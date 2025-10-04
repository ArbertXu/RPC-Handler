use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::{Buf, BufMut, BytesMut};
use flate2::{write::ZlibEncoder, Compression};
use lz4_flex::block::compress_prepend_size as lz4_compress_prepend_size;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{io::Write, net::SocketAddr, sync::Arc};
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf},
    net::{TcpListener, TcpStream},
    sync::mpsc,
};
use tracing::{error, info, instrument, warn};
use tracing_subscriber::FmtSubscriber;

#[derive(Debug, Error)]
enum RpcError {
    #[error("unknown function '{0}'")]
    UnknownFunction(String),
    #[error("invalid params: {0}")]
    InvalidParams(String),
    #[error("internal error: {0}")]
    Internal(String),
}

#[derive(Debug, Deserialize)]
struct RpcRequest {
    request_id: String,
    func: String,
    #[serde(default)]
    r#async: bool,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Serialize, Clone)]
struct RpcResponse {
    request_id: String,
    status: ResponseStatus,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
enum ResponseStatus {
    Accepted,
    Completed,
    Error,
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder().with_max_level(tracing::Level::INFO).finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let bind_addr = std::env::var("RPC_BIND").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let listener = TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("failed to bind to {bind_addr}"))?;
    info!("RPC server listening on {}", bind_addr);

    loop {
        let (socket, peer) = listener.accept().await?;
        tokio::spawn(handle_connection(socket, peer));
    }
}

#[instrument(skip(socket))]
async fn handle_connection(socket: TcpStream, peer: SocketAddr) {
    info!(%peer, "client connected");
    let (mut reader, mut writer) = tokio::io::split(socket);
    let (tx, mut rx) = mpsc::unbounded_channel::<RpcResponse>();
    
    // Spawn dedicated writer task
    let writer_task = tokio::spawn(async move {
        while let Some(resp) = rx.recv().await {
            if let Err(e) = send_response_direct(&mut writer, &resp).await {
                error!("writer task error: {}", e);
                break;
            }
        }
    });
    
    let mut buf = BytesMut::with_capacity(8 * 1024);

    loop {
        if let Err(e) = ensure_read_exact(&mut reader, &mut buf, 4).await {
            info!(%peer, "client disconnected: {}", e);
            break;
        }

        let mut len_bytes = &buf[..4];
        let frame_len = len_bytes.get_u32() as usize;
        buf.advance(4);

        if let Err(e) = ensure_read_exact(&mut reader, &mut buf, frame_len).await {
            error!(%peer, "frame read error: {}", e);
            break;
        }
        let payload = buf.split_to(frame_len).freeze();

        let req: RpcRequest = match serde_json::from_slice(&payload) {
            Ok(r) => r,
            Err(e) => {
                error!(%peer, "invalid JSON: {}", e);
                break;
            }
        };

        let tx_clone = tx.clone();
        let request_id = req.request_id.clone();
        let func_name = req.func.clone();

        if req.r#async {
            info!(%peer, %request_id, "handling async request: {}", func_name);
            
            let accepted = RpcResponse {
                request_id: request_id.clone(),
                status: ResponseStatus::Accepted,
                ok: true,
                result: None,
                error: None,
            };
            if tx.send(accepted).is_err() {
                error!(%peer, "failed to send accepted");
                break;
            }
            
            info!(%peer, %request_id, "sent accepted response");

            tokio::spawn(async move {
                info!(%request_id, "starting async processing");
                let resp = process_request(req).await;
                info!(%request_id, ok=%resp.ok, "processing completed, sending response");
                
                if tx_clone.send(resp).is_err() {
                    error!(%request_id, "failed to send completed");
                }
            });
        } else {
            info!(%peer, %request_id, "handling sync request: {}", func_name);
            let resp = process_request(req).await;
            if tx.send(resp).is_err() {
                error!(%peer, "failed to send response");
                break;
            }
            info!(%peer, %request_id, "sent sync response");
        }
    }
    
    drop(tx);
    let _ = writer_task.await;
}

async fn process_request(req: RpcRequest) -> RpcResponse {
    let rid = req.request_id.clone();

    let result = match req.func.as_str() {
        "hash_compute" => op_hash_compute(&req.params).map_err(|e| e.to_string()),
        "sort_array" => op_sort_array(&req.params).map_err(|e| e.to_string()),
        "matrix_multiply" => {
            let params = req.params.clone();
            match tokio::task::spawn_blocking(move || op_matrix_multiply(&params)).await {
                Ok(res) => res.map_err(|e| e.to_string()),
                Err(e) => Err(format!("join error: {}", e)),
            }
        }
        "compress_data" => {
            let params = req.params.clone();
            match tokio::task::spawn_blocking(move || op_compress_data(&params)).await {
                Ok(res) => res.map_err(|e| e.to_string()),
                Err(e) => Err(format!("join error: {}", e)),
            }
        }
        other => Err(RpcError::UnknownFunction(other.to_string()).to_string()),
    };

    match result {
        Ok(payload) => RpcResponse {
            request_id: rid,
            status: ResponseStatus::Completed,
            ok: true,
            result: Some(payload),
            error: None,
        },
        Err(err) => RpcResponse {
            request_id: rid,
            status: ResponseStatus::Error,
            ok: false,
            result: None,
            error: Some(err),
        },
    }
}

/* -------------------- Operations -------------------- */

#[derive(Deserialize)]
struct HashParams {
    data_base64: String,
}

fn op_hash_compute(params: &serde_json::Value) -> Result<serde_json::Value, RpcError> {
    let p: HashParams = serde_json::from_value(params.clone())
        .map_err(|e| RpcError::InvalidParams(e.to_string()))?;
    let data = STANDARD
        .decode(p.data_base64)
        .map_err(|e| RpcError::InvalidParams(e.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let digest = hasher.finalize();
    Ok(serde_json::json!({ "hex": hex::encode(digest) }))
}

#[derive(Deserialize)]
struct SortParams {
    values: Vec<i32>,
}

fn op_sort_array(params: &serde_json::Value) -> Result<serde_json::Value, RpcError> {
    let mut p: SortParams = serde_json::from_value(params.clone())
        .map_err(|e| RpcError::InvalidParams(e.to_string()))?;
    p.values.sort_unstable();
    Ok(serde_json::json!({ "values": p.values }))
}

#[derive(Deserialize)]
struct MatMulParams {
    n: usize,
    a: Vec<f64>,
    b: Vec<f64>,
}

fn op_matrix_multiply(params: &serde_json::Value) -> Result<serde_json::Value, RpcError> {
    let p: MatMulParams = serde_json::from_value(params.clone())
        .map_err(|e| RpcError::InvalidParams(e.to_string()))?;
    let n = p.n;
    if p.a.len() != n * n || p.b.len() != n * n {
        return Err(RpcError::InvalidParams("matrix size mismatch".to_string()));
    }
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = p.a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * p.b[k * n + j];
            }
        }
    }
    Ok(serde_json::json!({ "c": c }))
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Algo {
    Zlib,
    Lz4,
}

#[derive(Deserialize)]
struct CompressParams {
    algo: Algo,
    data_base64: String,
}

fn op_compress_data(params: &serde_json::Value) -> Result<serde_json::Value, RpcError> {
    let p: CompressParams = serde_json::from_value(params.clone())
        .map_err(|e| RpcError::InvalidParams(e.to_string()))?;
    let data = STANDARD
        .decode(p.data_base64)
        .map_err(|e| RpcError::InvalidParams(e.to_string()))?;

    let compressed = match p.algo {
        Algo::Zlib => {
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
            enc.write_all(&data).map_err(|e| RpcError::Internal(e.to_string()))?;
            enc.finish().map_err(|e| RpcError::Internal(e.to_string()))?
        }
        Algo::Lz4 => lz4_compress_prepend_size(&data),
    };

    Ok(serde_json::json!({ "compressed_base64": STANDARD.encode(compressed) }))
}

/* -------------------- Framing -------------------- */

async fn ensure_read_exact(reader: &mut ReadHalf<TcpStream>, buf: &mut BytesMut, needed: usize) -> Result<()> {
    buf.reserve(needed);
    while buf.len() < needed {
        let mut tmp = vec![0u8; needed - buf.len()];
        let n = reader.read(&mut tmp).await?;
        if n == 0 { return Err(anyhow!("peer closed")); }
        buf.put_slice(&tmp[..n]);
    }
    Ok(())
}

async fn send_response_direct(writer: &mut WriteHalf<TcpStream>, resp: &RpcResponse) -> Result<()> {
    let payload = serde_json::to_vec(resp)?;
    info!(request_id=%resp.request_id, status=?resp.status, payload_len=%payload.len(), "sending response");
    
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.put_u32(payload.len() as u32);
    frame.extend_from_slice(&payload);
    
    writer.write_all(&frame).await?;
    writer.flush().await?;
    
    info!(request_id=%resp.request_id, "response sent and flushed");
    Ok(())
}