use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use bytes::{Buf, BufMut, BytesMut};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::Duration};
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf},
    net::TcpStream,
    sync::{mpsc, RwLock, Mutex},
    time,
};
use tracing::{warn, info};

type RespTx = mpsc::UnboundedSender<RpcResponse>;
type RespRx = mpsc::UnboundedReceiver<RpcResponse>;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("connection closed")]
    Closed,
    #[error("server error: {0}")]
    Server(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("timeout waiting for response")]
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RpcRequest {
    request_id: String,
    func: String,
    #[serde(default)]
    r#async: bool,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ResponseStatus {
    Accepted,
    Completed,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RpcResponse {
    request_id: String,
    status: ResponseStatus,
    ok: bool,
    #[serde(default)]
    result: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<String>,
}

pub struct RpcClient {
    writer: Arc<Mutex<WriteHalf<TcpStream>>>,
    demux: Arc<RwLock<HashMap<String, RespTx>>>,
    timeout: Option<Duration>,
}

impl RpcClient {
    pub async fn connect(addr: &str, timeout: Option<Duration>) -> Result<Self> {
        let socket_addr: SocketAddr = addr.parse()?;
        let stream = TcpStream::connect(socket_addr).await?;
        stream.set_nodelay(true)?;
        let (reader, writer) = tokio::io::split(stream);
        let writer = Arc::new(Mutex::new(writer));
        let demux = Arc::new(RwLock::new(HashMap::new()));

        tokio::spawn(reader_loop(reader, demux.clone()));

        Ok(Self { writer, demux, timeout })
    }

    async fn call(
        &self,
        func: &str,
        params: serde_json::Value,
        async_mode: bool,
    ) -> Result<CallHandle> {
        let request_id = uuid::Uuid::new_v4().to_string();
        // info!(%request_id, %func, %async_mode, "creating call");
        
        let (tx, rx) = mpsc::unbounded_channel();
        self.demux.write().await.insert(request_id.clone(), tx);

        let req = RpcRequest {
            request_id: request_id.clone(),
            func: func.to_string(),
            r#async: async_mode,
            params,
        };

        let payload = serde_json::to_vec(&req)?;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.put_u32(payload.len() as u32);
        frame.extend_from_slice(&payload);

        // info!(%request_id, frame_len=%frame.len(), "sending frame");
        {
            let mut guard = self.writer.lock().await;
            // info!(%request_id, "acquired writer lock");
            guard.write_all(&frame).await?;
            guard.flush().await?;
            // info!(%request_id, "flushed");
        }
        // info!(%request_id, "frame sent");

        Ok(CallHandle {
            request_id,
            rx: Some(rx),
            timeout: self.timeout,
        })
    }

    pub async fn hash_compute(&self, data: &[u8], async_mode: bool) -> Result<String> {
        let params = serde_json::json!({ "data_base64": STANDARD.encode(data) });
        let mut h = self.call("hash_compute", params, async_mode).await?;
        let resp = h.wait_completed().await?;
        Ok(resp.result.unwrap()["hex"].as_str().unwrap().to_string())
    }

    pub async fn sort_array(&self, values: Vec<i32>, async_mode: bool) -> Result<Vec<i32>> {
        let params = serde_json::json!({ "values": values });
        let mut h = self.call("sort_array", params, async_mode).await?;
        let resp = h.wait_completed().await?;
        Ok(serde_json::from_value(resp.result.unwrap()["values"].clone())?)
    }

    pub async fn matrix_multiply(
        &self,
        n: usize,
        a: Vec<f64>,
        b: Vec<f64>,
        async_mode: bool,
    ) -> Result<Vec<f64>> {
        let params = serde_json::json!({ "n": n, "a": a, "b": b });
        let mut h = self.call("matrix_multiply", params, async_mode).await?;
        let resp = h.wait_completed().await?;
        Ok(serde_json::from_value(resp.result.unwrap()["c"].clone())?)
    }

    pub async fn compress_data(&self, algo: &str, data: &[u8], async_mode: bool) -> Result<Vec<u8>> {
        let params = serde_json::json!({
            "algo": algo,
            "data_base64": STANDARD.encode(data)
        });
        let mut h = self.call("compress_data", params, async_mode).await?;
        let resp = h.wait_completed().await?;
        Ok(STANDARD.decode(
            resp.result.unwrap()["compressed_base64"].as_str().unwrap(),
        )?)
    }
}

struct CallHandle {
    request_id: String,
    rx: Option<RespRx>,
    timeout: Option<Duration>,
}

impl CallHandle {
    async fn wait_completed(&mut self) -> Result<RpcResponse> {
        let rx = self.rx.as_mut().unwrap();
        // info!(request_id=%self.request_id, "waiting for response");

        loop {
            let fut = rx.recv();
            let msg = if let Some(to) = self.timeout {
                match time::timeout(to, fut).await {
                    Ok(Some(msg)) => {
                        // info!(request_id=%self.request_id, status=?msg.status, "received message");
                        msg
                    }
                    Ok(None) => {
                        warn!(request_id=%self.request_id, "channel closed");
                        return Err(anyhow!(ClientError::Closed));
                    }
                    Err(_) => {
                        warn!(request_id=%self.request_id, "timeout");
                        return Err(anyhow!(ClientError::Timeout));
                    }
                }
            } else {
                let msg = fut.await.ok_or_else(|| anyhow!(ClientError::Closed))?;
                // info!(request_id=%self.request_id, status=?msg.status, "received message");
                msg
            };

            match msg.status {
                ResponseStatus::Accepted => {
                    // info!(request_id=%self.request_id, "got accepted, waiting for completed");
                    continue;
                }
                ResponseStatus::Completed if msg.ok => {
                    // info!(request_id=%self.request_id, "got completed");
                    return Ok(msg);
                }
                ResponseStatus::Error | ResponseStatus::Completed => {
                    return Err(anyhow!(ClientError::Server(
                        msg.error.unwrap_or_else(|| "unknown".into())
                    )))
                }
            }
        }
    }
}

async fn reader_loop(mut reader: ReadHalf<TcpStream>, demux: Arc<RwLock<HashMap<String, RespTx>>>) {
    let mut buf = BytesMut::with_capacity(8192);
    // info!("reader_loop started");

    loop {
        if let Err(e) = ensure_read_exact(&mut reader, &mut buf, 4).await {
            warn!("reader error: {}", e);
            break;
        }
        let mut len_bytes = &buf[..4];
        let frame_len = len_bytes.get_u32() as usize;
        buf.advance(4);
        // info!(frame_len, "read frame length");

        if let Err(e) = ensure_read_exact(&mut reader, &mut buf, frame_len).await {
            warn!("reader error: {}", e);
            break;
        }
        let payload = buf.split_to(frame_len).freeze();

        match serde_json::from_slice::<RpcResponse>(&payload) {
            Ok(resp) => {
                // info!(request_id=%resp.request_id, status=?resp.status, "parsed response");
                let key = resp.request_id.clone();
                let tx = demux.read().await.get(&key).cloned();
                
                if let Some(tx) = tx {
                    // info!(request_id=%key, "sending to demux channel");
                    let _ = tx.send(resp.clone());
                    if matches!(resp.status, ResponseStatus::Completed | ResponseStatus::Error) {
                        demux.write().await.remove(&key);
                    }
                } else {
                    warn!(request_id=%key, "no channel found for request_id");
                }
            }
            Err(e) => warn!("bad response: {}", e),
        }
    }
}

async fn ensure_read_exact(reader: &mut ReadHalf<TcpStream>, buf: &mut BytesMut, needed: usize) -> Result<()> {
    buf.reserve(needed);
    while buf.len() < needed {
        let mut tmp = vec![0u8; needed - buf.len()];
        let n = reader.read(&mut tmp).await?;
        if n == 0 {
            return Err(anyhow!("eof"));
        }
        buf.put_slice(&tmp[..n]);
    }
    Ok(())
}