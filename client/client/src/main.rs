use client::RpcClient;
use anyhow::Result;
use std::time::Duration;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    // info!("Connecting to server...");
    let client = RpcClient::connect("127.0.0.1:8080", Some(Duration::from_secs(5))).await?;
    // info!("Connected!");

    // info!("Test 1: Computing hash (sync)...");
    let h = client.hash_compute(b"abc", false).await?;
    println!("hash = {}", h);

    // info!("Test 2: Sorting array (async)...");
    let sorted = client.sort_array(vec![3, 1, -5, 7, 1], true).await?;
    println!("sorted = {:?}", sorted);

    // info!("Test 3: Matrix multiply (async)...");
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = client.matrix_multiply(2, a, b, true).await?;
    println!("matmul result = {:?}", c);

    // info!("Test 4: Compressing data (async)...");
    let comp = client.compress_data("zlib", b"hello hello hello", true).await?;
    println!("compressed len = {}", comp.len());

    info!("All tests completed!");
    Ok(())
}