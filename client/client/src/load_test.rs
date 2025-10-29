use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::{interval, MissedTickBehavior};
use tokio::sync::mpsc;
use std::fs::File;
use std::io::Write as IoWrite;
use tracing::{info, warn};

use client::RpcClient;

/// Run a single load test at specified RPS
async fn run_single_load_test(
    addr: &str,
    rps: usize,
    duration_secs: u64,
    async_mode: bool,
) -> Result<(Vec<f64>, usize)> {
    // Connection pool size based on sqrt(RPS)
    let pool_size = ((rps as f64).sqrt().ceil() as usize).max(4).min(64);
    let mut pool = Vec::with_capacity(pool_size);
    
    for _ in 0..pool_size {
        let client = RpcClient::connect(addr, Some(Duration::from_secs(10))).await?;
        pool.push(Arc::new(Mutex::new(client)));
    }
    
    // Channel to collect latencies (ms)
    let (tx, mut rx) = mpsc::unbounded_channel::<f64>();
    
    // Open-loop ticker
    let mut tick = interval(Duration::from_nanos(1_000_000_000 / rps.max(1) as u64));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    
    let end_time = Instant::now() + Duration::from_secs(duration_secs);
    let mut i = 0usize;
    let mut requests_sent = 0usize;
    
    // Deterministic RNG for the op mix
    let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(0xC0FFEE)));
    
    while Instant::now() < end_time {
        tick.tick().await;
        
        let cli = pool[i % pool_size].clone();
        i += 1;
        requests_sent += 1;
        let txc = tx.clone();
        let rngc = rng.clone();
        
        tokio::spawn(async move {
            // Choose operation (50% hash, 20% sort, 10% matmul, 20% compress)
            let mut rng = rngc.lock().await;
            let p: f64 = rng.gen();
            drop(rng);
            
            let which = if p < 0.5 { "hash" }
            else if p < 0.7 { "sort" }
            else if p < 0.8 { "matmul" }
            else { "compress" };
            
            let start = Instant::now();
            let res: Result<()> = async {
                let mut c = cli.lock().await;
                match which {
                    "hash" => {
                        let mut data = vec![0u8; 256];
                        for (i, b) in data.iter_mut().enumerate() {
                            *b = (i as u8).wrapping_mul(31).wrapping_add(7);
                        }
                        c.hash_compute(&data, async_mode).await?;
                    }
                    "sort" => {
                        let mut vals = vec![0i32; 1000];
                        for (i, v) in vals.iter_mut().enumerate() {
                            let x = ((i as u64 * 1_103_515_245u64 + 12_345u64) >> 8) as u32;
                            *v = (x as i32) ^ 0x5a5a5a5a;
                        }
                        c.sort_array(vals, async_mode).await?;
                    }
                    "matmul" => {
                        let n = 16usize;
                        let mut a = vec![0.0f64; n * n];
                        let mut b = vec![0.0f64; n * n];
                        for i in 0..n * n {
                            a[i] = (i as f64).sin();
                            b[i] = (i as f64).cos();
                        }
                        c.matrix_multiply(n, a, b, async_mode).await?;
                    }
                    "compress" => {
                        let mut data = vec![0u8; 512];
                        for (i, b) in data.iter_mut().enumerate() {
                            *b = (i as u8).wrapping_mul(17).wrapping_add(3);
                        }
                        c.compress_data("zlib", &data, async_mode).await?;
                    }
                    _ => unreachable!(),
                }
                Ok(())
            }.await;
            
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if res.is_ok() {
                let _ = txc.send(elapsed);
            } else if let Err(e) = res {
                warn!("request error: {e}");
            }
        });
    }
    
    drop(tx);
    let mut lats = Vec::<f64>::new();
    while let Some(ms) = rx.recv().await {
        lats.push(ms);
    }
    
    Ok((lats, requests_sent))
}

fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len();
    let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
    v[idx]
}

// Test 1: Load-Latency Curve
async fn test_load_latency_curve() -> Result<()> {
    println!("\n=== TEST 1: Load-Latency Curve ===");
    println!("Testing with async mode\n");
    
    let load_levels = vec![10, 50, 100, 500, 1000, 2000, 5000, 10000, 11000, 12000, 13000, 15000];
    let mut results = Vec::new();
    
    for &rps in &load_levels {
        print!("Testing {} rps... ", rps);
        
        let (mut lats, sent) = run_single_load_test("127.0.0.1:8080", rps, 30, true).await?;
        
        if lats.is_empty() {
            println!("No samples collected.");
            continue;
        }
        
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let sum: f64 = lats.iter().sum();
        let avg = sum / lats.len() as f64;
        let p50 = percentile(&lats, 50.0);
        let p95 = percentile(&lats, 95.0);
        let p99 = percentile(&lats, 99.0);
        let actual_rps = lats.len() as f64 / 30.0;
        let success_rate = (lats.len() as f64 / sent as f64) * 100.0;
        
        println!("Done (p95: {:.2}ms, actual: {:.0}/{} rps, {:.1}% success)", 
                 p95, actual_rps, rps, success_rate);
        
        results.push((rps, actual_rps, avg, p50, p95, p99, success_rate));
        
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    
    // Save results
    let mut file = File::create("load_latency_curve.csv")?;
    writeln!(file, "Target_RPS,Actual_RPS,Avg_ms,p50_ms,p95_ms,p99_ms,Success_Rate")?;
    for (target, actual, avg, p50, p95, p99, success) in &results {
        writeln!(file, "{},{:.0},{:.3},{:.3},{:.3},{:.3},{:.2}",
                 target, actual, avg, p50, p95, p99, success)?;
    }
    
    println!("\n✓ Saved: load_latency_curve.csv");
    Ok(())
}

// Test 2: Throughput Analysis
async fn test_throughput_analysis() -> Result<()> {
    println!("\n=== TEST 2: Throughput Analysis ===");
    println!("Finding maximum sustainable throughput (p99 < 15ms, success > 99%)\n");
    
    let mut current_rps = 100;
    let mut results = Vec::new();
    let target_p99_ms = 15.0;
    let mut max_sustainable_rps = 0;
    
    loop {
        print!("Testing {} rps... ", current_rps);
        
        let (mut lats, sent) = run_single_load_test("127.0.0.1:8080", current_rps, 30, true).await?;
        
        if lats.is_empty() {
            println!("No samples collected.");
            break;
        }
        
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let sum: f64 = lats.iter().sum();
        let avg = sum / lats.len() as f64;
        let p99 = percentile(&lats, 99.0);
        let actual_rps = lats.len() as f64 / 30.0;
        let success_rate = (lats.len() as f64 / sent as f64) * 100.0;
        let achievement_rate = (actual_rps / current_rps as f64) * 100.0;
        
        println!("Done (p99: {:.2}ms, {:.1}% success)", p99, success_rate);
        
        results.push((current_rps, actual_rps, avg, p99, success_rate, achievement_rate));
        
        if p99 < target_p99_ms && success_rate > 99.0 && achievement_rate > 90.0 {
            max_sustainable_rps = actual_rps as usize;
        }
        
        if p99 >= target_p99_ms || success_rate < 99.0 || achievement_rate < 90.0 {
            println!("\n✓ Found saturation point");
            break;
        }
        
        if current_rps > 20000 {
            println!("\n✓ Reached test limit");
            break;
        }
        
        current_rps = match current_rps {
            r if r < 1000 => r + 200,
            r if r < 5000 => r + 500,
            r if r < 20000 => r + 1000,
            _ => current_rps + 5000,
        };
        
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    
    // Save results
    let mut file = File::create("throughput_analysis.csv")?;
    writeln!(file, "Target_RPS,Actual_RPS,Avg_ms,p99_ms,Success_Rate,Achievement_Rate")?;
    for (target, actual, avg, p99, success, achievement) in &results {
        writeln!(file, "{},{:.0},{:.3},{:.3},{:.2},{:.2}",
                 target, actual, avg, p99, success, achievement)?;
    }
    
    println!("\n✓ Saved: throughput_analysis.csv");
    println!("  Max sustainable: {} rps", max_sustainable_rps);
    
    Ok(())
}

// Test 3: Operation Comparison
async fn test_operation_comparison() -> Result<()> {
    println!("\n=== TEST 3: Operation Comparison ===\n");
    
    let operations = vec![
        ("hash", "hash"),
        ("sort", "sort"),
        ("matmul", "matmul"),
        ("compress", "compress"),
    ];
    
    let test_rps_levels = vec![100, 500, 1000, 2000, 5000, 10000, 12000, 13000, 15000];
    let mut all_results = Vec::new();
    
    for (op_name, _op_code) in operations {
        println!("Testing {} operation:", op_name);
        
        for &rps in &test_rps_levels {
            print!("  {} rps... ", rps);
            
            let (mut lats, sent) = run_single_load_test("127.0.0.1:8080", rps, 30, true).await?;
            
            if lats.is_empty() {
                println!("No samples collected.");
                continue;
            }
            
            lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let sum: f64 = lats.iter().sum();
            let avg = sum / lats.len() as f64;
            let p50 = percentile(&lats, 50.0);
            let p95 = percentile(&lats, 95.0);
            let p99 = percentile(&lats, 99.0);
            let actual_rps = lats.len() as f64 / 30.0;
            let success_rate = (lats.len() as f64 / sent as f64) * 100.0;
            
            println!("Done (avg: {:.2}ms, p95: {:.2}ms)", avg, p95);
            
            all_results.push((op_name, rps, actual_rps, avg, p50, p95, p99, success_rate));
            
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        println!();
    }
    
    // Save results
    let mut file = File::create("operation_comparison.csv")?;
    writeln!(file, "Operation,Target_RPS,Actual_RPS,Avg_ms,p50_ms,p95_ms,p99_ms,Success_Rate")?;
    for (op, target, actual, avg, p50, p95, p99, success) in &all_results {
        writeln!(file, "{},{},{:.0},{:.3},{:.3},{:.3},{:.3},{:.2}",
                 op, target, actual, avg, p50, p95, p99, success)?;
    }
    
    println!("✓ Saved: operation_comparison.csv");
    
    Ok(())
}

// Test 4: Sync vs Async Comparison
async fn test_sync_vs_async() -> Result<()> {
    println!("\n=== TEST 4: Sync vs Async Mode Comparison ===\n");
    
    let test_rps_levels = vec![100, 500, 1000, 2000, 5000, 10000, 12000, 13000, 15000];
    let mut results = Vec::new();
    
    for async_mode in [false, true] {
        let mode_name = if async_mode { "Async" } else { "Sync" };
        println!("Testing {} mode:", mode_name);
        
        for &rps in &test_rps_levels {
            print!("  {} rps... ", rps);
            
            let (mut lats, sent) = run_single_load_test("127.0.0.1:8080", rps, 30, async_mode).await?;
            
            if lats.is_empty() {
                println!("No samples collected.");
                continue;
            }
            
            lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let sum: f64 = lats.iter().sum();
            let avg = sum / lats.len() as f64;
            let p50 = percentile(&lats, 50.0);
            let p95 = percentile(&lats, 95.0);
            let p99 = percentile(&lats, 99.0);
            let actual_rps = lats.len() as f64 / 30.0;
            let success_rate = (lats.len() as f64 / sent as f64) * 100.0;
            
            println!("Done (avg: {:.2}ms, p95: {:.2}ms)", avg, p95);
            
            results.push((mode_name, rps, actual_rps, avg, p50, p95, p99, success_rate));
            
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        println!();
    }
    
    // Save results
    let mut file = File::create("sync_vs_async_comparison.csv")?;
    writeln!(file, "Mode,Target_RPS,Actual_RPS,Avg_ms,p50_ms,p95_ms,p99_ms,Success_Rate")?;
    for (mode, target, actual, avg, p50, p95, p99, success) in &results {
        writeln!(file, "{},{},{:.0},{:.3},{:.3},{:.3},{:.3},{:.2}",
                 mode, target, actual, avg, p50, p95, p99, success)?;
    }
    
    println!("✓ Saved: sync_vs_async_comparison.csv");
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    
    println!("RPC Load Testing Suite");
    println!("======================\n");
    
    test_load_latency_curve().await?;
    tokio::time::sleep(Duration::from_secs(3)).await;
    
    test_throughput_analysis().await?;
    tokio::time::sleep(Duration::from_secs(3)).await;
    
    test_operation_comparison().await?;
    tokio::time::sleep(Duration::from_secs(3)).await;
    
    test_sync_vs_async().await?;
    
    println!("All tests complete!");
    
    Ok(())
}