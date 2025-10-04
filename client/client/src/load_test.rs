use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;

use client::{RpcClient, ClientError};

struct LoadTestConfig {
    target_rps: usize,           // requests per second
    duration_secs: u64,          // test duration
    num_connections: usize,      // concurrent connections
    operation: TestOperation,
}

#[derive(Clone, Debug)]
enum TestOperation {
    Hash,
    Sort,
    MatrixMul,
    Compress,
    Mixed, // Mix of all operations
}

#[derive(Debug)]
struct LatencyStats {
    measurements: Vec<u128>, // latencies in microseconds
    errors: usize,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            measurements: Vec::new(),
            errors: 0,
        }
    }

    fn record_success(&mut self, latency_us: u128) {
        self.measurements.push(latency_us);
    }

    fn record_error(&mut self) {
        self.errors += 1;
    }

    fn percentile(&mut self, p: f64) -> u128 {
        if self.measurements.is_empty() {
            return 0;
        }
        self.measurements.sort_unstable();
        let idx = ((self.measurements.len() as f64) * p) as usize;
        self.measurements[idx.min(self.measurements.len() - 1)]
    }

    fn avg(&self) -> f64 {
        if self.measurements.is_empty() {
            return 0.0;
        }
        self.measurements.iter().sum::<u128>() as f64 / self.measurements.len() as f64
    }

    fn total_requests(&self) -> usize {
        self.measurements.len() + self.errors
    }

    fn success_rate(&self) -> f64 {
        let total = self.total_requests();
        if total == 0 {
            return 0.0;
        }
        100.0 * self.measurements.len() as f64 / total as f64
    }
}

async fn run_load_test(config: LoadTestConfig) -> Result<LatencyStats> {
    println!("Starting load test:");
    println!("  Target: {} req/s", config.target_rps);
    println!("  Duration: {}s", config.duration_secs);
    println!("  Connections: {}", config.num_connections);
    println!("  Operation: {:?}", config.operation);
    println!();

    let stats = Arc::new(tokio::sync::Mutex::new(LatencyStats::new()));
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(config.duration_secs);
    
    let mut tasks = vec![];
    
    // Calculate requests per worker
    let rps_per_worker = config.target_rps / config.num_connections;
    let interval_us = 1_000_000 / rps_per_worker.max(1);
    
    for worker_id in 0..config.num_connections {
        let stats_clone = stats.clone();
        let op = config.operation.clone();
        
        let task = tokio::spawn(async move {
            // Each worker creates its own connection
            let client = match RpcClient::connect("127.0.0.1:8080", Some(Duration::from_secs(10))).await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Worker {} failed to connect: {}", worker_id, e);
                    return;
                }
            };
            
            let mut next_request = Instant::now();
            let interval = Duration::from_micros(interval_us as u64);
            
            while next_request < end_time {
                let req_start = Instant::now();
                
                // Execute request based on operation type
                let result = match &op {
                    TestOperation::Hash => {
                        client.hash_compute(b"benchmark data", true).await.map(|_| ())
                    }
                    TestOperation::Sort => {
                        client.sort_array(vec![9, 7, 5, 3, 1, 2, 4, 6, 8, 0], true).await.map(|_| ())
                    }
                    TestOperation::MatrixMul => {
                        let a = vec![1.0, 2.0, 3.0, 4.0];
                        let b = vec![5.0, 6.0, 7.0, 8.0];
                        client.matrix_multiply(2, a, b, true).await.map(|_| ())
                    }
                    TestOperation::Compress => {
                        client.compress_data("lz4", b"benchmark data for compression", true)
                            .await.map(|_| ())
                    }
                    TestOperation::Mixed => {
                        // Randomly pick operation
                        match worker_id % 4 {
                            0 => client.hash_compute(b"data", true).await.map(|_| ()),
                            1 => client.sort_array(vec![5, 2, 8], true).await.map(|_| ()),
                            2 => {
                                let a = vec![1.0, 2.0, 3.0, 4.0];
                                let b = vec![5.0, 6.0, 7.0, 8.0];
                                client.matrix_multiply(2, a, b, true).await.map(|_| ())
                            }
                            _ => client.compress_data("lz4", b"data", true).await.map(|_| ()),
                        }
                    }
                };
                
                let latency_us = req_start.elapsed().as_micros();
                
                // Record result
                let mut stats_guard = stats_clone.lock().await;
                match result {
                    Ok(_) => stats_guard.record_success(latency_us),
                    Err(_) => stats_guard.record_error(),
                }
                drop(stats_guard);
                
                // Open loop: schedule next request regardless of completion
                next_request += interval;
                let now = Instant::now();
                if next_request > now {
                    time::sleep(next_request - now).await;
                }
            }
        });
        
        tasks.push(task);
    }
    
    // Wait for all workers
    for task in tasks {
        let _ = task.await;
    }
    
    let final_stats = Arc::try_unwrap(stats).unwrap().into_inner();
    Ok(final_stats)
}

async fn generate_load_latency_curve() -> Result<()> {
    println!("=== Generating Load-Latency Curve ===\n");
    
    // Test at different load levels
    let load_levels = vec![10, 25, 50, 100, 150, 200, 300, 400, 500];
    let mut results = HashMap::new();
    
    for &rps in &load_levels {
        println!("Testing at {} req/s...", rps);
        
        let config = LoadTestConfig {
            target_rps: rps,
            duration_secs: 10, // 10 seconds per test
            num_connections: 10,
            operation: TestOperation::Mixed,
        };
        
        let mut stats = run_load_test(config).await?;
        
        results.insert(rps, (
            stats.avg(),
            stats.percentile(0.50),
            stats.percentile(0.95),
            stats.percentile(0.99),
            stats.success_rate(),
        ));
        
        // Brief pause between tests
        time::sleep(Duration::from_secs(2)).await;
    }
    
    // Print results table
    println!("\n=== Load-Latency Analysis ===\n");
    println!("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}", 
             "Load (rps)", "Avg (ms)", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Success %");
    println!("{}", "-".repeat(80));
    
    for &rps in &load_levels {
        if let Some(&(avg, p50, p95, p99, success)) = results.get(&rps) {
            println!("{:<12} {:<12.2} {:<12.2} {:<12.2} {:<12.2} {:<12.1}", 
                     rps,
                     avg / 1000.0,
                     p50 as f64 / 1000.0,
                     p95 as f64 / 1000.0,
                     p99 as f64 / 1000.0,
                     success);
        }
    }
    
    // Find saturation point (where p99 > 100ms or success < 99%)
    println!("\n=== Analysis ===");
    for &rps in &load_levels {
        if let Some(&(_, _, _, p99, success)) = results.get(&rps) {
            if p99 > 100_000 || success < 99.0 {
                println!("System saturation detected at ~{} req/s", rps);
                println!("  p99 latency: {:.2}ms", p99 as f64 / 1000.0);
                println!("  success rate: {:.1}%", success);
                break;
            }
        }
    }
    
    Ok(())
}

async fn run_sustained_load_test() -> Result<()> {
    println!("=== Sustained Load Test ===\n");
    
    let config = LoadTestConfig {
        target_rps: 200,
        duration_secs: 30,
        num_connections: 10,
        operation: TestOperation::Mixed,
    };
    
    let mut stats = run_load_test(config).await?;
    
    println!("\n=== Results ===");
    println!("Total requests: {}", stats.total_requests());
    println!("Successful: {}", stats.measurements.len());
    println!("Failed: {}", stats.errors);
    println!("Success rate: {:.2}%", stats.success_rate());
    println!("\nLatency Statistics:");
    println!("  Average: {:.2}ms", stats.avg() / 1000.0);
    println!("  p50: {:.2}ms", stats.percentile(0.50) as f64 / 1000.0);
    println!("  p95: {:.2}ms", stats.percentile(0.95) as f64 / 1000.0);
    println!("  p99: {:.2}ms", stats.percentile(0.99) as f64 / 1000.0);
    println!("  p99.9: {:.2}ms", stats.percentile(0.999) as f64 / 1000.0);
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("RPC Load Testing Tool\n");
    println!("1. Running sustained load test (200 req/s for 30s)...\n");
    run_sustained_load_test().await?;
    
    println!("\n\n2. Generating load-latency curve...\n");
    time::sleep(Duration::from_secs(3)).await;
    generate_load_latency_curve().await?;
    
    Ok(())
}