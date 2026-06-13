//! Tiny matmul benchmark — no criterion dep; just `Instant`.
//!
//! Run scalar then SIMD:
//! ```bash
//! cargo run --release --example matmul_bench -p nn
//! cargo run --release --example matmul_bench -p nn --features simd
//! ```
//!
//! Reads `NN_BENCH_ITERS` (default 200) to set the iteration count.

use std::time::Instant;

use nn::backend::cpu;

fn bench(label: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.231).cos()).collect();

    // Warmup.
    for _ in 0..10 {
        let _ = cpu::matmul(&a, &[m, k], &b, &[k, n]);
    }

    let t = Instant::now();
    for _ in 0..iters {
        let _ = cpu::matmul(&a, &[m, k], &b, &[k, n]);
    }
    let elapsed = t.elapsed();
    let flops = 2.0 * (m * k * n) as f64 * iters as f64;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;
    println!(
        "{label:<24} {m}×{k} @ {k}×{n}  | {iters} iters in {:>7.2} ms | {:.2} GFLOP/s",
        elapsed.as_secs_f64() * 1e3,
        gflops,
    );
}

fn main() {
    let iters: usize = std::env::var("NN_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let feature = if cfg!(feature = "simd") {
        "SIMD (f32x8)"
    } else {
        "scalar"
    };
    println!("nn matmul bench — backend: {feature}");
    println!();

    bench("tiny     (XOR-sized)", 4, 2, 8, iters * 100);
    bench("small    (hidden=64)", 32, 64, 64, iters * 10);
    bench("medium   (hidden=256)", 64, 256, 256, iters);
    bench("large    (hidden=512)", 128, 512, 512, iters / 4);
}
