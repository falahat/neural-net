//! Benchmark the GPU backend of our `nn` library (cubecl + wgpu).
//!
//! Two passes per shape:
//!   - **one-shot** — the slice-based `backend::gpu::matmul(a, b)`
//!     API. Each call uploads inputs + downloads output, so the
//!     timing includes the full host↔device round trip. This is
//!     what a user calling the convenience API actually pays.
//!   - **pooled** — `GpuContext` with persistent `GpuTensor`s.
//!     Uploads happen ONCE per shape (outside the timing loop);
//!     output buffers come from the per-size pool, so steady-state
//!     iterations do zero GPU allocation.
//!
//! Each pass writes its own JSON file. The dashboard plots them
//! side-by-side so the buffer-pool win is visible at every shape.
//!
//! Build:
//! ```bash
//! cargo run --profile release-bench -p nn-bench --example nn_gpu --features gpu
//! ```

use std::sync::Arc;

use nn_bench::*;

#[cfg(feature = "gpu")]
use nn::backend::gpu::GpuContext;

fn bench_oneshot(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    for &(label, m, k, n, _max_iter, max_ms) in MATMUL_SHAPES {
        // Cap iterations: v1 has no recycling on the one-shot path,
        // so each iter does a fresh GPU alloc that wgpu doesn't
        // immediately reclaim. Capping keeps VRAM bounded.
        let max_iter = 100usize.min(if m * n > 256 * 256 { 30 } else { 100 });
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.231).cos()).collect();
        let (total, samples, resources) = time_op(
            || {
                let _ = nn::backend::gpu::matmul(&a, &[m, k], &b, &[k, n]);
            },
            max_iter,
            max_ms,
            gpu.clone(),
        );
        let mut run = Run::new("op", "matmul", format!("{}: {}x{}x{}", label, m, k, n));
        run.m = m;
        run.k = k;
        run.n = n;
        run.iters = samples.len();
        run.total_ns = total;
        run.samples = samples;
        run.resources = resources;
        let flops = 2.0 * (m as u64 * k as u64 * n as u64) as f64 * run.iters as f64;
        run.gflops = Some(flops / (total as f64 / 1e9) / 1e9);
        println!("[oneshot] matmul {:<14} {:>4}x{:>4}x{:>4}  iters={:>6}  mean={:>9.2}μs  GFLOP/s={:>6.2}",
                 label, m, k, n, run.iters, run.mean_ns() / 1000.0, run.gflops.unwrap());
        runs.push(run);
    }
}

#[cfg(feature = "gpu")]
fn bench_pooled(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    // ONE GpuContext for the entire bench. Persists upload buffers,
    // pools output buffers. Inputs go up to the device ONCE per
    // shape; only the kernel + checkout/release happens inside the
    // timing loop.
    let ctx = GpuContext::new();

    for &(label, m, k, n, max_iter, max_ms) in MATMUL_SHAPES {
        // Persistent residency means VRAM use stays bounded across
        // iterations (pool reuses output buffers), so we can run as
        // many iterations as time allows.
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.231).cos()).collect();
        // Uploads happen here, OUTSIDE the timing loop — they
        // shouldn't be counted as per-call cost.
        let a_gpu = ctx.upload(&a_data, vec![m, k]);
        let b_gpu = ctx.upload(&b_data, vec![k, n]);

        let (total, samples, resources) = time_op(
            || {
                // Kernel dispatch + sync. Without the sync, cubecl/wgpu
                // just queues the kernel and returns; we'd be measuring
                // queue-insert latency (microseconds) instead of actual
                // compute (which is the whole point of the bench).
                let c = ctx.matmul(&a_gpu, &b_gpu);
                ctx.sync();
                ctx.release(c);
            },
            max_iter,
            max_ms,
            gpu.clone(),
        );

        let mut run = Run::new("op", "matmul", format!("{}: {}x{}x{}", label, m, k, n));
        run.m = m;
        run.k = k;
        run.n = n;
        run.iters = samples.len();
        run.total_ns = total;
        run.samples = samples;
        run.resources = resources;
        let flops = 2.0 * (m as u64 * k as u64 * n as u64) as f64 * run.iters as f64;
        run.gflops = Some(flops / (total as f64 / 1e9) / 1e9);
        println!("[pooled]  matmul {:<14} {:>4}x{:>4}x{:>4}  iters={:>6}  mean={:>9.2}μs  GFLOP/s={:>6.2}",
                 label, m, k, n, run.iters, run.mean_ns() / 1000.0, run.gflops.unwrap());
        runs.push(run);
    }

    let (hits, misses) = ctx.pool_stats();
    println!(
        "\n[pooled]  pool stats: {} hits, {} misses ({}% reuse)",
        hits,
        misses,
        if hits + misses > 0 {
            hits * 100 / (hits + misses)
        } else {
            0
        }
    );
}

#[cfg(not(feature = "gpu"))]
fn bench_pooled(_runs: &mut Vec<Run>, _gpu: Option<Arc<GpuHandle>>) {
    eprintln!("pooled bench needs --features gpu");
}

fn main() -> std::io::Result<()> {
    let (label_oneshot, out_oneshot) = (
        "nn-gpu-oneshot".to_string(),
        "crates/ml/nn/bench_results/nn_gpu_oneshot.json".to_string(),
    );
    let (label_pooled, out_pooled) = (
        "nn-gpu-pooled".to_string(),
        "crates/ml/nn/bench_results/nn_gpu_pooled.json".to_string(),
    );
    let gpu_handle = GpuHandle::try_init();

    println!("nn GPU bench (cubecl/wgpu)");
    if let Some(h) = gpu_handle.as_ref() {
        println!("monitoring: {} ({} MB VRAM)", h.name, h.vram_total_mb);
    }
    println!();

    let machine_json = capture_machine_info(gpu_handle.as_ref());

    // Pass 1 — one-shot API (each call uploads + downloads).
    println!("── one-shot API: matmul(&[f32], …) — upload + dispatch + download per call ──");
    let mut runs_oneshot: Vec<Run> = Vec::new();
    bench_oneshot(&mut runs_oneshot, gpu_handle.clone());
    ensure_parent_dir(&out_oneshot)?;
    write_results(
        &label_oneshot,
        &machine_json,
        "cubecl-wgpu-oneshot",
        &runs_oneshot,
        &out_oneshot,
    )?;
    println!("wrote {} runs to {}", runs_oneshot.len(), out_oneshot);

    // Pass 2 — pooled API (uploads outside loop; output buffers pooled).
    println!("\n── pooled API: GpuContext, uploads outside loop, pooled output buffers ──");
    let mut runs_pooled: Vec<Run> = Vec::new();
    bench_pooled(&mut runs_pooled, gpu_handle.clone());
    ensure_parent_dir(&out_pooled)?;
    write_results(
        &label_pooled,
        &machine_json,
        "cubecl-wgpu-pooled",
        &runs_pooled,
        &out_pooled,
    )?;
    println!("wrote {} runs to {}", runs_pooled.len(), out_pooled);

    Ok(())
}
