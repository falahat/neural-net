//! Peer benchmark: `dfdx` (coreylowman).
//!
//! Why this peer: very different design (compile-time shape
//! checking via const generics; no tape). Useful to see whether
//! our tape overhead is real at the small-shape end of the ladder.
//!
//! Because dfdx's shapes are TYPE PARAMETERS, each (m, k, n) tuple
//! needs its own `Rank2<M, K>`/`Rank2<K, N>` annotation. A small
//! macro keeps the repetition tolerable.
//!
//! Build:
//! ```bash
//! cargo run --profile release-bench -p nn-bench --example dfdx --features dfdx -- \
//!     --label dfdx --out crates/ml/nn/bench_results/dfdx.json
//! ```

use std::sync::Arc;

use dfdx::prelude::*;
use nn_bench::*;

macro_rules! bench_matmul_shape {
    ($runs:expr, $gpu:expr, $label:expr, $M:literal, $K:literal, $N:literal, $max_iter:expr, $max_ms:expr) => {{
        let dev: Cpu = Default::default();
        let a: Tensor<Rank2<$M, $K>, f32, _> = dev.sample_normal();
        let b: Tensor<Rank2<$K, $N>, f32, _> = dev.sample_normal();
        let (total, samples, resources) = time_op(
            || {
                // `matmul` consumes; clone the inputs each iter so the
                // measurement reflects what a real forward pass costs.
                let _ = a.clone().matmul(b.clone());
            },
            $max_iter,
            $max_ms,
            $gpu.clone(),
        );
        let mut run = Run::new("op", "matmul", format!("{}: {}x{}x{}", $label, $M, $K, $N));
        run.m = $M;
        run.k = $K;
        run.n = $N;
        run.iters = samples.len();
        run.total_ns = total;
        run.samples = samples;
        run.resources = resources;
        let flops = 2.0 * ($M as u64 * $K as u64 * $N as u64) as f64 * run.iters as f64;
        run.gflops = Some(flops / (total as f64 / 1e9) / 1e9);
        println!(
            "matmul {:<14} {:>4}x{:>4}x{:>4}  iters={:>6}  mean={:>9.2}μs  GFLOP/s={:>6.2}",
            $label,
            $M,
            $K,
            $N,
            run.iters,
            run.mean_ns() / 1000.0,
            run.gflops.unwrap()
        );
        $runs.push(run);
    }};
}

fn bench_matmul(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    // Mirror MATMUL_SHAPES. dfdx needs literal dimensions so we can't
    // iterate over the slice — these are hand-aligned with the shared
    // ladder in nn_bench::MATMUL_SHAPES.
    bench_matmul_shape!(runs, gpu, "widget-tiny", 4, 2, 8, 50_000, 600);
    bench_matmul_shape!(runs, gpu, "widget-small", 32, 16, 16, 50_000, 600);
    bench_matmul_shape!(runs, gpu, "widget-med", 64, 64, 64, 20_000, 1000);
    bench_matmul_shape!(runs, gpu, "RL-small", 128, 32, 32, 20_000, 1000);
    bench_matmul_shape!(runs, gpu, "RL-mid", 256, 128, 128, 5_000, 2000);
    bench_matmul_shape!(runs, gpu, "RL-large", 512, 256, 256, 1_000, 3000);
    bench_matmul_shape!(runs, gpu, "GPU-warm", 1024, 512, 512, 100, 5000);
    bench_matmul_shape!(runs, gpu, "GPU-hot", 2048, 1024, 1024, 20, 8000);
}

macro_rules! bench_elemwise_shape {
    ($runs:expr, $gpu:expr, $label:expr, $N:literal, $max_iter:expr, $max_ms:expr) => {{
        let dev: Cpu = Default::default();
        let a: Tensor<Rank1<$N>, f32, _> = dev.sample_normal();
        let b: Tensor<Rank1<$N>, f32, _> = dev.sample_normal();

        let (t, sm, res) = time_op(
            || {
                let _ = a.clone() + b.clone();
            },
            $max_iter,
            $max_ms,
            $gpu.clone(),
        );
        let mut r = Run::new("op", "add", format!("{}: n={}", $label, $N));
        r.n = $N;
        r.iters = sm.len();
        r.total_ns = t;
        r.samples = sm;
        r.resources = res;
        println!(
            "add    {:<14} n={:>7}  iters={:>6}  mean={:>9.2}μs",
            $label,
            $N,
            r.iters,
            r.mean_ns() / 1000.0
        );
        $runs.push(r);

        let (t, sm, res) = time_op(
            || {
                let _ = a.clone() * b.clone();
            },
            $max_iter,
            $max_ms,
            $gpu.clone(),
        );
        let mut r = Run::new("op", "mul", format!("{}: n={}", $label, $N));
        r.n = $N;
        r.iters = sm.len();
        r.total_ns = t;
        r.samples = sm;
        r.resources = res;
        $runs.push(r);

        let (t, sm, res) = time_op(
            || {
                let _ = a.clone().relu();
            },
            $max_iter,
            $max_ms,
            $gpu.clone(),
        );
        let mut r = Run::new("op", "relu", format!("{}: n={}", $label, $N));
        r.n = $N;
        r.iters = sm.len();
        r.total_ns = t;
        r.samples = sm;
        r.resources = res;
        $runs.push(r);

        let (t, sm, res) = time_op(
            || {
                let _ = a.clone().sigmoid();
            },
            $max_iter,
            $max_ms,
            $gpu.clone(),
        );
        let mut r = Run::new("op", "sigmoid", format!("{}: n={}", $label, $N));
        r.n = $N;
        r.iters = sm.len();
        r.total_ns = t;
        r.samples = sm;
        r.resources = res;
        $runs.push(r);
    }};
}

fn bench_elemwise(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    bench_elemwise_shape!(runs, gpu, "widget-tiny", 64, 100_000, 600);
    bench_elemwise_shape!(runs, gpu, "widget-small", 256, 100_000, 600);
    bench_elemwise_shape!(runs, gpu, "widget-med", 4_096, 20_000, 1000);
    bench_elemwise_shape!(runs, gpu, "RL-mid", 16_384, 10_000, 1000);
    bench_elemwise_shape!(runs, gpu, "RL-large", 262_144, 1_000, 2000);
    bench_elemwise_shape!(runs, gpu, "GPU-warm", 1_048_576, 200, 3000);
}

fn main() -> std::io::Result<()> {
    let (label, out) = parse_args("dfdx", "crates/ml/nn/bench_results/dfdx.json");
    let gpu = GpuHandle::try_init();

    println!("dfdx peer benchmark: label={label}, out={out}");
    println!();

    let machine_json = capture_machine_info(gpu.as_ref());
    let mut runs: Vec<Run> = Vec::new();

    println!("── matmul ──");
    bench_matmul(&mut runs, gpu.clone());
    println!("\n── elementwise ──");
    bench_elemwise(&mut runs, gpu.clone());

    ensure_parent_dir(&out)?;
    write_results(&label, &machine_json, "dfdx-cpu", &runs, &out)?;
    println!("\nwrote {} runs to {}", runs.len(), out);
    Ok(())
}
