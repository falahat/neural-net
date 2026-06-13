//! Peer benchmark: `burn` (tracel-ai) on the ndarray CPU backend.
//!
//! Why this peer: same team that ships cubecl (our chosen GPU JIT);
//! a like-for-like CPU comparison helps calibrate when the cubecl
//! path lands later.
//!
//! Build:
//! ```bash
//! cargo run --profile release-bench -p nn-bench --example burn --features burn -- \
//!     --label burn --out crates/ml/nn/bench_results/burn.json
//! ```

use std::sync::Arc;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Distribution, Tensor};
use nn_bench::*;

type B = NdArray<f32>;

fn bench_matmul(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    let device = NdArrayDevice::default();
    for &(label, m, k, n, max_iter, max_ms) in MATMUL_SHAPES {
        let a = Tensor::<B, 2>::random([m, k], Distribution::Normal(0.0, 1.0), &device);
        let b = Tensor::<B, 2>::random([k, n], Distribution::Normal(0.0, 1.0), &device);
        let (total, samples, resources) = time_op(
            || {
                let _ = a.clone().matmul(b.clone());
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
        let flops = 2.0 * (m * k * n) as f64 * run.iters as f64;
        run.gflops = Some(flops / (total as f64 / 1e9) / 1e9);
        println!(
            "matmul {:<14} {:>4}x{:>4}x{:>4}  iters={:>6}  mean={:>9.2}μs  GFLOP/s={:>6.2}",
            label,
            m,
            k,
            n,
            run.iters,
            run.mean_ns() / 1000.0,
            run.gflops.unwrap()
        );
        runs.push(run);
    }
}

fn bench_elemwise(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    let device = NdArrayDevice::default();
    for &(label, n, max_iter, max_ms) in ELEMWISE_SIZES {
        let a = Tensor::<B, 1>::random([n], Distribution::Normal(0.0, 1.0), &device);
        let b = Tensor::<B, 1>::random([n], Distribution::Normal(0.0, 1.0), &device);

        let (t, sm, res) = time_op(
            || {
                let _ = a.clone().add(b.clone());
            },
            max_iter,
            max_ms,
            gpu.clone(),
        );
        let mut r = Run::new("op", "add", format!("{}: n={}", label, n));
        r.n = n;
        r.iters = sm.len();
        r.total_ns = t;
        r.samples = sm;
        r.resources = res;
        println!(
            "add    {:<14} n={:>7}  iters={:>6}  mean={:>9.2}μs",
            label,
            n,
            r.iters,
            r.mean_ns() / 1000.0
        );
        runs.push(r);

        let (t, sm, res) = time_op(
            || {
                let _ = a.clone().mul(b.clone());
            },
            max_iter,
            max_ms,
            gpu.clone(),
        );
        let mut r = Run::new("op", "mul", format!("{}: n={}", label, n));
        r.n = n;
        r.iters = sm.len();
        r.total_ns = t;
        r.samples = sm;
        r.resources = res;
        runs.push(r);

        let (t, sm, res) = time_op(
            || {
                let _ = burn::tensor::activation::relu(a.clone());
            },
            max_iter,
            max_ms,
            gpu.clone(),
        );
        let mut r = Run::new("op", "relu", format!("{}: n={}", label, n));
        r.n = n;
        r.iters = sm.len();
        r.total_ns = t;
        r.samples = sm;
        r.resources = res;
        runs.push(r);

        let (t, sm, res) = time_op(
            || {
                let _ = burn::tensor::activation::sigmoid(a.clone());
            },
            max_iter,
            max_ms,
            gpu.clone(),
        );
        let mut r = Run::new("op", "sigmoid", format!("{}: n={}", label, n));
        r.n = n;
        r.iters = sm.len();
        r.total_ns = t;
        r.samples = sm;
        r.resources = res;
        runs.push(r);
    }
}

fn main() -> std::io::Result<()> {
    let (label, out) = parse_args("burn", "crates/ml/nn/bench_results/burn.json");
    let gpu = GpuHandle::try_init();

    println!("burn peer benchmark (ndarray backend): label={label}, out={out}");
    println!();

    let machine_json = capture_machine_info(gpu.as_ref());
    let mut runs: Vec<Run> = Vec::new();

    println!("── matmul ──");
    bench_matmul(&mut runs, gpu.clone());
    println!("\n── elementwise ──");
    bench_elemwise(&mut runs, gpu.clone());

    ensure_parent_dir(&out)?;
    write_results(&label, &machine_json, "burn-ndarray", &runs, &out)?;
    println!("\nwrote {} runs to {}", runs.len(), out);
    Ok(())
}
