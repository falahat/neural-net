//! Peer benchmark: `tch-rs` — Rust bindings to libtorch.
//!
//! Why this peer: this IS PyTorch (well, libtorch — the C++ side).
//! Sets the upper bound on "what's actually fast on this hardware"
//! because PyTorch routes matmul through MKL / oneDNN / cuBLAS.
//!
//! **Setup requirements** (this peer is the only one with a native
//! prerequisite):
//!   1. Download libtorch from https://pytorch.org/get-started/locally/
//!      (CPU build is fine for this bench).
//!   2. Unzip somewhere; set `LIBTORCH=<that path>` in the shell.
//!   3. On Linux/macOS, also extend `LD_LIBRARY_PATH` (or
//!      `DYLD_LIBRARY_PATH` on macOS) to <LIBTORCH>/lib.
//!   4. Or set `LIBTORCH_USE_PYTORCH=1` if you have PyTorch installed
//!      via pip and want tch-rs to use that.
//!
//! Build:
//! ```bash
//! cargo run --profile release-bench -p nn-bench --example tch --features tch -- \
//!     --label tch --out crates/ml/nn/bench_results/tch.json
//! ```

use std::sync::Arc;

use nn_bench::*;
use tch::{Device, Kind, Tensor};

fn bench_matmul(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    for &(label, m, k, n, max_iter, max_ms) in MATMUL_SHAPES {
        let a = Tensor::randn([m as i64, k as i64], (Kind::Float, Device::Cpu));
        let b = Tensor::randn([k as i64, n as i64], (Kind::Float, Device::Cpu));
        let (total, samples, resources) = time_op(
            || {
                let _ = a.matmul(&b);
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
    for &(label, n, max_iter, max_ms) in ELEMWISE_SIZES {
        let a = Tensor::randn([n as i64], (Kind::Float, Device::Cpu));
        let b = Tensor::randn([n as i64], (Kind::Float, Device::Cpu));

        let (t, sm, res) = time_op(
            || {
                let _ = &a + &b;
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
                let _ = &a * &b;
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
                let _ = a.relu();
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
                let _ = a.sigmoid();
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
    let (label, out) = parse_args("tch", "crates/ml/nn/bench_results/tch.json");
    let gpu = GpuHandle::try_init();

    println!("tch-rs (libtorch) peer benchmark: label={label}, out={out}");
    println!();

    let machine_json = capture_machine_info(gpu.as_ref());
    let mut runs: Vec<Run> = Vec::new();

    println!("── matmul ──");
    bench_matmul(&mut runs, gpu.clone());
    println!("\n── elementwise ──");
    bench_elemwise(&mut runs, gpu.clone());

    ensure_parent_dir(&out)?;
    write_results(&label, &machine_json, "tch-cpu", &runs, &out)?;
    println!("\nwrote {} runs to {}", runs.len(), out);
    Ok(())
}
