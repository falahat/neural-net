//! Peer benchmark: `ndarray + matrixmultiply` baseline.
//!
//! Why this peer (per `docs/designs/nn_benchmark_harness.md` §4):
//! `matrixmultiply` is the canonical hand-tuned sgemm crate in the
//! Rust ecosystem (used by `ndarray.dot()` for f32). It's NOT a
//! neural-net library — no autograd, no losses, no optimisers — but
//! it answers the question "how much of the gap between our matmul
//! and `candle`'s is *just* missing sgemm vs. missing autograd
//! overhead?" If swapping our matmul kernel for `matrixmultiply::sgemm`
//! closes 80%+ of the gap, we know where to optimise.
//!
//! Only matmul + elementwise here — no train_step (would require us
//! to hand-roll backprop, defeating the point of using ndarray).

use std::sync::Arc;

use ndarray::Array2;
use nn_bench::*;

/// Hand-call into `matrixmultiply::sgemm`. The flat-Vec layout +
/// row-major strides match ndarray's default.
fn sgemm_via_mm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1, // row-stride = k, col-stride = 1
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

fn bench_matmul(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    for &(label, m, k, n, max_iter, max_ms) in MATMUL_SHAPES {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.231).cos()).collect();
        let mut c: Vec<f32> = vec![0.0; m * n];
        let (total, samples, resources) = time_op(
            || {
                sgemm_via_mm(&a, &b, &mut c, m, k, n);
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
        // Build as 1-D ndarrays so we exercise ndarray's broadcast machinery
        // (not just raw slice arithmetic — that would be the same as ours).
        let a = ndarray::Array1::from_vec((0..n).map(|i| (i as f32 * 0.137).sin()).collect());
        let b = ndarray::Array1::from_vec((0..n).map(|i| (i as f32 * 0.231).cos()).collect());

        // add — `&a + &b` returns a new Array1.
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

        // mul — elementwise product (ndarray Array1 * Array1).
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

        // relu — `mapv` allocates a new array; closest analog to our `map_unary`.
        let (t, sm, res) = time_op(
            || {
                let _ = a.mapv(|x| x.max(0.0));
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

        // sigmoid via stable branched form (same numerical contract as ours).
        let (t, sm, res) = time_op(
            || {
                let _ = a.mapv(|x| {
                    if x >= 0.0 {
                        1.0 / (1.0 + (-x).exp())
                    } else {
                        let e = x.exp();
                        e / (1.0 + e)
                    }
                });
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
    // Marker so a future reader sees we DID also try ndarray's `.dot()`
    // (which dispatches to matrixmultiply); identical numbers, so we only
    // keep the direct-sgemm matmul series above to avoid double-reporting.
    let _ = Array2::<f32>::zeros((1, 1));
}

fn main() -> std::io::Result<()> {
    let (label, out) = parse_args("ndarray-mm", "crates/ml/nn/bench_results/ndarray_mm.json");
    let gpu = GpuHandle::try_init();

    println!("ndarray_mm peer benchmark: label={label}, out={out}");
    println!();

    let machine_json = capture_machine_info(gpu.as_ref());
    let mut runs: Vec<Run> = Vec::new();

    println!("── matmul (matrixmultiply::sgemm) ──");
    bench_matmul(&mut runs, gpu.clone());
    println!("\n── elementwise (ndarray Array1 ops) ──");
    bench_elemwise(&mut runs, gpu.clone());

    ensure_parent_dir(&out)?;
    write_results(&label, &machine_json, "default", &runs, &out)?;
    println!("\nwrote {} runs to {}", runs.len(), out);
    Ok(())
}
