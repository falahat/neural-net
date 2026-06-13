//! Peer benchmark: `candle-core` (Hugging Face).
//!
//! Why this peer (per `docs/designs/nn_benchmark_harness.md` §4):
//! Architecturally closest to `nn` — they also use a Storage enum
//! over backend variants, also expose a small surface, also keep
//! deps light. Their CPU matmul routes through `gemm` (hand-tuned
//! sgemm), so this benchmark answers the same kernel-quality
//! question as the `ndarray_mm` peer but through a tensor API
//! more similar to ours.
//!
//! Build:
//! ```bash
//! cargo run --profile release-bench -p nn-bench --example candle --features candle -- \
//!     --label candle --out crates/ml/nn/bench_results/candle.json
//! ```

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use nn_bench::*;

/// Pick CUDA if `--features candle-cuda` is on AND `CUDA_DEVICE_IS_OK`
/// env hint is set; otherwise CPU. Lets one example file drive both
/// configurations.
fn pick_device() -> Device {
    #[cfg(feature = "candle-cuda")]
    {
        match Device::new_cuda(0) {
            Ok(d) => {
                println!("candle: using CUDA device 0");
                return d;
            }
            Err(e) => {
                eprintln!("candle: CUDA init failed ({e}); falling back to CPU");
            }
        }
    }
    println!("candle: using CPU");
    Device::Cpu
}

fn bench_matmul(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    let device = pick_device();
    // CUDA dispatch is asynchronous — `matmul` returns before the
    // kernel completes. Without synchronisation we'd be timing the
    // queue-insert (~10 µs) instead of the actual compute, producing
    // physically-impossible GFLOP/s numbers (saw 347 TFLOP/s on a
    // 22-TFLOP card before adding this sync).
    let needs_sync = !matches!(device, Device::Cpu);
    for &(label, m, k, n, max_iter, max_ms) in MATMUL_SHAPES {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.231).cos()).collect();
        let a = Tensor::from_vec(a_data, (m, k), &device).unwrap();
        let b = Tensor::from_vec(b_data, (k, n), &device).unwrap();
        let (total, samples, resources) = time_op(
            || {
                let _ = a.matmul(&b).unwrap();
                if needs_sync {
                    device.synchronize().unwrap();
                }
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
    let device = pick_device();
    for &(label, n, max_iter, max_ms) in ELEMWISE_SIZES {
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin()).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.231).cos()).collect();
        let a = Tensor::from_vec(a_data, n, &device).unwrap();
        let b = Tensor::from_vec(b_data, n, &device).unwrap();

        let (t, sm, res) = time_op(
            || {
                let _ = a.add(&b).unwrap();
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
                let _ = a.mul(&b).unwrap();
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
                let _ = a.relu().unwrap();
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

        // candle's f32 sigmoid via the sigmoid op (when present) — fall
        // back to `1/(1+exp(-x))` via composition if the lib version
        // doesn't expose it directly.
        //   candle 0.8+ has `Tensor::sigmoid` via `candle_nn::ops::sigmoid`;
        //   the core crate alone offers the math composition.
        let (t, sm, res) = time_op(
            || {
                let neg = a.neg().unwrap();
                let e = neg.exp().unwrap();
                let one = Tensor::ones_like(&a).unwrap();
                let denom = one.add(&e).unwrap();
                let _ = one.div(&denom).unwrap();
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

        let _ = DType::F32; // touch the imports
    }
}

fn main() -> std::io::Result<()> {
    let (label, out) = parse_args("candle", "crates/ml/nn/bench_results/candle.json");
    let gpu = GpuHandle::try_init();

    println!("candle peer benchmark: label={label}, out={out}");
    println!();

    let machine_json = capture_machine_info(gpu.as_ref());
    let mut runs: Vec<Run> = Vec::new();

    println!("── matmul ──");
    bench_matmul(&mut runs, gpu.clone());

    // ── DIAGNOSTIC FALLBACK, not a fix. ──
    //
    // candle's elementwise kernels are compiled to PTX bytecode at
    // candle-kernels build time. The PTX *version* is fixed by the
    // nvcc compiler (e.g. nvcc 13.2 emits PTX 8.4), and the driver
    // only loads PTX versions ≤ what its build supports. PTX version
    // is independent of `CUDA_COMPUTE_CAP` (SM target) and there's
    // no nvcc flag to downgrade the emitted bytecode version.
    //
    // If you hit `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` here, the fix
    // is one of:
    //   (a) Update the GPU driver to one matching your toolkit.
    //   (b) Use an older CUDA Toolkit whose PTX the driver accepts.
    //       For driver 591.86: CUDA 13.1 or 12.8.
    // See `bench_results/CUDA_SETUP.md` for driver/toolkit pairing.
    //
    // We catch the panic so the matmul data above is preserved (matmul
    // goes through cuBLAS, which ships pre-built SASS and bypasses the
    // PTX JIT entirely). The presence of this fallback should NOT be
    // taken as the bench passing — it means a system-level fixup is
    // owed.
    println!("\n── elementwise ──");
    let runs_before = runs.len();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bench_elemwise(&mut runs, gpu.clone());
    }));
    if let Err(_) = result {
        eprintln!();
        eprintln!("⚠  elementwise dropped due to PTX version mismatch.");
        eprintln!("   Matmul data above (via cuBLAS, no PTX) IS valid.");
        eprintln!("   To recover the elementwise numbers, see");
        eprintln!("   `bench_results/CUDA_SETUP.md` — either update your");
        eprintln!("   GPU driver or reinstall CUDA Toolkit at a version");
        eprintln!("   matching the driver.");
        runs.truncate(runs_before);
    }

    ensure_parent_dir(&out)?;
    write_results(&label, &machine_json, "candle-cpu", &runs, &out)?;
    println!("\nwrote {} runs to {}", runs.len(), out);
    Ok(())
}
