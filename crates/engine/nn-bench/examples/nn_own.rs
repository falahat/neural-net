//! Benchmark the `nn` library itself. Same harness as the peer
//! libraries — outputs JSON in the same schema so the dashboard
//! can plot all of them on shared axes.

use std::sync::Arc;

use nn_bench::*;

use nn::activation::Tanh;
use nn::backend::cpu;
use nn::init::Init;
use nn::loss::Mse;
use nn::module::{Linear, Sequential};
use nn::optim::Adam;
use nn::rng::SplitMix64;
use nn::tensor::Tensor;
use nn::train::Trainer;

fn bench_matmul(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    for &(label, m, k, n, max_iter, max_ms) in MATMUL_SHAPES {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.231).cos()).collect();
        let (total, samples, resources) = time_op(
            || {
                let _ = cpu::matmul(&a, &[m, k], &b, &[k, n]);
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
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.231).cos()).collect();

        let (t, sm, res) = time_op(
            || {
                let _ = cpu::add(&a, &b);
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
                let _ = cpu::mul(&a, &b);
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
                let _ = cpu::map_unary(&a, |x| x.max(0.0));
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
                let _ = cpu::map_unary(&a, nn::math::stable::sigmoid);
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

fn bench_train_step(runs: &mut Vec<Run>, gpu: Option<Arc<GpuHandle>>) {
    for &(label, batch, hidden, max_iter, max_ms) in TRAIN_CONFIGS {
        let mut rng = SplitMix64::seeded(42);
        let mut trainer = Trainer::builder()
            .model(Box::new(Sequential::new(vec![
                Box::new(Linear::new(4, hidden, Init::Xavier, &mut rng)),
                Box::new(Tanh),
                Box::new(Linear::new(hidden, 1, Init::Xavier, &mut rng)),
            ])))
            .loss(Box::new(Mse))
            .optim(Box::new(Adam::new(0.01)))
            .build();

        let xs: Vec<f32> = (0..batch * 4).map(|i| (i as f32 * 0.07).sin()).collect();
        let ys: Vec<f32> = (0..batch).map(|i| (i as f32 * 0.13).cos()).collect();
        let x = Tensor::from_data(xs, &[batch, 4]);
        let y = Tensor::from_data(ys, &[batch, 1]);

        let (t, sm, res) = time_op(
            || {
                trainer.train_step(&x, &y);
            },
            max_iter,
            max_ms,
            gpu.clone(),
        );
        let mut run = Run::new(
            "e2e",
            "train_step",
            format!("{}: batch={} hidden={}", label, batch, hidden),
        );
        run.batch = batch;
        run.hidden = hidden;
        run.iters = sm.len();
        run.total_ns = t;
        run.samples = sm;
        run.resources = res;
        println!(
            "train  {:<14} batch={:<3} hidden={:<4} iters={:>6}  mean={:>9.2}μs",
            label,
            batch,
            hidden,
            run.iters,
            run.mean_ns() / 1000.0
        );
        runs.push(run);
    }
}

fn main() -> std::io::Result<()> {
    let default_label = if cfg!(feature = "simd") {
        "nn-simd"
    } else {
        "nn-scalar"
    };
    let (label, out) = parse_args(default_label, "crates/ml/nn/bench_results/scalar.json");
    let gpu = GpuHandle::try_init();
    let features = if cfg!(feature = "simd") {
        "simd"
    } else {
        "default"
    };

    println!("nn-bench: label={label}, out={out}");
    println!("backend: {features}");
    println!();

    let machine_json = capture_machine_info(gpu.as_ref());
    let mut runs: Vec<Run> = Vec::new();

    println!("── matmul ──");
    bench_matmul(&mut runs, gpu.clone());
    println!("\n── elementwise ──");
    bench_elemwise(&mut runs, gpu.clone());
    println!("\n── train_step ──");
    bench_train_step(&mut runs, gpu.clone());

    ensure_parent_dir(&out)?;
    write_results(&label, &machine_json, features, &runs, &out)?;
    println!("\nwrote {} runs to {}", runs.len(), out);
    Ok(())
}
