//! Shared benchmark harness for the `nn` library and its peer
//! comparisons. Three things live here:
//!
//! 1. **`Sampler`** — a background thread that polls the WHOLE
//!    SYSTEM (not just this process) every 50 ms for CPU%, RAM%,
//!    thread count, and (if NVML init succeeds) GPU% + VRAM%.
//!    The point of "whole system" sampling is to make machine-
//!    level interference visible — if Slack ate a core during
//!    your `RL-mid` matmul, the chart row for that bench will
//!    show it.
//!
//! 2. **`Run`** — one (op, shape, label) measurement; carries
//!    timing samples + aggregated resource stats. Serialises to
//!    the schema v2 JSON shape the dashboard ingests.
//!
//! 3. **`time_op` / `capture_machine_info` / `write_results`** —
//!    the actual harness API used by each example driver.
//!
//! See `crates/ml/nn/bench_results/README.md` for the schema and
//! `docs/designs/nn_benchmark_harness.md` for the design rationale.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sysinfo::System;

// ─── Sampler ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Sample {
    pub cpu_total_pct: f32,
    pub mem_used_pct: f32,
    pub thread_count: u32,
    pub gpu_pct: Option<f32>,
    pub gpu_mem_pct: Option<f32>,
}

pub struct Sampler {
    samples: Arc<Mutex<Vec<Sample>>>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Sampler {
    pub fn start(interval_ms: u64, gpu: Option<Arc<GpuHandle>>) -> Self {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let s_clone = Arc::clone(&samples);
        let stop_clone = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let mut sys = System::new();
            sys.refresh_cpu_all();
            sys.refresh_memory();
            sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
            while !stop_clone.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(interval_ms));
                sys.refresh_cpu_all();
                sys.refresh_memory();
                sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
                let cpus = sys.cpus();
                let n = cpus.len().max(1) as f32;
                let avg = cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / n;
                let mem_pct = if sys.total_memory() > 0 {
                    (sys.used_memory() as f32 / sys.total_memory() as f32) * 100.0
                } else {
                    0.0
                };
                let thread_count: u32 = sys
                    .processes()
                    .values()
                    .map(|p| p.tasks().map(|t| t.len()).unwrap_or(1) as u32)
                    .sum();
                let (gpu_pct, gpu_mem_pct) = match gpu.as_deref() {
                    Some(h) => h.sample(),
                    None => (None, None),
                };
                s_clone.lock().unwrap().push(Sample {
                    cpu_total_pct: avg,
                    mem_used_pct: mem_pct,
                    thread_count,
                    gpu_pct,
                    gpu_mem_pct,
                });
            }
        });
        Self {
            samples,
            stop,
            handle: Some(handle),
        }
    }

    pub fn stop_and_drain(mut self) -> Vec<Sample> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        std::mem::take(&mut *self.samples.lock().unwrap())
    }
}

// ─── NVML GPU handle ────────────────────────────────────────────────

pub struct GpuHandle {
    inner: Mutex<nvml_wrapper::Nvml>,
    device_idx: u32,
    pub name: String,
    pub vram_total_mb: u64,
}

impl GpuHandle {
    pub fn try_init() -> Option<Arc<Self>> {
        let nvml = nvml_wrapper::Nvml::init().ok()?;
        let device = nvml.device_by_index(0).ok()?;
        let name = device.name().ok()?;
        let mem = device.memory_info().ok()?;
        let vram_total_mb = mem.total / (1024 * 1024);
        Some(Arc::new(Self {
            inner: Mutex::new(nvml),
            device_idx: 0,
            name,
            vram_total_mb,
        }))
    }
    pub fn sample(&self) -> (Option<f32>, Option<f32>) {
        let guard = self.inner.lock().unwrap();
        let device = match guard.device_by_index(self.device_idx) {
            Ok(d) => d,
            Err(_) => return (None, None),
        };
        let util = device.utilization_rates().ok();
        let mem = device.memory_info().ok();
        let u = util.map(|u| u.gpu as f32);
        let m = mem.map(|m| {
            if m.total > 0 {
                (m.used as f64 / m.total as f64 * 100.0) as f32
            } else {
                0.0
            }
        });
        (u, m)
    }
}

// ─── Stats + ResourceStats ──────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Stats {
    pub mean: f32,
    pub max: f32,
    pub p95: f32,
    pub min: f32,
}

impl Stats {
    pub fn from_samples<I: IntoIterator<Item = f32>>(it: I) -> Self {
        let mut v: Vec<f32> = it.into_iter().collect();
        if v.is_empty() {
            return Self {
                mean: 0.0,
                max: 0.0,
                p95: 0.0,
                min: 0.0,
            };
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let p95 = v[((v.len() as f32 * 0.95) as usize).min(v.len() - 1)];
        Self {
            mean,
            max: *v.last().unwrap(),
            p95,
            min: *v.first().unwrap(),
        }
    }
    fn to_json(&self) -> String {
        format!(
            r#"{{"mean":{:.2},"max":{:.2},"p95":{:.2},"min":{:.2}}}"#,
            self.mean, self.max, self.p95, self.min
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct ResourceStats {
    pub samples: usize,
    pub cpu_total_pct: Option<Stats>,
    pub mem_used_pct: Option<Stats>,
    pub thread_count: Option<Stats>,
    pub gpu_pct: Option<Stats>,
    pub gpu_mem_pct: Option<Stats>,
}

impl ResourceStats {
    pub fn from_samples(samples: &[Sample]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        Self {
            samples: samples.len(),
            cpu_total_pct: Some(Stats::from_samples(samples.iter().map(|s| s.cpu_total_pct))),
            mem_used_pct: Some(Stats::from_samples(samples.iter().map(|s| s.mem_used_pct))),
            thread_count: Some(Stats::from_samples(
                samples.iter().map(|s| s.thread_count as f32),
            )),
            gpu_pct: if samples.iter().any(|s| s.gpu_pct.is_some()) {
                Some(Stats::from_samples(
                    samples.iter().filter_map(|s| s.gpu_pct),
                ))
            } else {
                None
            },
            gpu_mem_pct: if samples.iter().any(|s| s.gpu_mem_pct.is_some()) {
                Some(Stats::from_samples(
                    samples.iter().filter_map(|s| s.gpu_mem_pct),
                ))
            } else {
                None
            },
        }
    }
    fn to_json(&self) -> String {
        let mut parts = vec![format!(r#""samples":{}"#, self.samples)];
        if let Some(ref s) = self.cpu_total_pct {
            parts.push(format!(r#""cpu_total_pct":{}"#, s.to_json()));
        }
        if let Some(ref s) = self.mem_used_pct {
            parts.push(format!(r#""mem_used_pct":{}"#, s.to_json()));
        }
        if let Some(ref s) = self.thread_count {
            parts.push(format!(r#""thread_count":{}"#, s.to_json()));
        }
        if let Some(ref s) = self.gpu_pct {
            parts.push(format!(r#""gpu_pct":{}"#, s.to_json()));
        }
        if let Some(ref s) = self.gpu_mem_pct {
            parts.push(format!(r#""gpu_mem_pct":{}"#, s.to_json()));
        }
        format!("{{{}}}", parts.join(","))
    }
}

// ─── Run ────────────────────────────────────────────────────────────

pub struct Run {
    pub category: &'static str,
    pub op: &'static str,
    pub shape: String,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub batch: usize,
    pub hidden: usize,
    pub iters: usize,
    pub total_ns: u128,
    pub samples: Vec<u64>,
    pub gflops: Option<f64>,
    pub resources: ResourceStats,
}

impl Run {
    pub fn new(category: &'static str, op: &'static str, shape: impl Into<String>) -> Self {
        Self {
            category,
            op,
            shape: shape.into(),
            m: 0,
            k: 0,
            n: 0,
            batch: 0,
            hidden: 0,
            iters: 0,
            total_ns: 0,
            samples: Vec::new(),
            gflops: None,
            resources: ResourceStats::default(),
        }
    }
    pub fn mean_ns(&self) -> f64 {
        self.total_ns as f64 / self.iters as f64
    }
    pub fn stddev_ns(&self) -> f64 {
        let m = self.mean_ns();
        let v: f64 = self
            .samples
            .iter()
            .map(|&x| (x as f64 - m).powi(2))
            .sum::<f64>()
            / self.iters.max(1) as f64;
        v.sqrt()
    }
    fn to_json_obj(&self) -> String {
        let mean_us = self.mean_ns() / 1000.0;
        let stddev_us = self.stddev_ns() / 1000.0;
        let total_ms = self.total_ns as f64 / 1_000_000.0;
        let gflops = self
            .gflops
            .map(|g| format!(r#""gflops":{:.3},"#, g))
            .unwrap_or_default();
        format!(
            r#"{{"category":"{}","op":"{}","shape":"{}","m":{},"k":{},"n":{},"batch":{},"hidden":{},"iters":{},"total_ms":{:.4},"mean_us":{:.4},"stddev_us":{:.4},{}"mean_ns":{:.1},"resources":{}}}"#,
            self.category,
            self.op,
            escape(&self.shape),
            self.m,
            self.k,
            self.n,
            self.batch,
            self.hidden,
            self.iters,
            total_ms,
            mean_us,
            stddev_us,
            gflops,
            self.mean_ns(),
            self.resources.to_json(),
        )
    }
}

fn escape(s: &str) -> String {
    s.replace('"', "\\\"")
}

// ─── Time-budgeted op runner ────────────────────────────────────────

pub fn time_op<F: FnMut()>(
    mut op: F,
    max_iters: usize,
    max_total_ms: u64,
    gpu: Option<Arc<GpuHandle>>,
) -> (u128, Vec<u64>, ResourceStats) {
    for _ in 0..3.min(max_iters) {
        op();
    }

    let sampler = Sampler::start(50, gpu);

    let mut samples = Vec::with_capacity(max_iters);
    let deadline = Instant::now() + Duration::from_millis(max_total_ms);
    let mut total: u128 = 0;
    for _ in 0..max_iters {
        let t = Instant::now();
        op();
        let dt = t.elapsed().as_nanos() as u64;
        samples.push(dt);
        total += dt as u128;
        if Instant::now() > deadline {
            break;
        }
    }

    let resource_samples = sampler.stop_and_drain();
    let resources = ResourceStats::from_samples(&resource_samples);
    (total, samples, resources)
}

// ─── Shape ladder (re-exported so every driver uses the same shapes) ─

pub const MATMUL_SHAPES: &[(&str, usize, usize, usize, usize, u64)] = &[
    ("widget-tiny", 4, 2, 8, 50_000, 600),
    ("widget-small", 32, 16, 16, 50_000, 600),
    ("widget-med", 64, 64, 64, 20_000, 1000),
    ("RL-small", 128, 32, 32, 20_000, 1000),
    ("RL-mid", 256, 128, 128, 5_000, 2000),
    ("RL-large", 512, 256, 256, 1_000, 3000),
    ("GPU-warm", 1024, 512, 512, 100, 5000),
    ("GPU-hot", 2048, 1024, 1024, 20, 8000),
];

pub const ELEMWISE_SIZES: &[(&str, usize, usize, u64)] = &[
    ("widget-tiny", 64, 100_000, 600),
    ("widget-small", 256, 100_000, 600),
    ("widget-med", 4_096, 20_000, 1000),
    ("RL-mid", 16_384, 10_000, 1000),
    ("RL-large", 262_144, 1_000, 2000),
    ("GPU-warm", 1_048_576, 200, 3000),
];

pub const TRAIN_CONFIGS: &[(&str, usize, usize, usize, u64)] = &[
    ("widget-tiny", 4, 8, 20_000, 1000),
    ("widget-small", 32, 32, 10_000, 1500),
    ("widget-med", 32, 128, 5_000, 2000),
    ("RL-mid", 256, 128, 1_000, 3000),
    ("RL-large", 256, 512, 200, 4000),
];

// ─── Machine info ───────────────────────────────────────────────────

pub fn capture_machine_info(gpu: Option<&Arc<GpuHandle>>) -> String {
    let mut sys = System::new();
    sys.refresh_all();
    let cpus = sys.cpus();
    let cpu_brand = cpus
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "?".to_string());
    let logical = cpus.len();
    let physical = sys.physical_core_count().unwrap_or(logical);
    let ram_total_mb = sys.total_memory() / (1024 * 1024);

    let os = format!(
        "{} {}",
        System::name().unwrap_or_else(|| "?".to_string()),
        System::os_version().unwrap_or_else(|| "?".to_string())
    );
    let kernel = System::kernel_version().unwrap_or_else(|| "?".to_string());

    let gpu_json = match gpu {
        Some(h) => format!(
            r#"{{"available":true,"name":"{}","vram_total_mb":{}}}"#,
            escape(&h.name),
            h.vram_total_mb
        ),
        None => r#"{"available":false}"#.to_string(),
    };

    format!(
        r#"{{"os":"{}","kernel":"{}","cpu_brand":"{}","cpu_cores_physical":{},"cpu_cores_logical":{},"ram_total_mb":{},"gpu":{}}}"#,
        escape(&os),
        escape(&kernel),
        escape(&cpu_brand),
        physical,
        logical,
        ram_total_mb,
        gpu_json,
    )
}

// ─── JSON writer ────────────────────────────────────────────────────

pub fn write_results(
    label: &str,
    machine_json: &str,
    library_features: &str,
    runs: &[Run],
    path: &str,
) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let target_arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "other"
    };

    writeln!(f, "{{")?;
    writeln!(f, r#"  "schema": 2,"#)?;
    writeln!(f, r#"  "label": "{label}","#)?;
    writeln!(f, r#"  "metadata": {{"#)?;
    writeln!(f, r#"    "timestamp": {timestamp},"#)?;
    writeln!(f, r#"    "features": "{library_features}","#)?;
    writeln!(f, r#"    "arch": "{target_arch}""#)?;
    writeln!(f, r#"  }},"#)?;
    writeln!(f, r#"  "machine": {machine_json},"#)?;
    writeln!(f, r#"  "runs": ["#)?;
    for (i, run) in runs.iter().enumerate() {
        let comma = if i + 1 == runs.len() { "" } else { "," };
        writeln!(f, "    {}{}", run.to_json_obj(), comma)?;
    }
    writeln!(f, r#"  ]"#)?;
    writeln!(f, "}}")?;
    Ok(())
}

pub fn parse_args(default_label: &str, default_out: &str) -> (String, String) {
    let args: Vec<String> = std::env::args().collect();
    let mut label = default_label.to_string();
    let mut out = default_out.to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--label" if i + 1 < args.len() => {
                label = args[i + 1].clone();
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }
    (label, out)
}

pub fn ensure_parent_dir(path: &str) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}
