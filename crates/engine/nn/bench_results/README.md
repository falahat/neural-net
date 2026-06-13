# nn — performance benchmark results

This directory holds the raw JSON results + the generated HTML
dashboard for the `nn` crate's benchmark suite.

## How to (re)run

Every driver lives in `crates/engine/nn-bench/examples/`. The harness
itself (Sampler, Run, JSON writers) is `nn-bench`'s `lib.rs` so
each peer driver reuses the same machine-info capture + 50ms
system-wide resource sampling.

```bash
# ── nn itself, scalar (default — LLVM autovec on AVX2). ──
cargo run --profile release-bench -p nn-bench --example nn_own -- \
    --label nn-scalar --out crates/engine/nn/bench_results/scalar.json

# ── nn itself, SIMD (portable f32x8 via `wide`). ──
cargo run --profile release-bench -p nn-bench --example nn_own --features simd -- \
    --label nn-simd --out crates/engine/nn/bench_results/simd.json

# ── peer: ndarray + matrixmultiply (hand-tuned sgemm, no autograd). ──
cargo run --profile release-bench -p nn-bench --example ndarray_mm -- \
    --label ndarray-mm --out crates/engine/nn/bench_results/ndarray_mm.json

# ── peer: candle-core (Hugging Face; uses the `gemm` crate for sgemm). ──
cargo run --profile release-bench -p nn-bench --example candle --features candle -- \
    --label candle --out crates/engine/nn/bench_results/candle.json

# Combine ALL *.json in this dir into a single self-contained HTML
# dashboard. Picks up new files automatically.
cargo run --release --example build_viz -p nn
```

## Peer libraries

All five peers from `docs/designs/nn_benchmark_harness.md` §4
have drivers. Four run out of the box; one needs libtorch
installed.

| # | Library | Status | Driver | How to run |
|---|---|---|---|---|
| 1 | **candle-core** (Hugging Face) | ✓ runs | `examples/candle.rs` | `--features candle` |
| 2 | **ndarray + matrixmultiply** | ✓ runs | `examples/ndarray_mm.rs` | no features |
| 3 | **burn-ndarray** (tracel-ai) | ✓ runs | `examples/burn.rs` | `--features burn` |
| 4 | **dfdx** (coreylowman) | ✓ runs | `examples/dfdx.rs` | `--features dfdx` |
| 5 | **tch-rs** (libtorch) | code complete · libtorch required | `examples/tch.rs` | `--features tch`; needs `LIBTORCH` env var |

To enable `tch-rs`: download libtorch from pytorch.org, point
`LIBTORCH=<install path>`, then run the example. See the driver's
docstring for the full setup recipe (Linux/macOS/Windows specifics).

When you add a new peer driver, drop it in
`crates/engine/nn-bench/examples/<name>.rs` and write its JSON to a
new file in this directory. The dashboard picks it up
automatically on the next `build_viz` run.

## Peak matmul on the current dev machine

Intel Core Ultra 9 275HX (24 cores) + RTX 5070 Ti Laptop GPU.

| Library | Peak GFLOP/s | At shape | Notes |
|---|---:|---|---|
| **nn-parallel** | **1026** | 2048×1024×1024 | **us, `--features parallel`** — `gemm` crate. |
| **candle** | 852 | 2048×1024×1024 | also `gemm` crate. |
| **dfdx** | 749 | 2048×1024×1024 | also `gemm`. |
| **burn-ndarray** | 208 | 2048×1024×1024 | multi-threaded but ndarray backend. |
| **ndarray-mm** | 119 | 512×256×256 | single-thread `matrixmultiply::sgemm`. |
| **nn-scalar** | 27 | 256×128×128 | our LLVM-autovec baseline. |
| **nn-simd** | 25 | 256×128×128 | our hand-SIMD path. |
| **nn-gpu** | 12.7 | 1024×512×512 | **us, `--features gpu`** — cubecl/wgpu v1 (naive kernel; tiling + pooling still to come). |

Take-aways:

1. **`--features parallel` makes us competitive** with candle / dfdx at
   the cost of bit-determinism. The default scalar path remains
   bit-exact for `tests/determinism.rs`; the parallel path is gated.
2. **GPU works** but isn't winning yet — our v1 cubecl kernel is
   one-thread-per-output-element + per-call buffer transfer. Tiled
   kernel + buffer pooling is the next chunk.
3. **The middle band** (ndarray-mm at 119 GFLOP/s, single-thread)
   used to be option 1 in the proposal but is now superseded by
   `--features parallel`. If we ever want a deterministic-but-fast
   middle ground (e.g. for the determinism CI lane), we could add
   a `--features matmul-mm` flag pointing at `matrixmultiply::sgemm`.

Open `viz.html` directly in a browser — no server needed (data is
inlined, Observable Plot loads from a CDN).

## What's in the JSON

Schema v2 (`docs/designs/nn_benchmark_harness.md` for the rationale):

```json
{
  "schema": 2,
  "label": "nn-scalar",
  "metadata": { "timestamp": ..., "features": "default", "arch": "x86_64" },

  "machine": {
    "os": "Windows 11 (26200)",
    "cpu_brand": "Intel(R) Core(TM) Ultra 9 275HX",
    "cpu_cores_physical": 24,
    "cpu_cores_logical": 24,
    "ram_total_mb": 32189,
    "gpu": { "available": true, "name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU", "vram_total_mb": 12227 }
  },

  "runs": [
    {
      "category": "op", "op": "matmul",
      "shape": "RL-mid: 256x128x128", "m": 256, "k": 128, "n": 128,
      "iters": 5000, "total_ms": ..., "mean_us": ..., "stddev_us": ..., "gflops": 25.32,

      "resources": {
        "samples": 47,
        "cpu_total_pct":   { "mean": 18.3, "max": 33.2, "p95": 30.1, "min": 11.0 },
        "mem_used_pct":    { "mean": 81.3, "max": 81.8, "p95": 81.6, "min": 81.0 },
        "thread_count":    { "mean": 445,  "max": 446,  "p95": 446,  "min": 444  },
        "gpu_pct":         { "mean": 0.0,  "max": 0.0,  "p95": 0.0,  "min": 0.0  },
        "gpu_mem_pct":     { "mean": 6.1,  "max": 6.1,  "p95": 6.1,  "min": 6.1  }
      }
    },
    ...
  ]
}
```

**Resource sampling is system-wide, not per-process.** A background
thread polls `sysinfo::System` (CPU / memory / thread count) and
`nvml-wrapper` (GPU utilisation / VRAM) every 50ms during each
benchmark, then reports `mean / max / p95 / min` of each. The point
is to make MACHINE-LEVEL pressure visible — if another process was
hammering the CPU when a run executed, you'll see it here instead
of accepting the timing as gospel.

Very fast benchmarks (e.g. `matmul widget-tiny` at ~5 ms total) may
complete in under 50 ms and so have `samples: 0`. The dashboard
notes this; sub-50 ms work doesn't generate meaningful pressure
signal anyway.

## Adding more configs

Drop another labelled JSON file in this directory and re-run
`build_viz`. The dashboard auto-picks it up — e.g. a future
`candle.json` from the cross-library lane will appear alongside
the existing scalar / SIMD curves.

See `docs/designs/nn_benchmark_harness.md` §4 for the five peer
libraries the cross-library lane targets.
