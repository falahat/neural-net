# neural-net

A from-scratch neural-network library in Rust — reverse-mode autodiff over a
Wengert tape, MLP modules, hot-swappable optimisers / losses / activations,
numerically-stable math, and a benchmark harness that compares against peer
libraries (candle, ndarray, burn, dfdx, tch). Every line of the `nn` crate is
hand-written; borrowed *ideas* (Wengert tape, Adam, Glorot/He init, Welford,
Box–Muller, log-sum-exp) are cited at point of use.

## Crates

- [`crates/engine/nn`](crates/engine/nn) — the library (autodiff, modules,
  optimisers, losses, numerically-stable math, optional SIMD / parallel / GPU /
  WASM backends).
- [`crates/engine/nn-bench`](crates/engine/nn-bench) — the benchmark harness +
  per-library example drivers (the heavy peer-lib deps live here so they don't
  pollute `nn`'s tree).

## Build

```sh
cargo test -p nn
cargo run --profile release-bench --example matmul_bench -p nn
```

Optional `nn` features: `simd` (portable f32x8), `parallel` (multi-threaded
sgemm via `gemm`), `gpu` / `gpu-cuda` (CubeCL), `wasm` (wasm-bindgen bridge).
The GPU feature on Windows DX12 needs the `[patch.crates-io]` one-liner noted in
`crates/engine/nn/src/backend/gpu.rs`.

Extracted from the [`falahat/simulator`](https://github.com/falahat/simulator)
monorepo; the full design rationale + attribution table lives there in
`docs/designs/neural_network_library.md`.
