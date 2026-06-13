//! Compute primitives — one module per execution target.
//!
//! `cpu.rs` is the public entry point; its matmul/elementwise
//! functions dispatch via `#[cfg(feature = …)]` to the highest-
//! effort variant enabled:
//!
//! ```text
//!     parallel  >  simd  >  scalar      (mutually exclusive at the cpu.rs level)
//! ```
//!
//! Scalar is always present as the deterministic fallback. GPU is
//! reached separately via `backend::gpu::{matmul, GpuContext}` —
//! it's async and needs device/buffer management that doesn't fit
//! a synchronous fn-on-slice signature.
//!
//! Five primitives cover every layer we ship: matmul (2D@2D),
//! elementwise unary, elementwise binary, reductions, shape ops.
//! Conv / LayerNorm / softmax / attention would all compose from
//! these — none are implemented yet because no widget needs them.

pub mod cpu;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(any(feature = "gpu", feature = "gpu-cuda"))]
pub mod gpu;
