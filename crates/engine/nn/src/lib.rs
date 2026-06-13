//! # `nn` — from-scratch autodiff + MLP library
//!
//! Built for the simulator and the textbook's live widgets. Every
//! line of compute is ours; borrowed *ideas* (log-sum-exp, Adam,
//! Xavier/He, Box-Muller, Welford, BLIS tiling) are cited where
//! they appear. Design: `docs/designs/neural_network_library.md`.
//! Benchmarks: `bench_results/viz.html`.
//!
//! ## Module map
//!
//! | Concern | Module |
//! |---|---|
//! | Tensors | [`tensor`] |
//! | Autograd tape + Op enum | [`autograd`] |
//! | Compute kernels | [`backend`] (cpu / simd / parallel / gpu) |
//! | Stable numerics | [`math::stable`] |
//! | Layers | [`module`] |
//! | Activations | [`activation`] |
//! | Losses | [`loss`] |
//! | Optimisers | [`optim`] |
//! | RNG + init | [`rng`], [`init`] |
//! | Training loop | [`train`] |
//! | Save/load | [`checkpoint`] |
//!
//! ## 30-second tour
//!
//! ```
//! use nn::*;
//! let mut rng = SplitMix64::seeded(42);
//! let mut trainer = Trainer::builder()
//!     .model(Box::new(Sequential::new(vec![
//!         Box::new(Linear::new(2, 16, Init::He, &mut rng)),
//!         Box::new(Relu),
//!         Box::new(Linear::new(16, 1, Init::Xavier, &mut rng)),
//!     ])))
//!     .loss(Box::new(loss::Mse))
//!     .optim(Box::new(optim::Adam::new(0.01)))
//!     .build();
//!
//! // XOR: 4 rows of 2 inputs, 4 rows of 1 target.
//! let inputs = Tensor::from_data(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
//! let targets = Tensor::from_data(vec![0.0, 1.0, 1.0, 0.0], &[4, 1]);
//!
//! let first = trainer.train_step(&inputs, &targets);
//! for _ in 0..200 { trainer.train_step(&inputs, &targets); }
//! let last = trainer.train_step(&inputs, &targets);
//! assert!(last < first); // the loss falls as it learns
//!
//! // Hot-swap loss / optim mid-training — every field on Trainer is Box<dyn …>,
//! // ParamIds are preserved so optimiser state initialises lazily on the new path.
//! trainer.loss  = Box::new(loss::Huber { delta: 1.0 });
//! trainer.optim = Box::new(optim::Sgd::new(0.05));
//! trainer.train_step(&inputs, &targets); // keeps training on the new path
//! ```
//!
//! ## Features
//!
//! | Feature | Effect |
//! |---|---|
//! | (default) | scalar CPU; deterministic. `tests/determinism.rs` only runs here. |
//! | `simd` | f32x8 via `wide`. Still deterministic. Wins at small shapes + WASM. |
//! | `parallel` | multi-threaded sgemm via `gemm` crate. **Not bit-deterministic.** |
//! | `gpu` | cubecl/wgpu naive matmul. Cross-platform. |
//! | `gpu-cuda` | cubecl CUDA backend. Needs CUDA toolkit. |
//! | `wasm` | wasm-bindgen surface for browser widgets. |

pub mod activation;
pub mod autograd;
pub mod backend;
pub mod init;
pub mod loss;
pub mod math;
pub mod module;
pub mod optim;
pub mod rng;
pub mod tensor;
pub mod train;

pub mod checkpoint;

#[cfg(feature = "wasm")]
pub mod wasm;

// Public surface re-exports. Anything a typical user touches is at the
// crate root. Specialised types stay in their submodules.
pub use crate::activation::{Gelu, LeakyRelu, Relu, Sigmoid, Tanh};
pub use crate::autograd::{GradStore, ParamId, Tape};
pub use crate::init::Init;
pub use crate::module::{Linear, Module, ParamVisitor, ParamVisitorMut, Sequential};
pub use crate::rng::{Rng, SplitMix64};
pub use crate::tensor::Tensor;
pub use crate::train::{Callback, Trainer, TrainerBuilder};
