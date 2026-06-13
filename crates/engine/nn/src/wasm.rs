//! WASM bridge — `WasmMlp` exposes the library through `wasm-bindgen`.
//!
//! A widget instantiates `new WasmMlp([2, 16, 1], 42)` from JS, calls
//! `set_loss("mse")` / `set_optim("adam", 0.01)` to configure, then
//! drives training step-by-step with `train_step(x, y) -> loss`. The
//! loss / optim setters are the hot-swap demo's surface.

use wasm_bindgen::prelude::*;

use crate::activation::{Relu, Sigmoid, Tanh};
use crate::init::Init;
use crate::loss::{CrossEntropy, Huber, Mse};
use crate::module::{Linear, Sequential};
use crate::optim::{Adam, AdamW, Sgd};
use crate::rng::SplitMix64;
use crate::tensor::Tensor;
use crate::train::Trainer;

#[wasm_bindgen]
pub struct WasmMlp {
    trainer: Trainer,
    in_dim: usize,
    out_dim: usize,
}

#[wasm_bindgen]
impl WasmMlp {
    /// Build an MLP from `layer_sizes`. Activation between hidden
    /// layers is ReLU; the last layer has no activation (linear
    /// output — chain with a loss that expects logits / raw output).
    ///
    /// `layer_sizes` is `[input, hidden_1, ..., hidden_k, output]`.
    #[wasm_bindgen(constructor)]
    pub fn new(layer_sizes: Vec<u32>, seed: u32) -> Result<WasmMlp, JsValue> {
        if layer_sizes.len() < 2 {
            return Err(JsValue::from_str(
                "layer_sizes must have at least 2 entries",
            ));
        }
        let mut rng = SplitMix64::seeded(seed as u64);
        let mut layers: Vec<Box<dyn crate::module::Module>> = Vec::new();
        let n = layer_sizes.len();
        for i in 0..(n - 1) {
            let in_dim = layer_sizes[i] as usize;
            let out_dim = layer_sizes[i + 1] as usize;
            layers.push(Box::new(Linear::new(in_dim, out_dim, Init::He, &mut rng)));
            if i + 2 < n {
                // ReLU on every layer except the output one
                layers.push(Box::new(Relu));
            }
        }
        let trainer = Trainer::builder()
            .model(Box::new(Sequential::new(layers)))
            .loss(Box::new(Mse))
            .optim(Box::new(Adam::new(0.01)))
            .build();
        Ok(WasmMlp {
            trainer,
            in_dim: layer_sizes[0] as usize,
            out_dim: layer_sizes[n - 1] as usize,
        })
    }

    /// Hot-swap the loss without rebuilding the network. The next
    /// `train_step` uses the new loss.
    pub fn set_loss(&mut self, name: &str) -> Result<(), JsValue> {
        self.trainer.loss = match name {
            "mse" => Box::new(Mse),
            "huber" => Box::new(Huber { delta: 1.0 }),
            "cross_entropy" => Box::new(CrossEntropy),
            other => {
                return Err(JsValue::from_str(&format!(
                    "unknown loss '{other}'; try 'mse', 'huber', 'cross_entropy'"
                )))
            }
        };
        Ok(())
    }

    /// Hot-swap the optimiser. The new optimiser starts with zeroed
    /// state; param identity persists.
    pub fn set_optim(&mut self, name: &str, lr: f32) -> Result<(), JsValue> {
        self.trainer.optim = match name {
            "sgd" => Box::new(Sgd::new(lr)),
            "momentum" => Box::new(Sgd::with_momentum(lr, 0.9)),
            "adam" => Box::new(Adam::new(lr)),
            "adamw" => Box::new(AdamW::new(lr, 1e-4)),
            other => {
                return Err(JsValue::from_str(&format!(
                    "unknown optim '{other}'; try 'sgd', 'momentum', 'adam', 'adamw'"
                )))
            }
        };
        Ok(())
    }

    /// Add a hidden-layer activation hot-swap. For the simple widgets
    /// we just rewire the activation at position `idx` in the
    /// Sequential. `name` is one of "relu", "sigmoid", "tanh".
    pub fn set_activation(&mut self, idx: usize, name: &str) -> Result<(), JsValue> {
        let act: Box<dyn crate::module::Module> = match name {
            "relu" => Box::new(Relu),
            "sigmoid" => Box::new(Sigmoid),
            "tanh" => Box::new(Tanh),
            other => return Err(JsValue::from_str(&format!("unknown activation '{other}'"))),
        };
        // Downcast back to Sequential to swap a slot. Trainer.model
        // is `Box<dyn Module>`; we own it via &mut self.
        let seq = self
            .trainer
            .model
            .as_any_mut()
            .downcast_mut::<Sequential>()
            .ok_or_else(|| JsValue::from_str("set_activation: model is not a Sequential"))?;
        seq.replace(idx, act);
        Ok(())
    }

    /// One training step. `x` is `[batch * in_dim]` flat; `y` is
    /// `[batch * out_dim]` flat. Returns the scalar loss.
    pub fn train_step(&mut self, x: Vec<f32>, y: Vec<f32>) -> Result<f32, JsValue> {
        let batch = x.len() / self.in_dim;
        if batch * self.in_dim != x.len() {
            return Err(JsValue::from_str("x length not divisible by in_dim"));
        }
        if y.len() != batch * self.out_dim {
            return Err(JsValue::from_str("y length doesn't match batch * out_dim"));
        }
        let xt = Tensor::from_data(x, &[batch, self.in_dim]);
        let yt = Tensor::from_data(y, &[batch, self.out_dim]);
        Ok(self.trainer.train_step(&xt, &yt))
    }

    /// Forward-only inference. `x` flat `[batch * in_dim]`; returns
    /// flat `[batch * out_dim]`.
    pub fn forward(&mut self, x: Vec<f32>) -> Result<Vec<f32>, JsValue> {
        let batch = x.len() / self.in_dim;
        if batch * self.in_dim != x.len() {
            return Err(JsValue::from_str("x length not divisible by in_dim"));
        }
        let xt = Tensor::from_data(x, &[batch, self.in_dim]);
        let pred = self.trainer.predict(&xt);
        Ok(pred.data().to_vec())
    }

    /// Number of completed training steps. Useful for x-axis on
    /// learning-curve plots without the widget bookkeeping that.
    pub fn step_count(&self) -> u64 {
        self.trainer.ctx.step
    }

    /// Current learning rate.
    pub fn lr(&self) -> f32 {
        self.trainer.optim.lr()
    }

    /// Save weights as a self-describing byte buffer.
    pub fn save_weights(&self) -> Vec<u8> {
        crate::checkpoint::save_bytes(&*self.trainer.model)
    }

    /// Load weights from a buffer produced by `save_weights`.
    pub fn load_weights(&mut self, blob: Vec<u8>) -> Result<(), JsValue> {
        crate::checkpoint::load_bytes(&mut *self.trainer.model, &blob)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
