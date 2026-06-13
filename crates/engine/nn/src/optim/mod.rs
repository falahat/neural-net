//! Optimisers. Each is `dyn Optimizer` so the trainer can hot-swap.
//!
//! The state is keyed on `ParamId` (not on storage address) so that
//! parameters survive optimiser swaps: replace `Adam` with `Sgd` and
//! the model's params keep their identity. Shape follows
//! PyTorch / Candle / Optax.

use std::collections::HashMap;

use crate::autograd::{GradStore, ParamId};
use crate::module::Module;
use crate::tensor::Tensor;

pub trait Optimizer: Send {
    fn step(&mut self, module: &mut dyn Module, grads: &GradStore);
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}

// ─── SGD with momentum ─────────────────────────────────────────────────
//
// **Citation.** B. T. Polyak, "Some methods of speeding up the
// convergence of iteration methods," USSR Computational Mathematics
// and Mathematical Physics 4(5), 1964. The momentum form below
// (`v_t = μ v_{t−1} + g_t`; `θ ← θ − η v_t`) is the classical one.

pub struct Sgd {
    pub lr: f32,
    pub momentum: f32,
    state: HashMap<ParamId, Vec<f32>>,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            state: HashMap::new(),
        }
    }
    pub fn with_momentum(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            state: HashMap::new(),
        }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, module: &mut dyn Module, grads: &GradStore) {
        let lr = self.lr;
        let momentum = self.momentum;
        let state = &mut self.state;

        module.visit_params_mut(&mut |_path: &str, p: &mut Tensor, id: ParamId| {
            if let Some((g, _)) = grads.get(id) {
                let n = p.numel();
                let data = p.data_mut();
                if momentum == 0.0 {
                    for k in 0..n {
                        data[k] -= lr * g[k];
                    }
                } else {
                    let v = state.entry(id).or_insert_with(|| vec![0.0; n]);
                    for k in 0..n {
                        v[k] = momentum * v[k] + g[k];
                        data[k] -= lr * v[k];
                    }
                }
            }
        });
    }

    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ─── Adam ──────────────────────────────────────────────────────────────
//
// **Citation.** D. P. Kingma & J. Ba, "Adam: A Method for Stochastic
// Optimization," ICLR 2015. The recurrence and bias correction below
// are Algorithm 1 from the paper.

struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
}

/// One Adam step over a parameter buffer (Kingma & Ba, Algorithm 1),
/// shared by `Adam` (`weight_decay = 0.0`) and `AdamW`. A non-zero
/// `weight_decay` applies the decoupled decay of Loshchilov & Hutter:
/// shrink the param *before* the Adam update.
#[allow(clippy::too_many_arguments)]
fn adam_update(
    data: &mut [f32],
    g: &[f32],
    s: &mut AdamState,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    weight_decay: f32,
) {
    for k in 0..data.len() {
        s.m[k] = b1 * s.m[k] + (1.0 - b1) * g[k];
        s.v[k] = b2 * s.v[k] + (1.0 - b2) * g[k] * g[k];
        let m_hat = s.m[k] / bc1;
        let v_hat = s.v[k] / bc2;
        if weight_decay != 0.0 {
            data[k] -= lr * weight_decay * data[k];
        }
        data[k] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    state: HashMap<ParamId, AdamState>,
    step_count: u64,
}

impl Adam {
    /// Default β values (0.9, 0.999) and ε = 1e-8 from the Adam paper.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            state: HashMap::new(),
            step_count: 0,
        }
    }
    pub fn with_betas(lr: f32, beta1: f32, beta2: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps: 1e-8,
            state: HashMap::new(),
            step_count: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, module: &mut dyn Module, grads: &GradStore) {
        self.step_count += 1;
        let t = self.step_count.min(i32::MAX as u64) as i32;
        let (lr, b1, b2, eps) = (self.lr, self.beta1, self.beta2, self.eps);
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        let state = &mut self.state;

        module.visit_params_mut(&mut |_path: &str, p: &mut Tensor, id: ParamId| {
            if let Some((g, _)) = grads.get(id) {
                let n = p.numel();
                let s = state.entry(id).or_insert_with(|| AdamState {
                    m: vec![0.0; n],
                    v: vec![0.0; n],
                });
                adam_update(p.data_mut(), g, s, lr, b1, b2, eps, bc1, bc2, 0.0);
            }
        });
    }

    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ─── AdamW (decoupled weight decay) ────────────────────────────────────
//
// **Citation.** I. Loshchilov & F. Hutter, "Decoupled Weight Decay
// Regularization," ICLR 2019. Same as Adam but applies weight decay
// directly on the parameter rather than as a gradient term — fixes a
// long-standing subtle bug in L2 regularisation + Adam.

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    state: HashMap<ParamId, AdamState>,
    step_count: u64,
}

impl AdamW {
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            state: HashMap::new(),
            step_count: 0,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, module: &mut dyn Module, grads: &GradStore) {
        self.step_count += 1;
        let t = self.step_count.min(i32::MAX as u64) as i32;
        let (lr, b1, b2, eps, wd) = (self.lr, self.beta1, self.beta2, self.eps, self.weight_decay);
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        let state = &mut self.state;

        module.visit_params_mut(&mut |_path: &str, p: &mut Tensor, id: ParamId| {
            if let Some((g, _)) = grads.get(id) {
                let n = p.numel();
                let s = state.entry(id).or_insert_with(|| AdamState {
                    m: vec![0.0; n],
                    v: vec![0.0; n],
                });
                adam_update(p.data_mut(), g, s, lr, b1, b2, eps, bc1, bc2, wd);
            }
        });
    }

    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}
