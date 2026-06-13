//! `Linear` — dense layer. `y = x @ Wᵀ + b` for `x:[batch, in]`,
//! `W:[out, in]` (PyTorch convention; saves a shape juggle when
//! loading safetensors), `b:[out]`.
//!
//! Hot-swap: `w_id` and `b_id` are stable across optimiser
//! replacement so the new optimiser's state finds the same params.

use crate::autograd::{ParamId, Tape};
use crate::init::Init;
use crate::module::{Module, ParamVisitor, ParamVisitorMut};
use crate::rng::Rng;
use crate::tensor::Tensor;

pub struct Linear {
    pub(crate) weight: Tensor, // [out_dim, in_dim]
    pub(crate) bias: Tensor,   // [out_dim]
    pub(crate) w_id: ParamId,
    pub(crate) b_id: ParamId,
    /// Cached dims, used to validate input shape before the forward pass.
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, init: Init, rng: &mut dyn Rng) -> Self {
        let weight = init.tensor(&[out_dim, in_dim], rng);
        let bias = Tensor::zeros(&[out_dim]);
        Self {
            weight,
            bias,
            w_id: ParamId::new(),
            b_id: ParamId::new(),
            in_dim,
            out_dim,
        }
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
}

impl Module for Linear {
    fn forward(&self, tape: &mut Tape, x: &Tensor) -> Tensor {
        assert_eq!(
            x.rank(),
            2,
            "Linear: input must be 2D [batch, in_dim], got rank {}",
            x.rank()
        );
        assert_eq!(
            x.shape()[1],
            self.in_dim,
            "Linear: input feature dim {} != layer in_dim {}",
            x.shape()[1],
            self.in_dim
        );

        // Register params on this tape (no-ops if already there).
        let w = tape.leaf(&self.weight, Some(self.w_id));
        let b = tape.leaf(&self.bias, Some(self.b_id));

        // y = x @ Wᵀ + bias_broadcast.
        let wt = tape.transpose(&w);
        let prod = tape.matmul(x, &wt);
        let bb = tape.broadcast_row(&b, prod.shape());
        tape.add(&prod, &bb)
    }

    fn visit_params(&self, v: &mut dyn ParamVisitor) {
        v.visit("weight", &self.weight, self.w_id);
        v.visit("bias", &self.bias, self.b_id);
    }

    fn visit_params_mut(&mut self, v: &mut dyn ParamVisitorMut) {
        v.visit("weight", &mut self.weight, self.w_id);
        v.visit("bias", &mut self.bias, self.b_id);
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
