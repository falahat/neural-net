//! `Sequential` — `Vec<Box<dyn Module>>` chain. Use when the
//! architecture is data (width/depth from config) or when you want
//! to swap a layer at runtime via `replace(idx, new)`. For
//! statically-known shapes prefer tuples — `(Linear, Relu, Linear)`
//! is also a `Module` and avoids the dyn dispatch.
//!
//! Param paths are `"<idx>.weight"` / `"<idx>.bias"` where `<idx>`
//! is the layer's position in the Vec, NOT a renumbering — so
//! activation slots leave gaps. A `[Linear, Relu, Linear]` yields
//! `"0.weight"`, `"0.bias"`, `"2.weight"`, `"2.bias"`.

use crate::autograd::Tape;
use crate::module::{visitor, Module, ParamVisitor, ParamVisitorMut};
use crate::tensor::Tensor;

pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Replace the layer at `idx`, the runtime hot-swap point. Returns
    /// the old layer.
    pub fn replace(&mut self, idx: usize, m: Box<dyn Module>) -> Box<dyn Module> {
        std::mem::replace(&mut self.layers[idx], m)
    }
}

impl Module for Sequential {
    fn forward(&self, tape: &mut Tape, x: &Tensor) -> Tensor {
        let mut y = self.layers[0].forward(tape, x);
        for layer in &self.layers[1..] {
            y = layer.forward(tape, &y);
        }
        y
    }

    fn visit_params(&self, v: &mut dyn ParamVisitor) {
        for (i, layer) in self.layers.iter().enumerate() {
            let mut p = visitor::Prefixed::new(&i.to_string(), v);
            layer.visit_params(&mut p);
        }
    }

    fn visit_params_mut(&mut self, v: &mut dyn ParamVisitorMut) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let mut p = visitor::PrefixedMut::new(&i.to_string(), v);
            layer.visit_params_mut(&mut p);
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
