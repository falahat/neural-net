//! `Module` — the trait every learnable component implements.
//!
//! The `Module: forward + visit_params` shape follows burn's `Module` +
//! `ModuleVisitor` pattern (simplified by dropping the `<B: Backend>`
//! generic); tuple composition (`(A, B, C): Module`) follows dfdx.

pub mod linear;
pub mod sequential;
pub mod visitor;

pub use linear::Linear;
pub use sequential::Sequential;
pub use visitor::{ParamVisitor, ParamVisitorMut};

use crate::autograd::Tape;
use crate::tensor::Tensor;

pub trait Module: Send {
    /// Forward pass. Records ops on `tape`; returns the tape-bound
    /// output tensor.
    fn forward(&self, tape: &mut Tape, x: &Tensor) -> Tensor;

    /// Depth-first walk of learnable parameters in declaration order.
    /// Allocation-free; the visitor carries a path string for
    /// save/load.
    fn visit_params(&self, v: &mut dyn ParamVisitor) {
        let _ = v;
    }

    /// Same, mutable, for parameter updates / device transfer / load.
    fn visit_params_mut(&mut self, v: &mut dyn ParamVisitorMut) {
        let _ = v;
    }

    /// Checked-downcast hook so holders of a `Box<dyn Module>` can
    /// recover the concrete type (e.g. the WASM bridge re-wiring a
    /// `Sequential` slot) without unsafe pointer casts.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

// ─── Tuple composition (idea from dfdx) ─────────────────────────────
//
// `(A, B)` is a Module whose forward is `B(A(x))`. Param paths get a
// numeric prefix so safetensors keys look like "0.weight", "1.weight".

impl<A: Module + 'static, B: Module + 'static> Module for (A, B) {
    fn forward(&self, tape: &mut Tape, x: &Tensor) -> Tensor {
        let y = self.0.forward(tape, x);
        self.1.forward(tape, &y)
    }
    fn visit_params(&self, v: &mut dyn ParamVisitor) {
        let mut p0 = visitor::Prefixed::new("0", v);
        self.0.visit_params(&mut p0);
        let mut p1 = visitor::Prefixed::new("1", v);
        self.1.visit_params(&mut p1);
    }
    fn visit_params_mut(&mut self, v: &mut dyn ParamVisitorMut) {
        let mut p0 = visitor::PrefixedMut::new("0", v);
        self.0.visit_params_mut(&mut p0);
        let mut p1 = visitor::PrefixedMut::new("1", v);
        self.1.visit_params_mut(&mut p1);
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl<A: Module + 'static, B: Module + 'static, C: Module + 'static> Module for (A, B, C) {
    fn forward(&self, tape: &mut Tape, x: &Tensor) -> Tensor {
        let y = self.0.forward(tape, x);
        let y = self.1.forward(tape, &y);
        self.2.forward(tape, &y)
    }
    fn visit_params(&self, v: &mut dyn ParamVisitor) {
        for (i, m) in [&self.0 as &dyn Module, &self.1, &self.2]
            .iter()
            .enumerate()
        {
            let mut p = visitor::Prefixed::new(&i.to_string(), v);
            m.visit_params(&mut p);
        }
    }
    fn visit_params_mut(&mut self, v: &mut dyn ParamVisitorMut) {
        let mut p0 = visitor::PrefixedMut::new("0", v);
        self.0.visit_params_mut(&mut p0);
        let mut p1 = visitor::PrefixedMut::new("1", v);
        self.1.visit_params_mut(&mut p1);
        let mut p2 = visitor::PrefixedMut::new("2", v);
        self.2.visit_params_mut(&mut p2);
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Convenience: count total learnable scalar parameters.
pub fn num_params<M: Module + ?Sized>(m: &M) -> usize {
    struct Counter(usize);
    impl ParamVisitor for Counter {
        fn visit(&mut self, _path: &str, p: &Tensor, _id: crate::autograd::ParamId) {
            self.0 += p.numel();
        }
    }
    let mut c = Counter(0);
    m.visit_params(&mut c);
    c.0
}
