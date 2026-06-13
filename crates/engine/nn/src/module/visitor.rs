//! `ParamVisitor` — a closure that visits every learnable parameter
//! of a module. One walk, many consumers (optimiser step, checkpoint
//! save, param count, init re-do).

use crate::autograd::ParamId;
use crate::tensor::Tensor;

pub trait ParamVisitor {
    fn visit(&mut self, path: &str, param: &Tensor, id: ParamId);
}

pub trait ParamVisitorMut {
    fn visit(&mut self, path: &str, param: &mut Tensor, id: ParamId);
}

/// Blanket: any `FnMut(&str, &Tensor, ParamId)` is a `ParamVisitor`.
impl<F: FnMut(&str, &Tensor, ParamId) + Send> ParamVisitor for F {
    fn visit(&mut self, path: &str, p: &Tensor, id: ParamId) {
        (self)(path, p, id)
    }
}

impl<F: FnMut(&str, &mut Tensor, ParamId) + Send> ParamVisitorMut for F {
    fn visit(&mut self, path: &str, p: &mut Tensor, id: ParamId) {
        (self)(path, p, id)
    }
}

/// `"<prefix>.<path>"`, or just the prefix when `path` is empty (a
/// leaf visited at the prefix itself).
fn join_path(prefix: &str, path: &str) -> String {
    if path.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}.{path}")
    }
}

/// Wrap a visitor with a path prefix. Used by Sequential and tuple
/// Module impls so safetensors keys look like `"0.weight"`.
pub(crate) struct Prefixed<'a> {
    prefix: String,
    inner: &'a mut dyn ParamVisitor,
}
impl<'a> Prefixed<'a> {
    pub fn new(prefix: &str, inner: &'a mut dyn ParamVisitor) -> Self {
        Self {
            prefix: prefix.to_string(),
            inner,
        }
    }
}
impl<'a> ParamVisitor for Prefixed<'a> {
    fn visit(&mut self, path: &str, p: &Tensor, id: ParamId) {
        self.inner.visit(&join_path(&self.prefix, path), p, id);
    }
}

pub(crate) struct PrefixedMut<'a> {
    prefix: String,
    inner: &'a mut dyn ParamVisitorMut,
}
impl<'a> PrefixedMut<'a> {
    pub fn new(prefix: &str, inner: &'a mut dyn ParamVisitorMut) -> Self {
        Self {
            prefix: prefix.to_string(),
            inner,
        }
    }
}
impl<'a> ParamVisitorMut for PrefixedMut<'a> {
    fn visit(&mut self, path: &str, p: &mut Tensor, id: ParamId) {
        self.inner.visit(&join_path(&self.prefix, path), p, id);
    }
}
