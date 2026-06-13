//! Loss functions — each is a `dyn Loss` so it can be hot-swapped on
//! a `Trainer` between training steps.

use crate::autograd::Tape;
use crate::tensor::Tensor;

pub trait Loss: Send {
    fn forward(&self, tape: &mut Tape, pred: &Tensor, target: &Tensor) -> Tensor;
}

/// Blanket: a free closure `Fn(&mut Tape, &Tensor, &Tensor) -> Tensor`
/// is automatically a `Loss`. Convenient for ad-hoc widget losses.
impl<F> Loss for F
where
    F: Fn(&mut Tape, &Tensor, &Tensor) -> Tensor + Send + Sync,
{
    fn forward(&self, t: &mut Tape, p: &Tensor, y: &Tensor) -> Tensor {
        (self)(t, p, y)
    }
}

// ─── Mse ───────────────────────────────────────────────────────────────

/// `MSE = mean((pred − target)²)`. Suitable for regression.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mse;
impl Loss for Mse {
    fn forward(&self, tape: &mut Tape, pred: &Tensor, target: &Tensor) -> Tensor {
        let d = tape.sub(pred, target);
        let s = tape.square(&d);
        tape.mean_all(&s)
    }
}

// ─── CrossEntropy (fused with softmax) ─────────────────────────────────

/// Fused softmax + cross-entropy. Numerically stable; see
/// `math::stable::softmax_cross_entropy`.
///
/// `pred` is logits of shape `[batch, classes]`; `target` is a
/// `[batch]` tensor of class indices stored as `f32` (we don't
/// have a separate integer tensor type in v1 — indices fit
/// losslessly in f32 up to 16,777,216 classes).
#[derive(Debug, Clone, Copy, Default)]
pub struct CrossEntropy;
impl Loss for CrossEntropy {
    fn forward(&self, tape: &mut Tape, pred: &Tensor, target: &Tensor) -> Tensor {
        tape.softmax_xe(pred, target)
    }
}

// ─── Huber ─────────────────────────────────────────────────────────────

/// **Huber loss** (P. J. Huber, "Robust estimation of a location
/// parameter," Annals of Mathematical Statistics 35(1), 1964).
/// Quadratic for small residuals (`|r| ≤ δ`), linear for large —
/// the robust regression workhorse.
///
/// `L(r) = ½ r²              if |r| ≤ δ`
/// `L(r) = δ (|r| − ½ δ)    otherwise`
///
/// Built from primitive ops, so backward goes through the autograd
/// graph rather than a hand-written kernel. Loss is the mean over all
/// examples.
#[derive(Debug, Clone, Copy)]
pub struct Huber {
    pub delta: f32,
}

impl Default for Huber {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl Loss for Huber {
    fn forward(&self, tape: &mut Tape, pred: &Tensor, target: &Tensor) -> Tensor {
        // The two pieces of the piecewise loss are equivalently
        //     L = min(½ r², δ |r| − ½ δ²),
        // computed on tape and min'd via min(a,b) = a − relu(a − b).
        let d = tape.sub(pred, target);
        let r2 = tape.square(&d);
        let half_r2 = tape.scale(&r2, 0.5);

        // |r| via sqrt(r² + ε). ε tiny; only matters at r=0 where the
        // gradient of |r| isn't defined anyway.
        let eps = Tensor::from_data(vec![1e-12; r2.numel()], r2.shape());
        let r2e = tape.add(&r2, &eps);
        // sqrt isn't a primitive; use r² ** 0.5 via log/exp.
        let l = tape.log(&r2e);
        let l2 = tape.scale(&l, 0.5);
        let abs_r = tape.exp(&l2);

        // linear branch: δ |r| − ½ δ²
        let delta_abs = tape.scale(&abs_r, self.delta);
        let const_off =
            Tensor::from_data(vec![0.5 * self.delta * self.delta; r2.numel()], r2.shape());
        let lin = tape.sub(&delta_abs, &const_off);

        // Choose pointwise min(half_r2, lin) via: min(a,b) = a − relu(a − b).
        let diff = tape.sub(&half_r2, &lin);
        let excess = tape.relu(&diff);
        let mn = tape.sub(&half_r2, &excess);
        tape.mean_all(&mn)
    }
}
