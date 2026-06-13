//! The `Op` enum + its `backward` dispatch.
//!
//! **Why an enum instead of `Box<dyn Backward>`:**
//!   1. Pedagogy. A reader can `grep` this file and see every op's
//!      forward and backward side by side. The pattern-match *is*
//!      the teaching artefact.
//!   2. No allocator pressure on the hot path.
//!   3. Future kernel fusion can walk the enum; closures are opaque.
//!
//! Each variant captures everything backward needs that *isn't* the
//! op's output (which the tape already caches on each node).
//! E.g. `Sigmoid` doesn't store its inputs — the backward formula
//! `dx = dy * y * (1 - y)` uses the *output* `y`, which the tape has.

use crate::backend::cpu;
use crate::tensor::{shape, NodeId};

use super::TapeNode;

/// One variant per recordable op. Backward formula in the doc on
/// each variant: `y` = this node's output, `dy` = ∂L/∂y, `dx_i` =
/// the contribution this op makes to its i-th input's gradient.
#[derive(Debug)]
pub(crate) enum Op {
    /// Leaf — input or param. No backward; if `param_id.is_some()`
    /// the accumulated grad gets exported to `GradStore`.
    Leaf,

    // ── elementwise binary (same shape) ─────────────────────────────
    /// `da = dy`, `db = dy`.
    Add { lhs: NodeId, rhs: NodeId },
    /// `da = dy`, `db = -dy`.
    Sub { lhs: NodeId, rhs: NodeId },
    /// `da = dy * b`, `db = dy * a`.
    Mul { lhs: NodeId, rhs: NodeId },
    /// `da = dy / b`, `db = -dy * a / b²`.
    Div { lhs: NodeId, rhs: NodeId },

    // ── elementwise unary ───────────────────────────────────────────
    /// `dx = -dy`.
    Neg { input: NodeId },
    /// `dx = dy * factor`.
    Scale { input: NodeId, factor: f32 },
    /// `dx = dy * 1[x > 0]` (subgradient = 0 at x = 0).
    Relu { input: NodeId },
    /// `dx = dy * y * (1 - y)` — reads cached OUTPUT (avoids re-exp).
    Sigmoid { input: NodeId },
    /// `dx = dy * (1 - y²)` — reads cached OUTPUT.
    Tanh { input: NodeId },
    /// `dx = dy * y` — reads cached OUTPUT.
    Exp { input: NodeId },
    /// `dx = dy / x`.
    Log { input: NodeId },
    /// `dx = 2 * x * dy`. Own op (not `Mul(x, x)`) so backward
    /// doesn't try to credit the same input twice.
    Square { input: NodeId },

    // ── linear algebra ──────────────────────────────────────────────
    /// `A:[m,k]` · `B:[k,n]` → `Y:[m,n]`. `dA = dY · Bᵀ`, `dB = Aᵀ · dY`.
    Matmul { lhs: NodeId, rhs: NodeId },
    /// `dX = dYᵀ`.
    Transpose { input: NodeId },

    // ── reductions ──────────────────────────────────────────────────
    /// `y = Σ xᵢ`; `dxᵢ = dy` for all i.
    SumAll { input: NodeId },
    /// `y = Σ xᵢ / n`; `dxᵢ = dy / n`.
    MeanAll { input: NodeId },

    // ── shape ops ───────────────────────────────────────────────────
    /// Pure view; backward reshapes the grad back to `from`.
    Reshape { input: NodeId },
    /// `[cols] → [rows, cols]`, each row copies the input. Backward
    /// is column-sum across rows. Only broadcast we have — enough
    /// for `Linear` bias.
    BroadcastRow { input: NodeId, to: Vec<usize> },

    // ── fused for stability ─────────────────────────────────────────
    /// Fused softmax + cross-entropy. Forward: `loss = mean_b(LSE(logits[b]) - logits[b][t_b])`.
    /// Backward: `∂loss/∂logits = (softmax - one_hot(targets)) / batch`.
    /// `softmax` is cached at forward so backward doesn't recompute.
    SoftmaxXE {
        logits: NodeId,
        targets: NodeId,
        softmax: Vec<f32>,
    },
}

/// One step of the backward sweep — read `grads[i]` (the dy for
/// this node) and accumulate `dx_*` into each input's grad slot.
pub(crate) fn backward(tape: &super::Tape, i: usize, grads: &mut [Vec<f32>]) {
    // Snapshot the upstream gradient so the borrow checker lets us
    // simultaneously index into other entries of `grads` below.
    let dy: Vec<f32> = grads[i].clone();
    let node: &TapeNode = &tape.nodes[i];

    match &node.op {
        Op::Leaf => { /* nothing to do; grad sits in slot i */ }

        Op::Add { lhs, rhs } => {
            // d(a + b)/da = 1, d(a + b)/db = 1.
            accumulate(&mut grads[lhs.0 as usize], &dy);
            accumulate(&mut grads[rhs.0 as usize], &dy);
        }

        Op::Sub { lhs, rhs } => {
            // d(a - b)/da = 1, d(a - b)/db = -1.
            accumulate(&mut grads[lhs.0 as usize], &dy);
            accumulate_neg(&mut grads[rhs.0 as usize], &dy);
        }

        Op::Mul { lhs, rhs } => {
            // d(a * b)/da = b, d(a * b)/db = a.
            let a = tape.value(*lhs).to_vec();
            let b = tape.value(*rhs).to_vec();
            accumulate_scaled(&mut grads[lhs.0 as usize], &dy, |k| b[k]);
            accumulate_scaled(&mut grads[rhs.0 as usize], &dy, |k| a[k]);
        }

        Op::Div { lhs, rhs } => {
            // d(a / b)/da = 1/b ; d(a / b)/db = -a / b^2.
            let a = tape.value(*lhs).to_vec();
            let b = tape.value(*rhs).to_vec();
            accumulate_scaled(&mut grads[lhs.0 as usize], &dy, |k| 1.0 / b[k]);
            accumulate_scaled(&mut grads[rhs.0 as usize], &dy, |k| -a[k] / (b[k] * b[k]));
        }

        Op::Neg { input } => {
            accumulate_neg(&mut grads[input.0 as usize], &dy);
        }

        Op::Scale { input, factor } => {
            accumulate_scaled(&mut grads[input.0 as usize], &dy, |_| *factor);
        }

        Op::Relu { input } => {
            // dx = dy * 1[x > 0]. We use the input value, not the output,
            // because ReLU is non-differentiable at 0 and convention is
            // that the subgradient there is 0.
            let x = tape.value(*input).to_vec();
            accumulate_scaled(&mut grads[input.0 as usize], &dy, |k| {
                if x[k] > 0.0 {
                    1.0
                } else {
                    0.0
                }
            });
        }

        Op::Sigmoid { input } => {
            // dx = dy * y * (1 - y) where y is THIS node's output.
            let y = &node.value;
            accumulate_scaled(&mut grads[input.0 as usize], &dy, |k| y[k] * (1.0 - y[k]));
        }

        Op::Tanh { input } => {
            // dx = dy * (1 - y^2).
            let y = &node.value;
            accumulate_scaled(&mut grads[input.0 as usize], &dy, |k| 1.0 - y[k] * y[k]);
        }

        Op::Exp { input } => {
            // dx = dy * y.
            let y = &node.value;
            accumulate_scaled(&mut grads[input.0 as usize], &dy, |k| y[k]);
        }

        Op::Log { input } => {
            // dx = dy / x.
            let x = tape.value(*input).to_vec();
            accumulate_scaled(&mut grads[input.0 as usize], &dy, |k| 1.0 / x[k]);
        }

        Op::Square { input } => {
            // dx = dy * 2x.
            let x = tape.value(*input).to_vec();
            accumulate_scaled(&mut grads[input.0 as usize], &dy, |k| 2.0 * x[k]);
        }

        Op::Matmul { lhs, rhs } => {
            matmul_backward(tape, *lhs, *rhs, &dy, &node.shape, grads);
        }

        Op::Transpose { input } => {
            // d/dA (A^T) = transpose the grad back.
            let (back, _) = cpu::transpose_2d(&dy, &node.shape);
            accumulate(&mut grads[input.0 as usize], &back);
        }

        Op::SumAll { input } => {
            // y = sum(x) ⇒ dx = dy * 1, but dy is scalar so we
            // broadcast that one number across the input shape.
            let n = grads[input.0 as usize].len();
            let s = dy[0];
            for k in 0..n {
                grads[input.0 as usize][k] += s;
            }
        }

        Op::MeanAll { input } => {
            let n = grads[input.0 as usize].len();
            let s = dy[0] / n as f32;
            for k in 0..n {
                grads[input.0 as usize][k] += s;
            }
        }

        Op::Reshape { input } => {
            // Reshape is a no-op on data; grad is the same flat values
            // with the input's shape.
            accumulate(&mut grads[input.0 as usize], &dy);
        }

        Op::BroadcastRow { input, to } => {
            // dx = sum-rows of dy (since each row of y replicated x).
            let in_shape = tape.shape(*input).to_vec();
            let back = shape::unbroadcast_to(&dy, to, &in_shape);
            accumulate(&mut grads[input.0 as usize], &back);
        }

        Op::SoftmaxXE {
            logits,
            targets,
            softmax,
        } => {
            softmax_xe_backward(tape, *logits, *targets, softmax, &dy, grads);
        }
    }
}

/// Backward for `y = a @ b` with `a:[m,k]`, `b:[k,n]`, `dy:[m,n]`:
///   `da = dy @ bᵀ  :[m, k]`
///   `db = aᵀ @ dy  :[k, n]`
fn matmul_backward(
    tape: &super::Tape,
    lhs: NodeId,
    rhs: NodeId,
    dy: &[f32],
    dy_shape: &[usize],
    grads: &mut [Vec<f32>],
) {
    let a_shape = tape.shape(lhs).to_vec();
    let b_shape = tape.shape(rhs).to_vec();
    let a = tape.value(lhs).to_vec();
    let b = tape.value(rhs).to_vec();

    let (b_t, b_t_shape) = cpu::transpose_2d(&b, &b_shape);
    let (da, _) = cpu::matmul(dy, dy_shape, &b_t, &b_t_shape);
    accumulate(&mut grads[lhs.0 as usize], &da);

    let (a_t, a_t_shape) = cpu::transpose_2d(&a, &a_shape);
    let (db, _) = cpu::matmul(&a_t, &a_t_shape, dy, dy_shape);
    accumulate(&mut grads[rhs.0 as usize], &db);
}

/// Backward for the fused softmax + cross-entropy.
/// Forward: `loss = mean over batch of (LSE(logits) - logits[target])`.
/// Gradient w.r.t. logits: `(softmax - one_hot) / batch`.
/// `targets` is treated as a leaf (its grad is unused).
fn softmax_xe_backward(
    tape: &super::Tape,
    logits: NodeId,
    targets: NodeId,
    softmax: &[f32],
    dy: &[f32],
    grads: &mut [Vec<f32>],
) {
    let logits_shape = tape.shape(logits).to_vec();
    assert_eq!(logits_shape.len(), 2, "SoftmaxXE: logits must be 2D");
    let (batch, classes) = (logits_shape[0], logits_shape[1]);
    let tvals = tape.value(targets).to_vec();
    assert_eq!(
        tvals.len(),
        batch,
        "SoftmaxXE: targets must have len = batch"
    );

    let s = dy[0] / batch as f32;
    let g = &mut grads[logits.0 as usize];
    for b in 0..batch {
        let t = tvals[b] as usize;
        for c in 0..classes {
            let mut val = softmax[b * classes + c];
            if c == t {
                val -= 1.0;
            }
            g[b * classes + c] += s * val;
        }
    }
}

#[inline]
fn accumulate(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

#[inline]
fn accumulate_neg(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d -= s;
    }
}

/// Indexed scaled accumulate: `dst[k] += dy[k] * per_elem(k)` — the
/// shared loop shape of every elementwise backward arm.
#[inline]
fn accumulate_scaled(dst: &mut [f32], dy: &[f32], per_elem: impl Fn(usize) -> f32) {
    debug_assert_eq!(dst.len(), dy.len());
    for (k, (d, dyk)) in dst.iter_mut().zip(dy).enumerate() {
        *d += dyk * per_elem(k);
    }
}
