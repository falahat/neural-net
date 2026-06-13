//! Reverse-mode autodiff via a Wengert tape (R. E. Wengert, "A simple
//! automatic derivative evaluation program," CACM 7(8), 1964). The tape
//! is a `Vec` of recorded ops indexed by integer.
//!
//! ## How forward works
//!
//! Every op that produces a tensor calls `tape.record_*(...)`. The
//! tape stores:
//!   - the op variant (so backward knows what derivative to apply),
//!   - any cached intermediate values (e.g. the sigmoid output),
//!   - the output tensor itself (so backward can read its value),
//!   - the NodeIds of the op's inputs (for grad propagation).
//!
//! The returned `Tensor` carries a `Some(node_id)` so subsequent ops
//! that consume it can record the dependency.
//!
//! ## How backward works
//!
//! `Tape::backward(loss)` is one reverse iteration over the recorded
//! nodes. For each node, the accumulated output-gradient is dispatched
//! through `op::backward` (see `op.rs`), which scatters partial
//! gradients into the input nodes' grad accumulators. Multiple
//! consumers of the same input correctly sum because of accumulation.
//!
//! Determinism: ops are visited in fixed reverse order, summations
//! are left-to-right, no parallelism. Same seed + same data ⇒
//! bit-exact gradients.

pub mod op;

use std::collections::HashMap;

use crate::tensor::{NodeId, Tensor};

pub(crate) use op::Op;

/// Stable identity for a learnable parameter. Survives optimiser
/// swaps so `state[&param_id]` keeps working across `trainer.optim = …`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(pub u64);

impl ParamId {
    /// Allocate a fresh id from the global monotonic counter.
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for ParamId {
    fn default() -> Self {
        Self::new()
    }
}

/// One recorded operation on the tape, plus its forward output.
pub(crate) struct TapeNode {
    pub op: Op,
    /// Forward result of this op, used for backward formulas that
    /// need it (e.g. sigmoid backward uses `sigmoid * (1 - sigmoid)`,
    /// where `sigmoid` is *this node's* value, cached by reference).
    pub value: Vec<f32>,
    pub shape: Vec<usize>,
    /// Set when this node is a learnable param. After backward, the
    /// node's accumulated grad is exported into the `GradStore` under
    /// this id.
    pub param_id: Option<ParamId>,
}

/// The tape itself. Construct fresh per forward/backward pair —
/// reusing requires `Tape::clear()` between training steps.
pub struct Tape {
    pub(crate) nodes: Vec<TapeNode>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(64),
        }
    }

    /// Drop all nodes. Call between training steps so the tape's
    /// memory stays bounded.
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Register a leaf tensor (input or learnable param) on the tape.
    /// The returned `Tensor` carries this tape's node id so ops can
    /// reference it. The `param_id` is `Some` for params (their grad
    /// will be exported) and `None` for transient inputs.
    pub fn leaf(&mut self, t: &Tensor, param_id: Option<ParamId>) -> Tensor {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(TapeNode {
            op: Op::Leaf,
            value: t.data().to_vec(),
            shape: t.shape().to_vec(),
            param_id,
        });
        Tensor::with_node(t.data().to_vec(), t.shape().to_vec(), id)
    }

    /// Lower-level record: caller has already computed the forward
    /// `value`/`shape` and provides the `Op` describing how it was
    /// produced. Returns the new tensor handle.
    pub(crate) fn record(&mut self, op: Op, value: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(TapeNode {
            op,
            value: value.clone(),
            shape: shape.clone(),
            param_id: None,
        });
        Tensor::with_node(value, shape, id)
    }

    /// Run the backward pass starting from `loss` (a scalar tensor on
    /// this tape). Returns a `GradStore` keyed by `ParamId` so the
    /// optimiser can step.
    pub fn backward(&mut self, loss: &Tensor) -> GradStore {
        let loss_node = loss
            .node()
            .expect("backward: loss tensor has no tape node — was it created via tape ops?");
        assert!(
            self.nodes[loss_node.0 as usize].value.len() == 1,
            "backward: loss must be a scalar (numel == 1); got shape {:?}",
            self.nodes[loss_node.0 as usize].shape
        );

        // Accumulators: one grad buffer per node. Initialised to zero;
        // the loss node seeds with 1.0.
        let n = self.nodes.len();
        let mut grads: Vec<Vec<f32>> = self
            .nodes
            .iter()
            .map(|n| vec![0.0; n.value.len()])
            .collect();
        grads[loss_node.0 as usize][0] = 1.0;

        // Visit nodes in reverse tape order. Each op's `backward`
        // reads its own node's accumulated grad and writes into its
        // inputs' grad slots.
        for i in (0..n).rev() {
            op::backward(self, i, &mut grads);
        }

        // Export grads for nodes that are flagged as params.
        let mut store = GradStore::default();
        for (i, node) in self.nodes.iter().enumerate() {
            if let Some(id) = node.param_id {
                store.insert(id, grads[i].clone(), node.shape.clone());
            }
        }
        store
    }

    pub(crate) fn value(&self, id: NodeId) -> &[f32] {
        &self.nodes[id.0 as usize].value
    }
    pub(crate) fn shape(&self, id: NodeId) -> &[usize] {
        &self.nodes[id.0 as usize].shape
    }

    /// Internal helper — resolve a tensor's node id, or auto-register
    /// it as a transient (non-param) leaf if it isn't yet on the tape.
    fn ensure_node(&mut self, t: &Tensor) -> NodeId {
        if let Some(id) = t.node() {
            return id;
        }
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(TapeNode {
            op: Op::Leaf,
            value: t.data().to_vec(),
            shape: t.shape().to_vec(),
            param_id: None,
        });
        id
    }

    /// Record an elementwise unary op: forward maps `f` over `x`, the
    /// `Op` variant is built from the input's node id.
    fn record_unary(
        &mut self,
        x: &Tensor,
        op: impl FnOnce(NodeId) -> Op,
        f: impl Fn(f32) -> f32,
    ) -> Tensor {
        let input = self.ensure_node(x);
        let out = crate::backend::cpu::map_unary(x.data(), f);
        let shape = x.shape().to_vec();
        self.record(op(input), out, shape)
    }

    /// Record an elementwise binary op (same-shape operands): forward
    /// runs the backend kernel `f`, the `Op` variant is built from the
    /// operands' node ids.
    fn record_binary(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        op: impl FnOnce(NodeId, NodeId) -> Op,
        f: fn(&[f32], &[f32]) -> Vec<f32>,
    ) -> Tensor {
        let lhs = self.ensure_node(a);
        let rhs = self.ensure_node(b);
        let out = f(a.data(), b.data());
        let shape = a.shape().to_vec();
        self.record(op(lhs, rhs), out, shape)
    }

    // ──────────────────────────────────────────────────────────────────
    // Forward ops. Each `Tape::xxx` runs the CPU forward, records the
    // op, and returns a tape-bound tensor. The user-visible API to
    // build a forward graph.
    // ──────────────────────────────────────────────────────────────────

    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        self.record_binary(
            a,
            b,
            |lhs, rhs| Op::Add { lhs, rhs },
            crate::backend::cpu::add,
        )
    }

    pub fn sub(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        self.record_binary(
            a,
            b,
            |lhs, rhs| Op::Sub { lhs, rhs },
            crate::backend::cpu::sub,
        )
    }

    pub fn mul(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        self.record_binary(
            a,
            b,
            |lhs, rhs| Op::Mul { lhs, rhs },
            crate::backend::cpu::mul,
        )
    }

    pub fn div(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        self.record_binary(
            a,
            b,
            |lhs, rhs| Op::Div { lhs, rhs },
            crate::backend::cpu::div,
        )
    }

    pub fn neg(&mut self, x: &Tensor) -> Tensor {
        self.record_unary(x, |input| Op::Neg { input }, |v| -v)
    }

    pub fn scale(&mut self, x: &Tensor, factor: f32) -> Tensor {
        self.record_unary(x, |input| Op::Scale { input, factor }, |v| v * factor)
    }

    pub fn relu(&mut self, x: &Tensor) -> Tensor {
        self.record_unary(x, |input| Op::Relu { input }, |v| v.max(0.0))
    }

    pub fn sigmoid(&mut self, x: &Tensor) -> Tensor {
        self.record_unary(
            x,
            |input| Op::Sigmoid { input },
            crate::math::stable::sigmoid,
        )
    }

    pub fn tanh(&mut self, x: &Tensor) -> Tensor {
        self.record_unary(x, |input| Op::Tanh { input }, |v| v.tanh())
    }

    pub fn exp(&mut self, x: &Tensor) -> Tensor {
        self.record_unary(x, |input| Op::Exp { input }, f32::exp)
    }

    pub fn log(&mut self, x: &Tensor) -> Tensor {
        self.record_unary(x, |input| Op::Log { input }, f32::ln)
    }

    pub fn square(&mut self, x: &Tensor) -> Tensor {
        self.record_unary(x, |input| Op::Square { input }, |v| v * v)
    }

    pub fn matmul(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let lhs = self.ensure_node(a);
        let rhs = self.ensure_node(b);
        let (out, shape) = crate::backend::cpu::matmul(a.data(), a.shape(), b.data(), b.shape());
        self.record(Op::Matmul { lhs, rhs }, out, shape)
    }

    pub fn transpose(&mut self, x: &Tensor) -> Tensor {
        let input = self.ensure_node(x);
        let (out, shape) = crate::backend::cpu::transpose_2d(x.data(), x.shape());
        self.record(Op::Transpose { input }, out, shape)
    }

    pub fn sum_all(&mut self, x: &Tensor) -> Tensor {
        let input = self.ensure_node(x);
        let s = crate::backend::cpu::sum_all(x.data());
        self.record(Op::SumAll { input }, vec![s], vec![])
    }

    pub fn mean_all(&mut self, x: &Tensor) -> Tensor {
        let input = self.ensure_node(x);
        let m = crate::backend::cpu::mean_all(x.data());
        self.record(Op::MeanAll { input }, vec![m], vec![])
    }

    pub fn reshape(&mut self, x: &Tensor, new_shape: &[usize]) -> Tensor {
        let input = self.ensure_node(x);
        let from = x.shape().to_vec();
        assert_eq!(
            from.iter().product::<usize>(),
            new_shape.iter().product::<usize>(),
            "reshape: numel mismatch"
        );
        self.record(Op::Reshape { input }, x.data().to_vec(), new_shape.to_vec())
    }

    /// Row-broadcast a 1D tensor `[cols]` to 2D `[rows, cols]`.
    pub fn broadcast_row(&mut self, x: &Tensor, target: &[usize]) -> Tensor {
        let input = self.ensure_node(x);
        let out = crate::tensor::shape::broadcast_row(x.data(), x.shape(), target);
        self.record(
            Op::BroadcastRow {
                input,
                to: target.to_vec(),
            },
            out,
            target.to_vec(),
        )
    }

    /// Fused softmax + cross-entropy. `logits: [batch, classes]`,
    /// `targets: [batch]` of class indices stored as `f32`.
    pub fn softmax_xe(&mut self, logits: &Tensor, targets: &Tensor) -> Tensor {
        let logits_id = self.ensure_node(logits);
        let targets_id = self.ensure_node(targets);
        assert_eq!(logits.rank(), 2, "softmax_xe: logits must be 2D");
        let (batch, classes) = (logits.shape()[0], logits.shape()[1]);
        assert_eq!(
            targets.shape(),
            &[batch],
            "softmax_xe: targets shape must be [batch]"
        );

        let logits_data = logits.data();
        let target_data = targets.data();
        let mut softmax = vec![0.0; batch * classes];
        let mut loss = 0.0_f32;
        for b in 0..batch {
            let row = &logits_data[b * classes..(b + 1) * classes];
            let (l, p) = crate::math::stable::softmax_cross_entropy(row, target_data[b] as usize);
            loss += l;
            softmax[b * classes..(b + 1) * classes].copy_from_slice(&p);
        }
        loss /= batch as f32;
        self.record(
            Op::SoftmaxXE {
                logits: logits_id,
                targets: targets_id,
                softmax,
            },
            vec![loss],
            vec![],
        )
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of `Tape::backward`. Stores one gradient tensor per
/// learnable parameter, keyed by `ParamId`. Optimisers consume this.
#[derive(Default, Debug)]
pub struct GradStore {
    grads: HashMap<ParamId, (Vec<f32>, Vec<usize>)>,
}

impl GradStore {
    pub fn insert(&mut self, id: ParamId, data: Vec<f32>, shape: Vec<usize>) {
        self.grads.insert(id, (data, shape));
    }

    pub fn get(&self, id: ParamId) -> Option<(&[f32], &[usize])> {
        self.grads
            .get(&id)
            .map(|(d, s)| (d.as_slice(), s.as_slice()))
    }

    pub fn len(&self) -> usize {
        self.grads.len()
    }
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = ParamId> + '_ {
        self.grads.keys().copied()
    }
}
