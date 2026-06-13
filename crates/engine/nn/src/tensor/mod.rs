//! `Tensor` — `Arc<TensorInner>` handle plus an `Option<NodeId>` into
//! the active tape. Clone is a refcount bump; the `Vec<f32>` data is
//! never copied.
//!
//! `node` is `Some` when the tensor came from a recorded op (or was
//! explicitly leafed via `tape.leaf()`); ops on a `None` tensor
//! auto-register it as a transient leaf on first use.
//!
//! v1 supports rank 0 (scalar — losses), 1 (vectors — biases), 2
//! (matrices — everything else). Higher ranks aren't blocked by the
//! type; just not implemented because no current widget needs them.
//! GPU residency lives in `backend::gpu::GpuTensor`, not here.

pub mod shape;

use std::fmt;
use std::sync::Arc;

/// Identifier for a node on a tape. Internal; users never construct these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u32);

#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<TensorInner>,
}

#[derive(Clone)]
pub(crate) struct TensorInner {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    /// `Some(id)` when this tensor was produced (or registered) on a
    /// tape and gradients can flow through it. `None` for "detached"
    /// tensors (raw constants, freshly-loaded weights before any
    /// forward pass).
    pub node: Option<NodeId>,
}

impl Tensor {
    pub fn from_data(data: Vec<f32>, shape: &[usize]) -> Self {
        let n_expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n_expected,
            "Tensor::from_data: shape {shape:?} expects {n_expected} elements, got {}",
            data.len()
        );
        Self {
            inner: Arc::new(TensorInner {
                shape: shape.to_vec(),
                data,
                node: None,
            }),
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::from_data(vec![0.0; n], shape)
    }

    /// Borrow the data slice. Cheap.
    pub fn data(&self) -> &[f32] {
        &self.inner.data
    }

    /// Mutable access to the data slice — copy-on-write through the
    /// `Arc` (clones the buffer only if other handles share it).
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut Arc::make_mut(&mut self.inner).data
    }
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }
    pub fn rank(&self) -> usize {
        self.inner.shape.len()
    }
    pub fn numel(&self) -> usize {
        self.inner.data.len()
    }

    /// `Some(scalar)` only when the tensor is rank-0 or has a single element.
    pub fn item(&self) -> Option<f32> {
        if self.numel() == 1 {
            Some(self.inner.data[0])
        } else {
            None
        }
    }

    pub(crate) fn node(&self) -> Option<NodeId> {
        self.inner.node
    }

    /// Construct a tape-bound tensor. Internal — used by `Tape::record_*`.
    pub(crate) fn with_node(data: Vec<f32>, shape: Vec<usize>, node: NodeId) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Self {
            inner: Arc::new(TensorInner {
                shape,
                data,
                node: Some(node),
            }),
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.inner.shape)
            .field("data", &Preview(&self.inner.data))
            .field("node", &self.inner.node)
            .finish()
    }
}

struct Preview<'a>(&'a [f32]);
impl fmt::Debug for Preview<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0.len().min(8);
        write!(f, "[")?;
        for (i, v) in self.0[..n].iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:.4}")?;
        }
        if self.0.len() > n {
            write!(f, ", … ({} more)", self.0.len() - n)?;
        }
        write!(f, "]")
    }
}
