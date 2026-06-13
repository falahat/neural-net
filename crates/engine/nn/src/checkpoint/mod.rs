//! Native checkpoint format. Layout:
//!
//! ```text
//!  ┌──────────────────────────────────────┐
//!  │ u32_le  header_len                   │
//!  │ [u8; header_len]  JSON manifest      │   ← shapes, names, byte offsets
//!  │ [f32; total]      tensor data        │   ← little-endian, contiguous
//!  └──────────────────────────────────────┘
//! ```
//!
//! Manifest grammar (hand-rolled JSON, no serde dep):
//!
//! ```json
//! {"version": 1, "params": [
//!     {"path": "0.weight", "shape": [16, 2], "offset": 0,   "len": 32},
//!     {"path": "0.bias",   "shape": [16],    "offset": 128, "len": 16},
//!     ...
//! ]}
//! ```
//!
//! This is a structural cousin of HuggingFace's safetensors format
//! (which uses a 64-bit header length and a more elaborate per-tensor
//! dict); the header + flat-data shape keeps the lib dependency-free.

mod native;
pub use native::{load_bytes, save_bytes, CheckpointError};
