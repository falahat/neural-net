//! Scalar CPU compute primitives.
//!
//! Every operation is a pure function on slices: no `&self`, no
//! global state, no thread pool. Deterministic by construction
//! (fixed left-to-right summation order, no parallel reductions).
//!
//! These are the operations the autograd tape (`autograd::op`) dispatches
//! to. Nothing here knows about the tape; this file is a numerics
//! library that happens to be the v1 backend.
//!
//! ## SIMD acceleration
//!
//! When `--features simd` is enabled, the hot functions in this file
//! (`matmul`, `add`, `sub`, `mul`, `div`, `relu`, `sigmoid`) become
//! one-line wrappers that delegate to `backend::simd`.

// ─── Primitive 1: matmul (2D × 2D) ─────────────────────────────────────

/// `c = a @ b` for `a: [m, k]`, `b: [k, n]`, result `[m, n]`.
///
/// Triple-loop, ijk order. Order chosen so the inner accumulator is
/// scalar (helps the autovectoriser) and `b` access is row-major
/// stride-1 — most cache-friendly for the typical [batch, hidden] @
/// [hidden, out] case.
#[cfg(feature = "parallel")]
pub fn matmul(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    super::parallel::matmul(a, a_shape, b, b_shape)
}

#[cfg(all(feature = "simd", not(feature = "parallel")))]
pub fn matmul(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    super::simd::matmul(a, a_shape, b, b_shape)
}

#[cfg(not(any(feature = "simd", feature = "parallel")))]
pub fn matmul(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(a_shape.len(), 2, "matmul: lhs must be 2D, got {a_shape:?}");
    assert_eq!(b_shape.len(), 2, "matmul: rhs must be 2D, got {b_shape:?}");
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    assert_eq!(k, k2, "matmul: inner dims disagree ({k} vs {k2})");

    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for kk in 0..k {
            let aik = a[i * k + kk];
            for j in 0..n {
                out[i * n + j] += aik * b[kk * n + j];
            }
        }
    }
    (out, vec![m, n])
}

// ─── Primitive 2: elementwise unary ────────────────────────────────────

pub fn map_unary(t: &[f32], f: impl Fn(f32) -> f32) -> Vec<f32> {
    t.iter().map(|&x| f(x)).collect()
}

// ─── Primitive 3: elementwise binary ───────────────────────────────────
//
// All four require identical shapes. Broadcasting is the caller's job
// (see `tensor::shape::broadcast_row`); keeping it that way means
// these stay simple and obviously vectorisable.

#[cfg(feature = "simd")]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    super::simd::add(a, b)
}
#[cfg(feature = "simd")]
pub fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    super::simd::sub(a, b)
}
#[cfg(feature = "simd")]
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    super::simd::mul(a, b)
}
#[cfg(feature = "simd")]
pub fn div(a: &[f32], b: &[f32]) -> Vec<f32> {
    super::simd::div(a, b)
}

#[cfg(not(feature = "simd"))]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(
        a.len(),
        b.len(),
        "add: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

#[cfg(not(feature = "simd"))]
pub fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x - y).collect()
}

#[cfg(not(feature = "simd"))]
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

#[cfg(not(feature = "simd"))]
pub fn div(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x / y).collect()
}

// ─── Primitive 4: reductions ───────────────────────────────────────────

/// Sum every element. Axis-restricted reductions follow below.
pub fn sum_all(t: &[f32]) -> f32 {
    // Left-to-right; deterministic. No parallel reductions in v1.
    let mut acc = 0.0_f32;
    for &x in t {
        acc += x;
    }
    acc
}

pub fn mean_all(t: &[f32]) -> f32 {
    if t.is_empty() {
        return 0.0;
    }
    sum_all(t) / t.len() as f32
}

// ─── Primitive 5: shape ops ────────────────────────────────────────────

pub fn transpose_2d(t: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(shape.len(), 2);
    let (rows, cols) = (shape[0], shape[1]);
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = t[r * cols + c];
        }
    }
    (out, vec![cols, rows])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x2_identity() {
        let i = vec![1.0, 0.0, 0.0, 1.0];
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let (c, s) = matmul(&i, &[2, 2], &a, &[2, 2]);
        assert_eq!(s, &[2, 2]);
        assert_eq!(c, a);
    }

    #[test]
    fn matmul_3x2_times_2x4() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]
        let b = vec![1.0; 8]; // [2, 4]
        let (c, s) = matmul(&a, &[3, 2], &b, &[2, 4]);
        assert_eq!(s, &[3, 4]);
        assert_eq!(
            c,
            vec![3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 7.0, 11.0, 11.0, 11.0, 11.0]
        );
    }

    #[test]
    fn transpose_round_trip() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (b, bs) = transpose_2d(&a, &[2, 3]);
        let (c, cs) = transpose_2d(&b, &bs);
        assert_eq!(cs, &[2, 3]);
        assert_eq!(c, a);
    }
}
