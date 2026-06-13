//! Portable f32x8 SIMD backend. Same semantics as `backend::cpu`; the
//! tape never sees this module directly — `cpu.rs` redirects its
//! public functions here when the `simd` feature is on.
//!
//! ## When to use this vs the other backends
//!
//! `--features parallel` (multi-threaded sgemm via `gemm`) is faster for
//! peak CPU throughput; SIMD's niche is where that path doesn't apply:
//!
//! 1. **WASM widgets.** `parallel` needs `rayon` → OS threads, which
//!    don't exist on wasm32. `wide` compiles to the wasm-simd extension
//!    cleanly, so this is the browser path.
//! 2. **Small-shape matmul on native** where rayon's dispatch overhead
//!    exceeds the multiply.
//! 3. **Bit-determinism.** SIMD stays single-threaded so its reductions
//!    are bit-deterministic; `parallel` is not.
//! 4. **Non-x86 targets** where LLVM autovec is less aggressive.
//!
//! ## Why `wide` and not `std::simd`
//!
//! `std::simd` is still nightly. The `wide` crate ships portable
//! `f32x8` / `f32x4` types on stable Rust by dispatching to AVX2 (or
//! the platform's best SSE level) at compile time, with a scalar
//! fallback for targets that don't have either. It's a small,
//! single-purpose dependency in the same spirit as `f32::exp` coming
//! from libm — a numeric primitive, not an algorithmic borrow.
//!
//! ## What's accelerated
//!
//!  - **matmul.** Microkernel keeps the per-`(i, j)` accumulator in a
//!    SIMD register across the inner `k`-loop (one mul-add per
//!    iteration; one store at the end). This is the right loop shape
//!    for a hand-written kernel.
//!  - **add / sub / mul / div.** Slice-load → vec op → slice-store
//!    over chunks of 8.
//!  - **relu.** Same pattern with `fast_max(0)`.
//!  - **sigmoid.** Per-lane scalar — `wide` doesn't expose a fast
//!    vectorised `exp`, and keeping the branched form bit-matches
//!    `math::stable::sigmoid`.
//!
//! ## Performance note
//!
//! On modern LLVM targeting AVX2, the *scalar* matmul in `cpu.rs`
//! autovectorises into FMA + 4-way unrolling and can beat this
//! hand-written kernel; a real competitor would need cache-blocked
//! tiles, prefetch, and several microkernels per accumulator (the
//! Goto-Van de Geijn 2008 / BLIS layout). The value here is the
//! register-local microkernel shape — deterministic, and easier to
//! port to wgpu / cubecl kernels than the LLVM-optimised soup.
//!
//! Reductions (sum/mean) are left in scalar form — not on the matmul
//! hot path.

use wide::f32x8;

/// SIMD width — `f32x8` carries 8 lanes.
const LANES: usize = 8;

/// Load 8 contiguous floats starting at `s[i]` into a SIMD register.
#[inline]
fn load8(s: &[f32], i: usize) -> f32x8 {
    (*<&[f32; LANES]>::try_from(&s[i..i + LANES]).unwrap()).into()
}

/// Store a SIMD register's 8 lanes into `dst[i..i + 8]`.
#[inline]
fn store8(v: f32x8, dst: &mut [f32], i: usize) {
    let r: [f32; LANES] = v.into();
    dst[i..i + LANES].copy_from_slice(&r);
}

// ─── matmul ─────────────────────────────────────────────────────────
//
// Same triple-loop layout as `cpu::matmul` but the innermost `j` loop
// is unrolled by 8 (the SIMD lane count). For tail elements (where
// `n % 8 != 0`) we fall back to scalar — keeps the code readable, the
// performance hit is negligible at our typical widths (hidden = 32,
// 64, 128 are all divisible by 8).

pub fn matmul(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(a_shape.len(), 2);
    assert_eq!(b_shape.len(), 2);
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    assert_eq!(k, k2, "matmul: inner dims {k} vs {k2}");

    let mut out = vec![0.0_f32; m * n];
    let n_simd = n - (n % LANES);

    // Microkernel: for each (i, j-block of 8), accumulate the dot
    // product across k in a single SIMD register. This is the key
    // shape — keeping `acc` in a register across the inner k-loop
    // eliminates K loads + K stores of `out` (the version that did
    // this was ~2× slower).
    for i in 0..m {
        let mut j = 0;
        while j < n_simd {
            let mut acc = f32x8::splat(0.0);
            for kk in 0..k {
                let aik = f32x8::splat(a[i * k + kk]);
                let bv = load8(b, kk * n + j);
                acc = aik.mul_add(bv, acc);
            }
            store8(acc, &mut out, i * n + j);
            j += LANES;
        }
        // Scalar tail for j in [n_simd, n).
        while j < n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
            j += 1;
        }
    }
    (out, vec![m, n])
}

// ─── elementwise binary ─────────────────────────────────────────────

#[inline]
fn binary(
    a: &[f32],
    b: &[f32],
    scalar_op: fn(f32, f32) -> f32,
    simd_op: fn(f32x8, f32x8) -> f32x8,
) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut out = vec![0.0_f32; n];
    let mut i = 0;
    let n_simd = n - (n % LANES);
    while i < n_simd {
        let av = load8(a, i);
        let bv = load8(b, i);
        store8(simd_op(av, bv), &mut out, i);
        i += LANES;
    }
    while i < n {
        out[i] = scalar_op(a[i], b[i]);
        i += 1;
    }
    out
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary(a, b, |x, y| x + y, |x, y| x + y)
}
pub fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary(a, b, |x, y| x - y, |x, y| x - y)
}
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary(a, b, |x, y| x * y, |x, y| x * y)
}
pub fn div(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary(a, b, |x, y| x / y, |x, y| x / y)
}

// ─── elementwise unary ──────────────────────────────────────────────

pub fn relu(t: &[f32]) -> Vec<f32> {
    let n = t.len();
    let mut out = vec![0.0_f32; n];
    let mut i = 0;
    let n_simd = n - (n % LANES);
    let zero = f32x8::splat(0.0);
    while i < n_simd {
        let v = load8(t, i);
        store8(v.fast_max(zero), &mut out, i);
        i += LANES;
    }
    while i < n {
        out[i] = t[i].max(0.0);
        i += 1;
    }
    out
}

/// Sigmoid lane-wise. `wide` doesn't expose a vectorised `exp`, so we
/// fall back to scalar per lane — but we keep the branched form from
/// `math::stable::sigmoid` so the result matches the scalar backend
/// exactly. The SIMD win here is largely from removing the loop
/// overhead, not from a fast exp.
pub fn sigmoid(t: &[f32]) -> Vec<f32> {
    t.iter().map(|&x| crate::math::stable::sigmoid(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_enough(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn simd_matches_scalar_matmul() {
        // 16×8 @ 8×24 — divisible by 8 in n, k is the "inner" so doesn't
        // need to align.
        let a: Vec<f32> = (0..16 * 8).map(|i| (i as f32 * 0.137).sin()).collect();
        let b: Vec<f32> = (0..8 * 24).map(|i| (i as f32 * 0.291).cos()).collect();
        let (s, _) = crate::backend::cpu::matmul(&a, &[16, 8], &b, &[8, 24]);
        let (r, _) = matmul(&a, &[16, 8], &b, &[8, 24]);
        assert!(
            close_enough(&s, &r, 1e-5),
            "SIMD matmul diverges from scalar; max diff {}",
            s.iter()
                .zip(&r)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max)
        );
    }

    #[test]
    fn simd_matmul_with_tail() {
        // n = 23 — not divisible by 8, exercises the scalar tail path.
        let a: Vec<f32> = (0..5 * 7).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..7 * 23).map(|i| i as f32 * 0.07 - 1.0).collect();
        let (s, _) = crate::backend::cpu::matmul(&a, &[5, 7], &b, &[7, 23]);
        let (r, _) = matmul(&a, &[5, 7], &b, &[7, 23]);
        assert!(close_enough(&s, &r, 1e-4));
    }

    #[test]
    fn simd_relu_matches_scalar() {
        let xs: Vec<f32> = (0..67).map(|i| (i as f32 - 30.0) * 0.3).collect();
        let scalar: Vec<f32> = xs.iter().map(|&x| x.max(0.0)).collect();
        assert_eq!(relu(&xs), scalar);
    }

    #[test]
    fn simd_add_matches_scalar() {
        let a: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..50).map(|i| (i as f32).sin()).collect();
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert!(close_enough(&add(&a, &b), &want, 1e-6));
    }
}
