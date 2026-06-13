//! Multi-threaded CPU sgemm via the [`gemm`](https://crates.io/crates/gemm)
//! crate (Faer / Sarah Williams) — same crate candle uses. ~38× our
//! scalar matmul thanks to BLIS-style tiling + AVX2/AVX-512
//! microkernel + rayon parallelism over the M dimension.
//!
//! Only `matmul` is here; parallel elementwise wouldn't pay back the
//! rayon dispatch cost at our shape sizes (memory-bandwidth bound,
//! not FMA bound).
//!
//! Parallel reductions reorder fp adds, so results vary by ~1 ULP
//! between runs. `tests/determinism.rs` skips itself when this
//! feature is on; the default scalar path stays bit-exact.

use gemm::{gemm, Parallelism};

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

    // gemm computes `C = α·A·B + β·C` with arbitrary row/col strides.
    // We have row-major A:[m,k], B:[k,n], C:[m,n].
    //   row_stride_a = k   (one row spans k floats)
    //   col_stride_a = 1
    //   row_stride_b = n
    //   col_stride_b = 1
    //   row_stride_c = n
    //   col_stride_c = 1
    //
    // `Parallelism::Rayon(0)` = let gemm pick the thread count.

    let mut c = vec![0.0_f32; m * n];
    unsafe {
        gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1 as isize, // col-stride of C (one f32)
            n as isize, // row-stride of C
            false,      // c_read_only — we're overwriting C, β=0
            a.as_ptr(),
            1 as isize, // col-stride of A
            k as isize, // row-stride of A
            b.as_ptr(),
            1 as isize, // col-stride of B
            n as isize, // row-stride of B
            0.0_f32,    // β (multiplier on existing C — we zero it)
            1.0_f32,    // α
            false,
            false,
            false, // conj A, B, C
            Parallelism::Rayon(0),
        );
    }
    (c, vec![m, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_enough(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    /// Equivalent to scalar `cpu::matmul` within ~1e-3 absolute
    /// (parallel reduction reorders adds; not bit-exact).
    #[test]
    fn parallel_matches_scalar_within_tolerance() {
        let a: Vec<f32> = (0..64 * 32).map(|i| (i as f32 * 0.137).sin()).collect();
        let b: Vec<f32> = (0..32 * 48).map(|i| (i as f32 * 0.231).cos()).collect();
        let (par, _) = matmul(&a, &[64, 32], &b, &[32, 48]);

        // Run the scalar reference: a direct triple loop, NOT through
        // `backend::cpu::matmul` (which under `--features parallel`
        // would dispatch to us — recursion).
        let (m, k, n) = (64, 32, 48);
        let mut ref_c = vec![0.0_f32; m * n];
        for i in 0..m {
            for kk in 0..k {
                let aik = a[i * k + kk];
                for j in 0..n {
                    ref_c[i * n + j] += aik * b[kk * n + j];
                }
            }
        }

        assert!(
            close_enough(&par, &ref_c, 1e-3),
            "parallel matmul diverges from scalar; max diff {}",
            par.iter()
                .zip(&ref_c)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max)
        );
    }
}
