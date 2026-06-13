//! Smoke test: GPU matmul produces the same values as scalar matmul,
//! within 1e-3 absolute (wgpu reductions aren't bit-deterministic).
//!
//! Only runs under `--features gpu`. Skipped silently otherwise.

#![cfg(feature = "gpu")]

#[test]
fn gpu_matches_scalar_matmul() {
    let (m, k, n) = (16, 8, 24);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.231).cos()).collect();

    // Reference triple loop.
    let mut ref_c = vec![0.0_f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let aik = a[i * k + kk];
            for j in 0..n {
                ref_c[i * n + j] += aik * b[kk * n + j];
            }
        }
    }

    // GPU.
    let (gpu_c, shape) = nn::backend::gpu::matmul(&a, &[m, k], &b, &[k, n]);
    assert_eq!(shape, vec![m, n]);
    assert_eq!(gpu_c.len(), m * n);

    let mut max_diff = 0.0_f32;
    for i in 0..m * n {
        let d = (gpu_c[i] - ref_c[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(
        max_diff < 1e-3,
        "GPU matmul diverges from scalar; max diff = {max_diff}"
    );
    println!("GPU matmul ok: max diff vs scalar = {max_diff:.6}");
}
