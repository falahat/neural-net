//! Shape utilities — the narrow row-broadcast helpers `Linear` needs for
//! bias addition + its backward pass. Shapes themselves are plain
//! `[usize]` / `Vec<usize>` slices throughout the crate.

/// Broadcast `src` (1D bias) row-wise to a 2D `target` shape. Returns
/// the flat data for the broadcast result. Used by `Linear` to add
/// bias to the matmul output.
///
/// Why so narrow? Full numpy/PyTorch broadcasting is ~150 lines and
/// every textbook widget only ever needs `[out_dim] -> [batch, out_dim]`
/// or `[1, out_dim] -> [batch, out_dim]`. We handle exactly those.
pub fn broadcast_row(src: &[f32], src_shape: &[usize], target: &[usize]) -> Vec<f32> {
    assert_eq!(target.len(), 2, "broadcast_row: target must be 2D");
    let (rows, cols) = (target[0], target[1]);
    match src_shape {
        // Scalar broadcast: fill rows*cols with the single value.
        [] | [1] => vec![src[0]; rows * cols],
        // 1D vector of length `cols` (or its `[1, cols]` 2D spelling):
        // repeat down the rows.
        [c] | [1, c] if *c == cols => {
            let mut out = Vec::with_capacity(rows * cols);
            for _ in 0..rows {
                out.extend_from_slice(src);
            }
            out
        }
        _ => panic!("broadcast_row: cannot broadcast {src_shape:?} to {target:?}"),
    }
}

/// Sum a broadcast result back into its source shape — the dual operation
/// needed when computing backward through a broadcast. (E.g. bias grad is
/// the column-sum of the output grad.)
pub fn unbroadcast_to(grad: &[f32], grad_shape: &[usize], target: &[usize]) -> Vec<f32> {
    if grad_shape == target {
        return grad.to_vec();
    }
    assert_eq!(grad_shape.len(), 2, "unbroadcast_to: only 2D supported");
    let (rows, cols) = (grad_shape[0], grad_shape[1]);
    match target {
        [] | [1] => vec![grad.iter().sum()],
        [c] | [1, c] if *c == cols => {
            let mut out = vec![0.0; cols];
            for r in 0..rows {
                for k in 0..cols {
                    out[k] += grad[r * cols + k];
                }
            }
            out
        }
        _ => panic!("unbroadcast_to: cannot unbroadcast {grad_shape:?} -> {target:?}"),
    }
}
