//! Weight initialisation schemes.
//!
//! **Citations.**
//!  - `Xavier`/`Glorot`: X. Glorot & Y. Bengio, "Understanding the
//!    difficulty of training deep feedforward neural networks,"
//!    AISTATS 2010.
//!  - `He`: K. He, X. Zhang, S. Ren & J. Sun, "Delving Deep into
//!    Rectifiers: Surpassing Human-Level Performance on ImageNet
//!    Classification," ICCV 2015.
//!
//! Both papers analyse forward-activation and backward-gradient
//! variance as a function of layer fan-in/fan-out, then pick a
//! distribution whose variance keeps the per-layer signal magnitude
//! roughly constant.

use crate::rng::Rng;
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub enum Init {
    /// All zeros. Useful for biases, almost never for weights.
    Zeros,
    /// `N(0, σ)` with caller-provided `σ`.
    Normal { sigma: f32 },
    /// **Xavier/Glorot (2010).** `U(−a, a)` with
    /// `a = √(6 / (fan_in + fan_out))`. Good for `tanh`/`sigmoid`.
    Xavier,
    /// **He (2015).** `N(0, σ)` with `σ = √(2 / fan_in)`. Good for
    /// ReLU-family activations.
    He,
}

impl Init {
    /// Materialise a weight tensor of `shape = [out_dim, in_dim]`.
    /// (We follow the PyTorch convention: weights are `[out, in]`,
    /// the forward pass is `y = W @ x.T` for batched-row inputs.)
    pub fn tensor(self, shape: &[usize], rng: &mut dyn Rng) -> Tensor {
        let n: usize = shape.iter().product();
        let data = match self {
            Init::Zeros => vec![0.0; n],
            Init::Normal { sigma } => (0..n).map(|_| rng.normal(0.0, sigma)).collect(),
            Init::Xavier => {
                let (fan_in, fan_out) = fan(shape);
                let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
                (0..n)
                    .map(|_| {
                        // U(-bound, bound) from U(0, 1).
                        (rng.next_f32() * 2.0 - 1.0) * bound
                    })
                    .collect()
            }
            Init::He => {
                let (fan_in, _) = fan(shape);
                let sigma = (2.0 / fan_in as f32).sqrt();
                (0..n).map(|_| rng.normal(0.0, sigma)).collect()
            }
        };
        Tensor::from_data(data, shape)
    }
}

/// For shape `[out, in]` returns `(fan_in, fan_out) = (in, out)`.
fn fan(shape: &[usize]) -> (usize, usize) {
    match shape {
        [out, inp] => (*inp, *out),
        [n] => (*n, *n),
        _ => panic!("init: only rank-1 and rank-2 shapes are supported, got {shape:?}"),
    }
}
