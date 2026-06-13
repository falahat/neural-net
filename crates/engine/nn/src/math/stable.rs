//! Stable elementary functions for neural-net workloads.
//!
//! Each routine ships with the derivation that makes it safe at
//! extreme inputs (where the textbook formula NaNs or overflows).

/// `log(Σ exp(xᵢ))` without overflow.
///
/// **Citation.** Shift-by-max identity is folklore (Boyd & Vandenberghe,
/// *Convex Optimization* §3.1.5, CUP 2004). Forward-error analysis:
/// Blanchard, Higham & Mary, "Accurately Computing the Log-Sum-Exp
/// and Softmax Functions," IMA J. Numer. Anal. 41(4), 2021.
///
/// ## Derivation
///
/// For any constant `c`:
/// ```text
///     Σ exp(xᵢ) = exp(c) · Σ exp(xᵢ − c)
/// ⟹  log Σ exp(xᵢ) = c + log Σ exp(xᵢ − c).
/// ```
/// Choose `c = max xᵢ`. Then the largest summand is `exp(0) = 1`, the
/// sum lies in `[1, n]`, and `log` of that is in `[0, log n]` —
/// comfortably representable. Other summands are in `(0, 1]`; any that
/// underflow to zero contribute negligibly.
pub fn log_sum_exp(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return f32::NEG_INFINITY;
    }
    let m = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return m;
    } // all -inf ⇒ stay -inf
    let s: f32 = xs.iter().map(|&x| (x - m).exp()).sum();
    m + s.ln()
}

/// `σ(x) = 1 / (1 + e⁻ˣ)`, branched to avoid overflow on either tail.
///
/// **Citation.** Branched form is standard numerical-analysis
/// practice (Higham, *Accuracy and Stability of Numerical Algorithms*
/// 2nd ed., SIAM 2002, ch. 1) applied to the logistic.
///
/// ## Derivation
///
/// `σ(x) = 1/(1+e⁻ˣ)` blows up at large `−x`. Symmetric form `eˣ/(1+eˣ)`
/// blows up at large `+x`. Branching on sign so each branch only
/// computes `e^{-|x|} ∈ (0, 1]` is overflow-free in both directions:
/// ```text
///     σ(x) = 1 / (1 + e⁻ˣ)      x ≥ 0
///          = eˣ / (1 + eˣ)      x < 0
/// ```
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `softplus(x) = log(1 + eˣ)`, stable form.
///
/// **Citation.** `max(x, 0) + log(1 + e^{-|x|})` identity is in
/// Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press 2016)
/// §6.3.3. `log1p` itself is C89 / IEEE-754 (libm, §7.5.3).
///
/// ## Derivation
///
/// Naïve `log(1 + eˣ)` overflows for `x ≳ 89` in `f32`. For `x > 0`:
/// ```text
///     log(1 + eˣ) = log(eˣ(e⁻ˣ + 1)) = x + log(1 + e⁻ˣ).
/// ```
/// Combining the two branches:
/// ```text
///     softplus(x) = max(x, 0) + log(1 + e^{-|x|}).
/// ```
/// Argument of `log1p` is in `(0, 2]` — never near 1, so `ln_1p`
/// avoids catastrophic cancellation.
pub fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Fused softmax + cross-entropy on a single example's logits.
///
/// Returns `(loss, softmax_probs)`. The probabilities are returned so
/// the backward pass can reuse them instead of recomputing.
///
/// **Citation.** `ℓ = LSE(x) − x_t` identity is standard (Goodfellow,
/// Bengio & Courville, *Deep Learning* (MIT Press 2016) §6.2.2.2,
/// eq. 6.30 for the gradient). Fused versions ship in PyTorch & JAX
/// under various names.
///
/// ## Derivation
///
/// ```text
///     ℓ = −log p_t = −log( e^{x_t} / Σ e^{x_j} ) = LSE(x) − x_t.
/// ```
/// No exponential materialises in the forward path of the loss. The
/// gradient w.r.t. logits is `softmax(x) − one_hot(t)`, bounded in
/// `[−1, 1]` — robust by construction.
pub fn softmax_cross_entropy(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    debug_assert!(target < logits.len());
    let lse = log_sum_exp(logits);
    let loss = lse - logits[target];
    let probs: Vec<f32> = logits.iter().map(|&x| (x - lse).exp()).collect();
    (loss, probs)
}

/// Welford's online variance algorithm — numerically stable variance
/// over a stream of values.
///
/// **Citation.** B. P. Welford, "Note on a method for calculating
/// corrected sums of squares and products," Technometrics 4(3),
/// August 1962, pp. 419–420. Popularised in Knuth, *TAOCP* Vol. 2
/// §4.2.2.
///
/// ## Why the naïve form fails
///
/// `Var(X) = E[X²] − E[X]²` at `xᵢ ∈ [10⁶, 10⁶+1]` computes
/// `≈ 10¹² − 10¹²` — both sides have ~10⁵ absolute error in f32 and
/// the result is dominated by cancellation noise (can go negative,
/// breaking sqrt).
///
/// ## Welford recurrence
///
/// ```text
///     x̄_n = x̄_{n−1} + (xₙ − x̄_{n−1}) / n
///     M₂_n = M₂_{n−1} + (xₙ − x̄_{n−1})(xₙ − x̄_n)
/// ```
/// The product uses `x̄` *before* and *after* the update — the trick
/// that keeps `M₂` provably non-negative.
#[derive(Debug, Clone, Default)]
pub struct WelfordVariance {
    n: u64,
    mean: f64,
    m2: f64,
}

impl WelfordVariance {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, x: f32) {
        self.n += 1;
        let dx_old = x as f64 - self.mean;
        self.mean += dx_old / self.n as f64;
        let dx_new = x as f64 - self.mean;
        self.m2 += dx_old * dx_new;
    }

    pub fn variance(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            (self.m2 / self.n as f64) as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lse_extreme_positive() {
        let v = log_sum_exp(&[1000.0, 1000.0]);
        assert!(
            (v - (1000.0 + 2.0_f32.ln())).abs() < 1e-3,
            "expected 1000+ln 2, got {v}"
        );
    }

    #[test]
    fn lse_extreme_negative() {
        // -1000 and -999: max=-999, sum = e^{-1} + 1 = 1 + e^{-1}.
        let v = log_sum_exp(&[-1000.0, -999.0]);
        let expected = -999.0 + (1.0_f32 + (-1.0_f32).exp()).ln();
        assert!((v - expected).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_extremes_no_nan() {
        assert!((sigmoid(-1000.0) - 0.0).abs() < 1e-30);
        assert!((sigmoid(1000.0) - 1.0).abs() < 1e-30);
        assert!(sigmoid(0.0).abs() - 0.5 < 1e-6);
    }

    #[test]
    fn softplus_no_overflow() {
        // Naïve log(1 + exp(100)) would be infinity; stable gives 100.
        assert!((softplus(100.0) - 100.0).abs() < 1e-3);
        assert!((softplus(-100.0)).abs() < 1e-30);
    }

    #[test]
    fn softmax_xe_confident_correct() {
        let (loss, p) = softmax_cross_entropy(&[10.0, -10.0], 0);
        assert!(
            loss.abs() < 1e-6,
            "confident-correct loss should be ~0, got {loss}"
        );
        assert!((p[0] - 1.0).abs() < 1e-6);
        assert!(p[1].abs() < 1e-6);
    }

    #[test]
    fn welford_cancellation_safe() {
        let mut w = WelfordVariance::new();
        w.push(1e6);
        w.push(1e6 + 1.0);
        // True variance of [1e6, 1e6+1] is 0.25 (population).
        assert!((w.variance() - 0.25).abs() < 1e-3);
    }
}
