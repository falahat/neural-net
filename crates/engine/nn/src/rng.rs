//! Deterministic PRNGs.
//!
//! Every NN function that consumes randomness takes `&mut dyn Rng` —
//! no thread-locals, no globals. Same seed + same sequence of calls
//! ⇒ bit-identical results across runs and machines.
//!
//! **Citations.**
//!  - `Rng::normal`: G. E. P. Box & M. E. Muller, "A note on the
//!    generation of random normal deviates," Annals of Mathematical
//!    Statistics 29(2), June 1958, pp. 610–611.
//!  - `SplitMix64`: Sebastiano Vigna, "An experimental exploration of
//!    Marsaglia's xorshift generators, scrambled," ACM Trans. on
//!    Math. Software 42(4), 2016. Also Java 8's `SplittableRandom`.

pub trait Rng: Send {
    fn next_u32(&mut self) -> u32;

    /// Uniform in `[0, 1)`. Default impl uses the top 24 bits — ample
    /// for `f32` mantissa. Override on backends that have native f32
    /// uniform sources.
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits / 2^24 ∈ [0, 1).
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Standard normal sample via Box-Muller. Sigma defaults to 1.
    ///
    /// Box-Muller transform: given `u₁, u₂` iid `U(0, 1)`,
    /// ```text
    ///     z = sqrt(-2 ln u₁) · cos(2π u₂)
    /// ```
    /// is `N(0, 1)`. We reject `u₁ == 0` (would give `log(0) = -inf`)
    /// by resampling; for a uniform draw the rejection rate is `2⁻²⁴`.
    fn normal(&mut self, mu: f32, sigma: f32) -> f32 {
        let mut u1 = self.next_f32();
        while u1 == 0.0 {
            u1 = self.next_f32();
        }
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let z = r * (2.0 * std::f32::consts::PI * u2).cos();
        mu + sigma * z
    }
}

/// `SplitMix64`. A small, fast, statistically excellent PRNG that
/// produces 64 bits per step from a 64-bit state. `next_u32` takes the
/// high 32 bits each step (the better-mixed half); this discards half the
/// entropy per call but is fine for neural-net init.
///
/// `nn` keeps its OWN copy of this algorithm — the crate has zero
/// workspace dependencies so it stays WASM-reachable — rather than sharing
/// the canonical `determinism` / `kernel` implementation. The
/// [`tests::splitmix64_matches_canonical_known_vectors`] pin proves the
/// copy computes the same function, so it cannot silently drift.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed with any `u64`; seeding with 0 is fine — the constants
    /// below decorrelate even a zero seed.
    pub fn seeded(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    /// Advance the state and return the full 64-bit output.
    /// Constants are from Vigna 2016 / Java 8's `SplittableRandom.mix64`.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Rng for SplitMix64 {
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = SplitMix64::seeded(42);
        let mut b = SplitMix64::seeded(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    /// Known-vector pin against the canonical SplitMix64 reference (Vigna
    /// 2016). `nn` can't share the workspace `determinism` copy (zero-dep,
    /// WASM-reachable), so this guards the copy from silently drifting.
    /// `seeded(s)` initialises state to `s + GAMMA`, so `seeded(0)`'s stream
    /// is the canonical `seed = 0` stream advanced by one — its k-th output
    /// equals the reference's (k+1)-th. `next_u32` keeps the high 32 bits.
    #[test]
    fn splitmix64_matches_canonical_known_vectors() {
        let mut r = SplitMix64::seeded(0);
        assert_eq!(r.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(r.next_u64(), 0x06c4_5d18_8009_454f);
        assert_eq!(r.next_u64(), 0xf88b_b8a8_724c_81ec);
        // `next_u32` returns the high 32 bits of the next 64-bit step.
        let mut r32 = SplitMix64::seeded(0);
        assert_eq!(r32.next_u32(), 0x6e78_9e6a);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::seeded(42);
        let mut b = SplitMix64::seeded(43);
        let mut diff = 0;
        for _ in 0..100 {
            if a.next_u32() != b.next_u32() {
                diff += 1;
            }
        }
        assert!(diff > 90, "expected most samples to differ, got {diff}/100");
    }

    #[test]
    fn next_f32_in_range() {
        let mut r = SplitMix64::seeded(7);
        for _ in 0..1000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x), "next_f32 out of range: {x}");
        }
    }

    #[test]
    fn normal_approx_zero_mean_unit_var() {
        let mut r = SplitMix64::seeded(1);
        let n = 10_000;
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for i in 1..=n {
            let x = r.normal(0.0, 1.0) as f64;
            let dx = x - mean;
            mean += dx / i as f64;
            m2 += dx * (x - mean);
        }
        let var = m2 / n as f64;
        // 10k samples; tolerate ~0.05 wobble.
        assert!(mean.abs() < 0.05, "mean too far from 0: {mean}");
        assert!((var - 1.0).abs() < 0.05, "var too far from 1: {var}");
    }
}
