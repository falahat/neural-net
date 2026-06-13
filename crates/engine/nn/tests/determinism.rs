//! Same seed → bit-identical training trajectory. The determinism
//! gate (design doc §14.4).
//!
//! **Skipped when `--features parallel` or `--features gpu` is on.**
//! Both paths use parallel reductions whose ordering depends on the
//! thread scheduler; fp adds are not associative, so bit-exact
//! results require the deterministic single-threaded scalar path.
//! The CPU-deterministic guarantee remains the contract of the
//! DEFAULT build — that's what gets tested in CI's deterministic
//! lane.

#![cfg(not(any(feature = "parallel", feature = "gpu", feature = "gpu-cuda")))]

use nn::activation::Tanh;
use nn::init::Init;
use nn::loss::Mse;
use nn::module::{Linear, Sequential};
use nn::optim::Adam;
use nn::rng::SplitMix64;
use nn::tensor::Tensor;
use nn::train::Trainer;

fn build_and_train(seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::seeded(seed);
    let mut trainer = Trainer::builder()
        .model(Box::new(Sequential::new(vec![
            Box::new(Linear::new(2, 4, Init::Xavier, &mut rng)),
            Box::new(Tanh),
            Box::new(Linear::new(4, 1, Init::Xavier, &mut rng)),
        ])))
        .loss(Box::new(Mse))
        .optim(Box::new(Adam::new(0.05)))
        .build();
    let x = Tensor::from_data(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let y = Tensor::from_data(vec![0.0, 1.0, 1.0, 0.0], &[4, 1]);

    let mut losses = Vec::with_capacity(50);
    for _ in 0..50 {
        losses.push(trainer.train_step(&x, &y));
    }
    losses
}

#[test]
fn same_seed_same_trajectory() {
    let a = build_and_train(2026);
    let b = build_and_train(2026);
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        // Bit-identical: no parallel reductions, fixed order.
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "step {i}: a={} b={} (bits differ)",
            a[i],
            b[i]
        );
    }
}

#[test]
fn different_seeds_diverge() {
    let a = build_and_train(1);
    let b = build_and_train(2);
    let any_diff = a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits());
    assert!(
        any_diff,
        "different seeds should produce different trajectories"
    );
}
