//! Round-trip: train an MLP, save, build a *fresh* MLP, load the
//! saved bytes, verify the fresh model produces bit-identical outputs.

use nn::activation::Tanh;
use nn::checkpoint::{load_bytes, save_bytes};
use nn::init::Init;
use nn::loss::Mse;
use nn::module::{Linear, Sequential};
use nn::optim::Adam;
use nn::rng::SplitMix64;
use nn::tensor::Tensor;
use nn::train::Trainer;

fn build_model(seed: u64) -> Sequential {
    let mut rng = SplitMix64::seeded(seed);
    Sequential::new(vec![
        Box::new(Linear::new(3, 5, Init::Xavier, &mut rng)),
        Box::new(Tanh),
        Box::new(Linear::new(5, 2, Init::Xavier, &mut rng)),
    ])
}

#[test]
fn checkpoint_round_trip() {
    // Train a bit so weights are non-trivial.
    let model = build_model(42);
    let mut trainer = Trainer::builder()
        .model(Box::new(model))
        .loss(Box::new(Mse))
        .optim(Box::new(Adam::new(0.01)))
        .build();
    let x = Tensor::from_data(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.1, 0.0, 0.1, 0.9, -0.5, 0.2],
        &[4, 3],
    );
    let y = Tensor::from_data(vec![1.0, 0.0, 0.5, 0.5, -0.5, 1.0, 0.0, -1.0], &[4, 2]);
    for _ in 0..50 {
        trainer.train_step(&x, &y);
    }

    // Save.
    let blob = save_bytes(&*trainer.model);
    let trained_pred = trainer.predict(&x);

    // Build a fresh (different seed) model, prove its prediction differs.
    let fresh = build_model(99);
    let mut fresh_trainer = Trainer::builder()
        .model(Box::new(fresh))
        .loss(Box::new(Mse))
        .optim(Box::new(Adam::new(0.01)))
        .build();
    let fresh_pred = fresh_trainer.predict(&x);
    assert_ne!(
        trained_pred.data(),
        fresh_pred.data(),
        "fresh model with different seed should differ from trained model"
    );

    // Load and compare. Now they should be bit-identical.
    load_bytes(&mut *fresh_trainer.model, &blob).expect("load_bytes");
    let loaded_pred = fresh_trainer.predict(&x);
    assert_eq!(
        trained_pred.data(),
        loaded_pred.data(),
        "after load, predictions should be bit-identical"
    );
}
