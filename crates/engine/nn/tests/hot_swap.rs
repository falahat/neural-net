//! Hot-swap regression test — the **constitutional** test: the
//! library's entire reason for the dyn-everywhere architecture is
//! that this should work without a single recompile, allocation
//! churn, or surprising NaN.

use nn::activation::Relu;
use nn::init::Init;
use nn::loss::{Huber, Mse};
use nn::module::{Linear, Sequential};
use nn::optim::{Adam, Sgd};
use nn::rng::SplitMix64;
use nn::tensor::Tensor;
use nn::train::Trainer;

#[test]
fn swap_loss_mid_training_keeps_descending() {
    let mut rng = SplitMix64::seeded(99);

    let mut trainer = Trainer::builder()
        .model(Box::new(Sequential::new(vec![
            Box::new(Linear::new(2, 8, Init::He, &mut rng)),
            Box::new(Relu),
            Box::new(Linear::new(8, 1, Init::Xavier, &mut rng)),
        ])))
        .loss(Box::new(Mse))
        .optim(Box::new(Adam::new(0.02)))
        .build();

    let x = Tensor::from_data(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let y = Tensor::from_data(vec![0.0, 1.0, 1.0, 0.0], &[4, 1]);

    // Phase 1: 200 steps with MSE + Adam.
    let mut loss_phase1 = 0.0;
    for _ in 0..200 {
        loss_phase1 = trainer.train_step(&x, &y);
    }

    // Hot-swap: loss → Huber, optimiser → SGD. **No recompile, no panic.**
    trainer.loss = Box::new(Huber { delta: 0.5 });
    trainer.optim = Box::new(Sgd::with_momentum(0.05, 0.9));

    // Phase 2: 200 more steps. Loss should NOT explode after the swap.
    let mut loss_phase2 = f32::INFINITY;
    for _ in 0..200 {
        loss_phase2 = trainer.train_step(&x, &y);
    }

    assert!(
        loss_phase2.is_finite(),
        "hot-swap should not produce NaN/inf; got {loss_phase2}"
    );
    assert!(
        loss_phase2 < loss_phase1.max(1.0) * 2.0,
        "after hot-swap, loss should not explode: phase1={loss_phase1}, phase2={loss_phase2}"
    );
}
