//! Linear regression — `y = 2x + 3 + noise`. A single Linear layer
//! with MSE + SGD should recover (2, 3) within tolerance.

use nn::init::Init;
use nn::loss::Mse;
use nn::module::Linear;
use nn::optim::Sgd;
use nn::rng::{Rng, SplitMix64};
use nn::tensor::Tensor;
use nn::train::Trainer;

#[test]
fn fit_linear() {
    let mut rng = SplitMix64::seeded(7);

    // Generate 256 points y = 2x + 3.
    let mut xs = Vec::with_capacity(256);
    let mut ys = Vec::with_capacity(256);
    for _ in 0..256 {
        let x = rng.normal(0.0, 1.0);
        xs.push(x);
        ys.push(2.0 * x + 3.0);
    }
    let x = Tensor::from_data(xs, &[256, 1]);
    let y = Tensor::from_data(ys, &[256, 1]);

    let mut trainer = Trainer::builder()
        .model(Box::new(Linear::new(1, 1, Init::Xavier, &mut rng)))
        .loss(Box::new(Mse))
        .optim(Box::new(Sgd::new(0.05)))
        .build();

    let mut last = f32::INFINITY;
    for _ in 0..500 {
        last = trainer.train_step(&x, &y);
    }
    assert!(last < 1e-3, "linreg should converge: loss = {last}");

    // Inspect recovered (w, b).
    use nn::autograd::ParamId;
    use nn::tensor::Tensor as T;
    let mut w_val: f32 = 0.0;
    let mut b_val: f32 = 0.0;
    trainer
        .model
        .visit_params(&mut |path: &str, t: &T, _id: ParamId| {
            if path.contains("weight") {
                w_val = t.data()[0];
            }
            if path.contains("bias") {
                b_val = t.data()[0];
            }
        });
    assert!((w_val - 2.0).abs() < 0.05, "w should be ~2.0, got {w_val}");
    assert!((b_val - 3.0).abs() < 0.05, "b should be ~3.0, got {b_val}");
}
