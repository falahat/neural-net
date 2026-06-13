//! XOR — the canonical "a network with one hidden layer can learn a
//! non-linearly-separable function" test. If this fails, the library
//! is unusable for anything meaningful.

use nn::activation::Tanh;
use nn::init::Init;
use nn::loss::Mse;
use nn::module::{Linear, Sequential};
use nn::optim::Adam;
use nn::rng::SplitMix64;
use nn::tensor::Tensor;
use nn::train::Trainer;

#[test]
fn mlp_overfits_xor() {
    let mut rng = SplitMix64::seeded(42);

    let mut trainer = Trainer::builder()
        .model(Box::new(Sequential::new(vec![
            Box::new(Linear::new(2, 8, Init::Xavier, &mut rng)),
            Box::new(Tanh),
            Box::new(Linear::new(8, 1, Init::Xavier, &mut rng)),
        ])))
        .loss(Box::new(Mse))
        .optim(Box::new(Adam::new(0.05)))
        .build();

    // All four XOR rows, one batch.
    let x = Tensor::from_data(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let y = Tensor::from_data(vec![0.0, 1.0, 1.0, 0.0], &[4, 1]);

    let mut final_loss = f32::INFINITY;
    for _ in 0..2000 {
        final_loss = trainer.train_step(&x, &y);
        if final_loss < 1e-4 {
            break;
        }
    }
    assert!(
        final_loss < 1e-2,
        "XOR should converge: loss = {final_loss} after 2000 steps"
    );

    let pred = trainer.predict(&x);
    let p = pred.data();
    // Sigmoid-ish thresholding for classification.
    let yd = y.data();
    for (i, (&pi, &expected)) in p.iter().zip(yd).enumerate() {
        let got = pi.round().clamp(0.0, 1.0);
        assert_eq!(
            got, expected,
            "XOR row {i}: predicted {pi} -> {got}, expected {expected}"
        );
    }
}
