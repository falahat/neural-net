//! Central-difference gradient checks for every Op variant.
//!
//! For each forward function `f(x)` returning a scalar, we verify:
//! ```text
//!     ∂f/∂xᵢ ≈ (f(x + ε·eᵢ) − f(x − ε·eᵢ)) / (2ε)
//! ```
//! to relative tolerance ~1e-3. This is the single test that prevents
//! "loss goes down but means nothing" silent-correctness bugs across
//! the autograd implementation. See design doc §14.1.

use nn::autograd::Tape;
use nn::tensor::Tensor;

const EPS: f32 = 1e-3;
const RTOL: f32 = 1e-2;
const ATOL: f32 = 1e-3;

/// Compare analytic gradient (from backward) to a central-difference
/// estimate. `forward` builds the graph and returns the scalar loss.
fn check_grad<F>(x: Vec<f32>, shape: Vec<usize>, mut forward: F)
where
    F: FnMut(&mut Tape, &Tensor) -> Tensor,
{
    // ── 1. Analytic gradient via backward.
    let mut tape = Tape::new();
    let xt = Tensor::from_data(x.clone(), &shape);
    let xnode = tape.leaf(&xt, Some(nn::autograd::ParamId::new()));
    let loss = forward(&mut tape, &xnode);
    let grads = tape.backward(&loss);
    // This test builds a single leaf, so the first (only) grad id is x's.
    let analytic = grads.ids().next().expect("no grads emitted");
    let (g, _) = grads.get(analytic).unwrap();
    let g: Vec<f32> = g.to_vec();

    // ── 2. Numerical gradient via central differences.
    let mut num = vec![0.0; x.len()];
    for i in 0..x.len() {
        let mut xp = x.clone();
        xp[i] += EPS;
        let mut xm = x.clone();
        xm[i] -= EPS;
        let lp = {
            let mut tape = Tape::new();
            let xt = Tensor::from_data(xp, &shape);
            let xn = tape.leaf(&xt, None);
            forward(&mut tape, &xn).item().unwrap()
        };
        let lm = {
            let mut tape = Tape::new();
            let xt = Tensor::from_data(xm, &shape);
            let xn = tape.leaf(&xt, None);
            forward(&mut tape, &xn).item().unwrap()
        };
        num[i] = (lp - lm) / (2.0 * EPS);
    }

    // ── 3. Compare.
    for i in 0..x.len() {
        let err = (g[i] - num[i]).abs();
        let denom = num[i].abs().max(g[i].abs()).max(1e-6);
        let rel = err / denom;
        assert!(
            err < ATOL || rel < RTOL,
            "grad mismatch at index {i}: analytic={a:.6}, numeric={n:.6}, err={err:.6}, rel={rel:.6}",
            a = g[i], n = num[i],
        );
    }
}

#[test]
fn grad_add_self() {
    // f(x) = sum(x + x) = 2 sum(x); df/dx = 2.
    check_grad(vec![0.5, -1.2, 3.0], vec![3], |tape, x| {
        let y = tape.add(x, x);
        tape.sum_all(&y)
    });
}

#[test]
fn grad_sub() {
    // f(x) = sum(x - c) for fixed c; gradient = 1.
    check_grad(vec![0.7, -0.3, 1.1], vec![3], |tape, x| {
        let c = Tensor::from_data(vec![0.1, 0.2, 0.3], &[3]);
        let y = tape.sub(x, &c);
        tape.sum_all(&y)
    });
}

#[test]
fn grad_mul_const() {
    check_grad(vec![1.5, -0.8, 2.0], vec![3], |tape, x| {
        let c = Tensor::from_data(vec![0.5, -1.0, 2.0], &[3]);
        let y = tape.mul(x, &c);
        tape.sum_all(&y)
    });
}

#[test]
fn grad_div_const() {
    check_grad(vec![1.5, 2.5, 3.5], vec![3], |tape, x| {
        let c = Tensor::from_data(vec![0.5, 1.0, 2.0], &[3]);
        let y = tape.div(x, &c);
        tape.sum_all(&y)
    });
}

#[test]
fn grad_neg_scale() {
    check_grad(vec![1.0, -2.0, 3.0], vec![3], |tape, x| {
        let y = tape.neg(x);
        let z = tape.scale(&y, 0.7);
        tape.sum_all(&z)
    });
}

#[test]
fn grad_relu_positive_and_negative() {
    check_grad(vec![0.5, -0.5, 1.2, -0.1], vec![4], |tape, x| {
        let y = tape.relu(x);
        tape.sum_all(&y)
    });
}

#[test]
fn grad_sigmoid() {
    check_grad(vec![0.1, -0.5, 1.2, -2.0], vec![4], |tape, x| {
        let y = tape.sigmoid(x);
        tape.sum_all(&y)
    });
}

#[test]
fn grad_tanh() {
    check_grad(vec![0.0, 0.5, -1.2, 2.0], vec![4], |tape, x| {
        let y = tape.tanh(x);
        tape.sum_all(&y)
    });
}

#[test]
fn grad_exp_log() {
    // f(x) = sum(log(exp(x))) = sum(x); gradient = 1.
    check_grad(vec![0.3, -0.7, 1.1], vec![3], |tape, x| {
        let e = tape.exp(x);
        let l = tape.log(&e);
        tape.sum_all(&l)
    });
}

#[test]
fn grad_square() {
    check_grad(vec![0.5, -1.0, 2.0], vec![3], |tape, x| {
        let y = tape.square(x);
        tape.sum_all(&y)
    });
}

#[test]
fn grad_matmul_input() {
    // f(x) = sum(W @ x), x is [2,3], W is [4,2] constant. Result [4,3].
    check_grad(
        vec![0.5, 1.0, -1.0, 2.0, 0.1, 0.7],
        vec![2, 3],
        |tape, x| {
            let w = Tensor::from_data(vec![1.0, 2.0, -1.0, 0.5, 0.3, -0.7, 1.1, 0.2], &[4, 2]);
            let y = tape.matmul(&w, x); // [4,3]
            tape.sum_all(&y)
        },
    );
}

#[test]
fn grad_transpose() {
    check_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], |tape, x| {
        let t = tape.transpose(x);
        tape.sum_all(&t)
    });
}

#[test]
fn grad_mean() {
    check_grad(vec![1.0, 2.0, 3.0, 4.0], vec![4], |tape, x| {
        tape.mean_all(x)
    });
}

#[test]
fn grad_broadcast() {
    // f(b) = sum(broadcast_row(b, [3, 2]))
    check_grad(vec![1.0, -0.5], vec![2], |tape, x| {
        let bc = tape.broadcast_row(x, &[3, 2]);
        tape.sum_all(&bc)
    });
}

#[test]
fn grad_chain_mse() {
    // f(x) = mean((x - y)^2). dx = 2/(n) * (x - y).
    check_grad(vec![1.0, 0.5, -0.3, 2.1], vec![4], |tape, x| {
        let y = Tensor::from_data(vec![0.2, 0.0, 0.1, 1.0], &[4]);
        let d = tape.sub(x, &y);
        let s = tape.square(&d);
        tape.mean_all(&s)
    });
}

#[test]
fn grad_softmax_xe() {
    // df/d(logits) = (softmax - one_hot)/batch.
    check_grad(
        vec![1.0, 2.0, -0.5, 0.3, 0.1, 2.0],
        vec![2, 3],
        |tape, x| {
            let targets = Tensor::from_data(vec![0.0, 2.0], &[2]);
            tape.softmax_xe(x, &targets)
        },
    );
}
