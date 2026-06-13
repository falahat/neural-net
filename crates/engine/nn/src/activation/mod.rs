//! Activations. Each is a zero-sized `Module` whose forward records
//! one op on the tape. They double as `Module`s so they can sit
//! inside `Sequential` / tuples.

use crate::autograd::Tape;
use crate::module::Module;
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy, Default)]
pub struct Relu;
impl Module for Relu {
    fn forward(&self, tape: &mut Tape, x: &Tensor) -> Tensor {
        tape.relu(x)
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// `LeakyRelu` with a slope on the negative side. `slope = 0.01` is
/// Maas, Hannun & Ng's original choice (ICML 2013).
#[derive(Debug, Clone, Copy)]
pub struct LeakyRelu {
    pub slope: f32,
}
impl Default for LeakyRelu {
    fn default() -> Self {
        Self { slope: 0.01 }
    }
}
impl Module for LeakyRelu {
    fn forward(&self, tape: &mut Tape, x: &Tensor) -> Tensor {
        // ReLU(x) + slope * (-ReLU(-x))   ≡   max(x, slope*x).
        // Implemented via two relu-record ops so backward gets exact
        // subgradients per branch without a new Op variant.
        let pos = tape.relu(x);
        let neg_in = tape.neg(x);
        let neg = tape.relu(&neg_in);
        let neg_scaled = tape.scale(&neg, -self.slope);
        tape.add(&pos, &neg_scaled)
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, tape: &mut Tape, x: &Tensor) -> Tensor {
        tape.sigmoid(x)
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh;
impl Module for Tanh {
    fn forward(&self, tape: &mut Tape, x: &Tensor) -> Tensor {
        tape.tanh(x)
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// **GELU** (Hendrycks & Gimpel, "Gaussian Error Linear Units," 2016).
/// We use the tanh approximation
/// The tanh-approximation cubic coefficient (Hendrycks & Gimpel 2016).
const GELU_CUBIC_COEF: f32 = 0.044715;

/// `0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))` — same as PyTorch's
/// `nn.GELU(approximate="tanh")`. The exact erf form is also fine
/// numerically but requires an erf primitive we don't have yet.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gelu;
impl Module for Gelu {
    fn forward(&self, tape: &mut Tape, x: &Tensor) -> Tensor {
        // 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))
        const C: f32 = 0.797_884_6; // √(2/π)
        let x2 = tape.square(x);
        let x3 = tape.mul(&x2, x);
        let x3s = tape.scale(&x3, GELU_CUBIC_COEF);
        let inner = tape.add(x, &x3s);
        let inner = tape.scale(&inner, C);
        let t = tape.tanh(&inner);
        let one = Tensor::from_data(vec![1.0; t.numel()], t.shape());
        let one_p = tape.add(&t, &one);
        let half_x = tape.scale(x, 0.5);
        tape.mul(&half_x, &one_p)
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
