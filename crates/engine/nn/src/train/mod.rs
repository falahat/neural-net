//! The `Trainer` — glue around model + loss + optimiser + callbacks.
//!
//! Every field is `Box<dyn …>` so all four can be hot-swapped between
//! training steps.
//!
//! ## Hot-swap example
//!
//! ```ignore
//! let mut trainer = Trainer::builder()
//!     .model(Box::new(my_mlp))
//!     .loss(Box::new(Mse))
//!     .optim(Box::new(Adam::new(0.01)))
//!     .build();
//!
//! for epoch in 0..1000 {
//!     trainer.train_step(&x, &y);
//!     if epoch == 500 { trainer.loss  = Box::new(Huber { delta: 1.0 }); }
//!     if epoch == 750 { trainer.optim = Box::new(Sgd::new(0.05)); }
//! }
//! ```

pub mod callbacks;

use crate::autograd::Tape;
use crate::loss::Loss;
use crate::module::Module;
use crate::optim::Optimizer;
use crate::tensor::Tensor;

/// Shared mutable context passed to every callback method. Carries
/// the epoch + step counter; widget code reads these for x-axis values.
#[derive(Debug, Default)]
pub struct TrainCtx {
    pub epoch: u64,
    pub step: u64,
    pub lr: f32,
}

/// Statistics rolled up at epoch boundaries.
#[derive(Debug, Default, Clone)]
pub struct EpochStats {
    pub mean_loss: f32,
    pub n_steps: u64,
}

pub trait Callback: Send {
    fn on_epoch_start(&mut self, _ctx: &mut TrainCtx) {}
    fn on_batch_start(&mut self, _ctx: &mut TrainCtx) {}
    fn on_loss_computed(&mut self, _ctx: &mut TrainCtx, _loss: f32) {}
    fn on_step_done(&mut self, _ctx: &mut TrainCtx) {}
    fn on_batch_end(&mut self, _ctx: &mut TrainCtx) {}
    fn on_epoch_end(&mut self, _ctx: &mut TrainCtx, _stats: &EpochStats) {}
}

pub struct Trainer {
    pub model: Box<dyn Module>,
    pub loss: Box<dyn Loss>,
    pub optim: Box<dyn Optimizer>,
    pub callbacks: Vec<Box<dyn Callback>>,
    pub ctx: TrainCtx,
    /// The tape is reused across steps (cleared each step) to avoid
    /// reallocating the op arena every batch.
    tape: Tape,
}

impl Trainer {
    pub fn builder() -> TrainerBuilder {
        TrainerBuilder::default()
    }

    /// One forward + backward + step + callback fanout. Returns the
    /// scalar loss for plotting / logging.
    pub fn train_step(&mut self, x: &Tensor, y: &Tensor) -> f32 {
        self.ctx.step += 1;
        self.ctx.lr = self.optim.lr();
        for cb in self.callbacks.iter_mut() {
            cb.on_batch_start(&mut self.ctx);
        }

        self.tape.clear();
        let pred = self.model.forward(&mut self.tape, x);
        let loss = self.loss.forward(&mut self.tape, &pred, y);
        let loss_value = loss.item().expect("Trainer: loss tensor must be scalar");
        for cb in self.callbacks.iter_mut() {
            cb.on_loss_computed(&mut self.ctx, loss_value);
        }

        let grads = self.tape.backward(&loss);
        self.optim.step(&mut *self.model, &grads);
        for cb in self.callbacks.iter_mut() {
            cb.on_step_done(&mut self.ctx);
        }
        for cb in self.callbacks.iter_mut() {
            cb.on_batch_end(&mut self.ctx);
        }

        loss_value
    }

    /// Run one full pass over `(xs, ys)`. Each `(x, y)` pair is one
    /// batch — the user pre-batches the data.
    pub fn epoch(&mut self, batches: &[(Tensor, Tensor)]) -> EpochStats {
        self.ctx.epoch += 1;
        for cb in self.callbacks.iter_mut() {
            cb.on_epoch_start(&mut self.ctx);
        }
        let mut total = 0.0;
        for (x, y) in batches {
            total += self.train_step(x, y);
        }
        let stats = EpochStats {
            mean_loss: total / batches.len() as f32,
            n_steps: batches.len() as u64,
        };
        for cb in self.callbacks.iter_mut() {
            cb.on_epoch_end(&mut self.ctx, &stats);
        }
        stats
    }

    /// Forward-only — for inference / validation. Doesn't touch the
    /// optimiser. Tape is cleared internally; gradients aren't computed.
    pub fn predict(&mut self, x: &Tensor) -> Tensor {
        self.tape.clear();
        self.model.forward(&mut self.tape, x)
    }
}

#[derive(Default)]
pub struct TrainerBuilder {
    model: Option<Box<dyn Module>>,
    loss: Option<Box<dyn Loss>>,
    optim: Option<Box<dyn Optimizer>>,
    callbacks: Vec<Box<dyn Callback>>,
}

impl TrainerBuilder {
    pub fn model(mut self, m: Box<dyn Module>) -> Self {
        self.model = Some(m);
        self
    }
    pub fn loss(mut self, l: Box<dyn Loss>) -> Self {
        self.loss = Some(l);
        self
    }
    pub fn optim(mut self, o: Box<dyn Optimizer>) -> Self {
        self.optim = Some(o);
        self
    }
    pub fn callback(mut self, c: Box<dyn Callback>) -> Self {
        self.callbacks.push(c);
        self
    }
    pub fn callbacks(mut self, cs: Vec<Box<dyn Callback>>) -> Self {
        self.callbacks = cs;
        self
    }

    pub fn build(self) -> Trainer {
        Trainer {
            model: self.model.expect("TrainerBuilder: model() is required"),
            loss: self.loss.expect("TrainerBuilder: loss() is required"),
            optim: self.optim.expect("TrainerBuilder: optim() is required"),
            callbacks: self.callbacks,
            ctx: TrainCtx::default(),
            tape: Tape::new(),
        }
    }
}
