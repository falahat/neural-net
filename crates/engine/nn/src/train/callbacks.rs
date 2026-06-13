//! Bundled callbacks: enough to drive a textbook widget without
//! writing a custom impl. Users can drop in their own by impl-ing
//! `Callback`.

use super::{Callback, EpochStats, TrainCtx};

/// Collect every batch loss into a `Vec`. Cheap; ~8 bytes per step.
#[derive(Debug, Default)]
pub struct LossLogger {
    pub losses: Vec<f32>,
}
impl Callback for LossLogger {
    fn on_loss_computed(&mut self, _ctx: &mut TrainCtx, loss: f32) {
        self.losses.push(loss);
    }
}

/// Print one line per epoch. Useful for sanity checks; widgets won't
/// usually want this.
#[derive(Debug, Default)]
pub struct PrintEpochLoss;
impl Callback for PrintEpochLoss {
    fn on_epoch_end(&mut self, ctx: &mut TrainCtx, stats: &EpochStats) {
        println!("epoch {:>4} | mean loss {:.6}", ctx.epoch, stats.mean_loss);
    }
}

/// Early-stopping by absolute loss threshold. When the *mean* epoch
/// loss drops below `threshold`, callers can read `triggered`.
#[derive(Debug)]
pub struct EarlyStopAtLoss {
    pub threshold: f32,
    pub triggered: bool,
}
impl EarlyStopAtLoss {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            triggered: false,
        }
    }
}
impl Callback for EarlyStopAtLoss {
    fn on_epoch_end(&mut self, _ctx: &mut TrainCtx, stats: &EpochStats) {
        if stats.mean_loss < self.threshold {
            self.triggered = true;
        }
    }
}
