use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{Result, SymError};
use crate::tensor::{argmax, softmax, Tensor};

#[derive(Debug, Clone)]
pub struct TrainingTrace {
    pub loss: f32,
    pub accuracy: f32,
    pub grad_norm: f32,
}

#[derive(Debug, Clone)]
pub struct LinearReadout {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LinearReadout {
    pub fn new(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let normal = StandardNormal;
        let scale = (1.0 / input_dim.max(1) as f32).sqrt();
        let weights = (0..input_dim * output_dim)
            .map(|_| {
                let z: f32 = normal.sample(rng);
                z * scale
            })
            .collect();
        Self {
            input_dim,
            output_dim,
            weights,
            bias: vec![0.0; output_dim],
        }
    }

    pub fn zeros(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            weights: vec![0.0; input_dim * output_dim],
            bias: vec![0.0; output_dim],
        }
    }

    pub fn logits(&self, features: &Tensor) -> Result<Tensor> {
        features.ensure_cols(self.input_dim, "LinearReadout features")?;
        let mut out = features.matmul_right_transposed(&self.weights, self.output_dim)?;
        out.add_row_bias(&self.bias)?;
        Ok(out)
    }

    pub fn predict(&self, features: &Tensor) -> Result<usize> {
        let logits = self.logits(features)?;
        Ok(argmax(logits.row(0)))
    }

    pub fn train_step(
        &mut self,
        features: &Tensor,
        target: usize,
        lr: f32,
    ) -> Result<TrainingTrace> {
        if target >= self.output_dim {
            return Err(SymError::InvalidArgument(format!(
                "target {target} outside output dim {}",
                self.output_dim
            )));
        }
        let logits = self.logits(features)?;
        let probs = softmax(logits.row(0));
        let loss = -probs[target].max(1e-8).ln();
        let pred = argmax(&probs);
        let mut grad_norm = 0.0;
        for (o, prob) in probs.iter().copied().enumerate() {
            let delta = prob - if o == target { 1.0 } else { 0.0 };
            grad_norm += delta * delta;
            self.bias[o] -= lr * delta;
            for i in 0..self.input_dim {
                let grad = delta * features.get(0, i);
                grad_norm += grad * grad;
                self.weights[o * self.input_dim + i] -= lr * grad;
            }
        }
        Ok(TrainingTrace {
            loss,
            accuracy: if pred == target { 1.0 } else { 0.0 },
            grad_norm: grad_norm.sqrt(),
        })
    }

    pub fn train_batch(
        &mut self,
        features: &Tensor,
        targets: &[usize],
        lr: f32,
    ) -> Result<TrainingTrace> {
        features.ensure_cols(self.input_dim, "LinearReadout batch features")?;
        if targets.len() != features.rows {
            return Err(SymError::Shape(format!(
                "target batch expected {}, got {}",
                features.rows,
                targets.len()
            )));
        }
        if let Some(target) = targets
            .iter()
            .copied()
            .find(|target| *target >= self.output_dim)
        {
            return Err(SymError::InvalidArgument(format!(
                "target {target} outside output dim {}",
                self.output_dim
            )));
        }

        let logits = self.logits(features)?;
        let mut grad_w = vec![0.0; self.weights.len()];
        let mut grad_b = vec![0.0; self.bias.len()];
        let mut loss = 0.0;
        let mut correct = 0usize;

        for (row, target) in targets.iter().copied().enumerate() {
            let probs = softmax(logits.row(row));
            loss += -probs[target].max(1e-8).ln();
            correct += usize::from(argmax(&probs) == target);
            for (o, prob) in probs.iter().copied().enumerate() {
                let delta = prob - if o == target { 1.0 } else { 0.0 };
                grad_b[o] += delta;
                for i in 0..self.input_dim {
                    grad_w[o * self.input_dim + i] += delta * features.get(row, i);
                }
            }
        }

        let scale = 1.0 / features.rows.max(1) as f32;
        let mut grad_norm = 0.0;
        for (weight, grad) in self.weights.iter_mut().zip(grad_w) {
            let grad = grad * scale;
            grad_norm += grad * grad;
            *weight -= lr * grad;
        }
        for (bias, grad) in self.bias.iter_mut().zip(grad_b) {
            let grad = grad * scale;
            grad_norm += grad * grad;
            *bias -= lr * grad;
        }

        Ok(TrainingTrace {
            loss: loss * scale,
            accuracy: correct as f32 * scale,
            grad_norm: grad_norm.sqrt(),
        })
    }

    pub fn train_batch_target_bags(
        &mut self,
        features: &Tensor,
        target_bags: &[usize],
        targets_per_sample: usize,
        lr: f32,
    ) -> Result<TrainingTrace> {
        features.ensure_cols(self.input_dim, "LinearReadout bag features")?;
        if targets_per_sample == 0 {
            return Err(SymError::InvalidArgument(
                "targets_per_sample must be greater than zero".to_string(),
            ));
        }
        if target_bags.len() != features.rows * targets_per_sample {
            return Err(SymError::Shape(format!(
                "target bag batch expected {}, got {}",
                features.rows * targets_per_sample,
                target_bags.len()
            )));
        }
        if let Some(target) = target_bags
            .iter()
            .copied()
            .find(|target| *target >= self.output_dim)
        {
            return Err(SymError::InvalidArgument(format!(
                "target {target} outside output dim {}",
                self.output_dim
            )));
        }

        let logits = self.logits(features)?;
        let mut grad_w = vec![0.0; self.weights.len()];
        let mut grad_b = vec![0.0; self.bias.len()];
        let mut loss = 0.0;
        let mut correct = 0usize;
        let inv_targets = 1.0 / targets_per_sample as f32;

        for row in 0..features.rows {
            let probs = softmax(logits.row(row));
            let bag = &target_bags[row * targets_per_sample..(row + 1) * targets_per_sample];
            let pred = argmax(&probs);
            correct += usize::from(bag.contains(&pred));

            let mut target_counts = vec![0.0; self.output_dim];
            for target in bag {
                target_counts[*target] += inv_targets;
                loss += -probs[*target].max(1e-8).ln() * inv_targets;
            }

            for (o, prob) in probs.iter().copied().enumerate() {
                let delta = prob - target_counts[o];
                grad_b[o] += delta;
                for i in 0..self.input_dim {
                    grad_w[o * self.input_dim + i] += delta * features.get(row, i);
                }
            }
        }

        let scale = 1.0 / features.rows.max(1) as f32;
        let mut grad_norm = 0.0;
        for (weight, grad) in self.weights.iter_mut().zip(grad_w) {
            let grad = grad * scale;
            grad_norm += grad * grad;
            *weight -= lr * grad;
        }
        for (bias, grad) in self.bias.iter_mut().zip(grad_b) {
            let grad = grad * scale;
            grad_norm += grad * grad;
            *bias -= lr * grad;
        }

        Ok(TrainingTrace {
            loss: loss * scale,
            accuracy: correct as f32 * scale,
            grad_norm: grad_norm.sqrt(),
        })
    }

    pub fn evaluate_batch(&self, features: &Tensor, targets: &[usize]) -> Result<TrainingTrace> {
        features.ensure_cols(self.input_dim, "LinearReadout eval features")?;
        if targets.len() != features.rows {
            return Err(SymError::Shape(format!(
                "target batch expected {}, got {}",
                features.rows,
                targets.len()
            )));
        }
        if let Some(target) = targets
            .iter()
            .copied()
            .find(|target| *target >= self.output_dim)
        {
            return Err(SymError::InvalidArgument(format!(
                "target {target} outside output dim {}",
                self.output_dim
            )));
        }

        let logits = self.logits(features)?;
        let mut loss = 0.0;
        let mut correct = 0usize;
        for (row, target) in targets.iter().copied().enumerate() {
            let probs = softmax(logits.row(row));
            loss += -probs[target].max(1e-8).ln();
            correct += usize::from(argmax(&probs) == target);
        }
        let scale = 1.0 / targets.len().max(1) as f32;
        Ok(TrainingTrace {
            loss: loss * scale,
            accuracy: correct as f32 * scale,
            grad_norm: 0.0,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}
