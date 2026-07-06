use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{Result, SymError};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Reservoir {
    pub hidden_dim: usize,
    pub reservoir_dim: usize,
    pub alpha: f32,
    pub spectral_radius_target: f32,
    pub sparsity: f32,
    pub trainable: bool,
    pub recurrent: Vec<f32>,
    pub input: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Reservoir {
    pub fn new(
        hidden_dim: usize,
        reservoir_dim: usize,
        spectral_radius: f32,
        sparsity: f32,
        trainable: bool,
        rng: &mut impl Rng,
    ) -> Result<Self> {
        if hidden_dim == 0 || reservoir_dim == 0 {
            return Err(SymError::InvalidArgument(
                "Reservoir requires nonzero dimensions".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&sparsity) {
            return Err(SymError::InvalidArgument(
                "reservoir sparsity must be in [0, 1)".to_string(),
            ));
        }
        let normal = StandardNormal;
        let mut recurrent = vec![0.0; reservoir_dim * reservoir_dim];
        for value in &mut recurrent {
            if rng.gen::<f32>() >= sparsity {
                let z: f32 = normal.sample(rng);
                *value = z / (reservoir_dim as f32).sqrt();
            }
        }
        let radius = estimate_spectral_radius(&recurrent, reservoir_dim, 48).max(1e-6);
        let scale = spectral_radius / radius;
        for value in &mut recurrent {
            *value *= scale;
        }
        let input = (0..reservoir_dim * hidden_dim)
            .map(|_| {
                let z: f32 = normal.sample(rng);
                z / (hidden_dim as f32).sqrt()
            })
            .collect();
        Ok(Self {
            hidden_dim,
            reservoir_dim,
            alpha: 0.25,
            spectral_radius_target: spectral_radius,
            sparsity,
            trainable,
            recurrent,
            input,
            bias: vec![0.0; reservoir_dim],
        })
    }

    pub fn forward(&self, h: &Tensor, r_prev: &Tensor) -> Result<Tensor> {
        h.ensure_cols(self.hidden_dim, "Reservoir input")?;
        r_prev.ensure_cols(self.reservoir_dim, "Reservoir state")?;
        if h.rows != r_prev.rows {
            return Err(SymError::Shape(format!(
                "Reservoir batch mismatch: input rows {}, state rows {}",
                h.rows, r_prev.rows
            )));
        }
        let mut out = Tensor::zeros(h.rows, self.reservoir_dim);
        for b in 0..h.rows {
            for j in 0..self.reservoir_dim {
                let mut pre = self.bias[j];
                for k in 0..self.reservoir_dim {
                    pre += self.recurrent[j * self.reservoir_dim + k] * r_prev.get(b, k);
                }
                for i in 0..self.hidden_dim {
                    pre += self.input[j * self.hidden_dim + i] * h.get(b, i);
                }
                let updated = (1.0 - self.alpha) * r_prev.get(b, j) + self.alpha * pre.tanh();
                out.set(b, j, updated);
            }
        }
        Ok(out)
    }

    pub fn spectral_radius_estimate(&self) -> f32 {
        estimate_spectral_radius(&self.recurrent, self.reservoir_dim, 64)
    }

    pub fn parameter_count(&self) -> usize {
        if self.trainable {
            self.recurrent.len() + self.input.len() + self.bias.len()
        } else {
            self.input.len() + self.bias.len()
        }
    }
}

pub fn estimate_spectral_radius(matrix: &[f32], dim: usize, iterations: usize) -> f32 {
    if dim == 0 || matrix.len() != dim * dim {
        return 0.0;
    }
    let mut v = vec![1.0 / (dim as f32).sqrt(); dim];
    let mut radius = 0.0;
    for _ in 0..iterations.max(1) {
        let mut next = vec![0.0; dim];
        for row in 0..dim {
            for col in 0..dim {
                next[row] += matrix[row * dim + col] * v[col];
            }
        }
        let norm = next.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        radius = norm;
        for i in 0..dim {
            v[i] = next[i] / norm;
        }
    }
    radius
}
