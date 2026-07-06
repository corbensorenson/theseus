use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{Result, SymError};
use crate::tensor::{softplus, Tensor};

#[derive(Debug, Clone)]
pub struct LiquidCell {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub dt: f32,
    pub steps: usize,
    pub eps: f32,
    candidate_w: Vec<f32>,
    candidate_b: Vec<f32>,
    tau_w: Vec<f32>,
    tau_b: Vec<f32>,
}

impl LiquidCell {
    pub fn new(input_dim: usize, hidden_dim: usize, rng: &mut impl Rng) -> Result<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(SymError::InvalidArgument(
                "LiquidCell requires nonzero input and hidden dims".to_string(),
            ));
        }
        let concat = input_dim + hidden_dim;
        let scale = (1.0 / concat as f32).sqrt();
        let normal = StandardNormal;
        let mut sample = || {
            let z: f32 = normal.sample(rng);
            z * scale
        };
        Ok(Self {
            input_dim,
            hidden_dim,
            dt: 0.1,
            steps: 1,
            eps: 1e-3,
            candidate_w: (0..hidden_dim * concat).map(|_| sample()).collect(),
            candidate_b: vec![0.0; hidden_dim],
            tau_w: (0..hidden_dim * concat).map(|_| sample()).collect(),
            tau_b: vec![0.5; hidden_dim],
        })
    }

    pub fn forward(&self, x: &Tensor, h_prev: &Tensor) -> Result<Tensor> {
        x.ensure_cols(self.input_dim, "LiquidCell input")?;
        h_prev.ensure_cols(self.hidden_dim, "LiquidCell hidden state")?;
        if x.rows != h_prev.rows {
            return Err(SymError::Shape(format!(
                "LiquidCell batch mismatch: input rows {}, hidden rows {}",
                x.rows, h_prev.rows
            )));
        }
        let mut h = h_prev.clone();
        let concat_dim = self.input_dim + self.hidden_dim;
        let steps = self.steps.max(1);
        for _ in 0..steps {
            let mut next = Tensor::zeros(x.rows, self.hidden_dim);
            for b in 0..x.rows {
                for j in 0..self.hidden_dim {
                    let mut cand_pre = self.candidate_b[j];
                    let mut tau_pre = self.tau_b[j];
                    for i in 0..self.input_dim {
                        let value = x.get(b, i);
                        cand_pre += self.candidate_w[j * concat_dim + i] * value;
                        tau_pre += self.tau_w[j * concat_dim + i] * value;
                    }
                    for i in 0..self.hidden_dim {
                        let value = h.get(b, i);
                        let offset = self.input_dim + i;
                        cand_pre += self.candidate_w[j * concat_dim + offset] * value;
                        tau_pre += self.tau_w[j * concat_dim + offset] * value;
                    }
                    let candidate = cand_pre.tanh();
                    let tau = softplus(tau_pre) + self.eps;
                    let dh = (-h.get(b, j) + candidate) / tau;
                    next.set(b, j, h.get(b, j) + self.dt * dh);
                }
            }
            h = next;
        }
        Ok(h)
    }

    pub fn parameter_count(&self) -> usize {
        self.candidate_w.len() + self.candidate_b.len() + self.tau_w.len() + self.tau_b.len()
    }
}
