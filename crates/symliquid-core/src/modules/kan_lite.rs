use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{Result, SymError};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct KANLiteLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub num_basis: usize,
    pub centers: Vec<f32>,
    pub sigma: f32,
    pub coeffs: Vec<f32>,
    pub linear: Vec<f32>,
    pub bias: Vec<f32>,
    pub residual_linear: bool,
}

impl KANLiteLayer {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        num_basis: usize,
        rng: &mut impl Rng,
    ) -> Result<Self> {
        if in_dim == 0 || out_dim == 0 || num_basis < 2 {
            return Err(SymError::InvalidArgument(
                "KANLiteLayer requires nonzero dims and at least two basis functions".to_string(),
            ));
        }
        let denom = (num_basis - 1) as f32;
        let centers = (0..num_basis)
            .map(|i| -1.0 + 2.0 * i as f32 / denom)
            .collect::<Vec<_>>();
        let sigma = (2.0 / denom).max(0.1);
        let normal = StandardNormal;
        let coeffs = (0..out_dim * in_dim * num_basis)
            .map(|_| {
                let z: f32 = normal.sample(rng);
                z * 0.04
            })
            .collect();
        let linear = (0..out_dim * in_dim)
            .map(|_| {
                let z: f32 = normal.sample(rng);
                z * (1.0 / in_dim as f32).sqrt() * 0.2
            })
            .collect();
        Ok(Self {
            in_dim,
            out_dim,
            num_basis,
            centers,
            sigma,
            coeffs,
            linear,
            bias: vec![0.0; out_dim],
            residual_linear: true,
        })
    }

    pub fn basis_values(&self, x: f32) -> Vec<f32> {
        let denom = 2.0 * self.sigma * self.sigma;
        self.centers
            .iter()
            .map(|center| {
                let d = x - *center;
                (-(d * d) / denom).exp()
            })
            .collect()
    }

    pub fn edge_value(&self, input_idx: usize, output_idx: usize, x: f32) -> Result<f32> {
        if input_idx >= self.in_dim || output_idx >= self.out_dim {
            return Err(SymError::InvalidArgument(format!(
                "edge ({input_idx}, {output_idx}) is outside KAN shape [{}, {}]",
                self.in_dim, self.out_dim
            )));
        }
        let basis = self.basis_values(x);
        let mut acc = 0.0;
        let base = (output_idx * self.in_dim + input_idx) * self.num_basis;
        for (k, phi) in basis.iter().enumerate() {
            acc += self.coeffs[base + k] * *phi;
        }
        Ok(acc)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.ensure_cols(self.in_dim, "KANLiteLayer input")?;
        let mut out = Tensor::zeros(x.rows, self.out_dim);
        for b in 0..x.rows {
            for o in 0..self.out_dim {
                let mut acc = self.bias[o];
                for i in 0..self.in_dim {
                    let value = x.get(b, i);
                    let basis = self.basis_values(value);
                    let coeff_base = (o * self.in_dim + i) * self.num_basis;
                    for (k, phi) in basis.iter().enumerate() {
                        acc += self.coeffs[coeff_base + k] * *phi;
                    }
                    if self.residual_linear {
                        acc += self.linear[o * self.in_dim + i] * value;
                    }
                }
                out.set(b, o, acc);
            }
        }
        Ok(out)
    }

    pub fn regularization(&self, l1_weight: f32, smooth_weight: f32) -> f32 {
        let l1 = self.coeffs.iter().map(|v| v.abs()).sum::<f32>() * l1_weight;
        let mut smooth = 0.0;
        if self.num_basis >= 3 {
            for edge in 0..self.out_dim * self.in_dim {
                let base = edge * self.num_basis;
                for k in 1..self.num_basis - 1 {
                    let second = self.coeffs[base + k - 1] - 2.0 * self.coeffs[base + k]
                        + self.coeffs[base + k + 1];
                    smooth += second * second;
                }
            }
        }
        l1 + smooth_weight * smooth
    }

    pub fn parameter_count(&self) -> usize {
        self.coeffs.len() + self.linear.len() + self.bias.len()
    }
}
