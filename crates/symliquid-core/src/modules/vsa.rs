use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{Result, SymError};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct VSAMemory {
    pub input_dim: usize,
    pub hv_dim: usize,
    pub projection: Vec<f32>,
}

impl VSAMemory {
    pub fn new(input_dim: usize, hv_dim: usize, rng: &mut impl Rng) -> Result<Self> {
        if input_dim == 0 || hv_dim == 0 {
            return Err(SymError::InvalidArgument(
                "VSAMemory requires nonzero dimensions".to_string(),
            ));
        }
        let normal = StandardNormal;
        let scale = 1.0 / (input_dim as f32).sqrt();
        let projection = (0..hv_dim * input_dim)
            .map(|_| {
                let z: f32 = normal.sample(rng);
                z * scale
            })
            .collect();
        Ok(Self {
            input_dim,
            hv_dim,
            projection,
        })
    }

    pub fn project(&self, x: &Tensor, hard: bool) -> Result<Tensor> {
        x.ensure_cols(self.input_dim, "VSAMemory projection input")?;
        let mut out = Tensor::zeros(x.rows, self.hv_dim);
        for b in 0..x.rows {
            for j in 0..self.hv_dim {
                let mut pre = 0.0;
                for i in 0..self.input_dim {
                    pre += self.projection[j * self.input_dim + i] * x.get(b, i);
                }
                let value = if hard {
                    if pre >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    pre.tanh()
                };
                out.set(b, j, value);
            }
        }
        Ok(out)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        memory: Option<&Tensor>,
        decay: f32,
        hard: bool,
    ) -> Result<(Tensor, Tensor)> {
        let hv = self.project(x, hard)?;
        let memory = match memory {
            Some(prev) => Self::bundle(prev, &hv, decay)?,
            None => hv.clone(),
        };
        Ok((hv, memory))
    }

    pub fn bind(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.ensure_same_shape(b, "VSA bind")?;
        Ok(Tensor {
            rows: a.rows,
            cols: a.cols,
            data: a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect(),
        })
    }

    pub fn unbind(bound: &Tensor, key: &Tensor) -> Result<Tensor> {
        Self::bind(bound, key)
    }

    pub fn bundle(memory: &Tensor, x: &Tensor, decay: f32) -> Result<Tensor> {
        memory.ensure_same_shape(x, "VSA bundle")?;
        Ok(Tensor {
            rows: memory.rows,
            cols: memory.cols,
            data: memory
                .data
                .iter()
                .zip(&x.data)
                .map(|(m, v)| decay * *m + *v)
                .collect(),
        })
    }

    pub fn permute(x: &Tensor, shift: isize) -> Tensor {
        let mut out = Tensor::zeros(x.rows, x.cols);
        let n = x.cols as isize;
        for row in 0..x.rows {
            for col in 0..x.cols {
                let src = (col as isize - shift).rem_euclid(n) as usize;
                out.set(row, col, x.get(row, src));
            }
        }
        out
    }

    pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Vec<f32>> {
        a.ensure_same_shape(b, "VSA cosine similarity")?;
        let mut out = vec![0.0; a.rows];
        for (row, score) in out.iter_mut().enumerate() {
            let mut dot = 0.0;
            let mut an = 0.0;
            let mut bn = 0.0;
            for col in 0..a.cols {
                let av = a.get(row, col);
                let bv = b.get(row, col);
                dot += av * bv;
                an += av * av;
                bn += bv * bv;
            }
            *score = dot / (an.sqrt() * bn.sqrt()).max(1e-8);
        }
        Ok(out)
    }

    pub fn cleanup(query: &Tensor, symbol_table: &Tensor) -> Result<(usize, f32)> {
        query.ensure_cols(symbol_table.cols, "cleanup query")?;
        if query.rows != 1 {
            return Err(SymError::Shape(
                "cleanup currently expects a single query row".to_string(),
            ));
        }
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for row in 0..symbol_table.rows {
            let mut dot = 0.0;
            let mut qn = 0.0;
            let mut sn = 0.0;
            for col in 0..symbol_table.cols {
                let q = query.get(0, col);
                let s = symbol_table.get(row, col);
                dot += q * s;
                qn += q * q;
                sn += s * s;
            }
            let score = dot / (qn.sqrt() * sn.sqrt()).max(1e-8);
            if score > best_score {
                best = row;
                best_score = score;
            }
        }
        Ok((best, best_score))
    }

    pub fn random_bipolar(rows: usize, cols: usize, rng: &mut impl Rng) -> Tensor {
        let data = (0..rows * cols)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        Tensor { rows, cols, data }
    }

    pub fn consistency_loss(role: &Tensor, filler: &Tensor) -> Result<f32> {
        let bound = Self::bind(role, filler)?;
        let recovered = Self::unbind(&bound, role)?;
        recovered.ensure_same_shape(filler, "VSA consistency")?;
        let mse = recovered
            .data
            .iter()
            .zip(&filler.data)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            / recovered.len().max(1) as f32;
        Ok(mse)
    }

    pub fn parameter_count(&self) -> usize {
        self.projection.len()
    }
}
