use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{Result, SymError};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if rows * cols != data.len() {
            return Err(SymError::Shape(format!(
                "expected {} values for shape [{rows}, {cols}], got {}",
                rows * cols,
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![1.0; rows * cols],
        }
    }

    pub fn random_normal(rows: usize, cols: usize, scale: f32, rng: &mut impl Rng) -> Self {
        let normal = StandardNormal;
        let data = (0..rows * cols)
            .map(|_| {
                let z: f32 = normal.sample(rng);
                z * scale
            })
            .collect();
        Self { rows, cols, data }
    }

    pub fn from_row(data: Vec<f32>) -> Self {
        Self {
            rows: 1,
            cols: data.len(),
            data,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    pub fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [f32] {
        let start = row * self.cols;
        &mut self.data[start..start + self.cols]
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().copied().map(f).collect(),
        }
    }

    pub fn ensure_cols(&self, expected: usize, label: &str) -> Result<()> {
        if self.cols != expected {
            return Err(SymError::Shape(format!(
                "{label} expected {} columns, got {}",
                expected, self.cols
            )));
        }
        Ok(())
    }

    pub fn ensure_same_shape(&self, other: &Self, label: &str) -> Result<()> {
        if self.shape() != other.shape() {
            return Err(SymError::Shape(format!(
                "{label} expected same shapes, got {:?} and {:?}",
                self.shape(),
                other.shape()
            )));
        }
        Ok(())
    }

    pub fn matmul_right_transposed(&self, weights: &[f32], out_dim: usize) -> Result<Self> {
        if weights.len() != out_dim * self.cols {
            return Err(SymError::Shape(format!(
                "weight matrix expected {} values for [{out_dim}, {}], got {}",
                out_dim * self.cols,
                self.cols,
                weights.len()
            )));
        }
        let mut out = Tensor::zeros(self.rows, out_dim);
        for b in 0..self.rows {
            for o in 0..out_dim {
                let mut acc = 0.0;
                for i in 0..self.cols {
                    acc += self.get(b, i) * weights[o * self.cols + i];
                }
                out.set(b, o, acc);
            }
        }
        Ok(out)
    }

    pub fn add_row_bias(&mut self, bias: &[f32]) -> Result<()> {
        if bias.len() != self.cols {
            return Err(SymError::Shape(format!(
                "bias expected {} values, got {}",
                self.cols,
                bias.len()
            )));
        }
        for row in 0..self.rows {
            for (col, value) in bias.iter().enumerate() {
                self.data[row * self.cols + col] += value;
            }
        }
        Ok(())
    }
}

pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let sum = exps.iter().sum::<f32>().max(1e-12);
    for v in &mut exps {
        *v /= sum;
    }
    exps
}

pub fn normalize_probs(values: &[f32]) -> Vec<f32> {
    let mut clipped: Vec<f32> = values.iter().map(|v| v.max(1e-8)).collect();
    let sum = clipped.iter().sum::<f32>().max(1e-8);
    for v in &mut clipped {
        *v /= sum;
    }
    clipped
}

pub fn entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .copied()
        .filter(|p| *p > 0.0)
        .map(|p| -p * p.max(1e-8).ln())
        .sum()
}

pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best_value {
            best = idx;
            best_value = value;
        }
    }
    best
}

pub fn project_or_pad(x: &Tensor, out_dim: usize) -> Tensor {
    if x.cols == out_dim {
        return x.clone();
    }
    let mut out = Tensor::zeros(x.rows, out_dim);
    let copy = x.cols.min(out_dim);
    for row in 0..x.rows {
        for col in 0..copy {
            out.set(row, col, x.get(row, col));
        }
    }
    out
}

pub fn one_hot(index: usize, dim: usize) -> Tensor {
    let mut out = Tensor::zeros(1, dim);
    out.set(0, index, 1.0);
    out
}
