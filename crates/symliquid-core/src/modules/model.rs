use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::config::SymLiquidConfig;
use crate::error::Result;
use crate::modules::fep::FEPLayer;
use crate::modules::kan_lite::KANLiteLayer;
use crate::modules::liquid::LiquidCell;
use crate::modules::reservoir::Reservoir;
use crate::modules::vsa::VSAMemory;
use crate::tensor::{entropy, project_or_pad, softmax, Tensor};

#[derive(Debug, Clone)]
pub struct ModelState {
    pub h: Tensor,
    pub r: Tensor,
}

#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub logits: Tensor,
    pub h: Tensor,
    pub r: Tensor,
    pub hv: Tensor,
    pub memory: Tensor,
    pub belief: Tensor,
    pub expected_free_energy: Option<Vec<f32>>,
    pub residual: f32,
    pub loss_terms: Vec<(String, f32)>,
}

#[derive(Debug, Clone)]
pub struct SymLiquidFEPNet {
    pub config: SymLiquidConfig,
    pub encoder: KANLiteLayer,
    pub liquid: LiquidCell,
    pub reservoir: Reservoir,
    pub vsa: VSAMemory,
    pub fep: FEPLayer,
    readout_w: Vec<f32>,
    readout_b: Vec<f32>,
}

impl SymLiquidFEPNet {
    pub fn new(config: SymLiquidConfig, seed: u64) -> Result<Self> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let encoder = KANLiteLayer::new(
            config.input_dim,
            config.hidden_dim,
            config.kan_basis,
            &mut rng,
        )?;
        let liquid = LiquidCell::new(config.hidden_dim, config.hidden_dim, &mut rng)?;
        let reservoir = Reservoir::new(
            config.hidden_dim,
            config.reservoir_dim,
            0.9,
            0.75,
            false,
            &mut rng,
        )?;
        let vsa = VSAMemory::new(config.reservoir_dim, config.hv_dim, &mut rng)?;
        let fep = FEPLayer::new(
            config.latent_dim,
            config.obs_dim,
            config.action_dim,
            &mut rng,
        )?;
        let normal = StandardNormal;
        let scale = (1.0 / config.hv_dim as f32).sqrt();
        let readout_w = (0..config.output_dim * config.hv_dim)
            .map(|_| {
                let z: f32 = normal.sample(&mut rng);
                z * scale
            })
            .collect();
        let readout_b = vec![0.0; config.output_dim];
        Ok(Self {
            config,
            encoder,
            liquid,
            reservoir,
            vsa,
            fep,
            readout_w,
            readout_b,
        })
    }

    pub fn zero_state(&self, batch: usize) -> ModelState {
        ModelState {
            h: Tensor::zeros(batch, self.config.hidden_dim),
            r: Tensor::zeros(batch, self.config.reservoir_dim),
        }
    }

    pub fn forward_step(
        &self,
        x: &Tensor,
        state: Option<&ModelState>,
        memory: Option<&Tensor>,
        preferences: Option<&[f32]>,
    ) -> Result<ModelOutput> {
        x.ensure_cols(self.config.input_dim, "SymLiquidFEPNet input")?;
        let prev = state.cloned().unwrap_or_else(|| self.zero_state(x.rows));

        let encoded = if self.config.ablations.use_kan {
            self.encoder.forward(x)?
        } else {
            project_or_pad(x, self.config.hidden_dim)
        };

        let h = if self.config.ablations.use_liquid {
            self.liquid.forward(&encoded, &prev.h)?
        } else {
            encoded.clone()
        };

        let r = if self.config.ablations.use_reservoir {
            self.reservoir.forward(&h, &prev.r)?
        } else {
            project_or_pad(&h, self.config.reservoir_dim)
        };

        let (hv, memory) = if self.config.ablations.use_vsa {
            self.vsa.forward(&r, memory, 0.98, false)?
        } else {
            let hv = project_or_pad(&r, self.config.hv_dim);
            let memory = match memory {
                Some(prev) => VSAMemory::bundle(prev, &hv, 0.98)?,
                None => hv.clone(),
            };
            (hv, memory)
        };

        let logits = self.readout_logits(&memory)?;
        let belief = self.belief_from_memory(&memory);
        let residual = if belief.rows > 0 {
            entropy(belief.row(0))
        } else {
            0.0
        };
        let expected_free_energy = if self.config.ablations.use_fep {
            match preferences {
                Some(pref) => Some(self.fep.expected_free_energy(belief.row(0), pref)?),
                None => None,
            }
        } else {
            None
        };

        let mut loss_terms = vec![(
            "kan_regularization".to_string(),
            if self.config.ablations.use_kan {
                self.encoder.regularization(1e-4, 1e-4)
            } else {
                0.0
            },
        )];
        if self.config.ablations.use_vsa && hv.rows > 0 {
            loss_terms.push(("memory_norm".to_string(), row_norm(memory.row(0))));
        }

        Ok(ModelOutput {
            logits,
            h,
            r,
            hv,
            memory,
            belief,
            expected_free_energy,
            residual,
            loss_terms,
        })
    }

    pub fn readout_logits(&self, features: &Tensor) -> Result<Tensor> {
        features.ensure_cols(self.config.hv_dim, "readout features")?;
        let mut logits =
            features.matmul_right_transposed(&self.readout_w, self.config.output_dim)?;
        logits.add_row_bias(&self.readout_b)?;
        Ok(logits)
    }

    pub fn train_readout_sgd(
        &mut self,
        features: &Tensor,
        targets: &[usize],
        lr: f32,
    ) -> Result<(f32, f32)> {
        features.ensure_cols(self.config.hv_dim, "readout training features")?;
        if targets.len() != features.rows {
            return Err(crate::error::SymError::Shape(format!(
                "target batch expected {}, got {}",
                features.rows,
                targets.len()
            )));
        }
        let logits = self.readout_logits(features)?;
        let mut loss = 0.0;
        let mut correct = 0usize;
        let mut grad_w = vec![0.0; self.readout_w.len()];
        let mut grad_b = vec![0.0; self.readout_b.len()];
        for (row, target) in targets.iter().copied().enumerate() {
            let probs = softmax(logits.row(row));
            loss += -probs[target].max(1e-8).ln();
            let pred = crate::tensor::argmax(&probs);
            if pred == target {
                correct += 1;
            }
            for o in 0..self.config.output_dim {
                let delta = probs[o] - if o == target { 1.0 } else { 0.0 };
                grad_b[o] += delta;
                for i in 0..self.config.hv_dim {
                    grad_w[o * self.config.hv_dim + i] += delta * features.get(row, i);
                }
            }
        }
        let scale = 1.0 / features.rows as f32;
        for (w, grad) in self.readout_w.iter_mut().zip(grad_w) {
            *w -= lr * grad * scale;
        }
        for (b, grad) in self.readout_b.iter_mut().zip(grad_b) {
            *b -= lr * grad * scale;
        }
        Ok((loss * scale, correct as f32 / features.rows as f32))
    }

    pub fn parameter_count(&self) -> usize {
        self.encoder.parameter_count()
            + self.liquid.parameter_count()
            + self.reservoir.parameter_count()
            + self.vsa.parameter_count()
            + self.fep.parameter_count()
            + self.readout_w.len()
            + self.readout_b.len()
    }

    fn belief_from_memory(&self, memory: &Tensor) -> Tensor {
        let mut out = Tensor::zeros(memory.rows, self.config.latent_dim);
        for row in 0..memory.rows {
            let mut logits = vec![0.0; self.config.latent_dim];
            for (idx, logit) in logits.iter_mut().enumerate() {
                *logit = memory.get(row, idx % memory.cols);
            }
            let probs = softmax(&logits);
            for (idx, prob) in probs.iter().copied().enumerate() {
                out.set(row, idx, prob);
            }
        }
        out
    }
}

fn row_norm(row: &[f32]) -> f32 {
    row.iter().map(|v| v * v).sum::<f32>().sqrt()
}
