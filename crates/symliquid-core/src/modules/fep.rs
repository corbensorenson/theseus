use rand::Rng;

use crate::error::{Result, SymError};
use crate::tensor::{entropy, normalize_probs};

#[derive(Debug, Clone)]
pub struct FEPLayer {
    pub latent_dim: usize,
    pub obs_dim: usize,
    pub action_dim: usize,
    pub epistemic_weight: f32,
    transition: Vec<f32>,
    likelihood: Vec<f32>,
}

impl FEPLayer {
    pub fn new(
        latent_dim: usize,
        obs_dim: usize,
        action_dim: usize,
        rng: &mut impl Rng,
    ) -> Result<Self> {
        if latent_dim == 0 || obs_dim == 0 || action_dim == 0 {
            return Err(SymError::InvalidArgument(
                "FEPLayer requires nonzero latent, observation, and action dims".to_string(),
            ));
        }
        let mut transition = vec![0.0; action_dim * latent_dim * latent_dim];
        for action in 0..action_dim {
            for from in 0..latent_dim {
                for to in 0..latent_dim {
                    let same = if from == to {
                        0.82
                    } else {
                        0.18 / (latent_dim - 1).max(1) as f32
                    };
                    let jitter = rng.gen::<f32>() * 0.01;
                    transition[action * latent_dim * latent_dim + from * latent_dim + to] =
                        same + jitter;
                }
                normalize_transition_row(&mut transition, action, from, latent_dim);
            }
        }
        let mut likelihood = vec![0.0; latent_dim * obs_dim];
        for z in 0..latent_dim {
            for o in 0..obs_dim {
                likelihood[z * obs_dim + o] = if o == z % obs_dim {
                    0.8
                } else {
                    0.2 / (obs_dim - 1).max(1) as f32
                };
            }
        }
        Ok(Self {
            latent_dim,
            obs_dim,
            action_dim,
            epistemic_weight: 0.5,
            transition,
            likelihood,
        })
    }

    pub fn expected_free_energy(&self, belief: &[f32], preferences: &[f32]) -> Result<Vec<f32>> {
        if belief.len() != self.latent_dim {
            return Err(SymError::Shape(format!(
                "belief expected {} values, got {}",
                self.latent_dim,
                belief.len()
            )));
        }
        if preferences.len() != self.obs_dim {
            return Err(SymError::Shape(format!(
                "preferences expected {} values, got {}",
                self.obs_dim,
                preferences.len()
            )));
        }
        let belief = normalize_probs(belief);
        let prefs = normalize_probs(preferences);
        let prior_entropy = entropy(&belief);
        let mut scores = vec![0.0; self.action_dim];
        for (action, score) in scores.iter_mut().enumerate() {
            let next = self.predict_latent(&belief, action);
            let obs = self.predict_obs(&next);
            let risk = obs
                .iter()
                .zip(&prefs)
                .map(|(p, pref)| -p * pref.max(1e-8).ln())
                .sum::<f32>();
            let ambiguity = entropy(&obs);
            let info_gain = (prior_entropy - entropy(&next)).max(0.0);
            *score = risk + ambiguity - self.epistemic_weight * info_gain;
        }
        Ok(scores)
    }

    pub fn select_action(&self, belief: &[f32], preferences: &[f32]) -> Result<usize> {
        let scores = self.expected_free_energy(belief, preferences)?;
        let mut best = 0;
        let mut best_score = f32::INFINITY;
        for (idx, score) in scores.iter().copied().enumerate() {
            if score < best_score {
                best = idx;
                best_score = score;
            }
        }
        Ok(best)
    }

    pub fn update_belief(
        &self,
        prior: &[f32],
        action: usize,
        observation: usize,
    ) -> Result<Vec<f32>> {
        if action >= self.action_dim || observation >= self.obs_dim {
            return Err(SymError::InvalidArgument(format!(
                "invalid action {action} or observation {observation}"
            )));
        }
        let prior = normalize_probs(prior);
        let predicted = self.predict_latent(&prior, action);
        let mut posterior = vec![0.0; self.latent_dim];
        for z in 0..self.latent_dim {
            posterior[z] = predicted[z] * self.likelihood[z * self.obs_dim + observation].max(1e-8);
        }
        Ok(normalize_probs(&posterior))
    }

    pub fn belief_loss(&self, belief: &[f32], target: usize) -> Result<f32> {
        if target >= self.latent_dim {
            return Err(SymError::InvalidArgument(format!(
                "target latent state {target} outside {}",
                self.latent_dim
            )));
        }
        let belief = normalize_probs(belief);
        Ok(-belief[target].max(1e-8).ln())
    }

    pub fn set_transition(&mut self, action: usize, from: usize, probs: &[f32]) -> Result<()> {
        if action >= self.action_dim || from >= self.latent_dim || probs.len() != self.latent_dim {
            return Err(SymError::Shape(
                "set_transition expected a latent-sized probability row".to_string(),
            ));
        }
        let probs = normalize_probs(probs);
        let base = action * self.latent_dim * self.latent_dim + from * self.latent_dim;
        self.transition[base..base + self.latent_dim].copy_from_slice(&probs);
        Ok(())
    }

    pub fn set_likelihood(&mut self, latent: usize, probs: &[f32]) -> Result<()> {
        if latent >= self.latent_dim || probs.len() != self.obs_dim {
            return Err(SymError::Shape(
                "set_likelihood expected an observation-sized probability row".to_string(),
            ));
        }
        let probs = normalize_probs(probs);
        let base = latent * self.obs_dim;
        self.likelihood[base..base + self.obs_dim].copy_from_slice(&probs);
        Ok(())
    }

    fn predict_latent(&self, belief: &[f32], action: usize) -> Vec<f32> {
        let mut next = vec![0.0; self.latent_dim];
        for (from, belief_value) in belief.iter().copied().enumerate().take(self.latent_dim) {
            for (to, next_value) in next.iter_mut().enumerate().take(self.latent_dim) {
                let idx = action * self.latent_dim * self.latent_dim + from * self.latent_dim + to;
                *next_value += belief_value * self.transition[idx];
            }
        }
        normalize_probs(&next)
    }

    fn predict_obs(&self, latent: &[f32]) -> Vec<f32> {
        let mut obs = vec![0.0; self.obs_dim];
        for (z, latent_value) in latent.iter().copied().enumerate().take(self.latent_dim) {
            for (o, obs_value) in obs.iter_mut().enumerate().take(self.obs_dim) {
                *obs_value += latent_value * self.likelihood[z * self.obs_dim + o];
            }
        }
        normalize_probs(&obs)
    }

    pub fn parameter_count(&self) -> usize {
        self.transition.len() + self.likelihood.len()
    }
}

fn normalize_transition_row(values: &mut [f32], action: usize, from: usize, latent_dim: usize) {
    let base = action * latent_dim * latent_dim + from * latent_dim;
    let sum = values[base..base + latent_dim]
        .iter()
        .sum::<f32>()
        .max(1e-8);
    for value in &mut values[base..base + latent_dim] {
        *value /= sum;
    }
}
