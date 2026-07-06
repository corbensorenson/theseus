use std::time::Instant;

use rand::{Rng, SeedableRng};

use crate::cgs::{CgsAccounting, VerificationReport};
use crate::config::AblationConfig;
use crate::eval::TaskReport;
use crate::tensor::{entropy, normalize_probs};

#[derive(Debug, Clone)]
pub struct ActiveClassificationConfig {
    pub episodes: usize,
    pub seed: u64,
    pub feature_dim: usize,
    pub max_inspections: usize,
    pub ablations: AblationConfig,
    pub variant: String,
}

impl Default for ActiveClassificationConfig {
    fn default() -> Self {
        Self {
            episodes: 100,
            seed: 0,
            feature_dim: 3,
            max_inspections: 3,
            ablations: AblationConfig::full(),
            variant: "full".to_string(),
        }
    }
}

pub fn run(config: ActiveClassificationConfig) -> crate::error::Result<TaskReport> {
    let start = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let states = 1usize << config.feature_dim;
    let mut correct = 0usize;
    let mut total_loss = 0.0;
    let mut total_inspections = 0usize;

    for _ in 0..config.episodes {
        let hidden = rng.gen_range(0..states);
        let target_label = label(hidden, config.feature_dim);
        let mut belief = vec![1.0 / states as f32; states];
        let mut observed = vec![None; config.feature_dim];
        let mut classified = false;

        for _turn in 0..=config.max_inspections {
            let action = if config.ablations.use_fep {
                select_fep_action(&belief, &observed, config.feature_dim)
            } else {
                random_unobserved_or_classify(&observed, &mut rng)
            };
            if action < config.feature_dim {
                let bit = ((hidden >> action) & 1) as u8;
                observed[action] = Some(bit);
                belief = condition_on_feature(&belief, action, bit, config.feature_dim);
                total_inspections += 1;
            } else {
                let pred_label = marginal_label(&belief, config.feature_dim);
                if pred_label == target_label {
                    correct += 1;
                }
                total_loss += if pred_label == target_label { 0.0 } else { 1.0 };
                classified = true;
                break;
            }
        }

        if !classified {
            let pred_label = marginal_label(&belief, config.feature_dim);
            if pred_label == target_label {
                correct += 1;
            }
            total_loss += if pred_label == target_label { 0.0 } else { 1.0 };
        }
    }

    let accuracy = correct as f32 / config.episodes.max(1) as f32;
    let residual = total_loss / config.episodes.max(1) as f32;
    let governance_power = (accuracy - 0.5).max(0.0);
    let avg_inspections = total_inspections as f32 / config.episodes.max(1) as f32;
    let states = 1usize << config.feature_dim;
    let cgs = CgsAccounting {
        seed_cost: config.feature_dim as f32,
        rule_cost: states as f32 * config.feature_dim as f32,
        memory_cost: states as f32,
        residual_cost: residual,
        verification_cost: config.episodes as f32,
        governance_cost: avg_inspections,
        target_cost: (config.episodes * states) as f32,
        fidelity: accuracy,
        governance_power,
    };

    Ok(TaskReport {
        task: "active_classification".to_string(),
        variant: config.variant,
        seed: config.seed,
        steps: config.episodes,
        accuracy,
        loss: residual,
        params: 0,
        runtime_ms: start.elapsed().as_millis(),
        residual,
        governance_power,
        cgs,
        verification: VerificationReport::new(
            config.episodes,
            correct,
            "classification_failure_rate",
            1.0 - accuracy,
        ),
        notes: vec![
            format!("Average inspections: {:.2}", avg_inspections),
            "CGS residual is the rate of governed classification failures.".to_string(),
        ],
    })
}

fn select_fep_action(belief: &[f32], observed: &[Option<u8>], feature_dim: usize) -> usize {
    let current_entropy = entropy(&normalize_probs(belief));
    let mut best_action = feature_dim;
    let mut best_score = classify_risk(belief, feature_dim);

    for (feature, observed_value) in observed.iter().enumerate().take(feature_dim) {
        if observed_value.is_some() {
            continue;
        }
        let p_one = belief
            .iter()
            .enumerate()
            .filter(|(state, _)| ((*state >> feature) & 1) == 1)
            .map(|(_, p)| *p)
            .sum::<f32>();
        let p_zero = 1.0 - p_one;
        let posterior_zero = condition_on_feature(belief, feature, 0, feature_dim);
        let posterior_one = condition_on_feature(belief, feature, 1, feature_dim);
        let expected_entropy = p_zero * entropy(&posterior_zero) + p_one * entropy(&posterior_one);
        let info_gain = (current_entropy - expected_entropy).max(0.0);
        let ambiguity = entropy(&[p_zero.max(1e-8), p_one.max(1e-8)]);
        let inspection_cost = 0.05;
        let efe = inspection_cost + ambiguity - info_gain;
        if efe < best_score {
            best_score = efe;
            best_action = feature;
        }
    }
    best_action
}

fn random_unobserved_or_classify(observed: &[Option<u8>], rng: &mut impl Rng) -> usize {
    let choices = observed
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| if value.is_none() { Some(idx) } else { None })
        .collect::<Vec<_>>();
    if choices.is_empty() || rng.gen::<f32>() < 0.35 {
        observed.len()
    } else {
        choices[rng.gen_range(0..choices.len())]
    }
}

fn condition_on_feature(
    belief: &[f32],
    feature: usize,
    value: u8,
    _feature_dim: usize,
) -> Vec<f32> {
    let mut next = belief.to_vec();
    for (state, p) in next.iter_mut().enumerate() {
        if ((state >> feature) & 1) as u8 != value {
            *p = 0.0;
        }
    }
    normalize_probs(&next)
}

fn marginal_label(belief: &[f32], feature_dim: usize) -> u8 {
    let mut scores = [0.0, 0.0];
    for (state, p) in belief.iter().copied().enumerate() {
        scores[label(state, feature_dim) as usize] += p;
    }
    if scores[1] >= scores[0] {
        1
    } else {
        0
    }
}

fn classify_risk(belief: &[f32], feature_dim: usize) -> f32 {
    let mut scores = [0.0, 0.0];
    for (state, p) in belief.iter().copied().enumerate() {
        scores[label(state, feature_dim) as usize] += p;
    }
    1.0 - scores[0].max(scores[1])
}

fn label(state: usize, feature_dim: usize) -> u8 {
    let ones = (0..feature_dim)
        .filter(|feature| ((state >> feature) & 1) == 1)
        .count();
    if ones * 2 >= feature_dim {
        1
    } else {
        0
    }
}
