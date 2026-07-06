use std::time::Instant;

use rand::{Rng, SeedableRng};

use crate::cgs::{CgsAccounting, VerificationReport};
use crate::config::AblationConfig;
use crate::eval::TaskReport;
use crate::tensor::{entropy, normalize_probs};

#[derive(Debug, Clone)]
pub struct ActiveGridworldConfig {
    pub episodes: usize,
    pub seed: u64,
    pub size: usize,
    pub max_steps: usize,
    pub ablations: AblationConfig,
    pub variant: String,
}

impl Default for ActiveGridworldConfig {
    fn default() -> Self {
        Self {
            episodes: 100,
            seed: 0,
            size: 4,
            max_steps: 12,
            ablations: AblationConfig::full(),
            variant: "full".to_string(),
        }
    }
}

pub fn run(config: ActiveGridworldConfig) -> crate::error::Result<TaskReport> {
    let start = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let size = config.size.max(2);
    let goals = [(size - 1, 0), (size - 1, size - 1)];
    let mut successes = 0usize;
    let mut total_steps = 0usize;
    let mut total_uncertainty_reduction = 0.0;

    for _ in 0..config.episodes {
        let hidden_goal = rng.gen_range(0..2);
        let mut pos = (0usize, 0usize);
        let mut belief = vec![0.5, 0.5];
        let mut inspected = false;
        let mut success = false;

        for step in 0..config.max_steps {
            if pos == goals[hidden_goal] {
                successes += 1;
                total_steps += step;
                success = true;
                break;
            }

            let action = if config.ablations.use_fep {
                fep_grid_action(&belief, inspected, pos, goals)
            } else {
                rng.gen_range(0..5)
            };

            if action == 4 {
                if !inspected {
                    let prior_entropy = entropy(&belief);
                    belief = if hidden_goal == 0 {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    };
                    total_uncertainty_reduction += prior_entropy - entropy(&belief);
                    inspected = true;
                }
            } else {
                pos = step_position(pos, action, size);
            }
        }

        if !success {
            if pos == goals[hidden_goal] {
                successes += 1;
            }
            total_steps += config.max_steps;
        }
    }

    let accuracy = successes as f32 / config.episodes.max(1) as f32;
    let residual = 1.0 - accuracy;
    let avg_steps = total_steps as f32 / config.episodes.max(1) as f32;
    let avg_info_gain = total_uncertainty_reduction / config.episodes.max(1) as f32;
    let governance_power = (accuracy - 0.5).max(0.0);
    let state_count = size * size * 2;
    let cgs = CgsAccounting {
        seed_cost: 2.0,
        rule_cost: 5.0 * state_count as f32,
        memory_cost: state_count as f32,
        residual_cost: residual,
        verification_cost: config.episodes as f32,
        governance_cost: avg_steps,
        target_cost: (config.episodes * state_count) as f32,
        fidelity: accuracy,
        governance_power,
    };

    Ok(TaskReport {
        task: "active_gridworld".to_string(),
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
            successes,
            "gridworld_failure_rate",
            residual,
        ),
        notes: vec![
            format!("Average steps: {:.2}", avg_steps),
            format!("Average information gain: {:.3}", avg_info_gain),
            "The FEP policy inspects when hidden-goal uncertainty is high, then navigates."
                .to_string(),
        ],
    })
}

fn fep_grid_action(
    belief: &[f32],
    inspected: bool,
    pos: (usize, usize),
    goals: [(usize, usize); 2],
) -> usize {
    let belief = normalize_probs(belief);
    if !inspected && entropy(&belief) > 0.05 {
        return 4;
    }
    let target = if belief[1] > belief[0] {
        goals[1]
    } else {
        goals[0]
    };
    move_toward(pos, target)
}

fn move_toward(pos: (usize, usize), target: (usize, usize)) -> usize {
    if pos.0 < target.0 {
        1
    } else if pos.0 > target.0 {
        0
    } else if pos.1 < target.1 {
        3
    } else if pos.1 > target.1 {
        2
    } else {
        4
    }
}

fn step_position(pos: (usize, usize), action: usize, size: usize) -> (usize, usize) {
    match action {
        0 => (pos.0.saturating_sub(1), pos.1),
        1 => ((pos.0 + 1).min(size - 1), pos.1),
        2 => (pos.0, pos.1.saturating_sub(1)),
        3 => (pos.0, (pos.1 + 1).min(size - 1)),
        _ => pos,
    }
}
