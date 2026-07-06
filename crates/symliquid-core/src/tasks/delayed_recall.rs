use std::time::Instant;

use rand::{Rng, SeedableRng};

use crate::cgs::{CgsAccounting, VerificationReport};
use crate::config::AblationConfig;
use crate::error::Result;
use crate::eval::TaskReport;
use crate::modules::reservoir::Reservoir;
use crate::modules::vsa::VSAMemory;
use crate::tensor::{one_hot, project_or_pad, Tensor};
use crate::train::LinearReadout;

#[derive(Debug, Clone)]
pub struct DelayedRecallConfig {
    pub steps: usize,
    pub seed: u64,
    pub seq_len: usize,
    pub delay: usize,
    pub vocab_size: usize,
    pub reservoir_dim: usize,
    pub hv_dim: usize,
    pub ablations: AblationConfig,
    pub variant: String,
}

impl Default for DelayedRecallConfig {
    fn default() -> Self {
        Self {
            steps: 200,
            seed: 0,
            seq_len: 6,
            delay: 4,
            vocab_size: 8,
            reservoir_dim: 128,
            hv_dim: 512,
            ablations: AblationConfig::full(),
            variant: "full".to_string(),
        }
    }
}

pub fn run(config: DelayedRecallConfig) -> Result<TaskReport> {
    let start = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let input_dim = config.vocab_size + config.seq_len;
    let reservoir = Reservoir::new(input_dim, config.reservoir_dim, 0.9, 0.7, false, &mut rng)?;
    let mut readout = LinearReadout::new(config.reservoir_dim, config.vocab_size, &mut rng);
    let positions = VSAMemory::random_bipolar(config.seq_len, config.hv_dim, &mut rng);
    let tokens = VSAMemory::random_bipolar(config.vocab_size, config.hv_dim, &mut rng);
    let mut total_loss = 0.0;
    let mut total_acc = 0.0;
    let lr = 0.05;

    for _step in 0..config.steps {
        let sequence = (0..config.seq_len)
            .map(|_| rng.gen_range(0..config.vocab_size))
            .collect::<Vec<_>>();
        let query_pos = rng.gen_range(0..config.seq_len);
        let mut r = Tensor::zeros(1, config.reservoir_dim);
        let mut symbolic_memory = Tensor::zeros(1, config.hv_dim);

        for (pos, token) in sequence.iter().copied().enumerate() {
            let x = token_position_input(token, pos, config.vocab_size, config.seq_len)?;
            if config.ablations.use_reservoir {
                r = reservoir.forward(&x, &r)?;
            }
            if config.ablations.use_vsa {
                let pos_hv = Tensor::new(1, config.hv_dim, positions.row(pos).to_vec())?;
                let token_hv = Tensor::new(1, config.hv_dim, tokens.row(token).to_vec())?;
                let bound = VSAMemory::bind(&pos_hv, &token_hv)?;
                symbolic_memory = VSAMemory::bundle(&symbolic_memory, &bound, 1.0)?;
            }
        }
        for _ in 0..config.delay {
            let x = Tensor::zeros(1, input_dim);
            if config.ablations.use_reservoir {
                r = reservoir.forward(&x, &r)?;
            }
        }
        let query = query_input(query_pos, config.vocab_size, config.seq_len)?;
        let target = sequence[query_pos];
        if config.ablations.use_vsa {
            if config.ablations.use_reservoir {
                let _ = reservoir.forward(&query, &r)?;
            }
            let query_hv = Tensor::new(1, config.hv_dim, positions.row(query_pos).to_vec())?;
            let recovered = VSAMemory::unbind(&symbolic_memory, &query_hv)?;
            let (pred, score) = VSAMemory::cleanup(&recovered, &tokens)?;
            total_acc += if pred == target { 1.0 } else { 0.0 };
            total_loss += 1.0 - score;
        } else {
            let features = if config.ablations.use_reservoir {
                reservoir.forward(&query, &r)?
            } else {
                project_or_pad(&query, config.reservoir_dim)
            };
            let trace = readout.train_step(&features, target, lr)?;
            total_loss += trace.loss;
            total_acc += trace.accuracy;
        }
    }

    let accuracy = total_acc / config.steps.max(1) as f32;
    let residual = total_loss / config.steps.max(1) as f32;
    let chance = 1.0 / config.vocab_size.max(1) as f32;
    let governance_power = (accuracy - chance).max(0.0);
    let params = reservoir.parameter_count()
        + readout.parameter_count()
        + if config.ablations.use_vsa {
            config.hv_dim * (config.seq_len + config.vocab_size)
        } else {
            0
        };
    let cgs = CgsAccounting {
        seed_cost: input_dim as f32,
        rule_cost: params as f32,
        memory_cost: (config.reservoir_dim
            + if config.ablations.use_vsa {
                config.hv_dim
            } else {
                0
            }) as f32,
        residual_cost: residual,
        verification_cost: config.steps as f32,
        governance_cost: config.delay as f32 + config.seq_len as f32,
        target_cost: (config.steps * config.seq_len * config.vocab_size) as f32,
        fidelity: accuracy,
        governance_power,
    };

    Ok(TaskReport {
        task: "delayed_recall".to_string(),
        variant: config.variant,
        seed: config.seed,
        steps: config.steps,
        accuracy,
        loss: residual,
        params,
        runtime_ms: start.elapsed().as_millis(),
        residual,
        governance_power,
        cgs,
        verification: VerificationReport::new(
            config.steps,
            (accuracy * config.steps as f32).round() as usize,
            if config.ablations.use_vsa {
                "delayed_recall_retrieval_error"
            } else {
                "delayed_recall_cross_entropy"
            },
            residual,
        ),
        notes: vec![
            "Full path binds token hypervectors to position hypervectors before delay.".to_string(),
            "No-VSA ablation falls back to a small readout over fixed reservoir features."
                .to_string(),
            "CGS residual is retrieval error for VSA runs or cross-entropy for no-VSA runs."
                .to_string(),
        ],
    })
}

fn token_position_input(
    token: usize,
    pos: usize,
    vocab_size: usize,
    seq_len: usize,
) -> Result<Tensor> {
    let mut x = Tensor::zeros(1, vocab_size + seq_len);
    x.set(0, token, 1.0);
    x.set(0, vocab_size + pos, 1.0);
    Ok(x)
}

fn query_input(pos: usize, vocab_size: usize, seq_len: usize) -> Result<Tensor> {
    let mut x = one_hot(vocab_size + pos, vocab_size + seq_len);
    x.set(0, vocab_size + pos, 2.0);
    Ok(x)
}
