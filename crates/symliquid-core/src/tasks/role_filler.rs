use std::time::Instant;

use rand::{Rng, SeedableRng};

use crate::cgs::{CgsAccounting, VerificationReport};
use crate::config::AblationConfig;
use crate::error::Result;
use crate::eval::TaskReport;
use crate::modules::vsa::VSAMemory;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct RoleFillerConfig {
    pub steps: usize,
    pub seed: u64,
    pub hv_dim: usize,
    pub vocab_size: usize,
    pub facts_per_sample: usize,
    pub ablations: AblationConfig,
    pub variant: String,
}

impl Default for RoleFillerConfig {
    fn default() -> Self {
        Self {
            steps: 200,
            seed: 0,
            hv_dim: 512,
            vocab_size: 24,
            facts_per_sample: 3,
            ablations: AblationConfig::full(),
            variant: "full".to_string(),
        }
    }
}

pub fn run(config: RoleFillerConfig) -> Result<TaskReport> {
    let start = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let roles = VSAMemory::random_bipolar(config.facts_per_sample, config.hv_dim, &mut rng);
    let fillers = VSAMemory::random_bipolar(config.vocab_size, config.hv_dim, &mut rng);
    let mut correct = 0usize;
    let mut loss = 0.0;

    for _ in 0..config.steps {
        let assignments = (0..config.facts_per_sample)
            .map(|_| rng.gen_range(0..config.vocab_size))
            .collect::<Vec<_>>();
        let query_role = rng.gen_range(0..config.facts_per_sample);
        let mut memory = Tensor::zeros(1, config.hv_dim);

        if config.ablations.use_vsa {
            for (role_idx, filler_idx) in assignments.iter().copied().enumerate() {
                let role = Tensor::new(1, config.hv_dim, roles.row(role_idx).to_vec())?;
                let filler = Tensor::new(1, config.hv_dim, fillers.row(filler_idx).to_vec())?;
                let bound = VSAMemory::bind(&role, &filler)?;
                memory = VSAMemory::bundle(&memory, &bound, 1.0)?;
            }
            let query = Tensor::new(1, config.hv_dim, roles.row(query_role).to_vec())?;
            let recovered = VSAMemory::unbind(&memory, &query)?;
            let (pred, score) = VSAMemory::cleanup(&recovered, &fillers)?;
            if pred == assignments[query_role] {
                correct += 1;
            }
            loss += 1.0 - score;
        } else {
            let pred = rng.gen_range(0..config.vocab_size);
            if pred == assignments[query_role] {
                correct += 1;
            }
            loss += 1.0;
        }
    }

    let accuracy = correct as f32 / config.steps.max(1) as f32;
    let residual = loss / config.steps.max(1) as f32;
    let chance = 1.0 / config.vocab_size.max(1) as f32;
    let governance_power = (accuracy - chance).max(0.0);
    let params = config.hv_dim * (config.vocab_size + config.facts_per_sample);
    let cgs = CgsAccounting {
        seed_cost: config.facts_per_sample as f32,
        rule_cost: if config.ablations.use_vsa {
            config.hv_dim as f32
        } else {
            1.0
        },
        memory_cost: config.hv_dim as f32,
        residual_cost: residual,
        verification_cost: config.steps as f32,
        governance_cost: config.facts_per_sample as f32,
        target_cost: (config.steps * config.facts_per_sample * config.hv_dim) as f32,
        fidelity: accuracy,
        governance_power,
    };
    Ok(TaskReport {
        task: "role_filler".to_string(),
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
            correct,
            "cleanup_retrieval_error",
            1.0 - accuracy,
        ),
        notes: vec![
            "Uses bipolar VSA role-filler binding with cleanup retrieval.".to_string(),
            "CGS residual is one minus cleanup cosine/retrieval fidelity.".to_string(),
            "No external dataset is used.".to_string(),
        ],
    })
}
