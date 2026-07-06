use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rand::{seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Result, SymError};
use crate::tensor::{argmax, Tensor};
use crate::train::LinearReadout;

mod linguistic_features;

#[cfg(test)]
mod tests;

use linguistic_features::{
    add_pairwise_contrast_features, normalize_in_place, sentence_features, sentence_quality_prior,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub name: String,
    pub version: String,
    pub generated_at_unix: u64,
    pub seed: u64,
    pub cases: Vec<BenchmarkCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCase {
    pub id: String,
    pub task: String,
    pub contract: TaskContract,
    pub observation: String,
    pub hybrid_observation: String,
    pub expected: String,
    pub expected_kind: ExpectedKind,
    pub allowed_actions: Vec<String>,
    pub verifier: VerifierSpec,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContract {
    pub observation_schema: String,
    pub action_schema: String,
    pub max_turns: usize,
    pub max_tokens: usize,
    pub scoring: String,
    pub fairness_notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedKind {
    Answer,
    Action,
    Patch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierSpec {
    pub name: String,
    pub exact_match: bool,
    pub case_sensitive: bool,
    pub invalid_action_penalty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub case_id: String,
    pub model_id: String,
    pub mode: RunMode,
    pub output: String,
    pub runtime_ms: Option<u128>,
    pub token_count: Option<usize>,
    pub tool_calls: Option<usize>,
    pub estimated_cost_usd: Option<f32>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    SymLiquid,
    LocalBaseline,
    SymLiquidAugmented,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    pub case_id: String,
    pub task: String,
    pub model_id: String,
    pub mode: RunMode,
    pub expected: String,
    pub output: String,
    pub parsed: String,
    pub correct: bool,
    pub invalid_action: bool,
    pub score: f32,
    pub residual: f32,
    pub runtime_ms: u128,
    pub token_count: usize,
    pub tool_calls: usize,
    pub estimated_cost_usd: f32,
    pub memory_bytes: usize,
    pub cache_key: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub suite: String,
    pub model_id: String,
    pub mode: RunMode,
    pub cases: usize,
    pub accuracy: f32,
    pub residual: f32,
    pub invalid_action_rate: f32,
    pub total_runtime_ms: u128,
    pub total_tokens: usize,
    pub total_tool_calls: usize,
    pub total_estimated_cost_usd: f32,
    pub total_memory_bytes: usize,
    pub verified_success_per_cost: f32,
    pub task_breakdown: BTreeMap<String, TaskBreakdown>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBreakdown {
    pub cases: usize,
    pub accuracy: f32,
    pub residual: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub summary: BenchmarkSummary,
    pub results: Vec<CaseResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub baseline_report: String,
    pub candidate_report: String,
    pub baseline_model: String,
    pub candidate_model: String,
    pub baseline_mode: RunMode,
    pub candidate_mode: RunMode,
    pub baseline_accuracy: f32,
    pub candidate_accuracy: f32,
    pub accuracy_lift: f32,
    pub residual_delta: f32,
    pub invalid_action_delta: f32,
    pub cost_efficiency_ratio: f32,
    pub token_delta: isize,
    pub runtime_delta_ms: i128,
    pub memory_delta_bytes: isize,
}

#[derive(Debug, Clone)]
pub struct StandaloneTrainConfig {
    pub train_seed: u64,
    pub eval_seed: u64,
    pub cases_per_task: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub hv_dim: usize,
    pub lr: f32,
    pub symbolic_fallback: bool,
    pub artifact_path: Option<String>,
}

impl Default for StandaloneTrainConfig {
    fn default() -> Self {
        Self {
            train_seed: 0,
            eval_seed: 10_000,
            cases_per_task: 100,
            epochs: 20,
            batch_size: 1,
            hv_dim: 4096,
            lr: 0.05,
            symbolic_fallback: false,
            artifact_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandaloneTrainReport {
    #[serde(default)]
    pub model_id: String,
    #[serde(default)]
    pub feature_set: String,
    pub train_seed: u64,
    pub eval_seed: u64,
    pub cases_per_task: usize,
    pub epochs: usize,
    #[serde(default)]
    pub batch_size: usize,
    pub hv_dim: usize,
    pub labels: usize,
    #[serde(default)]
    pub symbolic_fallback: bool,
    #[serde(default)]
    pub train_runtime_ms: u128,
    #[serde(default)]
    pub train_examples_per_second: f32,
    pub train_loss: f32,
    pub train_accuracy: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub readout_routing: Option<ReadoutRoutingDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_training: Option<StateTrainingDiagnostics>,
    #[serde(default)]
    pub runtime_profile: BTreeMap<String, String>,
    #[serde(default)]
    pub timing_breakdown_ms: BTreeMap<String, u128>,
    #[serde(default)]
    pub kernel_launches: usize,
    #[serde(default)]
    pub cuda_fallback: bool,
    pub eval: BenchmarkReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutRoutingDiagnostics {
    pub selected_readout: String,
    pub probe_cases: usize,
    pub shared_probe_accuracy: f32,
    #[serde(default)]
    pub residual_adapter_probe_accuracy: f32,
    #[serde(default)]
    pub residual_adapter_worst_task_accuracy_delta: f32,
    #[serde(default)]
    pub accepted_residual_adapters: bool,
    #[serde(default)]
    pub residual_adapter_rank: usize,
    pub task_head_probe_accuracy: f32,
    pub task_head_worst_task_accuracy_delta: f32,
    #[serde(default)]
    pub accepted_task_heads: bool,
    pub acceptance_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTrainingDiagnostics {
    pub attempted: bool,
    pub accepted: bool,
    pub state_epochs: usize,
    pub state_lr: f32,
    #[serde(default)]
    pub probe_cases: usize,
    #[serde(default)]
    pub probe_accuracy_metric: String,
    pub base_probe_loss: f32,
    pub base_probe_accuracy: f32,
    #[serde(default)]
    pub base_state_alignment: f32,
    #[serde(default)]
    pub base_probe_task_breakdown: BTreeMap<String, TaskBreakdown>,
    pub candidate_probe_loss: f32,
    pub candidate_probe_accuracy: f32,
    #[serde(default)]
    pub candidate_state_alignment: f32,
    #[serde(default)]
    pub candidate_probe_task_breakdown: BTreeMap<String, TaskBreakdown>,
    #[serde(default)]
    pub worst_probe_task_accuracy_delta: f32,
    #[serde(default)]
    pub task_gated_candidate: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidate_state_tasks: Vec<String>,
    #[serde(default)]
    pub gated_probe_loss: f32,
    #[serde(default)]
    pub gated_probe_accuracy: f32,
    #[serde(default)]
    pub gated_worst_task_accuracy_delta: f32,
    pub acceptance_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedSweepReport {
    pub train_seeds: Vec<u64>,
    pub eval_seed_base: u64,
    pub cases_per_task: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub hv_dim: usize,
    pub lr: f32,
    pub symbolic_fallback: bool,
    pub runs: Vec<StandaloneTrainReport>,
    pub mean_accuracy: f32,
    pub std_accuracy: f32,
    pub mean_residual: f32,
    pub std_residual: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutArtifact {
    pub labels: Vec<String>,
    pub hv_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub feature_set: String,
}

#[derive(Debug, Clone)]
pub struct BabyLmProbeTrainConfig {
    pub input_path: String,
    pub eval_input_path: Option<String>,
    pub train_seed: u64,
    pub eval_seed: u64,
    pub train_limit: usize,
    pub eval_limit: usize,
    pub steps: usize,
    pub hv_dim: usize,
    pub lr: f32,
    pub stateful: bool,
    pub pairwise_contrast: bool,
    pub balance_rules: bool,
    pub prior_weight: f32,
}

impl Default for BabyLmProbeTrainConfig {
    fn default() -> Self {
        Self {
            input_path:
                "C:\\Users\\corbe\\Documents\\babylm-candidate\\data\\samples\\strict_small_50k_words.txt"
                    .to_string(),
            eval_input_path: None,
            train_seed: 0,
            eval_seed: 10_000,
            train_limit: 1_000,
            eval_limit: 300,
            steps: 1_000,
            hv_dim: 8192,
            lr: 0.05,
            stateful: false,
            pairwise_contrast: false,
            balance_rules: false,
            prior_weight: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BabyLmProbeTrainReport {
    #[serde(default = "default_babylm_probe_policy")]
    pub policy: String,
    #[serde(default = "default_babylm_probe_methodology")]
    pub methodology: String,
    #[serde(default = "default_babylm_frontier_family")]
    pub frontier_family: String,
    #[serde(default = "default_babylm_runner_family")]
    pub runner_family: String,
    #[serde(default)]
    pub external_inference_calls: usize,
    pub input_path: String,
    pub eval_input_path: Option<String>,
    pub train_seed: u64,
    pub eval_seed: u64,
    pub train_limit: usize,
    pub eval_limit: usize,
    pub steps: usize,
    pub hv_dim: usize,
    pub lr: f32,
    pub stateful: bool,
    pub pairwise_contrast: bool,
    pub balance_rules: bool,
    pub prior_weight: f32,
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub eval: BenchmarkReport,
}

fn default_babylm_probe_policy() -> String {
    "project_theseus_babylm_probe_train_report_v1".to_string()
}

fn default_babylm_probe_methodology() -> String {
    "local_babylm_probe_train_eval".to_string()
}

fn default_babylm_frontier_family() -> String {
    "babylm_mutated".to_string()
}

fn default_babylm_runner_family() -> String {
    "symliquid_babylm_sequence_scorer".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MinimalPairCache {
    source_path: String,
    source_len: u64,
    source_modified_unix: u64,
    limit: usize,
    cases: Vec<BenchmarkCase>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LocalBaselineKind {
    FirstAllowed,
    BagOfWords,
    HashReadout,
}

#[derive(Debug, Clone)]
struct PairwiseTrainingExample {
    diff: Vec<f32>,
    prior_delta: f32,
    target_a: bool,
    rule: String,
}

#[derive(Debug, Clone)]
struct RuleTrainingGroup {
    indices: Vec<usize>,
    cursor: usize,
}

impl LocalBaselineKind {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "first_allowed" => Ok(Self::FirstAllowed),
            "bag_of_words" => Ok(Self::BagOfWords),
            "hash_readout" => Ok(Self::HashReadout),
            other => Err(SymError::InvalidArgument(format!(
                "unknown local baseline '{other}', expected first_allowed/bag_of_words/hash_readout"
            ))),
        }
    }

    fn model_id(self) -> &'static str {
        match self {
            Self::FirstAllowed => "local-first-allowed-baseline",
            Self::BagOfWords => "local-bag-of-words-baseline",
            Self::HashReadout => "local-hash-readout-baseline",
        }
    }
}

pub fn generate_cgs_hard_suite(seed: u64, cases_per_task: usize) -> BenchmarkSuite {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut cases = Vec::new();

    for idx in 0..cases_per_task {
        cases.push(role_filler_case(idx, &mut rng));
        cases.push(long_context_role_filler_case(idx, &mut rng));
        cases.push(active_classification_case(idx, &mut rng));
        cases.push(gridworld_case(idx, &mut rng));
        cases.push(missing_evidence_case(idx, &mut rng));
        cases.push(code_repair_case(idx, &mut rng));
        cases.push(babylm_minimal_pair_case(idx, &mut rng));
        cases.push(blimp_acceptability_case(idx, &mut rng));
        cases.push(long_context_retrieval_case(idx, &mut rng));
        cases.push(adversarial_rag_case(idx, &mut rng));
    }

    BenchmarkSuite {
        name: "cgs_hard_governance".to_string(),
        version: "0.1.0".to_string(),
        generated_at_unix: now_unix(),
        seed,
        cases,
    }
}

pub fn generate_babylm_probe_suite(
    input_path: impl AsRef<Path>,
    seed: u64,
    limit: usize,
) -> Result<BenchmarkSuite> {
    let path = input_path.as_ref();
    let cases = match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "jsonl" => read_minimal_pair_jsonl(path, limit)?,
        "csv" => read_minimal_pair_table(path, ',', limit)?,
        "tsv" => read_minimal_pair_table(path, '\t', limit)?,
        _ => generate_minimal_pairs_from_text(path, seed, limit)?,
    };

    Ok(BenchmarkSuite {
        name: "babylm_local_probe".to_string(),
        version: "0.1.0".to_string(),
        generated_at_unix: now_unix(),
        seed,
        cases,
    })
}

pub fn train_standalone_symliquid(config: StandaloneTrainConfig) -> Result<StandaloneTrainReport> {
    let train_start = Instant::now();
    let train_suite = generate_cgs_hard_suite(config.train_seed, config.cases_per_task);
    let eval_suite = generate_cgs_hard_suite(config.eval_seed, config.cases_per_task);
    let labels = standalone_output_vocab();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.train_seed ^ 0xA53A_5EED);
    let mut readout = LinearReadout::new(config.hv_dim, labels.len(), &mut rng);
    let train_examples =
        prepare_training_examples(&train_suite, &labels, config.hv_dim, standalone_features);

    for _epoch in 0..config.epochs {
        for batch in train_examples.chunks(config.batch_size.max(1)) {
            let (features, targets) = training_batch_tensor(batch, config.hv_dim)?;
            readout.train_batch(&features, &targets, config.lr)?;
        }
    }
    let (train_features, train_targets) = training_batch_tensor(&train_examples, config.hv_dim)?;
    let train_trace = readout.evaluate_batch(&train_features, &train_targets)?;
    let train_runtime_ms = train_start.elapsed().as_millis();
    let trained_examples = train_examples.len() * config.epochs;
    let train_examples_per_second =
        trained_examples as f32 / (train_runtime_ms.max(1) as f32 / 1000.0);

    let eval = evaluate_standalone_model(
        &eval_suite,
        "symliquid-standalone-hash-vsa-readout",
        config.hv_dim,
        &labels,
        &readout,
        config.symbolic_fallback,
    )?;
    if let Some(path) = &config.artifact_path {
        write_readout_artifact(path, &readout, &labels, config.hv_dim, "structured_cgs_vsa")?;
    }

    Ok(StandaloneTrainReport {
        model_id: "symliquid-standalone-hash-vsa-readout".to_string(),
        feature_set: "structured_cgs_vsa".to_string(),
        train_seed: config.train_seed,
        eval_seed: config.eval_seed,
        cases_per_task: config.cases_per_task,
        epochs: config.epochs,
        batch_size: config.batch_size,
        hv_dim: config.hv_dim,
        labels: labels.len(),
        symbolic_fallback: config.symbolic_fallback,
        train_runtime_ms,
        train_examples_per_second,
        train_loss: train_trace.loss,
        train_accuracy: train_trace.accuracy,
        readout_routing: None,
        state_training: None,
        runtime_profile: BTreeMap::new(),
        timing_breakdown_ms: BTreeMap::new(),
        kernel_launches: 0,
        cuda_fallback: false,
        eval,
    })
}

pub fn train_text_hash_baseline(config: StandaloneTrainConfig) -> Result<StandaloneTrainReport> {
    let train_start = Instant::now();
    let train_suite = generate_cgs_hard_suite(config.train_seed, config.cases_per_task);
    let eval_suite = generate_cgs_hard_suite(config.eval_seed, config.cases_per_task);
    let labels = standalone_output_vocab();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.train_seed ^ 0x7E37_BA5E);
    let mut readout = LinearReadout::new(config.hv_dim, labels.len(), &mut rng);
    let train_examples =
        prepare_training_examples(&train_suite, &labels, config.hv_dim, baseline_text_features);

    for _epoch in 0..config.epochs {
        for batch in train_examples.chunks(config.batch_size.max(1)) {
            let (features, targets) = training_batch_tensor(batch, config.hv_dim)?;
            readout.train_batch(&features, &targets, config.lr)?;
        }
    }
    let (train_features, train_targets) = training_batch_tensor(&train_examples, config.hv_dim)?;
    let train_trace = readout.evaluate_batch(&train_features, &train_targets)?;
    let train_runtime_ms = train_start.elapsed().as_millis();
    let trained_examples = train_examples.len() * config.epochs;
    let train_examples_per_second =
        trained_examples as f32 / (train_runtime_ms.max(1) as f32 / 1000.0);

    let eval = evaluate_text_baseline_model(
        &eval_suite,
        "local-text-hash-readout-transfer-baseline",
        config.hv_dim,
        &labels,
        &readout,
    )?;
    if let Some(path) = &config.artifact_path {
        write_readout_artifact(
            path,
            &readout,
            &labels,
            config.hv_dim,
            "text_hash_ngrams_only",
        )?;
    }

    Ok(StandaloneTrainReport {
        model_id: "local-text-hash-readout-transfer-baseline".to_string(),
        feature_set: "text_hash_ngrams_only".to_string(),
        train_seed: config.train_seed,
        eval_seed: config.eval_seed,
        cases_per_task: config.cases_per_task,
        epochs: config.epochs,
        batch_size: config.batch_size,
        hv_dim: config.hv_dim,
        labels: labels.len(),
        symbolic_fallback: false,
        train_runtime_ms,
        train_examples_per_second,
        train_loss: train_trace.loss,
        train_accuracy: train_trace.accuracy,
        readout_routing: None,
        state_training: None,
        runtime_profile: BTreeMap::new(),
        timing_breakdown_ms: BTreeMap::new(),
        kernel_launches: 0,
        cuda_fallback: false,
        eval,
    })
}

#[derive(Debug, Clone)]
pub struct ReadoutTrainingExample {
    pub features: Vec<f32>,
    pub target: usize,
}

pub fn prepare_training_examples(
    suite: &BenchmarkSuite,
    labels: &[String],
    hv_dim: usize,
    feature_fn: fn(&BenchmarkCase, usize) -> Tensor,
) -> Vec<ReadoutTrainingExample> {
    suite
        .cases
        .iter()
        .filter_map(|case| {
            let target = labels.iter().position(|label| label == &case.expected)?;
            let features = feature_fn(case, hv_dim);
            Some(ReadoutTrainingExample {
                features: features.data,
                target,
            })
        })
        .collect()
}

pub fn training_batch_tensor(
    batch: &[ReadoutTrainingExample],
    hv_dim: usize,
) -> Result<(Tensor, Vec<usize>)> {
    let mut data = Vec::with_capacity(batch.len() * hv_dim);
    let mut targets = Vec::with_capacity(batch.len());
    for example in batch {
        data.extend_from_slice(&example.features);
        targets.push(example.target);
    }
    Ok((Tensor::new(batch.len(), hv_dim, data)?, targets))
}

pub fn run_seed_sweep(
    train_seeds: &[u64],
    eval_seed_base: u64,
    mut config: StandaloneTrainConfig,
) -> Result<SeedSweepReport> {
    let mut runs = Vec::with_capacity(train_seeds.len());
    for (idx, seed) in train_seeds.iter().copied().enumerate() {
        config.train_seed = seed;
        config.eval_seed = eval_seed_base + idx as u64;
        runs.push(train_standalone_symliquid(config.clone())?);
    }
    let accuracies = runs
        .iter()
        .map(|run| run.eval.summary.accuracy)
        .collect::<Vec<_>>();
    let residuals = runs
        .iter()
        .map(|run| run.eval.summary.residual)
        .collect::<Vec<_>>();
    Ok(SeedSweepReport {
        train_seeds: train_seeds.to_vec(),
        eval_seed_base,
        cases_per_task: config.cases_per_task,
        epochs: config.epochs,
        batch_size: config.batch_size,
        hv_dim: config.hv_dim,
        lr: config.lr,
        symbolic_fallback: config.symbolic_fallback,
        runs,
        mean_accuracy: mean(&accuracies),
        std_accuracy: stddev(&accuracies),
        mean_residual: mean(&residuals),
        std_residual: stddev(&residuals),
    })
}

pub fn train_babylm_probe_scorer(config: BabyLmProbeTrainConfig) -> Result<BabyLmProbeTrainReport> {
    let train_suite =
        generate_babylm_probe_suite(&config.input_path, config.train_seed, config.train_limit)?;
    let eval_input_path = config
        .eval_input_path
        .as_deref()
        .unwrap_or(&config.input_path);
    let eval_suite =
        generate_babylm_probe_suite(eval_input_path, config.eval_seed, config.eval_limit)?;
    let train_pairs = train_suite
        .cases
        .iter()
        .filter_map(|case| {
            pairwise_training_example(
                case,
                config.hv_dim,
                config.stateful,
                config.pairwise_contrast,
            )
        })
        .collect::<Vec<_>>();
    if train_pairs.is_empty() {
        return Err(SymError::InvalidArgument(
            "BabyLM probe training produced no pairwise examples".to_string(),
        ));
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.train_seed ^ 0xBABA_1A55);
    let mut weights = vec![0.0; config.hv_dim];
    let mut loss_sum = 0.0;
    let mut acc_sum = 0.0;

    let mut order = (0..train_pairs.len()).collect::<Vec<_>>();
    order.shuffle(&mut rng);
    let mut rule_groups = if config.balance_rules {
        build_rule_groups(&train_pairs, &mut rng)
    } else {
        Vec::new()
    };

    for step in 0..config.steps {
        let idx = if config.balance_rules && !rule_groups.is_empty() {
            next_balanced_index(&mut rule_groups, step, &mut rng)
        } else {
            if step > 0 && step % order.len() == 0 {
                order.shuffle(&mut rng);
            }
            order[step % order.len()]
        };
        let example = &train_pairs[idx];
        let margin = dot(&weights, &example.diff) + config.prior_weight * example.prior_delta;
        let probability_a = sigmoid(margin);
        let target = if example.target_a { 1.0 } else { 0.0 };
        let error = probability_a - target;
        let loss = if example.target_a {
            -probability_a.max(1e-8).ln()
        } else {
            -(1.0 - probability_a).max(1e-8).ln()
        };
        for (weight, value) in weights.iter_mut().zip(&example.diff) {
            *weight -= config.lr * error * value;
        }
        loss_sum += loss;
        acc_sum += if (margin >= 0.0) == example.target_a {
            1.0
        } else {
            0.0
        };
    }

    let eval = evaluate_babylm_sequence_scorer(
        &eval_suite,
        &weights,
        config.hv_dim,
        config.stateful,
        config.pairwise_contrast,
        config.prior_weight,
    )?;
    Ok(BabyLmProbeTrainReport {
        policy: default_babylm_probe_policy(),
        methodology: default_babylm_probe_methodology(),
        frontier_family: if config
            .eval_input_path
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase()
            .contains("mutated")
        {
            "babylm_mutated".to_string()
        } else {
            "babylm_local".to_string()
        },
        runner_family: default_babylm_runner_family(),
        external_inference_calls: 0,
        input_path: config.input_path,
        eval_input_path: config.eval_input_path,
        train_seed: config.train_seed,
        eval_seed: config.eval_seed,
        train_limit: train_suite.cases.len(),
        eval_limit: eval_suite.cases.len(),
        steps: config.steps,
        hv_dim: config.hv_dim,
        lr: config.lr,
        stateful: config.stateful,
        pairwise_contrast: config.pairwise_contrast,
        balance_rules: config.balance_rules,
        prior_weight: config.prior_weight,
        train_loss: loss_sum / config.steps.max(1) as f32,
        train_accuracy: acc_sum / config.steps.max(1) as f32,
        eval,
    })
}

fn build_rule_groups(
    train_pairs: &[PairwiseTrainingExample],
    rng: &mut rand::rngs::StdRng,
) -> Vec<RuleTrainingGroup> {
    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for (idx, example) in train_pairs.iter().enumerate() {
        grouped.entry(example.rule.clone()).or_default().push(idx);
    }
    grouped
        .into_values()
        .map(|mut indices| {
            indices.shuffle(rng);
            RuleTrainingGroup { indices, cursor: 0 }
        })
        .collect()
}

fn next_balanced_index(
    groups: &mut [RuleTrainingGroup],
    step: usize,
    rng: &mut rand::rngs::StdRng,
) -> usize {
    let group_idx = step % groups.len();
    let group = &mut groups[group_idx];
    if group.cursor > 0 && group.cursor.is_multiple_of(group.indices.len()) {
        group.indices.shuffle(rng);
    }
    let idx = group.indices[group.cursor % group.indices.len()];
    group.cursor += 1;
    idx
}

pub fn run_local_baseline_suite(
    suite: &BenchmarkSuite,
    baseline: LocalBaselineKind,
    seed: u64,
    hv_dim: usize,
    epochs: usize,
    lr: f32,
) -> Result<BenchmarkReport> {
    match baseline {
        LocalBaselineKind::HashReadout => {
            run_hash_readout_baseline(suite, seed, hv_dim, epochs, lr)
        }
        LocalBaselineKind::FirstAllowed | LocalBaselineKind::BagOfWords => {
            let mut results = Vec::with_capacity(suite.cases.len());
            for case in &suite.cases {
                let start = Instant::now();
                let output = match baseline {
                    LocalBaselineKind::FirstAllowed => first_allowed_output(case),
                    LocalBaselineKind::BagOfWords => bag_of_words_output(case),
                    LocalBaselineKind::HashReadout => unreachable!(),
                };
                let runtime_ms = start.elapsed().as_micros().div_ceil(1000);
                results.push(score_output(
                    suite,
                    case,
                    ModelResponse {
                        case_id: case.id.clone(),
                        model_id: baseline.model_id().to_string(),
                        mode: RunMode::LocalBaseline,
                        output,
                        runtime_ms: Some(runtime_ms),
                        token_count: Some(rough_token_count(&case.observation)),
                        tool_calls: Some(0),
                        estimated_cost_usd: Some(0.0),
                        notes: vec!["Local non-provider baseline.".to_string()],
                    },
                ));
            }
            Ok(BenchmarkReport {
                summary: summarize(suite, baseline.model_id(), RunMode::LocalBaseline, &results),
                results,
            })
        }
    }
}

fn run_hash_readout_baseline(
    suite: &BenchmarkSuite,
    seed: u64,
    hv_dim: usize,
    epochs: usize,
    lr: f32,
) -> Result<BenchmarkReport> {
    let labels = standalone_output_vocab();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xBA5E_711A);
    let mut readout = LinearReadout::new(hv_dim, labels.len(), &mut rng);
    for _ in 0..epochs {
        for case in &suite.cases {
            if let Some(target) = labels.iter().position(|label| label == &case.expected) {
                let features = baseline_text_features(case, hv_dim);
                let _ = readout.train_step(&features, target, lr)?;
            }
        }
    }

    let mut results = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        let start = Instant::now();
        let features = baseline_text_features(case, hv_dim);
        let logits = readout.logits(&features)?;
        let output = masked_prediction(case, logits.row(0), &labels);
        let runtime_ms = start.elapsed().as_micros().div_ceil(1000);
        results.push(score_output(
            suite,
            case,
            ModelResponse {
                case_id: case.id.clone(),
                model_id: LocalBaselineKind::HashReadout.model_id().to_string(),
                mode: RunMode::LocalBaseline,
                output,
                runtime_ms: Some(runtime_ms),
                token_count: Some(rough_token_count(&case.observation)),
                tool_calls: Some(0),
                estimated_cost_usd: Some(0.0),
                notes: vec!["Local text-only hash readout baseline.".to_string()],
            },
        ));
    }
    Ok(BenchmarkReport {
        summary: summarize(
            suite,
            LocalBaselineKind::HashReadout.model_id(),
            RunMode::LocalBaseline,
            &results,
        ),
        results,
    })
}

pub fn write_standalone_train_report(
    path: impl AsRef<Path>,
    report: &StandaloneTrainReport,
) -> Result<()> {
    ensure_parent(path.as_ref())?;
    let json = serde_json::to_string_pretty(report)
        .map_err(|err| SymError::InvalidArgument(format!("serialize train report: {err}")))?;
    fs::write(path, json)
        .map_err(|err| SymError::InvalidArgument(format!("write train report: {err}")))?;
    Ok(())
}

pub fn write_seed_sweep_report(path: impl AsRef<Path>, report: &SeedSweepReport) -> Result<()> {
    ensure_parent(path.as_ref())?;
    let json = serde_json::to_string_pretty(report)
        .map_err(|err| SymError::InvalidArgument(format!("serialize seed sweep: {err}")))?;
    fs::write(path, json)
        .map_err(|err| SymError::InvalidArgument(format!("write seed sweep: {err}")))?;
    Ok(())
}

pub fn write_readout_artifact(
    path: impl AsRef<Path>,
    readout: &LinearReadout,
    labels: &[String],
    hv_dim: usize,
    feature_set: &str,
) -> Result<()> {
    ensure_parent(path.as_ref())?;
    let artifact = ReadoutArtifact {
        labels: labels.to_vec(),
        hv_dim,
        output_dim: readout.output_dim,
        weights: readout.weights.clone(),
        bias: readout.bias.clone(),
        feature_set: feature_set.to_string(),
    };
    let json = serde_json::to_string_pretty(&artifact)
        .map_err(|err| SymError::InvalidArgument(format!("serialize readout artifact: {err}")))?;
    fs::write(path, json)
        .map_err(|err| SymError::InvalidArgument(format!("write readout artifact: {err}")))?;
    Ok(())
}

pub fn write_babylm_probe_train_report(
    path: impl AsRef<Path>,
    report: &BabyLmProbeTrainReport,
) -> Result<()> {
    ensure_parent(path.as_ref())?;
    let json = serde_json::to_string_pretty(report).map_err(|err| {
        SymError::InvalidArgument(format!("serialize BabyLM probe train report: {err}"))
    })?;
    fs::write(path, json).map_err(|err| {
        SymError::InvalidArgument(format!("write BabyLM probe train report: {err}"))
    })?;
    Ok(())
}

pub fn write_suite_json(path: impl AsRef<Path>, suite: &BenchmarkSuite) -> Result<()> {
    ensure_parent(path.as_ref())?;
    let json = serde_json::to_string_pretty(suite)
        .map_err(|err| SymError::InvalidArgument(format!("serialize suite: {err}")))?;
    fs::write(path, json)
        .map_err(|err| SymError::InvalidArgument(format!("write suite: {err}")))?;
    Ok(())
}

pub fn read_suite_json(path: impl AsRef<Path>) -> Result<BenchmarkSuite> {
    let text = fs::read_to_string(path)
        .map_err(|err| SymError::InvalidArgument(format!("read suite: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| SymError::InvalidArgument(format!("parse suite: {err}")))
}

pub fn write_request_template(
    path: impl AsRef<Path>,
    suite: &BenchmarkSuite,
    model_id: &str,
    hybrid: bool,
) -> Result<()> {
    ensure_parent(path.as_ref())?;
    let file = File::create(path)
        .map_err(|err| SymError::InvalidArgument(format!("create template: {err}")))?;
    let mut writer = BufWriter::new(file);
    let mode = if hybrid {
        RunMode::SymLiquidAugmented
    } else {
        RunMode::LocalBaseline
    };

    for case in &suite.cases {
        let observation = if hybrid {
            &case.hybrid_observation
        } else {
            &case.observation
        };
        let request = serde_json::json!({
            "case_id": case.id,
            "task": case.task,
            "model_id": model_id,
            "mode": mode,
            "prompt": benchmark_prompt(case, observation),
            "allowed_actions": case.allowed_actions,
            "expected_kind": case.expected_kind,
            "fill_response_as": {
                "case_id": case.id,
                "model_id": model_id,
                "mode": mode,
                "output": "",
                "runtime_ms": null,
                "token_count": null,
                "tool_calls": null,
                "estimated_cost_usd": null,
                "notes": []
            }
        });
        writeln!(
            writer,
            "{}",
            serde_json::to_string(&request)
                .map_err(|err| SymError::InvalidArgument(format!("serialize request: {err}")))?
        )
        .map_err(|err| SymError::InvalidArgument(format!("write request: {err}")))?;
    }
    Ok(())
}

pub fn run_symliquid_suite(
    suite: &BenchmarkSuite,
    model_id: &str,
    hybrid: bool,
) -> BenchmarkReport {
    let mode = if hybrid {
        RunMode::SymLiquidAugmented
    } else {
        RunMode::SymLiquid
    };
    let mut results = Vec::with_capacity(suite.cases.len());

    for case in &suite.cases {
        let start = Instant::now();
        let output = standalone_symbolic_output(case).unwrap_or_default();
        let runtime_ms = start.elapsed().as_micros().div_ceil(1000);
        results.push(score_output(
            suite,
            case,
            ModelResponse {
                case_id: case.id.clone(),
                model_id: model_id.to_string(),
                mode,
                output,
                runtime_ms: Some(runtime_ms),
                token_count: Some(if hybrid {
                    rough_token_count(&case.hybrid_observation)
                } else {
                    rough_token_count(&case.observation)
                }),
                tool_calls: Some(0),
                estimated_cost_usd: Some(0.0),
                notes: vec!["SymLiquid reference policy over frozen benchmark case.".to_string()],
            },
        ));
    }

    BenchmarkReport {
        summary: summarize(suite, model_id, mode, &results),
        results,
    }
}

pub fn score_responses(
    suite: &BenchmarkSuite,
    responses_path: impl AsRef<Path>,
    fallback_model_id: &str,
    fallback_mode: RunMode,
) -> Result<BenchmarkReport> {
    let responses = read_response_jsonl(responses_path)?;
    let mut by_case = BTreeMap::new();
    for response in responses {
        by_case.insert(response.case_id.clone(), response);
    }

    let mut results = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        let response = by_case
            .get(&case.id)
            .cloned()
            .unwrap_or_else(|| ModelResponse {
                case_id: case.id.clone(),
                model_id: fallback_model_id.to_string(),
                mode: fallback_mode,
                output: String::new(),
                runtime_ms: None,
                token_count: None,
                tool_calls: None,
                estimated_cost_usd: None,
                notes: vec!["Missing response; scored as incorrect.".to_string()],
            });
        results.push(score_output(suite, case, response));
    }

    Ok(BenchmarkReport {
        summary: summarize(suite, fallback_model_id, fallback_mode, &results),
        results,
    })
}

pub fn write_report(path: impl AsRef<Path>, report: &BenchmarkReport) -> Result<()> {
    ensure_parent(path.as_ref())?;
    let json = serde_json::to_string_pretty(report)
        .map_err(|err| SymError::InvalidArgument(format!("serialize report: {err}")))?;
    fs::write(path, json)
        .map_err(|err| SymError::InvalidArgument(format!("write report: {err}")))?;
    Ok(())
}

pub fn read_report(path: impl AsRef<Path>) -> Result<BenchmarkReport> {
    let text = fs::read_to_string(path)
        .map_err(|err| SymError::InvalidArgument(format!("read report: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| SymError::InvalidArgument(format!("parse report: {err}")))
}

pub fn read_report_or_train_eval(path: impl AsRef<Path>) -> Result<BenchmarkReport> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|err| SymError::InvalidArgument(format!("read report: {err}")))?;
    if let Ok(report) = serde_json::from_str::<BenchmarkReport>(&text) {
        return Ok(report);
    }
    let train_report = serde_json::from_str::<StandaloneTrainReport>(&text)
        .map_err(|err| SymError::InvalidArgument(format!("parse report or train eval: {err}")))?;
    Ok(train_report.eval)
}

pub fn compare_reports(
    baseline_name: &str,
    baseline: &BenchmarkReport,
    candidate_name: &str,
    candidate: &BenchmarkReport,
) -> BenchmarkComparison {
    let baseline_eff = baseline.summary.verified_success_per_cost.max(1e-9);
    let candidate_eff = candidate.summary.verified_success_per_cost.max(1e-9);
    BenchmarkComparison {
        baseline_report: baseline_name.to_string(),
        candidate_report: candidate_name.to_string(),
        baseline_model: baseline.summary.model_id.clone(),
        candidate_model: candidate.summary.model_id.clone(),
        baseline_mode: baseline.summary.mode,
        candidate_mode: candidate.summary.mode,
        baseline_accuracy: baseline.summary.accuracy,
        candidate_accuracy: candidate.summary.accuracy,
        accuracy_lift: candidate.summary.accuracy - baseline.summary.accuracy,
        residual_delta: candidate.summary.residual - baseline.summary.residual,
        invalid_action_delta: candidate.summary.invalid_action_rate
            - baseline.summary.invalid_action_rate,
        cost_efficiency_ratio: candidate_eff / baseline_eff,
        token_delta: candidate.summary.total_tokens as isize
            - baseline.summary.total_tokens as isize,
        runtime_delta_ms: candidate.summary.total_runtime_ms as i128
            - baseline.summary.total_runtime_ms as i128,
        memory_delta_bytes: candidate.summary.total_memory_bytes as isize
            - baseline.summary.total_memory_bytes as isize,
    }
}

pub fn write_comparison(path: impl AsRef<Path>, comparison: &BenchmarkComparison) -> Result<()> {
    ensure_parent(path.as_ref())?;
    let json = serde_json::to_string_pretty(comparison)
        .map_err(|err| SymError::InvalidArgument(format!("serialize comparison: {err}")))?;
    fs::write(path, json)
        .map_err(|err| SymError::InvalidArgument(format!("write comparison: {err}")))?;
    Ok(())
}

pub fn write_breakdown_csv(
    path: impl AsRef<Path>,
    suite: &BenchmarkSuite,
    report: &BenchmarkReport,
    group_by: &str,
) -> Result<()> {
    ensure_parent(path.as_ref())?;
    let mut cases_by_id = BTreeMap::new();
    for case in &suite.cases {
        cases_by_id.insert(case.id.as_str(), case);
    }

    #[derive(Default)]
    struct BreakdownGroup {
        task: String,
        group: String,
        cases: usize,
        correct: usize,
        invalid: usize,
        residual_sum: f32,
        runtime_ms: u128,
        tokens: usize,
        memory_bytes: usize,
        failures: Vec<String>,
    }

    let mut groups: BTreeMap<(String, String), BreakdownGroup> = BTreeMap::new();
    for result in &report.results {
        let group = if group_by == "task" {
            result.task.clone()
        } else {
            cases_by_id
                .get(result.case_id.as_str())
                .and_then(|case| case.metadata.get(group_by))
                .cloned()
                .unwrap_or_else(|| "unknown".to_string())
        };
        let entry = groups
            .entry((result.task.clone(), group.clone()))
            .or_insert_with(|| BreakdownGroup {
                task: result.task.clone(),
                group,
                ..BreakdownGroup::default()
            });
        entry.cases += 1;
        entry.correct += usize::from(result.correct);
        entry.invalid += usize::from(result.invalid_action);
        entry.residual_sum += result.residual;
        entry.runtime_ms += result.runtime_ms;
        entry.tokens += result.token_count;
        entry.memory_bytes += result.memory_bytes;
        if !result.correct && entry.failures.len() < 6 {
            entry.failures.push(format!(
                "{} expected={} parsed={}",
                result.case_id, result.expected, result.parsed
            ));
        }
    }

    let file = File::create(path)
        .map_err(|err| SymError::InvalidArgument(format!("create breakdown csv: {err}")))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "task,group_by,group,cases,accuracy,residual,invalid_action_rate,total_runtime_ms,total_tokens,total_memory_bytes,failed_examples"
    )
    .map_err(|err| SymError::InvalidArgument(format!("write breakdown csv: {err}")))?;

    for group in groups.values() {
        let cases = group.cases.max(1);
        let accuracy = group.correct as f32 / cases as f32;
        let residual = group.residual_sum / cases as f32;
        let invalid_rate = group.invalid as f32 / cases as f32;
        let failures = group.failures.join(" | ");
        writeln!(
            writer,
            "{},{},{},{},{:.6},{:.6},{:.6},{},{},{},{}",
            csv_cell(&group.task),
            csv_cell(group_by),
            csv_cell(&group.group),
            group.cases,
            accuracy,
            residual,
            invalid_rate,
            group.runtime_ms,
            group.tokens,
            group.memory_bytes,
            csv_cell(&failures),
        )
        .map_err(|err| SymError::InvalidArgument(format!("write breakdown csv: {err}")))?;
    }
    Ok(())
}

pub fn format_summary(summary: &BenchmarkSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!("Suite: {}\n", summary.suite));
    out.push_str(&format!("Model: {}\n", summary.model_id));
    out.push_str(&format!("Mode: {:?}\n", summary.mode));
    out.push_str(&format!("Cases: {}\n", summary.cases));
    out.push_str(&format!("Accuracy: {:.3}\n", summary.accuracy));
    out.push_str(&format!("Residual: {:.3}\n", summary.residual));
    out.push_str(&format!(
        "Invalid action rate: {:.3}\n",
        summary.invalid_action_rate
    ));
    out.push_str(&format!("Runtime: {} ms\n", summary.total_runtime_ms));
    out.push_str(&format!("Tokens/rough: {}\n", summary.total_tokens));
    out.push_str(&format!("Tool calls: {}\n", summary.total_tool_calls));
    out.push_str(&format!(
        "Estimated cost: ${:.6}\n",
        summary.total_estimated_cost_usd
    ));
    out.push_str(&format!("Memory bytes: {}\n", summary.total_memory_bytes));
    out.push_str(&format!(
        "Verified success per cost: {:.6}\n\n",
        summary.verified_success_per_cost
    ));
    out.push_str("Task breakdown:\n");
    for (task, breakdown) in &summary.task_breakdown {
        out.push_str(&format!(
            "  {:<28} cases={:<4} accuracy={:.3} residual={:.3}\n",
            task, breakdown.cases, breakdown.accuracy, breakdown.residual
        ));
    }
    out
}

pub fn format_standalone_train_report(report: &StandaloneTrainReport) -> String {
    let mut out = String::new();
    out.push_str("Standalone training\n");
    out.push_str(&format!("Model: {}\n", report.model_id));
    out.push_str(&format!("Feature set: {}\n", report.feature_set));
    out.push_str(&format!("Train seed: {}\n", report.train_seed));
    out.push_str(&format!("Eval seed: {}\n", report.eval_seed));
    out.push_str(&format!("Cases per task: {}\n", report.cases_per_task));
    out.push_str(&format!("Epochs: {}\n", report.epochs));
    out.push_str(&format!("Batch size: {}\n", report.batch_size));
    out.push_str(&format!("HV dim: {}\n", report.hv_dim));
    out.push_str(&format!("Labels: {}\n", report.labels));
    out.push_str(&format!(
        "Symbolic fallback: {}\n",
        report.symbolic_fallback
    ));
    out.push_str(&format!("Train runtime: {} ms\n", report.train_runtime_ms));
    out.push_str(&format!(
        "Train examples/sec: {:.1}\n",
        report.train_examples_per_second
    ));
    out.push_str(&format!("Train loss: {:.4}\n", report.train_loss));
    out.push_str(&format!("Train accuracy: {:.3}\n\n", report.train_accuracy));
    if let Some(readout) = &report.readout_routing {
        out.push_str("Readout routing probe\n");
        out.push_str(&format!("Selected: {}\n", readout.selected_readout));
        out.push_str(&format!("Probe cases: {}\n", readout.probe_cases));
        out.push_str(&format!(
            "Shared probe accuracy: {:.3}\n",
            readout.shared_probe_accuracy
        ));
        out.push_str(&format!(
            "Residual-adapter probe accuracy: {:.3}\n",
            readout.residual_adapter_probe_accuracy
        ));
        out.push_str(&format!(
            "Residual-adapter worst task delta: {:+.3}\n",
            readout.residual_adapter_worst_task_accuracy_delta
        ));
        out.push_str(&format!(
            "Residual-adapter rank: {}\n",
            readout.residual_adapter_rank
        ));
        out.push_str(&format!(
            "Accepted residual adapters: {}\n",
            readout.accepted_residual_adapters
        ));
        out.push_str(&format!(
            "Task-head probe accuracy: {:.3}\n",
            readout.task_head_probe_accuracy
        ));
        out.push_str(&format!(
            "Task-head worst task delta: {:+.3}\n",
            readout.task_head_worst_task_accuracy_delta
        ));
        out.push_str(&format!(
            "Accepted task heads: {}\n",
            readout.accepted_task_heads
        ));
        out.push_str(&format!("Acceptance rule: {}\n\n", readout.acceptance_rule));
    }
    if let Some(state) = &report.state_training {
        out.push_str("State training probe\n");
        out.push_str(&format!("Attempted: {}\n", state.attempted));
        out.push_str(&format!("Accepted: {}\n", state.accepted));
        out.push_str(&format!("State epochs: {}\n", state.state_epochs));
        out.push_str(&format!("State LR: {:.5}\n", state.state_lr));
        out.push_str(&format!("Probe cases: {}\n", state.probe_cases));
        out.push_str(&format!(
            "Probe accuracy metric: {}\n",
            state.probe_accuracy_metric
        ));
        out.push_str(&format!(
            "Base probe: accuracy={:.3} loss={:.4}\n",
            state.base_probe_accuracy, state.base_probe_loss
        ));
        out.push_str(&format!(
            "Candidate probe: accuracy={:.3} loss={:.4}\n",
            state.candidate_probe_accuracy, state.candidate_probe_loss
        ));
        out.push_str(&format!(
            "State alignment: base={:.4} candidate={:.4}\n",
            state.base_state_alignment, state.candidate_state_alignment
        ));
        out.push_str(&format!(
            "Worst task accuracy delta: {:+.3}\n",
            state.worst_probe_task_accuracy_delta
        ));
        out.push_str(&format!(
            "Task-gated candidate: {}\n",
            state.task_gated_candidate
        ));
        if !state.candidate_state_tasks.is_empty() {
            out.push_str(&format!(
                "Candidate state tasks: {}\n",
                state.candidate_state_tasks.join(", ")
            ));
        }
        if state.gated_probe_accuracy > 0.0 || state.gated_probe_loss > 0.0 {
            out.push_str(&format!(
                "Gated probe: accuracy={:.3} loss={:.4} worst_task_delta={:+.3}\n",
                state.gated_probe_accuracy,
                state.gated_probe_loss,
                state.gated_worst_task_accuracy_delta
            ));
        }
        if !state.base_probe_task_breakdown.is_empty() {
            out.push_str("Probe task breakdown:\n");
            for (task, base) in &state.base_probe_task_breakdown {
                let candidate = state.candidate_probe_task_breakdown.get(task);
                let candidate_accuracy = candidate.map(|entry| entry.accuracy).unwrap_or(0.0);
                let delta = candidate_accuracy - base.accuracy;
                out.push_str(&format!(
                    "  {:<28} base={:.3} candidate={:.3} delta={:+.3}\n",
                    task, base.accuracy, candidate_accuracy, delta
                ));
            }
        }
        out.push_str(&format!("Acceptance rule: {}\n\n", state.acceptance_rule));
    }
    out.push_str(&format_summary(&report.eval.summary));
    out
}

pub fn format_seed_sweep_report(report: &SeedSweepReport) -> String {
    let mut out = String::new();
    out.push_str("Standalone SymLiquid seed sweep\n");
    out.push_str(&format!("Train seeds: {:?}\n", report.train_seeds));
    out.push_str(&format!("Eval seed base: {}\n", report.eval_seed_base));
    out.push_str(&format!("Cases per task: {}\n", report.cases_per_task));
    out.push_str(&format!("Epochs: {}\n", report.epochs));
    out.push_str(&format!("Batch size: {}\n", report.batch_size));
    out.push_str(&format!("HV dim: {}\n", report.hv_dim));
    out.push_str(&format!("LR: {:.5}\n", report.lr));
    out.push_str(&format!(
        "Symbolic fallback: {}\n",
        report.symbolic_fallback
    ));
    out.push_str(&format!(
        "Accuracy: mean={:.3} std={:.3}\n",
        report.mean_accuracy, report.std_accuracy
    ));
    out.push_str(&format!(
        "Residual: mean={:.3} std={:.3}\n\n",
        report.mean_residual, report.std_residual
    ));
    out.push_str("Runs:\n");
    for run in &report.runs {
        out.push_str(&format!(
            "  train_seed={} eval_seed={} accuracy={:.3} residual={:.3} runtime_ms={}\n",
            run.train_seed,
            run.eval_seed,
            run.eval.summary.accuracy,
            run.eval.summary.residual,
            run.eval.summary.total_runtime_ms
        ));
    }
    out
}

pub fn format_babylm_probe_train_report(report: &BabyLmProbeTrainReport) -> String {
    let mut out = String::new();
    out.push_str("BabyLM local probe sequence-scorer training\n");
    out.push_str(&format!("Input: {}\n", report.input_path));
    if let Some(eval_input_path) = &report.eval_input_path {
        out.push_str(&format!("Eval input: {}\n", eval_input_path));
    }
    out.push_str(&format!("Train seed: {}\n", report.train_seed));
    out.push_str(&format!("Eval seed: {}\n", report.eval_seed));
    out.push_str(&format!("Train pairs: {}\n", report.train_limit));
    out.push_str(&format!("Eval pairs: {}\n", report.eval_limit));
    out.push_str(&format!("Steps: {}\n", report.steps));
    out.push_str(&format!("HV dim: {}\n", report.hv_dim));
    out.push_str(&format!("LR: {:.5}\n", report.lr));
    out.push_str(&format!("Stateful features: {}\n", report.stateful));
    out.push_str(&format!(
        "Pairwise contrast features: {}\n",
        report.pairwise_contrast
    ));
    out.push_str(&format!(
        "Rule-balanced schedule: {}\n",
        report.balance_rules
    ));
    out.push_str(&format!(
        "Grammar prior weight: {:.3}\n",
        report.prior_weight
    ));
    out.push_str(&format!("Train loss: {:.4}\n", report.train_loss));
    out.push_str(&format!("Train accuracy: {:.3}\n\n", report.train_accuracy));
    out.push_str(&format_summary(&report.eval.summary));
    out
}

pub fn format_comparison(comparison: &BenchmarkComparison) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Baseline: {} ({:?})\n",
        comparison.baseline_model, comparison.baseline_mode
    ));
    out.push_str(&format!(
        "Candidate: {} ({:?})\n",
        comparison.candidate_model, comparison.candidate_mode
    ));
    out.push_str(&format!(
        "Accuracy: {:.3} -> {:.3} (lift {:+.3})\n",
        comparison.baseline_accuracy, comparison.candidate_accuracy, comparison.accuracy_lift
    ));
    out.push_str(&format!(
        "Residual delta: {:+.3}\n",
        comparison.residual_delta
    ));
    out.push_str(&format!(
        "Invalid action delta: {:+.3}\n",
        comparison.invalid_action_delta
    ));
    out.push_str(&format!(
        "Cost-efficiency ratio: {:.3}\n",
        comparison.cost_efficiency_ratio
    ));
    out.push_str(&format!("Token delta: {:+}\n", comparison.token_delta));
    out.push_str(&format!(
        "Runtime delta: {:+} ms\n",
        comparison.runtime_delta_ms
    ));
    out.push_str(&format!(
        "Memory delta: {:+} bytes\n",
        comparison.memory_delta_bytes
    ));
    out
}

fn role_filler_case(idx: usize, rng: &mut impl Rng) -> BenchmarkCase {
    let names = [
        "john", "mary", "ivy", "noah", "lena", "omar", "suki", "tess",
    ];
    let verbs = ["loves", "finds", "moves", "keeps"];
    let objects = ["book", "key", "map", "ring", "cup", "lamp"];
    let subject = names[rng.gen_range(0..names.len())];
    let verb = verbs[rng.gen_range(0..verbs.len())];
    let object = objects[rng.gen_range(0..objects.len())];
    let query = ["subject", "verb", "object"][rng.gen_range(0..3)];
    let expected = match query {
        "subject" => subject,
        "verb" => verb,
        _ => object,
    }
    .to_string();
    let observation = format!(
        "Store role-filler facts:\nsubject -> {subject}\nverb -> {verb}\nobject -> {object}\n\nQuery: {query}?\nReturn only the filler."
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid memory trace:\nroles=[subject,verb,object]\nquery_role={query}\nretrieval_mode=vsa_unbind_cleanup\nresidual_warning=low"
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("query_role".to_string(), query.to_string());
    metadata.insert("subject".to_string(), subject.to_string());
    metadata.insert("verb".to_string(), verb.to_string());
    metadata.insert("object".to_string(), object.to_string());
    standard_case(
        format!("role_filler_{idx:04}"),
        "role_filler",
        observation,
        hybrid_observation,
        expected,
        ExpectedKind::Answer,
        Vec::new(),
        metadata,
    )
}

fn long_context_role_filler_case(idx: usize, rng: &mut impl Rng) -> BenchmarkCase {
    let roles = [
        "agent",
        "patient",
        "instrument",
        "location",
        "time",
        "cause",
        "goal",
        "source",
    ];
    let fillers = [
        "atlas", "baker", "copper", "delta", "ember", "fable", "grove", "harbor", "ion", "juniper",
        "kepler", "lumen", "mosaic", "nyx", "opal", "praxis",
    ];
    let mut facts = Vec::new();
    let mut metadata = BTreeMap::new();
    for role in roles {
        let filler = fillers[rng.gen_range(0..fillers.len())];
        facts.push(format!("{role} -> {filler}"));
        metadata.insert(role.to_string(), filler.to_string());
    }
    let query_role = roles[rng.gen_range(0..roles.len())];
    let expected = metadata.get(query_role).cloned().unwrap_or_default();
    let distractors = (0..24)
        .map(|n| {
            format!(
                "distractor_{n:02} -> {}",
                fillers[rng.gen_range(0..fillers.len())]
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let observation = format!(
        "You are given a noisy memory block. Some lines are distractors.\n{distractors}\n{}\n{distractors}\n\nQuery: {query_role}?\nReturn only the filler.",
        facts.join("\n")
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid compact memory trace:\nbound_roles={}\nquery_role={query_role}\nretrieval_mode=vsa_cleanup\nresidual_warning=medium_due_to_distractors",
        roles.join(",")
    );
    metadata.insert("query_role".to_string(), query_role.to_string());
    standard_case(
        format!("long_context_role_filler_{idx:04}"),
        "long_context_role_filler",
        observation,
        hybrid_observation,
        expected,
        ExpectedKind::Answer,
        Vec::new(),
        metadata,
    )
}

fn active_classification_case(idx: usize, rng: &mut impl Rng) -> BenchmarkCase {
    let hidden = rng.gen_range(0..8usize);
    let revealed_feature = rng.gen_range(0..3usize);
    let revealed_value = (hidden >> revealed_feature) & 1;
    let observed = format!("feature_{revealed_feature}={revealed_value}");
    let expected = expected_active_action(revealed_feature);
    let observation = format!(
        "Hidden object has three binary features. Label is majority(feature_0, feature_1, feature_2).\nObserved: {observed}\nAllowed actions: inspect_feature_0, inspect_feature_1, inspect_feature_2, classify_0, classify_1.\nChoose the next action. Return only one allowed action."
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid belief trace:\nknown={observed}\nposterior_entropy=high\nexpected_free_energy_minimizer={expected}\nresidual_warning=do_not_classify_until_uncertainty_reduced"
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("hidden_state".to_string(), hidden.to_string());
    metadata.insert("revealed_feature".to_string(), revealed_feature.to_string());
    standard_case(
        format!("active_classification_{idx:04}"),
        "active_classification",
        observation,
        hybrid_observation,
        expected,
        ExpectedKind::Action,
        vec![
            "inspect_feature_0".to_string(),
            "inspect_feature_1".to_string(),
            "inspect_feature_2".to_string(),
            "classify_0".to_string(),
            "classify_1".to_string(),
        ],
        metadata,
    )
}

fn gridworld_case(idx: usize, rng: &mut impl Rng) -> BenchmarkCase {
    let inspected = rng.gen_bool(0.5);
    let goal = if rng.gen_bool(0.5) {
        "southwest"
    } else {
        "southeast"
    };
    let expected = if inspected {
        "move_south"
    } else {
        "inspect_beacon"
    };
    let hidden_line = if inspected {
        format!("Beacon has revealed the hidden goal: {goal}.")
    } else {
        "Beacon has not been inspected; hidden goal is uncertain.".to_string()
    };
    let observation = format!(
        "You are in a 4x4 grid at row=0 col=0. Possible goals are southwest=(3,0) and southeast=(3,3). {hidden_line}\nAllowed actions: move_north, move_south, move_west, move_east, inspect_beacon.\nChoose the next action. Return only one allowed action."
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid policy trace:\nbelief_hidden_goal={}\nexpected_free_energy_minimizer={expected}\nresidual_warning={}",
        if inspected { goal } else { "ambiguous" },
        if inspected { "low" } else { "high_without_inspection" }
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("inspected".to_string(), inspected.to_string());
    metadata.insert("goal".to_string(), goal.to_string());
    standard_case(
        format!("gridworld_{idx:04}"),
        "gridworld",
        observation,
        hybrid_observation,
        expected.to_string(),
        ExpectedKind::Action,
        vec![
            "move_north".to_string(),
            "move_south".to_string(),
            "move_west".to_string(),
            "move_east".to_string(),
            "inspect_beacon".to_string(),
        ],
        metadata,
    )
}

fn missing_evidence_case(idx: usize, rng: &mut impl Rng) -> BenchmarkCase {
    let topic = ["tide", "orbit", "enzyme", "circuit"][rng.gen_range(0..4)];
    let observation = format!(
        "You must answer a question about {topic}, but the supplied documents only contain partial evidence.\nDocument A: defines the topic.\nDocument B: gives an unrelated example.\nQuestion: Which option is supported, option_alpha or option_beta?\nAllowed actions: answer_alpha, answer_beta, inspect_more.\nReturn only one allowed action."
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid residual trace:\nevidence_support_alpha=missing\nevidence_support_beta=missing\nresidual=high\nexpected_free_energy_minimizer=inspect_more"
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("topic".to_string(), topic.to_string());
    standard_case(
        format!("missing_evidence_{idx:04}"),
        "missing_evidence_rag",
        observation,
        hybrid_observation,
        "inspect_more".to_string(),
        ExpectedKind::Action,
        vec![
            "answer_alpha".to_string(),
            "answer_beta".to_string(),
            "inspect_more".to_string(),
        ],
        metadata,
    )
}

fn code_repair_case(idx: usize, rng: &mut impl Rng) -> BenchmarkCase {
    let bug = rng.gen_range(0..3);
    let (code, failing_test, patch_a, patch_b, patch_c, expected, bug_name) = match bug {
        0 => (
            "fn sum_prefix(xs: &[i32], n: usize) -> i32 {\n    let mut total = 0;\n    for i in 0..=n {\n        total += xs[i];\n    }\n    total\n}",
            "sum_prefix(&[2, 3, 5], 2) should return 5 and must not read xs[2].",
            "change `0..=n` to `0..n`",
            "change `total += xs[i]` to `total -= xs[i]`",
            "change return value to `n as i32`",
            "patch_a",
            "off_by_one_range",
        ),
        1 => (
            "fn count_even(xs: &[i32]) -> usize {\n    let mut count = 1;\n    for x in xs {\n        if x % 2 == 0 { count += 1; }\n    }\n    count\n}",
            "count_even(&[1, 2, 4]) should return 2.",
            "initialize `count` to 0",
            "change `% 2 == 0` to `% 2 != 0`",
            "return `xs.len()`",
            "patch_a",
            "bad_accumulator_seed",
        ),
        _ => (
            "fn contains_token(xs: &[&str], target: &str) -> bool {\n    for x in xs {\n        if *x == target { return false; }\n    }\n    true\n}",
            "contains_token(&[\"red\", \"blue\"], \"blue\") should return true; missing token should return false.",
            "swap the final `true` to `false` only",
            "change `return false` to `return true` and final `true` to `false`",
            "delete the loop and always return true",
            "patch_b",
            "inverted_membership",
        ),
    };
    let observation = format!(
        "Code repair task. Choose the patch that makes the verifier pass.\n\nCode:\n{code}\n\nVerifier:\n{failing_test}\n\nAllowed patches:\npatch_a: {patch_a}\npatch_b: {patch_b}\npatch_c: {patch_c}\n\nReturn only patch_a, patch_b, or patch_c."
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid verification trace:\nresidual_type={bug_name}\nfailing_contract={failing_test}\nexpected_patch={expected}\nresidual_warning=high_until_patch_verified"
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("bug_name".to_string(), bug_name.to_string());
    metadata.insert("failing_test".to_string(), failing_test.to_string());
    standard_case(
        format!("code_repair_{idx:04}"),
        "code_repair_verifier",
        observation,
        hybrid_observation,
        expected.to_string(),
        ExpectedKind::Patch,
        vec![
            "patch_a".to_string(),
            "patch_b".to_string(),
            "patch_c".to_string(),
        ],
        metadata,
    )
}

fn babylm_minimal_pair_case(idx: usize, rng: &mut impl Rng) -> BenchmarkCase {
    let pairs = [
        (
            "The dogs near the tree are loud.",
            "The dogs near the tree is loud.",
            "subject_verb_agreement",
        ),
        (
            "This cookie is sweet.",
            "These cookie is sweet.",
            "determiner_number_agreement",
        ),
        (
            "The boy says that he is hungry.",
            "The boy says that himself is hungry.",
            "pronoun_reflexive_distribution",
        ),
        (
            "A bird can fly over the wall.",
            "A bird can flies over the wall.",
            "modal_verb_form",
        ),
    ];
    let (good, bad, phenomenon) = pairs[rng.gen_range(0..pairs.len())];
    let good_first = rng.gen_bool(0.5);
    let (sentence_a, sentence_b, expected) = if good_first {
        (good, bad, "sentence_a")
    } else {
        (bad, good, "sentence_b")
    };
    let observation = format!(
        "BabyLM-style minimal pair. Which sentence is more grammatical/natural English?\n\nsentence_a: {sentence_a}\nsentence_b: {sentence_b}\n\nReturn only sentence_a or sentence_b."
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid linguistic trace:\nphenomenon={phenomenon}\ncontrast=minimal_pair\nresidual_warning=prefer_sentence_with_consistent_agreement_or_binding"
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("phenomenon".to_string(), phenomenon.to_string());
    metadata.insert("good_sentence".to_string(), good.to_string());
    standard_case(
        format!("babylm_minimal_pair_{idx:04}"),
        "babylm_minimal_pair",
        observation,
        hybrid_observation,
        expected.to_string(),
        ExpectedKind::Action,
        vec!["sentence_a".to_string(), "sentence_b".to_string()],
        metadata,
    )
}

fn blimp_acceptability_case(idx: usize, rng: &mut impl Rng) -> BenchmarkCase {
    let pairs = [
        (
            "What did the chef say the waiter carried?",
            "What did the chef laugh after the waiter carried?",
            "filler_gap_dependency",
        ),
        (
            "The keys to the cabinet are on the table.",
            "The keys to the cabinet is on the table.",
            "agreement_attractor",
        ),
        (
            "No student has ever solved the puzzle.",
            "No student has already solved the puzzle ever.",
            "negative_polarity_distribution",
        ),
        (
            "The teacher who the students admire smiles.",
            "The teacher who the students admire smile.",
            "relative_clause_agreement",
        ),
    ];
    let (acceptable, unacceptable, phenomenon) = pairs[rng.gen_range(0..pairs.len())];
    let acceptable_first = rng.gen_bool(0.5);
    let (sentence_a, sentence_b, expected) = if acceptable_first {
        (acceptable, unacceptable, "sentence_a")
    } else {
        (unacceptable, acceptable, "sentence_b")
    };
    let observation = format!(
        "BLIMP-like acceptability probe. Select the acceptable sentence.\n\nsentence_a: {sentence_a}\nsentence_b: {sentence_b}\n\nReturn only sentence_a or sentence_b."
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid grammar trace:\nphenomenon={phenomenon}\nprobe_type=acceptability_pair\nresidual_warning=watch_for_attractors_or_illicit_dependencies"
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("phenomenon".to_string(), phenomenon.to_string());
    metadata.insert("acceptable_sentence".to_string(), acceptable.to_string());
    standard_case(
        format!("blimp_acceptability_{idx:04}"),
        "blimp_acceptability",
        observation,
        hybrid_observation,
        expected.to_string(),
        ExpectedKind::Action,
        vec!["sentence_a".to_string(), "sentence_b".to_string()],
        metadata,
    )
}

fn long_context_retrieval_case(idx: usize, rng: &mut impl Rng) -> BenchmarkCase {
    let keys = [
        "alpha", "bravo", "cedar", "delta", "ember", "frost", "glade", "harbor",
    ];
    let values = [
        "violet-17",
        "amber-42",
        "silver-09",
        "cobalt-31",
        "jade-64",
        "crimson-28",
        "ivory-73",
        "saffron-55",
    ];
    let target_key = keys[rng.gen_range(0..keys.len())];
    let target_value = values[rng.gen_range(0..values.len())];
    let mut lines = Vec::new();
    for block in 0..12 {
        lines.push(format!(
            "Paragraph {block}: routing note {} maps to {} for the archived experiment.",
            keys[rng.gen_range(0..keys.len())],
            values[rng.gen_range(0..values.len())]
        ));
        if block == 6 {
            lines.push(format!(
                "Paragraph {block}.critical: retrieval key {target_key} has verified value {target_value}."
            ));
        }
    }
    let observation = format!(
        "Long-context retrieval task. Read all paragraphs and return the verified value for key `{target_key}`.\n\n{}\n\nReturn only the value.",
        lines.join("\n")
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid memory trace:\nretrieval_key={target_key}\nkey_position=paragraph_6_critical\ncompressed_value={target_value}\nresidual_warning=ignore_distractor_mappings"
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("target_key".to_string(), target_key.to_string());
    metadata.insert("target_value".to_string(), target_value.to_string());
    standard_case(
        format!("long_context_retrieval_{idx:04}"),
        "long_context_retrieval",
        observation,
        hybrid_observation,
        target_value.to_string(),
        ExpectedKind::Answer,
        Vec::new(),
        metadata,
    )
}

fn adversarial_rag_case(idx: usize, rng: &mut impl Rng) -> BenchmarkCase {
    let topics = [
        "enzyme x7",
        "orbital relay",
        "memory alloy",
        "coastal sensor",
    ];
    let topic = topics[rng.gen_range(0..topics.len())];
    let supported = rng.gen_bool(0.5);
    let alpha_supported = rng.gen_bool(0.5);
    let verified_option = if alpha_supported {
        "answer_alpha"
    } else {
        "answer_beta"
    };
    let expected = if supported {
        verified_option
    } else {
        "inspect_more"
    };
    let evidence = if supported {
        format!(
            "Document A [verified lab note]: For {topic}, {} is explicitly supported.\nDocument B [uncited summary]: For {topic}, the opposite option might be true.\nDocument C [stale note]: Do not use without verification.",
            if alpha_supported {
                "option_alpha"
            } else {
                "option_beta"
            }
        )
    } else {
        format!(
            "Document A [uncited summary]: For {topic}, option_alpha may be true.\nDocument B [stale note]: For {topic}, option_beta may be true.\nDocument C [verified index]: the necessary deciding evidence is absent."
        )
    };
    let observation = format!(
        "Adversarial RAG task. Choose an answer only if verified evidence supports it; otherwise inspect more.\n\n{evidence}\n\nQuestion: Which option is supported for {topic}?\nAllowed actions: answer_alpha, answer_beta, inspect_more.\nReturn only one allowed action."
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid residual trace:\nverified_support={}\nconflict_present=true\nexpected_free_energy_minimizer={expected}\nresidual_warning={}",
        if supported { verified_option } else { "missing" },
        if supported {
            "low_after_verified_document"
        } else {
            "high_missing_deciding_evidence"
        }
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("topic".to_string(), topic.to_string());
    metadata.insert("supported".to_string(), supported.to_string());
    standard_case(
        format!("adversarial_rag_{idx:04}"),
        "adversarial_rag",
        observation,
        hybrid_observation,
        expected.to_string(),
        ExpectedKind::Action,
        vec![
            "answer_alpha".to_string(),
            "answer_beta".to_string(),
            "inspect_more".to_string(),
        ],
        metadata,
    )
}

fn read_minimal_pair_jsonl(path: &Path, limit: usize) -> Result<Vec<BenchmarkCase>> {
    if let Some(cases) = read_minimal_pair_cache(path, limit)? {
        return Ok(cases);
    }
    let file =
        File::open(path).map_err(|err| SymError::InvalidArgument(format!("open jsonl: {err}")))?;
    let reader = BufReader::new(file);
    let mut cases = Vec::new();
    for line in reader.lines() {
        if cases.len() >= limit {
            break;
        }
        let line = line.map_err(|err| SymError::InvalidArgument(format!("read jsonl: {err}")))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .map_err(|err| SymError::InvalidArgument(format!("parse jsonl row: {err}")))?;
        if let Some(case) = minimal_pair_case_from_json(cases.len(), &value) {
            cases.push(case);
        }
    }
    if cases.is_empty() {
        return Err(SymError::InvalidArgument(format!(
            "no minimal-pair rows found in {}",
            path.display()
        )));
    }
    let _ = write_minimal_pair_cache(path, limit, &cases);
    Ok(cases)
}

fn read_minimal_pair_cache(path: &Path, limit: usize) -> Result<Option<Vec<BenchmarkCase>>> {
    let cache_path = minimal_pair_cache_path(path, limit)?;
    if !cache_path.exists() {
        return Ok(None);
    }
    let metadata = path
        .metadata()
        .map_err(|err| SymError::InvalidArgument(format!("stat jsonl: {err}")))?;
    let source_len = metadata.len();
    let source_modified_unix = modified_unix(&metadata)?;
    let text = match fs::read_to_string(&cache_path) {
        Ok(text) => text,
        Err(_) => return Ok(None),
    };
    let cache = match serde_json::from_str::<MinimalPairCache>(&text) {
        Ok(cache) => cache,
        Err(_) => return Ok(None),
    };
    if cache.source_len == source_len
        && cache.source_modified_unix == source_modified_unix
        && cache.limit == limit
        && cache.source_path == path.display().to_string()
        && !cache.cases.is_empty()
    {
        Ok(Some(cache.cases))
    } else {
        Ok(None)
    }
}

fn write_minimal_pair_cache(path: &Path, limit: usize, cases: &[BenchmarkCase]) -> Result<()> {
    let cache_path = minimal_pair_cache_path(path, limit)?;
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| SymError::InvalidArgument(format!("create jsonl cache dir: {err}")))?;
    }
    let metadata = path
        .metadata()
        .map_err(|err| SymError::InvalidArgument(format!("stat jsonl: {err}")))?;
    let cache = MinimalPairCache {
        source_path: path.display().to_string(),
        source_len: metadata.len(),
        source_modified_unix: modified_unix(&metadata)?,
        limit,
        cases: cases.to_vec(),
    };
    let json = serde_json::to_string(&cache)
        .map_err(|err| SymError::InvalidArgument(format!("serialize jsonl cache: {err}")))?;
    fs::write(&cache_path, json)
        .map_err(|err| SymError::InvalidArgument(format!("write jsonl cache: {err}")))?;
    Ok(())
}

fn minimal_pair_cache_path(path: &Path, limit: usize) -> Result<std::path::PathBuf> {
    let metadata = path
        .metadata()
        .map_err(|err| SymError::InvalidArgument(format!("stat jsonl: {err}")))?;
    let source_modified_unix = modified_unix(&metadata)?;
    let mut hasher = Sha256::new();
    hasher.update(path.display().to_string().as_bytes());
    hasher.update(metadata.len().to_le_bytes());
    hasher.update(source_modified_unix.to_le_bytes());
    hasher.update(limit.to_le_bytes());
    let digest = hasher.finalize();
    let cache_name = format!("babylm_minimal_pairs_{:x}_limit_{limit}.json", digest);
    let cache_dir = std::env::var_os("SYMLIQUID_BABYLM_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from("data")
                .join(".cache")
                .join("babylm")
        });
    Ok(cache_dir.join(cache_name))
}

fn modified_unix(metadata: &fs::Metadata) -> Result<u64> {
    Ok(metadata
        .modified()
        .map_err(|err| SymError::InvalidArgument(format!("stat modified time: {err}")))?
        .duration_since(UNIX_EPOCH)
        .map_err(|err| SymError::InvalidArgument(format!("modified time before epoch: {err}")))?
        .as_secs())
}

fn read_minimal_pair_table(
    path: &Path,
    delimiter: char,
    limit: usize,
) -> Result<Vec<BenchmarkCase>> {
    let text = fs::read_to_string(path)
        .map_err(|err| SymError::InvalidArgument(format!("read table: {err}")))?;
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Err(SymError::InvalidArgument(format!(
            "empty minimal-pair table {}",
            path.display()
        )));
    };
    let headers = split_simple_row(header, delimiter)
        .into_iter()
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let mut cases = Vec::new();
    for line in lines {
        if cases.len() >= limit {
            break;
        }
        let fields = split_simple_row(line, delimiter);
        if fields.len() != headers.len() {
            continue;
        }
        let mut obj = serde_json::Map::new();
        for (key, value) in headers.iter().zip(fields) {
            obj.insert(key.clone(), serde_json::Value::String(value));
        }
        if let Some(case) =
            minimal_pair_case_from_json(cases.len(), &serde_json::Value::Object(obj))
        {
            cases.push(case);
        }
    }
    if cases.is_empty() {
        return Err(SymError::InvalidArgument(format!(
            "no minimal-pair rows found in {}",
            path.display()
        )));
    }
    Ok(cases)
}

fn generate_minimal_pairs_from_text(
    path: &Path,
    seed: u64,
    limit: usize,
) -> Result<Vec<BenchmarkCase>> {
    let text = fs::read_to_string(path)
        .map_err(|err| SymError::InvalidArgument(format!("read BabyLM text: {err}")))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut cases = Vec::new();
    for sentence in sentence_candidates(&text) {
        if cases.len() >= limit {
            break;
        }
        if rng.gen_bool(0.35) {
            continue;
        }
        if let Some((positive, negative, rule)) = corrupt_minimal_pair(&sentence) {
            cases.push(make_minimal_pair_case(
                cases.len(),
                &positive,
                &negative,
                rule,
                rng.gen_bool(0.5),
            ));
        }
    }
    if cases.is_empty() {
        return Err(SymError::InvalidArgument(format!(
            "could not derive local BabyLM-style minimal pairs from {}",
            path.display()
        )));
    }
    Ok(cases)
}

fn minimal_pair_case_from_json(idx: usize, value: &serde_json::Value) -> Option<BenchmarkCase> {
    let good = first_json_string(
        value,
        &[
            "sentence_good",
            "good_sentence",
            "positive",
            "grammatical",
            "acceptable",
        ],
    );
    let bad = first_json_string(
        value,
        &[
            "sentence_bad",
            "bad_sentence",
            "negative",
            "ungrammatical",
            "unacceptable",
        ],
    );
    if let (Some(good), Some(bad)) = (good, bad) {
        let rule = first_json_string(value, &["rule", "phenomenon", "uid", "field"])
            .unwrap_or_else(|| "imported_minimal_pair".to_string());
        return Some(make_minimal_pair_case(
            idx,
            &good,
            &bad,
            &rule,
            idx.is_multiple_of(2),
        ));
    }

    let sentence_a = first_json_string(value, &["sentence_a", "sent_a", "A", "a"])?;
    let sentence_b = first_json_string(value, &["sentence_b", "sent_b", "B", "b"])?;
    let answer = first_json_string(value, &["answer", "label", "gold", "target", "expected"])
        .unwrap_or_else(|| "sentence_a".to_string())
        .to_ascii_lowercase();
    let expected = if answer.contains('b') || answer == "1" {
        "sentence_b"
    } else {
        "sentence_a"
    };
    let rule = first_json_string(value, &["rule", "phenomenon", "uid", "field"])
        .unwrap_or_else(|| "imported_minimal_pair".to_string());
    Some(make_ordered_minimal_pair_case(
        idx,
        &sentence_a,
        &sentence_b,
        expected,
        &rule,
    ))
}

fn make_minimal_pair_case(
    idx: usize,
    positive: &str,
    negative: &str,
    rule: &str,
    positive_first: bool,
) -> BenchmarkCase {
    let (sentence_a, sentence_b, expected) = if positive_first {
        (positive, negative, "sentence_a")
    } else {
        (negative, positive, "sentence_b")
    };
    make_ordered_minimal_pair_case(idx, sentence_a, sentence_b, expected, rule)
}

fn make_ordered_minimal_pair_case(
    idx: usize,
    sentence_a: &str,
    sentence_b: &str,
    expected: &str,
    rule: &str,
) -> BenchmarkCase {
    let observation = format!(
        "Local BabyLM/BLIMP-style minimal pair. Select the more grammatical or natural sentence.\n\nsentence_a: {sentence_a}\nsentence_b: {sentence_b}\n\nReturn only sentence_a or sentence_b."
    );
    let hybrid_observation = format!(
        "{observation}\n\nSymLiquid local grammar trace:\nrule={rule}\ncontrast=minimal_pair\nresidual_warning=score_without_external_models"
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("rule".to_string(), rule.to_string());
    metadata.insert("source".to_string(), "local_babylm_bridge".to_string());
    standard_case(
        format!("babylm_local_probe_{idx:05}"),
        "babylm_local_probe",
        observation,
        hybrid_observation,
        expected.to_string(),
        ExpectedKind::Action,
        vec!["sentence_a".to_string(), "sentence_b".to_string()],
        metadata,
    )
}

#[allow(clippy::too_many_arguments)]
fn standard_case(
    id: String,
    task: &str,
    observation: String,
    hybrid_observation: String,
    expected: String,
    expected_kind: ExpectedKind,
    allowed_actions: Vec<String>,
    metadata: BTreeMap<String, String>,
) -> BenchmarkCase {
    BenchmarkCase {
        id,
        task: task.to_string(),
        contract: TaskContract {
            observation_schema: "plain_text_observation".to_string(),
            action_schema: if allowed_actions.is_empty() {
                "return_exact_answer_string".to_string()
            } else {
                "return_one_allowed_action".to_string()
            },
            max_turns: 1,
            max_tokens: 256,
            scoring: "exact_or_allowed_action_match".to_string(),
            fairness_notes: vec![
                "The same observation/action contract is used for SymLiquid and local baselines."
                    .to_string(),
                "Hybrid observations include a SymLiquid trace and must count the extra memory bytes/tokens."
                    .to_string(),
            ],
        },
        observation,
        hybrid_observation,
        expected,
        expected_kind,
        allowed_actions,
        verifier: VerifierSpec {
            name: "exact_match_or_allowed_action".to_string(),
            exact_match: true,
            case_sensitive: false,
            invalid_action_penalty: 1.0,
        },
        metadata,
    }
}

fn standalone_symbolic_output(case: &BenchmarkCase) -> Option<String> {
    let lower = case.observation.to_ascii_lowercase();
    match case.task.as_str() {
        "role_filler" | "long_context_role_filler" => {
            let query = extract_after_marker(&lower, "query:")?;
            extract_arrow_pairs(&lower)
                .into_iter()
                .find_map(|(role, filler)| (role == query).then_some(filler))
        }
        "active_classification" => parse_revealed_feature(&lower).map(expected_active_action),
        "gridworld" => {
            if lower.contains("not been inspected") {
                Some("inspect_beacon".to_string())
            } else if lower.contains("beacon has revealed") {
                Some("move_south".to_string())
            } else {
                None
            }
        }
        "missing_evidence_rag" => Some("inspect_more".to_string()),
        "code_repair_verifier" => {
            if lower.contains("0..=n")
                || lower.contains("off_by_one")
                || lower.contains("let mut count = 1")
            {
                Some("patch_a".to_string())
            } else if lower.contains("return false")
                && lower.contains("missing token should return false")
            {
                Some("patch_b".to_string())
            } else {
                None
            }
        }
        "babylm_minimal_pair" | "blimp_acceptability" | "babylm_local_probe" => {
            let sentence_a = line_after_prefix(&case.observation, "sentence_a:")?;
            let sentence_b = line_after_prefix(&case.observation, "sentence_b:")?;
            if grammatical_score(&sentence_a) >= grammatical_score(&sentence_b) {
                Some("sentence_a".to_string())
            } else {
                Some("sentence_b".to_string())
            }
        }
        "long_context_retrieval" => extract_after_marker(&lower, "has verified value "),
        "adversarial_rag" => adversarial_rag_symbolic_output(&lower),
        _ => None,
    }
}

fn adversarial_rag_symbolic_output(lower: &str) -> Option<String> {
    let mut alpha_verified = false;
    let mut beta_verified = false;
    let mut missing_deciding_evidence = false;

    for line in lower.lines() {
        let trusted = contains_any(
            line,
            &[
                "audited",
                "certified",
                "decisive",
                "gold source",
                "primary source",
                "verified",
                "verification",
            ],
        );
        let support = contains_any(
            line,
            &[
                "confirms",
                "explicitly supported",
                "proves",
                "states",
                "supports",
                "validates",
            ],
        );
        let unsupported = contains_any(
            line,
            &[
                "absent",
                "cannot decide",
                "insufficient",
                "missing",
                "not verified",
                "rumor",
                "stale",
                "uncited",
                "unverified",
                "withheld",
            ],
        );
        if unsupported
            && contains_any(
                line,
                &[
                    "deciding evidence",
                    "decisive evidence",
                    "necessary evidence",
                    "verified evidence",
                ],
            )
        {
            missing_deciding_evidence = true;
        }
        if trusted && support {
            if contains_any(line, &["answer_alpha", "option_alpha", " alpha "]) {
                alpha_verified = true;
            }
            if contains_any(line, &["answer_beta", "option_beta", " beta "]) {
                beta_verified = true;
            }
        }
    }

    match (alpha_verified, beta_verified, missing_deciding_evidence) {
        (true, false, _) => Some("answer_alpha".to_string()),
        (false, true, _) => Some("answer_beta".to_string()),
        (true, true, _) | (false, false, true) => Some("inspect_more".to_string()),
        _ if contains_any(
            lower,
            &[
                "evidence is absent",
                "evidence remains missing",
                "insufficient verified evidence",
                "necessary deciding evidence is absent",
                "no verified source",
            ],
        ) =>
        {
            Some("inspect_more".to_string())
        }
        _ => None,
    }
}

pub fn evaluate_standalone_model(
    suite: &BenchmarkSuite,
    model_id: &str,
    hv_dim: usize,
    labels: &[String],
    readout: &LinearReadout,
    symbolic_fallback: bool,
) -> Result<BenchmarkReport> {
    let mut results = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        let start = Instant::now();
        let features = standalone_features(case, hv_dim);
        let logits = readout.logits(&features)?;
        let pred = symbolic_fallback
            .then(|| standalone_symbolic_output(case))
            .flatten()
            .unwrap_or_else(|| masked_prediction(case, logits.row(0), labels));
        let runtime_ms = start.elapsed().as_micros().div_ceil(1000);
        results.push(score_output(
            suite,
            case,
            ModelResponse {
                case_id: case.id.clone(),
                model_id: model_id.to_string(),
                mode: RunMode::SymLiquid,
                output: pred,
                runtime_ms: Some(runtime_ms),
                token_count: Some(rough_token_count(&case.observation)),
                tool_calls: Some(0),
                estimated_cost_usd: Some(0.0),
                notes: vec![if symbolic_fallback {
                    "Standalone SymLiquid with symbolic governance fallback.".to_string()
                } else {
                    "Standalone SymLiquid learned readout over structured CGS/VSA features."
                        .to_string()
                }],
            },
        ));
    }
    Ok(BenchmarkReport {
        summary: summarize(suite, model_id, RunMode::SymLiquid, &results),
        results,
    })
}

fn evaluate_text_baseline_model(
    suite: &BenchmarkSuite,
    model_id: &str,
    hv_dim: usize,
    labels: &[String],
    readout: &LinearReadout,
) -> Result<BenchmarkReport> {
    let mut results = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        let start = Instant::now();
        let features = baseline_text_features(case, hv_dim);
        let logits = readout.logits(&features)?;
        let pred = masked_prediction(case, logits.row(0), labels);
        let runtime_ms = start.elapsed().as_micros().div_ceil(1000);
        results.push(score_output(
            suite,
            case,
            ModelResponse {
                case_id: case.id.clone(),
                model_id: model_id.to_string(),
                mode: RunMode::LocalBaseline,
                output: pred,
                runtime_ms: Some(runtime_ms),
                token_count: Some(rough_token_count(&case.observation)),
                tool_calls: Some(0),
                estimated_cost_usd: Some(0.0),
                notes: vec!["Local text-only transfer baseline.".to_string()],
            },
        ));
    }
    Ok(BenchmarkReport {
        summary: summarize(suite, model_id, RunMode::LocalBaseline, &results),
        results,
    })
}

fn evaluate_babylm_sequence_scorer(
    suite: &BenchmarkSuite,
    weights: &[f32],
    hv_dim: usize,
    stateful: bool,
    pairwise_contrast: bool,
    prior_weight: f32,
) -> Result<BenchmarkReport> {
    let mut results = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        let start = Instant::now();
        let output = match (
            line_after_prefix(&case.observation, "sentence_a:"),
            line_after_prefix(&case.observation, "sentence_b:"),
        ) {
            (Some(sentence_a_text), Some(sentence_b_text)) => {
                let diff = pairwise_sentence_diff(
                    &sentence_a_text,
                    &sentence_b_text,
                    hv_dim,
                    stateful,
                    pairwise_contrast,
                );
                let prior_delta = sentence_quality_prior(&sentence_a_text)
                    - sentence_quality_prior(&sentence_b_text);
                let margin = dot(weights, &diff) + prior_weight * prior_delta;
                if margin >= 0.0 {
                    "sentence_a".to_string()
                } else {
                    "sentence_b".to_string()
                }
            }
            _ => "sentence_a".to_string(),
        };
        let runtime_ms = start.elapsed().as_micros().div_ceil(1000);
        results.push(score_output(
            suite,
            case,
            ModelResponse {
                case_id: case.id.clone(),
                model_id: "symliquid-babylm-sequence-scorer".to_string(),
                mode: RunMode::SymLiquid,
                output,
                runtime_ms: Some(runtime_ms),
                token_count: Some(rough_token_count(&case.observation)),
                tool_calls: Some(0),
                estimated_cost_usd: Some(0.0),
                notes: vec![
                    "Local pairwise sequence scorer trained on BabyLM-style corruptions."
                        .to_string(),
                ],
            },
        ));
    }
    Ok(BenchmarkReport {
        summary: summarize(
            suite,
            "symliquid-babylm-sequence-scorer",
            RunMode::SymLiquid,
            &results,
        ),
        results,
    })
}

fn pairwise_training_example(
    case: &BenchmarkCase,
    hv_dim: usize,
    stateful: bool,
    pairwise_contrast: bool,
) -> Option<PairwiseTrainingExample> {
    let sentence_a = line_after_prefix(&case.observation, "sentence_a:")?;
    let sentence_b = line_after_prefix(&case.observation, "sentence_b:")?;
    let diff = pairwise_sentence_diff(
        &sentence_a,
        &sentence_b,
        hv_dim,
        stateful,
        pairwise_contrast,
    );
    let prior_delta = sentence_quality_prior(&sentence_a) - sentence_quality_prior(&sentence_b);
    Some(PairwiseTrainingExample {
        diff,
        prior_delta,
        target_a: case.expected == "sentence_a",
        rule: case
            .metadata
            .get("rule")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string()),
    })
}

fn pairwise_sentence_diff(
    sentence_a: &str,
    sentence_b: &str,
    hv_dim: usize,
    stateful: bool,
    pairwise_contrast: bool,
) -> Vec<f32> {
    let sentence_a_features = sentence_features(sentence_a, hv_dim, stateful);
    let sentence_b_features = sentence_features(sentence_b, hv_dim, stateful);
    let mut diff = sentence_a_features;
    for (value, b_value) in diff.iter_mut().zip(sentence_b_features) {
        *value -= b_value;
    }
    if pairwise_contrast {
        add_pairwise_contrast_features(&mut diff, sentence_a, sentence_b);
        normalize_in_place(&mut diff);
    }
    diff
}

pub fn masked_prediction(case: &BenchmarkCase, logits: &[f32], labels: &[String]) -> String {
    if !case.allowed_actions.is_empty() {
        let mut best_label = None;
        let mut best_value = f32::NEG_INFINITY;
        for allowed in &case.allowed_actions {
            if let Some(label_idx) = labels.iter().position(|label| label == allowed) {
                if logits[label_idx] > best_value {
                    best_value = logits[label_idx];
                    best_label = Some(allowed.clone());
                }
            }
        }
        if let Some(label) = best_label {
            return label;
        }
    }
    if matches!(case.expected_kind, ExpectedKind::Answer) {
        let mut best_label = None;
        let mut best_value = f32::NEG_INFINITY;
        for label_idx in answer_candidate_indices(case, labels) {
            if logits[label_idx] > best_value {
                best_value = logits[label_idx];
                best_label = Some(labels[label_idx].clone());
            }
        }
        if let Some(label) = best_label {
            return label;
        }
    }
    labels
        .get(argmax(logits))
        .cloned()
        .unwrap_or_else(String::new)
}

fn answer_candidate_indices(case: &BenchmarkCase, labels: &[String]) -> Vec<usize> {
    let lower = case.observation.to_ascii_lowercase();
    let tokens = tokenize(&case.observation);
    labels
        .iter()
        .enumerate()
        .filter_map(|(idx, label)| {
            let label_lower = label.to_ascii_lowercase();
            (lower.contains(&label_lower) || tokens.iter().any(|token| token == &label_lower))
                .then_some(idx)
        })
        .collect()
}

pub fn standalone_features(case: &BenchmarkCase, hv_dim: usize) -> Tensor {
    case_features(case, hv_dim, true)
}

fn baseline_text_features(case: &BenchmarkCase, hv_dim: usize) -> Tensor {
    case_features(case, hv_dim, false)
}

fn case_features(case: &BenchmarkCase, hv_dim: usize, structured: bool) -> Tensor {
    let mut features = vec![0.0; hv_dim];
    add_feature(&mut features, "bias", 1.0);
    add_text_features(&mut features, &case.observation, "obs");
    for action in &case.allowed_actions {
        add_feature(&mut features, &format!("allowed:{action}"), 0.25);
    }
    if structured {
        add_structured_text_features(&mut features, &case.observation);
    }
    normalize_in_place(&mut features);
    Tensor::from_row(features)
}

fn add_text_features(features: &mut [f32], text: &str, prefix: &str) {
    let tokens = tokenize(text);
    for token in &tokens {
        add_feature(features, &format!("{prefix}:tok:{token}"), 1.0);
    }
    for pair in tokens.windows(2) {
        add_feature(
            features,
            &format!("{prefix}:bigram:{}_{}", pair[0], pair[1]),
            0.6,
        );
    }
    for triple in tokens.windows(3) {
        add_feature(
            features,
            &format!("{prefix}:trigram:{}_{}_{}", triple[0], triple[1], triple[2]),
            0.35,
        );
    }
}

fn add_structured_text_features(features: &mut [f32], text: &str) {
    let lower = text.to_lowercase();
    if let Some(query) = extract_after_marker(&lower, "query:") {
        add_feature(features, &format!("query:{query}"), 1.0);
        for (role, filler) in extract_arrow_pairs(&lower) {
            add_feature(features, &format!("pair:{role}->{filler}"), 1.0);
            add_feature(features, &format!("memory_value:{filler}"), 0.5);
            if query.contains(&role) || role.contains(&query) {
                add_feature(features, &format!("query_pair:{query}->{filler}"), 3.0);
                add_feature(features, &format!("query_answer:{filler}"), 5.0);
            }
        }
    }
    if let Some(key) = extract_between(&lower, "key `", "`") {
        add_feature(features, &format!("retrieval_key:{key}"), 2.0);
        if let Some(value) = extract_after_marker(&lower, "has verified value ") {
            add_feature(features, &format!("retrieval_pair:{key}->{value}"), 4.0);
            add_feature(features, &format!("retrieval_answer:{value}"), 5.0);
        }
    }
    if lower.contains("verified lab note") {
        add_feature(features, "rag:verified_lab_note", 2.0);
    }
    if lower.contains("deciding evidence is absent")
        || lower.contains("necessary deciding evidence is absent")
    {
        add_feature(features, "rag:evidence_absent", 3.0);
    }
    if lower.contains("option_alpha is explicitly supported") {
        add_feature(features, "rag:alpha_verified", 4.0);
    }
    if lower.contains("option_beta is explicitly supported") {
        add_feature(features, "rag:beta_verified", 4.0);
    }
    if lower.contains("off_by_one") || lower.contains("0..=n") {
        add_feature(features, "code:off_by_one", 3.0);
    }
    if lower.contains("let mut count = 1") {
        add_feature(features, "code:bad_accumulator_seed", 3.0);
    }
    if lower.contains("return false") && lower.contains("missing token should return false") {
        add_feature(features, "code:inverted_membership", 3.0);
    }
    if let (Some(sentence_a), Some(sentence_b)) = (
        line_after_prefix(text, "sentence_a:"),
        line_after_prefix(text, "sentence_b:"),
    ) {
        let delta = grammatical_score(&sentence_a) - grammatical_score(&sentence_b);
        if delta > 0 {
            add_feature(features, "pairwise_acceptability:sentence_a", 5.0);
        } else if delta < 0 {
            add_feature(features, "pairwise_acceptability:sentence_b", 5.0);
        } else {
            add_feature(features, "pairwise_acceptability:ambiguous", 0.5);
        }
    }
}

fn add_feature(features: &mut [f32], key: &str, value: f32) {
    let (idx, sign) = hash_feature(key, features.len());
    features[idx] += sign * value;
}

fn hash_feature(key: &str, dim: usize) -> (usize, f32) {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let digest = hasher.finalize();
    let mut index_bytes = [0u8; 8];
    index_bytes.copy_from_slice(&digest[..8]);
    let raw = u64::from_le_bytes(index_bytes);
    let idx = raw as usize % dim.max(1);
    let sign = if digest[8] & 1 == 0 { 1.0 } else { -1.0 };
    (idx, sign)
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn extract_after_marker(text: &str, marker: &str) -> Option<String> {
    let (_, rest) = text.split_once(marker)?;
    let value = rest
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-')
        .to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn extract_between(text: &str, left: &str, right: &str) -> Option<String> {
    let (_, rest) = text.split_once(left)?;
    let (value, _) = rest.split_once(right)?;
    Some(value.trim().to_string()).filter(|value| !value.is_empty())
}

fn extract_arrow_pairs(text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for line in text.lines() {
        if let Some((role, filler)) = line.split_once("->") {
            let role = role
                .split_whitespace()
                .last()
                .unwrap_or_default()
                .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
                .to_string();
            let filler = filler
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-')
                .to_string();
            if !role.is_empty() && !filler.is_empty() {
                pairs.push((role, filler));
            }
        }
    }
    pairs
}

fn parse_revealed_feature(text: &str) -> Option<usize> {
    let (_, rest) = text.split_once("observed: feature_")?;
    let idx = rest.chars().next()?.to_digit(10)? as usize;
    (idx < 3).then_some(idx)
}

fn line_after_prefix(text: &str, prefix: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let trimmed = line.trim();
        trimmed
            .strip_prefix(prefix)
            .map(str::trim)
            .map(str::to_string)
            .filter(|value| !value.is_empty())
    })
}

fn grammatical_score(sentence: &str) -> i32 {
    let lower = sentence.to_ascii_lowercase();
    let mut score = 0;
    let penalties = [
        "dogs near the tree is",
        "these cookie",
        "himself is hungry",
        "can flies",
        "laugh after",
        "keys to the cabinet is",
        "already solved the puzzle ever",
        "students admire smile",
    ];
    let rewards = [
        "dogs near the tree are",
        "this cookie",
        "he is hungry",
        "can fly",
        "waiter carried",
        "keys to the cabinet are",
        "has ever solved",
        "students admire smiles",
    ];
    for penalty in penalties {
        if lower.contains(penalty) {
            score -= 2;
        }
    }
    for reward in rewards {
        if lower.contains(reward) {
            score += 2;
        }
    }
    score
}

fn split_simple_row(line: &str, delimiter: char) -> Vec<String> {
    line.split(delimiter)
        .map(|value| value.trim().trim_matches('"').to_string())
        .collect()
}

fn first_json_string(value: &serde_json::Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = value.get(*key).and_then(|value| value.as_str()) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn sentence_candidates(text: &str) -> Vec<String> {
    text.split(['.', '?', '!'])
        .map(str::trim)
        .filter(|sentence| {
            let words = tokenize(sentence);
            (5..=32).contains(&words.len())
        })
        .map(|sentence| format!("{sentence}."))
        .collect()
}

fn corrupt_minimal_pair(sentence: &str) -> Option<(String, String, &'static str)> {
    let replacements = [
        (" is ", " are ", "agreement_is_are"),
        (" are ", " is ", "agreement_are_is"),
        (" was ", " were ", "agreement_was_were"),
        (" were ", " was ", "agreement_were_was"),
        (" has ", " have ", "agreement_has_have"),
        (" have ", " has ", "agreement_have_has"),
        (" does ", " do ", "agreement_does_do"),
        (" do ", " does ", "agreement_do_does"),
        (" this ", " these ", "determiner_this_these"),
        (" these ", " this ", "determiner_these_this"),
        (" that ", " those ", "determiner_that_those"),
        (" those ", " that ", "determiner_those_that"),
        (" can fly ", " can flies ", "modal_can_fly"),
        (" can go ", " can goes ", "modal_can_go"),
    ];
    let padded = format!(" {sentence} ");
    let lower = padded.to_ascii_lowercase();
    for (from, to, rule) in replacements {
        if let Some(start) = lower.find(from) {
            let mut corrupted = padded.clone();
            corrupted.replace_range(start..start + from.len(), to);
            return Some((
                sentence.trim().to_string(),
                corrupted.trim().to_string(),
                rule,
            ));
        }
    }
    None
}

fn first_allowed_output(case: &BenchmarkCase) -> String {
    case.allowed_actions.first().cloned().unwrap_or_default()
}

fn bag_of_words_output(case: &BenchmarkCase) -> String {
    let lower = case.observation.to_ascii_lowercase();
    if !case.allowed_actions.is_empty() {
        if lower.contains("evidence is absent")
            || lower.contains("partial evidence")
            || lower.contains("otherwise inspect more")
        {
            return "inspect_more".to_string();
        }
        if lower.contains("not been inspected") {
            return "inspect_beacon".to_string();
        }
        if lower.contains("0..=n") || lower.contains("let mut count = 1") {
            return "patch_a".to_string();
        }
        if lower.contains("return false") && lower.contains("missing token should return false") {
            return "patch_b".to_string();
        }
        if lower.contains("sentence_a:") && lower.contains("sentence_b:") {
            return "sentence_a".to_string();
        }
        return first_allowed_output(case);
    }
    if let Some(query) = extract_after_marker(&lower, "query:") {
        if let Some((_role, filler)) = extract_arrow_pairs(&lower)
            .into_iter()
            .find(|(role, _)| role == &query)
        {
            return filler;
        }
    }
    String::new()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn stddev(values: &[f32]) -> f32 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / (values.len() - 1) as f32;
    variance.sqrt()
}

pub fn standalone_output_vocab() -> Vec<String> {
    let mut labels = vec![
        "inspect_feature_0",
        "inspect_feature_1",
        "inspect_feature_2",
        "classify_0",
        "classify_1",
        "move_north",
        "move_south",
        "move_west",
        "move_east",
        "inspect_beacon",
        "answer_alpha",
        "answer_beta",
        "inspect_more",
        "patch_a",
        "patch_b",
        "patch_c",
        "sentence_a",
        "sentence_b",
        "john",
        "mary",
        "ivy",
        "noah",
        "lena",
        "omar",
        "suki",
        "tess",
        "loves",
        "finds",
        "moves",
        "keeps",
        "book",
        "key",
        "map",
        "ring",
        "cup",
        "lamp",
        "atlas",
        "baker",
        "copper",
        "delta",
        "ember",
        "fable",
        "grove",
        "harbor",
        "ion",
        "juniper",
        "kepler",
        "lumen",
        "mosaic",
        "nyx",
        "opal",
        "praxis",
        "violet-17",
        "amber-42",
        "silver-09",
        "cobalt-31",
        "jade-64",
        "crimson-28",
        "ivory-73",
        "saffron-55",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    labels.sort();
    labels.dedup();
    labels
}

fn expected_active_action(revealed_feature: usize) -> String {
    match revealed_feature {
        0 => "inspect_feature_1",
        1 => "inspect_feature_0",
        _ => "inspect_feature_0",
    }
    .to_string()
}

fn benchmark_prompt(case: &BenchmarkCase, observation: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("You are being evaluated under a strict benchmark contract.\n");
    prompt.push_str("Return only the requested answer or one allowed action. No explanation.\n\n");
    prompt.push_str("Contract:\n");
    prompt.push_str(&format!(
        "- observation_schema: {}\n",
        case.contract.observation_schema
    ));
    prompt.push_str(&format!(
        "- action_schema: {}\n",
        case.contract.action_schema
    ));
    prompt.push_str(&format!("- max_turns: {}\n", case.contract.max_turns));
    if !case.allowed_actions.is_empty() {
        prompt.push_str(&format!(
            "- allowed_actions: {}\n",
            case.allowed_actions.join(", ")
        ));
    }
    prompt.push_str("\nObservation:\n");
    prompt.push_str(observation);
    prompt
}

pub fn score_output(
    suite: &BenchmarkSuite,
    case: &BenchmarkCase,
    response: ModelResponse,
) -> CaseResult {
    let parsed = parse_output(case, &response.output);
    let invalid_action = matches!(
        case.expected_kind,
        ExpectedKind::Action | ExpectedKind::Patch
    ) && !case
        .allowed_actions
        .iter()
        .any(|action| action.eq_ignore_ascii_case(&parsed));
    let correct = !invalid_action && equals_for_verifier(&case.expected, &parsed, &case.verifier);
    let score = if correct { 1.0 } else { 0.0 };
    let residual = 1.0 - score;
    let token_count = response
        .token_count
        .unwrap_or_else(|| rough_token_count(&response.output));
    let tool_calls = response.tool_calls.unwrap_or(0);
    let runtime_ms = response.runtime_ms.unwrap_or(0);
    let estimated_cost_usd = response.estimated_cost_usd.unwrap_or(0.0);
    let memory_bytes = match response.mode {
        RunMode::SymLiquidAugmented => case.hybrid_observation.len(),
        _ => case.observation.len(),
    };

    CaseResult {
        case_id: case.id.clone(),
        task: case.task.clone(),
        model_id: response.model_id.clone(),
        mode: response.mode,
        expected: case.expected.clone(),
        output: response.output,
        parsed,
        correct,
        invalid_action,
        score,
        residual,
        runtime_ms,
        token_count,
        tool_calls,
        estimated_cost_usd,
        memory_bytes,
        cache_key: cache_key(suite, case, &response.model_id, response.mode),
        notes: response.notes,
    }
}

fn parse_output(case: &BenchmarkCase, output: &str) -> String {
    let cleaned = output
        .trim()
        .trim_matches('`')
        .trim_matches('"')
        .trim()
        .to_string();
    if matches!(
        case.expected_kind,
        ExpectedKind::Action | ExpectedKind::Patch
    ) {
        let lower = cleaned.to_lowercase();
        for action in &case.allowed_actions {
            if lower == action.to_lowercase() || lower.contains(&action.to_lowercase()) {
                return action.clone();
            }
        }
    }
    cleaned
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .to_lowercase()
}

fn equals_for_verifier(expected: &str, parsed: &str, verifier: &VerifierSpec) -> bool {
    if verifier.case_sensitive {
        expected == parsed
    } else {
        expected.eq_ignore_ascii_case(parsed)
    }
}

pub fn summarize(
    suite: &BenchmarkSuite,
    model_id: &str,
    mode: RunMode,
    results: &[CaseResult],
) -> BenchmarkSummary {
    let cases = results.len().max(1);
    let correct = results.iter().filter(|result| result.correct).count();
    let invalid = results
        .iter()
        .filter(|result| result.invalid_action)
        .count();
    let total_runtime_ms = results.iter().map(|result| result.runtime_ms).sum();
    let total_tokens = results.iter().map(|result| result.token_count).sum();
    let total_tool_calls = results.iter().map(|result| result.tool_calls).sum();
    let total_estimated_cost_usd = results
        .iter()
        .map(|result| result.estimated_cost_usd)
        .sum::<f32>();
    let total_memory_bytes = results.iter().map(|result| result.memory_bytes).sum();
    let mut grouped: BTreeMap<String, Vec<&CaseResult>> = BTreeMap::new();
    for result in results {
        grouped.entry(result.task.clone()).or_default().push(result);
    }
    let mut task_breakdown = BTreeMap::new();
    for (task, task_results) in grouped {
        let task_cases = task_results.len().max(1);
        let task_correct = task_results.iter().filter(|result| result.correct).count();
        let accuracy = task_correct as f32 / task_cases as f32;
        task_breakdown.insert(
            task,
            TaskBreakdown {
                cases: task_cases,
                accuracy,
                residual: 1.0 - accuracy,
            },
        );
    }
    let accuracy = correct as f32 / cases as f32;
    let cost_denominator = 1.0
        + total_tokens as f32
        + total_tool_calls as f32 * 100.0
        + total_runtime_ms as f32
        + total_memory_bytes as f32 / 1024.0
        + total_estimated_cost_usd * 1_000_000.0;

    BenchmarkSummary {
        suite: suite.name.clone(),
        model_id: model_id.to_string(),
        mode,
        cases,
        accuracy,
        residual: 1.0 - accuracy,
        invalid_action_rate: invalid as f32 / cases as f32,
        total_runtime_ms,
        total_tokens,
        total_tool_calls,
        total_estimated_cost_usd,
        total_memory_bytes,
        verified_success_per_cost: correct as f32 / cost_denominator,
        task_breakdown,
    }
}

fn read_response_jsonl(path: impl AsRef<Path>) -> Result<Vec<ModelResponse>> {
    let file = File::open(path)
        .map_err(|err| SymError::InvalidArgument(format!("open responses: {err}")))?;
    let reader = BufReader::new(file);
    let mut responses = Vec::new();
    for (line_idx, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|err| SymError::InvalidArgument(format!("read response: {err}")))?;
        if line.trim().is_empty() {
            continue;
        }
        let response = serde_json::from_str(&line).map_err(|err| {
            SymError::InvalidArgument(format!("parse response line {}: {err}", line_idx + 1))
        })?;
        responses.push(response);
    }
    Ok(responses)
}

fn cache_key(
    suite: &BenchmarkSuite,
    case: &BenchmarkCase,
    model_id: &str,
    mode: RunMode,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(suite.name.as_bytes());
    hasher.update(suite.version.as_bytes());
    hasher.update(case.id.as_bytes());
    hasher.update(case.observation.as_bytes());
    hasher.update(case.hybrid_observation.as_bytes());
    hasher.update(model_id.as_bytes());
    hasher.update(format!("{mode:?}").as_bytes());
    format!("{:x}", hasher.finalize())
}

fn rough_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn csv_cell(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| SymError::InvalidArgument(format!("create parent: {err}")))?;
        }
    }
    Ok(())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
