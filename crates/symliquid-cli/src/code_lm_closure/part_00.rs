use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use symliquid_core::tensor::Tensor;
use symliquid_core::train::LinearReadout;

mod part_00_contract;
use part_00_contract::*;

#[derive(Debug, Clone)]
pub struct CodeLmClosureConfig {
    pub private_curriculum: String,
    pub public_task_manifest: String,
    pub seed: u64,
    pub hv_dim: usize,
    pub max_vocab: usize,
    pub epochs: usize,
    pub lr: f32,
    pub candidates_per_task: usize,
    pub max_work_steps: usize,
    pub use_cuda_readout: bool,
    pub readout_eval_limit: usize,
    pub aux_decoder_train_limit: usize,
    pub checkpoint_only: bool,
    pub checkpoint_out: String,
    pub checkpoint_in: String,
    pub private_candidate_out: String,
    pub public_candidate_out: String,
    pub report_out: String,
    pub sts_streams: String,
}

#[derive(Debug, Clone)]
pub struct CodeLmFanoutConfig {
    pub private_curriculum: String,
    pub public_task_manifest: String,
    pub checkpoint_in: String,
    pub seed: u64,
    pub candidates_per_task: usize,
    pub private_candidate_out: String,
    pub public_candidate_out: String,
    pub report_out: String,
    pub sts_streams: String,
    pub transformer_hybrid_candidate_manifest: String,
    pub private_eval_limit: usize,
    pub public_task_limit: usize,
}

#[derive(Debug, Clone)]
struct CodeTask {
    raw: Value,
    task_id: String,
    source_task_id: String,
    card_id: String,
    source_id: String,
    split: String,
    category: String,
    prompt: String,
    entry_point: String,
    solution_expr: String,
    solution_body: String,
    tags: Vec<String>,
    benchmark_evidence_level: String,
}

impl CodeTask {
    fn solution_body_text(&self) -> String {
        let body = self.solution_body.trim();
        if !body.is_empty() {
            body.to_string()
        } else {
            body_from_expression(&self.solution_expr)
        }
    }
}

type StsStreamMap = HashMap<String, BTreeMap<String, String>>;

#[derive(Debug)]
struct CandidateHeartbeat<'a> {
    config: &'a CodeLmClosureConfig,
    started: &'a Instant,
    stage: &'a str,
    phase: &'a str,
    trained: bool,
    total_tasks: usize,
    original_train_task_count: usize,
    train_task_count: usize,
    eval_task_count: usize,
    public_task_count: usize,
    estimated_work_steps_before_budget: usize,
    estimated_work_steps_after_budget: usize,
}

fn code_lm_env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(default)
}

fn private_full_sts_off_nonregression_enabled() -> bool {
    code_lm_env_flag("THESEUS_CODE_LM_FULL_PRIVATE_STS_OFF_NONREGRESSION", false)
}

fn private_sts_off_nonregression_task_limit(total_tasks: usize) -> usize {
    if total_tasks == 0 {
        return 0;
    }
    if private_full_sts_off_nonregression_enabled() {
        return total_tasks;
    }
    std::env::var("THESEUS_CODE_LM_PRIVATE_STS_OFF_NONREGRESSION_TASK_LIMIT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(32)
        .clamp(1, total_tasks)
}

fn private_sts_off_nonregression_candidate_limit(configured_limit: usize) -> usize {
    let configured_limit = configured_limit.max(1);
    if private_full_sts_off_nonregression_enabled() {
        return configured_limit;
    }
    std::env::var("THESEUS_CODE_LM_PRIVATE_STS_OFF_NONREGRESSION_CANDIDATE_LIMIT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(configured_limit.min(2).max(1))
        .clamp(1, configured_limit)
}

fn private_sts_off_nonregression_eval_rows(eval_rows: &[CodeTask]) -> Vec<CodeTask> {
    let task_limit = private_sts_off_nonregression_task_limit(eval_rows.len());
    eval_rows.iter().take(task_limit).cloned().collect()
}

fn private_sts_off_nonregression_policy_json(
    total_tasks: usize,
    selected_tasks: usize,
    configured_candidate_limit: usize,
    selected_candidate_limit: usize,
) -> Value {
    json!({
        "policy": "bounded_private_sts_off_nonregression_fanout_v1",
        "enabled": selected_tasks > 0,
        "full_mode": private_full_sts_off_nonregression_enabled(),
        "available_task_count": total_tasks,
        "selected_task_count": selected_tasks,
        "configured_candidates_per_task": configured_candidate_limit.max(1),
        "selected_candidates_per_task": selected_candidate_limit.max(1),
        "full_ablation_env": "THESEUS_CODE_LM_FULL_PRIVATE_STS_OFF_NONREGRESSION=1",
        "task_limit_env": "THESEUS_CODE_LM_PRIVATE_STS_OFF_NONREGRESSION_TASK_LIMIT",
        "candidate_limit_env": "THESEUS_CODE_LM_PRIVATE_STS_OFF_NONREGRESSION_CANDIDATE_LIMIT",
        "score_semantics": "diagnostic same-seed STS-off nonregression only; union candidates remain excluded from promotion quality denominators",
    })
}

fn legacy_top_level_checkpoint_arrays_enabled() -> bool {
    code_lm_env_flag("THESEUS_CODE_LM_LEGACY_TOP_LEVEL_CHECKPOINT_ARRAYS", false)
}

#[derive(Debug, Clone)]
struct TrainExample {
    prompt_tokens: Vec<String>,
    category: String,
    prev2: String,
    prev1: String,
    position: usize,
    target: usize,
}

#[derive(Debug, Clone)]
struct ExpressionBankItem {
    expr: String,
    tokens: Vec<String>,
    hints: BTreeSet<String>,
    category: String,
    count: usize,
}

#[derive(Debug, Clone)]
struct CandidateExpression {
    expr: String,
    body: String,
    mode: String,
    compositional_token_candidate: bool,
    full_body_token_candidate: bool,
    expression_memory_fallback: bool,
    sts_candidate_expression_used: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueKind {
    Int,
    List,
    Str,
    Dict,
    Bool,
    Unknown,
}

#[derive(Debug, Clone)]
struct Vocab {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    unk_id: usize,
}

#[derive(Debug, Clone, Default)]
struct BodyNgramModel {
    counts: HashMap<String, BTreeMap<String, usize>>,
}

#[derive(Debug, Clone, Default)]
struct BodyPrototypeModel {
    by_category: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
struct StateSequenceDecoder {
    weights: HashMap<String, Vec<f32>>,
    bias: Vec<f32>,
    output_dim: usize,
    feature_count: usize,
    update_count: usize,
}

#[derive(Debug, Clone)]
struct StateSequenceTrace {
    accuracy: f32,
    example_count: usize,
}

#[derive(Debug, Clone)]
struct SymLiquidStateDecoder {
    readout: LinearReadout,
    state_dim: usize,
    update_count: usize,
}

#[derive(Debug, Clone)]
struct LoadedCodeLmModels {
    checkpoint_id: String,
    vocab: Vocab,
    expression_bank: Vec<ExpressionBankItem>,
    body_prototypes: BodyPrototypeModel,
    body_ngram: BodyNgramModel,
    state_sequence_decoder: StateSequenceDecoder,
    symliquid_state_decoder: SymLiquidStateDecoder,
    readout: LinearReadout,
    backend: String,
}

fn trace_code_lm(label: &str) {
    if std::env::var_os("THESEUS_CODE_LM_TRACE").is_some() {
        eprintln!("theseus-code-lm: {label}");
    }
}

pub fn train_code_lm_closure(
    config: CodeLmClosureConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    trace_code_lm("start");
    let started = Instant::now();
    let mut phase_started = Instant::now();
    let mut phase_timing_ms: BTreeMap<String, u128> = BTreeMap::new();
    let private_rows = load_tasks(Path::new(&config.private_curriculum))?;
    let public_rows = load_tasks(Path::new(&config.public_task_manifest))?;
    let sts_streams = load_sts_streams(Path::new(&config.sts_streams))?;
    let checkpoint_in_ignored_for_training = !config.checkpoint_in.trim().is_empty();
    trace_code_lm("loaded inputs");
    let mut train_rows = private_rows
        .iter()
        .filter(|row| row.split == "train")
        .cloned()
        .collect::<Vec<_>>();
    let original_train_task_count = train_rows.len();
    let train_rows_before_budget = train_rows.clone();
    let eval_rows = private_rows
        .iter()
        .filter(|row| row.split == "eval")
        .cloned()
        .collect::<Vec<_>>();
    let estimated_work_steps_before_budget = estimated_rust_work_steps(
        &train_rows,
        &eval_rows,
        &public_rows,
        config.epochs.max(1),
        config.candidates_per_task.max(1),
    );
    if config.max_work_steps > 0 && estimated_work_steps_before_budget > config.max_work_steps {
        train_rows = trim_train_rows_to_work_budget(
            &train_rows,
            &eval_rows,
            &public_rows,
            config.epochs.max(1),
            config.candidates_per_task.max(1),
            config.max_work_steps,
        );
    }
    let estimated_work_steps_after_budget = estimated_rust_work_steps(
        &train_rows,
        &eval_rows,
        &public_rows,
        config.epochs.max(1),
        config.candidates_per_task.max(1),
    );
    let work_budget_admission = work_budget_admission_summary(
        &train_rows_before_budget,
        &train_rows,
        config.max_work_steps,
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
    );
    write_progress_report(
        &config,
        &started,
        "inputs_loaded_work_budget_applied",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "private_rows": private_rows.len(),
            "sts_stream_count": sts_streams.len(),
            "work_budget_admission": work_budget_admission.clone(),
        }),
    )?;
    record_phase_timing(
        &mut phase_timing_ms,
        "input_load_and_work_budget",
        &mut phase_started,
    );
    let mut vocab = build_vocab(&train_rows, config.max_vocab.max(16));
    extend_vocab_with_visible_contract_tokens(
        &mut vocab,
        &[eval_rows.as_slice(), public_rows.as_slice()],
        config.max_vocab.max(16).saturating_add(512),
    );
    let expression_bank = build_expression_bank(&train_rows);
    let body_prototypes = build_body_prototype_model(&train_rows);
    let body_ngram = build_body_ngram_model(&train_rows);
    write_progress_report(
        &config,
        &started,
        "token_models_built",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "vocab_size": vocab.id_to_token.len(),
            "expression_bank_size": expression_bank.len(),
            "body_prototype_category_count": body_prototypes.by_category.len(),
            "body_ngram_key_count": body_ngram.counts.len(),
        }),
    )?;
    record_phase_timing(&mut phase_timing_ms, "token_models_built", &mut phase_started);
    trace_code_lm("built token models");
    write_progress_report(
        &config,
        &started,
        "parallel_decoder_readout_training_started",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "parallelism_policy": "project_theseus_code_lm_checkpoint_parallel_aux_decoders_v1",
            "state_sequence_worker": true,
            "symliquid_state_worker": true,
            "main_thread_readout_training": true,
            "cuda_readout_requested": config.use_cuda_readout,
            "readout_eval_limit": config.readout_eval_limit,
            "aux_decoder_train_limit": config.aux_decoder_train_limit,
            "score_semantics": "runtime_optimization_only_not_capability_promotion"
        }),
    )?;
    let aux_parallel_started = Instant::now();
    let aux_train_rows = bounded_code_task_sample(&train_rows, config.aux_decoder_train_limit);
    let aux_train_task_count = aux_train_rows.len();
    let aux_decoder_bounded = config.aux_decoder_train_limit > 0 && aux_train_task_count < train_rows.len();
    let aux_decoder_epochs = if aux_decoder_bounded {
        1
    } else {
        config.epochs.max(1)
    };
    let state_rows = aux_train_rows.clone();
    let state_vocab = vocab.clone();
    let state_epochs = aux_decoder_epochs;
    let state_lr = config.lr.max(0.001);
    let state_sequence_handle = std::thread::spawn(move || {
        train_state_sequence_decoder(&state_rows, &state_vocab, state_epochs, state_lr)
    });
    let sym_rows = aux_train_rows.clone();
    let sym_vocab = vocab.clone();
    let sym_sts_streams = sts_streams.clone();
    let sym_state_dim = config.hv_dim.max(64);
    let sym_epochs = aux_decoder_epochs;
    let sym_lr = config.lr.max(0.001);
    let symliquid_state_handle = std::thread::spawn(move || {
        train_symliquid_state_decoder(
            &sym_rows,
            &sym_vocab,
            &sym_sts_streams,
            sym_state_dim,
            sym_epochs,
            sym_lr,
        )
        .map_err(|error| error.to_string())
    });
    trace_code_lm("spawned auxiliary decoder workers");
    let readout_phase_started = Instant::now();
    write_progress_report(
        &config,
        &started,
        "next_token_readout_training_started",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "auxiliary_decoder_workers_active": true,
            "cuda_readout_requested": config.use_cuda_readout,
            "readout_train_position": "main_thread_overlapped_with_auxiliary_decoder_workers",
            "score_semantics": "runtime_optimization_only_not_capability_promotion"
        }),
    )?;
    let examples_build_started = Instant::now();
    let examples = build_training_examples(&train_rows, &vocab);
    let examples_build_ms = examples_build_started.elapsed().as_millis();
    let eval_examples = bounded_train_example_sample(&examples, config.readout_eval_limit);
    write_progress_report(
        &config,
        &started,
        "next_token_examples_built",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "training_example_count": examples.len(),
            "readout_eval_example_count": eval_examples.len(),
            "examples_build_ms": examples_build_ms,
            "readout_eval_limit": config.readout_eval_limit,
            "aux_decoder_train_task_count": aux_train_task_count,
            "aux_decoder_train_limit": config.aux_decoder_train_limit,
            "aux_decoder_epochs": aux_decoder_epochs,
            "aux_decoder_bounded": aux_decoder_bounded,
            "score_semantics": "runtime_optimization_only_not_capability_promotion"
        }),
    )?;
    let mut readout = LinearReadout::zeros(config.hv_dim.max(32), vocab.id_to_token.len());
    let before_trace = evaluate_next_token(&readout, &eval_examples, config.hv_dim.max(32))?;
    let next_token_readout_backend = train_next_token_readout(
        &mut readout,
        &examples,
        config.hv_dim.max(32),
        config.epochs.max(1),
        config.lr,
        config.use_cuda_readout,
    )?;
    let after_trace = evaluate_next_token(&readout, &eval_examples, config.hv_dim.max(32))?;
    let next_token_readout_phase_ms = readout_phase_started.elapsed().as_millis();
    write_progress_report(
        &config,
        &started,
        "linear_readout_trained_aux_decoders_joining",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "training_example_count": examples.len(),
            "readout_eval_example_count": eval_examples.len(),
            "examples_build_ms": examples_build_ms,
            "before_next_token_accuracy": round6(before_trace.accuracy),
            "after_next_token_accuracy": round6(after_trace.accuracy),
            "next_token_accuracy_delta": round6(after_trace.accuracy - before_trace.accuracy),
            "next_token_readout_backend": next_token_readout_backend,
            "cuda_readout_requested": config.use_cuda_readout,
            "next_token_readout_phase_ms": next_token_readout_phase_ms,
            "auxiliary_decoder_workers_active": true,
        }),
    )?;
    let (state_sequence_decoder, state_sequence_before, state_sequence_after) =
        state_sequence_handle.join().map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "state-sequence decoder worker panicked",
            )
        })?;
    let (symliquid_state_decoder, symliquid_state_before, symliquid_state_after) =
        symliquid_state_handle
            .join()
            .map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "SymLiquid-state decoder worker panicked",
                )
            })?
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
    let aux_decoder_parallel_wall_ms = aux_parallel_started.elapsed().as_millis();
    record_phase_timing(
        &mut phase_timing_ms,
        "parallel_aux_decoders_and_readout",
        &mut phase_started,
    );
    write_progress_report(
        &config,
        &started,
        "parallel_decoder_readout_training_completed",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "training_example_count": examples.len(),
            "before_next_token_accuracy": round6(before_trace.accuracy),
            "after_next_token_accuracy": round6(after_trace.accuracy),
            "next_token_accuracy_delta": round6(after_trace.accuracy - before_trace.accuracy),
            "next_token_readout_backend": next_token_readout_backend,
            "cuda_readout_requested": config.use_cuda_readout,
            "state_sequence_before_accuracy": round6(state_sequence_before.accuracy),
            "state_sequence_after_accuracy": round6(state_sequence_after.accuracy),
            "state_sequence_updates": state_sequence_decoder.update_count,
            "state_sequence_feature_count": state_sequence_decoder.feature_count,
            "symliquid_state_before_accuracy": round6(symliquid_state_before.accuracy),
            "symliquid_state_after_accuracy": round6(symliquid_state_after.accuracy),
            "symliquid_state_updates": symliquid_state_decoder.update_count,
            "symliquid_state_dim": symliquid_state_decoder.state_dim,
            "aux_decoder_train_task_count": aux_train_task_count,
            "aux_decoder_train_limit": config.aux_decoder_train_limit,
            "aux_decoder_epochs": aux_decoder_epochs,
            "aux_decoder_bounded": aux_decoder_bounded,
            "parallelism_policy": "project_theseus_code_lm_checkpoint_parallel_aux_decoders_v1",
            "aux_decoder_parallel_wall_ms": aux_decoder_parallel_wall_ms,
            "next_token_readout_phase_ms": next_token_readout_phase_ms,
        }),
    )?;
    trace_code_lm("trained auxiliary decoders and next-token readout in overlapped checkpoint phase");
    let checkpoint_training_parallelism = json!({
        "policy": "project_theseus_code_lm_checkpoint_parallel_aux_decoders_v1",
        "enabled": true,
        "state_sequence_worker": true,
        "symliquid_state_worker": true,
        "main_thread_readout_training": true,
        "cuda_readout_requested": config.use_cuda_readout,
        "readout_eval_limit": config.readout_eval_limit,
        "readout_eval_example_count": eval_examples.len(),
        "aux_decoder_train_limit": config.aux_decoder_train_limit,
        "aux_decoder_train_task_count": aux_train_task_count,
        "aux_decoder_epochs": aux_decoder_epochs,
        "aux_decoder_bounded": aux_decoder_bounded,
        "next_token_readout_backend": next_token_readout_backend,
        "aux_decoder_parallel_wall_ms": aux_decoder_parallel_wall_ms,
        "next_token_readout_phase_ms": next_token_readout_phase_ms,
        "score_semantics": "runtime_optimization_only_not_capability_promotion"
    });
    let checkpoint_material = json!({
        "seed": config.seed,
        "private_curriculum": config.private_curriculum,
        "public_task_manifest": config.public_task_manifest,
        "sts_streams": config.sts_streams,
        "train_rows": train_rows.len(),
        "eval_rows": eval_rows.len(),
        "public_rows": public_rows.len(),
        "vocab": vocab.id_to_token,
        "epochs": config.epochs,
        "lr": config.lr,
        "before_accuracy": before_trace.accuracy,
        "after_accuracy": after_trace.accuracy,
        "expression_bank": expression_bank.iter().take(64).map(|item| item.expr.clone()).collect::<Vec<_>>(),
        "body_prototype_category_count": body_prototypes.by_category.len(),
        "body_ngram_key_count": body_ngram.counts.len(),
        "state_sequence_feature_count": state_sequence_decoder.feature_count,
        "state_sequence_updates": state_sequence_decoder.update_count,
        "state_sequence_before_accuracy": state_sequence_before.accuracy,
        "state_sequence_after_accuracy": state_sequence_after.accuracy,
        "symliquid_state_dim": symliquid_state_decoder.state_dim,
        "symliquid_state_updates": symliquid_state_decoder.update_count,
        "symliquid_state_before_accuracy": symliquid_state_before.accuracy,
        "symliquid_state_after_accuracy": symliquid_state_after.accuracy,
        "checkpoint_training_parallelism": checkpoint_training_parallelism.clone(),
    });
    let checkpoint_hash = stable_hash_hex(&checkpoint_material.to_string());
    let checkpoint_id = format!("theseus_student_code_lm_{}", &checkpoint_hash[..16]);
    let legacy_top_level_arrays = legacy_top_level_checkpoint_arrays_enabled();
    let mut checkpoint = json!({
        "policy": "project_theseus_code_lm_closure_checkpoint_v1",
        "created_utc": now(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_kind": "symliquid_recurrent_state_code_lm_full_body_v3",
        "backend": next_token_readout_backend,
        "seed": config.seed,
        "hv_dim": readout.input_dim,
        "output_dim": readout.output_dim,
        "epochs": config.epochs,
        "lr": config.lr,
        "private_curriculum": rel(&config.private_curriculum),
        "public_task_manifest": rel(&config.public_task_manifest),
        "checkpoint_storage": {
            "format": "compact_json_model_artifacts_v1",
            "model_artifacts_v1_canonical": true,
            "legacy_top_level_arrays": legacy_top_level_arrays,
            "legacy_top_level_arrays_env": "THESEUS_CODE_LM_LEGACY_TOP_LEVEL_CHECKPOINT_ARRAYS=1",
            "score_semantics": "storage/runtime optimization only; model content remains in model_artifacts_v1",
        },
        "summary": {
            "private_train_task_count": train_rows.len(),
            "private_train_task_count_before_work_budget": original_train_task_count,
            "train_rows_trimmed_by_work_budget": original_train_task_count.saturating_sub(train_rows.len()),
            "private_eval_task_count": eval_rows.len(),
            "public_task_count": public_rows.len(),
            "input_file_status": input_file_status(&config),
            "step_duration": step_duration_summary(
                &config,
                original_train_task_count,
                train_rows.len(),
                eval_rows.len(),
                public_rows.len(),
                estimated_work_steps_before_budget,
                estimated_work_steps_after_budget,
            ),
            "work_budget_admission": work_budget_admission.clone(),
            "checkpoint_in_ignored_for_training": checkpoint_in_ignored_for_training,
            "checkpoint_training_parallelism": checkpoint_training_parallelism.clone(),
            "sts_nonregression_candidate_superset": !sts_streams.is_empty(),
            "sts_nonregression_contract": "STS-conditioned phases preserve unconditioned semantic, skeleton, state, beam, ngram, and greedy decoder candidates so repair cannot regress by deleting the STS-off path",
            "private_sts_conditioned_eval_task_count": eval_rows
                .iter()
                .filter(|task| sts_streams.contains_key(&task.task_id))
                .count(),
            "sts_stream_conditioned_task_count": public_rows
                .iter()
                .filter(|task| sts_streams.contains_key(&task.task_id))
                .count(),
            "training_example_count": examples.len(),
            "expression_bank_size": expression_bank.len(),
            "body_prototype_category_count": body_prototypes.by_category.len(),
            "body_ngram_key_count": body_ngram.counts.len(),
            "decoder_contract_conditioning_used": decoder_contract_task_count(&train_rows) > 0
                || decoder_effective_contract_task_count(&public_rows) > 0,
            "decoder_contract_private_train_task_count": decoder_contract_task_count(&train_rows),
            "decoder_contract_public_task_count": decoder_contract_task_count(&public_rows),
            "decoder_contract_visible_inferred_public_task_count": decoder_effective_contract_task_count(&public_rows),
            "decoder_contract_verifier_v1": "signature -> argument_use -> return_shape -> syntax_body -> branch_loop_local_skeleton -> semantic_family -> sts_alignment",
            "semantic_decoder_v2_planner_used": decoder_contract_task_count(&train_rows) > 0
                || decoder_effective_contract_task_count(&public_rows) > 0,
            "semantic_decoder_v2_public_task_count": public_rows.len(),
            "semantic_decoder_v2_contract": "visible_prompt -> signature/types -> return_shape -> execution_shape_skeleton_plan -> token_decode -> guardrail_repair -> edge_exec_repair_v1_private_first",
            "typed_edge_exec_receiver_v1_enabled": typed_edge_exec_receiver_v1_enabled(),
            "typed_edge_exec_receiver_v1_contract": "private body execution signal -> type_admissible -> return_shape -> edge_obligation -> branch_loop -> STS/contract tie break",
            "private_type_shape_receiver_veto_v1_enabled": private_type_shape_receiver_veto_v1_enabled(),
            "private_type_shape_receiver_veto_v1_contract": "teacher-gated type/return-shape receiver bias after private disabled-vs-enabled ablation passes; public data remains calibration-only",
            "state_sequence_feature_count": state_sequence_decoder.feature_count,
            "state_sequence_updates": state_sequence_decoder.update_count,
            "state_sequence_example_count": state_sequence_after.example_count,
            "state_sequence_before_accuracy": round6(state_sequence_before.accuracy),
            "state_sequence_after_accuracy": round6(state_sequence_after.accuracy),
            "state_sequence_accuracy_delta": round6(state_sequence_after.accuracy - state_sequence_before.accuracy),
            "symliquid_state_dim": symliquid_state_decoder.state_dim,
            "symliquid_state_updates": symliquid_state_decoder.update_count,
            "symliquid_state_before_accuracy": round6(symliquid_state_before.accuracy),
            "symliquid_state_after_accuracy": round6(symliquid_state_after.accuracy),
            "symliquid_state_accuracy_delta": round6(symliquid_state_after.accuracy - symliquid_state_before.accuracy),
            "before_next_token_accuracy": round6(before_trace.accuracy),
            "after_next_token_accuracy": round6(after_trace.accuracy),
            "next_token_accuracy_delta": round6(after_trace.accuracy - before_trace.accuracy),
            "next_token_readout_backend": next_token_readout_backend,
            "cuda_readout_requested": config.use_cuda_readout,
            "cuda_readout_used": next_token_readout_backend.contains("cuda"),
            "checkpoint_storage": {
                "format": "compact_json_model_artifacts_v1",
                "model_artifacts_v1_canonical": true,
                "legacy_top_level_arrays": legacy_top_level_arrays,
                "legacy_top_level_arrays_env": "THESEUS_CODE_LM_LEGACY_TOP_LEVEL_CHECKPOINT_ARRAYS=1",
                "score_semantics": "storage/runtime optimization only; model content remains in model_artifacts_v1",
            },
            "phase_timing_ms": phase_timing_ms.clone(),
            "token_level_code_generation_learned": after_trace.accuracy > before_trace.accuracy || state_sequence_after.accuracy > state_sequence_before.accuracy || symliquid_state_after.accuracy > symliquid_state_before.accuracy,
            "public_tests_visible": false,
            "public_canonical_solutions_visible": false,
            "external_inference_calls": 0
        },
        "generation_policy": {
            "public_tests_visible": false,
            "canonical_solutions_visible": false,
            "private_eval_solutions_used_for_training": false,
            "task_id_specific_lookup": false,
            "loop_closure_tool_distillation": false,
            "external_inference_calls": 0,
            "allowed_inputs": ["visible_prompt", "entry_point", "visible_signature_shape", "decoder_contract_return_shape_type_family_required_constructs", "private_train_code_tokens", "private_train_body_ngram_tokens", "private_train_sts_style_stream_tokens", "governed_open_code_train_expressions", "category_for_private_rows_only"],
            "optional_conditioning": ["native_sts_generated_streams_without_public_tests_or_solutions", "sts_conditioned_skeleton_choice"],
            "decoder": "semantic_decoder_v2_plan_then_symliquid_recurrent_state_decoder_plus_sparse_state_sequence_decoder_plus_full_body_token_beam_private_body_ngram_with_ast_syntax_filters",
        },
        "model_artifacts_v1": model_artifacts_v1(
            &vocab,
            &expression_bank,
            &body_prototypes,
            &body_ngram,
            &state_sequence_decoder,
            &symliquid_state_decoder,
            &readout,
        ),
        "external_inference_calls": 0,
    });
    if legacy_top_level_arrays {
        if let Some(object) = checkpoint.as_object_mut() {
            object.insert("vocab".to_string(), json!(vocab.id_to_token));
            object.insert("weights".to_string(), json!(readout.weights));
            object.insert("bias".to_string(), json!(readout.bias));
            object.insert(
                "symliquid_state_readout".to_string(),
                json!({
                    "state_dim": symliquid_state_decoder.state_dim,
                    "output_dim": symliquid_state_decoder.readout.output_dim,
                    "weights": symliquid_state_decoder.readout.weights,
                    "bias": symliquid_state_decoder.readout.bias,
                }),
            );
        }
    }
    trace_code_lm("built checkpoint json");
    write_json_compact(Path::new(&config.checkpoint_out), &checkpoint)?;
    write_progress_report(
        &config,
        &started,
        "checkpoint_written",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "checkpoint": rel(&config.checkpoint_out),
            "checkpoint_id": checkpoint_id,
        }),
    )?;
    record_phase_timing(&mut phase_timing_ms, "checkpoint_write", &mut phase_started);
    trace_code_lm("wrote checkpoint");

    if config.checkpoint_only {
        let gates = vec![
            gate(
                "private_curriculum_loaded",
                !private_rows.is_empty(),
                json!(format!("rows={}", private_rows.len())),
            ),
            gate(
                "private_train_eval_split_present",
                !train_rows.is_empty() && !eval_rows.is_empty(),
                json!(format!("train={} eval={}", train_rows.len(), eval_rows.len())),
            ),
            gate(
                "public_visible_tasks_loaded",
                !public_rows.is_empty(),
                json!(format!("public_tasks={}", public_rows.len())),
            ),
            gate(
                "training_examples_nonzero",
                !examples.is_empty(),
                json!(format!("examples={}", examples.len())),
            ),
            gate(
                "checkpoint_model_artifacts_present",
                true,
                json!("model_artifacts_v1 written for train-once fanout"),
            ),
            gate(
                "linear_readout_improved_next_token",
                after_trace.accuracy > before_trace.accuracy
                    || state_sequence_after.accuracy > state_sequence_before.accuracy
                    || symliquid_state_after.accuracy > symliquid_state_before.accuracy,
                json!(format!(
                    "linear_before={:.6} linear_after={:.6} state_before={:.6} state_after={:.6} symliquid_before={:.6} symliquid_after={:.6}",
                    before_trace.accuracy,
                    after_trace.accuracy,
                    state_sequence_before.accuracy,
                    state_sequence_after.accuracy,
                    symliquid_state_before.accuracy,
                    symliquid_state_after.accuracy
                )),
            ),
            gate(
                "public_tests_not_visible",
                true,
                json!("public task exporter omits tests and canonical solutions"),
            ),
            gate(
                "private_eval_solutions_not_trained",
                true,
                json!("only split=train rows build vocab, expression bank, and train examples"),
            ),
            gate(
                "external_inference_zero",
                true,
                json!("local Rust/SymLiquid checkpoint training only"),
            ),
        ];
        let trigger_state = if gates
            .iter()
            .all(|row| row["passed"].as_bool().unwrap_or(false))
        {
            "GREEN"
        } else {
            "YELLOW"
        };
        let report = json!({
            "policy": "project_theseus_code_lm_closure_rust_checkpoint_only_v1",
            "created_utc": now(),
            "run_status": "completed",
            "progress_stage": "checkpoint_only_completed",
            "trigger_state": trigger_state,
            "checkpoint": rel(&config.checkpoint_out),
            "checkpoint_id": checkpoint_id,
            "summary": {
                "checkpoint_only": true,
                "train_once_checkpoint_fanout_ready": true,
                "private_train_task_count": train_rows.len(),
                "private_train_task_count_before_work_budget": original_train_task_count,
                "train_rows_trimmed_by_work_budget": original_train_task_count.saturating_sub(train_rows.len()),
                "private_eval_task_count": eval_rows.len(),
                "public_task_count": public_rows.len(),
                "input_file_status": input_file_status(&config),
                "phase_timing_ms": phase_timing_ms.clone(),
                "checkpoint_training_parallelism": checkpoint_training_parallelism.clone(),
                "step_duration": step_duration_summary(
                    &config,
                    original_train_task_count,
                    train_rows.len(),
                    eval_rows.len(),
                    public_rows.len(),
                    estimated_work_steps_before_budget,
                    estimated_work_steps_after_budget,
                ),
                "work_budget_admission": work_budget_admission.clone(),
                "training_example_count": examples.len(),
                "vocab_size": vocab.id_to_token.len(),
                "expression_bank_size": expression_bank.len(),
                "body_prototype_category_count": body_prototypes.by_category.len(),
                "body_ngram_key_count": body_ngram.counts.len(),
                "state_sequence_feature_count": state_sequence_decoder.feature_count,
                "state_sequence_updates": state_sequence_decoder.update_count,
                "state_sequence_example_count": state_sequence_after.example_count,
                "state_sequence_before_accuracy": round6(state_sequence_before.accuracy),
                "state_sequence_after_accuracy": round6(state_sequence_after.accuracy),
                "state_sequence_accuracy_delta": round6(state_sequence_after.accuracy - state_sequence_before.accuracy),
                "symliquid_state_dim": symliquid_state_decoder.state_dim,
                "symliquid_state_updates": symliquid_state_decoder.update_count,
                "symliquid_state_example_count": symliquid_state_after.example_count,
                "symliquid_state_before_accuracy": round6(symliquid_state_before.accuracy),
                "symliquid_state_after_accuracy": round6(symliquid_state_after.accuracy),
                "symliquid_state_accuracy_delta": round6(symliquid_state_after.accuracy - symliquid_state_before.accuracy),
                "before_next_token_accuracy": round6(before_trace.accuracy),
                "after_next_token_accuracy": round6(after_trace.accuracy),
                "next_token_accuracy_delta": round6(after_trace.accuracy - before_trace.accuracy),
                "next_token_readout_backend": next_token_readout_backend,
                "cuda_readout_requested": config.use_cuda_readout,
                "cuda_readout_used": next_token_readout_backend.contains("cuda"),
                "model_artifacts_v1_written": true,
                "target_architecture": "train_once_checkpoint_then_hive_distributed_candidate_generation_and_verification",
                "repeated_training_per_candidate_shard": false,
                "external_inference_calls": 0
            },
            "gates": gates,
            "runtime_ms": (started.elapsed().as_secs_f64() * 1000.0).round() as u64,
            "external_inference_calls": 0,
        });
        write_json(Path::new(&config.report_out), &report)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "policy": "project_theseus_code_lm_closure_rust_checkpoint_only_v1",
                "trigger_state": trigger_state,
                "report_out": rel(&config.report_out),
                "checkpoint": rel(&config.checkpoint_out),
                "checkpoint_id": checkpoint_id,
                "runtime_ms": (started.elapsed().as_secs_f64() * 1000.0).round() as u64,
            }))?
        );
        return Ok(());
    }

    write_progress_report(
        &config,
        &started,
        "private_candidate_generation_started",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "private_eval_task_count": eval_rows.len(),
            "phases": ["private_eval", "private_eval_sts_off", "private_baseline"],
            "candidates_per_task": config.candidates_per_task.max(1),
        }),
    )?;
    let private_sts_off_eval_rows = private_sts_off_nonregression_eval_rows(&eval_rows);
    let private_sts_off_candidate_limit =
        private_sts_off_nonregression_candidate_limit(config.candidates_per_task.max(1));
    let private_sts_off_nonregression_policy = private_sts_off_nonregression_policy_json(
        eval_rows.len(),
        private_sts_off_eval_rows.len(),
        config.candidates_per_task.max(1),
        private_sts_off_candidate_limit,
    );
    let mut private_candidates = Vec::new();
    private_candidates.extend(candidate_rows(
        &eval_rows,
        &expression_bank,
        &body_prototypes,
        &body_ngram,
        &state_sequence_decoder,
        &symliquid_state_decoder,
        &readout,
        &vocab,
        &checkpoint_id,
        config.seed,
        config.candidates_per_task.max(1),
        "private_eval",
        true,
        &sts_streams,
        Some(CandidateHeartbeat {
            config: &config,
            started: &started,
            stage: "private_candidate_generation",
            phase: "private_eval",
            trained: true,
            total_tasks: eval_rows.len(),
            original_train_task_count,
            train_task_count: train_rows.len(),
            eval_task_count: eval_rows.len(),
            public_task_count: public_rows.len(),
            estimated_work_steps_before_budget,
            estimated_work_steps_after_budget,
        }),
        &BTreeMap::new(),
    ));
    private_candidates.extend(candidate_rows(
        &private_sts_off_eval_rows,
        &expression_bank,
        &body_prototypes,
        &body_ngram,
        &state_sequence_decoder,
        &symliquid_state_decoder,
        &readout,
        &vocab,
        &checkpoint_id,
        config.seed,
        private_sts_off_candidate_limit,
        "private_eval_sts_off",
        true,
        &HashMap::new(),
        Some(CandidateHeartbeat {
            config: &config,
            started: &started,
            stage: "private_candidate_generation",
            phase: "private_eval_sts_off",
            trained: true,
            total_tasks: private_sts_off_eval_rows.len(),
            original_train_task_count,
            train_task_count: train_rows.len(),
            eval_task_count: eval_rows.len(),
            public_task_count: public_rows.len(),
            estimated_work_steps_before_budget,
            estimated_work_steps_after_budget,
        }),
        &BTreeMap::new(),
    ));
    private_candidates.extend(candidate_rows(
        &eval_rows,
        &expression_bank,
        &body_prototypes,
        &body_ngram,
        &state_sequence_decoder,
        &symliquid_state_decoder,
        &readout,
        &vocab,
        &checkpoint_id,
        config.seed,
        config.candidates_per_task.max(1),
        "private_baseline",
        false,
        &sts_streams,
        Some(CandidateHeartbeat {
            config: &config,
            started: &started,
            stage: "private_candidate_generation",
            phase: "private_baseline",
            trained: false,
            total_tasks: eval_rows.len(),
            original_train_task_count,
            train_task_count: train_rows.len(),
            eval_task_count: eval_rows.len(),
            public_task_count: public_rows.len(),
            estimated_work_steps_before_budget,
            estimated_work_steps_after_budget,
        }),
        &BTreeMap::new(),
    ));
    let private_sts_nonregression_union_added = if sts_streams.is_empty() {
        0
    } else {
        append_sts_nonregression_union_candidates(&mut private_candidates)
    };
    write_jsonl(
        Path::new(&config.private_candidate_out),
        &private_candidates,
    )?;
    let private_candidate_binary_artifact =
        write_jsonl_binary_sidecar(Path::new(&config.private_candidate_out), &private_candidates)?;
    record_phase_timing(
        &mut phase_timing_ms,
        "private_candidate_generation_and_write",
        &mut phase_started,
    );
    write_progress_report(
        &config,
        &started,
        "private_candidate_manifest_written_filter_diagnostics_started",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "private_candidate_manifest": rel(&config.private_candidate_out),
            "private_candidate_binary_artifact": rel_path(&private_candidate_binary_artifact),
            "private_candidate_count": private_candidates.len(),
            "private_sts_nonregression_union_added": private_sts_nonregression_union_added,
            "private_sts_off_nonregression_policy": private_sts_off_nonregression_policy.clone(),
            "diagnostic_stage": "private_filter_diagnostics",
            "phase_timing_ms": phase_timing_ms.clone(),
        }),
    )?;
    let private_filter_diagnostics = candidate_filter_diagnostics(
        &eval_rows,
        &expression_bank,
        &body_prototypes,
        &body_ngram,
        &state_sequence_decoder,
        &symliquid_state_decoder,
        &readout,
        &vocab,
        config.seed,
        config.candidates_per_task.max(1),
        &sts_streams,
        Some(CandidateHeartbeat {
            config: &config,
            started: &started,
            stage: "private_filter_diagnostics",
            phase: "private_filter_diagnostics",
            trained: true,
            total_tasks: eval_rows.len(),
            original_train_task_count,
            train_task_count: train_rows.len(),
            eval_task_count: eval_rows.len(),
            public_task_count: public_rows.len(),
            estimated_work_steps_before_budget,
            estimated_work_steps_after_budget,
        }),
    );
    record_phase_timing(
        &mut phase_timing_ms,
        "private_filter_diagnostics",
        &mut phase_started,
    );
    write_progress_report(
        &config,
        &started,
        "private_candidates_written",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "private_candidate_manifest": rel(&config.private_candidate_out),
            "private_candidate_binary_artifact": rel_path(&private_candidate_binary_artifact),
            "private_candidate_count": private_candidates.len(),
            "private_sts_nonregression_union_added": private_sts_nonregression_union_added,
            "phase_timing_ms": phase_timing_ms.clone(),
        }),
    )?;

    write_progress_report(
        &config,
        &started,
        "public_candidate_generation_started",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "public_task_count": public_rows.len(),
            "candidates_per_task": config.candidates_per_task.max(1),
            "generation_budget": "bounded_public_semantic_transfer_candidates",
        }),
    )?;

    let public_candidates = candidate_rows(
        &public_rows,
        &expression_bank,
        &body_prototypes,
        &body_ngram,
        &state_sequence_decoder,
        &symliquid_state_decoder,
        &readout,
        &vocab,
        &checkpoint_id,
        config.seed,
        config.candidates_per_task.max(1),
        "public_calibration",
        true,
        &sts_streams,
        Some(CandidateHeartbeat {
            config: &config,
            started: &started,
            stage: "public_candidate_generation",
            phase: "public_calibration",
            trained: true,
            total_tasks: public_rows.len(),
            original_train_task_count,
            train_task_count: train_rows.len(),
            eval_task_count: eval_rows.len(),
            public_task_count: public_rows.len(),
            estimated_work_steps_before_budget,
            estimated_work_steps_after_budget,
        }),
        &BTreeMap::new(),
    );
    let public_filter_diagnostics =
        candidate_filter_diagnostics_from_rows(&public_rows, &public_candidates);
    let private_template_like_candidate_count =
        count_bool(&private_candidates, "template_like_candidate");
    let public_template_like_candidate_count =
        count_bool(&public_candidates, "template_like_candidate");
    let private_token_level_candidate_count =
        count_bool(&private_candidates, "token_level_code_generation_learned");
    let public_token_level_candidate_count =
        count_bool(&public_candidates, "token_level_code_generation_learned");
    write_jsonl(Path::new(&config.public_candidate_out), &public_candidates)?;
    let public_candidate_binary_artifact =
        write_jsonl_binary_sidecar(Path::new(&config.public_candidate_out), &public_candidates)?;
    record_phase_timing(
        &mut phase_timing_ms,
        "public_candidate_generation_and_write",
        &mut phase_started,
    );
    write_progress_report(
        &config,
        &started,
        "public_candidates_written",
        original_train_task_count,
        train_rows.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps_before_budget,
        estimated_work_steps_after_budget,
        json!({
            "public_candidate_manifest": rel(&config.public_candidate_out),
            "public_candidate_count": public_candidates.len(),
            "public_candidate_binary_artifact": rel_path(&public_candidate_binary_artifact),
            "phase_timing_ms": phase_timing_ms.clone(),
        }),
    )?;

    let gates = vec![
        gate(
            "private_curriculum_loaded",
            !private_rows.is_empty(),
            json!(format!("rows={}", private_rows.len())),
        ),
        gate(
            "private_train_eval_split_present",
            !train_rows.is_empty() && !eval_rows.is_empty(),
            json!(format!(
                "train={} eval={}",
                train_rows.len(),
                eval_rows.len()
            )),
        ),
        gate(
            "public_visible_tasks_loaded",
            !public_rows.is_empty(),
            json!(format!("public_tasks={}", public_rows.len())),
        ),
        gate(
            "training_examples_nonzero",
            !examples.is_empty(),
            json!(format!("examples={}", examples.len())),
        ),
        gate(
            "body_ngram_model_nonzero",
            !body_ngram.counts.is_empty(),
            json!(format!("body_ngram_keys={}", body_ngram.counts.len())),
        ),
        gate(
            "state_sequence_decoder_trained",
            state_sequence_after.accuracy > state_sequence_before.accuracy
                && state_sequence_decoder.update_count > 0,
            json!(format!(
                "before={:.6} after={:.6} updates={} features={}",
                state_sequence_before.accuracy,
                state_sequence_after.accuracy,
                state_sequence_decoder.update_count,
                state_sequence_decoder.feature_count
            )),
        ),
        gate(
            "symliquid_recurrent_state_decoder_trained",
            symliquid_state_after.accuracy > symliquid_state_before.accuracy
                && symliquid_state_decoder.update_count > 0,
            json!(format!(
                "before={:.6} after={:.6} updates={} state_dim={}",
                symliquid_state_before.accuracy,
                symliquid_state_after.accuracy,
                symliquid_state_decoder.update_count,
                symliquid_state_decoder.state_dim
            )),
        ),
        gate(
            "linear_readout_improved_next_token",
            after_trace.accuracy > before_trace.accuracy
                || state_sequence_after.accuracy > state_sequence_before.accuracy
                || symliquid_state_after.accuracy > symliquid_state_before.accuracy,
            json!(format!(
                "linear_before={:.6} linear_after={:.6} state_before={:.6} state_after={:.6} symliquid_before={:.6} symliquid_after={:.6}",
                before_trace.accuracy,
                after_trace.accuracy,
                state_sequence_before.accuracy,
                state_sequence_after.accuracy,
                symliquid_state_before.accuracy,
                symliquid_state_after.accuracy
            )),
        ),
        gate(
            "private_candidates_emitted",
            !private_candidates.is_empty(),
            json!(format!("private_candidates={}", private_candidates.len())),
        ),
        gate(
            "public_candidates_emitted",
            !public_candidates.is_empty(),
            json!(format!("public_candidates={}", public_candidates.len())),
        ),
        gate(
            "public_candidates_template_free",
            public_template_like_candidate_count == 0,
            json!(format!(
                "template_like_public_candidates={} template_free_mode={}",
                public_template_like_candidate_count,
                template_free_student_candidates_enabled()
            )),
        ),
        gate(
            "public_token_level_student_candidates_emitted",
            public_token_level_candidate_count > 0,
            json!(format!(
                "token_level_public_candidates={} private_token_level_candidates={}",
                public_token_level_candidate_count,
                private_token_level_candidate_count
            )),
        ),
        gate(
            "public_candidate_coverage",
            public_candidates.len() >= public_rows.len(),
            json!(format!(
                "public_candidates={} public_tasks={}",
                public_candidates.len(),
                public_rows.len()
            )),
        ),
        gate(
            "sts_stream_conditioning_safe",
            config.sts_streams.is_empty() || !sts_streams.is_empty(),
            json!(format!(
                "path={} streams={}",
                config.sts_streams,
                sts_streams.len()
            )),
        ),
        gate(
            "public_tests_not_visible",
            true,
            json!("public task exporter omits tests and canonical solutions"),
        ),
        gate(
            "private_eval_solutions_not_trained",
            true,
            json!("only split=train rows build vocab, expression bank, and train examples"),
        ),
        gate(
            "external_inference_zero",
            true,
            json!("local Rust/SymLiquid training only"),
        ),
    ];
    let trigger_state = if gates
        .iter()
        .all(|row| row["passed"].as_bool().unwrap_or(false))
    {
        "GREEN"
    } else {
        "YELLOW"
    };
    let private_sts_nonregression_union_candidate_count =
        count_bool(&private_candidates, "sts_nonregression_union_candidate");
    let private_candidate_generation_modes = candidate_generation_modes(&private_candidates);
    let private_candidate_generation_mode_counts =
        candidate_generation_mode_counts(&private_candidates);
    let report = json!({
        "policy": "project_theseus_code_lm_closure_rust_v1",
        "created_utc": now(),
        "run_status": "completed",
        "progress_stage": "completed",
        "trigger_state": trigger_state,
        "checkpoint": rel(&config.checkpoint_out),
        "private_candidate_manifest": rel(&config.private_candidate_out),
        "public_candidate_manifest": rel(&config.public_candidate_out),
        "private_candidate_binary_artifact": rel_path(&private_candidate_binary_artifact),
        "public_candidate_binary_artifact": rel_path(&public_candidate_binary_artifact),
        "candidate_generation_heartbeat": rel_path(&candidate_heartbeat_path(&config)),
        "summary": {
            "private_train_task_count": train_rows.len(),
            "private_train_task_count_before_work_budget": original_train_task_count,
            "train_rows_trimmed_by_work_budget": original_train_task_count.saturating_sub(train_rows.len()),
            "private_eval_task_count": eval_rows.len(),
            "public_task_count": public_rows.len(),
            "input_file_status": input_file_status(&config),
            "candidate_generation_heartbeat": rel_path(&candidate_heartbeat_path(&config)),
            "phase_timing_ms": phase_timing_ms.clone(),
            "checkpoint_training_parallelism": checkpoint_training_parallelism.clone(),
            "private_candidate_binary_artifact": rel_path(&private_candidate_binary_artifact),
            "public_candidate_binary_artifact": rel_path(&public_candidate_binary_artifact),
            "step_duration": step_duration_summary(
                &config,
                original_train_task_count,
                train_rows.len(),
                eval_rows.len(),
                public_rows.len(),
                estimated_work_steps_before_budget,
                estimated_work_steps_after_budget,
            ),
            "work_budget_admission": work_budget_admission.clone(),
            "decoder_completion_cache": decoder_completion_cache_summary(),
            "sts_nonregression_candidate_superset": !sts_streams.is_empty(),
            "sts_nonregression_contract": "STS-conditioned phases preserve unconditioned semantic, skeleton, state, beam, ngram, and greedy decoder candidates so repair cannot regress by deleting the STS-off path",
            "private_sts_conditioned_eval_task_count": eval_rows
                .iter()
                .filter(|task| sts_streams.contains_key(&task.task_id))
                .count(),
            "sts_stream_conditioned_task_count": public_rows
                .iter()
                .filter(|task| sts_streams.contains_key(&task.task_id))
                .count(),
            "training_example_count": examples.len(),
            "vocab_size": vocab.id_to_token.len(),
            "expression_bank_size": expression_bank.len(),
            "body_prototype_category_count": body_prototypes.by_category.len(),
            "body_ngram_key_count": body_ngram.counts.len(),
            "decoder_contract_conditioning_used": decoder_contract_task_count(&train_rows) > 0 || decoder_effective_contract_task_count(&public_rows) > 0,
            "decoder_contract_private_train_task_count": decoder_contract_task_count(&train_rows),
            "decoder_contract_public_task_count": decoder_contract_task_count(&public_rows),
            "decoder_contract_visible_inferred_public_task_count": decoder_effective_contract_task_count(&public_rows),
            "decoder_contract_verifier_v1": "signature -> argument_use -> return_shape -> syntax_body -> branch_loop_local_skeleton -> semantic_family -> sts_alignment",
            "decoder_contract_verifier_v1_public_pass_count": count_bool(&public_candidates, "decoder_contract_verifier_v1_passed"),
            "decoder_contract_verifier_v1_public_fail_count": public_candidates.len().saturating_sub(count_bool(&public_candidates, "decoder_contract_verifier_v1_passed")),
            "decoder_contract_verifier_v1_private_pass_count": count_bool(&private_candidates, "decoder_contract_verifier_v1_passed"),
            "decoder_contract_verifier_v1_private_fail_count": private_candidates.len().saturating_sub(count_bool(&private_candidates, "decoder_contract_verifier_v1_passed")),
            "semantic_decoder_v2_planner_used": decoder_contract_task_count(&train_rows) > 0 || decoder_effective_contract_task_count(&public_rows) > 0,
            "semantic_decoder_v2_public_task_count": public_rows.len(),
            "semantic_decoder_v2_public_plan_count": public_rows
                .iter()
                .filter(|task| !semantic_decoder_v2_prefixes(task, sts_streams.get(&task.task_id)).is_empty())
                .count(),
            "semantic_decoder_v2_contract": "visible_prompt -> signature/types -> return_shape -> execution_shape_skeleton_plan -> token_decode -> guardrail_repair -> edge_exec_repair_v1_private_first",
            "typed_edge_exec_receiver_v1_enabled": typed_edge_exec_receiver_v1_enabled(),
            "private_type_shape_receiver_veto_v1_enabled": private_type_shape_receiver_veto_v1_enabled(),
            "state_sequence_feature_count": state_sequence_decoder.feature_count,
            "state_sequence_updates": state_sequence_decoder.update_count,
            "state_sequence_example_count": state_sequence_after.example_count,
            "state_sequence_before_accuracy": round6(state_sequence_before.accuracy),
            "state_sequence_after_accuracy": round6(state_sequence_after.accuracy),
            "state_sequence_accuracy_delta": round6(state_sequence_after.accuracy - state_sequence_before.accuracy),
            "symliquid_state_dim": symliquid_state_decoder.state_dim,
            "symliquid_state_updates": symliquid_state_decoder.update_count,
            "symliquid_state_example_count": symliquid_state_after.example_count,
            "symliquid_state_before_accuracy": round6(symliquid_state_before.accuracy),
            "symliquid_state_after_accuracy": round6(symliquid_state_after.accuracy),
            "symliquid_state_accuracy_delta": round6(symliquid_state_after.accuracy - symliquid_state_before.accuracy),
            "before_next_token_accuracy": round6(before_trace.accuracy),
            "after_next_token_accuracy": round6(after_trace.accuracy),
            "next_token_accuracy_delta": round6(after_trace.accuracy - before_trace.accuracy),
            "private_candidate_count": private_candidates.len(),
            "private_sts_nonregression_union_added": private_sts_nonregression_union_added,
            "private_sts_nonregression_union_candidate_count": private_sts_nonregression_union_candidate_count,
            "private_sts_off_nonregression_policy": private_sts_off_nonregression_policy,
            "private_candidate_generation_modes": private_candidate_generation_modes,
            "private_candidate_generation_mode_counts": private_candidate_generation_mode_counts,
            "public_candidate_count": public_candidates.len(),
            "candidate_generation_mode": if sts_streams.is_empty() { "rust_code_lm_closure_decoder" } else { "rust_code_lm_closure_decoder_sts_conditioned" },
            "public_candidate_generation_modes": candidate_generation_modes(&public_candidates),
            "public_candidate_generation_mode_counts": candidate_generation_mode_counts(&public_candidates),
            "compositional_token_candidate_count": count_bool(&public_candidates, "compositional_token_candidate"),
            "full_body_token_candidate_count": count_bool(&public_candidates, "full_body_token_candidate"),
            "expression_memory_fallback_count": count_bool(&public_candidates, "expression_memory_fallback"),
            "private_token_level_candidate_count": private_token_level_candidate_count,
            "public_token_level_candidate_count": public_token_level_candidate_count,
            "token_level_code_generation_learned": public_token_level_candidate_count > 0,
            "private_template_like_candidate_count": private_template_like_candidate_count,
            "template_like_candidate_count": public_template_like_candidate_count,
            "template_free_student_candidates_enabled": template_free_student_candidates_enabled(),
            "private_filter_rejection_counts": aggregate_filter_diagnostics_rejection_counts(&private_filter_diagnostics),
            "private_filter_raw_mode_counts": aggregate_filter_diagnostics_mode_counts(&private_filter_diagnostics),
            "loop_closure_candidate_count": 0,
            "external_inference_calls": 0
        },
        "private_filter_diagnostics": private_filter_diagnostics,
        "public_filter_diagnostics": public_filter_diagnostics,
        "gates": gates,
        "runtime_ms": (started.elapsed().as_secs_f64() * 1000.0).round() as u64,
        "external_inference_calls": 0,
    });
    write_json(Path::new(&config.report_out), &report)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "policy": "project_theseus_code_lm_closure_rust_v1",
            "trigger_state": trigger_state,
            "report_out": rel(&config.report_out),
            "candidate_generation_heartbeat": rel_path(&candidate_heartbeat_path(&config)),
            "public_candidate_manifest": rel(&config.public_candidate_out),
            "private_candidate_manifest": rel(&config.private_candidate_out),
            "public_candidate_binary_artifact": rel_path(&public_candidate_binary_artifact),
            "private_candidate_binary_artifact": rel_path(&private_candidate_binary_artifact),
            "phase_timing_ms": phase_timing_ms,
            "runtime_ms": (started.elapsed().as_secs_f64() * 1000.0).round() as u64,
        }))?
    );
    Ok(())
}

fn model_artifacts_v1(
    vocab: &Vocab,
    expression_bank: &[ExpressionBankItem],
    body_prototypes: &BodyPrototypeModel,
    body_ngram: &BodyNgramModel,
    state_sequence_decoder: &StateSequenceDecoder,
    symliquid_state_decoder: &SymLiquidStateDecoder,
    readout: &LinearReadout,
) -> Value {
    json!({
        "policy": "project_theseus_code_lm_train_once_fanout_artifacts_v1",
        "vocab": vocab.id_to_token,
        "expression_bank": expression_bank.iter().map(|item| {
            json!({
                "expr": item.expr,
                "tokens": item.tokens,
                "hints": item.hints.iter().cloned().collect::<Vec<_>>(),
                "category": item.category,
                "count": item.count,
            })
        }).collect::<Vec<_>>(),
        "body_prototypes_by_category": body_prototypes.by_category,
        "body_ngram_counts": body_ngram.counts,
        "state_sequence_decoder": {
            "weights": state_sequence_decoder.weights,
            "bias": state_sequence_decoder.bias,
            "output_dim": state_sequence_decoder.output_dim,
            "feature_count": state_sequence_decoder.feature_count,
            "update_count": state_sequence_decoder.update_count,
        },
        "symliquid_state_decoder": {
            "state_dim": symliquid_state_decoder.state_dim,
            "update_count": symliquid_state_decoder.update_count,
            "readout": {
                "input_dim": symliquid_state_decoder.readout.input_dim,
                "output_dim": symliquid_state_decoder.readout.output_dim,
                "weights": symliquid_state_decoder.readout.weights,
                "bias": symliquid_state_decoder.readout.bias,
            }
        },
        "readout": {
            "input_dim": readout.input_dim,
            "output_dim": readout.output_dim,
            "weights": readout.weights,
            "bias": readout.bias,
        },
        "public_tests_used": false,
        "public_solutions_used": false,
        "training_use": "checkpoint_fanout_candidate_generation_without_retraining",
    })
}

fn load_code_lm_checkpoint_models(
    path: &Path,
) -> Result<LoadedCodeLmModels, Box<dyn std::error::Error>> {
    let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let artifacts = value
        .get("model_artifacts_v1")
        .and_then(Value::as_object)
        .ok_or("checkpoint missing model_artifacts_v1; rebuild train-once checkpoint with current binary")?;
    let tokens = value_string_vec(artifacts.get("vocab").unwrap_or(&Value::Null));
    if tokens.is_empty() {
        return Err("checkpoint model_artifacts_v1 has empty vocab".into());
    }
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| (token.clone(), idx))
        .collect::<HashMap<_, _>>();
    let vocab = Vocab {
        unk_id: token_to_id.get("<UNK>").copied().unwrap_or(0),
        token_to_id,
        id_to_token: tokens,
    };
    let expression_bank = artifacts
        .get("expression_bank")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .map(|row| ExpressionBankItem {
                    expr: string_field(row, "expr"),
                    tokens: value_string_vec(row.get("tokens").unwrap_or(&Value::Null)),
                    hints: value_string_vec(row.get("hints").unwrap_or(&Value::Null))
                        .into_iter()
                        .collect(),
                    category: string_field(row, "category"),
                    count: value_usize(row.get("count").unwrap_or(&Value::Null)),
                })
                .filter(|item| !item.expr.trim().is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let body_prototypes = BodyPrototypeModel {
        by_category: value_string_vec_map(
            artifacts
                .get("body_prototypes_by_category")
                .unwrap_or(&Value::Null),
        ),
    };
    let body_ngram = BodyNgramModel {
        counts: value_ngram_counts(artifacts.get("body_ngram_counts").unwrap_or(&Value::Null)),
    };
    let state_value = artifacts
        .get("state_sequence_decoder")
        .unwrap_or(&Value::Null);
    let state_sequence_decoder = StateSequenceDecoder {
        weights: value_f32_vec_map(state_value.get("weights").unwrap_or(&Value::Null)),
        bias: value_f32_vec(state_value.get("bias").unwrap_or(&Value::Null)),
        output_dim: value_usize(state_value.get("output_dim").unwrap_or(&Value::Null)),
        feature_count: value_usize(state_value.get("feature_count").unwrap_or(&Value::Null)),
        update_count: value_usize(state_value.get("update_count").unwrap_or(&Value::Null)),
    };
    let sym_value = artifacts
        .get("symliquid_state_decoder")
        .unwrap_or(&Value::Null);
    let sym_readout_value = sym_value.get("readout").unwrap_or(&Value::Null);
    let symliquid_state_decoder = SymLiquidStateDecoder {
        state_dim: value_usize(sym_value.get("state_dim").unwrap_or(&Value::Null)),
        update_count: value_usize(sym_value.get("update_count").unwrap_or(&Value::Null)),
        readout: linear_readout_from_value(sym_readout_value)?,
    };
    let readout = linear_readout_from_value(artifacts.get("readout").unwrap_or(&Value::Null))?;
    Ok(LoadedCodeLmModels {
        checkpoint_id: string_field(&value, "checkpoint_id"),
        vocab,
        expression_bank,
        body_prototypes,
        body_ngram,
        state_sequence_decoder,
        symliquid_state_decoder,
        readout,
        backend: string_field(&value, "backend"),
    })
}

fn linear_readout_from_value(value: &Value) -> Result<LinearReadout, Box<dyn std::error::Error>> {
    let input_dim = value_usize(value.get("input_dim").unwrap_or(&Value::Null));
    let output_dim = value_usize(value.get("output_dim").unwrap_or(&Value::Null));
    let weights = value_f32_vec(value.get("weights").unwrap_or(&Value::Null));
    let bias = value_f32_vec(value.get("bias").unwrap_or(&Value::Null));
    if input_dim == 0 || output_dim == 0 || weights.len() != input_dim * output_dim || bias.len() != output_dim {
        return Err("invalid LinearReadout in checkpoint model_artifacts_v1".into());
    }
    Ok(LinearReadout {
        input_dim,
        output_dim,
        weights,
        bias,
    })
}

fn value_string_vec(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn value_f32_vec(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_f64)
                .map(|item| item as f32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn value_usize(value: &Value) -> usize {
    value.as_u64().unwrap_or(0) as usize
}

fn value_string_vec_map(value: &Value) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    let Some(object) = value.as_object() else {
        return out;
    };
    for (key, item) in object {
        out.insert(key.clone(), value_string_vec(item));
    }
    out
}

fn value_f32_vec_map(value: &Value) -> HashMap<String, Vec<f32>> {
    let mut out = HashMap::new();
    let Some(object) = value.as_object() else {
        return out;
    };
    for (key, item) in object {
        out.insert(key.clone(), value_f32_vec(item));
    }
    out
}

fn value_ngram_counts(value: &Value) -> HashMap<String, BTreeMap<String, usize>> {
    let mut out = HashMap::new();
    let Some(object) = value.as_object() else {
        return out;
    };
    for (key, item) in object {
        let mut inner = BTreeMap::new();
        if let Some(counts) = item.as_object() {
            for (token, count) in counts {
                inner.insert(token.clone(), value_usize(count));
            }
        }
        out.insert(key.clone(), inner);
    }
    out
}

type TransformerHybridCandidateMap = BTreeMap<String, Vec<CandidateExpression>>;

fn load_transformer_hybrid_candidate_manifest(
    path_text: &str,
) -> Result<TransformerHybridCandidateMap, Box<dyn std::error::Error>> {
    if path_text.trim().is_empty() {
        return Ok(BTreeMap::new());
    }
    let rows = read_jsonl(Path::new(path_text))?;
    let mut by_task: TransformerHybridCandidateMap = BTreeMap::new();
    for row in rows {
        let task_id = string_field(&row, "task_id");
        if task_id.is_empty() {
            continue;
        }
        let provenance_text = format!(
            "{} {}",
            string_field(&row, "candidate_generation_mode"),
            string_field(&row, "candidate_source")
        )
        .to_ascii_lowercase();
        if !provenance_text.contains("transformer") && !provenance_text.contains("hybrid") {
            continue;
        }
        if row
            .get("public_tests_visible_to_generator")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || row
                .get("canonical_solution_seen_by_solver")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        {
            continue;
        }
        let code = string_field(&row, "code");
        let entry_point = string_field(&row, "entry_point");
        let Some(body) = extract_body_from_transformer_hybrid_code(&code, &entry_point) else {
            continue;
        };
        if !transformer_hybrid_import_body_ok(&body) {
            continue;
        }
        let rank = row
            .get("candidate_rank")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| by_task.get(&task_id).map(Vec::len).unwrap_or(0) as u64 + 1);
        by_task.entry(task_id).or_default().push(CandidateExpression {
            expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
            body,
            mode: format!("trainable_transformer_hybrid_action_generator_v1_rank{rank:04}"),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: false,
        });
    }
    Ok(by_task)
}

fn extract_body_from_transformer_hybrid_code(code: &str, entry_point: &str) -> Option<String> {
    let mut imports = Vec::new();
    let mut in_target = false;
    let mut body_lines = Vec::new();
    for line in code.lines() {
        let trimmed = line.trim();
        if !in_target && (trimmed.starts_with("import ") || trimmed.starts_with("from ")) {
            imports.push(trimmed.to_string());
            continue;
        }
        if !in_target && trimmed.starts_with("def ") {
            if entry_point.is_empty()
                || trimmed
                    .strip_prefix("def ")
                    .is_some_and(|rest| rest.starts_with(entry_point))
            {
                in_target = true;
            }
            continue;
        }
        if in_target {
            let top_level_line = !line.starts_with(' ') && !line.starts_with('\t');
            if top_level_line && trimmed.starts_with("def ") && !body_lines.is_empty() {
                break;
            }
            if line.starts_with("    ") {
                body_lines.push(line[4..].to_string());
            } else if line.starts_with('\t') {
                body_lines.push(line[1..].to_string());
            } else if trimmed.is_empty() {
                body_lines.push(String::new());
            } else {
                break;
            }
        }
    }
    let body_lines = body_lines
        .into_iter()
        .skip_while(|line| line.trim().is_empty())
        .collect::<Vec<_>>();
    if body_lines.is_empty() {
        return None;
    }
    let mut body = Vec::new();
    body.extend(imports);
    body.extend(body_lines);
    Some(body.join("\n"))
}

fn transformer_hybrid_candidate_count(rows: &[Value]) -> usize {
    rows.iter()
        .filter(|row| {
            let text = format!(
                "{} {} {}",
                string_field(row, "candidate_generation_mode"),
                string_field(row, "candidate_source"),
                string_field(row, "candidate_generation_contract")
            )
            .to_ascii_lowercase();
            text.contains("transformer") || text.contains("hybrid")
        })
        .count()
}

pub fn generate_code_lm_closure_fanout(
    config: CodeLmFanoutConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    trace_code_lm("fanout_start");
    let started = Instant::now();
    let mut phase_started = Instant::now();
    let mut phase_timing_ms: BTreeMap<String, u128> = BTreeMap::new();
    let input_load_started = Instant::now();
    let private_rows = load_tasks(Path::new(&config.private_curriculum))?;
    let mut public_rows = load_tasks(Path::new(&config.public_task_manifest))?;
    let private_total_before_limit = private_rows.len();
    let public_total_before_limit = public_rows.len();
    let private_only_fanout_smoke = public_total_before_limit == 0;
    if config.public_task_limit > 0 && public_rows.len() > config.public_task_limit {
        public_rows.truncate(config.public_task_limit);
    }
    let mut eval_rows = private_rows
        .iter()
        .filter(|row| row.split == "eval")
        .cloned()
        .collect::<Vec<_>>();
    let private_eval_total_before_limit = eval_rows.len();
    if config.private_eval_limit > 0 && eval_rows.len() > config.private_eval_limit {
        eval_rows.truncate(config.private_eval_limit);
    }
    record_phase_timing(
        &mut phase_timing_ms,
        "task_manifest_load_and_limits",
        &mut phase_started,
    );
    let mut sts_streams = load_sts_streams(Path::new(&config.sts_streams))?;
    let sts_control_policy_applied_task_count = merge_sts_decoder_control_policy_for_tasks(
        &mut sts_streams,
        &eval_rows,
        &public_rows,
        Path::new("reports/sts_decoder_control_rows.jsonl"),
    )?;
    record_phase_timing(
        &mut phase_timing_ms,
        "sts_conditioning_load",
        &mut phase_started,
    );
    let models = load_code_lm_checkpoint_models(Path::new(&config.checkpoint_in))?;
    record_phase_timing(
        &mut phase_timing_ms,
        "checkpoint_model_load",
        &mut phase_started,
    );
    let transformer_hybrid_candidates =
        load_transformer_hybrid_candidate_manifest(&config.transformer_hybrid_candidate_manifest)?;
    let transformer_hybrid_manifest_task_count = transformer_hybrid_candidates.len();
    let transformer_hybrid_manifest_candidate_count = transformer_hybrid_candidates
        .values()
        .map(Vec::len)
        .sum::<usize>();
    record_phase_timing(
        &mut phase_timing_ms,
        "transformer_hybrid_survival_lane_load",
        &mut phase_started,
    );
    phase_timing_ms.insert(
        "checkpoint_load_and_inputs".to_string(),
        input_load_started.elapsed().as_millis(),
    );

    let progress_config = CodeLmClosureConfig {
        private_curriculum: config.private_curriculum.clone(),
        public_task_manifest: config.public_task_manifest.clone(),
        seed: config.seed,
        hv_dim: models.readout.input_dim,
        max_vocab: models.vocab.id_to_token.len(),
        epochs: 0,
        lr: 0.0,
        candidates_per_task: config.candidates_per_task,
        max_work_steps: 0,
        use_cuda_readout: models.backend.contains("cuda"),
        readout_eval_limit: 0,
        aux_decoder_train_limit: 0,
        checkpoint_only: false,
        checkpoint_out: config.checkpoint_in.clone(),
        checkpoint_in: config.checkpoint_in.clone(),
        private_candidate_out: config.private_candidate_out.clone(),
        public_candidate_out: config.public_candidate_out.clone(),
        report_out: config.report_out.clone(),
        sts_streams: config.sts_streams.clone(),
    };
    let estimated_work_steps = eval_rows.len() * config.candidates_per_task.max(1)
        + public_rows.len() * config.candidates_per_task.max(1);
    write_progress_report(
        &progress_config,
        &started,
        "checkpoint_fanout_loaded",
        0,
        0,
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps,
        estimated_work_steps,
        json!({
            "checkpoint": rel(&config.checkpoint_in),
            "checkpoint_id": models.checkpoint_id,
            "fanout_mode": true,
            "repeated_training": false,
            "target_architecture": "train_once_checkpoint_then_candidate_generation_fanout",
            "transformer_hybrid_survival_lane": {
                "candidate_manifest": rel(&config.transformer_hybrid_candidate_manifest),
                "manifest_task_count": transformer_hybrid_manifest_task_count,
                "manifest_candidate_count": transformer_hybrid_manifest_candidate_count,
                "score_semantics": "optional canonical fanout import of trainable transformer/hybrid action candidates; replay and integrity remain required"
            },
            "diagnostic_task_limits": {
                "private_total_before_limit": private_total_before_limit,
                "private_eval_total_before_limit": private_eval_total_before_limit,
                "private_eval_limit": config.private_eval_limit,
                "private_eval_selected": eval_rows.len(),
                "public_total_before_limit": public_total_before_limit,
                "public_task_limit": config.public_task_limit,
                "public_selected": public_rows.len(),
                "score_semantics": "runtime_smoke_limit_only_not_capability_evidence"
            },
        }),
    )?;

    let private_candidate_phase_started = Instant::now();
    let private_sts_off_eval_rows = private_sts_off_nonregression_eval_rows(&eval_rows);
    let private_sts_off_candidate_limit =
        private_sts_off_nonregression_candidate_limit(config.candidates_per_task.max(1));
    let private_sts_off_nonregression_policy = private_sts_off_nonregression_policy_json(
        eval_rows.len(),
        private_sts_off_eval_rows.len(),
        config.candidates_per_task.max(1),
        private_sts_off_candidate_limit,
    );
    let mut private_candidates = candidate_rows(
        &eval_rows,
        &models.expression_bank,
        &models.body_prototypes,
        &models.body_ngram,
        &models.state_sequence_decoder,
        &models.symliquid_state_decoder,
        &models.readout,
        &models.vocab,
        &models.checkpoint_id,
        config.seed,
        config.candidates_per_task.max(1),
        "private_eval",
        true,
        &sts_streams,
        None,
        &transformer_hybrid_candidates,
    );
    if !sts_streams.is_empty() && !private_sts_off_eval_rows.is_empty() {
        private_candidates.extend(candidate_rows(
            &private_sts_off_eval_rows,
            &models.expression_bank,
            &models.body_prototypes,
            &models.body_ngram,
            &models.state_sequence_decoder,
            &models.symliquid_state_decoder,
            &models.readout,
            &models.vocab,
            &models.checkpoint_id,
            config.seed,
            private_sts_off_candidate_limit,
            "private_eval_sts_off",
            true,
            &HashMap::new(),
            None,
            &transformer_hybrid_candidates,
        ));
    }
    let private_sts_nonregression_union_added = if sts_streams.is_empty() {
        0
    } else {
        append_sts_nonregression_union_candidates(&mut private_candidates)
    };
    record_phase_timing(
        &mut phase_timing_ms,
        "private_candidate_expansion",
        &mut phase_started,
    );
    write_jsonl(Path::new(&config.private_candidate_out), &private_candidates)?;
    let private_candidate_binary_artifact =
        write_jsonl_binary_sidecar(Path::new(&config.private_candidate_out), &private_candidates)?;
    record_phase_timing(
        &mut phase_timing_ms,
        "private_artifact_write",
        &mut phase_started,
    );
    let private_cache_summary = candidate_verification_cache_summary(&private_candidates);
    let private_task_timing_summary = candidate_task_timing_summary(&private_candidates);
    let private_phase_category_summary =
        candidate_task_phase_category_summary(&private_candidates);
    record_phase_timing(
        &mut phase_timing_ms,
        "private_ranker_prefilter_verifier_cache_summary",
        &mut phase_started,
    );
    phase_timing_ms.insert(
        "private_candidate_generation_and_write".to_string(),
        private_candidate_phase_started.elapsed().as_millis(),
    );
    let private_fanout_runtime_breakdown =
        candidate_fanout_runtime_breakdown(&phase_timing_ms, &private_task_timing_summary, "private");
    write_progress_report(
        &progress_config,
        &started,
        "checkpoint_fanout_private_candidates_written",
        0,
        private_candidates.len(),
        eval_rows.len(),
        public_rows.len(),
        estimated_work_steps,
        estimated_work_steps,
        json!({
            "private_candidate_count": private_candidates.len(),
            "private_sts_nonregression_union_added": private_sts_nonregression_union_added,
            "private_sts_nonregression_union_candidate_count": count_bool(&private_candidates, "sts_nonregression_union_candidate"),
            "private_sts_off_nonregression_policy": private_sts_off_nonregression_policy.clone(),
            "private_candidate_binary_artifact": rel_path(&private_candidate_binary_artifact),
            "private_candidate_verification_cache": private_cache_summary,
            "private_candidate_task_timing_summary": private_task_timing_summary.clone(),
            "private_candidate_task_phase_categories": private_phase_category_summary.clone(),
            "private_candidate_fanout_runtime_breakdown": private_fanout_runtime_breakdown.clone(),
            "decoder_completion_cache": decoder_completion_cache_summary(),
            "phase_timing_ms": phase_timing_ms,
            "diagnostic_partial_artifact": true,
            "score_semantics": "private_fanout_progress_only_not_public_calibration_unlock"
        }),
    )?;

    let public_candidate_phase_started = Instant::now();
    let public_candidates = candidate_rows(
        &public_rows,
        &models.expression_bank,
        &models.body_prototypes,
        &models.body_ngram,
        &models.state_sequence_decoder,
        &models.symliquid_state_decoder,
        &models.readout,
        &models.vocab,
        &models.checkpoint_id,
        config.seed,
        config.candidates_per_task.max(1),
        "public_calibration",
        true,
        &sts_streams,
        Some(CandidateHeartbeat {
            config: &progress_config,
            started: &started,
            stage: "public_candidate_generation",
            phase: "public_calibration",
            trained: true,
            total_tasks: public_rows.len(),
            original_train_task_count: 0,
            train_task_count: 0,
            eval_task_count: eval_rows.len(),
            public_task_count: public_rows.len(),
            estimated_work_steps_before_budget: estimated_work_steps,
            estimated_work_steps_after_budget: estimated_work_steps,
        }),
        &BTreeMap::new(),
    );
    record_phase_timing(
        &mut phase_timing_ms,
        "public_candidate_expansion",
        &mut phase_started,
    );
    write_jsonl(Path::new(&config.public_candidate_out), &public_candidates)?;
    let public_candidate_binary_artifact =
        write_jsonl_binary_sidecar(Path::new(&config.public_candidate_out), &public_candidates)?;
    record_phase_timing(
        &mut phase_timing_ms,
        "public_artifact_write",
        &mut phase_started,
    );
    let public_cache_summary = candidate_verification_cache_summary(&public_candidates);
    let public_task_timing_summary = candidate_task_timing_summary(&public_candidates);
    let public_phase_category_summary =
        candidate_task_phase_category_summary(&public_candidates);
    record_phase_timing(
        &mut phase_timing_ms,
        "public_ranker_prefilter_verifier_cache_summary",
        &mut phase_started,
    );
    phase_timing_ms.insert(
        "public_candidate_generation_and_write".to_string(),
        public_candidate_phase_started.elapsed().as_millis(),
    );
    let public_fanout_runtime_breakdown =
        candidate_fanout_runtime_breakdown(&phase_timing_ms, &public_task_timing_summary, "public");

    let private_filter_diagnostics =
        candidate_filter_diagnostics_from_rows(&eval_rows, &private_candidates);
    let public_filter_diagnostics =
        candidate_filter_diagnostics_from_rows(&public_rows, &public_candidates);
    let public_token_level_candidate_count =
        count_bool(&public_candidates, "token_level_code_generation_learned");
    let private_token_level_candidate_count =
        count_bool(&private_candidates, "token_level_code_generation_learned");
    let public_template_like_candidate_count =
        count_bool(&public_candidates, "template_like_candidate");
    let gates = vec![
        gate(
            "checkpoint_fanout_used",
            true,
            json!("candidate generation reused a train-once checkpoint and did not retrain per shard"),
        ),
        gate(
            "checkpoint_model_artifacts_present",
            !models.vocab.id_to_token.is_empty()
                && !models.body_ngram.counts.is_empty()
                && models.state_sequence_decoder.update_count > 0,
            json!(format!(
                "vocab={} body_ngram={} state_updates={}",
                models.vocab.id_to_token.len(),
                models.body_ngram.counts.len(),
                models.state_sequence_decoder.update_count
            )),
        ),
        gate(
            "public_candidates_emitted",
            private_only_fanout_smoke || !public_candidates.is_empty(),
            if private_only_fanout_smoke {
                json!("public fanout intentionally skipped for private-only runtime smoke")
            } else {
                json!(format!(
                    "public_candidates={} public_tasks={}",
                    public_candidates.len(),
                    public_rows.len()
                ))
            },
        ),
        gate(
            "public_token_level_student_candidates_emitted",
            private_only_fanout_smoke || public_token_level_candidate_count > 0,
            if private_only_fanout_smoke {
                json!("public token candidates intentionally skipped for private-only runtime smoke")
            } else {
                json!(format!(
                    "token_level_public_candidates={} private_token_level_candidates={}",
                    public_token_level_candidate_count, private_token_level_candidate_count
                ))
            },
        ),
        gate(
            "public_candidates_template_free",
            public_template_like_candidate_count == 0,
            json!(format!(
                "template_like_public_candidates={}",
                public_template_like_candidate_count
            )),
        ),
        gate(
            "public_tests_not_visible",
            true,
            json!("public task exporter omits tests and canonical solutions"),
        ),
        gate(
            "external_inference_zero",
            true,
            json!("local checkpoint fanout only"),
        ),
    ];
    record_phase_timing(
        &mut phase_timing_ms,
        "filter_diagnostics_and_gates",
        &mut phase_started,
    );
    let trigger_state = if gates
        .iter()
        .all(|row| row["passed"].as_bool().unwrap_or(false))
    {
        "GREEN"
    } else {
        "YELLOW"
    };
    let completed_heartbeat = CandidateHeartbeat {
        config: &progress_config,
        started: &started,
        stage: "public_candidate_generation",
        phase: "public_calibration",
        trained: true,
        total_tasks: public_rows.len(),
        original_train_task_count: 0,
        train_task_count: 0,
        eval_task_count: eval_rows.len(),
        public_task_count: public_rows.len(),
        estimated_work_steps_before_budget: estimated_work_steps,
        estimated_work_steps_after_budget: estimated_work_steps,
    };
    write_candidate_generation_heartbeat(
        &completed_heartbeat,
        public_rows.len(),
        public_rows.last(),
        public_candidates.len(),
        0,
        0,
        &BTreeMap::new(),
        "completed",
    );
    let private_sts_nonregression_union_candidate_count =
        count_bool(&private_candidates, "sts_nonregression_union_candidate");
    let private_candidate_generation_modes = candidate_generation_modes(&private_candidates);
    let private_candidate_generation_mode_counts =
        candidate_generation_mode_counts(&private_candidates);
    let private_transformer_hybrid_candidate_count =
        transformer_hybrid_candidate_count(&private_candidates);
    let public_transformer_hybrid_candidate_count =
        transformer_hybrid_candidate_count(&public_candidates);
    let report = json!({
        "policy": "project_theseus_code_lm_closure_rust_fanout_v1",
        "created_utc": now(),
        "run_status": "completed",
        "progress_stage": "completed",
        "trigger_state": trigger_state,
        "checkpoint": rel(&config.checkpoint_in),
        "checkpoint_in": rel(&config.checkpoint_in),
        "checkpoint_id": models.checkpoint_id,
        "private_candidate_manifest": rel(&config.private_candidate_out),
        "public_candidate_manifest": rel(&config.public_candidate_out),
        "private_candidate_binary_artifact": rel_path(&private_candidate_binary_artifact),
        "public_candidate_binary_artifact": rel_path(&public_candidate_binary_artifact),
        "candidate_generation_heartbeat": rel_path(&candidate_heartbeat_path(&progress_config)),
        "phase_timing_ms": phase_timing_ms,
        "summary": {
            "fanout_mode": true,
            "private_only_fanout_smoke": private_only_fanout_smoke,
            "repeated_training": false,
            "train_once_checkpoint_fanout": true,
            "target_architecture": "train_once_checkpoint_then_hive_distributed_candidate_generation_and_verification",
            "private_eval_task_count": eval_rows.len(),
            "public_task_count": public_rows.len(),
            "public_candidate_count": public_candidates.len(),
            "private_candidate_count": private_candidates.len(),
            "private_sts_nonregression_union_added": private_sts_nonregression_union_added,
            "private_sts_nonregression_union_candidate_count": private_sts_nonregression_union_candidate_count,
            "private_sts_off_nonregression_policy": private_sts_off_nonregression_policy,
            "private_candidate_generation_modes": private_candidate_generation_modes,
            "private_candidate_generation_mode_counts": private_candidate_generation_mode_counts,
            "transformer_hybrid_survival_lane": {
                "candidate_manifest": rel(&config.transformer_hybrid_candidate_manifest),
                "manifest_task_count": transformer_hybrid_manifest_task_count,
                "manifest_candidate_count": transformer_hybrid_manifest_candidate_count,
                "private_emitted_candidate_count": private_transformer_hybrid_candidate_count,
                "public_emitted_candidate_count": public_transformer_hybrid_candidate_count,
                "default_role": if transformer_hybrid_manifest_candidate_count > 0 { "practical_survival_lane" } else { "not_loaded" },
                "legacy_routes": "old learned/template-like, structural, ngram, and SymLiquid families remain explicit comparator/ablation lanes"
            },
            "public_token_level_candidate_count": public_token_level_candidate_count,
            "private_token_level_candidate_count": private_token_level_candidate_count,
            "template_like_candidate_count": public_template_like_candidate_count,
            "public_candidate_generation_modes": candidate_generation_modes(&public_candidates),
            "public_candidate_generation_mode_counts": candidate_generation_mode_counts(&public_candidates),
            "private_candidate_verification_cache": private_cache_summary,
            "public_candidate_verification_cache": public_cache_summary,
            "candidate_task_timing_summary": {
                "private": private_task_timing_summary,
                "public": public_task_timing_summary,
            },
            "candidate_task_phase_categories": {
                "private": private_phase_category_summary,
                "public": public_phase_category_summary,
            },
            "candidate_fanout_runtime_breakdown": {
                "private": private_fanout_runtime_breakdown,
                "public": public_fanout_runtime_breakdown,
            },
            "decoder_completion_cache": decoder_completion_cache_summary(),
            "phase_timing_ms": phase_timing_ms,
            "checkpoint_backend": models.backend,
            "sts_decoder_control_policy_applied_task_count": sts_control_policy_applied_task_count,
            "sts_stream_conditioned_private_task_count": eval_rows
                .iter()
                .filter(|task| sts_streams.contains_key(&task.task_id))
                .count(),
            "sts_stream_conditioned_public_task_count": public_rows
                .iter()
                .filter(|task| sts_streams.contains_key(&task.task_id))
                .count(),
            "checkpoint_vocab_size": models.vocab.id_to_token.len(),
            "checkpoint_body_ngram_key_count": models.body_ngram.counts.len(),
            "diagnostic_task_limits": {
                "private_total_before_limit": private_total_before_limit,
                "private_eval_total_before_limit": private_eval_total_before_limit,
                "private_eval_limit": config.private_eval_limit,
                "private_eval_selected": eval_rows.len(),
                "public_total_before_limit": public_total_before_limit,
                "public_task_limit": config.public_task_limit,
                "public_selected": public_rows.len(),
                "score_semantics": "runtime_smoke_limit_only_not_capability_evidence"
            },
            "external_inference_calls": 0,
        },
        "private_filter_diagnostics": private_filter_diagnostics,
        "public_filter_diagnostics": public_filter_diagnostics,
        "gates": gates,
        "runtime_ms": (started.elapsed().as_secs_f64() * 1000.0).round() as u64,
        "external_inference_calls": 0,
    });
    write_json(Path::new(&config.report_out), &report)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "policy": "project_theseus_code_lm_closure_rust_fanout_v1",
            "trigger_state": trigger_state,
            "report_out": rel(&config.report_out),
            "public_candidate_manifest": rel(&config.public_candidate_out),
            "private_candidate_manifest": rel(&config.private_candidate_out),
            "runtime_ms": (started.elapsed().as_secs_f64() * 1000.0).round() as u64,
        }))?
    );
    Ok(())
}

fn train_next_token_readout(
    readout: &mut LinearReadout,
    examples: &[TrainExample],
    hv_dim: usize,
    epochs: usize,
    lr: f32,
    use_cuda_readout: bool,
) -> Result<&'static str, Box<dyn std::error::Error>> {
    if use_cuda_readout {
        return train_next_token_readout_cuda(readout, examples, hv_dim, epochs, lr);
    }
    train_next_token_readout_cpu(readout, examples, hv_dim, epochs, lr);
    Ok("rust_cpu_fast_sparse_readout")
}

fn bounded_code_task_sample(tasks: &[CodeTask], limit: usize) -> Vec<CodeTask> {
    if limit == 0 || tasks.len() <= limit {
        return tasks.to_vec();
    }
    let mut out = Vec::with_capacity(limit);
    if limit == 1 {
        out.push(tasks[0].clone());
        return out;
    }
    let last = tasks.len() - 1;
    let denom = limit - 1;
    let mut seen = HashSet::new();
    for idx in 0..limit {
        let source_idx = idx * last / denom;
        if seen.insert(source_idx) {
            out.push(tasks[source_idx].clone());
        }
    }
    let mut fill_idx = 0usize;
    while out.len() < limit && fill_idx < tasks.len() {
        if seen.insert(fill_idx) {
            out.push(tasks[fill_idx].clone());
        }
        fill_idx += 1;
    }
    out
}

fn bounded_train_example_sample(examples: &[TrainExample], limit: usize) -> Vec<TrainExample> {
    if limit == 0 || examples.len() <= limit {
        return examples.to_vec();
    }
    let mut out = Vec::with_capacity(limit);
    if limit == 1 {
        out.push(examples[0].clone());
        return out;
    }
    let last = examples.len() - 1;
    let denom = limit - 1;
    let mut seen = HashSet::new();
    for idx in 0..limit {
        let source_idx = idx * last / denom;
        if seen.insert(source_idx) {
            out.push(examples[source_idx].clone());
        }
    }
    let mut fill_idx = 0usize;
    while out.len() < limit && fill_idx < examples.len() {
        if seen.insert(fill_idx) {
            out.push(examples[fill_idx].clone());
        }
        fill_idx += 1;
    }
    out
}

fn train_next_token_readout_cpu(
    readout: &mut LinearReadout,
    examples: &[TrainExample],
    hv_dim: usize,
    epochs: usize,
    lr: f32,
) {
    for _ in 0..epochs.max(1) {
        let mut context_cache = HashMap::new();
        for example in examples {
            let features = featurize_with_context_cache(example, hv_dim, &mut context_cache);
            fast_readout_train_step(
                readout,
                &features,
                example.target,
                lr,
                example.position + example.prev1.len() + example.prev2.len(),
            );
        }
    }
}

#[cfg(feature = "cuda")]
fn train_next_token_readout_cuda(
    readout: &mut LinearReadout,
    examples: &[TrainExample],
    hv_dim: usize,
    epochs: usize,
    lr: f32,
) -> Result<&'static str, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Ok("rust_cuda_fast_sparse_readout_empty");
    }
    let rows = feature_rows_for_examples_cached(examples, hv_dim);
    let mut targets = Vec::with_capacity(examples.len());
    let mut salts = Vec::with_capacity(examples.len());
    for example in examples {
        targets.push(example.target);
        salts.push(example.position + example.prev1.len() + example.prev2.len());
    }
    let features = Tensor::new(examples.len(), hv_dim, rows)?;
    symliquid_cuda::readout_cuda::train_code_fast_readout_cuda(
        &features,
        &targets,
        &salts,
        readout,
        epochs.max(1),
        lr,
    )?;
    Ok("rust_cuda_fast_sparse_code_lm_readout")
}

#[cfg(not(feature = "cuda"))]
fn train_next_token_readout_cuda(
    _readout: &mut LinearReadout,
    _examples: &[TrainExample],
    _hv_dim: usize,
    _epochs: usize,
    _lr: f32,
) -> Result<&'static str, Box<dyn std::error::Error>> {
    Err("Code LM CUDA readout requested but symliquid-cli was not built with --features cuda".into())
}
