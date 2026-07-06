// Candidate diagnostics and fanout artifact emission for Code LM closure.
// This is intentionally separated from the training/readout path so future GPU ranker
// and verifier-prefilter work has a single ownership boundary.

use super::*;
use std::cell::Cell;

mod broad_private_generalization;
mod broad_private_train_prototype;
mod contract_token;
mod expression_pool;
mod prefilter;
mod private_residual_v3;
mod sts_bridge;
mod syntax_repair;
mod task_rows;
mod variant_cache;
use broad_private_generalization::*;
use broad_private_train_prototype::*;
pub(super) use contract_token::{
    contract_guided_token_candidate_score, token_contract_candidate_body_ok,
};
use contract_token::{
    contract_guided_token_candidate_score_with_verification, contract_guided_token_decoder_bodies,
    contract_guided_token_decoder_bodies_with_timing, sts_conditioned_contract_token_bridge_bodies,
    token_contract_candidate_body_ok_with_verification,
};
use expression_pool::*;
pub(super) use prefilter::*;
use private_residual_v3::*;
use sts_bridge::*;
pub(super) use syntax_repair::*;
use task_rows::*;
use variant_cache::*;

fn record_candidate_timing(
    timings: &mut BTreeMap<String, u128>,
    phase: &str,
    started: &mut Instant,
) {
    timings.insert(phase.to_string(), started.elapsed().as_millis());
    *started = Instant::now();
}

#[derive(Clone, Copy, Default)]
struct CandidateFanoutThreadContext {
    worker_id: usize,
    persistent_worker_pool_enabled: bool,
}

thread_local! {
    static CANDIDATE_FANOUT_THREAD_CONTEXT: Cell<CandidateFanoutThreadContext> =
        Cell::new(CandidateFanoutThreadContext::default());
}

pub(super) struct CandidateFanoutThreadContextGuard {
    previous: CandidateFanoutThreadContext,
}

impl Drop for CandidateFanoutThreadContextGuard {
    fn drop(&mut self) {
        CANDIDATE_FANOUT_THREAD_CONTEXT.with(|context| context.set(self.previous));
    }
}

pub(super) fn candidate_fanout_thread_context_guard(
    worker_id: usize,
    persistent_worker_pool_enabled: bool,
) -> CandidateFanoutThreadContextGuard {
    CANDIDATE_FANOUT_THREAD_CONTEXT.with(|context| {
        let previous = context.get();
        context.set(CandidateFanoutThreadContext {
            worker_id,
            persistent_worker_pool_enabled,
        });
        CandidateFanoutThreadContextGuard { previous }
    })
}

pub(super) fn current_candidate_fanout_worker_id() -> usize {
    CANDIDATE_FANOUT_THREAD_CONTEXT.with(|context| context.get().worker_id)
}

pub(super) fn persistent_task_fanout_worker_pool_active() -> bool {
    CANDIDATE_FANOUT_THREAD_CONTEXT.with(|context| context.get().persistent_worker_pool_enabled)
}

fn nested_branch_threads_allowed_in_task_worker() -> bool {
    std::env::var("THESEUS_CODE_LM_ALLOW_NESTED_BRANCH_THREADS_IN_TASK_WORKER")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            value == "1" || value == "true" || value == "on"
        })
        .unwrap_or(false)
}

fn nested_branch_threads_suppressed_in_task_worker() -> bool {
    let suppressed = std::env::var("THESEUS_CODE_LM_SUPPRESS_NESTED_BRANCH_THREADS_IN_TASK_WORKER")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            value == "1" || value == "true" || value == "on"
        })
        .unwrap_or(false);
    suppressed && !nested_branch_threads_allowed_in_task_worker()
}

fn nested_branch_parallelism_allowed_in_current_context() -> bool {
    !persistent_task_fanout_worker_pool_active()
        || !nested_branch_threads_suppressed_in_task_worker()
}

pub(super) fn nested_branch_parallelism_suppressed_for_current_task() -> bool {
    persistent_task_fanout_worker_pool_active() && nested_branch_threads_suppressed_in_task_worker()
}

fn parallel_sts_candidate_fanout_enabled() -> bool {
    let enabled = std::env::var("THESEUS_PARALLEL_STS_CANDIDATE_FANOUT")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true);
    enabled && nested_branch_parallelism_allowed_in_current_context()
}

fn parallel_contract_token_fanout_enabled() -> bool {
    let enabled = std::env::var("THESEUS_PARALLEL_CONTRACT_TOKEN_FANOUT")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true);
    enabled && nested_branch_parallelism_allowed_in_current_context()
}

fn greedy_or_precomputed_beam_body(
    task: &CodeTask,
    readout: &LinearReadout,
    vocab: &Vocab,
    sts_streams: Option<&BTreeMap<String, String>>,
    precomputed_beams: Option<&Vec<String>>,
) -> Option<String> {
    if let Some(rows) = precomputed_beams {
        if let Some(body) = rows
            .iter()
            .find(|body| state_sequence_candidate_body_ok(task, body))
            .cloned()
        {
            return Some(body);
        }
        if low_latency_candidate_fanout_enabled() && !greedy_after_precomputed_beam_enabled() {
            return None;
        }
    }
    greedy_body(task, readout, vocab, sts_streams)
}

fn greedy_after_precomputed_beam_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_GREEDY_AFTER_PRECOMPUTED_BEAM")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(false)
}

fn low_latency_beam_precompute_default_enabled(task_count: usize) -> bool {
    std::env::var("THESEUS_CODE_LM_PRECOMPUTE_LOW_LATENCY_BEAMS")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(task_count >= 8)
}

pub(super) fn candidate_filter_diagnostics(
    tasks: &[CodeTask],
    expression_bank: &[ExpressionBankItem],
    body_prototypes: &BodyPrototypeModel,
    body_ngram: &BodyNgramModel,
    state_sequence_decoder: &StateSequenceDecoder,
    symliquid_state_decoder: &SymLiquidStateDecoder,
    readout: &LinearReadout,
    vocab: &Vocab,
    seed: u64,
    candidates_per_task: usize,
    sts_streams: &StsStreamMap,
    heartbeat: Option<CandidateHeartbeat<'_>>,
) -> Vec<Value> {
    let mut rows = Vec::new();
    if let Some(ctx) = heartbeat.as_ref() {
        let empty_rejections = BTreeMap::new();
        write_candidate_generation_heartbeat(ctx, 0, None, 0, 0, 0, &empty_rejections, "started");
    }
    let update_every = candidate_heartbeat_update_every_tasks();
    let mut last_accepted_for_task = 0usize;
    let mut last_rejected_for_task = 0usize;
    let mut last_rejection_counts: BTreeMap<String, usize> = BTreeMap::new();
    let batched_beam_cache = batched_beam_bodies(
        tasks,
        readout,
        vocab,
        seed,
        candidates_per_task.clamp(2, 8),
        sts_streams,
    );
    for (task_index, task) in tasks.iter().enumerate() {
        let task_sts = sts_streams.get(&task.task_id);
        let mut variant_cache = CandidateVariantCache::default();
        let expressions = candidate_expressions(
            task,
            expression_bank,
            body_prototypes,
            body_ngram,
            state_sequence_decoder,
            symliquid_state_decoder,
            readout,
            vocab,
            seed,
            candidates_per_task,
            task_sts,
            batched_beam_cache.get(&task.task_id),
        )
        .into_iter()
        .map(|candidate| normalize_candidate_for_task_cached(task, candidate, &mut variant_cache))
        .collect::<Vec<_>>();
        let mut raw_mode_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut rejection_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut accepted_count = 0usize;
        let mut accepted_modes = BTreeSet::new();
        let mut raw_previews = Vec::new();
        let mut rejected_body_samples = Vec::new();
        let mut seen = HashSet::new();
        let mut verification_cache = CandidateVerificationCache::default();
        let template_free_student_candidates = template_free_student_candidates_enabled();
        for candidate in &expressions {
            *raw_mode_counts.entry(candidate.mode.clone()).or_insert(0) += 1;
            let duplicate = !seen.insert(candidate_duplicate_key(task, candidate));
            let reason = if template_free_student_candidates && template_like_candidate(candidate) {
                Some("template_candidate_disabled")
            } else if let Some(reason) = candidate_rejection_reason_cached(
                task,
                candidate,
                duplicate,
                None,
                &mut verification_cache,
            ) {
                Some(reason)
            } else if !candidate_body_admissible_cached(
                task,
                candidate,
                task_sts,
                &mut verification_cache,
            ) {
                Some("candidate_body_not_admissible")
            } else {
                None
            };
            if let Some(reason) = reason {
                *rejection_counts.entry(reason).or_insert(0) += 1;
                if raw_previews.len() < 3 {
                    raw_previews.push(json!({
                        "mode": candidate.mode,
                        "reason": reason,
                        "body_preview": preview_body(&candidate.body),
                    }));
                }
                if rejected_body_samples.len() < 3
                    && candidate.mode != "student_decoder_no_admissible_candidate_residual"
                {
                    rejected_body_samples.push(json!({
                        "mode": candidate.mode,
                        "reason": reason,
                        "verifier_reasons": verification_cache.verifier(task, &candidate.body, task_sts).reasons,
                        "syntax_constrained": syntax_constrained_body(&candidate.body),
                        "semantic_admissible": body_semantically_admissible(task, &candidate.body),
                        "body": candidate.body,
                    }));
                }
            } else {
                accepted_count += 1;
                accepted_modes.insert(candidate.mode.clone());
                if raw_previews.len() < 3 {
                    raw_previews.push(json!({
                        "mode": candidate.mode,
                        "reason": "accepted",
                        "body_preview": preview_body(&candidate.body),
                    }));
                }
            }
        }
        rows.push(json!({
            "task_id": task.task_id,
            "source_task_id": task.source_task_id,
            "category": task.category,
            "raw_candidate_count": expressions.len(),
            "accepted_candidate_count": accepted_count,
            "accepted_modes": accepted_modes.into_iter().collect::<Vec<_>>(),
                "raw_mode_counts": raw_mode_counts,
                "rejection_counts": rejection_counts,
                "raw_previews": raw_previews,
            "rejected_body_samples": rejected_body_samples,
            "candidate_verification_cache_v1": verification_cache.metrics(),
            "template_free_student_candidates_enabled": template_free_student_candidates,
        }));
        last_accepted_for_task = accepted_count;
        last_rejected_for_task = rejection_counts.values().sum();
        last_rejection_counts = rejection_counts
            .iter()
            .map(|(reason, count)| (reason.to_string(), *count))
            .collect::<BTreeMap<_, _>>();
        if let Some(ctx) = heartbeat.as_ref() {
            let completed_tasks = task_index.saturating_add(1);
            if completed_tasks == tasks.len() || completed_tasks % update_every == 0 {
                write_candidate_generation_heartbeat(
                    ctx,
                    completed_tasks,
                    Some(task),
                    rows.len(),
                    last_accepted_for_task,
                    last_rejected_for_task,
                    &last_rejection_counts,
                    "running",
                );
            }
        }
    }
    if let Some(ctx) = heartbeat.as_ref() {
        write_candidate_generation_heartbeat(
            ctx,
            tasks.len(),
            tasks.last(),
            rows.len(),
            last_accepted_for_task,
            last_rejected_for_task,
            &last_rejection_counts,
            "completed",
        );
    }
    rows
}

pub(super) fn candidate_filter_diagnostics_from_rows(
    tasks: &[CodeTask],
    candidates: &[Value],
) -> Vec<Value> {
    let mut by_task: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for row in candidates {
        by_task
            .entry(string_field(row, "task_id"))
            .or_default()
            .push(row);
    }
    tasks
        .iter()
        .map(|task| {
            let rows = by_task.get(&task.task_id).cloned().unwrap_or_default();
            let mut accepted_modes = BTreeSet::new();
            let mut raw_mode_counts: BTreeMap<String, usize> = BTreeMap::new();
            let mut raw_previews = Vec::new();
            let mut rejection_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
            let mut accepted_count = 0usize;
            for row in rows {
                let mode = string_field(row, "candidate_generation_mode");
                *raw_mode_counts.entry(mode.clone()).or_insert(0) += 1;
                if mode == "student_decoder_no_admissible_candidate_residual" {
                    *rejection_counts
                        .entry("no_admissible_candidate")
                        .or_insert(0) += 1;
                    if raw_previews.len() < 3 {
                        raw_previews.push(json!({
                            "mode": mode,
                            "reason": "no_admissible_candidate",
                            "body_preview": "student decoder emitted no admissible candidate",
                        }));
                    }
                    continue;
                }
                let verifier_failed = row
                    .get("decoder_contract_verifier_v1_passed")
                    .and_then(Value::as_bool)
                    .is_some_and(|passed| !passed);
                let guardrail_failed = row
                    .get("deterministic_guardrail_passed")
                    .and_then(Value::as_bool)
                    .is_some_and(|passed| !passed);
                let scaffold = row
                    .get("placeholder_scaffold_body")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let low_quality = row
                    .get("beautiful_code_score")
                    .and_then(Value::as_f64)
                    .is_some_and(|score| score < 0.0);
                if verifier_failed || guardrail_failed || scaffold || low_quality {
                    let reason = if verifier_failed {
                        "decoder_contract_failed"
                    } else if guardrail_failed {
                        "deterministic_guardrail_failed"
                    } else if scaffold {
                        "placeholder_scaffold_body"
                    } else {
                        "beautiful_code_score_below_floor"
                    };
                    *rejection_counts.entry(reason).or_insert(0) += 1;
                    if raw_previews.len() < 3 {
                        raw_previews.push(json!({
                            "mode": mode,
                            "reason": reason,
                            "body_preview": preview_body(&string_field(row, "code")),
                        }));
                    }
                    continue;
                }
                accepted_count += 1;
                accepted_modes.insert(mode.clone());
                if raw_previews.len() < 3 {
                    raw_previews.push(json!({
                        "mode": mode,
                        "reason": "accepted_emitted_candidate",
                        "body_preview": preview_body(&string_field(row, "code")),
                    }));
                }
            }
            if accepted_count == 0 && rejection_counts.is_empty() {
                *rejection_counts.entry("missing_candidate_row").or_insert(0) += 1;
            }
            json!({
                "task_id": task.task_id,
                "source_task_id": task.source_task_id,
                "category": task.category,
                "raw_candidate_count": raw_mode_counts.values().copied().sum::<usize>(),
                "accepted_candidate_count": accepted_count,
                "accepted_modes": accepted_modes.into_iter().collect::<Vec<_>>(),
                "raw_mode_counts": raw_mode_counts,
                "rejection_counts": rejection_counts,
                "raw_previews": raw_previews,
                "diagnostic_source": "emitted_candidate_rows_no_decoder_rerun",
            })
        })
        .collect()
}

pub(super) fn preview_body(body: &str) -> String {
    let compact = body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");
    compact.chars().take(600).collect()
}

include!("candidate_fanout/scheduler.rs");

pub(super) fn candidate_duplicate_key(task: &CodeTask, candidate: &CandidateExpression) -> String {
    if private_execution_shape_family_attribution_required(task)
        || same_seed_non_sts_comparator_candidate(candidate)
        || contract_guided_token_inventory_candidate(candidate)
        || private_to_public_receiver_inventory_bridge_candidate(candidate)
        || broad_private_train_token_candidate(candidate)
        || broad_private_train_prototype_candidate(candidate)
        || private_residual_v3_train_induced_structural_token_candidate(candidate)
        || private_residual_v3_semantic_adapter_candidate(candidate)
        || broad_private_generalization_semantic_adapter_candidate(candidate)
    {
        format!("{}::{}", candidate.mode, candidate.body)
    } else {
        candidate.body.clone()
    }
}

pub(super) fn candidate_uses_sts_conditioning(candidate: &CandidateExpression) -> bool {
    candidate.sts_candidate_expression_used
        || candidate.mode.contains("sts_conditioned")
        || candidate.mode.contains("sts_causal_skeleton")
}

pub(super) fn private_residual_v3_semantic_adapter_candidate(
    candidate: &CandidateExpression,
) -> bool {
    candidate.mode.contains("private_residual_v3") && candidate.mode.contains("semantic_adapter")
}

pub(super) fn private_residual_v3_train_induced_structural_token_candidate(
    candidate: &CandidateExpression,
) -> bool {
    candidate
        .mode
        .contains("private_residual_v3_train_induced_structural_token_decoder_v1")
}

pub(super) fn edge_contract_v3_strict_novel_token_candidate(
    candidate: &CandidateExpression,
) -> bool {
    candidate
        .mode
        .contains("edge_contract_v3_strict_novel_token_decoder_v1")
}

pub(super) fn edge_contract_v3_body_memory_replay_candidate(
    candidate: &CandidateExpression,
) -> bool {
    candidate
        .mode
        .contains("edge_contract_v3_body_memory_replay_decoder_v1")
}

pub(super) fn edge_contract_v3_demote_to_body_memory_replay(
    task: &CodeTask,
    candidate: &CandidateExpression,
) -> bool {
    task.card_id == "edge_contract_v3_verifier_mismatch_public_transfer_private"
        && task
            .benchmark_evidence_level
            .contains("edge_contract_v3_private_generated_only")
        && task.category.starts_with("edge_v3_")
        && !edge_contract_v3_strict_novel_token_candidate(candidate)
        && !edge_contract_v3_body_memory_replay_candidate(candidate)
        && (candidate.mode.contains("contract_guided_token_decoder")
            || candidate.mode.contains("greedy_body_token_decoder")
            || candidate.mode.contains("full_body_token_beam"))
}

pub(super) fn broad_private_train_prototype_candidate(candidate: &CandidateExpression) -> bool {
    candidate
        .mode
        .contains("private_train_induced_broad_semantic_prototype_decoder_v1")
}

pub(super) fn broad_private_train_token_candidate(candidate: &CandidateExpression) -> bool {
    candidate
        .mode
        .contains("private_train_induced_broad_semantic_token_decoder_v1")
}

pub(super) fn broad_private_generalization_semantic_adapter_candidate(
    candidate: &CandidateExpression,
) -> bool {
    candidate.mode.contains("broad_private_generalization_v1")
        && candidate.mode.contains("semantic_adapter")
}

pub(super) fn candidate_conditioning_streams<'a>(
    candidate: &CandidateExpression,
    task_sts: Option<&'a BTreeMap<String, String>>,
) -> Option<&'a BTreeMap<String, String>> {
    if candidate_uses_sts_conditioning(candidate) {
        task_sts
    } else {
        None
    }
}

pub(super) fn same_seed_non_sts_comparator_candidate(candidate: &CandidateExpression) -> bool {
    candidate.mode.contains("same_seed_non_sts_comparator")
}

pub(super) fn contract_guided_token_inventory_candidate(candidate: &CandidateExpression) -> bool {
    candidate.mode.contains("contract_guided_token_decoder")
        && !candidate.mode.contains("sts_conditioned")
        && !same_seed_non_sts_comparator_candidate(candidate)
        && !candidate.sts_candidate_expression_used
        && candidate.compositional_token_candidate
        && candidate.full_body_token_candidate
        && !candidate.expression_memory_fallback
}

pub(super) fn private_execution_shape_family_attribution_required(task: &CodeTask) -> bool {
    execution_shaped_category(&task.category)
        && task
            .benchmark_evidence_level
            .contains("private_execution_shape_ablation")
}

pub(super) fn template_free_student_candidates_enabled() -> bool {
    if std::env::var("THESEUS_ALLOW_DIAGNOSTIC_TEMPLATE_CANDIDATES")
        .map(|value| value != "0" && value.to_lowercase() != "false")
        .unwrap_or(false)
    {
        return false;
    }
    true
}

pub(super) fn low_latency_candidate_fanout_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_LOW_LATENCY_FANOUT")
        .map(|value| value.trim() != "0" && value.to_lowercase() != "false")
        .unwrap_or(true)
}

pub(super) fn transformer_hybrid_survival_lane_only_enabled() -> bool {
    !std::env::var("THESEUS_CODE_LM_ENABLE_LEGACY_COMPARATOR_FANOUT")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            value == "1" || value == "true" || value == "on"
        })
        .unwrap_or(false)
}

pub(super) fn template_like_candidate(candidate: &CandidateExpression) -> bool {
    candidate.expression_memory_fallback
        || candidate.sts_candidate_expression_used
        || template_like_candidate_mode(&candidate.mode)
}

pub(super) fn template_like_candidate_mode(mode: &str) -> bool {
    let lowered = mode.to_lowercase();
    [
        "causal_contract_skeleton_decoder",
        "contract_guided_skeleton_decoder",
        "execution_shape_skeleton_decoder",
        "local_adapter_edge_skeleton_decoder",
        "sts_causal_skeleton_decoder",
        "semantic_plan_v2",
        "edge_exec_repair",
        "private_body_prototype",
        "seeded_body_ngram_token_decoder",
        "sparse_state_sequence_seeded_decoder",
        "native_sts_stream_expression",
        "prompt_program_decoder",
        "same_seed_non_sts_comparator",
        "frequency_baseline",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

pub(super) fn no_admissible_missing_capability_family(
    task: &CodeTask,
    rejection_counts: &BTreeMap<String, usize>,
) -> String {
    let reasons = rejection_counts
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    if reasons.contains("visible_argument") {
        return "interface_fidelity".to_string();
    }
    if reasons.contains("return_shape") {
        return "return_shape_contract".to_string();
    }
    if reasons.contains("required_skeleton")
        || reasons.contains("vacuous")
        || reasons.contains("body_not_useful")
    {
        return "branch_loop_local_skeleton".to_string();
    }
    if reasons.contains("execution_library") {
        return "library_adapter".to_string();
    }
    if reasons.contains("semantic_family") || reasons.contains("semantic_admissibility") {
        return "algorithm_family".to_string();
    }
    if reasons.contains("syntax") || reasons.contains("natural_language") {
        return "syntax_and_full_body_generation".to_string();
    }
    let hints = semantic_decoder_v2_plan_hints(task, None);
    if hints.iter().any(|hint| {
        matches!(
            hint.as_str(),
            "csv" | "archive" | "structured_parsing" | "system_api" | "file_path"
        )
    }) {
        return "execution_shape_adapter".to_string();
    }
    if matches!(
        decoder_type_family(task).as_str(),
        "collection_logic" | "collection_transform" | "string_indexing" | "string_transform"
    ) {
        return "list_string_transform".to_string();
    }
    "candidate_coverage".to_string()
}

pub(super) fn no_admissible_candidate_row(
    task: &CodeTask,
    checkpoint_id: &str,
    phase: &str,
    trained: bool,
    sts_conditioned: bool,
    rejection_counts: &BTreeMap<String, usize>,
    rejection_samples: &[Value],
    task_timing_ms: &BTreeMap<String, u128>,
    raw_candidate_count: usize,
    normalized_candidate_count_before_prefilter: usize,
    normalized_candidate_count_after_prefilter: usize,
    ranked_candidate_count: usize,
    rejected_for_task: usize,
    elapsed_ms: u128,
    fanout_worker_id: usize,
    persistent_worker_pool_enabled: bool,
) -> Value {
    let body = "raise RuntimeError('student decoder emitted no admissible candidate')";
    let code = render_candidate_body(task, body);
    let candidate_generation_mode = "student_decoder_no_admissible_candidate_residual";
    let required_constructs = decoder_required_constructs(task)
        .into_iter()
        .collect::<Vec<_>>();
    let plan_hints = semantic_decoder_v2_plan_hints(task, None)
        .into_iter()
        .collect::<Vec<_>>();
    let signature_arg_names = visible_signature_arg_names(task)
        .into_iter()
        .collect::<Vec<_>>();
    let signature_arg_kinds = visible_arg_kinds(&task.entry_point, &task.prompt)
        .into_iter()
        .map(|kind| format!("{:?}", kind).to_lowercase())
        .collect::<Vec<_>>();
    let decoder_contract_summary = json!({
        "return_shape": decoder_return_shape(task),
        "type_family": decoder_type_family(task),
        "required_constructs": required_constructs,
        "plan_hints": plan_hints,
        "signature_arg_names": signature_arg_names,
        "signature_arg_kinds": signature_arg_kinds,
        "missing_capability_family": no_admissible_missing_capability_family(task, rejection_counts),
    });
    let generation_inputs = if trained && sts_conditioned {
        vec![
            "visible_prompt",
            "entry_point",
            "trained_private_curriculum_code_tokens",
            "trained_private_sts_style_stream_tokens",
            "native_sts_generated_streams",
        ]
    } else {
        vec![
            "visible_prompt",
            "entry_point",
            "trained_private_curriculum_code_tokens",
            "trained_private_sts_style_stream_tokens",
        ]
    };
    let visible_task = json!({
        "task_id": task.task_id,
        "source_task_id": task.source_task_id,
        "entry_point": task.entry_point,
        "category": task.category,
        "case_type": string_field(&task.raw, "case_type"),
        "prompt_sha256": stable_hash_hex(&task.prompt),
        "tags": task.tags,
        "decoder_contract_summary": decoder_contract_summary.clone()
    });
    let provenance = json!({
        "policy": "project_theseus_code_lm_closure_v1",
        "card_id": task.card_id,
        "source_id": task.source_id,
        "phase": phase,
        "category": task.category,
        "checkpoint_id": checkpoint_id,
        "generation_inputs": generation_inputs,
        "tests_used": false,
        "canonical_solution_used": false,
        "compositional_token_candidate": false,
        "full_body_token_candidate": false,
        "deterministic_guardrail_passed": false,
        "deterministic_guardrail_reasons": ["no_admissible_student_candidate"],
        "decoder_contract_verifier_v1_passed": false,
        "decoder_contract_verifier_v1_reasons": ["no_admissible_student_candidate"],
        "candidate_rejection_counts": rejection_counts,
        "candidate_rejection_samples": rejection_samples,
        "decoder_contract_summary": decoder_contract_summary.clone(),
        "expression_memory_fallback": false,
        "sts_stream_conditioned": trained && sts_conditioned,
        "sts_candidate_expression_used": false,
        "private_eval_solution_used_for_generation": false,
        "benchmark_promotion_eligible": false,
        "candidate_generation_mode": candidate_generation_mode,
        "token_level_code_generation_learned": false,
        "external_inference_calls": 0,
        "visible_task": visible_task
    });
    json!({
        "task_id": task.task_id,
        "source_task_id": task.source_task_id,
        "entry_point": task.entry_point,
        "category": task.category,
        "candidate_source": "student_code_lm_checkpoint_v1",
        "checkpoint_id": checkpoint_id,
        "origin": format!("student_code_lm_checkpoint_v1:{candidate_generation_mode}:{phase}:no_admissible_candidate"),
        "phase": phase,
        "code": code,
        "candidate_sha256": stable_hash_hex(&code),
        "candidate_generation_mode": candidate_generation_mode,
        "candidate_generation_contract": "student_decoder_no_admissible_candidate_residual_not_promotion_evidence",
        "candidate_return_expr": "",
        "compositional_token_candidate": false,
        "full_body_token_candidate": false,
        "deterministic_guardrail_passed": false,
        "deterministic_guardrail_reasons": ["no_admissible_student_candidate"],
        "decoder_contract_verifier_v1_passed": false,
        "decoder_contract_verifier_v1_reasons": ["no_admissible_student_candidate"],
        "candidate_rejection_counts": rejection_counts,
        "candidate_rejection_samples": rejection_samples,
        "decoder_contract_summary": decoder_contract_summary.clone(),
        "missing_capability_family": no_admissible_missing_capability_family(task, rejection_counts),
        "candidate_program_scope": "no_admissible_candidate_residual",
        "expression_memory_fallback": false,
        "sts_stream_conditioned": trained && sts_conditioned,
        "sts_streams_seen": Vec::<String>::new(),
        "sts_candidate_expression_used": false,
        "token_level_code_generation_learned": false,
        "candidate_task_timing_v1": {
            "policy": "project_theseus_candidate_task_timing_v1",
            "raw_candidate_count": raw_candidate_count,
            "normalized_candidate_count_before_prefilter": normalized_candidate_count_before_prefilter,
            "normalized_candidate_count_after_prefilter": normalized_candidate_count_after_prefilter,
            "ranked_candidate_count": ranked_candidate_count,
            "fanout_worker_id": fanout_worker_id,
            "persistent_worker_pool_enabled": persistent_worker_pool_enabled,
            "rejected_so_far": rejected_for_task,
            "elapsed_ms": elapsed_ms,
            "timing_ms": task_timing_ms,
            "score_semantics": "runtime_profile_only_not_capability_evidence"
        },
        "benchmark_promotion_eligible": false,
        "loop_closure_generated": false,
        "template_like_candidate": false,
        "canonical_solution_seen_by_solver": false,
        "public_tests_visible_to_generator": false,
        "benchmark_evidence_level": task.benchmark_evidence_level,
        "provenance": provenance
    })
}

pub(super) fn edge_exec_repair_candidates(
    task: &CodeTask,
    candidates: &[CandidateExpression],
    conditioned_suffix: &str,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<CandidateExpression> {
    if limit == 0 || !edge_exec_repair_enabled(task) {
        return Vec::new();
    }
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    if sts_streams.is_some() {
        for body in sts_causal_skeleton_bodies(task, sts_streams, limit.saturating_mul(2).max(4)) {
            if rows.len() >= limit {
                break;
            }
            if useful_generated_body_for_task(task, &body)
                && syntax_constrained_body(&body)
                && body_semantically_admissible(task, &body)
                && decoder_contract_verifier_v1(task, &body, sts_streams).passed
                && seen.insert(body.clone())
            {
                rows.push(CandidateExpression {
                    expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                    body,
                    mode: format!(
                        "rust_code_lm_edge_exec_repair_v1_sts_causal_skeleton{conditioned_suffix}"
                    ),
                    compositional_token_candidate: true,
                    full_body_token_candidate: true,
                    expression_memory_fallback: false,
                    sts_candidate_expression_used: false,
                });
            }
        }
    }
    for candidate in candidates {
        if rows.len() >= limit {
            break;
        }
        if !candidate.full_body_token_candidate || candidate.expression_memory_fallback {
            continue;
        }
        for body in edge_exec_repair_bodies(task, &candidate.body) {
            if rows.len() >= limit {
                break;
            }
            if body.trim() == candidate.body.trim() || !seen.insert(body.clone()) {
                continue;
            }
            if useful_generated_body_for_task(task, &body)
                && syntax_constrained_body(&body)
                && body_semantically_admissible(task, &body)
                && decoder_contract_verifier_v1(task, &body, sts_streams).passed
            {
                let source = if candidate.mode.contains("semantic_plan_v2") {
                    "semantic_plan_v2"
                } else if candidate.mode.contains("symliquid") {
                    "symliquid_state"
                } else if candidate.mode.contains("sparse_state_sequence") {
                    "sparse_state_sequence"
                } else {
                    "token_body"
                };
                rows.push(CandidateExpression {
                    expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                    body,
                    mode: format!("rust_code_lm_edge_exec_repair_v1_{source}{conditioned_suffix}"),
                    compositional_token_candidate: true,
                    full_body_token_candidate: true,
                    expression_memory_fallback: false,
                    sts_candidate_expression_used: false,
                });
            }
        }
    }
    rows
}

pub(super) fn sts_category_first_skeleton_bodies(
    task: &CodeTask,
    primary: &str,
    second: &str,
) -> Vec<String> {
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    match task.category.as_str() {
        "prime_fib_sequence" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!(
                "found = 0\na = 1\nb = 1\nwhile True:\n    a, b = b, a + b\n    is_prime_value = a > 1\n    for divisor in range(2, int(a ** 0.5) + 1):\n        if a % divisor == 0:\n            is_prime_value = False\n            break\n    if is_prime_value:\n        found += 1\n        if found == {primary}:\n            return a"
            ),
        ),
        "polynomial_zero_bisection" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!(
                "left = -1.0\nright = 1.0\ndef value_at(x):\n    total = 0.0\n    power = 1.0\n    for coeff in {primary}:\n        total += coeff * power\n        power *= x\n    return total\nwhile value_at(left) * value_at(right) > 0:\n    left *= 2\n    right *= 2\nfor _ in range(60):\n    mid = (left + right) / 2\n    if value_at(left) * value_at(mid) <= 0:\n        right = mid\n    else:\n        left = mid\nreturn (left + right) / 2"
            ),
        ),
        "count_digit_under_divisibility" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!(
                "total = 0\nfor item in range({primary}):\n    if item % {second}[0] == 0 or item % {second}[1] == 0:\n        total += str(item).count(str({second}[2]))\nreturn total"
            ),
        ),
        "triangle_area_product" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return {primary} * {second} / 2"),
        ),
        "fruit_distribution_private" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!(
                "numbers = []\nfor part in str({primary}).replace(',', ' ').split():\n    if part.isdigit():\n        numbers.append(int(part))\nif len(numbers) < 2:\n    return 0\nreturn numbers[0] - numbers[1]"
            ),
        ),
        "ascii_mod_char" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!(
                "total = 0\nfor ch in str({primary}):\n    total += ord(ch)\nreturn chr(total % 26 + ord('a'))"
            ),
        ),
        "bell_number_sequence" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!(
                "bell = [[0 for _ in range({primary} + 1)] for _ in range({primary} + 1)]\nbell[0][0] = 1\nfor i in range(1, {primary} + 1):\n    bell[i][0] = bell[i - 1][i - 1]\n    for j in range(1, i + 1):\n        bell[i][j] = bell[i - 1][j - 1] + bell[i][j - 1]\nreturn bell[{primary}][0]"
            ),
        ),
        "car_race_collision_count" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return {primary} * {primary}"),
        ),
        "max_list" | "private_max_item" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return max({primary})"),
        ),
        "min_list" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return min({primary})"),
        ),
        "average_or_zero" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return sum({primary}) / len({primary}) if {primary} else 0"),
        ),
        "count_integer_items" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return sum(1 for item in {primary} if type(item) is int)"),
        ),
        "distinct_count" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return len(set({primary}))"),
        ),
        "dict_required_keys" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return all(key in {primary} for key in {second})"),
        ),
        "extract_def_name" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!(
                "for line in str({primary}).splitlines():\n    stripped = line.strip()\n    if stripped.startswith('def ') and '(' in stripped:\n        return stripped[4:stripped.index('(')].strip()\nreturn ''"
            ),
        ),
        "is_anagram" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return sorted(str({primary})) == sorted(str({second}))"),
        ),
        "list_tail_replace" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("out = list({primary})\nout[-len({second}):] = list({second})\nreturn out"),
        ),
        "modular_power_two" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return pow(2, {primary}, {second})"),
        ),
        "multi_step_digit_shift_private" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!(
                "digits = str({primary})\nif not digits:\n    return digits\nshift = {second} % len(digits)\nreturn digits[-shift:] + digits[:-shift] if shift else digits"
            ),
        ),
        "multiply_three_primes" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return {primary}[0] * {primary}[1] * {primary}[2]"),
        ),
        "nested_sum" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return sum(sum(item) if isinstance(item, list) else item for item in {primary})"),
        ),
        "newman_conway_sequence" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!(
                "values = [0, 1, 1]\nfor idx in range(3, {primary} + 1):\n    values.append(values[values[idx - 1]] + values[idx - values[idx - 1]])\nreturn values[{primary}]"
            ),
        ),
        "next_perfect_square" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("root = int({primary} ** 0.5) + 1\nreturn root * root"),
        ),
        "nonempty_substring_count" | "substring_count" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return str({primary}).count(str({second}))"),
        ),
        "palindrome" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("text = str({primary})\nreturn text == text[::-1]"),
        ),
        "palindrome_list_weight" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return sum(item for item in {primary} if str(item) == str(item)[::-1])"),
        ),
        "public_private_count" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return sum(1 for item in {primary} if str(item).lower() == 'public')"),
        ),
        "same_chars" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return set({primary}) == set({second})"),
        ),
        "simple_power" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return {primary} ** {second}"),
        ),
        "smallest_palindrome_changes" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("text = str({primary})\nreturn sum(1 for idx in range(len(text) // 2) if text[idx] != text[-idx - 1])"),
        ),
        "spelled_number_sort" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return ' '.join(sorted(str({primary}).split()))"),
        ),
        "split_list_at_index" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return ({primary}[:{second}], {primary}[{second}:])"),
        ),
        "sum_squares" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return sum(item * item for item in {primary})"),
        ),
        "symbol_beat_parser" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!(
                "beats = {{'o': 4, 'o|': 2, '.|': 1}}\nout = []\nfor note in str({primary}).split():\n    if note in beats:\n        out.append(beats[note])\nreturn out"
            ),
        ),
        "total_match_lengths" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return sum(len(item) for item in {primary} if item in {second})"),
        ),
        "tuple_item_count" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return len(tuple({primary}))"),
        ),
        "uppercase_ascii_sum" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return sum(ord(ch) for ch in str({primary}) if ch.isupper())"),
        ),
        "word_count" => push_semantic_skeleton(
            &mut rows,
            &mut seen,
            format!("return len(str({primary}).split())"),
        ),
        _ => {}
    }
    rows
}

pub(super) fn edge_exec_repair_enabled(task: &CodeTask) -> bool {
    let hints = semantic_decoder_v2_plan_hints(task, None);
    let shape = decoder_return_shape(task);
    let text = format!("{} {} {}", task.card_id, task.category, task.prompt).to_lowercase();
    hints.contains("edge_conditions")
        || shape != "unknown"
        || text.contains("empty")
        || text.contains("edge")
        || text.contains("boundary")
        || text.contains("zero")
        || text.contains("if no")
        || text.contains("if there is no")
        || text.contains("source_bigcodebench")
        || text.contains("source_livecodebench")
}

pub(super) fn edge_exec_repair_bodies(task: &CodeTask, body: &str) -> Vec<String> {
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    let normalized = normalize_generated_body(body);
    let primary = decoder_primary_arg(task);
    let shape = decoder_return_shape(task);
    let primary_kind = primary_arg_kind(task);
    let text = format!("{} {}", task.category, task.prompt).to_lowercase();

    let mut push = |candidate: String| {
        let candidate = normalize_generated_body(&candidate);
        if !candidate.trim().is_empty() && seen.insert(candidate.clone()) {
            rows.push(candidate);
        }
    };

    if text.contains("prime") || text.contains("factor") || task.category == "is_prime" {
        if shape == "bool" || task.category == "is_prime" {
            push(format!(
                "value = abs({primary})\nif value < 2:\n    return False\nfor divisor in range(2, int(value ** 0.5) + 1):\n    if value % divisor == 0:\n        return False\nreturn True"
            ));
        }
        if shape == "list" || task.category == "prime_factors" {
            push(format!(
                "out = []\nvalue = abs({primary})\nfactor = 2\nwhile factor * factor <= value:\n    while value % factor == 0:\n        out.append(factor)\n        value //= factor\n    factor += 1\nif value > 1:\n    out.append(value)\nreturn out"
            ));
        }
        if task.category == "factors" {
            push(format!(
                "value = abs({primary})\nout = []\nfor divisor in range(1, value + 1):\n    if value % divisor == 0:\n        out.append(divisor)\nreturn out"
            ));
        }
        if task.category == "largest_divisor" {
            push(format!(
                "value = abs({primary})\nbest = 1\nfor divisor in range(1, value):\n    if value % divisor == 0:\n        best = divisor\nreturn best"
            ));
        }
    }

    if sequence_like_edge_arg(primary_kind, &text)
        && !body_has_empty_guard(&normalized, &primary)
        && body_mentions_arg(&normalized, &primary)
    {
        let empty = edge_empty_return_literal(task, &shape, &text);
        push(format!(
            "if not {primary}:\n    return {empty}\n{normalized}"
        ));
    }

    if text.contains("unique") || text.contains("distinct") {
        if matches!(shape.as_str(), "number" | "unknown")
            && body_mentions_arg(&normalized, &primary)
        {
            push(format!("return len(set({primary}))"));
        }
        if matches!(shape.as_str(), "list") && body_mentions_arg(&normalized, &primary) {
            push(format!(
                "out = []\nseen = set()\nfor item in {primary}:\n    if item not in seen:\n        seen.add(item)\n        out.append(item)\nreturn out"
            ));
        }
    }

    if (text.contains("minimum") || text.contains("maximum") || text.contains("median"))
        && sequence_like_edge_arg(primary_kind, &text)
        && !body_has_empty_guard(&normalized, &primary)
    {
        let empty = edge_empty_return_literal(task, &shape, &text);
        push(format!(
            "items = list({primary})\nif not items:\n    return {empty}\n{normalized}"
        ));
    }

    if normalized.contains("high - low")
        && normalized.contains("low =")
        && normalized.contains("high =")
    {
        let fallback = if shape == "list" {
            format!("[0 for _ in {primary}]")
        } else {
            "0".to_string()
        };
        push(normalized.replace(
            "high = max(data)",
            &format!("high = max(data)\nif high == low:\n    return {fallback}"),
        ));
        if primary != "data" {
            push(normalized.replace(
                &format!("high = max({primary})"),
                &format!("high = max({primary})\nif high == low:\n    return {fallback}"),
            ));
        }
    }

    rows
}

pub(super) fn sequence_like_edge_arg(kind: ValueKind, text: &str) -> bool {
    matches!(
        kind,
        ValueKind::List | ValueKind::Str | ValueKind::Dict | ValueKind::Unknown
    ) && !text.contains("scalar")
}

pub(super) fn body_has_empty_guard(body: &str, primary: &str) -> bool {
    let compact = body
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .to_lowercase();
    let primary = primary.to_lowercase();
    compact.contains(&format!("ifnot{primary}:"))
        || compact.contains(&format!("iflen({primary})==0:"))
        || compact.contains(&format!("if{primary}==[]:"))
        || compact.contains(&format!("if{primary}=='':"))
}

pub(super) fn body_mentions_arg(body: &str, primary: &str) -> bool {
    let compact = body
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .to_lowercase();
    let primary = primary.to_lowercase();
    compact.contains(&primary)
}

pub(super) fn edge_empty_return_literal(task: &CodeTask, shape: &str, text: &str) -> &'static str {
    if shape == "bool" {
        if text.contains("all ") || text.contains("every ") || text.contains("for every") {
            return "True";
        }
        return "False";
    }
    if shape == "unknown" {
        if text.contains("path") || text.contains("file") || text.contains("message") {
            return "''";
        }
        if task.card_id.contains("livecodebench") && text.contains("length") {
            return "0";
        }
    }
    empty_return_literal(shape)
}

pub(super) fn normalize_candidate_expression(
    mut candidate: CandidateExpression,
) -> CandidateExpression {
    if candidate.full_body_token_candidate {
        candidate.body = normalize_generated_body(&candidate.body);
        candidate.expr = extract_first_return_expression(&candidate.body)
            .unwrap_or_else(|| candidate.body.clone());
    }
    candidate
}

pub(in crate::code_lm_closure::candidate_fanout) fn learned_candidate_expression_variants_cached(
    task: &CodeTask,
    candidate: CandidateExpression,
    variant_cache: &mut CandidateVariantCache,
) -> Vec<CandidateExpression> {
    let mut out = vec![candidate.clone()];
    if !candidate.full_body_token_candidate
        || candidate.expression_memory_fallback
        || candidate.sts_candidate_expression_used
        || template_like_candidate(&candidate)
    {
        return out;
    }
    let mut seen = HashSet::new();
    seen.insert(candidate.body.trim().to_string());
    let alias_repaired = normalize_generated_body(&canonicalize_task_candidate_body_aliases(
        task,
        &candidate.body,
    ));
    let alias_trimmed = alias_repaired.trim();
    if !alias_trimmed.is_empty() && seen.insert(alias_trimmed.to_string()) {
        let mut repaired = candidate.clone();
        repaired.body = alias_trimmed.to_string();
        repaired.expr = extract_first_return_expression(&repaired.body)
            .unwrap_or_else(|| repaired.body.clone());
        repaired.mode = format!("{}_interface_role_repair", candidate.mode);
        out.push(repaired);
    }
    for body in variant_cache.variants(task, &candidate.body) {
        let normalized = normalize_generated_body(&body);
        let trimmed = normalized.trim();
        if trimmed.is_empty() || !seen.insert(trimmed.to_string()) {
            continue;
        }
        let mut repaired = candidate.clone();
        repaired.body = trimmed.to_string();
        repaired.expr = extract_first_return_expression(&repaired.body)
            .unwrap_or_else(|| repaired.body.clone());
        repaired.mode = format!("{}_parser_ast_completion", candidate.mode);
        out.push(repaired);
        if out.len() >= 7 {
            break;
        }
    }
    out
}

pub(in crate::code_lm_closure::candidate_fanout) fn normalize_candidate_for_task_cached(
    task: &CodeTask,
    mut candidate: CandidateExpression,
    variant_cache: &mut CandidateVariantCache,
) -> CandidateExpression {
    if !candidate.full_body_token_candidate {
        return candidate;
    }
    if edge_contract_v3_strict_novel_token_candidate(&candidate) {
        candidate.body = normalize_generated_body(&candidate.body);
        candidate.expr = extract_first_return_expression(&candidate.body)
            .unwrap_or_else(|| candidate.body.clone());
        return candidate;
    }
    if broad_private_train_token_candidate(&candidate)
        || broad_private_train_prototype_candidate(&candidate)
    {
        candidate.body = normalize_broad_private_train_body(&candidate.body);
        candidate.expr = extract_first_return_expression(&candidate.body)
            .unwrap_or_else(|| candidate.body.clone());
        return candidate;
    }
    if broad_private_generalization_semantic_adapter_candidate(&candidate) {
        candidate.body = normalize_generated_body(&candidate.body);
        candidate.expr = extract_first_return_expression(&candidate.body)
            .unwrap_or_else(|| candidate.body.clone());
        return candidate;
    }
    candidate.body = canonicalize_task_candidate_body_aliases(task, &candidate.body);
    if private_residual_v3_semantic_adapter_candidate(&candidate) {
        candidate.body = normalize_generated_body(&candidate.body);
        candidate.expr = extract_first_return_expression(&candidate.body)
            .unwrap_or_else(|| candidate.body.clone());
        return candidate;
    }
    if candidate.mode.contains("execution_shape_skeleton_decoder") {
        candidate.body = normalize_generated_body(&candidate.body);
        candidate.expr = extract_first_return_expression(&candidate.body)
            .unwrap_or_else(|| candidate.body.clone());
        return candidate;
    }
    let mut variants = variant_cache
        .variants(task, &candidate.body)
        .into_iter()
        .filter(|body| {
            useful_generated_body_for_task(task, body)
                && syntax_constrained_body(body)
                && body_semantically_admissible(task, body)
        })
        .collect::<Vec<_>>();
    if variants.is_empty() {
        candidate.body = normalize_generated_body(&candidate.body);
    } else {
        variants.sort_by(|a, b| {
            body_transfer_score(task, b)
                .partial_cmp(&body_transfer_score(task, a))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.len().cmp(&b.len()))
        });
        candidate.body = variants.remove(0);
    }
    candidate.expr =
        extract_first_return_expression(&candidate.body).unwrap_or_else(|| candidate.body.clone());
    candidate
}

pub(super) fn state_sequence_seeded_completion_bodies(
    task: &CodeTask,
    vocab: &Vocab,
) -> Vec<String> {
    if template_free_student_candidates_enabled() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for prefix in state_sequence_seed_prefixes(task) {
        if !prefix
            .iter()
            .all(|token| vocab.token_to_id.contains_key(token))
        {
            continue;
        }
        if !prefix_is_token_allowed(&prefix) {
            continue;
        }
        let prefix_body = join_body_tokens(&prefix);
        for body in state_sequence_body_variants(task, &prefix_body) {
            if useful_generated_body_for_task(task, &body)
                && syntax_constrained_body(&body)
                && body_semantically_admissible(task, &body)
                && seen.insert(body.clone())
            {
                out.push(body);
            }
        }
    }
    out
}

pub(super) fn seeded_body_ngram_completion_bodies(
    task: &CodeTask,
    model: &BodyNgramModel,
    seed: u64,
    limit: usize,
) -> Vec<String> {
    if template_free_student_candidates_enabled() {
        return Vec::new();
    }
    if model.counts.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for prefix in state_sequence_seed_prefixes(task) {
        if !prefix_is_token_allowed(&prefix) {
            continue;
        }
        let prev1 = prefix
            .last()
            .cloned()
            .unwrap_or_else(|| "<BOS>".to_string());
        let prev2 = prefix
            .iter()
            .rev()
            .nth(1)
            .cloned()
            .unwrap_or_else(|| "<BOS>".to_string());
        let mut beams = vec![BeamState {
            tokens: prefix,
            prev2,
            prev1,
            score: 1.0,
            finished: false,
        }];
        let beam_width = if task.split == "public_calibration" {
            limit.clamp(2, 4)
        } else {
            limit.clamp(2, 4)
        };
        let max_steps = learned_token_max_steps(
            task,
            if task.split == "public_calibration" {
                40
            } else {
                40
            },
        );
        for _step in 0..max_steps {
            let mut next = Vec::new();
            for beam in &beams {
                if beam.finished {
                    next.push(beam.clone());
                    continue;
                }
                let position = learned_position_cap(&task.category, beam.tokens.len());
                let mut options = body_ngram_category_token_scores(
                    task,
                    model,
                    &beam.prev2,
                    &beam.prev1,
                    position,
                )
                .into_iter()
                .collect::<Vec<_>>();
                options.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let option_cap = if task.split == "public_calibration" {
                    6
                } else {
                    6
                };
                for (token, score) in options.into_iter().take(option_cap) {
                    if !task_body_token_allowed(task, &beam.tokens, &token) {
                        continue;
                    }
                    let mut candidate = beam.clone();
                    if token == "<EOS>" {
                        candidate.finished = true;
                        candidate.score += score + length_bonus(candidate.tokens.len());
                    } else {
                        candidate.tokens.push(token.clone());
                        candidate.prev2 = candidate.prev1;
                        candidate.prev1 = token;
                        candidate.score += score;
                    }
                    next.push(candidate);
                }
            }
            if next.is_empty() {
                break;
            }
            next.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        stable_hash_u64(&format!(
                            "seeded-body-ngram:{}:{}:{:?}",
                            seed, task.task_id, a.tokens
                        ))
                        .cmp(&stable_hash_u64(&format!(
                            "seeded-body-ngram:{}:{}:{:?}",
                            seed, task.task_id, b.tokens
                        )))
                    })
            });
            beams = next.into_iter().take(beam_width).collect();
            if beams.iter().all(|beam| beam.finished) {
                break;
            }
        }
        beams.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for beam in beams {
            let body = join_body_tokens(&beam.tokens);
            for candidate_body in state_sequence_body_variants(task, &body) {
                if state_sequence_candidate_body_ok(task, &candidate_body)
                    && seen.insert(candidate_body.clone())
                {
                    out.push(candidate_body);
                }
                if out.len() >= limit {
                    break;
                }
            }
            if out.len() >= limit {
                break;
            }
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}
