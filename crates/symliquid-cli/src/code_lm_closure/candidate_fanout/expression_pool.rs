use super::*;

pub(super) fn candidate_expressions(
    task: &CodeTask,
    expression_bank: &[ExpressionBankItem],
    body_prototypes: &BodyPrototypeModel,
    body_ngram: &BodyNgramModel,
    state_sequence_decoder: &StateSequenceDecoder,
    symliquid_state_decoder: &SymLiquidStateDecoder,
    readout: &LinearReadout,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
    precomputed_beams: Option<&Vec<String>>,
) -> Vec<CandidateExpression> {
    candidate_expressions_with_timing(
        task,
        expression_bank,
        body_prototypes,
        body_ngram,
        state_sequence_decoder,
        symliquid_state_decoder,
        readout,
        vocab,
        seed,
        limit,
        sts_streams,
        precomputed_beams,
        None,
        None,
        None,
        None,
    )
    .0
}

fn full_body_candidate_expression(body: String, mode: String) -> CandidateExpression {
    CandidateExpression {
        expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
        body,
        mode,
        compositional_token_candidate: true,
        full_body_token_candidate: true,
        expression_memory_fallback: false,
        sts_candidate_expression_used: false,
    }
}

fn push_contract_transduction_rows(
    rows: &mut Vec<CandidateExpression>,
    bodies: &[String],
    conditioned_suffix: &str,
) -> usize {
    let before = rows.len();
    for body in bodies {
        rows.push(CandidateExpression {
            expr: extract_first_return_expression(body).unwrap_or_else(|| body.clone()),
            body: body.clone(),
            mode: format!("rust_code_lm_contract_transduced_token_decoder{conditioned_suffix}"),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: false,
        });
    }
    rows.len().saturating_sub(before)
}

fn precomputed_decoder_family_rows(
    precomputed_rows: Option<&Vec<String>>,
    budget: usize,
) -> Option<Vec<String>> {
    precomputed_rows.map(|rows| rows.iter().take(budget).cloned().collect())
}

fn low_latency_full_body_candidate_accepted(
    task: &CodeTask,
    candidate: &CandidateExpression,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> bool {
    let verification_streams = if candidate.mode.contains("visible_contract_fallback") {
        None
    } else {
        sts_streams
    };
    let promotion_capable_for_public =
        task.split != "public_calibration" || learned_token_decoder_candidate(candidate);
    candidate.full_body_token_candidate
        && !candidate.expression_memory_fallback
        && !candidate.sts_candidate_expression_used
        && !template_like_candidate(candidate)
        && promotion_capable_for_public
        && decoder_contract_verifier_v1(task, &candidate.body, verification_streams).passed
}

fn low_latency_accepted_candidate_count(
    task: &CodeTask,
    rows: &[CandidateExpression],
    sts_streams: Option<&BTreeMap<String, String>>,
) -> usize {
    rows.iter()
        .filter(|candidate| low_latency_full_body_candidate_accepted(task, candidate, sts_streams))
        .count()
}

fn low_latency_expensive_rescue_enabled(task: &CodeTask) -> bool {
    if task.split == "public_calibration" {
        return true;
    }
    std::env::var("THESEUS_CODE_LM_LOW_LATENCY_EXPENSIVE_RESCUE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(false)
}

fn public_metadata_skip_full_sts_bridge_when_family_ready(task: &CodeTask) -> bool {
    if task.split != "public_calibration" {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_PUBLIC_METADATA_SKIP_FULL_STS_BRIDGE_FALLBACK")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn skip_full_sts_bridge_when_family_ready(task: &CodeTask) -> bool {
    if public_metadata_skip_full_sts_bridge_when_family_ready(task) {
        return true;
    }
    if task.split == "public_calibration" {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_PRIVATE_SKIP_FULL_STS_BRIDGE_FALLBACK_WHEN_FAMILY_READY")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn fast_sts_off_from_precomputed_family_enabled(task: &CodeTask) -> bool {
    if task.split == "public_calibration" {
        let fast_public_metadata =
            std::env::var("THESEUS_CODE_LM_PUBLIC_METADATA_FAST_STS_OFF_COMPARATOR")
                .map(|value| {
                    let value = value.trim().to_ascii_lowercase();
                    !(value == "0" || value == "false" || value == "off")
                })
                .unwrap_or(true);
        let full_public_metadata =
            std::env::var("THESEUS_CODE_LM_PUBLIC_METADATA_FULL_STS_OFF_COMPARATOR")
                .map(|value| {
                    let value = value.trim().to_ascii_lowercase();
                    !(value == "0" || value == "false" || value == "off")
                })
                .unwrap_or(false);
        return fast_public_metadata && !full_public_metadata;
    }
    std::env::var("THESEUS_CODE_LM_FAST_STS_OFF_FROM_PRECOMPUTED_FAMILY")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn fast_sts_off_contract_rows_from_precomputed_family(
    task: &CodeTask,
    body_prototypes: &BodyPrototypeModel,
    precomputed_state_sequence_no_sts: Option<&Vec<String>>,
    precomputed_symliquid_state_no_sts: Option<&Vec<String>>,
    budget: usize,
) -> Option<Vec<String>> {
    if !fast_sts_off_from_precomputed_family_enabled(task) {
        return None;
    }
    let budget = budget.clamp(1, 16);
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    let push_row = |body: String, rows: &mut Vec<String>, seen: &mut HashSet<String>| {
        let normalized = normalize_generated_body(&body);
        let canonical = canonicalize_task_candidate_body_aliases(task, &normalized);
        let trimmed = canonical.trim().to_string();
        if !trimmed.is_empty() && seen.insert(trimmed.clone()) {
            rows.push(trimmed);
        }
    };

    for body in prompt_contract_bodies_with_visible_fallback(task, budget.min(4), None).0 {
        push_row(body, &mut rows, &mut seen);
    }
    for body in learned_contract_transduction_bodies(task, body_prototypes, budget.min(4), None) {
        push_row(body, &mut rows, &mut seen);
    }
    for body in precomputed_decoder_family_rows(precomputed_state_sequence_no_sts, budget)
        .unwrap_or_default()
    {
        push_row(body, &mut rows, &mut seen);
    }
    for body in precomputed_decoder_family_rows(precomputed_symliquid_state_no_sts, budget)
        .unwrap_or_default()
    {
        push_row(body, &mut rows, &mut seen);
    }
    for body in contract_guided_skeleton_bodies(task, budget.min(4), None) {
        push_row(body, &mut rows, &mut seen);
    }
    if rows.len() >= budget.min(2) {
        rows.truncate(budget);
        Some(rows)
    } else {
        None
    }
}

fn finalize_transfer_score_prefilter_budget(candidate_limit: usize, low_latency: bool) -> usize {
    let default_multiplier = if low_latency { 6 } else { 24 };
    let multiplier = std::env::var("THESEUS_CODE_LM_FINALIZE_PREFILTER_MULTIPLIER")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default_multiplier)
        .clamp(3, 64);
    let default_cap = if low_latency { 64 } else { 192 };
    let cap = std::env::var("THESEUS_CODE_LM_FINALIZE_PREFILTER_CAP")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default_cap)
        .clamp(16, 512);
    candidate_limit
        .max(1)
        .saturating_mul(multiplier)
        .clamp(16, cap)
}

fn private_low_latency_multi_candidate_fanout_enabled(task: &CodeTask, limit: usize) -> bool {
    let max_private_limit = if broad_transfer_residual_policy_enabled()
        && broad_transfer_residual_policy(task).active()
    {
        8
    } else {
        4
    };
    if task.split == "public_calibration" || limit > max_private_limit {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_PRIVATE_LOW_LATENCY_MULTI_CANDIDATE_FANOUT")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn public_metadata_low_latency_multi_candidate_fanout_enabled(
    task: &CodeTask,
    limit: usize,
) -> bool {
    if task.split != "public_calibration" || limit > public_metadata_low_latency_candidate_limit() {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_PUBLIC_METADATA_LOW_LATENCY_MULTI_CANDIDATE_FANOUT")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn public_metadata_low_latency_candidate_limit() -> usize {
    std::env::var("THESEUS_CODE_LM_PUBLIC_METADATA_LOW_LATENCY_CANDIDATE_LIMIT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|value| value.clamp(1, 16))
        .unwrap_or(8)
}

fn public_metadata_single_accepted_lazy_exit_enabled(task: &CodeTask) -> bool {
    if task.split != "public_calibration" {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_PUBLIC_METADATA_SINGLE_ACCEPTED_LAZY_EXIT")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn low_latency_acceptance_target(task: &CodeTask, limit: usize) -> usize {
    if public_metadata_single_accepted_lazy_exit_enabled(task) {
        1
    } else if private_residual_single_accepted_lazy_exit_enabled(task) {
        1
    } else {
        limit.max(1)
    }
}

fn private_residual_single_accepted_lazy_exit_enabled(task: &CodeTask) -> bool {
    if task.split == "public_calibration" {
        return false;
    }
    if !broad_transfer_residual_policy_enabled() || !broad_transfer_residual_policy(task).active() {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_PRIVATE_RESIDUAL_SINGLE_ACCEPTED_LAZY_EXIT")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn low_latency_same_seed_comparator_target() -> usize {
    std::env::var("THESEUS_CODE_LM_LOW_LATENCY_SAME_SEED_COMPARATOR_TARGET")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|value| value.clamp(1, 8))
        .unwrap_or(1)
}

fn low_latency_comparator_uses_prototype_transduction(task: &CodeTask) -> bool {
    if task.split == "public_calibration" {
        return true;
    }
    std::env::var("THESEUS_CODE_LM_LOW_LATENCY_COMPARATOR_TRANSDUCTION")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(false)
}

fn prompt_contract_bodies_with_visible_fallback(
    task: &CodeTask,
    budget: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> (Vec<String>, &'static str) {
    let rows = contract_guided_skeleton_bodies(task, budget, sts_streams);
    let suffix = if sts_streams.is_some() {
        "_sts_conditioned"
    } else {
        ""
    };
    if !rows.is_empty() {
        let verified_rows = rows
            .iter()
            .filter(|body| {
                syntax_constrained_body(body)
                    && decoder_contract_verifier_v1(task, body, sts_streams).passed
            })
            .cloned()
            .collect::<Vec<_>>();
        if !verified_rows.is_empty() {
            return (verified_rows, suffix);
        }
    }
    if sts_streams.is_some() {
        let fallback_rows = contract_guided_skeleton_bodies(task, budget, None);
        let verified_fallback_rows = fallback_rows
            .iter()
            .filter(|body| {
                syntax_constrained_body(body)
                    && decoder_contract_verifier_v1(task, body, None).passed
            })
            .cloned()
            .collect::<Vec<_>>();
        if !verified_fallback_rows.is_empty() {
            return (verified_fallback_rows, "_visible_contract_fallback");
        }
    }
    (rows, suffix)
}

fn ensure_low_latency_same_seed_non_sts_comparator(
    task: &CodeTask,
    body_prototypes: &BodyPrototypeModel,
    rows: &mut Vec<CandidateExpression>,
    expression_timing_ms: &mut BTreeMap<String, u128>,
    already_added: &mut bool,
    enabled: bool,
) {
    if !enabled || *already_added {
        return;
    }
    let started = Instant::now();
    let before = rows.len();
    let budget = 4usize;
    let target = low_latency_same_seed_comparator_target();
    let mut seen = HashSet::new();
    let prompt_started = Instant::now();
    let prompt_contract_rows = prompt_contract_bodies_with_visible_fallback(task, budget, None).0;
    append_valid_same_seed_comparator_rows(
        task,
        rows,
        &mut seen,
        prompt_contract_rows,
        before,
        target,
    );
    expression_timing_ms.insert(
        "same_seed_non_sts_comparator_prompt_contract_ms".to_string(),
        prompt_started.elapsed().as_millis(),
    );
    let mut transduction_fallback_used = false;
    if rows.len().saturating_sub(before) < target
        && low_latency_comparator_uses_prototype_transduction(task)
    {
        transduction_fallback_used = true;
        let transduction_started = Instant::now();
        let transduced_rows =
            learned_contract_transduction_bodies(task, body_prototypes, budget, None);
        append_valid_same_seed_comparator_rows(
            task,
            rows,
            &mut seen,
            transduced_rows,
            before,
            target,
        );
        expression_timing_ms.insert(
            "same_seed_non_sts_comparator_transduction_fallback_ms".to_string(),
            transduction_started.elapsed().as_millis(),
        );
    } else {
        expression_timing_ms.insert(
            "same_seed_non_sts_comparator_transduction_fallback_ms".to_string(),
            0,
        );
    }
    expression_timing_ms.insert(
        "same_seed_non_sts_comparator_transduction_fallback_used".to_string(),
        if transduction_fallback_used { 1 } else { 0 },
    );
    expression_timing_ms.insert(
        "same_seed_non_sts_comparator_low_latency_ms".to_string(),
        started.elapsed().as_millis(),
    );
    expression_timing_ms.insert(
        "same_seed_non_sts_comparator_low_latency_count".to_string(),
        rows.len().saturating_sub(before) as u128,
    );
    *already_added = true;
}

fn append_valid_same_seed_comparator_rows(
    task: &CodeTask,
    rows: &mut Vec<CandidateExpression>,
    seen: &mut HashSet<String>,
    bodies: Vec<String>,
    before: usize,
    target: usize,
) {
    for body in bodies {
        if rows.len().saturating_sub(before) >= target {
            break;
        }
        if !seen.insert(body.clone()) {
            continue;
        }
        if !useful_generated_body_for_task(task, &body)
            || !syntax_constrained_body(&body)
            || !body_semantically_admissible(task, &body)
            || !decoder_contract_verifier_v1(task, &body, None).passed
        {
            continue;
        }
        rows.push(full_body_candidate_expression(
            body,
            "rust_code_lm_contract_guided_token_decoder_same_seed_non_sts_comparator".to_string(),
        ));
    }
}

fn append_broad_transfer_residual_retry_candidates_with_timing(
    task: &CodeTask,
    rows: &mut Vec<CandidateExpression>,
    limit: usize,
    expression_timing_ms: &mut BTreeMap<String, u128>,
    timing_key: &str,
) {
    let started = Instant::now();
    let added = append_broad_transfer_residual_retry_candidates(task, rows, limit);
    expression_timing_ms.insert(format!("{timing_key}_added"), added as u128);
    expression_timing_ms.insert(format!("{timing_key}_ms"), started.elapsed().as_millis());
}

fn append_eligible_receiver_inventory_router_candidates_with_timing(
    task: &CodeTask,
    rows: &mut Vec<CandidateExpression>,
    limit: usize,
    expression_timing_ms: &mut BTreeMap<String, u128>,
    timing_key: &str,
) {
    let started = Instant::now();
    let added = append_eligible_receiver_inventory_router_candidates(task, rows, limit);
    expression_timing_ms.insert(format!("{timing_key}_added"), added as u128);
    expression_timing_ms.insert(format!("{timing_key}_ms"), started.elapsed().as_millis());
}

fn append_private_to_public_receiver_inventory_bridge_candidates_with_timing(
    task: &CodeTask,
    rows: &mut Vec<CandidateExpression>,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
    expression_timing_ms: &mut BTreeMap<String, u128>,
    timing_key: &str,
) {
    let started = Instant::now();
    let added = append_private_to_public_receiver_inventory_bridge_candidates(task, rows, limit);
    expression_timing_ms.insert(format!("{timing_key}_added"), added as u128);
    expression_timing_ms.insert(format!("{timing_key}_ms"), started.elapsed().as_millis());
    if let Some(streams) = sts_streams {
        let sts_started = Instant::now();
        let added = append_sts_conditioned_private_to_public_receiver_inventory_bridge_candidates(
            task, rows, limit, streams,
        );
        expression_timing_ms.insert(format!("{timing_key}_sts_conditioned_added"), added as u128);
        expression_timing_ms.insert(
            format!("{timing_key}_sts_conditioned_ms"),
            sts_started.elapsed().as_millis(),
        );
    }
}

fn append_low_latency_local_adapter_edge_skeletons_with_timing(
    task: &CodeTask,
    rows: &mut Vec<CandidateExpression>,
    budget: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
    expression_timing_ms: &mut BTreeMap<String, u128>,
    timing_key: &str,
) {
    let started = Instant::now();
    let before = rows.len();
    let conditioned_suffix = if sts_streams.is_some() {
        "_sts_conditioned"
    } else {
        ""
    };
    for body in local_adapter_edge_skeleton_bodies(task, budget, sts_streams) {
        rows.push(full_body_candidate_expression(
            body,
            format!("rust_code_lm_local_adapter_edge_skeleton_decoder{conditioned_suffix}"),
        ));
    }
    expression_timing_ms.insert(
        format!("{timing_key}_added"),
        rows.len().saturating_sub(before) as u128,
    );
    expression_timing_ms.insert(format!("{timing_key}_ms"), started.elapsed().as_millis());
}

fn append_low_latency_learned_category_ngram_contract_candidates_with_timing(
    task: &CodeTask,
    rows: &mut Vec<CandidateExpression>,
    body_ngram: &BodyNgramModel,
    seed: u64,
    budget: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
    expression_timing_ms: &mut BTreeMap<String, u128>,
    timing_key: &str,
) {
    if task.split == "public_calibration" || !task.category.starts_with("edge_v3_") {
        return;
    }
    let started = Instant::now();
    let before = rows.len();
    let conditioned_suffix = if sts_streams.is_some() {
        "_sts_conditioned"
    } else {
        ""
    };
    let budget = budget.clamp(2, 8);
    for body in body_ngram_bodies(task, body_ngram, seed, budget, sts_streams) {
        rows.push(full_body_candidate_expression(
            body,
            format!(
                "rust_code_lm_contract_guided_token_decoder_category_context{conditioned_suffix}"
            ),
        ));
    }
    expression_timing_ms.insert(
        format!("{timing_key}_added"),
        rows.len().saturating_sub(before) as u128,
    );
    expression_timing_ms.insert(format!("{timing_key}_ms"), started.elapsed().as_millis());
}

pub(super) fn candidate_expressions_with_timing(
    task: &CodeTask,
    expression_bank: &[ExpressionBankItem],
    body_prototypes: &BodyPrototypeModel,
    body_ngram: &BodyNgramModel,
    state_sequence_decoder: &StateSequenceDecoder,
    symliquid_state_decoder: &SymLiquidStateDecoder,
    readout: &LinearReadout,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
    precomputed_beams: Option<&Vec<String>>,
    precomputed_state_sequence: Option<&Vec<String>>,
    precomputed_symliquid_state: Option<&Vec<String>>,
    precomputed_state_sequence_no_sts: Option<&Vec<String>>,
    precomputed_symliquid_state_no_sts: Option<&Vec<String>>,
) -> (Vec<CandidateExpression>, BTreeMap<String, u128>) {
    let mut rows = Vec::new();
    let mut expression_timing_ms: BTreeMap<String, u128> = BTreeMap::new();
    expression_timing_ms.insert(
        "candidate_task_context_worker_id".to_string(),
        current_candidate_fanout_worker_id() as u128,
    );
    expression_timing_ms.insert(
        "nested_branch_parallelism_suppressed".to_string(),
        nested_branch_parallelism_suppressed_for_current_task() as u128,
    );
    let mut expression_stage_started = Instant::now();
    let template_free = template_free_student_candidates_enabled();
    let conditioned_suffix = if sts_streams.is_some() {
        "_sts_conditioned"
    } else {
        ""
    };
    let public_calibration = task.split == "public_calibration";
    let low_latency_fanout = low_latency_candidate_fanout_enabled()
        && (limit <= 1
            || private_low_latency_multi_candidate_fanout_enabled(task, limit)
            || public_metadata_low_latency_multi_candidate_fanout_enabled(task, limit));
    let low_latency_accept_count = low_latency_acceptance_target(task, limit);
    let mut low_latency_non_sts_comparator_added = false;
    let mut contract_transduction_added = false;
    let mut low_latency_contract_transduction_cache: Option<Vec<String>> = None;
    let token_inventory_budget = if template_free {
        if low_latency_fanout {
            5
        } else {
            limit.saturating_mul(2).clamp(8, 24)
        }
    } else {
        limit.max(1)
    };
    let beam_budget = if template_free {
        if low_latency_fanout {
            2
        } else {
            limit.clamp(4, 8)
        }
    } else if public_calibration {
        (limit / 3 + 1).clamp(2, 4)
    } else {
        (limit / 4 + 1).clamp(1, 2)
    };
    let state_budget = if template_free {
        if low_latency_fanout {
            2
        } else {
            limit.clamp(4, 8)
        }
    } else if public_calibration {
        (limit / 3 + 1).clamp(2, 4)
    } else {
        (limit / 4 + 1).clamp(1, 2)
    };
    let symliquid_budget = if template_free {
        if low_latency_fanout {
            2
        } else {
            limit.clamp(4, 8)
        }
    } else if public_calibration {
        (limit / 3 + 1).clamp(2, 4)
    } else {
        (limit / 4 + 1).clamp(1, 2)
    };
    let semantic_plan_budget = if public_calibration {
        (limit / 3 + 1).clamp(2, 4)
    } else {
        (limit / 4 + 1).clamp(1, 2)
    };
    let execution_shape_budget = if public_calibration {
        (limit / 3 + 2).clamp(2, 5)
    } else {
        (limit / 5 + 1).clamp(1, 2)
    };
    let contract_guided_budget = if public_calibration {
        (limit / 3 + 2).clamp(2, 5)
    } else {
        (limit / 5 + 1).clamp(1, 2)
    };
    let prompt_contract_budget = if public_calibration {
        limit.clamp(4, 12)
    } else {
        limit.clamp(2, 8)
    };
    let local_adapter_budget = if public_calibration {
        (limit / 3 + 2).clamp(2, 5)
    } else {
        (limit / 5 + 1).clamp(1, 3)
    };
    let causal_contract_budget = if public_calibration {
        (limit / 3 + 3).clamp(3, 6)
    } else {
        (limit / 4 + 1).clamp(1, 3)
    };
    let contract_token_budget = if template_free {
        if low_latency_fanout {
            token_inventory_budget.clamp(4, 8)
        } else {
            token_inventory_budget.clamp(8, 24)
        }
    } else if public_calibration {
        (limit / 3 + 1).clamp(2, 4)
    } else {
        (limit / 4 + 1).clamp(1, 3)
    };
    if template_free && low_latency_fanout {
        let prompt_contract_started = Instant::now();
        let (prompt_contract_bodies, prompt_contract_mode_suffix) =
            prompt_contract_bodies_with_visible_fallback(task, prompt_contract_budget, sts_streams);
        for body in prompt_contract_bodies {
            rows.push(full_body_candidate_expression(
                body,
                format!(
                    "rust_code_lm_contract_guided_prompt_program_decoder{prompt_contract_mode_suffix}"
                ),
            ));
        }
        append_low_latency_learned_category_ngram_contract_candidates_with_timing(
            task,
            &mut rows,
            body_ngram,
            seed ^ 0xE6E3_7A11,
            contract_token_budget,
            sts_streams,
            &mut expression_timing_ms,
            "low_latency_learned_category_ngram_contract",
        );
        expression_timing_ms.insert(
            "prompt_contract_program_decoder_fast_path".to_string(),
            prompt_contract_started.elapsed().as_millis(),
        );
        ensure_low_latency_same_seed_non_sts_comparator(
            task,
            body_prototypes,
            &mut rows,
            &mut expression_timing_ms,
            &mut low_latency_non_sts_comparator_added,
            sts_streams.is_some(),
        );
        let sts_off_inventory_started = Instant::now();
        let mut sts_off_inventory_added = 0usize;
        if sts_streams.is_some() {
            let budget = (contract_token_budget / 2).clamp(2, 8);
            if let Some(sts_off_rows) = fast_sts_off_contract_rows_from_precomputed_family(
                task,
                body_prototypes,
                precomputed_state_sequence_no_sts,
                precomputed_symliquid_state_no_sts,
                budget,
            ) {
                for body in sts_off_rows {
                    rows.push(full_body_candidate_expression(
                        body,
                        "rust_code_lm_contract_guided_token_decoder".to_string(),
                    ));
                    sts_off_inventory_added += 1;
                }
            }
        }
        expression_timing_ms.insert(
            "sts_off_contract_guided_token_decoder_fast_inventory_added".to_string(),
            sts_off_inventory_added as u128,
        );
        expression_timing_ms.insert(
            "sts_off_contract_guided_token_decoder_fast_inventory_ms".to_string(),
            sts_off_inventory_started.elapsed().as_millis(),
        );
        append_eligible_receiver_inventory_router_candidates_with_timing(
            task,
            &mut rows,
            limit,
            &mut expression_timing_ms,
            "eligible_receiver_inventory_low_latency_prompt_contract",
        );
        append_private_to_public_receiver_inventory_bridge_candidates_with_timing(
            task,
            &mut rows,
            limit,
            sts_streams,
            &mut expression_timing_ms,
            "private_to_public_receiver_inventory_bridge_low_latency_prompt_contract",
        );
        append_broad_transfer_residual_retry_candidates_with_timing(
            task,
            &mut rows,
            limit,
            &mut expression_timing_ms,
            "broad_transfer_residual_low_latency_prompt_contract",
        );
        let prompt_contract_accepted =
            low_latency_accepted_candidate_count(task, &rows, sts_streams);
        expression_timing_ms.insert(
            "low_latency_prompt_contract_pre_transduction_accepted".to_string(),
            prompt_contract_accepted as u128,
        );
        if !public_calibration && prompt_contract_accepted >= low_latency_accept_count {
            expression_timing_ms.insert(
                "low_latency_prompt_contract_transduction_skipped_sufficient_candidates"
                    .to_string(),
                1,
            );
            expression_timing_ms.insert(
                "low_latency_prompt_contract_transduction_added".to_string(),
                0,
            );
            expression_timing_ms
                .insert("low_latency_prompt_contract_transduction_ms".to_string(), 0);
            let mut finalize_started = Instant::now();
            let finalized = finalize_candidate_expression_rows(task, rows, sts_streams, limit);
            record_candidate_timing(
                &mut expression_timing_ms,
                "low_latency_prompt_contract_fast_finalize",
                &mut finalize_started,
            );
            return (finalized, expression_timing_ms);
        }
        let prompt_contract_transduction_started = Instant::now();
        let transduced_rows = learned_contract_transduction_bodies(
            task,
            body_prototypes,
            contract_token_budget,
            sts_streams,
        );
        let prompt_contract_transduction_added =
            push_contract_transduction_rows(&mut rows, &transduced_rows, conditioned_suffix);
        contract_transduction_added = prompt_contract_transduction_added > 0;
        low_latency_contract_transduction_cache = Some(transduced_rows);
        expression_timing_ms.insert(
            "low_latency_prompt_contract_transduction_added".to_string(),
            prompt_contract_transduction_added as u128,
        );
        expression_timing_ms.insert(
            "low_latency_prompt_contract_transduction_ms".to_string(),
            prompt_contract_transduction_started.elapsed().as_millis(),
        );
        let mut finalize_started = Instant::now();
        let finalized = finalize_candidate_expression_rows(task, rows, sts_streams, limit);
        record_candidate_timing(
            &mut expression_timing_ms,
            "low_latency_prompt_contract_fast_finalize",
            &mut finalize_started,
        );
        if low_latency_accepted_candidate_count(task, &finalized, sts_streams)
            >= low_latency_accept_count
        {
            return (finalized, expression_timing_ms);
        }
        rows = finalized;
    }
    if template_free && low_latency_fanout {
        if contract_transduction_added {
            expression_timing_ms.insert(
                "low_latency_pre_contract_transduction_reused".to_string(),
                1,
            );
            expression_timing_ms
                .insert("low_latency_pre_contract_transduction_added".to_string(), 0);
            expression_timing_ms.insert("low_latency_pre_contract_transduction".to_string(), 0);
        } else {
            let transduction_started = Instant::now();
            let transduced_rows = low_latency_contract_transduction_cache
                .take()
                .unwrap_or_else(|| {
                    learned_contract_transduction_bodies(
                        task,
                        body_prototypes,
                        contract_token_budget,
                        sts_streams,
                    )
                });
            let transduction_added =
                push_contract_transduction_rows(&mut rows, &transduced_rows, conditioned_suffix);
            contract_transduction_added = transduction_added > 0;
            expression_timing_ms.insert(
                "low_latency_pre_contract_transduction_reused".to_string(),
                0,
            );
            expression_timing_ms.insert(
                "low_latency_pre_contract_transduction_added".to_string(),
                transduction_added as u128,
            );
            expression_timing_ms.insert(
                "low_latency_pre_contract_transduction".to_string(),
                transduction_started.elapsed().as_millis(),
            );
        }
        ensure_low_latency_same_seed_non_sts_comparator(
            task,
            body_prototypes,
            &mut rows,
            &mut expression_timing_ms,
            &mut low_latency_non_sts_comparator_added,
            sts_streams.is_some(),
        );
        append_eligible_receiver_inventory_router_candidates_with_timing(
            task,
            &mut rows,
            limit,
            &mut expression_timing_ms,
            "eligible_receiver_inventory_low_latency_transduction",
        );
        append_private_to_public_receiver_inventory_bridge_candidates_with_timing(
            task,
            &mut rows,
            limit,
            sts_streams,
            &mut expression_timing_ms,
            "private_to_public_receiver_inventory_bridge_low_latency_transduction",
        );
        append_broad_transfer_residual_retry_candidates_with_timing(
            task,
            &mut rows,
            limit,
            &mut expression_timing_ms,
            "broad_transfer_residual_low_latency_transduction",
        );
        let mut finalize_started = Instant::now();
        let finalized = finalize_candidate_expression_rows(task, rows, sts_streams, limit);
        record_candidate_timing(
            &mut expression_timing_ms,
            "low_latency_pre_contract_transduction_finalize",
            &mut finalize_started,
        );
        if low_latency_accepted_candidate_count(task, &finalized, sts_streams)
            >= low_latency_accept_count
        {
            return (finalized, expression_timing_ms);
        }
        rows = finalized;
    }
    if template_free && low_latency_fanout && !low_latency_expensive_rescue_enabled(task) {
        append_low_latency_local_adapter_edge_skeletons_with_timing(
            task,
            &mut rows,
            local_adapter_budget.max(2).min(4),
            sts_streams,
            &mut expression_timing_ms,
            "low_latency_local_adapter_edge_skeleton_rescue",
        );
        let mut finalize_started = Instant::now();
        let finalized = finalize_candidate_expression_rows(task, rows, sts_streams, limit);
        record_candidate_timing(
            &mut expression_timing_ms,
            "low_latency_local_adapter_edge_skeleton_rescue_finalize",
            &mut finalize_started,
        );
        expression_timing_ms.insert("low_latency_expensive_rescue_skipped".to_string(), 1);
        return (finalized, expression_timing_ms);
    }
    if template_free && sts_streams.is_some() && !low_latency_fanout {
        let streams = sts_streams.expect("checked above");
        let parallel_ngram_budget = limit.clamp(4, 10);
        let parallel_started = Instant::now();
        let (
            contract_rows,
            contract_ms,
            transduced_rows,
            transduced_ms,
            sts_off_contract_rows,
            sts_off_contract_ms,
            sts_off_contract_fast_precomputed_used,
            symliquid_rows,
            symliquid_ms,
            state_rows,
            state_ms,
            beam_rows,
            beam_ms,
            ngram_rows,
            ngram_ms,
            greedy_rows,
            greedy_ms,
            prompt_contract_rows,
            prompt_contract_mode_suffix,
            prompt_contract_ms,
        ) = std::thread::scope(|scope| {
            let contract = scope.spawn(|| {
                let started = Instant::now();
                let rows = contract_guided_token_decoder_bodies(
                    task,
                    body_ngram,
                    state_sequence_decoder,
                    symliquid_state_decoder,
                    readout,
                    vocab,
                    seed ^ 0xC0A7_5EED,
                    contract_token_budget,
                    Some(streams),
                    precomputed_beams,
                    precomputed_state_sequence,
                    precomputed_symliquid_state,
                );
                (rows, started.elapsed().as_millis())
            });
            let transduced = scope.spawn(|| {
                let started = Instant::now();
                let rows = learned_contract_transduction_bodies(
                    task,
                    body_prototypes,
                    contract_token_budget,
                    Some(streams),
                );
                (rows, started.elapsed().as_millis())
            });
            let sts_off_contract = scope.spawn(|| {
                let started = Instant::now();
                let budget = (contract_token_budget / 2).clamp(2, 8);
                if let Some(rows) = fast_sts_off_contract_rows_from_precomputed_family(
                    task,
                    body_prototypes,
                    precomputed_state_sequence_no_sts,
                    precomputed_symliquid_state_no_sts,
                    budget,
                ) {
                    (rows, started.elapsed().as_millis(), true)
                } else {
                    let rows = contract_guided_token_decoder_bodies(
                        task,
                        body_ngram,
                        state_sequence_decoder,
                        symliquid_state_decoder,
                        readout,
                        vocab,
                        seed ^ 0xC0A7_0FF5,
                        budget,
                        None,
                        None,
                        precomputed_state_sequence_no_sts,
                        precomputed_symliquid_state_no_sts,
                    );
                    (rows, started.elapsed().as_millis(), false)
                }
            });
            let symliquid = scope.spawn(|| {
                let started = Instant::now();
                let rows =
                    precomputed_decoder_family_rows(precomputed_symliquid_state, symliquid_budget)
                        .unwrap_or_else(|| {
                            symliquid_state_bodies(
                                task,
                                symliquid_state_decoder,
                                body_ngram,
                                vocab,
                                seed,
                                symliquid_budget,
                                Some(streams),
                            )
                        });
                (rows, started.elapsed().as_millis())
            });
            let state = scope.spawn(|| {
                let started = Instant::now();
                let rows =
                    precomputed_decoder_family_rows(precomputed_state_sequence, state_budget)
                        .unwrap_or_else(|| {
                            state_sequence_bodies(
                                task,
                                state_sequence_decoder,
                                body_ngram,
                                vocab,
                                seed,
                                state_budget,
                                Some(streams),
                            )
                        });
                (rows, started.elapsed().as_millis())
            });
            let beam = scope.spawn(|| {
                let started = Instant::now();
                let rows = precomputed_beams.cloned().unwrap_or_else(|| {
                    beam_bodies(task, readout, vocab, seed, beam_budget, Some(streams))
                });
                (rows, started.elapsed().as_millis())
            });
            let ngram = scope.spawn(|| {
                let started = Instant::now();
                let rows =
                    body_ngram_bodies(task, body_ngram, seed, parallel_ngram_budget, Some(streams));
                (rows, started.elapsed().as_millis())
            });
            let greedy = scope.spawn(|| {
                let started = Instant::now();
                let rows = greedy_or_precomputed_beam_body(
                    task,
                    readout,
                    vocab,
                    Some(streams),
                    precomputed_beams,
                )
                .into_iter()
                .collect::<Vec<_>>();
                (rows, started.elapsed().as_millis())
            });
            let prompt_contract = scope.spawn(|| {
                let started = Instant::now();
                let (rows, mode_suffix) = prompt_contract_bodies_with_visible_fallback(
                    task,
                    prompt_contract_budget,
                    Some(streams),
                );
                (rows, mode_suffix, started.elapsed().as_millis())
            });
            let (contract_rows, contract_ms) = contract.join().unwrap_or_else(|_| (Vec::new(), 0));
            let (transduced_rows, transduced_ms) =
                transduced.join().unwrap_or_else(|_| (Vec::new(), 0));
            let (sts_off_contract_rows, sts_off_contract_ms, sts_off_contract_fast_used) =
                sts_off_contract
                    .join()
                    .unwrap_or_else(|_| (Vec::new(), 0, false));
            let (symliquid_rows, symliquid_ms) =
                symliquid.join().unwrap_or_else(|_| (Vec::new(), 0));
            let (state_rows, state_ms) = state.join().unwrap_or_else(|_| (Vec::new(), 0));
            let (beam_rows, beam_ms) = beam.join().unwrap_or_else(|_| (Vec::new(), 0));
            let (ngram_rows, ngram_ms) = ngram.join().unwrap_or_else(|_| (Vec::new(), 0));
            let (greedy_rows, greedy_ms) = greedy.join().unwrap_or_else(|_| (Vec::new(), 0));
            let (prompt_contract_rows, prompt_contract_mode_suffix, prompt_contract_ms) =
                prompt_contract
                    .join()
                    .unwrap_or_else(|_| (Vec::new(), "_sts_conditioned", 0));
            (
                contract_rows,
                contract_ms,
                transduced_rows,
                transduced_ms,
                sts_off_contract_rows,
                sts_off_contract_ms,
                sts_off_contract_fast_used,
                symliquid_rows,
                symliquid_ms,
                state_rows,
                state_ms,
                beam_rows,
                beam_ms,
                ngram_rows,
                ngram_ms,
                greedy_rows,
                greedy_ms,
                prompt_contract_rows,
                prompt_contract_mode_suffix,
                prompt_contract_ms,
            )
        });
        let bridge_started = Instant::now();
        let mut bridge_pool = Vec::new();
        bridge_pool.extend(contract_rows.iter().cloned());
        bridge_pool.extend(transduced_rows.iter().cloned());
        bridge_pool.extend(sts_off_contract_rows.iter().cloned());
        bridge_pool.extend(symliquid_rows.iter().cloned());
        bridge_pool.extend(state_rows.iter().cloned());
        bridge_pool.extend(beam_rows.iter().cloned());
        bridge_pool.extend(ngram_rows.iter().cloned());
        bridge_pool.extend(greedy_rows.iter().cloned());
        let conditioned_family_count = contract_rows.len()
            + transduced_rows.len()
            + symliquid_rows.len()
            + state_rows.len()
            + beam_rows.len()
            + ngram_rows.len()
            + greedy_rows.len()
            + prompt_contract_rows.len();
        let demote_sts_preference = sts_decoder_control_demotes_sts_preference(Some(streams));
        let bridge_limit = if demote_sts_preference {
            token_inventory_budget.clamp(4, 12)
        } else {
            token_inventory_budget.clamp(12, 32)
        };
        let mut sts_bridge_rows =
            sts_bridge_ranked_outputs_from_pool(task, bridge_pool, bridge_limit, streams);
        let reused_bridge_ms = bridge_started.elapsed().as_millis();
        let mut full_bridge_fallback_ms = 0u128;
        let family_ready_target = if low_latency_fanout {
            low_latency_accept_count
        } else {
            limit.max(1)
        };
        let demoted_sts_has_useful_non_sts_surface = demote_sts_preference
            && (!sts_off_contract_rows.is_empty()
                || !prompt_contract_rows.is_empty()
                || !contract_rows.is_empty()
                || conditioned_family_count >= family_ready_target);
        let skip_full_bridge_fallback = (skip_full_sts_bridge_when_family_ready(task)
            && conditioned_family_count >= family_ready_target)
            || demoted_sts_has_useful_non_sts_surface;
        if sts_bridge_rows.len() < limit.max(1) {
            if skip_full_bridge_fallback {
                expression_timing_ms.insert(
                    "sts_conditioned_contract_token_bridge_full_fallback_skipped_family_ready"
                        .to_string(),
                    1,
                );
                if demoted_sts_has_useful_non_sts_surface {
                    expression_timing_ms.insert(
                        "sts_conditioned_contract_token_bridge_full_fallback_skipped_demoted_policy"
                            .to_string(),
                        1,
                    );
                }
            } else {
                let fallback_started = Instant::now();
                let fallback_rows = sts_conditioned_contract_token_bridge_bodies(
                    task,
                    body_ngram,
                    state_sequence_decoder,
                    symliquid_state_decoder,
                    readout,
                    vocab,
                    seed ^ 0x57A7_C0DE,
                    bridge_limit,
                    streams,
                    precomputed_beams,
                );
                full_bridge_fallback_ms = fallback_started.elapsed().as_millis();
                let mut seen = sts_bridge_rows.iter().cloned().collect::<HashSet<_>>();
                for body in fallback_rows {
                    if seen.insert(body.clone()) {
                        sts_bridge_rows.push(body);
                    }
                    if sts_bridge_rows.len() >= bridge_limit {
                        break;
                    }
                }
            }
        }
        expression_timing_ms.insert(
            "sts_conditioned_contract_token_bridge_conditioned_family_count".to_string(),
            conditioned_family_count as u128,
        );
        expression_timing_ms.insert(
            "sts_conditioned_contract_token_bridge_sts_demoted_by_control".to_string(),
            demote_sts_preference as u128,
        );
        expression_timing_ms.insert(
            "sts_conditioned_contract_token_bridge_fallback_skipped".to_string(),
            if skip_full_bridge_fallback { 1 } else { 0 },
        );
        expression_timing_ms.insert(
            "template_free_parallel_family_wall".to_string(),
            parallel_started.elapsed().as_millis(),
        );
        expression_timing_ms.insert("contract_guided_token_decoder".to_string(), contract_ms);
        expression_timing_ms.insert(
            "sts_conditioned_contract_token_bridge".to_string(),
            reused_bridge_ms + full_bridge_fallback_ms,
        );
        expression_timing_ms.insert(
            "sts_conditioned_contract_token_bridge_reused_family_pool".to_string(),
            reused_bridge_ms,
        );
        expression_timing_ms.insert(
            "sts_conditioned_contract_token_bridge_full_fallback".to_string(),
            full_bridge_fallback_ms,
        );
        expression_timing_ms.insert("learned_contract_transduction".to_string(), transduced_ms);
        expression_timing_ms.insert(
            "sts_off_contract_guided_token_decoder".to_string(),
            sts_off_contract_ms,
        );
        expression_timing_ms.insert(
            "sts_off_contract_guided_token_decoder_fast_precomputed_family_used".to_string(),
            sts_off_contract_fast_precomputed_used as u128,
        );
        expression_timing_ms.insert(
            "sts_off_contract_guided_token_decoder_public_fast_metadata_policy".to_string(),
            (task.split == "public_calibration"
                && fast_sts_off_from_precomputed_family_enabled(task)) as u128,
        );
        expression_timing_ms.insert("symliquid_state_bodies".to_string(), symliquid_ms);
        expression_timing_ms.insert("state_sequence_bodies".to_string(), state_ms);
        expression_timing_ms.insert("beam_bodies".to_string(), beam_ms);
        expression_timing_ms.insert("body_ngram_bodies".to_string(), ngram_ms);
        expression_timing_ms.insert("greedy_body".to_string(), greedy_ms);
        expression_timing_ms.insert(
            "prompt_contract_program_decoder".to_string(),
            prompt_contract_ms,
        );
        rows.extend(contract_rows.into_iter().map(|body| {
            full_body_candidate_expression(
                body,
                format!("rust_code_lm_contract_guided_token_decoder{conditioned_suffix}"),
            )
        }));
        rows.extend(prompt_contract_rows.into_iter().map(|body| {
            full_body_candidate_expression(
                body,
                format!("rust_code_lm_contract_guided_prompt_program_decoder{prompt_contract_mode_suffix}"),
            )
        }));
        rows.extend(sts_bridge_rows.into_iter().map(|body| {
            full_body_candidate_expression(
                body,
                "rust_code_lm_sts_conditioned_contract_guided_token_decoder".to_string(),
            )
        }));
        rows.extend(transduced_rows.into_iter().map(|body| {
            full_body_candidate_expression(
                body,
                format!("rust_code_lm_contract_transduced_token_decoder{conditioned_suffix}"),
            )
        }));
        rows.extend(sts_off_contract_rows.into_iter().map(|body| {
            full_body_candidate_expression(
                body,
                "rust_code_lm_contract_guided_token_decoder".to_string(),
            )
        }));
        rows.extend(symliquid_rows.into_iter().map(|body| {
            full_body_candidate_expression(
                body,
                format!("rust_code_lm_symliquid_recurrent_state_decoder{conditioned_suffix}"),
            )
        }));
        rows.extend(state_rows.into_iter().map(|body| {
            full_body_candidate_expression(
                body,
                format!("rust_code_lm_sparse_state_sequence_decoder{conditioned_suffix}"),
            )
        }));
        rows.extend(beam_rows.into_iter().map(|body| {
            full_body_candidate_expression(
                body,
                format!("rust_code_lm_full_body_token_beam{conditioned_suffix}"),
            )
        }));
        rows.extend(ngram_rows.into_iter().map(|body| {
            full_body_candidate_expression(
                body,
                format!("rust_code_lm_private_body_ngram_token_decoder{conditioned_suffix}"),
            )
        }));
        rows.extend(greedy_rows.into_iter().map(|body| {
            full_body_candidate_expression(
                body,
                format!("rust_code_lm_greedy_body_token_decoder{conditioned_suffix}"),
            )
        }));
        ensure_low_latency_same_seed_non_sts_comparator(
            task,
            body_prototypes,
            &mut rows,
            &mut expression_timing_ms,
            &mut low_latency_non_sts_comparator_added,
            sts_streams.is_some(),
        );
        append_eligible_receiver_inventory_router_candidates_with_timing(
            task,
            &mut rows,
            limit,
            &mut expression_timing_ms,
            "eligible_receiver_inventory_parallel_sts_final_retry",
        );
        append_private_to_public_receiver_inventory_bridge_candidates_with_timing(
            task,
            &mut rows,
            limit,
            sts_streams,
            &mut expression_timing_ms,
            "private_to_public_receiver_inventory_bridge_parallel_sts_final_retry",
        );
        append_broad_transfer_residual_retry_candidates_with_timing(
            task,
            &mut rows,
            limit,
            &mut expression_timing_ms,
            "broad_transfer_residual_parallel_sts_final_retry",
        );
        let mut finalize_started = Instant::now();
        let finalized = finalize_candidate_expression_rows(task, rows, sts_streams, limit);
        record_candidate_timing(
            &mut expression_timing_ms,
            "finalize_candidate_expression_rows",
            &mut finalize_started,
        );
        return (finalized, expression_timing_ms);
    }
    let (contract_guided_bodies, contract_guided_internal_timing) =
        contract_guided_token_decoder_bodies_with_timing(
            task,
            body_ngram,
            state_sequence_decoder,
            symliquid_state_decoder,
            readout,
            vocab,
            seed ^ 0xC0A7_5EED,
            contract_token_budget,
            sts_streams,
            precomputed_beams,
            precomputed_state_sequence,
            precomputed_symliquid_state,
        );
    for body in contract_guided_bodies {
        rows.push(CandidateExpression {
            expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
            body,
            mode: format!("rust_code_lm_contract_guided_token_decoder{conditioned_suffix}"),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: false,
        });
    }
    for (phase, elapsed_ms) in contract_guided_internal_timing {
        expression_timing_ms.insert(format!("contract_guided_internal_{phase}"), elapsed_ms);
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "contract_guided_token_decoder",
        &mut expression_stage_started,
    );
    if low_latency_fanout && sts_streams.is_some() {
        ensure_low_latency_same_seed_non_sts_comparator(
            task,
            body_prototypes,
            &mut rows,
            &mut expression_timing_ms,
            &mut low_latency_non_sts_comparator_added,
            true,
        );
        append_broad_transfer_residual_retry_candidates_with_timing(
            task,
            &mut rows,
            limit,
            &mut expression_timing_ms,
            "broad_transfer_residual_low_latency_pre_sts",
        );
        let finalized = finalize_candidate_expression_rows(task, rows, sts_streams, limit);
        record_candidate_timing(
            &mut expression_timing_ms,
            "low_latency_pre_sts_finalize",
            &mut expression_stage_started,
        );
        if finalized
            .iter()
            .filter(|candidate| {
                low_latency_full_body_candidate_accepted(task, candidate, sts_streams)
            })
            .count()
            >= low_latency_accept_count
        {
            return (finalized, expression_timing_ms);
        }
        rows = finalized;
    }
    if let Some(streams) = sts_streams {
        let demote_sts_preference = sts_decoder_control_demotes_sts_preference(Some(streams));
        let sts_contract_token_budget = if template_free {
            if low_latency_fanout {
                token_inventory_budget.clamp(4, 8)
            } else if demote_sts_preference {
                token_inventory_budget.clamp(4, 12)
            } else {
                token_inventory_budget.clamp(12, 32)
            }
        } else if public_calibration {
            (limit / 2 + 2).clamp(3, 8)
        } else {
            (limit / 3 + 2).clamp(3, 6)
        };
        let mut bridge_rows = Vec::new();
        let mut bridge_seen = HashSet::new();
        if low_latency_fanout {
            let bridge_started = Instant::now();
            let mut bridge_pool = rows
                .iter()
                .filter(|candidate| {
                    candidate.full_body_token_candidate
                        && !candidate.expression_memory_fallback
                        && !candidate.sts_candidate_expression_used
                })
                .map(|candidate| candidate.body.clone())
                .collect::<Vec<_>>();
            if let Some(precomputed) = precomputed_beams {
                bridge_pool.extend(precomputed.iter().take(sts_contract_token_budget).cloned());
            }
            for body in sts_bridge_ranked_outputs_from_pool(
                task,
                bridge_pool,
                sts_contract_token_budget,
                streams,
            ) {
                if bridge_seen.insert(body.clone()) {
                    bridge_rows.push(body);
                }
            }
            expression_timing_ms.insert(
                "sts_conditioned_contract_token_bridge_reused_low_latency_pool".to_string(),
                bridge_started.elapsed().as_millis(),
            );
        }
        let accepted_non_sts_rows = rows
            .iter()
            .filter(|candidate| {
                candidate.full_body_token_candidate
                    && !candidate.expression_memory_fallback
                    && !candidate.sts_candidate_expression_used
                    && low_latency_full_body_candidate_accepted(task, candidate, sts_streams)
            })
            .count();
        let demoted_sts_low_latency_surface_ready = demote_sts_preference
            && (accepted_non_sts_rows >= low_latency_accept_count || !bridge_rows.is_empty());
        if low_latency_fanout
            && template_free
            && bridge_rows.len() < limit.max(1)
            && !demoted_sts_low_latency_surface_ready
        {
            let transduction_started = Instant::now();
            for body in learned_contract_transduction_bodies(
                task,
                body_prototypes,
                contract_token_budget,
                sts_streams,
            ) {
                rows.push(CandidateExpression {
                    expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                    body,
                    mode: format!(
                        "rust_code_lm_contract_transduced_token_decoder{conditioned_suffix}"
                    ),
                    compositional_token_candidate: true,
                    full_body_token_candidate: true,
                    expression_memory_fallback: false,
                    sts_candidate_expression_used: false,
                });
            }
            contract_transduction_added = true;
            expression_timing_ms.insert(
                "low_latency_pre_recursive_contract_transduction".to_string(),
                transduction_started.elapsed().as_millis(),
            );
            ensure_low_latency_same_seed_non_sts_comparator(
                task,
                body_prototypes,
                &mut rows,
                &mut expression_timing_ms,
                &mut low_latency_non_sts_comparator_added,
                true,
            );
            let finalized = finalize_candidate_expression_rows(task, rows, sts_streams, limit);
            record_candidate_timing(
                &mut expression_timing_ms,
                "low_latency_pre_recursive_transduction_finalize",
                &mut expression_stage_started,
            );
            if finalized
                .iter()
                .filter(|candidate| {
                    low_latency_full_body_candidate_accepted(task, candidate, sts_streams)
                })
                .count()
                >= low_latency_accept_count
            {
                return (finalized, expression_timing_ms);
            }
            rows = finalized;
        }
        if template_free {
            let prompt_contract_started = Instant::now();
            let (prompt_contract_bodies, prompt_contract_mode_suffix) =
                prompt_contract_bodies_with_visible_fallback(
                    task,
                    prompt_contract_budget,
                    sts_streams,
                );
            for body in prompt_contract_bodies {
                rows.push(CandidateExpression {
                    expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                    body,
                    mode: format!(
                        "rust_code_lm_contract_guided_prompt_program_decoder{prompt_contract_mode_suffix}"
                    ),
                    compositional_token_candidate: true,
                    full_body_token_candidate: true,
                    expression_memory_fallback: false,
                    sts_candidate_expression_used: false,
                });
            }
            expression_timing_ms.insert(
                "prompt_contract_program_decoder".to_string(),
                prompt_contract_started.elapsed().as_millis(),
            );
            ensure_low_latency_same_seed_non_sts_comparator(
                task,
                body_prototypes,
                &mut rows,
                &mut expression_timing_ms,
                &mut low_latency_non_sts_comparator_added,
                true,
            );
            let finalized = finalize_candidate_expression_rows(task, rows, sts_streams, limit);
            record_candidate_timing(
                &mut expression_timing_ms,
                "low_latency_prompt_contract_finalize",
                &mut expression_stage_started,
            );
            if finalized
                .iter()
                .filter(|candidate| {
                    low_latency_full_body_candidate_accepted(task, candidate, sts_streams)
                })
                .count()
                >= low_latency_accept_count
            {
                return (finalized, expression_timing_ms);
            }
            rows = finalized;
        }
        let accepted_non_sts_rows = rows
            .iter()
            .filter(|candidate| {
                candidate.full_body_token_candidate
                    && !candidate.expression_memory_fallback
                    && !candidate.sts_candidate_expression_used
                    && low_latency_full_body_candidate_accepted(task, candidate, sts_streams)
            })
            .count();
        let skip_recursive_sts_fallback_for_demoted_policy = demote_sts_preference
            && (accepted_non_sts_rows >= low_latency_accept_count || !bridge_rows.is_empty());
        expression_timing_ms.insert(
            "sts_conditioned_contract_token_bridge_sts_demoted_by_control".to_string(),
            demote_sts_preference as u128,
        );
        let needs_recursive_sts_fallback = (!low_latency_fanout
            || bridge_rows.len() < limit.max(1))
            && !skip_recursive_sts_fallback_for_demoted_policy;
        if needs_recursive_sts_fallback {
            let fallback_started = Instant::now();
            let fallback_rows = sts_conditioned_contract_token_bridge_bodies(
                task,
                body_ngram,
                state_sequence_decoder,
                symliquid_state_decoder,
                readout,
                vocab,
                seed ^ 0x57A7_C0DE,
                sts_contract_token_budget,
                streams,
                precomputed_beams,
            );
            for body in fallback_rows {
                if bridge_seen.insert(body.clone()) {
                    bridge_rows.push(body);
                }
            }
            expression_timing_ms.insert(
                "sts_conditioned_contract_token_bridge_recursive_fallback".to_string(),
                fallback_started.elapsed().as_millis(),
            );
        } else if skip_recursive_sts_fallback_for_demoted_policy {
            expression_timing_ms.insert(
                "sts_conditioned_contract_token_bridge_recursive_fallback_skipped_demoted_policy"
                    .to_string(),
                1,
            );
        }
        for body in bridge_rows {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_sts_conditioned_contract_guided_token_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "sts_conditioned_contract_token_bridge",
        &mut expression_stage_started,
    );
    if template_free && !contract_transduction_added {
        for body in learned_contract_transduction_bodies(
            task,
            body_prototypes,
            contract_token_budget,
            sts_streams,
        ) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: format!("rust_code_lm_contract_transduced_token_decoder{conditioned_suffix}"),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "learned_contract_transduction",
        &mut expression_stage_started,
    );
    if low_latency_fanout {
        ensure_low_latency_same_seed_non_sts_comparator(
            task,
            body_prototypes,
            &mut rows,
            &mut expression_timing_ms,
            &mut low_latency_non_sts_comparator_added,
            sts_streams.is_some(),
        );
        let finalized = finalize_candidate_expression_rows(task, rows, sts_streams, limit);
        record_candidate_timing(
            &mut expression_timing_ms,
            "low_latency_initial_finalize",
            &mut expression_stage_started,
        );
        if finalized
            .iter()
            .filter(|candidate| {
                low_latency_full_body_candidate_accepted(task, candidate, sts_streams)
            })
            .count()
            >= low_latency_accept_count
        {
            return (finalized, expression_timing_ms);
        }
        rows = finalized;
    }
    if sts_streams.is_some() {
        let budget = (contract_token_budget / 2).clamp(2, 8);
        let mut fast_precomputed_used = false;
        let sts_off_rows = if let Some(rows) = fast_sts_off_contract_rows_from_precomputed_family(
            task,
            body_prototypes,
            precomputed_state_sequence_no_sts,
            precomputed_symliquid_state_no_sts,
            budget,
        ) {
            fast_precomputed_used = true;
            rows
        } else {
            contract_guided_token_decoder_bodies(
                task,
                body_ngram,
                state_sequence_decoder,
                symliquid_state_decoder,
                readout,
                vocab,
                seed ^ 0xC0A7_0FF5,
                budget,
                None,
                None,
                precomputed_state_sequence_no_sts,
                precomputed_symliquid_state_no_sts,
            )
        };
        expression_timing_ms.insert(
            "sts_off_contract_guided_token_decoder_fast_precomputed_family_used".to_string(),
            fast_precomputed_used as u128,
        );
        expression_timing_ms.insert(
            "sts_off_contract_guided_token_decoder_public_fast_metadata_policy".to_string(),
            (task.split == "public_calibration"
                && fast_sts_off_from_precomputed_family_enabled(task)) as u128,
        );
        for body in sts_off_rows {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_contract_guided_token_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "sts_off_contract_guided_token_decoder",
        &mut expression_stage_started,
    );
    if !template_free {
        for body in causal_contract_skeleton_bodies(task, causal_contract_budget, sts_streams) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: format!("rust_code_lm_causal_contract_skeleton_decoder{conditioned_suffix}"),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in contract_guided_skeleton_bodies(task, contract_guided_budget, sts_streams) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: format!("rust_code_lm_contract_guided_skeleton_decoder{conditioned_suffix}"),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in local_adapter_edge_skeleton_bodies(task, local_adapter_budget, sts_streams) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: format!(
                    "rust_code_lm_local_adapter_edge_skeleton_decoder{conditioned_suffix}"
                ),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in execution_shape_skeleton_bodies(task, execution_shape_budget, sts_streams) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: format!("rust_code_lm_execution_shape_skeleton_decoder{conditioned_suffix}"),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "template_skeleton_families",
        &mut expression_stage_started,
    );
    if !template_free && sts_streams.is_some() {
        let sts_skeleton_budget = if public_calibration {
            (limit / 3 + 2).clamp(2, 5)
        } else {
            (limit / 3 + 3).clamp(4, 8)
        };
        for body in sts_causal_skeleton_bodies(task, sts_streams, sts_skeleton_budget) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: format!("rust_code_lm_sts_causal_skeleton_decoder{conditioned_suffix}"),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "sts_causal_skeleton_families",
        &mut expression_stage_started,
    );
    if !template_free {
        for body in semantic_plan_v2_bodies(
            task,
            body_ngram,
            seed ^ 0xDEC0_DE20,
            semantic_plan_budget,
            sts_streams,
        ) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: format!("rust_code_lm_semantic_plan_v2_token_decoder{conditioned_suffix}"),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "semantic_plan_v2",
        &mut expression_stage_started,
    );
    let state_family_wall_started = Instant::now();
    let (symliquid_rows, symliquid_ms, state_rows, state_ms) =
        if parallel_sts_candidate_fanout_enabled() {
            std::thread::scope(|scope| {
                let symliquid = scope.spawn(|| {
                    let started = Instant::now();
                    let rows = precomputed_decoder_family_rows(
                        precomputed_symliquid_state,
                        symliquid_budget,
                    )
                    .unwrap_or_else(|| {
                        symliquid_state_bodies(
                            task,
                            symliquid_state_decoder,
                            body_ngram,
                            vocab,
                            seed,
                            symliquid_budget,
                            sts_streams,
                        )
                    });
                    (rows, started.elapsed().as_millis())
                });
                let state = scope.spawn(|| {
                    let started = Instant::now();
                    let rows =
                        precomputed_decoder_family_rows(precomputed_state_sequence, state_budget)
                            .unwrap_or_else(|| {
                                state_sequence_bodies(
                                    task,
                                    state_sequence_decoder,
                                    body_ngram,
                                    vocab,
                                    seed,
                                    state_budget,
                                    sts_streams,
                                )
                            });
                    (rows, started.elapsed().as_millis())
                });
                let (symliquid_rows, symliquid_ms) =
                    symliquid.join().unwrap_or_else(|_| (Vec::new(), 0));
                let (state_rows, state_ms) = state.join().unwrap_or_else(|_| (Vec::new(), 0));
                (symliquid_rows, symliquid_ms, state_rows, state_ms)
            })
        } else {
            let symliquid_started = Instant::now();
            let symliquid_rows =
                precomputed_decoder_family_rows(precomputed_symliquid_state, symliquid_budget)
                    .unwrap_or_else(|| {
                        symliquid_state_bodies(
                            task,
                            symliquid_state_decoder,
                            body_ngram,
                            vocab,
                            seed,
                            symliquid_budget,
                            sts_streams,
                        )
                    });
            let symliquid_ms = symliquid_started.elapsed().as_millis();
            let state_started = Instant::now();
            let state_rows =
                precomputed_decoder_family_rows(precomputed_state_sequence, state_budget)
                    .unwrap_or_else(|| {
                        state_sequence_bodies(
                            task,
                            state_sequence_decoder,
                            body_ngram,
                            vocab,
                            seed,
                            state_budget,
                            sts_streams,
                        )
                    });
            let state_ms = state_started.elapsed().as_millis();
            (symliquid_rows, symliquid_ms, state_rows, state_ms)
        };
    expression_timing_ms.insert("symliquid_state_bodies".to_string(), symliquid_ms);
    expression_timing_ms.insert("state_sequence_bodies".to_string(), state_ms);
    expression_timing_ms.insert(
        "state_symliquid_parallel_family_wall".to_string(),
        state_family_wall_started.elapsed().as_millis(),
    );
    expression_stage_started = Instant::now();
    for body in symliquid_rows {
        rows.push(CandidateExpression {
            expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
            body,
            mode: format!("rust_code_lm_symliquid_recurrent_state_decoder{conditioned_suffix}"),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: false,
        });
    }
    if !template_free {
        for body in state_sequence_seeded_completion_bodies(task, vocab)
            .into_iter()
            .take(3)
        {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: format!(
                    "rust_code_lm_sparse_state_sequence_seeded_decoder{conditioned_suffix}"
                ),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in seeded_body_ngram_completion_bodies(task, body_ngram, seed, state_budget)
            .into_iter()
            .take(state_budget)
        {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: format!("rust_code_lm_seeded_body_ngram_token_decoder{conditioned_suffix}"),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "template_seeded_state_and_ngram",
        &mut expression_stage_started,
    );
    for body in state_rows {
        rows.push(CandidateExpression {
            expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
            body,
            mode: format!("rust_code_lm_sparse_state_sequence_decoder{conditioned_suffix}"),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: false,
        });
    }
    expression_stage_started = Instant::now();
    if !template_free && !public_calibration {
        for body in body_prototype_bodies(task, body_prototypes, 2, sts_streams) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: format!(
                    "rust_code_lm_private_body_prototype_token_decoder{conditioned_suffix}"
                ),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "private_body_prototypes",
        &mut expression_stage_started,
    );
    let beam_bodies_for_task = precomputed_beams
        .cloned()
        .unwrap_or_else(|| beam_bodies(task, readout, vocab, seed, beam_budget, sts_streams));
    let beam_rows = beam_bodies_for_task
        .into_iter()
        .map(|body| CandidateExpression {
            expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
            body,
            mode: format!("rust_code_lm_full_body_token_beam{conditioned_suffix}"),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: false,
        })
        .collect::<Vec<_>>();
    rows.extend(beam_rows.iter().take(beam_budget).cloned());
    record_candidate_timing(
        &mut expression_timing_ms,
        "beam_bodies",
        &mut expression_stage_started,
    );
    let ngram_budget = if template_free {
        limit.clamp(4, 10)
    } else if public_calibration {
        limit.clamp(2, 4)
    } else {
        limit.clamp(1, 2)
    };
    for body in body_ngram_bodies(task, body_ngram, seed, ngram_budget, sts_streams) {
        rows.push(CandidateExpression {
            expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
            body,
            mode: format!("rust_code_lm_private_body_ngram_token_decoder{conditioned_suffix}"),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: false,
        });
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "body_ngram_bodies",
        &mut expression_stage_started,
    );
    if !template_free && !public_calibration {
        for expr in sts_candidate_expressions(sts_streams).into_iter().take(2) {
            rows.push(CandidateExpression {
                body: body_from_expression(&expr),
                expr,
                mode: "rust_code_lm_native_sts_stream_expression".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: false,
                expression_memory_fallback: false,
                sts_candidate_expression_used: true,
            });
        }
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "native_sts_stream_expressions",
        &mut expression_stage_started,
    );
    if let Some(body) =
        greedy_or_precomputed_beam_body(task, readout, vocab, sts_streams, precomputed_beams)
    {
        rows.push(CandidateExpression {
            expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
            body,
            mode: format!("rust_code_lm_greedy_body_token_decoder{conditioned_suffix}"),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: false,
        });
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "greedy_body",
        &mut expression_stage_started,
    );
    if !template_free && sts_streams.is_some() {
        let off_state_budget = (limit / 4 + 1).clamp(2, 3);
        for body in causal_contract_skeleton_bodies(task, off_state_budget, None) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_causal_contract_skeleton_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in contract_guided_skeleton_bodies(task, off_state_budget, None) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_contract_guided_skeleton_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in local_adapter_edge_skeleton_bodies(task, off_state_budget, None) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_local_adapter_edge_skeleton_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in execution_shape_skeleton_bodies(task, off_state_budget, None) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_execution_shape_skeleton_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in
            semantic_plan_v2_bodies(task, body_ngram, seed ^ 0xDE51_570F, off_state_budget, None)
        {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_semantic_plan_v2_token_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in symliquid_state_bodies(
            task,
            symliquid_state_decoder,
            body_ngram,
            vocab,
            seed ^ 0x51A7_3F00,
            off_state_budget,
            None,
        ) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_symliquid_recurrent_state_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in state_sequence_bodies(
            task,
            state_sequence_decoder,
            body_ngram,
            vocab,
            seed ^ 0x5EED_5EED,
            off_state_budget,
            None,
        ) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_sparse_state_sequence_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in beam_bodies(task, readout, vocab, seed ^ 0xBEE0, off_state_budget, None) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_full_body_token_beam".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        for body in body_ngram_bodies(task, body_ngram, seed ^ 0xC0DE, off_state_budget, None) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_private_body_ngram_token_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
        if let Some(body) = greedy_body(task, readout, vocab, None) {
            rows.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                body,
                mode: "rust_code_lm_greedy_body_token_decoder".to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "sts_off_state_family_bodies",
        &mut expression_stage_started,
    );
    if !template_free {
        let edge_repair_budget = if public_calibration {
            (limit / 4 + 1).clamp(2, 4)
        } else {
            (limit / 4 + 2).clamp(2, 5)
        };
        let edge_repairs = edge_exec_repair_candidates(
            task,
            &rows,
            conditioned_suffix,
            edge_repair_budget,
            sts_streams,
        );
        rows.extend(edge_repairs);
    }
    record_candidate_timing(
        &mut expression_timing_ms,
        "edge_exec_repair_candidates",
        &mut expression_stage_started,
    );
    append_eligible_receiver_inventory_router_candidates_with_timing(
        task,
        &mut rows,
        limit,
        &mut expression_timing_ms,
        "eligible_receiver_inventory_final_retry",
    );
    append_broad_transfer_residual_retry_candidates_with_timing(
        task,
        &mut rows,
        limit,
        &mut expression_timing_ms,
        "broad_transfer_residual_final_retry",
    );
    append_private_to_public_receiver_inventory_bridge_candidates_with_timing(
        task,
        &mut rows,
        limit,
        sts_streams,
        &mut expression_timing_ms,
        "private_to_public_receiver_inventory_bridge_final_retry",
    );
    let _ = (expression_bank, readout);
    rows.extend(beam_rows.into_iter().skip(beam_budget));
    let finalized = finalize_candidate_expression_rows(task, rows, sts_streams, limit);
    record_candidate_timing(
        &mut expression_timing_ms,
        "finalize_candidate_expression_rows",
        &mut expression_stage_started,
    );
    (finalized, expression_timing_ms)
}

pub(super) fn finalize_candidate_expression_rows(
    task: &CodeTask,
    rows: Vec<CandidateExpression>,
    sts_streams: Option<&BTreeMap<String, String>>,
    candidate_limit: usize,
) -> Vec<CandidateExpression> {
    let mut variant_cache = CandidateVariantCache::default();
    let mut rows = rows
        .into_iter()
        .map(normalize_candidate_expression)
        .flat_map(|candidate| {
            learned_candidate_expression_variants_cached(task, candidate, &mut variant_cache)
        })
        .collect::<Vec<_>>();
    let low_latency_prefilter = low_latency_candidate_fanout_enabled() && candidate_limit <= 8;
    let budget =
        cheap_prefilter_budget(candidate_limit, if low_latency_prefilter { 12 } else { 48 });
    if rows.len() > budget {
        truncate_by_cheap_prefilter(task, &mut rows, sts_streams, budget);
    }
    let transfer_score_budget =
        finalize_transfer_score_prefilter_budget(candidate_limit, low_latency_prefilter);
    if rows.len() > transfer_score_budget {
        truncate_by_cheap_prefilter(task, &mut rows, sts_streams, transfer_score_budget);
    }
    let mut scored_rows = rows
        .into_iter()
        .map(|candidate| {
            let score = candidate_transfer_score(task, &candidate, sts_streams);
            (candidate, score)
        })
        .collect::<Vec<_>>();
    scored_rows.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.mode.cmp(&b.0.mode))
            .then_with(|| a.0.body.cmp(&b.0.body))
    });
    scored_rows
        .into_iter()
        .map(|(candidate, _score)| candidate)
        .collect()
}
