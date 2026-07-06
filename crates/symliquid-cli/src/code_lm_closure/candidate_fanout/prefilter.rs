use super::*;

#[derive(Default)]
pub(super) struct CandidateVerificationCache {
    verifier: HashMap<String, DecoderContractVerification>,
    guardrail: HashMap<String, DeterministicGuardrail>,
    verifier_calls: usize,
    verifier_hits: usize,
    guardrail_calls: usize,
    guardrail_hits: usize,
}

impl CandidateVerificationCache {
    pub(super) fn verifier(
        &mut self,
        task: &CodeTask,
        body: &str,
        sts_streams: Option<&BTreeMap<String, String>>,
    ) -> DecoderContractVerification {
        let key = verification_cache_key(body, sts_streams);
        if let Some(cached) = self.verifier.get(&key) {
            self.verifier_hits = self.verifier_hits.saturating_add(1);
            return cached.clone();
        }
        self.verifier_calls = self.verifier_calls.saturating_add(1);
        let verification = decoder_contract_verifier_v1(task, body, sts_streams);
        self.verifier.insert(key, verification.clone());
        verification
    }

    pub(super) fn guardrail(&mut self, task: &CodeTask, body: &str) -> DeterministicGuardrail {
        if let Some(cached) = self.guardrail.get(body) {
            self.guardrail_hits = self.guardrail_hits.saturating_add(1);
            return cached.clone();
        }
        self.guardrail_calls = self.guardrail_calls.saturating_add(1);
        let guardrail = deterministic_full_body_guardrail(task, body);
        self.guardrail.insert(body.to_string(), guardrail.clone());
        guardrail
    }

    pub(super) fn metrics(&self) -> Value {
        json!({
            "policy": "project_theseus_candidate_verification_cache_v1",
            "scope": "single_task_candidate_fanout",
            "verifier_unique_body_calls": self.verifier_calls,
            "verifier_cache_hits": self.verifier_hits,
            "guardrail_unique_body_calls": self.guardrail_calls,
            "guardrail_cache_hits": self.guardrail_hits,
            "score_semantics": "runtime_prefilter_efficiency_only_not_capability_evidence",
        })
    }
}

fn verification_cache_key(body: &str, sts_streams: Option<&BTreeMap<String, String>>) -> String {
    let sts_key = sts_streams.map_or_else(
        || "no_sts".to_string(),
        |streams| {
            let joined = streams
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!("sts:{}", stable_hash_hex(&joined))
        },
    );
    format!("{sts_key}\0{body}")
}

pub(super) fn cached_decoder_contract_verification(
    task: &CodeTask,
    body: &str,
    sts_streams: Option<&BTreeMap<String, String>>,
    cache: &mut HashMap<String, DecoderContractVerification>,
) -> DecoderContractVerification {
    let key = verification_cache_key(body, sts_streams);
    if let Some(cached) = cache.get(&key) {
        return cached.clone();
    }
    let verification = decoder_contract_verifier_v1(task, body, sts_streams);
    cache.insert(key, verification.clone());
    verification
}

pub(super) fn candidate_rejection_reason_cached(
    task: &CodeTask,
    candidate: &CandidateExpression,
    duplicate: bool,
    sts_streams: Option<&BTreeMap<String, String>>,
    cache: &mut CandidateVerificationCache,
) -> Option<&'static str> {
    if duplicate {
        return Some("duplicate_body");
    }
    if broad_private_train_prototype_candidate(candidate) {
        if !syntax_constrained_body(&candidate.body) {
            return Some("syntax_rejected");
        }
        if natural_language_leakage_in_body(&candidate.body)
            || scaffold_placeholder_body(&candidate.body)
        {
            return Some("prototype_safety_rejected");
        }
        return None;
    }
    if candidate.expression_memory_fallback || candidate.sts_candidate_expression_used {
        if !useful_generated_expression(&candidate.expr) {
            return Some("expression_not_useful");
        }
        if !useful_generated_body_for_task(task, &candidate.body) {
            return Some("body_not_useful");
        }
        if !syntax_constrained_body(&candidate.body) {
            return Some("syntax_rejected");
        }
        return cache
            .verifier(task, &candidate.body, sts_streams)
            .reasons
            .into_iter()
            .next();
    }
    if candidate.full_body_token_candidate {
        let verification_streams = if candidate.mode.contains("visible_contract_fallback") {
            None
        } else {
            sts_streams
        };
        let verification = cache.verifier(task, &candidate.body, verification_streams);
        if verification.passed && syntax_constrained_body(&candidate.body) {
            return None;
        }
        if !useful_generated_body_for_task(task, &candidate.body) {
            return Some("body_not_useful");
        }
        if !syntax_constrained_body(&candidate.body) {
            return Some("syntax_rejected");
        }
        return verification.reasons.into_iter().next();
    }
    if !useful_generated_body_for_task(task, &candidate.body) {
        return Some("body_not_useful");
    }
    cache
        .verifier(task, &candidate.body, sts_streams)
        .reasons
        .into_iter()
        .next()
}

pub(super) fn candidate_body_admissible_cached(
    task: &CodeTask,
    candidate: &CandidateExpression,
    sts_streams: Option<&BTreeMap<String, String>>,
    cache: &mut CandidateVerificationCache,
) -> bool {
    let verification_streams = if candidate.mode.contains("visible_contract_fallback") {
        None
    } else {
        sts_streams
    };
    let contract_verified = cache
        .verifier(task, &candidate.body, verification_streams)
        .passed;
    if broad_private_train_prototype_candidate(candidate) {
        return syntax_constrained_body(&candidate.body)
            && !natural_language_leakage_in_body(&candidate.body)
            && !scaffold_placeholder_body(&candidate.body);
    }
    if candidate.mode.contains("execution_shape_skeleton_decoder")
        && candidate.full_body_token_candidate
    {
        let hints = semantic_decoder_v2_plan_hints(task, sts_streams);
        let contract_hints = decoder_required_constructs(task);
        return contract_verified
            && (execution_shape_contract_ok(task, &candidate.body, &hints)
                || execution_shape_contract_ok(task, &candidate.body, &contract_hints));
    }
    if candidate.expression_memory_fallback || candidate.sts_candidate_expression_used {
        return useful_generated_expression(&candidate.expr)
            && useful_generated_body_for_task(task, &candidate.body)
            && contract_verified;
    }
    if candidate.full_body_token_candidate {
        return contract_verified
            && (useful_generated_body_for_task(task, &candidate.body)
                || syntax_constrained_body(&candidate.body));
    }
    useful_generated_body_for_task(task, &candidate.body) && contract_verified
}

pub(super) fn cheap_candidate_prefilter_score(
    task: &CodeTask,
    candidate: &CandidateExpression,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> f32 {
    let body = candidate.body.as_str();
    let lowered = body.to_lowercase();
    let hints = decoder_required_constructs(task);
    let mut score = beautiful_body_score(task, body) + body_transfer_score(task, body);
    if candidate.full_body_token_candidate {
        score += 2.0;
    }
    if candidate.compositional_token_candidate {
        score += 1.0;
    }
    if candidate.expression_memory_fallback || candidate.sts_candidate_expression_used {
        score -= 4.0;
    }
    if template_like_candidate(candidate) {
        score -= 6.0;
    }
    if visible_argument_contract_ok(task, body) {
        score += 2.0;
    } else {
        score -= 3.5;
    }
    if return_shape_contract_ok(task, &lowered) {
        score += 2.0;
    } else {
        score -= 4.0;
    }
    if required_construct_contract_ok_for_task(task, body, &hints) {
        score += 1.2;
    }
    if syntax_constrained_body(body) {
        score += 1.0;
    } else {
        score -= 8.0;
    }
    if scaffold_placeholder_body(body) || natural_language_leakage_in_body(body) {
        score -= 8.0;
    }
    if candidate.mode.contains("contract_guided_token_decoder") {
        score += 1.8;
    }
    if task.category.starts_with("edge_v3_")
        && candidate
            .mode
            .contains("contract_guided_token_decoder_category_context")
    {
        score += 14.0;
    }
    if broad_private_train_prototype_candidate(candidate) {
        score += 140.0;
    }
    if private_residual_v3_semantic_adapter_candidate(candidate) {
        score += 100.0;
    }
    if candidate.mode.contains("sts_conditioned") {
        if sts_decoder_control_demotes_sts_preference(sts_streams) {
            score -= 2.0;
        } else {
            score += 0.8;
        }
        if sts_streams.is_some() && !sts_decoder_control_demotes_sts_preference(sts_streams) {
            score += sts_skeleton_alignment_score(body, sts_streams).min(1.5);
        }
    }
    score += broad_public_floor_recovery_prefilter_score(task, body, &candidate.mode);
    score += edge_full_body_contract_bridge_v2_prefilter_score(task, candidate);
    score
}

fn edge_full_body_contract_bridge_v2_prefilter_score(
    task: &CodeTask,
    candidate: &CandidateExpression,
) -> f32 {
    let body = candidate.body.to_ascii_lowercase();
    let task_text =
        format!("{} {} {}", task.category, task.prompt, task.tags.join(" ")).to_ascii_lowercase();
    let required = decoder_required_constructs(task);
    let generation_hints = decoder_contract_generation_hints(task);
    let has_hint = |needle: &str| {
        required.contains(needle) || generation_hints.contains(needle) || task_text.contains(needle)
    };
    let mut score = 0.0f32;

    let inventory_candidate = candidate
        .mode
        .contains("eligible_receiver_inventory_router_v1")
        || candidate
            .mode
            .contains("private_to_public_receiver_inventory_bridge_v1")
        || candidate
            .mode
            .contains("local_adapter_edge_skeleton_decoder");
    if inventory_candidate {
        score += 0.4;
    }

    let nested_string_path_target = has_hint("nested_structure")
        || has_hint("nested_walk_helper")
        || has_hint("dict_and_list_branches")
        || has_hint("path_state_local")
        || (task_text.contains("nested")
            && task_text.contains("string")
            && task_text.contains("path"))
        || (task_text.contains("string leaves") && task_text.contains("nested"));
    if nested_string_path_target {
        let has_dict_branch = body.contains("isinstance(value, dict)")
            || body.contains("isinstance(item, dict)")
            || body.contains("isinstance(child, dict)");
        let has_list_branch = body.contains("isinstance(value, (list, tuple))")
            || body.contains("isinstance(value, list)")
            || body.contains("isinstance(item, (list, tuple))")
            || body.contains("enumerate(");
        let has_string_leaf = body.contains("isinstance(value, str)")
            || body.contains("isinstance(item, str)")
            || body.contains("isinstance(child, str)");
        let has_path_state = body.contains("path")
            && (body.contains("'/'.join")
                || body.contains("\"/\".join")
                || body.contains(" + '/' + "));
        let has_walk = body.contains("stack")
            || body.contains("def walk")
            || body.contains("while stack")
            || body.contains("walk(");
        if has_dict_branch && has_list_branch && has_string_leaf && has_path_state && has_walk {
            score += 8.0;
            if inventory_candidate {
                score += 1.5;
            }
        } else if has_dict_branch && has_list_branch && has_path_state {
            score += 3.0;
        } else {
            score -= 2.5;
        }
        if body.contains("factor * factor")
            || body.contains("range(2,")
            || body.contains(" % 2")
            || body.contains("out[j][1]")
            || body.contains("left[idx]")
        {
            score -= 7.5;
        }
    }

    let pairwise_numeric_zip_target = has_hint("zip_both_arguments")
        || has_hint("numeric_pair_guard")
        || has_hint("two_arg_interface")
        || (task_text.contains("pairwise") && task_text.contains("sum"))
        || (task_text.contains("shorter sequence") && task_text.contains("numeric"));
    if pairwise_numeric_zip_target {
        let uses_secondary = decoder_secondary_arg(task)
            .as_deref()
            .map(|secondary| body.contains(&secondary.to_ascii_lowercase()))
            .unwrap_or(false)
            || body.contains("right_items")
            || body.contains("other");
        let has_pair_iteration = body.contains("zip(")
            || body.contains("min(len(")
            || (body.contains("left_items") && body.contains("right_items"));
        let has_numeric_guard = body.contains("isinstance(left, (int, float))")
            || body.contains("isinstance(left, int)")
            || body.contains("isinstance(left, float)");
        let has_bool_guard = body.contains("isinstance(left, bool)")
            || body.contains("isinstance(right, bool)")
            || body.contains("not isinstance(left, bool)");
        let has_sum_append = body.contains("append(left + right)")
            || body.contains("append(left+right)")
            || body.contains("left + right");
        if uses_secondary
            && has_pair_iteration
            && has_numeric_guard
            && has_bool_guard
            && has_sum_append
        {
            score += 8.0;
            if inventory_candidate {
                score += 1.25;
            }
        } else if uses_secondary && has_pair_iteration && has_sum_append {
            score += 3.0;
        } else {
            score -= 2.0;
        }
        if body.contains("% 2")
            || body.contains("== right")
            || body.contains("append(left[idx]")
            || body.contains("append(left %")
            || body.contains("return true")
            || body.contains("return false")
        {
            score -= 7.0;
        }
    }

    let dependency_or_adapter_target = has_hint("local_adapter")
        || task_text.contains("json")
        || task_text.contains("csv")
        || task_text.contains("path")
        || task_text.contains("file")
        || task_text.contains("archive")
        || task_text.contains("dependency");
    if dependency_or_adapter_target {
        if body.contains("try:")
            && body.contains("except")
            && (body.contains("return []")
                || body.contains("return {}")
                || body.contains("return ''")
                || body.contains("return false"))
        {
            score += 2.5;
        }
        if task_text.contains("json") && body.contains("import json") {
            score += 1.0;
        }
        if task_text.contains("csv") && body.contains("import csv") {
            score += 1.0;
        }
        if (task_text.contains("path")
            || task_text.contains("file")
            || task_text.contains("archive"))
            && (body.contains("import os") || body.contains("import pathlib"))
        {
            score += 1.0;
        }
    }

    score
}

pub(super) const CHEAP_PREFILTER_FEATURE_DIM: usize = 21;
#[allow(dead_code)]
const CHEAP_PREFILTER_WEIGHTS: [f32; CHEAP_PREFILTER_FEATURE_DIM] = [
    1.0,   // beautiful_body_score
    1.0,   // body_transfer_score
    2.0,   // full_body_token_candidate
    1.0,   // compositional_token_candidate
    -4.0,  // expression_memory_fallback or native STS expression
    -6.0,  // template-like candidate
    2.0,   // visible argument contract passes
    -3.5,  // visible argument contract fails
    2.0,   // return-shape contract passes
    -4.0,  // return-shape contract fails
    1.2,   // required constructs present
    1.0,   // syntax constrained
    -8.0,  // syntax rejected
    -8.0,  // scaffold/natural language leakage
    1.8,   // contract-guided token decoder
    0.8,   // STS-conditioned candidate family
    1.0,   // bounded STS alignment score
    1.0,   // broad public floor recovery prefilter score
    1.0,   // private edge/full-body contract bridge score
    140.0, // private train broad semantic prototype
    100.0, // private residual v3 semantic adapter diagnostic
];

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CheapPrefilterStats {
    pub input_count: usize,
    pub output_count: usize,
    pub budget: usize,
    pub used_cuda: bool,
    pub feature_dim: usize,
}

#[allow(dead_code)]
fn cheap_candidate_prefilter_feature_row(
    task: &CodeTask,
    candidate: &CandidateExpression,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> [f32; CHEAP_PREFILTER_FEATURE_DIM] {
    let body = candidate.body.as_str();
    let lowered = body.to_lowercase();
    let hints = decoder_required_constructs(task);
    let visible_ok = visible_argument_contract_ok(task, body);
    let return_ok = return_shape_contract_ok(task, &lowered);
    let syntax_ok = syntax_constrained_body(body);
    let scaffold_or_leak =
        scaffold_placeholder_body(body) || natural_language_leakage_in_body(body);
    let sts_conditioned = candidate.mode.contains("sts_conditioned");
    let demote_sts = sts_conditioned && sts_decoder_control_demotes_sts_preference(sts_streams);
    [
        beautiful_body_score(task, body),
        body_transfer_score(task, body),
        candidate.full_body_token_candidate as u8 as f32,
        candidate.compositional_token_candidate as u8 as f32,
        (candidate.expression_memory_fallback || candidate.sts_candidate_expression_used) as u8
            as f32,
        template_like_candidate(candidate) as u8 as f32,
        visible_ok as u8 as f32,
        (!visible_ok) as u8 as f32,
        return_ok as u8 as f32,
        (!return_ok) as u8 as f32,
        required_construct_contract_ok_for_task(task, body, &hints) as u8 as f32,
        syntax_ok as u8 as f32,
        (!syntax_ok) as u8 as f32,
        scaffold_or_leak as u8 as f32,
        candidate.mode.contains("contract_guided_token_decoder") as u8 as f32,
        if demote_sts {
            -2.5
        } else {
            sts_conditioned as u8 as f32
        },
        if sts_conditioned && sts_streams.is_some() && !demote_sts {
            sts_skeleton_alignment_score(body, sts_streams).min(1.5)
        } else {
            0.0
        },
        broad_public_floor_recovery_prefilter_score(task, body, &candidate.mode),
        edge_full_body_contract_bridge_v2_prefilter_score(task, candidate),
        broad_private_train_prototype_candidate(candidate) as u8 as f32,
        private_residual_v3_semantic_adapter_candidate(candidate) as u8 as f32,
    ]
}

fn cheap_candidate_prefilter_scores_batch(
    task: &CodeTask,
    candidates: &[CandidateExpression],
    sts_streams: Option<&BTreeMap<String, String>>,
) -> (Vec<f32>, bool) {
    if candidates.is_empty() {
        return (Vec::new(), false);
    }
    #[cfg(feature = "cuda")]
    {
        if cuda_candidate_prefilter_enabled() {
            let feature_rows = candidates
                .iter()
                .flat_map(|candidate| {
                    let candidate_sts = candidate_conditioning_streams(candidate, sts_streams);
                    cheap_candidate_prefilter_feature_row(task, candidate, candidate_sts)
                })
                .collect::<Vec<_>>();
            if let Ok(features) = symliquid_core::tensor::Tensor::new(
                candidates.len(),
                CHEAP_PREFILTER_FEATURE_DIM,
                feature_rows.clone(),
            ) {
                if let Ok(scores) = symliquid_cuda::readout_cuda::weighted_feature_scores_cuda(
                    &features,
                    &CHEAP_PREFILTER_WEIGHTS,
                    0.0,
                ) {
                    return (scores, true);
                }
            }
        }
    }
    (
        candidates
            .iter()
            .map(|candidate| {
                let candidate_sts = candidate_conditioning_streams(candidate, sts_streams);
                cheap_candidate_prefilter_score(task, candidate, candidate_sts)
            })
            .collect(),
        false,
    )
}

pub(super) fn cheap_candidate_ranker_scores_batch(
    task: &CodeTask,
    candidates: &[CandidateExpression],
    sts_streams: Option<&BTreeMap<String, String>>,
) -> (Vec<f32>, bool) {
    cheap_candidate_prefilter_scores_batch(task, candidates, sts_streams)
}

#[cfg(feature = "cuda")]
fn cuda_candidate_prefilter_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_CUDA_PREFILTER")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

pub(in crate::code_lm_closure) fn cheap_prefilter_budget(
    candidate_limit: usize,
    default_multiplier: usize,
) -> usize {
    let multiplier = std::env::var("THESEUS_CODE_LM_CHEAP_PREFILTER_MULTIPLIER")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default_multiplier)
        .clamp(8, 64);
    let cap = std::env::var("THESEUS_CODE_LM_CHEAP_PREFILTER_CAP")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(96)
        .clamp(24, 256);
    candidate_limit
        .max(1)
        .saturating_mul(multiplier)
        .clamp(24, cap)
}

pub(in crate::code_lm_closure) fn pre_verification_prefilter_budget(
    candidate_limit: usize,
    low_latency: bool,
) -> usize {
    let default_multiplier = if low_latency { 8 } else { 16 };
    let multiplier = std::env::var("THESEUS_CODE_LM_PRE_VERIFICATION_PREFILTER_MULTIPLIER")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default_multiplier)
        .clamp(4, 64);
    let cap = std::env::var("THESEUS_CODE_LM_PRE_VERIFICATION_PREFILTER_CAP")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(if low_latency { 64 } else { 128 })
        .clamp(16, 512);
    candidate_limit
        .max(1)
        .saturating_mul(multiplier)
        .clamp(16, cap)
}

fn pre_verification_contract_prefilter_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_CONTRACT_TOKEN_PREFILTER")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

pub(super) fn prefilter_bodies_before_contract_verifier(
    task: &CodeTask,
    bodies: Vec<String>,
    sts_streams: Option<&BTreeMap<String, String>>,
    limit: usize,
    low_latency: bool,
    mode: &str,
) -> (Vec<String>, CheapPrefilterStats) {
    let input_count = bodies.len();
    if !pre_verification_contract_prefilter_enabled() || input_count <= limit.max(1) {
        return (
            bodies,
            CheapPrefilterStats {
                input_count,
                output_count: input_count,
                budget: input_count,
                used_cuda: false,
                feature_dim: 0,
            },
        );
    }
    let budget = pre_verification_prefilter_budget(limit, low_latency).min(input_count);
    let candidates = bodies
        .into_iter()
        .map(|body| CandidateExpression {
            expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
            body,
            mode: mode.to_string(),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: mode.contains("sts_conditioned"),
        })
        .collect::<Vec<_>>();
    let (scores, used_cuda) = cheap_candidate_ranker_scores_batch(task, &candidates, sts_streams);
    let mut ranked = candidates
        .into_iter()
        .enumerate()
        .map(|(idx, candidate)| {
            let score = scores.get(idx).copied().unwrap_or(0.0);
            (score, candidate)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.body.len().cmp(&b.1.body.len()))
            .then_with(|| a.1.body.cmp(&b.1.body))
    });
    let output = ranked
        .into_iter()
        .take(budget)
        .map(|(_, candidate)| candidate.body)
        .collect::<Vec<_>>();
    let output_count = output.len();
    (
        output,
        CheapPrefilterStats {
            input_count,
            output_count,
            budget,
            used_cuda,
            feature_dim: CHEAP_PREFILTER_FEATURE_DIM,
        },
    )
}

pub(super) fn verified_low_latency_contract_outputs_cached_with_stats(
    task: &CodeTask,
    pool: &[String],
    sts_streams: Option<&BTreeMap<String, String>>,
    limit: usize,
    verifier_cache: &mut HashMap<String, DecoderContractVerification>,
) -> (Vec<String>, CheapPrefilterStats) {
    let mut base_scored = Vec::new();
    let mut base_seen = HashSet::new();
    let mut normalized_pool = Vec::new();
    for body in pool {
        let canonical =
            canonicalize_task_candidate_body_aliases(task, &normalize_generated_body(body));
        let trimmed = canonical.trim().to_string();
        if trimmed.is_empty() || !base_seen.insert(trimmed.clone()) {
            continue;
        }
        normalized_pool.push(trimmed);
    }
    let mode = if sts_streams.is_some() {
        "rust_code_lm_sts_conditioned_contract_guided_token_decoder"
    } else {
        "rust_code_lm_contract_guided_token_decoder"
    };
    let (prefiltered_pool, stats) = prefilter_bodies_before_contract_verifier(
        task,
        normalized_pool,
        sts_streams,
        limit,
        true,
        mode,
    );
    for trimmed in prefiltered_pool {
        let verification =
            cached_decoder_contract_verification(task, &trimmed, sts_streams, verifier_cache);
        if !token_contract_candidate_body_ok_with_verification(task, &trimmed, &verification) {
            continue;
        }
        let score = contract_guided_token_candidate_score_with_verification(
            task,
            &trimmed,
            sts_streams,
            &verification,
        );
        base_scored.push((score, trimmed));
    }
    base_scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.len().cmp(&b.1.len()))
            .then_with(|| a.1.cmp(&b.1))
    });
    let mut out = Vec::new();
    let mut emitted = HashSet::new();
    for (_score, body) in base_scored {
        if emitted.insert(body.clone()) {
            out.push(body);
        }
        if out.len() >= limit {
            break;
        }
    }
    (out, stats)
}

pub(super) fn truncate_by_cheap_prefilter(
    task: &CodeTask,
    candidates: &mut Vec<CandidateExpression>,
    sts_streams: Option<&BTreeMap<String, String>>,
    budget: usize,
) -> CheapPrefilterStats {
    let input_count = candidates.len();
    if candidates.len() <= budget {
        return CheapPrefilterStats {
            input_count,
            output_count: candidates.len(),
            budget,
            used_cuda: false,
            feature_dim: CHEAP_PREFILTER_FEATURE_DIM,
        };
    }
    let (scores, used_cuda) = cheap_candidate_prefilter_scores_batch(task, candidates, sts_streams);
    let mut scored = candidates.drain(..).zip(scores).collect::<Vec<_>>();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.mode.cmp(&b.0.mode))
            .then_with(|| a.0.body.cmp(&b.0.body))
    });
    let mut selected_indices = BTreeSet::new();
    let mut selected_modes = HashSet::new();
    if let Some((idx, _)) = scored
        .iter()
        .enumerate()
        .find(|(_, (candidate, _score))| same_seed_non_sts_comparator_candidate(candidate))
    {
        selected_indices.insert(idx);
    }
    for (idx, (candidate, _score)) in scored.iter().enumerate() {
        if selected_indices.len() >= budget {
            break;
        }
        if selected_modes.insert(candidate.mode.clone()) {
            selected_indices.insert(idx);
        }
    }
    for idx in 0..scored.len() {
        if selected_indices.len() >= budget {
            break;
        }
        selected_indices.insert(idx);
    }
    candidates.extend(
        scored
            .into_iter()
            .enumerate()
            .filter_map(|(idx, (candidate, _score))| {
                if selected_indices.contains(&idx) {
                    Some(candidate)
                } else {
                    None
                }
            }),
    );
    CheapPrefilterStats {
        input_count,
        output_count: candidates.len(),
        budget,
        used_cuda,
        feature_dim: CHEAP_PREFILTER_FEATURE_DIM,
    }
}
