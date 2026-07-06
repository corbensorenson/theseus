use super::*;

pub(super) struct CandidateTaskRows {
    pub rows: Vec<Value>,
    pub accepted_count: usize,
    pub rejected_count: usize,
    pub rejection_counts: BTreeMap<String, usize>,
}

pub(super) fn candidate_rows_for_task(
    task: &CodeTask,
    expression_bank: &[ExpressionBankItem],
    body_prototypes: &BodyPrototypeModel,
    body_ngram: &BodyNgramModel,
    state_sequence_decoder: &StateSequenceDecoder,
    symliquid_state_decoder: &SymLiquidStateDecoder,
    readout: &LinearReadout,
    vocab: &Vocab,
    checkpoint_id: &str,
    seed: u64,
    candidates_per_task: usize,
    phase: &str,
    trained: bool,
    task_sts: Option<&BTreeMap<String, String>>,
    transformer_hybrid_survival_candidates: Option<&Vec<CandidateExpression>>,
    precomputed_beams: Option<&Vec<String>>,
    precomputed_state_sequence: Option<&Vec<String>>,
    precomputed_symliquid_state: Option<&Vec<String>>,
    precomputed_state_sequence_no_sts: Option<&Vec<String>>,
    precomputed_symliquid_state_no_sts: Option<&Vec<String>>,
    batched_beam_cache_ms: u128,
    batched_beam_cache_precompute_enabled: bool,
    batched_state_sequence_cache_ms: u128,
    batched_state_sequence_cache_precompute_enabled: bool,
    batched_symliquid_state_cache_ms: u128,
    batched_symliquid_state_cache_precompute_enabled: bool,
    batched_state_sequence_no_sts_cache_ms: u128,
    batched_state_sequence_no_sts_cache_precompute_enabled: bool,
    batched_symliquid_state_no_sts_cache_ms: u128,
    batched_symliquid_state_no_sts_cache_precompute_enabled: bool,
    parallel_shared_decoder_precompute_enabled: bool,
    shared_decoder_precompute_wall_ms: u128,
    record_shared_decoder_precompute_timing: bool,
    fanout_worker_id: usize,
    persistent_worker_pool_enabled: bool,
) -> CandidateTaskRows {
    let _fanout_context =
        candidate_fanout_thread_context_guard(fanout_worker_id, persistent_worker_pool_enabled);
    let task_started = Instant::now();
    let mut stage_started = Instant::now();
    let mut task_timing_ms: BTreeMap<String, u128> = BTreeMap::new();
    task_timing_ms.insert(
        "batched_beam_cache_precompute_shared_ms".to_string(),
        batched_beam_cache_ms,
    );
    task_timing_ms.insert(
        "batched_beam_cache_precompute_enabled".to_string(),
        if batched_beam_cache_precompute_enabled {
            1
        } else {
            0
        },
    );
    task_timing_ms.insert(
        "batched_state_sequence_cache_precompute_shared_ms".to_string(),
        batched_state_sequence_cache_ms,
    );
    task_timing_ms.insert(
        "batched_state_sequence_cache_precompute_enabled".to_string(),
        if batched_state_sequence_cache_precompute_enabled {
            1
        } else {
            0
        },
    );
    task_timing_ms.insert(
        "batched_symliquid_state_cache_precompute_shared_ms".to_string(),
        batched_symliquid_state_cache_ms,
    );
    task_timing_ms.insert(
        "batched_symliquid_state_cache_precompute_enabled".to_string(),
        if batched_symliquid_state_cache_precompute_enabled {
            1
        } else {
            0
        },
    );
    task_timing_ms.insert(
        "batched_state_sequence_no_sts_cache_precompute_shared_ms".to_string(),
        batched_state_sequence_no_sts_cache_ms,
    );
    task_timing_ms.insert(
        "batched_state_sequence_no_sts_cache_precompute_enabled".to_string(),
        if batched_state_sequence_no_sts_cache_precompute_enabled {
            1
        } else {
            0
        },
    );
    task_timing_ms.insert(
        "batched_symliquid_state_no_sts_cache_precompute_shared_ms".to_string(),
        batched_symliquid_state_no_sts_cache_ms,
    );
    task_timing_ms.insert(
        "batched_symliquid_state_no_sts_cache_precompute_enabled".to_string(),
        if batched_symliquid_state_no_sts_cache_precompute_enabled {
            1
        } else {
            0
        },
    );
    task_timing_ms.insert(
        "batched_state_sequence_no_sts_cache_reused_from_conditioned".to_string(),
        (task_sts.is_none()
            && !batched_state_sequence_no_sts_cache_precompute_enabled
            && precomputed_state_sequence_no_sts.is_some()) as u128,
    );
    task_timing_ms.insert(
        "batched_symliquid_state_no_sts_cache_reused_from_conditioned".to_string(),
        (task_sts.is_none()
            && !batched_symliquid_state_no_sts_cache_precompute_enabled
            && precomputed_symliquid_state_no_sts.is_some()) as u128,
    );
    task_timing_ms.insert(
        "parallel_shared_decoder_precompute_enabled".to_string(),
        (parallel_shared_decoder_precompute_enabled && record_shared_decoder_precompute_timing)
            as u128,
    );
    task_timing_ms.insert(
        "shared_decoder_precompute_wall_ms".to_string(),
        if record_shared_decoder_precompute_timing {
            shared_decoder_precompute_wall_ms
        } else {
            0
        },
    );
    task_timing_ms.insert(
        "candidate_task_fanout_worker_id".to_string(),
        fanout_worker_id as u128,
    );
    task_timing_ms.insert(
        "candidate_task_persistent_worker_pool_enabled".to_string(),
        persistent_worker_pool_enabled as u128,
    );
    task_timing_ms.insert(
        "candidate_task_context_worker_id".to_string(),
        current_candidate_fanout_worker_id() as u128,
    );
    task_timing_ms.insert(
        "candidate_task_nested_branch_parallelism_suppressed".to_string(),
        nested_branch_parallelism_suppressed_for_current_task() as u128,
    );
    let mut expression_branch_timing_ms = BTreeMap::new();
    let survival_lane_only = trained
        && transformer_hybrid_survival_candidates.is_some()
        && transformer_hybrid_survival_lane_only_enabled();
    task_timing_ms.insert(
        "transformer_hybrid_survival_lane_only_enabled".to_string(),
        survival_lane_only as u128,
    );
    let mut expressions = if trained && !survival_lane_only {
        let (expressions, branch_timing_ms) = candidate_expressions_with_timing(
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
            precomputed_beams,
            precomputed_state_sequence,
            precomputed_symliquid_state,
            precomputed_state_sequence_no_sts,
            precomputed_symliquid_state_no_sts,
        );
        expression_branch_timing_ms = branch_timing_ms;
        expressions
    } else if trained {
        Vec::new()
    } else {
        baseline_expressions(task, expression_bank, seed, candidates_per_task * 2)
            .into_iter()
            .map(|expr| CandidateExpression {
                body: body_from_expression(&expr),
                expr,
                mode: "rust_code_lm_private_frequency_baseline".to_string(),
                compositional_token_candidate: false,
                full_body_token_candidate: false,
                expression_memory_fallback: true,
                sts_candidate_expression_used: false,
            })
            .collect::<Vec<_>>()
    };
    let mut transformer_hybrid_survival_lane_added = 0usize;
    if trained {
        if let Some(imported) = transformer_hybrid_survival_candidates {
            for candidate in imported.iter().take(candidates_per_task.max(1)) {
                expressions.push(candidate.clone());
                transformer_hybrid_survival_lane_added += 1;
            }
        }
    }
    task_timing_ms.insert(
        "transformer_hybrid_survival_lane_imported_count".to_string(),
        transformer_hybrid_survival_lane_added as u128,
    );
    let raw_candidate_count = expressions.len();
    record_candidate_timing(
        &mut task_timing_ms,
        "candidate_expression_generation_ms",
        &mut stage_started,
    );
    for (phase, elapsed_ms) in expression_branch_timing_ms {
        task_timing_ms.insert(
            format!("candidate_expression_branch_{phase}_ms"),
            elapsed_ms,
        );
    }
    let mut variant_cache = CandidateVariantCache::default();
    let mut normalized_candidates = expressions
        .drain(..)
        .map(|candidate| normalize_candidate_for_task_cached(task, candidate, &mut variant_cache))
        .collect::<Vec<_>>();
    let normalized_candidate_count_before_prefilter = normalized_candidates.len();
    record_candidate_timing(
        &mut task_timing_ms,
        "candidate_normalization_ms",
        &mut stage_started,
    );
    task_timing_ms.insert(
        "candidate_variant_cache_entries".to_string(),
        variant_cache.entries() as u128,
    );
    task_timing_ms.insert(
        "candidate_variant_cache_hits".to_string(),
        variant_cache.hits() as u128,
    );
    let low_latency_prefilter = low_latency_candidate_fanout_enabled() && candidates_per_task <= 8;
    let budget = cheap_prefilter_budget(
        candidates_per_task,
        if low_latency_prefilter { 16 } else { 48 },
    );
    let cheap_prefilter_stats =
        truncate_by_cheap_prefilter(task, &mut normalized_candidates, task_sts, budget);
    let normalized_candidate_count_after_prefilter = normalized_candidates.len();
    record_candidate_timing(
        &mut task_timing_ms,
        "cheap_prefilter_ms",
        &mut stage_started,
    );
    task_timing_ms.insert(
        "cheap_prefilter_cuda_ranker_used".to_string(),
        cheap_prefilter_stats.used_cuda as u128,
    );
    task_timing_ms.insert(
        "cheap_prefilter_input_count".to_string(),
        cheap_prefilter_stats.input_count as u128,
    );
    task_timing_ms.insert(
        "cheap_prefilter_output_count".to_string(),
        cheap_prefilter_stats.output_count as u128,
    );
    task_timing_ms.insert(
        "cheap_prefilter_budget".to_string(),
        cheap_prefilter_stats.budget as u128,
    );
    task_timing_ms.insert(
        "cheap_prefilter_feature_dim".to_string(),
        cheap_prefilter_stats.feature_dim as u128,
    );
    let pre_verification_budget =
        pre_verification_prefilter_budget(candidates_per_task, low_latency_prefilter);
    let pre_verification_stats = truncate_by_cheap_prefilter(
        task,
        &mut normalized_candidates,
        task_sts,
        pre_verification_budget,
    );
    record_candidate_timing(
        &mut task_timing_ms,
        "pre_verification_gpu_prefilter_ms",
        &mut stage_started,
    );
    task_timing_ms.insert(
        "pre_verification_gpu_prefilter_cuda_ranker_used".to_string(),
        pre_verification_stats.used_cuda as u128,
    );
    task_timing_ms.insert(
        "pre_verification_gpu_prefilter_input_count".to_string(),
        pre_verification_stats.input_count as u128,
    );
    task_timing_ms.insert(
        "pre_verification_gpu_prefilter_output_count".to_string(),
        pre_verification_stats.output_count as u128,
    );
    task_timing_ms.insert(
        "pre_verification_gpu_prefilter_budget".to_string(),
        pre_verification_stats.budget as u128,
    );
    let interface_floor_candidates =
        interface_floor_candidate_expressions(task, task_sts.is_some());
    let mut interface_floor_added = 0usize;
    for candidate in interface_floor_candidates {
        let normalized = normalize_candidate_for_task_cached(task, candidate, &mut variant_cache);
        if normalized_candidates
            .iter()
            .any(|existing| existing.body.trim() == normalized.body.trim())
        {
            continue;
        }
        normalized_candidates.push(normalized);
        interface_floor_added += 1;
    }
    task_timing_ms.insert(
        "interface_floor_candidate_added".to_string(),
        interface_floor_added as u128,
    );
    let edge_contract_v3_strict_novel_token_candidates =
        edge_contract_v3_strict_novel_token_candidates(task, task_sts.is_some());
    let mut edge_contract_v3_strict_novel_token_added = 0usize;
    for candidate in edge_contract_v3_strict_novel_token_candidates {
        let normalized = normalize_candidate_for_task_cached(task, candidate, &mut variant_cache);
        normalized_candidates.push(normalized);
        edge_contract_v3_strict_novel_token_added += 1;
    }
    task_timing_ms.insert(
        "edge_contract_v3_strict_novel_token_added".to_string(),
        edge_contract_v3_strict_novel_token_added as u128,
    );
    task_timing_ms.insert(
        "edge_contract_v3_strict_novel_token_normalized_count".to_string(),
        normalized_candidates
            .iter()
            .filter(|candidate| edge_contract_v3_strict_novel_token_candidate(candidate))
            .count() as u128,
    );
    let private_residual_v3_structural_token_candidates =
        private_residual_v3_train_induced_structural_token_candidates(task, task_sts.is_some());
    let mut private_residual_v3_structural_token_added = 0usize;
    for candidate in private_residual_v3_structural_token_candidates {
        let normalized = normalize_candidate_for_task_cached(task, candidate, &mut variant_cache);
        normalized_candidates.push(normalized);
        private_residual_v3_structural_token_added += 1;
    }
    task_timing_ms.insert(
        "private_residual_v3_structural_token_added".to_string(),
        private_residual_v3_structural_token_added as u128,
    );
    task_timing_ms.insert(
        "private_residual_v3_structural_token_normalized_count".to_string(),
        normalized_candidates
            .iter()
            .filter(|candidate| {
                private_residual_v3_train_induced_structural_token_candidate(candidate)
            })
            .count() as u128,
    );
    let private_residual_v3_candidates =
        private_residual_v3_semantic_adapter_candidates(task, task_sts.is_some());
    let mut private_residual_v3_added = 0usize;
    for candidate in private_residual_v3_candidates {
        let normalized = normalize_candidate_for_task_cached(task, candidate, &mut variant_cache);
        normalized_candidates.push(normalized);
        private_residual_v3_added += 1;
    }
    task_timing_ms.insert(
        "private_residual_v3_semantic_adapter_added".to_string(),
        private_residual_v3_added as u128,
    );
    task_timing_ms.insert(
        "private_residual_v3_semantic_adapter_normalized_count".to_string(),
        normalized_candidates
            .iter()
            .filter(|candidate| private_residual_v3_semantic_adapter_candidate(candidate))
            .count() as u128,
    );
    let broad_private_train_composition_candidates =
        broad_private_train_composition_token_candidates(task, task_sts.is_some());
    let mut broad_private_train_composition_added = 0usize;
    for candidate in broad_private_train_composition_candidates {
        let normalized = normalize_candidate_for_task_cached(task, candidate, &mut variant_cache);
        normalized_candidates.push(normalized);
        broad_private_train_composition_added += 1;
    }
    task_timing_ms.insert(
        "broad_private_train_composition_token_added".to_string(),
        broad_private_train_composition_added as u128,
    );
    task_timing_ms.insert(
        "broad_private_train_composition_token_normalized_count".to_string(),
        normalized_candidates
            .iter()
            .filter(|candidate| {
                broad_private_train_token_candidate(candidate)
                    && candidate.mode.contains("novel_composition_v1")
            })
            .count() as u128,
    );
    let broad_private_train_token_candidates =
        broad_private_train_token_candidates(task, task_sts.is_some());
    let mut broad_private_train_token_added = 0usize;
    for candidate in broad_private_train_token_candidates {
        let normalized = normalize_candidate_for_task_cached(task, candidate, &mut variant_cache);
        normalized_candidates.push(normalized);
        broad_private_train_token_added += 1;
    }
    task_timing_ms.insert(
        "broad_private_train_token_added".to_string(),
        broad_private_train_token_added as u128,
    );
    task_timing_ms.insert(
        "broad_private_train_token_normalized_count".to_string(),
        normalized_candidates
            .iter()
            .filter(|candidate| broad_private_train_token_candidate(candidate))
            .count() as u128,
    );
    let broad_private_train_prototype_candidates =
        broad_private_train_prototype_candidates(task, task_sts.is_some());
    let mut broad_private_train_prototype_added = 0usize;
    for candidate in broad_private_train_prototype_candidates {
        let normalized = normalize_candidate_for_task_cached(task, candidate, &mut variant_cache);
        normalized_candidates.push(normalized);
        broad_private_train_prototype_added += 1;
    }
    task_timing_ms.insert(
        "broad_private_train_prototype_added".to_string(),
        broad_private_train_prototype_added as u128,
    );
    task_timing_ms.insert(
        "broad_private_train_prototype_normalized_count".to_string(),
        normalized_candidates
            .iter()
            .filter(|candidate| broad_private_train_prototype_candidate(candidate))
            .count() as u128,
    );
    let broad_private_generalization_candidates =
        broad_private_generalization_semantic_adapter_candidates(task, task_sts.is_some());
    let mut broad_private_generalization_added = 0usize;
    for candidate in broad_private_generalization_candidates {
        let normalized = normalize_candidate_for_task_cached(task, candidate, &mut variant_cache);
        normalized_candidates.push(normalized);
        broad_private_generalization_added += 1;
    }
    task_timing_ms.insert(
        "broad_private_generalization_semantic_adapter_added".to_string(),
        broad_private_generalization_added as u128,
    );
    task_timing_ms.insert(
        "broad_private_generalization_semantic_adapter_normalized_count".to_string(),
        normalized_candidates
            .iter()
            .filter(|candidate| broad_private_generalization_semantic_adapter_candidate(candidate))
            .count() as u128,
    );
    let (rank_scores, ranker_used_cuda) =
        cheap_candidate_ranker_scores_batch(task, &normalized_candidates, task_sts);
    task_timing_ms.insert(
        "rank_score_output_count".to_string(),
        rank_scores.len() as u128,
    );
    let mut verification_cache = CandidateVerificationCache::default();
    let mut ranked_candidates = normalized_candidates
        .into_iter()
        .zip(rank_scores)
        .map(|(candidate, score)| {
            let tie_breaker = stable_hash_u64(&format!(
                "candidate-floor-v2:{}:{}:{}",
                seed, task.task_id, candidate.body
            ));
            (candidate, score, tie_breaker)
        })
        .collect::<Vec<_>>();
    let ranked_candidate_count = ranked_candidates.len();
    task_timing_ms.insert(
        "private_residual_v3_structural_token_ranked_count".to_string(),
        ranked_candidates
            .iter()
            .filter(|(candidate, _score, _tie_breaker)| {
                private_residual_v3_train_induced_structural_token_candidate(candidate)
            })
            .count() as u128,
    );
    task_timing_ms.insert(
        "private_residual_v3_semantic_adapter_ranked_count".to_string(),
        ranked_candidates
            .iter()
            .filter(|(candidate, _score, _tie_breaker)| {
                private_residual_v3_semantic_adapter_candidate(candidate)
            })
            .count() as u128,
    );
    task_timing_ms.insert(
        "edge_contract_v3_strict_novel_token_ranked_count".to_string(),
        ranked_candidates
            .iter()
            .filter(|(candidate, _score, _tie_breaker)| {
                edge_contract_v3_strict_novel_token_candidate(candidate)
            })
            .count() as u128,
    );
    task_timing_ms.insert(
        "broad_private_train_token_ranked_count".to_string(),
        ranked_candidates
            .iter()
            .filter(|(candidate, _score, _tie_breaker)| {
                broad_private_train_token_candidate(candidate)
            })
            .count() as u128,
    );
    task_timing_ms.insert(
        "broad_private_train_prototype_ranked_count".to_string(),
        ranked_candidates
            .iter()
            .filter(|(candidate, _score, _tie_breaker)| {
                broad_private_train_prototype_candidate(candidate)
            })
            .count() as u128,
    );
    task_timing_ms.insert(
        "broad_private_generalization_semantic_adapter_ranked_count".to_string(),
        ranked_candidates
            .iter()
            .filter(|(candidate, _score, _tie_breaker)| {
                broad_private_generalization_semantic_adapter_candidate(candidate)
            })
            .count() as u128,
    );
    ranked_candidates.sort_by(|a, b| {
        let a_transformer_hybrid = transformer_hybrid_survival_lane_candidate(&a.0);
        let b_transformer_hybrid = transformer_hybrid_survival_lane_candidate(&b.0);
        let a_broad_train_token = broad_private_train_token_candidate(&a.0);
        let b_broad_train_token = broad_private_train_token_candidate(&b.0);
        let a_broad_train_composition =
            a_broad_train_token && a.0.mode.contains("novel_composition_v1");
        let b_broad_train_composition =
            b_broad_train_token && b.0.mode.contains("novel_composition_v1");
        let a_broad_train_novel = a_broad_train_token && a.0.mode.contains("train_novel_body_v1");
        let b_broad_train_novel = b_broad_train_token && b.0.mode.contains("train_novel_body_v1");
        let a_broad_train_prototype = broad_private_train_prototype_candidate(&a.0);
        let b_broad_train_prototype = broad_private_train_prototype_candidate(&b.0);
        let a_edge_v3_strict = edge_contract_v3_strict_novel_token_candidate(&a.0);
        let b_edge_v3_strict = edge_contract_v3_strict_novel_token_candidate(&b.0);
        let a_private_v3_structural =
            private_residual_v3_train_induced_structural_token_candidate(&a.0);
        let b_private_v3_structural =
            private_residual_v3_train_induced_structural_token_candidate(&b.0);
        let a_private_v3 = private_residual_v3_semantic_adapter_candidate(&a.0);
        let b_private_v3 = private_residual_v3_semantic_adapter_candidate(&b.0);
        let a_broad_private = broad_private_generalization_semantic_adapter_candidate(&a.0);
        let b_broad_private = broad_private_generalization_semantic_adapter_candidate(&b.0);
        let a_edge_category_context = task.category.starts_with("edge_v3_")
            && a.0
                .mode
                .contains("contract_guided_token_decoder_category_context");
        let b_edge_category_context = task.category.starts_with("edge_v3_")
            && b.0
                .mode
                .contains("contract_guided_token_decoder_category_context");
        b_transformer_hybrid
            .cmp(&a_transformer_hybrid)
            .then_with(|| {
                if a_transformer_hybrid && b_transformer_hybrid {
                    transformer_hybrid_survival_lane_rank(&a.0)
                        .cmp(&transformer_hybrid_survival_lane_rank(&b.0))
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then_with(|| b_broad_train_composition.cmp(&a_broad_train_composition))
            .then_with(|| b_broad_train_novel.cmp(&a_broad_train_novel))
            .then_with(|| b_broad_train_token.cmp(&a_broad_train_token))
            .then_with(|| b_broad_train_prototype.cmp(&a_broad_train_prototype))
            .then_with(|| b_edge_v3_strict.cmp(&a_edge_v3_strict))
            .then_with(|| b_private_v3_structural.cmp(&a_private_v3_structural))
            .then_with(|| b_broad_private.cmp(&a_broad_private))
            .then_with(|| b_private_v3.cmp(&a_private_v3))
            .then_with(|| b_edge_category_context.cmp(&a_edge_category_context))
            .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.2.cmp(&b.2))
    });
    record_candidate_timing(
        &mut task_timing_ms,
        "rank_score_and_sort_ms",
        &mut stage_started,
    );
    task_timing_ms.insert(
        "rank_score_gpu_prefilter_cuda_ranker_used".to_string(),
        ranker_used_cuda as u128,
    );
    task_timing_ms.insert(
        "rank_score_gpu_prefilter_input_count".to_string(),
        ranked_candidate_count as u128,
    );
    task_timing_ms.insert(
        "transformer_hybrid_survival_lane_ranked_count".to_string(),
        ranked_candidates
            .iter()
            .filter(|(candidate, _score, _tie_breaker)| {
                transformer_hybrid_survival_lane_candidate(candidate)
            })
            .count() as u128,
    );
    let mut rows = Vec::new();
    let mut emitted_for_task = 0usize;
    let mut emitted_primary_for_task = 0usize;
    let mut emitted_same_seed_non_sts_comparator = false;
    let mut emitted_contract_guided_token_inventory = false;
    let mut seen = HashSet::new();
    let mut rejection_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut rejection_samples: Vec<Value> = Vec::new();
    let template_free_student_candidates = template_free_student_candidates_enabled();
    let mut rejected_for_task = 0usize;
    let primary_emission_limit =
        private_residual_low_latency_primary_emission_limit(task, candidates_per_task);
    task_timing_ms.insert(
        "private_residual_primary_emission_limit".to_string(),
        primary_emission_limit as u128,
    );
    for (mut candidate, _score, _tie_breaker) in ranked_candidates {
        let same_seed_non_sts_comparator = same_seed_non_sts_comparator_candidate(&candidate);
        if !same_seed_non_sts_comparator
            && edge_contract_v3_demote_to_body_memory_replay(task, &candidate)
        {
            candidate.mode = format!(
                "rust_code_lm_edge_contract_v3_body_memory_replay_decoder_v1::{}",
                candidate.mode
            );
        }
        let contract_guided_token_inventory = contract_guided_token_inventory_candidate(&candidate);
        if same_seed_non_sts_comparator && emitted_same_seed_non_sts_comparator {
            continue;
        }
        if contract_guided_token_inventory && emitted_contract_guided_token_inventory {
            continue;
        }
        let preserve_contract_inventory = task_sts.is_some()
            && contract_guided_token_inventory
            && !emitted_contract_guided_token_inventory;
        if !same_seed_non_sts_comparator && emitted_primary_for_task >= primary_emission_limit {
            if task_sts.is_some()
                && (!emitted_same_seed_non_sts_comparator || preserve_contract_inventory)
            {
                if !preserve_contract_inventory {
                    continue;
                }
            } else if task_sts.is_some() && !emitted_same_seed_non_sts_comparator {
                continue;
            } else {
                break;
            }
        }
        let candidate_sts = candidate_conditioning_streams(&candidate, task_sts);
        let candidate_template_like = template_like_candidate(&candidate);
        let transformer_hybrid_survival_lane_stage =
            transformer_hybrid_survival_lane_candidate(&candidate);
        let transformer_hybrid_import_syntax_release =
            transformer_hybrid_survival_lane_stage
                && transformer_hybrid_import_body_ok(&candidate.body)
                && !candidate_template_like;
        let rejection = if template_free_student_candidates && candidate_template_like {
            Some("template_like_candidate".to_string())
        } else if !candidate_body_admissible_cached(
            task,
            &candidate,
            candidate_sts,
            &mut verification_cache,
        ) && !transformer_hybrid_import_syntax_release
        {
            Some(
                candidate_rejection_reason_cached(
                    task,
                    &candidate,
                    false,
                    candidate_sts,
                    &mut verification_cache,
                )
                .unwrap_or("candidate_body_not_admissible")
                .to_string(),
            )
        } else {
            let duplicate_key = candidate_duplicate_key(task, &candidate);
            if !seen.insert(duplicate_key) {
                Some("duplicate_body".to_string())
            } else {
                None
            }
        };
        if let Some(reason) = rejection {
            rejected_for_task = rejected_for_task.saturating_add(1);
            *rejection_counts.entry(reason.clone()).or_insert(0) += 1;
            if rejection_samples.len() < 8 {
                rejection_samples.push(json!({
                    "reason": reason,
                    "mode": candidate.mode,
                    "body_preview": preview_body(&candidate.body),
                }));
            }
            continue;
        }
        let rank = emitted_primary_for_task + 1;
        let code = render_candidate_body(task, &candidate.body);
        let beautiful_code_score = beautiful_body_score(task, &candidate.body);
        let placeholder_scaffold_body = scaffold_placeholder_body(&candidate.body);
        let sts_conditioned = trained && candidate_sts.is_some();
        let candidate_generation_mode = candidate.mode.clone();
        let sts_streams_seen = candidate_sts
            .map(|streams| streams.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let generation_inputs = if sts_conditioned {
            vec![
                "visible_prompt",
                "entry_point",
                "decoder_v2_semantic_plan",
                "visible_signature_return_shape_type_family",
                "trained_private_curriculum_code_tokens",
                "trained_private_sts_style_stream_tokens",
                "native_sts_generated_streams",
                "sts_conditioned_skeleton_choice",
            ]
        } else {
            vec![
                "visible_prompt",
                "entry_point",
                "decoder_v2_semantic_plan",
                "visible_signature_return_shape_type_family",
                "trained_private_curriculum_code_tokens",
                "trained_private_sts_style_stream_tokens",
            ]
        };
        let mut generation_inputs = generation_inputs
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let eligible_receiver_inventory_stage =
            eligible_receiver_inventory_router_candidate(&candidate);
        let eligible_receiver_inventory_policy = eligible_receiver_inventory_policy_summary(task);
        if eligible_receiver_inventory_stage {
            generation_inputs.extend(eligible_receiver_inventory_generation_inputs(task));
        }
        let private_to_public_receiver_inventory_bridge_stage =
            private_to_public_receiver_inventory_bridge_candidate(&candidate);
        let private_to_public_receiver_inventory_bridge_policy =
            private_to_public_receiver_inventory_bridge_policy_summary(task);
        if private_to_public_receiver_inventory_bridge_stage {
            generation_inputs
                .extend(private_to_public_receiver_inventory_bridge_generation_inputs(task));
        }
        let broad_transfer_residual_stage = broad_transfer_residual_candidate(&candidate);
        let broad_transfer_residual_policy = broad_transfer_residual_policy_summary(task);
        if broad_transfer_residual_stage {
            generation_inputs.extend(broad_transfer_residual_generation_inputs(task));
        }
        if transformer_hybrid_survival_lane_stage {
            generation_inputs = vec![
                "visible_prompt".to_string(),
                "entry_point".to_string(),
                "visible_signature".to_string(),
            ];
            generation_inputs.push("trainable_transformer_hybrid_action_selector".to_string());
            generation_inputs.push("grammar_safe_renderer".to_string());
            generation_inputs.push("private_train_only_no_public_tests_or_solutions".to_string());
            generation_inputs
                .push("heldout_solutions_and_tests_not_used_for_generation".to_string());
            generation_inputs.push("canonical_fanout_import_rank_verify_emit".to_string());
        }
        let edge_exec_repair_stage = candidate.mode.contains("edge_exec_repair");
        if edge_exec_repair_stage {
            generation_inputs
                .push("private_source_agnostic_edge_execution_feedback_repair".to_string());
        }
        let execution_shape_skeleton_stage =
            candidate.mode.contains("execution_shape_skeleton_decoder");
        if execution_shape_skeleton_stage {
            generation_inputs.push("sts_conditioned_execution_shape_skeleton_choice".to_string());
            generation_inputs
                .push("private_execution_shape_io_library_control_flow_contract".to_string());
        }
        let sts_causal_skeleton_stage = candidate.mode.contains("sts_causal_skeleton_decoder");
        if sts_causal_skeleton_stage {
            generation_inputs.push("sts_residual_stream_to_skeleton_choice".to_string());
            generation_inputs.push("private_residual_family_control_flow_contract".to_string());
        }
        let contract_transduced_stage =
            candidate.mode.contains("contract_transduced_token_decoder");
        if contract_transduced_stage {
            generation_inputs.push("private_body_prototype_contract_transduction".to_string());
            generation_inputs
                .push("diagnostic_contract_repair_not_next_token_promotion".to_string());
        }
        let causal_contract_skeleton_stage =
            candidate.mode.contains("causal_contract_skeleton_decoder");
        if causal_contract_skeleton_stage {
            generation_inputs.push("signature_argument_roles_return_contract_first".to_string());
            generation_inputs
                .push("verifier_guided_skeleton_choice_before_token_decode".to_string());
            generation_inputs.push("symliquid_sts_route_bias_to_candidate_order".to_string());
        }
        let contract_guided_token_stage = candidate.mode.contains("contract_guided_token_decoder");
        if contract_guided_token_stage {
            generation_inputs.push("signature_argument_roles_return_contract_first".to_string());
            generation_inputs.push("verifier_guided_token_candidate_selection".to_string());
            generation_inputs.push("learned_next_token_body_inventory_no_templates".to_string());
            if sts_conditioned {
                generation_inputs.push("sts_conditioned_token_candidate_bias".to_string());
            }
        }
        let interface_floor_stage = candidate.mode.contains("interface_floor_token_decoder");
        if interface_floor_stage {
            generation_inputs.push("interface_floor_semantic_skeleton".to_string());
            generation_inputs
                .push("diagnostic_structural_floor_not_promotion_evidence".to_string());
        }
        let broad_private_train_token_stage = broad_private_train_token_candidate(&candidate);
        if broad_private_train_token_stage {
            generation_inputs
                .push("private_train_broad_generalization_token_transitions".to_string());
            generation_inputs.push("semantic_family_token_decoder".to_string());
            generation_inputs.push("private_train_only_no_public_tests_or_solutions".to_string());
            generation_inputs
                .push("heldout_solutions_and_tests_not_used_for_generation".to_string());
            generation_inputs.push("private_train_distilled_token_decoder_evidence".to_string());
        }
        let broad_private_train_prototype_stage =
            broad_private_train_prototype_candidate(&candidate);
        if broad_private_train_prototype_stage {
            generation_inputs
                .push("private_train_broad_generalization_solution_body_prototype".to_string());
            generation_inputs.push("semantic_family_prototype_index".to_string());
            generation_inputs.push("private_train_only_no_public_tests_or_solutions".to_string());
            generation_inputs
                .push("heldout_solutions_and_tests_not_used_for_generation".to_string());
            generation_inputs.push("prototype_transfer_non_public_promotion_evidence".to_string());
        }
        let edge_contract_v3_body_memory_replay_stage =
            edge_contract_v3_body_memory_replay_candidate(&candidate);
        if edge_contract_v3_body_memory_replay_stage {
            generation_inputs.push("edge_contract_v3_private_train_body_memory_replay".to_string());
            generation_inputs
                .push("diagnostic_replay_excluded_from_learned_token_evidence".to_string());
        }
        let edge_contract_v3_strict_novel_token_stage =
            edge_contract_v3_strict_novel_token_candidate(&candidate);
        if edge_contract_v3_strict_novel_token_stage {
            generation_inputs.push("edge_contract_v3_private_public_transfer_contract".to_string());
            generation_inputs.push("strict_novel_token_body_variant_decoder".to_string());
            generation_inputs
                .push("private_generated_curriculum_only_no_public_tests_or_solutions".to_string());
            generation_inputs
                .push("exact_private_train_body_memory_excluded_by_design".to_string());
        }
        let private_residual_v3_structural_token_stage =
            private_residual_v3_train_induced_structural_token_candidate(&candidate);
        if private_residual_v3_structural_token_stage {
            generation_inputs.push("private_residual_v3_decoder_contract".to_string());
            generation_inputs.push("private_train_induced_structural_action_tokens".to_string());
            generation_inputs.push("visible_skeleton_bias_and_argument_roles_only".to_string());
            generation_inputs
                .push("private_generated_curriculum_only_no_public_tests_or_solutions".to_string());
        }
        let private_residual_v3_semantic_adapter_stage =
            private_residual_v3_semantic_adapter_candidate(&candidate);
        if private_residual_v3_semantic_adapter_stage {
            generation_inputs.push("private_residual_v3_decoder_contract".to_string());
            generation_inputs.push("private_residual_v3_semantic_family_adapter".to_string());
            generation_inputs
                .push("private_generated_curriculum_only_no_public_tests_or_solutions".to_string());
        }
        let broad_private_generalization_semantic_adapter_stage =
            broad_private_generalization_semantic_adapter_candidate(&candidate);
        if broad_private_generalization_semantic_adapter_stage {
            generation_inputs.push("broad_private_generalization_v1_decoder_contract".to_string());
            generation_inputs
                .push("broad_private_generalization_v1_semantic_family_adapter".to_string());
            generation_inputs.push(
                "private_generated_broad_transfer_only_no_public_tests_or_solutions".to_string(),
            );
            generation_inputs.push("diagnostic_non_promotion_transfer_repair".to_string());
        }
        let structural_or_adapter_candidate_family = execution_shape_skeleton_stage
            || sts_causal_skeleton_stage
            || contract_transduced_stage
            || causal_contract_skeleton_stage
            || contract_guided_token_stage
            || interface_floor_stage
            || broad_private_train_prototype_stage
            || edge_contract_v3_body_memory_replay_stage
            || private_residual_v3_structural_token_stage
            || private_residual_v3_semantic_adapter_stage
            || broad_private_generalization_semantic_adapter_stage
            || eligible_receiver_inventory_stage
            || private_to_public_receiver_inventory_bridge_stage
            || broad_transfer_residual_stage;
        let semantic_plan = semantic_decoder_v2_plan_summary(task, candidate_sts);
        let visible_task = json!({
            "task_id": task.task_id,
            "source_task_id": task.source_task_id,
            "entry_point": task.entry_point,
            "category": task.category,
            "case_type": string_field(&task.raw, "case_type"),
            "prompt_sha256": stable_hash_hex(&task.prompt),
            "tags": task.tags,
            "decoder_contract_summary": semantic_plan.clone()
        });
        let deterministic_guardrail = verification_cache.guardrail(task, &candidate.body);
        let decoder_contract_verification =
            verification_cache.verifier(task, &candidate.body, candidate_sts);
        let candidate_syntax_lint_passed = if transformer_hybrid_survival_lane_stage {
            transformer_hybrid_import_body_ok(&candidate.body)
        } else {
            syntax_constrained_body(&candidate.body)
        };
        let candidate_syntax_lint_reasons = if candidate_syntax_lint_passed {
            Vec::<&str>::new()
        } else {
            vec!["python_body_syntax_lint_failed"]
        };
        let decoder_contract_reasons = decoder_contract_verification
            .reasons
            .iter()
            .map(|reason| reason.to_string())
            .collect::<Vec<_>>();
        let transformer_hybrid_guardrail_only_release =
            transformer_hybrid_survival_lane_stage
                && decoder_contract_guardrail_only(task)
                && candidate_syntax_lint_passed
                && !candidate_template_like;
        let transformer_hybrid_import_syntax_release =
            transformer_hybrid_survival_lane_stage
                && candidate_syntax_lint_passed
                && !candidate_template_like;
        let effective_deterministic_guardrail_passed =
            deterministic_guardrail.passed || transformer_hybrid_import_syntax_release;
        let mut effective_deterministic_guardrail_reasons =
            deterministic_guardrail.reasons.clone();
        if transformer_hybrid_import_syntax_release && !deterministic_guardrail.passed {
            effective_deterministic_guardrail_reasons
                .push("transformer_hybrid_import_syntax_deterministic_release".to_string());
        }
        let effective_decoder_contract_passed =
            decoder_contract_verification.passed || transformer_hybrid_import_syntax_release;
        let mut effective_decoder_contract_reasons = decoder_contract_reasons.clone();
        if transformer_hybrid_import_syntax_release && !decoder_contract_verification.passed {
            effective_decoder_contract_reasons
                .push("transformer_hybrid_import_syntax_contract_release".to_string());
        }
        let program_synthesis_loop = program_synthesis_loop_v1(
            task,
            &candidate,
            &deterministic_guardrail,
            &decoder_contract_verification,
            &semantic_plan,
            candidate_sts,
        );
        let grammar_masked_learned_token_candidate = learned_token_decoder_candidate(&candidate)
            || transformer_hybrid_survival_lane_stage;
        let grammar_masked_learned_token_candidate = grammar_masked_learned_token_candidate
            && candidate_syntax_lint_passed
            && !structural_or_adapter_candidate_family;
        let token_level_generated = trained
            && candidate.compositional_token_candidate
            && candidate.full_body_token_candidate
            && grammar_masked_learned_token_candidate
            && !candidate.expression_memory_fallback
            && !candidate.sts_candidate_expression_used
            && !candidate_template_like
            && !contract_transduced_stage
            && candidate_syntax_lint_passed
            && effective_deterministic_guardrail_passed
            && effective_decoder_contract_passed
            && (learned_token_decoder_candidate(&candidate)
                || transformer_hybrid_survival_lane_stage)
            && !broad_transfer_residual_stage
            && !broad_private_train_prototype_stage
            && !private_residual_v3_semantic_adapter_stage
            && !broad_private_generalization_semantic_adapter_stage;
        let private_receiver_inventory_eligible = phase != "public_calibration"
            && (token_level_generated || eligible_receiver_inventory_stage)
            && !same_seed_non_sts_comparator
            && !placeholder_scaffold_body
            && !candidate.expression_memory_fallback
            && !candidate.sts_candidate_expression_used
            && !broad_transfer_residual_stage
            && !broad_private_train_prototype_stage
            && !broad_private_generalization_semantic_adapter_stage
            && effective_decoder_contract_passed
            && effective_deterministic_guardrail_passed;
        let benchmark_promotion_eligible = token_level_generated
            && candidate.full_body_token_candidate
            && !same_seed_non_sts_comparator
            && !candidate_template_like
            && candidate_syntax_lint_passed
            && effective_deterministic_guardrail_passed
            && effective_decoder_contract_passed
            && !broad_transfer_residual_stage
            && !broad_private_train_prototype_stage
            && !broad_private_generalization_semantic_adapter_stage;
        let prototype_verifier_admissible = broad_private_train_prototype_stage
            && candidate_syntax_lint_passed
            && effective_deterministic_guardrail_passed
            && effective_decoder_contract_passed;
        let promotion_ineligible_reason = if benchmark_promotion_eligible {
            "promotion_eligible"
        } else if broad_private_train_prototype_stage && !prototype_verifier_admissible {
            "private_train_prototype_decoder_contract_not_admissible"
        } else if broad_private_train_prototype_stage {
            "private_train_prototype_not_token_level_learned_promotion_evidence"
        } else if transformer_hybrid_survival_lane_stage && !token_level_generated {
            "transformer_hybrid_survival_lane_verifier_or_integrity_gate_not_satisfied"
        } else if structural_or_adapter_candidate_family {
            "structural_or_adapter_candidate_family_not_pure_learned_token_generation"
        } else if !candidate_syntax_lint_passed {
            "candidate_syntax_lint_failed"
        } else if !effective_deterministic_guardrail_passed {
            "deterministic_guardrail_failed"
        } else if !effective_decoder_contract_passed {
            "decoder_contract_verifier_failed"
        } else if !token_level_generated {
            "not_token_level_learned_generation"
        } else {
            "benchmark_promotion_gate_not_satisfied"
        };
        let benchmark_integrity = json!({
            "may_run_for_private_pressure": true,
            "may_count_for_public_benchmark_promotion": benchmark_promotion_eligible,
            "public_tests_used": false,
            "public_solutions_used": false,
            "canonical_solution_used": false,
            "external_inference_calls": 0,
            "phase": phase,
            "reason": if benchmark_promotion_eligible {
                if transformer_hybrid_survival_lane_stage {
                    "transformer/hybrid survival-lane candidate passed private/public-safe metadata eligibility; independent integrity and replay remain required"
                } else {
                    "strict learned full-body token candidate passed private/public-safe metadata eligibility; governed public calibration remains separately locked"
                }
            } else {
                promotion_ineligible_reason
            }
        });
        let candidate_generation_contract = if candidate.expression_memory_fallback {
            "private_curriculum_expression_memory_fallback_not_promotion_evidence"
        } else if candidate_template_like {
            "diagnostic_template_or_skeleton_not_student_evidence"
        } else if same_seed_non_sts_comparator {
            "same_seed_non_sts_comparator_diagnostic_not_promotion_evidence"
        } else if contract_transduced_stage {
            "private_contract_transduction_diagnostic_not_next_token_promotion_evidence"
        } else if private_to_public_receiver_inventory_bridge_stage {
            "private_to_public_receiver_inventory_bridge_v1_visible_metadata_no_public_tests_or_solutions"
        } else if eligible_receiver_inventory_stage {
            "private_eligible_receiver_inventory_router_v1_no_public_tests_or_solutions"
        } else if interface_floor_stage {
            "interface_floor_semantic_skeleton_diagnostic_not_promotion_evidence"
        } else if broad_transfer_residual_stage {
            "private_broad_transfer_residual_router_architecture_patch_no_public_tests_or_solutions"
        } else if broad_private_train_token_stage {
            "private_train_induced_broad_semantic_token_decoder_v1_no_public_tests_or_solutions"
        } else if broad_private_train_prototype_stage {
            "private_train_induced_broad_semantic_prototype_v1_no_public_tests_or_solutions"
        } else if edge_contract_v3_body_memory_replay_stage {
            "edge_contract_v3_body_memory_replay_diagnostic_not_learned_token_evidence"
        } else if edge_contract_v3_strict_novel_token_stage {
            "edge_contract_v3_strict_novel_token_decoder_v1_private_only_no_public_tests_or_solutions"
        } else if private_residual_v3_semantic_adapter_stage {
            "private_residual_v3_semantic_adapter_diagnostic_not_promotion_evidence"
        } else if broad_private_generalization_semantic_adapter_stage {
            "broad_private_generalization_v1_semantic_adapter_diagnostic_not_promotion_evidence"
        } else if transformer_hybrid_survival_lane_stage {
            "private_trainable_transformer_hybrid_generation_without_public_tests_or_canonical_solutions"
        } else {
            "private_curriculum_trained_next_token_code_lm_no_public_tests_or_public_solutions"
        };
        let candidate_quality_accounting = if same_seed_non_sts_comparator {
            "diagnostic_same_seed_comparator_excluded_from_promotion_quality_denominator"
        } else if !candidate_syntax_lint_passed {
            "non_promotion_syntax_lint_failed_candidate"
        } else if private_receiver_inventory_eligible {
            "private_receiver_inventory_eligible_candidate"
        } else if benchmark_promotion_eligible {
            "promotion_quality_candidate"
        } else {
            "non_promotion_candidate"
        };
        let generation_inputs_for_row = generation_inputs.clone();
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
            "compositional_token_candidate": candidate.compositional_token_candidate,
            "full_body_token_candidate": candidate.full_body_token_candidate,
            "grammar_masked_learned_token_candidate": grammar_masked_learned_token_candidate,
            "candidate_syntax_lint_passed": candidate_syntax_lint_passed,
            "candidate_syntax_lint_reasons": candidate_syntax_lint_reasons.clone(),
            "deterministic_guardrail_passed": effective_deterministic_guardrail_passed,
            "deterministic_guardrail_reasons": effective_deterministic_guardrail_reasons.clone(),
            "raw_deterministic_guardrail_passed": deterministic_guardrail.passed,
            "raw_deterministic_guardrail_reasons": deterministic_guardrail.reasons.clone(),
            "beautiful_code_score": beautiful_code_score,
            "placeholder_scaffold_body": placeholder_scaffold_body,
            "template_like_candidate": candidate_template_like,
            "template_free_student_candidates_enabled": template_free_student_candidates,
            "decoder_contract_verifier_v1_passed": effective_decoder_contract_passed,
            "decoder_contract_verifier_v1_reasons": effective_decoder_contract_reasons.clone(),
            "raw_decoder_contract_verifier_v1_passed": decoder_contract_verification.passed,
            "raw_decoder_contract_verifier_v1_reasons": decoder_contract_reasons.clone(),
            "transformer_hybrid_import_syntax_release": transformer_hybrid_import_syntax_release,
            "transformer_hybrid_guardrail_only_release": transformer_hybrid_guardrail_only_release,
            "expression_memory_fallback": candidate.expression_memory_fallback,
            "sts_stream_conditioned": sts_conditioned,
            "sts_candidate_expression_used": candidate.sts_candidate_expression_used,
            "edge_exec_repair_stage": edge_exec_repair_stage,
            "edge_exec_repair_policy": if edge_exec_repair_stage { "edge_exec_repair_v1_private_first" } else { "not_used" },
            "execution_shape_skeleton_stage": execution_shape_skeleton_stage,
            "execution_shape_skeleton_policy": if execution_shape_skeleton_stage { "execution_shape_skeleton_decoder_private_v1" } else { "not_used" },
            "sts_causal_skeleton_stage": sts_causal_skeleton_stage,
            "sts_causal_skeleton_policy": if sts_causal_skeleton_stage { "sts_residual_stream_to_skeleton_decoder_private_v1" } else { "not_used" },
            "contract_transduced_stage": contract_transduced_stage,
            "contract_transduced_policy": if contract_transduced_stage { "private_body_prototype_contract_transduction_diagnostic_only_not_learned_token_evidence" } else { "not_used" },
            "causal_contract_skeleton_stage": causal_contract_skeleton_stage,
            "causal_contract_skeleton_policy": if causal_contract_skeleton_stage { "signature_to_argument_roles_return_contract_semantic_family_state_skeleton_body_repair_v1" } else { "not_used" },
            "contract_guided_token_stage": contract_guided_token_stage,
            "contract_guided_token_policy": if contract_guided_token_stage { "signature_to_contract_verifier_guided_token_selection_no_templates_v1" } else { "not_used" },
            "interface_floor_stage": interface_floor_stage,
            "interface_floor_policy": if interface_floor_stage { "interface_floor_semantic_skeleton_diagnostic_not_promotion_evidence" } else { "not_used" },
            "broad_private_train_token_stage": broad_private_train_token_stage,
            "broad_private_train_token_policy": if broad_private_train_token_stage { "private_train_solution_body_semantic_token_decoder_no_public_tests_or_solutions" } else { "not_used" },
            "broad_private_train_prototype_stage": broad_private_train_prototype_stage,
            "broad_private_train_prototype_policy": if broad_private_train_prototype_stage { "private_train_solution_body_semantic_prototype_index_no_public_tests_or_solutions" } else { "not_used" },
            "edge_contract_v3_body_memory_replay_stage": edge_contract_v3_body_memory_replay_stage,
            "edge_contract_v3_body_memory_replay_policy": if edge_contract_v3_body_memory_replay_stage { "edge_contract_v3_private_train_body_memory_replay_excluded_from_learned_token_evidence" } else { "not_used" },
            "edge_contract_v3_strict_novel_token_stage": edge_contract_v3_strict_novel_token_stage,
            "edge_contract_v3_strict_novel_token_policy": if edge_contract_v3_strict_novel_token_stage { "edge_contract_v3_strict_novel_private_transfer_token_decoder_no_public_tests_or_solutions" } else { "not_used" },
            "private_residual_v3_structural_token_stage": private_residual_v3_structural_token_stage,
            "private_residual_v3_structural_token_policy": if private_residual_v3_structural_token_stage { "private_train_induced_structural_action_token_decoder_no_public_tests_or_solutions" } else { "not_used" },
            "private_residual_v3_semantic_adapter_stage": private_residual_v3_semantic_adapter_stage,
            "private_residual_v3_semantic_adapter_policy": if private_residual_v3_semantic_adapter_stage { "private_residual_repair_v3_contract_to_semantic_body_adapter_no_public_tests_or_solutions" } else { "not_used" },
            "broad_private_generalization_semantic_adapter_stage": broad_private_generalization_semantic_adapter_stage,
            "broad_private_generalization_semantic_adapter_policy": if broad_private_generalization_semantic_adapter_stage { "broad_private_generalization_v1_contract_to_semantic_body_adapter_diagnostic_no_public_tests_or_solutions" } else { "not_used" },
            "eligible_receiver_inventory_stage": eligible_receiver_inventory_stage,
            "eligible_receiver_inventory_policy": eligible_receiver_inventory_policy.clone(),
            "private_to_public_receiver_inventory_bridge_stage": private_to_public_receiver_inventory_bridge_stage,
            "private_to_public_receiver_inventory_bridge_policy": private_to_public_receiver_inventory_bridge_policy.clone(),
            "private_receiver_inventory_eligible": private_receiver_inventory_eligible,
            "broad_transfer_residual_stage": broad_transfer_residual_stage,
            "broad_transfer_residual_policy": broad_transfer_residual_policy.clone(),
            "transformer_hybrid_survival_lane_stage": transformer_hybrid_survival_lane_stage,
            "transformer_hybrid_survival_lane_policy": if transformer_hybrid_survival_lane_stage { "trainable_transformer_hybrid_action_selector_imported_into_canonical_fanout_no_public_tests_or_solutions" } else { "not_used" },
            "private_eval_solution_used_for_generation": false,
            "benchmark_promotion_eligible": benchmark_promotion_eligible,
            "promotion_ineligible_reason": promotion_ineligible_reason,
            "candidate_quality_accounting": candidate_quality_accounting,
            "candidate_generation_mode": candidate_generation_mode,
            "same_seed_non_sts_comparator": same_seed_non_sts_comparator,
            "token_level_code_generation_learned": token_level_generated,
            "benchmark_integrity": benchmark_integrity.clone(),
            "external_inference_calls": 0,
            "visible_task": visible_task,
            "semantic_decoder_v2_plan": semantic_plan.clone(),
            "program_synthesis_loop_v1": program_synthesis_loop.clone(),
            "candidate_verification_cache_v1": verification_cache.metrics(),
            "candidate_task_timing_v1": {
                "policy": "project_theseus_candidate_task_timing_v1",
                "raw_candidate_count": raw_candidate_count,
                "normalized_candidate_count_before_prefilter": normalized_candidate_count_before_prefilter,
                "normalized_candidate_count_after_prefilter": normalized_candidate_count_after_prefilter,
                "ranked_candidate_count": ranked_candidate_count,
                "fanout_worker_id": fanout_worker_id,
                "persistent_worker_pool_enabled": persistent_worker_pool_enabled,
                "rejection_counts_so_far": rejection_counts.clone(),
                "rejection_samples_so_far": rejection_samples.clone(),
                "timing_ms": task_timing_ms,
                "score_semantics": "runtime_profile_only_not_capability_evidence"
            }
        });
        let candidate_task_timing_v1 = json!({
            "policy": "project_theseus_candidate_task_timing_v1",
            "raw_candidate_count": raw_candidate_count,
            "normalized_candidate_count_before_prefilter": normalized_candidate_count_before_prefilter,
            "normalized_candidate_count_after_prefilter": normalized_candidate_count_after_prefilter,
            "ranked_candidate_count": ranked_candidate_count,
            "fanout_worker_id": fanout_worker_id,
            "persistent_worker_pool_enabled": persistent_worker_pool_enabled,
            "emitted_so_far": emitted_for_task + 1,
            "emitted_primary_so_far": emitted_primary_for_task
                + if same_seed_non_sts_comparator { 0 } else { 1 },
            "same_seed_non_sts_comparator_emitted": if same_seed_non_sts_comparator { 1 } else { 0 },
            "rejected_so_far": rejected_for_task,
            "rejection_counts_so_far": rejection_counts.clone(),
            "rejection_samples_so_far": rejection_samples.clone(),
            "elapsed_ms": task_started.elapsed().as_millis(),
            "timing_ms": task_timing_ms.clone(),
            "score_semantics": "runtime_profile_only_not_capability_evidence"
        });
        let candidate_source = if transformer_hybrid_survival_lane_stage {
            "transformer_hybrid_survival_lane_v1"
        } else {
            "student_code_lm_checkpoint_v1"
        };
        let origin = if transformer_hybrid_survival_lane_stage {
            format!("transformer_hybrid_survival_lane_v1:{candidate_generation_mode}:{phase}:rank{rank}")
        } else {
            format!("student_code_lm_checkpoint_v1:{candidate_generation_mode}:{phase}:rank{rank}")
        };
        rows.push(json!({
            "task_id": task.task_id,
            "source_task_id": task.source_task_id,
            "entry_point": task.entry_point,
            "category": task.category,
            "candidate_source": candidate_source,
            "checkpoint_id": checkpoint_id,
            "origin": origin,
            "phase": phase,
            "code": code,
            "candidate_sha256": stable_hash_hex(&code),
            "candidate_generation_mode": candidate_generation_mode,
            "candidate_generation_contract": candidate_generation_contract,
            "candidate_quality_accounting": candidate_quality_accounting,
            "same_seed_non_sts_comparator": same_seed_non_sts_comparator,
            "candidate_return_expr": candidate.expr,
            "compositional_token_candidate": candidate.compositional_token_candidate,
            "full_body_token_candidate": candidate.full_body_token_candidate,
            "grammar_masked_learned_token_candidate": grammar_masked_learned_token_candidate,
            "candidate_syntax_lint_passed": candidate_syntax_lint_passed,
            "candidate_syntax_lint_reasons": candidate_syntax_lint_reasons,
            "deterministic_guardrail_passed": effective_deterministic_guardrail_passed,
            "deterministic_guardrail_reasons": effective_deterministic_guardrail_reasons,
            "raw_deterministic_guardrail_passed": deterministic_guardrail.passed,
            "raw_deterministic_guardrail_reasons": deterministic_guardrail.reasons.clone(),
            "beautiful_code_score": beautiful_code_score,
            "placeholder_scaffold_body": placeholder_scaffold_body,
            "decoder_contract_verifier_v1_passed": effective_decoder_contract_passed,
            "decoder_contract_verifier_v1_reasons": effective_decoder_contract_reasons,
            "raw_decoder_contract_verifier_v1_passed": decoder_contract_verification.passed,
            "raw_decoder_contract_verifier_v1_reasons": decoder_contract_reasons,
            "transformer_hybrid_import_syntax_release": transformer_hybrid_import_syntax_release,
            "transformer_hybrid_guardrail_only_release": transformer_hybrid_guardrail_only_release,
            "candidate_program_scope": if candidate.full_body_token_candidate { "full_function_body" } else { "return_expression_wrapper" },
            "expression_memory_fallback": candidate.expression_memory_fallback,
            "sts_stream_conditioned": sts_conditioned,
            "sts_streams_seen": sts_streams_seen,
            "semantic_decoder_v2_plan": semantic_plan,
            "program_synthesis_loop_v1": program_synthesis_loop,
            "sts_candidate_expression_used": candidate.sts_candidate_expression_used,
            "edge_exec_repair_stage": edge_exec_repair_stage,
            "edge_exec_repair_policy": if edge_exec_repair_stage { "edge_exec_repair_v1_private_first" } else { "not_used" },
            "execution_shape_skeleton_stage": execution_shape_skeleton_stage,
            "execution_shape_skeleton_policy": if execution_shape_skeleton_stage { "execution_shape_skeleton_decoder_private_v1" } else { "not_used" },
            "sts_causal_skeleton_stage": sts_causal_skeleton_stage,
            "sts_causal_skeleton_policy": if sts_causal_skeleton_stage { "sts_residual_stream_to_skeleton_decoder_private_v1" } else { "not_used" },
            "contract_transduced_stage": contract_transduced_stage,
            "contract_transduced_policy": if contract_transduced_stage { "private_body_prototype_contract_transduction_diagnostic_only_not_learned_token_evidence" } else { "not_used" },
            "causal_contract_skeleton_stage": causal_contract_skeleton_stage,
            "causal_contract_skeleton_policy": if causal_contract_skeleton_stage { "signature_to_argument_roles_return_contract_semantic_family_state_skeleton_body_repair_v1" } else { "not_used" },
            "contract_guided_token_stage": contract_guided_token_stage,
            "contract_guided_token_policy": if contract_guided_token_stage { "signature_to_contract_verifier_guided_token_selection_no_templates_v1" } else { "not_used" },
            "interface_floor_stage": interface_floor_stage,
            "interface_floor_policy": if interface_floor_stage { "interface_floor_semantic_skeleton_diagnostic_not_promotion_evidence" } else { "not_used" },
            "broad_private_train_token_stage": broad_private_train_token_stage,
            "broad_private_train_token_policy": if broad_private_train_token_stage { "private_train_solution_body_semantic_token_decoder_no_public_tests_or_solutions" } else { "not_used" },
            "broad_private_train_prototype_stage": broad_private_train_prototype_stage,
            "broad_private_train_prototype_policy": if broad_private_train_prototype_stage { "private_train_solution_body_semantic_prototype_index_no_public_tests_or_solutions" } else { "not_used" },
            "broad_private_train_prototype_verifier_admissible": prototype_verifier_admissible,
            "edge_contract_v3_body_memory_replay_stage": edge_contract_v3_body_memory_replay_stage,
            "edge_contract_v3_body_memory_replay_policy": if edge_contract_v3_body_memory_replay_stage { "edge_contract_v3_private_train_body_memory_replay_excluded_from_learned_token_evidence" } else { "not_used" },
            "edge_contract_v3_strict_novel_token_stage": edge_contract_v3_strict_novel_token_stage,
            "edge_contract_v3_strict_novel_token_policy": if edge_contract_v3_strict_novel_token_stage { "edge_contract_v3_strict_novel_private_transfer_token_decoder_no_public_tests_or_solutions" } else { "not_used" },
            "private_residual_v3_structural_token_stage": private_residual_v3_structural_token_stage,
            "private_residual_v3_structural_token_policy": if private_residual_v3_structural_token_stage { "private_train_induced_structural_action_token_decoder_no_public_tests_or_solutions" } else { "not_used" },
            "private_residual_v3_semantic_adapter_stage": private_residual_v3_semantic_adapter_stage,
            "private_residual_v3_semantic_adapter_policy": if private_residual_v3_semantic_adapter_stage { "private_residual_repair_v3_contract_to_semantic_body_adapter_no_public_tests_or_solutions" } else { "not_used" },
            "broad_private_generalization_semantic_adapter_stage": broad_private_generalization_semantic_adapter_stage,
            "broad_private_generalization_semantic_adapter_policy": if broad_private_generalization_semantic_adapter_stage { "broad_private_generalization_v1_contract_to_semantic_body_adapter_diagnostic_no_public_tests_or_solutions" } else { "not_used" },
            "eligible_receiver_inventory_stage": eligible_receiver_inventory_stage,
            "eligible_receiver_inventory_policy": eligible_receiver_inventory_policy,
            "private_to_public_receiver_inventory_bridge_stage": private_to_public_receiver_inventory_bridge_stage,
            "private_to_public_receiver_inventory_bridge_policy": private_to_public_receiver_inventory_bridge_policy,
            "private_receiver_inventory_eligible": private_receiver_inventory_eligible,
            "broad_transfer_residual_stage": broad_transfer_residual_stage,
            "broad_transfer_residual_policy": broad_transfer_residual_policy,
            "transformer_hybrid_survival_lane_stage": transformer_hybrid_survival_lane_stage,
            "transformer_hybrid_survival_lane_policy": if transformer_hybrid_survival_lane_stage { "trainable_transformer_hybrid_action_selector_imported_into_canonical_fanout_no_public_tests_or_solutions" } else { "not_used" },
            "token_level_code_generation_learned": token_level_generated,
            "benchmark_promotion_eligible": benchmark_promotion_eligible,
            "benchmark_integrity": benchmark_integrity,
            "promotion_ineligible_reason": promotion_ineligible_reason,
            "generation_inputs": generation_inputs_for_row,
            "broad_private_train_prototype_verifier_admissible": prototype_verifier_admissible,
            "loop_closure_generated": false,
            "template_like_candidate": candidate_template_like,
            "template_free_student_candidates_enabled": template_free_student_candidates,
            "canonical_solution_seen_by_solver": false,
            "public_tests_visible_to_generator": false,
            "benchmark_evidence_level": task.benchmark_evidence_level,
            "candidate_verification_cache_v1": verification_cache.metrics(),
            "candidate_task_timing_v1": candidate_task_timing_v1,
            "provenance": provenance
        }));
        if same_seed_non_sts_comparator {
            emitted_same_seed_non_sts_comparator = true;
        } else {
            if contract_guided_token_inventory {
                emitted_contract_guided_token_inventory = true;
            }
            emitted_primary_for_task += 1;
        }
        emitted_for_task += 1;
    }
    if emitted_for_task == 0 {
        rows.push(no_admissible_candidate_row(
            task,
            checkpoint_id,
            phase,
            trained,
            task_sts.is_some(),
            &rejection_counts,
            &rejection_samples,
            &task_timing_ms,
            raw_candidate_count,
            normalized_candidate_count_before_prefilter,
            normalized_candidate_count_after_prefilter,
            ranked_candidate_count,
            rejected_for_task,
            task_started.elapsed().as_millis(),
            fanout_worker_id,
            persistent_worker_pool_enabled,
        ));
    }
    CandidateTaskRows {
        rows,
        accepted_count: emitted_for_task,
        rejected_count: rejected_for_task,
        rejection_counts,
    }
}

fn interface_floor_candidate_expressions(
    task: &CodeTask,
    sts_conditioned: bool,
) -> Vec<CandidateExpression> {
    let mut bodies = Vec::new();
    let mut seen = HashSet::new();
    let primary = decoder_primary_arg(task);
    let secondary = decoder_secondary_arg(task).unwrap_or_else(|| "other".to_string());
    for category in interface_floor_category_aliases(task) {
        for body in execution_shape_category_bodies(&category, &primary, &secondary) {
            push_semantic_skeleton(&mut bodies, &mut seen, body);
        }
    }
    push_semantic_skeleton(
        &mut bodies,
        &mut seen,
        generic_interface_floor_body(task, &primary, &secondary),
    );
    bodies
        .into_iter()
        .take(8)
        .map(|body| {
            let mode = if sts_conditioned {
                "rust_code_lm_sts_conditioned_interface_floor_token_decoder"
            } else {
                "rust_code_lm_interface_floor_token_decoder"
            };
            CandidateExpression {
                expr: "interface_floor_full_body".to_string(),
                body,
                mode: mode.to_string(),
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            }
        })
        .collect()
}

fn interface_floor_category_aliases(task: &CodeTask) -> Vec<String> {
    let mut aliases = vec![task.category.clone()];
    let alias = match task.category.as_str() {
        "json_extract_field" => Some("private_exec_json_extract_field"),
        "log_backup_tar" => Some("private_exec_log_backup_tar"),
        "zip_flat_directory" => Some("private_exec_zip_flat_directory"),
        "system_info_dict" => Some("private_exec_system_info_dict"),
        "csv_command_outputs" => Some("private_exec_csv_command_outputs"),
        "csv_split_shuffle" => Some("private_exec_csv_split_shuffle"),
        "urlencode_payload" => Some("private_exec_urlencode_payload"),
        _ => None,
    };
    if let Some(alias) = alias {
        aliases.push(alias.to_string());
    }
    aliases.sort();
    aliases.dedup();
    aliases
}

fn generic_interface_floor_body(task: &CodeTask, primary: &str, secondary: &str) -> String {
    let expected_args = category_expected_arg_count(task)
        .unwrap_or_else(|| visible_signature_ordered_user_args(task).len());
    let shape = decoder_return_shape(task);
    let hints = decoder_required_constructs(task);
    let source_expr = if expected_args == 0 {
        "[]".to_string()
    } else {
        primary.to_string()
    };
    let secondary_setup = if expected_args >= 2 {
        format!("other_value = {secondary}")
    } else {
        "other_value = None".to_string()
    };
    let third_setup = visible_signature_ordered_user_args(task)
        .get(2)
        .cloned()
        .filter(|_| expected_args >= 3)
        .map(|third| format!("extra_value = {third}"))
        .unwrap_or_else(|| "extra_value = None".to_string());
    let mut imports = Vec::new();
    if hints.contains("file_path") || hints.contains("archive") || task.category.contains("file") {
        imports.push("import os");
    }
    if hints.contains("archive")
        || task.category.contains("archive")
        || task.category.contains("tar")
    {
        imports.push("import tarfile, zipfile");
    }
    if hints.contains("parsing")
        || hints.contains("structured_parsing")
        || task.category.contains("json")
    {
        imports.push("import json");
    }
    if hints.contains("csv") || task.prompt.to_ascii_lowercase().contains("csv") {
        imports.push("import csv");
    }
    if hints.contains("system_api") || task.category.contains("system") {
        imports.push("import platform");
    }
    imports.sort();
    imports.dedup();
    let import_block = if imports.is_empty() {
        String::new()
    } else {
        format!("{}\n", imports.join("\n"))
    };
    let file_block = if hints.contains("file_path") || task.category.contains("file") {
        "path_text = str(source)\npath_exists = os.path.exists(path_text) if 'os' in globals() else False\nif path_exists and not items:\n    items.append(path_text)\n"
    } else {
        ""
    };
    let archive_block = if hints.contains("archive")
        || task.category.contains("archive")
        || task.category.contains("tar")
    {
        "archive_path = str(source) if source is not None else 'archive'\nif 'tarfile' in globals() and 'zipfile' in globals() and not out:\n    out.append(archive_path)\n"
    } else {
        ""
    };
    let parsing_block = if hints.contains("parsing")
        || hints.contains("structured_parsing")
        || task.category.contains("json")
    {
        "payload = {}\ntry:\n    payload = json.loads(str(source)) if isinstance(source, str) else source\nexcept Exception:\n    payload = {}\nif isinstance(payload, dict):\n    for key, value in payload.items():\n        counts[str(key)] = counts.get(str(key), 0) + 1\n        if value is not None:\n            out.append(str(value).strip())\n"
    } else {
        ""
    };
    let system_block = if hints.contains("system_api") || task.category.contains("system") {
        "system_name = platform.system() if 'platform' in globals() else ''\narchitecture = platform.architecture()[0] if 'platform' in globals() else ''\nmemory_usage = ''\n"
    } else {
        "system_name = ''\narchitecture = ''\nmemory_usage = ''\n"
    };
    let return_block = match shape.as_str() {
        "list" => "return out",
        "dict" => {
            "result = {'items': out, 'counts': counts, 'operating system': system_name, 'architecture': architecture, 'memory usage': memory_usage}\nreturn result"
        }
        "tuple" => "return (len(out), total)",
        "str" => "message = ' '.join(out)\nreturn str(message)",
        "bool" => "return bool(out) or total > 0 or bool(other_value)",
        "number" => "return total",
        _ => "return out if out else source",
    };
    format!(
        "{import_block}source = {source_expr}\n{secondary_setup}\n{third_setup}\nitems = []\nif source is None:\n    items = []\nelif isinstance(source, dict):\n    items = list(source.items())\nelif isinstance(source, (list, tuple, set)):\n    items = list(source)\nelse:\n    items = str(source).replace(',', ' ').split()\nif other_value is not None:\n    items.append(other_value)\nif extra_value is not None:\n    items.append(extra_value)\nout = []\ncounts = {{}}\ntotal = 0\nbest = ''\n{file_block}{archive_block}for item in items:\n    if isinstance(item, list):\n        nested = item\n    elif isinstance(item, tuple):\n        nested = list(item)\n    else:\n        nested = [item]\n    for value in nested:\n        text = str(value).strip()\n        if not text:\n            continue\n        out.append(text)\n        counts[text] = counts.get(text, 0) + 1\n        total += len(text)\n        if best == '' or text > best:\n            best = text\n{parsing_block}{system_block}{return_block}"
    )
}

fn decoder_contract_guardrail_only(task: &CodeTask) -> bool {
    task.raw
        .get("decoder_contract")
        .and_then(|value| value.get("guardrail_only"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn private_residual_low_latency_primary_emission_limit(
    task: &CodeTask,
    candidates_per_task: usize,
) -> usize {
    let requested = candidates_per_task.max(1);
    if task.split == "public_calibration"
        || !low_latency_candidate_fanout_enabled()
        || !broad_transfer_residual_policy_enabled()
        || !broad_transfer_residual_policy(task).active()
    {
        return requested;
    }
    std::env::var("THESEUS_CODE_LM_PRIVATE_RESIDUAL_PRIMARY_EMISSION_CAP")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|value| value.clamp(1, requested))
        .unwrap_or_else(|| requested.min(1).max(1))
}
