use super::*;

pub(super) fn sts_bridge_ranked_outputs_from_pool(
    task: &CodeTask,
    pool: Vec<String>,
    limit: usize,
    sts_streams: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut scored = Vec::new();
    let mut variant_seen = HashSet::new();
    let mut verifier_cache: HashMap<String, DecoderContractVerification> = HashMap::new();
    let mut variant_cache = CandidateVariantCache::default();
    for body in pool {
        let canonical =
            canonicalize_task_candidate_body_aliases(task, &normalize_generated_body(&body));
        for variant in variant_cache.variants(task, &canonical) {
            let normalized =
                canonicalize_task_candidate_body_aliases(task, &normalize_generated_body(&variant));
            let trimmed = normalized.trim().to_string();
            if trimmed.is_empty() || !variant_seen.insert(trimmed.clone()) {
                continue;
            }
            let verification = cached_decoder_contract_verification(
                task,
                &trimmed,
                Some(sts_streams),
                &mut verifier_cache,
            );
            if !token_contract_candidate_body_ok_with_verification(task, &trimmed, &verification) {
                continue;
            }
            let score = contract_guided_token_candidate_score_with_verification(
                task,
                &trimmed,
                Some(sts_streams),
                &verification,
            ) + sts_conditioned_token_bridge_score(task, &trimmed, sts_streams);
            scored.push((score, trimmed));
        }
    }
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.len().cmp(&b.1.len()))
            .then_with(|| a.1.cmp(&b.1))
    });
    scored
        .into_iter()
        .map(|(_, body)| body)
        .take(limit)
        .collect()
}

fn sts_conditioned_token_bridge_score(
    task: &CodeTask,
    body: &str,
    sts_streams: &BTreeMap<String, String>,
) -> f32 {
    let lowered = body.to_lowercase();
    let required = decoder_required_constructs(task);
    let mut score = if sts_decoder_control_demotes_sts_preference(Some(sts_streams)) {
        -1.25 + sts_skeleton_alignment_score(body, Some(sts_streams)).max(0.0) * 0.25
    } else {
        1.0 + sts_skeleton_alignment_score(body, Some(sts_streams)).max(0.0) * 1.4
    };
    if visible_argument_contract_ok(task, body) {
        score += 0.9;
    } else {
        score -= 1.1;
    }
    if return_shape_contract_ok(task, &lowered) {
        score += 0.7 + return_shape_builder_bias(task, &lowered).min(0.8);
    }
    if required_construct_contract_ok_for_task(task, body, &required) {
        score += 0.8;
    }
    if execution_shape_contract_ok(task, body, &required) {
        score += 0.6;
    }
    if body_semantically_admissible(task, body) {
        score += 0.6;
    }
    if body.contains('\n')
        && (lowered.contains("if ") || lowered.contains("for ") || lowered.contains("while "))
    {
        score += 0.35;
    }
    score
}
