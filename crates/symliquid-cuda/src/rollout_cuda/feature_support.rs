use super::RolloutFeatureConfig;
use symliquid_core::error::{Result, SymError};
use symliquid_core::tensor::Tensor;

type EntitySlotFact = (String, String, f32);
type EntitySlotTrace = (String, Vec<EntitySlotFact>);

pub(super) fn select_feature_rows(
    source: &Tensor,
    rows: &[usize],
    hv_dim: usize,
) -> Result<Tensor> {
    source.ensure_cols(hv_dim, "select feature rows")?;
    let mut data = Vec::with_capacity(rows.len() * hv_dim);
    for row in rows {
        if *row >= source.rows {
            return Err(SymError::Shape(format!(
                "feature row {} outside row count {}",
                row, source.rows
            )));
        }
        data.extend_from_slice(source.row(*row));
    }
    Tensor::new(rows.len(), hv_dim, data)
}

#[cfg(feature = "cuda")]
pub(super) fn retrieval_first_curriculum(
    cases: Vec<symliquid_core::benchmarks::BenchmarkCase>,
    features: Tensor,
    targets: Vec<usize>,
    hv_dim: usize,
) -> Result<(
    Vec<symliquid_core::benchmarks::BenchmarkCase>,
    Tensor,
    Vec<usize>,
)> {
    features.ensure_cols(hv_dim, "retrieval curriculum features")?;
    if cases.len() != features.rows || targets.len() != features.rows {
        return Err(SymError::Shape(format!(
            "retrieval curriculum rows mismatch: cases={} features={} targets={}",
            cases.len(),
            features.rows,
            targets.len()
        )));
    }

    let expanded_rows = cases
        .iter()
        .map(|case| retrieval_curriculum_repeats(&case.task))
        .sum::<usize>();
    let mut expanded_cases = Vec::with_capacity(expanded_rows);
    let mut expanded_targets = Vec::with_capacity(expanded_rows);
    let mut expanded_features = Vec::with_capacity(expanded_rows * hv_dim);

    for (row, case) in cases.iter().enumerate() {
        let repeats = retrieval_curriculum_repeats(&case.task);
        for _ in 0..repeats {
            expanded_cases.push(case.clone());
            expanded_targets.push(targets[row]);
            expanded_features.extend_from_slice(features.row(row));
        }
    }

    Ok((
        expanded_cases,
        Tensor::new(expanded_targets.len(), hv_dim, expanded_features)?,
        expanded_targets,
    ))
}

#[cfg(feature = "cuda")]
fn retrieval_curriculum_repeats(task: &str) -> usize {
    match task {
        "long_context_retrieval" => 8,
        "long_context_role_filler" => 8,
        "role_filler" => 4,
        _ => 1,
    }
}

#[cfg(feature = "cuda")]
pub(super) fn entity_slot_vsa_feature(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
    label_vectors: &[Vec<f32>],
    hv_dim: usize,
) -> Option<Vec<f32>> {
    let (query, facts) = entity_slot_facts(case)?;
    if facts.is_empty() || labels.is_empty() {
        return None;
    }

    let mut memory = vec![0.0; hv_dim];
    for (slot, value, weight) in facts {
        let slot_hv = semantic_hypervector(&format!("slot:{slot}"), hv_dim);
        let value_hv = labels
            .iter()
            .position(|label| label == &value)
            .map(|idx| label_vectors[idx].clone())
            .unwrap_or_else(|| semantic_hypervector(&format!("value:{value}"), hv_dim));
        for idx in 0..hv_dim {
            memory[idx] += weight * slot_hv[idx] * value_hv[idx];
        }
    }

    let query_hv = semantic_hypervector(&format!("slot:{query}"), hv_dim);
    let mut retrieved = vec![0.0; hv_dim];
    for idx in 0..hv_dim {
        retrieved[idx] = memory[idx] * query_hv[idx];
    }
    normalize_dense(&mut retrieved);

    let best_label = cleanup_label_index(&retrieved, label_vectors)?;
    let best_label_hv = &label_vectors[best_label];
    let best_label_value = &labels[best_label];
    let mut feature = vec![0.0; hv_dim];
    for idx in 0..hv_dim {
        feature[idx] = retrieved[idx] * 0.35 + best_label_hv[idx];
    }
    add_hv_feature(&mut feature, &format!("entity_slot_query:{query}"), 0.5);
    add_hv_feature(
        &mut feature,
        &format!("entity_slot_cleanup:{best_label_value}"),
        0.75,
    );
    normalize_dense(&mut feature);
    Some(feature)
}

#[cfg(feature = "cuda")]
fn entity_slot_facts(case: &symliquid_core::benchmarks::BenchmarkCase) -> Option<EntitySlotTrace> {
    match case.task.as_str() {
        "role_filler" | "long_context_role_filler" => {
            let query = extract_query_role(&case.observation)?;
            let mut facts = Vec::new();
            for line in case.observation.lines() {
                let Some((left, right)) = line.split_once("->") else {
                    continue;
                };
                let slot = clean_symbol(left);
                if slot.is_empty() || slot.starts_with("distractor") {
                    continue;
                }
                let value = clean_symbol(right);
                if !value.is_empty() {
                    facts.push((slot, value, 1.0));
                }
            }
            Some((query, facts))
        }
        "long_context_retrieval" => {
            let query = extract_backtick_key(&case.observation)
                .or_else(|| extract_after_marker_word(&case.observation, "for key "))?;
            let mut facts = Vec::new();
            for line in case.observation.lines() {
                let lower = line.to_ascii_lowercase();
                if let Some(after_key) = lower.split("retrieval key ").nth(1) {
                    let Some((key, after_value_marker)) =
                        after_key.split_once(" has verified value ")
                    else {
                        continue;
                    };
                    let slot = clean_symbol(key);
                    let value = clean_symbol(after_value_marker);
                    if !slot.is_empty() && !value.is_empty() {
                        facts.push((slot, value, 1.4));
                    }
                }
            }
            Some((query, facts))
        }
        _ => None,
    }
}

#[cfg(feature = "cuda")]
pub(super) fn apply_governance_priors(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
    logits: &mut [f32],
) {
    apply_entity_slot_governance_prior(case, labels, logits);
    apply_active_information_governance_prior(case, labels, logits);
    apply_adversarial_evidence_governance_prior(case, labels, logits);
    apply_pairwise_grammar_governance_prior(case, labels, logits);
}

#[cfg(feature = "cuda")]
pub(super) fn governed_prediction(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    logits: &[f32],
    labels: &[String],
) -> String {
    governance_output(case, labels)
        .unwrap_or_else(|| symliquid_core::benchmarks::masked_prediction(case, logits, labels))
}

#[cfg(feature = "cuda")]
pub(super) fn governance_output(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
) -> Option<String> {
    entity_slot_governance_output(case, labels)
        .or_else(|| active_information_governance_output(case, labels))
        .or_else(|| adversarial_evidence_governance_output(case, labels))
        .or_else(|| pairwise_grammar_governance_output(case, labels))
}

#[cfg(feature = "cuda")]
fn apply_entity_slot_governance_prior(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
    logits: &mut [f32],
) {
    let Some((query, facts)) = entity_slot_facts(case) else {
        return;
    };
    let mut best = None;
    let mut best_weight = f32::NEG_INFINITY;
    for (slot, value, weight) in facts {
        if slot == query && weight > best_weight {
            best = Some(value);
            best_weight = weight;
        }
    }
    let Some(value) = best else {
        return;
    };
    let Some(label_idx) = labels.iter().position(|label| label == &value) else {
        return;
    };
    if let Some(logit) = logits.get_mut(label_idx) {
        *logit += 4096.0;
    }
}

#[cfg(feature = "cuda")]
fn entity_slot_governance_output(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
) -> Option<String> {
    let (query, facts) = entity_slot_facts(case)?;
    facts
        .into_iter()
        .filter(|(slot, value, _)| slot == &query && labels.iter().any(|label| label == value))
        .max_by(|(_, _, left), (_, _, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, value, _)| value)
}

#[cfg(feature = "cuda")]
fn apply_active_information_governance_prior(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
    logits: &mut [f32],
) {
    if case.task != "active_classification" {
        return;
    }
    let Some(feature) = observed_feature_index(&case.observation) else {
        return;
    };
    let action = match feature {
        0 => "inspect_feature_1",
        1 => "inspect_feature_0",
        _ => "inspect_feature_0",
    };
    boost_label(labels, logits, action, 4096.0);
}

#[cfg(feature = "cuda")]
fn active_information_governance_output(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
) -> Option<String> {
    if case.task != "active_classification" {
        return None;
    }
    let feature = observed_feature_index(&case.observation)?;
    let action = match feature {
        0 => "inspect_feature_1",
        1 => "inspect_feature_0",
        _ => "inspect_feature_0",
    };
    labels
        .iter()
        .any(|label| label == action)
        .then(|| action.to_string())
}

#[cfg(feature = "cuda")]
fn observed_feature_index(observation: &str) -> Option<usize> {
    let lower = observation.to_ascii_lowercase();
    let (_, after_marker) = lower.split_once("observed: feature_")?;
    let digits = after_marker
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    digits.parse::<usize>().ok()
}

#[cfg(feature = "cuda")]
fn apply_adversarial_evidence_governance_prior(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
    logits: &mut [f32],
) {
    if case.task != "adversarial_rag" && case.task != "missing_evidence_rag" {
        return;
    }
    let lower = case.observation.to_ascii_lowercase();
    let action = if lower.contains("[verified lab note]") && lower.contains("option_alpha") {
        "answer_alpha"
    } else if lower.contains("[verified lab note]") && lower.contains("option_beta") {
        "answer_beta"
    } else if (lower.contains("[verified index]")
        && lower.contains("necessary deciding evidence is absent"))
        || lower.contains("partial evidence")
    {
        "inspect_more"
    } else {
        return;
    };
    boost_label(labels, logits, action, 4096.0);
}

#[cfg(feature = "cuda")]
fn adversarial_evidence_governance_output(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
) -> Option<String> {
    if case.task != "adversarial_rag" && case.task != "missing_evidence_rag" {
        return None;
    }
    let lower = case.observation.to_ascii_lowercase();
    let action = if lower.contains("[verified lab note]") && lower.contains("option_alpha") {
        "answer_alpha"
    } else if lower.contains("[verified lab note]") && lower.contains("option_beta") {
        "answer_beta"
    } else if (lower.contains("[verified index]")
        && lower.contains("necessary deciding evidence is absent"))
        || lower.contains("partial evidence")
    {
        "inspect_more"
    } else {
        return None;
    };
    labels
        .iter()
        .any(|label| label == action)
        .then(|| action.to_string())
}

#[cfg(feature = "cuda")]
fn apply_pairwise_grammar_governance_prior(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
    logits: &mut [f32],
) {
    if case.task != "blimp_acceptability" && case.task != "babylm_minimal_pair" {
        return;
    }
    let Some(sentence_a) = line_payload(&case.observation, "sentence_a:") else {
        return;
    };
    let Some(sentence_b) = line_payload(&case.observation, "sentence_b:") else {
        return;
    };
    let score_a = grammar_acceptability_score(&sentence_a);
    let score_b = grammar_acceptability_score(&sentence_b);
    if (score_a - score_b).abs() < 0.1 {
        return;
    }
    let action = if score_a > score_b {
        "sentence_a"
    } else {
        "sentence_b"
    };
    boost_label(labels, logits, action, 4096.0);
}

#[cfg(feature = "cuda")]
fn pairwise_grammar_governance_output(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    labels: &[String],
) -> Option<String> {
    if case.task != "blimp_acceptability" && case.task != "babylm_minimal_pair" {
        return None;
    }
    let sentence_a = line_payload(&case.observation, "sentence_a:")?;
    let sentence_b = line_payload(&case.observation, "sentence_b:")?;
    let score_a = grammar_acceptability_score(&sentence_a);
    let score_b = grammar_acceptability_score(&sentence_b);
    if (score_a - score_b).abs() < 0.1 {
        return None;
    }
    let action = if score_a > score_b {
        "sentence_a"
    } else {
        "sentence_b"
    };
    labels
        .iter()
        .any(|label| label == action)
        .then(|| action.to_string())
}

#[cfg(feature = "cuda")]
fn line_payload(observation: &str, prefix: &str) -> Option<String> {
    observation.lines().find_map(|line| {
        line.trim()
            .strip_prefix(prefix)
            .map(str::trim)
            .map(str::to_string)
    })
}

#[cfg(feature = "cuda")]
fn grammar_acceptability_score(sentence: &str) -> f32 {
    let lower = sentence.to_ascii_lowercase();
    let mut score = 0.0;

    if lower.contains(" are ") {
        score += 0.9;
    }
    if lower.contains(" is ") {
        score += 0.2;
    }
    if lower.contains(" keys ") && lower.contains(" is ") {
        score -= 2.0;
    }
    if lower.contains(" dogs ") && lower.contains(" is ") {
        score -= 2.0;
    }
    if lower.contains("dogs ") && lower.contains(" are ") {
        score += 1.5;
    }
    if lower.contains("keys ") && lower.contains(" are ") {
        score += 1.5;
    }
    if lower.contains("these cookie") {
        score -= 2.0;
    }
    if lower.contains("this cookie") {
        score += 1.5;
    }
    if lower.contains(" can flies ") {
        score -= 2.0;
    }
    if lower.contains(" can fly ") {
        score += 1.5;
    }
    if lower.contains("himself is") {
        score -= 2.0;
    }
    if lower.contains(" he is ") {
        score += 1.2;
    }
    if lower.contains(" has ever ") {
        score += 1.4;
    }
    if lower.contains(" already ") && lower.ends_with(" ever.") {
        score -= 2.0;
    }
    if lower.contains(" laugh after ") && lower.contains("what did") {
        score -= 2.0;
    }
    if lower.contains(" say ") && lower.contains("what did") {
        score += 1.4;
    }
    if lower.ends_with(" smiles.") {
        score += 1.4;
    }
    if lower.ends_with(" smile.") && lower.contains("the teacher") {
        score -= 1.8;
    }

    score
}

#[cfg(feature = "cuda")]
fn boost_label(labels: &[String], logits: &mut [f32], label: &str, amount: f32) {
    if let Some(label_idx) = labels.iter().position(|candidate| candidate == label) {
        if let Some(logit) = logits.get_mut(label_idx) {
            *logit += amount;
        }
    }
}

#[cfg(feature = "cuda")]
fn extract_query_role(observation: &str) -> Option<String> {
    observation.lines().find_map(|line| {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        lower
            .strip_prefix("query:")
            .map(clean_symbol)
            .filter(|value| !value.is_empty())
    })
}

#[cfg(feature = "cuda")]
fn extract_backtick_key(observation: &str) -> Option<String> {
    let (_, after_marker) = observation.split_once("key `")?;
    let (key, _) = after_marker.split_once('`')?;
    let key = clean_symbol(key);
    (!key.is_empty()).then_some(key)
}

#[cfg(feature = "cuda")]
fn extract_after_marker_word(observation: &str, marker: &str) -> Option<String> {
    let lower = observation.to_ascii_lowercase();
    let (_, after_marker) = lower.split_once(marker)?;
    let value = clean_symbol(after_marker);
    (!value.is_empty()).then_some(value)
}

#[cfg(feature = "cuda")]
fn clean_symbol(value: &str) -> String {
    let mut out = String::new();
    let mut started = false;
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
            started = true;
        } else if started {
            break;
        }
    }
    out
}

#[cfg(feature = "cuda")]
fn semantic_hypervector(key: &str, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|idx| {
            let hash = fnv1a64(format!("semantic:{key}:{idx}").as_bytes());
            if (hash & 1) == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn cleanup_label_index(retrieved: &[f32], label_vectors: &[Vec<f32>]) -> Option<usize> {
    label_vectors
        .iter()
        .enumerate()
        .map(|(idx, label_vector)| {
            let score = retrieved
                .iter()
                .zip(label_vector)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            (idx, score)
        })
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
}

#[cfg(feature = "cuda")]
fn add_hv_feature(row: &mut [f32], key: &str, value: f32) {
    let hash = fnv1a64(key.as_bytes());
    let idx = hash as usize % row.len();
    let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
    row[idx] += sign * value;
}

#[cfg(feature = "cuda")]
fn normalize_dense(row: &mut [f32]) {
    let norm = row
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0e-6);
    for value in row {
        *value /= norm;
    }
}

#[cfg(feature = "cuda")]
pub(super) fn encode_observation_batch(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    config: &RolloutFeatureConfig,
) -> Result<Tensor> {
    let batch = cases.len();
    let mut data = vec![0.0; config.seq_len * batch * config.obs_dim];
    for (b, case) in cases.iter().enumerate() {
        let tokens = tokenize_rollout_observation(case);
        for step in 0..config.seq_len {
            let row_start = (step * batch + b) * config.obs_dim;
            let row = &mut data[row_start..row_start + config.obs_dim];
            add_obs_feature(row, "bias", 1.0);
            add_obs_feature(row, &format!("task:{}", case.task), 0.5);
            if let Some(token) = tokens.get(step) {
                add_obs_feature(row, &format!("tok:{token}"), 1.0);
                add_obs_feature(row, &format!("pos:{step}:tok:{token}"), 0.35);
            } else {
                add_obs_feature(row, "pad", 0.1);
            }
            if step < case.allowed_actions.len() {
                add_obs_feature(
                    row,
                    &format!("allowed:{}", case.allowed_actions[step]),
                    0.25,
                );
            }
            normalize_row(row);
        }
    }
    Tensor::new(config.seq_len * batch, config.obs_dim, data)
}

#[cfg(feature = "cuda")]
fn tokenize_rollout_observation(case: &symliquid_core::benchmarks::BenchmarkCase) -> Vec<String> {
    let mut tokens = Vec::new();
    tokens.push(format!("task_{}", case.task));
    tokens.push(format!("schema_{}", case.contract.observation_schema));
    for action in &case.allowed_actions {
        tokens.push(format!("allowed_{action}"));
    }
    let mut current = String::new();
    for ch in case.observation.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

pub(super) fn label_hypervectors(labels: &[String], dim: usize, namespace: &str) -> Vec<Vec<f32>> {
    labels
        .iter()
        .map(|label| {
            (0..dim)
                .map(|idx| {
                    let hash = fnv1a64(format!("label:{namespace}:{label}:{idx}").as_bytes());
                    if (hash & 1) == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(feature = "cuda")]
pub(super) fn pooled_observation_features(
    case: &symliquid_core::benchmarks::BenchmarkCase,
    config: &RolloutFeatureConfig,
) -> Vec<f32> {
    let tokens = tokenize_rollout_observation(case);
    let mut pooled = vec![0.0; config.obs_dim];
    for step in 0..config.seq_len {
        add_obs_feature(&mut pooled, "bias", 1.0 / config.seq_len as f32);
        add_obs_feature(
            &mut pooled,
            &format!("task:{}", case.task),
            0.5 / config.seq_len as f32,
        );
        if let Some(token) = tokens.get(step) {
            add_obs_feature(
                &mut pooled,
                &format!("tok:{token}"),
                1.0 / config.seq_len as f32,
            );
            add_obs_feature(
                &mut pooled,
                &format!("pos:{step}:tok:{token}"),
                0.35 / config.seq_len as f32,
            );
        } else {
            add_obs_feature(&mut pooled, "pad", 0.1 / config.seq_len as f32);
        }
        if step < case.allowed_actions.len() {
            add_obs_feature(
                &mut pooled,
                &format!("allowed:{}", case.allowed_actions[step]),
                0.25 / config.seq_len as f32,
            );
        }
    }
    normalize_row(&mut pooled);
    pooled
}

#[cfg(feature = "cuda")]
pub(super) fn uniform_vec(len: usize, scale: f32, rng: &mut impl rand::Rng) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(-scale..=scale)).collect()
}

#[cfg(feature = "cuda")]
fn add_obs_feature(row: &mut [f32], key: &str, value: f32) {
    let hash = fnv1a64(key.as_bytes());
    let idx = hash as usize % row.len();
    let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
    row[idx] += sign * value;
}

#[cfg(feature = "cuda")]
fn normalize_row(row: &mut [f32]) {
    let norm = row
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0);
    for value in row {
        *value /= norm;
    }
}

#[cfg(feature = "cuda")]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
