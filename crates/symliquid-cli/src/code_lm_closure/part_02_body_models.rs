use super::*;
use super::part_02_beam_decode::BeamState;

pub(super) fn build_body_ngram_model(tasks: &[CodeTask]) -> BodyNgramModel {
    let mut model = BodyNgramModel::default();
    for task in tasks {
        let repeats = if semantic_hard_target_category(&task.category) {
            symliquid_training_repeats(task)
        } else {
            1
        };
        for _ in 0..repeats {
            let mut tokens = solution_body_tokens(task);
            if tokens.is_empty() {
                continue;
            }
            tokens.push("<EOS>".to_string());
            let mut prev2 = "<BOS>".to_string();
            let mut prev1 = "<BOS>".to_string();
            for (position, token) in tokens.into_iter().enumerate() {
                for key in body_ngram_keys(task, &prev2, &prev1, position) {
                    *model
                        .counts
                        .entry(key)
                        .or_default()
                        .entry(token.clone())
                        .or_insert(0) += 1;
                }
                prev2 = prev1;
                prev1 = token;
            }
        }
    }
    model
}

pub(super) fn build_body_prototype_model(tasks: &[CodeTask]) -> BodyPrototypeModel {
    let mut counts: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for task in tasks {
        let body = task.solution_body_text();
        if useful_generated_body_for_task(task, &body) && syntax_constrained_body(&body) {
            for key in body_prototype_keys(task) {
                *counts
                    .entry(key)
                    .or_default()
                    .entry(body.clone())
                    .or_insert(0) += 1;
            }
        }
    }
    let by_category = counts
        .into_iter()
        .map(|(category, rows)| {
            let mut bodies = rows.into_iter().collect::<Vec<_>>();
            bodies.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            (
                category,
                bodies.into_iter().map(|(body, _)| body).collect::<Vec<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    BodyPrototypeModel { by_category }
}

pub(super) fn body_prototype_bodies(
    task: &CodeTask,
    model: &BodyPrototypeModel,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<String> {
    let prompt_tokens = prompt_tokens_with_sts(task, sts_streams);
    let stream_tokens = prototype_stream_tokens(sts_streams);
    let low_latency =
        task.split != "public_calibration" && low_latency_candidate_fanout_enabled() && limit <= 8;
    let low_latency_scan_limit = if low_latency {
        std::env::var("THESEUS_CODE_LM_LOW_LATENCY_PROTOTYPE_SCAN_PER_KEY")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .map(|value| value.clamp(limit.max(1), 256))
            .unwrap_or_else(|| limit.saturating_mul(8).clamp(24, 96))
    } else {
        usize::MAX
    };
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    let category_priority = body_prototype_key_priority(task);
    for key in body_prototype_keys(task) {
        let Some(candidates) = model.by_category.get(&key) else {
            continue;
        };
        let key_priority = category_priority.get(&key).copied().unwrap_or(0.0);
        for body in candidates.iter().take(low_latency_scan_limit) {
            if !seen.insert(body.clone()) || !syntax_constrained_body(body) {
                continue;
            }
            let body_tokens = tokenize_body(body)
                .into_iter()
                .map(|token| token.to_lowercase())
                .collect::<BTreeSet<_>>();
            let overlap = body_tokens.intersection(&prompt_tokens).count() as f32;
            let stream_bonus = prototype_stream_bonus_from_tokens(&body_tokens, &stream_tokens);
            let contract_score = contract_family_overlap_score(task, body);
            rows.push((
                key_priority + overlap + stream_bonus + contract_score,
                body.clone(),
            ));
        }
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    rows.into_iter().take(limit).map(|(_, body)| body).collect()
}

fn body_prototype_key_priority(task: &CodeTask) -> BTreeMap<String, f32> {
    let mut priorities = BTreeMap::new();
    for category in normalized_category_keys(&task.category) {
        priorities.insert(format!("category:{category}"), 25.0);
    }
    let shape = decoder_return_shape(task);
    let family = decoder_type_family(task);
    if shape != "unknown" && family != "unknown" {
        priorities.insert(format!("contract:{family}:{shape}"), 8.0);
    }
    if family != "unknown" {
        priorities.insert(format!("type_family:{family}"), 5.0);
    }
    if shape != "unknown" {
        priorities.insert(format!("return_shape:{shape}"), 3.0);
    }
    for required in decoder_required_constructs(task).into_iter().take(16) {
        priorities.entry(format!("requires:{required}")).or_insert(1.0);
    }
    priorities
}

pub(super) fn body_prototype_keys(task: &CodeTask) -> Vec<String> {
    let mut keys = BTreeSet::new();
    for category in normalized_category_keys(&task.category) {
        keys.insert(format!("category:{category}"));
    }
    let shape = decoder_return_shape(task);
    if shape != "unknown" {
        keys.insert(format!("return_shape:{shape}"));
    }
    let family = decoder_type_family(task);
    if family != "unknown" {
        keys.insert(format!("type_family:{family}"));
    }
    if shape != "unknown" && family != "unknown" {
        keys.insert(format!("contract:{family}:{shape}"));
    }
    for required in decoder_required_constructs(task).into_iter().take(16) {
        keys.insert(format!("requires:{required}"));
    }
    if execution_shaped_category(&task.category)
        || decoder_required_constructs(task).contains("execution_shaped_program")
    {
        keys.insert("lane:execution_shape".to_string());
    }
    keys.into_iter().collect()
}

pub(super) fn normalized_category_keys(category: &str) -> Vec<String> {
    let category = category.trim();
    if category.is_empty() {
        return vec!["general_semantics".to_string()];
    }
    let mut keys = BTreeSet::new();
    keys.insert(category.to_string());
    for prefix in ["private_exec_", "private_", "source_"] {
        if let Some(stripped) = category.strip_prefix(prefix) {
            if !stripped.is_empty() {
                keys.insert(stripped.to_string());
            }
        }
    }
    if category == "general_semantics" {
        keys.insert("general".to_string());
    }
    keys.into_iter().collect()
}

fn contract_family_overlap_score(task: &CodeTask, body: &str) -> f32 {
    let lowered = body.to_lowercase();
    let mut score = 0.0;
    let required = decoder_required_constructs(task);
    for (hint, needles) in [
        ("loop", vec!["for ", "while "]),
        ("branch", vec!["if ", "try:", "except "]),
        ("locals", vec![" = "]),
        ("archive", vec!["zipfile", "tarfile", "shutil", "make_archive"]),
        ("csv", vec!["csv.", "csvreader", "reader("]),
        ("structured_parsing", vec!["json", "payload", "load("]),
        ("file_path", vec!["os.path", "open(", "exists", "isfile", "isdir"]),
        ("collection_ops", vec!["append", "set(", "list(", "dict(", "sorted"]),
    ] {
        if required.contains(hint) && needles.iter().any(|needle| lowered.contains(needle)) {
            score += 0.7;
        }
    }
    score
}

fn prototype_stream_tokens(sts_streams: Option<&BTreeMap<String, String>>) -> Vec<String> {
    let mut tokens = Vec::new();
    let Some(streams) = sts_streams else {
        return tokens;
    };
    for stream in ["solver_stream", "patch_stream", "critic_stream"] {
        let Some(text) = streams.get(stream) else {
            continue;
        };
        for token in tokenize_code(text) {
            tokens.push(token.to_lowercase());
        }
    }
    tokens
}

fn prototype_stream_bonus_from_tokens(
    body_tokens: &BTreeSet<String>,
    stream_tokens: &[String],
) -> f32 {
    if stream_tokens.is_empty() {
        return 0.0;
    }
    let mut score = 0.0;
    for token in stream_tokens {
        if body_tokens.contains(token) {
            score += 0.4;
        }
    }
    score
}

fn fast_contract_transduction_prefilter_enabled(task: &CodeTask, limit: usize) -> bool {
    if task.split == "public_calibration" || limit > 8 {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_FAST_CONTRACT_TRANSDUCTION_PREFILTER")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn fast_contract_transduction_candidate_ok(
    task: &CodeTask,
    body: &str,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> bool {
    if !useful_generated_body_for_task(task, body) || !syntax_constrained_body(body) {
        return false;
    }
    let lowered = body.to_ascii_lowercase();
    let hints = decoder_required_constructs(task);
    visible_argument_contract_ok(task, body)
        || return_shape_contract_ok(task, &lowered)
        || required_construct_contract_ok_for_task(task, body, &hints)
        || body_semantically_admissible(task, body)
        || sts_streams.is_some_and(|streams| sts_skeleton_alignment_score(body, Some(streams)) > 0.5)
}

pub(super) fn learned_contract_transduction_bodies(
    task: &CodeTask,
    model: &BodyPrototypeModel,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let fast_prefilter = fast_contract_transduction_prefilter_enabled(task, limit);
    let low_latency =
        task.split != "public_calibration" && low_latency_candidate_fanout_enabled() && limit <= 8;
    let prototype_budget = if low_latency {
        std::env::var("THESEUS_CODE_LM_LOW_LATENCY_TRANSDUCTION_PROTOTYPE_BUDGET")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .map(|value| value.clamp(limit.max(1), 64))
            .unwrap_or_else(|| limit.saturating_mul(4).clamp(12, 32))
    } else {
        limit.saturating_mul(10).max(24)
    };
    for body in body_prototype_bodies(task, model, prototype_budget, sts_streams) {
        let adapted = canonicalize_task_candidate_body_aliases(task, &body);
        for variant in state_sequence_body_variants(task, &adapted) {
            let repaired = canonicalize_task_candidate_body_aliases(task, &variant);
            if !seen.insert(repaired.clone()) {
                continue;
            }
            let accepted = if fast_prefilter {
                fast_contract_transduction_candidate_ok(task, &repaired, sts_streams)
            } else {
                token_contract_candidate_body_ok(task, &repaired, sts_streams)
            };
            if accepted {
                out.push(repaired);
            }
            if out.len() >= limit {
                return out;
            }
        }
    }
    out
}

pub(super) fn body_ngram_bodies(
    task: &CodeTask,
    model: &BodyNgramModel,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<String> {
    if model.counts.is_empty() {
        return Vec::new();
    }
    let mut out = dominant_body_ngram_bodies(task, model, seed, limit);
    let mut seen = out.iter().cloned().collect::<HashSet<_>>();
    if out.len() >= limit {
        return out;
    }
    let prompt = prompt_tokens_with_sts(task, sts_streams)
        .into_iter()
        .collect::<Vec<_>>();
    let mut beams = vec![BeamState {
        tokens: Vec::new(),
        prev2: "<BOS>".to_string(),
        prev1: "<BOS>".to_string(),
        score: 0.0,
        finished: false,
    }];
    let beam_width = if execution_shaped_category(&task.category) {
        limit.clamp(2, 4)
    } else if task.split == "public_calibration" {
        limit.clamp(2, 4)
    } else {
        limit.clamp(2, 4)
    };
    let max_steps = learned_body_ngram_max_steps(
        task,
        if task.split == "public_calibration" {
            40
        } else {
            40
        },
    );
    for position in 0..max_steps {
        let mut next = Vec::new();
        for beam in &beams {
            if beam.finished {
                next.push(beam.clone());
                continue;
            }
            let options = if let Some(forced) = forced_block_token_options(&beam.tokens) {
                forced
            } else {
                let option_limit = if execution_shaped_category(&task.category) {
                    5
                } else if task.split == "public_calibration" {
                    6
                } else {
                    6
                };
                body_ngram_token_options(
                    task,
                    model,
                    &beam.prev2,
                    &beam.prev1,
                    &prompt,
                    position,
                    option_limit,
                )
            };
            for (token, score) in options {
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
                        "body-ngram:{}:{}:{:?}",
                        seed, task.task_id, a.tokens
                    ))
                    .cmp(&stable_hash_u64(&format!(
                        "body-ngram:{}:{}:{:?}",
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
            if body_ngram_candidate_body_ok(task, &candidate_body)
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
    out
}

fn dominant_body_ngram_bodies(
    task: &CodeTask,
    model: &BodyNgramModel,
    seed: u64,
    limit: usize,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    if let Some(body) = greedy_dominant_body_ngram_body(task, model) {
        if seen.insert(body.clone()) {
            out.push(body);
        }
    }
    if out.len() >= limit {
        return out;
    }
    let mut beams = vec![BeamState {
        tokens: Vec::new(),
        prev2: "<BOS>".to_string(),
        prev1: "<BOS>".to_string(),
        score: 0.0,
        finished: false,
    }];
    let beam_width = if execution_shaped_category(&task.category) {
        limit.clamp(2, 4)
    } else if task.split == "public_calibration" {
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
    for position in 0..max_steps {
        let mut next = Vec::new();
        for beam in &beams {
            if beam.finished {
                next.push(beam.clone());
                continue;
            }
            let options = if let Some(forced) = forced_block_token_options(&beam.tokens) {
                forced
            } else {
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
                options
            };
            let option_cap = if execution_shaped_category(&task.category) {
                6
            } else if task.split == "public_calibration" {
                5
            } else {
                5
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
                        "dominant-body-ngram:{}:{}:{:?}",
                        seed, task.task_id, a.tokens
                    ))
                    .cmp(&stable_hash_u64(&format!(
                        "dominant-body-ngram:{}:{}:{:?}",
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
            if body_ngram_candidate_body_ok(task, &candidate_body)
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
    out
}

fn greedy_dominant_body_ngram_body(task: &CodeTask, model: &BodyNgramModel) -> Option<String> {
    let mut tokens = Vec::new();
    let mut prev2 = "<BOS>".to_string();
    let mut prev1 = "<BOS>".to_string();
    for position in 0..learned_body_ngram_max_steps(task, 96) {
        let options = if let Some(forced) = forced_block_token_options(&tokens) {
            forced
        } else {
            let mut options =
                body_ngram_category_token_scores(task, model, &prev2, &prev1, position)
                    .into_iter()
                    .collect::<Vec<_>>();
            options.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            options
        };
        let Some((token, _)) = options
            .into_iter()
            .find(|(token, _)| task_body_token_allowed(task, &tokens, token))
        else {
            break;
        };
        if token == "<EOS>" {
            break;
        }
        tokens.push(token.clone());
        prev2 = prev1;
        prev1 = token;
    }
    let body = join_body_tokens(&tokens);
    for candidate in state_sequence_body_variants(task, &body) {
        if body_ngram_candidate_body_ok(task, &candidate) {
            return Some(candidate);
        }
    }
    None
}

fn body_ngram_candidate_body_ok(task: &CodeTask, body: &str) -> bool {
    state_sequence_candidate_body_ok(task, body)
        && (!contract_requires_extended_learned_body(task) || body_has_top_level_return(body))
}

fn body_has_top_level_return(body: &str) -> bool {
    body.lines()
        .any(|line| line.starts_with("return ") || line.trim() == "return")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_prototype_retrieval_prefers_exact_category_over_broad_contract() {
        let exact_body = "\
keep = []
drop = []
threshold = other
for value in data:
    if value >= threshold:
        keep.append(value)
    else:
        drop.append(value)
return (keep, drop)"
            .to_string();
        let broad_body = "\
out = []
for pair in data:
    if isinstance(pair, (list, tuple)):
        out.append(pair)
return tuple(out)"
            .to_string();
        let model = build_body_prototype_model(&[
            test_task("edge_v3_partition_tuple_shape", &exact_body),
            test_task("edge_v3_broad_tuple_shape", &broad_body),
        ]);
        let query = test_task("edge_v3_partition_tuple_shape", "");
        let bodies = body_prototype_bodies(&query, &model, 1, None);
        assert_eq!(bodies, vec![exact_body]);
    }

    #[test]
    fn body_ngram_replays_edge_v3_category_conditioned_body() {
        let exact_body = "\
keep = []
drop = []
threshold = other if other is not None else 0
for value in data:
    if value >= threshold:
        keep.append(value)
    else:
        drop.append(value)
return (keep, drop)"
            .to_string();
        let model = build_body_ngram_model(&[test_task(
            "edge_v3_partition_tuple_shape",
            &exact_body,
        )]);
        let query = test_task("edge_v3_partition_tuple_shape", "");
        let prompt = prompt_tokens_with_sts(&query, None)
            .into_iter()
            .collect::<Vec<_>>();
        let options = body_ngram_token_options(
            &query,
            &model,
            "other",
            "if",
            &prompt,
            14,
            8,
        );
        assert!(
            options.iter().any(|(token, _)| token == "other"),
            "expected inline conditional continuation option after `other if`, got {options:?}"
        );
        let prefix = vec![
            "keep".to_string(),
            "=".to_string(),
            "[".to_string(),
            "]".to_string(),
            "<NL>".to_string(),
            "drop".to_string(),
            "=".to_string(),
            "[".to_string(),
            "]".to_string(),
            "<NL>".to_string(),
            "threshold".to_string(),
            "=".to_string(),
            "other".to_string(),
            "if".to_string(),
        ];
        assert!(
            task_body_token_allowed(&query, &prefix, "other"),
            "grammar mask should allow inline conditional continuation after `other if`"
        );
        let mut inline_prefix = prefix.clone();
        inline_prefix.push("other".to_string());
        assert!(
            task_body_token_allowed(&query, &inline_prefix, "is"),
            "grammar mask should allow inline conditional `is` after `other if other`"
        );
        let bodies = body_ngram_bodies(&query, &model, 7, 4, None);
        assert!(
            bodies.iter().any(|body| body.contains("return (keep, drop)")),
            "expected learned category ngram body in {bodies:?}"
        );
    }

    #[test]
    fn body_ngram_replays_edge_v3_try_except_same_container_body() {
        let exact_body = "\
items = []
for value in data:
    try:
        items.append(int(str(value).strip()))
    except Exception:
        items.append(0)
if isinstance(data, tuple):
    return tuple(items)
return items"
            .to_string();
        let model = build_body_ngram_model(&[test_task(
            "edge_v3_same_container_transform",
            &exact_body,
        )]);
        let query = test_task("edge_v3_same_container_transform", "");
        let bodies = body_ngram_bodies(&query, &model, 11, 4, None);
        assert!(
            bodies.iter().any(|body| {
                let compact = body.replace(' ', "");
                body.contains("except Exception")
                    && compact.contains("returntuple(items)")
                    && body.contains("return items")
            }),
            "expected learned try/except same-container body in {bodies:?}"
        );
    }

    #[test]
    fn body_ngram_replays_edge_v3_weighted_interval_body() {
        let exact_body = "\
jobs = []
for item in data:
    if len(item) >= 3:
        start, end, weight = item[0], item[1], item[2]
        if start > end:
            start, end = end, start
        jobs.append((end, start, weight))
jobs.sort()
dp = [0]
ends = []
for end, start, weight in jobs:
    lo, hi = 0, len(ends)
    while lo < hi:
        mid = (lo + hi) // 2
        if ends[mid] <= start:
            lo = mid + 1
        else:
            hi = mid
    best = dp[lo] + weight
    dp.append(max(dp[-1], best))
    ends.append(end)
return dp[-1]"
            .to_string();
        let task = test_contract_task(
            "edge_v3_weighted_interval_best",
            "algorithmic_planning",
            "number",
            vec![
                "loop",
                "branch",
                "locals",
                "binary_search",
                "dynamic_programming",
            ],
            "weighted_interval_dynamic_programming",
            &exact_body,
        );
        let model = build_body_ngram_model(&[task]);
        let query = test_contract_task(
            "edge_v3_weighted_interval_best",
            "algorithmic_planning",
            "number",
            vec![
                "loop",
                "branch",
                "locals",
                "binary_search",
                "dynamic_programming",
            ],
            "weighted_interval_dynamic_programming",
            "",
        );
        let bodies = body_ngram_bodies(&query, &model, 13, 4, None);
        assert!(
            bodies.iter().any(|body| {
                let compact = body.replace(' ', "");
                compact.contains("jobs.sort()")
                    && body.contains("while lo < hi")
                    && compact.contains("dp.append(max(dp[-1],best))")
                    && compact.contains("returndp[-1]")
            }),
            "expected learned weighted-interval body in {bodies:?}"
        );
    }

    #[test]
    fn body_ngram_replays_edge_v3_state_machine_bodies() {
        let balance_body = "\
floor, ceiling = other
balance = 0
out = []
for delta in data:
    balance += delta
    if balance < floor:
        balance = floor
    if balance > ceiling:
        balance = ceiling
    out.append(balance)
return out"
            .to_string();
        let stack_body = "\
inverse = other if isinstance(other, dict) else {}
stack = []
for token in data:
    if stack and inverse.get(token) == stack[-1]:
        stack.pop()
    else:
        stack.append(token)
return stack"
            .to_string();
        let balance_assignment_prefix = vec![
            "balance".to_string(),
            "=".to_string(),
            "ceiling".to_string(),
        ];
        assert!(
            task_body_token_allowed(
                &test_contract_task(
                    "edge_v3_capped_running_balance",
                    "state_machine",
                    "list",
                    vec!["loop", "branch", "locals", "state_update"],
                    "bounded_state_update",
                    "",
                ),
                &balance_assignment_prefix,
                "<NL>",
            ),
            "state-local assignment should be allowed to finish as its own line"
        );
        assert!(
            !task_body_token_allowed(
                &test_contract_task(
                    "edge_v3_capped_running_balance",
                    "state_machine",
                    "list",
                    vec!["loop", "branch", "locals", "state_update"],
                    "bounded_state_update",
                    "",
                ),
                &balance_assignment_prefix,
                ".",
            ),
            "state-local assignment should not glue the next method call onto the RHS"
        );
        let repaired_balance_body = normalize_generated_body(
            "\
floor, ceiling = other
balance = 0
out = []
for delta in data:
    balance += delta
    if balance < floor:
        balance = floor
    if balance > ceiling:
        balance = ceiling.out.append(balance)
return out",
        );
        assert!(
            repaired_balance_body.contains("        balance = ceiling\n    out.append(balance)"),
            "glued state assignment append should split and dedent: {repaired_balance_body}"
        );
        assert!(
            decoder_contract_verifier_v1(
                &test_contract_task(
                    "edge_v3_capped_running_balance",
                    "state_machine",
                    "list",
                    vec!["loop", "branch", "locals", "state_update"],
                    "bounded_state_update",
                    "",
                ),
                &repaired_balance_body,
                None,
            )
            .passed,
            "repaired capped balance body should satisfy the strict verifier"
        );
        for (category, body, semantic_family, required) in [
            (
                "edge_v3_capped_running_balance",
                balance_body.as_str(),
                "bounded_state_update",
                vec!["loop", "branch", "locals", "state_update"],
            ),
            (
                "edge_v3_stack_cancel_tokens",
                stack_body.as_str(),
                "stack_cancellation",
                vec!["loop", "branch", "locals", "stack"],
            ),
        ] {
            let task = test_contract_task(
                category,
                "state_machine",
                "list",
                required.clone(),
                semantic_family,
                body,
            );
            let model = build_body_ngram_model(&[task]);
            let query =
                test_contract_task(category, "state_machine", "list", required, semantic_family, "");
            let bodies = body_ngram_bodies(&query, &model, 17, 4, None);
            assert!(
                bodies.iter().any(|candidate| {
                    let compact = candidate.replace(' ', "");
                    if category == "edge_v3_capped_running_balance" {
                        candidate.contains("balance += delta")
                            && candidate.contains("out.append")
                            && compact.contains("returnout")
                    } else {
                        candidate.contains("inverse.get")
                            && compact.contains("stack.pop()")
                            && compact.contains("returnstack")
                    }
                }),
                "expected learned state body for {category} in {bodies:?}"
            );
        }
    }

    #[test]
    fn body_ngram_replays_edge_v3_graph_distance_body() {
        let exact_body = "\
start = other
from collections import deque
graph = {}
for a, b in data:
    graph.setdefault(a, []).append(b)
    graph.setdefault(b, []).append(a)
dist = {start: 0}
queue = deque([start])
while queue:
    node = queue.popleft()
    for nxt in graph.get(node, []):
        if nxt not in dist:
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
return dist"
            .to_string();
        let task = test_contract_task(
            "edge_v3_graph_distance_labels",
            "graph_search_algorithm",
            "dict",
            vec!["loop", "branch", "locals", "queue", "graph"],
            "graph_shortest_hops",
            &exact_body,
        );
        let model = build_body_ngram_model(&[task]);
        let query = test_contract_task(
            "edge_v3_graph_distance_labels",
            "graph_search_algorithm",
            "dict",
            vec!["loop", "branch", "locals", "queue", "graph"],
            "graph_shortest_hops",
            "",
        );
        let bodies = body_ngram_bodies(&query, &model, 19, 4, None);
        assert!(
            bodies.iter().any(|body| {
                let compact = body.replace(' ', "");
                body.contains("from collections import deque")
                    && compact.contains("graph.setdefault(a,[]).append(b)")
                    && body.contains("while queue")
                    && body.contains("queue.append")
                    && compact.contains("returndist")
            }),
            "expected learned graph-distance body in {bodies:?}"
        );
    }

    #[test]
    fn edge_v3_verifier_rejects_shape_correct_but_semantically_wrong_bodies() {
        let weighted_task = test_contract_task(
            "edge_v3_weighted_interval_best",
            "algorithmic_planning",
            "number",
            vec![
                "loop",
                "branch",
                "locals",
                "binary_search",
                "dynamic_programming",
            ],
            "weighted_interval_dynamic_programming",
            "",
        );
        let weighted_good = "\
jobs = []
for item in data:
    if len(item) >= 3:
        start, end, weight = item[0], item[1], item[2]
        if start > end:
            start, end = end, start
        jobs.append((end, start, weight))
jobs.sort()
dp = [0]
ends = []
for end, start, weight in jobs:
    lo, hi = 0, len(ends)
    while lo < hi:
        mid = (lo + hi) // 2
        if ends[mid] <= start:
            lo = mid + 1
        else:
            hi = mid
    best = dp[lo] + weight
    dp.append(max(dp[-1], best))
    ends.append(end)
return dp[-1]";
        let weighted_bad = "\
jobs = []
for item in data:
    if len(item) <= start:
        start, end, weight = item[0], item[1], item[2]
        jobs.append((end, start, weight))
jobs.sort()
dp = [0]
ends = []
for end, start, weight in jobs:
    lo, hi = 0, len(ends)
    while lo < hi:
        mid = (lo + hi) // 2
        if ends[mid] <= start:
            lo = mid + 1
        else:
            hi = mid
    best = dp
return best";
        let weighted_good_verifier =
            decoder_contract_verifier_v1(&weighted_task, weighted_good, None);
        assert!(
            weighted_good_verifier.passed,
            "exact weighted interval body should satisfy the edge-v3 contract: {:?}",
            weighted_good_verifier.reasons
        );
        assert!(
            !decoder_contract_verifier_v1(&weighted_task, weighted_bad, None).passed,
            "unbound/truncated weighted interval body should be rejected"
        );

        let graph_task = test_contract_task(
            "edge_v3_graph_distance_labels",
            "graph_search_algorithm",
            "dict",
            vec!["loop", "branch", "locals", "queue", "graph"],
            "graph_shortest_hops",
            "",
        );
        let graph_good = "\
start = other
from collections import deque
graph = {}
for a, b in data:
    graph.setdefault(a, []).append(b)
    graph.setdefault(b, []).append(a)
dist = {start: 0}
queue = deque([start])
while queue:
    node = queue.popleft()
    for nxt in graph.get(node, []):
        if nxt not in dist:
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
return dist";
        let graph_bad = "\
if data is None:
    return {}
out = {}
for item in data:
    out[item] = out.get(item, 0) + 1
out[other] = out.get(other, 0)
return out";
        let graph_good_verifier = decoder_contract_verifier_v1(&graph_task, graph_good, None);
        assert!(
            graph_good_verifier.passed,
            "BFS distance body should satisfy the dict return-shape and graph/queue contract: {:?}",
            graph_good_verifier.reasons
        );
        assert!(
            !decoder_contract_verifier_v1(&graph_task, graph_bad, None).passed,
            "shape-correct frequency body should not satisfy the graph distance contract"
        );

        let balance_task = test_contract_task(
            "edge_v3_capped_running_balance",
            "state_machine",
            "list",
            vec!["loop", "branch", "locals", "state_update"],
            "bounded_state_update",
            "",
        );
        let balance_good = "\
floor, ceiling = other
balance = 0
out = []
for delta in data:
    balance += delta
    if balance < floor:
        balance = floor
    if balance > ceiling:
        balance = ceiling
    out.append(balance)
return out";
        let balance_bad = "\
items = list(data) if isinstance(data, (list, tuple, set)) else [data]
out = []
for item in items:
    if item is None:
        continue
    out.append(item)
if other is not None:
    out.append(other)
return out";
        let balance_good_verifier =
            decoder_contract_verifier_v1(&balance_task, balance_good, None);
        assert!(
            balance_good_verifier.passed,
            "capped running balance body should satisfy the state update contract: {:?}",
            balance_good_verifier.reasons
        );
        assert!(
            !decoder_contract_verifier_v1(&balance_task, balance_bad, None).passed,
            "list-cleanup body should not satisfy the capped balance contract"
        );

        let stack_task = test_contract_task(
            "edge_v3_stack_cancel_tokens",
            "state_machine",
            "list",
            vec!["loop", "branch", "locals", "stack"],
            "stack_cancellation",
            "",
        );
        let stack_good = "\
inverse = other if isinstance(other, dict) else {}
stack = []
for token in data:
    if stack and inverse.get(token) == stack[-1]:
        stack.pop()
    else:
        stack.append(token)
return stack";
        let stack_bad = "\
items = list(data) if isinstance(data, (list, tuple, set)) else []
out = []
for item in items:
    if item is None:
        continue
    out.append(item)
if other is not None and out:
    out[-1] = other
return out";
        let stack_good_verifier = decoder_contract_verifier_v1(&stack_task, stack_good, None);
        assert!(
            stack_good_verifier.passed,
            "inverse stack cancellation body should satisfy the stack contract: {:?}",
            stack_good_verifier.reasons
        );
        assert!(
            !decoder_contract_verifier_v1(&stack_task, stack_bad, None).passed,
            "generic list body should not satisfy the stack cancellation contract"
        );
    }

    fn test_task(category: &str, body: &str) -> CodeTask {
        CodeTask {
            raw: json!({
                "decoder_contract": {
                    "return_shape": "tuple",
                    "type_family": "return_shape_contract",
                    "required_constructs": ["loop", "branch", "locals", "type_and_return_shape"]
                }
            }),
            task_id: format!("unit_{category}"),
            source_task_id: "unit".to_string(),
            card_id: "edge_contract_v3".to_string(),
            source_id: "unit".to_string(),
            split: "eval".to_string(),
            category: category.to_string(),
            prompt: "Partition values by predicate and always return a two-list tuple."
                .to_string(),
            entry_point: category.to_string(),
            solution_expr: String::new(),
            solution_body: body.to_string(),
            tags: vec![],
            benchmark_evidence_level: "private_generated_unit".to_string(),
        }
    }

    fn test_contract_task(
        category: &str,
        type_family: &str,
        return_shape: &str,
        required_constructs: Vec<&str>,
        semantic_family: &str,
        body: &str,
    ) -> CodeTask {
        let prompt_signature = if category == "edge_v3_weighted_interval_best" {
            format!("def {category}(data):")
        } else {
            format!("def {category}(data, other):")
        };
        CodeTask {
            raw: json!({
                "decoder_contract": {
                    "return_shape": return_shape,
                    "type_family": type_family,
                    "required_constructs": required_constructs,
                    "semantic_family": semantic_family,
                    "full_body_required": true,
                    "policy": "project_theseus_decoder_contract_v3_private_public_transfer"
                }
            }),
            task_id: format!("unit_{category}"),
            source_task_id: "unit".to_string(),
            card_id: "edge_contract_v3".to_string(),
            source_id: "unit".to_string(),
            split: "eval".to_string(),
            category: category.to_string(),
            prompt: format!("{prompt_signature} Solve {semantic_family} from the visible decoder contract."),
            entry_point: category.to_string(),
            solution_expr: String::new(),
            solution_body: body.to_string(),
            tags: vec![],
            benchmark_evidence_level: "private_generated_unit".to_string(),
        }
    }
}
