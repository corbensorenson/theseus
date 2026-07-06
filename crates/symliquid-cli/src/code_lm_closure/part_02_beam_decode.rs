use super::*;

pub(super) fn greedy_body(
    task: &CodeTask,
    readout: &LinearReadout,
    vocab: &Vocab,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Option<String> {
    let prompt = prompt_tokens_with_sts(task, sts_streams)
        .into_iter()
        .collect::<Vec<_>>();
    let feature_context =
        token_feature_static_context(&prompt, &task.category, readout.input_dim);
    let mut prev2 = "<BOS>".to_string();
    let mut prev1 = "<BOS>".to_string();
    let mut out = Vec::new();
    let max_steps = learned_token_max_steps(
        task,
        if low_latency_candidate_fanout_enabled() {
            40
        } else {
            96
        },
    );
    for position in 0..max_steps {
        let features = Tensor::from_row(featurize_with_static_context(
            &feature_context,
            &prev2,
            &prev1,
            position,
        ));
        let logits = readout.logits(&features).ok()?;
        let mut ranked = logits
            .row(0)
            .iter()
            .copied()
            .enumerate()
            .map(|(id, value)| {
                let token = &vocab.id_to_token[id];
                (
                    id,
                    value
                        + token_alignment_bonus(token, &prompt)
                        + body_position_bonus(token, position)
                        + category_body_token_bonus(task, token)
                        + category_position_token_bonus(task, token, position),
                )
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let token = ranked
            .into_iter()
            .map(|(id, _)| vocab.id_to_token[id].clone())
            .find(|token| token != "<UNK>" && task_body_token_allowed(task, &out, token))?;
        if token == "<EOS>" {
            break;
        }
        out.push(token.clone());
        prev2 = prev1;
        prev1 = token;
    }
    let body = join_body_tokens(&out);
    state_sequence_candidate_body_ok(task, &body).then_some(body)
}

#[derive(Debug, Clone)]
pub(super) struct BeamState {
    pub(super) tokens: Vec<String>,
    pub(super) prev2: String,
    pub(super) prev1: String,
    pub(super) score: f32,
    pub(super) finished: bool,
}

pub(super) fn beam_bodies(
    task: &CodeTask,
    readout: &LinearReadout,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<String> {
    let prompt = prompt_tokens_with_sts(task, sts_streams)
        .into_iter()
        .collect::<Vec<_>>();
    let feature_context =
        token_feature_static_context(&prompt, &task.category, readout.input_dim);
    let mut beams = vec![BeamState {
        tokens: Vec::new(),
        prev2: "<BOS>".to_string(),
        prev1: "<BOS>".to_string(),
        score: 0.0,
        finished: false,
    }];
    let seed_prefixes = if template_free_student_candidates_enabled() {
        contract_decode_seed_prefixes(task)
    } else {
        state_sequence_seed_prefixes(task)
    };
    for prefix in seed_prefixes {
        if !prefix
            .iter()
            .all(|token| vocab.token_to_id.contains_key(token))
        {
            continue;
        }
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
        beams.push(BeamState {
            tokens: prefix,
            prev2,
            prev1,
            score: if template_free_student_candidates_enabled() {
                1.35
            } else {
                2.0
            },
            finished: false,
        });
    }
    let beam_width = if execution_shaped_category(&task.category) {
        limit.clamp(2, 4)
    } else if task.split == "public_calibration" {
        limit.clamp(2, 5)
    } else {
        limit.clamp(2, 5)
    };
    let branch_width = if execution_shaped_category(&task.category) {
        8usize
    } else if task.split == "public_calibration" {
        12usize
    } else {
        10usize
    };
    let max_steps = learned_token_max_steps(
        task,
        if task.split == "public_calibration" {
            48
        } else {
            48
        },
    );
    for position in 0..max_steps {
        let mut next = Vec::new();
        let mut active_beams = Vec::new();
        let mut feature_rows = Vec::new();
        for beam in &beams {
            if beam.finished {
                next.push(beam.clone());
                continue;
            }
            feature_rows.extend(featurize_with_static_context(
                &feature_context,
                &beam.prev2,
                &beam.prev1,
                position,
            ));
            active_beams.push(beam);
        }
        if active_beams.is_empty() && next.is_empty() {
            break;
        }
        if !active_beams.is_empty() {
            let Ok(features) = Tensor::new(active_beams.len(), readout.input_dim, feature_rows)
            else {
                break;
            };
            let Ok(token_options) =
                top_token_log_probs_batch(&features, readout, vocab, branch_width)
            else {
                break;
            };
            for (row_idx, beam) in active_beams.into_iter().enumerate() {
                for (token, log_prob) in token_options
                    .get(row_idx)
                    .into_iter()
                    .flat_map(|row| row.iter().cloned())
                {
                    if token == "<UNK>" {
                        continue;
                    }
                    if !task_body_token_allowed(task, &beam.tokens, &token) {
                        continue;
                    }
                    if token == "<EOS>" && beam.tokens.is_empty() {
                        continue;
                    }
                    let mut candidate = beam.clone();
                    if token == "<EOS>" {
                        candidate.finished = true;
                        candidate.score += log_prob + length_bonus(candidate.tokens.len());
                    } else {
                        let repetition_penalty = if candidate
                            .tokens
                            .iter()
                            .rev()
                            .take(4)
                            .any(|item| item == &token)
                        {
                            -1.2
                        } else {
                            0.0
                        };
                        candidate.tokens.push(token.clone());
                        candidate.prev2 = candidate.prev1;
                        candidate.prev1 = token.clone();
                        candidate.score += log_prob
                            + token_alignment_bonus(&token, &prompt)
                            + body_position_bonus(&token, position)
                            + category_body_token_bonus(task, &token)
                            + category_position_token_bonus(task, &token, position)
                            + repetition_penalty;
                    }
                    next.push(candidate);
                }
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
                    stable_hash_u64(&format!("{}:{}:{:?}", seed, task.task_id, a.tokens)).cmp(
                        &stable_hash_u64(&format!("{}:{}:{:?}", seed, task.task_id, b.tokens)),
                    )
                })
        });
        beams = next.into_iter().take(beam_width).collect();
        if beams.iter().all(|beam| beam.finished) {
            break;
        }
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    beams.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for beam in beams {
        let body = join_body_tokens(&beam.tokens);
        if state_sequence_candidate_body_ok(task, &body) && seen.insert(body.clone()) {
            out.push(body);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

pub(super) fn batched_beam_bodies(
    tasks: &[CodeTask],
    readout: &LinearReadout,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: &StsStreamMap,
) -> HashMap<String, Vec<String>> {
    struct BatchedTask<'a> {
        task: &'a CodeTask,
        prompt: Vec<String>,
        feature_context: TokenFeatureStaticContext,
        beams: Vec<BeamState>,
        beam_width: usize,
        branch_width: usize,
        max_steps: usize,
    }

    let mut states = Vec::with_capacity(tasks.len());
    for task in tasks {
        let task_sts = sts_streams.get(&task.task_id);
        let prompt = prompt_tokens_with_sts(task, task_sts)
            .into_iter()
            .collect::<Vec<_>>();
        let feature_context =
            token_feature_static_context(&prompt, &task.category, readout.input_dim);
        let mut beams = vec![BeamState {
            tokens: Vec::new(),
            prev2: "<BOS>".to_string(),
            prev1: "<BOS>".to_string(),
            score: 0.0,
            finished: false,
        }];
        let seed_prefixes = if template_free_student_candidates_enabled() {
            contract_decode_seed_prefixes(task)
        } else {
            state_sequence_seed_prefixes(task)
        };
        for prefix in seed_prefixes {
            if !prefix
                .iter()
                .all(|token| vocab.token_to_id.contains_key(token))
                || !prefix_is_token_allowed(&prefix)
            {
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
            beams.push(BeamState {
                tokens: prefix,
                prev2,
                prev1,
                score: if template_free_student_candidates_enabled() {
                    1.35
                } else {
                    2.0
                },
                finished: false,
            });
        }
        let beam_width = if execution_shaped_category(&task.category) {
            limit.clamp(2, 4)
        } else if task.split == "public_calibration" {
            limit.clamp(2, 5)
        } else {
            limit.clamp(2, 5)
        };
        let branch_width = if execution_shaped_category(&task.category) {
            8usize
        } else if task.split == "public_calibration" {
            12usize
        } else {
            10usize
        };
        states.push(BatchedTask {
            task,
            prompt,
            feature_context,
            beams,
            beam_width,
            branch_width,
            max_steps: learned_token_max_steps(task, 48),
        });
    }

    let max_steps = states
        .iter()
        .map(|state| state.max_steps)
        .max()
        .unwrap_or(0);
    for position in 0..max_steps {
        let mut next_by_task: Vec<Vec<BeamState>> = vec![Vec::new(); states.len()];
        let mut active = Vec::new();
        let mut feature_rows = Vec::new();
        let mut branch_width = 0usize;
        for (task_idx, state) in states.iter().enumerate() {
            if position >= state.max_steps {
                next_by_task[task_idx].extend(state.beams.clone());
                continue;
            }
            branch_width = branch_width.max(state.branch_width);
            for beam in &state.beams {
                if beam.finished {
                    next_by_task[task_idx].push(beam.clone());
                    continue;
                }
                feature_rows.extend(featurize_with_static_context(
                    &state.feature_context,
                    &beam.prev2,
                    &beam.prev1,
                    position,
                ));
                active.push((task_idx, beam.clone()));
            }
        }
        if active.is_empty() {
            break;
        }
        let Ok(features) = Tensor::new(active.len(), readout.input_dim, feature_rows) else {
            break;
        };
        let Ok(token_options) = top_token_log_probs_batch(&features, readout, vocab, branch_width)
        else {
            break;
        };
        for (row_idx, (task_idx, beam)) in active.into_iter().enumerate() {
            let state = &states[task_idx];
            for (token, log_prob) in token_options
                .get(row_idx)
                .into_iter()
                .flat_map(|row| row.iter().take(state.branch_width).cloned())
            {
                if token == "<UNK>" {
                    continue;
                }
                if !task_body_token_allowed(state.task, &beam.tokens, &token) {
                    continue;
                }
                if token == "<EOS>" && beam.tokens.is_empty() {
                    continue;
                }
                let mut candidate = beam.clone();
                if token == "<EOS>" {
                    candidate.finished = true;
                    candidate.score += log_prob + length_bonus(candidate.tokens.len());
                } else {
                    let repetition_penalty = if candidate
                        .tokens
                        .iter()
                        .rev()
                        .take(4)
                        .any(|item| item == &token)
                    {
                        -1.2
                    } else {
                        0.0
                    };
                    candidate.tokens.push(token.clone());
                    candidate.prev2 = candidate.prev1;
                    candidate.prev1 = token.clone();
                    candidate.score += log_prob
                        + token_alignment_bonus(&token, &state.prompt)
                        + body_position_bonus(&token, position)
                        + category_body_token_bonus(state.task, &token)
                        + category_position_token_bonus(state.task, &token, position)
                        + repetition_penalty;
                }
                next_by_task[task_idx].push(candidate);
            }
        }
        let mut all_finished = true;
        for (task_idx, state) in states.iter_mut().enumerate() {
            let mut next = std::mem::take(&mut next_by_task[task_idx]);
            if next.is_empty() {
                next = state.beams.clone();
            }
            next.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        stable_hash_u64(&format!("{}:{}:{:?}", seed, state.task.task_id, a.tokens))
                            .cmp(&stable_hash_u64(&format!(
                                "{}:{}:{:?}",
                                seed, state.task.task_id, b.tokens
                            )))
                    })
            });
            state.beams = next.into_iter().take(state.beam_width).collect();
            if !state.beams.iter().all(|beam| beam.finished) {
                all_finished = false;
            }
        }
        if all_finished {
            break;
        }
    }

    let mut out = HashMap::new();
    for mut state in states {
        state.beams.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut bodies = Vec::new();
        let mut seen = HashSet::new();
        for beam in state.beams {
            let body = join_body_tokens(&beam.tokens);
            if state_sequence_candidate_body_ok(state.task, &body) && seen.insert(body.clone()) {
                bodies.push(body);
            }
            if bodies.len() >= limit {
                break;
            }
        }
        out.insert(state.task.task_id.clone(), bodies);
    }
    out
}

fn top_token_log_probs_batch(
    features: &Tensor,
    readout: &LinearReadout,
    vocab: &Vocab,
    limit: usize,
) -> Result<Vec<Vec<(String, f32)>>, Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    {
        if cuda_decode_logits_enabled() {
            if let Ok(rows) = symliquid_cuda::readout_cuda::linear_readout_topk_log_probs_cuda(
                features,
                readout,
                limit,
                cuda_decode_top_p(),
            ) {
                return Ok(rows
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .filter_map(|(idx, log_prob)| {
                                vocab.id_to_token.get(idx).map(|token| (token.clone(), log_prob))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect());
            }
        }
    }
    let logits = readout.logits(features)?;
    Ok((0..features.rows)
        .map(|row| top_token_log_probs(logits.row(row), vocab, limit))
        .collect())
}

#[cfg(feature = "cuda")]
fn cuda_decode_logits_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_CUDA_DECODE_LOGITS")
        .map(|value| value.trim() != "0")
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn cuda_decode_top_p() -> f32 {
    std::env::var("THESEUS_CODE_LM_CUDA_TOP_P")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(1.0)
}

fn top_token_log_probs(row: &[f32], vocab: &Vocab, limit: usize) -> Vec<(String, f32)> {
    let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom = row
        .iter()
        .map(|value| (*value - max_logit).exp())
        .sum::<f32>()
        .max(1e-8);
    let mut ranked = row
        .iter()
        .copied()
        .enumerate()
        .map(|(id, value)| {
            (
                vocab.id_to_token[id].clone(),
                value - max_logit - denom.ln(),
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.into_iter().take(limit).collect()
}

pub(super) fn contract_decode_seed_prefixes(task: &CodeTask) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    let hints = decoder_required_constructs(task);
    let shape = decoder_return_shape(task);
    let family = decoder_type_family(task);
    let text = task_contract_text(task);
    let primary = decoder_primary_arg(task);
    let secondary = decoder_secondary_arg(task).unwrap_or_else(|| "other".to_string());
    let mut add = |rows: &mut Vec<Vec<String>>, source: String| {
        let tokens = tokenize_body(&source);
        if tokens.is_empty() || tokens.iter().any(|token| token == "return") {
            return;
        }
        if !prefix_is_token_allowed(&tokens) {
            return;
        }
        let key = tokens.join(" ");
        if seen.insert(key) {
            rows.push(tokens);
        }
    };

    if shape == "list" && hints.contains("locals") && hints.contains("loop") {
        add(&mut rows, format!("out = []\nfor item in {primary}:\n    out.append"));
        if hints.contains("index_or_string_ops") || family == "string_indexing" {
            add(
                &mut rows,
                format!("out = []\nfor idx in range(len({primary})):\n    out.append"),
            );
        }
        if text.contains("prefix") {
            add(
                &mut rows,
                format!("out = []\nfor i in range(1, len({primary}) + 1):\n    out.append"),
            );
        }
    }
    if shape == "dict" && (hints.contains("frequency") || hints.contains("locals")) {
        add(
            &mut rows,
            format!("counts = {{}}\nfor item in {primary}:\n    counts[item] = counts.get"),
        );
    }
    if shape == "number" && hints.contains("loop") {
        add(
            &mut rows,
            format!("total = 0\nfor item in {primary}:\n    total +="),
        );
    }
    if shape == "bool" && hints.contains("two_arg_interface") {
        add(
            &mut rows,
            format!("for left, right in zip({primary}, {secondary}):\n    if"),
        );
    }
    if shape == "str" && hints.contains("loop") {
        add(&mut rows, format!("out = []\nfor ch in {primary}:\n    out.append"));
    }
    rows.into_iter().take(8).collect()
}

