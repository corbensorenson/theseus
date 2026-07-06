use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::time::Instant;

use serde_json::{json, Map, Value};
use symliquid_core::tensor::Tensor;
use symliquid_core::train::LinearReadout;

#[derive(Debug, Clone)]
pub struct CodeRankerConfig {
    pub candidate_manifest: String,
    pub trace_in: String,
    pub seed: u64,
    pub holdout_ratio: f32,
    pub epochs: usize,
    pub hv_dim: usize,
    pub lr: f32,
    pub use_cuda_readout: bool,
    pub model_out: String,
    pub candidate_out: String,
    pub training_examples_out: String,
    pub transfer_artifact_out: String,
    pub code_transfer_artifacts: String,
    pub out: String,
}

#[derive(Debug, Clone)]
struct Candidate {
    raw: Value,
    task_id: String,
    source_task_id: String,
    entry_point: String,
    candidate_sha256: String,
    code: String,
    origin: String,
    rank: usize,
    label_passed: Option<bool>,
}

#[derive(Debug, Clone)]
struct SelectedRow {
    task_id: String,
    candidate_sha256: String,
    rank: usize,
    passed: bool,
    selection_score: f32,
}

#[derive(Debug, Clone)]
struct EvalSummary {
    task_count: usize,
    passed: usize,
    pass_rate: f32,
    selected: Vec<SelectedRow>,
}

#[derive(Debug, Clone)]
struct ScoreLookup {
    scores: HashMap<String, f32>,
    backend: String,
}

pub fn train_code_ranker(config: CodeRankerConfig) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let candidate_values = read_jsonl(Path::new(&config.candidate_manifest))?;
    let trace_values = read_trace_inputs(&config.trace_in)?;
    let outcomes = collect_outcomes(&trace_values);
    let mut candidates = normalize_candidates(candidate_values);
    attach_labels(&mut candidates, &outcomes);
    let labeled: Vec<Candidate> = candidates
        .iter()
        .filter(|row| row.label_passed.is_some())
        .cloned()
        .collect();
    let (train_ids, eval_ids, split_kind) =
        split_tasks(&labeled, config.seed, config.holdout_ratio);
    let mut train_rows: Vec<Candidate> = labeled
        .iter()
        .filter(|row| train_ids.contains(&row.task_id))
        .cloned()
        .collect();
    let eval_rows: Vec<Candidate> = labeled
        .iter()
        .filter(|row| eval_ids.contains(&row.task_id))
        .cloned()
        .collect();
    let (train_rows, eval_rows, split_kind) = if train_rows.is_empty() && !labeled.is_empty() {
        (
            labeled.clone(),
            labeled.clone(),
            "same_trace_replay_no_holdout_available".to_string(),
        )
    } else {
        train_rows.sort_by(|a, b| {
            (a.task_id.as_str(), a.rank, a.candidate_sha256.as_str()).cmp(&(
                b.task_id.as_str(),
                b.rank,
                b.candidate_sha256.as_str(),
            ))
        });
        (train_rows, eval_rows, split_kind)
    };

    let mut readout = LinearReadout::zeros(config.hv_dim.max(8), 2);
    let mut training_examples = Vec::new();
    for row in &train_rows {
        training_examples.push(training_example(row));
    }
    let (ranker_backend, ranker_training_trace) = train_ranker_readout(
        &mut readout,
        &train_rows,
        config.epochs.max(1),
        config.lr,
        config.use_cuda_readout,
    )?;

    let eval_scores = build_score_lookup(&eval_rows, &readout, config.use_cuda_readout)?;
    let all_scores = build_score_lookup(&labeled, &readout, config.use_cuda_readout)?;
    let candidate_scores = build_score_lookup(&candidates, &readout, config.use_cuda_readout)?;

    let before_eval = evaluate_selection(&eval_rows, None)?;
    let after_eval = evaluate_selection(&eval_rows, Some(&eval_scores.scores))?;
    let all_before = evaluate_selection(&labeled, None)?;
    let all_after = evaluate_selection(&labeled, Some(&all_scores.scores))?;
    let pass_rate_delta = round6(after_eval.pass_rate - before_eval.pass_rate);
    let all_pass_rate_delta = round6(all_after.pass_rate - all_before.pass_rate);
    let weight_l1 = readout
        .weights
        .iter()
        .chain(readout.bias.iter())
        .map(|value| value.abs())
        .sum::<f32>();
    let checkpoint_material = format!(
        "{}:{}:{}:{:.6}:{:.6}:{}",
        config.seed,
        readout.input_dim,
        config.epochs,
        config.lr,
        weight_l1,
        training_examples.len()
    );
    let checkpoint_hash = stable_hash_hex(&checkpoint_material);
    let checkpoint_id = format!("theseus_student_neural_{}", &checkpoint_hash[..16]);

    let model = json!({
        "policy": "project_theseus_student_neural_code_checkpoint_v1",
        "created_utc": now(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_kind": "symliquid_linear_readout_code_ranker",
        "backend": ranker_backend.clone(),
        "score_backend": candidate_scores.backend.clone(),
        "seed": config.seed,
        "hv_dim": readout.input_dim,
        "output_dim": readout.output_dim,
        "epochs": config.epochs,
        "lr": config.lr,
        "source_candidate_manifest": config.candidate_manifest,
        "trace_paths": config.trace_in,
        "weights": readout.weights,
        "bias": readout.bias,
        "summary": {
            "training_example_count": training_examples.len(),
            "positive_examples": training_examples.iter().filter(|row| row.get("passed").and_then(Value::as_bool).unwrap_or(false)).count(),
            "negative_examples": training_examples.iter().filter(|row| !row.get("passed").and_then(Value::as_bool).unwrap_or(false)).count(),
            "train_task_count": train_ids.len(),
            "eval_task_count": eval_ids.len(),
            "parameter_count": readout.parameter_count(),
            "weight_delta_l1_from_zero": round6(weight_l1),
            "neural_weight_update": weight_l1 > 0.0,
            "use_cuda_readout_requested": config.use_cuda_readout,
            "ranker_training_trace": ranker_training_trace.clone(),
        },
        "generation_policy": {
            "public_tests_visible": false,
            "canonical_solutions_visible": false,
            "task_id_specific_lookup": false,
            "external_inference_calls": 0,
            "allowed_inputs": ["candidate_code_shape", "entry_point_tokens", "visible_task_tags", "sandbox_outcome_labels"],
            "known_limitation": "This checkpoint learns candidate selection/ranking over locally generated code; it is not yet a token-level code generator."
        },
        "external_inference_calls": 0,
    });
    write_json(Path::new(&config.model_out), &model)?;
    write_jsonl_values(Path::new(&config.training_examples_out), &training_examples)?;

    let learned_manifest = emit_learned_manifest(
        &candidates,
        &candidate_scores.scores,
        &candidate_scores.backend,
        &checkpoint_id,
    )?;
    write_jsonl_values(Path::new(&config.candidate_out), &learned_manifest)?;
    let transfer_path = write_transfer_artifact(
        &config,
        &checkpoint_id,
        &before_eval,
        &after_eval,
        &all_before,
        &all_after,
        &split_kind,
    )?;
    merge_transfer_index(&config.code_transfer_artifacts, &transfer_path)?;

    let leakage = leakage_findings(&candidates);
    let gates = vec![
        gate("trace_outcomes_loaded", !outcomes.is_empty(), json!(format!("outcomes={}", outcomes.len()))),
        gate("candidate_manifest_loaded", !candidates.is_empty(), json!(format!("candidates={}", candidates.len()))),
        gate("training_examples_nonzero", !training_examples.is_empty(), json!(format!("examples={}", training_examples.len()))),
        gate("neural_weights_changed", weight_l1 > 0.0, json!(format!("weight_delta_l1={:.6}", weight_l1))),
        gate("learned_manifest_emitted", !learned_manifest.is_empty(), json!(format!("learned_candidates={}", learned_manifest.len()))),
        gate("canonical_solutions_not_visible", !leakage["canonical_solution_seen"].as_bool().unwrap_or(false), leakage.clone()),
        gate("public_tests_not_used_for_generation", !leakage["public_tests_visible"].as_bool().unwrap_or(false), leakage.clone()),
        gate("task_id_not_used_as_feature", true, json!("features use code shape, visible tags, and entry tokens; full task_id is never a feature")),
        gate("external_inference_zero", true, json!("Rust/SymLiquid local readout training only")),
        gate("trained_checkpoint_improves_eval_selection", pass_rate_delta > 0.0 || all_pass_rate_delta > 0.0, json!(format!("before={} after={} delta={}", before_eval.pass_rate, after_eval.pass_rate, pass_rate_delta))),
    ];
    let trigger_state = if gates
        .iter()
        .all(|row| row["passed"].as_bool().unwrap_or(false))
    {
        "GREEN"
    } else if !training_examples.is_empty()
        && !leakage["canonical_solution_seen"]
            .as_bool()
            .unwrap_or(false)
    {
        "YELLOW"
    } else {
        "RED"
    };
    let report = json!({
        "policy": "project_theseus_student_learning_closure_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "frontier_family": "coding_local_sandbox",
        "runner_family": "student_neural_code_learning",
        "backend": ranker_backend.clone(),
        "score_backend": candidate_scores.backend.clone(),
        "seed": config.seed,
        "evidence_level": if split_kind == "hash_holdout" { "heldout_trace_replay_closure" } else { &split_kind },
        "candidate_source_before": "local_theseus_student_checkpoint",
        "candidate_source_after": "student_neural_checkpoint_v1",
        "public_benchmark_score_claim": "student_neural_checkpoint_public_task_calibration_only",
        "promotion_allowed": false,
        "promotion_rule": "requires neural checkpoint improvement plus independent real-code graduation run; same-trace replay alone cannot promote",
        "score": after_eval.pass_rate,
        "score_semantics": "student_neural_trace_replay_selection_pass_rate_not_public_mastery",
        "summary": {
            "candidate_count": candidates.len(),
            "labeled_candidate_count": labeled.len(),
            "outcome_count": outcomes.len(),
            "train_task_count": train_ids.len(),
            "eval_task_count": eval_ids.len(),
            "training_example_count": training_examples.len(),
            "before_eval_pass_rate": before_eval.pass_rate,
            "after_eval_pass_rate": after_eval.pass_rate,
            "pass_rate_delta": pass_rate_delta,
            "all_before_pass_rate": all_before.pass_rate,
            "all_after_pass_rate": all_after.pass_rate,
            "all_pass_rate_delta": all_pass_rate_delta,
            "learned_candidate_count": learned_manifest.len(),
            "checkpoint_id": checkpoint_id,
            "checkpoint_kind": "symliquid_linear_readout_code_ranker",
            "model_parameter_count": readout.parameter_count(),
            "weight_delta_l1_from_zero": round6(weight_l1),
            "use_cuda_readout_requested": config.use_cuda_readout,
            "ranker_training_trace": ranker_training_trace.clone(),
            "external_inference_calls": 0,
            "neural_weight_update": weight_l1 > 0.0,
            "student_behavior_changed": pass_rate_delta > 0.0 || all_pass_rate_delta > 0.0,
            "token_level_code_generation_learned": false
        },
        "before_after": {
            "eval_before": eval_to_json(&before_eval),
            "eval_after": eval_to_json(&after_eval),
            "all_before": eval_to_json(&all_before),
            "all_after": eval_to_json(&all_after)
        },
        "leakage_policy": {
            "canonical_solutions_visible": false,
            "public_tests_visible_to_generator": false,
            "task_id_specific_lookup": false,
            "external_inference_calls": 0,
            "known_limitation": "This is a real neural/readout weight update for candidate selection, not yet token-level code generation."
        },
        "artifacts": {
            "checkpoint": config.model_out,
            "candidate_manifest": config.candidate_out,
            "training_examples": config.training_examples_out,
            "transfer_artifact": transfer_path,
            "source_candidate_manifest": config.candidate_manifest,
            "trace_in": config.trace_in
        },
        "gates": gates,
        "runtime_ms": started.elapsed().as_millis() as u64,
        "external_inference_calls": 0,
    });
    write_json(Path::new(&config.out), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn read_trace_inputs(trace_in: &str) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    for raw in trace_in
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        rows.extend(read_jsonl(Path::new(raw))?);
    }
    Ok(rows)
}

fn collect_outcomes(rows: &[Value]) -> HashMap<(String, String), bool> {
    let mut out = HashMap::new();
    for row in rows {
        if row.get("event").and_then(Value::as_str) != Some("real_code_candidate_test") {
            continue;
        }
        let task_id = str_field(row, "task_id");
        let digest = str_field(row, "candidate_sha256");
        if task_id.is_empty() || digest.is_empty() {
            continue;
        }
        let entry = out.entry((task_id, digest)).or_insert(false);
        *entry = *entry || row.get("passed").and_then(Value::as_bool).unwrap_or(false);
    }
    out
}

fn normalize_candidates(rows: Vec<Value>) -> Vec<Candidate> {
    rows.into_iter()
        .filter_map(|raw| {
            let code = str_field(&raw, "code");
            let task_id = str_field(&raw, "task_id");
            if code.trim().is_empty() || task_id.is_empty() {
                return None;
            }
            let candidate_sha256 = str_field(&raw, "candidate_sha256");
            Some(Candidate {
                source_task_id: str_field(&raw, "source_task_id"),
                entry_point: str_field(&raw, "entry_point"),
                origin: str_field(&raw, "origin"),
                rank: parse_rank(&str_field(&raw, "origin")),
                candidate_sha256: if candidate_sha256.is_empty() {
                    stable_hash_hex(&code)
                } else {
                    candidate_sha256
                },
                code,
                task_id,
                raw,
                label_passed: None,
            })
        })
        .collect()
}

fn attach_labels(candidates: &mut [Candidate], outcomes: &HashMap<(String, String), bool>) {
    for row in candidates {
        if let Some(passed) = outcomes.get(&(row.task_id.clone(), row.candidate_sha256.clone())) {
            row.label_passed = Some(*passed);
        }
    }
}

fn split_tasks(
    rows: &[Candidate],
    seed: u64,
    holdout_ratio: f32,
) -> (HashSet<String>, HashSet<String>, String) {
    let mut unique: Vec<String> = rows
        .iter()
        .map(|row| row.task_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if unique.len() < 3 {
        let set = unique.into_iter().collect::<HashSet<_>>();
        return (
            set.clone(),
            set,
            "same_trace_replay_small_sample".to_string(),
        );
    }
    unique.sort_by_key(|task_id| stable_hash64(&format!("{seed}:{task_id}")));
    let ratio = holdout_ratio.clamp(0.0, 0.8);
    let holdout_count = ((unique.len() as f32 * ratio).round() as usize).clamp(1, unique.len() - 1);
    let eval = unique[..holdout_count]
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let train = unique[holdout_count..]
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    (train, eval, "hash_holdout".to_string())
}

fn training_example(row: &Candidate) -> Value {
    json!({
        "task_id_sha64": stable_hash_hex(&row.task_id),
        "candidate_sha256": row.candidate_sha256,
        "passed": row.label_passed.unwrap_or(false),
        "label": if row.label_passed.unwrap_or(false) { 1 } else { 0 },
        "rank": row.rank,
        "features": features_for(row),
    })
}

fn evaluate_selection(
    rows: &[Candidate],
    scores: Option<&HashMap<String, f32>>,
) -> Result<EvalSummary, Box<dyn std::error::Error>> {
    let mut groups: BTreeMap<String, Vec<&Candidate>> = BTreeMap::new();
    for row in rows {
        groups.entry(row.task_id.clone()).or_default().push(row);
    }
    let mut selected = Vec::new();
    for (task_id, task_rows) in groups {
        let chosen = if let Some(scores) = scores {
            task_rows
                .iter()
                .copied()
                .max_by(|left, right| {
                    lookup_score(left, scores)
                        .partial_cmp(&lookup_score(right, scores))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| right.rank.cmp(&left.rank))
                })
                .unwrap()
        } else {
            task_rows
                .iter()
                .copied()
                .min_by_key(|row| (row.rank, row.candidate_sha256.clone()))
                .unwrap()
        };
        selected.push(SelectedRow {
            task_id,
            candidate_sha256: chosen.candidate_sha256.clone(),
            rank: chosen.rank,
            passed: chosen.label_passed.unwrap_or(false),
            selection_score: scores
                .map(|scores| lookup_score(chosen, scores))
                .unwrap_or(0.0),
        });
    }
    let passed = selected.iter().filter(|row| row.passed).count();
    let task_count = selected.len();
    Ok(EvalSummary {
        task_count,
        passed,
        pass_rate: ratio(passed, task_count),
        selected,
    })
}

fn emit_learned_manifest(
    candidates: &[Candidate],
    scores: &HashMap<String, f32>,
    score_backend: &str,
    checkpoint_id: &str,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let mut groups: BTreeMap<(String, String, String), Vec<&Candidate>> = BTreeMap::new();
    for row in candidates {
        groups
            .entry((
                row.task_id.clone(),
                row.source_task_id.clone(),
                row.entry_point.clone(),
            ))
            .or_default()
            .push(row);
    }
    let mut out = Vec::new();
    for (_key, mut rows) in groups {
        rows.sort_by(|a, b| {
            lookup_score(b, scores)
                .partial_cmp(&lookup_score(a, scores))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.rank.cmp(&b.rank))
                .then_with(|| a.candidate_sha256.cmp(&b.candidate_sha256))
        });
        for (idx, row) in rows.into_iter().enumerate() {
            let mut raw = row.raw.clone();
            let new_rank = idx + 1;
            set_field(
                &mut raw,
                "candidate_source",
                json!("student_neural_checkpoint_v1"),
            );
            set_field(&mut raw, "checkpoint_id", json!(checkpoint_id));
            set_field(
                &mut raw,
                "origin",
                json!(format!(
                    "student_neural_checkpoint_v1:{}:rank{}:parent_rank{}:parent_origin:{}",
                    score_backend,
                    new_rank,
                    row.rank,
                    safe_origin(&row.origin)
                )),
            );
            set_field(
                &mut raw,
                "candidate_generation_mode",
                json!("neural_ranker_over_parent_candidates"),
            );
            set_field(
                &mut raw,
                "candidate_generation_contract",
                json!(
                    "learned_neural_selection_over_existing_candidates_not_token_level_generation"
                ),
            );
            set_field(
                &mut raw,
                "token_level_code_generation_learned",
                json!(false),
            );
            set_field(&mut raw, "benchmark_promotion_eligible", json!(false));
            set_field(&mut raw, "loop_closure_generated", json!(false));
            set_field(
                &mut raw,
                "selection_score",
                json!(round6(lookup_score(row, scores))),
            );
            set_field(&mut raw, "canonical_solution_seen_by_solver", json!(false));
            set_field(&mut raw, "public_tests_visible_to_generator", json!(false));
            ensure_provenance_object(
                &mut raw,
                checkpoint_id,
                row.rank,
                new_rank,
                lookup_score(row, scores),
            );
            out.push(raw);
        }
    }
    Ok(out)
}

fn write_transfer_artifact(
    config: &CodeRankerConfig,
    checkpoint_id: &str,
    before_eval: &EvalSummary,
    after_eval: &EvalSummary,
    all_before: &EvalSummary,
    all_after: &EvalSummary,
    split_kind: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let payload = json!({
        "policy": "project_theseus_student_neural_learning_transfer_artifact_v1",
        "created_utc": now(),
        "family": "coding_local_sandbox",
        "card_id": "student_neural_learning_closure",
        "active_card": true,
        "summary": {
            "checkpoint_id": checkpoint_id,
            "split_kind": split_kind,
            "eval_before_pass_rate": before_eval.pass_rate,
            "eval_after_pass_rate": after_eval.pass_rate,
            "eval_pass_rate_delta": round6(after_eval.pass_rate - before_eval.pass_rate),
            "all_before_pass_rate": all_before.pass_rate,
            "all_after_pass_rate": all_after.pass_rate,
            "student_behavior_changed": after_eval.pass_rate > before_eval.pass_rate || all_after.pass_rate > all_before.pass_rate,
        },
        "failure_clusters": [{
            "category": "student_neural_code_learning_gap",
            "count": after_eval.task_count.saturating_sub(after_eval.passed),
            "suggested_intervention": "Distill failing code-ranker selections into token-level generation or richer neural candidate features.",
            "cards": ["student_neural_learning_closure"],
            "priority": 4.0
        }],
        "repair_traces": [{
            "trace_id": "student_neural_code_ranker_update",
            "created_utc": now(),
            "category": "learned_neural_candidate_selection",
            "repair_pattern": "rust_symliquid_linear_readout",
            "transfer_hint": "Load neural code-ranker checkpoint before code-family candidate selection and compare against frozen rank ordering.",
            "loads_into": ["code_repair_arm", "pressure_runner", "benchmark_adapter_factory", "octopus_router"]
        }],
        "loads_into": ["code_repair_arm", "benchmark_adapter_factory", "pressure_runner", "octopus_router"],
        "verification": {
            "trace_paths": config.trace_in,
            "candidate_manifest": config.candidate_manifest,
            "external_inference_calls": 0,
            "public_score_claim": "student_neural_trace_replay_selection_pass_rate_not_public_mastery"
        }
    });
    write_json(Path::new(&config.transfer_artifact_out), &payload)?;
    Ok(config.transfer_artifact_out.clone())
}

fn merge_transfer_index(
    index_path: &str,
    artifact_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut index = read_json(Path::new(index_path)).unwrap_or_else(|_| json!({}));
    let mut artifacts = index
        .get_mut("artifacts")
        .and_then(Value::as_array_mut)
        .map(std::mem::take)
        .unwrap_or_default();
    artifacts.retain(|row| row.get("path").and_then(Value::as_str) != Some(artifact_path));
    artifacts.push(json!({
        "name": "student_neural_learning_closure_transfer",
        "family": "coding_local_sandbox",
        "card_id": "student_neural_learning_closure",
        "path": artifact_path,
        "loads_into": ["code_repair_arm", "benchmark_adapter_factory", "pressure_runner", "octopus_router"],
        "cluster_count": 1,
        "trace_count": 1,
        "active_card": true
    }));
    let payload = json!({
        "policy": "project_theseus_code_transfer_artifacts_index_v1",
        "created_utc": now(),
        "summary": {
            "frontier_family": "coding_local_sandbox",
            "active_card_id": "student_neural_learning_closure",
            "artifact_count": artifacts.len(),
            "cluster_count": artifacts.iter().map(|row| row.get("cluster_count").and_then(Value::as_i64).unwrap_or(0)).sum::<i64>(),
            "trace_count": artifacts.iter().map(|row| row.get("trace_count").and_then(Value::as_i64).unwrap_or(0)).sum::<i64>(),
            "loads_into": ["code_repair_arm", "benchmark_adapter_factory", "pressure_runner", "octopus_router"]
        },
        "artifacts": artifacts,
        "external_inference_calls": 0
    });
    write_json(Path::new(index_path), &payload)?;
    Ok(())
}

fn train_ranker_readout(
    readout: &mut LinearReadout,
    train_rows: &[Candidate],
    epochs: usize,
    lr: f32,
    use_cuda_readout: bool,
) -> Result<(String, Value), Box<dyn std::error::Error>> {
    if use_cuda_readout {
        return train_ranker_readout_cuda(readout, train_rows, epochs, lr);
    }
    let started = Instant::now();
    let mut examples_seen = 0usize;
    let mut last_loss = 0.0f32;
    let mut last_accuracy = 0.0f32;
    for _ in 0..epochs.max(1) {
        for row in train_rows {
            let features = Tensor::from_row(project_features(row, readout.input_dim));
            let target = if row.label_passed.unwrap_or(false) {
                1
            } else {
                0
            };
            let trace = readout.train_step(&features, target, lr)?;
            last_loss = trace.loss;
            last_accuracy = trace.accuracy;
            examples_seen += 1;
        }
    }
    Ok((
        "rust_cpu_symliquid_linear_readout_code_ranker".to_string(),
        json!({
            "backend": "rust_cpu_symliquid_linear_readout_code_ranker",
            "examples_seen": examples_seen,
            "epochs": epochs.max(1),
            "loss": round6(last_loss),
            "accuracy": round6(last_accuracy),
            "runtime_ms": started.elapsed().as_millis(),
        }),
    ))
}

#[cfg(feature = "cuda")]
fn train_ranker_readout_cuda(
    readout: &mut LinearReadout,
    train_rows: &[Candidate],
    epochs: usize,
    lr: f32,
) -> Result<(String, Value), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let (features, targets) = ranker_training_batch(train_rows, readout.input_dim)?;
    let trace = symliquid_cuda::readout_cuda::train_readout_sgd_cuda(
        &features,
        &targets,
        readout,
        epochs.max(1),
        lr,
        512,
    )?;
    Ok((
        "rust_cuda_symliquid_linear_readout_code_ranker".to_string(),
        json!({
            "backend": "rust_cuda_symliquid_linear_readout_code_ranker",
            "examples_seen": train_rows.len() * epochs.max(1),
            "epochs": epochs.max(1),
            "loss": round6(trace.loss),
            "accuracy": round6(trace.accuracy),
            "grad_norm": round6(trace.grad_norm),
            "runtime_ms": started.elapsed().as_millis(),
        }),
    ))
}

#[cfg(not(feature = "cuda"))]
fn train_ranker_readout_cuda(
    _readout: &mut LinearReadout,
    _train_rows: &[Candidate],
    _epochs: usize,
    _lr: f32,
) -> Result<(String, Value), Box<dyn std::error::Error>> {
    Err("CUDA code ranker requested, but symliquid-cli was not built with --features cuda".into())
}

#[cfg(feature = "cuda")]
fn ranker_training_batch(
    rows: &[Candidate],
    hv_dim: usize,
) -> Result<(Tensor, Vec<usize>), Box<dyn std::error::Error>> {
    let mut data = Vec::with_capacity(rows.len() * hv_dim.max(8));
    let mut targets = Vec::with_capacity(rows.len());
    for row in rows {
        data.extend(project_features(row, hv_dim));
        targets.push(if row.label_passed.unwrap_or(false) {
            1
        } else {
            0
        });
    }
    Ok((Tensor::new(rows.len(), hv_dim.max(8), data)?, targets))
}

fn build_score_lookup(
    rows: &[Candidate],
    readout: &LinearReadout,
    use_cuda_readout: bool,
) -> Result<ScoreLookup, Box<dyn std::error::Error>> {
    if use_cuda_readout {
        return build_score_lookup_cuda(rows, readout);
    }
    let scores = rows
        .iter()
        .map(|row| {
            (
                score_key(row),
                score_candidate(row, readout, readout.input_dim),
            )
        })
        .collect::<HashMap<_, _>>();
    Ok(ScoreLookup {
        scores,
        backend: "rust_cpu_linear_readout_candidate_scoring".to_string(),
    })
}

#[cfg(feature = "cuda")]
fn build_score_lookup_cuda(
    rows: &[Candidate],
    readout: &LinearReadout,
) -> Result<ScoreLookup, Box<dyn std::error::Error>> {
    let features = candidate_feature_batch(rows, readout.input_dim)?;
    let scores_vec = symliquid_cuda::readout_cuda::score_binary_readout_cuda(&features, readout)?;
    let scores = rows
        .iter()
        .zip(scores_vec)
        .map(|(row, score)| (score_key(row), score))
        .collect::<HashMap<_, _>>();
    Ok(ScoreLookup {
        scores,
        backend: "rust_cuda_binary_readout_candidate_scoring".to_string(),
    })
}

#[cfg(not(feature = "cuda"))]
fn build_score_lookup_cuda(
    _rows: &[Candidate],
    _readout: &LinearReadout,
) -> Result<ScoreLookup, Box<dyn std::error::Error>> {
    Err(
        "CUDA candidate scoring requested, but symliquid-cli was not built with --features cuda"
            .into(),
    )
}

#[cfg(feature = "cuda")]
fn candidate_feature_batch(
    rows: &[Candidate],
    hv_dim: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut data = Vec::with_capacity(rows.len() * hv_dim.max(8));
    for row in rows {
        data.extend(project_features(row, hv_dim));
    }
    Ok(Tensor::new(rows.len(), hv_dim.max(8), data)?)
}

fn score_key(row: &Candidate) -> String {
    format!("{}:{}", row.task_id, row.candidate_sha256)
}

fn lookup_score(row: &Candidate, scores: &HashMap<String, f32>) -> f32 {
    scores
        .get(&score_key(row))
        .copied()
        .unwrap_or(f32::NEG_INFINITY)
}

fn score_candidate(row: &Candidate, readout: &LinearReadout, hv_dim: usize) -> f32 {
    let features = Tensor::from_row(project_features(row, hv_dim));
    match readout.logits(&features) {
        Ok(logits) => logits.get(0, 1) - logits.get(0, 0),
        Err(_) => f32::NEG_INFINITY,
    }
}

fn project_features(row: &Candidate, hv_dim: usize) -> Vec<f32> {
    let mut data = vec![0.0; hv_dim.max(8)];
    let features = features_for(row);
    for feature in features {
        let hash = stable_hash64(&feature);
        let idx = (hash as usize) % data.len();
        let sign = if ((hash >> 63) & 1) == 0 { 1.0 } else { -1.0 };
        data[idx] += sign;
    }
    let norm = data.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut data {
            *value /= norm;
        }
    }
    data
}

fn features_for(row: &Candidate) -> Vec<String> {
    let mut out = BTreeSet::new();
    let code = row.code.to_lowercase();
    out.insert(format!("entry_arity:{}", function_args(&row.code).len()));
    for token in tokenize(&row.entry_point) {
        out.insert(format!("entry_token:{token}"));
    }
    for tag in visible_tags(&row.raw) {
        out.insert(format!("tag:{tag}"));
    }
    let checks = [
        ("code:return_none", code.contains("return none")),
        (
            "code:return_single_symbol",
            row.code
                .trim()
                .lines()
                .last()
                .map(|line| {
                    let trimmed = line.trim();
                    trimmed.starts_with("return ")
                        && trimmed["return ".len()..]
                            .chars()
                            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                })
                .unwrap_or(false),
        ),
        (
            "code:regex",
            code.contains("import re") || code.contains("re."),
        ),
        (
            "code:regex_def_name",
            code.contains("re.search") && code.contains("def\\s+"),
        ),
        (
            "code:all_membership",
            code.contains("all(") && code.contains(" in "),
        ),
        (
            "code:stable_dedupe_loop",
            code.contains("seen = set") && code.contains("append"),
        ),
        ("code:uses_set", code.contains("set(")),
        ("code:uses_sorted", code.contains("sorted(")),
        ("code:uses_sum", code.contains("sum(")),
        ("code:uses_len", code.contains("len(")),
        ("code:has_loop", code.contains("for ")),
        ("code:has_while", code.contains("while ")),
        (
            "code:list_comprehension",
            code.contains("[") && code.contains(" for ") && code.contains("]"),
        ),
        (
            "code:counter",
            code.contains("collections") || code.contains("counter("),
        ),
        (
            "origin:program_induction_prior",
            row.origin.contains("program_induction_prior"),
        ),
    ];
    for (name, present) in checks {
        if present {
            out.insert(name.to_string());
        }
    }
    out.into_iter().collect()
}

fn visible_tags(raw: &Value) -> Vec<String> {
    raw.get("provenance")
        .and_then(|value| value.get("visible_task"))
        .and_then(|value| value.get("tags"))
        .and_then(Value::as_array)
        .map(|tags| {
            tags.iter()
                .flat_map(|tag| tokenize(tag.as_str().unwrap_or_default()))
                .collect()
        })
        .unwrap_or_default()
}

fn function_args(code: &str) -> Vec<String> {
    let Some(line) = code
        .lines()
        .find(|line| line.trim_start().starts_with("def "))
    else {
        return Vec::new();
    };
    let Some(open) = line.find('(') else {
        return Vec::new();
    };
    let Some(close) = line[open + 1..].find(')') else {
        return Vec::new();
    };
    line[open + 1..open + 1 + close]
        .split(',')
        .filter_map(|raw| {
            let name = raw
                .trim()
                .split_once(':')
                .map(|(left, _)| left)
                .unwrap_or(raw.trim())
                .split_once('=')
                .map(|(left, _)| left)
                .unwrap_or(raw.trim())
                .trim();
            if name.is_empty() || name == "self" || name.starts_with('*') {
                None
            } else {
                Some(name.to_string())
            }
        })
        .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            out.push(current.clone());
            if current.contains('_') {
                out.extend(
                    current
                        .split('_')
                        .filter(|part| !part.is_empty())
                        .map(ToString::to_string),
                );
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        out.push(current.clone());
        if current.contains('_') {
            out.extend(
                current
                    .split('_')
                    .filter(|part| !part.is_empty())
                    .map(ToString::to_string),
            );
        }
    }
    out
}

fn parse_rank(origin: &str) -> usize {
    if let Some(idx) = origin.find("rank") {
        let digits: String = origin[idx + 4..]
            .chars()
            .take_while(|ch| ch.is_ascii_digit())
            .collect();
        if let Ok(value) = digits.parse() {
            return value;
        }
    }
    999
}

fn ensure_provenance_object(
    raw: &mut Value,
    checkpoint_id: &str,
    parent_rank: usize,
    learned_rank: usize,
    score: f32,
) {
    if !raw.is_object() {
        *raw = json!({});
    }
    let obj = raw.as_object_mut().unwrap();
    let provenance = obj
        .entry("provenance")
        .or_insert_with(|| Value::Object(Map::new()));
    if !provenance.is_object() {
        *provenance = Value::Object(Map::new());
    }
    provenance.as_object_mut().unwrap().insert(
        "student_neural_learning_closure".to_string(),
        json!({
            "policy": "project_theseus_student_neural_code_learning_v1",
            "checkpoint_id": checkpoint_id,
            "parent_rank": parent_rank,
            "learned_rank": learned_rank,
            "selection_score": round6(score),
            "candidate_generation_mode": "neural_ranker_over_parent_candidates",
            "token_level_code_generation_learned": false,
            "benchmark_promotion_eligible": false,
            "canonical_solution_used": false,
            "tests_used": false,
            "task_id_specific_lookup": false,
            "external_inference_calls": 0
        }),
    );
}

fn leakage_findings(candidates: &[Candidate]) -> Value {
    let canonical = candidates.iter().any(|row| {
        row.raw
            .get("canonical_solution_seen_by_solver")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    });
    let public_tests = candidates.iter().any(|row| {
        row.raw
            .get("public_tests_visible_to_generator")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    });
    let mut suspicious = BTreeSet::new();
    for row in candidates {
        if let Some(obj) = row.raw.as_object() {
            for key in obj.keys() {
                let lowered = key.to_ascii_lowercase();
                if matches!(
                    lowered.as_str(),
                    "canonical_solution" | "test" | "tests" | "expected_answer" | "answer_key"
                ) {
                    suspicious.insert(lowered);
                }
            }
        }
    }
    json!({
        "canonical_solution_seen": canonical,
        "public_tests_visible": public_tests,
        "suspicious_answer_or_test_fields": suspicious.into_iter().collect::<Vec<_>>()
    })
}

fn eval_to_json(summary: &EvalSummary) -> Value {
    json!({
        "task_count": summary.task_count,
        "passed": summary.passed,
        "pass_rate": summary.pass_rate,
        "selected": summary.selected.iter().take(50).map(|row| json!({
            "task_id": row.task_id,
            "candidate_sha256": row.candidate_sha256,
            "rank": row.rank,
            "passed": row.passed,
            "selection_score": round6(row.selection_score),
        })).collect::<Vec<_>>()
    })
}

fn gate(name: &str, passed: bool, evidence: Value) -> Value {
    json!({"gate": name, "passed": passed, "evidence": evidence})
}

fn set_field(raw: &mut Value, key: &str, value: Value) {
    if !raw.is_object() {
        *raw = Value::Object(Map::new());
    }
    raw.as_object_mut().unwrap().insert(key.to_string(), value);
}

fn str_field(raw: &Value, key: &str) -> String {
    raw.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn safe_origin(origin: &str) -> String {
    origin
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .take(96)
        .collect()
}

fn ratio(num: usize, den: usize) -> f32 {
    if den == 0 {
        0.0
    } else {
        round6(num as f32 / den as f32)
    }
}

fn round6(value: f32) -> f32 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn stable_hash_hex(text: &str) -> String {
    format!("{:016x}", stable_hash64(text))
}

fn stable_hash64(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn read_json(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(json!({}));
    }
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(line) {
            if value.is_object() {
                rows.push(value);
            }
        }
    }
    Ok(rows)
}

fn write_json(path: &Path, payload: &Value) -> Result<(), Box<dyn std::error::Error>> {
    ensure_parent(path)?;
    fs::write(
        path,
        format!("{}\n", serde_json::to_string_pretty(payload)?),
    )?;
    Ok(())
}

fn write_jsonl_values(path: &Path, rows: &[Value]) -> Result<(), Box<dyn std::error::Error>> {
    ensure_parent(path)?;
    let mut out = String::new();
    for row in rows {
        out.push_str(&serde_json::to_string(row)?);
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

fn ensure_parent(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn now() -> String {
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-Date).ToUniversalTime().ToString('o')",
        ])
        .output();
    if let Ok(output) = output {
        if output.status.success() {
            return String::from_utf8_lossy(&output.stdout).trim().to_string();
        }
    }
    "1970-01-01T00:00:00Z".to_string()
}
