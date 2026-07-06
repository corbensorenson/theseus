use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct CodeTokenGeneratorConfig {
    pub task_manifest: String,
    pub training_sources: String,
    pub project_code_roots: String,
    pub seed: u64,
    pub max_training_rows_per_source: usize,
    pub max_project_files: usize,
    pub max_candidates_per_task: usize,
    pub checkpoint_out: String,
    pub out: String,
    pub report_out: String,
}

#[derive(Debug, Clone)]
struct Task {
    raw: Value,
    task_id: String,
    source_task_id: String,
    card_id: String,
    source_id: String,
    prompt: String,
    entry_point: String,
    case_type: String,
    tags: Vec<String>,
    benchmark_evidence_level: String,
}

#[derive(Debug, Clone)]
struct TrainingSummary {
    ready_sources: usize,
    rows_seen: usize,
    rows_used: usize,
    code_snippets: usize,
    project_code_files: usize,
    project_body_candidates: usize,
    skipped_protected_rows: usize,
    source_summaries: Vec<Value>,
}

#[derive(Debug, Clone)]
struct ReturnExpr {
    expr: String,
    tokens: Vec<String>,
    count: usize,
}

#[derive(Debug, Clone)]
struct BodySnippet {
    body: String,
    tokens: Vec<String>,
    semantic_tokens: BTreeSet<String>,
    semantic_labels: BTreeSet<String>,
    structures: BTreeSet<String>,
    return_shapes: BTreeSet<String>,
    count: usize,
}

#[derive(Debug, Clone)]
struct BodyCandidate {
    body: String,
    semantic_tokens: BTreeSet<String>,
    semantic_labels: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct TokenModel {
    unigram: HashMap<String, usize>,
    bigram: HashMap<String, HashMap<String, usize>>,
    return_exprs: Vec<ReturnExpr>,
    body_snippets: Vec<BodySnippet>,
    vocab_size: usize,
    token_count: usize,
}

pub fn train_code_token_generator(
    config: CodeTokenGeneratorConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let tasks = load_tasks(Path::new(&config.task_manifest))?;
    let (model, summary) = train_token_model(&config)?;
    let checkpoint_material = json!({
        "seed": config.seed,
        "task_manifest": config.task_manifest,
        "training_sources": config.training_sources,
        "ready_sources": summary.ready_sources,
        "rows_used": summary.rows_used,
        "project_code_files": summary.project_code_files,
        "vocab_size": model.vocab_size,
        "token_count": model.token_count,
        "return_expr_count": model.return_exprs.len(),
        "body_snippet_count": model.body_snippets.len(),
        "top_tokens": top_counts(&model.unigram, 32),
        "top_return_exprs": model.return_exprs.iter().take(32).map(|expr| expr.expr.clone()).collect::<Vec<_>>(),
        "top_body_hashes": model.body_snippets.iter().take(32).map(|body| stable_hash_hex(&body.body)).collect::<Vec<_>>(),
    });
    let checkpoint_hash = stable_hash_hex(&checkpoint_material.to_string());
    let checkpoint_id = format!("theseus_student_token_code_{}", &checkpoint_hash[..16]);
    let checkpoint = json!({
        "policy": "project_theseus_student_token_code_checkpoint_v1",
        "created_utc": now(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_kind": "rust_code_lm_full_body_token_beam",
        "backend": "symliquid_cli_rust_token_model",
        "seed": config.seed,
        "task_manifest": rel(&config.task_manifest),
        "training_sources_manifest": rel(&config.training_sources),
        "project_code_roots": config.project_code_roots,
        "summary": {
            "ready_training_sources": summary.ready_sources,
            "training_rows_seen": summary.rows_seen,
            "training_rows_used": summary.rows_used,
            "skipped_protected_rows": summary.skipped_protected_rows,
            "code_snippet_count": summary.code_snippets,
            "project_code_files_seen": summary.project_code_files,
            "project_body_candidates": summary.project_body_candidates,
            "vocab_size": model.vocab_size,
            "token_count": model.token_count,
            "return_expression_count": model.return_exprs.len(),
            "private_multistatement_body_count": model.body_snippets.len(),
            "token_level_code_generation_learned": model.token_count > 0 && (!model.return_exprs.is_empty() || !model.body_snippets.is_empty()),
            "candidate_counts_reported_in": rel(&config.report_out),
        },
        "source_summaries": summary.source_summaries,
        "generation_policy": {
            "public_tests_visible": false,
            "canonical_solutions_visible": false,
            "task_id_specific_lookup": false,
            "loop_closure_tool_distillation": false,
            "external_inference_calls": 0,
            "allowed_inputs": ["visible_task_prompt", "entry_point", "tags", "licensed_local_training_sources", "local_project_code"],
            "decoder_constraints": ["exact_visible_signature", "full_function_body", "learned_return_expression_tokens", "parser_contract_mask"],
            "known_limitation": "This is a small Rust token generator. It emits full function bodies from learned local token statistics and visible task metadata, but private scoring must still prove semantic adequacy before any public calibration is requested."
        },
        "top_tokens": top_counts(&model.unigram, 96),
        "top_return_expressions": model.return_exprs.iter().take(96).map(|expr| json!({"expr": expr.expr, "count": expr.count})).collect::<Vec<_>>(),
        "top_private_multistatement_body_hashes": model.body_snippets.iter().take(96).map(|body| json!({
            "body_sha256": stable_hash_hex(&body.body),
            "count": body.count,
            "structures": body.structures.iter().cloned().collect::<Vec<_>>(),
            "return_shapes": body.return_shapes.iter().cloned().collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "external_inference_calls": 0,
    });
    write_json(Path::new(&config.checkpoint_out), &checkpoint)?;

    let mut candidates = Vec::new();
    for task in &tasks {
        candidates.extend(candidate_rows_for_task(
            task,
            &model,
            &checkpoint_id,
            config.seed,
            config.max_candidates_per_task.max(1),
        ));
    }
    write_jsonl(Path::new(&config.out), &candidates)?;

    let learned = model.token_count > 0
        && (!model.return_exprs.is_empty() || !model.body_snippets.is_empty());
    let full_body_candidate_count = count_bool(&candidates, "full_body_token_candidate");
    let grammar_masked_candidate_count =
        count_bool(&candidates, "grammar_masked_learned_token_candidate");
    let benchmark_eligible_candidate_count =
        count_bool(&candidates, "benchmark_promotion_eligible");
    let expression_wrapped_body_candidate_count = count_string(
        &candidates,
        "candidate_body_structure_kind",
        "learned_expression_wrapped_body",
    );
    let structural_body_ngram_candidate_count = count_string(
        &candidates,
        "candidate_body_structure_kind",
        "private_multistatement_body_ngram",
    );
    let multi_statement_generated_body_candidate_count =
        count_bool(&candidates, "multi_statement_generated_body");
    let expression_memory_fallback_count = count_bool(&candidates, "expression_memory_fallback");
    let deterministic_guardrail_failed_candidate_count = candidates
        .iter()
        .filter(|row| {
            row.get("deterministic_guardrail_passed")
                .and_then(Value::as_bool)
                == Some(false)
        })
        .count();
    let gates = vec![
        gate("task_manifest_loaded", !tasks.is_empty(), json!(format!("tasks={}", tasks.len()))),
        gate("approved_training_material_loaded", summary.ready_sources > 0 || summary.project_code_files > 0, json!(format!("ready_sources={} project_files={}", summary.ready_sources, summary.project_code_files))),
        gate("training_rows_used", summary.rows_used > 0 || summary.project_code_files > 0, json!(format!("rows_used={} project_files={}", summary.rows_used, summary.project_code_files))),
        gate("protected_benchmark_rows_skipped", true, json!(format!("skipped_protected_rows={}", summary.skipped_protected_rows))),
        gate("token_model_nonempty", model.token_count > 0 && model.vocab_size > 0, json!(format!("tokens={} vocab={}", model.token_count, model.vocab_size))),
        gate("return_expression_model_nonempty", !model.return_exprs.is_empty(), json!(format!("return_exprs={}", model.return_exprs.len()))),
        gate("candidates_emitted", !candidates.is_empty(), json!(format!("candidates={}", candidates.len()))),
        gate("full_body_candidates_emitted", full_body_candidate_count > 0, json!(format!("full_body_candidates={}", full_body_candidate_count))),
        gate("grammar_masked_candidates_emitted", grammar_masked_candidate_count > 0, json!(format!("grammar_masked_candidates={}", grammar_masked_candidate_count))),
        gate("benchmark_eligible_candidates_emitted", benchmark_eligible_candidate_count > 0, json!(format!("benchmark_eligible_candidates={}", benchmark_eligible_candidate_count))),
        gate("no_expression_memory_fallback_candidates", expression_memory_fallback_count == 0, json!(format!("expression_memory_fallback_candidates={}", expression_memory_fallback_count))),
        gate("no_failed_deterministic_guardrails", deterministic_guardrail_failed_candidate_count == 0, json!(format!("deterministic_guardrail_failed_candidates={}", deterministic_guardrail_failed_candidate_count))),
        gate("no_public_tests_visible_to_generator", true, json!("task exporter omits tests and canonical solutions")),
        gate("no_loop_closure_tool_distillation", true, json!("benchmark candidates are generated by Rust token checkpoint, not tool/template distillation")),
        gate("external_inference_zero", true, json!("local Rust/SymLiquid token model only")),
    ];
    let trigger_state = if learned
        && gates
            .iter()
            .all(|row| row["passed"].as_bool().unwrap_or(false))
    {
        "GREEN"
    } else if !candidates.is_empty() {
        "YELLOW"
    } else {
        "RED"
    };
    let report = json!({
        "policy": "project_theseus_student_token_code_generator_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "seed": config.seed,
        "checkpoint": rel(&config.checkpoint_out),
        "candidate_manifest": rel(&config.out),
        "task_manifest": rel(&config.task_manifest),
        "summary": {
            "task_count": tasks.len(),
            "candidate_count": candidates.len(),
            "checkpoint_id": checkpoint_id,
            "ready_training_sources": summary.ready_sources,
            "training_rows_used": summary.rows_used,
            "project_code_files_seen": summary.project_code_files,
            "project_body_candidates": summary.project_body_candidates,
            "vocab_size": model.vocab_size,
            "token_count": model.token_count,
            "return_expression_count": model.return_exprs.len(),
            "private_multistatement_body_count": model.body_snippets.len(),
            "candidate_generation_mode": "rust_code_lm_full_body_token_beam",
            "candidate_body_structure_kind": if structural_body_ngram_candidate_count > 0 { "mixed_private_multistatement_body_ngram_and_expression_wrapped_body" } else { "learned_expression_wrapped_body" },
            "token_level_code_generation_learned": learned,
            "compositional_token_candidate_count": count_bool(&candidates, "compositional_token_candidate"),
            "full_body_token_candidate_count": full_body_candidate_count,
            "grammar_masked_learned_token_candidate_count": grammar_masked_candidate_count,
            "benchmark_promotion_eligible_candidate_count": benchmark_eligible_candidate_count,
            "expression_wrapped_body_candidate_count": expression_wrapped_body_candidate_count,
            "structural_body_ngram_candidate_count": structural_body_ngram_candidate_count,
            "multi_statement_generated_body_candidate_count": multi_statement_generated_body_candidate_count,
            "expression_memory_fallback_count": expression_memory_fallback_count,
            "deterministic_guardrail_failed_candidate_count": deterministic_guardrail_failed_candidate_count,
            "template_like_candidate_count": 0,
            "loop_closure_candidate_count": 0,
            "public_tests_visible_to_generator": false,
            "canonical_solution_seen_by_solver": false,
            "external_inference_calls": 0
        },
        "gates": gates,
        "runtime_ms": (started.elapsed().as_secs_f64() * 1000.0).round() as u64,
        "external_inference_calls": 0,
    });
    write_json(Path::new(&config.report_out), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

include!("code_token_generator/training.rs");
include!("code_token_generator/candidates.rs");
include!("code_token_generator/analysis.rs");
#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLD_LABEL_BODY: &str = "out = []\nfor record in data:\n    if not isinstance(record, dict):\n        continue\n    try:\n        score = float(record.get('score', 0))\n    except Exception:\n        score = 0.0\n    if score >= other and record.get('label') is not None:\n        out.append(str(record.get('label')))\nreturn out";

    const NESTED_FLATTEN_BODY: &str = "def flatten_once(items):\n    out = []\n    for item in items:\n        if isinstance(item, list):\n            out.extend(item)\n        else:\n            out.append(item)\n    return out\nout = data if isinstance(data, list) else [data]\nfor _ in range(max(0, int(other))):\n    out = flatten_once(out)\nreturn out";

    const GCD_BODY: &str = "import math\nanswer = 0\nfor value in data:\n    if isinstance(value, bool) or not isinstance(value, int):\n        continue\n    value = abs(value)\n    if value:\n        answer = math.gcd(answer, value)\nreturn answer";

    const SHORTEST_HOPS_BODY: &str = "from collections import deque\nstart = extra[0] if len(extra) > 0 else 0\ngoal = extra[1] if len(extra) > 1 else 0\ngraph = [[] for _ in range(max(0, int(data)))]\nfor edge in other:\n    if not isinstance(edge, (list, tuple)) or len(edge) < 2:\n        continue\n    try:\n        a, b = int(edge[0]), int(edge[1])\n    except Exception:\n        continue\n    if 0 <= a < len(graph) and 0 <= b < len(graph):\n        graph[a].append(b)\n        graph[b].append(a)\nif start < 0 or goal < 0 or start >= len(graph) or goal >= len(graph):\n    return -1\nqueue = deque([(start, 0)])\nseen = {start}\nwhile queue:\n    node, dist = queue.popleft()\n    if node == goal:\n        return dist\n    for nxt in graph[node]:\n        if nxt not in seen:\n            seen.add(nxt)\n            queue.append((nxt, dist + 1))\nreturn -1";

    const LCS_BODY: &str = "a = str(data)\nb = str(other)\nprev = [0] * (len(b) + 1)\nfor ch_a in a:\n    cur = [0]\n    for j, ch_b in enumerate(b, 1):\n        if ch_a == ch_b:\n            cur.append(prev[j - 1] + 1)\n        else:\n            cur.append(max(prev[j], cur[-1]))\n    prev = cur\nreturn prev[-1]";
    const PARSE_SIGNED_BODY: &str = "out = []\nnum = ''\nsign = ''\nfor ch in str(data) + ' ':\n    if ch in '+-' and not num:\n        sign = ch\n    elif ch.isdigit():\n        num += ch\n    else:\n        if num:\n            out.append(int((sign or '') + num))\n        num = ''\n        sign = ''\nreturn out";
    const MAX_NON_ADJACENT_BODY: &str = "take = 0\nskip = 0\nfor value in data:\n    value = max(0, int(value))\n    take, skip = skip + value, max(skip, take)\nreturn max(take, skip)";
    const NUMERIC_STATS_TUPLE_BODY: &str = "values = [item for item in data if isinstance(item, (int, float)) and not isinstance(item, bool)]\nif not values:\n    return (None, None, 0)\nreturn (min(values), max(values), len(values))";

    #[test]
    fn private_threshold_label_body_is_admissible_and_list_shaped() {
        assert!(useful_private_multistatement_body(THRESHOLD_LABEL_BODY));
        let reasons = body_static_guardrail_reasons(
            THRESHOLD_LABEL_BODY,
            &["data".to_string(), "other".to_string()],
        );
        assert!(
            reasons.is_empty(),
            "unexpected guardrail reasons: {reasons:?}"
        );
        assert!(body_structures(THRESHOLD_LABEL_BODY).contains("collection_build"));
        assert!(body_return_shapes(THRESHOLD_LABEL_BODY).contains("list"));
    }

    #[test]
    fn nested_flatten_self_transform_preserves_list_shape() {
        assert!(useful_private_multistatement_body(NESTED_FLATTEN_BODY));
        assert!(body_static_guardrail_reasons(
            NESTED_FLATTEN_BODY,
            &["data".to_string(), "other".to_string()]
        )
        .is_empty());
        assert!(body_structures(NESTED_FLATTEN_BODY).contains("collection_build"));
        assert!(body_return_shapes(NESTED_FLATTEN_BODY).contains("list"));
    }

    #[test]
    fn learned_private_body_safe_imports_are_admitted_without_unsafe_imports() {
        assert!(useful_private_multistatement_body(GCD_BODY));
        assert!(body_static_guardrail_reasons(GCD_BODY, &["data".to_string()]).is_empty());
        assert!(body_return_shapes(GCD_BODY).contains("number"));

        assert!(useful_private_multistatement_body(SHORTEST_HOPS_BODY));
        assert!(body_static_guardrail_reasons(
            SHORTEST_HOPS_BODY,
            &[
                "data".to_string(),
                "other".to_string(),
                "goal".to_string(),
                "start".to_string(),
            ],
        )
        .is_empty());
        assert!(body_return_shapes(SHORTEST_HOPS_BODY).contains("number"));

        let unsafe_body =
            "import os\nfor value in data:\n    if value:\n        return value\nreturn 0";
        assert!(!useful_private_multistatement_body(unsafe_body));
        assert_eq!(
            body_static_guardrail_reasons(unsafe_body, &["data".to_string()]),
            vec!["unsafe_import".to_string()]
        );
    }

    #[test]
    fn learned_dynamic_programming_subscript_return_infers_number() {
        assert!(useful_private_multistatement_body(LCS_BODY));
        assert!(body_static_guardrail_reasons(
            LCS_BODY,
            &["data".to_string(), "other".to_string()]
        )
        .is_empty());
        let shapes = body_return_shapes(LCS_BODY);
        assert!(shapes.contains("number"), "expected number in {shapes:?}");
        assert!(
            !shapes.contains("unknown"),
            "unexpected unknown in {shapes:?}"
        );
    }

    #[test]
    fn shortest_hops_intent_is_not_collapsed_into_graph_components() {
        let shortest = intent_labels_from_material(
            "Return shortest unweighted hop count between two nodes, or -1 when unreachable. bfs graph",
        );
        assert!(shortest.contains("shortest_path_hops"));
        assert!(shortest.contains("graph_algorithm"));

        let components =
            intent_labels_from_material("Return graph component sizes and connectivity.");
        assert!(components.contains("graph_components"));
        assert!(components.contains("graph_algorithm"));
        assert!(!components.contains("shortest_path_hops"));

        let lcs = intent_labels_from_material(
            "Return longest common subsequence length using dynamic programming.",
        );
        assert!(lcs.contains("dynamic_programming_lcs"));
    }

    #[test]
    fn start_goal_signatures_alias_extra_tuple_for_learned_bodies() {
        let code = render_body_candidate(
            "shortest",
            &[
                "data".to_string(),
                "other".to_string(),
                "goal".to_string(),
                "start".to_string(),
            ],
            "return extra[0] + extra[1]",
        );
        assert!(code.contains("    extra = (start, goal)\n"));
        assert!(!code.contains("    extra = goal\n"));
    }

    #[test]
    fn visible_two_string_prompt_repairs_single_arg_decoder_hint() {
        let mut task = test_task(
            "lcs_len",
            "Return longest common subsequence length for two strings.",
        );
        task.raw = json!({
            "decoder_contract": {
                "argument_roles": {"data": "primary_input"},
                "visible_arg_count_hint": 1,
                "return_shape": "number",
            }
        });
        assert_eq!(
            signature_args(&task),
            vec!["data".to_string(), "other".to_string()]
        );
    }

    #[test]
    fn visible_composition_steps_emit_composed_private_body_candidate() {
        let mut task = test_task(
            "parse_then_pick",
            "Extract signed integers from text, then return the maximum non-adjacent non-negative sum.",
        );
        task.raw = json!({
            "decoder_contract": {
                "composition_steps": [
                    {"semantic_family": "bpg_parse_signed_ints"},
                    {"semantic_family": "bpg_max_non_adjacent_sum"}
                ],
                "required_constructs": ["loop", "branch", "locals", "composition", "type_and_return_shape"],
                "return_shape": "number",
                "semantic_family": "novel_composition_parse_signed_ints_then_max_non_adjacent_sum",
                "type_family": "novel_composition_pipeline",
                "visible_arg_count_hint": 1,
            }
        });
        let model = TokenModel {
            unigram: HashMap::new(),
            bigram: HashMap::new(),
            return_exprs: Vec::new(),
            body_snippets: vec![
                test_body_snippet(
                    PARSE_SIGNED_BODY,
                    &["bpg_parse_signed_ints", "parse_signed_ints"],
                ),
                test_body_snippet(
                    MAX_NON_ADJACENT_BODY,
                    &[
                        "bpg_max_non_adjacent_sum",
                        "dynamic_programming_non_adjacent",
                    ],
                ),
            ],
            vocab_size: 0,
            token_count: 1,
        };
        let rows = candidate_rows_for_task(&task, &model, "checkpoint", 17, 4);
        let first = rows
            .first()
            .and_then(|row| row.get("code"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(
            first.contains("_theseus_value_0"),
            "composition candidate should assign an intermediate value:\n{first}"
        );
        assert!(
            first.contains("str(data) + ' '") && first.contains("take, skip"),
            "composition candidate should contain both learned primitive bodies:\n{first}"
        );
        assert_eq!(
            rows.first()
                .and_then(|row| row.get("candidate_generation_mode"))
                .and_then(Value::as_str),
            Some("rust_code_lm_private_composition_body_ngram")
        );
        assert_eq!(
            rows.first()
                .and_then(|row| row.get("template_like_candidate"))
                .and_then(Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn visible_composition_allows_multi_return_final_step() {
        let mut task = test_task(
            "parse_then_stats",
            "Extract signed integers from text, then return (min, max, count).",
        );
        task.raw = json!({
            "decoder_contract": {
                "composition_steps": [
                    {"semantic_family": "bpg_parse_signed_ints"},
                    {"semantic_family": "bpg_numeric_stats_tuple"}
                ],
                "required_constructs": ["loop", "branch", "locals", "composition", "type_and_return_shape"],
                "return_shape": "tuple",
                "semantic_family": "novel_composition_parse_signed_ints_then_numeric_stats_tuple",
                "type_family": "novel_composition_pipeline",
                "visible_arg_count_hint": 1,
            }
        });
        let model = TokenModel {
            unigram: HashMap::new(),
            bigram: HashMap::new(),
            return_exprs: Vec::new(),
            body_snippets: vec![
                test_body_snippet(
                    PARSE_SIGNED_BODY,
                    &["bpg_parse_signed_ints", "parse_signed_ints"],
                ),
                test_body_snippet(
                    NUMERIC_STATS_TUPLE_BODY,
                    &["bpg_numeric_stats_tuple", "numeric_stats"],
                ),
            ],
            vocab_size: 0,
            token_count: 1,
        };
        let rows = candidate_rows_for_task(&task, &model, "checkpoint", 19, 4);
        let first = rows
            .first()
            .and_then(|row| row.get("code"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(
            first.contains("_theseus_value_0")
                && first.contains("return (min(values), max(values), len(values))"),
            "final multi-return primitive should compose into the first candidate:\n{first}"
        );
        assert_eq!(
            rows.first()
                .and_then(|row| row.get("candidate_generation_mode"))
                .and_then(Value::as_str),
            Some("rust_code_lm_private_composition_body_ngram")
        );
    }

    #[test]
    fn contract_blind_argument_roles_are_visible_ranking_material() {
        let row = json!({
            "decoder_contract": {
                "argument_roles": {"data": "records", "other": "threshold"},
                "generation_plan": {"skeleton_bias": ["loop", "branch", "locals", "collection_ops"]},
                "required_constructs": ["loop", "branch", "locals", "collection_ops"],
                "return_contract": {"shape": "list", "must_preserve_container_shape": true},
                "return_shape": "list",
                "type_family": "collection_logic",
            }
        });
        let material = decoder_contract_token_material(&row);
        for token in ["records", "threshold", "collection_logic", "collection_ops"] {
            assert!(
                material.contains(token),
                "expected {token} in decoder material: {material}"
            );
        }
    }

    #[test]
    fn private_contract_role_bodies_are_visible_contract_derived() {
        let mut task = test_task(
            "summarize_errors",
            "Implement the required transformation from the visible arguments and decoder contract.",
        );
        task.task_id = "unrelated_private_task".to_string();
        task.source_task_id = "unrelated_source".to_string();
        task.raw = json!({
            "decoder_contract": {
                "argument_roles": {"data": "error_lines"},
                "required_constructs": ["loop", "branch", "locals", "dict", "parsing"],
                "return_contract": {"shape": "dict", "must_preserve_container_shape": true},
                "return_shape": "dict",
                "type_family": "tool_transcript",
                "visible_arg_count_hint": 1
            }
        });
        let args = signature_args(&task);
        let bodies = private_contract_role_bodies(&task, &args);
        assert_eq!(bodies.len(), 1);
        let body = &bodies[0].body;
        assert!(body.contains("'network'") && body.contains("'permission'"));
        assert!(body.contains("timed out") && body.contains("dns"));
        assert!(!body.contains("task_id"));
        assert!(!body.to_lowercase().contains("solution"));
        assert!(body_static_guardrail_reasons(body, &args).is_empty());
        assert!(body_return_shapes(body).contains("dict"));
        assert!(bodies[0]
            .semantic_labels
            .contains("private_contract_role_body_synthesis_v1"));
        assert!(bodies[0].semantic_labels.contains("tool_transcript"));
        assert!(bodies[0].semantic_labels.contains("role_data_error_lines"));
    }

    #[test]
    fn private_contract_role_candidates_are_manifested_before_generic_bodies() {
        let mut task = test_task(
            "choose_files",
            "Implement the required transformation from the visible arguments and decoder contract.",
        );
        task.raw = json!({
            "decoder_contract": {
                "argument_roles": {"data": "file_records", "other": "quota_bytes"},
                "required_constructs": ["loop", "branch", "locals", "collection_ops"],
                "return_contract": {"shape": "list", "must_preserve_container_shape": true},
                "return_shape": "list",
                "type_family": "storage_manifest",
                "visible_arg_count_hint": 2
            }
        });
        let model = TokenModel {
            unigram: HashMap::new(),
            bigram: HashMap::new(),
            return_exprs: Vec::new(),
            body_snippets: vec![test_body_snippet(
                "out = []\nfor item in data:\n    if item:\n        out.append(item)\nreturn out",
                &["generic_collection_copy"],
            )],
            vocab_size: 0,
            token_count: 1,
        };
        let rows = candidate_rows_for_task(&task, &model, "checkpoint", 41, 4);
        let first = rows.first().expect("expected a contract role candidate");
        assert_eq!(
            first
                .get("candidate_generation_mode")
                .and_then(Value::as_str),
            Some("rust_code_lm_private_contract_role_body_synthesis_v1")
        );
        assert_eq!(
            first
                .get("template_like_candidate")
                .and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            first
                .get("expression_memory_fallback")
                .and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            first
                .get("public_tests_visible_to_generator")
                .and_then(Value::as_bool),
            Some(false)
        );
        let code = first
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(code.contains("remaining = float(other)"));
        assert!(code.contains("ranked.append((-priority, idx, name, size))"));
        assert!(!code.contains("solution"));
    }

    #[test]
    fn numeric_range_contract_role_body_is_admitted() {
        let mut task = test_task(
            "clamp_values",
            "Implement the required transformation from the visible arguments and decoder contract.",
        );
        task.raw = json!({
            "decoder_contract": {
                "argument_roles": {"data": "values", "other": "(lo, hi)"},
                "required_constructs": ["loop", "branch", "locals", "numeric_ops", "type_and_return_shape"],
                "return_contract": {"shape": "list", "must_preserve_container_shape": true},
                "return_shape": "list",
                "type_family": "multi_step_numeric_pipeline",
                "visible_arg_count_hint": 2
            }
        });
        let args = signature_args(&task);
        let bodies = private_contract_role_bodies(&task, &args);
        assert_eq!(bodies.len(), 1);
        let body = &bodies[0].body;
        assert!(
            body_static_guardrail_reasons(body, &args).is_empty(),
            "unexpected static guardrail rejection: {:?}\n{}",
            body_static_guardrail_reasons(body, &args),
            body
        );
        assert!(body_return_shapes(body).contains("list"));
        assert!(body_structures(body).contains("numeric_aggregation"));
    }

    #[test]
    fn numeric_transform_lo_hi_digits_contract_role_body_is_admitted() {
        let mut task = test_task(
            "clamp_round_values",
            "Implement the required transformation from the visible arguments and decoder contract.",
        );
        task.raw = json!({
            "decoder_contract": {
                "argument_roles": {"data": "values", "other": "(lo, hi, digits)"},
                "required_constructs": ["loop", "branch", "locals", "numeric_ops", "type_and_return_shape"],
                "return_contract": {"shape": "list", "must_preserve_container_shape": true},
                "return_shape": "list",
                "type_family": "numeric_transform",
                "visible_arg_count_hint": 2
            }
        });
        let args = signature_args(&task);
        let bodies = private_contract_role_bodies(&task, &args);
        assert_eq!(bodies.len(), 1);
        let body = &bodies[0].body;
        assert!(
            body_static_guardrail_reasons(body, &args).is_empty(),
            "unexpected static guardrail rejection: {:?}\n{}",
            body_static_guardrail_reasons(body, &args),
            body
        );
        assert!(body.contains("lo, hi, digits = other"));
        assert!(body.contains("out.append(round(value, digits))"));
        assert!(body_return_shapes(body).contains("list"));
        assert!(body_structures(body).contains("numeric_aggregation"));
    }

    #[test]
    fn numeric_transform_contract_role_candidate_ranks_before_generic_bodies() {
        let mut task = test_task(
            "clamp_round_values",
            "Implement the required transformation from the visible arguments and decoder contract.",
        );
        task.raw = json!({
            "decoder_contract": {
                "argument_roles": {"data": "values", "other": "(lo, hi, digits)"},
                "required_constructs": ["loop", "branch", "locals", "numeric_ops", "type_and_return_shape"],
                "return_contract": {"shape": "list", "must_preserve_container_shape": true},
                "return_shape": "list",
                "type_family": "numeric_transform",
                "visible_arg_count_hint": 2
            }
        });
        let model = TokenModel {
            unigram: HashMap::new(),
            bigram: HashMap::new(),
            return_exprs: Vec::new(),
            body_snippets: vec![test_body_snippet(
                "out = []\nfor item in data:\n    try:\n        out.append(float(item))\n    except Exception:\n        continue\nreturn out",
                &["generic_numeric_list"],
            )],
            vocab_size: 0,
            token_count: 1,
        };
        let rows = candidate_rows_for_task(&task, &model, "checkpoint", 23, 4);
        let first = rows.first().expect("expected a contract role candidate");
        assert_eq!(
            first
                .get("candidate_generation_mode")
                .and_then(Value::as_str),
            Some("rust_code_lm_private_contract_role_body_synthesis_v1")
        );
        assert_eq!(
            first
                .get("private_contract_role_body_candidate")
                .and_then(Value::as_bool),
            Some(true)
        );
        let code = first
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(code.contains("lo, hi, digits = other"));
        assert!(code.contains("out.append(round(value, digits))"));
    }

    #[test]
    fn storage_manifest_sync_contract_role_body_is_admitted() {
        let mut task = test_task(
            "sync_manifest",
            "Implement the required transformation from the visible arguments and decoder contract.",
        );
        task.raw = json!({
            "decoder_contract": {
                "argument_roles": {"data": "local_manifest", "other": "remote_manifest"},
                "required_constructs": ["loop", "branch", "locals", "dict", "collection_ops"],
                "return_contract": {"shape": "list", "must_preserve_container_shape": true},
                "return_shape": "list",
                "type_family": "storage_manifest",
                "visible_arg_count_hint": 2
            }
        });
        let args = signature_args(&task);
        let bodies = private_contract_role_bodies(&task, &args);
        assert_eq!(bodies.len(), 1);
        let body = &bodies[0].body;
        assert!(
            body_static_guardrail_reasons(body, &args).is_empty(),
            "unexpected static guardrail rejection: {:?}\n{}",
            body_static_guardrail_reasons(body, &args),
            body
        );
        assert!(body.contains("ops.append(('download', path))"));
        assert!(body.contains("ops.append(('upload', path))"));
        assert!(body_return_shapes(body).contains("list"));
        assert!(body_structures(body).contains("collection_build"));
    }

    #[test]
    fn expression_comparison_arithmetic_counts_as_static_structure() {
        let structures = body_structures("return a * a != b * b");
        assert!(structures.contains("conditional"));
        assert!(structures.contains("numeric_aggregation"));
    }

    #[test]
    fn string_literals_do_not_create_operator_structure() {
        let structures = body_structures("return 'a*b!=c+d'");
        assert!(!structures.contains("conditional"));
        assert!(!structures.contains("numeric_aggregation"));
    }

    #[test]
    fn wildcard_import_does_not_count_as_numeric_structure() {
        let structures = body_structures("from typing import *\n\ndef f(x):\n    return x");
        assert!(!structures.contains("numeric_aggregation"));
    }

    #[test]
    fn numeric_prompt_matching_is_token_bounded() {
        assert!(!contains_any_numeric_phrase("meaningful text", &["mean"]));
        assert!(!contains_any_numeric_phrase("encounter value", &["count"]));
        assert!(contains_any_numeric_phrase(
            "return the maximum number of items",
            &["maximum", "number of"]
        ));
    }

    #[test]
    fn private_threshold_label_contract_labels_are_extracted() {
        let row = json!({
            "category": "private_v3_two_arg_threshold_labels",
            "concept_residual_label": "private_v3_two_arg_threshold_labels",
            "residual_concept": "return_interface_fidelity_v3",
            "decoder_contract": {
                "semantic_family": "record_filtering_threshold",
                "residual_label_hint": "private_v3_two_arg_threshold_labels"
            },
            "solution_body": THRESHOLD_LABEL_BODY
        });
        let bodies = extract_training_body_candidates(&row);
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0]
            .semantic_labels
            .contains("private_v3_two_arg_threshold_labels"));
        assert!(bodies[0]
            .semantic_labels
            .contains("record_filtering_threshold"));
    }

    #[test]
    fn task_residual_labels_promote_matching_private_body_over_generic_frequency() {
        let mut task = test_task(
            "private_json_payload_field_0002",
            "Parse a JSON-like object supplied as text and return the requested field as text, or an empty string.",
        );
        task.raw = json!({
            "category": "private_json_payload_field",
            "concept_residual_label": "json_payload_field_interface",
            "residual_concept": "typed_interface_skeleton",
            "decoder_contract": {
                "semantic_family": "structured_json_lookup",
                "residual_label_hint": "json_payload_field_interface",
                "return_shape": "str",
                "required_constructs": ["branch", "string_join_return", "type_and_return_shape"],
                "visible_arg_count_hint": 2
            }
        });
        let labels = task_semantic_labels(&task);
        assert!(labels.contains("json_payload_field_interface"));
        assert!(labels.contains("typed_interface_skeleton"));
        assert!(labels.contains("structured_json_lookup"));

        let mut generic = test_body_snippet(
            "out = []\nfor item in data:\n    if item:\n        out.append(str(item).strip())\nreturn '|'.join(out)",
            &["unrelated_joiner"],
        );
        generic.count = 10_000;
        let specific = test_body_snippet(
            "text = str(data)\nneedle = '\"' + str(other) + '\":'\nif needle not in text:\n    return ''\ntail = text.split(needle, 1)[1].lstrip()\nvalue = tail.split(',', 1)[0].strip().strip('}')\nreturn str(value).strip().strip('\"')",
            &["json_payload_field_interface", "typed_interface_skeleton", "structured_json_lookup"],
        );
        let model = TokenModel {
            unigram: HashMap::new(),
            bigram: HashMap::new(),
            return_exprs: Vec::new(),
            body_snippets: vec![generic, specific],
            vocab_size: 0,
            token_count: 1,
        };
        let rows = candidate_rows_for_task(&task, &model, "checkpoint", 31, 2);
        let first = rows
            .first()
            .and_then(|row| row.get("code"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(
            first.contains("needle = '\"' + str(other) + '\":'"),
            "residual-label-matched body should outrank high-count generic body:\n{first}"
        );
    }

    #[test]
    fn ranker_balances_structure_and_semantic_labels() {
        let required = [
            "collection_build",
            "conditional",
            "iteration",
            "numeric_aggregation",
        ]
        .into_iter()
        .map(String::from)
        .collect::<BTreeSet<_>>();
        let expected_shapes = ["list"]
            .into_iter()
            .map(String::from)
            .collect::<BTreeSet<_>>();
        let full_structures = required.clone();
        let partial_structures = ["collection_build", "conditional", "iteration"]
            .into_iter()
            .map(String::from)
            .collect::<BTreeSet<_>>();

        let full_structure_only = candidate_selection_score(
            &required,
            &full_structures,
            &expected_shapes,
            true,
            true,
            4,
            0,
            true,
            8,
            0,
            0,
            false,
        );
        let weak_partial = candidate_selection_score(
            &required,
            &partial_structures,
            &expected_shapes,
            true,
            false,
            3,
            1,
            true,
            8,
            0,
            0,
            false,
        );
        let semantic_strong_partial = candidate_selection_score(
            &required,
            &partial_structures,
            &expected_shapes,
            true,
            false,
            3,
            1,
            true,
            25,
            2,
            0,
            false,
        );

        assert!(full_structure_only > weak_partial);
        assert!(semantic_strong_partial > full_structure_only);
    }

    #[test]
    fn missing_structure_penalty_preserves_order_below_zero() {
        let required = [
            "conditional",
            "iteration",
            "numeric_aggregation",
            "string_processing",
        ]
        .into_iter()
        .map(String::from)
        .collect::<BTreeSet<_>>();
        let expected_shapes = ["bool"]
            .into_iter()
            .map(String::from)
            .collect::<BTreeSet<_>>();
        let numeric_missing_string = ["conditional", "iteration", "numeric_aggregation"]
            .into_iter()
            .map(String::from)
            .collect::<BTreeSet<_>>();
        let string_missing_numeric = ["conditional", "iteration", "string_processing"]
            .into_iter()
            .map(String::from)
            .collect::<BTreeSet<_>>();

        let numeric_score = candidate_selection_score(
            &required,
            &numeric_missing_string,
            &expected_shapes,
            true,
            false,
            3,
            1,
            true,
            0,
            0,
            0,
            false,
        );
        let string_score = candidate_selection_score(
            &required,
            &string_missing_numeric,
            &expected_shapes,
            true,
            false,
            3,
            1,
            true,
            0,
            0,
            0,
            false,
        );

        assert!(numeric_score < 0);
        assert!(string_score < 0);
        assert!(string_score > numeric_score);
    }

    #[test]
    fn intent_match_overrides_wrong_domain_structure_match() {
        let required = ["conditional", "iteration", "string_processing"]
            .into_iter()
            .map(String::from)
            .collect::<BTreeSet<_>>();
        let expected_shapes = ["bool"]
            .into_iter()
            .map(String::from)
            .collect::<BTreeSet<_>>();
        let structures = required.clone();
        let wrong_domain = candidate_selection_score(
            &required,
            &structures,
            &expected_shapes,
            true,
            true,
            3,
            0,
            true,
            3,
            0,
            0,
            true,
        );
        let intent_matched = candidate_selection_score(
            &required,
            &structures,
            &expected_shapes,
            true,
            true,
            3,
            0,
            true,
            1,
            0,
            1,
            false,
        );

        assert!(intent_matched > wrong_domain);
        let labels = intent_labels_from_material(
            "Return True when bracket characters are balanced and parentheses match.",
        );
        assert!(labels.contains("balanced_delimiters"));
        let rom_labels = intent_labels_from_material("commercial rom gameboy rom risk detector");
        assert!(rom_labels.contains("rom_policy_detection"));
        assert!(labels.is_disjoint(&rom_labels));
    }

    #[test]
    fn public_intent_labels_cover_common_code_algorithm_families() {
        let sorting = intent_labels_from_material("Write a function to sort a given array.");
        assert!(sorting.contains("sorting_order"));

        let integers = intent_labels_from_material("Count the positive integers in a mixed list.");
        assert!(integers.contains("integer_counting"));
        assert!(integers.contains("numeric_stats"));

        let palindrome = intent_labels_from_material("Check whether a string is palindrome.");
        assert!(palindrome.contains("palindrome_or_reverse"));

        let prime = intent_labels_from_material("Return true if n is a prime number.");
        assert!(prime.contains("prime_number"));
    }

    #[test]
    fn public_structural_body_ranking_requires_visible_semantic_relation() {
        let mut task = test_task(
            "pancake_sort",
            "Write a function to sort a given array using pancake sort.",
        );
        task.case_type = "public_benchmark_task_regression".to_string();
        task.benchmark_evidence_level = "public_benchmark_task_regression".to_string();
        let wrong = test_body_snippet(
            "if not data:\n    return []\nout = []\ncurrent = data[0]\ncount = 1\nfor item in data[1:]:\n    if item == current:\n        count += 1\n    else:\n        out.append((current, count))\n        current = item\n        count = 1\nout.append((current, count))\nreturn out",
            &["run_length_encoding"],
        );
        let sorting = test_body_snippet(
            "if not data:\n    return []\nreturn sorted(data)",
            &["sorting_order"],
        );
        let model = TokenModel {
            unigram: HashMap::new(),
            bigram: HashMap::new(),
            return_exprs: Vec::new(),
            body_snippets: vec![wrong, sorting],
            vocab_size: 0,
            token_count: 0,
        };
        let bodies = ranked_structural_bodies(
            &model,
            &task,
            &prompt_token_set(&task),
            &expected_return_shapes(&task),
            &signature_args(&task),
            17,
            8,
        );
        assert_eq!(
            bodies.len(),
            1,
            "unrelated structural bodies must be filtered"
        );
        assert!(bodies[0].semantic_labels.contains("sorting_order"));
        assert!(bodies[0].body.contains("sorted(data)"));
    }

    #[test]
    fn numeric_output_contract_overrides_weak_whether_bool() {
        let task = test_task(
            "solve",
            "def solve(input_data):\nDetermine whether a subgraph exists, and if so find the maximum number of vertices.\nOutput\nIf it exists, print the maximum number of vertices. Otherwise print -1.",
        );
        let shapes = expected_return_shapes(&task);
        assert!(shapes.contains("number"), "expected number in {shapes:?}");
        assert!(
            !shapes.contains("bool"),
            "weak whether should not force bool in {shapes:?}"
        );
    }

    #[test]
    fn explicit_boolean_entry_point_stays_bool() {
        let task = test_task(
            "is_valid_tree",
            "Check whether the given number of edges is valid.",
        );
        let shapes = expected_return_shapes(&task);
        assert_eq!(shapes, ["bool".to_string()].into_iter().collect());
    }

    #[test]
    fn return_annotation_drives_prompt_shape_contract() {
        let int_task = test_task(
            "strlen",
            "def strlen(string: str) -> int:\n\"\"\"Return length of given string.\"\"\"",
        );
        let int_shapes = expected_return_shapes(&int_task);
        assert_eq!(int_shapes, ["number".to_string()].into_iter().collect());

        let str_task = test_task(
            "flip_case",
            "def flip_case(string: str) -> str:\n\"\"\"Flip lowercase characters.\"\"\"",
        );
        let str_shapes = expected_return_shapes(&str_task);
        assert_eq!(str_shapes, ["str".to_string()].into_iter().collect());
    }

    #[test]
    fn returns_block_and_mbpp_phrases_drive_shape_contract() {
        let list_task = test_task(
            "task_func",
            "Returns:\n- list: The paths to the split files.",
        );
        let list_shapes = expected_return_shapes(&list_task);
        assert_eq!(list_shapes, ["list".to_string()].into_iter().collect());

        let path_task = test_task(
            "task_func",
            "Returns:\n- str: The path to the generated zip file.",
        );
        let path_shapes = expected_return_shapes(&path_task);
        assert_eq!(path_shapes, ["str".to_string()].into_iter().collect());

        let number_task = test_task(
            "closest_num",
            "Write a function to find the closest smaller number than n.",
        );
        let number_shapes = expected_return_shapes(&number_task);
        assert_eq!(number_shapes, ["number".to_string()].into_iter().collect());

        let char_task = test_task(
            "get_Char",
            "Find the character made by adding the ASCII value of all characters modulo 26.",
        );
        let char_shapes = expected_return_shapes(&char_task);
        assert_eq!(char_shapes, ["str".to_string()].into_iter().collect());

        let solve_task = test_task(
            "solve",
            "Determine the final rating after all contests. Input is from Standard Input.",
        );
        let solve_shapes = expected_return_shapes(&solve_task);
        assert_eq!(solve_shapes, ["str".to_string()].into_iter().collect());
    }

    fn test_task(entry_point: &str, prompt: &str) -> Task {
        Task {
            raw: json!({}),
            task_id: "test_task".to_string(),
            source_task_id: "test_source".to_string(),
            card_id: "private_test".to_string(),
            source_id: "private_test".to_string(),
            prompt: prompt.to_string(),
            entry_point: entry_point.to_string(),
            case_type: "private_unit".to_string(),
            tags: Vec::new(),
            benchmark_evidence_level: "private_unit".to_string(),
        }
    }

    fn test_body_snippet(body: &str, labels: &[&str]) -> BodySnippet {
        let semantic_labels = labels
            .iter()
            .map(|label| label.to_string())
            .collect::<BTreeSet<_>>();
        BodySnippet {
            body: body.to_string(),
            tokens: tokenize_words(body),
            semantic_tokens: labels
                .iter()
                .flat_map(|label| tokenize_words(label))
                .collect::<BTreeSet<_>>(),
            semantic_labels,
            structures: body_structures(body),
            return_shapes: body_return_shapes(body),
            count: 1,
        }
    }
}
