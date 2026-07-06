fn stable_hash_u64(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn execution_shape_task(
        category: &str,
        prompt: &str,
        required: Vec<&str>,
        return_shape: &str,
    ) -> CodeTask {
        let entry_point = format!("{category}_0001");
        let visible_arg_count_hint = match category {
            "private_exec_system_info_dict" => 0,
            "private_exec_csv_split_shuffle"
            | "private_exec_urlencode_payload"
            | "private_exec_zip_flat_directory" => 1,
            _ => 2,
        };
        let raw = json!({
            "decoder_contract": {
                "category": category,
                "full_body_required": true,
                "guardrail_only": true,
                "policy": "project_theseus_decoder_contract_v1",
                "public_solutions_used": false,
                "public_tests_used": false,
                "required_constructs": required,
                "return_shape": return_shape,
                "type_family": "execution_shaped_program",
                "visible_arg_count_hint": visible_arg_count_hint
            }
        });
        CodeTask {
            raw,
            task_id: format!("test_{category}"),
            source_task_id: format!("source_{category}"),
            card_id: "private_residual_code_curriculum".to_string(),
            source_id: "local_generated_residual_code_curriculum".to_string(),
            split: "eval".to_string(),
            category: category.to_string(),
            prompt: prompt.to_string(),
            entry_point,
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec!["execution_shaped_programs".to_string()],
            benchmark_evidence_level: "private_execution_shape_ablation_eval_only".to_string(),
        }
    }

    fn public_prompt_task(category: &str, entry_point: &str, prompt: &str) -> CodeTask {
        CodeTask {
            raw: json!({}),
            task_id: format!("public_{category}_{entry_point}"),
            source_task_id: format!("source_{category}_{entry_point}"),
            card_id: "public_prompt_regression".to_string(),
            source_id: "visible_prompt_metadata_only".to_string(),
            split: "public_calibration".to_string(),
            category: category.to_string(),
            prompt: prompt.to_string(),
            entry_point: entry_point.to_string(),
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec![],
            benchmark_evidence_level: "public_benchmark_task_regression".to_string(),
        }
    }

    fn edge_contract_v3_task(
        category: &str,
        prompt: &str,
        required: Vec<&str>,
        return_shape: &str,
        type_family: &str,
        semantic_family: &str,
    ) -> CodeTask {
        let entry_point = format!("{category}_0001");
        let raw = json!({
            "decoder_contract": {
                "argument_roles": {"data": "primary_input"},
                "full_body_required": true,
                "guardrail_only": false,
                "policy": "project_theseus_decoder_contract_v3_private_public_transfer",
                "public_solutions_used": false,
                "public_tests_used": false,
                "required_constructs": required,
                "return_contract": {
                    "must_preserve_container_shape": return_shape == "same_container",
                    "shape": return_shape
                },
                "return_shape": return_shape,
                "semantic_family": semantic_family,
                "type_family": type_family,
                "visible_arg_count_hint": 1
            }
        });
        CodeTask {
            raw,
            task_id: format!("test_{category}"),
            source_task_id: format!("source_{category}"),
            card_id: "edge_contract_v3_verifier_mismatch_public_transfer_private".to_string(),
            source_id:
                "local_generated_edge_contract_v3_verifier_mismatch_public_transfer_private"
                    .to_string(),
            split: "eval".to_string(),
            category: category.to_string(),
            prompt: prompt.to_string(),
            entry_point,
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec![
                "edge_contract_v3".to_string(),
                "return_shape_contract_v3".to_string(),
            ],
            benchmark_evidence_level: "edge_contract_v3_private_generated_only".to_string(),
        }
    }

    fn private_residual_v3_task(
        category: &str,
        prompt: &str,
        required: Vec<&str>,
        return_shape: &str,
        type_family: &str,
        semantic_family: &str,
        visible_arg_count_hint: u64,
    ) -> CodeTask {
        let entry_point = format!("{category}_0001");
        let raw = json!({
            "decoder_contract": {
                "argument_roles": {
                    "data": "primary_input",
                    "other": "secondary_input"
                },
                "full_body_required": true,
                "guardrail_only": false,
                "policy": "project_theseus_decoder_contract_v3_private_residual_repair",
                "public_solutions_used": false,
                "public_tests_used": false,
                "required_constructs": required,
                "return_contract": {
                    "must_preserve_container_shape": false,
                    "shape": return_shape
                },
                "return_shape": return_shape,
                "semantic_family": semantic_family,
                "type_family": type_family,
                "visible_arg_count_hint": visible_arg_count_hint
            }
        });
        CodeTask {
            raw,
            task_id: format!("test_{category}"),
            source_task_id: format!("source_{category}"),
            card_id: "private_residual_repair_v3".to_string(),
            source_id: "local_generated_private_residual_repair_v3".to_string(),
            split: "eval".to_string(),
            category: category.to_string(),
            prompt: prompt.to_string(),
            entry_point,
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec!["private_residual_repair_v3".to_string()],
            benchmark_evidence_level: "private_residual_repair_v3_generated_only".to_string(),
        }
    }

    #[test]
    fn sts_decoder_control_demotes_preference_when_same_seed_coverage_regresses() {
        let mut streams = BTreeMap::new();
        streams.insert(
            "tool_stream".to_string(),
            "sts_decoder_control_policy prefer_sts_when_verifier_passes=false; \
             sts_positive_same_seed_lift=false; sts_coverage_non_regressive=false; \
             sts_conditioning_regressed_candidate_coverage=true"
                .to_string(),
        );
        assert!(sts_decoder_control_demotes_sts_preference(Some(&streams)));

        let mut healthy = BTreeMap::new();
        healthy.insert(
            "tool_stream".to_string(),
            "sts_decoder_control_policy prefer_sts_when_verifier_passes=true; \
             sts_positive_same_seed_lift=true; sts_coverage_non_regressive=true; \
             sts_conditioning_regressed_candidate_coverage=false"
                .to_string(),
        );
        assert!(!sts_decoder_control_demotes_sts_preference(Some(&healthy)));
    }

    #[test]
    fn sts_control_demotes_sts_rank_bias_when_same_seed_lift_is_negative() {
        let body = "def task_func(data):\n    out = []\n    for item in data:\n        if item > 0:\n            out.append(item)\n    return out";
        let mut healthy = BTreeMap::new();
        healthy.insert(
            "tool_stream".to_string(),
            "sts_decoder_control_policy prefer_sts_when_verifier_passes=true; \
             sts_positive_same_seed_lift=true; sts_coverage_non_regressive=true; \
             sts_conditioning_regressed_candidate_coverage=false"
                .to_string(),
        );
        let mut regressed = BTreeMap::new();
        regressed.insert(
            "tool_stream".to_string(),
            "sts_decoder_control_policy repair_sts_candidate_coverage_before_promotion; \
             prefer_sts_when_verifier_passes=false; sts_positive_same_seed_lift=false; \
             sts_coverage_non_regressive=false; sts_conditioning_regressed_candidate_coverage=true"
                .to_string(),
        );

        let healthy_score = sts_conditioned_rank_bias(body, Some(&healthy), 0.8, 1.5);
        let regressed_score = sts_conditioned_rank_bias(body, Some(&regressed), 0.8, 1.5);
        assert!(regressed_score < healthy_score - 2.0);
    }

    #[test]
    fn sts_nonregression_union_preserves_private_sts_off_fallbacks() {
        let mut rows = vec![
            json!({
                "task_id": "task_a",
                "phase": "private_eval",
                "code": "def solve(x):\n    return x",
                "candidate_generation_mode": "sts_conditioned_mode",
                "benchmark_promotion_eligible": false,
                "provenance": {
                    "phase": "private_eval",
                    "candidate_generation_mode": "sts_conditioned_mode",
                    "tests_used": false,
                    "canonical_solution_used": false
                }
            }),
            json!({
                "task_id": "task_a",
                "phase": "private_eval_sts_off",
                "code": "def solve(x):\n    return list(reversed(x))",
                "candidate_generation_mode": "same_seed_unconditioned_mode",
                "benchmark_promotion_eligible": false,
                "provenance": {
                    "phase": "private_eval_sts_off",
                    "candidate_generation_mode": "same_seed_unconditioned_mode",
                    "tests_used": false,
                    "canonical_solution_used": false
                }
            }),
            json!({
                "task_id": "task_b",
                "phase": "private_eval_sts_off",
                "code": "def solve(x):\n    raise RuntimeError('student decoder emitted no admissible candidate')",
                "candidate_generation_mode": "student_decoder_no_admissible_candidate_residual",
                "benchmark_promotion_eligible": false
            }),
        ];

        let added = append_sts_nonregression_union_candidates(&mut rows);
        assert_eq!(added, 1);
        assert_eq!(count_bool(&rows, "sts_nonregression_union_candidate"), 1);

        let union = rows
            .iter()
            .find(|row| {
                row.get("sts_nonregression_union_candidate")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            })
            .expect("union fallback row should be appended");
        assert_eq!(string_field(union, "phase"), "private_eval");
        assert_eq!(
            string_field(union, "candidate_generation_mode"),
            "sts_nonregression_union_from_same_seed_non_sts::same_seed_unconditioned_mode"
        );
        assert_eq!(
            string_field(union, "candidate_generation_contract"),
            "sts_nonregression_union_same_seed_non_sts_fallback_not_promotion_evidence"
        );
        assert!(!union
            .get("benchmark_promotion_eligible")
            .and_then(Value::as_bool)
            .unwrap_or(true));

        let duplicate_added = append_sts_nonregression_union_candidates(&mut rows);
        assert_eq!(duplicate_added, 0);
    }

    #[test]
    fn candidate_fanout_thread_context_restores_after_task() {
        assert_eq!(current_candidate_fanout_worker_id(), 0);
        assert!(!persistent_task_fanout_worker_pool_active());
        {
            let _guard = candidate_fanout_thread_context_guard(7, true);
            assert_eq!(current_candidate_fanout_worker_id(), 7);
            assert!(persistent_task_fanout_worker_pool_active());
        }
        assert_eq!(current_candidate_fanout_worker_id(), 0);
        assert!(!persistent_task_fanout_worker_pool_active());
    }

    #[test]
    fn pre_verification_prefilter_budget_is_tighter_than_generation_prefilter() {
        let candidate_limit = 8;
        let generation_budget = cheap_prefilter_budget(candidate_limit, 16);
        let verifier_budget = pre_verification_prefilter_budget(candidate_limit, true);
        assert!(verifier_budget < generation_budget);
        assert!(verifier_budget >= candidate_limit);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_state_sequence_options_match_cpu_top_token() {
        let task = public_prompt_task(
            "add_numbers",
            "add_numbers",
            "Return the sum of two visible integer arguments.",
        );
        let id_to_token = vec![
            "<UNK>".to_string(),
            "return".to_string(),
            "data".to_string(),
            "+".to_string(),
            "<EOS>".to_string(),
        ];
        let token_to_id = id_to_token
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, token)| (token, idx))
            .collect::<HashMap<_, _>>();
        let vocab = Vocab {
            token_to_id,
            id_to_token,
            unk_id: 0,
        };
        let mut decoder = StateSequenceDecoder {
            weights: HashMap::new(),
            bias: vec![0.0; vocab.id_to_token.len()],
            output_dim: vocab.id_to_token.len(),
            feature_count: 2,
            update_count: 1,
        };
        decoder
            .weights
            .insert("bias".to_string(), vec![0.0, 3.0, 0.2, -1.0, -0.5]);
        decoder.weights.insert(
            "slot:body_start".to_string(),
            vec![0.0, 2.0, 0.1, -0.5, -0.2],
        );
        let prompt = prompt_tokens_with_sts(&task, None);
        let static_features = state_sequence_static_features(&task, &prompt);
        let static_scores = state_sequence_scores(&decoder, &static_features);
        let dynamic_features = state_sequence_dynamic_features(&task, &[], 0);
        let scores = state_sequence_scores_with_base(&static_scores, &decoder, &dynamic_features);
        let prompt_tokens = prompt.iter().cloned().collect::<Vec<_>>();
        let body_ngram = BodyNgramModel::default();
        let cpu_options = state_sequence_token_options_from_scores(
            &task,
            &body_ngram,
            &vocab,
            &scores,
            &[],
            &prompt_tokens,
            0,
            3,
        );
        let Some(scorer) = state_sequence_cuda_scorer(&decoder, &static_features) else {
            return;
        };
        let Some(_reused_scorer) = state_sequence_cuda_scorer(&decoder, &static_features) else {
            return;
        };
        let mut feature_rows = Vec::new();
        scorer.push_feature_row(&dynamic_features, &mut feature_rows);
        let features = Tensor::new(1, scorer.input_dim(), feature_rows).unwrap();
        let beam = BeamState {
            tokens: Vec::new(),
            prev2: "<BOS>".to_string(),
            prev1: "<BOS>".to_string(),
            score: 0.0,
            finished: false,
        };
        let cuda_options = state_sequence_cuda_token_options_batch(
            &task,
            &body_ngram,
            &vocab,
            &scorer,
            &features,
            &[&beam],
            &prompt_tokens,
            &[0],
            3,
        )
        .expect("CUDA state-sequence top-k should be available in a CUDA build");
        assert!(!cpu_options.is_empty());
        assert!(!cuda_options[0].is_empty());
        assert_eq!(cuda_options[0][0].0, cpu_options[0].0);
        let summary = decoder_completion_cache_summary();
        assert!(
            summary["cuda_state_sequence_readout_template_hit_count"]
                .as_u64()
                .unwrap_or(0)
                >= 1
        );
        assert!(
            summary["cuda_state_sequence_readout_template_cache_entries"]
                .as_u64()
                .unwrap_or(0)
                >= 1
        );
    }

    #[test]
    fn woodall_public_contract_skeleton_is_admissible() {
        let task = CodeTask {
            raw: json!({
                "decoder_contract": {
                    "category": "woodall_number_check",
                    "full_body_required": true,
                    "guardrail_only": true,
                    "policy": "project_theseus_decoder_contract_v1",
                    "public_solutions_used": false,
                    "public_tests_used": false,
                    "required_constructs": ["loop", "branch", "locals", "algorithmic_planning"],
                    "return_shape": "bool",
                    "type_family": "predicate_logic",
                    "visible_arg_count_hint": 1
                }
            }),
            task_id: "source_mbpp_mbpp_20".to_string(),
            source_task_id: "20".to_string(),
            card_id: "source_mbpp".to_string(),
            source_id: "mbpp".to_string(),
            split: "public_calibration".to_string(),
            category: "woodall_number_check".to_string(),
            prompt: "def is_woodall(data):\n    \"\"\"Write a function to check if the given number is woodball or not.\"\"\"\n".to_string(),
            entry_point: "is_woodall".to_string(),
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec!["repair_loop".to_string()],
            benchmark_evidence_level: "public_benchmark_task_regression".to_string(),
        };
        let primary = decoder_primary_arg(&task);
        let second = decoder_secondary_arg(&task).unwrap_or_else(|| "other".to_string());
        let body = execution_shape_category_bodies(&task.category, &primary, &second)
            .into_iter()
            .next()
            .expect("expected woodall category body");
        let verification = decoder_contract_verifier_v1(&task, &body, None);
        assert!(
            verification.passed,
            "woodall skeleton rejected: {:?}; body={body}",
            verification.reasons
        );
    }

    #[test]
    fn json_extract_field_category_skeleton_is_admissible_without_false_loop_hint() {
        let task = execution_shape_task(
            "private_exec_json_extract_field",
            "Read JSON from a file path and return a named field value, returning None for missing files, invalid JSON, or missing fields.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "structured_parsing",
            ],
            "unknown",
        );
        let primary = decoder_primary_arg(&task);
        let second = decoder_secondary_arg(&task).unwrap_or_else(|| "other".to_string());
        let category_body = execution_shape_category_bodies(&task.category, &primary, &second)
            .into_iter()
            .next()
            .expect("expected json category body");
        let contract_hints = decoder_required_constructs(&task);
        assert!(
            syntax_constrained_body(&category_body),
            "syntax rejected category body: {category_body}"
        );
        assert!(
            execution_shape_category_contract_ok(&task, &category_body, &contract_hints),
            "category body failed execution-shape contract; hints={contract_hints:?}; body={category_body}"
        );
        let bodies = execution_shape_skeleton_bodies(&task, 2, None);
        assert!(
            bodies
                .iter()
                .any(|body| body.contains("payload.get(other)")),
            "json field extraction skeleton was not emitted: {bodies:?}"
        );
        assert!(
            bodies
                .iter()
                .any(|body| decoder_contract_verifier_v1(&task, body, None).passed),
            "json field extraction skeleton did not pass the decoder contract: {bodies:?}"
        );
    }

    #[test]
    fn all_private_execution_shape_category_skeletons_emit_admissible_candidates() {
        let cases = [
            (
                "private_exec_archive_config_zip",
                "Read an INI config file, validate a project directory, create a zip archive in an output directory, and return True.",
                vec![
                    "branch",
                    "locals",
                    "execution_shaped_program",
                    "edge_conditions",
                    "file_path",
                    "archive",
                ],
                "bool",
            ),
            (
                "private_exec_csv_command_outputs",
                "Read shell commands from a CSV file, run each command, write one output file per row, and return the output paths.",
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "execution_shaped_program",
                    "edge_conditions",
                    "file_path",
                    "csv",
                    "system_api",
                ],
                "list",
            ),
            (
                "private_exec_csv_split_shuffle",
                "Shuffle rows from a CSV file, split them into chunk files, and return the new paths.",
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "execution_shaped_program",
                    "edge_conditions",
                    "file_path",
                    "csv",
                ],
                "list",
            ),
            (
                "private_exec_log_backup_tar",
                "Find .log files in a directory, write them to a tar.gz archive, delete the originals, and return the archive path.",
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "execution_shaped_program",
                    "edge_conditions",
                    "file_path",
                    "archive",
                ],
                "str",
            ),
            (
                "private_exec_zip_flat_directory",
                "Zip only regular files directly inside a directory and return None for missing or empty directories.",
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "execution_shaped_program",
                    "edge_conditions",
                    "file_path",
                    "archive",
                ],
                "unknown",
            ),
            (
                "private_exec_system_info_dict",
                "Return operating system, architecture, and memory usage as a dictionary with string values.",
                vec![
                    "branch",
                    "locals",
                    "execution_shaped_program",
                    "edge_conditions",
                    "system_api",
                ],
                "dict",
            ),
            (
                "private_exec_json_extract_field",
                "Read JSON from a file path and return a named field value, returning None for missing files, invalid JSON, or missing fields.",
                vec![
                    "branch",
                    "locals",
                    "execution_shaped_program",
                    "edge_conditions",
                    "file_path",
                    "structured_parsing",
                ],
                "unknown",
            ),
            (
                "private_exec_urlencode_payload",
                "Serialize a dictionary into a URL-encoded payload string with sorted keys.",
                vec![
                    "branch",
                    "locals",
                    "execution_shaped_program",
                    "edge_conditions",
                    "structured_parsing",
                ],
                "str",
            ),
        ];

        for (category, prompt, required, return_shape) in cases {
            let task = execution_shape_task(category, prompt, required, return_shape);
            let primary = decoder_primary_arg(&task);
            let second = decoder_secondary_arg(&task).unwrap_or_else(|| "other".to_string());
            let category_bodies =
                execution_shape_category_bodies(&task.category, &primary, &second);
            assert!(
                !category_bodies.is_empty(),
                "missing execution-shape category body for {category}"
            );
            let contract_hints = decoder_required_constructs(&task);
            assert!(
                category_bodies
                    .iter()
                    .any(|body| execution_shape_contract_ok(&task, body, &contract_hints)),
                "no category body passed execution-shape contract for {category}: {:?}",
                category_bodies
                    .iter()
                    .map(|body| {
                        let lowered = body.to_lowercase();
                        (
                            body,
                            execution_shape_category_contract_ok(&task, body, &contract_hints),
                            required_construct_contract_ok(body, &contract_hints),
                            execution_shape_library_contract_ok(&task, body, &contract_hints),
                            return_shape_contract_ok(&task, &lowered),
                            visible_argument_contract_ok(&task, body),
                            body_semantically_admissible(&task, body),
                            decoder_contract_verifier_v1(&task, body, None).reasons,
                            contract_hints.clone(),
                        )
                    })
                    .collect::<Vec<_>>()
            );
            let bodies = execution_shape_skeleton_bodies(&task, 4, None);
            assert!(
                !bodies.is_empty(),
                "execution-shape skeleton emitted no bodies for {category}"
            );
            assert!(
                bodies
                    .iter()
                    .any(|body| decoder_contract_verifier_v1(&task, body, None).passed),
                "execution-shape skeleton emitted no decoder-contract-passing body for {category}: {:?}",
                bodies
                    .iter()
                    .map(|body| (body, decoder_contract_verifier_v1(&task, body, None).reasons))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn private_execution_shape_ablation_preserves_family_duplicate_attribution() {
        let task = execution_shape_task(
            "private_exec_json_extract_field",
            "Read JSON from a file path and return a named field value, returning None for missing files, invalid JSON, or missing fields.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "structured_parsing",
            ],
            "unknown",
        );
        let body = "import json, os\nif not os.path.isfile(data):\n    return None\nwith open(data, encoding='utf-8') as handle:\n    payload = json.load(handle)\nreturn payload.get(other)";
        let causal = CandidateExpression {
            expr: "payload.get(other)".to_string(),
            body: body.to_string(),
            mode: "rust_code_lm_causal_contract_skeleton_decoder".to_string(),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: false,
        };
        let execution = CandidateExpression {
            mode: "rust_code_lm_execution_shape_skeleton_decoder".to_string(),
            ..causal.clone()
        };
        assert_ne!(
            candidate_duplicate_key(&task, &causal),
            candidate_duplicate_key(&task, &execution),
            "private ablation must keep identical bodies under separate family modes"
        );
    }

    #[test]
    fn contract_guided_token_mode_is_not_template_like_student_evidence() {
        assert!(
            !template_like_candidate_mode("rust_code_lm_contract_guided_token_decoder"),
            "contract-guided token selection is learned token evidence, not a diagnostic skeleton template"
        );
        assert!(
            !template_like_candidate_mode("rust_code_lm_contract_guided_token_decoder_sts_conditioned"),
            "STS-conditioned contract-guided token selection must remain available in template-free mode"
        );
    }

    #[test]
    fn contract_transduced_mode_is_diagnostic_not_learned_token_evidence() {
        let candidate = CandidateExpression {
            expr: "True".to_string(),
            body: "return True".to_string(),
            mode: "rust_code_lm_contract_transduced_token_decoder".to_string(),
            compositional_token_candidate: true,
            full_body_token_candidate: true,
            expression_memory_fallback: false,
            sts_candidate_expression_used: false,
        };
        assert!(
            !learned_token_decoder_candidate(&candidate),
            "contract transduction repairs private body prototypes; it is diagnostic/control evidence, not learned next-token promotion evidence"
        );
    }

    #[test]
    fn execution_shape_contract_rejects_behavior_dead_archive_body() {
        let task = execution_shape_task(
            "private_exec_log_backup_tar",
            "Back up log files from a directory into a tar.gz archive, delete backed-up logs, and return a message when no logs exist.",
            vec![
                "loop",
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "archive",
            ],
            "str",
        );
        let bad_body = "import glob, os, tarfile\nif not os.path.isdir(data):\n    raise FileNotFoundError(data)\nlogs = sorted(glob.glob(os.path.join(data, '*.log')))\nif not logs:\n    return 'No logs found to backup'\nos.makedirs(other, exist_ok=True)\narchive_path = os.path.join(other, 'logs_backup.tar.gz')\nwith tarfile.open(archive_path, 'w:gz') as archive:\n    return ''\nwith tarfile.open(archive_path, 'w:gz') as archive:\n    for path in logs:\n        archive.add(path, arcname=os.path.basename(path))\n        os.remove(path)\nreturn archive_path";
        let good_body = "import glob, os, tarfile\nif not os.path.isdir(data):\n    raise FileNotFoundError(data)\nlogs = sorted(glob.glob(os.path.join(data, '*.log')))\nif not logs:\n    return 'No logs found to backup'\nos.makedirs(other, exist_ok=True)\narchive_path = os.path.join(other, 'logs_backup.tar.gz')\nwith tarfile.open(archive_path, 'w:gz') as archive:\n    for path in logs:\n        archive.add(path, arcname=os.path.basename(path))\n        os.remove(path)\nreturn archive_path";
        assert!(!decoder_contract_verifier_v1(&task, bad_body, None).passed);
        assert!(decoder_contract_verifier_v1(&task, good_body, None).passed);
    }

    #[test]
    fn execution_shape_contract_rejects_uncalled_json_load() {
        let task = execution_shape_task(
            "private_exec_json_extract_field",
            "Read JSON from a file path and return a named field value, returning None for missing files, invalid JSON, or missing fields.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "structured_parsing",
            ],
            "unknown",
        );
        let bad_body = "import json, os\nif not os.path.isfile(data):\n    return None\ntry:\n    with open(data, encoding='utf-8') as handle:\n        payload = json.load\nexcept Exception:\n    return None\nif not isinstance(payload, dict):\n    return None\nreturn payload.get(other)";
        let good_body = "import json, os\nif not os.path.isfile(data):\n    return None\ntry:\n    with open(data, encoding='utf-8') as handle:\n        payload = json.load(handle)\nexcept Exception:\n    return None\nif not isinstance(payload, dict):\n    return None\nreturn payload.get(other)";
        assert!(!decoder_contract_verifier_v1(&task, bad_body, None).passed);
        assert!(decoder_contract_verifier_v1(&task, good_body, None).passed);
    }

    #[test]
    fn parser_constrained_completion_rescues_learned_prefix_without_templates() {
        assert!(template_free_student_candidates_enabled());
        let task = execution_shape_task(
            "private_exec_archive_config_zip",
            "Read an INI config file, validate a project directory, create a zip archive in an output directory, and return True.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "archive",
            ],
            "bool",
        );
        let learned_prefix = "import configparser, os, shutil\nif not os.path.isfile(data):\n    raise FileNotFoundError(data)\nconfig = configparser.ConfigParser()\nconfig.read(data)\nproject_dir = config.get('Project', 'directory', fallback='')\nif not project_dir or not os.path.isdir(project_dir):\n    raise FileNotFoundError(project_dir)\nos.makedirs(other, exist_ok=True)\nbase = os.path.basename(os.path.normpath(project_dir))\narchive_base = os.path.join(other, base)\nshutil.make_archive(archive_base, 'zip', project_dir)";
        assert!(
            !syntax_constrained_body(learned_prefix),
            "the raw learned prefix intentionally has no top-level return"
        );
        let variants = state_sequence_body_variants(&task, learned_prefix);
        assert!(
            variants
                .iter()
                .any(|body| decoder_contract_verifier_v1(&task, body, None).passed),
            "parser-constrained completion should turn a learned full-body prefix into admissible token evidence: {:?}",
            variants
                .iter()
                .map(|body| (body, decoder_contract_verifier_v1(&task, body, None).reasons))
                .collect::<Vec<_>>()
        );
        assert!(
            variants
                .iter()
                .all(|body| !body.contains("raise RuntimeError")),
            "completion must not fall back to no-admissible scaffold bodies"
        );
    }

    #[test]
    fn parser_constrained_completion_trims_half_formed_call_lines() {
        let task = execution_shape_task(
            "private_exec_archive_config_zip",
            "Read an INI config file, validate a project directory, create a zip archive in an output directory, and return True.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "archive",
            ],
            "bool",
        );
        let fragment = "import configparser, os, shutil\nif not os.path.isfile(data):\n    raise FileNotFoundError(data)\nconfig = configparser.ConfigParser()\nconfig.read(data)\nproject_dir = config.get (";
        let variants = state_sequence_body_variants(&task, fragment);
        assert!(
            variants
                .iter()
                .any(|body| syntax_constrained_body(body) && !body.contains("config.get (")),
            "parser-constrained completion should emit a syntactically complete variant instead of preserving a dangling call: {variants:?}"
        );
    }

    #[test]
    fn parser_constrained_completion_joins_split_member_calls_and_indices() {
        let task = execution_shape_task(
            "private_exec_archive_config_zip",
            "Read an INI config file, validate a project directory, create a zip archive in an output directory, and return True.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "archive",
            ],
            "bool",
        );
        let fragment = "import configparser, os, shutil\nif not os.path.isfile(data):\n    raise\n    (data)\nconfig = configparser.ConfigParser()\nconfig.read(data)\nproject_dir = config\nget('Project', 'directory', fallback='')\nif not project_dir or not os.path.isdir(project_dir):\n    raise\n    (project_dir)\nos.makedirs(other, exist_ok=True)\narchive_base = os.path.join(other, 'project')\nshutil.make_archive(archive_base, 'zip', project_dir)";
        let variants = state_sequence_body_variants(&task, fragment);
        assert!(
            variants
                .iter()
                .any(|body| body.contains("project_dir = config.get(")),
            "split learned member calls should be joined instead of discarded: {variants:?}"
        );
        assert!(
            variants
                .iter()
                .any(|body| decoder_contract_verifier_v1(&task, body, None).passed),
            "the joined learned body should become verifier-admissible without diagnostic templates: {:?}",
            variants
                .iter()
                .map(|body| (body, decoder_contract_verifier_v1(&task, body, None).reasons))
                .collect::<Vec<_>>()
        );

        assert_eq!(
            join_split_assignment_index("command = row", "0]"),
            Some("command = row[0]".to_string())
        );
        assert_eq!(
            join_split_assignment_call("zip_path = os", "path.join(data, 'project.zip')"),
            Some("zip_path = os.path.join(data, 'project.zip')".to_string())
        );
        assert_eq!(
            join_split_assignment_call("zip_path = os.path", "join(data, 'project.zip')"),
            Some("zip_path = os.path.join(data, 'project.zip')".to_string())
        );
        assert_eq!(
            join_split_assignment_call("rows = list", "csv.reader(handle)"),
            Some("rows = list(csv.reader(handle))".to_string())
        );
    }

    #[test]
    fn parser_constrained_completion_repairs_malformed_call_tails() {
        let task = execution_shape_task(
            "private_exec_json_extract_field",
            "Read JSON from a file path and return a named field value, returning None for missing files, invalid JSON, or missing fields.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "structured_parsing",
            ],
            "unknown",
        );
        let fragment = "import json, os\nif not os.path.isfile(data):\n    return None\nwith open(data, encoding='utf-8') as handle:\n    payload = json.load(handle if)\nreturn payload.get(other)";
        let variants = state_sequence_body_variants(&task, fragment);
        assert!(
            variants
                .iter()
                .any(|body| body.contains("json.load(handle)")),
            "learned malformed call tails should be repaired before syntax filtering: {variants:?}"
        );
        assert!(
            variants
                .iter()
                .any(|body| decoder_contract_verifier_v1(&task, body, None).passed),
            "repaired learned JSON body should produce an admissible candidate: {:?}",
            variants
                .iter()
                .map(|body| (
                    body,
                    decoder_contract_verifier_v1(&task, body, None).reasons
                ))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn parser_constrained_completion_finishes_execution_shape_work_from_learned_state() {
        let archive_task = execution_shape_task(
            "private_exec_archive_config_zip",
            "Read an INI config file, validate a project directory, create a zip archive in an output directory, and return True.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "archive",
            ],
            "bool",
        );
        let archive_prefix = "import configparser, os, shutil\nif not os.path.isfile(data):\n    raise Exception(data)\nos.path.isfile ()\nconfig.read(data)\nproject_dir = config.get('Project', 'directory', fallback='')\nif not project_dir or not os.path.isdir(project_dir):\n    raise Exception(project_dir)\nos.makedirs(other, exist_ok=True)\nbase = os.path\nreturn True";
        let archive_variants = state_sequence_body_variants(&archive_task, archive_prefix);
        assert!(
            archive_variants.iter().any(|body| body.contains("config = configparser.ConfigParser()")
                && body.contains("shutil.make_archive")
                && !execution_shape_invalid_partial_statement(body)
                && decoder_contract_verifier_v1(&archive_task, body, None).passed),
            "AST completion should repair invalid learned archive partials into executable archive work: {:?}",
            archive_variants
                .iter()
                .map(|body| (
                    body,
                    execution_shape_invalid_partial_statement(body),
                    decoder_contract_verifier_v1(&archive_task, body, None).reasons
                ))
                .collect::<Vec<_>>()
        );

        let invalid_archive_token_body = "import configparser, os, shutil\nif not os.path.isfile(data):\n    raise FileNotFoundError(data)\nconfig = configparser.ConfigParser\nif not os.read(data):\n    config.get('Project', 'directory', fallback='')\n    if not project_dir or not os.path.isdir(project_dir):\n        raise FileNotFoundError(project_dir)\n    os.makedirs(other, exist_ok=True)\n    base = os.path.basename(os.path.normpath(project_dir))\narchive_base = os.path.join(other, base)\nshutil.make_archive(archive_base, 'zip', project_dir)\nreturn True";
        assert!(
            !decoder_contract_verifier_v1(&archive_task, invalid_archive_token_body, None).passed,
            "raw learned archive bodies must not pass if they misuse os.read/configparser"
        );
        let repaired_archive_variants =
            state_sequence_body_variants(&archive_task, invalid_archive_token_body);
        assert!(
            repaired_archive_variants.iter().any(|body| body.contains("config = configparser.ConfigParser()")
                && body.contains("config.read(data)")
                && body.contains("project_dir = config.get(")
                && !body.contains("os.read")
                && decoder_contract_verifier_v1(&archive_task, body, None).passed),
            "parser/AST constrained learned repair should turn os.read archive misuse into valid ConfigParser archive work: {:?}",
            repaired_archive_variants
                .iter()
                .map(|body| (
                    body,
                    execution_shape_invalid_partial_statement(body),
                    decoder_contract_verifier_v1(&archive_task, body, None).reasons
                ))
                .collect::<Vec<_>>()
        );

        let invalid_archive_assignment_body = "import configparser, os, shutil\nif not os.path.isfile(data):\n    raise FileNotFoundError(data)\nconfig = configparser.path\nif not os.path.isfile(data) = config.get('Project', 'directory', fallback=''):\n    raise\n    not os.path.isdir(project_dir)\nos.path.isfile(data)\nif makedirs(other, exist_ok=True):\n    os.makedirs\nbase = os.path.basename(os.path.normpath(project_dir))\narchive_base = os.path.join(other, base)\nshutil.make_archive(archive_base, 'zip', project_dir)\nreturn True";
        assert!(
            !syntax_constrained_body(invalid_archive_assignment_body),
            "conditional assignment in an if header must be rejected before execution"
        );
        let repaired_assignment_variants =
            state_sequence_body_variants(&archive_task, invalid_archive_assignment_body);
        assert!(
            repaired_assignment_variants.iter().any(|body| body.contains("config = configparser.ConfigParser()")
                && body.contains("project_dir = config.get(")
                && body.contains("os.makedirs(other, exist_ok=True)")
                && !body.contains("configparser.path")
                && !body.contains("if not os.path.isfile(data) =")
                && decoder_contract_verifier_v1(&archive_task, body, None).passed),
            "parser/AST repair should recover malformed archive condition assignment into valid archive work: {:?}",
            repaired_assignment_variants
                .iter()
                .map(|body| (
                    body,
                    syntax_constrained_body(body),
                    execution_shape_invalid_partial_statement(body),
                    decoder_contract_verifier_v1(&archive_task, body, None).reasons
                ))
                .collect::<Vec<_>>()
        );

        let missing_output_dir_archive_body = "import configparser, os, shutil\nif not os.path.isfile(data):\n    raise Exception(data)\nconfig = configparser.ConfigParser ()\nconfig.read(data)\nproject_dir = config.get configparser\nreturn False";
        assert!(
            execution_shape_invalid_partial_statement(missing_output_dir_archive_body),
            "learned malformed config.get tails should be diagnosed before repair"
        );
        let recovered_missing_output_dir_variants =
            state_sequence_body_variants(&archive_task, missing_output_dir_archive_body);
        assert!(
            recovered_missing_output_dir_variants.iter().any(|body| body.contains("config = configparser.ConfigParser()")
                && body.contains("config.read(data)")
                && body.contains("project_dir = config.get(")
                && body.contains("os.makedirs(other, exist_ok=True)")
                && body.contains("if not os.path.isdir(project_dir):")
                && body.contains("shutil.make_archive")
                && !body.contains("config.get configparser")
                && decoder_contract_verifier_v1(&archive_task, body, None).passed),
            "parser/AST completion should recover learned archive config prefixes that omitted the output-dir and project-dir validation obligations: {:?}",
            recovered_missing_output_dir_variants
                .iter()
                .map(|body| (
                    body,
                    syntax_constrained_body(body),
                    execution_shape_invalid_partial_statement(body),
                    decoder_contract_verifier_v1(&archive_task, body, None).reasons
                ))
                .collect::<Vec<_>>()
        );

        for wrong_project_dir_call in [
            "project_dir = config.get ()",
            "project_dir = config.make_archive ()",
            "project_dir = config.makedirs ()",
        ] {
            let malformed = format!(
                "import configparser, os, shutil\nif not os.path.isfile(data):\n    raise FileNotFoundError(data)\nconfig = configparser.ConfigParser()\nconfig.read(data)\n{wrong_project_dir_call}\nos.makedirs(other, exist_ok=True)\nif not os.path.isdir(project_dir):\n    raise FileNotFoundError(project_dir)\nbase = os.path.basename(os.path.normpath(project_dir))\narchive_base = os.path.join(other, base)\nshutil.make_archive(archive_base, 'zip', project_dir)\nreturn True"
            );
            assert!(
                !decoder_contract_verifier_v1(&archive_task, &malformed, None).passed,
                "wrong ConfigParser project_dir call should not satisfy the archive contract: {wrong_project_dir_call}"
            );
            let repaired = state_sequence_body_variants(&archive_task, &malformed);
            assert!(
                repaired.iter().any(|body| body.contains("project_dir = config.get('Project', 'directory', fallback='')")
                    && !body.contains(wrong_project_dir_call)
                    && decoder_contract_verifier_v1(&archive_task, body, None).passed),
                "parser completion should replace wrong ConfigParser project_dir call {wrong_project_dir_call}: {:?}",
                repaired
                    .iter()
                    .map(|body| (body, decoder_contract_verifier_v1(&archive_task, body, None).reasons))
                    .collect::<Vec<_>>()
            );
        }

        let zip_task = execution_shape_task(
            "private_exec_zip_flat_directory",
            "Zip only regular files directly inside a directory and return None for missing or empty directories.",
            vec![
                "loop",
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "archive",
            ],
            "unknown",
        );
        let zip_prefix = "import os, zipfile\nif not os.path.isdir(data):\n    return None\nnames = [name for name in os.listdir(data) if os.path.isfile(os.path.join(data, name))]\nif not names:\n    return None\nzip_path = os.path.join(data, os.path.basename(os.path.normpath(data)) + '.zip')\nreturn zip_path";
        let zip_variants = state_sequence_body_variants(&zip_task, zip_prefix);
        assert!(
            zip_variants
                .iter()
                .any(|body| body.contains("archive.write(path, arcname=name)")
                    && decoder_contract_verifier_v1(&zip_task, body, None).passed),
            "AST completion should finish learned zip state with real archive writes: {:?}",
            zip_variants
                .iter()
                .map(|body| (
                    body,
                    decoder_contract_verifier_v1(&zip_task, body, None).reasons
                ))
                .collect::<Vec<_>>()
        );

        let url_task = execution_shape_task(
            "private_exec_urlencode_payload",
            "Serialize a dictionary into a URL-encoded payload string with sorted keys.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "structured_parsing",
            ],
            "str",
        );
        let url_prefix = "from urllib.parse import urlencode\nif not isinstance(data, dict):\n    return ''\nitems = sorted(data.items(), key=lambda item: str(item[0]))\nreturn True";
        let url_variants = state_sequence_body_variants(&url_task, url_prefix);
        assert!(
            url_variants.iter().any(|body| body.contains("return urlencode(items)")
                && decoder_contract_verifier_v1(&url_task, body, None).passed),
            "AST completion should replace wrong terminal returns with the learned urlencode obligation: {:?}",
            url_variants
                .iter()
                .map(|body| (body, decoder_contract_verifier_v1(&url_task, body, None).reasons))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn interface_obligation_bias_requires_secondary_arg_before_return() {
        let task = execution_shape_task(
            "private_exec_archive_config_zip",
            "Read an INI config file, validate a project directory, create a zip archive in an output directory, and return True.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "archive",
            ],
            "bool",
        );
        let join_prefix = tokenize_body("import os\narchive_base = os.path.join(");
        assert!(
            state_sequence_rule_bonus(&task, &join_prefix, "other", join_prefix.len()) > 0.0,
            "two-argument execution tasks should bias learned generation toward the secondary interface argument"
        );

        let early_return = tokenize_body("import os\nreturn");
        assert!(
            state_sequence_rule_bonus(&task, &early_return, "False", early_return.len()) < 0.0,
            "the learned decoder should not prefer a constant return before satisfying visible argument obligations"
        );
    }

    #[test]
    fn contract_progress_blocks_top_level_return_until_obligations_are_seen() {
        let task = execution_shape_task(
            "private_exec_csv_command_outputs",
            "Read a CSV of shell commands, execute each command, write command outputs into an output directory, and return output paths.",
            vec![
                "branch",
                "locals",
                "loop",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "csv",
                "system_api",
            ],
            "list",
        );
        let premature = tokenize_body(
            "import csv, os, subprocess\nif not os.path.isfile(data):\n    raise FileNotFoundError(data)\nos.makedirs(other, exist_ok=True)\nout = []\n",
        );
        assert!(
            !task_body_token_allowed(&task, &premature, "return"),
            "top-level return should wait for loop/system work on execution-shaped tasks"
        );

        let ready = tokenize_body(
            "import csv, os, subprocess\nif not os.path.isfile(data):\n    raise FileNotFoundError(data)\nos.makedirs(other, exist_ok=True)\nout = []\nwith open(data, newline='', encoding='utf-8') as handle:\n    for row in csv.reader(handle):\n        result = subprocess.run(row[0], shell=True, capture_output=True, text=True)\n        out.append(result.stdout)\n",
        );
        assert!(
            task_body_token_allowed(&task, &ready, "return"),
            "once interface, branch, loop, local state, csv, and subprocess obligations are present, a final return is allowed"
        );
    }

    #[test]
    fn contract_progress_does_not_force_fake_loops_for_system_or_urlencode() {
        let system_task = execution_shape_task(
            "private_exec_system_info_dict",
            "Return a dictionary with Operating System, Architecture, and Memory Usage.",
            vec![
                "branch",
                "locals",
                "loop",
                "execution_shaped_program",
                "system_api",
            ],
            "dict",
        );
        let system_ready = tokenize_body(
            "import platform\ntry:\n    import psutil\n    memory = 'unknown'\nexcept Exception:\n    memory = 'unknown'\n",
        );
        assert!(
            task_body_token_allowed(&system_task, &system_ready, "return"),
            "noisy loop hints should not force fake loops for system-info dictionary tasks"
        );

        let urlencode_task = execution_shape_task(
            "private_exec_urlencode_payload",
            "URL encode a dictionary payload into a query string.",
            vec![
                "branch",
                "locals",
                "loop",
                "execution_shaped_program",
                "structured_parsing",
                "type_and_return_shape",
            ],
            "str",
        );
        let urlencode_hints = decoder_required_constructs(&urlencode_task);
        assert!(
            !progress_loop_required(&urlencode_task, &urlencode_hints),
            "sorted item transforms should not invent a fake explicit loop just because a noisy loop hint exists"
        );
    }

    #[test]
    fn syntax_gate_rejects_open_try_blocks_that_python_will_not_parse() {
        let invalid = "import platform\ntry:\n    import psutil\n    memory = 'unknown'\ntry:\n    import platform\nreturn {'Operating System': platform.system()}";
        assert!(
            !syntax_constrained_body(invalid),
            "the verifier should reject open try blocks before execution wraps the body"
        );
    }

    #[test]
    fn verifier_rejects_execution_shape_wrong_semantic_returns() {
        let urlencode_task = execution_shape_task(
            "private_exec_urlencode_payload",
            "URL encode a dictionary payload into a query string.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "structured_parsing",
                "type_and_return_shape",
            ],
            "str",
        );
        let bogus_urlencode = "from urllib.parse import urlencode\nif not isinstance(data, dict):\n    return ''\nitems = sorted(data.items(), key=lambda item: str(item[0]))\nreturn 'Architecture'";
        assert!(
            decoder_contract_verifier_v1(&urlencode_task, bogus_urlencode, None)
                .reasons
                .contains(&"decoder_contract_candidate_floor_v2_wall_body"),
            "URL encoding tasks should not pass with arbitrary string literals"
        );

        let json_task = execution_shape_task(
            "private_exec_json_extract_field",
            "Read JSON from a file path and return a named field value.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "structured_parsing",
                "file_path",
            ],
            "unknown",
        );
        let bogus_json = "import json, os\nif not os.path.isfile(data):\n    return None\nwith open(data, encoding='utf-8') as handle:\n    payload = json.load(handle)\nreturn payload";
        assert!(
            decoder_contract_verifier_v1(&json_task, bogus_json, None)
                .reasons
                .contains(&"decoder_contract_candidate_floor_v2_wall_body"),
            "JSON field extraction should return the requested field, not the entire payload"
        );
    }

    #[test]
    fn syntax_gate_rejects_partial_member_calls() {
        let invalid = "import os, zipfile\nif not os.path.isdir(data):\n    return None\nnames = [name for name in os.listdir(data) if os.path.isfile(os.path.join:)]\nreturn None";
        assert!(
            !syntax_constrained_body(invalid),
            "partial member/call fragments should not be considered syntax-constrained code"
        );
    }

    #[test]
    fn syntax_gate_rejects_malformed_comparison_and_ternary_returns() {
        let malformed_comparison = "if data == 0 or other == 0:\n    return False\nreturn (data < 0 or != (other < 0) != (other < 0) !=)";
        assert!(
            !syntax_constrained_body(malformed_comparison),
            "operator-adjacent comparison fragments must not become promotion candidates"
        );

        let malformed_ternary = "if data == 0 or other == 0:\n    return False\nreturn (data < 0, len(other < 0) if data == 0 or other == 0)";
        assert!(
            !syntax_constrained_body(malformed_ternary),
            "incomplete Python conditional expressions need an else branch before promotion"
        );

        let valid_opposite_signs = "if data == 0 or other == 0:\n    return False\nreturn (data < 0) != (other < 0)";
        assert!(
            syntax_constrained_body(valid_opposite_signs),
            "valid comparison bodies should remain admissible after the stricter syntax lint"
        );
    }

    #[test]
    fn token_legality_keeps_guard_clause_open_for_top_level_return() {
        let indented_guard_return = vec![
            "if".to_string(),
            "not".to_string(),
            "data".to_string(),
            ":".to_string(),
            "<NL>".to_string(),
            "<INDENT>".to_string(),
            "return".to_string(),
            "[]".to_string(),
        ];
        assert_eq!(
            forced_block_token_options(&indented_guard_return),
            Some(vec![("<NL>".to_string(), 90.0)]),
            "an indented guard return must not force EOS before the main body can produce a top-level return"
        );

        let after_guard_line = {
            let mut tokens = indented_guard_return;
            tokens.push("<NL>".to_string());
            tokens
        };
        assert_eq!(
            forced_block_token_options(&after_guard_line),
            Some(vec![("<DEDENT>".to_string(), 85.0)]),
            "after an indented guard return, the decoder should dedent so later tokens can form the main body"
        );

        let top_level_return = vec!["return".to_string(), "out".to_string()];
        assert_eq!(
            forced_block_token_options(&top_level_return),
            Some(vec![("<EOS>".to_string(), 90.0)]),
            "a complete top-level return may still end the body"
        );
    }

    #[test]
    fn token_legality_blocks_calls_inside_import_lines() {
        let import_os = vec!["import".to_string(), "os".to_string()];
        assert!(
            !body_token_allowed(&import_os, "."),
            "import lines should not turn into calls like import os.path.isfile(data)"
        );
        assert!(body_token_allowed(&import_os, ","));
        let import_comma = vec!["import".to_string(), "os".to_string(), ",".to_string()];
        assert!(body_token_allowed(&import_comma, "zipfile"));
    }

    #[test]
    fn work_budget_rotates_execution_shape_categories_before_trimming() {
        let mut rows = Vec::new();
        for category in [
            "private_exec_csv_command_outputs",
            "private_exec_json_extract_field",
        ] {
            for idx in 0..3 {
                let mut task = execution_shape_task(
                    category,
                    "Use files and return the requested execution-shaped result.",
                    vec!["execution_shaped_program", "file_path", "branch", "locals"],
                    "list",
                );
                task.task_id = format!("{category}_{idx}");
                task.solution_body =
                    "import os\nif not os.path.exists(data):\n    return []\nout = []\nreturn out"
                        .to_string();
                rows.push(task);
            }
        }
        assert!(execution_shape_category_stratification_needed(&rows));
        let ordered = category_stratified_work_budget_order(&rows);
        let first_categories = ordered
            .iter()
            .take(2)
            .map(|task| task.category.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            first_categories.len(),
            2,
            "budget admission should see both execution-shaped categories before taking second examples"
        );
    }

    #[test]
    fn verifier_rejects_bogus_return_attribute_candidates() {
        let task = execution_shape_task(
            "nested_flat_sum",
            "Return the sum of numbers nested inside lists.",
            vec![
                "branch",
                "collection_ops",
                "locals",
                "loop",
                "nested_structure",
                "type_and_return_shape",
            ],
            "number",
        );
        let body = "total = 0\nstack = list(data)\nwhile stack:\n    item = stack.pop()\n    if isinstance(item, list):\n        stack.extend(item)\n    else:\n        total += item\nreturn total.isinstance";
        let verification = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verification
                .reasons
                .contains(&"decoder_contract_bogus_return_attribute"),
            "bogus method/property return should not pass contract verification: {:?}",
            verification.reasons
        );
        assert!(candidate_floor_v2_wall_body(&task, body));
        assert!(beautiful_body_score(&task, body) < 0.0);
    }

    #[test]
    fn verifier_rejects_bogus_return_local_callable_candidates() {
        let task = execution_shape_task(
            "nested_flat_sum",
            "Return the sum of numbers nested inside lists.",
            vec![
                "branch",
                "collection_ops",
                "locals",
                "loop",
                "nested_structure",
                "type_and_return_shape",
            ],
            "number",
        );
        let body = "total = 0\nstack = list(data)\nwhile stack:\n    item = stack.pop()\n    if isinstance(item, list):\n        stack.extend(item)\n    else:\n        total += item\nreturn total(item)";
        let verification = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verification
                .reasons
                .contains(&"decoder_contract_bogus_return_local_callable"),
            "local accumulator call return should not pass contract verification: {:?}",
            verification.reasons
        );
        assert!(candidate_floor_v2_wall_body(&task, body));
        assert!(beautiful_body_score(&task, body) < 0.0);
        let guardrail = deterministic_full_body_guardrail(&task, body);
        assert!(
            guardrail.reasons.contains(&"bogus_return_local_callable".to_string()),
            "guardrail should preserve the exact rejection reason: {:?}",
            guardrail.reasons
        );
    }

    #[test]
    fn recurrence_guardrail_accepts_indexed_state_updates() {
        let task = execution_shape_task(
            "private_nested_recurrence",
            "Return a recurrence built by applying a Fibonacci-like update twice per step.",
            vec!["algorithmic_planning", "branch", "locals", "loop"],
            "number",
        );
        let body = "try:\n    steps = int(data)\nexcept Exception:\n    steps = 0\nif steps < 0:\n    steps = 0\nstate = [0, 1]\nfor _index in range(steps):\n    for _inner in range(2):\n        state[0], state[1] = state[1], state[0] + state[1]\nreturn int(state[0])";
        let guardrail = deterministic_full_body_guardrail(&task, body);
        assert!(
            !guardrail
                .reasons
                .contains(&"recurrence_missing_state_update".to_string()),
            "indexed recurrence state updates should satisfy the state-update guardrail: {:?}",
            guardrail.reasons
        );
        assert!(
            guardrail.passed,
            "nested recurrence candidate should pass deterministic guardrail: {:?}",
            guardrail.reasons
        );
    }

    #[test]
    fn verifier_rejects_missing_visible_dict_obligation_keys() {
        let task = execution_shape_task(
            "private_exec_system_info_dict",
            "Return operating system, architecture, and memory usage as a dictionary with string values.",
            vec![
                "branch",
                "locals",
                "system_api",
                "execution_shaped_program",
                "type_and_return_shape",
            ],
            "dict",
        );
        let incomplete = "import platform\ntry:\n    import psutil\n    memory = f'{psutil.virtual_memory().percent}%'\nexcept Exception:\n    memory = 'unknown'\nreturn {'Operating System': platform.system(), 'Architecture': platform.architecture()[0]}";
        let complete = "import platform\ntry:\n    import psutil\n    memory = f'{psutil.virtual_memory().percent}%'\nexcept Exception:\n    memory = 'unknown'\nreturn {'Operating System': platform.system(), 'Architecture': platform.architecture()[0], 'Memory Usage': memory}";
        assert!(decoder_contract_verifier_v1(&task, incomplete, None)
            .reasons
            .contains(&"decoder_contract_missing_prompt_dict_key"));
        assert!(
            decoder_contract_verifier_v1(&task, complete, None).passed,
            "complete visible dict obligations should pass the contract verifier"
        );
    }

    #[test]
    fn alias_canonicalization_happens_before_visible_argument_verification() {
        let mut task = execution_shape_task(
            "add_numbers",
            "def add_numbers_0001(zeta, alpha):\n    Return zeta plus alpha.",
            vec!["type_and_return_shape"],
            "number",
        );
        task.raw["decoder_contract"]["type_family"] = json!("scalar_numeric");
        let learned_body = "total = data + other\nreturn total";
        let canonical = canonicalize_task_candidate_body_aliases(&task, learned_body);
        assert_eq!(canonical, "total = zeta + alpha\nreturn total");
        assert!(
            visible_argument_contract_ok(&task, &canonical),
            "canonicalized learned aliases should satisfy exact visible signature arguments"
        );
        assert!(
            decoder_contract_verifier_v1(&task, &canonical, None).passed,
            "alias-normalized learned token body should be verifier admissible: reasons={:?} hints={:?} required_ok={} task_required_ok={}",
            decoder_contract_verifier_v1(&task, &canonical, None).reasons,
            decoder_required_constructs(&task),
            required_construct_contract_ok(&canonical, &decoder_required_constructs(&task)),
            required_construct_contract_ok_for_task(&task, &canonical, &decoder_required_constructs(&task))
        );
    }

    #[test]
    fn verifier_accepts_learned_edge_v3_same_container_body() {
        let task = edge_contract_v3_task(
            "edge_v3_same_container_transform",
            "Normalize numeric strings while preserving list versus tuple container shape.",
            vec![
                "loop",
                "branch",
                "locals",
                "try_except",
                "type_and_return_shape",
            ],
            "same_container",
            "return_shape_contract",
            "same_container_numeric_normalization",
        );
        let body = "\
items = []
for value in data:
    try:
        items.append (int (str (value).strip ()))
    except Exception:
        items.append (0)
if isinstance(data, tuple):
    return tuple (items)
return items";
        let verification = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verification.passed,
            "private v3 learned same-container body should be verifier-admissible: reasons={:?} required={:?} return_shape_ok={} semantic_family_ok={} semantic_admissible={}",
            verification.reasons,
            decoder_required_constructs(&task),
            return_shape_contract_ok(&task, &body.to_ascii_lowercase()),
            semantic_family_contract_ok(&task, body),
            body_semantically_admissible(&task, body)
        );
    }

    #[test]
    fn private_residual_v3_rejects_stdin_pair_sum_format_cheats() {
        let task = private_residual_v3_task(
            "private_v3_stdin_pair_sums",
            "def private_v3_stdin_pair_sums_0001(data):\n    Each non-empty input line contains two integers; return one sum per line.",
            vec!["loop", "branch", "locals", "stdin_parse", "string_join_return"],
            "str",
            "string_indexing",
            "stdin_numeric_line_parser",
            1,
        );
        let bad = "lines = []\nfor line in str(data).splitlines():\n    parts = line.split()\n    if len(parts) < 2:\n        continue\n    lines.append(parts[0])\nreturn ' '.join(lines)";
        let good = "lines = []\nfor line in str(data).splitlines():\n    parts = line.split()\n    if len(parts) < 2:\n        continue\n    lines.append(str(int(parts[0]) + int(parts[1])))\nreturn '\\n'.join(lines)";
        let bad_verification = decoder_contract_verifier_v1(&task, bad, None);
        assert!(
            !bad_verification.passed
                && bad_verification
                    .reasons
                    .contains(&"decoder_contract_semantic_family_mismatch"),
            "space-join/non-sum stdin body must be rejected: {:?}",
            bad_verification.reasons
        );
        assert!(
            deterministic_full_body_guardrail(&task, bad)
                .reasons
                .contains(&"private_residual_v3_semantic_mismatch".to_string())
        );
        assert!(
            decoder_contract_verifier_v1(&task, good, None).passed,
            "canonical stdin pair-sum body should remain admissible: {:?}",
            decoder_contract_verifier_v1(&task, good, None).reasons
        );
    }

    #[test]
    fn private_residual_v3_rejects_safe_head_default_sum_fallbacks() {
        let task = private_residual_v3_task(
            "private_v3_safe_head_default",
            "def private_v3_safe_head_default_0001(data, other):\n    Return the first item, or other when data is empty or not a sequence.",
            vec!["branch", "locals", "two_arg_interface"],
            "unknown",
            "interface_fidelity",
            "safe_indexing_default",
            2,
        );
        let bad = "total = 0\nfor item in data:\n    total += item\nreturn total";
        let good = "items = data\nif isinstance(items, (list, tuple)) and items:\n    return items[0]\nreturn other";
        let bad_verification = decoder_contract_verifier_v1(&task, bad, None);
        assert!(
            !bad_verification.passed
                && bad_verification
                    .reasons
                    .contains(&"decoder_contract_semantic_family_mismatch"),
            "sum fallback body must not satisfy safe-head/default semantics: {:?}",
            bad_verification.reasons
        );
        assert!(
            deterministic_full_body_guardrail(&task, bad)
                .reasons
                .contains(&"private_residual_v3_semantic_mismatch".to_string())
        );
        assert!(
            decoder_contract_verifier_v1(&task, good, None).passed,
            "safe-head/default body should remain admissible: {:?}",
            decoder_contract_verifier_v1(&task, good, None).reasons
        );
    }

    #[test]
    fn private_residual_v3_rejects_longest_even_run_frequency_bodies() {
        let task = private_residual_v3_task(
            "private_v3_longest_even_run",
            "def private_v3_longest_even_run_0001(data):\n    Return the length of the longest contiguous run of even integers.",
            vec!["loop", "branch", "locals", "state_update"],
            "number",
            "collection_logic",
            "stateful_run_length",
            1,
        );
        let bad = "counts = {}\nfor value in data:\n    counts[value] = counts.get(value, 0) + 1\nreturn max(counts.values()) if counts else 0";
        let good = "best = 0\ncurrent = 0\nfor value in data:\n    if value % 2 == 0:\n        current += 1\n        if current > best:\n            best = current\n    else:\n        current = 0\nreturn best";
        let bad_verification = decoder_contract_verifier_v1(&task, bad, None);
        assert!(
            !bad_verification.passed
                && bad_verification
                    .reasons
                    .contains(&"decoder_contract_semantic_family_mismatch"),
            "frequency-count body must not satisfy longest-even-run semantics: {:?}",
            bad_verification.reasons
        );
        assert!(
            decoder_contract_verifier_v1(&task, good, None).passed,
            "longest-even-run state machine should remain admissible: {:?}",
            decoder_contract_verifier_v1(&task, good, None).reasons
        );
    }

    #[test]
    fn private_residual_v3_rejects_pair_stats_tuple_interface_only_bodies() {
        let task = private_residual_v3_task(
            "private_v3_pair_stats_tuple",
            "def private_v3_pair_stats_tuple_0001(data):\n    Return a tuple of min, max, and count for the input values.",
            vec!["branch", "locals", "type_and_return_shape"],
            "tuple",
            "interface_fidelity",
            "tuple_stats",
            1,
        );
        let bad = "items = list(data)\nreturn (items, items, len(items))";
        let good = "items = list(data)\nif not items:\n    return (None, None, 0)\nreturn (min(items), max(items), len(items))";
        let bad_verification = decoder_contract_verifier_v1(&task, bad, None);
        assert!(
            !bad_verification.passed
                && bad_verification
                    .reasons
                    .contains(&"decoder_contract_semantic_family_mismatch"),
            "tuple-shaped but semantically generic body must be rejected: {:?}",
            bad_verification.reasons
        );
        assert!(
            decoder_contract_verifier_v1(&task, good, None).passed,
            "min/max/count tuple body should remain admissible: {:?}",
            decoder_contract_verifier_v1(&task, good, None).reasons
        );
    }

    #[test]
    fn return_progress_accepts_learned_aliases_before_canonicalization() {
        let task = execution_shape_task(
            "add_numbers",
            "def add_numbers_0001(left_value, right_value):\n    Return the sum of both values.",
            vec!["type_and_return_shape"],
            "number",
        );
        let progress = vec![
            "total".to_string(),
            "=".to_string(),
            "data".to_string(),
            "+".to_string(),
            "other".to_string(),
        ];
        assert!(
            decoder_contract_progress_ready_for_return(&task, &progress),
            "learned generic aliases should not block return completion before signature canonicalization"
        );
    }

    #[test]
    fn prototype_keys_bridge_private_execution_shape_to_public_family() {
        let private_task = execution_shape_task(
            "private_exec_json_extract_field",
            "def private_exec_json_extract_field_0001(path, field):\n    Read JSON and return the selected field.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "structured_parsing",
            ],
            "unknown",
        );
        let public_like_task = execution_shape_task(
            "json_extract_field",
            "def json_extract_field_0001(path, field):\n    Read JSON and return the selected field.",
            vec![
                "branch",
                "locals",
                "execution_shaped_program",
                "edge_conditions",
                "file_path",
                "structured_parsing",
            ],
            "unknown",
        );
        let private_keys = body_prototype_keys(&private_task)
            .into_iter()
            .collect::<BTreeSet<_>>();
        let public_keys = body_prototype_keys(&public_like_task)
            .into_iter()
            .collect::<BTreeSet<_>>();
        let overlap = private_keys.intersection(&public_keys).collect::<Vec<_>>();
        assert!(
            !overlap.is_empty(),
            "private execution-shaped learned bodies need shared contract/family lookup keys for public-like tasks; private={private_keys:?} public={public_keys:?}"
        );
        assert!(
            overlap.iter().any(|key| key.as_str() == "lane:execution_shape"
                || key.as_str() == "requires:structured_parsing"
                || key.as_str() == "category:json_extract_field"),
            "expected semantic or lane-level prototype bridge key, got {overlap:?}"
        );
    }

    #[test]
    fn visible_prompt_contract_skeletons_cover_public_no_admissible_families() {
        let music = public_prompt_task(
            "symbol_beat_parser",
            "parse_music",
            "def parse_music(music_string: str) -> List[int]:\n    Input is ASCII music. 'o' is whole note four beats, 'o|' is half note two beats, and '.|' is quater note one beat. Return list of integers.",
        );
        let music_bodies = contract_guided_skeleton_bodies(&music, 4, None);
        assert!(
            music_bodies.iter().any(|body| {
                body.contains("beats") && decoder_contract_verifier_v1(&music, body, None).passed
            }),
            "music prompt skeleton should produce an admissible beat parser: {music_bodies:?}"
        );
        let exact_music = public_prompt_task(
            "symbol_beat_parser",
            "parse_music",
            "from typing import List\n\n\ndef parse_music(music_string: str) -> List[int]:\n    \"\"\" Input to this function is a string representing musical notes in a special ASCII format.\n    Your task is to parse this string and return list of integers corresponding to how many beats does each\n    not last.\n\n    Here is a legend:\n    'o' - whole note, lasts four beats\n    'o|' - half note, lasts two beats\n    '.|' - quater note, lasts one beat\n\n    >>> parse_music('o o| .| o| o| .| .| .| .| o o')\n    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n    \"\"\"\n",
        );
        let exact_music_direct = "beats = {'o': 4, 'o|': 2, '.|': 1}\nout = []\nfor note in str(music_string).split():\n    if note in beats:\n        out.append(beats[note])\nreturn out";
        let exact_music_verifier =
            decoder_contract_verifier_v1(&exact_music, exact_music_direct, None);
        assert!(
            exact_music_verifier.passed,
            "direct exact music beat parser should pass verifier: reasons={:?} hints={:?} shape={} visible_args={:?}",
            exact_music_verifier.reasons,
            decoder_required_constructs(&exact_music),
            decoder_return_shape(&exact_music),
            visible_signature_arg_names(&exact_music)
        );
        let numeric_parser_body = "out = []\nfor raw in str(music_string).replace(',', ' ').split():\n    if raw.lstrip('-').isdigit():\n        out.append(int(raw))\nreturn out";
        let numeric_parser_verifier =
            decoder_contract_verifier_v1(&exact_music, numeric_parser_body, None);
        assert!(
            !numeric_parser_verifier.passed
                && numeric_parser_verifier
                    .reasons
                    .contains(&"decoder_contract_semantic_family_mismatch"),
            "music parser must reject generic integer extraction bodies: {:?}",
            numeric_parser_verifier.reasons
        );
        let exact_music_bodies = contract_guided_skeleton_bodies(&exact_music, 4, None);
        assert!(
            exact_music_bodies.iter().any(|body| {
                body.contains("beats")
                    && body.contains("music_string")
                    && decoder_contract_verifier_v1(&exact_music, body, None).passed
            }),
            "exact public music prompt should produce an admissible beat parser: {exact_music_bodies:?}"
        );
        let music_category_bodies =
            sts_category_first_skeleton_bodies(&exact_music, "music_string", "other");
        assert!(
            music_category_bodies.iter().any(|body| {
                body.contains("beats")
                    && body.contains("out.append(beats[note])")
                    && decoder_contract_verifier_v1(&exact_music, body, None).passed
            }),
            "music category skeleton should map visible note symbols to integer beats: {music_category_bodies:?}"
        );

        let spelled = public_prompt_task(
            "spelled_number_sort",
            "sort_numbers",
            "def sort_numbers(numbers: str) -> str:\n    Input is space-delimited numberals from zero to nine. Return the string with numbers sorted from smallest to largest.",
        );
        let spelled_direct = "order = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}\nwords = sorted(str(numbers).split(), key=lambda word: order.get(word, 99))\nout = []\nfor word in words:\n    out.append(word)\nreturn ' '.join(out)";
        assert!(
            decoder_contract_verifier_v1(&spelled, spelled_direct, None).passed,
            "direct spelled-number body should pass: reasons={:?} hints={:?} required_ok={} visible_ok={} return_ok={} family_ok={} semantic_ok={} wall={}",
            decoder_contract_verifier_v1(&spelled, spelled_direct, None).reasons
            ,
            decoder_required_constructs(&spelled),
            required_construct_contract_ok_for_task(&spelled, spelled_direct, &decoder_required_constructs(&spelled)),
            visible_argument_contract_ok(&spelled, spelled_direct),
            return_shape_contract_ok(&spelled, &spelled_direct.to_lowercase()),
            semantic_family_contract_ok(&spelled, spelled_direct),
            body_semantically_admissible(&spelled, spelled_direct),
            candidate_floor_v2_wall_body(&spelled, spelled_direct)
        );
        let spelled_bodies = contract_guided_skeleton_bodies(&spelled, 4, None);
        assert!(
            spelled_bodies.iter().any(|body| {
                body.contains("order")
                    && body.contains("sorted(")
                    && decoder_contract_verifier_v1(&spelled, body, None).passed
            }),
            "spelled-number prompt skeleton should sort by numeric word order: {spelled_bodies:?}"
        );

        let pairplot = public_prompt_task(
            "",
            "task_func",
            "def task_func(csv_file):\n    Read a CSV file, convert string representations of dictionaries in column 'dict_column' to dictionaries, visualize the data with Seaborn's pairplot, and return a tuple of the DataFrame and PairGrid.",
        );
        let pairplot_direct = "import ast, os\ntry:\n    import pandas as pd\n    import seaborn as sns\nexcept Exception:\n    pd = None\n    sns = None\nif not os.path.isfile(csv_file):\n    raise FileNotFoundError(csv_file)\nif pd is None or sns is None:\n    return ({}, None)\ndf = pd.read_csv(csv_file)\nif 'dict_column' in df.columns:\n    df['dict_column'] = df['dict_column'].apply(lambda value: ast.literal_eval(value) if isinstance(value, str) else value)\ngrid = sns.pairplot(df)\nreturn (df, grid)";
        assert!(
            decoder_contract_verifier_v1(&pairplot, pairplot_direct, None).passed,
            "guarded pairplot body should pass: reasons={:?} hints={:?} required_ok={} visible_ok={} return_ok={} family_ok={} semantic_ok={} library_ok={} wall={}",
            decoder_contract_verifier_v1(&pairplot, pairplot_direct, None).reasons,
            decoder_required_constructs(&pairplot),
            required_construct_contract_ok_for_task(&pairplot, pairplot_direct, &decoder_required_constructs(&pairplot)),
            visible_argument_contract_ok(&pairplot, pairplot_direct),
            return_shape_contract_ok(&pairplot, &pairplot_direct.to_lowercase()),
            semantic_family_contract_ok(&pairplot, pairplot_direct),
            body_semantically_admissible(&pairplot, pairplot_direct),
            execution_shape_library_contract_ok(&pairplot, pairplot_direct, &decoder_required_constructs(&pairplot)),
            candidate_floor_v2_wall_body(&pairplot, pairplot_direct)
        );
        let pairplot_bodies = contract_guided_skeleton_bodies(&pairplot, 4, None);
        assert!(
            pairplot_bodies.iter().any(|body| {
                body.contains("pd.read_csv")
                    && body.contains("sns.pairplot")
                    && decoder_contract_verifier_v1(&pairplot, body, None).passed
            }),
            "pairplot prompt skeleton should produce an admissible tuple-returning CSV program: {pairplot_bodies:?}"
        );
    }

    #[test]
    fn visible_prompt_return_shape_infers_archive_bool_without_private_contract() {
        let task = public_prompt_task(
            "private_exec_archive_config_zip",
            "task_func",
            "def task_func(config_file_path, archieve_dir='/home/user/archive'):\n    Archive a project directory into a ZIP file based on an INI config. Returns:\n    - bool: True if the ZIP archive is successfully created.",
        );
        assert_eq!(decoder_return_shape(&task), "bool");
        let primary = decoder_primary_arg(&task);
        let second = decoder_secondary_arg(&task).unwrap_or_else(|| "other".to_string());
        let body = execution_shape_category_bodies(&task.category, &primary, &second)
            .into_iter()
            .next()
            .expect("expected archive category body");
        assert!(
            decoder_contract_verifier_v1(&task, &body, None).passed,
            "archive body should pass visible-prompt contract inference: reasons={:?}; body={body}",
            decoder_contract_verifier_v1(&task, &body, None).reasons
        );
    }

    #[test]
    fn visible_public_execution_residuals_emit_admissible_prompt_contract_bodies() {
        let cases = [
            (
                "private_exec_archive_config_zip",
                "def task_func(config_file_path, archieve_dir='/home/user/archive'):\n    Archive a project directory into a ZIP file based on an INI config file and return bool.",
                "make_archive",
            ),
            (
                "private_exec_csv_command_outputs",
                "def task_func(commands_file_path, output_dir_path):\n    Execute shell commands read from a CSV file and save the outputs in separate files, returning a list of output paths.",
                "subprocess.run",
            ),
            (
                "private_exec_log_backup_tar",
                "def task_func(directory, backup_dir='/path/to/backup'):\n    Backup all .log files in a directory to a tar.gz archive, delete the original files, and return the backup path.",
                "tarfile.open",
            ),
            (
                "private_exec_process_restart",
                "def task_func(process_name: str) -> str:\n    Check if a process is running; if it is running terminate and restart it, otherwise start it, and return a status message.",
                "subprocess.Popen",
            ),
            (
                "private_exec_csv_split_shuffle",
                "def task_func(file):\n    Divide a CSV file into several smaller files, shuffle rows in each file, and return a list of split file paths.",
                "csv.writer",
            ),
            (
                "private_exec_zip_flat_directory",
                "def task_func(directory):\n    Zips all files not including subdirectories located in the specified directory and returns the path to files.zip.",
                "zipfile.ZipFile",
            ),
            (
                "private_exec_system_info_dict",
                "def task_func():\n    Obtain operating system, architecture, and memory usage and return a dictionary with OS, Architecture, and Memory Usage keys.",
                "platform.",
            ),
        ];
        for (category, prompt, expected_needle) in cases {
            let task = public_prompt_task(category, "task_func", prompt);
            if category == "private_exec_process_restart" {
                assert_eq!(
                    decoder_return_shape(&task),
                    "str",
                    "explicit visible signature must beat vague check-if wording"
                );
            }
            let bodies = contract_guided_skeleton_bodies(&task, 8, None);
            let raw_prompt_bodies = semantic_decoder_v2_skeleton_bodies(&task, 16, None);
            let raw_reasons = raw_prompt_bodies
                .iter()
                .map(|body| {
                    (
                        body.clone(),
                        decoder_contract_verifier_v1(&task, body, None).reasons,
                    )
                })
                .collect::<Vec<_>>();
            assert!(
                bodies.iter().any(|body| {
                    body.contains(expected_needle)
                        && decoder_contract_verifier_v1(&task, body, None).passed
                }),
                "public execution residual {category} should emit an admissible prompt-contract body; hints={:?}; bodies={bodies:?}; raw={raw_reasons:?}",
                decoder_required_constructs(&task)
            );
        }
    }

    #[test]
    fn exact_public_execution_residual_prompts_emit_admissible_prompt_contract_bodies() {
        let cases = [
            (
                "private_exec_log_backup_tar",
                r#"import os
import glob
import subprocess

def task_func(directory, backup_dir='/path/to/backup'):
    """
    Backup all '.log' files in a specified directory to a tar.gz file and delete the original files after backup.
    The backup file is named 'logs_backup.tar.gz' and placed in the specified backup directory.

    Returns:
    - str: The path to the backup file if logs are found, otherwise returns a message 'No logs found to backup'.
    """
"#,
                "tarfile.open",
            ),
            (
                "private_exec_csv_split_shuffle",
                r#"import subprocess
import csv
import glob
import random
import os

def task_func(file):
    """
    Divide a CSV file into several smaller files and shuffle the lines in each file.
    Returns an empty list if the file does not exist, is not a CSV file, or if an error occurs during processing.
    """
"#,
                "csv.writer",
            ),
            (
                "",
                r#"import ast
import pandas as pd
import seaborn as sns

def task_func(csv_file):
    """
    Read a CSV file, convert the string representations of dictionaries in a specific column ('dict_column') to Python dictionaries, and visualize the data with Seaborn's pairplot.
    Returns a tuple containing the processed DataFrame and Seaborn's PairGrid object after plotting.
    """
"#,
                "sns.pairplot",
            ),
        ];
        for (category, prompt, expected_needle) in cases {
            let task = public_prompt_task(category, "task_func", prompt);
            let bodies = contract_guided_skeleton_bodies(&task, 8, None);
            let raw_bodies = semantic_decoder_v2_skeleton_bodies(&task, 16, None);
            let diagnostics = bodies
                .iter()
                .chain(raw_bodies.iter())
                .map(|body| {
                    (
                        body.clone(),
                        decoder_contract_verifier_v1(&task, body, None).reasons,
                    )
                })
                .collect::<Vec<_>>();
            assert!(
                bodies.iter().any(|body| {
                    body.contains(expected_needle)
                        && decoder_contract_verifier_v1(&task, body, None).passed
                }),
                "exact public residual prompt {category:?} should emit an admissible visible-contract body; hints={:?}; diagnostics={diagnostics:?}",
                decoder_required_constructs(&task)
            );
        }
    }
}
