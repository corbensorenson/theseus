use super::*;

fn private_task(tags: Vec<&str>, return_shape: &str) -> CodeTask {
    CodeTask {
            raw: json!({
                "residual_concept": tags.first().copied().unwrap_or("edge_case"),
                "decoder_contract": {
                    "return_shape": return_shape,
                    "required_constructs": ["branch", "locals"],
                    "type_family": "type_and_return_shape",
                    "visible_arg_count_hint": 2
                }
            }),
            task_id: "private_residual_test".to_string(),
            source_task_id: "private_residual_test".to_string(),
            card_id: "private_residual_code_curriculum".to_string(),
            source_id: "local_generated_residual_code_curriculum".to_string(),
            split: "eval".to_string(),
            category: "private_edge_type_shape".to_string(),
            prompt: "def repair(data, other):\n    \"\"\"Return a typed result with edge case handling.\"\"\"".to_string(),
            entry_point: "repair".to_string(),
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: tags.into_iter().map(str::to_string).collect(),
            benchmark_evidence_level: "private_residual_generated_eval_only".to_string(),
        }
}

fn public_metadata_task(
    category: &str,
    entry_point: &str,
    prompt: &str,
    return_shape: &str,
    type_family: &str,
    required: Vec<&str>,
) -> CodeTask {
    CodeTask {
        raw: json!({
            "decoder_contract": {
                "return_shape": return_shape,
                "required_constructs": required,
                "type_family": type_family,
                "public_tests_used": false,
                "public_solutions_used": false,
                "policy": "project_theseus_decoder_contract_v1"
            }
        }),
        task_id: format!("public_metadata_{category}"),
        source_task_id: format!("metadata_only_{category}"),
        card_id: "source_bigcodebench".to_string(),
        source_id: "visible_public_metadata_only".to_string(),
        split: "public_calibration".to_string(),
        category: category.to_string(),
        prompt: prompt.to_string(),
        entry_point: entry_point.to_string(),
        solution_expr: String::new(),
        solution_body: String::new(),
        tags: vec![],
        benchmark_evidence_level: "public_benchmark_metadata_only_no_tests".to_string(),
    }
}

#[test]
fn broad_transfer_residual_policy_detects_private_type_edge_pressure() {
    let task = private_task(vec!["edge_case", "type_and_return_shape"], "list");
    let policy = broad_transfer_residual_policy(&task);
    assert!(policy.edge_case);
    assert!(policy.type_handling);
    assert!(!policy.algorithm_choice);
}

#[test]
fn broad_transfer_residual_policy_detects_runtime_dependency_pressure() {
    let task = private_task(vec!["runtime_load_failure", "pandas"], "dict");
    let policy = broad_transfer_residual_policy(&task);
    assert!(policy.local_adapter);
    assert!(policy.runtime_dependency);
    assert!(policy
        .family_names()
        .contains(&"adapter_runtime_dependency_handling"));
}

#[test]
fn optional_dependency_prefilter_requires_guarded_imports() {
    let task = private_task(
        vec!["runtime_load_failure", "type_and_return_shape"],
        "list",
    );
    let guarded =
            "try:\n    import numpy as np\nexcept Exception:\n    np = None\nvalues = list(data) if isinstance(data, (list, tuple, set)) else []\nif other is not None:\n    values.append(other)\nout = []\nfor item in values:\n    out.append(item)\nreturn out";
    let unguarded = "import numpy as np\nvalues = np.asarray(data)\nreturn values.tolist()";
    let guarded_verifier = decoder_contract_verifier_v1(&task, guarded, None);
    assert!(
        guarded_verifier.passed,
        "guarded optional dependency body should pass: {:?}",
        guarded_verifier.reasons
    );
    assert!(
            broad_public_floor_recovery_prefilter_score(
                &task,
                guarded,
                "rust_code_lm_eligible_receiver_inventory_router_v1_runtime_dependency_guard_token_decoder",
            ) > broad_public_floor_recovery_prefilter_score(
                &task,
                unguarded,
                "rust_code_lm_eligible_receiver_inventory_router_v1_runtime_dependency_guard_token_decoder",
            )
        );
}

#[test]
fn runtime_dependency_receiver_uses_guarded_fallback_without_network_fetch() {
    let task = private_task(vec!["runtime_load_failure", "beautifulsoup4"], "str");
    let body = runtime_dependency_receiver_body("data", "other", true, "str", "''");

    assert!(body.contains("try:"));
    assert!(body.contains("BeautifulSoup = None"));
    assert!(!body.contains("requests.get"));
    assert!(!body.contains("http://"));

    let verifier = decoder_contract_verifier_v1(&task, &body, None);
    assert!(
        verifier.passed,
        "guarded runtime dependency receiver should pass: {:?}\n{}",
        verifier.reasons, body
    );
}

#[test]
fn optional_dependency_specific_inventory_covers_private_runtime_rows() {
    let cases = [
            (
                "private_runtime_optional_numpy_sum",
                "Return the numeric sum of mixed values, using numpy only when it is already available and falling back to pure Python otherwise.",
                "number",
                "runtime_dependency_numpy_sum",
                "general_semantics",
            ),
            (
                "private_runtime_optional_pandas_records",
                "Normalize tabular input into a list of dictionaries, using pandas only when present and otherwise accepting lists or dictionaries.",
                "list",
                "runtime_dependency_pandas_records",
                "collection_logic",
            ),
            (
                "private_runtime_optional_html_text",
                "Extract visible text from markup, using BeautifulSoup only when present and otherwise using a simple local fallback.",
                "str",
                "runtime_dependency_html_text",
                "string_indexing",
            ),
            (
                "private_runtime_optional_plot_summary",
                "Summarize plotting input without requiring matplotlib or seaborn, returning count, min, and max values.",
                "dict",
                "runtime_dependency_plot_summary",
                "collection_logic",
            ),
            (
                "private_runtime_optional_sklearn_labels",
                "Return deterministic integer labels for class labels, using sklearn only when available and falling back locally.",
                "list",
                "runtime_dependency_sklearn_labels",
                "collection_logic",
            ),
            (
                "private_runtime_optional_requests_query",
                "Return URL query parameters as a dictionary without performing network access; requests may be imported only behind a guard.",
                "dict",
                "runtime_dependency_requests_query",
                "string_indexing",
            ),
        ];
    for (category, prompt, shape, expected_family, type_family) in cases {
        let mut task = private_task(
            vec![
                "adapter_runtime_dependency_handling",
                "local_code_generation_adapter_needed",
                "optional_dependency",
                "runtime_load_failure",
            ],
            shape,
        );
        task.category = category.to_string();
        task.prompt = format!("def repair(data):\n    \"\"\"{prompt}\"\"\"");
        task.raw = json!({
            "residual_concept": "adapter_runtime_dependency_handling",
            "decoder_contract": {
                "return_shape": shape,
                "required_constructs": ["branch", "loop", "locals"],
                "type_family": type_family,
                "visible_arg_count_hint": 1
            }
        });

        let policy = broad_transfer_residual_policy(&task);
        assert!(policy.runtime_dependency);
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| panic!("missing {expected_family} receiver for {category}"));

        assert!(optional_dependency_import_contract_ok(body));
        assert!(optional_dependency_fallback_bonus(body) > 0.0);
        let hints = decoder_required_constructs(&task);
        assert!(
            required_construct_contract_ok_for_task(&task, body, &hints),
            "{expected_family} should satisfy private required constructs {:?}\n{}",
            hints,
            body
        );
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verifier.passed,
            "{expected_family} should pass private decoder contract: {:?}\n{}",
            verifier.reasons, body
        );
        if expected_family == "runtime_dependency_requests_query" {
            let generic_numeric_parser =
                "out = []\nfor raw in str(data).replace(',', ' ').split():\n    if raw.lstrip('-').isdigit():\n        out.append(int(raw))\nreturn out";
            assert!(
                broad_public_floor_recovery_prefilter_score(
                    &task,
                    body,
                    "rust_code_lm_private_shadow_private_to_public_receiver_inventory_bridge_v1_runtime_dependency_requests_query_token_decoder",
                ) > broad_public_floor_recovery_prefilter_score(
                    &task,
                    generic_numeric_parser,
                    "rust_code_lm_contract_guided_token_decoder",
                ),
                "requests query receiver should outrank generic numeric parsing\n{}\n{}",
                body,
                generic_numeric_parser
            );
        }
    }
}

#[test]
fn execution_shape_inventory_promotes_private_exec_rows_as_first_class_receivers() {
    let cases = [
            (
                "private_exec_archive_config_zip",
                "Read an INI config file, validate a project directory, create a zip archive in an output directory, and return True.",
                "bool",
                2usize,
                "execution_shape_archive_config_zip",
            ),
            (
                "private_exec_csv_command_outputs",
                "Read shell commands from a CSV file, run each command, write one output file per row, and return the output paths.",
                "list",
                2usize,
                "execution_shape_csv_command_outputs",
            ),
            (
                "private_exec_csv_split_shuffle",
                "Split a CSV file into smaller shuffled CSV chunks and return the created file paths, or an empty list for invalid input.",
                "list",
                1usize,
                "execution_shape_csv_split_shuffle",
            ),
            (
                "private_exec_json_extract_field",
                "Read JSON from a file path and return a named field value, returning None for missing files, invalid JSON, or missing fields.",
                "unknown",
                2usize,
                "execution_shape_json_extract_field",
            ),
            (
                "private_exec_log_backup_tar",
                "Back up log files from a directory into a tar.gz archive, delete backed-up logs, and return a message when no logs exist.",
                "str",
                2usize,
                "execution_shape_log_backup_tar",
            ),
            (
                "private_exec_system_info_dict",
                "Return operating system, architecture, and memory usage as a dictionary with string values.",
                "dict",
                0usize,
                "execution_shape_system_info_dict",
            ),
            (
                "private_exec_urlencode_payload",
                "Serialize a dictionary into a URL-encoded payload string with sorted keys.",
                "str",
                1usize,
                "execution_shape_urlencode_payload",
            ),
            (
                "private_exec_zip_flat_directory",
                "Zip only regular files directly inside a directory and return None for missing or empty directories.",
                "str",
                1usize,
                "execution_shape_zip_flat_directory",
            ),
        ];
    for (category, prompt, shape, visible_arg_count_hint, expected_family) in cases {
        let mut task = private_task(
            vec![
                "execution_shaped_programs",
                "local_code_generation_adapter_needed",
                "algorithm_choice",
                "private_residual_curriculum",
            ],
            shape,
        );
        task.category = category.to_string();
        let signature = match visible_arg_count_hint {
            0 => "def repair():",
            1 => "def repair(data):",
            _ => "def repair(data, other):",
        };
        task.prompt = format!("{signature}\n    \"\"\"{prompt}\"\"\"");
        task.raw = json!({
            "residual_concept": "execution_shaped_programs",
            "decoder_contract": {
                "return_shape": shape,
                "required_constructs": ["branch", "locals"],
                "type_family": "execution_shaped_program",
                "visible_arg_count_hint": visible_arg_count_hint
            }
        });

        let policy = broad_transfer_residual_policy(&task);
        assert!(
            policy.algorithm_choice && policy.local_adapter,
            "{category} should activate execution-shaped local adapter and algorithm policy"
        );
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| panic!("missing {expected_family} receiver for {category}"));
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verifier.passed,
            "{expected_family} should pass private execution-shaped decoder contract: {:?}\n{}",
            verifier.reasons, body
        );

        let mut rows = Vec::new();
        assert!(
            append_eligible_receiver_inventory_router_candidates(&task, &mut rows, 8) > 0,
            "{expected_family} should survive receiver prefiltering"
        );
        assert!(
            rows.iter()
                .any(|candidate| candidate.mode.contains(expected_family)),
            "{expected_family} should be emitted as a first-class mode: {:?}",
            rows.iter().map(|row| row.mode.as_str()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn runtime_dependency_prefilter_requires_actual_guarded_optional_dependency() {
    let task = private_task(vec!["runtime_load_failure", "pandas"], "dict");
    let no_import = "out = {}\nif isinstance(data, dict):\n    out.update(data)\nif other is not None:\n    out[other] = out.get(other, 0)\nreturn out";
    let guarded = runtime_dependency_receiver_body("data", "other", true, "dict", "{}");
    let mode =
        "rust_code_lm_eligible_receiver_inventory_router_v1_runtime_dependency_guard_token_decoder";

    assert!(
        broad_public_floor_recovery_prefilter_score(&task, &guarded, mode)
            > broad_public_floor_recovery_prefilter_score(&task, no_import, mode),
        "runtime dependency pressure should prefer guarded optional imports plus fallback"
    );
}

#[test]
fn eligible_receiver_inventory_reuses_private_residual_adapter_bodies() {
    let mut task = private_task(
        vec![
            "local_code_generation_adapter_needed",
            "csv",
            "type_and_return_shape",
        ],
        "list",
    );
    task.category = "private_exec_csv_split_shuffle".to_string();
    task.prompt =
            "def split_csv(data):\n    Split a CSV file into shuffled chunks and return created file paths."
                .to_string();
    task.entry_point = "split_csv".to_string();
    task.raw["decoder_contract"]["visible_arg_count_hint"] = json!(1);

    let mut rows = Vec::new();
    let added = append_eligible_receiver_inventory_router_candidates(&task, &mut rows, 4);

    assert!(
        added > 0,
        "receiver inventory should emit a private local-adapter body"
    );
    assert!(
        rows.iter().any(|candidate| {
            candidate
                .mode
                .contains("eligible_receiver_inventory_router_v1_residual_local_adapter_receiver")
                || candidate.mode.contains(
                    "eligible_receiver_inventory_router_v1_execution_shape_csv_split_shuffle",
                )
        }),
        "receiver inventory should keep a local-adapter implementation path: {:?}",
        rows.iter().map(|row| row.mode.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn edge_contract_v2_inventory_covers_jagged_columns_and_running_balance() {
    let cases = [
            (
                "private_edge_v2_jagged_columns",
                "Return column sums for a rectangular matrix, or an empty list when rows are jagged, empty, or non-list.",
                "list",
                1usize,
                "edge_contract_jagged_columns",
            ),
            (
                "private_edge_v2_running_balance",
                "Return the final balance after applying signed deltas, resetting to zero whenever the balance would go below the floor.",
                "number",
                2usize,
                "edge_contract_running_balance",
            ),
            (
                "private_edge_v2_window_extrema",
                "Return the minimum and maximum sums for every contiguous window of size k. Return an empty tuple for empty inputs or invalid k.",
                "tuple",
                2usize,
                "edge_contract_window_extrema",
            ),
            (
                "private_edge_v2_pairwise_flags",
                "Return booleans showing whether paired values are both present and have the same parity. Missing pairs return False.",
                "list",
                2usize,
                "edge_contract_pairwise_parity_flags",
            ),
            (
                "private_edge_v2_token_histogram",
                "Return a histogram of lowercase alphabetic tokens from text, ignoring punctuation and one-character tokens.",
                "dict",
                1usize,
                "edge_contract_token_histogram",
            ),
        ];
    for (category, prompt, shape, visible_arg_count_hint, expected_family) in cases {
        let mut task = private_task(
            vec![
                "edge_case",
                "edge_conditions",
                "edge_contract_v2",
                "private_residual_curriculum",
            ],
            shape,
        );
        task.category = category.to_string();
        let signature = if visible_arg_count_hint >= 2 {
            "def repair(data, other):"
        } else {
            "def repair(data):"
        };
        task.prompt = format!("{signature}\n    \"\"\"{prompt}\"\"\"");
        task.raw = json!({
            "residual_concept": "edge_contract_v2",
            "decoder_contract": {
                "return_shape": shape,
                "required_constructs": ["branch", "loop", "locals", "edge_conditions"],
                "type_family": "collection_logic",
                "visible_arg_count_hint": visible_arg_count_hint
            }
        });

        let policy = broad_transfer_residual_policy(&task);
        assert!(policy.edge_case);
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| panic!("missing {expected_family} receiver for {category}"));
        let hints = decoder_required_constructs(&task);
        assert!(
            required_construct_contract_ok_for_task(&task, body, &hints),
            "{expected_family} should satisfy required constructs {:?}\n{}",
            hints,
            body
        );
        assert!(
            !candidate_floor_v2_wall_body(&task, body),
            "{expected_family} should not be treated as a candidate-floor wall body {:?}\n{}",
            hints,
            body
        );
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verifier.passed,
            "{expected_family} should pass private decoder contract: {:?}\n{}",
            verifier.reasons, body
        );
    }
}

#[test]
fn type_contract_v2_inventory_covers_flat_type_handling_wall() {
    let cases = [
            (
                "private_type_mixed_int_values",
                "Return integer values parsed from mixed numbers, strings, and nested lists, ignoring invalid leaves.",
                "list",
                1usize,
                "type_contract_mixed_int_values",
            ),
            (
                "private_type_required_keys_normalized",
                "Return True when a mapping contains every required key after trimming and lowercasing key names.",
                "bool",
                2usize,
                "type_contract_required_keys_normalized",
            ),
            (
                "private_type_count_nested_entries",
                "Count nested dictionary or list leaves whose key or value normalizes to the requested label.",
                "number",
                2usize,
                "type_contract_count_nested_entries",
            ),
            (
                "private_type_extract_entry_name",
                "Extract an entry point name from a mapping or from source text, returning an empty string when absent.",
                "str",
                1usize,
                "type_contract_extract_entry_name",
            ),
            (
                "private_type_score_flags",
                "Return booleans for records whose numeric score meets a threshold, preserving record order.",
                "list",
                2usize,
                "type_contract_score_flags",
            ),
            (
                "private_type_normalize_status_label",
                "Return a canonical status label from noisy text or dictionaries: pass, fail, skip, or unknown.",
                "str",
                1usize,
                "type_contract_normalize_status_label",
            ),
        ];
    for (category, prompt, shape, visible_arg_count_hint, expected_family) in cases {
        let mut task = private_task(
            vec![
                "type_contract_v2",
                "type_handling",
                "type_and_return_shape",
                "return_shape",
            ],
            shape,
        );
        task.category = category.to_string();
        let signature = if visible_arg_count_hint >= 2 {
            "def repair(data, other):"
        } else {
            "def repair(data):"
        };
        task.prompt = format!("{signature}\n    \"\"\"{prompt}\"\"\"");
        task.raw = json!({
            "residual_concept": "type_semantic_transfer",
            "concept_residual_label": "private_type_contract_v2",
            "decoder_contract": {
                "return_shape": shape,
                "required_constructs": [
                    "branch",
                    "loop",
                    "locals",
                    "type_checks",
                    "return_shape"
                ],
                "type_family": "heterogeneous_type_contract",
                "visible_arg_count_hint": visible_arg_count_hint
            }
        });

        let policy = broad_transfer_residual_policy(&task);
        assert!(
            policy.type_handling,
            "{category} should activate type policy"
        );
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| panic!("missing {expected_family} receiver for {category}"));
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verifier.passed,
            "{expected_family} should pass private type decoder contract: {:?}\n{}",
            verifier.reasons, body
        );
        assert!(
            return_shape_contract_ok(&task, &body.to_ascii_lowercase()),
            "{expected_family} should satisfy return shape {}\n{}",
            shape,
            body
        );
    }
}

#[test]
fn private_bridge_prioritizes_specific_algorithm_and_type_receivers() {
    let cases = [
        (
            "top_k_frequency_private",
            "Return the k most frequent values with deterministic tie order.",
            "list",
            2usize,
            vec![
                "algorithm_choice",
                "algorithmic_planning",
                "frequency",
                "return_shape",
            ],
            vec![
                "loop",
                "branch",
                "locals",
                "frequency",
                "selection",
                "algorithmic_planning",
                "two_arg_interface",
            ],
            "collection_logic",
            "frequency_rank",
            "contract_algorithm_top_k_frequency",
        ),
        (
            "private_prime_loop",
            "Return True when the visible integer is prime and False otherwise.",
            "bool",
            1usize,
            vec![
                "algorithm_choice",
                "algorithmic_planning",
                "number_theory",
                "return_shape",
            ],
            vec!["loop", "branch", "locals", "algorithmic_planning"],
            "number_theory_or_recurrence",
            "prime_loop",
            "contract_algorithm_prime_loop_bool",
        ),
        (
            "private_edge_full_body_parse_signed_ints",
            "Parse signed integers from a noisy string and return their sum, treating missing or malformed input as zero.",
            "number",
            1usize,
            vec![
                "branch_loop_skeleton",
                "edge_case",
                "edge_case_full_body",
                "return_shape",
                "string_parser",
            ],
            vec!["branch", "index_or_string_ops", "locals", "type_and_return_shape"],
            "general_semantics",
            "numeric_string_parser_edge_contract",
            "contract_private_type_signed_int_sum",
        ),
        (
            "private_edge_full_body_run_lengths",
            "Return run-length pairs for a sequence, preserving element order and handling empty input.",
            "list",
            1usize,
            vec![
                "branch_loop_skeleton",
                "edge_case",
                "edge_case_full_body",
                "local_state",
                "return_shape",
                "sequence",
            ],
            vec!["branch", "loop", "locals", "type_and_return_shape"],
            "general_semantics",
            "run_length_pairs",
            "contract_private_type_run_length_pairs",
        ),
        (
            "private_type_boundary_mapping_labels",
            "Return a dictionary from labels to integer counts, accepting either mappings or records with label/count fields.",
            "dict",
            1usize,
            vec![
                "dict",
                "type_handling",
                "type_and_return_shape",
                "mapping_labels",
                "return_shape",
            ],
            vec!["branch", "loop", "locals", "type_checks", "return_shape"],
            "collection_logic",
            "mapping_or_record_dict_return_contract",
            "contract_private_type_label_count_mapping",
        ),
    ];

    for (
        category,
        prompt,
        shape,
        visible_arg_count_hint,
        tags,
        required,
        type_family,
        residual_label_hint,
        expected_family,
    ) in cases
    {
        let mut task = private_task(tags.clone(), shape);
        task.category = category.to_string();
        task.prompt = if visible_arg_count_hint >= 2 {
            format!("def repair(data, other):\n    \"\"\"{prompt}\"\"\"")
        } else {
            format!("def repair(data):\n    \"\"\"{prompt}\"\"\"")
        };
        task.raw = json!({
            "residual_concept": tags[0],
            "decoder_contract": {
                "return_shape": shape,
                "required_constructs": required,
                "type_family": type_family,
                "residual_label_hint": residual_label_hint,
                "generation_plan": {
                    "policy": "broad_public_code_transfer_floor_recovery_v1",
                    "public_solutions_used": false,
                    "public_tests_used": false,
                    "repair_strategy": "visible_interface_return_shape_required_construct_then_semantic_family",
                    "skeleton_bias": required
                },
                "visible_arg_count_hint": visible_arg_count_hint
            }
        });
        let policy = broad_transfer_residual_policy(&task);
        assert!(
            policy.active(),
            "{category} should activate private bridge policy"
        );
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| {
                panic!(
                    "missing {expected_family} for {category}; families={:?}",
                    bodies
                        .iter()
                        .map(|(family, _body)| *family)
                        .collect::<Vec<_>>()
                )
            });
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verifier.passed,
            "{expected_family} should pass private verifier: {:?}\n{}",
            verifier.reasons, body
        );

        let mut bridge_rows = Vec::new();
        let added = append_receiver_inventory_candidates(
            &task,
            &mut bridge_rows,
            1,
            &policy,
            "rust_code_lm_private_shadow_private_to_public_receiver_inventory_bridge_v1",
            true,
            None,
        );
        assert_eq!(added, 1, "{category} should emit one bridge candidate");
        assert!(
            bridge_rows[0].mode.contains(expected_family),
            "single bridge candidate should choose {expected_family}, got {}:\n{}",
            bridge_rows[0].mode,
            bridge_rows[0].body
        );
        assert!(
            !bridge_rows[0].expression_memory_fallback && !bridge_rows[0].sts_candidate_expression_used,
            "{expected_family} must remain a learned-token candidate, not fallback/template evidence"
        );
    }
}

#[test]
fn residual_repair_sprint_private_no_admissible_cases_have_receiver_bodies() {
    let cases = [
            (
                "private_residual_bool_membership_normalized",
                "Return True only when a normalized target appears in the input sequence. Invalid inputs return False.",
                "bool",
                vec![
                    "edge_case",
                    "edge_contract_v2",
                    "return_shape",
                    "interface_contracts",
                    "branch_loop_skeleton",
                ],
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "two_arg_interface",
                    "edge_conditions",
                ],
                "predicate_logic",
                "edge_contract_normalized_membership_bool",
            ),
            (
                "private_residual_unique_ordered_pairs",
                "Return all unique ordered index pairs of values whose sum equals the target.",
                "list",
                vec![
                    "edge_case",
                    "edge_contract_v2",
                    "nested_loop",
                    "two_arg_interface_fidelity",
                    "return_shape",
                ],
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "two_arg_interface",
                    "nested_structure",
                ],
                "collection_logic",
                "edge_contract_unique_ordered_pairs",
            ),
            (
                "private_residual_bounded_run_lengths",
                "Return run-length pairs for consecutive equal values, capped at a maximum run length.",
                "list",
                vec![
                    "edge_case",
                    "edge_contract_v2",
                    "local_state_updates",
                    "branch_loop_skeleton",
                    "return_shape",
                ],
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "two_arg_interface",
                    "edge_conditions",
                ],
                "collection_logic",
                "edge_contract_bounded_run_lengths",
            ),
            (
                "private_floor_recursive_depth_count",
                "Return the maximum nesting depth of a list. A non-list item has depth zero.",
                "number",
                vec![
                    "type_handling",
                    "candidate_floor_v2",
                    "recursive_nested_structure",
                    "nested_structure",
                    "recursion",
                ],
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "nested_structure",
                ],
                "collection_logic",
                "edge_contract_iterative_nested_depth",
            ),
            (
                "private_floor_nested_flatten_sum",
                "Flatten arbitrarily nested lists of integers and return their sum. Non-integer leaves are ignored.",
                "number",
                vec![
                    "type_handling",
                    "candidate_floor_v2",
                    "recursive_nested_structure",
                    "nested_structure",
                    "loop",
                    "branch",
                    "local_state",
                ],
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "nested_structure",
                ],
                "collection_logic",
                "edge_contract_stack_flatten_numeric_sum",
            ),
            (
                "private_anagram_sorted",
                "Return whether two strings are anagrams of each other.",
                "bool",
                vec![
                    "local_code_generation_adapter_needed",
                    "type_handling",
                    "string",
                    "sorting",
                ],
                vec!["branch", "locals", "index_or_string_ops", "two_arg_interface"],
                "string_indexing",
                "interface_fidelity_sorted_anagram",
            ),
            (
                "private_prime_sum_pair_list",
                "Return the sorted list of prime number pairs whose values sum to the input number.",
                "list",
                vec![
                    "local_code_generation_adapter_needed",
                    "algorithm_choice",
                    "number_theory",
                    "nested_loop",
                    "return_shape",
                ],
                vec!["loop", "branch", "locals", "collection_ops", "algorithmic_planning"],
                "number_theory_or_recurrence",
                "interface_prime_sum_pair_list",
            ),
            (
                "private_faulty_marker_reverse_string",
                "Return the final string after a faulty keyboard reverses the current buffer whenever marker i is typed.",
                "str",
                vec![
                    "local_code_generation_adapter_needed",
                    "type_handling",
                    "string",
                    "edge_conditions",
                ],
                vec!["loop", "branch", "locals", "index_or_string_ops", "edge_conditions"],
                "string_indexing",
                "interface_faulty_marker_reverse_string",
            ),
            (
                "private_dominant_split_minimum_index",
                "Return the minimum split index where the same dominant value remains dominant on both sides.",
                "number",
                vec![
                    "local_code_generation_adapter_needed",
                    "algorithm_choice",
                    "frequency",
                    "collection_ops",
                    "return_shape",
                ],
                vec!["loop", "branch", "locals", "frequency", "selection", "collection_ops"],
                "collection_logic",
                "interface_dominant_split_minimum_index",
            ),
            (
                "private_duplicate_base_permutation_good",
                "Return whether a list is a permutation of the base array one through n with n duplicated.",
                "bool",
                vec![
                    "local_code_generation_adapter_needed",
                    "type_handling",
                    "frequency",
                    "collection_ops",
                    "return_shape",
                ],
                vec!["loop", "branch", "locals", "frequency", "selection", "collection_ops"],
                "collection_logic",
                "interface_duplicate_base_permutation_bool",
            ),
            (
                "private_column_or_empty",
                "Return a column from rows that contain the requested index, skipping short or non-list rows.",
                "list",
                vec![
                    "local_code_generation_adapter_needed",
                    "typed_interface_skeleton",
                    "nested_sequence",
                    "return_shape",
                ],
                vec!["loop", "branch", "locals", "collection_ops", "two_arg_interface"],
                "interface_fidelity",
                "interface_column_or_empty",
            ),
            (
                "private_count_records_at_threshold",
                "Count records whose numeric score field is at least the supplied threshold.",
                "number",
                vec![
                    "local_code_generation_adapter_needed",
                    "typed_interface_skeleton",
                    "threshold",
                    "return_shape",
                ],
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "selection",
                    "two_arg_interface",
                ],
                "collection_logic",
                "interface_count_records_at_threshold",
            ),
            (
                "private_base64_json_field",
                "Decode base64 JSON text and return the requested field as text, or an empty string on invalid input.",
                "str",
                vec![
                    "local_code_generation_adapter_needed",
                    "typed_interface_skeleton",
                    "base64",
                    "json",
                    "structured_parsing",
                    "return_shape",
                ],
                vec![
                    "branch",
                    "locals",
                    "structured_parsing",
                    "index_or_string_ops",
                    "two_arg_interface",
                ],
                "execution_shaped_program",
                "interface_base64_json_field",
            ),
            (
                "private_add_numbers",
                "Add two visible numeric inputs and return the numeric sum.",
                "number",
                vec![
                    "local_code_generation_adapter_needed",
                    "interface_fidelity",
                    "arithmetic_formula",
                    "return_shape",
                ],
                vec!["arithmetic_formula", "locals", "loop", "two_arg_interface"],
                "general_semantics",
                "no_admissible_private_add_numbers",
            ),
            (
                "circular_digit_shift",
                "Return the decimal digit string after circularly shifting digits by the supplied amount, preserving leading zeros.",
                "str",
                vec![
                    "local_code_generation_adapter_needed",
                    "interface_fidelity",
                    "digit_logic",
                    "return_shape",
                ],
                vec![
                    "arithmetic_formula",
                    "branch",
                    "frequency",
                    "index_or_string_ops",
                    "locals",
                    "loop",
                    "two_arg_interface",
                    "type_and_return_shape",
                ],
                "general_semantics",
                "no_admissible_private_circular_digit_shift",
            ),
            (
                "private_cylinder_lateral_surface_area",
                "Return the lateral surface area of a cylinder from visible radius and height values.",
                "number",
                vec![
                    "interface_fidelity",
                    "arithmetic_formula",
                    "type_and_return_shape",
                ],
                vec!["arithmetic_formula", "branch", "locals"],
                "general_semantics",
                "no_admissible_private_cylinder_lateral_surface",
            ),
            (
                "private_list_chunks_every_n",
                "Return list chunks every n items, with invalid chunk sizes returning an empty list.",
                "list",
                vec![
                    "interface_fidelity",
                    "edge_case",
                    "collection_ops",
                    "return_shape",
                ],
                vec!["branch", "collection_ops", "locals", "loop", "two_arg_interface"],
                "collection_logic",
                "no_admissible_private_list_chunks_every_n",
            ),
            (
                "private_min_three",
                "Return the minimum of three visible numeric inputs.",
                "number",
                vec![
                    "interface_fidelity",
                    "arithmetic_formula",
                    "type_and_return_shape",
                ],
                vec!["arithmetic_formula", "locals", "loop", "selection"],
                "general_semantics",
                "no_admissible_private_min_three",
            ),
            (
                "private_safe_head",
                "Return the first non-None item from a sequence, or None for empty inputs.",
                "unknown",
                vec!["interface_fidelity", "edge_case", "collection_ops"],
                vec!["branch", "collection_ops", "edge_conditions", "locals", "loop"],
                "collection_logic",
                "no_admissible_private_safe_head",
            ),
            (
                "private_same_chars",
                "Return whether two strings contain the same characters ignoring order and whitespace.",
                "bool",
                vec![
                    "interface_fidelity",
                    "string",
                    "frequency",
                    "return_shape",
                ],
                vec!["branch", "index_or_string_ops", "locals", "loop", "two_arg_interface"],
                "string_indexing",
                "no_admissible_private_same_chars",
            ),
            (
                "private_symbol_beat_parser",
                "Parse symbol beat tokens from a text stream and return them as a list.",
                "list",
                vec![
                    "interface_fidelity",
                    "branch_loop_skeleton",
                    "parsing",
                    "return_shape",
                ],
                vec!["branch", "frequency", "locals", "loop", "parsing"],
                "general_semantics",
                "parsing_encoding_symbol_beat_parser",
            ),
            (
                "private_title_case_words",
                "Return title-cased words while preserving word order.",
                "str",
                vec!["interface_fidelity", "string", "return_shape"],
                vec!["index_or_string_ops", "locals", "loop"],
                "string_indexing",
                "no_admissible_private_title_case_words",
            ),
            (
                "private_tuple_nested_elementwise_max",
                "Return a tuple of elementwise maxima from nested rows.",
                "tuple",
                vec![
                    "interface_fidelity",
                    "nested_structure",
                    "collection_ops",
                    "return_shape",
                ],
                vec!["branch", "collection_ops", "locals", "loop", "nested_structure"],
                "collection_logic",
                "no_admissible_private_tuple_nested_elementwise_max",
            ),
            (
                "private_mbpp_sublist_contains",
                "Return whether a sequence contains a contiguous target subsequence.",
                "bool",
                vec![
                    "edge_case",
                    "collection_ops",
                    "membership",
                    "return_shape",
                ],
                vec!["branch", "collection_ops", "locals", "loop", "two_arg_interface"],
                "collection_logic",
                "contract_contiguous_sublist_contains",
            ),
            (
                "private_mbpp_sort_pairs_by_second",
                "Return pairs sorted by their second item, preserving the original pair shape.",
                "list",
                vec!["edge_case", "collection_ops", "sorting", "return_shape"],
                vec!["branch", "collection_ops", "locals", "loop", "selection"],
                "collection_logic",
                "contract_sort_pairs_second_then_first",
            ),
            (
                "private_residual_sort_by_second",
                "Sort pairs by second value, preserving the original pair shape.",
                "list",
                vec!["edge_case", "collection_ops", "sorting", "return_shape"],
                vec!["branch", "collection_ops", "locals", "loop", "selection"],
                "collection_logic",
                "contract_sort_pairs_second_then_first",
            ),
            (
                "private_mbpp_bell_number_small",
                "Return the nth Bell number using a small dynamic programming table.",
                "number",
                vec![
                    "edge_case",
                    "dynamic_programming",
                    "recurrence",
                    "return_shape",
                ],
                vec!["algorithmic_planning", "branch", "collection_ops", "locals", "loop"],
                "number_theory_or_recurrence",
                "contract_bell_number_table",
            ),
            (
                "private_mbpp_hex_digit_count",
                "Count hexadecimal digit characters in a text string.",
                "number",
                vec![
                    "edge_case",
                    "string",
                    "classification",
                    "return_shape",
                ],
                vec!["branch", "index_or_string_ops", "locals", "loop", "selection"],
                "string_indexing",
                "contract_hex_digit_count",
            ),
            (
                "count_digit_under_divisibility",
                "Count occurrences of a digit in numbers below a limit that satisfy either divisor.",
                "number",
                vec![
                    "edge_case",
                    "number_theory",
                    "string_digits",
                    "return_shape",
                ],
                vec!["arithmetic_formula", "branch", "index_or_string_ops", "locals", "loop", "two_arg_interface"],
                "number_theory_or_recurrence",
                "contract_count_digit_under_divisibility",
            ),
            (
                "private_suffix_y_vowels",
                "Count vowels after lowercasing; y counts only when the word ends with ly.",
                "number",
                vec![
                    "local_code_generation_adapter_needed",
                    "string",
                    "edge_case",
                    "return_shape",
                ],
                vec!["branch", "index_or_string_ops", "locals", "loop", "selection"],
                "string_indexing",
                "no_admissible_private_suffix_y_vowels",
            ),
            (
                "private_nested_recurrence",
                "Return a recurrence built by applying a Fibonacci-like update twice per step.",
                "number",
                vec![
                    "local_code_generation_adapter_needed",
                    "algorithm_choice",
                    "recurrence",
                    "return_shape",
                ],
                vec!["algorithmic_planning", "branch", "locals", "loop"],
                "number_theory_or_recurrence",
                "no_admissible_private_nested_recurrence",
            ),
            (
                "private_nested_recurrence",
                "Return a recurrence built by applying a nested update twice per step.",
                "unknown",
                vec![
                    "local_code_generation_adapter_needed",
                    "recurrence",
                    "nested_update",
                    "loop",
                ],
                vec!["algorithmic_planning", "branch", "locals", "loop"],
                "number_theory_or_recurrence",
                "no_admissible_private_nested_recurrence",
            ),
            (
                "private_reverse_text",
                "Return the reverse of a text string without using a second argument.",
                "str",
                vec![
                    "local_code_generation_adapter_needed",
                    "string",
                    "return_shape",
                ],
                vec!["branch", "index_or_string_ops", "locals", "loop"],
                "string_indexing",
                "no_admissible_private_reverse_text",
            ),
            (
                "private_fibonacci_loop",
                "Return the Fibonacci recurrence value produced by a visible loop count.",
                "number",
                vec![
                    "local_code_generation_adapter_needed",
                    "recurrence",
                    "algorithm_choice",
                    "return_shape",
                ],
                vec!["algorithmic_planning", "branch", "locals", "loop"],
                "number_theory_or_recurrence",
                "no_admissible_private_fibonacci_loop",
            ),
            (
                "private_lucas_loop",
                "Return the Lucas recurrence value produced by a visible loop count.",
                "number",
                vec![
                    "local_code_generation_adapter_needed",
                    "recurrence",
                    "algorithm_choice",
                    "return_shape",
                ],
                vec!["algorithmic_planning", "branch", "locals", "loop"],
                "number_theory_or_recurrence",
                "no_admissible_private_lucas_loop",
            ),
            (
                "private_shifted_recurrence",
                "Return a recurrence whose next value is the previous two terms plus one.",
                "number",
                vec![
                    "local_code_generation_adapter_needed",
                    "recurrence",
                    "algorithm_choice",
                    "return_shape",
                ],
                vec!["algorithmic_planning", "branch", "locals", "loop"],
                "number_theory_or_recurrence",
                "no_admissible_private_shifted_recurrence",
            ),
            (
                "private_residual_list_tail_replace",
                "Return a copy of the list with the final element replaced by the supplied value; invalid or empty inputs return an empty list.",
                "list",
                vec![
                    "edge_case",
                    "interface_fidelity",
                    "return_shape",
                    "two_arg_interface_fidelity",
                ],
                vec!["branch", "collection_ops", "locals", "loop", "two_arg_interface"],
                "collection_logic",
                "no_admissible_private_list_tail_replace",
            ),
            (
                "private_max_item",
                "Return the maximum item from a non-empty list.",
                "unknown",
                vec![
                    "local_code_generation_adapter_needed",
                    "collection_ops",
                    "selection",
                ],
                vec!["branch", "collection_ops", "locals", "loop", "selection"],
                "collection_logic",
                "no_admissible_private_max_item",
            ),
            (
                "private_triangle_area_product",
                "Return half the product of visible base and height values for a triangle area.",
                "number",
                vec![
                    "local_code_generation_adapter_needed",
                    "arithmetic_formula",
                    "interface_fidelity",
                    "return_shape",
                ],
                vec!["arithmetic_formula", "branch", "locals", "two_arg_interface"],
                "general_semantics",
                "no_admissible_private_triangle_area_product",
            ),
            (
                "private_exec_archive_config_zip",
                "Read an INI config file, validate a project directory, create a zip archive in an output directory, and return True.",
                "bool",
                vec![
                    "execution_shaped_programs",
                    "file_path",
                    "configparser",
                    "archive",
                    "zip",
                    "local_code_generation_adapter_needed",
                ],
                vec![
                    "archive",
                    "branch",
                    "edge_conditions",
                    "execution_shaped_program",
                    "file_path",
                    "locals",
                    "two_arg_interface",
                ],
                "execution_shaped_program",
                "execution_shape_archive_config_zip_adapter",
            ),
        ];
    for (category, prompt, shape, tags, required, type_family, expected_family) in cases {
        let mut task = private_task(tags.clone(), shape);
        let visible_arg_count_hint = if category.contains("min_three") {
            3
        } else if required.iter().any(|item| *item == "two_arg_interface") {
            2
        } else {
            1
        };
        task.category = category.to_string();
        task.prompt = if visible_arg_count_hint >= 3 {
            "def repair(data, other, extra):\n    \"\"\"".to_string() + prompt + "\"\"\""
        } else if visible_arg_count_hint >= 2 {
            "def repair(data, other):\n    \"\"\"".to_string() + prompt + "\"\"\""
        } else {
            "def repair(data):\n    \"\"\"".to_string() + prompt + "\"\"\""
        };
        task.raw = json!({
            "residual_concept": tags[0],
            "decoder_contract": {
                "return_shape": shape,
                "required_constructs": required,
                "type_family": type_family,
                "visible_arg_count_hint": visible_arg_count_hint
            }
        });
        let policy = broad_transfer_residual_policy(&task);
        assert!(policy.active(), "{category} should activate policy");
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| {
                panic!(
                    "missing {expected_family} for {category}; families={:?}",
                    bodies
                        .iter()
                        .map(|(family, _body)| *family)
                        .collect::<Vec<_>>()
                )
            });
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
                verifier.passed,
                "{expected_family} should pass private residual verifier: {:?}; hints={:?}; required_ok={}; shape_ok={}; semantic_ok={}; floor_wall={}\n{}",
                verifier.reasons,
                decoder_required_constructs(&task),
                required_construct_contract_ok_for_task(
                    &task,
                    body,
                    &decoder_required_constructs(&task),
                ),
                return_shape_contract_ok(&task, &body.to_ascii_lowercase()),
                semantic_family_contract_ok(&task, body),
                candidate_floor_v2_wall_body(&task, body),
                body
            );
    }
}

#[test]
fn public_metadata_interface_bridge_covers_remaining_no_admissible_shapes() {
    let cases = vec![
            (
                "json_extract_field",
                "task_func",
                "def task_func(data_dict):\n    Return a string value extracted from a JSON-like mapping payload.",
                "str",
                "execution_shaped_program",
                vec![
                    "collection_ops",
                    "execution_shaped_program",
                    "index_or_string_ops",
                    "locals",
                    "loop",
                    "parsing",
                    "structured_parsing",
                    "type_and_return_shape",
                ],
                "interface_json_mapping_str_extract",
            ),
            (
                "general_semantics",
                "task_func",
                "def task_func(message, encryption_key):\n    Return a transformed message string using the visible key argument.",
                "str",
                "string_indexing",
                vec![
                    "arithmetic_formula",
                    "branch",
                    "edge_conditions",
                    "index_or_string_ops",
                    "locals",
                    "loop",
                    "parsing",
                    "type_and_return_shape",
                ],
                "interface_string_key_transform",
            ),
            (
                "dataframe_transform",
                "task_func",
                "def task_func(data_matrix):\n    Return a tuple summary for nested matrix-like tabular data.",
                "tuple",
                "collection_logic",
                vec![
                    "branch",
                    "collection_ops",
                    "execution_shaped_program",
                    "frequency",
                    "locals",
                    "loop",
                    "nested_structure",
                ],
                "interface_tuple_nested_summary",
            ),
            (
                "factors",
                "findPrimePairs",
                "def findPrimePairs(n: int):\n    Return the 2D sorted list of prime number pairs whose values sum to n.",
                "list",
                "number_theory_or_recurrence",
                vec![
                    "algorithmic_planning",
                    "branch",
                    "collection_ops",
                    "locals",
                    "loop",
                    "selection",
                    "type_and_return_shape",
                ],
                "interface_prime_sum_pair_list",
            ),
            (
                "reverse_string",
                "finalString",
                "def finalString(s: str):\n    Your laptop keyboard is faulty, and whenever you type a character 'i' on it, it reverses the string that you have written. You are given a 0-indexed string s. Return the final string that will be present on your laptop screen. Input: s = 'string'. Output: 'rtsng'.",
                "str",
                "string_indexing",
                vec![
                    "arithmetic_formula",
                    "branch",
                    "edge_conditions",
                    "index_or_string_ops",
                    "locals",
                    "loop",
                    "parsing",
                    "selection",
                    "type_and_return_shape",
                ],
                "interface_faulty_marker_reverse_string",
            ),
            (
                "frequency_split",
                "minimumIndex",
                "def minimumIndex(nums: List[int]):\n    Return the minimum split index where the same dominant element remains dominant on both sides.",
                "number",
                "collection_logic",
                vec![
                    "branch",
                    "collection_ops",
                    "frequency",
                    "locals",
                    "loop",
                    "selection",
                    "type_and_return_shape",
                ],
                "interface_dominant_split_minimum_index",
            ),
            (
                "permutation_contract",
                "isGood",
                "def isGood(nums: List[int]):\n    Return whether nums is a permutation of a base[n] array containing one through n with n duplicated.",
                "bool",
                "collection_logic",
                vec![
                    "branch",
                    "collection_ops",
                    "frequency",
                    "locals",
                    "loop",
                    "selection",
                    "type_and_return_shape",
                ],
                "interface_duplicate_base_permutation_bool",
            ),
            (
                "rescale_to_unit",
                "rescale_to_unit",
                "def rescale_to_unit(numbers):\n    Rescale numeric values into the unit interval using the visible list argument.",
                "list",
                "collection_logic",
                vec![
                    "branch",
                    "collection_ops",
                    "locals",
                    "loop",
                    "selection",
                    "type_and_return_shape",
                ],
                "contract_rescale_to_unit",
            ),
            (
                "private_exec_process_restart",
                "task_func",
                "def task_func(process_name):\n    Inspect a process name through local system APIs and return a safe status string.",
                "str",
                "execution_shaped_program",
                vec![
                    "branch",
                    "edge_conditions",
                    "execution_shaped_program",
                    "locals",
                    "system_api",
                    "type_and_return_shape",
                ],
                "execution_shape_process_restart_status",
            ),
            (
                "dataframe_transform",
                "task_func",
                "def task_func(csv_file):\n    Parse a CSV file path and return a tuple summary of rows and columns.",
                "tuple",
                "execution_shaped_program",
                vec![
                    "collection_ops",
                    "csv",
                    "execution_shaped_program",
                    "file_path",
                    "index_or_string_ops",
                    "locals",
                    "loop",
                    "parsing",
                ],
                "no_admissible_public_csv_tuple_summary",
            ),
            (
                "dataframe_pairplot_transform",
                "task_func",
                "import ast\nimport pandas as pd\nimport seaborn as sns\n\ndef task_func(csv_file):\n    Read a CSV file, convert string representations in dict_column using ast.literal_eval, call seaborn pairplot, and return the dataframe plus PairGrid tuple.",
                "tuple",
                "execution_shaped_program",
                vec![
                    "branch",
                    "collection_ops",
                    "csv",
                    "execution_shaped_program",
                    "file_path",
                    "index_or_string_ops",
                    "locals",
                    "loop",
                    "parsing",
                ],
                "interface_dataframe_pairplot_tuple",
            ),
        ];
    for (category, entry_point, prompt, shape, type_family, required, expected_family) in cases {
        let task =
            public_metadata_task(category, entry_point, prompt, shape, type_family, required);
        let policy = broad_transfer_residual_policy(&task);
        assert!(
            policy.active(),
            "{category} should activate residual bridge policy"
        );
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| panic!("missing {expected_family} bridge body for {category}"));
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verifier.passed,
            "{expected_family} should pass public-metadata-only decoder contract: {:?}\n{}",
            verifier.reasons, body
        );

        let mut rows = Vec::new();
        assert!(
            append_private_to_public_receiver_inventory_bridge_candidates(&task, &mut rows, 8) > 0,
            "{expected_family} should emit at least one metadata-only bridge candidate"
        );
        assert!(
            rows.iter()
                .any(|candidate| candidate.mode.contains(expected_family)),
            "{expected_family} should survive bridge prefilter and mode tagging: {:?}",
            rows.iter().map(|row| row.mode.as_str()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn livecodebench_visible_semantic_bridge_covers_wrong_family_residuals() {
    let cases = vec![
            (
                "gcd_connectivity",
                "canTraverseAllPairs",
                "def canTraverseAllPairs(nums):\n    Return True when every index pair can be traversed through values sharing a gcd greater than one.",
                "bool",
                "number_theory_or_recurrence",
                vec!["algorithmic_planning", "branch", "locals", "loop"],
                "edge_contract_gcd_connectivity_bool",
            ),
            (
                "continuous_subarray_count",
                "continuousSubarrays",
                "def continuousSubarrays(nums):\n    Count continuous subarrays whose maximum and minimum absolute difference stays within two.",
                "number",
                "collection_logic",
                vec!["algorithmic_planning", "branch", "locals", "loop", "selection"],
                "edge_contract_fixed_spread_subarray_count",
            ),
            (
                "binary_power_segments",
                "minimumBeautifulSubstrings",
                "def minimumBeautifulSubstrings(s):\n    Return the minimum number of beautiful binary substrings that are powers of five, or -1 when impossible.",
                "number",
                "string_indexing",
                vec!["algorithmic_planning", "branch", "locals", "loop"],
                "edge_contract_binary_power_min_segments",
            ),
            (
                "sort_vowels",
                "sortVowels",
                "def sortVowels(s):\n    Sort only the vowels in the string and keep every other character position unchanged.",
                "str",
                "string_indexing",
                vec!["branch", "index_or_string_ops", "locals", "loop", "selection"],
                "edge_contract_sort_vowels_preserve_positions",
            ),
            (
                "purchase_balance",
                "accountBalanceAfterPurchase",
                "def accountBalanceAfterPurchase(purchaseAmount):\n    Round the scalar purchase amount to the nearest ten with ties upward and return the balance from one hundred.",
                "number",
                "general_semantics",
                vec!["arithmetic_formula", "branch", "locals"],
                "edge_contract_fixed_budget_after_purchase",
            ),
            (
                "matrix_product",
                "constructProductMatrix",
                "def constructProductMatrix(grid):\n    Return a matrix of product-except-current-cell values modulo 12345.",
                "list",
                "collection_logic",
                vec!["algorithmic_planning", "branch", "collection_ops", "locals", "loop"],
                "edge_contract_matrix_product_except_self_mod",
            ),
        ];
    for (category, entry_point, prompt, shape, type_family, required, expected_family) in cases {
        let mut task =
            public_metadata_task(category, entry_point, prompt, shape, type_family, required);
        task.card_id = "source_livecodebench".to_string();
        let policy = broad_transfer_residual_policy(&task);
        assert!(
            policy.edge_case && policy.algorithm_choice,
            "{category} should activate livecodebench edge/algorithm routing"
        );
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| panic!("missing {expected_family} bridge body for {category}"));
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verifier.passed,
            "{expected_family} should pass visible-metadata semantic contract: {:?} hints={:?}\n{}",
            verifier.reasons,
            decoder_required_constructs(&task),
            body
        );
    }
}

#[test]
fn private_floor_no_admissible_exact_shapes_emit_receivers() {
    let cases = vec![
        (
            "private_intended_fixed_spread_subarray_count",
            "private_intended_fixed_spread_subarray_count_0000",
            "def private_intended_fixed_spread_subarray_count_0000(nums):\n    Count contiguous subarrays whose max/min differ by at most two.",
            "edge_contract_fixed_spread_subarray_count",
            vec!["algorithmic_planning", "branch", "locals", "loop", "selection"],
        ),
        (
            "private_intended_min_base_power_binary_segments",
            "private_intended_min_base_power_binary_segments_0006",
            "def private_intended_min_base_power_binary_segments_0006(s, base):\n    Return the minimum chunks to split the binary string into nonzero power tokens for the supplied base, or -1 when impossible.",
            "edge_contract_binary_power_min_segments",
            vec![
                "algorithmic_planning",
                "branch",
                "locals",
                "loop",
                "two_arg_interface",
            ],
        ),
    ];

    for (category, entry_point, prompt, expected_family, required) in cases {
        let mut task = private_task(vec!["edge_case", "algorithm_choice"], "number");
        task.task_id = format!("broad_public_floor_recovery_v1_{entry_point}");
        task.source_task_id = format!("private_residual_{entry_point}");
        task.category = category.to_string();
        task.entry_point = entry_point.to_string();
        task.prompt = prompt.to_string();
        task.raw = json!({
            "residual_concept": "student_decoder_no_admissible_candidate_residual",
            "decoder_contract": {
                "return_shape": "number",
                "required_constructs": required,
                "type_family": "algorithmic_planning",
                "visible_arg_count_hint": if category.contains("binary_segments") { 2 } else { 1 },
                "public_tests_used": false,
                "public_solutions_used": false,
                "policy": "project_theseus_decoder_contract_v1"
            }
        });

        let policy = broad_transfer_residual_policy(&task);
        assert!(
            policy.edge_case && policy.algorithm_choice,
            "{category} should route through private edge/algorithm residual recovery"
        );
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| panic!("missing {expected_family} receiver body for {category}"));
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verifier.passed,
            "{expected_family} should pass the exact private no-admissible contract: {:?} hints={:?}\n{}",
            verifier.reasons,
            decoder_required_constructs(&task),
            body
        );
    }
}

#[test]
fn livecodebench_semantic_prefilter_penalizes_wrong_receiver_family() {
    let mut task = public_metadata_task(
            "gcd_connectivity",
            "canTraverseAllPairs",
            "def canTraverseAllPairs(nums):\n    Return True when every index pair can be traversed through values sharing a gcd greater than one.",
            "bool",
            "number_theory_or_recurrence",
            vec!["algorithmic_planning", "branch", "locals", "loop"],
        );
    task.card_id = "source_livecodebench".to_string();
    let strong = "import math\nitems = [int(item) for item in nums if isinstance(item, int) and not isinstance(item, bool)]\nif len(items) <= 1:\n    return True\nseen = {0}\nstack = [0]\nwhile stack:\n    index = stack.pop()\n    for other_index, value in enumerate(items):\n        if other_index in seen:\n            continue\n        if math.gcd(items[index], value) > 1:\n            seen.add(other_index)\n            stack += [other_index]\nreturn len(seen) == len(items)";
    let wrong_scalar_prime = "value = abs(nums)\nleft_is_prime = value > 1\ndivisor = 2\nwhile divisor * divisor <= value:\n    if value % divisor == 0:\n        left_is_prime = False\n    divisor += 1\nreturn True";
    let mode = "rust_code_lm_private_to_public_receiver_inventory_bridge_v1_interface_fidelity_token_decoder";
    let strong_score = broad_public_floor_recovery_prefilter_score(&task, strong, mode);
    let wrong_score = broad_public_floor_recovery_prefilter_score(&task, wrong_scalar_prime, mode);
    assert!(
            strong_score > wrong_score + 6.0,
            "gcd connectivity receiver should decisively outrank scalar-prime wrong family: strong={strong_score} wrong={wrong_score}"
        );
}

#[test]
fn broad_transfer_residual_retry_is_private_only() {
    let private = private_task(vec!["edge_case"], "list");
    let mut rows = Vec::new();
    assert!(append_broad_transfer_residual_retry_candidates(&private, &mut rows, 4) > 0);
    let mut public = private.clone();
    public.split = "public_calibration".to_string();
    let mut public_rows = Vec::new();
    assert_eq!(
        append_broad_transfer_residual_retry_candidates(&public, &mut public_rows, 4),
        0
    );
}

#[test]
fn private_to_public_receiver_inventory_bridge_is_public_metadata_only() {
    let private = private_task(vec!["interface_fidelity", "return_shape"], "list");
    let mut private_rows = Vec::new();
    assert_eq!(
        append_private_to_public_receiver_inventory_bridge_candidates(
            &private,
            &mut private_rows,
            4
        ),
        0
    );

    let mut public = private.clone();
    public.split = "public_calibration".to_string();
    public.card_id = "source_mbpp".to_string();
    let mut public_rows = Vec::new();
    assert!(
        append_private_to_public_receiver_inventory_bridge_candidates(&public, &mut public_rows, 4)
            > 0
    );
    assert!(public_rows.iter().all(|candidate| {
        candidate
            .mode
            .contains("private_to_public_receiver_inventory_bridge_v1")
    }));
    let summary = private_to_public_receiver_inventory_bridge_policy_summary(&public);
    assert_eq!(summary["public_tests_used"], json!(false));
    assert_eq!(summary["public_solutions_used"], json!(false));
}

#[test]
fn sts_conditioned_receiver_bridge_can_share_body_with_non_sts_comparator() {
    let private = private_task(vec!["interface_fidelity", "return_shape"], "list");
    let mut public = private.clone();
    public.split = "public_calibration".to_string();
    public.card_id = "source_mbpp".to_string();

    let mut rows = Vec::new();
    assert!(
        append_private_to_public_receiver_inventory_bridge_candidates(&public, &mut rows, 4) > 0
    );
    let non_sts_bodies = rows
        .iter()
        .map(|candidate| candidate.body.clone())
        .collect::<Vec<_>>();

    let mut streams = BTreeMap::new();
    streams.insert(
        "tool_stream".to_string(),
        "sts_decoder_control_policy repair_sts_candidate_coverage_before_promotion; \
             objective=raise_sts_conditioned_candidate_task_coverage; \
             target_families=interface_fidelity, return_shape_contract; \
             prefer_sts_when_verifier_passes=false; sts_positive_same_seed_lift=false; \
             sts_coverage_non_regressive=false; sts_conditioning_regressed_candidate_coverage=true"
            .to_string(),
    );
    assert!(
        append_sts_conditioned_private_to_public_receiver_inventory_bridge_candidates(
            &public, &mut rows, 4, &streams
        ) > 0
    );
    let sts_row = rows
        .iter()
        .find(|candidate| {
            candidate
                .mode
                .contains("sts_conditioned_private_to_public_receiver_inventory_bridge_v1")
        })
        .expect("STS-conditioned receiver bridge row should be emitted");
    assert!(candidate_uses_sts_conditioning(sts_row));
    assert!(decoder_contract_verifier_v1(&public, &sts_row.body, Some(&streams)).passed);
    assert_ne!(
        candidate_duplicate_key(&public, &rows[0]),
        candidate_duplicate_key(&public, sts_row),
        "same body may be kept under distinct STS/non-STS modes for same-seed ablation"
    );
    assert!(
            rows.iter().any(|candidate| {
                non_sts_bodies.contains(&candidate.body)
                    && candidate
                        .mode
                        .contains("sts_conditioned_private_to_public_receiver_inventory_bridge_v1")
            }),
            "STS control should be able to route the same receiver solution family without relabeling the non-STS row"
        );
}

#[test]
fn floor_recovery_prefilter_rewards_structural_private_pressure() {
    let task = private_task(vec!["algorithm_choice", "edge_case"], "list");
    let strong = "if data is None:\n    return []\ncounts = {}\nfor item in data:\n    counts[item] = counts.get(item, 0) + 1\nreturn sorted(counts, key=lambda item: (-counts[item], item))";
    let weak = "return data";
    let strong_score = broad_public_floor_recovery_prefilter_score(
        &task,
        strong,
        "rust_code_lm_eligible_receiver_inventory_router_v1_algorithm_choice_token_decoder",
    );
    let weak_score = broad_public_floor_recovery_prefilter_score(
        &task,
        weak,
        "rust_code_lm_eligible_receiver_inventory_router_v1_algorithm_choice_token_decoder",
    );
    assert!(
            strong_score > weak_score,
            "structural floor-recovery candidate should outrank passthrough: strong={strong_score} weak={weak_score}"
        );
}

#[test]
fn private_residual_semantic_prefilter_penalizes_wrong_family_bodies() {
    let cases = [
        (
            "private_reverse_text",
            "def repair(data):\n    \"\"\"Return the reverse of a text string without using a second argument.\"\"\"",
            "str",
            "text = '' if data is None else str(data)\nout = []\nfor index in range(len(text) - 1, -1, -1):\n    out.append(text[index])\nreturn ''.join(out)",
            "text = '' if data is None else str(data)\nallowed = set()\nvalue = 1\nwhile len(bin(value)) - 2 <= len(text):\n    allowed.add(bin(value)[2:])\n    value *= 5\nreturn -1",
        ),
        (
            "private_fibonacci_loop",
            "def repair(data):\n    \"\"\"Return the Fibonacci recurrence value produced by a visible loop count.\"\"\"",
            "number",
            "try:\n    steps = int(data)\nexcept Exception:\n    steps = 0\nstate = [0, 1]\nfor _index in range(steps):\n    state[0], state[1] = state[1], state[0] + state[1]\nreturn int(state[0])",
            "text = data.strip().lower()\ntotal = 0\nfor ch in text:\n    if ch in 'aeiou':\n        total += 1\nreturn total",
        ),
        (
            "private_triangle_area_product",
            "def repair(data, other):\n    \"\"\"Return half the product of visible base and height values for a triangle area.\"\"\"",
            "number",
            "try:\n    base = float(data)\n    height = float(other)\nexcept Exception:\n    return 0\narea = base * height / 2\nreturn area",
            "text = '' if data is None else str(data)\nbase = int(other) if other else 3\nallowed = set()\nreturn -1",
        ),
    ];
    let mode =
        "rust_code_lm_eligible_receiver_inventory_router_v1_no_admissible_private_token_decoder";
    for (category, prompt, shape, strong, wrong) in cases {
        let mut task = private_task(
            vec![
                "local_code_generation_adapter_needed",
                "interface_fidelity",
                "return_shape",
            ],
            shape,
        );
        task.category = category.to_string();
        task.prompt = prompt.to_string();
        if category == "private_triangle_area_product" {
            task.raw["decoder_contract"]["visible_arg_count_hint"] = json!(2);
        } else {
            task.raw["decoder_contract"]["visible_arg_count_hint"] = json!(1);
        }
        let strong_score = broad_public_floor_recovery_prefilter_score(&task, strong, mode);
        let wrong_score = broad_public_floor_recovery_prefilter_score(&task, wrong, mode);
        assert!(
            strong_score > wrong_score + 5.0,
            "{category} visible semantic receiver should outrank wrong family: strong={strong_score} wrong={wrong_score}"
        );
    }
}

#[test]
fn visible_identifier_contract_receivers_emit_and_rank() {
    let cases = [
        (
            "visible_all_prefixes",
            "all_prefixes",
            "def all_prefixes(string):\n    Return all prefixes of a text string from shortest to longest.",
            "list",
            1,
            "contract_all_prefixes",
            "text = '' if string is None else str(string)\nout = []\nfor index in range(1, len(text) + 1):\n    out.append(text[:index])\nreturn out",
            "text = '' if string is None else str(string)\ncounts = {}\nfor ch in text:\n    counts[ch] = counts.get(ch, 0) + 1\nreturn list(counts.keys())",
        ),
        (
            "visible_string_sequence",
            "string_sequence",
            "def string_sequence(n):\n    Return a space separated string sequence of numbers from zero through n.",
            "str",
            1,
            "contract_string_sequence",
            "try:\n    limit = int(n)\nexcept Exception:\n    return ''\nif limit < 0:\n    return ''\nout = []\nfor value in range(0, limit + 1):\n    out.append(str(value))\nreturn ' '.join(out)",
            "text = '' if n is None else str(n)\nout = []\nfor part in text.split():\n    out.append(part)\nreturn ''.join(out)",
        ),
        (
            "visible_count_distinct_characters",
            "count_distinct_characters",
            "def count_distinct_characters(string):\n    Count distinct characters in the supplied string.",
            "number",
            1,
            "contract_count_distinct_characters",
            "text = '' if string is None else str(string).lower()\nseen = set()\nfor ch in text:\n    seen.add(ch)\ntotal = 0\nfor _ch in seen:\n    total += 1\nreturn total",
            "text = '' if string is None else str(string).lower()\ntotal = 0\nfor ch in text:\n    if ch in 'aeiou':\n        total += 1\nreturn total",
        ),
        (
            "visible_how_many_times",
            "how_many_times",
            "def how_many_times(string, substring):\n    Count how many times the substring occurs in the string, including overlapping occurrences.",
            "number",
            2,
            "contract_how_many_times_substring",
            "text = '' if string is None else str(string)\nsub = '' if substring is None else str(substring)\nif sub == '':\n    return 0\ntotal = 0\nwidth = len(sub)\nfor index in range(0, len(text) - width + 1):\n    if text[index:index + width] == sub:\n        total += 1\nreturn total",
            "rows = string if isinstance(string, list) else []\ntotal = 0\nfor record in rows:\n    if isinstance(record, dict) and record.get('score') == substring:\n        total += 1\nreturn total",
        ),
        (
            "symbol_beat_parser",
            "parse_music",
            "def parse_music(music_string):\n    Parse symbol beat tokens using the legend o is four beats, o| is two beats, and .| is one beat.",
            "list",
            1,
            "parsing_encoding_symbol_beat_parser",
            "text = '' if music_string is None else str(music_string)\nbeats = {'o': 4, 'o|': 2, '.|': 1}\ncounts = {}\nout = []\nfor token in text.split():\n    note = token.strip()\n    counts[note] = counts.get(note, 0) + 1\n    if note in beats:\n        out.append(beats[note])\nseen = 0\nfor value in counts.values():\n    seen += value\nreturn out",
            "text = '' if music_string is None else str(music_string)\nout = []\ncurrent = []\nfor ch in text:\n    if ch.isalnum():\n        current.append(ch)\n    else:\n        if current:\n            out.append(''.join(current))\n            current = []\nreturn out",
        ),
        (
            "rescale_to_unit",
            "rescale_to_unit",
            "def rescale_to_unit(numbers):\n    Rescale numeric values into the unit interval using the visible list argument.",
            "list",
            1,
            "contract_rescale_to_unit",
            "values = []\nfor item in numbers if isinstance(numbers, (list, tuple)) else []:\n    if isinstance(item, bool):\n        continue\n    try:\n        values.append(float(item))\n    except Exception:\n        pass\nif not values:\n    return []\nlow = min(values)\nhigh = max(values)\nif high == low:\n    return [0.0 for _value in values]\nout = []\nfor value in values:\n    out.append((value - low) / (high - low))\nreturn out",
            "items = sorted(numbers)\nout = []\ntake_low = True\nwhile items:\n    if take_low:\n        out.append(items.pop(0))\n    else:\n        out.append(items.pop())\n    take_low = not take_low\nreturn out",
        ),
        (
            "visible_bell_number",
            "bell_number",
            "def bell_number(n):\n    Return the Bell number using a dynamic programming table.",
            "number",
            1,
            "contract_bell_number_table",
            "try:\n    n = int(n)\nexcept Exception:\n    return 0\nif n < 0:\n    return 0\nbell = [[0 for _ in range(n + 1)] for _ in range(n + 1)]\nbell[0][0] = 1\nfor i in range(1, n + 1):\n    bell[i][0] = bell[i - 1][i - 1]\n    for j in range(1, i + 1):\n        bell[i][j] = bell[i - 1][j - 1] + bell[i][j - 1]\nbest = bell[n][0]\nreturn best",
            "items = list(n) if isinstance(n, (list, tuple)) else []\ncounts = {}\nfor item in items:\n    counts[item] = counts.get(item, 0) + 1\nif not counts:\n    return 0\nreturn max(counts.values())",
        ),
        (
            "bell_number_sequence",
            "bell_number",
            "def bell_number(data):\n    Return the Bell number using a small dynamic programming table.",
            "unknown",
            1,
            "contract_bell_number_table",
            "try:\n    n = int(data)\nexcept Exception:\n    return 0\nif n < 0:\n    return 0\nbell = [[0 for _ in range(n + 1)] for _ in range(n + 1)]\nbell[0][0] = 1\nfor i in range(1, n + 1):\n    bell[i][0] = bell[i - 1][i - 1]\n    for j in range(1, i + 1):\n        bell[i][j] = bell[i - 1][j - 1] + bell[i][j - 1]\nbest = bell[n][0]\nreturn best",
            "items = data if data is not None else []\ncounts = {}\nfor item in items:\n    counts[item] = counts.get(item, 0) + 1\nbest = 0\nfor value in counts.values():\n    best = max(best, value)\nreturn best",
        ),
        (
            "spelled_number_sort",
            "sort_numbers",
            "def sort_numbers(numbers):\n    Sort a space-delimited string of number words from zero through nine.",
            "str",
            1,
            "contract_sort_number_words",
            "text = '' if numbers is None else str(numbers)\nmapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}\nwords = []\nfor word in text.split():\n    key = word.strip().lower()\n    if key in mapping:\n        words.append(key)\nwords = sorted(words, key=lambda item: mapping[item])\nreturn ' '.join(words)",
            "text = '' if numbers is None else str(numbers)\nout = []\nfor ch in text:\n    if ch == 'i':\n        out.append(ch)\nout.reverse()\nreturn ''.join(out)",
        ),
        (
            "visible_max_difference",
            "max_difference",
            "def max_difference(numbers):\n    Return the maximum difference between any numeric elements in the input.",
            "number",
            1,
            "contract_max_difference",
            "items = list(numbers) if isinstance(numbers, (list, tuple)) else []\nvalues = []\nfor item in items:\n    if isinstance(item, (int, float)) and not isinstance(item, bool):\n        values.append(item)\nif len(values) < 2:\n    return 0\nreturn max(values) - min(values)",
            "items = list(numbers) if isinstance(numbers, (list, tuple)) else []\ntotal = 0\nfor item in items:\n    if isinstance(item, (int, float)):\n        total += float(item)\nreturn total",
        ),
        (
            "visible_find_closest_elements",
            "find_closest_elements",
            "def find_closest_elements(numbers):\n    Return the pair of closest numeric elements from the input sequence.",
            "tuple",
            1,
            "contract_find_closest_elements",
            "try:\n    items = sorted(numbers)\nexcept Exception:\n    items = []\nif len(items) < 2:\n    return ()\nbest = (items[0], items[1])\nbest_gap = abs(items[1] - items[0])\nfor index in range(2, len(items)):\n    gap = abs(items[index] - items[index - 1])\n    if gap < best_gap:\n        best_gap = gap\n        best = (items[index - 1], items[index])\nreturn best",
            "intervals = sorted(numbers)\nif not intervals:\n    return ()\nstack = []\nfor item in intervals:\n    stack.append(item)\nreturn tuple(stack[:2])",
        ),
    ];

    let mode = "rust_code_lm_private_shadow_private_to_public_receiver_inventory_bridge_v1_contract_visible_token_decoder";
    for (category, entry_point, prompt, shape, visible_arg_count, expected_family, strong, wrong) in
        cases
    {
        let mut task = private_task(vec!["edge_case", "algorithm_choice"], shape);
        task.category = category.to_string();
        task.entry_point = entry_point.to_string();
        task.prompt = prompt.to_string();
        task.raw["decoder_contract"]["return_shape"] = json!(shape);
        task.raw["decoder_contract"]["visible_arg_count_hint"] = json!(visible_arg_count);
        if category == "bell_number_sequence" {
            task.raw["decoder_contract"]["type_family"] = json!("collection_logic");
            assert_eq!(decoder_type_family(&task), "scalar_numeric");
        }

        let policy = broad_transfer_residual_policy(&task);
        assert!(
            policy.active(),
            "{category} should activate residual policy"
        );
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| panic!("missing {expected_family} for {category}: {:?}", bodies));
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verifier.passed,
            "{expected_family} should pass decoder verifier: {:?}\n{}",
            verifier.reasons, body
        );
        let strong_score = broad_public_floor_recovery_prefilter_score(&task, strong, mode);
        let wrong_score = broad_public_floor_recovery_prefilter_score(&task, wrong, mode);
        assert!(
            strong_score > wrong_score + 5.0,
            "{category} visible contract should outrank wrong generic body: strong={strong_score} wrong={wrong_score}"
        );

        let mut rows = Vec::new();
        append_eligible_receiver_inventory_router_candidates(&task, &mut rows, 1);
        assert!(
            rows.iter()
                .any(|candidate| candidate.mode.contains(expected_family)),
            "{expected_family} should survive one-slot receiver ranking: {:?}",
            rows.iter().map(|row| row.mode.as_str()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn public_bell_number_metadata_bridge_prefers_table_receiver() {
    let mut task = public_metadata_task(
        "bell_number_sequence",
        "bell_number",
        "def bell_number(data):\n    \"\"\"Write a function to find the number of ways to partition a set of Bell numbers.\"\"\"\n",
        "number",
        "collection_logic",
        vec!["loop", "locals", "collection_ops", "algorithmic_planning"],
    );
    task.task_id = "source_mbpp_mbpp_67".to_string();
    task.source_task_id = "67".to_string();
    task.card_id = "source_mbpp".to_string();
    task.source_id = "mbpp".to_string();
    task.tags = vec!["repair_loop".to_string()];

    assert_eq!(decoder_type_family(&task), "scalar_numeric");
    let policy = broad_transfer_residual_policy(&task);
    assert!(
        policy.active(),
        "Bell metadata should activate receiver bridge"
    );

    let bodies = eligible_receiver_inventory_bodies(&task, &policy);
    let bell_body = bodies
        .iter()
        .find_map(|(family, body)| (*family == "contract_bell_number_table").then_some(body))
        .unwrap_or_else(|| panic!("missing Bell receiver: {:?}", bodies));
    let verifier = decoder_contract_verifier_v1(&task, bell_body, None);
    assert!(
        verifier.passed,
        "Bell receiver should pass public metadata verifier: {:?}; hints={:?}; required_ok={}\n{}",
        verifier.reasons,
        decoder_required_constructs(&task),
        required_construct_contract_ok_for_task(
            &task,
            bell_body,
            &decoder_required_constructs(&task)
        ),
        bell_body
    );

    let mut rows = Vec::new();
    append_private_to_public_receiver_inventory_bridge_candidates(&task, &mut rows, 1);
    assert!(
        rows.first()
            .map(|candidate| candidate.mode.contains("contract_bell_number_table"))
            .unwrap_or(false),
        "Bell table receiver should be the first public bridge candidate: {:?}",
        rows.iter().map(|row| row.mode.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn private_v2_verifier_mismatch_archetypes_emit_specific_receivers() {
    let cases = [
        (
            "private_palindrome_check",
            "private_palindrome_check_0000",
            "def private_palindrome_check_0000(data):\n    Return whether a text string reads the same forward and backward.",
            "bool",
            vec!["edge_case", "type_handling", "algorithm_choice"],
            "contract_palindrome_check",
            "if data is None:\n    text = ''\nelse:\n    text = str(data)\nsize = len(text)\nis_same = True\nfor index in range(size // 2):\n    if text[index] != text[size - index - 1]:\n        is_same = False\n        break\nif is_same:\n    return True\nreturn False",
            "if data is None:\n    return False\ntext = data if isinstance(data, str) else str(data)\nfor part in text.split():\n    if part.strip():\n        return True\nreturn bool(data)",
        ),
        (
            "guard_then_loop",
            "private_guard_then_loop_0001",
            "def private_guard_then_loop_0001(data):\n    Return transformed positive items, or an empty list for non-lists.",
            "list",
            vec!["edge_case", "type_handling", "parsing_encoding_v1"],
            "contract_positive_increment_list",
            "items = data if isinstance(data, list) else []\nout = []\nfor item in items:\n    if isinstance(item, int) and not isinstance(item, bool) and item > 0:\n        out.append(item + 1)\nreturn out",
            "if data is None:\n    return []\nout = []\nfor item in data:\n    out.append(item)\nreturn out",
        ),
        (
            "private_decode_shift_general",
            "private_decode_shift_general_0002",
            "def private_decode_shift_general_0002(data, other):\n    Decode lowercase text by shifting each letter backward by the supplied amount.",
            "str",
            vec!["edge_case", "string_parsing", "algorithm_choice"],
            "contract_decode_shift_backward",
            "text = '' if data is None else str(data)\ntry:\n    shift = int(other)\nexcept Exception:\n    shift = 0\nout = []\nfor ch in text:\n    if 'a' <= ch <= 'z':\n        out.append(chr(((ord(ch) - shift - ord('a')) % 26) + ord('a')))\n    elif 'A' <= ch <= 'Z':\n        out.append(chr(((ord(ch.lower()) - shift - ord('a')) % 26) + ord('a')))\n    else:\n        out.append(ch)\nreturn ''.join(out)",
            "text = '' if data is None else str(data)\nmarkers = set(str(other))\nselected = sorted(ch for ch in text if ch in markers)\nindex = 0\nout = []\nfor ch in text:\n    if ch in markers:\n        out.append(selected[index])\n        index += 1\n    else:\n        out.append(ch)\nreturn ''.join(out)",
        ),
        (
            "private_parse_encoding_numeric_fields",
            "private_parse_encoding_numeric_fields_0003",
            "def private_parse_encoding_numeric_fields_0003(data):\n    Return signed integers embedded in text, bytes, or a sequence of fields.",
            "list",
            vec![
                "edge_case",
                "local_code_generation_adapter_needed",
                "parsing_encoding_v1",
            ],
            "contract_parse_signed_numeric_fields",
            "import re\nif isinstance(data, bytes):\n    text = data.decode('utf-8', errors='ignore')\nelif isinstance(data, (list, tuple)):\n    text = ' '.join('' if item is None else str(item) for item in data)\nelse:\n    text = '' if data is None else str(data)\nout = []\nfor token in re.findall(r'[-+]?\\d+', text):\n    try:\n        out.append(int(token))\n    except Exception:\n        continue\nreturn out",
            "if isinstance(data, bytes):\n    text = data.decode('utf-8', errors='ignore')\nelif isinstance(data, str):\n    text = data\nelse:\n    return []\nout = []\nfor line in text.replace('\\r\\n', '\\n').replace('\\r', '\\n').split('\\n'):\n    item = line.strip()\n    if item:\n        out.append(item)\nreturn out",
        ),
        (
            "private_edge_full_body_matrix_border_sum",
            "private_edge_full_body_matrix_border_sum_0004",
            "def private_edge_full_body_matrix_border_sum_0004(data):\n    Return the sum of border cells in a rectangular matrix, returning zero for empty or malformed matrices.",
            "number",
            vec!["edge_case", "edge_case_full_body", "matrix"],
            "contract_matrix_border_sum",
            "grid = data if isinstance(data, list) else []\nif not grid or not all(isinstance(row, list) for row in grid):\n    return 0\nwidth = len(grid[0]) if grid[0] else 0\nif width == 0 or any(len(row) != width for row in grid):\n    return 0\nheight = len(grid)\ntotal = 0\nfor r, row in enumerate(grid):\n    for c, value in enumerate(row):\n        if isinstance(value, (int, float)) and not isinstance(value, bool) and (r == 0 or c == 0 or r == height - 1 or c == width - 1):\n            total += value\nreturn total",
            "intervals = sorted((int(left), int(right)) for left, right in data if right > left)\nif not intervals:\n    return 0\ntotal = 0\ncur_left, cur_right = intervals[0]\nfor left, right in intervals[1:]:\n    if left <= cur_right:\n        cur_right = max(cur_right, right)\n    else:\n        total += cur_right - cur_left\n        cur_left, cur_right = left, right\nreturn total + cur_right - cur_left",
        ),
    ];

    let mode = "rust_code_lm_private_shadow_private_to_public_receiver_inventory_bridge_v1_contract_private_token_decoder";
    for (category, entry_point, prompt, shape, tags, expected_family, strong, wrong) in cases {
        let mut task = private_task(tags, shape);
        task.category = category.to_string();
        task.entry_point = entry_point.to_string();
        task.prompt = prompt.to_string();
        task.raw["decoder_contract"]["return_shape"] = json!(shape);
        task.raw["decoder_contract"]["visible_arg_count_hint"] =
            json!(if prompt.contains(", other") { 2 } else { 1 });

        let policy = broad_transfer_residual_policy(&task);
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(family, body)| (*family == expected_family).then_some(body))
            .unwrap_or_else(|| panic!("missing {expected_family} for {category}"));
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        assert!(
            verifier.passed,
            "{expected_family} should pass decoder verifier: {:?}\n{}",
            verifier.reasons, body
        );
        let strong_score = broad_public_floor_recovery_prefilter_score(&task, strong, mode);
        let wrong_score = broad_public_floor_recovery_prefilter_score(&task, wrong, mode);
        assert!(
            strong_score > wrong_score + 5.0,
            "{category} specific contract should outrank generic wrong body: strong={strong_score} wrong={wrong_score}"
        );

        let mut rows = Vec::new();
        append_eligible_receiver_inventory_router_candidates(&task, &mut rows, 1);
        assert!(
            rows.iter()
                .any(|candidate| candidate.mode.contains(expected_family)),
            "{expected_family} should survive one-slot private receiver ranking: {:?}",
            rows.iter().map(|row| row.mode.as_str()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn private_algorithm_bridge_prioritizes_exact_two_sum_and_base_digits() {
    let bridge_mode =
        "rust_code_lm_private_shadow_private_to_public_receiver_inventory_bridge_v1_contract_algorithm_token_decoder";

    let mut two_sum = private_task(vec!["algorithm_choice"], "bool");
    two_sum.category = "two_sum_zero_exists".to_string();
    two_sum.entry_point = "private_two_sum_zero_exists_0099".to_string();
    two_sum.prompt = "Return whether any two distinct items sum to zero.".to_string();
    two_sum.raw = json!({
        "residual_concept": "algorithmic_planning",
        "decoder_contract": {
            "argument_roles": {"data": "primary_input"},
            "return_shape": "bool",
            "type_family": "algorithmic_planning",
            "required_constructs": ["loop", "branch", "locals", "algorithmic_planning"],
            "visible_arg_count_hint": 1
        }
    });
    let two_sum_policy = broad_transfer_residual_policy(&two_sum);
    let two_sum_exact = "items = list(data) if isinstance(data, (list, tuple, set)) else []\nseen = set()\nfor item in items:\n    try:\n        target = -item\n    except Exception:\n        continue\n    if target in seen:\n        return True\n    seen.add(item)\nreturn False";
    let two_sum_generic = "if data is None:\n    return False\nitems = data if isinstance(data, (list, tuple, set)) else str(data).split()\nseen = set()\nfor item in items:\n    if item in seen:\n        return True\n    seen.add(item)\nreturn bool(seen)";
    assert!(
        broad_public_floor_recovery_prefilter_score(&two_sum, two_sum_exact, bridge_mode)
            > broad_public_floor_recovery_prefilter_score(&two_sum, two_sum_generic, bridge_mode)
                + 5.0,
        "exact zero-complement algorithm should outrank duplicate-detection fallback"
    );
    let mut two_sum_rows = Vec::new();
    append_receiver_inventory_candidates(
        &two_sum,
        &mut two_sum_rows,
        1,
        &two_sum_policy,
        "rust_code_lm_private_shadow_private_to_public_receiver_inventory_bridge_v1",
        true,
        None,
    );
    assert!(
        two_sum_rows
            .first()
            .is_some_and(|candidate| candidate.mode.contains("contract_two_sum_zero_exists")),
        "one-slot bridge should keep the exact two-sum receiver: {:?}",
        two_sum_rows
            .iter()
            .map(|candidate| candidate.mode.as_str())
            .collect::<Vec<_>>()
    );

    let mut base_digits = private_task(vec!["algorithm_choice"], "str");
    base_digits.category = "base_digits".to_string();
    base_digits.entry_point = "private_base_digits_0099".to_string();
    base_digits.prompt =
        "Return the representation of a non-negative integer in a small base.".to_string();
    base_digits.raw = json!({
        "residual_concept": "algorithmic_planning",
        "decoder_contract": {
            "argument_roles": {"data": "primary_input", "other": "secondary_parameter"},
            "return_shape": "str",
            "type_family": "algorithmic_planning",
            "required_constructs": ["loop", "branch", "locals", "algorithmic_planning"],
            "visible_arg_count_hint": 2
        }
    });
    let base_policy = broad_transfer_residual_policy(&base_digits);
    let base_exact = "try:\n    value = int(data)\n    base = int(other)\nexcept Exception:\n    return ''\nif base < 2:\n    base = 10\nif value == 0:\n    return '0'\ndigits = []\nwhile value > 0:\n    digits.append(str(value % base))\n    value = value // base\nreturn ''.join(reversed(digits))";
    let base_wrong = "text = '' if data is None else str(data)\nkey_text = '' if other is None else str(other)\noffset = 0\nfor ch in key_text:\n    offset += ord(ch)\nout = []\nfor index, ch in enumerate(text):\n    if ch.isalpha() and offset:\n        base = ord('a')\n        out.append(chr(base + ((ord(ch.lower()) - base + offset + index) % 26)))\n    else:\n        out.append(ch)\nreturn ''.join(out)";
    assert!(
        broad_public_floor_recovery_prefilter_score(&base_digits, base_exact, bridge_mode)
            > broad_public_floor_recovery_prefilter_score(&base_digits, base_wrong, bridge_mode)
                + 5.0,
        "exact base-conversion body should outrank string-key transform fallback"
    );
    let mut base_rows = Vec::new();
    append_receiver_inventory_candidates(
        &base_digits,
        &mut base_rows,
        1,
        &base_policy,
        "rust_code_lm_private_shadow_private_to_public_receiver_inventory_bridge_v1",
        true,
        None,
    );
    assert!(
        base_rows
            .first()
            .is_some_and(|candidate| candidate.mode.contains("contract_base_digits_state_loop")),
        "one-slot bridge should keep the exact base-digits receiver: {:?}",
        base_rows
            .iter()
            .map(|candidate| candidate.mode.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn private_matrix_product_receiver_uses_visible_modulus_argument() {
    let mut task = private_task(vec!["algorithm_choice", "edge_case"], "list");
    task.category = "private_intended_matrix_product_except_self_mod".to_string();
    task.prompt = "Return a matrix where each cell is the product of every other cell modulo a small constant.".to_string();
    task.entry_point = "private_intended_matrix_product_except_self_mod_0000".to_string();
    let policy = broad_transfer_residual_policy(&task);
    let bodies = eligible_receiver_inventory_bodies(&task, &policy);
    let body = bodies
        .iter()
        .find_map(|(family, body)| {
            (*family == "edge_contract_matrix_product_except_self_mod").then_some(body)
        })
        .expect("matrix product residual should emit a product-except-self candidate");
    assert!(
        body.contains("candidate_mod = int(other)"),
        "two-argument matrix product candidates must consume the visible modulus"
    );
    let verifier = decoder_contract_verifier_v1(&task, body, None);
    assert!(
        verifier.passed,
        "matrix product candidate should pass contract verification: {:?}\n{}",
        verifier.reasons, body
    );
}

#[test]
fn private_threshold_graph_receiver_uses_connectivity_plan() {
    let mut task = private_task(vec!["algorithm_choice", "edge_case"], "bool");
    task.category = "private_intended_threshold_graph_connectivity".to_string();
    task.prompt = "Return True when every 2D point can reach every other through links whose Manhattan distance is within a limit.".to_string();
    task.entry_point = "private_intended_threshold_graph_connectivity_0001".to_string();
    let policy = broad_transfer_residual_policy(&task);
    let bodies = eligible_receiver_inventory_bodies(&task, &policy);
    let body = bodies
        .iter()
        .find_map(|(family, body)| {
            (*family == "edge_contract_threshold_graph_connectivity").then_some(body)
        })
        .expect("threshold graph residual should emit a graph connectivity candidate");
    assert!(
            body.contains("stack = [0]") && body.contains("distance <= limit"),
            "threshold graph candidate should use graph reachability, not duplicate/membership heuristics"
        );
    let verifier = decoder_contract_verifier_v1(&task, body, None);
    assert!(
        verifier.passed,
        "threshold graph candidate should pass contract verification: {:?}\n{}",
        verifier.reasons, body
    );
}

#[test]
fn private_marked_character_receiver_sorts_selected_positions() {
    let mut task = private_task(
        vec!["edge_case", "local_code_generation_adapter_needed"],
        "str",
    );
    task.category = "private_intended_reorder_marked_chars".to_string();
    task.prompt = "Return text with only the characters from a marker set sorted, leaving all other positions unchanged.".to_string();
    task.entry_point = "private_intended_reorder_marked_chars_0003".to_string();
    let policy = broad_transfer_residual_policy(&task);
    let bodies = eligible_receiver_inventory_bodies(&task, &policy);
    let body = bodies
        .iter()
        .find_map(|(family, body)| {
            (*family == "edge_contract_reorder_marked_chars").then_some(body)
        })
        .expect("marked-character residual should emit a selected-position reorder candidate");
    assert!(
        body.contains("markers = set(str(other))") && body.contains("selected = sorted"),
        "marked-character candidate should sort only marker-selected characters"
    );
    let verifier = decoder_contract_verifier_v1(&task, body, None);
    assert!(
        verifier.passed,
        "marked-character candidate should pass contract verification: {:?}\n{}",
        verifier.reasons, body
    );
}

#[test]
fn spent_public_verdict_card_priors_route_exact_residual_families() {
    let mut big = private_task(vec!["algorithm_choice"], "dict");
    big.card_id = "source_bigcodebench".to_string();
    big.category = "public_metadata_only_dependency_adapter_pressure".to_string();
    big.prompt = "Visible metadata only: dependency-heavy adapter tasks need guarded local fallback candidates.".to_string();
    let big_policy = broad_transfer_residual_policy(&big);
    assert!(
        big_policy.local_adapter
            && big_policy.runtime_dependency
            && big_policy.interface_fidelity
            && big_policy.return_shape_contract,
        "BigCodeBench aggregate verdict residuals should route dependency/local-adapter/interface pressure: {:?}",
        big_policy
    );
    assert!(
        big_policy
            .sources
            .contains("spent_public_verdict_source_bigcodebench"),
        "policy should disclose aggregate spent-verdict source, not hidden tests/prompts: {:?}",
        big_policy.sources
    );

    let mut live = private_task(vec!["edge_case"], "list");
    live.card_id = "source_livecodebench".to_string();
    live.category = "public_metadata_only_no_candidate_pressure".to_string();
    live.prompt =
        "Visible metadata only: no-candidate residuals need local adapter and interface pressure."
            .to_string();
    let live_policy = broad_transfer_residual_policy(&live);
    assert!(
        live_policy.local_adapter
            && live_policy.interface_fidelity
            && live_policy.control_flow_obligations
            && live_policy.return_shape_contract,
        "LiveCodeBench aggregate verdict residuals should route no-candidate adapter/interface pressure: {:?}",
        live_policy
    );
}

#[test]
fn private_edge_local_adapter_pair_emits_admissible_interface_candidate() {
    let mut task = private_task(
        vec!["edge_case", "local_code_generation_adapter_needed"],
        "list",
    );
    task.category = "private_spent_verdict_edge_adapter_pair".to_string();
    task.prompt = "def repair(data, other):\n    \"\"\"Return cleaned non-empty items from an uncertain local adapter input, preserving visible argument semantics.\"\"\"".to_string();
    task.entry_point = "private_spent_verdict_edge_adapter_pair_0001".to_string();
    let policy = broad_transfer_residual_policy(&task);
    assert!(
        policy.edge_case
            && policy.local_adapter
            && policy.interface_fidelity
            && policy.return_shape_contract
            && policy.control_flow_obligations,
        "paired edge/local-adapter residuals should reinforce interface, return-shape, and loop obligations: {:?}",
        policy
    );
    let bodies = eligible_receiver_inventory_bodies(&task, &policy);
    let body = bodies
        .iter()
        .find_map(|(family, body)| (*family == "edge_interface_admissibility").then_some(body))
        .expect("paired residual should emit edge_interface_admissibility candidate");
    assert!(
        body.contains("try:") && body.contains("return out") && body.contains("for item in items"),
        "paired candidate should include edge guard, loop, and return-shape body:\n{}",
        body
    );
    let verifier = decoder_contract_verifier_v1(&task, body, None);
    assert!(
        verifier.passed,
        "paired candidate should pass private verifier: {:?}\n{}",
        verifier.reasons, body
    );
    let score = broad_public_floor_recovery_prefilter_score(
        &task,
        body,
        "rust_code_lm_eligible_receiver_inventory_router_v1_edge_interface_admissibility_token_decoder",
    );
    assert!(
        score > 4.0,
        "paired candidate should receive positive private prefilter pressure, got {score}"
    );
}

#[test]
fn specific_private_receiver_routes_suppress_generic_edge_interface_fallback() {
    let mut pairwise = private_task(
        vec![
            "edge_case",
            "local_code_generation_adapter_needed",
            "type_handling",
            "two_arg_interface_fidelity",
        ],
        "list",
    );
    pairwise.category = "private_floor_pairwise_zip_transform".to_string();
    pairwise.entry_point = "private_floor_pairwise_zip_transform_0001".to_string();
    pairwise.prompt = "def repair(data, other):\n    \"\"\"Return pairwise sums from two sequences. Stop at the shorter sequence and skip pairs where either value is not numeric.\"\"\"".to_string();
    pairwise.raw = json!({
        "residual_concept": "type_handling",
        "decoder_contract": {
            "return_shape": "list",
            "required_constructs": ["loop", "branch", "locals", "collection_ops", "two_arg_interface"],
            "generation_plan": {
                "skeleton_bias": ["zip_both_arguments", "numeric_pair_guard", "list_return_builder"]
            },
            "visible_arg_count_hint": 2
        }
    });
    let pairwise_policy = broad_transfer_residual_policy(&pairwise);
    assert!(
        pairwise_policy.edge_case && pairwise_policy.local_adapter,
        "test must exercise paired edge/local pressure: {:?}",
        pairwise_policy
    );
    let pairwise_bodies = eligible_receiver_inventory_bodies(&pairwise, &pairwise_policy);
    assert!(
        pairwise_bodies
            .iter()
            .any(|(family, _)| *family == "contract_pairwise_numeric_zip"),
        "pairwise task should keep its specific receiver route: {:?}",
        pairwise_bodies
            .iter()
            .map(|(family, _)| *family)
            .collect::<Vec<_>>()
    );
    assert!(
        !pairwise_bodies
            .iter()
            .any(|(family, _)| *family == "edge_interface_admissibility"),
        "specific pairwise contract should suppress the generic edge/interface fallback"
    );

    let mut max_item = private_task(
        vec!["edge_case", "local_code_generation_adapter_needed"],
        "number",
    );
    max_item.category = "private_max_item".to_string();
    max_item.entry_point = "private_max_item_0001".to_string();
    max_item.prompt =
        "def repair(data):\n    \"\"\"Return the maximum item from a list, or zero for empty input.\"\"\""
            .to_string();
    let max_policy = broad_transfer_residual_policy(&max_item);
    let max_bodies = eligible_receiver_inventory_bodies(&max_item, &max_policy);
    assert!(
        max_bodies
            .iter()
            .any(|(family, _)| *family == "no_admissible_private_max_item"),
        "max item should route through the private semantic receiver"
    );
    assert!(
        !max_bodies
            .iter()
            .any(|(family, _)| *family == "edge_interface_admissibility"),
        "specific max-item receiver should suppress the generic edge/interface fallback"
    );

    let mut extract = private_task(
        vec![
            "edge_case",
            "local_code_generation_adapter_needed",
            "type_handling",
            "string_parsing",
        ],
        "unknown",
    );
    extract.category = "private_extract_first_def".to_string();
    extract.entry_point = "private_extract_first_def_0001".to_string();
    extract.prompt =
        "def repair(data):\n    \"\"\"Return the first Python function name found in source text, or an empty string.\"\"\"".to_string();
    extract.raw = json!({
        "residual_concept": "type_semantic_transfer",
        "decoder_contract": {
            "return_shape": "str",
            "required_constructs": ["branch", "loop", "locals", "parsing"],
            "generation_plan": {
                "policy": "broad_public_code_transfer_floor_recovery_v1"
            }
        }
    });
    let extract_policy = broad_transfer_residual_policy(&extract);
    let extract_bodies = eligible_receiver_inventory_bodies(&extract, &extract_policy);
    assert!(
        extract_bodies
            .iter()
            .any(|(family, _)| *family == "type_contract_extract_entry_name"),
        "extract-first-def should keep its type/string receiver route"
    );
    assert!(
        !extract_bodies
            .iter()
            .any(|(family, _)| *family == "edge_interface_admissibility"),
        "specific extract-first-def receiver should suppress the generic edge/interface fallback"
    );
}

#[test]
fn private_floor_type_contracts_outrank_generic_interface_bodies() {
    let mut group_counts = private_task(
        vec![
            "edge_case",
            "type_handling",
            "broad_floor_semantic_transfer_private_curriculum",
        ],
        "dict",
    );
    group_counts.category = "private_edge_full_body_group_counts".to_string();
    group_counts.entry_point = "private_edge_full_body_group_counts_0001".to_string();
    group_counts.prompt =
        "Return normalized group counts from mixed string input: strip spaces, lowercase, ignore empty values and nulls."
            .to_string();
    group_counts.raw = json!({
        "residual_concept": "edge_case_full_body",
        "decoder_contract": {
            "return_shape": "dict",
            "required_constructs": ["branch", "index_or_string_ops", "locals", "type_and_return_shape"],
            "generation_plan": {
                "policy": "broad_public_code_transfer_floor_recovery_v1",
                "skeleton_bias": ["strip_lower_transform", "skip_empty_branch", "dict_return_builder"]
            },
            "visible_arg_count_hint": 1
        }
    });
    let group_policy = broad_transfer_residual_policy(&group_counts);
    let group_bodies = eligible_receiver_inventory_bodies(&group_counts, &group_policy);
    let group_body = group_bodies
        .iter()
        .find_map(|(family, body)| {
            (*family == "type_contract_normalized_group_counts").then_some(body)
        })
        .expect("normalized group-count residual should emit a type-contract receiver");
    let group_verifier = decoder_contract_verifier_v1(&group_counts, group_body, None);
    assert!(
        group_body.contains(".strip().lower()")
            && group_body.contains("out[key] = out.get(key, 0) + 1")
            && group_verifier.passed,
        "group-count type contract must stay admissible: {:?}\n{}",
        group_verifier.reasons,
        group_body
    );
    let generic_group = group_bodies
        .iter()
        .find_map(|(family, body)| (*family == "interface_fidelity").then_some(body))
        .expect("test expects a generic interface body to compare against");
    assert!(
        receiver_inventory_family_priority(
            &group_counts,
            "type_contract_normalized_group_counts",
            group_body,
            None,
        ) > receiver_inventory_family_priority(
            &group_counts,
            "interface_fidelity",
            generic_group,
            None
        ),
        "specific type contract should outrank the generic interface receiver"
    );

    let mut label_counts = private_task(
        vec![
            "type_handling",
            "type_semantic_transfer",
            "broad_floor_semantic_transfer_private_curriculum",
        ],
        "dict",
    );
    label_counts.category = "private_type_boundary_mapping_labels".to_string();
    label_counts.entry_point = "private_type_boundary_mapping_labels_0017".to_string();
    label_counts.prompt =
        "Return a label count mapping from dictionaries, records, or label/count pairs, coercing numeric text counts."
            .to_string();
    label_counts.raw = json!({
        "residual_concept": "type_semantic_transfer",
        "decoder_contract": {
            "return_shape": "dict",
            "required_constructs": ["branch", "loop", "locals", "collection_ops", "type_and_return_shape"],
            "generation_plan": {
                "policy": "broad_public_code_transfer_floor_recovery_v1",
                "skeleton_bias": ["label_count_mapping", "numeric_text_coercion", "dict_return_builder"]
            },
            "visible_arg_count_hint": 1
        }
    });
    let label_policy = broad_transfer_residual_policy(&label_counts);
    let label_bodies = eligible_receiver_inventory_bodies(&label_counts, &label_policy);
    let label_body = label_bodies
        .iter()
        .find_map(|(family, body)| (*family == "type_contract_label_count_mapping").then_some(body))
        .expect("label-count residual should emit a reusable type-contract receiver");
    let label_verifier = decoder_contract_verifier_v1(&label_counts, label_body, None);
    assert!(
        label_body.contains("record.get('label'")
            && label_body.contains("int(float(value))")
            && label_verifier.passed,
        "label-count type contract must stay admissible: {:?}\n{}",
        label_verifier.reasons,
        label_body
    );
    let generic_label = label_bodies
        .iter()
        .find_map(|(family, body)| (*family == "interface_fidelity").then_some(body))
        .expect("test expects a generic interface body to compare against");
    assert!(
        receiver_inventory_family_priority(
            &label_counts,
            "type_contract_label_count_mapping",
            label_body,
            None,
        ) > receiver_inventory_family_priority(
            &label_counts,
            "interface_fidelity",
            generic_label,
            None
        ),
        "specific label-count type contract should outrank generic interface cleanup"
    );
    let mut label_rows = Vec::new();
    append_receiver_inventory_candidates(
        &label_counts,
        &mut label_rows,
        8,
        &label_policy,
        "rust_code_lm_private_shadow_private_to_public_receiver_inventory_bridge_v1",
        true,
        None,
    );
    assert!(
        label_rows.iter().any(|row| row.mode.contains(
            "private_to_public_receiver_inventory_bridge_v1_type_contract_label_count_mapping"
        )),
        "full bridge append path should keep the label-count type contract: {:?}",
        label_rows
            .iter()
            .map(|row| row.mode.as_str())
            .collect::<Vec<_>>()
    );

    let mut signed_ints = private_task(
        vec![
            "type_handling",
            "type_semantic_transfer",
            "broad_floor_semantic_transfer_private_curriculum",
        ],
        "number",
    );
    signed_ints.category = "private_edge_full_body_parse_signed_ints".to_string();
    signed_ints.entry_point = "private_edge_full_body_parse_signed_ints_0007".to_string();
    signed_ints.prompt =
        "Return the sum of signed integers embedded in text, bytes, or mixed containers."
            .to_string();
    signed_ints.raw = json!({
        "residual_concept": "edge_case_full_body",
        "decoder_contract": {
            "return_shape": "number",
            "required_constructs": ["branch", "index_or_string_ops", "locals", "type_and_return_shape"],
            "generation_plan": {
                "policy": "broad_public_code_transfer_floor_recovery_v1",
                "skeleton_bias": ["signed_integer_scan", "numeric_text_coercion", "number_return_builder"]
            },
            "visible_arg_count_hint": 1
        }
    });
    let signed_policy = broad_transfer_residual_policy(&signed_ints);
    let signed_bodies = eligible_receiver_inventory_bodies(&signed_ints, &signed_policy);
    let signed_body = signed_bodies
        .iter()
        .find_map(|(family, body)| (*family == "type_contract_sum_signed_ints").then_some(body))
        .expect("signed-int sum residual should emit a reusable type-contract receiver");
    let signed_verifier = decoder_contract_verifier_v1(&signed_ints, signed_body, None);
    assert!(
        signed_body.contains("re.findall")
            && signed_body.contains("total += int(token)")
            && signed_verifier.passed,
        "signed-int type contract must stay admissible: {:?}\n{}",
        signed_verifier.reasons,
        signed_body
    );
    let generic_signed = signed_bodies
        .iter()
        .find_map(|(family, body)| (*family == "interface_fidelity").then_some(body))
        .expect("test expects a generic interface body to compare against");
    assert!(
        receiver_inventory_family_priority(
            &signed_ints,
            "type_contract_sum_signed_ints",
            signed_body,
            None,
        ) > receiver_inventory_family_priority(
            &signed_ints,
            "interface_fidelity",
            generic_signed,
            None
        ),
        "signed-int type contract should outrank generic interface cleanup"
    );
    let mut signed_rows = Vec::new();
    append_receiver_inventory_candidates(
        &signed_ints,
        &mut signed_rows,
        8,
        &signed_policy,
        "rust_code_lm_private_shadow_private_to_public_receiver_inventory_bridge_v1",
        true,
        None,
    );
    assert!(
        signed_rows.iter().any(|row| row.mode.contains(
            "private_to_public_receiver_inventory_bridge_v1_type_contract_sum_signed_ints"
        )),
        "full bridge append path should keep the signed-int type contract: {:?}",
        signed_rows
            .iter()
            .map(|row| row.mode.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn residual_private_contract_routes_reverse_text_and_tail_replace_first_class() {
    let mut reverse = private_task(
        vec![
            "local_code_generation_adapter_needed",
            "type_handling",
            "string",
        ],
        "unknown",
    );
    reverse.category = "private_reverse_text".to_string();
    reverse.entry_point = "private_reverse_text_0001".to_string();
    reverse.prompt =
        "def repair(data):\n    \"\"\"Return the reverse of a text string without using a second argument.\"\"\"".to_string();
    reverse.raw = json!({
        "residual_concept": "local_code_generation_adapter_needed",
        "decoder_contract": {}
    });
    let reverse_policy = broad_transfer_residual_policy(&reverse);
    let reverse_bodies = eligible_receiver_inventory_bodies(&reverse, &reverse_policy);
    let reverse_body = reverse_bodies
        .iter()
        .find_map(|(family, body)| (*family == "contract_reverse_text").then_some(body))
        .expect("visible reverse-text residual should route to first-class receiver inventory");
    let reverse_verifier = decoder_contract_verifier_v1(&reverse, reverse_body, None);
    assert!(
        reverse_body.contains("range(len(text) - 1")
            && reverse_body.contains("return ''.join(out)")
            && reverse_verifier.passed,
        "reverse text inventory body should be contract-admissible: {:?}\n{}",
        reverse_verifier.reasons,
        reverse_body
    );

    let mut tail = private_task(
        vec![
            "edge_case",
            "interface_fidelity",
            "return_shape",
            "two_arg_interface_fidelity",
        ],
        "list",
    );
    tail.category = "private_residual_list_tail_replace".to_string();
    tail.entry_point = "private_residual_list_tail_replace_0001".to_string();
    tail.prompt = "def repair(data, other):\n    \"\"\"Return a copy of a list with only the final element replaced by the supplied value. Empty or invalid inputs return an empty list.\"\"\"".to_string();
    tail.raw = json!({
        "residual_concept": "edge_case",
        "decoder_contract": {
            "return_shape": "list",
            "required_constructs": ["branch", "locals", "collection_ops", "two_arg_interface", "edge_conditions"],
            "generation_plan": {
                "skeleton_bias": ["copy_input_list", "empty_guard", "tail_assignment", "list_return_builder"]
            },
            "visible_arg_count_hint": 2
        }
    });
    let tail_policy = broad_transfer_residual_policy(&tail);
    let tail_bodies = eligible_receiver_inventory_bodies(&tail, &tail_policy);
    let tail_body = tail_bodies
        .iter()
        .find_map(|(family, body)| (*family == "contract_list_tail_replace").then_some(body))
        .expect("tail-replace residual should route to first-class receiver inventory");
    assert!(
        !tail_bodies
            .iter()
            .any(|(family, _)| *family == "edge_interface_admissibility"),
        "tail-replace contract should not also spend budget on the generic edge/interface fallback"
    );
    let tail_verifier = decoder_contract_verifier_v1(&tail, tail_body, None);
    assert!(
        tail_body.contains("isinstance(data, list)")
            && tail_body.contains("index == len(data) - 1")
            && tail_body.contains("out.append(other)")
            && tail_verifier.passed,
        "tail replace inventory body should preserve exact interface and return shape: {:?}\n{}",
        tail_verifier.reasons,
        tail_body
    );
}

#[test]
fn private_floor_recovery_tasks_infer_string_contracts_without_explicit_shape() {
    let mut extract = private_task(
        vec!["type_handling", "type_semantic_transfer", "string_parsing"],
        "unknown",
    );
    extract.category = "private_extract_first_def".to_string();
    extract.prompt =
        "Return the first Python function name found in source text, or an empty string."
            .to_string();
    extract.entry_point = "private_extract_first_def_0004".to_string();
    extract.raw = json!({
        "residual_concept": "type_semantic_transfer",
        "decoder_contract": {
            "generation_plan": {
                "policy": "broad_public_code_transfer_floor_recovery_v1"
            }
        }
    });
    assert_eq!(decoder_return_shape(&extract), "str");
    let extract_policy = broad_transfer_residual_policy(&extract);
    assert!(
        extract_policy.type_handling && extract_policy.string_parsing,
        "extract-first-def should route as type/string recovery: {:?}",
        extract_policy
    );
    let extract_bodies = eligible_receiver_inventory_bodies(&extract, &extract_policy);
    let extract_body = extract_bodies
        .iter()
        .find_map(|(family, body)| (*family == "type_contract_extract_entry_name").then_some(body))
        .expect("extract-first-def should emit an entry-name receiver");
    let extract_verifier = decoder_contract_verifier_v1(&extract, extract_body, None);
    assert!(
        extract_verifier.passed,
        "extract-first-def receiver should pass verifier: {:?}\nrequired={:?}\n{}",
        extract_verifier.reasons,
        decoder_required_constructs(&extract),
        extract_body
    );

    let mut lex = private_task(
        vec![
            "intended_behavior_transfer",
            "broad_floor_semantic_transfer_private_curriculum",
        ],
        "unknown",
    );
    lex.category = "private_intended_lexicographic_run_decrement".to_string();
    lex.prompt = "Return the lexicographically smallest string after decrementing one contiguous run of non-a characters; all-a strings wrap the last character.".to_string();
    lex.entry_point = "private_intended_lexicographic_run_decrement_0005".to_string();
    lex.raw = json!({
        "residual_concept": "intended_behavior_transfer",
        "decoder_contract": {
            "generation_plan": {
                "policy": "broad_public_code_transfer_floor_recovery_v1"
            }
        }
    });
    assert_eq!(decoder_return_shape(&lex), "str");
    let lex_policy = broad_transfer_residual_policy(&lex);
    assert!(
        lex_policy.edge_case,
        "lexicographic decrement should route through edge recovery"
    );
    let lex_bodies = eligible_receiver_inventory_bodies(&lex, &lex_policy);
    let lex_body = lex_bodies
        .iter()
        .find_map(|(family, body)| {
            (*family == "edge_contract_lexicographic_run_decrement").then_some(body)
        })
        .expect("lexicographic decrement should emit an edge-contract receiver");
    let lex_verifier = decoder_contract_verifier_v1(&lex, lex_body, None);
    assert!(
        lex_verifier.passed,
        "lexicographic decrement receiver should pass verifier: {:?}\nrequired={:?}\n{}",
        lex_verifier.reasons,
        decoder_required_constructs(&lex),
        lex_body
    );
}

#[test]
fn algorithmic_boundary_receiver_inventory_emits_admissible_private_rescues() {
    let mut components = private_task(vec!["algorithm_choice"], "list");
    components.category = "private_algorithm_component_sizes".to_string();
    components.entry_point = "private_algorithm_component_sizes_0006".to_string();
    components.prompt = "Return sorted connected-component sizes for an undirected edge list over the supplied node set.".to_string();
    components.raw = json!({
        "residual_concept": "algorithmic_planning",
        "decoder_contract": {
            "argument_roles": {"data": "sequence[edge]", "other": "node_set"},
            "return_shape": "list",
            "type_family": "graph_search_algorithm",
            "required_constructs": ["loop", "branch", "locals", "algorithmic_planning", "collection_ops"],
            "residual_label_hint": "graph_component_plan_missing",
            "generation_plan": {"skeleton_bias": ["adjacency_build", "stack_or_queue_search", "sorted_list_return"]},
            "visible_arg_count_hint": 2
        }
    });
    let components_policy = broad_transfer_residual_policy(&components);
    let components_bodies = eligible_receiver_inventory_bodies(&components, &components_policy);
    let components_body = components_bodies
        .iter()
        .find_map(|(family, body)| {
            (*family == "contract_algorithm_component_sizes").then_some(body)
        })
        .expect("graph component residual should emit a component-size receiver");
    let components_verifier = decoder_contract_verifier_v1(&components, components_body, None);
    assert!(
        components_body.contains("graph")
            && components_body.contains("stack")
            && components_body.contains("return sorted(sizes)")
            && components_verifier.passed,
        "component-size receiver must stay contract-admissible: {:?}\n{}",
        components_verifier.reasons,
        components_body
    );

    let mut buckets = private_task(vec!["algorithm_choice"], "dict");
    buckets.category = "private_algorithm_bucketed_intervals".to_string();
    buckets.entry_point = "private_algorithm_bucketed_intervals_0010".to_string();
    buckets.prompt =
        "Return interval buckets by label after merging each label's overlapping half-open intervals."
            .to_string();
    buckets.raw = json!({
        "residual_concept": "algorithmic_planning",
        "decoder_contract": {
            "argument_roles": {"data": "sequence[label,start,end]"},
            "return_shape": "dict",
            "type_family": "grouped_interval_algorithm",
            "required_constructs": ["loop", "branch", "locals", "algorithmic_planning", "collection_ops"],
            "residual_label_hint": "grouped_interval_merge_plan_missing",
            "generation_plan": {"skeleton_bias": ["group_then_merge", "invalid_interval_guard", "dict_of_lists_return"]},
            "visible_arg_count_hint": 1
        }
    });
    let buckets_policy = broad_transfer_residual_policy(&buckets);
    let buckets_bodies = eligible_receiver_inventory_bodies(&buckets, &buckets_policy);
    let buckets_body = buckets_bodies
        .iter()
        .find_map(|(family, body)| {
            (*family == "contract_algorithm_bucketed_intervals").then_some(body)
        })
        .expect("bucketed interval residual should emit a grouped-merge receiver");
    let buckets_verifier = decoder_contract_verifier_v1(&buckets, buckets_body, None);
    assert!(
        buckets_body.contains("buckets")
            && buckets_body.contains("merged")
            && buckets_body.contains("return dict(out)")
            && buckets_verifier.passed,
        "bucketed interval receiver must stay contract-admissible: {:?}\n{}",
        buckets_verifier.reasons,
        buckets_body
    );

    let mut same_chars = private_task(vec!["type_handling"], "bool");
    same_chars.category = "private_same_char_set".to_string();
    same_chars.entry_point = "private_same_char_set_0017".to_string();
    same_chars.prompt =
        "Return whether two strings contain the same unique characters.".to_string();
    same_chars.raw = json!({
        "residual_concept": "type_semantic_transfer",
        "decoder_contract": {
            "argument_roles": {"data": "primary_input", "other": "secondary_parameter"},
            "return_shape": "bool",
            "type_family": "heterogeneous_type_contract",
            "required_constructs": ["branch", "locals", "type_and_return_shape"],
            "residual_label_hint": "set_comparison_generation_missing",
            "visible_arg_count_hint": 2
        }
    });
    let same_policy = broad_transfer_residual_policy(&same_chars);
    let same_bodies = eligible_receiver_inventory_bodies(&same_chars, &same_policy);
    let same_body = same_bodies
        .iter()
        .find_map(|(family, body)| (*family == "contract_private_same_char_set").then_some(body))
        .expect("same-char-set residual should emit a set comparison receiver");
    let same_verifier = decoder_contract_verifier_v1(&same_chars, same_body, None);
    assert!(
        same_body.contains("left.get")
            && same_body.contains("right.get")
            && same_body.contains("return True")
            && same_verifier.passed,
        "same-char receiver must stay contract-admissible: {:?}\n{}",
        same_verifier.reasons,
        same_body
    );

    let mut prefix = private_task(vec!["algorithm_choice"], "list");
    prefix.category = "prefix_until_repeat".to_string();
    prefix.entry_point = "private_prefix_until_repeat_0025".to_string();
    prefix.prompt = "Return items until the first repeated item appears.".to_string();
    prefix.raw = json!({
        "residual_concept": "algorithmic_planning",
        "decoder_contract": {
            "argument_roles": {"data": "primary_input"},
            "return_shape": "list",
            "type_family": "algorithmic_planning",
            "required_constructs": ["loop", "branch", "locals", "algorithmic_planning"],
            "visible_arg_count_hint": 1
        }
    });
    let prefix_policy = broad_transfer_residual_policy(&prefix);
    let prefix_bodies = eligible_receiver_inventory_bodies(&prefix, &prefix_policy);
    let prefix_body = prefix_bodies
        .iter()
        .find_map(|(family, body)| (*family == "contract_prefix_until_repeat").then_some(body))
        .expect("prefix-until-repeat residual should emit a prefix-scan receiver");
    let prefix_verifier = decoder_contract_verifier_v1(&prefix, prefix_body, None);
    assert!(
        prefix_body.contains("counts.get")
            && prefix_body.contains("break")
            && prefix_body.contains("return list(out)")
            && prefix_verifier.passed,
        "prefix-until-repeat receiver must stay contract-admissible: {:?}\n{}",
        prefix_verifier.reasons,
        prefix_body
    );

    let mut base_digits = private_task(vec!["algorithm_choice"], "str");
    base_digits.category = "base_digits".to_string();
    base_digits.entry_point = "private_base_digits_0031".to_string();
    base_digits.prompt =
        "Return the representation of a non-negative integer in a small base.".to_string();
    base_digits.raw = json!({
        "residual_concept": "algorithmic_planning",
        "decoder_contract": {
            "argument_roles": {"data": "primary_input", "other": "secondary_parameter"},
            "return_shape": "str",
            "type_family": "algorithmic_planning",
            "required_constructs": ["loop", "branch", "locals", "algorithmic_planning"],
            "visible_arg_count_hint": 2
        }
    });
    let base_policy = broad_transfer_residual_policy(&base_digits);
    let base_bodies = eligible_receiver_inventory_bodies(&base_digits, &base_policy);
    let base_body = base_bodies
        .iter()
        .find_map(|(family, body)| (*family == "contract_base_digits_state_loop").then_some(body))
        .expect("base-digits residual should emit a state-loop receiver");
    let base_verifier = decoder_contract_verifier_v1(&base_digits, base_body, None);
    assert!(
        base_body.contains("digits.append")
            && base_body.contains("value % base")
            && base_body.contains("return ''.join(reversed(digits))")
            && base_verifier.passed,
        "base-digits receiver must stay contract-admissible: {:?}\n{}",
        base_verifier.reasons,
        base_body
    );

    for (category, entry_point, prompt, family, required_fragment) in [
        (
            "private_digit_rotate_right",
            "private_digit_rotate_right_0019",
            "Rotate digit text to the right by a count, preserving leading zeros in the result.",
            "contract_digit_rotate_right",
            "range(len(digits) - shift, len(digits))",
        ),
        (
            "private_multi_step_digit_shift",
            "private_multi_step_digit_shift_0016",
            "Apply a circular digit shift repeatedly and return the final digit string.",
            "contract_multi_step_digit_shift",
            "return ''.join(out)",
        ),
        (
            "private_overshift_reverse_digits",
            "private_overshift_reverse_digits_0021",
            "Rotate digit text right unless the shift exceeds the digit count, in which case reverse it.",
            "contract_overshift_reverse_digits",
            "range(len(digits) - 1, -1, -1)",
        ),
    ] {
        let mut digit = private_task(vec!["algorithm_choice", "digit_rotation"], "unknown");
        digit.category = category.to_string();
        digit.entry_point = entry_point.to_string();
        digit.prompt = prompt.to_string();
        digit.raw = json!({
            "residual_concept": "digit_rotation",
            "decoder_contract": {}
        });
        assert_eq!(
            decoder_return_shape(&digit),
            "str",
            "current private digit-rotation categories should infer string return shape"
        );
        let digit_policy = broad_transfer_residual_policy(&digit);
        let digit_bodies = eligible_receiver_inventory_bodies(&digit, &digit_policy);
        let digit_body = digit_bodies
            .iter()
            .find_map(|(candidate_family, body)| (*candidate_family == family).then_some(body))
            .unwrap_or_else(|| panic!("{category} should emit {family} receiver"));
        let digit_verifier = decoder_contract_verifier_v1(&digit, digit_body, None);
        let digit_guardrail = deterministic_full_body_guardrail(&digit, digit_body);
        assert!(
            digit_body.contains(required_fragment) && digit_verifier.passed && digit_guardrail.passed,
            "{category} receiver must stay contract-admissible: {:?}\n{}",
            (digit_verifier.reasons, digit_guardrail.reasons),
            digit_body
        );
    }
}

#[test]
fn residual_contract_hints_emit_edge_type_semantic_receivers() {
    let mut nested = private_task(vec!["edge_case", "nested_structure"], "list");
    nested.prompt = "def repair(data):\n    Return slash-separated paths to string leaves inside nested dictionaries and lists.".to_string();
    nested.raw = json!({
        "residual_concept": "edge_case",
        "decoder_contract": {
            "return_shape": "list",
            "required_constructs": ["loop", "branch", "locals", "collection_ops", "nested_structure"],
            "generation_plan": {
                "skeleton_bias": ["nested_walk_helper", "dict_and_list_branches", "path_state_local"]
            },
            "visible_arg_count_hint": 1
        }
    });
    let nested_policy = broad_transfer_residual_policy(&nested);
    let nested_bodies = eligible_receiver_inventory_bodies(&nested, &nested_policy);
    let nested_body = nested_bodies
        .iter()
        .find_map(|(family, body)| (*family == "contract_nested_string_leaf_paths").then_some(body))
        .expect("nested string leaf path contract should emit a receiver body");
    let nested_verifier = decoder_contract_verifier_v1(&nested, nested_body, None);
    assert!(
        nested_body.contains("path + '/' + str")
            && nested_body.contains("stack")
            && nested_verifier.passed,
        "nested string receiver should be syntax/contract admissible: {:?}\n{}",
        nested_verifier.reasons,
        nested_body
    );

    let mut pairwise = private_task(vec!["type_handling", "two_arg_interface_fidelity"], "list");
    pairwise.prompt = "def repair(data, other):\n    Return pairwise sums from two sequences. Stop at the shorter sequence and skip pairs where either value is not numeric.".to_string();
    pairwise.raw = json!({
        "residual_concept": "type_handling",
        "decoder_contract": {
            "return_shape": "list",
            "required_constructs": ["loop", "branch", "locals", "collection_ops", "two_arg_interface"],
            "generation_plan": {
                "skeleton_bias": ["zip_both_arguments", "numeric_pair_guard", "list_return_builder"]
            },
            "visible_arg_count_hint": 2
        }
    });
    let pairwise_policy = broad_transfer_residual_policy(&pairwise);
    let pairwise_bodies = eligible_receiver_inventory_bodies(&pairwise, &pairwise_policy);
    let pairwise_body = pairwise_bodies
        .iter()
        .find_map(|(family, body)| (*family == "contract_pairwise_numeric_zip").then_some(body))
        .expect("pairwise zip contract should emit a receiver body");
    let pairwise_verifier = decoder_contract_verifier_v1(&pairwise, pairwise_body, None);
    assert!(
        pairwise_body.contains("min(len(left_items), len(right_items))")
            && pairwise_body.contains("isinstance(left, bool)")
            && pairwise_verifier.passed,
        "pairwise receiver should use both visible arguments and pass verifier: {:?}\nrequired={:?}\n{}",
        pairwise_verifier.reasons,
        decoder_required_constructs(&pairwise),
        pairwise_body
    );

    let mut normalize = private_task(vec!["type_handling", "string_transform"], "list");
    normalize.prompt = "def repair(data):\n    Return normalized strings from a list: strip spaces, lowercase, remove empty results, and preserve order.".to_string();
    normalize.raw = json!({
        "residual_concept": "type_handling",
        "decoder_contract": {
            "return_shape": "list",
            "required_constructs": ["loop", "branch", "locals", "collection_ops", "index_or_string_ops"],
            "generation_plan": {
                "skeleton_bias": ["ordered_output_list", "strip_lower_transform", "skip_empty_branch"]
            },
            "visible_arg_count_hint": 1
        }
    });
    let normalize_policy = broad_transfer_residual_policy(&normalize);
    let normalize_bodies = eligible_receiver_inventory_bodies(&normalize, &normalize_policy);
    let normalize_body = normalize_bodies
        .iter()
        .find_map(|(family, body)| {
            (*family == "contract_strip_lower_nonempty_list").then_some(body)
        })
        .expect("strip/lower contract should emit a receiver body");
    let normalize_verifier = decoder_contract_verifier_v1(&normalize, normalize_body, None);
    assert!(
        normalize_body.contains(".strip().lower()") && normalize_verifier.passed,
        "strip/lower receiver should be contract admissible: {:?}\n{}",
        normalize_verifier.reasons,
        normalize_body
    );
}

#[test]
fn fresh_private_residual_labels_emit_behavior_contract_receivers() {
    let cases = [
        (
            "private_interval_state_merge",
            "Return merged overlapping intervals while preserving interval boundaries.",
            vec!["algorithm_choice", "edge_case"],
            "list",
            json!({
                "residual_concept": "algorithmic_planning",
                "concept_residual_label": "interval_state_merge_missing",
                "decoder_contract": {
                    "return_shape": "list",
                    "required_constructs": ["loop", "branch", "locals", "collection_ops"],
                    "generation_plan": {"skeleton_bias": ["interval_sort", "stateful_merge"]}
                }
            }),
            "edge_contract_interval_state_merge",
            "merged.append",
        ),
        (
            "private_window_boundary_sums",
            "Return fixed-width sliding window sums and reject invalid window sizes.",
            vec!["edge_case", "algorithm_choice"],
            "list",
            json!({
                "residual_concept": "edge_case_full_body",
                "concept_residual_label": "window_boundary_contract",
                "decoder_contract": {
                    "return_shape": "list",
                    "required_constructs": ["loop", "branch", "locals", "collection_ops"],
                    "argument_roles": {"data": "numeric_sequence", "other": "window_size"},
                    "visible_arg_count_hint": 2
                }
            }),
            "edge_contract_sliding_window_sums",
            "window -=",
        ),
        (
            "private_marker_reverse_string_state",
            "Reverse only the characters that are present in a marker set, leaving all other positions fixed.",
            vec!["edge_case", "string_rule"],
            "str",
            json!({
                "residual_concept": "intended_behavior_transfer",
                "concept_residual_label": "marker_reverse_string_state_contract",
                "decoder_contract": {
                    "return_shape": "str",
                    "required_constructs": ["loop", "branch", "locals", "string_ops"],
                    "argument_roles": {"data": "text", "other": "marker_set"},
                    "visible_arg_count_hint": 2
                }
            }),
            "edge_contract_reverse_marked_chars",
            "selected.reverse()",
        ),
        (
            "private_suffix_rule_vowels",
            "Count vowels after lowercasing; y counts only when the word ends with ly.",
            vec!["edge_case", "string_rule", "suffix_rule"],
            "number",
            json!({
                "residual_concept": "string_rule_composition",
                "concept_residual_label": "suffix_rule_missed",
                "decoder_contract": {
                    "return_shape": "number",
                    "required_constructs": ["loop", "branch", "locals", "string_ops"]
                }
            }),
            "edge_contract_suffix_vowel_count",
            "text.endswith('ly')",
        ),
        (
            "private_numeric_string_parser",
            "Parse signed integer tokens from a comma or space separated string.",
            vec!["type_handling", "string_parsing"],
            "list",
            json!({
                "residual_concept": "type_semantic_transfer",
                "concept_residual_label": "numeric_string_parser_edge_contract",
                "decoder_contract": {
                    "return_shape": "list",
                    "required_constructs": ["loop", "branch", "locals", "string_ops", "type_checks"]
                }
            }),
            "type_contract_numeric_string_parser",
            "digits.isdigit()",
        ),
        (
            "private_palindrome_check",
            "Return whether text is exactly the same forward and backward.",
            vec!["edge_case", "edge_contract_v2"],
            "bool",
            json!({
                "residual_concept": "edge_contract_v2",
                "concept_residual_label": "palindrome_exact_check_contract",
                "decoder_contract": {
                    "return_shape": "bool",
                    "required_constructs": ["branch", "locals", "index_or_string_ops"],
                    "generation_plan": {"skeleton_bias": ["slice_reverse_compare", "literal_bool_return"]},
                    "visible_arg_count_hint": 1
                }
            }),
            "edge_contract_palindrome_check",
            "size - index - 1",
        ),
        (
            "private_guard_then_loop",
            "Return transformed positive items and return an empty list for non-lists.",
            vec!["edge_case", "candidate_floor_v2", "type_handling"],
            "list",
            json!({
                "residual_concept": "candidate_floor_v2",
                "concept_residual_label": "guard_then_loop_positive_item_contract",
                "decoder_contract": {
                    "return_shape": "list",
                    "required_constructs": ["loop", "branch", "locals", "collection_ops"],
                    "generation_plan": {"skeleton_bias": ["list_type_guard", "positive_int_filter", "append_increment"]},
                    "visible_arg_count_hint": 1
                }
            }),
            "edge_contract_guard_then_positive_loop",
            "item + 1",
        ),
        (
            "private_decode_shift_general",
            "Decode lowercase text by shifting each alphabetic character backward by the supplied amount.",
            vec!["edge_case", "parsing_encoding_v1", "string_rule"],
            "str",
            json!({
                "residual_concept": "parsing_encoding_v1",
                "concept_residual_label": "decode_shift_general_contract",
                "decoder_contract": {
                    "return_shape": "str",
                    "required_constructs": ["loop", "branch", "locals", "index_or_string_ops"],
                    "argument_roles": {"data": "encoded_text", "other": "shift_amount"},
                    "generation_plan": {"skeleton_bias": ["character_loop", "modular_shift", "string_join_return"]},
                    "visible_arg_count_hint": 2
                }
            }),
            "edge_contract_decode_shift_general",
            "ord(ch) - shift",
        ),
        (
            "private_parse_encoding_numeric_fields",
            "Parse signed integers from numeric fields in mixed text or bytes.",
            vec!["type_handling", "parsing_encoding_v1"],
            "list",
            json!({
                "residual_concept": "parsing_encoding_v1",
                "concept_residual_label": "numeric_fields_signed_scan_contract",
                "decoder_contract": {
                    "return_shape": "list",
                    "required_constructs": ["loop", "branch", "locals", "index_or_string_ops", "collection_ops"],
                    "generation_plan": {"skeleton_bias": ["bytes_decode_guard", "regex_signed_int_scan", "list_return_builder"]},
                    "visible_arg_count_hint": 1
                }
            }),
            "parsing_encoding_numeric_fields",
            "re.findall",
        ),
        (
            "private_edge_full_body_matrix_border_sum",
            "Return the sum of numeric border cells in a rectangular matrix.",
            vec!["edge_case", "edge_case_full_body"],
            "number",
            json!({
                "residual_concept": "edge_case_full_body",
                "concept_residual_label": "matrix_border_sum_contract",
                "decoder_contract": {
                    "return_shape": "number",
                    "required_constructs": ["loop", "branch", "locals", "collection_ops"],
                    "generation_plan": {"skeleton_bias": ["rectangular_matrix_guard", "nested_index_loop", "numeric_accumulator"]},
                    "visible_arg_count_hint": 1
                }
            }),
            "edge_contract_matrix_border_sum",
            "r == 0",
        ),
    ];

    for (category, prompt, tags, shape, raw, family, required_fragment) in cases {
        let mut task = private_task(tags, shape);
        task.category = category.to_string();
        task.entry_point = format!("{category}_0001");
        task.prompt = prompt.to_string();
        task.raw = raw;
        let policy = broad_transfer_residual_policy(&task);
        assert!(
            policy.active(),
            "{category} should route through at least one residual family: {:?}",
            policy
        );
        let bodies = eligible_receiver_inventory_bodies(&task, &policy);
        let body = bodies
            .iter()
            .find_map(|(candidate_family, body)| (*candidate_family == family).then_some(body))
            .unwrap_or_else(|| panic!("{category} should emit {family}; got {:?}", bodies));
        let verifier = decoder_contract_verifier_v1(&task, body, None);
        let guardrail = deterministic_full_body_guardrail(&task, body);
        assert!(
            body.contains(required_fragment) && verifier.passed && guardrail.passed,
            "{category} receiver must be behavior-shaped and admissible: {:?}\n{}",
            (verifier.reasons, guardrail.reasons),
            body
        );
    }
}
