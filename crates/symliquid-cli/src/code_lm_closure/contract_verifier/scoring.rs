use super::*;

pub(super) fn semantic_decoder_v2_token_bonus(
    task: &CodeTask,
    plan: &BTreeSet<String>,
    existing: &[String],
    token: &str,
) -> f32 {
    let mut score = decoder_return_shape_token_bonus(task, token);
    if plan.contains("loop") && matches!(token, "for" | "while" | "range") {
        score += 0.8;
    }
    if plan.contains("branch") && matches!(token, "if" | "else" | "elif") {
        score += 0.65;
    }
    if plan.contains("locals")
        && matches!(
            token,
            "total" | "count" | "out" | "items" | "counts" | "best"
        )
    {
        score += 0.45;
    }
    if plan.contains("frequency") && matches!(token, "counts" | "get" | "{}") {
        score += 0.7;
    }
    if plan.contains("selection") && matches!(token, "best" | "min" | "max" | "sorted") {
        score += 0.7;
    }
    if plan.contains("parsing") && matches!(token, "split" | "replace" | "isdigit" | "int") {
        score += 0.75;
    }
    if plan.contains("file_path")
        && matches!(
            token,
            "os" | "path" | "isfile" | "isdir" | "exists" | "open"
        )
    {
        score += 0.75;
    }
    if plan.contains("csv") && matches!(token, "csv" | "reader" | "writer" | "with" | "open") {
        score += 0.75;
    }
    if plan.contains("archive")
        && matches!(
            token,
            "zipfile" | "tarfile" | "shutil" | "make_archive" | "archive"
        )
    {
        score += 0.75;
    }
    if plan.contains("system_api")
        && matches!(
            token,
            "platform" | "psutil" | "subprocess" | "Popen" | "run"
        )
    {
        score += 0.65;
    }
    if plan.contains("structured_parsing")
        && matches!(token, "json" | "load" | "urlencode" | "items" | "get")
    {
        score += 0.65;
    }
    if plan.contains("algorithmic_planning")
        && matches!(token, "for" | "while" | "if" | "return" | "append")
    {
        score += 0.45;
    }
    if plan.contains("edge_conditions") && matches!(token, "if" | "not" | "return" | "[]" | "False")
    {
        score += 0.55;
    }
    if plan.contains("type_and_return_shape")
        && previous_meaningful_token(existing).as_deref() == Some("return")
    {
        match decoder_return_shape(task).as_str() {
            "list" if matches!(token, "out" | "[]" | "list") => score += 0.8,
            "dict" if matches!(token, "counts" | "{}" | "dict") => score += 0.8,
            "str" if matches!(token, "''" | "join" | "str") => score += 0.7,
            "bool" if matches!(token, "True" | "False") => score += 0.7,
            "number" if matches!(token, "total" | "count" | "best" | "0") => score += 0.7,
            _ => {}
        }
    }
    if (plan.contains("type_checks")
        || plan.contains("type_contract_v2")
        || plan.contains("type_and_return_shape"))
        && matches!(
            token,
            "isinstance" | "str" | "int" | "float" | "dict" | "list" | "tuple" | "set"
        )
    {
        score += 0.75;
    }
    if (plan.contains("normalization")
        || plan.contains("numeric_text")
        || plan.contains("normalize_mapping_keys"))
        && matches!(token, "strip" | "lower" | "replace" | "split" | "isdigit")
    {
        score += 0.75;
    }
    if plan.contains("nested_structure")
        && matches!(
            token,
            "stack" | "pop" | "extend" | "append" | "while" | "for"
        )
    {
        score += 0.65;
    }
    if plan.contains("interface_contracts")
        && matches!(token, "get" | "keys" | "other" | "return" | "if")
    {
        score += 0.55;
    }
    if previous_meaningful_token(existing).as_deref() == Some(":") && token == "<NL>" {
        score += 5.0;
    }
    score
}

pub(super) fn candidate_transfer_score(
    task: &CodeTask,
    candidate: &CandidateExpression,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> f32 {
    let mut score = body_transfer_score(task, &candidate.body);
    score += beautiful_body_score(task, &candidate.body);
    let verifier = decoder_contract_verifier_v1(task, &candidate.body, sts_streams);
    if verifier.passed {
        score += 1.15;
    } else {
        score -= (verifier.reasons.len() as f32).min(4.0) * 0.55;
    }
    if candidate.mode.contains("sparse_state_sequence") {
        score += 1.35;
    }
    if candidate.mode.contains("symliquid_recurrent_state") {
        score += 1.75;
    }
    if candidate.mode.contains("semantic_plan_v2") {
        score += 2.15;
    }
    if candidate.mode.contains("contract_guided_skeleton_decoder") {
        score += 2.65;
    }
    if candidate.mode.contains("contract_guided_token_decoder") {
        score += 2.05;
    }
    if candidate
        .mode
        .contains("sts_conditioned_contract_guided_token_decoder")
    {
        score += 2.35;
        if verifier.passed {
            score += 1.0;
        }
        if visible_argument_contract_ok(task, &candidate.body) {
            score += 0.8;
        }
    }
    if candidate.mode.contains("interface_role_repair") {
        score += 0.7;
    }
    if candidate.mode.contains("parser_ast_completion") {
        score += 0.45;
    }
    if candidate.mode.contains("causal_contract_skeleton_decoder") {
        score += 3.15;
    }
    if candidate
        .mode
        .contains("local_adapter_edge_skeleton_decoder")
    {
        score += 2.95;
    }
    if candidate.mode.contains("execution_shape_skeleton_decoder") {
        score += 2.35;
    }
    if candidate.mode.contains("sts_causal_skeleton_decoder") {
        score += 2.75;
    }
    if candidate.mode.contains("edge_exec_repair") {
        score += 1.15;
    }
    if optional_dependency_import_contract_ok(&candidate.body) {
        score += optional_dependency_fallback_bonus(&candidate.body);
    } else {
        score -= 8.0;
    }
    if candidate
        .mode
        .contains("eligible_receiver_inventory_router_v1")
        || candidate
            .mode
            .contains("private_to_public_receiver_inventory_bridge_v1")
    {
        score += if candidate
            .mode
            .contains("private_to_public_receiver_inventory_bridge_v1")
        {
            4.1
        } else {
            3.2
        };
        if verifier.passed {
            score += 1.2;
        }
        if visible_argument_contract_ok(task, &candidate.body) {
            score += 1.0;
        }
        if return_shape_contract_ok(task, &candidate.body.to_ascii_lowercase()) {
            score += 1.0 + return_shape_builder_bias(task, &candidate.body.to_ascii_lowercase());
        }
        let hints = decoder_required_constructs(task);
        if required_construct_contract_ok_for_task(task, &candidate.body, &hints) {
            score += 0.9;
        }
        if body_semantically_admissible(task, &candidate.body) {
            score += 0.5;
        }
    }
    if candidate.full_body_token_candidate {
        score += 0.6;
    }
    if candidate.expression_memory_fallback || candidate.sts_candidate_expression_used {
        score -= 2.0;
    }
    if sts_streams.is_some() && candidate.mode.contains("sts_conditioned") {
        score += sts_conditioned_rank_bias(&candidate.body, sts_streams, 0.25, 1.8);
        if candidate.mode.contains("execution_shape_skeleton_decoder") {
            score += if sts_decoder_control_demotes_sts_preference(sts_streams) {
                0.10
            } else {
                0.85
            };
        } else if candidate.mode.contains("causal_contract_skeleton_decoder") {
            score += if sts_decoder_control_demotes_sts_preference(sts_streams) {
                0.20
            } else {
                1.35
            };
        } else if candidate.mode.contains("contract_guided_skeleton_decoder") {
            score += if sts_decoder_control_demotes_sts_preference(sts_streams) {
                0.15
            } else {
                1.05
            };
        } else if candidate.mode.contains("contract_guided_token_decoder") {
            score += if sts_decoder_control_demotes_sts_preference(sts_streams) {
                0.15
            } else {
                0.95
            };
        } else if candidate
            .mode
            .contains("local_adapter_edge_skeleton_decoder")
        {
            score += if sts_decoder_control_demotes_sts_preference(sts_streams) {
                0.20
            } else {
                1.25
            };
        } else if candidate.mode.contains("sts_causal_skeleton_decoder") {
            score += if sts_decoder_control_demotes_sts_preference(sts_streams) {
                0.15
            } else {
                1.15
            };
        } else if candidate.mode.contains("semantic_plan_v2") {
            score += if sts_decoder_control_demotes_sts_preference(sts_streams) {
                0.05
            } else {
                0.45
            };
        } else if candidate.mode.contains("edge_exec_repair") {
            score += if sts_decoder_control_demotes_sts_preference(sts_streams) {
                0.05
            } else {
                0.25
            };
        }
    }
    if typed_edge_exec_receiver_v1_enabled() {
        score += typed_edge_exec_receiver_v1_score(task, candidate, &verifier, sts_streams);
    }
    if private_type_shape_receiver_veto_v1_enabled() {
        score += private_type_shape_receiver_veto_v1_score(task, candidate, &verifier, sts_streams);
    }
    score += broad_transfer_residual_candidate_score(task, candidate, &verifier, sts_streams);
    score
}

pub(super) fn typed_edge_exec_receiver_v1_enabled() -> bool {
    std::env::var_os("THESEUS_TYPED_EDGE_EXEC_RECEIVER_V1").is_some()
}

pub(super) fn private_type_shape_receiver_veto_v1_enabled() -> bool {
    std::env::var_os("THESEUS_PRIVATE_TYPE_SHAPE_RECEIVER_VETO_V1").is_some()
}

pub(super) fn typed_edge_exec_receiver_v1_score(
    task: &CodeTask,
    candidate: &CandidateExpression,
    verifier: &DecoderContractVerification,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> f32 {
    let mut score = 0.0f32;
    let lowered = candidate.body.to_lowercase();
    let hints = decoder_required_constructs(task);

    // Teacher diagnosis 2026-05-23: private verifier rates were saturated while
    // private body execution remained weak, so treat executable typed-edge
    // semantics as a receiver priority instead of another after-the-fact report.
    if verifier.passed {
        score += 1.4;
    } else {
        score -= 1.2 * verifier.reasons.len() as f32;
    }
    if visible_argument_contract_ok(task, &candidate.body) {
        score += 0.9;
    } else {
        score -= 1.1;
    }
    if return_shape_contract_ok(task, &lowered) {
        score += 1.0;
    } else {
        score -= 1.3;
    }
    if required_construct_contract_ok_for_task(task, &candidate.body, &hints) {
        score += 0.7;
    }
    if body_semantically_admissible(task, &candidate.body) {
        score += 0.7;
    }
    if execution_shape_contract_ok(task, &candidate.body, &hints) {
        score += 0.8;
    }
    if candidate.full_body_token_candidate {
        score += 0.9;
    } else {
        score -= 1.5;
    }
    if candidate.expression_memory_fallback || candidate.sts_candidate_expression_used {
        score -= 2.4;
    }
    if candidate
        .mode
        .contains("local_adapter_edge_skeleton_decoder")
    {
        score += 2.2;
    }
    if candidate.mode.contains("edge_exec_repair") {
        score += 2.0;
    }
    if candidate.mode.contains("causal_contract_skeleton_decoder") {
        score += 1.6;
    }
    if candidate.mode.contains("contract_guided_skeleton_decoder") {
        score += 1.2;
    }
    if candidate.mode.contains("contract_guided_token_decoder") {
        score += 1.4;
    }
    if candidate.mode.contains("semantic_plan_v2") {
        score += 0.25;
    }
    if sts_streams.is_some() && candidate.mode.contains("sts_conditioned") {
        score += sts_conditioned_rank_bias(&candidate.body, sts_streams, 0.6, 1.4);
    }
    if lowered.contains("raise runtimeerror") || candidate.mode.contains("no_admissible") {
        score -= 8.0;
    }
    score
}

pub(super) fn private_type_shape_receiver_veto_v1_score(
    task: &CodeTask,
    candidate: &CandidateExpression,
    verifier: &DecoderContractVerification,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> f32 {
    let lowered = candidate.body.to_lowercase();
    let mut score = 0.0f32;
    if !syntax_constrained_body(&candidate.body)
        || natural_language_leakage_in_body(&candidate.body)
    {
        return -24.0;
    }
    if lowered.contains("raise runtimeerror") || candidate.mode.contains("no_admissible") {
        return -18.0;
    }
    if unbound_item_reference(&candidate.body) {
        score -= 8.0;
    }
    if visible_argument_contract_ok(task, &candidate.body) {
        score += 1.8;
    } else {
        score -= 3.0;
    }
    if return_shape_contract_ok(task, &lowered) {
        score += 2.1;
        score += return_shape_builder_bias(task, &lowered);
    } else {
        score -= 5.5;
    }
    if semantic_family_contract_ok(task, &candidate.body) {
        score += 1.1;
    } else {
        score -= 2.1;
    }
    if body_semantically_admissible(task, &candidate.body) {
        score += 1.0;
    } else {
        score -= 1.6;
    }
    if verifier.passed {
        score += 0.9;
    } else {
        score -= 0.65 * verifier.reasons.len() as f32;
    }
    if candidate.full_body_token_candidate {
        score += 0.8;
    } else {
        score -= 1.4;
    }
    if candidate.expression_memory_fallback || candidate.sts_candidate_expression_used {
        score -= 2.2;
    }
    if candidate.mode.contains("causal_contract_skeleton_decoder")
        || candidate.mode.contains("contract_guided_skeleton_decoder")
        || candidate.mode.contains("contract_guided_token_decoder")
        || candidate
            .mode
            .contains("eligible_receiver_inventory_router_v1")
        || candidate
            .mode
            .contains("private_to_public_receiver_inventory_bridge_v1")
        || candidate
            .mode
            .contains("local_adapter_edge_skeleton_decoder")
    {
        score += 1.2;
    }
    if sts_streams.is_some() && candidate.mode.contains("sts_conditioned") {
        score += sts_conditioned_rank_bias(&candidate.body, sts_streams, 0.5, 1.0);
    }
    score
}

pub(super) fn return_shape_builder_bias(task: &CodeTask, lowered_body: &str) -> f32 {
    match decoder_return_shape(task).as_str() {
        "list" => {
            if lowered_body.contains('[')
                || lowered_body.contains(".append")
                || lowered_body.contains(".extend")
                || lowered_body.contains("sorted(")
                || lowered_body.contains("list(")
            {
                0.8
            } else {
                -1.0
            }
        }
        "dict" => {
            if lowered_body.contains('{')
                || lowered_body.contains("dict(")
                || lowered_body.contains(".update")
            {
                0.8
            } else {
                -1.0
            }
        }
        "tuple" => {
            if lowered_body.contains('(') || lowered_body.contains("tuple(") {
                0.55
            } else {
                -0.7
            }
        }
        "str" => {
            if lowered_body.contains("join(")
                || lowered_body.contains("str(")
                || lowered_body.contains(".replace")
                || lowered_body.contains(".strip")
            {
                0.5
            } else {
                0.0
            }
        }
        "bool" => {
            if lowered_body.contains("true")
                || lowered_body.contains("false")
                || lowered_body.contains("==")
                || lowered_body.contains(" in ")
                || lowered_body.contains(".startswith")
                || lowered_body.contains(".endswith")
            {
                0.45
            } else {
                -0.4
            }
        }
        "number" => {
            if lowered_body.contains("sum(")
                || lowered_body.contains("len(")
                || lowered_body.contains("count")
                || lowered_body.contains("+")
                || lowered_body.contains("-")
                || lowered_body.contains("*")
                || lowered_body.contains("/")
                || lowered_body.contains("%")
            {
                0.45
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}
