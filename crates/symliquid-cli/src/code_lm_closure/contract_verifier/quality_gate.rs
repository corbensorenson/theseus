// Decoder contract verification and cheap quality gates used before expensive sandbox work.

use super::*;

#[derive(Debug, Clone)]
pub(in crate::code_lm_closure) struct DecoderContractVerification {
    pub(in crate::code_lm_closure) passed: bool,
    pub(in crate::code_lm_closure) reasons: Vec<&'static str>,
}

pub(in crate::code_lm_closure) fn decoder_contract_verifier_v1(
    task: &CodeTask,
    body: &str,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> DecoderContractVerification {
    let mut reasons = Vec::new();
    let lowered = body.to_lowercase();
    let hints = decoder_required_constructs(task);
    if !syntax_constrained_body(body) {
        reasons.push("decoder_contract_ast_or_body_syntax_rejected");
    }
    if natural_language_leakage_in_body(body) {
        reasons.push("decoder_contract_natural_language_leakage");
    }
    if !optional_dependency_import_contract_ok(body) {
        reasons.push("decoder_contract_unguarded_optional_dependency_import");
    }
    if scaffold_placeholder_body(body) {
        reasons.push("decoder_contract_placeholder_scaffold_body");
    }
    if unbound_item_reference(body) {
        reasons.push("decoder_contract_unbound_item_reference");
    }
    if execution_shape_invalid_partial_statement(body) {
        reasons.push("decoder_contract_invalid_execution_shape_partial_statement");
    }
    if execution_shape_behavioral_antipattern(task, body) {
        reasons.push("decoder_contract_behavior_dead_execution_shape");
    }
    if !visible_argument_contract_ok(task, body) {
        reasons.push("decoder_contract_visible_argument_mismatch");
    }
    if !return_shape_contract_ok(task, &lowered) {
        reasons.push("decoder_contract_return_shape_mismatch");
    }
    if !prompt_dict_key_contract_ok(task, &lowered) {
        reasons.push("decoder_contract_missing_prompt_dict_key");
    }
    if !required_construct_contract_ok_for_task(task, body, &hints) {
        reasons.push("decoder_contract_missing_required_skeleton");
    }
    if vacuous_contract_body(task, body, &hints) {
        reasons.push("decoder_contract_vacuous_body");
    }
    if candidate_floor_v2_wall_body(task, body) {
        reasons.push("decoder_contract_candidate_floor_v2_wall_body");
    }
    if bogus_return_attribute_body(body) {
        reasons.push("decoder_contract_bogus_return_attribute");
    }
    if bogus_return_local_callable_body(body) {
        reasons.push("decoder_contract_bogus_return_local_callable");
    }
    if !execution_shape_library_contract_ok(task, body, &hints) {
        reasons.push("decoder_contract_execution_library_mismatch");
    }
    if !semantic_family_contract_ok(task, body) {
        reasons.push("decoder_contract_semantic_family_mismatch");
    }
    if !body_semantically_admissible(task, body) {
        reasons.push("decoder_contract_semantic_admissibility_rejected");
    }
    if sts_streams.is_some()
        && !sts_decoder_v2_hints(sts_streams).is_empty()
        && sts_skeleton_alignment_score(body, sts_streams) <= -1.0
        && !private_residual_v3_sts_contract_alignment_ok(task, body, &lowered, sts_streams)
    {
        reasons.push("decoder_contract_sts_skeleton_misaligned");
    }
    DecoderContractVerification {
        passed: reasons.is_empty(),
        reasons,
    }
}

fn private_residual_v3_sts_contract_alignment_ok(
    task: &CodeTask,
    body: &str,
    lowered_body: &str,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> bool {
    let Some(streams) = sts_streams else {
        return false;
    };
    let contract = task.raw.get("decoder_contract").and_then(Value::as_object);
    let policy = contract
        .and_then(|contract| contract.get("policy"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if task.card_id != "private_residual_repair_v3"
        || !task
            .benchmark_evidence_level
            .contains("private_residual_repair_v3_generated_only")
        || policy != "project_theseus_decoder_contract_v3_private_residual_repair"
    {
        return false;
    }
    let stream_text = streams
        .values()
        .map(|value| value.to_lowercase())
        .collect::<Vec<_>>()
        .join("\n");
    if !stream_text.contains("private_residual_v3_decoder_contract_stream") {
        return false;
    }
    let hints = decoder_required_constructs(task);
    syntax_constrained_body(body)
        && visible_argument_contract_ok(task, body)
        && return_shape_contract_ok(task, lowered_body)
        && required_construct_contract_ok_for_task(task, body, &hints)
        && semantic_family_contract_ok(task, body)
        && body_semantically_admissible(task, body)
}

fn private_public_transfer_decoder_contract(task: &CodeTask) -> bool {
    task.raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("policy"))
        .and_then(Value::as_str)
        .is_some_and(|policy| {
            policy == "project_theseus_decoder_contract_v3_private_public_transfer"
        })
}

fn prompt_dict_key_contract_ok(task: &CodeTask, lowered_body: &str) -> bool {
    if decoder_return_shape(task) != "dict" {
        return true;
    }
    let text = task_contract_text(task);
    for (needle, body_key) in [
        ("operating system", "operating system"),
        ("architecture", "architecture"),
        ("memory usage", "memory usage"),
    ] {
        if text.contains(needle) && !lowered_body.contains(body_key) {
            return false;
        }
    }
    true
}

pub(in crate::code_lm_closure) fn scaffold_placeholder_body(body: &str) -> bool {
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if compact.contains("args=(")
        || compact.contains("data=args[")
        || compact.contains("other=args[")
        || compact.contains("extra=args[")
        || compact.contains("extra=()")
    {
        return true;
    }
    for line in lowered.lines().map(str::trim) {
        if line.starts_with("args =")
            || line.starts_with("args=")
            || line.starts_with("extra =")
            || line.starts_with("extra=")
            || line == "other = none"
            || line == "other=none"
        {
            return true;
        }
    }
    for empty in [
        "false", "true", "none", "0", "1", "[]", "{}", "()", "''", "\"\"",
    ] {
        let spaced = format!("result = {empty}");
        let tight = format!("result={empty}");
        if (lowered.contains(&spaced) || lowered.contains(&tight))
            && compact.contains("returnresult")
            && !lowered.contains("result +=")
            && !lowered.contains("result+=")
            && !lowered.contains("result.append")
            && !lowered.contains("result.extend")
            && !lowered.contains("result.update")
        {
            return true;
        }
    }
    false
}

pub(in crate::code_lm_closure) fn beautiful_body_score(task: &CodeTask, body: &str) -> f32 {
    let lowered = body.to_lowercase();
    let hints = decoder_required_constructs(task);
    let mut score = 0.0f32;
    let non_empty_lines = body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .count();

    if scaffold_placeholder_body(body) {
        score -= 8.0;
    } else {
        score += 1.0;
    }
    if visible_argument_contract_ok(task, body) {
        score += 1.0;
    } else {
        score -= 2.0;
    }
    if return_shape_contract_ok(task, &lowered) {
        score += 1.0 + return_shape_builder_bias(task, &lowered).max(0.0);
    } else {
        score -= 2.5;
    }
    if required_construct_contract_ok_for_task(task, body, &hints) {
        score += 0.7;
    } else {
        score -= 1.5;
    }
    if semantic_family_contract_ok(task, body) {
        score += 0.7;
    } else {
        score -= 1.4;
    }
    if optional_dependency_import_contract_ok(body) {
        score += optional_dependency_fallback_bonus(body);
    } else {
        score -= 6.0;
    }
    if body_semantically_admissible(task, body) {
        score += 0.8;
    } else {
        score -= 1.8;
    }
    if vacuous_contract_body(task, body, &hints) {
        score -= 4.0;
    }
    if bogus_return_attribute_body(body) {
        score -= 8.0;
    }
    if bogus_return_local_callable_body(body) {
        score -= 8.0;
    }
    if non_empty_lines >= 2 {
        score += 0.35;
    }
    if non_empty_lines > 24 {
        score -= 0.35;
    }
    let overcomposed_lines = body
        .lines()
        .map(str::trim)
        .filter(|line| invalid_overcomposed_generated_line(line))
        .count();
    if overcomposed_lines > 0 {
        score -= 4.0 * overcomposed_lines as f32;
    }
    if lowered.contains("try:")
        && lowered.contains("except")
        && (lowered.contains("return []")
            || lowered.contains("return {}")
            || lowered.contains("return false")
            || lowered.contains("return none"))
    {
        score -= 0.6;
    }
    if lowered.contains("pass\n") || lowered.ends_with("pass") {
        score -= 3.0;
    }
    score
}

fn vacuous_contract_body(task: &CodeTask, body: &str, hints: &BTreeSet<String>) -> bool {
    let Some(return_expr) = first_top_level_return_expr(body) else {
        return false;
    };
    let lowered = body.to_lowercase();
    let body_compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if execution_shaped_category(&task.category)
        && syntax_constrained_body(body)
        && visible_argument_contract_ok(task, body)
        && required_construct_contract_ok_for_task(task, body, hints)
        && execution_shape_library_contract_ok(task, body, hints)
        && category_body_transfer_bonus(&task.category, &lowered, &body_compact) >= 2.0
    {
        return false;
    }
    if decoder_return_shape(task) == "bool"
        && syntax_constrained_body(body)
        && visible_argument_contract_ok(task, body)
        && required_construct_contract_ok_for_task(task, body, hints)
        && semantic_family_contract_ok(task, body)
        && body_has_any(&lowered, &["for ", "while "])
        && body_has_any(&lowered, &["return true"])
        && body_has_any(&lowered, &["return false"])
    {
        return false;
    }
    if optional_requests_query_body_nontrivial(task, body, &lowered, hints) {
        return false;
    }
    if private_type_contract_body_nontrivial(task, body, &lowered, hints) {
        return false;
    }
    if broad_private_generated_decoder_contract(task)
        && task.category == "bpg_safe_head_default"
        && syntax_constrained_body(body)
        && visible_argument_contract_ok(task, body)
        && required_construct_contract_ok_for_task(task, body, hints)
        && body_has_any(&lowered, &["return data[0]", "return data [0]"])
        && lowered.contains("return other")
    {
        return false;
    }
    if private_residual_v3_decoder_contract(task)
        && task.category == "private_v3_safe_head_default"
        && syntax_constrained_body(body)
        && visible_argument_contract_ok(task, body)
        && required_construct_contract_ok_for_task(task, body, hints)
        && semantic_family_contract_ok(task, body)
        && body_semantically_admissible(task, body)
    {
        return false;
    }
    let compact = return_expr
        .to_lowercase()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let trivial_return = matches!(
        compact.as_str(),
        "data"
            | "other"
            | "args"
            | "extra"
            | "result"
            | "out"
            | "values"
            | "items"
            | "true"
            | "false"
            | "none"
            | "0"
            | "1"
            | "[]"
            | "{}"
            | "()"
            | "''"
            | "\"\""
            | "bool(data)"
            | "len(data)"
            | "list(data)"
            | "tuple(data)"
            | "dict(data)"
            | "str(data)"
            | "int(data)"
    ) || (compact == "returnvalue");
    if !trivial_return {
        return false;
    }
    let family = category_semantic_family(&task.category);
    let needs_real_body = hints.contains("loop")
        || hints.contains("branch")
        || hints.contains("locals")
        || hints.contains("frequency")
        || hints.contains("selection")
        || hints.contains("algorithmic_planning")
        || hints.contains("arithmetic_formula")
        || hints.contains("execution_shaped_program")
        || matches!(
            family,
            "collection_transform"
                | "string_transform"
                | "membership_lookup"
                | "scalar_recurrence"
                | "execution_shaped_program"
        );
    if !needs_real_body
        || (simple_return_category(&task.category)
            && !hints.contains("arithmetic_formula")
            && !hints.contains("algorithmic_planning"))
    {
        return false;
    }
    if compact == "result"
        && (lowered.contains("result = false")
            || lowered.contains("result=false")
            || lowered.contains("result = true")
            || lowered.contains("result=true")
            || lowered.contains("result = 0")
            || lowered.contains("result=0")
            || lowered.contains("result = []")
            || lowered.contains("result=[]"))
    {
        return true;
    }
    if returned_collection_builder_body_is_nontrivial(task, body, &compact) {
        return false;
    }
    true
}

pub(in crate::code_lm_closure) fn optional_dependency_import_contract_ok(body: &str) -> bool {
    unguarded_optional_dependency_imports(body).is_empty()
}

pub(in crate::code_lm_closure) fn optional_dependency_fallback_bonus(body: &str) -> f32 {
    let imports = optional_dependency_import_count(body) + optional_dependency_probe_count(body);
    if imports == 0 {
        return 0.0;
    }
    let lowered = body.to_ascii_lowercase();
    if !optional_dependency_import_contract_ok(body) {
        return -6.0;
    }
    let mut score = 0.5;
    if lowered.contains("except exception") || lowered.contains("except importerror") {
        score += 0.7;
    }
    if body_has_any(
        &lowered,
        &[
            " is none",
            " = none",
            "return []",
            "return {}",
            "return ''",
            "return 0",
            "return false",
            "list(",
            "dict(",
        ],
    ) {
        score += 0.5;
    }
    score
}

fn optional_dependency_import_count(body: &str) -> usize {
    body.lines()
        .filter(|line| !optional_dependency_modules_in_import_line(line).is_empty())
        .count()
}

fn optional_dependency_probe_count(body: &str) -> usize {
    let lowered = body.to_ascii_lowercase();
    optional_dependency_module_names()
        .iter()
        .filter(|module| {
            lowered.contains(&format!("find_spec('{module}')"))
                || lowered.contains(&format!("find_spec(\"{module}\")"))
        })
        .count()
}

fn unguarded_optional_dependency_imports(body: &str) -> Vec<String> {
    let lines = body.lines().collect::<Vec<_>>();
    let mut out = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        let modules = optional_dependency_modules_in_import_line(line);
        if modules.is_empty() || import_line_is_inside_try_with_handler(&lines, idx) {
            continue;
        }
        out.extend(modules.into_iter().map(str::to_string));
    }
    out
}

fn optional_dependency_modules_in_import_line(line: &str) -> Vec<&'static str> {
    let trimmed = line.trim();
    let mut out = Vec::new();
    if let Some(rest) = trimmed.strip_prefix("import ") {
        for raw in rest.split(',') {
            let base = raw
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("")
                .split('.')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();
            if let Some(module) = optional_dependency_module_name(&base) {
                out.push(module);
            }
        }
    } else if let Some(rest) = trimmed.strip_prefix("from ") {
        if let Some((module, _names)) = rest.split_once(" import ") {
            let base = module
                .trim()
                .split('.')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();
            if let Some(module) = optional_dependency_module_name(&base) {
                out.push(module);
            }
        }
    }
    out
}

fn optional_dependency_module_name(module: &str) -> Option<&'static str> {
    optional_dependency_module_names()
        .iter()
        .copied()
        .find(|candidate| *candidate == module)
        .or(match module {
            "bs4" | "beautifulsoup4" => Some("beautifulsoup4"),
            _ => None,
        })
}

fn optional_dependency_module_names() -> [&'static str; 10] {
    [
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "seaborn",
        "matplotlib",
        "requests",
        "nltk",
        "wordcloud",
        "psutil",
    ]
}

fn import_line_is_inside_try_with_handler(lines: &[&str], import_idx: usize) -> bool {
    let import_indent = leading_space_count(lines[import_idx]);
    let mut cursor = import_idx;
    while cursor > 0 {
        cursor -= 1;
        let trimmed = lines[cursor].trim();
        if trimmed.is_empty() {
            continue;
        }
        let indent = leading_space_count(lines[cursor]);
        if indent >= import_indent {
            continue;
        }
        if trimmed == "try:" && block_has_except_handler(lines, cursor, indent) {
            return true;
        }
        break;
    }
    false
}

fn block_has_except_handler(lines: &[&str], try_idx: usize, try_indent: usize) -> bool {
    for line in lines.iter().skip(try_idx + 1) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let indent = leading_space_count(line);
        if indent < try_indent {
            return false;
        }
        if indent == try_indent && trimmed.starts_with("except ") {
            return true;
        }
    }
    false
}

fn leading_space_count(line: &str) -> usize {
    line.chars().take_while(|ch| *ch == ' ').count()
}

fn returned_collection_builder_body_is_nontrivial(
    task: &CodeTask,
    body: &str,
    return_compact: &str,
) -> bool {
    if !matches!(return_compact, "out" | "result" | "values" | "items") {
        return false;
    }
    let lowered = body.to_lowercase();
    let returned_name = return_compact;
    let append_prefix = format!("{returned_name}.append(");
    let append_spaced_prefix = format!("{returned_name}.append (");
    let extend_prefix = format!("{returned_name}.extend(");
    let extend_spaced_prefix = format!("{returned_name}.extend (");
    let item_assign_prefix = format!("{returned_name}[");
    let has_builder_mutation = lowered.lines().map(str::trim).any(|line| {
        ((line.starts_with(&append_prefix) || line.starts_with(&append_spaced_prefix))
            && !line.ends_with(".append(")
            && !line.ends_with(".append ("))
            || ((line.starts_with(&extend_prefix) || line.starts_with(&extend_spaced_prefix))
                && !line.ends_with(".extend(")
                && !line.ends_with(".extend ("))
            || (line.starts_with(&item_assign_prefix) && line.contains('='))
    });
    if !has_builder_mutation {
        return false;
    }
    let has_loop = lowered.contains("for ") || lowered.contains("while ");
    let visible_args = visible_signature_arg_names(task);
    let uses_visible_arg = if visible_args.is_empty() {
        ["data", "other", "args", "extra"]
            .iter()
            .any(|arg| body_mentions_token(body, arg))
    } else {
        visible_args
            .iter()
            .any(|arg| body_mentions_token(body, arg))
    };
    has_loop && uses_visible_arg
}

pub(in crate::code_lm_closure) fn candidate_floor_v2_wall_body(
    task: &CodeTask,
    body: &str,
) -> bool {
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let hints = decoder_required_constructs(task);
    let Some(return_expr) =
        first_top_level_return_expr(body).or_else(|| first_return_expr_any_indent(body))
    else {
        return true;
    };
    let return_compact = return_expr
        .to_lowercase()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if (broad_private_generated_decoder_contract(task)
        || private_public_transfer_decoder_contract(task)
        || private_residual_v3_decoder_contract(task))
        && syntax_constrained_body(body)
        && visible_argument_contract_ok(task, body)
        && return_shape_contract_ok(task, &lowered)
        && required_construct_contract_ok_for_task(task, body, &hints)
        && semantic_family_contract_ok(task, body)
        && body_semantically_admissible(task, body)
    {
        return false;
    }

    if hints.contains("two_arg_interface")
        || category_expected_arg_count(task).is_some_and(|count| count >= 2)
    {
        let visible_args = visible_signature_arg_names(task);
        let uses_second_visible = visible_args
            .iter()
            .nth(1)
            .is_some_and(|name| body_mentions_token(body, name));
        if !uses_second_visible && !body_mentions_token(body, "other") {
            return true;
        }
    }

    if hints.contains("nested_structure") && !nested_structure_body_ok(body) {
        return true;
    }
    if private_type_contract_body_nontrivial(task, body, &lowered, &hints) {
        return false;
    }
    if optional_requests_query_body_nontrivial(task, body, &lowered, &hints) {
        return false;
    }

    if transform_or_collection_body_required(task, &hints) {
        if matches!(
            return_compact.as_str(),
            "data"
                | "text"
                | "items"
                | "list(data)"
                | "tuple(data)"
                | "str(data)"
                | "bool(data)"
                | "len(data)"
                | "[]"
                | "''"
                | "\"\""
        ) {
            return true;
        }
        let has_builder = body_has_any(
            &lowered,
            &[
                "append(",
                "extend(",
                "join(",
                "replace(",
                "split(",
                "strip(",
                "lower(",
                "sorted(",
                "set(",
                "zip(",
                "enumerate(",
                "read_csv",
                ".apply(",
                "pairplot",
                "df[",
                "range(",
                "while ",
                "for ",
            ],
        );
        if !has_builder {
            return true;
        }
    }

    if hints.contains("arithmetic_formula")
        && !body_has_any(&lowered, &["+", "-", "*", "/", "%", "**", "pow(", "math."])
    {
        return true;
    }
    if execution_shaped_semantic_return_wall(task, body, &return_compact) {
        return true;
    }
    if bogus_return_attribute_body(body) {
        return true;
    }
    if bogus_return_local_callable_body(body) {
        return true;
    }

    compact.contains("returnresult")
        && body_has_any(
            &lowered,
            &["result = false", "result=false", "result = []", "result=[]"],
        )
        && transform_or_collection_body_required(task, &hints)
}

fn first_return_expr_any_indent(body: &str) -> Option<String> {
    for raw_line in body.lines() {
        let line = raw_line.trim();
        if let Some(expr) = line.strip_prefix("return ") {
            return Some(expr.trim().to_string());
        }
    }
    None
}

fn private_type_contract_body_nontrivial(
    task: &CodeTask,
    body: &str,
    lowered: &str,
    hints: &BTreeSet<String>,
) -> bool {
    let text = format!(
        "{} {} {} {}",
        task.category,
        task.prompt,
        task.tags.join(" "),
        task.raw
            .get("residual_concept")
            .and_then(Value::as_str)
            .unwrap_or("")
    )
    .to_ascii_lowercase();
    if !(hints.contains("type_checks")
        || hints.contains("nested_structure")
        || text.contains("type_contract_v2")
        || text.contains("type_semantic_transfer")
        || text.contains("type_and_return_shape"))
    {
        return false;
    }
    visible_argument_contract_ok(task, body)
        && return_shape_contract_ok(task, lowered)
        && required_construct_contract_ok_for_task(task, body, hints)
        && body_has_any(
            lowered,
            &[
                "isinstance",
                "str(",
                ".strip",
                ".lower",
                ".replace",
                ".split",
                ".get(",
                "stack",
                "while ",
                "for ",
            ],
        )
}

fn optional_requests_query_body_nontrivial(
    task: &CodeTask,
    body: &str,
    lowered: &str,
    hints: &BTreeSet<String>,
) -> bool {
    let text = task_contract_text(task);
    let optional_query_contract = task.category.contains("optional_requests_query")
        || text.contains("optional_requests_query")
        || (text.contains("query parameters")
            && (text.contains("requests") || text.contains("url")));
    optional_query_contract
        && decoder_return_shape(task) == "dict"
        && syntax_constrained_body(body)
        && visible_argument_contract_ok(task, body)
        && return_shape_contract_ok(task, lowered)
        && required_construct_contract_ok_for_task(task, body, hints)
        && optional_dependency_import_contract_ok(body)
        && lowered.contains("parse_qs(")
        && lowered.contains("urlparse(")
        && lowered.contains("for ")
        && lowered.contains("out[")
        && lowered.contains("return out")
        && !body_has_any(
            lowered,
            &[
                "requests.get(",
                "requests.post(",
                "requests.request(",
                "urlopen(",
            ],
        )
}
