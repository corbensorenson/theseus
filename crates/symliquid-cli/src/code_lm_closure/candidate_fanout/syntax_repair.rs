use super::*;

pub(in crate::code_lm_closure) fn canonicalize_task_candidate_body_aliases(
    task: &CodeTask,
    body: &str,
) -> String {
    let entry = sanitize_ident(&task.entry_point);
    let signature = visible_signature(&entry, &task.prompt);
    canonicalize_candidate_body_aliases(body, &signature)
}

pub(in crate::code_lm_closure) fn normalize_generated_body(body: &str) -> String {
    let trimmed = trim_after_first_top_level_return(body);
    repair_generated_token_syntax(&trimmed).trim().to_string()
}

pub(in crate::code_lm_closure) fn repair_generated_token_syntax(body: &str) -> String {
    body.lines()
        .flat_map(repair_generated_line_syntax)
        .collect::<Vec<_>>()
        .join("\n")
}

pub(in crate::code_lm_closure) fn repair_generated_line_syntax(line: &str) -> Vec<String> {
    if let Some(repaired) = repair_state_assignment_glued_append(line) {
        return repaired;
    }
    let repaired =
        repair_malformed_call_tail_tokens(&repair_f_string_spacing(&repair_lambda_as_syntax(line)));
    let trimmed = repaired.trim();
    if trimmed.starts_with("with ") && !trimmed.contains(':') {
        if let Some((header, tail)) = split_line_before_keyword(&repaired, " return ") {
            return vec![
                format!("{}:", header.trim_end()),
                format!("    return {}", tail.trim()),
            ];
        }
    }
    if trimmed.starts_with("try ") && !trimmed.starts_with("try:") {
        let tail = trimmed.trim_start_matches("try").trim();
        if !tail.is_empty() {
            return vec!["try:".to_string(), format!("    {tail}")];
        }
    }
    vec![repaired]
}

fn repair_state_assignment_glued_append(line: &str) -> Option<Vec<String>> {
    let indent_len = line.chars().take_while(|ch| *ch == ' ').count();
    let indent = " ".repeat(indent_len);
    let trimmed = line.trim_start();
    let (lhs, rhs) = trimmed.split_once(" = ")?;
    if !matches!(
        lhs.trim(),
        "balance" | "state" | "current" | "total" | "best"
    ) {
        return None;
    }
    let append_idx = rhs.find(".out.append(")?;
    let rhs_head = rhs[..append_idx].trim();
    if rhs_head.is_empty()
        || !rhs_head
            .chars()
            .all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        return None;
    }
    let append_tail = &rhs[append_idx + 1..];
    let append_indent = if indent_len >= 4 {
        " ".repeat(indent_len - 4)
    } else {
        indent.clone()
    };
    Some(vec![
        format!("{indent}{} = {rhs_head}", lhs.trim()),
        format!("{append_indent}{append_tail}"),
    ])
}

pub(in crate::code_lm_closure) fn repair_malformed_call_tail_tokens(line: &str) -> String {
    line.replace(" if)", ")")
        .replace(" if,", ",")
        .replace(" if]", "]")
        .replace(" if}", "}")
}

pub(in crate::code_lm_closure) fn split_line_before_keyword(
    line: &str,
    keyword: &str,
) -> Option<(String, String)> {
    let idx = line.find(keyword)?;
    Some((
        line[..idx].to_string(),
        line[idx + keyword.len()..].to_string(),
    ))
}

pub(in crate::code_lm_closure) fn repair_f_string_spacing(line: &str) -> String {
    let mut out = line.replace("f '", "f'");
    out = out.replace("f \"", "f\"");
    out
}

pub(in crate::code_lm_closure) fn repair_lambda_as_syntax(line: &str) -> String {
    let Some(lambda_pos) = line.find("lambda ") else {
        return line.to_string();
    };
    let after_lambda = &line[lambda_pos + "lambda ".len()..];
    let delimiter = after_lambda
        .find(" as ")
        .map(|idx| (idx, " as ".len()))
        .or_else(|| after_lambda.find("] ").map(|idx| (idx, "] ".len())))
        .or_else(|| after_lambda.find(") ").map(|idx| (idx, ") ".len())));
    let Some((delimiter_pos, delimiter_len)) = delimiter else {
        return line.to_string();
    };
    let name = after_lambda[..delimiter_pos].trim();
    if !is_identifier(name) {
        return line.to_string();
    }
    let before = &line[..lambda_pos + "lambda ".len()];
    let after = &after_lambda[delimiter_pos + delimiter_len..];
    format!("{before}{name}: {after}")
}

pub(in crate::code_lm_closure) fn trim_after_first_top_level_return(body: &str) -> String {
    let mut out = Vec::new();
    for raw_line in body.lines() {
        if raw_line.trim().is_empty() {
            continue;
        }
        out.push(raw_line.trim_end().to_string());
        let indent = raw_line.chars().take_while(|ch| *ch == ' ').count();
        if indent == 0 && raw_line.trim_start().starts_with("return ") {
            break;
        }
    }
    out.join("\n")
}
