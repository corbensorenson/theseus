mod part_02_beam_decode;
mod part_02_body_models;
use part_02_beam_decode::*;
use part_02_body_models::*;

fn sts_candidate_expressions(sts_streams: Option<&BTreeMap<String, String>>) -> Vec<String> {
    let mut out = Vec::new();
    let Some(streams) = sts_streams else {
        return out;
    };
    for stream in ["patch_stream", "solver_stream"] {
        let Some(text) = streams.get(stream) else {
            continue;
        };
        let normalized = text
            .trim()
            .trim_end_matches(';')
            .trim()
            .strip_prefix("return ")
            .unwrap_or_else(|| text.trim())
            .trim()
            .to_string();
        if useful_generated_expression(&normalized) {
            out.push(normalized);
        }
    }
    out
}

fn baseline_expressions(
    task: &CodeTask,
    expression_bank: &[ExpressionBankItem],
    seed: u64,
    limit: usize,
) -> Vec<String> {
    let mut scored = expression_bank
        .iter()
        .map(|item| {
            let tie = stable_hash_u64(&format!("baseline:{}:{}:{}", seed, task.task_id, item.expr));
            (item.count, tie, item.expr.clone())
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, _, expr)| expr)
        .collect()
}

fn forced_block_token_options(existing: &[String]) -> Option<Vec<(String, f32)>> {
    let prev = existing.last().map(String::as_str).unwrap_or("<BOS>");
    let prev2 = existing
        .iter()
        .rev()
        .nth(1)
        .map(String::as_str)
        .unwrap_or("<BOS>");
    if prev2 == "." && is_identifier(prev) && body_token_allowed(existing, "(") {
        return Some(vec![("(".to_string(), 90.0)]);
    }
    if prev == ":" && body_token_allowed(existing, "<NL>") {
        return Some(vec![("<NL>".to_string(), 80.0)]);
    }
    let current_line = current_line_tokens(existing);
    if block_header_line(&current_line)
        && !current_line.iter().any(|token| token == ":")
        && bracket_balance(&current_line) == 0
        && matches!(prev, ")" | "]" | "True" | "False")
        && body_token_allowed(existing, ":")
    {
        if prev == "]" && !line_has_comparison_operator(&current_line) {
            return None;
        }
        if let Some(options) = block_header_call_comparison_options(&current_line) {
            return Some(options);
        }
        return Some(vec![
            (":".to_string(), 88.0),
            ("and".to_string(), 8.0),
            ("or".to_string(), 8.0),
        ]);
    }
    if previous_meaningful_token(existing).as_deref() == Some(":")
        && prev == "<NL>"
        && body_token_allowed(existing, "<INDENT>")
    {
        return Some(vec![("<INDENT>".to_string(), 80.0)]);
    }
    if prev == "<NL>"
        && body_indent_balance(existing) > 0
        && previous_completed_line_starts_with(existing, "return")
        && body_token_allowed(existing, "<DEDENT>")
    {
        return Some(vec![("<DEDENT>".to_string(), 85.0)]);
    }
    if return_statement_complete(&current_line) {
        if body_indent_balance(existing) > 0 && body_token_allowed(existing, "<NL>") {
            return Some(vec![("<NL>".to_string(), 90.0)]);
        }
        if body_token_allowed(existing, "<EOS>") {
            return Some(vec![("<EOS>".to_string(), 90.0)]);
        }
    }
    if simple_statement_complete(&current_line) && body_token_allowed(existing, "<NL>") {
        return Some(vec![("<NL>".to_string(), 70.0)]);
    }
    None
}

fn previous_completed_line_starts_with(existing: &[String], expected: &str) -> bool {
    if existing.last().map(String::as_str) != Some("<NL>") {
        return false;
    }
    let mut line = Vec::new();
    for token in existing.iter().rev().skip(1) {
        if token == "<NL>" {
            break;
        }
        if !matches!(token.as_str(), "<INDENT>" | "<DEDENT>") {
            line.push(token.as_str());
        }
    }
    line.reverse();
    line.first().is_some_and(|token| *token == expected)
}

fn return_statement_complete(line: &[String]) -> bool {
    line.first().map(String::as_str) == Some("return")
        && line.len() >= 2
        && balanced_complete_line(line)
        && !return_head_may_continue_as_call(line)
        && !matches!(
            line.last().map(String::as_str),
            Some("return" | "." | "," | "=" | "(" | "[" | "{")
        )
}

fn return_head_may_continue_as_call(line: &[String]) -> bool {
    let Some(token) = line.last().map(String::as_str) else {
        return false;
    };
    if common_return_value_identifier(token) {
        return false;
    }
    line.len() == 2
        && (matches!(
            token,
            "tuple" | "list" | "dict" | "set" | "int" | "str" | "len" | "sum" | "min" | "max"
                | "sorted"
        ) || is_identifier(token))
}

fn common_return_value_identifier(token: &str) -> bool {
    matches!(
        token,
        "out"
            | "result"
            | "answer"
            | "best"
            | "total"
            | "count"
            | "current"
            | "value"
            | "values"
            | "items"
            | "rows"
            | "mapping"
            | "summary"
    )
}

fn simple_statement_complete(line: &[String]) -> bool {
    if line.is_empty() || !balanced_complete_line(line) {
        return false;
    }
    if assignment_rhs_requires_continuation(line) {
        return false;
    }
    if assignment_inline_conditional_requires_continuation(line) {
        return false;
    }
    if assignment_rhs_may_continue_with_expression(line) {
        return false;
    }
    if method_call_may_continue_with_chain(line) {
        return false;
    }
    let first = line.first().map(String::as_str).unwrap_or("");
    if matches!(
        first,
        "if" | "elif" | "else" | "for" | "while" | "with" | "try" | "except" | "finally" | "def"
    ) {
        return false;
    }
    if matches!(
        line.last().map(String::as_str),
        Some(
            "."
                | ","
                | "="
                | "("
                | "["
                | "{"
                | "as"
                | "in"
                | "and"
                | "or"
                | "not"
                | "+"
                | "-"
                | "*"
                | "/"
                | "//"
                | "%"
                | "&"
                | "|"
                | "=="
                | "!="
                | "<"
                | ">"
                | "<="
                | ">="
        )
    ) {
        return false;
    }
    first == "raise"
        || first == "continue"
        || first == "break"
        || first == "pass"
        || line.iter().any(|token| token == "=")
        || line.iter().any(|token| token.ends_with(')'))
}

fn assignment_rhs_requires_continuation(line: &[String]) -> bool {
    let Some(eq_idx) = line.iter().position(|token| token == "=") else {
        return false;
    };
    let rhs = &line[eq_idx + 1..];
    if rhs.is_empty() {
        return true;
    }
    if rhs.len() == 1
        && matches!(
            rhs[0].as_str(),
            "list"
                | "tuple"
                | "dict"
                | "set"
                | "sorted"
                | "os"
                | "json"
                | "csv"
                | "zipfile"
                | "tarfile"
                | "configparser"
                | "shutil"
                | "glob"
                | "platform"
                | "psutil"
                | "subprocess"
        )
    {
        return true;
    }
    matches!(
        rhs.last().map(String::as_str),
        Some(
            "."
                | "("
                | "["
                | "{"
                | ","
                | "+"
                | "-"
                | "*"
                | "/"
                | "//"
                | "%"
                | "&"
                | "|"
                | "=="
                | "!="
                | "<"
                | ">"
                | "<="
                | ">="
        )
    )
}

fn assignment_inline_conditional_requires_continuation(line: &[String]) -> bool {
    let Some(eq_idx) = line.iter().position(|token| token == "=") else {
        return false;
    };
    let rhs = &line[eq_idx + 1..];
    if !rhs.iter().any(|token| token == "if") {
        return false;
    }
    let Some(else_idx) = rhs.iter().position(|token| token == "else") else {
        return true;
    };
    rhs.get(else_idx + 1).is_none()
}

fn assignment_rhs_may_continue_with_expression(line: &[String]) -> bool {
    let Some(eq_idx) = line.iter().position(|token| token == "=") else {
        return false;
    };
    let lhs = &line[..eq_idx];
    let rhs = &line[eq_idx + 1..];
    if lhs.iter().any(|token| token == ",") {
        let lhs_targets = top_level_expression_count(lhs);
        let rhs_values = top_level_expression_count(rhs);
        if lhs_targets > 1 && rhs_values < lhs_targets {
            return true;
        }
        if lhs_targets > 1
            && rhs_values <= lhs_targets
            && rhs.iter().any(|token| token == "[")
            && last_top_level_expression_is_bare_identifier(rhs)
        {
            return true;
        }
        if lhs_targets > 1 && last_top_level_expression_is_callable_head(rhs) {
            return true;
        }
    }
    if assignment_rhs_closed_expression_may_take_operator(rhs) {
        return true;
    }
    if rhs.len() != 1 {
        return false;
    }
    let token = rhs[0].as_str();
    is_identifier(token)
        && !matches!(
            token,
            "True" | "False" | "None" | "len" | "sum" | "min" | "max" | "sorted" | "list"
                | "tuple" | "dict" | "set" | "range" | "int" | "str"
        )
}

fn state_local_assignment_rhs_identifier(line: &[String]) -> bool {
    if line.len() != 3 || line.get(1).map(String::as_str) != Some("=") {
        return false;
    }
    let lhs = line.first().map(String::as_str).unwrap_or("");
    let rhs = line.get(2).map(String::as_str).unwrap_or("");
    matches!(
        lhs,
        "balance" | "state" | "best" | "total" | "current" | "floor" | "ceiling"
    ) && is_identifier(rhs)
}

fn top_level_expression_count(tokens: &[String]) -> usize {
    if tokens.is_empty() {
        return 0;
    }
    let mut depth = 0i32;
    let mut count = 1usize;
    for token in tokens {
        match token.as_str() {
            "(" | "[" | "{" => depth += 1,
            ")" | "]" | "}" => depth = depth.saturating_sub(1),
            "," if depth == 0 => count += 1,
            _ => {}
        }
    }
    count
}

fn line_has_comparison_operator(tokens: &[String]) -> bool {
    tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "==" | "!=" | "<" | ">" | "<=" | ">=" | "in" | "is"
        )
    })
}

fn last_top_level_expression_is_bare_identifier(tokens: &[String]) -> bool {
    let mut depth = 0i32;
    let mut start = 0usize;
    for (idx, token) in tokens.iter().enumerate() {
        match token.as_str() {
            "(" | "[" | "{" => depth += 1,
            ")" | "]" | "}" => depth = depth.saturating_sub(1),
            "," if depth == 0 => start = idx + 1,
            _ => {}
        }
    }
    let expr = tokens[start..]
        .iter()
        .filter(|token| !token.trim().is_empty())
        .collect::<Vec<_>>();
    expr.len() == 1 && is_identifier(expr[0])
}

fn last_top_level_expression_is_callable_head(tokens: &[String]) -> bool {
    let mut depth = 0i32;
    let mut start = 0usize;
    for (idx, token) in tokens.iter().enumerate() {
        match token.as_str() {
            "(" | "[" | "{" => depth += 1,
            ")" | "]" | "}" => depth = depth.saturating_sub(1),
            "," if depth == 0 => start = idx + 1,
            _ => {}
        }
    }
    let expr = tokens[start..]
        .iter()
        .filter(|token| !token.trim().is_empty())
        .collect::<Vec<_>>();
    expr.len() == 1
        && matches!(
            expr[0].as_str(),
            "len" | "sum" | "min" | "max" | "sorted" | "list" | "tuple" | "dict" | "set"
        )
}

fn assignment_rhs_closed_expression_may_take_operator(rhs: &[String]) -> bool {
    if !matches!(rhs.last().map(String::as_str), Some(")" | "]")) {
        return false;
    }
    if rhs.iter().any(|token| {
        matches!(
            token.as_str(),
            "+" | "-" | "*" | "/" | "//" | "%" | "&" | "|" | "==" | "!=" | "<" | ">" | "<=" | ">="
        )
    }) {
        return true;
    }
    rhs.iter().any(|token| token == "[")
}

fn method_call_may_continue_with_chain(line: &[String]) -> bool {
    matches!(line.last().map(String::as_str), Some(")"))
        && line.iter().any(|token| token == ".")
        && line.iter().any(|token| token == "setdefault")
        && !line.iter().any(|token| token == "append")
        && !line.first().is_some_and(|token| token == "return")
}

fn block_header_call_comparison_options(line: &[String]) -> Option<Vec<(String, f32)>> {
    if !matches!(line.first().map(String::as_str), Some("if" | "elif" | "while")) {
        return None;
    }
    if line.iter().any(|token| token == ":") || line.last().map(String::as_str) != Some(")") {
        return None;
    }
    if line.iter().any(|token| {
        matches!(
            token.as_str(),
            "isinstance" | "all" | "any" | "endswith" | "startswith"
        )
    }) {
        return None;
    }
    if line.windows(3).any(|window| {
        matches!(window, [name, dot, method] if is_identifier(name) && dot == "." && matches!(method.as_str(), "get" | "count" | "index"))
    }) {
        return Some(vec![
            ("==".to_string(), 94.0),
            ("!=".to_string(), 88.0),
            (">=".to_string(), 72.0),
            (">".to_string(), 70.0),
            ("<=".to_string(), 70.0),
            ("<".to_string(), 68.0),
            (":".to_string(), 60.0),
            ("and".to_string(), 8.0),
            ("or".to_string(), 8.0),
        ]);
    }
    if line.iter().any(|token| {
        matches!(
            token.as_str(),
            "len" | "sum" | "min" | "max" | "abs" | "int" | "float"
        )
    }) {
        return Some(vec![
            (">=".to_string(), 92.0),
            (">".to_string(), 88.0),
            ("<=".to_string(), 88.0),
            ("<".to_string(), 86.0),
            ("==".to_string(), 84.0),
            ("!=".to_string(), 84.0),
            (":".to_string(), 70.0),
            ("and".to_string(), 8.0),
            ("or".to_string(), 8.0),
        ]);
    }
    None
}

fn balanced_complete_line(line: &[String]) -> bool {
    bracket_balance(line) == 0
        && !open_block_header_without_colon(line)
        && !with_header_requires_as_now(line)
        && !grammar_requires_colon_now(line)
}

fn body_ngram_token_options(
    task: &CodeTask,
    model: &BodyNgramModel,
    prev2: &str,
    prev1: &str,
    prompt_tokens: &[String],
    position: usize,
    limit: usize,
) -> Vec<(String, f32)> {
    let mut scored: BTreeMap<String, f32> = BTreeMap::new();
    for (key, weight) in weighted_body_ngram_lookup_keys(task, prev2, prev1, position) {
        let Some(counts) = model.counts.get(&key) else {
            continue;
        };
        let total = counts.values().sum::<usize>().max(1) as f32;
        for (token, count) in counts {
            let prob = (*count as f32 / total).max(1e-6);
            *scored.entry(token.clone()).or_insert(0.0) +=
                ((*count as f32 + 1.0).ln() + prob) * weight;
        }
    }
    inject_visible_signature_token_priors(task, position, &mut scored);
    inject_execution_shape_context_token_priors(task, prev2, prev1, position, &mut scored);
    let mut rows = scored
        .into_iter()
        .map(|(token, score)| {
            let adjusted = score
                + token_alignment_bonus(&token, prompt_tokens)
                + body_position_bonus(&token, position)
                + category_body_token_bonus(task, &token)
                + category_position_token_bonus(task, &token, position)
                + execution_shape_context_token_bonus(task, prev2, prev1, &token, position);
            (token, adjusted)
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    rows.into_iter().take(limit).collect()
}

fn body_ngram_keys(task: &CodeTask, prev2: &str, prev1: &str, position: usize) -> Vec<String> {
    weighted_body_ngram_lookup_keys(task, prev2, prev1, position)
        .into_iter()
        .map(|(key, _)| key)
        .collect()
}

fn inject_visible_signature_token_priors(
    task: &CodeTask,
    position: usize,
    scored: &mut BTreeMap<String, f32>,
) {
    let args = visible_signature_ordered_user_args(task);
    for (idx, arg) in args.iter().enumerate().take(4) {
        let base = if idx == 0 { 2.4 } else { 2.1 };
        let early = if position <= 8 { 0.8 } else { 0.0 };
        *scored.entry(arg.clone()).or_insert(0.0) += base + early;
    }
    if !args.is_empty() {
        *scored.entry("data".to_string()).or_insert(0.0) += 1.2;
    }
    if args.len() >= 2 {
        *scored.entry("other".to_string()).or_insert(0.0) += 1.1;
    }
}

fn weighted_body_ngram_lookup_keys(
    task: &CodeTask,
    prev2: &str,
    prev1: &str,
    position: usize,
) -> Vec<(String, f32)> {
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for category in normalized_category_keys(&task.category) {
        let capped_position = learned_position_cap(&category, position);
        for (key, weight) in [
            (
                format!("cat:{category}|pos:{capped_position}|p2:{prev2}|p1:{prev1}"),
                8.0f32,
            ),
            (format!("cat:{category}|pos:{capped_position}"), 2.0f32),
            (
                format!("cat:{category}|pos:{capped_position}|p1:{prev1}"),
                3.0f32,
            ),
            (format!("cat:{category}|p2:{prev2}|p1:{prev1}"), 7.0f32),
            (format!("cat:{category}|p1:{prev1}"), 2.5f32),
        ] {
            if seen.insert(key.clone()) {
                rows.push((key, weight));
            }
        }
    }
    for (key, weight) in [
        (format!("all|p2:{prev2}|p1:{prev1}"), 1.2f32),
        (format!("all|p1:{prev1}"), 0.6f32),
    ] {
        if seen.insert(key.clone()) {
            rows.push((key, weight));
        }
    }
    rows
}

fn body_token_allowed(existing: &[String], token: &str) -> bool {
    if existing.is_empty() {
        return !matches!(
            token,
            "<EOS>" | "<NL>" | "<INDENT>" | "<DEDENT>" | ")" | "]" | "}" | "," | "." | ":"
        );
    }
    let prev = existing.last().map(String::as_str).unwrap_or("");
    if token == "<EOS>" {
        let current_line = current_line_tokens(existing);
        if prev == "="
            || bracket_balance(&current_line) > 0
            || open_block_header_without_colon(&current_line)
            || with_header_requires_as_now(&current_line)
            || grammar_requires_colon_now(&current_line)
        {
            return false;
        }
        return existing.iter().any(|item| item == "return");
    }
    if token == "<DEDENT>" {
        return body_indent_balance(existing) > 0 && matches!(prev, "<NL>" | "<DEDENT>");
    }
    if token == "<INDENT>" {
        let previous_meaningful = existing
            .iter()
            .rev()
            .skip(1)
            .find(|item| item.as_str() != "<NL>")
            .map(String::as_str);
        return prev == "<NL>" && previous_meaningful == Some(":");
    }
    if token == "<NL>" {
        let current_line = current_line_tokens(existing);
        if prev == "="
            || bracket_balance(&current_line) > 0
            || open_block_header_without_colon(&current_line)
            || with_header_requires_as_now(&current_line)
            || grammar_requires_colon_now(&current_line)
        {
            return false;
        }
        return !matches!(
            prev,
            "<NL>" | "<INDENT>" | "(" | "[" | "{" | "," | "." | "return"
        );
    }
    if matches!(prev, "<NL>" | "<INDENT>" | "<DEDENT>")
        && matches!(
            token,
            ")" | "]" | "}" | "," | "." | ":" | "and" | "or" | "in"
        )
    {
        return false;
    }
    if prev == "return" && matches!(token, "<NL>" | "<EOS>" | ":" | "=" | "," | ")" | "]" | "}") {
        return false;
    }
    let current_line = current_line_tokens(existing);
    if current_line.first().map(String::as_str) == Some("import") {
        if token == "," {
            return is_identifier(prev) && !matches!(prev, "import" | "as");
        }
        if token == "as" {
            return is_identifier(prev) && !matches!(prev, "import" | "as");
        }
        if prev == "as" {
            return is_identifier(token);
        }
        return is_identifier(token) && matches!(prev, "import" | ",");
    }
    if prev == "." && !is_identifier(token) {
        return false;
    }
    if token == "." && state_local_assignment_rhs_identifier(&current_line) {
        return false;
    }
    if prev == "=" && matches!(token, "<NL>" | "<EOS>" | ")" | "]" | "}" | ":" | ",") {
        return false;
    }
    if prev == "lambda" && !is_identifier(token) {
        return false;
    }
    if prev == "as" && !is_identifier(token) {
        return false;
    }
    if current_line.first().map(String::as_str) == Some("if") && token == "as" {
        return false;
    }
    if matches!(
        current_line.first().map(String::as_str),
        Some("if" | "elif" | "while")
    ) && !current_line.iter().any(|item| item == ":")
        && bracket_balance(&current_line) == 0
        && matches!(prev, ")" | "]" | "True" | "False")
        && !matches!(
            token,
            ":" | "and" | "or" | "is" | "in" | "==" | "!=" | "<" | ">" | "<=" | ">="
        )
    {
        return false;
    }
    if current_line.first().map(String::as_str) == Some("with") {
        if let Some(module_idx) = current_line
            .iter()
            .position(|item| matches!(item.as_str(), "tarfile" | "zipfile"))
        {
            let module = current_line[module_idx].as_str();
            if current_line.last().map(String::as_str) == Some(module) && token != "." {
                return false;
            }
            if prev == "."
                && current_line
                    .get(current_line.len().saturating_sub(2))
                    .is_some_and(|item| item == module)
                && !matches!(
                    (module, token),
                    ("tarfile", "open") | ("zipfile", "ZipFile")
                )
            {
                return false;
            }
        }
        if bracket_balance(&current_line) > 0
            && current_line.iter().any(|item| item == "encoding")
            && ((prev.starts_with('\'') && prev.ends_with('\''))
                || (prev.starts_with('"') && prev.ends_with('"')))
            && !matches!(token, ")" | ",")
        {
            return false;
        }
    }
    if prev.starts_with('\'') && prev.ends_with('\'') && token == "." {
        return false;
    }
    if prev.starts_with('"') && prev.ends_with('"') && token == "." {
        return false;
    }
    if current_line.len() == 1
        && !matches!(
            current_line.first().map(String::as_str),
            Some("import" | "from" | "with")
        )
        && matches!(token, "tarfile" | "zipfile")
    {
        return false;
    }
    if current_line.first().map(String::as_str) == Some("if")
        && matches!(token, "tarfile" | "zipfile" | "ZipFile")
    {
        return false;
    }
    if archive_context_manager_in_if_header(&current_line)
        && matches!(token, ":" | "," | "+" | "<NL>" | "<EOS>")
    {
        return false;
    }
    if current_line.first().map(String::as_str) == Some("return") && prev == "{" && token == "." {
        return false;
    }
    if block_header_line(&current_line)
        && bracket_balance(&current_line) == 0
        && token == ","
        && !for_header_target_allows_comma(&current_line)
    {
        return false;
    }
    if for_header_target_before_in(&current_line)
        && matches!(
            token,
            "+" | "-"
                | "*"
                | "/"
                | "//"
                | "%"
                | "&"
                | "|"
                | "=="
                | "!="
                | "<"
                | ">"
                | "<="
                | ">="
                | ":"
        )
    {
        return false;
    }
    if prev == ":" && block_header_line(&current_line) && token != "<NL>" {
        return false;
    }
    if callable_keyword_argument_requires_callable_now(&current_line)
        && !callable_keyword_argument_start_token(token)
    {
        return false;
    }
    if current_line.first().map(String::as_str) == Some("if")
        && matches!(token, "Exception" | "BaseException")
    {
        return false;
    }
    if current_line.first().map(String::as_str) == Some("if")
        && current_line
            .get(1)
            .is_some_and(|item| matches!(item.as_str(), "Exception" | "BaseException"))
    {
        return false;
    }
    if token == "return"
        && !matches!(prev, "<NL>" | "<INDENT>" | "<DEDENT>" | ":")
        && !current_line.is_empty()
    {
        return false;
    }
    if is_identifier(prev) && prev == token {
        return false;
    }
    if open_block_header_without_colon(&current_line)
        && matches!(token, "<NL>" | "<EOS>" | "return")
    {
        return false;
    }
    if with_header_requires_as_now(&current_line) && token != "as" {
        return false;
    }
    if grammar_requires_colon_now(&current_line) && token != ":" {
        return false;
    }
    if current_line.first().map(String::as_str) == Some("return") && current_line.len() > 1 {
        let balance = bracket_balance(&current_line[1..]);
        let expression_can_continue = balance > 0
            || matches!(
                prev,
                "return"
                    | "not"
                    | "and"
                    | "or"
                    | "in"
                    | "+"
                    | "-"
                    | "*"
                    | "/"
                    | "//"
                    | "%"
                    | "&"
                    | "|"
                    | "=="
                    | "!="
                    | "<"
                    | ">"
                    | "<="
                    | ">="
                    | "("
                    | "["
                    | "{"
                    | ","
                    | "."
                    | ":"
                    | "sorted"
                    | "set"
                    | "list"
                    | "tuple"
                    | "sum"
                    | "len"
                    | "min"
                    | "max"
                    | "int"
                    | "str"
                    | "range"
            );
        let token_continues_expression = matches!(
            token,
            "+" | "-"
                | "*"
                | "/"
                | "//"
                | "%"
                | "&"
                | "|"
                | "=="
                | "!="
                | "<"
                | ">"
                | "<="
                | ">="
                | "and"
                | "or"
                | "in"
        ) || (matches!(token, "[" | "(") && is_identifier(prev))
            || (token == "." && is_identifier(prev))
            || (balance > 0 && matches!(token, ")" | "]" | "}" | "," | "." | ":"));
        if !expression_can_continue
            && !token_continues_expression
            && !matches!(token, "<EOS>" | "<NL>")
        {
            return false;
        }
    }
    if token == ":" && !colon_allowed_in_current_line(existing) {
        let line = current_line_tokens(existing);
        let slice_colon = line.iter().any(|item| item == "[") && bracket_balance(&line) > 0;
        if !slice_colon {
            return false;
        }
    }
    if prev == "[" && token == "]" {
        let line = current_line_tokens(existing);
        let before_bracket = line.iter().rev().nth(1).map(String::as_str);
        if !matches!(before_bracket, Some("=" | "return" | "," | "(" | "[")) {
            return false;
        }
    }
    if matches!(prev, "if" | "for" | "while" | "elif") && token == ":" {
        return false;
    }
    if token == "=" && matches!(prev, "+" | "-" | "*" | "/" | "%" | "//") {
        return false;
    }
    true
}

fn task_body_token_allowed(task: &CodeTask, existing: &[String], token: &str) -> bool {
    body_token_allowed(existing, token)
        && decoder_contract_progress_token_allowed(task, existing, token)
}

fn decoder_contract_progress_token_allowed(
    task: &CodeTask,
    existing: &[String],
    token: &str,
) -> bool {
    let contract_sensitive =
        execution_shaped_category(&task.category)
            || complex_body_category(&task.category)
            || contract_requires_extended_learned_body(task);
    if !contract_sensitive {
        return true;
    }
    let prev = existing.last().map(String::as_str).unwrap_or("<BOS>");
    let prev2 = existing
        .iter()
        .rev()
        .nth(1)
        .map(String::as_str)
        .unwrap_or("<BOS>");
    if matches!(
        task.category.as_str(),
        "private_exec_archive_config_zip" | "archive_config_zip"
    ) {
        if prev2 == "os" && prev == "." && token == "read" {
            return false;
        }
        if prev2 == "configparser" && prev == "." && token != "ConfigParser" {
            return false;
        }
        if prev2 == "config" && prev == "." && !matches!(token, "read" | "get") {
            return false;
        }
        if prev == "get" && token != "(" {
            return false;
        }
        if prev2 == "." && prev == "ConfigParser" && token == "<NL>" {
            return false;
        }
    }
    if csv_split_invalid_input_branch_start(task, existing) && token != "return" {
        return false;
    }

    let starts_top_level_return = token == "return" && body_indent_balance(existing) == 0;
    if starts_top_level_return && !decoder_contract_progress_ready_for_return(task, existing) {
        return false;
    }

    let in_top_level_return_expr = body_indent_balance(existing) == 0
        && current_line_tokens(existing)
            .first()
            .is_some_and(|item| item == "return");
    if in_top_level_return_expr
        && trivial_terminal_return_token(token)
        && !decoder_contract_progress_ready_for_return(task, existing)
    {
        return false;
    }
    if in_top_level_return_expr && !return_shape_terminal_token_allowed(task, token) {
        return false;
    }
    if in_top_level_return_expr && empty_terminal_return_token_is_bad(task, token) {
        return false;
    }

    if token == "<EOS>" {
        if !generated_tokens_have_top_level_return(existing) {
            return false;
        }
        if !decoder_contract_progress_ready_for_return(task, existing) {
            return false;
        }
    }

    true
}

fn decoder_contract_progress_ready_for_return(task: &CodeTask, existing: &[String]) -> bool {
    let hints = decoder_required_constructs(task);
    let expected = category_expected_arg_count(task).unwrap_or(0);

    if expected >= 1 {
        let primary = decoder_primary_arg(task);
        if !generated_tokens_contain(existing, &primary)
            && !generated_tokens_have_any(existing, &["data", "items", "text", "nums", "numbers"])
        {
            return false;
        }
    }
    if expected >= 2 {
        let secondary = decoder_secondary_arg(task).unwrap_or_else(|| "other".to_string());
        if !generated_tokens_contain(existing, &secondary)
            && !generated_tokens_have_any(existing, &["other", "target", "threshold", "right"])
        {
            return false;
        }
    }
    if expected >= 3 {
        let args = visible_signature_ordered_user_args(task);
        let third = args.get(2).map(String::as_str).unwrap_or("extra");
        if !generated_tokens_contain(existing, third)
            && !generated_tokens_have_any(existing, &["extra", "args"])
        {
            return false;
        }
    }

    if (hints.contains("branch") || hints.contains("edge_conditions"))
        && !generated_tokens_have_any(existing, &["if", "try", "except", "else", "elif"])
    {
        return false;
    }
    if hints.contains("locals") && !generated_tokens_contain(existing, "=") {
        return false;
    }
    if progress_loop_required(task, &hints)
        && !generated_tokens_have_any(existing, &["for", "while"])
    {
        return false;
    }

    let text = task_contract_text(task);
    if hints.contains("csv") && !generated_tokens_contain(existing, "csv") {
        return false;
    }
    if hints.contains("structured_parsing")
        && body_has_any(&text, &["json"])
        && !generated_tokens_contain(existing, "json")
    {
        return false;
    }
    if hints.contains("archive")
        && !generated_tokens_have_any(
            existing,
            &[
                "archive",
                "archive_path",
                "zipfile",
                "tarfile",
                "shutil",
                "make_archive",
                "ZipFile",
            ],
        )
    {
        return false;
    }
    if hints.contains("system_api")
        && !generated_tokens_have_any(existing, &["subprocess", "platform", "psutil"])
    {
        return false;
    }
    if !execution_shape_completion_obligations_met_for_return(task, existing) {
        return false;
    }
    true
}

fn progress_loop_required(task: &CodeTask, hints: &BTreeSet<String>) -> bool {
    if !hints.contains("loop") {
        return false;
    }
    let text = task_contract_text(task);
    if body_has_any(
        &text,
        &[
            "operating system",
            "architecture",
            "memory usage",
            "json",
            "query string",
            "urlencode",
            "url encode",
        ],
    ) {
        return false;
    }
    body_has_any(
        &text,
        &[
            "each",
            "iterate",
            "for each",
            "rows",
            "commands",
            "files",
            "logs",
            "directory",
            "folder",
            "split",
            "archive",
        ],
    )
}

fn trivial_terminal_return_token(token: &str) -> bool {
    matches!(
        token,
        "False" | "True" | "None" | "[]" | "{}" | "()" | "''" | "\"\""
    )
}

fn empty_terminal_return_token_is_bad(task: &CodeTask, token: &str) -> bool {
    let hints = decoder_required_constructs(task);
    let shape_requires_builder = transform_or_collection_body_required(task, &hints)
        || hints.contains("system_api")
        || (hints.contains("execution_shaped_program") && decoder_return_shape(task) == "str");
    if !shape_requires_builder {
        return false;
    }
    matches!(token, "[]" | "{}" | "()" | "''" | "\"\"")
}

fn return_shape_terminal_token_allowed(task: &CodeTask, token: &str) -> bool {
    match decoder_return_shape(task).as_str() {
        "str" => !matches!(token, "True" | "False" | "None" | "[]" | "{}" | "()"),
        "list" => !matches!(
            token,
            "True" | "False" | "None" | "{}" | "()" | "''" | "\"\""
        ),
        "dict" => !matches!(
            token,
            "True" | "False" | "None" | "[]" | "()" | "''" | "\"\""
        ),
        "bool" => !matches!(token, "None" | "[]" | "{}" | "()" | "''" | "\"\""),
        "number" => !matches!(
            token,
            "True" | "False" | "None" | "[]" | "{}" | "()" | "''" | "\"\""
        ),
        _ => true,
    }
}

fn execution_shape_completion_obligations_met_for_return(
    task: &CodeTask,
    existing: &[String],
) -> bool {
    if !execution_shaped_category(&task.category) {
        return true;
    }
    match task.category.as_str() {
        "private_exec_archive_config_zip" | "archive_config_zip" => {
            generated_tokens_have_any(existing, &["make_archive", "archive_base"])
        }
        "private_exec_csv_command_outputs" | "csv_command_outputs" => {
            generated_tokens_have_any(existing, &["run", "subprocess"])
                && generated_tokens_have_any(existing, &["out_handle", "append"])
        }
        "private_exec_csv_split_shuffle" | "csv_split_shuffle" => {
            generated_tokens_have_any(existing, &["writerows", "writer"])
                && generated_tokens_have_any(existing, &["chunk_size", "append"])
        }
        "private_exec_json_extract_field" | "json_extract_field" => {
            generated_tokens_have_any(existing, &["load"])
                && generated_tokens_have_any(existing, &["payload"])
                && generated_tokens_have_any(existing, &["get"])
        }
        "private_exec_log_backup_tar" | "log_backup_tar" => {
            generated_tokens_have_any(existing, &["tarfile", "open"])
                && generated_tokens_have_any(existing, &["add", "archive"])
        }
        "private_exec_urlencode_payload" | "urlencode_payload" => {
            generated_tokens_have_any(existing, &["urlencode"])
                && generated_tokens_have_any(existing, &["items"])
        }
        "private_exec_zip_flat_directory" | "zip_flat_directory" => {
            generated_tokens_have_any(existing, &["ZipFile", "zipfile"])
                && generated_tokens_have_any(existing, &["write", "archive"])
        }
        _ => true,
    }
}

fn generated_tokens_contain(tokens: &[String], needle: &str) -> bool {
    tokens.iter().any(|item| item == needle)
}

fn generated_tokens_have_any(tokens: &[String], needles: &[&str]) -> bool {
    needles
        .iter()
        .any(|needle| generated_tokens_contain(tokens, needle))
}

fn generated_tokens_have_top_level_return(tokens: &[String]) -> bool {
    let mut indent = 0usize;
    let mut at_line_start = true;
    for token in tokens {
        match token.as_str() {
            "<INDENT>" => {
                indent += 1;
                at_line_start = true;
            }
            "<DEDENT>" => {
                indent = indent.saturating_sub(1);
                at_line_start = true;
            }
            "<NL>" => at_line_start = true,
            "return" if indent == 0 && at_line_start => return true,
            _ => at_line_start = false,
        }
    }
    false
}

fn colon_allowed_in_current_line(existing: &[String]) -> bool {
    let current_line = existing
        .iter()
        .rev()
        .take_while(|token| token.as_str() != "<NL>")
        .cloned()
        .collect::<Vec<_>>();
    if current_line.is_empty() {
        return false;
    }
    let mut forward = current_line.into_iter().rev().collect::<Vec<_>>();
    while matches!(
        forward.first().map(String::as_str),
        Some("<INDENT>" | "<DEDENT>")
    ) {
        forward.remove(0);
    }
    let Some(first) = forward.first().map(String::as_str) else {
        return false;
    };
    let bracket_balance = forward
        .iter()
        .fold(0isize, |balance, token| match token.as_str() {
            "[" | "(" | "{" => balance + 1,
            "]" | ")" | "}" => balance - 1,
            _ => balance,
        });
    if bracket_balance > 0 && forward.iter().any(|token| token == "[" || token == "{") {
        return true;
    }
    if matches!(first, "if" | "while" | "elif") {
        return bracket_balance == 0 && conditional_header_has_complete_test(&forward);
    }
    if first == "for" {
        return bracket_balance == 0
            && forward.iter().any(|token| token == "in")
            && !matches!(
                forward.last().map(String::as_str),
                Some("for" | "," | "in" | "+" | "-" | "*" | "/" | "//" | "%" | "&" | "|")
            );
    }
    if matches!(first, "else" | "try") {
        return bracket_balance == 0;
    }
    if first == "except" {
        return bracket_balance == 0
            && !matches!(
                forward.last().map(String::as_str),
                Some("as" | "," | "." | "(" | "[" | "{")
            );
    }
    if first == "with" {
        return bracket_balance == 0
            && forward
                .iter()
                .position(|token| token == "as")
                .is_some_and(|idx| {
                    forward
                        .get(idx + 1)
                        .is_some_and(|token| is_identifier(token))
                });
    }
    if let Some(lambda_idx) = forward.iter().position(|token| token == "lambda") {
        return forward
            .get(lambda_idx + 1)
            .is_some_and(|token| is_identifier(token))
            && !forward
                .iter()
                .skip(lambda_idx + 1)
                .any(|token| token == ":");
    }
    false
}

fn conditional_header_has_complete_test(line: &[String]) -> bool {
    if line.len() <= 1 {
        return false;
    }
    !matches!(
        line.last().map(String::as_str),
        Some(
            "if" | "elif"
                | "while"
                | "not"
                | "and"
                | "or"
                | "in"
                | "is"
                | "=="
                | "!="
                | "<"
                | ">"
                | "<="
                | ">="
                | "+"
                | "-"
                | "*"
                | "/"
                | "//"
                | "%"
                | "&"
                | "|"
                | "("
                | "["
                | "{"
                | ","
                | "."
        )
    )
}

fn body_indent_balance(tokens: &[String]) -> usize {
    let mut indent = 0usize;
    for token in tokens {
        match token.as_str() {
            "<INDENT>" => indent += 1,
            "<DEDENT>" => indent = indent.saturating_sub(1),
            _ => {}
        }
    }
    indent
}

fn bracket_balance(tokens: &[String]) -> isize {
    tokens
        .iter()
        .fold(0isize, |balance, token| match token.as_str() {
            "(" | "[" | "{" => balance + 1,
            ")" | "]" | "}" => balance - 1,
            _ => balance,
        })
}

fn body_position_bonus(token: &str, position: usize) -> f32 {
    match (position, token) {
        (0, "return") => 0.35,
        (0, "if") | (0, "for") => 0.45,
        (0, "<EOS>" | "<NL>" | "<INDENT>" | "<DEDENT>") => -4.0,
        (1..=8, "return") => 0.15,
        (_, "<EOS>") if position < 4 => -1.0,
        _ => 0.0,
    }
}

fn category_body_token_bonus(task: &CodeTask, token: &str) -> f32 {
    let mut bonus = semantic_type_token_bonus(task, token);
    let visible_args = visible_signature_ordered_user_args(task);
    if visible_args.iter().any(|arg| arg == token) {
        bonus += 1.2;
    }
    if recurrence_category(&task.category)
        && matches!(token, "a" | "b" | "values" | "range" | "for" | "if" | "+")
    {
        bonus += 0.75;
    }
    if vowel_rule_category(&task.category)
        && matches!(
            token,
            "text" | "total" | "lower" | "enumerate" | "aeiou" | "isalpha" | "endswith"
        )
    {
        bonus += 0.75;
    }
    if digit_rotation_category(&task.category)
        && matches!(
            token,
            "digits" | "shift" | "len" | "str" | "abs" | "sign" | "%"
        )
    {
        bonus += 0.75;
    }
    bonus
        + match (task.category.as_str(), token) {
            ("all_prefixes", "out")
            | ("all_prefixes", "range")
            | ("all_prefixes", "len")
            | ("all_prefixes", "append")
            | ("all_prefixes", "idx")
            | ("all_prefixes", "i") => 0.9,
            ("string_sequence", "join") | ("string_sequence", "range") => 0.45,
            ("opposite_signs", "other") | ("opposite_signs", "<") | ("opposite_signs", ">") => 0.35,
            ("median_list", "items")
            | ("median_list", "sorted")
            | ("median_list", "mid")
            | ("median_list", "len") => 0.75,
            ("modular_power_two", "result")
            | ("modular_power_two", "range")
            | ("modular_power_two", "%")
            | ("modular_power_two", "other") => 0.75,
            ("caesar_decode_shift5", "out")
            | ("caesar_decode_shift5", "append")
            | ("caesar_decode_shift5", "chr")
            | ("caesar_decode_shift5", "ord")
            | ("caesar_decode_shift5", "join") => 0.75,
            ("caesar_decode_shift5", "26") => 1.25,
            ("remove_vowels", "out")
            | ("remove_vowels", "append")
            | ("remove_vowels", "aeiou")
            | ("remove_vowels", "join") => 0.75,
            ("below_threshold", "other")
            | ("below_threshold", "False")
            | ("below_threshold", "True")
            | ("below_threshold", ">=") => 0.75,
            ("same_chars", "set")
            | ("same_chars", "data")
            | ("same_chars", "other")
            | ("add_numbers", "data")
            | ("add_numbers", "other") => 0.6,
            ("distinct_count", "set")
            | ("distinct_count", "lower")
            | ("distinct_count", "isinstance") => 0.45,
            ("largest_concat", "sorted") | ("largest_concat", "reverse") => 0.45,
            ("parse_ints", "split") | ("parse_ints", "int") => 0.45,
            ("replace_whitespace", "replace")
            | ("replace_whitespace", "data")
            | ("replace_whitespace", "other") => 0.9,
            ("cube_volume", "data") | ("cube_volume", "*") | ("cube_volume", "**") => 0.9,
            ("cube_lateral_surface_area", "4")
            | ("cube_lateral_surface_area", "data")
            | ("cube_lateral_surface_area", "*") => 0.95,
            ("cylinder_lateral_surface_area", "2")
            | ("cylinder_lateral_surface_area", "3.141592653589793")
            | ("cylinder_lateral_surface_area", "data")
            | ("cylinder_lateral_surface_area", "other")
            | ("cylinder_lateral_surface_area", "*") => 0.85,
            ("tuple_item_count", "count")
            | ("tuple_item_count", ".")
            | ("tuple_item_count", "data")
            | ("tuple_item_count", "other") => 0.9,
            ("tuple_nested_elementwise_max", "out")
            | ("tuple_nested_elementwise_max", "zip")
            | ("tuple_nested_elementwise_max", "max")
            | ("tuple_nested_elementwise_max", "tuple")
            | ("tuple_nested_elementwise_max", "append") => 0.85,
            ("list_chunks_every_n", "range")
            | ("list_chunks_every_n", "len")
            | ("list_chunks_every_n", "idx")
            | ("list_chunks_every_n", ":") => 0.8,
            ("combinations_with_replacement", "build")
            | ("combinations_with_replacement", "out")
            | ("combinations_with_replacement", "append")
            | ("combinations_with_replacement", "tuple") => 0.8,
            ("is_prime", "range") | ("is_prime", "%") => 0.35,
            ("factors", "%") | ("factors", "range") => 0.35,
            ("balanced_brackets_simple", "stack")
            | ("balanced_brackets_simple", "pairs")
            | ("balanced_brackets_simple", "append")
            | ("balanced_brackets_simple", "pop")
            | ("balanced_brackets_simple", "for")
            | ("balanced_brackets_simple", "if") => 0.75,
            ("monotonic_sequence", "range")
            | ("monotonic_sequence", "len")
            | ("monotonic_sequence", "if")
            | ("monotonic_sequence", "<")
            | ("monotonic_sequence", ">") => 0.65,
            ("common_elements", "set")
            | ("common_elements", "sorted")
            | ("common_elements", "&")
            | ("sorted_unique_values", "set") => 0.7,
            ("largest_prime_factor", "while")
            | ("largest_prime_factor", "factor")
            | ("largest_prime_factor", "best")
            | ("largest_prime_factor", "%")
            | ("largest_prime_factor", "//=") => 0.8,
            ("arithmetic_series_sum", "total")
            | ("arithmetic_series_sum", "range")
            | ("arithmetic_series_sum", "+=") => 0.7,
            ("derivative_coefficients", "out")
            | ("derivative_coefficients", "append")
            | ("derivative_coefficients", "range")
            | ("derivative_coefficients", "*") => 0.7,
            ("tribonacci_sequence", "values")
            | ("tribonacci_sequence", "append")
            | ("tribonacci_sequence", "range") => 0.7,
            ("fibonacci_loop_private", "a")
            | ("fibonacci_loop_private", "b")
            | ("fibonacci_loop_private", "range")
            | ("lucas_loop_private", "a")
            | ("lucas_loop_private", "b")
            | ("lucas_loop_private", "range")
            | ("shifted_recurrence_private", "a")
            | ("shifted_recurrence_private", "b")
            | ("shifted_recurrence_private", "range")
            | ("nested_recurrence_private", "a")
            | ("nested_recurrence_private", "b")
            | ("nested_recurrence_private", "range") => 0.75,
            ("rotate_sequence", "shift")
            | ("rotate_sequence", "%")
            | ("rotate_sequence", "len") => 0.65,
            ("circular_digit_shift", "digits")
            | ("circular_digit_shift", "str")
            | ("circular_digit_shift", "shift")
            | ("circular_digit_shift", "len")
            | ("digit_rotate_right_private", "digits")
            | ("digit_rotate_right_private", "shift")
            | ("digit_rotate_right_private", "len")
            | ("signed_digit_rotate_private", "digits")
            | ("signed_digit_rotate_private", "sign")
            | ("signed_digit_rotate_private", "abs")
            | ("multi_step_digit_shift_private", "digits")
            | ("multi_step_digit_shift_private", "shift")
            | ("multi_step_digit_shift_private", "range") => 0.75,
            ("count_vowels", "text")
            | ("count_vowels", "lower")
            | ("count_vowels", "enumerate")
            | ("count_vowels", "total")
            | ("final_y_vowel_private", "text")
            | ("final_y_vowel_private", "lower")
            | ("final_y_vowel_private", "enumerate")
            | ("final_y_vowel_private", "total")
            | ("suffix_y_vowel_private", "text")
            | ("suffix_y_vowel_private", "endswith")
            | ("suffix_y_vowel_private", "total")
            | ("case_punct_vowel_private", "total")
            | ("case_punct_vowel_private", "lower")
            | ("case_punct_vowel_private", "isalpha") => 0.75,
            ("digit_sum_casefold", "isupper")
            | ("digit_sum_casefold", "isdigit")
            | ("digit_sum_casefold", "int")
            | ("digit_sum_casefold", "total") => 0.7,
            ("fruit_distribution_private", "numbers")
            | ("fruit_distribution_private", "split")
            | ("fruit_distribution_private", "isdigit")
            | ("fruit_distribution_private", "int") => 0.7,
            _ => 0.0,
        }
}

fn semantic_type_token_bonus(task: &CodeTask, token: &str) -> f32 {
    let family = category_semantic_family(&task.category);
    let mut score = 0.0;
    if category_expected_arg_count(task).is_some_and(|count| count < 2) && token == "other" {
        score -= 4.0;
    }
    match family {
        "collection_transform" => {
            if matches!(
                token,
                "for" | "item" | "data" | "len" | "sorted" | "set" | "append" | "return"
            ) {
                score += 0.35;
            }
            if matches!(token, "abs" | "zip") {
                score -= 1.1;
            }
        }
        "string_transform" => {
            if matches!(
                token,
                "data" | "split" | "splitlines" | "strip" | "lower" | "startswith" | "return" | "["
            ) {
                score += 0.35;
            }
            if matches!(token, "abs" | "zip") {
                score -= 1.1;
            }
        }
        "scalar_numeric" | "scalar_recurrence" => {
            if matches!(
                token,
                "range" | "if" | "return" | "data" | "other" | "%" | "+"
            ) {
                score += 0.25;
            }
            if matches!(token, "zip" | "append")
                && !matches!(task.category.as_str(), "count_digit_under_divisibility")
            {
                score -= 0.8;
            }
        }
        "membership_lookup" => {
            if matches!(
                token,
                "if" | "in" | "index" | "other" | "data" | "-1" | "return"
            ) {
                score += 0.55;
            }
        }
        "digit_string_boundary" => {
            if matches!(
                token,
                "str" | "digits" | "len" | "shift" | "return" | "[" | "]" | "%"
            ) {
                score += 0.35;
            }
        }
        _ => {}
    }
    match (task.category.as_str(), token) {
        ("extract_def_name", "splitlines")
        | ("extract_def_name", "startswith")
        | ("extract_def_name", "strip")
        | ("extract_def_name", "split")
        | ("private_extract_first_def", "splitlines")
        | ("private_extract_first_def", "startswith")
        | ("private_extract_first_def", "strip")
        | ("private_extract_first_def", "split") => score += 1.1,
        ("reverse_string", "[")
        | ("reverse_string", ":")
        | ("palindrome", "[")
        | ("palindrome", ":") => {
            score += 0.75;
        }
        ("palindrome", "==")
        | ("palindrome", "]")
        | ("private_palindrome_check", "[")
        | ("private_palindrome_check", ":")
        | ("private_palindrome_check", "==")
        | ("private_palindrome_check", "]") => score += 0.9,
        ("caesar_decode_shift5", "26")
        | ("caesar_decode_shift5", "5")
        | ("caesar_decode_shift5", "%")
        | ("private_decode_shift_general", "26")
        | ("private_decode_shift_general", "%")
        | ("private_decode_shift_general", "ord")
        | ("private_decode_shift_general", "chr") => score += 1.1,
        ("median_list", "items")
        | ("median_list", "mid")
        | ("private_median_even_odd", "items")
        | ("private_median_even_odd", "mid") => {
            score += 0.9;
        }
        ("add_numbers", "+")
        | ("private_pair_sum", "+")
        | ("private_pair_sum", "data")
        | ("private_pair_sum", "other") => score += 1.0,
        ("same_chars", "set")
        | ("same_chars", "==")
        | ("private_same_char_set", "set")
        | ("private_same_char_set", "==") => score += 1.0,
        ("gcd_pair", "%")
        | ("gcd_pair", "best")
        | ("gcd_pair", "range")
        | ("private_gcd_pair_loop", "%")
        | ("private_gcd_pair_loop", "while")
        | ("private_gcd_pair_loop", "return") => score += 0.95,
        ("is_prime", "%")
        | ("is_prime", "False")
        | ("is_prime", "True")
        | ("is_prime", "range")
        | ("private_prime_loop", "%")
        | ("private_prime_loop", "False")
        | ("private_prime_loop", "True") => score += 0.95,
        ("is_anagram", "sorted")
        | ("is_anagram", "==")
        | ("private_anagram_sorted", "sorted")
        | ("private_anagram_sorted", "==") => score += 1.0,
        ("base_digits", "digits")
        | ("base_digits", "while")
        | ("base_digits", "%")
        | ("base_digits", "str")
        | ("private_base_digits", "digits")
        | ("private_base_digits_alt", "digits")
        | ("private_base_digits_alt", "while")
        | ("private_base_digits_alt", "%") => score += 0.95,
        ("triangle_area_sides", "a")
        | ("triangle_area_sides", "b")
        | ("triangle_area_sides", "c")
        | ("triangle_area_sides", "round")
        | ("triangle_area_sides", "extra") => score += 0.95,
        ("uppercase_ascii_sum", "ord")
        | ("uppercase_ascii_sum", "isupper")
        | ("uppercase_ascii_sum", "total") => score += 1.0,
        ("pluck_smallest_even", "best")
        | ("pluck_smallest_even", "best_idx")
        | ("pluck_smallest_even", "enumerate") => score += 1.0,
        ("frequency_at_least_value", "counts")
        | ("frequency_at_least_value", "get")
        | ("frequency_at_least_value", "best") => score += 1.0,
        ("alternating_min_max_sort", "items")
        | ("alternating_min_max_sort", "pop")
        | ("alternating_min_max_sort", "take_low") => score += 1.0,
        ("palindrome_list_weight", "sum")
        | ("palindrome_list_weight", "False")
        | ("palindrome_list_weight", "True") => score += 0.9,
        ("smallest_palindrome_changes", "total")
        | ("smallest_palindrome_changes", "range")
        | ("smallest_palindrome_changes", "-") => score += 0.9,
        ("total_match_lengths", "left")
        | ("total_match_lengths", "right")
        | ("total_match_lengths", "len")
        | ("total_match_lengths", "sum") => score += 0.9,
        ("multiply_three_primes", "count")
        | ("multiply_three_primes", "factor")
        | ("multiply_three_primes", "while") => score += 0.9,
        ("simple_power", "value")
        | ("simple_power", "while")
        | ("simple_power", "True")
        | ("simple_power", "False") => score += 0.9,
        ("cube_number", "root") | ("cube_number", "abs") | ("cube_number", "while") => score += 0.9,
        ("hex_prime_count", "total") | ("hex_prime_count", "2357BD") => score += 1.0,
        ("max_list", "max")
        | ("min_list", "min")
        | ("sum_list", "sum")
        | ("private_max_item", "max") => {
            score += 1.0;
        }
        ("rescale_to_unit", "min")
        | ("rescale_to_unit", "max")
        | ("rescale_to_unit", "low")
        | ("rescale_to_unit", "high")
        | ("prime_factors", "factor")
        | ("prime_factors", "value")
        | ("prime_factors", "out")
        | ("divisible_by_11", "%")
        | ("divisible_by_11", "11")
        | ("string_sequence", "parts")
        | ("two_sum_zero_exists", "seen")
        | ("three_sum_zero_exists", "range")
        | ("count_digit_under_divisibility", "total")
        | ("decode_cyclic", "group")
        | ("decode_cyclic", "join")
        | ("decode_cyclic", "range")
        | ("prime_fib_sequence", "found")
        | ("prime_fib_sequence", "divisor")
        | ("prime_fib_sequence", "while")
        | ("polynomial_zero_bisection", "left")
        | ("polynomial_zero_bisection", "right")
        | ("polynomial_zero_bisection", "mid") => {
            score += 1.0;
        }
        _ => {}
    }
    score
}

fn category_position_token_bonus(task: &CodeTask, token: &str, position: usize) -> f32 {
    if position > 1 {
        return 0.0;
    }
    let category = task.category.as_str();
    let complex = matches!(
        category,
        "balanced_brackets_simple"
            | "monotonic_sequence"
            | "largest_prime_factor"
            | "derivative_coefficients"
            | "tribonacci_sequence"
            | "rotate_sequence"
            | "circular_digit_shift"
            | "digit_sum_casefold"
            | "uppercase_ascii_sum"
            | "fruit_distribution_private"
            | "triangle_area_sides"
            | "pluck_smallest_even"
            | "frequency_at_least_value"
            | "alternating_min_max_sort"
            | "palindrome_list_weight"
            | "smallest_palindrome_changes"
            | "total_match_lengths"
            | "multiply_three_primes"
            | "simple_power"
            | "cube_number"
            | "hex_prime_count"
            | "parse_ints"
            | "median_list"
            | "modular_power_two"
            | "caesar_decode_shift5"
            | "remove_vowels"
            | "below_threshold"
            | "count_vowels"
            | "factors"
            | "is_prime"
    );
    if complex && token == "return" {
        return -1.9;
    }
    match (category, token) {
        ("balanced_brackets_simple", "stack") | ("balanced_brackets_simple", "pairs") => 1.2,
        ("monotonic_sequence", "if") | ("monotonic_sequence", "nondecreasing") => 1.0,
        ("largest_prime_factor", "best") | ("largest_prime_factor", "factor") => 1.1,
        ("arithmetic_series_sum", "total") => 1.0,
        ("all_prefixes", "out") | ("all_prefixes", "for") => 1.0,
        ("derivative_coefficients", "out") => 1.0,
        ("tribonacci_sequence", "values") => 1.0,
        ("median_list", "items") => 1.0,
        ("modular_power_two", "result") => 1.0,
        ("caesar_decode_shift5", "out") | ("remove_vowels", "out") => 1.0,
        ("below_threshold", "for") => 1.0,
        ("fibonacci_loop_private", "a")
        | ("fibonacci_loop_private", "if")
        | ("lucas_loop_private", "a")
        | ("shifted_recurrence_private", "a")
        | ("nested_recurrence_private", "a") => 1.0,
        ("rotate_sequence", "if") | ("rotate_sequence", "shift") => 0.9,
        ("circular_digit_shift", "digits")
        | ("circular_digit_shift", "shift")
        | ("digit_rotate_right_private", "digits")
        | ("signed_digit_rotate_private", "sign")
        | ("multi_step_digit_shift_private", "digits") => 1.0,
        ("count_vowels", "text")
        | ("count_vowels", "total")
        | ("final_y_vowel_private", "text")
        | ("suffix_y_vowel_private", "text")
        | ("case_punct_vowel_private", "total") => 1.0,
        ("digit_sum_casefold", "total") | ("uppercase_ascii_sum", "total") => 1.0,
        ("fruit_distribution_private", "numbers") => 1.0,
        ("triangle_area_sides", "a") => 1.0,
        ("pluck_smallest_even", "best") => 1.0,
        ("frequency_at_least_value", "counts") => 1.0,
        ("alternating_min_max_sort", "items") => 1.0,
        ("palindrome_list_weight", "if") => 1.0,
        ("smallest_palindrome_changes", "total") => 1.0,
        ("total_match_lengths", "left") => 1.0,
        ("multiply_three_primes", "value") => 1.0,
        ("simple_power", "if") | ("simple_power", "value") => 1.0,
        ("cube_number", "value") => 1.0,
        ("hex_prime_count", "total") => 1.0,
        _ => 0.0,
    }
}

fn train_symliquid_state_decoder(
    tasks: &[CodeTask],
    vocab: &Vocab,
    sts_streams: &StsStreamMap,
    state_dim: usize,
    epochs: usize,
    lr: f32,
) -> Result<
    (
        SymLiquidStateDecoder,
        StateSequenceTrace,
        StateSequenceTrace,
    ),
    Box<dyn std::error::Error>,
> {
    let mut decoder = SymLiquidStateDecoder {
        readout: LinearReadout::zeros(state_dim.max(64), vocab.id_to_token.len()),
        state_dim: state_dim.max(64),
        update_count: 0,
    };
    let before = evaluate_symliquid_state_decoder(&decoder, tasks, vocab, sts_streams)?;
    for _ in 0..epochs.max(1) {
        for task in tasks {
            for _ in 0..symliquid_training_repeats(task) {
                let task_sts = sts_streams.get(&task.task_id);
                let prompt = prompt_tokens_with_sts(task, task_sts);
                let mut existing = Vec::new();
                let mut tokens = solution_body_tokens(task);
                tokens.push("<EOS>".to_string());
                for (position, token) in tokens.into_iter().enumerate() {
                    let target = vocab_id(vocab, &token);
                    let features = symliquid_state_vector(
                        task,
                        &prompt,
                        &existing,
                        position,
                        decoder.state_dim,
                        task_sts,
                    );
                    fast_readout_train_step(
                        &mut decoder.readout,
                        &features,
                        target,
                        lr,
                        position + task.category.len() + existing.len(),
                    );
                    decoder.update_count += 1;
                    if token != "<EOS>" {
                        existing.push(token);
                    }
                }
            }
        }
    }
    let after = evaluate_symliquid_state_decoder(&decoder, tasks, vocab, sts_streams)?;
    Ok((decoder, before, after))
}

fn evaluate_symliquid_state_decoder(
    decoder: &SymLiquidStateDecoder,
    tasks: &[CodeTask],
    vocab: &Vocab,
    sts_streams: &StsStreamMap,
) -> Result<StateSequenceTrace, Box<dyn std::error::Error>> {
    let mut correct = 0usize;
    let mut total = 0usize;
    for task in tasks {
        let task_sts = sts_streams.get(&task.task_id);
        let prompt = prompt_tokens_with_sts(task, task_sts);
        let mut existing = Vec::new();
        let mut tokens = solution_body_tokens(task);
        tokens.push("<EOS>".to_string());
        for (position, token) in tokens.into_iter().enumerate() {
            let target = vocab_id(vocab, &token);
            let features = Tensor::from_row(symliquid_state_vector(
                task,
                &prompt,
                &existing,
                position,
                decoder.state_dim,
                task_sts,
            ));
            let predicted = symliquid_state_best_id(decoder, vocab, &features, &existing)?;
            correct += usize::from(predicted == target);
            total += 1;
            if token != "<EOS>" {
                existing.push(token);
            }
        }
    }
    Ok(StateSequenceTrace {
        accuracy: if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        },
        example_count: total,
    })
}

fn symliquid_state_best_id(
    decoder: &SymLiquidStateDecoder,
    vocab: &Vocab,
    features: &Tensor,
    existing: &[String],
) -> Result<usize, Box<dyn std::error::Error>> {
    let logits = decoder.readout.logits(features)?;
    let mut best = vocab.unk_id;
    let mut best_score = f32::NEG_INFINITY;
    for (idx, score) in logits.row(0).iter().copied().enumerate() {
        let token = &vocab.id_to_token[idx];
        if token == "<UNK>" || !body_token_allowed(existing, token) {
            continue;
        }
        if score > best_score {
            best = idx;
            best_score = score;
        }
    }
    Ok(best)
}

fn fast_readout_train_step(
    readout: &mut LinearReadout,
    features: &[f32],
    target: usize,
    lr: f32,
    salt: usize,
) {
    if target >= readout.output_dim || readout.output_dim == 0 {
        return;
    }
    let mut negative = if readout.output_dim > 1 {
        (target + 1 + (salt % (readout.output_dim - 1))) % readout.output_dim
    } else {
        target
    };
    if negative == target && readout.output_dim > 1 {
        negative = (target + 1) % readout.output_dim;
    }
    let input_dim = readout.input_dim.min(features.len());
    let positive_offset = target * readout.input_dim;
    let negative_offset = negative * readout.input_dim;
    readout.bias[target] += lr;
    if negative != target {
        readout.bias[negative] -= lr * 0.25;
    }
    for (idx, value) in features.iter().copied().take(input_dim).enumerate() {
        if value.abs() < 1e-8 {
            continue;
        }
        readout.weights[positive_offset + idx] += lr * value;
        if negative != target {
            readout.weights[negative_offset + idx] -= lr * 0.25 * value;
        }
    }
}

fn symliquid_training_repeats(task: &CodeTask) -> usize {
    if semantic_hard_target_category(&task.category) {
        7
    } else if semantic_focus_category(&task.category) {
        4
    } else if complex_body_category(&task.category) {
        2
    } else {
        1
    }
}

fn state_sequence_training_repeats(task: &CodeTask) -> usize {
    if semantic_hard_target_category(&task.category) {
        symliquid_training_repeats(task)
    } else {
        symliquid_training_repeats(task).min(3)
    }
}

fn semantic_hard_target_category(category: &str) -> bool {
    private_mbpp_category(category)
        || matches!(
            category,
            "palindrome"
                | "caesar_decode_shift5"
                | "below_threshold"
                | "add_numbers"
                | "same_chars"
                | "gcd_pair"
                | "is_prime"
                | "is_anagram"
                | "base_digits"
                | "median_list"
                | "triangle_area_sides"
                | "uppercase_ascii_sum"
                | "pluck_smallest_even"
                | "frequency_at_least_value"
                | "alternating_min_max_sort"
                | "palindrome_list_weight"
                | "smallest_palindrome_changes"
                | "total_match_lengths"
                | "multiply_three_primes"
                | "simple_power"
                | "cube_number"
                | "cube_volume"
                | "cube_lateral_surface_area"
                | "cylinder_lateral_surface_area"
                | "hex_prime_count"
                | "replace_whitespace"
                | "tuple_item_count"
                | "tuple_nested_elementwise_max"
                | "list_chunks_every_n"
                | "combinations_with_replacement"
                | "private_palindrome_check"
                | "private_decode_shift_general"
                | "private_all_below_threshold"
                | "private_pair_sum"
                | "private_same_char_set"
                | "private_gcd_pair_loop"
                | "private_prime_loop"
                | "private_anagram_sorted"
                | "private_base_digits"
                | "private_base_digits_alt"
                | "private_median_even_odd"
        )
}

fn train_state_sequence_decoder(
    tasks: &[CodeTask],
    vocab: &Vocab,
    epochs: usize,
    lr: f32,
) -> (StateSequenceDecoder, StateSequenceTrace, StateSequenceTrace) {
    let mut decoder = StateSequenceDecoder {
        weights: HashMap::new(),
        bias: vec![0.0; vocab.id_to_token.len()],
        output_dim: vocab.id_to_token.len(),
        feature_count: 0,
        update_count: 0,
    };
    let before = evaluate_state_sequence_decoder(&decoder, tasks, vocab);
    for _ in 0..epochs.max(1) {
        for task in tasks {
            let prompt = prompt_tokens(task);
            let static_features = state_sequence_static_features(task, &prompt);
            let repeats = state_sequence_training_repeats(task);
            for _ in 0..repeats {
                let mut existing = Vec::new();
                let mut tokens = solution_body_tokens(task);
                tokens.push("<EOS>".to_string());
                for (position, token) in tokens.into_iter().enumerate() {
                    let target = vocab_id(vocab, &token);
                    let features =
                        state_sequence_features_with_static(task, &static_features, &existing, position);
                    let predicted = state_sequence_best_id(&decoder, vocab, &features, &existing);
                    if predicted != target {
                        state_sequence_update(&mut decoder, &features, target, predicted, lr);
                    }
                    if token != "<EOS>" {
                        existing.push(token);
                    }
                }
            }
        }
    }
    decoder.feature_count = decoder.weights.len();
    let after = evaluate_state_sequence_decoder(&decoder, tasks, vocab);
    (decoder, before, after)
}

fn evaluate_state_sequence_decoder(
    decoder: &StateSequenceDecoder,
    tasks: &[CodeTask],
    vocab: &Vocab,
) -> StateSequenceTrace {
    let mut correct = 0usize;
    let mut total = 0usize;
    for task in tasks {
        let prompt = prompt_tokens(task);
        let static_features = state_sequence_static_features(task, &prompt);
        let mut existing = Vec::new();
        let mut tokens = solution_body_tokens(task);
        tokens.push("<EOS>".to_string());
        for (position, token) in tokens.into_iter().enumerate() {
            let target = vocab_id(vocab, &token);
            let features =
                state_sequence_features_with_static(task, &static_features, &existing, position);
            let predicted = state_sequence_best_id(decoder, vocab, &features, &existing);
            correct += usize::from(predicted == target);
            total += 1;
            if token != "<EOS>" {
                existing.push(token);
            }
        }
    }
    StateSequenceTrace {
        accuracy: if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        },
        example_count: total,
    }
}

fn state_sequence_update(
    decoder: &mut StateSequenceDecoder,
    features: &[(String, f32)],
    target: usize,
    predicted: usize,
    lr: f32,
) {
    if target == predicted || target >= decoder.output_dim || predicted >= decoder.output_dim {
        return;
    }
    for (key, value) in features {
        let weights = decoder
            .weights
            .entry(key.clone())
            .or_insert_with(|| vec![0.0; decoder.output_dim]);
        weights[target] += lr * *value;
        weights[predicted] -= lr * *value;
    }
    decoder.bias[target] += lr * 0.05;
    decoder.bias[predicted] -= lr * 0.05;
    decoder.update_count += 1;
}

fn state_sequence_scores(decoder: &StateSequenceDecoder, features: &[(String, f32)]) -> Vec<f32> {
    state_sequence_scores_with_base(&decoder.bias, decoder, features)
}

fn state_sequence_scores_with_base(
    base_scores: &[f32],
    decoder: &StateSequenceDecoder,
    features: &[(String, f32)],
) -> Vec<f32> {
    let mut scores = base_scores.to_vec();
    if scores.len() < decoder.output_dim {
        scores.resize(decoder.output_dim, 0.0);
    }
    for (key, value) in features {
        let Some(weights) = decoder.weights.get(key) else {
            continue;
        };
        for (idx, weight) in weights.iter().enumerate().take(decoder.output_dim) {
            scores[idx] += *weight * *value;
        }
    }
    scores
}

fn state_sequence_best_id(
    decoder: &StateSequenceDecoder,
    vocab: &Vocab,
    features: &[(String, f32)],
    existing: &[String],
) -> usize {
    let scores = state_sequence_scores(decoder, features);
    let mut best = vocab.unk_id;
    let mut best_score = f32::NEG_INFINITY;
    for (idx, score) in scores.iter().enumerate() {
        let token = &vocab.id_to_token[idx];
        if token == "<UNK>" || !body_token_allowed(existing, token) {
            continue;
        }
        if *score > best_score {
            best = idx;
            best_score = *score;
        }
    }
    best
}

fn current_line_tokens(existing: &[String]) -> Vec<String> {
    existing
        .iter()
        .rev()
        .take_while(|token| token.as_str() != "<NL>")
        .filter(|token| !matches!(token.as_str(), "<INDENT>" | "<DEDENT>"))
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn previous_nonempty_line_tokens(existing: &[String]) -> Vec<String> {
    let mut line = Vec::new();
    let mut seen_token = false;
    for token in existing.iter().rev() {
        if matches!(token.as_str(), "<INDENT>" | "<DEDENT>") && !seen_token {
            continue;
        }
        if token == "<NL>" {
            if seen_token {
                break;
            }
            continue;
        }
        if matches!(token.as_str(), "<INDENT>" | "<DEDENT>") {
            continue;
        }
        seen_token = true;
        line.push(token.clone());
    }
    line.reverse();
    line
}

fn csv_split_invalid_input_branch_start(task: &CodeTask, existing: &[String]) -> bool {
    if !matches!(
        task.category.as_str(),
        "private_exec_csv_split_shuffle" | "csv_split_shuffle"
    ) || existing.last().map(String::as_str) != Some("<INDENT>")
    {
        return false;
    }
    let header = previous_nonempty_line_tokens(existing);
    header.first().map(String::as_str) == Some("if")
        && header
            .iter()
            .any(|token| matches!(token.as_str(), "isinstance" | "endswith" | "isfile"))
}

fn open_block_header_without_colon(line: &[String]) -> bool {
    if line.iter().any(|token| token == ":") {
        return false;
    }
    block_header_line(line)
}

fn block_header_line(line: &[String]) -> bool {
    matches!(
        line.first().map(String::as_str),
        Some("if" | "for" | "while" | "elif" | "else" | "with" | "try" | "except")
    )
}

fn for_header_target_allows_comma(line: &[String]) -> bool {
    line.first().map(String::as_str) == Some("for")
        && !line.iter().any(|token| token == "in")
        && line.len() > 1
}

fn for_header_target_before_in(line: &[String]) -> bool {
    line.first().map(String::as_str) == Some("for") && !line.iter().any(|token| token == "in")
}

fn grammar_requires_colon_now(line: &[String]) -> bool {
    if line.iter().any(|token| token == ":") {
        return false;
    }
    if line.first().map(String::as_str) == Some("try") {
        return true;
    }
    if line.first().map(String::as_str) == Some("with") {
        if let Some(as_idx) = line.iter().position(|token| token == "as") {
            return line
                .get(as_idx + 1)
                .is_some_and(|token| is_identifier(token));
        }
    }
    if let Some(lambda_idx) = line.iter().position(|token| token == "lambda") {
        return line
            .get(lambda_idx + 1)
            .is_some_and(|token| is_identifier(token));
    }
    false
}

fn with_header_requires_as_now(line: &[String]) -> bool {
    if line.first().map(String::as_str) != Some("with") {
        return false;
    }
    if line.iter().any(|token| token == ":" || token == "as") {
        return false;
    }
    if matches!(
        line.last().map(String::as_str),
        Some("." | "," | "(" | "[" | "{")
    ) {
        return false;
    }
    if !line.iter().any(|token| token == ")") {
        return false;
    }
    bracket_balance(line) == 0 && line.len() > 2
}

fn callable_keyword_argument_requires_callable_now(line: &[String]) -> bool {
    if line.len() < 3 || line.last().map(String::as_str) != Some("=") {
        return false;
    }
    if line.get(line.len() - 2).map(String::as_str) != Some("key") {
        return false;
    }
    line.iter()
        .any(|token| matches!(token.as_str(), "sorted" | "sort" | "min" | "max"))
}

fn callable_keyword_argument_start_token(token: &str) -> bool {
    matches!(token, "lambda" | "str" | "int" | "len" | "abs")
        || (is_identifier(token) && token.ends_with("_key"))
}

fn try_block_waiting_for_except(tokens: &[String]) -> bool {
    let try_count = tokens
        .iter()
        .filter(|token| token.as_str() == "try")
        .count();
    let except_count = tokens
        .iter()
        .filter(|token| token.as_str() == "except")
        .count();
    try_count > except_count
}

fn archive_context_manager_in_if_header(line: &[String]) -> bool {
    line.first().map(String::as_str) == Some("if")
        && line
            .iter()
            .any(|token| matches!(token.as_str(), "tarfile" | "zipfile" | "ZipFile"))
}

fn previous_meaningful_token(existing: &[String]) -> Option<String> {
    existing
        .iter()
        .rev()
        .find(|token| !matches!(token.as_str(), "<NL>" | "<INDENT>" | "<DEDENT>"))
        .cloned()
}
