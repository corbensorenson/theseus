fn expected_return_shapes(task: &Task) -> BTreeSet<String> {
    if let Some(contract) = task.raw.get("decoder_contract").and_then(Value::as_object) {
        if let Some(shape) = contract.get("return_shape").and_then(Value::as_str) {
            let normalized = shape.trim().to_lowercase();
            if matches!(
                normalized.as_str(),
                "bool" | "list" | "tuple" | "dict" | "set" | "str" | "number"
            ) {
                return [normalized].into_iter().collect();
            }
        }
    }
    if let Some(shape) = prompt_type_annotation_return_shape(&task.prompt) {
        return [shape].into_iter().collect();
    }
    let text = normalize_words(&format!("{} {}", task.entry_point, task.prompt));
    let mut shapes = BTreeSet::new();
    let bool_from_name = task.entry_point.to_lowercase().starts_with("is_")
        || task.entry_point.to_lowercase().starts_with("has_")
        || task.entry_point.to_lowercase().starts_with("can_")
        || task.entry_point.to_lowercase().starts_with("are_");
    let bool_from_text = contains_any(
        &text,
        &[
            "check whether",
            "check if",
            "whether",
            "true or false",
            "return true",
            "return false",
            "is valid",
            "are valid",
        ],
    );
    if bool_from_text || bool_from_name {
        shapes.insert("bool".to_string());
    }
    if contains_any(
        &text,
        &[
            "return a list",
            "return list",
            "returns list",
            "returns a list",
            "returns an empty list",
            "return only positive numbers",
            "paths to the split files",
            "find all",
            "all elements",
            "list of",
            "array",
            "remove",
            "sort",
            "filter",
            "tuples which",
            "given list",
            "from the list",
        ],
    ) {
        shapes.insert("list".to_string());
    }
    if contains_any(
        &text,
        &["return a tuple", "return tuple", "find tuples", "tuple of"],
    ) {
        shapes.insert("tuple".to_string());
    }
    if contains_any(
        &text,
        &[
            "return a dictionary",
            "return dictionary",
            "dict",
            "frequency",
            "mapping",
        ],
    ) {
        shapes.insert("dict".to_string());
    }
    if contains_any(&text, &["return a set", "return set", "unique elements"]) {
        shapes.insert("set".to_string());
    }
    if contains_any(
        &text,
        &[
            "return a string",
            "return string",
            "returns string",
            "returns a string",
            "returns decoded string",
            "returns encoded string",
            "returns str",
            "return str",
            "returns the path",
            "returns path",
            "path to the backup file",
            "path to the generated zip file",
            "base64 encoded encrypted message",
            "encrypted message",
            "character made",
            "ascii value",
            "text",
            "sentence",
            "word",
            "substring",
            "binary string",
            "roman",
        ],
    ) {
        shapes.insert("str".to_string());
    }
    if contains_any_numeric_phrase(
        &text,
        &[
            "count",
            "number of",
            "sum",
            "maximum",
            "minimum",
            "largest",
            "smallest",
            "area",
            "volume",
            "product",
            "gcd",
            "lcm",
            "index",
            "nth",
            "average",
            "mean",
            "difference",
            "distance",
            "length of given string",
            "return length",
            "closest smaller number",
            "calculate the value",
            "value of",
            "multiply all the numbers",
            "divide with the length",
            "n-th number",
            "n th number",
        ],
    ) {
        shapes.insert("number".to_string());
    }
    if shapes.contains("bool")
        && shapes.contains("number")
        && !bool_from_name
        && numeric_output_contract_overrides_weak_bool(&text)
    {
        shapes.remove("bool");
    }
    if shapes.contains("str")
        && shapes.contains("number")
        && string_output_contract_overrides_numeric(&text)
    {
        shapes.remove("number");
    }
    if shapes.is_empty() && task.entry_point == "solve" {
        return ["str".to_string()].into_iter().collect();
    }
    if shapes.contains("bool") {
        return ["bool".to_string()].into_iter().collect();
    }
    shapes
}

fn prompt_type_annotation_return_shape(prompt: &str) -> Option<String> {
    let lower = prompt.to_lowercase();
    let bytes = lower.as_bytes();
    let mut cursor = 0usize;
    while let Some(offset) = lower[cursor..].find("->") {
        let start = cursor + offset + 2;
        let end = bytes[start..]
            .iter()
            .position(|ch| matches!(*ch as char, ':' | '\n' | '\r'))
            .map(|pos| start + pos)
            .unwrap_or(lower.len());
        if let Some(shape) = type_annotation_return_shape(&lower[start..end]) {
            return Some(shape);
        }
        cursor = end.saturating_add(1);
        if cursor >= lower.len() {
            break;
        }
    }
    if let Some(shape) = returns_block_type_shape(&lower) {
        return Some(shape);
    }
    if let Some(offset) = lower.find("returns:") {
        let start = offset + "returns:".len();
        let end = bytes[start..]
            .iter()
            .position(|ch| matches!(*ch as char, '\n' | '\r' | '.'))
            .map(|pos| start + pos)
            .unwrap_or(lower.len());
        if let Some(shape) =
            type_annotation_return_shape(lower[start..end].trim_start_matches(&[' ', '-'][..]))
        {
            return Some(shape);
        }
    }
    None
}

fn returns_block_type_shape(lower_prompt: &str) -> Option<String> {
    let lines: Vec<&str> = lower_prompt.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        if line.trim() != "returns:" && line.trim() != "return:" {
            continue;
        }
        for candidate in lines.iter().skip(index + 1).take(7) {
            let cleaned = candidate.trim().trim_start_matches(&['-', '*', ' '][..]);
            if cleaned.is_empty() {
                continue;
            }
            let typed_prefix = cleaned.split(':').next().unwrap_or(cleaned);
            if let Some(shape) = type_annotation_return_shape(typed_prefix) {
                return Some(shape);
            }
            if let Some(shape) = type_annotation_return_shape(cleaned) {
                return Some(shape);
            }
        }
        return None;
    }
    None
}

fn type_annotation_return_shape(annotation: &str) -> Option<String> {
    let normalized = normalize_phrase_words(annotation);
    if normalized.is_empty() {
        return None;
    }
    if contains_any(&normalized, &["bool", "boolean"]) {
        return Some("bool".to_string());
    }
    if contains_any(&normalized, &["list", "array", "sequence"]) {
        return Some("list".to_string());
    }
    if normalized.contains("tuple") {
        return Some("tuple".to_string());
    }
    if contains_any(&normalized, &["dict", "dictionary", "mapping"]) {
        return Some("dict".to_string());
    }
    if normalized.contains("set") {
        return Some("set".to_string());
    }
    if contains_any(&normalized, &["str", "string", "path", "message"]) {
        return Some("str".to_string());
    }
    if contains_any(&normalized, &["int", "float", "number", "numeric"]) {
        return Some("number".to_string());
    }
    None
}

fn numeric_output_contract_overrides_weak_bool(text: &str) -> bool {
    contains_any_numeric_phrase(
        text,
        &[
            "print the maximum",
            "print maximum",
            "print the minimum",
            "print minimum",
            "print the largest",
            "print largest",
            "print the smallest",
            "print smallest",
            "print the number of",
            "print number of",
            "print the count",
            "print count",
            "find the maximum",
            "find maximum",
            "find the minimum",
            "find minimum",
            "maximum number",
            "minimum number",
            "maximum value",
            "minimum value",
            "output the maximum",
            "output maximum",
            "output the minimum",
            "output minimum",
            "otherwise print -1",
            "print -1",
        ],
    )
}

fn string_output_contract_overrides_numeric(text: &str) -> bool {
    contains_any(
        text,
        &[
            "character made",
            "ascii value",
            "return character",
            "return a character",
        ],
    )
}

fn required_structures(task: &Task) -> BTreeSet<String> {
    let text = normalize_words(&format!("{} {}", task.entry_point, task.prompt));
    let mut required = BTreeSet::new();
    if let Some(contract) = task.raw.get("decoder_contract").and_then(Value::as_object) {
        if let Some(items) = contract
            .get("required_constructs")
            .and_then(Value::as_array)
        {
            for item in items {
                let Some(text) = item.as_str() else {
                    continue;
                };
                match text.trim().to_lowercase().as_str() {
                    "loop" => {
                        required.insert("iteration".to_string());
                    }
                    "branch" => {
                        required.insert("conditional".to_string());
                    }
                    "collection_ops" => {
                        required.insert("collection_build".to_string());
                    }
                    "dict" => {
                        required.insert("collection_build".to_string());
                    }
                    "parsing" => {
                        required.insert("string_processing".to_string());
                    }
                    "numeric_ops" => {
                        required.insert("numeric_aggregation".to_string());
                    }
                    "stdin_parse" | "string_join_return" => {
                        required.insert("string_processing".to_string());
                    }
                    "algorithmic_planning" | "graph" => {
                        required.insert("iteration".to_string());
                        required.insert("conditional".to_string());
                    }
                    "composition" => {
                        required.insert("composition".to_string());
                    }
                    _ => {}
                }
            }
        }
    }
    if contains_any(
        &text,
        &[
            "list",
            "array",
            "tuple",
            "each",
            "every",
            "all elements",
            "elements",
            "filter",
            "sort",
            "remove",
            "find all",
            "strings",
            "numbers",
            "items",
            "pairs",
            "subsequence",
            "subarray",
        ],
    ) {
        required.insert("iteration".to_string());
    }
    if contains_any(
        &text,
        &[
            "if",
            "whether",
            "check",
            "valid",
            "invalid",
            "positive",
            "negative",
            "empty",
            "non empty",
            "divisible",
            "greater than",
            "less than",
            "same",
            "different",
        ],
    ) || task.entry_point.to_lowercase().starts_with("is_")
        || task.entry_point.to_lowercase().starts_with("has_")
        || task.entry_point.to_lowercase().starts_with("can_")
        || task.entry_point.to_lowercase().starts_with("are_")
    {
        required.insert("conditional".to_string());
    }
    if contains_any(
        &text,
        &[
            "string",
            "substring",
            "word",
            "character",
            "text",
            "sentence",
            "roman",
            "binary string",
        ],
    ) {
        required.insert("string_processing".to_string());
    }
    if contains_any(
        &text,
        &[
            "return a list",
            "return list",
            "list of",
            "dictionary",
            "dict",
            "mapping",
            "frequency",
            "return a set",
            "unique",
            "return tuple",
            "tuple of",
        ],
    ) {
        required.insert("collection_build".to_string());
    }
    if contains_any_numeric_phrase(
        &text,
        &[
            "count",
            "number of",
            "sum",
            "maximum",
            "minimum",
            "largest",
            "smallest",
            "product",
            "average",
            "mean",
            "gcd",
            "lcm",
            "difference",
            "distance",
        ],
    ) {
        required.insert("numeric_aggregation".to_string());
    }
    required
}

fn expression_return_shapes(expr: &str) -> BTreeSet<String> {
    let normalized = normalize_expression(expr);
    let lowered = normalized.to_lowercase();
    let trimmed = normalized.trim();
    let mut shapes = BTreeSet::new();
    if trimmed.is_empty() {
        shapes.insert("unknown".to_string());
        return shapes;
    }
    if let Some((then_expr, else_expr)) = top_level_conditional_branches(trimmed) {
        shapes.extend(expression_return_shapes(then_expr));
        shapes.extend(expression_return_shapes(else_expr));
        return concrete_shapes_or_unknown(shapes);
    }
    if trimmed == "true"
        || trimmed == "false"
        || lowered.starts_with("not ")
        || lowered.starts_with("any (")
        || lowered.starts_with("all (")
        || lowered.starts_with("isinstance (")
        || has_top_level_comparison(&lowered)
        || has_top_level_bool_method_call(&lowered)
    {
        shapes.insert("bool".to_string());
    }
    if trimmed.starts_with('[') || lowered.starts_with("list (") || lowered.starts_with("sorted (")
    {
        shapes.insert("list".to_string());
    }
    if trimmed.starts_with('(') || lowered.starts_with("tuple (") {
        shapes.insert("tuple".to_string());
    }
    if trimmed.starts_with('{') || lowered.starts_with("dict (") {
        shapes.insert("dict".to_string());
    }
    if lowered.starts_with("set (") {
        shapes.insert("set".to_string());
    }
    if trimmed.starts_with('"')
        || trimmed.starts_with('\'')
        || lowered.starts_with("str (")
        || lowered.starts_with("chr (")
        || lowered.contains(".join")
    {
        shapes.insert("str".to_string());
    }
    if lowered.starts_with("int (")
        || lowered.starts_with("float (")
        || lowered.starts_with("len (")
        || lowered.starts_with("sum (")
        || lowered.starts_with("max (")
        || lowered.starts_with("min (")
        || lowered.starts_with("abs (")
        || lowered.starts_with("round (")
        || trimmed.parse::<f64>().is_ok()
        || trimmed.replace(' ', "").parse::<f64>().is_ok()
    {
        shapes.insert("number".to_string());
    }
    if shapes.is_empty() {
        shapes.insert("unknown".to_string());
    }
    shapes
}

fn body_return_shapes(body: &str) -> BTreeSet<String> {
    let mut local_shapes: HashMap<String, BTreeSet<String>> = HashMap::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some((name, expr)) = simple_assignment(trimmed) {
            let shapes = expression_return_shapes_with_locals(expr, &local_shapes);
            if !shapes.is_empty() {
                let preserve_self_shape = shapes.len() == 1
                    && shapes.contains("unknown")
                    && expr
                        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
                        .any(|token| token == name);
                if preserve_self_shape {
                    if let Some(previous) = local_shapes.get(name).cloned() {
                        local_shapes.insert(name.to_string(), previous);
                        continue;
                    }
                }
                local_shapes.insert(name.to_string(), shapes);
            }
        }
    }
    let mut shapes = BTreeSet::new();
    let mut return_exprs = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim_start();
        if let Some(expr) = trimmed.strip_prefix("return ") {
            return_exprs.push(expr.to_string());
            shapes.extend(expression_return_shapes_with_locals(expr, &local_shapes));
        }
    }
    let mut shapes = concrete_shapes_or_unknown(shapes);
    if shapes.contains("unknown")
        && return_exprs
            .iter()
            .any(|expr| code_has_subscript_expression(expr))
        && body_structures(body).contains("numeric_aggregation")
    {
        shapes.remove("unknown");
        shapes.insert("number".to_string());
    }
    shapes
}

fn expression_return_shapes_with_locals(
    expr: &str,
    local_shapes: &HashMap<String, BTreeSet<String>>,
) -> BTreeSet<String> {
    let raw_trimmed = expr.trim();
    if is_identifier(raw_trimmed) {
        if let Some(shapes) = local_shapes.get(raw_trimmed) {
            return shapes.clone();
        }
    }
    let normalized = normalize_expression(expr);
    if let Some((then_expr, else_expr)) = top_level_conditional_branches(&normalized) {
        let mut shapes = BTreeSet::new();
        shapes.extend(expression_return_shapes_with_locals(
            then_expr,
            local_shapes,
        ));
        shapes.extend(expression_return_shapes_with_locals(
            else_expr,
            local_shapes,
        ));
        return concrete_shapes_or_unknown(shapes);
    }
    if is_identifier(&normalized) {
        if let Some(shapes) = local_shapes.get(&normalized) {
            return shapes.clone();
        }
    }
    expression_return_shapes(expr)
}

fn concrete_shapes_or_unknown(mut shapes: BTreeSet<String>) -> BTreeSet<String> {
    if shapes.len() > 1 {
        shapes.remove("unknown");
    }
    if shapes.is_empty() {
        shapes.insert("unknown".to_string());
    }
    shapes
}

fn top_level_conditional_branches(expr: &str) -> Option<(&str, &str)> {
    let if_idx = find_top_level_keyword(expr, "if")?;
    let else_idx = find_top_level_keyword(&expr[if_idx + 4..], "else")? + if_idx + 4;
    let then_expr = expr[..if_idx].trim();
    let else_expr = expr[else_idx + 6..].trim();
    if then_expr.is_empty() || else_expr.is_empty() {
        return None;
    }
    Some((then_expr, else_expr))
}

fn find_top_level_keyword(expr: &str, keyword: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let pattern = format!(" {keyword} ");
    for (idx, ch) in expr.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == q {
                quote = None;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            continue;
        }
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = (depth - 1).max(0),
            _ => {}
        }
        if depth == 0 && expr[idx..].starts_with(&pattern) {
            return Some(idx);
        }
    }
    None
}

fn simple_assignment(line: &str) -> Option<(&str, &str)> {
    if line.starts_with("return ")
        || line.starts_with("if ")
        || line.starts_with("elif ")
        || line.starts_with("for ")
        || line.starts_with("while ")
        || line.starts_with("except ")
        || line.starts_with("with ")
        || line.contains("==")
        || line.contains("!=")
        || line.contains(">=")
        || line.contains("<=")
    {
        return None;
    }
    let (left, right) = line.split_once(" = ")?;
    let name = left.split(':').next().unwrap_or(left).trim();
    let expr = right.trim();
    if name.is_empty()
        || expr.is_empty()
        || name.contains(',')
        || name.contains('.')
        || name.contains('[')
        || !is_identifier(name)
    {
        return None;
    }
    Some((name, expr))
}

fn body_structures(body: &str) -> BTreeSet<String> {
    let analysis_body = structure_analysis_text(body);
    let lowered = analysis_body.to_lowercase();
    let mut structures = BTreeSet::new();
    if contains_any(
        &lowered,
        &[
            "\nfor ",
            "\nwhile ",
            " for ",
            ".items()",
            "enumerate(",
            "sorted(",
            ".sort(",
        ],
    ) || lowered.trim_start().starts_with("for ")
        || lowered.trim_start().starts_with("while ")
    {
        structures.insert("iteration".to_string());
    }
    if contains_any(
        &lowered,
        &["\nif ", "\nelif ", "\nelse:", " if ", " and ", " or "],
    ) || lowered.trim_start().starts_with("if ")
        || code_has_comparison_operator(&analysis_body)
    {
        structures.insert("conditional".to_string());
    }
    if contains_any(
        &lowered,
        &[
            ".split",
            ".join",
            ".strip",
            ".lstrip",
            ".rstrip",
            ".lower",
            ".upper",
            ".replace",
            ".find",
            ".count",
            ".startswith",
            ".endswith",
            ".isdigit",
            ".isalpha",
            ".isalnum",
            "str(",
            "chr(",
            "ord(",
        ],
    ) {
        structures.insert("string_processing".to_string());
    }
    if contains_any(
        &lowered,
        &[
            "out = []", " = []", " = [", " = {}", " = {", "return [", "return {", "return (",
            " else []", " else [", "set(", "dict(", "list(", "tuple(", "sorted(", ".append",
            ".add",
        ],
    ) {
        structures.insert("collection_build".to_string());
    }
    if contains_any(
        &lowered,
        &[
            "total", "count", "sum(", "max(", "min(", "abs(", "round(", "+=", "-=", "*=", "len(",
        ],
    ) || code_has_arithmetic_operator(&analysis_body)
    {
        structures.insert("numeric_aggregation".to_string());
    }
    structures
}

fn structure_analysis_text(code: &str) -> String {
    code.lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !(trimmed.starts_with("import ")
                || (trimmed.starts_with("from ") && trimmed.contains(" import ")))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn code_has_comparison_operator(code: &str) -> bool {
    let chars = code.chars().collect::<Vec<_>>();
    let mut idx = 0usize;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    while idx < chars.len() {
        let ch = chars[idx];
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == q {
                quote = None;
            }
            idx += 1;
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            idx += 1;
            continue;
        }
        let next = chars.get(idx + 1).copied();
        if matches!(ch, '=' | '!' | '<' | '>') && next == Some('=') {
            return true;
        }
        if matches!(ch, '<' | '>') && next != Some('-') {
            return true;
        }
        idx += 1;
    }
    false
}

fn code_has_arithmetic_operator(code: &str) -> bool {
    let chars = code.chars().collect::<Vec<_>>();
    let mut idx = 0usize;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    while idx < chars.len() {
        let ch = chars[idx];
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == q {
                quote = None;
            }
            idx += 1;
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            idx += 1;
            continue;
        }
        if matches!(ch, '*' | '/' | '%' | '+') {
            return true;
        }
        if ch == '-' && arithmetic_minus_context(&chars, idx) {
            return true;
        }
        idx += 1;
    }
    false
}

fn arithmetic_minus_context(chars: &[char], idx: usize) -> bool {
    let prev = previous_non_space_with_index(chars, idx).map(|(_, ch)| ch);
    let next = chars
        .get(idx + 1..)
        .and_then(|tail| tail.iter().find(|ch| !ch.is_whitespace()).copied());
    match (prev, next) {
        (Some(left), Some(right)) => {
            (left.is_ascii_alphanumeric() || matches!(left, ')' | ']'))
                && (right.is_ascii_alphanumeric() || matches!(right, '(' | '['))
        }
        _ => false,
    }
}

fn structure_coverage_score(
    required: &BTreeSet<String>,
    actual: &BTreeSet<String>,
    expected_shapes: &BTreeSet<String>,
) -> usize {
    required
        .iter()
        .filter(|name| actual.contains(*name))
        .map(|name| match name.as_str() {
            "collection_build" => {
                if expected_shapes
                    .iter()
                    .any(|shape| matches!(shape.as_str(), "list" | "dict" | "set" | "tuple"))
                {
                    2600
                } else {
                    2100
                }
            }
            "string_processing" => 2400,
            "numeric_aggregation" => 1700,
            "composition" => 3200,
            "conditional" => 1500,
            "iteration" => 1500,
            _ => 1200,
        })
        .sum()
}

fn has_top_level_comparison(expr: &str) -> bool {
    let mut depth = 0i32;
    let mut chars = expr.chars().peekable();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    while let Some(ch) = chars.next() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == q {
                quote = None;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            continue;
        }
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = (depth - 1).max(0),
            '=' | '!' | '<' | '>' if depth == 0 => {
                let next = chars.peek().copied();
                if matches!(ch, '=' | '!' | '<' | '>') && next == Some('=') {
                    return true;
                }
                if matches!(ch, '<' | '>') {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn has_top_level_bool_method_call(expr: &str) -> bool {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (idx, ch) in expr.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == q {
                quote = None;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            continue;
        }
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = (depth - 1).max(0),
            '.' if depth == 0 => {
                let tail = &expr[idx..];
                if tail.starts_with(".startswith")
                    || tail.starts_with(".endswith")
                    || tail.starts_with(".isdigit")
                    || tail.starts_with(".isalpha")
                    || tail.starts_with(".isalnum")
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn return_shape_compatible(expected: &BTreeSet<String>, actual: &BTreeSet<String>) -> bool {
    if expected.is_empty() {
        return true;
    }
    if actual.contains("unknown") {
        return false;
    }
    if expected.iter().any(|shape| actual.contains(shape)) {
        return true;
    }
    (expected.contains("list") && actual.contains("tuple"))
        || (expected.contains("tuple") && actual.contains("list"))
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn contains_any_numeric_phrase(text: &str, phrases: &[&str]) -> bool {
    let padded = format!(" {} ", normalize_phrase_words(text));
    phrases.iter().any(|phrase| {
        let normalized = normalize_phrase_words(phrase);
        !normalized.is_empty() && padded.contains(&format!(" {} ", normalized))
    })
}

fn normalize_phrase_words(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_words(value: &str) -> String {
    value
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn generate_ngram_expression(
    model: &TokenModel,
    prompt_tokens: &BTreeSet<String>,
    seed: u64,
) -> Option<String> {
    let mut out = Vec::new();
    let mut prev = "return".to_string();
    for step in 0..18usize {
        let choices = model.bigram.get(&prev)?;
        let mut ranked = choices
            .iter()
            .map(|(token, count)| {
                let prompt_boost = if prompt_tokens.contains(&token.to_lowercase()) {
                    200usize
                } else {
                    0usize
                };
                let score = *count + prompt_boost;
                let tie = stable_hash_u64(&format!("{}:{}:{}:{}", seed, step, prev, token));
                (score, tie, token.clone())
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let token = ranked.first()?.2.clone();
        if token == ":" || token == ";" || token == "def" || token == "class" {
            break;
        }
        out.push(token.clone());
        prev = token;
    }
    let expr = normalize_expression(&join_tokens(&out));
    useful_expression(&expr).then_some(expr)
}

fn signature_args(task: &Task) -> Vec<String> {
    let public_unknown = public_or_unknown_signature_task(task);
    if let Some(args) = args_from_decoder_contract(&task.raw) {
        if args.len() == 1 {
            if let Some(prompt_args) = args_from_visible_prompt_semantics(&task.prompt) {
                return prompt_args;
            }
        }
        return args;
    }
    if let Some(args) = args_from_prompt_signature(&task.prompt, &task.entry_point) {
        return args;
    }
    if let Some(args) = args_from_visible_prompt_semantics(&task.prompt) {
        return args;
    }
    let hint = task
        .raw
        .get("decoder_contract")
        .and_then(|contract| contract.get("visible_arg_count_hint"))
        .and_then(Value::as_u64)
        .unwrap_or(1) as usize;
    let count = hint.max(1);
    if public_unknown {
        return args_for_count(count);
    }
    args_for_count(count)
}

fn public_or_unknown_signature_task(task: &Task) -> bool {
    let evidence = task.benchmark_evidence_level.to_ascii_lowercase();
    let case_type = task.case_type.to_ascii_lowercase();
    evidence.contains("public_benchmark") || case_type.starts_with("public_")
}

fn args_from_visible_prompt_semantics(prompt: &str) -> Option<Vec<String>> {
    let text = normalize_phrase_words(prompt);
    if phrase_any(
        &text,
        &[
            "for two strings",
            "of two strings",
            "two string",
            "pair of strings",
            "two sequences",
        ],
    ) {
        return Some(vec!["data".to_string(), "other".to_string()]);
    }
    None
}

fn args_from_decoder_contract(raw: &Value) -> Option<Vec<String>> {
    let roles = raw
        .get("decoder_contract")
        .and_then(|contract| contract.get("argument_roles"))
        .and_then(Value::as_object)?;
    if roles.is_empty() {
        return None;
    }
    let mut args = Vec::new();
    for preferred in ["data", "other", "extra"] {
        if roles.contains_key(preferred) {
            args.push(preferred.to_string());
        }
    }
    let mut rest = roles
        .keys()
        .filter(|key| !["data", "other", "extra"].contains(&key.as_str()))
        .map(|key| sanitize_ident(key))
        .collect::<Vec<_>>();
    rest.sort();
    args.extend(rest);
    Some(unique_args(args))
}

fn args_from_prompt_signature(prompt: &str, entry_point: &str) -> Option<Vec<String>> {
    let entry = sanitize_ident(entry_point);
    let exact_marker = format!("def {}(", entry);
    if let Some(start) = prompt.find(&exact_marker) {
        let args_start = start + exact_marker.len();
        let rest = &prompt[args_start..];
        let args_end = rest.find(')')?;
        let raw_args = &rest[..args_end];
        let args = raw_args
            .split(',')
            .filter_map(clean_signature_arg)
            .collect::<Vec<_>>();
        if !args.is_empty() {
            return Some(unique_args(args));
        }
    }
    if let Some(start) = prompt.find("def ") {
        let rest_after_def = &prompt[start + 4..];
        let open = rest_after_def.find('(')?;
        let rest = &rest_after_def[open + 1..];
        let args_end = rest.find(')')?;
        let raw_args = &rest[..args_end];
        let args = raw_args
            .split(',')
            .filter_map(clean_signature_arg)
            .collect::<Vec<_>>();
        if !args.is_empty() {
            return Some(unique_args(args));
        }
    }
    None
}

fn clean_signature_arg(raw: &str) -> Option<String> {
    let mut text = raw.trim();
    if text.is_empty() || text == "/" || text == "*" {
        return None;
    }
    while text.starts_with('*') {
        text = text.trim_start_matches('*').trim();
    }
    let head = text
        .split('=')
        .next()
        .unwrap_or(text)
        .split(':')
        .next()
        .unwrap_or(text)
        .trim();
    if head.is_empty() {
        return None;
    }
    let ident = sanitize_ident(head);
    (!ident.is_empty()).then_some(ident)
}

fn args_for_count(count: usize) -> Vec<String> {
    let names = ["data", "other", "extra"];
    (0..count.max(1))
        .map(|idx| {
            names
                .get(idx)
                .map(|value| value.to_string())
                .unwrap_or_else(|| format!("arg{}", idx + 1))
        })
        .collect()
}

fn unique_args(args: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (idx, arg) in args.into_iter().enumerate() {
        if arg.trim() == "*args" {
            if seen.insert("args".to_string()) {
                out.push("*args".to_string());
            }
            continue;
        }
        let mut ident = sanitize_ident(&arg);
        if ident.is_empty() {
            ident = format!("arg{}", idx + 1);
        }
        if seen.insert(ident.clone()) {
            out.push(ident);
        }
    }
    if out.is_empty() {
        out.push("data".to_string());
    }
    out
}

fn program_synthesis_loop() -> Value {
    json!({
        "policy": "project_theseus_program_synthesis_loop_v1",
        "loop_shape": [
            "visible_contract_parse",
            "constrained_token_decode",
            "parser_contract_mask",
            "expression_alias_guardrail",
            "full_body_emit",
            "static_guardrail"
        ],
        "decode_control": {
            "constrained_token_decode": true,
            "parser_contract_mask": true,
            "exact_interface_claim": true,
            "grammar_masked_learned_token_candidate": true,
            "template_or_memory_fallback": false
        },
        "public_tests_used": false,
        "canonical_solution_used": false,
        "external_inference_calls": 0
    })
}

fn count_bool(rows: &[Value], key: &str) -> usize {
    rows.iter()
        .filter(|row| row.get(key).and_then(Value::as_bool) == Some(true))
        .count()
}

fn count_string(rows: &[Value], key: &str, expected: &str) -> usize {
    rows.iter()
        .filter(|row| row.get(key).and_then(Value::as_str) == Some(expected))
        .count()
}

fn expression_static_guardrail_reasons(expr: &str, args: &[String]) -> Vec<String> {
    let params = unique_args(args.to_vec());
    let uses_varargs = params.iter().any(|arg| arg == "*args");
    if uses_varargs {
        return vec!["erased_varargs_signature".to_string()];
    }
    let mut allowed = [
        "None",
        "True",
        "False",
        "len",
        "sum",
        "min",
        "max",
        "sorted",
        "list",
        "tuple",
        "set",
        "dict",
        "str",
        "int",
        "float",
        "bool",
        "range",
        "enumerate",
        "zip",
        "any",
        "all",
        "abs",
        "round",
        "reversed",
    ]
    .into_iter()
    .map(String::from)
    .collect::<BTreeSet<_>>();
    for param in &params {
        if param == "*args" {
            allowed.insert("args".to_string());
        } else {
            allowed.insert(param.clone());
        }
    }
    if uses_varargs {
        allowed.insert("args".to_string());
        allowed.insert("data".to_string());
        allowed.insert("other".to_string());
        allowed.insert("extra".to_string());
    }
    if !params.is_empty() {
        allowed.insert("data".to_string());
    }
    if params.len() >= 2 {
        allowed.insert("other".to_string());
    }
    if params.len() >= 3 {
        allowed.insert("extra".to_string());
    }

    let tokens = tokenize_code(expr);
    for idx in 0..tokens.len().saturating_sub(2) {
        if tokens[idx] == "for" {
            for target in loop_target_identifiers(&tokens, idx + 1) {
                allowed.insert(target);
            }
        }
    }

    let mut reasons = BTreeSet::new();
    for (idx, token) in tokens.iter().enumerate() {
        if !is_identifier(token) {
            continue;
        }
        if idx > 0 && tokens.get(idx - 1).map(String::as_str) == Some(".") {
            continue;
        }
        if allowed.contains(token) {
            continue;
        }
        if token
            .chars()
            .next()
            .map(|ch| ch.is_ascii_uppercase())
            .unwrap_or(false)
        {
            continue;
        }
        reasons.insert(format!("undefined_identifier:{token}"));
    }
    reasons.into_iter().collect()
}

fn body_static_guardrail_reasons(body: &str, args: &[String]) -> Vec<String> {
    let params = unique_args(args.to_vec());
    let uses_varargs = params.iter().any(|arg| arg == "*args");
    let mut reasons = BTreeSet::new();
    if uses_varargs {
        reasons.insert("erased_varargs_signature".to_string());
    }
    if generator_scaffold_placeholder_body(body) {
        reasons.insert("placeholder_scaffold_body".to_string());
    }
    let Some(imported) = safe_imported_identifiers(body) else {
        return vec!["unsafe_import".to_string()];
    };
    let mut allowed = [
        "None",
        "True",
        "False",
        "len",
        "sum",
        "min",
        "max",
        "sorted",
        "list",
        "tuple",
        "dict",
        "set",
        "str",
        "int",
        "float",
        "bool",
        "range",
        "enumerate",
        "zip",
        "any",
        "all",
        "abs",
        "round",
        "reversed",
        "isinstance",
        "for",
        "if",
        "elif",
        "else",
        "while",
        "return",
        "in",
        "not",
        "and",
        "or",
        "is",
        "continue",
        "break",
        "try",
        "except",
        "Exception",
        "def",
        "e",
    ]
    .into_iter()
    .map(String::from)
    .collect::<BTreeSet<_>>();
    for param in &params {
        if param == "*args" {
            allowed.insert("args".to_string());
        } else {
            allowed.insert(param.clone());
        }
    }
    allowed.extend(imported);
    if uses_varargs {
        allowed.insert("args".to_string());
        allowed.insert("data".to_string());
        allowed.insert("other".to_string());
        allowed.insert("extra".to_string());
    }
    if !params.is_empty() {
        allowed.insert("data".to_string());
    }
    if params.len() >= 2 {
        allowed.insert("other".to_string());
    }
    if params.len() >= 3 {
        allowed.insert("extra".to_string());
    }
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some((left, _)) = trimmed.split_once(" = ") {
            for target in assignment_target_identifiers(left) {
                allowed.insert(target);
            }
        }
        if let Some((name, params)) = nested_function_signature(trimmed) {
            allowed.insert(name);
            for param in params {
                allowed.insert(param);
            }
        }
    }

    let guardrail_body = structure_analysis_text(body);
    let tokens = tokenize_code(&guardrail_body);
    for idx in 0..tokens.len() {
        let token = &tokens[idx];
        if token == "for" {
            for target in loop_target_identifiers(&tokens, idx + 1) {
                allowed.insert(target);
            }
        }
        if is_identifier(token) && tokens.get(idx + 1).map(String::as_str) == Some("=") {
            allowed.insert(token.clone());
        }
        if is_identifier(token)
            && matches!(
                tokens.get(idx + 1).map(String::as_str),
                Some("+") | Some("-") | Some("*") | Some("/")
            )
            && tokens.get(idx + 2).map(String::as_str) == Some("=")
        {
            allowed.insert(token.clone());
        }
        if token == "lambda" && idx + 1 < tokens.len() && is_identifier(&tokens[idx + 1]) {
            allowed.insert(tokens[idx + 1].clone());
        }
        if token == "except" && idx + 2 < tokens.len() && tokens[idx + 1] == "Exception" {
            allowed.insert(tokens[idx + 2].clone());
        }
    }

    for (idx, token) in tokens.iter().enumerate() {
        if !is_identifier(token) {
            continue;
        }
        if idx > 0 && tokens.get(idx - 1).map(String::as_str) == Some(".") {
            continue;
        }
        if allowed.contains(token) {
            continue;
        }
        if token
            .chars()
            .next()
            .map(|ch| ch.is_ascii_uppercase())
            .unwrap_or(false)
        {
            continue;
        }
        reasons.insert(format!("undefined_identifier:{token}"));
    }
    reasons.into_iter().collect()
}

fn generator_scaffold_placeholder_body(body: &str) -> bool {
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

fn loop_target_identifiers(tokens: &[String], start: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut idx = start;
    while idx < tokens.len() {
        let token = &tokens[idx];
        if token == "in" {
            break;
        }
        if is_identifier(token) {
            out.push(token.clone());
        }
        idx += 1;
    }
    out
}

fn assignment_target_identifiers(left: &str) -> Vec<String> {
    left.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|part| {
            let name = part.trim();
            if is_identifier(name) {
                Some(name.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn nested_function_signature(trimmed: &str) -> Option<(String, Vec<String>)> {
    let rest = trimmed.strip_prefix("def ")?;
    let name_end = rest.find('(')?;
    let name = rest[..name_end].trim();
    if !is_identifier(name) {
        return None;
    }
    let close = rest[name_end + 1..].find(')')? + name_end + 1;
    let params = rest[name_end + 1..close]
        .split(',')
        .filter_map(clean_signature_arg)
        .collect::<Vec<_>>();
    Some((name.to_string(), params))
}

fn render_body_candidate(entry_point: &str, args: &[String], body: &str) -> String {
    let entry = sanitize_ident(entry_point);
    let params = unique_args(args.to_vec());
    let mut lines = Vec::new();
    lines.push("from typing import *".to_string());
    lines.push(String::new());
    lines.push(format!("def {entry}({}):", params.join(", ")));
    for alias in body_alias_lines(&params) {
        lines.push(alias);
    }
    for raw_line in body.lines() {
        if raw_line.trim().is_empty() {
            continue;
        }
        lines.push(format!("    {}", raw_line));
    }
    lines.push(String::new());
    lines.join("\n")
}

fn render_candidate(entry_point: &str, args: &[String], expression: &str) -> String {
    let entry = sanitize_ident(entry_point);
    let params = unique_args(args.to_vec());
    let alias_lines = body_alias_lines(&params);
    let aliases = if alias_lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", alias_lines.join("\n"))
    };
    format!(
        "from typing import *\n\n\
def {entry}({params}):\n{aliases}    \
result = {expr}\n    \
return result\n",
        params = params.join(", "),
        expr = expression.trim()
    )
}

fn body_alias_lines(params: &[String]) -> Vec<String> {
    let mut alias_lines = Vec::new();
    if params.iter().any(|arg| arg == "*args") {
        alias_lines.push("    data = args[0] if len(args) > 0 else None".to_string());
        alias_lines.push("    other = args[1] if len(args) > 1 else None".to_string());
        alias_lines.push("    extra = args[2:] if len(args) > 2 else ()".to_string());
        return alias_lines;
    }
    if let Some(first) = params.first() {
        if first != "data" {
            alias_lines.push(format!("    data = {first}"));
        }
    }
    if let Some(second) = params.get(1) {
        if second != "other" {
            alias_lines.push(format!("    other = {second}"));
        }
    }
    if params.iter().any(|arg| arg == "start") && params.iter().any(|arg| arg == "goal") {
        alias_lines.push("    extra = (start, goal)".to_string());
    } else if let Some(third) = params.get(2) {
        if third != "extra" {
            alias_lines.push(format!("    extra = {third}"));
        }
    }
    alias_lines
}

fn normalize_expression(expr: &str) -> String {
    let tokens = tokenize_code(expr);
    if tokens.is_empty() {
        return "None".to_string();
    }
    let keep = [
        "None",
        "True",
        "False",
        "len",
        "sum",
        "min",
        "max",
        "sorted",
        "list",
        "tuple",
        "set",
        "str",
        "int",
        "float",
        "bool",
        "range",
        "enumerate",
        "zip",
        "any",
        "all",
        "abs",
        "round",
        "reversed",
        "data",
        "other",
        "extra",
        "for",
        "in",
        "if",
        "else",
        "and",
        "or",
        "not",
        "is",
    ];
    let keep: HashSet<&str> = keep.into_iter().collect();
    let mut vars: BTreeMap<String, String> = BTreeMap::new();
    let names = ["data", "other", "extra", "item", "value"];
    let mut normalized = Vec::new();
    for token in tokens {
        if is_identifier(&token)
            && !keep.contains(token.as_str())
            && !token.chars().next().unwrap_or('_').is_uppercase()
        {
            let next_name = names[vars.len().min(names.len() - 1)].to_string();
            let mapped = vars.entry(token).or_insert(next_name);
            normalized.push(mapped.clone());
        } else {
            normalized.push(token);
        }
    }
    let joined = join_tokens(&normalized);
    if joined.trim().is_empty() {
        "None".to_string()
    } else {
        joined
    }
}

fn useful_expression(expr: &str) -> bool {
    let trimmed = expr.trim();
    if trimmed.is_empty() || trimmed.len() > 160 {
        return false;
    }
    let compact = trimmed.replace(' ', "");
    let blocked_exact = [
        "None", "True", "False", "0", "1", "[]", "{}", "()", "''", "\"\"",
    ];
    if blocked_exact.contains(&compact.as_str()) {
        return false;
    }
    if tokenize_code(trimmed).iter().any(|token| token == "None") {
        return false;
    }
    let lowered = trimmed.to_lowercase();
    let blocked = [
        "open(",
        "exec(",
        "eval(",
        "__",
        "import ",
        "assert ",
        "sys.",
        "subprocess",
        "os.",
        "task_id",
        "canonical",
        "solution",
    ];
    !blocked.iter().any(|token| lowered.contains(token)) && delimiters_balanced(trimmed)
}

fn delimiters_balanced(text: &str) -> bool {
    let mut stack = Vec::new();
    for ch in text.chars() {
        match ch {
            '(' | '[' | '{' => stack.push(ch),
            ')' => {
                if stack.pop() != Some('(') {
                    return false;
                }
            }
            ']' => {
                if stack.pop() != Some('[') {
                    return false;
                }
            }
            '}' => {
                if stack.pop() != Some('{') {
                    return false;
                }
            }
            _ => {}
        }
    }
    stack.is_empty()
}

fn tokenize_code(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars = text.chars().collect::<Vec<_>>();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '"' || ch == '\'' {
            tokens.push("\"STR\"".to_string());
            i += 1;
            while i < chars.len() && chars[i] != ch {
                if chars[i] == '\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect::<String>());
            continue;
        }
        if ch.is_ascii_digit() {
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            tokens.push("0".to_string());
            continue;
        }
        if i + 1 < chars.len() {
            let two = format!("{}{}", chars[i], chars[i + 1]);
            if ["==", "!=", "<=", ">=", "//", "**", "->"].contains(&two.as_str()) {
                tokens.push(two);
                i += 2;
                continue;
            }
        }
        tokens.push(ch.to_string());
        i += 1;
    }
    tokens
}

fn tokenize_words(text: &str) -> Vec<String> {
    tokenize_code(text)
        .into_iter()
        .filter(|token| is_identifier(token))
        .map(|token| token.to_lowercase())
        .collect()
}

fn join_tokens(tokens: &[String]) -> String {
    let mut out = String::new();
    for token in tokens {
        let no_space_before = [")", "]", "}", ",", ".", ":"].contains(&token.as_str());
        let no_space_after_prev =
            out.ends_with('(') || out.ends_with('[') || out.ends_with('{') || out.ends_with('.');
        if !out.is_empty() && !no_space_before && !no_space_after_prev {
            out.push(' ');
        }
        out.push_str(token);
    }
    out
}

fn sanitize_ident(value: &str) -> String {
    let mut out = String::new();
    for (idx, ch) in value.chars().enumerate() {
        if (idx == 0 && (ch.is_ascii_alphabetic() || ch == '_'))
            || (idx > 0 && (ch.is_ascii_alphanumeric() || ch == '_'))
        {
            out.push(ch);
        }
    }
    if out.is_empty() {
        "solve".to_string()
    } else {
        out
    }
}

fn is_identifier(token: &str) -> bool {
    let mut chars = token.chars();
    match chars.next() {
        Some(ch) if ch.is_ascii_alphabetic() || ch == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn top_counts(counts: &HashMap<String, usize>, limit: usize) -> Vec<Value> {
    let mut rows = counts
        .iter()
        .map(|(token, count)| (token.clone(), *count))
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows.into_iter()
        .take(limit)
        .map(|(token, count)| json!({"token": token, "count": count}))
        .collect()
}

fn read_json(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(line) {
            rows.push(value);
        }
    }
    Ok(rows)
}

fn write_json(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}

fn write_jsonl(path: &Path, rows: &[Value]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = String::new();
    for row in rows {
        text.push_str(&serde_json::to_string(row)?);
        text.push('\n');
    }
    fs::write(path, text)?;
    Ok(())
}

fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn gate(name: &str, passed: bool, evidence: Value) -> Value {
    json!({"gate": name, "passed": passed, "evidence": evidence})
}

fn now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}

fn rel(value: &str) -> String {
    value.replace('\\', "/")
}

fn stable_hash_hex(text: &str) -> String {
    format!(
        "{:016x}{:016x}",
        stable_hash_u64(text),
        stable_hash_u64(&format!("salt:{text}"))
    )
}

fn stable_hash_u64(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
