fn load_tasks(path: &Path) -> Result<Vec<Task>, Box<dyn std::error::Error>> {
    let mut tasks = Vec::new();
    for raw in read_jsonl(path)? {
        let task_id = string_field(&raw, "task_id");
        let prompt = string_field(&raw, "prompt");
        let entry_point = string_field(&raw, "entry_point");
        if task_id.is_empty() || prompt.is_empty() || entry_point.is_empty() {
            continue;
        }
        tasks.push(Task {
            task_id,
            source_task_id: string_field(&raw, "source_task_id"),
            card_id: string_field(&raw, "card_id"),
            source_id: string_field(&raw, "source_id"),
            prompt,
            entry_point,
            case_type: string_field(&raw, "case_type"),
            tags: raw
                .get("tags")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
            benchmark_evidence_level: string_field(&raw, "benchmark_evidence_level"),
            raw,
        });
    }
    Ok(tasks)
}

fn train_token_model(
    config: &CodeTokenGeneratorConfig,
) -> Result<(TokenModel, TrainingSummary), Box<dyn std::error::Error>> {
    let manifest = read_json(Path::new(&config.training_sources))?;
    let ready_sources = manifest
        .get("ready_sources")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|row| {
            row.get("training_use_state").and_then(Value::as_str) == Some("ready_local_verified")
                && row.get("sha256_verified").and_then(Value::as_bool) == Some(true)
        })
        .collect::<Vec<_>>();
    let mut snippets = Vec::new();
    let mut body_candidates = Vec::new();
    let mut source_summaries = Vec::new();
    let mut rows_seen = 0usize;
    let mut rows_used = 0usize;
    let mut skipped_protected_rows = 0usize;
    for source in &ready_sources {
        let local_path = string_field(source, "local_path");
        let path = PathBuf::from(&local_path);
        let mut source_rows_seen = 0usize;
        let mut source_rows_used = 0usize;
        let mut source_snippets = 0usize;
        if path.exists() {
            let text = fs::read_to_string(&path).unwrap_or_default();
            for line in text.lines() {
                if source_rows_seen >= config.max_training_rows_per_source.max(1) {
                    break;
                }
                source_rows_seen += 1;
                rows_seen += 1;
                let raw: Value = match serde_json::from_str(line) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
                if protected_training_row(&raw) {
                    skipped_protected_rows += 1;
                    continue;
                }
                let extracted = extract_training_code_snippets(&raw);
                if extracted.is_empty() {
                    continue;
                }
                body_candidates.extend(extract_training_body_candidates(&raw));
                rows_used += 1;
                source_rows_used += 1;
                source_snippets += extracted.len();
                snippets.extend(extracted);
            }
        }
        source_summaries.push(json!({
            "dataset_id": source.get("dataset_id").cloned().unwrap_or(Value::Null),
            "local_path": local_path,
            "rows_seen": source_rows_seen,
            "rows_used_for_code": source_rows_used,
            "code_snippets": source_snippets,
            "sha256_verified": source.get("sha256_verified").and_then(Value::as_bool).unwrap_or(false),
            "training_use_state": source.get("training_use_state").cloned().unwrap_or(Value::Null),
        }));
    }

    let project_files =
        collect_project_code_files(&config.project_code_roots, config.max_project_files);
    let mut project_body_candidates = 0usize;
    for path in &project_files {
        if let Ok(text) = fs::read_to_string(path) {
            let extracted_bodies = extract_project_function_body_candidates(&text);
            project_body_candidates += extracted_bodies.len();
            body_candidates.extend(extracted_bodies);
            snippets.push(text);
        }
    }

    let model = build_model(&snippets, &body_candidates);
    let summary = TrainingSummary {
        ready_sources: ready_sources.len(),
        rows_seen,
        rows_used,
        code_snippets: snippets.len(),
        project_code_files: project_files.len(),
        project_body_candidates,
        skipped_protected_rows,
        source_summaries,
    };
    Ok((model, summary))
}

fn protected_training_row(row: &Value) -> bool {
    if row.get("public_benchmark").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    if row
        .get("public_benchmark_payload_detected")
        .and_then(Value::as_bool)
        == Some(true)
    {
        return true;
    }
    if row.get("public_tests_used").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    if row.get("public_solutions_used").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    if row
        .get("protected_holdout_overlap")
        .and_then(Value::as_bool)
        == Some(true)
    {
        return true;
    }
    if row
        .get("benchmark_ids")
        .and_then(Value::as_array)
        .map(|items| !items.is_empty())
        .unwrap_or(false)
    {
        return true;
    }
    row.get("split")
        .and_then(Value::as_str)
        .map(|split| split.eq_ignore_ascii_case("test"))
        .unwrap_or(false)
}

fn extract_training_code_snippets(row: &Value) -> Vec<String> {
    let mut snippets = Vec::new();
    for key in [
        "solution_body",
        "function_body",
        "body",
        "candidate_body",
        "code",
        "candidate_code",
        "answer",
    ] {
        let value = string_field(row, key);
        if value.trim().is_empty() {
            continue;
        }
        snippets.extend(extract_code_snippets(&value));
        if looks_like_code(&value) || value.lines().any(looks_like_code_line) {
            snippets.push(value);
        }
    }
    let expr = string_field(row, "solution_expr");
    if !expr.trim().is_empty() && useful_expression(&expr) {
        snippets.push(format!("return {}", normalize_expression(&expr)));
    }
    snippets
}

fn extract_training_body_candidates(row: &Value) -> Vec<BodyCandidate> {
    let mut bodies = Vec::new();
    let semantic_tokens = training_row_semantic_tokens(row);
    let semantic_labels = training_row_semantic_labels(row);
    for key in ["solution_body", "function_body", "body", "candidate_body"] {
        let value = string_field(row, key);
        let normalized = normalize_body_text(&value);
        if useful_private_multistatement_body(&normalized) {
            bodies.push(BodyCandidate {
                body: normalized,
                semantic_tokens: semantic_tokens.clone(),
                semantic_labels: semantic_labels.clone(),
            });
        }
    }
    bodies
}

fn extract_project_function_body_candidates(text: &str) -> Vec<BodyCandidate> {
    let lines = text.lines().collect::<Vec<_>>();
    let mut bodies = Vec::new();
    let mut idx = 0usize;
    while idx < lines.len() {
        let line = lines[idx];
        let trimmed = line.trim_start();
        if !is_project_function_def_line(trimmed) {
            idx += 1;
            continue;
        }
        let def_indent = leading_indent_width(line);
        let params = params_from_def_line(trimmed);
        let mut raw_body = Vec::new();
        idx += 1;
        while idx < lines.len() {
            let current = lines[idx];
            let current_trimmed = current.trim_start();
            if current_trimmed.is_empty() {
                raw_body.push(String::new());
                idx += 1;
                continue;
            }
            let current_indent = leading_indent_width(current);
            if current_indent <= def_indent {
                break;
            }
            raw_body.push(current.to_string());
            idx += 1;
        }
        let dedented = dedent_body_lines(&raw_body);
        let normalized_params = normalize_project_body_parameters(&dedented, &params);
        let normalized = normalize_body_text(&normalized_params);
        if useful_private_multistatement_body(&normalized)
            && project_body_candidate_safe(&normalized)
        {
            bodies.push(BodyCandidate {
                body: normalized.clone(),
                semantic_tokens: tokenize_words(&normalized).into_iter().collect(),
                semantic_labels: intent_labels_from_material(&normalized),
            });
        }
    }
    bodies
}

fn training_row_semantic_tokens(row: &Value) -> BTreeSet<String> {
    let mut material = vec![
        string_field(row, "task_id"),
        string_field(row, "source_task_id"),
        string_field(row, "card_id"),
        string_field(row, "source_id"),
        string_field(row, "category"),
        string_field(row, "prompt"),
        string_field(row, "entry_point"),
        string_field(row, "targeted_private_residual_family_v3"),
        string_field(row, "concept_residual_label"),
        string_field(row, "residual_concept"),
    ];
    if let Some(tags) = row.get("tags").and_then(Value::as_array) {
        for tag in tags {
            if let Some(text) = tag.as_str() {
                material.push(text.to_string());
            }
        }
    }
    if let Some(contract) = row.get("decoder_contract").and_then(Value::as_object) {
        for key in [
            "semantic_family",
            "residual_label_hint",
            "type_family",
            "return_shape",
        ] {
            if let Some(text) = contract.get(key).and_then(Value::as_str) {
                material.push(text.to_string());
            }
        }
        if let Some(items) = contract
            .get("required_constructs")
            .and_then(Value::as_array)
        {
            for item in items {
                if let Some(text) = item.as_str() {
                    material.push(text.to_string());
                }
            }
        }
    }
    tokenize_words(&material.join(" ")).into_iter().collect()
}

fn training_row_semantic_labels(row: &Value) -> BTreeSet<String> {
    let mut labels = BTreeSet::new();
    for key in ["category", "concept_residual_label", "residual_concept"] {
        insert_semantic_label(&mut labels, &string_field(row, key));
    }
    if let Some(contract) = row.get("decoder_contract").and_then(Value::as_object) {
        for key in ["semantic_family", "residual_label_hint"] {
            if let Some(text) = contract.get(key).and_then(Value::as_str) {
                insert_semantic_label(&mut labels, text);
            }
        }
    }
    labels.extend(intent_labels_from_material(&semantic_material_for_row(row)));
    labels
}

fn semantic_material_for_row(row: &Value) -> String {
    let mut material = vec![
        string_field(row, "task_id"),
        string_field(row, "source_task_id"),
        string_field(row, "card_id"),
        string_field(row, "source_id"),
        string_field(row, "category"),
        string_field(row, "prompt"),
        string_field(row, "entry_point"),
        string_field(row, "targeted_private_residual_family_v3"),
        string_field(row, "concept_residual_label"),
        string_field(row, "residual_concept"),
    ];
    if let Some(tags) = row.get("tags").and_then(Value::as_array) {
        for tag in tags {
            if let Some(text) = tag.as_str() {
                material.push(text.to_string());
            }
        }
    }
    material.push(decoder_contract_token_material(row));
    material.join(" ")
}

fn insert_semantic_label(labels: &mut BTreeSet<String>, value: &str) {
    let normalized = value.trim().to_lowercase();
    if !normalized.is_empty() {
        labels.insert(normalized);
    }
}

fn intent_labels_from_material(value: &str) -> BTreeSet<String> {
    let text = normalize_phrase_words(value);
    let mut labels = BTreeSet::new();
    let mut add = |label: &str| {
        labels.insert(label.to_string());
    };
    if phrase_any(
        &text,
        &[
            "balanced parens",
            "balanced parentheses",
            "balanced brackets",
            "bracket characters",
            "valid parentheses",
        ],
    ) {
        add("balanced_delimiters");
    }
    if phrase_any(
        &text,
        &[
            "parse query string",
            "query string",
            "url query",
            "urlencode",
            "key value pairs",
        ],
    ) {
        add("query_string_parse");
    }
    if phrase_any(
        &text,
        &[
            "signed ints",
            "signed integers",
            "parse signed",
            "extract integers",
            "parse ints",
        ],
    ) {
        add("parse_signed_ints");
    }
    if phrase_any(
        &text,
        &["rle", "run length", "run lengths", "roundtrip rle"],
    ) {
        add("run_length_encoding");
    }
    if phrase_any(
        &text,
        &[
            "sort",
            "sorted",
            "sorting",
            "ascending",
            "descending",
            "lexicographic order",
            "order the",
            "ordered list",
        ],
    ) {
        add("sorting_order");
    }
    if phrase_any(
        &text,
        &[
            "count integer",
            "count integers",
            "count number",
            "count numbers",
            "number of integer",
            "number of integers",
            "integer count",
            "positive integers",
            "negative integers",
        ],
    ) {
        add("integer_counting");
    }
    if phrase_any(
        &text,
        &[
            "stable dedup",
            "first seen unique",
            "preserve encounter order",
            "unique text tokens",
            "casefold",
        ],
    ) {
        add("stable_dedup_normalization");
    }
    if phrase_any(&text, &["top k", "most frequent", "frequency", "frequent"]) {
        add("frequency_top_k");
    }
    if phrase_any(
        &text,
        &[
            "palindrome",
            "palindromic",
            "same forward and backward",
            "reverse string",
            "reversed string",
        ],
    ) {
        add("palindrome_or_reverse");
    }
    if phrase_any(
        &text,
        &[
            "prime",
            "primality",
            "is prime",
            "prime number",
            "composite number",
        ],
    ) {
        add("prime_number");
    }
    if phrase_any(
        &text,
        &["fibonacci", "tribonacci", "nth number", "n th number"],
    ) {
        add("recurrence_sequence");
    }
    if phrase_any(
        &text,
        &[
            "merge intervals",
            "interval coverage",
            "covered half open interval",
            "half open interval",
            "overlaps",
        ],
    ) {
        add("interval_merge_coverage");
    }
    if phrase_any(
        &text,
        &["longest even run", "windowed deltas", "subarray", "window"],
    ) {
        add("sequence_window_scan");
    }
    if phrase_any(
        &text,
        &[
            "gcd",
            "greatest common divisor",
            "absolute positive integer",
            "positive integer values",
        ],
    ) {
        add("gcd_numeric");
    }
    if phrase_any(&text, &["clamp", "round", "rounded"]) {
        add("clamp_round_numeric");
    }
    if phrase_any(
        &text,
        &[
            "numeric stats",
            "sum",
            "count",
            "number of",
            "mean",
            "average",
            "minimum",
            "maximum",
        ],
    ) {
        add("numeric_stats");
    }
    if phrase_any(
        &text,
        &["group records", "project table", "records", "table"],
    ) {
        add("record_table_transform");
    }
    let graph_components = phrase_any(
        &text,
        &[
            "graph components",
            "component sizes",
            "connectivity",
            "connected components",
        ],
    );
    if graph_components {
        add("graph_components");
    }
    let shortest_path_hops = phrase_any(
        &text,
        &[
            "shortest hops",
            "shortest hop",
            "shortest unweighted hop",
            "hop count",
            "unreachable",
            "bfs",
        ],
    );
    if shortest_path_hops {
        add("shortest_path_hops");
    }
    if graph_components || shortest_path_hops || phrase_any(&text, &["dfs", "edge list", "graph"]) {
        add("graph_algorithm");
    }
    if phrase_any(
        &text,
        &[
            "lcs",
            "longest common subsequence",
            "subsequence length",
            "dynamic programming",
        ],
    ) {
        add("dynamic_programming_lcs");
    }
    if phrase_any(
        &text,
        &["max non adjacent", "non adjacent sum", "house robber"],
    ) {
        add("dynamic_programming_non_adjacent");
    }
    if phrase_any(&text, &["stdin", "standard input", "print", "output"]) {
        add("stdin_io_contract");
    }
    if phrase_any(&text, &["threshold labels", "threshold", "score", "label"]) {
        add("threshold_labeling");
    }
    if phrase_any(
        &text,
        &["rom", "gameboy", "game boy", "gb rom", "commercial rom"],
    ) {
        add("rom_policy_detection");
    }
    labels
}

fn phrase_any(text: &str, phrases: &[&str]) -> bool {
    phrases.iter().any(|phrase| {
        let normalized = normalize_phrase_words(phrase);
        !normalized.is_empty() && text.contains(&normalized)
    })
}

fn is_project_function_def_line(trimmed: &str) -> bool {
    (trimmed.starts_with("def ") || trimmed.starts_with("async def "))
        && trimmed.contains('(')
        && trimmed.contains(')')
        && trimmed.ends_with(':')
}

fn params_from_def_line(trimmed: &str) -> Vec<String> {
    let Some(open) = trimmed.find('(') else {
        return Vec::new();
    };
    let Some(close) = trimmed[open + 1..].find(')') else {
        return Vec::new();
    };
    trimmed[open + 1..open + 1 + close]
        .split(',')
        .filter_map(clean_signature_arg)
        .filter(|name| name != "self" && name != "cls")
        .take(3)
        .collect()
}

fn leading_indent_width(line: &str) -> usize {
    line.chars()
        .take_while(|ch| ch.is_whitespace() && *ch != '\n')
        .map(|ch| if ch == '\t' { 4 } else { 1 })
        .sum()
}

fn dedent_body_lines(lines: &[String]) -> String {
    let min_indent = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| leading_indent_width(line))
        .min()
        .unwrap_or(0);
    lines
        .iter()
        .map(|line| strip_indent_width(line, min_indent))
        .collect::<Vec<_>>()
        .join("\n")
}

fn strip_indent_width(line: &str, width: usize) -> String {
    let mut consumed = 0usize;
    let mut byte_idx = 0usize;
    for (idx, ch) in line.char_indices() {
        let value = if ch == '\t' {
            4
        } else if ch.is_whitespace() {
            1
        } else {
            break;
        };
        if consumed + value > width {
            break;
        }
        consumed += value;
        byte_idx = idx + ch.len_utf8();
    }
    line[byte_idx..].to_string()
}

fn normalize_project_body_parameters(body: &str, params: &[String]) -> String {
    let mut normalized = body.to_string();
    for (from, to) in params.iter().zip(["data", "other", "extra"]) {
        if from != to {
            normalized = replace_identifier_outside_strings(&normalized, from, to);
        }
    }
    normalized
}

fn replace_identifier_outside_strings(text: &str, from: &str, to: &str) -> String {
    if from.is_empty() || from == to {
        return text.to_string();
    }
    let chars = text.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut idx = 0usize;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    while idx < chars.len() {
        let ch = chars[idx];
        if let Some(q) = quote {
            out.push(ch);
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
            out.push(ch);
            idx += 1;
            continue;
        }
        if is_ident_char(ch) && (idx == 0 || !is_ident_char(chars[idx - 1])) {
            let start = idx;
            idx += 1;
            while idx < chars.len() && is_ident_char(chars[idx]) {
                idx += 1;
            }
            let token = chars[start..idx].iter().collect::<String>();
            if token == from {
                out.push_str(to);
            } else {
                out.push_str(&token);
            }
            continue;
        }
        out.push(ch);
        idx += 1;
    }
    out
}

fn is_ident_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn project_body_candidate_safe(body: &str) -> bool {
    let lowered = body.to_lowercase();
    let blocked = [
        "public",
        "benchmark",
        "calibration",
        "operator_unlock",
        "teacher",
        "network",
        "launchagent",
        "subprocess",
        "request",
        "socket",
        "file",
        "path",
        "report",
    ];
    !blocked.iter().any(|token| lowered.contains(token))
}

fn collect_project_code_files(roots: &str, max_files: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = roots
        .split(',')
        .map(str::trim)
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    while let Some(path) = stack.pop() {
        if files.len() >= max_files {
            break;
        }
        if excluded_path(&path) {
            continue;
        }
        if path.is_dir() {
            if let Ok(read_dir) = fs::read_dir(&path) {
                for entry in read_dir.flatten() {
                    stack.push(entry.path());
                }
            }
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("py") {
            files.push(path);
        }
    }
    files.sort();
    files.truncate(max_files);
    files
}

fn excluded_path(path: &Path) -> bool {
    let lowered = path.to_string_lossy().replace('\\', "/").to_lowercase();
    let blocked = [
        "/.git/",
        "/target/",
        "/reports/",
        "/benchmarks/",
        "/data/",
        "/resource_pantry/",
        "__pycache__",
        "mbpp",
        "humaneval",
        "human_eval",
        "evalplus",
        "bigcodebench",
        "livecodebench",
        "candidate_generator",
        "synthetic_benchmark",
    ];
    blocked.iter().any(|token| lowered.contains(token))
}

fn extract_code_snippets(text: &str) -> Vec<String> {
    let mut snippets = Vec::new();
    let mut in_fence = false;
    let mut current = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            if in_fence {
                let joined = current.join("\n");
                if looks_like_code(&joined) {
                    snippets.push(joined);
                }
                current.clear();
                in_fence = false;
            } else {
                in_fence = true;
            }
            continue;
        }
        if in_fence {
            current.push(line.to_string());
        }
    }
    let code_lines = text
        .lines()
        .filter(|line| looks_like_code_line(line))
        .map(str::to_string)
        .collect::<Vec<_>>();
    if code_lines.len() >= 2 {
        snippets.push(code_lines.join("\n"));
    }
    snippets
}

fn normalize_body_text(text: &str) -> String {
    text.lines()
        .map(str::trim_end)
        .skip_while(|line| line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn useful_private_multistatement_body(body: &str) -> bool {
    let body = body.trim();
    if body.is_empty() || body.len() > 2600 {
        return false;
    }
    if !private_body_imports_safe(body) {
        return false;
    }
    let lowered = body.to_lowercase();
    let blocked = [
        "subprocess",
        "open(",
        " os.",
        "os.",
        "shutil",
        "zipfile",
        "tarfile",
        "pathlib",
        "requests",
        "urllib",
        "socket",
        "eval(",
        "exec(",
        "__",
        "task_id",
        "canonical",
        "solution",
        "return none",
        "result = none",
    ];
    if blocked.iter().any(|token| lowered.contains(token)) {
        return false;
    }
    let nonempty = body.lines().filter(|line| !line.trim().is_empty()).count();
    nonempty >= 3
        && body
            .lines()
            .any(|line| line.trim_start().starts_with("return "))
        && (body_structures(body).contains("iteration")
            || body_structures(body).contains("conditional")
            || body_structures(body).contains("collection_build"))
}

fn private_body_imports_safe(body: &str) -> bool {
    body.lines().all(|line| {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            matches!(trimmed, "import math")
        } else if trimmed.starts_with("from ") && trimmed.contains(" import ") {
            matches!(trimmed, "from collections import deque")
        } else {
            true
        }
    })
}

fn safe_imported_identifiers(body: &str) -> Option<BTreeSet<String>> {
    let mut identifiers = BTreeSet::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            "import math" => {
                identifiers.insert("math".to_string());
            }
            "from collections import deque" => {
                identifiers.insert("deque".to_string());
            }
            _ if trimmed.starts_with("import ")
                || (trimmed.starts_with("from ") && trimmed.contains(" import ")) =>
            {
                return None;
            }
            _ => {}
        }
    }
    Some(identifiers)
}

fn looks_like_code(text: &str) -> bool {
    text.contains("def ")
        || text.contains("return ")
        || text.contains("import ")
        || text.contains("class ")
}

fn looks_like_code_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("def ")
        || trimmed.starts_with("return ")
        || trimmed.starts_with("if ")
        || trimmed.starts_with("for ")
        || trimmed.starts_with("while ")
        || trimmed.starts_with("import ")
        || trimmed.starts_with("from ")
        || trimmed.contains(" = ")
}

fn build_model(snippets: &[String], body_candidates: &[BodyCandidate]) -> TokenModel {
    let mut unigram = HashMap::new();
    let mut bigram: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut return_expr_counts: HashMap<String, usize> = HashMap::new();
    let mut token_count = 0usize;
    for snippet in snippets {
        let tokens = tokenize_code(snippet);
        let mut prev = "<BOS>".to_string();
        for token in tokens {
            *unigram.entry(token.clone()).or_insert(0) += 1;
            *bigram
                .entry(prev)
                .or_default()
                .entry(token.clone())
                .or_insert(0) += 1;
            prev = token;
            token_count += 1;
        }
        for line in snippet.lines() {
            if let Some(expr) = line.trim_start().strip_prefix("return ") {
                let normalized = normalize_expression(expr);
                if useful_expression(&normalized) {
                    *return_expr_counts.entry(normalized).or_insert(0) += 1;
                }
            }
        }
    }
    let mut return_exprs = return_expr_counts
        .into_iter()
        .map(|(expr, count)| ReturnExpr {
            tokens: tokenize_code(&expr),
            expr,
            count,
        })
        .collect::<Vec<_>>();
    return_exprs.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.expr.cmp(&b.expr)));
    let mut body_counts: HashMap<String, (usize, BTreeSet<String>, BTreeSet<String>)> =
        HashMap::new();
    for candidate in body_candidates {
        let normalized = normalize_body_text(&candidate.body);
        if useful_private_multistatement_body(&normalized) {
            let entry = body_counts
                .entry(normalized)
                .or_insert_with(|| (0, BTreeSet::new(), BTreeSet::new()));
            entry.0 += 1;
            entry.1.extend(candidate.semantic_tokens.iter().cloned());
            entry.2.extend(candidate.semantic_labels.iter().cloned());
        }
    }
    let mut body_snippets = body_counts
        .into_iter()
        .map(
            |(body, (count, semantic_tokens, semantic_labels))| BodySnippet {
                tokens: tokenize_words(&body),
                semantic_tokens,
                semantic_labels,
                structures: body_structures(&body),
                return_shapes: body_return_shapes(&body),
                body,
                count,
            },
        )
        .filter(|body| {
            body.structures.contains("iteration")
                || body.structures.contains("conditional")
                || body.structures.contains("string_processing")
                || body.structures.contains("collection_build")
        })
        .collect::<Vec<_>>();
    body_snippets.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| b.structures.len().cmp(&a.structures.len()))
            .then_with(|| a.body.cmp(&b.body))
    });
    let vocab_size = unigram.len();
    TokenModel {
        unigram,
        bigram,
        return_exprs,
        body_snippets,
        vocab_size,
        token_count,
    }
}
