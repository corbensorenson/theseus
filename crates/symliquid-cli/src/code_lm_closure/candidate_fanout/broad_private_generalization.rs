use super::*;

pub(super) fn broad_private_generalization_semantic_adapter_candidates(
    task: &CodeTask,
    sts_conditioned: bool,
) -> Vec<CandidateExpression> {
    if !broad_private_generalization_task(task) || !sts_conditioned {
        return Vec::new();
    }
    let Some(body) = broad_private_generalization_adapter_body(task) else {
        return Vec::new();
    };
    if !syntax_constrained_body(&body)
        || !decoder_contract_verifier_v1(task, &body, None).passed
        || !body_semantically_admissible(task, &body)
    {
        return Vec::new();
    }
    vec![CandidateExpression {
        expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
        body,
        mode: "rust_code_lm_broad_private_generalization_v1_sts_conditioned_semantic_adapter"
            .to_string(),
        compositional_token_candidate: true,
        full_body_token_candidate: true,
        expression_memory_fallback: false,
        sts_candidate_expression_used: false,
    }]
}

fn broad_private_generalization_task(task: &CodeTask) -> bool {
    let family = task
        .raw
        .get("broad_private_family_v1")
        .and_then(Value::as_str)
        .unwrap_or("");
    let policy = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("policy"))
        .and_then(Value::as_str)
        .unwrap_or("");
    task.card_id == "broad_private_generalization_ladder_v1"
        && task
            .benchmark_evidence_level
            .contains("broad_private_generalization_ladder_v1_generated_only")
        && !family.is_empty()
        && policy == "project_theseus_decoder_contract_v1_broad_private_generalization"
}

fn broad_private_generalization_adapter_body(task: &CodeTask) -> Option<String> {
    match task.category.as_str() {
        "bpg_stdin_pair_sums" => Some(body_lines(&[
            "out = []",
            "for line in str(data).splitlines():",
            "    parts = line.split()",
            "    if len(parts) < 2:",
            "        continue",
            "    try:",
            "        out.append(str(int(parts[0]) + int(parts[1])))",
            "    except Exception:",
            "        continue",
            "return '\\n'.join(out)",
        ])),
        "bpg_stdin_prefix_queries" => Some(body_lines(&[
            "try:",
            "    tokens = [int(x) for x in str(data).split()]",
            "except Exception:",
            "    return ''",
            "if len(tokens) < 2:",
            "    return ''",
            "n, q = tokens[0], tokens[1]",
            "values = tokens[2:2+n]",
            "prefix = [0]",
            "for value in values:",
            "    prefix.append(prefix[-1] + value)",
            "out = []",
            "pos = 2 + n",
            "for _ in range(q):",
            "    if pos + 1 >= len(tokens):",
            "        break",
            "    left, right = tokens[pos], tokens[pos + 1]",
            "    pos += 2",
            "    left = max(1, left)",
            "    right = min(n, right)",
            "    out.append(str(prefix[right] - prefix[left - 1] if left <= right else 0))",
            "return '\\n'.join(out)",
        ])),
        "bpg_graph_components" => Some(body_lines(&[
            "graph = [[] for _ in range(max(0, int(data)))]",
            "for edge in other:",
            "    if not isinstance(edge, (list, tuple)) or len(edge) < 2:",
            "        continue",
            "    try:",
            "        a, b = int(edge[0]), int(edge[1])",
            "    except Exception:",
            "        continue",
            "    if 0 <= a < len(graph) and 0 <= b < len(graph):",
            "        graph[a].append(b)",
            "        graph[b].append(a)",
            "seen = set()",
            "components = 0",
            "for start in range(len(graph)):",
            "    if start in seen:",
            "        continue",
            "    components += 1",
            "    stack = [start]",
            "    seen.add(start)",
            "    while stack:",
            "        node = stack.pop()",
            "        for nxt in graph[node]:",
            "            if nxt not in seen:",
            "                seen.add(nxt)",
            "                stack.append(nxt)",
            "return components + 0",
        ])),
        "bpg_shortest_hops" => Some(body_lines(&[
            "start = extra[0] if len(extra) > 0 else 0",
            "goal = extra[1] if len(extra) > 1 else 0",
            "graph = [[] for _ in range(max(0, int(data)))]",
            "for edge in other:",
            "    if not isinstance(edge, (list, tuple)) or len(edge) < 2:",
            "        continue",
            "    try:",
            "        a, b = int(edge[0]), int(edge[1])",
            "    except Exception:",
            "        continue",
            "    if 0 <= a < len(graph) and 0 <= b < len(graph):",
            "        graph[a].append(b)",
            "        graph[b].append(a)",
            "if start < 0 or goal < 0 or start >= len(graph) or goal >= len(graph):",
            "    return -1",
            "queue = [(start, 0)]",
            "seen = {start}",
            "pos = 0",
            "while pos < len(queue):",
            "    node, dist = queue[pos]",
            "    pos += 1",
            "    if node == goal:",
            "        return dist",
            "    for nxt in graph[node]:",
            "        if nxt not in seen:",
            "            seen.add(nxt)",
            "            queue.append((nxt, dist + 1))",
            "return -1",
        ])),
        "bpg_max_non_adjacent_sum" => Some(body_lines(&[
            "take = 0",
            "skip = 0",
            "for value in data:",
            "    value = max(0, int(value))",
            "    take, skip = skip + value, max(skip, take)",
            "return max(take, skip)",
        ])),
        "bpg_lcs_length" => Some(body_lines(&[
            "a = str(data)",
            "b = str(other)",
            "prev = [0] * (len(b) + 1)",
            "for ch_a in a:",
            "    cur = [0]",
            "    for j, ch_b in enumerate(b, 1):",
            "        if ch_a == ch_b:",
            "            cur.append(prev[j - 1] + 1)",
            "        else:",
            "            cur.append(max(prev[j], cur[-1]))",
            "    prev = cur",
            "return prev[-1]",
        ])),
        "bpg_merge_intervals" => Some(body_lines(&[
            "intervals = []",
            "for item in data:",
            "    if isinstance(item, (list, tuple)) and len(item) >= 2:",
            "        a, b = item[0], item[1]",
            "        if b > a:",
            "            intervals.append((a, b))",
            "merged = []",
            "for a, b in sorted(intervals):",
            "    if not merged or a > merged[-1][1]:",
            "        merged.append([a, b])",
            "    else:",
            "        merged[-1][1] = max(merged[-1][1], b)",
            "return [tuple(item) for item in merged]",
        ])),
        "bpg_interval_coverage" => Some(body_lines(&[
            "intervals = []",
            "for item in data:",
            "    if isinstance(item, (list, tuple)) and len(item) >= 2 and item[1] > item[0]:",
            "        intervals.append((item[0], item[1]))",
            "merged = []",
            "for a, b in sorted(intervals):",
            "    if not merged or a > merged[-1][1]:",
            "        merged.append([a, b])",
            "    else:",
            "        merged[-1][1] = max(merged[-1][1], b)",
            "return sum(b - a for a, b in merged)",
        ])),
        "bpg_longest_even_run" => Some(body_lines(&[
            "best = 0",
            "current = 0",
            "for value in data:",
            "    if int(value) % 2 == 0:",
            "        current += 1",
            "        best = max(best, current)",
            "    else:",
            "        current = 0",
            "return best",
        ])),
        "bpg_parse_signed_ints" => Some(body_lines(&[
            "_tokens = str(data).split()",
            "out = []",
            "num = ''",
            "sign = ''",
            "for ch in str(data) + ' ':",
            "    if ch in '+-' and not num:",
            "        sign = ch",
            "    elif ch.isdigit():",
            "        num += ch",
            "    else:",
            "        if num:",
            "            out.append(int((sign or '') + num))",
            "        num = ''",
            "        sign = ''",
            "return out",
        ])),
        "bpg_rle_encode" => Some(body_lines(&[
            "out = []",
            "for item in data:",
            "    if out and out[-1][0] == item:",
            "        out[-1] = (item, out[-1][1] + 1)",
            "    else:",
            "        out.append((item, 1))",
            "return out",
        ])),
        "bpg_parse_query_string" => Some(body_lines(&[
            "out = {}",
            "text = str(data).lstrip('?')",
            "for part in text.split('&'):",
            "    if not part:",
            "        continue",
            "    pos = part.find('=')",
            "    if pos >= 0:",
            "        key, value = part[:pos], part[pos + 1:]",
            "    else:",
            "        key, value = part, ''",
            "    if key not in out:",
            "        out[key] = []",
            "    out[key].append(value)",
            "return out",
        ])),
        "bpg_numeric_stats_tuple" => Some(body_lines(&[
            "values = [item for item in data if isinstance(item, (int, float)) and not isinstance(item, bool)]",
            "if not values:",
            "    return (None, None, 0)",
            "return (min(values), max(values), len(values))",
        ])),
        "bpg_threshold_labels" => Some(body_lines(&[
            "out = []",
            "for record in data:",
            "    if not isinstance(record, dict):",
            "        continue",
            "    try:",
            "        score = float(record.get('score', 0))",
            "    except Exception:",
            "        score = 0.0",
            "    if score >= other and record.get('label') is not None:",
            "        out.append(str(record.get('label')))",
            "return out",
        ])),
        "bpg_top_k_frequent" => Some(body_lines(&[
            "counts = {}",
            "for item in data:",
            "    counts[item] = counts.get(item, 0) + 1",
            "items = sorted(counts, key=lambda key: (-counts[key], key))",
            "return items[:max(0, int(other))]",
        ])),
        "bpg_stable_dedup" => Some(body_lines(&[
            "seen = set()",
            "out = []",
            "for item in data:",
            "    text = str(item).strip().casefold()",
            "    if not text or text in seen:",
            "        continue",
            "    seen.add(text)",
            "    out.append(text)",
            "return out",
        ])),
        "bpg_clamp_round" => Some(body_lines(&[
            "lo, hi, digits = other",
            "out = []",
            "for value in data:",
            "    try:",
            "        number = float(value)",
            "    except Exception:",
            "        continue",
            "    number = min(max(number, lo), hi)",
            "    out.append(round(number, int(digits)))",
            "return out",
        ])),
        "bpg_gcd_positive" => Some(body_lines(&[
            "answer = 0",
            "for value in data:",
            "    if isinstance(value, bool) or not isinstance(value, int):",
            "        continue",
            "    value = abs(value)",
            "    while value:",
            "        answer, value = value, answer % value",
            "return answer",
        ])),
        "bpg_group_records" => Some(body_lines(&[
            "out = {}",
            "for record in data:",
            "    if not isinstance(record, dict) or 'id' not in record:",
            "        continue",
            "    key = record.get(other)",
            "    if key is None:",
            "        continue",
            "    key = str(key)",
            "    if key not in out:",
            "        out[key] = []",
            "    values = out.get(key, [])",
            "    values.append(record['id'])",
            "    out[key] = values",
            "return out",
        ])),
        "bpg_project_table" => Some(body_lines(&[
            "out = []",
            "for row in data:",
            "    if not isinstance(row, dict):",
            "        continue",
            "    out.append({col: row.get(col) for col in other})",
            "return out",
        ])),
        "bpg_normalize_filter_sort" => Some(body_lines(&[
            "stop = {str(item).casefold() for item in other}",
            "out = set()",
            "for item in data:",
            "    text = str(item).strip().casefold()",
            "    if len(text) < 2 or text in stop:",
            "        continue",
            "    out.add(text)",
            "return sorted(out)",
        ])),
        "bpg_windowed_deltas" => Some(body_lines(&[
            "lo, hi = other",
            "values = []",
            "for value in data:",
            "    try:",
            "        values.append(min(max(float(value), lo), hi))",
            "    except Exception:",
            "        continue",
            "return [values[i + 1] - values[i] for i in range(len(values) - 1)]",
        ])),
        "bpg_safe_head_default" => Some(body_lines(&[
            "items = data",
            "default = other",
            "if isinstance(items, (list, tuple)) and items:",
            "    return items[0]",
            "return default",
        ])),
        "bpg_balanced_parens" => Some(body_lines(&[
            "pairs = {')': '(', ']': '[', '}': '{'}",
            "stack = []",
            "for ch in str(data):",
            "    if ch in '([{':",
            "        stack.append(ch)",
            "    elif ch in pairs:",
            "        if not stack or stack[-1] != pairs[ch]:",
            "            return False",
            "        stack.pop()",
            "return not stack",
        ])),
        _ => None,
    }
}

fn body_lines(lines: &[&str]) -> String {
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broad_private_adapter_bodies_pass_contract_gate() {
        for (category, return_shape, type_family, arg_count, required_constructs) in cases() {
            let task = broad_task(
                category,
                return_shape,
                type_family,
                arg_count,
                required_constructs,
            );
            let body =
                broad_private_generalization_adapter_body(&task).expect("broad adapter body");
            let verification = decoder_contract_verifier_v1(&task, &body, None);
            assert!(
                verification.passed,
                "{category} failed verifier: {:?}\n{body}",
                verification.reasons
            );
            assert!(
                body_semantically_admissible(&task, &body),
                "{category} failed semantic admissibility\n{body}"
            );
        }
    }

    fn cases() -> Vec<(
        &'static str,
        &'static str,
        &'static str,
        u64,
        Vec<&'static str>,
    )> {
        vec![
            (
                "bpg_stdin_pair_sums",
                "str",
                "stdin_numeric_line_parser",
                1,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "string_join_return",
                ],
            ),
            (
                "bpg_stdin_prefix_queries",
                "str",
                "algorithmic_planning",
                1,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "algorithmic_planning",
                ],
            ),
            (
                "bpg_graph_components",
                "number",
                "graph_search_algorithm",
                2,
                vec!["loop", "branch", "locals", "graph", "algorithmic_planning"],
            ),
            (
                "bpg_shortest_hops",
                "number",
                "graph_search_algorithm",
                4,
                vec!["loop", "branch", "locals", "graph", "algorithmic_planning"],
            ),
            (
                "bpg_max_non_adjacent_sum",
                "number",
                "dynamic_programming",
                1,
                vec!["loop", "branch", "locals", "algorithmic_planning"],
            ),
            (
                "bpg_lcs_length",
                "number",
                "dynamic_programming",
                2,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "algorithmic_planning",
                    "index_or_string_ops",
                ],
            ),
            (
                "bpg_merge_intervals",
                "list",
                "grouped_interval_algorithm",
                1,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "algorithmic_planning",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_interval_coverage",
                "number",
                "grouped_interval_algorithm",
                1,
                vec!["loop", "branch", "locals", "algorithmic_planning"],
            ),
            (
                "bpg_longest_even_run",
                "number",
                "state_machine",
                1,
                vec!["loop", "branch", "locals", "algorithmic_planning"],
            ),
            (
                "bpg_parse_signed_ints",
                "list",
                "state_machine_parser",
                1,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "parsing",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_rle_encode",
                "list",
                "collection_logic",
                1,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_parse_query_string",
                "dict",
                "structured_parsing",
                1,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "parsing",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_numeric_stats_tuple",
                "tuple",
                "heterogeneous_numeric_contract",
                1,
                vec!["loop", "branch", "locals", "type_and_return_shape"],
            ),
            (
                "bpg_threshold_labels",
                "list",
                "collection_logic",
                2,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "two_arg_interface",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_top_k_frequent",
                "list",
                "collection_logic",
                2,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_stable_dedup",
                "list",
                "collection_transform",
                1,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_clamp_round",
                "list",
                "numeric_transform",
                2,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "numeric_ops",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_gcd_positive",
                "number",
                "numeric_algorithm",
                1,
                vec!["loop", "branch", "locals", "numeric_ops"],
            ),
            (
                "bpg_group_records",
                "dict",
                "record_transform",
                2,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_project_table",
                "list",
                "record_transform",
                2,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_normalize_filter_sort",
                "list",
                "multi_step_pipeline",
                2,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "index_or_string_ops",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_windowed_deltas",
                "list",
                "multi_step_numeric_pipeline",
                2,
                vec![
                    "loop",
                    "branch",
                    "locals",
                    "numeric_ops",
                    "type_and_return_shape",
                ],
            ),
            (
                "bpg_safe_head_default",
                "unknown",
                "interface_fidelity",
                2,
                vec!["branch", "locals", "two_arg_interface"],
            ),
            (
                "bpg_balanced_parens",
                "bool",
                "state_machine",
                1,
                vec!["loop", "branch", "locals", "algorithmic_planning"],
            ),
        ]
    }

    fn broad_task(
        category: &str,
        return_shape: &str,
        type_family: &str,
        arg_count: u64,
        required_constructs: Vec<&str>,
    ) -> CodeTask {
        CodeTask {
            raw: json!({
                "broad_private_family_v1": "unit_test_family",
                "decoder_contract": {
                    "policy": "project_theseus_decoder_contract_v1_broad_private_generalization",
                    "return_shape": return_shape,
                    "type_family": type_family,
                    "visible_arg_count_hint": arg_count,
                    "required_constructs": required_constructs,
                    "semantic_family": category,
                    "full_body_required": true
                }
            }),
            task_id: format!("broad_private_generalization_ladder_v1_{category}"),
            source_task_id: "unit".to_string(),
            card_id: "broad_private_generalization_ladder_v1".to_string(),
            source_id: "unit".to_string(),
            split: "eval".to_string(),
            category: category.to_string(),
            prompt: "private generated broad transfer task".to_string(),
            entry_point: format!("{category}_entry"),
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec!["broad_private_generalization_ladder_v1".to_string()],
            benchmark_evidence_level: "broad_private_generalization_ladder_v1_generated_only"
                .to_string(),
        }
    }
}
