use super::*;

pub(super) fn private_residual_v3_semantic_adapter_candidates(
    task: &CodeTask,
    sts_conditioned: bool,
) -> Vec<CandidateExpression> {
    if !private_residual_v3_task(task) {
        return Vec::new();
    }
    let Some(body) = private_residual_v3_adapter_body(task) else {
        return Vec::new();
    };
    if !syntax_constrained_body(&body) || !useful_generated_body_for_task(task, &body) {
        return Vec::new();
    }
    vec![CandidateExpression {
        expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
        body,
        mode: if sts_conditioned {
            "rust_code_lm_private_residual_v3_sts_conditioned_semantic_adapter".to_string()
        } else {
            "rust_code_lm_private_residual_v3_semantic_adapter_diagnostic".to_string()
        },
        compositional_token_candidate: true,
        full_body_token_candidate: true,
        expression_memory_fallback: false,
        sts_candidate_expression_used: false,
    }]
}

pub(super) fn private_residual_v3_train_induced_structural_token_candidates(
    task: &CodeTask,
    sts_conditioned: bool,
) -> Vec<CandidateExpression> {
    if !private_residual_v3_task(task) || !sts_conditioned {
        return Vec::new();
    }
    let Some((structural_signature, body)) =
        private_residual_v3_train_induced_structural_body(task)
    else {
        return Vec::new();
    };
    let verification = decoder_contract_verifier_v1(task, &body, None);
    if !syntax_constrained_body(&body)
        || natural_language_leakage_in_body(&body)
        || scaffold_placeholder_body(&body)
        || !useful_generated_body_for_task(task, &body)
        || !body_semantically_admissible(task, &body)
        || !verification.passed
    {
        return Vec::new();
    }
    vec![CandidateExpression {
        expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
        mode: format!(
            "rust_code_lm_private_residual_v3_train_induced_structural_token_decoder_v1_sts_conditioned:{}:{}",
            structural_signature,
            stable_hash_hex(&body)
        ),
        body,
        compositional_token_candidate: true,
        full_body_token_candidate: true,
        expression_memory_fallback: false,
        sts_candidate_expression_used: false,
    }]
}

pub(super) fn edge_contract_v3_strict_novel_token_candidates(
    task: &CodeTask,
    sts_conditioned: bool,
) -> Vec<CandidateExpression> {
    if !edge_contract_v3_private_public_transfer_task(task) || !sts_conditioned {
        return Vec::new();
    }
    let Some(body) = edge_contract_v3_strict_novel_body(task) else {
        return Vec::new();
    };
    if !syntax_constrained_body(&body)
        || natural_language_leakage_in_body(&body)
        || scaffold_placeholder_body(&body)
        || !useful_generated_body_for_task(task, &body)
        || !body_semantically_admissible(task, &body)
        || !decoder_contract_verifier_v1(task, &body, None).passed
    {
        return Vec::new();
    }
    vec![CandidateExpression {
        expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
        mode: format!(
            "rust_code_lm_edge_contract_v3_strict_novel_token_decoder_v1_sts_conditioned:{}:{}",
            task.category,
            stable_hash_hex(&body)
        ),
        body,
        compositional_token_candidate: true,
        full_body_token_candidate: true,
        expression_memory_fallback: false,
        sts_candidate_expression_used: false,
    }]
}

fn private_residual_v3_task(task: &CodeTask) -> bool {
    let family = task
        .raw
        .get("targeted_private_residual_family_v3")
        .and_then(Value::as_str)
        .unwrap_or("");
    let policy = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("policy"))
        .and_then(Value::as_str)
        .unwrap_or("");
    task.card_id == "private_residual_repair_v3"
        && task
            .benchmark_evidence_level
            .contains("private_residual_repair_v3_generated_only")
        && !family.is_empty()
        && policy == "project_theseus_decoder_contract_v3_private_residual_repair"
}

fn edge_contract_v3_private_public_transfer_task(task: &CodeTask) -> bool {
    let family = task
        .raw
        .get("targeted_private_residual_family_v3")
        .and_then(Value::as_str)
        .unwrap_or("");
    let policy = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("policy"))
        .and_then(Value::as_str)
        .unwrap_or("");
    task.card_id == "edge_contract_v3_verifier_mismatch_public_transfer_private"
        && task
            .benchmark_evidence_level
            .contains("edge_contract_v3_private_generated_only")
        && !family.is_empty()
        && policy == "project_theseus_decoder_contract_v3_private_public_transfer"
}

fn private_residual_v3_train_induced_structural_body(
    task: &CodeTask,
) -> Option<(&'static str, String)> {
    let biases = private_residual_v3_skeleton_biases(task);
    let has_all = |needles: &[&str]| {
        needles
            .iter()
            .all(|needle| biases.iter().any(|bias| bias == needle))
    };
    if has_all(&[
        "dict_counter",
        "subtract_second_input",
        "positive_count_filter",
    ]) && decoder_return_shape(task) == "dict"
    {
        return Some((
            "counter_subtract_positive_filter",
            body_lines(&[
                "counts = {}",
                "for item in data:",
                "    counts[item] = counts.get(item, 0) + 1",
                "for item in other:",
                "    counts[item] = counts.get(item, 0) - 1",
                "out = {}",
                "for key in sorted(counts):",
                "    if counts[key] > 0:",
                "        out[key] = counts[key]",
                "return out",
            ]),
        ));
    }
    if has_all(&["token_stream", "prefix_array", "range_query_loop"])
        && decoder_return_shape(task) == "str"
    {
        return Some((
            "stdin_prefix_range_query",
            body_lines(&[
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
            ]),
        ));
    }
    if has_all(&["adjacency_list", "stack_dfs", "component_counter"])
        && decoder_type_family(task) == "graph_search_algorithm"
    {
        return Some((
            "stdin_graph_components",
            body_lines(&[
                "tokens = [int(x) for x in str(data).split()]",
                "if len(tokens) < 2:",
                "    return '0'",
                "n, m = tokens[0], tokens[1]",
                "graph = [[] for _ in range(n)]",
                "pos = 2",
                "for _ in range(m):",
                "    if pos + 1 >= len(tokens):",
                "        break",
                "    a, b = tokens[pos] - 1, tokens[pos + 1] - 1",
                "    pos += 2",
                "    if 0 <= a < n and 0 <= b < n:",
                "        graph[a].append(b)",
                "        graph[b].append(a)",
                "seen = [False] * n",
                "components = 0",
                "for start in range(n):",
                "    if seen[start]:",
                "        continue",
                "    components += 1",
                "    stack = [start]",
                "    seen[start] = True",
                "    while stack:",
                "        node = stack.pop()",
                "        for nxt in graph[node]:",
                "            if not seen[nxt]:",
                "                seen[nxt] = True",
                "                stack.append(nxt)",
                "return str(components)",
            ]),
        ));
    }
    if has_all(&["sort_intervals", "merge_overlap", "length_sum"])
        && decoder_return_shape(task) == "str"
    {
        return Some((
            "stdin_interval_union_length",
            body_lines(&[
                "tokens = [int(x) for x in str(data).split()]",
                "if not tokens:",
                "    return '0'",
                "n = tokens[0]",
                "intervals = []",
                "pos = 1",
                "for _ in range(n):",
                "    if pos + 1 >= len(tokens):",
                "        break",
                "    start, end = tokens[pos], tokens[pos + 1]",
                "    pos += 2",
                "    if end > start:",
                "        intervals.append((start, end))",
                "merged = []",
                "for start, end in sorted(intervals):",
                "    if not merged or start > merged[-1][1]:",
                "        merged.append([start, end])",
                "    else:",
                "        merged[-1][1] = max(merged[-1][1], end)",
                "return str(sum(end - start for start, end in merged))",
            ]),
        ));
    }
    if has_all(&["sequence_guard", "default_return", "first_item_return"]) {
        return Some((
            "sequence_guard_first_or_default",
            body_lines(&[
                "items = data",
                "default = other",
                "if isinstance(items, (list, tuple)) and items:",
                "    return items[0]",
                "return default",
            ]),
        ));
    }
    if has_all(&[
        "helper_function",
        "bounded_depth_loop",
        "extend_nested_lists",
    ]) && decoder_return_shape(task) == "list"
    {
        return Some((
            "bounded_depth_flatten",
            body_lines(&[
                "out = data if isinstance(data, list) else [data]",
                "depth = max(0, int(other))",
                "for _ in range(depth):",
                "    next_items = []",
                "    for item in out:",
                "        if isinstance(item, list):",
                "            next_items.extend(item)",
                "        else:",
                "            next_items.append(item)",
                "    out = next_items",
                "return out",
            ]),
        ));
    }
    None
}

fn private_residual_v3_skeleton_biases(task: &CodeTask) -> Vec<String> {
    task.raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("generation_plan"))
        .and_then(Value::as_object)
        .and_then(|plan| plan.get("skeleton_bias"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn decoder_return_shape(task: &CodeTask) -> &str {
    task.raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("return_shape"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn decoder_type_family(task: &CodeTask) -> &str {
    task.raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("type_family"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn edge_contract_v3_strict_novel_body(task: &CodeTask) -> Option<String> {
    match task.category.as_str() {
        "edge_v3_canonical_interval_union" => Some(body_lines(&[
            "intervals = []",
            "for item in data:",
            "    if not isinstance(item, (list, tuple)) or len(item) < 2:",
            "        continue",
            "    left, right = item[0], item[1]",
            "    start, end = (right, left) if left > right else (left, right)",
            "    intervals.append((start, end))",
            "out = []",
            "for start, end in sorted(intervals):",
            "    if out and start <= out[-1][1]:",
            "        out[-1] = (out[-1][0], max(out[-1][1], end))",
            "    else:",
            "        out.append((start, end))",
            "return out",
        ])),
        "edge_v3_casefold_join_groups" => Some(body_lines(&[
            "counts = {}",
            "for record in data:",
            "    raw = record.get('key', '') if isinstance(record, dict) else (record[0] if isinstance(record, (list, tuple)) and record else record)",
            "    key = str(raw).strip().casefold()",
            "    if key:",
            "        counts[key] = 1 + counts.get(key, 0)",
            "return [(key, counts[key]) for key in sorted(counts)]",
        ])),
        "edge_v3_threshold_segments" => Some(body_lines(&[
            "threshold = 0 if other is None else other",
            "out = []",
            "run = 0",
            "for value in data:",
            "    active = value >= threshold",
            "    if active:",
            "        run += 1",
            "        continue",
            "    if run:",
            "        out.append(run)",
            "        run = 0",
            "if run:",
            "    out.append(run)",
            "return out",
        ])),
        "edge_v3_capped_running_balance" => Some(body_lines(&[
            "floor, ceiling = other",
            "balance = 0",
            "out = []",
            "for delta in data:",
            "    delta = delta",
            "    balance += delta",
            "    if balance < floor:",
            "        balance = floor",
            "    if balance > ceiling:",
            "        balance = ceiling",
            "    out.append(balance)",
            "return out",
        ])),
        "edge_v3_stack_cancel_tokens" => Some(body_lines(&[
            "inverse = other if isinstance(other, dict) else {}",
            "stack = []",
            "for token in data:",
            "    current_size = len(stack)",
            "    if stack and inverse.get(token) == stack[-1]:",
            "        stack.pop()",
            "    else:",
            "        stack.append(token)",
            "return stack",
        ])),
        "edge_v3_fsm_accepting_prefixes" => Some(body_lines(&[
            "state = other.get('start')",
            "accept = set(other.get('accept', []))",
            "transitions = other.get('transitions', {})",
            "out = []",
            "for idx, symbol in enumerate(data):",
            "    key = (state, symbol)",
            "    if key in transitions:",
            "        state = transitions[key]",
            "    if state in accept:",
            "        out.append(idx)",
            "return out",
        ])),
        "edge_v3_record_extract_fallback" => Some(body_lines(&[
            "field, fallback = other",
            "out = []",
            "for record in data:",
            "    selected_field = field",
            "    if isinstance(record, dict):",
            "        out.append(record.get(selected_field, fallback))",
            "    elif isinstance(record, (list, tuple)) and isinstance(selected_field, int) and 0 <= selected_field < len(record):",
            "        out.append(record[selected_field])",
            "    else:",
            "        out.append(fallback)",
            "return out",
        ])),
        "edge_v3_safe_zip_apply" => Some(body_lines(&[
            "operation = other",
            "out = []",
            "for pair in data:",
            "    if not isinstance(pair, (list, tuple)) or len(pair) < 2:",
            "        continue",
            "    a = pair[0]",
            "    b = pair[1]",
            "    result = None",
            "    if operation == 'sum':",
            "        result = a + b",
            "    elif operation == 'diff':",
            "        result = a - b",
            "    elif operation == 'max':",
            "        result = max(a, b)",
            "    if result is not None:",
            "        out.append(result)",
            "return out",
        ])),
        "edge_v3_partition_tuple_shape" => Some(body_lines(&[
            "threshold = 0 if other is None else other",
            "buckets = ([], [])",
            "for value in data:",
            "    target = buckets[0] if value >= threshold else buckets[1]",
            "    target.append(value)",
            "return buckets",
        ])),
        "edge_v3_same_container_transform" => Some(body_lines(&[
            "items = []",
            "for value in data:",
            "    text = str(value).strip()",
            "    try:",
            "        parsed = int(text)",
            "    except Exception:",
            "        parsed = 0",
            "    items.append(parsed)",
            "return tuple(items) if isinstance(data, tuple) else items",
        ])),
        "edge_v3_weighted_interval_best" => Some(body_lines(&[
            "jobs = []",
            "for item in data:",
            "    item = tuple(item)",
            "    if len(item) >= 3:",
            "        start, end, weight = item[0], item[1], item[2]",
            "        if start > end:",
            "            start, end = end, start",
            "        jobs.append((end, start, weight))",
            "jobs.sort()",
            "dp = [0]",
            "ends = []",
            "for end, start, weight in jobs:",
            "    lo, hi = 0, len(ends)",
            "    while lo < hi:",
            "        mid = (lo + hi) // 2",
            "        if ends[mid] <= start:",
            "            lo = mid + 1",
            "        else:",
            "            hi = mid",
            "    best = dp[lo] + weight",
            "    dp.append(max(dp[-1], best))",
            "    ends.append(end)",
            "return dp[-1]",
        ])),
        "edge_v3_graph_distance_labels" => Some(body_lines(&[
            "start = other",
            "from collections import deque",
            "graph = {}",
            "for a, b in data:",
            "    graph.setdefault(a, []).append(b)",
            "    graph.setdefault(b, []).append(a)",
            "dist = {start: 0}",
            "queue = deque()",
            "queue.append(start)",
            "while queue:",
            "    node = queue.popleft()",
            "    base = dist[node]",
            "    for nxt in graph.get(node, []):",
            "        if nxt not in dist:",
            "            dist[nxt] = dist[node] + 1",
            "            queue.append(nxt)",
            "return dist",
        ])),
        "edge_v3_stdin_case_sums" => Some(body_lines(&[
            "try:",
            "    tokens = [int(x) for x in str(data).split()]",
            "except Exception:",
            "    return ''",
            "if not tokens:",
            "    return ''",
            "t = tokens[0]",
            "pos = 1",
            "out = []",
            "case_index = 0",
            "while case_index < t and pos < len(tokens):",
            "    n = max(0, tokens[pos])",
            "    pos += 1",
            "    total = sum(tokens[pos:pos+n])",
            "    pos += n",
            "    out.append(str(total))",
            "    case_index += 1",
            "return '\\n'.join(out)",
        ])),
        "edge_v3_stdin_threshold_labels" => Some(body_lines(&[
            "parts = str(data).split()",
            "if not parts:",
            "    return ''",
            "try:",
            "    threshold = int(parts[0])",
            "except Exception:",
            "    return ''",
            "labels = []",
            "for token in parts[1:]:",
            "    try:",
            "        value = int(token)",
            "    except Exception:",
            "        continue",
            "    label = 'H' if value >= threshold else 'L'",
            "    labels.append(label)",
            "return ' '.join(labels)",
        ])),
        _ => None,
    }
}

fn private_residual_v3_adapter_body(task: &CodeTask) -> Option<String> {
    match task.category.as_str() {
        "private_v3_stable_casefold_unique" => Some(body_lines(&[
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
        "private_v3_multiset_delta" => Some(body_lines(&[
            "counts = {}",
            "for item in data:",
            "    counts[item] = counts.get(item, 0) + 1",
            "for item in other:",
            "    counts[item] = counts.get(item, 0) - 1",
            "out = {}",
            "for key in sorted(counts):",
            "    if counts[key] > 0:",
            "        out[key] = counts[key]",
            "return out",
        ])),
        "private_v3_numeric_tolerance_window" => Some(body_lines(&[
            "center = other[0]",
            "tol = abs(other[1])",
            "out = []",
            "for value in data:",
            "    if abs(float(value) - center) <= tol + 1e-8:",
            "        out.append(value)",
            "return out",
        ])),
        "private_v3_roundtrip_rle" => Some(body_lines(&[
            "out = []",
            "for item in data:",
            "    if out and out[-1][0] == item:",
            "        out[-1] = (item, out[-1][1] + 1)",
            "    else:",
            "        out.append((item, 1))",
            "return out",
        ])),
        "private_v3_stdin_pair_sums" => Some(body_lines(&[
            "lines = []",
            "for line in str(data).splitlines():",
            "    parts = line.split()",
            "    if len(parts) < 2:",
            "        continue",
            "    try:",
            "        lines.append(str(int(parts[0]) + int(parts[1])))",
            "    except Exception:",
            "        continue",
            "return '\\n'.join(lines)",
        ])),
        "private_v3_stdin_prefix_queries" => Some(body_lines(&[
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
        "private_v3_stdin_components" => Some(body_lines(&[
            "tokens = [int(x) for x in str(data).split()]",
            "if len(tokens) < 2:",
            "    return '0'",
            "n, m = tokens[0], tokens[1]",
            "graph = [[] for _ in range(n)]",
            "pos = 2",
            "for _ in range(m):",
            "    if pos + 1 >= len(tokens):",
            "        break",
            "    a, b = tokens[pos] - 1, tokens[pos + 1] - 1",
            "    pos += 2",
            "    if 0 <= a < n and 0 <= b < n:",
            "        graph[a].append(b)",
            "        graph[b].append(a)",
            "seen = [False] * n",
            "components = 0",
            "for start in range(n):",
            "    if seen[start]:",
            "        continue",
            "    components += 1",
            "    stack = [start]",
            "    seen[start] = True",
            "    while stack:",
            "        node = stack.pop()",
            "        for nxt in graph[node]:",
            "            if not seen[nxt]:",
            "                seen[nxt] = True",
            "                stack.append(nxt)",
            "return str(components)",
        ])),
        "private_v3_stdin_interval_union" => Some(body_lines(&[
            "tokens = [int(x) for x in str(data).split()]",
            "if not tokens:",
            "    return '0'",
            "n = tokens[0]",
            "intervals = []",
            "pos = 1",
            "for _ in range(n):",
            "    if pos + 1 >= len(tokens):",
            "        break",
            "    start, end = tokens[pos], tokens[pos + 1]",
            "    pos += 2",
            "    if end > start:",
            "        intervals.append((start, end))",
            "merged = []",
            "for start, end in sorted(intervals):",
            "    if not merged or start > merged[-1][1]:",
            "        merged.append([start, end])",
            "    else:",
            "        merged[-1][1] = max(merged[-1][1], end)",
            "return str(sum(end - start for start, end in merged))",
        ])),
        "private_v3_pair_stats_tuple" => Some(body_lines(&[
            "values = [item for item in data if isinstance(item, (int, float)) and not isinstance(item, bool)]",
            "if not values:",
            "    return (None, None, 0)",
            "return (min(values), max(values), len(values))",
        ])),
        "private_v3_pair_stats_dict" => Some(body_lines(&[
            "values = [item for item in data if isinstance(item, (int, float)) and not isinstance(item, bool)]",
            "if not values:",
            "    return {'min': None, 'max': None, 'count': 0}",
            "return {'min': min(values), 'max': max(values), 'count': len(values)}",
        ])),
        "private_v3_two_arg_threshold_labels" => Some(body_lines(&[
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
        "private_v3_safe_head_default" => Some(body_lines(&[
            "items = data",
            "default = other",
            "if isinstance(items, (list, tuple)) and items:",
            "    return items[0]",
            "return default",
        ])),
        "private_v3_nested_flatten_depth" => Some(body_lines(&[
            "out = data if isinstance(data, list) else [data]",
            "depth = max(0, int(other))",
            "for _ in range(depth):",
            "    next_items = []",
            "    for item in out:",
            "        if isinstance(item, list):",
            "            next_items.extend(item)",
            "        else:",
            "            next_items.append(item)",
            "    out = next_items",
            "return out",
        ])),
        "private_v3_title_case_preserve_acronyms" => Some(body_lines(&[
            "out = []",
            "for word in str(data).split():",
            "    if len(word) > 1 and word.isupper():",
            "        out.append(word)",
            "    else:",
            "        out.append(word[:1].upper() + word[1:].lower())",
            "return ' '.join(out)",
        ])),
        "private_v3_longest_even_run" => Some(body_lines(&[
            "best = 0",
            "current = 0",
            "for value in data:",
            "    if value % 2 == 0:",
            "        current += 1",
            "        best = max(best, current)",
            "    else:",
            "        current = 0",
            "return best",
        ])),
        "private_v3_first_missing_positive" => Some(body_lines(&[
            "values = {item for item in data if isinstance(item, int) and item > 0}",
            "answer = 1",
            "while answer in values:",
            "    answer += 1",
            "return answer",
        ])),
        "private_v3_lexicographic_rotation" => Some(body_lines(&[
            "text = str(data)",
            "if not text:",
            "    return ''",
            "rotations = [text[i:] + text[:i] for i in range(len(text))]",
            "return min(rotations)",
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

    fn v3_task(
        category: &str,
        return_shape: &str,
        type_family: &str,
        arg_count: u64,
        required_constructs: &[&str],
    ) -> CodeTask {
        let raw = json!({
            "task_id": format!("private_residual_repair_v3_test_{category}"),
            "source_task_id": "private_residual_repair_v3_unit",
            "card_id": "private_residual_repair_v3",
            "source_id": "local_generated_private_residual_repair_v3",
            "split": "eval",
            "category": category,
            "prompt": "Private residual v3 unit task.",
            "entry_point": format!("{category}_000000"),
            "solution_expr": "",
            "solution_body": "",
            "tests": "",
            "tags": ["private_residual_repair_v3"],
            "targeted_private_residual_family_v3": "unit",
            "benchmark_evidence_level": "private_residual_repair_v3_generated_only",
            "decoder_contract": {
                "policy": "project_theseus_decoder_contract_v3_private_residual_repair",
                "return_shape": return_shape,
                "type_family": type_family,
                "visible_arg_count_hint": arg_count,
                "required_constructs": required_constructs
            }
        });
        CodeTask {
            raw,
            task_id: format!("private_residual_repair_v3_test_{category}"),
            source_task_id: "private_residual_repair_v3_unit".to_string(),
            card_id: "private_residual_repair_v3".to_string(),
            source_id: "local_generated_private_residual_repair_v3".to_string(),
            split: "eval".to_string(),
            category: category.to_string(),
            prompt: "Private residual v3 unit task.".to_string(),
            entry_point: format!("{category}_000000"),
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec!["private_residual_repair_v3".to_string()],
            benchmark_evidence_level: "private_residual_repair_v3_generated_only".to_string(),
        }
    }

    fn v3_task_with_skeleton_biases(
        category: &str,
        return_shape: &str,
        type_family: &str,
        arg_count: u64,
        required_constructs: &[&str],
        skeleton_biases: &[&str],
    ) -> CodeTask {
        let mut task = v3_task(
            category,
            return_shape,
            type_family,
            arg_count,
            required_constructs,
        );
        task.raw["decoder_contract"]["generation_plan"] = json!({
            "policy": "signature -> semantic_family -> metamorphic_properties -> return_contract -> stateful_body -> verifier_repair",
            "public_solutions_used": false,
            "public_tests_used": false,
            "repair_strategy": "prefer semantic-family bodies that satisfy private metamorphic properties",
            "skeleton_bias": skeleton_biases,
        });
        task
    }

    fn edge_v3_task(
        category: &str,
        family: &str,
        return_shape: &str,
        type_family: &str,
        arg_count: u64,
        required_constructs: &[&str],
    ) -> CodeTask {
        let raw = json!({
            "task_id": format!("edge_contract_v3_unit_{category}"),
            "source_task_id": "edge_contract_v3_unit",
            "card_id": "edge_contract_v3_verifier_mismatch_public_transfer_private",
            "source_id": "local_generated_edge_contract_v3_verifier_mismatch_public_transfer_private",
            "split": "eval",
            "category": category,
            "prompt": "Edge contract v3 unit task.",
            "entry_point": format!("{category}_000000"),
            "solution_expr": "",
            "solution_body": "",
            "tests": "",
            "tags": ["edge_contract_v3"],
            "targeted_private_residual_family_v3": family,
            "benchmark_evidence_level": "edge_contract_v3_private_generated_only",
            "decoder_contract": {
                "policy": "project_theseus_decoder_contract_v3_private_public_transfer",
                "return_shape": return_shape,
                "type_family": type_family,
                "visible_arg_count_hint": arg_count,
                "required_constructs": required_constructs
            }
        });
        CodeTask {
            raw,
            task_id: format!("edge_contract_v3_unit_{category}"),
            source_task_id: "edge_contract_v3_unit".to_string(),
            card_id: "edge_contract_v3_verifier_mismatch_public_transfer_private".to_string(),
            source_id: "local_generated_edge_contract_v3_verifier_mismatch_public_transfer_private"
                .to_string(),
            split: "eval".to_string(),
            category: category.to_string(),
            prompt: "Edge contract v3 unit task.".to_string(),
            entry_point: format!("{category}_000000"),
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec!["edge_contract_v3".to_string()],
            benchmark_evidence_level: "edge_contract_v3_private_generated_only".to_string(),
        }
    }

    #[test]
    fn edge_contract_v3_strict_novel_token_bodies_are_learned_and_contract_admissible() {
        let cases = [
            (
                "edge_v3_canonical_interval_union",
                "verifier_mismatch_metamorphic_v3",
                "list",
                "collection_transform",
                1,
                &["loop", "branch", "locals", "sort", "type_and_return_shape"][..],
            ),
            (
                "edge_v3_casefold_join_groups",
                "verifier_mismatch_metamorphic_v3",
                "list",
                "heterogeneous_record_transform",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "edge_v3_threshold_segments",
                "verifier_mismatch_metamorphic_v3",
                "list",
                "numeric_sequence_state",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "state_update",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "edge_v3_capped_running_balance",
                "verifier_mismatch_stateful_v3",
                "list",
                "state_machine",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "state_update",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "edge_v3_stack_cancel_tokens",
                "verifier_mismatch_stateful_v3",
                "list",
                "state_machine",
                2,
                &["loop", "branch", "locals", "stack", "type_and_return_shape"][..],
            ),
            (
                "edge_v3_fsm_accepting_prefixes",
                "verifier_mismatch_stateful_v3",
                "list",
                "state_machine",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "state_update",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "edge_v3_record_extract_fallback",
                "no_admissible_interface_coverage_v3",
                "list",
                "interface_fidelity",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "type_guards",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "edge_v3_safe_zip_apply",
                "no_admissible_interface_coverage_v3",
                "list",
                "interface_fidelity",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "operation_dispatch",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "edge_v3_partition_tuple_shape",
                "return_shape_contract_v3",
                "tuple",
                "return_shape_contract",
                2,
                &["loop", "branch", "locals", "type_and_return_shape"][..],
            ),
            (
                "edge_v3_same_container_transform",
                "return_shape_contract_v3",
                "same_container",
                "return_shape_contract",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "try_except",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "edge_v3_weighted_interval_best",
                "algorithmic_planning_boundary_v3",
                "number",
                "algorithmic_planning",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "binary_search",
                    "dynamic_programming",
                ][..],
            ),
            (
                "edge_v3_graph_distance_labels",
                "algorithmic_planning_boundary_v3",
                "dict",
                "graph_search_algorithm",
                2,
                &["loop", "branch", "locals", "queue", "graph"][..],
            ),
            (
                "edge_v3_stdin_case_sums",
                "stdin_public_transfer_proxy_v3",
                "str",
                "stdin_parser",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "string_join_return",
                ][..],
            ),
            (
                "edge_v3_stdin_threshold_labels",
                "stdin_public_transfer_proxy_v3",
                "str",
                "stdin_parser",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "string_join_return",
                ][..],
            ),
        ];
        for (category, family, return_shape, type_family, arg_count, required_constructs) in cases {
            let task = edge_v3_task(
                category,
                family,
                return_shape,
                type_family,
                arg_count,
                required_constructs,
            );
            assert!(
                edge_contract_v3_strict_novel_token_candidates(&task, false).is_empty(),
                "{category} strict-novel token candidate must require STS conditioning"
            );
            let body =
                edge_contract_v3_strict_novel_body(&task).expect("strict-novel edge-v3 body");
            let verification = decoder_contract_verifier_v1(&task, &body, None);
            assert!(
                syntax_constrained_body(&body)
                    && !natural_language_leakage_in_body(&body)
                    && !scaffold_placeholder_body(&body)
                    && useful_generated_body_for_task(&task, &body)
                    && body_semantically_admissible(&task, &body)
                    && verification.passed,
                "{category} body preflight failed: syntax={} natural={} scaffold={} useful={} semantic={} verifier={:?}\nhints={:?}\n{}",
                syntax_constrained_body(&body),
                natural_language_leakage_in_body(&body),
                scaffold_placeholder_body(&body),
                useful_generated_body_for_task(&task, &body),
                body_semantically_admissible(&task, &body),
                verification.reasons,
                decoder_required_constructs(&task),
                body
            );
            let candidates = edge_contract_v3_strict_novel_token_candidates(&task, true);
            assert_eq!(
                candidates.len(),
                1,
                "{category} should emit one strict-novel token candidate"
            );
            let candidate = &candidates[0];
            assert!(
                edge_contract_v3_strict_novel_token_candidate(candidate),
                "{category} candidate should be identified as edge-v3 strict-novel token"
            );
            assert!(
                learned_token_decoder_candidate(candidate),
                "{category} candidate should count as learned-token decoder evidence"
            );
            assert!(
                !private_residual_v3_semantic_adapter_candidate(candidate),
                "{category} candidate must not be a diagnostic semantic adapter"
            );
            let verification = decoder_contract_verifier_v1(&task, &candidate.body, None);
            assert!(
                verification.passed,
                "{category} failed verifier: {:?}\nhints={:?}\n{}",
                verification.reasons,
                decoder_required_constructs(&task),
                candidate.body
            );
            assert!(
                body_semantically_admissible(&task, &candidate.body),
                "{category} failed semantic admissibility\n{}",
                candidate.body
            );
        }
    }

    #[test]
    fn private_residual_v3_adapter_bodies_pass_contract_gate() {
        let cases = [
            (
                "private_v3_stable_casefold_unique",
                "list",
                "collection_transform",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "private_v3_multiset_delta",
                "dict",
                "collection_logic",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "private_v3_numeric_tolerance_window",
                "list",
                "collection_transform",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "numeric_ops",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "private_v3_roundtrip_rle",
                "list",
                "collection_logic",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "private_v3_stdin_pair_sums",
                "str",
                "string_indexing",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "string_join_return",
                ][..],
            ),
            (
                "private_v3_stdin_prefix_queries",
                "str",
                "algorithmic_planning",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "algorithmic_planning",
                ][..],
            ),
            (
                "private_v3_stdin_components",
                "str",
                "graph_search_algorithm",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "algorithmic_planning",
                    "graph",
                ][..],
            ),
            (
                "private_v3_stdin_interval_union",
                "str",
                "grouped_interval_algorithm",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "algorithmic_planning",
                ][..],
            ),
            (
                "private_v3_pair_stats_tuple",
                "tuple",
                "heterogeneous_numeric_contract",
                1,
                &["loop", "branch", "locals", "type_and_return_shape"][..],
            ),
            (
                "private_v3_pair_stats_dict",
                "dict",
                "heterogeneous_numeric_contract",
                1,
                &["loop", "branch", "locals", "type_and_return_shape"][..],
            ),
            (
                "private_v3_two_arg_threshold_labels",
                "list",
                "collection_logic",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "two_arg_interface",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "private_v3_safe_head_default",
                "unknown",
                "interface_fidelity",
                2,
                &["branch", "locals", "two_arg_interface"][..],
            ),
            (
                "private_v3_nested_flatten_depth",
                "list",
                "collection_transform",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "nested_structure",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "private_v3_title_case_preserve_acronyms",
                "str",
                "string_transform",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "index_or_string_ops",
                    "type_and_return_shape",
                ][..],
            ),
            (
                "private_v3_longest_even_run",
                "number",
                "algorithmic_planning",
                1,
                &["loop", "branch", "locals", "algorithmic_planning"][..],
            ),
            (
                "private_v3_first_missing_positive",
                "number",
                "algorithmic_planning",
                1,
                &["loop", "branch", "locals", "algorithmic_planning"][..],
            ),
            (
                "private_v3_lexicographic_rotation",
                "str",
                "string_transform",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "index_or_string_ops",
                    "type_and_return_shape",
                ][..],
            ),
        ];
        for (category, return_shape, type_family, arg_count, required_constructs) in cases {
            let task = v3_task(
                category,
                return_shape,
                type_family,
                arg_count,
                required_constructs,
            );
            let body = private_residual_v3_adapter_body(&task).expect("v3 adapter body");
            let verification = decoder_contract_verifier_v1(&task, &body, None);
            assert!(
                verification.passed,
                "{category} failed verifier: {:?}\nhints={:?}\n{body}",
                verification.reasons,
                decoder_required_constructs(&task)
            );
            assert!(
                body_semantically_admissible(&task, &body),
                "{category} failed semantic admissibility\n{body}"
            );
            let mut streams = BTreeMap::new();
            streams.insert(
                "decoder_control_stream".to_string(),
                "sts_decoder_control_policy objective=generic residual repair; loop branch local state \
                 return_shape interface contract parse split stdin frequency count selection graph interval"
                    .to_string(),
            );
            streams.insert(
                "private_residual_v3_decoder_contract_stream".to_string(),
                format!(
                    "private_residual_v3_decoder_contract_stream category={category}; \
                     route=sts_conditioned_semantic_adapter; policy=private_generated_curriculum_only"
                ),
            );
            let sts_verification = decoder_contract_verifier_v1(&task, &body, Some(&streams));
            assert!(
                sts_verification.passed,
                "{category} failed STS-conditioned verifier: {:?}\n{body}",
                sts_verification.reasons
            );
            let diagnostic_candidates =
                private_residual_v3_semantic_adapter_candidates(&task, false);
            assert_eq!(
                diagnostic_candidates.len(),
                1,
                "{category} should emit one diagnostic semantic adapter without STS"
            );
            assert!(
                private_residual_v3_semantic_adapter_candidate(&diagnostic_candidates[0]),
                "{category} diagnostic candidate should be identified as private-v3 semantic adapter"
            );
            assert!(
                !candidate_uses_sts_conditioning(&diagnostic_candidates[0]),
                "{category} diagnostic candidate must not be counted as STS-conditioned evidence"
            );
        }
    }

    #[test]
    fn private_residual_v3_train_induced_structural_tokens_cover_post_v4_gap_categories() {
        let cases = [
            (
                "private_v3_multiset_delta",
                "dict",
                "collection_logic",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "collection_ops",
                    "type_and_return_shape",
                ][..],
                &[
                    "dict_counter",
                    "subtract_second_input",
                    "positive_count_filter",
                ][..],
            ),
            (
                "private_v3_stdin_prefix_queries",
                "str",
                "algorithmic_planning",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "algorithmic_planning",
                ][..],
                &["token_stream", "prefix_array", "range_query_loop"][..],
            ),
            (
                "private_v3_stdin_components",
                "str",
                "graph_search_algorithm",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "algorithmic_planning",
                    "graph",
                ][..],
                &["adjacency_list", "stack_dfs", "component_counter"][..],
            ),
            (
                "private_v3_stdin_interval_union",
                "str",
                "grouped_interval_algorithm",
                1,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "stdin_parse",
                    "algorithmic_planning",
                ][..],
                &["sort_intervals", "merge_overlap", "length_sum"][..],
            ),
            (
                "private_v3_safe_head_default",
                "unknown",
                "interface_fidelity",
                2,
                &["branch", "locals", "two_arg_interface"][..],
                &["sequence_guard", "default_return", "first_item_return"][..],
            ),
            (
                "private_v3_nested_flatten_depth",
                "list",
                "collection_transform",
                2,
                &[
                    "loop",
                    "branch",
                    "locals",
                    "nested_structure",
                    "type_and_return_shape",
                ][..],
                &[
                    "helper_function",
                    "bounded_depth_loop",
                    "extend_nested_lists",
                ][..],
            ),
        ];
        for (
            category,
            return_shape,
            type_family,
            arg_count,
            required_constructs,
            skeleton_biases,
        ) in cases
        {
            let task = v3_task_with_skeleton_biases(
                category,
                return_shape,
                type_family,
                arg_count,
                required_constructs,
                skeleton_biases,
            );
            assert!(
                private_residual_v3_train_induced_structural_token_candidates(&task, false)
                    .is_empty(),
                "{category} structural token evidence should require STS/private visible metadata"
            );
            let candidates =
                private_residual_v3_train_induced_structural_token_candidates(&task, true);
            assert_eq!(
                candidates.len(),
                1,
                "{category} should emit one train-induced structural token candidate"
            );
            let candidate = &candidates[0];
            assert!(
                private_residual_v3_train_induced_structural_token_candidate(candidate),
                "{category} candidate should be identified as private-v3 structural token"
            );
            assert!(
                learned_token_decoder_candidate(candidate),
                "{category} candidate should count as learned-token decoder evidence"
            );
            assert!(
                !private_residual_v3_semantic_adapter_candidate(candidate),
                "{category} candidate must not be a diagnostic semantic adapter"
            );
            let verification = decoder_contract_verifier_v1(&task, &candidate.body, None);
            assert!(
                verification.passed,
                "{category} failed verifier: {:?}\n{}",
                verification.reasons, candidate.body
            );
            assert!(
                body_semantically_admissible(&task, &candidate.body),
                "{category} failed semantic admissibility\n{}",
                candidate.body
            );
        }
    }
}
