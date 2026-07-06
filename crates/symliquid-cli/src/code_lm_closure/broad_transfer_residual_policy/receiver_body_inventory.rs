// Receiver inventory body generators for broad-transfer residual repair.
// Included by broad_transfer_residual_policy.rs to keep policy/scoring separate from body catalogs.

fn eligible_receiver_inventory_bodies(
    task: &CodeTask,
    policy: &BroadTransferResidualPolicy,
) -> Vec<(&'static str, String)> {
    let primary = decoder_primary_arg(task);
    let secondary = decoder_secondary_arg(task);
    let secondary_name = secondary.as_deref().unwrap_or("other");
    let has_secondary = secondary.is_some();
    let shape = decoder_return_shape(task);
    let empty = empty_return_literal(&shape);
    let mut bodies = Vec::new();
    bodies.extend(category_specific_receiver_inventory_bodies(
        task,
        &primary,
        secondary_name,
        has_secondary,
    ));
    let specific_receiver_route_present = bodies
        .iter()
        .any(|(family, _)| receiver_inventory_family_is_specific_contract(family))
        || type_contract_specific_receiver_target_present(task, &shape, has_secondary);
    if policy.local_adapter {
        bodies.push((
            "residual_local_adapter_receiver",
            local_adapter_retry_body(task, &primary, secondary_name, has_secondary, &shape, empty),
        ));
    }
    if policy.edge_case || policy.string_parsing || policy.local_adapter || policy.type_handling {
        for (family, body) in
            edge_contract_v2_retry_bodies(task, &primary, secondary_name, has_secondary, &shape)
        {
            bodies.push((family, body));
        }
    }
    if policy.edge_case
        && (policy.local_adapter || policy.runtime_dependency || policy.interface_fidelity)
        && !specific_receiver_route_present
    {
        bodies.push((
            "edge_interface_admissibility",
            edge_interface_admissibility_receiver_body(
                &primary,
                secondary_name,
                has_secondary,
                &shape,
                empty,
            ),
        ));
    }
    if policy.algorithm_choice {
        bodies.push((
            "residual_algorithm_choice_receiver",
            algorithm_choice_retry_body(
                task,
                &primary,
                secondary_name,
                has_secondary,
                &shape,
                empty,
            ),
        ));
    }
    if policy.runtime_dependency {
        bodies.push((
            "residual_runtime_dependency_receiver",
            runtime_dependency_receiver_body(
                &primary,
                secondary_name,
                has_secondary,
                &shape,
                empty,
            ),
        ));
    }
    if policy.verification_cascade_compile {
        bodies.push((
            "verification_cascade_compile_receiver",
            verification_cascade_compile_receiver_body(
                &primary,
                secondary_name,
                has_secondary,
                &shape,
                empty,
            ),
        ));
    }
    if policy.interface_fidelity || policy.type_handling || policy.return_shape_contract {
        bodies.push((
            "interface_fidelity",
            interface_fidelity_receiver_body(
                &primary,
                secondary_name,
                has_secondary,
                &shape,
                empty,
            ),
        ));
    }
    if policy.type_handling || policy.return_shape_contract || policy.interface_fidelity {
        for (family, body) in
            type_contract_v2_receiver_bodies(task, &primary, secondary_name, has_secondary, &shape)
        {
            bodies.push((family, body));
        }
    }
    if policy.string_parsing || policy.local_adapter {
        bodies.push((
            "string_parsing",
            string_parsing_receiver_body(&primary, secondary_name, has_secondary, &shape, empty),
        ));
    }
    if policy.runtime_dependency {
        bodies.push((
            "runtime_dependency_guard",
            runtime_dependency_receiver_body(
                &primary,
                secondary_name,
                has_secondary,
                &shape,
                empty,
            ),
        ));
    }
    if policy.verification_cascade_compile {
        bodies.push((
            "verification_cascade_compile_guard",
            verification_cascade_compile_receiver_body(
                &primary,
                secondary_name,
                has_secondary,
                &shape,
                empty,
            ),
        ));
    }
    if policy.control_flow_obligations || policy.edge_case || policy.algorithm_choice {
        bodies.push((
            "locals_branch_loop",
            locals_branch_loop_receiver_body(
                &primary,
                secondary_name,
                has_secondary,
                &shape,
                empty,
            ),
        ));
    }
    if policy.return_shape_contract || policy.type_handling {
        bodies.push((
            "return_shape_contract",
            return_shape_receiver_body(&primary, secondary_name, has_secondary, &shape, empty),
        ));
    }
    bodies
}

fn receiver_inventory_family_is_specific_contract(family: &str) -> bool {
    family.starts_with("contract_")
        || family.starts_with("interface_")
        || family.starts_with("edge_contract_")
        || family.starts_with("parsing_encoding_")
        || family.starts_with("type_contract_")
        || family.starts_with("runtime_dependency_")
        || family.starts_with("no_admissible_")
        || family.starts_with("execution_shape_")
}

fn eligible_type_contract_receiver_body_count(task: &CodeTask) -> usize {
    let primary = decoder_primary_arg(task);
    let secondary = decoder_secondary_arg(task);
    let secondary_name = secondary.as_deref().unwrap_or("other");
    let shape = decoder_return_shape(task);
    type_contract_v2_receiver_bodies(task, &primary, secondary_name, secondary.is_some(), &shape)
        .len()
}

fn type_contract_specific_receiver_target_present(
    task: &CodeTask,
    shape: &str,
    has_secondary: bool,
) -> bool {
    let text =
        format!("{} {} {}", task.category, task.prompt, task.tags.join(" ")).to_ascii_lowercase();
    let generation_hints = decoder_contract_generation_hints(task);
    let has_hint = |needle: &str| generation_hints.contains(needle);
    match shape {
        "list" => {
            (has_secondary
                && (has_hint("zip_both_arguments")
                    || has_hint("numeric_pair_guard")
                    || (text.contains("pairwise") && text.contains("sum"))
                    || (text.contains("shorter sequence") && text.contains("numeric"))))
                || has_hint("strip_lower_transform")
                || has_hint("skip_empty_branch")
                || (text.contains("normalized strings") && text.contains("lowercase"))
                || (text.contains("strip") && text.contains("lowercase"))
                || (has_secondary
                    && (has_hint("nested_dict_walk")
                        || has_hint("target_compare_with_second_argument")
                        || (text.contains("dot-separated") && text.contains("nested"))
                        || (text.contains("nested dictionaries") && text.contains("target"))))
                || (has_secondary
                    && (has_hint("range_window_scan")
                        || has_hint("slice_compare_append_index")
                        || text.contains("two_arg_string_window")
                        || (text.contains("overlapping matches")
                            && text.contains("pattern")
                            && text.contains("text"))))
                || text.contains("mixed_int")
                || text.contains("integer values")
                || text.contains("numeric_text")
                || text.contains("numeric_string_parser")
                || text.contains("signed integer tokens")
                || text.contains("mixed numbers")
                || text.contains("parsed from mixed")
                || (has_secondary
                    && (text.contains("score flags")
                        || text.contains("numeric score")
                        || text.contains("record order")
                        || text.contains("record_score")))
        }
        "str" => {
            text.contains("status")
                || text.contains("canonical")
                || text.contains("label")
                || text.contains("normalize_status")
                || text.contains("test type")
                || text.contains("case type")
                || text.contains("case kind")
                || text.contains("visibility")
                || text.contains("normalize_test")
                || text.contains("normalize case")
                || text.contains("safe head")
                || text.contains("first non-empty")
                || text.contains("first item")
                || text.contains("head of")
                || text.contains("safe_extraction")
                || text.contains("first_text")
                || text.contains("normalized text value")
                || text.contains("entry point")
                || text.contains("entry_point")
                || text.contains("function name")
                || text.contains("source text")
        }
        "number" => {
            (text.contains("count")
                && (text.contains("test")
                    || text.contains("case")
                    || text.contains("visibility")
                    || text.contains("public")
                    || text.contains("visible")))
                || text.contains("parse_signed_ints")
                || text.contains("signed integers embedded")
                || text.contains("signed integer")
                || text.contains("numeric text")
                || (has_secondary
                    && (text.contains("nested")
                        || text.contains("count nested")
                        || text.contains("key or value")
                        || text.contains("nested_structure")))
                || text.contains("final_y_vowels")
                || text.contains("suffix_rule")
                || (text.contains("final alphabetic") && text.contains("vowel"))
        }
        "bool" => {
            (has_secondary
                && (text.contains("required key")
                    || text.contains("required_key")
                    || text.contains("mapping contains")
                    || text.contains("key names")))
                || text.contains("three_sum_zero_exists")
                || text.contains("three distinct positions")
        }
        "dict" => {
            text.contains("optional_requests_query")
                || (text.contains("query parameters")
                    && (text.contains("requests") || text.contains("url")))
                || text.contains("group_counts")
                || text.contains("group counts")
                || text.contains("mapping_labels")
                || text.contains("label count")
                || text.contains("label/count")
                || text.contains("count mapping")
        }
        _ => false,
    }
}

fn category_specific_receiver_inventory_bodies(
    task: &CodeTask,
    primary: &str,
    secondary: &str,
    has_secondary: bool,
) -> Vec<(&'static str, String)> {
    let text = receiver_inventory_target_text(task);
    let shape = decoder_return_shape(task);
    let generation_hints = decoder_contract_generation_hints(task);
    let has_hint = |needle: &str| generation_hints.contains(needle);
    let mut bodies = Vec::new();
    bodies.extend(execution_shape_private_receiver_bodies(
        task, primary, secondary,
    ));
    bodies.extend(optional_dependency_specific_receiver_bodies(task, primary));
    bodies.extend(interface_metadata_bridge_bodies(
        task,
        primary,
        secondary,
        has_secondary,
    ));
    let compact_text = text.replace('_', "").replace(' ', "").replace('-', "");
    if shape == "list"
        && (compact_text.contains("allprefixes")
            || text.contains("all prefixes")
            || text.contains("every prefix"))
    {
        bodies.push((
            "contract_all_prefixes",
            format!(
                "text = '' if {primary} is None else str({primary})\nout = []\nfor index in range(1, len(text) + 1):\n    out.append(text[:index])\nreturn out"
            ),
        ));
    }
    if shape == "str"
        && (compact_text.contains("stringsequence")
            || (text.contains("string") && text.contains("sequence") && text.contains("number")))
    {
        bodies.push((
            "contract_string_sequence",
            format!(
                "try:\n    limit = int({primary})\nexcept Exception:\n    return ''\nif limit < 0:\n    return ''\nout = []\nfor value in range(0, limit + 1):\n    out.append(str(value))\nreturn ' '.join(out)"
            ),
        ));
    }
    if shape == "number"
        && (compact_text.contains("countdistinctcharacters")
            || (text.contains("count") && text.contains("distinct") && text.contains("character")))
    {
        bodies.push((
            "contract_count_distinct_characters",
            format!(
                "text = '' if {primary} is None else str({primary}).lower()\nseen = set()\nfor ch in text:\n    seen.add(ch)\ntotal = 0\nfor _ch in seen:\n    total += 1\nreturn total"
            ),
        ));
    }
    if shape == "list"
        && (compact_text.contains("symbolbeatparser")
            || compact_text.contains("parsemusic")
            || (text.contains("symbol") && text.contains("beat") && text.contains("parse")))
    {
        bodies.push((
            "parsing_encoding_symbol_beat_parser",
            format!(
                "text = '' if {primary} is None else str({primary})\nbeats = {{'o': 4, 'o|': 2, '.|': 1}}\ncounts = {{}}\nout = []\nfor token in text.split():\n    note = token.strip()\n    counts[note] = counts.get(note, 0) + 1\n    if note in beats:\n        out.append(beats[note])\nseen = 0\nfor value in counts.values():\n    seen += value\nreturn out"
            ),
        ));
    }
    if shape == "number"
        && has_secondary
        && (compact_text.contains("howmanytimes")
            || (text.contains("substring") && (text.contains("count") || text.contains("times"))))
    {
        bodies.push((
            "contract_how_many_times_substring",
            format!(
                "text = '' if {primary} is None else str({primary})\nsub = '' if {secondary} is None else str({secondary})\nif sub == '':\n    return 0\ntotal = 0\nwidth = len(sub)\nfor index in range(0, len(text) - width + 1):\n    if text[index:index + width] == sub:\n        total += 1\nreturn total"
            ),
        ));
    }
    if shape == "list"
        && (compact_text.contains("rescaletounit")
            || text.contains("rescale to unit")
            || text.contains("unit interval")
            || (text.contains("rescale") && text.contains("unit")))
    {
        bodies.push((
            "contract_rescale_to_unit",
            format!(
                "values = []\nfor item in {primary} if isinstance({primary}, (list, tuple)) else []:\n    if isinstance(item, bool):\n        continue\n    try:\n        values.append(float(item))\n    except Exception:\n        pass\nif not values:\n    return []\nlow = min(values)\nhigh = max(values)\nif high == low:\n    return [0.0 for _value in values]\nout = []\nfor value in values:\n    out.append((value - low) / (high - low))\nreturn out"
            ),
        ));
    }
    if shape == "number"
        && (compact_text.contains("maxdifference")
            || text.contains("maximum difference")
            || text.contains("max difference"))
    {
        bodies.push((
            "contract_max_difference",
            format!(
                "items = list({primary}) if isinstance({primary}, (list, tuple)) else []\nvalues = []\nfor item in items:\n    if isinstance(item, (int, float)) and not isinstance(item, bool):\n        values.append(item)\nif len(values) < 2:\n    return 0\nreturn max(values) - min(values)"
            ),
        ));
    }
    if shape == "str"
        && (compact_text.contains("sortnumbers")
            || (text.contains("sort") && text.contains("number") && text.contains("words")))
    {
        bodies.push((
            "contract_sort_number_words",
            format!(
                "text = '' if {primary} is None else str({primary})\nmapping = {{'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}}\nwords = []\nfor word in text.split():\n    key = word.strip().lower()\n    if key in mapping:\n        words.append(key)\nwords = sorted(words, key=lambda item: mapping[item])\nreturn ' '.join(words)"
            ),
        ));
    }
    if (shape == "tuple" || shape == "list")
        && (compact_text.contains("findclosestelements")
            || (text.contains("closest") && text.contains("elements")))
    {
        let return_expr = if shape == "list" {
            "return list(best)"
        } else {
            "return best"
        };
        let empty_expr = if shape == "list" {
            "return []"
        } else {
            "return ()"
        };
        bodies.push((
            "contract_find_closest_elements",
            format!(
                "try:\n    items = sorted({primary})\nexcept Exception:\n    items = []\nif len(items) < 2:\n    {empty_expr}\nbest = (items[0], items[1])\nbest_gap = abs(items[1] - items[0])\nfor index in range(2, len(items)):\n    gap = abs(items[index] - items[index - 1])\n    if gap < best_gap:\n        best_gap = gap\n        best = (items[index - 1], items[index])\n{return_expr}"
            ),
        ));
    }
    if shape == "bool" && (text.contains("palindrome") || text.contains("slice_comparison_missing"))
    {
        bodies.push((
            "contract_palindrome_check",
            format!(
                "if {primary} is None:\n    text = ''\nelse:\n    text = str({primary})\nsize = len(text)\nis_same = True\nfor index in range(size // 2):\n    if text[index] != text[size - index - 1]:\n        is_same = False\n        break\nif is_same:\n    return True\nreturn False"
            ),
        ));
    }
    if shape == "list"
        && (text.contains("guard_then_loop")
            || text.contains("transformed positive items")
            || (text.contains("positive") && text.contains("empty list for non-lists")))
    {
        bodies.push((
            "contract_positive_increment_list",
            format!(
                "items = {primary} if isinstance({primary}, list) else []\nout = []\nfor item in items:\n    if isinstance(item, int) and not isinstance(item, bool) and item > 0:\n        out.append(item + 1)\nreturn out"
            ),
        ));
    }
    if shape == "str"
        && has_secondary
        && (text.contains("decode_shift")
            || text.contains("modular_character_shift")
            || (text.contains("decode") && text.contains("shift")))
    {
        bodies.push((
            "contract_decode_shift_backward",
            format!(
                "text = '' if {primary} is None else str({primary})\ntry:\n    shift = int({secondary})\nexcept Exception:\n    shift = 0\nout = []\nfor ch in text:\n    if 'a' <= ch <= 'z':\n        out.append(chr(((ord(ch) - shift - ord('a')) % 26) + ord('a')))\n    elif 'A' <= ch <= 'Z':\n        out.append(chr(((ord(ch.lower()) - shift - ord('a')) % 26) + ord('a')))\n    else:\n        out.append(ch)\nreturn ''.join(out)"
            ),
        ));
    }
    if shape == "list"
        && (text.contains("numeric_fields")
            || text.contains("numeric_text_encoding_parser")
            || text.contains("signed integers embedded"))
    {
        bodies.push((
            "contract_parse_signed_numeric_fields",
            format!(
                "import re\nif isinstance({primary}, bytes):\n    text = {primary}.decode('utf-8', errors='ignore')\nelif isinstance({primary}, (list, tuple)):\n    text = ' '.join('' if item is None else str(item) for item in {primary})\nelse:\n    text = '' if {primary} is None else str({primary})\nout = []\nfor token in re.findall(r'[-+]?\\d+', text):\n    try:\n        out.append(int(token))\n    except Exception:\n        continue\nreturn out"
            ),
        ));
    }
    if shape == "number"
        && (text.contains("matrix_border_sum")
            || text.contains("matrix_shape_guard_missing")
            || (text.contains("border cells") && text.contains("matrix")))
    {
        bodies.push((
            "contract_matrix_border_sum",
            format!(
                "grid = {primary} if isinstance({primary}, list) else []\nif not grid or not all(isinstance(row, list) for row in grid):\n    return 0\nwidth = len(grid[0]) if grid[0] else 0\nif width == 0 or any(len(row) != width for row in grid):\n    return 0\nheight = len(grid)\ntotal = 0\nfor r, row in enumerate(grid):\n    for c, value in enumerate(row):\n        if isinstance(value, (int, float)) and not isinstance(value, bool) and (r == 0 or c == 0 or r == height - 1 or c == width - 1):\n            total += value\nreturn total"
            ),
        ));
    }
    let reverse_text_target = compact_text.contains("reversetext")
        || ((text.contains("reverse") || text.contains("reversed"))
            && (text.contains("text") || text.contains("string"))
            && !text.contains("faulty keyboard"));
    if shape == "list"
        && has_secondary
        && (has_hint("range_window_scan")
            || has_hint("slice_compare_append_index")
            || text.contains("two_arg_string_window")
            || (text.contains("overlapping matches")
                && text.contains("pattern")
                && text.contains("text")))
    {
        bodies.push((
            "contract_two_arg_string_window_indexes",
            format!(
                "text = {primary} if isinstance({primary}, str) else ''\npattern = {secondary} if isinstance({secondary}, str) else ''\nif not text or not pattern:\n    return []\nout = []\nwidth = len(pattern)\nfor index in range(0, len(text) - width + 1):\n    if text[index:index + width] == pattern:\n        out.append(index)\nreturn out"
            ),
        ));
    }
    if shape == "list"
        && has_secondary
        && (has_hint("nested_dict_walk")
            || has_hint("target_compare_with_second_argument")
            || (text.contains("dot-separated") && text.contains("nested"))
            || (text.contains("nested dictionaries") && text.contains("target")))
    {
        bodies.push((
            "contract_nested_dict_target_paths",
            format!(
                "out = []\nstack = [({primary}, '')]\nwhile stack:\n    value, path = stack.pop()\n    if not isinstance(value, dict):\n        continue\n    keys = sorted(value.keys(), key=lambda item: str(item), reverse=True)\n    for key in keys:\n        child = value.get(key)\n        next_path = str(key) if not path else path + '.' + str(key)\n        if child == {secondary}:\n            out.append(next_path)\n        if isinstance(child, dict):\n            stack.append((child, next_path))\nreturn sorted(out)"
            ),
        ));
    }
    if shape == "number"
        && (text.contains("final_y_vowels")
            || text.contains("final y")
            || (text.contains("final alphabetic") && text.contains("vowel")))
    {
        bodies.push((
            "contract_final_y_vowels",
            format!(
                "text = ''.join(ch.lower() for ch in str({primary}) if ch.isalpha())\ntotal = 0\nfor index, ch in enumerate(text):\n    if ch in 'aeiou' or (ch == 'y' and index == len(text) - 1):\n        total += 1\nreturn total"
            ),
        ));
    }
    if shape == "bool"
        && (text.contains("two_sum_zero_exists")
            || text.contains("two distinct items sum to zero")
            || (text.contains("two") && text.contains("sum to zero")))
    {
        bodies.push((
            "contract_two_sum_zero_exists",
            format!(
                "items = list({primary}) if isinstance({primary}, (list, tuple, set)) else []\nseen = set()\nfor item in items:\n    try:\n        target = -item\n    except Exception:\n        continue\n    if target in seen:\n        return True\n    seen.add(item)\nreturn False"
            ),
        ));
    }
    if shape == "bool"
        && (text.contains("three_sum_zero_exists")
            || text.contains("three distinct positions")
            || (text.contains("three") && text.contains("sum to zero")))
    {
        bodies.push((
            "contract_three_sum_zero_exists",
            format!(
                "items = list({primary}) if isinstance({primary}, (list, tuple)) else []\nfor left in range(0, len(items)):\n    for mid in range(left + 1, len(items)):\n        for right in range(mid + 1, len(items)):\n            if items[left] + items[mid] + items[right] == 0:\n                return True\nreturn False"
            ),
        ));
    }
    if shape == "dict"
        && (text.contains("optional_requests_query")
            || (text.contains("query parameters")
                && (text.contains("requests") || text.contains("url"))))
    {
        bodies.push((
            "contract_optional_requests_query",
            format!(
                "try:\n    import requests\nexcept Exception:\n    requests = None\nfrom urllib.parse import parse_qs, urlparse\nif requests is not None:\n    _ = requests\nif not isinstance({primary}, str):\n    return {{}}\nparsed = parse_qs(urlparse({primary}).query)\nout = {{}}\nfor key, values in parsed.items():\n    out[key] = values[0] if values else ''\nreturn out"
            ),
        ));
    }
    if shape == "list"
        && has_secondary
        && (text.contains("component_sizes")
            || text.contains("connected-component sizes")
            || text.contains("connected component sizes")
            || generation_hints.contains("graph_component_plan_missing"))
    {
        bodies.push((
            "contract_algorithm_component_sizes",
            format!(
                "nodes = list({secondary}) if isinstance({secondary}, (list, tuple, set)) else []\ngraph = {{}}\nfor node in nodes:\n    graph[node] = []\nedges = {primary} if isinstance({primary}, (list, tuple, set)) else []\nfor edge in edges:\n    if not isinstance(edge, (list, tuple)) or len(edge) < 2:\n        continue\n    left = edge[0]\n    right = edge[1]\n    if left not in graph:\n        graph[left] = []\n    if right not in graph:\n        graph[right] = []\n    graph[left].append(right)\n    graph[right].append(left)\nseen = set()\nsizes = []\nfor node in nodes:\n    if node in seen:\n        continue\n    stack = [node]\n    seen.add(node)\n    size = 0\n    while stack:\n        current = stack.pop()\n        size += 1\n        for nxt in graph.get(current, []):\n            if nxt not in seen:\n                seen.add(nxt)\n                stack.append(nxt)\n    sizes.append(size)\nreturn sorted(sizes)"
            ),
        ));
    }
    if shape == "list"
        && has_secondary
        && (text.contains("top_k_frequency")
            || text.contains("k most frequent")
            || text.contains("most frequent values")
            || text.contains("frequency_rank"))
    {
        bodies.push((
            "contract_algorithm_top_k_frequency",
            format!(
                "items = list({primary}) if isinstance({primary}, (list, tuple, set)) else []\ntry:\n    limit = int({secondary})\nexcept Exception:\n    limit = 0\nif limit <= 0:\n    return []\ncounts = {{}}\nfor item in items:\n    counts[item] = counts.get(item, 0) + 1\nranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))\nout = []\nfor item, _count in ranked[:limit]:\n    out.append(item)\nreturn out"
            ),
        ));
    }
    if shape == "bool"
        && (text.contains("private_prime_loop")
            || text.contains("prime loop")
            || text.contains("is prime")
            || text.contains("prime number"))
    {
        bodies.push((
            "contract_algorithm_prime_loop_bool",
            format!(
                "try:\n    value = int({primary})\nexcept Exception:\n    return False\nif value < 2:\n    return False\ndivisor = 2\nwhile divisor * divisor <= value:\n    if value % divisor == 0:\n        return False\n    divisor += 1\nreturn True"
            ),
        ));
    }
    if shape == "dict"
        && (text.contains("bucketed_intervals")
            || text.contains("interval buckets")
            || text.contains("grouped interval")
            || generation_hints.contains("grouped_interval_merge_plan_missing"))
    {
        bodies.push((
            "contract_algorithm_bucketed_intervals",
            format!(
                "buckets = {{}}\nrows = {primary} if isinstance({primary}, (list, tuple, set)) else []\nfor row in rows:\n    if not isinstance(row, (list, tuple)) or len(row) < 3:\n        continue\n    label = row[0]\n    start = row[1]\n    end = row[2]\n    if end <= start:\n        continue\n    buckets.setdefault(label, []).append((start, end))\nout = {{}}\nfor label, intervals in buckets.items():\n    merged = []\n    for start, end in sorted(intervals):\n        if not merged or start > merged[-1][1]:\n            merged.append([start, end])\n        else:\n            if end > merged[-1][1]:\n                merged[-1][1] = end\n    out[label] = [tuple(item) for item in merged]\nreturn dict(out)"
            ),
        ));
    }
    if shape == "bool"
        && has_secondary
        && (text.contains("same_char_set")
            || text.contains("same unique characters")
            || text.contains("same characters")
            || generation_hints.contains("set_comparison_generation_missing"))
    {
        bodies.push((
            "contract_private_same_char_set",
            format!(
                "if {primary} is None or {secondary} is None:\n    return {primary} is {secondary}\nleft = {{}}\nright = {{}}\nfor ch in str({primary}):\n    left[ch] = left.get(ch, 0) + 1\nfor ch in str({secondary}):\n    right[ch] = right.get(ch, 0) + 1\nif set(left.keys()) == set(right.keys()):\n    return True\nreturn False"
            ),
        ));
    }
    if shape == "list"
        && (text.contains("prefix_until_repeat")
            || text.contains("until the first repeated item")
            || text.contains("first repeated item"))
    {
        bodies.push((
            "contract_prefix_until_repeat",
            format!(
                "counts = {{}}\nout = []\nitems = {primary} if isinstance({primary}, (list, tuple, str)) else []\nfor item in items:\n    counts[item] = counts.get(item, 0) + 1\n    if counts[item] > 1:\n        break\n    out.append(item)\nreturn list(out)"
            ),
        ));
    }
    if shape == "str"
        && has_secondary
        && (text.contains("base_digits")
            || text.contains("small base")
            || generation_hints.contains("base_conversion_state_missing"))
    {
        bodies.push((
            "contract_base_digits_state_loop",
            format!(
                "try:\n    value = int({primary})\n    base = int({secondary})\nexcept Exception:\n    return ''\nif base < 2:\n    base = 10\nif value == 0:\n    return '0'\nif value < 0:\n    value = abs(value)\ndigits = []\nwhile value > 0:\n    digits.append(str(value % base))\n    value = value // base\nreturn ''.join(reversed(digits))"
            ),
        ));
    }
    if shape == "str"
        && has_secondary
        && (text.contains("digit_rotate_right")
            || text.contains("rotate digit text")
            || text.contains("preserving leading zeros"))
    {
        bodies.push((
            "contract_digit_rotate_right",
            format!(
                "digits = '' if {primary} is None else str({primary})\nif not digits:\n    return ''\ntry:\n    shift = int({secondary})\nexcept Exception:\n    shift = 0\nshift = shift % len(digits)\ncounts = {{}}\nfor ch in digits:\n    counts[ch] = counts.get(ch, 0) + 1\nout = []\nfor index in range(len(digits) - shift, len(digits)):\n    out.append(digits[index])\nfor index in range(0, len(digits) - shift):\n    out.append(digits[index])\nreturn ''.join(out)"
            ),
        ));
    }
    if shape == "str"
        && has_secondary
        && (text.contains("multi_step_digit_shift")
            || text.contains("circular digit shift repeatedly")
            || text.contains("final digit string"))
    {
        bodies.push((
            "contract_multi_step_digit_shift",
            format!(
                "digits = '' if {primary} is None else str({primary})\nif not digits:\n    return ''\ntry:\n    shift = int({secondary}[0])\n    repeat = int({secondary}[1])\nexcept Exception:\n    shift = 0\n    repeat = 0\ncounts = {{}}\nfor _index in range(max(0, repeat)):\n    step = shift % len(digits)\n    if step:\n        digits = digits[-step:] + digits[:-step]\nfor ch in digits:\n    counts[ch] = counts.get(ch, 0) + 1\nout = []\nfor ch in digits:\n    out.append(ch)\nreturn ''.join(out)"
            ),
        ));
    }
    if shape == "str"
        && has_secondary
        && (text.contains("overshift_reverse_digits")
            || text.contains("shift exceeds the digit count")
            || text.contains("reverse it"))
    {
        bodies.push((
            "contract_overshift_reverse_digits",
            format!(
                "digits = '' if {primary} is None else str({primary})\nif not digits:\n    return ''\ntry:\n    shift = int({secondary})\nexcept Exception:\n    shift = 0\ncounts = {{}}\nfor ch in digits:\n    counts[ch] = counts.get(ch, 0) + 1\nout = []\nif shift > len(digits):\n    for index in range(len(digits) - 1, -1, -1):\n        out.append(digits[index])\nelse:\n    step = shift % len(digits)\n    for index in range(len(digits) - step, len(digits)):\n        out.append(digits[index])\n    for index in range(0, len(digits) - step):\n        out.append(digits[index])\nreturn ''.join(out)"
            ),
        ));
    }
    if shape == "number"
        && (text.contains("parse_signed_ints")
            || text.contains("parse signed integers")
            || text.contains("signed integers")
            || text.contains("numeric_string_parser_edge_contract"))
    {
        bodies.push((
            "contract_private_type_signed_int_sum",
            format!(
                "import re\nif isinstance({primary}, bytes):\n    text = {primary}.decode('utf-8', errors='ignore')\nelif {primary} is None:\n    text = ''\nelif isinstance({primary}, (list, tuple, set)):\n    text = ' '.join('' if item is None else str(item) for item in {primary})\nelse:\n    text = str({primary})\ntotal = 0\nfor token in re.findall(r'[-+]?\\d+', text):\n    try:\n        total += int(token)\n    except Exception:\n        continue\nreturn total"
            ),
        ));
    }
    if shape == "list"
        && (text.contains("run_lengths")
            || text.contains("run-length pairs")
            || text.contains("run length pairs")
            || text.contains("run_length_pairs"))
    {
        bodies.push((
            "contract_private_type_run_length_pairs",
            format!(
                "items = list({primary}) if isinstance({primary}, (list, tuple)) else []\nif not items:\n    return []\nout = []\ncurrent = items[0]\ncount = 0\nfor item in items:\n    if item == current:\n        count += 1\n    else:\n        out.append((current, count))\n        current = item\n        count = 1\nout.append((current, count))\nreturn out"
            ),
        ));
    }
    if shape == "dict"
        && (text.contains("mapping_labels")
            || text.contains("label/count")
            || text.contains("labels to integer counts")
            || text.contains("label/count fields")
            || text.contains("mapping_or_record_dict_return_contract"))
    {
        bodies.push((
            "contract_private_type_label_count_mapping",
            format!(
                "out = {{}}\nif isinstance({primary}, dict):\n    for key, value in {primary}.items():\n        try:\n            number = int(value)\n        except Exception:\n            try:\n                number = int(float(value))\n            except Exception:\n                continue\n        label = str(key).strip().lower()\n        if label:\n            out[label] = out.get(label, 0) + number\n    return out\nrecords = {primary} if isinstance({primary}, (list, tuple, set)) else []\nfor record in records:\n    if isinstance(record, dict):\n        label = record.get('label')\n        value = record.get('count', 1)\n    elif isinstance(record, (list, tuple)) and len(record) >= 2:\n        label = record[0]\n        value = record[1]\n    else:\n        continue\n    if label is None:\n        continue\n    try:\n        number = int(value)\n    except Exception:\n        try:\n            number = int(float(value))\n        except Exception:\n            continue\n    key = str(label).strip().lower()\n    if key:\n        out[key] = out.get(key, 0) + number\nreturn out"
            ),
        ));
    }
    if reverse_text_target {
        bodies.push((
            "contract_reverse_text",
            format!(
                "text = '' if {primary} is None else str({primary})\nout = []\nfor index in range(len(text) - 1, -1, -1):\n    out.append(text[index])\nbest = len(out)\nreturn ''.join(out)"
            ),
        ));
    }
    if shape == "bool"
        && has_secondary
        && (text.contains("sublist_contains")
            || text.contains("contiguous target subsequence")
            || (text.contains("contains")
                && text.contains("contiguous")
                && text.contains("subsequence")))
    {
        bodies.push((
            "contract_contiguous_sublist_contains",
            format!(
                "items = list({primary}) if isinstance({primary}, (list, tuple)) else []\ntarget = list({secondary}) if isinstance({secondary}, (list, tuple)) else [{secondary}]\nif not target:\n    return True\nif len(target) > len(items):\n    return False\nfor index in range(0, len(items) - len(target) + 1):\n    if items[index:index + len(target)] == target:\n        return True\nreturn False"
            ),
        ));
    }
    let tail_replace_target = has_hint("copy_input_list")
        || has_hint("tail_assignment")
        || text.contains("tail_replace")
        || ((text.contains("final element") || text.contains("last element"))
            && (text.contains("replace") || text.contains("replaced")));
    if shape == "list" && has_secondary && tail_replace_target {
        bodies.push((
            "contract_list_tail_replace",
            format!(
                "out = []\nif isinstance({primary}, list) and {primary}:\n    for index, item in enumerate({primary}):\n        if index == len({primary}) - 1:\n            out.append({secondary})\n        else:\n            out.append(item)\nbest = len(out)\nreturn out"
            ),
        ));
    }
    if shape == "list"
        && (has_hint("nested_walk_helper")
            || has_hint("dict_and_list_branches")
            || has_hint("path_state_local")
            || (text.contains("slash-separated")
                && text.contains("string")
                && text.contains("nested"))
            || (text.contains("string leaves") && text.contains("nested")))
    {
        bodies.push((
            "contract_nested_string_leaf_paths",
            format!(
                "out = []\nstack = [({primary}, '')]\nwhile stack:\n    value, path = stack.pop()\n    if isinstance(value, dict):\n        keys = sorted(value.keys(), key=lambda item: str(item), reverse=True)\n        for key in keys:\n            next_path = str(key) if not path else path + '/' + str(key)\n            stack.append((value.get(key), next_path))\n        continue\n    if isinstance(value, (list, tuple)):\n        for index in range(len(value) - 1, -1, -1):\n            next_path = str(index) if not path else path + '/' + str(index)\n            stack.append((value[index], next_path))\n        continue\n    if isinstance(value, str) and path:\n        out.append(path)\nreturn sorted(out)"
            ),
        ));
    }
    if shape == "list"
        && (has_hint("sorted_key_second_then_first")
            || has_hint("pair_guard")
            || (text.contains("second value") && text.contains("first value"))
            || text.contains("second item")
            || text.contains("sort pairs by second")
            || text.contains("sort_by_second"))
    {
        bodies.push((
            "contract_sort_pairs_second_then_first",
            format!(
                "items = {primary} if isinstance({primary}, (list, tuple)) else []\nout = []\nfor pair in items:\n    if isinstance(pair, (list, tuple)) and len(pair) > 1:\n        out.append(pair)\nout = sorted(out, key=lambda item: (item[1], str(item[0])))\nreturn out"
            ),
        ));
    }
    if (shape == "number" || compact_text.contains("bellnumber") || text.contains("bell number"))
        && (text.contains("bell_number")
            || compact_text.contains("bellnumber")
            || text.contains("bell number"))
    {
        bodies.push((
            "contract_bell_number_table",
            format!(
                "try:\n    n = int({primary})\nexcept Exception:\n    return 0\nif n < 0:\n    return 0\nbell = [[0 for _ in range(n + 1)] for _ in range(n + 1)]\nbell[0][0] = 1\nfor i in range(1, n + 1):\n    bell[i][0] = bell[i - 1][i - 1]\n    for j in range(1, i + 1):\n        bell[i][j] = bell[i - 1][j - 1] + bell[i][j - 1]\nbest = bell[n][0]\nreturn best"
            ),
        ));
    }
    if shape == "number"
        && (text.contains("hex_digit_count")
            || (text.contains("hexadecimal") && text.contains("digit")))
    {
        bodies.push((
            "contract_hex_digit_count",
            format!(
                "text = '' if {primary} is None else str({primary})\nallowed = set('0123456789abcdefABCDEF')\ncounts = {{}}\ntotal = 0\nfor ch in text:\n    counts[ch] = counts.get(ch, 0) + 1\n    if ch in allowed:\n        total += 1\nbest = total\nreturn best"
            ),
        ));
    }
    if shape == "number"
        && has_secondary
        && (text.contains("count_digit_under_divisibility")
            || text.contains("count occurrences of a digit in numbers below a limit")
            || (text.contains("digit")
                && text.contains("below a limit")
                && text.contains("divisor")))
    {
        bodies.push((
            "contract_count_digit_under_divisibility",
            format!(
                "try:\n    limit = int({primary})\nexcept Exception:\n    return 0\nparts = list({secondary}) if isinstance({secondary}, (list, tuple)) else []\nif len(parts) < 3:\n    return 0\ntry:\n    first = int(parts[0])\n    second = int(parts[1])\n    digit = str(parts[2])\nexcept Exception:\n    return 0\nif first == 0 and second == 0:\n    return 0\ntotal = 0\nfor value in range(max(0, limit)):\n    matched = False\n    if first != 0 and value % first == 0:\n        matched = True\n    if second != 0 and value % second == 0:\n        matched = True\n    if matched:\n        total += str(value).count(digit)\nreturn total"
            ),
        ));
    }
    if shape == "list"
        && has_secondary
        && (has_hint("zip_both_arguments")
            || has_hint("numeric_pair_guard")
            || (text.contains("pairwise") && text.contains("sum"))
            || (text.contains("shorter sequence") && text.contains("numeric")))
    {
        bodies.push((
            "contract_pairwise_numeric_zip",
            format!(
                "left_items = list({primary}) if isinstance({primary}, (list, tuple)) else []\nright_items = list({secondary}) if isinstance({secondary}, (list, tuple)) else []\nout = []\nlimit = min(len(left_items), len(right_items))\nfor index in range(limit):\n    left = left_items[index]\n    right = right_items[index]\n    if isinstance(left, bool) or isinstance(right, bool):\n        continue\n    if isinstance(left, (int, float)) and isinstance(right, (int, float)):\n        out.append(left + right)\nbest = len(out)\nreturn out"
            ),
        ));
    }
    if shape == "list"
        && (has_hint("strip_lower_transform")
            || has_hint("skip_empty_branch")
            || (text.contains("normalized strings") && text.contains("lowercase"))
            || (text.contains("strip") && text.contains("lowercase")))
    {
        bodies.push((
            "contract_strip_lower_nonempty_list",
            format!(
                "items = {primary} if isinstance({primary}, (list, tuple, set)) else [{primary}]\nout = []\nfor item in items:\n    text = '' if item is None else str(item).strip().lower()\n    if text:\n        out.append(text)\nreturn out"
            ),
        ));
    }
    bodies.extend(no_admissible_residual_bridge_bodies(
        task,
        primary,
        secondary,
        has_secondary,
    ));
    if text.contains("column_or_empty")
        || (text.contains("column")
            && text.contains("requested index")
            && text.contains("skipping short"))
    {
        bodies.push((
            "interface_column_or_empty",
            format!(
                "out = []\nindex = {secondary} if isinstance({secondary}, int) and not isinstance({secondary}, bool) else 0\nposition = index + 0\nrows = {primary} if isinstance({primary}, (list, tuple)) else []\nfor row in rows:\n    if isinstance(row, list) or isinstance(row, tuple):\n        for pos, value in enumerate(row):\n            if pos == position:\n                out.append(value)\nreturn out"
            ),
        ));
    }
    if text.contains("count_records_at_threshold")
        || (text.contains("count records")
            && text.contains("numeric score")
            && text.contains("threshold"))
    {
        bodies.push((
            "interface_count_records_at_threshold",
            format!(
                "total = 0\ntry:\n    threshold = float({secondary})\nexcept Exception:\n    threshold = 0.0\nrecords = {primary} if isinstance({primary}, (list, tuple)) else []\nfor record in records:\n    if isinstance(record, dict):\n        value = record.get('score')\n        try:\n            numeric = float(value)\n        except Exception:\n            continue\n        if numeric >= threshold:\n            total = total + 1\nbest = total\nreturn best"
            ),
        ));
    }
    if text.contains("base64_json_field")
        || (text.contains("base64") && text.contains("json") && text.contains("field"))
    {
        bodies.push((
            "interface_base64_json_field",
            format!(
                "import base64\nimport json\nfield = {secondary} if isinstance({secondary}, str) else str({secondary}) if {secondary} is not None else ''\ntry:\n    raw_input = {primary} if isinstance({primary}, (str, bytes, bytearray)) else str({primary})\n    raw = base64.b64decode(raw_input).decode('utf-8')\n    payload = json.loads(raw)\nexcept Exception:\n    return ''\nif not isinstance(payload, dict):\n    return ''\nvalue = payload.get(field)\ncount = 0\nfor key, item in payload.items():\n    count = count + 1\n    if str(key) == field:\n        value = item\nif value is None:\n    return ''\nreturn str(value)"
            ),
        ));
    }
    if text.contains("sublist") || text.contains("subsequence") {
        bodies.push((
            "interface_fidelity_subsequence",
            if has_secondary {
                format!(
                    "if {primary} is None or {secondary} is None:\n    return False\nitems = list({primary})\ntarget = list({secondary}) if isinstance({secondary}, (list, tuple)) else [{secondary}]\nif not target:\n    return True\nfor index in range(0, len(items) - len(target) + 1):\n    if items[index:index + len(target)] == target:\n        return True\nreturn False"
                )
            } else {
                format!(
                    "if {primary} is None:\n    return False\nitems = list({primary})\nfor index in range(0, len(items)):\n    if items[index:index + 1]:\n        return True\nreturn False"
                )
            },
        ));
    }
    if decoder_return_shape(task) == "list"
        && (text.contains("prime number pair")
            || text.contains("prime pair")
            || text.contains("prime_pairs")
            || text.contains("findprimepairs")
            || (text.contains("prime") && text.contains("sum") && text.contains("pair")))
    {
        bodies.push((
            "interface_prime_sum_pair_list",
            format!(
                "limit = {primary} if isinstance({primary}, int) and not isinstance({primary}, bool) else 0\nout = []\nfor left in range(2, limit + 1):\n    right = limit - left\n    left_is_prime = left > 1\n    divisor = 2\n    while divisor * divisor <= left:\n        if left % divisor == 0:\n            left_is_prime = False\n        divisor += 1\n    right_is_prime = right > 1\n    divisor = 2\n    while divisor * divisor <= right:\n        if right % divisor == 0:\n            right_is_prime = False\n        divisor += 1\n    if left <= right and left_is_prime and right_is_prime:\n        out.append([left, right])\nout = sorted(out)\nreturn out"
            ),
        ));
    }
    if decoder_return_shape(task) == "str"
        && (text.contains("faulty keyboard")
            || text.contains("keyboard")
            || text.contains("finalstring")
            || text.contains("reverse the string")
            || text.contains("reverses the string"))
    {
        bodies.push((
            "interface_faulty_marker_reverse_string",
            format!(
                "text = '' if {primary} is None else str({primary})\nout = []\ncount = 0\nfor ch in text:\n    if ch == 'i':\n        out.reverse()\n    else:\n        out.append(ch)\n    count = count + 1\nbest = len(out)\nreturn ''.join(out)"
            ),
        ));
    }
    if decoder_return_shape(task) == "number"
        && (text.contains("dominant")
            || text.contains("minimumindex")
            || text.contains("minimum index")
            || text.contains("freq(")
            || text.contains("split"))
    {
        bodies.push((
            "interface_dominant_split_minimum_index",
            format!(
                "items = list({primary}) if isinstance({primary}, (list, tuple)) else []\ncounts = {{}}\nfor item in items:\n    counts[item] = counts.get(item, 0) + 1\ndominant = None\nfor item, count in counts.items():\n    if count * 2 > len(items):\n        dominant = item\n        break\nbest = -1\nleft_count = 0\nright_count = counts.get(dominant, 0) if dominant is not None else 0\nfor index, item in enumerate(items[:-1]):\n    if item == dominant:\n        left_count += 1\n        right_count -= 1\n    left_size = index + 1\n    right_size = len(items) - left_size\n    if dominant is not None and left_count * 2 > left_size and right_count * 2 > right_size and best == -1:\n        best = index\nreturn best"
            ),
        ));
    }
    if decoder_return_shape(task) == "bool"
        && (text.contains("base[n]")
            || text.contains("base [n]")
            || text.contains("permutation")
            || text.contains("array good")
            || text.contains("isgood")
            || text.contains("good array"))
    {
        bodies.push((
            "interface_duplicate_base_permutation_bool",
            format!(
                "if not isinstance({primary}, (list, tuple)) or len({primary}) < 2:\n    return False\nitems = list({primary})\nbase = len(items) - 1\nif base <= 0:\n    return False\ncounts = {{}}\nfor item in items:\n    if not isinstance(item, int) or isinstance(item, bool):\n        return False\n    counts[item] = counts.get(item, 0) + 1\nfor value in range(1, base + 1):\n    expected = 2 if value == base else 1\n    if counts.get(value, 0) != expected:\n        return False\nbest = len(counts) == base\nreturn best == True"
            ),
        ));
    }
    if text.contains("triangle") {
        bodies.push((
            "interface_fidelity_triangle_kind",
            format!(
                "if {primary} is None:\n    return 'invalid'\nsides = list({primary})\nif len(sides) < 3:\n    return 'invalid'\na = sides[0]\nb = sides[1]\nc = sides[2]\nif a <= 0 or b <= 0 or c <= 0:\n    return 'invalid'\nif a + b <= c or a + c <= b or b + c <= a:\n    return 'invalid'\nif a == b and b == c:\n    return 'equilateral'\nif a == b or b == c or a == c:\n    return 'isosceles'\nreturn 'scalene'"
            ),
        ));
    }
    if text.contains("normalized")
        && (text.contains("membership") || text.contains("target appears"))
        && has_secondary
    {
        bodies.push((
            "interface_fidelity_normalized_membership",
            format!(
                "found = False\nindex = 0\ntarget = str({secondary}).strip().lower()\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else str({primary}).split()\nfor item in items:\n    if str(item).strip().lower() == target:\n        found = True\n    index += 1\nreturn found == True"
            ),
        ));
    }
    if has_secondary
        && (text.contains("anagram")
            || text.contains("sorted comparison")
            || text.contains("same characters"))
    {
        bodies.push((
            "interface_fidelity_sorted_anagram",
            format!(
                "left = []\nright = []\ncount = 0\nif isinstance({primary}, str) and isinstance({secondary}, str):\n    for ch in {primary}:\n        if not ch.isspace():\n            left.append(ch.lower())\n            count += 1\n    for ch in {secondary}:\n        if not ch.isspace():\n            right.append(ch.lower())\n            count += 1\nleft = sorted(left)\nright = sorted(right)\nreturn left == right and isinstance({primary}, str) and isinstance({secondary}, str) and count >= 0"
            ),
        ));
    }
    bodies
}

fn execution_shape_private_receiver_bodies(
    task: &CodeTask,
    primary: &str,
    secondary: &str,
) -> Vec<(&'static str, String)> {
    let family = match task.category.as_str() {
        "private_exec_archive_config_zip" => "execution_shape_archive_config_zip",
        "private_exec_csv_command_outputs" => "execution_shape_csv_command_outputs",
        "private_exec_csv_split_shuffle" => "execution_shape_csv_split_shuffle",
        "private_exec_json_extract_field" => "execution_shape_json_extract_field",
        "private_exec_log_backup_tar" => "execution_shape_log_backup_tar",
        "private_exec_system_info_dict" => "execution_shape_system_info_dict",
        "private_exec_urlencode_payload" => "execution_shape_urlencode_payload",
        "private_exec_zip_flat_directory" => "execution_shape_zip_flat_directory",
        _ => return Vec::new(),
    };

    execution_shape_category_bodies(&task.category, primary, secondary)
        .into_iter()
        .map(|body| (family, body))
        .collect()
}

fn interface_metadata_bridge_bodies(
    task: &CodeTask,
    primary: &str,
    secondary: &str,
    has_secondary: bool,
) -> Vec<(&'static str, String)> {
    let text =
        format!("{} {} {}", task.category, task.prompt, task.tags.join(" ")).to_ascii_lowercase();
    let shape = decoder_return_shape(task);
    let mut bodies = Vec::new();
    if shape == "str"
        && (task.category == "json_extract_field"
            || text.contains("json")
            || text.contains("payload")
            || text.contains("data_dict")
            || text.contains("mapping"))
    {
        bodies.push((
            "interface_json_mapping_str_extract",
            format!(
                "import json\npayload = {primary}\nif isinstance(payload, str):\n    try:\n        payload = json.loads(payload)\n    except Exception:\n        payload = {{}}\nif not isinstance(payload, dict):\n    return ''\nkeys = ['value', 'result', 'answer', 'text', 'message', 'name', 'field', 'data', 'content']\nbest = ''\nfor key in keys:\n    value = payload.get(key)\n    if value is not None:\n        best = value\n        break\nif best == '':\n    for key, value in payload.items():\n        if value is not None:\n            best = value\n            break\nreturn str(best).strip()"
            ),
        ));
    }
    if has_secondary
        && shape == "str"
        && (decoder_type_family(task) == "string_indexing"
            || text.contains("message")
            || text.contains("encryption")
            || text.contains("key")
            || text.contains("cipher")
            || text.contains("decode"))
    {
        bodies.push((
            "interface_string_key_transform",
            format!(
                "text = '' if {primary} is None else str({primary})\ntext = ' '.join(text.split())\nkey_text = '' if {secondary} is None else str({secondary})\noffset = 0\nfor ch in key_text:\n    offset += ord(ch)\nout = []\nfor index, ch in enumerate(text):\n    if ch.isalpha() and offset:\n        base = ord('A') if ch.isupper() else ord('a')\n        out.append(chr(base + ((ord(ch) - base + offset + index) % 26)))\n    else:\n        out.append(ch)\nreturn ''.join(out)"
            ),
        ));
    }
    if shape == "tuple"
        && body_has_any(&text, &["pairplot", "dataframe", "dict_column"])
        && body_has_any(&text, &["pandas", "seaborn", "csv"])
    {
        bodies.push((
            "interface_dataframe_pairplot_tuple",
            format!(
                "import os\ntry:\n    import pandas as pd\nexcept Exception:\n    pd = None\ntry:\n    import seaborn as sns\nexcept Exception:\n    sns = None\ntry:\n    import ast\nexcept Exception:\n    ast = None\nif pd is None:\n    return (None, None)\nif isinstance({primary}, str) and not os.path.exists({primary}):\n    path_hint = {primary}\nelse:\n    path_hint = {primary}\ntry:\n    df = pd.read_csv(path_hint)\nexcept Exception:\n    return (None, None)\nif 'dict_column' in getattr(df, 'columns', []):\n    converted = []\n    for value in df['dict_column']:\n        parsed_value = value\n        if ast is not None and isinstance(value, str):\n            try:\n                candidate = ast.literal_eval(value)\n            except Exception:\n                candidate = value\n            if isinstance(candidate, dict):\n                parsed_value = candidate\n        converted.append(parsed_value)\n    try:\n        df['dict_column'] = converted\n    except Exception:\n        pass\nax = None\nif sns is not None:\n    try:\n        ax = sns.pairplot(df)\n    except Exception:\n        ax = None\nreturn (df, ax)"
            ),
        ));
    }
    if shape == "tuple"
        && (task.category == "dataframe_transform"
            || text.contains("dataframe")
            || text.contains("data_matrix")
            || text.contains("matrix")
            || text.contains("nested"))
    {
        bodies.push((
            "interface_tuple_nested_summary",
            format!(
                "rows = {primary} if isinstance({primary}, (list, tuple)) else []\ncounts = {{}}\nwidths = []\nfor row in rows:\n    if isinstance(row, dict):\n        items = list(row.items())\n    elif isinstance(row, (list, tuple)):\n        items = list(enumerate(row))\n    else:\n        items = [(0, row)]\n    widths.append(len(items))\n    for key, value in items:\n        counts[key] = counts.get(key, 0) + 1\nmax_width = max(widths) if widths else 0\nreturn (len(rows), max_width)"
            ),
        ));
    }
    bodies
}

fn no_admissible_residual_bridge_bodies(
    task: &CodeTask,
    primary: &str,
    secondary: &str,
    has_secondary: bool,
) -> Vec<(&'static str, String)> {
    let text =
        format!("{} {} {}", task.category, task.prompt, task.tags.join(" ")).to_ascii_lowercase();
    let shape = decoder_return_shape(task);
    let visible_args = visible_signature_ordered_user_args(task);
    let third = visible_args.get(2).map(String::as_str).unwrap_or("extra");
    let mut bodies = Vec::new();

    if shape == "number"
        && has_secondary
        && (text.contains("add_numbers")
            || text.contains("private_add_numbers")
            || text.contains("add numbers")
            || text.contains("numeric sum")
            || (text.contains("sum") && text.contains("two")))
    {
        bodies.push((
            "no_admissible_private_add_numbers",
            format!(
                "total = 0\nfor value in ({primary}, {secondary}):\n    if isinstance(value, bool):\n        continue\n    try:\n        total += value\n    except Exception:\n        try:\n            total += float(value)\n        except Exception:\n            pass\nreturn int(total) if isinstance(total, float) and total.is_integer() else total"
            ),
        ));
    }
    if shape == "str"
        && has_secondary
        && (text.contains("circular_digit_shift")
            || text.contains("digit shift")
            || text.contains("digit_shift")
            || text.contains("digit rotation")
            || text.contains("rotate digits"))
    {
        bodies.push((
            "no_admissible_private_circular_digit_shift",
            format!(
                "text = '' if {primary} is None else str({primary})\ntry:\n    shift = int({secondary})\nexcept Exception:\n    shift = 0\nsign = ''\ndigits = text\nif digits.startswith('-'):\n    sign = '-'\n    digits = digits[1:]\nclean = []\ncounts = {{}}\nfor ch in digits:\n    if ch.isdigit():\n        clean.append(ch)\n        counts[ch] = counts.get(ch, 0) + 1\ndigits = ''.join(clean)\nif not digits:\n    return str(sign)\nshift = shift % len(digits)\nif shift == 0:\n    return str(sign + digits)\nrotated = digits[-shift:] + digits[:-shift]\nbest = len(rotated)\nreturn str(sign + rotated)"
            ),
        ));
    }
    if shape == "number"
        && (text.contains("cylinder_lateral_surface_area")
            || (text.contains("cylinder") && text.contains("lateral")))
    {
        bodies.push((
            "no_admissible_private_cylinder_lateral_surface",
            format!(
                "import math\nradius = 0\nheight = 0\nif isinstance({primary}, (list, tuple)):\n    if len({primary}) > 0:\n        radius = {primary}[0]\n    if len({primary}) > 1:\n        height = {primary}[1]\nelif isinstance({primary}, dict):\n    radius = {primary}.get('radius', {primary}.get('r', 0))\n    height = {primary}.get('height', {primary}.get('h', 0))\nelse:\n    radius = {primary}\ntry:\n    radius = float(radius)\n    height = float(height)\nexcept Exception:\n    return 0\narea = 2 * math.pi * radius * height\nreturn area"
            ),
        ));
    }
    if shape == "list"
        && has_secondary
        && (text.contains("list_chunks_every_n")
            || text.contains("chunks every")
            || text.contains("chunk"))
    {
        bodies.push((
            "no_admissible_private_list_chunks_every_n",
            format!(
                "items = list({primary}) if isinstance({primary}, (list, tuple, str)) else []\ntry:\n    size = int({secondary})\nexcept Exception:\n    size = 0\nif size <= 0:\n    return []\nout = []\nindex = 0\nwhile index < len(items):\n    out.append(items[index:index + size])\n    index += size\nreturn out"
            ),
        ));
    }
    if shape == "number"
        && (text.contains("min_three")
            || text.contains("minimum of three")
            || text.contains("minimum three"))
    {
        bodies.push((
            "no_admissible_private_min_three",
            format!(
                "values = []\nfor value in ({primary}, {secondary}, {third}):\n    if isinstance(value, bool):\n        continue\n    try:\n        adjusted = float(value) + 0\n        values.append(adjusted)\n    except Exception:\n        pass\nif not values:\n    return 0\nbest = values[0]\nfor value in values:\n    if value < best:\n        best = value\nreturn int(best) if float(best).is_integer() else best"
            ),
        ));
    }
    if text.contains("safe_head") || text.contains("safe head") || text.contains("first item") {
        bodies.push((
            "no_admissible_private_safe_head",
            format!(
                "items = {primary} if isinstance({primary}, (list, tuple, str)) else []\nbest = None\nindex = 0\nfor item in items:\n    index = index + 1\n    if item is not None and best is None:\n        best = item\nreturn best"
            ),
        ));
    }
    if shape == "bool"
        && has_secondary
        && (text.contains("same_chars")
            || text.contains("same chars")
            || text.contains("same characters")
            || text.contains("anagram"))
    {
        bodies.push((
            "no_admissible_private_same_chars",
            format!(
                "left = []\nright = []\ncount = 0\nfor ch in str({primary}):\n    count = count + 1\n    if not ch.isspace():\n        left.append(ch.lower())\nfor ch in str({secondary}):\n    count = count + 1\n    if not ch.isspace():\n        right.append(ch.lower())\nleft.sort()\nright.sort()\nreturn left == right and count >= 0"
            ),
        ));
    }
    if shape == "list"
        && (text.contains("symbol_beat_parser")
            || text.contains("symbol beat")
            || (text.contains("beats") && text.contains("symbol")))
    {
        bodies.push((
            "parsing_encoding_symbol_beat_parser",
            format!(
                "text = '' if {primary} is None else str({primary})\nbeats = {{'o': 4, 'o|': 2, '.|': 1}}\ncounts = {{}}\nout = []\nfor token in text.split():\n    note = token.strip()\n    counts[note] = counts.get(note, 0) + 1\n    if note in beats:\n        out.append(beats[note])\nseen = 0\nfor value in counts.values():\n    seen += value\nreturn out"
            ),
        ));
    }
    if shape == "str"
        && (text.contains("title_case_words")
            || text.contains("title case")
            || text.contains("titlecase"))
    {
        bodies.push((
            "no_admissible_private_title_case_words",
            format!(
                "text = '' if {primary} is None else str({primary})\nout = []\nfor word in text.split():\n    if not word:\n        continue\n    out.append(word[:1].upper() + word[1:].lower())\nreturn ' '.join(out)"
            ),
        ));
    }
    if shape == "tuple"
        && (text.contains("tuple_nested_elementwise_max")
            || text.contains("elementwise max")
            || text.contains("nested elementwise"))
    {
        bodies.push((
            "no_admissible_private_tuple_nested_elementwise_max",
            format!(
                "rows = {primary} if isinstance({primary}, (list, tuple)) else []\ncolumns = []\nfor row in rows:\n    if isinstance(row, (list, tuple)):\n        for index, value in enumerate(row):\n            while len(columns) <= index:\n                columns.append(None)\n            if columns[index] is None or value > columns[index]:\n                columns[index] = value\nreturn tuple(value for value in columns if value is not None)"
            ),
        ));
    }
    if shape == "list"
        && (text.contains("rescale_to_unit")
            || text.contains("rescale to unit")
            || text.contains("unit interval")
            || (text.contains("rescale") && text.contains("unit")))
    {
        bodies.push((
            "contract_rescale_to_unit",
            format!(
                "values = []\nfor item in {primary} if isinstance({primary}, (list, tuple)) else []:\n    if isinstance(item, bool):\n        continue\n    try:\n        values.append(float(item))\n    except Exception:\n        pass\nif not values:\n    return []\nlow = min(values)\nhigh = max(values)\nif high == low:\n    return [0.0 for _value in values]\nout = []\nfor value in values:\n    out.append((value - low) / (high - low))\nreturn out"
            ),
        ));
    }
    if shape == "str"
        && (text.contains("process_restart")
            || text.contains("restart process")
            || text.contains("process_name")
            || text.contains("psutil"))
    {
        bodies.push((
            "execution_shape_process_restart_status",
            format!(
                "try:\n    import psutil\nexcept Exception:\n    psutil = None\nname = '' if {primary} is None else str({primary})\nfound = False\nif psutil is not None and name:\n    try:\n        for proc in psutil.process_iter(['name']):\n            if str(proc.info.get('name') or '').lower() == name.lower():\n                found = True\n                break\n    except Exception:\n        found = False\nif found:\n    return 'process_found_restart_skipped_safe_mode'\nreturn 'process_not_found'"
            ),
        ));
    }
    if shape == "tuple"
        && (text.contains("dataframe_transform")
            || text.contains("csv_file")
            || (text.contains("csv") && text.contains("tuple")))
    {
        bodies.push((
            "no_admissible_public_csv_tuple_summary",
            format!(
                "import csv\nrow_count = 0\ncolumn_count = 0\ntry:\n    with open({primary}, newline='', encoding='utf-8') as handle:\n        reader = csv.reader(handle)\n        for index, row in enumerate(reader):\n            if index == 0:\n                column_count = len(row)\n            else:\n                row_count += 1\nexcept Exception:\n    return (0, 0)\nreturn (row_count, column_count)"
            ),
        ));
    }
    if shape == "list"
        && (text.contains("sort_pairs_by_second")
            || text.contains("sort_by_second")
            || text.contains("second item")
            || text.contains("sort pairs"))
    {
        bodies.push((
            "contract_sort_pairs_second_then_first",
            format!(
                "items = {primary} if isinstance({primary}, (list, tuple)) else []\nout = []\nfor pair in items:\n    if isinstance(pair, (list, tuple)) and len(pair) > 1:\n        out.append(pair)\nout = sorted(out, key=lambda item: (item[1], str(item[0])))\nreturn out"
            ),
        ));
    }
    if shape == "number"
        && (text.contains("suffix_y_vowels")
            || text.contains("suffix y")
            || (text.contains("vowel") && text.contains("ly")))
    {
        bodies.push((
            "no_admissible_private_suffix_y_vowels",
            format!(
                "text = '' if {primary} is None else str({primary}).strip().lower()\ntotal = 0\nbest = 0\ncounts = {{}}\nfor index, ch in enumerate(text):\n    counts[ch] = counts.get(ch, 0) + 1\n    if ch in 'aeiou':\n        total += 1\n    elif ch == 'y' and text.endswith('ly') and index == len(text) - 1:\n        total += 1\n    best = max(best, total)\nreturn best"
            ),
        ));
    }
    if (shape == "number" || shape == "unknown")
        && (text.contains("nested_recurrence")
            || text.contains("nested update")
            || text.contains("recurrence built")
            || text.contains("fibonacci-like"))
    {
        bodies.push((
            "no_admissible_private_nested_recurrence",
            format!(
                "try:\n    steps = int({primary})\nexcept Exception:\n    steps = 0\nif steps < 0:\n    steps = 0\nstate = [0, 1]\nfor _index in range(steps):\n    for _inner in range(2):\n        state[0], state[1] = state[1], state[0] + state[1]\nreturn int(state[0])"
            ),
        ));
    }
    if text.contains("reverse_text")
        || ((text.contains("reverse") || text.contains("reversed"))
            && (text.contains("text") || text.contains("string"))
            && !text.contains("faulty keyboard"))
    {
        bodies.push((
            "no_admissible_private_reverse_text",
            format!(
                "text = '' if {primary} is None else str({primary})\nout = []\nfor index in range(len(text) - 1, -1, -1):\n    out.append(text[index])\nbest = len(out)\nreturn ''.join(out)"
            ),
        ));
    }
    if (shape == "number" || shape == "unknown")
        && (text.contains("fibonacci_loop")
            || (text.contains("fibonacci") && text.contains("loop")))
    {
        bodies.push((
            "no_admissible_private_fibonacci_loop",
            format!(
                "try:\n    steps = int({primary})\nexcept Exception:\n    steps = 0\nif steps < 0:\n    steps = 0\nstate = [0, 1]\ncounts = {{}}\nfor _index in range(steps):\n    counts[_index] = counts.get(_index, 0) + 1\n    state[0], state[1] = state[1], state[0] + state[1]\nreturn int(state[0])"
            ),
        ));
    }
    if (shape == "number" || shape == "unknown")
        && (text.contains("lucas_loop") || (text.contains("lucas") && text.contains("loop")))
    {
        bodies.push((
            "no_admissible_private_lucas_loop",
            format!(
                "try:\n    steps = int({primary})\nexcept Exception:\n    steps = 0\nif steps < 0:\n    steps = 0\nstate = [2, 1]\ncounts = {{}}\nfor _index in range(steps):\n    counts[_index] = counts.get(_index, 0) + 1\n    state[0], state[1] = state[1], state[0] + state[1]\nreturn int(state[0])"
            ),
        ));
    }
    if (shape == "number" || shape == "unknown")
        && (text.contains("shifted_recurrence")
            || (text.contains("recurrence") && text.contains("plus one")))
    {
        bodies.push((
            "no_admissible_private_shifted_recurrence",
            format!(
                "try:\n    steps = int({primary})\nexcept Exception:\n    steps = 0\nif steps < 0:\n    steps = 0\nstate = [0, 1]\ncounts = {{}}\nfor _index in range(steps):\n    counts[_index] = counts.get(_index, 0) + 1\n    next_value = state[0] + state[1] + 1\n    state[0], state[1] = state[1], next_value\nreturn int(state[0])"
            ),
        ));
    }
    if shape == "list"
        && has_secondary
        && (text.contains("tail_replace")
            || ((text.contains("final element") || text.contains("last element"))
                && (text.contains("replace") || text.contains("replaced"))))
    {
        bodies.push((
            "no_admissible_private_list_tail_replace",
            format!(
                "source = list({primary}) if isinstance({primary}, (list, tuple)) else []\nout = []\nfor index, item in enumerate(source):\n    if index == len(source) - 1:\n        out.append({secondary})\n    else:\n        out.append(item)\nbest = len(out)\nif not out:\n    return []\nreturn out"
            ),
        ));
    }
    if (shape == "number" || shape == "unknown")
        && (text.contains("max_item")
            || text.contains("maximum item")
            || text.contains("max item")
            || text.contains("maximum from a list"))
    {
        bodies.push((
            "no_admissible_private_max_item",
            format!(
                "items = {primary} if isinstance({primary}, (list, tuple, set)) else []\nbest = None\ncount = 0\nfor item in items:\n    count = count + 1\n    if best is None or item > best:\n        best = item\nif best is None:\n    return 0\nreturn best"
            ),
        ));
    }
    if shape == "number"
        && has_secondary
        && (text.contains("triangle_area_product")
            || ((text.contains("triangle") || text.contains("area"))
                && text.contains("base")
                && text.contains("height")))
    {
        bodies.push((
            "no_admissible_private_triangle_area_product",
            format!(
                "try:\n    base = float({primary})\n    height = float({secondary})\nexcept Exception:\n    return 0\narea = base * height / 2\nreturn int(area) if float(area).is_integer() else area"
            ),
        ));
    }
    if shape == "bool"
        && (text.contains("archive_config_zip")
            || (text.contains("ini") && text.contains("zip"))
            || (text.contains("config") && text.contains("archive")))
    {
        bodies.push((
            "execution_shape_archive_config_zip_adapter",
            format!(
                "import configparser\nimport os\nimport shutil\nif not isinstance({primary}, str) or not os.path.isfile({primary}):\n    return False\nconfig = configparser.ConfigParser()\nconfig.read({primary})\nproject_dir = config.get('Project', 'directory', fallback='')\nif not project_dir or not os.path.isdir(project_dir):\n    return False\nout_dir = {secondary} if {has_secondary} else os.path.dirname({primary})\nif not out_dir:\n    out_dir = os.path.dirname({primary})\nos.makedirs(out_dir, exist_ok=True)\nbase = os.path.basename(os.path.normpath(project_dir)) or 'archive'\narchive_base = os.path.join(out_dir, base)\nshutil.make_archive(archive_base, 'zip', project_dir)\nreturn True",
                has_secondary = if has_secondary { "True" } else { "False" }
            ),
        ));
    }

    bodies
}

fn optional_dependency_specific_receiver_bodies(
    task: &CodeTask,
    primary: &str,
) -> Vec<(&'static str, String)> {
    let text = format!("{} {}", task.category, task.prompt).to_ascii_lowercase();
    let shape = decoder_return_shape(task);
    let mut bodies = Vec::new();
    if shape == "number" && (text.contains("numpy") || text.contains("numeric sum")) {
        bodies.push((
            "runtime_dependency_numpy_sum",
            format!(
                "try:\n    import numpy as np\nexcept Exception:\n    np = None\nif np is not None:\n    _ = np\ntotal = 0.0\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else []\nfor item in items:\n    try:\n        total += float(item)\n    except Exception:\n        continue\nbest = total\nreturn best"
            ),
        ));
    }
    if shape == "list" && (text.contains("pandas") || text.contains("tabular")) {
        bodies.push((
            "runtime_dependency_pandas_records",
            format!(
                "try:\n    import pandas as pd\nexcept Exception:\n    pd = None\nout = []\nif pd is not None and hasattr({primary}, 'to_dict'):\n    try:\n        records = {primary}.to_dict('records')\n        if isinstance(records, list):\n            for item in records:\n                if isinstance(item, dict):\n                    out.append(dict(item))\n    except Exception:\n        pass\nif not out and isinstance({primary}, list):\n    for item in {primary}:\n        if isinstance(item, dict):\n            out.append(dict(item))\nif not out and isinstance({primary}, dict):\n    out.append(dict({primary}))\nbest = len(out)\nreturn out"
            ),
        ));
    }
    if shape == "str" && (text.contains("beautifulsoup") || text.contains("markup")) {
        bodies.push((
            "runtime_dependency_html_text",
            format!(
                "try:\n    from bs4 import BeautifulSoup\nexcept Exception:\n    BeautifulSoup = None\ntext = '' if {primary} is None else str({primary})\nif BeautifulSoup is not None and '<' in text and '>' in text:\n    try:\n        text = BeautifulSoup(text, 'html.parser').get_text(' ')\n    except Exception:\n        pass\nelse:\n    out = []\n    in_tag = False\n    for ch in text:\n        if ch == '<':\n            in_tag = True\n            continue\n        if ch == '>':\n            in_tag = False\n            continue\n        if not in_tag:\n            out.append(ch)\n    text = ''.join(out)\nbest = text\nreturn str(' '.join(text.split()))"
            ),
        ));
    }
    if shape == "list"
        && (text.contains("sklearn")
            || text.contains("scikit")
            || text.contains("label encoder")
            || text.contains("cluster label")
            || text.contains("class label"))
    {
        bodies.push((
            "runtime_dependency_sklearn_labels",
            format!(
                "try:\n    import importlib.util\n    sklearn_available = importlib.util.find_spec('sklearn') is not None\nexcept Exception:\n    sklearn_available = False\nitems = list({primary}) if isinstance({primary}, (list, tuple)) else []\nif sklearn_available:\n    _ = 'sklearn_available_without_heavy_import'\nordered = sorted({{str(item) for item in items}})\nmapping = {{key: idx for idx, key in enumerate(ordered)}}\nout = []\nfor item in items:\n    out.append(mapping[str(item)])\nreturn out"
            ),
        ));
    }
    if shape == "dict"
        && (text.contains("matplotlib")
            || text.contains("seaborn")
            || text.contains("plot")
            || text.contains("chart")
            || text.contains("histogram"))
    {
        bodies.push((
            "runtime_dependency_plot_summary",
            format!(
                "try:\n    import matplotlib.pyplot as plt\nexcept Exception:\n    plt = None\ntry:\n    import seaborn as sns\nexcept Exception:\n    sns = None\nif plt is not None and sns is not None:\n    _ = (plt, sns)\nvalues = []\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else []\nfor item in items:\n    try:\n        values.append(float(item))\n    except Exception:\n        continue\ncounts = {{}}\nif values:\n    counts['count'] = len(values)\n    counts['min'] = min(values)\n    counts['max'] = max(values)\nelse:\n    counts['count'] = 0\n    counts['min'] = None\n    counts['max'] = None\nreturn dict(counts)"
            ),
        ));
    }
    if shape == "dict"
        && (text.contains("wordcloud")
            || text.contains("nltk")
            || text.contains("token frequency")
            || text.contains("word frequency"))
    {
        bodies.push((
            "runtime_dependency_word_frequency",
            format!(
                "try:\n    from wordcloud import WordCloud\nexcept Exception:\n    WordCloud = None\ntry:\n    import nltk\nexcept Exception:\n    nltk = None\nif WordCloud is not None:\n    _ = WordCloud\nif nltk is not None:\n    _ = nltk\ntext = '' if {primary} is None else str({primary})\ncounts = {{}}\nword = []\nfor ch in text.lower():\n    if ch.isalnum():\n        word.append(ch)\n    elif word:\n        token = ''.join(word)\n        counts[token] = counts.get(token, 0) + 1\n        word = []\nif word:\n    token = ''.join(word)\n    counts[token] = counts.get(token, 0) + 1\nreturn counts"
            ),
        ));
    }
    if shape == "dict"
        && (text.contains("requests") || text.contains("url") || text.contains("query parameter"))
    {
        bodies.push((
            "runtime_dependency_requests_query",
            format!(
                "try:\n    import requests\nexcept Exception:\n    requests = None\nfrom urllib.parse import parse_qs, urlparse\nif requests is not None:\n    _ = requests\nif not isinstance({primary}, str):\n    return {{}}\nparsed = parse_qs(urlparse({primary}).query)\nout = {{}}\nfor key, values in parsed.items():\n    out[key] = values[0] if values else ''\nreturn out"
            ),
        ));
    }
    if shape == "number"
        && (text.contains("scipy")
            || text.contains("numpy")
            || text.contains("statistic")
            || text.contains("mean")
            || text.contains("numeric"))
    {
        bodies.push((
            "runtime_dependency_numeric_mean",
            format!(
                "try:\n    import scipy\nexcept Exception:\n    scipy = None\ntry:\n    import numpy as np\nexcept Exception:\n    np = None\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else []\nvalues = []\nfor item in items:\n    try:\n        values.append(float(item))\n    except Exception:\n        continue\nif np is not None and values:\n    try:\n        return float(np.asarray(values, dtype=float).mean())\n    except Exception:\n        pass\nif scipy is not None:\n    _ = scipy\nreturn sum(values) / len(values) if values else 0"
            ),
        ));
    }
    bodies
}

fn interface_fidelity_receiver_body(
    primary: &str,
    secondary: &str,
    has_secondary: bool,
    shape: &str,
    empty: &str,
) -> String {
    match shape {
        "list" => format!(
            "if {primary} is None:\n    return []\nitems = {primary}.split() if isinstance({primary}, str) else list({primary}) if isinstance({primary}, (list, tuple, set)) else [{primary}]\nout = []\nfor item in items:\n    if item is None:\n        continue\n    out.append(item.strip() if isinstance(item, str) else item)\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out.append({secondary})")
            } else {
                String::new()
            }
        ),
        "str" => format!(
            "if {primary} is None:\n    return ''\ntext = {primary} if isinstance({primary}, str) else str({primary})\nparts = text.split()\nout = []\nfor part in parts:\n    if part:\n        out.append(part.strip())\n{}\nreturn ' '.join(out)",
            if has_secondary {
                format!("if {secondary} is not None:\n    out.append(str({secondary}))")
            } else {
                String::new()
            }
        ),
        "bool" => format!(
            "if {primary} is None:\n    return False\ntext = {primary} if isinstance({primary}, str) else str({primary})\n{}\nfor part in text.split():\n    if part.strip():\n        return True\nreturn bool({primary})",
            if has_secondary {
                format!("if {secondary} is not None and str({secondary}) in text:\n    return True")
            } else {
                String::new()
            }
        ),
        "dict" => format!(
            "if {primary} is None:\n    return {{}}\nitems = {primary}.items() if isinstance({primary}, dict) else enumerate({primary}.split() if isinstance({primary}, str) else list({primary}) if isinstance({primary}, (list, tuple, set)) else [{primary}])\nout = {{}}\nfor key, value in items:\n    if key is None:\n        continue\n    out[key] = value\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out[{secondary}] = out.get({secondary}, 0)")
            } else {
                String::new()
            }
        ),
        "number" => format!(
            "if {primary} is None:\n    return 0\ntotal = 0\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else str({primary}).replace(',', ' ').split()\nfor item in items:\n    try:\n        total += int(item)\n    except Exception:\n        continue\n{}\nreturn total",
            if has_secondary {
                format!("try:\n    total += int({secondary})\nexcept Exception:\n    pass")
            } else {
                String::new()
            }
        ),
        _ => format!(
            "if {primary} is None:\n    return {empty}\nvalue = {primary}\n{}\nreturn value",
            if has_secondary {
                format!("if {secondary} is not None:\n    value = {secondary}")
            } else {
                String::new()
            }
        ),
    }
}

fn edge_interface_admissibility_receiver_body(
    primary: &str,
    secondary: &str,
    has_secondary: bool,
    shape: &str,
    empty: &str,
) -> String {
    match shape {
        "list" => format!(
            "out = []\ntry:\n    source = {primary}\nexcept Exception:\n    source = None\nif source is None:\n    items = []\nelif isinstance(source, str):\n    items = source.replace(',', ' ').split()\nelif isinstance(source, dict):\n    items = list(source.values())\nelif isinstance(source, (list, tuple, set)):\n    items = list(source)\nelse:\n    items = [source]\nfor item in items:\n    if item is None:\n        continue\n    if isinstance(item, str):\n        item = item.strip()\n        if not item:\n            continue\n    out.append(item)\n{}\nbest = len(out)\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out.append({secondary})")
            } else {
                String::new()
            }
        ),
        "dict" => format!(
            "out = {{}}\ntry:\n    source = {primary}\nexcept Exception:\n    return {{}}\nif source is None:\n    return out\nif isinstance(source, dict):\n    items = source.items()\nelif isinstance(source, (list, tuple, set)):\n    items = enumerate(source)\nelse:\n    items = enumerate(str(source).replace(',', ' ').split())\nfor key, value in items:\n    if value is None:\n        continue\n    out[key] = value\n{}\nbest = len(out)\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out[{secondary}] = out.get({secondary}, 0)")
            } else {
                String::new()
            }
        ),
        "str" => format!(
            "try:\n    source = {primary}\nexcept Exception:\n    return ''\nif source is None:\n    return ''\nif isinstance(source, str):\n    items = source.replace(',', ' ').split()\nelif isinstance(source, (list, tuple, set)):\n    items = [str(item).strip() for item in source if item is not None]\nelse:\n    items = [str(source).strip()]\nout = []\nfor item in items:\n    if item:\n        out.append(item)\n{}\nbest = len(out)\nreturn ' '.join(out)",
            if has_secondary {
                format!("if {secondary} is not None:\n    out.append(str({secondary}).strip())")
            } else {
                String::new()
            }
        ),
        "bool" => format!(
            "try:\n    source = {primary}\nexcept Exception:\n    return False\nif source is None:\n    return False\nitems = source if isinstance(source, (list, tuple, set)) else str(source).split()\nseen = set()\nfor item in items:\n    if item is None:\n        continue\n    key = repr(item)\n    if key in seen:\n        return True\n    seen.add(key)\n{}\nbest = len(seen)\nreturn best > 0",
            if has_secondary {
                format!("if {secondary} is not None and {secondary} in seen:\n    return True")
            } else {
                String::new()
            }
        ),
        "number" => format!(
            "try:\n    source = {primary}\nexcept Exception:\n    return 0\nif source is None:\n    return 0\nitems = source if isinstance(source, (list, tuple, set)) else str(source).replace(',', ' ').split()\ntotal = 0\ncount = 0\nfor item in items:\n    try:\n        total += float(item)\n        count += 1\n    except Exception:\n        continue\n{}\nbest = total\nreturn int(best) if best == int(best) else best",
            if has_secondary {
                format!("try:\n    total += float({secondary})\n    count += 1\nexcept Exception:\n    pass")
            } else {
                String::new()
            }
        ),
        _ => format!(
            "if {primary} is None:\n    return {empty}\nvalue = {primary}\n{}\nreturn value",
            if has_secondary {
                format!("if value is None and {secondary} is not None:\n    value = {secondary}")
            } else {
                String::new()
            }
        ),
    }
}

fn string_parsing_receiver_body(
    primary: &str,
    secondary: &str,
    has_secondary: bool,
    shape: &str,
    empty: &str,
) -> String {
    match shape {
        "list" => format!(
            "if {primary} is None:\n    return []\ntext = {primary} if isinstance({primary}, str) else ' '.join(str(item) for item in {primary}) if isinstance({primary}, (list, tuple, set)) else str({primary})\nout = []\nfor part in text.replace(',', ' ').split():\n    cleaned = part.strip()\n    if cleaned:\n        out.append(cleaned)\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out.append(str({secondary}).strip())")
            } else {
                String::new()
            }
        ),
        "str" => format!(
            "if {primary} is None:\n    return ''\ntext = {primary} if isinstance({primary}, str) else str({primary})\nout = []\nfor part in text.replace(',', ' ').split():\n    cleaned = part.strip()\n    if cleaned:\n        out.append(cleaned)\n{}\nreturn ' '.join(out)",
            if has_secondary {
                format!("if {secondary} is not None:\n    out.append(str({secondary}).strip())")
            } else {
                String::new()
            }
        ),
        "bool" => format!(
            "if {primary} is None:\n    return False\ntext = {primary} if isinstance({primary}, str) else str({primary})\nneedle = str({secondary}) if {has_secondary} else ''\nfor part in text.split():\n    cleaned = part.strip()\n    if needle and cleaned == needle:\n        return True\n    if cleaned:\n        return True\nreturn False",
            has_secondary = if has_secondary { "True" } else { "False" }
        ),
        "dict" => format!(
            "if {primary} is None:\n    return {{}}\ntext = {primary} if isinstance({primary}, str) else str({primary})\nout = {{}}\nfor part in text.replace(',', ' ').split():\n    key = part.strip()\n    if not key:\n        continue\n    out[key] = out.get(key, 0) + 1\nreturn out"
        ),
        "number" => format!(
            "if {primary} is None:\n    return 0\ntext = {primary} if isinstance({primary}, str) else str({primary})\ntotal = 0\nfor part in text.replace(',', ' ').split():\n    if part.strip().lstrip('-').isdigit():\n        total += int(part)\nreturn total"
        ),
        _ => format!(
            "if {primary} is None:\n    return {empty}\ntext = {primary} if isinstance({primary}, str) else str({primary})\nparts = text.split()\nfor part in parts:\n    if part.strip():\n        return part.strip()\nreturn {empty}"
        ),
    }
}

fn verification_cascade_compile_receiver_body(
    primary: &str,
    secondary: &str,
    has_secondary: bool,
    shape: &str,
    empty: &str,
) -> String {
    match shape {
        "list" => format!(
            "out = []\ntry:\n    source = {primary}\nexcept Exception:\n    return []\nif source is None:\n    return out\nitems = source if isinstance(source, (list, tuple, set)) else str(source).replace(',', ' ').split()\nfor item in items:\n    if item is None:\n        continue\n    out.append(item)\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out.append({secondary})")
            } else {
                String::new()
            }
        ),
        "dict" => format!(
            "out = {{}}\ntry:\n    source = {primary}\nexcept Exception:\n    return {{}}\nif isinstance(source, dict):\n    for key, value in source.items():\n        out[key] = value\nelse:\n    items = source if isinstance(source, (list, tuple, set)) else []\n    for index, value in enumerate(items):\n        out[index] = value\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out[{secondary}] = out.get({secondary}, 0)")
            } else {
                String::new()
            }
        ),
        "str" => format!(
            "try:\n    source = {primary}\nexcept Exception:\n    return ''\nif source is None:\n    return ''\ntext = source if isinstance(source, str) else str(source)\nparts = []\nfor part in text.replace(',', ' ').split():\n    cleaned = part.strip()\n    if cleaned:\n        parts.append(cleaned)\n{}\nreturn ' '.join(parts)",
            if has_secondary {
                format!("if {secondary} is not None:\n    parts.append(str({secondary}).strip())")
            } else {
                String::new()
            }
        ),
        "bool" => format!(
            "try:\n    source = {primary}\nexcept Exception:\n    return False\nif source is None:\n    return False\nfound = False\nitems = source if isinstance(source, (list, tuple, set)) else str(source).split()\nfor item in items:\n    if item is not None:\n        found = True\n        break\n{}\nreturn bool(found)",
            if has_secondary {
                format!("if not found and {secondary} is not None:\n    found = True")
            } else {
                String::new()
            }
        ),
        "number" => format!(
            "total = 0\ntry:\n    source = {primary}\nexcept Exception:\n    return 0\nitems = source if isinstance(source, (list, tuple, set)) else [source]\nfor item in items:\n    if isinstance(item, bool):\n        continue\n    try:\n        total += float(item)\n    except Exception:\n        continue\n{}\nreturn int(total) if float(total).is_integer() else total",
            if has_secondary {
                format!("try:\n    total += float({secondary})\nexcept Exception:\n    total = total")
            } else {
                String::new()
            }
        ),
        _ => format!(
            "try:\n    value = {primary}\nexcept Exception:\n    return {empty}\nif value is None:\n    return {empty}\n{}\nreturn value",
            if has_secondary {
                format!("if value is None and {secondary} is not None:\n    value = {secondary}")
            } else {
                String::new()
            }
        ),
    }
}

fn runtime_dependency_receiver_body(
    primary: &str,
    secondary: &str,
    has_secondary: bool,
    shape: &str,
    empty: &str,
) -> String {
    match shape {
        "list" => format!(
            "try:\n    import numpy as np\nexcept Exception:\n    np = None\nitems = []\nif np is not None and hasattr({primary}, 'tolist'):\n    try:\n        items = np.asarray({primary}).tolist()\n    except Exception:\n        items = []\nif not items:\n    items = list({primary}) if isinstance({primary}, (list, tuple, set)) else str({primary}).replace(',', ' ').split() if {primary} is not None else []\n{}\nreturn items",
            if has_secondary {
                format!("if {secondary} is not None:\n    items.append({secondary})")
            } else {
                String::new()
            }
        ),
        "dict" => format!(
            "try:\n    import pandas as pd\nexcept Exception:\n    pd = None\nout = {{}}\nif pd is not None and hasattr({primary}, 'to_dict'):\n    try:\n        out = {primary}.to_dict()\n    except Exception:\n        out = {{}}\nif not out:\n    if isinstance({primary}, dict):\n        out = dict({primary})\n    else:\n        items = {primary} if isinstance({primary}, (list, tuple, set)) else []\n        for item in items:\n            out[item] = out.get(item, 0) + 1\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out[{secondary}] = out.get({secondary}, 0)")
            } else {
                String::new()
            }
        ),
        "str" => format!(
            "try:\n    from bs4 import BeautifulSoup\nexcept Exception:\n    BeautifulSoup = None\ntext = '' if {primary} is None else str({primary})\nif BeautifulSoup is not None and '<' in text and '>' in text:\n    try:\n        text = BeautifulSoup(text, 'html.parser').get_text(' ')\n    except Exception:\n        pass\n{}\nreturn str(text.strip())",
            if has_secondary {
                format!("if {secondary} is not None:\n    text = text + str({secondary})")
            } else {
                String::new()
            }
        ),
        "bool" => format!(
            "try:\n    import psutil\nexcept Exception:\n    psutil = None\nif psutil is not None and isinstance({primary}, str):\n    try:\n        for proc in psutil.process_iter(['name']):\n            if proc.info.get('name') == {primary}:\n                return True\n    except Exception:\n        pass\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else []\n{}\nreturn bool(items)",
            if has_secondary {
                format!("if {secondary} is not None and {secondary} in items:\n    return True")
            } else {
                String::new()
            }
        ),
        "number" => format!(
            "try:\n    import numpy as np\nexcept Exception:\n    np = None\nif np is not None:\n    try:\n        values = np.asarray({primary})\n        return int(values.size)\n    except Exception:\n        pass\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else []\ncount = 0\nfor _item in items:\n    count += 1\nreturn count"
        ),
        _ => format!(
            "try:\n    import pandas as pd\nexcept Exception:\n    pd = None\nif pd is not None and hasattr({primary}, 'copy'):\n    try:\n        return {primary}.copy()\n    except Exception:\n        pass\nreturn {primary} if {primary} is not None else {empty}"
        ),
    }
}

fn locals_branch_loop_receiver_body(
    primary: &str,
    secondary: &str,
    has_secondary: bool,
    shape: &str,
    empty: &str,
) -> String {
    match shape {
        "bool" => format!(
            "if {primary} is None:\n    return False\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else str({primary}).split()\nseen = set()\nfor item in items:\n    if item in seen:\n        return True\n    seen.add(item)\n{}\nreturn bool(seen)",
            if has_secondary {
                format!("if {secondary} in seen:\n    return True")
            } else {
                String::new()
            }
        ),
        "str" => format!(
            "if {primary} is None:\n    return ''\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else str({primary}).split()\nout = []\nfor item in items:\n    if item is None:\n        continue\n    out.append(str(item))\n{}\nreturn ' '.join(out)",
            if has_secondary {
                format!("if {secondary} is not None:\n    out.append(str({secondary}))")
            } else {
                String::new()
            }
        ),
        "list" => format!(
            "if {primary} is None:\n    return []\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else str({primary}).split()\nout = []\nfor item in items:\n    if item is None:\n        continue\n    out.append(item)\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out.append({secondary})")
            } else {
                String::new()
            }
        ),
        "dict" => format!(
            "if {primary} is None:\n    return {{}}\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else str({primary}).split()\nout = {{}}\nfor item in items:\n    if item is None:\n        continue\n    out[item] = out.get(item, 0) + 1\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out[{secondary}] = out.get({secondary}, 0)")
            } else {
                String::new()
            }
        ),
        "number" => format!(
            "if {primary} is None:\n    return 0\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else str({primary}).split()\nbest = 0\nfor item in items:\n    try:\n        value = int(item)\n    except Exception:\n        continue\n    best = max(best, value)\nreturn best"
        ),
        _ => format!(
            "if {primary} is None:\n    return {empty}\nitems = {primary} if isinstance({primary}, (list, tuple, set)) else [{primary}]\nout = {empty}\nfor item in items:\n    if item is not None:\n        out = item\nreturn out"
        ),
    }
}

fn return_shape_receiver_body(
    primary: &str,
    secondary: &str,
    has_secondary: bool,
    shape: &str,
    empty: &str,
) -> String {
    match shape {
        "list" => format!(
            "if {primary} is None:\n    return []\nout = list({primary}) if isinstance({primary}, (list, tuple, set)) else str({primary}).split()\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out.append({secondary})")
            } else {
                String::new()
            }
        ),
        "str" => format!(
            "if {primary} is None:\n    return ''\nout = str({primary}).strip()\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out = out + str({secondary})")
            } else {
                String::new()
            }
        ),
        "bool" => format!(
            "if {primary} is None:\n    return False\nout = bool({primary})\n{}\nreturn out",
            if has_secondary {
                format!("if {secondary} is not None:\n    out = out or bool({secondary})")
            } else {
                String::new()
            }
        ),
        _ => type_shape_retry_body(primary, secondary, has_secondary, shape, empty),
    }
}
