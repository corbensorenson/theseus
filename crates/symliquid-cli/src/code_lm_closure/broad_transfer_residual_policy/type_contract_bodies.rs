use super::*;

pub(super) fn type_contract_v2_receiver_bodies(
    task: &CodeTask,
    primary: &str,
    secondary: &str,
    has_secondary: bool,
    shape: &str,
) -> Vec<(&'static str, String)> {
    let text =
        format!("{} {} {}", task.category, task.prompt, task.tags.join(" ")).to_ascii_lowercase();
    let mut bodies = Vec::new();
    let has_secondary_literal = if has_secondary { "True" } else { "False" };
    let generation_hints = decoder_contract_generation_hints(task);
    let has_hint = |needle: &str| generation_hints.contains(needle);
    if has_secondary
        && shape == "list"
        && (has_hint("zip_both_arguments")
            || has_hint("numeric_pair_guard")
            || (text.contains("pairwise") && text.contains("sum"))
            || (text.contains("shorter sequence") && text.contains("numeric")))
    {
        bodies.push((
            "type_contract_pairwise_numeric_zip",
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
            "type_contract_strip_lower_nonempty_list",
            format!(
                "items = {primary} if isinstance({primary}, (list, tuple, set)) else [{primary}]\nout = []\nfor item in items:\n    text = '' if item is None else str(item).strip().lower()\n    if text:\n        out.append(text)\nreturn out"
            ),
        ));
    }
    if has_secondary
        && shape == "list"
        && (has_hint("nested_dict_walk")
            || has_hint("target_compare_with_second_argument")
            || (text.contains("dot-separated") && text.contains("nested"))
            || (text.contains("nested dictionaries") && text.contains("target")))
    {
        bodies.push((
            "type_contract_nested_dict_target_paths",
            format!(
                "out = []\nstack = [({primary}, '')]\nwhile stack:\n    value, path = stack.pop()\n    if not isinstance(value, dict):\n        continue\n    keys = sorted(value.keys(), key=lambda item: str(item), reverse=True)\n    for key in keys:\n        child = value.get(key)\n        next_path = str(key) if not path else path + '.' + str(key)\n        if child == {secondary}:\n            out.append(next_path)\n        if isinstance(child, dict):\n            stack.append((child, next_path))\nreturn sorted(out)"
            ),
        ));
    }
    if shape == "list"
        && (text.contains("mixed_int")
            || text.contains("integer values")
            || text.contains("numeric_text")
            || text.contains("mixed numbers")
            || text.contains("parsed from mixed"))
    {
        bodies.push((
            "type_contract_mixed_int_values",
            format!(
                "out = []\nstack = list({primary}) if isinstance({primary}, (list, tuple, set)) else [{primary}]\nwhile stack:\n    item = stack.pop(0)\n    if isinstance(item, (list, tuple, set)):\n        stack.extend(item)\n        continue\n    if isinstance(item, bool):\n        continue\n    if isinstance(item, int):\n        out.append(item)\n        continue\n    if isinstance(item, float) and item.is_integer():\n        out.append(int(item))\n        continue\n    for raw in str(item).replace(',', ' ').split():\n        part = raw.strip()\n        sign = -1 if part.startswith('-') else 1\n        digits = part[1:] if part.startswith('-') else part\n        if digits.isdigit():\n            out.append(sign * int(digits))\nreturn out"
            ),
        ));
    }
    if shape == "list"
        && (text.contains("numeric_string_parser")
            || text.contains("signed integer tokens")
            || text.contains("comma or space separated")
            || (text.contains("parse") && text.contains("integer") && text.contains("string")))
    {
        bodies.push((
            "type_contract_numeric_string_parser",
            format!(
                "source = '' if {primary} is None else str({primary})\nout = []\nfor raw in source.replace(',', ' ').split():\n    part = raw.strip()\n    if not part:\n        continue\n    sign = -1 if part.startswith('-') else 1\n    digits = part[1:] if part.startswith('-') else part\n    if digits.isdigit():\n        out.append(sign * int(digits))\nreturn out"
            ),
        ));
    }
    if shape == "str"
        && (text.contains("status")
            || text.contains("canonical")
            || text.contains("label")
            || text.contains("normalize_status"))
    {
        bodies.push((
            "type_contract_normalize_status_label",
            format!(
                "value = {primary}\nif isinstance(value, dict):\n    for key in ('status', 'result', 'state', 'label'):\n        if key in value:\n            value = value[key]\n            break\ntext = str(value).strip().lower().replace('-', ' ').replace('_', ' ')\nif text in {{'pass', 'passed', 'ok', 'success', 'green'}}:\n    return 'pass'\nif text in {{'fail', 'failed', 'error', 'red'}}:\n    return 'fail'\nif text in {{'skip', 'skipped', 'ignore', 'yellow'}}:\n    return 'skip'\nreturn 'unknown'"
            ),
        ));
    }
    if shape == "str"
        && (text.contains("test type")
            || text.contains("case type")
            || text.contains("case kind")
            || text.contains("visibility")
            || text.contains("normalize_test")
            || text.contains("normalize case"))
    {
        bodies.push((
            "type_contract_case_kind_label",
            format!(
                "value = {primary}\nif isinstance(value, dict):\n    for key in ('kind', 'type', 'case_type', 'visibility', 'name', 'label'):\n        if key in value:\n            value = value[key]\n            break\ntext = str(value).strip().lower().replace('-', ' ').replace('_', ' ')\nif text in {{'public', 'visible', 'open', 'shown'}}:\n    return 'visible'\nif text in {{'hidden', 'secret', 'withheld'}}:\n    return 'hidden'\nif text in {{'sample', 'example', 'demo'}}:\n    return 'sample'\nif text in {{'private', 'local', 'internal'}}:\n    return 'private'\nreturn 'unknown'"
            ),
        ));
    }
    if shape == "number"
        && text.contains("count")
        && (text.contains("test")
            || text.contains("case")
            || text.contains("visibility")
            || text.contains("public")
            || text.contains("visible"))
    {
        bodies.push((
            "type_contract_count_case_kind",
            format!(
                "target = str({secondary}) if {has_secondary_literal} else 'visible'\ntarget = target.strip().lower().replace('-', ' ').replace('_', ' ')\nif target == 'public':\n    target = 'visible'\nrecords = {primary} if isinstance({primary}, (list, tuple)) else []\ntotal = 0\nfor record in records:\n    value = record\n    if isinstance(record, dict):\n        for key in ('kind', 'type', 'case_type', 'visibility', 'name', 'label'):\n            if key in record:\n                value = record[key]\n                break\n    text = str(value).strip().lower().replace('-', ' ').replace('_', ' ')\n    if text in {{'public', 'open', 'shown'}}:\n        text = 'visible'\n    if text == target:\n        total += 1\nreturn total",
            ),
        ));
    }
    if shape == "str"
        && (text.contains("safe head")
            || text.contains("first non-empty")
            || text.contains("first item")
            || text.contains("head of"))
    {
        bodies.push((
            "type_contract_safe_head_text",
            format!(
                "items = {primary} if isinstance({primary}, (list, tuple)) else []\nfor item in items:\n    text = '' if item is None else str(item).strip()\n    if text:\n        return text\nreturn ''"
            ),
        ));
    }
    if has_secondary
        && shape == "bool"
        && (text.contains("required key")
            || text.contains("required_key")
            || text.contains("mapping contains")
            || text.contains("key names"))
    {
        bodies.push((
            "type_contract_required_keys_normalized",
            format!(
                "if not isinstance({primary}, dict):\n    return False\npresent = {{str(key).strip().lower() for key in {primary}.keys()}}\nrequired = {secondary} if isinstance({secondary}, (list, tuple, set)) else [{secondary}]\nfor key in required:\n    if str(key).strip().lower() not in present:\n        return False\nreturn True"
            ),
        ));
    }
    if shape == "str"
        && (text.contains("first non-empty")
            || text.contains("safe_extraction")
            || text.contains("first_text")
            || text.contains("normalized text value"))
    {
        bodies.push((
            "type_contract_first_text_value",
            format!(
                "stack = list({primary}) if isinstance({primary}, (list, tuple)) else [{primary}]\nwhile stack:\n    item = stack.pop(0)\n    if isinstance(item, (list, tuple)):\n        stack = list(item) + stack\n        continue\n    if isinstance(item, dict):\n        stack = list(item.values()) + stack\n        continue\n    text = str(item).strip().lower() if item is not None else ''\n    if text:\n        return text\nreturn ''"
            ),
        ));
    }
    if has_secondary
        && shape == "number"
        && (text.contains("nested")
            || text.contains("count nested")
            || text.contains("key or value")
            || text.contains("nested_structure"))
    {
        bodies.push((
            "type_contract_count_nested_entries",
            format!(
                "target = str({secondary}).strip().lower()\ncount = 0\nstack = [{primary}]\nwhile stack:\n    item = stack.pop()\n    if isinstance(item, dict):\n        for key, value in item.items():\n            if str(key).strip().lower() == target:\n                count += 1\n            stack.append(value)\n    elif isinstance(item, (list, tuple, set)):\n        stack.extend(item)\n    elif str(item).strip().lower() == target:\n        count += 1\nreturn count"
            ),
        ));
    }
    if shape == "number"
        && (text.contains("parse_signed_ints")
            || text.contains("signed integers embedded")
            || text.contains("signed integer")
            || text.contains("numeric text"))
    {
        bodies.push((
            "type_contract_sum_signed_ints",
            format!(
                "import re\nif isinstance({primary}, bytes):\n    text = {primary}.decode('utf-8', errors='ignore')\nelif {primary} is None:\n    text = ''\nelif isinstance({primary}, (list, tuple, set)):\n    text = ' '.join('' if item is None else str(item) for item in {primary})\nelse:\n    text = str({primary})\ntotal = 0\nfor token in re.findall(r'[-+]?\\d+', text):\n    try:\n        total += int(token)\n    except Exception:\n        continue\nreturn total"
            ),
        ));
    }
    if shape == "str"
        && (text.contains("entry point")
            || text.contains("entry_point")
            || text.contains("function name")
            || text.contains("source text"))
    {
        bodies.push((
            "type_contract_extract_entry_name",
            format!(
                "best = ''\nif isinstance({primary}, dict):\n    for key in ('entry_point', 'entry', 'name', 'function'):\n        value = {primary}.get(key)\n        if isinstance(value, str) and value.strip():\n            best = value.strip()\n            break\nif best == '':\n    text = str({primary})\n    for line in text.splitlines():\n        stripped = line.strip()\n        if stripped.startswith('def ') and '(' in stripped:\n            parts = stripped[4:].split('(', 1)\n            best = parts[0].strip()\n            break\nreturn str(best)"
            ),
        ));
    }
    if has_secondary
        && shape == "list"
        && (text.contains("score flags")
            || text.contains("numeric score")
            || text.contains("record order")
            || text.contains("record_score"))
    {
        bodies.push((
            "type_contract_score_flags",
            format!(
                "out = []\nrecords = {primary} if isinstance({primary}, (list, tuple)) else []\nfor record in records:\n    score = record.get('score') if isinstance(record, dict) else None\n    try:\n        value = float(score)\n    except Exception:\n        out.append(False)\n        continue\n    out.append(value >= {secondary})\nreturn out"
            ),
        ));
    }
    if shape == "dict"
        && (text.contains("group_counts")
            || text.contains("group counts")
            || (text.contains("normalize") && text.contains("count"))
            || (text.contains("strip") && text.contains("lower") && text.contains("count")))
    {
        bodies.push((
            "type_contract_normalized_group_counts",
            format!(
                "items = {primary} if isinstance({primary}, (list, tuple, set)) else []\nout = {{}}\nfor item in items:\n    if item is None:\n        continue\n    key = str(item).strip().lower()\n    if not key:\n        continue\n    out[key] = out.get(key, 0) + 1\nreturn out"
            ),
        ));
    }
    if shape == "dict"
        && (text.contains("mapping_labels")
            || text.contains("label count")
            || text.contains("label/count")
            || text.contains("label")
            || text.contains("count mapping"))
    {
        bodies.push((
            "type_contract_label_count_mapping",
            format!(
                "out = {{}}\nif isinstance({primary}, dict):\n    for key, value in {primary}.items():\n        try:\n            number = int(value)\n        except Exception:\n            try:\n                number = int(float(value))\n            except Exception:\n                continue\n        out[str(key)] = out.get(str(key), 0) + number\n    return out\nrecords = {primary} if isinstance({primary}, (list, tuple, set)) else []\nfor record in records:\n    if isinstance(record, dict):\n        label = record.get('label', record.get('key', record.get('name')))\n        value = record.get('count', record.get('value', 1))\n    elif isinstance(record, (list, tuple)) and len(record) >= 2:\n        label = record[0]\n        value = record[1]\n    else:\n        continue\n    if label is None:\n        continue\n    try:\n        number = int(value)\n    except Exception:\n        try:\n            number = int(float(value))\n        except Exception:\n            number = 1\n    key = str(label).strip().lower()\n    if key:\n        out[key] = out.get(key, 0) + number\nreturn out"
            ),
        ));
    }
    bodies
}
