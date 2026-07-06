"""Decoder contract construction and public contract preflight for Code LM closure.

This module uses visible task metadata only. Public tests and public solutions stay
out of the contract/preflight path; the output is a routing/guardrail signal.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Any

from code_lm_private_verifier import concept_residual_label

def attach_decoder_contracts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["decoder_contract"] = merged_decoder_contract(item)
        out.append(item)
    return out


def merged_decoder_contract(task: dict[str, Any]) -> dict[str, Any]:
    base = decoder_contract_for_task(task)
    override = task.get("decoder_contract")
    if not isinstance(override, dict) or not override:
        return base
    allowed_keys = {
        "policy",
        "return_shape",
        "type_family",
        "visible_arg_count_hint",
        "required_constructs",
        "residual_label_hint",
        "full_body_required",
        "guardrail_only",
        "feedback_weight",
        "score_semantics",
        "argument_roles",
        "return_contract",
        "generation_plan",
        "skeleton_bias",
        "repair_strategy",
        "verifier_feedback",
    }
    for key in allowed_keys:
        value = override.get(key)
        if value not in (None, "", []):
            if key == "visible_arg_count_hint":
                base_value = int_or_none(base.get(key))
                override_value = int_or_none(value)
                if base_value is not None and override_value is not None:
                    base[key] = max(base_value, override_value)
                elif override_value is not None:
                    base[key] = override_value
                continue
            base[key] = value
    arg_count = int_or_none(base.get("visible_arg_count_hint"))
    if arg_count is not None:
        inferred_roles = argument_roles_for_count(
            arg_count,
            str(task.get("category") or ""),
            str(task.get("prompt") or ""),
        )
        existing_roles = base.get("argument_roles") if isinstance(base.get("argument_roles"), dict) else {}
        base["argument_roles"] = {**inferred_roles, **existing_roles, **inferred_roles}
    base["base_policy"] = "project_theseus_decoder_contract_v1"
    base["feedback_override_used"] = True
    base["public_tests_used"] = False
    base["public_solutions_used"] = False
    return base


def decoder_contract_for_task(task: dict[str, Any]) -> dict[str, Any]:
    category = str(task.get("category") or "")
    prompt = str(task.get("prompt") or "")
    constructs = required_constructs_for_task(category, prompt)
    return_shape = return_shape_for_category(category, prompt)
    type_family = type_family_for_category(category, prompt)
    arg_count = visible_arg_count_hint_for_task(task)
    return {
        "policy": "project_theseus_decoder_contract_v1",
        "category": category,
        "return_shape": return_shape,
        "type_family": type_family,
        "visible_arg_count_hint": arg_count,
        "required_constructs": constructs,
        "residual_label_hint": concept_residual_label(task, ""),
        "full_body_required": bool(constructs),
        "guardrail_only": True,
        "argument_roles": argument_roles_for_count(arg_count or 0, category, prompt),
        "return_contract": return_contract_for_shape(return_shape, category, prompt),
        "generation_plan": decoder_generation_plan(
            category=category,
            prompt=prompt,
            return_shape=return_shape,
            type_family=type_family,
            constructs=constructs,
        ),
        "public_tests_used": False,
        "public_solutions_used": False,
        "score_semantics": "visible prompt/category contract for routing and decoding pressure, not benchmark evidence",
    }


def argument_roles_for_category(category: str, prompt: str) -> dict[str, str]:
    arg_count = visible_arg_count_hint_for_category(category)
    return argument_roles_for_count(arg_count or 0, category, prompt)


def argument_roles_for_count(arg_count: int, category: str, prompt: str) -> dict[str, str]:
    text = f"{category} {prompt}".lower()
    if arg_count == 0:
        return {}
    roles = {"data": "primary_visible_input"}
    if arg_count and arg_count >= 2:
        if "threshold" in text or "below" in text:
            roles["other"] = "threshold_or_limit"
        elif "window" in text or "chunk" in text or "top k" in text:
            roles["other"] = "size_or_count"
        elif "target" in text or "needle" in text:
            roles["other"] = "target_or_lookup_value"
        else:
            roles["other"] = "secondary_visible_input"
    if arg_count and arg_count >= 3:
        roles["extra"] = "remaining_visible_inputs"
    return roles


def return_contract_for_shape(shape: str, category: str, prompt: str) -> dict[str, Any]:
    text = f"{category} {prompt}".lower()
    empty_defaults = {
        "list": "[]",
        "dict": "{}",
        "tuple": "()",
        "str": "''",
        "bool": "False",
        "number": "0",
    }
    return {
        "shape": shape,
        "empty_or_invalid_behavior": empty_defaults.get(shape, "None")
        if any(token in text for token in ["empty", "invalid", "missing", "none", "no "])
        else "infer_from_visible_prompt",
        "must_preserve_container_shape": shape in {"list", "dict", "tuple", "str"},
        "source": "visible_prompt_and_private_contract_metadata_only",
    }


def decoder_generation_plan(
    *,
    category: str,
    prompt: str,
    return_shape: str,
    type_family: str,
    constructs: list[str],
) -> dict[str, Any]:
    text = f"{category} {prompt}".lower()
    skeleton_bias: list[str] = []
    if "loop" in constructs or any(token in text for token in ["each", "every", "scan", "iterate", "window", "prefix"]):
        skeleton_bias.append("loop_over_primary_input")
    if "branch" in constructs or any(token in text for token in ["empty", "invalid", "missing", "fallback", "if "]):
        skeleton_bias.append("boundary_guard_branch")
    if "locals" in constructs or any(token in text for token in ["count", "sum", "best", "state", "balance"]):
        skeleton_bias.append("named_local_state")
    if return_shape in {"list", "dict"}:
        skeleton_bias.append(f"{return_shape}_return_builder")
    if return_shape == "tuple":
        skeleton_bias.append("tuple_return_builder")
    if type_family == "execution_shaped_program":
        skeleton_bias.append("library_action_plan")
    if not skeleton_bias:
        skeleton_bias.append("minimal_visible_signature_body")
    return {
        "policy": "signature -> argument_roles -> return_contract -> semantic_family -> state_variables -> branch_loop_skeleton -> body -> repair",
        "skeleton_bias": skeleton_bias,
        "repair_strategy": "use verifier rejection reasons to choose a stricter skeleton before token-level repair",
        "verifier_feedback": [
            "visible_argument_mismatch",
            "return_shape_mismatch",
            "missing_required_skeleton",
            "execution_library_mismatch",
            "semantic_family_mismatch",
            "semantic_admissibility_rejected",
        ],
        "public_tests_used": False,
        "public_solutions_used": False,
    }


def return_shape_for_category(category: str, prompt: str) -> str:
    text = f"{category} {prompt}".lower()
    bool_categories = {
        "below_threshold",
        "same_chars",
        "opposite_signs",
        "palindrome",
        "is_prime",
        "is_anagram",
        "dict_required_keys",
        "balanced_brackets_simple",
        "monotonic_sequence",
        "two_sum_zero_exists",
        "three_sum_zero_exists",
        "divisible_by_11",
        "palindrome_list_weight",
        "multiply_three_primes",
        "simple_power",
        "cube_number",
        "sublist_contains",
        "equal_tuple_lengths",
        "difference_of_squares_check",
        "same_pattern_sequence",
        "odd_length_check",
        "private_exec_archive_config_zip",
    }
    number_categories = {
        "sum_list",
        "add_numbers",
        "abs_diff",
        "max_list",
        "min_list",
        "min_three",
        "median_list",
        "median_odd",
        "length",
        "distinct_count",
        "string_char_count",
        "nonempty_substring_count",
        "tuple_item_count",
        "count_integer_items",
        "count_primes_below",
        "count_truthy",
        "count_vowels",
        "final_y_vowel_private",
        "suffix_y_vowel_private",
        "case_punct_vowel_private",
        "word_count",
        "positive_count",
        "negative_count",
        "count_digit_under_divisibility",
        "hex_prime_count",
        "arithmetic_series_sum",
        "harmonic_sum",
        "frequency_at_least_value",
        "largest_divisor",
        "largest_prime_factor",
        "next_perfect_square",
        "gcd_pair",
        "modular_power_two",
        "triangle_area_product",
        "triangle_area_sides",
        "cube_volume",
        "cube_lateral_surface_area",
        "cylinder_lateral_surface_area",
        "sphere_volume",
        "sphere_surface_area",
        "polygonal_octagonal_number",
        "polygonal_tetrahedral_number",
        "polygonal_centered_hexagonal_number",
        "tribonacci_sequence",
        "fibonacci_loop_private",
        "lucas_loop_private",
        "shifted_recurrence_private",
        "nested_recurrence_private",
        "polynomial_zero_bisection",
    }
    list_categories = {
        "sorted_unique_values",
        "all_prefixes",
        "filter_integers",
        "list_tail_replace",
        "stable_negative_partition",
        "top_k_largest",
        "insert_before_each",
        "list_chunks_every_n",
        "prime_factors",
        "factors",
        "derivative_coefficients",
        "positive_filter",
        "sort_by_second",
        "stable_dedupe",
        "matrix_diagonal",
        "transpose_matrix",
        "remove_none",
        "parse_ints",
        "symbol_beat_parser",
        "powers_of_two",
        "pluck_smallest_even",
        "alternating_min_max_sort",
        "total_match_lengths",
        "tuple_all_divisible",
        "unique_once_stable",
        "filter_by_prefix",
        "sort_indices_multiple_three",
        "private_exec_csv_command_outputs",
        "private_exec_csv_split_shuffle",
    }
    str_categories = {
        "caesar_decode_shift5",
        "remove_vowels",
        "reverse_string",
        "string_sequence",
        "base_digits",
        "circular_digit_shift",
        "decode_cyclic",
        "parse_ints_text",
        "extract_def_name",
        "replace_whitespace",
        "spelled_number_sort",
        "flip_case",
        "concat_strings",
        "ascii_mod_char",
        "digit_rotate_right_private",
        "signed_digit_rotate_private",
        "multi_step_digit_shift_private",
        "normalize_string",
        "remove_spaces",
        "title_case_words",
        "private_exec_log_backup_tar",
        "private_exec_urlencode_payload",
    }
    dict_categories = {
        "tuple_frequency_dict",
        "frequency_dict",
        "dict_merge_three",
        "private_exec_system_info_dict",
    }
    tuple_categories = {
        "split_list_at_index",
        "swap_pair",
        "closest_pair",
        "closest_pair_sorted",
        "tuple_elementwise_division",
        "tuple_elementwise_max",
        "tuple_nested_elementwise_max",
    }
    mixed_shape_categories = {"rotate_sequence", "take_every_other", "safe_head"}
    if category in bool_categories or any(needle in text for needle in ["return whether", "return true", "returns true", "check if", "whether"]):
        return "bool"
    if category in number_categories:
        return "number"
    if category in list_categories:
        return "list"
    if category in str_categories:
        return "str"
    if category in dict_categories:
        return "dict"
    if category in tuple_categories:
        return "tuple"
    if category in mixed_shape_categories:
        return "unknown"
    explicit = explicit_return_shape_from_prompt(prompt)
    if explicit:
        return explicit
    if any(needle in text for needle in ["count", "sum", "area", "volume", "number", "median", "minimum", "maximum", "largest", "smallest", "gcd"]):
        return "number"
    if "tuple" in text or category in {"swap_pair", "closest_pair", "closest_pair_sorted"}:
        return "tuple"
    if "dictionary" in text or "dict" in text:
        return "dict"
    if "list" in text or "array" in text:
        return "list"
    if any(needle in text for needle in ["string", "text", "word", "decode", "characters"]):
        return "str"
    return "unknown"


def explicit_return_shape_from_prompt(prompt: str) -> str:
    """Infer the requested output shape from visible return-language only.

    This intentionally looks at the prompt/starter text, not tests or reference
    solutions. It prevents argument annotations such as ``s: str`` from
    overpowering instructions like "Return an integer".
    """

    raw = str(prompt or "")
    text = " ".join(raw.lower().split())
    match = re.search(
        r"\breturns?\s*:?(.*?)(?:\braises?\s*:|\brequirements?\s*:|\bexample\s*:|\bnotes?\s*:|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return_text = match.group(1).strip()
    else:
        return_windows = re.findall(r"\breturns?\b[^.\n]*", text)
        return_text = " ".join(return_windows[:4])
    if not return_text:
        return ""

    # Prefer explicit return type declarations over incidental words in the
    # prose. BigCodeBench-style prompts often mention input lists/dicts before
    # saying "Returns: tuple/DataFrame/Axes"; using the full prompt made the
    # decoder choose list-shaped bodies for tuple-return tasks.
    leading = return_text[:180]
    if any(
        token in leading
        for token in [
            "whether",
            "true if",
            "false otherwise",
            "boolean",
            "bool",
        ]
    ):
        return "bool"
    if any(token in leading for token in ["tuple[", "tuple:", "a tuple", "tuple containing", "pairgrid object after plotting"]):
        return "tuple"
    if any(token in leading for token in ["collections.counter", "counter:", "counter object"]):
        return "dict"
    if (
        any(token in return_text for token in ["dataframe", "data frame"])
        and any(token in return_text for token in ["axes", "axis", "figure", "plot", "pairgrid", "list[axes"])
    ):
        return "tuple"
    if any(token in leading for token in ["dict:", "dictionary:", "mapping:", "a dictionary"]):
        return "dict"
    if any(
        token in leading
        for token in [
            "list:",
            "list[",
            "list of",
            "a list",
            "array:",
            "array of",
            "a sequence",
        ]
    ):
        return "list"
    if any(token in leading for token in ["str:", "string:", "a string", "encoded string", "text content"]):
        return "str"
    if any(token in leading for token in ["int:", "integer:", "float:", "number:", "numeric", "a numeric"]):
        return "number"
    if any(token in return_text for token in ["whether", "true if", "false otherwise", "boolean", "bool"]):
        return "bool"
    if any(token in return_text for token in ["tuple", "pair of"]):
        return "tuple"
    if any(token in return_text for token in ["dictionary", "dict", "counter", "mapping"]):
        return "dict"
    if any(token in return_text for token in ["list of", "array of", "return a list", "return an array"]):
        return "list"
    if any(
        token in return_text
        for token in [
            "integer",
            "an int",
            "a number",
            "the number",
            "count",
            "length",
            "minimum",
            "maximum",
            "sum",
            "median",
            "gcd",
        ]
    ):
        return "number"
    if any(token in return_text for token in ["string", "str", "lexicographically"]):
        return "str"
    return ""


def type_family_for_category(category: str, prompt: str) -> str:
    text = f"{category} {prompt}".lower()
    if any(token in text for token in ["archive", "zip file", "zipfile", "tar", "csv", "file", "directory", "subprocess", "system", "json", "payload", "urlencode"]):
        return "execution_shaped_program"
    if any(token in text for token in ["prime", "factor", "gcd", "divisor", "fibonacci", "recurrence", "tribonacci"]):
        return "number_theory_or_recurrence"
    if any(token in text for token in ["string", "text", "char", "word", "vowel", "decode", "substring"]):
        return "string_indexing"
    if any(token in text for token in ["list", "array", "tuple", "elements", "sequence"]):
        return "collection_logic"
    if any(token in text for token in ["threshold", "whether", "true", "false", "check"]):
        return "predicate_logic"
    return "general_semantics"


def visible_arg_count_hint_for_category(category: str) -> int | None:
    two_arg = {
        "abs_diff",
        "below_threshold",
        "add_numbers",
        "same_chars",
        "gcd_pair",
        "is_anagram",
        "base_digits",
        "replace_whitespace",
        "top_k_largest",
        "split_list_at_index",
        "swap_pair",
        "substring_count",
        "index_or_minus_one",
        "common_elements",
        "list_chunks_every_n",
        "safe_head",
        "clamp_number",
        "rotate_sequence",
        "circular_digit_shift",
        "palindrome_list_weight",
        "tuple_all_divisible",
        "substring_in_list",
        "filter_by_prefix",
    }
    three_arg = {"min_three", "triangle_area_sides", "dict_merge_three"}
    if category in three_arg:
        return 3
    if category in {
        "private_exec_archive_config_zip",
        "private_exec_csv_command_outputs",
        "private_exec_log_backup_tar",
        "private_exec_json_extract_field",
    }:
        return 2
    if category in {
        "private_exec_zip_flat_directory",
        "private_exec_csv_split_shuffle",
        "private_exec_system_info_dict",
        "private_exec_urlencode_payload",
    }:
        return 1 if category != "private_exec_system_info_dict" else 0
    if category in two_arg:
        return 2
    return 1 if category else None


def visible_arg_count_hint_for_task(task: dict[str, Any]) -> int | None:
    category = str(task.get("category") or "")
    hints = [visible_arg_count_hint_for_category(category), visible_arg_count_from_prompt(task)]
    if not bool(task.get("public_benchmark")):
        hints.extend(
            [
                visible_arg_count_from_private_tests(task),
                visible_arg_count_from_private_solution(task),
            ]
        )
    observed = [hint for hint in hints if hint is not None]
    return max(observed) if observed else None


def visible_arg_count_from_prompt(task: dict[str, Any]) -> int | None:
    text = f"{task.get('category') or ''} {task.get('prompt') or ''}".lower()
    if any(token in text for token in ["three scalar", "three numbers", "three values", "three list"]):
        return 3
    if any(
        token in text
        for token in [
            "two numbers",
            "two scalar",
            "two values",
            "two strings",
            "two lists",
            "two tuples",
            "a list and",
            "a string and",
            "provided character",
            "with the provided",
        ]
    ):
        return 2
    return None


def visible_arg_count_from_private_tests(task: dict[str, Any]) -> int | None:
    tests = str(task.get("tests") or "")
    entry_point = str(task.get("entry_point") or "")
    if not tests or not entry_point:
        return None
    counts: list[int] = []
    try:
        tree = ast.parse(tests)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == entry_point:
                counts.append(len(node.args) + len(node.keywords))
    except SyntaxError:
        pattern = re.compile(rf"\b{re.escape(entry_point)}\s*\((.*?)\)", re.DOTALL)
        for match in pattern.finditer(tests):
            counts.append(count_call_args_from_text(match.group(1)))
    return max(counts) if counts else None


def visible_arg_count_from_private_solution(task: dict[str, Any]) -> int | None:
    body = f"{task.get('solution_body') or ''}\n{task.get('solution_expr') or ''}"
    if re.search(r"\bextra\s*\[", body) or re.search(r"\bextra\b", body):
        return 3
    if re.search(r"\bother\b", body):
        return 2
    if re.search(r"\bdata\b", body):
        return 1
    return None


def count_call_args_from_text(text: str) -> int:
    depth = 0
    in_string = ""
    escaped = False
    count = 1 if text.strip() else 0
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = ""
            continue
        if ch in {"'", '"'}:
            in_string = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            count += 1
    return count


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def required_constructs_for_task(category: str, prompt: str) -> list[str]:
    constructs = list(required_constructs_for_category(category))
    seen = set(constructs)
    text = f"{category} {prompt}".lower()

    def add(*items: str) -> None:
        for item in items:
            if item and item not in seen:
                seen.add(item)
                constructs.append(item)

    loop_needles = [
        "all ",
        "any ",
        "array",
        "bell",
        "count",
        "digit",
        "divisor",
        "each",
        "every",
        "factor",
        "frequency",
        "given list",
        "iterate",
        "largest",
        "list",
        "matrix",
        "newman",
        "prime",
        "range",
        "recurrence",
        "reverse",
        "scan",
        "sequence",
        "smallest",
        "string",
        "substring",
        "sum",
        "tuple",
        "window",
        "woodall",
    ]
    branch_needles = [
        "check",
        "empty",
        "exist",
        "false",
        "given number",
        "if ",
        "invalid",
        "is ",
        "missing",
        "not ",
        "or not",
        "palindrome",
        "true",
        "valid",
        "whether",
    ]
    local_needles = [
        "balance",
        "best",
        "count",
        "current",
        "frequency",
        "index",
        "largest",
        "maximum",
        "minimum",
        "result",
        "reverse",
        "smallest",
        "state",
        "total",
    ]
    selection_needles = [
        "largest",
        "maximum",
        "median",
        "minimum",
        "smallest",
        "sort",
        "sorted",
        "top",
    ]
    frequency_needles = ["count", "frequency", "occurrence", "repeated", "unique"]
    collection_needles = ["array", "dict", "dictionary", "element", "list", "matrix", "set", "tuple"]
    string_needles = [
        "character",
        "decode",
        "digit",
        "replace",
        "reverse",
        "rotate",
        "string",
        "substring",
        "vowel",
        "word",
    ]
    algorithmic_needles = [
        "bell",
        "divisible",
        "factor",
        "gcd",
        "newman",
        "prime",
        "recurrence",
        "woodall",
    ]
    arithmetic_needles = [
        "area",
        "centered hexagonal",
        "cube",
        "cylinder",
        "divisible by",
        "lateral surface",
        "modulo",
        "nth",
        "octagonal",
        "perfect square",
        "power",
        "sphere",
        "square",
        "surface area",
        "tetrahedral",
        "volume",
    ]

    if any(token in text for token in loop_needles):
        add("loop")
    if any(token in text for token in branch_needles):
        add("branch")
    if any(token in text for token in local_needles) or {"loop", "branch"} & seen:
        add("locals")
    if any(token in text for token in frequency_needles):
        add("frequency")
    if any(token in text for token in selection_needles):
        add("selection")
    if any(token in text for token in collection_needles):
        add("collection_ops")
    if any(token in text for token in string_needles):
        add("index_or_string_ops")
    if any(token in text for token in algorithmic_needles):
        add("algorithmic_planning")
    if any(token in text for token in arithmetic_needles):
        add("arithmetic_formula")
    return constructs


def public_decoder_contract_preflight(public_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Block expensive training when public receiver contracts are visibly weak.

    This uses only visible prompts/categories/contracts. It does not inspect
    public tests or reference solutions.
    """

    algorithmic_needles = [
        "array",
        "bell",
        "check",
        "count",
        "digit",
        "divisor",
        "element",
        "factor",
        "frequency",
        "given list",
        "largest",
        "list",
        "matrix",
        "newman",
        "palindrome",
        "prime",
        "recurrence",
        "reverse",
        "sequence",
        "smallest",
        "sort",
        "string",
        "substring",
        "sum",
        "tuple",
        "whether",
        "woodall",
        "nth",
        "octagonal",
        "tetrahedral",
        "hexagonal",
        "sphere",
        "volume",
        "surface area",
        "lateral surface",
        "power",
        "square",
        "modulo",
    ]
    varargs_tasks: list[str] = []
    missing_contract_tasks: list[str] = []
    weak_required_construct_tasks: list[str] = []
    weak_full_body_tasks: list[str] = []
    suspicious_tasks: list[dict[str, Any]] = []
    construct_counts: Counter[str] = Counter()
    for task in public_tasks:
        task_id = str(task.get("task_id") or task.get("source_task_id") or "")
        category = str(task.get("category") or "")
        prompt = str(task.get("prompt") or "")
        text = f"{category} {prompt}".lower()
        contract = task.get("decoder_contract") if isinstance(task.get("decoder_contract"), dict) else {}
        constructs = [str(item) for item in contract.get("required_constructs") or []]
        construct_counts.update(constructs)
        needs_real_body = any(token in text for token in algorithmic_needles)
        if re.search(r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(\s*\*args", prompt):
            varargs_tasks.append(task_id)
        if not contract:
            missing_contract_tasks.append(task_id)
        if needs_real_body and not constructs:
            weak_required_construct_tasks.append(task_id)
        if constructs and not bool(contract.get("full_body_required")):
            weak_full_body_tasks.append(task_id)
        if len(suspicious_tasks) < 12 and (
            task_id in varargs_tasks
            or task_id in missing_contract_tasks
            or task_id in weak_required_construct_tasks
            or task_id in weak_full_body_tasks
        ):
            suspicious_tasks.append(
                {
                    "task_id": task_id,
                    "card_id": task.get("card_id"),
                    "category": category,
                    "prompt_header": prompt.splitlines()[0] if prompt.splitlines() else "",
                    "required_constructs": constructs,
                    "full_body_required": bool(contract.get("full_body_required")),
                }
            )
    total = len(public_tasks)
    varargs_rate = len(varargs_tasks) / total if total else 0.0
    hard_blockers = []
    if missing_contract_tasks:
        hard_blockers.append("missing_decoder_contract")
    if varargs_rate > 0.05:
        hard_blockers.append("public_signature_varargs_rate_too_high")
    if weak_required_construct_tasks:
        hard_blockers.append("visible_algorithmic_tasks_missing_required_constructs")
    if weak_full_body_tasks:
        hard_blockers.append("construct_tasks_not_marked_full_body_required")
    return {
        "policy": "project_theseus_public_decoder_contract_preflight_v1",
        "passed": not hard_blockers,
        "public_tests_used": False,
        "public_solutions_used": False,
        "public_task_count": total,
        "varargs_task_count": len(varargs_tasks),
        "varargs_rate": round(varargs_rate, 6),
        "missing_contract_count": len(missing_contract_tasks),
        "weak_required_construct_count": len(weak_required_construct_tasks),
        "weak_full_body_count": len(weak_full_body_tasks),
        "construct_counts": dict(sorted(construct_counts.items())),
        "hard_blockers": hard_blockers,
        "examples": suspicious_tasks,
    }


def required_constructs_for_category(category: str) -> list[str]:
    constructs: list[str] = []
    loop_categories = {
        "below_threshold",
        "caesar_decode_shift5",
        "count_vowels",
        "is_prime",
        "base_digits",
        "prime_factors",
        "factors",
        "largest_prime_factor",
        "balanced_brackets_simple",
        "decode_cyclic",
        "parse_ints",
        "pluck_smallest_even",
        "frequency_at_least_value",
        "alternating_min_max_sort",
        "count_primes_below",
        "list_chunks_every_n",
        "two_sum_zero_exists",
        "three_sum_zero_exists",
        "private_exec_csv_command_outputs",
        "private_exec_log_backup_tar",
        "private_exec_zip_flat_directory",
        "private_exec_csv_split_shuffle",
    }
    branch_categories = {
        "below_threshold",
        "median_list",
        "palindrome",
        "is_prime",
        "base_digits",
        "balanced_brackets_simple",
        "safe_head",
        "dict_required_keys",
        "triangle_area_sides",
        "sublist_contains",
        "same_pattern_sequence",
        "private_exec_archive_config_zip",
        "private_exec_csv_command_outputs",
        "private_exec_log_backup_tar",
        "private_exec_zip_flat_directory",
        "private_exec_csv_split_shuffle",
        "private_exec_json_extract_field",
        "private_exec_urlencode_payload",
        "private_exec_process_restart",
    }
    local_categories = loop_categories | branch_categories | {"same_chars", "add_numbers"}
    if category in loop_categories:
        constructs.append("loop")
    if category in branch_categories:
        constructs.append("branch")
    if category in local_categories:
        constructs.append("locals")
    if category in {"common_elements", "same_chars", "sorted_unique_values", "frequency_dict", "tuple_frequency_dict"}:
        constructs.append("collection_ops")
    if category in {"caesar_decode_shift5", "decode_cyclic", "circular_digit_shift", "base_digits", "count_vowels"}:
        constructs.append("index_or_string_ops")
    if category.startswith("private_exec_") or "exec_" in category:
        constructs.extend(["execution_shaped_program", "edge_conditions"])
        if category in {
            "private_exec_archive_config_zip",
            "private_exec_csv_command_outputs",
            "private_exec_log_backup_tar",
            "private_exec_zip_flat_directory",
            "private_exec_csv_split_shuffle",
            "private_exec_json_extract_field",
        }:
            constructs.append("file_path")
        if category in {"private_exec_csv_command_outputs", "private_exec_csv_split_shuffle"}:
            constructs.append("csv")
        if category in {"private_exec_json_extract_field", "private_exec_urlencode_payload"}:
            constructs.append("structured_parsing")
        if category in {"private_exec_process_restart", "private_exec_csv_command_outputs"}:
            constructs.append("system_api")
    return constructs


