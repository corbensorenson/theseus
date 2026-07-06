"""Visible public task export and SymLiquid routing helpers for Code LM closure.

This module owns calibration-manifest shaping and visible-category routing.
It never exports public tests or reference solutions into training rows.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import real_code_benchmark_graduation as real_code
from public_code_case_manifest import filter_tasks_for_card, load_case_manifest, manifest_pool_size

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path, default: Any = None) -> Any:
    default = {} if default is None else default
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current

def export_public_visible_tasks(
    cards: list[str],
    *,
    seed: int,
    max_cases: int,
    case_manifest: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_by_card = load_case_manifest(case_manifest)
    manifest_enabled = bool(str(case_manifest or "").strip())
    for card_id in cards:
        card = read_json(ROOT / "benchmarks" / "cards" / f"{card_id}.json", {})
        source_id = str(card.get("source_id") or card_id.replace("source_", ""))
        source_path = real_code.resolve_source_path(card)
        manifest_rows = manifest_by_card.get(card_id, [])
        load_limit = manifest_pool_size(max_cases, {card_id: manifest_rows}) if manifest_rows else max_cases
        if source_path.exists():
            tasks, evidence_level, _semantics = real_code.load_cases(card_id, source_id, source_path, seed, load_limit)
        else:
            tasks = []
            evidence_level = "public_metadata_contract_only"
        if manifest_rows:
            tasks, _missing = filter_tasks_for_card(tasks, manifest_rows)
        if not manifest_enabled and len(tasks) < max_cases:
            tasks = supplement_with_metadata_contract_tasks(
                tasks,
                card_id=card_id,
                source_id=source_id,
                card=card,
                seed=seed,
                max_cases=max_cases,
                evidence_level=evidence_level,
            )
        for task in tasks:
            prompt = str(task.get("prompt") or "")
            entry = str(task.get("entry_point") or "")
            rows.append(
                {
                    "task_id": str(task.get("task_id") or ""),
                    "source_task_id": str(task.get("source_task_id") or ""),
                    "card_id": card_id,
                    "source_id": source_id,
                    "split": "public_calibration",
                    "case_type": str(task.get("case_type") or ""),
                    "category": str(task.get("category") or "")
                    or infer_visible_category(prompt=prompt, entry_point=entry),
                    "prompt": prompt,
                    "entry_point": entry,
                    "tags": [str(tag) for tag in task.get("tags", [])] if isinstance(task.get("tags"), list) else [],
                    "benchmark_evidence_level": evidence_level,
                    "case_manifest_selected": bool(manifest_rows),
                    "visible_task_only": True,
                    "tests_exported": False,
                    "canonical_solution_exported": False,
                    "public_tests_used": False,
                    "public_solutions_used": False,
                }
            )
    return rows


def supplement_with_metadata_contract_tasks(
    tasks: list[dict[str, Any]],
    *,
    card_id: str,
    source_id: str,
    card: dict[str, Any],
    seed: int,
    max_cases: int,
    evidence_level: str,
) -> list[dict[str, Any]]:
    """Fill sparse public surfaces from visible benchmark-card metadata only.

    On machines without the public benchmark payloads staged, the old fallback
    used tiny loader repair manifests that include local test strings. Decoder
    fanout only needs visible prompt/signature metadata, so this fallback emits
    source-specific adapter contract prompts and explicitly omits tests and
    reference answers.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        if task_id and task_id not in seen:
            seen.add(task_id)
            out.append(task)
    if len(out) >= max_cases:
        return out[:max_cases]

    needed = max_cases - len(out)
    for row in rotate_metadata_contracts(metadata_contract_tasks(card_id, source_id, card), seed):
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        row["benchmark_evidence_level"] = (
            evidence_level if evidence_level != "source_staged_no_local_task_cases" else "public_metadata_contract_only"
        )
        out.append(row)
        needed -= 1
        if needed <= 0:
            break
    return out[:max_cases]


def metadata_contract_tasks(card_id: str, source_id: str, card: dict[str, Any]) -> list[dict[str, Any]]:
    source_slug = safe_identifier(source_id or card_id.replace("source_", ""))
    card_slug = safe_identifier(card_id.replace("source_", ""))
    label = str(card.get("name") or source_id or card_id).replace("\n", " ")[:96]
    license_id = str(card.get("license_spdx") or "unknown")
    adapter_type = str(card.get("adapter_type") or "code_eval_adapter")
    capability = str(card.get("capability_target") or "local coding benchmark")
    base_tags = ["public_metadata", "benchmark_adapter", source_slug]
    specs = [
        (
            "extract_entry_point",
            f"{source_slug}_extract_entry_point",
            "extract_def_name",
            f"Write a Python function named {source_slug}_extract_entry_point. "
            f"Given visible {label} prompt text, return the first Python function name declared in it.",
            ["parsing", "entry_point"],
        ),
        (
            "has_required_keys",
            f"{source_slug}_has_required_keys",
            "dict_required_keys",
            f"Write a Python function named {source_slug}_has_required_keys. "
            f"Given a metadata row and a list of required keys for {label}, return whether every key is present.",
            ["schema", "type_handling"],
        ),
        (
            "normalize_signature",
            f"{source_slug}_normalize_signature",
            "string_parsing",
            f"Write a Python function named {source_slug}_normalize_signature. "
            "Normalize a visible Python signature by trimming whitespace and removing duplicate spaces.",
            ["signature", "string_parsing"],
        ),
        (
            "stable_dedupe",
            f"{source_slug}_stable_dedupe",
            "edge_case",
            f"Write a Python function named {source_slug}_stable_dedupe. "
            f"Return unique {label} metadata tags while preserving their first-seen order.",
            ["edge_case", "list_ops"],
        ),
        (
            "route_capability",
            f"{source_slug}_route_capability",
            "algorithm_choice",
            f"Write a Python function named {source_slug}_route_capability. "
            f"Route a visible benchmark metadata row to one of parsing, type_handling, algorithm_choice, or edge_case.",
            ["routing", "algorithm_choice"],
        ),
        (
            "safe_head",
            f"{source_slug}_safe_head",
            "edge_case",
            f"Write a Python function named {source_slug}_safe_head. "
            f"Return the first visible {label} case id, or a default value when the list is empty.",
            ["edge_case", "safe_access"],
        ),
        (
            "license_allowed",
            f"{source_slug}_license_allowed",
            "interface_fidelity",
            f"Write a Python function named {source_slug}_license_allowed. "
            f"Return whether a benchmark card license string is compatible with local calibration metadata use. "
            f"This card advertises {license_id}.",
            ["license", "policy"],
        ),
        (
            "summarize_adapter",
            f"{source_slug}_summarize_adapter",
            "locals_branch_loop_obligations",
            f"Write a Python function named {source_slug}_summarize_adapter. "
            f"Build a compact dictionary describing adapter={adapter_type}, source={source_id}, and capability={capability}.",
            ["adapter", "dict"],
        ),
        (
            "count_visible_cases",
            f"{source_slug}_count_visible_cases",
            "type_handling",
            f"Write a Python function named {source_slug}_count_visible_cases. "
            f"Count visible metadata rows for {label} while ignoring rows marked hidden or private.",
            ["counting", "metadata"],
        ),
        (
            "split_task_id",
            f"{source_slug}_split_task_id",
            "string_parsing",
            f"Write a Python function named {source_slug}_split_task_id. "
            f"Split a {label} task id into benchmark prefix and local item id.",
            ["parsing", "task_id"],
        ),
        (
            "merge_metadata",
            f"{source_slug}_merge_metadata",
            "dict_required_keys",
            f"Write a Python function named {source_slug}_merge_metadata. "
            f"Merge two visible metadata dictionaries for {label}, preferring non-empty values from the second dictionary.",
            ["dict", "metadata"],
        ),
        (
            "sort_by_priority",
            f"{source_slug}_sort_by_priority",
            "sort_by_second",
            f"Write a Python function named {source_slug}_sort_by_priority. "
            f"Sort visible {label} routing tuples by their numeric priority field.",
            ["sorting", "priority"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for index, (suffix, entry_point, category, prompt, tags) in enumerate(specs):
        rows.append(
            {
                "task_id": f"{card_slug}_metadata_{suffix}_{index:02d}",
                "source_task_id": f"{card_id}:{suffix}",
                "card_id": card_id,
                "source_id": source_id,
                "split": "public_calibration",
                "category": category,
                "prompt": prompt,
                "entry_point": entry_point,
                "tags": base_tags + tags,
                "benchmark_evidence_level": "public_metadata_contract_only",
                "visible_task_only": True,
                "tests_exported": False,
                "canonical_solution_exported": False,
                "public_tests_used": False,
                "public_solutions_used": False,
                "metadata_contract_only": True,
                "score_semantics": "visible public benchmark-card metadata only; no public tests or solutions",
            }
        )
    return rows


def rotate_metadata_contracts(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    offset = seed % len(rows)
    return rows[offset:] + rows[:offset]


def safe_identifier(value: str) -> str:
    ident = re.sub(r"\W+", "_", value.lower()).strip("_")
    if not ident:
        return "source"
    if ident[0].isdigit():
        ident = f"source_{ident}"
    return ident


def prioritize_private_rows_for_public_categories(
    private_rows: list[dict[str, Any]],
    public_tasks: list[dict[str, Any]],
    *,
    symliquid_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Move visible-public-category lookalikes ahead of generic train rows.

    Step-budgeted Rust runs trim train rows by work, so frontier-relevant
    private families need to be early in the private train stream. This uses
    only visible prompt category labels from public tasks; it does not inspect
    public tests, public solutions, or task outcomes.
    """
    categories = []
    seen = set()
    for task in public_tasks:
        category = str(task.get("category") or "").strip()
        if category and category not in seen:
            seen.add(category)
            categories.append(category)
    category_set = set(categories)
    state_categories = set(symliquid_priority_categories(symliquid_state or {}))
    priority_set = category_set | state_categories
    indexed = list(enumerate(private_rows))

    def priority(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        idx, row = item
        split = str(row.get("split") or "")
        category = str(row.get("category") or "")
        if split == "train" and category in priority_set:
            return (0, idx)
        if split == "train":
            return (1, idx)
        if category in priority_set:
            return (2, idx)
        return (3, idx)

    prioritized = [row for _idx, row in sorted(indexed, key=priority)]
    train_hits = sum(
        1
        for row in prioritized
        if str(row.get("split") or "") == "train" and str(row.get("category") or "") in priority_set
    )
    return prioritized, {
        "policy": "visible_public_category_priority_no_public_answers_v1",
        "category_count": len(categories),
        "categories": categories,
        "symliquid_priority_categories": sorted(state_categories),
        "prioritized_private_train_rows": train_hits,
        "symliquid_state_loaded": bool((symliquid_state or {}).get("loaded")),
        "public_tests_used": False,
        "public_solutions_used": False,
        "score_semantics": "curriculum_ordering_only_not_public_evidence",
    }


def load_symliquid_state(args: argparse.Namespace) -> dict[str, Any]:
    path = resolve(getattr(args, "symliquid_state_engine_report", "reports/symliquid_state_engine.json"))
    payload = read_json(path, {})
    if payload.get("policy") != "project_theseus_symliquid_state_engine_v1":
        return {
            "loaded": False,
            "path": rel(path),
            "reason": "missing_or_wrong_policy",
            "priority_categories": [],
            "context": "",
        }
    categories = symliquid_priority_categories(payload)
    route_hints = [
        f"{row.get('kind')}:{row.get('route')}:{row.get('weight')}"
        for row in payload.get("route_hints", [])
        if isinstance(row, dict)
    ][:8]
    active_slots = [
        f"{row.get('capability')}={row.get('activation')}"
        for row in payload.get("state_slots", [])
        if isinstance(row, dict) and row.get("status") == "active"
    ][:8]
    return {
        "loaded": True,
        "path": rel(path),
        "priority_categories": categories,
        "strongest_action_kind": get_path(payload, ["summary", "strongest_action_kind"], ""),
        "route_hints": route_hints,
        "active_slots": active_slots,
        "code_lm_closure_args": payload.get("code_lm_closure_args", {}),
        "context": "; ".join(route_hints + active_slots),
        "public_tests_used": False,
        "public_solutions_used": False,
        "promotion_evidence": False,
    }


def symliquid_priority_categories(state: dict[str, Any]) -> list[str]:
    strongest = str(get_path(state, ["summary", "strongest_action_kind"], "") or "")
    hints = state.get("route_hints") if isinstance(state.get("route_hints"), list) else []
    kinds = {strongest}
    for row in hints:
        if isinstance(row, dict):
            kinds.add(str(row.get("kind") or ""))
    categories: set[str] = set()
    if "train_private_semantic_residual_family" in kinds:
        categories.update(
            {
                "below_threshold",
                "dict_required_keys",
                "extract_def_name",
                "is_prime",
                "max_list",
                "pluck_smallest_even",
                "same_chars",
                "sort_by_second",
                "sum_list",
                "three_sum_zero_exists",
                "triangle_area_product",
                "two_sum_zero_exists",
                "list_tail_replace",
                "opposite_signs",
            }
        )
    if "train_repo_repair_trace_checkpoint" in kinds:
        categories.update(
            {
                "off_by_one_loop",
                "string_parsing_edge",
                "type_shape_mismatch",
                "selection_logic",
                "repo_repair",
            }
        )
    if "run_same_seed_sts_repair_ablation" in kinds:
        categories.update({"recurrence_state", "string_rule_composition", "digit_rotation"})
    if "train_edge_contract_v2_private_gate" in kinds:
        categories.update(
            {
                "private_edge_v2_window_extrema",
                "private_edge_v2_normalized_lookup",
                "private_edge_v2_until_gap",
                "private_edge_v2_record_filter",
                "private_edge_v2_running_balance",
                "private_edge_v2_jagged_columns",
                "private_edge_v2_token_histogram",
                "private_edge_v2_pairwise_flags",
                "interface_contracts",
                "return_shape",
                "branch_loop_skeleton",
                "local_state_updates",
                "type_family_handling",
            }
        )
    return sorted(category for category in categories if category)


def symliquid_sts_context(state: dict[str, Any]) -> str:
    if not state.get("loaded"):
        return ""
    parts = []
    strongest = str(state.get("strongest_action_kind") or "")
    if strongest:
        parts.append(f"strongest_action={strongest}")
    categories = state.get("priority_categories") if isinstance(state.get("priority_categories"), list) else []
    if categories:
        parts.append("priority_categories=" + ",".join(str(item) for item in categories[:12]))
    route_hints = state.get("route_hints") if isinstance(state.get("route_hints"), list) else []
    if route_hints:
        parts.append("routes=" + "|".join(str(item) for item in route_hints[:4]))
    return " ; ".join(parts)


def infer_visible_category(*, prompt: str, entry_point: str) -> str:
    text = f"{entry_point} {prompt}".lower()
    if "divide a csv file" in text and ("shuffle" in text or "split_" in text):
        return "private_exec_csv_split_shuffle"
    if "zip" in text and "not including subdirectories" in text:
        return "private_exec_zip_flat_directory"
    if "shell commands" in text and "csv file" in text and ("separate files" in text or "output directory" in text):
        return "private_exec_csv_command_outputs"
    if "config file" in text and "project directory" in text and ("zip file" in text or "archive" in text):
        return "private_exec_archive_config_zip"
    if "operating system" in text and "architecture" in text and "memory usage" in text:
        return "private_exec_system_info_dict"
    if ("url-encoded" in text or "urlencode" in text) and "payload" in text:
        return "private_exec_urlencode_payload"
    if "json" in text and ("file path" in text or "json file" in text) and ("retrieve" in text or "field" in text):
        return "private_exec_json_extract_field"
    if "process is running" in text and "restart" in text:
        return "private_exec_process_restart"
    rules = [
        ("private_exec_archive_config_zip", ["configuration file format", "project directory", "archive directory"]),
        ("private_exec_csv_command_outputs", ["shell commands read from a csv", "commands_file_path", "separate files", "command's output"]),
        ("private_exec_log_backup_tar", [".log files", "tar.gz", "delete the original", "logs_backup"]),
        ("private_exec_csv_split_shuffle", ["divide a csv file", "smaller files", "shuffle the lines", "split_"]),
        ("private_exec_zip_flat_directory", ["zip all files", "not including subdirectories", "specified directory"]),
        ("private_exec_system_info_dict", ["operating system", "architecture", "memory usage"]),
        ("private_exec_json_extract_field", ["json file path", "retrieve a field", "payload.get"]),
        ("private_exec_urlencode_payload", ["url-encoded payload", "urlencode", "payload dictionary"]),
        ("private_exec_process_restart", ["process is running", "restart", "process_name"]),
        ("remove_vowels", ["remove_vowels", "without vowels", "string without vowels"]),
        ("caesar_decode_shift5", ["decode_shift", "encoded with encode_shift", "shifting every character by 5"]),
        ("modular_power_two", ["modp", "2^n modulo", "2 ** n modulo", "two raised to n modulo"]),
        ("below_threshold", ["below_threshold", "below threshold", "all numbers", "strictly below"]),
        ("add_numbers", ["def add", "add two numbers", "sum of two numbers"]),
        ("same_chars", ["same_chars", "same characters", "same set of characters"]),
        ("median_list", ["def median", "return median", "median of elements"]),
        ("string_odd_index_remove", ["odd index values", "odd indices", "characters which have odd index"]),
        ("min_three", ["minimum of three", "smallest of three"]),
        ("stable_negative_partition", ["negative elements appear before positive", "relative order among negative and positive", "re-arranges the first n elements"]),
        ("replace_whitespace", ["replace blank spaces", "replaces blank spaces", "replace spaces in the string"]),
        ("top_k_largest", ["n largest items", "largest items from the list", "n largest"]),
        ("cube_lateral_surface_area", ["lateral surface area of a cube", "lateralsurface_cube", "lateral surface cube"]),
        ("cylinder_lateral_surface_area", ["lateral surface area of a cylinder", "lateralsuface_cylinder", "lateralsurface_cylinder"]),
        ("cube_volume", ["volume of a cube", "cube given its side length"]),
        ("string_char_count", ["total number of characters in a string", "count the total number of characters"]),
        ("nonempty_substring_count", ["number of non-empty substrings", "non-empty substrings"]),
        ("list_tail_replace", ["replaces the last element", "replace the last element"]),
        ("tuple_frequency_dict", ["dictionary mapping each unique tuple", "unique tuple to the number of times"]),
        ("tuple_item_count", ["counts the occcurences", "counts the occurrences", "count occurrences of the element", "element in the tuple"]),
        ("count_integer_items", ["number of integer elements", "integer elements in a given list"]),
        ("split_list_at_index", ["splits the given list into two parts", "length of the first part"]),
        ("swap_pair", ["tuple with the second number", "second number and then the first"]),
        ("tuple_elementwise_division", ["division operation element-wise", "element-wise across the given tuples"]),
        ("tuple_nested_elementwise_max", ["maximize_elements", "maximize the given two tuples", "maximise the given two tuples"]),
        ("tuple_elementwise_max", ["element-wise maxima", "elementwise maxima"]),
        ("insert_before_each", ["inserts the element before each element", "insert the element before each element"]),
        ("count_primes_below", ["number of prime numbers less than", "prime numbers less than"]),
        ("next_perfect_square", ["next perfect square", "perfect square greater"]),
        ("harmonic_sum", ["harmonic sum"]),
        ("list_chunks_every_n", ["splits a list for every nth element", "split a list for every nth element", "list_split"]),
        ("combinations_with_replacement", ["combinations (with repetition)", "combinations with repetition", "combinations_colors"]),
        ("rescale_to_unit", ["rescale_to_unit", "linear transform", "smallest number will become 0", "largest will become 1"]),
        ("decode_cyclic", ["decode_cyclic", "encoded with encode_cyclic", "returns encoded string by cycling groups of three"]),
        ("prime_fib_sequence", ["prime_fib", "fibonacci number and it's also prime", "fibonacci number and it is also prime"]),
        ("polynomial_zero_bisection", ["find_zero", "zero point", "coefficients of a polynomial"]),
        ("largest_divisor", ["largest_divisor", "largest number that divides", "divides n evenly"]),
        ("opposite_signs", ["opposite sign", "opposite_sign"]),
        ("all_prefixes", ["all prefixes", "prefixes"]),
        ("string_sequence", ["space-delimited numbers", "starting from 0", "0 upto n", "0 through n"]),
        ("one_less_than_twice_reverse", ["one less than twice", "twice its reverse", "twice the reverse"]),
        ("distinct_count", ["distinct characters", "how many distinct", "count_distinct"]),
        ("largest_concat", ["given list of digits", "formed with the given list of digits", "formed with the given list"]),
        ("circular_digit_shift", ["circular_shift", "circular shift", "shift the digits", "digits right", "return digits reversed"]),
        ("reverse_string", ["reverse"]),
        ("palindrome", ["palindrome"]),
        ("largest_prime_factor", ["largest prime factor"]),
        ("is_prime", ["def is_prime", "check if a number is prime", "whether a number is prime"]),
        ("prime_factors", ["prime factors", "factorize"]),
        ("factors", ["factor"]),
        ("largest_divisor", ["proper divisor", "largest divisor"]),
        ("safe_head", ["safe_head", "first item", "fallback"]),
        ("dict_required_keys", ["has_required_keys", "required keys", "required_keys"]),
        ("public_private_count", ["count_public_tests", "public test cases", "public tests"]),
        ("extract_def_name", ["extract_entry_point", "entry point", "function name"]),
        ("symbol_beat_parser", ["musical notes", "whole note", "half note", "quater note", "quarter note", "parse_music"]),
        ("common_elements", ["common elements", "common values", "def common", "intersection of two arrays"]),
        ("sorted_unique_values", ["sorted unique", "unique elements", "return sorted unique", "def unique"]),
        ("sort_even_index_values", ["even indicies", "even indices", "sort_even", "even positions"]),
        ("count_digit_under_divisibility", ["digit 7", "fizz_buzz"]),
        ("divisible_by_11", ["divisible by 11"]),
        ("woodall_number_check", ["woodall", "woodball"]),
        ("polygonal_octagonal_number", ["octagonal number"]),
        ("polygonal_tetrahedral_number", ["tetrahedral number"]),
        ("polygonal_centered_hexagonal_number", ["centered hexagonal"]),
        ("sphere_volume", ["volume of a sphere"]),
        ("sphere_surface_area", ["surface area of a sphere"]),
        ("sort_by_second", ["sort a list of tuples using the second value", "using the second value", "sort pairs by the second"]),
        ("nested_flat_sum", ["flatten a list and sum", "flatten a list", "sum all of its elements"]),
        ("positive_count", ["count the number of positive numbers", "positive numbers in a list"]),
        ("positive_filter", ["return only positive numbers", "only positive numbers in the list", "get_positive"]),
        ("sublist_contains", ["contains the given sublist", "given sublist"]),
        ("equal_tuple_lengths", ["tuples have equal length", "all the given tuples have equal length"]),
        ("sort_list", ["sort a list of elements", "comb_sort"]),
        ("difference_of_squares_check", ["difference of two squares"]),
        ("same_pattern_sequence", ["samepatterns", "same pattern", "sequence given in the patterns"]),
        ("tuple_all_divisible", ["tuples which have all elements divisible", "all elements divisible by k"]),
        ("odd_length_check", ["length of the word is odd", "length is odd"]),
        ("ascii_mod_char", ["ascii value of all the characters", "modulo 26"]),
        ("dict_merge_three", ["merge three dictionaries"]),
        ("frequency_dict", ["frequency of all the elements", "returned as a dictionary", "freq_count"]),
        ("closest_smaller_number", ["closest smaller number"]),
        ("longest_word_length", ["length of the longest word"]),
        ("substring_in_list", ["string is present as a substring", "substring in a given list"]),
        ("overlapping_substring_count", ["overlaping cases", "overlapping cases", "how_many_times"]),
        ("spelled_number_sort", ["space-delimited string of numberals", "numbers sorted from smallest", "sort_numbers"]),
        ("closest_pair_sorted", ["select and return two that are the closest", "closest to each other"]),
        ("unique_once_stable", ["remove all elements that occur more than once", "remove_duplicates"]),
        ("flip_case", ["flip lowercase characters", "flip_case", "uppercase to lowercase"]),
        ("concat_strings", ["concatenate list of strings", "concatenate("]),
        ("filter_by_prefix", ["starts with a given prefix", "filter_by_prefix"]),
        ("sort_indices_multiple_three", ["indices that are divisible by three", "sort_third"]),
        ("car_race_collision_count", ["car_race_collision", "n cars are driving left to right"]),
        ("digit_substring_length_sum_count", ["sum of digits equal to their length", "count_substrings"]),
        ("bell_number_sequence", ["bell number", "bell numbers"]),
        ("newman_conway_sequence", ["newman conway", "newman-conway"]),
        ("three_sum_zero_exists", ["three distinct", "three elements", "triples_sum_to_zero"]),
        ("two_sum_zero_exists", ["two distinct", "two elements", "pairs_sum_to_zero"]),
        ("base_digits", ["change numerical base", "change_base", "string representation after the conversion"]),
        ("increment_each_item", ["incremented by 1", "incr_list", "elements incremented"]),
        ("tribonacci_sequence", ["fibfib", "fibonacci-like", "fibonacci like"]),
        ("triangle_area_sides", ["three sides", "valid triangle", "three sides form a valid triangle"]),
        ("triangle_area_product", ["side and high", "side and height"]),
        ("balanced_brackets_simple", ["correct bracketing", "correct_bracketing", "balanced bracket", "parentheses", "brackets"]),
        ("monotonic_sequence", ["monotonic", "non-decreasing", "non-increasing", "nondecreasing", "nonincreasing"]),
        ("arithmetic_series_sum", ["sum_to_n", "sum to n", "sum from 0", "sum from zero"]),
        ("derivative_coefficients", ["derivative", "polynomial"]),
        ("fibonacci_loop_private", ["def fib", "n-th fibonacci", "nth fibonacci", "fibonacci number"]),
        ("rotate_sequence", ["circular shift", "circular_shift", "rotate"]),
        ("uppercase_ascii_sum", ["digitsum", "upper characters", "ascii codes", "uppercase letters"]),
        ("digit_sum_casefold", ["digit sum", "sum of digits"]),
        ("fruit_distribution_private", ["fruit_distribution", "apples", "oranges", "mangoes", "fruits"]),
        ("pluck_smallest_even", ["pluck", "smallest even", "smallest index"]),
        ("frequency_at_least_value", ["frequency greater than or equal", "frequency is the number of times", "greatest integer"]),
        ("alternating_min_max_sort", ["strange_sort", "strange sort", "minimum value, then maximum"]),
        ("palindrome_list_weight", ["will_it_fly", "palindromic list", "maximum possible weight"]),
        ("smallest_palindrome_changes", ["smallest_change", "make the array palindromic", "minimum number of elements"]),
        ("total_match_lengths", ["total_match", "total number of chars", "lists of strings"]),
        ("multiply_three_primes", ["multiplication of 3 prime", "multiply_prime", "3 prime numbers"]),
        ("simple_power", ["simple_power", "n**int", "exact power"]),
        ("cube_number", ["iscube", "is a cube", "perfect cube"]),
        ("hex_prime_count", ["hex_key", "hexadecimal digits", "2, 3, 5, 7, b", "prime digits"]),
        ("parse_ints", ["parse integers", "parse integer", "extract integers", "integer-looking tokens"]),
        ("count_vowels", ["vowel"]),
        ("sum_list", ["sum of a list", "sum of list", "sum the list", "return the sum"]),
        ("filter_integers", ["filter given list", "only for integers", "filter_integers"]),
        ("max_tuple_difference", ["maximum difference between available pairs", "maximum difference between pairs"]),
        ("max_list", ["maximum element", "max_element", "return maximum"]),
        ("min_list", ["minimum element", "smallest number in a list", "return minimum"]),
        ("length", ["length of a collection", "length of a list", "length of a string", "length of given string", "strlen"]),
    ]
    for category, needles in rules:
        if any(needle in text for needle in needles):
            return category
    return ""
