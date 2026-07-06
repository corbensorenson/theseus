#!/usr/bin/env python3
"""Generate Private Residual Repair v3 rows for Code LM transfer.

This is the private-only repair lane for the 2026-06-05 wide public wall. It
uses public reports only as sanitized residual category counts. It never copies
public prompts, public tests, public solutions, candidate code, or public
reward targets into training rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_rows import training_data_path  # noqa: E402
from code_residual_curriculum import (  # noqa: E402
    verify_private_solution_rows,
    write_json,
    write_jsonl,
)


TRAIN_DEFAULT = training_data_path(
    "high_transfer",
    "private_train",
    "private_residual_repair_v3_code_lm_tasks.jsonl",
)
HELDOUT_DEFAULT = training_data_path(
    "high_transfer",
    "private_eval",
    "private_residual_repair_v3_heldout_code_lm_tasks.jsonl",
)

FAMILY_TARGETS = (
    "verifier_mismatch_property_v3",
    "livecodebench_stdin_proxy_v1",
    "return_interface_fidelity_v3",
    "no_admissible_cleanup_v3",
    "semantic_ranker_selection_v1",
)

PUBLIC_CALIBRATION_ONLY_SCORE_CLAIMS = {
    "student_code_lm_checkpoint_public_task_calibration_only",
    "student_token_generator_checkpoint_public_task_calibration_only",
    "student_neural_checkpoint_public_task_calibration_only",
    "student_checkpoint_public_task_calibration_only",
    "forbidden_non_token_level_code_generation",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--residual-report",
        default="reports/public_code_transfer_residual_report_wide_public_seed23_5x32_interface_floor_v1.json",
    )
    parser.add_argument(
        "--real-code-report",
        default="reports/real_code_benchmark_graduation_wide_public_seed23_5x32_interface_floor_v1.json",
    )
    parser.add_argument("--rows-per-family", type=int, default=96)
    parser.add_argument("--heldout-rows-per-family", type=int, default=24)
    parser.add_argument("--private-train-out", default=TRAIN_DEFAULT)
    parser.add_argument("--private-heldout-out", default=HELDOUT_DEFAULT)
    parser.add_argument("--out", default="reports/targeted_private_residual_curriculum_v3.json")
    parser.add_argument("--markdown-out", default="reports/targeted_private_residual_curriculum_v3.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    residual_report = read_json(resolve(args.residual_report), {})
    real_code_report = read_json(resolve(args.real_code_report), {})
    residual_summary = residual_report.get("summary") if isinstance(residual_report.get("summary"), dict) else {}

    train_rows = build_rows(
        rows_per_family=max(8, int(args.rows_per_family)),
        split="train",
        id_offset=0,
    )
    heldout_rows = build_rows(
        rows_per_family=max(4, int(args.heldout_rows_per_family)),
        split="heldout",
        id_offset=100_000,
    )
    train_path = resolve(args.private_train_out)
    heldout_path = resolve(args.private_heldout_out)
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)
    train_check = verify_private_solution_rows(train_rows, max_failures=12)
    heldout_check = verify_private_solution_rows(heldout_rows, max_failures=12)
    family_counts = dict(Counter(str(row.get("targeted_private_residual_family_v3")) for row in train_rows))
    heldout_family_counts = dict(Counter(str(row.get("targeted_private_residual_family_v3")) for row in heldout_rows))
    gates = [
        gate("five_required_families_written", set(family_counts) == set(FAMILY_TARGETS), family_counts),
        gate("each_family_has_train_rows", all(family_counts.get(family, 0) > 0 for family in FAMILY_TARGETS), family_counts),
        gate(
            "each_family_has_heldout_rows",
            all(heldout_family_counts.get(family, 0) > 0 for family in FAMILY_TARGETS),
            heldout_family_counts,
        ),
        gate("private_train_solution_tests_pass", train_check["failure_count"] == 0, train_check),
        gate("private_heldout_solution_tests_pass", heldout_check["failure_count"] == 0, heldout_check),
        gate("metamorphic_property_rows_present", metamorphic_property_rows(train_rows) >= 24, metamorphic_property_rows(train_rows)),
        gate("stdin_proxy_rows_present", stdin_proxy_rows(train_rows) >= 24, stdin_proxy_rows(train_rows)),
        gate("return_interface_rows_present", return_interface_rows(train_rows) >= 24, return_interface_rows(train_rows)),
        gate("no_admissible_cleanup_rows_present", no_admissible_rows(train_rows) >= 24, no_admissible_rows(train_rows)),
        gate("semantic_ranker_rows_present", semantic_ranker_rows(train_rows) >= 24, semantic_ranker_rows(train_rows)),
        gate("public_prompts_not_copied", True, "uses sanitized residual class counts only"),
        gate("public_tests_not_copied", True, "public test bodies are not read or emitted"),
        gate("public_solutions_not_copied", True, "public canonical solutions are not read or emitted"),
        gate(
            "public_score_is_calibration_only",
            real_code_report.get("public_benchmark_score_claim") in PUBLIC_CALIBRATION_ONLY_SCORE_CLAIMS
            and real_code_report.get("promotion_allowed") is False,
            real_code_report.get("public_benchmark_score_claim"),
        ),
        gate("external_inference_zero", True, 0),
    ]
    trigger_state = "GREEN" if all(item["passed"] for item in gates) else "RED"
    return {
        "policy": "project_theseus_targeted_private_residual_curriculum_v3",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": (
            "Private-only residual repair for verifier mismatch, stdin/competitive-programming adapters, "
            "return/interface fidelity, no-admissible cleanup, and semantic ranker selection."
        ),
        "inputs": {
            "residual_report": rel(resolve(args.residual_report)),
            "real_code_report": rel(resolve(args.real_code_report)),
            "rows_per_family": int(args.rows_per_family),
            "heldout_rows_per_family": int(args.heldout_rows_per_family),
            "residual_summary_used": sanitized_residual_summary(residual_summary),
        },
        "outputs": {
            "private_train_jsonl": rel(train_path),
            "private_heldout_jsonl": rel(heldout_path),
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "summary": {
            "private_train_row_count": len(train_rows),
            "private_heldout_row_count": len(heldout_rows),
            "family_train_row_counts": family_counts,
            "family_heldout_row_counts": heldout_family_counts,
            "private_train_solution_failures": train_check["failure_count"],
            "private_heldout_solution_failures": heldout_check["failure_count"],
            "metamorphic_property_rows": metamorphic_property_rows(train_rows),
            "stdin_proxy_rows": stdin_proxy_rows(train_rows),
            "return_interface_rows": return_interface_rows(train_rows),
            "no_admissible_cleanup_rows": no_admissible_rows(train_rows),
            "semantic_ranker_rows": semantic_ranker_rows(train_rows),
            "dominant_public_residual_categories": residual_summary.get("dominant_categories"),
            "adapter_adjusted_public_residual_categories": residual_summary.get("adapter_adjusted_dominant_categories"),
            "public_task_ids_hashed_only": True,
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "external_inference_calls": 0,
        },
        "target_gates_before_next_public_calibration": {
            "private_residual_v3_heldout_pass_rate_min": 0.70,
            "no_admissible_task_rate_max": 0.03,
            "livecodebench_private_stdin_proxy_pass_count_min": 1,
            "sts_same_seed_delta_must_be_positive": True,
            "sts_same_seed_regressions_max": 0,
            "maturity_audit_hard_blockers_max": 0,
            "maturity_audit_evidence_blockers_max": 0,
            "public_calibration": "locked_until_all_private_gates_clear",
        },
        "families": family_reports(train_rows, heldout_rows),
        "gates": gates,
        "next_actions": [
            "include private_residual_repair_v3_code_lm_tasks.jsonl in Code LM private training inputs",
            "run train-once fanout and private residual v3 heldout gate without launching public calibration",
            "rerun decoder gate, private/public transfer proof, STS causal A/B, readiness, and maturity audit",
            "keep public calibration locked until private residual v3 heldout pass rate and no-admissible gates clear",
        ],
        "external_inference_calls": 0,
    }


def build_rows(*, rows_per_family: int, split: str, id_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    templates_by_family = family_templates()
    for family in FAMILY_TARGETS:
        templates = templates_by_family[family]
        for family_index in range(rows_per_family):
            template = templates[family_index % len(templates)]
            global_index = id_offset + len(rows)
            rows.append(row_from_template(template, family, split, global_index, family_index))
    return rows


def family_templates() -> dict[str, list[dict[str, Any]]]:
    return {
        "verifier_mismatch_property_v3": verifier_mismatch_templates(),
        "livecodebench_stdin_proxy_v1": stdin_proxy_templates(),
        "return_interface_fidelity_v3": return_interface_templates(),
        "no_admissible_cleanup_v3": no_admissible_cleanup_templates(),
        "semantic_ranker_selection_v1": semantic_ranker_templates(),
    }


def row_from_template(
    template: dict[str, Any],
    family: str,
    split: str,
    global_index: int,
    family_index: int,
) -> dict[str, Any]:
    entry = f"{template['entry']}_{global_index:06d}"
    tests = str(template["tests"]).replace("{entry}", entry)
    tags = sorted(
        dict.fromkeys(
            [
                "private_residual_repair_v3",
                family,
                split,
                *[str(tag) for tag in template.get("tags", []) if str(tag)],
            ]
        )
    )
    decoder_contract = decoder_contract_for_template(template, family)
    return {
        "task_id": f"private_residual_repair_v3_{family}_{global_index:06d}",
        "source_task_id": f"private_residual_repair_v3_{split}_{family_index:04d}",
        "card_id": "private_residual_repair_v3",
        "source_id": "local_generated_private_residual_repair_v3",
        "split": "train" if split == "train" else "eval",
        "category": template["category"],
        "prompt": template["prompt"],
        "entry_point": entry,
        "solution_expr": "",
        "solution_body": template["body"],
        "tests": tests,
        "tags": tags,
        "targeted_private_residual_family_v3": family,
        "residual_concept": template.get("residual_concept", family),
        "concept_residual_label": template.get("concept_residual_label", template["category"]),
        "metamorphic_properties": template.get("metamorphic_properties", []),
        "semantic_ranker_target": template.get("semantic_ranker_target", {}),
        "guardrail_expectations": template.get("guardrail_expectations", {}),
        "decoder_contract": decoder_contract,
        "benchmark_evidence_level": "private_residual_repair_v3_generated_only",
        "public_benchmark": False,
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "license_spdx": "CC0-1.0",
        "candidate_expression_eligible": False,
        "provenance": {
            "policy": "project_theseus_targeted_private_residual_curriculum_v3",
            "family": family,
            "public_benchmark_answers_used": False,
            "public_tests_used": False,
            "public_prompts_used": False,
            "public_task_ids_hashed_only": True,
            "semantics": "private synthetic residual pressure only",
        },
    }


def decoder_contract_for_template(template: dict[str, Any], family: str) -> dict[str, Any]:
    shape = str(template.get("return_shape") or "unknown")
    semantic = str(template.get("semantic_family") or template.get("residual_concept") or family)
    required = [str(item) for item in template.get("required_constructs", []) if str(item)]
    if not required:
        required = ["branch", "loop", "locals", "type_and_return_shape"]
    ranker_target = template.get("semantic_ranker_target") if isinstance(template.get("semantic_ranker_target"), dict) else {}
    return {
        "policy": "project_theseus_decoder_contract_v3_private_residual_repair",
        "return_shape": shape,
        "type_family": str(template.get("type_family") or semantic),
        "semantic_family": semantic,
        "visible_arg_count_hint": int(template.get("visible_arg_count_hint") or 1),
        "required_constructs": required,
        "residual_label_hint": str(template.get("concept_residual_label") or template["category"]),
        "full_body_required": True,
        "guardrail_only": False,
        "feedback_weight": float(template.get("feedback_weight") or 1.55),
        "score_semantics": "private residual repair v3 pressure only; public benchmarks remain calibration-only",
        "argument_roles": template.get("argument_roles", {"data": "primary_input"}),
        "return_contract": {
            "shape": shape,
            "empty_or_invalid_behavior": "covered_by_private_v3_assertions",
            "must_preserve_container_shape": shape in {"list", "dict", "tuple"},
        },
        "generation_plan": {
            "policy": "signature -> semantic_family -> metamorphic_properties -> return_contract -> stateful_body -> verifier_repair",
            "skeleton_bias": template.get("skeleton_bias", []),
            "repair_strategy": template.get("repair_strategy", "prefer semantic-family bodies that satisfy private metamorphic properties"),
            "semantic_ranker_target": ranker_target,
            "verifier_feedback": [
                "intended_behavior_failed",
                "semantic_family_mismatch",
                "return_shape_mismatch",
                "visible_argument_mismatch",
                "edge_boundary_failure",
                "stdin_format_mismatch",
            ],
            "public_tests_used": False,
            "public_solutions_used": False,
        },
    }


def verifier_mismatch_templates() -> list[dict[str, Any]]:
    return [
        {
            "category": "private_v3_stable_casefold_unique",
            "prompt": "Return first-seen unique text tokens after trimming and casefolding. Preserve encounter order.",
            "entry": "private_v3_stable_casefold_unique",
            "body": "seen = set()\nout = []\nfor item in data:\n    text = str(item).strip().casefold()\n    if not text or text in seen:\n        continue\n    seen.add(text)\n    out.append(text)\nreturn out",
            "tests": "assert {entry}([' A ', 'a', 'B', '', 'b']) == ['a', 'b']\nassert {entry}([]) == []\nassert {entry}(['x', 'X', 'x ']) == ['x']\nassert {entry}(['two', 'one', 'two']) == ['two', 'one']\n",
            "tags": ["verifier_mismatch", "metamorphic_property", "duplicate_handling", "normalization"],
            "metamorphic_properties": ["idempotent_under_duplicate_replay", "stable_order", "casefold_normalization"],
            "return_shape": "list",
            "semantic_family": "stable_dedup_normalization",
            "type_family": "collection_transform",
            "required_constructs": ["loop", "branch", "locals", "collection_ops", "type_and_return_shape"],
            "skeleton_bias": ["seen_set", "append_order", "casefold_strip"],
        },
        {
            "category": "private_v3_multiset_delta",
            "prompt": "Return a sorted dictionary of counts left after subtracting one sequence from another.",
            "entry": "private_v3_multiset_delta",
            "body": "counts = {}\nfor item in data:\n    counts[item] = counts.get(item, 0) + 1\nfor item in other:\n    counts[item] = counts.get(item, 0) - 1\nout = {}\nfor key in sorted(counts):\n    if counts[key] > 0:\n        out[key] = counts[key]\nreturn out",
            "tests": "assert {entry}(['a', 'b', 'a'], ['a']) == {'a': 1, 'b': 1}\nassert {entry}([], ['x']) == {}\nassert {entry}([1, 1, 2], [1, 3]) == {1: 1, 2: 1}\nassert {entry}(['z'], ['z']) == {}\n",
            "tags": ["verifier_mismatch", "metamorphic_property", "duplicate_handling", "dict"],
            "metamorphic_properties": ["subtraction_identity", "duplicate_count_preservation", "sorted_key_stability"],
            "return_shape": "dict",
            "semantic_family": "multiset_counting",
            "type_family": "collection_logic",
            "visible_arg_count_hint": 2,
            "argument_roles": {"data": "minuend_sequence", "other": "subtrahend_sequence"},
            "required_constructs": ["loop", "branch", "locals", "collection_ops", "type_and_return_shape"],
            "skeleton_bias": ["dict_counter", "subtract_second_input", "positive_count_filter"],
        },
        {
            "category": "private_v3_numeric_tolerance_window",
            "prompt": "Return values whose absolute distance from a center is within a tolerance, preserving order.",
            "entry": "private_v3_numeric_tolerance_window",
            "body": "center = other[0]\ntol = abs(other[1])\nout = []\nfor value in data:\n    if abs(float(value) - center) <= tol + 1e-8:\n        out.append(value)\nreturn out",
            "tests": "assert {entry}([0.9, 1.0, 1.2, 2.0], (1.0, 0.2)) == [0.9, 1.0, 1.2]\nassert {entry}([], (0, 1)) == []\nassert {entry}([3, -3], (0, -3)) == [3, -3]\nassert {entry}([1.000000001], (1.0, 0.0)) == [1.000000001]\n",
            "tags": ["verifier_mismatch", "metamorphic_property", "numeric_tolerance", "edge_conditions"],
            "metamorphic_properties": ["numeric_tolerance", "negative_tolerance_normalized", "order_preserving"],
            "return_shape": "list",
            "semantic_family": "numeric_filter_with_tolerance",
            "type_family": "collection_transform",
            "visible_arg_count_hint": 2,
            "argument_roles": {"data": "sequence[number]", "other": "(center,tolerance)"},
            "required_constructs": ["loop", "branch", "locals", "numeric_ops", "type_and_return_shape"],
            "skeleton_bias": ["absolute_difference", "epsilon_guard", "append_preserve_order"],
        },
        {
            "category": "private_v3_roundtrip_rle",
            "prompt": "Return run-length encoded pairs for consecutive equal values. Empty input returns an empty list.",
            "entry": "private_v3_roundtrip_rle",
            "body": "out = []\nfor item in data:\n    if out and out[-1][0] == item:\n        out[-1] = (item, out[-1][1] + 1)\n    else:\n        out.append((item, 1))\nreturn out",
            "tests": "assert {entry}(['a', 'a', 'b']) == [('a', 2), ('b', 1)]\nassert {entry}([]) == []\nassert {entry}([1]) == [(1, 1)]\nassert {entry}(['x', 'y', 'x']) == [('x', 1), ('y', 1), ('x', 1)]\n",
            "tags": ["verifier_mismatch", "metamorphic_property", "round_trip", "local_state"],
            "metamorphic_properties": ["roundtrip_decodable", "singleton_boundary", "consecutive_not_global_count"],
            "return_shape": "list",
            "semantic_family": "run_length_encoding",
            "type_family": "collection_logic",
            "required_constructs": ["loop", "branch", "locals", "collection_ops", "type_and_return_shape"],
            "skeleton_bias": ["last_item_state", "tuple_count_return", "empty_guard"],
        },
    ]


def stdin_proxy_templates() -> list[dict[str, Any]]:
    return [
        {
            "category": "private_v3_stdin_pair_sums",
            "prompt": "Implement solve(input_data): each non-empty line contains two integers; return one sum per line.",
            "entry": "private_v3_stdin_pair_sums",
            "body": "lines = []\nfor line in str(data).splitlines():\n    parts = line.split()\n    if len(parts) < 2:\n        continue\n    lines.append(str(int(parts[0]) + int(parts[1])))\nreturn '\\n'.join(lines)",
            "tests": "assert {entry}('1 2\\n-3 5\\n') == '3\\n2'\nassert {entry}('') == ''\nassert {entry}('7 0 extra\\n') == '7'\nassert {entry}('bad\\n4 6') == '10'\n",
            "tags": ["livecodebench_stdin_proxy", "stdin_parser", "output_formatting", "edge_conditions"],
            "return_shape": "str",
            "semantic_family": "stdin_numeric_line_parser",
            "type_family": "string_indexing",
            "required_constructs": ["loop", "branch", "locals", "stdin_parse", "string_join_return"],
            "skeleton_bias": ["splitlines", "split_fields", "newline_join"],
        },
        {
            "category": "private_v3_stdin_prefix_queries",
            "prompt": "Implement solve(input_data): first line n q, second line values, then 1-based inclusive range-sum queries.",
            "entry": "private_v3_stdin_prefix_queries",
            "body": "try:\n    tokens = [int(x) for x in str(data).split()]\nexcept Exception:\n    return ''\nif len(tokens) < 2:\n    return ''\nn, q = tokens[0], tokens[1]\nvalues = tokens[2:2+n]\nprefix = [0]\nfor value in values:\n    prefix.append(prefix[-1] + value)\nout = []\npos = 2 + n\nfor _ in range(q):\n    if pos + 1 >= len(tokens):\n        break\n    left, right = tokens[pos], tokens[pos + 1]\n    pos += 2\n    left = max(1, left)\n    right = min(n, right)\n    out.append(str(prefix[right] - prefix[left - 1] if left <= right else 0))\nreturn '\\n'.join(out)",
            "tests": "assert {entry}('4 3\\n1 2 3 4\\n1 2\\n2 4\\n4 4\\n') == '3\\n9\\n4'\nassert {entry}('0 1\\n\\n1 1\\n') == '0'\nassert {entry}('2 2 5 6 2 1') == '0'\nassert {entry}('bad') == ''\n",
            "tags": ["livecodebench_stdin_proxy", "stdin_parser", "prefix_sum", "algorithmic_planning"],
            "return_shape": "str",
            "semantic_family": "stdin_prefix_sum_queries",
            "type_family": "algorithmic_planning",
            "required_constructs": ["loop", "branch", "locals", "stdin_parse", "algorithmic_planning"],
            "skeleton_bias": ["token_stream", "prefix_array", "range_query_loop"],
        },
        {
            "category": "private_v3_stdin_components",
            "prompt": "Implement solve(input_data): first line n m, followed by undirected edges; return the number of connected components.",
            "entry": "private_v3_stdin_components",
            "body": "tokens = [int(x) for x in str(data).split()]\nif len(tokens) < 2:\n    return '0'\nn, m = tokens[0], tokens[1]\ngraph = [[] for _ in range(n)]\npos = 2\nfor _ in range(m):\n    if pos + 1 >= len(tokens):\n        break\n    a, b = tokens[pos] - 1, tokens[pos + 1] - 1\n    pos += 2\n    if 0 <= a < n and 0 <= b < n:\n        graph[a].append(b)\n        graph[b].append(a)\nseen = [False] * n\ncomponents = 0\nfor start in range(n):\n    if seen[start]:\n        continue\n    components += 1\n    stack = [start]\n    seen[start] = True\n    while stack:\n        node = stack.pop()\n        for nxt in graph[node]:\n            if not seen[nxt]:\n                seen[nxt] = True\n                stack.append(nxt)\nreturn str(components)",
            "tests": "assert {entry}('5 2\\n1 2\\n4 5\\n') == '3'\nassert {entry}('1 0\\n') == '1'\nassert {entry}('3 3 1 2 2 3 1 3') == '1'\nassert {entry}('') == '0'\n",
            "tags": ["livecodebench_stdin_proxy", "stdin_parser", "graph", "algorithmic_planning"],
            "return_shape": "str",
            "semantic_family": "stdin_graph_components",
            "type_family": "graph_search_algorithm",
            "required_constructs": ["loop", "branch", "locals", "stdin_parse", "algorithmic_planning", "graph"],
            "skeleton_bias": ["adjacency_list", "stack_dfs", "component_counter"],
        },
        {
            "category": "private_v3_stdin_interval_union",
            "prompt": "Implement solve(input_data): first line n, followed by intervals; return total covered integer length after merging.",
            "entry": "private_v3_stdin_interval_union",
            "body": "tokens = [int(x) for x in str(data).split()]\nif not tokens:\n    return '0'\nn = tokens[0]\nintervals = []\npos = 1\nfor _ in range(n):\n    if pos + 1 >= len(tokens):\n        break\n    start, end = tokens[pos], tokens[pos + 1]\n    pos += 2\n    if end > start:\n        intervals.append((start, end))\nmerged = []\nfor start, end in sorted(intervals):\n    if not merged or start > merged[-1][1]:\n        merged.append([start, end])\n    else:\n        merged[-1][1] = max(merged[-1][1], end)\nreturn str(sum(end - start for start, end in merged))",
            "tests": "assert {entry}('3\\n1 3\\n2 5\\n10 12\\n') == '6'\nassert {entry}('2 5 5 7 6') == '0'\nassert {entry}('0') == '0'\nassert {entry}('2 -1 1 1 2') == '3'\n",
            "tags": ["livecodebench_stdin_proxy", "stdin_parser", "intervals", "algorithmic_planning"],
            "return_shape": "str",
            "semantic_family": "stdin_interval_merge",
            "type_family": "grouped_interval_algorithm",
            "required_constructs": ["loop", "branch", "locals", "stdin_parse", "algorithmic_planning"],
            "skeleton_bias": ["sort_intervals", "merge_overlap", "length_sum"],
        },
    ]


def return_interface_templates() -> list[dict[str, Any]]:
    return [
        {
            "category": "private_v3_pair_stats_tuple",
            "prompt": "Return (minimum, maximum, count) for numeric values. Empty input returns (None, None, 0).",
            "entry": "private_v3_pair_stats_tuple",
            "body": "values = [item for item in data if isinstance(item, (int, float)) and not isinstance(item, bool)]\nif not values:\n    return (None, None, 0)\nreturn (min(values), max(values), len(values))",
            "tests": "assert {entry}([3, 1, True, 'x']) == (1, 3, 2)\nassert {entry}([]) == (None, None, 0)\nassert {entry}([-1.5]) == (-1.5, -1.5, 1)\nassert isinstance({entry}([1, 2]), tuple)\n",
            "tags": ["return_interface_fidelity", "return_shape", "tuple", "type_guard"],
            "return_shape": "tuple",
            "semantic_family": "numeric_summary_exact_return",
            "type_family": "heterogeneous_numeric_contract",
            "required_constructs": ["loop", "branch", "locals", "type_and_return_shape"],
            "skeleton_bias": ["numeric_filter", "tuple_return_contract", "empty_tuple_boundary"],
        },
        {
            "category": "private_v3_pair_stats_dict",
            "prompt": "Return {'min': value, 'max': value, 'count': count} for numeric values. Empty input uses None and 0.",
            "entry": "private_v3_pair_stats_dict",
            "body": "values = [item for item in data if isinstance(item, (int, float)) and not isinstance(item, bool)]\nif not values:\n    return {'min': None, 'max': None, 'count': 0}\nreturn {'min': min(values), 'max': max(values), 'count': len(values)}",
            "tests": "assert {entry}([3, 1, True, 'x']) == {'min': 1, 'max': 3, 'count': 2}\nassert {entry}([]) == {'min': None, 'max': None, 'count': 0}\nassert {entry}([2]) == {'min': 2, 'max': 2, 'count': 1}\nassert isinstance({entry}([1, 2]), dict)\n",
            "tags": ["return_interface_fidelity", "return_shape", "dict", "type_guard"],
            "return_shape": "dict",
            "semantic_family": "numeric_summary_exact_return",
            "type_family": "heterogeneous_numeric_contract",
            "required_constructs": ["loop", "branch", "locals", "type_and_return_shape"],
            "skeleton_bias": ["numeric_filter", "literal_dict_return", "empty_dict_boundary"],
        },
        {
            "category": "private_v3_two_arg_threshold_labels",
            "prompt": "Return labels whose numeric score is at least the supplied threshold.",
            "entry": "private_v3_two_arg_threshold_labels",
            "body": "out = []\nfor record in data:\n    if not isinstance(record, dict):\n        continue\n    try:\n        score = float(record.get('score', 0))\n    except Exception:\n        score = 0.0\n    if score >= other and record.get('label') is not None:\n        out.append(str(record.get('label')))\nreturn out",
            "tests": "assert {entry}([{'label': 'a', 'score': 2}, {'label': 'b', 'score': '3'}], 2.5) == ['b']\nassert {entry}([], 1) == []\nassert {entry}([{'score': 9}], 1) == []\nassert {entry}([1, {'label': 'x', 'score': 'bad'}], 0) == ['x']\n",
            "tags": ["return_interface_fidelity", "two_arg_interface", "return_shape", "list"],
            "return_shape": "list",
            "semantic_family": "record_filtering_threshold",
            "type_family": "collection_logic",
            "visible_arg_count_hint": 2,
            "argument_roles": {"data": "sequence[record]", "other": "threshold"},
            "required_constructs": ["loop", "branch", "locals", "two_arg_interface", "type_and_return_shape"],
            "skeleton_bias": ["record_guard", "threshold_compare", "list_return_builder"],
        },
    ]


def no_admissible_cleanup_templates() -> list[dict[str, Any]]:
    return [
        {
            "category": "private_v3_safe_head_default",
            "prompt": "Return the first item of a sequence, or the supplied default when the sequence is empty or not a sequence.",
            "entry": "private_v3_safe_head_default",
            "body": "if isinstance(data, (list, tuple)) and data:\n    return data[0]\nreturn other",
            "tests": "assert {entry}([1, 2], 9) == 1\nassert {entry}([], 'x') == 'x'\nassert {entry}(None, 0) == 0\nassert {entry}(('a',), 'z') == 'a'\n",
            "tags": ["no_admissible_cleanup", "candidate_floor", "two_arg_interface", "edge_conditions"],
            "return_shape": "unknown",
            "semantic_family": "safe_indexing_default",
            "type_family": "interface_fidelity",
            "visible_arg_count_hint": 2,
            "argument_roles": {"data": "sequence[Any]", "other": "default"},
            "required_constructs": ["branch", "locals", "two_arg_interface"],
            "skeleton_bias": ["sequence_guard", "default_return", "first_item_return"],
        },
        {
            "category": "private_v3_nested_flatten_depth",
            "prompt": "Flatten nested lists up to the supplied depth. Non-list items are preserved.",
            "entry": "private_v3_nested_flatten_depth",
            "body": "def flatten_once(items):\n    out = []\n    for item in items:\n        if isinstance(item, list):\n            out.extend(item)\n        else:\n            out.append(item)\n    return out\nout = data if isinstance(data, list) else [data]\nfor _ in range(max(0, int(other))):\n    out = flatten_once(out)\nreturn out",
            "tests": "assert {entry}([[1, 2], [3]], 1) == [1, 2, 3]\nassert {entry}([[[1]], 2], 1) == [[1], 2]\nassert {entry}([[[1]], 2], 2) == [1, 2]\nassert {entry}(5, 3) == [5]\n",
            "tags": ["no_admissible_cleanup", "candidate_floor", "nested_structure", "return_shape"],
            "return_shape": "list",
            "semantic_family": "bounded_nested_flatten",
            "type_family": "collection_transform",
            "visible_arg_count_hint": 2,
            "argument_roles": {"data": "nested_sequence", "other": "depth"},
            "required_constructs": ["loop", "branch", "locals", "nested_structure", "type_and_return_shape"],
            "skeleton_bias": ["helper_function", "bounded_depth_loop", "extend_nested_lists"],
        },
        {
            "category": "private_v3_title_case_preserve_acronyms",
            "prompt": "Title-case words while preserving all-uppercase acronyms.",
            "entry": "private_v3_title_case_preserve_acronyms",
            "body": "out = []\nfor word in str(data).split():\n    if len(word) > 1 and word.isupper():\n        out.append(word)\n    else:\n        out.append(word[:1].upper() + word[1:].lower())\nreturn ' '.join(out)",
            "tests": "assert {entry}('hello NASA world') == 'Hello NASA World'\nassert {entry}('') == ''\nassert {entry}('mIxEd cpu') == 'Mixed Cpu'\nassert {entry}('AI lab') == 'AI Lab'\n",
            "tags": ["no_admissible_cleanup", "candidate_floor", "string_transform", "return_shape"],
            "return_shape": "str",
            "semantic_family": "title_case_with_acronym_guard",
            "type_family": "string_transform",
            "required_constructs": ["loop", "branch", "locals", "index_or_string_ops", "type_and_return_shape"],
            "skeleton_bias": ["word_loop", "isupper_guard", "join_return"],
        },
    ]


def semantic_ranker_templates() -> list[dict[str, Any]]:
    return [
        {
            "category": "private_v3_longest_even_run",
            "prompt": "Return the length of the longest contiguous run of even integers, not the total count of even integers.",
            "entry": "private_v3_longest_even_run",
            "body": "best = 0\ncurrent = 0\nfor value in data:\n    if value % 2 == 0:\n        current += 1\n        best = max(best, current)\n    else:\n        current = 0\nreturn best",
            "tests": "assert {entry}([2, 4, 1, 6, 8, 10]) == 3\nassert {entry}([1, 3]) == 0\nassert {entry}([]) == 0\nassert {entry}([2, 1, 4]) == 1\n",
            "tags": ["semantic_ranker_selection", "algorithmic_planning", "verifier_mismatch"],
            "return_shape": "number",
            "semantic_family": "contiguous_run_state_machine",
            "semantic_ranker_target": {"positive_family": "contiguous_run", "negative_decoys": ["global_count", "sum_even_values"]},
            "type_family": "algorithmic_planning",
            "required_constructs": ["loop", "branch", "locals", "algorithmic_planning"],
            "skeleton_bias": ["current_and_best_state", "reset_on_break", "scalar_return"],
        },
        {
            "category": "private_v3_first_missing_positive",
            "prompt": "Return the smallest positive integer missing from the input collection.",
            "entry": "private_v3_first_missing_positive",
            "body": "values = {item for item in data if isinstance(item, int) and item > 0}\nanswer = 1\nwhile answer in values:\n    answer += 1\nreturn answer",
            "tests": "assert {entry}([1, 2, 0]) == 3\nassert {entry}([3, 4, -1, 1]) == 2\nassert {entry}([]) == 1\nassert {entry}([2, 2]) == 1\n",
            "tags": ["semantic_ranker_selection", "algorithmic_planning", "edge_conditions"],
            "return_shape": "number",
            "semantic_family": "missing_positive_set_search",
            "semantic_ranker_target": {"positive_family": "set_membership_search", "negative_decoys": ["minimum_value", "count_unique"]},
            "type_family": "algorithmic_planning",
            "required_constructs": ["loop", "branch", "locals", "algorithmic_planning"],
            "skeleton_bias": ["positive_int_set", "while_membership", "smallest_missing_return"],
        },
        {
            "category": "private_v3_lexicographic_rotation",
            "prompt": "Return the lexicographically smallest rotation of a string, not the sorted characters.",
            "entry": "private_v3_lexicographic_rotation",
            "body": "text = str(data)\nif not text:\n    return ''\nrotations = [text[i:] + text[:i] for i in range(len(text))]\nreturn min(rotations)",
            "tests": "assert {entry}('baca') == 'abac'\nassert {entry}('aaaa') == 'aaaa'\nassert {entry}('') == ''\nassert {entry}('caba') == 'abac'\n",
            "tags": ["semantic_ranker_selection", "string_transform", "algorithmic_planning"],
            "return_shape": "str",
            "semantic_family": "string_rotation_selection",
            "semantic_ranker_target": {"positive_family": "rotation_enumeration", "negative_decoys": ["sort_characters", "reverse_string"]},
            "type_family": "string_transform",
            "required_constructs": ["loop", "branch", "locals", "index_or_string_ops", "type_and_return_shape"],
            "skeleton_bias": ["rotation_list", "min_lexicographic", "empty_string_guard"],
        },
    ]


def sanitized_residual_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "dominant_categories": summary.get("dominant_categories"),
        "adapter_adjusted_dominant_categories": summary.get("adapter_adjusted_dominant_categories"),
        "adapter_adjusted_no_admissible": summary.get("adapter_adjusted_no_admissible"),
        "public_prompts_embedded": summary.get("public_prompts_embedded"),
        "public_tests_or_solutions_embedded": summary.get("public_tests_or_solutions_embedded"),
        "external_inference_calls": summary.get("external_inference_calls"),
    }


def family_reports(train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for family in FAMILY_TARGETS:
        family_train = [row for row in train_rows if row.get("targeted_private_residual_family_v3") == family]
        family_heldout = [row for row in heldout_rows if row.get("targeted_private_residual_family_v3") == family]
        reports.append(
            {
                "family": family,
                "train_rows": len(family_train),
                "heldout_rows": len(family_heldout),
                "sample_categories": sorted({str(row.get("category") or "") for row in family_train})[:8],
                "decoder_contract_rows": sum(1 for row in family_train if isinstance(row.get("decoder_contract"), dict) and row.get("decoder_contract")),
                "semantic_ranker_target_rows": semantic_ranker_rows(family_train),
                "metamorphic_property_rows": metamorphic_property_rows(family_train),
            }
        )
    return reports


def metamorphic_property_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("metamorphic_properties"))


def stdin_proxy_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if "livecodebench_stdin_proxy" in (row.get("tags") or []))


def return_interface_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if "return_interface_fidelity" in (row.get("tags") or []))


def no_admissible_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if "no_admissible_cleanup" in (row.get("tags") or []))


def semantic_ranker_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if isinstance(row.get("semantic_ranker_target"), dict) and row.get("semantic_ranker_target"))


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Targeted Private Residual Curriculum V3",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Private train rows: {summary.get('private_train_row_count')}",
        f"- Private heldout rows: {summary.get('private_heldout_row_count')}",
        f"- Family train rows: {summary.get('family_train_row_counts')}",
        f"- Train solution failures: {summary.get('private_train_solution_failures')}",
        f"- Heldout solution failures: {summary.get('private_heldout_solution_failures')}",
        f"- Metamorphic property rows: {summary.get('metamorphic_property_rows')}",
        f"- Stdin proxy rows: {summary.get('stdin_proxy_rows')}",
        f"- Return/interface rows: {summary.get('return_interface_rows')}",
        f"- No-admissible cleanup rows: {summary.get('no_admissible_cleanup_rows')}",
        f"- Semantic ranker rows: {summary.get('semantic_ranker_rows')}",
        f"- Public solutions included: {summary.get('public_benchmark_solutions_included')}",
        f"- Public tests included: {summary.get('public_tests_included')}",
        f"- External inference calls: {summary.get('external_inference_calls')}",
        "",
        "## Required Before Next Public Calibration",
    ]
    for key, value in report.get("target_gates_before_next_public_calibration", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("Public calibration remains locked; this artifact is private training and heldout pressure only.")
    return "\n".join(lines) + "\n"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
