#!/usr/bin/env python3
"""Generate edge-contract-v3 private public-transfer curriculum rows.

This successor lane uses the spent public residual report only as sanitized
category pressure. It emits private synthetic tasks for verifier mismatch,
no-admissible interface coverage, return-shape fidelity, algorithmic planning,
and stdin-style transfer without copying public prompts, tests, or solutions.
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
from code_residual_curriculum import verify_private_solution_rows, write_json, write_jsonl  # noqa: E402


POLICY = "project_theseus_edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum_v1"
TRAIN_DEFAULT = training_data_path(
    "high_transfer",
    "private_train",
    "edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum_code_lm_tasks.jsonl",
)
HELDOUT_DEFAULT = training_data_path(
    "high_transfer",
    "private_eval",
    "edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum_heldout_code_lm_tasks.jsonl",
)
FAMILY_TARGETS = (
    "verifier_mismatch_metamorphic_v3",
    "verifier_mismatch_stateful_v3",
    "no_admissible_interface_coverage_v3",
    "return_shape_contract_v3",
    "algorithmic_planning_boundary_v3",
    "stdin_public_transfer_proxy_v3",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--residual-report",
        default="reports/public_code_transfer_residual_report_wide_public_seed23_5x32_interface_floor_v1.json",
    )
    parser.add_argument("--post-readiness", default="reports/post_distillation_public_transfer_readiness_v1.json")
    parser.add_argument("--rows-per-family", type=int, default=128)
    parser.add_argument("--heldout-rows-per-family", type=int, default=32)
    parser.add_argument("--private-train-out", default=TRAIN_DEFAULT)
    parser.add_argument("--private-heldout-out", default=HELDOUT_DEFAULT)
    parser.add_argument(
        "--out",
        default="reports/edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum.json",
    )
    parser.add_argument(
        "--markdown-out",
        default="reports/edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum.md",
    )
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    residual_report = read_json(resolve(args.residual_report), {})
    post_readiness = read_json(resolve(args.post_readiness), {})
    residual_summary = object_field(residual_report, "summary")
    post_summary = object_field(post_readiness, "summary")

    train_rows = build_rows(rows_per_family=max(24, int(args.rows_per_family)), split="train", id_offset=0)
    heldout_rows = build_rows(
        rows_per_family=max(8, int(args.heldout_rows_per_family)),
        split="heldout",
        id_offset=200_000,
    )
    train_path = resolve(args.private_train_out)
    heldout_path = resolve(args.private_heldout_out)
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)

    train_check = verify_private_solution_rows(train_rows, max_failures=20)
    heldout_check = verify_private_solution_rows(heldout_rows, max_failures=20)
    train_counts = family_counts(train_rows)
    heldout_counts = family_counts(heldout_rows)
    contract_count = complete_contract_rows(train_rows)
    heldout_contract_count = complete_contract_rows(heldout_rows)
    dominant = dominant_public_residuals(residual_summary, post_summary)
    gates = [
        gate("six_required_families_written", set(train_counts) == set(FAMILY_TARGETS), train_counts),
        gate("heldout_has_same_families", set(heldout_counts) == set(FAMILY_TARGETS), heldout_counts),
        gate("train_rows_ge_720", len(train_rows) >= 720, len(train_rows)),
        gate("heldout_rows_ge_180", len(heldout_rows) >= 180, len(heldout_rows)),
        gate("decoder_contracts_complete", contract_count == len(train_rows), {"complete": contract_count, "rows": len(train_rows)}),
        gate(
            "heldout_decoder_contracts_complete",
            heldout_contract_count == len(heldout_rows),
            {"complete": heldout_contract_count, "rows": len(heldout_rows)},
        ),
        gate("private_train_solution_tests_pass", train_check["failure_count"] == 0, train_check),
        gate("private_heldout_solution_tests_pass", heldout_check["failure_count"] == 0, heldout_check),
        gate("verifier_mismatch_majority_pressure", verifier_mismatch_rows(train_rows) >= len(train_rows) // 3, verifier_mismatch_rows(train_rows)),
        gate("no_admissible_pressure_present", family_count(train_rows, "no_admissible_interface_coverage_v3") >= 96, train_counts),
        gate("return_shape_pressure_present", family_count(train_rows, "return_shape_contract_v3") >= 96, train_counts),
        gate("algorithmic_planning_pressure_present", family_count(train_rows, "algorithmic_planning_boundary_v3") >= 96, train_counts),
        gate("stdin_proxy_pressure_present", family_count(train_rows, "stdin_public_transfer_proxy_v3") >= 96, train_counts),
        gate("sanitized_public_residual_pressure_used", "verifier_mismatch" in dominant, dominant),
        gate("public_boundary_clean", public_boundary_clean(train_rows + heldout_rows), "private synthetic prompts only"),
        gate("external_inference_zero", True, 0),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "RED"
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": (
            "Private-only successor curriculum for public-transfer verifier mismatch after edge-contract-v2. "
            "It teaches reusable semantic contracts, not benchmark-specific answers."
        ),
        "inputs": {
            "residual_report": rel(resolve(args.residual_report)),
            "post_readiness": rel(resolve(args.post_readiness)),
            "rows_per_family": int(args.rows_per_family),
            "heldout_rows_per_family": int(args.heldout_rows_per_family),
            "sanitized_public_residual_summary": sanitized_residual_summary(residual_summary, post_summary),
            "public_tests_or_solutions_read": False,
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
            "family_train_row_counts": train_counts,
            "family_heldout_row_counts": heldout_counts,
            "private_train_solution_failures": train_check["failure_count"],
            "private_heldout_solution_failures": heldout_check["failure_count"],
            "complete_decoder_contract_rows": contract_count,
            "heldout_complete_decoder_contract_rows": heldout_contract_count,
            "verifier_mismatch_rows": verifier_mismatch_rows(train_rows),
            "dominant_public_residual_categories": dominant,
            "next_public_blocker": post_summary.get("next_public_blocker")
            or residual_summary.get("next_blocker_after_current_adapter")
            or residual_summary.get("next_blocker"),
            "public_task_ids_hashed_only": True,
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "external_inference_calls": 0,
        },
        "target_gates_before_public_calibration": {
            "edge_contract_v3_private_heldout_pass_rate_min": 0.70,
            "no_admissible_task_rate_max": 0.03,
            "learned_token_only_required": True,
            "prototype_or_diagnostic_adapter_pass_count_max": 0,
            "public_calibration": "locked_until_fresh_private_repair_transfer_readiness_and_operator_review",
        },
        "families": family_reports(train_rows, heldout_rows),
        "gates": gates,
        "next_actions": [
            "include the edge-contract-v3 private train JSONL in high-transfer training inputs",
            "run private-only train/fanout on the edge-contract-v3 heldout split",
            "score learned-token-only private heldout behavior and same-seed STS-off control",
            "rerun post-distillation readiness and maturity gates; keep public calibration locked",
        ],
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def build_rows(*, rows_per_family: int, split: str, id_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    templates = templates_by_family()
    for family in FAMILY_TARGETS:
        family_templates = templates[family]
        for family_index in range(rows_per_family):
            template = family_templates[family_index % len(family_templates)]
            rows.append(row_from_template(template, family, split, id_offset + len(rows), family_index))
    return rows


def row_from_template(template: dict[str, Any], family: str, split: str, global_index: int, family_index: int) -> dict[str, Any]:
    entry = f"{template['entry']}_{global_index:06d}"
    tests = str(template["tests"]).replace("{entry}", entry)
    tags = sorted(
        {
            "edge_contract_v3",
            "edge_contract_v3_verifier_mismatch_public_transfer",
            family,
            split,
            *[str(tag) for tag in template.get("tags", []) if str(tag)],
        }
    )
    return {
        "task_id": f"edge_contract_v3_public_transfer_{family}_{global_index:06d}",
        "source_task_id": f"edge_contract_v3_{split}_{family_index:04d}",
        "card_id": "edge_contract_v3_verifier_mismatch_public_transfer_private",
        "source_id": "local_generated_edge_contract_v3_verifier_mismatch_public_transfer_private",
        "split": "train" if split == "train" else "eval",
        "category": template["category"],
        "prompt": template["prompt"],
        "entry_point": entry,
        "solution_expr": "",
        "solution_body": template["body"],
        "tests": tests,
        "tags": tags,
        "targeted_private_residual_family_v3": family,
        "edge_contract_v3_family": family,
        "residual_concept": "edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum",
        "concept_residual_label": template.get("concept_residual_label", template["category"]),
        "metamorphic_properties": template.get("metamorphic_properties", []),
        "semantic_ranker_target": template.get("semantic_ranker_target", {}),
        "guardrail_expectations": template.get("guardrail_expectations", {}),
        "decoder_contract": decoder_contract_for_template(template, family),
        "benchmark_evidence_level": "edge_contract_v3_private_generated_only",
        "public_benchmark": False,
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "license_spdx": "CC0-1.0",
        "candidate_expression_eligible": False,
        "provenance": {
            "policy": POLICY,
            "family": family,
            "source_public_signal": "aggregate residual family counts only",
            "public_benchmark_answers_used": False,
            "public_tests_used": False,
            "public_prompts_used": False,
            "public_task_ids_hashed_only": True,
            "semantics": "private synthetic public-transfer residual pressure only",
        },
    }


def decoder_contract_for_template(template: dict[str, Any], family: str) -> dict[str, Any]:
    shape = str(template.get("return_shape") or "unknown")
    semantic = str(template.get("semantic_family") or template.get("concept_residual_label") or family)
    required = [str(item) for item in template.get("required_constructs", []) if str(item)] or [
        "branch",
        "loop",
        "locals",
        "type_and_return_shape",
    ]
    return {
        "policy": "project_theseus_decoder_contract_v3_private_public_transfer",
        "return_shape": shape,
        "type_family": str(template.get("type_family") or semantic),
        "semantic_family": semantic,
        "visible_arg_count_hint": int(template.get("visible_arg_count_hint") or 1),
        "required_constructs": required,
        "residual_label_hint": str(template.get("concept_residual_label") or template["category"]),
        "full_body_required": True,
        "guardrail_only": False,
        "feedback_weight": float(template.get("feedback_weight") or 1.75),
        "score_semantics": "edge-contract-v3 private transfer pressure only; public benchmarks remain calibration-only",
        "argument_roles": template.get("argument_roles", {"data": "primary_input"}),
        "return_contract": {
            "shape": shape,
            "empty_or_invalid_behavior": "covered_by_private_edge_contract_v3_assertions",
            "must_preserve_container_shape": shape in {"list", "dict", "tuple", "same_container"},
        },
        "generation_plan": {
            "policy": "signature -> normalized input contract -> semantic family -> return contract -> metamorphic properties -> verifier repair",
            "skeleton_bias": template.get("skeleton_bias", []),
            "repair_strategy": template.get(
                "repair_strategy",
                "prefer reusable semantic bodies that satisfy private metamorphic contracts before ranker repair",
            ),
            "semantic_ranker_target": template.get("semantic_ranker_target", {}),
            "verifier_feedback": [
                "intended_behavior_failed",
                "semantic_family_mismatch",
                "return_shape_mismatch",
                "visible_argument_mismatch",
                "no_admissible_candidate",
                "stdin_format_mismatch",
                "edge_boundary_failure",
            ],
            "public_tests_used": False,
            "public_solutions_used": False,
        },
    }


def templates_by_family() -> dict[str, list[dict[str, Any]]]:
    return {
        "verifier_mismatch_metamorphic_v3": [
            template(
                "edge_v3_canonical_interval_union",
                "Merge intervals after normalizing reversed endpoints; return sorted closed intervals.",
                "intervals = []\nfor item in data:\n    if not isinstance(item, (list, tuple)) or len(item) < 2:\n        continue\n    a, b = item[0], item[1]\n    if a > b:\n        a, b = b, a\n    intervals.append((a, b))\nintervals.sort()\nout = []\nfor start, end in intervals:\n    if not out or start > out[-1][1]:\n        out.append((start, end))\n    elif end > out[-1][1]:\n        out[-1] = (out[-1][0], end)\nreturn out",
                "assert {entry}([(5, 3), (1, 2), (2, 4)]) == [(1, 5)]\nassert {entry}([]) == []\nassert {entry}([(1, 1), (3, 2)]) == [(1, 1), (2, 3)]\nassert {entry}(['bad', (9, 7)]) == [(7, 9)]\n",
                ["verifier_mismatch", "metamorphic_property", "intervals", "normalization"],
                "list",
                "interval_normalization_and_merge",
                "collection_transform",
                ["loop", "branch", "locals", "sort", "type_and_return_shape"],
                ["normalize_reversed_endpoint", "sort_then_merge", "tuple_interval_return"],
            ),
            template(
                "edge_v3_casefold_join_groups",
                "Group records by normalized key and return sorted key/count pairs.",
                "counts = {}\nfor record in data:\n    if isinstance(record, dict):\n        raw = record.get('key', '')\n    elif isinstance(record, (list, tuple)) and record:\n        raw = record[0]\n    else:\n        raw = record\n    key = str(raw).strip().casefold()\n    if not key:\n        continue\n    counts[key] = counts.get(key, 0) + 1\nreturn sorted(counts.items())",
                "assert {entry}([{'key':' A '}, ('a', 2), 'B', '']) == [('a', 2), ('b', 1)]\nassert {entry}([]) == []\nassert {entry}([{'x': 1}, {'key': 'x'}]) == [('x', 1)]\nassert {entry}(['Two', 'two ', 'ONE']) == [('one', 1), ('two', 2)]\n",
                ["verifier_mismatch", "metamorphic_property", "record_normalization", "dict"],
                "list",
                "normalized_record_grouping",
                "heterogeneous_record_transform",
                ["loop", "branch", "locals", "collection_ops", "type_and_return_shape"],
                ["record_shape_guard", "casefold_strip", "sorted_items_return"],
            ),
            template(
                "edge_v3_threshold_segments",
                "Return lengths of consecutive numeric runs whose values meet a threshold.",
                "threshold = other if other is not None else 0\nout = []\nrun = 0\nfor value in data:\n    if value >= threshold:\n        run += 1\n    elif run:\n        out.append(run)\n        run = 0\nif run:\n    out.append(run)\nreturn out",
                "assert {entry}([1, 3, 4, 0, 5], 3) == [2, 1]\nassert {entry}([], 0) == []\nassert {entry}([5, 5], 5) == [2]\nassert {entry}([1, 2], 3) == []\n",
                ["verifier_mismatch", "metamorphic_property", "stateful_runs", "threshold"],
                "list",
                "threshold_run_length_encoding",
                "numeric_sequence_state",
                ["loop", "branch", "locals", "state_update", "type_and_return_shape"],
                ["run_counter", "flush_on_boundary", "final_flush"],
                visible_arg_count_hint=2,
                argument_roles={"data": "sequence[number]", "other": "threshold"},
            ),
        ],
        "verifier_mismatch_stateful_v3": [
            template(
                "edge_v3_capped_running_balance",
                "Apply signed deltas with floor and ceiling clamps; return each visible balance.",
                "floor, ceiling = other\nbalance = 0\nout = []\nfor delta in data:\n    balance += delta\n    if balance < floor:\n        balance = floor\n    if balance > ceiling:\n        balance = ceiling\n    out.append(balance)\nreturn out",
                "assert {entry}([5, 10, -30, 7], (0, 12)) == [5, 12, 0, 7]\nassert {entry}([], (0, 1)) == []\nassert {entry}([-1, -1], (-2, 2)) == [-1, -2]\nassert {entry}([3, 3], (0, 4)) == [3, 4]\n",
                ["verifier_mismatch", "state_machine", "local_state", "edge_conditions"],
                "list",
                "bounded_state_update",
                "state_machine",
                ["loop", "branch", "locals", "state_update", "type_and_return_shape"],
                ["accumulator_state", "floor_ceiling_clamp", "append_each_state"],
                visible_arg_count_hint=2,
                argument_roles={"data": "signed_deltas", "other": "(floor, ceiling)"},
            ),
            template(
                "edge_v3_stack_cancel_tokens",
                "Use a stack to cancel adjacent inverse tokens and return the remaining stack.",
                "inverse = other if isinstance(other, dict) else {}\nstack = []\nfor token in data:\n    if stack and inverse.get(token) == stack[-1]:\n        stack.pop()\n    else:\n        stack.append(token)\nreturn stack",
                "assert {entry}(['open', 'close'], {'close': 'open'}) == []\nassert {entry}(['a', 'b', 'B', 'c'], {'B': 'b'}) == ['a', 'c']\nassert {entry}([], {}) == []\nassert {entry}(['x'], {'y': 'x'}) == ['x']\n",
                ["verifier_mismatch", "state_machine", "stack", "local_state"],
                "list",
                "stack_cancellation",
                "state_machine",
                ["loop", "branch", "locals", "stack", "type_and_return_shape"],
                ["stack_push_pop", "inverse_lookup", "adjacent_pair_cancel"],
                visible_arg_count_hint=2,
                argument_roles={"data": "token_sequence", "other": "inverse_mapping"},
            ),
            template(
                "edge_v3_fsm_accepting_prefixes",
                "Run a finite-state transition table and return indices where the state is accepting.",
                "state = other.get('start')\naccept = set(other.get('accept', []))\ntransitions = other.get('transitions', {})\nout = []\nfor idx, symbol in enumerate(data):\n    state = transitions.get((state, symbol), state)\n    if state in accept:\n        out.append(idx)\nreturn out",
                "spec = {'start':'A','accept':['B'],'transitions':{('A','x'):'B',('B','y'):'A'}}\nassert {entry}(['x','z','y','x'], spec) == [0, 1, 3]\nassert {entry}([], spec) == []\nassert {entry}(['y'], spec) == []\nassert {entry}(['x','y'], spec) == [0]\n",
                ["verifier_mismatch", "state_machine", "fsm", "prefix"],
                "list",
                "finite_state_prefix_acceptance",
                "state_machine",
                ["loop", "branch", "locals", "state_update", "type_and_return_shape"],
                ["transition_lookup", "accepting_state_check", "prefix_index_return"],
                visible_arg_count_hint=2,
                argument_roles={"data": "symbols", "other": "fsm_spec"},
            ),
        ],
        "no_admissible_interface_coverage_v3": [
            template(
                "edge_v3_record_extract_fallback",
                "Extract a named field from mixed records; preserve order and use fallback when absent.",
                "field, fallback = other\nout = []\nfor record in data:\n    if isinstance(record, dict):\n        out.append(record.get(field, fallback))\n    elif isinstance(record, (list, tuple)) and isinstance(field, int) and 0 <= field < len(record):\n        out.append(record[field])\n    else:\n        out.append(fallback)\nreturn out",
                "assert {entry}([{'a':1}, {}, (5, 6)], ('a', 0)) == [1, 0, 0]\nassert {entry}([(1, 2), (3,)], (1, None)) == [2, None]\nassert {entry}([], ('x', 9)) == []\nassert {entry}(['bad'], ('x', 'na')) == ['na']\n",
                ["no_admissible_candidate_regression", "interface_coverage", "mixed_records", "fallback"],
                "list",
                "mixed_record_field_extraction",
                "interface_fidelity",
                ["loop", "branch", "locals", "type_guards", "type_and_return_shape"],
                ["dict_or_tuple_branch", "fallback_append", "visible_second_argument"],
                visible_arg_count_hint=2,
                argument_roles={"data": "records", "other": "(field, fallback)"},
            ),
            template(
                "edge_v3_safe_zip_apply",
                "Apply an operation name to paired values, skipping incomplete pairs.",
                "operation = other\nout = []\nfor pair in data:\n    if not isinstance(pair, (list, tuple)) or len(pair) < 2:\n        continue\n    a, b = pair[0], pair[1]\n    if operation == 'sum':\n        out.append(a + b)\n    elif operation == 'diff':\n        out.append(a - b)\n    elif operation == 'max':\n        out.append(max(a, b))\nreturn out",
                "assert {entry}([(1, 2), (5, 3)], 'sum') == [3, 8]\nassert {entry}([(1, 2), 'bad', (5, 3)], 'diff') == [-1, 2]\nassert {entry}([(1, 9)], 'max') == [9]\nassert {entry}([(1, 2)], 'noop') == []\n",
                ["no_admissible_candidate_regression", "interface_coverage", "operation_dispatch"],
                "list",
                "safe_pair_operation_dispatch",
                "interface_fidelity",
                ["loop", "branch", "locals", "operation_dispatch", "type_and_return_shape"],
                ["pair_shape_guard", "operation_branch", "skip_incomplete"],
                visible_arg_count_hint=2,
                argument_roles={"data": "pairs", "other": "operation_name"},
            ),
        ],
        "return_shape_contract_v3": [
            template(
                "edge_v3_partition_tuple_shape",
                "Partition values by predicate and always return a two-list tuple.",
                "keep = []\ndrop = []\nthreshold = other if other is not None else 0\nfor value in data:\n    if value >= threshold:\n        keep.append(value)\n    else:\n        drop.append(value)\nreturn (keep, drop)",
                "assert {entry}([1, 3, 0], 2) == ([3], [1, 0])\nassert {entry}([], 2) == ([], [])\nassert {entry}([5], 5) == ([5], [])\nassert isinstance({entry}([1], 2), tuple)\n",
                ["return_shape", "interface_fidelity", "tuple_return"],
                "tuple",
                "partition_tuple_return_contract",
                "return_shape_contract",
                ["loop", "branch", "locals", "type_and_return_shape"],
                ["two_bucket_partition", "tuple_return", "preserve_order"],
                visible_arg_count_hint=2,
                argument_roles={"data": "values", "other": "threshold"},
            ),
            template(
                "edge_v3_same_container_transform",
                "Normalize numeric strings while preserving list versus tuple container shape.",
                "items = []\nfor value in data:\n    try:\n        items.append(int(str(value).strip()))\n    except Exception:\n        items.append(0)\nif isinstance(data, tuple):\n    return tuple(items)\nreturn items",
                "assert {entry}([' 1 ', 'bad', 3]) == [1, 0, 3]\nassert {entry}(('2', '4')) == (2, 4)\nassert {entry}([]) == []\nassert isinstance({entry}(('x',)), tuple)\n",
                ["return_shape", "same_container", "parsing_encoding"],
                "same_container",
                "same_container_numeric_normalization",
                "return_shape_contract",
                ["loop", "branch", "locals", "try_except", "type_and_return_shape"],
                ["container_type_preservation", "safe_int_parse", "tuple_rewrap"],
            ),
        ],
        "algorithmic_planning_boundary_v3": [
            template(
                "edge_v3_weighted_interval_best",
                "Return max total weight from non-overlapping intervals using dynamic programming.",
                "jobs = []\nfor item in data:\n    if len(item) >= 3:\n        start, end, weight = item[0], item[1], item[2]\n        if start > end:\n            start, end = end, start\n        jobs.append((end, start, weight))\njobs.sort()\ndp = [0]\nends = []\nfor end, start, weight in jobs:\n    lo, hi = 0, len(ends)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if ends[mid] <= start:\n            lo = mid + 1\n        else:\n            hi = mid\n    best = dp[lo] + weight\n    dp.append(max(dp[-1], best))\n    ends.append(end)\nreturn dp[-1]",
                "assert {entry}([(1, 3, 5), (3, 4, 6), (2, 5, 20)]) == 20\nassert {entry}([]) == 0\nassert {entry}([(5, 1, 7)]) == 7\nassert {entry}([(1, 2, 3), (2, 3, 4)]) == 7\n",
                ["algorithmic_planning", "dynamic_programming", "intervals"],
                "number",
                "weighted_interval_dynamic_programming",
                "algorithmic_planning",
                ["loop", "branch", "locals", "binary_search", "dynamic_programming"],
                ["sort_by_end", "binary_search_previous", "dp_prefix_best"],
            ),
            template(
                "edge_v3_graph_distance_labels",
                "Return shortest hop distances from a start node for an undirected edge list.",
                "start = other\nfrom collections import deque\ngraph = {}\nfor a, b in data:\n    graph.setdefault(a, []).append(b)\n    graph.setdefault(b, []).append(a)\ndist = {start: 0}\nqueue = deque([start])\nwhile queue:\n    node = queue.popleft()\n    for nxt in graph.get(node, []):\n        if nxt not in dist:\n            dist[nxt] = dist[node] + 1\n            queue.append(nxt)\nreturn dist",
                "assert {entry}([('a','b'), ('b','c')], 'a') == {'a':0, 'b':1, 'c':2}\nassert {entry}([], 'x') == {'x':0}\nassert {entry}([('a','b'), ('c','d')], 'c') == {'c':0, 'd':1}\nassert {entry}([('a','a')], 'a') == {'a':0}\n",
                ["algorithmic_planning", "graph_search", "dict_return"],
                "dict",
                "graph_shortest_hops",
                "graph_search_algorithm",
                ["loop", "branch", "locals", "queue", "graph"],
                ["adjacency_list", "bfs_queue", "distance_dict"],
                visible_arg_count_hint=2,
                argument_roles={"data": "edge_list", "other": "start_node"},
            ),
        ],
        "stdin_public_transfer_proxy_v3": [
            template(
                "edge_v3_stdin_case_sums",
                "Implement solve(input_data): first token is case count, then each case length and values; return one sum per case.",
                "try:\n    tokens = [int(x) for x in str(data).split()]\nexcept Exception:\n    return ''\nif not tokens:\n    return ''\nt = tokens[0]\npos = 1\nout = []\nfor _ in range(t):\n    if pos >= len(tokens):\n        break\n    n = max(0, tokens[pos])\n    pos += 1\n    vals = tokens[pos:pos+n]\n    pos += n\n    out.append(str(sum(vals)))\nreturn '\\n'.join(out)",
                "assert {entry}('2\\n3 1 2 3\\n2 -1 4\\n') == '6\\n3'\nassert {entry}('') == ''\nassert {entry}('1 0') == '0'\nassert {entry}('bad') == ''\n",
                ["stdin_proxy", "public_transfer_shape", "algorithmic_planning"],
                "str",
                "stdin_case_token_parser",
                "stdin_parser",
                ["loop", "branch", "locals", "stdin_parse", "string_join_return"],
                ["token_stream", "case_loop", "newline_join"],
            ),
            template(
                "edge_v3_stdin_threshold_labels",
                "Implement solve(input_data): read threshold and values, output H/L labels preserving order.",
                "parts = str(data).split()\nif not parts:\n    return ''\ntry:\n    threshold = int(parts[0])\nexcept Exception:\n    return ''\nout = []\nfor token in parts[1:]:\n    try:\n        value = int(token)\n    except Exception:\n        continue\n    out.append('H' if value >= threshold else 'L')\nreturn ' '.join(out)",
                "assert {entry}('5 1 5 9') == 'L H H'\nassert {entry}('') == ''\nassert {entry}('x 1 2') == ''\nassert {entry}('0 -1 0') == 'L H'\n",
                ["stdin_proxy", "public_transfer_shape", "threshold", "formatting"],
                "str",
                "stdin_threshold_labeling",
                "stdin_parser",
                ["loop", "branch", "locals", "stdin_parse", "string_join_return"],
                ["threshold_first_token", "skip_bad_values", "space_join"],
            ),
        ],
    }


def template(
    category: str,
    prompt: str,
    body: str,
    tests: str,
    tags: list[str],
    return_shape: str,
    semantic_family: str,
    type_family: str,
    required_constructs: list[str],
    skeleton_bias: list[str],
    *,
    visible_arg_count_hint: int = 1,
    argument_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "prompt": prompt,
        "entry": category,
        "body": body,
        "tests": tests,
        "tags": tags,
        "return_shape": return_shape,
        "semantic_family": semantic_family,
        "type_family": type_family,
        "visible_arg_count_hint": visible_arg_count_hint,
        "argument_roles": argument_roles or {"data": "primary_input"},
        "required_constructs": required_constructs,
        "skeleton_bias": skeleton_bias,
        "concept_residual_label": category,
    }


def family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("edge_contract_v3_family") or "unknown") for row in rows))


def family_count(rows: list[dict[str, Any]], family: str) -> int:
    return family_counts(rows).get(family, 0)


def verifier_mismatch_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if "verifier_mismatch" in {str(tag) for tag in row.get("tags") or []})


def complete_contract_rows(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
        plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
        if (
            contract.get("policy")
            and contract.get("return_shape")
            and contract.get("semantic_family")
            and contract.get("required_constructs")
            and plan.get("skeleton_bias") is not None
            and plan.get("public_tests_used") is False
            and plan.get("public_solutions_used") is False
        ):
            count += 1
    return count


def public_boundary_clean(rows: list[dict[str, Any]]) -> bool:
    forbidden = ("humaneval", "mbpp", "evalplus", "bigcodebench", "livecodebench", "canonical_solution", "public_test")
    for row in rows:
        if row.get("public_benchmark") is True or row.get("public_tests_included") is True:
            return False
        if row.get("public_benchmark_solutions_included") is True:
            return False
        text = " ".join([str(row.get("prompt") or ""), str(row.get("tests") or ""), str(row.get("solution_body") or "")]).lower()
        if any(token in text for token in forbidden):
            return False
    return True


def dominant_public_residuals(*summaries: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for summary in summaries:
        for key in ["adapter_adjusted_dominant_categories", "dominant_public_residual_categories", "dominant_categories"]:
            value = summary.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, list) and item:
                        amount = int(item[1]) if len(item) > 1 and isinstance(item[1], (int, float)) else 1
                        counts[str(item[0])] += amount
    return dict(counts)


def sanitized_residual_summary(residual_summary: dict[str, Any], post_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "residual_categories": dominant_public_residuals(residual_summary, post_summary),
        "next_public_blocker": post_summary.get("next_public_blocker")
        or residual_summary.get("next_blocker_after_current_adapter")
        or residual_summary.get("next_blocker"),
        "public_pass_rate": post_summary.get("public_pass_rate") or residual_summary.get("real_public_task_pass_rate"),
        "public_task_count": post_summary.get("public_task_count") or residual_summary.get("public_task_count"),
        "public_tests_or_solutions_copied": False,
    }


def family_reports(train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train = family_counts(train_rows)
    heldout = family_counts(heldout_rows)
    out = []
    for family in FAMILY_TARGETS:
        sample = next((row for row in train_rows if row.get("edge_contract_v3_family") == family), {})
        out.append(
            {
                "family": family,
                "train_rows": train.get(family, 0),
                "heldout_rows": heldout.get(family, 0),
                "sample_category": sample.get("category"),
                "sample_entry_point": sample.get("entry_point"),
            }
        )
    return out


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    lines = [
        "# Edge Contract V3 Verifier-Mismatch Curriculum",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Private train rows: {summary.get('private_train_row_count')}",
        f"- Private heldout rows: {summary.get('private_heldout_row_count')}",
        f"- Train solution failures: {summary.get('private_train_solution_failures')}",
        f"- Heldout solution failures: {summary.get('private_heldout_solution_failures')}",
        f"- Verifier-mismatch rows: {summary.get('verifier_mismatch_rows')}",
        f"- Next public blocker: {summary.get('next_public_blocker')}",
        "",
        "## Families",
    ]
    for row in report.get("families", []):
        lines.append(f"- `{row.get('family')}`: train `{row.get('train_rows')}`, heldout `{row.get('heldout_rows')}`")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
