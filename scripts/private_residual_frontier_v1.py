#!/usr/bin/env python3
"""Private residual frontier gate.

The previous private ratchet exhausted its queue while the spent public score
remained far below floor. This gate opens a new public-safe lane: use only the
aggregate public residual categories to choose harder private residual
compositions, then test whether STS-conditioned learned token behavior can
solve them without diagnostic adapters, prototypes, public prompts, public
tests, public solutions, or external inference.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from broad_private_novel_composition_gate_v1 import private_train_bodies, render_solution_body  # noqa: E402
from code_residual_curriculum import verify_private_solution_rows  # noqa: E402
from theseus_archive_resolver import read_jsonl_follow_pointer  # noqa: E402


REPORTS = ROOT / "reports"
HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_residual_frontier_v1_heldout_code_lm_tasks.jsonl"
EMPTY_PUBLIC = REPORTS / "code_lm_public_tasks_private_residual_frontier_v1_empty.jsonl"
EMPTY_STS = REPORTS / "private_residual_frontier_v1_empty_sts_streams.jsonl"
STS_STREAMS = REPORTS / "private_residual_frontier_v1_private_safe_sts_streams.jsonl"
STS_STREAMS_REPORT = REPORTS / "private_residual_frontier_v1_private_safe_sts_streams.json"
SHARD_DIR = REPORTS / "private_residual_frontier_v1_shards"
PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_private_residual_frontier_v1.jsonl"
PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_private_residual_frontier_v1_empty_public.jsonl"
CONTROL_PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_private_residual_frontier_v1_sts_off.jsonl"
CONTROL_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_private_residual_frontier_v1_sts_off_empty_public.jsonl"
FRONTIER_CANDIDATES = REPORTS / "code_lm_private_candidates_private_residual_frontier_v1_frontier_only.jsonl"
FANOUT_REPORT = REPORTS / "private_residual_frontier_v1_fanout.json"
CONTROL_FANOUT_REPORT = REPORTS / "private_residual_frontier_v1_sts_off_fanout.json"
SCORE = REPORTS / "private_residual_frontier_v1_score.json"
SCORE_MD = REPORTS / "private_residual_frontier_v1_score.md"
FRONTIER_SCORE = REPORTS / "private_residual_frontier_v1_frontier_only_score.json"
FRONTIER_SCORE_MD = REPORTS / "private_residual_frontier_v1_frontier_only_score.md"
DEFAULT_OUT = REPORTS / "private_residual_frontier_v1.json"
DEFAULT_MD = REPORTS / "private_residual_frontier_v1.md"
DEFAULT_QUEUE = REPORTS / "private_residual_frontier_v1_queue.jsonl"

PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"
READINESS_PACKET = REPORTS / "public_calibration_readiness_packet.json"
OPERATOR_DRY_RUN = REPORTS / "operator_bounded_public_calibration_dry_run.json"
OPERATOR_EXECUTE = REPORTS / "operator_bounded_public_calibration_execute.json"
OPERATOR_APPROVAL = REPORTS / "public_calibration_operator_approval_post_v4_seed23_5x32.json"
PUBLIC_RESIDUAL = REPORTS / "public_code_transfer_residual_report_wide_public_seed23_5x32_interface_floor_v1.json"
DEFAULT_CHECKPOINT = REPORTS / "student_code_lm_checkpoint_private_residual_repair_v3_private_proof.json"
RELEASE = ROOT / "target" / "release" / ("symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli")

FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS = [
    REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json",
    REPORTS / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl",
    REPORTS / "student_code_candidates_post_v4_seed23_5x32.jsonl",
    REPORTS / "operator_bounded_public_calibration_post_v4_seed23_5x32.json",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rows", type=int, default=672)
    parser.add_argument("--min-rows", type=int, default=672)
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument("--candidates-per-task", type=int, default=3)
    parser.add_argument("--score-timeout-seconds", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=105)
    parser.add_argument("--floor", type=float, default=0.70)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--checkpoint-in", default=rel(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--queue-out", default=rel(DEFAULT_QUEUE))
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.queue_out), queue_rows(report, args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    bodies = private_train_bodies()
    heldout_report = write_frontier_heldout(HELDOUT, bodies, rows=max(1, int(args.rows)), seed=int(args.seed))
    ensure_sidecars()
    preflight = preflight_report(args, heldout_report)
    commands: list[dict[str, Any]] = []

    if args.execute and preflight["ready"]:
        deadline = started + max(0.1, float(args.max_hours)) * 3600.0
        commands = run_execution_phases(args, deadline)

    candidates = read_jsonl(PRIVATE_CANDIDATES) if args.execute and PRIVATE_CANDIDATES.exists() else []
    frontier_candidates = [row for row in candidates if frontier_token_candidate(row)]
    if args.execute:
        write_jsonl(FRONTIER_CANDIDATES, frontier_candidates)
    score = read_json(SCORE, {}) if args.execute else {}
    frontier_score = read_json(FRONTIER_SCORE, {}) if args.execute else {}
    score_summary = object_field(score, "summary")
    frontier_summary = object_field(frontier_score, "summary")
    inventory = candidate_inventory(candidates, frontier_candidates)
    pass_inventory = pass_inventory_summary(score, candidates)
    floor = float(args.floor)
    pass_rate = first_number(score_summary.get("pass_rate"), 0.0)
    frontier_rate = first_number(frontier_summary.get("pass_rate"), 0.0)
    public_residual = read_json(PUBLIC_RESIDUAL, {})
    residual_summary = object_field(public_residual, "summary")
    leakage = public_leakage_scan(read_jsonl(HELDOUT))

    gates = [
        gate("public_calibration_operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK), "hard"),
        gate(
            "post_v4_public_artifacts_approved_or_absent",
            post_v4_public_artifact_state()["allowed"],
            post_v4_public_artifact_state(),
            "hard",
        ),
        gate(
            "public_residual_report_is_aggregate_only",
            public_residual.get("trigger_state") == "GREEN"
            and residual_summary.get("public_tests_or_solutions_embedded") is False
            and residual_summary.get("public_prompts_embedded") is False,
            {
                "trigger_state": public_residual.get("trigger_state"),
                "public_tests_or_solutions_embedded": residual_summary.get("public_tests_or_solutions_embedded"),
                "public_prompts_embedded": residual_summary.get("public_prompts_embedded"),
                "dominant_categories": residual_summary.get("adapter_adjusted_dominant_categories") or residual_summary.get("dominant_categories"),
            },
            "hard",
        ),
        gate("private_frontier_rows_ge_minimum", heldout_report["row_count"] >= int(args.min_rows), heldout_report, "warning"),
        gate("private_frontier_spec_diversity_ge_26", heldout_report["frontier_spec_count"] >= 26, heldout_report, "warning"),
        gate("private_solution_tests_pass", heldout_report["private_solution_failure_count"] == 0, heldout_report, "hard"),
        gate("preflight_ready", preflight["ready"], preflight, "hard"),
        gate("execute_requested", bool(args.execute), bool(args.execute), "warning"),
        gate("fanout_commands_succeeded", commands_succeeded(commands, ["build_private_safe_sts_streams", "fanout_sts_on", "fanout_sts_off_control"]), command_evidence(commands), "warning"),
        gate("score_commands_succeeded", commands_succeeded(commands, ["score_all_candidates", "score_frontier_only"]), command_evidence(commands), "warning"),
        gate("pass_rate_floor", pass_rate >= floor, {"observed": pass_rate, "minimum": floor}, "warning"),
        gate("frontier_only_pass_rate_floor", frontier_rate >= floor, {"observed": frontier_rate, "minimum": floor}, "warning"),
        gate("frontier_token_passes_present", pass_inventory["frontier_token_pass_count"] > 0, pass_inventory, "warning"),
        gate("diagnostic_adapter_pass_count_zero", pass_inventory["diagnostic_adapter_pass_count"] == 0, pass_inventory, "warning"),
        gate("prototype_pass_count_zero", pass_inventory["prototype_pass_count"] == 0, pass_inventory, "warning"),
        gate("sts_control_lower_than_frontier", first_number(score_summary.get("control_pass_rate"), 0.0) < pass_rate, score_summary, "warning"),
        gate("public_candidate_manifests_empty", file_empty(PUBLIC_CANDIDATES) and file_empty(CONTROL_PUBLIC_CANDIDATES) and file_empty(EMPTY_PUBLIC), {
            "public_candidates": file_size(PUBLIC_CANDIDATES),
            "control_public_candidates": file_size(CONTROL_PUBLIC_CANDIDATES),
            "public_manifest": file_size(EMPTY_PUBLIC),
        }, "hard"),
        gate("public_data_leakage_zero", leakage["hit_count"] == 0, leakage, "hard"),
        gate("external_inference_zero", top_external_calls(score) + top_external_calls(frontier_score) == 0, {
            "score": top_external_calls(score),
            "frontier_score": top_external_calls(frontier_score),
        }, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    failed = [row for row in gates if not row["passed"]]
    trigger_state = "RED" if hard_failed else "YELLOW" if failed else "GREEN"
    return {
        "policy": "project_theseus_private_residual_frontier_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "completion_evidence_status": "private_residual_frontier_ready" if trigger_state == "GREEN" else "private_residual_frontier_pending",
            "row_count": heldout_report["row_count"],
            "frontier_spec_count": heldout_report["frontier_spec_count"],
            "dominant_public_residual_categories": residual_summary.get("adapter_adjusted_dominant_categories") or residual_summary.get("dominant_categories") or [],
            "candidate_row_count": len(candidates),
            "frontier_candidate_row_count": len(frontier_candidates),
            "pass_count": score_summary.get("pass_count"),
            "pass_rate": score_summary.get("pass_rate"),
            "frontier_only_pass_count": frontier_summary.get("pass_count"),
            "frontier_only_pass_rate": frontier_summary.get("pass_rate"),
            "control_pass_rate": score_summary.get("control_pass_rate"),
            "frontier_token_pass_count": pass_inventory["frontier_token_pass_count"],
            "diagnostic_adapter_pass_count": pass_inventory["diagnostic_adapter_pass_count"],
            "prototype_pass_count": pass_inventory["prototype_pass_count"],
            "hard_failed_gate_count": len(hard_failed),
            "failed_gate_count": len(failed),
            "elapsed_seconds": round(time.time() - started, 3),
            "public_calibration_allowed": False,
            "public_tests_used": False,
            "public_solutions_used": False,
            "external_inference_calls": 0,
            "score_semantics": "private residual frontier only; not promotion evidence and not public calibration",
        },
        "inputs": {
            "execute": bool(args.execute),
            "rows": int(args.rows),
            "min_rows": int(args.min_rows),
            "seed": int(args.seed),
            "candidates_per_task": int(args.candidates_per_task),
            "shard_size": int(args.shard_size),
            "public_residual": rel(PUBLIC_RESIDUAL),
            "aggregate_public_residual_categories_only": True,
            "public_calibration": "locked",
        },
        "heldout_report": heldout_report,
        "preflight": preflight,
        "candidate_inventory": inventory,
        "pass_inventory": pass_inventory,
        "gates": gates,
        "blockers": failed,
        "commands": commands,
        "artifacts": artifacts(),
        "next_actions": next_actions(trigger_state, failed, pass_rate, frontier_rate, floor),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def frontier_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "stdin_pair_sums_then_parse_then_gcd_positive",
            "steps": ["bpg_stdin_pair_sums", "bpg_parse_signed_ints", "bpg_gcd_positive"],
            "prompt": "Sum integer pairs from a stdin-style string, parse the sums, then return their positive gcd.",
            "cases": [("12 18\n5 7\nbad\n-2 4", None, 2), ("3 9\n10 5", None, 3)],
            "return_shape": "number",
            "type_family": "algorithmic_planning",
            "required_constructs": ["loop", "branch", "locals", "composition", "stdin_parser"],
            "visible_arg_count_hint": 1,
        },
        {
            "id": "threshold_labels_then_stable_dedup_then_top_k",
            "steps": ["bpg_threshold_labels", "bpg_stable_dedup", "bpg_top_k_frequent"],
            "prompt": "Select high-scoring labels, normalize/deduplicate them, then return the top-k stable labels.",
            "cases": [
                ([{"label": "A", "score": 3}, {"label": "a", "score": 4}, {"label": "B", "score": 2}, {"label": "c", "score": 5}], 2, ["a", "b"]),
                ([{"label": "x", "score": 1}, {"label": "Y", "score": 2}, {"label": "y", "score": 3}], 2, ["y"]),
            ],
            "return_shape": "list",
            "type_family": "return_shape_record_pipeline",
            "required_constructs": ["loop", "branch", "locals", "composition", "type_and_return_shape"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "clamp_round_then_rle_encode",
            "steps": ["bpg_clamp_round", "bpg_rle_encode"],
            "prompt": "Clamp and round numeric values, then run-length encode repeated rounded outputs.",
            "cases": [([1.234, 2.6, 2.6, 9, -5], (0, 3, 0), [(1.0, 1), (3.0, 3), (0.0, 1)]), ([4.2, 4.4, 1], (0, 4, 0), [(4.0, 2), (1.0, 1)])],
            "return_shape": "list",
            "type_family": "return_shape_numeric_pipeline",
            "required_constructs": ["loop", "branch", "locals", "composition", "type_and_return_shape"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "windowed_deltas_then_longest_even_run",
            "steps": ["bpg_windowed_deltas", "bpg_longest_even_run"],
            "prompt": "Clip numeric readings, compute adjacent deltas, then return the longest even-delta run.",
            "cases": [([0, 2, 4, 7, 9], (0, 10), 2), ([5, 7, 9, 10, 12], (0, 20), 2)],
            "return_shape": "number",
            "type_family": "algorithmic_planning",
            "required_constructs": ["loop", "branch", "locals", "composition", "state_machine"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "parse_signed_ints_then_clamp_round_then_rle_encode",
            "steps": ["bpg_parse_signed_ints", "bpg_clamp_round", "bpg_rle_encode"],
            "prompt": "Extract signed integers, clamp/round them, then run-length encode the shaped numeric output.",
            "cases": [("7 7 -1 3 +4", (0, 5, 0), [(5.0, 2), (0.0, 1), (3.0, 1), (4.0, 1)]), ("-5 -5 2", (0, 3, 0), [(0.0, 2), (2.0, 1)])],
            "return_shape": "list",
            "type_family": "verifier_mismatch_shape_pipeline",
            "required_constructs": ["loop", "branch", "locals", "composition", "type_and_return_shape"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "parse_signed_ints_then_top_k_frequent",
            "steps": ["bpg_parse_signed_ints", "bpg_top_k_frequent"],
            "prompt": "Extract signed integers from noisy text, then return the most frequent parsed values.",
            "cases": [("5 5 2 2 2 7", 2, [2, 5]), ("-1 -1 3 3 3", 1, [3])],
            "return_shape": "list",
            "type_family": "no_admissible_candidate_regression",
            "required_constructs": ["loop", "branch", "locals", "composition", "collection_ops"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "threshold_labels_then_rle_encode",
            "steps": ["bpg_threshold_labels", "bpg_rle_encode"],
            "prompt": "Filter labels by score and run-length encode the resulting label stream.",
            "cases": [([{"label": "a", "score": 3}, {"label": "a", "score": 4}, {"label": "b", "score": 2}], 2, [("a", 2), ("b", 1)]), ([{"label": "x", "score": 0}], 1, [])],
            "return_shape": "list",
            "type_family": "return_shape_record_pipeline",
            "required_constructs": ["loop", "branch", "locals", "composition", "record_filter"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "stable_dedup_then_safe_head_default",
            "steps": ["bpg_stable_dedup", "bpg_safe_head_default"],
            "prompt": "Normalize/deduplicate tokens, then return the first normalized token or the supplied default.",
            "cases": [(["", " A ", "a", "B"], "none", "a"), ([], "missing", "missing")],
            "return_shape": "unknown",
            "type_family": "interface_fidelity",
            "required_constructs": ["loop", "branch", "locals", "composition", "default_return"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "parse_signed_ints_then_numeric_stats_tuple",
            "steps": ["bpg_parse_signed_ints", "bpg_numeric_stats_tuple"],
            "prompt": "Extract signed integers from noisy text, then return the typed numeric stats tuple.",
            "cases": [("z -5 12 0", None, (-5, 12, 3)), ("none", None, (None, None, 0))],
            "return_shape": "tuple",
            "type_family": "return_shape_numeric_tuple",
            "required_constructs": ["loop", "branch", "locals", "composition", "type_and_return_shape"],
            "visible_arg_count_hint": 1,
        },
        {
            "id": "parse_signed_ints_then_max_non_adjacent_sum",
            "steps": ["bpg_parse_signed_ints", "bpg_max_non_adjacent_sum"],
            "prompt": "Extract signed integers from noisy text, then solve a non-adjacent dynamic-programming selection.",
            "cases": [("1 2 9 4", None, 10), ("-5 7 6 5", None, 12)],
            "return_shape": "number",
            "type_family": "algorithmic_planning",
            "required_constructs": ["loop", "branch", "locals", "composition", "dynamic_programming"],
            "visible_arg_count_hint": 1,
        },
        {
            "id": "merge_intervals_then_interval_coverage",
            "steps": ["bpg_merge_intervals", "bpg_interval_coverage"],
            "prompt": "Merge interval records, then return total half-open coverage.",
            "cases": [([(1, 3), (2, 5), (10, 12)], None, 6), ([(0, 1), (1, 4)], None, 4)],
            "return_shape": "number",
            "type_family": "algorithmic_planning_interval_contract",
            "required_constructs": ["loop", "branch", "locals", "composition", "type_and_return_shape"],
            "visible_arg_count_hint": 1,
        },
        {
            "id": "normalize_filter_sort_then_rle_encode",
            "steps": ["bpg_normalize_filter_sort", "bpg_rle_encode"],
            "prompt": "Normalize/filter a token stream, then preserve the shaped list contract through run-length encoding.",
            "cases": [([" A ", "bb", "a", "CC", "bb"], ["cc"], [("bb", 1)]), (["xx", "YY", "xx", "zz"], [], [("xx", 1), ("yy", 1), ("zz", 1)])],
            "return_shape": "list",
            "type_family": "return_shape_collection_pipeline",
            "required_constructs": ["loop", "branch", "locals", "composition", "collection_ops", "type_and_return_shape"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "threshold_labels_then_top_k_frequent",
            "steps": ["bpg_threshold_labels", "bpg_top_k_frequent"],
            "prompt": "Select labels by a numeric threshold, then rank the visible labels by frequency.",
            "cases": [
                ([{"label": "a", "score": 3}, {"label": "b", "score": 4}, {"label": "a", "score": 5}, {"label": "c", "score": 1}], 2, ["a", "b"]),
                ([{"label": "x", "score": 9}, {"label": "y", "score": 1}, {"label": "x", "score": 9}], 1, ["x"]),
            ],
            "return_shape": "list",
            "type_family": "no_admissible_candidate_regression",
            "required_constructs": ["loop", "branch", "locals", "composition", "collection_ops", "type_and_return_shape"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "clamp_round_then_numeric_stats_tuple",
            "steps": ["bpg_clamp_round", "bpg_numeric_stats_tuple"],
            "prompt": "Clamp and round numeric values, then return typed stats for the shaped numeric list.",
            "cases": [([1.234, 2.6, 9, -5], (0, 3, 0), (0.0, 3.0, 4)), (["bad", 2.2, 4.8], (0, 5, 0), (2.0, 5.0, 2))],
            "return_shape": "tuple",
            "type_family": "return_shape_numeric_tuple",
            "required_constructs": ["loop", "branch", "locals", "composition", "numeric_ops", "type_and_return_shape"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "windowed_deltas_then_numeric_stats_tuple",
            "steps": ["bpg_windowed_deltas", "bpg_numeric_stats_tuple"],
            "prompt": "Clip a numeric sequence, compute deltas, then return the typed stats tuple for those deltas.",
            "cases": [([-5, 0, 10], (0, 5), (0.0, 5.0, 2)), ([1, 3, 6, 10], (0, 10), (2.0, 4.0, 3))],
            "return_shape": "tuple",
            "type_family": "algorithmic_planning_return_shape",
            "required_constructs": ["loop", "branch", "locals", "composition", "numeric_ops", "type_and_return_shape"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "parse_query_string_then_safe_head_default",
            "steps": ["bpg_parse_query_string", "bpg_safe_head_default"],
            "prompt": "Parse a query string into a mapping, then intentionally preserve the default interface when the next step expects a sequence.",
            "cases": [("?a=1&b=2", "fallback", "fallback"), ("", "missing", "missing")],
            "return_shape": "unknown",
            "type_family": "verifier_mismatch_interface_boundary",
            "required_constructs": ["branch", "locals", "composition", "default_return", "type_and_return_shape"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "stdin_prefix_queries_then_parse_signed_ints_then_numeric_stats_tuple",
            "steps": ["bpg_stdin_prefix_queries", "bpg_parse_signed_ints", "bpg_numeric_stats_tuple"],
            "prompt": "Parse prefix-query stdin output, re-parse the emitted answers, then return a typed stats tuple.",
            "cases": [
                ("5 3 1 2 3 4 5 1 3 2 5 4 4", None, (4, 14, 3)),
                ("3 2 -1 10 4 1 2 2 3", None, (9, 14, 2)),
            ],
            "return_shape": "tuple",
            "type_family": "verifier_mismatch_stdin_return_shape",
            "required_constructs": ["loop", "branch", "locals", "composition", "stdin_parser", "type_and_return_shape"],
            "visible_arg_count_hint": 1,
        },
        {
            "id": "stdin_prefix_queries_then_parse_signed_ints_then_top_k_frequent",
            "steps": ["bpg_stdin_prefix_queries", "bpg_parse_signed_ints", "bpg_top_k_frequent"],
            "prompt": "Parse prefix-query stdin output, re-parse the answers, then rank the most frequent answer values.",
            "cases": [
                ("4 3 1 2 1 4 1 2 2 3 4 4", 1, [3]),
                ("5 4 2 2 2 3 3 1 1 2 2 4 4 4 5", 2, [2, 3]),
            ],
            "return_shape": "list",
            "type_family": "no_admissible_candidate_regression",
            "required_constructs": ["loop", "branch", "locals", "composition", "stdin_parser", "collection_ops"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "stdin_pair_sums_then_parse_signed_ints_then_max_non_adjacent_sum",
            "steps": ["bpg_stdin_pair_sums", "bpg_parse_signed_ints", "bpg_max_non_adjacent_sum"],
            "prompt": "Sum stdin-style integer pairs, re-parse the sums, then solve a non-adjacent DP selection.",
            "cases": [
                ("1 2\n5 6\n2 4\n7 3", None, 21),
                ("4 4\n9 -2\n1 1", None, 10),
            ],
            "return_shape": "number",
            "type_family": "algorithmic_planning_stdin_dp",
            "required_constructs": ["loop", "branch", "locals", "composition", "stdin_parser", "dynamic_programming"],
            "visible_arg_count_hint": 1,
        },
        {
            "id": "parse_signed_ints_then_windowed_deltas_then_numeric_stats_tuple",
            "steps": ["bpg_parse_signed_ints", "bpg_windowed_deltas", "bpg_numeric_stats_tuple"],
            "prompt": "Extract signed integers, clamp into a window, compute deltas, then return typed numeric stats.",
            "cases": [
                ("0 10 -5 5", (0, 6), (-6.0, 6.0, 3)),
                ("3 3 9", (0, 4), (0.0, 1.0, 2)),
            ],
            "return_shape": "tuple",
            "type_family": "algorithmic_planning_return_shape",
            "required_constructs": ["loop", "branch", "locals", "composition", "numeric_ops", "type_and_return_shape"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "project_table_then_safe_head_default",
            "steps": ["bpg_project_table", "bpg_safe_head_default"],
            "prompt": "Project table rows to requested columns, then preserve the first-row/default interface contract.",
            "cases": [
                ([{"id": 1, "x": 2}, {"id": 2}], ["id", "x"], {"id": 1, "x": 2}),
                ([], ["id"], ["id"]),
            ],
            "return_shape": "unknown",
            "type_family": "verifier_mismatch_interface_boundary",
            "required_constructs": ["loop", "branch", "locals", "composition", "record_filter", "default_return"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "graph_components",
            "steps": ["bpg_graph_components"],
            "prompt": "Build an undirected graph from edge records and return the number of connected components.",
            "cases": [
                (5, [(0, 1), (1, 2), (3, 4)], 2),
                (4, [(0, 1), (2, 3), (1, 2)], 1),
            ],
            "return_shape": "number",
            "type_family": "algorithmic_planning_graph",
            "required_constructs": ["loop", "branch", "locals", "graph_traversal"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "shortest_hops",
            "steps": ["bpg_shortest_hops"],
            "prompt": "Build an undirected graph from edge records and return the shortest hop distance between two visible nodes.",
            "cases": [
                (5, ([(0, 1), (1, 2), (2, 4), (0, 3)], 0, 4), 3),
                (4, ([(0, 1), (2, 3)], 0, 3), -1),
            ],
            "return_shape": "number",
            "type_family": "algorithmic_planning_graph",
            "required_constructs": ["loop", "branch", "locals", "graph_traversal", "queue"],
            "visible_arg_count_hint": 4,
        },
        {
            "id": "lcs_length",
            "steps": ["bpg_lcs_length"],
            "prompt": "Compute the longest common subsequence length for two visible strings.",
            "cases": [
                ("abcde", "ace", 3),
                ("theseus", "hive", 2),
            ],
            "return_shape": "number",
            "type_family": "algorithmic_planning_string_dp",
            "required_constructs": ["loop", "branch", "locals", "dynamic_programming"],
            "visible_arg_count_hint": 2,
        },
        {
            "id": "balanced_parens",
            "steps": ["bpg_balanced_parens"],
            "prompt": "Validate balanced parentheses, brackets, and braces in a visible string.",
            "cases": [
                ("([]{})", None, True),
                ("([)]", None, False),
            ],
            "return_shape": "bool",
            "type_family": "algorithmic_planning_stack",
            "required_constructs": ["loop", "branch", "locals", "stack"],
            "visible_arg_count_hint": 1,
        },
        {
            "id": "group_records",
            "steps": ["bpg_group_records"],
            "prompt": "Group record IDs by a visible record field and preserve the typed mapping contract.",
            "cases": [
                ([{"id": 1, "room": "kitchen"}, {"id": 2, "room": "lab"}, {"id": 3, "room": "kitchen"}], "room", {"kitchen": [1, 3], "lab": [2]}),
                ([{"id": 4, "kind": "sensor"}, {"name": "skip"}, {"id": 5, "kind": "node"}], "kind", {"sensor": [4], "node": [5]}),
            ],
            "return_shape": "dict",
            "type_family": "return_shape_record_pipeline",
            "required_constructs": ["loop", "branch", "locals", "record_filter", "type_and_return_shape"],
            "visible_arg_count_hint": 2,
        },
    ]


def write_frontier_heldout(out: Path, bodies: dict[str, str], *, rows: int, seed: int) -> dict[str, Any]:
    specs = frontier_specs()
    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index in range(rows):
        spec = specs[(index + seed) % len(specs)]
        data, other, expected = spec["cases"][(index // len(specs)) % len(spec["cases"])]
        task_index = 4_000_000 + index
        entry = f"private_residual_frontier_{spec['id']}_{task_index:07d}"
        try:
            body = render_solution_body(spec["steps"], bodies)
        except Exception as exc:
            body = ""
            failures.append({"task_id": f"private_residual_frontier_v1_{task_index:07d}", "error": f"{exc.__class__.__name__}: {exc}"})
        row = {
            "task_id": f"private_residual_frontier_v1_{spec['id']}_{task_index:07d}",
            "source_task_id": f"private_residual_frontier_v1_private_{seed}_{task_index:07d}",
            "card_id": "broad_private_generalization_ladder_v1",
            "source_id": "local_generated_private_residual_frontier_v1",
            "split": "eval",
            "category": f"private_residual_frontier_{spec['id']}",
            "prompt": f"Private residual frontier contract: {spec['prompt']}",
            "entry_point": entry,
            "solution_expr": "",
            "solution_body": body,
            "tests": render_test(entry, data, other, expected, visible_arg_count=int(spec.get("visible_arg_count_hint") or 2)),
            "candidate_expression_eligible": False,
            "public_benchmark": False,
            "public_prompts_included": False,
            "public_tests_included": False,
            "public_benchmark_solutions_included": False,
            "public_score_labels_included": False,
            "license_spdx": "LicenseRef-Private-Theseus-Synthetic",
            "provenance": "private_residual_frontier_v1_generated_from_private_compositions",
            "benchmark_evidence_level": "broad_private_generalization_ladder_v1_generated_only;private_residual_frontier_v1_generated_only",
            "broad_private_family_v1": "private_residual_frontier_v1",
            "targeted_private_residual_family_v3": "private_residual_frontier_v1",
            "concept_residual_label": "public_aggregate_residual_private_frontier",
            "residual_concept": "private residual composition, not public benchmark content",
            "private_residual_frontier_v1": {
                "aggregate_public_residual_categories_only": True,
                "source_public_residual_report": rel(PUBLIC_RESIDUAL),
                "spec_id": spec["id"],
                "steps": [{"semantic_family": step} for step in spec["steps"]],
            },
            "novel_composition_v1": {
                "steps": [{"semantic_family": step} for step in spec["steps"]],
                "frontier": "private_residual_frontier_v1",
            },
            "decoder_contract": {
                "policy": "project_theseus_decoder_contract_v1_broad_private_generalization",
                "semantic_family": f"private_residual_frontier_{spec['id']}",
                "residual_label_hint": spec["id"],
                "type_family": spec["type_family"],
                "return_shape": spec["return_shape"],
                "required_constructs": spec["required_constructs"],
                "composition_steps": [{"semantic_family": step} for step in spec["steps"]],
                "visible_arg_count_hint": int(spec.get("visible_arg_count_hint") or 2),
                "full_body_required": True,
                "guardrail_only": False,
                "feedback_weight": 1.65,
                "score_semantics": "private residual frontier only",
                "generation_plan": {
                    "policy": "aggregate_public_residual_category -> private composition specs -> learned token decoder",
                    "repair_strategy": "internalize residual-family behavior through private reusable composition, not public benchmark replay",
                    "public_tests_used": False,
                    "public_solutions_used": False,
                },
            },
            "tags": [
                "private_residual_frontier_v1",
                "heldout",
                "aggregate_public_residual_private_proxy",
                *spec["steps"],
            ],
        }
        output.append(row)
    verify = verify_private_solution_rows(output, max_failures=24)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, output)
    counts = Counter(str(row.get("category") or "") for row in output)
    leakage = public_leakage_scan(output)
    return {
        "heldout": rel(out),
        "row_count": len(output),
        "frontier_spec_count": len(specs),
        "category_counts": dict(sorted(counts.items())),
        "private_solution_failure_count": int(verify.get("failure_count") or 0) + len(failures),
        "private_solution_sample_failures": list(verify.get("sample_failures") or []) + failures[:8],
        "public_data_leakage_hit_count": leakage["hit_count"],
        "public_tests_used": False,
        "public_solutions_used": False,
    }


def render_test(entry: str, data: Any, other: Any, expected: Any, *, visible_arg_count: int) -> str:
    if visible_arg_count <= 1:
        return f"assert {entry}({data!r}) == {expected!r}\n"
    if visible_arg_count > 2 and isinstance(other, (list, tuple)):
        return f"assert {entry}({data!r}, {', '.join(repr(item) for item in other)}) == {expected!r}\n"
    return f"assert {entry}({data!r}, {other!r}) == {expected!r}\n"


def preflight_report(args: argparse.Namespace, heldout_report: dict[str, Any]) -> dict[str, Any]:
    packet = read_json(READINESS_PACKET, {})
    dry_run = read_json(OPERATOR_DRY_RUN, {})
    dry_summary = object_field(dry_run, "summary")
    disk = shutil.disk_usage(ROOT)
    free_gb = disk.free / (1024**3)
    battery = battery_state()
    checkpoint = resolve(args.checkpoint_in)
    post_v4_state = post_v4_public_artifact_state()
    gates = [
        gate("release_binary_present", RELEASE.exists(), rel(RELEASE), "hard"),
        gate("checkpoint_present", checkpoint.exists(), rel(checkpoint), "hard"),
        gate("operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK), "hard"),
        gate("public_calibration_disallowed", packet.get("public_calibration_allowed") is False, packet.get("public_calibration_allowed"), "hard"),
        gate("operator_dry_run_not_executed", dry_summary.get("executed") is False, dry_summary.get("executed"), "hard"),
        gate("post_v4_public_artifacts_approved_or_absent", post_v4_state["allowed"], post_v4_state, "hard"),
        gate("heldout_private_solution_tests_pass", heldout_report["private_solution_failure_count"] == 0, heldout_report, "hard"),
        gate("free_disk_ge_min", free_gb >= float(args.min_free_gb), {"free_gb": round(free_gb, 3), "min_free_gb": float(args.min_free_gb)}, "hard"),
        gate("battery_allowed_or_ac_power", bool(args.allow_battery or not battery.get("on_battery")), battery, "hard"),
    ]
    blockers = [row for row in gates if not row["passed"]]
    return {
        "ready": not blockers,
        "blockers": blockers,
        "gates": gates,
        "free_gb": round(free_gb, 3),
        "battery": battery,
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def phase_commands(args: argparse.Namespace) -> list[tuple[str, list[str], dict[str, str]]]:
    rows = max(1, int(args.rows))
    return [
        (
            "build_private_safe_sts_streams",
            [sys.executable, "scripts/private_task_sts_streams.py", "--tasks", rel(HELDOUT), "--out", rel(STS_STREAMS), "--report-out", rel(STS_STREAMS_REPORT), "--task-limit", str(rows)],
            {},
        ),
        ("fanout_sts_on", fanout_command(args, HELDOUT, PRIVATE_CANDIDATES, PUBLIC_CANDIDATES, FANOUT_REPORT, STS_STREAMS, rows), fanout_env(sts_on=True)),
        ("fanout_sts_off_control", fanout_command(args, HELDOUT, CONTROL_PRIVATE_CANDIDATES, CONTROL_PUBLIC_CANDIDATES, CONTROL_FANOUT_REPORT, EMPTY_STS, rows), fanout_env(sts_on=False)),
        ("score_all_candidates", score_command(args, PRIVATE_CANDIDATES, SCORE, SCORE_MD), {}),
        ("score_frontier_only", score_command(args, FRONTIER_CANDIDATES, FRONTIER_SCORE, FRONTIER_SCORE_MD), {}),
    ]


def run_execution_phases(args: argparse.Namespace, deadline: float) -> list[dict[str, Any]]:
    rows = max(1, int(args.rows))
    shard_size = int(args.shard_size)
    clear_execution_outputs()
    if shard_size <= 0 or rows <= shard_size:
        return run_monolithic_execution(args, deadline)
    return run_sharded_execution(args, deadline, rows, max(1, shard_size))


def run_monolithic_execution(args: argparse.Namespace, deadline: float) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for name, command, env in phase_commands(args):
        timeout = phase_timeout(deadline, name)
        if timeout <= 60 and name != "score_frontier_only":
            commands.append(timeout_record(name))
            break
        if name == "score_frontier_only":
            current_candidates = read_jsonl(PRIVATE_CANDIDATES)
            write_jsonl(FRONTIER_CANDIDATES, [row for row in current_candidates if frontier_token_candidate(row)])
        commands.append(run_command(name, command, env=env, timeout=timeout))
        if commands[-1]["returncode"] != 0:
            break
    return commands


def run_sharded_execution(args: argparse.Namespace, deadline: float, rows: int, shard_size: int) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    heldout_rows = read_jsonl(HELDOUT)[:rows]
    shards = write_shards(heldout_rows, shard_size)
    sts_candidates: list[Path] = []
    sts_public: list[Path] = []
    control_candidates: list[Path] = []
    control_public: list[Path] = []
    fanout_reports: list[Path] = []
    control_fanout_reports: list[Path] = []
    sts_stream_files: list[Path] = []
    sts_reports: list[Path] = []

    for shard in shards:
        timeout = phase_timeout(deadline, f"build_private_safe_sts_streams_{shard['index']:03d}")
        if timeout <= 60:
            commands.append(timeout_record("build_private_safe_sts_streams"))
            return commands
        commands.append(
            run_command(
                f"build_private_safe_sts_streams_{shard['index']:03d}",
                [
                    sys.executable,
                    "scripts/private_task_sts_streams.py",
                    "--tasks",
                    rel(shard["heldout"]),
                    "--out",
                    rel(shard["sts_streams"]),
                    "--report-out",
                    rel(shard["sts_report"]),
                    "--task-limit",
                    str(shard["row_count"]),
                ],
                timeout=timeout,
            )
        )
        if commands[-1]["returncode"] != 0:
            return commands
        sts_stream_files.append(shard["sts_streams"])
        sts_reports.append(shard["sts_report"])

        timeout = phase_timeout(deadline, f"fanout_sts_on_{shard['index']:03d}")
        if timeout <= 60:
            commands.append(timeout_record("fanout_sts_on"))
            return commands
        commands.append(
            run_command(
                f"fanout_sts_on_{shard['index']:03d}",
                fanout_command(
                    args,
                    shard["heldout"],
                    shard["private_candidates"],
                    shard["public_candidates"],
                    shard["fanout_report"],
                    shard["sts_streams"],
                    shard["row_count"],
                ),
                env=fanout_env(sts_on=True),
                timeout=timeout,
            )
        )
        if commands[-1]["returncode"] != 0:
            return commands
        sts_candidates.append(shard["private_candidates"])
        sts_public.append(shard["public_candidates"])
        fanout_reports.append(shard["fanout_report"])

        timeout = phase_timeout(deadline, f"fanout_sts_off_control_{shard['index']:03d}")
        if timeout <= 60:
            commands.append(timeout_record("fanout_sts_off_control"))
            return commands
        commands.append(
            run_command(
                f"fanout_sts_off_control_{shard['index']:03d}",
                fanout_command(
                    args,
                    shard["heldout"],
                    shard["control_private_candidates"],
                    shard["control_public_candidates"],
                    shard["control_fanout_report"],
                    EMPTY_STS,
                    shard["row_count"],
                ),
                env=fanout_env(sts_on=False),
                timeout=timeout,
            )
        )
        if commands[-1]["returncode"] != 0:
            return commands
        control_candidates.append(shard["control_private_candidates"])
        control_public.append(shard["control_public_candidates"])
        control_fanout_reports.append(shard["control_fanout_report"])

    concatenate_jsonl(sts_candidates, PRIVATE_CANDIDATES)
    concatenate_jsonl(control_candidates, CONTROL_PRIVATE_CANDIDATES)
    concatenate_jsonl(sts_public, PUBLIC_CANDIDATES)
    concatenate_jsonl(control_public, CONTROL_PUBLIC_CANDIDATES)
    concatenate_jsonl(sts_stream_files, STS_STREAMS)
    aggregate_reports(STS_STREAMS_REPORT, sts_reports, "private_residual_frontier_v1_sts_streams_sharded")
    aggregate_reports(FANOUT_REPORT, fanout_reports, "private_residual_frontier_v1_fanout_sharded")
    aggregate_reports(CONTROL_FANOUT_REPORT, control_fanout_reports, "private_residual_frontier_v1_control_fanout_sharded")
    commands.extend(
        [
            aggregate_command_record("build_private_safe_sts_streams", commands, "build_private_safe_sts_streams_"),
            aggregate_command_record("fanout_sts_on", commands, "fanout_sts_on_"),
            aggregate_command_record("fanout_sts_off_control", commands, "fanout_sts_off_control_"),
        ]
    )

    for name, command, env in [
        ("score_all_candidates", score_command(args, PRIVATE_CANDIDATES, SCORE, SCORE_MD), {}),
        ("score_frontier_only", score_command(args, FRONTIER_CANDIDATES, FRONTIER_SCORE, FRONTIER_SCORE_MD), {}),
    ]:
        timeout = phase_timeout(deadline, name)
        if timeout <= 60 and name != "score_frontier_only":
            commands.append(timeout_record(name))
            break
        if name == "score_frontier_only":
            current_candidates = read_jsonl(PRIVATE_CANDIDATES)
            write_jsonl(FRONTIER_CANDIDATES, [row for row in current_candidates if frontier_token_candidate(row)])
        commands.append(run_command(name, command, env=env, timeout=timeout))
        if commands[-1]["returncode"] != 0:
            break
    return commands


def write_shards(rows: list[dict[str, Any]], shard_size: int) -> list[dict[str, Any]]:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    for path in SHARD_DIR.glob("private_residual_frontier_v1_shard_*"):
        if path.is_file():
            path.unlink()
    shards = []
    for index, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start : start + shard_size]
        prefix = SHARD_DIR / f"private_residual_frontier_v1_shard_{index:03d}"
        shard = {
            "index": index,
            "row_count": len(shard_rows),
            "heldout": SHARD_DIR / f"{prefix.name}_tasks.jsonl",
            "sts_streams": SHARD_DIR / f"{prefix.name}_sts_streams.jsonl",
            "sts_report": SHARD_DIR / f"{prefix.name}_sts_streams.json",
            "private_candidates": SHARD_DIR / f"{prefix.name}_private_candidates.jsonl",
            "public_candidates": SHARD_DIR / f"{prefix.name}_empty_public_candidates.jsonl",
            "control_private_candidates": SHARD_DIR / f"{prefix.name}_control_private_candidates.jsonl",
            "control_public_candidates": SHARD_DIR / f"{prefix.name}_control_empty_public_candidates.jsonl",
            "fanout_report": SHARD_DIR / f"{prefix.name}_fanout.json",
            "control_fanout_report": SHARD_DIR / f"{prefix.name}_control_fanout.json",
        }
        write_jsonl(shard["heldout"], shard_rows)
        write_text(shard["public_candidates"], "")
        write_text(shard["control_public_candidates"], "")
        shards.append(shard)
    return shards


def clear_execution_outputs() -> None:
    for path in [
        STS_STREAMS,
        STS_STREAMS_REPORT,
        PRIVATE_CANDIDATES,
        PUBLIC_CANDIDATES,
        CONTROL_PRIVATE_CANDIDATES,
        CONTROL_PUBLIC_CANDIDATES,
        FRONTIER_CANDIDATES,
        FANOUT_REPORT,
        CONTROL_FANOUT_REPORT,
        SCORE,
        SCORE_MD,
        FRONTIER_SCORE,
        FRONTIER_SCORE_MD,
    ]:
        if path.exists():
            path.unlink()
    ensure_sidecars()


def concatenate_jsonl(inputs: list[Path], out: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        rows.extend(read_jsonl(path))
    write_jsonl(out, rows)


def aggregate_reports(out: Path, inputs: list[Path], policy: str) -> None:
    reports = [read_json(path, {}) for path in inputs]
    summaries = [object_field(report, "summary") for report in reports]
    write_json(
        out,
        {
            "policy": policy,
            "created_utc": now(),
            "shard_count": len(inputs),
            "shards": [rel(path) for path in inputs],
            "summary": {
                "shard_count": len(inputs),
                "total_candidate_rows": sum(int(first_number(summary.get("candidate_row_count"), summary.get("private_candidate_row_count"), 0)) for summary in summaries),
                "external_inference_calls": sum(int(first_number(report.get("external_inference_calls"), object_field(report, "summary").get("external_inference_calls"), 0)) for report in reports),
            },
        },
    )


def aggregate_command_record(name: str, commands: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    rows = [row for row in commands if str(row.get("name") or "").startswith(prefix)]
    return {
        "name": name,
        "returncode": 0 if rows and all(row.get("returncode") == 0 for row in rows) else 1,
        "elapsed_seconds": round(sum(first_number(row.get("elapsed_seconds")) for row in rows), 3),
        "shard_count": len(rows),
        "stdout_tail": "",
        "stderr_tail": "",
    }


def phase_timeout(deadline: float, name: str) -> int:
    return max(60, int(deadline - time.time()))


def timeout_record(name: str) -> dict[str, Any]:
    return {"name": name, "returncode": 124, "stderr_tail": "time budget exhausted before phase", "stdout_tail": ""}


def fanout_command(args: argparse.Namespace, private_curriculum: Path, private_out: Path, public_out: Path, report_out: Path, sts_streams: Path, eval_limit: int) -> list[str]:
    return [
        rel(RELEASE),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        rel(private_curriculum),
        "--public-task-manifest",
        rel(EMPTY_PUBLIC),
        "--checkpoint-in",
        rel(resolve(args.checkpoint_in)),
        "--seed",
        str(int(args.seed)),
        "--candidates-per-task",
        str(max(1, int(args.candidates_per_task))),
        "--private-candidate-out",
        rel(private_out),
        "--public-candidate-out",
        rel(public_out),
        "--report-out",
        rel(report_out),
        "--public-task-limit",
        "0",
        "--private-eval-limit",
        str(eval_limit),
        "--sts-streams",
        rel(sts_streams),
    ]


def score_command(args: argparse.Namespace, candidates: Path, out: Path, markdown: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/broad_private_generalization_score_v1.py",
        "--heldout",
        rel(HELDOUT),
        "--candidates",
        rel(candidates),
        "--control-candidates",
        rel(CONTROL_PRIVATE_CANDIDATES),
        "--timeout-seconds",
        str(max(1, int(args.score_timeout_seconds))),
        "--min-heldout-rows",
        str(max(1, int(args.min_rows))),
        "--out",
        rel(out),
        "--markdown-out",
        rel(markdown),
    ]


def fanout_env(*, sts_on: bool) -> dict[str, str]:
    env = {
        "THESEUS_CODE_LM_LOW_LATENCY_FANOUT": "1",
        "THESEUS_CODE_LM_PRIVATE_LOW_LATENCY_MULTI_CANDIDATE_FANOUT": "1",
        "THESEUS_CODE_LM_LOW_LATENCY_EXPENSIVE_RESCUE": "0",
    }
    if not sts_on:
        env["THESEUS_CODE_LM_DISABLE_STS_DECODER_CONTROL_POLICY"] = "1"
    return env


def ensure_sidecars() -> None:
    for path in [EMPTY_PUBLIC, EMPTY_STS, PUBLIC_CANDIDATES, CONTROL_PUBLIC_CANDIDATES]:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("", encoding="utf-8")


def candidate_inventory(candidates: list[dict[str, Any]], frontier_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    modes = Counter(str(row.get("candidate_generation_mode") or "") for row in candidates)
    return {
        "candidate_rows": len(candidates),
        "frontier_candidate_rows": len(frontier_candidates),
        "frontier_token_rows": sum(frontier_token_candidate(row) for row in candidates),
        "diagnostic_adapter_rows": sum(diagnostic_adapter_candidate(row) for row in candidates),
        "prototype_rows": sum(true(row.get("broad_private_train_prototype_stage")) for row in candidates),
        "top_modes": dict(modes.most_common(20)),
    }


def pass_inventory_summary(score: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, int]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        index.setdefault((str(row.get("task_id") or ""), str(row.get("candidate_generation_mode") or "")), []).append(row)
    out = {"passed_result_count": 0, "frontier_token_pass_count": 0, "diagnostic_adapter_pass_count": 0, "prototype_pass_count": 0}
    for result in score.get("results") if isinstance(score.get("results"), list) else []:
        if not true(result.get("passed")):
            continue
        out["passed_result_count"] += 1
        rows = index.get((str(result.get("task_id") or ""), str(result.get("pass_candidate_mode") or "")), [])
        if any(frontier_token_candidate(row) for row in rows):
            out["frontier_token_pass_count"] += 1
        if any(diagnostic_adapter_candidate(row) for row in rows):
            out["diagnostic_adapter_pass_count"] += 1
        if any(true(row.get("broad_private_train_prototype_stage")) for row in rows):
            out["prototype_pass_count"] += 1
    return out


def frontier_token_candidate(row: dict[str, Any]) -> bool:
    mode = str(row.get("candidate_generation_mode") or "").lower()
    return true(row.get("compositional_token_candidate")) and "novel_composition_v1" in mode and true(row.get("token_level_code_generation_learned"))


def diagnostic_adapter_candidate(row: dict[str, Any]) -> bool:
    return true(row.get("broad_private_generalization_semantic_adapter_stage")) or true(row.get("private_residual_v3_semantic_adapter_stage"))


def commands_succeeded(commands: list[dict[str, Any]], names: list[str]) -> bool:
    by_name = {str(row.get("name") or ""): row for row in commands}
    return bool(commands) and all(name in by_name and by_name[name].get("returncode") == 0 for name in names)


def command_evidence(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": row.get("name"), "returncode": row.get("returncode"), "elapsed_seconds": row.get("elapsed_seconds")} for row in commands]


def run_command(name: str, command: list[str], *, env: dict[str, str] | None = None, timeout: int = 3600) -> dict[str, Any]:
    started = time.time()
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    try:
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=proc_env, timeout=max(60, timeout))
        return {
            "name": name,
            "command": command,
            "returncode": proc.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": command,
            "returncode": 124,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "timeout",
        }


def next_actions(trigger_state: str, failed: list[dict[str, Any]], pass_rate: float, frontier_rate: float, floor: float) -> list[str]:
    if trigger_state == "RED":
        return ["Fix hard safety/preflight blockers before running private residual frontier work."]
    if not failed:
        return ["Refresh the generalization governor; this is private-only evidence and does not unlock public calibration."]
    if pass_rate < floor or frontier_rate < floor:
        return ["Inspect frontier failure clusters and patch learned decoder composition/ranking before any public calibration."]
    return ["Rerun with a larger row count for overnight residual-frontier pressure, then refresh the governor."]


def queue_rows(report: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if report["trigger_state"] != "GREEN":
        rows.append(
            queue_item(
                "run_private_residual_frontier_v1",
                "Run private residual frontier composition pressure without public calibration.",
                [
                    sys.executable,
                    "scripts/private_residual_frontier_v1.py",
                    "--execute",
                    "--rows",
                    str(max(int(args.rows), int(args.min_rows))),
                    "--min-rows",
                    str(int(args.min_rows)),
                    "--max-hours",
                    str(float(args.max_hours)),
                    "--shard-size",
                    str(int(args.shard_size)),
                ],
                priority=10,
            )
        )
    rows.append(
        queue_item(
            "refresh_generalization_governor",
            "Refresh the governor after private residual frontier evidence; public calibration remains locked.",
            [
                sys.executable,
                "scripts/theseus_generalization_governor_v1.py",
                "--out",
                "reports/theseus_generalization_governor_v1.json",
                "--markdown-out",
                "reports/theseus_generalization_governor_v1.md",
                "--queue-out",
                "reports/theseus_generalization_governor_v1_queue.jsonl",
            ],
            priority=90,
        )
    )
    return rows


def queue_item(kind: str, title: str, command: list[str], *, priority: int) -> dict[str, Any]:
    return {
        "policy": "project_theseus_private_residual_frontier_queue_item_v1",
        "queue": "private_residual_frontier_v1",
        "kind": kind,
        "title": title,
        "priority": priority,
        "status": "pending",
        "command": command,
        "safe_to_execute_without_operator_public_approval": True,
        "requires_operator_public_unlock": False,
        "public_calibration_allowed": False,
    }


def artifacts() -> dict[str, str]:
    return {
        "heldout": rel(HELDOUT),
        "sts_streams": rel(STS_STREAMS),
        "shards": rel(SHARD_DIR),
        "private_candidates": rel(PRIVATE_CANDIDATES),
        "control_private_candidates": rel(CONTROL_PRIVATE_CANDIDATES),
        "frontier_candidates": rel(FRONTIER_CANDIDATES),
        "score": rel(SCORE),
        "frontier_score": rel(FRONTIER_SCORE),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    lines = [
        "# Private Residual Frontier v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- row_count: `{summary.get('row_count')}`",
        f"- frontier_spec_count: `{summary.get('frontier_spec_count')}`",
        f"- pass_rate: `{summary.get('pass_rate')}`",
        f"- frontier_only_pass_rate: `{summary.get('frontier_only_pass_rate')}`",
        f"- control_pass_rate: `{summary.get('control_pass_rate')}`",
        f"- frontier_token_pass_count: `{summary.get('frontier_token_pass_count')}`",
        f"- diagnostic_adapter_pass_count: `{summary.get('diagnostic_adapter_pass_count')}`",
        f"- prototype_pass_count: `{summary.get('prototype_pass_count')}`",
        f"- public_calibration_allowed: `{summary.get('public_calibration_allowed')}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates") or []:
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}` ({row.get('severity')})")
    lines.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions") or []:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any, severity: str = "warning") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def battery_state() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {"available": False, "on_battery": False, "raw": ""}
    try:
        proc = subprocess.run(["pmset", "-g", "batt"], text=True, capture_output=True, timeout=5)
    except Exception as exc:
        return {"available": False, "on_battery": False, "raw": f"{exc.__class__.__name__}: {exc}"}
    raw = proc.stdout.strip()
    return {"available": proc.returncode == 0, "on_battery": "Battery Power" in raw, "raw": raw}


def forbidden_public_artifacts() -> list[str]:
    state = post_v4_public_artifact_state()
    return [] if state["allowed"] else list(state["present_artifacts"])


def post_v4_public_artifact_state() -> dict[str, Any]:
    present = [rel(path) for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()]
    if not present:
        return {
            "allowed": True,
            "mode": "absent",
            "present_artifacts": [],
            "approval_valid": False,
            "execute_report_valid": False,
            "operator_lock_active": PUBLIC_LOCK.exists(),
        }
    approval = read_json(OPERATOR_APPROVAL, {})
    execute = read_json(OPERATOR_EXECUTE, {})
    summary = object_field(execute, "summary")
    approval_valid = (
        approval.get("policy") == "project_theseus_public_calibration_operator_approval_v1"
        and approval.get("approved") is True
        and approval.get("proposed_slug") == "post_v4_seed23_5x32"
        and int(approval.get("max_runs") or 0) == 1
    )
    execute_valid = (
        execute.get("policy") == "project_theseus_operator_bounded_public_calibration_v1"
        and execute.get("trigger_state") == "GREEN"
        and summary.get("executed") is True
        and summary.get("proposed_slug") == "post_v4_seed23_5x32"
        and summary.get("output_exists_after") is True
        and summary.get("operator_lock_present_after") is True
        and int(summary.get("run_returncode") or -1) == 0
    )
    required_outputs_present = all(path.exists() for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS[:3])
    allowed = approval_valid and execute_valid and required_outputs_present and PUBLIC_LOCK.exists()
    return {
        "allowed": allowed,
        "mode": "approved_spent_one_shot" if allowed else "unapproved_or_incomplete",
        "present_artifacts": present,
        "approval_valid": approval_valid,
        "execute_report_valid": execute_valid,
        "required_outputs_present": required_outputs_present,
        "operator_lock_active": PUBLIC_LOCK.exists(),
        "rules": "post-v4 public artifacts may exist only after the approved one-shot calibration completed and relocked",
    }


def public_leakage_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    needles = ["humaneval", "mbpp", "evalplus", "bigcodebench", "livecodebench", "canonical_solution", "public_test"]
    hits = []
    for row in rows:
        text = "\n".join(leakage_strings(row)).lower()
        for needle in needles:
            if needle in text:
                hits.append({"task_id": row.get("task_id"), "needle": needle})
                break
    return {"hit_count": len(hits), "sample_hits": hits[:8]}


def leakage_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(leakage_strings(child))
        return out
    if isinstance(value, str):
        return [value]
    return []


def file_empty(path: Path) -> bool:
    return (not path.exists()) or path.stat().st_size == 0


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def top_external_calls(report: dict[str, Any]) -> int:
    return int(first_number(report.get("external_inference_calls"), object_field(report, "summary").get("external_inference_calls"), 0))


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() in {"1", "true", "yes"})


def object_field(value: Any, key: str) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get(key), dict):
        return value[key]
    return {}


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return value if isinstance(value, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(value).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
