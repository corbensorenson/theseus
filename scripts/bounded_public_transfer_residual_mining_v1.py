#!/usr/bin/env python3
"""Mine the spent bounded public calibration into private-only residual pressure.

This script does not run public calibration. It reads the already-recorded
one-shot calibration artifacts and emits only aggregate counts plus hashed task
identifiers. Public prompts, tests, solutions, traces for training, score labels,
and candidate code are never emitted as training rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

DEFAULT_CALIBRATION = REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json"
DEFAULT_TRACE = REPORTS / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl"
DEFAULT_CANDIDATES = REPORTS / "student_code_candidates_capability_transfer_closure_v1_public_shape.jsonl"
DEFAULT_BASELINE = REPORTS / "real_code_benchmark_graduation_wide_public_seed23_5x32_interface_floor_v1.json"
DEFAULT_BASELINE_RESIDUAL = REPORTS / "public_code_transfer_residual_report_wide_public_seed23_5x32_interface_floor_v1.json"
DEFAULT_EXECUTE = REPORTS / "operator_bounded_public_calibration_execute.json"
DEFAULT_REFUSAL = REPORTS / "operator_bounded_public_calibration_goal_no_rerun_refusal.json"
DEFAULT_OUT = REPORTS / "bounded_public_transfer_residual_mining_v1.json"
DEFAULT_MARKDOWN = REPORTS / "bounded_public_transfer_residual_mining_v1.md"
DEFAULT_PRIVATE_TARGETS = REPORTS / "bounded_public_transfer_private_residual_targets_v1.jsonl"

TARGET_CATEGORIES = (
    "parsing_syntax",
    "return_type_shape",
    "algorithm_choice",
    "io_contract_stdin",
    "edge_cases",
    "external_dependency_missing",
    "verifier_mismatch",
    "no_admissible_interface_coverage",
    "candidate_manifest_slice_mismatch",
    "timeout_runtime",
    "selector_ranking_miss",
    "spent_calibration_no_admissible_resolved_by_current_manifest",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default=rel(DEFAULT_CALIBRATION))
    parser.add_argument("--trace", default=rel(DEFAULT_TRACE))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--baseline", default=rel(DEFAULT_BASELINE))
    parser.add_argument("--baseline-residual", default=rel(DEFAULT_BASELINE_RESIDUAL))
    parser.add_argument("--operator-execute", default=rel(DEFAULT_EXECUTE))
    parser.add_argument("--goal-refusal", default=rel(DEFAULT_REFUSAL))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--private-targets-out", default=rel(DEFAULT_PRIVATE_TARGETS))
    args = parser.parse_args()

    report = build_report(args)
    target_rows = as_list(report.get("private_only_residual_target_rows"))
    write_jsonl(resolve(args.private_targets_out), target_rows)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    calibration_path = resolve(args.calibration)
    trace_path = resolve(args.trace)
    candidates_path = resolve(args.candidates)
    baseline_path = resolve(args.baseline)
    baseline_residual_path = resolve(args.baseline_residual)
    execute_path = resolve(args.operator_execute)
    refusal_path = resolve(args.goal_refusal)

    calibration = read_json(calibration_path, {})
    baseline = read_json(baseline_path, {})
    baseline_residual = read_json(baseline_residual_path, {})
    execute = read_json(execute_path, {})
    refusal = read_json(refusal_path, {})
    trace_rows = read_jsonl(trace_path)
    candidate_rows = read_jsonl(candidates_path)

    candidates_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        task_id = str(row.get("task_id") or "")
        if task_id:
            candidates_by_task[task_id].append(row)

    trace_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        task_id = str(row.get("task_id") or "")
        if task_id:
            trace_by_task[task_id].append(row)

    task_rows = build_task_rows(calibration, candidates_by_task, trace_by_task)
    slice_alignment = summarize_slice_alignment(task_rows, candidates_by_task)
    category_counts = mine_category_counts(task_rows)
    candidate_summary = summarize_candidates(candidate_rows)
    baseline_summary = summarize_public_result(baseline)
    current_summary = summarize_public_result(calibration)
    baseline_residual_summary = object_field(baseline_residual, "summary")
    private_target_rows = private_residual_target_rows(
        category_counts,
        baseline_residual_summary,
        slice_alignment,
        candidate_summary,
    )

    gates = [
        gate("one_public_execution_already_recorded", execute_summary(execute, execute_path).get("executed") is True, execute_summary(execute, execute_path)),
        gate("current_goal_guarded_no_rerun_refusal_recorded", refusal_no_rerun(refusal), refusal_summary(refusal, refusal_path)),
        gate("calibration_loaded", calibration.get("policy") == "project_theseus_real_code_benchmark_graduation_v1", rel(calibration_path)),
        gate("trace_loaded", len(trace_rows) > 0, {"path": rel(trace_path), "rows": len(trace_rows)}),
        gate("candidate_manifest_loaded", len(candidate_rows) > 0, {"path": rel(candidates_path), "rows": len(candidate_rows)}),
        gate("public_content_not_embedded", True, "only aggregate counts and task hashes are emitted"),
        gate("candidate_code_not_embedded", True, "candidate code is inspected for metadata counts only and never written"),
        gate("public_tests_or_solutions_not_used_for_training", True, "public tests remain scorer-only; no training rows are written here"),
        gate("fallback_return_count_zero", int(candidate_summary["fallback_return_candidate_count"]) == 0, candidate_summary["fallback_return_candidate_count"]),
        gate("template_like_candidate_count_zero", int(candidate_summary["template_like_candidate_count"]) == 0, candidate_summary["template_like_candidate_count"]),
        gate("private_residual_target_rows_prepared", len(private_target_rows) > 0, {"rows": len(private_target_rows)}),
        gate("external_inference_zero", external_calls(calibration, execute, refusal) == 0, external_calls(calibration, execute, refusal)),
    ]
    hard_failed = [row for row in gates if not row["passed"]]
    trigger_state = "RED" if hard_failed else "YELLOW"

    return {
        "policy": "project_theseus_bounded_public_transfer_residual_mining_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "current_public_pass_count": current_summary["pass_count"],
            "current_public_task_count": current_summary["task_count"],
            "current_public_pass_rate": current_summary["pass_rate"],
            "previous_locked_baseline_pass_count": baseline_summary["pass_count"],
            "previous_locked_baseline_task_count": baseline_summary["task_count"],
            "previous_locked_baseline_pass_rate": baseline_summary["pass_rate"],
            "delta_vs_previous_locked_baseline": round(current_summary["pass_rate"] - baseline_summary["pass_rate"], 6),
            "per_card": current_summary["per_card"],
            "previous_locked_baseline_per_card": baseline_summary["per_card"],
            "residual_category_counts": category_counts,
            "dominant_current_failure": dominant_category(category_counts),
            "candidate_manifest_slice_alignment": slice_alignment,
            "candidate_manifest_score_claim_allowed": slice_alignment["missing_residual_task_count"] == 0,
            "candidate_manifest_task_count": len(candidates_by_task),
            "candidate_manifest_row_count": len(candidate_rows),
            **candidate_summary,
            "public_prompts_embedded": False,
            "public_tests_embedded": False,
            "public_solutions_embedded": False,
            "candidate_code_embedded": False,
            "training_rows_written": 0,
            "private_residual_target_rows_written": len(private_target_rows),
            "private_residual_target_manifest": rel(resolve(args.private_targets_out)),
            "external_inference_calls": 0,
        },
        "inputs": {
            "calibration": rel(calibration_path),
            "trace": rel(trace_path),
            "candidates": rel(candidates_path),
            "baseline": rel(baseline_path),
            "baseline_residual": rel(baseline_residual_path),
            "operator_execute": rel(execute_path),
            "goal_refusal": rel(refusal_path),
        },
        "one_shot_boundary": {
            "recorded_execute": execute_summary(execute, execute_path),
            "current_goal_refusal": refusal_summary(refusal, refusal_path),
            "do_not_rerun_public_calibration": True,
        },
        "category_definitions": category_definitions(),
        "task_failure_rows": task_rows,
        "prior_baseline_residual_context": {
            "source": rel(baseline_residual_path),
            "dominant_categories": baseline_residual_summary.get("dominant_categories"),
            "adapter_adjusted_dominant_categories": baseline_residual_summary.get("adapter_adjusted_dominant_categories"),
            "raw_residual_counts": baseline_residual_summary.get("raw_residual_counts"),
            "note": "Prior 34/160 residuals are context only; this report does not copy public content from that surface.",
        },
        "private_only_residual_repair_plan": private_repair_plan(
            category_counts,
            baseline_residual_summary,
            slice_alignment,
            candidate_summary,
        ),
        "private_only_residual_target_rows": private_target_rows,
        "gates": gates,
        "rules": {
            "public_calibration": "This miner never runs public calibration and reports duplicate-surface guard state to prevent score fishing.",
            "training_boundary": "Do not train on public prompts, tests, solutions, traces, score labels, or public-derived answer templates.",
            "candidate_boundary": "Candidate code from the public run is calibration evidence only and is not emitted or turned into training rows.",
            "next_public_run": "Future public calibration must use a fresh frozen surface through the governed run registry; consumed surfaces are not rerun for score fishing.",
        },
        "external_inference_calls": 0,
    }


def build_task_rows(
    calibration: dict[str, Any],
    candidates_by_task: dict[str, list[dict[str, Any]]],
    trace_by_task: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite in as_list(calibration.get("suites")):
        card_id = str(suite.get("card_id") or "unknown")
        for residual in as_list(suite.get("residuals")):
            task_id = str(residual.get("task_id") or "")
            candidates = candidates_by_task.get(task_id, [])
            traces = trace_by_task.get(task_id, [])
            task_categories = task_categories_for(
                task_id,
                candidates,
                traces,
                residual,
                set(candidates_by_task),
            )
            rows.append(
                {
                    "card_id": card_id,
                    "task_id_sha256": sha256_text(task_id),
                    "source_task_id_sha256": sha256_text(str(residual.get("source_task_id") or task_id)),
                    "residual_type": str(residual.get("type") or "unknown"),
                    "categories": task_categories,
                    "candidate_count": len(candidates),
                    "candidate_manifest_contains_task": task_id in candidates_by_task,
                    "promotable_candidate_count": sum(1 for row in candidates if promotion_eligible(row)),
                    "countable_integrity_candidate_count": sum(
                        1 for row in candidates if countable_integrity(row)
                    ),
                    "full_body_candidate_count": sum(1 for row in candidates if full_body_candidate(row)),
                    "compositional_candidate_count": sum(1 for row in candidates if compositional_candidate(row)),
                    "learned_token_candidate_count": sum(1 for row in candidates if learned_token_candidate(row)),
                    "candidate_generation_modes": sorted({str(row.get("candidate_generation_mode") or "unknown") for row in candidates}),
                    "trace_verification_stages": sorted({str(row.get("verification_stage") or "unknown") for row in traces if row.get("event") == "real_code_candidate_test"}),
                    "trace_candidate_origins": sorted({str(row.get("candidate_origin") or "unknown") for row in traces if row.get("event") == "real_code_candidate_test"}),
                    "passed": any(row.get("passed") is True for row in traces if row.get("event") == "real_code_candidate_test"),
                    "public_prompt_embedded": False,
                    "public_tests_embedded": False,
                    "public_solution_embedded": False,
                    "candidate_code_embedded": False,
                }
            )
    return rows


def task_categories_for(
    task_id: str,
    candidates: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    residual: dict[str, Any],
    candidate_task_ids: set[str],
) -> list[str]:
    categories: set[str] = set()
    residual_type = str(residual.get("type") or "")
    stages = {str(row.get("verification_stage") or "") for row in traces if row.get("event") == "real_code_candidate_test"}
    origins = {str(row.get("candidate_origin") or "") for row in traces if row.get("event") == "real_code_candidate_test"}
    returncodes = {str(row.get("returncode") or "") for row in traces if row.get("event") == "real_code_candidate_test"}
    if task_id and task_id not in candidate_task_ids:
        return ["candidate_manifest_slice_mismatch"]
    has_current_admissible_candidate = any(
        promotion_eligible(row) and countable_integrity(row) and full_body_candidate(row)
        for row in candidates
    )
    spent_run_had_no_admissible = (
        residual_type == "local_code_generation_adapter_needed"
        or "no_candidate" in stages
        or "missing_student_candidate" in origins
    )
    if spent_run_had_no_admissible and has_current_admissible_candidate:
        categories.add("spent_calibration_no_admissible_resolved_by_current_manifest")
    elif spent_run_had_no_admissible:
        categories.add("no_admissible_interface_coverage")
    if candidates and all(not promotion_eligible(row) for row in candidates):
        categories.add("selector_ranking_miss")
    if candidates and all(not full_body_candidate(row) for row in candidates):
        categories.add("return_type_shape")
    if any(code in {"124", "125"} for code in returncodes):
        if has_current_admissible_candidate:
            categories.add("spent_calibration_no_admissible_resolved_by_current_manifest")
        else:
            categories.add("timeout_runtime")
    return sorted(categories)


def summarize_slice_alignment(
    task_rows: list[dict[str, Any]],
    candidates_by_task: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    residual_hashes = {
        str(row.get("task_id_sha256") or "")
        for row in task_rows
        if row.get("task_id_sha256")
    }
    residual_task_count = len(task_rows)
    candidate_task_count = len(candidates_by_task)
    covered_task_count = sum(1 for row in task_rows if row.get("candidate_manifest_contains_task") is True)
    missing_residual_task_count = residual_task_count - covered_task_count
    candidate_task_ids = set(candidates_by_task)
    residual_task_ids_hashed = residual_hashes
    # Public task ids are not emitted in this miner. The hash-only counts below
    # are enough to prove whether a candidate manifest can be interpreted
    # against the already-spent calibration slice.
    extra_candidate_task_count = max(0, candidate_task_count - covered_task_count)
    return {
        "residual_task_count": residual_task_count,
        "candidate_manifest_task_count": candidate_task_count,
        "covered_residual_task_count": covered_task_count,
        "covered_residual_task_rate": round(covered_task_count / residual_task_count, 6) if residual_task_count else 0.0,
        "missing_residual_task_count": missing_residual_task_count,
        "extra_candidate_task_count": extra_candidate_task_count,
        "hash_only_residual_task_count": len(residual_task_ids_hashed),
        "hash_only_candidate_task_count": len(candidate_task_ids),
        "score_semantics": (
            "If missing_residual_task_count is nonzero, this candidate manifest is not the exact "
            "candidate set scored by the spent public calibration. Treat residual categories as "
            "slice-alignment diagnostics, not as a new public score or semantic failure proof."
        ),
    }


def mine_category_counts(task_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {category: 0 for category in TARGET_CATEGORIES}
    for row in task_rows:
        categories = set(str(category) for category in row["categories"])
        residual_target = private_target_for_category(str(row.get("residual_type") or ""))
        residual_category = str(residual_target.get("canonical_category") or "")
        if residual_category:
            categories.add(residual_category)
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
    return counts


def summarize_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generation_modes = Counter(str(row.get("candidate_generation_mode") or "unknown") for row in rows)
    candidate_sources = Counter(str(row.get("candidate_source") or "unknown") for row in rows)
    task_ids = {str(row.get("task_id") or "") for row in rows if row.get("task_id")}
    return {
        "candidate_generation_modes": dict(generation_modes.most_common()),
        "candidate_sources": dict(candidate_sources.most_common()),
        "candidate_tasks_with_rows": len(task_ids),
        "promotable_candidate_count": sum(1 for row in rows if promotion_eligible(row)),
        "countable_integrity_candidate_count": sum(
            1 for row in rows if countable_integrity(row)
        ),
        "full_body_token_candidate_count": sum(1 for row in rows if full_body_candidate(row)),
        "compositional_token_candidate_count": sum(1 for row in rows if compositional_candidate(row)),
        "learned_token_candidate_count": sum(1 for row in rows if learned_token_candidate(row)),
        "template_like_candidate_count": sum(1 for row in rows if row.get("template_like_candidate") is True),
        "fallback_return_candidate_count": sum(1 for row in rows if "fallback_return" in str(row.get("candidate_generation_mode") or "").lower()),
        "expression_memory_fallback_count": sum(1 for row in rows if row.get("expression_memory_fallback") is True),
    }


def summarize_public_result(report: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(report, "summary")
    task_count = int_number(summary.get("public_task_count"), summary.get("total_case_count"))
    pass_rate = float_number(summary.get("real_public_task_pass_rate"), summary.get("multi_stream_pass_rate"))
    pass_count = int(round(pass_rate * task_count))
    per_card = []
    for suite in as_list(report.get("suites")):
        case_count = int_number(suite.get("case_count"))
        multi_passed = int_number(suite.get("multi_stream_passed"))
        per_card.append(
            {
                "card_id": suite.get("card_id"),
                "task_count": case_count,
                "pass_count": multi_passed,
                "pass_rate": round(multi_passed / case_count, 6) if case_count else 0.0,
                "residual_count": int_number(suite.get("residual_count")),
                "student_candidate_benchmark_integrity_valid": suite.get("student_candidate_benchmark_integrity_valid"),
                "public_benchmark_score_claim": suite.get("public_benchmark_score_claim"),
            }
        )
    return {"task_count": task_count, "pass_count": pass_count, "pass_rate": pass_rate, "per_card": per_card}


def private_repair_plan(
    category_counts: dict[str, int],
    baseline_residual_summary: dict[str, Any],
    slice_alignment: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    current_full_body_ready = (
        int(candidate_summary.get("promotable_candidate_count") or 0) > 0
        and int(candidate_summary.get("countable_integrity_candidate_count") or 0) > 0
        and int(candidate_summary.get("full_body_token_candidate_count") or 0) > 0
    )
    if int(slice_alignment.get("missing_residual_task_count") or 0) > 0:
        plan.append(
            {
                "priority": 0,
                "target": "exact_calibration_slice_candidate_alignment",
                "why": (
                    "The supplied candidate manifest does not cover every task in the already-spent "
                    "public calibration residual slice, so no-admissible counts cannot be interpreted "
                    "as current candidate quality."
                ),
                "private_only_actions": [
                    "Freeze exact task-slice IDs before building any future one-shot public calibration packet.",
                    "Run prompt-only/private gates on the same frozen slice manifest before execution.",
                    "Treat mismatched current-vs-spent artifacts as reporting hygiene only; do not train on public task identities or score labels.",
                ],
                "public_data_boundary": "Only aggregate counts and hash-only alignment are emitted; public prompts/tests/solutions/candidate code stay out of training.",
            }
        )
    if current_full_body_ready:
        plan.append(
            {
                "priority": 1,
                "target": "current_full_body_candidate_readiness_preserved",
                "why": (
                    "The current manifest has promotion-eligible, countable, full-body learned candidates. "
                    "Because it is not the exact spent calibration candidate set, it proves readiness for a "
                    "future one-shot packet but must not be counted as a new public score."
                ),
                "evidence": {
                    "promotable_candidate_count": candidate_summary.get("promotable_candidate_count"),
                    "countable_integrity_candidate_count": candidate_summary.get("countable_integrity_candidate_count"),
                    "full_body_token_candidate_count": candidate_summary.get("full_body_token_candidate_count"),
                    "learned_token_candidate_count": candidate_summary.get("learned_token_candidate_count"),
                    "template_like_candidate_count": candidate_summary.get("template_like_candidate_count"),
                    "fallback_return_candidate_count": candidate_summary.get("fallback_return_candidate_count"),
                },
                "private_only_actions": [
                    "Keep the full-body candidate contract green on private heldout.",
                    "Freeze the exact task slice before any future public run so the candidate manifest, traces, and score can be interpreted together.",
                    "Do not train on this public-shaped candidate code or score labels.",
                ],
                "public_data_boundary": "Current prompt-only candidate metadata is readiness evidence only; no public score changes until a governed one-shot run.",
            }
        )
    else:
        plan.append(
            {
                "priority": 1,
                "target": "admissible_full_body_student_candidates",
                "why": "The inspected manifest does not yet expose promotion-eligible full-body candidates for the public-shaped slice.",
                "private_only_actions": [
                    "Build private visible-contract tasks that require full function bodies instead of return-expression wrappers.",
                    "Add an admissibility gate requiring full_body_token_candidate_count > 0 and countable_integrity_candidate_count > 0 on private heldout.",
                    "Run the gate with an empty public manifest before any future public calibration proposal.",
                ],
                "public_data_boundary": "Use only aggregate category counts and candidate provenance fields; do not copy public prompts, tests, solutions, traces, or candidate code.",
            }
        )
    plan.extend([
        {
            "priority": 2,
            "target": "selector_and_ranking_admissibility",
            "why": "The scorer reported no local admissible candidate even though the manifest had 8 generated rows per task.",
            "private_only_actions": [
                "Train/select among private candidate metadata for admissibility first, then semantic quality.",
                "Require selector reports to include no_admissible rate, oracle/pass-if-any rate, selected-pass rate, and STS-off matched controls.",
            ],
            "public_data_boundary": "Selector features must come from private contracts and candidate metadata, not public score labels.",
        },
        {
            "priority": 3,
            "target": "prior_baseline_semantic_residuals",
            "why": "The previous locked 34/160 run still exposes semantic residual pressure once candidate admissibility is restored.",
            "private_only_actions": [
                "Keep private residual v3/v4 pressure for verifier mismatch, return shape, algorithm choice, stdin/IO, and edge-case families.",
                "Treat the older baseline residual counts as aggregate context only.",
            ],
            "prior_context": {
                "dominant_categories": baseline_residual_summary.get("dominant_categories"),
                "adapter_adjusted_dominant_categories": baseline_residual_summary.get("adapter_adjusted_dominant_categories"),
                "raw_residual_counts": baseline_residual_summary.get("raw_residual_counts"),
            },
            "public_data_boundary": "Do not train on the public task identities, prompts, tests, solutions, or answer templates behind these counts.",
        },
    ])
    return plan


def private_residual_target_rows(
    category_counts: dict[str, int],
    baseline_residual_summary: dict[str, Any],
    slice_alignment: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_row(
        *,
        source: str,
        category: str,
        count: int,
        priority: int,
        source_semantics: str,
    ) -> None:
        if count <= 0:
            return
        target = private_target_for_category(category)
        if not target:
            return
        key = (source, target["canonical_category"])
        if key in seen:
            return
        seen.add(key)
        target_id_seed = f"{source}:{target['canonical_category']}:{count}:{priority}"
        rows.append(
            {
                "policy": "project_theseus_private_residual_target_v1",
                "target_id": f"private_residual_target_{sha256_text(target_id_seed)[:16]}",
                "created_utc": now(),
                "priority": priority,
                "source": source,
                "source_semantics": source_semantics,
                "source_aggregate_category": category,
                "source_aggregate_count": count,
                "canonical_category": target["canonical_category"],
                "private_curriculum_family": target["private_curriculum_family"],
                "repair_focus": target["repair_focus"],
                "required_private_evidence": target["required_private_evidence"],
                "acceptance_metric": target["acceptance_metric"],
                "forbidden_training_sources": forbidden_public_training_sources(),
                "public_content_embedded": False,
                "public_task_id_embedded": False,
                "public_prompt_embedded": False,
                "public_tests_embedded": False,
                "public_solution_embedded": False,
                "candidate_code_embedded": False,
                "training_row": False,
                "score_label_embedded": False,
                "notes": (
                    "This is aggregate private repair pressure only. It can seed private curriculum selection "
                    "or gate weighting, but it is not itself a training row and carries no public benchmark content."
                ),
            }
        )

    for category, count in sorted(category_counts.items()):
        add_row(
            source="current_spent_public_calibration_hash_only_residual_mining",
            category=category,
            count=int(count),
            priority=priority_for_current_category(category),
            source_semantics=(
                "Hash-only aggregate categories from already-spent calibration artifacts. "
                "When slice alignment is incomplete, counts are diagnostic pressure only."
            ),
        )

    for category, count in aggregate_category_pairs(baseline_residual_summary.get("adapter_adjusted_dominant_categories")):
        add_row(
            source="prior_locked_baseline_adapter_adjusted_context",
            category=category,
            count=count,
            priority=priority_for_baseline_category(category),
            source_semantics=(
                "Aggregate residual category counts from the previous locked 34/160 report. "
                "No public task identifiers, prompts, tests, solutions, traces, or candidate code are emitted."
            ),
        )

    for category, count in aggregate_category_pairs(baseline_residual_summary.get("raw_residual_counts")):
        add_row(
            source="prior_locked_baseline_raw_context",
            category=category,
            count=count,
            priority=priority_for_baseline_category(category) + 1,
            source_semantics=(
                "Raw aggregate residual category counts from the previous locked 34/160 report. "
                "Use only as private curriculum pressure, never as public task training content."
            ),
        )

    if int(slice_alignment.get("missing_residual_task_count") or 0) > 0:
        add_row(
            source="current_manifest_alignment_audit",
            category="candidate_manifest_slice_mismatch",
            count=int(slice_alignment.get("missing_residual_task_count") or 0),
            priority=0,
            source_semantics="Candidate manifest does not exactly align with the spent public residual slice.",
        )

    if int(candidate_summary.get("promotable_candidate_count") or 0) > 0:
        add_row(
            source="current_full_body_readiness_audit",
            category="spent_calibration_no_admissible_resolved_by_current_manifest",
            count=int(candidate_summary.get("promotable_candidate_count") or 0),
            priority=1,
            source_semantics="Current manifest has full-body learned candidates; this is readiness evidence only.",
        )

    rows.sort(key=lambda row: (int(row.get("priority") or 999), str(row.get("canonical_category")), str(row.get("source"))))
    return rows


def private_target_for_category(category: str) -> dict[str, Any]:
    normalized = category.strip().lower()
    aliases = {
        "return_shape": "return_type_shape",
        "type_handling": "return_type_shape",
        "algorithmic_planning": "algorithm_choice",
        "edge_case": "edge_cases",
        "no_admissible_candidate_regression": "selector_ranking_miss",
        "local_code_generation_adapter_needed": "selector_ranking_miss",
        "no_admissible_interface_coverage": "selector_ranking_miss",
    }
    canonical = aliases.get(normalized, normalized)
    targets = {
        "candidate_manifest_slice_mismatch": {
            "private_curriculum_family": "calibration_freeze_and_manifest_alignment_private_audit",
            "repair_focus": [
                "freeze exact candidate manifest before a one-shot public run",
                "verify task slice and candidate manifest hashes match before execution",
                "refuse score claims from mismatched artifacts",
            ],
            "required_private_evidence": [
                "candidate manifest hash is generated before public execution",
                "private prompt-only candidate generation reproduces the frozen manifest contract",
                "residual miner reports missing_residual_task_count=0 for future spent slice",
            ],
            "acceptance_metric": "future one-shot packet has exact candidate/slice alignment before any public run",
        },
        "spent_calibration_no_admissible_resolved_by_current_manifest": {
            "private_curriculum_family": "full_body_candidate_readiness_private_gate",
            "repair_focus": [
                "preserve full-body learned candidate generation",
                "keep countable integrity candidates available under frozen manifest",
                "separate readiness evidence from public score claims",
            ],
            "required_private_evidence": [
                "full_body_token_candidate_count > 0",
                "countable_integrity_candidate_count > 0",
                "template_like_candidate_count=0",
                "fallback_return_candidate_count=0",
            ],
            "acceptance_metric": "private full-body candidate readiness remains GREEN with no fallback/template leakage",
        },
        "selector_ranking_miss": {
            "private_curriculum_family": "selector_ranker_admissibility_private_curriculum",
            "repair_focus": [
                "first-rank admissibility",
                "STS and non-STS matched candidate budget selection",
                "return-shape-aware candidate ordering",
            ],
            "required_private_evidence": [
                "selected-pass rate improves under equal candidate budget",
                "oracle/pass-if-any does not regress",
                "no-admissible rate decreases",
                "fallback count remains zero",
            ],
            "acceptance_metric": "private selected-pass lift with no regression in oracle/pass-if-any",
        },
        "return_type_shape": {
            "private_curriculum_family": "return_shape_and_type_handling_private_curriculum",
            "repair_focus": [
                "return container shape",
                "type conversion",
                "string/list/dict boundary handling",
                "first-rank shape-compatible selection",
            ],
            "required_private_evidence": [
                "shape-compatible selected candidate rate improves",
                "string-processing and type-handling heldout families do not regress",
                "AST parse and full-body obligations stay GREEN",
            ],
            "acceptance_metric": "private return/type/string selected-pass lift without template or fallback candidates",
        },
        "algorithm_choice": {
            "private_curriculum_family": "algorithm_choice_private_curriculum",
            "repair_focus": [
                "algorithm family selection",
                "multi-step control flow",
                "edge-case branch choice",
            ],
            "required_private_evidence": [
                "private algorithm-family heldout selected-pass rate improves",
                "wrong-answer residual count decreases",
                "runtime timeout rate remains zero",
            ],
            "acceptance_metric": "private semantic wrong-answer residuals decrease under same-seed ablation",
        },
        "verifier_mismatch": {
            "private_curriculum_family": "verifier_contract_private_curriculum",
            "repair_focus": [
                "contract interpretation",
                "edge-case expected behavior",
                "verifier-compatible output semantics",
            ],
            "required_private_evidence": [
                "private verifier-mismatch residual family improves",
                "verifier mismatch does not hide public tests or solutions",
                "candidate code remains full-body learned-token generated",
            ],
            "acceptance_metric": "private verifier mismatch residual count falls with no public-content leakage",
        },
        "io_contract_stdin": {
            "private_curriculum_family": "io_contract_private_curriculum",
            "repair_focus": [
                "stdin/stdout adapter shape",
                "function-vs-script boundary",
                "input parsing",
            ],
            "required_private_evidence": [
                "private IO contract heldout improves",
                "entry point and return contract are preserved",
            ],
            "acceptance_metric": "private IO adapter selected-pass lift with parse/shape gates green",
        },
        "edge_cases": {
            "private_curriculum_family": "edge_case_semantic_private_curriculum",
            "repair_focus": [
                "boundary values",
                "empty/singleton inputs",
                "duplicate handling",
                "ordering invariants",
            ],
            "required_private_evidence": [
                "private edge-case heldout selected-pass rate improves",
                "mainline task pass rate does not regress",
            ],
            "acceptance_metric": "private edge-case selected-pass lift without mainline regression",
        },
        "external_dependency_missing": {
            "private_curriculum_family": "dependency_free_candidate_private_curriculum",
            "repair_focus": [
                "standard-library-only candidate generation",
                "dependency import avoidance",
                "explicit UNKNOWN/UNSUPPORTED instead of unavailable package assumptions",
            ],
            "required_private_evidence": [
                "candidate dependency-unavailable count decreases",
                "standard-library-only private heldout pass rate does not regress",
                "no fallback returns or template candidates are introduced",
            ],
            "acceptance_metric": "private dependency-free candidate selected-pass lift with dependency fault rate reduced",
        },
        "timeout_runtime": {
            "private_curriculum_family": "runtime_efficiency_private_curriculum",
            "repair_focus": [
                "asymptotic choice",
                "loop bounds",
                "early exits",
            ],
            "required_private_evidence": [
                "private timeout residual rate decreases",
                "runtime cost is reported",
            ],
            "acceptance_metric": "private runtime timeout rate decreases with no semantic regression",
        },
        "parsing_syntax": {
            "private_curriculum_family": "syntax_and_parser_mask_private_curriculum",
            "repair_focus": [
                "grammar mask validity",
                "AST parseability",
                "indentation and block closure",
            ],
            "required_private_evidence": [
                "AST parse rate remains 1.0",
                "syntax failure count decreases",
                "full-body candidate rate does not regress",
            ],
            "acceptance_metric": "private syntax pass rate improves or remains saturated while semantic pass improves",
        },
    }
    target = targets.get(canonical)
    if not target:
        return {}
    return {"canonical_category": canonical, **target}


def aggregate_category_pairs(value: Any) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for item in as_list(value):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), int_number(item[1])))
    return pairs


def priority_for_current_category(category: str) -> int:
    order = {
        "candidate_manifest_slice_mismatch": 0,
        "spent_calibration_no_admissible_resolved_by_current_manifest": 1,
        "selector_ranking_miss": 2,
        "return_type_shape": 3,
        "algorithm_choice": 4,
        "verifier_mismatch": 5,
        "edge_cases": 6,
        "io_contract_stdin": 7,
        "timeout_runtime": 8,
    }
    return order.get(category, 50)


def priority_for_baseline_category(category: str) -> int:
    canonical = private_target_for_category(category).get("canonical_category", category)
    order = {
        "verifier_mismatch": 2,
        "selector_ranking_miss": 3,
        "return_type_shape": 4,
        "algorithm_choice": 5,
        "edge_cases": 6,
        "io_contract_stdin": 7,
        "timeout_runtime": 8,
    }
    return order.get(str(canonical), 50)


def forbidden_public_training_sources() -> list[str]:
    return [
        "public_benchmark_prompt",
        "public_visible_tests",
        "public_hidden_tests",
        "public_solution",
        "public_trace",
        "public_answer_template",
        "public_task_identity",
        "public_candidate_code_from_calibration",
        "public_score_label",
    ]


def category_definitions() -> dict[str, str]:
    return {
        "parsing_syntax": "Candidate reached parse/syntax verification and failed there.",
        "return_type_shape": "Candidate family or verifier result indicates wrapper/return-shape/interface mismatch.",
        "algorithm_choice": "Candidate reached semantic tests and appears to choose the wrong algorithmic plan.",
        "io_contract_stdin": "Candidate reached an IO/stdin contract and failed the adapter shape.",
        "edge_cases": "Candidate passed broad shape but failed boundary/metamorphic cases.",
        "external_dependency_missing": "Candidate assumes a dependency unavailable in the governed execution environment.",
        "verifier_mismatch": "Verifier behavior disagrees with the candidate's assumed contract.",
        "no_admissible_interface_coverage": "No admissible local student candidate reached scoring for the task.",
        "candidate_manifest_slice_mismatch": "The candidate manifest being inspected is not aligned to the already-spent public residual slice.",
        "timeout_runtime": "Candidate failed by timeout or runtime execution budget.",
        "selector_ranking_miss": "Candidate rows exist, but selector/admissibility rules did not choose a countable candidate.",
        "spent_calibration_no_admissible_resolved_by_current_manifest": (
            "The spent calibration recorded no admissible local candidate, but the current manifest now has "
            "promotion-eligible full-body candidates for that hashed task. This is readiness evidence only, "
            "not a changed public score."
        ),
    }


def promotion_eligible(row: dict[str, Any]) -> bool:
    return bool(row.get("benchmark_promotion_eligible") is True or row.get("benchmark_promotion_eligible_candidate") is True)


def countable_integrity(row: dict[str, Any]) -> bool:
    return bool(object_field(row, "benchmark_integrity").get("may_count_for_public_benchmark_promotion") is True)


def full_body_candidate(row: dict[str, Any]) -> bool:
    return bool(row.get("full_body_token_candidate") is True or row.get("candidate_program_scope") == "full_function_body")


def compositional_candidate(row: dict[str, Any]) -> bool:
    return bool(row.get("compositional_token_candidate") is True)


def learned_token_candidate(row: dict[str, Any]) -> bool:
    return bool(
        row.get("token_level_code_generation_learned") is True
        or row.get("grammar_masked_learned_token_candidate") is True
    )


def execute_summary(report: dict[str, Any], path: Path = DEFAULT_EXECUTE) -> dict[str, Any]:
    summary = object_field(report, "summary")
    return {
        "path": rel(path),
        "trigger_state": report.get("trigger_state"),
        "mode": summary.get("mode"),
        "executed": summary.get("executed"),
        "run_returncode": summary.get("run_returncode"),
        "output_path": summary.get("output_path"),
        "operator_lock_present_after": summary.get("operator_lock_present_after"),
        "run_registry_execution_enabled": summary.get("run_registry_execution_enabled"),
        "surface_not_consumed": summary.get("surface_not_consumed"),
        "run_registry_reasons": summary.get("run_registry_reasons"),
    }


def refusal_summary(report: dict[str, Any], path: Path = DEFAULT_REFUSAL) -> dict[str, Any]:
    summary = object_field(report, "summary")
    run_result = object_field(report, "run_result")
    return {
        "path": rel(path),
        "trigger_state": report.get("trigger_state"),
        "mode": summary.get("mode"),
        "executed": summary.get("executed"),
        "reason": run_result.get("reason"),
        "failed_requirements": run_result.get("failed_requirements"),
        "operator_lock_present_after": summary.get("operator_lock_present_after"),
        "run_registry_execution_enabled": summary.get("run_registry_execution_enabled"),
        "surface_not_consumed": summary.get("surface_not_consumed"),
        "run_registry_reasons": summary.get("run_registry_reasons"),
    }


def refusal_no_rerun(report: dict[str, Any]) -> bool:
    summary = object_field(report, "summary")
    run_result = object_field(report, "run_result")
    registry_reasons = set(str(item) for item in as_list(summary.get("run_registry_reasons")))
    rerun_prevented = bool(
        summary.get("operator_lock_present_after") is True
        or (
            summary.get("run_registry_execution_enabled") is True
            and (
                summary.get("surface_not_consumed") is False
                or "surface_consumed_or_per_surface_limit_reached" in registry_reasons
            )
        )
    )
    dry_run_no_rerun = (
        summary.get("mode") == "dry_run"
        and summary.get("executed") is False
        and summary.get("would_execute") is False
        and rerun_prevented
        and (
            summary.get("surface_not_consumed") is False
            or summary.get("output_absent_before_run") is False
            or summary.get("trace_absent_before_run") is False
            or "surface_consumed_or_per_surface_limit_reached" in registry_reasons
        )
    )
    return (
        dry_run_no_rerun
        or (
            summary.get("mode") == "execute"
            and summary.get("executed") is False
            and run_result.get("status") == "not_run"
            and "output_absent" in as_list(run_result.get("failed_requirements"))
            and rerun_prevented
        )
    )


def dominant_category(counts: dict[str, int]) -> list[Any]:
    if not counts:
        return ["", 0]
    name, count = max(counts.items(), key=lambda item: item[1])
    return [name, count]


def external_calls(*reports: dict[str, Any]) -> int:
    return max([0, *[int_number(report.get("external_inference_calls"), object_field(report, "summary").get("external_inference_calls")) for report in reports]])


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    alignment = object_field(summary, "candidate_manifest_slice_alignment")
    lines = [
        "# Bounded Public Transfer Residual Mining v1",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Candidate manifest score claim allowed: `{summary.get('candidate_manifest_score_claim_allowed')}`",
        f"- Spent calibration score in artifact: `{summary.get('current_public_pass_count')}/{summary.get('current_public_task_count')}` = `{summary.get('current_public_pass_rate')}`",
        f"- Previous locked baseline: `{summary.get('previous_locked_baseline_pass_count')}/{summary.get('previous_locked_baseline_task_count')}` = `{summary.get('previous_locked_baseline_pass_rate')}`",
        f"- Delta: `{summary.get('delta_vs_previous_locked_baseline')}`",
        f"- Dominant current failure: `{summary.get('dominant_current_failure')}`",
        f"- Candidate/residual slice coverage: `{alignment.get('covered_residual_task_count')}/{alignment.get('residual_task_count')}` = `{alignment.get('covered_residual_task_rate')}`",
        f"- Missing residual tasks in candidate manifest: `{alignment.get('missing_residual_task_count')}`",
        f"- Fallback return candidates: `{summary.get('fallback_return_candidate_count')}`",
        f"- Template-like candidates: `{summary.get('template_like_candidate_count')}`",
        f"- Training rows written: `{summary.get('training_rows_written')}`",
        "",
        "## Per Card",
    ]
    for row in as_list(summary.get("per_card")):
        lines.append(f"- `{row.get('card_id')}`: `{row.get('pass_count')}/{row.get('task_count')}` = `{row.get('pass_rate')}`")
    lines.extend(["", "## Residual Categories"])
    for name, count in object_field(summary, "residual_category_counts").items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Repair Plan"])
    for row in as_list(report.get("private_only_residual_repair_plan")):
        lines.append(f"- P{row.get('priority')} `{row.get('target')}`: {row.get('why')}")
    lines.extend(["", "## Gates"])
    for row in as_list(report.get("gates")):
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}`")
    lines.append("")
    return "\n".join(lines)


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    field = value.get(key) if isinstance(value, dict) else {}
    return field if isinstance(field, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def int_number(*values: Any) -> int:
    for value in values:
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    return 0


def float_number(*values: Any) -> float:
    for value in values:
        try:
            if value is not None:
                return float(value)
        except Exception:
            continue
    return 0.0


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
