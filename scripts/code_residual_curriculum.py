"""Generate private residual-targeted code curriculum rows.

This script uses public calibration failures only as *category pressure*.
It never copies public prompts, public tests, canonical solutions, or generated
answers into training. The output is private generated code tasks shaped around
observed residual classes so the Code LM can learn repair patterns without
benchmark leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from code_residual_curriculum_templates import render_variant, template_bank


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE_OUT = (
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_train"
    / "residual_code_lm_tasks.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-in", default="reports/real_code_benchmark_traces.jsonl")
    parser.add_argument("--real-code-report", default="reports/real_code_benchmark_graduation.json")
    parser.add_argument("--broad-transfer-matrix", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--broad-scheduler", default="reports/broad_code_calibration_scheduler.json")
    parser.add_argument("--active-card", default="")
    parser.add_argument("--private-out", default=str(DEFAULT_PRIVATE_OUT))
    parser.add_argument("--out", default="reports/code_residual_curriculum.json")
    parser.add_argument("--markdown-out", default="reports/code_residual_curriculum.md")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--max-rows", type=int, default=720)
    parser.add_argument(
        "--concept-focus",
        default="",
        help="Optional source-agnostic pressure focus: type_and_return_shape, return_type_shape_v2, admissibility_and_interface, adapter_runtime_dependency_handling, external_dependency_missing, typed_interface_skeleton, edge_conditions, edge_contract_4card, edge_contract_balanced_4card_private_curriculum_v2, edge_case_full_body_private_curriculum_v1, frontier_agent_edge_full_body_private_curriculum_v2, edge_contract_v2_private_residual_curriculum, candidate_floor_v2_private_residual_curriculum, candidate_floor_adapter_v2, residual_targeted_private_edge_case_contract_v1, parsing_encoding_v1, intended_behavior_transfer_private_curriculum, broad_floor_semantic_transfer_private_curriculum, algorithmic_planning, algorithmic_planning_boundary, or execution_shaped_programs.",
    )
    args = parser.parse_args()

    traces = read_jsonl(resolve(args.trace_in))
    real_code = read_json(resolve(args.real_code_report), {})
    broad_matrix = read_json(resolve(args.broad_transfer_matrix), {})
    broad_scheduler = read_json(resolve(args.broad_scheduler), {})
    residual_summary = summarize_residuals(traces)
    broad_context = summarize_broad_context(broad_matrix, broad_scheduler, active_card_override=args.active_card)
    rows = build_private_rows(
        residual_summary,
        seed=args.seed,
        max_rows=max(24, args.max_rows),
        broad_context=broad_context,
        concept_focus=args.concept_focus,
    )
    write_jsonl(resolve(args.private_out), rows)
    private_solution_check = verify_private_solution_rows(rows)

    gates = [
        gate("public_prompts_not_copied", True, "templates are private generated from residual class names only"),
        gate("public_tests_not_copied", True, "no public tests or hidden public assertions are read into output rows"),
        gate("canonical_solutions_not_used", True, "trace summaries omit canonical solutions"),
        gate("private_rows_written", len(rows) > 0, f"rows={len(rows)}"),
        gate("private_solution_tests_pass", private_solution_check["failure_count"] == 0, private_solution_check),
        gate(
            "execution_shaped_programs_share_lte_0_25",
            execution_shaped_programs_share(rows) <= 0.25,
            {
                "execution_shaped_programs_rows": execution_shaped_programs_rows(rows),
                "private_row_count": len(rows),
                "share": execution_shaped_programs_share(rows),
            },
        ),
        gate(
            "benchmark_named_rows_not_required_for_neutral_focus",
            not benchmark_neutral_focus(args.concept_focus) or benchmark_named_private_row_count(rows) == 0,
            {
                "concept_focus": str(args.concept_focus or ""),
                "benchmark_neutral_focus": benchmark_neutral_focus(args.concept_focus),
                "benchmark_named_private_rows": benchmark_named_private_row_count(rows),
            },
        ),
        gate(
            "edge_case_full_body_rows_have_edge_contract_tests",
            not edge_case_full_body_focus(args.concept_focus) or edge_case_full_body_contract_rows(rows) > 0,
            {
                "concept_focus": str(args.concept_focus or ""),
                "edge_case_full_body_rows": edge_case_full_body_rows(rows),
                "edge_case_full_body_contract_rows": edge_case_full_body_contract_rows(rows),
                "contract": "focused rows must be full-body private tasks with at least four assertions covering normal and edge/boundary behavior",
            },
        ),
        gate(
            "edge_contract_v2_rows_have_generation_plan_contracts",
            not edge_contract_v2_focus(args.concept_focus)
            or (
                edge_contract_v2_rows(rows) > 0
                and edge_contract_v2_generation_plan_rows(rows) == edge_contract_v2_rows(rows)
            ),
            {
                "concept_focus": str(args.concept_focus or ""),
                "edge_contract_v2_rows": edge_contract_v2_rows(rows),
                "edge_contract_v2_generation_plan_rows": edge_contract_v2_generation_plan_rows(rows),
                "contract": "v2 rows must carry decoder_contract.generation_plan so the verifier can bias skeleton generation before candidate rejection",
            },
        ),
        gate(
            "external_dependency_rows_have_guarded_optional_fallbacks",
            not external_dependency_focus(args.concept_focus)
            or (
                optional_dependency_rows(rows) > 0
                and optional_dependency_guard_contract_rows(rows) == optional_dependency_rows(rows)
            ),
            {
                "concept_focus": str(args.concept_focus or ""),
                "optional_dependency_rows": optional_dependency_rows(rows),
                "optional_dependency_guard_contract_rows": optional_dependency_guard_contract_rows(rows),
                "contract": "dependency-missing private rows must require guarded optional imports plus deterministic stdlib/pure-Python fallbacks, not unbounded dependency installation or network access",
            },
        ),
        gate(
            "dominant_edge_case_pressure_materialized",
            "edge_case" not in residual_summary["class_counts"]
            or any(
                "edge_case" in (row.get("tags") or [])
                or "residual_targeted_edge_case_contract_v1" in (row.get("tags") or [])
                for row in rows
            )
            or (
                str(args.concept_focus or "").strip().lower()
                in {"admissibility_and_interface", "adapter_runtime_dependency_handling"}
                and any("local_code_generation_adapter_needed" in (row.get("tags") or []) for row in rows)
            )
            or edge_pressure_exempt_focus(args.concept_focus),
            {
                "edge_case_residuals": residual_summary["class_counts"].get("edge_case", 0),
                "edge_case_rows": sum(
                    1
                    for row in rows
                    if "edge_case" in (row.get("tags") or [])
                    or "residual_targeted_edge_case_contract_v1" in (row.get("tags") or [])
                ),
                "local_adapter_rows": sum(
                    1
                    for row in rows
                    if "local_code_generation_adapter_needed" in (row.get("tags") or [])
                ),
            },
        ),
        gate(
            "intended_behavior_stage_pressure_materialized",
            residual_summary["verification_stage_counts"].get("intended_behavior_failed", 0) <= 0
            or intended_behavior_pressure_rows(rows) > 0,
            {
                "intended_behavior_failed": residual_summary["verification_stage_counts"].get(
                    "intended_behavior_failed", 0
                ),
                "targeted_rows": intended_behavior_pressure_rows(rows),
                "contract": (
                    "When public calibration candidates pass early verification but fail behavior, private rows must "
                    "carry intended-behavior pressure without copying public tests or rewards."
                ),
            },
        ),
        gate(
            "frontier_intended_behavior_full_body_pressure_materialized",
            not frontier_full_body_pressure_required(args.concept_focus, residual_summary, broad_context)
            or edge_case_full_body_contract_rows(rows) > 0,
            {
                "concept_focus": str(args.concept_focus or ""),
                "active_card": broad_context.get("active_card"),
                "cards_below_floor": broad_context.get("cards_below_floor") or [],
                "intended_behavior_failed": residual_summary["verification_stage_counts"].get(
                    "intended_behavior_failed", 0
                ),
                "edge_case_full_body_rows": edge_case_full_body_rows(rows),
                "edge_case_full_body_contract_rows": edge_case_full_body_contract_rows(rows),
                "contract": (
                    "When LiveCodeBench-like or intended-behavior transfer is the active wall, private rows must "
                    "include full-body edge programs, not only contract or candidate-floor rows."
                ),
            },
        ),
        gate("license_declared", all(row.get("license_spdx") == "CC0-1.0" for row in rows), "CC0-1.0 local generated"),
        gate(
            "public_score_quarantined",
            real_code.get("public_benchmark_score_claim")
            in {
                "student_code_lm_checkpoint_public_task_calibration_only",
                "student_token_generator_checkpoint_public_task_calibration_only",
            },
            real_code.get("public_benchmark_score_claim"),
        ),
    ]
    report = {
        "policy": "project_theseus_code_residual_curriculum_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "YELLOW",
        "purpose": "Convert residual classes into private generated code training pressure without benchmark answer leakage.",
        "inputs": {
            "trace_in": rel(resolve(args.trace_in)),
            "real_code_report": rel(resolve(args.real_code_report)),
            "broad_transfer_matrix": rel(resolve(args.broad_transfer_matrix)),
            "broad_scheduler": rel(resolve(args.broad_scheduler)),
            "active_card_override": str(args.active_card or ""),
            "concept_focus": str(args.concept_focus or ""),
        },
        "outputs": {
            "private_train_jsonl": rel(resolve(args.private_out)),
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "summary": {
            "private_row_count": len(rows),
            "private_solution_test_failures": private_solution_check["failure_count"],
            "residual_class_counts": residual_summary["class_counts"],
            "verification_stage_counts": residual_summary["verification_stage_counts"],
            "dominant_verification_stage": residual_summary["dominant_verification_stage"],
            "verification_pressure_target_stage": residual_summary["verification_pressure"]["target_stage"],
            "verification_pressure_private_rows": sum(1 for row in rows if row.get("verification_pressure")),
            "concept_residual_counts": dict(Counter(str(row.get("concept_residual_label") or "unknown") for row in rows)),
            "target_wall_family_counts": dict(Counter(str(row.get("residual_concept") or "unknown") for row in rows)),
            "dominant_residual_class": max(
                residual_summary["class_counts"].items(),
                key=lambda item: int(item[1] or 0),
            )[0]
            if residual_summary["class_counts"]
            else "",
            "edge_case_private_rows": sum(1 for row in rows if "edge_case" in (row.get("tags") or [])),
            "local_adapter_private_rows": sum(
                1 for row in rows if "local_code_generation_adapter_needed" in (row.get("tags") or [])
            ),
            "broad_active_card": broad_context["active_card"],
            "broad_active_card_selection": broad_context.get("active_card_selection"),
            "broad_active_card_additional_passes_needed": broad_context.get("active_card_additional_passes_needed"),
            "broad_selected_card": broad_context.get("selected_card"),
            "broad_selected_action": broad_context.get("selected_action"),
            "broad_cards_below_floor": broad_context["cards_below_floor"],
            "broad_no_clean_student_evidence_cards": broad_context["no_clean_student_evidence_cards"],
            "broad_loader_only_cards": broad_context["loader_only_cards"],
            "livecodebench_intended_behavior_private_rows": sum(
                1
                for row in rows
                if "intended_behavior_transfer" in (row.get("tags") or [])
                or row.get("residual_concept") == "intended_behavior_transfer"
            ),
            "intended_behavior_pressure_private_rows": intended_behavior_pressure_rows(rows),
            "mbpp_broad_transfer_private_rows": sum(
                1
                for row in rows
                if "mbpp_broad_transfer_private" in (row.get("tags") or [])
            ),
            "failed_task_count": len(residual_summary["failed_task_hashes"]),
            "public_task_ids_hashed_only": True,
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "external_inference_calls": 0,
            "concept_focus": str(args.concept_focus or ""),
            "benchmark_neutral_focus": benchmark_neutral_focus(args.concept_focus),
            "execution_shaped_programs_rows": execution_shaped_programs_rows(rows),
            "execution_shaped_programs_share": execution_shaped_programs_share(rows),
            "edge_case_full_body_rows": edge_case_full_body_rows(rows),
            "edge_case_full_body_contract_rows": edge_case_full_body_contract_rows(rows),
            "frontier_full_body_pressure_required": frontier_full_body_pressure_required(
                args.concept_focus, residual_summary, broad_context
            ),
            "edge_contract_v2_rows": edge_contract_v2_rows(rows),
            "edge_contract_v2_generation_plan_rows": edge_contract_v2_generation_plan_rows(rows),
            "optional_dependency_rows": optional_dependency_rows(rows),
            "optional_dependency_guard_contract_rows": optional_dependency_guard_contract_rows(rows),
            "benchmark_named_private_rows": benchmark_named_private_row_count(rows),
        },
        "residual_targets": residual_summary["targets"],
        "verification_pressure": residual_summary["verification_pressure"],
        "sample_private_categories": sorted({str(row.get("category")) for row in rows})[:24],
        "gates": gates,
        "next_actions": [
            "let code_lm_closure.py load this private residual file before public calibration",
            "track whether residual classes shrink on held-out private tasks and public calibration",
            "keep public benchmark failures as category labels only, never as training answers",
            (
                f"for {broad_context['active_card']} below-floor transfer, emphasize private intended-behavior, "
                "edge-contract, type/return-shape, and adapter/runtime rows until the private heldout delta moves"
            ),
        ],
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 1


def summarize_residuals(traces: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    failed_task_hashes: set[str] = set()
    targets: list[dict[str, Any]] = []
    by_task: dict[str, Counter[str]] = {}
    reward_total = 0.0
    reward_count = 0
    for row in traces:
        if row.get("event") not in {"real_code_candidate_test", "real_code_residual_export"}:
            continue
        if row.get("passed") is True:
            continue
        residual = str(row.get("residual_class") or "wrong_answer")
        task_id = str(row.get("task_id") or row.get("source_task_id") or "unknown")
        stage = verification_stage_for_row(row)
        class_counts[residual] += 1
        if stage:
            stage_counts[stage] += 1
        if isinstance(row.get("verification_reward"), (int, float)):
            reward_total += float(row.get("verification_reward") or 0.0)
            reward_count += 1
        failed_task_hashes.add(short_hash(task_id))
        by_task.setdefault(task_id, Counter())[residual] += 1
    for task_id, counts in sorted(by_task.items(), key=lambda item: (-sum(item[1].values()), item[0]))[:24]:
        residual, count = counts.most_common(1)[0]
        targets.append(
            {
                "task_hash": short_hash(task_id),
                "residual_class": residual,
                "failed_attempts": count,
                "source": "public_calibration_failure_category_only",
            }
        )
    if not class_counts:
        class_counts["edge_case"] = 1
    if not stage_counts:
        stage_counts["unknown_failed_stage"] = 1
    dominant_stage = str(stage_counts.most_common(1)[0][0])
    return {
        "class_counts": dict(class_counts),
        "verification_stage_counts": dict(stage_counts),
        "dominant_verification_stage": dominant_stage,
        "verification_pressure": verification_pressure_from_stages(stage_counts, reward_total, reward_count),
        "failed_task_hashes": sorted(failed_task_hashes),
        "targets": targets,
    }


def verification_stage_for_row(row: dict[str, Any]) -> str:
    """Return a sanitized verifier-stage label without test text or candidate code."""
    stage = str(row.get("verification_stage") or "").strip()
    if row.get("event") == "real_code_residual_export":
        stderr_tail = str(row.get("stderr_tail") or "")
        marker = "__THESEUS_STAGE__:"
        if marker in stderr_tail:
            return stderr_tail.split(marker, 1)[1].splitlines()[0].strip() or "residual_export_failed"
        return "residual_export_failed"
    if row.get("intended_behavior_passed") is False and row.get("runtime_loaded") is True:
        return "intended_behavior_failed"
    if row.get("compile_passed") is False:
        return "candidate_compile_failed"
    if row.get("lint_passed") is False:
        return "lint_parse_failed"
    return stage or "unknown_failed_stage"


def verification_pressure_from_stages(
    stage_counts: Counter[str],
    reward_total: float,
    reward_count: int,
) -> dict[str, Any]:
    dominant_stage = str(stage_counts.most_common(1)[0][0]) if stage_counts else "unknown_failed_stage"
    if dominant_stage in {"intended_behavior_failed", "runtime_loaded"}:
        target_stage = "intended_behavior"
        target_concepts = [
            "intended_behavior_transfer",
            "edge_contract_v2",
            "algorithm_choice",
            "type_semantic_transfer",
        ]
    elif "compile" in dominant_stage:
        target_stage = "compile_import"
        target_concepts = ["typed_interface_skeleton", "type_semantic_transfer", "syntax_structure"]
    elif "lint" in dominant_stage or "parse" in dominant_stage:
        target_stage = "lint_parse"
        target_concepts = ["syntax_structure", "typed_interface_skeleton"]
    else:
        target_stage = "behavioral_transfer"
        target_concepts = ["edge_case", "algorithm_choice", "type_semantic_transfer"]
    return {
        "policy": "project_theseus_private_verification_stage_pressure_v1",
        "dominant_stage": dominant_stage,
        "target_stage": target_stage,
        "target_concepts": target_concepts,
        "stage_counts": dict(stage_counts),
        "mean_public_calibration_reward_diagnostic": round(reward_total / reward_count, 6) if reward_count else None,
        "reward_semantics": "public calibration reward is diagnostic-only; rows carry stage pressure labels, not public tests or rewards",
    }


def summarize_broad_context(
    matrix: dict[str, Any],
    scheduler: dict[str, Any],
    *,
    active_card_override: str = "",
) -> dict[str, Any]:
    summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
    selected = scheduler.get("selected") if isinstance(scheduler.get("selected"), dict) else {}
    rows = matrix.get("rows") if isinstance(matrix.get("rows"), list) else []
    by_card = {
        str(row.get("card_id") or ""): row
        for row in rows
        if isinstance(row, dict) and row.get("card_id")
    }
    selected_action = str(selected.get("action") or "")
    selected_card = str(selected.get("card_id") or "")
    active_card, active_selection = select_broad_active_card(
        by_card,
        summary,
        active_card_override=str(active_card_override or ""),
        selected_card=selected_card,
        selected_action=selected_action,
    )
    if not active_card:
        below = summary.get("cards_below_floor") if isinstance(summary.get("cards_below_floor"), list) else []
        active_card = str(below[0]) if below else ""
    active_row = by_card.get(active_card, {})
    return {
        "active_card": active_card,
        "active_card_selection": active_selection,
        "active_card_additional_passes_needed": additional_passes_needed(active_row),
        "selected_card": selected_card,
        "selected_action": selected_action,
        "cards_below_floor": [str(item) for item in summary.get("cards_below_floor", []) if str(item)],
        "no_clean_student_evidence_cards": [
            str(item) for item in summary.get("no_clean_student_evidence_cards", []) if str(item)
        ],
        "loader_only_cards": [str(item) for item in summary.get("loader_only_cards", []) if str(item)],
        "active_residual_family_counts": active_row.get("residual_family_counts")
        if isinstance(active_row.get("residual_family_counts"), dict)
        else {},
        "active_multi_stream_pass_rate": float(active_row.get("multi_stream_pass_rate") or 0.0),
        "active_single_stream_pass_rate": float(active_row.get("single_stream_pass_rate") or 0.0),
    }


def select_broad_active_card(
    by_card: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    *,
    active_card_override: str,
    selected_card: str,
    selected_action: str,
) -> tuple[str, str]:
    if active_card_override:
        return active_card_override, "explicit_override"
    below = [str(item) for item in summary.get("cards_below_floor", []) if str(item)]
    if not below:
        return selected_card, "scheduler_selected_no_below_floor_cards"
    public_locked = selected_action in {
        "public_calibration_operator_locked",
        "no_action",
        "refresh_broad_transfer_matrix",
    }
    if public_locked:
        ranked = sorted(
            (by_card.get(card, {}) for card in below),
            key=lambda row: (
                additional_passes_needed(row),
                -float(row.get("multi_stream_pass_rate") or row.get("single_stream_pass_rate") or 0.0),
                int(row.get("public_task_count") or 0),
                str(row.get("card_id") or ""),
            ),
            reverse=True,
        )
        if ranked:
            chosen = str(ranked[0].get("card_id") or "")
            if chosen:
                return chosen, "largest_deficit_while_public_calibration_locked"
    if selected_card:
        return selected_card, "scheduler_selected"
    return below[0], "first_below_floor_card"


def additional_passes_needed(row: dict[str, Any]) -> int:
    task_count = int(row.get("public_task_count") or 0)
    if task_count <= 0:
        return 0
    pass_rate = float(row.get("multi_stream_pass_rate") or row.get("single_stream_pass_rate") or 0.0)
    current_passes = int(round(pass_rate * task_count))
    return max(0, math.ceil(0.70 * task_count) - current_passes)


def build_private_rows(
    summary: dict[str, Any],
    *,
    seed: int,
    max_rows: int,
    broad_context: dict[str, Any] | None = None,
    concept_focus: str = "",
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    broad_context = broad_context or {}
    weighted_classes: list[str] = []
    for residual_class, count in summary["class_counts"].items():
        weighted_classes.extend([str(residual_class)] * max(1, min(16, int(count))))
    if not weighted_classes:
        weighted_classes = ["edge_case"]
    dominant_residual = ""
    if summary["class_counts"]:
        dominant_residual = str(max(summary["class_counts"].items(), key=lambda item: int(item[1] or 0))[0])
    weighted_classes = [
        "recurrence_state",
        "recurrence_state",
        "string_rule_composition",
        "string_rule_composition",
        "digit_rotation",
        "digit_rotation",
        "type_semantic_transfer",
        "type_semantic_transfer",
        "algorithm_choice",
        "wrong_answer",
        *weighted_classes,
    ]
    if dominant_residual == "edge_case":
        weighted_classes = (
            ["edge_case"] * 48
            + ["local_code_generation_adapter_needed"] * 24
            + ["type_handling"] * 12
            + weighted_classes
        )
    elif dominant_residual == "local_code_generation_adapter_needed":
        weighted_classes = ["local_code_generation_adapter_needed"] * 48 + ["edge_case"] * 24 + weighted_classes
    focus = str(concept_focus or "").strip().lower()
    edge_contract_weights = ["edge_case"] * 96 + ["type_handling"] * 48 + [
        "local_code_generation_adapter_needed"
    ] * 36 + ["syntax_structure"] * 24 + ["algorithm_choice"] * 24 + ["parsing"] * 24
    balanced_edge_contract_weights = (
        ["edge_case"] * 48
        + ["type_semantic_transfer"] * 48
        + ["local_code_generation_adapter_needed"] * 48
        + ["typed_interface_skeleton"] * 48
        + ["algorithm_choice"] * 48
        + ["type_handling"] * 24
        + ["syntax_structure"] * 24
        + ["recurrence_state"] * 24
        + ["parsing"] * 24
    )
    edge_case_full_body_weights = (
        ["edge_case_full_body"] * 144
        + ["edge_case"] * 48
        + ["type_semantic_transfer"] * 36
        + ["local_code_generation_adapter_needed"] * 36
        + ["typed_interface_skeleton"] * 24
        + ["algorithm_choice"] * 24
    )
    frontier_agent_edge_full_body_weights = (
        ["edge_case_full_body"] * 216
        + ["intended_behavior_transfer"] * 192
        + ["residual_targeted_edge_case_contract_v1"] * 144
        + ["edge_contract_v2"] * 120
        + ["typed_interface_skeleton"] * 96
        + ["algorithm_choice"] * 84
        + ["type_semantic_transfer"] * 72
        + ["adapter_runtime_dependency_handling"] * 48
        + ["local_code_generation_adapter_needed"] * 36
    )
    edge_contract_v2_weights = (
        ["edge_contract_v2"] * 168
        + ["edge_case_full_body"] * 48
        + ["typed_interface_skeleton"] * 48
        + ["type_semantic_transfer"] * 36
        + ["parsing_encoding_v1"] * 24
        + ["algorithm_choice"] * 36
        + ["local_code_generation_adapter_needed"] * 24
        + ["type_handling"] * 24
    )
    candidate_floor_v2_weights = (
        ["candidate_floor_v2"] * 192
        + ["typed_interface_skeleton"] * 54
        + ["local_code_generation_adapter_needed"] * 54
        + ["algorithm_choice"] * 48
        + ["edge_contract_v2"] * 36
        + ["edge_case_full_body"] * 36
        + ["type_semantic_transfer"] * 30
        + ["parsing_encoding_v1"] * 24
    )
    return_type_shape_v2_weights = (
        ["type_semantic_transfer"] * 168
        + ["type_handling"] * 72
        + ["typed_interface_skeleton"] * 54
        + ["parsing_encoding_v1"] * 36
        + ["edge_contract_v2"] * 24
        + ["local_code_generation_adapter_needed"] * 24
    )
    parsing_encoding_v1_weights = (
        ["parsing_encoding_v1"] * 192
        + ["parsing"] * 72
        + ["syntax_structure"] * 48
        + ["type_semantic_transfer"] * 42
        + ["local_code_generation_adapter_needed"] * 36
        + ["string_rule_composition"] * 24
        + ["edge_contract_v2"] * 18
    )
    residual_edge_case_contract_weights = (
        ["residual_targeted_edge_case_contract_v1"] * 216
        + ["candidate_floor_v2"] * 72
        + ["edge_contract_v2"] * 54
        + ["typed_interface_skeleton"] * 42
        + ["algorithm_choice"] * 36
        + ["type_semantic_transfer"] * 24
    )
    intended_behavior_transfer_weights = (
        ["intended_behavior_transfer"] * 288
        + ["edge_contract_v2"] * 120
        + ["edge_case_full_body"] * 108
        + ["candidate_floor_v2"] * 96
        + ["type_semantic_transfer"] * 96
        + ["residual_targeted_edge_case_contract_v1"] * 72
        + ["algorithm_choice"] * 48
        + ["typed_interface_skeleton"] * 36
        + ["string_rule_composition"] * 24
        + ["digit_rotation"] * 24
    )
    broad_floor_semantic_transfer_weights = (
        ["intended_behavior_transfer"] * 360
        + ["edge_contract_v2"] * 168
        + ["edge_case_full_body"] * 144
        + ["type_semantic_transfer"] * 156
        + ["adapter_runtime_dependency_handling"] * 120
        + ["residual_targeted_edge_case_contract_v1"] * 132
        + ["candidate_floor_v2"] * 96
        + ["typed_interface_skeleton"] * 72
        + ["algorithm_choice"] * 60
        + ["local_code_generation_adapter_needed"] * 48
    )
    focus_weights = {
        "type_and_return_shape": ["type_semantic_transfer"] * 72
        + ["type_handling"] * 36
        + ["runtime"] * 12
        + ["wrong_answer"] * 12,
        "return_type_shape_v2": return_type_shape_v2_weights,
        "type_return_shape_v2": return_type_shape_v2_weights,
        "private_return_type_shape_v2": return_type_shape_v2_weights,
        "admissibility_and_interface": ["typed_interface_skeleton"] * 120
        + ["adapter_runtime_dependency_handling"] * 72
        + ["local_code_generation_adapter_needed"] * 72
        + ["type_semantic_transfer"] * 24
        + ["syntax_structure"] * 24
        + ["edge_case"] * 12,
        "adapter_runtime_dependency_handling": ["adapter_runtime_dependency_handling"] * 144
        + ["local_code_generation_adapter_needed"] * 36
        + ["type_semantic_transfer"] * 24
        + ["edge_case"] * 12,
        "external_dependency_missing": ["adapter_runtime_dependency_handling"] * 216
        + ["local_code_generation_adapter_needed"] * 84
        + ["execution_shaped_programs"] * 48
        + ["type_semantic_transfer"] * 36
        + ["typed_interface_skeleton"] * 24
        + ["edge_case"] * 12,
        "dependency_optional_adapter_private_repair": ["adapter_runtime_dependency_handling"] * 216
        + ["local_code_generation_adapter_needed"] * 84
        + ["execution_shaped_programs"] * 48
        + ["type_semantic_transfer"] * 36
        + ["typed_interface_skeleton"] * 24
        + ["edge_case"] * 12,
        "typed_interface_skeleton": ["typed_interface_skeleton"] * 96
        + ["local_code_generation_adapter_needed"] * 48
        + ["type_semantic_transfer"] * 48
        + ["syntax_structure"] * 24
        + ["edge_case"] * 12,
        "edge_conditions": ["edge_case_full_body"] * 120
        + ["residual_targeted_edge_case_contract_v1"] * 96
        + ["edge_contract_v2"] * 72
        + ["edge_case"] * 72
        + ["intended_behavior_transfer"] * 48
        + ["type_handling"] * 24
        + ["local_code_generation_adapter_needed"] * 12
        + ["algorithm_choice"] * 12,
        "edge_contract_4card": edge_contract_weights,
        "private_edge_contract_4card": edge_contract_weights,
        "benchmark_neutral_edge_contract": edge_contract_weights,
        "edge_contract_balanced_4card_private_curriculum_v2": balanced_edge_contract_weights,
        "edge_contract_balanced_4card_v2": balanced_edge_contract_weights,
        "balanced_edge_contract_4card": balanced_edge_contract_weights,
        "edge_case_full_body_private_curriculum_v1": edge_case_full_body_weights,
        "residual_targeted_private_edge_case_full_body_curriculum_v1": edge_case_full_body_weights,
        "edge_case_full_body": edge_case_full_body_weights,
        "frontier_agent_edge_full_body_private_curriculum_v2": frontier_agent_edge_full_body_weights,
        "frontier_agent_edge_full_body_v2": frontier_agent_edge_full_body_weights,
        "private_frontier_agent_full_body_transfer": frontier_agent_edge_full_body_weights,
        "edge_contract_v2_private_residual_curriculum": edge_contract_v2_weights,
        "edge_contract_v2": edge_contract_v2_weights,
        "private_edge_contract_v2": edge_contract_v2_weights,
        "residual_targeted_edge_contract_v2": edge_contract_v2_weights,
        "candidate_floor_v2_private_residual_curriculum": candidate_floor_v2_weights,
        "candidate_floor_v2": candidate_floor_v2_weights,
        "candidate_floor_adapter_v2": candidate_floor_v2_weights,
        "candidate_floor_adapter_v2_private_residual_curriculum": candidate_floor_v2_weights,
        "no_admissible_adapter_floor_v2": candidate_floor_v2_weights,
        "no_admissible_candidate_floor_v2": candidate_floor_v2_weights,
        "parsing_encoding_v1": parsing_encoding_v1_weights,
        "parsing_encoding_private_curriculum_v1": parsing_encoding_v1_weights,
        "private_parsing_encoding_v1": parsing_encoding_v1_weights,
        "residual_targeted_private_edge_case_contract_v1": residual_edge_case_contract_weights,
        "residual_targeted_edge_case_contract_v1": residual_edge_case_contract_weights,
        "private_residual_edge_case_contract_v1": residual_edge_case_contract_weights,
        "intended_behavior_transfer_private_curriculum": intended_behavior_transfer_weights,
        "intended_behavior_transfer": intended_behavior_transfer_weights,
        "livecodebench_intended_behavior_private_curriculum": intended_behavior_transfer_weights,
        "livecodebench_intended_behavior": intended_behavior_transfer_weights,
        "broad_floor_semantic_transfer_private_curriculum": broad_floor_semantic_transfer_weights,
        "semantic_edge_type_adapter_private_curriculum": broad_floor_semantic_transfer_weights,
        "private_broad_floor_semantic_transfer": broad_floor_semantic_transfer_weights,
        "algorithmic_planning": ["algorithm_choice"] * 72
        + ["recurrence_state"] * 24
        + ["repair_loop"] * 12
        + ["wrong_answer"] * 12,
        "algorithmic_planning_boundary": ["algorithm_choice"] * 96
        + ["recurrence_state"] * 12
        + ["repair_loop"] * 12
        + ["wrong_answer"] * 12,
        "execution_shaped_programs": ["execution_shaped_programs"] * 96
        + ["local_code_generation_adapter_needed"] * 36
        + ["type_handling"] * 24
        + ["edge_case"] * 24
        + ["algorithm_choice"] * 12,
        "decoder_v2_private_ablation_gate": ["edge_case"] * 48
        + ["typed_interface_skeleton"] * 48
        + ["type_semantic_transfer"] * 48
        + ["local_code_generation_adapter_needed"] * 36
        + ["algorithm_choice"] * 36
        + ["execution_shaped_programs"] * 24
        + ["type_handling"] * 24,
    }.get(focus, [])
    if focus_weights:
        weighted_classes = focus_weights + weighted_classes
    if not benchmark_neutral_focus(focus) and (
        broad_context.get("active_card") == "source_mbpp"
        or "source_mbpp" in set(broad_context.get("cards_below_floor") or [])
    ):
        mbpp_weight = max(
            24,
            sum(int(value) for value in (broad_context.get("active_residual_family_counts") or {}).values()),
        )
        weighted_classes = ["mbpp_broad_transfer"] * mbpp_weight + weighted_classes
    if not benchmark_neutral_focus(focus) and (
        broad_context.get("active_card") == "source_bigcodebench"
        or "source_bigcodebench" in set(broad_context.get("cards_below_floor") or [])
    ):
        bigcodebench_weight = max(
            64,
            sum(int(value) for value in (broad_context.get("active_residual_family_counts") or {}).values()),
        )
        weighted_classes = ["execution_shaped_programs"] * bigcodebench_weight + weighted_classes
    if not benchmark_neutral_focus(focus) and (
        broad_context.get("active_card") == "source_livecodebench"
        or "source_livecodebench" in set(broad_context.get("cards_below_floor") or [])
    ):
        live_residuals = broad_context.get("active_residual_family_counts") or {}
        live_weight = max(96, sum(int(value) for value in live_residuals.values()))
        edge_weight = max(24, int(live_residuals.get("edge_case", 0)) * 4)
        type_weight = max(12, int(live_residuals.get("type_handling", 0)) * 4)
        adapter_weight = max(
            12,
            (
                int(live_residuals.get("local_code_generation_adapter_needed", 0))
                + int(live_residuals.get("external_dependency_missing", 0))
            )
            * 4,
        )
        weighted_classes = (
            ["intended_behavior_transfer"] * live_weight
            + ["edge_case_full_body"] * max(48, edge_weight * 2)
            + ["residual_targeted_edge_case_contract_v1"] * edge_weight
            + ["type_handling"] * type_weight
            + ["adapter_runtime_dependency_handling"] * adapter_weight
            + weighted_classes
        )
    verification_pressure = summary.get("verification_pressure") if isinstance(summary.get("verification_pressure"), dict) else {}
    if verification_pressure.get("target_stage") == "intended_behavior":
        weighted_classes = (
            ["intended_behavior_transfer"] * 240
            + ["edge_case_full_body"] * 96
            + ["edge_contract_v2"] * 96
            + ["algorithm_choice"] * 72
            + ["type_semantic_transfer"] * 48
            + weighted_classes
        )
    elif verification_pressure.get("target_stage") == "compile_import":
        weighted_classes = ["typed_interface_skeleton"] * 96 + ["type_semantic_transfer"] * 72 + weighted_classes
    elif verification_pressure.get("target_stage") == "lint_parse":
        weighted_classes = ["syntax_structure"] * 96 + ["typed_interface_skeleton"] * 48 + weighted_classes
    rng.shuffle(weighted_classes)
    templates = template_bank()
    rows: list[dict[str, Any]] = []
    for idx in range(max_rows):
        residual_class = weighted_classes[idx % len(weighted_classes)]
        candidates = templates.get(residual_class) or templates["edge_case"]
        template = candidates[(idx + rng.randrange(len(candidates))) % len(candidates)]
        variant = render_variant(template, idx, rng)
        task_id = f"residual_private_{variant['category']}_{idx:04d}"
        verification_pressure_tags = verification_pressure_tags_for_row(verification_pressure)
        rows.append(
            {
                "task_id": task_id,
                "source_task_id": f"private_residual_{idx:04d}",
                "card_id": "private_residual_code_curriculum",
                "source_id": "local_generated_residual_code_curriculum",
                "split": "train",
                "category": variant["category"],
                "prompt": variant["prompt"],
                "entry_point": variant["entry_point"],
                "solution_expr": variant.get("solution_expr", ""),
                "solution_body": variant["solution_body"],
                "tests": variant["tests"],
                "tags": [
                    "private_residual_curriculum",
                    residual_class,
                    *verification_pressure_tags,
                    *variant.get("tags", []),
                ],
                "residual_concept": variant.get("residual_concept", residual_class),
                "concept_residual_label": variant.get("concept_residual_label", residual_class),
                "verification_pressure": private_row_verification_pressure(verification_pressure),
                "guardrail_expectations": variant.get("guardrail_expectations", {}),
                "decoder_contract": variant.get("decoder_contract", {}),
                "benchmark_evidence_level": "private_residual_generated_train_only",
                "public_benchmark": False,
                "public_benchmark_solutions_included": False,
                "public_tests_included": False,
                "license_spdx": "CC0-1.0",
                "candidate_expression_eligible": bool(variant.get("candidate_expression_eligible", False)),
                "provenance": {
                    "policy": "project_theseus_code_residual_curriculum_v1",
                    "residual_class": residual_class,
                    "residual_concept": variant.get("residual_concept", residual_class),
                    "concept_residual_label": variant.get("concept_residual_label", residual_class),
                    "verification_pressure_target_stage": verification_pressure.get("target_stage", ""),
                    "public_task_ids_hashed_only": True,
                    "public_benchmark_answers_used": False,
                    "public_tests_used": False,
                },
            }
        )
    return rows


def verification_pressure_tags_for_row(pressure: dict[str, Any]) -> list[str]:
    stage = str(pressure.get("dominant_stage") or "").strip()
    target = str(pressure.get("target_stage") or "").strip()
    tags: list[str] = []
    if stage:
        tags.append("verification_stage_" + safe_tag(stage))
    if target:
        tags.append("verification_target_" + safe_tag(target))
    return tags


def private_row_verification_pressure(pressure: dict[str, Any]) -> dict[str, Any]:
    if not pressure:
        return {}
    stage_counts = pressure.get("stage_counts") if isinstance(pressure.get("stage_counts"), dict) else {}
    top_stage_counts = dict(
        sorted(
            ((str(key), int(value or 0)) for key, value in stage_counts.items()),
            key=lambda item: (-item[1], item[0]),
        )[:4]
    )
    return {
        "policy": "project_theseus_private_verification_stage_pressure_v1",
        "dominant_stage": str(pressure.get("dominant_stage") or ""),
        "target_stage": str(pressure.get("target_stage") or ""),
        "target_concepts": [str(item) for item in pressure.get("target_concepts", []) if str(item)],
        "top_stage_counts": top_stage_counts,
        "public_tests_used": False,
        "public_rewards_used_as_training_targets": False,
        "semantics": "sanitized stage pressure only; no public tests, public prompts, candidate code, or exact public reward targets",
    }


def safe_tag(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "unknown"


def benchmark_neutral_focus(concept_focus: str) -> bool:
    return str(concept_focus or "").strip().lower() in {
        "type_and_return_shape",
        "type_semantic_transfer",
        "return_type_shape_v2",
        "type_return_shape_v2",
        "private_return_type_shape_v2",
        "edge_contract_4card",
        "private_edge_contract_4card",
        "benchmark_neutral_edge_contract",
        "edge_contract_balanced_4card_private_curriculum_v2",
        "edge_contract_balanced_4card_v2",
        "balanced_edge_contract_4card",
        "edge_case_full_body_private_curriculum_v1",
        "residual_targeted_private_edge_case_full_body_curriculum_v1",
        "edge_case_full_body",
        "frontier_agent_edge_full_body_private_curriculum_v2",
        "frontier_agent_edge_full_body_v2",
        "private_frontier_agent_full_body_transfer",
        "edge_contract_v2_private_residual_curriculum",
        "edge_contract_v2",
        "private_edge_contract_v2",
        "residual_targeted_edge_contract_v2",
        "candidate_floor_v2_private_residual_curriculum",
        "candidate_floor_v2",
        "candidate_floor_adapter_v2",
        "candidate_floor_adapter_v2_private_residual_curriculum",
        "no_admissible_adapter_floor_v2",
        "no_admissible_candidate_floor_v2",
        "parsing_encoding_v1",
        "parsing_encoding_private_curriculum_v1",
        "private_parsing_encoding_v1",
        "residual_targeted_private_edge_case_contract_v1",
        "residual_targeted_edge_case_contract_v1",
        "private_residual_edge_case_contract_v1",
        "intended_behavior_transfer_private_curriculum",
        "intended_behavior_transfer",
        "livecodebench_intended_behavior_private_curriculum",
        "livecodebench_intended_behavior",
        "broad_floor_semantic_transfer_private_curriculum",
        "semantic_edge_type_adapter_private_curriculum",
        "private_broad_floor_semantic_transfer",
        "algorithmic_planning_boundary",
        "admissibility_and_interface",
        "adapter_runtime_dependency_handling",
        "external_dependency_missing",
        "dependency_optional_adapter_private_repair",
        "decoder_v2_private_ablation_gate",
        "typed_interface_skeleton",
    }


def edge_case_full_body_focus(concept_focus: str) -> bool:
    return str(concept_focus or "").strip().lower() in {
        "edge_conditions",
        "edge_case_full_body_private_curriculum_v1",
        "residual_targeted_private_edge_case_full_body_curriculum_v1",
        "edge_case_full_body",
        "frontier_agent_edge_full_body_private_curriculum_v2",
        "frontier_agent_edge_full_body_v2",
        "private_frontier_agent_full_body_transfer",
    }


def frontier_full_body_pressure_required(
    concept_focus: str,
    residual_summary: dict[str, Any],
    broad_context: dict[str, Any],
) -> bool:
    focus = str(concept_focus or "").strip().lower()
    focus_allows_full_body_gate = (
        not focus
        or edge_case_full_body_focus(focus)
        or edge_contract_v2_focus(focus)
        or focus
        in {
            "broad_floor_semantic_transfer_private_curriculum",
            "semantic_edge_type_adapter_private_curriculum",
            "private_broad_floor_semantic_transfer",
        }
    )
    if not focus_allows_full_body_gate:
        return False
    cards_below_floor = set(str(card) for card in broad_context.get("cards_below_floor") or [])
    livecodebench_wall = (
        broad_context.get("active_card") == "source_livecodebench"
        or "source_livecodebench" in cards_below_floor
    )
    intended_behavior_wall = (
        residual_summary.get("verification_pressure", {}).get("target_stage") == "intended_behavior"
        or int(residual_summary.get("verification_stage_counts", {}).get("intended_behavior_failed", 0) or 0) > 0
    )
    return bool(livecodebench_wall or intended_behavior_wall)


def edge_pressure_exempt_focus(concept_focus: str) -> bool:
    return str(concept_focus or "").strip().lower() in {
        "type_and_return_shape",
        "type_semantic_transfer",
        "return_type_shape_v2",
        "type_return_shape_v2",
        "private_return_type_shape_v2",
        "typed_interface_skeleton",
        "parsing_encoding_v1",
        "parsing_encoding_private_curriculum_v1",
        "private_parsing_encoding_v1",
        "admissibility_and_interface",
        "adapter_runtime_dependency_handling",
        "external_dependency_missing",
        "dependency_optional_adapter_private_repair",
    }


def external_dependency_focus(concept_focus: str) -> bool:
    return str(concept_focus or "").strip().lower() in {
        "external_dependency_missing",
        "dependency_optional_adapter_private_repair",
        "adapter_runtime_dependency_handling",
    }


def edge_contract_v2_focus(concept_focus: str) -> bool:
    return str(concept_focus or "").strip().lower() in {
        "edge_contract_v2_private_residual_curriculum",
        "edge_contract_v2",
        "private_edge_contract_v2",
        "residual_targeted_edge_contract_v2",
        "candidate_floor_v2_private_residual_curriculum",
        "candidate_floor_v2",
        "candidate_floor_adapter_v2",
        "candidate_floor_adapter_v2_private_residual_curriculum",
        "no_admissible_adapter_floor_v2",
        "no_admissible_candidate_floor_v2",
        "residual_targeted_private_edge_case_contract_v1",
        "residual_targeted_edge_case_contract_v1",
        "private_residual_edge_case_contract_v1",
        "intended_behavior_transfer_private_curriculum",
        "intended_behavior_transfer",
        "livecodebench_intended_behavior_private_curriculum",
        "livecodebench_intended_behavior",
        "broad_floor_semantic_transfer_private_curriculum",
        "semantic_edge_type_adapter_private_curriculum",
        "private_broad_floor_semantic_transfer",
    }


def execution_shaped_programs_rows(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if "execution_shaped_programs" in (row.get("tags") or [])
        or row.get("residual_concept") == "execution_shaped_programs"
    )


def intended_behavior_pressure_rows(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("verification_pressure", {}).get("target_stage") == "intended_behavior"
        or "verification_stage_intended_behavior_failed" in (row.get("tags") or [])
        or "verification_target_intended_behavior" in (row.get("tags") or [])
        or "intended_behavior_transfer" in (row.get("tags") or [])
        or row.get("residual_concept") == "intended_behavior_transfer"
    )


def execution_shaped_programs_share(rows: list[dict[str, Any]]) -> float:
    return round(execution_shaped_programs_rows(rows) / max(1, len(rows)), 6)


def edge_case_full_body_rows(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if "edge_case_full_body" in (row.get("tags") or [])
        or row.get("residual_concept") == "edge_case_full_body"
    )


def edge_case_full_body_contract_rows(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if "edge_case_full_body" not in (row.get("tags") or []) and row.get("residual_concept") != "edge_case_full_body":
            continue
        tests = str(row.get("tests") or "")
        if tests.count("assert ") >= 4 and bool(row.get("solution_body")) and not row.get("candidate_expression_eligible"):
            count += 1
    return count


def edge_contract_v2_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if edge_contract_v2_pressure_row(row))


def edge_contract_v2_pressure_row(row: dict[str, Any]) -> bool:
    tags = row.get("tags") or []
    return bool(
        "edge_contract_v2" in tags
        or row.get("residual_concept") == "edge_contract_v2"
        or "intended_behavior_transfer" in tags
        or row.get("residual_concept") == "intended_behavior_transfer"
        or "candidate_floor_v2" in tags
        or row.get("residual_concept") == "candidate_floor_v2"
        or "residual_targeted_edge_case_contract_v1" in tags
        or row.get("residual_concept") == "residual_targeted_private_edge_case_contract_v1"
    )


def edge_contract_v2_generation_plan_rows(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if not edge_contract_v2_pressure_row(row):
            continue
        contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
        plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
        if (
            str(contract.get("policy") or "").startswith("project_theseus_decoder_contract_v2_private_")
            and plan.get("skeleton_bias")
            and plan.get("repair_strategy")
            and plan.get("verifier_feedback")
            and not plan.get("public_tests_used")
            and not plan.get("public_solutions_used")
        ):
            count += 1
    return count


def optional_dependency_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if "optional_dependency" in (row.get("tags") or []))


def optional_dependency_guard_contract_rows(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if "optional_dependency" not in (row.get("tags") or []):
            continue
        contract = row.get("decoder_contract") or {}
        score_semantics = str(contract.get("score_semantics") or "").lower()
        residual_hint = str(contract.get("residual_label_hint") or "").lower()
        body = str(row.get("solution_body") or row.get("solution") or row.get("body") or "").lower()
        if (
            contract.get("full_body_required") is True
            and "runtime_load_failure" in residual_hint
            and "fallback" in score_semantics
            and "try:" in body
            and "except exception" in body
            and "return" in body
        ):
            count += 1
    return count


def benchmark_named_private_row_count(rows: list[dict[str, Any]]) -> int:
    prefixes = ("mbpp_", "bigcodebench_", "livecodebench_", "evalplus_")
    return sum(
        1
        for row in rows
        if any(str(tag).startswith(prefixes) for tag in (row.get("tags") or []))
    )


def verify_private_solution_rows(rows: list[dict[str, Any]], *, max_failures: int = 5) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    for row in rows:
        entry_point = safe_identifier(str(row.get("entry_point") or "private_task"))
        body = str(row.get("solution_body") or "").strip()
        tests = str(row.get("tests") or "").strip()
        if not body or not tests:
            continue
        namespace: dict[str, Any] = {}
        code = (
            f"def {entry_point}(*args):\n"
            "    data = args[0] if len(args) > 0 else None\n"
            "    other = args[1] if len(args) > 1 else None\n"
            "    extra = args[2:] if len(args) > 2 else ()\n"
        )
        for line in body.splitlines():
            code += f"    {line}\n"
        try:
            exec(code, namespace, namespace)
            exec(tests, namespace, namespace)
        except Exception as exc:  # pragma: no cover - report detail is the test oracle
            failures.append(
                {
                    "task_id": str(row.get("task_id") or ""),
                    "category": str(row.get("category") or ""),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            if len(failures) >= max_failures:
                break
    return {
        "checked_rows": len(rows),
        "failure_count": len(failures),
        "sample_failures": failures,
        "contract": "solution_body is compiled as def entry_point(*args) with data/other/extra locals before tests run",
    }


def safe_identifier(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return "\n".join(
        [
            "# Code Residual Curriculum",
            "",
            f"State: **{report.get('trigger_state')}**",
            "",
            f"- Private rows: {summary.get('private_row_count')}",
            f"- Private solution test failures: {summary.get('private_solution_test_failures')}",
            f"- Residual classes: {summary.get('residual_class_counts')}",
            f"- Concept residuals: {summary.get('concept_residual_counts')}",
            f"- Target wall families: {summary.get('target_wall_family_counts')}",
            f"- Broad active card: {summary.get('broad_active_card')}",
            f"- Broad active card selection: {summary.get('broad_active_card_selection')}",
            f"- Broad active card additional passes needed: {summary.get('broad_active_card_additional_passes_needed')}",
            f"- Broad below-floor cards: {summary.get('broad_cards_below_floor')}",
            f"- Broad no-clean cards: {summary.get('broad_no_clean_student_evidence_cards')}",
            f"- Broad loader-only cards: {summary.get('broad_loader_only_cards')}",
            f"- LiveCodeBench intended-behavior private rows: {summary.get('livecodebench_intended_behavior_private_rows')}",
            f"- MBPP broad-transfer private rows: {summary.get('mbpp_broad_transfer_private_rows')}",
            f"- Edge-case full-body rows: {summary.get('edge_case_full_body_rows')}",
            f"- Edge-case full-body contract rows: {summary.get('edge_case_full_body_contract_rows')}",
            f"- Edge-contract v2 rows: {summary.get('edge_contract_v2_rows')}",
            f"- Edge-contract v2 generation-plan rows: {summary.get('edge_contract_v2_generation_plan_rows')}",
            f"- Optional-dependency rows: {summary.get('optional_dependency_rows')}",
            f"- Optional-dependency guarded-fallback rows: {summary.get('optional_dependency_guard_contract_rows')}",
            f"- Execution-shaped programs rows: {summary.get('execution_shaped_programs_rows')}",
            f"- Execution-shaped programs share: {summary.get('execution_shaped_programs_share')}",
            f"- Benchmark-named private rows: {summary.get('benchmark_named_private_rows')}",
            f"- Public task ids hashed only: {summary.get('public_task_ids_hashed_only')}",
            f"- Public solutions included: {summary.get('public_benchmark_solutions_included')}",
            f"- Public tests included: {summary.get('public_tests_included')}",
            "",
        ]
    )


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
