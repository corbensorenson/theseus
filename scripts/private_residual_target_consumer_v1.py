#!/usr/bin/env python3
"""Consume public-transfer residual targets as private-only repair pressure.

The input target manifest is intentionally aggregate metadata only. This script
validates that boundary, joins each target against current private evidence, and
emits a private repair queue. It does not write training rows, copy public
benchmark content, run public calibration, or inspect public prompts/tests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

DEFAULT_TARGETS = REPORTS / "bounded_public_transfer_private_residual_targets_current_private_full_body_v1.jsonl"
DEFAULT_ADMISSIBILITY = REPORTS / "private_full_body_candidate_admissibility_gate_capability_transfer_closure_v1.json"
DEFAULT_HELDOUT = REPORTS / "private_residual_repair_v3_heldout_score_capability_transfer_closure_v1.json"
DEFAULT_RESIDUAL_GATE = REPORTS / "private_residual_repair_v3_gate_capability_transfer_closure_v1.json"
DEFAULT_STS_POLICY = REPORTS / "sts_ranker_policy_v1.json"
DEFAULT_ALIGNMENT_PREFLIGHT = REPORTS / "public_calibration_alignment_preflight.json"
DEFAULT_CANDIDATE_FLOOR = REPORTS / "candidate_floor_v2_private_token_probe.json"
DEFAULT_PUBLIC_RESIDUAL_REPORT = (
    REPORTS / "bounded_public_transfer_residual_mining_public_transfer_measurement_diagnostic_seed1_5x3.json"
)
DEFAULT_FRESH_RESIDUAL_PRIVATE_PROBE = REPORTS / "candidate_floor_v2_fresh_residual_queue_probe.json"
DEFAULT_OUT = REPORTS / "private_residual_target_consumer_v1.json"
DEFAULT_MD = REPORTS / "private_residual_target_consumer_v1.md"
DEFAULT_QUEUE = REPORTS / "private_residual_repair_queue_v1.jsonl"

FORBIDDEN_FLAGS = (
    "public_content_embedded",
    "public_task_id_embedded",
    "public_prompt_embedded",
    "public_tests_embedded",
    "public_solution_embedded",
    "candidate_code_embedded",
    "score_label_embedded",
    "training_row",
)

FORBIDDEN_SOURCES = {
    "public_benchmark_prompt",
    "public_visible_tests",
    "public_hidden_tests",
    "public_solution",
    "public_trace",
    "public_answer_template",
    "public_task_identity",
    "public_candidate_code_from_calibration",
    "public_score_label",
}

PRIVATE_TRANSFER_SEMANTIC_FLOOR = 0.70


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=rel(DEFAULT_TARGETS))
    parser.add_argument("--private-admissibility", default=rel(DEFAULT_ADMISSIBILITY))
    parser.add_argument("--private-heldout", default=rel(DEFAULT_HELDOUT))
    parser.add_argument("--private-residual-gate", default=rel(DEFAULT_RESIDUAL_GATE))
    parser.add_argument("--sts-policy", default=rel(DEFAULT_STS_POLICY))
    parser.add_argument("--alignment-preflight", default=rel(DEFAULT_ALIGNMENT_PREFLIGHT))
    parser.add_argument("--candidate-floor-probe", default=rel(DEFAULT_CANDIDATE_FLOOR))
    parser.add_argument("--public-residual-report", default=rel(DEFAULT_PUBLIC_RESIDUAL_REPORT))
    parser.add_argument("--fresh-residual-private-probe", default=rel(DEFAULT_FRESH_RESIDUAL_PRIVATE_PROBE))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--queue-out", default=rel(DEFAULT_QUEUE))
    args = parser.parse_args()

    report = build_report(args)
    write_jsonl(resolve(args.queue_out), as_list(report.get("repair_queue")))
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    targets_path = resolve(args.targets)
    admissibility_path = resolve(args.private_admissibility)
    heldout_path = resolve(args.private_heldout)
    residual_gate_path = resolve(args.private_residual_gate)
    sts_policy_path = resolve(args.sts_policy)
    alignment_preflight_path = resolve(args.alignment_preflight)
    candidate_floor_path = resolve(args.candidate_floor_probe)
    public_residual_path = resolve(args.public_residual_report)
    fresh_residual_private_probe_path = resolve(args.fresh_residual_private_probe)

    targets = read_jsonl(targets_path)
    admissibility = read_json(admissibility_path, {})
    heldout = read_json(heldout_path, {})
    residual_gate = read_json(residual_gate_path, {})
    sts_policy = read_json(sts_policy_path, {})
    alignment_preflight = read_json(alignment_preflight_path, {})
    candidate_floor = read_json(candidate_floor_path, {})
    public_residual = read_json(public_residual_path, {})
    fresh_residual_private_probe = read_json(fresh_residual_private_probe_path, {})

    adm = obj(admissibility, "summary")
    heldout_summary = obj(heldout, "summary")
    residual_summary = obj(residual_gate, "summary")
    residual_summary["_trigger_state"] = residual_gate.get("trigger_state")
    sts_summary = obj(sts_policy, "summary")
    alignment_summary = obj(alignment_preflight, "summary")
    alignment_summary["_trigger_state"] = alignment_preflight.get("trigger_state")
    alignment_summary["_alignment_preflight_ready"] = alignment_preflight.get("alignment_preflight_ready")
    candidate_floor_summary = obj(candidate_floor, "summary")
    candidate_floor_quality = candidate_floor_quality_summary(candidate_floor, candidate_floor_summary)
    fresh_probe_quality = fresh_residual_private_probe_quality(fresh_residual_private_probe)
    public_residual_summary = obj(public_residual, "summary")
    fresh_public_residual_counts = public_residual_failure_counts(public_residual)

    validation_rows = [validate_target(row) for row in targets]
    invalid_rows = [row for row in validation_rows if not row["valid"]]
    queue = build_repair_queue(
        targets,
        adm,
        heldout_summary,
        residual_summary,
        sts_summary,
        alignment_summary,
        candidate_floor_quality,
        fresh_probe_quality,
        fresh_public_residual_counts,
    )
    coverage_counts = Counter(row["state"] for row in queue)
    unresolved_rows = [
        row
        for row in queue
        if row.get("state") in {"ready", "needs_private_ablation", "blocked"}
    ]
    unresolved_by_state = Counter(str(row.get("state") or "unknown") for row in unresolved_rows)
    unresolved_by_category = Counter(str(row.get("canonical_category") or "unknown") for row in unresolved_rows)
    target_families = Counter(str(row.get("private_curriculum_family") or "unknown") for row in targets)
    target_categories = Counter(str(row.get("canonical_category") or "unknown") for row in targets)

    gates = [
        gate("target_manifest_loaded", len(targets) > 0, {"path": rel(targets_path), "rows": len(targets)}),
        gate("target_manifest_metadata_only", not invalid_rows, {"invalid_rows": len(invalid_rows)}),
        gate("private_evidence_loaded", bool(adm) and bool(heldout_summary), {
            "admissibility_state": admissibility.get("trigger_state"),
            "heldout_state": heldout.get("trigger_state"),
            "residual_gate_state": residual_gate.get("trigger_state"),
        }),
        gate("queue_written_without_training_rows", all(row.get("training_row") is False for row in queue), {
            "queue_rows": len(queue),
            "training_rows": sum(1 for row in queue if row.get("training_row") is True),
        }),
        gate("public_content_not_embedded", all(not row.get(flag) for row in queue for flag in FORBIDDEN_FLAGS), {
            "queue_rows": len(queue),
        }),
        gate("fallback_returns_zero", int_number(adm.get("fallback_return_candidate_count")) == 0, {
            "fallback_return_candidate_count": int_number(adm.get("fallback_return_candidate_count")),
        }),
        gate("public_training_rows_zero", True, {"training_rows_written": 0}),
        gate(
            "repair_queue_covers_every_valid_target",
            len(queue) == len(targets) - len(invalid_rows),
            {
                "repair_queue_rows": len(queue),
                "valid_target_rows": len(targets) - len(invalid_rows),
            },
            severity="hard",
        ),
        gate(
            "all_valid_targets_closed_by_current_private_evidence",
            not unresolved_rows,
            {
                "unresolved_target_count": len(unresolved_rows),
                "unresolved_by_state": dict(unresolved_by_state),
                "unresolved_by_category": dict(unresolved_by_category),
            },
            severity="warning",
        ),
        gate(
            "candidate_floor_semantic_quality_ready",
            bool(candidate_floor_quality.get("semantic_quality_ready")),
            candidate_floor_quality,
            severity="warning",
        ),
    ]
    hard_failures = [row for row in gates if not row["passed"] and row.get("severity") == "hard"]
    warning_failures = [row for row in gates if not row["passed"] and row.get("severity") == "warning"]
    trigger_state = "RED" if hard_failures else "YELLOW" if warning_failures else "GREEN"

    return {
        "policy": "project_theseus_private_residual_target_consumer_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "target_rows": len(targets),
            "valid_target_rows": len(targets) - len(invalid_rows),
            "invalid_target_rows": len(invalid_rows),
            "repair_queue_rows": len(queue),
            "ready_target_count": coverage_counts.get("ready", 0),
            "covered_target_count": coverage_counts.get("covered_by_current_private_evidence", 0),
            "needs_private_ablation_count": coverage_counts.get("needs_private_ablation", 0),
            "blocked_target_count": coverage_counts.get("blocked", 0),
            "unresolved_target_count": len(unresolved_rows),
            "unresolved_target_state_counts": dict(unresolved_by_state),
            "unresolved_target_category_counts": dict(unresolved_by_category),
            "target_coverage_rate": (
                coverage_counts.get("covered_by_current_private_evidence", 0) / len(targets)
                if targets
                else 0.0
            ),
            "target_categories": dict(target_categories),
            "target_families": dict(target_families),
            "private_selected_pass_rate": number(adm.get("selected_pass_rate")),
            "private_pass_if_any_rate": number(adm.get("pass_if_any_rate")),
            "private_no_admissible_task_rate": number(adm.get("no_admissible_task_rate")),
            "private_heldout_pass_rate": number(heldout_summary.get("private_residual_v3_heldout_pass_rate")),
            "private_learned_candidate_pass_rate": number(heldout_summary.get("learned_candidate_task_pass_rate")),
            "sts_policy_selected_pass_rate": number(sts_summary.get("sts_policy_selected_pass_rate")),
            "sts_policy_delta": number(sts_summary.get("selected_pass_delta_sts_policy_minus_non_sts_policy")),
            "fallback_return_candidate_count": int_number(adm.get("fallback_return_candidate_count")),
            "template_like_candidate_count": int_number(adm.get("template_like_candidate_count")),
            "public_leakage_count": int_number(adm.get("public_leakage_count")),
            "public_training_rows_written": 0,
            "training_rows_written": 0,
            "external_inference_calls": 0,
            "alignment_preflight_state": alignment_preflight.get("trigger_state"),
            "alignment_preflight_ready": alignment_preflight.get("alignment_preflight_ready"),
            "alignment_case_manifest": alignment_summary.get("case_manifest"),
            "alignment_case_manifest_row_count": alignment_summary.get("case_manifest_row_count"),
            "alignment_candidate_manifest_bound_to_case_manifest": alignment_summary.get(
                "candidate_manifest_bound_to_case_manifest"
            ),
            "candidate_floor_probe": rel(candidate_floor_path),
            "candidate_floor_trigger_state": candidate_floor_quality.get("trigger_state"),
            "candidate_floor_candidate_coverage_rate": candidate_floor_quality.get("candidate_coverage_rate"),
            "candidate_floor_private_trained_pass_rate": candidate_floor_quality.get("private_trained_pass_rate"),
            "candidate_floor_private_trained_passed": candidate_floor_quality.get("private_trained_passed"),
            "candidate_floor_private_eval_task_count": candidate_floor_quality.get("private_eval_task_count"),
            "candidate_floor_semantic_quality_ready": candidate_floor_quality.get("semantic_quality_ready"),
            "candidate_floor_admissibility_ready": candidate_floor_quality.get("admissibility_ready"),
            "candidate_floor_weak_family_count": candidate_floor_quality.get("weak_family_count"),
            "candidate_floor_weak_families": candidate_floor_quality.get("weak_families"),
            "fresh_residual_private_probe": rel(fresh_residual_private_probe_path),
            "fresh_residual_private_probe_state": fresh_probe_quality.get("trigger_state"),
            "fresh_residual_private_probe_pass_rate": fresh_probe_quality.get("private_trained_pass_rate"),
            "fresh_residual_private_probe_task_count": fresh_probe_quality.get("private_eval_task_count"),
            "fresh_residual_private_probe_covered_categories": fresh_probe_quality.get("covered_categories"),
            "fresh_residual_private_probe_unresolved_categories": fresh_probe_quality.get("unresolved_categories"),
            "fresh_residual_private_probe_missing_categories": fresh_probe_quality.get("missing_categories"),
            "fresh_public_residual_report": rel(public_residual_path),
            "fresh_public_residual_state": public_residual.get("trigger_state"),
            "fresh_public_pass_rate": number(public_residual_summary.get("current_public_pass_rate")),
            "fresh_public_task_count": int_number(public_residual_summary.get("current_public_task_count")),
            "fresh_public_residual_counts": fresh_public_residual_counts,
            "fresh_public_residual_open_target_count": sum(
                1 for row in queue if int_number(obj(row, "current_private_evidence").get("fresh_public_residual_count")) > 0
            ),
            "all_valid_targets_closed_by_current_private_evidence": not unresolved_rows,
            "queue_path": rel(resolve(args.queue_out)),
        },
        "inputs": {
            "targets": rel(targets_path),
            "private_admissibility": rel(admissibility_path),
            "private_heldout": rel(heldout_path),
            "private_residual_gate": rel(residual_gate_path),
            "sts_policy": rel(sts_policy_path),
            "alignment_preflight": rel(alignment_preflight_path),
            "candidate_floor_probe": rel(candidate_floor_path),
            "public_residual_report": rel(public_residual_path),
            "fresh_residual_private_probe": rel(fresh_residual_private_probe_path),
        },
        "validation_rows": validation_rows,
        "repair_queue": queue,
        "gates": gates,
        "rules": {
            "no_public_training_rows": True,
            "no_public_prompt_or_test_content": True,
            "no_candidate_code_from_public_calibration": True,
            "queue_is_not_training_data": True,
            "public_calibration_not_run": True,
        },
        "external_inference_calls": 0,
    }


def candidate_floor_quality_summary(report: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    family_rates = summary.get("private_family_pass_rates")
    family_counts = summary.get("private_family_counts")
    family_rates = family_rates if isinstance(family_rates, dict) else {}
    family_counts = family_counts if isinstance(family_counts, dict) else {}
    weak_families = []
    for family, value in sorted(family_rates.items()):
        rate = number(value)
        count = int_number(family_counts.get(family))
        if count > 0 and rate < PRIVATE_TRANSFER_SEMANTIC_FLOOR:
            weak_families.append({"family": family, "pass_rate": rate, "task_count": count})
    weak_families.sort(key=lambda row: (row["pass_rate"], -row["task_count"], row["family"]))

    fallback_count = int_number(summary.get("expression_memory_fallback_count"))
    template_count = int_number(summary.get("template_like_candidate_count"))
    external_calls = int_number(summary.get("external_inference_calls"), summary.get("runtime_external_inference_calls"))
    candidate_coverage_rate = number(summary.get("candidate_coverage_rate"))
    private_pass_rate = number(summary.get("private_trained_pass_rate"))
    admissibility_ready = bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and candidate_coverage_rate >= 1.0
        and int_number(summary.get("no_candidate_task_count")) == 0
        and int_number(summary.get("full_body_token_candidate_count")) > 0
        and int_number(summary.get("grammar_masked_learned_token_candidate_count")) > 0
        and fallback_count == 0
        and template_count == 0
        and external_calls == 0
    )
    semantic_quality_ready = bool(
        admissibility_ready
        and private_pass_rate >= PRIVATE_TRANSFER_SEMANTIC_FLOOR
    )
    return {
        "trigger_state": report.get("trigger_state"),
        "semantic_floor": PRIVATE_TRANSFER_SEMANTIC_FLOOR,
        "admissibility_ready": admissibility_ready,
        "semantic_quality_ready": semantic_quality_ready,
        "candidate_coverage_rate": candidate_coverage_rate,
        "no_candidate_task_count": int_number(summary.get("no_candidate_task_count")),
        "private_trained_pass_rate": private_pass_rate,
        "private_trained_passed": int_number(summary.get("private_trained_passed")),
        "private_eval_task_count": int_number(summary.get("private_eval_task_count")),
        "full_body_token_candidate_count": int_number(summary.get("full_body_token_candidate_count")),
        "grammar_masked_learned_token_candidate_count": int_number(
            summary.get("grammar_masked_learned_token_candidate_count")
        ),
        "fallback_return_candidate_count": fallback_count,
        "template_like_candidate_count": template_count,
        "external_inference_calls": external_calls,
        "weak_family_count": len(weak_families),
        "weak_families": weak_families[:12],
        "score_semantics": (
            "Private candidate-floor evidence only. Coverage/admissibility can be green while semantic "
            "quality remains too weak for public-transfer repair closure."
        ),
    }


def fresh_residual_private_probe_quality(report: dict[str, Any]) -> dict[str, Any]:
    summary = obj(report, "summary")
    queue_eval = obj(summary, "residual_queue_eval")
    open_categories = [str(item) for item in as_list(queue_eval.get("open_categories")) if str(item)]
    missing_categories = {str(item) for item in as_list(queue_eval.get("missing_open_categories")) if str(item)}
    min_rates = obj(queue_eval, "targeted_private_eval_min_family_pass_rate_by_category")
    covered_categories = []
    unresolved_categories = set(missing_categories)
    for category in open_categories:
        if category in missing_categories:
            continue
        rate = number(min_rates.get(category))
        if rate >= 1.0:
            covered_categories.append(category)
        else:
            unresolved_categories.add(category)
    return {
        "trigger_state": report.get("trigger_state"),
        "private_trained_pass_rate": number(summary.get("private_trained_pass_rate")),
        "private_eval_task_count": int_number(summary.get("private_eval_task_count")),
        "candidate_coverage_rate": number(summary.get("candidate_coverage_rate")),
        "fallback_return_candidate_count": int_number(summary.get("expression_memory_fallback_count")),
        "template_like_candidate_count": int_number(summary.get("template_like_candidate_count")),
        "external_inference_calls": int_number(summary.get("external_inference_calls")),
        "covered_categories": sorted(covered_categories),
        "unresolved_categories": sorted(unresolved_categories),
        "missing_categories": sorted(missing_categories),
        "category_min_pass_rates": min_rates,
        "targeted_private_eval_category_counts": queue_eval.get("targeted_private_eval_category_counts")
        if isinstance(queue_eval.get("targeted_private_eval_category_counts"), dict)
        else {},
        "score_semantics": (
            "Fresh public residual categories are represented only as category labels; "
            "private heldout rows and private verifier results close categories."
        ),
    }


def public_residual_failure_counts(report: dict[str, Any]) -> dict[str, int]:
    summary_counts = obj(obj(report, "summary"), "residual_category_counts")
    counts: Counter[str] = Counter()
    for category, value in summary_counts.items():
        canonical = canonical_public_residual_category(str(category))
        count = int_number(value)
        if canonical and count > 0:
            counts[canonical] += count
    if counts:
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    task_rows = report.get("task_failure_rows")
    if isinstance(task_rows, list):
        for row in task_rows:
            if not isinstance(row, dict):
                continue
            raw = str(row.get("residual_type") or "")
            canonical = canonical_public_residual_category(raw)
            if canonical:
                counts[canonical] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def canonical_public_residual_category(category: str) -> str:
    normalized = category.strip().lower()
    aliases = {
        "return_shape": "return_type_shape",
        "type_handling": "return_type_shape",
        "algorithmic_planning": "algorithm_choice",
        "edge_case": "edge_cases",
        "edge_cases": "edge_cases",
        "external_dependency_missing": "external_dependency_missing",
        "dependency_unavailable": "external_dependency_missing",
        "candidate_dependency_unavailable": "external_dependency_missing",
        "local_code_generation_adapter_needed": "selector_ranking_miss",
        "no_admissible_candidate_regression": "selector_ranking_miss",
        "no_admissible_interface_coverage": "selector_ranking_miss",
        "verifier_mismatch": "verifier_mismatch",
        "io_contract_stdin": "io_contract_stdin",
        "timeout_runtime": "timeout_runtime",
        "parsing_syntax": "parsing_syntax",
        "candidate_manifest_slice_mismatch": "candidate_manifest_slice_mismatch",
        "spent_calibration_no_admissible_resolved_by_current_manifest": (
            "spent_calibration_no_admissible_resolved_by_current_manifest"
        ),
        "selector_ranking_miss": "selector_ranking_miss",
        "algorithm_choice": "algorithm_choice",
        "return_type_shape": "return_type_shape",
    }
    return aliases.get(normalized, "")


def build_repair_queue(
    targets: list[dict[str, Any]],
    adm: dict[str, Any],
    heldout: dict[str, Any],
    residual_gate: dict[str, Any],
    sts_policy: dict[str, Any],
    alignment_preflight: dict[str, Any],
    candidate_floor_quality: dict[str, Any],
    fresh_probe_quality: dict[str, Any],
    fresh_public_residual_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        validation = validate_target(target)
        category = str(target.get("canonical_category") or "unknown")
        evidence = evidence_for_category(
            category,
            adm,
            heldout,
            residual_gate,
            sts_policy,
            alignment_preflight,
            candidate_floor_quality,
            fresh_probe_quality,
            fresh_public_residual_counts,
        )
        state = target_state(category, validation, evidence)
        rows.append(
            {
                "policy": "project_theseus_private_residual_repair_queue_v1",
                "queue_id": f"private_repair_queue_{sha256_text(str(target.get('target_id')) + ':' + category)[:16]}",
                "created_utc": now(),
                "source_target_id": target.get("target_id"),
                "state": state,
                "priority": int_number(target.get("priority"), 99),
                "canonical_category": category,
                "private_curriculum_family": target.get("private_curriculum_family"),
                "repair_focus": target.get("repair_focus") if isinstance(target.get("repair_focus"), list) else [],
                "required_private_evidence": target.get("required_private_evidence")
                if isinstance(target.get("required_private_evidence"), list)
                else [],
                "acceptance_metric": target.get("acceptance_metric"),
                "current_private_evidence": evidence,
                "validation": validation,
                "next_private_action": next_private_action(category, state),
                "public_content_embedded": False,
                "public_task_id_embedded": False,
                "public_prompt_embedded": False,
                "public_tests_embedded": False,
                "public_solution_embedded": False,
                "candidate_code_embedded": False,
                "score_label_embedded": False,
                "training_row": False,
                "training_rows_written": 0,
                "external_inference_calls": 0,
            }
        )
    rows.sort(key=lambda row: (int(row.get("priority") or 99), state_order(str(row.get("state"))), str(row.get("canonical_category"))))
    return rows


def validate_target(row: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if row.get("policy") != "project_theseus_private_residual_target_v1":
        failures.append("unexpected_policy")
    for flag in FORBIDDEN_FLAGS:
        if row.get(flag) is True:
            failures.append(f"{flag}_true")
    forbidden = set(str(item) for item in as_list(row.get("forbidden_training_sources")))
    missing_forbidden = sorted(FORBIDDEN_SOURCES - forbidden)
    if missing_forbidden:
        failures.append("forbidden_source_boundary_incomplete")
    if not row.get("target_id"):
        failures.append("target_id_missing")
    if not row.get("canonical_category"):
        failures.append("canonical_category_missing")
    return {
        "target_id": row.get("target_id"),
        "valid": not failures,
        "failures": failures,
        "missing_forbidden_sources": missing_forbidden,
    }


def evidence_for_category(
    category: str,
    adm: dict[str, Any],
    heldout: dict[str, Any],
    residual_gate: dict[str, Any],
    sts_policy: dict[str, Any],
    alignment_preflight: dict[str, Any],
    candidate_floor_quality: dict[str, Any],
    fresh_probe_quality: dict[str, Any],
    fresh_public_residual_counts: dict[str, int],
) -> dict[str, Any]:
    family_rates = obj(adm, "family_rates")
    heldout_family_rates = obj(heldout, "family_rates")
    base = {
        "selected_pass_rate": number(adm.get("selected_pass_rate")),
        "pass_if_any_rate": number(adm.get("pass_if_any_rate")),
        "no_admissible_task_rate": number(adm.get("no_admissible_task_rate")),
        "fallback_return_candidate_count": int_number(adm.get("fallback_return_candidate_count")),
        "template_like_candidate_count": int_number(adm.get("template_like_candidate_count")),
        "public_leakage_count": int_number(adm.get("public_leakage_count")),
        "private_heldout_pass_rate": number(heldout.get("private_residual_v3_heldout_pass_rate")),
        "learned_candidate_task_pass_rate": number(heldout.get("learned_candidate_task_pass_rate")),
        "residual_gate_trigger_state": residual_gate.get("_trigger_state"),
        "matched_sts_selected_pass_rate_delta": number(residual_gate.get("matched_sts_selected_pass_rate_delta")),
        "sts_policy_selected_pass_rate": number(sts_policy.get("sts_policy_selected_pass_rate")),
        "sts_policy_delta": number(sts_policy.get("selected_pass_delta_sts_policy_minus_non_sts_policy")),
        "candidate_floor_trigger_state": candidate_floor_quality.get("trigger_state"),
        "candidate_floor_admissibility_ready": bool(candidate_floor_quality.get("admissibility_ready")),
        "candidate_floor_semantic_quality_ready": bool(candidate_floor_quality.get("semantic_quality_ready")),
        "candidate_floor_candidate_coverage_rate": number(candidate_floor_quality.get("candidate_coverage_rate")),
        "candidate_floor_no_candidate_task_count": int_number(candidate_floor_quality.get("no_candidate_task_count")),
        "candidate_floor_private_trained_pass_rate": number(candidate_floor_quality.get("private_trained_pass_rate")),
        "candidate_floor_private_trained_passed": int_number(candidate_floor_quality.get("private_trained_passed")),
        "candidate_floor_private_eval_task_count": int_number(candidate_floor_quality.get("private_eval_task_count")),
        "candidate_floor_weak_family_count": int_number(candidate_floor_quality.get("weak_family_count")),
        "candidate_floor_weak_families": candidate_floor_quality.get("weak_families"),
        "fresh_private_probe_trigger_state": fresh_probe_quality.get("trigger_state"),
        "fresh_private_probe_pass_rate": number(fresh_probe_quality.get("private_trained_pass_rate")),
        "fresh_private_probe_task_count": int_number(fresh_probe_quality.get("private_eval_task_count")),
        "fresh_private_probe_candidate_coverage_rate": number(fresh_probe_quality.get("candidate_coverage_rate")),
        "fresh_private_probe_covered_categories": fresh_probe_quality.get("covered_categories"),
        "fresh_private_probe_unresolved_categories": fresh_probe_quality.get("unresolved_categories"),
        "fresh_private_probe_missing_categories": fresh_probe_quality.get("missing_categories"),
        "fresh_private_probe_category_min_pass_rates": fresh_probe_quality.get("category_min_pass_rates"),
        "fresh_private_probe_targeted_private_eval_category_counts": fresh_probe_quality.get(
            "targeted_private_eval_category_counts"
        ),
        "fresh_private_probe_fallback_return_candidate_count": int_number(
            fresh_probe_quality.get("fallback_return_candidate_count")
        ),
        "fresh_private_probe_template_like_candidate_count": int_number(
            fresh_probe_quality.get("template_like_candidate_count")
        ),
        "fresh_private_probe_external_inference_calls": int_number(fresh_probe_quality.get("external_inference_calls")),
        "fresh_public_residual_count": int_number(fresh_public_residual_counts.get(category)),
        "fresh_public_residual_counts": fresh_public_residual_counts,
    }
    family = {
        "selector_ranking_miss": "semantic_ranker_selection_v1",
        "return_type_shape": "return_interface_fidelity_v3",
        "verifier_mismatch": "verifier_mismatch_property_v3",
        "io_contract_stdin": "livecodebench_stdin_proxy_v1",
        "spent_calibration_no_admissible_resolved_by_current_manifest": "no_admissible_cleanup_v3",
        "edge_cases": "edge_case_semantic_private_curriculum",
        "external_dependency_missing": "dependency_free_candidate_private_curriculum",
    }.get(category)
    if family:
        base["mapped_private_family"] = family
        base["mapped_private_family_admissibility"] = family_rates.get(family)
        base["mapped_private_family_heldout"] = heldout_family_rates.get(family)
    if category == "candidate_manifest_slice_mismatch":
        base["alignment_evidence_required"] = "future one-shot candidate manifest must be frozen against the exact public slice before execution"
        base["alignment_preflight_trigger_state"] = alignment_preflight.get("_trigger_state")
        base["alignment_preflight_ready"] = alignment_preflight.get("_alignment_preflight_ready")
        base["alignment_case_manifest"] = alignment_preflight.get("case_manifest")
        base["alignment_case_manifest_row_count"] = alignment_preflight.get("case_manifest_row_count")
        base["alignment_candidate_manifest_bound_to_case_manifest"] = alignment_preflight.get(
            "candidate_manifest_bound_to_case_manifest"
        )
        base["alignment_candidate_manifest_preexists_before_run"] = alignment_preflight.get(
            "candidate_manifest_preexists_before_run"
        )
        base["alignment_public_tests_used"] = alignment_preflight.get("public_tests_used")
        base["alignment_public_solutions_used"] = alignment_preflight.get("public_solutions_used")
        base["alignment_training_rows_written"] = alignment_preflight.get("training_rows_written")
        base["alignment_external_inference_calls"] = alignment_preflight.get("external_inference_calls")
    if category == "algorithm_choice":
        base["nearest_private_family"] = "semantic_ranker_selection_v1"
    return base


def target_state(category: str, validation: dict[str, Any], evidence: dict[str, Any]) -> str:
    if not validation.get("valid"):
        return "blocked"
    if has_no_cheat_failure(evidence):
        return "blocked"
    if int_number(evidence.get("fresh_public_residual_count")) > 0:
        covered_categories = set(str(item) for item in as_list(evidence.get("fresh_private_probe_covered_categories")))
        if category in covered_categories:
            return "covered_by_current_private_evidence"
        return "needs_private_ablation"
    if category == "candidate_manifest_slice_mismatch":
        if alignment_preflight_closes_target(evidence):
            return "covered_by_current_private_evidence"
        return "ready"
    if fresh_private_probe_closes_category(category, evidence):
        return "covered_by_current_private_evidence"
    if category == "spent_calibration_no_admissible_resolved_by_current_manifest":
        if evidence.get("candidate_floor_admissibility_ready") is True:
            return "covered_by_current_private_evidence"
        return "needs_private_ablation"
    if category in {
        "algorithm_choice",
        "return_type_shape",
        "verifier_mismatch",
        "selector_ranking_miss",
        "io_contract_stdin",
        "edge_cases",
        "external_dependency_missing",
    }:
        if evidence.get("candidate_floor_semantic_quality_ready") is not True:
            return "needs_private_ablation"
    family_adm = evidence.get("mapped_private_family_admissibility")
    family_heldout = evidence.get("mapped_private_family_heldout")
    if isinstance(family_adm, dict) and isinstance(family_heldout, dict):
        if number(family_adm.get("selected_pass_rate")) >= 1.0 and number(family_heldout.get("pass_rate")) >= 1.0:
            return "covered_by_current_private_evidence"
        return "needs_private_ablation"
    if category == "algorithm_choice":
        if number(evidence.get("selected_pass_rate")) >= 1.0 and number(evidence.get("learned_candidate_task_pass_rate")) >= 1.0:
            return "covered_by_current_private_evidence"
        return "needs_private_ablation"
    return "needs_private_ablation"


def fresh_private_probe_closes_category(category: str, evidence: dict[str, Any]) -> bool:
    covered_categories = set(str(item) for item in as_list(evidence.get("fresh_private_probe_covered_categories")))
    unresolved_categories = set(str(item) for item in as_list(evidence.get("fresh_private_probe_unresolved_categories")))
    missing_categories = set(str(item) for item in as_list(evidence.get("fresh_private_probe_missing_categories")))
    min_rates = obj(evidence, "fresh_private_probe_category_min_pass_rates")
    category_counts = obj(evidence, "fresh_private_probe_targeted_private_eval_category_counts")
    return bool(
        category in covered_categories
        and category not in unresolved_categories
        and category not in missing_categories
        and number(min_rates.get(category)) >= 1.0
        and int_number(category_counts.get(category)) > 0
        and evidence.get("fresh_private_probe_trigger_state") in {"GREEN", "YELLOW"}
        and number(evidence.get("fresh_private_probe_pass_rate")) >= PRIVATE_TRANSFER_SEMANTIC_FLOOR
        and number(evidence.get("fresh_private_probe_candidate_coverage_rate")) >= 1.0
        and int_number(evidence.get("fresh_private_probe_fallback_return_candidate_count")) == 0
        and int_number(evidence.get("fresh_private_probe_template_like_candidate_count")) == 0
        and int_number(evidence.get("fresh_private_probe_external_inference_calls")) == 0
    )


def has_no_cheat_failure(evidence: dict[str, Any]) -> bool:
    return (
        int_number(evidence.get("fallback_return_candidate_count")) > 0
        or int_number(evidence.get("template_like_candidate_count")) > 0
        or int_number(evidence.get("public_leakage_count")) > 0
    )


def alignment_preflight_closes_target(evidence: dict[str, Any]) -> bool:
    return bool(
        evidence.get("alignment_preflight_trigger_state") in {"GREEN", "YELLOW"}
        and evidence.get("alignment_preflight_ready") is True
        and int_number(evidence.get("alignment_case_manifest_row_count")) == 320
        and evidence.get("alignment_candidate_manifest_bound_to_case_manifest") is True
        and evidence.get("alignment_candidate_manifest_preexists_before_run") is False
        and evidence.get("alignment_public_tests_used") is False
        and evidence.get("alignment_public_solutions_used") is False
        and int_number(evidence.get("alignment_training_rows_written")) == 0
        and int_number(evidence.get("alignment_external_inference_calls")) == 0
    )


def next_private_action(category: str, state: str) -> str:
    if state == "blocked":
        return "Fix metadata/no-cheat validation before consuming this target."
    if category == "candidate_manifest_slice_mismatch":
        return "Freeze exact one-shot task slice and candidate manifest together before any future public calibration."
    if state == "covered_by_current_private_evidence":
        return "Keep this target in regression coverage; do not generate more private rows merely to reprove saturation."
    if category == "selector_ranking_miss":
        return "Run same-seed private selector ablation under equal candidate budget and require selected-pass lift with no oracle regression."
    if category == "return_type_shape":
        return "Run private return/type/string shape ablation and require first-rank shape-compatible selected-pass lift."
    if category == "verifier_mismatch":
        return "Run private verifier-contract ablation and require residual mismatch count reduction without public content."
    if category == "algorithm_choice":
        return "Run private algorithm-choice semantic ablation and require wrong-answer residual reduction."
    if category == "edge_cases":
        return "Run private edge-case semantic ablation and require boundary residual reduction without mainline regression."
    if category == "external_dependency_missing":
        return "Run private dependency-free candidate ablation and require dependency-unavailable faults to fall without fallback returns."
    return "Run private-only same-seed ablation for this target family."


def state_order(state: str) -> int:
    return {
        "blocked": 0,
        "needs_private_ablation": 1,
        "ready": 2,
        "covered_by_current_private_evidence": 3,
    }.get(state, 9)


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = obj(report, "summary")
    lines = [
        "# Private Residual Target Consumer v1",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Target rows: `{summary.get('target_rows')}`",
        f"- Valid target rows: `{summary.get('valid_target_rows')}`",
        f"- Repair queue rows: `{summary.get('repair_queue_rows')}`",
        f"- Ready targets: `{summary.get('ready_target_count')}`",
        f"- Covered targets: `{summary.get('covered_target_count')}`",
        f"- Needs private ablation: `{summary.get('needs_private_ablation_count')}`",
        f"- Blocked targets: `{summary.get('blocked_target_count')}`",
        f"- Unresolved targets: `{summary.get('unresolved_target_count')}`",
        f"- Target coverage rate: `{summary.get('target_coverage_rate')}`",
        f"- Queue path: `{summary.get('queue_path')}`",
        f"- Training rows written: `{summary.get('training_rows_written')}`",
        "",
        "## Queue",
    ]
    for row in as_list(report.get("repair_queue")):
        lines.append(
            f"- `{row.get('state')}` P{row.get('priority')} `{row.get('canonical_category')}` -> `{row.get('private_curriculum_family')}`"
        )
        if row.get("state") != "covered_by_current_private_evidence":
            lines.append(f"  Next: {row.get('next_private_action')}")
    lines.extend(["", "## Gates"])
    for row in as_list(report.get("gates")):
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}`")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path, default: Any) -> Any:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return data if isinstance(data, dict) else default


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def obj(value: Any, key: str) -> dict[str, Any]:
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


def number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def sha256_text(value: str) -> str:
    import hashlib

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
