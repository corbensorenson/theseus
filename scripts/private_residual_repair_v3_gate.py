#!/usr/bin/env python3
"""Gate Private Residual Repair v3 before any future public calibration.

This script does not launch training or public benchmarks. It records whether
the v3 private curriculum exists and whether downstream private-only evidence
has cleared the thresholds required before proposing another bounded public
calibration.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

PRIVATE_HELDOUT_PASS_RATE_MIN = 0.70
PRIVATE_LEARNED_CANDIDATE_PASS_RATE_MIN = 0.70
NO_ADMISSIBLE_RATE_MAX = 0.03
LIVE_STDIN_PROXY_PASS_COUNT_MIN = 1
PRIVATE_CANDIDATE_TASK_COVERAGE_MIN = 0.97
PRIVATE_FULL_BODY_CANDIDATE_COUNT_MIN = 1
PRIVATE_HELDOUT_TASK_COUNT_MIN = 240


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", default="reports/targeted_private_residual_curriculum_v3.json")
    parser.add_argument("--private-heldout", default="")
    parser.add_argument("--decoder-gate", default="")
    parser.add_argument("--sts-causal", default="")
    parser.add_argument("--sts-ablation", default="reports/private_residual_v3_sts_ablation.json")
    parser.add_argument("--maturity-audit", default="reports/maturity_integrity_audit_wide_public_seed23_5x32_interface_floor_v1.json")
    parser.add_argument("--operator-lock", default="reports/public_calibration_operator_lock.flag")
    parser.add_argument("--out", default="reports/private_residual_repair_v3_gate.json")
    parser.add_argument("--markdown-out", default="reports/private_residual_repair_v3_gate.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    curriculum_path = resolve(args.curriculum)
    heldout_path = resolve(args.private_heldout) if args.private_heldout else None
    decoder_path = resolve(args.decoder_gate) if args.decoder_gate else None
    sts_path = resolve(args.sts_causal) if args.sts_causal else None
    sts_ablation_path = resolve(args.sts_ablation) if args.sts_ablation else None
    maturity_path = resolve(args.maturity_audit)
    lock_path = resolve(args.operator_lock)

    curriculum = read_json(curriculum_path, {})
    heldout = read_json(heldout_path, {}) if heldout_path is not None else {}
    decoder = read_json(decoder_path, {}) if decoder_path is not None else {}
    sts = read_json(sts_path, {}) if sts_path is not None else {}
    sts_ablation = read_json(sts_ablation_path, {}) if sts_ablation_path is not None else {}
    maturity = read_json(maturity_path, {})

    curriculum_summary = object_field(curriculum, "summary")
    heldout_summary = object_field(heldout, "summary")
    decoder_summary = object_field(decoder, "summary") or decoder
    sts_summary = object_field(sts, "summary") or sts
    sts_ablation_summary = object_field(sts_ablation, "summary")
    maturity_summary = object_field(maturity, "summary")
    private_manifest = object_field(decoder_summary, "private_candidate_manifest_diagnostics")
    public_manifest = object_field(decoder_summary, "public_candidate_manifest_diagnostics")

    private_heldout_pass_rate = first_number(
        heldout_summary,
        "private_residual_v3_heldout_pass_rate",
        "private_heldout_pass_rate",
        "heldout_pass_rate",
        "private_pass_rate",
    )
    private_heldout_task_count = first_number(
        heldout_summary,
        "private_residual_v3_heldout_task_count",
        "heldout_task_count",
        "task_count",
    )
    private_heldout_task_limit = first_number(heldout_summary, "private_residual_v3_heldout_task_limit")
    full_heldout_score_present = (
        private_heldout_task_count is not None
        and private_heldout_task_count >= PRIVATE_HELDOUT_TASK_COUNT_MIN
        and (private_heldout_task_limit is None or private_heldout_task_limit == 0)
    )
    live_stdin_proxy_pass_count = first_number(
        heldout_summary,
        "livecodebench_private_stdin_proxy_pass_count",
        "stdin_proxy_pass_count",
        "livecodebench_stdin_proxy_pass_count",
    )
    learned_candidate_pass_rate = first_number(heldout_summary, "learned_candidate_task_pass_rate")
    learned_candidate_passes = first_number(heldout_summary, "learned_candidate_task_passes")
    structural_action_pass_rate = first_number(heldout_summary, "structural_action_candidate_task_pass_rate")
    structural_action_passes = first_number(heldout_summary, "structural_action_candidate_task_passes")
    diagnostic_adapter_pass_rate = first_number(heldout_summary, "diagnostic_adapter_task_pass_rate")
    diagnostic_adapter_passes = first_number(heldout_summary, "diagnostic_adapter_task_passes")
    adapter_off_scoring = heldout_summary.get("adapter_off_scoring")
    heldout_fallback_flag_count = first_number(heldout_summary, "fallback_return_candidate_count")
    no_admissible_rate = first_number(
        heldout_summary,
        "no_admissible_task_rate",
        "private_no_admissible_task_rate",
        "private_no_admissible_rate",
    )
    if no_admissible_rate is None:
        no_admissible_rate = first_number(
            decoder_summary,
            "private_no_admissible_task_rate",
            "public_no_admissible_task_rate",
            "no_admissible_task_rate",
        )
    private_candidate_task_coverage = first_number(private_manifest, "task_coverage")
    if no_admissible_rate is None and private_candidate_task_coverage is not None:
        no_admissible_rate = round(max(0.0, 1.0 - private_candidate_task_coverage), 6)
    heldout_sts_delta = first_number(
        heldout_summary,
        "private_residual_v3_sts_delta",
    )
    sts_delta = heldout_sts_delta
    if sts_delta is None:
        sts_delta = first_number(
            sts_summary,
            "private_residual_v3_sts_delta",
            "sts_private_eligible_coverage_delta",
            "sts_public_eligible_coverage_delta",
            "eval_token_accuracy_delta",
            "max_abs_delta",
        )
    sts_regressions = first_number(
        heldout_summary,
        "private_residual_v3_sts_regressions",
    )
    if sts_regressions is None:
        sts_regressions = first_number(
            sts_summary,
            "private_residual_v3_sts_regressions",
            "task_level_regressions",
            "regression_count",
        )
    sts_token_accuracy_delta = first_number(sts_summary, "eval_token_accuracy_delta")
    matched_sts_task_count = first_number(sts_ablation_summary, "heldout_task_count")
    matched_sts_budget_equal = sts_ablation_summary.get("matched_candidate_budget_equal")
    matched_sts_selected_delta = first_number(sts_ablation_summary, "selected_pass_rate_delta_sts_minus_non_sts")
    matched_sts_oracle_delta = first_number(sts_ablation_summary, "oracle_pass_rate_delta_sts_minus_non_sts")
    matched_sts_selection_gap_delta = first_number(sts_ablation_summary, "selection_gap_delta_non_sts_minus_sts")
    matched_sts_effect = str(sts_ablation_summary.get("effect_interpretation") or "")
    matched_sts_clean = (
        sts_ablation.get("trigger_state") == "GREEN"
        and matched_sts_task_count is not None
        and matched_sts_task_count >= PRIVATE_HELDOUT_TASK_COUNT_MIN
        and matched_sts_budget_equal is True
        and first_number(sts_ablation_summary, "public_candidate_rows") == 0
        and first_number(sts_ablation_summary, "fallback_return_candidate_count") == 0
        and first_number(sts_ablation_summary, "external_inference_calls") == 0
        and sts_ablation_summary.get("public_tests_used") is False
        and sts_ablation_summary.get("public_solutions_used") is False
        and sts_ablation_summary.get("teacher_rows_used") is False
    )
    train_once_completed = (
        decoder.get("trigger_state") == "GREEN"
        and str(decoder.get("run_status") or decoder_summary.get("run_status") or "").lower() == "completed"
        and (decoder.get("private_only") is True or decoder_summary.get("private_only") is True)
    )
    private_full_body_candidate_count = first_number(private_manifest, "full_body_candidate_count")
    private_candidate_row_count = first_number(private_manifest, "row_count")
    private_program_synthesis_promotion_ready_count = first_number(
        private_manifest,
        "program_synthesis_promotion_ready_count",
    )
    private_template_like_candidate_count = first_number(private_manifest, "template_like_candidate_count")
    private_placeholder_scaffold_count = first_number(private_manifest, "placeholder_scaffold_count")
    public_candidate_row_count = first_number(public_manifest, "row_count")
    decoder_external_inference_calls = (
        first_number(decoder, "external_inference_calls")
        or first_number(decoder_summary, "external_inference_calls")
        or 0.0
    )
    private_manifest_safety = object_field(private_manifest, "safety")
    fallback_flag_count = fallback_flag_count_from_manifest(str(private_manifest.get("path") or ""))
    private_manifest_public_clean = (
        public_candidate_row_count == 0
        and decoder_external_inference_calls == 0
        and private_manifest_safety.get("public_tests_or_solutions_used") is False
    )
    fallback_returns_clean = (
        fallback_flag_count is not None
        and fallback_flag_count == 0
        and (heldout_fallback_flag_count is None or heldout_fallback_flag_count == 0)
        and (private_placeholder_scaffold_count or 0) == 0
    )
    hard_blockers = int(number(maturity_summary.get("hard_blocker_count")))
    evidence_blockers = int(number(maturity_summary.get("evidence_blocker_count")))
    leak_hits = int(number(maturity_summary.get("manifest_public_leak_hit_count")))

    gates = [
        gate("curriculum_report_green", curriculum.get("trigger_state") == "GREEN", rel_or_missing(curriculum_path)),
        gate("v3_private_train_rows_written", int(number(curriculum_summary.get("private_train_row_count"))) >= 480, curriculum_summary.get("private_train_row_count")),
        gate("v3_private_heldout_rows_written", int(number(curriculum_summary.get("private_heldout_row_count"))) >= 120, curriculum_summary.get("private_heldout_row_count")),
        gate("v3_private_solution_tests_pass", int(number(curriculum_summary.get("private_train_solution_failures"))) == 0 and int(number(curriculum_summary.get("private_heldout_solution_failures"))) == 0, {
            "train_failures": curriculum_summary.get("private_train_solution_failures"),
            "heldout_failures": curriculum_summary.get("private_heldout_solution_failures"),
        }),
        readiness_gate(
            "private_train_once_candidate_manifest_green",
            train_once_completed,
            {
                "trigger_state": decoder.get("trigger_state"),
                "run_status": decoder.get("run_status") or decoder_summary.get("run_status"),
                "private_only": decoder.get("private_only") or decoder_summary.get("private_only"),
                "source": rel_or_missing(decoder_path),
            },
            missing=not decoder,
        ),
        readiness_gate(
            "private_candidate_manifest_task_coverage_floor",
            private_candidate_task_coverage is not None
            and private_candidate_task_coverage >= PRIVATE_CANDIDATE_TASK_COVERAGE_MIN,
            {
                "observed": private_candidate_task_coverage,
                "minimum": PRIVATE_CANDIDATE_TASK_COVERAGE_MIN,
                "source": rel_or_missing(decoder_path),
            },
            missing=private_candidate_task_coverage is None,
        ),
        readiness_gate(
            "private_full_body_candidates_present",
            private_full_body_candidate_count is not None
            and private_full_body_candidate_count >= PRIVATE_FULL_BODY_CANDIDATE_COUNT_MIN,
            {
                "observed": private_full_body_candidate_count,
                "minimum": PRIVATE_FULL_BODY_CANDIDATE_COUNT_MIN,
                "source": rel_or_missing(decoder_path),
            },
            missing=private_full_body_candidate_count is None,
        ),
        readiness_gate(
            "private_candidate_manifest_public_boundary_clean",
            private_manifest_public_clean,
            {
                "public_candidate_row_count": public_candidate_row_count,
                "decoder_external_inference_calls": decoder_external_inference_calls,
                "private_manifest_public_tests_or_solutions_used": private_manifest_safety.get("public_tests_or_solutions_used"),
                "source": rel_or_missing(decoder_path),
            },
            missing=public_candidate_row_count is None,
        ),
        readiness_gate(
            "fallback_returns_not_used",
            fallback_returns_clean,
            {
                "expression_memory_fallback_count": fallback_flag_count,
                "heldout_fallback_return_candidate_count": heldout_fallback_flag_count,
                "placeholder_scaffold_count": private_placeholder_scaffold_count,
                "source": rel_or_missing(decoder_path),
                "rule": "fallback returns are disallowed; candidate manifests must use learned/verified bodies or explicit no-candidate residuals",
            },
            missing=fallback_flag_count is None,
        ),
        readiness_gate(
            "private_residual_v3_full_heldout_score_present",
            full_heldout_score_present,
            {
                "observed_task_count": private_heldout_task_count,
                "minimum_task_count": PRIVATE_HELDOUT_TASK_COUNT_MIN,
                "task_limit": private_heldout_task_limit,
                "source": rel_or_missing(heldout_path),
                "rule": "canary slices are diagnostic only; full v3 gate readiness needs the full private heldout score",
            },
            missing=private_heldout_task_count is None,
        ),
        readiness_gate(
            "private_residual_v3_adapter_off_scoring_present",
            adapter_off_scoring is True,
            {
                "observed": adapter_off_scoring,
                "source": rel_or_missing(heldout_path),
                "rule": "diagnostic semantic adapters are diagnostic only and must be excluded from heldout pass credit",
            },
            missing=adapter_off_scoring is None,
        ),
        readiness_gate(
            "private_residual_v3_heldout_pass_rate_floor",
            private_heldout_pass_rate is not None and private_heldout_pass_rate >= PRIVATE_HELDOUT_PASS_RATE_MIN,
            {
                "observed": private_heldout_pass_rate,
                "minimum": PRIVATE_HELDOUT_PASS_RATE_MIN,
                "source": rel_or_missing(heldout_path),
            },
            missing=private_heldout_pass_rate is None,
        ),
        readiness_gate(
            "private_residual_v3_learned_candidate_pass_rate_floor",
            learned_candidate_pass_rate is not None
            and learned_candidate_pass_rate >= PRIVATE_LEARNED_CANDIDATE_PASS_RATE_MIN,
            {
                "observed": learned_candidate_pass_rate,
                "minimum": PRIVATE_LEARNED_CANDIDATE_PASS_RATE_MIN,
                "learned_candidate_passes": learned_candidate_passes,
                "structural_action_candidate_pass_rate": structural_action_pass_rate,
                "structural_action_candidate_passes": structural_action_passes,
                "diagnostic_adapter_pass_rate": diagnostic_adapter_pass_rate,
                "diagnostic_adapter_passes": diagnostic_adapter_passes,
                "source": rel_or_missing(heldout_path),
                "rule": "diagnostic semantic adapters can repair candidate-floor behavior, but they do not count as student learning or promotion evidence",
            },
            missing=learned_candidate_pass_rate is None,
        ),
        readiness_gate(
            "no_admissible_rate_floor",
            no_admissible_rate is not None and no_admissible_rate <= NO_ADMISSIBLE_RATE_MAX,
            {
                "observed": no_admissible_rate,
                "maximum": NO_ADMISSIBLE_RATE_MAX,
                "source": rel_or_missing(decoder_path),
            },
            missing=no_admissible_rate is None,
        ),
        readiness_gate(
            "livecodebench_private_stdin_proxy_nonzero",
            live_stdin_proxy_pass_count is not None and live_stdin_proxy_pass_count >= LIVE_STDIN_PROXY_PASS_COUNT_MIN,
            {
                "observed": live_stdin_proxy_pass_count,
                "minimum": LIVE_STDIN_PROXY_PASS_COUNT_MIN,
                "source": rel_or_missing(heldout_path),
            },
            missing=live_stdin_proxy_pass_count is None,
        ),
        readiness_gate(
            "matched_sts_generation_ablation_clean",
            matched_sts_clean,
            {
                "source": rel_or_missing(sts_ablation_path),
                "trigger_state": sts_ablation.get("trigger_state"),
                "heldout_task_count": matched_sts_task_count,
                "matched_candidate_budget_equal": matched_sts_budget_equal,
                "selected_pass_rate_delta_sts_minus_non_sts": matched_sts_selected_delta,
                "oracle_pass_rate_delta_sts_minus_non_sts": matched_sts_oracle_delta,
                "selection_gap_delta_non_sts_minus_sts": matched_sts_selection_gap_delta,
                "effect_interpretation": matched_sts_effect,
                "public_candidate_rows": first_number(sts_ablation_summary, "public_candidate_rows"),
                "fallback_return_candidate_count": first_number(sts_ablation_summary, "fallback_return_candidate_count"),
                "external_inference_calls": first_number(sts_ablation_summary, "external_inference_calls"),
            },
            missing=not sts_ablation,
        ),
        gate("maturity_hard_and_evidence_clean", hard_blockers == 0 and evidence_blockers == 0 and leak_hits == 0, {
            "hard_blockers": hard_blockers,
            "evidence_blockers": evidence_blockers,
            "manifest_public_leak_hit_count": leak_hits,
            "source": rel_or_missing(maturity_path),
        }),
        gate("public_calibration_operator_lock_active", lock_path.exists(), rel_or_missing(lock_path)),
    ]
    hard_failures = [row for row in gates if row["status"] == "FAILED"]
    pending = [row for row in gates if row["status"] == "PENDING"]
    trigger_state = "RED" if hard_failures else ("YELLOW" if pending else "GREEN")
    return {
        "policy": "project_theseus_private_residual_repair_v3_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "public_calibration_allowed": trigger_state == "GREEN" and not lock_path.exists(),
        "summary": {
            "curriculum_ready": curriculum.get("trigger_state") == "GREEN",
            "private_heldout_pass_rate": private_heldout_pass_rate,
            "private_heldout_pass_rate_minimum": PRIVATE_HELDOUT_PASS_RATE_MIN,
            "private_heldout_adapter_off_scoring": adapter_off_scoring,
            "private_learned_candidate_pass_rate": learned_candidate_pass_rate,
            "private_learned_candidate_pass_rate_minimum": PRIVATE_LEARNED_CANDIDATE_PASS_RATE_MIN,
            "private_learned_candidate_passes": learned_candidate_passes,
            "private_structural_action_candidate_pass_rate": structural_action_pass_rate,
            "private_structural_action_candidate_passes": structural_action_passes,
            "private_diagnostic_adapter_pass_rate": diagnostic_adapter_pass_rate,
            "private_diagnostic_adapter_passes": diagnostic_adapter_passes,
            "private_heldout_task_count": private_heldout_task_count,
            "private_heldout_task_count_minimum": PRIVATE_HELDOUT_TASK_COUNT_MIN,
            "private_heldout_task_limit": private_heldout_task_limit,
            "full_heldout_score_present": full_heldout_score_present,
            "no_admissible_rate": no_admissible_rate,
            "no_admissible_rate_maximum": NO_ADMISSIBLE_RATE_MAX,
            "livecodebench_private_stdin_proxy_pass_count": live_stdin_proxy_pass_count,
            "sts_delta": sts_delta,
            "sts_regressions": sts_regressions,
            "sts_token_accuracy_delta": sts_token_accuracy_delta,
            "matched_sts_ablation_clean": matched_sts_clean,
            "matched_sts_ablation_trigger_state": sts_ablation.get("trigger_state"),
            "matched_sts_task_count": matched_sts_task_count,
            "matched_sts_candidate_budget_equal": matched_sts_budget_equal,
            "matched_sts_selected_pass_rate_delta": matched_sts_selected_delta,
            "matched_sts_oracle_pass_rate_delta": matched_sts_oracle_delta,
            "matched_sts_selection_gap_delta": matched_sts_selection_gap_delta,
            "matched_sts_effect_interpretation": matched_sts_effect,
            "private_train_once_completed": train_once_completed,
            "private_candidate_task_coverage": private_candidate_task_coverage,
            "private_candidate_task_coverage_minimum": PRIVATE_CANDIDATE_TASK_COVERAGE_MIN,
            "private_candidate_row_count": private_candidate_row_count,
            "private_full_body_candidate_count": private_full_body_candidate_count,
            "private_program_synthesis_promotion_ready_count": private_program_synthesis_promotion_ready_count,
            "private_template_like_candidate_count": private_template_like_candidate_count,
            "private_placeholder_scaffold_count": private_placeholder_scaffold_count,
            "public_candidate_row_count": public_candidate_row_count,
            "candidate_manifest_fallback_flag_count": fallback_flag_count,
            "heldout_fallback_return_candidate_count": heldout_fallback_flag_count,
            "fallback_returns_clean": fallback_returns_clean,
            "maturity_hard_blockers": hard_blockers,
            "maturity_evidence_blockers": evidence_blockers,
            "manifest_public_leak_hit_count": leak_hits,
            "pending_gate_count": len(pending),
            "failed_gate_count": len(hard_failures),
            "public_calibration_operator_lock_active": lock_path.exists(),
        },
        "inputs": {
            "curriculum": rel_or_missing(curriculum_path),
            "private_heldout": rel_or_missing(heldout_path),
            "decoder_gate": rel_or_missing(decoder_path),
            "sts_causal": rel_or_missing(sts_path),
            "sts_ablation": rel_or_missing(sts_ablation_path),
            "maturity_audit": rel_or_missing(maturity_path),
            "operator_lock": rel_or_missing(lock_path),
        },
        "gates": gates,
        "next_actions": next_actions(trigger_state, pending, hard_failures, heldout_summary),
        "rules": {
            "public_calibration": "Do not unlock another public calibration until this gate is GREEN under private-only evidence.",
            "public_data": "Public benchmark reports may supply sanitized residual category counts only; public tests and solutions must remain excluded.",
        },
    }


def next_actions(
    trigger_state: str,
    pending: list[dict[str, Any]],
    hard_failures: list[dict[str, Any]],
    heldout_summary: dict[str, Any],
) -> list[str]:
    if hard_failures:
        return ["Fix failed v3 curriculum/integrity gates before training or public calibration."]
    if trigger_state == "YELLOW":
        names = [row["gate"] for row in pending]
        actions: list[str] = []
        learned_rate = first_number(heldout_summary, "learned_candidate_task_pass_rate")
        diagnostic_rate = first_number(heldout_summary, "diagnostic_adapter_task_pass_rate")
        if "private_residual_v3_learned_candidate_pass_rate_floor" in names:
            actions.append(
                "Improve learned/student candidate generation or ranking; diagnostic adapters fixed candidate-floor behavior but do not count as learning evidence."
            )
            actions.append(
                f"Current learned/student pass rate: {learned_rate}; diagnostic adapter pass rate: {diagnostic_rate}."
            )
        if "private_residual_v3_heldout_pass_rate_floor" in names:
            actions.append("Repair private v3 heldout semantics before public calibration.")
        if "no_admissible_rate_floor" in names:
            actions.append("Repair candidate-floor generation for any remaining no-admissible v3 families.")
        if "livecodebench_private_stdin_proxy_nonzero" in names:
            actions.append("Repair stdin parser/output-format candidates until LiveCodeBench-style private stdin proxy is nonzero.")
        if "matched_sts_generation_ablation_clean" in names:
            actions.append("Produce a clean equal-budget STS-on/non-STS structural ablation before attributing STS lift.")
        actions.append(f"Pending gates: {', '.join(names)}")
        return actions
    return ["Private v3 gates are clear; operator may decide whether to propose one new bounded public calibration surface, then relock immediately."]


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "status": "PASSED" if passed else "FAILED", "evidence": evidence}


def pending_gate(name: str, passed: bool, evidence: Any, *, missing: bool) -> dict[str, Any]:
    if missing:
        return {"gate": name, "passed": False, "status": "PENDING", "evidence": evidence}
    return gate(name, passed, evidence)


def readiness_gate(name: str, passed: bool, evidence: Any, *, missing: bool) -> dict[str, Any]:
    if passed:
        return {"gate": name, "passed": True, "status": "PASSED", "evidence": evidence}
    status = "PENDING" if missing or not passed else "PASSED"
    return {"gate": name, "passed": False, "status": status, "evidence": evidence}


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def first_number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row and row.get(key) is not None:
            return number(row.get(key))
    return None


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fallback_flag_count_from_manifest(path_value: str) -> int | None:
    if not path_value:
        return None
    path = resolve(path_value)
    if not path.exists():
        return None
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                mode = str(row.get("candidate_generation_mode") or "").lower()
                if bool(row.get("expression_memory_fallback")) or (
                    "fallback" in mode and "fallback_skipped" not in mode
                ):
                    count += 1
    except OSError:
        return None
    return count


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Private Residual Repair V3 Gate",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Curriculum ready: {summary.get('curriculum_ready')}",
        f"- Full heldout score present: {summary.get('full_heldout_score_present')}",
        f"- Heldout tasks: {summary.get('private_heldout_task_count')} / {summary.get('private_heldout_task_count_minimum')} (limit={summary.get('private_heldout_task_limit')})",
        f"- Private heldout pass rate: {summary.get('private_heldout_pass_rate')}",
        f"- Adapter-off scoring: {summary.get('private_heldout_adapter_off_scoring')}",
        f"- Learned/student pass rate: {summary.get('private_learned_candidate_pass_rate')}",
        f"- Structural-action pass rate: {summary.get('private_structural_action_candidate_pass_rate')}",
        f"- Diagnostic adapter pass rate: {summary.get('private_diagnostic_adapter_pass_rate')}",
        f"- Private candidate coverage: {summary.get('private_candidate_task_coverage')}",
        f"- Full-body private candidates: {summary.get('private_full_body_candidate_count')}",
        f"- Fallback return flags: {summary.get('candidate_manifest_fallback_flag_count')} manifest / {summary.get('heldout_fallback_return_candidate_count')} heldout",
        f"- No-admissible rate: {summary.get('no_admissible_rate')}",
        f"- LiveCodeBench stdin proxy pass count: {summary.get('livecodebench_private_stdin_proxy_pass_count')}",
        f"- STS delta: {summary.get('sts_delta')}",
        f"- STS regressions: {summary.get('sts_regressions')}",
        f"- Matched STS ablation clean: {summary.get('matched_sts_ablation_clean')}",
        f"- Matched STS selected delta: {summary.get('matched_sts_selected_pass_rate_delta')}",
        f"- Matched STS oracle delta: {summary.get('matched_sts_oracle_pass_rate_delta')}",
        f"- Matched STS interpretation: {summary.get('matched_sts_effect_interpretation')}",
        f"- Pending gates: {summary.get('pending_gate_count')}",
        f"- Failed gates: {summary.get('failed_gate_count')}",
        f"- Public calibration lock active: {summary.get('public_calibration_operator_lock_active')}",
        "",
        "## Next Actions",
    ]
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_or_missing(path: Path | None) -> str:
    if path is None:
        return ""
    rel_path = rel(path)
    return rel_path if path.exists() else f"{rel_path} (missing)"


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
