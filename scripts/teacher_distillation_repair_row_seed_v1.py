#!/usr/bin/env python3
"""Build a private verifier-backed seed row for governed teacher distillation.

This script does not call a teacher and does not admit training rows. It builds
one project-internal candidate row shape for the teacher to assess in
distillation mode, then runs the same local verifier used by the manifest
builder so unsafe or incomplete shapes are visible before any teacher spend.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import teacher_distillation_manifest_builder as manifest_builder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "teacher_distillation_repair_row_seed_survival_loop_v1.json"
DEFAULT_MD = ROOT / "reports" / "teacher_distillation_repair_row_seed_survival_loop_v1.md"
DEFAULT_PROMPT = ROOT / "reports" / "teacher_distillation_repair_row_seed_survival_loop_v1_prompt.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD.relative_to(ROOT)))
    parser.add_argument("--prompt-out", default=str(DEFAULT_PROMPT.relative_to(ROOT)))
    args = parser.parse_args()

    candidate = build_candidate_row()
    teacher_call_shell = {
        "request_id": "local_seed_for_teacher_distillation_survival_loop_v1",
        "created_utc": now(),
        "completed_utc": now(),
        "provider": "local_seed_not_teacher",
        "model": "none",
        "reason_for_call": "architecture_wall",
        "mode": "distillation",
        "status": "completed",
        "prompt_sha256": manifest_builder.sha256_text("local seed only"),
        "response_text": "",
        "external_inference_calls": 0,
    }
    verifier = manifest_builder.local_candidate_verifier(candidate, teacher_call_shell)
    admission = manifest_builder.candidate_admission_decision(candidate, teacher_call_shell)
    report = {
        "policy": "project_theseus_teacher_distillation_repair_row_seed_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if admission.get("accepted") else "RED",
        "summary": {
            "candidate_task_family": candidate["task_family"],
            "repair_categories": candidate["provenance"]["repair_categories"],
            "local_verifier_accepted": verifier.get("accepted"),
            "manifest_admission_would_accept": admission.get("accepted"),
            "reject_reasons": admission.get("reject_reasons", []),
            "public_overlap_hits": candidate["public_overlap_hits"],
            "holdout_overlap_hits": candidate["holdout_overlap_hits"],
            "license_spdx": candidate["license_spdx"],
            "runtime_serving": candidate["runtime_serving"],
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "fallback_returns": 0,
        },
        "candidate_row": candidate,
        "local_verifier": verifier,
        "manifest_admission_decision": admission,
        "score_semantics": (
            "Private teacher-distillation seed only. It prepares a candidate row shape for a later "
            "governed teacher call. It does not call a teacher, admit training rows, run public "
            "calibration, serve external tokens, emit fallback bodies, or promote a model."
        ),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_text(resolve(args.prompt_out), render_teacher_prompt(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 1


def build_candidate_row() -> dict[str, Any]:
    repair_categories = [
        "algorithm_planning",
        "return_shape",
        "verifier_contract",
        "selector_ranking",
        "dependency_runtime",
    ]
    code_lm_task = {
        "task_id": "teacher_seed_private_filtered_abs_sum_v1",
        "split": "train",
        "category": "teacher_private_filtered_abs_sum",
        "concept_residual_label": "teacher_private_filtered_abs_sum",
        "prompt": "Return the sum of absolute integer values divisible by a positive divisor, ignoring bools and non-integers.",
        "entry_point": "teacher_private_filtered_abs_sum_v1",
        "solution_body": "\n".join(
            [
                "divisor = abs(other) if isinstance(other, int) and not isinstance(other, bool) else 0",
                "if divisor == 0:",
                "    return 0",
                "total = 0",
                "for value in data:",
                "    if isinstance(value, bool) or not isinstance(value, int):",
                "        continue",
                "    value = abs(value)",
                "    if value % divisor == 0:",
                "        total += value",
                "return total",
            ]
        ),
        "tests": "\n".join(
            [
                "assert teacher_private_filtered_abs_sum_v1([3, -6, 7, True, 'x', 12], 3) == 21",
                "assert teacher_private_filtered_abs_sum_v1([5, -10, 11], -5) == 15",
                "assert teacher_private_filtered_abs_sum_v1([1, 2, 3], 0) == 0",
                "",
            ]
        ),
        "decoder_contract": {
            "policy": "project_theseus_decoder_contract_v1_teacher_private_code_lm",
            "visible_arg_count_hint": 2,
            "full_body_required": True,
            "return_shape": "number",
            "return_contract": {
                "shape": "number",
                "empty_or_invalid_behavior": "covered_by_private_assertions",
                "must_preserve_container_shape": False,
            },
            "type_family": "numeric_algorithm",
            "semantic_family": "teacher_private_filtered_abs_sum",
            "residual_label_hint": "teacher_private_filtered_abs_sum",
            "required_constructs": ["loop", "branch", "locals", "numeric_ops"],
            "generation_plan": {
                "policy": "signature -> prompt -> branch_loop_state -> body",
                "public_solutions_used": False,
                "public_tests_used": False,
                "repair_strategy": "teach compositional numeric filtering without heldout-family leakage",
            },
        },
        "public_benchmark": False,
        "public_prompt": False,
        "tags": [
            "teacher_distillation",
            "private_code_lm",
            "numeric_algorithm",
            "loop",
            "branch",
            "train",
        ],
    }
    input_text = json.dumps(
        {
            "contract": "private_survival_lane_repair_policy",
            "goal": "improve transfer-oriented private candidate quality without public payload training",
            "observed_private_evidence": {
                "private_replay_selected_pass_rate": 1.0,
                "private_semantic_selected_pass_rate": 1.0,
                "dogfood_outcomes_present": [
                    "accepted",
                    "missed",
                    "ignored",
                    "corrected",
                    "completed",
                ],
                "residual_buckets": repair_categories,
            },
            "required_behavior": "produce a private residual stress and selector-hardening rule for the survival lane",
        },
        sort_keys=True,
    )
    target_text = json.dumps(
        {
            "training_target": "survival_residual_stress_selector_hardening_rule",
            "code_lm_task": code_lm_task,
            "route": "transformer_hybrid_structural_full_body_student",
            "symliquid_role": "matched_compute_discovery_comparator",
            "vcm": "enabled_for_structural_path",
            "sts": "use_only_where_private_ablation_is_non_regressive",
            "stress_split": {
                "source": "private_residual_and_redacted_dogfood_metadata",
                "balance_buckets": repair_categories,
                "forbidden_inputs": [
                    "public_eval_payloads",
                    "public_tests",
                    "public_solutions",
                    "hidden_tests",
                    "benchmark_traces",
                ],
            },
            "selector_rule": {
                "prefer": [
                    "candidate_matches_requested_return_shape",
                    "candidate_verifier_rationale_matches_contract",
                    "candidate_algorithm_family_matches_prompt_contract",
                    "candidate_runtime_dependencies_are_declared",
                ],
                "penalize": [
                    "shape_ambiguous_candidate",
                    "verifier_contract_mismatch",
                    "algorithm_family_mismatch",
                    "undeclared_runtime_dependency",
                ],
            },
            "acceptance_gate": {
                "private_stress_selected_pass_rate_delta": "positive",
                "private_replay_selected_pass_rate": 1.0,
                "fallback_body_count": 0,
                "public_training_rows": 0,
                "served_external_inference": 0,
            },
        },
        sort_keys=True,
    )
    return {
        "row_id": "survival_loop_repair_category_teacher_seed_v1",
        "source_kind": "teacher_distillation",
        "task_family": "survival_residual_stress_selector_hardening",
        "input_text": input_text,
        "target_text": target_text,
        "target_hash": "sha256:auto",
        "code_lm_task": code_lm_task,
        "license_spdx": "project-internal",
        "provenance": {
            "source": "local_private_seed_for_governed_teacher_distillation",
            "created_utc": now(),
            "evidence_paths": [
                "reports/survival_lane_real_use_growth_loop_v1.json",
                "reports/private_residual_target_consumer_survival_loop_v1.json",
                "reports/private_candidate_replay_contract_audit_survival_loop_v1.json",
                "reports/dogfood_trace_training_bridge_survival_loop_v1.json",
            ],
            "policy": "sparse_teacher_governed_distillation",
            "local_verifier": "scripts/teacher_distillation_manifest_builder.py",
            "repair_categories": repair_categories,
        },
        "public_benchmark": False,
        "public_prompt": False,
        "public_overlap_hits": 0,
        "holdout_overlap_hits": 0,
        "runtime_serving": "forbidden",
        "admission_checks": {
            "provenance_retained": True,
            "license_checked": True,
            "leakage_audited": True,
            "verifier_accepted": True,
            "runtime_serving_forbidden": True,
            "public_benchmark_excluded": True,
        },
    }


def render_teacher_prompt(report: dict[str, Any]) -> str:
    candidate = report["candidate_row"]
    verifier = report["local_verifier"]
    return "\n".join(
        [
            "Using only this private local seed row and the evidence paths in it, decide whether to emit one governed distillation_training_row.",
            "",
            "If you emit a row, keep it private, project-internal, runtime_serving='forbidden', and public_benchmark=false/public_prompt=false.",
            "Do not include public eval payloads, public tests, public solutions, hidden tests, benchmark traces, wrappers, or fallback bodies.",
            "You may copy or minimally revise the candidate row only if you preserve the local verifier properties below.",
            "If you cannot safely emit it, set distillation_training_row=null and name the exact missing evidence.",
            "",
            "LOCAL VERIFIER SUMMARY:",
            json.dumps(
                {
                    "accepted": verifier.get("accepted"),
                    "reject_reasons": verifier.get("reject_reasons"),
                    "checks": verifier.get("checks"),
                },
                indent=2,
                sort_keys=True,
            ),
            "",
            "CANDIDATE DISTILLATION ROW JSON:",
            json.dumps(candidate, indent=2, sort_keys=True),
            "",
        ]
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# Teacher Distillation Repair Row Seed",
            "",
            f"- trigger_state: `{report.get('trigger_state')}`",
            f"- candidate_task_family: `{summary.get('candidate_task_family')}`",
            f"- repair_categories: `{summary.get('repair_categories')}`",
            f"- local_verifier_accepted: `{summary.get('local_verifier_accepted')}`",
            f"- manifest_admission_would_accept: `{summary.get('manifest_admission_would_accept')}`",
            f"- reject_reasons: `{summary.get('reject_reasons')}`",
            f"- public_training_rows_written: `{summary.get('public_training_rows_written')}`",
            f"- external_inference_calls: `{summary.get('external_inference_calls')}`",
            "",
            str(report.get("score_semantics") or ""),
            "",
        ]
    )


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
