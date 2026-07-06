#!/usr/bin/env python3
"""Audit the survival-lane real-use growth loop.

This script is a gate and evidence packet for the current Theseus growth loop.
It reads existing private/governance artifacts, checks each goal requirement,
and writes a concise report. It does not run public calibration, call a
teacher, serve external tokens, or manufacture dogfood events.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "survival_lane_real_use_growth_loop_v1.json"
DEFAULT_MD = REPORTS / "survival_lane_real_use_growth_loop_v1.md"
REQUIRED_DOGFOOD_OUTCOMES = {"accepted", "missed", "ignored", "corrected", "completed"}
FORBIDDEN_DOGFOOD_KEYS = {
    "raw_user_text",
    "raw_assistant_text",
    "prompt",
    "completion",
    "secret_values",
    "private_file_contents",
    "public_benchmark_prompt_or_solution",
    "fallback_return",
    "fallback_return_used",
}


DEFAULT_INPUTS = {
    "dogfood_readiness": "reports/dogfood_trace_readiness_survival_loop_v1.json",
    "dogfood_bridge": "reports/dogfood_trace_training_bridge_survival_loop_v1.json",
    "teacher_gate": "reports/teacher_distillation_gate_survival_loop_v1.json",
    "teacher_manifest": "reports/teacher_distillation_manifest.json",
    "teacher_smoke": "reports/teacher_distillation_admission_smoke_survival_loop_v1.json",
    "private_replay": "reports/private_candidate_replay_contract_audit_survival_loop_v1.json",
    "semantic_ablation": "reports/private_full_body_semantic_quality_ablation_survival_loop_v1.json",
    "private_residual_consumer": "reports/private_residual_target_consumer_survival_loop_v1.json",
    "survival_decision": "reports/broad_capability_survival_lane_decision_survival_loop_v1.json",
    "promotion_gate": "reports/broad_capability_survival_promotion_gate_survival_loop_v1.json",
    "vcm_ablation": "reports/broad_capability_structural_vcm_ablation_v1.json",
    "vcm_policy": "configs/vcm_structural_survival_feature_policy_v1.json",
    "sts_policy": "configs/sts_broad_survival_policy_v1.json",
    "resource_governor": "reports/resource_governor_survival_loop_v1.json",
    "mlx_diagnosis": "reports/macos_mlx_environment_diagnosis_survival_loop_v1.json",
    "vcm_runtime": "reports/vcm_native_runtime_probe_survival_loop_v1.json",
    "maturity_audit": "reports/maturity_integrity_audit_survival_loop_v1.json",
    "candidate_promotion": "reports/candidate_promotion_gate.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=default)
    parser.add_argument("--dogfood-trace", default="runtime/dogfood/daily_use_events.jsonl")
    parser.add_argument(
        "--dogfood-training-rows",
        default="data/training_data/high_transfer/private_train/dogfood_daily_use_trace_training_rows.jsonl",
    )
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    inputs = {key: read_json(resolve(getattr(args, key))) for key in DEFAULT_INPUTS}
    dogfood_events = read_jsonl(resolve(args.dogfood_trace))
    dogfood_training_rows = read_jsonl(resolve(args.dogfood_training_rows))

    dogfood = dogfood_summary(inputs["dogfood_bridge"], dogfood_events, dogfood_training_rows)
    teacher = teacher_summary(inputs["teacher_gate"], inputs["teacher_manifest"], inputs["teacher_smoke"])
    private_learning = private_learning_summary(
        inputs["private_replay"],
        inputs["semantic_ablation"],
        inputs["private_residual_consumer"],
        inputs["survival_decision"],
        inputs["promotion_gate"],
    )
    routing = routing_summary(inputs["vcm_ablation"], inputs["vcm_policy"], inputs["sts_policy"])
    runtime = runtime_summary(inputs["resource_governor"], inputs["mlx_diagnosis"], inputs["vcm_runtime"])
    public_guard = public_guard_summary(inputs["maturity_audit"], inputs["candidate_promotion"])
    safety = safety_summary(inputs, dogfood, teacher, private_learning, runtime)

    checks = build_checks(dogfood, teacher, private_learning, routing, runtime, public_guard, safety)
    hard_failed = [row for row in checks if row["severity"] == "hard" and not row["passed"]]
    incomplete = [row for row in checks if row["severity"] == "incomplete" and not row["passed"]]
    trigger_state = "RED" if hard_failed else "YELLOW" if incomplete else "GREEN"

    return {
        "policy": "project_theseus_survival_lane_real_use_growth_loop_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "hard_failure_count": len(hard_failed),
            "incomplete_requirement_count": len(incomplete),
            "hard_failures": [row["name"] for row in hard_failed],
            "incomplete_requirements": [row["name"] for row in incomplete],
            "dogfood": dogfood,
            "teacher": teacher,
            "private_learning": private_learning,
            "routing": routing,
            "runtime": runtime,
            "public_guard": public_guard,
            "safety": safety,
            "recommendation": recommendation(trigger_state, dogfood, teacher, private_learning, runtime),
        },
        "inputs": {
            **{key: rel(resolve(getattr(args, key))) for key in DEFAULT_INPUTS},
            "dogfood_trace": rel(resolve(args.dogfood_trace)),
            "dogfood_training_rows": rel(resolve(args.dogfood_training_rows)),
        },
        "checks": checks,
        "score_semantics": (
            "Survival-lane real-use growth-loop audit only. It consumes private reports and redacted local dogfood "
            "metadata. It does not run public calibration, train on public benchmark payloads, call a teacher, "
            "serve external tokens, emit fallback returns, or promote a public model."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def dogfood_summary(
    bridge: dict[str, Any],
    events: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    event_outcomes = Counter(str(row.get("outcome") or "") for row in events if isinstance(row, dict))
    training_outcomes = Counter(str(row.get("outcome_label") or "") for row in training_rows if isinstance(row, dict))
    forbidden_event_keys = sorted(FORBIDDEN_DOGFOOD_KEYS.intersection(flat_keys(events)))
    forbidden_training_keys = sorted(FORBIDDEN_DOGFOOD_KEYS.intersection(flat_keys(training_rows)))
    present_outcomes = set(event_outcomes) | set(training_outcomes)
    non_accepted = sorted(outcome for outcome in present_outcomes if outcome and outcome != "accepted")
    missing_required = sorted(REQUIRED_DOGFOOD_OUTCOMES - present_outcomes)
    bridge_summary = dict_or_empty(bridge.get("summary"))
    return {
        "event_count": len(events),
        "training_row_count": len(training_rows),
        "new_training_event_count": bridge_summary.get("new_training_event_count"),
        "training_rows_written_latest": bridge.get("training_rows_written"),
        "event_outcome_counts": dict(event_outcomes.most_common()),
        "training_outcome_counts": dict(training_outcomes.most_common()),
        "required_outcomes": sorted(REQUIRED_DOGFOOD_OUTCOMES),
        "missing_required_outcomes": missing_required,
        "non_accepted_outcomes_present": non_accepted,
        "accepted_only": present_outcomes == {"accepted"},
        "raw_text_capture_enabled": bridge_summary.get("raw_text_capture_enabled"),
        "trained_on_raw_text": bridge_summary.get("trained_on_raw_text"),
        "forbidden_event_keys": forbidden_event_keys,
        "forbidden_training_keys": forbidden_training_keys,
        "public_training_rows": bridge_summary.get("public_training_rows", 0),
        "external_inference_calls": bridge_summary.get("external_inference_calls", 0),
        "fallback_returns": bridge_summary.get("fallback_returns", 0),
        "bridge_state": bridge.get("trigger_state"),
        "bridge_write_blocker": bridge.get("write_blocker"),
    }


def teacher_summary(gate: dict[str, Any], manifest: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    gate_summary = dict_or_empty(gate.get("summary"))
    manifest_summary = dict_or_empty(manifest.get("summary"))
    manifest_rows = manifest.get("rows") if isinstance(manifest.get("rows"), list) else []
    task_families = sorted({str(row.get("task_family") or "") for row in manifest_rows if isinstance(row, dict)})
    return {
        "gate_state": gate.get("trigger_state"),
        "distillation_allowed": gate.get("distillation_allowed"),
        "manifest_row_count": manifest_summary.get("row_count", len(manifest_rows)),
        "manifest_verifier_pass_rate": manifest_summary.get("verifier_pass_rate"),
        "manifest_public_overlap_hits": manifest_summary.get("public_overlap_hits", 0),
        "manifest_holdout_overlap_hits": manifest_summary.get("holdout_overlap_hits", 0),
        "manifest_admission_safety_checks_clean": manifest_summary.get("admission_safety_checks_clean"),
        "runtime_external_tokens_forbidden": gate_summary.get("runtime_external_tokens_forbidden"),
        "teacher_accepted_row_share": gate_summary.get("teacher_accepted_row_share"),
        "teacher_share_within_cap": gate_summary.get("teacher_share_within_cap"),
        "task_families": task_families,
        "external_inference_calls": gate_summary.get("external_inference_calls", 0),
        "smoke_state": smoke.get("trigger_state"),
        "smoke_real_training_rows_written": get_path(smoke, ["summary", "real_training_rows_written"], 0),
    }


def private_learning_summary(
    replay: dict[str, Any],
    semantic: dict[str, Any],
    residual: dict[str, Any],
    decision: dict[str, Any],
    promotion: dict[str, Any],
) -> dict[str, Any]:
    replay_summary = dict_or_empty(replay.get("summary"))
    semantic_summary = dict_or_empty(semantic.get("summary"))
    residual_summary = dict_or_empty(residual.get("summary"))
    promotion_summary = dict_or_empty(promotion.get("summary"))
    return {
        "replay_state": replay.get("trigger_state"),
        "replay_task_count": replay_summary.get("task_count"),
        "replay_selected_pass_rate": replay_summary.get("selected_intended_behavior_pass_rate"),
        "replay_unexplained_no_candidate_count": replay_summary.get("unexplained_no_candidate_count"),
        "semantic_state": semantic.get("trigger_state"),
        "semantic_best_selected_pass_rate": semantic_summary.get("best_private_public_shaped_selected_pass_rate"),
        "semantic_best_pass_if_any_rate": semantic_summary.get("best_private_public_shaped_pass_if_any_rate"),
        "semantic_sts_delta": semantic_summary.get("sts_delta"),
        "residual_state": residual.get("trigger_state"),
        "residual_target_rows": residual_summary.get("target_rows"),
        "residual_unresolved_target_count": residual_summary.get("unresolved_target_count"),
        "residual_categories": residual_summary.get("target_categories", {}),
        "survival_decision_state": decision.get("trigger_state"),
        "survival_decision": decision.get("decision"),
        "promotion_state": promotion.get("trigger_state"),
        "promotion_scope": promotion_summary.get("model_promotion_scope"),
        "promotion_public_calibration_allowed": promotion_summary.get("public_calibration_allowed"),
        "promotion_serving_allowed": promotion_summary.get("serving_allowed"),
        "survival_arm": promotion_summary.get("arm_id"),
        "baseline_pass_rate": promotion_summary.get("baseline_pass_rate"),
        "structural_only_pass_rate": promotion_summary.get("structural_only_pass_rate"),
        "augmented_pass_rate": promotion_summary.get("augmented_pass_rate"),
        "fallback_return_count": max(
            int_number(replay_summary.get("fallback_return_candidate_count")),
            int_number(semantic_summary.get("fallback_return_candidate_count")),
        ),
        "public_training_rows_written": int_number(semantic_summary.get("public_training_rows_written")),
        "external_inference_calls": int_number(semantic_summary.get("external_inference_calls")),
    }


def routing_summary(vcm_ablation: dict[str, Any], vcm_policy: dict[str, Any], sts_policy: dict[str, Any]) -> dict[str, Any]:
    vcm_summary = dict_or_empty(vcm_ablation.get("summary"))
    return {
        "vcm_ablation_state": vcm_ablation.get("trigger_state"),
        "vcm_policy_action": vcm_policy.get("action"),
        "promoted_transformer_vcm_mode": vcm_policy.get("promoted_transformer_vcm_mode"),
        "vcm_transformer_augmented_delta": get_path(vcm_policy, ["transformer_control", "summary", "augmented_delta"]),
        "vcm_transformer_structural_delta": get_path(vcm_policy, ["transformer_control", "summary", "structural_only_delta"]),
        "vcm_rows_with_context": vcm_summary.get("vcm_rows_with_context"),
        "sts_policy_action": sts_policy.get("action"),
        "sts_effective_fields": sts_policy.get("effective_sts_on_fields", []),
        "sts_promotion_rule": sts_policy.get("promotion_rule"),
        "legacy_body_template_sts_disabled": sts_policy.get("action") == "disable_sts_for_broad_body_template_selector",
    }


def runtime_summary(resource: dict[str, Any], mlx: dict[str, Any], vcm_runtime: dict[str, Any]) -> dict[str, Any]:
    resource_summary = dict_or_empty(resource.get("summary"))
    mlx_summary = dict_or_empty(mlx.get("summary"))
    vcm_summary = dict_or_empty(vcm_runtime.get("summary"))
    return {
        "resource_state": resource.get("trigger_state"),
        "can_run_requested_profile": resource_summary.get("can_run_requested_profile"),
        "execution_owner": resource_summary.get("execution_owner"),
        "efficiency_score": resource_summary.get("efficiency_score"),
        "disk_free_gib": resource_summary.get("disk_free_gib"),
        "mlx_state": mlx.get("trigger_state"),
        "mlx_usable": bool(
            resource_summary.get("mlx_usable") is True
            or mlx_summary.get("active_python_status") == "usable"
            or mlx_summary.get("mlx_usable") is True
        ),
        "metal_usable": bool(resource_summary.get("metal_usable") is True or mlx_summary.get("metal_usable") is True),
        "recommended_python": mlx_summary.get("recommended_python") or vcm_summary.get("recommended_python"),
        "vcm_runtime_state": vcm_runtime.get("trigger_state"),
        "vcm_native_runtime_claimable": vcm_summary.get("native_runtime_claimable"),
        "vcm_native_runtime_claim_scope": vcm_summary.get("native_runtime_claim_scope"),
        "recommended_backend": vcm_summary.get("recommended_backend"),
        "recommended_backend_native_runtime_claimable": vcm_summary.get("recommended_backend_native_runtime_claimable"),
        "scheduler_native_kv_route_allowed_for_recommended_backend": vcm_summary.get(
            "scheduler_native_kv_route_allowed_for_recommended_backend"
        ),
        "accelerator_kv_parity_claimed": vcm_summary.get("accelerator_kv_parity_claimed"),
        "fallback_return_count": vcm_summary.get("fallback_return_count", 0),
        "public_training_rows_written": vcm_summary.get("public_training_rows_written", 0),
        "external_inference_calls": vcm_summary.get("external_inference_calls", 0),
    }


def public_guard_summary(maturity: dict[str, Any], candidate_promotion: dict[str, Any]) -> dict[str, Any]:
    maturity_summary = dict_or_empty(maturity.get("summary"))
    return {
        "maturity_state": maturity.get("trigger_state"),
        "latest_public_pass_rate": maturity_summary.get("latest_public_pass_rate")
        or maturity_summary.get("broad_public_pass_rate"),
        "public_calibration_allowed": maturity_summary.get("public_calibration_allowed", False),
        "model_growth_allowed": maturity_summary.get("model_growth_allowed"),
        "maturity_blockers": maturity_summary.get("maturity_blockers", []),
        "candidate_promote": candidate_promotion.get("promote"),
        "candidate_passed": candidate_promotion.get("passed"),
        "candidate_total": candidate_promotion.get("total"),
        "candidate_failed_gates": [
            row.get("gate")
            for row in as_list(candidate_promotion.get("checks"))
            if isinstance(row, dict) and row.get("passed") is False
        ],
    }


def safety_summary(
    inputs: dict[str, dict[str, Any]],
    dogfood: dict[str, Any],
    teacher: dict[str, Any],
    private_learning: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    public_training_rows = max(
        int_number(dogfood.get("public_training_rows")),
        int_number(private_learning.get("public_training_rows_written")),
        int_number(runtime.get("public_training_rows_written")),
    )
    fallback_returns = max(
        int_number(dogfood.get("fallback_returns")),
        int_number(private_learning.get("fallback_return_count")),
        int_number(runtime.get("fallback_return_count")),
    )
    external_calls = max(
        int_number(dogfood.get("external_inference_calls")),
        int_number(private_learning.get("external_inference_calls")),
        int_number(runtime.get("external_inference_calls")),
    )
    return {
        "public_calibration_run_by_this_script": False,
        "public_training_rows_written": public_training_rows,
        "fallback_return_count": fallback_returns,
        "external_inference_calls": external_calls,
        "runtime_serving_external_tokens": "forbidden" if teacher.get("runtime_external_tokens_forbidden") else "unknown",
        "teacher_smoke_real_training_rows_written": teacher.get("smoke_real_training_rows_written"),
        "report_states": {key: value.get("trigger_state") for key, value in inputs.items()},
    }


def build_checks(
    dogfood: dict[str, Any],
    teacher: dict[str, Any],
    private_learning: dict[str, Any],
    routing: dict[str, Any],
    runtime: dict[str, Any],
    public_guard: dict[str, Any],
    safety: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        check(
            "dogfood_real_pressure_not_accepted_only",
            not dogfood["accepted_only"] and bool(dogfood["non_accepted_outcomes_present"]),
            "incomplete",
            {
                "outcomes": dogfood["event_outcome_counts"],
                "training_outcomes": dogfood["training_outcome_counts"],
                "missing_required_outcomes": dogfood["missing_required_outcomes"],
            },
        ),
        check(
            "dogfood_required_outcomes_observed",
            not dogfood["missing_required_outcomes"],
            "incomplete",
            {"missing_required_outcomes": dogfood["missing_required_outcomes"]},
        ),
        check(
            "dogfood_redacted_private_training_only",
            not dogfood["forbidden_event_keys"]
            and not dogfood["forbidden_training_keys"]
            and dogfood.get("raw_text_capture_enabled") is False
            and dogfood.get("trained_on_raw_text") is False,
            "hard",
            {
                "forbidden_event_keys": dogfood["forbidden_event_keys"],
                "forbidden_training_keys": dogfood["forbidden_training_keys"],
                "raw_text_capture_enabled": dogfood.get("raw_text_capture_enabled"),
                "trained_on_raw_text": dogfood.get("trained_on_raw_text"),
            },
        ),
        check(
            "governed_teacher_distillation_clean",
            teacher.get("gate_state") == "GREEN"
            and teacher.get("distillation_allowed") is True
            and int_number(teacher.get("manifest_row_count")) > 0
            and number(teacher.get("manifest_verifier_pass_rate")) >= 0.95
            and int_number(teacher.get("manifest_public_overlap_hits")) == 0
            and int_number(teacher.get("manifest_holdout_overlap_hits")) == 0
            and teacher.get("manifest_admission_safety_checks_clean") is True
            and teacher.get("runtime_external_tokens_forbidden") is True,
            "hard",
            teacher,
        ),
        check(
            "teacher_repair_category_coverage_started",
            int_number(teacher.get("manifest_row_count")) >= 1,
            "incomplete",
            {"manifest_row_count": teacher.get("manifest_row_count"), "task_families": teacher.get("task_families")},
        ),
        check(
            "private_replay_and_semantic_green",
            private_learning.get("replay_state") == "GREEN"
            and private_learning.get("semantic_state") == "GREEN"
            and number(private_learning.get("replay_selected_pass_rate")) >= 1.0
            and int_number(private_learning.get("replay_unexplained_no_candidate_count")) == 0
            and number(private_learning.get("semantic_best_selected_pass_rate")) > 0.0,
            "hard",
            private_learning,
        ),
        check(
            "private_residual_categories_closed_or_queued",
            private_learning.get("residual_state") == "GREEN"
            and int_number(private_learning.get("residual_target_rows")) > 0
            and int_number(private_learning.get("residual_unresolved_target_count")) == 0,
            "hard",
            {
                "target_rows": private_learning.get("residual_target_rows"),
                "unresolved": private_learning.get("residual_unresolved_target_count"),
                "categories": private_learning.get("residual_categories"),
            },
        ),
        check(
            "transformer_survival_lane_selected_symliquid_comparator_only",
            private_learning.get("survival_decision") == "promote_new_survival_candidate"
            and private_learning.get("survival_arm") == "transformer_control"
            and private_learning.get("promotion_public_calibration_allowed") is False
            and private_learning.get("promotion_serving_allowed") is False,
            "hard",
            private_learning,
        ),
        check(
            "vcm_and_sts_path_specific_routing",
            routing.get("vcm_policy_action") == "enable_vcm_for_promoted_transformer_structural_path"
            and routing.get("promoted_transformer_vcm_mode") == "on"
            and number(routing.get("vcm_transformer_augmented_delta")) >= 0.0
            and routing.get("legacy_body_template_sts_disabled") is True,
            "hard",
            routing,
        ),
        check(
            "mac_runtime_cost_and_acceleration_reported",
            runtime.get("resource_state") == "GREEN"
            and runtime.get("can_run_requested_profile") is True
            and runtime.get("mlx_usable") is True
            and runtime.get("metal_usable") is True
            and runtime.get("accelerator_kv_parity_claimed") is False,
            "hard",
            runtime,
        ),
        check(
            "public_calibration_not_run_and_not_allowed",
            safety.get("public_calibration_run_by_this_script") is False
            and public_guard.get("public_calibration_allowed") is False,
            "hard",
            {"public_guard": public_guard, "safety": safety},
        ),
        check("public_training_rows_zero", int_number(safety.get("public_training_rows_written")) == 0, "hard", safety),
        check("fallback_returns_zero", int_number(safety.get("fallback_return_count")) == 0, "hard", safety),
        check("served_external_inference_zero", int_number(safety.get("external_inference_calls")) == 0, "hard", safety),
        check(
            "candidate_promotion_still_requires_public_transfer",
            public_guard.get("candidate_promote") is False,
            "incomplete",
            public_guard,
        ),
    ]


def recommendation(
    trigger_state: str,
    dogfood: dict[str, Any],
    teacher: dict[str, Any],
    private_learning: dict[str, Any],
    runtime: dict[str, Any],
) -> str:
    if trigger_state == "RED":
        return "Fix hard safety or evidence failures before any longer growth run."
    missing = dogfood.get("missing_required_outcomes") or []
    if missing:
        return (
            "Continue private repair, but first collect real redacted dogfood events for: "
            + ", ".join(missing)
            + ". The survival lane should not be called daily-use trained while accepted-only pressure dominates."
        )
    if int_number(teacher.get("manifest_row_count")) < 5:
        return "Continue private repair with more governed teacher-distillation rows across algorithm, return-shape, verifier, and selector/ranking categories."
    if runtime.get("scheduler_native_kv_route_allowed_for_recommended_backend") is False:
        return "Continue private repair and move the VCM/native cache hot path toward MLX before claiming Mac parity."
    if private_learning.get("survival_decision") == "promote_new_survival_candidate":
        return "Private survival loop is ready to propose one future governed public calibration review, not to rerun public surfaces automatically."
    return "Continue private repair."


def check(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Survival-Lane Real-Use Growth Loop v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- hard_failure_count: `{summary.get('hard_failure_count')}`",
        f"- incomplete_requirement_count: `{summary.get('incomplete_requirement_count')}`",
        f"- recommendation: {summary.get('recommendation')}",
        "",
        "## Dogfood",
        "",
        f"- event outcomes: `{summary.get('dogfood', {}).get('event_outcome_counts')}`",
        f"- training outcomes: `{summary.get('dogfood', {}).get('training_outcome_counts')}`",
        f"- missing required outcomes: `{summary.get('dogfood', {}).get('missing_required_outcomes')}`",
        "",
        "## Survival Lane",
        "",
        f"- decision: `{summary.get('private_learning', {}).get('survival_decision')}`",
        f"- arm: `{summary.get('private_learning', {}).get('survival_arm')}`",
        f"- replay selected pass: `{summary.get('private_learning', {}).get('replay_selected_pass_rate')}`",
        f"- semantic selected pass: `{summary.get('private_learning', {}).get('semantic_best_selected_pass_rate')}`",
        "",
        "## Teacher And Runtime",
        "",
        f"- teacher rows: `{summary.get('teacher', {}).get('manifest_row_count')}`",
        f"- teacher share: `{summary.get('teacher', {}).get('teacher_accepted_row_share')}`",
        f"- execution owner: `{summary.get('runtime', {}).get('execution_owner')}`",
        f"- MLX usable: `{summary.get('runtime', {}).get('mlx_usable')}`",
        f"- accelerator KV parity claimed: `{summary.get('runtime', {}).get('accelerator_kv_parity_claimed')}`",
        "",
        "## Checks",
        "",
    ]
    for row in report.get("checks", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    lines.extend(["", "## Semantics", "", str(report.get("score_semantics") or ""), ""])
    return "\n".join(lines)


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def flat_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(flat_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(flat_keys(item))
    return keys


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_number(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
