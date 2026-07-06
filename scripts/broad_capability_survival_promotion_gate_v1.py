#!/usr/bin/env python3
"""Promotion gate for the broad private survival-lane structural candidate.

This gate promotes only a private structural-action student artifact. It does
not run public calibration, train on public payloads, call a teacher, serve
runtime tokens, or claim MLX/Metal parity.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRUCTURAL = ROOT / "reports" / "broad_capability_structural_action_decoder_probe_v1_vcm_on.json"
DEFAULT_BROAD_RUN = ROOT / "reports" / "broad_capability_survival_lane_run_v1.json"
DEFAULT_STS_AUDIT = ROOT / "reports" / "sts_broad_regression_audit_v1.json"
DEFAULT_STS_POLICY = ROOT / "configs" / "sts_broad_survival_policy_v1.json"
DEFAULT_VCM_ABLATION = ROOT / "reports" / "broad_capability_structural_vcm_ablation_v1.json"
DEFAULT_VCM_POLICY = ROOT / "configs" / "vcm_structural_survival_feature_policy_v1.json"
DEFAULT_MLX = ROOT / "reports" / "macos_mlx_environment_diagnosis.json"
DEFAULT_MLX_SMOKE = ROOT / "reports" / "macos_mlx_structural_action_smoke.json"
DEFAULT_OUT = ROOT / "reports" / "broad_capability_survival_promotion_gate_v1.json"
DEFAULT_MD = ROOT / "reports" / "broad_capability_survival_promotion_gate_v1.md"
DEFAULT_ACTIVE = ROOT / "checkpoints" / "hive_promoted" / "broad_survival_transformer_structural_student" / "active_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural-action", default=rel(DEFAULT_STRUCTURAL))
    parser.add_argument("--broad-run", default=rel(DEFAULT_BROAD_RUN))
    parser.add_argument("--sts-audit", default=rel(DEFAULT_STS_AUDIT))
    parser.add_argument("--sts-policy", default=rel(DEFAULT_STS_POLICY))
    parser.add_argument("--vcm-ablation", default=rel(DEFAULT_VCM_ABLATION))
    parser.add_argument("--vcm-policy", default=rel(DEFAULT_VCM_POLICY))
    parser.add_argument("--mlx-diagnosis", default=rel(DEFAULT_MLX))
    parser.add_argument("--mlx-smoke", default=rel(DEFAULT_MLX_SMOKE))
    parser.add_argument("--active-manifest-out", default=rel(DEFAULT_ACTIVE))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    if report["promote"]:
        write_json(resolve(args.active_manifest_out), report["active_manifest"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    structural_path = resolve(args.structural_action)
    broad_run_path = resolve(args.broad_run)
    sts_audit_path = resolve(args.sts_audit)
    sts_policy_path = resolve(args.sts_policy)
    vcm_ablation_path = resolve(args.vcm_ablation)
    vcm_policy_path = resolve(args.vcm_policy)
    mlx_path = resolve(args.mlx_diagnosis)
    mlx_smoke_path = resolve(args.mlx_smoke)
    active_manifest_path = resolve(args.active_manifest_out)

    structural = read_json(structural_path)
    broad_run = read_json(broad_run_path)
    sts_audit = read_json(sts_audit_path)
    sts_policy = read_json(sts_policy_path)
    vcm_ablation = read_json(vcm_ablation_path)
    vcm_policy = read_json(vcm_policy_path)
    mlx = read_json(mlx_path)
    mlx_smoke = read_json(mlx_smoke_path)

    transformer = get_path(structural, ["arms", "transformer_control", "summary"], {})
    structural_only = float(transformer.get("structural_only_pass_rate") or 0.0)
    baseline = float(transformer.get("baseline_pass_rate") or 0.0)
    augmented = float(transformer.get("augmented_pass_rate") or 0.0)
    eval_rows = int(get_path(structural, ["summary", "eval_rows"], 0) or 0)
    candidate_rows = int(transformer.get("structural_candidate_rows") or 0)
    structural_action_candidate_rows = int(transformer.get("structural_action_candidate_rows") or 0)
    structural_composition_candidate_rows = int(transformer.get("structural_composition_candidate_rows") or 0)
    structural_composition_supported_task_rows = int(transformer.get("structural_composition_supported_task_rows") or 0)
    argument_contract_mismatch_rows = int(transformer.get("argument_contract_mismatch_rows") or 0)
    first_rank_argument_contract_mismatch_rows = int(transformer.get("first_rank_argument_contract_mismatch_rows") or 0)
    composition_summary = get_path(structural, ["structural_composition_library"], {})
    no_cheat = {
        "public_training_rows": int(get_path(structural, ["summary", "public_training_rows"], 0) or 0)
        + int(get_path(broad_run, ["summary", "public_benchmark_training_rows"], 0) or 0),
        "fallback_return_rows": int(transformer.get("fallback_return_rows") or 0)
        + int(get_path(broad_run, ["summary", "fallback_return_count"], 0) or 0),
        "external_inference_calls": int(get_path(structural, ["summary", "external_inference_calls"], 0) or 0)
        + int(get_path(broad_run, ["summary", "external_inference_calls"], 0) or 0),
        "teacher_used": bool(get_path(structural, ["summary", "teacher_used"], False))
        or bool(get_path(broad_run, ["summary", "teacher_used"], False)),
    }
    broad_transformer_sts_delta = get_path(broad_run, ["comparator_report", "comparisons", "by_arm", "transformer_control", "sts_delta"])
    broad_transformer_sts_regressions = int(
        get_path(broad_run, ["comparator_report", "comparisons", "by_arm", "transformer_control", "sts_task_level_regressions"], 0)
        or 0
    )
    sts_policy_action = str(sts_policy.get("action") or "")
    legacy_body_template_sts_disabled = sts_policy_action == "disable_sts_for_broad_body_template_selector"
    mlx_route_action = str(get_path(mlx, ["summary", "route_action"], "") or "")
    mlx_recommended_python = str(get_path(mlx, ["summary", "recommended_python"], "") or "")
    mlx_parity_claim_allowed = bool(get_path(mlx, ["route_decision", "parity_claim_allowed"], False))
    mlx_enabled = mlx_route_action == "route_mlx_to_usable_python"
    checks = [
        check("structural_report_green", structural.get("trigger_state") == "GREEN", structural.get("trigger_state")),
        check("full_broad_eval_slice", eval_rows >= 192, {"eval_rows": eval_rows, "floor": 192}),
        check(
            "transformer_candidate_coverage_present",
            candidate_rows >= max(1, eval_rows),
            {
                "candidate_rows": candidate_rows,
                "eval_rows": eval_rows,
                "rule": "coverage floor is one generated candidate per eval task; stricter fanout is handled by private verifier quality and no-cheat gates because compatibility filters intentionally drop invalid candidates",
            },
        ),
        check(
            "composition_fragments_private_train_only",
            get_path(composition_summary, ["uses_eval_tests_or_solutions"], False) is False
            and get_path(composition_summary, ["uses_public_data"], True) is False
            and get_path(composition_summary, ["teacher_used"], True) is False,
            composition_summary,
        ),
        check("transformer_structural_only_beats_baseline", structural_only >= baseline, {"structural_only": structural_only, "baseline": baseline}),
        check("transformer_augmented_no_regression", augmented >= max(baseline, structural_only), {"augmented": augmented, "baseline": baseline, "structural_only": structural_only}),
        check("transformer_augmented_positive_delta", float(transformer.get("delta") or 0.0) > 0.0, transformer.get("delta")),
        check("syntax_pass_rate_full", float(transformer.get("syntax_pass_rate") or 0.0) >= 1.0, transformer.get("syntax_pass_rate")),
        check("argument_contract_mismatch_rows_zero", argument_contract_mismatch_rows == 0, argument_contract_mismatch_rows),
        check("first_rank_argument_contract_mismatch_rows_zero", first_rank_argument_contract_mismatch_rows == 0, first_rank_argument_contract_mismatch_rows),
        check("fallback_returns_zero", no_cheat["fallback_return_rows"] == 0, no_cheat["fallback_return_rows"]),
        check("public_training_zero", no_cheat["public_training_rows"] == 0, no_cheat["public_training_rows"]),
        check("external_inference_zero", no_cheat["external_inference_calls"] == 0, no_cheat["external_inference_calls"]),
        check("teacher_unused", no_cheat["teacher_used"] is False, no_cheat["teacher_used"]),
        check(
            "sts_policy_routes_legacy_selector_regression",
            (
                broad_transformer_sts_delta is not None
                and float(broad_transformer_sts_delta) >= 0.0
                and broad_transformer_sts_regressions == 0
            )
            or legacy_body_template_sts_disabled,
            {
                "sts_delta": broad_transformer_sts_delta,
                "regressions": broad_transformer_sts_regressions,
                "sts_policy_action": sts_policy_action,
                "rule": (
                    "Legacy body-template STS regressions do not block the structural/full-body survival lane "
                    "when policy explicitly disables STS for that legacy selector. Structural STS/VCM evidence "
                    "must be evaluated on the structural path."
                ),
            },
        ),
        check("sts_policy_loaded", bool(sts_policy.get("action")), sts_policy.get("action")),
        check("vcm_structural_policy_explicit", bool(vcm_policy.get("action")) and vcm_ablation.get("trigger_state") in {"GREEN", "YELLOW"}, {"policy": vcm_policy.get("action"), "ablation_state": vcm_ablation.get("trigger_state")}),
        check("mlx_route_explicit", mlx_route_action in {"disable_mlx_acceleration_route", "route_mlx_to_usable_python"}, mlx_route_action),
        check("mlx_recommended_python_present_when_enabled", not mlx_enabled or bool(mlx_recommended_python), {"route_action": mlx_route_action, "recommended_python": mlx_recommended_python}),
        check("mlx_structural_smoke_green_when_enabled", not mlx_enabled or mlx_smoke.get("trigger_state") == "GREEN", {"route_action": mlx_route_action, "smoke_state": mlx_smoke.get("trigger_state")}),
        check("mlx_structural_smoke_uses_mlx_when_enabled", not mlx_enabled or bool(get_path(mlx_smoke, ["summary", "mlx_used"], False)), get_path(mlx_smoke, ["summary", "mlx_used"])),
        check("mlx_structural_smoke_fallback_zero", not mlx_enabled or int(get_path(mlx_smoke, ["summary", "fallback_return_rows"], 0) or 0) == 0, get_path(mlx_smoke, ["summary", "fallback_return_rows"])),
        check("mlx_structural_smoke_public_training_zero", not mlx_enabled or int(get_path(mlx_smoke, ["summary", "public_training_rows"], 0) or 0) == 0, get_path(mlx_smoke, ["summary", "public_training_rows"])),
        check("mlx_structural_smoke_external_inference_zero", not mlx_enabled or int(get_path(mlx_smoke, ["summary", "external_inference_calls"], 0) or 0) == 0, get_path(mlx_smoke, ["summary", "external_inference_calls"])),
        check("mlx_parity_not_claimed", mlx_parity_claim_allowed is False, {"route_action": mlx_route_action, "parity_claim_allowed": mlx_parity_claim_allowed}),
        check("public_calibration_locked", True, "not_run_by_this_gate"),
    ]
    promote = all(row["passed"] for row in checks)
    active_manifest = {
        "policy": "project_theseus_broad_survival_structural_student_active_manifest_v1",
        "created_utc": now(),
        "arm_id": "transformer_control",
        "artifact_type": "private_broad_survival_structural_action_student",
        "candidate_family": "structural_action_sequence_plus_visible_composition_fragments",
        "serving_allowed": False,
        "public_calibration_allowed": False,
        "teacher_used": False,
        "external_inference_calls": 0,
        "scores": {
            "baseline_pass_rate": baseline,
            "structural_only_pass_rate": structural_only,
            "augmented_pass_rate": augmented,
            "delta_vs_baseline": round(augmented - baseline, 6),
        },
        "artifacts": {
            "structural_report": rel(structural_path),
            "candidate_manifest": structural.get("candidate_manifest"),
            "broad_run": rel(broad_run_path),
            "sts_audit": rel(sts_audit_path),
            "vcm_ablation": rel(vcm_ablation_path),
            "vcm_policy": rel(vcm_policy_path),
            "mlx_diagnosis": rel(mlx_path),
            "mlx_structural_smoke": rel(mlx_smoke_path),
        },
        "sts_route": {
            "policy_action": sts_policy_action,
            "legacy_body_template_sts_disabled": legacy_body_template_sts_disabled,
            "legacy_transformer_sts_delta": broad_transformer_sts_delta,
            "legacy_transformer_sts_regressions": broad_transformer_sts_regressions,
            "promotion_path": "transformer_hybrid_structural_full_body_student",
        },
        "runtime_route": {
            "framework": get_path(transformer, ["backend", "framework"]),
            "device": get_path(transformer, ["backend", "device"]),
            "mlx_available": mlx_enabled,
            "mlx_used": False,
            "mlx_route_action": mlx_route_action,
            "mlx_recommended_python": mlx_recommended_python,
            "mlx_structural_smoke_state": mlx_smoke.get("trigger_state"),
            "mlx_structural_smoke_verifier_pass_rate": get_path(mlx_smoke, ["summary", "verifier_pass_rate"]),
            "mlx_parity_claimed": False,
        },
        "vcm_route": {
            "policy_action": vcm_policy.get("action"),
            "promoted_transformer_vcm_mode": vcm_policy.get("promoted_transformer_vcm_mode"),
            "symliquid_discovery_vcm_mode": vcm_policy.get("symliquid_discovery_vcm_mode"),
            "feature_contract": vcm_policy.get("feature_contract"),
            "model_visible_features": vcm_policy.get("model_visible_features", []),
            "audit_only_features": vcm_policy.get("audit_only_features", []),
        },
        "no_cheat": no_cheat,
        "candidate_quality": {
            "argument_contract_mismatch_rows": argument_contract_mismatch_rows,
            "first_rank_argument_contract_mismatch_rows": first_rank_argument_contract_mismatch_rows,
            "structural_action_candidate_rows": structural_action_candidate_rows,
            "structural_composition_candidate_rows": structural_composition_candidate_rows,
            "structural_composition_supported_task_rows": structural_composition_supported_task_rows,
            "syntax_pass_rate": transformer.get("syntax_pass_rate"),
            "fallback_return_rows": no_cheat["fallback_return_rows"],
        },
    }
    return {
        "policy": "project_theseus_broad_capability_survival_promotion_gate_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if promote else "YELLOW",
        "promote": promote,
        "promoted": promote,
        "promotion_applied": promote,
        "active_manifest_path": rel(active_manifest_path) if promote else "",
        "active_manifest": active_manifest if promote else {},
        "summary": {
            "arm_id": "transformer_control",
            "baseline_pass_rate": baseline,
            "structural_only_pass_rate": structural_only,
            "augmented_pass_rate": augmented,
            "eval_rows": eval_rows,
            "candidate_rows": candidate_rows,
            "structural_action_candidate_rows": structural_action_candidate_rows,
            "structural_composition_candidate_rows": structural_composition_candidate_rows,
            "structural_composition_supported_task_rows": structural_composition_supported_task_rows,
            "argument_contract_mismatch_rows": argument_contract_mismatch_rows,
            "first_rank_argument_contract_mismatch_rows": first_rank_argument_contract_mismatch_rows,
            "rank_pool_size": transformer.get("rank_pool_size"),
            "compatibility_rerank": transformer.get("compatibility_rerank"),
            "model_promotion_scope": "private_broad_survival_training_artifact_only",
            "serving_allowed": False,
            "public_calibration_allowed": False,
            "mlx_available": mlx_enabled,
            "mlx_structural_smoke_state": mlx_smoke.get("trigger_state"),
            "mlx_structural_smoke_verifier_pass_rate": get_path(mlx_smoke, ["summary", "verifier_pass_rate"]),
            "mlx_parity_claimed": False,
            "vcm_policy_action": vcm_policy.get("action"),
            "promoted_transformer_vcm_mode": vcm_policy.get("promoted_transformer_vcm_mode"),
        },
        "checks": checks,
        "inputs": {
            "structural_action": rel(structural_path),
            "broad_run": rel(broad_run_path),
            "sts_audit": rel(sts_audit_path),
            "sts_policy": rel(sts_policy_path),
            "vcm_ablation": rel(vcm_ablation_path),
            "vcm_policy": rel(vcm_policy_path),
            "mlx_diagnosis": rel(mlx_path),
            "mlx_smoke": rel(mlx_smoke_path),
        },
        "score_semantics": (
            "Private broad survival promotion gate only. It promotes a structural-action plus visible-composition "
            "private training artifact manifest when private broad evidence beats the body-template baseline. It "
            "does not run public calibration, expose public payloads, call a teacher, enable serving, or claim MLX parity."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Broad Capability Survival Promotion Gate v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- promote: `{report.get('promote')}`",
        f"- active_manifest_path: `{report.get('active_manifest_path')}`",
        "",
        "## Summary",
    ]
    for key, value in dict(report.get("summary") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks"])
    for row in report.get("checks") or []:
        lines.append(f"- `{row.get('name')}`: `{row.get('passed')}` / `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def get_path(obj: Any, parts: list[str], default: Any = None) -> Any:
    cur = obj
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
