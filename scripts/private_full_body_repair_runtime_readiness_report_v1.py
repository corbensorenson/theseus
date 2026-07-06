#!/usr/bin/env python3
"""Consolidate the private full-body repair/runtime cleanup evidence packet.

This report intentionally consumes existing private/governance reports only. It
does not run public calibration, write training rows, or call a teacher.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPORTS: dict[str, str] = {
    "candidate_replay": "reports/private_candidate_replay_contract_audit_v1.json",
    "semantic_ablation": "reports/private_full_body_semantic_quality_ablation_private_repair_runtime_v1.json",
    "residual_consumer": "reports/private_residual_target_consumer_private_repair_runtime_v1.json",
    "promotion_gate": "reports/broad_capability_survival_promotion_gate_private_repair_runtime_v1.json",
    "survival_decision": "reports/broad_capability_survival_lane_decision_private_repair_runtime_v1.json",
    "autonomy_readiness": "reports/autonomy_launch_readiness_private_repair_runtime_v1.json",
    "artifact_retention": "reports/theseus_artifact_retention_private_repair_runtime_v1.json",
    "resource_governor": "reports/resource_governor_private_repair_runtime_v1.json",
    "mlx_diagnosis": "reports/macos_mlx_environment_diagnosis_private_repair_runtime_v1.json",
    "vcm_runtime": "reports/vcm_native_runtime_probe_private_repair_runtime_v1.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"trigger_state": "MISSING", "summary": {"missing_path": str(path)}}
    except json.JSONDecodeError as exc:
        return {
            "trigger_state": "UNREADABLE",
            "summary": {"path": str(path), "error": f"{exc.__class__.__name__}: {exc}"},
        }


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def hard_failures(packet: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []

    replay = packet["candidate_replay"]
    replay_summary = replay.get("summary", {})
    if replay.get("trigger_state") != "GREEN":
        failures.append("candidate_replay_not_green")
    if replay_summary.get("unexplained_no_candidate_count") != 0:
        failures.append("candidate_replay_has_unexplained_no_candidate")
    if replay_summary.get("selected_intended_behavior_pass_rate") != 1.0:
        failures.append("candidate_replay_selected_pass_below_1")
    if replay_summary.get("fallback_return_candidate_count") != 0:
        failures.append("candidate_replay_has_fallback_returns")

    residual = packet["residual_consumer"]
    residual_summary = residual.get("summary", {})
    if residual.get("trigger_state") != "GREEN":
        failures.append("residual_consumer_not_green")
    if residual_summary.get("unresolved_target_count") not in (0, None):
        failures.append("private_residual_targets_unresolved")
    if residual_summary.get("public_training_rows_written") not in (0, None):
        failures.append("residual_consumer_wrote_public_training_rows")

    semantic = packet["semantic_ablation"]
    semantic_summary = semantic.get("summary", {})
    if semantic.get("trigger_state") != "GREEN":
        failures.append("semantic_ablation_not_green")
    if semantic_summary.get("fallback_return_candidate_count") not in (0, None):
        failures.append("semantic_ablation_has_fallback_returns")
    if semantic_summary.get("public_leakage_count") not in (0, None):
        failures.append("semantic_ablation_has_public_leakage")

    promotion = packet["promotion_gate"]
    promotion_summary = promotion.get("summary", {})
    if promotion.get("trigger_state") != "GREEN":
        failures.append("promotion_gate_not_green")
    if promotion_summary.get("public_calibration_allowed") is not False:
        failures.append("promotion_gate_allows_public_calibration")
    if promotion_summary.get("serving_allowed") is not False:
        failures.append("private_promotion_allows_runtime_serving")

    autonomy = packet["autonomy_readiness"]
    if autonomy.get("trigger_state") != "GREEN":
        failures.append("autonomy_readiness_not_green")
    if nested(autonomy, "summary", "ready_for_autonomous_training") is not True:
        failures.append("overnight_autonomous_training_not_ready")

    retention = packet["artifact_retention"]
    retention_summary = retention.get("summary", {})
    if retention.get("trigger_state") != "GREEN":
        failures.append("artifact_retention_not_green")
    if retention_summary.get("failed_count", 0) != 0:
        failures.append("artifact_retention_failures")

    resource = packet["resource_governor"]
    if resource.get("trigger_state") != "GREEN":
        failures.append("resource_governor_not_green")
    if nested(resource, "summary", "can_run_requested_profile") is not True:
        failures.append("resource_governor_cannot_run_profile")

    mlx = packet["mlx_diagnosis"]
    resource_summary = packet["resource_governor"].get("summary", {})
    if mlx.get("trigger_state") != "GREEN":
        failures.append("mlx_diagnosis_not_green")
    mlx_summary = mlx.get("summary", {})
    mlx_usable = (
        mlx_summary.get("mlx_usable") is True
        or mlx_summary.get("active_python_status") == "usable"
        or resource_summary.get("mlx_usable") is True
    )
    if not mlx_usable:
        failures.append("mlx_not_usable")

    vcm = packet["vcm_runtime"]
    if vcm.get("trigger_state") != "GREEN":
        failures.append("vcm_runtime_not_green")
    if nested(vcm, "summary", "fallback_return_count") not in (0, None):
        failures.append("vcm_runtime_has_fallback_returns")

    return failures


def compact_summary(packet: dict[str, dict[str, Any]], failures: list[str]) -> dict[str, Any]:
    replay = packet["candidate_replay"].get("summary", {})
    semantic = packet["semantic_ablation"].get("summary", {})
    residual = packet["residual_consumer"].get("summary", {})
    promotion = packet["promotion_gate"].get("summary", {})
    decision = packet["survival_decision"]
    autonomy = packet["autonomy_readiness"].get("summary", {})
    retention = packet["artifact_retention"].get("summary", {})
    resource = packet["resource_governor"].get("summary", {})
    mlx = packet["mlx_diagnosis"].get("summary", {})
    vcm = packet["vcm_runtime"].get("summary", {})

    recommended_lane = "block_on_failures"
    if not failures:
        recommended_lane = "promote_transformer_hybrid_structural_private_training_lane"

    return {
        "trigger_state": "GREEN" if not failures else "RED",
        "created_utc": utc_now(),
        "policy": "project_theseus_private_full_body_repair_runtime_readiness_v1",
        "hard_failure_count": len(failures),
        "hard_failures": failures,
        "no_public_calibration_run": True,
        "public_training_rows_written": max(
            int(residual.get("public_training_rows_written") or 0),
            int(semantic.get("public_training_rows_written") or 0),
            int(vcm.get("public_training_rows_written") or 0),
        ),
        "external_inference_calls": max(
            int(residual.get("external_inference_calls") or 0),
            int(semantic.get("external_inference_calls") or 0),
            int(vcm.get("external_inference_calls") or 0),
        ),
        "fallback_return_count": max(
            int(replay.get("fallback_return_candidate_count") or 0),
            int(semantic.get("fallback_return_candidate_count") or 0),
            int(vcm.get("fallback_return_count") or 0),
        ),
        "candidate_replay": {
            "task_count": replay.get("task_count"),
            "candidate_row_count": replay.get("candidate_row_count"),
            "eligible_candidate_count": replay.get("eligible_candidate_count"),
            "selected_compile_pass_rate": replay.get("selected_compile_pass_rate"),
            "selected_runtime_load_rate": replay.get("selected_runtime_load_rate"),
            "selected_intended_behavior_pass_rate": replay.get("selected_intended_behavior_pass_rate"),
            "pass_if_any_rate": replay.get("pass_if_any_rate"),
            "unexplained_no_candidate_count": replay.get("unexplained_no_candidate_count"),
        },
        "semantic_repair": {
            "best_private_public_shaped_selected_pass_rate": semantic.get(
                "best_private_public_shaped_selected_pass_rate"
            ),
            "best_private_public_shaped_pass_if_any_rate": semantic.get(
                "best_private_public_shaped_pass_if_any_rate"
            ),
            "post_v4_current_release_selected_pass_rate": semantic.get(
                "post_v4_current_release_selected_pass_rate"
            ),
            "v4_strict_novel_learned_only_pass_rate": semantic.get(
                "v4_strict_novel_learned_only_pass_rate"
            ),
            "matched_sts_off_private_pass_rate": semantic.get("matched_sts_off_private_pass_rate"),
            "sts_delta": semantic.get("sts_delta"),
        },
        "residual_targets": {
            "target_rows": residual.get("target_rows"),
            "covered_target_count": residual.get("covered_target_count"),
            "unresolved_target_count": residual.get("unresolved_target_count"),
            "repair_queue_rows": residual.get("repair_queue_rows"),
            "target_categories": residual.get("target_categories", {}),
        },
        "survival_lane": {
            "recommended_lane": recommended_lane,
            "arm_id": promotion.get("arm_id"),
            "baseline_pass_rate": promotion.get("baseline_pass_rate"),
            "structural_only_pass_rate": promotion.get("structural_only_pass_rate"),
            "augmented_pass_rate": promotion.get("augmented_pass_rate"),
            "vcm_policy_action": promotion.get("vcm_policy_action"),
            "promotion_scope": promotion.get("model_promotion_scope"),
            "serving_allowed": promotion.get("serving_allowed"),
            "public_calibration_allowed": promotion.get("public_calibration_allowed"),
            "decision": decision.get("decision"),
        },
        "autonomy": {
            "ready_for_autonomous_training": autonomy.get("ready_for_autonomous_training"),
            "ready_for_teacher_enabled_run": autonomy.get("ready_for_teacher_enabled_run"),
            "blocker_failure_count": autonomy.get("blocker_failure_count"),
            "warning_failure_count": autonomy.get("warning_failure_count"),
            "warning_failures": autonomy.get("warning_failures", []),
        },
        "runtime_cleanup": {
            "archived_count": retention.get("archived_count"),
            "failed_count": retention.get("failed_count"),
            "estimated_reclaimed_gib": retention.get("estimated_reclaimed_gib"),
            "manifest": retention.get("manifest"),
        },
        "mac_native": {
            "resource_governor_state": packet["resource_governor"].get("trigger_state"),
            "can_run_requested_profile": resource.get("can_run_requested_profile"),
            "execution_owner": resource.get("execution_owner"),
            "mlx_diagnosis_state": packet["mlx_diagnosis"].get("trigger_state"),
            "mlx_usable": mlx.get("mlx_usable", resource.get("mlx_usable")),
            "metal_usable": mlx.get("metal_usable", resource.get("metal_usable")),
            "mlx_lm_available": mlx.get("mlx_lm_available"),
            "recommended_python": mlx.get("recommended_python"),
            "mlx_parity_claimed": False,
        },
        "vcm_runtime": {
            "semantic_runtime_lifecycle_green": vcm.get("semantic_runtime_lifecycle_green"),
            "native_runtime_claimable": vcm.get("native_runtime_claimable"),
            "native_runtime_claim_scope": vcm.get("native_runtime_claim_scope"),
            "native_runtime_claim_backend": vcm.get("native_runtime_claim_backend"),
            "native_runtime_claim_backend_matches_recommended_execution_backend": vcm.get(
                "native_runtime_claim_backend_matches_recommended_execution_backend"
            ),
            "recommended_backend": vcm.get("recommended_backend"),
            "recommended_backend_native_runtime_claimable": vcm.get(
                "recommended_backend_native_runtime_claimable"
            ),
            "scheduler_vcm_descriptor_route_allowed_for_recommended_backend": vcm.get(
                "scheduler_vcm_descriptor_route_allowed_for_recommended_backend"
            ),
            "scheduler_native_kv_route_allowed_for_recommended_backend": vcm.get(
                "scheduler_native_kv_route_allowed_for_recommended_backend"
            ),
            "accelerator_kv_parity_claimed": vcm.get("accelerator_kv_parity_claimed"),
        },
        "remaining_walls": [
            "Latest public transfer score remains the locked calibration result until a future governed one-shot public run.",
            "Private full-body semantics are green, but public transfer is not proven by this private-only packet.",
            "MLX is usable and descriptor routing is ready, but MLX/Metal native KV parity is not claimed.",
            "Retention reclaimed the largest active checkpoint JSONs, but reports/data/dist still need ongoing retention discipline.",
            "Autonomy is GREEN for overnight private training, while legacy breadth-freeze warnings remain outside the current priority.",
        ],
        "recommendation": (
            "Use the transformer/hybrid structural full-body student as the private training survival lane, "
            "keep SymLiquid as a protected matched-compute comparator, keep public calibration locked until a "
            "separate one-shot review, and keep moving Mac hot loops toward real MLX/Metal parity."
        ),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Private Full-Body Repair Runtime Readiness v1",
        "",
        f"- State: **{summary['trigger_state']}**",
        f"- Created UTC: {summary['created_utc']}",
        f"- Hard failures: {summary['hard_failure_count']}",
        f"- Public calibration run: {'no' if summary['no_public_calibration_run'] else 'yes'}",
        f"- Public training rows written: {summary['public_training_rows_written']}",
        f"- External inference calls: {summary['external_inference_calls']}",
        f"- Fallback returns: {summary['fallback_return_count']}",
        "",
        "## Candidate Replay",
        "",
        f"- Tasks replayed: {summary['candidate_replay']['task_count']}",
        f"- Candidate rows: {summary['candidate_replay']['candidate_row_count']}",
        f"- Eligible candidates: {summary['candidate_replay']['eligible_candidate_count']}",
        f"- Selected compile/load/pass rates: {summary['candidate_replay']['selected_compile_pass_rate']} / {summary['candidate_replay']['selected_runtime_load_rate']} / {summary['candidate_replay']['selected_intended_behavior_pass_rate']}",
        f"- Unexplained no-candidate count: {summary['candidate_replay']['unexplained_no_candidate_count']}",
        "",
        "## Semantic Repair",
        "",
        f"- Best private public-shaped selected/pass-if-any: {summary['semantic_repair']['best_private_public_shaped_selected_pass_rate']} / {summary['semantic_repair']['best_private_public_shaped_pass_if_any_rate']}",
        f"- Post-v4 current release selected pass: {summary['semantic_repair']['post_v4_current_release_selected_pass_rate']}",
        f"- Strict novel learned-only pass: {summary['semantic_repair']['v4_strict_novel_learned_only_pass_rate']}",
        f"- Matched STS-off private pass: {summary['semantic_repair']['matched_sts_off_private_pass_rate']}",
        "",
        "## Residual Targets",
        "",
        f"- Target rows: {summary['residual_targets']['target_rows']}",
        f"- Covered targets: {summary['residual_targets']['covered_target_count']}",
        f"- Unresolved targets: {summary['residual_targets']['unresolved_target_count']}",
        f"- Repair queue rows: {summary['residual_targets']['repair_queue_rows']}",
        "",
        "## Survival Lane",
        "",
        f"- Recommended lane: {summary['survival_lane']['recommended_lane']}",
        f"- Arm: {summary['survival_lane']['arm_id']}",
        f"- Baseline / structural / augmented pass rates: {summary['survival_lane']['baseline_pass_rate']} / {summary['survival_lane']['structural_only_pass_rate']} / {summary['survival_lane']['augmented_pass_rate']}",
        f"- VCM policy: {summary['survival_lane']['vcm_policy_action']}",
        f"- Runtime serving allowed: {summary['survival_lane']['serving_allowed']}",
        f"- Public calibration allowed: {summary['survival_lane']['public_calibration_allowed']}",
        "",
        "## Runtime And Mac",
        "",
        f"- Autonomy private overnight ready: {summary['autonomy']['ready_for_autonomous_training']}",
        f"- Teacher-enabled run ready: {summary['autonomy']['ready_for_teacher_enabled_run']}",
        f"- Retention archived / reclaimed GiB: {summary['runtime_cleanup']['archived_count']} / {summary['runtime_cleanup']['estimated_reclaimed_gib']}",
        f"- Mac execution owner: {summary['mac_native']['execution_owner']}",
        f"- MLX usable / Metal usable / mlx-lm available: {summary['mac_native']['mlx_usable']} / {summary['mac_native']['metal_usable']} / {summary['mac_native']['mlx_lm_available']}",
        f"- VCM native runtime claim scope: {summary['vcm_runtime']['native_runtime_claim_scope']}",
        f"- VCM recommended backend native KV route allowed: {summary['vcm_runtime']['scheduler_native_kv_route_allowed_for_recommended_backend']}",
        "",
        "## Remaining Walls",
        "",
    ]
    lines.extend(f"- {wall}" for wall in summary["remaining_walls"])
    lines.extend(["", "## Recommendation", "", summary["recommendation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/private_full_body_repair_runtime_readiness_v1.json")
    parser.add_argument("--markdown-out", default="reports/private_full_body_repair_runtime_readiness_v1.md")
    args = parser.parse_args()

    packet = {name: read_json(Path(path)) for name, path in DEFAULT_REPORTS.items()}
    failures = hard_failures(packet)
    summary = compact_summary(packet, failures)
    report = {
        "trigger_state": summary["trigger_state"],
        "created_utc": summary["created_utc"],
        "summary": summary,
        "source_reports": DEFAULT_REPORTS,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(args.markdown_out), summary)
    print(json.dumps({"trigger_state": report["trigger_state"], "out": args.out, "markdown_out": args.markdown_out}, sort_keys=True))
    return 0 if summary["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
