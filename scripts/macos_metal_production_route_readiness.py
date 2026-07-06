"""Fail-closed production-route readiness review for Mac Rust/Metal training.

This report is deliberately not a route enabler. It consolidates the guarded
Metal proof/canary evidence and names the exact blockers that must remain
cleared before the scheduler can route production work to Metal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "macos_metal_production_route_readiness.json"
DEFAULT_MD = ROOT / "reports" / "macos_metal_production_route_readiness.md"
DEFAULT_APPROVAL = ROOT / "configs" / "macos_metal_production_route_approval.json"

MACOS_MLX_PARITY_AUDIT = ROOT / "reports" / "macos_mlx_parity_audit.json"
ACCELERATOR_PARITY_MANIFEST = ROOT / "reports" / "accelerator_parity_manifest.json"
METAL_STATE_TRAINING_PROOF = ROOT / "reports" / "macos_metal_rollout_state_training_proof.json"


SURFACES: list[dict[str, Any]] = [
    {
        "surface": "standalone_readout_cli",
        "cuda_equivalent": "train-standalone-cuda",
        "metal_command": "train-standalone-metal",
        "report": ROOT / "reports" / "symliquid_standalone_metal_train_report.json",
        "route_policy": ROOT / "configs" / "macos_metal_standalone_route_policy.json",
        "scheduler_canary": ROOT / "reports" / "macos_metal_standalone_scheduler_canary.json",
        "parity_claim_key": "train_standalone_parity_claim_allowed",
        "contract_key": "matches_train_standalone_cli_surface",
        "requires_state_training_native": False,
    },
    {
        "surface": "rollout_cli",
        "cuda_equivalent": "train-rollout-cuda",
        "metal_command": "train-rollout-metal",
        "report": ROOT / "reports" / "symliquid_rollout_metal_train_report.json",
        "route_policy": ROOT / "configs" / "macos_metal_route_policy.json",
        "scheduler_canary": ROOT / "reports" / "macos_metal_scheduler_canary.json",
        "parity_claim_key": "train_rollout_parity_claim_allowed",
        "contract_key": "matches_train_rollout_cli_surface",
        "requires_state_training_native": False,
    },
    {
        "surface": "rollout_sweep_cli",
        "cuda_equivalent": "train-rollout-cuda-sweep",
        "metal_command": "train-rollout-metal-sweep",
        "report": ROOT / "reports" / "symliquid_rollout_metal_sweep.json",
        "route_policy": ROOT / "configs" / "macos_metal_rollout_sweep_route_policy.json",
        "scheduler_canary": ROOT / "reports" / "macos_metal_rollout_sweep_scheduler_canary.json",
        "state_training_proof": METAL_STATE_TRAINING_PROOF,
        "parity_claim_key": "train_rollout_sweep_parity_claim_allowed",
        "contract_key": "matches_train_rollout_sweep_cli_surface",
        "requires_state_training_native": True,
    },
    {
        "surface": "token_superposition_cli",
        "cuda_equivalent": "train-token-superposition-cuda",
        "metal_command": "train-token-superposition-metal",
        "report": ROOT / "reports" / "token_superposition_metal_training.json",
        "route_policy": ROOT / "configs" / "macos_metal_token_superposition_route_policy.json",
        "scheduler_canary": ROOT / "reports" / "macos_metal_token_superposition_scheduler_canary.json",
        "parity_claim_key": "train_token_superposition_parity_claim_allowed",
        "contract_key": "matches_train_token_superposition_cli_surface",
        "requires_state_training_native": False,
    },
]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path | None, default: Any = None) -> Any:
    if path is None:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for part in path:
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        elif isinstance(cur, list) and isinstance(part, int) and 0 <= part < len(cur):
            cur = cur[part]
        else:
            return default
    return cur


def int_value(data: Any, paths: list[list[Any]], default: int = -1) -> int:
    for path in paths:
        value = get_path(data, path, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
    return default


def bool_value(data: Any, paths: list[list[Any]], default: bool = False) -> bool:
    for path in paths:
        value = get_path(data, path, None)
        if isinstance(value, bool):
            return value
    return default


def artifact_write_ok(report: dict[str, Any]) -> bool:
    artifact = report.get("artifact_write") or {}
    if not artifact:
        artifact = get_path(report, ["best_run", "artifact_write"], {}) or {}
    if not artifact:
        return int_value(report, [["summary", "artifact_count"]], 0) > 0
    return bool(
        artifact.get("attempted")
        and artifact.get("kind") == "canonical_readout_artifact"
        and artifact.get("production_checkpoint_compatible") is True
        and artifact.get("promotion_allowed") is False
    )


def evidence_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_approval(path: Path, expected_hash: str) -> dict[str, Any]:
    approval = read_json(path, {})
    reasons = []
    approved_surfaces = approval.get("approved_surfaces")
    if approved_surfaces is None:
        approved_surfaces = [surface["surface"] for surface in SURFACES]
    if not isinstance(approved_surfaces, list):
        approved_surfaces = []
    approved_surfaces = sorted(str(item) for item in approved_surfaces)
    known_surfaces = sorted(surface["surface"] for surface in SURFACES)
    if approval.get("policy") != "project_theseus_macos_metal_production_route_approval_v1":
        reasons.append("approval_policy_missing")
    if approval.get("approved") is not True:
        reasons.append("approved_not_true")
    if approval.get("evidence_sha256") != expected_hash:
        reasons.append("evidence_sha256_mismatch")
    if approval.get("max_surfaces") != len(approved_surfaces):
        reasons.append("max_surfaces_mismatch")
    if not approved_surfaces:
        reasons.append("approved_surfaces_empty")
    if any(surface not in known_surfaces for surface in approved_surfaces):
        reasons.append("approved_surfaces_unknown")
    if approval.get("allow_remote_task_scope_change") is not False:
        reasons.append("remote_task_scope_change_not_forbidden")
    if approval.get("allow_public_training_rows") is not False:
        reasons.append("public_training_rows_not_forbidden")
    return {
        "path": rel(path),
        "exists": path.exists(),
        "valid": not reasons,
        "reasons": reasons,
        "approved_surfaces": approved_surfaces,
        "approval": approval if approval else {},
    }


def route_policy_checks(policy: dict[str, Any], surface: dict[str, Any]) -> dict[str, bool]:
    if not policy:
        return {
            "route_policy_present": False,
            "command_matches": False,
            "backend_matches": False,
            "scope_unchanged": False,
            "no_arbitrary_remote_execution": False,
            "production_scheduler_routing_enabled": False,
        }
    return {
        "route_policy_present": True,
        "command_matches": policy.get("command") == surface["metal_command"],
        "backend_matches": policy.get("backend") == "apple_metal",
        "scope_unchanged": policy.get("does_not_change_hive_remote_task_scope") is True,
        "no_arbitrary_remote_execution": policy.get("no_arbitrary_remote_execution") is True,
        "production_scheduler_routing_enabled": policy.get("production_scheduler_routing_enabled") is True,
    }


def canary_checks(report: dict[str, Any]) -> dict[str, bool]:
    guardrails = report.get("guardrails") or {}
    return {
        "canary_present": bool(report),
        "canary_ok": report.get("ok") is True,
        "local_only": guardrails.get("local_only") is True,
        "remote_task_not_submitted": guardrails.get("remote_task_submitted") is False,
        "registers_worker_chunk_false": guardrails.get("registers_worker_chunk") is False,
        "production_scheduler_routing_disabled": guardrails.get("production_scheduler_routing_enabled") is False,
        "model_promotion_locked": guardrails.get("model_promotion_allowed") is False,
        "native_parity_claim_locked": guardrails.get("native_hot_loop_parity_claim_allowed") is False,
        "teacher_disabled": guardrails.get("teacher_used") is False,
        "external_inference_zero": int(guardrails.get("external_inference_calls") or 0) == 0,
        "no_fallback_returns": guardrails.get("no_fallback_returns") is True,
    }


def state_training_proof_checks(report: dict[str, Any]) -> dict[str, bool]:
    guardrails = report.get("guardrails") or {}
    state_training = report.get("state_training") if isinstance(report.get("state_training"), dict) else {}
    return {
        "state_training_proof_present": bool(report),
        "state_training_proof_ok": report.get("ok") is True and report.get("state") == "GREEN",
        "policy_matches": report.get("policy")
        == "project_theseus_macos_metal_rollout_state_training_proof_v0",
        "state_training_semantics_proof": report.get("state_training_semantics_proof") is True,
        "state_training_native_ported": report.get("state_training_native_ported") is True,
        "state_update_on_metal": get_path(report, ["runtime_profile", "state_update_on_metal"]) is True,
        "rollout_state_on_metal": get_path(report, ["runtime_profile", "rollout_state_on_metal"]) is True,
        "python_mlx_bridge_not_used": get_path(report, ["runtime_profile", "python_mlx_bridge_used"]) is False,
        "decision_matches": state_training.get("decision_matches") is True,
        "params_changed": state_training.get("params_changed_from_base") is True,
        "kernel_launches_positive": int(report.get("kernel_launches") or 0) > 0,
        "external_inference_zero": int(report.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": report.get("teacher_used") is False,
        "public_training_zero": int(report.get("public_training_rows") or 0) == 0,
        "fallback_returns_zero": int(report.get("fallback_returns") or 0) == 0,
        "no_fallback_returns": guardrails.get("no_fallback_returns") is True,
        "no_public_calibration": guardrails.get("no_public_calibration") is True,
        "no_public_training_rows": guardrails.get("no_public_training_rows") is True,
        "no_teacher": guardrails.get("no_teacher") is True,
        "no_external_inference": guardrails.get("no_external_inference") is True,
        "production_scheduler_routing_disabled": report.get("production_scheduler_routing_enabled") is False,
        "cuda_state_training_parity_claim_locked": report.get("cuda_state_training_parity_claim_allowed")
        is False,
        "train_rollout_sweep_parity_claim_locked": report.get("train_rollout_sweep_parity_claim_allowed")
        is False,
        "full_cli_parity_claim_locked": report.get("full_cli_parity_claim_allowed") is False,
        "model_promotion_locked": report.get("model_promotion_allowed") is False,
    }


def surface_row(surface: dict[str, Any], approval_valid: bool, approved_surfaces: set[str]) -> dict[str, Any]:
    report = read_json(surface["report"], {})
    policy = read_json(surface["route_policy"], {}) if surface.get("route_policy") else {}
    canary = read_json(surface["scheduler_canary"], {}) if surface.get("scheduler_canary") else {}
    state_proof = read_json(surface.get("state_training_proof"), {}) if surface.get("state_training_proof") else {}
    guardrails = report.get("guardrails") or {}
    summary = report.get("summary") or {}
    contract = report.get("report_contract") or {}
    surface_approved = surface["surface"] in approved_surfaces

    route_checks = route_policy_checks(policy, surface)
    scheduler_checks = canary_checks(canary)
    state_checks = state_training_proof_checks(state_proof) if surface.get("state_training_proof") else {}
    state = report.get("state") or report.get("trigger_state")
    evidence_checks = {
        "report_present": bool(report),
        "report_ok": report.get("ok") is True,
        "state_green_or_not_required": state in {None, "GREEN"},
        "command_matches": report.get("command") == surface["metal_command"],
        "backend_matches": report.get("backend") == "apple_metal",
        "parity_for_matches": report.get("parity_for") == surface["cuda_equivalent"],
        "contract_matches_surface": contract.get(surface["contract_key"]) is True,
        "python_mlx_bridge_not_used": contract.get("python_mlx_bridge_used") is False,
        "report_scheduler_routing_disabled": contract.get("scheduler_routing_enabled") is False
        or guardrails.get("scheduler_routing_enabled") is False,
        "artifact_write_ok": artifact_write_ok(report),
        "external_inference_zero": int(report.get("external_inference_calls") or summary.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": (report.get("teacher_used") if "teacher_used" in report else summary.get("teacher_used")) is False,
        "public_training_zero": int(report.get("public_training_rows") or summary.get("public_training_rows") or 0) == 0,
        "no_fallback_returns": guardrails.get("no_fallback_returns") is True
        and int(summary.get("fallback_returns") or 0) == 0,
        "model_promotion_locked": report.get("model_promotion_allowed") is False
        or summary.get("model_promotion_allowed") is False,
        "surface_parity_claim_locked": report.get(surface["parity_claim_key"]) is False
        or summary.get(surface["parity_claim_key"]) is False,
        "full_cli_parity_claim_locked": report.get("full_cli_parity_claim_allowed") is False,
        "production_scheduler_routing_disabled": summary.get("production_scheduler_routing_enabled") is False
        or guardrails.get("production_scheduler_routing_enabled") is False
        or contract.get("scheduler_routing_enabled") is False,
    }
    production_checks = {
        "operator_approval_valid": approval_valid and surface_approved,
        "production_scheduler_routing_enabled": route_checks.get("production_scheduler_routing_enabled") is True,
    }
    if surface.get("requires_state_training_native"):
        production_checks["state_training_native_ported"] = (
            bool(state_checks) and all(state_checks.values())
        )
        production_checks["cuda_state_training_parity_claim_allowed"] = (
            summary.get("cuda_state_training_parity_claim_allowed") is True
        )

    guarded_evidence_ok = all(evidence_checks.values())
    route_evidence_ok = all(
        value
        for name, value in route_checks.items()
        if name != "production_scheduler_routing_enabled"
    )
    canary_required = surface.get("scheduler_canary") is not None
    canary_evidence_ok = all(scheduler_checks.values()) if canary_required else False
    production_route_ready = bool(
        surface_approved
        and
        guarded_evidence_ok
        and route_evidence_ok
        and canary_evidence_ok
        and all(production_checks.values())
    )

    blockers: list[str] = []
    if not guarded_evidence_ok:
        blockers.extend(f"guarded_evidence:{name}" for name, passed in evidence_checks.items() if not passed)
    if not route_evidence_ok:
        blockers.extend(f"route_policy:{name}" for name, passed in route_checks.items() if name != "production_scheduler_routing_enabled" and not passed)
    if surface.get("route_policy") is None:
        blockers.append("route_policy:missing_for_surface")
    if canary_required and not canary_evidence_ok:
        blockers.extend(f"scheduler_canary:{name}" for name, passed in scheduler_checks.items() if not passed)
    if surface.get("scheduler_canary") is None:
        blockers.append("scheduler_canary:missing_for_surface")
    if surface.get("requires_state_training_native") and not all(state_checks.values()):
        blockers.extend(f"state_training_proof:{name}" for name, passed in state_checks.items() if not passed)
    if surface_approved:
        blockers.extend(
            f"production_prerequisite:{name}"
            for name, passed in production_checks.items()
            if not passed
        )

    return {
        "surface": surface["surface"],
        "surface_approved_for_production": surface_approved,
        "cuda_equivalent": surface["cuda_equivalent"],
        "metal_command": surface["metal_command"],
        "report": rel(surface["report"]),
        "route_policy": rel(surface["route_policy"]),
        "scheduler_canary": rel(surface["scheduler_canary"]),
        "state_training_proof": rel(surface.get("state_training_proof")),
        "guarded_evidence_ok": guarded_evidence_ok,
        "route_evidence_ok": route_evidence_ok,
        "scheduler_canary_evidence_ok": canary_evidence_ok,
        "production_route_ready": production_route_ready,
        "checks": {
            "guarded_evidence": evidence_checks,
            "route_policy": route_checks,
            "scheduler_canary": scheduler_checks,
            "state_training_proof": state_checks,
            "production_prerequisites": production_checks,
        },
        "blockers": sorted(set(blockers)),
    }


def build_report(approval_path: Path) -> dict[str, Any]:
    evidence_paths = [
        MACOS_MLX_PARITY_AUDIT,
        ACCELERATOR_PARITY_MANIFEST,
        *(surface["report"] for surface in SURFACES),
        *(surface["route_policy"] for surface in SURFACES if surface.get("route_policy")),
        *(surface["scheduler_canary"] for surface in SURFACES if surface.get("scheduler_canary")),
        *(surface["state_training_proof"] for surface in SURFACES if surface.get("state_training_proof")),
    ]
    evidence_sha = evidence_hash(evidence_paths)
    approval = validate_approval(approval_path, evidence_sha)
    approved_surfaces = set(approval.get("approved_surfaces") or [])
    rows = [surface_row(surface, bool(approval["valid"]), approved_surfaces) for surface in SURFACES]

    parity_audit = read_json(MACOS_MLX_PARITY_AUDIT, {})
    accelerator_manifest = read_json(ACCELERATOR_PARITY_MANIFEST, {})
    guarded_ok = [row for row in rows if row["guarded_evidence_ok"]]
    route_ok = [row for row in rows if row["route_evidence_ok"]]
    canary_ok = [row for row in rows if row["scheduler_canary_evidence_ok"]]
    state_training_ok = [
        row
        for row in rows
        if row.get("checks", {}).get("state_training_proof")
        and all(row.get("checks", {}).get("state_training_proof", {}).values())
    ]
    approved_rows = [row for row in rows if row.get("surface_approved_for_production")]
    production_ready = [row for row in approved_rows if row["production_route_ready"]]
    all_blockers = sorted({blocker for row in rows for blocker in row["blockers"]})
    hard_failures = [
        row
        for row in rows
        if not row["guarded_evidence_ok"]
        or not get_path(row, ["checks", "guarded_evidence", "external_inference_zero"], False)
        or not get_path(row, ["checks", "guarded_evidence", "teacher_disabled"], False)
        or not get_path(row, ["checks", "guarded_evidence", "public_training_zero"], False)
        or not get_path(row, ["checks", "guarded_evidence", "no_fallback_returns"], False)
    ]
    no_cheat_ok = not hard_failures
    production_route_allowed = bool(approved_rows) and len(production_ready) == len(approved_rows)
    if production_route_allowed:
        next_actions = [
            "Bounded local production Metal routing is ready for the approved surfaces only.",
            "Keep rollout-sweep production routing disabled until its route policy enables scheduler routing and a separate approval covers that surface.",
            "Do not treat route readiness as CUDA/Metal parity, model-promotion approval, public calibration approval, teacher approval, or remote task-scope expansion.",
        ]
    else:
        next_actions = [
            "Keep production routing disabled until explicit operator route approval exists.",
            "Add missing route policies/canaries for any surface intended for production routing.",
            "Do not route rollout-sweep work to Metal as production until parity-claim approval and scheduler routing approval are both explicit.",
        ]

    return {
        "ok": no_cheat_ok,
        "policy": "project_theseus_macos_metal_production_route_readiness_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if production_route_allowed else "YELLOW" if no_cheat_ok else "RED",
        "score_semantics": (
            "Production-route readiness review only. This report does not enable "
            "Metal scheduler routing, register remote worker chunks, spend public "
            "calibration, call a teacher, call external inference, promote a model, "
            "or claim full CUDA/MLX/Metal parity."
        ),
        "summary": {
            "surface_count": len(rows),
            "guarded_evidence_ok_count": len(guarded_ok),
            "route_evidence_ok_count": len(route_ok),
            "scheduler_canary_evidence_ok_count": len(canary_ok),
            "state_training_native_proof_ok_count": len(state_training_ok),
            "production_route_ready_count": len(production_ready),
            "approved_surface_count": len(approved_rows),
            "approved_surfaces": sorted(approved_surfaces),
            "production_route_allowed": production_route_allowed,
            "operator_approval_valid": bool(approval["valid"]),
            "hard_failure_count": len(hard_failures),
            "blocker_count": len(all_blockers),
            "kernel_parity_ready_count": int_value(parity_audit, [["summary", "kernel_parity_ready_count"]], 0),
            "kernel_parity_pending_count": int_value(parity_audit, [["summary", "kernel_parity_pending_count"]], 0),
            "native_hot_loop_parity_claim_allowed": bool_value(
                parity_audit,
                [["summary", "native_hot_loop_parity_claim_allowed"]],
                False,
            ),
            "accelerator_manifest_ok": accelerator_manifest.get("ok") is True
            and accelerator_manifest.get("trigger_state") == "GREEN",
            "external_inference_calls": 0,
            "teacher_used_count": 0,
            "public_training_rows": 0,
            "model_promotion_allowed_count": 0,
        },
        "approval": approval,
        "approval_template": {
            "policy": "project_theseus_macos_metal_production_route_approval_v1",
            "approved": False,
            "max_surfaces": 0,
            "approved_surfaces": [],
            "evidence_sha256": evidence_sha,
            "allow_remote_task_scope_change": False,
            "allow_public_training_rows": False,
            "operator_note": (
                "Set approved=true only after reviewing every surface blocker and "
                "intentionally enabling bounded production Metal routing. Do not use "
                "this to enable arbitrary remote execution or public training."
            ),
        },
        "surfaces": rows,
        "blockers": all_blockers,
        "hard_failures": [row["surface"] for row in hard_failures],
        "guardrails": {
            "does_not_enable_scheduler_routing": True,
            "does_not_register_worker_chunks": True,
            "does_not_change_remote_task_scope": True,
            "no_arbitrary_remote_execution": True,
            "no_public_calibration": True,
            "no_public_training_rows": True,
            "no_teacher": True,
            "no_external_inference": True,
            "no_fallback_returns_required": True,
            "no_model_promotion": True,
        },
        "next_actions": next_actions,
        "external_inference_calls": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# macOS Metal Production Route Readiness",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Production route allowed: `{get_path(report, ['summary', 'production_route_allowed'])}`",
        f"- Guarded evidence ok: `{get_path(report, ['summary', 'guarded_evidence_ok_count'])}/{get_path(report, ['summary', 'surface_count'])}`",
        f"- Production route ready: `{get_path(report, ['summary', 'production_route_ready_count'])}/{get_path(report, ['summary', 'surface_count'])}`",
        f"- Native state-training proofs ok: `{get_path(report, ['summary', 'state_training_native_proof_ok_count'])}`",
        f"- Operator approval valid: `{get_path(report, ['summary', 'operator_approval_valid'])}`",
        f"- Hard failures: `{get_path(report, ['summary', 'hard_failure_count'])}`",
        "",
        "## Surface Status",
        "",
        "| Surface | Guarded Evidence | Route Evidence | Canary Evidence | Production Ready | Blockers |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("surfaces", []):
        blockers = ", ".join(row.get("blockers") or [])
        lines.append(
            "| {surface} | `{guarded}` | `{route}` | `{canary}` | `{ready}` | {blockers} |".format(
                surface=row.get("surface"),
                guarded=row.get("guarded_evidence_ok"),
                route=row.get("route_evidence_ok"),
                canary=row.get("scheduler_canary_evidence_ok"),
                ready=row.get("production_route_ready"),
                blockers=blockers or "",
            )
        )
    lines.extend(
        [
            "",
            "## Rules",
            "",
            report.get("score_semantics", ""),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Review Mac Metal production-route readiness without enabling it.")
    parser.add_argument("--approval", default=str(DEFAULT_APPROVAL.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD.relative_to(ROOT)))
    args = parser.parse_args()

    report = build_report(ROOT / args.approval)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = ROOT / args.markdown_out
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("trigger_state") != "RED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
    surface_approved = surface["surface"] in approved_surfaces
