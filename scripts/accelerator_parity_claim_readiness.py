#!/usr/bin/env python3
"""Fail-closed readiness gate for full accelerator parity claims.

Routing approval and parity-claim approval are separate decisions. This script
does not enable scheduler routing and does not set any parity flags. It only
checks whether enough apples-to-apples CUDA-vs-Metal evidence exists to support
a future human-reviewed parity claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_MANIFEST = REPORTS / "accelerator_parity_manifest.json"
DEFAULT_ROUTE = REPORTS / "macos_metal_production_route_readiness.json"
DEFAULT_APPROVAL = ROOT / "configs" / "accelerator_parity_claim_approval.json"
DEFAULT_OUT = REPORTS / "accelerator_parity_claim_readiness.json"
DEFAULT_MD = REPORTS / "accelerator_parity_claim_readiness.md"

CUDA_REFERENCE_REPORTS = {
    "standalone_readout_cli": REPORTS / "symliquid_standalone_cuda_train_report.json",
    "rollout_cli": REPORTS / "symliquid_rollout_cuda_train_report.json",
    "rollout_sweep_cli": REPORTS / "symliquid_rollout_cuda_sweep.json",
    "token_superposition_cli": REPORTS / "token_superposition_cuda_training.json",
}

RESULT_METRIC_KEYS = {
    "train_accuracy",
    "eval_accuracy",
    "train_loss",
    "eval_loss",
    "best_eval_accuracy",
    "best_eval_loss",
    "baseline_combined_loss",
    "best_variant_combined_loss",
    "loss_final",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=rel(DEFAULT_MANIFEST))
    parser.add_argument("--route-readiness", default=rel(DEFAULT_ROUTE))
    parser.add_argument("--approval", default=rel(DEFAULT_APPROVAL))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    report = build_report(
        manifest_path=resolve(args.manifest),
        route_path=resolve(args.route_readiness),
        approval_path=resolve(args.approval),
        tolerance=max(0.0, float(args.tolerance)),
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") != "RED" else 2


def build_report(manifest_path: Path, route_path: Path, approval_path: Path, tolerance: float) -> dict[str, Any]:
    manifest = read_json(manifest_path, {})
    route = read_json(route_path, {})
    evidence_hash = sha256_paths([manifest_path, route_path, *CUDA_REFERENCE_REPORTS.values()])
    approval = validate_approval(approval_path, evidence_hash)
    rows = [
        parity_surface_row(row, tolerance)
        for row in manifest.get("rows", [])
        if isinstance(row, dict) and row.get("metal_equivalent")
    ]
    no_cheat_ok = bool(
        manifest.get("ok")
        and get_path(manifest, ["summary", "external_inference_calls"], 1) == 0
        and get_path(manifest, ["summary", "teacher_used_count"], 1) == 0
        and get_path(manifest, ["summary", "public_training_rows"], 1) == 0
        and get_path(manifest, ["summary", "model_promotion_allowed_count"], 1) == 0
        and get_path(manifest, ["summary", "production_routing_enabled_count"], 1) == 0
        and get_path(route, ["summary", "hard_failure_count"], 1) == 0
    )
    parity_ready = bool(rows) and all(row["parity_evidence_ok"] for row in rows)
    parity_claim_allowed = bool(no_cheat_ok and parity_ready and approval["valid"])
    blockers = sorted({blocker for row in rows for blocker in row["blockers"]})
    if not approval["valid"]:
        blockers.extend(f"approval:{reason}" for reason in approval["reasons"])
    if not no_cheat_ok:
        blockers.append("no_cheat_evidence_not_green")
    return {
        "ok": no_cheat_ok,
        "policy": "project_theseus_accelerator_parity_claim_readiness_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if parity_claim_allowed else "YELLOW" if no_cheat_ok else "RED",
        "score_semantics": (
            "Parity-claim readiness only. This report does not enable production routing, "
            "does not promote a model, does not spend public calibration, does not call a "
            "teacher or external inference, and does not set full CUDA/MLX/Metal parity flags."
        ),
        "summary": {
            "surface_count": len(rows),
            "parity_ready_surface_count": sum(1 for row in rows if row["parity_evidence_ok"]),
            "no_cheat_ok": no_cheat_ok,
            "approval_valid": approval["valid"],
            "parity_claim_allowed": parity_claim_allowed,
            "tolerance": tolerance,
            "external_inference_calls": 0,
            "teacher_used_count": 0,
            "public_training_rows": 0,
            "model_promotion_allowed_count": 0,
            "production_routing_enabled_count": 0,
        },
        "inputs": {
            "accelerator_parity_manifest": rel(manifest_path),
            "macos_metal_production_route_readiness": rel(route_path),
            "approval": rel(approval_path),
        },
        "approval": approval,
        "approval_template": {
            "policy": "project_theseus_accelerator_parity_claim_approval_v1",
            "approved": False,
            "evidence_sha256": evidence_hash,
            "max_surfaces": len(rows),
            "claim_scope": "audit_statement_only_no_runtime_routing",
            "allow_scheduler_routing": False,
            "allow_model_promotion": False,
            "allow_public_training_rows": False,
            "operator_note": (
                "Set approved=true only after reviewing apples-to-apples CUDA and Metal "
                "result equivalence across every listed real training surface. This approval "
                "must not be used to enable scheduler routing or model promotion."
            ),
        },
        "surfaces": rows,
        "blockers": sorted(set(blockers)),
        "guardrails": {
            "routing_approval_is_separate": True,
            "does_not_enable_scheduler_routing": True,
            "does_not_change_remote_task_scope": True,
            "no_public_calibration": True,
            "no_public_training_rows": True,
            "no_teacher": True,
            "no_external_inference": True,
            "no_fallback_returns_required": True,
            "no_model_promotion": True,
            "full_parity_flags_remain_false": True,
        },
        "next_actions": next_actions(no_cheat_ok, parity_ready, approval),
        "external_inference_calls": 0,
    }


def parity_surface_row(row: dict[str, Any], tolerance: float) -> dict[str, Any]:
    surface = str(row.get("surface") or "")
    cuda_path = CUDA_REFERENCE_REPORTS.get(surface)
    metal_path_text = get_path(row, ["metal", "path"], "")
    cuda = read_json(cuda_path, {}) if cuda_path else {}
    metal = read_json(resolve(metal_path_text), {}) if metal_path_text else {}
    cuda_metrics = extract_metrics(cuda)
    metal_metrics = extract_metrics(metal)
    common = sorted((set(cuda_metrics) & set(metal_metrics)) & RESULT_METRIC_KEYS)
    metric_deltas = {
        key: abs(float(cuda_metrics[key]) - float(metal_metrics[key]))
        for key in common
        if is_number(cuda_metrics.get(key)) and is_number(metal_metrics.get(key))
    }
    max_delta = max(metric_deltas.values()) if metric_deltas else None
    checks = {
        "cuda_reference_report_present": bool(cuda),
        "cuda_reference_is_cuda": report_is_cuda(cuda, row),
        "metal_report_present": bool(metal),
        "metal_report_ok": metal.get("ok") is True,
        "metal_parity_for_matches": metal.get("parity_for") == row.get("cuda_equivalent")
        or get_path(row, ["metal", "parity_for"], "") == row.get("cuda_equivalent"),
        "common_result_metrics_present": bool(common),
        "metric_delta_within_tolerance": bool(metric_deltas) and max_delta is not None and max_delta <= tolerance,
        "no_fallback_returns": get_path(metal, ["guardrails", "no_fallback_returns"], False) is True,
        "external_inference_zero": int(metal.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": metal.get("teacher_used") is False,
        "public_training_zero": int(metal.get("public_training_rows") or 0) == 0,
        "parity_claim_locked": row.get("full_parity_claim_allowed") is False
        and metal.get("full_cli_parity_claim_allowed") is False,
        "production_routing_locked": row.get("production_scheduler_routing_enabled") is False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "surface": surface,
        "cuda_equivalent": row.get("cuda_equivalent"),
        "metal_equivalent": row.get("metal_equivalent"),
        "cuda_reference_report": rel(cuda_path) if cuda_path else "",
        "metal_report": metal_path_text,
        "parity_evidence_ok": all(checks.values()),
        "checks": checks,
        "common_result_metrics": common,
        "metric_deltas": metric_deltas,
        "max_metric_delta": max_delta,
        "blockers": blockers,
    }


def validate_approval(path: Path, expected_hash: str) -> dict[str, Any]:
    approval = read_json(path, {})
    reasons: list[str] = []
    if approval.get("policy") != "project_theseus_accelerator_parity_claim_approval_v1":
        reasons.append("approval_policy_missing")
    if approval.get("approved") is not True:
        reasons.append("approved_not_true")
    if approval.get("evidence_sha256") != expected_hash:
        reasons.append("evidence_sha256_mismatch")
    if approval.get("claim_scope") != "audit_statement_only_no_runtime_routing":
        reasons.append("claim_scope_mismatch")
    if approval.get("allow_scheduler_routing") is not False:
        reasons.append("scheduler_routing_not_forbidden")
    if approval.get("allow_model_promotion") is not False:
        reasons.append("model_promotion_not_forbidden")
    if approval.get("allow_public_training_rows") is not False:
        reasons.append("public_training_rows_not_forbidden")
    return {
        "path": rel(path),
        "exists": path.exists(),
        "valid": not reasons,
        "reasons": reasons,
        "evidence_sha256": expected_hash,
    }


def report_is_cuda(report: dict[str, Any], row: dict[str, Any]) -> bool:
    if not report:
        return False
    text = " ".join(
        str(value)
        for value in [
            report.get("backend"),
            report.get("command"),
            report.get("policy"),
            report.get("parity_for"),
        ]
    ).lower()
    expected = str(row.get("cuda_equivalent") or "").lower()
    return "cuda" in text and (not expected or expected in text or report.get("command") == row.get("cuda_equivalent"))


def extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    if isinstance(report.get("metrics"), dict):
        return report["metrics"]
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    best = report.get("best_run") if isinstance(report.get("best_run"), dict) else {}
    best_metrics = best.get("metrics") if isinstance(best.get("metrics"), dict) else {}
    merged: dict[str, Any] = {}
    merged.update(summary)
    merged.update(best_metrics)
    return merged


def next_actions(no_cheat_ok: bool, parity_ready: bool, approval: dict[str, Any]) -> list[str]:
    if not no_cheat_ok:
        return ["Fix no-cheat accelerator evidence before considering any parity claim."]
    if not parity_ready:
        return [
            "Generate matching CUDA reference reports for each real training surface on a CUDA node.",
            "Compare CUDA and Metal result metrics under the same config before claiming full parity.",
        ]
    if not approval["valid"]:
        return ["Parity evidence is ready for operator review, but full parity remains locked until explicit approval."]
    return ["Parity claim readiness is approved for audit wording only; production routing remains separately locked."]


def sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(rel(path).encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def read_json(path: Path | None, default: Any) -> Any:
    if path is None:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Accelerator Parity Claim Readiness",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Parity claim allowed: `{summary.get('parity_claim_allowed')}`",
        f"- No-cheat evidence ok: `{summary.get('no_cheat_ok')}`",
        f"- Surfaces ready: `{summary.get('parity_ready_surface_count')}/{summary.get('surface_count')}`",
        f"- Approval valid: `{summary.get('approval_valid')}`",
        "",
        "| Surface | Ready | CUDA Reference | Metal Report | Blockers |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in report.get("surfaces", []):
        lines.append(
            "| {surface} | `{ready}` | `{cuda}` | `{metal}` | {blockers} |".format(
                surface=row.get("surface"),
                ready=row.get("parity_evidence_ok"),
                cuda=row.get("cuda_reference_report"),
                metal=row.get("metal_report"),
                blockers=", ".join(row.get("blockers") or []) or "",
            )
        )
    lines.extend(["", str(report.get("score_semantics") or ""), ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
