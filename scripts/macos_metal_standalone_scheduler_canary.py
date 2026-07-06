#!/usr/bin/env python3
"""Run a guarded local-only train-standalone-metal scheduler canary.

This is route-readiness evidence only. It does not register a Hive task, submit
remote work, enable production routing, spend public calibration, call a
teacher, use external inference, allow fallback returns, or promote a model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "macos_metal_standalone_scheduler_canary.json"
DEFAULT_MD = ROOT / "reports" / "macos_metal_standalone_scheduler_canary.md"
ROUTE_POLICY = ROOT / "configs" / "macos_metal_standalone_route_policy.json"
CANARY_POLICY = ROOT / "configs" / "macos_metal_standalone_scheduler_canary_policy.json"
PARITY_AUDIT = ROOT / "reports" / "macos_mlx_parity_audit.json"
BASE_REPORT = ROOT / "reports" / "symliquid_standalone_metal_train_report.json"
BASE_ARTIFACT = ROOT / "reports" / "macos_metal_train_standalone_readout_artifact.json"
TRAIN_REPORT = ROOT / "reports" / "macos_metal_standalone_scheduler_canary_train_report.json"
ARTIFACT = ROOT / "reports" / "macos_metal_standalone_scheduler_canary_readout_artifact.json"
TASK_KIND = "train_standalone_metal_local_canary"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = run_canary(started, execute=args.execute)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report.get("trigger_state") == "RED" else 0


def run_canary(started: float, *, execute: bool) -> dict[str, Any]:
    route_policy = read_json(ROUTE_POLICY, {})
    canary_policy = read_json(CANARY_POLICY, {})
    parity_audit = read_json(PARITY_AUDIT, {})
    base_report = read_json(BASE_REPORT, {})
    base_artifact = read_json(BASE_ARTIFACT, {})
    bounds = dict(canary_policy.get("bounds") if isinstance(canary_policy.get("bounds"), dict) else {})
    command = build_command(bounds)
    preflight = preflight_checks(route_policy, canary_policy, parity_audit, base_report, base_artifact, bounds)

    execution: dict[str, Any] = {"attempted": False, "reason": "execute_not_requested"}
    if execute and all(preflight.values()):
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=240,
        )
        execution = {
            "attempted": True,
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    elif execute:
        execution = {
            "attempted": False,
            "reason": "preflight_failed",
            "failed_checks": [name for name, ok in preflight.items() if not ok],
        }

    train_report = read_json(TRAIN_REPORT, {}) if execute else {}
    artifact = read_json(ARTIFACT, {}) if execute else {}
    result_checks = validate_result(train_report, artifact, bounds) if execute else {"not_applicable_until_execute": True}
    ok = all(preflight.values()) and (not execute or (bool(execution.get("ok")) and all(result_checks.values())))
    guardrails = {
        "local_only": True,
        "remote_task_submitted": False,
        "production_scheduler_routing_enabled": False,
        "registers_worker_chunk": False,
        "model_promotion_allowed": False,
        "train_standalone_parity_claim_allowed": False,
        "native_hot_loop_parity_claim_allowed": False,
        "public_benchmark_training_used": False,
        "public_calibration_run": False,
        "teacher_used": False,
        "external_inference_calls": 0,
        "no_fallback_returns": True,
    }
    return {
        "ok": ok,
        "policy": "project_theseus_macos_metal_standalone_scheduler_canary_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if ok and execute else "YELLOW" if ok else "RED",
        "mode": "execute" if execute else "plan",
        "route_policy": rel(ROUTE_POLICY),
        "canary_policy": rel(CANARY_POLICY),
        "task_kind": TASK_KIND,
        "planned_placement": {
            "target": "local",
            "reason": "Reviewed local-only scheduler canary for train-standalone-metal; production routing remains disabled.",
            "payload": {
                "profile": "reviewed_local_standalone_canary",
                "job_id": "job_macos_metal_train_standalone_scheduler_canary",
                "job_family": "macos_metal_train_standalone_canary",
                "arm_id": "apple_metal_control_arm",
                "backend_requirements": ["apple_metal"],
                "command": "train-standalone-metal",
                "parity_for": "train-standalone-cuda",
                "route_policy": rel(ROUTE_POLICY),
                "canary_policy": rel(CANARY_POLICY),
                "production_scheduler_routing_enabled": False,
                "merge_policy": "append_report_only_no_promotion",
                "priority": 20,
                "lease_seconds": 600,
                "output_artifacts": [
                    {"type": "worker_report", "path": rel(TRAIN_REPORT)},
                    {"type": "readout_artifact", "path": rel(ARTIFACT)},
                ],
                **bounds,
            },
        },
        "command": command,
        "execution": execution,
        "train_report": rel(TRAIN_REPORT),
        "artifact": rel(ARTIFACT),
        "canary_bounds": bounds,
        "checks": {
            "preflight": preflight,
            "result": result_checks,
        },
        "guardrails": guardrails,
        "score_semantics": "Local scheduler-adjacent canary evidence only; not a production routing enablement or parity/promotion claim.",
        "next_action": "keep production routing locked; any route enablement requires a separate operator-reviewed policy change",
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def build_command(bounds: dict[str, Any]) -> list[str]:
    return [
        "cargo",
        "run",
        "-p",
        "symliquid-cli",
        "--",
        "train-standalone-metal",
        "--train-seed",
        str(int(bounds.get("train_seed") or 0)),
        "--eval-seed",
        str(int(bounds.get("eval_seed") or 10000)),
        "--cases-per-task",
        str(int(bounds.get("cases_per_task") or 1)),
        "--epochs",
        str(int(bounds.get("epochs") or 1)),
        "--samples-per-launch",
        str(int(bounds.get("samples_per_launch") or 1)),
        "--hv-dim",
        str(int(bounds.get("hv_dim") or 8)),
        "--lr",
        str(float(bounds.get("lr") or 0.03)),
        "--model-out",
        rel(ARTIFACT),
        "--out",
        rel(TRAIN_REPORT),
    ]


def preflight_checks(
    route_policy: dict[str, Any],
    canary_policy: dict[str, Any],
    parity_audit: dict[str, Any],
    base_report: dict[str, Any],
    base_artifact: dict[str, Any],
    bounds: dict[str, Any],
) -> dict[str, bool]:
    requires = canary_policy.get("requires") if isinstance(canary_policy.get("requires"), dict) else {}
    guardrails = base_report.get("guardrails") if isinstance(base_report.get("guardrails"), dict) else {}
    contract = base_report.get("report_contract") if isinstance(base_report.get("report_contract"), dict) else {}
    artifact_write = base_report.get("artifact_write") if isinstance(base_report.get("artifact_write"), dict) else {}
    return {
        "route_policy_present": bool(route_policy),
        "route_policy_matches": route_policy.get("policy") == "project_theseus_macos_metal_standalone_route_policy_v0",
        "route_policy_command_matches": route_policy.get("command") == "train-standalone-metal",
        "route_policy_backend_matches": route_policy.get("backend") == "apple_metal",
        "route_policy_production_disabled": route_policy.get("production_scheduler_routing_enabled") is False,
        "canary_policy_present": bool(canary_policy),
        "canary_policy_matches": canary_policy.get("policy") == "project_theseus_macos_metal_standalone_scheduler_canary_policy_v0",
        "command_matches": canary_policy.get("command") == "train-standalone-metal",
        "task_kind_matches": canary_policy.get("task_kind") == TASK_KIND,
        "backend_matches": canary_policy.get("backend") == "apple_metal",
        "local_only_policy": canary_policy.get("local_only") is True,
        "remote_task_not_submitted": canary_policy.get("remote_task_submitted") is False,
        "canary_not_registered_worker_chunk": canary_policy.get("registers_worker_chunk") is False,
        "production_scheduler_routing_disabled": canary_policy.get("production_scheduler_routing_enabled") is False,
        "no_remote_task_scope_change": canary_policy.get("does_not_change_hive_remote_task_scope") is True,
        "no_arbitrary_remote_execution": canary_policy.get("no_arbitrary_remote_execution") is True,
        "base_report_ok": base_report.get("ok") is True and base_report.get("command") == "train-standalone-metal",
        "base_artifact_loaded": bool(base_artifact),
        "base_artifact_shape_valid": readout_shape_ok(base_artifact, int(base_report.get("hv_dim") or 0), int(base_report.get("labels") or 0)),
        "base_contract_matches_surface": contract.get("matches_train_standalone_cli_surface") is True,
        "base_python_mlx_bridge_not_used": contract.get("python_mlx_bridge_used") is False,
        "base_scheduler_routing_disabled": contract.get("scheduler_routing_enabled") is False,
        "base_artifact_canonical": artifact_write.get("kind") == "canonical_readout_artifact"
        and artifact_write.get("production_checkpoint_compatible") is True,
        "parity_audit_counts_standalone_artifact": requires.get("macos_mlx_parity_audit_counts_standalone_artifact") is True
        and int(get_path(parity_audit, ["summary", "native_train_standalone_artifact_equivalence_count"], 0) or 0) >= 1,
        "required_reports_declared": all(
            path in canary_policy.get("required_reports", [])
            for path in [
                "reports/symliquid_standalone_metal_train_report.json",
                "reports/macos_metal_train_standalone_readout_artifact.json",
                "reports/macos_mlx_parity_audit.json",
            ]
        ),
        "bounds_cases_bounded": 0 < int(bounds.get("cases_per_task") or 0) <= 16,
        "bounds_epochs_bounded": 0 < int(bounds.get("epochs") or 0) <= 4,
        "bounded_kernel_launch_cap_declared": 0 < int(bounds.get("max_kernel_launches") or 0) <= 128,
        "promotion_locked_by_policy": requires.get("model_promotion_allowed") is False,
        "parity_claim_locked_by_policy": requires.get("train_standalone_parity_claim_allowed") is False
        and requires.get("native_hot_loop_parity_claim_allowed") is False,
        "no_external_inference_by_policy": requires.get("external_inference_calls") == 0,
        "teacher_disabled_by_policy": requires.get("teacher_used") is False,
        "public_training_zero_by_policy": requires.get("public_training_rows") == 0,
        "public_calibration_not_run_by_policy": requires.get("public_calibration_not_run") is True,
        "no_fallback_returns_by_policy": requires.get("no_fallback_returns") is True
        and guardrails.get("no_fallback_returns") is True,
    }


def validate_result(train_report: dict[str, Any], artifact: dict[str, Any], bounds: dict[str, Any]) -> dict[str, bool]:
    guardrails = train_report.get("guardrails") if isinstance(train_report.get("guardrails"), dict) else {}
    contract = train_report.get("report_contract") if isinstance(train_report.get("report_contract"), dict) else {}
    runtime = train_report.get("runtime_profile") if isinstance(train_report.get("runtime_profile"), dict) else {}
    artifact_write = train_report.get("artifact_write") if isinstance(train_report.get("artifact_write"), dict) else {}
    expected_hv_dim = int(train_report.get("hv_dim") or 0)
    expected_output_dim = int(train_report.get("labels") or 0)
    kernel_launches = int(train_report.get("kernel_launches") or 0)
    return {
        "train_report_ok": train_report.get("ok") is True and train_report.get("state") == "GREEN",
        "command_matches": train_report.get("command") == "train-standalone-metal",
        "parity_for_matches": train_report.get("parity_for") == "train-standalone-cuda",
        "backend_matches": train_report.get("backend") == "apple_metal",
        "implementation_matches": train_report.get("implementation") == "rust_metal_structured_cgs_vsa_readout_cli",
        "mode_is_native_metal": runtime.get("backend") == "apple_metal",
        "native_rust_owned": runtime.get("native_rust_owned") == "true",
        "python_mlx_bridge_not_used": runtime.get("python_mlx_bridge_used") == "false"
        and contract.get("python_mlx_bridge_used") is False,
        "symbolic_fallback_false": train_report.get("symbolic_fallback") is False,
        "cuda_fallback_false": train_report.get("cuda_fallback") is False,
        "args_match_bounds": int(get_path(train_report, ["args", "cases_per_task"], 0) or 0) == int(bounds.get("cases_per_task") or 0)
        and int(get_path(train_report, ["args", "epochs"], 0) or 0) == int(bounds.get("epochs") or 0)
        and int(get_path(train_report, ["args", "samples_per_launch"], 0) or 0) == int(bounds.get("samples_per_launch") or 0)
        and int(get_path(train_report, ["args", "hv_dim"], 0) or 0) == int(bounds.get("hv_dim") or 0),
        "kernel_launches_bounded": 0 < kernel_launches <= int(bounds.get("max_kernel_launches") or 0),
        "work_receipt_accepted": get_path(train_report, ["work_receipt", "accepted"], False) is True
        and get_path(train_report, ["work_receipt", "task_kind"], "") == "train_standalone_metal_cli",
        "contract_matches_surface": contract.get("matches_train_standalone_cli_surface") is True,
        "scheduler_routing_still_disabled": contract.get("scheduler_routing_enabled") is False,
        "artifact_write_attempted": artifact_write.get("attempted") is True,
        "artifact_write_canonical": artifact_write.get("kind") == "canonical_readout_artifact",
        "artifact_write_production_compatible": artifact_write.get("production_checkpoint_compatible") is True,
        "artifact_file_loaded": bool(artifact),
        "artifact_shape_valid": readout_shape_ok(artifact, expected_hv_dim, expected_output_dim),
        "feature_set_matches_report": artifact.get("feature_set") == artifact_write.get("feature_set")
        == "structured_cgs_vsa_metal_readout",
        "promotion_still_locked": train_report.get("model_promotion_allowed") is False
        and get_path(train_report, ["promotion_decision", "promote_to_training_lane"], True) is False
        and artifact_write.get("promotion_allowed") is False,
        "parity_claim_still_locked": train_report.get("train_standalone_parity_claim_allowed") is False
        and train_report.get("full_cli_parity_claim_allowed") is False
        and artifact_write.get("train_standalone_parity_claim_allowed") is False,
        "external_inference_zero": int(train_report.get("external_inference_calls") or 0) == 0
        and guardrails.get("no_external_inference") is True,
        "teacher_disabled": train_report.get("teacher_used") is False and guardrails.get("no_teacher") is True,
        "public_training_zero": int(train_report.get("public_training_rows") or 0) == 0
        and guardrails.get("no_public_training_rows") is True,
        "public_calibration_not_run": guardrails.get("no_public_calibration") is True,
        "no_fallback_returns": guardrails.get("no_fallback_returns") is True,
    }


def readout_shape_ok(artifact: dict[str, Any], hv_dim: int, output_dim: int) -> bool:
    labels = artifact.get("labels") if isinstance(artifact.get("labels"), list) else []
    weights = artifact.get("weights") if isinstance(artifact.get("weights"), list) else []
    bias = artifact.get("bias") if isinstance(artifact.get("bias"), list) else []
    return (
        bool(artifact)
        and artifact.get("hv_dim") == hv_dim
        and artifact.get("output_dim") == output_dim
        and len(labels) == output_dim
        and len(weights) == hv_dim * output_dim
        and len(bias) == output_dim
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = [
        "# macOS Metal Standalone Scheduler Canary",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- OK: `{report.get('ok')}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Task kind: `{report.get('task_kind')}`",
        f"- Route policy: `{report.get('route_policy')}`",
        f"- Canary policy: `{report.get('canary_policy')}`",
        f"- Train report: `{report.get('train_report')}`",
        f"- Artifact: `{report.get('artifact')}`",
        "",
        "## Failed Checks",
    ]
    failed: list[str] = []
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    for group, rows in checks.items():
        if isinstance(rows, dict):
            failed.extend(f"{group}:{name}" for name, ok in rows.items() if ok is False)
    if failed:
        summary.extend(f"- `{item}`" for item in sorted(failed))
    else:
        summary.append("- none")
    summary.extend(
        [
            "",
            "## Guardrails",
            "- local-only, no remote task submission, no worker scope change",
            "- production scheduler routing remains disabled",
            "- no public calibration/training rows, no teacher, no external inference",
            "- no fallback returns, no parity claim, no model promotion",
            "",
        ]
    )
    return "\n".join(summary)


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


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
