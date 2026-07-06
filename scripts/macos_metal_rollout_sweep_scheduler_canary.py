#!/usr/bin/env python3
"""Run a guarded local-only train-rollout-metal-sweep scheduler canary.

This is route-readiness evidence only. It does not register a Hive task, submit
remote work, enable production routing, spend public calibration, call a
teacher, use external inference, allow fallback returns, claim CUDA
state-training parity, or promote a model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "macos_metal_rollout_sweep_scheduler_canary.json"
DEFAULT_MD = ROOT / "reports" / "macos_metal_rollout_sweep_scheduler_canary.md"
ROUTE_POLICY = ROOT / "configs" / "macos_metal_rollout_sweep_route_policy.json"
CANARY_POLICY = ROOT / "configs" / "macos_metal_rollout_sweep_scheduler_canary_policy.json"
PARITY_AUDIT = ROOT / "reports" / "macos_mlx_parity_audit.json"
BASE_REPORT = ROOT / "reports" / "symliquid_rollout_metal_sweep.json"
TRAIN_REPORT = ROOT / "reports" / "macos_metal_rollout_sweep_scheduler_canary_train_report.json"
ARTIFACT_DIR = ROOT / "reports" / "macos_metal_rollout_sweep_scheduler_canary_artifacts"
TASK_KIND = "train_rollout_metal_sweep_local_canary"


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
    bounds = dict(canary_policy.get("bounds") if isinstance(canary_policy.get("bounds"), dict) else {})
    command = build_command(bounds)
    preflight = preflight_checks(route_policy, canary_policy, parity_audit, base_report, bounds)

    execution: dict[str, Any] = {"attempted": False, "reason": "execute_not_requested"}
    if execute and all(preflight.values()):
        if ARTIFACT_DIR.exists():
            shutil.rmtree(ARTIFACT_DIR)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
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
    result_checks = validate_result(train_report, bounds) if execute else {"not_applicable_until_execute": True}
    ok = all(preflight.values()) and (not execute or (bool(execution.get("ok")) and all(result_checks.values())))
    guardrails = {
        "local_only": True,
        "remote_task_submitted": False,
        "production_scheduler_routing_enabled": False,
        "registers_worker_chunk": False,
        "model_promotion_allowed": False,
        "train_rollout_sweep_parity_claim_allowed": False,
        "native_hot_loop_parity_claim_allowed": False,
        "state_training_native_ported": False,
        "cuda_state_training_parity_claim_allowed": False,
        "public_benchmark_training_used": False,
        "public_calibration_run": False,
        "teacher_used": False,
        "external_inference_calls": 0,
        "no_fallback_returns": True,
    }
    return {
        "ok": ok,
        "policy": "project_theseus_macos_metal_rollout_sweep_scheduler_canary_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if ok and execute else "YELLOW" if ok else "RED",
        "mode": "execute" if execute else "plan",
        "route_policy": rel(ROUTE_POLICY),
        "canary_policy": rel(CANARY_POLICY),
        "task_kind": TASK_KIND,
        "planned_placement": {
            "target": "local",
            "reason": "Reviewed local-only scheduler canary for train-rollout-metal-sweep; production routing remains disabled.",
            "payload": {
                "profile": "reviewed_local_rollout_sweep_canary",
                "job_id": "job_macos_metal_train_rollout_sweep_scheduler_canary",
                "job_family": "macos_metal_train_rollout_sweep_canary",
                "arm_id": "apple_metal_control_arm",
                "backend_requirements": ["apple_metal"],
                "command": "train-rollout-metal-sweep",
                "parity_for": "train-rollout-cuda-sweep",
                "route_policy": rel(ROUTE_POLICY),
                "canary_policy": rel(CANARY_POLICY),
                "production_scheduler_routing_enabled": False,
                "merge_policy": "append_report_only_no_promotion",
                "priority": 20,
                "lease_seconds": 600,
                "output_artifacts": [
                    {"type": "worker_report", "path": rel(TRAIN_REPORT)},
                    {"type": "readout_artifacts", "path": rel(ARTIFACT_DIR)},
                ],
                **bounds,
            },
        },
        "command": command,
        "execution": execution,
        "train_report": rel(TRAIN_REPORT),
        "artifact_dir": rel(ARTIFACT_DIR),
        "canary_bounds": bounds,
        "checks": {
            "preflight": preflight,
            "result": result_checks,
        },
        "guardrails": guardrails,
        "score_semantics": "Local scheduler-adjacent rollout-sweep canary evidence only; not production routing, state-training parity, promotion, or public calibration.",
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
        "train-rollout-metal-sweep",
        "--train-seeds",
        str(bounds.get("train_seeds") or "0"),
        "--eval-seed-base",
        str(int(bounds.get("eval_seed_base") or 10000)),
        "--cases-per-task",
        str(int(bounds.get("cases_per_task") or 1)),
        "--epochs",
        str(int(bounds.get("epochs") or 1)),
        "--state-epochs",
        str(bounds.get("state_epochs") or "0"),
        "--state-lrs",
        str(bounds.get("state_lrs") or "0.0"),
        "--samples-per-launch",
        str(int(bounds.get("samples_per_launch") or 1)),
        "--probe-cases-per-task",
        str(int(bounds.get("probe_cases_per_task") or 0)),
        "--rollout-batch",
        str(int(bounds.get("rollout_batch") or 1)),
        "--obs-dim",
        str(int(bounds.get("obs_dim") or 1)),
        "--hidden-dim",
        str(int(bounds.get("hidden_dim") or 1)),
        "--reservoir-dim",
        str(int(bounds.get("reservoir_dim") or 1)),
        "--hv-dim",
        str(int(bounds.get("hv_dim") or 8)),
        "--seq-len",
        str(int(bounds.get("seq_len") or 1)),
        "--lr",
        str(float(bounds.get("lr") or 0.03)),
        "--output-dim",
        str(int(bounds.get("output_dim") or 1)),
        "--tolerance",
        str(float(bounds.get("tolerance") or 0.0001)),
        "--artifact-dir",
        rel(ARTIFACT_DIR),
        "--out",
        rel(TRAIN_REPORT),
    ]


def preflight_checks(
    route_policy: dict[str, Any],
    canary_policy: dict[str, Any],
    parity_audit: dict[str, Any],
    base_report: dict[str, Any],
    bounds: dict[str, Any],
) -> dict[str, bool]:
    requires = canary_policy.get("requires") if isinstance(canary_policy.get("requires"), dict) else {}
    guardrails = base_report.get("guardrails") if isinstance(base_report.get("guardrails"), dict) else {}
    contract = base_report.get("report_contract") if isinstance(base_report.get("report_contract"), dict) else {}
    summary = base_report.get("summary") if isinstance(base_report.get("summary"), dict) else {}
    return {
        "route_policy_present": bool(route_policy),
        "route_policy_matches": route_policy.get("policy") == "project_theseus_macos_metal_rollout_sweep_route_policy_v0",
        "route_policy_command_matches": route_policy.get("command") == "train-rollout-metal-sweep",
        "route_policy_backend_matches": route_policy.get("backend") == "apple_metal",
        "route_policy_production_disabled": route_policy.get("production_scheduler_routing_enabled") is False,
        "canary_policy_present": bool(canary_policy),
        "canary_policy_matches": canary_policy.get("policy") == "project_theseus_macos_metal_rollout_sweep_scheduler_canary_policy_v0",
        "command_matches": canary_policy.get("command") == "train-rollout-metal-sweep",
        "task_kind_matches": canary_policy.get("task_kind") == TASK_KIND,
        "backend_matches": canary_policy.get("backend") == "apple_metal",
        "local_only_policy": canary_policy.get("local_only") is True,
        "remote_task_not_submitted": canary_policy.get("remote_task_submitted") is False,
        "canary_not_registered_worker_chunk": canary_policy.get("registers_worker_chunk") is False,
        "production_scheduler_routing_disabled": canary_policy.get("production_scheduler_routing_enabled") is False,
        "no_remote_task_scope_change": canary_policy.get("does_not_change_hive_remote_task_scope") is True,
        "no_arbitrary_remote_execution": canary_policy.get("no_arbitrary_remote_execution") is True,
        "base_report_ok": base_report.get("ok") is True and base_report.get("command") == "train-rollout-metal-sweep",
        "base_contract_matches_surface": contract.get("matches_train_rollout_sweep_cli_surface") is True,
        "base_python_mlx_bridge_not_used": contract.get("python_mlx_bridge_used") is False,
        "base_scheduler_routing_disabled": contract.get("scheduler_routing_enabled") is False,
        "base_state_training_not_claimed": contract.get("state_training_native_ported") is False
        and summary.get("state_training_native_ported") is False
        and summary.get("cuda_state_training_parity_claim_allowed") is False,
        "base_child_artifacts_present": int(summary.get("artifact_count") or 0) > 0,
        "parity_audit_counts_sweep_artifacts": requires.get("macos_mlx_parity_audit_counts_sweep_artifacts") is True
        and int(get_path(parity_audit, ["summary", "native_train_rollout_sweep_child_artifact_count"], 0) or 0) >= 1,
        "required_reports_declared": all(
            path in canary_policy.get("required_reports", [])
            for path in [
                "reports/symliquid_rollout_metal_sweep.json",
                "reports/macos_mlx_parity_audit.json",
            ]
        ),
        "bounds_cases_bounded": 0 < int(bounds.get("cases_per_task") or 0) <= 16,
        "bounds_epochs_bounded": 0 < int(bounds.get("epochs") or 0) <= 4,
        "bounded_child_count_declared": 0 < int(bounds.get("expected_child_runs") or 0) <= 6,
        "bounded_kernel_launch_cap_declared": 0 < int(bounds.get("max_kernel_launches") or 0) <= 256,
        "state_training_locked_by_policy": requires.get("state_training_native_ported") is False
        and requires.get("cuda_state_training_parity_claim_allowed") is False,
        "promotion_locked_by_policy": requires.get("model_promotion_allowed") is False,
        "parity_claim_locked_by_policy": requires.get("train_rollout_sweep_parity_claim_allowed") is False
        and requires.get("native_hot_loop_parity_claim_allowed") is False,
        "no_external_inference_by_policy": requires.get("external_inference_calls") == 0,
        "teacher_disabled_by_policy": requires.get("teacher_used") is False,
        "public_training_zero_by_policy": requires.get("public_training_rows") == 0,
        "public_calibration_not_run_by_policy": requires.get("public_calibration_not_run") is True,
        "no_fallback_returns_by_policy": requires.get("no_fallback_returns") is True
        and guardrails.get("no_fallback_returns") is True,
    }


def validate_result(report: dict[str, Any], bounds: dict[str, Any]) -> dict[str, bool]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    contract = report.get("report_contract") if isinstance(report.get("report_contract"), dict) else {}
    children = report.get("children") if isinstance(report.get("children"), list) else []
    child_validations = [validate_child(child) for child in children if isinstance(child, dict)]
    child_ok_count = sum(1 for row in child_validations if row.get("ok"))
    total_kernel_launches = int(summary.get("total_kernel_launches") or 0)
    return {
        "train_report_ok": report.get("ok") is True and report.get("trigger_state") == "GREEN",
        "command_matches": report.get("command") == "train-rollout-metal-sweep",
        "parity_for_matches": report.get("parity_for") == "train-rollout-cuda-sweep",
        "backend_matches": report.get("backend") == "apple_metal",
        "implementation_matches": report.get("implementation") == "rust_metal_rollout_sweep_guarded_proof",
        "contract_matches_surface": contract.get("matches_train_rollout_sweep_cli_surface") is True,
        "child_command_declared": contract.get("child_command") == "train-rollout-metal",
        "python_mlx_bridge_not_used": contract.get("python_mlx_bridge_used") is False,
        "scheduler_routing_still_disabled": contract.get("scheduler_routing_enabled") is False
        and summary.get("production_scheduler_routing_enabled") is False
        and guardrails.get("production_scheduler_routing_enabled") is False,
        "state_training_not_claimed": contract.get("state_training_native_ported") is False
        and summary.get("state_training_native_ported") is False
        and summary.get("cuda_state_training_parity_claim_allowed") is False
        and guardrails.get("does_not_claim_cuda_state_training_parity") is True,
        "expected_child_runs_match": int(summary.get("run_count") or 0) == int(bounds.get("expected_child_runs") or 0),
        "children_present": len(children) == int(bounds.get("expected_child_runs") or 0),
        "all_children_ok": child_ok_count == len(children) and child_ok_count > 0,
        "artifact_count_matches": int(summary.get("artifact_count") or 0) == len(children),
        "kernel_launches_bounded": 0 < total_kernel_launches <= int(bounds.get("max_kernel_launches") or 0),
        "work_receipt_accepted": get_path(report, ["work_receipt", "accepted"], False) is True
        and get_path(report, ["work_receipt", "task_kind"], "") == "train_rollout_metal_sweep_cli",
        "promotion_still_locked": report.get("model_promotion_allowed") is False
        and get_path(report, ["promotion_decision", "promote_to_training_lane"], True) is False,
        "parity_claim_still_locked": report.get("train_rollout_sweep_parity_claim_allowed") is False
        and report.get("full_cli_parity_claim_allowed") is False
        and summary.get("native_hot_loop_parity_claim_allowed") is False,
        "external_inference_zero": int(report.get("external_inference_calls") or 0) == 0
        and int(summary.get("external_inference_calls") or 0) == 0
        and guardrails.get("no_external_inference") is True,
        "teacher_disabled": report.get("teacher_used") is False
        and summary.get("teacher_used") is False
        and guardrails.get("no_teacher") is True,
        "public_training_zero": int(report.get("public_training_rows") or 0) == 0
        and int(summary.get("public_training_rows") or 0) == 0
        and guardrails.get("no_public_training_rows") is True,
        "public_calibration_not_run": guardrails.get("no_public_calibration") is True,
        "no_fallback_returns": int(summary.get("fallback_returns") or 0) == 0
        and guardrails.get("no_fallback_returns") is True,
    }


def validate_child(child: dict[str, Any]) -> dict[str, Any]:
    path = get_path(child, ["sweep_child", "child_report_path"], "")
    full = read_json(resolve(path), {}) if isinstance(path, str) and path else {}
    if not full:
        full = child
    artifact_path = get_path(full, ["artifact_write", "path"], "")
    artifact = read_json(resolve(artifact_path), {}) if isinstance(artifact_path, str) and artifact_path else {}
    hv_dim = int(full.get("hv_dim") or 0)
    output_dim = int(full.get("labels") or 0)
    checks = {
        "child_report_ok": full.get("ok") is True and full.get("state") == "GREEN",
        "child_command_matches": full.get("command") == "train-rollout-metal",
        "child_backend_matches": full.get("backend") == "apple_metal",
        "child_parent_matches": get_path(full, ["sweep_child", "parent_command"], "") == "train-rollout-metal-sweep",
        "child_state_training_not_claimed": get_path(full, ["sweep_child", "state_training_native_ported"], True) is False,
        "artifact_loaded": bool(artifact),
        "artifact_shape_valid": readout_shape_ok(artifact, hv_dim, output_dim),
        "child_scheduler_routing_disabled": get_path(full, ["report_contract", "scheduler_routing_enabled"], True) is False,
        "child_python_mlx_bridge_not_used": get_path(full, ["report_contract", "python_mlx_bridge_used"], True) is False,
        "child_no_fallback_returns": get_path(full, ["guardrails", "no_fallback_returns"], False) is True,
        "child_external_inference_zero": int(full.get("external_inference_calls") or 0) == 0,
        "child_teacher_disabled": full.get("teacher_used") is False,
        "child_public_training_zero": int(full.get("public_training_rows") or 0) == 0,
        "child_promotion_locked": full.get("model_promotion_allowed") is False
        and get_path(full, ["promotion_decision", "promote_to_training_lane"], True) is False,
        "child_parity_claim_locked": full.get("train_rollout_parity_claim_allowed") is False
        and full.get("full_cli_parity_claim_allowed") is False,
    }
    return {"ok": all(checks.values()), "path": path, "artifact": artifact_path, "checks": checks}


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
    lines = [
        "# macOS Metal Rollout-Sweep Scheduler Canary",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- OK: `{report.get('ok')}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Task kind: `{report.get('task_kind')}`",
        f"- Route policy: `{report.get('route_policy')}`",
        f"- Canary policy: `{report.get('canary_policy')}`",
        f"- Train report: `{report.get('train_report')}`",
        f"- Artifact dir: `{report.get('artifact_dir')}`",
        "",
        "## Failed Checks",
    ]
    failed: list[str] = []
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    for group, rows in checks.items():
        if isinstance(rows, dict):
            failed.extend(f"{group}:{name}" for name, ok in rows.items() if ok is False)
    lines.extend(f"- `{item}`" for item in sorted(failed)) if failed else lines.append("- none")
    lines.extend(
        [
            "",
            "## Guardrails",
            "- local-only, no remote task submission, no worker scope change",
            "- production scheduler routing remains disabled",
            "- no public calibration/training rows, no teacher, no external inference",
            "- no fallback returns, no state-training parity claim, no model promotion",
            "",
        ]
    )
    return "\n".join(lines)


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
    except (FileNotFoundError, json.JSONDecodeError, IsADirectoryError):
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
