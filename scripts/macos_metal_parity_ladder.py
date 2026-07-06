#!/usr/bin/env python3
"""Run bounded Rust/Metal train-rollout parity ladder evidence.

This script intentionally does not register a Hive task, route scheduler work,
use public data, call a teacher, enable production Metal routing, or promote a
model. It only runs local bounded `train-rollout-metal` CLI tiers and validates
that each tier writes a canonical readout artifact with no-cheat guardrails.
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
DEFAULT_OUT = ROOT / "reports" / "macos_metal_parity_ladder.json"
DEFAULT_MD = ROOT / "reports" / "macos_metal_parity_ladder.md"
TIER_DIR = ROOT / "reports" / "macos_metal_parity_ladder"


TIERS: list[dict[str, Any]] = [
    {
        "id": "tiny_guarded",
        "cases_per_task": 4,
        "epochs": 2,
        "samples_per_launch": 2,
        "rollout_batch": 2,
        "obs_dim": 3,
        "hidden_dim": 4,
        "reservoir_dim": 5,
        "hv_dim": 8,
        "seq_len": 3,
        "output_dim": 4,
        "tolerance": 0.0005,
        "max_kernel_launches": 64,
    },
    {
        "id": "small_guarded",
        "cases_per_task": 8,
        "epochs": 3,
        "samples_per_launch": 4,
        "rollout_batch": 4,
        "obs_dim": 4,
        "hidden_dim": 6,
        "reservoir_dim": 8,
        "hv_dim": 16,
        "seq_len": 4,
        "output_dim": 4,
        "tolerance": 0.0005,
        "max_kernel_launches": 128,
    },
    {
        "id": "medium_guarded",
        "cases_per_task": 12,
        "epochs": 3,
        "samples_per_launch": 4,
        "rollout_batch": 4,
        "obs_dim": 5,
        "hidden_dim": 8,
        "reservoir_dim": 10,
        "hv_dim": 24,
        "seq_len": 5,
        "output_dim": 5,
        "tolerance": 0.0005,
        "max_kernel_launches": 192,
    },
    {
        "id": "wide_guarded",
        "cases_per_task": 16,
        "epochs": 4,
        "samples_per_launch": 4,
        "rollout_batch": 4,
        "obs_dim": 6,
        "hidden_dim": 10,
        "reservoir_dim": 12,
        "hv_dim": 32,
        "seq_len": 6,
        "output_dim": 6,
        "tolerance": 0.0005,
        "max_kernel_launches": 256,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.execute:
        report = run_ladder(started)
    else:
        report = planned_report(started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def planned_report(started: float) -> dict[str, Any]:
    return {
        "policy": "project_theseus_macos_metal_parity_ladder_v0",
        "created_utc": now(),
        "trigger_state": "YELLOW",
        "execute": False,
        "tiers": [planned_tier(row) for row in TIERS],
        "summary": {
            "tier_count": len(TIERS),
            "tier_ok_count": 0,
            "max_kernel_launches": max(int(row["max_kernel_launches"]) for row in TIERS),
            "production_scheduler_routing_enabled": False,
            "model_promotion_allowed": False,
            "train_rollout_parity_claim_allowed": False,
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
        },
        "guardrails": guardrails(),
        "next_action": "Run with --execute to produce bounded local Metal ladder evidence.",
        "score_semantics": score_semantics(),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def run_ladder(started: float) -> dict[str, Any]:
    TIER_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for tier in TIERS:
        rows.append(run_tier(tier))
    hard_failures = [
        {"tier_id": row["tier_id"], "failed_checks": [name for name, ok in row["checks"].items() if not ok]}
        for row in rows
        if not row.get("ok")
    ]
    all_ok = not hard_failures
    return {
        "policy": "project_theseus_macos_metal_parity_ladder_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if all_ok else "RED",
        "execute": True,
        "tiers": rows,
        "summary": {
            "tier_count": len(rows),
            "tier_ok_count": sum(1 for row in rows if row.get("ok")),
            "max_kernel_launches": max(int(row.get("kernel_launches") or 0) for row in rows) if rows else 0,
            "total_kernel_launches": sum(int(row.get("kernel_launches") or 0) for row in rows),
            "max_report_wall_ms": max(float(row.get("report_wall_ms") or 0.0) for row in rows) if rows else 0.0,
            "artifact_count": sum(1 for row in rows if row.get("artifact_validated")),
            "production_scheduler_routing_enabled": False,
            "remote_task_submitted": False,
            "model_promotion_allowed": False,
            "train_rollout_parity_claim_allowed": False,
            "native_hot_loop_parity_claim_allowed": False,
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "fallback_returns": 0,
            "hard_failures": hard_failures,
        },
        "guardrails": guardrails(),
        "next_action": (
            "Keep production routing locked; next Mac-native work is either a larger explicitly bounded ladder "
            "or a separately reviewed scheduler canary policy."
            if all_ok
            else "Fix failed Metal ladder checks before expanding size or changing any scheduler policy."
        ),
        "score_semantics": score_semantics(),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def run_tier(tier: dict[str, Any]) -> dict[str, Any]:
    tier_id = str(tier["id"])
    report_path = TIER_DIR / f"{tier_id}_train_report.json"
    artifact_path = TIER_DIR / f"{tier_id}_readout_artifact.json"
    command = [
        "cargo",
        "run",
        "-p",
        "symliquid-cli",
        "--",
        "train-rollout-metal",
        "--cases-per-task",
        str(tier["cases_per_task"]),
        "--epochs",
        str(tier["epochs"]),
        "--samples-per-launch",
        str(tier["samples_per_launch"]),
        "--rollout-batch",
        str(tier["rollout_batch"]),
        "--obs-dim",
        str(tier["obs_dim"]),
        "--hidden-dim",
        str(tier["hidden_dim"]),
        "--reservoir-dim",
        str(tier["reservoir_dim"]),
        "--hv-dim",
        str(tier["hv_dim"]),
        "--seq-len",
        str(tier["seq_len"]),
        "--output-dim",
        str(tier["output_dim"]),
        "--tolerance",
        str(tier["tolerance"]),
        "--model-out",
        rel(artifact_path),
        "--out",
        rel(report_path),
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    report = read_json(report_path)
    artifact = read_json(artifact_path)
    checks = validate_tier(tier, proc, report, artifact)
    return {
        "tier_id": tier_id,
        "ok": all(checks.values()),
        "config": tier,
        "command": command,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "report": rel(report_path),
        "artifact": rel(artifact_path),
        "kernel_launches": report.get("kernel_launches"),
        "report_wall_ms": get_path(report, ["timing", "report_wall_ms"]),
        "metal_wall_ms": get_path(report, ["timing", "metal_wall_ms"]),
        "cpu_wall_ms": get_path(report, ["timing", "cpu_wall_ms"]),
        "train_metrics": report.get("train_metrics") if isinstance(report.get("train_metrics"), dict) else {},
        "eval_metrics": report.get("eval_metrics") if isinstance(report.get("eval_metrics"), dict) else {},
        "parity_metrics": report.get("parity_metrics") if isinstance(report.get("parity_metrics"), dict) else {},
        "work_receipt": report.get("work_receipt") if isinstance(report.get("work_receipt"), dict) else {},
        "artifact_validated": bool(
            checks.get("artifact_file_loaded")
            and checks.get("artifact_shape_valid")
            and checks.get("artifact_feature_set_matches")
        ),
        "checks": checks,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def validate_tier(
    tier: dict[str, Any],
    proc: subprocess.CompletedProcess[str],
    report: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, bool]:
    output_dim = int(tier["output_dim"])
    hv_dim = int(tier["hv_dim"])
    expected_weights = hv_dim * output_dim
    tolerance = float(tier["tolerance"])
    guard = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    runtime = report.get("runtime_profile") if isinstance(report.get("runtime_profile"), dict) else {}
    contract = report.get("report_contract") if isinstance(report.get("report_contract"), dict) else {}
    receipt = report.get("work_receipt") if isinstance(report.get("work_receipt"), dict) else {}
    artifact_write = report.get("artifact_write") if isinstance(report.get("artifact_write"), dict) else {}
    return {
        "command_returned_zero": proc.returncode == 0,
        "report_loaded": bool(report),
        "report_ok": bool(report.get("ok") and report.get("state") == "GREEN"),
        "command_matches": report.get("command") == "train-rollout-metal",
        "backend_matches": report.get("backend") == "apple_metal" and runtime.get("backend") == "apple_metal",
        "native_rust_owned": runtime.get("native_rust_owned") is True,
        "python_mlx_bridge_not_used": runtime.get("python_mlx_bridge_used") is False,
        "kernel_launches_bounded": 0 < int(report.get("kernel_launches") or 0) <= int(tier["max_kernel_launches"]),
        "train_rows_match": int(report.get("train_rows") or 0) == int(tier["cases_per_task"]),
        "eval_rows_match": int(report.get("eval_rows") or 0) == int(tier["cases_per_task"]),
        "hv_dim_matches": int(report.get("hv_dim") or 0) == hv_dim,
        "output_dim_matches": int(report.get("labels") or 0) == output_dim,
        "tolerance_declared": abs(float(get_path(report, ["args", "tolerance"], 0.0) or 0.0) - tolerance) <= 1.0e-8,
        "tolerance_bounded": 0.0 < tolerance <= 0.0005,
        "work_receipt_accepted": receipt.get("accepted") is True
        and receipt.get("backend") == "apple_metal"
        and receipt.get("task_kind") == "train_rollout_metal_cli",
        "artifact_write_attempted": artifact_write.get("attempted") is True,
        "artifact_write_canonical": artifact_write.get("kind") == "canonical_readout_artifact",
        "artifact_write_production_compatible": artifact_write.get("production_checkpoint_compatible") is True,
        "artifact_file_loaded": bool(artifact),
        "artifact_shape_valid": int(artifact.get("hv_dim") or 0) == hv_dim
        and int(artifact.get("output_dim") or 0) == output_dim
        and len(artifact.get("weights") if isinstance(artifact.get("weights"), list) else []) == expected_weights
        and len(artifact.get("bias") if isinstance(artifact.get("bias"), list) else []) == output_dim
        and len(artifact.get("labels") if isinstance(artifact.get("labels"), list) else []) == output_dim,
        "artifact_feature_set_matches": artifact.get("feature_set")
        == "metal_rollout_memory_readout_sgd_private_synthetic_train_eval",
        "scheduler_routing_still_disabled": contract.get("scheduler_routing_enabled") is False
        and guard.get("does_not_route_scheduler_to_metal") is True,
        "promotion_still_locked": report.get("model_promotion_allowed") is False
        and get_path(report, ["promotion_decision", "promote_to_training_lane"]) is False,
        "parity_claim_still_locked": report.get("train_rollout_parity_claim_allowed") is False
        and report.get("full_cli_parity_claim_allowed") is False
        and report.get("parity_claim_allowed") is False,
        "external_inference_zero": int(report.get("external_inference_calls") or 0) == 0
        and guard.get("no_external_inference") is True,
        "teacher_disabled": report.get("teacher_used") is False and guard.get("no_teacher") is True,
        "public_training_zero": int(report.get("public_training_rows") or 0) == 0
        and guard.get("no_public_training_rows") is True,
        "no_fallback_returns": report.get("symbolic_fallback") is False
        and guard.get("no_fallback_returns") is True,
    }


def planned_tier(tier: dict[str, Any]) -> dict[str, Any]:
    tier_id = str(tier["id"])
    return {
        "tier_id": tier_id,
        "ok": False,
        "config": tier,
        "report": rel(TIER_DIR / f"{tier_id}_train_report.json"),
        "artifact": rel(TIER_DIR / f"{tier_id}_readout_artifact.json"),
        "checks": {"not_executed": False},
    }


def guardrails() -> dict[str, Any]:
    return {
        "local_only": True,
        "remote_task_submitted": False,
        "production_scheduler_routing_enabled": False,
        "model_promotion_allowed": False,
        "train_rollout_parity_claim_allowed": False,
        "native_hot_loop_parity_claim_allowed": False,
        "public_benchmark_training_used": False,
        "teacher_used": False,
        "external_inference_calls": 0,
        "fallback_returns_allowed": False,
        "public_calibration_run": False,
    }


def score_semantics() -> str:
    return (
        "Mac Rust/Metal bounded ladder only. Tiers use deterministic private synthetic train/eval rollout features "
        "and CPU-vs-Metal parity checks through train-rollout-metal with an explicit <=5e-4 f32 numerical tolerance. "
        "This does not run public calibration, train on "
        "public data, call a teacher, use external inference, submit a remote task, enable production scheduler "
        "routing, promote a model, or claim full CUDA-equivalent parity."
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# macOS Metal Parity Ladder",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Execute: `{report.get('execute')}`",
        f"- Tier OK: `{summary.get('tier_ok_count')}/{summary.get('tier_count')}`",
        f"- Max kernel launches: `{summary.get('max_kernel_launches')}`",
        f"- Production scheduler routing enabled: `{summary.get('production_scheduler_routing_enabled')}`",
        f"- Model promotion allowed: `{summary.get('model_promotion_allowed')}`",
        f"- Parity claim allowed: `{summary.get('train_rollout_parity_claim_allowed')}`",
        "",
        "## Tiers",
    ]
    for row in report.get("tiers", []):
        if not isinstance(row, dict):
            continue
        failed = [name for name, ok in (row.get("checks") or {}).items() if not ok]
        lines.append(
            f"- `{row.get('tier_id')}`: ok=`{row.get('ok')}` launches=`{row.get('kernel_launches')}` "
            f"report=`{row.get('report')}` failed=`{failed[:6]}`"
        )
    lines.extend(["", "## Boundary", "", report.get("score_semantics", "")])
    return "\n".join(lines) + "\n"


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
