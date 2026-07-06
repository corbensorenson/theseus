#!/usr/bin/env python3
"""Run bounded Rust/Metal token-superposition ladder evidence.

This script intentionally does not register a Hive task, route scheduler work,
use public benchmark data, call a teacher, enable production Metal routing, or
promote a model. It only runs local bounded `train-token-superposition-metal`
tiers over private residual curriculum files and validates the report contract.
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
DEFAULT_OUT = ROOT / "reports" / "macos_metal_token_superposition_ladder.json"
DEFAULT_MD = ROOT / "reports" / "macos_metal_token_superposition_ladder.md"
TIER_DIR = ROOT / "reports" / "macos_metal_token_superposition_ladder"
ROUTE_POLICY = ROOT / "configs" / "macos_metal_token_superposition_route_policy.json"
PRIVATE_INPUT_ROOT = ROOT / "data" / "training_data" / "high_transfer" / "private_train"

PRIVATE_INPUTS = [
    "data/training_data/high_transfer/private_train/targeted_private_residual_curriculum_v2_residual_code_lm_tasks.jsonl",
    "data/training_data/high_transfer/private_train/algorithmic_planning_residual_code_lm_tasks.jsonl",
    "data/training_data/high_transfer/private_train/parsing_encoding_v1_private_residual_curriculum_residual_code_lm_tasks.jsonl",
]

TIERS: list[dict[str, Any]] = [
    {
        "id": "tiny_private_guarded",
        "train_seed": 2026061401,
        "max_language_rows": 48,
        "max_vocab": 24,
        "hv_dim": 32,
        "train_samples": 48,
        "eval_samples": 24,
        "baseline_epochs": 1,
        "bag_sizes": "2",
        "recovery_ratios": "0.5",
        "samples_per_launch": 8,
        "gate_tolerance": 0.002,
        "max_kernel_launches": 32,
    },
    {
        "id": "small_private_guarded",
        "train_seed": 2026061402,
        "max_language_rows": 96,
        "max_vocab": 32,
        "hv_dim": 64,
        "train_samples": 96,
        "eval_samples": 32,
        "baseline_epochs": 2,
        "bag_sizes": "4",
        "recovery_ratios": "0.5",
        "samples_per_launch": 8,
        "gate_tolerance": 0.002,
        "max_kernel_launches": 96,
    },
    {
        "id": "medium_private_guarded",
        "train_seed": 2026061403,
        "max_language_rows": 144,
        "max_vocab": 48,
        "hv_dim": 96,
        "train_samples": 128,
        "eval_samples": 48,
        "baseline_epochs": 2,
        "bag_sizes": "4,8",
        "recovery_ratios": "0.25,0.5",
        "samples_per_launch": 8,
        "gate_tolerance": 0.002,
        "max_kernel_launches": 160,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    route_policy = read_json(ROUTE_POLICY)
    if args.execute:
        report = run_ladder(started, route_policy)
    else:
        report = planned_report(started, route_policy)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def planned_report(started: float, route_policy: dict[str, Any]) -> dict[str, Any]:
    route_policy_validation = validate_route_policy(route_policy)
    return {
        "policy": "project_theseus_macos_metal_token_superposition_ladder_v0",
        "created_utc": now(),
        "trigger_state": "YELLOW",
        "execute": False,
        "route_policy": route_policy_summary(route_policy, route_policy_validation),
        "tiers": [planned_tier(row) for row in TIERS],
        "summary": {
            "tier_count": len(TIERS),
            "tier_ok_count": 0,
            "max_kernel_launches": max(int(row["max_kernel_launches"]) for row in TIERS),
            "production_scheduler_routing_enabled": False,
            "remote_task_submitted": False,
            "model_promotion_allowed": False,
            "train_token_superposition_parity_claim_allowed": False,
            "native_hot_loop_parity_claim_allowed": False,
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "fallback_returns": 0,
        },
        "guardrails": guardrails(),
        "next_action": "Run with --execute to produce bounded local Metal token-superposition ladder evidence.",
        "score_semantics": score_semantics(),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def run_ladder(started: float, route_policy: dict[str, Any]) -> dict[str, Any]:
    TIER_DIR.mkdir(parents=True, exist_ok=True)
    route_policy_validation = validate_route_policy(route_policy)
    rows = [run_tier(tier, route_policy_validation) for tier in TIERS]
    hard_failures = [
        {"tier_id": row["tier_id"], "failed_checks": [name for name, ok in row["checks"].items() if not ok]}
        for row in rows
        if not row.get("ok")
    ]
    all_ok = bool(route_policy_validation.get("ok")) and not hard_failures
    return {
        "policy": "project_theseus_macos_metal_token_superposition_ladder_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if all_ok else "RED",
        "execute": True,
        "route_policy": route_policy_summary(route_policy, route_policy_validation),
        "tiers": rows,
        "summary": {
            "tier_count": len(rows),
            "tier_ok_count": sum(1 for row in rows if row.get("ok")),
            "max_kernel_launches": max(int(row.get("kernel_launches") or 0) for row in rows) if rows else 0,
            "total_kernel_launches": sum(int(row.get("kernel_launches") or 0) for row in rows),
            "max_report_wall_ms": max(float(row.get("total_timing_ms") or 0.0) for row in rows) if rows else 0.0,
            "production_scheduler_routing_enabled": False,
            "remote_task_submitted": False,
            "model_promotion_allowed": False,
            "train_token_superposition_parity_claim_allowed": False,
            "native_hot_loop_parity_claim_allowed": False,
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "fallback_returns": 0,
            "hard_failures": hard_failures,
        },
        "guardrails": guardrails(),
        "next_action": (
            "Keep production routing locked; next Mac-native work is a larger explicitly bounded ladder, "
            "artifact-equivalence work, or an operator-reviewed route gate."
            if all_ok
            else "Fix failed Metal token-superposition ladder checks before expanding size or changing any scheduler policy."
        ),
        "score_semantics": score_semantics(),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def run_tier(tier: dict[str, Any], route_policy_validation: dict[str, Any]) -> dict[str, Any]:
    tier_id = str(tier["id"])
    report_path = TIER_DIR / f"{tier_id}_report.json"
    command = [
        "cargo",
        "run",
        "-p",
        "symliquid-cli",
        "--",
        "train-token-superposition-metal",
        "--input",
        ",".join(PRIVATE_INPUTS),
        "--train-seed",
        str(tier["train_seed"]),
        "--max-language-rows",
        str(tier["max_language_rows"]),
        "--max-code-files",
        "0",
        "--max-chars-per-doc",
        "6000",
        "--max-vocab",
        str(tier["max_vocab"]),
        "--hv-dim",
        str(tier["hv_dim"]),
        "--train-samples",
        str(tier["train_samples"]),
        "--eval-samples",
        str(tier["eval_samples"]),
        "--baseline-epochs",
        str(tier["baseline_epochs"]),
        "--bag-sizes",
        str(tier["bag_sizes"]),
        "--recovery-ratios",
        str(tier["recovery_ratios"]),
        "--lr",
        "0.03",
        "--samples-per-launch",
        str(tier["samples_per_launch"]),
        "--gate-tolerance",
        str(tier["gate_tolerance"]),
        "--out",
        rel(report_path),
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    report = read_json(report_path)
    checks = validate_tier(tier, proc, report, route_policy_validation)
    return {
        "tier_id": tier_id,
        "ok": all(checks.values()),
        "config": tier,
        "command": command,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "report": rel(report_path),
        "kernel_launches": kernel_launches(report),
        "total_timing_ms": get_path(report, ["timing_breakdown_ms", "total"]),
        "dataset": report.get("dataset") if isinstance(report.get("dataset"), dict) else {},
        "metrics": report.get("metrics") if isinstance(report.get("metrics"), dict) else {},
        "raw_gate_promotion_decision": report.get("raw_gate_promotion_decision")
        if isinstance(report.get("raw_gate_promotion_decision"), dict)
        else {},
        "work_receipt": report.get("work_receipt") if isinstance(report.get("work_receipt"), dict) else {},
        "checks": checks,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def validate_tier(
    tier: dict[str, Any],
    proc: subprocess.CompletedProcess[str],
    report: dict[str, Any],
    route_policy_validation: dict[str, Any],
) -> dict[str, bool]:
    guard = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    contract = report.get("report_contract") if isinstance(report.get("report_contract"), dict) else {}
    receipt = report.get("work_receipt") if isinstance(report.get("work_receipt"), dict) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    args = report.get("args") if isinstance(report.get("args"), dict) else {}
    baseline = report.get("baseline") if isinstance(report.get("baseline"), dict) else {}
    variants = report.get("variants") if isinstance(report.get("variants"), list) else []
    best_variant = report.get("best_variant") if isinstance(report.get("best_variant"), dict) else {}
    launches = kernel_launches(report)
    input_paths = str(args.get("input") or "").split(",")
    return {
        "command_returned_zero": proc.returncode == 0,
        "route_policy_ok": bool(route_policy_validation.get("ok")),
        "report_loaded": bool(report),
        "report_ok": bool(report.get("ok")),
        "policy_matches": report.get("policy") == "project_theseus_token_superposition_metal_report_v1",
        "command_matches": report.get("command") == "train-token-superposition-metal",
        "parity_for_matches": report.get("parity_for") == "train-token-superposition-cuda",
        "backend_matches": report.get("backend") == "apple_metal",
        "implementation_matches": report.get("implementation") == "rust_metal_token_superposition_readout_cli",
        "cuda_fallback_false": report.get("cuda_fallback") is False,
        "private_inputs_only": all(is_private_input(path) for path in input_paths if path),
        "project_code_excluded": args.get("include_project_code") is False
        and int(args.get("max_code_files") or 0) == 0,
        "dataset_present": int(dataset.get("train_tokens") or 0) >= int(tier["train_samples"])
        and int(dataset.get("vocab_size") or 0) == int(tier["max_vocab"])
        and int(dataset.get("hv_dim") or 0) == int(tier["hv_dim"]),
        "code_docs_excluded": int(dataset.get("code_train_docs") or 0) == 0
        and int(dataset.get("code_eval_docs") or 0) == 0,
        "baseline_present": baseline.get("id") == "baseline_ar_metal",
        "variant_present": bool(variants) and bool(best_variant),
        "variant_ids_are_metal": all(
            isinstance(row, dict) and str(row.get("id") or "").endswith("_metal")
            for row in variants
        ),
        "kernel_launches_bounded": 0 < launches <= int(tier["max_kernel_launches"]),
        "work_receipt_matches": receipt.get("accepted") is True
        and receipt.get("backend") == "apple_metal"
        and receipt.get("task_kind") == "train_token_superposition_metal_cli"
        and int(receipt.get("claimed_work_units") or 0) == launches,
        "contract_matches_surface": contract.get("matches_train_token_superposition_cli_surface") is True,
        "mirrors_mlx_command": contract.get("mirrors_command") == "train-token-superposition-mlx",
        "scheduler_routing_still_disabled": contract.get("scheduler_routing_enabled") is False
        and guard.get("does_not_route_scheduler_to_metal") is True,
        "python_mlx_bridge_not_used": contract.get("python_mlx_bridge_used") is False,
        "native_readout_subpath_declared": bool(contract.get("native_readout_subpath")),
        "promotion_still_locked": report.get("model_promotion_allowed") is False
        and get_path(report, ["promotion_decision", "promote_to_training_lane"]) is False
        and guard.get("promotion_locked_by_macos_contract") is True,
        "raw_gate_decision_retained": isinstance(report.get("raw_gate_promotion_decision"), dict)
        and bool(report.get("raw_gate_promotion_decision")),
        "parity_claim_still_locked": report.get("train_token_superposition_parity_claim_allowed") is False
        and report.get("full_cli_parity_claim_allowed") is False
        and guard.get("does_not_claim_full_kernel_parity") is True
        and guard.get("does_not_claim_training_lane_parity") is True,
        "external_inference_zero": int(report.get("external_inference_calls") or 0) == 0
        and guard.get("no_external_inference") is True,
        "teacher_disabled": report.get("teacher_used") is False and guard.get("no_teacher") is True,
        "public_training_zero": int(report.get("public_training_rows") or 0) == 0
        and guard.get("no_public_training_rows") is True,
        "public_calibration_not_run": guard.get("no_public_calibration") is True,
        "no_fallback_returns": guard.get("no_fallback_returns") is True,
    }


def validate_route_policy(policy: dict[str, Any]) -> dict[str, Any]:
    requires = policy.get("requires") if isinstance(policy.get("requires"), dict) else {}
    allowed = policy.get("allowed_input_scope") if isinstance(policy.get("allowed_input_scope"), list) else []
    checks = {
        "policy_loaded": bool(policy),
        "policy_matches": policy.get("policy")
        == "project_theseus_macos_metal_token_superposition_route_policy_v0",
        "command_matches": policy.get("command") == "train-token-superposition-metal",
        "parity_for_matches": policy.get("parity_for") == "train-token-superposition-cuda",
        "backend_matches": policy.get("backend") == "apple_metal",
        "guarded_ladder_only": policy.get("route_state") == "guarded_ladder_only",
        "production_routing_disabled": policy.get("production_scheduler_routing_enabled") is False,
        "remote_scope_unchanged": policy.get("does_not_change_hive_remote_task_scope") is True,
        "no_arbitrary_remote_execution": policy.get("no_arbitrary_remote_execution") is True,
        "private_input_scope_declared": "data/training_data/high_transfer/private_train" in allowed,
        "requires_contract_ok": requires.get("train_token_superposition_metal_contract_ok") is True,
        "requires_scheduler_locked": requires.get("scheduler_routing_enabled_in_report") is False,
        "requires_promotion_locked": requires.get("model_promotion_allowed") is False,
        "requires_parity_locked": requires.get("train_token_superposition_parity_claim_allowed") is False
        and requires.get("native_hot_loop_parity_claim_allowed") is False,
        "requires_no_cheat_locks": int(requires.get("external_inference_calls", -1)) == 0
        and requires.get("teacher_used") is False
        and int(requires.get("public_training_rows", -1)) == 0
        and requires.get("no_fallback_returns") is True
        and requires.get("public_calibration_not_run") is True,
    }
    return {"ok": all(checks.values()), "checks": checks}


def route_policy_summary(policy: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": rel(ROUTE_POLICY),
        "ok": bool(validation.get("ok")),
        "policy": policy.get("policy"),
        "command": policy.get("command"),
        "route_state": policy.get("route_state"),
        "production_scheduler_routing_enabled": policy.get("production_scheduler_routing_enabled"),
        "allowed_input_scope": policy.get("allowed_input_scope") if isinstance(policy.get("allowed_input_scope"), list) else [],
        "checks": validation.get("checks") if isinstance(validation.get("checks"), dict) else {},
    }


def planned_tier(tier: dict[str, Any]) -> dict[str, Any]:
    tier_id = str(tier["id"])
    return {
        "tier_id": tier_id,
        "ok": False,
        "config": tier,
        "report": rel(TIER_DIR / f"{tier_id}_report.json"),
        "checks": {"not_executed": False},
    }


def guardrails() -> dict[str, Any]:
    return {
        "local_only": True,
        "remote_task_submitted": False,
        "production_scheduler_routing_enabled": False,
        "model_promotion_allowed": False,
        "train_token_superposition_parity_claim_allowed": False,
        "native_hot_loop_parity_claim_allowed": False,
        "public_benchmark_training_used": False,
        "teacher_used": False,
        "external_inference_calls": 0,
        "fallback_returns_allowed": False,
        "public_calibration_run": False,
        "private_input_scope_only": True,
    }


def score_semantics() -> str:
    return (
        "Mac Rust/Metal bounded token-superposition ladder only. Tiers use private residual curriculum JSONL "
        "under data/training_data/high_transfer/private_train, exclude project-code mixing, and validate the "
        "train-token-superposition-metal report contract. This does not run public calibration, train on public "
        "benchmark data, call a teacher, use external inference, submit a remote task, enable production "
        "scheduler routing, promote a model, or claim full CUDA-equivalent parity."
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# macOS Metal Token-Superposition Ladder",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Execute: `{report.get('execute')}`",
        f"- Tier OK: `{summary.get('tier_ok_count')}/{summary.get('tier_count')}`",
        f"- Max kernel launches: `{summary.get('max_kernel_launches')}`",
        f"- Production scheduler routing enabled: `{summary.get('production_scheduler_routing_enabled')}`",
        f"- Model promotion allowed: `{summary.get('model_promotion_allowed')}`",
        f"- Token-superposition parity claim allowed: `{summary.get('train_token_superposition_parity_claim_allowed')}`",
        "",
        "## Tiers",
    ]
    for row in report.get("tiers", []):
        if not isinstance(row, dict):
            continue
        failed = [name for name, ok in (row.get("checks") or {}).items() if not ok]
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        lines.append(
            f"- `{row.get('tier_id')}`: ok=`{row.get('ok')}` launches=`{row.get('kernel_launches')}` "
            f"best=`{metrics.get('best_variant_id')}` report=`{row.get('report')}` failed=`{failed[:6]}`"
        )
    lines.extend(["", "## Boundary", "", report.get("score_semantics", "")])
    return "\n".join(lines) + "\n"


def kernel_launches(report: dict[str, Any]) -> int:
    baseline = report.get("baseline") if isinstance(report.get("baseline"), dict) else {}
    variants = report.get("variants") if isinstance(report.get("variants"), list) else []
    return int(baseline.get("kernel_launches") or 0) + sum(
        int(row.get("kernel_launches") or 0)
        for row in variants
        if isinstance(row, dict)
    )


def is_private_input(path: str) -> bool:
    candidate = resolve(path)
    try:
        candidate.resolve().relative_to(PRIVATE_INPUT_ROOT.resolve())
    except ValueError:
        return False
    return candidate.exists() and candidate.suffix == ".jsonl"


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
