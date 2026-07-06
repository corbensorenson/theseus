#!/usr/bin/env python3
"""Audit whether Code LM sharding is being used intelligently.

The chunked recovery lane exists to make long Code LM runs resumable and
crash-safe. It is not yet the ideal distributed training substrate: today each
shard owns its own bounded closure, including a fresh readout/STSes pass, then
the completed candidate manifests are merged. This audit makes that tradeoff
explicit so the watchdog and hive do not mistake "many shards ran" for a clean
train-once/distribute-many architecture.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATA_CURRICULUM = ROOT / "data" / "private_code_curriculum"
DEFAULT_SLUG = "private_pressure_private_recovery_cuda_program_loop_v6"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--out", default="reports/code_lm_shard_strategy_audit.json")
    parser.add_argument("--markdown-out", default="reports/code_lm_shard_strategy_audit.md")
    args = parser.parse_args()

    slug = args.slug
    shard_count = max(1, int(args.shard_count))
    artifacts = collect_artifacts(slug)
    shard_rows = [shard_summary(slug, index, shard_count, artifacts) for index in range(shard_count)]
    completed = [row for row in shard_rows if row["completed"]]
    completed_count = len(completed)
    total_bytes = sum(row["artifact_bytes"] for row in shard_rows)
    completed_bytes = sum(row["artifact_bytes"] for row in completed)
    avg_completed_bytes = int(completed_bytes / completed_count) if completed_count else 0
    projected_bytes = max(total_bytes, avg_completed_bytes * shard_count)
    free_bytes = shutil.disk_usage(REPORTS).free
    active = active_workers(slug)
    duplicate_active = duplicate_active_shards(active)
    lease_summary = read_lease_summary(slug, shard_count)
    repeated_training = repeated_training_detected(shard_rows, active)

    blockers: list[str] = []
    warnings: list[str] = []
    if duplicate_active:
        blockers.append("duplicate_active_shard_workers")
    if free_bytes < 2 * 1024**3:
        blockers.append("low_report_drive_free_space_under_2gb")
    if projected_bytes > max(2 * 1024**3, free_bytes // 2):
        warnings.append("projected_shard_artifacts_large_relative_to_free_space")
    if repeated_training:
        warnings.append("current_shards_repeat_training_per_shard")
    if any(row["active_leases"] > 1 for row in lease_summary["shards"]):
        blockers.append("multiple_hive_leases_for_same_shard")

    trigger_state = "RED" if blockers else ("YELLOW" if warnings else "GREEN")
    report = {
        "policy": "project_theseus_code_lm_shard_strategy_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "slug": slug,
        "shard_count": shard_count,
        "machine": socket.gethostname(),
        "summary": {
            "completed_shards": completed_count,
            "active_worker_count": len(active),
            "duplicate_active_shards": duplicate_active,
            "current_artifact_mb": round(total_bytes / 1024**2, 3),
            "projected_artifact_mb": round(projected_bytes / 1024**2, 3),
            "report_drive_free_mb": round(free_bytes / 1024**2, 3),
            "repeated_training_per_shard_detected": repeated_training,
            "hive_lease_files": lease_summary["lease_file_count"],
            "blockers": blockers,
            "warnings": warnings,
        },
        "architecture_assessment": {
            "current_role": "crash_recovery_and_candidate_eval_sharding",
            "not_a_long_term_training_substrate": True,
            "why_it_exists": [
                "preserves completed candidate/eval artifacts after crashes or timeouts",
                "bounds each Rust/SymLiquid process so the PC remains usable",
                "lets the hive distribute disjoint public/private receiver shards without public leakage",
            ],
            "current_cost": [
                "readout and STS conditioning are repeated per shard",
                "checkpoint artifacts are duplicated per shard",
                "throughput scales worse than a train-once/distribute-candidate-generation design",
            ],
            "target_architecture": "train_once_checkpoint_then_hive_distributed_candidate_generation_and_verification",
            "promotion_rule": "sharding is acceptable as a recovery envelope, but should not be credited as architecture progress until repeated training per shard is removed or amortized",
        },
        "hive_compatibility": {
            "deterministic_shard_ids": True,
            "merge_order_independent": True,
            "public_training_forbidden": True,
            "lease_manifest_present": lease_summary["lease_file_count"] > 0,
            "risk": "safe for one active local worker and mostly safe for shared-filesystem hive leases; not ideal for high-throughput multi-device training until a durable central scheduler or train-once checkpoint fanout exists",
            "leases": lease_summary,
        },
        "active_workers": active,
        "shards": shard_rows,
        "next_actions": next_actions(blockers, warnings, repeated_training),
    }
    write_json(resolve(args.out), report)
    write_markdown(resolve(args.markdown_out), report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state != "RED" else 2


def collect_artifacts(slug: str) -> list[Path]:
    files: list[Path] = []
    for root in [REPORTS, DATA_CURRICULUM]:
        if root.exists():
            files.extend(path for path in root.glob(f"*{slug}*") if path.is_file())
    return files


def shard_summary(slug: str, index: int, count: int, artifacts: list[Path]) -> dict[str, Any]:
    shard = shard_slug(slug, index, count)
    shard_artifacts = [path for path in artifacts if shard in path.name]
    closure = read_json(REPORTS / f"code_lm_closure_{shard}.json", {})
    rust = read_json(REPORTS / f"code_lm_closure_rust_{shard}.json", {})
    completed = closure.get("run_status") == "completed" or rust.get("run_status") == "completed"
    command = get_path(closure, ["phase", "command"], [])
    command_text = " ".join(command) if isinstance(command, list) else ""
    checkpoint = first_string([rust.get("checkpoint"), closure.get("checkpoint")])
    return {
        "index": index,
        "shard_slug": shard,
        "completed": bool(completed),
        "closure_run_status": closure.get("run_status"),
        "rust_run_status": rust.get("run_status"),
        "artifact_count": len(shard_artifacts),
        "artifact_mb": round(sum(path.stat().st_size for path in shard_artifacts) / 1024**2, 3),
        "artifact_bytes": sum(path.stat().st_size for path in shard_artifacts),
        "private_candidate_rows": count_jsonl_rows(resolve(first_string([rust.get("private_candidate_manifest"), closure.get("private_candidate_manifest")]))),
        "public_candidate_rows": count_jsonl_rows(resolve(first_string([rust.get("public_candidate_manifest"), closure.get("public_candidate_manifest")]))),
        "checkpoint": checkpoint,
        "uses_cuda_readout": "--use-cuda-readout" in command_text or bool(get_path(rust, ["summary", "cuda_readout_used"], False)),
        "repeats_training": "--epochs" in command_text and "--checkpoint-out" in command_text,
        "elapsed_seconds": get_path(closure, ["phase", "elapsed_seconds"], None),
    }


def active_workers(slug: str) -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match 'code_lm_chunked_recovery.py|code_lm_closure.py|train-code-lm-closure|symliquid-cli' } | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], cwd=ROOT, text=True, capture_output=True, timeout=10)
        payload = json.loads(result.stdout or "[]")
    except Exception:
        return []
    rows = payload if isinstance(payload, list) else [payload]
    active: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        command_line = str(row.get("CommandLine") or "")
        if "Get-CimInstance Win32_Process" in command_line:
            continue
        if slug not in command_line:
            continue
        shard_match = re.search(r"shard(\d+)of(\d+)", command_line)
        active.append(
            {
                "pid": row.get("ProcessId"),
                "name": row.get("Name"),
                "shard_index": int(shard_match.group(1)) if shard_match else None,
                "shard_count": int(shard_match.group(2)) if shard_match else None,
                "uses_cuda_readout": "--use-cuda-readout" in command_line,
                "command_kind": command_kind(command_line),
            }
        )
    return active


def command_kind(command_line: str) -> str:
    if "train-code-lm-closure" in command_line:
        return "rust_code_lm_closure"
    if "train-sts-parallel-decoder" in command_line:
        return "rust_sts_parallel_decoder"
    if "code_lm_closure.py" in command_line:
        return "python_closure_wrapper"
    if "code_lm_chunked_recovery.py" in command_line:
        return "python_chunked_driver"
    return "unknown"


def duplicate_active_shards(active: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(row.get("shard_index"))
        for row in active
        if row.get("shard_index") is not None and row.get("command_kind") in {"rust_code_lm_closure", "python_closure_wrapper"}
    )
    return {key: value for key, value in counts.items() if value > 2}


def read_lease_summary(slug: str, shard_count: int) -> dict[str, Any]:
    lease_dir = REPORTS / "code_lm_chunked_recovery_leases"
    rows: list[dict[str, Any]] = []
    file_count = 0
    for index in range(shard_count):
        shard = shard_slug(slug, index, shard_count)
        leases = sorted(lease_dir.glob(f"{shard}*.json")) if lease_dir.exists() else []
        file_count += len(leases)
        active = 0
        statuses: list[str] = []
        for path in leases:
            payload = read_json(path, {})
            status = str(payload.get("status") or "unknown")
            statuses.append(status)
            if status in {"claimed", "running"}:
                active += 1
        rows.append({"index": index, "shard_slug": shard, "lease_files": len(leases), "active_leases": active, "statuses": statuses})
    return {"lease_file_count": file_count, "shards": rows}


def repeated_training_detected(shards: list[dict[str, Any]], active: list[dict[str, Any]]) -> bool:
    if any(row.get("repeats_training") for row in shards):
        return True
    return any(row.get("command_kind") == "rust_code_lm_closure" for row in active)


def next_actions(blockers: list[str], warnings: list[str], repeated_training: bool) -> list[str]:
    actions: list[str] = []
    if blockers:
        actions.append("Do not launch more shard work until blockers are cleared.")
    if repeated_training:
        actions.append("Keep the current run only as bounded recovery evidence; next architecture work should split training from candidate-generation shards.")
    if "projected_shard_artifacts_large_relative_to_free_space" in warnings:
        actions.append("Archive or compact completed diagnostic artifacts before starting another full shard cycle.")
    if not actions:
        actions.append("Current sharding envelope is acceptable for recovery; continue merge/gate path.")
    return actions


def shard_slug(slug: str, index: int, count: int) -> str:
    return f"{slug}_shard{index:02d}of{count:02d}"


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return payload if isinstance(payload, dict) else default
    except Exception:
        return default
    return default


def count_jsonl_rows(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def get_path(payload: Any, path: list[str], default: Any = None) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def first_string(values: list[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Code LM Shard Strategy Audit",
        "",
        f"- Status: **{report['trigger_state']}**",
        f"- Completed shards: `{summary['completed_shards']}/{report['shard_count']}`",
        f"- Current shard artifact size: `{summary['current_artifact_mb']} MB`",
        f"- Projected full-run artifact size: `{summary['projected_artifact_mb']} MB`",
        f"- Free report-drive space: `{summary['report_drive_free_mb']} MB`",
        f"- Repeats training per shard: `{summary['repeated_training_per_shard_detected']}`",
        "",
        "## Assessment",
        "",
        "Shards are acceptable as a crash-recovery envelope, but they are not the target long-term training substrate.",
        "The target architecture is train-once checkpoint fanout into distributed candidate generation and verification.",
        "",
        "## Next Actions",
        "",
    ]
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
