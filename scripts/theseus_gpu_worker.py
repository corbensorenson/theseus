#!/usr/bin/env python3
"""Resident GPU worker for bounded Theseus hot-path jobs.

The worker is intentionally conservative: it warms and verifies the CUDA node,
checks resource policy before code-heavy work, executes only allowlisted local
commands, and writes a status ledger. It is a service wrapper, not a promotion
shortcut; public calibration and public benchmark training remain blocked by
the normal gates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_QUEUE = REPORTS / "theseus_gpu_worker_queue.jsonl"
DEFAULT_STATUS = REPORTS / "theseus_gpu_worker_status.json"
DEFAULT_LEDGER = REPORTS / "theseus_gpu_worker_ledger.jsonl"

ALLOWLIST = {
    "scripts/code_lm_chunked_recovery.py",
    "scripts/sts_causal_decoder_ablation.py",
    "scripts/symliquid_state_engine.py",
    "scripts/agent_lane_transfer_gate.py",
    "scripts/maturity_integrity_audit.py",
    "scripts/a_plus_operating_scorecard.py",
    "scripts/pufferlib4_rl_lane.py",
    "scripts/windows_cuda_doctor.py",
    "scripts/resource_aware_execution_policy.py",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--enqueue-code-lm-v5", action="store_true")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE.relative_to(ROOT)))
    parser.add_argument("--status-out", default=str(DEFAULT_STATUS.relative_to(ROOT)))
    parser.add_argument("--ledger-out", default=str(DEFAULT_LEDGER.relative_to(ROOT)))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-idle-seconds", type=int, default=0)
    args = parser.parse_args()

    queue = resolve(args.queue)
    status_out = resolve(args.status_out)
    ledger_out = resolve(args.ledger_out)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if args.enqueue_code_lm_v5:
        append_job(queue, code_lm_v5_job())

    if not args.serve and not args.run_once:
        status = status_payload("PLANNED", "no_execution_requested", queue)
        write_json(status_out, status)
        print(json.dumps(status, indent=2))
        return 0

    warm = run_command(
        [
            sys.executable,
            "scripts/windows_cuda_doctor.py",
            "--refresh",
            "--out",
            "reports/windows_cuda_doctor.json",
            "--markdown-out",
            "reports/windows_cuda_doctor.md",
        ],
        timeout=120,
    )
    append_ledger(ledger_out, {"event": "cuda_doctor", "result": warm})

    idle_started = time.monotonic()
    while True:
        job = next_pending_job(queue)
        if not job:
            status = status_payload("IDLE", "queue_empty", queue)
            write_json(status_out, status)
            if args.run_once:
                print(json.dumps(status, indent=2))
                return 0
            if args.max_idle_seconds and time.monotonic() - idle_started >= args.max_idle_seconds:
                return 0
            time.sleep(max(1, args.poll_seconds))
            continue

        idle_started = time.monotonic()
        command = normalize_command(job.get("command") or [])
        allowed, reason = command_allowed(command)
        if not allowed:
            mark_job(queue, job["job_id"], "REJECTED", reason)
            append_ledger(ledger_out, {"event": "job_rejected", "job": job, "reason": reason})
            continue

        if is_code_lm_job(command):
            policy = resource_policy()
            budget = ((policy.get("recommended_code_lm_budget") or {}) if isinstance(policy, dict) else {})
            if not (budget.get("start_new_code_closure") or budget.get("start_new_chunked_code_closure")):
                status = status_payload("DEFERRED", "resource_policy_deferred_code_lm", queue)
                status["resource_policy"] = policy
                write_json(status_out, status)
                append_ledger(ledger_out, {"event": "job_deferred", "job": job, "resource_policy": policy})
                if args.run_once:
                    print(json.dumps(status, indent=2))
                    return 0
                time.sleep(max(1, args.poll_seconds))
                continue

        mark_job(queue, job["job_id"], "RUNNING", "started")
        status = status_payload("RUNNING", "executing_job", queue)
        status["job"] = job
        write_json(status_out, status)
        started = time.monotonic()
        result = run_command(command, timeout=int(job.get("timeout_seconds") or 21600))
        result["runtime_ms"] = int((time.monotonic() - started) * 1000)
        final_state = "DONE" if result.get("returncode") == 0 else "FAILED"
        mark_job(queue, job["job_id"], final_state, result.get("stderr_tail") or result.get("stdout_tail") or final_state)
        append_ledger(ledger_out, {"event": "job_completed", "job": job, "result": result, "final_state": final_state})
        status = status_payload(final_state, "job_completed", queue)
        status["last_result"] = result
        write_json(status_out, status)
        if args.run_once:
            print(json.dumps(status, indent=2))
            return 0 if final_state == "DONE" else 2


def code_lm_v5_job() -> dict[str, Any]:
    return {
        "job_id": f"code_lm_cuda_decode_v5_{int(time.time())}",
        "created_utc": now(),
        "status": "PENDING",
        "kind": "code_lm_cuda_decode_v5",
        "timeout_seconds": 21600,
        "command": [
            sys.executable,
            "scripts/code_lm_chunked_recovery.py",
            "--execute",
            "--slug",
            "private_pressure_private_recovery_cuda_decode_v5",
            "--shard-count",
            "16",
            "--continuous",
            "--max-wall-seconds",
            "21600",
            "--out",
            "reports/code_lm_chunked_recovery.json",
            "--markdown-out",
            "reports/code_lm_chunked_recovery.md",
        ],
        "rules": {
            "public_calibration": "locked_by_decoder_and_transfer_gates",
            "cuda_required": "--use-cuda-readout added by chunked driver",
            "monolithic_closure": "forbidden",
        },
    }


def append_job(queue: Path, job: dict[str, Any]) -> None:
    queue.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(queue)
    if any(row.get("job_id") == job["job_id"] for row in rows):
        return
    with queue.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(job, sort_keys=True) + "\n")


def next_pending_job(queue: Path) -> dict[str, Any] | None:
    for row in read_jsonl(queue):
        if row.get("status") == "PENDING":
            return row
    return None


def mark_job(queue: Path, job_id: str, status: str, detail: str) -> None:
    rows = read_jsonl(queue)
    for row in rows:
        if row.get("job_id") == job_id:
            row["status"] = status
            row["updated_utc"] = now()
            row["status_detail"] = detail[:1000]
            break
    write_jsonl(queue, rows)


def command_allowed(command: list[str]) -> tuple[bool, str]:
    if len(command) < 2:
        return False, "command_too_short"
    script = command[1].replace("\\", "/")
    if script not in ALLOWLIST:
        return False, f"script_not_allowlisted:{script}"
    text = " ".join(command).lower()
    forbidden = [
        "--allow-public-training",
        "--teacher-apply",
        "public_gateway",
        "bulk_download",
        "download-rom",
    ]
    for token in forbidden:
        if token in text:
            return False, f"forbidden_token:{token}"
    return True, "allowed"


def is_code_lm_job(command: list[str]) -> bool:
    text = " ".join(command)
    return "code_lm_chunked_recovery.py" in text or "code_lm_closure.py" in text


def resource_policy() -> dict[str, Any]:
    result = run_command(
        [
            sys.executable,
            "scripts/resource_aware_execution_policy.py",
            "--out",
            "reports/resource_aware_execution_policy.json",
            "--markdown-out",
            "reports/resource_aware_execution_policy.md",
        ],
        timeout=90,
    )
    path = REPORTS / "resource_aware_execution_policy.json"
    payload = read_json(path)
    payload["_worker_result"] = result
    return payload


def run_command(command: list[str], timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("THESEUS_CODE_LM_CUDA_DECODE_LOGITS", "1")
    env.setdefault("THESEUS_CODE_LM_CUDA_TOP_P", "1.0")
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "runtime_ms": int((time.monotonic() - started) * 1000),
            "stdout_tail": tail(result.stdout),
            "stderr_tail": tail(result.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "runtime_ms": int((time.monotonic() - started) * 1000),
            "stdout_tail": tail(exc.stdout or ""),
            "stderr_tail": tail(exc.stderr or ""),
            "timeout": True,
        }


def normalize_command(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def status_payload(state: str, reason: str, queue: Path) -> dict[str, Any]:
    return {
        "policy": "project_theseus_resident_gpu_worker_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if state in {"IDLE", "DONE"} else "YELLOW" if state in {"RUNNING", "DEFERRED", "PLANNED"} else "RED",
        "state": state,
        "reason": reason,
        "queue": rel(queue),
        "pending_jobs": sum(1 for row in read_jsonl(queue) if row.get("status") == "PENDING"),
        "rules": {
            "allowlist_only": sorted(ALLOWLIST),
            "public_benchmark_training": "forbidden",
            "teacher_apply": "forbidden",
            "duplicate_heavy_workers": "deferred_to_resource_policy",
        },
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_utc": now(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def tail(text: str, limit: int = 4000) -> str:
    return (text or "")[-limit:]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
