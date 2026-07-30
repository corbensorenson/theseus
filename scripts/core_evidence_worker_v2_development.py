#!/usr/bin/env python3
"""Run Worker v2 on consumed development tasks only.

This is an engineering diagnostic, never blind evidence. It intentionally
does not inspect or score target commits; the separate evaluator owns that
after a candidate is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "core_evidence_local_8b_worker.json"
E0 = ROOT / "reports" / "core_evidence_e0_preregistration.json"
OUT = ROOT / "reports" / "core_evidence_worker_v2_development.json"
EVENTS = ROOT / "runtime" / "core_evidence_worker_v2_development_events.jsonl"
MLX_PYTHON = Path("/Users/corbensorenson/miniforge3/bin/python")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=1)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    report = run(args.task_index, args.task_limit)
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "attempted": len(report["tasks"]),
        "patch_count": sum(bool(row.get("patch_unified_diff")) for row in report["tasks"]),
        "verified_count": sum(row.get("candidate_verification_green") is True for row in report["tasks"]),
        "fault_count": len(report["faults"]),
    }, indent=2, sort_keys=True))
    return 0 if not report["faults"] else 2


def run(task_index: int, task_limit: int) -> dict[str, Any]:
    started = time.perf_counter()
    if EVENTS.exists():
        EVENTS.unlink()
    e0 = read_json(E0)
    config = read_json(CONFIG)
    public_tasks = [
        row for row in dicts(dict_value(e0.get("public_packet")).get("tasks"))
        if row.get("partition") == "development"
        and row.get("denominator") == "D1_DEVELOPMENT"
    ]
    selected = public_tasks[task_index:task_index + task_limit]
    rows = []
    faults = []
    for task in selected:
        try:
            rows.append(run_task(task, config))
        except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            faults.append({
                "opaque_task_id": task.get("opaque_task_id"),
                "fault": f"{type(exc).__name__}:{exc}",
            })
    report = {
        "policy": "project_theseus_worker_v2_consumed_development_diagnostic_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "scope": "consumed_development_only_not_blind_evidence",
        "source": {
            "commit": git("rev-parse", "HEAD"),
            "worker_sha256": sha256_file(ROOT / "scripts" / "core_evidence_worker_v2.py"),
            "config_sha256": sha256_file(CONFIG),
            "E0_preregistration_sha256": e0.get("preregistration_sha256"),
        },
        "tasks": rows,
        "faults": faults,
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "learned_generation_credit": sum(
                int(row.get("learned_generation_credit") or 0) for row in rows
            ),
            "user_facing_effects": 0,
        },
        "runtime": {
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "maximum_inference": (
            "This report can diagnose Worker v2 mechanics on consumed development "
            "tasks. It cannot support capability, generalization, or E2 claims."
        ),
    }
    report["report_payload_sha256"] = stable_hash({
        key: value for key, value in report.items()
        if key not in {"created_utc", "runtime", "report_payload_sha256"}
    })
    return report


def run_task(task: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    visible = {
        key: task[key] for key in (
            "natural_request", "parent_source_commit", "allowed_runtime_context",
            "authority_grant",
        )
    }
    with tempfile.TemporaryDirectory(prefix="theseus-worker-v2-dev-") as tmp:
        root = Path(tmp)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        archive = root / "parent.tar"
        worker_input = root / "input.json"
        worker_output = root / "output.json"
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", str(visible["parent_source_commit"])],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        with tarfile.open(archive) as bundle:
            safe_extract(bundle, snapshot)
        if (snapshot / ".git").exists():
            raise ValueError("git metadata entered worker snapshot")
        worker_input.write_text(
            json.dumps(visible, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        started_utc = now()
        started = time.perf_counter()
        process = subprocess.run(
            [
                str(MLX_PYTHON),
                str(ROOT / "scripts" / "core_evidence_worker_v2.py"),
                "--input", str(worker_input),
                "--snapshot-root", str(snapshot),
                "--out", str(worker_output),
                "--config", str(CONFIG),
                "--events-out", str(EVENTS),
            ],
            cwd=snapshot,
            capture_output=True,
            text=True,
            timeout=1800,
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "NO_PROXY": "*",
                "no_proxy": "*",
                "PYTHONHASHSEED": "0",
            },
            check=False,
        )
        finished_utc = now()
        worker_wall_ms = round((time.perf_counter() - started) * 1000.0, 3)
        if process.returncode != 0 or not worker_output.is_file():
            raise ValueError(
                f"worker_failed:returncode={process.returncode}:stderr={process.stderr[-1000:]}"
            )
        candidate = read_json(worker_output)
        receipts = dicts(candidate.get("verification_receipts"))
        seal = {
            "candidate_output_sha256": sha256_file(worker_output),
            "worker_input_sha256": sha256_file(worker_input),
            "parent_archive_sha256": sha256_file(archive),
            "worker_source_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_worker_v2.py"
            ),
            "config_sha256": sha256_file(CONFIG),
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "worker_wall_ms": worker_wall_ms,
            "target_opened_before_seal": False,
        }
        return {
            "opaque_task_id": task.get("opaque_task_id"),
            "candidate_output": candidate,
            "candidate_seal": seal,
            "candidate_output_sha256": seal["candidate_output_sha256"],
            "worker_input_sha256": seal["worker_input_sha256"],
            "parent_archive_sha256": seal["parent_archive_sha256"],
            "sealed_before_target_open": True,
            "patch_unified_diff": candidate.get("patch_unified_diff"),
            "proposed_paths": candidate.get("proposed_paths"),
            "candidate_verification_green": bool(receipts and receipts[-1].get("passed") is True),
            "verification_receipts": receipts,
            "effect_inventory": candidate.get("effect_inventory"),
            "action_summary": candidate.get("action_summary"),
            "repair_attempts": candidate.get("repair_attempts"),
            "format_repairs": candidate.get("format_repairs"),
            "terminal_reason": candidate.get("terminal_reason"),
            "residuals": candidate.get("residuals"),
            "learned_generation_credit": candidate.get("learned_generation_credit"),
            "local_model_inference_calls": candidate.get("local_model_inference_calls"),
            "model_identity": candidate.get("model_identity"),
            "external_inference_calls": candidate.get("external_inference_calls"),
            "teacher_calls": candidate.get("teacher_calls"),
            "public_calibration_cases_consumed": candidate.get("public_calibration_cases_consumed"),
            "D2_cases_consumed": candidate.get("D2_cases_consumed"),
            "worker_stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
            "worker_stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
        }


def safe_extract(bundle: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in bundle.getmembers():
        resolved = (destination / member.name).resolve()
        if not resolved.is_relative_to(root) or member.issym() or member.islnk():
            raise ValueError("unsafe archive member")
    bundle.extractall(destination, filter="data")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
