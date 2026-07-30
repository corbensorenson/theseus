#!/usr/bin/env python3
"""Run one preregistered local model on the consumed development cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import core_evidence_worker_v2_development as development


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "core_evidence_local_model_bakeoff.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run(args.candidate_id)
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "candidate_id": report["candidate_id"],
        "trigger_state": report["trigger_state"],
        "attempted": report["denominators"]["attempted"],
        "sealed": report["denominators"]["sealed"],
        "fault_count": len(report["faults"]),
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def run(candidate_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    plan = read_json(PLAN)
    candidate = next(
        (
            row for row in plan["candidates"]
            if row["candidate_id"] == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError("candidate_not_preregistered")
    task_manifest_path = ROOT / plan["task_manifest"]
    task_manifest = read_json(task_manifest_path)
    config_path = ROOT / candidate["config"]
    config = read_json(config_path)
    if (
        config["model"]["repo_id"] != candidate["repo_id"]
        or config["model"]["revision"] != candidate["revision"]
    ):
        raise ValueError("candidate_model_identity_mismatch")
    runtime_python = ROOT / plan["runtime_python"]
    events_path = (
        ROOT / "runtime" / f"core_evidence_model_bakeoff_{candidate_id}.jsonl"
    )
    if events_path.exists():
        events_path.unlink()
    rows: list[dict[str, Any]] = []
    faults: list[dict[str, str]] = []
    for task in task_manifest["tasks"]:
        try:
            rows.append(development.run_task(
                task,
                config,
                config_path=config_path,
                events_path=events_path,
                mlx_python=runtime_python,
            ))
        except Exception as exc:
            faults.append({
                "opaque_task_id": str(task.get("opaque_task_id") or ""),
                "fault": f"{type(exc).__name__}:{exc}",
            })
    attempted = len(rows) + len(faults)
    sealed = sum(
        int(row.get("sealed_before_target_open") is True) for row in rows
    )
    report = {
        "policy": "project_theseus_local_model_bakeoff_candidates_v1",
        "created_utc": now(),
        "trigger_state": (
            "GREEN"
            if attempted == len(task_manifest["tasks"])
            and sealed == len(task_manifest["tasks"])
            and not faults else "RED"
        ),
        "scope": "consumed_development_model_selection_no_capability_claim",
        "candidate_id": candidate_id,
        "model_identity": {
            "repo_id": candidate["repo_id"],
            "revision": candidate["revision"],
            "config_sha256": sha256_file(config_path),
        },
        "source_identities": {
            "plan_sha256": sha256_file(PLAN),
            "task_manifest_sha256": sha256_file(task_manifest_path),
            "worker_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_worker_v2.py"
            ),
            "development_runner_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_worker_v2_development.py"
            ),
            "bakeoff_runner_sha256": sha256_file(Path(__file__)),
        },
        "tasks": rows,
        "faults": faults,
        "denominators": {
            "planned": len(task_manifest["tasks"]),
            "attempted": attempted,
            "sealed": sealed,
            "infrastructure_failed": len(faults),
            "skipped": len(task_manifest["tasks"]) - attempted,
        },
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "user_facing_effects": 0,
        },
        "runtime": {
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "maximum_inference": (
            "This consumed-task run supports local model selection only. It "
            "cannot support competence, generalization, or subsystem claims."
        ),
    }
    return report


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object:{path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
