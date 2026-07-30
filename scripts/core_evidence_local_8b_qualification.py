#!/usr/bin/env python3
"""Generate and seal the prospective local-8B qualification candidates.

This process reads only the public task manifest and the pre-generation freeze.
It never imports or opens the evaluator manifest or hidden functional tests.
"""

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
PUBLIC = ROOT / "configs" / "core_evidence_local_8b_qualification_public.json"
FREEZE = ROOT / "configs" / "core_evidence_local_8b_qualification_freeze.json"
WORKER = ROOT / "scripts" / "core_evidence_worker_v2.py"
WORKER_CONFIG = ROOT / "configs" / "core_evidence_local_8b_worker.json"
DEVELOPMENT_RUNNER = ROOT / "scripts" / "core_evidence_worker_v2_development.py"
OUT = ROOT / "reports" / "core_evidence_local_8b_qualification_candidates.json"
POLICY = "project_theseus_local_8b_functional_qualification_candidates_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", default=str(PUBLIC))
    parser.add_argument("--freeze", default=str(FREEZE))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    report = run(Path(args.public_manifest), Path(args.freeze))
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "attempted": report["denominators"]["attempted"],
        "sealed": report["denominators"]["sealed"],
        "fault_count": len(report["faults"]),
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def run(public_path: Path, freeze_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    public = read_json(public_path)
    freeze = read_json(freeze_path)
    validate_frozen_inputs(public, freeze, public_path)
    config = read_json(WORKER_CONFIG)
    rows: list[dict[str, Any]] = []
    faults: list[dict[str, str]] = []
    for task in public["tasks"]:
        try:
            rows.append(development.run_task(task, config))
        except Exception as exc:  # preserve a complete attempted denominator
            faults.append({
                "opaque_task_id": str(task.get("opaque_task_id") or ""),
                "fault": f"{type(exc).__name__}:{exc}",
            })
    attempted = len(rows) + len(faults)
    sealed = sum(
        int(row.get("sealed_before_target_open") is True) for row in rows
    )
    report = {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": (
            "GREEN"
            if attempted == len(public["tasks"])
            and sealed == len(public["tasks"])
            and not faults
            else "RED"
        ),
        "scope": "prospective_target_blind_candidate_generation_only",
        "cohort_id": public["cohort_id"],
        "public_manifest_sha256": sha256_file(public_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "source_identities": current_source_identities(),
        "tasks": rows,
        "faults": faults,
        "denominators": {
            "planned": len(public["tasks"]),
            "attempted": attempted,
            "sealed": sealed,
            "infrastructure_failed": len(faults),
            "skipped": len(public["tasks"]) - attempted,
        },
        "counters": {
            "evaluator_manifest_opened": 0,
            "hidden_test_files_opened": 0,
            "target_commits_opened": 0,
            "target_patches_opened": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "user_facing_effects": 0,
            "learned_generation_credit": sum(
                int(row.get("learned_generation_credit") or 0)
                for row in rows
            ),
        },
        "runtime": {
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "maximum_inference": (
            "This report proves only that target-blind candidates were generated "
            "and sealed. Competence requires the frozen independent evaluator."
        ),
    }
    report["report_payload_sha256"] = stable_hash({
        key: value for key, value in report.items()
        if key not in {"created_utc", "runtime", "report_payload_sha256"}
    })
    return report


def validate_frozen_inputs(
    public: dict[str, Any],
    freeze: dict[str, Any],
    public_path: Path,
) -> None:
    if public.get("policy") != (
        "project_theseus_local_8b_functional_qualification_public_v1"
    ):
        raise ValueError("unexpected_public_policy")
    if freeze.get("policy") != (
        "project_theseus_local_8b_functional_qualification_freeze_v1"
    ):
        raise ValueError("unexpected_freeze_policy")
    if public.get("cohort_id") != freeze.get("cohort_id"):
        raise ValueError("cohort_identity_mismatch")
    if sha256_file(public_path) != freeze.get("public_manifest_sha256"):
        raise ValueError("public_manifest_mutated_after_freeze")
    if current_source_identities() != freeze.get("candidate_source_identities"):
        raise ValueError("candidate_source_mutated_after_freeze")
    if public.get("competence_floor") != {
        "minimum_attempted_tasks": 3,
        "minimum_useful_rate": 0.5,
        "minimum_weakest_family_rate": 0.34,
    }:
        raise ValueError("competence_floor_changed")
    tasks = public.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("qualification_requires_three_tasks")
    ids = [str(task.get("opaque_task_id") or "") for task in tasks]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("task_identity_invalid")
    expected_fields = {
        "opaque_task_id",
        "family",
        "natural_request",
        "parent_source_commit",
        "allowed_runtime_context",
        "authority_grant",
    }
    for task in tasks:
        if set(task) != expected_fields:
            raise ValueError("unexpected_public_task_field")
        if len(str(task["natural_request"]).split()) < 24:
            raise ValueError("natural_request_inadequate")
        if task["authority_grant"] != "temporary_effect_with_exact_rollback":
            raise ValueError("authority_grant_changed")


def current_source_identities() -> dict[str, str]:
    return {
        "worker_sha256": sha256_file(WORKER),
        "worker_config_sha256": sha256_file(WORKER_CONFIG),
        "development_runner_sha256": sha256_file(DEVELOPMENT_RUNNER),
        "qualification_runner_sha256": sha256_file(Path(__file__)),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object:{path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
