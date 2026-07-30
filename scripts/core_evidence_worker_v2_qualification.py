#!/usr/bin/env python3
"""Run the frozen Worker v2 on a fresh target-blind qualification cohort.

This runner reads only the public task manifest. Authoritative target commits
remain in the evaluator-only manifest and are not opened until every candidate
has been sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import core_evidence_worker_v2_development as development


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MANIFEST = (
    ROOT / "configs" / "core_evidence_worker_v2_qualification_public.json"
)
FREEZE_MANIFEST = (
    ROOT / "configs" / "core_evidence_worker_v2_qualification_freeze.json"
)
WORKER_CONFIG = ROOT / "configs" / "core_evidence_worker_v2_development.json"
WORKER = ROOT / "scripts" / "core_evidence_worker_v2.py"
EVALUATOR = ROOT / "scripts" / "core_evidence_worker_v2_evaluator.py"
OUT = ROOT / "reports" / "core_evidence_worker_v2_qualification_candidates.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", default=str(PUBLIC_MANIFEST))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    report = run(Path(args.public_manifest))
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "attempted": len(report["tasks"]),
        "sealed": sum(
            row.get("sealed_before_target_open") is True
            for row in report["tasks"]
        ),
        "fault_count": len(report["faults"]),
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def run(public_manifest_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = read_json(public_manifest_path)
    freeze = read_json(FREEZE_MANIFEST)
    validate_manifest(manifest, freeze, public_manifest_path)
    config = read_json(WORKER_CONFIG)
    rows = []
    faults = []
    for task in manifest["tasks"]:
        try:
            rows.append(development.run_task(task, config))
        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:
            faults.append({
                "opaque_task_id": task.get("opaque_task_id"),
                "fault": f"{type(exc).__name__}:{exc}",
            })
    report = {
        "policy": "project_theseus_worker_v2_fresh_qualification_candidates_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "scope": "fresh_source_disjoint_qualification_candidates_not_yet_evaluated",
        "qualification_public_manifest": relative(public_manifest_path),
        "qualification_public_manifest_sha256": sha256_file(public_manifest_path),
        "qualification_freeze_manifest": relative(FREEZE_MANIFEST),
        "qualification_freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST),
        "source": {
            "worker_sha256": sha256_file(WORKER),
            "worker_config_sha256": sha256_file(WORKER_CONFIG),
            "qualification_runner_sha256": sha256_file(Path(__file__)),
            "evaluator_sha256": sha256_file(EVALUATOR),
        },
        "tasks": rows,
        "faults": faults,
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "targets_opened_to_worker": 0,
            "targets_opened_before_all_candidates_sealed": 0,
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
            "Candidate generation alone supports no competence claim. "
            "The independent evaluator must open targets only after every "
            "candidate seal validates."
        ),
    }
    report["report_payload_sha256"] = stable_hash({
        key: value for key, value in report.items()
        if key not in {"created_utc", "runtime", "report_payload_sha256"}
    })
    return report


def validate_manifest(
    manifest: dict[str, Any],
    freeze_manifest: dict[str, Any],
    path: Path,
) -> None:
    if manifest.get("policy") != (
        "project_theseus_worker_v2_fresh_qualification_public_v1"
    ):
        raise ValueError("unexpected qualification manifest policy")
    if freeze_manifest.get("policy") != (
        "project_theseus_worker_v2_fresh_qualification_freeze_v1"
    ):
        raise ValueError("unexpected qualification freeze policy")
    freeze = mapping(freeze_manifest.get("source_identities"))
    expected = {
        "worker_sha256": sha256_file(WORKER),
        "worker_config_sha256": sha256_file(WORKER_CONFIG),
        "qualification_runner_sha256": sha256_file(Path(__file__)),
        "evaluator_sha256": sha256_file(EVALUATOR),
    }
    if any(freeze.get(key) != value for key, value in expected.items()):
        raise ValueError("qualification source freeze mismatch")
    floor = mapping(manifest.get("competence_floor"))
    if floor != {
        "minimum_attempted_tasks": 3,
        "minimum_task_family_rate": 0.34,
        "minimum_useful_rate": 0.5,
    }:
        raise ValueError("competence floor changed")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("qualification requires exactly three frozen tasks")
    opaque_ids = set()
    parents = set()
    for task in tasks:
        if set(task) != {
            "opaque_task_id",
            "family",
            "natural_request",
            "parent_source_commit",
            "allowed_runtime_context",
            "authority_grant",
        }:
            raise ValueError("unexpected public task field")
        opaque_ids.add(str(task["opaque_task_id"]))
        parents.add(str(task["parent_source_commit"]))
        audit = request_adequacy(str(task["natural_request"]))
        if not audit["passed"]:
            raise ValueError(
                f"request adequacy failed:{task['opaque_task_id']}:{audit['faults']}"
            )
    if len(opaque_ids) != len(tasks) or len(parents) != len(tasks):
        raise ValueError("qualification tasks are not source-disjoint")
    if sha256_file(path) != freeze_manifest.get("public_manifest_sha256"):
        raise ValueError("public qualification manifest mutated after freeze")


def request_adequacy(request: str) -> dict[str, Any]:
    words = re.findall(r"[A-Za-z0-9_./`'-]+", request)
    observable = re.findall(
        r"\b(?:must|only|when|preserve|reject|raise|return|remain|keep|"
        r"without|before|after|strictly|exact)\b",
        request,
        flags=re.IGNORECASE,
    )
    repository_identifiers = re.findall(
        r"(?:[A-Za-z0-9_-]+/)+[A-Za-z0-9_.-]+|"
        r"`[A-Za-z_][A-Za-z0-9_]*`|"
        r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b",
        request,
    )
    faults = []
    if len(words) < 24:
        faults.append("fewer_than_24_words")
    if len(observable) < 3:
        faults.append("fewer_than_3_observable_acceptance_terms")
    if len(repository_identifiers) < 2:
        faults.append("insufficient_repository_surface_identity")
    if not re.search(r"\btest|verification|assert", request, re.IGNORECASE):
        faults.append("verification_expectation_missing")
    return {
        "passed": not faults,
        "word_count": len(words),
        "observable_acceptance_term_count": len(observable),
        "repository_identifier_count": len(repository_identifiers),
        "verification_expectation_present": not any(
            fault == "verification_expectation_missing" for fault in faults
        ),
        "faults": faults,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
