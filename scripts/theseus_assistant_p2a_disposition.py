#!/usr/bin/env python3
"""Independently classify a consumed P2A instrument-adequacy run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p2a_terminal_disposition_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = build_disposition(resolve(args.candidate_report), resolve(args.evaluation_report), resolve(args.task))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(compact_summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_disposition(candidate_path: Path, evaluation_path: Path, task_path: Path) -> dict[str, Any]:
    candidate = read_json(candidate_path)
    evaluation = read_json(evaluation_path)
    task = read_json(task_path)
    faults: list[str] = []

    if candidate.get("task_sha256") != sha256_file(task_path):
        faults.append("candidate_task_digest_mismatch")
    if evaluation.get("candidate_report_sha256") != sha256_file(candidate_path):
        faults.append("evaluation_candidate_digest_mismatch")
    if candidate.get("trigger_state") not in {"GREEN", "YELLOW"}:
        faults.append("candidate_run_invalid")
    if evaluation.get("trigger_state") != "GREEN":
        faults.append("evaluation_invalid")

    denominators = mapping(candidate.get("denominators"))
    pair = mapping(candidate.get("matched_pair"))
    attempts = dicts(candidate.get("attempts"))
    runtime_receipts: list[dict[str, Any]] = []
    for attempt in attempts:
        arm = str(attempt.get("arm_id") or "")
        for call in dicts(attempt.get("runtime_calls")):
            runtime_path = resolve(str(call.get("report_path") or ""))
            if not runtime_path.is_file():
                faults.append(f"runtime_report_missing:{arm}:{call.get('call_number')}")
                continue
            observed_sha = sha256_file(runtime_path)
            if observed_sha != str(call.get("report_sha256") or ""):
                faults.append(f"runtime_report_digest_mismatch:{arm}:{call.get('call_number')}")
            runtime = read_json(runtime_path)
            failed = [
                {"name": str(row.get("name") or ""), "severity": str(row.get("severity") or "")}
                for row in dicts(runtime.get("gates")) if row.get("passed") is False
            ]
            runtime_receipts.append({
                "arm_id": arm,
                "call_number": int(call.get("call_number") or 0),
                "path": relative(runtime_path),
                "sha256": observed_sha,
                "trigger_state": runtime.get("trigger_state"),
                "route_integrity_ready": mapping(runtime.get("route_integrity")).get("ready") is True,
                "failed_gates": failed,
            })

    parse_faults = {
        str(row.get("arm_id") or ""): {
            "first_call": sorted({
                fault
                for history in dicts(row.get("repair_fault_history"))
                for fault in strings(history.get("parse_faults"))
            }),
            "final_call": sorted(set(strings(row.get("parse_faults")))),
        }
        for row in attempts
    }
    namespace = namespace_audit(task)
    evaluated = int(mapping(evaluation.get("denominators")).get("correctness_evaluated_candidates") or 0)
    parseable = int(denominators.get("parseable_candidates") or 0)
    expected_shape = (
        pair.get("ready") is True
        and int(denominators.get("arms") or 0) == 2
        and int(denominators.get("model_loads") or 0) == 1
        and int(denominators.get("model_calls") or 0) <= 4
        and len(attempts) == 2
    )
    if not expected_shape:
        faults.append("matched_instrument_shape_invalid")

    inadequate = not faults and parseable == 0 and evaluated == 0
    disposition = (
        "P2A_TERMINAL_INCONCLUSIVE_INSTRUMENT_AND_TASK_NAMESPACE"
        if inadequate and namespace["repo_relative_path_ambiguity_present"]
        else "P2A_TERMINAL_INCONCLUSIVE_EXACT_INSTRUMENT"
        if inadequate
        else "P2A_TERMINAL_REVIEW_REQUIRED"
    )
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "scientific_status": "INCONCLUSIVE_IMPLEMENTATION" if inadequate else "UNCLASSIFIED",
        "terminal_disposition": disposition,
        "source_identities": {
            "candidate_report": source_identity(candidate_path),
            "evaluation_report": source_identity(evaluation_path),
            "task_manifest": source_identity(task_path),
            "instrument_sha256": candidate.get("instrument_sha256"),
            "runtime_reports": runtime_receipts,
        },
        "recomputed_checks": {
            "matched_pair_ready": pair.get("ready") is True,
            "one_persistent_model_load": int(denominators.get("model_loads") or 0) == 1,
            "model_calls_within_four_call_pair_budget": int(denominators.get("model_calls") or 0) <= 4,
            "parseable_candidates": parseable,
            "correctness_evaluated_candidates": evaluated,
            "route_integrity_ready_for_every_call": bool(runtime_receipts)
            and all(row["route_integrity_ready"] for row in runtime_receipts),
            "external_inference_calls": int(mapping(candidate.get("counters")).get("external_inference_calls") or 0),
            "user_facing_effects": int(mapping(candidate.get("counters")).get("user_facing_effects") or 0),
        },
        "observed_residuals": {
            "typed_action_parse_faults_by_arm": parse_faults,
            "task_path_namespace": namespace,
            "integrated_runtime_red_is_separate_from_route_integrity": any(
                row["arm_id"] == "integrated_local_model"
                and row["trigger_state"] == "RED"
                and row["route_integrity_ready"]
                for row in runtime_receipts
            ),
            "integrated_failed_gate_names": sorted({
                gate["name"]
                for row in runtime_receipts if row["arm_id"] == "integrated_local_model"
                for gate in row["failed_gates"]
            }),
        },
        "consumption": {
            "task_consumed": True,
            "eligible_for_exact_rerun": False,
            "eligible_for_training": False,
            "eligible_for_D1_or_D2": False,
        },
        "next_stage": {
            "id": "P2B_NEW_INSTRUMENT_AND_SOURCE_DISJOINT_TASK",
            "requirements": [
                "freeze a new instrument identity before opening a new task",
                "use one canonical repo-relative path namespace in request, context, edits, verifier, and evaluator",
                "treat a top-level runtime RED separately from a route-integrity receipt",
                "select the local model prospectively from retained model-selection evidence",
                "consume a fresh licensed source-disjoint development task exactly once",
            ],
        },
        "maximum_inference": (
            "The exact frozen TMax P2A instrument/task combination did not reach independent correctness evaluation. "
            "This does not estimate a Theseus subsystem effect, compare direct with integrated quality, falsify the "
            "ASI Stack mechanism, or establish general model competence."
        ),
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "training_rows_written": 0,
            "user_facing_effects": 0,
        },
    }


def namespace_audit(task: dict[str, Any]) -> dict[str, Any]:
    allowed = strings(task.get("allowed_effect_paths"))
    request = str(task.get("natural_request") or "")
    referenced_suffixes = sorted({
        part.strip(".,;:()[]{}'")
        for part in request.split()
        if "/" in part and not part.startswith(("http://", "https://"))
    })
    ambiguous = sorted({
        suffix for suffix in referenced_suffixes
        if suffix not in allowed and any(path.endswith("/" + suffix) for path in allowed)
    })
    top_levels = sorted({Path(path).parts[0] for path in allowed if Path(path).parts})
    return {
        "allowed_effect_paths": allowed,
        "request_repo_relative_paths": referenced_suffixes,
        "request_paths_rejected_but_allowed_with_archive_prefix": ambiguous,
        "archive_top_level_prefixes": top_levels,
        "repo_relative_path_ambiguity_present": bool(ambiguous),
    }


def compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    checks = mapping(report.get("recomputed_checks"))
    return {
        "trigger_state": report.get("trigger_state"),
        "scientific_status": report.get("scientific_status"),
        "terminal_disposition": report.get("terminal_disposition"),
        "parseable_candidates": checks.get("parseable_candidates"),
        "correctness_evaluated_candidates": checks.get("correctness_evaluated_candidates"),
        "next_stage": mapping(report.get("next_stage")).get("id"),
        "faults": report.get("faults"),
    }


def source_identity(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def strings(value: Any) -> list[str]:
    return [str(row) for row in value if isinstance(row, str)] if isinstance(value, list) else []


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
