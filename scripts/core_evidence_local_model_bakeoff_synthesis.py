#!/usr/bin/env python3
"""Synthesize the preregistered consumed-task local-model bakeoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "core_evidence_local_model_bakeoff.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("CANDIDATE_REPORT", "EVALUATION_REPORT"),
        required=True,
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    pairs = [(Path(a).resolve(), Path(b).resolve()) for a, b in args.pair]
    report = synthesize(plan_path, pairs)
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "selected_candidate_id": report["selection"]["selected_candidate_id"],
        "terminal_disposition": report["terminal_disposition"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def synthesize(
    plan_path: Path,
    pairs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    plan = read_json(plan_path)
    task_manifest_path = ROOT / str(plan["task_manifest"])
    task_manifest = read_json(task_manifest_path)
    expected_ids = [
        str(candidate["candidate_id"]) for candidate in plan["candidates"]
    ]
    family_by_task = {
        str(task["opaque_task_id"]): str(task["family"])
        for task in task_manifest["tasks"]
    }
    rows = [
        candidate_row(plan, family_by_task, candidate_path, evaluation_path)
        for candidate_path, evaluation_path in pairs
    ]
    ids = [row["candidate_id"] for row in rows]
    complete = sorted(ids) == sorted(expected_ids) and len(ids) == len(set(ids))
    eligible = [row for row in rows if row["eligible"]]
    ranked = sorted(eligible, key=selection_key)
    selected = ranked[0] if ranked else None
    adequate = bool(selected and selected["adequacy"]["passes"])
    terminal = (
        "FREEZE_WINNER_FOR_NEW_SOURCE_DISJOINT_QUALIFICATION"
        if adequate
        else "NO_LOCAL_MODEL_ADEQUATE_FOR_FRESH_QUALIFICATION"
    )
    diagnostic = sorted(
        rows,
        key=lambda row: (
            -int(row["metrics"]["useful"]),
            int(row["metrics"]["unsafe"]),
            int(row["metrics"]["malformed_or_abstained_tasks"]),
            int(row["metrics"]["local_model_calls"]),
            float(row["metrics"]["worker_wall_ms"]),
        ),
    )[0] if rows else None
    trigger_state = "GREEN" if complete else "RED"
    return {
        "policy": "project_theseus_local_model_bakeoff_synthesis_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "scope": "consumed_development_model_selection_no_capability_claim",
        "source_identities": {
            "plan": relative_identity(plan_path),
            "task_manifest": relative_identity(task_manifest_path),
            "synthesis_source": relative_identity(Path(__file__).resolve()),
        },
        "candidates": rows,
        "selection": {
            "eligible_candidate_ids": [
                row["candidate_id"] for row in ranked
            ],
            "selected_candidate_id": (
                selected["candidate_id"] if selected else None
            ),
            "selection_rule": plan["selection_rule"],
            "diagnostic_signal_leader": (
                diagnostic["candidate_id"] if diagnostic else None
            ),
            "diagnostic_signal_leader_is_eligible": bool(
                diagnostic and diagnostic["eligible"]
            ),
            "diagnostic_signal_is_not_qualification": True,
        },
        "adequacy_floor": plan["adequacy_floor"],
        "selected_candidate_passes_adequacy_floor": adequate,
        "terminal_disposition": terminal,
        "next_action": (
            "Freeze the selected model and worker for a new source-disjoint "
            "qualification cohort."
            if adequate else
            "Do not consume a fresh cohort. Repair the measured model/runtime/"
            "worker walls on consumed development tasks, preregister a successor "
            "bakeoff, and retain all subsystem arms sealed."
        ),
        "counters": {
            "external_inference_calls": sum(
                int(row["counters"]["external_inference_calls"]) for row in rows
            ),
            "teacher_calls": sum(
                int(row["counters"]["teacher_calls"]) for row in rows
            ),
            "public_calibration_cases_consumed": sum(
                int(row["counters"]["public_calibration_cases_consumed"])
                for row in rows
            ),
            "D2_cases_consumed": sum(
                int(row["counters"]["D2_cases_consumed"]) for row in rows
            ),
        },
        "maximum_inference": (
            "This consumed-task bakeoff can select repair priorities and a "
            "candidate for future qualification. It cannot establish fresh-task "
            "competence, generalization, or any VCM, planning, routing, "
            "governance, or reuse effect."
        ),
    }


def candidate_row(
    plan: dict[str, Any],
    family_by_task: dict[str, str],
    candidate_path: Path,
    evaluation_path: Path,
) -> dict[str, Any]:
    candidate = read_json(candidate_path)
    evaluation = read_json(evaluation_path)
    candidate_id = str(candidate["candidate_id"])
    preregistered = next(
        (
            row for row in plan["candidates"]
            if str(row["candidate_id"]) == candidate_id
        ),
        None,
    )
    if preregistered is None:
        raise ValueError(f"unregistered_candidate:{candidate_id}")
    if evaluation.get("candidate_report_sha256") != sha256_file(candidate_path):
        raise ValueError(f"evaluation_candidate_hash_mismatch:{candidate_id}")
    denominators = candidate["denominators"]
    evaluated = evaluation["denominators"]
    planned = int(denominators["planned"])
    task_results = {
        str(task["opaque_task_id"]): task for task in evaluation["tasks"]
    }
    family_counts: dict[str, dict[str, int]] = {}
    malformed_or_abstained = 0
    for task_id, family in family_by_task.items():
        bucket = family_counts.setdefault(family, {"planned": 0, "useful": 0})
        bucket["planned"] += 1
        result = task_results.get(task_id)
        if result is None:
            continue
        bucket["useful"] += int(result.get("useful") or 0)
        malformed_or_abstained += int(
            bool(result.get("malformed")) or bool(result.get("abstained"))
        )
    family_rates = {
        family: (
            values["useful"] / values["planned"] if values["planned"] else 0.0
        )
        for family, values in family_counts.items()
    }
    useful = int(evaluated["useful"])
    useful_rate = useful / planned if planned else 0.0
    weakest_family_rate = min(family_rates.values(), default=0.0)
    counters = candidate["counters"]
    eligibility_checks = {
        "attempted_tasks": int(denominators["attempted"]) == planned,
        "infrastructure_failures": (
            int(denominators["infrastructure_failed"]) == 0
        ),
        "sealed_tasks": int(denominators["sealed"]) == planned,
        "unsafe": int(evaluated["unsafe"]) == 0,
        "rollback_verified": int(evaluated["rollback_verified"]) == planned,
        "external_inference_calls": (
            int(counters["external_inference_calls"]) == 0
        ),
        "teacher_calls": int(counters["teacher_calls"]) == 0,
        "public_calibration_cases_consumed": (
            int(counters["public_calibration_cases_consumed"]) == 0
        ),
        "D2_cases_consumed": int(counters["D2_cases_consumed"]) == 0,
    }
    eligible = all(eligibility_checks.values())
    floor = plan["adequacy_floor"]
    adequacy_checks = {
        "minimum_attempted_tasks": (
            int(denominators["attempted"])
            >= int(floor["minimum_attempted_tasks"])
        ),
        "minimum_useful_rate": (
            useful_rate >= float(floor["minimum_useful_rate"])
        ),
        "minimum_weakest_family_rate": (
            weakest_family_rate >= float(floor["minimum_weakest_family_rate"])
        ),
        "zero_unsafe": (
            not bool(floor["zero_unsafe_required"])
            or int(evaluated["unsafe"]) == 0
        ),
        "exact_rollback": (
            not bool(floor["exact_rollback_required"])
            or int(evaluated["rollback_verified"]) == planned
        ),
    }
    calls = sum(
        int(task["local_model_inference_calls"]) for task in candidate["tasks"]
    )
    return {
        "candidate_id": candidate_id,
        "model_identity": candidate["model_identity"],
        "eligible": eligible,
        "eligibility_checks": eligibility_checks,
        "adequacy": {
            "passes": eligible and all(adequacy_checks.values()),
            "checks": adequacy_checks,
            "useful_rate": round(useful_rate, 6),
            "family_useful_rates": {
                key: round(value, 6) for key, value in family_rates.items()
            },
            "weakest_family_rate": round(weakest_family_rate, 6),
        },
        "metrics": {
            "planned": planned,
            "attempted": int(denominators["attempted"]),
            "sealed": int(denominators["sealed"]),
            "infrastructure_failed": int(
                denominators["infrastructure_failed"]
            ),
            "evaluated": int(evaluated["attempted"]),
            "useful": useful,
            "unsafe": int(evaluated["unsafe"]),
            "malformed_or_abstained_tasks": malformed_or_abstained,
            "rollback_verified": int(evaluated["rollback_verified"]),
            "local_model_calls": calls,
            "worker_wall_ms": float(candidate["runtime"]["wall_ms"]),
        },
        "faults": candidate["faults"],
        "counters": counters,
        "source_identities": {
            "candidate_report": relative_identity(candidate_path),
            "evaluation_report": relative_identity(evaluation_path),
        },
    }


def selection_key(row: dict[str, Any]) -> tuple[float, int, int, int, float]:
    metrics = row["metrics"]
    return (
        -int(metrics["useful"]),
        int(metrics["unsafe"]),
        int(metrics["malformed_or_abstained_tasks"]),
        int(metrics["local_model_calls"]),
        float(metrics["worker_wall_ms"]),
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def relative_identity(path: Path) -> dict[str, str]:
    try:
        shown = str(path.relative_to(ROOT))
    except ValueError:
        shown = str(path)
    return {"path": shown, "sha256": sha256_file(path)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
