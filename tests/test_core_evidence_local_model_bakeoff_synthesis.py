from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.core_evidence_local_model_bakeoff_synthesis as module


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def candidate(path: Path, candidate_id: str, *, faults: int = 0) -> dict:
    value = {
        "candidate_id": candidate_id,
        "model_identity": {"repo_id": candidate_id, "revision": "fixed"},
        "denominators": {
            "planned": 3,
            "attempted": 3,
            "sealed": 3 - faults,
            "infrastructure_failed": faults,
        },
        "tasks": [
            {"local_model_inference_calls": 4} for _ in range(3 - faults)
        ],
        "faults": [{"fault": "timeout"} for _ in range(faults)],
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "runtime": {"wall_ms": 100.0},
    }
    write_json(path, value)
    return value


def evaluation(
    path: Path,
    candidate_path: Path,
    useful: int,
    *,
    attempted: int = 3,
) -> None:
    tasks = []
    ids = ["a", "b", "c"]
    for index in range(attempted):
        is_useful = index < useful
        tasks.append({
            "opaque_task_id": ids[index],
            "useful": int(is_useful),
            "unsafe": 0,
            "malformed": int(not is_useful),
            "abstained": int(not is_useful),
        })
    write_json(path, {
        "candidate_report_sha256": hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        "denominators": {
            "attempted": attempted,
            "useful": useful,
            "unsafe": 0,
            "rollback_verified": attempted,
        },
        "tasks": tasks,
    })


def plan() -> dict:
    return {
        "candidates": [
            {"candidate_id": "small"},
            {"candidate_id": "promising_but_timed_out"},
        ],
        "adequacy_floor": {
            "minimum_attempted_tasks": 3,
            "minimum_useful_rate": 0.5,
            "minimum_weakest_family_rate": 0.34,
            "zero_unsafe_required": True,
            "exact_rollback_required": True,
        },
    }


def test_ineligible_signal_leader_cannot_win(tmp_path: Path) -> None:
    small_path = tmp_path / "small.json"
    signal_path = tmp_path / "signal.json"
    candidate(small_path, "small")
    candidate(signal_path, "promising_but_timed_out", faults=2)
    small_eval_path = tmp_path / "small_eval.json"
    signal_eval_path = tmp_path / "signal_eval.json"
    evaluation(small_eval_path, small_path, 0)
    evaluation(signal_eval_path, signal_path, 1, attempted=1)

    small = module.candidate_row(
        plan(),
        {"a": "repo", "b": "repo", "c": "repo"},
        small_path,
        small_eval_path,
    )
    signal = module.candidate_row(
        plan(),
        {"a": "repo", "b": "repo", "c": "repo"},
        signal_path,
        signal_eval_path,
    )

    assert small["eligible"] is True
    assert small["adequacy"]["passes"] is False
    assert signal["eligible"] is False
    assert signal["metrics"]["useful"] == 1
    assert signal["adequacy"]["passes"] is False


def test_selection_key_prefers_useful_then_calls() -> None:
    base = {
        "metrics": {
            "useful": 0,
            "unsafe": 0,
            "malformed_or_abstained_tasks": 3,
            "local_model_calls": 20,
            "worker_wall_ms": 100.0,
        }
    }
    fewer_calls = {
        "metrics": {**base["metrics"], "local_model_calls": 10}
    }
    useful = {"metrics": {**base["metrics"], "useful": 1}}

    assert module.selection_key(useful) < module.selection_key(fewer_calls)
    assert module.selection_key(fewer_calls) < module.selection_key(base)
