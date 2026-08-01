#!/usr/bin/env python3
"""Select a successor P2B local model from retained, consumed evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p2b_local_model_selection_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis", default="reports/core_evidence_local_model_bakeoff_synthesis.json")
    parser.add_argument("--out", default="reports/theseus_p2b_local_model_selection.json")
    args = parser.parse_args()
    report = select_model(resolve(args.synthesis))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "selection_state": report["selection_state"],
        "selected_candidate_id": report["selected_candidate_id"],
        "selected_model": report["selected_model_identity"],
        "faults": report["faults"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def select_model(synthesis_path: Path) -> dict[str, Any]:
    synthesis = read_json(synthesis_path)
    faults: list[str] = []
    if synthesis.get("trigger_state") != "GREEN":
        faults.append("source_synthesis_not_green")
    if synthesis.get("terminal_disposition") != "NO_LOCAL_MODEL_ADEQUATE_FOR_FRESH_QUALIFICATION":
        faults.append("source_synthesis_disposition_unexpected")
    candidates: list[dict[str, Any]] = []
    for row in dicts(synthesis.get("candidates")):
        model = mapping(row.get("model_identity"))
        preflight_path = preflight_for(str(row.get("candidate_id") or ""))
        preflight = read_json(preflight_path) if preflight_path.is_file() else {}
        preflight_model = mapping(preflight.get("model_identity"))
        preflight_ready = (
            preflight.get("trigger_state") == "GREEN"
            and mapping(preflight.get("output")).get("exact_action_valid") is True
            and all(preflight_model.get(key) == model.get(key) for key in ("repo_id", "revision"))
        )
        metrics = mapping(row.get("metrics"))
        candidates.append({
            "candidate_id": row.get("candidate_id"),
            "model_identity": model,
            "preflight": source_identity(preflight_path) if preflight_path.is_file() else {
                "path": preflight_path.resolve().relative_to(ROOT).as_posix(),
                "missing": True,
            },
            "preflight_ready": preflight_ready,
            "attempted": int(metrics.get("attempted") or 0),
            "evaluated": int(metrics.get("evaluated") or 0),
            "useful": int(metrics.get("useful") or 0),
            "infrastructure_failed": int(metrics.get("infrastructure_failed") or 0),
            "old_worker_adequacy_passed": mapping(row.get("adequacy")).get("passes") is True,
        })
    ready = [row for row in candidates if row["preflight_ready"]]
    if not ready:
        faults.append("no_runtime_ready_candidate")
        selected: dict[str, Any] = {}
    else:
        selected = max(
            ready,
            key=lambda row: (
                row["useful"],
                row["useful"] / max(1, row["evaluated"]),
                int("9B" in str(mapping(row["model_identity"]).get("repo_id") or "")),
            ),
        )
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": faults,
        "state": "FROZEN_BEFORE_P2B_TASK_ACQUISITION",
        "selection_state": "SELECTED_FOR_P2B_INSTRUMENT_ONLY_NOT_QUALIFIED",
        "selection_rule": (
            "Among exact-action preflight-GREEN installed candidates, maximize retained useful outcomes, "
            "then useful/evaluated rate; selection is diagnostic because no candidate passed the old adequacy floor."
        ),
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_model_identity": selected.get("model_identity", {}),
        "candidates": candidates,
        "source_identity": source_identity(synthesis_path),
        "source_terminal_disposition": synthesis.get("terminal_disposition"),
        "model_qualified": False,
        "P3_eligible": False,
        "next_decision": "Run once under the new P2B instrument on a fresh licensed source-disjoint development task.",
        "maximum_inference": (
            "Qwen3.5-9B is the strongest retained diagnostic candidate for this bounded successor instrument. "
            "The selection does not establish repository competence, P3 eligibility, general superiority, or a Theseus effect."
        ),
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "fresh_tasks_consumed": 0,
        },
    }


def preflight_for(candidate_id: str) -> Path:
    names = {
        "qwen3_8b_general": "reports/core_evidence_qwen3_8b_preflight.json",
        "qwen25_coder_7b": "reports/core_evidence_qwen25_coder_7b_preflight.json",
        "qwen35_9b_general": "reports/core_evidence_qwen35_9b_preflight.json",
    }
    if candidate_id not in names:
        raise ValueError(f"unregistered candidate: {candidate_id}")
    return resolve(names[candidate_id])


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def source_identity(path: Path) -> dict[str, str]:
    return {"path": path.resolve().relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
