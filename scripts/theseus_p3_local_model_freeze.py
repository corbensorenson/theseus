#!/usr/bin/env python3
"""Freeze the best retained local denominator before P3 task acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p3_local_model_freeze_v1"
SELECTION_STATE = (
    "FROZEN_AS_BEST_RETAINED_P3_DEVELOPMENT_LOCAL_DENOMINATOR_NOT_CAPABILITY_QUALIFIED"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-selection", required=True)
    parser.add_argument("--p2c-disposition", required=True)
    parser.add_argument("--p2c-instrument", required=True)
    parser.add_argument("--worker-config", required=True)
    parser.add_argument("--runtime-preflight", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = build(
        resolve(args.previous_selection),
        resolve(args.p2c_disposition),
        resolve(args.p2c_instrument),
        resolve(args.worker_config),
        resolve(args.runtime_preflight),
    )
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "selection_state": report["selection_state"],
        "selected_candidate_id": report["selected_candidate_id"],
        "P3_eligible": report["P3_eligible"],
        "model_capability_qualified": report["model_capability_qualified"],
        "faults": report["faults"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build(
    previous_path: Path,
    disposition_path: Path,
    instrument_path: Path,
    worker_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    previous = read_json(previous_path)
    disposition = read_json(disposition_path)
    instrument = read_json(instrument_path)
    worker = read_json(worker_path)
    preflight = read_json(preflight_path)
    faults: list[str] = []

    if previous.get("trigger_state") != "GREEN" or previous.get("selected_candidate_id") != "qwen35_9b_general":
        faults.append("retained_selection_invalid")
    if disposition.get("trigger_state") != "GREEN" or disposition.get("scientific_status") != "INSTRUMENT_ADEQUATE_TASK_NOT_SOLVED":
        faults.append("p2c_instrument_adequacy_not_established")
    if disposition.get("terminal_disposition") != "P2C_TERMINAL_INSTRUMENT_ADEQUATE_ZERO_USEFUL":
        faults.append("p2c_terminal_disposition_invalid")
    if preflight.get("trigger_state") != "GREEN":
        faults.append("runtime_preflight_not_green")

    frozen = mapping(instrument.get("frozen_model"))
    previous_identity = mapping(previous.get("selected_model_identity"))
    preflight_identity = mapping(preflight.get("model_identity"))
    worker_identity = mapping(worker.get("model"))
    for field in ("repo_id", "revision"):
        values = {
            str(frozen.get(field) or ""),
            str(previous_identity.get(field) or ""),
            str(preflight_identity.get(field) or ""),
            str(worker_identity.get(field) or ""),
        }
        if len(values) != 1 or "" in values:
            faults.append(f"model_{field}_mismatch")
    snapshots = {
        str(frozen.get("snapshot_manifest_sha256") or ""),
        str(preflight_identity.get("snapshot_manifest_sha256") or ""),
    }
    if len(snapshots) != 1 or "" in snapshots:
        faults.append("model_snapshot_mismatch")

    eligible = not faults
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "GREEN" if eligible else "RED",
        "faults": sorted(set(faults)),
        "state": "FROZEN_BEFORE_P3_TASK_POOL_ACQUISITION",
        "selection_state": SELECTION_STATE,
        "selected_candidate_id": "qwen35_9b_general",
        "P3_eligible": eligible,
        "model_capability_qualified": False,
        "selected_model_identity": {
            "repo_id": frozen.get("repo_id"),
            "revision": frozen.get("revision"),
            "snapshot_manifest_sha256": frozen.get("snapshot_manifest_sha256"),
            "identity_sha256": frozen.get("identity_sha256"),
            "decoder_sha256": frozen.get("decoder_sha256"),
            "worker_config_sha256": sha256_file(worker_path),
        },
        "selection_rule": (
            "Freeze the strongest retained installed candidate before opening P3 tasks: Qwen3.5 was the only prior candidate with a useful retained result, passed exact-action runtime preflight, and then exercised the complete parse/apply/seal/blind-evaluate path under P2C. No installed candidate has established broad capability competence."
        ),
        "source_identities": {
            "previous_selection": source_identity(previous_path),
            "p2c_disposition": source_identity(disposition_path),
            "p2c_instrument": source_identity(instrument_path),
            "worker_config": source_identity(worker_path),
            "runtime_preflight": source_identity(preflight_path),
        },
        "known_limits": [
            "The P2C task produced zero useful candidates.",
            "The retained three-task bakeoff qualified no local model under its old worker adequacy floor.",
            "P3 is a development residual campaign, not a model promotion or general competence evaluation.",
        ],
        "maximum_inference": (
            "Qwen3.5-9B is frozen as the best retained local P3 development denominator and is adequate to exercise the exact harness. This does not establish general repository competence, model superiority, serving promotion, a Theseus effect, or an ASI Stack claim."
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


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def source_identity(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
