#!/usr/bin/env python3
"""Classify the consumed P2B run without trusting its candidate parser."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p2b_terminal_disposition_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = build(
        resolve(args.run), resolve(args.evaluation), resolve(args.instrument), resolve(args.task)
    )
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "scientific_status": report["scientific_status"],
        "terminal_disposition": report["terminal_disposition"],
        "next_stage": report["next_stage"]["id"],
        "faults": report["faults"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build(run_path: Path, evaluation_path: Path, instrument_path: Path, task_path: Path) -> dict[str, Any]:
    run = read_json(run_path)
    evaluation = read_json(evaluation_path)
    instrument = read_json(instrument_path)
    faults: list[str] = []
    if run.get("instrument_sha256") != sha256_file(instrument_path):
        faults.append("run_instrument_digest_mismatch")
    if run.get("task_sha256") != sha256_file(task_path):
        faults.append("run_task_digest_mismatch")
    if evaluation.get("candidate_report_sha256") != sha256_file(run_path):
        faults.append("evaluation_run_digest_mismatch")
    if mapping(run.get("matched_pair")).get("ready") is not True:
        faults.append("matched_pair_invalid")
    if evaluation.get("trigger_state") != "GREEN":
        faults.append("blind_evaluation_invalid")

    grammar = str(mapping(instrument.get("candidate_protocol")).get("grammar") or "")
    grammar_transport = {
        "contains_literal_backslash_n": "\\n" in grammar,
        "contains_actual_newline": "\n" in grammar,
        "sha256": sha256_text(grammar),
    }
    receipts: list[dict[str, Any]] = []
    for attempt in dicts(run.get("attempts")):
        arm = str(attempt.get("arm_id") or "")
        for call in dicts(attempt.get("runtime_calls")):
            path = resolve(str(call.get("report_path") or ""))
            if not path.is_file() or sha256_file(path) != str(call.get("report_sha256") or ""):
                faults.append(f"runtime_report_binding_invalid:{arm}:{call.get('call_number')}")
                continue
            runtime = read_json(path)
            text = str(runtime.get("assistant_text") or "")
            receipts.append({
                "arm_id": arm,
                "call_number": int(call.get("call_number") or 0),
                "path": relative(path),
                "sha256": sha256_file(path),
                "route_integrity_ready": mapping(runtime.get("route_integrity")).get("ready") is True,
                "runtime_trigger_state": runtime.get("trigger_state"),
                "assistant_output_sha256": sha256_text(text),
                "assistant_output_characters": len(text),
                "contains_literal_backslash_n": "\\n" in text,
                "contains_actual_newline": "\n" in text,
                "contains_header": "THESEUS_EDIT_V1" in text,
                "contains_replace_token": "REPLACE " in text,
                "contains_authorized_repo_relative_path": "src/requests/models.py" in text,
                "contains_open_marker": "<<<" in text,
                "contains_close_marker": ">>>" in text,
                "contains_end_marker": "END" in text,
            })
    escaped_transport_reproduced = (
        grammar_transport["contains_literal_backslash_n"]
        and not grammar_transport["contains_actual_newline"]
        and len(receipts) == 4
        and all(row["contains_literal_backslash_n"] and not row["contains_actual_newline"] for row in receipts)
    )
    if not receipts or not all(row["route_integrity_ready"] for row in receipts):
        faults.append("runtime_route_receipts_invalid")
    denominators = mapping(run.get("denominators"))
    evaluated = int(mapping(evaluation.get("denominators")).get("correctness_evaluated_candidates") or 0)
    exact_inconclusive = (
        not faults
        and escaped_transport_reproduced
        and int(denominators.get("model_loads") or 0) == 1
        and int(denominators.get("model_calls") or 0) == 4
        and int(denominators.get("parseable_candidates") or 0) == 0
        and evaluated == 0
    )
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "scientific_status": "INCONCLUSIVE_IMPLEMENTATION" if exact_inconclusive else "UNCLASSIFIED",
        "terminal_disposition": (
            "P2B_TERMINAL_INCONCLUSIVE_LITERAL_GRAMMAR_TRANSPORT"
            if exact_inconclusive else "P2B_TERMINAL_REVIEW_REQUIRED"
        ),
        "source_identities": {
            "run": source_identity(run_path),
            "evaluation": source_identity(evaluation_path),
            "instrument": source_identity(instrument_path),
            "task": source_identity(task_path),
            "runtime_reports": receipts,
        },
        "recomputed_checks": {
            "matched_pair_ready": mapping(run.get("matched_pair")).get("ready") is True,
            "one_persistent_model_load": int(denominators.get("model_loads") or 0) == 1,
            "four_model_calls": int(denominators.get("model_calls") or 0) == 4,
            "parseable_candidates": int(denominators.get("parseable_candidates") or 0),
            "correctness_evaluated_candidates": evaluated,
            "route_integrity_ready_for_every_call": bool(receipts)
            and all(row["route_integrity_ready"] for row in receipts),
            "escaped_grammar_transport_reproduced_in_every_output": escaped_transport_reproduced,
            "three_of_four_outputs_contain_replace_token": sum(int(row["contains_replace_token"]) for row in receipts) == 3,
            "every_output_uses_authorized_repo_relative_path": bool(receipts)
            and all(row["contains_authorized_repo_relative_path"] for row in receipts),
        },
        "grammar_transport": grammar_transport,
        "consumption": {
            "task_consumed": True,
            "eligible_for_exact_rerun": False,
            "eligible_for_training": False,
            "eligible_for_D1_or_D2": False,
        },
        "next_stage": {
            "id": "P2C_NEW_INSTRUMENT_AND_SOURCE_DISJOINT_TASK",
            "single_intended_change": "Encode the displayed edit grammar with actual newline characters and add an audit that rendered grammar round-trips through the parser.",
            "requirements": [
                "freeze a new instrument identity before opening a fresh task",
                "retain Qwen3.5, decoder, persistence, arm order, budgets, runtime overlay, and repository-relative namespace",
                "unit-test the exact rendered prompt grammar against the exact parser",
                "consume a fresh licensed source-disjoint development task exactly once",
            ],
        },
        "maximum_inference": (
            "The exact P2B prompt serialized newline separators as literal backslash-n text and all four Qwen3.5 outputs reproduced that transport, while the parser required actual newlines. "
            "P2B therefore cannot assess model competence, direct-versus-integrated quality, any Theseus subsystem effect, or any ASI Stack claim."
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
