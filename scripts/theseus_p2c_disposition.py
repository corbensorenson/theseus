#!/usr/bin/env python3
"""Independently classify the consumed P2C instrument-adequacy canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p2c_terminal_disposition_v1"


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
            receipts.append({
                "arm_id": arm,
                "call_number": int(call.get("call_number") or 0),
                "path": relative(path),
                "sha256": sha256_file(path),
                "route_integrity_ready": mapping(runtime.get("route_integrity")).get("ready") is True,
                "runtime_trigger_state": runtime.get("trigger_state"),
                "assistant_output_sha256": sha256_text(str(runtime.get("assistant_text") or "")),
            })
    if not receipts or not all(row["route_integrity_ready"] for row in receipts):
        faults.append("runtime_route_receipts_invalid")

    attempts = {str(row.get("arm_id") or ""): row for row in dicts(run.get("attempts"))}
    direct = mapping(attempts.get("direct_local_model"))
    integrated = mapping(attempts.get("integrated_local_model"))
    results = dicts(evaluation.get("results"))
    result = results[0] if len(results) == 1 else {}
    verification = mapping(result.get("verification"))
    run_counts = mapping(run.get("denominators"))
    evaluation_counts = mapping(evaluation.get("denominators"))

    exact_adequate_unsolved = (
        not faults
        and grammar_transport["contains_actual_newline"]
        and not grammar_transport["contains_literal_backslash_n"]
        and int(run_counts.get("model_loads") or 0) == 1
        and int(run_counts.get("model_calls") or 0) == 3
        and int(run_counts.get("parseable_candidates") or 0) == 1
        and direct.get("parseable_candidate") is False
        and integrated.get("parseable_candidate") is True
        and int(evaluation_counts.get("correctness_evaluated_candidates") or 0) == 1
        and int(evaluation_counts.get("useful_candidates") or 0) == 0
        and result.get("arm_id") == "integrated_local_model"
        and int(result.get("actions_applied") or 0) == 1
        and int(result.get("allowed_effects") or 0) == 1
        and int(result.get("rollback_verified") or 0) == 1
        and int(result.get("unsafe") or 0) == 0
        and verification.get("passed") is False
        and "optional choice metavar must not be double bracketed"
        in str(verification.get("stderr_tail") or "")
    )

    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "scientific_status": (
            "INSTRUMENT_ADEQUATE_TASK_NOT_SOLVED" if exact_adequate_unsolved else "UNCLASSIFIED"
        ),
        "terminal_disposition": (
            "P2C_TERMINAL_INSTRUMENT_ADEQUATE_ZERO_USEFUL"
            if exact_adequate_unsolved else "P2C_TERMINAL_REVIEW_REQUIRED"
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
            "one_persistent_model_load": int(run_counts.get("model_loads") or 0) == 1,
            "three_model_calls": int(run_counts.get("model_calls") or 0) == 3,
            "actual_newline_grammar_transport": grammar_transport["contains_actual_newline"]
            and not grammar_transport["contains_literal_backslash_n"],
            "direct_candidate_parseable": direct.get("parseable_candidate") is True,
            "integrated_candidate_parseable": integrated.get("parseable_candidate") is True,
            "correctness_evaluated_candidates": int(
                evaluation_counts.get("correctness_evaluated_candidates") or 0
            ),
            "useful_candidates": int(evaluation_counts.get("useful_candidates") or 0),
            "route_integrity_ready_for_every_call": bool(receipts)
            and all(row["route_integrity_ready"] for row in receipts),
            "rollback_verified": int(result.get("rollback_verified") or 0) == 1,
            "unsafe_candidates": int(result.get("unsafe") or 0),
        },
        "grammar_transport": grammar_transport,
        "observed_residuals": [
            "The direct arm did not emit a parseable typed edit after its one allowed repair.",
            "The integrated arm emitted one safe, authorized, parseable edit on its first call.",
            "The integrated edit preserved the original double-bracketing defect and failed the sealed hidden oracle.",
        ],
        "consumption": {
            "task_consumed": True,
            "eligible_for_exact_rerun": False,
            "eligible_for_training": False,
            "eligible_for_D1_or_D2": False,
        },
        "next_stage": {
            "id": "P3_TEN_TASK_MATCHED_RESIDUAL_CAMPAIGN",
            "requirements": [
                "freeze a fresh ten-task licensed source-disjoint development pool before candidate generation",
                "retain blind route-hidden evaluation and disposable-snapshot effects",
                "report direct and integrated denominators separately, including malformed, failed, useful, unsafe, rollback, latency, and weak-tail outcomes",
                "treat the governed Luna-xhigh 2x2 as a separate hosted reference denominator only after a callable transport is source-bound",
                "select no P4 mechanism until P3 exposes a nonzero, decision-relevant residual on an adequate instrument",
            ],
        },
        "maximum_inference": (
            "P2C proves that the exact Qwen3.5 direct/integrated harness can produce, apply, seal, and independently score an authorized edit under the frozen budget. "
            "The one Click task was not solved. A single integrated parseable candidate cannot establish a direct-versus-integrated effect, model competence, any Theseus subsystem benefit, or any ASI Stack claim."
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
