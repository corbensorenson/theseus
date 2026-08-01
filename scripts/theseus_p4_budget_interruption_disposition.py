#!/usr/bin/env python3
"""Bind the interrupted P4-v1 run without turning it into claim evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402


POLICY = "project_theseus_p4_budget_interruption_disposition_v1"
INSTRUMENT = ROOT / "configs" / "theseus_p4_cognitive_compilation_instrument.json"
POOL = ROOT / "configs" / "theseus_p4_task_pool.json"
PROGRESS = ROOT / "reports" / "theseus_p4_campaign_progress.json"
COMPLETION_POLICY = ROOT / "configs" / "theseus_generation_completion_policy.json"


def build_report() -> dict[str, Any]:
    pool = p2a.read_json(POOL)
    tasks = p2a.dicts(pool.get("tasks"))
    runtime_reports = sorted(
        (ROOT / "runtime" / "p2a").glob("p2a_p4_p4-cognitive-compilation-*.json")
    )
    rows: list[dict[str, Any]] = []
    for runtime_path in runtime_reports:
        runtime = p2a.read_json(runtime_path)
        backend_path = ROOT / str(p2a.mapping(runtime.get("generation_backend")).get("out") or "")
        backend = p2a.read_json(backend_path)
        metrics = p2a.mapping(backend.get("metrics"))
        answer = str(runtime.get("assistant_text") or "")
        generated = int(metrics.get("generated_tokens") or 0)
        maximum = int(
            p2a.mapping(
                p2a.mapping(p2a.mapping(runtime.get("local_model_contract")).get("identity")).get("decoder")
            ).get("maximum_tokens")
            or 0
        )
        rows.append({
            "runtime_report": p2a.rel(runtime_path),
            "runtime_report_sha256": p2a.sha256_file(runtime_path),
            "backend_report": p2a.rel(backend_path),
            "backend_report_sha256": p2a.sha256_file(backend_path),
            "generated_tokens": generated,
            "requested_maximum_tokens": maximum,
            "protocol_terminal_end_observed": answer.strip().endswith("END"),
            "observed_ceiling_hit": bool(maximum and generated >= maximum),
            "explicit_backend_finish_reason_recorded": bool(metrics.get("termination_reason")),
        })
    completed = []
    consumed = []
    unopened = []
    for task in tasks:
        index = int(task.get("campaign_index") or 0)
        stem = str(task.get("stem") or "")
        suffix = stem.removeprefix("p4_")
        run_path = ROOT / "reports" / f"theseus_p4_{suffix}_run.json"
        evaluation_path = ROOT / "reports" / f"theseus_p4_{suffix}_evaluation.json"
        task_receipts = [row for row in rows if f"-{index:02d}_" in row["runtime_report"]]
        if task_receipts:
            consumed.append(index)
        else:
            unopened.append(index)
        if run_path.is_file() and evaluation_path.is_file():
            completed.append({
                "campaign_index": index,
                "stem": stem,
                "run": p2a.rel(run_path),
                "run_sha256": p2a.sha256_file(run_path),
                "evaluation": p2a.rel(evaluation_path),
                "evaluation_sha256": p2a.sha256_file(evaluation_path),
            })
    all_terminated = bool(rows) and all(row["protocol_terminal_end_observed"] for row in rows)
    ceiling_hits = sum(int(row["observed_ceiling_hit"]) for row in rows)
    max_generated = max((int(row["generated_tokens"]) for row in rows), default=0)
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "YELLOW",
        "campaign_state": "P4_V1_INTERRUPTED_BUDGET_POLICY_REJECTED",
        "scientific_status": "INCONCLUSIVE_EXPERIMENT",
        "reason": "The inherited 1,536-token cap was not adequacy-derived and the v1 backend did not retain an explicit finish reason. The campaign was stopped before further task consumption.",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "instrument": p2a.rel(INSTRUMENT),
        "instrument_sha256": p2a.sha256_file(INSTRUMENT),
        "pool": p2a.rel(POOL),
        "pool_sha256": p2a.sha256_file(POOL),
        "campaign_progress": p2a.rel(PROGRESS),
        "campaign_progress_sha256": p2a.sha256_file(PROGRESS),
        "replacement_completion_policy": p2a.rel(COMPLETION_POLICY),
        "replacement_completion_policy_sha256": p2a.sha256_file(COMPLETION_POLICY),
        "execution_custody": {
            "complete_tasks_with_run_and_blind_evaluation": len(completed),
            "complete_task_evidence": completed,
            "retained_model_call_reports": len(rows),
            "retained_calls_inside_complete_task_reports": 12,
            "retained_partial_task_3_calls": max(0, len(rows) - 12),
            "additional_call_interrupted_without_durable_report": 1,
            "consumed_campaign_indices": consumed,
            "unopened_campaign_indices": unopened,
            "runtime_reports": rows,
        },
        "budget_audit": {
            "requested_fixed_tokens_per_call": 1536,
            "maximum_observed_generated_tokens": max_generated,
            "observed_ceiling_hits": ceiling_hits,
            "all_retained_outputs_reached_protocol_terminal_end": all_terminated,
            "explicit_finish_reason_present_in_v1_receipts": all(
                row["explicit_backend_finish_reason_recorded"] for row in rows
            ),
            "retained_outputs_observed_truncated": bool(ceiling_hits or not all_terminated),
            "finding": "The retained evidence shows no observed truncation, but the cap remains scientifically arbitrary and v1 termination telemetry is inadequate for a decision campaign."
        },
        "evidence_disposition": {
            "tasks_1_and_2": "Retain as completed mechanics-and-budget pilot rows only; do not aggregate into a P4 treatment effect.",
            "task_3": "Consumed by two retained Semantic-IR calls and one interrupted attempted call; never replay for fresh credit.",
            "tasks_4_through_10": "Unopened and eligible only for a prospectively frozen repaired instrument; preserve exact source/evaluator custody.",
            "claim_support_state_effect": "none",
            "model_or_mechanism_negative_inference": "forbidden"
        },
        "successor_requirements": [
            "stop on a complete declared artifact envelope or model EOS",
            "derive the only numeric token boundary from the pinned model context residual rather than a project-selected answer length",
            "retain exact prompt tokens, generated tokens, finish reason, termination reason, and ceiling-hit status",
            "invalidate rather than score any observation that reaches the physical context ceiling or a host emergency stop",
            "freeze the repaired runner and completion policy before opening another task",
            "use the seven unopened tasks plus three new source-disjoint licensed tasks to restore a ten-task decision denominator"
        ],
        "maximum_inference": "This report establishes interruption custody and a generation-budget adequacy defect only. It cannot rank arms, qualify P4, falsify cognitive compilation, move a book claim, qualify serving, or enter D1/D2/training evidence."
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="reports/theseus_p4_budget_interruption_disposition.json",
    )
    args = parser.parse_args()
    report = build_report()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "campaign_state": report["campaign_state"],
        "retained_model_call_reports": report["execution_custody"]["retained_model_call_reports"],
        "consumed_campaign_indices": report["execution_custody"]["consumed_campaign_indices"],
        "unopened_campaign_indices": report["execution_custody"]["unopened_campaign_indices"],
        "observed_ceiling_hits": report["budget_audit"]["observed_ceiling_hits"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
