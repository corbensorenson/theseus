#!/usr/bin/env python3
"""Run the sealed ten-task P4 campaign exactly once, with crash-safe resumption."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4_cognitive_compilation_evaluator as p4_evaluator  # noqa: E402


POLICY = "project_theseus_p4_cognitive_compilation_campaign_v1"
POOL = ROOT / "configs" / "theseus_p4_task_pool.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4_cognitive_compilation_instrument.json"
POOL_SEAL_COMMIT = "1aa756e2a83ade8a144dfa1ef309ca2934b50720"
POOL_SHA256 = "27204dbc4009d54181aee176e97fc7a42e42c51089804178fdcb49f23504bd13"
PROGRESS = ROOT / "reports" / "theseus_p4_campaign_progress.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = audit_campaign()
    if not args.audit_only and report["trigger_state"] == "GREEN":
        report = run_campaign()
    p2a.write_json(PROGRESS, report)
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "complete_tasks": report["complete_tasks"],
        "pending_tasks": report["pending_tasks"],
        "model_calls_retained": report["model_calls_retained"],
        "faults": report["faults"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def run_campaign() -> dict[str, Any]:
    pool = p2a.read_json(POOL)
    for row in p2a.dicts(pool.get("tasks")):
        paths = result_paths(row)
        if not paths["run"].is_file():
            partial = runtime_reports(row)
            if partial:
                return audit_campaign([f"partial_unsealed_runtime_receipts:{row['stem']}"])
            run = p4.run_experiment(INSTRUMENT, ROOT / str(row["task"]))
            p2a.write_json(paths["run"], run)
            if run.get("trigger_state") not in {"GREEN", "YELLOW"}:
                return audit_campaign([f"candidate_run_red:{row['stem']}"])
        if not paths["evaluation"].is_file():
            evaluation = p4_evaluator.evaluate_report(
                paths["run"], ROOT / str(row["evaluator"])
            )
            p2a.write_json(paths["evaluation"], evaluation)
            if evaluation.get("trigger_state") != "GREEN":
                return audit_campaign([f"blind_evaluation_red:{row['stem']}"])
        p2a.write_json(PROGRESS, audit_campaign())
    return audit_campaign()


def audit_campaign(extra_faults: list[str] | None = None) -> dict[str, Any]:
    faults = list(extra_faults or [])
    if p2a.sha256_file(POOL) != POOL_SHA256:
        faults.append("task_pool_digest_mismatch")
    pool = p2a.read_json(POOL)
    if pool.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("task_pool_not_sealed")
    if pool.get("instrument_freeze_commit") != "4ef352303a4d4c93288d1db3da659a874663c6d3":
        faults.append("instrument_freeze_binding_invalid")
    if p2a.sha256_file(INSTRUMENT) != str(pool.get("instrument_sha256") or ""):
        faults.append("instrument_digest_mismatch")
    rows = p2a.dicts(pool.get("tasks"))
    if len(rows) != 10:
        faults.append("task_count_invalid")
    status_rows: list[dict[str, Any]] = []
    total_calls = 0
    for expected, row in enumerate(rows, 1):
        stem = str(row.get("stem") or "")
        if int(row.get("campaign_index") or 0) != expected:
            faults.append(f"campaign_index_invalid:{stem}")
        task_path = ROOT / str(row.get("task") or "")
        evaluator_path = ROOT / str(row.get("evaluator") or "")
        if p2a.sha256_file(task_path) != str(row.get("task_sha256") or ""):
            faults.append(f"task_binding_invalid:{stem}")
        if p2a.sha256_file(evaluator_path) != str(row.get("evaluator_sha256") or ""):
            faults.append(f"evaluator_binding_invalid:{stem}")
        paths = result_paths(row)
        run_exists = paths["run"].is_file()
        evaluation_exists = paths["evaluation"].is_file()
        receipts = runtime_reports(row)
        if evaluation_exists and not run_exists:
            faults.append(f"evaluation_without_run:{stem}")
        if receipts and not run_exists:
            faults.append(f"partial_unsealed_runtime_receipts:{stem}")
        run: dict[str, Any] = {}
        evaluation: dict[str, Any] = {}
        if run_exists:
            run = p2a.read_json(paths["run"])
            if run.get("instrument_sha256") != p2a.sha256_file(INSTRUMENT):
                faults.append(f"run_instrument_binding_invalid:{stem}")
            if run.get("task_sha256") != p2a.sha256_file(task_path):
                faults.append(f"run_task_binding_invalid:{stem}")
            if p2a.mapping(run.get("matched_set")).get("ready") is not True:
                faults.append(f"matched_set_invalid:{stem}")
            calls = int(p2a.mapping(run.get("denominators")).get("model_calls") or 0)
            total_calls += calls
            if calls != 6:
                faults.append(f"model_call_budget_invalid:{stem}")
            if int(p2a.mapping(run.get("denominators")).get("model_loads") or 0) != 1:
                faults.append(f"model_load_budget_invalid:{stem}")
            retained = sum(
                len(p2a.dicts(attempt.get("runtime_calls")))
                for attempt in p2a.dicts(run.get("attempts"))
            )
            if retained != 6 or len(receipts) != 6:
                faults.append(f"runtime_receipt_count_invalid:{stem}")
        if evaluation_exists:
            evaluation = p2a.read_json(paths["evaluation"])
            if evaluation.get("candidate_report_sha256") != p2a.sha256_file(paths["run"]):
                faults.append(f"evaluation_run_binding_invalid:{stem}")
            if evaluation.get("evaluator_sha256") != p2a.sha256_file(evaluator_path):
                faults.append(f"evaluation_evaluator_binding_invalid:{stem}")
            if evaluation.get("trigger_state") != "GREEN":
                faults.append(f"blind_evaluation_invalid:{stem}")
        status_rows.append({
            "campaign_index": expected,
            "stem": stem,
            "run": relative(paths["run"]) if run_exists else "",
            "run_sha256": p2a.sha256_file(paths["run"]),
            "evaluation": relative(paths["evaluation"]) if evaluation_exists else "",
            "evaluation_sha256": p2a.sha256_file(paths["evaluation"]),
            "runtime_receipts": len(receipts),
            "complete": run_exists and evaluation_exists,
        })
    complete = sum(row["complete"] for row in status_rows)
    pending = len(status_rows) - complete
    if complete == 10 and total_calls != 60:
        faults.append("campaign_total_call_count_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "scope": "Exact sealed P4 local development campaign; no hosted, D1, D2, serving, training, or automatic book-support authority.",
        "pool": relative(POOL),
        "pool_sha256": p2a.sha256_file(POOL),
        "pool_seal_commit": POOL_SEAL_COMMIT,
        "instrument": relative(INSTRUMENT),
        "instrument_sha256": p2a.sha256_file(INSTRUMENT),
        "complete_tasks": complete,
        "pending_tasks": pending,
        "model_calls_retained": total_calls,
        "tasks": status_rows,
        "hosted_reference": {
            "model": "gpt-5.6-luna",
            "effort": "xhigh",
            "state": "DEFINED_TRANSPORT_NOT_BOUND",
            "calls": 0,
            "P4_blocking": False,
        },
        "maximum_inference": "Campaign execution custody only; terminal scientific status is computed separately from sealed blind evaluations.",
    }


def result_paths(row: dict[str, Any]) -> dict[str, Path]:
    suffix = str(row.get("stem") or "").removeprefix("p4_")
    return {
        "run": ROOT / "reports" / f"theseus_p4_{suffix}_run.json",
        "evaluation": ROOT / "reports" / f"theseus_p4_{suffix}_evaluation.json",
    }


def runtime_reports(row: dict[str, Any]) -> list[Path]:
    task = p2a.read_json(ROOT / str(row.get("task") or ""))
    task_id = p2a.safe_slug(str(task.get("opaque_task_id") or ""))
    return sorted((ROOT / "runtime" / "p2a").glob(f"*{task_id}*.json"))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
