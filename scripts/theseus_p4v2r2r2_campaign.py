#!/usr/bin/env python3
"""Run the sealed all-new P4-v2r2-r2 campaign exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation_evaluator as evaluator  # noqa: E402
import theseus_p4v2r2r3_cognitive_compilation as candidate  # noqa: E402


POLICY = "project_theseus_p4v2r2r2_campaign_v1"
POOL = ROOT / "configs" / "theseus_p4v2r2r2_task_pool.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2r2_cognitive_compilation_instrument.json"
ROUTE_CANARY = ROOT / "reports" / "theseus_p4v2r2r2_route_canary_audit.json"
PROGRESS = ROOT / "reports" / "theseus_p4v2r2r2_attempt2_campaign_progress.json"
POOL_SHA256 = "7c78025dcbad76a637016a8287e2fd6c94b7f1dc580959a8c9fd3c4e1215ef1f"
INSTRUMENT_SHA256 = "86f8e1840dd6ced36e48257eca8a427377852cb662f2ad66d53441d048218b4d"
ROUTE_CANARY_SHA256 = "14b3c8775b835972d4866e9cad2ed3d64928d600583f266e832201bcf50a15be"
RUNTIME_ATTEMPT_NAMESPACE = "p4v2r2r3_attempt1"
NORMAL_TERMINATIONS = {"parser_complete", "model_eos"}


def result_paths(row: dict[str, Any]) -> dict[str, Path]:
    stem = p2a.safe_slug(str(row.get("stem") or "task"))
    return {
        "run": ROOT / "reports" / f"theseus_p4v2r2r2_attempt2_{stem}_run.json",
        "evaluation": ROOT / "reports" / f"theseus_p4v2r2r2_attempt2_{stem}_evaluation.json",
    }


def runtime_reports(row: dict[str, Any]) -> list[Path]:
    task = p2a.read_json(ROOT / str(row.get("task") or ""))
    task_id = p2a.safe_slug(str(task.get("opaque_task_id") or ""))
    return sorted(
        (ROOT / "runtime" / "p2a").glob(
            f"*{task_id}*{RUNTIME_ATTEMPT_NAMESPACE}*.json"
        )
    )


def route_custody(receipts: list[Path]) -> dict[str, Any]:
    faults: list[str] = []
    backend_paths: list[str] = []
    terminations: list[dict[str, Any]] = []
    for index, path in enumerate(receipts, start=1):
        runtime = p2a.read_json(path)
        route = p2a.mapping(runtime.get("route_integrity"))
        if runtime.get("trigger_state") != "GREEN":
            faults.append(f"runtime_red:{index}")
        if route.get("policy") != "project_theseus_live_route_integrity_v2":
            faults.append(f"route_policy_invalid:{index}")
        if route.get("ready") is not True or route.get("release_allowed") is not True:
            faults.append(f"route_release_invalid:{index}")
        checkpoint = p2a.mapping(runtime.get("checkpoint_chat"))
        backend_path = ROOT / str(checkpoint.get("out") or "")
        backend = p2a.read_json(backend_path) if backend_path.is_file() else {}
        if backend.get("policy") != "project_theseus_local_inference_backend_v2":
            faults.append(f"backend_policy_invalid:{index}")
        if backend.get("trigger_state") != "GREEN" or p2a.strings(backend.get("faults")):
            faults.append(f"backend_red:{index}")
        metrics = p2a.mapping(backend.get("metrics"))
        reason = str(metrics.get("termination_reason") or "")
        if reason not in NORMAL_TERMINATIONS:
            faults.append(f"termination_invalid:{index}")
        if metrics.get("physical_context_boundary_hit") is True:
            faults.append(f"physical_context_boundary_hit:{index}")
        if metrics.get("project_selected_quality_token_cap") is not None:
            faults.append(f"quality_token_cap_present:{index}")
        backend_paths.append(p2a.rel(backend_path))
        terminations.append(
            {
                "termination_reason": reason,
                "generated_tokens": metrics.get("generated_tokens"),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "model_context_window_tokens": metrics.get("model_context_window_tokens"),
                "physical_context_boundary_hit": metrics.get("physical_context_boundary_hit"),
                "project_selected_quality_token_cap": metrics.get("project_selected_quality_token_cap"),
            }
        )
    if len(set(backend_paths)) != len(backend_paths):
        faults.append("backend_receipt_path_reused")
    return {
        "passed": not faults,
        "faults": faults,
        "runtime_receipts": len(receipts),
        "unique_backend_receipts": len(set(backend_paths)),
        "backend_paths": backend_paths,
        "terminations": terminations,
    }


def audit_campaign(extra_faults: list[str] | None = None) -> dict[str, Any]:
    faults = list(extra_faults or [])
    for path, expected, owner in (
        (POOL, POOL_SHA256, "pool"),
        (INSTRUMENT, INSTRUMENT_SHA256, "instrument"),
        (ROUTE_CANARY, ROUTE_CANARY_SHA256, "route_canary"),
    ):
        if not path.is_file() or p2a.sha256_file(path) != expected:
            faults.append(f"binding_invalid:{owner}")
    pool = p2a.read_json(POOL)
    instrument = p2a.read_json(INSTRUMENT)
    canary = p2a.read_json(ROUTE_CANARY)
    if pool.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("pool_not_sealed")
    if p2a.strings(pool.get("faults")) or int(pool.get("task_count") or 0) != 10:
        faults.append("pool_adequacy_invalid")
    if not all(
        int(pool.get(key) or 0) == 10
        for key in (
            "green_evaluator_audits",
            "v2r2_oracle_replays_green",
            "dependency_corruptions_rejected",
        )
    ):
        faults.append("mechanics_floor_invalid")
    if instrument.get("runtime_attempt_namespace") != RUNTIME_ATTEMPT_NAMESPACE:
        faults.append("runtime_namespace_invalid")
    if candidate.audit_instrument(INSTRUMENT).get("trigger_state") != "GREEN":
        faults.append("instrument_audit_red")
    if canary.get("trigger_state") != "GREEN" or int(canary.get("task_candidate_or_control_calls") or 0) != 0:
        faults.append("route_canary_invalid")
    if p2a.mapping(instrument.get("generation_budget")).get("project_selected_quality_token_cap") is not None:
        faults.append("quality_token_cap_present")

    tasks: list[dict[str, Any]] = []
    total_calls = 0
    total_boundary_hits = 0
    for expected_index, row in enumerate(p2a.dicts(pool.get("tasks")), start=1):
        stem = str(row.get("stem") or "")
        if int(row.get("campaign_index") or 0) != expected_index:
            faults.append(f"campaign_index_invalid:{stem}")
        task_path = ROOT / str(row.get("task") or "")
        evaluator_path = ROOT / str(row.get("evaluator") or "")
        if p2a.sha256_file(task_path) != str(row.get("task_sha256") or ""):
            faults.append(f"task_binding_invalid:{stem}")
        if p2a.sha256_file(evaluator_path) != str(row.get("evaluator_sha256") or ""):
            faults.append(f"evaluator_binding_invalid:{stem}")
        paths = result_paths(row)
        receipts = runtime_reports(row)
        run_exists = paths["run"].is_file()
        evaluation_exists = paths["evaluation"].is_file()
        custody = route_custody(receipts)
        if receipts and not run_exists:
            faults.append(f"partial_unsealed_runtime_receipts:{stem}")
        if run_exists:
            run = p2a.read_json(paths["run"])
            calls = int(p2a.mapping(run.get("denominators")).get("model_calls") or 0)
            total_calls += calls
            if calls != 6 or len(receipts) != 6:
                faults.append(f"model_call_denominator_invalid:{stem}")
            if custody.get("passed") is not True:
                faults.extend(f"route_custody:{stem}:{fault}" for fault in custody["faults"])
            total_boundary_hits += sum(
                int(item.get("physical_context_boundary_hit") is True)
                for item in custody["terminations"]
            )
        if evaluation_exists:
            evaluation = p2a.read_json(paths["evaluation"])
            if evaluation.get("trigger_state") != "GREEN":
                faults.append(f"evaluation_red:{stem}")
            if not run_exists or evaluation.get("candidate_report_sha256") != p2a.sha256_file(paths["run"]):
                faults.append(f"evaluation_run_binding_invalid:{stem}")
        tasks.append(
            {
                "campaign_index": expected_index,
                "stem": stem,
                "run": p2a.rel(paths["run"]) if run_exists else "",
                "run_sha256": p2a.sha256_file(paths["run"]),
                "evaluation": p2a.rel(paths["evaluation"]) if evaluation_exists else "",
                "evaluation_sha256": p2a.sha256_file(paths["evaluation"]),
                "runtime_receipts": len(receipts),
                "unique_backend_receipts": custody["unique_backend_receipts"],
                "complete": run_exists and evaluation_exists,
            }
        )
    complete = sum(row["complete"] for row in tasks)
    if total_calls != complete * 6:
        faults.append("retained_model_call_count_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "pool": {"path": p2a.rel(POOL), "sha256": p2a.sha256_file(POOL)},
        "instrument": {"path": p2a.rel(INSTRUMENT), "sha256": p2a.sha256_file(INSTRUMENT)},
        "route_canary": {"path": p2a.rel(ROUTE_CANARY), "sha256": p2a.sha256_file(ROUTE_CANARY)},
        "runtime_attempt_namespace": RUNTIME_ATTEMPT_NAMESPACE,
        "complete_tasks": complete,
        "pending_tasks": len(tasks) - complete,
        "model_calls_retained": total_calls,
        "physical_context_boundary_hits": total_boundary_hits,
        "project_selected_quality_token_cap": None,
        "tasks": tasks,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "training_rows_written": 0,
        "maximum_inference": "Campaign custody only. Scientific disposition requires all ten sealed blind evaluations and an independent terminal aggregation.",
    }


def run_campaign() -> dict[str, Any]:
    pool = p2a.read_json(POOL)
    for row in p2a.dicts(pool.get("tasks")):
        paths = result_paths(row)
        if not paths["run"].is_file():
            if runtime_reports(row):
                return audit_campaign([f"partial_unsealed_runtime_receipts:{row['stem']}"])
            try:
                run = candidate.run_experiment(INSTRUMENT, ROOT / str(row["task"]))
            except Exception as exc:  # noqa: BLE001 - fail closed after first bad route.
                return audit_campaign([f"candidate_run_exception:{row['stem']}:{type(exc).__name__}:{exc}"])
            receipts = runtime_reports(row)
            custody = route_custody(receipts)
            if custody.get("passed") is not True or len(receipts) != 6:
                return audit_campaign([f"candidate_route_custody_red:{row['stem']}"])
            p2a.write_json(paths["run"], run)
            if run.get("trigger_state") not in {"GREEN", "YELLOW"}:
                return audit_campaign([f"candidate_run_red:{row['stem']}"])
        if not paths["evaluation"].is_file():
            blind = evaluator.evaluate_report(paths["run"], ROOT / str(row["evaluator"]))
            p2a.write_json(paths["evaluation"], blind)
            if blind.get("trigger_state") != "GREEN":
                return audit_campaign([f"blind_evaluation_red:{row['stem']}"])
        p2a.write_json(PROGRESS, audit_campaign())
    return audit_campaign()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = audit_campaign()
    if not args.audit_only and report.get("trigger_state") == "GREEN":
        report = run_campaign()
    p2a.write_json(PROGRESS, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "trigger_state",
                    "complete_tasks",
                    "pending_tasks",
                    "model_calls_retained",
                    "physical_context_boundary_hits",
                    "project_selected_quality_token_cap",
                    "faults",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
