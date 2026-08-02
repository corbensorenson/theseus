#!/usr/bin/env python3
"""Run the zero-call-repaired P4 campaign in an isolated receipt namespace."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import theseus_assistant_p2a as p2a
import theseus_p4v2r2r2_campaign as predecessor
import theseus_p4v2r2r4_cognitive_compilation as candidate


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p4v2r2r3_campaign_v1"
POOL = ROOT / "configs" / "theseus_p4v2r2r2_task_pool.json"
INSTRUMENT = (
    ROOT / "configs" / "theseus_p4v2r2r3_prompt_continuity_repair.json"
)
ROUTE_CANARY = ROOT / "reports" / "theseus_p4v2r2r2_route_canary_audit.json"
PRODUCTION_CONFORMANCE = (
    ROOT / "reports" / "theseus_p4v2r2r3_production_conformance.json"
)
PROGRESS = ROOT / "reports" / "theseus_p4v2r2r3_attempt1_campaign_progress.json"
POOL_SHA256 = "7c78025dcbad76a637016a8287e2fd6c94b7f1dc580959a8c9fd3c4e1215ef1f"
INSTRUMENT_SHA256 = "831fbefeab84d856b87233f3e8bfd63cf0655ea080eda5d24f58bc69260bab76"
ROUTE_CANARY_SHA256 = "14b3c8775b835972d4866e9cad2ed3d64928d600583f266e832201bcf50a15be"
PRODUCTION_CONFORMANCE_SHA256 = "c206e2bbbffd0e61b75d810a7c1760ef511ae5c73de671f688a71f4815bf555b"
RUNTIME_ATTEMPT_NAMESPACE = candidate.RUNTIME_ATTEMPT_NAMESPACE


def result_paths(row: dict[str, Any]) -> dict[str, Path]:
    stem = p2a.safe_slug(str(row.get("stem") or "task"))
    return {
        "run": ROOT
        / "reports"
        / f"theseus_p4v2r2r3_attempt1_{stem}_run.json",
        "evaluation": ROOT
        / "reports"
        / f"theseus_p4v2r2r3_attempt1_{stem}_evaluation.json",
    }


def runtime_reports(row: dict[str, Any]) -> list[Path]:
    task = p2a.read_json(ROOT / str(row.get("task") or ""))
    task_id = p2a.safe_slug(str(task.get("opaque_task_id") or ""))
    return sorted(
        (ROOT / "runtime" / "p2a").glob(
            f"*{task_id}*{RUNTIME_ATTEMPT_NAMESPACE}*.json"
        )
    )


def call_start_receipts(row: dict[str, Any]) -> list[Path]:
    task = p2a.read_json(ROOT / str(row.get("task") or ""))
    task_id = p2a.safe_slug(str(task.get("opaque_task_id") or ""))
    return sorted(
        candidate.CALL_START_DIRECTORY.glob(
            f"*{task_id}*{RUNTIME_ATTEMPT_NAMESPACE}*_call_*.json"
        )
    )


def audit_call_start_custody(paths: list[Path]) -> dict[str, Any]:
    faults: list[str] = []
    returned = 0
    runtime_paths: list[str] = []
    for index, path in enumerate(paths, start=1):
        receipt = p2a.read_json(path)
        if receipt.get("policy") != candidate.CALL_START_POLICY:
            faults.append(f"policy_invalid:{index}")
        if receipt.get("runtime_attempt_namespace") != RUNTIME_ATTEMPT_NAMESPACE:
            faults.append(f"namespace_invalid:{index}")
        if receipt.get("state") != "RETURNED_WITH_RUNTIME_RECEIPT":
            faults.append(f"not_returned:{index}")
            continue
        returned += 1
        report_path = p2a.resolve(str(receipt.get("runtime_report") or ""))
        runtime_paths.append(p2a.rel(report_path))
        if (
            not report_path.is_file()
            or p2a.sha256_file(report_path)
            != str(receipt.get("runtime_report_sha256") or "")
            or receipt.get("runtime_report_binding_valid") is not True
        ):
            faults.append(f"runtime_binding_invalid:{index}")
        if receipt.get("prompt_retained") is not False:
            faults.append(f"prompt_retention_invalid:{index}")
        if receipt.get("candidate_output_retained") is not False:
            faults.append(f"candidate_retention_invalid:{index}")
    if len(runtime_paths) != len(set(runtime_paths)):
        faults.append("runtime_path_reused")
    return {
        "passed": not faults,
        "faults": faults,
        "receipts": len(paths),
        "returned_receipts": returned,
        "runtime_paths": runtime_paths,
    }


@contextmanager
def bind_predecessor() -> Iterator[None]:
    rebound = {
        "POLICY": POLICY,
        "POOL": POOL,
        "INSTRUMENT": INSTRUMENT,
        "ROUTE_CANARY": ROUTE_CANARY,
        "PROGRESS": PROGRESS,
        "POOL_SHA256": POOL_SHA256,
        "INSTRUMENT_SHA256": INSTRUMENT_SHA256,
        "ROUTE_CANARY_SHA256": ROUTE_CANARY_SHA256,
        "RUNTIME_ATTEMPT_NAMESPACE": RUNTIME_ATTEMPT_NAMESPACE,
        "candidate": candidate,
        "result_paths": result_paths,
        "runtime_reports": runtime_reports,
    }
    original = {name: getattr(predecessor, name) for name in rebound}
    try:
        for name, value in rebound.items():
            setattr(predecessor, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(predecessor, name, value)


def audit_campaign(extra_faults: list[str] | None = None) -> dict[str, Any]:
    with bind_predecessor():
        report = predecessor.audit_campaign(extra_faults)
    faults = p2a.strings(report.get("faults"))
    conformance = (
        p2a.read_json(PRODUCTION_CONFORMANCE)
        if PRODUCTION_CONFORMANCE.is_file()
        else {}
    )
    if (
        p2a.sha256_file(PRODUCTION_CONFORMANCE) != PRODUCTION_CONFORMANCE_SHA256
        or conformance.get("trigger_state") != "GREEN"
        or p2a.strings(conformance.get("faults"))
        or int(conformance.get("candidate_or_control_calls") or 0) != 0
        or int(conformance.get("hidden_evaluator_calls") or 0) != 0
        or conformance.get("project_selected_quality_token_cap") is not None
        or int(p2a.mapping(conformance.get("coverage")).get("frozen_tasks") or 0)
        != 10
        or p2a.mapping(conformance.get("prompt_addressability")).get(
            "trigger_state"
        )
        != "GREEN"
    ):
        faults.append("production_conformance_binding_invalid")
    pool = p2a.read_json(POOL)
    task_rows = {
        str(row.get("stem") or ""): row for row in p2a.dicts(report.get("tasks"))
    }
    total_starts = 0
    for row in p2a.dicts(pool.get("tasks")):
        stem = str(row.get("stem") or "")
        paths = result_paths(row)
        starts = call_start_receipts(row)
        custody = audit_call_start_custody(starts)
        total_starts += len(starts)
        if starts and not paths["run"].is_file():
            faults.append(f"partial_call_start_without_sealed_run:{stem}")
        if paths["run"].is_file() and (
            len(starts) != 6 or custody.get("passed") is not True
        ):
            faults.append(f"call_start_custody_invalid:{stem}")
            faults.extend(
                f"call_start_custody:{stem}:{fault}" for fault in custody["faults"]
            )
        if stem in task_rows:
            task_rows[stem]["pre_inference_call_start_receipts"] = len(starts)
            task_rows[stem]["returned_call_start_receipts"] = custody[
                "returned_receipts"
            ]
    report["faults"] = sorted(set(faults))
    report["trigger_state"] = "GREEN" if not report["faults"] else "RED"
    report["pre_inference_call_start_receipts"] = total_starts
    report["production_conformance"] = {
        "path": p2a.rel(PRODUCTION_CONFORMANCE),
        "sha256": p2a.sha256_file(PRODUCTION_CONFORMANCE),
        "trigger_state": conformance.get("trigger_state"),
        "candidate_or_control_calls": conformance.get(
            "candidate_or_control_calls"
        ),
        "hidden_evaluator_calls": conformance.get("hidden_evaluator_calls"),
        "minimum_context_residual_tokens": p2a.mapping(
            conformance.get("prompt_addressability")
        ).get("minimum_context_residual_tokens"),
    }
    report["policy"] = POLICY
    report["prompt_continuity"] = {
        "complete_first_call_artifact_retained": True,
        "same_rule_all_learned_arms": True,
        "project_selected_first_artifact_character_cap": None,
        "project_selected_first_artifact_token_cap": None,
    }
    return report


def run_campaign() -> dict[str, Any]:
    with bind_predecessor():
        predecessor_report = predecessor.run_campaign()
    report = audit_campaign(p2a.strings(predecessor_report.get("faults")))
    report["prompt_continuity"] = {
        "complete_first_call_artifact_retained": True,
        "same_rule_all_learned_arms": True,
        "project_selected_first_artifact_character_cap": None,
        "project_selected_first_artifact_token_cap": None,
    }
    return report


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
                key: report.get(key)
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
    return 0 if report.get("trigger_state") == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
