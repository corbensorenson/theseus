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
PROGRESS = ROOT / "reports" / "theseus_p4v2r2r3_attempt1_campaign_progress.json"
POOL_SHA256 = "7c78025dcbad76a637016a8287e2fd6c94b7f1dc580959a8c9fd3c4e1215ef1f"
INSTRUMENT_SHA256 = "b3df79d20a8cdf266101df949178a82cea252e1f8fbdad0a301dae3a2eaf1945"
ROUTE_CANARY_SHA256 = "14b3c8775b835972d4866e9cad2ed3d64928d600583f266e832201bcf50a15be"
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
        report = predecessor.run_campaign()
    report["policy"] = POLICY
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
