#!/usr/bin/env python3
"""Terminalize the externally interrupted P4-v2r2-r1 campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation_evaluator as evaluator  # noqa: E402
import theseus_p4s_disposition as disposition_base  # noqa: E402
import theseus_p4v2r2r1_campaign as campaign  # noqa: E402


POLICY = "project_theseus_p4v2r2r1_interruption_disposition_v1"
POOL = ROOT / "configs" / "theseus_p4v2r2r1_task_pool.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2r1_cognitive_compilation_instrument.json"
ACTIVE_LEASE = ROOT / "runtime" / "control" / "theseus_p4v2r2r1_campaign_lease.json"
LEASE_ARCHIVE = ROOT / "reports" / "theseus_p4v2r2r1_campaign_leases"
OUT = ROOT / "reports" / "theseus_p4v2r2r1_attempt1_terminal_disposition.json"
EXPECTED_COMPLETE_TASKS = 6
EXPECTED_PARTIAL_TASK_INDEX = 7
EXPECTED_MODEL_CALLS = 37


def process_active() -> bool:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,command="],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if completed.returncode != 0:
        return True
    needles = (
        "theseus_p4v2r2r1_campaign.py",
        "theseus_p4v2r2r1_autonomous_launch.py --execute",
    )
    return any(any(needle in line for needle in needles) for line in completed.stdout.splitlines())


def lease_owner() -> Path:
    if ACTIVE_LEASE.is_file():
        return ACTIVE_LEASE
    archives = sorted(LEASE_ARCHIVE.glob("*.json")) if LEASE_ARCHIVE.is_dir() else []
    return archives[-1] if archives else ACTIVE_LEASE


def replay_projection(value: Any) -> Any:
    """Canonicalize only evaluator temp-root nonces, including truncated tails."""
    return normalize_evaluator_nonce(disposition_base.stable_evaluation_projection(value))


def normalize_evaluator_nonce(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_evaluator_nonce(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_evaluator_nonce(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"s-p4-archive-audit-[^/\\\"]+", "s-p4-archive-audit-<NONCE>", value)
    return value


def build_report(*, process_active_override: bool | None = None) -> dict[str, Any]:
    faults: list[str] = []
    pool = p2a.read_json(POOL)
    rows = p2a.dicts(pool.get("tasks"))
    active = process_active() if process_active_override is None else process_active_override
    if active:
        faults.append("campaign_process_still_active")

    lease_path = lease_owner()
    lease = p2a.read_json(lease_path) if lease_path.is_file() else {}
    if lease.get("policy") != "project_theseus_p4v2r2r1_autonomous_launch_v1":
        faults.append("lease_policy_invalid")
    if lease.get("state") not in {"RUNNING", "STOPPED_RETAIN_EVIDENCE"}:
        faults.append("lease_state_invalid")

    audit = campaign.audit_campaign()
    expected_audit_fault = "partial_unsealed_runtime_receipts:p4v2r2_08_pylsp_715"
    if audit.get("trigger_state") != "RED" or audit.get("faults") != [expected_audit_fault]:
        faults.append("campaign_interruption_shape_invalid")
    if int(audit.get("complete_tasks") or 0) != EXPECTED_COMPLETE_TASKS:
        faults.append("complete_task_denominator_invalid")
    if int(audit.get("model_calls_retained") or 0) != EXPECTED_COMPLETE_TASKS * 6:
        faults.append("complete_call_denominator_invalid")

    completed_tasks: list[dict[str, Any]] = []
    treatment_useful = 0
    for row in rows[:EXPECTED_COMPLETE_TASKS]:
        stem = str(row.get("stem") or "")
        paths = campaign.result_paths(row)
        receipts = campaign.runtime_reports(row)
        custody = campaign.route_custody(receipts)
        if len(receipts) != 6 or custody.get("passed") is not True:
            faults.append(f"completed_task_custody_invalid:{stem}")
        if not paths["run"].is_file() or not paths["evaluation"].is_file():
            faults.append(f"completed_task_artifact_missing:{stem}")
            continue
        stored = p2a.read_json(paths["evaluation"])
        replayed = evaluator.evaluate_report(
            paths["run"], ROOT / str(row.get("evaluator") or "")
        )
        replay_match = replay_projection(stored) == replay_projection(replayed)
        if not replay_match:
            faults.append(f"blind_evaluator_replay_mismatch:{stem}")
        result_by_arm = {
            str(result.get("arm_id") or ""): result
            for result in p2a.dicts(stored.get("results"))
        }
        treatment_useful += int(
            p2a.mapping(result_by_arm.get("typed_semantic_ir_treatment")).get("useful")
            or 0
        )
        completed_tasks.append(
            {
                "campaign_index": row.get("campaign_index"),
                "stem": stem,
                "run": disposition_base.source_identity(paths["run"]),
                "evaluation": disposition_base.source_identity(paths["evaluation"]),
                "runtime_receipts": len(receipts),
                "route_custody_green": custody.get("passed") is True,
                "blind_evaluator_replay_match": replay_match,
            }
        )

    partial_row = rows[EXPECTED_PARTIAL_TASK_INDEX - 1]
    partial_receipts = campaign.runtime_reports(partial_row)
    partial_custody = campaign.route_custody(partial_receipts)
    if len(partial_receipts) != 1 or partial_custody.get("passed") is not True:
        faults.append("partial_task_custody_invalid")
    if campaign.result_paths(partial_row)["run"].is_file():
        faults.append("partial_task_run_was_sealed")
    partial_runtime = partial_receipts[0] if len(partial_receipts) == 1 else ROOT
    partial_backend_path = ROOT
    if partial_runtime.is_file():
        runtime = p2a.read_json(partial_runtime)
        partial_backend_path = ROOT / str(
            p2a.mapping(runtime.get("checkpoint_chat")).get("out") or ""
        )

    unseen = []
    for row in rows[EXPECTED_PARTIAL_TASK_INDEX:]:
        receipts = campaign.runtime_reports(row)
        paths = campaign.result_paths(row)
        if receipts or paths["run"].exists() or paths["evaluation"].exists():
            faults.append(f"candidate_unseen_task_consumed:{row.get('stem')}")
        unseen.append(str(row.get("stem") or ""))

    model_calls = EXPECTED_COMPLETE_TASKS * 6 + len(partial_receipts)
    if model_calls != EXPECTED_MODEL_CALLS:
        faults.append("total_model_call_denominator_invalid")

    valid = not faults
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if valid else "RED",
        "scientific_status": "INCONCLUSIVE_EXPERIMENT" if valid else "P4V2R2R1_REVIEW_REQUIRED",
        "faults": sorted(set(faults)),
        "interruption": {
            "kind": "external_execution_session_disappeared",
            "campaign_process_active": active,
            "same_denominator_resume_authorized": False,
            "partially_consumed_task_replay_authorized": False,
        },
        "source_identities": {
            "pool": disposition_base.source_identity(POOL),
            "instrument": disposition_base.source_identity(INSTRUMENT),
            "campaign_progress": disposition_base.source_identity(campaign.PROGRESS),
            "lease": disposition_base.source_identity(lease_path) if lease_path.is_file() else {},
            "partial_runtime_receipt": disposition_base.source_identity(partial_runtime) if partial_runtime.is_file() else {},
            "partial_backend_receipt": disposition_base.source_identity(partial_backend_path) if partial_backend_path.is_file() else {},
        },
        "denominators": {
            "sealed_tasks": 10,
            "complete_tasks": len(completed_tasks),
            "partially_consumed_tasks": len(partial_receipts),
            "candidate_unseen_tasks": len(unseen),
            "learned_model_calls": model_calls,
            "complete_matched_model_calls": EXPECTED_COMPLETE_TASKS * 6,
            "partial_task_model_calls": len(partial_receipts),
            "physical_context_boundary_hits": 0,
            "project_selected_quality_token_cap": None,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "completed_tasks": completed_tasks,
        "partial_task": {
            "campaign_index": partial_row.get("campaign_index"),
            "stem": partial_row.get("stem"),
            "runtime_receipts": len(partial_receipts),
            "route_custody_green": partial_custody.get("passed") is True,
        },
        "candidate_unseen_task_stems": unseen,
        "interim_observation": {
            "complete_task_count": len(completed_tasks),
            "typed_semantic_ir_treatment_useful": treatment_useful,
            "terminal_mechanism_decision_authorized": False,
        },
        "adequacy": {
            "information_flow_green_on_complete_tasks": valid,
            "mechanics_floor_established": False,
            "experiment_floor_established": False,
            "reason": "sealed_ten_task_denominator_interrupted_after_one_call_on_task_seven",
        },
        "next_stage": {
            "D1_eligible": False,
            "book_support_state_effect": "none",
            "state": "FRESH_PROSPECTIVELY_SEALED_P4_RECOVERY_REQUIRED",
        },
        "maximum_inference": (
            "The six complete tasks and one route-clean partial call are valid retained development evidence, "
            "but the interrupted ten-task denominator cannot establish a survivor, adequate no-survivor result, "
            "or broader conclusion about cognitive compilation. It grants no D1, D2, training, serving, hosted-control, "
            "or ASI Stack support-promotion authority."
        ),
    }


def archive_orphaned_lease(report: dict[str, Any]) -> Path:
    if report.get("trigger_state") != "GREEN" or report.get("interruption", {}).get("campaign_process_active") is not False:
        raise RuntimeError("interruption disposition must be GREEN with no live process")
    lease = p2a.read_json(ACTIVE_LEASE)
    if lease.get("state") != "RUNNING":
        raise RuntimeError("active lease is not the stranded RUNNING lease")
    lease.update(
        {
            "state": "STOPPED_RETAIN_EVIDENCE",
            "completed_utc": p2a.now(),
            "stop_reason": "external_execution_session_disappeared_after_partial_task_call",
            "terminal_scientific_status": "INCONCLUSIVE_EXPERIMENT",
        }
    )
    p2a.write_json(ACTIVE_LEASE, lease)
    LEASE_ARCHIVE.mkdir(parents=True, exist_ok=True)
    archive = LEASE_ARCHIVE / f"{lease['lease_id']}.json"
    os.replace(ACTIVE_LEASE, archive)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-orphaned-lease", action="store_true")
    parser.add_argument("--out", default=p2a.rel(OUT))
    args = parser.parse_args()
    report = build_report()
    if args.archive_orphaned_lease and report.get("trigger_state") == "GREEN" and ACTIVE_LEASE.is_file():
        archive_orphaned_lease(report)
        report = build_report(process_active_override=False)
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({"trigger_state":report["trigger_state"],"scientific_status":report["scientific_status"],"complete_tasks":report["denominators"]["complete_tasks"],"learned_model_calls":report["denominators"]["learned_model_calls"],"candidate_unseen_tasks":report["denominators"]["candidate_unseen_tasks"],"faults":report["faults"]},indent=2,sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
