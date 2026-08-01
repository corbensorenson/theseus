#!/usr/bin/env python3
"""Build and audit the prospectively resealed P4 recovery instrument."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2_cognitive_compilation as candidate  # noqa: E402


BASE = ROOT / "configs" / "theseus_p4v2r2_cognitive_compilation_instrument.json"
POOL = ROOT / "configs" / "theseus_p4v2r2r1_task_pool.json"
OUT = ROOT / "configs" / "theseus_p4v2r2r1_cognitive_compilation_instrument.json"
BASE_SHA256 = "046f989e7eaa444a64e38d96fae8def10dabc4a6c3b1b2b2de3118ba4fe5d6ff"


def identity(path: str) -> dict[str, str]:
    resolved = ROOT / path
    return {"path": path, "sha256": p2a.sha256_file(resolved)}


def build() -> dict[str, Any]:
    if p2a.sha256_file(BASE) != BASE_SHA256:
        raise ValueError("base instrument binding changed")
    value = p2a.read_json(BASE)
    value.update(
        {
            "campaign_id": "ASI-THESEUS-FLAGSHIP-01-P4V2R2R1-RECOVERY",
            "runtime_attempt_namespace": "p4v2r2r1_attempt1",
            "state": "PROSPECTIVELY_RESEALED_AFTER_ROUTE_IMPLEMENTATION_FAILURE",
            "purpose": "Run the honest P4 recovery successor with frozen TMax held fixed: nine independently retained candidate-unseen tasks plus one new licensed source-disjoint task, exact v2 route binding, immutable per-call backend telemetry, and no project-selected generation cap.",
            "recovery_predecessor": {
                "interruption": identity("reports/theseus_p4v2r2_attempt2_interruption.json"),
                "terminal_disposition": identity("reports/theseus_p4v2r2_attempt2_terminal_disposition.json"),
                "scientific_status": "INCONCLUSIVE_IMPLEMENTATION",
                "consumed_tasks_excluded": 1,
                "candidate_unseen_tasks_carried": 9,
                "same_denominator_resume_authorized": False,
            },
            "sealed_task_pool": identity("configs/theseus_p4v2r2r1_task_pool.json"),
            "route_canary_contract": {
                "report": "reports/theseus_p4v2r2r1_route_canary.json",
                "required_state_before_task_calls": "GREEN",
                "real_frozen_model_call_required": True,
                "task_prompt_or_source_visible": False,
                "task_denominator_consumed": 0,
                "candidate_or_control_output": False,
                "route_backend_policy": "project_theseus_local_inference_backend_v2",
                "route_receipt_policy": "project_theseus_assistant_route_integrity_v2",
            },
        }
    )
    harness_paths = {
        "candidate_runner": "scripts/theseus_p4v2r2_cognitive_compilation.py",
        "blind_evaluator": "scripts/theseus_p4_cognitive_compilation_evaluator.py",
        "assistant_runtime": "scripts/theseus_assistant_runtime.py",
        "assistant_runtime_config": "configs/theseus_assistant_runtime.json",
        "route_integrity_base": "scripts/theseus_assistant_route_integrity.py",
        "route_integrity_v2": "scripts/theseus_assistant_route_integrity_v2.py",
        "local_backend": "scripts/theseus_local_inference_backend_v2.py",
        "semantic_parser": "scripts/theseus_semantic_ir_v2r2.py",
    }
    value["harness"] = {
        key: path
        for key, path in harness_paths.items()
    }
    value["harness"].update(
        {f"{key}_sha256": p2a.sha256_file(ROOT / path) for key, path in harness_paths.items()}
    )
    fresh = dict(value.get("fresh_task_pool_contract") or {})
    fresh.update(
        {
            "all_tasks_new_after_instrument_freeze": False,
            "nine_tasks_candidate_unseen_from_consumed_predecessor": True,
            "one_replacement_new_after_predecessor_failure": True,
            "replacement_source_disjoint_from_all_prior_development_sources": True,
            "predecessor_consumed_task_excluded": True,
            "selection_may_condition_on_candidate_or_control_output": False,
        }
    )
    value["fresh_task_pool_contract"] = fresh
    matched = dict(value.get("matched_arm_contract") or {})
    matched["immutable_backend_receipt_per_model_call"] = True
    matched["stop_immediately_on_any_route_receipt_failure"] = True
    value["matched_arm_contract"] = matched
    value["boundaries"] = {
        **dict(value.get("boundaries") or {}),
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "training_rows_written": 0,
        "user_gate": "none",
        "project_selected_quality_token_cap": None,
        "ASI_stack_support_state_effect": "none",
    }
    value["maximum_inference"] = (
        "A terminal result can decide only whether this exact recovered TMax plus "
        "Semantic-IR v2r2 implementation is a development survivor eligible for "
        "one fresh D1 qualification. It cannot establish general cognitive "
        "compilation, serving qualification, training eligibility, D2, or book support."
    )
    return value


def audit(value: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    path_faults = candidate.audit_instrument(OUT).get("faults", []) if OUT.is_file() else ["instrument_missing"]
    faults.extend(str(row) for row in path_faults)
    for owner, path in value.get("harness", {}).items():
        if owner.endswith("_sha256"):
            continue
        if p2a.sha256_file(ROOT / str(path)) != value["harness"].get(f"{owner}_sha256"):
            faults.append(f"harness_binding_invalid:{owner}")
    pool = p2a.read_json(POOL)
    if (
        pool.get("state") != "SEALED_BEFORE_SUCCESSOR_CANDIDATE_GENERATION"
        or pool.get("faults") != []
        or int(pool.get("task_count") or 0) != 10
    ):
        faults.append("recovery_pool_invalid")
    if value.get("sealed_task_pool", {}).get("sha256") != p2a.sha256_file(POOL):
        faults.append("recovery_pool_binding_invalid")
    if value.get("generation_budget", {}).get("project_selected_quality_token_cap") is not None:
        faults.append("quality_token_cap_present")
    return {
        "policy": "project_theseus_p4v2r2r1_instrument_audit_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "instrument": identity(str(OUT.relative_to(ROOT))),
        "task_pool": identity(str(POOL.relative_to(ROOT))),
        "runtime_attempt_namespace": value.get("runtime_attempt_namespace"),
        "project_selected_quality_token_cap": value.get("generation_budget", {}).get("project_selected_quality_token_cap"),
        "task_candidate_or_control_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        p2a.write_json(OUT, build())
    value = p2a.read_json(OUT)
    report = audit(value)
    p2a.write_json(ROOT / "reports" / "theseus_p4v2r2r1_instrument_audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
