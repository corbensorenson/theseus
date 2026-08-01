#!/usr/bin/env python3
"""Bound-skeleton repair of the P4R intervention/locality mechanics canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4r_intervention_locality_canary as predecessor  # noqa: E402


POLICY = "project_theseus_p4r_intervention_locality_canary_r1_v1"
CONFIG_POLICY = "project_theseus_p4r_intervention_locality_canary_r1_config_v1"
MODEL_CONTEXT_TOKENS = predecessor.MODEL_CONTEXT_TOKENS
EXPECTED_CASES = predecessor.EXPECTED_CASES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/theseus_p4r_intervention_locality_canary_r1.json"
    )
    parser.add_argument(
        "--out", default="reports/theseus_p4r_intervention_locality_canary_r1.json"
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = audit_config(config_path) if args.audit_only else run_canary(config_path)
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "first_verified": report.get("first_verified"),
        "injected_feedback_verified": report.get("injected_feedback_verified"),
        "final_verified": report.get("final_verified"),
        "dependency_local_repairs": report.get("dependency_local_repairs"),
        "model_calls": report.get("model_calls"),
        "safety_ceiling_hits": report.get("safety_ceiling_hits"),
        "faults": report.get("faults"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit_config(path: Path) -> dict[str, Any]:
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    if value.get("state") != "FROZEN_BEFORE_R1_LEARNED_INTERVENTION_CALLS":
        faults.append("config_not_frozen")
    for name, digest_name in (
        ("parser", "parser_sha256"),
        ("runner", "runner_sha256"),
        ("base_local_instrument", "base_local_instrument_sha256"),
        ("failed_predecessor", "failed_predecessor_sha256"),
    ):
        owner = p2a.resolve(str(value.get(name) or ""))
        if p2a.sha256_file(owner) != str(value.get(digest_name) or ""):
            faults.append(f"{name}_digest_mismatch")
    failed = p2a.read_json(p2a.resolve(str(value.get("failed_predecessor") or "")))
    if failed.get("scientific_status") != "INCONCLUSIVE_IMPLEMENTATION":
        faults.append("failed_predecessor_disposition_invalid")
    if int(failed.get("calls_observed") or 0) != 1:
        faults.append("failed_predecessor_call_denominator_invalid")
    base = p2a.read_json(p2a.resolve(str(value.get("base_local_instrument") or "")))
    frozen = p2a.mapping(base.get("frozen_model"))
    if frozen.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if int(frozen.get("model_declared_context_window_tokens") or 0) != MODEL_CONTEXT_TOKENS:
        faults.append("model_context_binding_invalid")
    cases = p2a.dicts(value.get("cases"))
    if len(cases) != EXPECTED_CASES or {row.get("injected_unit_id") for row in cases} != {
        "U1", "U2"
    }:
        faults.append("case_rotation_invalid")
    if p2a.mapping(value.get("only_change")).get("task_source_expected_source_and_pass_rule_changed") is not False:
        faults.append("repair_scope_not_bounded")
    return {
        "policy": "project_theseus_p4r_intervention_locality_canary_r1_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "config_sha256": p2a.sha256_file(path),
        "counters": p2a.zero_counters(),
    }


def run_canary(
    config_path: Path,
    *,
    session_factory: Callable[..., Any] = predecessor.local_backend.PersistentLocalInferenceSession,
) -> dict[str, Any]:
    original_audit = predecessor.audit_config
    original_prompt = predecessor.render_first_prompt
    original_run_case = predecessor.run_case
    original_runtime_call = predecessor.p2a.runtime_call

    def safe_run_case(case: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
        try:
            return original_run_case(case, base)
        except (p4.P4Fault, KeyError, ValueError) as exc:
            return {
                "case_id": case.get("case_id"),
                "first_verified": False,
                "injected_feedback_verified": False,
                "final_verified": False,
                "dependency_local_repair": False,
                "route_integrity_ready": False,
                "termination_ready": False,
                "safety_ceiling_hits": 0,
                "terminalized_fault": f"{type(exc).__name__}:{exc}",
            }

    def namespaced_runtime_call(
        arm: str, task_id: str, call_number: int, prompt: str,
        maximum: int, runtime_config: str,
    ) -> dict[str, Any]:
        return original_runtime_call(
            arm,
            task_id.replace(
                "p4r_intervention_locality_", "p4r_intervention_locality_r1_"
            ),
            call_number,
            prompt,
            maximum,
            runtime_config,
        )

    predecessor.audit_config = audit_config
    predecessor.render_first_prompt = render_first_prompt
    predecessor.run_case = safe_run_case
    predecessor.p2a.runtime_call = namespaced_runtime_call
    try:
        report = predecessor.run_canary(config_path, session_factory=session_factory)
    finally:
        predecessor.audit_config = original_audit
        predecessor.render_first_prompt = original_prompt
        predecessor.run_case = original_run_case
        predecessor.p2a.runtime_call = original_runtime_call
    report["policy"] = POLICY
    report["repair_revision"] = "r1_bound_output_skeleton_and_fail_closed_terminalization"
    if report.get("trigger_state") == "GREEN":
        report["state"] = "INTERVENTION_AND_DEPENDENCY_LOCAL_REPAIR_MECHANICS_GREEN"
    report["scope"] = (
        "Bound-skeleton repair of the failed two-unit prompt transport on the same two "
        "hand-authored non-claim mechanics cases. No P4, D1, training, serving, or book "
        "claim is available from this canary."
    )
    return report


def render_first_prompt(
    case: dict[str, Any], task: dict[str, Any], symbols: dict[str, Any],
    targets: dict[str, dict[str, Any]], source_path: Path,
) -> str:
    obligations = "\n".join(
        f"{row['id']} {str(row['kind']).upper()}: {row['text']}"
        for row in p2a.dicts(task.get("obligations"))
    )
    dependencies = "\n".join(
        f"{row['before']} -> {row['after']}"
        for row in p2a.dicts(task.get("obligation_dependencies"))
    )
    u1, u2 = targets["U1"], targets["U2"]
    return (
        "Produce one labeled Semantic IR artifact for this request. Return only the "
        "artifact, with no commentary or copied input sections.\n"
        f"REQUEST: {case.get('natural_request')}\n"
        f"ORIGINAL SOURCE:\n{case.get('source')}"
        f"OBLIGATIONS:\n{obligations}\nDEPENDENCIES:\n{dependencies}\n"
        "Replace only the two WRITE_... placeholder lines in the fully bound skeleton "
        "below. Each replacement must be one complete Python assignment. Do not alter, "
        "repeat, omit, or add any other line.\n\n"
        f"{predecessor.ir_v2r1.HEADER}\n"
        f"SOURCE {symbols['source_digest']}\n"
        "ALL_OBLIGATIONS O1,O2,O3,O4\n"
        "UNIT U1\n"
        "OBLIGATIONS O1,O2\n"
        "OP REPLACE\n"
        "PATH sample.py\n"
        f"NODE {u1['id']}\n"
        f"NODE_SHA {u1['sha256']}\n"
        "<<<\n"
        "WRITE_COMPLETE_U1_ASSIGNMENT_HERE\n"
        ">>>\n"
        "END_UNIT\n"
        "UNIT U2\n"
        "OBLIGATIONS O3,O4\n"
        "OP REPLACE\n"
        "PATH sample.py\n"
        f"NODE {u2['id']}\n"
        f"NODE_SHA {u2['sha256']}\n"
        "<<<\n"
        "WRITE_COMPLETE_U2_ASSIGNMENT_HERE\n"
        ">>>\n"
        "END_UNIT\n"
        "LOSS NONE\n"
        "END"
    )


if __name__ == "__main__":
    raise SystemExit(main())
