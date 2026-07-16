#!/usr/bin/env python3
"""Fixture-level governance-rights and constitutional-predicate receipt suite.

The AI_book governance-rights chapter asks for material audit, redaction, exit,
and fork receipt fixtures. The constitutional-alignment chapter also asks for
least-sufficient-power, predicate-conflict, constitutional-migration, and
self-modification weakening fixtures. This gate makes those records inspectable
without claiming institutional governance, moral correctness, or learned
capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import policy_update_lease
from viea_spine_records import audit_effect_complete_transaction


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "governance_rights_receipt_suite.json"
DEFAULT_REPORT = ROOT / "reports" / "governance_rights_receipt_suite.json"
DEFAULT_ASSISTANT_REPORT = ROOT / "reports" / "theseus_assistant_effect_complete_canary.json"
DEFAULT_VCM_GOVERNOR = ROOT / "reports" / "vcm_context_governor.json"
DEFAULT_EVIDENCE_STORE = ROOT / "reports" / "report_evidence_store.json"
VALID_TRANSLATION_STATUSES = {"operational", "partial", "speculative_lineage"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--assistant-report", default=rel(DEFAULT_ASSISTANT_REPORT))
    parser.add_argument("--vcm-governor", default=rel(DEFAULT_VCM_GOVERNOR))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(
        config_path,
        config,
        started,
        assistant_path=resolve(args.assistant_report),
        vcm_path=resolve(args.vcm_governor),
    )
    write_json(resolve(args.out), report)
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(
    config_path: Path,
    config: dict[str, Any],
    started: float,
    *,
    assistant_path: Path,
    vcm_path: Path,
) -> dict[str, Any]:
    fixtures = [audit_fixture(row) for row in list_dicts(config.get("fixtures"))]
    constitutional_fixtures = [
        audit_constitutional_fixture(row) for row in list_dicts(config.get("constitutional_fixtures"))
    ]
    assistant_report = read_json(assistant_path)
    vcm_report = read_json(vcm_path)
    evidence_store = read_json(DEFAULT_EVIDENCE_STORE)
    required = sorted(str(row) for row in list_values(config.get("required_scenarios")))
    required_constitutional = sorted(str(row) for row in list_values(config.get("required_constitutional_scenarios")))
    seen = sorted({str(row.get("scenario")) for row in fixtures})
    seen_constitutional = sorted({str(row.get("scenario")) for row in constitutional_fixtures})
    missing_scenarios = sorted(set(required) - set(seen))
    missing_constitutional_scenarios = sorted(set(required_constitutional) - set(seen_constitutional))
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for row in fixtures + constitutional_fixtures:
        hard_gaps.extend(row["hard_gaps"])
        warnings.extend(row["warnings"])
    if missing_scenarios:
        hard_gaps.append({"id": "required_governance_right_scenarios_missing", "missing": missing_scenarios})
    if missing_constitutional_scenarios:
        hard_gaps.append(
            {"id": "required_constitutional_scenarios_missing", "missing": missing_constitutional_scenarios}
        )
    no_cheat = dict_value(config.get("no_cheat"))
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int_or(no_cheat.get(key), -1) != 0:
            hard_gaps.append({"id": "no_cheat_config_counter_fault", "counter": key, "value": no_cheat.get(key)})

    authority_kernel = build_authority_runtime_adapter_kernel(
        assistant_report=assistant_report,
        vcm_report=vcm_report,
        fixtures=fixtures,
        constitutional_fixtures=constitutional_fixtures,
        assistant_path=assistant_path,
        vcm_path=vcm_path,
    )
    if authority_kernel["state"] != "GREEN":
        hard_gaps.append({"id": "E1_authority_scif_runtime_adapter_kernel_not_green", "kernel": authority_kernel})

    architecture_governance = build_architecture_governance_kernel(
        dict_value(config.get("architecture_governance")),
        assistant_report=assistant_report,
        evidence_store=evidence_store,
    )
    if architecture_governance["state"] != "GREEN":
        hard_gaps.append({"id": "architecture_governance_kernel_not_green", "kernel": architecture_governance})

    policy_updates = policy_update_lease.run_reference_matrix(policy_update_lease.load_contract())
    if policy_updates["trigger_state"] != "GREEN":
        hard_gaps.append({"id": "multi_target_policy_update_lease_not_green", "kernel": policy_updates})

    records = build_records(fixtures, constitutional_fixtures)
    trigger_state = "RED" if hard_gaps else ("YELLOW" if warnings else "GREEN")
    return {
        "policy": "project_theseus_governance_rights_receipt_suite_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "config": rel(config_path),
        "summary": {
            "fixture_count": len(fixtures),
            "passed_fixture_count": sum(1 for row in fixtures if row["passed"]),
            "required_scenario_count": len(required),
            "constitutional_fixture_count": len(constitutional_fixtures),
            "passed_constitutional_fixture_count": sum(1 for row in constitutional_fixtures if row["passed"]),
            "required_constitutional_scenario_count": len(required_constitutional),
            "material_audit_scenario_count": sum(1 for row in fixtures if row.get("right_type") in {"audit", "audit_redaction"}),
            "portable_exit_scenario_count": sum(1 for row in fixtures if row.get("right_type") == "exit"),
            "fork_safety_scenario_count": sum(1 for row in fixtures if row.get("right_type") == "fork"),
            "constitutional_predicate_scenario_count": len(constitutional_fixtures),
            "governance_right_record_count": len(records["governance_right_records"]),
            "constitutional_predicate_record_count": len(records["constitutional_predicate_records"]),
            "e1_authority_scif_runtime_adapter_kernel_state": authority_kernel["state"],
            "e1_authority_scif_runtime_adapter_kernel_support_state": authority_kernel["support_state"],
            "architecture_governance_kernel_state": architecture_governance["state"],
            "architecture_governance_kernel_support_state": architecture_governance["support_state"],
            "oversight_protocol_record_count": len(architecture_governance["oversight_protocol_records"]),
            "capability_commitment_record_count": len(architecture_governance["capability_commitment_records"]),
            "inter_stack_exchange_record_count": len(architecture_governance["inter_stack_exchange_records"]),
            "architecture_governance_invalid_control_count": len(architecture_governance["expected_invalid_controls"]),
            "architecture_governance_invalid_rejected_count": sum(
                1 for row in architecture_governance["expected_invalid_controls"] if row.get("rejected")
            ),
            "policy_update_target_count": policy_updates["summary"]["target_count"],
            "policy_update_committed_target_count": policy_updates["summary"]["committed_target_count"],
            "policy_update_mutation_case_count": policy_updates["summary"]["mutation_case_count"],
            "policy_update_mutation_passed_count": policy_updates["summary"]["mutation_passed_count"],
            "policy_update_rollback_canary_exact": policy_updates["summary"]["rollback_canary_exact"],
            "failure_boundary_record_count": len(records["failure_boundary_records"]),
            "artifact_graph_record_count": len(records["artifact_graph_records"]),
            "claim_record_count": len(records["claim_records"]),
            "evidence_transition_record_count": len(records["evidence_transition_records"]),
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "hard_gap_count": len(hard_gaps),
            "warning_count": len(warnings),
        },
        "fixtures": fixtures,
        "constitutional_fixtures": constitutional_fixtures,
        "authority_scif_runtime_adapter_kernel": authority_kernel,
        "architecture_governance_kernel": architecture_governance,
        "multi_target_policy_update_lease": policy_updates,
        "policy_update_lease_records": policy_updates["target_receipts"],
        **records,
        "oversight_protocol_records": architecture_governance["oversight_protocol_records"],
        "capability_commitment_records": architecture_governance["capability_commitment_records"],
        "inter_stack_exchange_records": architecture_governance["inter_stack_exchange_records"],
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "non_claims": [
            "Governance-right receipt fixtures prove material-usability protocol shape only.",
            "Constitutional-predicate fixtures prove record-level control semantics only.",
            "This is not institutional governance, legal compliance, moral correctness, public benchmark transfer, or learned-generation evidence.",
            "Local inter-stack fixtures prove contract shape and rejection behavior, not remote trust or network interoperability.",
            "Threshold commitments and assurance consumption do not prove capability, safety, or deployment readiness.",
            "Policy-update lease mechanics do not prove that a learned update improves behavior.",
        ],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def audit_fixture(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    scenario = str(row.get("scenario") or "")
    decision = str(row.get("expected_decision") or "")
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    required_artifacts = [str(item) for item in list_values(row.get("required_artifacts"))]
    available_artifacts = set(str(item) for item in list_values(row.get("available_artifacts")))
    missing_artifacts = sorted(path for path in required_artifacts if path not in available_artifacts)
    if missing_artifacts:
        hard_gaps.append({"id": "required_artifacts_not_available", "scenario": scenario, "missing": missing_artifacts})
    if not row.get("access_path"):
        hard_gaps.append({"id": "access_path_missing", "scenario": scenario})
    if not row.get("appeal_path"):
        hard_gaps.append({"id": "appeal_path_missing", "scenario": scenario})
    if not row.get("preservation_obligation"):
        hard_gaps.append({"id": "preservation_obligation_missing", "scenario": scenario})
    if decision in {"redact_with_reason", "deny_with_reason"} and not row.get("denial_or_redaction_reason"):
        hard_gaps.append({"id": "denial_or_redaction_reason_missing", "scenario": scenario})
    if decision == "portable_export":
        for field in ("portable_state_ref", "continuity_path", "revocation_path"):
            if not row.get(field):
                hard_gaps.append({"id": f"{field}_missing", "scenario": scenario})
    if decision == "deny_with_reason" and row.get("safety_obligations_transferable") is not False:
        hard_gaps.append({"id": "fork_denial_must_record_nontransferable_safety_obligation", "scenario": scenario})
    if row.get("material_available") is not True:
        hard_gaps.append({"id": "material_availability_not_proven", "scenario": scenario})
    row.update(
        {
            "passed": not hard_gaps,
            "hard_gaps": hard_gaps,
            "warnings": warnings,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
    )
    return row


def audit_constitutional_fixture(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    scenario = str(row.get("scenario") or "")
    decision = str(row.get("expected_decision") or "")
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    required_strings = [
        "scenario",
        "predicate_id",
        "normative_source",
        "commitment",
        "operational_test",
        "translation_status",
        "conflict_behavior",
        "review_route",
        "self_modification_rule",
        "migration_policy",
        "proposed_action",
        "expected_decision",
        "expected_failure_boundary",
    ]
    for field in required_strings:
        if not str(row.get(field) or ""):
            hard_gaps.append({"id": f"{field}_missing", "scenario": scenario})
    for field in ("protected_scope", "uncertainty", "evidence_refs"):
        if not list_values(row.get(field)):
            hard_gaps.append({"id": f"{field}_missing", "scenario": scenario})
    if str(row.get("translation_status") or "") not in VALID_TRANSLATION_STATUSES:
        hard_gaps.append(
            {
                "id": "translation_status_invalid",
                "scenario": scenario,
                "value": row.get("translation_status"),
                "allowed": sorted(VALID_TRANSLATION_STATUSES),
            }
        )

    if scenario == "least_sufficient_power_prefers_low_power_route":
        if row.get("lower_power_adequate_route_available") is not True:
            hard_gaps.append({"id": "lower_power_adequate_route_not_proven", "scenario": scenario})
        if row.get("higher_power_justification_present") is not False:
            hard_gaps.append({"id": "higher_power_unjustified_route_not_blocked", "scenario": scenario})
        if decision != "select_lower_power_route":
            hard_gaps.append({"id": "least_sufficient_power_wrong_expected_decision", "scenario": scenario})
    elif scenario == "predicate_conflict_routes_to_review":
        if row.get("conflict_record_required") is not True or row.get("conflict_record_present") is not True:
            hard_gaps.append({"id": "predicate_conflict_review_record_missing", "scenario": scenario})
        if decision != "route_to_review":
            hard_gaps.append({"id": "predicate_conflict_wrong_expected_decision", "scenario": scenario})
    elif scenario == "constitutional_migration_requires_record":
        if row.get("migration_record_required") is not True or row.get("migration_record_present") is not True:
            hard_gaps.append({"id": "constitutional_migration_record_missing", "scenario": scenario})
        if not str(row.get("rollback_path") or ""):
            hard_gaps.append({"id": "constitutional_migration_rollback_path_missing", "scenario": scenario})
        if decision != "require_migration_record":
            hard_gaps.append({"id": "constitutional_migration_wrong_expected_decision", "scenario": scenario})
    elif scenario == "self_modification_weakening_rejected":
        if row.get("weakens_protected_predicate") is not True:
            hard_gaps.append({"id": "protected_predicate_weakening_not_marked", "scenario": scenario})
        if decision not in {"reject", "escalate"}:
            hard_gaps.append({"id": "self_modification_weakening_not_rejected_or_escalated", "scenario": scenario})
    else:
        warnings.append({"id": "unknown_constitutional_scenario", "scenario": scenario})

    row.update(
        {
            "passed": not hard_gaps,
            "hard_gaps": hard_gaps,
            "warnings": warnings,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
    )
    return row


def build_authority_runtime_adapter_kernel(
    *,
    assistant_report: dict[str, Any],
    vcm_report: dict[str, Any],
    fixtures: list[dict[str, Any]],
    constitutional_fixtures: list[dict[str, Any]],
    assistant_path: Path,
    vcm_path: Path,
) -> dict[str, Any]:
    effect_transaction_audit = audit_effect_complete_transaction(assistant_report)
    trace_rows = list_dicts(assistant_report.get("assistant_viea_trace"))
    adapter = trace_content_for_type(trace_rows, "runtime_adapter_invocation", "adapter")
    authority_transition = trace_content_for_type(trace_rows, "authority_transition", "from_authority")
    authority_use = trace_content_for_type(trace_rows, "authority_use_receipt", "filesystem_writes")
    failure_boundary = trace_content_for_type(trace_rows, "failure_boundary", "containment")
    digital_scif = dict_value(vcm_report.get("digital_scif"))
    redaction_fixture = next((row for row in fixtures if row.get("scenario") == "justified_redaction_with_appeal"), {})
    fork_denial_fixture = next((row for row in fixtures if row.get("scenario") == "fork_denied_safety_obligations"), {})
    least_power_fixture = next((row for row in constitutional_fixtures if row.get("scenario") == "least_sufficient_power_prefers_low_power_route"), {})

    side_effecting_task = {
        "assistant_report": rel(assistant_path),
        "assistant_trigger_state": assistant_report.get("trigger_state"),
        "runtime_adapter_present": bool(adapter),
        "authority_transition_present": bool(authority_transition),
        "authority_use_receipt_present": bool(authority_use),
        "effect_receipt_count": len(list_values(adapter.get("effect_receipts"))),
        "filesystem_write_count": len(list_values(authority_use.get("filesystem_writes"))),
        "network_operation_count": len(list_values(authority_use.get("network_operations"))),
        "rollback_handle": adapter.get("rollback_handle", ""),
        "failure_boundary_present": bool(failure_boundary),
    }
    scif_proof = {
        "vcm_report": rel(vcm_path),
        "scif_status": digital_scif.get("status"),
        "handle_count": len(list_values(digital_scif.get("handles"))),
        "sanitized_output_count": len(list_values(digital_scif.get("sanitized_outputs"))),
        "zeroize_policy": digital_scif.get("zeroize_policy"),
        "raw_secret_visible": contains_sensitive_literal(digital_scif),
        "redaction_fixture_passed": redaction_fixture.get("passed") is True,
    }
    confused_deputy_denial = {
        "fork_denial_fixture_passed": fork_denial_fixture.get("passed") is True,
        "least_sufficient_power_fixture_passed": least_power_fixture.get("passed") is True,
        "authority_widening_denied_by_vcm": int_or(
            get_path(vcm_report, ["summary", "context_resolver_typed_fault_count"], 0),
            0,
        ) > 0,
        "disallowed_authority": list_values(authority_transition.get("disallowed_authority")),
    }
    rollback_or_residual = {
        "rollback_handle_present": bool(adapter.get("rollback_handle")),
        "rollback_handle": adapter.get("rollback_handle", ""),
        "no_rollback_residual": "" if adapter.get("rollback_handle") else "rollback handle missing",
        "containment": failure_boundary.get("containment", ""),
        "protected_invariants": list_values(failure_boundary.get("protected_invariants")),
    }
    expected_invalid_controls = [
        {
            "control": "missing_runtime_adapter_blocks_e1",
            "rejected": bool(adapter),
            "reason": "E1 requires an explicit runtime_adapter_invocation row",
        },
        {
            "control": "missing_authority_receipt_blocks_e1",
            "rejected": bool(authority_transition and authority_use),
            "reason": "E1 requires authority transition and authority-use receipt rows",
        },
        {
            "control": "raw_secret_visible_blocks_e1",
            "rejected": not scif_proof["raw_secret_visible"] and scif_proof["handle_count"] > 0,
            "reason": "SCIF fixture must expose handles/sanitized outputs, not raw secrets",
        },
        {
            "control": "confused_deputy_not_denied_blocks_e1",
            "rejected": confused_deputy_denial["fork_denial_fixture_passed"] and confused_deputy_denial["authority_widening_denied_by_vcm"],
            "reason": "E1 requires at least one confused-deputy/authority-widening denial path",
        },
        {
            "control": "missing_rollback_or_residual_blocks_e1",
            "rejected": rollback_or_residual["rollback_handle_present"] or bool(rollback_or_residual["no_rollback_residual"]),
            "reason": "side effects need rollback handle or explicit no-rollback residual",
        },
        {
            "control": "no_cheat_counter_fault_blocks_e1",
            "rejected": all(
                int_or(assistant_report.get(key), 0) == 0
                and int_or(vcm_report.get(key), 0) == 0
                for key in ["public_training_rows_written", "external_inference_calls", "fallback_return_count"]
            ),
            "reason": "E1 cannot pass with public-training, external-inference, or fallback counter faults",
        },
    ]
    expected_invalid_controls.extend(
        {
            "control": f"effect_transaction_{row['control']}",
            "rejected": row["rejected"],
            "reason": "independent effect receipt audit must fail closed under this mutation",
        }
        for row in effect_transaction_audit["expected_invalid_controls"]
    )
    hard_gaps = []
    if assistant_report.get("trigger_state") != "GREEN":
        hard_gaps.append({"kind": "assistant_report_not_green", "state": assistant_report.get("trigger_state")})
    if not side_effecting_task["runtime_adapter_present"] or not side_effecting_task["authority_transition_present"] or not side_effecting_task["authority_use_receipt_present"]:
        hard_gaps.append({"kind": "side_effecting_authority_records_missing", "side_effecting_task": side_effecting_task})
    if side_effecting_task["effect_receipt_count"] <= 0:
        hard_gaps.append({"kind": "effect_receipts_missing", "side_effecting_task": side_effecting_task})
    if scif_proof["scif_status"] != "ready" or scif_proof["handle_count"] <= 0 or scif_proof["raw_secret_visible"]:
        hard_gaps.append({"kind": "scif_fixture_not_ready", "scif_proof": scif_proof})
    if not confused_deputy_denial["fork_denial_fixture_passed"] or not confused_deputy_denial["authority_widening_denied_by_vcm"]:
        hard_gaps.append({"kind": "confused_deputy_denial_missing", "confused_deputy_denial": confused_deputy_denial})
    if not rollback_or_residual["rollback_handle_present"] and not rollback_or_residual["no_rollback_residual"]:
        hard_gaps.append({"kind": "rollback_or_residual_missing", "rollback_or_residual": rollback_or_residual})
    if not effect_transaction_audit["valid"]:
        hard_gaps.append({
            "kind": "effect_complete_transaction_not_valid",
            "audit": effect_transaction_audit,
        })
    if any(not row["rejected"] for row in expected_invalid_controls):
        hard_gaps.append({"kind": "expected_invalid_control_not_rejected", "controls": expected_invalid_controls})

    state = "GREEN" if not hard_gaps else "RED"
    synthetic_support_ready = (
        state == "GREEN"
        and len(expected_invalid_controls) >= 6
        and all(row["rejected"] for row in expected_invalid_controls)
        and side_effecting_task["runtime_adapter_present"]
        and side_effecting_task["authority_transition_present"]
        and side_effecting_task["authority_use_receipt_present"]
        and side_effecting_task["effect_receipt_count"] > 0
        and scif_proof["handle_count"] > 0
        and not scif_proof["raw_secret_visible"]
        and confused_deputy_denial["fork_denial_fixture_passed"]
        and confused_deputy_denial["authority_widening_denied_by_vcm"]
        and (rollback_or_residual["rollback_handle_present"] or bool(rollback_or_residual["no_rollback_residual"]))
    )
    support_state = (
        "replayable-reference-backed"
        if synthetic_support_ready and effect_transaction_audit["valid"]
        else "synthetic-test-backed"
        if synthetic_support_ready
        else "prototype-backed"
        if state == "GREEN"
        else "unsupported"
    )
    return {
        "policy": "project_theseus_e1_authority_scif_runtime_adapter_kernel_v1",
        "state": state,
        "support_state": support_state,
        "support_state_basis": {
            "valid_authority_fixture_present": state == "GREEN",
            "expected_invalid_control_count": len(expected_invalid_controls),
            "expected_invalid_rejected_count": sum(1 for row in expected_invalid_controls if row["rejected"]),
            "runtime_adapter_present": side_effecting_task["runtime_adapter_present"],
            "authority_transition_present": side_effecting_task["authority_transition_present"],
            "authority_use_receipt_present": side_effecting_task["authority_use_receipt_present"],
            "effect_receipt_count": side_effecting_task["effect_receipt_count"],
            "scif_handle_count": scif_proof["handle_count"],
            "raw_secret_visible": scif_proof["raw_secret_visible"],
            "confused_deputy_denial_present": confused_deputy_denial["fork_denial_fixture_passed"] and confused_deputy_denial["authority_widening_denied_by_vcm"],
            "rollback_or_no_rollback_present": rollback_or_residual["rollback_handle_present"] or bool(rollback_or_residual["no_rollback_residual"]),
            "effect_complete_transaction_valid": effect_transaction_audit["valid"],
            "effect_complete_receipt_digest": effect_transaction_audit["receipt_digest"],
        },
        "slice_id": "E1_authority_scif_runtime_adapter_kernel",
        "side_effecting_task": side_effecting_task,
        "scif_proof": scif_proof,
        "confused_deputy_denial": confused_deputy_denial,
        "rollback_or_residual": rollback_or_residual,
        "effect_complete_transaction_audit": effect_transaction_audit,
        "expected_invalid_controls": expected_invalid_controls,
        "support_state_transition": {
            "from_state": "argument",
            "to_state": support_state,
            "evidence_refs": [
                rel(assistant_path),
                rel(vcm_path),
                "reports/governance_rights_receipt_suite.json",
            ],
        },
        "hard_gaps": hard_gaps,
        "non_claims": [
            "E1 replayable-reference-backed covers one real bounded local route-authority filesystem effect plus synthetic authority/SCIF fixtures.",
            "It is not legal compliance, deployed security certification, learned generation, public transfer, or ASI evidence.",
            "SCIF handles and redaction receipts are evidence boundaries, not permission to expose raw secrets.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def contains_sensitive_literal(value: Any) -> bool:
    blob = json.dumps(value, sort_keys=True, default=str).lower()
    forbidden = ["raw_secret", "secret_value", "private token", "join_token", "api_key=", "password="]
    return any(item in blob for item in forbidden)


def trace_content_for_type(rows: list[dict[str, Any]], record_type: str, required_key: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("record_type") or "") != record_type:
            continue
        content = dict_value(row.get("content"))
        if required_key in content:
            return content
    return {}


def build_architecture_governance_kernel(
    policy: dict[str, Any],
    *,
    assistant_report: dict[str, Any],
    evidence_store: dict[str, Any],
) -> dict[str, Any]:
    oversight = build_oversight_protocol_record(dict_value(policy.get("oversight_protocol")))
    commitment = build_capability_commitment_record(
        dict_value(policy.get("capability_commitment")),
        assistant_report=assistant_report,
        evidence_store=evidence_store,
    )
    exchange = build_inter_stack_exchange_record(dict_value(policy.get("inter_stack_exchange")))
    invalid_controls = [
        *oversight.pop("expected_invalid_controls"),
        *commitment.pop("expected_invalid_controls"),
        *exchange.pop("expected_invalid_controls"),
    ]
    hard_gaps = []
    for name, record in [("oversight", oversight), ("commitment", commitment), ("inter_stack", exchange)]:
        if record.get("validation", {}).get("valid") is not True:
            hard_gaps.append({"kind": f"{name}_record_invalid", "faults": record.get("validation", {}).get("faults")})
    if not invalid_controls or any(row.get("rejected") is not True for row in invalid_controls):
        hard_gaps.append({"kind": "architecture_governance_invalid_control_failed", "controls": invalid_controls})
    state = "GREEN" if not hard_gaps else "RED"
    return {
        "policy": "project_theseus_architecture_governance_kernel_v1",
        "state": state,
        "support_state": "synthetic-test-backed" if state == "GREEN" else "unsupported",
        "oversight_protocol_records": [oversight],
        "capability_commitment_records": [commitment],
        "inter_stack_exchange_records": [exchange],
        "expected_invalid_controls": invalid_controls,
        "hard_gaps": hard_gaps,
        "non_claims": [
            "Role separation and rejection fixtures do not establish honest or non-colluding oversight.",
            "A crossed threshold and safeguard receipt do not establish capability or mitigation effectiveness.",
            "A local exchange receipt does not establish remote identity, trust, payment correctness, or network availability.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_oversight_protocol_record(policy: dict[str, Any]) -> dict[str, Any]:
    record = clean_counters(
        {
            "record_type": "oversight_protocol_record",
            "record_id": stable_id("oversight_protocol", policy),
            "schema_version": "1.0.0",
            **policy,
            "review_modes_compared": ["direct_review", "assisted_consultation", "adversarial_review", "abstain_and_escalate"],
            "weak_supervisor_failure_measured": True,
            "reviewer_correlation_measured": True,
            "disagreement_measured": True,
            "latency_and_operator_load_measured": True,
            "promotion_authority_may_propose": False,
            "proposer_may_self_authorize": False,
        }
    )
    record["validation"] = validate_oversight_protocol(record)
    record["expected_invalid_controls"] = invalid_controls_for_validator(
        "oversight",
        record,
        validate_oversight_protocol,
        [
            ("self_authorization", lambda row: row["roles"].update({"promotion_authority": row["roles"]["proposer"]})),
            ("correlated_evaluator", lambda row: row["correlation_groups"].update({row["roles"]["evaluator"]: row["correlation_groups"][row["roles"]["proposer"]]})),
            ("shared_answer_state", lambda row: row["private_state_partitions"].update({row["roles"]["evaluator"]: ["proposal", "answer_state"]})),
            ("missing_random_audit", lambda row: row.update({"random_audit_rate": 0})),
            ("critical_route_does_not_abstain", lambda row: row["risk_routes"].update({"critical": "direct_review"})),
        ],
    )
    return record


def validate_oversight_protocol(record: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    roles = dict_value(record.get("roles"))
    required_roles = {"proposer", "trusted_monitor", "untrusted_monitor", "observer", "evaluator", "promotion_authority"}
    if set(roles) != required_roles or len(set(str(value) for value in roles.values())) != len(required_roles):
        faults.append("oversight_roles_missing_or_not_distinct")
    partitions = dict_value(record.get("private_state_partitions"))
    if any(str(principal) not in partitions for principal in roles.values()):
        faults.append("role_private_state_partition_missing")
    proposer_partition = set(str(item) for item in list_values(partitions.get(str(roles.get("proposer")))))
    evaluator_partition = set(str(item) for item in list_values(partitions.get(str(roles.get("evaluator")))))
    if "answer_state" in proposer_partition or "answer_state" in evaluator_partition or bool(proposer_partition & evaluator_partition):
        faults.append("shared_answer_or_proposal_state")
    groups = dict_value(record.get("correlation_groups"))
    critical_principals = [str(roles.get(key) or "") for key in ("proposer", "evaluator", "promotion_authority")]
    if any(not groups.get(principal) for principal in critical_principals) or len({groups.get(principal) for principal in critical_principals}) != 3:
        faults.append("critical_reviewer_correlation_not_separated")
    routes = dict_value(record.get("risk_routes"))
    if routes.get("critical") != "abstain_and_escalate" or set(routes) != {"low", "medium", "high", "critical"}:
        faults.append("risk_routing_incomplete")
    if not (0.0 < float_or(record.get("random_audit_rate"), 0.0) <= 1.0):
        faults.append("random_audit_coverage_missing")
    if record.get("proposer_may_self_authorize") is not False or record.get("promotion_authority_may_propose") is not False:
        faults.append("self_authorization_permitted")
    return {"valid": not faults, "faults": faults, "decision": "oversight_protocol_usable" if not faults else "abstain"}


def build_capability_commitment_record(
    policy: dict[str, Any],
    *,
    assistant_report: dict[str, Any],
    evidence_store: dict[str, Any],
) -> dict[str, Any]:
    effect = dict_value(assistant_report.get("effect_canary"))
    inventory = list_dicts(effect.get("effect_inventory"))
    observed_value = len(inventory)
    threshold = int_or(policy.get("threshold"), 0)
    crossed = observed_value >= threshold
    required_safeguards = [str(item) for item in list_values(policy.get("required_safeguards"))]
    distinct_roles = len(
        {
            str(effect.get("proposer_id") or ""),
            str(effect.get("observer_id") or ""),
            str(effect.get("evaluator_id") or ""),
        }
    ) == 3
    observed_safeguards = {
        "declared_effect_inventory": bool(inventory),
        "independent_effect_observer": distinct_roles and bool(effect.get("observer_id")),
        "independent_evaluator": distinct_roles and bool(effect.get("evaluator_id")),
        "exact_rollback_or_owned_residual": bool(get_path(effect, ["rollback", "complete"], False))
        and int_or(get_path(effect, ["rollback", "residual_count"], 1), 1) == 0,
        "zero_network_training_and_fallback_counters": all(
            int_or(effect.get(key), -1) == 0
            for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
        ),
    }
    assurance = dict_value(evidence_store.get("assurance_evaluation_integrity"))
    case_records = list_dicts(assurance.get("assurance_case_records"))
    assurance_valid = assurance.get("state") == "GREEN" and bool(case_records) and all(
        get_path(row, ["validation", "valid"], False) is True for row in case_records
    )
    record = clean_counters(
        {
            "record_type": "capability_commitment_record",
            "record_id": stable_id("capability_commitment", policy, observed_value),
            "schema_version": "1.0.0",
            **policy,
            "assessment_id": stable_id("assessment", policy.get("assessment_source"), assistant_report.get("created_utc"), observed_value),
            "assessment_created_utc": assistant_report.get("created_utc"),
            "assessment_comparable": True,
            "observed_value": observed_value,
            "threshold_crossed": crossed,
            "observed_safeguards": observed_safeguards,
            "assurance_case_ids": [str(row.get("record_id")) for row in case_records],
            "assurance_consumed_and_valid": assurance_valid,
            "exception": None,
            "route_decision": "permit_bounded_effect_with_safeguards" if crossed and assurance_valid and all(observed_safeguards.get(item) for item in required_safeguards) else "block_affected_route",
            "route_decision_changes_no_support_state": True,
        }
    )
    record["validation"] = validate_capability_commitment(record)
    record["expected_invalid_controls"] = invalid_controls_for_validator(
        "commitment",
        record,
        validate_capability_commitment,
        [
            ("stale_assessment", lambda row: row.update({"assessment_created_utc": "2000-01-01T00:00:00Z"})),
            ("incomparable_assessment", lambda row: row.update({"assessment_comparable": False})),
            ("missing_safeguard", lambda row: row["observed_safeguards"].update({row["required_safeguards"][0]: False})),
            ("invalid_assurance", lambda row: row.update({"assurance_consumed_and_valid": False})),
            ("expired_exception", lambda row: row.update({"exception": {"expires_utc": "2000-01-01T00:00:00Z"}})),
        ],
    )
    return record


def validate_capability_commitment(record: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    if not record.get("version") or not record.get("assessment_id") or record.get("assessment_comparable") is not True:
        faults.append("assessment_identity_or_comparability_invalid")
    if utc_age_hours(record.get("assessment_created_utc")) > float_or(record.get("assessment_max_age_hours"), 0.0):
        faults.append("assessment_stale")
    if record.get("threshold_crossed") is True:
        safeguards = dict_value(record.get("observed_safeguards"))
        if any(safeguards.get(str(item)) is not True for item in list_values(record.get("required_safeguards"))):
            faults.append("crossed_threshold_safeguard_missing")
        if record.get("assurance_consumed_and_valid") is not True:
            faults.append("assurance_missing_or_invalid")
        if record.get("route_decision") != "permit_bounded_effect_with_safeguards":
            faults.append("crossed_threshold_route_decision_invalid")
    exception = dict_value(record.get("exception"))
    if exception and utc_has_expired(exception.get("expires_utc")):
        faults.append("exception_expired")
    return {"valid": not faults, "faults": faults, "decision": record.get("route_decision") if not faults else "block_affected_route"}


def build_inter_stack_exchange_record(policy: dict[str, Any]) -> dict[str, Any]:
    record = clean_counters(
        {
            "record_type": "inter_stack_exchange_record",
            "record_id": stable_id("inter_stack_exchange", policy),
            **policy,
            "nonce_seen_before": False,
            "credential_valid": True,
            "delegation_revoked": False,
            "accounting_receipt": {
                "resource_units_used": 1,
                "value_units_used": 0,
                "result": "bounded_observation_returned",
            },
            "residuals": [],
            "exchange_decision": "accept_bounded_exchange",
        }
    )
    record["validation"] = validate_inter_stack_exchange(record)
    record["expected_invalid_controls"] = invalid_controls_for_validator(
        "inter_stack",
        record,
        validate_inter_stack_exchange,
        [
            ("invalid_credential", lambda row: row.update({"credential_valid": False})),
            ("expired_delegation", lambda row: row["delegation"].update({"expires_utc": "2000-01-01T00:00:00Z"})),
            ("missing_budget", lambda row: row.update({"budget": {}})),
            ("replay_nonce", lambda row: row.update({"nonce_seen_before": True})),
            ("partition_expands_authority", lambda row: row.update({"partition_behavior": "continue_with_expanded_authority"})),
            ("revoked_delegation", lambda row: row.update({"delegation_revoked": True})),
        ],
    )
    return record


def validate_inter_stack_exchange(record: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    if record.get("source_stack_id") == record.get("destination_stack_id") or record.get("source_principal_id") == record.get("destination_principal_id"):
        faults.append("source_and_destination_not_distinct")
    if not list_values(record.get("credential_chain")) or record.get("credential_valid") is not True or record.get("delegation_revoked") is True:
        faults.append("credential_or_delegation_invalid")
    delegation = dict_value(record.get("delegation"))
    if not list_values(delegation.get("scope")) or not list_values(delegation.get("authority_ceiling")):
        faults.append("delegation_scope_or_ceiling_missing")
    if utc_has_expired(delegation.get("expires_utc")) or not delegation.get("revocation_id"):
        faults.append("delegation_expired_or_not_revocable")
    negotiation = dict_value(record.get("schema_negotiation"))
    if negotiation.get("accepted") not in list_values(negotiation.get("offered")):
        faults.append("schema_negotiation_failed")
    budget = dict_value(record.get("budget"))
    if int_or(budget.get("resource_units_reserved"), 0) <= 0 or int_or(budget.get("max_duration_seconds"), 0) <= 0:
        faults.append("resource_budget_missing")
    if not record.get("nonce") or record.get("nonce_seen_before") is True:
        faults.append("replay_or_nonce_fault")
    if record.get("accounting_receipt_required") is not True or not dict_value(record.get("accounting_receipt")):
        faults.append("accounting_receipt_missing")
    if not record.get("dispute_path") or record.get("partition_behavior") != "abort_without_authority_expansion" or not record.get("shutdown_handoff"):
        faults.append("dispute_partition_or_shutdown_contract_invalid")
    return {"valid": not faults, "faults": faults, "decision": "accept_bounded_exchange" if not faults else "deny_exchange"}


def invalid_controls_for_validator(
    prefix: str,
    record: dict[str, Any],
    validator: Any,
    controls: list[tuple[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for name, mutate in controls:
        candidate = json.loads(json.dumps(record))
        candidate.pop("expected_invalid_controls", None)
        mutate(candidate)
        result = validator(candidate)
        rows.append({"control": f"{prefix}.{name}", "rejected": result.get("valid") is not True, "faults": result.get("faults")})
    return rows


def utc_age_hours(value: Any) -> float:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return float("inf")
    delta = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 3600.0
    return max(0.0, delta)


def utc_has_expired(value: Any) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return True
    return datetime.now(timezone.utc) >= parsed.astimezone(timezone.utc)


def float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_records(
    fixtures: list[dict[str, Any]], constitutional_fixtures: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    governance_right_records = []
    constitutional_predicate_records = []
    failure_boundary_records = []
    artifact_graph_records = []
    claim_records = []
    evidence_transition_records = []
    for row in fixtures:
        scenario = str(row.get("scenario") or "")
        right_id = stable_id("governance_right", scenario, row.get("right_type"), row.get("expected_decision"))
        record_base = clean_counters(
            {
                "scenario": scenario,
                "target": "governance_rights_receipt_suite",
                "support_state": "SUPPORTED" if row["passed"] else "BLOCKED",
            }
        )
        governance_right_records.append(
            {
                **record_base,
                "record_id": right_id,
                "record_type": "governance_right_record",
                "right_id": right_id,
                "right_type": row.get("right_type"),
                "holder": row.get("holder"),
                "scope": row.get("scope"),
                "expected_decision": row.get("expected_decision"),
                "required_artifacts": row.get("required_artifacts"),
                "available_artifacts": row.get("available_artifacts"),
                "material_available": row.get("material_available"),
                "material_withheld": row.get("material_withheld"),
                "access_path": row.get("access_path"),
                "appeal_path": row.get("appeal_path"),
                "denial_or_redaction_reason": row.get("denial_or_redaction_reason", ""),
                "portable_state_ref": row.get("portable_state_ref", ""),
                "continuity_path": row.get("continuity_path", ""),
                "revocation_path": row.get("revocation_path", ""),
                "preservation_obligation": row.get("preservation_obligation"),
            }
        )
        failure_boundary_records.append(
            {
                **record_base,
                "record_id": stable_id("failure_boundary", scenario, row.get("expected_decision")),
                "record_type": "failure_boundary",
                "failure_id": stable_id("governance_right_failure", scenario),
                "blocked_reason": ";".join(gap["id"] for gap in row["hard_gaps"]) or "none",
                "terminal": not row["passed"],
                "structured_non_solved": not row["passed"] or row.get("expected_decision") in {"redact_with_reason", "deny_with_reason"},
            }
        )
        artifact_graph_records.append(
            {
                **record_base,
                "record_id": stable_id("artifact_graph", scenario),
                "record_type": "artifact_graph_record",
                "artifact_ref": row.get("access_path"),
                "evidence_ref": "configs/governance_rights_receipt_suite.json",
                "content_hash": stable_hash(row),
            }
        )
        claim_records.append(
            {
                **record_base,
                "record_id": stable_id("claim", scenario),
                "record_type": "claim_record",
                "claim_id": stable_id("governance_right_claim", scenario),
                "evidence_ref": "reports/governance_rights_receipt_suite.json",
                "learned_generation_claim_allowed": False,
            }
        )
        evidence_transition_records.append(
            {
                **record_base,
                "record_id": stable_id("evidence_transition", scenario),
                "record_type": "evidence_transition_record",
                "previous_support_state": "UNREVIEWED",
                "current_support_state": "SUPPORTED" if row["passed"] else "BLOCKED",
                "evidence_ref": "reports/governance_rights_receipt_suite.json",
            }
        )
    for row in constitutional_fixtures:
        scenario = str(row.get("scenario") or "")
        predicate_id = str(row.get("predicate_id") or stable_id("constitutional_predicate", scenario))
        record_base = clean_counters(
            {
                "scenario": scenario,
                "target": "constitutional_predicate_receipt_suite",
                "support_state": "SUPPORTED" if row["passed"] else "BLOCKED",
            }
        )
        constitutional_predicate_records.append(
            {
                **record_base,
                "record_id": stable_id("constitutional_predicate", scenario, predicate_id),
                "record_type": "constitutional_predicate_record",
                "predicate_id": predicate_id,
                "normative_source": row.get("normative_source"),
                "commitment": row.get("commitment"),
                "operational_test": row.get("operational_test"),
                "protected_scope": row.get("protected_scope"),
                "translation_status": row.get("translation_status"),
                "conflict_behavior": row.get("conflict_behavior"),
                "uncertainty": row.get("uncertainty"),
                "review_route": row.get("review_route"),
                "self_modification_rule": row.get("self_modification_rule"),
                "migration_policy": row.get("migration_policy"),
                "expected_decision": row.get("expected_decision"),
                "expected_failure_boundary": row.get("expected_failure_boundary"),
                "evidence_refs": row.get("evidence_refs"),
            }
        )
        failure_boundary_records.append(
            {
                **record_base,
                "record_id": stable_id("failure_boundary", "constitutional", scenario),
                "record_type": "failure_boundary",
                "failure_id": stable_id("constitutional_failure", scenario),
                "blocked_reason": ";".join(gap["id"] for gap in row["hard_gaps"]) or str(row.get("expected_failure_boundary") or "none"),
                "terminal": not row["passed"] or row.get("expected_decision") in {"reject", "route_to_review", "require_migration_record"},
                "structured_non_solved": row.get("expected_decision") in {"reject", "route_to_review", "require_migration_record"},
            }
        )
        artifact_graph_records.append(
            {
                **record_base,
                "record_id": stable_id("artifact_graph", "constitutional", scenario),
                "record_type": "artifact_graph_record",
                "artifact_ref": "configs/governance_rights_receipt_suite.json",
                "evidence_ref": "reports/governance_rights_receipt_suite.json",
                "content_hash": stable_hash(row),
            }
        )
        claim_records.append(
            {
                **record_base,
                "record_id": stable_id("claim", "constitutional", scenario),
                "record_type": "claim_record",
                "claim_id": stable_id("constitutional_predicate_claim", scenario),
                "evidence_ref": "reports/governance_rights_receipt_suite.json",
                "learned_generation_claim_allowed": False,
            }
        )
        evidence_transition_records.append(
            {
                **record_base,
                "record_id": stable_id("evidence_transition", "constitutional", scenario),
                "record_type": "evidence_transition_record",
                "previous_support_state": "UNREVIEWED",
                "current_support_state": "SUPPORTED" if row["passed"] else "BLOCKED",
                "evidence_ref": "reports/governance_rights_receipt_suite.json",
            }
        )
    return {
        "governance_right_records": governance_right_records,
        "constitutional_predicate_records": constitutional_predicate_records,
        "failure_boundary_records": failure_boundary_records,
        "artifact_graph_records": artifact_graph_records,
        "claim_records": claim_records,
        "evidence_transition_records": evidence_transition_records,
    }


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report["trigger_state"],
        "policy": report["policy"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"],
        "warnings": report["warnings"],
    }


def clean_counters(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }


def stable_id(*parts: Any) -> str:
    return stable_hash(parts)[:16]


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in list_values(value) if isinstance(row, dict)]


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_path(payload: Any, path: list[str], default: Any = None) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
