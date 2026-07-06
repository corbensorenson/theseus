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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "governance_rights_receipt_suite.json"
DEFAULT_REPORT = ROOT / "reports" / "governance_rights_receipt_suite.json"
DEFAULT_ASSISTANT_REPORT = ROOT / "reports" / "theseus_assistant_product_spine_smoke.json"
DEFAULT_VCM_GOVERNOR = ROOT / "reports" / "vcm_context_governor.json"
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
        **records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "non_claims": [
            "Governance-right receipt fixtures prove material-usability protocol shape only.",
            "Constitutional-predicate fixtures prove record-level control semantics only.",
            "This is not institutional governance, legal compliance, moral correctness, public benchmark transfer, or learned-generation evidence.",
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
    trace_rows = list_dicts(assistant_report.get("assistant_viea_trace"))
    rows_by_type = {str(row.get("record_type") or ""): row for row in trace_rows}
    adapter = dict_value(rows_by_type.get("runtime_adapter_invocation", {}).get("content"))
    authority_transition = dict_value(rows_by_type.get("authority_transition", {}).get("content"))
    authority_use = dict_value(rows_by_type.get("authority_use_receipt", {}).get("content"))
    failure_boundary = dict_value(rows_by_type.get("failure_boundary", {}).get("content"))
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
    support_state = "synthetic-test-backed" if synthetic_support_ready else ("prototype-backed" if state == "GREEN" else "unsupported")
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
        },
        "slice_id": "E1_authority_scif_runtime_adapter_kernel",
        "side_effecting_task": side_effecting_task,
        "scif_proof": scif_proof,
        "confused_deputy_denial": confused_deputy_denial,
        "rollback_or_residual": rollback_or_residual,
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
            "E1 synthetic-test-backed proves one local authority/SCIF/runtime-adapter fixture chain plus expected-invalid controls only.",
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
