#!/usr/bin/env python3
"""Validate shared VIEA spine record contracts across current producers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402
import viea_spine_records  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "viea_spine_record_contracts.json"
DEFAULT_OUT = ROOT / "reports" / "viea_spine_record_gate.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "viea_spine_record_gate.md"
DEFAULT_VIEW_OUT = ROOT / "reports" / "viea_spine_materialized_view.json"

ARTIFACT_GRAPH_REQUIRED_FIELDS = {
    "artifact_id",
    "artifact_type",
    "parent_job",
    "source_refs",
    "context_refs",
    "context_transaction_refs",
    "semantic_certificate_refs",
    "tool_refs",
    "claim_refs",
    "test_refs",
    "audit_events",
    "replay_metadata",
    "replay_grade",
    "environment_assumptions",
    "provenance_status",
    "replay_limits",
    "evidence_gate",
    "residuals",
    "non_claims",
}

CONTEXT_TRANSACTION_REQUIRED_FIELDS = {
    "transaction_id",
    "operation",
    "snapshot_id",
    "mounts",
    "read_set",
    "write_set",
    "branch_policy",
    "taint_labels",
    "deletion_obligations",
    "declassification_refs",
    "derivative_refs",
    "contradiction_refs",
    "materialization_state",
    "closure_state",
    "faults",
    "audit_refs",
    "replay_boundary",
    "non_claims",
}

COSTED_ROUTE_REQUIRED_FIELDS = {
    "task_id",
    "route_state",
    "task_contract_ref",
    "quality_predicate",
    "authority_ceiling",
    "candidate_routes",
    "selected_route",
    "rejected_lower_cost_routes",
    "verification_result",
    "outcome_state",
    "cost_accounting",
    "cost_classes",
    "hidden_cost_checks",
    "residual_obligations",
    "fallback_route",
    "promotion_candidate",
    "support_state_effect",
    "non_claims",
}

COMPRESSED_ARTIFACT_REQUIRED_FIELDS = {
    "artifact_id",
    "source_artifact",
    "task_family",
    "access_pattern",
    "admission_state",
    "compression_method",
    "reconstruction_contract",
    "declared_use_envelope",
    "ratio_claim_state",
    "codec_parameters",
    "metadata_costs",
    "residual_coding",
    "probe_plan",
    "fallback_artifact",
    "fallback_trigger",
    "decode_determinism",
    "exact_replay_status",
    "consumer_policy",
    "utility_tests",
    "support_state_effect",
    "evidence_refs",
    "non_claims",
}

COMPRESSION_RECEIPT_REQUIRED_FIELDS = {
    "artifact_id",
    "receipt_state",
    "reconstruction_contract",
    "public_law_family",
    "seed",
    "search_bound",
    "generated_regions",
    "verification_result",
    "repair_residual",
    "fallback_threshold",
    "interface_costs",
    "consumer_policy",
    "use_permissions",
    "proxy_rate_status",
    "final_serialization_status",
    "rate_accounting",
    "support_state_effect",
    "evidence_refs",
    "non_claims",
}

COMPACT_GENERATIVE_REQUIRED_FIELDS = {
    "system_id",
    "target_system",
    "compact_seed",
    "rule_system",
    "memory_state",
    "generation_status",
    "residual_channel",
    "correction_mechanism",
    "verification_contract",
    "verification_status",
    "verifier_independence",
    "governance_interface",
    "authority_boundary",
    "use_envelope",
    "burden_ledger",
    "cost_accounting",
    "generative_leverage",
    "hidden_complexity_risks",
    "fallback_path",
    "fallback_status",
    "residual_burden_status",
    "promotion_state",
    "promotion_blockers",
    "retirement_condition",
    "support_state_effect",
    "source_refs",
    "evidence_refs",
    "non_claims",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--view-out", default=rel(DEFAULT_VIEW_OUT))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(config_path, config, started)
    view_path = resolve(args.view_out)
    materialized_view = build_materialized_view(config_path, config, report, view_path)
    report["outputs"] = {"materialized_view": rel(view_path)}
    report["summary"]["materialized_record_count"] = materialized_view["summary"]["record_count"]
    report["summary"]["materialized_claim_ledger_entry_count"] = materialized_view["summary"]["claim_ledger_entry_count"]
    report["summary"]["materialized_compression_record_count"] = materialized_view["summary"].get("compression_record_count", 0)
    report["summary"]["materialized_defeater_record_count"] = materialized_view["summary"].get("defeater_record_count", 0)
    report_evidence_store.write_json_report(
        resolve(args.out),
        report,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(report),
    )
    report_evidence_store.write_json_report(view_path, materialized_view)
    print(json.dumps(gate_view(report) if args.gate else report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(config_path: Path, config: dict[str, Any], started: float) -> dict[str, Any]:
    aliases = dict_value(config.get("canonical_record_aliases"))
    profile_reports = []
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for profile in list_dicts(config.get("profiles")):
        profile_report = audit_profile(profile, aliases)
        profile_reports.append(profile_report)
        hard_gaps.extend(profile_report["hard_gaps"])
        warnings.extend(profile_report["warnings"])
    if str(config.get("policy") or "") != "project_theseus_viea_spine_record_contracts_v1":
        hard_gaps.append(gap("config", "wrong_or_missing_policy", {"policy": config.get("policy")}))
    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_viea_spine_record_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "config": rel(config_path),
            "profile_count": len(profile_reports),
            "passing_profile_count": sum(1 for row in profile_reports if row["passed"]),
            "hard_gap_count": len(hard_gaps),
            "warning_count": len(warnings),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "profiles": profile_reports,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rules": {
            "shared_contract": "Assistant, planner, tool, Hive, verifier, and future execution surfaces normalize to shared VIEA record families.",
            "no_cheat": "Public training rows, runtime external inference, and fallback returns must remain zero.",
            "non_claim": "Passing this gate is implementation cohesion evidence, not learned-generation promotion evidence.",
        },
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def audit_profile(profile: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    source_kind = str(profile.get("source_kind") or "")
    if source_kind == "jsonl_trace":
        return audit_jsonl_trace_profile(profile, aliases)
    if source_kind == "compiled_dags":
        return audit_compiled_dags_profile(profile, aliases)
    if source_kind == "tool_report_and_artifact_graph":
        return audit_tool_profile(profile)
    if source_kind == "execution_spine_report":
        return audit_execution_spine_profile(profile, aliases)
    if source_kind == "hive_scheduler_report":
        return audit_hive_scheduler_profile(profile, aliases)
    if source_kind == "candidate_integrity_report":
        return audit_candidate_integrity_profile(profile, aliases)
    if source_kind == "private_verifier_spine_report":
        return audit_private_verifier_profile(profile, aliases)
    if source_kind == "report_evidence_store_report":
        return audit_report_evidence_store_profile(profile, aliases)
    if source_kind == "record_list_report":
        return audit_record_list_report_profile(profile, aliases)
    return profile_result(profile, False, [], [gap(profile_id(profile), "unsupported_source_kind", {"source_kind": source_kind})], [])


def audit_jsonl_trace_profile(profile: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    pid = profile_id(profile)
    required = [str(item) for item in list_values(profile.get("required_record_types"))]
    paths = [resolve(str(path)) for path in list_values(profile.get("paths"))]
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    path_reports = []
    for path in paths:
        rows = read_jsonl(path)
        missing = viea_spine_records.missing_required_record_types(rows, required, aliases)
        raw_prompt_rows = [row for row in rows if row.get("raw_prompt_stored") is True]
        counter_faults = no_cheat_faults(rows)
        clean = path.exists() and not missing and not raw_prompt_rows and not counter_faults
        if not path.exists():
            hard_gaps.append(gap(pid, "trace_path_missing", {"path": rel(path)}))
        if missing:
            hard_gaps.append(gap(pid, "required_record_types_missing", {"path": rel(path), "missing": missing}))
        if raw_prompt_rows:
            hard_gaps.append(gap(pid, "raw_prompt_stored_in_trace", {"path": rel(path), "count": len(raw_prompt_rows)}))
        if counter_faults:
            hard_gaps.append(gap(pid, "no_cheat_counter_fault", {"path": rel(path), "faults": counter_faults[:8]}))
        path_reports.append(
            {
                "path": rel(path),
                "present": path.exists(),
                "row_count": len(rows),
                "observed_record_types": sorted(viea_spine_records.collect_record_types(rows, aliases)),
                "missing_record_types": missing,
                "raw_prompt_stored_count": len(raw_prompt_rows),
                "counter_fault_count": len(counter_faults),
                "passed": clean,
            }
        )
    return profile_result(profile, bool(paths) and all(row["passed"] for row in path_reports), path_reports, hard_gaps, warnings)


def audit_compiled_dags_profile(profile: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    pid = profile_id(profile)
    path = resolve(str(profile.get("path") or ""))
    payload = read_json(path)
    goals = list_dicts(payload.get("compiled_goals"))
    required_node = {viea_spine_records.canonical_record_type(item, aliases) for item in list_values(profile.get("required_node_record_types"))}
    required_goal = {viea_spine_records.canonical_record_type(item, aliases) for item in list_values(profile.get("required_goal_record_types"))}
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    node_reports = []
    goal_reports = []
    if not path.exists():
        hard_gaps.append(gap(pid, "compiled_dags_missing", {"path": rel(path)}))
    if not goals:
        hard_gaps.append(gap(pid, "compiled_goals_missing", {"path": rel(path)}))
    if not viea_spine_records.no_cheat_counters_clean(payload):
        hard_gaps.append(gap(pid, "compiled_dags_no_cheat_counter_fault", no_cheat_summary(payload)))
    for goal in goals:
        goal_id = str(goal.get("goal_id") or "")
        goal_records = dict_value(goal.get("goal_records"))
        observed_goal = {viea_spine_records.canonical_record_type(key, aliases) for key in goal_records}
        missing_goal = sorted(required_goal - observed_goal)
        if missing_goal:
            hard_gaps.append(gap(pid, "goal_records_missing", {"goal_id": goal_id, "missing": missing_goal}))
        goal_reports.append({"goal_id": goal_id, "missing_record_types": missing_goal, "observed_record_types": sorted(observed_goal)})
        for node in list_dicts(goal.get("nodes")):
            node_id = str(node.get("node_id") or "")
            records = dict_value(node.get("asi_stack_records"))
            observed = set()
            for key, value in records.items():
                canonical_key = viea_spine_records.canonical_record_type(key, aliases)
                observed.add(canonical_key)
                if isinstance(value, dict) and value.get("record_type"):
                    observed.add(viea_spine_records.canonical_record_type(value.get("record_type"), aliases))
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and item.get("record_type"):
                            observed.add(viea_spine_records.canonical_record_type(item.get("record_type"), aliases))
            missing = sorted(required_node - observed)
            if missing:
                hard_gaps.append(gap(pid, "node_records_missing", {"node_id": node_id, "missing": missing}))
            node_reports.append({"node_id": node_id, "missing_record_types": missing, "observed_record_types": sorted(observed)})
    return profile_result(
        profile,
        path.exists() and bool(goals) and not any(row["missing_record_types"] for row in node_reports + goal_reports),
        {"path": rel(path), "goal_count": len(goals), "node_count": len(node_reports), "goals": goal_reports[:20], "nodes": node_reports[:40]},
        hard_gaps,
        warnings,
    )


def audit_tool_profile(profile: dict[str, Any]) -> dict[str, Any]:
    pid = profile_id(profile)
    report_path = resolve(str(profile.get("report_path") or ""))
    graph_path = resolve(str(profile.get("artifact_graph_path") or ""))
    report = read_json(report_path)
    graph = read_json(graph_path)
    summary = dict_value(report.get("summary"))
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not report_path.exists():
        hard_gaps.append(gap(pid, "tool_report_missing", {"path": rel(report_path)}))
    if not graph_path.exists():
        hard_gaps.append(gap(pid, "tool_artifact_graph_missing", {"path": rel(graph_path)}))
    if report.get("trigger_state") != "GREEN":
        hard_gaps.append(gap(pid, "tool_report_not_green", {"trigger_state": report.get("trigger_state")}))
    if int(summary.get("result_count") or summary.get("private_case_result_count") or 0) <= 0:
        hard_gaps.append(gap(pid, "tool_results_missing", {"summary": summary}))
    if not viea_spine_records.no_cheat_counters_clean(report) or not viea_spine_records.no_cheat_counters_clean(summary):
        hard_gaps.append(gap(pid, "tool_no_cheat_counter_fault", {"report": no_cheat_summary(report), "summary": no_cheat_summary(summary)}))
    artifact_count = len(list_values(graph.get("artifacts"))) + len(list_values(graph.get("nodes")))
    claim_count = len(list_values(graph.get("claims"))) + len(list_values(report.get("claim_ledger")))
    if graph_path.exists() and artifact_count <= 0:
        warnings.append(gap(pid, "tool_artifact_graph_has_no_artifact_list", {"path": rel(graph_path)}, severity="warning"))
    return profile_result(
        profile,
        not hard_gaps,
        {
            "report": rel(report_path),
            "artifact_graph": rel(graph_path),
            "trigger_state": report.get("trigger_state"),
            "result_count": int(summary.get("result_count") or summary.get("private_case_result_count") or 0),
            "artifact_count": artifact_count,
            "claim_count": claim_count,
        },
        hard_gaps,
        warnings,
    )


def audit_execution_spine_profile(profile: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    pid = profile_id(profile)
    path = resolve(str(profile.get("path") or ""))
    payload = read_json(path)
    required = {viea_spine_records.canonical_record_type(item, aliases) for item in list_values(profile.get("required_record_types"))}
    results = list_dicts(payload.get("compiled_execution_results"))
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not path.exists():
        hard_gaps.append(gap(pid, "execution_spine_report_missing", {"path": rel(path)}))
    if payload.get("trigger_state") not in {"GREEN", "YELLOW"}:
        hard_gaps.append(gap(pid, "execution_spine_not_ready", {"trigger_state": payload.get("trigger_state")}))
    if not results:
        hard_gaps.append(gap(pid, "execution_spine_results_missing", {"path": rel(path)}))
    if not viea_spine_records.no_cheat_counters_clean(payload) or not viea_spine_records.no_cheat_counters_clean(dict_value(payload.get("summary"))):
        hard_gaps.append(gap(pid, "execution_spine_no_cheat_counter_fault", {"report": no_cheat_summary(payload), "summary": no_cheat_summary(dict_value(payload.get("summary")))}))
    result_reports = []
    for result in results:
        records = dict_value(result.get("asi_stack_execution_records"))
        observed = set()
        for key, value in records.items():
            observed.add(viea_spine_records.canonical_record_type(key, aliases))
            if isinstance(value, dict) and value.get("record_type"):
                observed.add(viea_spine_records.canonical_record_type(value.get("record_type"), aliases))
        missing = sorted(required - observed)
        if missing:
            hard_gaps.append(gap(pid, "execution_result_records_missing", {"case_id": result.get("case_id"), "missing": missing}))
        result_reports.append(
            {
                "case_id": result.get("case_id"),
                "node_id": result.get("node_id"),
                "observed_record_types": sorted(observed),
                "missing_record_types": missing,
                "verified": bool(result.get("verified")),
            }
        )
    return profile_result(
        profile,
        path.exists() and bool(results) and not any(row["missing_record_types"] for row in result_reports) and not hard_gaps,
        {"path": rel(path), "result_count": len(results), "results": result_reports[:40]},
        hard_gaps,
        warnings,
    )


def audit_hive_scheduler_profile(profile: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    pid = profile_id(profile)
    path = resolve(str(profile.get("path") or ""))
    payload = read_json(path)
    placements = list_dicts(payload.get("placements"))
    required = {viea_spine_records.canonical_record_type(item, aliases) for item in list_values(profile.get("required_record_types"))}
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not path.exists():
        hard_gaps.append(gap(pid, "hive_scheduler_report_missing", {"path": rel(path)}))
    if payload.get("policy") != "project_theseus_hive_scheduler_v0":
        hard_gaps.append(gap(pid, "hive_scheduler_policy_mismatch", {"policy": payload.get("policy")}))
    if not placements:
        hard_gaps.append(gap(pid, "hive_scheduler_placements_missing", {"path": rel(path)}))
    if not viea_spine_records.no_cheat_counters_clean(payload) or not viea_spine_records.no_cheat_counters_clean(dict_value(payload.get("viea_spine"))):
        hard_gaps.append(gap(pid, "hive_scheduler_no_cheat_counter_fault", {"report": no_cheat_summary(payload), "viea_spine": no_cheat_summary(dict_value(payload.get("viea_spine")))}))
    route_validator = dict_value(payload.get("route_validator_receipt"))
    if not route_validator:
        hard_gaps.append(gap(pid, "hive_scheduler_route_validator_receipt_missing", {"path": rel(path)}))
    elif not route_validator.get("ready"):
        receipt_evidence = {
            "missing_required_groups": route_validator.get("missing_required_groups"),
            "view_trigger_state": route_validator.get("view_trigger_state"),
            "no_cheat_fault_count": route_validator.get("no_cheat_fault_count"),
        }
        bootstrap_safe = (
            bool(profile.get("allow_route_validator_bootstrap"))
            and not list_values(route_validator.get("missing_required_groups"))
            and int(route_validator.get("no_cheat_fault_count") or 0) == 0
        )
        if not bootstrap_safe:
            hard_gaps.append(
                gap(
                    pid,
                    "hive_scheduler_route_validator_receipt_not_ready",
                    receipt_evidence,
                )
            )
    placement_reports = []

    def audit_record_set(scope: str, records: dict[str, Any], *, missing_gap_kind: str, counter_gap_kind: str) -> dict[str, Any]:
        observed = set()
        counter_faults = []
        for key, value in records.items():
            if not isinstance(value, dict):
                continue
            observed.add(viea_spine_records.canonical_record_type(key, aliases))
            if value.get("record_type"):
                observed.add(viea_spine_records.canonical_record_type(value.get("record_type"), aliases))
            if not viea_spine_records.no_cheat_counters_clean(value):
                counter_faults.append({"record": key, "counters": no_cheat_summary(value)})
            if value.get("fallback_return_used") is True:
                counter_faults.append({"record": key, "fallback_return_used": True})
        missing = sorted(required - observed)
        if missing:
            hard_gaps.append(gap(pid, missing_gap_kind, {"scope": scope, "missing": missing}))
        if counter_faults:
            hard_gaps.append(gap(pid, counter_gap_kind, {"scope": scope, "faults": counter_faults}))
        return {
            "scope": scope,
            "observed_record_types": sorted(observed),
            "missing_record_types": missing,
            "counter_fault_count": len(counter_faults),
        }

    for index, placement in enumerate(placements):
        placement_id = str(placement.get("placement_id") or f"placement:{index}")
        records = dict_value(placement.get("viea_route_records"))
        record_report = audit_record_set(
            f"placement:{placement_id}",
            records,
            missing_gap_kind="hive_scheduler_route_records_missing",
            counter_gap_kind="hive_scheduler_route_no_cheat_counter_fault",
        )
        placement_reports.append(
            {
                "placement_id": placement_id,
                "task_kind": placement.get("task_kind"),
                "target": placement.get("target"),
                "node_id": placement.get("node_id"),
                "observed_record_types": record_report["observed_record_types"],
                "missing_record_types": record_report["missing_record_types"],
                "counter_fault_count": record_report["counter_fault_count"],
            }
        )
    smoke = dict_value(payload.get("viea_execution_receipt_smoke"))
    if not smoke:
        hard_gaps.append(gap(pid, "hive_scheduler_execution_receipt_smoke_missing", {"path": rel(path)}))
    elif not smoke.get("ready"):
        hard_gaps.append(gap(pid, "hive_scheduler_execution_receipt_smoke_not_ready", {"reason": smoke.get("reason")}))
    if smoke and not viea_spine_records.no_cheat_counters_clean(smoke):
        hard_gaps.append(gap(pid, "hive_scheduler_execution_receipt_smoke_no_cheat_counter_fault", {"counters": no_cheat_summary(smoke)}))
    smoke_records = dict_value(smoke.get("viea_execution_records")) if smoke else {}
    smoke_report = audit_record_set(
        "execution_receipt_smoke",
        smoke_records,
        missing_gap_kind="hive_scheduler_execution_receipt_records_missing",
        counter_gap_kind="hive_scheduler_execution_receipt_no_cheat_counter_fault",
    )
    execution_reports = []
    for index, row in enumerate(list_dicts(payload.get("execution"))):
        scope = f"execution:{row.get('placement_id') or index}"
        records = dict_value(row.get("viea_execution_records"))
        if not records:
            hard_gaps.append(gap(pid, "hive_scheduler_execution_records_missing", {"scope": scope}))
            execution_reports.append({"scope": scope, "missing_record_types": sorted(required), "counter_fault_count": 0})
            continue
        execution_reports.append(
            audit_record_set(
                scope,
                records,
                missing_gap_kind="hive_scheduler_execution_records_missing",
                counter_gap_kind="hive_scheduler_execution_no_cheat_counter_fault",
            )
        )
    return profile_result(
        profile,
        path.exists() and bool(placements) and bool(smoke) and not hard_gaps,
        {
            "path": rel(path),
            "placement_count": len(placements),
            "viea_spine": dict_value(payload.get("viea_spine")),
            "route_validator_receipt": {
                "ready": route_validator.get("ready"),
                "receipt_id": route_validator.get("receipt_id"),
                "record_count": route_validator.get("record_count"),
                "missing_required_groups": route_validator.get("missing_required_groups"),
            },
            "placements": placement_reports[:80],
            "execution_receipt_smoke": smoke_report,
            "execution_receipt_count": len(list_dicts(payload.get("execution"))),
            "execution_receipts": execution_reports[:80],
        },
        hard_gaps,
        warnings,
    )


def audit_candidate_integrity_profile(profile: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    pid = profile_id(profile)
    path = resolve(str(profile.get("path") or ""))
    payload = read_json(path)
    required = {viea_spine_records.canonical_record_type(item, aliases) for item in list_values(profile.get("required_record_types"))}
    records = dict_value(payload.get("viea_integrity_records"))
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not path.exists():
        hard_gaps.append(gap(pid, "candidate_integrity_report_missing", {"path": rel(path)}))
    if payload.get("policy") != "project_theseus_candidate_integrity_audit_v1":
        hard_gaps.append(gap(pid, "candidate_integrity_policy_mismatch", {"policy": payload.get("policy")}))
    if payload.get("trigger_state") not in {"GREEN", "YELLOW"}:
        hard_gaps.append(gap(pid, "candidate_integrity_not_ready", {"trigger_state": payload.get("trigger_state")}))
    if int(dict_value(payload.get("summary")).get("candidate_count") or 0) <= 0:
        hard_gaps.append(gap(pid, "candidate_integrity_candidates_missing", {"summary": dict_value(payload.get("summary"))}))
    if not viea_spine_records.no_cheat_counters_clean(payload) or not viea_spine_records.no_cheat_counters_clean(dict_value(payload.get("summary"))):
        hard_gaps.append(gap(pid, "candidate_integrity_no_cheat_counter_fault", {"report": no_cheat_summary(payload), "summary": no_cheat_summary(dict_value(payload.get("summary")))}))
    observed = set()
    counter_faults = []
    for key, value in records.items():
        if isinstance(value, dict):
            observed.add(viea_spine_records.canonical_record_type(key, aliases))
            if value.get("record_type"):
                observed.add(viea_spine_records.canonical_record_type(value.get("record_type"), aliases))
            if not viea_spine_records.no_cheat_counters_clean(value):
                counter_faults.append({"record": key, "counters": no_cheat_summary(value)})
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    continue
                observed.add(viea_spine_records.canonical_record_type(key, aliases))
                if item.get("record_type"):
                    observed.add(viea_spine_records.canonical_record_type(item.get("record_type"), aliases))
                if not viea_spine_records.no_cheat_counters_clean(item):
                    counter_faults.append({"record": f"{key}:{index}", "counters": no_cheat_summary(item)})
    missing = sorted(required - observed)
    if missing:
        hard_gaps.append(gap(pid, "candidate_integrity_records_missing", {"missing": missing}))
    if counter_faults:
        hard_gaps.append(gap(pid, "candidate_integrity_record_counter_fault", {"faults": counter_faults[:20]}))
    return profile_result(
        profile,
        path.exists() and not missing and not counter_faults and not hard_gaps,
        {
            "path": rel(path),
            "trigger_state": payload.get("trigger_state"),
            "summary": dict_value(payload.get("summary")),
            "observed_record_types": sorted(observed),
            "missing_record_types": missing,
            "counter_fault_count": len(counter_faults),
        },
        hard_gaps,
        warnings,
    )


def audit_private_verifier_profile(profile: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    pid = profile_id(profile)
    path = resolve(str(profile.get("path") or ""))
    payload = read_json(path)
    required = {viea_spine_records.canonical_record_type(item, aliases) for item in list_values(profile.get("required_record_types"))}
    verification = dict_value(payload.get("private_verification"))
    records = dict_value(verification.get("viea_verifier_records"))
    summary = dict_value(payload.get("summary"))
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not path.exists():
        hard_gaps.append(gap(pid, "private_verifier_spine_report_missing", {"path": rel(path)}))
    if payload.get("policy") != "project_theseus_private_verifier_spine_smoke_v1":
        hard_gaps.append(gap(pid, "private_verifier_policy_mismatch", {"policy": payload.get("policy")}))
    if payload.get("trigger_state") != "GREEN":
        hard_gaps.append(gap(pid, "private_verifier_not_green", {"trigger_state": payload.get("trigger_state")}))
    if verification.get("policy") != "private_code_candidate_verification_cascade_v1":
        hard_gaps.append(gap(pid, "private_verification_summary_missing", {"policy": verification.get("policy")}))
    if int(summary.get("candidate_attempt_count") or 0) <= 0:
        hard_gaps.append(gap(pid, "private_verifier_attempts_missing", {"summary": summary}))
    if not viea_spine_records.no_cheat_counters_clean(payload) or not viea_spine_records.no_cheat_counters_clean(verification):
        hard_gaps.append(gap(pid, "private_verifier_no_cheat_counter_fault", {"report": no_cheat_summary(payload), "verification": no_cheat_summary(verification)}))
    observed = set()
    counter_faults = []
    for key, value in records.items():
        if not isinstance(value, dict):
            continue
        observed.add(viea_spine_records.canonical_record_type(key, aliases))
        if value.get("record_type"):
            observed.add(viea_spine_records.canonical_record_type(value.get("record_type"), aliases))
        if not viea_spine_records.no_cheat_counters_clean(value):
            counter_faults.append({"record": key, "counters": no_cheat_summary(value)})
    missing = sorted(required - observed)
    if missing:
        hard_gaps.append(gap(pid, "private_verifier_records_missing", {"missing": missing}))
    if counter_faults:
        hard_gaps.append(gap(pid, "private_verifier_record_counter_fault", {"faults": counter_faults[:20]}))
    return profile_result(
        profile,
        path.exists() and payload.get("trigger_state") == "GREEN" and not missing and not counter_faults and not hard_gaps,
        {
            "path": rel(path),
            "summary": summary,
            "observed_record_types": sorted(observed),
            "missing_record_types": missing,
            "counter_fault_count": len(counter_faults),
        },
        hard_gaps,
        warnings,
    )


def audit_report_evidence_store_profile(profile: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    pid = profile_id(profile)
    path = resolve(str(profile.get("path") or ""))
    payload = read_json(path)
    required = {viea_spine_records.canonical_record_type(item, aliases) for item in list_values(profile.get("required_record_types"))}
    records = (
        list_dicts(payload.get("compression_records"))
        + list_dicts(payload.get("compressed_artifact_records"))
        + list_dicts(payload.get("compression_receipts"))
        + list_dicts(payload.get("defeater_records"))
    )
    observed = viea_spine_records.collect_record_types(records, aliases)
    missing = sorted(required - observed)
    counter_faults = no_cheat_faults(records)
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not path.exists():
        hard_gaps.append(gap(pid, "report_evidence_store_missing", {"path": rel(path)}))
    if not viea_spine_records.no_cheat_counters_clean(payload):
        hard_gaps.append(gap(pid, "report_evidence_store_no_cheat_counter_fault", no_cheat_summary(payload)))
    if missing:
        hard_gaps.append(gap(pid, "report_evidence_store_records_missing", {"missing": missing}))
    if counter_faults:
        hard_gaps.append(gap(pid, "report_evidence_store_record_counter_fault", {"faults": counter_faults[:20]}))
    return profile_result(
        profile,
        path.exists() and not missing and not counter_faults and not hard_gaps,
        {
            "path": rel(path),
            "compression_record_count": len(list_dicts(payload.get("compression_records"))),
            "compressed_artifact_record_count": len(list_dicts(payload.get("compressed_artifact_records"))),
            "compression_receipt_count": len(list_dicts(payload.get("compression_receipts"))),
            "defeater_record_count": len(list_dicts(payload.get("defeater_records"))),
            "observed_record_types": sorted(observed),
            "missing_record_types": missing,
            "counter_fault_count": len(counter_faults),
        },
        hard_gaps,
        warnings,
    )


def audit_record_list_report_profile(profile: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    pid = profile_id(profile)
    path = resolve(str(profile.get("path") or ""))
    payload = read_json(path)
    summary = dict_value(payload.get("summary"))
    fields = [str(item) for item in list_values(profile.get("record_list_fields"))]
    required = {viea_spine_records.canonical_record_type(item, aliases) for item in list_values(profile.get("required_record_types"))}
    records: list[dict[str, Any]] = []
    field_counts = {}
    for field in fields:
        rows = list_dicts(payload.get(field))
        field_counts[field] = len(rows)
        records.extend(rows)
    observed = viea_spine_records.collect_record_types(records, aliases)
    missing = sorted(required - observed)
    action_count = viea_spine_records.int_or(summary.get("action_count"), 0)
    if bool(profile.get("allow_empty_when_no_actions")) and action_count == 0:
        missing = []
    counter_faults = no_cheat_faults(records)
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not path.exists():
        hard_gaps.append(gap(pid, "record_list_report_missing", {"path": rel(path)}))
    if not viea_spine_records.no_cheat_counters_clean(payload):
        hard_gaps.append(gap(pid, "record_list_report_no_cheat_counter_fault", no_cheat_summary(payload)))
    if missing:
        hard_gaps.append(gap(pid, "record_list_report_records_missing", {"missing": missing, "field_counts": field_counts}))
    if counter_faults:
        hard_gaps.append(gap(pid, "record_list_report_counter_fault", {"faults": counter_faults[:20]}))
    return profile_result(
        profile,
        path.exists() and not missing and not counter_faults and not hard_gaps,
        {
            "path": rel(path),
            "field_counts": field_counts,
            "action_count": action_count,
            "observed_record_types": sorted(observed),
            "missing_record_types": missing,
            "counter_fault_count": len(counter_faults),
        },
        hard_gaps,
        warnings,
    )


def build_materialized_view(config_path: Path, config: dict[str, Any], report: dict[str, Any], view_path: Path) -> dict[str, Any]:
    aliases = dict_value(config.get("canonical_record_aliases"))
    records: list[dict[str, Any]] = []
    sources = []
    for profile in list_dicts(config.get("profiles")):
        source_kind = str(profile.get("source_kind") or "")
        if source_kind == "jsonl_trace":
            for path_value in list_values(profile.get("paths")):
                path = resolve(str(path_value))
                sources.append(source_descriptor(profile, path))
                for index, row in enumerate(read_jsonl(path)):
                    if isinstance(row, dict):
                        records.append(materialized_record(row.get("record_type"), row, aliases, profile, path, f"row:{index}"))
        elif source_kind == "compiled_dags":
            path = resolve(str(profile.get("path") or ""))
            payload = read_json(path)
            sources.append(source_descriptor(profile, path))
            for goal in list_dicts(payload.get("compiled_goals")):
                goal_id = str(goal.get("goal_id") or "")
                for key, value in dict_value(goal.get("goal_records")).items():
                    records.extend(materialize_nested(key, value, aliases, profile, path, f"goal:{goal_id}:{key}", {"goal_id": goal_id}))
                for node in list_dicts(goal.get("nodes")):
                    node_id = str(node.get("node_id") or "")
                    for key, value in dict_value(node.get("asi_stack_records")).items():
                        records.extend(materialize_nested(key, value, aliases, profile, path, f"node:{node_id}:{key}", {"goal_id": goal_id, "node_id": node_id}))
        elif source_kind == "tool_report_and_artifact_graph":
            report_path = resolve(str(profile.get("report_path") or ""))
            graph_path = resolve(str(profile.get("artifact_graph_path") or ""))
            payload = read_json(report_path)
            graph = read_json(graph_path)
            sources.append(source_descriptor(profile, report_path))
            sources.append(source_descriptor(profile, graph_path))
            for index, row in enumerate(list_dicts(payload.get("claim_ledger"))):
                records.append(materialized_record(row.get("record_type") or "claim_record", row, aliases, profile, report_path, f"claim:{index}"))
            for index, row in enumerate(list_dicts(payload.get("tool_results"))):
                records.append(materialized_record("tool_result_evidence", row, aliases, profile, report_path, f"tool_result:{index}"))
            for index, row in enumerate(list_dicts(graph.get("artifacts")) + list_dicts(graph.get("nodes"))):
                records.append(materialized_record("artifact_graph_record", row, aliases, profile, graph_path, f"artifact:{index}"))
        elif source_kind == "execution_spine_report":
            path = resolve(str(profile.get("path") or ""))
            payload = read_json(path)
            sources.append(source_descriptor(profile, path))
            for result in list_dicts(payload.get("compiled_execution_results")):
                case_id = str(result.get("case_id") or "")
                node_id = str(result.get("node_id") or "")
                for key, value in dict_value(result.get("asi_stack_execution_records")).items():
                    records.extend(materialize_nested(key, value, aliases, profile, path, f"execution:{case_id}:{key}", {"case_id": case_id, "node_id": node_id}))
        elif source_kind == "hive_scheduler_report":
            path = resolve(str(profile.get("path") or ""))
            payload = read_json(path)
            sources.append(source_descriptor(profile, path))
            for index, placement in enumerate(list_dicts(payload.get("placements"))):
                placement_id = str(placement.get("placement_id") or f"placement:{index}")
                parent = {
                    "placement_id": placement_id,
                    "task_kind": placement.get("task_kind"),
                    "target": placement.get("target"),
                    "node_id": placement.get("node_id"),
                }
                for key, value in dict_value(placement.get("viea_route_records")).items():
                    records.extend(materialize_nested(key, value, aliases, profile, path, f"placement:{placement_id}:{key}", parent))
            smoke = dict_value(payload.get("viea_execution_receipt_smoke"))
            smoke_parent = {
                "placement_id": smoke.get("placement_id"),
                "task_kind": smoke.get("task_kind"),
                "target": smoke.get("target"),
                "node_id": smoke.get("node_id"),
                "execution_scope": "receipt_schema_smoke",
            }
            for key, value in dict_value(smoke.get("viea_execution_records")).items():
                records.extend(materialize_nested(key, value, aliases, profile, path, f"execution_smoke:{key}", smoke_parent))
            for index, row in enumerate(list_dicts(payload.get("execution"))):
                execution_parent = {
                    "placement_id": row.get("placement_id"),
                    "task_kind": row.get("task_kind"),
                    "target": row.get("target"),
                    "node_id": row.get("node_id"),
                    "execution_scope": row.get("placement_id") or f"execution:{index}",
                }
                for key, value in dict_value(row.get("viea_execution_records")).items():
                    records.extend(materialize_nested(key, value, aliases, profile, path, f"execution:{index}:{key}", execution_parent))
        elif source_kind == "candidate_integrity_report":
            path = resolve(str(profile.get("path") or ""))
            payload = read_json(path)
            sources.append(source_descriptor(profile, path))
            summary = dict_value(payload.get("summary"))
            parent = {
                "audit_scope": "candidate_family_integrity",
                "candidate_count": summary.get("candidate_count"),
                "integrity_verified_candidate_count": summary.get("integrity_verified_candidate_count"),
                "integrity_mismatch_count": summary.get("integrity_mismatch_count"),
            }
            for key, value in dict_value(payload.get("viea_integrity_records")).items():
                records.extend(materialize_nested(key, value, aliases, profile, path, f"candidate_integrity:{key}", parent))
        elif source_kind == "private_verifier_spine_report":
            path = resolve(str(profile.get("path") or ""))
            payload = read_json(path)
            sources.append(source_descriptor(profile, path))
            summary = dict_value(payload.get("summary"))
            parent = {
                "verifier_surface": "code_lm_private_verifier",
                "candidate_attempt_count": summary.get("candidate_attempt_count"),
                "runtime_load_rate": summary.get("runtime_load_rate"),
                "intended_behavior_pass_rate": summary.get("intended_behavior_pass_rate"),
            }
            verification = dict_value(payload.get("private_verification"))
            for key, value in dict_value(verification.get("viea_verifier_records")).items():
                records.extend(materialize_nested(key, value, aliases, profile, path, f"private_verifier:{key}", parent))
        elif source_kind == "report_evidence_store_report":
            path = resolve(str(profile.get("path") or ""))
            payload = read_json(path)
            sources.append(source_descriptor(profile, path))
            for index, row in enumerate(list_dicts(payload.get("compression_records"))):
                records.append(materialized_record(row.get("record_type") or "compression_record", row, aliases, profile, path, f"compression:{index}"))
            for index, row in enumerate(list_dicts(payload.get("compressed_artifact_records"))):
                records.append(materialized_record(row.get("record_type") or "compressed_artifact_record", row, aliases, profile, path, f"compressed_artifact:{index}"))
            for index, row in enumerate(list_dicts(payload.get("compression_receipts"))):
                records.append(materialized_record(row.get("record_type") or "compression_receipt", row, aliases, profile, path, f"compression_receipt:{index}"))
            for index, row in enumerate(list_dicts(payload.get("defeater_records"))):
                records.append(materialized_record(row.get("record_type") or "defeater_record", row, aliases, profile, path, f"defeater:{index}"))
        elif source_kind == "record_list_report":
            path = resolve(str(profile.get("path") or ""))
            payload = read_json(path)
            sources.append(source_descriptor(profile, path))
            for field in [str(item) for item in list_values(profile.get("record_list_fields"))]:
                for index, row in enumerate(list_dicts(payload.get(field))):
                    records.append(materialized_record(row.get("record_type") or field.rstrip("s"), row, aliases, profile, path, f"{field}:{index}"))
    deduped = dedupe_records(records)
    groups = group_materialized_records(deduped)
    schema_shape_gaps = schema_payload_gaps(deduped)
    no_cheat_faults = [
        row
        for row in deduped
        if any(viea_spine_records.int_or(row.get("compact_payload", {}).get(key), 0) != 0 for key in viea_spine_records.NO_CHEAT_COUNTERS)
    ]
    return {
        "policy": "project_theseus_viea_spine_materialized_view_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if report.get("trigger_state") == "GREEN" and not no_cheat_faults and not schema_shape_gaps else "RED",
        "source_config": rel(config_path),
        "view_path": rel(view_path),
        "summary": {
            "record_count": len(deduped),
            "source_count": len(sources),
            "canonical_record_type_count": len({row["canonical_record_type"] for row in deduped}),
            "claim_ledger_entry_count": len(groups["claim_ledger_entries"]),
            "semantic_ir_record_count": len(groups["semantic_ir_records"]),
            "simulation_fidelity_record_count": len(groups["simulation_fidelity_records"]),
            "governance_record_count": len(groups["governance_records"]),
            "failure_boundary_count": len(groups["failure_boundaries"]),
            "authority_record_count": len(groups["authority_records"]),
            "runtime_adapter_record_count": len(groups["runtime_adapter_records"]),
            "resource_route_record_count": len(groups["resource_route_records"]),
            "generation_mode_record_count": len(groups["generation_mode_records"]),
            "context_record_count": len(groups["context_records"]),
            "compression_record_count": len(groups["compression_records"]),
            "compressed_artifact_record_count": len(groups["compressed_artifact_records"]),
            "compression_receipt_count": len(groups["compression_receipts"]),
            "compact_generative_record_count": len(groups["compact_generative_records"]),
            "defeater_record_count": len(groups["defeater_records"]),
            "no_cheat_fault_count": len(no_cheat_faults),
            "schema_payload_gap_count": len(schema_shape_gaps),
        },
        "sources": sources,
        "record_counts": record_counts(deduped),
        "claim_ledger_entries": groups["claim_ledger_entries"],
        "semantic_ir_records": groups["semantic_ir_records"],
        "simulation_fidelity_records": groups["simulation_fidelity_records"],
        "governance_records": groups["governance_records"],
        "failure_boundaries": groups["failure_boundaries"],
        "evidence_transitions": groups["evidence_transitions"],
        "artifact_records": groups["artifact_records"],
        "authority_records": groups["authority_records"],
        "runtime_adapter_records": groups["runtime_adapter_records"],
        "resource_route_records": groups["resource_route_records"],
        "generation_mode_records": groups["generation_mode_records"],
        "context_records": groups["context_records"],
        "compression_records": groups["compression_records"],
        "compressed_artifact_records": groups["compressed_artifact_records"],
        "compression_receipts": groups["compression_receipts"],
        "compact_generative_records": groups["compact_generative_records"],
        "defeater_records": groups["defeater_records"],
        "records": deduped,
        "schema_payload_gaps": schema_shape_gaps[:100],
        "boundaries": {
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "raw_prompt_payload_stored": False,
            "runtime_external_inference_served": False,
        },
        "non_claims": [
            "This materialized view proves spine cohesion and traceability, not learned-generation capability.",
            "Tool and deterministic records stay tool-assisted evidence and cannot support learned-generation promotion claims.",
            "Planner compile-time records require execution/verifier receipts before they support completion claims.",
        ],
    }


def source_descriptor(profile: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "profile_id": profile_id(profile),
        "producer_surface": str(profile.get("producer_surface") or ""),
        "source_kind": str(profile.get("source_kind") or ""),
        "path": rel(path),
        "present": path.exists(),
    }


def materialize_nested(
    record_type: str,
    value: Any,
    aliases: dict[str, str],
    profile: dict[str, Any],
    path: Path,
    locator: str,
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            if isinstance(item, dict):
                rows.append(materialized_record(item.get("record_type") or record_type, {**parent, **item}, aliases, profile, path, f"{locator}:{index}"))
        return rows
    if isinstance(value, dict):
        return [materialized_record(value.get("record_type") or record_type, {**parent, **value}, aliases, profile, path, locator)]
    return []


def materialized_record(
    record_type: Any,
    payload: dict[str, Any],
    aliases: dict[str, str],
    profile: dict[str, Any],
    path: Path,
    locator: str,
) -> dict[str, Any]:
    canonical = viea_spine_records.canonical_record_type(record_type, aliases)
    normalized_payload = normalize_schema_payload(canonical, payload, profile, path, locator)
    compact = viea_spine_records.compact_record_payload(normalized_payload)
    record_id = str(
        compact.get("record_id")
        or compact.get("claim_id")
        or compact.get("proof_claim_id")
        or compact.get("predicate_id")
        or compact.get("right_type")
        or compact.get("artifact_id")
        or compact.get("transaction_id")
        or compact.get("simulation_id")
        or compact.get("failure_id")
        or viea_spine_records.stable_id("spine_record", profile_id(profile), rel(path), locator, canonical, normalized_payload)
    )
    return {
        "record_id": record_id,
        "canonical_record_type": canonical,
        "source_record_type": str(record_type or ""),
        "producer_surface": str(profile.get("producer_surface") or ""),
        "source_profile": profile_id(profile),
        "source_path": rel(path),
        "source_locator": locator,
        "content_hash": viea_spine_records.stable_hash(normalized_payload),
        "compact_payload": compact,
    }


def normalize_schema_payload(
    canonical: str,
    payload: dict[str, Any],
    profile: dict[str, Any],
    path: Path,
    locator: str,
) -> dict[str, Any]:
    row = dict(payload)
    if canonical == "artifact_graph_record":
        return normalize_artifact_graph_payload(row, profile, path, locator)
    if canonical == "context_transaction":
        return normalize_context_transaction_payload(row, profile, path, locator)
    if canonical == "costed_route":
        return normalize_costed_route_payload(row, profile, path, locator)
    return row


def normalize_artifact_graph_payload(
    row: dict[str, Any],
    profile: dict[str, Any],
    path: Path,
    locator: str,
) -> dict[str, Any]:
    artifacts = list_values(row.get("artifacts"))
    source_refs = list_values(row.get("source_refs"))
    if not source_refs:
        source_refs = list_values(row.get("parents"))
    if not source_refs:
        source_refs = [rel(path)]
    replay_limits = row.get("replay_limits")
    if isinstance(replay_limits, str):
        replay_limits = [replay_limits]
    if not isinstance(replay_limits, list):
        replay_limits = [
            "source report must remain available",
            "raw prompt/private text is not materialized",
        ]
    evidence_gate = row.get("evidence_gate")
    if not isinstance(evidence_gate, dict):
        evidence_gate = {
            "state": str(row.get("support_state") or row.get("verifier_state") or "STRUCTURAL_TRACE_ONLY"),
            "public_training_rows_written": viea_spine_records.int_or(row.get("public_training_rows_written"), 0),
            "external_inference_calls": viea_spine_records.int_or(row.get("external_inference_calls"), 0),
            "fallback_return_count": viea_spine_records.int_or(row.get("fallback_return_count"), 0),
        }
    return {
        **row,
        "record_type": row.get("record_type") or "artifact_graph_record",
        "artifact_id": str(
            row.get("artifact_id")
            or row.get("record_id")
            or viea_spine_records.stable_id("artifact", profile_id(profile), rel(path), locator, row)
        ),
        "artifact_type": str(row.get("artifact_type") or row.get("kind") or "materialized_viea_artifact_graph"),
        "parent_job": str(
            row.get("parent_job")
            or row.get("typed_job_id")
            or row.get("job_id")
            or row.get("node_id")
            or row.get("case_id")
            or row.get("run_id")
            or profile_id(profile)
        ),
        "source_refs": source_refs,
        "context_refs": list_values(row.get("context_refs")),
        "context_transaction_refs": list_values(row.get("context_transaction_refs")),
        "semantic_certificate_refs": list_values(row.get("semantic_certificate_refs"))
        or list_values(row.get("context_adequacy_refs"))
        or list_values(row.get("vcm_governor_refs")),
        "tool_refs": list_values(row.get("tool_refs")),
        "claim_refs": list_values(row.get("claim_refs")),
        "test_refs": list_values(row.get("test_refs")),
        "audit_events": list_values(row.get("audit_events")) or ["materialized_by_viea_spine_record_gate"],
        "replay_metadata": dict_value(row.get("replay_metadata"))
        or {
            "source_profile": profile_id(profile),
            "source_path": rel(path),
            "source_locator": locator,
            "source_content_hash": viea_spine_records.stable_hash(row),
            "artifact_count": len(artifacts),
        },
        "replay_grade": str(row.get("replay_grade") or "metadata_replayable_with_source_report"),
        "environment_assumptions": dict_value(row.get("environment_assumptions"))
        or {
            "producer_surface": str(profile.get("producer_surface") or ""),
            "source_kind": str(profile.get("source_kind") or ""),
            "materialized_view": "reports/viea_spine_materialized_view.json",
        },
        "provenance_status": str(row.get("provenance_status") or "materialized_from_registered_viea_source"),
        "replay_limits": replay_limits,
        "evidence_gate": evidence_gate,
        "residuals": list_values(row.get("residuals")),
        "non_claims": list_values(row.get("non_claims"))
        or [
            "not learned-generation evidence",
            "not a model-quality claim",
            "not a public benchmark result",
        ],
    }


def normalize_context_transaction_payload(
    row: dict[str, Any],
    profile: dict[str, Any],
    path: Path,
    locator: str,
) -> dict[str, Any]:
    faults = list_values(row.get("faults"))
    read_set = list_values(row.get("read_set"))
    if not read_set and row.get("context_hash"):
        read_set = [row.get("context_hash")]
    snapshot_id = str(row.get("snapshot_id") or row.get("context_hash") or row.get("run_id") or "no_snapshot")
    closure_state = row.get("closure_state")
    if not closure_state:
        closure_state = "closed" if not faults else "open_with_faults"
    return {
        **row,
        "record_type": row.get("record_type") or "context_transaction_record",
        "transaction_id": str(
            row.get("transaction_id")
            or row.get("record_id")
            or viea_spine_records.stable_id("context_transaction", profile_id(profile), rel(path), locator, row)
        ),
        "operation": str(row.get("operation") or ("read" if row.get("context_consumed") else "materialize")),
        "snapshot_id": snapshot_id,
        "mounts": list_values(row.get("mounts")),
        "read_set": read_set,
        "write_set": list_values(row.get("write_set")),
        "branch_policy": str(row.get("branch_policy") or "read_only_materialization"),
        "taint_labels": list_values(row.get("taint_labels")),
        "deletion_obligations": row.get("deletion_obligations")
        if isinstance(row.get("deletion_obligations"), (dict, list))
        else {
            "status": row.get("deletion_obligations") or row.get("deletion_closure_status") or "not_applicable",
        },
        "declassification_refs": list_values(row.get("declassification_refs")),
        "derivative_refs": list_values(row.get("derivative_refs")),
        "contradiction_refs": list_values(row.get("contradiction_refs")),
        "materialization_state": str(
            row.get("materialization_state")
            or ("materialized" if row.get("context_consumed") is not False else "not_materialized")
        ),
        "closure_state": str(closure_state),
        "faults": faults,
        "audit_refs": list_values(row.get("audit_refs")) or [rel(path)],
        "replay_boundary": str(row.get("replay_boundary") or "metadata_only_no_raw_prompt_or_private_text"),
        "non_claims": list_values(row.get("non_claims"))
        or [
            "not raw memory export",
            "not permission to reconstruct redacted text",
            "not native KV cache parity evidence",
        ],
    }


def normalize_costed_route_payload(
    row: dict[str, Any],
    profile: dict[str, Any],
    path: Path,
    locator: str,
) -> dict[str, Any]:
    task_id = str(
        row.get("task_id")
        or row.get("node_id")
        or row.get("route_id")
        or row.get("placement_id")
        or row.get("job_id")
        or row.get("case_id")
        or locator
    )
    selected_route = str(
        row.get("selected_route")
        or row.get("backend")
        or row.get("tool_id")
        or row.get("node_name")
        or row.get("route_phase")
        or row.get("task_kind")
        or profile_id(profile)
    )
    candidate_routes = list_values(row.get("candidate_routes"))
    if not candidate_routes:
        candidate_routes = [selected_route]
    rejected = list_values(row.get("rejected_lower_cost_routes")) or list_values(row.get("rejected_routes"))
    verification_result = str(
        row.get("verification_result")
        or row.get("verifier_state")
        or row.get("status")
        or "not_executed_metadata_route"
    )
    outcome_state = str(row.get("outcome_state") or row.get("status") or row.get("state") or "not_evaluated")
    cost_accounting = dict_value(row.get("cost_accounting"))
    if not cost_accounting:
        cost_accounting = {
            "estimated_latency_ms": viea_spine_records.int_or(row.get("estimated_latency_ms"), 0),
            "gas_estimate_micro_twc": viea_spine_records.int_or(row.get("gas_estimate_micro_twc"), 0),
            "provider_payout_micro_twc": viea_spine_records.int_or(row.get("provider_payout_micro_twc"), 0),
            "tool_count": viea_spine_records.int_or(row.get("tool_count"), 0),
        }
    return {
        **row,
        "record_type": row.get("record_type") or "costed_route_record",
        "task_id": task_id,
        "route_state": str(row.get("route_state") or row.get("state") or "candidate"),
        "task_contract_ref": str(
            row.get("task_contract_ref")
            or row.get("command_contract_ref")
            or row.get("goal_id")
            or row.get("task_kind")
            or rel(path)
        ),
        "quality_predicate": str(
            row.get("quality_predicate")
            or "selected route must preserve authority, readiness, verification, and no-cheat boundaries"
        ),
        "authority_ceiling": str(row.get("authority_ceiling") or row.get("authority_scope") or "registered_local_or_policy_allowed_task"),
        "candidate_routes": candidate_routes,
        "selected_route": selected_route,
        "rejected_lower_cost_routes": rejected,
        "verification_result": verification_result,
        "outcome_state": outcome_state,
        "cost_accounting": cost_accounting,
        "cost_classes": list_values(row.get("cost_classes"))
        or [
            "runtime",
            "context construction",
            "verification",
            "coordination",
            "residual",
        ],
        "hidden_cost_checks": list_values(row.get("hidden_cost_checks"))
        or [
            "authority boundary preserved",
            "verification cost remains explicit",
            "fallback/tool/router evidence is not learned-generation credit",
        ],
        "residual_obligations": list_values(row.get("residual_obligations"))
        or list_values(row.get("residuals"))
        or ["execution and verifier receipts required before completion or promotion claims"],
        "fallback_route": str(row.get("fallback_route") or row.get("fallback_backend") or "manual_or_full_verifier_route"),
        "promotion_candidate": bool(row.get("promotion_candidate", False)),
        "support_state_effect": str(row.get("support_state_effect") or "record_shape_only"),
        "non_claims": list_values(row.get("non_claims"))
        or [
            "costed route is not execution success",
            "route cost is not model capability evidence",
            "router/tool selection is not learned generation",
        ],
    }


def schema_payload_gaps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = {
        "artifact_graph_record": ARTIFACT_GRAPH_REQUIRED_FIELDS,
        "context_transaction": CONTEXT_TRANSACTION_REQUIRED_FIELDS,
        "costed_route": COSTED_ROUTE_REQUIRED_FIELDS,
        "compressed_artifact_record": COMPRESSED_ARTIFACT_REQUIRED_FIELDS,
        "compression_receipt": COMPRESSION_RECEIPT_REQUIRED_FIELDS,
        "compact_generative_record": COMPACT_GENERATIVE_REQUIRED_FIELDS,
    }
    gaps: list[dict[str, Any]] = []
    for row in records:
        canonical = str(row.get("canonical_record_type") or "")
        required = checks.get(canonical)
        if not required:
            continue
        payload = row.get("compact_payload") if isinstance(row.get("compact_payload"), dict) else {}
        missing = sorted(key for key in required if key not in payload)
        if missing:
            gaps.append(
                {
                    "record_id": row.get("record_id"),
                    "canonical_record_type": canonical,
                    "missing_fields": missing,
                    "source_path": row.get("source_path"),
                    "source_locator": row.get("source_locator"),
                }
            )
    return gaps


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in records:
        key = (row.get("record_id"), row.get("canonical_record_type"), row.get("content_hash"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def group_materialized_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    def pick(types: set[str]) -> list[dict[str, Any]]:
        return [row for row in records if row.get("canonical_record_type") in types]

    return {
        "claim_ledger_entries": pick({"claim_record", "proof_carrying_claim"}),
        "semantic_ir_records": pick({"semantic_atom", "semantic_node", "routing_decision"}),
        "simulation_fidelity_records": pick({"simulation_contract", "fidelity_record", "counterfactual_trace", "world_adapter_receipt"}),
        "governance_records": pick({"constitutional_predicate", "agency_rights_checklist", "value_conflict_record", "governance_right"}),
        "failure_boundaries": pick({"failure_boundary"}),
        "evidence_transitions": pick({"evidence_transition_record"}),
        "artifact_records": pick({"artifact_graph_record"}),
        "authority_records": pick({"authority_transition", "authority_use_receipt"}),
        "runtime_adapter_records": pick({"runtime_adapter_invocation"}),
        "resource_route_records": pick({"resource_budget", "costed_route"}),
        "generation_mode_records": pick({"generation_mode"}),
        "context_records": pick({"context_abi_record", "context_transaction", "context_adequacy"}),
        "compression_records": pick({"compression_record"}),
        "compressed_artifact_records": pick({"compressed_artifact_record"}),
        "compression_receipts": pick({"compression_receipt"}),
        "compact_generative_records": pick({"compact_generative_record"}),
        "defeater_records": pick({"defeater_record"}),
    }


def record_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        key = str(row.get("canonical_record_type") or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def profile_result(profile: dict[str, Any], passed: bool, details: Any, hard_gaps: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": profile_id(profile),
        "producer_surface": str(profile.get("producer_surface") or ""),
        "source_kind": str(profile.get("source_kind") or ""),
        "required": bool(profile.get("required") is True),
        "passed": passed,
        "details": details,
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def no_cheat_faults(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    faults = []
    for index, row in enumerate(rows):
        for key in viea_spine_records.NO_CHEAT_COUNTERS:
            if viea_spine_records.int_or(row.get(key), 0) != 0:
                faults.append({"row_index": index, "counter": key, "value": row.get(key)})
    return faults


def no_cheat_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: viea_spine_records.int_or(payload.get(key), 0) for key in viea_spine_records.NO_CHEAT_COUNTERS}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VIEA Spine Record Gate",
        "",
        f"- Trigger state: `{report['trigger_state']}`",
        f"- Profiles: `{report['summary']['passing_profile_count']}/{report['summary']['profile_count']}`",
        f"- Hard gaps: `{report['summary']['hard_gap_count']}`",
        f"- Warnings: `{report['summary']['warning_count']}`",
        "",
        "| Profile | Producer | Source | Passed | Hard gaps | Warnings |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for profile in report["profiles"]:
        lines.append(
            f"| `{profile['id']}` | `{profile['producer_surface']}` | `{profile['source_kind']}` | "
            f"`{profile['passed']}` | {profile['hard_gap_count']} | {profile['warning_count']} |"
        )
    if report["hard_gaps"]:
        lines.extend(["", "## Hard Gaps", ""])
        for item in report["hard_gaps"][:80]:
            lines.append(f"- `{item['id']}` {item['kind']}: {json.dumps(item.get('evidence', {}), sort_keys=True)}")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for item in report["warnings"][:80]:
            lines.append(f"- `{item['id']}` {item['kind']}: {json.dumps(item.get('evidence', {}), sort_keys=True)}")
    lines.append("")
    return "\n".join(lines)


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report["trigger_state"],
        "profile_count": report["summary"]["profile_count"],
        "passing_profile_count": report["summary"]["passing_profile_count"],
        "materialized_record_count": report["summary"].get("materialized_record_count", 0),
        "materialized_claim_ledger_entry_count": report["summary"].get("materialized_claim_ledger_entry_count", 0),
        "materialized_compression_record_count": report["summary"].get("materialized_compression_record_count", 0),
        "materialized_defeater_record_count": report["summary"].get("materialized_defeater_record_count", 0),
        "hard_gap_count": report["summary"]["hard_gap_count"],
        "warning_count": report["summary"]["warning_count"],
    }


def gap(identifier: str, kind: str, evidence: dict[str, Any], *, severity: str = "hard") -> dict[str, Any]:
    return {"id": identifier, "kind": kind, "severity": severity, "evidence": evidence}


def profile_id(profile: dict[str, Any]) -> str:
    return str(profile.get("id") or "unknown_profile")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {"record_type": "json_decode_error", "raw": line[:200]}
            if isinstance(row, dict):
                rows.append(row)
    return rows


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def list_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
