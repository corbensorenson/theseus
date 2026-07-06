#!/usr/bin/env python3
"""Unified control-plane snapshot for Project Theseus.

The control plane is not another trainer. It is the thin operational spine that
turns volatile reports into typed state, appends action decisions to a durable
ledger, and keeps heavyweight work behind one idempotent gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402
from code_lm_process_guard import (  # noqa: E402
    duplicate_code_lm_artifact_targets,
    windows_active_code_lm_process_rows,
)


REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "theseus_control_plane.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_control_plane.md"
DEFAULT_DB = report_evidence_store.DEFAULT_DB
CONTROL_PLANE_POLICY = "project_theseus_control_plane_v1"
ACTIVE_LEDGER_STATUSES = {"reserved", "running"}
HEAVY_ACTION_TYPES = {"heavy_code_worker", "code_lm_train_once_fanout", "code_lm_closure"}


@dataclass(frozen=True)
class ReportSpec:
    report_id: str
    path: Path
    role: str
    max_age_hours: float
    required: bool = True


CONTROL_REPORT_SPECS: tuple[ReportSpec, ...] = (
    ReportSpec("autonomy_watchdog", REPORTS / "autonomy_watchdog.json", "watchdog", 8.0),
    ReportSpec("learning_launch_supervisor", REPORTS / "learning_launch_supervisor.json", "supervisor", 12.0),
    ReportSpec("vacation_mode_supervisor", REPORTS / "vacation_mode_supervisor_overnight.json", "supervisor", 12.0, False),
    ReportSpec("service_process_hygiene", REPORTS / "service_process_hygiene.json", "services", 12.0),
    ReportSpec("resource_aware_execution_policy", REPORTS / "resource_aware_execution_policy.json", "resource", 8.0),
    ReportSpec("windows_cuda_doctor", REPORTS / "windows_cuda_doctor.json", "cuda", 24.0, False),
    ReportSpec("system_efficiency_audit", REPORTS / "system_efficiency_audit.json", "efficiency", 12.0),
    ReportSpec("asi_wall_breaker_governor", REPORTS / "asi_wall_breaker_governor.json", "governor", 12.0),
    ReportSpec("closed_loop_residual_ratchet", REPORTS / "closed_loop_residual_ratchet.json", "ratchet", 24.0, False),
    ReportSpec("theseus_project_registry", REPORTS / "theseus_project_registry.json", "project_registry", 24.0, False),
    ReportSpec("viea_spine_materialized_view", REPORTS / "viea_spine_materialized_view.json", "spine_view", 24.0),
    ReportSpec("teacher_distillation_gate", REPORTS / "teacher_distillation_gate.json", "teacher_governance", 24.0, False),
    ReportSpec("teacher_share_ledger_summary", REPORTS / "teacher_share_ledger_summary.json", "teacher_governance", 24.0, False),
    ReportSpec("theseus_workspace_hygiene_audit", REPORTS / "theseus_workspace_hygiene_audit.json", "workspace_hygiene", 24.0, False),
    ReportSpec("theseus_doc_link_audit", REPORTS / "theseus_doc_link_audit.json", "workspace_hygiene", 24.0, False),
    ReportSpec("theseus_deprecation_registry", REPORTS / "theseus_deprecation_registry.json", "deprecation_registry", 24.0, False),
    ReportSpec("theseus_artifact_retention", REPORTS / "theseus_artifact_retention.json", "artifact_retention", 24.0, False),
    ReportSpec("theseus_generated_artifact_gc", REPORTS / "theseus_generated_artifact_gc.json", "artifact_gc", 24.0, False),
    ReportSpec("theseus_dirty_workspace_review", REPORTS / "theseus_dirty_workspace_review.json", "dirty_workspace", 24.0, False),
    ReportSpec("attd_report", REPORTS / "attd_report.json", "attd", 24.0),
    ReportSpec("attd_maintenance_packets", REPORTS / "attd_maintenance_packets.json", "attd", 24.0),
    ReportSpec("theseus_plan_compiler", REPORTS / "theseus_plan_compiler.json", "planning", 24.0, False),
    ReportSpec("hive_work_board_executor", REPORTS / "hive_work_board_executor.json", "work_board", 12.0),
    ReportSpec("candidate_promotion_gate", REPORTS / "candidate_promotion_gate.json", "promotion", 24.0),
    ReportSpec(
        "private_candidate_replay_contract",
        REPORTS / "private_candidate_replay_contract_audit_v1.json",
        "code_gate",
        24.0,
        False,
    ),
    ReportSpec(
        "full_body_contract_transfer_recovery",
        REPORTS / "full_body_contract_transfer_recovery_v1.json",
        "code_gate",
        24.0,
        False,
    ),
    ReportSpec(
        "private_full_body_repair_runtime_readiness",
        REPORTS / "private_full_body_repair_runtime_readiness_v1.json",
        "code_gate",
        24.0,
        False,
    ),
    ReportSpec(
        "broad_capability_survival_promotion_gate",
        REPORTS / "broad_capability_survival_promotion_gate_v1.json",
        "promotion",
        24.0,
        False,
    ),
    ReportSpec(
        "broad_capability_survival_lane_decision",
        REPORTS / "broad_capability_survival_lane_decision_v1.json",
        "promotion",
        24.0,
        False,
    ),
    ReportSpec("coherence_delirium_gate", REPORTS / "coherence_delirium_gate.json", "promotion", 24.0, False),
    ReportSpec("maturity_integrity_audit", REPORTS / "maturity_integrity_audit.json", "integrity", 24.0),
    ReportSpec("agent_lane_transfer_gate", REPORTS / "agent_lane_transfer_gate.json", "agent_lanes", 24.0),
    ReportSpec("decoder_v2_private_ablation_gate", REPORTS / "decoder_v2_private_ablation_gate.json", "code_gate", 24.0),
    ReportSpec("private_public_transfer_proof", REPORTS / "private_public_transfer_proof.json", "code_gate", 24.0),
    ReportSpec("broad_transfer_matrix", REPORTS / "broad_transfer_matrix.json", "transfer", 24.0),
    ReportSpec("code_lm_train_once_fanout", REPORTS / "code_lm_train_once_fanout.json", "code_lm", 24.0, False),
    ReportSpec("a_plus_operating_scorecard", REPORTS / "a_plus_operating_scorecard.json", "scorecard", 24.0, False),
    ReportSpec("report_evidence_store", REPORTS / "report_evidence_store.json", "evidence", 24.0),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--action-key", default="", help="Optional action/job key to reserve in the control ledger.")
    parser.add_argument("--action-type", default="", help="Optional action type, e.g. heavy_code_worker.")
    parser.add_argument("--command", default="", help="Command associated with the requested action lease.")
    parser.add_argument("--lease-seconds", type=int, default=21600)
    parser.add_argument("--no-ingest", action="store_true", help="Skip evidence-store refresh for diagnostic runs.")
    args = parser.parse_args()

    started = time.perf_counter()
    db_path = resolve(args.db)
    control_paths = [spec.path for spec in CONTROL_REPORT_SPECS if spec.path.exists()]
    if not args.no_ingest:
        report_evidence_store.ingest_reports(db_path, control_paths)

    conn = report_evidence_store.connect(db_path)
    try:
        ensure_action_schema(conn)
        state = observe_state(db_path, control_paths)
        requested_action = {}
        if args.action_key or args.action_type or args.command:
            requested_action = request_action(
                conn,
                action_key=args.action_key or stable_id("action", args.action_type, args.command)[:16],
                action_type=args.action_type or "unknown",
                command=args.command,
                gates=state["gates"],
                active_workers=state["active_workers"]["code_lm"],
                lease_seconds=max(1, int(args.lease_seconds)),
            )
        action_decisions = build_action_decisions(state)
        append_control_run(conn, state, action_decisions, requested_action)
        ledger_summary = action_ledger_summary(conn)
        conn.commit()
    finally:
        conn.close()

    payload = build_report_payload(state, action_decisions, requested_action, ledger_summary, started)
    report_evidence_store.write_json_report(
        resolve(args.out),
        payload,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(payload),
        db_path=db_path,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] != "RED" else 2


def observe_state(db_path: Path, control_paths: list[Path]) -> dict[str, Any]:
    report_records = build_report_records(CONTROL_REPORT_SPECS)
    report_map = {row["id"]: row for row in report_records}
    payloads = {row["id"]: row.get("payload", {}) for row in report_records}
    current_pid = os.getpid()
    active_code_lm = [
        row for row in windows_active_code_lm_process_rows() if int(row.get("pid") or -1) != current_pid
    ]
    duplicate_targets = duplicate_code_lm_artifact_targets(active_code_lm, ROOT)
    store_summary = report_evidence_store.evidence_store_summary(db_path)
    current_index = report_evidence_store.current_report_index(db_path, control_paths)
    gates = build_gates(payloads, active_code_lm, duplicate_targets, current_index)
    maintenance_queue = build_maintenance_queue(payloads)
    stale_reports = [compact_report_record(row) for row in report_records if row.get("stale") or row.get("missing")]
    blockers = build_blockers(payloads, stale_reports, duplicate_targets, current_index, gates)
    typed_records = build_typed_records(report_records, gates, maintenance_queue, blockers, active_code_lm)
    return {
        "reports": [strip_payload(row) for row in report_records],
        "report_map": {key: strip_payload(value) for key, value in report_map.items()},
        "report_payloads": payloads,
        "stale_reports": stale_reports,
        "active_workers": {
            "code_lm": compact_process_rows(active_code_lm),
            "duplicate_code_lm_artifact_targets": duplicate_targets,
            "active_code_lm_process_count": len(active_code_lm),
        },
        "evidence_store": {
            "database": rel_or_abs(db_path),
            "summary": store_summary,
            "current_index": compact_current_index(current_index),
        },
        "gates": gates,
        "blockers": blockers,
        "maintenance_queue": maintenance_queue,
        "typed_records": typed_records,
    }


def build_report_records(specs: tuple[ReportSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        payload = read_json(spec.path, {})
        exists = spec.path.exists()
        created = report_created_utc(payload, spec.path if exists else None)
        age_hours = age_hours_since(created)
        stale = bool(exists and age_hours is not None and age_hours > spec.max_age_hours)
        missing = bool(spec.required and not exists)
        rows.append(
            {
                "record_type": "report_state",
                "id": spec.report_id,
                "path": rel_or_abs(spec.path),
                "role": spec.role,
                "required": spec.required,
                "exists": exists,
                "missing": missing,
                "created_utc": format_dt(created) if created else "",
                "age_hours": round(float(age_hours), 3) if age_hours is not None else None,
                "max_age_hours": spec.max_age_hours,
                "stale": stale,
                "trigger_state": str(payload.get("trigger_state") or ""),
                "passed": payload.get("passed"),
                "policy": str(payload.get("policy") or ""),
                "summary": compact_summary(payload.get("summary") if isinstance(payload.get("summary"), dict) else {}),
                "payload": payload,
            }
        )
    return rows


def current_private_transfer_contract_evidence(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    legacy = payloads.get("private_public_transfer_proof", {})
    legacy_summary = as_dict(legacy.get("summary"))
    legacy_ok = bool(
        legacy.get("trigger_state") == "GREEN"
        and (legacy.get("ready_for_public_calibration") is True or legacy_summary.get("ready_for_public_calibration") is True)
    )

    replay = payloads.get("private_candidate_replay_contract", {})
    replay_summary = as_dict(replay.get("summary"))
    replay_ok = bool(
        replay.get("trigger_state") == "GREEN"
        and float_value(replay_summary.get("selected_intended_behavior_pass_rate")) >= 1.0
        and int_value(replay_summary.get("unexplained_no_candidate_count")) == 0
        and int_value(replay_summary.get("fallback_return_candidate_count")) == 0
    )

    recovery = payloads.get("full_body_contract_transfer_recovery", {})
    recovery_summary = as_dict(recovery.get("summary"))
    required_rows = int_value(recovery_summary.get("required_readiness_eval_rows"))
    private_rows = int_value(recovery_summary.get("private_eval_rows"))
    recovery_ok = bool(
        recovery.get("trigger_state") == "GREEN"
        and recovery_summary.get("ready_for_future_governed_public_calibration") is True
        and float_value(recovery_summary.get("full_contract_selected_pass_rate")) >= 0.95
        and float_value(recovery_summary.get("selected_pass_delta_full_minus_minimal")) > 0.0
        and (required_rows == 0 or private_rows >= required_rows)
        and int_value(recovery_summary.get("fallback_return_count")) == 0
        and int_value(recovery_summary.get("template_like_candidate_count")) == 0
        and int_value(recovery_summary.get("public_training_rows")) == 0
        and int_value(recovery_summary.get("external_inference_calls")) == 0
    )

    readiness = payloads.get("private_full_body_repair_runtime_readiness", {})
    readiness_summary = as_dict(readiness.get("summary"))
    readiness_ok = bool(
        readiness.get("trigger_state") == "GREEN"
        and int_value(readiness_summary.get("hard_failure_count")) == 0
        and readiness_summary.get("no_public_calibration_run") is True
        and int_value(readiness_summary.get("public_training_rows_written")) == 0
        and int_value(readiness_summary.get("external_inference_calls")) == 0
        and int_value(readiness_summary.get("fallback_return_count")) == 0
    )
    current_ok = bool(replay_ok and recovery_ok and readiness_ok)
    return {
        "passed": bool(legacy_ok or current_ok),
        "source": "legacy_private_public_transfer_proof" if legacy_ok else "current_replay_full_body_repair_readiness",
        "legacy_ready": legacy_ok,
        "current_ready": current_ok,
        "candidate_replay_green": replay_ok,
        "full_body_recovery_green": recovery_ok,
        "private_runtime_readiness_green": readiness_ok,
        "candidate_replay_path": "reports/private_candidate_replay_contract_audit_v1.json",
        "full_body_recovery_path": "reports/full_body_contract_transfer_recovery_v1.json",
        "private_runtime_readiness_path": "reports/private_full_body_repair_runtime_readiness_v1.json",
        "selected_intended_behavior_pass_rate": replay_summary.get("selected_intended_behavior_pass_rate"),
        "full_contract_selected_pass_rate": recovery_summary.get("full_contract_selected_pass_rate"),
        "selected_pass_delta_full_minus_minimal": recovery_summary.get("selected_pass_delta_full_minus_minimal"),
        "hard_failure_count": readiness_summary.get("hard_failure_count"),
        "public_training_rows_written": readiness_summary.get("public_training_rows_written"),
        "fallback_return_count": readiness_summary.get("fallback_return_count"),
        "score_semantics": "private-only transfer-readiness evidence; not a public benchmark run or calibration unlock",
    }


def float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_gates(
    payloads: dict[str, dict[str, Any]],
    active_code_lm: list[dict[str, Any]],
    duplicate_targets: list[dict[str, Any]],
    current_index: dict[str, Any],
) -> dict[str, Any]:
    asi_summary = as_dict(payloads.get("asi_wall_breaker_governor", {}).get("summary"))
    resource = payloads.get("resource_aware_execution_policy", {})
    resource_summary = as_dict(resource.get("summary"))
    budget = as_dict(resource.get("recommended_code_lm_budget"))
    decoder = payloads.get("decoder_v2_private_ablation_gate", {})
    transfer = payloads.get("private_public_transfer_proof", {})
    maturity = payloads.get("maturity_integrity_audit", {})
    candidate = payloads.get("candidate_promotion_gate", {})
    project_registry = payloads.get("theseus_project_registry", {})
    project_summary = as_dict(project_registry.get("summary"))
    spine_view = payloads.get("viea_spine_materialized_view", {})
    spine_summary = as_dict(spine_view.get("summary"))
    teacher_share = payloads.get("teacher_share_ledger_summary", {})
    teacher_share_summary = as_dict(teacher_share.get("summary"))
    ratchet = payloads.get("closed_loop_residual_ratchet", {})
    ratchet_summary = as_dict(ratchet.get("summary"))
    ratchet_decision = as_dict(ratchet.get("decision"))

    decoder_ready = bool(decoder.get("ready_for_public_calibration"))
    transfer_contract = current_private_transfer_contract_evidence(payloads)
    transfer_ready = bool(transfer_contract.get("passed"))
    public_calibration_allowed = bool(asi_summary.get("public_calibration_allowed")) and decoder_ready and transfer_ready
    model_growth_allowed = bool(asi_summary.get("model_growth_allowed"))
    candidate_promotion_allowed = bool(asi_summary.get("candidate_promotion_allowed")) and bool(candidate.get("promote"))
    resource_allows_code = bool(budget.get("start_new_train_once_fanout") or budget.get("start_new_code_closure"))
    no_active_code_worker = not active_code_lm and not duplicate_targets
    heavy_code_work_allowed = bool(resource_allows_code and no_active_code_worker)
    evidence_store_clean = (
        int(current_index.get("current_unstored_count") or 0) == 0
        and int(current_index.get("current_truncated_without_snapshot_count") or 0) == 0
    )
    registry_governance_ready = bool(
        project_registry
        and project_registry.get("trigger_state") != "RED"
        and int(project_summary.get("abstraction_registry_gap_count") or 0) == 0
        and int(project_summary.get("stable_capability_field_gap_count") or 0) == 0
        and int(project_summary.get("stable_capability_field_health_red_count") or 0) == 0
        and int(project_summary.get("implementation_routing_blocker_count") or 0) == 0
        and int(project_summary.get("registry_hard_governance_violation_count") or 0) == 0
    )
    spine_view_ready = bool(
        spine_view
        and spine_view.get("trigger_state") == "GREEN"
        and int(spine_summary.get("record_count") or 0) > 0
        and int(spine_summary.get("claim_ledger_entry_count") or 0) > 0
        and int(spine_summary.get("semantic_ir_record_count") or 0) > 0
        and int(spine_summary.get("simulation_fidelity_record_count") or 0) > 0
        and int(spine_summary.get("governance_record_count") or 0) > 0
        and int(spine_summary.get("failure_boundary_count") or 0) > 0
        and int(spine_summary.get("no_cheat_fault_count") or 0) == 0
    )
    teacher_share_ready = bool(
        teacher_share
        and teacher_share.get("trigger_state") == "GREEN"
        and teacher_share_summary.get("metric_ready") is True
        and teacher_share_summary.get("teacher_share_within_cap") is True
        and int(teacher_share_summary.get("runtime_external_inference_calls") or 0) == 0
        and int(teacher_share_summary.get("public_training_rows_written") or 0) == 0
        and int(teacher_share_summary.get("teacher_accepted_rows") or 0) >= 0
    )

    return {
        "registry_governance_ready": gate_record(
            registry_governance_ready,
            "self-improvement/routing decisions require a current registry with abstraction and implementation contracts intact",
            {
                "registry_present": bool(project_registry),
                "registry_trigger_state": project_registry.get("trigger_state"),
                "abstraction_registry_gap_count": int(project_summary.get("abstraction_registry_gap_count") or 0),
                "stable_capability_field_gap_count": int(project_summary.get("stable_capability_field_gap_count") or 0),
                "stable_capability_field_health_red_count": int(project_summary.get("stable_capability_field_health_red_count") or 0),
                "implementation_routing_blocker_count": int(project_summary.get("implementation_routing_blocker_count") or 0),
                "registry_hard_governance_violation_count": int(project_summary.get("registry_hard_governance_violation_count") or 0),
                "routing_eligible_implementation_count": int(project_summary.get("routing_eligible_implementation_count") or 0),
                "registry_cleanup_queue_count": int(project_summary.get("registry_cleanup_queue_count") or 0),
            },
        ),
        "viea_spine_materialized_view_ready": gate_record(
            spine_view_ready,
            "route and governance decisions require the shared VIEA claim/evidence/semantic/governance/failure view",
            {
                "view_present": bool(spine_view),
                "view_trigger_state": spine_view.get("trigger_state"),
                "record_count": int(spine_summary.get("record_count") or 0),
                "claim_ledger_entry_count": int(spine_summary.get("claim_ledger_entry_count") or 0),
                "semantic_ir_record_count": int(spine_summary.get("semantic_ir_record_count") or 0),
                "simulation_fidelity_record_count": int(spine_summary.get("simulation_fidelity_record_count") or 0),
                "governance_record_count": int(spine_summary.get("governance_record_count") or 0),
                "failure_boundary_count": int(spine_summary.get("failure_boundary_count") or 0),
                "no_cheat_fault_count": int(spine_summary.get("no_cheat_fault_count") or 0),
            },
        ),
        "teacher_share_ledger_ready": gate_record(
            teacher_share_ready,
            "teacher governance requires a durable ledger summary with runtime external serving forbidden and public training rows at zero",
            {
                "report_present": bool(teacher_share),
                "trigger_state": teacher_share.get("trigger_state"),
                "metric_ready": teacher_share_summary.get("metric_ready"),
                "accepted_training_rows": teacher_share_summary.get("accepted_training_rows"),
                "teacher_accepted_rows": teacher_share_summary.get("teacher_accepted_rows"),
                "verified_self_generated_rows": teacher_share_summary.get("verified_self_generated_rows"),
                "teacher_share_of_accepted_training_rows": teacher_share_summary.get("teacher_share_of_accepted_training_rows"),
                "teacher_share_cap": teacher_share_summary.get("teacher_share_cap"),
                "teacher_share_within_cap": teacher_share_summary.get("teacher_share_within_cap"),
                "runtime_external_inference_calls": teacher_share_summary.get("runtime_external_inference_calls"),
                "public_training_rows_written": teacher_share_summary.get("public_training_rows_written"),
                "source_report": "reports/teacher_share_ledger_summary.json",
                "score_semantics": "teacher-share accounting only; not public-transfer or learned-generation evidence",
            },
        ),
        "public_calibration_allowed": gate_record(
            public_calibration_allowed,
            "existing governor, decoder, and private/public transfer gates must all allow public calibration",
            {
                "asi_governor_allowed": bool(asi_summary.get("public_calibration_allowed")),
                "operator_locked": bool(asi_summary.get("public_calibration_operator_locked")),
                "decoder_ready_for_public_calibration": decoder_ready,
                "transfer_ready_for_public_calibration": transfer_ready,
                "transfer_contract": transfer_contract,
            },
        ),
        "model_growth_allowed": gate_record(
            model_growth_allowed,
            "model growth remains locked until maturity, transfer, coherence, and promotion gates allow it",
            {
                "asi_governor_allowed": bool(asi_summary.get("model_growth_allowed")),
                "maturity_trigger_state": maturity.get("trigger_state"),
                "maturity_model_growth_allowed": as_dict(maturity.get("summary")).get("model_growth_allowed"),
            },
        ),
        "candidate_promotion_allowed": gate_record(
            candidate_promotion_allowed,
            "candidate promotion requires the ASI governor and candidate gate to agree",
            {
                "asi_governor_allowed": bool(asi_summary.get("candidate_promotion_allowed")),
                "candidate_promote": bool(candidate.get("promote")),
                "candidate_passed": candidate.get("passed"),
                "candidate_total": candidate.get("total"),
            },
        ),
        "heavy_code_work_allowed": gate_record(
            heavy_code_work_allowed,
            "new heavy Code LM work requires resource permission and no active duplicate worker",
            {
                "resource_allows_code": resource_allows_code,
                "budget_reason": budget.get("reason"),
                "active_code_lm_process_count": len(active_code_lm),
                "resource_active_code_lm_process_count": resource_summary.get("active_code_lm_process_count"),
                "duplicate_code_lm_artifact_target_count": len(duplicate_targets),
            },
        ),
        "evidence_store_clean": gate_record(
            evidence_store_clean,
            "current control reports must be indexed and oversized reports must have snapshots",
            {
                "current_unstored_count": int(current_index.get("current_unstored_count") or 0),
                "current_truncated_without_snapshot_count": int(
                    current_index.get("current_truncated_without_snapshot_count") or 0
                ),
            },
        ),
        "closed_loop_residual_ratchet_ready": gate_record(
            bool(
                ratchet.get("policy") == "project_theseus_closed_loop_residual_ratchet_v1"
                and ratchet.get("trigger_state") in {"GREEN", "YELLOW"}
                and ratchet_decision.get("kind") in {"promote", "rollback", "retry_private", "stop_blocker"}
            ),
            "the residual ratchet must emit exactly one promote/rollback/retry_private/stop_blocker decision",
            {
                "trigger_state": ratchet.get("trigger_state"),
                "decision": ratchet_decision.get("kind"),
                "decision_reason": ratchet_decision.get("reason"),
                "same_seed_private_semantic_lift": ratchet_summary.get("same_seed_private_semantic_lift"),
                "broad_public_pass_rate": ratchet_summary.get("broad_public_pass_rate"),
            },
        ),
    }


def build_maintenance_queue(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    packets = payloads.get("attd_maintenance_packets", {})
    for index, packet in enumerate(as_list(packets.get("packets"))):
        if not isinstance(packet, dict):
            continue
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "attd_maintenance_packets",
                "id": str(packet.get("packet_id") or f"attd_packet_{index + 1}"),
                "priority": str(packet.get("priority") or "medium"),
                "component": str(packet.get("component") or "attd"),
                "bounded_action": str(packet.get("bounded_action") or ""),
                "scope": as_list(packet.get("scope"))[:12],
                "verification": as_list(packet.get("verification"))[:6],
                "runtime_tier": packet.get("runtime_tier"),
                "risk_tier": packet.get("risk_tier"),
            }
        )

    efficiency = payloads.get("system_efficiency_audit", {})
    for index, item in enumerate(as_list(efficiency.get("loop_bottlenecks"))):
        if not isinstance(item, dict):
            continue
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "system_efficiency_audit.loop_bottlenecks",
                "id": str(item.get("id") or f"runtime_bottleneck_{index + 1}"),
                "priority": str(item.get("severity") or "high"),
                "component": str(item.get("component") or item.get("phase") or item.get("lane") or "runtime_bottleneck"),
                "bounded_action": str(item.get("big_gain_action") or item.get("recommended_action") or item.get("action") or ""),
                "scope": as_list(item.get("scope"))[:12],
                "verification": as_list(item.get("verification"))[:6],
            }
        )
    for index, item in enumerate(as_list(efficiency.get("stale_control_debt"))):
        if not isinstance(item, dict):
            continue
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "system_efficiency_audit.stale_control_debt",
                "id": str(item.get("id") or f"stale_control_debt_{index + 1}"),
                "priority": str(item.get("severity") or "medium"),
                "component": str(item.get("component") or "stale_control_debt"),
                "bounded_action": str(item.get("big_gain_action") or item.get("recommended_action") or item.get("action") or ""),
                "scope": as_list(item.get("scope"))[:12],
                "verification": as_list(item.get("verification"))[:6],
            }
        )
    for index, item in enumerate(as_list(efficiency.get("architecture_cleanup_queue"))):
        if not isinstance(item, dict):
            continue
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "system_efficiency_audit",
                "id": str(item.get("id") or f"system_efficiency_cleanup_{index + 1}"),
                "priority": str(item.get("priority") or item.get("severity") or "medium"),
                "component": str(item.get("component") or item.get("lane") or "runtime_bottleneck"),
                "bounded_action": str(item.get("recommended_action") or item.get("action") or item.get("big_gain_action") or ""),
                "scope": as_list(item.get("scope"))[:12],
                "verification": as_list(item.get("verification"))[:6],
            }
        )
    project_registry = payloads.get("theseus_project_registry", {})
    project_summary = as_dict(project_registry.get("summary"))
    governance_violations = [
        row for row in as_list(project_registry.get("governance_violations")) if isinstance(row, dict)
    ]
    if governance_violations:
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_project_registry",
                "id": "project_registry_evolution_contract_violations",
                "priority": "high"
                if any(str(row.get("severity") or "") == "hard" for row in governance_violations)
                else "medium",
                "component": "project_registry",
                "bounded_action": (
                    "Resolve registry evolution contract violations before creating new lanes: improve an existing "
                    "registered surface first, or add a complete successor/deprecation relationship."
                ),
                "scope": [
                    item
                    for row in governance_violations[:8]
                    for item in as_list(row.get("scope"))[:4]
                    if item
                ],
                "verification": [
                    "Run scripts/theseus_project_registry.py",
                    "Run scripts/attd_analyzer.py",
                    "Run scripts/theseus_control_plane.py",
                ],
                "evidence": {
                    "count": len(governance_violations),
                    "violations": governance_violations[:12],
                },
            }
        )
    if int(project_summary.get("unregistered_active_source_count") or 0) > 0:
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_project_registry",
                "id": "project_registry_unregistered_active_sources",
                "priority": "high",
                "component": "project_registry",
                "bounded_action": "Assign every unregistered active source/config/doc file to an owner surface or move it under deprecated/generated lifecycle state.",
                "scope": [
                    str(row.get("path"))
                    for row in as_list(project_registry.get("unregistered"))[:16]
                    if isinstance(row, dict)
                ],
                "verification": [
                    "Run scripts/theseus_project_registry.py",
                    "Run scripts/theseus_workspace_hygiene_audit.py",
                    "Run scripts/theseus_control_plane.py",
                ],
                "evidence": project_summary,
            }
        )
    if int(project_summary.get("duplicate_family_count") or 0) > 0:
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_project_registry",
                "id": "project_registry_duplicate_families",
                "priority": "medium",
                "component": "project_registry",
                "bounded_action": "Consolidate or explicitly register duplicate vN/seed/current/after source and report families.",
                "scope": [
                    f"{row.get('root')}/{row.get('family')}"
                    for row in as_list(project_registry.get("duplicate_families"))[:16]
                    if isinstance(row, dict)
                ],
                "verification": [
                    "Run scripts/theseus_project_registry.py",
                    "Run scripts/attd_analyzer.py",
                ],
                "evidence": project_summary,
            }
        )
    for index, item in enumerate(as_list(project_registry.get("cleanup_queue"))[:24]):
        if not isinstance(item, dict):
            continue
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_project_registry.cleanup_queue",
                "id": str(item.get("queue_id") or f"registry_cleanup_{index + 1}"),
                "priority": str(item.get("priority") or "medium"),
                "component": "project_registry",
                "bounded_action": str(item.get("bounded_action") or ""),
                "scope": as_list(item.get("scope"))[:12],
                "verification": as_list(item.get("verification"))[:6],
                "evidence": item.get("evidence", {}),
            }
        )
    registry_decisions = as_dict(project_registry.get("registry_decisions"))
    for index, item in enumerate(as_list(registry_decisions.get("decisions"))[:24]):
        if not isinstance(item, dict):
            continue
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_project_registry.registry_decisions",
                "id": str(item.get("source_queue_id") or f"registry_decision_{index + 1}"),
                "priority": str(item.get("priority") or "medium"),
                "component": "project_registry_decision",
                "bounded_action": str(item.get("bounded_action") or ""),
                "scope": as_list(item.get("scope"))[:12],
                "verification": [
                    "Run scripts/theseus_project_registry.py --gate",
                    "Run scripts/attd_analyzer.py",
                    "Run scripts/theseus_control_plane.py",
                ],
                "evidence": item,
            }
        )
    stale_registry_reports = [
        row
        for row in as_list(project_registry.get("report_outputs"))
        if isinstance(row, dict) and row.get("status") in {"stale", "missing"}
    ]
    if stale_registry_reports:
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_project_registry",
                "id": "project_registry_stale_report_outputs",
                "priority": "medium",
                "component": "project_registry",
                "bounded_action": "Refresh or intentionally retire stale/missing report outputs declared by the project registry.",
                "scope": [str(row.get("path")) for row in stale_registry_reports[:16]],
                "verification": [
                    "Run the declared verification command for each owning surface.",
                    "Run scripts/report_evidence_store.py",
                    "Run scripts/theseus_control_plane.py",
                ],
                "evidence": {"count": len(stale_registry_reports), "sample": stale_registry_reports[:20]},
            }
        )
    hygiene = payloads.get("theseus_workspace_hygiene_audit", {})
    for index, item in enumerate(as_list(hygiene.get("candidates"))):
        if not isinstance(item, dict):
            continue
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_workspace_hygiene_audit",
                "id": str(item.get("id") or f"workspace_hygiene_{index + 1}"),
                "priority": str(item.get("priority") or "medium"),
                "component": str(item.get("kind") or "workspace_hygiene"),
                "bounded_action": str(item.get("action") or ""),
                "scope": [str(item.get("path") or "")] if item.get("path") else [],
                "verification": [
                    "Run scripts/theseus_workspace_hygiene_audit.py",
                    "Run scripts/theseus_control_plane.py",
                    "Preserve report evidence before deprecation or archive moves.",
                ],
                "evidence": item.get("evidence", {}),
            }
        )
    registry = payloads.get("theseus_deprecation_registry", {})
    for index, item in enumerate(as_list(registry.get("entries"))):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "")
        if status not in {"deprecated", "compatibility_wrapper", "retained"}:
            continue
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_deprecation_registry",
                "id": f"deprecation_registry_{safe_queue_id(item.get('path'))}_{index + 1}",
                "priority": (
                    "high"
                    if status == "retained"
                    and item.get("artifact_type") == "artifact"
                    and "pending_manifest_backed_archive" in str(item.get("reason") or "")
                    else "medium"
                ),
                "component": f"deprecation_{item.get('artifact_type')}",
                "bounded_action": f"Resolve registry status={status} for {item.get('path')}",
                "scope": [str(item.get("path") or "")],
                "verification": [
                    "Run scripts/theseus_deprecation_registry.py",
                    "Run scripts/theseus_control_plane.py",
                ],
                "evidence": item,
            }
        )
    retention = payloads.get("theseus_artifact_retention", {})
    retention_summary = as_dict(retention.get("summary"))
    if int(retention_summary.get("candidate_count") or 0) > int(retention_summary.get("archived_count") or 0):
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_artifact_retention",
                "id": "artifact_retention_pending_candidates",
                "priority": "high",
                "component": "artifact_retention",
                "bounded_action": "Archive remaining heavyweight historical checkpoint artifacts through the manifest-backed retention service.",
                "scope": ["reports"],
                "verification": [
                    "Run scripts/theseus_artifact_retention.py --execute",
                    "Run scripts/theseus_deprecation_registry.py",
                    "Run scripts/theseus_control_plane.py",
                ],
                "evidence": retention_summary,
            }
        )
    gc_report = payloads.get("theseus_generated_artifact_gc", {})
    gc_summary = as_dict(gc_report.get("summary"))
    if int(gc_summary.get("candidate_count") or 0) > 0:
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_generated_artifact_gc",
                "id": "generated_artifact_gc_candidates",
                "priority": "medium",
                "component": "generated_artifact_gc",
                "bounded_action": "Quarantine stale generated artifacts with the GC service when the queue is large or disk pressure rises.",
                "scope": ["tmp", ".attd_tmp", "logs"],
                "verification": [
                    "Run scripts/theseus_generated_artifact_gc.py",
                    "Optionally run scripts/theseus_generated_artifact_gc.py --execute",
                ],
                "evidence": gc_summary,
            }
        )
    dirty_review = payloads.get("theseus_dirty_workspace_review", {})
    dirty_summary = as_dict(dirty_review.get("summary"))
    if int(dirty_summary.get("dirty_count") or 0) > 0:
        queue.append(
            {
                "record_type": "maintenance_queue_item",
                "source": "theseus_dirty_workspace_review",
                "id": "dirty_workspace_review_classified",
                "priority": "high",
                "component": "dirty_workspace",
                "bounded_action": "Resolve classified dirty workspace entries into source changes, generated artifacts, or ignored archive payloads.",
                "scope": [str(row.get("path")) for row in as_list(dirty_review.get("rows"))[:16] if isinstance(row, dict)],
                "verification": [
                    "Run scripts/theseus_dirty_workspace_review.py",
                    "Run git status --short",
                    "Run scripts/theseus_control_plane.py",
                ],
                "evidence": dirty_summary,
            }
        )
    return sorted(queue, key=maintenance_sort_key)


def build_blockers(
    payloads: dict[str, dict[str, Any]],
    stale_reports: list[dict[str, Any]],
    duplicate_targets: list[dict[str, Any]],
    current_index: dict[str, Any],
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    live_code_lm_count = int(gates.get("heavy_code_work_allowed", {}).get("evidence", {}).get("active_code_lm_process_count") or 0)
    resource_code_lm_count = int(
        gates.get("heavy_code_work_allowed", {}).get("evidence", {}).get("resource_active_code_lm_process_count") or 0
    )
    if resource_code_lm_count > live_code_lm_count:
        blockers.append(
            {
                "record_type": "blocker",
                "source": "theseus_control_plane",
                "id": "resource_policy_live_worker_mismatch",
                "severity": "medium",
                "status": "stale_control_state",
                "title": "Resource policy reports active Code LM workers that live process guard does not see.",
                "next_action": "refresh scripts/resource_aware_execution_policy.py before deciding heavy Code LM availability",
                "evidence": {
                    "resource_active_code_lm_process_count": resource_code_lm_count,
                    "live_active_code_lm_process_count": live_code_lm_count,
                },
            }
        )
    asi = payloads.get("asi_wall_breaker_governor", {})
    for wall in as_list(asi.get("walls")):
        if not isinstance(wall, dict) or wall.get("status") == "cleared":
            continue
        blockers.append(
            {
                "record_type": "blocker",
                "source": "asi_wall_breaker_governor",
                "id": str(wall.get("id") or "asi_wall"),
                "severity": str(wall.get("severity") or "medium"),
                "status": str(wall.get("status") or "blocked"),
                "title": str(wall.get("title") or ""),
                "next_action": str(wall.get("next_action") or ""),
            }
        )
    for row in stale_reports:
        blockers.append(
            {
                "record_type": "blocker",
                "source": "theseus_control_plane",
                "id": f"stale_or_missing_{row['id']}",
                "severity": "medium" if row.get("exists") else "high",
                "status": "blocked" if row.get("missing") else "stale",
                "title": f"{row['id']} is {'missing' if row.get('missing') else 'stale'}",
                "next_action": f"refresh {row.get('path')}",
            }
        )
    if duplicate_targets:
        blockers.append(
            {
                "record_type": "blocker",
                "source": "code_lm_process_guard",
                "id": "duplicate_code_lm_artifact_targets",
                "severity": "high",
                "status": "blocked",
                "title": "Duplicate Code LM workers target the same artifacts.",
                "next_action": "do not launch another heavy worker; let one finish or explicitly stop duplicates",
                "evidence": duplicate_targets[:5],
            }
        )
    if int(current_index.get("current_unstored_count") or 0) > 0:
        blockers.append(
            {
                "record_type": "blocker",
                "source": "report_evidence_store",
                "id": "current_reports_unstored",
                "severity": "high",
                "status": "blocked",
                "title": "Some current control reports are not in the evidence store.",
                "next_action": "run report evidence ingestion before trusting latest views",
            }
        )
    if int(current_index.get("current_truncated_without_snapshot_count") or 0) > 0:
        blockers.append(
            {
                "record_type": "blocker",
                "source": "report_evidence_store",
                "id": "current_truncated_reports_without_snapshots",
                "severity": "high",
                "status": "blocked",
                "title": "Some oversized current reports are indexed without snapshots.",
                "next_action": "snapshot oversized reports before overwrites can lose evidence",
            }
        )
    if not gates.get("registry_governance_ready", {}).get("passed"):
        blockers.append(
            {
                "record_type": "blocker",
                "source": "theseus_project_registry",
                "id": "registry_governance_not_ready",
                "severity": "hard",
                "status": "blocked",
                "title": "Registry abstraction/implementation governance is not ready for routing or self-improvement decisions.",
                "next_action": "run python3 scripts/theseus_project_registry.py --gate and resolve the reported contract/routing gaps",
                "evidence": gates.get("registry_governance_ready", {}).get("evidence", {}),
            }
        )
    ratchet = payloads.get("closed_loop_residual_ratchet", {})
    ratchet_decision = as_dict(ratchet.get("decision"))
    ratchet_kind = str(ratchet_decision.get("kind") or "")
    if ratchet_kind in {"rollback", "retry_private", "stop_blocker"}:
        blockers.append(
            {
                "record_type": "blocker",
                "source": "closed_loop_residual_ratchet",
                "id": f"residual_ratchet_{ratchet_kind}",
                "severity": "hard" if ratchet_kind == "rollback" else "medium",
                "status": ratchet_kind,
                "title": str(ratchet_decision.get("reason") or "Closed-loop residual ratchet requires action."),
                "next_action": " ".join(map(str, ratchet_decision.get("command") or [])),
                "evidence": {
                    "decision": ratchet_kind,
                    "trigger_state": ratchet.get("trigger_state"),
                    "summary": as_dict(ratchet.get("summary")),
                },
            }
        )
    for gate_name in ["public_calibration_allowed", "model_growth_allowed", "candidate_promotion_allowed"]:
        if not gates.get(gate_name, {}).get("passed"):
            blockers.append(
                {
                    "record_type": "blocker",
                    "source": "theseus_control_plane",
                    "id": f"{gate_name}_locked",
                    "severity": "policy",
                    "status": "locked",
                    "title": f"{gate_name} is locked by current gates.",
                    "next_action": gates.get(gate_name, {}).get("rule", ""),
                    "evidence": gates.get(gate_name, {}).get("evidence", {}),
                }
            )
    return blockers


def build_typed_records(
    report_records: list[dict[str, Any]],
    gates: dict[str, Any],
    maintenance_queue: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    active_code_lm: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(strip_payload(row) for row in report_records)
    for gate_name, gate in gates.items():
        records.append({"record_type": "gate_state", "id": gate_name, **gate})
    records.extend(maintenance_queue)
    records.extend(blockers)
    for row in compact_process_rows(active_code_lm):
        records.append({"record_type": "active_worker", "worker_type": "code_lm", **row})
    return records


def build_action_decisions(state: dict[str, Any]) -> list[dict[str, Any]]:
    gates = state["gates"]
    decisions = [
        action_decision(
            "registry_governance",
            "registry_gate",
            "ready" if gates["registry_governance_ready"]["passed"] else "blocked",
            gates["registry_governance_ready"]["rule"],
            gates["registry_governance_ready"]["evidence"],
        ),
        action_decision(
            "public_calibration",
            "public_calibration",
            "allowed" if gates["public_calibration_allowed"]["passed"] else "blocked",
            gates["public_calibration_allowed"]["rule"],
            gates["public_calibration_allowed"]["evidence"],
        ),
        action_decision(
            "model_growth",
            "model_growth",
            "allowed" if gates["model_growth_allowed"]["passed"] else "blocked",
            gates["model_growth_allowed"]["rule"],
            gates["model_growth_allowed"]["evidence"],
        ),
        action_decision(
            "candidate_promotion",
            "candidate_promotion",
            "allowed" if gates["candidate_promotion_allowed"]["passed"] else "blocked",
            gates["candidate_promotion_allowed"]["rule"],
            gates["candidate_promotion_allowed"]["evidence"],
        ),
        action_decision(
            "heavy_code_worker",
            "heavy_code_worker",
            "available" if gates["heavy_code_work_allowed"]["passed"] else "blocked",
            gates["heavy_code_work_allowed"]["rule"],
            gates["heavy_code_work_allowed"]["evidence"],
        ),
        action_decision(
            "closed_loop_residual_ratchet",
            "residual_ratchet",
            str(gates["closed_loop_residual_ratchet_ready"]["evidence"].get("decision") or "missing"),
            gates["closed_loop_residual_ratchet_ready"]["rule"],
            gates["closed_loop_residual_ratchet_ready"]["evidence"],
        ),
    ]
    return decisions


def request_action(
    conn: sqlite3.Connection,
    *,
    action_key: str,
    action_type: str,
    command: str,
    gates: dict[str, Any],
    active_workers: list[dict[str, Any]],
    lease_seconds: int,
) -> dict[str, Any]:
    now_dt = utcnow_dt()
    active_lease = active_lease_for(conn, action_key, now_dt)
    status = "reserved"
    reason = "action_lease_reserved"
    evidence: dict[str, Any] = {}
    if active_lease:
        status = "blocked"
        reason = "active_action_lease_exists"
        evidence = {
            "existing_ledger_id": active_lease.get("ledger_id"),
            "existing_status": active_lease.get("status"),
            "expires_utc": active_lease.get("expires_utc"),
        }
    elif action_type in HEAVY_ACTION_TYPES and active_workers:
        status = "blocked"
        reason = "active_heavy_code_worker_present"
        evidence = {"active_workers": active_workers[:6]}
    elif action_type in HEAVY_ACTION_TYPES and not gates.get("heavy_code_work_allowed", {}).get("passed"):
        status = "blocked"
        reason = "resource_policy_or_duplicate_guard_blocks_heavy_code_work"
        evidence = gates.get("heavy_code_work_allowed", {}).get("evidence", {})
    elif action_type == "public_calibration" and not gates.get("public_calibration_allowed", {}).get("passed"):
        status = "blocked"
        reason = "public_calibration_locked"
        evidence = gates.get("public_calibration_allowed", {}).get("evidence", {})
    elif action_type == "model_growth" and not gates.get("model_growth_allowed", {}).get("passed"):
        status = "blocked"
        reason = "model_growth_locked"
        evidence = gates.get("model_growth_allowed", {}).get("evidence", {})
    elif action_type == "candidate_promotion" and not gates.get("candidate_promotion_allowed", {}).get("passed"):
        status = "blocked"
        reason = "candidate_promotion_locked"
        evidence = gates.get("candidate_promotion_allowed", {}).get("evidence", {})

    expires = now_dt + timedelta(seconds=lease_seconds) if status in ACTIVE_LEDGER_STATUSES else None
    row = append_action_ledger(
        conn,
        action_key=action_key,
        action_type=action_type,
        command=command,
        status=status,
        reason=reason,
        lease_seconds=lease_seconds if expires else 0,
        expires_utc=format_dt(expires) if expires else "",
        payload={"evidence": evidence},
    )
    return row


def ensure_action_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS control_action_ledger (
            ledger_id TEXT PRIMARY KEY,
            created_utc TEXT NOT NULL,
            action_key TEXT NOT NULL,
            action_type TEXT NOT NULL,
            command_hash TEXT NOT NULL,
            command TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            lease_seconds INTEGER NOT NULL DEFAULT 0,
            expires_utc TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_control_action_key ON control_action_ledger(action_key, status, expires_utc);
        CREATE INDEX IF NOT EXISTS idx_control_action_created ON control_action_ledger(created_utc);
        """
    )


def append_control_run(
    conn: sqlite3.Connection,
    state: dict[str, Any],
    action_decisions: list[dict[str, Any]],
    requested_action: dict[str, Any],
) -> None:
    append_action_ledger(
        conn,
        action_key="theseus_control_plane_observation",
        action_type="control_plane_run",
        command="scripts/theseus_control_plane.py",
        status="observed",
        reason="control_plane_snapshot_refreshed",
        lease_seconds=0,
        expires_utc="",
        payload={
            "blocker_count": len(state.get("blockers", [])),
            "stale_report_count": len(state.get("stale_reports", [])),
            "action_decisions": action_decisions,
            "requested_action": requested_action,
        },
    )


def append_action_ledger(
    conn: sqlite3.Connection,
    *,
    action_key: str,
    action_type: str,
    command: str,
    status: str,
    reason: str,
    lease_seconds: int,
    expires_utc: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    created = now()
    command_hash = stable_id("command", command)
    payload_json = json.dumps(payload, sort_keys=True)
    ledger_id = stable_id("control_action", created, action_key, action_type, command_hash, status, reason, payload_json)
    conn.execute(
        """
        INSERT INTO control_action_ledger (
            ledger_id, created_utc, action_key, action_type, command_hash, command,
            status, reason, lease_seconds, expires_utc, payload_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ledger_id,
            created,
            action_key,
            action_type,
            command_hash,
            command,
            status,
            reason,
            int(lease_seconds),
            expires_utc,
            payload_json,
        ),
    )
    return {
        "record_type": "action_ledger_row",
        "ledger_id": ledger_id,
        "created_utc": created,
        "action_key": action_key,
        "action_type": action_type,
        "command_hash": command_hash,
        "status": status,
        "reason": reason,
        "lease_seconds": int(lease_seconds),
        "expires_utc": expires_utc,
        "payload": payload,
    }


def active_lease_for(conn: sqlite3.Connection, action_key: str, now_dt: datetime) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT * FROM control_action_ledger
        WHERE action_key=? AND status IN ('reserved', 'running') AND expires_utc != ''
        ORDER BY created_utc DESC
        """,
        (action_key,),
    ).fetchall()
    for row in rows:
        expires = parse_dt(str(row["expires_utc"] or ""))
        if expires and expires > now_dt:
            return dict(row)
    return {}


def action_ledger_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT ledger_id, created_utc, action_key, action_type, status, reason, expires_utc
        FROM control_action_ledger
        ORDER BY created_utc DESC
        LIMIT 24
        """
    ).fetchall()
    active = []
    now_dt = utcnow_dt()
    for row in rows:
        expires = parse_dt(str(row["expires_utc"] or ""))
        if str(row["status"]) in ACTIVE_LEDGER_STATUSES and expires and expires > now_dt:
            active.append(dict(row))
    total = int(conn.execute("SELECT COUNT(*) AS n FROM control_action_ledger").fetchone()["n"])
    return {
        "ledger_table": "control_action_ledger",
        "total_rows": total,
        "active_lease_count": len(active),
        "active_leases": active,
        "recent_rows": [dict(row) for row in rows],
    }


def build_report_payload(
    state: dict[str, Any],
    action_decisions: list[dict[str, Any]],
    requested_action: dict[str, Any],
    ledger_summary: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    hard_blockers = [row for row in state["blockers"] if row.get("severity") in {"hard", "high"}]
    policy_locks = [row for row in state["blockers"] if row.get("severity") == "policy"]
    stale_count = len(state["stale_reports"])
    evidence_clean = bool(state["gates"]["evidence_store_clean"]["passed"])
    trigger_state = "RED" if hard_blockers or not evidence_clean else ("YELLOW" if stale_count or policy_locks else "GREEN")
    next_work = choose_next_work(state)
    payload = {
        "policy": CONTROL_PLANE_POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "report_count": len(state["reports"]),
            "stale_report_count": stale_count,
            "missing_report_count": sum(1 for row in state["reports"] if row.get("missing")),
            "blocker_count": len(state["blockers"]),
            "hard_blocker_count": len(hard_blockers),
            "maintenance_queue_count": len(state["maintenance_queue"]),
            "active_code_lm_process_count": state["active_workers"]["active_code_lm_process_count"],
            "duplicate_code_lm_artifact_target_count": len(state["active_workers"]["duplicate_code_lm_artifact_targets"]),
            "registry_governance_ready": state["gates"]["registry_governance_ready"]["passed"],
            "registry_routing_eligible_implementation_count": state["gates"]["registry_governance_ready"]["evidence"].get("routing_eligible_implementation_count"),
            "registry_cleanup_queue_count": state["gates"]["registry_governance_ready"]["evidence"].get("registry_cleanup_queue_count"),
            "viea_spine_materialized_view_ready": state["gates"]["viea_spine_materialized_view_ready"]["passed"],
            "viea_spine_materialized_record_count": state["gates"]["viea_spine_materialized_view_ready"]["evidence"].get("record_count"),
            "viea_spine_claim_ledger_entry_count": state["gates"]["viea_spine_materialized_view_ready"]["evidence"].get("claim_ledger_entry_count"),
            "teacher_share_ledger_ready": state["gates"]["teacher_share_ledger_ready"]["passed"],
            "teacher_share_of_accepted_training_rows": state["gates"]["teacher_share_ledger_ready"]["evidence"].get("teacher_share_of_accepted_training_rows"),
            "teacher_accepted_rows": state["gates"]["teacher_share_ledger_ready"]["evidence"].get("teacher_accepted_rows"),
            "verified_self_generated_rows": state["gates"]["teacher_share_ledger_ready"]["evidence"].get("verified_self_generated_rows"),
            "teacher_share_within_cap": state["gates"]["teacher_share_ledger_ready"]["evidence"].get("teacher_share_within_cap"),
            "public_calibration_allowed": state["gates"]["public_calibration_allowed"]["passed"],
            "model_growth_allowed": state["gates"]["model_growth_allowed"]["passed"],
            "candidate_promotion_allowed": state["gates"]["candidate_promotion_allowed"]["passed"],
            "heavy_code_work_allowed": state["gates"]["heavy_code_work_allowed"]["passed"],
            "closed_loop_residual_ratchet_decision": state["gates"]["closed_loop_residual_ratchet_ready"]["evidence"].get("decision"),
            "evidence_store_current_unstored_count": state["evidence_store"]["current_index"]["current_unstored_count"],
            "evidence_store_current_truncated_without_snapshot_count": state["evidence_store"]["current_index"][
                "current_truncated_without_snapshot_count"
            ],
            "action_ledger_rows": ledger_summary["total_rows"],
            "active_action_lease_count": ledger_summary["active_lease_count"],
            "next_recommended_work": next_work.get("label", ""),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "control_plane_records": state["typed_records"],
        "reports": state["reports"],
        "stale_reports": state["stale_reports"],
        "gates": state["gates"],
        "blockers": state["blockers"],
        "maintenance_queue": state["maintenance_queue"],
        "active_workers": state["active_workers"],
        "action_decisions": action_decisions,
        "requested_action": requested_action,
        "action_ledger": ledger_summary,
        "evidence_store": state["evidence_store"],
        "next_recommended_work": next_work,
        "rules": {
            "reports_are_latest_views": "Stable report JSON files remain human-readable latest views only.",
            "durable_evidence": "Current control reports are ingested into report_evidence_store.sqlite before synthesis.",
            "large_report_snapshots": "Oversized current reports must have immutable snapshots before control state is trusted.",
            "idempotent_heavy_work": "Heavy Code LM actions must reserve a control_action_ledger lease and pass process/duplicate guards.",
            "public_calibration": "Public calibration stays locked unless existing gates explicitly allow it.",
            "model_growth": "Model growth stays locked unless existing gates explicitly allow it.",
            "teacher_share": "Teacher-share accounting must come from the durable ledger summary; training-time teacher calls are distinct from forbidden runtime external serving.",
            "registry_governance": "Self-improvement and routing decisions must pass the registry abstraction/implementation gate.",
            "viea_spine_materialized_view": "Control decisions consume the shared claim/evidence/semantic/governance/failure record view.",
        },
        "external_inference_calls": 0,
    }
    return payload


def choose_next_work(state: dict[str, Any]) -> dict[str, Any]:
    if state["stale_reports"]:
        row = state["stale_reports"][0]
        return {
            "label": "refresh_stale_control_report",
            "reason": f"{row['id']} is stale or missing",
            "command": ["python", "scripts/theseus_control_plane.py"],
            "target": row,
        }
    if not state["gates"]["evidence_store_clean"]["passed"]:
        return {
            "label": "repair_report_evidence_ingestion",
            "reason": "current reports are not fully indexed or snapshotted",
            "command": ["python", "scripts/report_evidence_store.py"],
        }
    for blocker in state["blockers"]:
        if blocker.get("id") == "resource_policy_live_worker_mismatch":
            return {
                "label": "refresh_resource_policy_live_state",
                "reason": blocker.get("title", ""),
                "command": [
                    "python",
                    "scripts/resource_aware_execution_policy.py",
                    "--out",
                    "reports/resource_aware_execution_policy.json",
                    "--markdown-out",
                    "reports/resource_aware_execution_policy.md",
                ],
                "target": blocker,
            }
    ratchet_payload = state["report_payloads"].get("closed_loop_residual_ratchet", {})
    ratchet_decision = as_dict(ratchet_payload.get("decision"))
    ratchet_kind = str(ratchet_decision.get("kind") or "")
    if ratchet_kind in {"rollback", "retry_private", "stop_blocker"}:
        fallback = locked_ratchet_fallback_work(state, ratchet_kind, ratchet_decision)
        if fallback:
            return fallback
        return {
            "label": f"closed_loop_residual_ratchet_{ratchet_kind}",
            "reason": str(ratchet_decision.get("reason") or ""),
            "command": ratchet_decision.get("command") or [],
            "target": {
                "decision": ratchet_kind,
                "report": "reports/closed_loop_residual_ratchet.json",
                "evidence": ratchet_decision.get("evidence", {}),
            },
        }
    if state["maintenance_queue"]:
        item = state["maintenance_queue"][0]
        return {
            "label": "consume_highest_priority_attd_or_runtime_cleanup_packet",
            "reason": item.get("bounded_action", ""),
            "target": item,
        }
    for blocker in state["blockers"]:
        if blocker.get("severity") in {"hard", "high"}:
            return {
                "label": "clear_highest_severity_control_blocker",
                "reason": blocker.get("title", ""),
                "target": blocker,
            }
    return {
        "label": "continue_private_transfer_architecture_cleanup",
        "reason": "no high-severity control-plane hygiene blocker is currently first in line",
    }


def locked_ratchet_fallback_work(
    state: dict[str, Any],
    ratchet_kind: str,
    ratchet_decision: dict[str, Any],
) -> dict[str, Any]:
    if ratchet_kind != "stop_blocker":
        return {}
    public_gate = as_dict(state["gates"].get("public_calibration_allowed"))
    public_evidence = as_dict(public_gate.get("evidence"))
    operator_locked = bool(public_evidence.get("operator_locked"))
    decoder_ready = bool(public_evidence.get("decoder_ready_for_public_calibration"))
    transfer_ready = bool(public_evidence.get("transfer_ready_for_public_calibration"))
    if not (operator_locked and decoder_ready and transfer_ready):
        return {}

    fallback = governor_non_public_fallback(state)
    if not fallback:
        fallback = coherence_repair_fallback(state)
    if not fallback:
        fallback = private_transfer_fallback(state)
    if not fallback:
        return {}

    fallback["locked_boundary"] = {
        "blocked_label": f"closed_loop_residual_ratchet_{ratchet_kind}",
        "blocked_reason": str(ratchet_decision.get("reason") or ""),
        "blocked_command": ratchet_decision.get("command") or [],
        "operator_locked": operator_locked,
        "decoder_ready_for_public_calibration": decoder_ready,
        "transfer_ready_for_public_calibration": transfer_ready,
        "policy": "public calibration remains locked; route autonomous work to the best non-public capability lane",
    }
    fallback["routing_delta"] = {
        "before": f"closed_loop_residual_ratchet_{ratchet_kind}",
        "after": fallback.get("label", ""),
        "improvement": "locked_public_boundary_replaced_by_actionable_private_work",
    }
    return fallback


def governor_non_public_fallback(state: dict[str, Any]) -> dict[str, Any]:
    governor = state["report_payloads"].get("asi_wall_breaker_governor", {})
    wall_priority = [
        "promotion_and_coherence_blocked",
        "learner_substrate_too_small",
        "private_to_public_transfer_gap",
        "autonomy_noisy",
        "non_code_lanes_not_frontier_grade",
        "codebase_complexity_drag",
    ]
    for wanted in wall_priority:
        wall = find_governor_wall(governor, wanted)
        if not wall:
            continue
        if str(wall.get("status") or "").lower() == "cleared":
            continue
        next_action = str(wall.get("next_action") or wall.get("title") or "")
        fallback_work = fallback_work_for_wall(wanted, wall, next_action)
        return {
            "label": fallback_work["label"],
            "reason": fallback_work["reason"],
            "command": fallback_work["command"],
            "target": {
                "source": "asi_wall_breaker_governor",
                "wall_id": wanted,
                "status": wall.get("status"),
                "severity": wall.get("severity"),
                "title": wall.get("title"),
                "evidence": wall.get("evidence", {}),
            },
        }
    return {}


def fallback_work_for_wall(wall_id: str, wall: dict[str, Any], next_action: str) -> dict[str, Any]:
    evidence = as_dict(wall.get("evidence"))
    failed_gates = {str(item) for item in as_list(evidence.get("candidate_failed_gates"))}
    if wall_id == "promotion_and_coherence_blocked":
        coherence_green = str(evidence.get("coherence_trigger_state") or "") == "GREEN"
        transfer_or_maturity_blocked = bool(
            failed_gates
            & {
                "broad_public_code_transfer_ready",
                "maturity_integrity_audit_green",
                "active_frontier_training_budget_sufficient",
                "candidate_profile_evidence_complete",
                "code_frontier_transfer_consumed",
            }
        )
        if coherence_green and transfer_or_maturity_blocked:
            return {
                "label": "broad_public_transfer_floor_private_repair",
                "reason": "coherence is GREEN; continue with private-only Code LM transfer repair for the remaining broad-transfer/maturity blockers",
                "command": private_code_lm_transfer_command(),
            }
    return {
        "label": f"{wall_id}_private_repair",
        "reason": next_action,
        "command": fallback_command_for_wall(wall_id),
    }


def find_governor_wall(governor: dict[str, Any], wall_id: str) -> dict[str, Any]:
    for key in ("walls", "blockers", "wall_records", "control_plane_records"):
        for row in as_list(governor.get(key)):
            if isinstance(row, dict) and str(row.get("id") or row.get("wall_id") or "") == wall_id:
                return row
    return {}


def fallback_command_for_wall(wall_id: str) -> list[str]:
    if wall_id == "promotion_and_coherence_blocked":
        return [
            "python",
            "scripts/coherence_delirium_gate.py",
            "--out",
            "reports/coherence_delirium_gate.json",
        ]
    if wall_id == "learner_substrate_too_small":
        return private_code_lm_transfer_command()
    if wall_id == "private_to_public_transfer_gap":
        return ["python", "scripts/private_public_transfer_proof.py"]
    if wall_id == "autonomy_noisy":
        return ["python", "scripts/autonomy_watchdog.py"]
    if wall_id == "non_code_lanes_not_frontier_grade":
        return ["python", "scripts/agent_lane_transfer_gate.py"]
    if wall_id == "codebase_complexity_drag":
        return ["python", "scripts/system_efficiency_audit.py"]
    return ["python", "scripts/asi_wall_breaker_governor.py"]


def private_code_lm_transfer_command() -> list[str]:
    return [
        "python",
        "scripts/code_lm_train_once_fanout.py",
        "--execute",
        "--private-only",
        "--slug",
        "open_code_transfer_expansion_v1",
    ]


def coherence_repair_fallback(state: dict[str, Any]) -> dict[str, Any]:
    for blocker in state["blockers"]:
        if blocker.get("id") == "candidate_promotion_allowed_locked":
            return {
                "label": "coherence_candidate_promotion_private_repair",
                "reason": "candidate promotion is locked; improve private coherence evidence before any promotion claim",
                "command": [
                    "python",
                    "scripts/coherence_delirium_gate.py",
                    "--out",
                    "reports/coherence_delirium_gate.json",
                ],
                "target": blocker,
            }
    return {}


def private_transfer_fallback(state: dict[str, Any]) -> dict[str, Any]:
    if state["gates"]["heavy_code_work_allowed"]["passed"]:
        return {
            "label": "private_code_lm_transfer_worker",
            "reason": "public calibration is locked, but resource policy allows private CUDA Code LM work",
            "command": private_code_lm_transfer_command(),
            "target": {
                "source": "resource_aware_execution_policy",
                "evidence": state["gates"]["heavy_code_work_allowed"]["evidence"],
            },
        }
    return {}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = as_dict(payload.get("summary"))
    lines = [
        "# Theseus Control Plane v1",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- reports tracked: `{summary.get('report_count')}`",
        f"- stale reports: `{summary.get('stale_report_count')}`",
        f"- blockers: `{summary.get('blocker_count')}` hard `{summary.get('hard_blocker_count')}`",
        f"- active Code LM workers: `{summary.get('active_code_lm_process_count')}`",
        f"- duplicate artifact targets: `{summary.get('duplicate_code_lm_artifact_target_count')}`",
        f"- registry governance ready: `{summary.get('registry_governance_ready')}`",
        f"- registry routing-eligible implementations: `{summary.get('registry_routing_eligible_implementation_count')}`",
        f"- registry cleanup queue: `{summary.get('registry_cleanup_queue_count')}`",
        f"- VIEA materialized view ready: `{summary.get('viea_spine_materialized_view_ready')}`",
        f"- VIEA materialized records: `{summary.get('viea_spine_materialized_record_count')}` claim/proof `{summary.get('viea_spine_claim_ledger_entry_count')}`",
        f"- teacher share ledger ready: `{summary.get('teacher_share_ledger_ready')}` share `{summary.get('teacher_share_of_accepted_training_rows')}`",
        f"- teacher/self rows: teacher `{summary.get('teacher_accepted_rows')}` verified self `{summary.get('verified_self_generated_rows')}` within cap `{summary.get('teacher_share_within_cap')}`",
        f"- maintenance queue: `{summary.get('maintenance_queue_count')}`",
        f"- evidence current unstored: `{summary.get('evidence_store_current_unstored_count')}`",
        f"- evidence current truncated without snapshot: `{summary.get('evidence_store_current_truncated_without_snapshot_count')}`",
        f"- action ledger rows: `{summary.get('action_ledger_rows')}` active leases `{summary.get('active_action_lease_count')}`",
        f"- next: `{summary.get('next_recommended_work')}`",
        "",
        "## Gates",
        "",
    ]
    for name, gate in as_dict(payload.get("gates")).items():
        lines.append(f"- `{name}` passed=`{gate.get('passed')}` rule={gate.get('rule')}")
    lines.extend(["", "## Action Decisions", ""])
    for row in as_list(payload.get("action_decisions")):
        lines.append(f"- `{row.get('action_key')}` status=`{row.get('status')}` reason={row.get('reason')}")
    if payload.get("requested_action"):
        row = as_dict(payload.get("requested_action"))
        lines.extend(["", "## Requested Action", ""])
        lines.append(f"- `{row.get('action_key')}` status=`{row.get('status')}` reason={row.get('reason')}")
    lines.extend(["", "## Blockers", ""])
    for row in as_list(payload.get("blockers"))[:12]:
        lines.append(f"- `{row.get('id')}` severity=`{row.get('severity')}` status=`{row.get('status')}` {row.get('title')}")
    lines.extend(["", "## Maintenance Queue", ""])
    for row in as_list(payload.get("maintenance_queue"))[:10]:
        lines.append(f"- `{row.get('id')}` priority=`{row.get('priority')}` component=`{row.get('component')}`")
    return "\n".join(lines) + "\n"


def action_decision(action_key: str, action_type: str, status: str, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "action_decision",
        "action_key": action_key,
        "action_type": action_type,
        "status": status,
        "reason": reason,
        "evidence": evidence,
    }


def gate_record(passed: bool, rule: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"passed": bool(passed), "rule": rule, "evidence": evidence}


def maintenance_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    priority = str(row.get("priority") or "").lower()
    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(priority, 2)
    return rank, str(row.get("id") or "")


def compact_report_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "path": row.get("path"),
        "role": row.get("role"),
        "exists": row.get("exists"),
        "missing": row.get("missing"),
        "stale": row.get("stale"),
        "created_utc": row.get("created_utc"),
        "age_hours": row.get("age_hours"),
        "max_age_hours": row.get("max_age_hours"),
        "trigger_state": row.get("trigger_state"),
    }


def strip_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "payload"}


def compact_process_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "pid": row.get("pid"),
                "name": row.get("name"),
                "command_preview": str(row.get("command_preview") or row.get("command") or "")[:360],
            }
        )
    return compact


def compact_current_index(index: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_report_count": int(index.get("current_report_count") or 0),
        "current_unstored_count": int(index.get("current_unstored_count") or 0),
        "current_truncated_without_snapshot_count": int(index.get("current_truncated_without_snapshot_count") or 0),
        "unstored_samples": as_list(index.get("unstored_samples"))[:8],
        "truncated_without_snapshot_samples": as_list(index.get("truncated_without_snapshot_samples"))[:8],
        "largest_current_reports": as_list(index.get("latest_samples"))[:8],
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "trigger_state",
        "profile",
        "gpu_name",
        "gpu_utilization_percent",
        "active_code_lm_process_count",
        "duplicate_code_lm_artifact_target_count",
        "hard_blocker_count",
        "soft_blocker_count",
        "public_calibration_allowed",
        "public_calibration_operator_locked",
        "model_growth_allowed",
        "candidate_promotion_allowed",
        "ready_for_public_calibration",
        "current_unstored_count",
        "current_truncated_without_snapshot_count",
        "hard_maintainability_hotspot_count",
        "attd_packet_count",
        "attd_runtime_overlap_count",
        "packet_count",
        "runtime_ms",
    ]
    out = {key: summary[key] for key in keys if key in summary}
    for key, value in summary.items():
        if key in out:
            continue
        if len(out) >= 20:
            break
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def report_created_utc(payload: dict[str, Any], path: Path | None) -> datetime | None:
    created = payload.get("created_utc") if isinstance(payload, dict) else None
    if not created and isinstance(payload.get("summary") if isinstance(payload, dict) else None, dict):
        created = payload["summary"].get("created_utc")
    parsed = parse_dt(str(created or ""))
    if parsed:
        return parsed
    if path and path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return None


def parse_dt(value: str) -> datetime | None:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def age_hours_since(created: datetime | None) -> float | None:
    if created is None:
        return None
    return max(0.0, (utcnow_dt() - created).total_seconds() / 3600.0)


def format_dt(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def utcnow_dt() -> datetime:
    return datetime.now(timezone.utc)


def now() -> str:
    return format_dt(utcnow_dt())


def read_json(path: Path, default: Any) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return payload if isinstance(payload, dict) else default


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def stable_id(*parts: Any) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]


def safe_queue_id(value: Any) -> str:
    text = str(value or "unknown")
    chars = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text]
    return "".join(chars).strip("._")[:80] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
