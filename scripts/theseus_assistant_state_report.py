#!/usr/bin/env python3
"""Concise state report for the canonical Theseus assistant surface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "theseus_assistant_state_report.json"
DEFAULT_MD = REPORTS / "theseus_assistant_state_report.md"
DEFAULT_PUBLIC_RUN_REGISTRY = REPORTS / "public_benchmark_run_registry.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant-e2e", default="reports/theseus_assistant_e2e.json")
    parser.add_argument("--assistant-runtime", default="reports/theseus_assistant_runtime.json")
    parser.add_argument("--code-probe", default="reports/assistant_code_private_replay_probe.json")
    parser.add_argument("--dogfood-bridge", default="reports/theseus_assistant_dogfood_training_bridge.json")
    parser.add_argument("--public-planner", default="reports/public_transfer_next_surface_planner.json")
    parser.add_argument("--public-packet", default="reports/public_transfer_readiness_packet_public_transfer_lift_seed5_5x64.json")
    parser.add_argument("--benchmark-measurement", default="reports/theseus_benchmark_measurement.json")
    parser.add_argument("--public-run-registry", default=rel(DEFAULT_PUBLIC_RUN_REGISTRY))
    parser.add_argument("--benchmark-operator-dry-run", default="reports/operator_bounded_public_calibration_dry_run.json")
    parser.add_argument(
        "--benchmark-operator-execute",
        default="reports/theseus_benchmark_public_operator_execute.json",
    )
    parser.add_argument(
        "--public-residual-report",
        default="",
    )
    parser.add_argument(
        "--private-residual-consumer",
        default="",
    )
    parser.add_argument(
        "--fresh-residual-private-probe",
        default="",
    )
    parser.add_argument("--candidate-lookup-audit", default="reports/student_candidate_lookup_contract_audit.json")
    parser.add_argument("--registry", default="reports/theseus_project_registry_assistant_runtime_current.json")
    parser.add_argument("--hive-operator-assistant", default="reports/hive_operator_assistant_latest.json")
    parser.add_argument("--hive-operator-feedback", default="reports/hive_operator_feedback_latest.json")
    parser.add_argument("--hive-operator-status", default="reports/hive_operator_status.json")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    e2e = read_json(resolve(args.assistant_e2e), {})
    runtime = read_json(resolve(args.assistant_runtime), {})
    code_probe = read_json(resolve(args.code_probe), {})
    dogfood = read_json(resolve(args.dogfood_bridge), {})
    public_planner = read_json(resolve(args.public_planner), {})
    public_packet = read_json(resolve(args.public_packet), {})
    benchmark_measurement = read_json(resolve(args.benchmark_measurement), {})
    benchmark_operator_dry_run = read_json(resolve(args.benchmark_operator_dry_run), {})
    benchmark_operator_execute = read_json(resolve(args.benchmark_operator_execute), {})
    public_registry_rows = read_jsonl(resolve(args.public_run_registry))
    latest_registry_row = latest_scored_registry_row(public_registry_rows)
    public_residual_path = resolve_report_path(
        args.public_residual_report,
        latest_registry_row.get("residual_report_path"),
    )
    private_residual_path = resolve_report_path(
        args.private_residual_consumer,
        inferred_private_residual_consumer(public_residual_path, latest_registry_row),
    )
    public_residual = read_json(public_residual_path, {})
    private_residual_consumer = read_json(private_residual_path, {})
    private_residual_consumer_summary_for_probe = as_dict(private_residual_consumer.get("summary"))
    fresh_residual_private_probe_path = resolve_report_path(
        args.fresh_residual_private_probe,
        private_residual_consumer_summary_for_probe.get("fresh_residual_private_probe")
        or "reports/candidate_floor_v2_fresh_residual_queue_probe.json",
    )
    fresh_residual_private_probe = read_json(fresh_residual_private_probe_path, {})
    candidate_lookup = read_json(resolve(args.candidate_lookup_audit), {})
    registry = read_json(resolve(args.registry), {})
    hive_operator_assistant = read_json(resolve(args.hive_operator_assistant), {})
    hive_operator_feedback = read_json(resolve(args.hive_operator_feedback), {})
    hive_operator_status = read_json(resolve(args.hive_operator_status), {})

    e2e_summary = as_dict(e2e.get("summary"))
    e2e_cli_case = as_dict(e2e.get("cli_case"))
    e2e_memory_case = as_dict(e2e.get("memory_case"))
    e2e_feedback_case = as_dict(e2e.get("feedback_case"))
    e2e_cli_report = as_dict(e2e_cli_case.get("report"))
    e2e_cli_summary = as_dict(e2e_cli_report.get("summary"))
    runtime_summary = as_dict(runtime.get("summary"))
    runtime_teacher = as_dict(runtime.get("teacher_policy"))
    code_summary = as_dict(code_probe.get("summary"))
    dogfood_summary = as_dict(dogfood.get("summary"))
    planner_summary = as_dict(public_planner.get("summary"))
    planner_capacity = as_dict(public_planner.get("source_capacity"))
    packet_summary = as_dict(public_packet.get("summary"))
    benchmark_summary = as_dict(benchmark_measurement.get("summary"))
    benchmark_capacity = as_dict(benchmark_measurement.get("source_capacity"))
    fresh_capacity = benchmark_capacity or planner_capacity
    benchmark_omitted_cards = benchmark_summary.get("omitted_cards")
    headline_surface_available = bool(benchmark_summary.get("fresh_headline_surface_available"))
    balanced_public_max_cases = benchmark_summary.get("balanced_max_cases_per_card_after_exclusions")
    if balanced_public_max_cases is None:
        balanced_public_max_cases = fresh_capacity.get("balanced_max_cases_per_card_after_exclusions")
    if isinstance(benchmark_omitted_cards, list):
        fresh_public_insufficient_count = len(benchmark_omitted_cards)
    else:
        fresh_public_insufficient_count = fresh_capacity.get("insufficient_card_count")
    benchmark_latest = as_dict(benchmark_summary.get("latest_public_run"))
    benchmark_operator_summary = public_runner_summary(benchmark_operator_dry_run)
    benchmark_execute_summary = public_runner_summary(benchmark_operator_execute)
    public_residual_summary = as_dict(public_residual.get("summary"))
    private_residual_consumer_summary = as_dict(private_residual_consumer.get("summary"))
    fresh_probe_summary = as_dict(fresh_residual_private_probe.get("summary"))
    fresh_probe_queue_eval = as_dict(fresh_probe_summary.get("residual_queue_eval"))
    fresh_probe_unresolved_categories = unresolved_fresh_probe_categories(fresh_probe_queue_eval)
    e2e_cli_evidence = {
        "case_id": e2e_cli_case.get("case_id"),
        "passed": e2e_cli_case.get("passed"),
        "returncode": e2e_cli_case.get("returncode"),
        "runtime_ms": e2e_cli_case.get("runtime_ms"),
        "intent": e2e_cli_summary.get("intent"),
        "code_private_probe_state": get_path(e2e_cli_report, ["code_private_probe", "trigger_state"]),
        "code_private_probe_selected_pass_rate": e2e_cli_summary.get("code_private_probe_selected_pass_rate"),
        "vcm_context_ready": e2e_cli_summary.get("vcm_context_ready"),
        "dogfood_event_written": e2e_cli_summary.get("dogfood_event_written"),
        "public_training_rows_written": e2e_cli_summary.get("public_training_rows_written"),
        "runtime_external_inference_calls": e2e_cli_summary.get("runtime_external_inference_calls"),
        "fallback_return_count": e2e_cli_summary.get("fallback_return_count"),
    }
    public_residual_failures = [
        row
        for row in public_residual.get("task_failure_rows", [])
        if isinstance(row, dict)
    ] if isinstance(public_residual.get("task_failure_rows"), list) else []
    public_residual_counts = count_by_key(public_residual_failures, "residual_type")
    candidate_lookup_summary = as_dict(candidate_lookup.get("summary"))
    registry_summary = as_dict(registry.get("summary"))
    hive_operator_assistant_summary = as_dict(hive_operator_assistant.get("summary"))
    hive_operator_assistant_teacher = as_dict(hive_operator_assistant.get("teacher_policy"))
    hive_operator_feedback_summary = as_dict(hive_operator_feedback)
    hive_operator_status_assistant = as_dict(hive_operator_status.get("assistant"))
    hive_operator_latest = as_dict(hive_operator_status_assistant.get("latest"))

    blockers = []
    if e2e.get("trigger_state") != "GREEN":
        blockers.append("assistant_e2e_not_green")
    if not e2e_cli_case.get("passed"):
        blockers.append("assistant_cli_route_not_verified")
    if not e2e_memory_case.get("passed"):
        blockers.append("assistant_session_memory_recall_not_verified")
    if not e2e_feedback_case.get("passed"):
        blockers.append("assistant_posthoc_cli_feedback_not_verified")
    if code_probe.get("trigger_state") != "GREEN":
        blockers.append("private_code_probe_not_green")
    if int_or_zero(fresh_capacity.get("insufficient_card_count")):
        blockers.append("fresh_balanced_public_5x64_surface_not_available_locally")
    latest_public_score = benchmark_latest.get("pass_rate")
    if latest_public_score is None:
        latest_public_score = packet_summary.get("latest_consumed_public_score")
    latest_public_surface = benchmark_latest.get("run_id") or packet_summary.get("latest_consumed_public_surface")
    latest_public_task_count = benchmark_latest.get("task_count")
    if latest_public_score is not None and float_or_zero(latest_public_score) < 0.25:
        blockers.append("public_code_transfer_score_still_weak")
    if benchmark_summary and not headline_surface_available:
        blockers.append("fresh_public_headline_surface_not_available_locally")
    if int_or_zero(latest_public_task_count) and int_or_zero(latest_public_task_count) < 160:
        blockers.append("latest_public_measurement_is_diagnostic_not_headline")
    if candidate_lookup and candidate_lookup.get("trigger_state") != "GREEN":
        blockers.append("student_candidate_lookup_contract_not_green")
    if int_or_zero(private_residual_consumer_summary.get("unresolved_target_count")):
        if fresh_probe_queue_eval:
            if fresh_probe_unresolved_categories:
                blockers.append("fresh_public_residual_private_probe_still_has_gaps")
        else:
            blockers.append("fresh_public_residual_targets_need_private_ablation")
    if fresh_residual_private_probe and fresh_residual_private_probe.get("trigger_state") not in {"GREEN", "YELLOW"}:
        blockers.append("fresh_public_residual_private_probe_not_clean")
    if int_or_zero(registry_summary.get("registry_hard_governance_violation_count")):
        blockers.append("registry_hard_governance_violation")
    if hive_operator_assistant and hive_operator_assistant.get("trigger_state") not in {"GREEN", "YELLOW"}:
        blockers.append("hive_operator_assistant_not_green")
    if hive_operator_feedback and hive_operator_feedback.get("ok") is not True:
        blockers.append("hive_operator_feedback_not_green")

    gates = [
        gate("assistant_e2e_green", e2e.get("trigger_state") == "GREEN", e2e_summary),
        gate(
            "assistant_tool_evidence_green",
            e2e_summary.get("tool_evidence_state") in {"GREEN", "YELLOW"}
            and int_or_zero(e2e_summary.get("tool_evidence_result_count")) > 0
            and float_or_zero(e2e_summary.get("tool_evidence_tool_on_solve_rate")) > 0.0,
            {
                "tool_evidence_state": e2e_summary.get("tool_evidence_state"),
                "tool_evidence_result_count": e2e_summary.get("tool_evidence_result_count"),
                "tool_evidence_exact_solve_rate": e2e_summary.get("tool_evidence_exact_solve_rate"),
                "tool_evidence_tool_on_solve_rate": e2e_summary.get("tool_evidence_tool_on_solve_rate"),
                "tool_evidence_trace": e2e_summary.get("tool_evidence_trace"),
            },
        ),
        gate(
            "teacher_policy_boundary_green",
            e2e_summary.get("teacher_runtime_external_tokens_forbidden") is True
            and runtime_teacher.get("runtime_external_tokens_forbidden") is True
            and runtime_teacher.get("teacher_apply_mode_forbidden") is True
            and runtime_teacher.get("public_benchmark_distillation_forbidden") is True
            and int_or_zero(runtime_teacher.get("runtime_external_inference_calls")) == 0
            and int_or_zero(runtime_teacher.get("public_training_rows_written")) == 0,
            {
                "e2e_gate_state": e2e_summary.get("teacher_distillation_gate_state"),
                "e2e_distillation_allowed": e2e_summary.get("teacher_distillation_allowed"),
                "e2e_teacher_share": e2e_summary.get("teacher_accepted_row_share"),
                "runtime_gate_state": runtime_teacher.get("gate_state"),
                "runtime_distillation_allowed": runtime_teacher.get("distillation_allowed"),
                "runtime_teacher_share": runtime_teacher.get("teacher_accepted_row_share"),
                "runtime_external_tokens_forbidden": runtime_teacher.get("runtime_external_tokens_forbidden"),
                "teacher_apply_mode_forbidden": runtime_teacher.get("teacher_apply_mode_forbidden"),
                "public_benchmark_distillation_forbidden": runtime_teacher.get("public_benchmark_distillation_forbidden"),
            },
        ),
        gate("assistant_cli_route_verified", bool(e2e_cli_case.get("passed")), e2e_cli_evidence),
        gate(
            "assistant_session_memory_exact_recall_verified",
            bool(e2e_memory_case.get("passed"))
            and e2e_memory_case.get("assistant_text_contains_expected_codename") is True
            and e2e_memory_case.get("assistant_text_contains_expected_constraint") is True,
            {
                "passed": e2e_memory_case.get("passed"),
                "expected_codename": e2e_memory_case.get("expected_codename"),
                "expected_constraint": e2e_memory_case.get("expected_constraint"),
                "assistant_text_contains_expected_codename": e2e_memory_case.get("assistant_text_contains_expected_codename"),
                "assistant_text_contains_expected_constraint": e2e_memory_case.get("assistant_text_contains_expected_constraint"),
                "second_history_turns_loaded": get_path(e2e_memory_case, ["second_report", "summary", "checkpoint_history_turns_loaded"]),
            },
        ),
        gate(
            "assistant_posthoc_cli_feedback_verified",
            bool(e2e_feedback_case.get("passed"))
            and get_path(e2e_feedback_case, ["report", "event_written"]) is True
            and get_path(e2e_feedback_case, ["report", "training_bridge_state"]) in {"GREEN", "YELLOW"}
            and int_or_zero(get_path(e2e_feedback_case, ["report", "public_training_rows_written"])) == 0
            and int_or_zero(get_path(e2e_feedback_case, ["report", "external_inference_calls"])) == 0
            and int_or_zero(get_path(e2e_feedback_case, ["report", "fallback_return_count"])) == 0,
            {
                "passed": e2e_feedback_case.get("passed"),
                "outcome": get_path(e2e_feedback_case, ["report", "outcome"]),
                "event_written": get_path(e2e_feedback_case, ["report", "event_written"]),
                "training_bridge_state": get_path(e2e_feedback_case, ["report", "training_bridge_state"]),
                "training_rows_written": get_path(e2e_feedback_case, ["report", "training_rows_written"]),
                "public_training_rows_written": get_path(e2e_feedback_case, ["report", "public_training_rows_written"]),
                "external_inference_calls": get_path(e2e_feedback_case, ["report", "external_inference_calls"]),
                "fallback_return_count": get_path(e2e_feedback_case, ["report", "fallback_return_count"]),
            },
        ),
        gate("private_code_probe_green", code_probe.get("trigger_state") == "GREEN", code_summary),
        gate("dogfood_training_rows_available", int_or_zero(dogfood_summary.get("trainable_event_count")) > 0, dogfood_summary),
        gate("benchmark_measurement_clean", benchmark_measurement.get("trigger_state") in {"GREEN", "YELLOW"}, benchmark_summary),
        gate("benchmark_operator_dry_run_clean", benchmark_operator_dry_run.get("trigger_state") in {"GREEN", "YELLOW", None}, benchmark_operator_summary),
        gate("benchmark_operator_execute_clean", benchmark_operator_execute.get("trigger_state") in {"GREEN", "YELLOW", None}, benchmark_execute_summary),
        gate("public_residual_report_clean", public_residual.get("trigger_state") in {"GREEN", "YELLOW", None}, public_residual_summary),
        gate("private_residual_consumer_clean", private_residual_consumer.get("trigger_state") in {"GREEN", "YELLOW", None}, private_residual_consumer_summary),
        gate("fresh_residual_private_probe_clean", fresh_residual_private_probe.get("trigger_state") in {"GREEN", "YELLOW", None}, fresh_probe_summary),
        gate("student_candidate_lookup_contract_green", candidate_lookup.get("trigger_state") in {"GREEN", None}, candidate_lookup_summary),
        gate("registry_governance_clean", int_or_zero(registry_summary.get("registry_hard_governance_violation_count")) == 0, registry_summary),
        gate(
            "hive_operator_chat_canonical_assistant_green",
            hive_operator_assistant.get("trigger_state") in {"GREEN", "YELLOW"}
            and hive_operator_assistant_summary.get("vcm_context_ready") is True
            and hive_operator_assistant_summary.get("dogfood_event_written") is True
            and int_or_zero(hive_operator_assistant_summary.get("public_training_rows_written")) == 0
            and int_or_zero(hive_operator_assistant_summary.get("runtime_external_inference_calls")) == 0
            and int_or_zero(hive_operator_assistant_summary.get("fallback_return_count")) == 0,
            hive_operator_assistant_summary,
        ),
        gate(
            "hive_operator_feedback_metadata_only_green",
            hive_operator_feedback.get("ok") is True
            and hive_operator_feedback.get("event_written") is True
            and hive_operator_feedback.get("training_bridge_state") in {"GREEN", "YELLOW"}
            and int_or_zero(hive_operator_feedback.get("public_training_rows_written")) == 0
            and int_or_zero(hive_operator_feedback.get("external_inference_calls")) == 0
            and int_or_zero(hive_operator_feedback.get("fallback_return_count")) == 0,
            {
                "ok": hive_operator_feedback.get("ok"),
                "outcome": hive_operator_feedback.get("outcome"),
                "event_written": hive_operator_feedback.get("event_written"),
                "training_bridge_state": hive_operator_feedback.get("training_bridge_state"),
                "training_rows_written": hive_operator_feedback.get("training_rows_written"),
                "event_report": hive_operator_feedback.get("event_report"),
            },
        ),
        gate("no_public_training_rows", max_public_training_rows(e2e, runtime, code_probe, dogfood, public_planner, public_packet, benchmark_measurement, benchmark_operator_dry_run, benchmark_operator_execute, public_residual, private_residual_consumer, fresh_residual_private_probe, candidate_lookup, hive_operator_assistant, hive_operator_feedback_summary) == 0, 0),
        gate("no_external_inference", max_external_inference(e2e, runtime, code_probe, dogfood, public_planner, public_packet, benchmark_measurement, benchmark_operator_dry_run, benchmark_operator_execute, public_residual, private_residual_consumer, fresh_residual_private_probe, candidate_lookup, hive_operator_assistant, hive_operator_feedback_summary) == 0, 0),
        gate("no_fallback_returns", max_fallback_returns(e2e, runtime, public_packet, hive_operator_assistant, hive_operator_feedback_summary) == 0, 0),
    ]
    hard_failed = [row for row in gates if not row["passed"]]
    trigger_state = "RED" if hard_failed else ("YELLOW" if blockers else "GREEN")
    return {
        "policy": "project_theseus_assistant_state_report_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "assistant_e2e_state": e2e.get("trigger_state"),
            "assistant_cases": f"{e2e_summary.get('passed_case_count')}/{e2e_summary.get('case_count')}",
            "assistant_cli_case_passed": e2e_cli_case.get("passed"),
            "assistant_cli_intent": e2e_cli_summary.get("intent"),
            "assistant_cli_code_probe_state": get_path(e2e_cli_report, ["code_private_probe", "trigger_state"]),
            "assistant_session_memory_case_passed": e2e_memory_case.get("passed"),
            "assistant_session_memory_expected_codename": e2e_memory_case.get("expected_codename"),
            "assistant_session_memory_expected_constraint": e2e_memory_case.get("expected_constraint"),
            "assistant_session_memory_exact_codename_recalled": e2e_memory_case.get("assistant_text_contains_expected_codename"),
            "assistant_session_memory_constraint_recalled": e2e_memory_case.get("assistant_text_contains_expected_constraint"),
            "assistant_session_memory_history_turns_loaded": e2e_summary.get("session_memory_history_turns_loaded"),
            "assistant_posthoc_feedback_case_passed": e2e_feedback_case.get("passed"),
            "assistant_posthoc_feedback_outcome": get_path(e2e_feedback_case, ["report", "outcome"]),
            "assistant_posthoc_feedback_event_written": get_path(e2e_feedback_case, ["report", "event_written"]),
            "assistant_posthoc_feedback_training_bridge_state": get_path(e2e_feedback_case, ["report", "training_bridge_state"]),
            "assistant_posthoc_feedback_training_rows_written": get_path(e2e_feedback_case, ["report", "training_rows_written"]),
            "assistant_tool_evidence_state": e2e_summary.get("tool_evidence_state"),
            "assistant_tool_evidence_result_count": e2e_summary.get("tool_evidence_result_count"),
            "assistant_tool_evidence_exact_solve_rate": e2e_summary.get("tool_evidence_exact_solve_rate"),
            "assistant_tool_evidence_tool_on_solve_rate": e2e_summary.get("tool_evidence_tool_on_solve_rate"),
            "assistant_tool_evidence_trace": e2e_summary.get("tool_evidence_trace"),
            "teacher_distillation_gate_state": e2e_summary.get("teacher_distillation_gate_state") or runtime_teacher.get("gate_state"),
            "teacher_distillation_allowed": e2e_summary["teacher_distillation_allowed"] if "teacher_distillation_allowed" in e2e_summary else runtime_teacher.get("distillation_allowed"),
            "teacher_accepted_row_share": e2e_summary.get("teacher_accepted_row_share", runtime_teacher.get("teacher_accepted_row_share")),
            "teacher_runtime_external_tokens_forbidden": e2e_summary.get("teacher_runtime_external_tokens_forbidden", runtime_teacher.get("runtime_external_tokens_forbidden")),
            "teacher_runtime_gate_state": runtime_teacher.get("gate_state"),
            "teacher_manifest_verifier_pass_rate": runtime_teacher.get("manifest_verifier_pass_rate"),
            "teacher_manifest_public_overlap_hits": runtime_teacher.get("manifest_public_overlap_hits"),
            "teacher_manifest_holdout_overlap_hits": runtime_teacher.get("manifest_holdout_overlap_hits"),
            "runtime_state": runtime.get("trigger_state"),
            "runtime_vcm_ready": runtime_summary.get("vcm_context_ready"),
            "runtime_vcm_pages": runtime_summary.get("vcm_selected_page_count"),
            "private_code_probe_state": code_probe.get("trigger_state"),
            "private_code_probe_tasks": code_summary.get("task_count"),
            "private_code_probe_selected_pass_rate": code_summary.get("selected_intended_behavior_pass_rate"),
            "private_code_probe_compile_rate": code_summary.get("selected_compile_pass_rate"),
            "private_code_probe_runtime_load_rate": code_summary.get("selected_runtime_load_rate"),
            "dogfood_trainable_events": dogfood_summary.get("trainable_event_count"),
            "dogfood_training_enabled": dogfood_summary.get("training_enabled"),
            "dogfood_raw_text_capture_enabled": dogfood_summary.get("raw_text_capture_enabled"),
            "latest_public_score": latest_public_score,
            "latest_public_surface": latest_public_surface,
            "latest_public_task_count": latest_public_task_count,
            "benchmark_latest_public_score": benchmark_latest.get("pass_rate"),
            "benchmark_latest_public_surface": benchmark_latest.get("run_id"),
            "benchmark_latest_public_task_count": benchmark_latest.get("task_count"),
            "benchmark_measurement_kind": benchmark_summary.get("measurement_kind"),
            "benchmark_fresh_headline_available": benchmark_summary.get("fresh_headline_surface_available"),
            "benchmark_diagnostic_packet": benchmark_summary.get("packet"),
            "benchmark_operator_dry_run_would_execute": benchmark_operator_summary.get("would_execute"),
            "benchmark_operator_dry_run_run_registry_allowed": benchmark_operator_summary.get("run_registry_allowed"),
            "benchmark_operator_dry_run_authorization_mode": benchmark_operator_summary.get("authorization_mode"),
            "benchmark_operator_execute_state": benchmark_operator_execute.get("trigger_state"),
            "benchmark_operator_execute_returncode": benchmark_execute_summary.get("run_returncode"),
            "benchmark_operator_execute_output": benchmark_execute_summary.get("output_path"),
            "public_residual_report_state": public_residual.get("trigger_state"),
            "public_residual_current_pass_rate": public_residual_summary.get("current_public_pass_rate"),
            "public_residual_current_task_count": public_residual_summary.get("current_public_task_count"),
            "public_residual_failure_counts": public_residual_counts,
            "private_residual_target_count": len(public_residual.get("private_only_residual_target_rows", []))
            if isinstance(public_residual.get("private_only_residual_target_rows"), list)
            else None,
            "private_residual_consumer_state": private_residual_consumer.get("trigger_state"),
            "private_residual_unresolved_target_count": private_residual_consumer_summary.get("unresolved_target_count"),
            "private_residual_unresolved_target_category_counts": private_residual_consumer_summary.get("unresolved_target_category_counts"),
            "private_residual_fresh_counts": private_residual_consumer_summary.get("fresh_public_residual_counts"),
            "fresh_residual_private_probe_state": fresh_residual_private_probe.get("trigger_state"),
            "fresh_residual_private_probe_task_count": fresh_probe_summary.get("private_eval_task_count"),
            "fresh_residual_private_probe_pass_rate": fresh_probe_summary.get("private_trained_pass_rate"),
            "fresh_residual_private_probe_candidate_coverage": fresh_probe_summary.get("candidate_coverage_rate"),
            "fresh_residual_private_probe_open_categories": fresh_probe_queue_eval.get("open_categories"),
            "fresh_residual_private_probe_category_counts": fresh_probe_queue_eval.get("targeted_private_eval_category_counts"),
            "fresh_residual_private_probe_min_pass_rates": fresh_probe_queue_eval.get("targeted_private_eval_min_family_pass_rate_by_category"),
            "fresh_residual_private_probe_missing_categories": fresh_probe_queue_eval.get("missing_open_categories"),
            "fresh_residual_private_probe_unresolved_categories": fresh_probe_unresolved_categories,
            "candidate_lookup_state": candidate_lookup.get("trigger_state"),
            "candidate_lookup_eligible_task_coverage_rate": candidate_lookup_summary.get("normalized_eligible_task_coverage_rate"),
            "candidate_lookup_tasks_without_eligible_candidates": candidate_lookup_summary.get("tasks_without_normalized_eligible_candidates"),
            "historical_no_candidate_tasks_now_covered": candidate_lookup_summary.get("historical_no_candidate_tasks_now_have_normalized_eligible_candidates"),
            "fresh_public_balanced_max_cases_per_card": balanced_public_max_cases,
            "fresh_public_insufficient_card_count": fresh_public_insufficient_count,
            "registry_state": registry.get("trigger_state"),
            "registry_governance_violations": registry_summary.get("registry_governance_violation_count"),
            "hive_operator_assistant_state": hive_operator_assistant.get("trigger_state"),
            "hive_operator_assistant_intent": hive_operator_assistant_summary.get("intent"),
            "hive_operator_assistant_feedback": hive_operator_assistant_summary.get("feedback"),
            "hive_operator_assistant_vcm_ready": hive_operator_assistant_summary.get("vcm_context_ready"),
            "hive_operator_assistant_history_turns": hive_operator_assistant_summary.get("checkpoint_history_turns_loaded"),
            "hive_operator_assistant_dogfood_event_written": hive_operator_assistant_summary.get("dogfood_event_written"),
            "hive_operator_assistant_report": hive_operator_assistant.get("report"),
            "hive_operator_assistant_teacher_gate_state": hive_operator_assistant_teacher.get("gate_state"),
            "hive_operator_assistant_teacher_runtime_tokens_forbidden": hive_operator_assistant_teacher.get("runtime_external_tokens_forbidden"),
            "hive_operator_status_assistant_state": hive_operator_status_assistant.get("state"),
            "hive_operator_status_latest_state": hive_operator_latest.get("trigger_state"),
            "hive_operator_feedback_ok": hive_operator_feedback.get("ok"),
            "hive_operator_feedback_outcome": hive_operator_feedback.get("outcome"),
            "hive_operator_feedback_event_written": hive_operator_feedback.get("event_written"),
            "hive_operator_feedback_training_bridge_state": hive_operator_feedback.get("training_bridge_state"),
            "blocker_count": len(blockers),
        },
        "blockers": blockers,
        "evidence": {
            "assistant_e2e": rel(resolve(args.assistant_e2e)),
            "assistant_runtime": rel(resolve(args.assistant_runtime)),
            "code_probe": rel(resolve(args.code_probe)),
            "dogfood_bridge": rel(resolve(args.dogfood_bridge)),
            "public_planner": rel(resolve(args.public_planner)),
            "public_packet": rel(resolve(args.public_packet)),
            "benchmark_measurement": rel(resolve(args.benchmark_measurement)),
            "public_run_registry": rel(resolve(args.public_run_registry)),
            "benchmark_operator_dry_run": rel(resolve(args.benchmark_operator_dry_run)),
            "benchmark_operator_execute": rel(resolve(args.benchmark_operator_execute)),
            "public_residual_report": rel(public_residual_path),
            "private_residual_consumer": rel(private_residual_path),
            "fresh_residual_private_probe": rel(fresh_residual_private_probe_path),
            "candidate_lookup_audit": rel(resolve(args.candidate_lookup_audit)),
            "registry": rel(resolve(args.registry)),
            "hive_operator_assistant": rel(resolve(args.hive_operator_assistant)),
            "hive_operator_feedback": rel(resolve(args.hive_operator_feedback)),
            "hive_operator_status": rel(resolve(args.hive_operator_status)),
        },
        "recommendation": recommendation(blockers, fresh_capacity, latest_public_score),
        "gates": gates,
        "public_benchmark_boundary": {
            "benchmarks_may_be_run_for_measurement": True,
            "train_on_public_prompts_tests_solutions_traces_or_scores": False,
            "current_fresh_balanced_5x64_available": headline_surface_available,
            "model_only_and_tool_assisted_scores_must_be_separate": True,
        },
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def recommendation(blockers: list[str], planner_capacity: dict[str, Any], latest_public_score: Any) -> list[str]:
    actions = []
    if "public_code_transfer_score_still_weak" in blockers:
        actions.append("Keep improving private semantic candidate quality and selection; current public transfer evidence remains weak.")
    if "fresh_public_residual_private_probe_still_has_gaps" in blockers:
        actions.append(
            "Close the remaining fresh residual private-probe gaps before the next headline public measurement: add dependency-free heldout eval and repair any listed category whose targeted private min pass rate is below 1.0."
        )
    if "fresh_public_residual_targets_need_private_ablation" in blockers:
        actions.append(
            "Run private ablations for the fresh public-safe residual categories before claiming transfer repair: algorithm choice, edge cases, return/type handling, and dependency-free candidates."
        )
    if "fresh_balanced_public_5x64_surface_not_available_locally" in blockers:
        actions.append(
            "Stage legitimate additional public calibration sources or run only a clearly labeled small diagnostic; do not rerun consumed public surfaces for a new headline score."
        )
        for row in planner_capacity.get("insufficient_cards", []) if isinstance(planner_capacity.get("insufficient_cards"), list) else []:
            actions.append(
                f"{row.get('card_id')}: {row.get('available_after_exclusions')}/{row.get('required_task_count')} unused rows available after exclusions."
            )
    if not actions:
        actions.append("Continue dogfood use and route code tasks through the private verifier-backed assistant path.")
    if latest_public_score is not None:
        actions.append(f"Latest consumed public score is {latest_public_score}; do not train on that public payload.")
    return actions


def render_markdown(report: dict[str, Any]) -> str:
    summary = as_dict(report.get("summary"))
    lines = [
        "# Theseus Assistant State Report",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- assistant E2E: `{summary.get('assistant_e2e_state')}` cases `{summary.get('assistant_cases')}`",
        f"- assistant CLI route: `{summary.get('assistant_cli_case_passed')}` intent `{summary.get('assistant_cli_intent')}` code probe `{summary.get('assistant_cli_code_probe_state')}`",
        f"- assistant session memory: exact codename `{summary.get('assistant_session_memory_exact_codename_recalled')}` constraint `{summary.get('assistant_session_memory_constraint_recalled')}` history `{summary.get('assistant_session_memory_history_turns_loaded')}`",
        f"- assistant post-hoc feedback: `{summary.get('assistant_posthoc_feedback_case_passed')}` outcome `{summary.get('assistant_posthoc_feedback_outcome')}` event `{summary.get('assistant_posthoc_feedback_event_written')}` bridge `{summary.get('assistant_posthoc_feedback_training_bridge_state')}` rows `{summary.get('assistant_posthoc_feedback_training_rows_written')}`",
        f"- assistant tool evidence: `{summary.get('assistant_tool_evidence_state')}` results `{summary.get('assistant_tool_evidence_result_count')}` exact_solve `{summary.get('assistant_tool_evidence_exact_solve_rate')}`",
        f"- teacher policy: `{summary.get('teacher_distillation_gate_state')}` allowed `{summary.get('teacher_distillation_allowed')}` share `{summary.get('teacher_accepted_row_share')}` runtime tokens forbidden `{summary.get('teacher_runtime_external_tokens_forbidden')}`",
        f"- VCM ready: `{summary.get('runtime_vcm_ready')}` pages `{summary.get('runtime_vcm_pages')}`",
        f"- private code probe: `{summary.get('private_code_probe_state')}` pass `{summary.get('private_code_probe_selected_pass_rate')}` tasks `{summary.get('private_code_probe_tasks')}`",
        f"- dogfood trainable events: `{summary.get('dogfood_trainable_events')}` raw text capture `{summary.get('dogfood_raw_text_capture_enabled')}`",
        f"- latest public score: `{summary.get('latest_public_score')}` surface `{summary.get('latest_public_surface')}`",
        f"- benchmark measurement: `{summary.get('benchmark_measurement_kind')}` latest `{summary.get('benchmark_latest_public_score')}` tasks `{summary.get('benchmark_latest_public_task_count')}` fresh headline `{summary.get('benchmark_fresh_headline_available')}`",
        f"- benchmark packet: `{summary.get('benchmark_diagnostic_packet')}` dry-run would execute `{summary.get('benchmark_operator_dry_run_would_execute')}` run registry `{summary.get('benchmark_operator_dry_run_run_registry_allowed')}`",
        f"- benchmark execute: `{summary.get('benchmark_operator_execute_state')}` returncode `{summary.get('benchmark_operator_execute_returncode')}` output `{summary.get('benchmark_operator_execute_output')}`",
        f"- residual mining: `{summary.get('public_residual_report_state')}` pass `{summary.get('public_residual_current_pass_rate')}` targets `{summary.get('private_residual_target_count')}` failures `{summary.get('public_residual_failure_counts')}`",
        f"- private residual consumer: `{summary.get('private_residual_consumer_state')}` unresolved `{summary.get('private_residual_unresolved_target_count')}` categories `{summary.get('private_residual_unresolved_target_category_counts')}`",
        f"- fresh residual private probe: `{summary.get('fresh_residual_private_probe_state')}` pass `{summary.get('fresh_residual_private_probe_pass_rate')}` coverage `{summary.get('fresh_residual_private_probe_candidate_coverage')}` unresolved `{summary.get('fresh_residual_private_probe_unresolved_categories')}`",
        f"- candidate lookup: `{summary.get('candidate_lookup_state')}` eligible coverage `{summary.get('candidate_lookup_eligible_task_coverage_rate')}` historical no-candidate tasks now covered `{summary.get('historical_no_candidate_tasks_now_covered')}`",
        f"- fresh public balanced max/card: `{summary.get('fresh_public_balanced_max_cases_per_card')}`",
        f"- registry: `{summary.get('registry_state')}` violations `{summary.get('registry_governance_violations')}`",
        f"- Hive operator assistant: `{summary.get('hive_operator_assistant_state')}` intent `{summary.get('hive_operator_assistant_intent')}` feedback `{summary.get('hive_operator_assistant_feedback')}` dogfood `{summary.get('hive_operator_assistant_dogfood_event_written')}`",
        f"- Hive operator feedback: ok `{summary.get('hive_operator_feedback_ok')}` outcome `{summary.get('hive_operator_feedback_outcome')}` event `{summary.get('hive_operator_feedback_event_written')}` bridge `{summary.get('hive_operator_feedback_training_bridge_state')}`",
        "",
        "## Blockers",
    ]
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Recommendation"])
    for action in report.get("recommendation", []):
        lines.append(f"- {action}")
    lines.extend(["", "## Gates"])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}`")
    return "\n".join(lines).rstrip() + "\n"


def max_public_training_rows(*reports: dict[str, Any]) -> int:
    return max([0, *[int_or_zero(report.get("public_training_rows_written"), as_dict(report.get("summary")).get("public_training_rows")) for report in reports]])


def max_external_inference(*reports: dict[str, Any]) -> int:
    return max([0, *[int_or_zero(report.get("external_inference_calls"), as_dict(report.get("summary")).get("external_inference_calls")) for report in reports]])


def max_fallback_returns(*reports: dict[str, Any]) -> int:
    return max([0, *[int_or_zero(report.get("fallback_return_count"), as_dict(report.get("summary")).get("fallback_return_count")) for report in reports]])


def public_runner_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = dict(as_dict(report.get("summary")))
    for key in list(summary):
        if key.startswith("calendar_or_monthly_"):
            summary.pop(key, None)
    summary.setdefault("calendar_throttle_enabled", False)
    authorization_mode = summary.get("authorization_mode")
    if authorization_mode and authorization_mode != "run_registry" and str(authorization_mode).endswith("_registry"):
        summary["authorization_mode"] = "run_registry_legacy_report"
    return summary


def latest_scored_registry_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        row
        for row in rows
        if row.get("consumed") is True
        and (row.get("score_recorded") is True or row.get("real_public_task_pass_rate") is not None)
    ]
    if not scored:
        return {}
    return sorted(scored, key=lambda row: str(row.get("created_utc") or row.get("registry_enriched_utc") or ""))[-1]


def resolve_report_path(requested: str | Path | None, inferred: Any) -> Path:
    if requested:
        return resolve(requested)
    if isinstance(inferred, str) and inferred:
        return resolve(inferred)
    return resolve("reports/missing_report.json")


def inferred_private_residual_consumer(public_residual_path: Path, latest_registry_row: dict[str, Any]) -> str:
    run_id = latest_registry_row.get("run_id")
    if isinstance(run_id, str) and run_id:
        candidate = REPORTS / f"private_residual_target_consumer_{run_id}.json"
        if candidate.exists():
            return rel(candidate)
    stem = public_residual_path.stem
    prefix = "bounded_public_transfer_residual_mining_"
    if stem.startswith(prefix):
        candidate = public_residual_path.with_name(f"private_residual_target_consumer_{stem[len(prefix):]}.json")
        if candidate.exists():
            return rel(candidate)
        return rel(candidate)
    return "reports/private_residual_target_consumer_public_transfer_measurement_mbpp_signature_repair_seed4_5x30.json"


def count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def unresolved_fresh_probe_categories(queue_eval: dict[str, Any]) -> list[str]:
    unresolved = set()
    missing = queue_eval.get("missing_open_categories")
    if isinstance(missing, list):
        unresolved.update(str(item) for item in missing if str(item))
    min_rates = queue_eval.get("targeted_private_eval_min_family_pass_rate_by_category")
    if isinstance(min_rates, dict):
        for category, rate in min_rates.items():
            try:
                value = float(rate)
            except (TypeError, ValueError):
                unresolved.add(str(category))
                continue
            if value < 1.0:
                unresolved.add(str(category))
    return sorted(unresolved)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return data if isinstance(data, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(candidate).replace("\\", "/")


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def int_or_zero(*values: Any) -> int:
    for value in values:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
