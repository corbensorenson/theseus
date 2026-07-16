#!/usr/bin/env python3
"""End-to-end verifier for the canonical Theseus assistant runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import reflexive_dispatch


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DOGFOOD_EVENTS = ROOT / "runtime" / "dogfood" / "daily_use_events.jsonl"
DEFAULT_OUT = REPORTS / "theseus_assistant_e2e.json"
DEFAULT_MD = REPORTS / "theseus_assistant_e2e.md"


CASES = [
    {
        "id": "chat_status",
        "intent": "chat",
        "feedback": "completed",
        "prompt": "Summarize current Theseus assistant status in one concise answer.",
    },
    {
        "id": "code_route",
        "intent": "code",
        "feedback": "corrected",
        "prompt": "Route this coding request: fix a Python parser that returns the wrong shape.",
    },
    {
        "id": "tool_route",
        "intent": "tool",
        "feedback": "accepted",
        "prompt": "Use deterministic tools where appropriate for math, search, or verification tasks.",
    },
    {
        "id": "planning_route",
        "intent": "planning",
        "feedback": "completed",
        "prompt": "Plan the next cohesive Theseus assistant improvement as a VCM-backed DAG.",
    },
    {
        "id": "composite_tool_planning_route",
        "intent": "chat",
        "feedback": "completed",
        "prompt": "/plan-tool inspect architecture evidence before planning",
    },
    {
        "id": "effect_route_rollback",
        "intent": "chat",
        "feedback": "completed",
        "prompt": "Exercise the bounded local effect transaction and prove rollback.",
        "extra_args": ["--effect-canary"],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--session-prefix", default="assistant_e2e")
    parser.add_argument("--skip-context-refresh-after-first", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.perf_counter()
    rows = []
    for index, case in enumerate(CASES):
        rows.append(run_case(case, args, index=index))
    cli_case = run_cli_case(args)
    feedback_case = run_feedback_case(args)
    memory_case = run_memory_case(args)
    gates = build_gates(rows)
    reflexbench_profile = reflexive_dispatch.evaluate_reflexbench_profile()
    reflexbench_verification = reflexive_dispatch.verify_reflexbench_result(reflexbench_profile)
    gates.extend(reflexbench_gates(reflexbench_profile, reflexbench_verification))
    usefulness_report = build_usefulness_report(rows, cli_case, feedback_case, memory_case)
    gates.append(
        gate(
            "cli_chat_uses_assistant_runtime",
            bool(cli_case.get("passed")),
            cli_case,
            "hard",
        )
    )
    gates.append(
        gate(
            "session_memory_loads_previous_turn",
            bool(memory_case.get("passed")),
            memory_case,
            "hard",
        )
    )
    gates.append(
        gate(
            "posthoc_cli_feedback_records_metadata",
            bool(feedback_case.get("passed")),
            feedback_case,
            "hard",
        )
    )
    hard_failures = [gate for gate in gates if gate["severity"] == "hard" and not gate["passed"]]
    warning_failures = [gate for gate in gates if gate["severity"] == "warning" and not gate["passed"]]
    trigger_state = "GREEN" if not hard_failures else "RED"
    if trigger_state == "GREEN" and warning_failures:
        trigger_state = "YELLOW"
    row_dogfood_rows = sum(int(get_path(row, ["report", "summary", "dogfood_training_rows_written"], 0) or 0) for row in rows)
    cli_dogfood_rows = int(get_path(cli_case, ["report", "summary", "dogfood_training_rows_written"], 0) or 0)
    row_public_rows = sum(int(get_path(row, ["report", "summary", "public_training_rows_written"], 0) or 0) for row in rows)
    cli_public_rows = int(get_path(cli_case, ["report", "public_training_rows_written"], 0) or 0)
    row_external_calls = sum(int(get_path(row, ["report", "external_inference_calls"], 0) or 0) for row in rows)
    cli_external_calls = int(get_path(cli_case, ["report", "external_inference_calls"], 0) or 0)
    row_fallback_count = sum(int(get_path(row, ["report", "fallback_return_count"], 0) or 0) for row in rows)
    cli_fallback_count = int(get_path(cli_case, ["report", "fallback_return_count"], 0) or 0)
    memory_public_rows = int(get_path(memory_case, ["second_report", "public_training_rows_written"], 0) or 0)
    memory_external_calls = int(get_path(memory_case, ["second_report", "external_inference_calls"], 0) or 0)
    memory_fallback_count = int(get_path(memory_case, ["second_report", "fallback_return_count"], 0) or 0)
    feedback_public_rows = int(get_path(feedback_case, ["report", "public_training_rows_written"], 0) or 0)
    feedback_external_calls = int(get_path(feedback_case, ["report", "external_inference_calls"], 0) or 0)
    feedback_fallback_count = int(get_path(feedback_case, ["report", "fallback_return_count"], 0) or 0)
    feedback_training_rows = int(get_path(feedback_case, ["report", "training_rows_written"], 0) or 0)
    tool_case = next((row for row in rows if row.get("case_id") == "tool_route"), {})
    chat_case = next((row for row in rows if row.get("case_id") == "chat_status"), {})
    report = {
        "policy": "project_theseus_assistant_e2e_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "case_count": len(rows),
            "passed_case_count": len([row for row in rows if row.get("passed")]),
            "user_facing_cli_case_passed": bool(cli_case.get("passed")),
            "cli_code_probe_state": get_path(cli_case, ["report", "code_private_probe", "trigger_state"], None),
            "posthoc_feedback_case_passed": bool(feedback_case.get("passed")),
            "posthoc_feedback_outcome": get_path(feedback_case, ["report", "outcome"], ""),
            "posthoc_feedback_event_written": get_path(feedback_case, ["report", "event_written"], False),
            "posthoc_feedback_training_bridge_state": get_path(feedback_case, ["report", "training_bridge_state"], ""),
            "posthoc_feedback_training_rows_written": feedback_training_rows,
            "session_memory_case_passed": bool(memory_case.get("passed")),
            "session_memory_history_turns_loaded": get_path(memory_case, ["second_report", "summary", "checkpoint_history_turns_loaded"], None),
            "tool_evidence_state": get_path(tool_case, ["report", "tool_evidence", "trigger_state"], ""),
            "tool_evidence_result_count": get_path(tool_case, ["report", "tool_evidence", "summary", "result_count"], 0),
            "tool_evidence_exact_solve_rate": get_path(tool_case, ["report", "tool_evidence", "summary", "exact_solve_rate"], None),
            "tool_evidence_tool_on_solve_rate": get_path(tool_case, ["report", "tool_evidence", "summary", "tool_on_solve_rate"], None),
            "tool_evidence_trace": get_path(tool_case, ["report", "tool_evidence", "trace"], ""),
            "teacher_distillation_gate_state": get_path(chat_case, ["report", "teacher_policy", "gate_state"], ""),
            "teacher_distillation_allowed": get_path(chat_case, ["report", "teacher_policy", "distillation_allowed"], None),
            "teacher_accepted_row_share": get_path(chat_case, ["report", "teacher_policy", "teacher_accepted_row_share"], None),
            "teacher_runtime_external_tokens_forbidden": get_path(chat_case, ["report", "teacher_policy", "runtime_external_tokens_forbidden"], None),
            "latest_public_run": get_path(chat_case, ["report", "benchmark_status", "latest_public_run_id"], ""),
            "latest_public_score": get_path(chat_case, ["report", "benchmark_status", "latest_public_pass_rate"], None),
            "latest_public_task_count": get_path(chat_case, ["report", "benchmark_status", "latest_public_task_count"], None),
            "latest_public_measurement_kind": get_path(chat_case, ["report", "benchmark_status", "measurement_kind"], None),
            "latest_public_dominant_residual": get_path(chat_case, ["report", "benchmark_status", "dominant_residual"], []),
            "dogfood_training_rows_written": row_dogfood_rows + cli_dogfood_rows + feedback_training_rows,
            "public_training_rows_written": row_public_rows + cli_public_rows + memory_public_rows + feedback_public_rows,
            "external_inference_calls": row_external_calls + cli_external_calls + memory_external_calls + feedback_external_calls,
            "fallback_return_count": row_fallback_count + cli_fallback_count + memory_fallback_count + feedback_fallback_count,
            "usefulness_recent_event_count": usefulness_report.get("recent_event_count"),
            "usefulness_trainable_event_count": usefulness_report.get("trainable_event_count"),
            "usefulness_completed_or_accepted_count": usefulness_report.get("completed_or_accepted_count"),
            "current_code_generator_wall": usefulness_report.get("current_code_generator_wall"),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "reflexbench_profile_id": reflexbench_profile.get("profile_id"),
            "reflexbench_case_count": reflexbench_profile.get("case_count"),
            "reflexbench_policy_count": reflexbench_profile.get("policy_count"),
            "reflexbench_full_mechanics_ready": reflexbench_profile.get("full_reflexive_pretraining_mechanics_ready"),
        },
        "usefulness_report": usefulness_report,
        "cases": rows,
        "cli_case": cli_case,
        "feedback_case": feedback_case,
        "memory_case": memory_case,
        "reflexbench_profile": reflexbench_profile,
        "reflexbench_verification": reflexbench_verification,
        "gates": gates,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if trigger_state == "RED" else 0


def run_case(case: dict[str, Any], args: argparse.Namespace, *, index: int) -> dict[str, Any]:
    case_id = str(case["id"])
    out = REPORTS / f"theseus_assistant_e2e_{case_id}.json"
    md = REPORTS / f"theseus_assistant_e2e_{case_id}.md"
    command = [
        sys.executable,
        "scripts/theseus_assistant_runtime.py",
        "--prompt",
        str(case["prompt"]),
        "--intent",
        str(case["intent"]),
        "--feedback",
        str(case["feedback"]),
        "--session-id",
        f"{args.session_prefix}_{case_id}",
        "--out",
        rel(out),
        "--markdown-out",
        rel(md),
        "--events-out",
        "reports/theseus_assistant_e2e_events.jsonl",
        "--print-answer",
    ]
    command.extend(str(value) for value in case.get("extra_args", []) if str(value))
    if args.skip_context_refresh_after_first and index > 0:
        command.append("--skip-context-refresh")
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=240)
    report = read_json(out, {})
    passed = (
        result.returncode == 0
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and bool(get_path(report, ["summary", "assistant_text_chars"], 0))
        and int(default_if_none(get_path(report, ["summary", "public_training_rows_written"], None), 1)) == 0
        and int(report.get("external_inference_calls") or 0) == 0
        and int(report.get("fallback_return_count") or 0) == 0
    )
    return {
        "case_id": case_id,
        "intent": case.get("intent"),
        "returncode": result.returncode,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "passed": passed,
        "out": rel(out),
        "markdown": rel(md),
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
        "report": compact_report(report),
    }


def run_cli_case(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/theseus_cli.py",
        "chat",
        "--json",
        "--session-id",
        f"{args.session_prefix}_cli_code",
        "--intent",
        "code",
        "--feedback",
        "corrected",
        "Route a CLI coding request through the assistant runtime.",
    ]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=240)
    report = parse_stdout_json(result.stdout)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    code_probe = report.get("code_private_probe") if isinstance(report.get("code_private_probe"), dict) else {}
    code_probe_summary = code_probe.get("summary") if isinstance(code_probe.get("summary"), dict) else {}
    passed = (
        result.returncode == 0
        and report.get("policy") == "project_theseus_assistant_runtime_v0"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and summary.get("intent") == "code"
        and bool(report.get("assistant_text"))
        and code_probe_executed_safely(code_probe)
        and int(report.get("public_training_rows_written") or 0) == 0
        and int(report.get("external_inference_calls") or 0) == 0
        and int(report.get("fallback_return_count") or 0) == 0
    )
    return {
        "case_id": "cli_code_chat",
        "returncode": result.returncode,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "passed": passed,
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
        "report": compact_report(report),
    }


def run_feedback_case(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/theseus_cli.py",
        "feedback",
        "accepted",
        "--session-id",
        f"{args.session_prefix}_cli_code",
        "--latest-report",
        "reports/checkpoint_chat_last.json",
        "--intent-summary-redacted",
        "assistant_e2e_posthoc_feedback",
        "--out",
        "reports/theseus_assistant_e2e_feedback.json",
        "--markdown-out",
        "reports/theseus_assistant_e2e_feedback.md",
    ]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=240)
    report = parse_stdout_json(result.stdout)
    passed = (
        result.returncode == 0
        and report.get("policy") == "project_theseus_cli_assistant_feedback_v0"
        and report.get("ok") is True
        and report.get("event_written") is True
        and report.get("training_bridge_state") in {"GREEN", "YELLOW"}
        and report.get("outcome") == "accepted"
        and int(report.get("public_training_rows_written") or 0) == 0
        and int(report.get("external_inference_calls") or 0) == 0
        and int(report.get("fallback_return_count") or 0) == 0
    )
    return {
        "case_id": "posthoc_cli_feedback",
        "returncode": result.returncode,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "passed": passed,
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
        "report": {
            "policy": report.get("policy"),
            "ok": report.get("ok"),
            "outcome": report.get("outcome"),
            "event_written": report.get("event_written"),
            "training_bridge_state": report.get("training_bridge_state"),
            "training_rows_written": report.get("training_rows_written"),
            "public_training_rows_written": report.get("public_training_rows_written"),
            "external_inference_calls": report.get("external_inference_calls"),
            "fallback_return_count": report.get("fallback_return_count"),
            "event_report": report.get("event_report"),
            "training_bridge_report": report.get("training_bridge_report"),
        },
    }


def run_memory_case(args: argparse.Namespace) -> dict[str, Any]:
    session_id = f"{args.session_prefix}_memory"
    first_command = [
        sys.executable,
        "scripts/theseus_cli.py",
        "chat",
        "--json",
        "--session-id",
        session_id,
        "--intent",
        "chat",
        "--feedback",
        "completed",
        "Remember for this session: the codename is Atlas Reed and the constraint is no fallback returns.",
    ]
    second_command = [
        sys.executable,
        "scripts/theseus_cli.py",
        "chat",
        "--json",
        "--session-id",
        session_id,
        "--intent",
        "chat",
        "--feedback",
        "completed",
        "What codename and constraint did I just give you?",
    ]
    started = time.perf_counter()
    first = subprocess.run(first_command, cwd=ROOT, text=True, capture_output=True, timeout=240)
    second = subprocess.run(second_command, cwd=ROOT, text=True, capture_output=True, timeout=240)
    first_report = parse_stdout_json(first.stdout)
    second_report = parse_stdout_json(second.stdout)
    second_summary = second_report.get("summary") if isinstance(second_report.get("summary"), dict) else {}
    assistant_text = str(second_report.get("assistant_text") or "")
    expected_codename = "Atlas Reed"
    expected_constraint = "no fallback returns"
    passed = (
        first.returncode == 0
        and second.returncode == 0
        and second_report.get("policy") == "project_theseus_assistant_runtime_v0"
        and int(second_summary.get("checkpoint_history_turns_loaded") or 0) >= 1
        and expected_codename in assistant_text
        and expected_constraint in assistant_text
        and int(second_report.get("public_training_rows_written") or 0) == 0
        and int(second_report.get("external_inference_calls") or 0) == 0
        and int(second_report.get("fallback_return_count") or 0) == 0
    )
    return {
        "case_id": "session_memory_chat",
        "returncode_first": first.returncode,
        "returncode_second": second.returncode,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "passed": passed,
        "expected_codename": expected_codename,
        "expected_constraint": expected_constraint,
        "assistant_text_contains_expected_codename": expected_codename in assistant_text,
        "assistant_text_contains_expected_constraint": expected_constraint in assistant_text,
        "first_stdout_tail": first.stdout[-1200:],
        "second_stdout_tail": second.stdout[-1200:],
        "first_stderr_tail": first.stderr[-1200:],
        "second_stderr_tail": second.stderr[-1200:],
        "first_report": compact_report(first_report),
        "second_report": compact_report(second_report),
    }


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "trigger_state": report.get("trigger_state"),
        "summary": summary,
        "code_route": report.get("code_route"),
        "code_private_probe": report.get("code_private_probe"),
        "tool_context": report.get("tool_context"),
        "tool_evidence": report.get("tool_evidence"),
        "plan_context": report.get("plan_context"),
        "effect_canary": report.get("effect_canary"),
        "teacher_policy": report.get("teacher_policy"),
        "benchmark_status": report.get("benchmark_status"),
        "vcm_context_packet": {
            "task_family_id": get_path(report, ["vcm_context_packet", "task_family_id"], ""),
            "ready": get_path(report, ["vcm_context_packet", "ready"], False),
            "selected_page_count": get_path(report, ["vcm_context_packet", "selected_page_count"], 0),
        },
        "external_inference_calls": report.get("external_inference_calls"),
        "public_training_rows_written": report.get("public_training_rows_written"),
        "fallback_return_count": report.get("fallback_return_count"),
    }


def code_probe_executed_safely(probe: Any) -> bool:
    if not isinstance(probe, dict):
        return False
    summary = probe.get("summary") if isinstance(probe.get("summary"), dict) else {}
    rules = probe.get("rules") if isinstance(probe.get("rules"), dict) else {}
    return (
        probe.get("active") is True
        and probe.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(default_if_none(summary.get("candidate_row_count"), 0)) > 0
        and int(default_if_none(summary.get("tasks_with_manifest_candidates"), 0)) > 0
        and int(default_if_none(summary.get("public_boundary_violation_count"), 1)) == 0
        and int(default_if_none(summary.get("fallback_return_candidate_count"), 1)) == 0
        and int(default_if_none(summary.get("unconditional_constant_return_candidate_count"), 1)) == 0
        and int(default_if_none(rules.get("public_training_rows_written"), 0)) == 0
        and int(default_if_none(rules.get("external_inference_calls"), 0)) == 0
        and rules.get("public_calibration_run") is not True
    )


def code_probe_wall(probe: Any) -> dict[str, Any]:
    if not isinstance(probe, dict):
        probe = {}
    summary = probe.get("summary") if isinstance(probe.get("summary"), dict) else {}
    selected_pass = float_or_zero(summary.get("selected_intended_behavior_pass_rate"))
    pass_if_any = float_or_zero(summary.get("pass_if_any_rate"))
    return {
        "safe_boundary": code_probe_executed_safely(probe),
        "trigger_state": probe.get("trigger_state"),
        "task_count": int(default_if_none(summary.get("task_count"), 0)),
        "candidate_row_count": int(default_if_none(summary.get("candidate_row_count"), 0)),
        "tasks_with_manifest_candidates": int(default_if_none(summary.get("tasks_with_manifest_candidates"), 0)),
        "eligible_candidate_count": int(default_if_none(summary.get("eligible_candidate_count"), 0)),
        "candidate_integrity_mismatch_count": int(default_if_none(summary.get("candidate_integrity_mismatch_count"), 0)),
        "selected_intended_behavior_pass_rate": selected_pass,
        "pass_if_any_rate": pass_if_any,
        "semantic_pass_currently_zero": selected_pass == 0.0 and pass_if_any == 0.0,
        "public_boundary_violation_count": int(default_if_none(summary.get("public_boundary_violation_count"), 0)),
        "fallback_return_candidate_count": int(default_if_none(summary.get("fallback_return_candidate_count"), 0)),
        "unconditional_constant_return_candidate_count": int(default_if_none(summary.get("unconditional_constant_return_candidate_count"), 0)),
    }


def code_probe_reports_wall(probe: Any) -> bool:
    wall = code_probe_wall(probe)
    return bool(
        wall.get("safe_boundary")
        and (
            float_or_zero(wall.get("selected_intended_behavior_pass_rate")) > 0.0
            or int(wall.get("eligible_candidate_count") or 0) == 0
            or int(wall.get("candidate_integrity_mismatch_count") or 0) > 0
        )
    )


def build_usefulness_report(
    rows: list[dict[str, Any]],
    cli_case: dict[str, Any],
    feedback_case: dict[str, Any],
    memory_case: dict[str, Any],
) -> dict[str, Any]:
    events = read_jsonl(DOGFOOD_EVENTS)
    recent_events = events[-80:]
    outcome_counts: dict[str, int] = {}
    lane_counts: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    latest_refs: list[dict[str, Any]] = []
    for event in recent_events:
        if not isinstance(event, dict):
            continue
        outcome = str(event.get("outcome") or "unknown")
        lane = str(event.get("assistant_lane") or "unknown")
        surface = str(event.get("surface") or "unknown")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        surface_counts[surface] = surface_counts.get(surface, 0) + 1
    for event in recent_events[-8:]:
        if isinstance(event, dict):
            latest_refs.append(
                {
                    "created_utc": event.get("created_utc"),
                    "surface": event.get("surface"),
                    "assistant_lane": event.get("assistant_lane"),
                    "outcome": event.get("outcome"),
                    "event_id": event.get("event_id"),
                    "artifact_count": len(event.get("artifact_refs") or []) if isinstance(event.get("artifact_refs"), list) else 0,
                }
            )
    by_id = {str(row.get("case_id")): row for row in rows}
    code_probe = get_path(by_id, ["code_route", "report", "code_private_probe"], {})
    tool_case = get_path(by_id, ["tool_route", "report"], {})
    vcm_ready_cases = [
        row.get("case_id")
        for row in rows
        if get_path(row, ["report", "vcm_context_packet", "ready"], False)
    ]
    trainable_event_count = sum(outcome_counts.get(key, 0) for key in ["accepted", "completed", "corrected", "missed", "ignored"])
    completed_or_accepted_count = outcome_counts.get("completed", 0) + outcome_counts.get("accepted", 0)
    return {
        "policy": "project_theseus_assistant_usefulness_report_v0",
        "recent_event_count": len(recent_events),
        "outcome_counts": outcome_counts,
        "lane_counts": lane_counts,
        "surface_counts": surface_counts,
        "latest_events": latest_refs,
        "trainable_event_count": trainable_event_count,
        "completed_or_accepted_count": completed_or_accepted_count,
        "code_probe_safe_boundary": code_probe_executed_safely(code_probe),
        "current_code_generator_wall": code_probe_wall(code_probe),
        "tool_evidence_state": get_path(tool_case, ["tool_evidence", "trigger_state"], ""),
        "tool_evidence_result_count": get_path(tool_case, ["tool_evidence", "summary", "result_count"], 0),
        "tool_evidence_tool_on_solve_rate": get_path(tool_case, ["tool_evidence", "summary", "tool_on_solve_rate"], None),
        "vcm_ready_case_count": len(vcm_ready_cases),
        "vcm_ready_cases": vcm_ready_cases,
        "cli_case_passed": bool(cli_case.get("passed")),
        "feedback_case_passed": bool(feedback_case.get("passed")),
        "memory_case_passed": bool(memory_case.get("passed")),
        "raw_text_training_allowed": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_gates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("case_id")): row for row in rows}
    return [
        gate("all_cases_passed", all(row.get("passed") for row in rows), {"failed": [row.get("case_id") for row in rows if not row.get("passed")]}, "hard"),
        gate(
            "all_cases_use_verified_reflexive_predispatch",
            all(
                get_path(row, ["report", "summary", "reflexive_dispatch_verified"], False) is True
                and get_path(row, ["report", "summary", "reflexive_dispatch_prepared"], False) is True
                and get_path(row, ["report", "summary", "reflexive_dispatch_downstream_skipped"], True) is False
                and get_path(row, ["report", "summary", "reflexive_dispatch_selected_capabilities"], [])
                for row in rows
            ),
            {
                row.get("case_id"): {
                    "terminal": get_path(row, ["report", "summary", "reflexive_dispatch_terminal_outcome"], ""),
                    "verified": get_path(row, ["report", "summary", "reflexive_dispatch_verified"], False),
                    "selected": get_path(row, ["report", "summary", "reflexive_dispatch_selected_capabilities"], []),
                    "skipped": get_path(row, ["report", "summary", "reflexive_dispatch_downstream_skipped"], None),
                }
                for row in rows
            },
            "hard",
        ),
        gate("chat_case_has_vcm", bool(get_path(by_id, ["chat_status", "report", "vcm_context_packet", "ready"], False)), get_path(by_id, ["chat_status", "report", "vcm_context_packet"], {}), "hard"),
        gate("code_case_routes_to_code_lane", bool(get_path(by_id, ["code_route", "report", "code_route", "active"], False)), get_path(by_id, ["code_route", "report", "code_route"], {}), "hard"),
        gate(
            "code_case_has_safe_private_verifier_probe",
            code_probe_executed_safely(get_path(by_id, ["code_route", "report", "code_private_probe"], {})),
            get_path(by_id, ["code_route", "report", "code_private_probe"], {}),
            "hard",
        ),
        gate(
            "code_case_reports_current_generator_wall",
            code_probe_reports_wall(get_path(by_id, ["code_route", "report", "code_private_probe"], {})),
            code_probe_wall(get_path(by_id, ["code_route", "report", "code_private_probe"], {})),
            "warning",
        ),
        gate("tool_case_has_registry", bool(get_path(by_id, ["tool_route", "report", "tool_context", "active"], False)), get_path(by_id, ["tool_route", "report", "tool_context"], {}), "hard"),
        gate(
            "tool_case_has_executed_evidence",
            bool(get_path(by_id, ["tool_route", "report", "tool_evidence", "active"], False))
            and get_path(by_id, ["tool_route", "report", "tool_evidence", "trigger_state"], "") in {"GREEN", "YELLOW"}
            and int(get_path(by_id, ["tool_route", "report", "tool_evidence", "summary", "result_count"], 0) or 0) > 0
            and float_or_zero(get_path(by_id, ["tool_route", "report", "tool_evidence", "summary", "tool_on_solve_rate"], 0)) > 0.0,
            get_path(by_id, ["tool_route", "report", "tool_evidence"], {}),
            "hard",
        ),
        gate("planning_case_has_compiler", bool(get_path(by_id, ["planning_route", "report", "plan_context", "active"], False)), get_path(by_id, ["planning_route", "report", "plan_context"], {}), "hard"),
        gate(
            "composite_case_executes_tool_then_planning",
            get_path(by_id, ["composite_tool_planning_route", "report", "summary", "reflexive_dispatch_selected_capabilities"], [])
            == ["assistant.deterministic_tool", "assistant.plan_dag"]
            and get_path(by_id, ["composite_tool_planning_route", "report", "tool_evidence", "active"], False) is True
            and get_path(by_id, ["composite_tool_planning_route", "report", "plan_context", "active"], False) is True,
            get_path(by_id, ["composite_tool_planning_route", "report"], {}),
            "hard",
        ),
        gate(
            "effect_case_binds_dispatch_and_rolls_back",
            get_path(by_id, ["effect_route_rollback", "report", "summary", "reflexive_dispatch_selected_capabilities"], [])
            == ["assistant.route_authority_effect"]
            and get_path(by_id, ["effect_route_rollback", "report", "effect_canary", "dispatch_bound"], False) is True
            and get_path(by_id, ["effect_route_rollback", "report", "effect_canary", "ready"], False) is True
            and get_path(by_id, ["effect_route_rollback", "report", "effect_canary", "rollback", "complete"], False) is True,
            get_path(by_id, ["effect_route_rollback", "report", "effect_canary"], {}),
            "hard",
        ),
        gate(
            "teacher_policy_carried_by_all_cases",
            all(
                get_path(row, ["report", "teacher_policy", "runtime_external_tokens_forbidden"], False) is True
                and get_path(row, ["report", "teacher_policy", "teacher_apply_mode_forbidden"], False) is True
                and get_path(row, ["report", "teacher_policy", "public_benchmark_distillation_forbidden"], False) is True
                for row in rows
            ),
            {
                row.get("case_id"): {
                    "gate_state": get_path(row, ["report", "teacher_policy", "gate_state"], ""),
                    "distillation_allowed": get_path(row, ["report", "teacher_policy", "distillation_allowed"], None),
                    "teacher_share": get_path(row, ["report", "teacher_policy", "teacher_accepted_row_share"], None),
                    "runtime_tokens_forbidden": get_path(row, ["report", "teacher_policy", "runtime_external_tokens_forbidden"], None),
                    "apply_mode_forbidden": get_path(row, ["report", "teacher_policy", "teacher_apply_mode_forbidden"], None),
                }
                for row in rows
            },
            "hard",
        ),
        gate(
            "teacher_distillation_clean_when_allowed",
            all(
                get_path(row, ["report", "teacher_policy", "distillation_allowed"], False) is not True
                or (
                    get_path(row, ["report", "teacher_policy", "gate_state"], "") == "GREEN"
                    and int(get_path(row, ["report", "teacher_policy", "gate_hard_blocker_count"], 0) or 0) == 0
                    and float_or_zero(get_path(row, ["report", "teacher_policy", "manifest_verifier_pass_rate"], 0)) >= 0.95
                    and int(get_path(row, ["report", "teacher_policy", "manifest_public_overlap_hits"], 0) or 0) == 0
                    and int(get_path(row, ["report", "teacher_policy", "manifest_holdout_overlap_hits"], 0) or 0) == 0
                    and get_path(row, ["report", "teacher_policy", "teacher_share_within_cap"], False) is True
                )
                for row in rows
            ),
            {row.get("case_id"): get_path(row, ["report", "teacher_policy"], {}) for row in rows},
            "hard",
        ),
        gate(
            "assistant_viea_trace_present_for_nontrivial_cases",
            assistant_viea_trace_present_for_nontrivial_cases(rows),
            {
                row.get("case_id"): {
                    "intent": row.get("intent"),
                    "required": row.get("intent") in {"code", "tool", "planning"},
                    "complete": get_path(row, ["report", "summary", "assistant_viea_trace_complete"], None),
                    "record_count": get_path(row, ["report", "summary", "assistant_viea_trace_record_count"], 0),
                    "trace_out": get_path(row, ["report", "summary", "assistant_viea_trace_out"], ""),
                }
                for row in rows
            },
            "hard",
        ),
        gate("no_public_training_rows", not any(int(get_path(row, ["report", "public_training_rows_written"], 0) or 0) for row in rows), 0, "hard"),
        gate("no_external_inference", not any(int(get_path(row, ["report", "external_inference_calls"], 0) or 0) for row in rows), 0, "hard"),
        gate("no_fallback_returns", not any(int(get_path(row, ["report", "fallback_return_count"], 0) or 0) for row in rows), 0, "hard"),
    ]


def reflexbench_gates(profile: dict[str, Any], verification: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = {str(row.get("policy_id")): row for row in profile.get("policy_metrics", []) if isinstance(row, dict)}
    full = metrics.get("full_reflexive", {})
    no_chronicle = metrics.get("reflexive_without_chronicle", {})
    no_compiler = metrics.get("reflexive_without_compiler", {})
    return [
        gate("reflexbench_result_replay_verified", verification.get("state") == "VERIFIED", verification, "hard"),
        gate(
            "reflexbench_frozen_8track_10policy_denominators_complete",
            profile.get("case_count") == 32
            and profile.get("track_count") == 8
            and profile.get("policy_count") == 10
            and profile.get("result_count") == 320,
            {key: profile.get(key) for key in ("profile_id", "profile_digest", "case_count", "track_count", "policy_count", "result_count")},
            "hard",
        ),
        gate(
            "reflexbench_full_mechanics_pass_without_authority_or_verification_escape",
            profile.get("full_reflexive_pretraining_mechanics_ready") is True
            and full.get("useful_rate") == 1.0
            and full.get("unauthorized_action_rate") == 0.0
            and full.get("verification_escape_rate") == 0.0
            and full.get("silent_fallback_rate") == 0.0,
            full,
            "hard",
        ),
        gate(
            "reflexbench_lifecycle_ablations_are_causal",
            get_path(full, ["per_track", "E_temporal_chronicle", "rate"], 0) == 1.0
            and get_path(no_chronicle, ["per_track", "E_temporal_chronicle", "rate"], 1) < 1.0
            and get_path(full, ["per_track", "H_reflex_compilation", "rate"], 0) == 1.0
            and get_path(no_compiler, ["per_track", "H_reflex_compilation", "rate"], 1) < 1.0,
            {"full": full.get("per_track"), "no_chronicle": no_chronicle.get("per_track"), "no_compiler": no_compiler.get("per_track")},
            "hard",
        ),
        gate(
            "reflexbench_information_boundary_and_no_cheat_clean",
            get_path(profile, ["no_cheat", "non_oracle_held_field_reads"], 1) == 0
            and get_path(profile, ["no_cheat", "learned_generation_credit"], 1) == 0
            and get_path(profile, ["no_cheat", "external_inference_calls"], 1) == 0
            and get_path(profile, ["no_cheat", "public_training_rows_written"], 1) == 0
            and get_path(profile, ["no_cheat", "fallback_return_count"], 1) == 0,
            profile.get("no_cheat"),
            "hard",
        ),
    ]


def assistant_viea_trace_present_for_nontrivial_cases(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("intent") not in {"code", "tool", "planning"}:
            continue
        if get_path(row, ["report", "summary", "assistant_viea_trace_complete"], False) is not True:
            return False
        if int(get_path(row, ["report", "summary", "assistant_viea_trace_record_count"], 0) or 0) < 16:
            return False
    return True


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    usefulness = report.get("usefulness_report") if isinstance(report.get("usefulness_report"), dict) else {}
    current_wall = usefulness.get("current_code_generator_wall") if isinstance(usefulness.get("current_code_generator_wall"), dict) else {}
    lines = [
        "# Theseus Assistant E2E",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- cases: `{summary.get('passed_case_count')}/{summary.get('case_count')}`",
        f"- user-facing CLI case: `{summary.get('user_facing_cli_case_passed')}` code probe `{summary.get('cli_code_probe_state')}`",
        f"- session memory case: `{summary.get('session_memory_case_passed')}` history `{summary.get('session_memory_history_turns_loaded')}`",
        f"- usefulness events: `{usefulness.get('recent_event_count')}` trainable=`{usefulness.get('trainable_event_count')}` completed_or_accepted=`{usefulness.get('completed_or_accepted_count')}`",
        f"- current code generator wall: safe=`{current_wall.get('safe_boundary')}` selected_pass=`{current_wall.get('selected_intended_behavior_pass_rate')}` pass_if_any=`{current_wall.get('pass_if_any_rate')}` eligible=`{current_wall.get('eligible_candidate_count')}` integrity_mismatches=`{current_wall.get('candidate_integrity_mismatch_count')}`",
        f"- teacher distillation: `{summary.get('teacher_distillation_gate_state')}` allowed `{summary.get('teacher_distillation_allowed')}` share `{summary.get('teacher_accepted_row_share')}` runtime tokens forbidden `{summary.get('teacher_runtime_external_tokens_forbidden')}`",
        f"- dogfood rows written: `{summary.get('dogfood_training_rows_written')}`",
        f"- public training rows: `{summary.get('public_training_rows_written')}`",
        f"- external inference calls: `{summary.get('external_inference_calls')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    return "\n".join(lines).rstrip() + "\n"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return data if isinstance(data, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def parse_stdout_json(stdout: str) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def default_if_none(value: Any, default: Any) -> Any:
    return default if value is None else value


def float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
