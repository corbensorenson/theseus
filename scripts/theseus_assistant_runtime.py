#!/usr/bin/env python3
"""Canonical local Theseus assistant runtime.

This composes the existing checkpoint chat, VCM context bridge, deterministic
tool registry, plan compiler, code-route metadata, and dogfood feedback bridge
into one user-facing contract. It does not train on public benchmark payloads,
serve external inference, or emit fallback answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import viea_spine_records
import vcm_consumer_abi
import reflexive_dispatch


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_assistant_runtime.json"
DEFAULT_OUT = REPORTS / "theseus_assistant_runtime.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_assistant_runtime.md"
DEFAULT_EVENTS = REPORTS / "theseus_assistant_conversation_events.jsonl"
DEFAULT_VIEA_TRACE = REPORTS / "theseus_assistant_viea_trace.jsonl"
DEFAULT_ASSISTANT_TRACE_SCHEMA = ROOT / "configs" / "assistant_trace_schema.json"
DEFAULT_PROCEDURAL_ADOPTION_REPORT = REPORTS / "procedural_memory_route_adoption.json"
DEFAULT_EFFECT_ROOT = ROOT / "runtime" / "assistant_effects"
DEFAULT_EFFECT_TARGET = DEFAULT_EFFECT_ROOT / "default_route_authority.json"
DOGFOOD_EVENTS = ROOT / "runtime" / "dogfood" / "daily_use_events.jsonl"
DEFAULT_ALLOWED_FEEDBACK = {"", "accepted", "missed", "ignored", "corrected", "completed"}
ASSISTANT_VIEA_REQUIRED_RECORD_TYPES = viea_spine_records.ASSISTANT_RUNTIME_REQUIRED_RECORDS
ASSISTANT_PRODUCT_VIEW_GROUPS = [
    "claim_ledger_entries",
    "artifact_records",
    "failure_boundaries",
    "authority_records",
    "runtime_adapter_records",
    "resource_route_records",
    "generation_mode_records",
    "context_records",
]
ROUTE_VALIDATOR_VIEW_GROUPS = [
    "governance_records",
    "failure_boundaries",
    "authority_records",
    "resource_route_records",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint-id", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--surface", default="local_assistant")
    parser.add_argument("--intent", default="auto", choices=["auto", "chat", "code", "tool", "planning"])
    parser.add_argument("--principal", default="local-user")
    parser.add_argument("--origin", default="local_user_control")
    parser.add_argument("--unauthenticated", action="store_true")
    parser.add_argument("--requested-route", default="")
    parser.add_argument("--fallback-policy", default="no_fallback", choices=["no_fallback", "explicit_only"])
    parser.add_argument("--effort", default="balanced", choices=["direct", "balanced", "deliberative"])
    parser.add_argument("--feedback", default="completed")
    parser.add_argument("--error-family", default="")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--events-out", default=rel(DEFAULT_EVENTS))
    parser.add_argument("--viea-trace-out", default=rel(DEFAULT_VIEA_TRACE))
    parser.add_argument("--skip-context-refresh", action="store_true")
    parser.add_argument("--skip-dogfood", action="store_true")
    parser.add_argument(
        "--effect-canary",
        action="store_true",
        help="Exercise a bounded local route-authority write, independent observation, and exact rollback.",
    )
    parser.add_argument("--effect-target", default=rel(DEFAULT_EFFECT_TARGET))
    parser.add_argument("--print-answer", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    append_jsonl(conversation_events_path(args.events_out), conversation_event(report))
    for row in report.get("assistant_viea_trace", []) if isinstance(report.get("assistant_viea_trace"), list) else []:
        append_jsonl(resolve(args.viea_trace_out), row)
    if args.print_answer:
        print(report.get("assistant_text") or "")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    config = read_json(resolve(args.config), {})
    trace_schema = load_assistant_trace_schema(config)
    allowed_feedback = allowed_feedback_values(trace_schema)
    created_utc = now()
    prompt = str(args.prompt or "")
    session_id = safe_slug(args.session_id or config.get("default_session_id") or "local_assistant")
    checkpoint_id = str(args.checkpoint_id or config.get("checkpoint_id") or "live")
    requested_intent = classify_intent(prompt, args.intent)
    context_refresh = [] if args.skip_context_refresh else refresh_context(config)
    materialized_view_receipt = assistant_materialized_view_receipt()
    route_validator_receipt = assistant_route_validator_receipt()
    private_verifier_receipt = private_verifier_receipt_packet()
    reflexive_trace = build_reflexive_dispatch_trace(
        prompt=prompt,
        requested_intent=requested_intent,
        args=args,
        config=config,
        materialized_view_receipt=materialized_view_receipt,
        route_validator_receipt=route_validator_receipt,
        private_verifier_receipt=private_verifier_receipt,
    )
    reflexive_verification = verify_reflexive_dispatch_trace(reflexive_trace, config)
    selected_capabilities = selected_reflexive_capabilities(reflexive_trace)
    intent = effective_intent_from_dispatch(reflexive_trace, requested_intent)
    route = route_for(config, intent)
    dispatch_prepared = reflexive_dispatch_prepared(reflexive_trace, reflexive_verification)
    vcm_governor = vcm_context_governor_packet()
    contexts = read_json(REPORTS / "vcm_task_contexts.json", {})
    selected_context = select_vcm_context(contexts, str(route.get("vcm_task_family") or "operator_chat"))
    selected_pages = selected_context.get("selected_pages") if isinstance(selected_context.get("selected_pages"), list) else []
    vcm_consumer_packet = vcm_consumer_abi.build_consumer_packet(
        consumer_id="theseus_assistant_runtime",
        purpose="assistant",
        read_set=["reports/vcm_context_governor.json", "reports/vcm_task_contexts.json"],
        write_set=[rel(resolve(args.out)), rel(resolve(args.viea_trace_out))],
        authority_ceiling=["local_assistant_context_read", "local_session_memory_write"],
        permitted_uses=["assistant_context", "conversation_continuity", "planning_and_tool_routing", "dogfood_metadata"],
        context_refs=[
            {
                "kind": "semantic_address",
                "ref": page.get("address") or page.get("source_path"),
                "required": True,
                "exists": bool(page.get("address") or page.get("source_path")),
                "sha256": page.get("content_hash") or page.get("sha256") or "",
                "taint_labels": page.get("taints", []),
                "contradiction_refs": page.get("contradiction_refs", []),
            }
            for page in selected_pages
            if isinstance(page, dict)
        ],
        taint_labels=sorted({str(taint) for page in selected_pages if isinstance(page, dict) for taint in list_value(page.get("taints"))}),
        deletion_obligations=["invalidate_session_context_derivatives_when_source_context_is_revoked"],
        contradiction_refs=[
            str(ref)
            for page in selected_pages
            if isinstance(page, dict)
            for ref in list_value(page.get("contradiction_refs"))
            if ref
        ],
        compression_loss=float(get_path(vcm_governor, ["summary", "mission_brief_compression_loss"], 0.0) or 0.0),
        audit_refs=["scripts/theseus_assistant_runtime.py", "reports/vcm_task_contexts.json"],
    )
    vcm_governor["consumer_abi"] = vcm_consumer_packet
    vcm_governor["ready"] = bool(vcm_governor.get("ready")) and bool(vcm_consumer_packet.get("ready"))
    checkpoint_out = REPORTS / f"theseus_assistant_checkpoint_chat_{session_id}.json"
    chat_result = (
        run_checkpoint_chat(
            prompt=prompt,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            out=checkpoint_out,
        )
        if dispatch_prepared
        else {
            "returncode": None,
            "skipped": True,
            "skip_reason": f"reflexive_dispatch_{reflexive_terminal_outcome(reflexive_trace)}",
            "stderr_tail": "",
        }
    )
    checkpoint_payload = read_json(checkpoint_out, {}) if dispatch_prepared else {}
    response = checkpoint_payload.get("response") if isinstance(checkpoint_payload.get("response"), dict) else {}
    checkpoint_session = checkpoint_payload.get("session") if isinstance(checkpoint_payload.get("session"), dict) else {}
    code_route = code_route_packet(intent, route) if dispatch_prepared else {"active": False, "skipped": True}
    code_private_probe = run_code_private_probe(intent, route) if dispatch_prepared else {"active": False, "skipped": True}
    tool_required = "assistant.deterministic_tool" in selected_capabilities
    planning_required = "assistant.plan_dag" in selected_capabilities
    tool_context = tool_context_packet(tool_required) if dispatch_prepared else {"active": False, "required": tool_required, "skipped": True}
    tool_evidence = run_tool_evidence(tool_required, route_for(config, "tool")) if dispatch_prepared else {"active": False, "required": tool_required, "skipped": True}
    plan_context = plan_context_packet(planning_required) if dispatch_prepared else {"active": False, "required": planning_required, "skipped": True}
    procedural_default_route = (
        procedural_default_route_packet(intent, route, config, surface=str(args.surface or "local_assistant"))
        if dispatch_prepared
        else {"active": False, "ready": False, "skipped": True}
    )
    effect_canary = run_local_effect_canary(
        enabled=(
            bool(args.effect_canary)
            and dispatch_prepared
            and selected_reflexive_capabilities(reflexive_trace) == ["assistant.route_authority_effect"]
        ),
        target=resolve(args.effect_target),
        allowed_root=DEFAULT_EFFECT_ROOT,
        session_id=session_id,
        intent=intent,
        prompt_hash=sha256_text(prompt),
        reflexive_dispatch_trace=reflexive_trace,
    )
    teacher_policy = teacher_policy_packet()
    benchmark_status = benchmark_status_packet(prompt)
    assistant_text = compose_assistant_text(
        intent=intent,
        base_text=str(response.get("answer") or ""),
        code_route=code_route,
        code_private_probe=code_private_probe,
        tool_context=tool_context,
        tool_evidence=tool_evidence,
        plan_context=plan_context,
        procedural_default_route=procedural_default_route,
        teacher_policy=teacher_policy,
        benchmark_status=benchmark_status,
        vcm_governor=vcm_governor,
        selected_context=selected_context,
        checkpoint_session=checkpoint_session,
    ) if dispatch_prepared else reflexive_terminal_text(reflexive_trace)
    feedback = normalize_feedback(args.feedback, allowed_feedback)
    dogfood = {}
    if dispatch_prepared and not args.skip_dogfood and feedback:
        dogfood = run_dogfood_feedback(
            feedback=feedback,
            surface=str(args.surface or "local_assistant"),
            lane=str(route.get("assistant_lane") or "chat_checkpoint"),
            intent_summary=redacted_intent_summary(prompt, intent),
            artifact_refs=artifact_refs_for_feedback(args.out, checkpoint_out, code_private_probe, tool_evidence, procedural_default_route),
            error_family=str(args.error_family or ""),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    assistant_viea_trace = build_assistant_viea_trace(
        created_utc=created_utc,
        prompt_hash=sha256_text(prompt),
        prompt_summary=redacted_intent_summary(prompt, intent),
        args=args,
        config=config,
        intent=intent,
        reflexive_dispatch_trace=reflexive_trace,
        route=route,
        session_id=session_id,
        checkpoint_id=checkpoint_id,
        selected_context=selected_context,
        checkpoint_out=checkpoint_out,
        chat_result=chat_result,
        checkpoint_session=checkpoint_session,
        code_route=code_route,
        code_private_probe=code_private_probe,
        tool_context=tool_context,
        tool_evidence=tool_evidence,
        plan_context=plan_context,
        procedural_default_route=procedural_default_route,
        teacher_policy=teacher_policy,
        benchmark_status=benchmark_status,
        materialized_view_receipt=materialized_view_receipt,
        route_validator_receipt=route_validator_receipt,
        private_verifier_receipt=private_verifier_receipt,
        vcm_governor=vcm_governor,
        dogfood=dogfood,
        feedback=feedback,
        assistant_text=assistant_text,
        trace_schema=trace_schema,
        effect_canary=effect_canary,
    )
    assistant_viea_trace.extend(
        row for row in vcm_consumer_packet.get("records", []) if isinstance(row, dict)
    )
    gates = build_gates(
        chat_result=chat_result,
        response=response,
        assistant_text=assistant_text,
        reflexive_dispatch_trace=reflexive_trace,
        reflexive_dispatch_verification=reflexive_verification,
        feedback=feedback,
        dogfood=dogfood,
        selected_context=selected_context,
        context_refresh=context_refresh,
        intent=intent,
        code_private_probe=code_private_probe,
        tool_evidence=tool_evidence,
        procedural_default_route=procedural_default_route,
        teacher_policy=teacher_policy,
        config=config,
        checkpoint_session=checkpoint_session,
        materialized_view_receipt=materialized_view_receipt,
        route_validator_receipt=route_validator_receipt,
        private_verifier_receipt=private_verifier_receipt,
        vcm_governor=vcm_governor,
        assistant_viea_trace=assistant_viea_trace,
        trace_schema=trace_schema,
        allowed_feedback=allowed_feedback,
        effect_canary=effect_canary,
    )
    gates.append(
        gate(
            "vcm_consumer_abi_ready",
            bool(vcm_consumer_packet.get("ready")) and bool(vcm_consumer_packet.get("validation", {}).get("passed")),
            {
                "packet_id": vcm_consumer_packet.get("packet_id"),
                "typed_faults": vcm_consumer_packet.get("typed_faults"),
                "validation": vcm_consumer_packet.get("validation"),
            },
            "hard",
        )
    )
    hard_failures = [gate for gate in gates if gate["severity"] == "hard" and not gate["passed"]]
    warning_failures = [gate for gate in gates if gate["severity"] == "warning" and not gate["passed"]]
    trigger_state = "GREEN" if not hard_failures else "RED"
    if trigger_state == "GREEN" and warning_failures:
        trigger_state = "YELLOW"
    summary = {
        "requested_intent": requested_intent,
        "intent": intent,
        "session_id": session_id,
        "checkpoint_id": checkpoint_id,
        "assistant_lane": route.get("assistant_lane"),
        "reflexive_dispatch_trace_id": reflexive_trace.get("trace_id"),
        "reflexive_dispatch_decision_digest": reflexive_trace.get("decision_digest"),
        "reflexive_dispatch_terminal_outcome": reflexive_terminal_outcome(reflexive_trace),
        "reflexive_dispatch_prepared": dispatch_prepared,
        "reflexive_dispatch_verified": reflexive_verification.get("state") == "VERIFIED",
        "reflexive_dispatch_selected_capabilities": selected_reflexive_capabilities(reflexive_trace),
        "reflexive_dispatch_downstream_skipped": not dispatch_prepared,
        "vcm_task_family": route.get("vcm_task_family"),
        "vcm_context_ready": selected_context.get("ready"),
        "vcm_selected_page_count": len(selected_context.get("selected_pages") or []),
        "vcm_context_governor_state": vcm_governor.get("trigger_state"),
        "vcm_context_governor_ready": vcm_governor.get("ready"),
        "vcm_context_adequacy_state": vcm_governor.get("adequacy_state"),
        "vcm_mission_brief_status": get_path(vcm_governor, ["summary", "mission_brief_status"], None),
        "vcm_deletion_closure_status": get_path(vcm_governor, ["summary", "deletion_closure_status"], None),
        "vcm_context_governor_hard_gap_count": get_path(vcm_governor, ["summary", "hard_gap_count"], 0),
        "vcm_consumer_abi_ready": vcm_consumer_packet.get("ready"),
        "vcm_consumer_abi_packet_id": vcm_consumer_packet.get("packet_id"),
        "vcm_consumer_abi_fault_count": len(vcm_consumer_packet.get("typed_faults") or []),
        "checkpoint_chat_returncode": chat_result.get("returncode"),
        "checkpoint_history_turns_loaded": checkpoint_session.get("history_turns_loaded"),
        "checkpoint_session_path": checkpoint_session.get("session_path"),
        "assistant_text_chars": len(assistant_text),
        "code_private_probe_state": code_private_probe.get("trigger_state") if code_private_probe.get("active") else "",
        "code_private_probe_selected_pass_rate": get_path(code_private_probe, ["summary", "selected_intended_behavior_pass_rate"], None),
        "code_private_probe_pass_if_any_rate": get_path(code_private_probe, ["summary", "pass_if_any_rate"], None),
        "code_private_probe_task_count": get_path(code_private_probe, ["summary", "task_count"], 0),
        "code_private_probe_candidate_integrity_mismatch_count": get_path(code_private_probe, ["summary", "candidate_integrity_mismatch_count"], 0),
        "code_private_probe_eligible_candidate_count": get_path(code_private_probe, ["summary", "eligible_candidate_count"], 0),
        "code_private_probe_safe_boundary": code_private_probe_safe(code_private_probe),
        "code_private_probe_current_wall": code_private_probe_wall(code_private_probe),
        "tool_evidence_state": tool_evidence.get("trigger_state") if tool_evidence.get("active") else "",
        "tool_evidence_result_count": get_path(tool_evidence, ["summary", "result_count"], 0),
        "tool_evidence_exact_solve_rate": get_path(tool_evidence, ["summary", "exact_solve_rate"], None),
        "tool_evidence_tool_on_solve_rate": get_path(tool_evidence, ["summary", "tool_on_solve_rate"], None),
        "tool_evidence_trace": tool_evidence.get("trace", ""),
        "procedural_default_route_active": procedural_default_route.get("active"),
        "procedural_default_route_ready": procedural_default_route.get("ready"),
        "procedural_default_route_state": procedural_default_route.get("trigger_state"),
        "procedural_default_route_id": get_path(procedural_default_route, ["selected_route", "id"], ""),
        "procedural_default_route_candidate_id": get_path(procedural_default_route, ["selected_route", "candidate_id"], ""),
        "procedural_default_route_guard_armed": get_path(procedural_default_route, ["selected_route", "continued_regression_guard", "armed"], False),
        "procedural_default_route_learned_generation_claim_allowed": get_path(procedural_default_route, ["selected_route", "learned_generation_claim_allowed"], None),
        "procedural_default_route_selection_matched": get_path(procedural_default_route, ["selection", "matched"], False),
        "procedural_default_route_selection_mode": get_path(procedural_default_route, ["selection", "selection_mode"], ""),
        "procedural_default_route_scope": get_path(procedural_default_route, ["selection", "route_scope"], ""),
        "procedural_default_route_runtime_consumers": get_path(procedural_default_route, ["selected_route", "runtime_consumers"], []),
        "procedural_default_route_public_training_rows_written": procedural_default_route.get("public_training_rows_written", 0),
        "procedural_default_route_external_inference_calls": procedural_default_route.get("external_inference_calls", 0),
        "procedural_default_route_fallback_return_count": procedural_default_route.get("fallback_return_count", 0),
        "feedback": feedback,
        "dogfood_event_written": get_path(dogfood, ["event", "event_written"], False),
        "dogfood_training_rows_written": get_path(dogfood, ["training_bridge", "training_rows_written"], 0),
        "dogfood_training_bridge_state": get_path(dogfood, ["training_bridge", "trigger_state"], ""),
        "teacher_distillation_gate_state": teacher_policy.get("gate_state"),
        "teacher_distillation_allowed": teacher_policy.get("distillation_allowed"),
        "teacher_accepted_row_share": teacher_policy.get("teacher_accepted_row_share"),
        "teacher_accepted_rows": teacher_policy.get("teacher_accepted_rows"),
        "teacher_proposal_rows_recorded": teacher_policy.get("teacher_proposal_rows_recorded"),
        "teacher_runtime_external_tokens_forbidden": teacher_policy.get("runtime_external_tokens_forbidden"),
        "latest_public_run": benchmark_status.get("latest_public_run_id"),
        "latest_public_cards": benchmark_status.get("latest_public_cards"),
        "latest_public_score": benchmark_status.get("latest_public_pass_rate"),
        "latest_public_passed": benchmark_status.get("latest_public_passed"),
        "latest_public_task_count": benchmark_status.get("latest_public_task_count"),
        "latest_public_measurement_kind": benchmark_status.get("measurement_kind"),
        "latest_public_dominant_residual": benchmark_status.get("dominant_residual"),
        "context_refresh_failed_count": len([row for row in context_refresh if row.get("returncode") not in {0, None}]),
        "public_training_rows_written": 0,
        "runtime_external_inference_calls": 0,
        "fallback_return_count": 0,
        "assistant_viea_trace_required": assistant_viea_trace_required(intent),
        "assistant_viea_trace_record_count": len(assistant_viea_trace),
        "assistant_viea_trace_complete": assistant_viea_trace_complete(assistant_viea_trace),
        "assistant_viea_trace_out": rel(resolve(args.viea_trace_out)),
        "viea_materialized_view_ready": materialized_view_receipt.get("ready"),
        "viea_materialized_view_record_count": materialized_view_receipt.get("record_count"),
        "viea_materialized_view_claim_count": materialized_view_receipt.get("claim_ledger_entry_count"),
        "viea_materialized_view_receipt_id": materialized_view_receipt.get("receipt_id"),
        "route_validator_receipt_ready": route_validator_receipt.get("ready"),
        "route_validator_receipt_id": route_validator_receipt.get("receipt_id"),
        "route_validator_governance_record_count": route_validator_receipt.get("governance_record_count"),
        "route_validator_failure_boundary_count": route_validator_receipt.get("failure_boundary_count"),
        "route_validator_authority_record_count": route_validator_receipt.get("authority_record_count"),
        "route_validator_resource_route_record_count": route_validator_receipt.get("resource_route_record_count"),
        "private_verifier_spine_ready": private_verifier_receipt.get("ready"),
        "private_verifier_spine_state": private_verifier_receipt.get("trigger_state"),
        "private_verifier_spine_record_count": private_verifier_receipt.get("viea_verifier_record_count"),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "assistant_trace_schema": trace_schema.get("path"),
        "assistant_trace_schema_policy": trace_schema.get("policy"),
        "assistant_trace_schema_version": trace_schema.get("schema_version"),
        "assistant_trace_schema_sha256": trace_schema.get("sha256"),
        "assistant_trace_schema_ready": assistant_trace_schema_ready(trace_schema),
        "assistant_trace_allowed_outcomes": sorted(value for value in allowed_feedback if value),
        "effect_canary_enabled": effect_canary.get("enabled"),
        "effect_canary_ready": effect_canary.get("ready"),
        "effect_canary_transaction_id": effect_canary.get("transaction_id"),
        "effect_canary_first_effect_identity": get_path(effect_canary, ["observation", "identity"], ""),
        "effect_canary_final_effect_identity": get_path(effect_canary, ["rollback", "final_identity"], ""),
        "effect_canary_rollback_complete": get_path(effect_canary, ["rollback", "complete"], False),
    }
    return {
        "policy": "project_theseus_assistant_runtime_v0",
        "created_utc": created_utc,
        "trigger_state": trigger_state,
        "summary": summary,
        "inputs": {
            "config": rel(resolve(args.config)),
            "prompt_sha256": sha256_text(prompt),
            "surface": str(args.surface or "local_assistant"),
            "principal": str(getattr(args, "principal", "local-user") or "local-user"),
            "origin": str(getattr(args, "origin", "local_user_control") or "local_user_control"),
        },
        "outputs": {
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
            "conversation_events": rel(conversation_events_path(args.events_out)),
            "assistant_viea_trace": rel(resolve(args.viea_trace_out)),
        },
        "response": response,
        "assistant_text": assistant_text,
        "reflexive_dispatch": reflexive_trace,
        "reflexive_dispatch_verification": reflexive_verification,
        "checkpoint_chat": {
            "returncode": chat_result.get("returncode"),
            "out": rel(checkpoint_out),
            "stderr_tail": chat_result.get("stderr_tail"),
            "session": checkpoint_session,
        },
        "vcm_context_packet": compact_vcm_context(selected_context),
        "vcm_context_governor": vcm_governor,
        "vcm_consumer_abi": vcm_consumer_packet,
        "code_route": code_route,
        "code_private_probe": code_private_probe,
        "tool_context": tool_context,
        "tool_evidence": tool_evidence,
        "plan_context": plan_context,
        "procedural_default_route": procedural_default_route,
        "effect_canary": effect_canary,
        "teacher_policy": teacher_policy,
        "benchmark_status": benchmark_status,
        "assistant_trace_schema": trace_schema,
        "viea_materialized_view_receipt": materialized_view_receipt,
        "route_validator_receipt": route_validator_receipt,
        "private_verifier_receipt": private_verifier_receipt,
        "dogfood": dogfood,
        "assistant_viea_trace": assistant_viea_trace,
        "context_refresh": context_refresh,
        "gates": gates,
        "public_benchmark_boundary": {
            "benchmarks_may_be_run_for_measurement": True,
            "train_on_public_prompts_tests_solutions_traces_or_scores": False,
            "tool_assisted_scores_reported_separately": True,
            "model_only_scores_reported_separately": True,
            "public_measurement_run_registry_required": True,
            "duplicate_measurements_must_be_labeled": True,
        },
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def refresh_context(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if config.get("refresh_context_by_default") is False:
        return rows
    for item in config.get("context_refresh_commands", []) if isinstance(config.get("context_refresh_commands"), list) else []:
        if not isinstance(item, dict):
            continue
        command = [str(part) for part in item.get("command", []) if str(part)]
        if not command:
            continue
        rows.append(run_command(str(item.get("id") or command[-1]), command, int(item.get("timeout_seconds") or 120)))
    return rows


def assistant_materialized_view_receipt() -> dict[str, Any]:
    return viea_spine_records.materialized_view_consumer_receipt(
        "theseus_assistant_runtime_product_trace",
        required_groups=ASSISTANT_PRODUCT_VIEW_GROUPS,
    )


def assistant_route_validator_receipt() -> dict[str, Any]:
    return viea_spine_records.materialized_view_consumer_receipt(
        "theseus_assistant_runtime_route_validator",
        required_groups=ROUTE_VALIDATOR_VIEW_GROUPS,
    )


def private_verifier_receipt_packet() -> dict[str, Any]:
    path = REPORTS / "private_verifier_spine_smoke.json"
    payload = read_json(path, {})
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    verification = payload.get("private_verification") if isinstance(payload.get("private_verification"), dict) else {}
    records = verification.get("viea_verifier_records") if isinstance(verification.get("viea_verifier_records"), dict) else {}
    counter_fault = any(
        int_or_zero(payload.get(key)) != 0 or int_or_zero(verification.get(key)) != 0
        for key in viea_spine_records.NO_CHEAT_COUNTERS
    )
    required_records = {
        "claim_record",
        "proof_carrying_claim",
        "authority_transition",
        "authority_use_receipt",
        "runtime_adapter_invocation",
        "resource_budget",
        "generation_mode",
        "failure_boundary",
        "artifact_graph_record",
        "evidence_transition_record",
    }
    observed = {
        viea_spine_records.canonical_record_type(row.get("record_type"))
        for row in records.values()
        if isinstance(row, dict)
    }
    missing = sorted(required_records - observed)
    ready = (
        path.exists()
        and payload.get("trigger_state") == "GREEN"
        and int_or_zero(summary.get("candidate_attempt_count")) > 0
        and int_or_zero(summary.get("viea_verifier_record_count")) >= len(required_records)
        and not missing
        and not counter_fault
    )
    return {
        "receipt_id": stable_id("assistant_private_verifier_receipt", rel(path), stable_hash(summary), sorted(observed)),
        "path": rel(path),
        "present": path.exists(),
        "ready": ready,
        "trigger_state": payload.get("trigger_state"),
        "candidate_attempt_count": summary.get("candidate_attempt_count"),
        "runtime_load_rate": summary.get("runtime_load_rate"),
        "intended_behavior_pass_rate": summary.get("intended_behavior_pass_rate"),
        "viea_verifier_record_count": summary.get("viea_verifier_record_count", len(records)),
        "observed_record_types": sorted(observed),
        "missing_record_types": missing,
        "counter_fault": counter_fault,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claim": "Private verifier receipt proves verifier-label plumbing for the assistant trace; it is not generation capability evidence.",
    }


def run_checkpoint_chat(*, prompt: str, session_id: str, checkpoint_id: str, out: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/checkpoint_chat.py",
        "--checkpoint-id",
        checkpoint_id,
        "--session-id",
        session_id,
        "--prompt",
        prompt,
        "--out",
        rel(out),
    ]
    return run_command("checkpoint_chat", command, 180)


def run_dogfood_feedback(
    *,
    feedback: str,
    surface: str,
    lane: str,
    intent_summary: str,
    artifact_refs: list[str],
    error_family: str,
    duration_ms: int,
) -> dict[str, Any]:
    event_out = REPORTS / "theseus_assistant_dogfood_event.json"
    event_md = REPORTS / "theseus_assistant_dogfood_event.md"
    event_cmd = [
        sys.executable,
        "scripts/dogfood_trace_event.py",
        "--execute",
        "--surface",
        surface,
        "--assistant-lane",
        lane,
        "--outcome",
        feedback,
        "--intent-summary-redacted",
        intent_summary,
        "--duration-ms",
        str(max(0, duration_ms)),
        "--out",
        rel(event_out),
        "--markdown-out",
        rel(event_md),
    ]
    for artifact in artifact_refs:
        event_cmd.extend(["--artifact-ref", artifact])
    if error_family:
        event_cmd.extend(["--error-family", error_family])
    event_run = run_command("dogfood_trace_event", event_cmd, 60)
    bridge_out = REPORTS / "theseus_assistant_dogfood_training_bridge.json"
    bridge_md = REPORTS / "theseus_assistant_dogfood_training_bridge.md"
    bridge_cmd = [
        sys.executable,
        "scripts/dogfood_trace_training_bridge.py",
        "--execute",
        "--compact-existing",
        "--out",
        rel(bridge_out),
        "--markdown-out",
        rel(bridge_md),
    ]
    bridge_run = run_command("dogfood_trace_training_bridge", bridge_cmd, 120)
    return {
        "event_run": event_run,
        "event": read_json(event_out, {}),
        "training_bridge_run": bridge_run,
        "training_bridge": read_json(bridge_out, {}),
    }


def classify_intent(prompt: str, requested: str) -> str:
    if requested != "auto":
        return requested
    text = prompt.lower()
    if has_any(text, ["implement", "code", "function", "class", "rust", "python", "bug", "stack trace", "compile", "test failure", "verifier"]):
        return "code"
    if has_any(text, ["plan", "goal", "dag", "schedule", "next steps", "orchestrate"]):
        return "planning"
    if has_any(text, ["solve", "equation", "calculate", "search", "retrieve", "tool", "sympy", "z3", "lean"]):
        return "tool"
    return "chat"


def build_reflexive_dispatch_trace(
    *,
    prompt: str,
    requested_intent: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    materialized_view_receipt: dict[str, Any],
    route_validator_receipt: dict[str, Any],
    private_verifier_receipt: dict[str, Any],
) -> dict[str, Any]:
    contract_path = resolve(str(config.get("reflexive_router_contract") or "configs/reflexive_router_contract.json"))
    try:
        contract = reflexive_dispatch.load_contract(contract_path)
        capabilities = contract.get("capabilities") if isinstance(contract.get("capabilities"), list) else []
        tool_registry = read_json(REPORTS / "deterministic_tool_registry.json", {})
        plan_report = read_json(REPORTS / "theseus_plan_compiler.json", {})
        tool_ready = (
            tool_registry.get("trigger_state") in {"GREEN", "YELLOW"}
            and isinstance(tool_registry.get("tools"), list)
            and bool(tool_registry.get("tools"))
        )
        plan_ready = (
            plan_report.get("trigger_state") in {"GREEN", "YELLOW"}
            and int_or_zero(get_path(plan_report, ["summary", "compiled_goal_count"], 0)) > 0
        )
        common_route_ready = bool(route_validator_receipt.get("ready")) and bool(materialized_view_receipt.get("ready"))
        route_health = {
            str(row.get("capability_id") or ""): common_route_ready
            and (
                str(row.get("capability_id") or "") != "assistant.code_candidate"
                or bool(private_verifier_receipt.get("ready"))
            )
            and (
                str(row.get("capability_id") or "") != "assistant.deterministic_tool"
                or tool_ready
            )
            and (
                str(row.get("capability_id") or "") != "assistant.plan_dag"
                or plan_ready
            )
            for row in capabilities
            if isinstance(row, dict) and row.get("capability_id")
        }
        event = reflexive_dispatch.canonical_event(
            payload=prompt,
            principal=str(getattr(args, "principal", "local-user") or "local-user"),
            authenticated=not bool(getattr(args, "unauthenticated", False)),
            origin=str(getattr(args, "origin", "local_user_control") or "local_user_control"),
            authority_refs=[str(value) for value in list_value(config.get("reflexive_router_authority_refs")) if value],
            context_handles=unique_strings(
                [
                    str(materialized_view_receipt.get("receipt_id") or ""),
                    str(route_validator_receipt.get("receipt_id") or ""),
                    str(private_verifier_receipt.get("receipt_id") or ""),
                ]
            ),
            deadline_ms=int(get_path(contract, ["resource_limits", "default_deadline_ms"], 30000) or 30000),
        )
        requested_route = str(getattr(args, "requested_route", "") or "")
        if bool(getattr(args, "effect_canary", False)) and not requested_route:
            requested_route = "assistant.route_authority_effect"
        return reflexive_dispatch.dispatch(
            event,
            intent=requested_intent,
            profile=str(config.get("reflexive_router_profile") or "local_private_assistant"),
            effort_profile=str(getattr(args, "effort", "balanced") or "balanced"),
            requested_route=requested_route,
            fallback_policy=str(getattr(args, "fallback_policy", "no_fallback") or "no_fallback"),
            route_health=route_health,
            contract=contract,
        )
    except reflexive_dispatch.ReflexiveDispatchFault as exc:
        return {
            "policy": "project_theseus_reflexive_dispatch_trace_v1",
            "source_contract": "project_theseus_reflexive_router_contract_v1",
            "trace_id": "",
            "decision_digest": "",
            "selection": {"selected_proposal_ids": [], "terminal_outcome": "rejected", "fallback_used": False},
            "result": {"terminal_outcome": "rejected", "effect_authority_granted": False},
            "effect": {"state": "blocked", "effect_authority_granted": False},
            "fault": exc.record(),
            "no_cheat": {
                "learned_generation_credit": 0,
                "fallback_return_count": 0,
                "external_inference_calls": 0,
                "public_training_rows_written": 0,
            },
        }


def verify_reflexive_dispatch_trace(trace: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if trace.get("fault"):
        return {"state": "REJECTED", "fault": trace.get("fault"), "effect_authority_granted": False}
    try:
        contract_path = resolve(str(config.get("reflexive_router_contract") or "configs/reflexive_router_contract.json"))
        return reflexive_dispatch.verify_trace(trace, reflexive_dispatch.load_contract(contract_path))
    except reflexive_dispatch.ReflexiveDispatchFault as exc:
        return {"state": "REJECTED", "fault": exc.record(), "effect_authority_granted": False}


def selected_reflexive_capabilities(trace: dict[str, Any]) -> list[str]:
    selection = trace.get("selection") if isinstance(trace.get("selection"), dict) else {}
    selected = set(str(value) for value in list_value(selection.get("selected_proposal_ids")))
    capabilities = set()
    for row in list_value(trace.get("proposals")):
        if not isinstance(row, dict) or str(row.get("proposal_id") or "") not in selected:
            continue
        capabilities.update(reflexive_dispatch.proposal_capability_ids(row))
    return sorted(capabilities)


def effective_intent_from_dispatch(trace: dict[str, Any], requested_intent: str) -> str:
    capability_to_intent = {
        "assistant.chat_checkpoint": "chat",
        "assistant.code_candidate": "code",
        "assistant.deterministic_tool": "tool",
        "assistant.plan_dag": "planning",
        "assistant.route_authority_effect": "chat",
    }
    selected = selected_reflexive_capabilities(trace)
    if "assistant.plan_dag" in selected:
        return "planning"
    return capability_to_intent.get(selected[0], requested_intent) if len(selected) == 1 else requested_intent


def reflexive_terminal_outcome(trace: dict[str, Any]) -> str:
    return str(get_path(trace, ["selection", "terminal_outcome"], "rejected") or "rejected")


def reflexive_dispatch_prepared(trace: dict[str, Any], verification: dict[str, Any]) -> bool:
    return (
        verification.get("state") == "VERIFIED"
        and reflexive_terminal_outcome(trace) == "prepared"
        and len(selected_reflexive_capabilities(trace)) >= 1
        and get_path(trace, ["effect", "effect_authority_granted"], False) is False
    )


def reflexive_terminal_text(trace: dict[str, Any]) -> str:
    outcome = reflexive_terminal_outcome(trace).upper()
    qualification_failures = sorted(
        {
            str(failure)
            for row in list_value(trace.get("qualification"))
            if isinstance(row, dict)
            for failure in list_value(row.get("failures"))
            if failure
        }
    )
    fault = get_path(trace, ["fault", "fault_type"], "")
    reasons = qualification_failures or ([str(fault)] if fault else [])
    suffix = f" Reasons: {', '.join(reasons)}." if reasons else ""
    return f"{outcome}: no qualified assistant route was executed.{suffix}"


def route_for(config: dict[str, Any], intent: str) -> dict[str, Any]:
    routes = config.get("intent_routes") if isinstance(config.get("intent_routes"), dict) else {}
    route = routes.get(intent) if isinstance(routes.get(intent), dict) else {}
    if route:
        return route
    return {"assistant_lane": "chat_checkpoint", "vcm_task_family": "operator_chat"}


def select_vcm_context(contexts: dict[str, Any], family_id: str) -> dict[str, Any]:
    for row in contexts.get("task_contexts", []) if isinstance(contexts.get("task_contexts"), list) else []:
        if isinstance(row, dict) and str(row.get("task_family_id") or "") == family_id:
            return row
    return {"task_family_id": family_id, "ready": False, "selected_pages": [], "blockers": ["vcm_task_family_missing"]}


def compact_vcm_context(context: dict[str, Any]) -> dict[str, Any]:
    pages = context.get("selected_pages") if isinstance(context.get("selected_pages"), list) else []
    return {
        "task_family_id": context.get("task_family_id"),
        "label": context.get("label"),
        "ready": context.get("ready"),
        "selected_hash": context.get("selected_hash"),
        "selected_page_count": len(pages),
        "pages": [
            {
                "address": page.get("address"),
                "title": page.get("title"),
                "score": page.get("score"),
                "source_path": page.get("source_path"),
            }
            for page in pages[:8]
            if isinstance(page, dict)
        ],
        "blockers": context.get("blockers") if isinstance(context.get("blockers"), list) else [],
    }


def vcm_context_governor_packet() -> dict[str, Any]:
    report_path = REPORTS / "vcm_context_governor.json"
    mission_path = REPORTS / "vcm_mission_brief.json"
    closure_path = REPORTS / "vcm_deletion_closure.json"
    report = read_json(report_path, {})
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    mission = read_json(mission_path, {})
    closure = read_json(closure_path, {})
    boundary_gates = report.get("boundary_gates") if isinstance(report.get("boundary_gates"), list) else []
    failed_boundary_gates = [
        str(row.get("name") or row.get("id") or row.get("kind") or "")
        for row in boundary_gates
        if isinstance(row, dict) and row.get("passed") is not True
    ]
    ready = bool(
        report.get("trigger_state") == "GREEN"
        and int_or_zero(summary.get("hard_gap_count")) == 0
        and summary.get("mission_brief_status") == "ready"
        and summary.get("deletion_closure_status") == "closed"
        and summary.get("scif_status") == "ready"
        and not failed_boundary_gates
    )
    return {
        "report": rel(report_path),
        "mission_brief": rel(mission_path),
        "deletion_closure": rel(closure_path),
        "trigger_state": report.get("trigger_state"),
        "ready": ready,
        "adequacy_state": "governed_sufficient" if ready else "fail_closed_or_partial",
        "summary": {
            "chunk_count": summary.get("chunk_count"),
            "pinned_chunk_count": summary.get("pinned_chunk_count"),
            "fail_closed_chunk_count": summary.get("fail_closed_chunk_count"),
            "mission_brief_status": summary.get("mission_brief_status"),
            "mission_brief_omission_count": summary.get("mission_brief_omission_count"),
            "mission_brief_compression_loss": summary.get("mission_brief_compression_loss"),
            "scif_status": summary.get("scif_status"),
            "deletion_closure_status": summary.get("deletion_closure_status"),
            "deletion_closure_fault_count": summary.get("deletion_closure_fault_count"),
            "hard_gap_count": summary.get("hard_gap_count"),
            "warning_count": summary.get("warning_count"),
        },
        "mission": {
            "id": mission.get("id"),
            "status": mission.get("status"),
            "selected_chunk_ids": mission.get("selected_chunk_ids") if isinstance(mission.get("selected_chunk_ids"), list) else [],
            "missing_required_chunk_ids": mission.get("missing_required_chunk_ids") if isinstance(mission.get("missing_required_chunk_ids"), list) else [],
            "omission_count": len(mission.get("omissions") or []) if isinstance(mission.get("omissions"), list) else 0,
            "authority_limits": mission.get("authority_limits") if isinstance(mission.get("authority_limits"), list) else [],
            "compression_loss": mission.get("compression_loss"),
        },
        "deletion": {
            "id": closure.get("id"),
            "status": closure.get("status"),
            "revoked_material": closure.get("revoked_material"),
            "descendant_count": closure.get("descendant_count"),
            "closure_fault_count": closure.get("closure_fault_count"),
        },
        "failed_boundary_gates": [item for item in failed_boundary_gates if item],
        "no_cheat": {
            "public_training_rows_written": 0,
            "runtime_external_inference_calls": 0,
            "fallback_return_count": 0,
        },
        "rules": {
            "larger_context_not_substitute_for_verified_context": True,
            "taint_and_omissions_survive_compression": True,
            "deleted_material_must_close_descendants": True,
            "runtime_external_inference_forbidden": True,
        },
    }


def code_route_packet(intent: str, route: dict[str, Any]) -> dict[str, Any]:
    if intent != "code":
        return {"active": False}
    decision = read_json(REPORTS / "broad_capability_survival_lane_decision_v1.json", {})
    promotion = read_json(REPORTS / "broad_capability_survival_promotion_gate_v1.json", {})
    return {
        "active": True,
        "practical_generator": route.get("practical_generator"),
        "generator_entrypoint": route.get("generator_entrypoint"),
        "verifier": route.get("verifier"),
        "symliquid_role": route.get("symliquid_role"),
        "decision_state": decision.get("trigger_state"),
        "promotion_state": promotion.get("trigger_state"),
        "no_fallback_returns": True,
        "public_benchmark_training_allowed": False,
        "score_reporting": "model_only_and_tool_assisted_must_be_separate",
    }


def code_private_probe_safe(probe: dict[str, Any]) -> bool:
    summary = probe.get("summary") if isinstance(probe.get("summary"), dict) else {}
    rules = probe.get("rules") if isinstance(probe.get("rules"), dict) else {}
    return (
        probe.get("active") is True
        and probe.get("trigger_state") in {"GREEN", "YELLOW"}
        and int_or_zero(summary.get("candidate_row_count")) > 0
        and int_or_zero(summary.get("tasks_with_manifest_candidates")) > 0
        and int_or_zero(summary.get("public_boundary_violation_count")) == 0
        and int_or_zero(summary.get("fallback_return_candidate_count")) == 0
        and int_or_zero(summary.get("unconditional_constant_return_candidate_count")) == 0
        and int_or_zero(rules.get("public_training_rows_written")) == 0
        and int_or_zero(rules.get("external_inference_calls")) == 0
        and rules.get("public_calibration_run") is not True
    )


def code_private_probe_wall(probe: dict[str, Any]) -> dict[str, Any]:
    summary = probe.get("summary") if isinstance(probe.get("summary"), dict) else {}
    selected_pass = float_or_zero(summary.get("selected_intended_behavior_pass_rate"))
    pass_if_any = float_or_zero(summary.get("pass_if_any_rate"))
    return {
        "safe_boundary": code_private_probe_safe(probe),
        "trigger_state": probe.get("trigger_state"),
        "task_count": int_or_zero(summary.get("task_count")),
        "candidate_row_count": int_or_zero(summary.get("candidate_row_count")),
        "tasks_with_manifest_candidates": int_or_zero(summary.get("tasks_with_manifest_candidates")),
        "eligible_candidate_count": int_or_zero(summary.get("eligible_candidate_count")),
        "candidate_integrity_mismatch_count": int_or_zero(summary.get("candidate_integrity_mismatch_count")),
        "candidate_integrity_mismatch_counts": summary.get("candidate_integrity_mismatch_counts")
        if isinstance(summary.get("candidate_integrity_mismatch_counts"), dict)
        else {},
        "selected_intended_behavior_pass_rate": selected_pass,
        "pass_if_any_rate": pass_if_any,
        "semantic_pass_currently_zero": selected_pass == 0.0 and pass_if_any == 0.0,
        "public_boundary_violation_count": int_or_zero(summary.get("public_boundary_violation_count")),
        "fallback_return_candidate_count": int_or_zero(summary.get("fallback_return_candidate_count")),
        "unconditional_constant_return_candidate_count": int_or_zero(summary.get("unconditional_constant_return_candidate_count")),
    }


def run_code_private_probe(intent: str, route: dict[str, Any]) -> dict[str, Any]:
    if intent != "code":
        return {"active": False}
    probe = route.get("private_probe") if isinstance(route.get("private_probe"), dict) else {}
    if probe.get("enabled") is False:
        return {"active": False, "reason": "disabled"}
    command = [str(part) for part in probe.get("command", []) if str(part)]
    if not command:
        return {"active": False, "reason": "no_private_probe_command"}
    command_result = run_command(
        str(probe.get("id") or "assistant_code_private_probe"),
        command,
        int(probe.get("timeout_seconds") or 180),
    )
    out = str(probe.get("out") or command_arg(command, "--out") or "")
    markdown = str(probe.get("markdown") or command_arg(command, "--markdown-out") or "")
    report = read_json(resolve(out), {}) if out else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "active": True,
        "trigger_state": report.get("trigger_state") or ("RED" if command_result.get("returncode") not in {0, None} else ""),
        "summary": {
            "task_count": summary.get("task_count", 0),
            "candidate_row_count": summary.get("candidate_row_count", 0),
            "candidate_integrity_mismatch_count": summary.get("candidate_integrity_mismatch_count", 0),
            "candidate_integrity_mismatch_counts": summary.get("candidate_integrity_mismatch_counts", {}),
            "eligible_candidate_count": summary.get("eligible_candidate_count", 0),
            "replayed_candidate_count": summary.get("replayed_candidate_count", 0),
            "tasks_with_manifest_candidates": summary.get("tasks_with_manifest_candidates", 0),
            "unexplained_no_candidate_count": summary.get("unexplained_no_candidate_count", 0),
            "selected_compile_pass_rate": summary.get("selected_compile_pass_rate"),
            "selected_runtime_load_rate": summary.get("selected_runtime_load_rate"),
            "selected_intended_behavior_pass_rate": summary.get("selected_intended_behavior_pass_rate"),
            "pass_if_any_rate": summary.get("pass_if_any_rate"),
            "fallback_return_candidate_count": summary.get("fallback_return_candidate_count", 0),
            "unconditional_constant_return_candidate_count": summary.get("unconditional_constant_return_candidate_count", 0),
            "public_boundary_violation_count": summary.get("public_boundary_violation_count", 0),
            "runtime_ms": summary.get("runtime_ms"),
        },
        "report": rel(resolve(out)) if out else "",
        "markdown": rel(resolve(markdown)) if markdown else "",
        "command_result": command_result,
        "rules": report.get("rules") if isinstance(report.get("rules"), dict) else {},
    }


def compose_assistant_text(
    *,
    intent: str,
    base_text: str,
    code_route: dict[str, Any],
    code_private_probe: dict[str, Any],
    tool_context: dict[str, Any],
    tool_evidence: dict[str, Any],
    plan_context: dict[str, Any],
    procedural_default_route: dict[str, Any],
    teacher_policy: dict[str, Any],
    benchmark_status: dict[str, Any],
    vcm_governor: dict[str, Any],
    selected_context: dict[str, Any],
    checkpoint_session: dict[str, Any],
) -> str:
    cleaned_base_text = sanitize_checkpoint_answer(
        base_text,
        drop_benchmark_status=bool(benchmark_status.get("active")),
    )
    lines = [cleaned_base_text] if cleaned_base_text else []
    family = str(selected_context.get("label") or selected_context.get("task_family_id") or "")
    history_turns = int_or_zero(checkpoint_session.get("history_turns_loaded"))
    if checkpoint_session.get("session_id"):
        lines.append(f"Session memory: {history_turns} previous turn(s) loaded for `{checkpoint_session.get('session_id')}`.")
    if intent == "code" and code_route.get("active"):
        lines.extend(
            [
                "",
                "Code path: I classified this as a coding task and attached the VCM code-training context.",
                (
                    "Practical lane: "
                    f"{code_route.get('practical_generator')} via {code_route.get('generator_entrypoint')}, "
                    f"checked by {code_route.get('verifier')}."
                ),
                (
                    "Architecture posture: transformer/hybrid structural full-body generation is the practical route; "
                    "SymLiquid stays as a matched-compute comparator until it wins repeat evidence."
                ),
                "Evidence rule: no fallback returns and no public benchmark payloads enter training.",
            ]
        )
        if code_private_probe.get("active"):
            probe_summary = code_private_probe.get("summary") if isinstance(code_private_probe.get("summary"), dict) else {}
            lines.append(
                "Private verifier probe: "
                f"{code_private_probe.get('trigger_state')} "
                f"selected_pass={probe_summary.get('selected_intended_behavior_pass_rate')} "
                f"pass_if_any={probe_summary.get('pass_if_any_rate')} "
                f"tasks={probe_summary.get('task_count')}."
            )
            wall = code_private_probe_wall(code_private_probe)
            if wall.get("semantic_pass_currently_zero") or wall.get("candidate_integrity_mismatch_count"):
                lines.append(
                    "Current generator wall: verifier replay is safe but not capable yet; "
                    f"eligible_candidates={wall.get('eligible_candidate_count')} "
                    f"integrity_mismatches={wall.get('candidate_integrity_mismatch_count')} "
                    f"selected_pass={wall.get('selected_intended_behavior_pass_rate')}. "
                    "Do not trust generated code unless an explicit verifier/tool pass backs it."
                )
    elif intent == "tool":
        evidence_summary = tool_evidence.get("summary") if isinstance(tool_evidence.get("summary"), dict) else {}
        lines.extend(
            [
                "",
                f"Tool path: deterministic tool registry is {tool_context.get('registry_state')} with {tool_context.get('tool_count')} tools.",
                "The model proposes and routes; exact local tools compute/check and emit evidence.",
            ]
        )
        if tool_evidence.get("active"):
            lines.append(
                "Tool evidence: "
                f"{tool_evidence.get('trigger_state')} "
                f"results={evidence_summary.get('result_count')} "
                f"exact_solve={evidence_summary.get('exact_solve_rate')} "
                f"tool_on={evidence_summary.get('tool_on_solve_rate')} "
                f"trace={tool_evidence.get('trace')}."
            )
    elif intent == "planning":
        lines.extend(
            [
                "",
                f"Planning path: plan compiler is {plan_context.get('planner_state')} with {plan_context.get('compiled_goal_count')} compiled goals.",
                "The assistant should compile work into VCM-backed DAGs, then route to existing registered executors.",
            ]
        )
        if tool_evidence.get("required"):
            evidence_summary = tool_evidence.get("summary") if isinstance(tool_evidence.get("summary"), dict) else {}
            lines.append(
                "Deterministic evidence step: "
                f"{tool_evidence.get('trigger_state')} results={evidence_summary.get('result_count')} "
                f"verified={evidence_summary.get('verified_solved_count')} trace={tool_evidence.get('trace')}."
            )
        if procedural_default_route.get("active"):
            selected_route = procedural_default_route.get("selected_route") if isinstance(procedural_default_route.get("selected_route"), dict) else {}
            lines.append(
                "Procedural memory: guarded default route "
                f"`{selected_route.get('id')}` is ready={procedural_default_route.get('ready')} "
                f"candidate=`{selected_route.get('candidate_id')}` guard_armed="
                f"{get_path(selected_route, ['continued_regression_guard', 'armed'], False)}."
            )
            lines.append(
                "Procedural boundary: this is local metadata workflow compression only; it is not learned generation, "
                "not public-transfer evidence, and must roll back if its regression guard fails."
            )
        elif procedural_default_route.get("required"):
            lines.append(
                "Procedural memory: required planning default route is not ready; planning remains explicit and uncompressed."
            )
    if family:
        lines.append(f"VCM context: {family}.")
    if vcm_governor.get("trigger_state"):
        lines.append(
            "VCM adequacy: "
            f"{vcm_governor.get('trigger_state')} "
            f"state={vcm_governor.get('adequacy_state')} "
            f"mission={get_path(vcm_governor, ['summary', 'mission_brief_status'], '')} "
            f"deletion={get_path(vcm_governor, ['summary', 'deletion_closure_status'], '')}."
        )
    if benchmark_status.get("include_in_answer"):
        residual = benchmark_status.get("dominant_residual")
        residual_text = ""
        if isinstance(residual, list) and residual:
            residual_text = f" dominant_residual={residual[0]}:{residual[1] if len(residual) > 1 else ''}"
        lines.append(
            "Public measurement: "
            f"{benchmark_status.get('latest_public_run_id')} "
            f"{benchmark_status.get('latest_public_passed')}/{benchmark_status.get('latest_public_task_count')} "
            f"= {benchmark_status.get('latest_public_pass_rate')} "
            f"({benchmark_status.get('measurement_kind')}; cards={','.join(benchmark_status.get('latest_public_cards') or [])})."
            f"{residual_text}"
        )
        lines.append(
            "Benchmark boundary: measurement-only, public training rows "
            f"{benchmark_status.get('public_training_rows_written')}, external inference "
            f"{benchmark_status.get('external_inference_calls')}, fallback returns "
            f"{benchmark_status.get('fallback_return_count')}."
        )
    if teacher_policy.get("runtime_external_tokens_forbidden") is True:
        lines.append("Teacher boundary: no external teacher tokens are served at runtime.")
    return "\n".join(lines).strip()


def sanitize_checkpoint_answer(base_text: str, *, drop_benchmark_status: bool) -> str:
    lines: list[str] = []
    skip_benchmark_block = False
    for raw_line in str(base_text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if drop_benchmark_status and stripped == "[benchmark_status]":
            skip_benchmark_block = True
            continue
        if skip_benchmark_block:
            if not stripped:
                continue
            if re.match(r"^[a-zA-Z0-9_./ -]+:\s*score=[0-9.]+,\s*residual=[0-9.]+", stripped):
                continue
            skip_benchmark_block = False
        if drop_benchmark_status and re.match(r"^[a-zA-Z0-9_./ -]+:\s*score=[0-9.]+,\s*residual=[0-9.]+", stripped):
            continue
        if drop_benchmark_status and stripped.lower().startswith("benchmark posture:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def tool_context_packet(required: bool) -> dict[str, Any]:
    registry = read_json(REPORTS / "deterministic_tool_registry.json", {})
    return {
        "active": required,
        "required": required,
        "registry_state": registry.get("trigger_state"),
        "tool_count": len(registry.get("tools") or []) if isinstance(registry.get("tools"), list) else 0,
        "strict_no_fallback_returns": registry.get("strict_no_fallback_returns"),
    }


def run_tool_evidence(required: bool, route: dict[str, Any]) -> dict[str, Any]:
    if not required:
        return {"active": False, "required": False}
    evidence = route.get("tool_evidence") if isinstance(route.get("tool_evidence"), dict) else {}
    if evidence.get("enabled") is False:
        return {"active": False, "required": True, "reason": "disabled"}
    command = [str(part) for part in evidence.get("command", []) if str(part)]
    if not command:
        return {"active": False, "required": True, "reason": "no_tool_evidence_command"}
    command_result = run_command(
        str(evidence.get("id") or "assistant_deterministic_tool_evidence"),
        command,
        int(evidence.get("timeout_seconds") or 180),
    )
    out = str(evidence.get("out") or command_arg(command, "--out") or "")
    markdown = str(evidence.get("markdown") or command_arg(command, "--markdown-out") or "")
    trace = str(evidence.get("trace") or command_arg(command, "--trace-out") or "")
    ablation = str(evidence.get("ablation") or command_arg(command, "--ablation-out") or "")
    artifact_graph = str(evidence.get("artifact_graph") or command_arg(command, "--artifact-graph-out") or "")
    report = read_json(resolve(out), {}) if out else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    tool_results = report.get("tool_results") if isinstance(report.get("tool_results"), list) else []
    compact_results = []
    for row in tool_results[:8]:
        if not isinstance(row, dict):
            continue
        compact_results.append(
            {
                "case_id": row.get("case_id"),
                "tool_id": row.get("tool_id"),
                "state": row.get("state"),
                "verified": row.get("verified"),
                "latency_ms": row.get("latency_ms"),
                "evidence_ref": row.get("evidence_ref"),
                "replay_checksum": row.get("replay_checksum"),
            }
        )
    return {
        "active": True,
        "required": True,
        "trigger_state": report.get("trigger_state") or ("RED" if command_result.get("returncode") not in {0, None} else ""),
        "summary": {
            "tool_count": summary.get("tool_count", summary.get("tool_card_count")),
            "result_count": summary.get("result_count", summary.get("private_case_result_count")),
            "solved_count": summary.get("solved_count"),
            "verified_solved_count": summary.get("verified_solved_count"),
            "exact_solve_rate": summary.get("exact_solve_rate"),
            "tool_on_solve_rate": summary.get("tool_on_solve_rate"),
            "abstention_rate": summary.get("abstention_rate"),
            "tool_fault_rate": summary.get("tool_fault_rate"),
            "fallback_return_count": summary.get("fallback_return_count", 0),
            "external_inference_calls": summary.get("external_inference_calls", 0),
            "public_training_rows_written": summary.get("public_training_rows_written", 0),
            "runtime_ms": summary.get("runtime_ms"),
        },
        "report": rel(resolve(out)) if out else "",
        "markdown": rel(resolve(markdown)) if markdown else "",
        "trace": rel(resolve(trace)) if trace else "",
        "ablation": rel(resolve(ablation)) if ablation else "",
        "artifact_graph": rel(resolve(artifact_graph)) if artifact_graph else "",
        "compact_results": compact_results,
        "command_result": command_result,
        "rules": {
            "tool_outputs_are_evidence_not_chat_memory": True,
            "strict_no_fallback_returns": True,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
        },
    }


def build_assistant_viea_trace(
    *,
    created_utc: str,
    prompt_hash: str,
    prompt_summary: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    intent: str,
    reflexive_dispatch_trace: dict[str, Any],
    route: dict[str, Any],
    session_id: str,
    checkpoint_id: str,
    selected_context: dict[str, Any],
    checkpoint_out: Path,
    chat_result: dict[str, Any],
    checkpoint_session: dict[str, Any],
    code_route: dict[str, Any],
    code_private_probe: dict[str, Any],
    tool_context: dict[str, Any],
    tool_evidence: dict[str, Any],
    plan_context: dict[str, Any],
    procedural_default_route: dict[str, Any],
    teacher_policy: dict[str, Any],
    benchmark_status: dict[str, Any],
    materialized_view_receipt: dict[str, Any],
    route_validator_receipt: dict[str, Any],
    private_verifier_receipt: dict[str, Any],
    vcm_governor: dict[str, Any],
    dogfood: dict[str, Any],
    feedback: str,
    assistant_text: str,
    trace_schema: dict[str, Any],
    effect_canary: dict[str, Any],
) -> list[dict[str, Any]]:
    run_id = stable_id("assistant_viea_run", created_utc, session_id, intent, prompt_hash)
    surface = str(args.surface or "local_assistant")
    vcm_packet = compact_vcm_context(selected_context)
    artifact_refs = artifact_refs_for_feedback(args.out, checkpoint_out, code_private_probe, tool_evidence, procedural_default_route)
    if plan_context.get("active"):
        artifact_refs.append("reports/theseus_plan_compiler.json")
        artifact_refs.append("reports/theseus_plan_trace_bundle.jsonl")
    if materialized_view_receipt.get("view_path"):
        artifact_refs.append(str(materialized_view_receipt.get("view_path")))
    if route_validator_receipt.get("view_path"):
        artifact_refs.append(str(route_validator_receipt.get("view_path")))
    if private_verifier_receipt.get("path"):
        artifact_refs.append(str(private_verifier_receipt.get("path")))
    for candidate in [
        vcm_governor.get("report"),
        vcm_governor.get("mission_brief"),
        vcm_governor.get("deletion_closure"),
    ]:
        if candidate:
            artifact_refs.append(str(candidate))
    if dogfood:
        for candidate in [
            get_path(dogfood, ["event", "out"], ""),
            get_path(dogfood, ["training_bridge", "out"], ""),
        ]:
            if candidate:
                artifact_refs.append(str(candidate))
    artifact_refs = unique_strings(artifact_refs)
    context_hash = stable_hash(vcm_packet)
    route_id = stable_id("assistant_route", intent, route.get("assistant_lane"), route.get("vcm_task_family"))
    node_id = stable_id("assistant_node", run_id, intent, route_id)
    trace_base = {
        "policy": "project_theseus_assistant_viea_trace_v1",
        "created_utc": created_utc,
        "assistant_run_id": run_id,
        "session_id": session_id,
        "intent": intent,
        "prompt_sha256": prompt_hash,
        "redacted_intent_summary": prompt_summary,
        "raw_prompt_stored": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }

    records: list[dict[str, Any]] = []

    def add(record_type: str, content: dict[str, Any]) -> str:
        record_id = stable_id(record_type, run_id, content)
        row = dict(trace_base)
        row.update(
            {
                "record_type": record_type,
                "record_id": record_id,
                "content": content,
            }
        )
        records.append(row)
        return record_id

    intent_id = add(
        "intent_contract",
        {
            "surface": surface,
            "requested_intent": intent,
            "prompt_visible_to_model": True,
            "prompt_persisted_as_hash_only": True,
            "permitted_context": ["VCM selected pages", "checkpoint session metadata", "registered tool evidence"],
            "materialized_view_receipt_id": materialized_view_receipt.get("receipt_id"),
            "route_validator_receipt_id": route_validator_receipt.get("receipt_id"),
            "private_verifier_receipt_id": private_verifier_receipt.get("receipt_id"),
            "forbidden_means": [
                "runtime external inference",
                "public benchmark training rows",
                "fallback returns",
                "raw prompt training rows",
            ],
            "required_outcome": "assistant answer plus evidence records; UNKNOWN/UNSUPPORTED when evidence is absent",
        },
    )
    command_id = add(
        "command_contract",
        {
            "intent_contract_id": intent_id,
            "runtime": "scripts/theseus_assistant_runtime.py",
            "config": rel(resolve(args.config)),
            "assistant_trace_schema": trace_schema.get("path"),
            "assistant_trace_schema_sha256": trace_schema.get("sha256"),
            "assistant_trace_schema_version": trace_schema.get("schema_version"),
            "output_report": rel(resolve(args.out)),
            "assistant_lane": route.get("assistant_lane"),
            "vcm_task_family": route.get("vcm_task_family"),
            "reflexive_dispatch": {
                "trace_id": reflexive_dispatch_trace.get("trace_id"),
                "decision_digest": reflexive_dispatch_trace.get("decision_digest"),
                "terminal_outcome": reflexive_terminal_outcome(reflexive_dispatch_trace),
                "selected_capabilities": selected_reflexive_capabilities(reflexive_dispatch_trace),
                "effect_authority_granted": get_path(reflexive_dispatch_trace, ["effect", "effect_authority_granted"], False),
                "no_cheat": reflexive_dispatch_trace.get("no_cheat"),
            },
            "procedural_default_route": {
                "active": procedural_default_route.get("active"),
                "ready": procedural_default_route.get("ready"),
                "report": procedural_default_route.get("report"),
                "route_id": get_path(procedural_default_route, ["selected_route", "id"], ""),
                "candidate_id": get_path(procedural_default_route, ["selected_route", "candidate_id"], ""),
                "guard_armed": get_path(procedural_default_route, ["selected_route", "continued_regression_guard", "armed"], False),
                "learned_generation_claim_allowed": get_path(procedural_default_route, ["selected_route", "learned_generation_claim_allowed"], None),
            },
            "checkpoint_id": checkpoint_id,
            "session_id": session_id,
            "side_effect_classes": ["reports_write", "conversation_event_append", "metadata_only_dogfood_event"],
            "bounded_effect_canary": {
                "enabled": effect_canary.get("enabled"),
                "transaction_id": effect_canary.get("transaction_id"),
                "target": effect_canary.get("target"),
            },
            "non_claims": ["assistant answer is not learned-generation promotion evidence", "tool evidence is not model-only skill"],
        },
    )
    context_id = add(
        "context_abi_record",
        {
            "command_contract_id": command_id,
            "context_hash": context_hash,
            "context_ready": bool(selected_context.get("ready")),
            "selected_page_count": len(selected_context.get("selected_pages") or []),
            "context_packet": vcm_packet,
            "viea_materialized_view": {
                "ready": materialized_view_receipt.get("ready"),
                "record_count": materialized_view_receipt.get("record_count"),
                "claim_ledger_entry_count": materialized_view_receipt.get("claim_ledger_entry_count"),
                "required_groups": materialized_view_receipt.get("required_groups"),
            },
            "route_validator_view": {
                "ready": route_validator_receipt.get("ready"),
                "receipt_id": route_validator_receipt.get("receipt_id"),
                "required_groups": route_validator_receipt.get("required_groups"),
                "governance_record_count": route_validator_receipt.get("governance_record_count"),
                "failure_boundary_count": route_validator_receipt.get("failure_boundary_count"),
                "authority_record_count": route_validator_receipt.get("authority_record_count"),
                "resource_route_record_count": route_validator_receipt.get("resource_route_record_count"),
            },
            "vcm_governor_receipt": {
                "report": vcm_governor.get("report"),
                "ready": vcm_governor.get("ready"),
                "trigger_state": vcm_governor.get("trigger_state"),
                "adequacy_state": vcm_governor.get("adequacy_state"),
                "mission_brief_status": get_path(vcm_governor, ["summary", "mission_brief_status"], None),
                "deletion_closure_status": get_path(vcm_governor, ["summary", "deletion_closure_status"], None),
                "hard_gap_count": get_path(vcm_governor, ["summary", "hard_gap_count"], 0),
            },
            "taint_labels": ["private_runtime_metadata"],
            "public_calibration_artifact_loaded": False,
            "raw_private_text_training_allowed": False,
            "adequacy": vcm_governor.get("adequacy_state") if vcm_governor.get("ready") else "partial",
        },
    )
    context_transaction_id = add(
        "context_transaction",
        {
            "command_contract_id": command_id,
            "context_abi_record_id": context_id,
            "transaction_id": stable_id("vcm_context_transaction", run_id, context_hash, vcm_governor.get("report")),
            "snapshot_id": vcm_packet.get("selected_hash") or context_hash,
            "mounts": [
                {
                    "mount_id": str(vcm_packet.get("task_family_id") or route.get("vcm_task_family") or "operator_chat"),
                    "label": vcm_packet.get("label"),
                    "page_count": vcm_packet.get("selected_page_count"),
                }
            ],
            "read_set": [
                {
                    "address": page.get("address"),
                    "source_path": page.get("source_path"),
                    "title": page.get("title"),
                }
                for page in vcm_packet.get("pages", [])
                if isinstance(page, dict)
            ],
            "write_set": artifact_refs,
            "taint_labels": ["private_runtime_metadata"],
            "deletion_obligations": {
                "closure_report": vcm_governor.get("deletion_closure"),
                "closure_status": get_path(vcm_governor, ["summary", "deletion_closure_status"], None),
                "closure_fault_count": get_path(vcm_governor, ["summary", "deletion_closure_fault_count"], 0),
            },
            "public_calibration_artifact_loaded": False,
            "raw_private_text_training_allowed": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
    )
    context_adequacy_id = add(
        "context_adequacy",
        {
            "command_contract_id": command_id,
            "context_abi_record_id": context_id,
            "context_transaction_id": context_transaction_id,
            "governor_report": vcm_governor.get("report"),
            "mission_brief": vcm_governor.get("mission_brief"),
            "deletion_closure": vcm_governor.get("deletion_closure"),
            "context_ready": bool(selected_context.get("ready")),
            "selected_page_count": len(selected_context.get("selected_pages") or []),
            "governor_ready": vcm_governor.get("ready"),
            "governor_trigger_state": vcm_governor.get("trigger_state"),
            "adequacy_state": vcm_governor.get("adequacy_state"),
            "mission_brief_status": get_path(vcm_governor, ["summary", "mission_brief_status"], None),
            "mission_brief_omission_count": get_path(vcm_governor, ["summary", "mission_brief_omission_count"], None),
            "mission_brief_compression_loss": get_path(vcm_governor, ["summary", "mission_brief_compression_loss"], None),
            "deletion_closure_status": get_path(vcm_governor, ["summary", "deletion_closure_status"], None),
            "scif_status": get_path(vcm_governor, ["summary", "scif_status"], None),
            "failed_boundary_gates": vcm_governor.get("failed_boundary_gates") if isinstance(vcm_governor.get("failed_boundary_gates"), list) else [],
            "fail_closed": vcm_governor.get("ready") is not True or not selected_context.get("ready"),
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
    )
    job_id = add(
        "typed_job",
        {
            "command_contract_id": command_id,
            "node_id": node_id,
            "job_type": f"assistant_{intent}",
            "lifecycle_state": "completed_with_report" if chat_result.get("returncode") == 0 else "tool_fault",
            "registered_surface": "theseus_assistant_runtime",
            "permissions": ["read_context_reports", "write_assistant_report", "append_local_metadata_event"],
            "approval_state": "local_operator_policy",
            "replay_status": "hash_replayable_without_raw_prompt",
            "failure_behavior": "return structured gate failure; no fallback answer credit",
        },
    )
    plan_id = add(
        "planforge_dag",
        {
            "command_contract_id": command_id,
            "dag_id": stable_id("assistant_plan_dag", run_id, node_id),
            "nodes": assistant_plan_nodes(intent, node_id, code_private_probe, tool_evidence, plan_context, procedural_default_route),
            "edges": [["context_read", "checkpoint_chat"], ["checkpoint_chat", "answer_compose"], ["answer_compose", "trace_emit"]],
            "route_id": route_id,
            "procedural_default_route_id": get_path(procedural_default_route, ["selected_route", "id"], ""),
            "procedural_default_route_ready": procedural_default_route.get("ready"),
            "route_validator_receipt_id": route_validator_receipt.get("receipt_id"),
            "route_validator_ready": route_validator_receipt.get("ready"),
            "vcm_context_record_id": context_id,
            "vcm_context_transaction_id": context_transaction_id,
            "context_adequacy_receipt_id": context_adequacy_id,
            "typed_job_id": job_id,
            "registered_route_only": True,
        },
    )
    adapter_id = add(
        "runtime_adapter_invocation",
        {
            "typed_job_id": job_id,
            "adapter": "checkpoint_chat_and_registered_assistant_subtools",
            "commands": [
                "python3 scripts/checkpoint_chat.py --prompt <redacted_prompt_sha256> --out " + rel(checkpoint_out),
                "registered_context_refresh_commands" if not args.skip_context_refresh else "context_refresh_skipped",
            ],
            "sandbox": "local_process",
            "effect_receipts": artifact_refs,
            "materialized_view_receipt": {
                "receipt_id": materialized_view_receipt.get("receipt_id"),
                "ready": materialized_view_receipt.get("ready"),
                "record_count": materialized_view_receipt.get("record_count"),
            },
            "route_validator_receipt": {
                "receipt_id": route_validator_receipt.get("receipt_id"),
                "ready": route_validator_receipt.get("ready"),
                "missing_required_groups": route_validator_receipt.get("missing_required_groups"),
            },
            "private_verifier_receipt": {
                "receipt_id": private_verifier_receipt.get("receipt_id"),
                "ready": private_verifier_receipt.get("ready"),
                "trigger_state": private_verifier_receipt.get("trigger_state"),
                "record_count": private_verifier_receipt.get("viea_verifier_record_count"),
            },
            "rollback_handle": "delete generated report/event rows if operator requests; model state unchanged",
            "returncode": chat_result.get("returncode"),
            "stderr_tail_present": bool(chat_result.get("stderr_tail")),
        },
    )
    if effect_canary.get("enabled"):
        effect_inventory_id = add(
            "effect_inventory",
            {
                "command_contract_id": command_id,
                "typed_job_id": job_id,
                "transaction_id": effect_canary.get("transaction_id"),
                "proposer_id": effect_canary.get("proposer_id"),
                "declared_effects": effect_canary.get("effect_inventory"),
                "undeclared_effects_permitted": False,
                "network_effects_permitted": False,
                "training_effects_permitted": False,
            },
        )
        effect_observation_id = add(
            "effect_observation_record",
            {
                "effect_inventory_record_id": effect_inventory_id,
                "transaction_id": effect_canary.get("transaction_id"),
                "observer_id": effect_canary.get("observer_id"),
                "observer_independent_from_proposer": effect_canary.get("observer_id") != effect_canary.get("proposer_id"),
                "observation": effect_canary.get("observation"),
            },
        )
        add(
            "rollback_completeness_record",
            {
                "effect_inventory_record_id": effect_inventory_id,
                "effect_observation_record_id": effect_observation_id,
                "transaction_id": effect_canary.get("transaction_id"),
                "evaluator_id": effect_canary.get("evaluator_id"),
                "evaluator_independent_from_proposer_and_observer": len(
                    {
                        str(effect_canary.get("proposer_id")),
                        str(effect_canary.get("observer_id")),
                        str(effect_canary.get("evaluator_id")),
                    }
                )
                == 3,
                "rollback": effect_canary.get("rollback"),
                "residuals": effect_canary.get("residuals"),
                "ready": effect_canary.get("ready"),
            },
        )
    if procedural_default_route.get("active"):
        add(
            "procedural_tool_record",
            {
                "command_contract_id": command_id,
                "typed_job_id": job_id,
                "route_id": get_path(procedural_default_route, ["selected_route", "id"], ""),
                "candidate_id": get_path(procedural_default_route, ["selected_route", "candidate_id"], ""),
                "source_canary_route_id": get_path(procedural_default_route, ["selected_route", "source_canary_route_id"], ""),
                "route_scope": get_path(procedural_default_route, ["selected_route", "route_scope"], ""),
                "route_binding_contract": get_path(procedural_default_route, ["selected_route", "route_binding_contract"], {}),
                "selection": procedural_default_route.get("selection"),
                "report": procedural_default_route.get("report"),
                "ready": procedural_default_route.get("ready"),
                "trigger_state": procedural_default_route.get("trigger_state"),
                "guard_id": get_path(procedural_default_route, ["selected_route", "continued_regression_guard", "guard_id"], ""),
                "guard_armed": get_path(procedural_default_route, ["selected_route", "continued_regression_guard", "armed"], False),
                "revalidate_command": get_path(procedural_default_route, ["selected_route", "continued_regression_guard", "revalidate_command"], ""),
                "rollback_criteria": get_path(procedural_default_route, ["selected_route", "continued_regression_guard", "rollback_criteria"], []),
                "mode": "guarded_local_metadata_route_only",
                "learned_generation_claim": False,
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
                "non_claims": procedural_default_route.get("non_claims") if isinstance(procedural_default_route.get("non_claims"), list) else [],
            },
        )
    add(
        "authority_transition",
        {
            "command_contract_id": command_id,
            "from_authority": "operator_prompt",
            "to_authority": "local_assistant_runtime",
            "authority_ceiling": ["read allowed project reports", "write local report artifacts", "append metadata-only dogfood"],
            "disallowed_authority": ["network teacher serving", "public benchmark training", "arbitrary shell beyond registered commands"],
            "prompt_raw_text_hidden_from_training_records": True,
        },
    )
    add(
        "authority_use_receipt",
        {
            "adapter_invocation_id": adapter_id,
            "secret_handles_used": [],
            "network_operations": [],
            "filesystem_writes": artifact_refs,
            "viea_materialized_view_receipt_id": materialized_view_receipt.get("receipt_id"),
            "route_validator_receipt_id": route_validator_receipt.get("receipt_id"),
            "private_verifier_receipt_id": private_verifier_receipt.get("receipt_id"),
            "runtime_external_inference_calls": 0,
            "public_training_rows_written": 0,
            "teacher_policy_runtime_external_tokens_forbidden": teacher_policy.get("runtime_external_tokens_forbidden"),
        },
    )
    add(
        "resource_budget",
        {
            "typed_job_id": job_id,
            "budget_class": "local_interactive_assistant",
            "timeout_policy_seconds": {"checkpoint_chat": 180, "context_refresh": 120, "tool_evidence": 180},
            "verification_tax_included": bool(code_private_probe.get("active") or tool_evidence.get("active")),
            "viea_materialized_view_record_count": materialized_view_receipt.get("record_count"),
            "route_validator_resource_route_record_count": route_validator_receipt.get("resource_route_record_count"),
            "private_verifier_record_count": private_verifier_receipt.get("viea_verifier_record_count"),
            "accepted_output_chars": len(assistant_text),
            "runtime_cost_units": "wall_clock_ms_and_report_writes",
        },
    )
    add(
        "generation_mode",
        {
            "typed_job_id": job_id,
            "mode": "local_checkpoint_chat_plus_registered_evidence",
            "model_only_claim": False,
            "tool_assisted_claim": bool(tool_evidence.get("active")),
            "private_verifier_receipt_ready": private_verifier_receipt.get("ready"),
            "procedural_default_route_active": procedural_default_route.get("active"),
            "procedural_default_route_ready": procedural_default_route.get("ready"),
            "learned_generation_claim": False,
            "deterministic_tool_credit_separate": True,
            "fallback_return_count": 0,
        },
    )
    add(
        "failure_boundary",
        {
            "typed_job_id": job_id,
            "protected_invariants": [
                "no public benchmark training payloads",
                "no runtime external teacher tokens",
                "no fallback return credit",
                "no raw prompt training rows",
            ],
            "detectors": ["assistant runtime gates", "teacher policy packet", "candidate integrity reports", "dogfood metadata bridge"],
            "materialized_view_ready": materialized_view_receipt.get("ready"),
            "route_validator_ready": route_validator_receipt.get("ready"),
            "private_verifier_ready": private_verifier_receipt.get("ready"),
            "current_faults": assistant_faults(intent, chat_result, code_private_probe, tool_evidence, selected_context),
            "containment": "surface in gates and residuals; do not promote capability claim",
        },
    )
    artifact_graph_id = add(
        "artifact_graph_record",
        {
            "planforge_dag_id": plan_id,
            "typed_job_id": job_id,
            "artifacts": artifact_refs,
            "context_refs": [context_id],
            "context_transaction_refs": [context_transaction_id],
            "context_adequacy_refs": [context_adequacy_id],
            "tool_refs": tool_refs_for_trace(tool_evidence),
            "verifier_refs": [private_verifier_receipt.get("path")] if private_verifier_receipt.get("path") else [],
            "materialized_view_refs": [materialized_view_receipt.get("view_path")] if materialized_view_receipt.get("view_path") else [],
            "route_validator_receipt_id": route_validator_receipt.get("receipt_id"),
            "checkpoint_refs": [checkpoint_id],
            "replay_limits": ["raw prompt not persisted", "checkpoint live state may change"],
        },
    )
    claim_id = add(
        "claim_record",
        {
            "artifact_graph_id": artifact_graph_id,
            "claim": "assistant runtime produced a local answer with governed evidence boundaries",
            "support_state": "supported_by_runtime_report" if chat_result.get("returncode") == 0 and assistant_text.strip() else "unsupported",
            "evidence_refs": artifact_refs,
            "materialized_view_receipt_id": materialized_view_receipt.get("receipt_id"),
            "route_validator_receipt_id": route_validator_receipt.get("receipt_id"),
            "private_verifier_receipt_id": private_verifier_receipt.get("receipt_id"),
            "negative_results": assistant_faults(intent, chat_result, code_private_probe, tool_evidence, selected_context),
            "non_claims": [
                "not a learned code-generation promotion",
                "not a public benchmark result",
                "not external inference serving",
            ],
        },
    )
    add(
        "evidence_transition_record",
        {
            "claim_id": claim_id,
            "from_state": "unmeasured",
            "to_state": "supported_by_runtime_report" if chat_result.get("returncode") == 0 and assistant_text.strip() else "blocked",
            "verification_refs": artifact_refs,
            "materialized_view_ready": materialized_view_receipt.get("ready"),
            "route_validator_ready": route_validator_receipt.get("ready"),
            "private_verifier_ready": private_verifier_receipt.get("ready"),
            "contradiction_refs": contradiction_refs_for_trace(code_private_probe, benchmark_status),
            "operator_feedback": feedback,
        },
    )
    add(
        "residual_record",
        {
            "claim_id": claim_id,
            "residuals": assistant_residuals(intent, code_private_probe, tool_evidence, selected_context, benchmark_status),
            "next_repair_target": "improve registered implementation behind the route, not a new sidecar",
            "public_training_rows_written": 0,
        },
    )
    add(
        "policy_optimization_record",
        {
            "claim_id": claim_id,
            "feedback": feedback,
            "assistant_trace_schema": {
                "path": trace_schema.get("path"),
                "policy": trace_schema.get("policy"),
                "schema_version": trace_schema.get("schema_version"),
                "sha256": trace_schema.get("sha256"),
                "ready": assistant_trace_schema_ready(trace_schema),
                "allowed_outcomes": trace_schema.get("allowed_outcomes"),
                "required_event_fields": trace_schema.get("required_event_fields"),
            },
            "dogfood_event_written": get_path(dogfood, ["event", "event_written"], False),
            "dogfood_training_rows_written": get_path(dogfood, ["training_bridge", "training_rows_written"], 0),
            "raw_text_training_allowed": config.get("raw_text_training_allowed") is True,
            "eligible_for_private_metadata_learning": feedback in set(trace_schema.get("allowed_outcomes") or []),
            "training_boundary": "metadata-only dogfood and governed private rows; public calibration payloads excluded",
        },
    )
    return records


def assistant_plan_nodes(
    intent: str,
    node_id: str,
    code_private_probe: dict[str, Any],
    tool_evidence: dict[str, Any],
    plan_context: dict[str, Any],
    procedural_default_route: dict[str, Any],
) -> list[dict[str, Any]]:
    nodes = [
        {"node_id": f"{node_id}:context_read", "kind": "context_read", "verifier": "context_adequacy_gate"},
        {"node_id": f"{node_id}:private_verifier_receipt", "kind": "private_verifier_spine_receipt", "verifier": "private_verifier_spine_v1"},
        {"node_id": f"{node_id}:checkpoint_chat", "kind": "local_checkpoint_chat", "verifier": "assistant_answer_present"},
        {"node_id": f"{node_id}:answer_compose", "kind": "answer_composition", "verifier": "no_external_runtime_inference"},
        {"node_id": f"{node_id}:trace_emit", "kind": "viea_trace_emit", "verifier": "assistant_viea_trace_complete"},
    ]
    if intent == "code":
        nodes.insert(2, {"node_id": f"{node_id}:private_replay_probe", "kind": "private_code_probe", "active": bool(code_private_probe.get("active")), "verifier": "candidate_integrity_and_replay"})
    if tool_evidence.get("required"):
        nodes.insert(2, {"node_id": f"{node_id}:tool_evidence", "kind": "deterministic_tool_evidence", "active": bool(tool_evidence.get("active")), "verifier": "tool_trace_and_artifact_graph"})
    if plan_context.get("required"):
        nodes.insert(2, {"node_id": f"{node_id}:plan_context", "kind": "plan_compiler_context", "active": bool(plan_context.get("active")), "verifier": "plan_compiler_gate"})
        nodes.insert(
            3,
            {
                "node_id": f"{node_id}:procedural_default_route",
                "kind": "procedural_memory_default_route",
                "active": bool(procedural_default_route.get("active")),
                "ready": bool(procedural_default_route.get("ready")),
                "route_id": get_path(procedural_default_route, ["selected_route", "id"], ""),
                "verifier": "procedural_route_adoption_gate",
            },
        )
    return nodes


def assistant_viea_trace_required(intent: str) -> bool:
    return intent in {"code", "tool", "planning"}


def assistant_viea_trace_complete(rows: list[dict[str, Any]]) -> bool:
    return viea_spine_records.trace_complete(rows, ASSISTANT_VIEA_REQUIRED_RECORD_TYPES)


def missing_assistant_viea_records(rows: list[dict[str, Any]]) -> list[str]:
    return viea_spine_records.missing_required_record_types(rows, ASSISTANT_VIEA_REQUIRED_RECORD_TYPES)


def assistant_trace_has_raw_prompt(rows: list[dict[str, Any]]) -> bool:
    forbidden_keys = {"prompt", "raw_prompt", "raw_user_text", "raw_text", "training_text"}

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key) in forbidden_keys and child not in {"", None, False}:
                    return True
                if walk(child):
                    return True
        elif isinstance(value, list):
            return any(walk(child) for child in value)
        return False

    return walk(rows)


def assistant_faults(
    intent: str,
    chat_result: dict[str, Any],
    code_private_probe: dict[str, Any],
    tool_evidence: dict[str, Any],
    selected_context: dict[str, Any],
) -> list[dict[str, Any]]:
    faults: list[dict[str, Any]] = []
    if chat_result.get("returncode") != 0:
        faults.append({"family": "checkpoint_chat_fault", "returncode": chat_result.get("returncode")})
    if not selected_context.get("ready") or not selected_context.get("selected_pages"):
        faults.append({"family": "context_partial_or_missing", "ready": selected_context.get("ready")})
    if intent == "code" and code_private_probe.get("active"):
        wall = code_private_probe_wall(code_private_probe)
        if wall.get("semantic_pass_currently_zero"):
            faults.append({"family": "learned_generator_semantic_pass_zero", "wall": wall})
        if int_or_zero(wall.get("candidate_integrity_mismatch_count")):
            faults.append({"family": "candidate_integrity_mismatch", "wall": wall})
    if tool_evidence.get("required") and tool_evidence.get("active") and tool_evidence.get("trigger_state") not in {"GREEN", "YELLOW"}:
        faults.append({"family": "tool_evidence_fault", "state": tool_evidence.get("trigger_state")})
    return faults


def assistant_residuals(
    intent: str,
    code_private_probe: dict[str, Any],
    tool_evidence: dict[str, Any],
    selected_context: dict[str, Any],
    benchmark_status: dict[str, Any],
) -> list[dict[str, Any]]:
    residuals = assistant_faults(intent, {"returncode": 0}, code_private_probe, tool_evidence, selected_context)
    dominant = benchmark_status.get("dominant_residual")
    if isinstance(dominant, list) and dominant:
        residuals.append({"family": "public_transfer_residual", "dominant_residual": dominant})
    if not residuals:
        residuals.append({"family": "none_for_this_runtime_call", "status": "no_runtime_fault_detected"})
    return residuals


def contradiction_refs_for_trace(code_private_probe: dict[str, Any], benchmark_status: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    if code_private_probe.get("report"):
        refs.append(str(code_private_probe.get("report")))
    if benchmark_status.get("measurement_report"):
        refs.append(str(benchmark_status.get("measurement_report")))
    return unique_strings(refs)


def tool_refs_for_trace(tool_evidence: dict[str, Any]) -> list[str]:
    refs = []
    if tool_evidence.get("report"):
        refs.append(str(tool_evidence.get("report")))
    if tool_evidence.get("trace"):
        refs.append(str(tool_evidence.get("trace")))
    if tool_evidence.get("artifact_graph"):
        refs.append(str(tool_evidence.get("artifact_graph")))
    return unique_strings(refs)


def plan_context_packet(required: bool) -> dict[str, Any]:
    plan = read_json(REPORTS / "theseus_plan_compiler.json", {})
    summary = plan.get("summary") if isinstance(plan.get("summary"), dict) else {}
    return {
        "active": required,
        "required": required,
        "planner_state": plan.get("trigger_state"),
        "compiled_goal_count": summary.get("compiled_goal_count"),
        "hard_failed_gate_count": summary.get("hard_failed_gate_count"),
    }


def run_local_effect_canary(
    *,
    enabled: bool,
    target: Path,
    allowed_root: Path,
    session_id: str,
    intent: str,
    prompt_hash: str,
    reflexive_dispatch_trace: dict[str, Any],
) -> dict[str, Any]:
    """Perform one real route-authority change and prove exact rollback.

    The proposer declares one filesystem effect. A separate observer reloads
    the bytes and route payload. A third evaluator compares the observation to
    the declaration and verifies rollback against the original byte identity.
    """
    if not enabled:
        return {
            "enabled": False,
            "ready": True,
            "non_claims": ["effect completeness was not exercised for this assistant call"],
        }

    proposer_id = "theseus_assistant_effect_proposer_v1"
    observer_id = "theseus_filesystem_effect_observer_v1"
    evaluator_id = "theseus_effect_rollback_evaluator_v1"
    dispatch_trace_id = str(reflexive_dispatch_trace.get("trace_id") or "")
    decision_digest = str(reflexive_dispatch_trace.get("decision_digest") or "")
    selected_capabilities = selected_reflexive_capabilities(reflexive_dispatch_trace)
    try:
        dispatch_verification = reflexive_dispatch.verify_trace(reflexive_dispatch_trace)
    except reflexive_dispatch.ReflexiveDispatchFault:
        dispatch_verification = {"state": "REJECTED"}
    dispatch_bound = bool(
        dispatch_verification.get("state") == "VERIFIED"
        and dispatch_trace_id
        and decision_digest
        and reflexive_terminal_outcome(reflexive_dispatch_trace) == "prepared"
        and selected_capabilities == ["assistant.route_authority_effect"]
        and get_path(reflexive_dispatch_trace, ["effect", "required"], False) is True
        and get_path(reflexive_dispatch_trace, ["effect", "effect_authority_granted"], True) is False
    )
    transaction_id = stable_id(
        "assistant_effect_transaction",
        session_id,
        intent,
        prompt_hash,
        dispatch_trace_id,
        decision_digest,
    )
    base = {
        "enabled": True,
        "ready": False,
        "policy": "project_theseus_bounded_local_effect_transaction_v1",
        "transaction_id": transaction_id,
        "proposer_id": proposer_id,
        "observer_id": observer_id,
        "evaluator_id": evaluator_id,
        "dispatch_trace_id": dispatch_trace_id,
        "dispatch_decision_digest": decision_digest,
        "selected_capability_ids": selected_capabilities,
        "dispatch_bound": dispatch_bound,
        "target": rel(target),
        "effect_inventory": [],
        "observation": {},
        "rollback": {"complete": False, "residual_count": 1},
        "residuals": [],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claims": [
            "this canary proves local effect accounting and rollback, not model capability",
            "the route-authority state is rolled back before the assistant call returns",
        ],
    }
    if not dispatch_bound:
        base["residuals"] = [{"kind": "effect_dispatch_binding_invalid"}]
        return base
    try:
        target = validate_effect_target(target, allowed_root)
    except ValueError as exc:
        base["residuals"] = [{"kind": "effect_target_denied", "detail": str(exc)}]
        return base

    before = observe_effect_target(target, observer_id=observer_id)
    candidate = {
        "policy": "project_theseus_local_route_authority_canary_v1",
        "transaction_id": transaction_id,
        "dispatch_trace_id": dispatch_trace_id,
        "dispatch_decision_digest": decision_digest,
        "capability_id": "assistant.route_authority_effect",
        "intent": intent,
        "assistant_surface": "local_assistant",
        "authority_ceiling": ["bounded_local_metadata_route"],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    candidate_bytes = (json.dumps(candidate, indent=2, sort_keys=True) + "\n").encode("utf-8")
    intended_sha = hashlib.sha256(candidate_bytes).hexdigest()
    base["effect_inventory"] = [
        {
            "effect_id": stable_id("declared_effect", transaction_id, rel(target), intended_sha),
            "path": rel(target),
            "operation": "replace" if before["exists"] else "create",
            "before_identity": before["identity"],
            "intended_content_sha256": intended_sha,
            "intended_mode": "0o600",
            "rollback_obligation": "restore exact prior bytes and mode" if before["exists"] else "remove created path",
        }
    ]

    prior_bytes = target.read_bytes() if before["exists"] else None
    prior_mode = int(before.get("mode", 0)) if before["exists"] else None
    write_fault = ""
    try:
        atomic_write_bytes(target, candidate_bytes, mode=0o600)
        observed = observe_effect_target(target, observer_id=observer_id, parse_json=True)
    except OSError as exc:
        observed = observe_effect_target(target, observer_id=observer_id)
        write_fault = f"{type(exc).__name__}: {exc}"

    parsed = observed.get("parsed_json") if isinstance(observed.get("parsed_json"), dict) else {}
    matches_intent = bool(
        observed.get("exists")
        and observed.get("sha256") == intended_sha
        and parsed.get("transaction_id") == transaction_id
        and parsed.get("dispatch_trace_id") == dispatch_trace_id
        and parsed.get("dispatch_decision_digest") == decision_digest
        and parsed.get("capability_id") == "assistant.route_authority_effect"
        and dispatch_bound
    )
    base["observation"] = {
        **observed,
        "matches_intent": matches_intent,
        "expected_content_sha256": intended_sha,
        "expected_dispatch_trace_id": dispatch_trace_id,
        "expected_dispatch_decision_digest": decision_digest,
        "write_fault": write_fault,
    }

    rollback_fault = ""
    try:
        if prior_bytes is None:
            target.unlink(missing_ok=True)
        else:
            atomic_write_bytes(target, prior_bytes, mode=prior_mode or 0o600)
    except OSError as exc:
        rollback_fault = f"{type(exc).__name__}: {exc}"
    final = observe_effect_target(target, observer_id=observer_id)
    rollback_complete = final["identity"] == before["identity"] and not rollback_fault
    residuals = []
    if not matches_intent:
        residuals.append({"kind": "effect_observation_mismatch", "observed_identity": observed.get("identity")})
    if not rollback_complete:
        residuals.append(
            {
                "kind": "rollback_identity_mismatch",
                "before_identity": before["identity"],
                "final_identity": final["identity"],
                "fault": rollback_fault,
            }
        )
    base["rollback"] = {
        "complete": rollback_complete,
        "before_identity": before["identity"],
        "first_effect_identity": observed.get("identity"),
        "final_identity": final["identity"],
        "prior_path_existed": before["exists"],
        "restored_prior_bytes": prior_bytes is not None and rollback_complete,
        "removed_new_path": prior_bytes is None and rollback_complete,
        "residual_count": len(residuals),
        "fault": rollback_fault,
    }
    base["residuals"] = residuals
    base["ready"] = matches_intent and rollback_complete and not residuals
    return base


def validate_effect_target(target: Path, allowed_root: Path) -> Path:
    allowed = allowed_root.resolve()
    resolved = target.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"effect target must remain under {allowed}") from exc
    if target.is_symlink():
        raise ValueError("effect target may not be a symlink")
    return resolved


def observe_effect_target(target: Path, *, observer_id: str, parse_json: bool = False) -> dict[str, Any]:
    if not target.exists():
        identity = stable_hash({"exists": False, "path": rel(target)})
        return {
            "observer_id": observer_id,
            "exists": False,
            "path": rel(target),
            "sha256": "",
            "size_bytes": 0,
            "mode": 0,
            "identity": identity,
        }
    content = target.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    mode = target.stat().st_mode & 0o777
    row: dict[str, Any] = {
        "observer_id": observer_id,
        "exists": True,
        "path": rel(target),
        "sha256": sha,
        "size_bytes": len(content),
        "mode": mode,
        "identity": stable_hash({"exists": True, "path": rel(target), "sha256": sha, "mode": mode}),
    }
    if parse_json:
        try:
            parsed = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        row["parsed_json"] = parsed if isinstance(parsed, dict) else {}
    return row


def atomic_write_bytes(path: Path, content: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        os.chmod(temporary, mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def procedural_default_route_packet(
    intent: str,
    route_config: dict[str, Any],
    config: dict[str, Any],
    *,
    surface: str = "local_assistant",
) -> dict[str, Any]:
    policy = config.get("procedural_memory_default_route") if isinstance(config.get("procedural_memory_default_route"), dict) else {}
    enabled = policy.get("enabled") is not False
    eligible_intents = {str(item) for item in policy.get("eligible_intents", []) if str(item)}
    required_intents = {str(item) for item in policy.get("required_for_intents", []) if str(item)}
    active = enabled and (not eligible_intents or intent in eligible_intents)
    required = enabled and intent in required_intents
    report_path = resolve(str(policy.get("report") or rel(DEFAULT_PROCEDURAL_ADOPTION_REPORT)))
    report = read_json(report_path, {})
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    default_routes = report.get("default_routes") if isinstance(report.get("default_routes"), list) else []
    selected_route: dict[str, Any] = {}
    for route in default_routes:
        if not isinstance(route, dict):
            continue
        guard = route.get("continued_regression_guard") if isinstance(route.get("continued_regression_guard"), dict) else {}
        if (
            route.get("default_route_adopted") is True
            and guard.get("armed") is True
            and route.get("learned_generation_claim_allowed") is False
            and procedural_route_matches_runtime(route, intent, route_config, policy, surface=surface)
            and int_or_zero(route.get("public_training_rows_written")) == 0
            and int_or_zero(route.get("external_inference_calls")) == 0
            and int_or_zero(route.get("fallback_return_count")) == 0
        ):
            selected_route = route
            break
    ready = bool(
        active
        and report_path.exists()
        and report.get("trigger_state") == "GREEN"
        and int_or_zero(summary.get("hard_gap_count")) == 0
        and int_or_zero(summary.get("default_route_adopted_count")) >= 1
        and int_or_zero(summary.get("default_route_guarded_count")) >= 1
        and int_or_zero(summary.get("learned_generation_claim_count")) == 0
        and int_or_zero(summary.get("public_training_rows_written")) == 0
        and int_or_zero(summary.get("external_inference_calls")) == 0
        and int_or_zero(summary.get("fallback_return_count")) == 0
        and bool(selected_route)
    )
    return {
        "active": active,
        "required": required,
        "ready": ready,
        "policy": policy,
        "report": rel(report_path),
        "present": report_path.exists(),
        "trigger_state": report.get("trigger_state"),
        "summary": {
            "default_route_adopted_count": summary.get("default_route_adopted_count", 0),
            "default_route_guarded_count": summary.get("default_route_guarded_count", 0),
            "transaction_count": summary.get("transaction_count", 0),
            "viea_route_adoption_record_count": summary.get("viea_route_adoption_record_count", 0),
            "hard_gap_count": summary.get("hard_gap_count", 0),
            "warning_count": summary.get("warning_count", 0),
        },
        "selected_route": selected_route,
        "selection": procedural_route_selection_evidence(selected_route, intent, route_config, policy, surface=surface),
        "public_training_rows_written": int_or_zero(summary.get("public_training_rows_written")),
        "external_inference_calls": int_or_zero(summary.get("external_inference_calls")),
        "fallback_return_count": int_or_zero(summary.get("fallback_return_count")),
        "learned_generation_claim_count": int_or_zero(summary.get("learned_generation_claim_count")),
        "non_claims": policy.get("non_claims") if isinstance(policy.get("non_claims"), list) else [],
        "failure_behavior": "fail_closed_for_required_intents; leave explicit route otherwise",
    }


def procedural_route_matches_runtime(
    route: dict[str, Any],
    intent: str,
    route_config: dict[str, Any],
    policy: dict[str, Any],
    *,
    surface: str = "local_assistant",
) -> bool:
    binding = route.get("route_binding_contract") if isinstance(route.get("route_binding_contract"), dict) else {}
    if not binding and policy.get("selection_mode") == "route_binding_contract":
        return False
    consumer = str(policy.get("required_runtime_consumer") or "theseus_assistant_runtime")
    consumers = {str(item) for item in list_value(route.get("runtime_consumers") or binding.get("runtime_consumers"))}
    surfaces = {str(item) for item in list_value(route.get("assistant_surfaces") or binding.get("assistant_surfaces"))}
    intents = {str(item) for item in list_value(route.get("assistant_intents") or binding.get("assistant_intents"))}
    lanes = {str(item) for item in list_value(route.get("assistant_lanes") or binding.get("assistant_lanes"))}
    families = {str(item) for item in list_value(route.get("vcm_task_families") or binding.get("vcm_task_families"))}
    lane = str(route_config.get("assistant_lane") or "")
    family = str(route_config.get("vcm_task_family") or "")
    return (
        (not consumers or consumer in consumers)
        and (not surfaces or surface in surfaces)
        and (not intents or intent in intents)
        and (not lanes or lane in lanes)
        and (not families or family in families)
    )


def procedural_route_selection_evidence(
    route: dict[str, Any],
    intent: str,
    route_config: dict[str, Any],
    policy: dict[str, Any],
    *,
    surface: str = "local_assistant",
) -> dict[str, Any]:
    binding = route.get("route_binding_contract") if isinstance(route.get("route_binding_contract"), dict) else {}
    return {
        "selection_mode": policy.get("selection_mode", "route_binding_contract"),
        "required_runtime_consumer": policy.get("required_runtime_consumer", "theseus_assistant_runtime"),
        "requested_surface": surface,
        "requested_intent": intent,
        "requested_assistant_lane": route_config.get("assistant_lane"),
        "requested_vcm_task_family": route_config.get("vcm_task_family"),
        "matched": bool(route) and procedural_route_matches_runtime(route, intent, route_config, policy, surface=surface),
        "route_scope": route.get("route_scope", ""),
        "route_binding_contract": binding,
        "selection_keys": route.get("selection_keys") or binding.get("selection_keys") or [],
    }


def benchmark_status_packet(prompt: str) -> dict[str, Any]:
    measurement = read_json(REPORTS / "theseus_benchmark_measurement.json", {})
    state = read_json(REPORTS / "theseus_assistant_state_report.json", {})
    measurement_summary = measurement.get("summary") if isinstance(measurement.get("summary"), dict) else {}
    state_summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
    latest = measurement_summary.get("latest_public_run")
    latest = latest if isinstance(latest, dict) else {}
    score_report_path = str(latest.get("score_report_path") or "")
    score_report = read_json(resolve(score_report_path), {}) if score_report_path else {}
    score_summary = score_report.get("summary") if isinstance(score_report.get("summary"), dict) else {}
    latest_card_counts = score_summary.get("case_manifest_card_counts")
    latest_cards = []
    if isinstance(latest_card_counts, dict):
        latest_cards = [str(card) for card in latest_card_counts]
    elif isinstance(score_report.get("cards"), list):
        latest_cards = [str(card) for card in score_report.get("cards", [])]
    dominant = latest.get("dominant_residual_categories")
    if isinstance(dominant, list) and dominant and isinstance(dominant[0], list):
        dominant_residual = dominant[0]
    else:
        dominant_residual = state_summary.get("public_residual_dominant_failure") or []
    text = prompt.lower()
    include = has_any(
        text,
        [
            "benchmark",
            "public",
            "score",
            "status",
            "state",
            "doing",
            "current",
            "result",
            "repair target",
            "next coding",
            "transfer",
        ],
    )
    return {
        "active": bool(measurement_summary or state_summary),
        "include_in_answer": include,
        "measurement_report": rel(REPORTS / "theseus_benchmark_measurement.json"),
        "state_report": rel(REPORTS / "theseus_assistant_state_report.json"),
        "measurement_kind": measurement_summary.get("measurement_kind"),
        "effective_cards": measurement_summary.get("effective_cards") if isinstance(measurement_summary.get("effective_cards"), list) else [],
        "latest_public_cards": latest_cards,
        "latest_public_run_id": latest.get("run_id") or state_summary.get("latest_public_surface"),
        "latest_public_passed": latest.get("passed"),
        "latest_public_task_count": latest.get("task_count") or state_summary.get("latest_public_task_count"),
        "latest_public_pass_rate": latest.get("pass_rate") if latest.get("pass_rate") is not None else state_summary.get("latest_public_score"),
        "dominant_residual": dominant_residual,
        "public_training_rows_written": int_or_zero(measurement_summary.get("public_training_rows_written")),
        "external_inference_calls": int_or_zero(measurement_summary.get("external_inference_calls")),
        "fallback_return_count": int_or_zero(latest.get("fallback_return_count")),
        "blockers": state.get("blockers") if isinstance(state.get("blockers"), list) else [],
    }


def teacher_policy_packet() -> dict[str, Any]:
    teacher_policy = read_json(ROOT / "configs" / "teacher_policy.json", {})
    distill_policy = read_json(ROOT / "configs" / "teacher_distillation_policy.json", {})
    gate = read_json(REPORTS / "teacher_distillation_gate.json", {})
    share_report = read_json(REPORTS / "teacher_share_ledger_summary.json", {})
    manifest = read_json(REPORTS / "teacher_distillation_manifest.json", {})
    gate_summary = gate.get("summary") if isinstance(gate.get("summary"), dict) else {}
    share_summary = share_report.get("summary") if isinstance(share_report.get("summary"), dict) else {}
    manifest_summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    distill_boundary = distill_policy.get("boundary") if isinstance(distill_policy.get("boundary"), dict) else {}
    teacher_budget = teacher_policy.get("budget") if isinstance(teacher_policy.get("budget"), dict) else {}
    teacher_prompt_contract = teacher_policy.get("teacher_prompt_contract") if isinstance(teacher_policy.get("teacher_prompt_contract"), dict) else {}
    return {
        "active": True,
        "teacher_policy": rel(ROOT / "configs" / "teacher_policy.json"),
        "distillation_policy": rel(ROOT / "configs" / "teacher_distillation_policy.json"),
        "gate_report": rel(REPORTS / "teacher_distillation_gate.json"),
        "teacher_share_ledger_summary": rel(REPORTS / "teacher_share_ledger_summary.json"),
        "teacher_share_ledger_state": share_report.get("trigger_state"),
        "teacher_share_metric_ready": share_summary.get("metric_ready"),
        "manifest": rel(REPORTS / "teacher_distillation_manifest.json"),
        "teacher_default_mode": teacher_policy.get("default_mode"),
        "proposal_mode_default": teacher_policy.get("default_mode") == "proposal",
        "apply_mode_enabled": teacher_budget.get("apply_mode_enabled") is True,
        "distillation_training_enabled_by_policy": bool(teacher_budget.get("distillation_training_enabled")),
        "distillation_allowed": gate_summary.get("distillation_allowed"),
        "gate_state": gate.get("trigger_state"),
        "gate_hard_blocker_count": gate_summary.get("hard_blocker_count"),
        "manifest_row_count": gate_summary.get("manifest_row_count", manifest_summary.get("row_count")),
        "manifest_verifier_pass_rate": gate_summary.get("manifest_verifier_pass_rate", manifest_summary.get("verifier_pass_rate")),
        "manifest_public_overlap_hits": gate_summary.get("manifest_public_overlap_hits", manifest_summary.get("public_overlap_hits")),
        "manifest_holdout_overlap_hits": gate_summary.get("manifest_holdout_overlap_hits", manifest_summary.get("holdout_overlap_hits")),
        "manifest_admission_safety_checks_clean": gate_summary.get("manifest_admission_safety_checks_clean", manifest_summary.get("admission_safety_checks_clean")),
        "teacher_accepted_row_share": share_summary.get("teacher_share_of_accepted_training_rows", gate_summary.get("teacher_accepted_row_share")),
        "teacher_accepted_rows": share_summary.get("teacher_accepted_rows", gate_summary.get("teacher_accepted_rows")),
        "verified_self_generated_rows": share_summary.get("verified_self_generated_rows", gate_summary.get("verified_self_generated_rows")),
        "teacher_proposal_rows_recorded": share_summary.get("teacher_proposal_rows", gate_summary.get("teacher_proposal_rows_recorded")),
        "teacher_rejected_rows_recorded": share_summary.get("teacher_rejected_rows", gate_summary.get("teacher_rejected_rows_recorded")),
        "teacher_share_within_cap": share_summary.get("teacher_share_within_cap", gate_summary.get("teacher_share_within_cap")),
        "teacher_share_target_trend": share_summary.get("teacher_share_target_trend"),
        "teacher_share_graduation_target": share_summary.get("teacher_share_graduation_target"),
        "runtime_external_tokens_forbidden": (
            distill_boundary.get("runtime_serving_external_tokens") == "forbidden"
            and distill_boundary.get("external_inference_at_runtime") == "forbidden"
        ),
        "teacher_apply_mode_forbidden": (
            distill_boundary.get("teacher_apply_mode") == "forbidden"
            and teacher_budget.get("apply_mode_enabled") is False
        ),
        "public_benchmark_distillation_forbidden": "public_benchmark_solution_distillation" in (distill_policy.get("blocked_uses") or []),
        "teacher_prompt_must_not_emit_benchmark_answers": teacher_prompt_contract.get("must_not_emit_benchmark_answers_or_templates") is True,
        "runtime_external_inference_calls": 0,
        "public_training_rows_written": 0,
    }


def build_gates(
    *,
    chat_result: dict[str, Any],
    response: dict[str, Any],
    assistant_text: str,
    reflexive_dispatch_trace: dict[str, Any],
    reflexive_dispatch_verification: dict[str, Any],
    feedback: str,
    dogfood: dict[str, Any],
    selected_context: dict[str, Any],
    context_refresh: list[dict[str, Any]],
    intent: str,
    code_private_probe: dict[str, Any],
    tool_evidence: dict[str, Any],
    procedural_default_route: dict[str, Any],
    teacher_policy: dict[str, Any],
    config: dict[str, Any],
    checkpoint_session: dict[str, Any],
    materialized_view_receipt: dict[str, Any],
    route_validator_receipt: dict[str, Any],
    private_verifier_receipt: dict[str, Any],
    vcm_governor: dict[str, Any],
    assistant_viea_trace: list[dict[str, Any]],
    trace_schema: dict[str, Any],
    allowed_feedback: set[str],
    effect_canary: dict[str, Any],
) -> list[dict[str, Any]]:
    event_state = get_path(dogfood, ["event", "trigger_state"], "") if dogfood else "skipped"
    bridge_state = get_path(dogfood, ["training_bridge", "trigger_state"], "") if dogfood else "skipped"
    code_probe_summary = code_private_probe.get("summary") if isinstance(code_private_probe.get("summary"), dict) else {}
    tool_summary = tool_evidence.get("summary") if isinstance(tool_evidence.get("summary"), dict) else {}
    dispatch_prepared = reflexive_dispatch_prepared(reflexive_dispatch_trace, reflexive_dispatch_verification)
    dispatch_terminal = reflexive_terminal_outcome(reflexive_dispatch_trace)
    dispatch_safely_stopped = (
        reflexive_dispatch_verification.get("state") == "VERIFIED"
        and dispatch_terminal in {"ambiguous", "insufficient_context", "insufficient_evidence", "conflicting", "stale", "unauthorized", "unsupported", "ood", "resource_exceeded", "escalate", "rejected"}
        and chat_result.get("skipped") is True
    )
    return [
        gate(
            "reflexive_dispatch_integrity_verified",
            reflexive_dispatch_verification.get("state") == "VERIFIED",
            reflexive_dispatch_verification,
            "hard",
        ),
        gate(
            "reflexive_dispatch_precedes_downstream_execution",
            (dispatch_prepared and chat_result.get("skipped") is not True) or dispatch_safely_stopped,
            {
                "terminal_outcome": dispatch_terminal,
                "selected_capabilities": selected_reflexive_capabilities(reflexive_dispatch_trace),
                "checkpoint_skipped": chat_result.get("skipped") is True,
            },
            "hard",
        ),
        gate("checkpoint_chat_completed", dispatch_safely_stopped or chat_result.get("returncode") == 0, chat_result, "hard"),
        gate(
            "checkpoint_session_memory_available",
            dispatch_safely_stopped or (bool(checkpoint_session.get("session_id")) and checkpoint_session.get("session_path") is not None),
            checkpoint_session,
            "hard",
        ),
        gate(
            "assistant_answer_present",
            bool(str(assistant_text or "").strip()) and (dispatch_safely_stopped or bool(str(response.get("answer") or "").strip())),
            {"mode": response.get("mode"), "dispatch_terminal": dispatch_terminal},
            "hard",
        ),
        gate(
            "code_private_probe_executed_and_safe",
            intent != "code" or code_private_probe_safe(code_private_probe),
            {
                "active": code_private_probe.get("active"),
                "trigger_state": code_private_probe.get("trigger_state"),
                "summary": code_probe_summary,
                "wall": code_private_probe_wall(code_private_probe),
            },
            "hard",
        ),
        gate(
            "code_private_probe_reports_current_capability_wall",
            intent != "code"
            or (
                code_private_probe_safe(code_private_probe)
                and (
                    float_or_zero(code_probe_summary.get("selected_intended_behavior_pass_rate")) > 0.0
                    or int_or_zero(code_probe_summary.get("eligible_candidate_count")) == 0
                    or int_or_zero(code_probe_summary.get("candidate_integrity_mismatch_count")) > 0
                )
            ),
            code_private_probe_wall(code_private_probe),
            "warning",
        ),
        gate("vcm_context_attached", bool(selected_context.get("ready")) and bool(selected_context.get("selected_pages")), compact_vcm_context(selected_context), "warning"),
        gate(
            "vcm_context_governor_ready",
            vcm_governor.get("ready") is True
            and vcm_governor.get("trigger_state") == "GREEN"
            and get_path(vcm_governor, ["summary", "mission_brief_status"], "") == "ready"
            and get_path(vcm_governor, ["summary", "deletion_closure_status"], "") == "closed"
            and get_path(vcm_governor, ["summary", "scif_status"], "") == "ready"
            and int_or_zero(get_path(vcm_governor, ["summary", "hard_gap_count"], 0)) == 0
            and int_or_zero(get_path(vcm_governor, ["summary", "deletion_closure_fault_count"], 0)) == 0
            and int_or_zero(get_path(vcm_governor, ["no_cheat", "public_training_rows_written"], 0)) == 0
            and int_or_zero(get_path(vcm_governor, ["no_cheat", "runtime_external_inference_calls"], 0)) == 0
            and int_or_zero(get_path(vcm_governor, ["no_cheat", "fallback_return_count"], 0)) == 0,
            vcm_governor,
            "hard",
        ),
        gate(
            "tool_evidence_recorded",
            tool_evidence.get("required") is not True
            or (
                tool_evidence.get("active") is True
                and tool_evidence.get("trigger_state") in {"GREEN", "YELLOW"}
                and int_or_zero(tool_summary.get("result_count")) > 0
                and float_or_zero(tool_summary.get("tool_on_solve_rate")) > 0.0
                and int_or_zero(tool_summary.get("public_training_rows_written")) == 0
                and int_or_zero(tool_summary.get("external_inference_calls")) == 0
                and int_or_zero(tool_summary.get("fallback_return_count")) == 0
                and bool(tool_evidence.get("trace"))
            ),
            {
                "active": tool_evidence.get("active"),
                "trigger_state": tool_evidence.get("trigger_state"),
                "summary": tool_summary,
                "trace": tool_evidence.get("trace"),
                "report": tool_evidence.get("report"),
            },
            "hard",
        ),
        gate(
            "procedural_default_route_guarded_for_planning",
            intent != "planning"
            or procedural_default_route.get("required") is not True
            or (
                procedural_default_route.get("active") is True
                and procedural_default_route.get("ready") is True
                and procedural_default_route.get("trigger_state") == "GREEN"
                and bool(get_path(procedural_default_route, ["selected_route", "id"], ""))
                and get_path(procedural_default_route, ["selected_route", "default_route_adopted"], False) is True
                and get_path(procedural_default_route, ["selected_route", "learned_generation_claim_allowed"], True) is False
                and get_path(procedural_default_route, ["selected_route", "continued_regression_guard", "armed"], False) is True
                and int_or_zero(procedural_default_route.get("learned_generation_claim_count")) == 0
                and int_or_zero(procedural_default_route.get("public_training_rows_written")) == 0
                and int_or_zero(procedural_default_route.get("external_inference_calls")) == 0
                and int_or_zero(procedural_default_route.get("fallback_return_count")) == 0
            ),
            procedural_default_route,
            "hard",
        ),
        gate(
            "procedural_default_route_binding_contract_enforced",
            procedural_default_route.get("active") is not True
            or procedural_default_route.get("required") is not True
            or (
                procedural_default_route.get("ready") is True
                and get_path(procedural_default_route, ["selection", "matched"], False) is True
                and bool(get_path(procedural_default_route, ["selection", "route_binding_contract"], {}))
                and "theseus_assistant_runtime"
                in {
                    str(item)
                    for item in list_value(
                        get_path(procedural_default_route, ["selected_route", "runtime_consumers"], [])
                    )
                }
            ),
            {
                "active": procedural_default_route.get("active"),
                "required": procedural_default_route.get("required"),
                "ready": procedural_default_route.get("ready"),
                "selection": procedural_default_route.get("selection"),
                "runtime_consumers": get_path(procedural_default_route, ["selected_route", "runtime_consumers"], []),
            },
            "hard",
        ),
        gate("context_refresh_no_hard_failures", not any(row.get("returncode") not in {0, None} for row in context_refresh), context_refresh, "warning"),
        gate(
            "assistant_trace_schema_ready",
            assistant_trace_schema_ready(trace_schema),
            {
                "path": trace_schema.get("path"),
                "policy": trace_schema.get("policy"),
                "schema_version": trace_schema.get("schema_version"),
                "sha256": trace_schema.get("sha256"),
                "allowed_outcomes": trace_schema.get("allowed_outcomes"),
                "missing_required_outcomes": missing_required_outcomes(trace_schema),
                "missing_required_event_fields": missing_required_event_fields(trace_schema),
            },
            "hard",
        ),
        gate("feedback_value_allowed", feedback in allowed_feedback, {"feedback": feedback, "allowed": sorted(allowed_feedback)}, "hard"),
        gate("dogfood_feedback_processed", (not feedback) or event_state in {"GREEN", "YELLOW"}, {"event_state": event_state, "bridge_state": bridge_state}, "warning"),
        gate(
            "teacher_policy_governed_runtime_boundary",
            teacher_policy.get("runtime_external_tokens_forbidden") is True
            and teacher_policy.get("teacher_apply_mode_forbidden") is True
            and teacher_policy.get("public_benchmark_distillation_forbidden") is True
            and int_or_zero(teacher_policy.get("runtime_external_inference_calls")) == 0
            and int_or_zero(teacher_policy.get("public_training_rows_written")) == 0,
            teacher_policy,
            "hard",
        ),
        gate(
            "teacher_share_ledger_metric_ready",
            teacher_policy.get("teacher_share_metric_ready") is True
            and teacher_policy.get("teacher_share_ledger_state") == "GREEN"
            and teacher_policy.get("teacher_share_within_cap") is True
            and int_or_zero(teacher_policy.get("runtime_external_inference_calls")) == 0
            and int_or_zero(teacher_policy.get("public_training_rows_written")) == 0,
            {
                "teacher_share_ledger_summary": teacher_policy.get("teacher_share_ledger_summary"),
                "teacher_share_ledger_state": teacher_policy.get("teacher_share_ledger_state"),
                "teacher_share_metric_ready": teacher_policy.get("teacher_share_metric_ready"),
                "teacher_accepted_row_share": teacher_policy.get("teacher_accepted_row_share"),
                "teacher_share_within_cap": teacher_policy.get("teacher_share_within_cap"),
            },
            "hard",
        ),
        gate(
            "teacher_distillation_gate_clean_when_allowed",
            teacher_policy.get("distillation_allowed") is not True
            or (
                teacher_policy.get("gate_state") == "GREEN"
                and int_or_zero(teacher_policy.get("gate_hard_blocker_count")) == 0
                and float_or_zero(teacher_policy.get("manifest_verifier_pass_rate")) >= 0.95
                and int_or_zero(teacher_policy.get("manifest_public_overlap_hits")) == 0
                and int_or_zero(teacher_policy.get("manifest_holdout_overlap_hits")) == 0
                and teacher_policy.get("teacher_share_within_cap") is True
            ),
            teacher_policy,
            "hard",
        ),
        gate(
            "assistant_viea_trace_complete",
            (not assistant_viea_trace_required(intent)) or assistant_viea_trace_complete(assistant_viea_trace),
            {
                "required": assistant_viea_trace_required(intent),
                "record_count": len(assistant_viea_trace),
                "missing_record_types": missing_assistant_viea_records(assistant_viea_trace),
            },
            "hard",
        ),
        gate(
            "assistant_materialized_view_receipt_ready",
            (not assistant_viea_trace_required(intent)) or materialized_view_receipt.get("ready") is True,
            materialized_view_receipt,
            "hard",
        ),
        gate(
            "assistant_route_validator_receipt_ready",
            (not assistant_viea_trace_required(intent)) or route_validator_receipt.get("ready") is True,
            route_validator_receipt,
            "hard",
        ),
        gate(
            "assistant_private_verifier_receipt_ready",
            (not assistant_viea_trace_required(intent)) or private_verifier_receipt.get("ready") is True,
            private_verifier_receipt,
            "hard",
        ),
        gate(
            "assistant_product_trace_exercises_required_surfaces",
            (not assistant_viea_trace_required(intent))
            or (
                bool(selected_context.get("ready"))
                and vcm_governor.get("ready") is True
                and assistant_viea_trace_complete(assistant_viea_trace)
                and materialized_view_receipt.get("ready") is True
                and route_validator_receipt.get("ready") is True
                and private_verifier_receipt.get("ready") is True
                and (tool_evidence.get("required") is not True or tool_evidence.get("active") is True)
                and ((not feedback) or event_state in {"GREEN", "YELLOW"})
            ),
            {
                "intent": intent,
                "vcm_ready": selected_context.get("ready"),
                "vcm_governor_ready": vcm_governor.get("ready"),
                "vcm_governor_state": vcm_governor.get("trigger_state"),
                "vcm_adequacy_state": vcm_governor.get("adequacy_state"),
                "trace_complete": assistant_viea_trace_complete(assistant_viea_trace),
                "materialized_view_ready": materialized_view_receipt.get("ready"),
                "route_validator_ready": route_validator_receipt.get("ready"),
                "private_verifier_ready": private_verifier_receipt.get("ready"),
                "tool_evidence_active": tool_evidence.get("active"),
                "dogfood_event_state": event_state,
            },
            "hard",
        ),
        gate(
            "assistant_viea_trace_avoids_raw_prompt_storage",
            not assistant_trace_has_raw_prompt(assistant_viea_trace),
            {
                "record_count": len(assistant_viea_trace),
                "raw_prompt_stored": assistant_trace_has_raw_prompt(assistant_viea_trace),
            },
            "hard",
        ),
        gate("raw_text_training_disabled", config.get("raw_text_training_allowed") is False, config.get("raw_text_training_allowed"), "hard"),
        gate("public_benchmark_training_disabled", config.get("public_benchmark_training_allowed") is False, config.get("public_benchmark_training_allowed"), "hard"),
        gate("runtime_external_inference_disabled", config.get("external_inference_at_runtime_allowed") is False, config.get("external_inference_at_runtime_allowed"), "hard"),
        gate("fallback_returns_disabled", config.get("fallback_returns_allowed") is False, config.get("fallback_returns_allowed"), "hard"),
        gate(
            "bounded_effect_canary_complete_when_requested",
            effect_canary.get("enabled") is not True
            or (
                effect_canary.get("ready") is True
                and get_path(effect_canary, ["observation", "matches_intent"], False) is True
                and get_path(effect_canary, ["rollback", "complete"], False) is True
                and get_path(effect_canary, ["rollback", "residual_count"], 1) == 0
                and len(
                    {
                        str(effect_canary.get("proposer_id")),
                        str(effect_canary.get("observer_id")),
                        str(effect_canary.get("evaluator_id")),
                    }
                )
                == 3
            ),
            effect_canary,
            "hard",
        ),
    ]


def artifact_refs_for_feedback(
    out: str,
    checkpoint_out: Path,
    code_private_probe: dict[str, Any],
    tool_evidence: dict[str, Any],
    procedural_default_route: dict[str, Any],
) -> list[str]:
    refs = [rel(resolve(out)), rel(checkpoint_out)]
    if code_private_probe.get("active") and code_private_probe.get("report"):
        refs.append(str(code_private_probe.get("report")))
    if tool_evidence.get("active"):
        for key in ["report", "trace", "ablation", "artifact_graph"]:
            if tool_evidence.get(key):
                refs.append(str(tool_evidence.get(key)))
    if procedural_default_route.get("active") and procedural_default_route.get("report"):
        refs.append(str(procedural_default_route.get("report")))
    return refs


def conversation_event(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "policy": "project_theseus_assistant_conversation_event_v0",
        "created_utc": report.get("created_utc"),
        "session_id": summary.get("session_id"),
        "intent": summary.get("intent"),
        "assistant_lane": summary.get("assistant_lane"),
        "prompt_sha256": get_path(report, ["inputs", "prompt_sha256"], ""),
        "assistant_text_sha256": sha256_text(str(report.get("assistant_text") or "")),
        "assistant_text_chars": summary.get("assistant_text_chars"),
        "checkpoint_history_turns_loaded": summary.get("checkpoint_history_turns_loaded"),
        "checkpoint_session_path": summary.get("checkpoint_session_path"),
        "feedback": summary.get("feedback"),
        "assistant_trace_schema": summary.get("assistant_trace_schema"),
        "assistant_trace_schema_sha256": summary.get("assistant_trace_schema_sha256"),
        "assistant_trace_schema_ready": summary.get("assistant_trace_schema_ready"),
        "dogfood_event_written": summary.get("dogfood_event_written"),
        "training_rows_written": summary.get("dogfood_training_rows_written"),
        "vcm_context_ready": summary.get("vcm_context_ready"),
        "vcm_context_governor_ready": summary.get("vcm_context_governor_ready"),
        "vcm_context_adequacy_state": summary.get("vcm_context_adequacy_state"),
        "vcm_mission_brief_status": summary.get("vcm_mission_brief_status"),
        "vcm_deletion_closure_status": summary.get("vcm_deletion_closure_status"),
        "tool_evidence_state": summary.get("tool_evidence_state"),
        "tool_evidence_result_count": summary.get("tool_evidence_result_count"),
        "tool_evidence_trace": summary.get("tool_evidence_trace"),
        "tool_evidence_report": get_path(report, ["tool_evidence", "report"], ""),
        "tool_evidence_trace_sha256": sha256_text(str(summary.get("tool_evidence_trace") or "")),
        "procedural_default_route_active": summary.get("procedural_default_route_active"),
        "procedural_default_route_ready": summary.get("procedural_default_route_ready"),
        "procedural_default_route_state": summary.get("procedural_default_route_state"),
        "procedural_default_route_id": summary.get("procedural_default_route_id"),
        "procedural_default_route_guard_armed": summary.get("procedural_default_route_guard_armed"),
        "procedural_default_route_learned_generation_claim_allowed": summary.get("procedural_default_route_learned_generation_claim_allowed"),
        "procedural_default_route_selection_matched": summary.get("procedural_default_route_selection_matched"),
        "procedural_default_route_scope": summary.get("procedural_default_route_scope"),
        "viea_materialized_view_ready": summary.get("viea_materialized_view_ready"),
        "viea_materialized_view_record_count": summary.get("viea_materialized_view_record_count"),
        "private_verifier_spine_ready": summary.get("private_verifier_spine_ready"),
        "private_verifier_spine_record_count": summary.get("private_verifier_spine_record_count"),
        "latest_public_run": summary.get("latest_public_run"),
        "latest_public_cards": summary.get("latest_public_cards"),
        "latest_public_score": summary.get("latest_public_score"),
        "latest_public_task_count": summary.get("latest_public_task_count"),
        "latest_public_measurement_kind": summary.get("latest_public_measurement_kind"),
        "latest_public_dominant_residual": summary.get("latest_public_dominant_residual"),
        "teacher_distillation_gate_state": summary.get("teacher_distillation_gate_state"),
        "teacher_distillation_allowed": summary.get("teacher_distillation_allowed"),
        "teacher_accepted_row_share": summary.get("teacher_accepted_row_share"),
        "teacher_runtime_external_tokens_forbidden": summary.get("teacher_runtime_external_tokens_forbidden"),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
    }


def conversation_events_path(path: str) -> Path:
    candidate = resolve(path)
    try:
        if candidate.resolve() == DOGFOOD_EVENTS.resolve():
            return DEFAULT_EVENTS
    except FileNotFoundError:
        if candidate == DOGFOOD_EVENTS:
            return DEFAULT_EVENTS
    return candidate


def redacted_intent_summary(prompt: str, intent: str) -> str:
    words = re.findall(r"[A-Za-z0-9_./:-]+", prompt.lower())
    hints = []
    for token in words[:80]:
        if len(token) < 3:
            continue
        if any(key in token for key in ["token", "secret", "password", "key"]):
            hints.append("[secret_like]")
        elif token in {"code", "python", "rust", "test", "benchmark", "memory", "vcm", "chat", "assistant", "plan", "tool", "hive"}:
            hints.append(token)
    unique = []
    for item in hints:
        if item not in unique:
            unique.append(item)
    return f"intent={intent}; hints={','.join(unique[:12]) or 'general'}; prompt_hash={sha256_text(prompt)[:12]}"


def load_assistant_trace_schema(config: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(config.get("assistant_trace_schema") or rel(DEFAULT_ASSISTANT_TRACE_SCHEMA))
    path = resolve(raw_path)
    schema = read_json(path, {})
    if not isinstance(schema, dict):
        schema = {}
    schema = dict(schema)
    schema["path"] = rel(path)
    schema["exists"] = path.exists()
    schema["sha256"] = sha256_file(path) if path.exists() else ""
    return schema


def allowed_feedback_values(schema: dict[str, Any]) -> set[str]:
    outcomes = {str(item).strip().lower() for item in schema.get("allowed_outcomes", []) if str(item).strip()}
    if not outcomes:
        outcomes = {item for item in DEFAULT_ALLOWED_FEEDBACK if item}
    if schema.get("empty_outcome_allowed") is not False:
        outcomes.add("")
    return outcomes


def missing_required_outcomes(schema: dict[str, Any]) -> list[str]:
    observed = {str(item).strip().lower() for item in schema.get("allowed_outcomes", []) if str(item).strip()}
    required = {item for item in DEFAULT_ALLOWED_FEEDBACK if item}
    return sorted(required - observed)


def missing_required_event_fields(schema: dict[str, Any]) -> list[str]:
    observed = {str(item).strip() for item in schema.get("required_event_fields", []) if str(item).strip()}
    required = {
        "session_id",
        "intent",
        "assistant_lane",
        "prompt_sha256",
        "assistant_text_sha256",
        "feedback",
        "dogfood_event_written",
        "vcm_context_ready",
        "vcm_context_governor_ready",
        "viea_materialized_view_ready",
        "private_verifier_spine_ready",
        "public_training_rows_written",
        "external_inference_calls",
    }
    return sorted(required - observed)


def assistant_trace_schema_ready(schema: dict[str, Any]) -> bool:
    boundaries = schema.get("boundaries") if isinstance(schema.get("boundaries"), dict) else {}
    return (
        schema.get("exists") is True
        and schema.get("policy") == "project_theseus_assistant_trace_schema_v1"
        and bool(schema.get("schema_version"))
        and not missing_required_outcomes(schema)
        and not missing_required_event_fields(schema)
        and boundaries.get("raw_text_training_allowed") is False
        and boundaries.get("public_benchmark_training_allowed") is False
        and boundaries.get("external_inference_at_runtime_allowed") is False
        and boundaries.get("fallback_returns_allowed") is False
    )


def normalize_feedback(value: str, allowed_feedback: set[str]) -> str:
    feedback = str(value or "").strip().lower()
    return feedback if feedback in allowed_feedback else "missed"


def run_command(command_id: str, command: list[str], timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=max(1, timeout_seconds))
        return {
            "id": command_id,
            "command": command,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-1200:],
            "stderr_tail": result.stderr[-1200:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "id": command_id,
            "command": command,
            "returncode": 124,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-1200:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-1200:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Theseus Assistant Runtime",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- intent: `{summary.get('intent')}`",
        f"- session: `{summary.get('session_id')}`",
        f"- lane: `{summary.get('assistant_lane')}`",
        f"- VCM: `{summary.get('vcm_context_ready')}` pages=`{summary.get('vcm_selected_page_count')}`",
        f"- VCM governor: `{summary.get('vcm_context_governor_state')}` ready=`{summary.get('vcm_context_governor_ready')}` adequacy=`{summary.get('vcm_context_adequacy_state')}` mission=`{summary.get('vcm_mission_brief_status')}` deletion=`{summary.get('vcm_deletion_closure_status')}`",
        f"- code probe: `{summary.get('code_private_probe_state')}` safe=`{summary.get('code_private_probe_safe_boundary')}` selected_pass=`{summary.get('code_private_probe_selected_pass_rate')}` pass_if_any=`{summary.get('code_private_probe_pass_if_any_rate')}` integrity_mismatches=`{summary.get('code_private_probe_candidate_integrity_mismatch_count')}`",
        f"- tool evidence: `{summary.get('tool_evidence_state')}` results=`{summary.get('tool_evidence_result_count')}` exact_solve=`{summary.get('tool_evidence_exact_solve_rate')}`",
        f"- procedural default route: active=`{summary.get('procedural_default_route_active')}` ready=`{summary.get('procedural_default_route_ready')}` matched=`{summary.get('procedural_default_route_selection_matched')}` route=`{summary.get('procedural_default_route_id')}` scope=`{summary.get('procedural_default_route_scope')}` guard=`{summary.get('procedural_default_route_guard_armed')}` learned_claim_allowed=`{summary.get('procedural_default_route_learned_generation_claim_allowed')}`",
        f"- VIEA trace: required=`{summary.get('assistant_viea_trace_required')}` complete=`{summary.get('assistant_viea_trace_complete')}` records=`{summary.get('assistant_viea_trace_record_count')}` out=`{summary.get('assistant_viea_trace_out')}`",
        f"- VIEA materialized view: ready=`{summary.get('viea_materialized_view_ready')}` records=`{summary.get('viea_materialized_view_record_count')}` claims=`{summary.get('viea_materialized_view_claim_count')}`",
        f"- private verifier receipt: ready=`{summary.get('private_verifier_spine_ready')}` state=`{summary.get('private_verifier_spine_state')}` records=`{summary.get('private_verifier_spine_record_count')}`",
        f"- latest public: `{summary.get('latest_public_run')}` score=`{summary.get('latest_public_score')}` tasks=`{summary.get('latest_public_task_count')}` kind=`{summary.get('latest_public_measurement_kind')}`",
        f"- answer chars: `{summary.get('assistant_text_chars')}`",
        f"- feedback: `{summary.get('feedback')}`",
        f"- assistant trace schema: `{summary.get('assistant_trace_schema')}` ready=`{summary.get('assistant_trace_schema_ready')}`",
        f"- dogfood event written: `{summary.get('dogfood_event_written')}`",
        f"- dogfood rows written: `{summary.get('dogfood_training_rows_written')}`",
        f"- teacher distillation: `{summary.get('teacher_distillation_gate_state')}` allowed=`{summary.get('teacher_distillation_allowed')}` share=`{summary.get('teacher_accepted_row_share')}` runtime_tokens_forbidden=`{summary.get('teacher_runtime_external_tokens_forbidden')}`",
        "",
        "## Answer",
        "",
        str(report.get("assistant_text") or "").strip() or "(empty)",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    return "\n".join(lines).rstrip() + "\n"


def has_any(text: str, tokens: list[str]) -> bool:
    return any(token in text for token in tokens)


def command_arg(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(command):
        return ""
    return str(command[index + 1])


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def unique_strings(values: list[Any]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{stable_hash(parts)[:16]}"


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def read_json(path: Path, default: Any | None = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())[:96].strip("._-")
    return slug or "local_assistant"


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
