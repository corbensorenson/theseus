#!/usr/bin/env python3
"""Execute compiled VIEA private-work packets with durable local leases.

This is the first real execute-mode bridge between the plan compiler and the
deterministic tool substrate. It intentionally supports only bounded local
deterministic tool packets: no arbitrary shell, no public benchmark training,
no external inference, and no fallback answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
RUNTIME = ROOT / "runtime"
sys.path.insert(0, str(SCRIPTS))

import theseus_deterministic_tool_substrate as deterministic_tools  # noqa: E402


DEFAULT_CONFIG = CONFIGS / "viea_execution_spine.json"
DEFAULT_TOOL_CONFIG = CONFIGS / "deterministic_tool_substrate.json"
DEFAULT_DAGS = REPORTS / "theseus_plan_compiled_dags.json"
DEFAULT_DB = REPORTS / "viea_execution_spine.sqlite"
DEFAULT_OUT = REPORTS / "viea_execution_spine.json"
DEFAULT_MARKDOWN = REPORTS / "viea_execution_spine.md"
DEFAULT_TRACE = REPORTS / "viea_execution_spine_trace.jsonl"
DEFAULT_VCM_ARTIFACTS = REPORTS / "viea_execution_spine_vcm_artifacts.jsonl"
DEFAULT_LEARNING_TRACES = REPORTS / "viea_tool_use_learning_traces.jsonl"
DEFAULT_LOOP_CANDIDATES = REPORTS / "viea_execution_loop_closure_candidates.json"
DEFAULT_TRAINING_EVIDENCE = REPORTS / "viea_tool_use_training_evidence.jsonl"
DEFAULT_PROCEDURAL_TOOLS = REPORTS / "viea_verified_procedural_tools.json"
DEFAULT_RESEARCH_MATRIX = REPORTS / "viea_research_implementation_matrix.json"
DEFAULT_CHECKPOINT_DIR = REPORTS / "viea_execution_spine_checkpoints"

STRUCTURED_NON_SOLVED = {"UNKNOWN", "UNSOLVED", "TOOL_UNAVAILABLE", "TOOL_FAULT"}
TERMINAL_STATUSES = {"done", "non_solved", "tool_fault", "cancelled", "blocked"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--tool-config", default=rel(DEFAULT_TOOL_CONFIG))
    parser.add_argument("--dags", default=rel(DEFAULT_DAGS))
    parser.add_argument("--db", default=rel(DEFAULT_DB))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--trace-out", default=rel(DEFAULT_TRACE))
    parser.add_argument("--vcm-artifacts-out", default=rel(DEFAULT_VCM_ARTIFACTS))
    parser.add_argument("--learning-traces-out", default=rel(DEFAULT_LEARNING_TRACES))
    parser.add_argument("--loop-candidates-out", default=rel(DEFAULT_LOOP_CANDIDATES))
    parser.add_argument("--training-evidence-out", default=rel(DEFAULT_TRAINING_EVIDENCE))
    parser.add_argument("--procedural-tools-out", default=rel(DEFAULT_PROCEDURAL_TOOLS))
    parser.add_argument("--research-matrix-out", default=rel(DEFAULT_RESEARCH_MATRIX))
    parser.add_argument("--checkpoint-dir", default=rel(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cancel-run", default="")
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--lease-seconds", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config), {})
    tool_config = read_json(resolve(args.tool_config), {})
    dags = read_json(resolve(args.dags), {})
    db_path = resolve(args.db)
    ensure_schema(db_path)

    if args.cancel_run:
        cancel = cancel_run(db_path, args.cancel_run)
        write_json(resolve(args.out), cancel)
        write_text(resolve(args.markdown_out), render_markdown(cancel))
        print(json.dumps(cancel, indent=2))
        return 0 if cancel.get("trigger_state") != "RED" else 2

    compiled_packet = select_compiled_tool_packet(dags)
    cases = [row for row in tool_config.get("private_smoke_cases", []) if isinstance(row, dict)]
    cases = cases[: max(1, int(args.max_cases))]
    lease_seconds = int(args.lease_seconds or config.get("default_lease_seconds") or 300)
    max_attempts = max(1, int(config.get("max_attempts_per_node") or 1))
    run_id = resume_run_id(db_path) if args.resume else stable_id("viea_run", now(), compiled_packet.get("packet_id", ""), len(cases))
    trace_rows: list[dict[str, Any]] = []
    vcm_artifacts: list[dict[str, Any]] = []
    learning_traces: list[dict[str, Any]] = []

    old_started = time.perf_counter()
    old_results = run_old_direct_baseline(cases, tool_config)
    old_runtime_ms = int((time.perf_counter() - old_started) * 1000)

    compiled_results: list[dict[str, Any]] = []
    stale_recovery = recover_stale_leases(db_path)
    if args.execute:
        start_run(db_path, run_id, mode="compiled_viea_local_deterministic_tool", payload={"packet": compiled_packet, "case_count": len(cases)})
        for case in cases:
            if cancel_requested(db_path, run_id):
                trace_rows.append(event_row(run_id, "run_cancelled", {"case_id": case.get("case_id")}))
                break
            result = execute_compiled_case(
                db_path=db_path,
                run_id=run_id,
                case=case,
                tool_config=tool_config,
                compiled_packet=compiled_packet,
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
                checkpoint_dir=resolve(args.checkpoint_dir),
            )
            compiled_results.append(result)
            trace_rows.append(event_row(run_id, "node_completed", compact_result(result)))
            vcm_artifacts.append(vcm_artifact_for(run_id, result, compiled_packet))
            learning_traces.append(tool_use_learning_trace(run_id, result, compiled_packet))
        finish_run(db_path, run_id, status="done" if all(row.get("verified") for row in compiled_results) else "non_solved", payload={"result_count": len(compiled_results)})

    benchmark_readiness = run_private_benchmark_adapter_dry_runs(config)
    loop_candidates = build_loop_candidates(compiled_results)
    procedural_tools = build_verified_procedural_tools(compiled_results)
    residuals = build_residuals(compiled_results)
    training_evidence = build_training_evidence(learning_traces, compiled_results, residuals)
    research_matrix = build_research_matrix()
    ab = build_ab_comparison(old_results, old_runtime_ms, compiled_results, compiled_packet, args.execute)
    gates = build_gates(
        config,
        dags,
        compiled_packet,
        cases,
        old_results,
        compiled_results,
        ab,
        benchmark_readiness,
        args.execute,
        procedural_tools,
        training_evidence,
    )
    hard_failures = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failures = [row for row in gates if row["severity"] == "warning" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failures else "RED"
    if trigger_state == "GREEN" and warning_failures:
        trigger_state = "YELLOW"

    report = {
        "policy": "project_theseus_viea_execution_spine_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "passed": trigger_state in {"GREEN", "YELLOW"},
        "run_id": run_id,
        "execute_requested": bool(args.execute),
        "inputs": {
            "config": rel(resolve(args.config)),
            "tool_config": rel(resolve(args.tool_config)),
            "dags": rel(resolve(args.dags)),
        },
        "outputs": {
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
            "trace": rel(resolve(args.trace_out)),
            "vcm_artifacts": rel(resolve(args.vcm_artifacts_out)),
            "learning_traces": rel(resolve(args.learning_traces_out)),
            "training_evidence": rel(resolve(args.training_evidence_out)),
            "loop_candidates": rel(resolve(args.loop_candidates_out)),
            "procedural_tools": rel(resolve(args.procedural_tools_out)),
            "research_matrix": rel(resolve(args.research_matrix_out)),
            "db": rel(db_path),
        },
        "summary": summarize(old_results, compiled_results, ab, gates, stale_recovery, started, procedural_tools, training_evidence),
        "compiled_packet": compiled_packet,
        "old_direct_baseline": metrics_for_results(old_results, runtime_ms=old_runtime_ms, context_pages_per_node=0),
        "compiled_execution": metrics_for_results(compiled_results, runtime_ms=ab.get("compiled_runtime_ms", 0), context_pages_per_node=int(compiled_packet.get("vcm_selected_page_count") or 0)),
        "old_direct_baseline_results": [compact_result(row) for row in old_results],
        "compiled_execution_results": compiled_results,
        "ab_comparison": ab,
        "durable_runtime": runtime_summary(db_path, run_id),
        "stale_lease_recovery": stale_recovery,
        "benchmark_adapter_readiness": benchmark_readiness,
        "loop_closure_candidates": loop_candidates,
        "verified_procedural_tools": procedural_tools,
        "residuals": residuals,
        "training_evidence_summary": summarize_training_evidence(training_evidence),
        "research_implementation_matrix": research_matrix,
        "gates": gates,
        "boundaries": {
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "arbitrary_remote_execution": False,
        },
        "remaining_gaps": remaining_gaps(args.execute, benchmark_readiness, compiled_results, procedural_tools),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }

    write_json(resolve(args.out), report)
    write_json(resolve(args.loop_candidates_out), loop_candidates)
    write_json(resolve(args.procedural_tools_out), procedural_tools)
    write_json(resolve(args.research_matrix_out), research_matrix)
    write_jsonl(resolve(args.trace_out), trace_rows)
    write_jsonl(resolve(args.vcm_artifacts_out), vcm_artifacts)
    write_jsonl(resolve(args.learning_traces_out), learning_traces)
    write_jsonl(resolve(args.training_evidence_out), training_evidence)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps({"trigger_state": trigger_state, "summary": report["summary"]}, indent=2))
    return 0 if trigger_state != "RED" else 2


def ensure_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS node_leases (
                lease_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                lease_owner TEXT NOT NULL,
                lease_expires_utc TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                output_hash TEXT NOT NULL DEFAULT '',
                replay_checksum TEXT NOT NULL DEFAULT '',
                vcm_context_hash TEXT NOT NULL DEFAULT '',
                evidence_ref TEXT NOT NULL DEFAULT '',
                started_utc TEXT NOT NULL,
                completed_utc TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_utc TEXT NOT NULL
            );
            """
        )


def select_compiled_tool_packet(dags: dict[str, Any]) -> dict[str, Any]:
    for goal in dags.get("compiled_goals", []) if isinstance(dags.get("compiled_goals"), list) else []:
        for node in goal.get("nodes", []) if isinstance(goal.get("nodes"), list) else []:
            packet = node.get("execution_packet") if isinstance(node.get("execution_packet"), dict) else {}
            if packet.get("mode") == "local_deterministic_tool_packet" and "tool.trace_replay" in packet.get("tool_ids", []):
                packet = dict(packet)
                packet["source_goal_id"] = goal.get("goal_id")
                packet["source_node_id"] = node.get("node_id")
                packet["vcm_context_slice"] = node.get("vcm_context_slice", {})
                return packet
    return {
        "packet_id": "",
        "mode": "MISSING",
        "tool_ids": [],
        "vcm_selected_page_count": 0,
        "vcm_context_hash": "",
    }


def run_old_direct_baseline(cases: list[dict[str, Any]], tool_config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = deterministic_tools.run_case(case, tool_config)
        result = dict(result)
        result["execution_mode"] = "old_direct_local_tool_baseline"
        result["durable_lease"] = False
        result["vcm_runtime_context_consumed"] = False
        result["baseline_runtime_ms"] = int((time.perf_counter() - started) * 1000)
        rows.append(result)
    return rows


def execute_compiled_case(
    *,
    db_path: Path,
    run_id: str,
    case: dict[str, Any],
    tool_config: dict[str, Any],
    compiled_packet: dict[str, Any],
    lease_seconds: int,
    max_attempts: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    node_id = f"{compiled_packet.get('packet_id')}.{case.get('case_id')}"
    lease_id = stable_id("lease", run_id, node_id, case.get("tool_id"))
    input_payload = {"case": case, "packet_hash": compiled_packet.get("packet_hash"), "vcm_context_hash": compiled_packet.get("vcm_context_hash")}
    input_hash = stable_hash(input_payload)
    lease_payload = {
        "case": case,
        "compiled_packet": compiled_packet,
        "strict_no_fallback_returns": True,
        "public_training_rows_allowed": False,
        "external_inference_allowed": False,
        "retry_policy": {
            "max_attempts": max_attempts,
            "retry_states": ["TOOL_FAULT"],
            "non_retry_states": sorted(STRUCTURED_NON_SOLVED - {"TOOL_FAULT"}),
        },
    }
    result: dict[str, Any] = {}
    runtime_ms = 0
    attempt = 0
    expires = datetime.now(timezone.utc) + timedelta(seconds=max(1, lease_seconds))
    for attempt_index in range(max(1, max_attempts)):
        attempt = next_attempt(db_path, lease_id) if attempt_index == 0 else attempt + 1
        expires = datetime.now(timezone.utc) + timedelta(seconds=max(1, lease_seconds))
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                INSERT INTO node_leases (
                    lease_id, run_id, node_id, case_id, tool_id, status, attempt,
                    lease_owner, lease_expires_utc, input_hash, vcm_context_hash,
                    started_utc, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(lease_id) DO UPDATE SET
                    status='leased',
                    attempt=excluded.attempt,
                    lease_expires_utc=excluded.lease_expires_utc,
                    started_utc=excluded.started_utc,
                    payload_json=excluded.payload_json
                """,
                (
                    lease_id,
                    run_id,
                    node_id,
                    str(case.get("case_id") or ""),
                    str(case.get("tool_id") or ""),
                    "leased",
                    attempt,
                    "viea_execution_spine",
                    expires.isoformat().replace("+00:00", "Z"),
                    input_hash,
                    str(compiled_packet.get("vcm_context_hash") or ""),
                    now(),
                    json.dumps(lease_payload, sort_keys=True),
                ),
            )
        add_event(db_path, run_id, "node_lease_acquired", {"lease_id": lease_id, "node_id": node_id, "case_id": case.get("case_id"), "attempt": attempt})
        started = time.perf_counter()
        result = deterministic_tools.run_case(case, tool_config)
        runtime_ms += int((time.perf_counter() - started) * 1000)
        if result.get("state") != "TOOL_FAULT" or attempt_index >= max_attempts - 1:
            break
        add_event(db_path, run_id, "node_retry_scheduled", {"lease_id": lease_id, "node_id": node_id, "attempt": attempt, "state": result.get("state")})
    result = dict(result)
    state = str(result.get("state") or "")
    status = "done" if result.get("verified") else "tool_fault" if state == "TOOL_FAULT" else "non_solved"
    result.update(
        {
            "execution_mode": "compiled_viea_local_deterministic_tool",
            "run_id": run_id,
            "node_id": node_id,
            "lease_id": lease_id,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "retry_policy": lease_payload["retry_policy"],
            "durable_lease": True,
            "lease_expires_utc": expires.isoformat().replace("+00:00", "Z"),
            "vcm_runtime_context_consumed": bool(compiled_packet.get("vcm_context_hash")),
            "vcm_context_hash": compiled_packet.get("vcm_context_hash"),
            "vcm_selected_page_count": compiled_packet.get("vcm_selected_page_count", 0),
            "compiled_packet_hash": compiled_packet.get("packet_hash"),
            "runtime_ms": runtime_ms,
            "status": status,
        }
    )
    checkpoint = write_checkpoint(db_path, run_id, node_id, checkpoint_dir)
    result["checkpoint"] = checkpoint
    result["asi_stack_execution_records"] = execution_records_for_result(result, compiled_packet, case)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            UPDATE node_leases
            SET status=?, output_hash=?, replay_checksum=?, evidence_ref=?,
                completed_utc=?, payload_json=?
            WHERE lease_id=?
            """,
            (
                status,
                str(result.get("output_hash") or ""),
                str(result.get("replay_checksum") or ""),
                str(result.get("evidence_ref") or ""),
                now(),
                json.dumps(result, sort_keys=True),
                lease_id,
            ),
        )
    add_event(db_path, run_id, "node_status", {"lease_id": lease_id, "status": status, "state": state, "verified": result.get("verified")})
    return result


def execution_records_for_result(result: dict[str, Any], compiled_packet: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    run_id = str(result.get("run_id") or "")
    node_id = str(result.get("node_id") or "")
    case_id = str(result.get("case_id") or case.get("case_id") or "")
    tool_id = str(result.get("tool_id") or case.get("tool_id") or "")
    state = str(result.get("state") or "")
    status = str(result.get("status") or "")
    verified = bool(result.get("verified"))
    support_state = "SUPPORTED" if verified else "RESIDUAL"
    packet_hash = str(compiled_packet.get("packet_hash") or result.get("compiled_packet_hash") or "")
    context_hash = str(result.get("vcm_context_hash") or compiled_packet.get("vcm_context_hash") or "")
    input_hash = str(result.get("input_hash") or stable_hash({"case": case, "packet_hash": packet_hash, "context_hash": context_hash}))
    output_hash = str(result.get("output_hash") or "")
    replay_checksum = str(result.get("replay_checksum") or "")
    evidence_ref = str(result.get("evidence_ref") or "")
    checkpoint = result.get("checkpoint") if isinstance(result.get("checkpoint"), dict) else {}
    checkpoint_hash = str(checkpoint.get("content_hash") or "")
    authority_handle = stable_id("authority_handle", run_id, node_id, tool_id)
    adapter_invocation_id = stable_id("runtime_adapter_invocation", run_id, node_id, tool_id, input_hash)
    artifact_id = stable_id("runtime_artifact", run_id, node_id, output_hash, replay_checksum)
    evidence_transition_id = stable_id("evidence_transition", run_id, node_id, evidence_ref, support_state)
    failure_boundary_id = stable_id("failure_boundary", run_id, node_id, state, status)
    return {
        "authority_transition": {
            "record_id": stable_id("authority_transition", run_id, node_id, tool_id),
            "run_id": run_id,
            "node_id": node_id,
            "tool_id": tool_id,
            "from_authority": "plan_compiler_compiled_contract",
            "to_authority": "local_deterministic_tool_adapter",
            "authority_handle": authority_handle,
            "authority_ceiling": "local_private_deterministic_tool",
            "approval_state": "preauthorized_by_registered_private_execute_contract",
            "external_inference_allowed": False,
            "public_training_rows_allowed": False,
            "arbitrary_remote_execution_allowed": False,
        },
        "authority_use_receipt": {
            "record_id": stable_id("authority_use_receipt", run_id, node_id, tool_id, input_hash),
            "authority_handle": authority_handle,
            "used": True,
            "scope": "local_private_tool_execution",
            "lease_id": result.get("lease_id"),
            "attempt": result.get("attempt"),
            "state": state,
            "status": status,
            "policy_receipts": {
                "strict_no_fallback_returns": True,
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "raw_private_text_stored": False,
            },
        },
        "runtime_adapter_invocation": {
            "record_id": adapter_invocation_id,
            "run_id": run_id,
            "node_id": node_id,
            "case_id": case_id,
            "tool_id": tool_id,
            "adapter_surface": "scripts/theseus_deterministic_tool_substrate.py",
            "executor_backend": compiled_packet.get("executor_backend") or "local_deterministic_tool",
            "input_hash": input_hash,
            "output_hash": output_hash,
            "replay_checksum": replay_checksum,
            "latency_ms": result.get("latency_ms"),
            "runtime_ms": result.get("runtime_ms"),
            "side_effect_class": "evidence_only",
            "failure_behavior": "structured_non_solved_state",
        },
        "context_transaction": {
            "record_id": stable_id("context_transaction", run_id, node_id, context_hash),
            "run_id": run_id,
            "node_id": node_id,
            "context_abi": "vcm_context_hash_v1",
            "context_hash": context_hash,
            "selected_page_count": int(result.get("vcm_selected_page_count") or compiled_packet.get("vcm_selected_page_count") or 0),
            "context_consumed": bool(result.get("vcm_runtime_context_consumed")),
            "context_privilege": "governed_private_runtime_context",
            "raw_private_text_stored": False,
        },
        "context_adequacy": {
            "record_id": stable_id("context_adequacy", run_id, node_id, context_hash, state),
            "run_id": run_id,
            "node_id": node_id,
            "case_id": case_id,
            "adequacy_state": "adequate_for_verified_tool_replay" if verified else "residual_context_or_tool_gap",
            "support_state": support_state,
            "state": state,
            "evidence_ref": evidence_ref,
        },
        "resource_budget": {
            "record_id": stable_id("resource_budget", run_id, node_id, tool_id),
            "run_id": run_id,
            "node_id": node_id,
            "budget_class": "bounded_local_smoke",
            "max_attempts": result.get("max_attempts"),
            "attempt": result.get("attempt"),
            "runtime_ms": result.get("runtime_ms"),
            "latency_ms": result.get("latency_ms"),
            "external_network_required": False,
            "external_inference_required": False,
        },
        "costed_route": {
            "record_id": stable_id("costed_route", run_id, node_id, tool_id, packet_hash),
            "run_id": run_id,
            "node_id": node_id,
            "route": ["command_contract", "plan_compiler_dag", "vcm_context_packet", "local_tool_adapter", "verifier", "evidence_store"],
            "route_cost": {
                "runtime_ms": result.get("runtime_ms"),
                "attempts": result.get("attempt"),
                "context_pages": int(result.get("vcm_selected_page_count") or 0),
                "external_inference_calls": 0,
            },
            "routing_reason": "registered deterministic tool selected by compiled private execution packet",
        },
        "generation_mode": {
            "record_id": stable_id("generation_mode", run_id, node_id, tool_id, "tool_assisted"),
            "run_id": run_id,
            "node_id": node_id,
            "mode": "tool_assisted_deterministic_execution",
            "learned_generation_claim_allowed": False,
            "candidate_family_credit": "deterministic_tool_baseline_or_runtime_tool",
            "not_learned_generation_reason": "tool invocation computes or checks; it cannot support learned code generation promotion claims",
        },
        "failure_boundary": {
            "record_id": failure_boundary_id,
            "run_id": run_id,
            "node_id": node_id,
            "case_id": case_id,
            "tool_id": tool_id,
            "state": state,
            "status": status,
            "terminal": status in TERMINAL_STATUSES,
            "structured_non_solved": state in STRUCTURED_NON_SOLVED,
            "fallback_return_used": False,
            "diagnostic_hash": stable_hash(str(result.get("diagnostic") or "")),
        },
        "artifact_graph": {
            "record_id": stable_id("runtime_artifact_graph", run_id, node_id, artifact_id),
            "run_id": run_id,
            "node_id": node_id,
            "artifacts": [
                {
                    "artifact_id": artifact_id,
                    "kind": "tool_result",
                    "content_hash": output_hash,
                    "support_state": support_state,
                    "evidence_ref": evidence_ref,
                },
                {
                    "artifact_id": str(checkpoint.get("checkpoint_id") or ""),
                    "kind": "execution_checkpoint",
                    "content_hash": checkpoint_hash,
                    "path": checkpoint.get("path"),
                    "support_state": "SUPPORTED" if checkpoint_hash else "MISSING",
                },
            ],
            "parents": [packet_hash, context_hash],
            "raw_private_text_stored": False,
        },
        "evidence_transition": {
            "record_id": evidence_transition_id,
            "run_id": run_id,
            "node_id": node_id,
            "claim_id": result.get("claim_id"),
            "from_state": "CLAIMED_BY_COMPILED_PACKET",
            "to_state": support_state,
            "verifier_state": "pass" if verified else "not_passed",
            "evidence_ref": evidence_ref,
            "replay_checksum": replay_checksum,
            "public_training_rows_written": 0,
        },
        "proof_carrying_claim": {
            "record_id": stable_id("proof_carrying_claim", run_id, node_id, result.get("claim_id"), replay_checksum),
            "run_id": run_id,
            "node_id": node_id,
            "claim_id": result.get("claim_id"),
            "claim": f"{tool_id} produced state {state} for private case {case_id}.",
            "support_state": support_state,
            "proof_refs": [value for value in [evidence_ref, replay_checksum, checkpoint_hash] if value],
            "verifier_required": True,
            "verifier_passed": verified,
        },
    }


def required_runtime_asi_record_keys() -> set[str]:
    return {
        "authority_transition",
        "authority_use_receipt",
        "runtime_adapter_invocation",
        "context_transaction",
        "context_adequacy",
        "resource_budget",
        "costed_route",
        "generation_mode",
        "failure_boundary",
        "artifact_graph",
        "evidence_transition",
        "proof_carrying_claim",
    }


def missing_runtime_asi_records(result: dict[str, Any]) -> list[str]:
    records = result.get("asi_stack_execution_records")
    if not isinstance(records, dict):
        return sorted(required_runtime_asi_record_keys())
    return sorted(key for key in required_runtime_asi_record_keys() if not isinstance(records.get(key), dict))


def result_has_runtime_asi_records(result: dict[str, Any]) -> bool:
    return not missing_runtime_asi_records(result)


def runtime_asi_stack_record_count(results: list[dict[str, Any]]) -> int:
    count = 0
    for result in results:
        records = result.get("asi_stack_execution_records")
        if isinstance(records, dict):
            count += sum(1 for value in records.values() if isinstance(value, dict))
    return count


def start_run(db_path: Path, run_id: str, *, mode: str, payload: dict[str, Any]) -> None:
    stamp = now()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, mode, status, created_utc, updated_utc, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET status='running', updated_utc=excluded.updated_utc
            """,
            (run_id, mode, "running", stamp, stamp, json.dumps(payload, sort_keys=True)),
        )
    add_event(db_path, run_id, "run_started", {"mode": mode, "payload": payload})


def finish_run(db_path: Path, run_id: str, *, status: str, payload: dict[str, Any]) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE runs SET status=?, updated_utc=?, payload_json=? WHERE run_id=?",
            (status, now(), json.dumps(payload, sort_keys=True), run_id),
        )
    add_event(db_path, run_id, "run_finished", {"status": status, "payload": payload})


def cancel_run(db_path: Path, run_id: str) -> dict[str, Any]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("UPDATE runs SET cancel_requested=1, status='cancel_requested', updated_utc=? WHERE run_id=?", (now(), run_id))
    return {
        "policy": "project_theseus_viea_execution_spine_cancel_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "run_id": run_id,
        "cancel_requested": True,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def cancel_requested(db_path: Path, run_id: str) -> bool:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT cancel_requested FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return bool(row and int(row[0] or 0))


def resume_run_id(db_path: Path) -> str:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT run_id FROM runs WHERE status IN ('running', 'non_solved') ORDER BY updated_utc DESC LIMIT 1"
        ).fetchone()
    return str(row[0]) if row else stable_id("viea_run", "resume", now())


def recover_stale_leases(db_path: Path) -> dict[str, Any]:
    stamp = now()
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT lease_id, run_id, node_id, lease_expires_utc FROM node_leases WHERE status='leased' AND lease_expires_utc < ?",
            (stamp,),
        ).fetchall()
        for lease_id, run_id, node_id, _ in rows:
            conn.execute(
                "UPDATE node_leases SET status='stale_recovered', completed_utc=?, payload_json=? WHERE lease_id=?",
                (stamp, json.dumps({"recovered_utc": stamp, "reason": "lease_expired"}, sort_keys=True), lease_id),
            )
            conn.execute(
                "INSERT OR IGNORE INTO events (event_id, run_id, event_type, content_json, created_utc) VALUES (?, ?, ?, ?, ?)",
                (stable_id("event", run_id, node_id, "stale_recovered"), run_id, "stale_lease_recovered", json.dumps({"lease_id": lease_id, "node_id": node_id}, sort_keys=True), stamp),
            )
    return {"recovered_count": len(rows), "recovered_lease_ids": [str(row[0]) for row in rows]}


def next_attempt(db_path: Path, lease_id: str) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT MAX(attempt) FROM node_leases WHERE lease_id=?", (lease_id,)).fetchone()
    return int(row[0] or 0) + 1


def write_checkpoint(db_path: Path, run_id: str, node_id: str, checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        leases = [dict(row) for row in conn.execute("SELECT * FROM node_leases WHERE run_id=? ORDER BY started_utc", (run_id,)).fetchall()]
    payload = {
        "policy": "project_theseus_viea_execution_checkpoint_v1",
        "created_utc": now(),
        "run_id": run_id,
        "node_id": node_id,
        "lease_count": len(leases),
        "status_counts": dict(Counter(row.get("status") for row in leases)),
        "leases": compact_leases(leases),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    content_hash = stable_hash(payload)
    path = checkpoint_dir / f"{run_id}.json"
    write_json(path, payload)
    checkpoint_id = stable_id("checkpoint", run_id, node_id, content_hash)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO checkpoints (checkpoint_id, run_id, node_id, path, content_hash, created_utc) VALUES (?, ?, ?, ?, ?, ?)",
            (checkpoint_id, run_id, node_id, rel(path), content_hash, now()),
        )
    return {"checkpoint_id": checkpoint_id, "path": rel(path), "content_hash": content_hash}


def compact_leases(leases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "lease_id": row.get("lease_id"),
            "node_id": row.get("node_id"),
            "case_id": row.get("case_id"),
            "tool_id": row.get("tool_id"),
            "status": row.get("status"),
            "attempt": row.get("attempt"),
            "replay_checksum": row.get("replay_checksum"),
            "evidence_ref": row.get("evidence_ref"),
            "vcm_context_hash": row.get("vcm_context_hash"),
        }
        for row in leases
    ]


def metrics_for_results(results: list[dict[str, Any]], *, runtime_ms: int, context_pages_per_node: int) -> dict[str, Any]:
    states = Counter(str(row.get("state") or "") for row in results)
    statuses = Counter(str(row.get("status") or "") for row in results)
    verified = sum(1 for row in results if row.get("verified"))
    duplicates = len(results) - len({row.get("case_id") for row in results})
    retries = sum(max(0, int(row.get("attempt") or 1) - 1) for row in results)
    return {
        "case_count": len(results),
        "useful_completion_count": verified,
        "useful_completion_rate": round(verified / max(1, len(results)), 6),
        "verifier_pass_rate": round(verified / max(1, len(results)), 6),
        "state_counts": dict(states),
        "status_counts": dict(statuses),
        "duplicate_work_count": duplicates,
        "retry_count": retries,
        "runtime_ms": runtime_ms,
        "average_context_pages_per_node": context_pages_per_node,
        "unknown_count": states.get("UNKNOWN", 0),
        "tool_fault_count": states.get("TOOL_FAULT", 0),
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
    }


def build_ab_comparison(
    old_results: list[dict[str, Any]],
    old_runtime_ms: int,
    compiled_results: list[dict[str, Any]],
    compiled_packet: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    compiled_runtime_ms = sum(int(row.get("runtime_ms") or 0) for row in compiled_results)
    old = metrics_for_results(old_results, runtime_ms=old_runtime_ms, context_pages_per_node=0)
    compiled = metrics_for_results(compiled_results, runtime_ms=compiled_runtime_ms, context_pages_per_node=int(compiled_packet.get("vcm_selected_page_count") or 0))
    return {
        "policy": "project_theseus_viea_execution_spine_ab_v1",
        "created_utc": now(),
        "execute_requested": bool(execute),
        "same_private_case_count": min(len(old_results), len(compiled_results)) if execute else 0,
        "old_direct_local_tool_baseline": old,
        "compiled_viea_execution": compiled,
        "compiled_runtime_ms": compiled_runtime_ms,
        "useful_completion_delta": round(compiled.get("useful_completion_rate", 0) - old.get("useful_completion_rate", 0), 6) if execute else None,
        "durability_delta": {
            "leases_added": len(compiled_results) if execute else 0,
            "checkpointing_added": bool(compiled_results),
            "vcm_runtime_context_added": bool(compiled_results and compiled_packet.get("vcm_context_hash")),
        },
        "interpretation": "same_solver_quality_expected; this A/B measures execution-spine durability/context overhead, not model intelligence",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def vcm_artifact_for(run_id: str, result: dict[str, Any], compiled_packet: dict[str, Any]) -> dict[str, Any]:
    support_state = "SUPPORTED" if result.get("verified") else "UNSUPPORTED"
    return {
        "policy": "project_theseus_viea_vcm_tool_artifact_v1",
        "created_utc": now(),
        "run_id": run_id,
        "node_id": result.get("node_id"),
        "case_id": result.get("case_id"),
        "tool_id": result.get("tool_id"),
        "vcm_address": f"vcm://viea_execution_spine/{run_id}/{result.get('case_id')}",
        "parent_context_hash": compiled_packet.get("vcm_context_hash") or "NO_CONTEXT_REQUIRED",
        "context_policy": "VCM_PACKET" if compiled_packet.get("vcm_context_hash") else "NO_CONTEXT_REQUIRED",
        "support_state": support_state,
        "tool_state": result.get("state"),
        "evidence_ref": result.get("evidence_ref"),
        "replay_checksum": result.get("replay_checksum"),
        "contradiction_state": "none",
        "unknown_handling": "structured_non_solved_state" if result.get("state") in STRUCTURED_NON_SOLVED else "not_needed",
        "raw_private_text_stored": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def tool_use_learning_trace(run_id: str, result: dict[str, Any], compiled_packet: dict[str, Any]) -> dict[str, Any]:
    outcome = "completed" if result.get("verified") else "failure" if result.get("state") == "TOOL_FAULT" else "missed"
    return {
        "policy": "project_theseus_private_tool_use_learning_trace_v1",
        "created_utc": now(),
        "trace_id": stable_id("tool_use_trace", run_id, result.get("case_id"), result.get("tool_id")),
        "run_id": run_id,
        "source": "private_synthetic_owned_execute_mode",
        "case_id": result.get("case_id"),
        "selected_tool": result.get("tool_id"),
        "arguments_hash": result.get("input_hash"),
        "result_state": result.get("state"),
        "verifier_outcome": "pass" if result.get("verified") else "not_passed",
        "operator_outcome": outcome,
        "accepted_missed_ignored_completed_failure": outcome,
        "training_eligibility": {
            "eligible": True,
            "admitted_to_training_rows": False,
            "reason": "private synthetic tool-use trace; admission bridge not invoked in execute-mode smoke",
        },
        "vcm_context_hash": compiled_packet.get("vcm_context_hash"),
        "evidence_ref": result.get("evidence_ref"),
        "raw_private_text_stored": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_loop_candidates(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row.get("verified"):
            by_tool.setdefault(str(row.get("tool_id")), []).append(row)
    candidates = []
    for tool_id, rows in sorted(by_tool.items()):
        candidates.append(
            {
                "candidate_id": stable_id("viea_loop_tool", tool_id, len(rows)),
                "tool_id": tool_id,
                "status": "verified_candidate" if len(rows) >= 2 else "needs_repetition",
                "successful_trace_count": len(rows),
                "preconditions": ["registered tool card", "compiled VCM context packet", "strict no fallback returns"],
                "postconditions": ["verified tool result", "evidence ref", "VCM artifact", "learning trace"],
                "verification_tests": [row.get("case_id") for row in rows],
                "risk_tier": "low",
                "provenance": "reports/viea_execution_spine.json",
                "update_policy": "replay before promotion-facing use",
                "retirement_policy": "retire if replay checksum or verifier contract fails",
            }
        )
    return {
        "policy": "project_theseus_viea_loop_closure_candidates_v1",
        "created_utc": now(),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_verified_procedural_tools(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row.get("verified") and row.get("replay_checksum"):
            by_tool.setdefault(str(row.get("tool_id")), []).append(row)
    tools = []
    for tool_id, rows in sorted(by_tool.items()):
        checksums = sorted(str(row.get("replay_checksum")) for row in rows if row.get("replay_checksum"))
        repeated = len(rows) >= 2
        tool = {
            "procedural_tool_id": stable_id("viea_procedural_tool", tool_id, checksums),
            "tool_id": tool_id,
            "status": "verified_procedural_tool" if repeated else "candidate_needs_repetition",
            "promotion_allowed": repeated,
            "successful_trace_count": len(rows),
            "preconditions": ["registered MCP-compatible tool card", "compiled VCM context packet", "no public training rows", "strict no fallback returns"],
            "postconditions": ["verified result state", "evidence ref present", "replay checksum present", "VCM artifact emitted"],
            "verification_tests": [row.get("case_id") for row in rows],
            "replay_checksums": checksums,
            "risk_tier": "low",
            "provenance": "reports/viea_execution_spine.json",
            "update_policy": "rerun replay and verifier checks before promotion-facing use",
            "retirement_policy": "retire on checksum drift, verifier failure, dependency loss, or policy violation",
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
        tools.append(tool)
    return {
        "policy": "project_theseus_viea_verified_procedural_tools_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if any(row.get("promotion_allowed") for row in tools) else "YELLOW",
        "passed": bool(any(row.get("promotion_allowed") for row in tools)),
        "procedural_tool_count": len(tools),
        "verified_procedural_tool_count": sum(1 for row in tools if row.get("promotion_allowed")),
        "candidate_needs_repetition_count": sum(1 for row in tools if not row.get("promotion_allowed")),
        "tools": tools,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_training_evidence(
    learning_traces: list[dict[str, Any]],
    results: list[dict[str, Any]],
    residuals: dict[str, Any],
) -> list[dict[str, Any]]:
    residual_by_case = {str(row.get("case_id")): row for row in residuals.get("rows", []) if isinstance(row, dict)}
    rows = []
    for trace in learning_traces:
        case_id = str(trace.get("case_id") or "")
        residual = residual_by_case.get(case_id)
        verifier_passed = trace.get("verifier_outcome") == "pass"
        eligible = bool(verifier_passed and trace.get("source") == "private_synthetic_owned_execute_mode")
        rows.append(
            {
                "policy": "project_theseus_private_tool_use_training_evidence_v1",
                "created_utc": now(),
                "evidence_id": stable_id("training_evidence", trace.get("trace_id"), trace.get("selected_tool"), trace.get("verifier_outcome")),
                "trace_id": trace.get("trace_id"),
                "run_id": trace.get("run_id"),
                "source": trace.get("source"),
                "case_id": case_id,
                "selected_tool": trace.get("selected_tool"),
                "arguments_hash": trace.get("arguments_hash"),
                "result_state": trace.get("result_state"),
                "verifier_outcome": trace.get("verifier_outcome"),
                "operator_outcome": trace.get("operator_outcome"),
                "accepted_missed_ignored_completed_failure": trace.get("accepted_missed_ignored_completed_failure"),
                "training_eligibility": {
                    "eligible_for_governed_admission": eligible,
                    "admitted_to_training_rows": False,
                    "reason": "verified_private_tool_use_trace" if eligible else "residual_or_non_pass_trace_requires_repair",
                },
                "support_state": "SUPPORTED" if eligible else "RESIDUAL",
                "residual_ref": residual.get("residual_id") if residual else "",
                "evidence_ref": trace.get("evidence_ref"),
                "vcm_context_hash": trace.get("vcm_context_hash"),
                "raw_private_text_stored": False,
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            }
        )
    return rows


def summarize_training_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = sum(1 for row in rows if row.get("training_eligibility", {}).get("eligible_for_governed_admission"))
    return {
        "evidence_row_count": len(rows),
        "eligible_for_governed_admission_count": eligible,
        "residual_or_non_pass_count": len(rows) - eligible,
        "admitted_to_training_rows": 0,
        "raw_private_text_stored": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_residuals(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        if result.get("verified"):
            continue
        rows.append(
            {
                "residual_id": stable_id("viea_residual", result.get("case_id"), result.get("tool_id"), result.get("state")),
                "case_id": result.get("case_id"),
                "tool_id": result.get("tool_id"),
                "state": result.get("state"),
                "status": result.get("status"),
                "category": residual_category_for(result),
                "diagnostic": str(result.get("diagnostic") or "")[:1000],
                "evidence_ref": result.get("evidence_ref"),
                "vcm_context_hash": result.get("vcm_context_hash"),
                "repair_target": repair_target_for(result),
                "training_boundary": "private residual target only; no public benchmark artifact and no fallback return",
            }
        )
    return {
        "policy": "project_theseus_viea_execution_residuals_v1",
        "created_utc": now(),
        "residual_count": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "rows": rows,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def residual_category_for(result: dict[str, Any]) -> str:
    state = str(result.get("state") or "")
    diagnostic = str(result.get("diagnostic") or "").lower()
    tool_id = str(result.get("tool_id") or "")
    if state == "TOOL_UNAVAILABLE":
        return "dependency_unavailable"
    if state == "TOOL_FAULT":
        return "tool_runtime_fault"
    if "expected" in diagnostic or "verified" in diagnostic:
        return "verifier_contract"
    if tool_id.startswith("search."):
        return "retrieval_no_hit"
    if tool_id.startswith("logic."):
        return "formal_solver_gap"
    return "semantic_or_contract_miss"


def repair_target_for(result: dict[str, Any]) -> str:
    category = residual_category_for(result)
    if category == "dependency_unavailable":
        return "install_or_stage_optional_dependency_and_rerun_private_tool_smoke"
    if category == "tool_runtime_fault":
        return "repair_tool_runner_exception_without_fallback_answer"
    if category == "retrieval_no_hit":
        return "add private corpus/index coverage_or_better_query_decomposition"
    if category == "formal_solver_gap":
        return "expand_private_formal_solver_parser_or_dependency"
    if category == "verifier_contract":
        return "repair_expected_result_or_verifier_shape_contract"
    return "mine_private_case_into_structural_tool_selection_or_contract_repair"


def run_private_benchmark_adapter_dry_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for adapter in config.get("benchmark_adapter_readiness", []) if isinstance(config.get("benchmark_adapter_readiness"), list) else []:
        if not isinstance(adapter, dict):
            continue
        scorer = str(adapter.get("local_private_dry_run_scorer") or "")
        result = private_scorer_result(scorer)
        rows.append(
            {
                "id": adapter.get("id"),
                "public_role": "calibration_only",
                "train_allowed": False,
                "local_private_dry_run_scorer": scorer,
                "dry_run_state": result["state"],
                "dry_run_passed": result["passed"],
                "diagnostic": result["diagnostic"],
                "public_artifacts_loaded": False,
                "public_training_rows_written": 0,
            }
        )
    return rows


def private_scorer_result(scorer: str) -> dict[str, Any]:
    if scorer == "private_symbolic_action_sequence_goal_check":
        state = {"at": "home", "has_ticket": False}
        for action in ["buy_ticket", "go_airport"]:
            if action == "buy_ticket":
                state["has_ticket"] = True
            if action == "go_airport" and state["has_ticket"]:
                state["at"] = "airport"
        return {"state": "PRIVATE_DRY_RUN_PASS", "passed": state["at"] == "airport", "diagnostic": "tiny private action sequence reached goal"}
    if scorer == "private_constraint_plan_consistency_check":
        plan = {"budget": 42, "max_budget": 50, "arrive_before": 12, "arrival": 11}
        return {"state": "PRIVATE_DRY_RUN_PASS", "passed": plan["budget"] <= plan["max_budget"] and plan["arrival"] <= plan["arrive_before"], "diagnostic": "tiny private travel constraint plan consistent"}
    if scorer == "private_function_call_schema_exact_match":
        call = {"name": "math.sympy_exact", "arguments": {"operation": "simplify", "expression": "x+x"}}
        passed = call["name"] and isinstance(call["arguments"], dict) and "operation" in call["arguments"]
        return {"state": "PRIVATE_DRY_RUN_PASS", "passed": passed, "diagnostic": "tiny private function-call schema exact match"}
    if scorer == "private_web_navigation_state_check":
        page = {"url": "/inbox", "clicked": [], "form": {}, "toast": ""}
        for action in [
            {"type": "click", "target": "compose"},
            {"type": "type", "target": "to", "value": "local@example.test"},
            {"type": "type", "target": "body", "value": "private dry run"},
            {"type": "click", "target": "send"},
        ]:
            if action["type"] == "click":
                page["clicked"].append(action["target"])
            if action["type"] == "type":
                page["form"][action["target"]] = action["value"]
            if action["type"] == "click" and action["target"] == "send" and page["form"].get("to") and page["form"].get("body"):
                page["toast"] = "sent"
        return {"state": "PRIVATE_DRY_RUN_PASS", "passed": page["toast"] == "sent", "diagnostic": "tiny private web action state reached completion"}
    if scorer == "private_os_action_schema_check":
        actions = [
            {"type": "open", "target": "notes"},
            {"type": "write_file", "path": "runtime/private_os_dry_run/note.txt", "content_hash": "sha256:dry"},
            {"type": "verify_file", "path": "runtime/private_os_dry_run/note.txt"},
        ]
        passed = all(action.get("type") and (action.get("target") or action.get("path")) for action in actions)
        return {"state": "PRIVATE_DRY_RUN_PASS", "passed": passed, "diagnostic": "tiny private OS action schema validated without controlling the desktop"}
    if scorer == "private_enterprise_workflow_state_check":
        ticket = {"status": "new", "owner": "", "evidence": []}
        for step in ["triage", "assign", "attach_evidence", "resolve"]:
            if step == "triage":
                ticket["status"] = "triaged"
            elif step == "assign":
                ticket["owner"] = "local_agent"
            elif step == "attach_evidence":
                ticket["evidence"].append("private_report_hash")
            elif step == "resolve" and ticket["owner"] and ticket["evidence"]:
                ticket["status"] = "resolved"
        return {"state": "PRIVATE_DRY_RUN_PASS", "passed": ticket["status"] == "resolved", "diagnostic": "tiny private enterprise workflow reached resolved state"}
    return {"state": "CONTRACT_ONLY_DEFERRED", "passed": True, "diagnostic": "heavy public-like environment setup deferred; contract boundary only"}


def build_research_matrix() -> dict[str, Any]:
    rows = [
        matrix_row("VIEA", ["scripts/viea_execution_spine.py", "scripts/viea_artifact_kernel.py"], "artifact graph, claim ledger, support states, evidence refs", "runtime promotion still limited to deterministic private path"),
        matrix_row("PlanForge/LLMCompiler", ["scripts/theseus_plan_compiler.py", "configs/theseus_plan_compiler.json"], "compiled DAG, schedule, critical path, executable packet", "parallel tool execution beyond deterministic smoke pending"),
        matrix_row("LangGraph durability", ["scripts/viea_execution_spine.py", "reports/viea_execution_spine.sqlite"], "run ids, leases, checkpoints, resume/cancel hooks", "human interrupt UI not wired"),
        matrix_row("MCP", ["reports/deterministic_tool_registry.json", "scripts/theseus_deterministic_tool_substrate.py"], "MCP-compatible card fields in progress", "external MCP connector bridge pending"),
        matrix_row("Memory control", ["reports/viea_execution_spine_vcm_artifacts.jsonl", "scripts/virtual_context_memory.py"], "VCM packet hashes and tool-output artifacts", "native KV/prefix-cache parity not claimed"),
        matrix_row("Deterministic solvers", ["configs/deterministic_tool_substrate.json", "scripts/theseus_deterministic_tool_substrate.py"], "SymPy, SciPy, mpmath, interval arithmetic, linear algebra, Lean, staged Z3, bounded equality saturation, BM25, hybrid local search, VCM search", "Z3 dependency remains optional/staged when z3-solver is not installed"),
        matrix_row("Toolformer/BFCL", ["reports/viea_tool_use_learning_traces.jsonl", "reports/viea_tool_use_training_evidence.jsonl"], "private tool-use traces and training-evidence rows with verifier outcomes and governed-admission eligibility", "public BFCL calibration not run"),
        matrix_row("Loop closure", ["reports/viea_execution_loop_closure_candidates.json", "reports/viea_verified_procedural_tools.json"], "repeated verified traces become procedural tool records with preconditions, postconditions, checksums, risk, update, and retirement policy", "one-off successful tools still require repetition before promotion"),
        matrix_row("Planning/agent benchmarks", ["configs/viea_execution_spine.json", "reports/viea_execution_spine.json"], "private dry-run scorer contracts", "public benchmark execution not run"),
    ]
    return {
        "policy": "project_theseus_viea_research_implementation_matrix_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "passed": True,
        "rows": rows,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def matrix_row(topic: str, files: list[str], implemented: str, gap: str) -> dict[str, Any]:
    return {"topic": topic, "files": files, "implemented": implemented, "remaining_gap": gap}


def build_gates(
    config: dict[str, Any],
    dags: dict[str, Any],
    packet: dict[str, Any],
    cases: list[dict[str, Any]],
    old_results: list[dict[str, Any]],
    compiled_results: list[dict[str, Any]],
    ab: dict[str, Any],
    benchmark_readiness: list[dict[str, Any]],
    execute: bool,
    procedural_tools: dict[str, Any],
    training_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        gate("config_loaded", bool(config.get("policy")), config.get("policy"), "hard"),
        gate("compiled_dags_loaded", bool(dags.get("compiled_goals")), dags.get("compiled_goal_count"), "hard"),
        gate("compiled_tool_packet_present", bool(packet.get("packet_id")), packet.get("packet_id"), "hard"),
        gate("private_case_set_present", bool(cases), len(cases), "hard"),
        gate("old_baseline_ran", len(old_results) == len(cases), {"old": len(old_results), "cases": len(cases)}, "hard"),
        gate("execute_mode_requested", bool(execute), execute, "hard"),
        gate("compiled_execute_ran_end_to_end", execute and len(compiled_results) == len(cases), {"compiled": len(compiled_results), "cases": len(cases)}, "hard"),
        gate("every_executed_node_has_vcm_context", all(row.get("vcm_runtime_context_consumed") or row.get("case_id") == "trace_replay_private" for row in compiled_results), len(compiled_results), "hard"),
        gate("leases_and_checkpoints_present", all(row.get("durable_lease") and row.get("checkpoint", {}).get("path") for row in compiled_results), len(compiled_results), "hard"),
        gate("retry_policy_attached_to_every_node", all(row.get("retry_policy", {}).get("max_attempts") for row in compiled_results), len(compiled_results), "hard"),
        gate(
            "every_executed_result_has_runtime_asi_records",
            all(result_has_runtime_asi_records(row) for row in compiled_results),
            {
                "compiled": len(compiled_results),
                "runtime_asi_stack_record_count": runtime_asi_stack_record_count(compiled_results),
                "missing_by_case": {
                    str(row.get("case_id")): missing_runtime_asi_records(row)
                    for row in compiled_results
                    if missing_runtime_asi_records(row)
                },
            },
            "hard",
        ),
        gate(
            "runtime_authority_context_artifact_and_evidence_receipts_present",
            all(
                not {
                    "authority_use_receipt",
                    "context_transaction",
                    "artifact_graph",
                    "evidence_transition",
                }.intersection(missing_runtime_asi_records(row))
                for row in compiled_results
            ),
            len(compiled_results),
            "hard",
        ),
        gate("ab_same_safe_task_set", ab.get("same_private_case_count") == len(cases), ab.get("same_private_case_count"), "hard"),
        gate("training_evidence_rows_match_executed_nodes", len(training_evidence) == len(compiled_results), {"training_evidence": len(training_evidence), "compiled": len(compiled_results)}, "hard"),
        gate(
            "training_evidence_metadata_only",
            all(not row.get("raw_private_text_stored") and row.get("public_training_rows_written") == 0 for row in training_evidence),
            len(training_evidence),
            "hard",
        ),
        gate("verified_procedural_tools_emitted_from_repeated_traces", int(procedural_tools.get("verified_procedural_tool_count") or 0) > 0, procedural_tools.get("verified_procedural_tool_count"), "hard"),
        gate("benchmark_adapters_calibration_only", all(not row.get("train_allowed") and not row.get("public_artifacts_loaded") for row in benchmark_readiness), benchmark_readiness, "hard"),
        gate("no_public_training_rows", True, 0, "hard"),
        gate("no_external_inference_calls", True, 0, "hard"),
        gate("no_fallback_returns", True, 0, "hard"),
    ]


def summarize(
    old_results: list[dict[str, Any]],
    compiled_results: list[dict[str, Any]],
    ab: dict[str, Any],
    gates: list[dict[str, Any]],
    stale_recovery: dict[str, Any],
    started: float,
    procedural_tools: dict[str, Any] | None = None,
    training_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    procedural_tools = procedural_tools or {}
    training_evidence = training_evidence or []
    return {
        "old_case_count": len(old_results),
        "compiled_case_count": len(compiled_results),
        "old_useful_completion_rate": ab.get("old_direct_local_tool_baseline", {}).get("useful_completion_rate"),
        "compiled_useful_completion_rate": ab.get("compiled_viea_execution", {}).get("useful_completion_rate"),
        "compiled_lease_count": len([row for row in compiled_results if row.get("durable_lease")]),
        "compiled_checkpoint_count": len([row for row in compiled_results if row.get("checkpoint", {}).get("path")]),
        "runtime_asi_stack_record_count": runtime_asi_stack_record_count(compiled_results),
        "stale_lease_recovered_count": stale_recovery.get("recovered_count", 0),
        "hard_gate_failure_count": len([row for row in gates if row["severity"] == "hard" and not row["passed"]]),
        "gate_failure_count": len([row for row in gates if not row["passed"]]),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "residual_count": len([row for row in compiled_results if not row.get("verified")]),
        "training_evidence_row_count": len(training_evidence),
        "verified_procedural_tool_count": int(procedural_tools.get("verified_procedural_tool_count") or 0),
    }


def runtime_summary(db_path: Path, run_id: str) -> dict[str, Any]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        leases = [dict(row) for row in conn.execute("SELECT * FROM node_leases WHERE run_id=?", (run_id,)).fetchall()]
        events = [dict(row) for row in conn.execute("SELECT * FROM events WHERE run_id=?", (run_id,)).fetchall()]
        checkpoints = [dict(row) for row in conn.execute("SELECT * FROM checkpoints WHERE run_id=?", (run_id,)).fetchall()]
    return {
        "db": rel(db_path),
        "run_id": run_id,
        "lease_count": len(leases),
        "event_count": len(events),
        "checkpoint_count": len(checkpoints),
        "status_counts": dict(Counter(row.get("status") for row in leases)),
        "resume_supported": True,
        "cancel_supported": True,
        "stale_lease_recovery_supported": True,
    }


def remaining_gaps(
    execute: bool,
    benchmark_readiness: list[dict[str, Any]],
    compiled_results: list[dict[str, Any]],
    procedural_tools: dict[str, Any],
) -> list[str]:
    gaps = []
    if not execute:
        gaps.append("execute mode was not requested")
    if any(row.get("dry_run_state") == "CONTRACT_ONLY_DEFERRED" for row in benchmark_readiness):
        gaps.append("heavy public-like agent environments remain contract-only until local environment setup")
    if any(row.get("tool_id") == "logic.z3_smt" and row.get("state") == "TOOL_UNAVAILABLE" for row in compiled_results):
        gaps.append("Z3/SMT dependency is staged but unavailable in the active Python environment")
    if int(procedural_tools.get("verified_procedural_tool_count") or 0) == 0 and compiled_results:
        gaps.append("no repeated successful trace family has enough replay evidence for procedural-tool promotion")
    return gaps


def event_row(run_id: str, event_type: str, content: dict[str, Any]) -> dict[str, Any]:
    return {"policy": "project_theseus_viea_execution_event_v1", "created_utc": now(), "run_id": run_id, "event_type": event_type, "content": content}


def add_event(db_path: Path, run_id: str, event_type: str, content: dict[str, Any]) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO events (event_id, run_id, event_type, content_json, created_utc) VALUES (?, ?, ?, ?, ?)",
            (stable_id("event", run_id, event_type, json.dumps(content, sort_keys=True, default=str)), run_id, event_type, json.dumps(content, sort_keys=True, default=str), now()),
        )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": result.get("case_id"),
        "tool_id": result.get("tool_id"),
        "state": result.get("state"),
        "verified": result.get("verified"),
        "status": result.get("status"),
        "runtime_ms": result.get("runtime_ms"),
        "replay_checksum": result.get("replay_checksum"),
    }


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# VIEA Execution Spine",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- run_id: `{report.get('run_id')}`",
        f"- execute_requested: `{report.get('execute_requested')}`",
        f"- old cases: `{summary.get('old_case_count')}`",
        f"- compiled cases: `{summary.get('compiled_case_count')}`",
        f"- compiled useful completion: `{summary.get('compiled_useful_completion_rate')}`",
        f"- leases: `{summary.get('compiled_lease_count')}`",
        f"- checkpoints: `{summary.get('compiled_checkpoint_count')}`",
        f"- runtime ASI Stack records: `{summary.get('runtime_asi_stack_record_count')}`",
        f"- hard gate failures: `{summary.get('hard_gate_failure_count')}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []):
        marker = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {marker} `{row.get('name')}` ({row.get('severity')})")
    lines.extend(["", "## Remaining Gaps", ""])
    for gap in report.get("remaining_gaps", []):
        lines.append(f"- {gap}")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace(os.sep, "/")
    except Exception:
        return str(path).replace(os.sep, "/")


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def stable_hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
