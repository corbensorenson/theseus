"""Hive Work Board Executor V1.

The Hive work board is the durable source of truth for unattended work. This
executor refreshes the board, assigns ready work to the current/best node,
executes one bounded task at a time, writes a replay ledger, routes evidence,
and leaves failed work retryable once before blocking it.

It is intentionally conservative: no shell interpretation, no public benchmark
training, no remote desktop side effects, and no arbitrary command execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hive_node_registry  # noqa: E402
import report_evidence_store  # noqa: E402
from hive_work_board_command_guards import (
    code_contract_preflight_command,
    code_contract_preflight_guard,
    code_lm_training_command_gate,
    decoder_relevant_source_mtime,
    execution_shape_no_template_smoke_guard,
    private_pressure_private_closure_needed,
    private_public_calibration_guard,
    private_type_shape_receiver_ablation_needed,
    post_edge_contract_v2_public_calibration_limit_reached,
)
from hive_work_board_commands import command_for_task
from progress_integrity_policy import (
    is_non_promotable_diagnostic_concept,
    non_promotable_diagnostic_reason,
    promotion_safe_replacement_concept,
)
from hive_work_board_executor_runtime import (
    task_rotation_lane,
    recent_rotation_lane_counts,
    rotation_lane_for_concept,
    lane_rotation_penalty,
    lane_rotation_rank,
    is_conversation_task,
    is_generalist_high_transfer_task,
    is_code_transfer_task,
    run_postprocess_commands,
    useful_frontier_evidence_returned,
    normalize_frontier_evidence_improvement,
    ingest_command_evidence,
    build_report,
    snapshot_improvement_metrics,
    classify_improvement,
    no_progress_demote_or_block,
    report_signals,
    signal,
    residual_cluster_for,
    maybe_enqueue_teacher_escalation,
    improvement_ledger_row,
    load_node_context,
    update_task,
    upsert_task,
    add_event,
    add_evidence,
    unblock_resolved_guard_blocks,
    run_subprocess,
    record_hook,
    hook_guards,
    feedback_row,
    ledger_row,
    compact_result,
    render_markdown,
    stable_id,
    get_path,
    parse_json,
    read_json,
    read_jsonl,
    write_json,
    write_text,
    append_jsonl,
    first_number,
    safe_float,
    improved,
    lower,
    resolve,
    rel,
    now,
)

CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
DEFAULT_DB = REPORTS / "hive_work_board.sqlite"
DEFAULT_OUT = REPORTS / "hive_work_board_executor.json"
DEFAULT_MARKDOWN = REPORTS / "hive_work_board_executor.md"
LEDGER = REPORTS / "hive_work_board_execution_ledger.jsonl"
COMMAND_LEDGER = REPORTS / "hive_live_command_ledger.jsonl"
HOOK_LEDGER = REPORTS / "hive_tool_hook_ledger.jsonl"
FEEDBACK_LEDGER = REPORTS / "hive_work_board_feedback.jsonl"
IMPROVEMENT_LEDGER = REPORTS / "hive_unattended_improvement_ledger.jsonl"
TEACHER_ESCALATION_LEDGER = REPORTS / "hive_teacher_auto_escalation_ledger.jsonl"
NO_PROGRESS_DEMOTION_LEDGER = REPORTS / "hive_no_progress_demotion_ledger.jsonl"
PAUSE_FLAG = REPORTS / "hive_work_board_pause.flag"
STOP_FLAGS = [REPORTS / "sparkstream_stop.flag", REPORTS / "unattended_autonomy_stop.flag", REPORTS / "hive_work_board_stop.flag"]

READY_STATUSES = {"ready", "queued", "failed"}
PRIORITY_SCORE = {"critical": 0, "high": 10, "medium": 20, "low": 30}
IMPROVEMENT_SIGNAL_KINDS = {
    "private_residual_shrank",
    "public_transfer_improved",
    "new_clean_evidence_produced",
    "adapter_repaired",
    "teacher_experiment_spec_produced",
    "stale_tool_retired",
    "useful_failure_residual_captured",
}
CODE_HIGH_TRANSFER_CONCEPTS = {
    "type_contract_diagnostic",
    "type_contract_four_card_calibration",
    "edge_exec_repair_four_card_calibration",
    "execution_shaped_four_card_calibration",
    "execution_shape_private_ablation",
    "type_and_return_shape",
    "typed_interface_skeleton",
    "typed_interface_private_closure",
    "edge_contract_4card",
    "edge_contract_private_closure",
    "edge_contract_balanced_4card_private_curriculum_v2",
    "edge_contract_balanced_private_closure_v2",
    "edge_case_full_body_private_curriculum_v1",
    "edge_case_full_body_private_closure_v1",
    "edge_contract_v2_private_residual_curriculum",
    "edge_contract_v2_private_closure",
    "candidate_floor_v2_private_residual_curriculum",
    "candidate_floor_v2_private_closure",
    "residual_targeted_private_edge_case_contract_v1",
    "decoder_v2_private_ablation_gate",
    "private_type_shape_receiver_veto_ablation",
    "private_pressure_private_closure",
    "admissibility_and_interface",
    "edge_conditions",
    "algorithmic_planning",
    "execution_shaped_programs",
    "private_pressure_four_card_recalibration",
}
GENERALIST_HIGH_TRANSFER_CONCEPTS = {
    "multi_turn_conversation_hard",
    "multi_turn_conversation_hard_v2",
    "multi_turn_conversation_hard_v3",
    "multi_turn_conversation_hard_v4",
    "board_game_rl",
    "pufferlib4_rl",
    "long_horizon_tool_use",
    "repo_repair",
    "cross_domain_sts_capsules",
}
PRIVATE_SOURCE_AGNOSTIC_CONCEPTS = {
    "type_and_return_shape",
    "typed_interface_skeleton",
    "edge_contract_4card",
    "edge_contract_private_closure",
    "edge_contract_balanced_4card_private_curriculum_v2",
    "edge_contract_balanced_private_closure_v2",
    "edge_case_full_body_private_curriculum_v1",
    "edge_case_full_body_private_closure_v1",
    "edge_contract_v2_private_residual_curriculum",
    "edge_contract_v2_private_closure",
    "candidate_floor_v2_private_residual_curriculum",
    "candidate_floor_v2_private_closure",
    "residual_targeted_private_edge_case_contract_v1",
    "decoder_v2_private_ablation_gate",
    "private_type_shape_receiver_veto_ablation",
    "private_pressure_private_closure",
    "admissibility_and_interface",
    "edge_conditions",
    "algorithmic_planning",
    "execution_shaped_programs",
    "repo_repair",
    "multi_turn_conversation_hard_v2",
    "multi_turn_conversation_hard_v3",
    "long_horizon_tool_use",
    "board_game_rl",
    "pufferlib4_rl",
    "cross_domain_sts_capsules",
}
LANE_ROTATION_ORDER = {
    "private_closure_gate": 0,
    "harder_conversation_v2": 1,
    "board_game_self_play_rl": 2,
    "pufferlib4_rl": 3,
    "long_horizon_tool_use": 4,
    "repo_repair": 5,
    "cross_domain_sts_capsules": 6,
    "typed_interface_skeleton": 7,
    "code_transfer": 8,
    "other": 20,
}
FRESH_TARGET_MAX_AGE_SECONDS = 24 * 3600
STALE_HIVE_QUEUE_SECONDS = 6 * 3600
SATISFIABLE_HIGH_TRANSFER_TARGETS = {
    "multi_turn_conversation_hard",
    "multi_turn_conversation_hard_v2",
    "multi_turn_conversation_hard_v3",
    "board_game_rl",
    "pufferlib4_rl",
    "long_horizon_tool_use",
    "repo_repair",
    "cross_domain_sts_capsules",
    "execution_shape_private_ablation",
    "private_type_shape_receiver_veto_ablation",
    "decoder_v2_private_ablation_gate",
}
FOUR_CARD_RECEIVER_SLUG = "source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32"
CODE_CONTRACT_PREFLIGHT_REPORT = REPORTS / "code_lm_closure_public_contract_preflight_seed23_32.json"
EXECUTION_SHAPE_NO_TEMPLATE_SMOKE_REPORT = REPORTS / "execution_shape_private_ablation_smoke.json"
DECODER_RELEVANT_SOURCES = (
    ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure.rs",
    ROOT / "scripts" / "code_lm_closure.py",
    ROOT / "scripts" / "code_residual_curriculum.py",
    ROOT / "scripts" / "type_contract_diagnostic.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--only-task-id", default="")
    parser.add_argument("--force-local-remote-assignment", action="store_true", help="Debug override: run a remotely assigned task on this local process.")
    parser.add_argument("--command-text", default="", help="Live command channel text, e.g. /background run broad transfer status.")
    parser.add_argument("--source-channel", default="dashboard")
    parser.add_argument("--enqueue-only", action="store_true")
    parser.add_argument("--no-refresh-board", action="store_true")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--resume-board", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.pause:
        PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
        PAUSE_FLAG.write_text("paused\n", encoding="utf-8")
    if args.resume_board:
        PAUSE_FLAG.unlink(missing_ok=True)

    db_path = resolve(args.db)
    ensure_schema(db_path)
    command_result: dict[str, Any] = {}
    if args.command_text:
        command_result = enqueue_live_command(db_path, args.command_text, source_channel=args.source_channel)

    refresh_result: dict[str, Any] = {}
    if not args.no_refresh_board:
        refresh_result = refresh_operator_board(db_path)
    high_transfer_sync_result = sync_high_transfer_scheduler_tasks(db_path)
    retirement_result = retire_regression_only_high_transfer_tasks(db_path)
    high_transfer_supersede_result = supersede_stale_high_transfer_tasks(db_path)
    feedback_guard_result = block_stale_feedback_actions(db_path)
    teacher_retirement_result = retire_reclassified_teacher_escalations(db_path)
    satisfied_target_result = satisfy_fresh_high_transfer_targets(db_path)
    stale_hive_queue_result = retire_stale_hive_task_queue_chunks(db_path)
    resolved_guard_block_result = unblock_resolved_guard_blocks(db_path)

    node_context = load_node_context()
    tasks = load_tasks(db_path)
    selected_task_id = args.only_task_id or str(command_result.get("task_id") or "")
    selected = select_tasks(tasks, node_context, only_task_id=selected_task_id, limit=max(1, int(args.max_tasks)))
    paused = PAUSE_FLAG.exists()
    stop = any(path.exists() for path in STOP_FLAGS)
    results: list[dict[str, Any]] = []
    should_execute = bool(args.execute and not args.status and not args.enqueue_only and not paused and not stop)
    if should_execute:
        for task in selected:
            results.append(execute_task(db_path, task, node_context, args=args))
    report = build_report(
        db_path,
        tasks=load_tasks(db_path),
        selected=selected,
        results=results,
        command_result=command_result,
        refresh_result=refresh_result,
        high_transfer_sync_result=high_transfer_sync_result,
        retirement_result=retirement_result,
        high_transfer_supersede_result=high_transfer_supersede_result,
        feedback_guard_result=feedback_guard_result,
        teacher_retirement_result=teacher_retirement_result,
        satisfied_target_result=satisfied_target_result,
        stale_hive_queue_result=stale_hive_queue_result,
        resolved_guard_block_result=resolved_guard_block_result,
        node_context=node_context,
        started=started,
        args=args,
        paused=paused,
        stop=stop,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, resolve(args.out), payload=report)
    print(json.dumps(report, indent=2))
    return 2 if report["trigger_state"] == "RED" else 0


def ensure_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                priority TEXT NOT NULL,
                assignee TEXT NOT NULL,
                node_id TEXT NOT NULL,
                command TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                blocked_reason TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                label TEXT NOT NULL,
                path TEXT NOT NULL,
                claim_role TEXT NOT NULL,
                created_utc TEXT NOT NULL
            );
            """
        )


def enqueue_live_command(db_path: Path, command_text: str, *, source_channel: str) -> dict[str, Any]:
    parsed = parse_live_command(command_text)
    task_id = stable_id("live_command", source_channel, command_text, now())
    status = "ready" if parsed["allowed"] else "blocked"
    blocked_reason = "" if parsed["allowed"] else parsed["reason"]
    task = {
        "task_id": task_id,
        "title": parsed["title"],
        "source": "live_command_channel",
        "kind": parsed["kind"],
        "status": status,
        "priority": parsed["priority"],
        "assignee": "hive_work_board_executor",
        "node_id": "local",
        "command": command_text,
        "evidence_json": json.dumps(
            {
                "source_channel": source_channel,
                "parsed_intent": parsed,
                "command_contract": {
                    "role": "Hive operator",
                    "objective": parsed["title"],
                    "context": "Live dashboard/tray/mobile command channel",
                    "constraints": "bounded local execution, no public training leakage, durable board ledger",
                    "verification": "task ledger, hook ledger, and output report",
                    "failure_behavior": "mark blocked or failed with residual reason",
                },
            },
            sort_keys=True,
        ),
        "retry_count": 0,
        "blocked_reason": blocked_reason,
    }
    upsert_task(db_path, task)
    add_event(db_path, task_id, "live_command_enqueued", {"command_text": command_text, "parsed": parsed})
    append_jsonl(COMMAND_LEDGER, {"created_utc": now(), "policy": "project_theseus_hive_live_command_v1", "task_id": task_id, "command_text": command_text, "parsed": parsed, "source_channel": source_channel})
    return {"ok": parsed["allowed"], "task_id": task_id, "parsed": parsed}


def parse_live_command(command_text: str) -> dict[str, Any]:
    text = " ".join(command_text.strip().split())
    lower = text.lower()
    if not lower.startswith("/background"):
        return {"allowed": False, "reason": "only_background_commands_enabled_v1", "kind": "unsupported_live_command", "priority": "medium", "title": f"Unsupported command: {text[:80]}"}
    if "broad" in lower and "transfer" in lower and "status" in lower:
        return {"allowed": True, "reason": "background_broad_transfer_status", "kind": "background_broad_transfer_status", "priority": "high", "title": "Background: broad transfer status"}
    if "operator" in lower and "status" in lower:
        return {"allowed": True, "reason": "background_operator_status", "kind": "background_operator_status", "priority": "medium", "title": "Background: Hive Operator OS status"}
    if "vacation" in lower and ("status" in lower or "check" in lower):
        return {"allowed": True, "reason": "background_vacation_status", "kind": "background_vacation_status", "priority": "medium", "title": "Background: vacation supervisor status"}
    return {"allowed": False, "reason": "unknown_background_command", "kind": "unsupported_live_command", "priority": "medium", "title": f"Unsupported background command: {text[:80]}"}


def refresh_operator_board(db_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/hive_operator_os.py",
        "--config",
        "configs/hive_operator_os.json",
        "--db",
        rel(db_path),
        "--out",
        "reports/hive_operator_os.json",
        "--markdown-out",
        "reports/hive_operator_os.md",
    ]
    return run_subprocess(command, timeout=120)


def load_tasks(db_path: Path, limit: int = 1000) -> list[dict[str, Any]]:
    ensure_schema(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM tasks
            ORDER BY
              CASE status
                WHEN 'ready' THEN 0
                WHEN 'queued' THEN 1
                WHEN 'failed' THEN 2
                WHEN 'retry_queued' THEN 3
                WHEN 'active' THEN 4
                WHEN 'blocked' THEN 5
                WHEN 'done' THEN 6
                ELSE 7
              END,
              CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
              updated_utc DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["evidence"] = parse_json(item.pop("evidence_json"), {})
        out.append(item)
    return out


def task_by_id(db_path: Path, task_id: str) -> dict[str, Any] | None:
    ensure_schema(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["evidence"] = parse_json(item.pop("evidence_json"), {})
    return item


def select_tasks(tasks: list[dict[str, Any]], node_context: dict[str, Any], *, only_task_id: str, limit: int) -> list[dict[str, Any]]:
    ready = []
    high_transfer_lifecycle = high_transfer_concept_lifecycle()
    ready_private_concepts = {
        concept
        for concept, row in high_transfer_lifecycle.items()
        if concept in PRIVATE_SOURCE_AGNOSTIC_CONCEPTS
        and not is_non_promotable_diagnostic_concept(concept)
        and str(row.get("status") or "") == "ready"
    }
    for task in tasks:
        if only_task_id and task.get("task_id") != only_task_id:
            continue
        if not only_task_id and is_non_promotable_diagnostic_concept(get_path(task, ["evidence", "concept"], "")):
            continue
        if not only_task_id and high_transfer_task_is_regression_only(task, high_transfer_lifecycle):
            continue
        if not only_task_id and ready_private_concepts and str(task.get("source") or "") == "broad_transfer_closure":
            continue
        if not only_task_id and str(task.get("source") or "") == "hive_task_queue":
            continue
        if not only_task_id and post_edge_contract_v2_public_calibration_limit_reached(task):
            continue
        if not only_task_id and task_assigned_to_other_node(task, node_context):
            continue
        if str(task.get("status") or "") not in READY_STATUSES:
            continue
        if str(task.get("status") or "") == "failed" and int(task.get("retry_count") or 0) > 0:
            continue
        assignment = assign_task(task, node_context)
        task["assignment"] = assignment
        ready.append(task)
    conversation_first = conversation_focus_enabled()
    recent_lanes = recent_rotation_lane_counts()
    return sorted(
        ready,
        key=lambda row: task_sort_key(row, conversation_first=conversation_first, recent_lanes=recent_lanes),
        reverse=False,
    )[:limit]


def task_sort_key(
    task: dict[str, Any],
    *,
    conversation_first: bool,
    recent_lanes: dict[str, int] | None = None,
) -> tuple[int, int, int, int, str]:
    score = PRIORITY_SCORE.get(str(task.get("priority") or ""), 99)
    concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
    if is_generalist_high_transfer_task(task):
        score -= 35
    elif is_code_transfer_task(task):
        score += 5
    if conversation_first:
        if is_conversation_task(task):
            score -= 50
        elif is_code_transfer_task(task):
            score += 25
    lane = task_rotation_lane(task)
    lane_counts = recent_lanes or {}
    score -= symliquid_control_bonus_for_task(task, lane)
    private_pressure_needed = private_pressure_private_closure_needed()
    type_shape_receiver_needed = private_type_shape_receiver_ablation_needed()
    no_template_execution_shape_needed = not execution_shape_no_template_smoke_guard().get("allowed")
    if concept == "private_pressure_private_closure" and private_pressure_needed:
        score -= 60
    elif concept == "decoder_v2_private_ablation_gate" and private_pressure_needed:
        score += 80
    if no_template_execution_shape_needed:
        if concept == "execution_shape_private_ablation":
            score -= 95
        elif concept.endswith("private_closure") or "private_closure" in concept:
            score += 95
    if concept == "private_type_shape_receiver_veto_ablation" and type_shape_receiver_needed:
        score -= 58
    elif concept == "private_pressure_four_card_recalibration" and type_shape_receiver_needed:
        score += 85
    elif concept == "private_pressure_four_card_recalibration":
        public_guard = private_public_calibration_guard()
        if public_guard.get("allowed"):
            score -= 70
        else:
            score += 80
    if lane == "private_closure_gate":
        recent_code_pressure = int(lane_counts.get("private_closure_gate", 0) or 0) + int(lane_counts.get("code_transfer", 0) or 0)
        score -= 38 if recent_code_pressure == 0 else 12
    elif lane in {"harder_conversation_v2", "board_game_self_play_rl", "long_horizon_tool_use", "repo_repair", "cross_domain_sts_capsules"}:
        if int(lane_counts.get(lane, 0) or 0) == 0:
            score -= 22
    score += lane_rotation_penalty(lane, lane_counts)
    return score, task_source_rank(task), lane_rotation_rank(lane, lane_counts), high_transfer_concept_rank(task), str(task.get("updated_utc") or "")






def symliquid_control_bonus_for_task(task: dict[str, Any], lane: str) -> int:
    """Let SymLiquid state act as a live routing signal, not just a report."""

    state = read_json(REPORTS / "symliquid_state_engine.json", {})
    weights = state.get("action_kind_weights") if isinstance(state.get("action_kind_weights"), dict) else {}
    concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
    key = ""
    if concept == "cross_domain_sts_capsules":
        key = "refresh_cross_domain_sts_capsules"
    elif concept == "repo_repair":
        key = "train_repo_repair_trace_checkpoint"
    elif concept == "long_horizon_tool_use":
        key = "run_long_horizon_tool_use"
    elif concept == "board_game_rl":
        key = "run_board_game_self_play"
    elif concept == "pufferlib4_rl":
        key = "run_pufferlib4_rl_lane"
    elif concept in {"multi_turn_conversation_hard", "multi_turn_conversation_hard_v2"}:
        key = "run_hard_conversation_frontier"
    elif concept == "edge_contract_v2_private_closure":
        key = "train_edge_contract_v2_private_gate"
    elif concept == "private_type_shape_receiver_veto_ablation":
        key = "train_private_semantic_residual_family"
    elif lane in {"private_closure_gate", "typed_interface_skeleton", "code_transfer"}:
        key = "train_private_semantic_residual_family"
    weight = safe_float(weights.get(key)) if key else 0.0
    if weight <= 0:
        return 0
    return int(min(24, max(0, weight * 2.0)))


def task_source_rank(task: dict[str, Any]) -> int:
    source = str(task.get("source") or "")
    kind = str(task.get("kind") or "")
    if source == "high_transfer_curriculum_scheduler":
        return 0
    if source == "feedback_action_queue" and kind == "expand_public_adapter_clean_slice":
        return 1
    if source == "feedback_action_queue":
        return 2
    return 3


def high_transfer_concept_rank(task: dict[str, Any]) -> int:
    if str(task.get("source") or "") != "high_transfer_curriculum_scheduler":
        return 99
    concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
    order = {
        "multi_turn_conversation_hard_v3": 0,
        "multi_turn_conversation_hard_v2": 0,
        "multi_turn_conversation_hard": 0,
        "board_game_rl": 1,
        "pufferlib4_rl": 2,
        "long_horizon_tool_use": 3,
        "repo_repair": 4,
        "cross_domain_sts_capsules": 5,
        "private_pressure_private_closure": 8,
        "decoder_v2_private_ablation_gate": 9,
        "private_type_shape_receiver_veto_ablation": 10,
        "private_pressure_four_card_recalibration": 11,
        "type_contract_diagnostic": 20,
        "execution_shape_private_ablation": 21,
        "execution_shaped_four_card_calibration": 22,
        "edge_exec_repair_four_card_calibration": 23,
        "edge_case_full_body_private_closure_v1": 24,
        "edge_contract_v2_private_closure": 25,
        "typed_interface_skeleton": 28,
        "edge_contract_private_closure": 29,
        "edge_contract_balanced_private_closure_v2": 30,
        "edge_contract_balanced_4card_private_curriculum_v2": 29,
        "edge_case_full_body_private_curriculum_v1": 29,
        "edge_contract_v2_private_residual_curriculum": 29,
        "type_contract_four_card_calibration": 29,
        "edge_contract_4card": 30,
        "typed_interface_private_closure": 32,
        "type_and_return_shape": 33,
        "admissibility_and_interface": 34,
        "edge_conditions": 35,
        "algorithmic_planning": 36,
        "execution_shaped_programs": 37,
        "execution_shape_skeleton_decoder_private_v1": 38,
        "multi_turn_conversation": 80,
        "open_conversation_pantry": 82,
    }
    return order.get(concept, 50)

























def conversation_focus_enabled() -> bool:
    policy = read_json(CONFIGS / "autonomy_policy.json", {})
    focus = str(get_path(policy, ["personality_core", "near_term_training_focus"], "") or "").lower()
    if broad_code_transfer_wall_active():
        return False
    if conversation_lane_graduated():
        return False
    return "conversation" in focus and ("before" in focus or "temporarily" in focus)


def broad_code_transfer_wall_active() -> bool:
    matrix = read_json(REPORTS / "broad_transfer_matrix.json", {})
    summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
    pass_rate = float(
        summary.get("real_public_pass_rate")
        or summary.get("pass_rate")
        or matrix.get("real_public_pass_rate")
        or 0.0
    )
    floor = float(summary.get("floor") or matrix.get("floor") or 0.70)
    below = summary.get("cards_below_floor") or matrix.get("cards_below_floor") or []
    return pass_rate < floor and bool(below)


def conversation_lane_graduated() -> bool:
    lifecycle = high_transfer_concept_lifecycle()
    row = lifecycle.get("multi_turn_conversation") or {}
    if row.get("status") == "regression_only":
        return True
    conversation = read_json(REPORTS / "high_transfer_multi_turn_conversation.json", {})
    return bool(get_path(conversation, ["summary", "graduated"], False) or get_path(conversation, ["summary", "saturated"], False))


def high_transfer_concept_lifecycle() -> dict[str, dict[str, Any]]:
    scheduler = read_json(REPORTS / "high_transfer_curriculum_scheduler.json", {})
    concepts = scheduler.get("concepts") if isinstance(scheduler.get("concepts"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in concepts:
        if not isinstance(row, dict):
            continue
        concept = str(row.get("concept") or "").lower()
        if concept:
            out[concept] = row
    return out


def high_transfer_task_is_regression_only(task: dict[str, Any], lifecycle: dict[str, dict[str, Any]]) -> bool:
    if str(task.get("source") or "") != "high_transfer_curriculum_scheduler":
        return False
    concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
    if not concept:
        return False
    return str((lifecycle.get(concept) or {}).get("status") or "") == "regression_only"


def retire_regression_only_high_transfer_tasks(db_path: Path) -> dict[str, Any]:
    lifecycle = high_transfer_concept_lifecycle()
    regression = {
        concept: row
        for concept, row in lifecycle.items()
        if str(row.get("status") or "") == "regression_only"
    }
    if not regression:
        return {"retired_count": 0, "retired": []}
    tasks = load_tasks(db_path, limit=1000)
    retired: list[dict[str, Any]] = []
    for task in tasks:
        if str(task.get("source") or "") != "high_transfer_curriculum_scheduler":
            continue
        if str(task.get("status") or "") not in READY_STATUSES:
            continue
        concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
        lifecycle_row = regression.get(concept)
        if not lifecycle_row:
            continue
        reason = str(get_path(lifecycle_row, ["evidence", "graduation_reason"], "") or "graduated_to_regression_only")
        detail = {
            "concept": concept,
            "reason": reason,
            "scheduler_status": lifecycle_row.get("status"),
            "task_id": task.get("task_id"),
        }
        update_task(db_path, str(task.get("task_id") or ""), status="done", blocked_reason=f"retired:{reason}")
        add_event(db_path, str(task.get("task_id") or ""), "retired_regression_only_task", detail)
        append_jsonl(FEEDBACK_LEDGER, {"created_utc": now(), "policy": "project_theseus_hive_work_board_feedback_v1", "event": "retired_regression_only_task", **detail})
        retired.append(detail)
    return {"retired_count": len(retired), "retired": retired[:50]}


def latest_high_transfer_task_ids_by_concept() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in current_high_transfer_scheduler_rows():
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        concept = str(payload.get("concept") or "").lower()
        task_id = str(row.get("task_id") or "")
        if concept and task_id:
            out.setdefault(concept, set()).add(task_id)
    return out


def current_high_transfer_scheduler_rows() -> list[dict[str, Any]]:
    """Return only the newest append-only scheduler batch.

    The scheduler JSONL is evidence, not a queue. Older ready rows must not be
    replayed after the current scheduler has moved a concept to waiting,
    regression-only, or blocked by a gate.
    """

    path = REPORTS / "high_transfer_curriculum_tasks.jsonl"
    if not path.exists():
        return []
    rows = [
        row
        for row in read_jsonl(path)[-500:]
        if isinstance(row, dict)
        and str(row.get("source") or "") == "high_transfer_curriculum_scheduler"
        and str(row.get("created_utc") or "")
    ]
    if not rows:
        return []
    latest_created = max(str(row.get("created_utc") or "") for row in rows)
    return [row for row in rows if str(row.get("created_utc") or "") == latest_created]


def sync_high_transfer_scheduler_tasks(db_path: Path) -> dict[str, Any]:
    """Mirror the scheduler's JSONL tasks into the durable work board.

    The scheduler owns the high-transfer curriculum decision, but unattended
    execution is board-driven. Keeping the latest scheduler tasks in SQLite
    prevents generic maintenance work from masking a ready critical training
    pressure after a calibration completes.
    """

    path = REPORTS / "high_transfer_curriculum_tasks.jsonl"
    if not path.exists():
        return {"synced_count": 0, "task_ids": [], "reason": "missing_high_transfer_tasks_jsonl"}
    synced: list[str] = []
    current_rows = current_high_transfer_scheduler_rows()
    for row in current_rows:
        task_id = str(row.get("task_id") or "")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if not task_id or not payload.get("concept"):
            continue
        command = row.get("command")
        if isinstance(command, list):
            command_text = " ".join(str(item) for item in command)
        else:
            command_text = str(command or "")
        desired_status = str(row.get("status") or "ready")
        previous = task_by_id(db_path, task_id)
        concept = str(payload.get("concept") or "").lower()
        diagnostic_reason = non_promotable_diagnostic_reason(concept)
        blocked_reason = ""
        if diagnostic_reason and desired_status in (READY_STATUSES | {"waiting_private_closure", "waiting_recalibration"}):
            previous_status = desired_status
            desired_status = "diagnostic_only"
            blocked_reason = diagnostic_reason
            payload["progress_integrity"] = {
                "promotion_safe": False,
                "diagnostic_only": True,
                "reason": diagnostic_reason,
                "previous_status": previous_status,
                "replacement_concept": promotion_safe_replacement_concept(concept) or None,
            }
            if not previous or str(previous.get("status") or "") != "diagnostic_only":
                append_jsonl(
                    NO_PROGRESS_DEMOTION_LEDGER,
                    {
                        "created_utc": now(),
                        "policy": "project_theseus_progress_integrity_non_promotable_demotion_v1",
                        "task_id": task_id,
                        "concept": concept,
                        "source": "high_transfer_curriculum_scheduler",
                        "status_before": previous_status,
                        "action": "demote",
                        "cluster": f"non_promotable_diagnostic_only:{concept}",
                        "reason": diagnostic_reason,
                        "replacement_concept": promotion_safe_replacement_concept(concept) or None,
                        "rule": "unattended frontier work must optimize semantic transfer, not template/skeleton/candidate-floor surfaces",
                    },
                )
        upsert_task(
            db_path,
            {
                "task_id": task_id,
                "title": str(row.get("title") or f"Train transferable concept: {payload.get('concept')}"),
                "source": "high_transfer_curriculum_scheduler",
                "kind": str(row.get("kind") or "high_transfer_concept_pressure"),
                "status": desired_status,
                "priority": str(row.get("priority") or "medium"),
                "assignee": "high_transfer_curriculum_scheduler",
                "node_id": str(row.get("target_node_id") or "best_training_node"),
                "command": command_text,
                "evidence_json": json.dumps(payload, sort_keys=True),
                "retry_count": 0,
                "blocked_reason": blocked_reason,
            },
        )
        if (
            previous
            and str(previous.get("source") or "") == "high_transfer_curriculum_scheduler"
            and str(previous.get("status") or "") == "done"
            and desired_status in READY_STATUSES
        ):
            update_task(db_path, task_id, status=desired_status, retry_count=0, blocked_reason="")
            add_event(db_path, task_id, "reopened_high_transfer_scheduler_task", {"concept": payload.get("concept"), "status": desired_status})
        add_event(db_path, task_id, "synced_high_transfer_scheduler_task", {"concept": payload.get("concept")})
        synced.append(task_id)
    return {
        "synced_count": len(synced),
        "task_ids": synced[:50],
        "source": str(path.relative_to(ROOT)),
        "scheduler_batch_created_utc": str(current_rows[0].get("created_utc") or "") if current_rows else "",
    }


def supersede_stale_high_transfer_tasks(db_path: Path) -> dict[str, Any]:
    current_ids_by_concept = latest_high_transfer_task_ids_by_concept()
    if not current_ids_by_concept:
        return {"superseded_count": 0, "superseded": []}
    tasks = load_tasks(db_path, limit=5000)
    superseded: list[dict[str, Any]] = []
    supersedable_statuses = READY_STATUSES | {"active"}
    for task in tasks:
        if str(task.get("source") or "") != "high_transfer_curriculum_scheduler":
            continue
        task_status = str(task.get("status") or "")
        if task_status not in supersedable_statuses:
            continue
        concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
        task_id = str(task.get("task_id") or "")
        current_ids = current_ids_by_concept.get(concept)
        if not concept or not task_id or (current_ids and task_id in current_ids):
            continue
        reason = (
            "superseded_by_latest_high_transfer_scheduler_task"
            if current_ids
            else "superseded_by_high_transfer_scheduler_no_longer_ready"
        )
        detail = {
            "task_id": task_id,
            "title": task.get("title"),
            "concept": concept,
            "previous_status": task_status,
            "current_task_ids": sorted(current_ids or []),
            "reason": reason,
        }
        update_task(db_path, task_id, status="done", blocked_reason=reason)
        add_event(db_path, task_id, "superseded_stale_high_transfer_task", detail)
        append_jsonl(FEEDBACK_LEDGER, {"created_utc": now(), "policy": "project_theseus_hive_work_board_feedback_v1", "event": "superseded_stale_high_transfer_task", **detail})
        superseded.append(detail)
    return {"superseded_count": len(superseded), "superseded": superseded[:50]}


def block_stale_feedback_actions(db_path: Path) -> dict[str, Any]:
    lifecycle = high_transfer_concept_lifecycle()
    managed_concepts = {
        concept
        for concept, row in lifecycle.items()
        if str(row.get("status") or "") in {"ready", "regression_only"}
    }
    ready_private_concepts = {
        concept
        for concept, row in lifecycle.items()
        if concept in PRIVATE_SOURCE_AGNOSTIC_CONCEPTS
        and str(row.get("status") or "") == "ready"
    }
    tasks = load_tasks(db_path, limit=5000)
    blocked: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for task in tasks:
        if str(task.get("source") or "") != "feedback_action_queue":
            continue
        if str(task.get("status") or "") not in READY_STATUSES:
            continue
        evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
        kind = str(task.get("kind") or "")
        concept = str(evidence.get("concept") or "").lower()
        blockers = {str(item).lower() for item in evidence.get("blockers") or []}
        next_action = str(evidence.get("next_action") or "")
        teacher_queue_allowed = evidence.get("teacher_queue_allowed")
        public_task_count = first_number(evidence.get("public_task_count"))
        task_id = str(task.get("task_id") or "")

        reason = ""
        terminal_status = ""
        if kind != "expand_public_adapter_clean_slice" and (
            "needs_32_plus_clean_tasks" in blockers
            or next_action == "upgrade_public_task_adapter_to_32_plus_clean_tasks"
            or (kind == "run_same_seed_sts_repair_ablation" and public_task_count is not None and public_task_count < 32)
        ):
            reason = "needs_32_plus_clean_tasks_before_training_or_sts"
            terminal_status = "blocked"
        elif kind == "train_private_semantic_residual_family" and (concept in managed_concepts or ready_private_concepts):
            reason = (
                "superseded_by_canonical_high_transfer_curriculum_task"
                if concept in managed_concepts
                else "superseded_by_ready_private_high_transfer_curriculum_task"
            )
            terminal_status = "done"
        elif kind == "promote_regression_surface" and ready_private_concepts:
            reason = "superseded_by_ready_private_high_transfer_curriculum_task"
            terminal_status = "done"
        elif kind == "write_repo_repair_tasks" and ready_private_concepts:
            reason = "superseded_by_ready_private_high_transfer_curriculum_task"
            terminal_status = "done"
        elif kind == "request_teacher_architecture_diagnosis" and (
            teacher_queue_allowed is False or ready_private_concepts
        ):
            reason = (
                "teacher_queue_not_allowed"
                if teacher_queue_allowed is False
                else "superseded_by_ready_private_high_transfer_curriculum_task"
            )
            terminal_status = "done"

        if not reason or not task_id:
            continue
        detail = {
            "task_id": task_id,
            "title": task.get("title"),
            "kind": kind,
            "concept": concept or None,
            "card_id": evidence.get("card_id"),
            "public_task_count": public_task_count,
            "reason": reason,
        }
        update_task(db_path, task_id, status=terminal_status, blocked_reason=reason)
        add_event(db_path, task_id, "feedback_action_guard", detail)
        append_jsonl(FEEDBACK_LEDGER, {"created_utc": now(), "policy": "project_theseus_hive_work_board_feedback_v1", "event": "feedback_action_guard", "status": terminal_status, **detail})
        if terminal_status == "blocked":
            blocked.append(detail)
        else:
            completed.append(detail)
    return {
        "blocked_count": len(blocked),
        "superseded_count": len(completed),
        "blocked": blocked[:50],
        "superseded": completed[:50],
    }


def retire_reclassified_teacher_escalations(db_path: Path) -> dict[str, Any]:
    live = read_json(REPORTS / "broad_transfer_closure_runner_source_livecodebench.json", {})
    live_summary = live.get("summary") if isinstance(live.get("summary"), dict) else {}
    live_coverage_repaired = (
        int(live_summary.get("public_task_count") or 0) >= 32
        and int(live_summary.get("hard_step_failure_count") or 0) == 0
    )
    hard_conversation = read_json(REPORTS / "high_transfer_multi_turn_conversation_hard.json", {})
    hard_conversation_summary = (
        hard_conversation.get("summary") if isinstance(hard_conversation.get("summary"), dict) else {}
    )
    hard_conversation_frontier_valid = (
        hard_conversation.get("policy") == "project_theseus_multi_turn_conversation_benchmark_v1"
        and hard_conversation.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(hard_conversation_summary.get("turn_count") or 0) > 0
        and int(hard_conversation_summary.get("personality_context_ready_turns") or 0)
        == int(hard_conversation_summary.get("turn_count") or 0)
    )
    if not live_coverage_repaired and not hard_conversation_frontier_valid:
        return {"retired_count": 0, "retired": []}

    tasks = load_tasks(db_path, limit=5000)
    retired: list[dict[str, Any]] = []
    for task in tasks:
        if str(task.get("source") or "") != "auto_teacher_escalation":
            continue
        if str(task.get("kind") or "") != "teacher_architecture_escalation":
            continue
        if str(task.get("status") or "") not in READY_STATUSES:
            continue
        evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
        residual_cluster = str(evidence.get("residual_cluster") or "")
        if residual_cluster == "no_progress:expand_public_adapter_clean_slice" and live_coverage_repaired:
            reason = "reclassified_as_livecodebench_32_task_adapter_coverage_success"
            detail = {
                "task_id": task.get("task_id"),
                "title": task.get("title"),
                "residual_cluster": evidence.get("residual_cluster"),
                "livecodebench_public_task_count": live_summary.get("public_task_count"),
                "livecodebench_public_pass_rate": live_summary.get("public_pass_rate"),
                "livecodebench_trigger_state": live.get("trigger_state"),
                "reason": reason,
            }
        elif residual_cluster == "command_failed:multi_turn_conversation_hard" and hard_conversation_frontier_valid:
            reason = "reclassified_as_valid_hard_conversation_frontier_evidence"
            detail = {
                "task_id": task.get("task_id"),
                "title": task.get("title"),
                "residual_cluster": evidence.get("residual_cluster"),
                "hard_conversation_trigger_state": hard_conversation.get("trigger_state"),
                "hard_conversation_accuracy": hard_conversation_summary.get("accuracy"),
                "hard_conversation_turn_count": hard_conversation_summary.get("turn_count"),
                "reason": reason,
            }
        else:
            continue
        update_task(db_path, str(task.get("task_id") or ""), status="done", blocked_reason=reason)
        add_event(db_path, str(task.get("task_id") or ""), "retired_reclassified_teacher_escalation", detail)
        append_jsonl(FEEDBACK_LEDGER, {"created_utc": now(), "policy": "project_theseus_hive_work_board_feedback_v1", "event": "retired_reclassified_teacher_escalation", **detail})
        retired.append(detail)
    return {"retired_count": len(retired), "retired": retired[:50]}


def satisfy_fresh_high_transfer_targets(db_path: Path) -> dict[str, Any]:
    """Retire ready high-transfer tasks whose target report is already useful.

    The scheduler can legitimately keep a lane marked ready while the work
    board still has an already-green target from the current rotation window.
    Without this guard, unattended loops churn reports such as cross-domain
    capsules instead of moving to the next frontier.
    """

    tasks = load_tasks(db_path, limit=5000)
    satisfied: list[dict[str, Any]] = []
    for task in tasks:
        if str(task.get("source") or "") != "high_transfer_curriculum_scheduler":
            continue
        if str(task.get("status") or "") not in READY_STATUSES:
            continue
        concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
        if concept not in SATISFIABLE_HIGH_TRANSFER_TARGETS:
            continue
        target = high_transfer_target_satisfied(concept)
        if not target.get("satisfied"):
            continue
        task_id = str(task.get("task_id") or "")
        reason = f"satisfied_fresh_target:{target.get('reason')}"
        detail = {
            "task_id": task_id,
            "title": task.get("title"),
            "concept": concept,
            "lane": task_rotation_lane(task),
            "target_report": target.get("report"),
            "target_age_seconds": target.get("age_seconds"),
            "reason": reason,
        }
        update_task(db_path, task_id, status="done", blocked_reason=reason)
        add_event(db_path, task_id, "satisfied_fresh_high_transfer_target", detail)
        improvement = {
            "passed": True,
            "signals": [signal("stale_tool_retired", 0, 1, target.get("report"))],
            "signal_kinds": ["stale_tool_retired"],
            "residual_cluster": f"progress:{concept}:fresh_target_already_satisfied",
            "satisfied_without_rerun": True,
        }
        append_jsonl(IMPROVEMENT_LEDGER, improvement_ledger_row(task, "done", reason, improvement))
        append_jsonl(
            FEEDBACK_LEDGER,
            {
                "created_utc": now(),
                "policy": "project_theseus_hive_work_board_feedback_v1",
                "event": "satisfied_fresh_high_transfer_target",
                **detail,
            },
        )
        satisfied.append(detail)
    return {"satisfied_count": len(satisfied), "satisfied": satisfied[:50]}


def retire_stale_hive_task_queue_chunks(db_path: Path) -> dict[str, Any]:
    """Retire old utilization chunks that can no longer be a useful lease.

    The utilization manager can enqueue short-lived maintenance/training chunks.
    If those chunks sit for hours, selecting them later is stale churn rather
    than learning. Fresh utilization work may still be enqueued by the manager;
    this only clears aged queue artifacts.
    """

    tasks = load_tasks(db_path, limit=10000)
    retired: list[dict[str, Any]] = []
    for task in tasks:
        if str(task.get("source") or "") != "hive_task_queue":
            continue
        if str(task.get("status") or "") not in READY_STATUSES:
            continue
        age = iso_age_seconds(str(task.get("created_utc") or task.get("updated_utc") or ""))
        if age is None or age < STALE_HIVE_QUEUE_SECONDS:
            continue
        task_id = str(task.get("task_id") or "")
        reason = "retired_stale_hive_utilization_chunk"
        detail = {
            "task_id": task_id,
            "title": task.get("title"),
            "kind": task.get("kind"),
            "age_seconds": age,
            "reason": reason,
        }
        update_task(db_path, task_id, status="done", blocked_reason=reason)
        add_event(db_path, task_id, "retired_stale_hive_task_queue_chunk", detail)
        improvement = {
            "passed": True,
            "signals": [signal("stale_tool_retired", 0, 1, "hive_task_queue")],
            "signal_kinds": ["stale_tool_retired"],
            "residual_cluster": "progress:hive_task_queue:stale_utilization_chunk_retired",
        }
        append_jsonl(IMPROVEMENT_LEDGER, improvement_ledger_row(task, "done", reason, improvement))
        append_jsonl(
            FEEDBACK_LEDGER,
            {
                "created_utc": now(),
                "policy": "project_theseus_hive_work_board_feedback_v1",
                "event": "retired_stale_hive_task_queue_chunk",
                **detail,
            },
        )
        retired.append(detail)
    return {"retired_count": len(retired), "retired": retired[:50], "max_age_seconds": STALE_HIVE_QUEUE_SECONDS}


def high_transfer_target_satisfied(concept: str) -> dict[str, Any]:
    concept = str(concept or "").lower()
    if concept in {"multi_turn_conversation_hard", "multi_turn_conversation_hard_v2", "multi_turn_conversation_hard_v3"}:
        return conversation_hard_target_satisfied(concept)
    if concept == "board_game_rl":
        return board_game_target_satisfied()
    if concept == "pufferlib4_rl":
        return pufferlib4_target_satisfied()
    if concept == "long_horizon_tool_use":
        return long_horizon_target_satisfied()
    if concept == "repo_repair":
        return repo_repair_target_satisfied()
    if concept == "cross_domain_sts_capsules":
        return cross_domain_target_satisfied()
    if concept == "execution_shape_private_ablation":
        return execution_shape_ablation_target_satisfied()
    if concept == "private_type_shape_receiver_veto_ablation":
        return private_type_shape_receiver_target_satisfied()
    if concept == "decoder_v2_private_ablation_gate":
        return decoder_v2_private_gate_target_satisfied()
    return {"satisfied": False, "reason": "no_satisfaction_policy"}


def fresh_report(path: Path) -> tuple[dict[str, Any], float | None]:
    report = read_json(path, {})
    if not report:
        return {}, None
    stamp = str(report.get("created_utc") or report.get("generated_at") or "")
    return report, iso_age_seconds(stamp)


def conversation_hard_target_satisfied(concept: str) -> dict[str, Any]:
    if concept == "multi_turn_conversation_hard_v3":
        path = REPORTS / "high_transfer_multi_turn_conversation_hard_v3.json"
        min_cases = 256
        min_accuracy = 0.95
    elif concept == "multi_turn_conversation_hard_v2":
        path = REPORTS / "high_transfer_multi_turn_conversation_hard_v2.json"
        min_cases = 128
        min_accuracy = 0.90
    else:
        path = REPORTS / "high_transfer_multi_turn_conversation_hard.json"
        min_cases = 64
        min_accuracy = 0.90
    report, age = fresh_report(path)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    accuracy = safe_float(summary.get("accuracy"))
    ok = bool(
        age is not None
        and age <= FRESH_TARGET_MAX_AGE_SECONDS
        and report.get("policy") == "project_theseus_multi_turn_conversation_benchmark_v1"
        and report.get("trigger_state") == "GREEN"
        and int(summary.get("case_count") or 0) >= min_cases
        and accuracy >= min_accuracy
        and bool(summary.get("graduated") or summary.get("saturated"))
    )
    return {
        "satisfied": ok,
        "report": rel(path),
        "age_seconds": age,
        "min_cases": min_cases,
        "min_accuracy": min_accuracy,
        "accuracy": accuracy,
        "reason": "hard_conversation_green_saturated" if ok else "not_fresh_green_saturated",
    }


def board_game_target_satisfied() -> dict[str, Any]:
    path = REPORTS / "board_game_rl_benchmark.json"
    report, age = fresh_report(path)
    gates = report.get("gates") if isinstance(report.get("gates"), list) else []
    trace_path = resolve(str(get_path(report, ["outputs", "traces"], "reports/board_game_rl_traces.jsonl")))
    ok = bool(
        age is not None
        and age <= FRESH_TARGET_MAX_AGE_SECONDS
        and report.get("policy") == "project_theseus_board_game_rl_benchmark_v1"
        and report.get("trigger_state") == "GREEN"
        and gates
        and all(bool(gate.get("ok")) for gate in gates if isinstance(gate, dict))
        and trace_path.exists()
    )
    return {"satisfied": ok, "report": rel(path), "age_seconds": age, "reason": "board_game_self_play_traces_green" if ok else "not_fresh_green_with_traces"}


def pufferlib4_target_satisfied() -> dict[str, Any]:
    path = REPORTS / "pufferlib4_rl_lane.json"
    report, age = fresh_report(path)
    trigger = str(report.get("trigger_state") or "")
    capsule_count = int(get_path(report, ["summary", "capsule_count"], 0) or 0)
    native_ready = bool(get_path(report, ["summary", "native_backend_ready"]))
    improvement_signal = str(get_path(report, ["summary", "improvement_signal"], "") or "")
    ok = bool(
        age is not None
        and age <= FRESH_TARGET_MAX_AGE_SECONDS
        and report.get("policy") == "project_theseus_pufferlib4_rl_lane_v1"
        and trigger in {"GREEN", "YELLOW"}
        and capsule_count > 0
        and improvement_signal in {"new_clean_evidence_produced", "useful_failure_residual_captured"}
    )
    if ok and not native_ready:
        reason = "pufferlib4_backend_residual_captured_with_capsules"
    elif ok:
        reason = "pufferlib4_ocean_lane_fresh_green"
    else:
        reason = "not_fresh_pufferlib4_rl_lane_with_capsules"
    return {"satisfied": ok, "report": rel(path), "age_seconds": age, "reason": reason}


def long_horizon_target_satisfied() -> dict[str, Any]:
    path = REPORTS / "high_transfer_long_horizon_tool_use.json"
    report, age = fresh_report(path)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    ok = bool(
        age is not None
        and age <= FRESH_TARGET_MAX_AGE_SECONDS
        and report.get("policy") == "project_theseus_long_horizon_tool_use_benchmark_v1"
        and report.get("trigger_state") == "GREEN"
        and int(summary.get("case_count") or 0) >= 64
        and safe_float(summary.get("pass_rate")) >= 0.85
        and int(summary.get("sts_rows") or 0) >= 64
    )
    return {"satisfied": ok, "report": rel(path), "age_seconds": age, "reason": "long_horizon_tool_use_64_case_green" if ok else "not_fresh_64_case_tool_use_green"}


def repo_repair_target_satisfied() -> dict[str, Any]:
    path = REPORTS / "high_transfer_repo_repair_learner.json"
    report, age = fresh_report(path)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    ok = bool(
        age is not None
        and age <= FRESH_TARGET_MAX_AGE_SECONDS
        and report.get("policy") == "project_theseus_viea_repo_repair_learner_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(summary.get("validated_private_trace_count") or 0) >= 128
        and int(summary.get("code_lm_row_count") or 0) >= 128
    )
    return {"satisfied": ok, "report": rel(path), "age_seconds": age, "reason": "repo_repair_frontier_traces_fresh" if ok else "not_fresh_repo_repair_frontier_traces"}


def cross_domain_target_satisfied() -> dict[str, Any]:
    path = REPORTS / "cross_domain_sts_capsules.json"
    report, age = fresh_report(path)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    ok = bool(
        age is not None
        and age <= FRESH_TARGET_MAX_AGE_SECONDS
        and report.get("policy") == "project_theseus_cross_domain_sts_capsules_v1"
        and report.get("trigger_state") == "GREEN"
        and int(summary.get("capsule_count") or 0) > 0
        and int(summary.get("sts_row_count") or 0) > 0
    )
    return {"satisfied": ok, "report": rel(path), "age_seconds": age, "reason": "cross_domain_sts_capsules_fresh_green" if ok else "not_fresh_capsules_green"}


def execution_shape_ablation_target_satisfied() -> dict[str, Any]:
    """Treat a fresh failed private execution-shape gate as useful evidence.

    The ablation is a private diagnostic gate before public calibration. If it
    already proved that the skeleton decoder still has no-admissible-candidate
    or zero-category coverage residuals, rerunning it without a decoder patch is
    churn. Mark the current board task satisfied so rotation can move to the
    next architecture-pressure step.
    """

    path = REPORTS / "execution_shape_private_ablation.json"
    report, age = fresh_report(path)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    trigger = str(report.get("trigger_state") or "")
    public_ready = bool(
        report.get("ready_for_public_calibration")
        or report.get("private_ablation_public_gate_ready")
        or summary.get("ready_for_public_calibration")
        or summary.get("private_ablation_public_gate_ready")
    )
    eval_count = int(summary.get("private_eval_task_count") or 0)
    candidate_rows = int(summary.get("candidate_rows") or 0)
    no_admissible = int(summary.get("skeleton_no_admissible_candidate_count") or 0)
    zero_categories = summary.get("skeleton_zero_pass_categories") or []
    dominant = str(summary.get("dominant_residual") or "")
    decoder_current = bool(path.exists() and decoder_relevant_source_mtime() <= path.stat().st_mtime)
    diagnostic = bool(
        trigger == "YELLOW"
        and not public_ready
        and eval_count > 0
        and candidate_rows > 0
        and (no_admissible > 0 or bool(zero_categories) or dominant)
    )
    ready = bool(trigger == "GREEN" and public_ready and eval_count > 0 and candidate_rows > 0)
    ok = bool(
        age is not None
        and age <= FRESH_TARGET_MAX_AGE_SECONDS
        and report.get("policy") == "project_theseus_execution_shape_private_ablation_v1"
        and decoder_current
        and (diagnostic or ready)
    )
    if ok and diagnostic:
        reason = "execution_shape_private_ablation_fresh_diagnostic_requires_decoder_patch"
    elif ok:
        reason = "execution_shape_private_ablation_green_ready"
    elif not decoder_current:
        reason = "execution_shape_private_ablation_stale_after_decoder_source_change"
    else:
        reason = "not_fresh_execution_shape_ablation_diagnostic"
    return {
        "satisfied": ok,
        "report": rel(path),
        "age_seconds": age,
        "reason": reason,
    }


def private_type_shape_receiver_target_satisfied() -> dict[str, Any]:
    path = REPORTS / "private_type_shape_receiver_ablation.json"
    report, age = fresh_report(path)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    ok = bool(
        age is not None
        and age <= FRESH_TARGET_MAX_AGE_SECONDS
        and report.get("policy") == "project_theseus_private_type_shape_receiver_veto_ablation_v1"
        and report.get("trigger_state") == "GREEN"
        and bool(report.get("ready_for_public_calibration") or summary.get("ready_for_public_calibration"))
        and int(summary.get("leakage_violation_count") or 0) == 0
        and int(summary.get("template_like_candidate_count") or 0) == 0
        and int(summary.get("wrapper_like_candidate_count") or 0) == 0
    )
    return {
        "satisfied": ok,
        "report": rel(path),
        "age_seconds": age,
        "reason": "private_type_shape_receiver_ablation_green_ready" if ok else "not_fresh_receiver_ablation_green_ready",
    }


def decoder_v2_private_gate_target_satisfied() -> dict[str, Any]:
    path = REPORTS / "decoder_v2_private_ablation_gate.json"
    report, age = fresh_report(path)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    ok = bool(
        age is not None
        and age <= FRESH_TARGET_MAX_AGE_SECONDS
        and report.get("policy") == "project_theseus_decoder_v2_private_ablation_gate_v1"
        and report.get("trigger_state") == "GREEN"
        and bool(report.get("ready_for_public_calibration") or summary.get("ready_for_public_calibration"))
        and bool(summary.get("private_signal_positive"))
    )
    return {
        "satisfied": ok,
        "report": rel(path),
        "age_seconds": age,
        "reason": "decoder_v2_private_ablation_gate_green_ready" if ok else "not_fresh_decoder_v2_gate_green_ready",
    }


def iso_age_seconds(stamp: str) -> float | None:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())
    except ValueError:
        return None


def task_assigned_to_other_node(task: dict[str, Any], node_context: dict[str, Any]) -> bool:
    node_id = str(task.get("node_id") or "")
    if node_id in {"", "local", "unassigned", "best_training_node"}:
        return False
    local = node_context.get("local_node") if isinstance(node_context.get("local_node"), dict) else {}
    local_names = {str(local.get("node_id") or ""), str(local.get("node_name") or "")}
    return node_id not in local_names


def assign_task(task: dict[str, Any], node_context: dict[str, Any]) -> dict[str, Any]:
    kind = str(task.get("kind") or "")
    source = str(task.get("source") or "")
    concept = str(get_path(task, ["evidence", "concept"], "") or "")
    scheduler_summary = node_context.get("scheduler_summary") if isinstance(node_context.get("scheduler_summary"), dict) else {}
    local = node_context.get("local_node") if isinstance(node_context.get("local_node"), dict) else {}
    target = "local"
    node_id = local.get("node_id") or "local"
    node_name = local.get("node_name") or "local"
    reason = "local_default"
    if source == "high_transfer_curriculum_scheduler" and concept in CODE_HIGH_TRANSFER_CONCEPTS:
        candidate = scheduler_summary.get("best_cuda_node") or scheduler_summary.get("best_training_node") or node_name
        reason = "cuda_node_for_code_semantic_training"
        node_name = str(candidate)
        node_id = str(candidate)
        target = "scheduler_selected"
    elif source == "high_transfer_curriculum_scheduler" and concept in {"open_conversation_pantry", "multi_turn_conversation", "multi_turn_conversation_hard"}:
        candidate = node_name
        reason = "local_conversation_first_lane"
        node_name = str(candidate)
        node_id = str(candidate)
        target = "local"
    elif source == "high_transfer_curriculum_scheduler" and concept in {"repo_repair", "long_horizon_tool_use", "cross_domain_sts_capsules"}:
        candidate = node_name
        reason = "cpu_or_coordinator_node_for_repo_and_long_horizon_work"
        node_name = str(candidate)
        node_id = str(candidate)
        target = "local"
    elif source == "high_transfer_curriculum_scheduler" and concept == "board_game_rl":
        candidate = node_name
        reason = "local_board_game_rl_lane"
        node_name = str(candidate)
        node_id = str(candidate)
        target = "local"
    elif source == "high_transfer_curriculum_scheduler" and concept == "pufferlib4_rl":
        candidate = node_name
        reason = "local_or_builder_node_for_pufferlib4_rl_lane"
        node_name = str(candidate)
        node_id = str(candidate)
        target = "local"
    elif "teacher" in kind:
        candidate = node_name
        reason = "coordinator_node_for_teacher_architecture_gate"
        node_name = str(candidate)
        node_id = str(candidate)
        target = "local"
    elif any(token in kind for token in ["train", "transfer", "repo_repair", "curriculum"]):
        candidate = scheduler_summary.get("best_cuda_node") or scheduler_summary.get("best_training_node") or node_name
        reason = "best_training_node_for_learning_task"
        node_name = str(candidate)
        node_id = str(candidate)
        target = "scheduler_selected"
    if node_context.get("version_drift_blocking"):
        return {"allowed": False, "target": target, "node_id": node_id, "node_name": node_name, "reason": "version_drift_blocking"}
    local_names = {str(local.get("node_id") or ""), str(local.get("node_name") or ""), "local", ""}
    dispatch_mode = "local_runner" if str(node_id) in local_names or str(node_name) in local_names else "remote_node_selected"
    return {"allowed": True, "target": target, "node_id": node_id, "node_name": node_name, "reason": reason, "dispatch_mode": dispatch_mode}


def execute_task(db_path: Path, task: dict[str, Any], node_context: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    assignment = task.get("assignment") if isinstance(task.get("assignment"), dict) else assign_task(task, node_context)
    if not assignment.get("allowed"):
        update_task(db_path, task_id, status="blocked", blocked_reason=str(assignment.get("reason") or "assignment_blocked"))
        add_event(db_path, task_id, "assignment_blocked", assignment)
        return {"task_id": task_id, "status": "blocked", "reason": assignment.get("reason")}
    if assignment.get("dispatch_mode") == "remote_node_selected" and not args.force_local_remote_assignment:
        update_task(db_path, task_id, status="queued", assignee=str(assignment.get("node_name") or ""), node_id=str(assignment.get("node_id") or ""))
        add_event(db_path, task_id, "assigned_to_remote_node", assignment)
        append_jsonl(LEDGER, ledger_row(task, "queued_remote", "assigned_to_remote_node", {"assignment": assignment}))
        return {"task_id": task_id, "title": task.get("title"), "status": "queued_remote", "reason": "assigned_to_remote_node", "assignment": assignment}

    update_task(db_path, task_id, status="active", assignee=str(assignment.get("node_name") or "local"), node_id=str(assignment.get("node_id") or "local"))
    add_event(db_path, task_id, "task_started", {"assignment": assignment})
    command_spec = command_for_task(task, args)
    if not command_spec["allowed"]:
        update_task(db_path, task_id, status="blocked", blocked_reason=command_spec["reason"])
        add_event(db_path, task_id, "command_blocked", command_spec)
        append_jsonl(LEDGER, ledger_row(task, "blocked", command_spec["reason"], command_spec))
        return {"task_id": task_id, "status": "blocked", "reason": command_spec["reason"]}
    training_gate = code_lm_training_command_gate(command_spec)
    if not training_gate.get("allowed"):
        update_task(db_path, task_id, status="blocked", blocked_reason=str(training_gate.get("reason")))
        add_event(db_path, task_id, "code_lm_training_gate_blocked", training_gate)
        append_jsonl(LEDGER, ledger_row(task, "blocked", str(training_gate.get("reason")), training_gate))
        return {
            "task_id": task_id,
            "status": "blocked",
            "reason": training_gate.get("reason"),
            "guard": training_gate,
        }

    command = command_spec["command"]
    hook_target = command_spec.get("hook_target", "board_task")
    record_hook("before", hook_target, task, command, {"timeout_seconds": args.timeout_seconds})
    before_metrics = snapshot_improvement_metrics()
    result = run_subprocess(
        command,
        timeout=max(60, int(args.timeout_seconds)),
        env_overrides=command_spec.get("env") if isinstance(command_spec.get("env"), dict) else None,
    )
    postprocess_results = run_postprocess_commands(command_spec, args=args) if (
        result.get("ok") or command_spec.get("postprocess_on_failure")
    ) else []
    ingested_evidence = ingest_command_evidence(command_spec)
    after_metrics = snapshot_improvement_metrics()
    improvement = classify_improvement(task, command_spec, result, before_metrics, after_metrics)
    ok = bool(result.get("ok"))
    frontier_ok = useful_frontier_evidence_returned(task, command_spec, result, improvement)
    if frontier_ok:
        ok = True
        improvement = normalize_frontier_evidence_improvement(task, improvement)
    status = "done" if ok else "failed"
    reason = (
        "frontier_report_returned_nonzero_but_valid_evidence"
        if frontier_ok
        else "command_returned_zero"
        if ok
        else f"returncode_{result.get('returncode')}"
    )
    retry_count = int(task.get("retry_count") or 0) + (0 if ok else 1)
    if not ok and retry_count > int(args.max_retries):
        status = "blocked"
        reason = "retry_limit_exceeded"
    no_progress_decision = no_progress_demote_or_block(task, status, reason, improvement, result)
    if no_progress_decision.get("action") == "block":
        status = "blocked"
        reason = str(no_progress_decision.get("reason") or "no_improvement_signal_repeated")
    elif no_progress_decision.get("action") == "demote" and status == "done":
        reason = str(no_progress_decision.get("reason") or "done_no_improvement_demoted")
    update_task(db_path, task_id, status=status, retry_count=retry_count, blocked_reason="" if status != "blocked" else reason)
    add_event(db_path, task_id, f"task_{status}", {"reason": reason, "result": compact_result(result), "postprocess_results": [compact_result(row) for row in postprocess_results], "improvement_contract": improvement, "no_progress_decision": no_progress_decision, "report_evidence_store": ingested_evidence})
    add_evidence(db_path, task_id, "hive_work_board_executor", rel(DEFAULT_OUT), "diagnostic")
    for row in ingested_evidence:
        if row.get("report_path"):
            add_evidence(db_path, task_id, f"report_evidence:{row.get('family')}", str(row.get("report_path")), "claim_bearing")
    append_jsonl(LEDGER, ledger_row(task, status, reason, {"command": command, "result": compact_result(result), "postprocess_results": [compact_result(row) for row in postprocess_results], "assignment": assignment, "improvement_contract": improvement, "no_progress_decision": no_progress_decision, "report_evidence_store": ingested_evidence}))
    append_jsonl(IMPROVEMENT_LEDGER, improvement_ledger_row(task, status, reason, improvement))
    append_jsonl(FEEDBACK_LEDGER, feedback_row(task, status, reason, result, improvement))
    escalation = maybe_enqueue_teacher_escalation(db_path, task, improvement, status=status, reason=reason)
    record_hook("after", hook_target, task, command, {"status": status, "reason": reason, "returncode": result.get("returncode"), "timeout_seconds": args.timeout_seconds})
    return {
        "task_id": task_id,
        "title": task.get("title"),
        "status": status,
        "reason": reason,
        "returncode": result.get("returncode"),
        "assignment": assignment,
        "runtime_ms": result.get("runtime_ms"),
        "postprocess_results": [compact_result(row) for row in postprocess_results],
        "report_evidence_store": ingested_evidence,
        "improvement_contract": improvement,
        "no_progress_decision": no_progress_decision,
        "teacher_escalation": escalation,
    }





if __name__ == "__main__":
    raise SystemExit(main())
