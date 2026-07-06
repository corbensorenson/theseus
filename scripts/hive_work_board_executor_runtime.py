"""Runtime/reporting helpers for Hive Work Board Executor.

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




def task_rotation_lane(task: dict[str, Any]) -> str:
    concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
    if concept.endswith("private_closure") or concept.endswith("private_closure_v1") or concept.endswith("private_closure_v2"):
        return "private_closure_gate"
    if concept in {"private_pressure_private_closure", "typed_interface_private_closure", "private_type_shape_receiver_veto_ablation"}:
        return "private_closure_gate"
    if concept == "decoder_v2_private_ablation_gate":
        return "private_closure_gate"
    if concept == "typed_interface_skeleton":
        return "typed_interface_skeleton"
    if concept in {"multi_turn_conversation_hard", "multi_turn_conversation_hard_v2", "multi_turn_conversation_hard_v3", "multi_turn_conversation_hard_v4"}:
        return "harder_conversation_v2"
    if concept == "board_game_rl":
        return "board_game_self_play_rl"
    if concept == "pufferlib4_rl":
        return "pufferlib4_rl"
    if concept == "long_horizon_tool_use":
        return "long_horizon_tool_use"
    if concept == "repo_repair":
        return "repo_repair"
    if concept == "cross_domain_sts_capsules":
        return "cross_domain_sts_capsules"
    if is_code_transfer_task(task):
        return "code_transfer"
    return "other"


def recent_rotation_lane_counts(limit: int = 18) -> dict[str, int]:
    counts: dict[str, int] = {}
    rows = read_jsonl(IMPROVEMENT_LEDGER)[-max(1, limit):]
    for row in rows:
        lane = rotation_lane_for_concept(str(row.get("concept") or ""), str(row.get("kind") or ""))
        counts[lane] = counts.get(lane, 0) + 1
    return counts


def rotation_lane_for_concept(concept: str, kind: str = "") -> str:
    concept = str(concept or "").lower()
    kind = str(kind or "").lower()
    if concept.endswith("private_closure") or concept.endswith("private_closure_v1") or concept.endswith("private_closure_v2"):
        return "private_closure_gate"
    if concept in {"private_pressure_private_closure", "typed_interface_private_closure", "private_type_shape_receiver_veto_ablation"}:
        return "private_closure_gate"
    if concept == "decoder_v2_private_ablation_gate":
        return "private_closure_gate"
    if concept == "typed_interface_skeleton":
        return "typed_interface_skeleton"
    if concept in {"multi_turn_conversation_hard", "multi_turn_conversation_hard_v2", "multi_turn_conversation_hard_v3", "multi_turn_conversation_hard_v4"} or "conversation" in kind:
        return "harder_conversation_v2"
    if concept == "board_game_rl":
        return "board_game_self_play_rl"
    if concept == "pufferlib4_rl":
        return "pufferlib4_rl"
    if concept == "long_horizon_tool_use":
        return "long_horizon_tool_use"
    if concept == "repo_repair" or "repo_repair" in kind:
        return "repo_repair"
    if concept == "cross_domain_sts_capsules":
        return "cross_domain_sts_capsules"
    if concept in CODE_HIGH_TRANSFER_CONCEPTS or "code" in kind or "transfer" in kind:
        return "code_transfer"
    return "other"


def lane_rotation_penalty(lane: str, recent_lanes: dict[str, int]) -> int:
    count = int(recent_lanes.get(lane, 0) or 0)
    if lane == "private_closure_gate":
        code_count = int(recent_lanes.get("code_transfer", 0) or 0)
        return -8 if (count + code_count) == 0 else 18 * (count + code_count)
    if lane == "code_transfer":
        return 45 * count
    if lane != "other" and count == 0:
        return -8
    return 25 * count


def lane_rotation_rank(lane: str, recent_lanes: dict[str, int]) -> int:
    base = LANE_ROTATION_ORDER.get(lane, LANE_ROTATION_ORDER["other"])
    return base + (10 if int(recent_lanes.get(lane, 0) or 0) else 0)


def is_conversation_task(task: dict[str, Any]) -> bool:
    concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
    haystack = " ".join(
        str(task.get(key) or "").lower()
        for key in ["title", "source", "kind", "command"]
    )
    return concept in {"open_conversation_pantry", "multi_turn_conversation", "multi_turn_conversation_hard", "multi_turn_conversation_hard_v2", "multi_turn_conversation_hard_v3"} or "conversation" in haystack or "checkpoint_chat" in haystack


def is_generalist_high_transfer_task(task: dict[str, Any]) -> bool:
    if str(task.get("source") or "") != "high_transfer_curriculum_scheduler":
        return False
    concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
    return concept in GENERALIST_HIGH_TRANSFER_CONCEPTS


def is_code_transfer_task(task: dict[str, Any]) -> bool:
    concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
    card_id = str(get_path(task, ["evidence", "card_id"], "") or "").lower()
    haystack = " ".join(
        str(task.get(key) or "").lower()
        for key in ["title", "source", "kind", "command"]
    )
    if concept in {"open_conversation_pantry", "multi_turn_conversation", "multi_turn_conversation_hard", "multi_turn_conversation_hard_v2", "multi_turn_conversation_hard_v3"}:
        return False
    return (
        "source_" in card_id
        or "broad_transfer" in haystack
        or "code" in haystack
        or "repo_repair" in haystack
        or "programming" in haystack
        or concept in CODE_HIGH_TRANSFER_CONCEPTS
        or concept == "repo_repair"
    )


def run_postprocess_commands(command_spec: dict[str, Any], *, args: argparse.Namespace) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for command in command_spec.get("postprocess_commands") or []:
        if not isinstance(command, list) or not command:
            continue
        timeout = min(max(60, int(args.timeout_seconds)), 600)
        results.append(run_subprocess([str(part) for part in command], timeout=timeout))
    return results


def useful_frontier_evidence_returned(
    task: dict[str, Any],
    command_spec: dict[str, Any],
    result: dict[str, Any],
    improvement: dict[str, Any],
) -> bool:
    """Treat YELLOW frontier reports as useful completed work, not process failure.

    Several benchmark/frontier scripts return a non-zero code when the surface
    remains below graduation threshold. For unattended learning, that is still
    a successful board step when a valid report was written and the improvement
    contract produced evidence. True RED reports and commands without evidence
    stay failed/blocked.
    """

    if result.get("ok"):
        return False
    if int(result.get("returncode") or 0) not in {1, 2}:
        return False
    if str(task.get("source") or "") != "high_transfer_curriculum_scheduler":
        return False
    if not bool(improvement.get("passed")):
        return False
    valid_policies = {
        "project_theseus_multi_turn_conversation_benchmark_v1",
        "project_theseus_broad_transfer_closure_runner_v1",
        "project_theseus_code_residual_curriculum_v1",
        "project_theseus_execution_shape_private_ablation_v1",
        "project_theseus_type_contract_diagnostic_v1",
        "project_theseus_viea_repo_repair_learner_v1",
        "project_theseus_board_game_rl_benchmark_v1",
        "project_theseus_pufferlib4_rl_lane_v1",
        "project_theseus_pufferlib4_capability_probe_v1",
        "project_theseus_long_horizon_tool_use_benchmark_v1",
        "project_theseus_cross_domain_sts_capsules_v1",
    }
    for raw_path in command_spec.get("evidence_paths") or []:
        path = resolve(str(raw_path))
        if path.suffix.lower() != ".json" or not path.exists():
            continue
        report = read_json(path, {})
        if str(report.get("policy") or "") not in valid_policies:
            continue
        if str(report.get("trigger_state") or "") in {"GREEN", "YELLOW"}:
            return True
    return False


def normalize_frontier_evidence_improvement(task: dict[str, Any], improvement: dict[str, Any]) -> dict[str, Any]:
    """Remove process-failure noise from valid below-floor frontier reports."""

    clean = dict(improvement)
    signals = [
        row
        for row in clean.get("signals", [])
        if not (
            isinstance(row, dict)
            and row.get("kind") == "useful_failure_residual_captured"
            and row.get("evidence") == "process_failure"
        )
    ]
    signal_kinds = {row.get("kind") for row in signals if isinstance(row, dict)}
    concept = str(get_path(task, ["evidence", "concept"], "") or task.get("kind") or "unknown")
    clean["signals"] = signals
    clean["signal_kinds"] = sorted(str(item) for item in signal_kinds if item)
    clean["passed"] = bool(signal_kinds & IMPROVEMENT_SIGNAL_KINDS)
    clean["residual_cluster"] = f"progress:{concept}:frontier_yellow_not_graduated"
    clean["frontier_report_nonzero_normalized"] = True
    return clean


def ingest_command_evidence(command_spec: dict[str, Any]) -> list[dict[str, Any]]:
    ingested: list[dict[str, Any]] = []
    for raw_path in command_spec.get("evidence_paths") or []:
        path = resolve(str(raw_path))
        if path.suffix.lower() != ".json" or not path.exists():
            continue
        row = report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, path)
        if row:
            ingested.append(row)
    return ingested






def build_report(
    db_path: Path,
    *,
    tasks: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    results: list[dict[str, Any]],
    command_result: dict[str, Any],
    refresh_result: dict[str, Any],
    high_transfer_sync_result: dict[str, Any],
    retirement_result: dict[str, Any],
    high_transfer_supersede_result: dict[str, Any],
    feedback_guard_result: dict[str, Any],
    teacher_retirement_result: dict[str, Any],
    satisfied_target_result: dict[str, Any],
    stale_hive_queue_result: dict[str, Any],
    resolved_guard_block_result: dict[str, Any],
    node_context: dict[str, Any],
    started: float,
    args: argparse.Namespace,
    paused: bool,
    stop: bool,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for task in tasks:
        status = str(task.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    failed = [row for row in results if row.get("status") in {"failed", "blocked"}]
    trigger = "RED" if any(row.get("status") == "failed" for row in results) else ("YELLOW" if paused or stop or failed else "GREEN")
    return {
        "policy": "project_theseus_hive_work_board_executor_v1",
        "created_utc": now(),
        "trigger_state": trigger,
        "summary": {
            "total_tasks": len(tasks),
            "ready_tasks": sum(1 for row in tasks if str(row.get("status") or "") in READY_STATUSES),
            "selected_tasks": len(selected),
            "executed_tasks": len(results),
            "completed_this_run": sum(1 for row in results if row.get("status") == "done"),
            "failed_this_run": sum(1 for row in results if row.get("status") == "failed"),
            "blocked_this_run": sum(1 for row in results if row.get("status") == "blocked"),
            "paused": paused,
            "stop_requested": stop,
            "execute_requested": bool(args.execute),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "improvement_contract_passed": sum(1 for row in results if get_path(row, ["improvement_contract", "passed"], False)),
            "teacher_escalations_queued": sum(1 for row in results if get_path(row, ["teacher_escalation", "queued"], False)),
            "fresh_targets_satisfied": int(satisfied_target_result.get("satisfied_count") or 0),
            "stale_hive_queue_retired": int(stale_hive_queue_result.get("retired_count") or 0),
            "resolved_guard_blocks_unblocked": int(resolved_guard_block_result.get("unblocked_count") or 0),
            "recent_lane_counts": recent_rotation_lane_counts(),
        },
        "database": rel(db_path),
        "counts": counts,
        "command_result": command_result,
        "refresh_result": compact_result(refresh_result),
        "high_transfer_sync_result": high_transfer_sync_result,
        "retirement_result": retirement_result,
        "high_transfer_supersede_result": high_transfer_supersede_result,
        "feedback_guard_result": feedback_guard_result,
        "teacher_retirement_result": teacher_retirement_result,
        "satisfied_target_result": satisfied_target_result,
        "stale_hive_queue_result": stale_hive_queue_result,
        "resolved_guard_block_result": resolved_guard_block_result,
        "node_assignment": node_context,
        "selected": selected[:20],
        "results": results,
        "rules": {
            "source_of_truth": "SQLite Hive work board owns task status; reports are views",
            "execution": "only mapped board task kinds become bounded subprocesses",
            "retry": "failed work is retried once, then blocked with evidence",
            "public_data": "public benchmark work remains calibration-only",
            "receiver_calibration": "public 4-card calibration requires a fresh private ablation gate and is consumed once per gate",
            "remote_control": "screen/keyboard/mouse control is permissioned separately with TTL and kill switch",
        },
        "external_inference_calls": 0,
    }


def snapshot_improvement_metrics() -> dict[str, Any]:
    broad = read_json(REPORTS / "broad_transfer_matrix.json", {})
    scoreboard = read_json(REPORTS / "learning_scoreboard.json", {})
    residual = read_json(REPORTS / "residual_escrow.json", {})
    return {
        "created_utc": now(),
        "broad_public_pass_rate": first_number(
            get_path(broad, ["summary", "aggregate_pass_rate"], None),
            get_path(broad, ["summary", "real_public_pass_rate"], None),
            get_path(scoreboard, ["broad_transfer_matrix", "real_public_pass_rate"], None),
        ),
        "public_transfer_pass_rate": first_number(get_path(scoreboard, ["public_transfer", "real_public_task_pass_rate"], None)),
        "residual_count": first_number(
            get_path(residual, ["summary", "active_residual_count"], None),
            get_path(residual, ["summary", "residual_count"], None),
        ),
    }


def classify_improvement(
    task: dict[str, Any],
    command_spec: dict[str, Any],
    result: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    signals: list[dict[str, Any]] = []
    evidence_paths = [str(path) for path in command_spec.get("evidence_paths") or []]
    reports = {path: read_json(resolve(path), {}) for path in evidence_paths if str(path).lower().endswith(".json")}
    if improved(after.get("broad_public_pass_rate"), before.get("broad_public_pass_rate")):
        signals.append(signal("public_transfer_improved", before.get("broad_public_pass_rate"), after.get("broad_public_pass_rate"), "broad_transfer_matrix"))
    if lower(after.get("residual_count"), before.get("residual_count")):
        signals.append(signal("private_residual_shrank", before.get("residual_count"), after.get("residual_count"), "residual_escrow"))
    for path, report in reports.items():
        sigs = report_signals(path, report, task)
        signals.extend(sigs)
    if not result.get("ok"):
        signals.append(signal("useful_failure_residual_captured", None, compact_result(result).get("stderr_tail") or result.get("reason"), "process_failure"))
    concept = str(get_path(task, ["evidence", "concept"], "") or "")
    if (
        concept == "private_pressure_four_card_recalibration"
        and not any(str(row.get("kind") or "") == "public_transfer_improved" for row in signals)
        and any(str(row.get("kind") or "") == "private_residual_shrank" for row in signals)
    ):
        signals.append(
            signal(
                "useful_failure_residual_captured",
                {
                    "broad_public_pass_rate": before.get("broad_public_pass_rate"),
                    "public_transfer_pass_rate": before.get("public_transfer_pass_rate"),
                },
                {
                    "reason": "public_receiver_flat_after_private_gate",
                    "broad_public_pass_rate": after.get("broad_public_pass_rate"),
                    "public_transfer_pass_rate": after.get("public_transfer_pass_rate"),
                    "teacher_next": "architecture_experiment_spec_only",
                },
                "broad_transfer_matrix",
            )
        )
    signal_kinds = {row.get("kind") for row in signals}
    passed = bool(signal_kinds & IMPROVEMENT_SIGNAL_KINDS)
    residual_cluster = residual_cluster_for(task, result, signals)
    return {
        "policy": "project_theseus_unattended_improvement_contract_v1",
        "passed": passed,
        "signals": signals,
        "signal_kinds": sorted(str(item) for item in signal_kinds if item),
        "required": sorted(IMPROVEMENT_SIGNAL_KINDS),
        "residual_cluster": residual_cluster,
        "concept": get_path(task, ["evidence", "concept"], ""),
        "evidence_paths": evidence_paths,
        "before": before,
        "after": after,
        "score_semantics": "unattended progress contract; not public promotion evidence",
    }


def no_progress_demote_or_block(
    task: dict[str, Any],
    status: str,
    reason: str,
    improvement: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    if status != "done" or bool(improvement.get("passed")):
        return {"action": "none", "reason": "improvement_signal_present_or_not_done"}
    concept = str(get_path(task, ["evidence", "concept"], "") or task.get("kind") or "unknown")
    source = str(task.get("source") or "")
    lane = task_rotation_lane(task)
    punishable = bool(
        source == "high_transfer_curriculum_scheduler"
        or concept in CODE_HIGH_TRANSFER_CONCEPTS
        or concept in GENERALIST_HIGH_TRANSFER_CONCEPTS
        or lane != "other"
    )
    if not punishable:
        return {"action": "none", "reason": "non_frontier_task"}
    cluster = str(improvement.get("residual_cluster") or f"no_progress:{concept}")
    prior = [
        row
        for row in read_jsonl(NO_PROGRESS_DEMOTION_LEDGER)[-200:]
        if str(row.get("cluster") or "") == cluster and str(row.get("action") or "") in {"demote", "block"}
    ]
    action = "block" if prior else "demote"
    decision = {
        "created_utc": now(),
        "policy": "project_theseus_no_progress_demotion_v1",
        "task_id": task.get("task_id"),
        "concept": concept,
        "lane": lane,
        "source": source,
        "status_before": status,
        "reason_before": reason,
        "cluster": cluster,
        "action": action,
        "reason": "no_improvement_signal_repeated" if action == "block" else "done_no_improvement_demoted",
        "returncode": result.get("returncode"),
        "signal_kinds": improvement.get("signal_kinds"),
        "rule": "frontier work must produce transfer, residual shrinkage, clean evidence, adapter repair, teacher spec, stale retirement, or useful diagnosis",
    }
    append_jsonl(NO_PROGRESS_DEMOTION_LEDGER, decision)
    return decision


def report_signals(path: str, report: dict[str, Any], task: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not report:
        return []
    policy = str(report.get("policy") or "")
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    out: list[dict[str, Any]] = []
    if policy == "project_theseus_code_residual_curriculum_v1":
        rows = int(summary.get("private_row_count") or 0)
        failures = int(summary.get("private_solution_test_failures") or 0)
        if rows > 0 and failures == 0:
            out.append(signal("new_clean_evidence_produced", 0, rows, path))
    elif policy == "project_theseus_code_lm_closure_v1":
        private_rows = int(summary.get("high_transfer_private_train_task_count") or 0)
        pass_delta = safe_float(summary.get("private_pass_rate_delta"))
        sts_delta = safe_float(summary.get("private_sts_repair_pass_rate_delta"))
        regressions = int(summary.get("private_sts_repair_task_level_regressions") or 0)
        if private_rows > 0 and report.get("run_status") == "completed":
            out.append(
                signal(
                    "new_clean_evidence_produced",
                    0,
                    {
                        "high_transfer_private_train_rows": private_rows,
                        "private_trained_pass_rate": summary.get("private_trained_pass_rate"),
                        "private_pass_rate_delta": pass_delta,
                        "private_sts_repair_delta": sts_delta,
                    },
                    path,
                )
            )
        if pass_delta > 0.0:
            out.append(signal("private_residual_shrank", 0, {"private_pass_rate_delta": pass_delta}, path))
        if sts_delta >= 0.0 and regressions == 0:
            out.append(signal("private_residual_shrank", 0, {"private_sts_repair_delta": sts_delta}, path))
    elif policy == "project_theseus_edge_contract_v2_private_verifier_v1":
        private_rows = int(summary.get("private_rows") or 0)
        generation_plan_rows = int(summary.get("generation_plan_rows") or 0)
        failures = int(summary.get("solution_test_failures") or 0)
        if private_rows > 0 and generation_plan_rows == private_rows and failures == 0:
            out.append(
                signal(
                    "new_clean_evidence_produced",
                    0,
                    {
                        "edge_contract_v2_private_rows": private_rows,
                        "generation_plan_rows": generation_plan_rows,
                        "ready_for_public_calibration": report.get("ready_for_public_calibration"),
                    },
                    path,
                )
            )
        if report.get("ready_for_public_calibration"):
            out.append(
                signal(
                    "private_residual_shrank",
                    0,
                    {
                        "private_delta": summary.get("private_delta"),
                        "candidate_verifier_rows": summary.get("candidate_verifier_rows"),
                    },
                    path,
                )
            )
        elif private_rows > 0:
            out.append(
                signal(
                    "useful_failure_residual_captured",
                    "edge_contract_v2_private_gate",
                    {
                        "reason": "private_v2_verifier_not_ready",
                        "trigger_state": report.get("trigger_state"),
                        "closure_trigger_state": summary.get("closure_trigger_state"),
                        "closure_run_status": summary.get("closure_run_status"),
                    },
                    path,
                )
            )
    elif policy == "project_theseus_multi_turn_conversation_benchmark_v1":
        turns = int(summary.get("turn_count") or 0)
        ready = int(summary.get("personality_context_ready_turns") or 0)
        if turns > 0:
            out.append(signal("new_clean_evidence_produced", 0, {"turns": turns, "personality_ready": ready}, path))
        if report.get("passed"):
            out.append(signal("private_residual_shrank", 0, "conversation_lane_passed", path))
    elif policy == "project_theseus_viea_repo_repair_learner_v1":
        traces = int(summary.get("validated_private_trace_count") or 0)
        rows = int(summary.get("code_lm_row_count") or 0)
        if traces > 0 and rows > 0:
            out.append(signal("new_clean_evidence_produced", 0, {"validated_traces": traces, "code_lm_rows": rows}, path))
    elif policy == "project_theseus_type_contract_diagnostic_v1":
        rows = int(summary.get("feedback_rows_written") or 0)
        contracts = int(summary.get("decoder_contract_rows") or 0)
        if rows > 0 and contracts > 0:
            out.append(signal("new_clean_evidence_produced", 0, {"decoder_feedback_rows": rows, "decoder_contract_rows": contracts}, path))
    elif policy == "project_theseus_teacher_architect_experiment_runner_v1":
        executed = int(summary.get("executed_stage_count") or 0)
        selected = int(summary.get("selected_experiments") or 0)
        if executed > 0 or selected > 0:
            out.append(signal("teacher_experiment_spec_produced", 0, {"selected": selected, "executed": executed}, path))
    elif policy == "project_theseus_hive_work_board_executor_v1":
        if int(summary.get("total_tasks") or 0) > 0:
            out.append(signal("new_clean_evidence_produced", 0, {"board_tasks": summary.get("total_tasks")}, path))
    elif policy == "project_theseus_broad_transfer_closure_runner_v1":
        public_tasks = int(summary.get("public_task_count") or 0)
        hard_failures = int(summary.get("hard_step_failure_count") or 0)
        prior_tasks = int(get_path(task or {}, ["evidence", "public_task_count"], 0) or 0)
        if hard_failures == 0 and public_tasks >= 32:
            out.append(signal("new_clean_evidence_produced", prior_tasks, {"public_task_count": public_tasks, "pass_rate": summary.get("public_pass_rate")}, path))
            if public_tasks > prior_tasks:
                out.append(signal("adapter_repaired", prior_tasks, public_tasks, path))
            if float(summary.get("public_pass_rate") or 0.0) <= 0.0:
                out.append(signal("useful_failure_residual_captured", "adapter_thin_slice", "semantic_transfer_zero_pass_on_32_clean_tasks", path))
    elif policy == "project_theseus_board_game_rl_benchmark_v1":
        gates = report.get("gates") if isinstance(report.get("gates"), list) else []
        ratings = report.get("games") if isinstance(report.get("games"), dict) else {}
        if gates and all(bool(gate.get("ok")) for gate in gates if isinstance(gate, dict)):
            out.append(
                signal(
                    "new_clean_evidence_produced",
                    0,
                    {
                        "games": sorted(ratings.keys()),
                        "gate_count": len(gates),
                        "ratings_path": get_path(report, ["outputs", "ratings"], ""),
                        "policy_train_rows": get_path(report, ["outputs", "policy_train_rows"], ""),
                    },
                    path,
                )
            )
    elif policy == "project_theseus_pufferlib4_rl_lane_v1":
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        capsule_count = int(summary.get("capsule_count") or 0)
        native_ready = bool(summary.get("native_backend_ready"))
        if capsule_count > 0:
            out.append(
                signal(
                    "new_clean_evidence_produced" if native_ready else "useful_failure_residual_captured",
                    0,
                    {
                        "capsule_count": capsule_count,
                        "native_backend_ready": native_ready,
                        "atari_enabled": summary.get("atari_enabled"),
                        "improvement_signal": summary.get("improvement_signal"),
                    },
                    path,
                )
            )
    elif policy == "project_theseus_long_horizon_tool_use_benchmark_v1":
        cases = int(summary.get("case_count") or 0)
        pass_rate = safe_float(summary.get("pass_rate"))
        sts_rows = int(summary.get("sts_rows") or 0)
        if cases > 0 and pass_rate >= 0.8 and sts_rows > 0:
            out.append(
                signal(
                    "new_clean_evidence_produced",
                    0,
                    {
                        "tool_use_cases": cases,
                        "pass_rate": pass_rate,
                        "sts_rows": sts_rows,
                        "skills": summary.get("skills"),
                    },
                    path,
                )
            )
        if pass_rate < 1.0:
            out.append(
                signal(
                    "useful_failure_residual_captured",
                    "long_horizon_tool_use",
                    {"pass_rate": pass_rate, "residuals": summary.get("residuals")},
                    path,
                )
            )
    elif policy == "project_theseus_cross_domain_sts_capsules_v1":
        capsules = int(summary.get("capsule_count") or 0)
        sts_rows = int(summary.get("sts_row_count") or 0)
        if capsules > 0 and sts_rows > 0:
            out.append(
                signal(
                    "new_clean_evidence_produced",
                    0,
                    {
                        "cross_domain_capsules": capsules,
                        "sts_rows": sts_rows,
                        "lane_counts": summary.get("lane_counts"),
                        "skill_counts": summary.get("skill_counts"),
                    },
                    path,
                )
            )
    elif policy == "project_theseus_execution_shape_private_ablation_v1":
        eval_count = int(summary.get("private_eval_task_count") or 0)
        candidate_rows = int(summary.get("candidate_rows") or 0)
        public_ready = bool(
            report.get("ready_for_public_calibration")
            or report.get("private_ablation_public_gate_ready")
            or summary.get("ready_for_public_calibration")
            or summary.get("private_ablation_public_gate_ready")
        )
        if eval_count > 0 and candidate_rows > 0:
            out.append(
                signal(
                    "new_clean_evidence_produced",
                    0,
                    {
                        "private_eval_task_count": eval_count,
                        "candidate_rows": candidate_rows,
                        "edge_exec_repair_v1_pass_rate": summary.get("edge_exec_repair_v1_pass_rate"),
                        "execution_shape_skeleton_pass_rate": summary.get("execution_shape_skeleton_pass_rate"),
                    },
                    path,
                )
            )
        if public_ready:
            out.append(
                signal(
                    "private_residual_shrank",
                    0,
                    {
                        "private_ablation_public_gate_ready": public_ready,
                        "execution_shape_skeleton_pass_rate": summary.get("execution_shape_skeleton_pass_rate"),
                    },
                    path,
                )
            )
        elif eval_count > 0 and candidate_rows > 0:
            out.append(
                signal(
                    "useful_failure_residual_captured",
                    "execution_shape_private_ablation",
                    {
                        "dominant_residual": summary.get("dominant_residual"),
                        "skeleton_no_admissible_candidate_count": summary.get("skeleton_no_admissible_candidate_count"),
                        "skeleton_zero_pass_categories": summary.get("skeleton_zero_pass_categories"),
                        "next_action": "patch_execution_shape_skeleton_decoder_before_rerun",
                    },
                    path,
                )
            )
    elif report.get("trigger_state") == "GREEN":
        out.append(signal("new_clean_evidence_produced", 0, report.get("trigger_state"), path))
    return out


def signal(kind: str, before: Any, after: Any, evidence: Any) -> dict[str, Any]:
    return {"kind": kind, "before": before, "after": after, "evidence": evidence}


def residual_cluster_for(task: dict[str, Any], result: dict[str, Any], signals: list[dict[str, Any]]) -> str:
    concept = str(get_path(task, ["evidence", "concept"], "") or task.get("kind") or "unknown")
    if concept == "private_pressure_four_card_recalibration":
        flat_after_private_gate = any(
            str(row.get("kind") or "") == "useful_failure_residual_captured"
            and str(get_path(row, ["after", "reason"], "") or "") == "public_receiver_flat_after_private_gate"
            for row in signals
        )
        if flat_after_private_gate:
            return f"no_public_transfer_after_private_gate:{concept}"
    if result.get("ok") and not signals:
        return f"no_progress:{concept}"
    if not result.get("ok"):
        stderr = str(result.get("stderr_tail") or "").lower()
        if "timeout" in stderr or result.get("returncode") == 124:
            return f"timeout:{concept}"
        if "no such file" in stderr or "not found" in stderr:
            return f"adapter_missing:{concept}"
        return f"command_failed:{concept}"
    return f"progress:{concept}"


def maybe_enqueue_teacher_escalation(
    db_path: Path,
    task: dict[str, Any],
    improvement: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    cluster = str(improvement.get("residual_cluster") or reason or "")
    if not cluster or cluster.startswith("progress:"):
        return {"queued": False, "reason": "no_escalation_needed", "cluster": cluster}
    prior = [
        row
        for row in read_jsonl(IMPROVEMENT_LEDGER)
        if str(get_path(row, ["improvement_contract", "residual_cluster"], "")) == cluster
    ]
    occurrences = len(prior) + 1
    required_occurrences = 1 if cluster.startswith("no_public_transfer_after_private_gate:") else 2
    if occurrences < required_occurrences:
        return {
            "queued": False,
            "reason": "waiting_for_second_occurrence",
            "cluster": cluster,
            "occurrences": occurrences,
            "required_occurrences": required_occurrences,
        }
    task_id = stable_id("teacher_escalation", cluster)
    teacher_task = {
        "task_id": task_id,
        "title": f"Teacher architecture diagnosis for {cluster}",
        "source": "auto_teacher_escalation",
        "kind": "teacher_architecture_escalation",
        "status": "ready",
        "priority": "high",
        "assignee": "teacher_architect_experiment_runner",
        "node_id": "local",
        "command": "teacher_architect_experiment_runner proposal only",
        "evidence_json": json.dumps(
            {
                "residual_cluster": cluster,
                "occurrences": occurrences,
                "source_task_id": task.get("task_id"),
                "source_kind": task.get("kind"),
                "source_status": status,
                "source_reason": reason,
                "loop": "residual cluster -> diagnosis -> experiment spec -> private eval -> public calibration",
                "measured_wall": "private gates improved but public receiver calibration stayed flat"
                if cluster.startswith("no_public_transfer_after_private_gate:")
                else "",
                "teacher_policy": "architecture guidance only; no public answers; no distillation",
            },
            sort_keys=True,
        ),
        "retry_count": 0,
        "blocked_reason": "",
    }
    upsert_task(db_path, teacher_task)
    add_event(db_path, task_id, "teacher_escalation_queued", {"cluster": cluster, "occurrences": occurrences})
    append_jsonl(TEACHER_ESCALATION_LEDGER, {"created_utc": now(), "policy": "project_theseus_teacher_auto_escalation_v1", "task_id": task_id, "cluster": cluster, "occurrences": occurrences})
    return {"queued": True, "task_id": task_id, "cluster": cluster, "occurrences": occurrences}


def improvement_ledger_row(task: dict[str, Any], status: str, reason: str, improvement: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_utc": now(),
        "policy": "project_theseus_unattended_improvement_event_v1",
        "task_id": task.get("task_id"),
        "kind": task.get("kind"),
        "source": task.get("source"),
        "concept": get_path(task, ["evidence", "concept"], ""),
        "status": status,
        "reason": reason,
        "improvement_contract": improvement,
        "review_step_count": improvement_review_step_count(improvement),
        "review_step_basis": "improvement_contract_fields",
        "maintenance_mode": maintenance_mode_from_values(task, improvement),
        "maintenance_mode_basis": "task_or_improvement_or_object_only_default",
        "human_edit_minutes": None,
        "human_edit_minutes_measured": False,
        "external_inference_calls": 0,
    }


def load_node_context() -> dict[str, Any]:
    policy = read_json(ROOT / "configs" / "hive_policy.json", {})
    registry = hive_node_registry.build_registry(policy)
    write_json(REPORTS / "hive_node_registry.json", registry)
    status = read_json(REPORTS / "hive_status.json", {})
    registry_nodes = registry.get("nodes") if isinstance(registry.get("nodes"), list) else []
    registry_local = next((node for node in registry_nodes if isinstance(node, dict) and node.get("is_local")), {})
    scheduler = read_json(REPORTS / "hive_scheduler.json", {})
    version = read_json(REPORTS / "hive_version_status.json", {})
    convergence = read_json(REPORTS / "hive_version_convergence.json", {})
    drift = []
    if version and version.get("trigger_state") == "RED":
        drift.append("local_version_red")
    if convergence and convergence.get("trigger_state") == "RED":
        drift.append("fleet_version_convergence_red")
    return {
        "policy": "project_theseus_hive_node_assignment_v1",
        "local_node": {
            "node_id": registry_local.get("node_id") or status.get("node_id") or "local",
            "node_name": registry_local.get("node_name") or status.get("node_name") or "local",
            "capabilities": registry_local.get("capabilities") or status.get("capabilities") or [],
            "api_url": registry_local.get("api_url") or status.get("api_url"),
        },
        "scheduler_summary": registry.get("summary") if isinstance(registry.get("summary"), dict) else scheduler.get("summary") if isinstance(scheduler.get("summary"), dict) else {},
        "node_registry": {
            "report": "reports/hive_node_registry.json",
            "summary": registry.get("summary", {}),
            "nodes": [
                {
                    "node_id": node.get("node_id"),
                    "node_name": node.get("node_name"),
                    "training_allowed": node.get("training_allowed"),
                    "light_task_allowed": node.get("light_task_allowed"),
                    "training_blockers": node.get("training_blockers"),
                }
                for node in registry.get("nodes", [])
                if isinstance(node, dict)
            ],
        },
        "version_status": {
            "local": version.get("trigger_state"),
            "convergence": convergence.get("trigger_state"),
            "drift": drift,
        },
        "version_drift_blocking": bool(drift),
    }


def update_task(
    db_path: Path,
    task_id: str,
    *,
    status: str | None = None,
    assignee: str | None = None,
    node_id: str | None = None,
    retry_count: int | None = None,
    blocked_reason: str | None = None,
) -> None:
    fields = []
    values: list[Any] = []
    if status is not None:
        fields.append("status=?")
        values.append(status)
    if assignee is not None:
        fields.append("assignee=?")
        values.append(assignee)
    if node_id is not None:
        fields.append("node_id=?")
        values.append(node_id)
    if retry_count is not None:
        fields.append("retry_count=?")
        values.append(int(retry_count))
    if blocked_reason is not None:
        fields.append("blocked_reason=?")
        values.append(blocked_reason)
    fields.append("updated_utc=?")
    values.append(now())
    values.append(task_id)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE task_id=?", values)


def upsert_task(db_path: Path, task: dict[str, Any]) -> None:
    stamp = now()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO tasks (
                task_id, title, source, kind, status, priority, assignee, node_id,
                command, evidence_json, created_utc, updated_utc, retry_count, blocked_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                title=excluded.title,
                source=excluded.source,
                kind=excluded.kind,
                status=CASE
                    WHEN tasks.status IN ('done', 'blocked') AND excluded.status IN ('ready', 'queued') THEN tasks.status
                    ELSE excluded.status
                END,
                priority=excluded.priority,
                assignee=excluded.assignee,
                node_id=excluded.node_id,
                command=excluded.command,
                evidence_json=excluded.evidence_json,
                updated_utc=excluded.updated_utc,
                retry_count=CASE
                    WHEN tasks.status IN ('done', 'blocked') AND excluded.status IN ('ready', 'queued') THEN tasks.retry_count
                    ELSE excluded.retry_count
                END,
                blocked_reason=CASE
                    WHEN tasks.status IN ('done', 'blocked') AND excluded.status IN ('ready', 'queued') THEN tasks.blocked_reason
                    ELSE excluded.blocked_reason
                END
            """,
            (
                task["task_id"],
                task["title"],
                task["source"],
                task["kind"],
                task["status"],
                task["priority"],
                task["assignee"],
                task["node_id"],
                task["command"],
                task["evidence_json"],
                stamp,
                stamp,
                int(task.get("retry_count") or 0),
                task.get("blocked_reason") or "",
            ),
        )


def add_event(db_path: Path, task_id: str, event_type: str, content: dict[str, Any]) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO events (event_id, task_id, event_type, content_json, created_utc) VALUES (?, ?, ?, ?, ?)",
            (stable_id("event", task_id, event_type, json.dumps(content, sort_keys=True)), task_id, event_type, json.dumps(content, sort_keys=True), now()),
        )


def add_evidence(db_path: Path, task_id: str, label: str, path: str, claim_role: str) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO evidence (evidence_id, task_id, label, path, claim_role, created_utc) VALUES (?, ?, ?, ?, ?, ?)",
            (stable_id("evidence", task_id, label, path), task_id, label, path, claim_role, now()),
        )


def unblock_resolved_guard_blocks(db_path: Path) -> dict[str, Any]:
    guard = execution_shape_no_template_smoke_guard()
    if not guard.get("allowed"):
        return {
            "unblocked_count": 0,
            "reason": "guard_still_blocked",
            "guard_reason": guard.get("reason"),
        }
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT task_id, title, blocked_reason
            FROM tasks
            WHERE status='blocked'
              AND title='Train transferable concept: private_pressure_private_closure'
              AND blocked_reason IN ('', 'private_pressure_closure_blocked_by_no_template_execution_shape_gate')
            """
        ).fetchall()
    unblocked: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row["task_id"])
        update_task(db_path, task_id, status="ready", retry_count=0, blocked_reason="")
        detail = {
            "task_id": task_id,
            "title": row["title"],
            "previous_blocked_reason": row["blocked_reason"],
            "guard_reason": guard.get("reason"),
            "targeted_zero_category_closure": guard.get("targeted_zero_category_closure"),
        }
        add_event(db_path, task_id, "resolved_guard_block_unblocked", detail)
        unblocked.append(detail)
    return {
        "unblocked_count": len(unblocked),
        "reason": "resolved_no_template_execution_shape_gate",
        "guard_reason": guard.get("reason"),
        "unblocked": unblocked,
    }


def run_subprocess(command: list[str], *, timeout: int, env_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    env = None
    if env_overrides:
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=max(1, int(timeout)), env=env)
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "command": command,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": 124,
            "command": command,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": exc.stdout[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": exc.stderr[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout",
        }


def record_hook(phase: str, target: str, task: dict[str, Any], command: list[str], payload: dict[str, Any]) -> None:
    guards = hook_guards(target, phase)
    append_jsonl(
        HOOK_LEDGER,
        {
            "created_utc": now(),
            "policy": "project_theseus_hive_tool_hook_event_v1",
            "phase": phase,
            "target": target,
            "task_id": task.get("task_id"),
            "kind": task.get("kind"),
            "title": task.get("title"),
            "command_hash": hashlib.sha256(json.dumps(command, sort_keys=True).encode("utf-8")).hexdigest()[:16],
            "guards": guards,
            "review_step_count": len(guards),
            "review_step_basis": "hook_guard_count",
            "maintenance_mode": maintenance_mode_from_values(task, payload),
            "maintenance_mode_basis": "task_or_payload_or_object_only_default",
            "human_edit_minutes": None,
            "human_edit_minutes_measured": False,
            "payload": {
                "status": payload.get("status"),
                "reason": payload.get("reason"),
                "returncode": payload.get("returncode"),
                "timeout_seconds": payload.get("timeout_seconds"),
            },
            "external_inference_calls": 0,
        },
    )


def hook_guards(target: str, phase: str) -> list[str]:
    if target == "background_task":
        return ["permission_envelope", "step_budget", "delivery_contract"] if phase == "before" else ["status_report", "notify_channel", "feedback_route"]
    if target == "viea_action_executor":
        return ["board_assignment", "resume_state", "public_data_guard"] if phase == "before" else ["ledger_update", "evidence_link", "residual_route"]
    if target == "training_launch":
        return ["resource_governor", "concept_transfer_guard"] if phase == "before" else ["donor_receiver_report", "residual_route"]
    if target == "conversation_training":
        return ["personality_core", "session_isolation", "turn_budget"] if phase == "before" else ["conversation_score", "correction_memory_route"]
    if target == "repo_repair_training":
        return ["private_repo_tasks_only", "hidden_private_tests"] if phase == "before" else ["repair_trace", "code_lm_row_route"]
    if target == "long_horizon_tool_use":
        return ["resume_state", "checkpoint_visibility"] if phase == "before" else ["run_ledger", "tool_rot_signal"]
    if target == "teacher_call":
        return ["teacher_budget", "architecture_only"] if phase == "before" else ["experiment_spec_capture", "no_answer_distillation_check"]
    return ["permission_envelope"] if phase == "before" else ["ledger_update"]


def feedback_row(task: dict[str, Any], status: str, reason: str, result: dict[str, Any], improvement: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_utc": now(),
        "policy": "project_theseus_hive_work_board_feedback_v1",
        "task_id": task.get("task_id"),
        "kind": task.get("kind"),
        "source": task.get("source"),
        "status": status,
        "reason": reason,
        "artifact_updates": ["hive_work_board.sqlite", "hive_work_board_executor.json"],
        "improvement_contract": improvement,
        "residual": None if improvement.get("passed") else {"cluster": improvement.get("residual_cluster") or reason, "stderr_tail": result.get("stderr_tail", "")[-1000:]},
        "training_eligible": bool(improvement.get("passed")) and task.get("source") == "high_transfer_curriculum_scheduler",
        "claim_role": "diagnostic",
    }


def ledger_row(task: dict[str, Any], status: str, reason: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    return {
        "created_utc": now(),
        "policy": "project_theseus_hive_work_board_execution_event_v1",
        "task_id": task.get("task_id"),
        "title": task.get("title"),
        "source": task.get("source"),
        "kind": task.get("kind"),
        "status": status,
        "reason": reason,
        "payload": payload,
        "runtime_ms": result.get("runtime_ms"),
        "review_step_count": work_board_review_step_count(payload),
        "review_step_basis": "assignment_command_result_postprocess_improvement_no_progress_evidence",
        "maintenance_mode": maintenance_mode_from_values(task, payload),
        "maintenance_mode_basis": "task_or_payload_or_object_only_default",
        "human_edit_minutes": None,
        "human_edit_minutes_measured": False,
        "external_inference_calls": 0,
    }


def work_board_review_step_count(payload: dict[str, Any]) -> int:
    steps = 0
    for key in ["assignment", "command", "result", "improvement_contract", "no_progress_decision"]:
        if payload.get(key):
            steps += 1
    postprocess = payload.get("postprocess_results")
    if isinstance(postprocess, list):
        steps += len(postprocess)
    evidence = payload.get("report_evidence_store")
    if isinstance(evidence, list):
        steps += len(evidence)
    return max(1, steps)


def improvement_review_step_count(improvement: dict[str, Any]) -> int:
    steps = 1 if improvement else 0
    for key in ["passed", "residual_cluster", "score", "fresh_target_satisfied"]:
        if key in improvement:
            steps += 1
    gates = improvement.get("gates")
    if isinstance(gates, list):
        steps += len(gates)
    return max(1, steps)


def maintenance_mode_from_values(*values: Any) -> str:
    for value in values:
        explicit = explicit_maintenance_mode(value)
        if explicit:
            return explicit
    return "object_only"


def explicit_maintenance_mode(value: Any) -> str:
    if isinstance(value, dict):
        for key in ["maintenance_mode", "maintenance_policy", "maintenance_label"]:
            normalized = normalize_maintenance_mode(value.get(key))
            if normalized:
                return normalized
        for key in ["payload", "orchestration", "improvement_contract"]:
            nested = value.get(key)
            if isinstance(nested, dict):
                normalized = explicit_maintenance_mode(nested)
                if normalized:
                    return normalized
        text = " ".join(str(value.get(key) or "") for key in ["task_id", "title", "source", "kind", "reason"])
        if "circle" in text.lower() and "seed" in text.lower() and "rebuild" in text.lower():
            return "circle_seed_rule_rebuild"
    return normalize_maintenance_mode(value)


def normalize_maintenance_mode(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ordinary": "object_only",
        "ordinary_current": "object_only",
        "baseline": "object_only",
        "object": "object_only",
        "object_only": "object_only",
        "circle": "circle_seed_rule_rebuild",
        "circle_seed_rule": "circle_seed_rule_rebuild",
        "circle_seed_rule_rebuild": "circle_seed_rule_rebuild",
        "seed_rule_rebuild": "circle_seed_rule_rebuild",
    }
    return aliases.get(text, "")


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result.get("ok"),
        "returncode": result.get("returncode"),
        "runtime_ms": result.get("runtime_ms"),
        "stdout_tail": str(result.get("stdout_tail") or "")[-1000:],
        "stderr_tail": str(result.get("stderr_tail") or "")[-1000:],
        "error": result.get("error"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Hive Work Board Executor",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- total_tasks: `{summary.get('total_tasks')}`",
        f"- ready_tasks: `{summary.get('ready_tasks')}`",
        f"- selected_tasks: `{summary.get('selected_tasks')}`",
        f"- executed_tasks: `{summary.get('executed_tasks')}`",
        f"- completed_this_run: `{summary.get('completed_this_run')}`",
        f"- failed_this_run: `{summary.get('failed_this_run')}`",
        f"- fresh_targets_satisfied: `{summary.get('fresh_targets_satisfied')}`",
        f"- stale_hive_queue_retired: `{summary.get('stale_hive_queue_retired')}`",
        f"- recent_lane_counts: `{json.dumps(summary.get('recent_lane_counts') or {}, sort_keys=True)}`",
        "",
        "## Results",
        "",
    ]
    for row in report.get("results", []):
        lines.append(f"- `{row.get('status')}` {row.get('title')} reason={row.get('reason')} runtime_ms={row.get('runtime_ms')}")
    if not report.get("results"):
        lines.append("- none")
    satisfied = get_path(report, ["satisfied_target_result", "satisfied"], []) or []
    lines.extend(["", "## Fresh Targets Satisfied", ""])
    for row in satisfied[:10]:
        lines.append(f"- `{row.get('concept')}` task={row.get('task_id')} report={row.get('target_report')} reason={row.get('reason')}")
    if not satisfied:
        lines.append("- none")
    lines.extend(["", "## Selected", ""])
    for row in report.get("selected", [])[:10]:
        lines.append(f"- `{row.get('status')}` {row.get('title')} task={row.get('task_id')} assignment={get_path(row, ['assignment', 'reason'], '')}")
    lines.append("")
    return "\n".join(lines)


def stable_id(*parts: Any) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"hive_task_{digest}"


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def parse_json(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = []
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
    except OSError:
        return []


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


def first_number(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def improved(after: Any, before: Any) -> bool:
    if after is None or before is None:
        return False
    try:
        return float(after) > float(before)
    except (TypeError, ValueError):
        return False


def lower(after: Any, before: Any) -> bool:
    if after is None or before is None:
        return False
    try:
        return float(after) < float(before)
    except (TypeError, ValueError):
        return False


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "task_rotation_lane",
    "recent_rotation_lane_counts",
    "rotation_lane_for_concept",
    "lane_rotation_penalty",
    "lane_rotation_rank",
    "is_conversation_task",
    "is_generalist_high_transfer_task",
    "is_code_transfer_task",
    "run_postprocess_commands",
    "useful_frontier_evidence_returned",
    "normalize_frontier_evidence_improvement",
    "ingest_command_evidence",
    "build_report",
    "snapshot_improvement_metrics",
    "classify_improvement",
    "no_progress_demote_or_block",
    "report_signals",
    "signal",
    "residual_cluster_for",
    "maybe_enqueue_teacher_escalation",
    "improvement_ledger_row",
    "load_node_context",
    "update_task",
    "upsert_task",
    "add_event",
    "add_evidence",
    "unblock_resolved_guard_blocks",
    "run_subprocess",
    "record_hook",
    "hook_guards",
    "feedback_row",
    "ledger_row",
    "compact_result",
    "render_markdown",
    "stable_id",
    "get_path",
    "parse_json",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_text",
    "append_jsonl",
    "first_number",
    "safe_float",
    "improved",
    "lower",
    "resolve",
    "rel",
    "now",
]
