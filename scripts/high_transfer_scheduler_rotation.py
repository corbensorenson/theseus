"""Rotation, report selection, and task rendering helpers for the scheduler."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import report_evidence_store
from high_transfer_scheduler_common import *  # noqa: F401,F403


def concept_transfer_pressure_index(transfer: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rows = transfer.get("concept_transfer_pressure") if isinstance(transfer.get("concept_transfer_pressure"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        concept = str(row.get("concept") or "")
        if concept:
            out[concept] = row
    return out


def generalist_rotation_state(
    *,
    broad: dict[str, Any],
    conversation_hard: dict[str, Any],
    conversation_hard_v2: dict[str, Any],
    conversation_hard_v3: dict[str, Any],
    conversation_hard_v4: dict[str, Any],
    board_game: dict[str, Any],
    pufferlib4_rl: dict[str, Any],
    long_horizon: dict[str, Any],
    repo: dict[str, Any],
    cross_domain_capsules: dict[str, Any],
) -> dict[str, Any]:
    """Prioritize broad learning lanes after a flat code calibration.

    Code remains important, but once a clean private-pressure receiver
    calibration stays flat, the unattended board should gather transferable
    pressure from conversation, games, long-horizon tool use, and repo repair
    before blindly scheduling another same-surface code pass.
    """

    now_ts = time.time()
    code_flat = latest_code_receiver_calibration_flat()
    summary = broad.get("summary") if isinstance(broad.get("summary"), dict) else {}
    pass_rate = safe_float(summary.get("real_public_pass_rate") or summary.get("aggregate_pass_rate"))
    floor = safe_float(summary.get("floor") or 0.70) or 0.70
    cards_below = summary.get("cards_below_floor") if isinstance(summary.get("cards_below_floor"), list) else []
    active = bool((code_flat.get("flat") or pass_rate < floor) and pass_rate < floor and cards_below)
    source_mtimes = {
        "multi_turn_conversation_hard": file_mtime(REPORTS / "high_transfer_multi_turn_conversation_hard.json"),
        "multi_turn_conversation_hard_v2": file_mtime(REPORTS / "high_transfer_multi_turn_conversation_hard_v2.json"),
        "multi_turn_conversation_hard_v3": file_mtime(REPORTS / "high_transfer_multi_turn_conversation_hard_v3.json"),
        "multi_turn_conversation_hard_v4": file_mtime(REPORTS / "high_transfer_multi_turn_conversation_hard_v4.json"),
        "board_game_rl": file_mtime(REPORTS / "board_game_rl_benchmark.json"),
        "pufferlib4_rl": file_mtime(REPORTS / "pufferlib4_rl_lane.json"),
        "long_horizon_tool_use": file_mtime(REPORTS / "high_transfer_long_horizon_tool_use.json"),
        "repo_repair": file_mtime(REPORTS / "high_transfer_repo_repair_learner.json") or file_mtime(REPORTS / "viea_repo_repair_learner.json"),
        "cross_domain_sts_capsules": file_mtime(REPORTS / "cross_domain_sts_capsules.json"),
    }
    due_reasons: dict[str, str] = {}

    hard_state = conversation_lifecycle(conversation_hard)
    hard_v2_state = conversation_lifecycle(conversation_hard_v2)
    hard_v3_state = conversation_lifecycle(conversation_hard_v3)
    hard_v4_state = conversation_lifecycle(conversation_hard_v4)
    if not source_mtimes["multi_turn_conversation_hard"]:
        due_reasons["multi_turn_conversation_hard"] = "missing_hard_conversation_report"
    elif active and hard_state["graduated"] and not source_mtimes["multi_turn_conversation_hard_v2"]:
        due_reasons["multi_turn_conversation_hard_v2"] = "hard_v1_saturated_missing_hard_v2_frontier"
    elif active and hard_v2_state["graduated"] and not source_mtimes["multi_turn_conversation_hard_v3"]:
        due_reasons["multi_turn_conversation_hard_v3"] = "hard_v2_saturated_missing_hard_v3_product_frontier"
    elif active and hard_v2_state["graduated"] and not hard_v3_state["graduated"]:
        due_reasons["multi_turn_conversation_hard_v3"] = "hard_v2_saturated_hard_v3_product_frontier_not_graduated"
    elif active and hard_v3_state["graduated"] and not source_mtimes["multi_turn_conversation_hard_v4"]:
        due_reasons["multi_turn_conversation_hard_v4"] = "hard_v3_saturated_missing_hard_v4_a_plus_frontier"
    elif active and hard_v3_state["graduated"] and not hard_v4_state["graduated"]:
        due_reasons["multi_turn_conversation_hard_v4"] = "hard_v3_saturated_hard_v4_a_plus_frontier_not_graduated"
    elif active and hard_v4_state["graduated"] and now_ts - source_mtimes["multi_turn_conversation_hard_v4"] >= GENERALIST_ROTATION_REFRESH_SECONDS:
        due_reasons["multi_turn_conversation_hard_v4"] = "hard_v4_refresh_due_after_flat_code_calibration"
    elif active and hard_v3_state["graduated"] and now_ts - source_mtimes["multi_turn_conversation_hard_v3"] >= GENERALIST_ROTATION_REFRESH_SECONDS:
        due_reasons["multi_turn_conversation_hard_v3"] = "hard_v3_refresh_due_after_flat_code_calibration"
    elif active and hard_state["graduated"] and not hard_v2_state["graduated"]:
        due_reasons["multi_turn_conversation_hard_v2"] = "hard_v1_saturated_hard_v2_frontier_not_graduated"
    elif active and hard_v2_state["graduated"] and now_ts - source_mtimes["multi_turn_conversation_hard_v2"] >= GENERALIST_ROTATION_REFRESH_SECONDS:
        due_reasons["multi_turn_conversation_hard_v2"] = "hard_v2_refresh_due_after_flat_code_calibration"
    elif active and not hard_state["graduated"] and now_ts - source_mtimes["multi_turn_conversation_hard"] >= GENERALIST_ROTATION_REFRESH_SECONDS:
        due_reasons["multi_turn_conversation_hard"] = "hard_conversation_refresh_due_after_flat_code_calibration"
    elif active and not hard_state["graduated"]:
        due_reasons["multi_turn_conversation_hard"] = "hard_conversation_frontier_not_graduated"

    if not source_mtimes["board_game_rl"]:
        due_reasons["board_game_rl"] = "missing_board_game_rl_report"
    elif active and now_ts - source_mtimes["board_game_rl"] >= GENERALIST_ROTATION_REFRESH_SECONDS:
        due_reasons["board_game_rl"] = "board_game_rl_refresh_due_after_flat_code_calibration"

    if not source_mtimes["pufferlib4_rl"]:
        due_reasons["pufferlib4_rl"] = "missing_pufferlib4_rl_lane_report"
    elif active and now_ts - source_mtimes["pufferlib4_rl"] >= GENERALIST_ROTATION_REFRESH_SECONDS:
        due_reasons["pufferlib4_rl"] = "pufferlib4_rl_refresh_due_after_flat_code_calibration"

    if not source_mtimes["long_horizon_tool_use"]:
        due_reasons["long_horizon_tool_use"] = "missing_long_horizon_tool_use_report"
    elif active and now_ts - source_mtimes["long_horizon_tool_use"] >= GENERALIST_ROTATION_REFRESH_SECONDS:
        due_reasons["long_horizon_tool_use"] = "long_horizon_tool_use_refresh_due_after_flat_code_calibration"

    repo_rows = int(get_path(repo, ["summary", "validated_private_trace_count"], get_path(repo, ["summary", "task_count"], 0)) or 0)
    if repo_rows <= 0:
        due_reasons["repo_repair"] = "missing_repo_repair_private_trace_evidence"
    elif active and now_ts - source_mtimes["repo_repair"] >= REPO_REPAIR_REFRESH_SECONDS:
        due_reasons["repo_repair"] = "repo_repair_refresh_due_after_flat_code_calibration"

    capsule_mtime = source_mtimes["cross_domain_sts_capsules"]
    latest_non_code_mtime = max(
        source_mtimes["multi_turn_conversation_hard"],
        source_mtimes["multi_turn_conversation_hard_v2"],
        source_mtimes["multi_turn_conversation_hard_v3"],
        source_mtimes["multi_turn_conversation_hard_v4"],
        source_mtimes["board_game_rl"],
        source_mtimes["pufferlib4_rl"],
        source_mtimes["long_horizon_tool_use"],
        source_mtimes["repo_repair"],
    )
    capsule_count = int(get_path(cross_domain_capsules, ["summary", "capsule_count"], 0) or 0)
    if capsule_count <= 0:
        due_reasons["cross_domain_sts_capsules"] = "missing_cross_domain_sts_capsules"
    elif latest_non_code_mtime and capsule_mtime < latest_non_code_mtime:
        due_reasons["cross_domain_sts_capsules"] = "non_code_source_newer_than_capsule_report"
    elif active and capsule_mtime and now_ts - capsule_mtime >= CROSS_DOMAIN_CAPSULE_REFRESH_SECONDS:
        due_reasons["cross_domain_sts_capsules"] = "cross_domain_capsule_refresh_due"

    order = [
        "multi_turn_conversation_hard_v4",
        "multi_turn_conversation_hard_v3",
        "multi_turn_conversation_hard_v2",
        "multi_turn_conversation_hard",
        "board_game_rl",
        "pufferlib4_rl",
        "long_horizon_tool_use",
        "repo_repair",
        "cross_domain_sts_capsules",
    ]
    due = [concept for concept in order if concept in due_reasons]
    payload = {
        "active": active,
        "flat_code_calibration": code_flat,
        "broad_pass_rate": pass_rate,
        "floor": floor,
        "cards_below_floor": cards_below,
        "due_reasons": due_reasons,
        "source_mtimes": source_mtimes,
        "latest_non_code_mtime": latest_non_code_mtime or None,
        "capsule_count": capsule_count,
    }
    return {
        **payload,
        "order": order,
        "due_concepts": due,
        "first_due_concept": due[0] if due else None,
        "rotation_epoch": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12],
        "policy": "below_floor_or_flat_code_receiver_opens_generalist_rotation_v2",
    }


def latest_code_receiver_calibration_flat() -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    for row in read_jsonl(REPORTS / "hive_unattended_improvement_ledger.jsonl"):
        if not isinstance(row, dict) or row.get("status") != "done":
            continue
        contract = row.get("improvement_contract") if isinstance(row.get("improvement_contract"), dict) else {}
        concept = str(row.get("concept") or contract.get("concept") or "")
        if concept != "private_pressure_four_card_recalibration":
            continue
        latest = row
    if not latest:
        return {"flat": False, "reason": "no_private_pressure_four_card_recalibration_seen"}
    contract = latest.get("improvement_contract") if isinstance(latest.get("improvement_contract"), dict) else {}
    before = contract.get("before") if isinstance(contract.get("before"), dict) else {}
    after = contract.get("after") if isinstance(contract.get("after"), dict) else {}
    before_broad = safe_float(before.get("broad_public_pass_rate"))
    after_broad = safe_float(after.get("broad_public_pass_rate"))
    before_public = safe_float(before.get("public_transfer_pass_rate"))
    after_public = safe_float(after.get("public_transfer_pass_rate"))
    improved = after_broad > before_broad or after_public > before_public
    return {
        "flat": not improved,
        "reason": "latest_receiver_calibration_flat" if not improved else "latest_receiver_calibration_improved",
        "task_id": latest.get("task_id"),
        "created_utc": latest.get("created_utc"),
        "before_broad_public_pass_rate": before_broad,
        "after_broad_public_pass_rate": after_broad,
        "before_public_transfer_pass_rate": before_public,
        "after_public_transfer_pass_rate": after_public,
    }


def report_age_seconds(path: Path) -> int | None:
    mtime = file_mtime(path)
    if not mtime:
        return None
    return int(max(0.0, time.time() - mtime))


def apply_dynamic_code_priority(
    concept: dict[str, Any],
    *,
    transfer_pressure_by_concept: dict[str, dict[str, Any]],
    max_shared_pressure: int,
    no_lift_cooldown_concepts: set[str],
) -> None:
    """Let the residual audit pick the next code concept instead of a fixed favorite."""

    if concept.get("concept") not in {
        "type_and_return_shape",
        "typed_interface_skeleton",
        "edge_conditions",
        "admissibility_and_interface",
        "algorithmic_planning",
    }:
        return
    concept_name = str(concept.get("concept") or "")
    row = transfer_pressure_by_concept.get(concept_name) or {}
    residual_count = int(row.get("residual_count") or 0)
    card_count = int(row.get("card_count") or 0)
    if concept_name == "typed_interface_skeleton":
        type_row = transfer_pressure_by_concept.get("type_and_return_shape") or {}
        interface_row = transfer_pressure_by_concept.get("admissibility_and_interface") or {}
        residual_count = (
            int(type_row.get("residual_count") or 0)
            + int(interface_row.get("residual_count") or 0)
        )
        card_count = max(
            int(type_row.get("card_count") or 0),
            int(interface_row.get("card_count") or 0),
        )
        concept["dynamic_priority_evidence"] = {
            "type_and_return_shape_residual_count": int(type_row.get("residual_count") or 0),
            "admissibility_and_interface_residual_count": int(interface_row.get("residual_count") or 0),
            "combined_residual_count": residual_count,
            "combined_card_count": card_count,
        }
    if concept.get("status") == "waiting_recalibration":
        concept["priority"] = "critical"
        concept["dynamic_priority_reason"] = "fresh_private_pressure_waiting_for_receiver_calibration"
    elif str(concept.get("concept") or "") in no_lift_cooldown_concepts:
        concept["priority"] = "medium"
        concept["dynamic_priority_reason"] = "no_lift_cooldown_after_fresh_private_recalibration"
    elif residual_count <= 0:
        concept["priority"] = "medium"
        concept["dynamic_priority_reason"] = "no_current_transfer_residual_pressure"
    elif card_count >= 2 and residual_count >= max_shared_pressure:
        concept["priority"] = "critical"
        concept["dynamic_priority_reason"] = "largest_shared_cross_card_residual_pressure"
    elif card_count >= 2:
        concept["priority"] = "high"
        concept["dynamic_priority_reason"] = "shared_cross_card_residual_pressure"
    elif residual_count >= 12:
        concept["priority"] = "high"
        concept["dynamic_priority_reason"] = "large_single_receiver_residual_pressure"
    else:
        concept["priority"] = "medium"
        concept["dynamic_priority_reason"] = "low_or_single_receiver_residual_pressure"


def no_lift_cooldown_concept_set() -> set[str]:
    """Prevent immediate same-concept churn after a clean no-lift calibration.

    The residual audit may keep the largest residual concept at the top even
    after its fresh private pressure has just been tested. If the latest board
    result shows a completed four-card recalibration with no public-transfer
    lift, cool down the newest calibrated private-pressure concept for one
    scheduler epoch so the board rotates to the next transferable wall.
    """

    transferable_concepts = {
        "type_and_return_shape",
        "typed_interface_skeleton",
        "edge_contract_4card",
        "edge_conditions",
        "admissibility_and_interface",
        "algorithmic_planning",
    }
    rows = read_jsonl(REPORTS / "hive_unattended_improvement_ledger.jsonl")
    latest_recalibration: dict[str, Any] | None = None
    latest_transfer_concept: str | None = None
    cooldown: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        contract = row.get("improvement_contract") if isinstance(row.get("improvement_contract"), dict) else {}
        concept = str(row.get("concept") or contract.get("concept") or "")
        if row.get("status") != "done":
            continue
        if concept in transferable_concepts:
            latest_transfer_concept = concept
            continue
        if concept != "private_pressure_four_card_recalibration":
            continue
        latest_recalibration = row
        before = contract.get("before") if isinstance(contract.get("before"), dict) else {}
        after = contract.get("after") if isinstance(contract.get("after"), dict) else {}
        before_broad = safe_float(before.get("broad_public_pass_rate"))
        after_broad = safe_float(after.get("broad_public_pass_rate"))
        before_public = safe_float(before.get("public_transfer_pass_rate"))
        after_public = safe_float(after.get("public_transfer_pass_rate"))
        if after_broad > before_broad or after_public > before_public:
            cooldown.clear()
        elif latest_transfer_concept:
            cooldown.add(latest_transfer_concept)
    if not latest_recalibration:
        return set()

    contract = latest_recalibration.get("improvement_contract") or {}
    before = contract.get("before") if isinstance(contract.get("before"), dict) else {}
    after = contract.get("after") if isinstance(contract.get("after"), dict) else {}
    before_broad = safe_float(before.get("broad_public_pass_rate"))
    after_broad = safe_float(after.get("broad_public_pass_rate"))
    before_public = safe_float(before.get("public_transfer_pass_rate"))
    after_public = safe_float(after.get("public_transfer_pass_rate"))
    if after_broad > before_broad or after_public > before_public:
        return set()
    return cooldown



def code_transfer_rotation_epoch(
    concept: str,
    *,
    transfer: dict[str, Any],
    broad: dict[str, Any],
    guidance: dict[str, Any],
) -> str:
    pressure = []
    for row in transfer.get("concept_transfer_pressure", []):
        if isinstance(row, dict) and str(row.get("concept") or "") == concept:
            pressure.append(
                {
                    "residual_count": row.get("residual_count"),
                    "card_count": row.get("card_count"),
                    "cards": row.get("cards"),
                }
            )
    payload = {
        "concept": concept,
        "broad": {
            "aggregate_pass_rate": get_path(broad, ["summary", "aggregate_pass_rate"], None),
            "aggregate_sts_delta": get_path(broad, ["summary", "aggregate_sts_delta"], None),
            "cards_below_floor": get_path(broad, ["summary", "cards_below_floor"], []),
        },
        "pressure": pressure,
        "guidance": {
            "created_utc": guidance.get("created_utc"),
            "dominant_residual": get_path(guidance, ["diagnosis", "dominant_residual"], None),
            "recommended_intervention": get_path(guidance, ["teacher", "response_json", "recommended_intervention"], None),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def task_from_concept(concept: dict[str, Any]) -> dict[str, Any]:
    command = [str(item) for item in concept.get("command") or []]
    task_id = stable_id("high_transfer", concept.get("concept"), json.dumps(command, sort_keys=True), concept.get("rotation_epoch") or "")
    return {
        "task_id": task_id,
        "created_utc": now(),
        "source": "high_transfer_curriculum_scheduler",
        "kind": "high_transfer_concept_pressure",
        "status": "ready",
        "priority": concept.get("priority", "medium"),
        "title": f"Train transferable concept: {concept.get('concept')}",
        "target_node_id": "best_training_node",
        "payload": {
            "concept": concept.get("concept"),
            "private_pressure": concept.get("private_pressure"),
            "donors": concept.get("donors"),
            "receivers": concept.get("receivers"),
            "transfer_checks": concept.get("transfer_checks"),
            "rotation_epoch": concept.get("rotation_epoch"),
            "public_data_rule": "public_benchmarks_calibration_only",
        },
        "command": command,
    }


def conversation_focus_enabled(policy: dict[str, Any]) -> bool:
    focus = str(get_path(policy, ["personality_core", "near_term_training_focus"], "") or "").lower()
    return "conversation" in focus and ("before" in focus or "temporarily" in focus)


def conversation_lifecycle(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    accuracy = float(summary.get("accuracy", summary.get("average_score", 0.0)) or 0.0)
    case_count = int(summary.get("case_count") or 0)
    turn_count = int(summary.get("turn_count") or 0)
    suite_mode = str(summary.get("suite_mode") or "unknown")
    passed = bool(report.get("passed")) or str(report.get("trigger_state") or "") == "GREEN"
    graduated = bool(summary.get("graduated")) or (
        passed
        and accuracy >= CONVERSATION_GRADUATION_ACCURACY
        and case_count >= CONVERSATION_GRADUATION_MIN_CASES
    )
    if graduated:
        reason = "large_suite_saturated_graduate_to_regression"
    elif passed and accuracy >= CONVERSATION_GRADUATION_ACCURACY and case_count < CONVERSATION_GRADUATION_MIN_CASES:
        reason = "small_suite_saturated_needs_large_calibration"
    elif case_count == 0:
        reason = "missing_conversation_evidence"
    else:
        reason = "active_frontier_until_large_suite_passes"
    return {
        "accuracy": accuracy,
        "case_count": case_count,
        "turn_count": turn_count,
        "suite_mode": suite_mode,
        "passed": passed,
        "graduated": graduated,
        "reason": reason,
    }


def conversation_rotation_epoch() -> str:
    seconds = int(datetime.now(timezone.utc).timestamp())
    return str(seconds - (seconds % CONVERSATION_ROTATION_SECONDS))


def latest_report(*paths: Path) -> dict[str, Any]:
    candidates = [path for path in paths if path.exists()]
    if not candidates:
        return {}
    return read_json(max(candidates, key=lambda path: path.stat().st_mtime), {})


def best_or_file(family: str, path: Path) -> dict[str, Any]:
    return report_evidence_store.best_payload_for_family(report_evidence_store.DEFAULT_DB, family) or read_json(path, {})


def latest_architecture_guidance() -> dict[str, Any]:
    candidates = []
    for path in REPORTS.glob("architecture_guidance_loop*.json"):
        if not path.is_file():
            continue
        report = read_json(path, {})
        recommendation = str(get_path(report, ["teacher", "response_json", "recommended_intervention"], "") or "")
        candidates.append(((1 if recommendation else 0, str(report.get("created_utc") or ""), path.stat().st_mtime), report))
    if not candidates:
        return {}
    return max(candidates, key=lambda item: item[0])[1]


def best_conversation_report(*paths: Path) -> dict[str, Any]:
    stored = report_evidence_store.best_payload_for_family(report_evidence_store.DEFAULT_DB, "conversation_multiturn")
    if stored:
        return stored
    candidates = []
    for path in paths:
        if not path.exists():
            continue
        report = read_json(path, {})
        if not isinstance(report, dict):
            continue
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        accuracy = float(summary.get("accuracy", summary.get("average_score", 0.0)) or 0.0)
        case_count = int(summary.get("case_count") or 0)
        suite_mode = str(summary.get("suite_mode") or "").lower()
        passed = bool(report.get("passed")) or str(report.get("trigger_state") or "") == "GREEN"
        graduated = bool(summary.get("graduated")) or (
            passed
            and accuracy >= CONVERSATION_GRADUATION_ACCURACY
            and case_count >= CONVERSATION_GRADUATION_MIN_CASES
        )
        report["_source_report"] = str(path.relative_to(ROOT)).replace("\\", "/")
        candidates.append(
            (
                (
                    1 if graduated else 0,
                    case_count,
                    1 if suite_mode == "large" else 0,
                    path.stat().st_mtime,
                ),
                report,
            )
        )
    if not candidates:
        return {}
    return max(candidates, key=lambda item: item[0])[1]


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# High-Transfer Curriculum Scheduler",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- concept_count: `{summary.get('concept_count')}`",
        f"- ready_task_count: `{summary.get('ready_task_count')}`",
        f"- donor_receiver_checks: `{summary.get('donor_receiver_checks')}`",
        "",
        "## Concepts",
        "",
    ]
    for row in report.get("concepts", []):
        lines.append(f"- `{row.get('concept')}` {row.get('status')} priority={row.get('priority')} receivers={len(row.get('receivers') or [])}")
    lines.append("")
    return "\n".join(lines)


