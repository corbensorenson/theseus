#!/usr/bin/env python3
"""Autonomy Rotation Governor V2.

This is the small policy surface that keeps unattended learning from getting
stuck in stale housekeeping. It refreshes the shared node view, retires
already-satisfied high-transfer targets, reports lane pressure, and exposes the
next selected board work without executing a training command itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hive_work_board_executor as board  # noqa: E402


REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "autonomy_rotation_governor_v2.json"
DEFAULT_MARKDOWN = REPORTS / "autonomy_rotation_governor_v2.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(board.DEFAULT_DB.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--execute", action="store_true", help="Apply board maintenance, but do not execute training commands.")
    parser.add_argument("--max-selected", type=int, default=8)
    args = parser.parse_args()

    started = time.perf_counter()
    db_path = board.resolve(args.db)
    board.ensure_schema(db_path)
    maintenance: dict[str, Any] = {}
    if args.execute:
        maintenance = {
            "high_transfer_sync_result": board.sync_high_transfer_scheduler_tasks(db_path),
            "retirement_result": board.retire_regression_only_high_transfer_tasks(db_path),
            "high_transfer_supersede_result": board.supersede_stale_high_transfer_tasks(db_path),
            "feedback_guard_result": board.block_stale_feedback_actions(db_path),
            "teacher_retirement_result": board.retire_reclassified_teacher_escalations(db_path),
            "satisfied_target_result": board.satisfy_fresh_high_transfer_targets(db_path),
            "stale_hive_queue_result": board.retire_stale_hive_task_queue_chunks(db_path),
        }

    node_context = board.load_node_context()
    tasks = board.load_tasks(db_path, limit=5000)
    selected = board.select_tasks(tasks, node_context, only_task_id="", limit=max(1, args.max_selected))
    lane_backlog = lane_counts(tasks, statuses=board.READY_STATUSES)
    recent_lanes = board.recent_rotation_lane_counts()
    registry = node_context.get("node_registry") if isinstance(node_context.get("node_registry"), dict) else {}
    nodes = registry.get("nodes") if isinstance(registry.get("nodes"), list) else []
    local_node = next((node for node in nodes if "windows" in str(node.get("node_name") or "").lower() or str(node.get("node_id") or "") == "local"), nodes[0] if nodes else {})
    readiness = read_json(REPORTS / "overnight_learning_readiness.json", {})
    selected_satisfied = [
        {
            "task_id": task.get("task_id"),
            "concept": board.get_path(task, ["evidence", "concept"], ""),
            "satisfaction": board.high_transfer_target_satisfied(str(board.get_path(task, ["evidence", "concept"], "") or "")),
        }
        for task in selected
        if board.get_path(task, ["evidence", "concept"], "") in board.SATISFIABLE_HIGH_TRANSFER_TARGETS
    ]
    stale_selected = [row for row in selected_satisfied if row.get("satisfaction", {}).get("satisfied")]
    trigger = "GREEN"
    if stale_selected:
        trigger = "YELLOW"
    if not selected and any(str(task.get("status") or "") in board.READY_STATUSES for task in tasks):
        trigger = "YELLOW"

    report = {
        "policy": "project_theseus_autonomy_rotation_governor_v2",
        "created_utc": now(),
        "trigger_state": trigger,
        "summary": {
            "execute_requested": bool(args.execute),
            "selected_count": len(selected),
            "ready_tasks": sum(1 for row in tasks if str(row.get("status") or "") in board.READY_STATUSES),
            "lane_backlog": lane_backlog,
            "recent_lane_counts": recent_lanes,
            "fresh_selected_targets_remaining": len(stale_selected),
            "windows_training_allowed": local_node.get("training_allowed"),
            "windows_training_blockers": local_node.get("training_blockers"),
            "overnight_launch_ready": readiness.get("overnight_launch_ready"),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "maintenance": maintenance,
        "node_registry": registry,
        "selected": selected,
        "selected_target_satisfaction": selected_satisfied,
        "rules": {
            "runtime_storage": "low source-drive space does not block training when heavy generated writes are redirected to a roomy runtime drive",
            "satisfied_targets": "fresh green/saturated non-code target reports are marked done so the board can rotate",
            "lane_quota": "selection penalizes recently used lanes and rotates typed-interface, conversation, games, tool use, repo repair, capsules, and code transfer",
            "public_data": "public benchmarks remain calibration-only; no public solutions or tests are admitted as training rows",
        },
        "external_inference_calls": 0,
    }
    write_json(board.resolve(args.out), report)
    write_text(board.resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if trigger in {"GREEN", "YELLOW"} else 2


def lane_counts(tasks: list[dict[str, Any]], *, statuses: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        if str(task.get("status") or "") not in statuses:
            continue
        lane = board.task_rotation_lane(task)
        counts[lane] = counts.get(lane, 0) + 1
    return counts


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Autonomy Rotation Governor V2",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- selected_count: `{summary.get('selected_count')}`",
        f"- ready_tasks: `{summary.get('ready_tasks')}`",
        f"- fresh_selected_targets_remaining: `{summary.get('fresh_selected_targets_remaining')}`",
        f"- windows_training_allowed: `{summary.get('windows_training_allowed')}`",
        f"- windows_training_blockers: `{summary.get('windows_training_blockers')}`",
        "",
        "## Lane Backlog",
        "",
    ]
    for lane, count in sorted((summary.get("lane_backlog") or {}).items()):
        lines.append(f"- `{lane}`: `{count}`")
    lines.extend(["", "## Selected", ""])
    for task in report.get("selected", [])[:10]:
        lines.append(f"- `{board.task_rotation_lane(task)}` {task.get('title')} task={task.get('task_id')}")
    if not report.get("selected"):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
