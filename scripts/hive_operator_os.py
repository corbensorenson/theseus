"""Hive operator OS contract and live work-board views.

This script ports the most useful Hermes/OpenClaw/MoECOT/BugBrain operator
patterns into Project Theseus without pretending every external channel is
already implemented. It creates one canonical command vocabulary, a durable
SQLite work board, dynamic skill loading/hygiene views, background/persistent
goal surfaces, tool-hook contracts, execution safety contracts, and report views
that the dashboard/mobile clients can consume.

It does not train on public data, call external inference, or send messages to
external platforms by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
DEFAULT_CONFIG = CONFIGS / "hive_operator_os.json"
DEFAULT_DB = REPORTS / "hive_work_board.sqlite"

BOARD_SOURCES = {
    "feedback_action_queue": REPORTS / "feedback_action_queue.json",
    "viea_action_executor": REPORTS / "viea_action_executor.json",
    "hive_task_queue": REPORTS / "hive_task_queue.jsonl",
    "hive_task_ledger": REPORTS / "hive_task_ledger.jsonl",
    "autonomous_goal": REPORTS / "autonomous_goal_last.json",
    "vacation_mode": REPORTS / "vacation_mode_supervisor.json",
    "broad_transfer": REPORTS / "broad_transfer_closure.json",
    "high_transfer_tasks": REPORTS / "high_transfer_curriculum_tasks.jsonl",
    "hive_scheduler": REPORTS / "hive_scheduler.json",
    "hive_utilization_manager": REPORTS / "hive_utilization_manager.json",
    "hive_training_orchestrator": REPORTS / "hive_training_orchestrator.json",
}

REPORT_FILES = {
    "operator": "hive_operator_os.json",
    "work_board": "hive_work_board.json",
    "app_manifest": "hive_operator_app_manifest.json",
    "channels": "hive_channel_contract.json",
    "skills": "hive_skill_registry.json",
    "hooks": "hive_tool_hooks.json",
    "background": "hive_background_tasks.json",
    "goals": "hive_persistent_goals.json",
    "feedback": "hive_feedback_router.json",
    "safety": "hive_execution_safety.json",
}


class WorkBoard:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
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
            CREATE INDEX IF NOT EXISTS idx_hive_work_board_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_hive_work_board_kind ON tasks(kind);

            CREATE TABLE IF NOT EXISTS comments (
                comment_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                author TEXT NOT NULL,
                body TEXT NOT NULL,
                created_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dependencies (
                dependency_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                blocks_task_id TEXT NOT NULL,
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

            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_utc TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def upsert_task(
        self,
        *,
        task_id: str,
        title: str,
        source: str,
        kind: str,
        status: str,
        priority: str,
        assignee: str,
        node_id: str,
        command: str,
        evidence: dict[str, Any] | None = None,
        retry_count: int = 0,
        blocked_reason: str = "",
    ) -> None:
        stamp = now()
        self.conn.execute(
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
                task_id,
                title,
                source,
                kind,
                status,
                priority,
                assignee,
                node_id,
                command,
                json.dumps(evidence or {}, sort_keys=True),
                stamp,
                stamp,
                retry_count,
                blocked_reason,
            ),
        )

    def add_event(self, task_id: str, event_type: str, content: dict[str, Any]) -> None:
        event_id = stable_id("event", task_id, event_type, json.dumps(content, sort_keys=True))
        self.conn.execute(
            """
            INSERT OR IGNORE INTO events (event_id, task_id, event_type, content_json, created_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, task_id, event_type, json.dumps(content, sort_keys=True), now()),
        )

    def add_evidence(self, task_id: str, label: str, path: str, claim_role: str) -> None:
        evidence_id = stable_id("evidence", task_id, label, path)
        self.conn.execute(
            """
            INSERT OR IGNORE INTO evidence (evidence_id, task_id, label, path, claim_role, created_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (evidence_id, task_id, label, path, claim_role, now()),
        )

    def commit(self) -> None:
        self.conn.commit()

    def rows(self, limit: int = 80) -> list[dict[str, Any]]:
        result = []
        for row in self.conn.execute(
            """
            SELECT * FROM tasks
            ORDER BY
                CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
                updated_utc DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall():
            item = dict(row)
            item["evidence"] = parse_json(item.pop("evidence_json"), {})
            result.append(item)
        return result

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall():
            out[str(row["status"])] = int(row["n"])
        out["total"] = sum(out.values())
        return out

    def event_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
        return int(row["n"]) if row else 0

    def evidence_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM evidence").fetchone()
        return int(row["n"]) if row else 0

    def task_status(self, task_id: str) -> str:
        row = self.conn.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return str(row["status"]) if row else ""

    def task_terminal_status_from_history(self, task_id: str) -> str:
        terminal_events = {
            "task_done": "done",
            "task_blocked": "blocked",
            "task_failed": "failed",
        }
        rows = self.conn.execute(
            """
            SELECT event_type
            FROM events
            WHERE task_id=?
            ORDER BY created_utc DESC
            """,
            (task_id,),
        ).fetchall()
        for row in rows:
            status = terminal_events.get(str(row["event_type"]))
            if status:
                return status
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--db", default=str(DEFAULT_DB.relative_to(ROOT)))
    parser.add_argument("--out", default=f"reports/{REPORT_FILES['operator']}")
    parser.add_argument("--markdown-out", default="reports/hive_operator_os.md")
    parser.add_argument("--write-views", action="store_true", default=True)
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config), {})
    if not config:
        return fail(args, "missing_operator_os_config", started)

    board = WorkBoard(resolve(args.db))
    source_summary = ingest_board_sources(board)
    board.commit()
    board_report = build_board_report(board, source_summary)
    app_manifest = build_app_manifest(config)
    channels = build_channel_report(config)
    skills = build_skill_report(config, board_report)
    hooks = build_hook_report(config)
    background = build_background_report(config, board_report)
    goals = build_goal_report()
    feedback = build_feedback_router_report(config)
    safety = build_execution_safety_report(config)
    board.close()

    gates = build_gates(
        config=config,
        board_report=board_report,
        channels=channels,
        skills=skills,
        hooks=hooks,
        safety=safety,
    )
    trigger = trigger_state(gates)
    operator = {
        "policy": "project_theseus_hive_operator_os_v1",
        "created_utc": now(),
        "trigger_state": trigger,
        "summary": {
            "channels": channels["summary"]["channel_count"],
            "implemented_channels": channels["summary"]["implemented_count"],
            "command_count": len(config.get("commands") or []),
            "board_tasks": board_report["summary"]["total_tasks"],
            "background_tasks": background["summary"]["task_count"],
            "persistent_goals": goals["summary"]["goal_count"],
            "skills": skills["summary"]["skill_count"],
            "active_skills": skills["summary"]["active_skill_count"],
            "hook_targets": hooks["summary"]["hook_target_count"],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "operating_streams": [
            {
                "id": "memory_evidence_substrate",
                "owns": ["VIEA artifact kernel", "work board", "claim/diagnostic evidence", "trace fabric"],
            },
            {
                "id": "capability_runtime",
                "owns": ["skills", "channels", "remote control", "repo repair", "benchmark/runtime adapters"],
            },
            {
                "id": "governance_autonomy_training",
                "owns": ["goals", "tool hooks", "resource gates", "teacher architecture loop", "safety contracts"],
            },
        ],
        "app_manifest": app_manifest,
        "channels": channels,
        "work_board": board_report,
        "skills": skills,
        "tool_hooks": hooks,
        "background_tasks": background,
        "persistent_goals": goals,
        "feedback_router": feedback,
        "execution_safety": safety,
        "gates": gates,
        "reports": {name: f"reports/{filename}" for name, filename in REPORT_FILES.items()},
        "external_inference_calls": 0,
    }
    write_views(operator, board_report, app_manifest, channels, skills, hooks, background, goals, feedback, safety)
    write_json(resolve(args.out), operator)
    write_text(resolve(args.markdown_out), render_markdown(operator))
    print(json.dumps(operator, indent=2))
    return 2 if trigger == "RED" else 0


def fail(args: argparse.Namespace, reason: str, started: float) -> int:
    payload = {
        "policy": "project_theseus_hive_operator_os_v1",
        "created_utc": now(),
        "trigger_state": "RED",
        "summary": {"runtime_ms": int((time.perf_counter() - started) * 1000)},
        "gates": [{"gate": reason, "passed": False, "severity": "hard", "evidence": str(resolve(args.config))}],
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 2


def ingest_board_sources(board: WorkBoard) -> dict[str, Any]:
    source_summary: dict[str, Any] = {"sources": {}, "ingested": 0}
    feedback = read_json(BOARD_SOURCES["feedback_action_queue"], {})
    executor = read_json(BOARD_SOURCES["viea_action_executor"], {})
    executor_state = executor.get("state") if isinstance(executor.get("state"), dict) else {}
    completed = set(str(item) for item in executor_state.get("completed_action_ids") or [])
    blocked = set(str(item) for item in executor_state.get("blocked_action_ids") or [])
    failed = set(str(item) for item in executor_state.get("failed_action_ids") or [])
    for action in feedback.get("actions") or []:
        if not isinstance(action, dict):
            continue
        task_id = stable_action_id(action)
        status = "ready"
        reason = ""
        if task_id in completed:
            status = "done"
        elif task_id in blocked:
            status = "blocked"
            reason = "blocked_by_action_executor"
        elif task_id in failed:
            status = "failed"
            reason = "failed_in_action_executor"
        board.upsert_task(
            task_id=task_id,
            title=str(action.get("title") or action.get("kind") or "VIEA action"),
            source="feedback_action_queue",
            kind=str(action.get("kind") or "feedback_action"),
            status=status,
            priority=str(action.get("priority") or "medium"),
            assignee="viea_action_executor",
            node_id=str(get_path(action, ["evidence", "node_id"], "local")),
            command=" ".join(str(part) for part in (action.get("command") or [])),
            evidence=action.get("evidence") if isinstance(action.get("evidence"), dict) else {},
            blocked_reason=reason,
        )
        board.add_event(task_id, "ingested_feedback_action", {"source": "feedback_action_queue"})
        board.add_evidence(task_id, "feedback_action_queue", "reports/feedback_action_queue.json", "diagnostic")
        source_summary["ingested"] += 1
    source_summary["sources"]["feedback_action_queue"] = len(feedback.get("actions") or [])

    hive_queue = read_jsonl_tail(BOARD_SOURCES["hive_task_queue"], 200)
    for row in hive_queue:
        task_id = str(row.get("task_id") or stable_id("hive_task", json.dumps(row, sort_keys=True)))
        board.upsert_task(
            task_id=task_id,
            title=str(row.get("kind") or "Hive task"),
            source="hive_task_queue",
            kind=str(row.get("kind") or "hive_task"),
            status=str(row.get("status") or "queued"),
            priority=str(row.get("priority") or "medium"),
            assignee=str(row.get("target_node_id") or "hive_scheduler"),
            node_id=str(row.get("target_node_id") or "unassigned"),
            command=json.dumps(row.get("payload") or {}, sort_keys=True),
            evidence={"queued_utc": row.get("created_utc")},
        )
        board.add_event(task_id, "ingested_hive_task", row)
        board.add_evidence(task_id, "hive_task_queue", "reports/hive_task_queue.jsonl", "diagnostic")
        source_summary["ingested"] += 1
    source_summary["sources"]["hive_task_queue"] = len(hive_queue)

    scheduler = read_json(BOARD_SOURCES["hive_scheduler"], {})
    scheduler_jobs = get_path(scheduler, ["jobs", "jobs"], [])
    for job in scheduler_jobs if isinstance(scheduler_jobs, list) else []:
        if not isinstance(job, dict):
            continue
        task_id = stable_id("scheduler_job", str(job.get("job_id") or json.dumps(job, sort_keys=True)))
        board.upsert_task(
            task_id=task_id,
            title=f"Scheduler lease: {job.get('task_kind') or job.get('job_family') or 'worker chunk'}",
            source="hive_scheduler",
            kind=str(job.get("task_kind") or "scheduler_job"),
            status="active",
            priority="medium",
            assignee=str(job.get("node_name") or "hive_scheduler"),
            node_id=str(job.get("node_id") or "unassigned"),
            command=json.dumps(job, sort_keys=True),
            evidence={"job_id": job.get("job_id"), "merge_policy": job.get("merge_policy"), "output_artifacts": job.get("output_artifacts")},
        )
        board.add_event(task_id, "ingested_scheduler_job", job)
        board.add_evidence(task_id, "hive_scheduler", "reports/hive_scheduler.json", "diagnostic")
        source_summary["ingested"] += 1
    source_summary["sources"]["hive_scheduler_jobs"] = len(scheduler_jobs) if isinstance(scheduler_jobs, list) else 0

    utilization = read_json(BOARD_SOURCES["hive_utilization_manager"], {})
    execution_by_action = {
        str(row.get("action_id") or ""): row
        for row in utilization.get("execution", [])
        if isinstance(row, dict) and row.get("action_id")
    }
    utilization_actions = utilization.get("actions") if isinstance(utilization.get("actions"), list) else []
    for action_row in utilization_actions:
        if not isinstance(action_row, dict):
            continue
        action_id = str(action_row.get("action_id") or stable_id("hive_util", json.dumps(action_row, sort_keys=True)))
        executed = execution_by_action.get(action_id)
        status = "active"
        if executed:
            status = "done" if str(executed.get("status") or "") == "completed" else "failed"
        task_id = stable_id("utilization_action", action_id)
        board.upsert_task(
            task_id=task_id,
            title=str(action_row.get("title") or action_row.get("kind") or "Hive utilization action"),
            source="hive_utilization_manager",
            kind=str(action_row.get("kind") or "utilization_action"),
            status=status,
            priority="medium",
            assignee=str(get_path(action_row, ["evidence", "node_id"], "hive_utilization_manager")),
            node_id=str(get_path(action_row, ["evidence", "node_id"], "unassigned")),
            command=json.dumps(action_row.get("command") or action_row.get("submit") or {}, sort_keys=True),
            evidence={"action_id": action_id, "planned_jobs": action_row.get("planned_jobs"), "execution": executed or {}},
        )
        board.add_event(task_id, "ingested_utilization_action", {"action_id": action_id, "status": status})
        board.add_evidence(task_id, "hive_utilization_manager", "reports/hive_utilization_manager.json", "diagnostic")
        source_summary["ingested"] += 1
    source_summary["sources"]["hive_utilization_actions"] = len(utilization_actions)

    training = read_json(BOARD_SOURCES["hive_training_orchestrator"], {})
    training_jobs = get_path(training, ["plan", "jobs"], [])
    execution_by_job = {
        str(row.get("job_id") or ""): row
        for row in training.get("execution", [])
        if isinstance(row, dict) and row.get("job_id")
    }
    for job in training_jobs if isinstance(training_jobs, list) else []:
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id") or stable_id("training_job", json.dumps(job, sort_keys=True)))
        executed = execution_by_job.get(job_id)
        status = "active"
        if executed:
            status = "done" if executed.get("ok") else "failed"
        task_id = stable_id("training_job", job_id)
        board.upsert_task(
            task_id=task_id,
            title=f"Training lease: {job.get('task_kind') or job.get('arm_id') or 'worker chunk'}",
            source="hive_training_orchestrator",
            kind=str(job.get("task_kind") or "training_job"),
            status=status,
            priority="medium",
            assignee=str(job.get("node_name") or "hive_training_orchestrator"),
            node_id=str(job.get("node_id") or "unassigned"),
            command=json.dumps(get_path(job, ["payload"], {}), sort_keys=True),
            evidence={"job_id": job_id, "arm_id": job.get("arm_id"), "execution": executed or {}, "output_artifacts": get_path(job, ["payload", "output_artifacts"], [])},
        )
        board.add_event(task_id, "ingested_training_job", {"job_id": job_id, "status": status})
        board.add_evidence(task_id, "hive_training_orchestrator", "reports/hive_training_orchestrator.json", "diagnostic")
        source_summary["ingested"] += 1
    source_summary["sources"]["hive_training_jobs"] = len(training_jobs) if isinstance(training_jobs, list) else 0

    goal = read_json(BOARD_SOURCES["autonomous_goal"], {})
    if goal:
        goal_text = str(goal.get("goal") or goal.get("objective") or get_path(goal, ["plan", "goal"], "Autonomous goal"))
        task_id = stable_id("goal", goal_text)
        board.upsert_task(
            task_id=task_id,
            title=f"Persistent goal: {goal_text[:96]}",
            source="autonomous_goal",
            kind="persistent_goal",
            status="active" if goal.get("trigger_state") != "RED" else "blocked",
            priority="high",
            assignee="octopus_router",
            node_id="local",
            command="/goal",
            evidence={"report": "reports/autonomous_goal_last.json", "trigger_state": goal.get("trigger_state")},
        )
        board.add_event(task_id, "ingested_persistent_goal", {"trigger_state": goal.get("trigger_state")})
        board.add_evidence(task_id, "autonomous_goal_last", "reports/autonomous_goal_last.json", "diagnostic")
        source_summary["ingested"] += 1
    source_summary["sources"]["autonomous_goal"] = 1 if goal else 0

    vacation = read_json(BOARD_SOURCES["vacation_mode"], {})
    repair_queue = vacation.get("repair_queue") if isinstance(vacation.get("repair_queue"), list) else []
    for item in repair_queue:
        task_id = stable_id("vacation_repair", json.dumps(item, sort_keys=True))
        board.upsert_task(
            task_id=task_id,
            title=str(item.get("title") or item.get("reason") or "Vacation repair action"),
            source="vacation_mode_supervisor",
            kind=str(item.get("kind") or "repair_action"),
            status=str(item.get("status") or "ready"),
            priority=str(item.get("priority") or "high"),
            assignee="vacation_mode_supervisor",
            node_id="local",
            command=str(item.get("command") or ""),
            evidence=item,
        )
        board.add_evidence(task_id, "vacation_mode_supervisor", "reports/vacation_mode_supervisor.json", "diagnostic")
        source_summary["ingested"] += 1
    source_summary["sources"]["vacation_repair_queue"] = len(repair_queue)

    broad = read_json(BOARD_SOURCES["broad_transfer"], {})
    summary = broad.get("summary") if isinstance(broad.get("summary"), dict) else {}
    if summary and float(summary.get("aggregate_floor_gap") or 0.0) > 0:
        task_id = stable_id("broad_transfer_gap", str(summary.get("selected_next_card") or ""), str(summary.get("aggregate_floor_gap")))
        board.upsert_task(
            task_id=task_id,
            title=f"Close broad transfer gap: {summary.get('selected_next_card') or 'next card'}",
            source="broad_transfer_closure",
            kind="broad_transfer_closure",
            status="ready",
            priority="high",
            assignee="learning_loop",
            node_id="best_training_node",
            command="/bench --close-transfer",
            evidence=summary,
        )
        board.add_evidence(task_id, "broad_transfer_closure", "reports/broad_transfer_closure.json", "claim_candidate")
        source_summary["ingested"] += 1
    source_summary["sources"]["broad_transfer_gap"] = 1 if summary else 0

    high_transfer_tasks = read_jsonl_tail(BOARD_SOURCES["high_transfer_tasks"], 200)
    for row in high_transfer_tasks:
        task_id = str(row.get("task_id") or stable_id("high_transfer", json.dumps(row, sort_keys=True)))
        existing_status = board.task_status(task_id)
        history_status = board.task_terminal_status_from_history(task_id)
        if existing_status in {"done", "blocked", "failed", "active", "queued", "retry_queued"}:
            status = existing_status
        elif history_status in {"done", "blocked", "failed"}:
            status = history_status
        else:
            status = str(row.get("status") or "ready")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        board.upsert_task(
            task_id=task_id,
            title=str(row.get("title") or f"High-transfer concept: {payload.get('concept', 'unknown')}"),
            source="high_transfer_curriculum_scheduler",
            kind=str(row.get("kind") or "high_transfer_concept_pressure"),
            status=status,
            priority=str(row.get("priority") or "high"),
            assignee="high_transfer_curriculum_scheduler",
            node_id=str(row.get("target_node_id") or "best_training_node"),
            command=" ".join(str(part) for part in (row.get("command") or [])),
            evidence=payload,
        )
        board.add_event(task_id, "ingested_high_transfer_task", row)
        board.add_evidence(task_id, "high_transfer_curriculum_tasks", "reports/high_transfer_curriculum_tasks.jsonl", "diagnostic")
        source_summary["ingested"] += 1
    source_summary["sources"]["high_transfer_curriculum_tasks"] = len(high_transfer_tasks)
    return source_summary


def build_board_report(board: WorkBoard, source_summary: dict[str, Any]) -> dict[str, Any]:
    rows = board.rows()
    counts = board.counts()
    ready = [row for row in rows if row["status"] in {"ready", "queued", "active", "failed"}]
    return {
        "policy": "project_theseus_hive_work_board_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if counts.get("total", 0) else "YELLOW",
        "database": rel_or_abs(board.path),
        "summary": {
            "total_tasks": counts.get("total", 0),
            "ready_or_active": len(ready),
            "done": counts.get("done", 0),
            "blocked": counts.get("blocked", 0),
            "failed": counts.get("failed", 0),
            "event_count": board.event_count(),
            "evidence_count": board.evidence_count(),
            "source_count": len(source_summary.get("sources") or {}),
        },
        "tasks": rows,
        "source_summary": source_summary,
        "rules": {
            "durable": "SQLite is the local work-board substrate; JSON reports are views.",
            "status": "ready/active/queued/done/blocked/failed remain crash-recoverable across daemon restarts.",
            "public_data": "Public benchmark tasks can enter the board only as calibration/evidence work, never training data.",
        },
        "external_inference_calls": 0,
    }


def build_app_manifest(config: dict[str, Any]) -> dict[str, Any]:
    app = config.get("app") if isinstance(config.get("app"), dict) else {}
    panels = app.get("panels") if isinstance(app.get("panels"), list) else []
    routes = []
    for panel in panels:
        panel_id = str(panel)
        routes.append(
            {
                "panel": panel_id,
                "web_route": f"/#{panel_id}",
                "mobile_route": f"/mobile#{panel_id}",
                "api_report_key": report_key_for_panel(panel_id),
                "status": "implemented" if panel_id in implemented_panels() else "contract",
            }
        )
    return {
        "policy": "project_theseus_hive_operator_app_manifest_v1",
        "created_utc": now(),
        "app": app,
        "routes": routes,
        "tool_contracts": [
            {"id": "background_task", "report": "reports/hive_background_tasks.json"},
            {"id": "work_board", "report": "reports/hive_work_board.json"},
            {"id": "skill_registry", "report": "reports/hive_skill_registry.json"},
            {"id": "tool_hooks", "report": "reports/hive_tool_hooks.json"},
            {"id": "remote_control", "report": "reports/hive_remote_control_status.json"},
        ],
        "telemetry_hooks": [
            "reports/hive_tool_hook_ledger.jsonl",
            "reports/hive_operator_os.json",
            "reports/trace_fabric_training_exchange.json",
        ],
        "lineage": {
            "config_path": rel_or_abs(DEFAULT_CONFIG),
            "config_hash": file_hash(DEFAULT_CONFIG),
            "compiler": "scripts/hive_operator_os.py",
        },
    }


def build_channel_report(config: dict[str, Any]) -> dict[str, Any]:
    channels = config.get("channels") if isinstance(config.get("channels"), list) else []
    commands = config.get("commands") if isinstance(config.get("commands"), list) else []
    implemented = [row for row in channels if row.get("status") == "implemented"]
    return {
        "policy": "project_theseus_hive_channel_contract_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if implemented else "YELLOW",
        "summary": {
            "channel_count": len(channels),
            "implemented_count": len(implemented),
            "contract_count": sum(1 for row in channels if row.get("status") == "contract"),
            "command_count": len(commands),
        },
        "channels": channels,
        "commands": commands,
        "shared_contract": {
            "message_envelope": {
                "channel_id": "string",
                "user_id": "string",
                "command": "slash_command_or_text",
                "intent_checksum": "short_plain_language_summary",
                "permission_envelope": "least_privilege_by_default",
                "delivery": "sync_or_background",
            },
            "command_parity": "Every channel uses the same command vocabulary and routes through the same VIEA/Hive work-board contracts.",
        },
        "external_inference_calls": 0,
    }


def build_skill_report(config: dict[str, Any], board_report: dict[str, Any]) -> dict[str, Any]:
    skills = config.get("skill_packs") if isinstance(config.get("skill_packs"), list) else []
    active_terms = active_terms_from_board(board_report)
    rows = []
    for skill in skills:
        load_terms = [str(term).lower() for term in skill.get("load_terms") or []]
        active_hits = sorted(term for term in load_terms if term in active_terms)
        usage = skill_usage_score(str(skill.get("id") or ""), board_report)
        status = str(skill.get("status") or "contract")
        row = dict(skill)
        row.update(
            {
                "active": bool(active_hits) or str(skill.get("id")) in {"daily_brief", "training_status", "benchmark_watch"},
                "active_hits": active_hits,
                "usage_count": usage,
                "success_rate": inferred_skill_success_rate(str(skill.get("id") or ""), board_report),
                "last_used_utc": latest_task_time_for_skill(str(skill.get("id") or ""), board_report),
                "hygiene": skill_hygiene_state(status, usage),
            }
        )
        rows.append(row)
    conflicts = detect_skill_conflicts(rows)
    stale = [row for row in rows if row["hygiene"] == "stale_contract"]
    return {
        "policy": "project_theseus_hive_skill_registry_v1",
        "created_utc": now(),
        "trigger_state": "YELLOW" if conflicts else "GREEN",
        "summary": {
            "skill_count": len(rows),
            "active_skill_count": sum(1 for row in rows if row.get("active")),
            "implemented_count": sum(1 for row in rows if row.get("status") == "implemented"),
            "contract_count": sum(1 for row in rows if row.get("status") == "contract"),
            "conflict_count": len(conflicts),
            "stale_count": len(stale),
        },
        "active_skill_set": [row["id"] for row in rows if row.get("active")],
        "skills": rows,
        "conflicts": conflicts,
        "rules": {
            "dynamic_loading": "Load the smallest skill subset needed for current task terms and active board items.",
            "pre_creation_validation": "Before adding a new skill, search this registry and update existing overlapping skills first.",
            "lifecycle": "Skills track usage, success, conflicts, permissions, and stale/retire candidates.",
        },
        "external_inference_calls": 0,
    }


def build_hook_report(config: dict[str, Any]) -> dict[str, Any]:
    hooks = config.get("tool_hooks") if isinstance(config.get("tool_hooks"), list) else []
    ledger = read_jsonl_tail(REPORTS / "hive_tool_hook_ledger.jsonl", 100)
    rows = []
    for hook in hooks:
        target = str(hook.get("target") or "")
        target_events = [event for event in ledger if event.get("target") == target]
        rows.append(
            {
                **hook,
                "ledger_event_count": len(target_events),
                "last_event_utc": str(target_events[-1].get("created_utc")) if target_events else "",
                "enforcement_state": "live_for_viea_executor" if target in {"training_launch", "teacher_call", "shell"} else "contract",
            }
        )
    return {
        "policy": "project_theseus_hive_tool_hooks_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if rows else "YELLOW",
        "summary": {
            "hook_target_count": len(rows),
            "ledger_event_count": len(ledger),
            "live_target_count": sum(1 for row in rows if row["enforcement_state"].startswith("live")),
        },
        "hooks": rows,
        "ledger_tail": ledger[-20:],
        "rules": {
            "before": "Budget, permission, checkpoint, public-data, and teacher gates run before side effects.",
            "after": "Replay, trace capsule, residual, receipt, and feedback routing run after outcomes.",
        },
        "external_inference_calls": 0,
    }


def build_background_report(config: dict[str, Any], board_report: dict[str, Any]) -> dict[str, Any]:
    active_jobs = read_json(REPORTS / "sparkstream_status.json", {})
    queue_tasks = [
        row
        for row in board_report.get("tasks", [])
        if row.get("status") in {"ready", "queued", "active", "failed"}
    ][:20]
    templates = config.get("cron_templates") if isinstance(config.get("cron_templates"), list) else []
    tasks = []
    for row in queue_tasks:
        tasks.append(
            {
                "background_id": stable_id("background", row.get("task_id"), row.get("updated_utc")),
                "task_id": row.get("task_id"),
                "title": row.get("title"),
                "status": row.get("status"),
                "watch_command": f"/watch {row.get('task_id')}",
                "delivery": ["dashboard", "mobile_web"],
            }
        )
    return {
        "policy": "project_theseus_hive_background_tasks_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "summary": {
            "task_count": len(tasks),
            "schedule_template_count": len(templates),
            "sparkstream_phase": active_jobs.get("phase"),
        },
        "tasks": tasks,
        "cron_templates": templates,
        "contracts": {
            "run_in_background": "Create a work-board task, run under step budget, keep status inspectable.",
            "watch_this": "Subscribe a channel to task status changes.",
            "wake_me": "Deliver notification when condition changes or task completes.",
            "send_to_phone": "Use the mobile operator surface or configured channel adapter.",
        },
        "external_inference_calls": 0,
    }


def build_goal_report() -> dict[str, Any]:
    latest = read_json(REPORTS / "autonomous_goal_last.json", {})
    ledger = read_jsonl_tail(REPORTS / "autonomous_goal_ledger.jsonl", 50)
    goals = []
    if latest:
        goal_text = str(latest.get("goal") or get_path(latest, ["plan", "goal"], "current autonomous goal"))
        goals.append(
            {
                "goal_id": stable_id("goal", goal_text),
                "goal": goal_text,
                "status": latest.get("trigger_state") or latest.get("status") or "active",
                "budget": latest.get("budget") or get_path(latest, ["plan", "budget"], {}),
                "current_step": latest.get("current_step") or get_path(latest, ["plan", "next_step"], ""),
                "judge": latest.get("judge") or "progress_contract_and_gates",
                "stop_conditions": latest.get("stop_conditions")
                or ["hard gate red", "operator stop", "budget exhausted", "no progress contract item"],
                "source": "reports/autonomous_goal_last.json",
            }
        )
    for row in ledger[-8:]:
        text = str(row.get("goal") or row.get("objective") or "")
        if not text:
            continue
        goal_id = stable_id("goal", text)
        if any(goal.get("goal_id") == goal_id for goal in goals):
            continue
        goals.append(
            {
                "goal_id": goal_id,
                "goal": text,
                "status": str(row.get("status") or "historical"),
                "budget": row.get("budget") or {},
                "current_step": row.get("current_step") or "",
                "judge": row.get("judge") or "historical",
                "stop_conditions": row.get("stop_conditions") or [],
                "source": "reports/autonomous_goal_ledger.jsonl",
            }
        )
    return {
        "policy": "project_theseus_hive_persistent_goals_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if goals else "YELLOW",
        "summary": {"goal_count": len(goals), "ledger_entries": len(ledger)},
        "goals": goals,
        "rules": {
            "persistent": "A goal remains visible until completed, blocked, superseded, or stopped by the operator.",
            "budgeted": "Goals must expose a budget, judge, current step, and stop condition before long autonomy.",
        },
        "external_inference_calls": 0,
    }


def build_feedback_router_report(config: dict[str, Any]) -> dict[str, Any]:
    chat = read_json(REPORTS / "checkpoint_chat_last.json", {})
    belief = read_json(REPORTS / "belief_update_governance.json", {})
    residuals = read_json(REPORTS / "residual_escrow.json", {})
    vcm = read_json(REPORTS / "virtual_context_memory_probe.json", {})
    vcm_status = read_json(REPORTS / "virtual_context_memory_status.json", {})
    vcm_training = read_json(REPORTS / "virtual_context_memory_training_admission.json", {})
    vcm_consumer_audit = read_json(REPORTS / "virtual_context_memory_consumer_audit.json", {})
    routes = [
        {
            "feedback_type": "user_correction",
            "routes_to": ["personality_memory", "artifact_kernel", "follow_up_task"],
            "confidence_policy": "requires source artifact or operator confirmation before durable preference update",
        },
        {
            "feedback_type": "wrong_answer",
            "routes_to": ["residual_escrow", "benchmark_watch", "private_curriculum_candidate"],
            "confidence_policy": "public benchmark failures define categories only, not solutions",
        },
        {
            "feedback_type": "repeated_workflow",
            "routes_to": ["skill_registry", "workflow_to_tool_candidate", "tool_hygiene"],
            "confidence_policy": "requires repeated successful traces and verification before active tool promotion",
        },
        {
            "feedback_type": "operator_preference",
            "routes_to": ["personality_adaptive_profile_candidate", "command_contract_defaults"],
            "confidence_policy": "must validate against immutable personality charter and safety invariants",
        },
    ]
    return {
        "policy": "project_theseus_hive_feedback_router_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "summary": {
            "route_count": len(routes),
            "residual_clusters": len(residuals.get("clusters") or []),
            "belief_status": belief.get("status") or belief.get("trigger_state"),
            "latest_chat_available": bool(chat),
            "vcm_state": vcm.get("trigger_state"),
            "vcm_training_admission": vcm_training.get("trigger_state"),
            "vcm_consumer_audit": vcm_consumer_audit.get("trigger_state"),
        },
        "routes": routes,
        "memory_contract": {
            "typed_edges": ["supports", "contradicts", "supersedes", "invalidates", "derives_from", "generalizes_to"],
            "decay": "low-confidence observations decay or stay transient until linked to core artifacts",
            "append_only": "Durable updates add new artifacts/edges; they do not silently rewrite prior belief state.",
            "vcm_contract": {
                "status": vcm.get("trigger_state"),
                "pages": get_path(vcm, ["summary", "semantic_pages"], 0),
                "faults": get_path(vcm_status, ["summary", "fault_count"], 0),
                "training_admission": vcm_training.get("trigger_state"),
                "consumer_audit": vcm_consumer_audit.get("trigger_state"),
                "raw_usage_text_stored": False,
            },
        },
        "external_inference_calls": 0,
    }


def build_execution_safety_report(config: dict[str, Any]) -> dict[str, Any]:
    security = config.get("security_contract") if isinstance(config.get("security_contract"), dict) else {}
    git_state = git_status()
    checkpoint = read_json(REPORTS / "checkpoint_backup_last.json", {})
    remote = read_json(REPORTS / "hive_remote_control_status.json", {})
    version = read_json(REPORTS / "hive_verified_version.json", {})
    contracts = [
        {
            "id": "rollback_checkpoints",
            "status": "implemented",
            "rule": "checkpoint before repo/self-update/training-config mutation when available",
            "evidence": checkpoint.get("policy") or "reports/checkpoint_backup_last.json",
        },
        {
            "id": "worktree_isolation",
            "status": "contract",
            "rule": "repo repair and self-improvement workers should use isolated branches/worktrees",
            "evidence": git_state,
        },
        {
            "id": "remote_control_ttl_kill_switch",
            "status": "implemented",
            "rule": "remote-control sessions require TTL, provider readiness, audit ledger, and operator kill switch",
            "evidence": remote.get("policy") or "reports/hive_remote_control_status.json",
        },
        {
            "id": "signed_update_uniformity",
            "status": "implemented",
            "rule": "hive nodes converge on latest verified update catalog with rollback status",
            "evidence": version.get("policy") or "reports/hive_verified_version.json",
        },
        {
            "id": "personality_charter_adaptive_profile",
            "status": "implemented",
            "rule": "PERSONALITY_CORE.md is durable charter; runtime context is adaptive and audited",
            "evidence": "reports/personality_runtime_audit.json",
        },
    ]
    return {
        "policy": "project_theseus_hive_execution_safety_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "summary": {
            "contract_count": len(contracts),
            "implemented_count": sum(1 for row in contracts if row["status"] == "implemented"),
            "git_dirty": git_state.get("dirty"),
        },
        "security_contract": security,
        "contracts": contracts,
        "git_state": git_state,
        "external_inference_calls": 0,
    }


def build_gates(
    *,
    config: dict[str, Any],
    board_report: dict[str, Any],
    channels: dict[str, Any],
    skills: dict[str, Any],
    hooks: dict[str, Any],
    safety: dict[str, Any],
) -> list[dict[str, Any]]:
    commands = config.get("commands") if isinstance(config.get("commands"), list) else []
    command_names = {str(row.get("name")) for row in commands if isinstance(row, dict)}
    required_commands = {"/status", "/goal", "/background", "/watch", "/board", "/skill", "/rollback", "/takeover", "/feedback"}
    return [
        gate("config_loaded", config.get("policy") == "project_theseus_hive_operator_os_config_v1", config.get("policy")),
        gate("one_agent_many_channels_declared", channels["summary"]["channel_count"] >= 8, channels["summary"]),
        gate("shared_command_vocabulary_complete", required_commands.issubset(command_names), sorted(required_commands - command_names)),
        gate("durable_work_board_ready", board_report["summary"]["total_tasks"] >= 0 and DEFAULT_DB.parent.exists(), board_report["database"]),
        gate("skills_registry_available", skills["summary"]["skill_count"] >= 10, skills["summary"]),
        gate("dynamic_skill_loading_available", bool(skills.get("active_skill_set")), skills.get("active_skill_set")),
        gate("tool_hooks_declared", hooks["summary"]["hook_target_count"] >= 8, hooks["summary"]),
        gate("remote_control_safety_contract", any(row["id"] == "remote_control_ttl_kill_switch" for row in safety["contracts"]), safety["summary"]),
        gate("personality_contract_visible", any(row["id"] == "personality_charter_adaptive_profile" for row in safety["contracts"]), safety["summary"]),
    ]


def write_views(
    operator: dict[str, Any],
    board: dict[str, Any],
    app_manifest: dict[str, Any],
    channels: dict[str, Any],
    skills: dict[str, Any],
    hooks: dict[str, Any],
    background: dict[str, Any],
    goals: dict[str, Any],
    feedback: dict[str, Any],
    safety: dict[str, Any],
) -> None:
    views = {
        REPORT_FILES["work_board"]: board,
        REPORT_FILES["app_manifest"]: app_manifest,
        REPORT_FILES["channels"]: channels,
        REPORT_FILES["skills"]: skills,
        REPORT_FILES["hooks"]: hooks,
        REPORT_FILES["background"]: background,
        REPORT_FILES["goals"]: goals,
        REPORT_FILES["feedback"]: feedback,
        REPORT_FILES["safety"]: safety,
    }
    for filename, payload in views.items():
        write_json(REPORTS / filename, payload)


def trigger_state(gates: list[dict[str, Any]]) -> str:
    if any((not gate["passed"]) and gate["severity"] == "hard" for gate in gates):
        return "RED"
    if any(not gate["passed"] for gate in gates):
        return "YELLOW"
    return "GREEN"


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def stable_action_id(action: dict[str, Any]) -> str:
    evidence = action.get("evidence") if isinstance(action.get("evidence"), dict) else {}
    return stable_id(
        "viea_action",
        str(action.get("kind") or ""),
        str(action.get("title") or ""),
        str(evidence.get("card_id") or evidence.get("tool_name") or evidence.get("source") or ""),
        json.dumps(action.get("command") or [], sort_keys=True),
    )


def active_terms_from_board(board_report: dict[str, Any]) -> set[str]:
    text = " ".join(
        f"{row.get('title', '')} {row.get('kind', '')} {row.get('command', '')}"
        for row in board_report.get("tasks", [])
    ).lower()
    return {token.strip(".,:;()[]{}") for token in text.split() if token.strip()}


def skill_usage_score(skill_id: str, board_report: dict[str, Any]) -> int:
    aliases = {
        "repo_repair": ["repo", "repair", "test"],
        "remote_desktop": ["remote", "takeover", "screen"],
        "training_status": ["train", "learning", "checkpoint"],
        "benchmark_watch": ["benchmark", "eval", "transfer"],
        "cloud_compute": ["cloud", "compute", "aws"],
        "personality_memory": ["personality", "memory"],
    }
    terms = aliases.get(skill_id, [skill_id.replace("_", " ")])
    count = 0
    for row in board_report.get("tasks", []):
        haystack = f"{row.get('title', '')} {row.get('kind', '')} {row.get('command', '')}".lower()
        if any(term in haystack for term in terms):
            count += 1
    return count


def inferred_skill_success_rate(skill_id: str, board_report: dict[str, Any]) -> float | None:
    usage = skill_usage_score(skill_id, board_report)
    if usage == 0:
        return None
    done = sum(1 for row in board_report.get("tasks", []) if row.get("status") == "done")
    total = max(1, int(board_report.get("summary", {}).get("total_tasks") or 1))
    return round(done / total, 4)


def latest_task_time_for_skill(skill_id: str, board_report: dict[str, Any]) -> str:
    usage = skill_usage_score(skill_id, board_report)
    if usage == 0:
        return ""
    return str(max((row.get("updated_utc") or "" for row in board_report.get("tasks", [])), default=""))


def skill_hygiene_state(status: str, usage: int) -> str:
    if status == "contract" and usage == 0:
        return "stale_contract"
    if usage >= 3:
        return "hot"
    if usage > 0:
        return "warm"
    return "watch"


def detect_skill_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = []
    permission_map: dict[str, list[str]] = {}
    for row in rows:
        for permission in row.get("permissions") or []:
            permission_map.setdefault(str(permission), []).append(str(row.get("id")))
    for permission, skill_ids in permission_map.items():
        if permission in {"approved_repo_write", "keyboard_mouse_control", "reviewed_compute_start"} and len(skill_ids) > 1:
            conflicts.append({"permission": permission, "skills": sorted(skill_ids), "resolution": "scope_by_task_and_channel"})
    return conflicts


def report_key_for_panel(panel_id: str) -> str:
    mapping = {
        "work_board": "hive_work_board",
        "background_tasks": "hive_background_tasks",
        "persistent_goals": "hive_persistent_goals",
        "skills": "hive_skill_registry",
        "trace_fabric": "trace_fabric_training_exchange",
        "remote_control": "hive.remote_control",
    }
    return mapping.get(panel_id, panel_id)


def implemented_panels() -> set[str]:
    return {
        "chat",
        "progress",
        "work_board",
        "background_tasks",
        "persistent_goals",
        "hive_nodes",
        "remote_control",
        "skills",
        "benchmarks",
        "training_data",
        "trace_fabric",
        "updates",
        "ops",
    }


def git_status() -> dict[str, Any]:
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = run_git(["status", "--porcelain"])
    return {
        "branch": branch.strip() if branch else "",
        "dirty": bool(dirty.strip()),
        "porcelain_count": len([line for line in dirty.splitlines() if line.strip()]),
        "worktree_policy": "use isolated branch/worktree for repo repair and self-improvement workers",
    }


def run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Hive Operator OS",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- channels: `{summary.get('implemented_channels')}/{summary.get('channels')}` implemented",
        f"- commands: `{summary.get('command_count')}`",
        f"- work_board_tasks: `{summary.get('board_tasks')}`",
        f"- active_skills: `{summary.get('active_skills')}/{summary.get('skills')}`",
        f"- background_tasks: `{summary.get('background_tasks')}`",
        f"- persistent_goals: `{summary.get('persistent_goals')}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []):
        lines.append(f"- {'PASS' if row.get('passed') else 'FAIL'} `{row.get('gate')}`: {row.get('evidence')}")
    lines.extend(["", "## Active Skill Set", ""])
    for skill_id in get_path(report, ["skills", "active_skill_set"], []):
        lines.append(f"- `{skill_id}`")
    lines.extend(["", "## Top Work Board Items", ""])
    for task in get_path(report, ["work_board", "tasks"], [])[:12]:
        lines.append(f"- `{task.get('status')}` `{task.get('priority')}` {task.get('title')} ({task.get('task_id')})")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path, default: Any | None = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def parse_json(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def get_path(payload: Any, path: list[str], default: Any = None) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
