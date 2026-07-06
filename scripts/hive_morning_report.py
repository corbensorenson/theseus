"""Readable unattended-learning morning report.

Summarizes what improved, what failed, what Theseus learned, what needs
approval, and what should run next. This is a report over ledgers; it does not
perform training or external inference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "hive_morning_report.json"
DEFAULT_MARKDOWN = REPORTS / "hive_morning_report.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    improvements = read_jsonl(REPORTS / "hive_unattended_improvement_ledger.jsonl")[-80:]
    board = read_json(REPORTS / "hive_work_board.json", {})
    board_exec = read_json(REPORTS / "hive_work_board_executor.json", {})
    vacation = read_json(REPORTS / "vacation_mode_supervisor.json", {})
    high_transfer = read_json(REPORTS / "high_transfer_curriculum_scheduler.json", {})
    conversation = best_report(
        REPORTS / "high_transfer_multi_turn_conversation_hard_v3.json",
        REPORTS / "high_transfer_multi_turn_conversation_hard_v2.json",
        REPORTS / "high_transfer_multi_turn_conversation_hard.json",
        REPORTS / "high_transfer_multi_turn_conversation.json",
        REPORTS / "multi_turn_conversation_benchmark.json",
    )
    tool_use = read_json(REPORTS / "high_transfer_long_horizon_tool_use.json", {})
    a_plus = read_json(REPORTS / "a_plus_operating_scorecard.json", {})
    edge_gate = best_report(
        REPORTS / "edge_obligation_decode_gate_v1_private_pressure_private.json",
        REPORTS / "edge_obligation_decode_gate_v1_private.json",
    )
    repo_repair = read_json(REPORTS / "high_transfer_repo_repair_learner.json", read_json(REPORTS / "viea_repo_repair_learner.json", {}))
    teacher = read_json(REPORTS / "hive_teacher_auto_escalation.json", read_json(REPORTS / "teacher_architect_experiment_runner.json", {}))
    version = read_json(REPORTS / "hive_version_convergence.json", {})

    improved = [row for row in improvements if get_path(row, ["improvement_contract", "passed"], False)]
    failed = [row for row in improvements if not get_path(row, ["improvement_contract", "passed"], False)]
    signal_counts = Counter(
        kind
        for row in improved
        for kind in (get_path(row, ["improvement_contract", "signal_kinds"], []) or [])
    )
    residual_counts = Counter(str(get_path(row, ["improvement_contract", "residual_cluster"], "unknown")) for row in failed)
    ready_tasks = [
        row
        for row in (board.get("tasks") or [])
        if isinstance(row, dict) and str(row.get("status") or "") in {"ready", "queued", "failed"}
    ][:10]
    blocked_tasks = [
        row
        for row in (board.get("tasks") or [])
        if isinstance(row, dict) and str(row.get("status") or "") == "blocked"
    ][:10]
    approvals = []
    if blocked_tasks:
        approvals.append("review blocked board tasks")
    if teacher.get("trigger_state") == "YELLOW":
        approvals.append("review teacher architecture queue")
    if version.get("trigger_state") == "RED":
        approvals.append("resolve hive version drift before remote execution")

    payload = {
        "policy": "project_theseus_hive_morning_report_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if improved or ready_tasks else "YELLOW",
        "summary": {
            "improvement_events": len(improved),
            "no_progress_or_failure_events": len(failed),
            "ready_task_count": get_path(board, ["summary", "ready_or_active"], get_path(board_exec, ["summary", "ready_tasks"], None)),
            "blocked_task_count": get_path(board, ["summary", "blocked"], None),
            "high_transfer_ready_tasks": get_path(high_transfer, ["summary", "ready_task_count"], None),
            "conversation_accuracy": get_path(conversation, ["summary", "accuracy"], None),
            "conversation_suite_mode": get_path(conversation, ["summary", "suite_mode"], None),
            "tool_use_cases": get_path(tool_use, ["summary", "case_count"], None),
            "tool_use_pass_rate": get_path(tool_use, ["summary", "pass_rate"], None),
            "edge_obligation_pass_rate": get_path(edge_gate, ["summary", "private_pass_rate"], None),
            "edge_obligation_ready": edge_gate.get("ready_for_public_calibration") if edge_gate else None,
            "repo_repair_rows": get_path(repo_repair, ["summary", "code_lm_row_count"], None),
            "teacher_stage_count": get_path(teacher, ["summary", "executed_stage_count"], None),
            "a_plus_grade": get_path(a_plus, ["summary", "overall_grade"], None),
            "a_plus_score": get_path(a_plus, ["summary", "overall_score"], None),
        },
        "what_improved": [
            {"kind": kind, "count": count}
            for kind, count in signal_counts.most_common()
        ],
        "what_failed": [
            {"cluster": cluster, "count": count}
            for cluster, count in residual_counts.most_common(10)
        ],
        "what_theseus_learned": learned_items(high_transfer, conversation, tool_use, repo_repair, teacher, a_plus, edge_gate),
        "needs_approval": approvals,
        "what_runs_next": [
            {
                "task_id": row.get("task_id"),
                "title": row.get("title"),
                "status": row.get("status"),
                "priority": row.get("priority"),
                "assignee": row.get("assignee"),
            }
            for row in ready_tasks[:8]
        ],
        "latest_vacation_state": {
            "trigger_state": vacation.get("trigger_state"),
            "summary": vacation.get("summary"),
        },
        "score_semantics": "operator morning report; diagnostic, not promotion evidence",
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0


def learned_items(high_transfer: dict[str, Any], conversation: dict[str, Any], tool_use: dict[str, Any], repo_repair: dict[str, Any], teacher: dict[str, Any], a_plus: dict[str, Any], edge_gate: dict[str, Any]) -> list[str]:
    items = []
    for concept in high_transfer.get("concepts", []) if isinstance(high_transfer.get("concepts"), list) else []:
        if isinstance(concept, dict) and concept.get("status") == "ready":
            items.append(f"queued transferable concept pressure: {concept.get('concept')}")
    if conversation:
        items.append(f"conversation {get_path(conversation, ['summary', 'suite_mode'], 'unknown')} accuracy={get_path(conversation, ['summary', 'accuracy'], 'unknown')}")
    if tool_use:
        items.append(f"tool-use cases={get_path(tool_use, ['summary', 'case_count'], 'unknown')} pass_rate={get_path(tool_use, ['summary', 'pass_rate'], 'unknown')}")
    if repo_repair:
        items.append(f"repo-repair private rows={get_path(repo_repair, ['summary', 'code_lm_row_count'], 'unknown')}")
    if edge_gate:
        items.append(f"edge-obligation gate pass_rate={get_path(edge_gate, ['summary', 'private_pass_rate'], 'unknown')} ready={edge_gate.get('ready_for_public_calibration')}")
    if teacher:
        items.append(f"teacher architecture stages={get_path(teacher, ['summary', 'executed_stage_count'], 'unknown')}")
    if a_plus:
        items.append(f"A+ operating scorecard={get_path(a_plus, ['summary', 'overall_grade'], 'unknown')} score={get_path(a_plus, ['summary', 'overall_score'], 'unknown')}")
    return items[:12]


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# Theseus Hive Morning Report",
        "",
        f"- State: `{payload.get('trigger_state')}`",
        f"- Improvement events: `{summary.get('improvement_events')}`",
        f"- No-progress/failure events: `{summary.get('no_progress_or_failure_events')}`",
        f"- Ready tasks: `{summary.get('ready_task_count')}`",
        f"- A+ grade: `{summary.get('a_plus_grade')}`",
        f"- Conversation accuracy: `{summary.get('conversation_accuracy')}`",
        f"- Tool-use cases: `{summary.get('tool_use_cases')}`",
        f"- Edge obligation ready: `{summary.get('edge_obligation_ready')}`",
        f"- Repo-repair rows: `{summary.get('repo_repair_rows')}`",
        "",
        "## What Improved",
        "",
    ]
    for row in payload.get("what_improved", []) or []:
        lines.append(f"- `{row.get('kind')}` x{row.get('count')}")
    if not payload.get("what_improved"):
        lines.append("- No confirmed improvement contract signal yet.")
    lines.extend(["", "## What Failed", ""])
    for row in payload.get("what_failed", []) or []:
        lines.append(f"- `{row.get('cluster')}` x{row.get('count')}")
    if not payload.get("what_failed"):
        lines.append("- No new residual failures captured.")
    lines.extend(["", "## What Theseus Learned", ""])
    for item in payload.get("what_theseus_learned", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Needs Approval", ""])
    for item in payload.get("needs_approval", []) or []:
        lines.append(f"- {item}")
    if not payload.get("needs_approval"):
        lines.append("- Nothing urgent.")
    lines.extend(["", "## What Runs Next", ""])
    for row in payload.get("what_runs_next", []) or []:
        lines.append(f"- `{row.get('priority')}` {row.get('title')} ({row.get('status')})")
    if not payload.get("what_runs_next"):
        lines.append("- No ready board tasks.")
    lines.append("")
    return "\n".join(lines)


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def best_report(*paths: Path) -> dict[str, Any]:
    for path in paths:
        report = read_json(path, {})
        if report:
            return report
    return {}


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                out.append(item)
        return out
    except OSError:
        return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
