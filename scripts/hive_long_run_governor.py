"""Long-run governor for unattended Theseus Hive learning.

This report is the operator-readable guard layer for multi-hour unattended
runs. It answers, on one page, whether the hive can keep going, what each node
is doing, what improved, what failed, whether artifacts synced, whether the
teacher was used, what got demoted, and what should happen next.

It is diagnostic only: no training, no teacher calls, no external inference.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "hive_long_run_governor.json"
DEFAULT_MARKDOWN = REPORTS / "hive_long_run_governor.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--stale-seconds", type=int, default=1800)
    args = parser.parse_args()

    payload = build_report(stale_seconds=max(60, int(args.stale_seconds)))
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 2 if payload["operations_state"] == "RED" else 0


def build_report(*, stale_seconds: int) -> dict[str, Any]:
    registry = read_json(REPORTS / "hive_node_registry.json", {})
    utilization = read_json(REPORTS / "hive_utilization_manager.json", {})
    board = read_json(REPORTS / "hive_work_board.json", {})
    board_exec = read_json(REPORTS / "hive_work_board_executor.json", {})
    training = read_json(REPORTS / "hive_training_orchestrator.json", {})
    morning = read_json(REPORTS / "hive_morning_report.json", {})
    overnight = read_json(REPORTS / "hive_overnight_proof.json", {})
    artifact_sync = read_json(REPORTS / "hive_artifact_sync.json", {})
    artifact_merge = read_json(REPORTS / "hive_artifact_merge_summary.json", {})
    broad = read_json(REPORTS / "broad_transfer_matrix.json", {})
    resource = read_json(REPORTS / "resource_governor.json", {})
    performance = read_json(REPORTS / "performance_optimizer.json", {})
    readiness = read_json(REPORTS / "hive_fleet_readiness.json", {})
    version = read_json(REPORTS / "hive_version_convergence.json", {})
    teacher = read_json(REPORTS / "hive_teacher_auto_escalation.json", read_json(REPORTS / "teacher_architect_experiment_runner.json", {}))
    conversation = read_json(REPORTS / "high_transfer_multi_turn_conversation.json", read_json(REPORTS / "multi_turn_conversation_benchmark.json", {}))
    repo_repair = read_json(REPORTS / "high_transfer_repo_repair_learner.json", read_json(REPORTS / "viea_repo_repair_learner.json", {}))
    no_progress = read_jsonl(REPORTS / "hive_no_progress_families.jsonl")[-80:]
    improvements = read_jsonl(REPORTS / "hive_unattended_improvement_ledger.jsonl")[-120:]
    teacher_ledger = read_jsonl(REPORTS / "hive_teacher_auto_escalation_ledger.jsonl")[-40:]

    services = {
        "dashboard_local": http_status("http://127.0.0.1:8787/api/health", timeout=2),
        "hive_local": http_status("http://127.0.0.1:8791/api/hive/health", timeout=2),
        "dashboard_lan": lan_dashboard_status(registry),
    }
    nodes = node_rows(registry, utilization, board_exec, training)
    gates = build_gates(
        registry=registry,
        utilization=utilization,
        board=board,
        board_exec=board_exec,
        morning=morning,
        overnight=overnight,
        artifact_sync=artifact_sync,
        artifact_merge=artifact_merge,
        broad=broad,
        resource=resource,
        performance=performance,
        readiness=readiness,
        version=version,
        services=services,
        stale_seconds=stale_seconds,
    )
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    soft_failed = [row for row in gates if row["severity"] == "soft" and not row["passed"]]
    operations_state = "RED" if hard_failed else "YELLOW" if soft_failed else "GREEN"
    cards_below_floor = get_path(broad, ["summary", "cards_below_floor"], []) or []
    learning_state = "YELLOW" if cards_below_floor else "GREEN"
    trigger_state = "RED" if operations_state == "RED" else "YELLOW" if operations_state == "YELLOW" or learning_state == "YELLOW" else "GREEN"

    improved_rows = [row for row in improvements if get_path(row, ["improvement_contract", "passed"], False)]
    failed_rows = [row for row in improvements if not get_path(row, ["improvement_contract", "passed"], False)]
    signal_counts = Counter(
        kind
        for row in improved_rows
        for kind in (get_path(row, ["improvement_contract", "signal_kinds"], []) or [])
    )
    residual_counts = Counter(str(get_path(row, ["improvement_contract", "residual_cluster"], "unknown")) for row in failed_rows)
    demotions = [
        row for row in no_progress if isinstance(row, dict) and str(row.get("action") or "") in {"demoted", "blocked"}
    ]

    payload = {
        "policy": "project_theseus_hive_long_run_governor_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "operations_state": operations_state,
        "learning_state": learning_state,
        "ready_for_soak": not hard_failed,
        "summary": {
            "trusted_nodes": get_path(registry, ["summary", "trusted_node_count"], 0),
            "training_eligible_nodes": get_path(registry, ["summary", "training_eligible_node_count"], 0),
            "ready_board_tasks": get_path(board_exec, ["summary", "ready_tasks"], get_path(board, ["summary", "ready_or_active"], None)),
            "executed_board_tasks_last_run": get_path(board_exec, ["summary", "executed_tasks"], 0),
            "artifact_sync_ok": bool(artifact_sync.get("ok")),
            "artifact_fetched_count": int(artifact_sync.get("fetched_count") or 0),
            "artifact_promoted_count": int(artifact_merge.get("promoted_count") or len(artifact_merge.get("promoted") or [])),
            "improvement_events": len(improved_rows),
            "no_progress_or_failure_events": len(failed_rows),
            "teacher_stage_count": get_path(teacher, ["summary", "executed_stage_count"], 0),
            "teacher_escalations_queued": len(teacher_ledger),
            "broad_public_pass_rate": get_path(broad, ["summary", "real_public_pass_rate"], get_path(broad, ["summary", "aggregate_pass_rate"], None)),
            "cards_below_floor": cards_below_floor,
            "conversation_accuracy": get_path(conversation, ["summary", "accuracy"], None),
            "repo_repair_rows": get_path(repo_repair, ["summary", "code_lm_row_count"], None),
        },
        "nodes": nodes,
        "services": services,
        "gates": gates,
        "what_improved": [{"kind": kind, "count": count} for kind, count in signal_counts.most_common()],
        "what_failed": [{"cluster": cluster, "count": count} for cluster, count in residual_counts.most_common(10)],
        "task_family_demotions": demotions[-10:],
        "teacher_use": {
            "latest_report": report_ref(teacher),
            "executed_stage_count": get_path(teacher, ["summary", "executed_stage_count"], 0),
            "selected_experiments": get_path(teacher, ["summary", "selected_experiments"], 0),
            "recent_escalations": teacher_ledger[-5:],
        },
        "next_actions": next_actions(
            gates=gates,
            broad=broad,
            board_exec=board_exec,
            services=services,
            demotions=demotions,
            conversation=conversation,
            repo_repair=repo_repair,
        ),
        "score_semantics": "long-run operations governor; diagnostic, not promotion evidence",
        "external_inference_calls": 0,
    }
    return payload


def build_gates(
    *,
    registry: dict[str, Any],
    utilization: dict[str, Any],
    board: dict[str, Any],
    board_exec: dict[str, Any],
    morning: dict[str, Any],
    overnight: dict[str, Any],
    artifact_sync: dict[str, Any],
    artifact_merge: dict[str, Any],
    broad: dict[str, Any],
    resource: dict[str, Any],
    performance: dict[str, Any],
    readiness: dict[str, Any],
    version: dict[str, Any],
    services: dict[str, Any],
    stale_seconds: int,
) -> list[dict[str, Any]]:
    artifact_promoted = int(artifact_merge.get("promoted_count") or len(artifact_merge.get("promoted") or []))
    overnight_unfed = int(get_path(overnight, ["summary", "unfed_node_count"], 0) or 0)
    ready_tasks = int(get_path(board_exec, ["summary", "ready_tasks"], get_path(board, ["summary", "ready_or_active"], 0)) or 0)
    public_leaks = int(get_path(broad, ["summary", "no_cheat_violation_count"], 0) or 0)
    board_stale = report_stale(REPORTS / "hive_work_board_executor.json", stale_seconds)
    util_stale = report_stale(REPORTS / "hive_utilization_manager.json", stale_seconds)
    return [
        gate("node_registry_trusted_nodes_present", int(get_path(registry, ["summary", "trusted_node_count"], 0) or 0) > 0, get_path(registry, ["summary"], {}), "hard"),
        gate("fleet_readiness_not_red", readiness.get("trigger_state") != "RED", readiness.get("trigger_state"), "hard"),
        gate("version_convergence_not_red", version.get("trigger_state") != "RED", version.get("trigger_state"), "hard"),
        gate("resource_governor_not_red", resource.get("trigger_state") != "RED", resource.get("trigger_state"), "hard"),
        gate("performance_optimizer_not_red", performance.get("trigger_state") != "RED", performance.get("trigger_state"), "hard"),
        gate("dashboard_api_responsive", bool(get_path(services, ["dashboard_local", "ok"], False)), services.get("dashboard_local"), "soft"),
        gate("hive_api_responsive", bool(get_path(services, ["hive_local", "ok"], False)), services.get("hive_local"), "hard"),
        gate("dashboard_lan_reachable_or_not_advertised", bool(get_path(services, ["dashboard_lan", "ok"], True)), services.get("dashboard_lan"), "soft"),
        gate("artifact_sync_returning_remote_evidence", bool(artifact_sync.get("ok")) and int(artifact_sync.get("fetched_count") or 0) > 0 and artifact_promoted > 0, {"ok": artifact_sync.get("ok"), "fetched": artifact_sync.get("fetched_count"), "promoted": artifact_promoted}, "hard"),
        gate("trusted_capable_nodes_fed", overnight_unfed == 0 and (int(get_path(utilization, ["summary", "planned_actions"], 0) or 0) > 0 or int(get_path(utilization, ["summary", "executed_actions"], 0) or 0) > 0), {"unfed": overnight_unfed, "utilization": utilization.get("summary")}, "hard"),
        gate("work_board_has_ready_or_recent_work", ready_tasks > 0 or int(get_path(board_exec, ["summary", "executed_tasks"], 0) or 0) > 0, {"ready_tasks": ready_tasks, "executed_last_run": get_path(board_exec, ["summary", "executed_tasks"], 0)}, "soft"),
        gate("work_board_report_fresh", not board_stale, {"stale": board_stale, "path": "reports/hive_work_board_executor.json"}, "soft"),
        gate("utilization_report_fresh", not util_stale, {"stale": util_stale, "path": "reports/hive_utilization_manager.json"}, "soft"),
        gate("progress_or_residual_signal_present", int(get_path(morning, ["summary", "improvement_events"], 0) or 0) > 0 or int(get_path(morning, ["summary", "no_progress_or_failure_events"], 0) or 0) > 0, morning.get("summary"), "hard"),
        gate("public_data_leak_checks_clean", public_leaks == 0, {"no_cheat_violation_count": public_leaks}, "hard"),
    ]


def node_rows(registry: dict[str, Any], utilization: dict[str, Any], board_exec: dict[str, Any], training: dict[str, Any]) -> list[dict[str, Any]]:
    works = work_by_node(utilization, board_exec, training)
    out = []
    for node in registry.get("nodes", []) if isinstance(registry.get("nodes"), list) else []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "")
        node_name = str(node.get("node_name") or "")
        idle = [
            {
                "slot_id": slot.get("slot_id"),
                "slot_type": slot.get("slot_type"),
                "capacity": slot.get("capacity"),
                "task_kinds": slot.get("task_kinds"),
            }
            for slot in node.get("slots", [])
            if isinstance(slot, dict) and slot.get("available")
        ]
        work_items = works.get(node_id) or works.get(node_name) or []
        out.append(
            {
                "node_id": node_id,
                "node_name": node_name,
                "is_local": bool(node.get("is_local")),
                "api_url": node.get("api_url"),
                "training_allowed": bool(node.get("training_allowed")),
                "light_task_allowed": bool(node.get("light_task_allowed")),
                "accelerators": node.get("accelerator_ids") or [],
                "idle_slots": idle,
                "blockers": {
                    "resource": node.get("resource_blockers") or [],
                    "training": node.get("training_blockers") or [],
                },
                "current_or_planned_work": work_items[:8] or [{"status": "idle_or_not_reported", "source": "latest_reports"}],
            }
        )
    return out


def work_by_node(utilization: dict[str, Any], board_exec: dict[str, Any], training: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in utilization.get("actions", []) if isinstance(utilization.get("actions"), list) else []:
        if not isinstance(row, dict):
            continue
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        node_ids = [str(evidence.get("node_id") or "")]
        node_ids.extend(str(item) for item in evidence.get("planned_node_ids", []) if item)
        for node_id in [item for item in node_ids if item]:
            out.setdefault(node_id, []).append({"source": "utilization_manager", "status": "planned", "kind": row.get("kind"), "title": row.get("title")})
    for row in utilization.get("execution", []) if isinstance(utilization.get("execution"), list) else []:
        if not isinstance(row, dict):
            continue
        node_id = str(get_path(row, ["evidence", "node_id"], "") or get_path(row, ["result", "task", "payload", "target_node_id"], ""))
        if node_id:
            out.setdefault(node_id, []).append({"source": "utilization_manager", "status": row.get("status"), "kind": row.get("kind"), "title": row.get("title")})
        for job in get_path(row, ["result", "payload", "plan", "jobs"], []) or []:
            if not isinstance(job, dict):
                continue
            add_training_job_work(out, job, source="training_orchestrator_via_utilization")
    for job in get_path(training, ["plan", "jobs"], []) or []:
        if isinstance(job, dict):
            add_training_job_work(out, job, source="training_orchestrator")
    for section, status in (("selected", "selected"), ("results", "executed")):
        for row in board_exec.get(section, []) if isinstance(board_exec.get(section), list) else []:
            if not isinstance(row, dict):
                continue
            assignment = row.get("assignment") if isinstance(row.get("assignment"), dict) else {}
            for key in ("node_id", "node_name"):
                value = str(assignment.get(key) or row.get(key) or "")
                if value:
                    out.setdefault(value, []).append({"source": "hive_work_board", "status": row.get("status") or status, "kind": row.get("kind"), "title": row.get("title")})
    return out


def add_training_job_work(out: dict[str, list[dict[str, Any]]], job: dict[str, Any], *, source: str) -> None:
    node_id = str(job.get("node_id") or get_path(job, ["payload", "target_node_id"], ""))
    node_name = str(job.get("node_name") or get_path(job, ["payload", "target_node_name"], ""))
    item = {
        "source": source,
        "status": job.get("status") or "planned",
        "kind": job.get("task_kind") or get_path(job, ["payload", "job_family"], ""),
        "title": job.get("display_name") or job.get("arm_id") or job.get("task_kind"),
        "target": job.get("target"),
        "lease_expires_utc": get_path(job, ["lease", "lease_expires_utc"], get_path(job, ["payload", "orchestration", "lease_expires_utc"], None)),
    }
    for key in (node_id, node_name):
        if key:
            out.setdefault(key, []).append(item)


def next_actions(
    *,
    gates: list[dict[str, Any]],
    broad: dict[str, Any],
    board_exec: dict[str, Any],
    services: dict[str, Any],
    demotions: list[dict[str, Any]],
    conversation: dict[str, Any],
    repo_repair: dict[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    failed = {row["gate"]: row for row in gates if not row["passed"]}
    if "dashboard_lan_reachable_or_not_advertised" in failed:
        actions.append(next_action("fix_dashboard_lan_binding", "Restart SparkStream dashboard bound to 0.0.0.0.", ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts/start_sparkstream.ps1", "-Restart", "-DashboardHost", "0.0.0.0"]))
    if "artifact_sync_returning_remote_evidence" in failed:
        actions.append(next_action("repair_artifact_sync", "Run artifact sync and merge before trusting remote learning.", ["python", "scripts/hive_artifact_sync.py", "--out", "reports/hive_artifact_sync.json", "--limit", "200"]))
    if "trusted_capable_nodes_fed" in failed:
        actions.append(next_action("feed_idle_hive_slots", "Run utilization sweep so reachable accelerator slots stay fed.", ["python", "scripts/hive_utilization_manager.py", "sweep", "--execute", "--max-new-jobs", "2", "--out", "reports/hive_utilization_manager.json"]))
    ready = int(get_path(board_exec, ["summary", "ready_tasks"], 0) or 0)
    if ready > 0 and int(get_path(board_exec, ["summary", "executed_tasks"], 0) or 0) == 0:
        actions.append(next_action("execute_board_step", "Let the board run one bounded real learning/action task.", ["python", "scripts/hive_work_board_executor.py", "--execute", "--resume", "--max-tasks", "1", "--max-steps", "1", "--timeout-seconds", "21600"]))
    cards = get_path(broad, ["summary", "cards_below_floor"], []) or []
    if cards:
        actions.append(next_action("broad_transfer_pressure", "Prioritize transferable semantic concepts before benchmark-specific grinding.", ["python", "scripts/high_transfer_curriculum_scheduler.py", "--out", "reports/high_transfer_curriculum_scheduler.json", "--markdown-out", "reports/high_transfer_curriculum_scheduler.md", "--tasks-out", "reports/high_transfer_curriculum_tasks.jsonl"]))
    if not conversation:
        actions.append(next_action("conversation_lane", "Refresh multi-turn conversation/personality retention lane.", ["python", "scripts/multi_turn_conversation_benchmark.py", "--out", "reports/high_transfer_multi_turn_conversation.json", "--markdown-out", "reports/high_transfer_multi_turn_conversation.md"]))
    if get_path(repo_repair, ["summary", "code_lm_row_count"], 0) in (None, 0):
        actions.append(next_action("repo_repair_lane", "Refresh private repo-repair traces for agentic programming transfer.", ["python", "scripts/viea_repo_repair_learner.py", "--max-tasks", "96", "--out", "reports/high_transfer_repo_repair_learner.json"]))
    if demotions:
        actions.append(next_action("teacher_architect_after_repeated_residual", "Repeated no-progress family exists; queue architecture diagnosis, not answers.", ["python", "scripts/teacher_architect_experiment_runner.py", "--execute", "--allow-teacher", "--max-experiments", "1", "--max-steps", "1"]))
    return actions[:8]


def next_action(kind: str, title: str, command: list[str]) -> dict[str, Any]:
    return {"kind": kind, "title": title, "command": command}


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def lan_dashboard_status(registry: dict[str, Any]) -> dict[str, Any]:
    local = next((node for node in registry.get("nodes", []) if isinstance(node, dict) and node.get("is_local")), {})
    url = str(local.get("dashboard_url") or "")
    if not url or "127.0.0.1" in url or "localhost" in url:
        return {"ok": True, "url": url, "reason": "no_lan_dashboard_advertised"}
    return http_status(url.rstrip("/") + "/api/health", timeout=2)


def http_status(url: str, *, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read(256 * 1024).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        return {"ok": True, "url": url, "runtime_ms": int((time.perf_counter() - started) * 1000), "policy": payload.get("policy"), "created_utc": payload.get("created_utc")}
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "url": url, "runtime_ms": int((time.perf_counter() - started) * 1000), "error": str(exc)}


def report_stale(path: Path, stale_seconds: int) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) > stale_seconds
    except OSError:
        return True


def report_ref(report: dict[str, Any]) -> dict[str, Any]:
    return {"policy": report.get("policy"), "trigger_state": report.get("trigger_state"), "created_utc": report.get("created_utc")}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Hive Long-Run Governor",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- operations_state: `{payload.get('operations_state')}`",
        f"- learning_state: `{payload.get('learning_state')}`",
        f"- ready_for_soak: `{payload.get('ready_for_soak')}`",
        f"- trusted_nodes: `{summary.get('trusted_nodes')}`",
        f"- training_eligible_nodes: `{summary.get('training_eligible_nodes')}`",
        f"- ready_board_tasks: `{summary.get('ready_board_tasks')}`",
        f"- artifact_sync_ok: `{summary.get('artifact_sync_ok')}` fetched=`{summary.get('artifact_fetched_count')}` promoted=`{summary.get('artifact_promoted_count')}`",
        f"- broad_public_pass_rate: `{summary.get('broad_public_pass_rate')}`",
        f"- cards_below_floor: `{summary.get('cards_below_floor')}`",
        "",
        "## Nodes",
        "",
    ]
    for node in payload.get("nodes", []) or []:
        work = "; ".join(f"{item.get('status')}:{item.get('kind') or item.get('source')}" for item in (node.get("current_or_planned_work") or [])[:3])
        lines.append(f"- `{node.get('node_name')}` training={node.get('training_allowed')} light={node.get('light_task_allowed')} accelerators={node.get('accelerators')} work={work}")
    lines.extend(["", "## Gates", ""])
    for row in payload.get("gates", []) or []:
        mark = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {mark} `{row.get('gate')}` ({row.get('severity')})")
    lines.extend(["", "## What Improved", ""])
    if payload.get("what_improved"):
        for row in payload.get("what_improved") or []:
            lines.append(f"- `{row.get('kind')}` x{row.get('count')}")
    else:
        lines.append("- No confirmed improvement signal in the recent ledger window.")
    lines.extend(["", "## What Failed", ""])
    if payload.get("what_failed"):
        for row in payload.get("what_failed") or []:
            lines.append(f"- `{row.get('cluster')}` x{row.get('count')}")
    else:
        lines.append("- No recent failure residuals in the ledger window.")
    lines.extend(["", "## Teacher", ""])
    teacher = payload.get("teacher_use") or {}
    lines.append(f"- executed_stage_count: `{teacher.get('executed_stage_count')}`")
    lines.append(f"- recent_escalations: `{len(teacher.get('recent_escalations') or [])}`")
    lines.extend(["", "## Next Actions", ""])
    if payload.get("next_actions"):
        for row in payload.get("next_actions") or []:
            lines.append(f"- `{row.get('kind')}`: {row.get('title')}")
    else:
        lines.append("- Keep Vacation Mode running; no operator correction needed from this snapshot.")
    lines.append("")
    return "\n".join(lines)


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def get_path(value: Any, path: list[Any], default: Any = None) -> Any:
    cur = value
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return value if isinstance(value, dict) else default


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


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
