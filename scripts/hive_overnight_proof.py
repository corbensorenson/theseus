"""Overnight unattended Hive proof gate.

This report answers the operator question: did the unattended hive actually keep
trusted devices fed and compound evidence without cheating or dropping remote
artifacts?
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
import hive_node_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/hive_policy.json")
    parser.add_argument("--out", default="reports/hive_overnight_proof.json")
    parser.add_argument("--markdown-out", default="reports/hive_overnight_proof.md")
    args = parser.parse_args()

    policy = read_json(resolve(args.policy), {})
    registry = hive_node_registry.build_registry(policy)
    write_json(REPORTS / "hive_node_registry.json", registry)
    report = build_report(registry)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(registry: dict[str, Any]) -> dict[str, Any]:
    utilization = read_json(REPORTS / "hive_utilization_manager.json", {})
    artifact_sync = read_json(REPORTS / "hive_artifact_sync.json", {})
    merge = read_json(REPORTS / "hive_artifact_merge_summary.json", {})
    training = read_json(REPORTS / "hive_training_orchestrator.json", {})
    morning = read_json(REPORTS / "hive_morning_report.json", {})
    broad = read_json(REPORTS / "broad_transfer_matrix.json", {})
    no_progress = read_jsonl(REPORTS / "hive_no_progress_families.jsonl")[-50:]
    board = read_json(REPORTS / "hive_work_board.json", {})

    fed = fed_nodes(registry, utilization, training)
    artifacts = artifact_gate(artifact_sync, merge)
    improvement = improvement_gate(morning)
    leak = public_leak_gate(broad)
    stuck = stuck_family_gate(no_progress, board)
    gates = [fed, artifacts, improvement, leak, stuck]
    hard_failed = [gate for gate in gates if gate["severity"] == "hard" and not gate["passed"]]
    soft_failed = [gate for gate in gates if gate["severity"] == "soft" and not gate["passed"]]
    return {
        "policy": "project_theseus_hive_overnight_proof_v1",
        "created_utc": now(),
        "trigger_state": "RED" if hard_failed else "YELLOW" if soft_failed else "GREEN",
        "summary": {
            "trusted_nodes": get_path(registry, ["summary", "trusted_node_count"], 0),
            "training_eligible_nodes": get_path(registry, ["summary", "training_eligible_node_count"], 0),
            "unfed_node_count": len(get_path(fed, ["evidence", "unfed_nodes"], [])),
            "artifact_sync_ok": artifacts["passed"],
            "improvement_events": get_path(morning, ["summary", "improvement_events"], 0),
            "no_progress_or_failure_events": get_path(morning, ["summary", "no_progress_or_failure_events"], 0),
            "public_no_cheat_violations": get_path(broad, ["summary", "no_cheat_violation_count"], 0),
        },
        "gates": gates,
        "node_registry": {"report": "reports/hive_node_registry.json", "summary": registry.get("summary", {})},
        "score_semantics": "overnight unattended operations proof; diagnostic, not promotion evidence",
        "external_inference_calls": 0,
    }


def fed_nodes(registry: dict[str, Any], utilization: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    executed_node_ids = {
        str(get_path(row, ["evidence", "node_id"], "") or get_path(row, ["result", "task", "payload", "target_node_id"], ""))
        for row in utilization.get("execution", [])
        if isinstance(row, dict) and row.get("status") == "completed"
    }
    planned_node_ids: set[str] = set()
    for row in utilization.get("actions", []):
        if not isinstance(row, dict):
            continue
        single = str(get_path(row, ["evidence", "node_id"], "") or get_path(row, ["submit", "payload", "target_node_id"], ""))
        if single:
            planned_node_ids.add(single)
        for node_id in get_path(row, ["evidence", "planned_node_ids"], []) or []:
            if node_id:
                planned_node_ids.add(str(node_id))
    queue_rows = read_jsonl(REPORTS / "hive_task_queue.jsonl")[-500:]
    queued_node_ids = {
        str(row.get("target_node_id") or get_path(row, ["payload", "target_node_id"], ""))
        for row in queue_rows
        if isinstance(row, dict) and str(row.get("status") or "queued") in {"queued", "active", "running"}
    }
    training_node_ids = {
        str(row.get("node_id") or get_path(row, ["payload", "target_node_id"], ""))
        for row in get_path(training, ["plan", "jobs"], []) or []
        if isinstance(row, dict) and str(row.get("status") or "ready") in {"ready", "queued", "active", "running", "completed"}
    }
    fed_ids = executed_node_ids | planned_node_ids | queued_node_ids | training_node_ids
    unfed = []
    for node in registry.get("nodes", []) if isinstance(registry.get("nodes"), list) else []:
        if not isinstance(node, dict) or not node.get("light_task_allowed"):
            continue
        node_id = str(node.get("node_id") or "")
        has_idle_slot = any(slot.get("available") for slot in node.get("slots", []) if isinstance(slot, dict))
        if has_idle_slot and node_id not in fed_ids and str(node.get("node_name") or "") not in fed_ids:
            unfed.append({"node_id": node_id, "node_name": node.get("node_name"), "slots": node.get("slots", [])})
    return {
        "gate": "no_idle_capable_node_unfed",
        "passed": not unfed,
        "severity": "hard",
        "evidence": {"fed_node_ids": sorted(item for item in fed_ids if item), "unfed_nodes": unfed},
    }


def artifact_gate(sync: dict[str, Any], merge: dict[str, Any]) -> dict[str, Any]:
    hard_errors = [
        err
        for err in sync.get("errors", [])
        if isinstance(err, dict) and not volatile_artifact_error(err)
    ]
    promoted = int(merge.get("promoted_count") or len(merge.get("promoted") or []))
    fetched = int(sync.get("fetched_count") or len(sync.get("fetched") or []))
    return {
        "gate": "remote_artifacts_returned_and_merged",
        "passed": fetched > 0 and promoted > 0 and not hard_errors,
        "severity": "hard",
        "evidence": {"fetched": fetched, "promoted": promoted, "hard_errors": hard_errors[:5]},
    }


def volatile_artifact_error(err: dict[str, Any]) -> bool:
    path = str(err.get("path") or "").replace("\\", "/").lower()
    return err.get("error") == "sha256_mismatch" and (
        path.endswith("_last.json")
        or path in {"reports/compute_market_status.json", "reports/compute_market_settlement_last.json"}
    )


def improvement_gate(morning: dict[str, Any]) -> dict[str, Any]:
    improvements = int(get_path(morning, ["summary", "improvement_events"], 0) or 0)
    residuals = int(get_path(morning, ["summary", "no_progress_or_failure_events"], 0) or 0)
    return {
        "gate": "improvement_or_residual_signal_present",
        "passed": improvements > 0 or residuals > 0,
        "severity": "hard",
        "evidence": {"improvement_events": improvements, "no_progress_or_failure_events": residuals},
    }


def public_leak_gate(broad: dict[str, Any]) -> dict[str, Any]:
    violations = int(get_path(broad, ["summary", "no_cheat_violation_count"], 0) or 0)
    return {
        "gate": "public_data_leak_checks_clean",
        "passed": violations == 0,
        "severity": "hard",
        "evidence": {"no_cheat_violation_count": violations, "score_semantics": broad.get("score_semantics")},
    }


def stuck_family_gate(no_progress: list[dict[str, Any]], board: dict[str, Any]) -> dict[str, Any]:
    blocked_no_progress = [
        row for row in no_progress if isinstance(row, dict) and str(row.get("action") or "") == "blocked"
    ]
    blocked_tasks = [
        row for row in board.get("tasks", []) if isinstance(row, dict) and str(row.get("blocked_reason") or "") == "no_progress_contract_twice"
    ]
    return {
        "gate": "no_stuck_repeated_task_family",
        "passed": not blocked_no_progress and not blocked_tasks,
        "severity": "soft",
        "evidence": {"blocked_no_progress_families": blocked_no_progress[-5:], "blocked_tasks": blocked_tasks[:5]},
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hive Overnight Proof",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- trusted_nodes: `{get_path(report, ['summary', 'trusted_nodes'], 0)}`",
        f"- training_eligible_nodes: `{get_path(report, ['summary', 'training_eligible_nodes'], 0)}`",
        f"- improvement_events: `{get_path(report, ['summary', 'improvement_events'], 0)}`",
        "",
        "## Gates",
        "",
    ]
    for gate in report.get("gates", []):
        mark = "PASS" if gate.get("passed") else "FAIL"
        lines.append(f"- {mark} `{gate.get('gate')}` ({gate.get('severity')})")
    lines.append("")
    return "\n".join(lines)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
