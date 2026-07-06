"""Summarize whether guarded teacher self-edits are actually proving out."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="reports/teacher_self_edit_traces.jsonl")
    parser.add_argument("--last", default="reports/teacher_self_edit_last.json")
    parser.add_argument("--out", default="reports/teacher_self_edit_proof.json")
    args = parser.parse_args()

    traces = read_jsonl(ROOT / args.trace)
    last = read_json(ROOT / args.last)
    report = build_report(traces, last)
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") != "blocked" else 1


def build_report(traces: list[dict[str, Any]], last: dict[str, Any]) -> dict[str, Any]:
    recent = traces[-25:]
    successes = [row for row in recent if row.get("success")]
    failures = [row for row in recent if row and not row.get("success")]
    total = len(recent)
    success_rate = len(successes) / total if total else 0.0
    latest_status = str(last.get("status") or (recent[-1].get("status") if recent else "missing"))
    blocked = latest_status.startswith("blocked") or latest_status in {
        "teacher_apply_failed",
        "checks_failed_branch_left_for_review",
        "failed_create_branch",
    }
    report = {
        "policy": "project_theseus_teacher_self_edit_proof_v0",
        "created_utc": now(),
        "status": "blocked" if blocked else ("proving" if successes else "needs_real_success_cycles"),
        "summary": {
            "recent_trace_count": total,
            "recent_successes": len(successes),
            "recent_failures": len(failures),
            "success_rate": round(success_rate, 4),
            "latest_status": latest_status,
            "latest_branch": last.get("branch") or (recent[-1].get("branch") if recent else None),
            "latest_changed_files": last.get("changed_files", []),
        },
        "proof_gates": [
            gate("trace_written", bool(traces), f"traces={len(traces)}"),
            gate("latest_not_blocked", not blocked, f"latest_status={latest_status}"),
            gate("has_successful_teacher_patch", bool(successes), f"successes={len(successes)}"),
            gate("success_rate_nonzero", success_rate > 0.0, f"success_rate={success_rate:.3f}"),
        ],
        "open_risks": open_risks(successes, failures, latest_status),
        "next_actions": next_actions(successes, failures, latest_status),
        "external_inference_calls": 0,
    }
    return report


def open_risks(successes: list[dict[str, Any]], failures: list[dict[str, Any]], latest_status: str) -> list[str]:
    risks: list[str] = []
    if not successes:
        risks.append("No successful guarded teacher patch has completed local checks yet.")
    if failures:
        risks.append("Recent teacher patch failures should become repair traces for the local self-debugging arm.")
    if latest_status == "blocked_dirty_worktree":
        risks.append("Dirty worktree blocks the guarded teacher lane.")
    if latest_status == "blocked_attd_red":
        risks.append("ATTD debt is blocking architecture/self-edit changes until maintenance packets are consumed.")
    return risks


def next_actions(successes: list[dict[str, Any]], failures: list[dict[str, Any]], latest_status: str) -> list[str]:
    actions: list[str] = []
    if latest_status == "blocked_dirty_worktree":
        actions.append("Commit or isolate existing local changes before teacher self-edit.")
    if not successes:
        actions.append("Run one bounded teacher maintenance or architecture-wall edit and verify checks.")
    if failures:
        actions.append("Distill failed teacher traces into the self-debugging and code-repair transfer evals.")
    if successes:
        actions.append("Compare successful teacher branch benchmarks, then merge only if regression gates hold.")
    return actions


def gate(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
