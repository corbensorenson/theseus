"""Audit sparse teacher budget without making a teacher call.

The teacher should be available for real architecture walls, but it should not
become a hidden answer source or an unbounded dependency. This report separates
budget availability from actual teacher use.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import teacher_oracle  # noqa: E402


DEFAULT_REASONS = [
    "architecture_wall",
    "benchmark_frontier_design",
    "frontier_exhausted",
    "residual_conflict",
    "promotion_gate_blocked",
    "safety_or_governance_uncertainty",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/teacher_policy.json")
    parser.add_argument("--out", default="reports/teacher_budget_audit.json")
    parser.add_argument("--markdown-out", default="reports/teacher_budget_audit.md")
    parser.add_argument("--reasons", default=",".join(DEFAULT_REASONS))
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    reasons = [item.strip() for item in args.reasons.split(",") if item.strip()]
    calls = read_jsonl(ROOT / policy.get("log_path", "reports/teacher_calls.jsonl"))
    today = datetime.now(timezone.utc).date().isoformat()
    completed_today = [row for row in calls if row.get("status") == "completed" and str(row.get("created_utc", "")).startswith(today)]
    completed_architecture_today = [
        row for row in completed_today if row.get("reason_for_call") == "architecture_wall"
    ]
    budget = policy.get("budget") or {}
    reason_decisions = {}
    for reason in reasons:
        reason_decisions[reason] = {
            "budget": teacher_oracle.call_budget_decision(policy, reason),
            "local_wall_evidence": teacher_oracle.local_wall_evidence_decision(policy, reason, []),
        }
    architecture = reason_decisions.get("architecture_wall", {})
    architecture_ready = bool(
        get_path(architecture, ["budget", "allowed"], False)
        and get_path(architecture, ["local_wall_evidence", "allowed"], False)
    )
    payload = {
        "policy": "project_theseus_teacher_budget_audit_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if architecture_ready else "YELLOW",
        "provider": policy.get("provider"),
        "model": policy.get("model"),
        "reasoning_effort": policy.get("reasoning_effort"),
        "proposal_only_no_distillation": bool(budget.get("proposal_only_no_distillation", True)),
        "subscription_budget_hint": budget.get("subscription_budget_hint", {}),
        "completed_today": len(completed_today),
        "completed_architecture_today": len(completed_architecture_today),
        "default_max_calls_per_day": budget.get("max_calls_per_day"),
        "architecture_wall_override": (budget.get("reason_overrides") or {}).get("architecture_wall", {}),
        "reason_decisions": reason_decisions,
        "latest_budget_status": read_json(ROOT / policy.get("budget_block_path", "reports/teacher_budget_last.json")),
        "score_semantics": "budget audit only; it does not spend teacher calls and is not learning evidence",
    }
    write_json(ROOT / args.out, payload)
    write_text(ROOT / args.markdown_out, render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] == "GREEN" else 1


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Teacher Budget Audit",
        "",
        f"- trigger_state: {payload.get('trigger_state')}",
        f"- provider/model: {payload.get('provider')} / {payload.get('model')}",
        f"- completed_today: {payload.get('completed_today')}",
        f"- completed_architecture_today: {payload.get('completed_architecture_today')}",
        f"- proposal_only_no_distillation: {payload.get('proposal_only_no_distillation')}",
        "",
        "## Reason Decisions",
    ]
    for reason, decision in (payload.get("reason_decisions") or {}).items():
        budget = decision.get("budget") or {}
        evidence = decision.get("local_wall_evidence") or {}
        lines.append(
            f"- {reason}: budget_allowed={budget.get('allowed')} "
            f"budget_reason={budget.get('reason')} wall_evidence={evidence.get('allowed')}"
        )
    return "\n".join(lines) + "\n"


def get_path(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
