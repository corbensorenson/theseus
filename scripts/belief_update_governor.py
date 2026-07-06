"""Govern observation-to-belief updates against the personality reality contract."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "belief_update_policy.json"

import sys

sys.path.insert(0, str(ROOT / "scripts"))
import personality_context_builder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--observation", default="")
    parser.add_argument("--inferred-belief", default="")
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--source", default="manual")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    out = ROOT / (args.out or policy.get("report") or "reports/belief_update_governance.json")
    if args.status_only or not args.observation.strip():
        report = status_report(policy)
        write_json(out, report)
        print(json.dumps(report, indent=2))
        return 0

    report = evaluate_update(
        policy,
        observation=args.observation,
        inferred_belief=args.inferred_belief,
        confidence=args.confidence,
        source=args.source,
    )
    append_jsonl(ROOT / str(policy.get("ledger") or "reports/belief_update_ledger.jsonl"), report["update"])
    status = status_report(policy)
    status["last_update"] = report["update"]
    status["status"] = "evaluated"
    write_json(out, status)
    print(json.dumps(status, indent=2))
    return 0 if report["update"].get("decision") != policy.get("quarantine_status", "quarantined") else 2


def evaluate_update(
    policy: dict[str, Any],
    *,
    observation: str,
    inferred_belief: str,
    confidence: float,
    source: str,
) -> dict[str, Any]:
    context = personality_context_builder.build_context(prompt=f"{observation}\n{inferred_belief}", task="belief_update")
    text = f"{observation}\n{inferred_belief}"
    conflicts = pattern_hits(text, policy.get("quarantine_patterns", []))
    review_flags = pattern_hits(text, policy.get("review_patterns", []))
    min_accept = float(policy.get("minimum_accept_confidence") or 0.72)
    if conflicts:
        decision = str(policy.get("quarantine_status") or "quarantined")
    elif confidence < min_accept or review_flags:
        decision = str(policy.get("review_status") or "needs_review")
    else:
        decision = str(policy.get("accepted_status") or "accepted")
    update = {
        "policy": "sparkstream_belief_update_v0",
        "created_utc": now(),
        "source": source,
        "observation": observation,
        "inferred_belief": inferred_belief,
        "confidence": confidence,
        "decision": decision,
        "conflict_with_inherited_core": bool(conflicts),
        "conflicts": conflicts,
        "review_flags": review_flags,
        "personality_context": compact_personality_context(context),
        "external_inference_calls": 0,
    }
    return {"update": update}


def status_report(policy: dict[str, Any]) -> dict[str, Any]:
    ledger_path = ROOT / str(policy.get("ledger") or "reports/belief_update_ledger.jsonl")
    rows = read_jsonl_tail(ledger_path, 2000)
    counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision") or "unknown") if isinstance(row, dict) else "unknown"
        counts[decision] = counts.get(decision, 0) + 1
    context = personality_context_builder.build_context(prompt="belief update governance status", task="belief_update_status")
    return {
        "policy": "sparkstream_belief_update_governance_v0",
        "created_utc": now(),
        "status": "ready",
        "ledger": rel(ledger_path),
        "summary": {
            "ledger_entries": len(rows),
            "accepted": counts.get(str(policy.get("accepted_status") or "accepted"), 0),
            "needs_review": counts.get(str(policy.get("review_status") or "needs_review"), 0),
            "quarantined": counts.get(str(policy.get("quarantine_status") or "quarantined"), 0),
            "personality_context_status": context.get("status"),
        },
        "personality_context": compact_personality_context(context),
        "policy_thresholds": {
            "minimum_accept_confidence": policy.get("minimum_accept_confidence"),
            "review_confidence_band": policy.get("review_confidence_band"),
        },
        "external_inference_calls": 0,
    }


def pattern_hits(text: str, patterns: Any) -> list[dict[str, str]]:
    hits = []
    for item in patterns if isinstance(patterns, list) else []:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern") or "")
        if not pattern:
            continue
        try:
            matched = re.search(pattern, text)
        except re.error:
            continue
        if matched:
            hits.append({"reason": str(item.get("reason") or "pattern_match"), "pattern": pattern})
    return hits


def compact_personality_context(context: dict[str, Any]) -> dict[str, Any]:
    ctx = context.get("context") if isinstance(context.get("context"), dict) else {}
    return {
        "status": context.get("status"),
        "source_report": context.get("source_report"),
        "summary": context.get("summary", {}),
        "compact_core": ctx.get("compact_core", ""),
        "hard_safety_invariants": personality_context_builder.get_path(ctx, ["reality_contract", "hard_safety_invariants"], [])[:5],
        "anti_drift_rules": (ctx.get("anti_drift_rules") or [])[:5],
    }


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines[-limit:]:
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
