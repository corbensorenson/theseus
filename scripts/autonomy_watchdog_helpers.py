"""Shared helpers for the SparkStream autonomy watchdog."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    severity: str,
    evidence: str,
    corrective_action: str,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "evidence": evidence,
            "corrective_action": corrective_action,
        }
    )


def action(kind: str, reason: str) -> dict[str, Any]:
    return {"kind": kind, "reason": reason, "applied": False}


def grammar_sucker_integrity(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    structurally_ok = bool(
        report.get("policy") == "project_theseus_grammar_suckers_v0"
        and report.get("trigger_state") in {"GREEN", "YELLOW", "RED"}
    )
    invalid_promotion = int(summary.get("python_invalid_promotion_eligible_count") or 0)
    return {
        "summary": summary,
        "structurally_ok": structurally_ok,
        "invalid_promotion": invalid_promotion,
        "integrity_ok": bool(structurally_ok and invalid_promotion == 0),
        "hard_action_needed": bool((not structurally_ok) or invalid_promotion > 0),
    }


def deterministic_taming_integrity(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    integrity_ok = bool(
        report.get("policy") == "project_theseus_deterministic_taming_stack_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(summary.get("hard_failure_count") or 0) == 0
    )
    return {
        "summary": summary,
        "integrity_ok": integrity_ok,
        "stale_severity": "YELLOW" if integrity_ok else "RED",
    }


def broad_transfer_blockers(summary: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in (
        "cards_below_floor",
        "no_clean_student_evidence_cards",
        "loader_only_cards",
        "coverage_warning_cards",
        "missing_cards",
    ):
        values = summary.get(key) if isinstance(summary, dict) else []
        if isinstance(values, list) and values:
            blockers.append(f"{key}={','.join(str(item) for item in values[:8])}")
    if isinstance(summary, dict) and float(summary.get("real_public_pass_rate") or 0.0) < 0.70:
        blockers.append(f"aggregate_pass_rate={summary.get('real_public_pass_rate')}")
    return blockers


def trailing_daemon_failures(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in reversed(rows):
        event = row.get("event")
        if event == "cycle_failed":
            count += 1
        elif event in {"cycle_complete", "cycle_start", "started"}:
            break
    return count


def recent_daemon_failures(rows: list[dict[str, Any]], *, minutes: int) -> int:
    cutoff = time.time() - max(1, minutes) * 60
    count = 0
    for row in rows:
        if row.get("event") != "cycle_failed":
            continue
        ts = row_time(row)
        if ts >= cutoff:
            count += 1
    return count


def latest_daemon_terminal_event(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        if row.get("event") in {"cycle_complete", "cycle_failed", "stopped", "completed"}:
            return row
    return {}


def selected_work_board_task(report: dict[str, Any]) -> dict[str, Any]:
    selected = report.get("selected") if isinstance(report.get("selected"), list) else []
    for row in selected:
        if isinstance(row, dict):
            return row
    return {}


def task_concept(task: dict[str, Any]) -> str:
    evidence = task.get("evidence") if isinstance(task.get("evidence"), dict) else {}
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    nested_payload = evidence.get("payload") if isinstance(evidence.get("payload"), dict) else {}
    for source in (evidence, payload, nested_payload, task):
        value = source.get("concept") if isinstance(source, dict) else ""
        if value:
            return str(value)
    return ""


def row_time(row: dict[str, Any]) -> float:
    for key in ("created_utc", "updated_utc", "time_utc", "utc"):
        value = row.get(key)
        if value:
            try:
                return datetime.fromisoformat(str(value)).timestamp()
            except ValueError:
                pass
    return 0.0


def trailing_same_frontier_streak(rows: list[dict[str, Any]]) -> int:
    identities: list[str] = []
    for row in rows:
        decision = row.get("decision") or {}
        family = str(decision.get("frontier_family") or "")
        if not family:
            continue
        if family == "rl_local":
            identity = f"rl:{decision.get('rl_frontier_env')}:{decision.get('rl_frontier_seed')}"
        else:
            identity = (
                f"{family}:{decision.get('pressure_card_id') or decision.get('frontier_seed')}:"
                f"{decision.get('frontier_report')}"
            )
        identities.append(identity)
    if not identities:
        return 0
    latest = identities[-1]
    count = 0
    for identity in reversed(identities):
        if identity != latest:
            break
        count += 1
    return count


def teacher_budget_blocks_since_completed(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in reversed(rows):
        if row.get("status") == "completed":
            break
        if row.get("status") == "blocked_by_teacher_budget":
            count += 1
    return count


def seed_from_text(value: str) -> int:
    match = re.search(r"seed(\d+)", value)
    return int(match.group(1)) if match else -1


CONVERSATION_FRONTIER_CARDS = {
    "multi_turn_conversation_benchmark",
    "high_transfer_multi_turn_conversation",
    "high_transfer_multi_turn_conversation_hard",
    "open_conversation_pantry",
}

CONVERSATION_FRONTIER_FAMILIES = {
    "tool_agent",
    "general",
    "conversation",
    "conversation_multiturn",
    "conversation_multiturn_local",
}

PROMOTION_FRONTIER_FAMILIES = {
    "babylm_mutated",
    "coding_local_sandbox",
    "minecraft_rl",
    "drone_rl",
    "web_agent_local",
    "transfer_eval",
}


def candidate_frontier_alignment_required(expected_family: str, expected_card: str, candidate_family: str) -> bool:
    """Only promotion-facing reports should make stale candidate gates RED.

    Candidate promotion is not the active evidence source when the curriculum
    legitimately rotates into conversation/regression lanes. In that case the
    watchdog should still check the live transfer/architecture reports, but a
    stale candidate-promotion artifact should be diagnostic rather than a hard
    blocker.
    """

    expected_family = str(expected_family or "")
    expected_card = str(expected_card or "")
    candidate_family = str(candidate_family or "")
    if expected_card in CONVERSATION_FRONTIER_CARDS:
        return False
    if expected_family in CONVERSATION_FRONTIER_FAMILIES:
        return False
    if expected_family in PROMOTION_FRONTIER_FAMILIES:
        return True
    return bool(candidate_family and candidate_family in PROMOTION_FRONTIER_FAMILIES)


def frontier_family_matches(expected_family: str, observed_family: str, expected_card: str) -> bool:
    if not expected_family or not observed_family:
        return True
    if expected_family == observed_family:
        return True
    if expected_card in CONVERSATION_FRONTIER_CARDS:
        return (
            expected_family in CONVERSATION_FRONTIER_FAMILIES
            and observed_family in CONVERSATION_FRONTIER_FAMILIES
        )
    return False


def active_frontier_alignment_ok(
    *,
    expected_family: str,
    expected_card: str,
    candidate_family: str,
    transfer_family: str,
    transfer_card: str,
    architecture_family: str,
    architecture_card: str,
) -> bool:
    if not expected_family:
        return True
    if not frontier_family_matches(expected_family, candidate_family, expected_card):
        return False
    if not frontier_family_matches(expected_family, transfer_family, expected_card):
        return False
    if not frontier_family_matches(expected_family, architecture_family, expected_card):
        return False
    if expected_card:
        for card in (transfer_card, architecture_card):
            if card and card != expected_card:
                return False
    return True


def latest_cycle_report(primary: dict[str, Any], correction: dict[str, Any]) -> dict[str, Any]:
    primary_ts = parse_time(primary.get("created_utc"))
    correction_ts = parse_time(correction.get("created_utc"))
    if correction and correction_ts >= primary_ts:
        return correction
    return primary


def parse_time(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0


def age_seconds(value: Any, now_ts: float) -> int:
    if not value:
        return 10**9
    try:
        stamp = str(value)
        if stamp.endswith("Z"):
            stamp = stamp[:-1] + "+00:00"
        return max(0, int(now_ts - datetime.fromisoformat(stamp).timestamp()))
    except ValueError:
        return 10**9


def file_age_seconds(path: Path, now_ts: float) -> int:
    try:
        return max(0, int(now_ts - path.stat().st_mtime))
    except OSError:
        return 10**9


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_latest_json(directory: Path, pattern: str) -> dict[str, Any]:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        return {}
    value = read_json(matches[-1])
    return value if isinstance(value, dict) else {}


def read_latest_json_with_path(directory: Path, pattern: str) -> dict[str, Any]:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        return {}
    path = matches[-1]
    value = read_json(path)
    if isinstance(value, dict):
        value = dict(value)
        value["__path"] = str(path)
        return value
    return {}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "item")).strip("_") or "item"


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
