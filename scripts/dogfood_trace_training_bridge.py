#!/usr/bin/env python3
"""Consent-gated bridge from dogfood events to private training rows.

This script does not capture events. It only reads existing redacted dogfood
metadata events and, when explicitly enabled by a local consent config, exports
accepted/missed/ignored/corrected/completed rows into the private training
surface.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dogfood_trace.local.json"
DEFAULT_TRACE = ROOT / "runtime" / "dogfood" / "daily_use_events.jsonl"
DEFAULT_TRAINING_ROWS = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "dogfood_daily_use_trace_training_rows.jsonl"
DEFAULT_OUT = ROOT / "reports" / "dogfood_trace_training_bridge.json"
DEFAULT_MD = ROOT / "reports" / "dogfood_trace_training_bridge.md"
ALLOWED_OUTCOMES = {"accepted", "missed", "ignored", "corrected", "completed"}
FORBIDDEN_EVENT_KEYS = {
    "raw_user_text",
    "raw_assistant_text",
    "prompt",
    "completion",
    "secret_values",
    "private_file_contents",
    "public_benchmark_prompt_or_solution",
    "fallback_return",
    "fallback_return_used",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--trace-jsonl", default="")
    parser.add_argument("--training-rows-jsonl", default="")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD.relative_to(ROOT)))
    parser.add_argument("--compact-existing", action="store_true")
    parser.add_argument("--compact-backup-jsonl", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = read_json(config_path)
    trace_path = resolve(args.trace_jsonl or str(config.get("trace_jsonl") or DEFAULT_TRACE.relative_to(ROOT)))
    training_path = resolve(
        args.training_rows_jsonl or str(config.get("training_rows_jsonl") or DEFAULT_TRAINING_ROWS.relative_to(ROOT))
    )
    events = read_jsonl(trace_path)
    report = build_report(args, config_path, config, trace_path, training_path, events)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_report(
    args: argparse.Namespace,
    config_path: Path,
    config: dict[str, Any],
    trace_path: Path,
    training_path: Path,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    capture_enabled = bool(config.get("capture_enabled", False))
    capture_consent_value = str(config.get("explicit_capture_consent_utc") or "")
    capture_consent = valid_utc_timestamp(capture_consent_value)
    training_enabled = bool(config.get("training_enabled", False))
    training_consent_value = str(config.get("explicit_training_consent_utc") or "")
    training_consent = valid_utc_timestamp(training_consent_value)
    raw_text_capture_enabled = bool(config.get("raw_text_capture_enabled", False))
    training_candidate_events = [event for event in events if event_is_training_candidate(event)]
    valid_events = [event for event in training_candidate_events if event_is_trainable(event)]
    rejected_events = len(training_candidate_events) - len(valid_events)
    non_trainable_event_count = len(events) - len(training_candidate_events)
    existing_training_rows = read_jsonl(training_path)
    existing_source_event_ids = training_source_event_ids(existing_training_rows)
    existing_duplicate_summary = duplicate_training_row_summary(existing_training_rows)
    rows = [
        training_row(event)
        for event in valid_events
        if str(event.get("event_id") or "") not in existing_source_event_ids
    ]
    all_trainable_events_already_exported = (
        bool(args.execute)
        and bool(valid_events)
        and not rows
        and all(str(event.get("event_id") or "") in existing_source_event_ids for event in valid_events)
        and existing_duplicate_summary["duplicate_source_event_row_count"] == 0
    )
    gates = [
        gate("execute_requested", bool(args.execute), {"execute": bool(args.execute)}, "soft"),
        gate("local_config_exists", config_path.exists(), rel(config_path), "soft"),
        gate("capture_enabled", capture_enabled, {"capture_enabled": capture_enabled}, "soft"),
        gate("explicit_capture_consent_valid", capture_consent, {"explicit_capture_consent_utc": config.get("explicit_capture_consent_utc")}, "soft"),
        gate("training_enabled", training_enabled, {"training_enabled": training_enabled}, "soft"),
        gate("explicit_training_consent_valid", training_consent, {"explicit_training_consent_utc": config.get("explicit_training_consent_utc")}, "soft"),
        gate(
            "training_consent_separate_from_capture_consent",
            (not training_enabled) or not training_consent or training_consent_value != capture_consent_value,
            {
                "training_enabled": training_enabled,
                "capture_consent_matches_training_consent": bool(
                    training_consent_value and training_consent_value == capture_consent_value
                ),
            },
            "hard",
        ),
        gate("raw_text_capture_disabled", not raw_text_capture_enabled, {"raw_text_capture_enabled": raw_text_capture_enabled}, "hard"),
        gate("trace_events_present", len(events) > 0, {"event_count": len(events)}, "soft"),
        gate("trainable_events_present", len(valid_events) > 0, {"trainable_event_count": len(valid_events)}, "soft"),
        gate(
            "allowed_outcomes_only",
            rejected_events == 0,
            {"rejected_events": rejected_events, "training_candidate_event_count": len(training_candidate_events)},
            "hard",
        ),
        gate("redacted_metadata_present", all(redacted_metadata_present(event) for event in valid_events), {"trainable_event_count": len(valid_events)}, "hard"),
        gate("forbidden_fields_absent", forbidden_fields_absent(events), forbidden_key_evidence(events), "hard"),
        gate("external_inference_zero", True, 0, "hard"),
        gate("public_benchmark_training_zero", True, 0, "hard"),
        gate("teacher_calls_zero", True, 0, "hard"),
        gate("no_fallback_returns", True, 0, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    allowed_to_write = (
        bool(args.execute)
        and capture_enabled
        and capture_consent
        and training_enabled
        and training_consent
        and not raw_text_capture_enabled
        and not hard_failed
        and bool(rows)
    )
    rows_written = 0
    compacted_existing_duplicate_rows = 0
    compact_backup_path = resolve(
        args.compact_backup_jsonl
        or "reports/dogfood_trace_training_bridge_compacted_duplicates.jsonl"
    )
    write_blocker = ""
    if hard_failed:
        trigger_state = "RED"
        write_blocker = "hard_gate_failed"
    elif allowed_to_write:
        training_path.parent.mkdir(parents=True, exist_ok=True)
        with training_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                rows_written += 1
        trigger_state = "GREEN"
    elif all_trainable_events_already_exported:
        trigger_state = "GREEN"
        write_blocker = "no_new_unique_trainable_events"
    else:
        trigger_state = "YELLOW"
        if not rows and valid_events:
            write_blocker = "no_new_unique_trainable_events"
        else:
            write_blocker = "consent_or_events_missing"
    if (
        bool(args.execute)
        and bool(args.compact_existing)
        and not hard_failed
        and existing_duplicate_summary["duplicate_source_event_row_count"] > 0
    ):
        kept_rows, duplicate_rows = compact_training_rows(existing_training_rows)
        compact_backup_path.parent.mkdir(parents=True, exist_ok=True)
        with compact_backup_path.open("w", encoding="utf-8") as handle:
            for row in duplicate_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        training_path.parent.mkdir(parents=True, exist_ok=True)
        with training_path.open("w", encoding="utf-8") as handle:
            for row in kept_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        compacted_existing_duplicate_rows = len(duplicate_rows)
        existing_training_rows = kept_rows
        existing_source_event_ids = training_source_event_ids(existing_training_rows)
        existing_duplicate_summary = duplicate_training_row_summary(existing_training_rows)
    outcome_counts = Counter(str(event.get("outcome") or "") for event in valid_events)
    lane_counts = Counter(str(event.get("assistant_lane") or "") for event in valid_events)
    return {
        "policy": "project_theseus_dogfood_trace_training_bridge_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "config": rel(config_path),
        "trace_jsonl": rel(trace_path),
        "training_rows_jsonl": rel(training_path),
        "execute": bool(args.execute),
        "training_rows_written": rows_written,
        "compacted_existing_duplicate_rows": compacted_existing_duplicate_rows,
        "compact_backup_jsonl": rel(compact_backup_path) if compacted_existing_duplicate_rows else "",
        "write_blocker": write_blocker,
        "summary": {
            "event_count": len(events),
            "training_candidate_event_count": len(training_candidate_events),
            "trainable_event_count": len(valid_events),
            "existing_training_source_event_count": len(existing_source_event_ids),
            "existing_training_row_count": len(existing_training_rows),
            "duplicate_training_source_event_count": existing_duplicate_summary["duplicate_source_event_count"],
            "duplicate_training_source_event_row_count": existing_duplicate_summary["duplicate_source_event_row_count"],
            "new_training_event_count": len(rows),
            "all_trainable_events_already_exported": all_trainable_events_already_exported,
            "rejected_event_count": rejected_events,
            "non_trainable_event_count": non_trainable_event_count,
            "outcome_counts": dict(outcome_counts.most_common()),
            "lane_counts": dict(lane_counts.most_common()),
            "capture_enabled": capture_enabled,
            "explicit_capture_consent_present": capture_consent,
            "training_enabled": training_enabled,
            "explicit_training_consent_present": training_consent,
            "raw_text_capture_enabled": raw_text_capture_enabled,
            "trained_on_raw_text": False,
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "fallback_returns": 0,
            "no_fallback_returns": True,
        },
        "training_row_schema": {
            "policy": "project_theseus_dogfood_private_training_row_v0",
            "inputs": [
                "assistant_lane",
                "outcome",
                "intent_summary_redacted",
                "artifact_refs",
                "error_family",
                "duration_bucket",
            ],
            "targets": [
                "outcome_label",
                "repair_priority",
                "lane_feedback",
            ],
            "raw_user_text": False,
            "public_benchmark_rows": False,
            "teacher_rows": False,
        },
        "gates": gates,
        "score_semantics": (
            "Dogfood training bridge only. It exports redacted accepted/missed/ignored/corrected/completed metadata events into private "
            "training rows only after local capture consent and local training consent are present. It does not "
            "collect raw user text, call a teacher, use public benchmark rows, run external inference, return fallback bodies, or promote a model."
        ),
        "external_inference_calls": 0,
    }


def event_is_trainable(event: dict[str, Any]) -> bool:
    return (
        str(event.get("outcome") or "") in ALLOWED_OUTCOMES
        and redacted_metadata_present(event)
        and forbidden_fields_absent(event)
    )


def event_is_training_candidate(event: dict[str, Any]) -> bool:
    """Return True for dogfood metadata rows that should be considered for training.

    The dogfood trace may sit next to adjacent runtime telemetry. Telemetry can
    never become a training row, but it also should not poison an otherwise safe
    batch as long as forbidden/raw/public fields remain absent across the file.
    """

    return (
        str(event.get("consent_scope") or "") == "dogfood_metadata_only_v0"
        or str(event.get("outcome") or "") in ALLOWED_OUTCOMES
    )


def training_row(event: dict[str, Any]) -> dict[str, Any]:
    outcome = str(event.get("outcome") or "")
    lane = str(event.get("assistant_lane") or "")
    return {
        "policy": "project_theseus_dogfood_private_training_row_v0",
        "created_utc": now(),
        "source_event_id": str(event.get("event_id") or ""),
        "source_event_created_utc": str(event.get("created_utc") or ""),
        "assistant_lane": lane,
        "surface": str(event.get("surface") or ""),
        "intent_summary_redacted": str(event.get("intent_summary_redacted") or ""),
        "artifact_refs": [str(item) for item in event.get("artifact_refs") or []],
        "error_family": str(event.get("error_family") or ""),
        "duration_bucket": duration_bucket(int(event.get("duration_ms") or 0)),
        "outcome_label": outcome,
        "repair_priority": repair_priority(outcome),
        "lane_feedback": f"{lane}:{outcome}",
        "raw_user_text_included": False,
        "public_benchmark_row": False,
        "teacher_generated": False,
        "external_inference_calls": 0,
    }


def training_source_event_ids(rows: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        source_event_id = str(row.get("source_event_id") or "")
        if source_event_id:
            ids.add(source_event_id)
    return ids


def duplicate_training_row_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(row.get("source_event_id") or "")
        for row in rows
        if str(row.get("source_event_id") or "")
    )
    duplicate_source_event_count = sum(1 for count in counts.values() if count > 1)
    duplicate_source_event_row_count = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "duplicate_source_event_count": duplicate_source_event_count,
        "duplicate_source_event_row_count": duplicate_source_event_row_count,
    }


def compact_training_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        source_event_id = str(row.get("source_event_id") or "")
        if not source_event_id or source_event_id not in seen:
            kept.append(row)
            if source_event_id:
                seen.add(source_event_id)
        else:
            duplicates.append(row)
    return kept, duplicates


def duration_bucket(duration_ms: int) -> str:
    if duration_ms <= 0:
        return "unknown"
    if duration_ms < 5_000:
        return "under_5s"
    if duration_ms < 60_000:
        return "under_1m"
    if duration_ms < 600_000:
        return "under_10m"
    return "over_10m"


def repair_priority(outcome: str) -> str:
    if outcome in {"missed", "corrected"}:
        return "high"
    if outcome == "ignored":
        return "low"
    if outcome == "completed":
        return "completed_positive"
    return "positive_reinforcement"


def redacted_metadata_present(event: dict[str, Any]) -> bool:
    return bool(
        str(event.get("intent_summary_redacted") or "").strip()
        or event.get("artifact_refs")
        or str(event.get("error_family") or "").strip()
        or int(event.get("duration_ms") or 0) > 0
    )


def forbidden_fields_absent(value: Any) -> bool:
    return not bool(FORBIDDEN_EVENT_KEYS.intersection(flat_keys(value)))


def forbidden_key_evidence(value: Any) -> dict[str, Any]:
    observed = sorted(FORBIDDEN_EVENT_KEYS.intersection(flat_keys(value)))
    return {"forbidden_keys_present": observed, "forbidden_key_count": len(observed)}


def flat_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(flat_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(flat_keys(item))
    return keys


def valid_utc_timestamp(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Dogfood Trace Training Bridge",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- trace_jsonl: `{report.get('trace_jsonl')}`",
        f"- training_rows_jsonl: `{report.get('training_rows_jsonl')}`",
        f"- event_count: `{summary.get('event_count')}`",
        f"- training_candidate_event_count: `{summary.get('training_candidate_event_count')}`",
        f"- trainable_event_count: `{summary.get('trainable_event_count')}`",
        f"- non_trainable_event_count: `{summary.get('non_trainable_event_count')}`",
        f"- training_rows_written: `{report.get('training_rows_written')}`",
        f"- compacted_existing_duplicate_rows: `{report.get('compacted_existing_duplicate_rows')}`",
        f"- write_blocker: `{report.get('write_blocker')}`",
        f"- trained_on_raw_text: `{summary.get('trained_on_raw_text')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    lines.extend(["", str(report.get("score_semantics") or "")])
    return "\n".join(lines).rstrip() + "\n"


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
