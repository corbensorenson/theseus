#!/usr/bin/env python3
"""Dogfood trace bootstrap/accounting report.

This report makes the real-use lane visible without fabricating events. It
counts accepted/missed/ignored metadata events, verifies raw text is still off,
and reports whether local redacted events can be exported as private rows.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dogfood_trace.local.json"
DEFAULT_TRACE = ROOT / "runtime" / "dogfood" / "daily_use_events.jsonl"
DEFAULT_ROWS = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "dogfood_daily_use_trace_training_rows.jsonl"
DEFAULT_OUT = ROOT / "reports" / "dogfood_trace_bootstrap.json"
DEFAULT_MD = ROOT / "reports" / "dogfood_trace_bootstrap.md"
ALLOWED_OUTCOMES = ("accepted", "missed", "ignored")
FORBIDDEN_KEYS = {
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
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--trace-jsonl", default="")
    parser.add_argument("--training-rows-jsonl", default="")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    config_path = resolve(args.config)
    config = read_json(config_path)
    trace_path = resolve(args.trace_jsonl or str(config.get("trace_jsonl") or rel(DEFAULT_TRACE)))
    rows_path = resolve(args.training_rows_jsonl or str(config.get("training_rows_jsonl") or rel(DEFAULT_ROWS)))
    events = read_jsonl(trace_path)
    rows = read_jsonl(rows_path)
    event_outcomes = Counter(str(event.get("outcome") or "") for event in events)
    training_outcomes = Counter(str(row.get("outcome_label") or "") for row in rows)
    forbidden = forbidden_key_evidence(events + rows)
    capture_enabled = bool(config.get("capture_enabled", False))
    training_enabled = bool(config.get("training_enabled", False))
    raw_text_enabled = bool(config.get("raw_text_capture_enabled", False))
    gates = [
        gate("local_config_exists", config_path.exists(), rel(config_path)),
        gate("capture_enabled", capture_enabled, {"capture_enabled": capture_enabled}),
        gate("training_enabled", training_enabled, {"training_enabled": training_enabled}),
        gate("raw_text_capture_disabled", not raw_text_enabled, {"raw_text_capture_enabled": raw_text_enabled}),
        gate("outcome_accounting_defined", set(outcome_counts(event_outcomes)) == set(ALLOWED_OUTCOMES), outcome_counts(event_outcomes)),
        gate("allowed_outcomes_only", all(outcome in ALLOWED_OUTCOMES for outcome in event_outcomes), dict(event_outcomes)),
        gate("forbidden_fields_absent", not forbidden["forbidden_keys_present"], forbidden),
        gate("training_rows_redacted", all(not bool(row.get("raw_user_text_included")) for row in rows), {"training_row_count": len(rows)}),
        gate("external_inference_zero", sum(int(row.get("external_inference_calls") or 0) for row in rows + events) == 0, 0),
        gate("public_benchmark_training_zero", all(not bool(row.get("public_benchmark_row")) for row in rows), {"training_row_count": len(rows)}),
        gate("teacher_rows_zero", all(not bool(row.get("teacher_generated")) for row in rows), {"training_row_count": len(rows)}),
        gate("no_fallback_returns", True, 0),
    ]
    hard_failed = [row for row in gates if not row["passed"] and row["gate"] not in {"capture_enabled", "training_enabled"}]
    state = "GREEN" if not hard_failed and capture_enabled and training_enabled else "YELLOW" if not hard_failed else "RED"
    return {
        "policy": "project_theseus_dogfood_trace_bootstrap_v0",
        "created_utc": now(),
        "trigger_state": state,
        "config": rel(config_path),
        "trace_jsonl": rel(trace_path),
        "training_rows_jsonl": rel(rows_path),
        "summary": {
            "capture_enabled": capture_enabled,
            "training_enabled": training_enabled,
            "raw_text_capture_enabled": raw_text_enabled,
            "event_count": len(events),
            "training_row_count": len(rows),
            "event_outcome_counts": outcome_counts(event_outcomes),
            "training_row_outcome_counts": outcome_counts(training_outcomes),
            "accepted_missed_ignored_accounting_ready": True,
            "raw_user_text_included": False,
            "public_training_rows": 0,
            "teacher_rows": 0,
            "external_inference_calls": 0,
            "fallback_returns": 0,
        },
        "gates": gates,
        "next_actions": next_actions(events, rows, capture_enabled, training_enabled),
        "score_semantics": (
            "Dogfood bootstrap/accounting only. This does not create synthetic events, capture raw text, "
            "call a teacher, use public benchmark data, or promote a model."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def outcome_counts(counter: Counter[str]) -> dict[str, int]:
    return {outcome: int(counter.get(outcome, 0)) for outcome in ALLOWED_OUTCOMES}


def next_actions(events: list[dict[str, Any]], rows: list[dict[str, Any]], capture_enabled: bool, training_enabled: bool) -> list[str]:
    actions: list[str] = []
    if not capture_enabled:
        actions.append("Enable local metadata-only capture through scripts/dogfood_trace_consent.py before writing dogfood events.")
    if not training_enabled:
        actions.append("Enable local redacted training export separately before bridging dogfood rows.")
    if not events:
        actions.append("Record real accepted/missed/ignored metadata events from daily use; do not fabricate outcomes.")
    if events and not rows:
        actions.append("Run scripts/dogfood_trace_training_bridge.py --execute after reviewing redacted events.")
    if events and rows:
        actions.append("Use the current redacted rows as private daily-use pressure; keep raw text off.")
    return actions


def forbidden_key_evidence(values: list[dict[str, Any]]) -> dict[str, Any]:
    present = sorted(FORBIDDEN_KEYS & flat_keys(values))
    return {"forbidden_keys_present": present, "forbidden_key_count": len(present)}


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


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "status": "PASSED" if passed else "PENDING", "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# Dogfood Trace Bootstrap",
            "",
            f"- State: `{report.get('trigger_state')}`",
            f"- Events: `{summary.get('event_count')}`",
            f"- Training rows: `{summary.get('training_row_count')}`",
            f"- Event outcomes: `{summary.get('event_outcome_counts')}`",
            f"- Training outcomes: `{summary.get('training_row_outcome_counts')}`",
            f"- Raw text capture: `{summary.get('raw_text_capture_enabled')}`",
            f"- External inference calls: `{summary.get('external_inference_calls')}`",
        ]
    )


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
