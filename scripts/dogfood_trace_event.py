#!/usr/bin/env python3
"""Consent-gated dogfood trace event logger.

By default this script refuses to write events because
configs/dogfood_trace.local.json is absent. It is a local operator tool, not a
training importer.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dogfood_trace.local.json"
DEFAULT_TRACE_PATH = ROOT / "runtime" / "dogfood" / "daily_use_events.jsonl"
DEFAULT_OUT = ROOT / "reports" / "dogfood_trace_event_check.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "dogfood_trace_event_check.md"
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
ALLOWED_LANES = {
    "tool_transcript",
    "structured_parsing",
    "state_machine_parser",
    "long_horizon_planning",
    "device_routing",
    "storage_operator",
    "chat_checkpoint",
    "code_assistant",
    "tool_assistant",
    "planning_assistant",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--trace-jsonl", default="")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--surface", default="cli")
    parser.add_argument("--assistant-lane", default="chat_checkpoint")
    parser.add_argument("--outcome", default="ignored")
    parser.add_argument("--intent-summary-redacted", default="")
    parser.add_argument("--artifact-ref", action="append", default=[])
    parser.add_argument("--error-family", default="")
    parser.add_argument("--duration-ms", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    config_path = resolve(args.config)
    local_config = read_json(config_path) if config_path.exists() else {}
    trace_path = resolve(args.trace_jsonl or str(local_config.get("trace_jsonl") or DEFAULT_TRACE_PATH.relative_to(ROOT)))
    capture_enabled = bool(local_config.get("capture_enabled", False))
    capture_consent = valid_utc_timestamp(str(local_config.get("explicit_capture_consent_utc") or ""))
    training_enabled = bool(local_config.get("training_enabled", False))
    raw_text_capture_enabled = bool(local_config.get("raw_text_capture_enabled", False))
    event = build_event(args)
    can_write = bool(args.execute and capture_enabled and capture_consent and not raw_text_capture_enabled)

    gates = [
        gate("execute_requested", bool(args.execute), {"execute": bool(args.execute)}, "soft"),
        gate("local_config_exists", config_path.exists(), rel(config_path), "soft"),
        gate("capture_enabled", capture_enabled, {"capture_enabled": capture_enabled}, "soft"),
        gate("explicit_capture_consent_valid", capture_consent, {"explicit_capture_consent_utc": local_config.get("explicit_capture_consent_utc")}, "soft"),
        gate("outcome_allowed", event["outcome"] in ALLOWED_OUTCOMES, event["outcome"], "hard"),
        gate("assistant_lane_allowed", event["assistant_lane"] in ALLOWED_LANES, event["assistant_lane"], "hard"),
        gate("redacted_metadata_present", redacted_metadata_present(event), redacted_metadata_evidence(event), "hard"),
        gate("forbidden_fields_absent", forbidden_fields_absent(event), sorted(FORBIDDEN_EVENT_KEYS.intersection(flat_keys(event))), "hard"),
        gate("training_not_performed_by_event_logger", True, {"training_enabled": training_enabled}, "hard"),
        gate("raw_text_capture_disabled", not raw_text_capture_enabled, {"raw_text_capture_enabled": raw_text_capture_enabled}, "hard"),
        gate("external_inference_zero", True, 0, "hard"),
        gate("public_benchmark_training_zero", True, 0, "hard"),
        gate("no_fallback_returns", True, 0, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    event_written = False
    write_blocker = ""
    if hard_failed:
        state = "RED"
        write_blocker = "hard_gate_failed"
    elif can_write:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        event_written = True
        state = "GREEN"
    else:
        state = "YELLOW"
        write_blocker = "capture_disabled_or_consent_missing"
    return {
        "policy": "project_theseus_dogfood_trace_event_logger_v0",
        "created_utc": now(),
        "trigger_state": state,
        "config": rel(config_path),
        "trace_jsonl": rel(trace_path),
        "event_written": event_written,
        "write_blocker": write_blocker,
        "event_preview": event,
        "summary": {
            "capture_enabled": capture_enabled,
            "explicit_capture_consent_present": capture_consent,
            "training_enabled": training_enabled,
            "raw_text_capture_enabled": raw_text_capture_enabled,
            "trained_on_user_text": False,
            "external_inference_calls": 0,
            "fallback_returns": 0,
            "no_fallback_returns": True,
        },
        "gates": gates,
        "score_semantics": (
            "Dogfood event logger only. It writes redacted metadata events only after explicit local "
            "capture consent. It does not train, collect raw user text by default, call a teacher, "
            "call external inference, use public benchmark rows, return fallback bodies, or unlock promotion."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def build_event(args: argparse.Namespace) -> dict[str, Any]:
    created = now()
    seed = "|".join(
        [
            created,
            str(args.surface),
            str(args.assistant_lane),
            str(args.outcome),
            str(args.intent_summary_redacted),
        ]
    )
    return {
        "event_id": stable_hash(seed)[:20],
        "created_utc": created,
        "surface": str(args.surface),
        "assistant_lane": str(args.assistant_lane),
        "outcome": str(args.outcome),
        "consent_scope": "dogfood_metadata_only_v0",
        "intent_summary_redacted": str(args.intent_summary_redacted or ""),
        "artifact_refs": [str(item) for item in args.artifact_ref],
        "error_family": str(args.error_family or ""),
        "duration_ms": max(0, int(args.duration_ms or 0)),
    }


def redacted_metadata_present(event: dict[str, Any]) -> bool:
    return bool(
        str(event.get("intent_summary_redacted") or "").strip()
        or event.get("artifact_refs")
        or str(event.get("error_family") or "").strip()
        or int(event.get("duration_ms") or 0) > 0
    )


def redacted_metadata_evidence(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent_summary_redacted_present": bool(str(event.get("intent_summary_redacted") or "").strip()),
        "artifact_ref_count": len(event.get("artifact_refs") or []),
        "error_family_present": bool(str(event.get("error_family") or "").strip()),
        "duration_ms_positive": int(event.get("duration_ms") or 0) > 0,
    }


def forbidden_fields_absent(value: Any) -> bool:
    return not bool(FORBIDDEN_EVENT_KEYS.intersection(flat_keys(value)))


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


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Dogfood Trace Event Check",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- event_written: `{report.get('event_written')}`",
        f"- write_blocker: `{report.get('write_blocker')}`",
        f"- trace_jsonl: `{report.get('trace_jsonl')}`",
        f"- capture_enabled: `{summary.get('capture_enabled')}`",
        f"- explicit_capture_consent_present: `{summary.get('explicit_capture_consent_present')}`",
        f"- trained_on_user_text: `{summary.get('trained_on_user_text')}`",
        "",
        "## Semantics",
        "",
        str(report.get("score_semantics") or ""),
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
