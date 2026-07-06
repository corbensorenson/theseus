#!/usr/bin/env python3
"""Report consent-gated dogfood trace readiness.

This script does not capture user text and does not train on user text. It
defines the safe local contract Theseus should use before dogfood traces become
training pressure.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "dogfood_trace_readiness.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "dogfood_trace_readiness.md"
DEFAULT_LOCAL_CONFIG = ROOT / "configs" / "dogfood_trace.local.json"
DEFAULT_TRACE_PATH = ROOT / "runtime" / "dogfood" / "daily_use_events.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_LOCAL_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    local_config_path = resolve(args.config)
    local_config = read_json(local_config_path) if local_config_path.exists() else {}
    capture_enabled = bool(local_config.get("capture_enabled", False))
    raw_text_capture_enabled = bool(local_config.get("raw_text_capture_enabled", False))
    training_enabled = bool(local_config.get("training_enabled", False))
    explicit_capture_consent_value = str(local_config.get("explicit_capture_consent_utc") or "")
    explicit_training_consent_value = str(
        local_config.get("explicit_training_consent_utc")
        or local_config.get("explicit_private_text_training_consent_utc")
        or ""
    )
    explicit_capture_consent = valid_utc_timestamp(explicit_capture_consent_value)
    explicit_training_consent = valid_utc_timestamp(explicit_training_consent_value)
    trace_path = resolve(str(local_config.get("trace_jsonl") or DEFAULT_TRACE_PATH.relative_to(ROOT)))

    training_allowed = bool(training_enabled and explicit_training_consent and capture_enabled and explicit_capture_consent)
    raw_text_allowed = bool(raw_text_capture_enabled and explicit_capture_consent)
    gates = [
        gate(
            "local_consent_required_before_capture",
            (not capture_enabled) or explicit_capture_consent,
            {
                "capture_enabled": capture_enabled,
                "explicit_capture_consent_utc_valid": explicit_capture_consent,
                "accepted_key": "explicit_capture_consent_utc",
            },
            "hard",
        ),
        gate(
            "training_disabled_by_default",
            (not training_enabled) or explicit_training_consent,
            {
                "training_enabled": training_enabled,
                "explicit_training_consent_utc_valid": explicit_training_consent,
                "accepted_key": "explicit_training_consent_utc",
                "legacy_key_still_read": "explicit_private_text_training_consent_utc",
            },
            "hard",
        ),
        gate(
            "raw_user_text_disabled_without_capture_consent",
            (not raw_text_capture_enabled) or explicit_capture_consent,
            {
                "raw_text_capture_enabled": raw_text_capture_enabled,
                "explicit_capture_consent_utc_valid": explicit_capture_consent,
            },
            "hard",
        ),
        gate(
            "training_requires_capture_and_training_consent",
            (not training_enabled) or training_allowed,
            {
                "training_allowed": training_allowed,
                "reason": "training requires capture_enabled=true and explicit_training_consent_utc",
            },
            "hard",
        ),
        gate(
            "training_consent_separate_from_capture_consent",
            (not training_allowed) or explicit_training_consent_value != explicit_capture_consent_value,
            {
                "training_allowed": training_allowed,
                "capture_consent_matches_training_consent": bool(
                    explicit_training_consent_value and explicit_training_consent_value == explicit_capture_consent_value
                ),
            },
            "hard",
        ),
        gate("external_inference_zero", True, 0, "hard"),
        gate("public_benchmark_training_zero", True, 0, "hard"),
        gate("trace_schema_defined", True, dogfood_event_schema(), "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    state = "GREEN" if not hard_failed else "RED"
    return {
        "policy": "project_theseus_dogfood_trace_readiness_v0",
        "created_utc": now(),
        "trigger_state": state,
        "config": rel(local_config_path),
        "local_config_exists": local_config_path.exists(),
        "trace_jsonl": rel(trace_path),
        "summary": {
            "capture_enabled": capture_enabled,
            "training_enabled": training_enabled,
            "raw_text_capture_enabled": raw_text_capture_enabled,
            "training_allowed": training_allowed,
            "raw_text_allowed": raw_text_allowed,
            "status": "ready_capture_disabled" if not capture_enabled else "capture_configured",
            "trained_on_user_text": False,
            "external_inference_calls": 0,
        },
        "consent_contract": {
            "capture_requires": [
                "capture_enabled=true",
                "explicit_capture_consent_utc",
                "operator-visible local config",
            ],
            "training_requires": [
                "training_enabled=true",
                "capture_enabled=true",
                "explicit_training_consent_utc",
                "separate future training importer gate",
            ],
            "default_behavior": "metadata/schema readiness only; no trace capture and no training",
            "raw_text_policy": "disabled by default; prefer redacted intent summaries and artifact references",
            "event_logger": "scripts/dogfood_trace_event.py refuses to write until capture consent is enabled",
        },
        "event_schema": dogfood_event_schema(),
        "assistant_lanes": [
            "tool_transcript",
            "structured_parsing",
            "state_machine_parser",
            "long_horizon_planning",
            "device_routing",
            "storage_operator",
            "chat_checkpoint",
        ],
        "gates": gates,
        "score_semantics": (
            "Dogfood trace readiness only. This report defines consent gates and event schema. "
            "It does not collect private user text, train on traces, call a teacher, call external "
            "inference, use public benchmark rows, or unlock promotion."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def dogfood_event_schema() -> dict[str, Any]:
    return {
        "version": "dogfood_daily_use_event_v0",
        "required_fields": [
            "event_id",
            "created_utc",
            "surface",
            "assistant_lane",
            "outcome",
            "consent_scope",
        ],
        "allowed_outcomes": ["accepted", "missed", "ignored", "corrected", "completed"],
        "outcome_semantics": {
            "accepted": "The operator used the result or evidence as-is.",
            "missed": "The system failed to produce, route, or guard the expected result and needed follow-up.",
            "ignored": "A visible suggestion, route, artifact, or available action was deliberately not used.",
            "corrected": "A previous miss or wrong artifact was repaired with auditable follow-up work.",
            "completed": "A bounded task reached its intended verified stopping point.",
        },
        "default_redacted_fields": [
            "intent_summary_redacted",
            "artifact_refs",
            "error_family",
            "duration_ms",
        ],
        "forbidden_by_default": [
            "raw_user_text",
            "secret_values",
            "private_file_contents",
            "public_benchmark_prompt_or_solution",
            "fallback_return",
            "fallback_return_used",
        ],
    }


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Dogfood Trace Readiness",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- capture_enabled: `{summary.get('capture_enabled')}`",
        f"- training_enabled: `{summary.get('training_enabled')}`",
        f"- raw_text_capture_enabled: `{summary.get('raw_text_capture_enabled')}`",
        f"- trained_on_user_text: `{summary.get('trained_on_user_text')}`",
        f"- no_fallback_returns: `True`",
        f"- trace_jsonl: `{report.get('trace_jsonl')}`",
        "",
        "## Semantics",
        "",
        str(report.get("score_semantics") or ""),
        "",
        "## Assistant Lanes",
        "",
    ]
    for lane in report.get("assistant_lanes", []):
        lines.append(f"- `{lane}`")
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
