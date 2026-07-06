#!/usr/bin/env python3
"""Operator-visible dogfood trace consent manager.

This tool exists so dogfood capture/training consent is deliberate and
machine-local. It is report-only by default. With --execute it writes
configs/dogfood_trace.local.json, but only after explicit acknowledgement flags
and valid UTC timestamps are provided.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dogfood_trace.local.json"
DEFAULT_OUT = ROOT / "reports" / "dogfood_trace_consent.json"
DEFAULT_MD = ROOT / "reports" / "dogfood_trace_consent.md"
DEFAULT_TRACE = "runtime/dogfood/daily_use_events.jsonl"
DEFAULT_ROWS = "data/training_data/high_transfer/private_train/dogfood_daily_use_trace_training_rows.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or apply local dogfood metadata consent.")
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--enable-capture", action="store_true")
    parser.add_argument("--explicit-capture-consent-utc", default="")
    parser.add_argument("--enable-training", action="store_true")
    parser.add_argument("--explicit-training-consent-utc", default="")
    parser.add_argument("--disable-capture", action="store_true")
    parser.add_argument("--disable-training", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-understand-metadata-only", action="store_true")
    parser.add_argument("--i-understand-no-raw-text", action="store_true")
    parser.add_argument("--i-understand-private-training-rows", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    current = read_json(config_path, {})
    proposed = proposed_config(current, args)
    report = build_report(args, config_path, current, proposed)
    if report["summary"]["would_write"]:
        write_json(config_path, proposed)
        report["summary"]["written"] = True
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] != "RED" else 2


def proposed_config(current: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = {
        "policy": "project_theseus_dogfood_trace_local_consent_v0",
        "capture_enabled": bool(current.get("capture_enabled", False)),
        "explicit_capture_consent_utc": str(current.get("explicit_capture_consent_utc") or ""),
        "training_enabled": bool(current.get("training_enabled", False)),
        "explicit_training_consent_utc": str(current.get("explicit_training_consent_utc") or ""),
        "raw_text_capture_enabled": False,
        "trace_jsonl": str(current.get("trace_jsonl") or DEFAULT_TRACE),
        "training_rows_jsonl": str(current.get("training_rows_jsonl") or DEFAULT_ROWS),
        "notes": [
            "Local machine-only consent file; do not commit.",
            "Capture is metadata-only accepted/missed/ignored events.",
            "Training export requires separate capture and training consent timestamps.",
            "raw_text_capture_enabled is forced false by this tool.",
        ],
    }
    if args.disable_capture:
        out["capture_enabled"] = False
        out["explicit_capture_consent_utc"] = ""
    if args.disable_training:
        out["training_enabled"] = False
        out["explicit_training_consent_utc"] = ""
    if args.enable_capture:
        out["capture_enabled"] = True
        out["explicit_capture_consent_utc"] = args.explicit_capture_consent_utc
    if args.enable_training:
        out["training_enabled"] = True
        out["explicit_training_consent_utc"] = args.explicit_training_consent_utc
    if not out["capture_enabled"]:
        out["training_enabled"] = False
        out["explicit_training_consent_utc"] = ""
    return out


def build_report(
    args: argparse.Namespace,
    config_path: Path,
    current: dict[str, Any],
    proposed: dict[str, Any],
) -> dict[str, Any]:
    capture_enabled = bool(proposed.get("capture_enabled"))
    training_enabled = bool(proposed.get("training_enabled"))
    capture_consent = valid_utc(str(proposed.get("explicit_capture_consent_utc") or ""))
    training_consent = valid_utc(str(proposed.get("explicit_training_consent_utc") or ""))
    separate_training_consent = (
        not training_enabled
        or str(proposed.get("explicit_training_consent_utc") or "")
        != str(proposed.get("explicit_capture_consent_utc") or "")
    )
    acknowledgements_ok = bool(
        args.i_understand_metadata_only
        and args.i_understand_no_raw_text
        and (not training_enabled or args.i_understand_private_training_rows)
    )
    gates = [
        gate("execute_requested", bool(args.execute), {"execute": bool(args.execute)}, "soft"),
        gate("raw_text_forced_disabled", proposed.get("raw_text_capture_enabled") is False, False, "hard"),
        gate("capture_consent_valid_when_enabled", (not capture_enabled) or capture_consent, proposed.get("explicit_capture_consent_utc"), "hard"),
        gate("training_requires_capture", (not training_enabled) or capture_enabled, {"capture_enabled": capture_enabled}, "hard"),
        gate("training_consent_valid_when_enabled", (not training_enabled) or training_consent, proposed.get("explicit_training_consent_utc"), "hard"),
        gate("training_consent_separate_from_capture", separate_training_consent, {}, "hard"),
        gate("operator_acknowledgements_present", acknowledgements_ok if args.execute else True, acknowledgement_evidence(args, training_enabled), "hard"),
        gate("public_benchmark_training_zero", True, 0, "hard"),
        gate("teacher_calls_zero", True, 0, "hard"),
        gate("external_inference_zero", True, 0, "hard"),
        gate("no_fallback_returns", True, 0, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    would_write = bool(args.execute and not hard_failed)
    return {
        "policy": "project_theseus_dogfood_trace_consent_manager_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if would_write else "YELLOW" if not hard_failed else "RED",
        "config": rel(config_path),
        "current": redact_config(current),
        "proposed": redact_config(proposed),
        "summary": {
            "execute": bool(args.execute),
            "would_write": would_write,
            "written": False,
            "capture_enabled": capture_enabled,
            "training_enabled": training_enabled,
            "raw_text_capture_enabled": False,
            "capture_consent_valid": capture_consent,
            "training_consent_valid": training_consent,
            "training_consent_separate_from_capture": separate_training_consent,
            "public_training_rows": 0,
            "teacher_calls": 0,
            "external_inference_calls": 0,
            "fallback_returns": 0,
        },
        "gates": gates,
        "next_actions": next_actions(args, capture_enabled, training_enabled, hard_failed),
        "score_semantics": (
            "Dogfood consent manager only. It can enable metadata-only local capture/training consent "
            "when explicitly executed by the operator, but it never enables raw text, public benchmark "
            "training, teacher calls, external inference, fallback returns, promotion, or model serving."
        ),
        "external_inference_calls": 0,
    }


def next_actions(args: argparse.Namespace, capture_enabled: bool, training_enabled: bool, hard_failed: list[dict[str, Any]]) -> list[str]:
    if hard_failed:
        return ["Fix failed hard consent gates before writing the local config."]
    if not args.execute:
        return [
            "Dry run only; no local consent config was changed.",
            "Use --execute with acknowledgement flags to deliberately apply this local metadata-only consent.",
        ]
    if capture_enabled and training_enabled:
        return ["Run dogfood_trace_event.py for real accepted/missed/ignored events, then dogfood_trace_training_bridge.py --execute."]
    if capture_enabled:
        return ["Capture is enabled; record real accepted/missed/ignored metadata events before enabling training export."]
    return ["Capture and training are disabled."]


def acknowledgement_evidence(args: argparse.Namespace, training_enabled: bool) -> dict[str, bool]:
    return {
        "i_understand_metadata_only": bool(args.i_understand_metadata_only),
        "i_understand_no_raw_text": bool(args.i_understand_no_raw_text),
        "i_understand_private_training_rows_required": bool(training_enabled),
        "i_understand_private_training_rows": bool(args.i_understand_private_training_rows),
    }


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": config.get("policy"),
        "capture_enabled": bool(config.get("capture_enabled", False)),
        "explicit_capture_consent_utc_present": bool(config.get("explicit_capture_consent_utc")),
        "training_enabled": bool(config.get("training_enabled", False)),
        "explicit_training_consent_utc_present": bool(config.get("explicit_training_consent_utc")),
        "raw_text_capture_enabled": bool(config.get("raw_text_capture_enabled", False)),
        "trace_jsonl": config.get("trace_jsonl"),
        "training_rows_jsonl": config.get("training_rows_jsonl"),
    }


def valid_utc(value: str) -> bool:
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


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return data if isinstance(data, dict) else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Dogfood Trace Consent",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Execute: `{summary.get('execute')}`",
        f"- Would write: `{summary.get('would_write')}`",
        f"- Written: `{summary.get('written')}`",
        f"- Capture enabled: `{summary.get('capture_enabled')}`",
        f"- Training enabled: `{summary.get('training_enabled')}`",
        f"- Raw text capture enabled: `{summary.get('raw_text_capture_enabled')}`",
        f"- Fallback returns: `{summary.get('fallback_returns')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
