"""Audit VCM predictive prefetch decisions against local usage evidence.

The audit uses only private/local VCM artifacts: compiled context forecasts,
non-model-visible staging records, model-visible promotions, page faults, and
redacted usage events. It does not load public benchmark payloads, call a
teacher, write training rows, or return fallback answers.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_COMPILED = REPORTS / "virtual_context_compiled_context.json"
DEFAULT_USAGE_EVENTS = REPORTS / "virtual_context_memory_usage_events.jsonl"
DEFAULT_OUT = REPORTS / "vcm_prefetch_regret_audit.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_prefetch_regret_audit.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", default=rel(DEFAULT_COMPILED))
    parser.add_argument("--usage-events", default=rel(DEFAULT_USAGE_EVENTS))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--waste-cost", type=float, default=0.25)
    parser.add_argument("--miss-cost", type=float, default=1.0)
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(
        compiled_path=resolve(args.compiled),
        usage_events_path=resolve(args.usage_events),
        waste_cost=args.waste_cost,
        miss_cost=args.miss_cost,
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(
    *,
    compiled_path: Path,
    usage_events_path: Path,
    waste_cost: float,
    miss_cost: float,
    started: float,
) -> dict[str, Any]:
    compiled = read_json(compiled_path)
    usage_events = read_jsonl(usage_events_path)
    staged = list_value(compiled.get("staging_cache"))
    visible = list_value(compiled.get("model_visible_pages"))
    forecasts = list_value(compiled.get("context_demand_forecast"))
    faults = list_value(compiled.get("semantic_page_faults"))
    staged_addresses = {str(row.get("address") or "") for row in staged if isinstance(row, dict)}
    promoted_addresses = {str(row.get("address") or "") for row in staged if isinstance(row, dict) and row.get("promoted") is True}
    visible_addresses = {str(row.get("address") or "") for row in visible if isinstance(row, dict)}
    promoted_addresses |= staged_addresses & visible_addresses
    unused_staged = sorted(staged_addresses - promoted_addresses)
    fault_addresses = {str(row.get("address") or "") for row in faults if isinstance(row, dict)}
    missed_faults = sorted(fault_addresses - staged_addresses)
    forecast_by_address = {
        str(row.get("address") or ""): row
        for row in forecasts
        if isinstance(row, dict) and row.get("address")
    }
    decision_time_complete = all(
        has_keys(row, ["address", "probability", "expected_value", "deadline_step", "required_level"])
        for row in forecasts[:20]
    )
    promoted_count = len(promoted_addresses)
    staged_count = len(staged_addresses)
    precision = promoted_count / max(1, staged_count)
    miss_rate = len(missed_faults) / max(1, len(fault_addresses))
    weighted_waste = sum(float(get_path(forecast_by_address.get(addr, {}), ["expected_value"], waste_cost) or waste_cost) for addr in unused_staged)
    weighted_miss = sum(float(get_path(forecast_by_address.get(addr, {}), ["expected_value"], miss_cost) or miss_cost) for addr in missed_faults)
    regret = round(weighted_waste * waste_cost + weighted_miss * miss_cost, 6)
    usage_private = all(
        row.get("raw_text_stored") is False
        and int(row.get("external_inference_calls") or 0) == 0
        and row.get("training_use_allowed") is False
        for row in usage_events
    )
    blockers = []
    if not compiled:
        blockers.append({"kind": "missing_compiled_context", "detail": rel(compiled_path)})
    if not staged:
        blockers.append({"kind": "missing_staging_cache", "detail": "No staged VCM prefetch decisions available."})
    if not forecasts:
        blockers.append({"kind": "missing_forecast", "detail": "No decision-time VCM demand forecast rows available."})
    if not decision_time_complete:
        blockers.append({"kind": "incomplete_decision_time_features", "detail": "Forecast rows must include probability, expected_value, deadline_step, and required_level."})
    if not usage_private:
        blockers.append({"kind": "usage_boundary_violation", "detail": "Usage events must be redacted local metadata with training disabled."})
    trigger_state = "GREEN" if not blockers else "RED"
    return {
        "policy": "project_theseus_vcm_prefetch_regret_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "staged_count": staged_count,
            "promoted_count": promoted_count,
            "unused_staged_count": len(unused_staged),
            "fault_count": len(fault_addresses),
            "missed_fault_count": len(missed_faults),
            "prefetch_precision": round(precision, 6),
            "prefetch_miss_rate": round(miss_rate, 6),
            "weighted_waste": round(weighted_waste, 6),
            "weighted_miss": round(weighted_miss, 6),
            "prefetch_regret": regret,
            "decision_time_features_complete": decision_time_complete,
            "usage_event_count": len(usage_events),
            "usage_events_private": usage_private,
            "external_inference_calls": 0,
            "public_training_rows_written": 0,
            "fallback_return_count": 0,
            "runtime_seconds": round(time.perf_counter() - started, 4),
        },
        "boundary": {
            "source_artifacts": [rel(compiled_path), rel(usage_events_path)],
            "public_payloads_loaded": False,
            "public_training_use_allowed": False,
            "external_inference_allowed": False,
            "fallback_returns_allowed": False,
            "raw_usage_text_stored": False,
        },
        "regret_components": {
            "unused_staged_sample": unused_staged[:20],
            "missed_fault_sample": missed_faults[:20],
            "waste_cost": waste_cost,
            "miss_cost": miss_cost,
        },
        "blockers": blockers,
        "recommendation": recommendation(trigger_state, precision, miss_rate, regret),
    }


def recommendation(trigger_state: str, precision: float, miss_rate: float, regret: float) -> str:
    if trigger_state != "GREEN":
        return "Fix VCM forecast/staging artifacts before claiming predictive accountability."
    if precision < 0.5 or miss_rate > 0.5:
        return "Keep Predictive VCM guarded and tune private prefetch heuristics before production use."
    if regret > 25.0:
        return "Prefetch is accountable but waste is high; mine private usage misses and tune forecast thresholds."
    return "Predictive VCM has private regret accounting and can move from missing-evidence YELLOW toward guarded GREEN."


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Prefetch Regret Audit",
        "",
        f"State: `{report['trigger_state']}`",
        "",
        "## Summary",
        "",
        f"- Staged: `{summary['staged_count']}`",
        f"- Promoted: `{summary['promoted_count']}`",
        f"- Unused staged: `{summary['unused_staged_count']}`",
        f"- Faults: `{summary['fault_count']}`",
        f"- Missed faults: `{summary['missed_fault_count']}`",
        f"- Prefetch precision: `{summary['prefetch_precision']}`",
        f"- Miss rate: `{summary['prefetch_miss_rate']}`",
        f"- Prefetch regret: `{summary['prefetch_regret']}`",
        f"- Usage events private: `{summary['usage_events_private']}`",
        f"- External inference calls: `{summary['external_inference_calls']}`",
        f"- Public training rows written: `{summary['public_training_rows_written']}`",
        f"- Fallback return count: `{summary['fallback_return_count']}`",
        "",
        f"Recommendation: {report['recommendation']}",
    ]
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker.get('kind')}`: {blocker.get('detail')}")
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def has_keys(row: Any, keys: list[str]) -> bool:
    return isinstance(row, dict) and all(key in row for key in keys)


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cursor = value
    for key in path:
        if not isinstance(cursor, dict):
            return default
        cursor = cursor.get(key)
    return default if cursor is None else cursor


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
