"""Phase-ledger helpers for long Code LM jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_phase_event(
    path: Path,
    phase: str,
    event: str,
    payload: dict[str, Any] | None = None,
    *,
    phase_contracts: dict[str, dict[str, Any]] | None = None,
) -> None:
    contract = (phase_contracts or {}).get(phase, {})
    row = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "event": event,
        "contract": contract,
        "payload": payload or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_phase_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def summarize_phase_ledger(
    path: Path,
    *,
    root: Path | None = None,
    phase_contracts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = read_phase_ledger(path)
    phases: dict[str, dict[str, Any]] = {}
    contracts = phase_contracts or {}
    for row in rows:
        phase = str(row.get("phase") or "")
        event = str(row.get("event") or "")
        if not phase:
            continue
        contract = contracts.get(phase, {})
        item = phases.setdefault(
            phase,
            {
                "event_count": 0,
                "latest_event": "",
                "latest_utc": "",
                "elapsed_seconds": None,
                "returncode": None,
                "timed_out": None,
                "target_max_seconds": _get_path(contract, ["target_max_seconds"], 0),
                "consumer": _get_path(contract, ["consumer"], ""),
                "evidence_semantics": _get_path(contract, ["evidence_semantics"], ""),
            },
        )
        item["event_count"] += 1
        item["latest_event"] = event
        item["latest_utc"] = row.get("created_utc")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if "elapsed_seconds" in payload:
            item["elapsed_seconds"] = payload.get("elapsed_seconds")
        if "returncode" in payload:
            item["returncode"] = payload.get("returncode")
        if "timed_out" in payload:
            item["timed_out"] = payload.get("timed_out")
    slow = []
    for phase, item in phases.items():
        elapsed = _number(item.get("elapsed_seconds"))
        target = _number(item.get("target_max_seconds"))
        if target and elapsed and elapsed > target:
            slow.append({"phase": phase, "elapsed_seconds": elapsed, "target_max_seconds": target})
    return {
        "path": _rel(path, root=root),
        "event_count": len(rows),
        "phases": phases,
        "slow_phases": slow,
        "score_semantics": "phase_timing_control_signal_not_capability_evidence",
    }


def _get_path(row: Any, path: list[str], default: Any = None) -> Any:
    current = row
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _number(value: Any) -> float:
    try:
        if value is None or isinstance(value, bool):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rel(path: Path, *, root: Path | None) -> str:
    value = path.resolve()
    if root is not None:
        try:
            return str(value.relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            pass
    return str(value).replace("\\", "/")
