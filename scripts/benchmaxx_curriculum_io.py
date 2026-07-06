"""Rendering and IO helpers for the Benchmaxx curriculum report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmaxx Curriculum",
        "",
        f"Generated: {payload.get('created_utc')}",
        "",
        "## Current",
        "",
        f"- Stage: {get_path(payload, ['current_stage', 'level'], '')} / {get_path(payload, ['current_stage', 'title'], '')}",
        f"- Status: {get_path(payload, ['current_stage', 'status'], '')}",
        f"- Next frontier: {get_path(payload, ['next_frontier', 'family'], '')}",
        f"- Runner: {get_path(payload, ['next_frontier', 'runner_family'], '')}",
        f"- Card/env: {get_path(payload, ['next_frontier', 'recommended_env'], '')}",
        f"- Runnable now: {get_path(payload, ['next_frontier', 'runnable_now'], '')}",
        f"- Action: {get_path(payload, ['next_frontier', 'action'], '')}",
        f"- Same-family rotation: {get_path(payload, ['next_frontier', 'same_family_rotation', 'reason'], '')}",
        f"- Transfer interleave: {get_path(payload, ['next_frontier', 'transfer_interleave', 'reason'], '')}",
        "",
        "## Ordered Course",
        "",
        "| Level | Stage | Status | Score | Next Action |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for stage in payload.get("ordered_course", []):
        lines.append(
            "| {level} | {title} | {status} | {score:.3f} | {action} |".format(
                level=stage.get("level"),
                title=md(stage.get("title")),
                status=stage.get("status"),
                score=number(stage.get("readiness_score")),
                action=md(stage.get("next_action")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_path(value: Any, path: list[Any], default: Any = None) -> Any:
    cur = value
    for key in path:
        if isinstance(cur, list) and isinstance(key, int):
            if 0 <= key < len(cur):
                cur = cur[key]
                continue
            return default
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
