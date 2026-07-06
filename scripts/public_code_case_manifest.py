"""Public code benchmark case-manifest helpers.

The manifest is a selector only. It may pin public task IDs and coarse metadata
for calibration, but it must not carry prompts, tests, reference solutions, or
candidate code into private training paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CASE_MANIFEST_POLICY = "project_theseus_public_code_case_manifest_v1"
PUBLIC_CASE_MANIFEST_CONTENT_POLICY = {
    "public_calibration_only": True,
    "prompts_exported": False,
    "tests_exported": False,
    "solutions_exported": False,
    "candidate_code_exported": False,
    "private_training_allowed": False,
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def load_case_manifest(path_value: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    raw = str(path_value or "").strip()
    if not raw:
        return {}
    rows_by_card: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(read_jsonl(resolve(raw))):
        card_id = str(row.get("card_id") or "").strip()
        task_id = str(row.get("task_id") or "").strip()
        if not card_id or not task_id:
            continue
        if row.get("prompts_exported") or row.get("tests_exported") or row.get("solutions_exported"):
            continue
        clean = dict(row)
        clean["manifest_index"] = index
        rows_by_card.setdefault(card_id, []).append(clean)
    return rows_by_card


def manifest_card_counts(rows_by_card: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {card_id: len(rows) for card_id, rows in sorted(rows_by_card.items())}


def manifest_pool_size(max_cases: int, rows_by_card: dict[str, list[dict[str, Any]]], *, default_pool: int = 4096) -> int:
    selected = sum(len(rows) for rows in rows_by_card.values())
    if selected <= 0:
        return max(1, int(max_cases))
    return max(max(1, int(max_cases)), default_pool, selected * 8)


def filter_tasks_for_card(
    tasks: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not manifest_rows:
        return tasks, []
    by_task_id = {str(task.get("task_id") or ""): task for task in tasks}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for row in manifest_rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        task = by_task_id.get(task_id)
        if task is None:
            missing.append(task_id)
            continue
        selected.append(task)
    return selected, missing


def manifest_context(path_value: str | Path | None, rows_by_card: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    raw = str(path_value or "").strip()
    return {
        "enabled": bool(raw),
        "path": rel(resolve(raw)) if raw else "",
        "policy": PUBLIC_CASE_MANIFEST_POLICY if raw else "",
        "content_policy": PUBLIC_CASE_MANIFEST_CONTENT_POLICY if raw else {},
        "card_counts": manifest_card_counts(rows_by_card),
        "selected_count": sum(len(rows) for rows in rows_by_card.values()),
    }
