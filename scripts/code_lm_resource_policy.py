"""Resource-policy adapter for Code LM train-once orchestration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


def run_resource_policy(root: Path, reports: Path) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "scripts/resource_aware_execution_policy.py",
            "--out",
            "reports/resource_aware_execution_policy.json",
            "--markdown-out",
            "reports/resource_aware_execution_policy.md",
        ],
        cwd=root,
        check=False,
        timeout=120,
    )
    return read_json(reports / "resource_aware_execution_policy.json", {})


def summarize_resource_policy(resource: dict[str, Any]) -> dict[str, Any]:
    budget = object_field(resource, "recommended_code_lm_budget")
    return {
        "profile": _get_path(resource, ["summary", "profile"], budget.get("profile", "")),
        "start_new_code_closure": bool(budget.get("start_new_code_closure")),
        "start_new_chunked_code_closure": bool(budget.get("start_new_chunked_code_closure")),
        "start_new_train_once_fanout": bool(budget.get("start_new_train_once_fanout")),
        "reason": budget.get("reason"),
    }


def resource_allows_code_work(budget: dict[str, Any]) -> bool:
    return bool(
        budget.get("start_new_train_once_fanout")
        or budget.get("start_new_code_closure")
        or budget.get("start_new_chunked_code_closure")
    )


def resource_deferral_is_self_observation(budget: dict[str, Any], *, active_worker_present: bool = False) -> bool:
    profile = str(budget.get("profile") or "")
    reason = str(budget.get("reason") or "").lower()
    if active_worker_present:
        return False
    return (
        not resource_allows_code_work(budget)
        and (
            profile == "defer_code_closure_existing_heavy_code_worker"
            or "existing code lm" in reason
            or "worker is active" in reason
        )
    )


def read_json(path: Path, default: Any) -> Any:
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def _get_path(row: Any, path: list[str], default: Any = None) -> Any:
    current = row
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current
