"""Shared utility helpers for local benchmark pressure runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def budget_summary(args: argparse.Namespace) -> dict[str, Any]:
    candidate_evals = max(1, int(args.train_iterations)) * max(4, int(args.train_population))
    return {
        "episodes": max(1, int(args.episodes)),
        "steps": max(1, int(args.steps)),
        "train_iterations": max(1, int(args.train_iterations)),
        "train_population": max(4, int(args.train_population)),
        "elite_count": max(1, int(args.elite_count)),
        "eval_seed_count": max(1, int(args.eval_seed_count or args.episodes)),
        "train_candidate_evaluations": candidate_evals,
        "train_env_steps_budget": candidate_evals * max(1, int(args.steps)),
        "min_train_candidate_evaluations": max(0, int(args.min_train_candidate_evals)),
        "min_train_env_steps": max(0, int(args.min_train_env_steps)),
        "budget_report": args.budget_report,
    }


def pressure_timeout_seconds(args: argparse.Namespace) -> int:
    candidate_evals = max(1, int(args.train_iterations)) * max(4, int(args.train_population))
    train_steps = candidate_evals * max(1, int(args.steps))
    eval_steps = max(1, int(args.eval_seed_count or args.episodes)) * max(1, int(args.steps))
    estimated = int((train_steps + eval_steps) * 0.025) + 180
    return max(180, min(7200, estimated))


def suite_name(frontier_family: str, card_id: str) -> str:
    prefix = safe_name(frontier_family or "pressure")
    card = safe_name(card_id)
    return f"{prefix}_{card}"


def safe_name(value: Any) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": str(evidence)[:1200]}


def clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def resolve_path(value: str | Path) -> Path:
    if not str(value):
        return ROOT / "__missing__"
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


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


def synthetic_available_arms() -> set[str]:
    registry = read_json(ROOT / "reports" / "arm_registry.json")
    arms = {
        str(row.get("arm_name"))
        for row in registry.get("arms", [])
        if isinstance(row, dict) and row.get("arm_name")
    }
    suckers = read_json(ROOT / "reports" / "arm_sucker_registry.json")
    for key in ("suckers", "arm_suckers"):
        rows = suckers.get(key, []) if isinstance(suckers.get(key), list) else []
        for row in rows:
            if isinstance(row, dict) and (row.get("sucker_name") or row.get("name")):
                arms.add(str(row.get("sucker_name") or row.get("name")))
    return arms


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def list_limited_files(path: Path, *, limit: int) -> list[Path]:
    ignored = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
    results: list[Path] = []
    stack = [path]
    while stack and len(results) < limit:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for item in entries:
            if len(results) >= limit:
                break
            if item.is_dir():
                if item.name not in ignored:
                    stack.append(item)
            else:
                results.append(item)
    return results


def command_available(command: str) -> bool:
    return command_path(command) is not None


def command_path(command: str) -> str | None:
    found = shutil.which(command)
    if found:
        return found
    local_toolchains = [
        ROOT / "data" / "external_benchmark_candidates" / "toolchains",
        Path("D:/ProjectTheseus/tools"),
    ]
    candidates = [command]
    if os.name == "nt":
        candidates.extend([f"{command}.cmd", f"{command}.exe", f"{command}.ps1"])
    for base in local_toolchains:
        direct = base / command
        for name in candidates:
            path = direct / name
            if path.exists():
                return str(path)
        for child in [base / command, base / command.lower(), base / command.upper()]:
            for name in candidates:
                path = child / name
                if path.exists():
                    return str(path)
    return None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
