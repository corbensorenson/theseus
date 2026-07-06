"""Helpers for passive Code LM train-once worker monitoring."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from code_lm_process_guard import extract_flag_value


def infer_active_worker_slug(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        command = str(row.get("command") or row.get("command_preview") or "")
        if "code_lm_train_once_fanout.py" not in command:
            continue
        slug = extract_flag_value(command, "--slug")
        if slug:
            return slug
    return ""


def phase_heartbeat_for_active_phase(paths: dict[str, Path], phase: str) -> Path | None:
    if phase == "train_once_checkpoint":
        return paths["checkpoint_phase_heartbeat"]
    if phase == "checkpoint_fanout_candidate_generation":
        return paths["fanout_phase_heartbeat"]
    if phase == "checkpoint_fanout_current_source_smoke":
        return paths["current_source_smoke_phase_heartbeat"]
    return None


def summarize_active_phase_heartbeat(path: Path | None, *, root: Path | None = None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False}
    exists = path.exists()
    row: dict[str, Any] = {
        "path": _rel(path, root=root),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "mtime": path.stat().st_mtime if exists else 0.0,
        "age_seconds": round(max(0.0, time.time() - path.stat().st_mtime), 3) if exists else None,
    }
    if not exists:
        return row
    heartbeat = _read_json(path, {})
    artifacts = heartbeat.get("progress_artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    refreshed_artifacts = _refresh_progress_artifacts(artifacts, root=root)
    latest_artifact = _latest_progress_artifact(refreshed_artifacts)
    row.update(
        {
            "phase": heartbeat.get("phase"),
            "status": heartbeat.get("status"),
            "elapsed_seconds": heartbeat.get("elapsed_seconds"),
            "timeout_seconds": heartbeat.get("timeout_seconds"),
            "progress_ratio": heartbeat.get("progress_ratio"),
            "returncode": heartbeat.get("returncode"),
            "timed_out": heartbeat.get("timed_out"),
            "consumer": heartbeat.get("consumer"),
            "evidence_semantics": heartbeat.get("evidence_semantics"),
            "progress_artifact_count": len(refreshed_artifacts),
            "latest_progress_artifact": latest_artifact.get("path", ""),
            "latest_progress_artifact_mtime": latest_artifact.get("mtime", 0.0),
            "latest_progress_stage": latest_artifact.get("progress_stage", ""),
            "latest_progress_run_status": latest_artifact.get("run_status", ""),
            "latest_progress_runtime_ms": latest_artifact.get("runtime_ms"),
            "latest_progress": latest_artifact.get("progress") if isinstance(latest_artifact.get("progress"), dict) else {},
        }
    )
    return row


def _refresh_progress_artifacts(artifacts: list[Any], *, root: Path | None) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        path_text = str(item.get("path") or "")
        path = _resolve_report_path(path_text, root=root)
        if path is not None and path.exists() and path.is_file() and path.suffix.lower() == ".json":
            refreshed.append(_summarize_progress_artifact(path, root=root))
            continue
        refreshed.append(dict(item))
    return refreshed


def _summarize_progress_artifact(path: Path, *, root: Path | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": _rel(path, root=root),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "mtime": path.stat().st_mtime if path.exists() else 0.0,
    }
    data = _read_json(path, {})
    if not isinstance(data, dict):
        return row
    progress = data.get("progress") if isinstance(data.get("progress"), dict) else {}
    row.update(
        {
            "trigger_state": data.get("trigger_state"),
            "run_status": data.get("run_status") or _get_path(data, ["summary", "run_status"], ""),
            "progress_stage": data.get("progress_stage"),
            "runtime_ms": data.get("runtime_ms") or _get_path(data, ["summary", "runtime_ms"]),
            "progress": progress,
        }
    )
    return row


def _latest_progress_artifact(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    progress_artifacts = [
        item
        for item in artifacts
        if isinstance(item, dict) and item.get("progress_stage")
    ]
    if not progress_artifacts:
        return {}
    return max(progress_artifacts, key=lambda item: float(item.get("mtime") or 0.0))


def _resolve_report_path(path_text: str, *, root: Path | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    if root is not None:
        return root / path
    return path


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _get_path(row: Any, path: list[str], default: Any = None) -> Any:
    cur = row
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _rel(path: Path, *, root: Path | None) -> str:
    value = path.resolve()
    if root is not None:
        try:
            return str(value.relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            pass
    return str(value).replace("\\", "/")
