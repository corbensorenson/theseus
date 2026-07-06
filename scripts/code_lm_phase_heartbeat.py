#!/usr/bin/env python3
"""Live phase heartbeat helpers for long Code LM subprocesses."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_command_with_optional_heartbeat(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    log_path: Path,
    phase: str | None = None,
    heartbeat_path: Path | None = None,
    progress_paths: list[Path] | None = None,
    phase_contracts: dict[str, dict[str, Any]] | None = None,
    heartbeat_interval_seconds: int = 30,
) -> dict[str, Any]:
    started = time.time()
    progress_paths = progress_paths or []
    phase_contracts = phase_contracts or {}
    try:
        if heartbeat_path is None:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
            timed_out = False
        else:
            result, timed_out = _run_with_heartbeat(
                command,
                cwd=cwd,
                env=env,
                timeout_seconds=timeout_seconds,
                started=started,
                log_path=log_path,
                phase=phase or "command",
                heartbeat_path=heartbeat_path,
                progress_paths=progress_paths,
                phase_contracts=phase_contracts,
                heartbeat_interval_seconds=max(1, heartbeat_interval_seconds),
            )
            write_phase_heartbeat(
                heartbeat_path,
                cwd=cwd,
                phase=phase or "command",
                command=command,
                started=started,
                timeout_seconds=timeout_seconds,
                log_path=log_path,
                progress_paths=progress_paths,
                phase_contracts=phase_contracts,
                status="completed" if result.returncode == 0 else "failed",
                returncode=result.returncode,
                timed_out=timed_out,
            )
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            command,
            124,
            stdout=_timeout_text(exc.stdout),
            stderr=_timeout_text(exc.stderr) + f"\nTimed out after {timeout_seconds}s",
        )
        timed_out = True
        if heartbeat_path is not None:
            write_phase_heartbeat(
                heartbeat_path,
                cwd=cwd,
                phase=phase or "command",
                command=command,
                started=started,
                timeout_seconds=timeout_seconds,
                log_path=log_path,
                progress_paths=progress_paths,
                phase_contracts=phase_contracts,
                status="timed_out",
                returncode=124,
                timed_out=True,
            )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (result.stdout or "") + "\n--- STDERR ---\n" + (result.stderr or ""),
        encoding="utf-8",
        errors="replace",
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "timed_out": timed_out,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "completed_utc": _now(),
        "elapsed_seconds": int(time.time() - started),
        "log_path": _rel(log_path, cwd),
        "heartbeat_path": _rel(heartbeat_path, cwd) if heartbeat_path is not None else "",
    }


def write_phase_heartbeat(
    path: Path,
    *,
    cwd: Path,
    phase: str,
    command: list[str],
    started: float,
    timeout_seconds: int,
    log_path: Path,
    progress_paths: list[Path],
    phase_contracts: dict[str, dict[str, Any]] | None = None,
    status: str,
    returncode: int | None,
    timed_out: bool,
) -> None:
    phase_contracts = phase_contracts or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    elapsed = int(time.time() - started)
    contract = phase_contracts.get(phase, {})
    row = {
        "policy": "project_theseus_code_lm_train_once_phase_heartbeat_v1",
        "created_utc": _now(),
        "phase": phase,
        "status": status,
        "elapsed_seconds": elapsed,
        "timeout_seconds": timeout_seconds,
        "progress_ratio": round(min(1.0, elapsed / max(1, timeout_seconds)), 6),
        "returncode": returncode,
        "timed_out": timed_out,
        "command_preview": _command_preview(command),
        "log_path": _rel(log_path, cwd),
        "progress_artifacts": [_summarize_progress_artifact(item, cwd) for item in progress_paths],
        "consumer": contract.get("consumer", ""),
        "evidence_semantics": contract.get("evidence_semantics", "phase_progress_only"),
        "score_semantics": "live_phase_progress_only_not_capability_evidence",
    }
    path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")


def _run_with_heartbeat(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    started: float,
    log_path: Path,
    phase: str,
    heartbeat_path: Path,
    progress_paths: list[Path],
    phase_contracts: dict[str, dict[str, Any]],
    heartbeat_interval_seconds: int,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    timed_out = False
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=heartbeat_interval_seconds)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.time() - started
            write_phase_heartbeat(
                heartbeat_path,
                cwd=cwd,
                phase=phase,
                command=command,
                started=started,
                timeout_seconds=timeout_seconds,
                log_path=log_path,
                progress_paths=progress_paths,
                phase_contracts=phase_contracts,
                status="running",
                returncode=None,
                timed_out=False,
            )
            if elapsed >= timeout_seconds:
                timed_out = True
                proc.kill()
                stdout, stderr = proc.communicate()
                stderr = (stderr or "") + f"\nTimed out after {timeout_seconds}s"
                break
    return subprocess.CompletedProcess(command, proc.returncode if not timed_out else 124, stdout, stderr), timed_out


def _summarize_progress_artifact(path: Path, cwd: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": _rel(path, cwd),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "mtime": path.stat().st_mtime if path.exists() else 0.0,
    }
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".json":
        return row
    data = _read_json(path, {})
    row.update(
        {
            "trigger_state": data.get("trigger_state"),
            "run_status": data.get("run_status") or _get_path(data, ["summary", "run_status"]),
            "progress_stage": data.get("progress_stage"),
            "runtime_ms": data.get("runtime_ms") or _get_path(data, ["summary", "runtime_ms"]),
            "progress": data.get("progress") if isinstance(data.get("progress"), dict) else {},
        }
    )
    return row


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


def _command_preview(command: list[str], limit: int = 16) -> list[str]:
    if len(command) <= limit:
        return command
    return command[:limit] + [f"...(+{len(command) - limit} args)"]


def _rel(path: str | Path, cwd: Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        value = cwd / value
    try:
        return str(value.relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return str(value).replace("\\", "/")


def _timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
