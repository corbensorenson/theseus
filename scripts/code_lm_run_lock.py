"""Run-lock helpers for Code LM launchers."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def acquire_run_lock(args: argparse.Namespace, root: Path) -> int | None:
    path = resolve_path(getattr(args, "lock_path", ""), root)
    path.parent.mkdir(parents=True, exist_ok=True)
    stale_after = lock_stale_after_seconds(args)
    if path.exists():
        try:
            age = time.time() - path.stat().st_mtime
            payload = read_json(path, {})
            pid = int(payload.get("pid") or 0) if isinstance(payload, dict) else 0
            if (pid > 0 and not process_is_running(pid)) or age > stale_after or (pid <= 0 and age > 60):
                path.unlink()
        except OSError:
            pass
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    payload = {
        "policy": "project_theseus_code_lm_closure_run_lock_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "argv": sys.argv,
        "stale_after_seconds": stale_after,
    }
    os.write(fd, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    return fd


def release_run_lock(fd: int, path: Path) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        path.unlink()
    except OSError:
        pass


def lock_stale_after_seconds(args: argparse.Namespace) -> int:
    return max(
        1800,
        timeout_arg(args, "rust_timeout_seconds", 7200)
        + timeout_arg(args, "public_timeout_seconds", 1800)
        + timeout_arg(args, "sts_timeout_seconds", 3600)
        + 900,
    )


def timeout_arg(args: argparse.Namespace, name: str, default_when_unbounded: int) -> int:
    value = int(getattr(args, name, 0) or 0)
    return value if value > 0 else default_when_unbounded


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except (OSError, SystemError, ValueError):
        return False


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
