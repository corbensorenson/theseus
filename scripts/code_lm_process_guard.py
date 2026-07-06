"""Shared process/artifact guards for Code LM launchers.

The launcher layer must treat duplicate output targets as a hard runtime
hazard: two wrappers writing the same checkpoint or candidate manifest corrupt
timing evidence and waste a heavy worker slot.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


CODE_LM_CLOSURE_FLAGS = [
    "--out",
    "--checkpoint-out",
    "--private-candidate-out",
    "--public-candidate-out",
    "--rust-report-out",
]
TRAIN_ONCE_FLAGS = ["--slug", "--out", "--markdown-out"]
CHUNKED_FLAGS = ["--slug", "--out", "--markdown-out"]
CODE_LM_WORKER_TOKENS = [
    "code_lm_train_once_fanout.py",
    "code_lm_chunked_recovery.py",
    "code_lm_closure.py",
    "train-code-lm-closure",
    "generate-code-lm-closure-fanout",
    "train-sts-parallel-decoder",
    "train-code-ranker",
]
CODE_LM_WORKER_PROCESS_PATTERN = "|".join(CODE_LM_WORKER_TOKENS)
EXECUTE_GATED_WRAPPERS = [
    "code_lm_train_once_fanout.py",
    "code_lm_chunked_recovery.py",
]


def duplicate_artifact_processes(args: argparse.Namespace, root: Path, *, current_pid: int | None = None) -> list[dict[str, Any]]:
    current = closure_artifact_fingerprint_from_args(args, root)
    if not current:
        return []
    duplicates: list[dict[str, Any]] = []
    for row in windows_code_lm_process_rows("code_lm_closure.py"):
        pid = int(row.get("pid") or 0)
        if pid <= 0 or pid == (current_pid or os.getpid()):
            continue
        command = str(row.get("command") or "")
        if process_artifact_fingerprint(command, root) != current:
            continue
        duplicates.append(
            {
                "pid": pid,
                "name": row.get("name"),
                "command_preview": command[:360],
                "fingerprint": current[:240],
            }
        )
    return duplicates


def duplicate_code_lm_artifact_targets(rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    rows = logical_code_lm_process_rows(rows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        fingerprint = process_artifact_fingerprint(str(row.get("command") or ""), root)
        if fingerprint:
            groups.setdefault(fingerprint, []).append(row)

    duplicates: list[dict[str, Any]] = []
    for fingerprint, grouped in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(grouped) <= 1:
            continue
        duplicates.append(
            {
                "fingerprint": fingerprint[:240],
                "process_count": len(grouped),
                "pids": [row.get("pid") for row in grouped],
                "names": [row.get("name") for row in grouped],
                "command_previews": [str(row.get("command_preview") or row.get("command") or "")[:360] for row in grouped[:3]],
                "policy": "same Code LM wrapper/output artifacts should not run from multiple launchers",
            }
        )
    return duplicates


def is_code_lm_worker_command(command: str) -> bool:
    """Return true only for actual Code LM closure/fanout workers.

    Generic cargo/rustc/symliquid-cli processes are heavy, but they are not all
    Code LM workers. Keeping this predicate narrow prevents unrelated CUDA
    ablations from freezing the train-once/fanout control plane.
    """
    lowered = str(command or "").lower()
    if "get-ciminstance win32_process" in lowered:
        return False
    for wrapper in EXECUTE_GATED_WRAPPERS:
        if wrapper in lowered and "--execute" not in lowered:
            return False
    return any(token in lowered for token in CODE_LM_WORKER_TOKENS)


def windows_active_code_lm_process_rows() -> list[dict[str, Any]]:
    rows = [
        row
        for row in windows_code_lm_process_rows(CODE_LM_WORKER_PROCESS_PATTERN)
        if is_code_lm_worker_command(str(row.get("command") or row.get("command_preview") or ""))
    ]
    return logical_code_lm_process_rows(rows)


def logical_code_lm_process_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse Windows venv redirector parent/child pairs into one logical worker.

    On Windows, launching ``.venv\\Scripts\\python.exe script.py ...`` can leave
    a short-lived redirector parent plus the real base-Python child with the
    same script and arguments. Counting both as separate Code LM workers creates
    false duplicate-artifact blockers and can starve the actual CUDA path.
    """
    by_pid: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            pid = int(row.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid > 0:
            by_pid[pid] = row

    parent_pids_to_drop: set[int] = set()
    for row in rows:
        try:
            pid = int(row.get("pid") or 0)
            parent_pid = int(row.get("parent_pid") or 0)
        except (TypeError, ValueError):
            continue
        if pid <= 0 or parent_pid <= 0 or parent_pid not in by_pid:
            continue
        parent = by_pid[parent_pid]
        if canonical_worker_command_key(parent.get("command") or "") == canonical_worker_command_key(row.get("command") or ""):
            parent_pids_to_drop.add(parent_pid)

    logical: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, str]] = set()
    for row in rows:
        try:
            pid = int(row.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid in parent_pids_to_drop:
            continue
        key = (pid, canonical_worker_command_key(row.get("command") or ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        logical.append(row)
    return logical


def canonical_worker_command_key(command: str) -> str:
    try:
        tokens = shlex.split(str(command or ""), posix=True)
    except ValueError:
        tokens = str(command or "").split()
    if tokens:
        executable = tokens[0].replace("\\", "/").lower()
        if executable.endswith("/python.exe") or executable.endswith("/python") or executable in {"python", "python.exe"}:
            tokens = tokens[1:]
    return " ".join(token.replace("\\", "/").lower() for token in tokens)


def closure_artifact_fingerprint_from_args(args: argparse.Namespace, root: Path) -> str:
    return artifact_fingerprint(
        "python_code_lm_closure",
        {
            "--out": getattr(args, "out", ""),
            "--checkpoint-out": getattr(args, "checkpoint_out", ""),
            "--private-candidate-out": getattr(args, "private_candidate_out", ""),
            "--public-candidate-out": getattr(args, "public_candidate_out", ""),
            "--rust-report-out": getattr(args, "rust_report_out", ""),
        },
        root,
    )


def process_artifact_fingerprint(command: str, root: Path) -> str:
    lowered = command.lower()
    if "code_lm_closure.py" in lowered:
        return artifact_fingerprint("python_code_lm_closure", flag_values(command, CODE_LM_CLOSURE_FLAGS), root)
    if "code_lm_train_once_fanout.py" in lowered:
        return artifact_fingerprint("python_code_lm_train_once_fanout", flag_values(command, TRAIN_ONCE_FLAGS), root)
    if "code_lm_chunked_recovery.py" in lowered:
        return artifact_fingerprint("python_code_lm_chunked_recovery", flag_values(command, CHUNKED_FLAGS), root)
    return ""


def artifact_fingerprint(kind: str, values: dict[str, Any], root: Path) -> str:
    parts = [f"{key}={normalize_arg_value(str(value or ''), root)}" for key, value in values.items()]
    parts = [part for part in parts if not part.endswith("=")]
    if not parts:
        return ""
    return kind + "|" + ";".join(parts)


def flag_values(command: str, flags: list[str]) -> dict[str, str]:
    return {flag: extract_flag_value(command, flag) for flag in flags}


def extract_flag_value(command: str, flag: str) -> str:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1].strip("\"'")
        if token.startswith(flag + "="):
            return token.split("=", 1)[1].strip("\"'")
    return ""


def normalize_arg_value(value: str, root: Path) -> str:
    cleaned = str(value or "").strip().strip("\"'")
    if not cleaned:
        return ""
    if cleaned.startswith("-"):
        return cleaned.lower()
    path = Path(cleaned)
    normalized = path if path.is_absolute() else root / path
    return str(normalized).replace("\\", "/").lower()


def windows_code_lm_process_rows(pattern: str) -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.Name -match 'python|symliquid|cargo|rustc' -and $_.CommandLine -match '{pattern}' }} | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=5)
        payload = json.loads(result.stdout or "[]")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    rows = payload if isinstance(payload, list) else [payload]
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        command_line = str(row.get("CommandLine") or "")
        if "Get-CimInstance Win32_Process" in command_line:
            continue
        normalized.append(
            {
                "pid": int(row.get("ProcessId") or 0),
                "parent_pid": int(row.get("ParentProcessId") or 0),
                "name": row.get("Name"),
                "command": command_line,
                "command_preview": command_line[:360],
            }
        )
    return normalized
