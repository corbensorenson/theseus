"""Launch the unattended Theseus learning loop as a durable background process.

This is a small operator wrapper around ``vacation_mode_supervisor.py``. It
does not decide what to learn. The Hive work board and high-transfer scheduler
remain the source of truth. This script only makes the launch repeatable:

* avoid duplicate unattended learning processes;
* clear stale pause/stop flags when explicitly requested;
* start Vacation Mode in the background with stable report/log paths;
* write a PID/report artifact for watchdogs and humans.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
PID_REPORT = REPORTS / "learning_launch_supervisor.json"
PID_MARKDOWN = REPORTS / "learning_launch_supervisor.md"
PID_STATE = REPORTS / "learning_launch_supervisor_pid.json"
STOP_FLAGS = [
    REPORTS / "sparkstream_stop.flag",
    REPORTS / "unattended_autonomy_stop.flag",
    REPORTS / "vacation_mode_stop.flag",
]
PAUSE_FLAGS = [
    REPORTS / "sparkstream_pause.flag",
    REPORTS / "viea_action_executor_pause.flag",
    REPORTS / "vacation_mode_pause.flag",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=0, help="0 means run until stopped.")
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--max-actions-per-cycle", type=int, default=1)
    parser.add_argument("--action-timeout-seconds", type=int, default=21600)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--start-services", action="store_true")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--clear-stale-flags", action="store_true")
    parser.add_argument("--force-new", action="store_true", help="Start even if the saved PID appears alive.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=str(PID_REPORT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(PID_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if args.clear_stale_flags:
        clear_flags(STOP_FLAGS + PAUSE_FLAGS)

    previous = read_json(PID_STATE, {})
    previous_pid = int(previous.get("pid") or 0)
    previous_alive = process_matches_command(previous_pid, "vacation_mode_supervisor.py") if previous_pid else False

    command = build_command(args)
    discovered = discover_existing_vacation_supervisors()
    discovered_pid = int(discovered[0]["pid"]) if discovered else 0
    discovered_alive = bool(discovered_pid)
    stdout_path = LOGS / "learning_launch_supervisor_stdout.log"
    stderr_path = LOGS / "learning_launch_supervisor_stderr.log"
    started = False
    pid = previous_pid if previous_alive and not args.force_new else (discovered_pid if discovered_alive and not args.force_new else None)
    error = None

    if args.dry_run:
        status = "DRY_RUN"
    elif previous_alive and not args.force_new:
        status = "ALREADY_RUNNING"
    elif discovered_alive and not args.force_new:
        status = "ALREADY_RUNNING_DISCOVERED"
    else:
        try:
            proc = launch(command, stdout_path=stdout_path, stderr_path=stderr_path)
            pid = int(proc.pid)
            started = True
            status = "STARTED"
        except Exception as exc:  # pragma: no cover - operator artifact path.
            status = "FAILED"
            error = repr(exc)

    report = {
        "policy": "project_theseus_learning_launch_supervisor_v1",
        "created_utc": now(),
        "trigger_state": "GREEN"
        if status in {"STARTED", "ALREADY_RUNNING", "ALREADY_RUNNING_DISCOVERED", "DRY_RUN"}
        else "RED",
        "status": status,
        "started_new_process": started,
        "pid": pid,
        "previous_pid": previous_pid or None,
        "previous_pid_alive": previous_alive,
        "discovered_existing_pid": discovered_pid or None,
        "discovered_existing_processes": discovered[:8],
        "duplicate_existing_process_count": max(0, len(discovered) - 1),
        "command": command,
        "cwd": str(ROOT),
        "stdout_log": str(stdout_path.relative_to(ROOT)),
        "stderr_log": str(stderr_path.relative_to(ROOT)),
        "vacation_report": "reports/vacation_mode_supervisor_overnight.json",
        "vacation_markdown": "reports/vacation_mode_supervisor_overnight.md",
        "watchdog_report": "reports/autonomy_watchdog.json",
        "board_report": "reports/hive_work_board_executor.json",
        "public_data_rule": "public_benchmarks_calibration_only",
        "teacher_mode": "proposal_only" if args.allow_teacher else "disabled",
        "stop_flags": [str(path.relative_to(ROOT)) for path in STOP_FLAGS],
        "pause_flags": [str(path.relative_to(ROOT)) for path in PAUSE_FLAGS],
        "error": error,
    }
    write_json(resolve(args.out), report)
    write_markdown(resolve(args.markdown_out), report)
    if pid:
        write_json(PID_STATE, {"pid": pid, "created_utc": report["created_utc"], "command": command})
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/vacation_mode_supervisor.py",
        "--cycles",
        str(max(0, int(args.cycles))),
        "--sleep-seconds",
        str(max(1, int(args.sleep_seconds))),
        "--max-actions-per-cycle",
        str(max(1, int(args.max_actions_per_cycle))),
        "--action-timeout-seconds",
        str(max(60, int(args.action_timeout_seconds))),
        "--out",
        "reports/vacation_mode_supervisor_overnight.json",
        "--markdown-out",
        "reports/vacation_mode_supervisor_overnight.md",
    ]
    if args.execute:
        command.append("--execute")
    if args.allow_teacher:
        command.append("--allow-teacher")
    if args.start_services:
        command.append("--start-services")
    if args.explore:
        command.append("--explore")
    if args.allow_network_fetch:
        command.append("--allow-network-fetch")
    return command


def launch(command: list[str], *, stdout_path: Path, stderr_path: Path) -> subprocess.Popen:
    stdout = stdout_path.open("ab")
    stderr = stderr_path.open("ab")
    kwargs: dict[str, Any] = {
        "cwd": ROOT,
        "stdout": stdout,
        "stderr": stderr,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


def discover_existing_vacation_supervisors() -> list[dict[str, Any]]:
    if os.name != "nt":
        return discover_existing_vacation_supervisors_posix()
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match 'vacation_mode_supervisor.py' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return []
    rows = payload if isinstance(payload, list) else [payload]
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = int(row.get("ProcessId") or 0)
        cmd = str(row.get("CommandLine") or "")
        lowered = cmd.lower()
        if "get-ciminstance" in lowered or "select-object processid" in lowered:
            continue
        if pid and "vacation_mode_supervisor.py" in cmd:
            out.append({"pid": pid, "command_preview": cmd[:500]})
    out.sort(key=lambda item: int(item.get("pid") or 0), reverse=True)
    return out


def discover_existing_vacation_supervisors_posix() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    out = []
    for line in (result.stdout or "").splitlines():
        if "vacation_mode_supervisor.py" not in line:
            continue
        pid_text, _, cmd = line.strip().partition(" ")
        if pid_text.isdigit():
            out.append({"pid": int(pid_text), "command_preview": cmd[:500]})
    out.sort(key=lambda item: int(item.get("pid") or 0), reverse=True)
    return out


def process_matches_command(pid: int, expected_command_substring: str) -> bool:
    if pid <= 0:
        return False
    expected = expected_command_substring.lower()
    if os.name == "nt":
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        f"Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" | "
                        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return False
        try:
            row = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return False
        if not isinstance(row, dict):
            return False
        return int(row.get("ProcessId") or 0) == pid and expected in str(row.get("CommandLine") or "").lower()
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True, timeout=10)
    except Exception:
        return False
    return result.returncode == 0 and expected in (result.stdout or "").lower()


def clear_flags(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Theseus Learning Launch Supervisor",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- status: `{report.get('status')}`",
        f"- pid: `{report.get('pid')}`",
        f"- teacher_mode: `{report.get('teacher_mode')}`",
        f"- vacation_report: `{report.get('vacation_report')}`",
        f"- board_report: `{report.get('board_report')}`",
        f"- stdout_log: `{report.get('stdout_log')}`",
        f"- stderr_log: `{report.get('stderr_log')}`",
        "",
        "## Command",
        "",
        "```text",
        " ".join(str(part) for part in report.get("command", [])),
        "```",
    ]
    if report.get("error"):
        lines.extend(["", "## Error", "", f"```text\n{report['error']}\n```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
