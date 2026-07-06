"""SparkStream daemon.

Runs autonomy cycles on an interval. The daemon is intentionally boring:
bounded commands, visible status, append-only ledger entries, and a stop flag.
Use the dashboard for interactive control during development.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "autonomy_policy.json"
STATUS_PATH = ROOT / "reports" / "sparkstream_status.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--profile", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--forbid-teacher", action="store_true")
    parser.add_argument("--forbid-network-fetch", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Disable teacher escalation and network fetches for daemon cycles.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--duration-hours", type=float, default=0.0)
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    paths = daemon_paths(policy)
    profile = args.profile or policy.get("default_profile", "inner_loop")
    teacher_forbidden = bool(args.offline or args.forbid_teacher)
    network_forbidden = bool(args.offline or args.forbid_network_fetch)
    allow_teacher = False if teacher_forbidden else bool(args.allow_teacher or policy.get("allow_teacher_by_default", False))
    allow_network_fetch = False if network_forbidden else bool(args.allow_network_fetch or policy.get("allow_network_fetch_by_default", False))
    max_cycles = args.max_cycles
    if max_cycles is None:
        max_cycles = int(policy.get("max_cycles_per_daemon_run", 0))
    interval = int(policy.get("cycle_interval_seconds", 300))
    stop_flag = paths["stop_flag"]
    pause_flag = paths["pause_flag"]
    ledger_path = paths["ledger_path"]
    stop_flag.unlink(missing_ok=True)
    pause_flag.unlink(missing_ok=True)
    started_ts = time.time()
    end_ts = started_ts + args.duration_hours * 3600 if args.duration_hours > 0 else None

    cycle = 0
    daemon_id = f"daemon_{int(started_ts * 1000)}"
    append_jsonl(
        ledger_path,
        daemon_event(
            daemon_id,
            "started",
            cycle,
            profile,
            {
                "execute": args.execute,
                "allow_teacher": allow_teacher,
                "allow_network_fetch": allow_network_fetch,
                "offline": bool(args.offline),
                "teacher_forbidden": teacher_forbidden,
                "network_fetch_forbidden": network_forbidden,
                "interval_seconds": interval,
                "max_cycles": max_cycles,
                "duration_hours": args.duration_hours,
            },
        ),
    )
    write_status("starting", profile, "SparkStream daemon starting.", cycle, daemon_id=daemon_id)
    while True:
        if stop_flag.exists():
            write_status("stopped", profile, "Stop flag detected.", cycle, daemon_id=daemon_id)
            append_jsonl(ledger_path, daemon_event(daemon_id, "stopped", cycle, profile, {"reason": "stop_flag"}))
            return 0
        if end_ts and time.time() >= end_ts:
            write_status("idle", profile, "Configured duration reached.", cycle, daemon_id=daemon_id)
            append_jsonl(ledger_path, daemon_event(daemon_id, "completed", cycle, profile, {"reason": "duration"}))
            return 0
        if args.once and cycle >= 1:
            write_status("idle", profile, "One-shot daemon complete.", cycle, daemon_id=daemon_id)
            append_jsonl(ledger_path, daemon_event(daemon_id, "completed", cycle, profile, {"reason": "once"}))
            return 0
        if max_cycles and cycle >= max_cycles:
            write_status("idle", profile, "Configured max cycles reached.", cycle, daemon_id=daemon_id)
            append_jsonl(ledger_path, daemon_event(daemon_id, "completed", cycle, profile, {"reason": "max_cycles"}))
            return 0
        while pause_flag.exists():
            if stop_flag.exists():
                write_status("stopped", profile, "Stop flag detected while paused.", cycle, daemon_id=daemon_id)
                append_jsonl(ledger_path, daemon_event(daemon_id, "stopped", cycle, profile, {"reason": "stop_flag_paused"}))
                return 0
            write_status("paused", profile, "Pause flag detected. Waiting to resume.", cycle, daemon_id=daemon_id)
            time.sleep(2)
        if policy.get("daemon", {}).get("policy_reload_each_cycle", True):
            policy = read_json(ROOT / args.policy)
            teacher_forbidden = bool(args.offline or args.forbid_teacher)
            network_forbidden = bool(args.offline or args.forbid_network_fetch)
            allow_teacher = False if teacher_forbidden else bool(args.allow_teacher or policy.get("allow_teacher_by_default", False))
            allow_network_fetch = False if network_forbidden else bool(args.allow_network_fetch or policy.get("allow_network_fetch_by_default", False))
            interval = int(policy.get("cycle_interval_seconds", interval))
            paths = daemon_paths(policy)
            stop_flag = paths["stop_flag"]
            pause_flag = paths["pause_flag"]
            ledger_path = paths["ledger_path"]

        cycle += 1
        write_status("cycle_start", profile, f"Starting autonomy cycle {cycle}.", cycle, daemon_id=daemon_id)
        append_jsonl(ledger_path, daemon_event(daemon_id, "cycle_start", cycle, profile, {}))
        command = [
            sys.executable,
            "scripts/autonomy_cycle.py",
            "--policy",
            args.policy,
            "--profile",
            profile,
            "--out",
            "reports/autonomy_cycle_last.json",
            *(["--execute"] if args.execute else []),
            *(["--allow-teacher"] if allow_teacher else []),
            *(["--allow-network-fetch"] if allow_network_fetch else []),
            *(["--forbid-teacher"] if teacher_forbidden else []),
            *(["--forbid-network-fetch"] if network_forbidden else []),
            *(["--offline"] if args.offline else []),
        ]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        runtime_ms = int((time.perf_counter() - started) * 1000)
        write_status(
            "cycle_complete" if result.returncode == 0 else "cycle_failed",
            profile,
            f"Cycle {cycle} finished with returncode {result.returncode}.",
            cycle,
            daemon_id=daemon_id,
            extra={
                "returncode": result.returncode,
                "runtime_ms": runtime_ms,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            },
        )
        append_jsonl(
            ledger_path,
            daemon_event(
                daemon_id,
                "cycle_complete" if result.returncode == 0 else "cycle_failed",
                cycle,
                profile,
                {"returncode": result.returncode, "runtime_ms": runtime_ms},
            ),
        )
        watchdog = run_watchdog(args.execute)
        append_jsonl(
            ledger_path,
            daemon_event(
                daemon_id,
                "watchdog",
                cycle,
                profile,
                {
                    "returncode": watchdog.get("returncode"),
                    "trigger_state": watchdog.get("trigger_state"),
                    "applied_actions": watchdog.get("applied_actions", []),
                },
            ),
        )
        if args.once:
            return result.returncode
        sleep_until = time.time() + interval
        while time.time() < sleep_until:
            if stop_flag.exists():
                write_status("stopped", profile, "Stop flag detected.", cycle, daemon_id=daemon_id)
                append_jsonl(ledger_path, daemon_event(daemon_id, "stopped", cycle, profile, {"reason": "stop_flag"}))
                return 0
            if pause_flag.exists():
                break
            write_status(
                "sleeping",
                profile,
                f"Next autonomy cycle in {max(0, int(sleep_until - time.time()))}s.",
                cycle,
                daemon_id=daemon_id,
                extra={"next_cycle_utc": datetime.fromtimestamp(sleep_until, timezone.utc).isoformat()},
            )
            time.sleep(1)


def write_status(
    phase: str,
    profile: str,
    message: str,
    cycle: int,
    *,
    daemon_id: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "policy": "sparkstream_daemon_status_v0",
        "updated_utc": now(),
        "daemon_id": daemon_id,
        "phase": phase,
        "profile": profile,
        "message": message,
        "cycle": cycle,
    }
    if extra:
        payload.update(extra)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def daemon_paths(policy: dict[str, Any]) -> dict[str, Path]:
    daemon = policy.get("daemon") or {}
    return {
        "stop_flag": ROOT / daemon.get("stop_flag", "reports/sparkstream_stop.flag"),
        "pause_flag": ROOT / daemon.get("pause_flag", "reports/sparkstream_pause.flag"),
        "ledger_path": ROOT / daemon.get("ledger_path", "reports/sparkstream_daemon_ledger.jsonl"),
    }


def daemon_event(
    daemon_id: str,
    event: str,
    cycle: int,
    profile: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "daemon_id": daemon_id,
        "event": event,
        "created_utc": now(),
        "cycle": cycle,
        "profile": profile,
        **extra,
    }


def run_watchdog(apply_fixes: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/autonomy_watchdog.py",
        "--out",
        "reports/autonomy_watchdog.json",
    ]
    if apply_fixes:
        command.append("--fix")
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    report = read_json(ROOT / "reports" / "autonomy_watchdog.json")
    return {
        "returncode": result.returncode,
        "trigger_state": report.get("trigger_state"),
        "applied_actions": [
            item.get("kind")
            for item in report.get("recommended_actions", [])
            if item.get("applied")
        ],
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
