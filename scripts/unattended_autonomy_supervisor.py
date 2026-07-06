"""Unattended VIEA + Theseus + SymLiquid supervisor.

This is the sleep/vacation runner. It does not grant new permissions by
itself; it repeatedly executes the existing watchdog, VIEA spine, one approved
VIEA action, residual reading, optional teacher-as-architect experiment closure,
and hive scheduler refresh under pause/stop flags.
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
REPORTS = ROOT / "reports"
LEDGER = REPORTS / "unattended_autonomy_supervisor_ledger.jsonl"
STOP_FLAGS = [REPORTS / "sparkstream_stop.flag", REPORTS / "unattended_autonomy_stop.flag"]
PAUSE_FLAGS = [REPORTS / "sparkstream_pause.flag", REPORTS / "viea_action_executor_pause.flag"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=1, help="0 means run until a stop flag is seen.")
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--max-actions-per-cycle", type=int, default=1)
    parser.add_argument("--max-steps-per-action", type=int, default=1)
    parser.add_argument("--action-timeout-seconds", type=int, default=21600)
    parser.add_argument("--teacher-timeout-seconds", type=int, default=21600)
    parser.add_argument("--out", default="reports/unattended_autonomy_supervisor.json")
    parser.add_argument("--markdown-out", default="reports/unattended_autonomy_supervisor.md")
    args = parser.parse_args()

    started = time.perf_counter()
    cycle_count = 0
    last_cycle: dict[str, Any] = {}
    while True:
        if stop_requested():
            break
        if args.cycles and cycle_count >= max(0, int(args.cycles)):
            break
        if paused():
            last_cycle = {
                "cycle": cycle_count + 1,
                "status": "paused",
                "paused_flags": [rel(path) for path in PAUSE_FLAGS if path.exists()],
                "created_utc": now(),
            }
            append_jsonl(LEDGER, last_cycle)
            write_snapshot(args, started, cycle_count, last_cycle, trigger_state="YELLOW")
            sleep_or_stop(max(1, int(args.sleep_seconds)))
            continue
        cycle_count += 1
        last_cycle = run_cycle(cycle_count, args=args)
        append_jsonl(LEDGER, last_cycle)
        write_snapshot(args, started, cycle_count, last_cycle, trigger_state=cycle_trigger(last_cycle))
        if last_cycle.get("trigger_state") == "RED":
            break
        if args.cycles and cycle_count >= max(0, int(args.cycles)):
            break
        sleep_or_stop(max(1, int(args.sleep_seconds)))

    final_trigger = "RED" if last_cycle.get("trigger_state") == "RED" else ("YELLOW" if stop_requested() or paused() else "GREEN")
    write_snapshot(args, started, cycle_count, last_cycle, trigger_state=final_trigger)
    return 2 if final_trigger == "RED" else 0


def run_cycle(cycle: int, *, args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.perf_counter() + max(60, int(args.action_timeout_seconds) + 3600)
    closure_report = latest_closure_report_path()
    steps = [
        step(
            "autonomy_watchdog",
            [sys.executable, "scripts/autonomy_watchdog.py", "--fix", "--out", "reports/autonomy_watchdog.json"],
            timeout=1800,
            allow_failure=True,
        ),
        step(
            "viea_autonomy_spine",
            [
                sys.executable,
                "scripts/viea_autonomy_spine.py",
                "--max-steps",
                "64",
                "--timeout-seconds",
                "1800",
                "--out",
                "reports/viea_autonomy_spine.json",
                "--markdown-out",
                "reports/viea_autonomy_spine.md",
            ],
            timeout=1800,
            allow_failure=True,
        ),
    ]
    action_command = [
        sys.executable,
        "scripts/viea_action_executor.py",
        "--resume",
        "--max-actions",
        str(max(1, int(args.max_actions_per_cycle))),
        "--max-steps",
        str(max(1, int(args.max_steps_per_action))),
        "--timeout-seconds",
        str(max(60, int(args.action_timeout_seconds))),
        "--out",
        "reports/viea_action_executor.json",
        "--markdown-out",
        "reports/viea_action_executor.md",
    ]
    if args.execute:
        action_command.append("--execute")
    if args.allow_teacher:
        action_command.append("--allow-teacher")
    steps.append(step("viea_action_executor", action_command, timeout=max(60, int(args.action_timeout_seconds)), allow_failure=False))
    steps.append(
        step(
            "broad_transfer_residual_reader",
            [
                sys.executable,
                "scripts/broad_transfer_residual_reader.py",
                "--closure-report",
                closure_report,
                "--out",
                "reports/broad_transfer_residual_reader.json",
                "--markdown-out",
                "reports/broad_transfer_residual_reader.md",
            ],
            timeout=300,
            allow_failure=True,
        )
    )
    rows = []
    for item in steps:
        if stop_requested() or time.perf_counter() >= deadline:
            rows.append({**item, "returncode": 124, "error": "cycle_deadline_or_stop", "runtime_ms": 0})
            break
        row = run_step(item)
        rows.append(row)
        if int(row.get("returncode") or 0) != 0 and not row.get("allow_failure"):
            break
    teacher_steps = teacher_steps_for_residual(args)
    for item in teacher_steps:
        if stop_requested() or time.perf_counter() >= deadline:
            rows.append({**item, "returncode": 124, "error": "cycle_deadline_or_stop", "runtime_ms": 0})
            break
        row = run_step(item) if item.get("command") else {**item, "returncode": 0, "runtime_ms": 0, "stdout_tail": "", "stderr_tail": ""}
        rows.append(row)
    hive_step = step(
        "hive_scheduler",
        [sys.executable, "scripts/hive_scheduler.py", "--out", "reports/hive_scheduler.json"],
        timeout=300,
        allow_failure=True,
    )
    if not stop_requested() and time.perf_counter() < deadline:
        rows.append(run_step(hive_step))
    hard_failures = [row for row in rows if int(row.get("returncode") or 0) != 0 and not row.get("allow_failure")]
    return {
        "policy": "project_theseus_unattended_autonomy_cycle_v1",
        "created_utc": now(),
        "cycle": cycle,
        "trigger_state": "RED" if hard_failures else "GREEN",
        "execute_requested": bool(args.execute),
        "teacher_allowed": bool(args.allow_teacher),
        "selected_closure_report": closure_report,
        "steps": rows,
        "hard_failure_count": len(hard_failures),
        "external_inference_calls": 0,
    }


def teacher_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/teacher_architect_experiment_runner.py",
        "--execute",
        "--max-experiments",
        "1",
        "--max-steps",
        "1",
        "--timeout-seconds",
        str(max(300, int(args.teacher_timeout_seconds))),
        "--out",
        "reports/teacher_architect_experiment_runner.json",
        "--markdown-out",
        "reports/teacher_architect_experiment_runner.md",
    ]
    if args.allow_teacher:
        command.append("--allow-teacher")
    return command


def teacher_steps_for_residual(args: argparse.Namespace) -> list[dict[str, Any]]:
    residual = read_json(REPORTS / "broad_transfer_residual_reader.json", {})
    if not args.allow_teacher:
        return skipped_teacher_steps("teacher_flag_not_set")
    should_run = bool(get_path(residual, ["summary", "teacher_should_run"], False))
    if not should_run:
        reason = get_path(residual, ["summary", "wall_type"], "residual_reader_not_ready")
        return skipped_teacher_steps(f"teacher_not_ready:{reason}")
    source = residual.get("source") if isinstance(residual.get("source"), dict) else {}
    public_report = str(source.get("public_report") or "")
    public_trace = str(source.get("public_trace") or "")
    if not public_report or not public_trace:
        return skipped_teacher_steps("residual_reader_missing_public_evidence")
    return [
        step(
            "architecture_guidance_loop",
            [
                sys.executable,
                "scripts/architecture_guidance_loop.py",
                "--real-code-report",
                public_report,
                "--trace-in",
                public_trace,
                "--out",
                "reports/architecture_guidance_loop.json",
                "--markdown-out",
                "reports/architecture_guidance_loop.md",
                "--allow-teacher",
            ],
            timeout=300,
            allow_failure=True,
        ),
        step(
            "teacher_architect_experiment_runner",
            teacher_command(args),
            timeout=max(300, int(args.teacher_timeout_seconds)),
            allow_failure=True,
        ),
    ]


def skipped_teacher_steps(reason: str) -> list[dict[str, Any]]:
    return [
        {
            "name": "architecture_guidance_loop",
            "command": [],
            "timeout": 0,
            "allow_failure": True,
            "skipped": True,
            "skip_reason": reason,
        },
        {
            "name": "teacher_architect_experiment_runner",
            "command": [],
            "timeout": 0,
            "allow_failure": True,
            "skipped": True,
            "skip_reason": reason,
        },
    ]


def step(name: str, command: list[str], *, timeout: int, allow_failure: bool) -> dict[str, Any]:
    return {"name": name, "command": command, "timeout": int(timeout), "allow_failure": bool(allow_failure)}


def run_step(spec: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            spec["command"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=max(60, int(spec["timeout"])),
        )
        return {
            **spec,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            **spec,
            "returncode": 124,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": exc.stdout[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": exc.stderr[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout_safety_fuse",
        }


def write_snapshot(args: argparse.Namespace, started: float, cycles: int, last_cycle: dict[str, Any], *, trigger_state: str) -> None:
    payload = {
        "policy": "project_theseus_unattended_autonomy_supervisor_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "cycles_completed": cycles,
            "execute_requested": bool(args.execute),
            "teacher_allowed": bool(args.allow_teacher),
            "paused": paused(),
            "stop_requested": stop_requested(),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "ledger": rel(LEDGER),
        },
        "last_cycle": last_cycle,
        "rules": {
            "actions_per_cycle": "one approved VIEA action by default",
            "teacher": "proposal-only architect loop; no answer distillation or apply mode",
            "public_benchmarks": "calibration-only; never training data",
            "pause_stop": "honors sparkstream and unattended pause/stop flags",
            "hive": "scheduler refresh only; remote execution still requires hive policy and secrets",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    last = payload.get("last_cycle", {})
    lines = [
        "# Unattended Autonomy Supervisor",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- cycles_completed: `{summary.get('cycles_completed')}`",
        f"- execute_requested: `{summary.get('execute_requested')}`",
        f"- teacher_allowed: `{summary.get('teacher_allowed')}`",
        f"- last_cycle_state: `{last.get('trigger_state') or last.get('status')}`",
        "",
        "## Last Steps",
        "",
    ]
    for row in last.get("steps", []) if isinstance(last.get("steps"), list) else []:
        lines.append(f"- `{row.get('returncode')}` `{row.get('name')}` runtime_ms={row.get('runtime_ms')} error={row.get('error', '')}")
    lines.append("")
    return "\n".join(lines)


def cycle_trigger(cycle: dict[str, Any]) -> str:
    return str(cycle.get("trigger_state") or "YELLOW")


def stop_requested() -> bool:
    return any(path.exists() for path in STOP_FLAGS)


def paused() -> bool:
    return any(path.exists() for path in PAUSE_FLAGS)


def sleep_or_stop(seconds: int) -> None:
    deadline = time.time() + max(1, seconds)
    while time.time() < deadline and not stop_requested():
        time.sleep(min(5, max(0.0, deadline - time.time())))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def latest_closure_report_path() -> str:
    candidates: list[tuple[float, Path]] = []
    for path in REPORTS.glob("broad_transfer_closure_runner_source_*.json"):
        payload = read_json(path, {})
        outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
        if outputs.get("public_report") and outputs.get("public_trace"):
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
    if not candidates:
        return "reports/broad_transfer_closure_runner_source_evalplus.json"
    return rel(max(candidates, key=lambda item: item[0])[1])


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
