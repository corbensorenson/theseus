#!/usr/bin/env python3
"""Report duplicate/missing Theseus service processes.

The long-running autonomy stack should be boring: one dashboard, one daemon,
one learning supervisor, one artifact sync loop, and one hive node agent per
host. This script is diagnostic-only by default; it never kills processes.
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
DEFAULT_OUT = REPORTS / "service_process_hygiene.json"
DEFAULT_MARKDOWN = REPORTS / "service_process_hygiene.md"

sys.path.insert(0, str(ROOT / "scripts"))
try:
    import report_evidence_store  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    report_evidence_store = None  # type: ignore


SERVICE_SPECS = [
    {
        "service": "sparkstream_daemon",
        "needles": ["sparkstream_daemon.py"],
        "expected_min": 1,
        "expected_max": 1,
        "severity": "YELLOW",
    },
    {
        "service": "hive_node",
        "needles": ["hive_node.py"],
        "expected_min": 1,
        "expected_max": 1,
        "severity": "YELLOW",
    },
    {
        "service": "vacation_learning_supervisor",
        "needles": ["vacation_mode_supervisor.py"],
        "expected_min": 1,
        "expected_max": 1,
        "severity": "YELLOW",
    },
    {
        "service": "hive_artifact_sync",
        "needles": ["hive_artifact_sync.py"],
        "expected_min": 0,
        "expected_max": 1,
        "severity": "YELLOW",
    },
    {
        "service": "dashboard",
        "needles": ["sparkstream_dashboard.py", "hive_operator_dashboard.py"],
        "expected_min": 1,
        "expected_max": 1,
        "severity": "YELLOW",
    },
]


def service_needles(service: str) -> list[str]:
    for spec in SERVICE_SPECS:
        if spec.get("service") == service:
            return [str(item) for item in spec.get("needles", [])]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--fix", action="store_true", help="Start missing required services. Duplicate stopping requires --dedupe.")
    parser.add_argument("--dedupe", action="store_true", help="With --fix, stop older duplicate allowlisted services. Omit for start-only repair.")
    args = parser.parse_args()

    started = time.perf_counter()
    processes = list_processes()
    service_rows = [classify_service(spec, processes) for spec in SERVICE_SPECS]
    fix_actions: list[dict[str, Any]] = []
    if args.fix:
        fix_actions = apply_fixes(service_rows, dedupe=bool(args.dedupe))
        if fix_actions:
            time.sleep(4.0)
            processes = list_processes()
            service_rows = [classify_service(spec, processes) for spec in SERVICE_SPECS]
    duplicate_rows = [row for row in service_rows if row["duplicate_count"] > 0]
    missing_required = [row for row in service_rows if row["missing_required"]]
    launch_guards = duplicate_prevention_guards()
    checks = [
        gate("process_snapshot_available", bool(processes), len(processes), "YELLOW"),
        gate("no_duplicate_service_processes", not duplicate_rows, [row["service"] for row in duplicate_rows], "YELLOW"),
        gate("required_service_processes_present", not missing_required, [row["service"] for row in missing_required], "YELLOW"),
        gate("duplicate_launch_paths_guarded", all(row["guarded"] for row in launch_guards), launch_guards, "YELLOW"),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in checks) else "YELLOW"
    report = {
        "policy": "project_theseus_service_process_hygiene_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": "Detect duplicate or missing long-running Theseus service processes without killing anything.",
        "summary": {
            "observed_process_count": len(processes),
            "service_count": len(service_rows),
            "duplicate_service_count": len(duplicate_rows),
            "missing_required_service_count": len(missing_required),
            "duplicate_launch_guard_count": sum(1 for row in launch_guards if row["guarded"]),
            "duplicate_launch_guard_expected_count": len(launch_guards),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "checks": checks,
        "services": service_rows,
        "duplicate_prevention_guards": launch_guards,
        "fix_requested": bool(args.fix),
        "dedupe_requested": bool(args.dedupe),
        "fix_actions": fix_actions,
        "next_actions": next_actions(duplicate_rows, missing_required),
    }
    out = resolve(args.out)
    md = resolve(args.markdown_out)
    write_json(out, report)
    write_text(md, render_markdown(report))
    if report_evidence_store is not None:
        try:
            report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, out, payload=report)
        except Exception:
            pass
    print(json.dumps(report, indent=2))
    return 0


def list_processes() -> list[dict[str, Any]]:
    if sys.platform.startswith("win"):
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 2",
        ]
    else:
        command = ["ps", "-eo", "pid=,ppid=,args="]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    except Exception as exc:
        return [{"pid": 0, "command_line": f"process_snapshot_error: {exc}"}]
    if result.returncode != 0:
        return [{"pid": 0, "command_line": f"process_snapshot_failed: {result.stderr[-300:]}"}]
    if sys.platform.startswith("win"):
        try:
            raw = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            raw = []
        if isinstance(raw, dict):
            raw = [raw]
        rows = []
        items = raw if isinstance(raw, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            command_line = str(item.get("CommandLine") or "")
            if not command_line:
                continue
            rows.append(
                {
                    "pid": int(item.get("ProcessId") or 0),
                    "parent_pid": int(item.get("ParentProcessId") or 0),
                    "command_line": command_line,
                }
            )
        return rows
    rows = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, rest = line.partition(" ")
        parent_pid, _, command_line = rest.strip().partition(" ")
        rows.append(
            {
                "pid": int(pid) if pid.isdigit() else 0,
                "parent_pid": int(parent_pid) if parent_pid.isdigit() else 0,
                "command_line": command_line,
            }
        )
    return rows


def classify_service(spec: dict[str, Any], processes: list[dict[str, Any]]) -> dict[str, Any]:
    needles = [str(item).lower() for item in spec["needles"]]
    raw_matches = [
        {"pid": row["pid"], "parent_pid": row.get("parent_pid", 0), "command_line": truncate(row["command_line"])}
        for row in processes
        if any(needle in str(row.get("command_line") or "").lower() for needle in needles)
    ]
    matches = collapse_windows_venv_launcher_pairs(raw_matches)
    expected_min = int(spec["expected_min"])
    expected_max = int(spec["expected_max"])
    count = len(matches)
    return {
        "service": spec["service"],
        "expected_min": expected_min,
        "expected_max": expected_max,
        "observed_count": count,
        "duplicate_count": max(0, count - expected_max),
        "missing_required": count < expected_min,
        "matches": matches[:8],
        "severity": spec.get("severity", "YELLOW"),
    }


def next_actions(duplicates: list[dict[str, Any]], missing: list[dict[str, Any]]) -> list[str]:
    actions = []
    if duplicates:
        actions.append("inspect duplicate service rows and keep the newest healthy lease before unattended runs")
    if missing:
        actions.append("restart missing service through the normal scheduled-task/service launcher")
    if not actions:
        actions.append("service process hygiene is clean")
    return actions


def collapse_windows_venv_launcher_pairs(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not matches:
        return []
    by_pid = {int(row.get("pid") or 0): row for row in matches}
    launcher_parent_pids: set[int] = set()
    for row in matches:
        parent_pid = int(row.get("parent_pid") or 0)
        parent = by_pid.get(parent_pid)
        if not parent:
            continue
        parent_command = str(parent.get("command_line") or "").lower()
        child_command = str(row.get("command_line") or "").lower()
        if ".venv-puffer" in parent_command and same_python_script_command(parent_command, child_command):
            launcher_parent_pids.add(parent_pid)
    collapsed = [row for row in matches if int(row.get("pid") or 0) not in launcher_parent_pids]
    return collapsed or matches


def same_python_script_command(left: str, right: str) -> bool:
    scripts = (
        "sparkstream_daemon.py",
        "hive_node.py",
        "vacation_mode_supervisor.py",
        "sparkstream_dashboard.py",
    )
    return any(script in left and script in right for script in scripts)


def duplicate_prevention_guards() -> list[dict[str, Any]]:
    dashboard = read_text(ROOT / "scripts" / "sparkstream_dashboard.py")
    return [
        launch_guard_row("dashboard_start_daemon", "sparkstream_daemon", dashboard),
        launch_guard_row("dashboard_start_hive_node", "hive_node", dashboard),
        launch_guard_row("dashboard_start_hive_relay", "hive_relay", dashboard),
    ]


def launch_guard_row(name: str, singleton_name: str, source: str) -> dict[str, Any]:
    guarded = (
        "SINGLETON_JOB_NEEDLES" in source
        and f'"{singleton_name}"' in source
        and "existing_singleton_processes" in source
        and '"status": "already_running"' in source
    )
    return {
        "name": name,
        "service": singleton_name,
        "guarded": bool(guarded),
        "evidence": "dashboard start_job process-level singleton guard" if guarded else "guard not found",
    }


def apply_fixes(service_rows: list[dict[str, Any]], *, dedupe: bool) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    hive_stack_started = False
    sparkstream_stack_started = False
    vacation_supervisor_started = False
    for row in service_rows:
        service = str(row.get("service") or "")
        matches = row.get("matches") if isinstance(row.get("matches"), list) else []
        if dedupe and int(row.get("duplicate_count") or 0) > 0:
            keep = preferred_service_match(matches)
            for match in matches:
                pid = int(match.get("pid") or 0)
                if not pid or pid == int(keep.get("pid") or 0):
                    continue
                stopped = stop_process(pid)
                actions.append(
                    {
                        "action": "stop_duplicate_service_process",
                        "service": service,
                        "pid": pid,
                        "kept_pid": int(keep.get("pid") or 0),
                        "applied": stopped,
                    }
                )
        if row.get("missing_required") and service == "sparkstream_daemon":
            launched = {"ok": True, "skipped": "sparkstream_daemon_already_started_this_pass"} if sparkstream_stack_started else start_sparkstream_daemon()
            sparkstream_stack_started = True
            actions.append(
                {
                    "action": "start_missing_sparkstream_daemon",
                    "service": service,
                    "applied": bool(launched.get("ok")),
                    **launched,
                }
            )
        if row.get("missing_required") and service == "hive_node":
            launched = {"ok": True, "skipped": "hive_node_already_started_this_pass"} if hive_stack_started else start_hive_node()
            hive_stack_started = True
            actions.append(
                {
                    "action": "start_missing_hive_or_dashboard",
                    "service": service,
                    "applied": bool(launched.get("ok")),
                    **launched,
                }
            )
        if row.get("missing_required") and service == "dashboard":
            launched = start_dashboard()
            actions.append(
                {
                    "action": "start_missing_dashboard",
                    "service": service,
                    "applied": bool(launched.get("ok")),
                    **launched,
                }
            )
        if row.get("missing_required") and service == "vacation_learning_supervisor":
            launched = (
                {"ok": True, "skipped": "vacation_supervisor_already_started_this_pass"}
                if vacation_supervisor_started
                else start_learning_launch_supervisor()
            )
            vacation_supervisor_started = True
            actions.append(
                {
                    "action": "start_missing_vacation_learning_supervisor",
                    "service": service,
                    "applied": bool(launched.get("ok")),
                    **launched,
                }
            )
    return actions


def preferred_service_match(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {}
    return sorted(
        matches,
        key=lambda row: (
            ".venv-puffer" not in str(row.get("command_line") or "").lower(),
            -int(row.get("pid") or 0),
        ),
    )[0]


def stop_process(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        command = ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"]
    else:
        command = ["kill", "-TERM", str(pid)]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=20)
    except Exception:
        return False
    return result.returncode == 0


def start_learning_launch_supervisor() -> dict[str, Any]:
    existing = existing_processes(service_needles("vacation_learning_supervisor"))
    if existing:
        return {
            "ok": True,
            "started_new_process": False,
            "skipped": "existing_vacation_learning_supervisor",
            "existing_processes": existing[:3],
        }
    python = project_python()
    command = [
        str(python),
        "scripts/learning_launch_supervisor.py",
        "--execute",
        "--allow-teacher",
        "--start-services",
        "--cycles",
        "0",
        "--sleep-seconds",
        "300",
        "--max-actions-per-cycle",
        "1",
        "--clear-stale-flags",
        "--out",
        "reports/learning_launch_supervisor.json",
        "--markdown-out",
        "reports/learning_launch_supervisor.md",
    ]
    try:
        if sys.platform.startswith("win"):
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            )
        else:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": command}
    return {"ok": True, "started_new_process": True, "pid": proc.pid, "command": command}


def start_sparkstream_daemon() -> dict[str, Any]:
    existing = existing_processes(service_needles("sparkstream_daemon"))
    if existing:
        return {
            "ok": True,
            "started_new_process": False,
            "skipped": "existing_sparkstream_daemon",
            "existing_processes": existing[:3],
        }
    command = [
        str(project_python()),
        "scripts/sparkstream_daemon.py",
        "--profile",
        "inner_loop",
        "--execute",
        "--allow-teacher",
        "--allow-network-fetch",
    ]
    return launch_background(command)


def start_hive_node() -> dict[str, Any]:
    existing = existing_processes(service_needles("hive_node"))
    if existing:
        return {
            "ok": True,
            "started_new_process": False,
            "skipped": "existing_hive_node",
            "existing_processes": existing[:3],
        }
    command = [
        str(project_python()),
        "scripts/hive_node.py",
        "daemon",
        "--port",
        "8791",
    ]
    return launch_background(command)


def start_dashboard() -> dict[str, Any]:
    existing = existing_processes(service_needles("dashboard"))
    if existing:
        return {
            "ok": True,
            "started_new_process": False,
            "skipped": "existing_dashboard",
            "existing_processes": existing[:3],
        }
    command = [
        str(project_python()),
        "scripts/sparkstream_dashboard.py",
        "--host",
        "0.0.0.0",
        "--port",
        "8787",
    ]
    return launch_background(command)


def launch_background(command: list[str]) -> dict[str, Any]:
    try:
        if sys.platform.startswith("win"):
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            )
        else:
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": command}
    return {"ok": True, "started_new_process": True, "pid": proc.pid, "command": command}


def existing_processes(needles: list[str]) -> list[dict[str, Any]]:
    lowered = [needle.lower() for needle in needles if needle]
    if not lowered:
        return []
    matches = []
    for row in list_processes():
        command_line = str(row.get("command_line") or "")
        if any(needle in command_line.lower() for needle in lowered):
            matches.append({"pid": int(row.get("pid") or 0), "command_line": truncate(command_line)})
    return matches


def project_python() -> Path:
    candidate = ROOT / ".venv-puffer" / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence, "severity": severity}


def truncate(value: str, limit: int = 420) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Service Process Hygiene",
        "",
        f"- State: {report['trigger_state']}",
        f"- Duplicate services: {report['summary']['duplicate_service_count']}",
        f"- Missing required services: {report['summary']['missing_required_service_count']}",
        f"- Duplicate launch guards: {report['summary'].get('duplicate_launch_guard_count', 0)}/{report['summary'].get('duplicate_launch_guard_expected_count', 0)}",
        "",
        "## Services",
    ]
    for row in report["services"]:
        lines.append(
            f"- {row['service']}: observed={row['observed_count']} expected={row['expected_min']}-{row['expected_max']} duplicates={row['duplicate_count']}"
        )
    lines.append("")
    lines.append("## Duplicate Prevention Guards")
    for row in report.get("duplicate_prevention_guards", []):
        lines.append(f"- {row['name']}: guarded={row['guarded']} evidence={row['evidence']}")
    lines.append("")
    lines.append("## Next Actions")
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
