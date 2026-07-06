"""Automatically checkpoint dirty work before ATTD governance.

Uncommitted source changes are provenance hygiene, not technical debt by
themselves. This helper turns a dirty workspace into a normal git checkpoint so
ATTD can judge the code's shape instead of the user's commit timing.
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
DEFAULT_AUTONOMY_POLICY = ROOT / "configs" / "autonomy_policy.json"
DEFAULT_ATTD_POLICY = ROOT / "configs" / "attd_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_AUTONOMY_POLICY.relative_to(ROOT)))
    parser.add_argument("--attd-policy", default=str(DEFAULT_ATTD_POLICY.relative_to(ROOT)))
    parser.add_argument("--attd-report", default="reports/attd_report.json")
    parser.add_argument("--packets-out", default="reports/attd_maintenance_packets.json")
    parser.add_argument("--markdown-out", default="reports/attd_report.md")
    parser.add_argument("--runtime-out", default="reports/legacy_port_runtime_enforcement.json")
    parser.add_argument("--out", default="reports/attd_dirty_workspace_checkpoint.json")
    parser.add_argument("--message", default="")
    parser.add_argument("--rerun-attd", action="store_true")
    parser.add_argument("--rerun-runtime-enforcement", action="store_true")
    args = parser.parse_args()

    autonomy_policy = read_json(resolve(args.policy))
    attd_policy = read_json(resolve(args.attd_policy))
    attd_report = read_json(resolve(args.attd_report))
    cfg = checkpoint_config(autonomy_policy, attd_policy)
    status_before = git_status()
    report: dict[str, Any] = {
        "policy": "attd_dirty_workspace_checkpoint_v0",
        "created_utc": now(),
        "enabled": bool(cfg.get("auto_commit_dirty_workspace", True)),
        "mode": str(cfg.get("auto_commit_mode", "any_dirty_workspace")),
        "stage_untracked": bool(cfg.get("auto_stage_untracked", True)),
        "message": args.message or str(cfg.get("auto_commit_message") or "Checkpoint workspace before ATTD gate"),
        "status_before": status_before,
        "attd_before": dirty_attd_summary(attd_report),
        "commands": [],
        "status": "pending",
        "commit_hash": "",
        "reran_attd": False,
        "reran_runtime_enforcement": False,
        "external_inference_calls": 0,
    }

    should_commit, reason = should_checkpoint(report, cfg, status_before, attd_report)
    if not should_commit:
        report["status"] = "skipped"
        report["reason"] = reason
        report["status_after"] = status_before
        write_json(resolve(args.out), report)
        print(json.dumps(report, indent=2))
        return 0

    stage_command = ["git", "add", "-A"] if cfg.get("auto_stage_untracked", True) else ["git", "add", "-u"]
    stage = run(stage_command, timeout=300)
    report["commands"].append(stage)
    if stage["returncode"] != 0:
        report["status"] = "stage_failed"
        report["status_after"] = git_status()
        write_json(resolve(args.out), report)
        print(json.dumps(report, indent=2))
        return 2

    staged_files = git_lines(["git", "diff", "--cached", "--name-only"])
    report["staged_files"] = staged_files
    if not staged_files:
        report["status"] = "skipped"
        report["reason"] = "dirty_status_had_no_stageable_changes"
        report["status_after"] = git_status()
        write_json(resolve(args.out), report)
        print(json.dumps(report, indent=2))
        return 0

    commit = run(["git", "commit", "-m", report["message"]], timeout=300)
    report["commands"].append(commit)
    report["status_after_commit_attempt"] = git_status()
    if commit["returncode"] != 0:
        report["status"] = "commit_failed"
        report["status_after"] = git_status()
        write_json(resolve(args.out), report)
        print(json.dumps(report, indent=2))
        return 2

    report["status"] = "committed"
    report["commit_hash"] = git_head()
    report["status_after"] = git_status()

    if args.rerun_attd:
        attd_command = [
            sys.executable,
            "scripts/attd_analyzer.py",
            "--policy",
            str(relative_to_root(resolve(args.attd_policy))),
            "--out",
            str(relative_to_root(resolve(args.attd_report))),
            "--packets-out",
            str(relative_to_root(resolve(args.packets_out))),
            "--markdown-out",
            str(relative_to_root(resolve(args.markdown_out))),
        ]
        attd = run(attd_command, timeout=180)
        report["commands"].append(attd)
        report["reran_attd"] = True
        report["attd_after"] = dirty_attd_summary(read_json(resolve(args.attd_report)))

    if args.rerun_runtime_enforcement:
        runtime = run(
            [
                sys.executable,
                "scripts/legacy_port_runtime_enforcer.py",
                "--out",
                str(relative_to_root(resolve(args.runtime_out))),
            ],
            timeout=240,
        )
        report["commands"].append(runtime)
        report["reran_runtime_enforcement"] = True

    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0


def checkpoint_config(autonomy_policy: dict[str, Any], attd_policy: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    attd_workspace = attd_policy.get("workspace_checkpoint")
    if isinstance(attd_workspace, dict):
        cfg.update(attd_workspace)
    autonomy_attd = autonomy_policy.get("attd")
    if isinstance(autonomy_attd, dict):
        cfg.update(autonomy_attd)
    return cfg


def should_checkpoint(
    report: dict[str, Any],
    cfg: dict[str, Any],
    status: dict[str, Any],
    attd_report: dict[str, Any],
) -> tuple[bool, str]:
    if not cfg.get("auto_commit_dirty_workspace", True):
        return False, "auto_commit_dirty_workspace_disabled"
    if not status.get("git_available"):
        return False, "git_unavailable"
    if not status.get("dirty"):
        return False, "workspace_clean"
    mode = str(cfg.get("auto_commit_mode", "any_dirty_workspace"))
    if mode == "dirty_attd_red_only" and not dirty_only_attd_red(attd_report):
        return False, "attd_not_red_due_only_to_dirty_residue"
    if mode not in {"any_dirty_workspace", "dirty_attd_red_only"}:
        report["unknown_mode"] = mode
        return False, "unknown_auto_commit_mode"
    return True, "workspace_dirty"


def dirty_only_attd_red(attd_report: dict[str, Any]) -> bool:
    if not attd_report or attd_report.get("trigger_state") != "RED":
        return False
    violations = []
    for row in get_path(attd_report, ["hard_caps", "violations"], []):
        if isinstance(row, dict):
            violations.append(str(row.get("gate") or ""))
    return bool(violations) and set(violations) <= {"max_dirty_residue_score"}


def dirty_attd_summary(attd_report: dict[str, Any]) -> dict[str, Any]:
    if not attd_report:
        return {"available": False}
    violations = [
        row.get("gate")
        for row in get_path(attd_report, ["hard_caps", "violations"], [])
        if isinstance(row, dict)
    ]
    return {
        "available": True,
        "trigger_state": attd_report.get("trigger_state"),
        "attd_score": attd_report.get("attd_score"),
        "dirty_residue_score": get_path(attd_report, ["history", "dirty_residue_score"]),
        "workspace_dirty_residue_score": get_path(attd_report, ["history", "workspace_dirty_residue_score"]),
        "hard_cap_violations": violations,
        "dirty_only_red": dirty_only_attd_red(attd_report),
    }


def git_status() -> dict[str, Any]:
    result = run(["git", "status", "--porcelain=v1"], timeout=120)
    if result["returncode"] != 0:
        return {
            "git_available": False,
            "dirty": True,
            "error": result["stderr_tail"] or result["stdout_tail"],
            "entries": [],
        }
    entries = [line for line in result["stdout_tail"].splitlines() if line.strip()]
    return {
        "git_available": True,
        "dirty": bool(entries),
        "entry_count": len(entries),
        "entries": entries[:200],
    }


def git_head() -> str:
    result = run(["git", "rev-parse", "HEAD"], timeout=60)
    if result["returncode"] != 0:
        return ""
    return result["stdout_tail"].strip()


def git_lines(command: list[str]) -> list[str]:
    result = run(command, timeout=120)
    if result["returncode"] != 0:
        return []
    return [line.strip() for line in result["stdout_tail"].splitlines() if line.strip()]


def run(command: list[str], *, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout",
        }


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve(path: str | Path) -> Path:
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    return ROOT / parsed


def relative_to_root(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
