"""Fetch and switch to the newest Project Theseus remote lane.

Windows is currently the fastest-moving coordinator. This helper keeps Mac
sessions from accidentally working on an older Codex branch after Windows has
pushed a newer `origin/*` tip.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch GitHub and select the newest origin branch by commit date.")
    parser.add_argument("--switch", action="store_true", help="Switch to the newest remote branch and fast-forward it.")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    fetch = run(["git", "fetch", "--all", "--prune", "--tags"])
    status = run(["git", "status", "--porcelain"])
    dirty = bool(status["stdout"].strip())
    branches = remote_branches(args.remote)
    selected = branches[0] if branches else {}
    switched: dict[str, Any] = {}
    if args.switch:
        if dirty:
            switched = {"ok": False, "error": "dirty_worktree", "message": "Commit, stash, or clean local changes before switching lanes."}
        elif not selected:
            switched = {"ok": False, "error": "no_remote_branches"}
        else:
            switched = switch_to(selected)

    report = {
        "ok": bool(fetch["ok"] and branches and (not args.switch or switched.get("ok"))),
        "policy": "project_theseus_latest_windows_lane_v0",
        "remote": args.remote,
        "dirty_worktree": dirty,
        "selected": selected,
        "branch_count": len(branches),
        "top_branches": branches[:10],
        "fetch": {"ok": fetch["ok"], "stderr_tail": fetch["stderr"][-1000:]},
        "switch": switched,
        "next_action": "Run with --switch to move the checkout to the selected branch." if not args.switch else "",
    }
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 2


def remote_branches(remote: str) -> list[dict[str, Any]]:
    fmt = "%(committerdate:iso8601-strict)%09%(refname:short)%09%(objectname:short)%09%(subject)"
    result = run(["git", "for-each-ref", "--sort=-committerdate", f"--format={fmt}", f"refs/remotes/{remote}"])
    rows = []
    for line in result["stdout"].splitlines():
        date, ref, short, subject = (line.split("\t", 3) + ["", "", "", ""])[:4]
        if ref.endswith("/HEAD"):
            continue
        branch = ref.split("/", 1)[1] if "/" in ref else ref
        rows.append({"committerdate": date, "remote_ref": ref, "branch": branch, "short_commit": short, "subject": subject})
    return rows


def switch_to(selected: dict[str, Any]) -> dict[str, Any]:
    branch = str(selected.get("branch") or "")
    remote_ref = str(selected.get("remote_ref") or "")
    local_exists = run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"])["returncode"] == 0
    if local_exists:
        switch = run(["git", "switch", branch])
    else:
        switch = run(["git", "switch", "--track", remote_ref])
    if not switch["ok"]:
        return {"ok": False, "error": "switch_failed", "stderr_tail": switch["stderr"][-1000:]}
    pull = run(["git", "pull", "--ff-only"])
    return {
        "ok": bool(pull["ok"]),
        "branch": branch,
        "remote_ref": remote_ref,
        "pull_stdout_tail": pull["stdout"][-1000:],
        "pull_stderr_tail": pull["stderr"][-1000:],
    }


def run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


if __name__ == "__main__":
    raise SystemExit(main())
