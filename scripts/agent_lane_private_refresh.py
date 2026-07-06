#!/usr/bin/env python3
"""Refresh private agent-lane transfer evidence without public calibration."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tool-cases", type=int, default=64)
    parser.add_argument("--max-capsules", type=int, default=256)
    parser.add_argument("--out", default="reports/agent_lane_private_refresh.json")
    parser.add_argument("--markdown-out", default="reports/agent_lane_private_refresh.md")
    args = parser.parse_args()

    started = time.perf_counter()
    steps = [
        step(
            "long_horizon_tool_use",
            [
                sys.executable,
                "scripts/long_horizon_tool_use_benchmark.py",
                "--max-cases",
                str(max(1, int(args.max_tool_cases))),
            ],
        ),
        step(
            "pufferlib4_rl_lane",
            [
                sys.executable,
                "scripts/pufferlib4_rl_lane.py",
                "--probe",
                "--out",
                "reports/pufferlib4_rl_lane.json",
                "--markdown-out",
                "reports/pufferlib4_rl_lane.md",
            ],
        ),
        step(
            "cross_domain_sts_capsules",
            [
                sys.executable,
                "scripts/cross_domain_sts_capsules.py",
                "--max-capsules",
                str(max(1, int(args.max_capsules))),
            ],
        ),
        step(
            "agent_lane_transfer_gate",
            [
                sys.executable,
                "scripts/agent_lane_transfer_gate.py",
                "--out",
                "reports/agent_lane_transfer_gate.json",
            ],
        ),
    ]
    report = {
        "policy": "project_theseus_agent_lane_private_refresh_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["returncode"] == 0 for row in steps) else "YELLOW",
        "summary": {
            "step_count": len(steps),
            "passed_steps": sum(1 for row in steps if row["returncode"] == 0),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "public_calibration_allowed": False,
            "external_inference_calls": 0,
        },
        "steps": steps,
        "rules": {
            "public_calibration": "not run",
            "training_data": "private local tool-use traces and metadata-only STS capsules",
            "external_inference": "not used",
        },
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    write_text(ROOT / args.markdown_out, render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] == "GREEN" else 2


def step(name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Agent Lane Private Refresh",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Passed steps: `{report.get('summary', {}).get('passed_steps')}/{report.get('summary', {}).get('step_count')}`",
        "",
        "## Steps",
        "",
    ]
    for row in report.get("steps", []):
        lines.append(f"- `{row.get('name')}` returncode=`{row.get('returncode')}`")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
