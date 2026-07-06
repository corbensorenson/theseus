#!/usr/bin/env python3
"""Run the code-transfer hard-blocker chain without public calibration.

Sequence:
1. Snapshot the current receiver gate as the stale baseline.
2. Run a fresh private Code LM closure with patched decoder/private rows.
3. Run decoder_v2_private_ablation_gate.
4. Run private_public_transfer_proof.
5. Refresh governor/scorecard reports.

This script never executes public benchmark tests. It writes a heartbeat-style
progress report while child processes run so long closures are distinguishable
from wedges.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from code_lm_private_rows import high_transfer_private_rows_string


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "code_transfer_hard_blocker_chain.json"
DEFAULT_MARKDOWN = REPORTS / "code_transfer_hard_blocker_chain.md"
PRIVATE_ROWS = high_transfer_private_rows_string()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    out = resolve(args.out)
    markdown = resolve(args.markdown_out)
    state: dict[str, Any] = {
        "policy": "project_theseus_code_transfer_hard_blocker_chain_v1",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "execute": bool(args.execute),
        "current_phase": "planned",
        "phases": [],
        "rules": {
            "public_calibration": "never run by this chain",
            "public_benchmark_training": "forbidden",
            "purpose": "prove private-to-public receiver candidate coverage before any public 4-card run",
        },
        "external_inference_calls": 0,
    }
    write_progress(out, markdown, state)
    if not args.execute:
        state["next_action"] = "rerun with --execute"
        write_progress(out, markdown, state)
        print(json.dumps(state, indent=2))
        return 0

    env = os.environ.copy()
    env.update(
        {
            "THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION": "1",
            "THESEUS_TARGET_FAMILY_STARVATION_RESCUE": "1",
            "THESEUS_TARGET_FAMILY_STARVATION_RESCUE_MIN_ROWS": "48",
            "THESEUS_TYPED_EDGE_EXEC_RECEIVER_V1": "1",
            "THESEUS_PRIVATE_TYPE_SHAPE_RECEIVER_VETO_V1": "1",
            "THESEUS_TEMPLATE_FREE_STUDENT_CANDIDATES": "1",
            "THESEUS_ALLOW_DIAGNOSTIC_TEMPLATE_CANDIDATES": "0",
        }
    )

    steps = [
        ("baseline_transfer_snapshot", [sys.executable, "scripts/private_public_transfer_proof.py", "--write-baseline"]),
        ("fresh_private_pressure_closure", private_closure_command(args.timeout_seconds)),
        ("decoder_v2_private_ablation_gate", [sys.executable, "scripts/decoder_v2_private_ablation_gate.py"]),
        ("private_public_transfer_proof", [sys.executable, "scripts/private_public_transfer_proof.py"]),
        ("asi_wall_breaker_governor", [sys.executable, "scripts/asi_wall_breaker_governor.py"]),
        ("symliquid_state_engine", [sys.executable, "scripts/symliquid_state_engine.py"]),
        ("a_plus_operating_scorecard", [sys.executable, "scripts/a_plus_operating_scorecard.py"]),
        ("model_growth_gate", [sys.executable, "scripts/model_growth_gate.py"]),
    ]

    state["trigger_state"] = "RUNNING"
    for name, command in steps:
        state["current_phase"] = name
        if name == "fresh_private_pressure_closure":
            attached = attach_existing_closure_phase(args.timeout_seconds, out, markdown, state)
            phase = attached if attached else run_phase(name, command, env, args.timeout_seconds, out, markdown, state)
        else:
            phase = run_phase(name, command, env, args.timeout_seconds, out, markdown, state)
        state["phases"].append(phase)
        if phase["returncode"] != 0:
            state["trigger_state"] = "RED"
            state["current_phase"] = "failed"
            state["failed_phase"] = name
            write_progress(out, markdown, state)
            print(json.dumps(state, indent=2))
            return int(phase["returncode"] or 1)

    state["trigger_state"] = "GREEN"
    state["current_phase"] = "completed"
    state["completed_utc"] = now()
    write_progress(out, markdown, state)
    print(json.dumps(state, indent=2))
    return 0


def private_closure_command(timeout_seconds: int) -> list[str]:
    return [
        sys.executable,
        "scripts/code_lm_closure.py",
        "--skip-public-calibration",
        "--public-cards",
        "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
        "--seed",
        "23",
        "--max-public-cases-per-card",
        "32",
        "--private-count",
        "320",
        "--epochs",
        "4",
        "--candidates-per-task",
        "8",
        "--disable-extra-private-train",
        "--disable-residual-private-train",
        "--disable-repo-repair-private-train",
        "--high-transfer-private-train-jsonl",
        PRIVATE_ROWS,
        "--max-high-transfer-private-train",
        "4800",
        "--max-rust-work-steps",
        "3000000",
        "--rust-timeout-seconds",
        str(max(60, int(timeout_seconds))),
        "--sts-timeout-seconds",
        str(max(60, min(int(timeout_seconds), 7200))),
        "--private-curriculum-out",
        "data/private_code_curriculum/code_lm_closure_private_pressure_private.jsonl",
        "--public-task-manifest-out",
        "reports/code_lm_public_tasks_private_pressure_private.jsonl",
        "--checkpoint-out",
        "reports/student_code_lm_checkpoint_private_pressure_private.json",
        "--private-candidate-out",
        "reports/code_lm_private_candidates_private_pressure_private.jsonl",
        "--public-candidate-out",
        "reports/student_code_candidates_private_pressure_private.jsonl",
        "--rust-report-out",
        "reports/code_lm_closure_rust_private_pressure_private.json",
        "--public-report-out",
        "reports/real_code_benchmark_graduation_private_pressure_private_skipped.json",
        "--public-trace-out",
        "reports/real_code_benchmark_traces_private_pressure_private_skipped.jsonl",
        "--out",
        "reports/code_lm_closure_private_pressure_private.json",
        "--sts-conditioning-input-out",
        "reports/code_lm_sts_conditioning_input_private_pressure_private.jsonl",
        "--sts-generation-out",
        "reports/code_lm_sts_public_generations_private_pressure_private.jsonl",
        "--sts-conditioning-checkpoint-out",
        "reports/code_lm_sts_conditioning_checkpoint_private_pressure_private.json",
        "--sts-conditioning-report-out",
        "reports/code_lm_sts_conditioning_report_private_pressure_private.json",
        "--lock-path",
        "reports/code_lm_closure_private_pressure_private.lock",
        "--typed-edge-exec-receiver-v1",
        "--edge-obligation-decode-gate-v1",
        "--private-type-shape-receiver-veto-v1",
        "--edge-obligation-report-out",
        "reports/edge_obligation_decode_gate_v1_private_pressure_private.json",
        "--edge-obligation-markdown-out",
        "reports/edge_obligation_decode_gate_v1_private_pressure_private.md",
    ]


def run_phase(
    name: str,
    command: list[str],
    env: dict[str, str],
    timeout_seconds: int,
    out: Path,
    markdown: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    log_path = REPORTS / f"{name}.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        phase = {
            "name": name,
            "started_utc": now(),
            "pid": proc.pid,
            "command": command,
            "log_path": rel_or_abs(log_path),
            "returncode": None,
        }
        state["active_child"] = phase
        while True:
            rc = proc.poll()
            elapsed = time.time() - started
            phase["elapsed_seconds"] = int(elapsed)
            phase["last_progress_utc"] = now()
            if rc is not None:
                phase["returncode"] = int(rc)
                phase["completed_utc"] = now()
                break
            if elapsed > timeout_seconds + 120:
                proc.kill()
                phase["returncode"] = 124
                phase["completed_utc"] = now()
                phase["timed_out"] = True
                break
            write_progress(out, markdown, state)
            time.sleep(30)
    state.pop("active_child", None)
    return phase


def attach_existing_closure_phase(
    timeout_seconds: int,
    out: Path,
    markdown: Path,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    lock = read_json(REPORTS / "code_lm_closure_private_pressure_private.lock", {})
    pid = int(lock.get("pid") or 0)
    if not pid or not pid_alive(pid):
        return None
    started = time.time()
    phase = {
        "name": "fresh_private_pressure_closure",
        "attached_to_existing": True,
        "started_utc": now(),
        "pid": pid,
        "lock_path": rel_or_abs(REPORTS / "code_lm_closure_private_pressure_private.lock"),
        "command": ["attach_existing_code_lm_closure_pid", str(pid)],
        "log_path": "reports/code_lm_closure_rust_private_pressure_private.json",
        "returncode": None,
    }
    state["active_child"] = phase
    while pid_alive(pid):
        elapsed = time.time() - started
        phase["elapsed_seconds"] = int(elapsed)
        phase["last_progress_utc"] = now()
        if elapsed > timeout_seconds + 120:
            phase["returncode"] = 124
            phase["completed_utc"] = now()
            phase["timed_out"] = True
            state.pop("active_child", None)
            return phase
        write_progress(out, markdown, state)
        time.sleep(30)
    time.sleep(5)
    report = read_json(REPORTS / "code_lm_closure_private_pressure_private.json", {})
    phase["elapsed_seconds"] = int(time.time() - started)
    phase["completed_utc"] = now()
    phase["returncode"] = 0 if report.get("run_status") == "completed" else 75
    phase["closure_trigger_state"] = report.get("trigger_state")
    phase["closure_run_status"] = report.get("run_status")
    state.pop("active_child", None)
    return phase


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else default
    except Exception:
        return default
    return default


def write_progress(out: Path, markdown: Path, state: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = now()
    out.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Code Transfer Hard Blocker Chain",
        "",
        f"- Status: **{state.get('trigger_state')}**",
        f"- Current phase: `{state.get('current_phase')}`",
        f"- Execute: `{state.get('execute')}`",
        f"- Updated: `{state.get('updated_utc')}`",
    ]
    active = state.get("active_child") if isinstance(state.get("active_child"), dict) else {}
    if active:
        lines.extend(
            [
                f"- Active child PID: `{active.get('pid')}`",
                f"- Active elapsed seconds: `{active.get('elapsed_seconds', 0)}`",
                f"- Active log: `{active.get('log_path')}`",
            ]
        )
    lines.extend(["", "## Completed Phases", ""])
    for phase in state.get("phases", []):
        lines.append(f"- `{phase.get('name')}` rc `{phase.get('returncode')}` elapsed `{phase.get('elapsed_seconds')}`s")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
