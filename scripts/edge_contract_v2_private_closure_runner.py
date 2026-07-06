#!/usr/bin/env python3
"""Run edge-contract-v2 private closure, then its private verifier.

This wrapper exists so the Hive work board can execute one safe, private-only
task that both trains/evaluates the held-out edge-contract v2 pressure and
materializes the verifier gate that decides whether a public receiver
calibration is allowed.
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
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_rows import training_data_path  # noqa: E402


SOURCE_PRIVATE_ROWS = Path(
    training_data_path(
        "high_transfer",
        "private_train",
        "edge_contract_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
    )
)
REPAIRED_PRIVATE_ROWS = Path(
    training_data_path(
        "high_transfer",
        "private_train",
        "edge_contract_v2_private_residual_curriculum_repaired_code_lm_tasks.jsonl",
    )
)
PRIVATE_ROWS = REPAIRED_PRIVATE_ROWS if REPAIRED_PRIVATE_ROWS.exists() else SOURCE_PRIVATE_ROWS
DEFAULT_OUT = ROOT / "reports" / "edge_contract_v2_private_closure_runner.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "edge_contract_v2_private_closure_runner.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-rows", default=str(PRIVATE_ROWS))
    parser.add_argument("--private-count", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--candidates-per-task", type=int, default=12)
    parser.add_argument("--max-high-transfer-private-train", type=int, default=960)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--rust-timeout-seconds", type=int, default=21600)
    parser.add_argument("--sts-timeout-seconds", type=int, default=7200)
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    private_rows = resolve(args.private_rows)
    steps: list[dict[str, Any]] = []
    steps.append(
        run_step(
            "code_lm_closure_private_only",
            [
                sys.executable,
                "scripts/code_lm_closure.py",
                "--skip-public-calibration",
                "--public-cards",
                "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                "--seed",
                "47",
                "--max-public-cases-per-card",
                "32",
                "--private-count",
                str(max(1, int(args.private_count))),
                "--epochs",
                str(max(1, int(args.epochs))),
                "--candidates-per-task",
                str(max(1, int(args.candidates_per_task))),
                "--disable-extra-private-train",
                "--disable-residual-private-train",
                "--disable-repo-repair-private-train",
                "--high-transfer-private-train-jsonl",
                str(private_rows).replace("\\", "/"),
                "--max-high-transfer-private-train",
                str(max(1, int(args.max_high_transfer_private_train))),
                "--max-rust-work-steps",
                "4000000",
                "--rust-timeout-seconds",
                str(max(60, int(args.rust_timeout_seconds))),
                "--sts-timeout-seconds",
                str(max(60, int(args.sts_timeout_seconds))),
                "--private-curriculum-out",
                "data/private_code_curriculum/code_lm_closure_edge_contract_v2_private.jsonl",
                "--public-task-manifest-out",
                "reports/code_lm_public_tasks_edge_contract_v2_private.jsonl",
                "--checkpoint-out",
                "reports/student_code_lm_checkpoint_edge_contract_v2_private.json",
                "--private-candidate-out",
                "reports/code_lm_private_candidates_edge_contract_v2_private.jsonl",
                "--public-candidate-out",
                "reports/student_code_candidates_edge_contract_v2_private.jsonl",
                "--rust-report-out",
                "reports/code_lm_closure_rust_edge_contract_v2_private.json",
                "--public-report-out",
                "reports/real_code_benchmark_graduation_edge_contract_v2_private_skipped.json",
                "--public-trace-out",
                "reports/real_code_benchmark_traces_edge_contract_v2_private_skipped.jsonl",
                "--out",
                "reports/code_lm_closure_edge_contract_v2_private.json",
                "--sts-conditioning-input-out",
                "reports/code_lm_sts_conditioning_input_edge_contract_v2_private.jsonl",
                "--sts-generation-out",
                "reports/code_lm_sts_public_generations_edge_contract_v2_private.jsonl",
                "--sts-conditioning-checkpoint-out",
                "reports/code_lm_sts_conditioning_checkpoint_edge_contract_v2_private.json",
                "--sts-conditioning-report-out",
                "reports/code_lm_sts_conditioning_report_edge_contract_v2_private.json",
                "--lock-path",
                "reports/code_lm_closure_edge_contract_v2_private.lock",
            ],
            timeout=max(60, int(args.timeout_seconds)),
        )
    )
    steps.append(
        run_step(
            "edge_contract_v2_private_verifier",
            [
                sys.executable,
                "scripts/edge_contract_v2_private_verifier.py",
                "--private-in",
                str(private_rows).replace("\\", "/"),
                "--closure-report",
                "reports/code_lm_closure_edge_contract_v2_private.json",
                "--private-candidates",
                "reports/code_lm_private_candidates_edge_contract_v2_private.jsonl",
                "--out",
                "reports/edge_contract_v2_private_verifier.json",
                "--markdown-out",
                "reports/edge_contract_v2_private_verifier.md",
            ],
            timeout=max(60, min(int(args.timeout_seconds), 1800)),
        )
    )

    closure = read_json(ROOT / "reports" / "code_lm_closure_edge_contract_v2_private.json", {})
    verifier = read_json(ROOT / "reports" / "edge_contract_v2_private_verifier.json", {})
    hard_failures = [step for step in steps if step["returncode"] not in (0,)]
    report = {
        "policy": "project_theseus_edge_contract_v2_private_closure_runner_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_failures and verifier.get("ready_for_public_calibration") else ("YELLOW" if not hard_failures else "RED"),
        "run_status": "completed" if not hard_failures else "failed",
        "summary": {
            "private_rows": count_jsonl(private_rows),
            "closure_trigger_state": closure.get("trigger_state"),
            "closure_run_status": closure.get("run_status"),
            "verifier_trigger_state": verifier.get("trigger_state"),
            "ready_for_public_calibration": verifier.get("ready_for_public_calibration"),
            "private_delta": get_path(verifier, ["summary", "private_delta"], None),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "public_benchmarks": "calibration_only_not_training",
            "external_inference_calls": 0,
        },
        "steps": steps,
        "outputs": {
            "closure_report": "reports/code_lm_closure_edge_contract_v2_private.json",
            "verifier_report": "reports/edge_contract_v2_private_verifier.json",
            "private_candidates": "reports/code_lm_private_candidates_edge_contract_v2_private.jsonl",
        },
        "public_tests_used_for_training": False,
        "public_solutions_used_for_training": False,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def run_step(name: str, command: list[str], *, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {
            "name": name,
            "command": command,
            "returncode": result.returncode,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": command,
            "returncode": 124,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stdout_tail": str(exc.stdout or "")[-4000:],
            "stderr_tail": str(exc.stderr or "")[-4000:],
        }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Edge Contract V2 Private Closure Runner",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- run_status: `{report.get('run_status')}`",
        f"- ready_for_public_calibration: `{summary.get('ready_for_public_calibration')}`",
        f"- private_delta: `{summary.get('private_delta')}`",
        "",
        "## Steps",
        "",
    ]
    for step in report.get("steps", []):
        lines.append(f"- `{step.get('name')}` returncode `{step.get('returncode')}` elapsed `{step.get('elapsed_seconds')}`")
    lines.append("")
    return "\n".join(lines)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def get_path(data: Any, path: list[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
