"""Run the RTX 2060 Super ablation matrix and summarize matched evidence."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/ablation_matrix_rtx2060super.json")
    parser.add_argument("--out", default="reports/ablation_matrix_rtx2060super_report.json")
    parser.add_argument("--workflow-trace-out", default="reports/workflow_routing_traces.jsonl")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    matrix = read_json(Path(args.matrix))
    comparisons = matrix.get("comparisons") or []
    rows: list[dict[str, Any]] = []
    ok = True
    for comparison in comparisons:
        started = time.perf_counter()
        row: dict[str, Any] = {
            "name": comparison.get("name"),
            "purpose": comparison.get("purpose"),
            "kind": "command" if comparison.get("command") else "source_report",
        }
        if command_text := comparison.get("command"):
            output_path = command_output_path(command_text)
            if args.skip_existing and output_path and output_path.exists():
                result = {
                    "command": command_text,
                    "returncode": 0,
                    "runtime_ms": 0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "skipped_existing": True,
                }
            else:
                result = run_command(command_text, timeout=args.timeout_seconds)
                append_workflow_trace(
                    Path(args.workflow_trace_out),
                    task=f"ablation matrix run: {comparison.get('name')}",
                    command=command_text,
                    returncode=result["returncode"],
                    runtime_ms=result["runtime_ms"],
                )
            row["run"] = result
            row["report_path"] = str(output_path) if output_path else None
            row["report_summary"] = summarize_report(output_path) if output_path else {}
            ok = ok and result["returncode"] == 0
        elif comparison.get("source_report"):
            path = Path(str(comparison["source_report"]))
            row["report_path"] = str(path)
            row["report_summary"] = summarize_report(path)
            row["metric_path"] = comparison.get("metric_path")
            row["metric_value"] = get_path(row["report_summary"].get("raw", {}), str(comparison.get("metric_path", "")).split("."), None)
        elif comparison.get("source_reports"):
            paths = [Path(str(path)) for path in comparison["source_reports"]]
            row["report_paths"] = [str(path) for path in paths]
            row["report_summaries"] = [summarize_report(path) for path in paths]
        else:
            row["error"] = "comparison_has_no_command_or_source_report"
            ok = False
        row["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        rows.append(row)

    report = {
        "policy": "local_only_no_external_inference",
        "methodology": "rtx2060super_matched_ablation_matrix",
        "ok": ok,
        "matrix": args.matrix,
        "comparison_count": len(rows),
        "completed_count": sum(1 for row in rows if not row.get("error") and (row.get("run", {}).get("returncode", 0) == 0)),
        "failed": [row for row in rows if row.get("run", {}).get("returncode", 0) != 0 or row.get("error")],
        "comparisons": rows,
        "promotion_rule": matrix.get("promotion_rule", {}),
        "external_inference_calls": 0,
    }
    write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def run_command(command_text: str, *, timeout: int) -> dict[str, Any]:
    command = shlex.split(command_text, posix=False)
    command = resolve_platform_command(command)
    uses_cargo = bool(command and command[0].lower() == "cargo")
    if command and command[0].lower() == "cargo":
        command[0] = resolve_cargo()
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    if uses_cargo:
        env.setdefault("CARGO_TARGET_DIR", default_cargo_target_dir())
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    return {
        "command": command_text,
        "argv": command,
        "returncode": result.returncode,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "cargo_target_dir": env.get("CARGO_TARGET_DIR") if uses_cargo else "",
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def resolve_platform_command(command: list[str]) -> list[str]:
    if not command:
        return command
    if sys.platform == "darwin":
        mlx_command = resolve_macos_mlx_command(command)
        if mlx_command is not None:
            return mlx_command
    exe = str(command[0]).replace("\\", "/")
    if not exe.endswith("target/release/symliquid-cli.exe"):
        return command
    native = ROOT / "target" / "release" / ("symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli")
    if native.exists():
        return [str(native), *command[1:]]
    return ["cargo", "run", "--release", "-p", "symliquid-cli", "--", *command[1:]]


def resolve_macos_mlx_command(command: list[str]) -> list[str] | None:
    if "--" not in command:
        return None
    subcommand_idx = command.index("--") + 1
    if subcommand_idx >= len(command):
        return None
    subcommand = command[subcommand_idx]
    if subcommand == "train-standalone-cuda":
        return [
            sys.executable,
            "scripts/macos_mlx_training.py",
            "train-standalone-mlx",
            *command[subcommand_idx + 1 :],
        ]
    if subcommand == "train-rollout-cuda":
        return [
            sys.executable,
            "scripts/macos_mlx_training.py",
            "train-rollout-mlx",
            *command[subcommand_idx + 1 :],
        ]
    return None


def resolve_cargo() -> str:
    cargo = Path.home() / ".cargo" / "bin" / "cargo.exe"
    return str(cargo) if cargo.exists() else "cargo"


def default_cargo_target_dir() -> str:
    d_drive = Path("D:/ProjectTheseus/runtime/cargo-target/ablation-matrix")
    if d_drive.drive and Path("D:/").exists():
        return str(d_drive)
    return str(ROOT / "target" / "ablation-matrix")


def command_output_path(command_text: str) -> Path | None:
    parts = shlex.split(command_text, posix=False)
    for idx, part in enumerate(parts):
        if part == "--out" and idx + 1 < len(parts):
            return Path(parts[idx + 1])
    return None


def summarize_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False}
    payload = read_json(path)
    summary = {
        "exists": True,
        "accuracy": accuracy(payload),
        "residual": get_path(payload, ["eval", "summary", "residual"], get_path(payload, ["summary", "residual"], payload.get("residual"))),
        "train_examples_per_second": payload.get("train_examples_per_second"),
        "train_runtime_ms": payload.get("train_runtime_ms"),
        "cuda_fallback": payload.get("cuda_fallback"),
        "runtime_profile_present": bool(payload.get("runtime_profile")),
        "timing_breakdown_present": bool(payload.get("timing_breakdown_ms")),
        "raw": payload,
    }
    return summary


def accuracy(payload: dict[str, Any]) -> float | None:
    value = get_path(payload, ["eval", "summary", "accuracy"], None)
    if value is None:
        value = get_path(payload, ["summary", "accuracy"], None)
    if value is None:
        value = payload.get("accuracy")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not key:
            continue
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def append_workflow_trace(
    path: Path, *, task: str, command: str, returncode: int, runtime_ms: int
) -> None:
    payload = {
        "trace_id": f"ablation_{int(time.time() * 1000)}_{abs(hash(command)) % 1000000}",
        "task": task,
        "workflow": "matched ablation matrix",
        "command": command,
        "selected_arms": ["benchmark_ratchet_arm", "rust_cuda_systems_arm"],
        "expected_arms": ["benchmark_ratchet_arm", "rust_cuda_systems_arm"],
        "risk": "low",
        "routing_pattern": "sequential",
        "returncode": returncode,
        "success": returncode == 0,
        "runtime_ms": runtime_ms,
        "split": "train",
        "source": "ablation_matrix_runner",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
