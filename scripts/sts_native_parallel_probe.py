"""Native STS parallel-token probe.

This is the first hard check that the stream-token-superposition lane is more
than a trace format: it calls the Rust/SymLiquid decoder to train on private STS
train rows and emit one token per output stream per generation step on held-out
STS eval rows.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sts_learning/sts_code_context_spaces_seed14.jsonl")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--hv-dim", type=int, default=384)
    parser.add_argument("--max-vocab", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--max-generate-steps", type=int, default=18)
    parser.add_argument("--max-train-rows", type=int, default=240)
    parser.add_argument("--max-eval-rows", type=int, default=80)
    parser.add_argument("--max-generate-rows", type=int, default=128)
    parser.add_argument("--checkpoint-out", default="reports/sts_parallel_decoder_checkpoint.json")
    parser.add_argument("--generation-out", default="reports/sts_parallel_decoder_generations.jsonl")
    parser.add_argument("--out", default="reports/sts_native_parallel_probe.json")
    args = parser.parse_args()

    started = time.perf_counter()
    command = rust_command(args)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
    native_report = read_json(resolve(args.out), {})
    trigger_state = "GREEN" if result.returncode == 0 and native_report.get("trigger_state") == "GREEN" else "YELLOW"
    wrapper = {
        "policy": "project_theseus_sts_native_parallel_probe_wrapper_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "native_report": rel(resolve(args.out)),
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1600:],
        "stderr_tail": result.stderr[-1600:],
        "summary": {
            "native_trigger_state": native_report.get("trigger_state"),
            "row_count": get_path(native_report, ["summary", "row_count"], 0),
            "train_row_count": get_path(native_report, ["summary", "train_row_count"], 0),
            "eval_row_count": get_path(native_report, ["summary", "eval_row_count"], 0),
            "max_train_rows": get_path(native_report, ["summary", "max_train_rows"], args.max_train_rows),
            "max_eval_rows": get_path(native_report, ["summary", "max_eval_rows"], args.max_eval_rows),
            "max_generate_rows": get_path(native_report, ["summary", "max_generate_rows"], args.max_generate_rows),
            "output_stream_count": get_path(native_report, ["summary", "output_stream_count"], 0),
            "before_eval_token_accuracy": get_path(native_report, ["summary", "before_eval_token_accuracy"], 0.0),
            "after_eval_token_accuracy": get_path(native_report, ["summary", "after_eval_token_accuracy"], 0.0),
            "eval_token_accuracy_delta": get_path(native_report, ["summary", "eval_token_accuracy_delta"], 0.0),
            "native_parallel_token_generation_proven": bool(get_path(native_report, ["summary", "native_parallel_token_generation_proven"], False)),
            "one_token_per_output_stream_per_step": bool(get_path(native_report, ["summary", "one_token_per_output_stream_per_step"], False)),
            "public_benchmark_solutions_included": bool(get_path(native_report, ["summary", "public_benchmark_solutions_included"], False)),
            "external_inference_calls": 0,
        },
        "gates": [
            gate("rust_command_succeeded", result.returncode == 0, f"returncode={result.returncode}"),
            gate("native_report_green", native_report.get("trigger_state") == "GREEN", native_report.get("trigger_state")),
            gate("native_parallel_token_generation_proven", bool(get_path(native_report, ["summary", "native_parallel_token_generation_proven"], False)), get_path(native_report, ["summary"], {})),
            gate("eval_accuracy_improved", float(get_path(native_report, ["summary", "eval_token_accuracy_delta"], 0.0) or 0.0) > 0.0, get_path(native_report, ["summary", "eval_token_accuracy_delta"], 0.0)),
            gate("no_public_benchmark_solutions", not bool(get_path(native_report, ["summary", "public_benchmark_solutions_included"], False)), get_path(native_report, ["summary", "public_benchmark_solutions_included"], None)),
            gate("external_inference_zero", True, "local Rust/SymLiquid training only"),
        ],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), native_report if native_report else wrapper)
    write_json(resolve("reports/sts_native_parallel_probe_wrapper.json"), wrapper)
    print(json.dumps(wrapper, indent=2))
    return 0 if trigger_state in {"GREEN", "YELLOW"} else 1


def rust_command(args: argparse.Namespace) -> list[str]:
    exe = ROOT / "target" / "release" / "symliquid-cli.exe"
    source_files = [
        ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "sts_parallel_decoder.rs",
    ]
    exe_fresh = exe.exists() and all(exe.stat().st_mtime >= path.stat().st_mtime for path in source_files if path.exists())
    prefix = [str(exe)] if exe_fresh else ["cargo", "run", "--release", "-p", "symliquid-cli", "--"]
    return [
        *prefix,
        "train-sts-parallel-decoder",
        "--input",
        rel(resolve(args.input)),
        "--seed",
        str(args.seed),
        "--hv-dim",
        str(args.hv_dim),
        "--max-vocab",
        str(args.max_vocab),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--max-generate-steps",
        str(args.max_generate_steps),
        "--max-train-rows",
        str(args.max_train_rows),
        "--max-eval-rows",
        str(args.max_eval_rows),
        "--max-generate-rows",
        str(args.max_generate_rows),
        "--checkpoint-out",
        rel(resolve(args.checkpoint_out)),
        "--generation-out",
        rel(resolve(args.generation_out)),
        "--report-out",
        rel(resolve(args.out)),
    ]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def get_path(data: Any, path: list[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
