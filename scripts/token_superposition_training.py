"""Run the real Token Superposition Training comparison.

This file is deliberately only an orchestrator. The apples-to-apples work lives
inside platform-native backends: Rust/CUDA on Windows/Linux CUDA nodes and MLX
on Apple Silicon Macs. Both paths use the same local corpus, feature
hash/readout shape, optimizer family, and baseline AR versus TST bag training
plus ordinary AR recovery.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/token_superposition_training.json")
    parser.add_argument("--out", default="")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--skip-if-evidence", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = read_json(config_path)
    rust = config.get("rust_cuda") if isinstance(config.get("rust_cuda"), dict) else {}
    out = args.out or get_path(config, ["reports", "out"], "reports/token_superposition_training.json")
    backend = selected_backend()
    command = backend_command(backend, rust, out)

    if args.skip_if_evidence:
        existing = read_json(ROOT / out)
        if valid_backend_evidence(existing, backend):
            existing["wrapper"] = wrapper_report(
                config,
                out,
                command,
                {},
                {},
                backend=backend,
                status=f"skipped_existing_{backend}_evidence",
            )
            write_json(ROOT / out, existing)
            print(json.dumps(existing, indent=2))
            return 0
        reusable = reusable_backend_evidence(backend)
        if reusable:
            reusable["wrapper"] = wrapper_report(
                config,
                out,
                command,
                {},
                {},
                backend=backend,
                status=f"reused_existing_{backend}_evidence",
            )
            write_json(ROOT / out, reusable)
            print(json.dumps(reusable, indent=2))
            return 0

    build_row = {}
    if args.build and backend == "rust_cuda":
        build_row = run_command(
            ["cargo", "build", "--release", "-p", "symliquid-cli", "--features", "cuda"],
            timeout=args.timeout_seconds,
        )
        if build_row["returncode"] != 0:
            report = wrapper_report(config, out, command, build_row, {}, status="build_failed")
            write_json(ROOT / out, report)
            print(json.dumps(report, indent=2))
            return 1

    run_row = run_command(command, timeout=args.timeout_seconds)
    if run_row["returncode"] != 0:
        report = wrapper_report(config, out, command, build_row, run_row, backend=backend, status="run_failed")
        write_json(ROOT / out, report)
        print(json.dumps(report, indent=2))
        return 1

    child = read_json(ROOT / out)
    child["wrapper"] = wrapper_report(config, out, command, build_row, run_row, backend=backend, status="completed")
    write_json(ROOT / out, child)
    print(json.dumps(child, indent=2))
    return 0


def selected_backend() -> str:
    if sys.platform == "darwin":
        return "mlx_apple"
    return "rust_cuda"


def valid_backend_evidence(report: Any, backend: str) -> bool:
    if not isinstance(report, dict) or int(report.get("external_inference_calls") or 0) != 0:
        return False
    if backend == "mlx_apple":
        return (
            report.get("policy") == "project_theseus_token_superposition_mlx_report_v1"
            and bool(report.get("ok"))
            and str(report.get("backend") or "") in {"mlx_apple", "apple_mlx"}
            and not bool(report.get("cuda_fallback"))
        )
    return report.get("policy") == "project_theseus_token_superposition_rust_cuda_report_v1"


def reusable_backend_evidence(backend: str) -> dict[str, Any]:
    candidates: list[Path] = []
    if backend == "mlx_apple":
        candidates.extend(
            [
                ROOT / "reports" / "macos_mlx_work_proof" / "cli_train_token_superposition_mlx.json",
                ROOT / "reports" / "token_superposition_mlx_training.json",
            ]
        )
    for candidate in candidates:
        report = read_json(candidate)
        if valid_backend_evidence(report, backend):
            report["reused_from"] = str(candidate.relative_to(ROOT))
            return report
    return {}


def backend_command(backend: str, rust: dict[str, Any], out: str) -> list[str]:
    if backend == "mlx_apple":
        return mlx_command(rust, out)
    return rust_command(rust, out)


def mlx_command(rust: dict[str, Any], out: str) -> list[str]:
    command = [
        sys.executable,
        "scripts/macos_mlx_training.py",
        "train-token-superposition-mlx",
        "--input",
        str(rust.get("input") or "data/babylm_blimp_filtered_train.jsonl"),
        "--project-code-roots",
        str(rust.get("project_code_roots") or "scripts,crates"),
        "--train-seed",
        str(int(rust.get("train_seed", 20260514))),
        "--max-language-rows",
        str(int(rust.get("max_language_rows", 8000))),
        "--max-code-files",
        str(int(rust.get("max_code_files", 160))),
        "--max-chars-per-doc",
        str(int(rust.get("max_chars_per_doc", 12000))),
        "--max-vocab",
        str(int(rust.get("max_vocab", 256))),
        "--hv-dim",
        str(int(rust.get("hv_dim", 4096))),
        "--train-samples",
        str(int(rust.get("train_samples", 32768))),
        "--eval-samples",
        str(int(rust.get("eval_samples", 4096))),
        "--baseline-epochs",
        str(int(rust.get("baseline_epochs", 6))),
        "--bag-sizes",
        str(rust.get("bag_sizes") or "4,8"),
        "--recovery-ratios",
        str(rust.get("recovery_ratios") or "0.2,0.4"),
        "--lr",
        str(float(rust.get("lr", 0.03))),
        "--samples-per-launch",
        str(int(rust.get("samples_per_launch", 512))),
        "--gate-tolerance",
        str(float(rust.get("gate_tolerance", 0.002))),
        "--min-nominal-speedup",
        str(float(rust.get("min_nominal_speedup", 1.2))),
        "--min-train-speedup",
        str(float(rust.get("min_train_speedup", 1.0))),
        "--out",
        out,
    ]
    if bool(rust.get("include_project_code", True)):
        command.insert(3, "--include-project-code")
    artifact = str(rust.get("model_out") or "")
    if artifact:
        command.extend(["--model-out", artifact])
    return command


def rust_command(rust: dict[str, Any], out: str) -> list[str]:
    binary = str(rust.get("binary") or default_symliquid_binary())
    if sys.platform != "win32" and binary.endswith(".exe"):
        binary = str(default_symliquid_binary())
    command = [
        binary,
        "train-token-superposition-cuda",
        "--input",
        str(rust.get("input") or "data/babylm_blimp_filtered_train.jsonl"),
        "--project-code-roots",
        str(rust.get("project_code_roots") or "scripts,crates"),
        "--train-seed",
        str(int(rust.get("train_seed", 20260514))),
        "--max-language-rows",
        str(int(rust.get("max_language_rows", 8000))),
        "--max-code-files",
        str(int(rust.get("max_code_files", 160))),
        "--max-chars-per-doc",
        str(int(rust.get("max_chars_per_doc", 12000))),
        "--max-vocab",
        str(int(rust.get("max_vocab", 256))),
        "--hv-dim",
        str(int(rust.get("hv_dim", 4096))),
        "--train-samples",
        str(int(rust.get("train_samples", 32768))),
        "--eval-samples",
        str(int(rust.get("eval_samples", 4096))),
        "--baseline-epochs",
        str(int(rust.get("baseline_epochs", 6))),
        "--bag-sizes",
        str(rust.get("bag_sizes") or "4,8"),
        "--recovery-ratios",
        str(rust.get("recovery_ratios") or "0.2,0.4"),
        "--lr",
        str(float(rust.get("lr", 0.03))),
        "--samples-per-launch",
        str(int(rust.get("samples_per_launch", 512))),
        "--gate-tolerance",
        str(float(rust.get("gate_tolerance", 0.002))),
        "--min-nominal-speedup",
        str(float(rust.get("min_nominal_speedup", 1.2))),
        "--min-train-speedup",
        str(float(rust.get("min_train_speedup", 1.0))),
        "--out",
        out,
    ]
    if bool(rust.get("include_project_code", True)):
        command.insert(4, "--include-project-code")
    artifact = str(rust.get("model_out") or "")
    if artifact:
        command.extend(["--model-out", artifact])
    return command


def default_symliquid_binary() -> Path:
    release_dir = ROOT / "target" / "release"
    binary = release_dir / ("symliquid-cli.exe" if sys.platform == "win32" else "symliquid-cli")
    if binary.exists():
        return binary
    cargo = shutil.which("cargo") or "cargo"
    return Path(cargo)


def run_command(command: list[str], *, timeout: int) -> dict[str, Any]:
    if command and Path(command[0]).name == "cargo":
        command = [command[0], "run", "--release", "-p", "symliquid-cli", "--", *command[1:]]
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
            "returncode": int(result.returncode),
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
    except OSError as exc:
        return {
            "command": command,
            "returncode": 127,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "error": "spawn_failed",
        }


def wrapper_report(
    config: dict[str, Any],
    out: str,
    command: list[str],
    build_row: dict[str, Any],
    run_row: dict[str, Any],
    *,
    backend: str,
    status: str,
) -> dict[str, Any]:
    return {
        "policy": "project_theseus_token_superposition_wrapper_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "config": config,
        "out": out,
        "command": command,
        "build": build_row,
        "run": run_row,
        "toy_proxy": False,
        "backend": backend,
        "backend_required": backend,
        "cuda_fallback": False,
        "external_inference_calls": 0,
    }


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


if __name__ == "__main__":
    raise SystemExit(main())
