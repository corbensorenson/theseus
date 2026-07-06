"""macOS dependency doctor/bootstrap for Project Theseus Hive installs.

This script is intentionally usable from a USB installer. It avoids assuming
Homebrew or Rust are present, creates the app-local venv, and installs only the
small Python dependency set needed for Hive/MLX workers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "macos_dependency_bootstrap.json"


def emit(message: str) -> None:
    print(f"[macos-deps] {message}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", default=str(ROOT / ".venv-puffer"))
    parser.add_argument("--runtime-root", default="")
    parser.add_argument("--install-missing", action="store_true")
    parser.add_argument("--assume-yes", action="store_true")
    parser.add_argument("--require-mlx", action="store_true")
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_REPORT.relative_to(ROOT)))
    args = parser.parse_args()

    report = bootstrap(args)
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 2


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    emit("Starting dependency bootstrap.")
    report: dict[str, Any] = {
        "ok": True,
        "policy": "project_theseus_macos_dependency_bootstrap_v0",
        "created_utc": now(),
        "root": str(ROOT),
        "platform": platform_report(),
        "install_missing": bool(args.install_missing),
        "checks": {},
        "actions": [],
        "next_actions": [],
    }
    is_macos = platform.system() == "Darwin"
    if not is_macos:
        report["ok"] = False
        report["error"] = "macos_required"
        return report
    if report["platform"].get("is_intel_mac"):
        report["actions"].append(
            {
                "action": "mac_architecture_profile",
                "role": "intel_mac_cpu_storage_operator_node",
                "mlx_expected": False,
            }
        )
        if args.require_mlx:
            report["ok"] = False
            report["error"] = "mlx_requires_apple_silicon_or_supported_linux_cuda"
            report["next_actions"].append(
                "Intel Macs should join without --require-mlx; they remain useful CPU, storage, operator, relay, and artifact-sync nodes."
            )
            return report

    emit("Checking Xcode Command Line Tools.")
    report["checks"]["xcode_select"] = command_check(["xcode-select", "-p"])
    if not report["checks"]["xcode_select"]["ok"]:
        report["next_actions"].append("Run xcode-select --install, then rerun the installer.")
        if args.install_missing:
            emit("Requesting Xcode Command Line Tools install prompt.")
            run_action(report, ["xcode-select", "--install"], timeout=30, allow_fail=True)

    emit("Checking Python, Homebrew, Cargo, and rustc.")
    report["checks"]["python3"] = command_check(["python3", "--version"])
    report["checks"]["brew"] = command_check(["brew", "--version"])
    report["checks"]["cargo"] = command_check(["cargo", "--version"])
    report["checks"]["rustc"] = command_check(["rustc", "--version"])

    if not report["checks"]["cargo"]["ok"]:
        if args.install_missing:
            emit("Cargo not found; installing Rust with rustup. This can take several minutes.")
            install_rust(report)
        else:
            report["next_actions"].append("Install Rust with rustup, or rerun with --install-missing.")

    if not report["checks"]["python3"]["ok"]:
        if args.install_missing and report["checks"]["brew"]["ok"]:
            run_action(report, ["brew", "install", "python"], timeout=1800, allow_fail=True)
        else:
            report["ok"] = False
            report["next_actions"].append("Install Python 3.11+ before running the Hive app.")
            return report

    venv_path = Path(args.venv).expanduser()
    python = ensure_venv(report, venv_path)
    install_python_deps(report, python, args)
    runtime = init_runtime(report, python, args.runtime_root)
    report["runtime_paths"] = runtime

    emit("Checking MLX import.")
    report["checks"]["mlx"] = python_check(python, "import mlx.core as mx; print('mlx.core ok')")
    if args.require_mlx and not report["checks"]["mlx"]["ok"]:
        report["ok"] = False
        report["next_actions"].append("MLX is required for this install but import mlx.core failed.")

    emit("Running Hive capability probe.")
    report["checks"]["hive_probe"] = run_json([str(python), "scripts/hive_node.py", "probe", "--out", "reports/hive_status.json"], timeout=120)
    if not report["checks"]["hive_probe"].get("ok"):
        report["ok"] = False
        report["next_actions"].append("Hive probe failed; inspect reports/macos_dependency_bootstrap.json.")

    report["venv_python"] = str(python)
    emit("Dependency bootstrap complete.")
    return report


def install_rust(report: dict[str, Any]) -> None:
    if not shutil.which("curl"):
        report["next_actions"].append("Install Rust manually from https://rustup.rs because curl is unavailable.")
        return
    command = "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    run_action(report, ["sh", "-c", command], timeout=1800, allow_fail=True)


def ensure_venv(report: dict[str, Any], venv_path: Path) -> Path:
    python = venv_path / "bin" / "python"
    if not python.exists():
        emit(f"Creating Python virtual environment at {venv_path}.")
        report["actions"].append({"action": "create_venv", "path": str(venv_path)})
        venv_path.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=True).create(str(venv_path))
    else:
        emit(f"Using existing Python virtual environment at {venv_path}.")
    return python


def install_python_deps(report: dict[str, Any], python: Path, args: argparse.Namespace) -> None:
    emit("Updating pip, wheel, and setuptools.")
    run_action(report, [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], timeout=600, allow_fail=False)
    packages = ["numpy"]
    machine = platform.machine().lower()
    if not args.skip_mlx and machine in {"arm64", "aarch64"}:
        packages.append("mlx")
    elif not args.skip_mlx and platform.system() == "Darwin":
        report["actions"].append({"action": "skip_mlx", "reason": "intel_mac_or_unsupported_macos_architecture", "machine": platform.machine()})
    for package in packages:
        probe = "import numpy; print('numpy ok')" if package == "numpy" else "import mlx.core as mx; print('mlx ok')"
        if python_check(python, probe)["ok"]:
            emit(f"Python package already present: {package}.")
            report["actions"].append({"action": "python_package_present", "package": package})
            continue
        emit(f"Installing Python package: {package}.")
        run_action(report, [str(python), "-m", "pip", "install", package], timeout=1800, allow_fail=(package == "mlx" and not args.require_mlx))


def init_runtime(report: dict[str, Any], python: Path, runtime_root: str) -> dict[str, Any]:
    emit("Initializing runtime directories.")
    command = [str(python), "scripts/runtime_paths.py", "init"]
    if runtime_root:
        command.extend(["--runtime-root", runtime_root])
    result = run_json(command, timeout=120)
    report["actions"].append({"action": "runtime_init", "ok": result.get("ok"), "runtime_root": runtime_root})
    return result


def command_check(command: list[str]) -> dict[str, Any]:
    return run_capture(command, timeout=20)


def python_check(python: Path, code: str) -> dict[str, Any]:
    return run_capture([str(python), "-c", code], timeout=30)


def run_json(command: list[str], *, timeout: int) -> dict[str, Any]:
    result = run_capture(command, timeout=timeout)
    if not result.get("ok"):
        return result
    try:
        parsed = json.loads(str(result.get("stdout") or "{}"))
    except json.JSONDecodeError:
        parsed = {"ok": True, "stdout": result.get("stdout", "")}
    if isinstance(parsed, dict):
        parsed.setdefault("ok", True)
        return parsed
    return {"ok": True, "value": parsed}


def run_action(report: dict[str, Any], command: list[str], *, timeout: int, allow_fail: bool) -> dict[str, Any]:
    emit("Running: " + " ".join(redact_command(command)))
    result = run_capture(command, timeout=timeout)
    emit(f"Finished with return code {result.get('returncode', 'unknown')}: " + " ".join(redact_command(command)))
    row = {"action": "run", "command": redact_command(command), **result}
    report["actions"].append(row)
    if not result.get("ok") and not allow_fail:
        report["ok"] = False
    return result


def run_capture(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def redact_command(command: list[str]) -> list[str]:
    return ["<redacted>" if "TOKEN" in part or "PASSWORD" in part else part for part in command]


def platform_report() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    machine_lower = machine.lower()
    is_macos = system == "Darwin"
    is_apple_silicon = is_macos and machine_lower in {"arm64", "aarch64"}
    is_intel_mac = is_macos and machine_lower in {"x86_64", "amd64"}
    return {
        "system": system,
        "release": platform.release(),
        "machine": machine,
        "python": platform.python_version(),
        "executable": sys.executable,
        "is_apple_silicon": is_apple_silicon,
        "is_intel_mac": is_intel_mac,
        "mlx_expected": is_apple_silicon,
        "hive_node_role": "apple_silicon_mlx_capable" if is_apple_silicon else "intel_cpu_storage_operator" if is_intel_mac else "unsupported_macos_architecture",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
