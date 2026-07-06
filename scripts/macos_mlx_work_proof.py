"""Run a bounded Apple MLX work proof for the macOS Hive lane.

This is release-gate evidence, not a benchmark chase. It proves that the Mac
runtime can execute the registered Hive MLX worker chunks and the user-facing
MLX command bridges, then records report paths, metrics, receipts, and ledger
growth in one compact artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROOF_DIR = REPORTS / "macos_mlx_work_proof"
LEDGER = REPORTS / "hive_worker_chunk_ledger.jsonl"
DEFAULT_OUT = REPORTS / "macos_mlx_work_proof.json"
DEFAULT_MARKDOWN = REPORTS / "macos_mlx_work_proof.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded macOS MLX worker and command proof.")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--skip-worker-smokes", action="store_true")
    parser.add_argument("--skip-cli-smokes", action="store_true")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve_path(args.out), report)
    if args.markdown_out:
        resolve_path(args.markdown_out).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("ok") else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    platform_info = platform_report()
    mlx = mlx_runtime_report()
    ledger_before = line_count(LEDGER)
    worker_smokes = [] if args.skip_worker_smokes else run_worker_smokes(args)
    cli_smokes = [] if args.skip_cli_smokes else run_cli_smokes(args)
    parity = run_parity_audit(args)
    ledger_after = line_count(LEDGER)

    is_apple_silicon = bool(platform_info.get("is_apple_silicon"))
    if not is_apple_silicon:
        state = "INTEL_CPU_STORAGE_OPERATOR_ONLY"
        ok = True
    else:
        worker_ok = bool(worker_smokes) and all(bool(row.get("ok")) for row in worker_smokes)
        cli_ok = bool(cli_smokes) and all(bool(row.get("ok")) for row in cli_smokes)
        parity_ok = bool(parity.get("ok")) and str(parity.get("state") or "") != "RED"
        receipts_ok = all(worker_receipt_ok(row) for row in worker_smokes)
        ok = bool(mlx.get("available") and worker_ok and cli_ok and parity_ok and receipts_ok)
        state = "GREEN" if ok else "RED"

    return {
        "ok": ok,
        "policy": "project_theseus_macos_mlx_work_proof_v0",
        "created_utc": now(),
        "state": state,
        "platform": platform_info,
        "mlx": mlx,
        "summary": {
            "worker_smoke_count": len(worker_smokes),
            "worker_smoke_ok_count": len([row for row in worker_smokes if row.get("ok")]),
            "cli_smoke_count": len(cli_smokes),
            "cli_smoke_ok_count": len([row for row in cli_smokes if row.get("ok")]),
            "ledger_rows_before": ledger_before,
            "ledger_rows_after": ledger_after,
            "ledger_rows_added": max(0, ledger_after - ledger_before),
            "external_inference_calls": 0,
            "teacher_used": False,
        },
        "parity_audit": compact_parity(parity),
        "worker_smokes": worker_smokes,
        "cli_smokes": cli_smokes,
        "guardrails": {
            "bounded": True,
            "registered_worker_chunks_only": True,
            "no_arbitrary_shell": True,
            "no_teacher": True,
            "no_public_benchmark_training": True,
            "intel_policy": "Intel Macs are valid CPU/storage/operator nodes and must not advertise MLX.",
        },
        "duration_seconds": round(time.perf_counter() - started, 3),
        "next_actions": next_actions(platform_info, mlx, worker_smokes, cli_smokes, parity),
        "external_inference_calls": 0,
    }


def run_parity_audit(args: argparse.Namespace) -> dict[str, Any]:
    out = PROOF_DIR / "macos_mlx_parity_audit.json"
    md = PROOF_DIR / "macos_mlx_parity_audit.md"
    return run_json(
        [
            str(preferred_mlx_python()),
            "scripts/macos_mlx_parity_audit.py",
            "--out",
            str(out.relative_to(ROOT)),
            "--markdown-out",
            str(md.relative_to(ROOT)),
        ],
        timeout=max(60, int(args.timeout_seconds)),
        out_path=out,
    )


def run_worker_smokes(args: argparse.Namespace) -> list[dict[str, Any]]:
    stamp = int(time.time())
    specs = [
        {
            "name": "mlx_eval_chunk",
            "kind": "mlx_babylm_eval",
            "task_kind": "mlx_eval_chunk",
            "payload": {
                "chunk_id": f"macos_gate_mlx_eval_{stamp}",
                "profile": "smoke",
                "train_limit": 4,
                "eval_limit": 4,
                "feature_dim": 64,
                "steps": 1,
                "source": "macos_mlx_work_proof",
            },
        },
        {
            "name": "mlx_training_chunk",
            "kind": "mlx_babylm_train",
            "task_kind": "mlx_training_chunk",
            "payload": {
                "chunk_id": f"macos_gate_mlx_train_{stamp}",
                "profile": "smoke",
                "train_limit": 4,
                "eval_limit": 4,
                "feature_dim": 64,
                "steps": 1,
                "source": "macos_mlx_work_proof",
            },
        },
        {
            "name": "mlx_rollout_chunk",
            "kind": "mlx_rollout_probe",
            "task_kind": "mlx_rollout_chunk",
            "payload": {
                "chunk_id": f"macos_gate_mlx_rollout_{stamp}",
                "profile": "smoke",
                "cases_per_task": 4,
                "eval_cases": 4,
                "epochs": 1,
                "seq_len": 8,
                "obs_dim": 8,
                "hv_dim": 64,
                "lr": 0.03,
                "source": "macos_mlx_work_proof",
            },
        },
    ]
    rows = []
    for spec in specs:
        out = PROOF_DIR / f"{spec['name']}.json"
        command = [
            str(preferred_mlx_python()),
            "scripts/hive_worker_chunk.py",
            "--kind",
            str(spec["kind"]),
            "--payload-json",
            json.dumps(spec["payload"]),
            "--out",
            str(out.relative_to(ROOT)),
        ]
        result = run_json(command, timeout=max(60, int(args.timeout_seconds)), out_path=out)
        rows.append(compact_worker_result(spec, out, result, command))
    return rows


def run_cli_smokes(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = [str(preferred_mlx_python()), "scripts/macos_mlx_training.py"]
    specs = [
        {
            "name": "train_standalone_mlx",
            "command": [
                *base,
                "train-standalone-mlx",
                "--cases-per-task",
                "4",
                "--epochs",
                "1",
                "--hv-dim",
                "64",
            ],
            "out": PROOF_DIR / "cli_train_standalone_mlx.json",
        },
        {
            "name": "train_rollout_mlx",
            "command": [
                *base,
                "train-rollout-mlx",
                "--cases-per-task",
                "4",
                "--probe-cases-per-task",
                "4",
                "--epochs",
                "1",
                "--state-epochs",
                "0",
                "--state-lr",
                "0.0",
                "--rollout-batch",
                "4",
                "--obs-dim",
                "8",
                "--hidden-dim",
                "8",
                "--reservoir-dim",
                "8",
                "--hv-dim",
                "64",
                "--seq-len",
                "8",
            ],
            "out": PROOF_DIR / "cli_train_rollout_mlx.json",
        },
        {
            "name": "train_rollout_mlx_sweep",
            "command": [
                *base,
                "train-rollout-mlx-sweep",
                "--train-seeds",
                "0",
                "--state-epochs",
                "0,1",
                "--state-lrs",
                "0.0",
                "--cases-per-task",
                "4",
                "--probe-cases-per-task",
                "4",
                "--epochs",
                "1",
                "--rollout-batch",
                "4",
                "--obs-dim",
                "8",
                "--hidden-dim",
                "8",
                "--reservoir-dim",
                "8",
                "--hv-dim",
                "64",
                "--seq-len",
                "8",
            ],
            "out": PROOF_DIR / "cli_train_rollout_mlx_sweep.json",
        },
        {
            "name": "train_token_superposition_mlx",
            "command": [
                *base,
                "train-token-superposition-mlx",
                "--max-language-rows",
                "20",
                "--max-code-files",
                "0",
                "--max-vocab",
                "32",
                "--hv-dim",
                "64",
                "--train-samples",
                "64",
                "--eval-samples",
                "32",
                "--baseline-epochs",
                "1",
                "--bag-sizes",
                "2",
                "--recovery-ratios",
                "0.5",
                "--samples-per-launch",
                "16",
            ],
            "out": PROOF_DIR / "cli_train_token_superposition_mlx.json",
        },
    ]
    rows = []
    for spec in specs:
        command = [*spec["command"], "--out", str(spec["out"].relative_to(ROOT))]
        result = run_json(command, timeout=max(60, int(args.timeout_seconds)), out_path=spec["out"])
        rows.append(compact_cli_result(spec, result, command))
    return rows


def compact_worker_result(spec: dict[str, Any], out: Path, result: dict[str, Any], command: list[str]) -> dict[str, Any]:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    telemetry = result.get("telemetry") if isinstance(result.get("telemetry"), dict) else {}
    receipt = result.get("work_receipt") if isinstance(result.get("work_receipt"), dict) else {}
    return {
        "name": spec.get("name"),
        "kind": spec.get("kind"),
        "task_kind": spec.get("task_kind"),
        "ok": bool(result.get("ok")),
        "backend": result.get("backend"),
        "report_path": str(out.relative_to(ROOT)),
        "command": redact_command(command),
        "returncode": result.get("returncode"),
        "runtime_ms": result.get("runtime_ms"),
        "runtime_ms_child": result.get("runtime_ms_child"),
        "metrics": compact_metrics(metrics),
        "telemetry": {
            "model_path": telemetry.get("model_path"),
            "mlx_platform": telemetry.get("mlx_platform"),
            "synthetic_control_task": telemetry.get("synthetic_control_task"),
        },
        "work_receipt": {
            "accepted": receipt.get("accepted"),
            "task_kind": receipt.get("task_kind"),
            "worker_kind": receipt.get("worker_kind"),
            "backend": receipt.get("backend"),
            "claimed_work_units": receipt.get("claimed_work_units"),
        },
        "compute_market_ok": bool(get_path(result, ["compute_market", "ok"], False)),
        "error": result.get("error"),
        "message": result.get("message"),
    }


def compact_cli_result(spec: dict[str, Any], result: dict[str, Any], command: list[str]) -> dict[str, Any]:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    child = result.get("child_report") if isinstance(result.get("child_report"), dict) else {}
    return {
        "name": spec.get("name"),
        "command_name": result.get("command"),
        "parity_for": result.get("parity_for"),
        "ok": bool(result.get("ok")),
        "backend": result.get("backend"),
        "implementation": result.get("implementation"),
        "report_path": str(spec["out"].relative_to(ROOT)),
        "child_report_path": result.get("child_report_path"),
        "command": redact_command(command),
        "runtime_ms": result.get("runtime_ms") or get_path(result, ["timing_breakdown_ms", "total"], None),
        "metrics": compact_metrics(metrics),
        "child": {
            "ok": child.get("ok"),
            "kind": child.get("kind"),
            "chunk_id": child.get("chunk_id"),
            "backend": child.get("backend"),
            "work_receipt": {
                "accepted": get_path(child, ["work_receipt", "accepted"], None),
                "task_kind": get_path(child, ["work_receipt", "task_kind"], None),
                "claimed_work_units": get_path(child, ["work_receipt", "claimed_work_units"], None),
            },
        },
        "promotion_decision": result.get("promotion_decision") if isinstance(result.get("promotion_decision"), dict) else {},
        "error": result.get("error"),
        "message": result.get("message"),
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "train_rows",
        "eval_rows",
        "train_cases",
        "eval_cases",
        "expanded_train_examples",
        "expanded_eval_examples",
        "feature_dim",
        "steps",
        "train_accuracy",
        "eval_accuracy",
        "train_return_proxy",
        "eval_return_proxy",
        "loss_initial",
        "loss_final",
        "examples_per_second",
        "mlx_transfer_ms",
        "mlx_train_ms",
        "mlx_eval_ms",
    ]
    return {key: metrics.get(key) for key in keep if key in metrics}


def compact_parity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": report.get("ok"),
        "state": report.get("state"),
        "summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
        "report_path": str((PROOF_DIR / "macos_mlx_parity_audit.json").relative_to(ROOT)),
    }


def worker_receipt_ok(row: dict[str, Any]) -> bool:
    receipt = row.get("work_receipt") if isinstance(row.get("work_receipt"), dict) else {}
    return bool(row.get("ok") and receipt.get("accepted") and receipt.get("task_kind"))


def next_actions(
    platform_info: dict[str, Any],
    mlx: dict[str, Any],
    worker_smokes: list[dict[str, Any]],
    cli_smokes: list[dict[str, Any]],
    parity: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    if not platform_info.get("is_apple_silicon"):
        actions.append("Run this proof on Apple Silicon for MLX training evidence; Intel Macs should remain CPU/storage/operator nodes.")
        return actions
    if not mlx.get("available"):
        actions.append("Repair MLX in the source or installed app venv before using this Mac as an MLX worker.")
    if any(not row.get("ok") for row in worker_smokes):
        actions.append("Inspect reports/macos_mlx_work_proof/*.json for the failed registered MLX worker chunk.")
    if any(not row.get("ok") for row in cli_smokes):
        actions.append("Inspect reports/macos_mlx_work_proof/cli_*.json for the failed MLX command bridge.")
    if parity.get("state") == "YELLOW":
        actions.append("Continue the Rust/Metal or Rust/MLX kernel ports; the Python MLX bridge is real but not final hot-loop parity.")
    if not actions:
        actions.append("MLX bounded work proof is green; proceed to Windows reachability and physical Intel canary gates.")
    return actions


def mlx_runtime_report() -> dict[str, Any]:
    checks = [mlx_runtime_check(name, python) for name, python in runtime_candidates()]
    available = any(row.get("available") for row in checks)
    preferred = next((row for row in checks if row.get("available")), checks[0] if checks else {})
    return {
        "available": available,
        "module": "mlx.core",
        "preferred_runtime": preferred.get("name"),
        "preferred_python": preferred.get("python"),
        "runtimes": checks,
    }


def runtime_candidates() -> list[tuple[str, Path]]:
    rows = []
    env_python = os.environ.get("THESEUS_MLX_PYTHON")
    if env_python:
        rows.append(("env_THESEUS_MLX_PYTHON", Path(env_python)))
    rows.extend(
        [
            ("source_venv", ROOT / ".venv-puffer" / "bin" / "python"),
            (
                "installed_app_venv",
                Path.home()
                / "Library"
                / "Application Support"
                / "Project Theseus Hive"
                / "app"
                / "current"
                / ".venv-puffer"
                / "bin"
                / "python",
            ),
            ("active_python", Path(sys.executable)),
        ]
    )
    deduped = []
    seen = set()
    for name, path in rows:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append((name, path))
    return deduped


def preferred_mlx_python() -> Path:
    for _, candidate in runtime_candidates():
        if candidate.exists() and python_has_mlx(candidate):
            return candidate
    return Path(sys.executable)


def python_has_mlx(python: Path) -> bool:
    if not python.exists():
        return False
    try:
        result = subprocess.run(
            [str(python), "-c", "import mlx.core as mx; x=mx.array([1.0]); mx.eval(x)"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except Exception:
        return False
    return result.returncode == 0


def mlx_runtime_check(name: str, python: Path) -> dict[str, Any]:
    if not python.exists():
        return {"name": name, "python": str(python), "available": False, "error": "python_missing"}
    code = "import json, mlx.core as mx; x=mx.array([1.0,2.0]); mx.eval(x); print(json.dumps({'available': True, 'probe': [float(v) for v in x.tolist()]}))"
    try:
        result = subprocess.run([str(python), "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=30)
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary.
        return {"name": name, "python": str(python), "available": False, "error": type(exc).__name__, "message": str(exc)}
    if result.returncode != 0:
        return {
            "name": name,
            "python": str(python),
            "available": False,
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-500:],
        }
    payload = parse_json(result.stdout.strip(), {"available": True, "stdout_tail": result.stdout[-500:]})
    return {"name": name, "python": str(python), **payload}


def run_json(command: list[str], *, timeout: int, out_path: Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, env=runtime_env())
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": "timeout",
            "command": redact_command(command),
            "timeout_seconds": timeout,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
        }
    payload = parse_json(result.stdout.strip(), {})
    if not isinstance(payload, dict) or not payload:
        payload = read_json(out_path, {})
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("ok", result.returncode == 0)
    payload.setdefault("returncode", result.returncode)
    payload.setdefault("runtime_ms_subprocess", int((time.perf_counter() - started) * 1000))
    payload.setdefault("stdout_tail", result.stdout[-1000:])
    payload.setdefault("stderr_tail", result.stderr[-1000:])
    payload.setdefault("command", redact_command(command))
    return json_sanitize(payload)


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def platform_report() -> dict[str, Any]:
    machine = platform.machine()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": machine,
        "python": platform.python_version(),
        "is_macos": platform.system() == "Darwin",
        "is_apple_silicon": platform.system() == "Darwin" and machine.lower() in {"arm64", "aarch64"},
        "is_intel_mac": platform.system() == "Darwin" and machine.lower() in {"x86_64", "amd64"},
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# macOS MLX Work Proof",
        "",
        f"- State: `{report.get('state')}`",
        f"- OK: `{report.get('ok')}`",
        f"- MLX available: `{get_path(report, ['mlx', 'available'], False)}`",
        f"- Worker smokes: `{get_path(report, ['summary', 'worker_smoke_ok_count'], 0)}/{get_path(report, ['summary', 'worker_smoke_count'], 0)}`",
        f"- CLI smokes: `{get_path(report, ['summary', 'cli_smoke_ok_count'], 0)}/{get_path(report, ['summary', 'cli_smoke_count'], 0)}`",
        f"- Ledger rows added: `{get_path(report, ['summary', 'ledger_rows_added'], 0)}`",
        "",
        "## Worker Chunks",
        "",
        "| Task | Backend | OK | Report |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.get("worker_smokes", []) or []:
        lines.append(f"| `{row.get('task_kind')}` | `{row.get('backend')}` | `{row.get('ok')}` | `{row.get('report_path')}` |")
    lines.extend(["", "## CLI Bridges", "", "| Command | Parity For | Backend | OK | Report |", "| --- | --- | --- | --- | --- |"])
    for row in report.get("cli_smokes", []) or []:
        lines.append(f"| `{row.get('command_name') or row.get('name')}` | `{row.get('parity_for')}` | `{row.get('backend')}` | `{row.get('ok')}` | `{row.get('report_path')}` |")
    lines.extend(["", "## Next Actions"])
    for item in report.get("next_actions", []) or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def redact_command(command: list[str]) -> list[str]:
    sensitive_flags = {"--secret", "--token", "--join-token", "--hive-secret", "--password"}
    sensitive_keys = {"secret", "token", "password"}
    redacted = []
    skip_secret_value = False
    for item in command:
        text = str(item)
        if skip_secret_value:
            redacted.append("<redacted>")
            skip_secret_value = False
            continue
        lowered = text.lower()
        if lowered in sensitive_flags:
            redacted.append(text)
            skip_secret_value = True
        elif "=" in text and any(key in text.split("=", 1)[0].lower() for key in sensitive_keys):
            redacted.append(f"{text.split('=', 1)[0]}=<redacted>")
        else:
            redacted.append(text)
    return redacted


def line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def parse_json(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_sanitize(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_sanitize(inner) for inner in value]
    if isinstance(value, tuple):
        return [json_sanitize(inner) for inner in value]
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None
    return value


def get_path(value: Any, path: list[str], default: Any) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
