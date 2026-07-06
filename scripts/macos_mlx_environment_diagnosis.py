#!/usr/bin/env python3
"""Diagnose the active macOS MLX Python runtime without crashing the parent.

MLX/Metal failures can abort the interpreter at native import time. This doctor
probes candidate Python runtimes only in child processes and writes an explicit
route decision for Hive/Theseus Mac acceleration.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "macos_mlx_environment_diagnosis.json"
DEFAULT_MD = ROOT / "reports" / "macos_mlx_environment_diagnosis.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--timeout-seconds", type=int, default=12)
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    probes = [probe_python(path, timeout_seconds=max(1, int(args.timeout_seconds))) for path in candidate_pythons()]
    usable = [row for row in probes if row.get("mlx_core_usable")]
    native_aborts = [row for row in probes if row.get("native_abort")]
    active = probes[0] if probes else {}
    route = route_decision(usable, native_aborts, active)
    gates = [
        gate("candidate_python_found", bool(probes), [row.get("python") for row in probes], "hard"),
        gate("parent_survived_native_probes", True, "all mlx.core checks ran in child processes", "hard"),
        gate("usable_mlx_runtime_or_safe_route", bool(usable) or route["action"] == "disable_mlx_acceleration_route", route, "hard"),
        gate("active_python_not_trusted_if_native_abort", not active.get("native_abort") or route["action"] == "disable_mlx_acceleration_route", active, "hard"),
        gate("external_inference_zero", True, 0, "hard"),
        gate("public_training_rows_zero", True, 0, "hard"),
        gate("fallback_return_zero", True, 0, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failed and usable else ("YELLOW" if not hard_failed else "RED")
    return {
        "policy": "project_theseus_macos_mlx_environment_diagnosis_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "candidate_python_count": len(probes),
            "usable_mlx_runtime_count": len(usable),
            "native_abort_count": len(native_aborts),
            "active_python": active.get("python"),
            "active_python_status": active.get("status"),
            "route_action": route["action"],
            "recommended_python": route.get("recommended_python"),
            "smallest_safe_fix": route.get("smallest_safe_fix"),
            "external_inference_calls": 0,
            "public_training_rows_written": 0,
            "fallback_return_count": 0,
        },
        "route_decision": route,
        "python_probes": probes,
        "gates": gates,
        "score_semantics": (
            "Local Mac MLX runtime diagnosis only. It does not install packages, train, call external "
            "inference, run public calibration, or promote an accelerator parity claim."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def candidate_pythons() -> list[Path]:
    values = [
        os.environ.get("THESEUS_MLX_PYTHON"),
        str(ROOT / ".venv-mlx" / "bin" / "python"),
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
        "/Users/corbensorenson/miniforge3/bin/python3",
        sys.executable,
        shutil.which("python3"),
        str(ROOT / ".venv" / "bin" / "python"),
    ]
    out: list[Path] = []
    seen = set()
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser()
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        out.append(path)
    return out


def probe_python(path: Path, *, timeout_seconds: int) -> dict[str, Any]:
    identity = run_child(path, IDENTITY_CODE, timeout_seconds=timeout_seconds)
    metadata = run_child(path, METADATA_CODE, timeout_seconds=timeout_seconds)
    core = run_child(path, CORE_CODE, timeout_seconds=timeout_seconds)
    native_abort = core.get("returncode", 0) < 0 or "NSException" in str(core.get("stderr_tail") or "")
    missing_mlx = "No module named 'mlx'" in str(core.get("stdout_tail") or "") or "No module named 'mlx'" in str(core.get("stderr_tail") or "")
    usable = core.get("returncode") == 0 and bool(get_json_stdout(core).get("ok"))
    status = "usable" if usable else ("native_abort" if native_abort else ("missing_mlx" if missing_mlx else "failed"))
    return {
        "python": str(path),
        "status": status,
        "mlx_core_usable": usable,
        "native_abort": native_abort,
        "missing_mlx": missing_mlx,
        "identity": get_json_stdout(identity),
        "metadata": get_json_stdout(metadata),
        "core_probe": core,
    }


def route_decision(usable: list[dict[str, Any]], native_aborts: list[dict[str, Any]], active: dict[str, Any]) -> dict[str, Any]:
    if usable:
        best = usable[0]
        return {
            "action": "route_mlx_to_usable_python",
            "recommended_python": best.get("python"),
            "reason": "At least one child-probed Python runtime imports mlx.core and runs a tensor eval.",
            "environment_export": f"export THESEUS_MLX_PYTHON='{best.get('python')}'",
            "production_routing_allowed": False,
            "parity_claim_allowed": False,
        }
    fix = (
        "Create a clean Apple-Silicon MLX runtime, then point Theseus at it. Recommended: "
        "`conda create -n theseus-mlx python=3.12 -y && conda run -n theseus-mlx python -m pip install -U pip mlx`, "
        "then set `THESEUS_MLX_PYTHON` to that environment's python and rerun this diagnosis. "
        "Do not use the current Miniforge Python for MLX while it aborts at native import."
    )
    reason = "No child-probed Python runtime can import mlx.core safely."
    if native_aborts:
        reason = "The active MLX install is discoverable but aborts with a native NSException at mlx.core import."
    return {
        "action": "disable_mlx_acceleration_route",
        "recommended_python": None,
        "reason": reason,
        "smallest_safe_fix": fix,
        "safe_current_route": "CPU/Torch for comparator work; no Apple MLX worker or parity claim until a clean child probe passes.",
        "production_routing_allowed": False,
        "parity_claim_allowed": False,
    }


def run_child(path: Path, code: str, *, timeout_seconds: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [str(path), "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": "timeout",
            "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else str(exc)[-2000:],
        }
    except OSError as exc:
        return {"returncode": "launch_failed", "stdout_tail": "", "stderr_tail": f"{type(exc).__name__}:{exc}"}
    return {
        "returncode": result.returncode,
        "stdout_tail": (result.stdout or "")[-1000:],
        "stderr_tail": (result.stderr or "")[-2000:],
    }


def get_json_stdout(result: dict[str, Any]) -> dict[str, Any]:
    text = str(result.get("stdout_tail") or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text.splitlines()[-1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


IDENTITY_CODE = (
    "import json, platform, sys\n"
    "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0], 'machine': platform.machine()}))\n"
)
METADATA_CODE = (
    "import json, importlib.metadata as md\n"
    "out={}\n"
    "for name in ['mlx','mlx-lm']:\n"
    "    try:\n"
    "        dist=md.distribution(name); out[name]={'version': dist.version, 'location': str(dist.locate_file(''))}\n"
    "    except BaseException as exc:\n"
    "        out[name]={'error': type(exc).__name__ + ':' + str(exc)}\n"
    "print(json.dumps(out))\n"
)
CORE_CODE = (
    "import json, platform, sys\n"
    "out={'executable': sys.executable, 'version': sys.version.split()[0], 'machine': platform.machine()}\n"
    "try:\n"
    "    import mlx.core as mx\n"
    "    x=mx.array([1.0,2.0]); mx.eval(x)\n"
    "    out.update({'ok': True, 'default_device': str(mx.default_device()), 'values': [float(v) for v in x.tolist()]})\n"
    "except BaseException as exc:\n"
    "    out.update({'ok': False, 'exc_type': type(exc).__name__, 'exc': str(exc)})\n"
    "print(json.dumps(out))\n"
)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    route = report.get("route_decision") if isinstance(report.get("route_decision"), dict) else {}
    lines = [
        "# macOS MLX Environment Diagnosis",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- active_python: `{summary.get('active_python')}`",
        f"- active_python_status: `{summary.get('active_python_status')}`",
        f"- usable_mlx_runtime_count: `{summary.get('usable_mlx_runtime_count')}`",
        f"- native_abort_count: `{summary.get('native_abort_count')}`",
        f"- route_action: `{summary.get('route_action')}`",
        f"- smallest_safe_fix: {route.get('smallest_safe_fix') or 'none'}",
        "",
        "## Python Probes",
    ]
    for row in report.get("python_probes", []):
        lines.append(f"- `{row.get('python')}`: `{row.get('status')}`")
    lines.extend(["", "## Failed Gates"])
    failed = [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("passed")]
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(f"- `{row.get('name')}` ({row.get('severity')}): `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


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
