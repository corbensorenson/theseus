#!/usr/bin/env python3
"""Portable Project Theseus demo readiness preflight.

This is intentionally not a training launcher. It checks whether the current
machine can run a safe travel demo and whether Theseus' governance reports are
coherent enough to show without accidentally claiming promotion, model growth,
or public calibration.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/travel_demo_preflight.json")
    parser.add_argument("--markdown-out", default="reports/travel_demo_preflight.md")
    parser.add_argument("--mode", default="parents_demo", choices=["parents_demo", "technical_demo"])
    parser.add_argument("--target", default="auto", choices=["auto", "apple_mlx", "windows_cuda", "cpu_fallback"])
    args = parser.parse_args()

    started = time.perf_counter()
    host = host_info()
    target = choose_target(args.target, host)
    backend = backend_probe(target)
    theseus = theseus_governance_probe()
    demo = demo_path(args.mode, target, host, backend, theseus)
    gates = build_gates(args.mode, target, host, backend, theseus, demo)
    trigger_state = state_from_gates(gates)
    report = {
        "policy": "project_theseus_travel_demo_preflight_v1",
        "created_utc": now(),
        "mode": args.mode,
        "target_backend": target,
        "trigger_state": trigger_state,
        "host": host,
        "backend": backend,
        "theseus_governance": theseus,
        "demo": demo,
        "gates": gates,
        "next_actions": next_actions(trigger_state, target, backend, theseus),
        "rules": {
            "public_calibration": "never launched by this preflight",
            "training": "never launched by this preflight",
            "model_growth": "must remain locked for a travel demo",
            "candidate_promotion": "must remain locked unless separately proven by governance",
            "parent_demo": "prefer honest, fast, cached evidence over long-running research jobs",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if trigger_state in {"GREEN", "YELLOW"} else 2


def host_info() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    info: dict[str, Any] = {
        "system": system,
        "machine": machine,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "is_apple_silicon": system == "Darwin" and machine in {"arm64", "aarch64"},
        "is_windows": system == "Windows",
        "is_linux": system == "Linux",
    }
    if system == "Darwin":
        info["macos_version"] = run_text(["sw_vers", "-productVersion"])
        info["cpu_brand"] = run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        memsize = run_text(["sysctl", "-n", "hw.memsize"])
        info["unified_memory_gb"] = round_int_gb(memsize)
    elif system == "Windows":
        info["cpu_brand"] = platform.processor()
        info["total_memory_gb"] = windows_memory_gb()
    else:
        info["cpu_brand"] = platform.processor()
    return info


def choose_target(requested: str, host: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    if host.get("is_apple_silicon"):
        return "apple_mlx"
    if host.get("is_windows"):
        return "windows_cuda"
    return "cpu_fallback"


def backend_probe(target: str) -> dict[str, Any]:
    if target == "apple_mlx":
        return apple_mlx_probe()
    if target == "windows_cuda":
        return windows_cuda_probe()
    return {
        "backend": "cpu_fallback",
        "ready": True,
        "status": "YELLOW",
        "findings": [
            finding("YELLOW", "cpu_fallback_only", "No accelerated backend target was selected.", "Use only cached reports and tiny private checks for demo."),
        ],
    }


def apple_mlx_probe() -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    payload = run_python_json(
        "import json\n"
        "out={'mlx_import': False, 'mlx_smoke_ok': False}\n"
        "try:\n"
        " import mlx.core as mx\n"
        " out['mlx_import'] = True\n"
        " a = mx.ones((64, 64), dtype=mx.float32)\n"
        " b = mx.matmul(a, a)\n"
        " mx.eval(b)\n"
        " out['mlx_smoke_ok'] = bool(float(b[0,0]) == 64.0)\n"
        " out['default_device'] = str(mx.default_device())\n"
        "except Exception as e:\n"
        " out['mlx_error'] = str(e)\n"
        "try:\n"
        " import torch\n"
        " out['torch_import'] = True\n"
        " out['torch_mps_available'] = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())\n"
        "except Exception as e:\n"
        " out['torch_import'] = False\n"
        " out['torch_error'] = str(e)\n"
        "print(json.dumps(out))\n"
    )
    if not payload.get("mlx_import"):
        findings.append(finding("RED", "mlx_not_importable", "Python cannot import mlx.core.", "On Apple Silicon macOS 14+, install with: python3 -m pip install mlx."))
    elif not payload.get("mlx_smoke_ok"):
        findings.append(finding("RED", "mlx_smoke_failed", "MLX imported but a tiny matmul smoke failed.", "Reinstall MLX in the active Python environment."))
    else:
        findings.append(finding("GREEN", "mlx_smoke_ok", "MLX imported and completed a tiny Metal/unified-memory smoke.", "Use MLX for Mac demo inference/small tensor checks."))
    if payload.get("torch_mps_available"):
        findings.append(finding("GREEN", "torch_mps_available", "PyTorch MPS is available as a secondary Apple GPU path.", "Keep MLX as primary for Apple Silicon demos."))
    else:
        findings.append(finding("YELLOW", "torch_mps_not_available", "PyTorch MPS is not available or PyTorch is absent.", "This does not block the MLX demo path."))
    return {
        "backend": "apple_mlx",
        "ready": bool(payload.get("mlx_import") and payload.get("mlx_smoke_ok")),
        "status": "GREEN" if payload.get("mlx_import") and payload.get("mlx_smoke_ok") else "RED",
        "probe": payload,
        "findings": findings,
    }


def windows_cuda_probe() -> dict[str, Any]:
    cuda_report = read_json(REPORTS / "windows_cuda_doctor.json", {})
    nvidia_smi = shutil.which("nvidia-smi")
    live = run_json_command([nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]) if nvidia_smi else {}
    ready = cuda_report.get("trigger_state") == "GREEN" or bool(live.get("stdout"))
    findings = []
    if cuda_report:
        findings.append(finding(str(cuda_report.get("trigger_state") or "YELLOW"), "windows_cuda_doctor_report", "Existing Windows/CUDA doctor report is present.", "Use Windows CUDA for heavy training, not the travel Mac demo."))
    if live.get("stdout"):
        findings.append(finding("GREEN", "nvidia_smi_visible", live["stdout"].strip(), "CUDA hardware is visible on this Windows node."))
    elif nvidia_smi:
        findings.append(finding("YELLOW", "nvidia_smi_no_gpu_rows", "nvidia-smi ran but did not report a GPU row.", "Refresh windows_cuda_doctor before heavy CUDA work."))
    else:
        findings.append(finding("YELLOW", "nvidia_smi_missing", "nvidia-smi is not on PATH.", "This blocks CUDA proof only on Windows, not Mac demo readiness."))
    return {
        "backend": "windows_cuda",
        "ready": ready,
        "status": "GREEN" if ready else "YELLOW",
        "windows_cuda_doctor": compact_report(cuda_report),
        "nvidia_smi": live,
        "findings": findings,
    }


def theseus_governance_probe() -> dict[str, Any]:
    readiness = read_json(REPORTS / "public_calibration_readiness_packet.json", {})
    decoder = read_json(REPORTS / "decoder_v2_private_ablation_gate.json", {})
    proof = read_json(REPORTS / "private_public_transfer_proof.json", {})
    maturity = read_json(REPORTS / "maturity_integrity_audit.json", {})
    promotion = read_json(REPORTS / "candidate_promotion_gate.json", {})
    growth = read_json(REPORTS / "model_growth_gate.json", {})
    walls = read_json(REPORTS / "code_transfer_governance_remaining_walls.json", {})
    residual_report = read_json(REPORTS / "public_code_transfer_residual_report.json", {})
    readiness_gates = {
        str(row.get("name")): row
        for row in readiness.get("gates", [])
        if isinstance(row, dict) and row.get("name")
    }
    canonical = readiness_gates.get("canonical_broad_floor_v2_artifacts", {}).get("detail", {})
    return {
        "readiness": compact_report(readiness),
        "decoder_gate": compact_report(decoder),
        "transfer_proof": compact_report(proof),
        "maturity_integrity": compact_report(maturity),
        "candidate_promotion": {"promote": bool(promotion.get("promote")), "passed": promotion.get("passed"), "total": promotion.get("total")},
        "model_growth": {"model_growth_allowed": bool(growth.get("model_growth_allowed")), "hard_blockers": growth.get("hard_blockers", [])},
        "canonical_broad_floor": {
            "passed": bool(canonical.get("passed")),
            "canonical_slug": canonical.get("canonical_slug"),
            "stale_artifact_count": int(canonical.get("stale_artifact_count") or 0),
            "noncanonical_artifact_count": int(canonical.get("noncanonical_artifact_count") or 0),
        },
        "remaining_walls": [row.get("id") for row in walls.get("remaining_walls", []) if isinstance(row, dict)],
        "fresh_residual_categories": residual_report.get("category_counts") or residual_report.get("failure_category_counts") or {},
    }


def demo_path(mode: str, target: str, host: dict[str, Any], backend: dict[str, Any], theseus: dict[str, Any]) -> dict[str, Any]:
    memory_gb = host.get("unified_memory_gb") or host.get("total_memory_gb")
    memory_class = "small_unified_memory" if target == "apple_mlx" and memory_gb and memory_gb <= 16 else "normal"
    live_steps = [
        {
            "name": "Open the demo dashboard/report",
            "command": "python scripts/travel_demo_preflight.py --mode parents_demo",
            "expected_seconds": 3,
        },
        {
            "name": "Show honesty gates",
            "talk_track": "Theseus can say what is ready, what is locked, and what still needs work.",
            "expected_seconds": 60,
        },
        {
            "name": "Show private A/B improvement",
            "talk_track": "The system found broad code-transfer residuals and improved private heldout behavior without training on public answers.",
            "expected_seconds": 90,
        },
        {
            "name": "Show remaining walls",
            "talk_track": "It does not claim victory; it names adapter/runtime, interface, return-shape, planning, and verifier gaps.",
            "expected_seconds": 90,
        },
    ]
    if mode == "technical_demo":
        live_steps.append(
            {
                "name": "Optional private-only recovery check",
                "command": "python scripts/broad_public_code_transfer_floor_recovery.py --execute-ablation --task-limit 8 --candidates-per-task 4",
                "expected_seconds": 60,
                "warning": "Run only if plugged in and the machine is cool.",
            }
        )
    return {
        "audience": "parents/non-specialists" if mode == "parents_demo" else "technical",
        "memory_class": memory_class,
        "recommended_demo_style": "cached_evidence_plus_tiny_backend_smoke",
        "avoid": [
            "long training",
            "public calibration",
            "model growth",
            "candidate promotion",
            "large local LLM inference on a 16GB M1 while screen-sharing",
        ],
        "live_steps": live_steps,
        "offline_fallback_reports": [
            "reports/travel_demo_preflight.md",
            "reports/code_transfer_governance_remaining_walls.md",
            "reports/public_calibration_readiness_packet.md",
            "reports/decoder_v2_private_ablation_gate.md",
            "reports/private_public_transfer_proof.md",
            "reports/maturity_integrity_audit.md",
        ],
    }


def build_gates(
    mode: str,
    target: str,
    host: dict[str, Any],
    backend: dict[str, Any],
    theseus: dict[str, Any],
    demo: dict[str, Any],
) -> list[dict[str, Any]]:
    readiness = theseus.get("readiness", {})
    decoder = theseus.get("decoder_gate", {})
    proof = theseus.get("transfer_proof", {})
    maturity = theseus.get("maturity_integrity", {})
    promotion = theseus.get("candidate_promotion", {})
    growth = theseus.get("model_growth", {})
    canonical = theseus.get("canonical_broad_floor", {})
    gates = [
        gate("backend_target_detected", target in {"apple_mlx", "windows_cuda", "cpu_fallback"}, "hard", {"target": target}),
        gate("accelerated_backend_ready_or_demo_fallback", backend.get("ready") or target == "cpu_fallback", "hard", backend),
        gate("mac_16gb_demo_profile_used", demo.get("memory_class") != "small_unified_memory" or mode == "parents_demo", "hard", demo.get("memory_class")),
        gate("canonical_broad_floor_v2", bool(canonical.get("passed")) and canonical.get("stale_artifact_count") == 0, "hard", canonical),
        gate("readiness_report_fresh_green", readiness.get("trigger_state") == "GREEN", "hard", readiness),
        gate("decoder_gate_green", decoder.get("trigger_state") == "GREEN" and decoder.get("ready_for_public_calibration") is True, "hard", decoder),
        gate("transfer_proof_green", proof.get("trigger_state") == "GREEN" and proof.get("ready_for_public_calibration") is True, "hard", proof),
        gate("public_calibration_locked", readiness.get("public_calibration_allowed") is False and readiness.get("operator_lock_active") is True, "hard", readiness),
        gate("candidate_promotion_locked", promotion.get("promote") is False, "hard", promotion),
        gate("model_growth_locked", growth.get("model_growth_allowed") is False, "hard", growth),
        gate("maturity_has_no_hard_blockers", int_or_default(maturity.get("hard_blocker_count"), 1) == 0, "hard", maturity),
        gate("remaining_walls_named", len(theseus.get("remaining_walls") or []) > 0, "warning", theseus.get("remaining_walls")),
    ]
    return gates


def state_from_gates(gates: list[dict[str, Any]]) -> str:
    hard_fail = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warn_fail = [row for row in gates if row["severity"] == "warning" and not row["passed"]]
    if hard_fail:
        return "RED"
    if warn_fail:
        return "YELLOW"
    return "GREEN"


def next_actions(trigger_state: str, target: str, backend: dict[str, Any], theseus: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if target == "apple_mlx" and not backend.get("ready"):
        actions.append("On the MacBook, install MLX in the active environment: python3 -m pip install mlx")
    if target == "apple_mlx":
        actions.append("Use the parents_demo profile on 16GB M1: cached reports plus tiny MLX smoke, not long training.")
    if trigger_state == "GREEN":
        actions.append("Demo-ready: run python scripts/travel_demo_preflight.py --mode parents_demo shortly before leaving.")
    else:
        actions.append("Fix RED demo gates before traveling; do not improvise with long training during the demo.")
    if not theseus.get("readiness", {}).get("operator_lock_active"):
        actions.append("Restore reports/public_calibration_operator_lock.flag before demo.")
    return actions


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    keys = [
        "trigger_state",
        "status",
        "created_utc",
        "ready_for_public_calibration",
        "technical_ready_for_one_bounded_4_card_calibration",
        "public_calibration_allowed",
        "operator_lock_active",
    ]
    out = {key: report.get(key) for key in keys if key in report}
    for key in [
        "broad_public_pass_rate",
        "public_calibration_allowed",
        "candidate_promotion_allowed",
        "model_growth_allowed",
        "hard_blocker_count",
        "maturity_blocker_count",
        "public_tests_or_solutions_visible",
        "stale_artifact_count",
    ]:
        if key in summary:
            out[key] = summary[key]
    return out


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def finding(state: str, finding_id: str, detail: str, action: str) -> dict[str, str]:
    return {"state": state, "id": finding_id, "detail": detail, "action": action}


def run_python_json(code: str) -> dict[str, Any]:
    try:
        proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=20)
        if proc.returncode != 0:
            return {"error": proc.stderr.strip() or proc.stdout.strip(), "returncode": proc.returncode}
        return json.loads(proc.stdout.strip() or "{}")
    except Exception as exc:
        return {"error": str(exc)}


def run_json_command(command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=15)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"error": str(exc)}


def run_text(command: list[str]) -> str:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=10)
        if proc.returncode == 0:
            return proc.stdout.strip()
        return ""
    except Exception:
        return ""


def windows_memory_gb() -> int | None:
    payload = run_python_json(
        "import json\n"
        "try:\n"
        " import ctypes\n"
        " class MEMORYSTATUSEX(ctypes.Structure):\n"
        "  _fields_=[('dwLength', ctypes.c_ulong),('dwMemoryLoad', ctypes.c_ulong),('ullTotalPhys', ctypes.c_ulonglong),('ullAvailPhys', ctypes.c_ulonglong),('ullTotalPageFile', ctypes.c_ulonglong),('ullAvailPageFile', ctypes.c_ulonglong),('ullTotalVirtual', ctypes.c_ulonglong),('ullAvailVirtual', ctypes.c_ulonglong),('sullAvailExtendedVirtual', ctypes.c_ulonglong)]\n"
        " stat=MEMORYSTATUSEX(); stat.dwLength=ctypes.sizeof(MEMORYSTATUSEX); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)); print(json.dumps({'gb': round(stat.ullTotalPhys/(1024**3))}))\n"
        "except Exception as e: print(json.dumps({'error':str(e)}))\n"
    )
    return payload.get("gb") if isinstance(payload.get("gb"), int) else None


def round_int_gb(value: str) -> int | None:
    try:
        return round(int(value) / (1024**3))
    except Exception:
        return None


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return ROOT / value


def get_path(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def render_markdown(report: dict[str, Any]) -> str:
    backend = report["backend"]
    host = report["host"]
    demo = report["demo"]
    lines = [
        "# Project Theseus Travel Demo Preflight",
        "",
        f"- Trigger state: `{report['trigger_state']}`",
        f"- Target backend: `{report['target_backend']}`",
        f"- Host: `{host.get('platform')}`",
        f"- Backend ready: `{backend.get('ready')}`",
        f"- Demo style: `{demo.get('recommended_demo_style')}`",
        "",
        "## Gates",
    ]
    for item in report["gates"]:
        lines.append(f"- `{item['name']}`: `{item['passed']}` ({item['severity']})")
    lines.extend(["", "## Live Demo Path"])
    for step in demo.get("live_steps", []):
        command = f" Command: `{step['command']}`." if step.get("command") else ""
        talk = f" {step['talk_track']}" if step.get("talk_track") else ""
        lines.append(f"- {step['name']}.{command}{talk}")
    lines.extend(["", "## Avoid"])
    for item in demo.get("avoid", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
