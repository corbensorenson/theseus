"""Windows/CUDA readiness doctor for the local Project Theseus node.

This report is intentionally operational and cheap: it gathers the Windows
NVIDIA toolchain, resource governor, Hive scheduler, and worker-chunk evidence
that is otherwise spread across several reports. It does not run training
smokes or mutate benchmark/training data.
"""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_OUT = REPORTS / "windows_cuda_doctor.json"
DEFAULT_MARKDOWN = REPORTS / "windows_cuda_doctor.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--refresh", action="store_true", help="Refresh cheap resource/scheduler/performance reports first.")
    parser.add_argument(
        "--refresh-preflight",
        action="store_true",
        help="Also refresh training_preflight without build checks or smokes.",
    )
    parser.add_argument("--stale-hours", type=float, default=6.0)
    args = parser.parse_args()

    commands = refresh_reports(args) if args.refresh or args.refresh_preflight else []
    reports = collect_reports()
    live = collect_live_environment()
    report = build_report(args, live, reports, commands)
    write_json(ROOT / args.out, report)
    write_text(ROOT / args.markdown_out, markdown_report(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] != "RED" else 2


def refresh_reports(args: argparse.Namespace) -> list[dict[str, Any]]:
    commands = [
        [sys.executable, "scripts/resource_governor.py", "--profile", args.profile, "--out", "reports/resource_governor.json"],
        [sys.executable, "scripts/hive_scheduler.py", "--out", "reports/hive_scheduler.json"],
        [
            sys.executable,
            "scripts/performance_optimizer.py",
            "--out",
            "reports/performance_optimizer.json",
            "--markdown-out",
            "reports/performance_optimizer.md",
        ],
    ]
    if args.refresh_preflight:
        commands.append([sys.executable, "scripts/training_preflight.py", "--out", "reports/training_preflight_report.json"])
    return [compact_command(run_command(command, timeout=180, allow_failure=True)) for command in commands]


def collect_reports() -> dict[str, Any]:
    return {
        "resource_governor": read_json(REPORTS / "resource_governor.json", {}),
        "performance_optimizer": read_json(REPORTS / "performance_optimizer.json", {}),
        "training_preflight": read_json(REPORTS / "training_preflight_report.json", {}),
        "hive_scheduler": read_json(REPORTS / "hive_scheduler.json", {}),
        "hive_status": read_json(REPORTS / "hive_status.json", {}),
        "worker_chunks": read_jsonl_tail(REPORTS / "hive_worker_chunk_ledger.jsonl", 80),
    }


def collect_live_environment() -> dict[str, Any]:
    nvidia = query_nvidia_smi()
    nvcc = run_command(["nvcc", "--version"], timeout=15, allow_failure=True)
    rustc_cmd = resolve_tool("rustc")
    cargo_cmd = resolve_tool("cargo")
    cl = run_command(["where.exe", "cl"], timeout=10, allow_failure=True) if platform.system() == "Windows" else missing_command("where.exe")
    power = (
        run_command(["powercfg", "/GETACTIVESCHEME"], timeout=10, allow_failure=True)
        if platform.system() == "Windows"
        else missing_command("powercfg")
    )
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "executable": sys.executable,
        },
        "nvidia_smi": nvidia,
        "nvcc": compact_command(nvcc),
        "cuda_toolkit": extract_cuda_release(nvcc.get("stdout", "")),
        "rustc_path": rustc_cmd[0],
        "cargo_path": cargo_cmd[0],
        "rustc": single_line(run_command([*rustc_cmd, "--version"], timeout=15, allow_failure=True).get("stdout")),
        "cargo": single_line(run_command([*cargo_cmd, "--version"], timeout=15, allow_failure=True).get("stdout")),
        "cl_visible": cl.get("returncode") == 0,
        "cl_where": cl.get("stdout", "").strip(),
        **check_msvc_dev_shell(),
        "power_plan": parse_power_plan(power.get("stdout", "")),
        "symliquid_release_binary": release_binary_status(),
    }


def build_report(
    args: argparse.Namespace,
    live: dict[str, Any],
    reports: dict[str, Any],
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    gpu = first_gpu(live.get("nvidia_smi", {}))
    resource = reports["resource_governor"]
    performance = reports["performance_optimizer"]
    preflight = reports["training_preflight"]
    scheduler = reports["hive_scheduler"]
    chunks = [row for row in reports["worker_chunks"] if isinstance(row, dict)]
    cuda_chunks = [row for row in chunks if str(row.get("backend")) == "rust_cuda"]
    recent_ok_cuda = [row for row in cuda_chunks if row.get("ok")]

    summary = {
        "node_role": "windows_cuda_hot_path",
        "gpu_name": gpu.get("name"),
        "driver_version": gpu.get("driver_version"),
        "cuda_toolkit": live.get("cuda_toolkit"),
        "compute_capability": gpu.get("compute_cap"),
        "vram_total_mib": number(gpu.get("memory_total_mib")),
        "vram_free_mib": number(gpu.get("memory_free_mib")),
        "vram_used_mib": number(gpu.get("memory_used_mib")),
        "gpu_utilization_percent": number(gpu.get("utilization_gpu_percent")),
        "gpu_temperature_c": number(gpu.get("temperature_c")),
        "gpu_power_w": number(gpu.get("power_draw_w")),
        "rustc": live.get("rustc"),
        "cargo": live.get("cargo"),
        "cl_visible": live.get("cl_visible"),
        "vsdevcmd_cl_visible": live.get("vsdevcmd_cl_visible"),
        "release_binary_present": get_path(live, ["symliquid_release_binary", "present"], False),
        "resource_can_run_profile": get_path(resource, ["decision", "can_run_requested_profile"], None),
        "resource_throttle_reasons": get_path(resource, ["decision", "throttle_reasons"], []),
        "resource_warnings": get_path(resource, ["decision", "warnings"], []),
        "recommended_profile": get_path(resource, ["decision", "recommended_profile"], None),
        "performance_state": performance.get("trigger_state"),
        "performance_score": performance.get("score"),
        "scheduler_worker_chunks": int(number(get_path(scheduler, ["summary", "real_worker_chunks"], 0))),
        "recent_cuda_worker_chunks": len(cuda_chunks),
        "recent_ok_cuda_worker_chunks": len(recent_ok_cuda),
        "best_cuda_node": get_path(scheduler, ["summary", "best_cuda_node"], None),
        "best_training_node": get_path(scheduler, ["summary", "best_training_node"], None),
        "runtime_root": get_path(resource, ["resource_envelope", "runtime_root"], None),
        "root_free_gib": get_path(resource, ["current_resources", "disk", "free_gib"], None),
        "spillover_free_gib": get_path(resource, ["current_resources", "spillover", "selected", "free_gib"], None),
        "preflight_heavy_training_allowed": preflight.get("heavy_training_allowed"),
        "preflight_blockers": [row.get("gate") for row in preflight.get("blockers", []) if isinstance(row, dict)],
        "power_plan": live.get("power_plan"),
    }

    findings = build_findings(args, live, reports, summary)
    trigger_state = doctor_state(findings)
    next_actions = build_next_actions(findings, summary)
    freshness = report_freshness(args, reports)
    readiness = {
        "rust_cuda_hot_path_ready": bool(
            gpu.get("available")
            and live.get("cuda_toolkit")
            and live.get("rustc")
            and live.get("cargo")
            and get_path(live, ["symliquid_release_binary", "present"], False)
        ),
        "native_windows_build_shell_ready": bool(live.get("cl_visible") or live.get("vsdevcmd_cl_visible")),
        "unattended_cuda_work_ready": bool(
            get_path(resource, ["decision", "can_run_requested_profile"], False)
            and performance.get("trigger_state") != "RED"
            and summary["scheduler_worker_chunks"] >= 3
        ),
        "report_stale": any(bool(row.get("stale")) for row in freshness.values()),
    }
    return {
        "ok": trigger_state != "RED",
        "policy": "project_theseus_windows_cuda_doctor_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "profile": args.profile,
        "summary": summary,
        "readiness": readiness,
        "findings": findings,
        "next_actions": next_actions,
        "refresh_commands": commands,
        "live_environment": live,
        "report_freshness": freshness,
        "source_reports": {
            "resource_governor": "reports/resource_governor.json",
            "performance_optimizer": "reports/performance_optimizer.json",
            "training_preflight": "reports/training_preflight_report.json",
            "hive_scheduler": "reports/hive_scheduler.json",
            "worker_chunks": "reports/hive_worker_chunk_ledger.jsonl",
        },
    }


def build_findings(
    args: argparse.Namespace,
    live: dict[str, Any],
    reports: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    gpu = first_gpu(live.get("nvidia_smi", {}))
    add = findings.append
    if not gpu.get("available"):
        add(finding("RED", "nvidia_smi_unavailable", "No NVIDIA GPU is visible through nvidia-smi.", "Install/repair NVIDIA driver or run this node as CPU-only."))
    elif not gpu_matches_configured_profile(gpu.get("name"), reports):
        expected = get_path(reports, ["resource_governor", "hardware_profile", "name"], "configured CUDA profile")
        add(finding("YELLOW", "unexpected_cuda_gpu", f"Visible GPU is {gpu.get('name')}; profiles are tuned for {expected}.", "Keep smoke profile until a matching profile is added."))
    if not live.get("cuda_toolkit"):
        add(finding("RED", "nvcc_unavailable", "CUDA toolkit is not visible through nvcc.", "Install CUDA toolkit or add nvcc to PATH."))
    if not live.get("rustc") or not live.get("cargo"):
        add(finding("RED", "rust_toolchain_unavailable", "Rust cargo/rustc are not visible.", "Install Rust or add %USERPROFILE%\\.cargo\\bin to PATH."))
    if not get_path(live, ["symliquid_release_binary", "present"], False):
        add(finding("RED", "symliquid_release_binary_missing", "target/release/symliquid-cli.exe is missing.", "Build with cargo build --release -p symliquid-cli --features cuda."))
    if not (live.get("cl_visible") or live.get("vsdevcmd_cl_visible")):
        add(finding("YELLOW", "msvc_compiler_not_visible", "MSVC cl.exe is not visible in the shell or VSDevCmd probe.", "Install Visual Studio Build Tools with C++ tools."))
    elif not live.get("cl_visible") and live.get("vsdevcmd_cl_visible"):
        add(finding("INFO", "msvc_requires_dev_shell", "cl.exe works after VsDevCmd but is not in the plain shell.", "Use scripts/use_msvc_dev_shell.ps1 for native-extension work."))

    temp = number(summary.get("gpu_temperature_c"))
    if temp >= 88:
        add(finding("RED", "gpu_temperature_critical", f"GPU temperature is {temp:.0f} C.", "Pause CUDA work and improve cooling before unattended training."))
    elif temp >= 82:
        add(finding("YELLOW", "gpu_temperature_high", f"GPU temperature is {temp:.0f} C.", "Prefer smoke jobs or pause long runs until cooling improves."))

    free = number(summary.get("vram_free_mib"))
    if free and free < 1400:
        add(finding("YELLOW", "low_free_vram", f"Free VRAM is {free:.0f} MiB.", "Close GPU-heavy apps or keep only smoke worker chunks."))
    util = number(summary.get("gpu_utilization_percent"))
    if util >= 92:
        add(finding("YELLOW", "gpu_busy", f"GPU utilization is {util:.0f}%.", "Defer new CUDA chunks until the current work quiets."))

    if summary.get("resource_can_run_profile") is False:
        reasons = ", ".join(str(row) for row in summary.get("resource_throttle_reasons") or [])
        add(finding("YELLOW", "resource_governor_throttled", reasons or "Resource governor blocked requested profile.", "Let the queue wait, free VRAM, or switch to the recommended profile."))
    for warning in summary.get("resource_warnings") or []:
        add(finding("INFO", "resource_governor_warning", str(warning), "Runtime/spillover is already routing generated data off C: where possible."))

    if str(summary.get("performance_state") or "").upper() == "RED":
        add(finding("RED", "performance_optimizer_red", "Performance optimizer is RED.", "Open reports/performance_optimizer.md before long unattended runs."))
    elif str(summary.get("performance_state") or "").upper() == "YELLOW":
        add(finding("YELLOW", "performance_optimizer_yellow", "Performance optimizer is YELLOW.", "Check bottlenecks and run bounded worker chunks before long jobs."))
    if summary["scheduler_worker_chunks"] < 3:
        add(finding("YELLOW", "scheduler_worker_chunks_thin", f"Scheduler planned {summary['scheduler_worker_chunks']} real worker chunks.", "Run theseus schedule --execute --worker-chunks after resources clear."))
    if summary["recent_ok_cuda_worker_chunks"] == 0:
        add(finding("YELLOW", "no_recent_ok_cuda_worker_chunk", "No recent accepted CUDA worker chunk is recorded.", "Run a bounded cuda_eval_chunk or performance smoke before trusting overnight CUDA work."))

    preflight = reports["training_preflight"]
    blockers = preflight.get("blockers") if isinstance(preflight.get("blockers"), list) else []
    for blocker in blockers:
        if isinstance(blocker, dict):
            gate = str(blocker.get("gate") or "preflight_blocker")
            add(finding("YELLOW", f"preflight_{gate}", str(blocker.get("evidence") or gate), "This blocks heavy training, not cheap CUDA readiness checks."))

    for name, report in {
        "resource_governor": reports["resource_governor"],
        "performance_optimizer": reports["performance_optimizer"],
        "training_preflight": reports["training_preflight"],
        "hive_scheduler": reports["hive_scheduler"],
    }.items():
        freshness = single_report_freshness(args, report)
        if not freshness["exists"]:
            add(finding("YELLOW", f"{name}_missing", f"{name} report is missing.", "Run this doctor with --refresh."))
        elif freshness["age_hours"] is None:
            add(finding("INFO", f"{name}_timestamp_missing", f"{name} does not record created_utc.", "Report is usable, but adding created_utc would improve stale-lane detection."))
        elif freshness["age_hours"] is not None and freshness["age_hours"] > args.stale_hours:
            add(finding("YELLOW", f"{name}_stale", f"{name} is {freshness['age_hours']:.1f} hours old.", "Run this doctor with --refresh."))
    return findings


def gpu_matches_configured_profile(gpu_name: Any, reports: dict[str, Any]) -> bool:
    expected = str(get_path(reports, ["resource_governor", "hardware_profile", "name"], "") or "")
    observed = str(gpu_name or "")
    if not expected or not observed:
        return False
    return normalize_gpu_name(expected) in normalize_gpu_name(observed)


def normalize_gpu_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def build_next_actions(findings: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    ids = {str(row.get("id")) for row in findings}
    actions = []
    if any(item.endswith("_stale") or (item.endswith("_missing") and not item.endswith("_timestamp_missing")) for item in ids):
        actions.append("python scripts\\windows_cuda_doctor.py --refresh --out reports\\windows_cuda_doctor.json --markdown-out reports\\windows_cuda_doctor.md")
    if "symliquid_release_binary_missing" in ids:
        actions.append("cargo build --release -p symliquid-cli --features cuda")
    if "msvc_requires_dev_shell" in ids:
        actions.append("powershell -ExecutionPolicy Bypass -File scripts\\use_msvc_dev_shell.ps1")
    if "resource_governor_throttled" in ids or "low_free_vram" in ids:
        actions.append("close_GPU_heavy_apps_or_wait_for_current_training_then_rerun_resource_governor")
    if "scheduler_worker_chunks_thin" in ids or "no_recent_ok_cuda_worker_chunk" in ids:
        actions.append("theseus schedule --execute --worker-chunks")
    if "performance_optimizer_yellow" in ids or "performance_optimizer_red" in ids:
        actions.append("open reports\\performance_optimizer.md and clear first bottleneck before vacation mode")
    if not actions:
        actions.append("windows_cuda_node_ready_for_bounded_cuda_worker_chunks")
    if summary.get("runtime_root"):
        actions.append(f"keep_generated_runtime_on {summary.get('runtime_root')}")
    return actions


def doctor_state(findings: list[dict[str, Any]]) -> str:
    severities = {str(row.get("severity")).upper() for row in findings}
    if "RED" in severities:
        return "RED"
    if "YELLOW" in severities:
        return "YELLOW"
    return "GREEN"


def query_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = run_command(command, timeout=10, allow_failure=True)
    if result["returncode"] != 0 or not result.get("stdout", "").strip():
        return {"available": False, "command": compact_command(result), "error": result.get("stderr") or "nvidia_smi_failed"}
    gpus = []
    for line in result["stdout"].strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 9:
            continue
        gpus.append(
            {
                "available": True,
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mib": parse_float(parts[2]),
                "memory_used_mib": parse_float(parts[3]),
                "memory_free_mib": parse_float(parts[4]),
                "utilization_gpu_percent": parse_float(parts[5]),
                "temperature_c": parse_float(parts[6]),
                "power_draw_w": parse_float(parts[7]),
                "compute_cap": parts[8],
            }
        )
    return {"available": bool(gpus), "gpus": gpus, "command": compact_command(result)}


def first_gpu(nvidia: dict[str, Any]) -> dict[str, Any]:
    gpus = nvidia.get("gpus") if isinstance(nvidia.get("gpus"), list) else []
    if gpus and isinstance(gpus[0], dict):
        return gpus[0]
    return {"available": False}


def check_msvc_dev_shell() -> dict[str, Any]:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return {"vswhere": None, "vsdevcmd": None, "vsdevcmd_cl_visible": False}
    install = run_command(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        timeout=30,
        allow_failure=True,
    )
    path = single_line(install.get("stdout"))
    if not path:
        return {"vswhere": str(vswhere), "vsdevcmd": None, "vsdevcmd_cl_visible": False}
    devcmd = Path(path) / "Common7" / "Tools" / "VsDevCmd.bat"
    if not devcmd.exists():
        return {"vswhere": str(vswhere), "vsdevcmd": str(devcmd), "vsdevcmd_cl_visible": False}
    probe = run_shell_command(f'cmd.exe /s /c ""{devcmd}" -arch=x64 -host_arch=x64 >nul && where cl"', timeout=60)
    return {
        "vswhere": str(vswhere),
        "vsdevcmd": str(devcmd),
        "vsdevcmd_cl_visible": probe["returncode"] == 0,
        "vsdevcmd_cl_where": probe.get("stdout", "").strip(),
    }


def release_binary_status() -> dict[str, Any]:
    path = ROOT / "target" / "release" / "symliquid-cli.exe"
    return {"path": str(path), "present": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}


def report_freshness(args: argparse.Namespace, reports: dict[str, Any]) -> dict[str, Any]:
    return {name: single_report_freshness(args, report) for name, report in reports.items() if name != "worker_chunks"}


def single_report_freshness(args: argparse.Namespace, report: Any) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"exists": False, "created_utc": None, "age_hours": None, "stale": True}
    created = str(report.get("created_utc") or "")
    if not created:
        return {"exists": True, "created_utc": None, "age_hours": None, "stale": False}
    age = age_hours(created)
    return {
        "exists": True,
        "created_utc": created or None,
        "age_hours": round(age, 3) if age is not None else None,
        "stale": bool(age is None or age > args.stale_hours),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Project Theseus Windows/CUDA Doctor",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- GPU: `{summary.get('gpu_name')}` driver `{summary.get('driver_version')}` compute `{summary.get('compute_capability')}`",
        f"- CUDA toolkit: `{summary.get('cuda_toolkit')}`",
        f"- VRAM free/total MiB: `{summary.get('vram_free_mib')}` / `{summary.get('vram_total_mib')}`",
        f"- GPU temp/util: `{summary.get('gpu_temperature_c')}` C / `{summary.get('gpu_utilization_percent')}`%",
        f"- Rust/Cargo: `{summary.get('rustc')}` / `{summary.get('cargo')}`",
        f"- MSVC: plain shell `{summary.get('cl_visible')}`, VSDevCmd `{summary.get('vsdevcmd_cl_visible')}`",
        f"- Resource can run `{summary.get('resource_can_run_profile')}`, recommended profile `{summary.get('recommended_profile')}`",
        f"- Performance: `{summary.get('performance_state')}` score `{summary.get('performance_score')}`",
        f"- Worker chunks planned/recent CUDA OK: `{summary.get('scheduler_worker_chunks')}` / `{summary.get('recent_ok_cuda_worker_chunks')}`",
        f"- Runtime root: `{summary.get('runtime_root')}`",
        "",
        "## Findings",
    ]
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    if findings:
        for row in findings:
            lines.append(f"- `{row.get('severity')}` `{row.get('id')}`: {row.get('detail')} Action: {row.get('action')}")
    else:
        lines.append("- No findings.")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- `{action}`")
    return "\n".join(lines) + "\n"


def finding(severity: str, item_id: str, detail: str, action: str) -> dict[str, str]:
    return {"severity": severity, "id": item_id, "detail": detail, "action": action}


def resolve_tool(name: str) -> list[str]:
    if platform.system() == "Windows":
        probe = run_command(["where.exe", name], timeout=10, allow_failure=True)
        if probe["returncode"] == 0:
            first = probe.get("stdout", "").splitlines()[0].strip()
            if first:
                return [first]
        cargo_home = Path.home() / ".cargo" / "bin" / f"{name}.exe"
        if cargo_home.exists():
            return [str(cargo_home)]
    return [name]


def run_command(command: list[str], *, timeout: int = 120, allow_failure: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        out = {
            "command": command,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
    except Exception as exc:
        out = {
            "command": command,
            "returncode": -1,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout": "",
            "stderr": str(exc),
        }
    if out["returncode"] != 0 and not allow_failure:
        return out
    return out


def run_shell_command(command: str, *, timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, shell=True)
        return {
            "command": command,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": -1,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout": "",
            "stderr": str(exc),
        }


def missing_command(name: str) -> dict[str, Any]:
    return {"command": [name], "returncode": -1, "runtime_ms": 0, "stdout": "", "stderr": "not_applicable"}


def compact_command(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": result.get("command"),
        "returncode": result.get("returncode"),
        "runtime_ms": result.get("runtime_ms"),
        "stdout": result.get("stdout", "")[-1200:],
        "stderr": result.get("stderr", "")[-1200:],
    }


def parse_power_plan(text: str) -> str | None:
    line = single_line(text)
    return line


def extract_cuda_release(text: str) -> str | None:
    for line in text.splitlines():
        if "release" in line:
            return line.strip()
    return None


def single_line(text: Any) -> str | None:
    if not text:
        return None
    stripped = str(text).strip()
    return stripped.splitlines()[0] if stripped else None


def parse_float(value: Any) -> float | None:
    try:
        text = str(value).strip().replace(" MiB", "").replace(" W", "")
        num = float(text)
        return num if math.isfinite(num) else None
    except (TypeError, ValueError):
        return None


def number(value: Any) -> float:
    parsed = parse_float(value)
    return parsed if parsed is not None else 0.0


def age_hours(value: str) -> float | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds() / 3600.0


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl_tail(path: Path, limit: int) -> list[Any]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
