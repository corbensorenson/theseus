#!/usr/bin/env python3
"""Choose bounded Theseus work budgets from current host resources."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
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

REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "resource_aware_execution_policy.json"
DEFAULT_MARKDOWN = REPORTS / "resource_aware_execution_policy.md"

from code_lm_active_worker_monitor import (  # noqa: E402
    infer_active_worker_slug,
    phase_heartbeat_for_active_phase,
    summarize_active_phase_heartbeat,
)
from code_lm_process_guard import (  # noqa: E402
    duplicate_code_lm_artifact_targets,
    is_code_lm_worker_command,
    logical_code_lm_process_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()
    started = time.perf_counter()

    memory = memory_status()
    disks = disk_status()
    processes = process_snapshot()
    gpu = nvidia_status()
    host = host_status()
    accelerator = accelerator_status(host=host, gpu=gpu)
    logical_cores = os.cpu_count() or 1
    heavy = [row for row in processes if heavy_process(row)]
    code_heavy = logical_code_lm_process_rows([row for row in processes if code_lm_process(row)])
    duplicate_code_lm_artifacts = duplicate_code_lm_artifact_targets(code_heavy, ROOT)
    code_lm_cpu_bound = any(cpu_bound_code_lm_process(row) for row in code_heavy)
    code_lm_cuda_readout = any(cuda_readout_code_lm_process(row) for row in code_heavy)
    latest_heartbeat = latest_code_lm_heartbeat()
    command_stage = active_code_lm_command_stage(code_heavy)
    active_phase_heartbeat = latest_train_once_phase_heartbeat(code_heavy, command_stage)
    code_lm_heartbeat = (
        latest_heartbeat
        if code_heavy
        and float(latest_heartbeat.get("age_seconds") or 1_000_000.0) <= 1800.0
        and heartbeat_matches_active_code_lm(latest_heartbeat, code_heavy)
        else {}
    )
    if active_phase_heartbeat:
        code_lm_heartbeat = active_phase_heartbeat
    code_lm_stage = str(
        code_lm_heartbeat.get("stage")
        or code_lm_heartbeat.get("progress_stage")
        or command_stage.get("stage")
        or ""
    )
    code_lm_phase = str(code_lm_heartbeat.get("phase") or command_stage.get("phase") or "")
    code_lm_progress_stage = str(
        code_lm_heartbeat.get("latest_progress_stage")
        or code_lm_heartbeat.get("progress_stage")
        or ""
    )
    gpu_util = gpu.get("utilization_gpu_percent")
    code_lm_gpu_low_utilization_bottleneck, code_lm_gpu_low_utilization_reason = low_gpu_utilization_bottleneck(
        code_lm_cuda_readout=code_lm_cuda_readout,
        gpu_utilization_percent=gpu_util,
        stage=code_lm_stage,
        phase=code_lm_phase,
        progress_stage=code_lm_progress_stage,
    )
    free_gb = memory.get("available_gb", 0.0)
    primary_disk = primary_disk_label(disks)
    primary_free_gb = disks.get(primary_disk, {}).get("free_gb", 0.0) if primary_disk else 0.0
    workspace_free_gb = disks.get("workspace", {}).get("free_gb", primary_free_gb)
    d_free_gb = disks.get("D:/", {}).get("free_gb", 0.0)
    c_free_gb = disks.get("C:/", {}).get("free_gb", 0.0)

    recent_timeout = recent_code_lm_timeout()
    repeated_recovery_timeout = recent_recovery_smoke_timeout()
    if code_heavy:
        profile = "defer_code_closure_existing_heavy_code_worker"
    elif free_gb < 6.0 or workspace_free_gb < 20.0 or len(heavy) >= max(6, logical_cores):
        profile = "conservative"
    elif free_gb < 12.0 or len(heavy) >= max(4, logical_cores // 2):
        profile = "balanced"
    else:
        profile = "throughput"
    if profile != "defer_code_closure_existing_heavy_code_worker" and repeated_recovery_timeout:
        profile = "defer_code_closure_repeated_recovery_timeout"
    elif profile != "defer_code_closure_existing_heavy_code_worker" and recent_timeout:
        profile = "timeout_recovery_smoke"

    budgets = budgets_for(profile, accelerator=accelerator)
    deferred = profile.startswith("defer_code_closure")
    trigger_state = "GREEN" if not deferred else "YELLOW"
    if code_lm_cpu_bound or code_lm_gpu_low_utilization_bottleneck or duplicate_code_lm_artifacts:
        trigger_state = "YELLOW"
    report = {
        "policy": "project_theseus_resource_aware_execution_policy_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "platform": host.get("platform"),
            "machine": host.get("machine"),
            "accelerator_backend": accelerator.get("backend"),
            "accelerator_available": accelerator.get("available"),
            "accelerator_detail": accelerator.get("detail"),
            "profile": profile,
            "logical_cores": logical_cores,
            "available_memory_gb": free_gb,
            "total_memory_gb": memory.get("total_gb"),
            "primary_disk": primary_disk,
            "primary_free_gb": primary_free_gb,
            "workspace_free_gb": workspace_free_gb,
            "c_free_gb": c_free_gb,
            "d_free_gb": d_free_gb,
            "heavy_process_count": len(heavy),
            "active_code_lm_process_count": len(code_heavy),
            "duplicate_code_lm_artifact_target_count": len(duplicate_code_lm_artifacts),
            "duplicate_code_lm_artifact_targets": duplicate_code_lm_artifacts[:5],
            "max_parallel_heavy_closures": 1,
            "run_public_calibration": False,
            "recent_code_lm_timeout": recent_timeout,
            "recent_recovery_smoke_timeout": repeated_recovery_timeout,
            "gpu_name": gpu.get("name"),
            "gpu_utilization_percent": gpu.get("utilization_gpu_percent"),
            "gpu_memory_free_mib": gpu.get("memory_free_mib"),
            "code_lm_cpu_bound_hot_path_active": code_lm_cpu_bound,
            "code_lm_cuda_readout_active": code_lm_cuda_readout,
            "active_code_lm_stage": code_lm_stage,
            "active_code_lm_phase": code_lm_phase,
            "active_code_lm_progress_stage": code_lm_progress_stage,
            "active_code_lm_heartbeat": code_lm_heartbeat,
            "active_train_once_phase_heartbeat": active_phase_heartbeat,
            "latest_code_lm_heartbeat": latest_heartbeat,
            "code_lm_gpu_low_utilization_bottleneck": code_lm_gpu_low_utilization_bottleneck,
            "code_lm_gpu_low_utilization_reason": code_lm_gpu_low_utilization_reason,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "recommended_code_lm_budget": budgets,
        "resource_snapshot": {
            "memory": memory,
            "disks": disks,
            "gpu": gpu,
            "heavy_processes": heavy[:24],
            "active_code_lm_processes": code_heavy[:12],
            "duplicate_code_lm_artifact_targets": duplicate_code_lm_artifacts[:12],
        },
        "rules": {
            "machine_usability": "one heavy code closure at a time; prefer bounded work-step budgets over wall-clock hope",
            "accelerator_hot_path": accelerator.get("routing_rule"),
            "code_lm_execution_envelope": "prefer train-once checkpoint fanout; chunked repeated-training shards are diagnostic/recovery debt, not the default training architecture",
            "low_gpu_utilization": "CUDA Code LM with low GPU utilization is YELLOW unless the heartbeat says it is in an explicit CPU-control phase such as parser/AST masking, candidate generation, filtering, artifact writing, or orchestration. Post-readout auxiliary decoder joins are tracked as a low-GPU bottleneck.",
            "public_calibration": "never launch from this policy; public calibration remains gated by decoder_v2_private_ablation_gate and transfer proof",
            "artifact_disk": artifact_disk_rule(host),
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0


def budgets_for(profile: str, *, accelerator: dict[str, Any] | None = None) -> dict[str, Any]:
    accelerator = accelerator or {"backend": "cpu", "flag": "", "available": False}
    if profile == "defer_code_closure_existing_heavy_code_worker":
        return {
            "start_new_code_closure": False,
            "start_new_chunked_code_closure": False,
            "start_new_train_once_fanout": False,
            "reason": "existing Code LM/SymLiquid/Cargo worker is active",
            "private_count": 0,
            "epochs": 0,
            "candidates_per_task": 0,
            "max_high_transfer_private_train": 0,
            "max_rust_work_steps": 0,
            "rust_timeout_seconds": 0,
            "sts_timeout_seconds": 0,
        }
    if profile == "defer_code_closure_repeated_recovery_timeout":
        return {
            "start_new_code_closure": False,
            "start_new_chunked_code_closure": False,
            "start_new_train_once_fanout": True,
            "reason": "recent timeout-recovery smoke also timed out; use train-once checkpoint fanout instead of repeated-training shards",
            "train_once_fanout_reason": "train a reusable checkpoint once, then generate/evaluate candidates from that checkpoint without retraining per shard",
            "chunk_private_count": 32,
            "chunk_epochs": 1,
            "chunk_candidates_per_task": 3,
            "chunk_max_high_transfer_private_train": 320,
            "chunk_max_rust_work_steps": 180_000,
            "chunk_rust_timeout_seconds": 1_200,
            "chunk_sts_timeout_seconds": 600,
            "private_count": 0,
            "epochs": 0,
            "candidates_per_task": 0,
            "max_high_transfer_private_train": 0,
            "max_rust_work_steps": 0,
            "rust_timeout_seconds": 0,
            "sts_timeout_seconds": 0,
        }
    table = {
        "timeout_recovery_smoke": (64, 1, 3, 640, 250_000, 1200, 600),
        "conservative": (240, 3, 6, 3600, 2_000_000, 3600, 1200),
        "balanced": (320, 4, 8, 4800, 3_000_000, 5400, 1800),
        "throughput": (480, 4, 8, 6400, 4_000_000, 7200, 2400),
    }
    private_count, epochs, candidates, max_train, max_steps, rust_timeout, sts_timeout = table.get(profile, table["balanced"])
    return {
        "start_new_code_closure": True,
        "start_new_chunked_code_closure": False,
        "start_new_train_once_fanout": True,
        "preferred_code_lm_backend": accelerator.get("code_lm_backend") or "cpu_structural_code_lm",
        "preferred_code_lm_flag": accelerator.get("code_lm_flag") or "",
        "preferred_accelerator_backend": accelerator.get("backend") or "cpu",
        "preferred_accelerator_available": bool(accelerator.get("available")),
        "profile": profile,
        "private_count": private_count,
        "epochs": epochs,
        "candidates_per_task": candidates,
        "max_high_transfer_private_train": max_train,
        "max_rust_work_steps": max_steps,
        "rust_timeout_seconds": rust_timeout,
        "sts_timeout_seconds": sts_timeout,
        "chunk_label": f"{profile}_bounded_recovery",
    }


def memory_status() -> dict[str, Any]:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return {
            "total_gb": round(stat.ullTotalPhys / (1024**3), 3),
            "available_gb": round(stat.ullAvailPhys / (1024**3), 3),
            "memory_load_percent": int(stat.dwMemoryLoad),
        }
    if sys.platform == "darwin":
        return macos_memory_status()
    if sys.platform.startswith("linux"):
        return linux_memory_status()
    return {"total_gb": 0.0, "available_gb": 0.0, "memory_load_percent": None}


def disk_status() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    labels = ["C:/", "D:/"] if os.name == "nt" else ["/", "workspace"]
    for label in labels:
        try:
            target = ROOT if label == "workspace" else Path(label)
            usage = shutil.disk_usage(target)
        except OSError:
            continue
        out[label] = {
            "total_gb": round(usage.total / (1024**3), 3),
            "free_gb": round(usage.free / (1024**3), 3),
            "used_percent": round((usage.used / usage.total) * 100.0, 2) if usage.total else 0.0,
        }
    return out


def process_snapshot() -> list[dict[str, Any]]:
    if os.name != "nt":
        return posix_process_snapshot()
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python|symliquid|cargo|rustc' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return []
    rows = payload if isinstance(payload, list) else [payload]
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cmd = str(row.get("CommandLine") or "")
        compact.append(
            {
                "pid": row.get("ProcessId"),
                "parent_pid": row.get("ParentProcessId"),
                "name": row.get("Name"),
                "command": cmd,
                "command_preview": cmd[:320],
            }
        )
    return compact


def host_status() -> dict[str, Any]:
    return {
        "platform": sys.platform,
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.executable,
    }


def primary_disk_label(disks: dict[str, dict[str, Any]]) -> str:
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        return "workspace" if "workspace" in disks else "/"
    if "D:/" in disks:
        return "D:/"
    if "C:/" in disks:
        return "C:/"
    return next(iter(disks), "")


def artifact_disk_rule(host: dict[str, Any]) -> str:
    if str(host.get("platform") or "").startswith("win"):
        return "artifacts and temporary training data should prefer the largest non-system data disk when generated outside repository reports"
    return "artifacts and temporary training data should prefer workspace/runtime paths with retention manifests; do not emit Windows drive assumptions on Unix hosts"


def accelerator_status(*, host: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    system = str(host.get("system") or "").lower()
    machine = str(host.get("machine") or "").lower()
    if os.name == "nt" and gpu.get("available"):
        return {
            "available": True,
            "backend": "cuda",
            "detail": gpu.get("name") or "nvidia_cuda",
            "code_lm_backend": "rust_cuda_fast_sparse_code_lm_readout",
            "code_lm_flag": "--use-cuda-readout",
            "routing_rule": "new Code LM closure work should request --use-cuda-readout on CUDA-capable Windows nodes",
        }
    if system == "darwin" and ("arm64" in machine or "aarch64" in machine):
        mlx = mlx_subprocess_probe()
        if mlx.get("available"):
            return {
                "available": True,
                "backend": "mlx_apple",
                "detail": mlx.get("detail") or "mlx.core import ok",
                "code_lm_backend": "macos_mlx_structural_routes",
                "code_lm_flag": "",
                "routing_rule": "route only MLX/Metal-supported bounded work on Apple Silicon; do not emit CUDA flags on macOS",
            }
        return {
            "available": False,
            "backend": "apple_silicon_cpu",
            "detail": mlx.get("detail") or mlx.get("error") or "mlx unavailable",
            "code_lm_backend": "cpu_structural_code_lm",
            "code_lm_flag": "",
            "routing_rule": "Apple Silicon detected but MLX is unavailable; use CPU-safe work until the MLX environment probe is green",
        }
    return {
        "available": False,
        "backend": "cpu",
        "detail": "no platform accelerator detected",
        "code_lm_backend": "cpu_structural_code_lm",
        "code_lm_flag": "",
        "routing_rule": "use CPU-safe bounded work unless a platform-specific accelerator probe is green",
    }


def mlx_subprocess_probe() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mlx.core as mx; print('mlx.core ok')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if result.returncode == 0:
        return {"available": True, "detail": (result.stdout or "").strip() or "mlx.core ok"}
    return {"available": False, "error": (result.stderr or result.stdout or "").strip()[:400]}


def macos_memory_status() -> dict[str, Any]:
    try:
        total = int(
            subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip()
        )
    except Exception:
        total = 0
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page_size = 4096
        if "page size of" in vm:
            marker = vm.split("page size of", 1)[1].split("bytes", 1)[0].strip()
            page_size = int(marker)
        pages: dict[str, int] = {}
        for line in vm.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                pages[key.strip().lower()] = int(digits)
        available_pages = (
            pages.get("pages free", 0)
            + pages.get("pages inactive", 0)
            + pages.get("pages speculative", 0)
        )
        available = available_pages * page_size
    except Exception:
        available = 0
    return {
        "total_gb": round(total / (1024**3), 3) if total else 0.0,
        "available_gb": round(available / (1024**3), 3) if available else 0.0,
        "memory_load_percent": round(100.0 - (available / total * 100.0), 2) if total and available else None,
    }


def linux_memory_status() -> dict[str, Any]:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
        return {
            "total_gb": round(total / (1024**3), 3) if total else 0.0,
            "available_gb": round(available / (1024**3), 3) if available else 0.0,
            "memory_load_percent": round(100.0 - (available / total * 100.0), 2) if total and available else None,
        }
    except Exception:
        return {"total_gb": 0.0, "available_gb": 0.0, "memory_load_percent": None}


def posix_process_snapshot() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,comm=,args="],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    compact: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, name, cmd = parts
        text = f"{name} {cmd}".lower()
        if not any(token in text for token in ["python", "symliquid", "cargo", "rustc"]):
            continue
        compact.append(
            {
                "pid": int(pid) if pid.isdigit() else pid,
                "parent_pid": int(ppid) if ppid.isdigit() else ppid,
                "name": name,
                "command": cmd,
                "command_preview": cmd[:320],
            }
        )
    return compact


def nvidia_status() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "error": result.stderr.strip()}
    parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    def number(idx: int) -> float | None:
        try:
            return float(parts[idx])
        except (IndexError, ValueError):
            return None
    return {
        "available": True,
        "name": parts[0] if parts else "",
        "memory_total_mib": number(1),
        "memory_used_mib": number(2),
        "memory_free_mib": number(3),
        "utilization_gpu_percent": number(4),
        "temperature_c": number(5),
        "power_draw_w": number(6),
    }


def heavy_process(row: dict[str, Any]) -> bool:
    text = f"{row.get('name')} {row.get('command')}".lower()
    if "code_lm_train_once_fanout.py" in text and "--execute" not in text:
        return False
    if "code_lm_chunked_recovery.py" in text and "--execute" not in text:
        return False
    return any(
        token in text
        for token in [
            "hive_work_board_executor",
            "vacation_mode_supervisor",
            "autonomy_cycle",
            "multi_turn_conversation_benchmark",
            "code_lm_train_once_fanout",
            "code_lm_chunked_recovery",
            "code_lm_closure",
            "train-code-lm-closure",
            "generate-code-lm-closure-fanout",
            "train-sts-parallel-decoder",
            "symliquid",
            "cargo",
            "rustc",
        ]
    )


def code_lm_process(row: dict[str, Any]) -> bool:
    return is_code_lm_worker_command(f"{row.get('name')} {row.get('command')}")


def cpu_bound_code_lm_process(row: dict[str, Any]) -> bool:
    text = f"{row.get('name')} {row.get('command')}".lower()
    hot_path = (
        "train-code-lm-closure" in text
        or "train-code-ranker" in text
        or "code_lm_closure.py" in text
        or "train-sts-parallel-decoder" in text
    )
    return hot_path and "--use-cuda-readout" not in text


def cuda_readout_code_lm_process(row: dict[str, Any]) -> bool:
    text = f"{row.get('name')} {row.get('command')}".lower()
    hot_path = (
        "train-code-lm-closure" in text
        or "train-code-ranker" in text
        or "code_lm_closure.py" in text
    )
    return hot_path and "--use-cuda-readout" in text


def active_code_lm_command_stage(rows: list[dict[str, Any]]) -> dict[str, str]:
    text = "\n".join(f"{row.get('name')} {row.get('command')}".lower() for row in rows)
    if "train-sts-parallel-decoder" in text:
        return {
            "stage": "sts_parallel_decoder_conditioning",
            "phase": "train_once_checkpoint_preconditioning",
        }
    if "generate-code-lm-closure-fanout" in text:
        return {
            "stage": "checkpoint_fanout_candidate_generation",
            "phase": "private_public_candidate_fanout",
        }
    if "train-code-lm-closure" in text:
        return {
            "stage": "cuda_code_lm_readout_training",
            "phase": "train_once_checkpoint",
        }
    if "code_lm_closure.py" in text and "--checkpoint-only" in text:
        return {
            "stage": "code_lm_checkpoint_orchestration",
            "phase": "train_once_checkpoint",
        }
    if "code_lm_train_once_fanout.py" in text:
        return {
            "stage": "train_once_fanout_supervisor",
            "phase": "orchestration",
        }
    if "code_lm_chunked_recovery.py" in text:
        return {
            "stage": "legacy_chunked_recovery_supervisor",
            "phase": "diagnostic_or_recovery",
        }
    if rows:
        return {"stage": "active_code_lm_worker", "phase": "unknown"}
    return {"stage": "", "phase": ""}


def heartbeat_matches_active_code_lm(heartbeat: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    if not heartbeat or not rows:
        return False
    text = "\n".join(str(row.get("command") or "").replace("\\", "/").lower() for row in rows)
    heartbeat_text = json.dumps(heartbeat, sort_keys=True).replace("\\", "/").lower()
    for marker in [
        "frontier_private_transfer_private_only_train_once_v1",
        "private_pressure_private_recovery_train_once_fanout_v1",
        "private_pressure_private_recovery_chunked",
        "train_once_fanout_smoke2",
    ]:
        if marker in heartbeat_text and marker in text:
            return True
    report_out = str(heartbeat.get("report_out") or "").replace("\\", "/").lower()
    if report_out and report_out in text:
        return True
    checkpoint = str(heartbeat.get("checkpoint") or "").replace("\\", "/").lower()
    if checkpoint and checkpoint in text:
        return True
    return False


def latest_train_once_phase_heartbeat(rows: list[dict[str, Any]], command_stage: dict[str, str]) -> dict[str, Any]:
    if not rows:
        return {}
    phase = str(command_stage.get("phase") or "")
    stage = str(command_stage.get("stage") or "")
    if not phase and not stage:
        return {}
    slug = infer_active_worker_slug(rows)
    if not slug:
        return {}
    paths = train_once_phase_paths(slug)
    heartbeat_path = phase_heartbeat_for_active_phase(paths, phase)
    if heartbeat_path is None and stage:
        heartbeat_path = phase_heartbeat_for_active_phase(paths, stage)
    summary = summarize_active_phase_heartbeat(heartbeat_path, root=ROOT)
    if not summary.get("exists"):
        return {}
    summary["stage"] = stage
    return summary


def train_once_phase_paths(slug: str) -> dict[str, Path]:
    return {
        "checkpoint_phase_heartbeat": REPORTS / f"code_lm_train_once_fanout_{slug}_checkpoint.phase_heartbeat.json",
        "fanout_phase_heartbeat": REPORTS / f"code_lm_train_once_fanout_{slug}_fanout.phase_heartbeat.json",
        "current_source_smoke_phase_heartbeat": REPORTS
        / f"code_lm_train_once_fanout_{slug}_current_source_smoke.phase_heartbeat.json",
    }


def latest_code_lm_heartbeat() -> dict[str, Any]:
    candidates = sorted(
        set(REPORTS.glob("code_lm_closure_rust*.heartbeat.json"))
        | set(REPORTS.glob("code_lm_fanout*.heartbeat.json")),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload = dict(payload)
        payload["path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        payload["age_seconds"] = round(time.time() - path.stat().st_mtime, 3)
        if str(payload.get("heartbeat_status") or "").lower() == "completed":
            payload["effective_run_status"] = "completed"
        else:
            payload["effective_run_status"] = payload.get("run_status") or payload.get("heartbeat_status")
        return payload
    return {}


def low_gpu_utilization_bottleneck(
    *,
    code_lm_cuda_readout: bool,
    gpu_utilization_percent: Any,
    stage: str,
    phase: str,
    progress_stage: str = "",
) -> tuple[bool, str]:
    if not code_lm_cuda_readout:
        return False, "no_active_cuda_code_lm"
    try:
        utilization = float(gpu_utilization_percent)
    except (TypeError, ValueError):
        return False, "gpu_utilization_unavailable"
    lowered = f"{stage} {phase} {progress_stage}".lower()
    if "sts_parallel_decoder" in lowered and utilization < 15.0:
        return True, f"sts_conditioning_active_with_gpu_utilization_{utilization:.1f}_percent"
    if "linear_readout_trained_aux_decoders_joining" in lowered and utilization < 15.0:
        return True, f"post_readout_aux_decoder_joining_with_gpu_utilization_{utilization:.1f}_percent"
    cpu_control_tokens = {
        "candidate_generation",
        "filter",
        "diagnostic",
        "checkpoint",
        "artifact",
        "write",
        "input_load",
        "token_models",
        "public_calibration",
        "private_eval",
    }
    if any(token in lowered for token in cpu_control_tokens):
        return False, f"explicit_cpu_control_phase:{stage or 'unknown'}:{phase or 'unknown'}"
    if utilization < 15.0:
        return True, f"cuda_code_lm_active_but_gpu_utilization_{utilization:.1f}_percent"
    return False, "gpu_utilization_healthy"


def recent_code_lm_timeout() -> bool:
    for path in [
        REPORTS / "code_lm_closure_rust_private_pressure_private_recovery.json",
        REPORTS / "code_transfer_bounded_recovery_chain.json",
    ]:
        try:
            if not path.exists() or time.time() - path.stat().st_mtime > 12 * 3600:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        text = json.dumps(payload).lower()
        if "timed_out_process_tree_killed" in text or "timed out after" in text:
            return True
    return False


def recent_recovery_smoke_timeout() -> bool:
    rust_path = REPORTS / "code_lm_closure_rust_private_pressure_private_recovery.json"
    chain_path = REPORTS / "code_transfer_bounded_recovery_chain.json"
    try:
        if not rust_path.exists() or time.time() - rust_path.stat().st_mtime > 12 * 3600:
            return False
        rust = json.loads(rust_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(rust.get("run_status") or "").lower() != "timed_out_process_tree_killed":
        return False
    summary = rust.get("summary") if isinstance(rust.get("summary"), dict) else {}
    previous = summary.get("previous_report_summary") if isinstance(summary.get("previous_report_summary"), dict) else {}
    step = previous.get("step_duration") if isinstance(previous.get("step_duration"), dict) else {}
    timeout_seconds = int(summary.get("timeout_seconds") or 0)
    private_eval_count = int(previous.get("private_eval_task_count") or 0)
    max_work_steps = int(step.get("max_work_steps") or 0)
    smoke_sized = timeout_seconds <= 1200 and private_eval_count <= 64 and max_work_steps <= 250_000
    if not smoke_sized:
        return False
    try:
        if chain_path.exists() and time.time() - chain_path.stat().st_mtime <= 12 * 3600:
            chain = json.loads(chain_path.read_text(encoding="utf-8"))
            if chain.get("failed_phase") == "bounded_private_pressure_closure":
                return True
    except Exception:
        pass
    return True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    budget = report["recommended_code_lm_budget"]
    return "\n".join(
        [
            "# Resource-Aware Execution Policy",
            "",
            f"- Status: **{report['trigger_state']}**",
            f"- Profile: `{summary['profile']}`",
            f"- Available memory: `{summary['available_memory_gb']} GB`",
            f"- D: free: `{summary['d_free_gb']} GB`",
            f"- Heavy process count: `{summary['heavy_process_count']}`",
            f"- Code LM CUDA readout active: `{summary['code_lm_cuda_readout_active']}`",
            f"- Code LM CPU-bound hot path active: `{summary['code_lm_cpu_bound_hot_path_active']}`",
            f"- Code LM low-GPU bottleneck: `{summary['code_lm_gpu_low_utilization_bottleneck']}`",
            f"- Code LM phase: `{summary['active_code_lm_stage'] or 'none'} / {summary['active_code_lm_phase'] or 'none'}`",
            f"- Start new Code LM closure: `{budget['start_new_code_closure']}`",
            f"- Recommended private count: `{budget['private_count']}`",
            f"- Recommended max work steps: `{budget['max_rust_work_steps']}`",
            "",
        ]
    )


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
