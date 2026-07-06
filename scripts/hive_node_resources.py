"""Resource probing and capability classification for the Theseus Hive node."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any

import compute_market
import hive_remote_control
import hive_storage
import hive_voice_following
import openai_compat_server
import update_manager
from hive_node_common import ROOT, command_available, get_path, read_json, to_float


def probe_resources(policy: dict[str, Any]) -> dict[str, Any]:
    nvidia = query_nvidia()
    mlx = query_mlx(policy, nvidia)
    return {
        "cpu": {
            "logical_cores": os.cpu_count() or 1,
            "architecture": platform.machine(),
        },
        "memory": memory_status(),
        "disk": disk_status(),
        "power": power_status(),
        "thermal": thermal_status(),
        "nvidia": nvidia,
        "mlx": mlx,
        "rust": command_available("cargo"),
        "python": sys.executable,
    }

def query_nvidia() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi_not_found"}
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip() or "nvidia_smi_failed"}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            gpus.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mib": to_float(parts[2]),
                    "memory_used_mib": to_float(parts[3]),
                    "memory_free_mib": to_float(parts[4]),
                    "utilization_gpu_percent": to_float(parts[5]),
                }
            )
    return {"available": bool(gpus), "gpus": gpus}

def query_mlx(policy: dict[str, Any], nvidia: dict[str, Any] | None = None) -> dict[str, Any]:
    module = str(get_path(policy, ["mac_support", "mlx_python_module"], "mlx.core"))
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    is_mac = platform.system() == "Darwin"
    machine = platform.machine().lower()
    is_apple_silicon = is_mac and machine in {"arm64", "aarch64"}
    has_nvidia = bool((nvidia or {}).get("available"))
    backend_ids: list[str] = []
    if spec and is_apple_silicon:
        backend_ids.append("mlx_apple")
    if spec and platform.system() == "Linux" and has_nvidia:
        backend_ids.append("mlx_cuda")
    return {
        "available": bool(backend_ids),
        "module_available": bool(spec),
        "module": module,
        "platform_is_macos": is_mac,
        "platform_machine": platform.machine(),
        "backend_ids": backend_ids,
        "recommended": bool(backend_ids),
        "install_notes": {
            "apple": "pip install mlx on native Apple Silicon Python with macOS >= 14",
            "cuda": "pip install mlx[cuda12] or mlx[cuda13] on supported Linux NVIDIA hosts",
        },
    }

def memory_status() -> dict[str, Any]:
    system = platform.system()
    if system == "Windows":
        return windows_memory_status()
    if system == "Darwin":
        try:
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], text=True, capture_output=True, timeout=3)
            total = int(result.stdout.strip()) if result.returncode == 0 else 0
        except (OSError, ValueError, subprocess.TimeoutExpired):
            total = 0
        return {"total_gib": round(total / 1024**3, 2) if total else None}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        rows = {}
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" in line:
                key, rest = line.split(":", 1)
                rows[key] = to_float(rest.strip().split()[0]) or 0.0
        total_kib = rows.get("MemTotal", 0.0)
        available_kib = rows.get("MemAvailable", 0.0)
        return {
            "total_gib": round(total_kib / 1024**2, 2) if total_kib else None,
            "available_gib": round(available_kib / 1024**2, 2) if available_kib else None,
        }
    return {"total_gib": None}

def windows_memory_status() -> dict[str, Any]:
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
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
        return {
            "total_gib": round(stat.ullTotalPhys / 1024**3, 2),
            "available_gib": round(stat.ullAvailPhys / 1024**3, 2),
            "load_percent": int(stat.dwMemoryLoad),
        }
    return {"total_gib": None}

def disk_status() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    return {
        "root": str(ROOT),
        "total_gib": round(usage.total / 1024**3, 2),
        "free_gib": round(usage.free / 1024**3, 2),
    }

def power_status() -> dict[str, Any]:
    system = platform.system()
    if system == "Darwin":
        return macos_power_status()
    if system == "Windows":
        return windows_power_status()
    if system == "Linux":
        return linux_power_status()
    return {"available": False, "reason": "unsupported_platform", "system": system}

def thermal_status() -> dict[str, Any]:
    system = platform.system()
    if system == "Darwin":
        return macos_thermal_status()
    if system == "Linux":
        return linux_thermal_status()
    if system == "Windows":
        return windows_thermal_status()
    return {"available": False, "reason": "unsupported_platform", "system": system}

def macos_thermal_status() -> dict[str, Any]:
    try:
        result = subprocess.run(["pmset", "-g", "therm"], text=True, capture_output=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "system": "Darwin", "error": str(exc)}
    if result.returncode != 0:
        return {"available": False, "system": "Darwin", "error": result.stderr.strip() or "pmset_therm_failed"}
    limits: dict[str, float] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed = to_float(value.strip().split()[0])
        if parsed is not None:
            limits[key.strip()] = parsed
    throttle = any(value < 100 for value in limits.values()) if limits else False
    return {
        "available": bool(limits),
        "system": "Darwin",
        "state": "throttled" if throttle else "nominal",
        "limits": limits,
        "raw_summary": " ".join(result.stdout.split())[:240],
    }

def linux_thermal_status() -> dict[str, Any]:
    root = Path("/sys/class/thermal")
    if not root.exists():
        return {"available": False, "system": "Linux", "reason": "thermal_sysfs_not_found"}
    temps = []
    for path in root.glob("thermal_zone*/temp"):
        try:
            raw = to_float(path.read_text(encoding="utf-8", errors="ignore").strip())
        except OSError:
            raw = None
        if raw is None:
            continue
        temp_c = raw / 1000.0 if raw > 500 else raw
        temps.append({"zone": path.parent.name, "temp_c": round(temp_c, 1)})
    max_temp = max((float(row["temp_c"]) for row in temps), default=None)
    return {
        "available": bool(temps),
        "system": "Linux",
        "state": "hot" if max_temp is not None and max_temp >= 85 else "nominal",
        "max_temp_c": max_temp,
        "zones": temps[:16],
    }

def windows_thermal_status() -> dict[str, Any]:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance -Namespace root/wmi MSAcpi_ThermalZoneTemperature | Select-Object -First 4 CurrentTemperature | ConvertTo-Json -Compress",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "system": "Windows", "error": str(exc)}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "system": "Windows", "reason": "thermal_query_unavailable"}
    try:
        value = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        value = []
    rows = value if isinstance(value, list) else [value]
    temps = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        kelvin_tenths = to_float(row.get("CurrentTemperature"))
        if kelvin_tenths is not None:
            temps.append(round(kelvin_tenths / 10.0 - 273.15, 1))
    max_temp = max(temps, default=None)
    return {
        "available": bool(temps),
        "system": "Windows",
        "state": "hot" if max_temp is not None and max_temp >= 85 else "nominal",
        "max_temp_c": max_temp,
        "temperatures_c": temps,
    }

def macos_power_status() -> dict[str, Any]:
    try:
        result = subprocess.run(["pmset", "-g", "batt"], text=True, capture_output=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc), "system": "Darwin"}
    raw = result.stdout.strip()
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip() or "pmset_failed", "system": "Darwin"}
    on_ac = "AC Power" in raw
    percent = None
    for token in raw.replace(";", " ").split():
        if token.endswith("%"):
            percent = to_float(token.strip("%"))
            break
    return {
        "available": True,
        "system": "Darwin",
        "on_ac_power": on_ac,
        "battery_percent": percent,
        "state": "ac_power" if on_ac else "battery_power",
        "raw_summary": " ".join(raw.split())[:200],
    }

def windows_power_status() -> dict[str, Any]:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Battery | Select-Object -First 1 EstimatedChargeRemaining,BatteryStatus | ConvertTo-Json -Compress",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc), "system": "Windows"}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "system": "Windows", "reason": "no_battery_or_query_failed"}
    try:
        value = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        value = {}
    status = int(value.get("BatteryStatus") or 0) if isinstance(value, dict) else 0
    percent = value.get("EstimatedChargeRemaining") if isinstance(value, dict) else None
    return {
        "available": True,
        "system": "Windows",
        "on_ac_power": status in {2, 6, 7, 8, 9, 11},
        "battery_percent": percent,
        "battery_status": status,
    }

def linux_power_status() -> dict[str, Any]:
    root = Path("/sys/class/power_supply")
    if not root.exists():
        return {"available": False, "system": "Linux", "reason": "power_supply_not_found"}
    batteries = [path for path in root.iterdir() if path.name.upper().startswith("BAT")]
    acs = [path for path in root.iterdir() if path.name.upper().startswith(("AC", "ADP", "MAINS"))]
    percent = None
    for battery in batteries:
        cap = battery / "capacity"
        if cap.exists():
            percent = to_float(cap.read_text(encoding="utf-8", errors="ignore").strip())
            break
    on_ac = None
    for ac in acs:
        online = ac / "online"
        if online.exists():
            on_ac = online.read_text(encoding="utf-8", errors="ignore").strip() == "1"
            break
    return {
        "available": bool(batteries or acs),
        "system": "Linux",
        "on_ac_power": on_ac,
        "battery_percent": percent,
        "battery_count": len(batteries),
        "ac_adapter_count": len(acs),
    }

def classify_capabilities(resources: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    caps = []
    nvidia = resources.get("nvidia") or {}
    if nvidia.get("available"):
        best = max(nvidia.get("gpus") or [{}], key=lambda gpu: float(gpu.get("memory_free_mib") or 0))
        caps.append({"id": "nvidia_cuda", "score": 0.9, "detail": best.get("name", "nvidia")})
    mlx = resources.get("mlx") or {}
    backend_ids = set(str(item) for item in mlx.get("backend_ids") or [])
    if "mlx_apple" in backend_ids:
        caps.append({"id": "mlx_apple", "score": 0.84, "detail": mlx.get("module")})
        caps.append({"id": "apple_mlx", "score": 0.82, "detail": "legacy alias for mlx_apple"})
    if "mlx_cuda" in backend_ids:
        caps.append({"id": "mlx_cuda", "score": 0.78, "detail": mlx.get("module")})
    cores = int(get_path(resources, ["cpu", "logical_cores"], 1))
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        caps.append({"id": "macos_hive_node", "score": 0.58, "detail": f"{platform.machine()} macOS node"})
        if machine in {"arm64", "aarch64"}:
            caps.append({"id": "macos_apple_silicon_node", "score": 0.66, "detail": "Apple Silicon Mac"})
        elif machine in {"x86_64", "amd64"}:
            caps.append(
                {
                    "id": "macos_intel_cpu_worker",
                    "score": min(0.68, 0.25 + cores / 48.0),
                    "detail": "Intel Mac CPU/storage/operator node",
                }
            )
    caps.append({"id": "cpu_worker", "score": min(0.75, 0.2 + cores / 32.0), "detail": f"{cores} logical cores"})
    if command_available("cargo").get("available"):
        caps.append({"id": "rust_build", "score": 0.65, "detail": "cargo available"})
    caps.append({"id": "checkpoint_chat_gateway", "score": 0.55, "detail": "grounded report/chat shim"})
    market_status = compute_market.status_report(write_report=True)
    caps.append(
        {
            "id": "compute_market_accounting",
            "score": 0.56 if market_status.get("enabled") else 0.2,
            "detail": f"{market_status.get('mode', 'accounting')} / {market_status.get('currency', {}).get('symbol', 'TWC')}",
        }
    )
    openai_status = openai_compat_server.status_report(write_report=True)
    if openai_status.get("live"):
        caps.append({"id": "openai_compatible_endpoint", "score": 0.6, "detail": str(openai_status.get("base_url") or "")})
    storage_status = hive_storage.status_report(policy=policy, write_report=True)
    if int(storage_status.get("share_count") or 0) > 0:
        caps.append({"id": "hive_storage_extension", "score": 0.62, "detail": f"{storage_status.get('share_count')} configured share(s)"})
    remote_control_status = hive_remote_control.status_report(policy=policy, write_report=True)
    ready_remote_control = int(remote_control_status.get("ready_provider_count") or 0)
    if ready_remote_control > 0:
        caps.append(
            {
                "id": "remote_control_operator",
                "score": 0.7,
                "detail": f"{ready_remote_control} provider(s), preferred {remote_control_status.get('preferred_provider_id') or 'auto'}",
            }
        )
    voice_status = hive_voice_following.status_report(policy=policy, write_report=False)
    if voice_status.get("enabled"):
        mic_ready = bool(get_path(voice_status, ["microphone", "ready"], False))
        speaker_ready = bool(get_path(voice_status, ["speaker", "ready"], False))
        if mic_ready or speaker_ready:
            caps.append(
                {
                    "id": "voice_following_node",
                    "score": 0.62 if mic_ready and speaker_ready else 0.48,
                    "detail": f"{get_path(voice_status, ['room', 'name'], 'unassigned')} listener={mic_ready} speaker={speaker_ready}",
                }
            )
        if mic_ready:
            caps.append({"id": "voice_presence_input", "score": 0.58, "detail": get_path(voice_status, ["room", "name"], "room")})
        if speaker_ready:
            caps.append({"id": "voice_response_output", "score": 0.58, "detail": get_path(voice_status, ["room", "name"], "room")})
    update_status = update_manager.status_report(write_report=True)
    caps.append(
        {
            "id": "candidate_update_client",
            "score": 0.58 if update_status.get("update_available") else 0.45,
            "detail": "update available" if update_status.get("update_available") else "watching accepted candidates",
        }
    )
    return caps

def resource_slots(resources: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    slot_policy = policy.get("resource_slots") if isinstance(policy.get("resource_slots"), dict) else {}
    if slot_policy.get("enabled", True) is False:
        return []
    slots: list[dict[str, Any]] = []
    cores = int(get_path(resources, ["cpu", "logical_cores"], 1))
    cpu_capacity = max(1, min(int(slot_policy.get("max_cpu_slots", 4)), max(1, cores // 2)))
    slots.append(
        {
            "slot_id": "cpu:general",
            "slot_type": "cpu",
            "capacity": cpu_capacity,
            "running": 0,
            "available": cpu_capacity > 0,
            "capabilities": ["cpu_worker"],
            "task_kinds": get_path(slot_policy, ["task_kinds_by_slot", "cpu"], []),
        }
    )
    nvidia = resources.get("nvidia") if isinstance(resources.get("nvidia"), dict) else {}
    if nvidia.get("available"):
        for idx, gpu in enumerate(nvidia.get("gpus") or []):
            slots.append(
                {
                    "slot_id": f"cuda:{idx}",
                    "slot_type": "cuda",
                    "capacity": int(slot_policy.get("max_cuda_slots_per_gpu", 1)),
                    "running": 0,
                    "available": int(slot_policy.get("max_cuda_slots_per_gpu", 1)) > 0,
                    "capabilities": ["nvidia_cuda", "rust_cuda"],
                    "task_kinds": get_path(slot_policy, ["task_kinds_by_slot", "cuda"], []),
                    "memory_free_mib": gpu.get("memory_free_mib"),
                    "memory_total_mib": gpu.get("memory_total_mib"),
                    "detail": gpu.get("name"),
                }
            )
    mlx = resources.get("mlx") if isinstance(resources.get("mlx"), dict) else {}
    backend_ids = set(str(item) for item in mlx.get("backend_ids") or [])
    if "mlx_apple" in backend_ids:
        slots.append(
            {
                "slot_id": "mlx:apple:0",
                "slot_type": "mlx_apple",
                "capacity": int(slot_policy.get("max_mlx_slots", 1)),
                "running": 0,
                "available": int(slot_policy.get("max_mlx_slots", 1)) > 0,
                "capabilities": ["mlx_apple", "apple_mlx"],
                "task_kinds": get_path(slot_policy, ["task_kinds_by_slot", "mlx"], []),
            }
        )
    if "mlx_cuda" in backend_ids:
        slots.append(
            {
                "slot_id": "mlx:cuda:0",
                "slot_type": "mlx_cuda",
                "capacity": int(slot_policy.get("max_mlx_cuda_slots", 1)),
                "running": 0,
                "available": int(slot_policy.get("max_mlx_cuda_slots", 1)) > 0,
                "capabilities": ["mlx_cuda", "nvidia_cuda"],
                "task_kinds": get_path(slot_policy, ["task_kinds_by_slot", "mlx"], []),
            }
        )
    return slots

def worker_thread_count(status: dict[str, Any], policy: dict[str, Any]) -> int:
    capacity = sum(max(1, int(slot.get("capacity") or 1)) for slot in status.get("slots") or [])
    max_threads = int(get_path(policy, ["resource_slots", "max_worker_threads"], 8))
    return max(1, min(max_threads, capacity or 1))
