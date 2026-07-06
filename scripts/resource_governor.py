"""Resource-aware execution governor for SparkStream.

The governor keeps the autonomous loop biased toward efficient local work:
Rust/CUDA hot loops, small profiles before large profiles, one training job at
a time, and teacher escalation only when local evidence says we are stuck.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "autonomy_policy.json"
DEFAULT_CUDA_PROFILES = ROOT / "configs" / "training_profiles_rtx2060super.json"
DEFAULT_MACOS_PROFILES = ROOT / "configs" / "training_profiles_macos_local.json"
DEFAULT_OUT = ROOT / "reports" / "resource_governor.json"
DEFAULT_LEDGER = ROOT / "reports" / "resource_governor_ledger.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    default_profiles = default_profiles_path()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--profiles", default=str(default_profiles.relative_to(ROOT)))
    parser.add_argument("--profile", default="inner_loop")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER.relative_to(ROOT)))
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    profiles = read_json(ROOT / args.profiles)
    report = build_report(policy, profiles, args.profile)
    write_json(ROOT / args.out, report)
    append_jsonl(ROOT / args.ledger, compact_ledger(report))
    print(json.dumps(report, indent=2))
    return 0


def build_report(policy: dict[str, Any], profiles: dict[str, Any], requested_profile: str) -> dict[str, Any]:
    started = time.perf_counter()
    profile = (profiles.get("profiles") or {}).get(requested_profile, {})
    hardware = platform_hardware_profile(profiles.get("hardware") or {})
    governor = policy.get("resource_governor") or {}
    gpu = query_gpu()
    disk = disk_status()
    spillover = resource_pantry_storage_status(policy)
    runtime_paths = theseus_runtime.runtime_report(create=True, write_report=True)
    active = active_training_jobs()
    profile_budget = profile_budget_for(profile)
    min_free_after = int(governor.get("min_free_vram_mib_after_allocation", 768))
    max_jobs = int(governor.get("max_concurrent_training_jobs", 1))
    gpu_busy = gpu.get("utilization_gpu_percent")
    free_vram = gpu.get("memory_free_mib")
    vram_ok = (
        free_vram is None
        or profile_budget.get("max_vram_mib", 0) <= 0
        or free_vram - profile_budget["max_vram_mib"] >= min_free_after
    )
    busy_ok = (
        gpu_busy is None
        or gpu_busy < int(governor.get("throttle_when_gpu_busy_percent", 92))
    )
    job_ok = active["training_job_count"] < max_jobs
    disk_warning_gib = float(governor.get("disk_free_gib_warning", 20))
    disk_minimum_gib = float(governor.get("disk_free_gib_minimum", 10))
    spillover_selected = get_path(spillover, ["selected"], {})
    spillover_tier = str(spillover_selected.get("tier") or "")
    spillover_ok = (
        bool(governor.get("allow_storage_spillover_for_disk_warning", True))
        and bool(spillover_selected.get("available"))
        and spillover_tier in {"preferred", "fallback"}
    )
    runtime_storage_ok = runtime_generated_storage_ok(runtime_paths, disk_minimum_gib)
    disk_ok = (
        disk["free_gib"] >= disk_warning_gib
        or (disk["free_gib"] >= disk_minimum_gib and spillover_ok)
        or runtime_storage_ok
    )
    execution_owner = platform_execution_owner(hardware, governor, gpu)
    can_run = bool(vram_ok and busy_ok and job_ok and disk_ok)
    reasons = []
    if not vram_ok:
        reasons.append("insufficient_free_vram_for_profile")
    if not busy_ok:
        reasons.append("gpu_busy")
    if not job_ok:
        reasons.append("training_job_already_running")
    warnings = []
    if disk["free_gib"] < disk_warning_gib and disk_ok and spillover_ok:
        warnings.append(
            f"root_disk_below_warning_using_spillover root_free_gib={disk['free_gib']} "
            f"warning_gib={disk_warning_gib} spillover_free_gib={spillover_selected.get('free_gib')}"
        )
    if disk["free_gib"] < disk_minimum_gib and runtime_storage_ok:
        warnings.append(
            f"root_disk_below_minimum_using_runtime_paths root_free_gib={disk['free_gib']} "
            f"runtime_root={get_path(runtime_paths, ['paths', 'runtime_root', 'path'], '')}"
        )
    if not disk_ok:
        reasons.append("low_disk_space")
    efficiency_score = score_efficiency(profile_budget, gpu, active, disk, spillover_ok, can_run)
    trigger_state = "GREEN" if can_run else "YELLOW"
    return {
        "policy": "sparkstream_resource_governor_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "requested_profile": requested_profile,
        "summary": {
            "can_run_requested_profile": can_run,
            "recommended_profile": recommend_profile(profiles, gpu, governor, active),
            "execution_owner": execution_owner,
            "python_role": hardware.get("python_role", governor.get("python_role", "orchestration_and_reporting")),
            "throttle_reason_count": len(reasons),
            "throttle_reasons": reasons,
            "warning_count": len(warnings),
            "warnings": warnings,
            "efficiency_score": efficiency_score,
            "disk_free_gib": disk["free_gib"],
            "root_disk_ok": disk_ok,
            "gpu_available": bool(gpu.get("available")),
            "accelerator": gpu.get("accelerator") or gpu.get("name"),
            "mlx_usable": bool(gpu.get("mlx_usable")),
            "metal_usable": bool(gpu.get("metal_usable")),
            "active_training_jobs": active["training_job_count"],
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "public_training_rows_written": 0,
        },
        "hardware_profile": hardware,
        "resource_policy": governor,
        "current_resources": {
            "gpu": gpu,
            "disk": disk,
            "spillover": spillover,
            "runtime_paths": runtime_paths,
            "active_jobs": active,
        },
        "profile_budget": profile_budget,
        "decision": {
            "can_run_requested_profile": can_run,
            "throttle_reasons": reasons,
            "warnings": warnings,
            "recommended_profile": recommend_profile(profiles, gpu, governor, active),
            "execution_owner": execution_owner,
            "python_role": hardware.get("python_role", governor.get("python_role", "orchestration_and_reporting")),
        },
        "resource_envelope": {
            "max_vram_mib": profile_budget.get("max_vram_mib"),
            "reserve_vram_mib": min_free_after,
            "max_concurrent_training_jobs": max_jobs,
            "network": "off_by_default",
            "external_inference": "teacher_only_when_allowed_and_budgeted",
            "storage_spillover": get_path(spillover, ["selected", "path"], None),
            "runtime_root": get_path(runtime_paths, ["paths", "runtime_root", "path"], None),
        },
        "efficiency": {
            "score": efficiency_score,
            "objectives": governor.get("efficiency_objectives", []),
            "hot_loop_guidance": hot_loop_guidance(execution_owner),
            "cache_guidance": "reuse parsed/tokenized data and generated holdouts before regenerating",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def query_gpu() -> dict[str, Any]:
    if platform.system() == "Darwin":
        return query_apple_accelerator()
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "error": result.stderr.strip() or "nvidia_smi_failed"}
    line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 8:
        return {"available": False, "raw": line, "error": "unexpected_nvidia_smi_output"}
    total = to_float(parts[2])
    used = to_float(parts[3])
    free = to_float(parts[4])
    return {
        "available": True,
        "name": parts[0],
        "driver_version": parts[1],
        "memory_total_mib": total,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "utilization_gpu_percent": to_float(parts[5]),
        "temperature_c": to_float(parts[6]),
        "power_draw_w": to_float(parts[7]),
    }


def query_apple_accelerator() -> dict[str, Any]:
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64"}:
        return {
            "available": False,
            "name": "Intel Mac CPU/storage/operator node",
            "machine": machine,
            "accelerator": "cpu",
            "mlx_usable": False,
            "metal_usable": False,
            "error": "apple_mlx_requires_apple_silicon",
        }
    probe = latest_mlx_probe()
    summary = probe.get("summary") if isinstance(probe.get("summary"), dict) else {}
    usable = probe.get("trigger_state") == "GREEN" and summary.get("route_action") == "route_mlx_to_usable_python"
    return {
        "available": bool(usable),
        "name": "Apple Silicon MLX/Metal",
        "machine": machine,
        "accelerator": "mlx_apple" if usable else "apple_silicon_probe_required",
        "mlx_usable": bool(usable),
        "metal_usable": bool(usable),
        "default_device": first_probe_device(probe),
        "probe_source": str(probe.get("_source_path") or ""),
        "recommended_python": summary.get("recommended_python"),
        "error": None if usable else "mlx_unsandboxed_probe_missing_or_not_green",
    }


def latest_mlx_probe() -> dict[str, Any]:
    candidates = sorted((ROOT / "reports").glob("macos_mlx_environment_diagnosis*.json"))
    best: dict[str, Any] = {}
    best_mtime = -1.0
    for path in candidates:
        if not path.exists():
            continue
        row = read_json(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        if isinstance(row, dict) and mtime > best_mtime:
            row["_source_path"] = rel(path)
            best = row
            best_mtime = mtime
    return best


def first_probe_device(report: dict[str, Any]) -> str | None:
    probes = report.get("python_probes")
    if not isinstance(probes, list):
        return None
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        core = probe.get("core_probe") if isinstance(probe.get("core_probe"), dict) else {}
        stdout = str(core.get("stdout_tail") or "")
        if "Device(gpu" in stdout:
            return "Device(gpu, 0)"
    return None


def disk_status() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    return {
        "root": str(ROOT),
        "total_gib": round(usage.total / 1024**3, 2),
        "used_gib": round(usage.used / 1024**3, 2),
        "free_gib": round(usage.free / 1024**3, 2),
    }


def resource_pantry_storage_status(policy: dict[str, Any]) -> dict[str, Any]:
    pantry_policy_path = ROOT / str(get_path(policy, ["resource_pantry", "policy"], "configs/resource_pantry.json"))
    pantry_policy = read_json(pantry_policy_path)
    storage = pantry_policy.get("storage") or {}
    preferred = expand_path(str(storage.get("preferred_clone_root") or ""))
    fallback = expand_path(str(storage.get("fallback_clone_root") or "data/external_benchmark_candidates/git_clones"))
    min_preferred = float(storage.get("min_free_gib_preferred", 100))
    min_fallback = float(storage.get("min_free_gib_fallback", 25))
    preferred_status = disk_status_for_path(preferred)
    fallback_status = disk_status_for_path(fallback)
    if preferred_status.get("available") and preferred_status.get("free_gib", 0) >= min_preferred:
        selected = preferred_status
        selected["tier"] = "preferred"
    elif fallback_status.get("available") and fallback_status.get("free_gib", 0) >= min_fallback:
        selected = fallback_status
        selected["tier"] = "fallback"
    else:
        selected = fallback_status
        selected["tier"] = "fallback_low_space"
    return {
        "policy": str(pantry_policy_path.relative_to(ROOT)) if pantry_policy_path.is_relative_to(ROOT) else str(pantry_policy_path),
        "selected": selected,
        "preferred": preferred_status,
        "fallback": fallback_status,
        "min_free_gib_preferred": min_preferred,
        "min_free_gib_fallback": min_fallback,
    }


def disk_status_for_path(path: Path) -> dict[str, Any]:
    try:
        anchor = path.anchor or str(ROOT.anchor)
        usage = shutil.disk_usage(anchor)
        return {
            "available": True,
            "path": str(path),
            "root": anchor,
            "total_gib": round(usage.total / 1024**3, 2),
            "used_gib": round(usage.used / 1024**3, 2),
            "free_gib": round(usage.free / 1024**3, 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(path), "error": str(exc)}


def runtime_generated_storage_ok(runtime_paths: dict[str, Any], min_free_gib: float) -> bool:
    paths = runtime_paths.get("paths") if isinstance(runtime_paths.get("paths"), dict) else {}
    required = ["reports_dir", "checkpoints_dir", "cargo_target_dir", "cache_dir"]
    for key in required:
        row = paths.get(key) if isinstance(paths.get(key), dict) else {}
        free = row.get("disk_free_gib")
        try:
            if float(free) < min_free_gib:
                return False
        except (TypeError, ValueError):
            return False
    return True


def expand_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        return path
    return ROOT / path


def active_training_jobs() -> dict[str, Any]:
    jobs_dir = ROOT / "reports" / "sparkstream_jobs"
    running = 0
    maintenance = 0
    known = 0
    if jobs_dir.exists():
        for path in jobs_dir.glob("*.out.log"):
            known += 1
    status = read_json(ROOT / "reports" / "sparkstream_status.json")
    phase = str(status.get("phase") or "")
    step_status = str(get_path(status, ["profile_step", "status"], ""))
    if phase == "running_profile":
        running += 1
    elif phase == "running_profile_step" and step_status in {"started", "running"}:
        running += 1
    elif phase == "refreshing_ratchet":
        maintenance += 1
    return {
        "training_job_count": running,
        "maintenance_job_count": maintenance,
        "known_job_logs": known,
        "sparkstream_phase": phase,
        "sparkstream_updated_utc": status.get("updated_utc"),
    }


def platform_execution_owner(hardware: dict[str, Any], governor: dict[str, Any], gpu: dict[str, Any]) -> str:
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mlx_apple" if gpu.get("mlx_usable") else "macos_cpu_until_mlx_probe_green"
    if platform.system() == "Darwin":
        return "rust_cpu"
    preferred = str(hardware.get("preferred_hot_loop_owner") or governor.get("prefer_hot_loop_owner") or "rust_cuda")
    if preferred == "auto_detect":
        return "rust_cuda"
    return preferred


def hot_loop_guidance(owner: str) -> str:
    if owner == "mlx_apple":
        return "prefer MLX for Apple Silicon rollout, scoring, optimizer state, and repeated eval kernels"
    if owner == "macos_cpu_until_mlx_probe_green":
        return "run the unsandboxed MLX diagnosis before routing Apple Silicon hot loops; use CPU only until it is GREEN"
    if owner == "rust_cpu":
        return "prefer Rust CPU paths for Intel Mac workers; advertise CPU/storage/operator capability, not MLX"
    return "prefer Rust/CUDA for rollout, scoring, optimizer state, and repeated eval kernels"


def profile_budget_for(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_runtime_minutes": profile.get("expected_runtime_minutes", 0),
        "max_vram_mib": profile.get("max_vram_mib", 0),
        "purpose": profile.get("purpose", ""),
    }


def recommend_profile(profiles: dict[str, Any], gpu: dict[str, Any], governor: dict[str, Any], active: dict[str, Any]) -> str:
    if int(active.get("training_job_count") or 0) > 0:
        return "smoke"
    free_vram = gpu.get("memory_free_mib")
    reserve = int(governor.get("min_free_vram_mib_after_allocation", 768))
    candidates = []
    rank = {"smoke": 0, "inner_loop": 1, "candidate": 2, "overnight": 3}
    for name, profile in (profiles.get("profiles") or {}).items():
        if name not in rank:
            continue
        budget = profile_budget_for(profile)
        max_vram = float(budget.get("max_vram_mib") or 0)
        runtime = float(budget.get("expected_runtime_minutes") or 0)
        fits = free_vram is None or max_vram <= 0 or free_vram - max_vram >= reserve
        if fits:
            candidates.append((rank.get(name, 0), runtime, max_vram, name))
    if not candidates:
        return "smoke"
    return sorted(candidates, reverse=True)[0][3]


def score_efficiency(
    profile_budget: dict[str, Any],
    gpu: dict[str, Any],
    active: dict[str, Any],
    disk: dict[str, Any],
    spillover_ok: bool,
    can_run: bool,
) -> float:
    score = 1.0 if can_run else 0.5
    if gpu.get("available"):
        util = float(gpu.get("utilization_gpu_percent") or 0)
        free = float(gpu.get("memory_free_mib") or 0)
        total = float(gpu.get("memory_total_mib") or 1)
        score += min(0.1, free / max(1.0, total) * 0.1)
        if util > 90:
            score -= 0.2
    if profile_budget.get("expected_runtime_minutes", 0) > 120:
        score -= 0.05
    if active.get("training_job_count", 0) > 0:
        score -= 0.25
    if disk.get("free_gib", 0) < 20 and not spillover_ok:
        score -= 0.2
    elif disk.get("free_gib", 0) < 20:
        score -= 0.05
    return round(max(0.0, min(1.0, score)), 4)


def compact_ledger(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_utc": report.get("created_utc"),
        "requested_profile": report.get("requested_profile"),
        "can_run": get_path(report, ["decision", "can_run_requested_profile"], False),
        "recommended_profile": get_path(report, ["decision", "recommended_profile"], ""),
        "efficiency_score": get_path(report, ["efficiency", "score"], 0.0),
        "gpu": get_path(report, ["current_resources", "gpu", "name"], ""),
        "execution_owner": get_path(report, ["decision", "execution_owner"], ""),
        "free_vram_mib": get_path(report, ["current_resources", "gpu", "memory_free_mib"], None),
        "throttle_reasons": get_path(report, ["decision", "throttle_reasons"], []),
    }


def default_profiles_path() -> Path:
    if platform.system() == "Darwin" and DEFAULT_MACOS_PROFILES.exists():
        return DEFAULT_MACOS_PROFILES
    return DEFAULT_CUDA_PROFILES


def platform_hardware_profile(configured: dict[str, Any]) -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return {
            **configured,
            "name": "Apple Silicon Mac",
            "compute_capability": "apple_metal_mlx",
            "vram_total_mib": 0,
            "max_target_vram_mib": 0,
            "preferred_hot_loop_owner": "mlx_apple_when_probe_green",
            "python_role": configured.get("python_role", "orchestration_and_reporting"),
            "profile_source": "platform_detected",
        }
    if system == "Darwin":
        return {
            **configured,
            "name": "Intel Mac",
            "compute_capability": "x86_64_cpu",
            "vram_total_mib": 0,
            "max_target_vram_mib": 0,
            "preferred_hot_loop_owner": "rust_cpu",
            "python_role": configured.get("python_role", "orchestration_and_reporting"),
            "profile_source": "platform_detected",
        }
    return configured


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
