"""Run bounded CUDA VRAM stress probes for configured training profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="configs/training_profiles_rtx2060super.json")
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--out", default="reports/profile_vram_stress_report.json")
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    config = read_json(Path(args.profiles))
    profiles = config.get("profiles") or {}
    selected = args.profile or ["inner_loop", "candidate"]
    rows = []
    ok = True
    for name in selected:
        profile = profiles.get(name)
        if not profile:
            rows.append({"profile": name, "passed": False, "error": "missing_profile"})
            ok = False
            continue
        row = run_profile_stress(
            name,
            profile,
            poll_seconds=args.poll_seconds,
            timeout=args.timeout_seconds,
        )
        rows.append(row)
        ok = ok and row.get("passed") is True

    report = {
        "policy": "local_only_no_external_inference",
        "methodology": "rtx2060super_profile_vram_stress",
        "ok": ok,
        "profiles": selected,
        "stress": rows,
        "external_inference_calls": 0,
    }
    write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def run_profile_stress(
    name: str,
    profile: dict[str, Any],
    *,
    poll_seconds: float,
    timeout: int,
) -> dict[str, Any]:
    if sys.platform == "darwin":
        return run_macos_mlx_stress_proxy(name, profile)

    rollout = profile.get("puffer_ocean_rollout_cuda") or {}
    report_path = Path(f"reports/vram_stress_{name}_rollout_cuda.json")
    command = [
        str(ROOT / "target" / "release" / "symliquid-cli.exe"),
        "train-rollout-cuda",
        "--cases-per-task",
        "2",
        "--epochs",
        "1",
        "--state-epochs",
        str(min(1, int(rollout.get("state_epochs", 0) or 0))),
        "--state-lr",
        str(rollout.get("state_lr", 0.0)),
        "--probe-cases-per-task",
        "2",
        "--samples-per-launch",
        str(rollout.get("samples_per_launch", 128)),
        "--rollout-batch",
        str(rollout.get("rollout_batch", 128)),
        "--obs-dim",
        str(rollout.get("obs_dim", 32)),
        "--hidden-dim",
        str(rollout.get("hidden_dim", 64)),
        "--reservoir-dim",
        str(rollout.get("reservoir_dim", 96)),
        "--hv-dim",
        str(rollout.get("hv_dim", 1024)),
        "--seq-len",
        str(rollout.get("seq_len", 32)),
        "--lr",
        str(rollout.get("lr", 0.03)),
        "--out",
        str(report_path),
    ]
    samples: list[dict[str, Any]] = []
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stop = threading.Event()
    poller = threading.Thread(
        target=poll_nvidia_smi,
        args=(samples, stop, poll_seconds),
        daemon=True,
    )
    started = time.perf_counter()
    poller.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    finally:
        stop.set()
        poller.join(timeout=2.0)
    runtime_ms = int((time.perf_counter() - started) * 1000)

    max_used = max((int(sample.get("memory_used_mib", 0)) for sample in samples), default=0)
    max_total = max((int(sample.get("memory_total_mib", 0)) for sample in samples), default=0)
    limit = int(profile.get("max_vram_mib", 0) or 0)
    payload = read_json(report_path)
    passed = (
        process.returncode == 0
        and bool(payload)
        and payload.get("cuda_fallback") is False
        and (limit <= 0 or max_used <= limit)
    )
    return {
        "profile": name,
        "passed": passed,
        "command": command,
        "returncode": process.returncode,
        "runtime_ms": runtime_ms,
        "stdout_tail": (stdout or "")[-4000:],
        "stderr_tail": (stderr or "")[-4000:],
        "report": str(report_path),
        "max_vram_used_mib": max_used,
        "max_vram_total_mib": max_total,
        "profile_max_vram_mib": limit,
        "sample_count": len(samples),
        "cuda_fallback": payload.get("cuda_fallback"),
        "train_examples_per_second": payload.get("train_examples_per_second"),
        "train_runtime_ms": payload.get("train_runtime_ms"),
    }


def run_macos_mlx_stress_proxy(name: str, profile: dict[str, Any]) -> dict[str, Any]:
    report_path = Path("reports/macos_training_preflight.json")
    payload = read_json(report_path)
    worker = get_path(payload, ["execution", "worker_report", "payload"], {})
    metrics = worker.get("metrics") if isinstance(worker, dict) else {}
    backend = str(worker.get("backend") or "") if isinstance(worker, dict) else ""
    external = int(get_path(payload, ["execution", "external_inference_calls"], 0) or 0)
    passed = bool(
        payload.get("state") == "GREEN"
        and payload.get("long_training_allowed") is True
        and get_path(payload, ["execution", "ok"], False) is True
        and backend in {"mlx_apple", "apple_mlx"}
        and isinstance(metrics, dict)
        and metrics
        and external == 0
    )
    return {
        "profile": name,
        "passed": passed,
        "methodology": "macos_mlx_training_preflight_proxy",
        "command": ["python3", "scripts/macos_mlx_work_proof.py"],
        "returncode": 0 if passed else 1,
        "runtime_ms": worker.get("runtime_ms") if isinstance(worker, dict) else None,
        "stdout_tail": "",
        "stderr_tail": "",
        "report": str(report_path),
        "backend": backend,
        "max_vram_used_mib": 0,
        "max_vram_total_mib": 0,
        "profile_max_vram_mib": int(profile.get("max_vram_mib", 0) or 0),
        "sample_count": 0,
        "cuda_fallback": False,
        "mlx_fallback": False,
        "external_inference_calls": external,
        "train_examples_per_second": metrics.get("examples_per_second") if isinstance(metrics, dict) else None,
        "train_runtime_ms": worker.get("runtime_ms") if isinstance(worker, dict) else None,
        "metrics": metrics if isinstance(metrics, dict) else {},
    }


def poll_nvidia_smi(samples: list[dict[str, Any]], stop: threading.Event, interval: float) -> None:
    while not stop.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            if len(parts) >= 3:
                samples.append(
                    {
                        "t": time.time(),
                        "memory_used_mib": int(float(parts[0])),
                        "memory_total_mib": int(float(parts[1])),
                        "gpu_utilization_percent": int(float(parts[2])),
                    }
                )
        time.sleep(interval)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
