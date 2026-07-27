#!/usr/bin/env python3
"""Qualify ANE SwiGLU recomputation against an independent Metal backward window."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_ane_activation_recomputation_qualification_v1"
DEFAULT_SOURCE = ROOT / "native/ane_metal/ane_swiglu_activation_recompute.m"
DEFAULT_NATIVE_BINARY = Path("/private/tmp/theseus_ane_swiglu_recompute")
MLX_WORKER = ROOT / "scripts/mlx_attention_backward_window.py"


class QualificationFault(ValueError):
    """Raised when a worker or evidence packet is malformed."""


def final_json(output: bytes) -> dict[str, Any]:
    for raw in reversed(output.decode("utf-8", errors="replace").splitlines()):
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise QualificationFault("worker_final_json_missing")


def build_native(source: Path, binary: Path) -> None:
    command = [
        "xcrun",
        "clang",
        "-fobjc-arc",
        "-O3",
        "-DACCELERATE_NEW_LAPACK",
        "-DACCELERATE_LAPACK_ILP64",
        "-framework",
        "Foundation",
        "-framework",
        "IOSurface",
        "-framework",
        "Metal",
        "-framework",
        "Accelerate",
        str(source),
        "-o",
        str(binary),
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    if completed.returncode:
        raise QualificationFault(
            "native_build_failed:"
            + completed.stdout.decode("utf-8", errors="replace")[-2000:]
        )


def run_worker(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise QualificationFault(
            f"worker_failed_{completed.returncode}:"
            + completed.stdout.decode("utf-8", errors="replace")[-2000:]
        )
    return final_json(completed.stdout)


def barrier_work(payload: dict[str, Any]) -> float:
    value = (payload.get("runtime") or {}).get("barrier_work_milliseconds")
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise QualificationFault("barrier_work_milliseconds_missing")
    return float(value)


def barrier_round(
    native_command: Sequence[str],
    mlx_command: Sequence[str],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="theseus_ane_recompute_", dir="/private/tmp"
    ) as temporary:
        root = Path(temporary)
        go = root / "go"
        native_ready = root / "native.ready"
        mlx_ready = root / "mlx.ready"
        native = subprocess.Popen(
            [
                *native_command,
                "--ready-file",
                str(native_ready),
                "--go-file",
                str(go),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        mlx = subprocess.Popen(
            [
                *mlx_command,
                "--ready-file",
                str(mlx_ready),
                "--go-file",
                str(go),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 30.0
        while not (native_ready.exists() and mlx_ready.exists()):
            if native.poll() is not None or mlx.poll() is not None:
                break
            if time.monotonic() >= deadline:
                native.kill()
                mlx.kill()
                raise QualificationFault("concurrent_ready_timeout")
            time.sleep(0.001)
        if not (native_ready.exists() and mlx_ready.exists()):
            native_output = native.communicate()[0] or b""
            mlx_output = mlx.communicate()[0] or b""
            raise QualificationFault(
                "concurrent_worker_exited_before_ready:"
                + (native_output + mlx_output).decode(
                    "utf-8", errors="replace"
                )[-2000:]
            )
        started = time.perf_counter()
        go.write_text("go\n", encoding="utf-8")
        native_output = native.communicate()[0] or b""
        mlx_output = mlx.communicate()[0] or b""
        joined_wall_milliseconds = (time.perf_counter() - started) * 1000.0
        if native.returncode or mlx.returncode:
            raise QualificationFault(
                f"concurrent_worker_failed:{native.returncode}:{mlx.returncode}:"
                + (native_output + mlx_output).decode(
                    "utf-8", errors="replace"
                )[-2000:]
            )
        native_payload = final_json(native_output)
        mlx_payload = final_json(mlx_output)
        return {
            "native": native_payload,
            "mlx": mlx_payload,
            "joined_wall_milliseconds_including_teardown": joined_wall_milliseconds,
            "critical_path_milliseconds": max(
                barrier_work(native_payload), barrier_work(mlx_payload)
            ),
        }


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def adjudicate(
    standalone_native: list[dict[str, Any]],
    standalone_mlx: list[dict[str, Any]],
    concurrent: list[dict[str, Any]],
    *,
    repetitions: int,
) -> dict[str, Any]:
    native_green = all(
        row.get("trigger_state") == "GREEN_RECOMPUTE_MECHANICS"
        and int((row.get("parity") or {}).get("mismatch_count", -1)) == 0
        and not bool(
            (row.get("custody") or {}).get(
                "intermediate_python_or_numpy_round_trip", True
            )
        )
        for row in standalone_native
        + [round_row["native"] for round_row in concurrent]
    )
    mlx_green = all(
        str(row.get("trigger_state") or "").startswith("GREEN")
        and bool((row.get("mechanics") or {}).get("all_gradients_finite"))
        for row in standalone_mlx
        + [round_row["mlx"] for round_row in concurrent]
    )
    control_per_iteration = [
        barrier_work(row) / repetitions for row in standalone_mlx
    ]
    candidate_per_iteration = [
        float(row["critical_path_milliseconds"]) / repetitions
        for row in concurrent
    ]
    concurrent_native_per_iteration = [
        barrier_work(row["native"]) / repetitions for row in concurrent
    ]
    concurrent_mlx_per_iteration = [
        barrier_work(row["mlx"]) / repetitions for row in concurrent
    ]
    recompute_hidden = all(
        barrier_work(row["native"]) <= barrier_work(row["mlx"])
        for row in concurrent
    )
    mean_speedup = (
        statistics.fmean(control_per_iteration)
        / statistics.fmean(candidate_per_iteration)
    )
    conservative_speedup = (
        min(control_per_iteration) / max(candidate_per_iteration)
    )
    wall_selected = (
        native_green
        and mlx_green
        and recompute_hidden
        and conservative_speedup > 1.0
    )
    memory = standalone_native[0]["memory"]
    theoretical_release = float(
        memory["maximum_twelve_layer_discarded_mib"]
    )
    measured_batch_six_deficit = 236.656
    memory_ceiling_covers_gap = (
        theoretical_release >= measured_batch_six_deficit
    )
    return {
        "native_mechanics_green": native_green,
        "mlx_backward_window_green": mlx_green,
        "ane_recompute_hidden_inside_mlx_window": recompute_hidden,
        "control_milliseconds_per_iteration": summary(control_per_iteration),
        "candidate_critical_path_milliseconds_per_iteration": summary(
            candidate_per_iteration
        ),
        "concurrent_native_milliseconds_per_iteration": summary(
            concurrent_native_per_iteration
        ),
        "concurrent_mlx_milliseconds_per_iteration": summary(
            concurrent_mlx_per_iteration
        ),
        "mean_speedup_control_over_candidate": mean_speedup,
        "conservative_speedup_control_over_candidate": conservative_speedup,
        "joined_wall_selected": wall_selected,
        "memory": {
            **memory,
            "historical_microbatch_six_live_reserve_deficit_mib": (
                measured_batch_six_deficit
            ),
            "theoretical_release_covers_historical_gap": (
                memory_ceiling_covers_gap
            ),
            "actual_mlx_allocator_release_measured": False,
            "larger_sustained_batch_qualified": False,
        },
        "production_eligible": False,
        "remaining_integration_gates": [
            "MLX autograd cannot consume the private ANE IOSurface output.",
            "A native Metal SwiGLU backward must consume the recomputed gate/up surfaces without host conversion.",
            "Actual full-model allocator release and a faster sustained batch remain unmeasured.",
            "Full-step replay, save/reload, resource, thermal, and independent audit remain open.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--native-binary", type=Path, default=DEFAULT_NATIVE_BINARY
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds < 2 or args.warmup < 1 or args.repetitions < 2:
        raise ValueError("rounds >= 2, warmup >= 1, repetitions >= 2 required")
    build_native(args.source, args.native_binary)
    native_command = [
        str(args.native_binary),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
    ]
    mlx_command = [
        sys.executable,
        str(MLX_WORKER),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
    ]
    standalone_native: list[dict[str, Any]] = []
    standalone_mlx: list[dict[str, Any]] = []
    concurrent: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        if round_index % 2 == 0:
            standalone_native.append(run_worker(native_command))
            standalone_mlx.append(run_worker(mlx_command))
        else:
            standalone_mlx.append(run_worker(mlx_command))
            standalone_native.append(run_worker(native_command))
        concurrent.append(barrier_round(native_command, mlx_command))
    decision = adjudicate(
        standalone_native,
        standalone_mlx,
        concurrent,
        repetitions=args.repetitions,
    )
    if decision["joined_wall_selected"]:
        trigger_state = "INCONCLUSIVE_IMPLEMENTATION_NATIVE_BRIDGE_REQUIRED"
        disposition = "OVERLAP_GREEN_NATIVE_METAL_BACKWARD_INTEGRATION_NEXT"
    else:
        trigger_state = "RED_RECOMPUTE_SCHEDULE_NOT_SELECTED"
        disposition = "RETAIN_MLX_CHECKPOINT_RECOMPUTATION"
    report = {
        "policy": POLICY,
        "trigger_state": trigger_state,
        "disposition": disposition,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "topology": {
            "discarded_activation": (
                "SwiGLU gate and up FP32 outputs for one lower decoder layer"
            ),
            "ane_recompute": (
                "six compile-once 2048x512 by 512x512 private-ANE projections "
                "covering gate/up width 1536"
            ),
            "independent_metal_window": (
                "compiled production-shape GQA/RoPE/SDPA self-attention forward/VJP "
                "from a later decoder layer"
            ),
            "dependency_valid": True,
            "fused_3072_and_1536_channel_mil_compile_attempts": (
                "INCOMPATIBLE_ON_THIS_M1_EXACT_512_CHANNEL_CHUNKS_USED"
            ),
        },
        "decision": decision,
        "raw": {
            "standalone_native": standalone_native,
            "standalone_mlx": standalone_mlx,
            "concurrent": concurrent,
        },
        "canonical_backend_changed": False,
        "canonical_checkpoint_mutated": False,
        "public_benchmark_rows_read": 0,
        "external_inference_calls": 0,
        "claim_scope": (
            "Exact-shape recomputation and independent-backward overlap mechanics. "
            "No full-model memory, optimizer, training-throughput, convergence, "
            "utility, serving, or capability claim."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.out)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
