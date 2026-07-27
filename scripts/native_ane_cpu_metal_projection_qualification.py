#!/usr/bin/env python3
"""Qualify the native IOSurface ANE+Accelerate+Metal q_proj transaction.

The native candidate owns its complete measured hot step. This driver builds
it, runs alternating sustained candidate/control rounds, independently
reconstructs the deterministic inputs, executes a matched compiled MLX
transaction, compares full station and optimizer artifacts, and emits one
fail-closed adjudication.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


POLICY = "project_theseus_native_ane_cpu_metal_projection_qualification_v1"
ROWS = 2048
DIM = 512
ACTIVATION_ELEMENTS = ROWS * DIM
WEIGHT_ELEMENTS = DIM * DIM
SOURCE = Path("native/ane_metal/ane_cpu_metal_projection_triad.m")
DEFAULT_BINARY = Path("/private/tmp/theseus_native_projection_triad")
NATIVE_POLICY = "project_theseus_native_ane_cpu_metal_projection_triad_v1"


class QualificationFault(ValueError):
    """Raised when an evidence packet cannot support a decision."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def compare(
    observed: np.ndarray, reference: np.ndarray, *, tolerance: float
) -> dict[str, float | int]:
    if observed.shape != reference.shape:
        raise QualificationFault(
            f"comparison_shape_mismatch:{observed.shape}:{reference.shape}"
        )
    delta = np.abs(
        observed.astype(np.float64) - reference.astype(np.float64)
    )
    return {
        "element_count": int(delta.size),
        "mismatch_count": int(np.count_nonzero(delta > tolerance)),
        "maximum_absolute_delta": float(delta.max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))),
        "tolerance": float(tolerance),
    }


def deterministic_values() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260727)
    x = (
        rng.standard_normal((ROWS, DIM), dtype=np.float32) * 0.125
    ).astype(np.float32)
    weight = (
        rng.standard_normal((DIM, DIM), dtype=np.float32) * 0.03125
    ).astype(np.float32)
    target = (
        rng.standard_normal((ROWS, DIM), dtype=np.float32) * 0.0625
    ).astype(np.float32)
    return x, weight, target


def write_frozen_inputs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    x, weight, target = deterministic_values()
    x.tofile(directory / "x_f32.bin")
    weight.tofile(directory / "weight_f32.bin")
    target.tofile(directory / "target_f32.bin")


def timing_summary(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2 or any(
        not math.isfinite(value) or value <= 0 for value in values
    ):
        raise QualificationFault("timings_require_two_finite_positive_values")
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(values),
        "minimum_milliseconds": min(values),
        "median_milliseconds": statistics.median(values),
        "mean_milliseconds": statistics.fmean(values),
        "p95_milliseconds": ordered[p95_index],
        "maximum_milliseconds": max(values),
    }


def build_native(source: Path, binary: Path) -> dict[str, Any]:
    command = [
        "xcrun",
        "clang",
        "-fobjc-arc",
        "-O3",
        "-DACCELERATE_NEW_LAPACK",
        "-Wall",
        "-Wextra",
        "-framework",
        "Foundation",
        "-framework",
        "CoreVideo",
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
    started = time.perf_counter()
    process = subprocess.run(
        command, text=True, capture_output=True, check=False
    )
    if process.returncode != 0:
        raise QualificationFault(
            "native_build_failed:" + process.stderr[-2000:]
        )
    return {
        "command": command,
        "wall_seconds": time.perf_counter() - started,
        "stderr": process.stderr,
    }


def run_native(
    binary: Path,
    *,
    root: Path,
    steps: int,
    warmup: int,
    input_root: Path,
) -> dict[str, Any]:
    report_path = root / "native_report.json"
    artifact_root = root / "native_artifacts"
    process = subprocess.run(
        [
            str(binary),
            "--out",
            str(report_path),
            "--artifact-dir",
            str(artifact_root),
            "--steps",
            str(steps),
            "--warmup",
            str(warmup),
            "--input-dir",
            str(input_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise QualificationFault(
            f"native_candidate_failed:{process.returncode}:"
            + process.stderr[-2000:]
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("policy") != NATIVE_POLICY:
        raise QualificationFault("native_report_policy_mismatch")
    return report


def build_mlx_control(
    mx: Any,
) -> Callable[..., tuple[Any, ...]]:
    learning_rate = 3.0e-5
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1.0e-8
    weight_decay = 0.01
    clip_norm = 1.0

    @mx.compile
    def control(
        weight: Any,
        first: Any,
        second: Any,
        x: Any,
        target: Any,
    ) -> tuple[Any, ...]:
        compute_weight = weight.astype(mx.float16)
        output = x.astype(mx.float16) @ compute_weight
        difference = output.astype(mx.float32) - target
        loss = (
            mx.array(0.5 / ACTIVATION_ELEMENTS, dtype=mx.float32)
            * mx.sum(mx.square(difference))
        )
        dy = difference * mx.array(
            1.0 / ACTIVATION_ELEMENTS, dtype=mx.float32
        )
        dx = dy.astype(mx.float16) @ compute_weight.T
        dw = x.astype(mx.float32).T @ dy
        gradient_norm = mx.sqrt(mx.sum(mx.square(dw)))
        clip_scale = mx.minimum(
            mx.array(1.0, dtype=mx.float32),
            mx.array(clip_norm, dtype=mx.float32)
            / mx.maximum(
                gradient_norm, mx.array(1.0e-6, dtype=mx.float32)
            ),
        )
        clipped = dw * clip_scale
        next_first = beta1 * first + (1.0 - beta1) * clipped
        next_second = beta2 * second + (1.0 - beta2) * mx.square(clipped)
        decayed = weight * (1.0 - learning_rate * weight_decay)
        next_weight = (
            decayed
            - learning_rate
            * next_first
            / (mx.sqrt(next_second) + epsilon)
        )
        return (
            loss,
            output,
            dx,
            dw,
            next_weight,
            next_first,
            next_second,
        )

    return control


def run_mlx(
    mx: Any,
    control: Callable[..., tuple[Any, ...]],
    *,
    steps: int,
    warmup: int,
) -> dict[str, Any]:
    x_host, weight_host, target_host = deterministic_values()
    x = mx.array(x_host, dtype=mx.float32)
    target = mx.array(target_host, dtype=mx.float32)
    initial_weight = mx.array(weight_host, dtype=mx.float32)
    initial_first = mx.zeros_like(initial_weight)
    initial_second = mx.zeros_like(initial_weight)
    for _ in range(warmup):
        outputs = control(
            initial_weight, initial_first, initial_second, x, target
        )
        mx.eval(*outputs)

    weight = initial_weight
    first = initial_first
    second = initial_second
    timings: list[float] = []
    first_station: dict[str, np.ndarray | float] | None = None
    loss_prefix: list[float] = []
    for step in range(steps):
        started = time.perf_counter()
        outputs = control(weight, first, second, x, target)
        mx.eval(*outputs)
        timings.append((time.perf_counter() - started) * 1000.0)
        (
            loss,
            output,
            dx,
            dw,
            weight,
            first,
            second,
        ) = outputs
        if len(loss_prefix) < 8:
            loss_prefix.append(float(loss.item()))
        if step == 0:
            first_station = {
                "loss": float(loss.item()),
                "output": np.asarray(output, dtype=np.float16),
                "dx": np.asarray(dx, dtype=np.float16),
                "dw": np.asarray(dw, dtype=np.float32),
            }
    if first_station is None:
        raise QualificationFault("mlx_first_station_missing")
    return {
        "timing": timing_summary(timings),
        "timing_samples_milliseconds": timings,
        "loss_prefix": loss_prefix,
        "first_station": first_station,
        "final_weight": np.asarray(weight, dtype=np.float32),
        "final_first": np.asarray(first, dtype=np.float32),
        "final_second": np.asarray(second, dtype=np.float32),
    }


def native_artifacts(report: dict[str, Any]) -> dict[str, np.ndarray]:
    artifacts = report.get("artifacts") or {}
    if artifacts.get("written") is not True:
        raise QualificationFault("native_artifacts_not_written")

    def read(key: str, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        path = Path(str(artifacts.get(key) or ""))
        values = np.fromfile(path, dtype=dtype)
        if values.size != math.prod(shape):
            raise QualificationFault(f"native_artifact_shape_invalid:{key}")
        return values.reshape(shape)

    return {
        "step1_output": read(
            "step1_output_f16", np.float16, (DIM, ROWS)
        ).T.copy(),
        "step1_dx": read("step1_dx_f16", np.float16, (DIM, ROWS)).T.copy(),
        "step1_dw": read("step1_dw_f32", np.float32, (DIM, DIM)),
        "final_weight": read(
            "final_weight_f32", np.float32, (DIM, DIM)
        ),
        "final_first": read(
            "final_first_moment_f32", np.float32, (DIM, DIM)
        ),
        "final_second": read(
            "final_second_moment_f32", np.float32, (DIM, DIM)
        ),
    }


def adjudicate(
    *,
    native_reports: list[dict[str, Any]],
    mlx_runs: list[dict[str, Any]],
    parity: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    native_means = [
        float(
            report["timing"]["summaries_milliseconds"]["joined"][
                "mean_milliseconds"
            ]
        )
        for report in native_reports
    ]
    mlx_means = [
        float(run["timing"]["mean_milliseconds"]) for run in mlx_runs
    ]
    conservative_speedup = min(mlx_means) / max(native_means)
    mean_speedup = statistics.fmean(mlx_means) / statistics.fmean(
        native_means
    )
    parity_green = all(
        int(record["mismatch_count"]) == 0 for record in parity.values()
    )
    mechanics_green = all(
        report["stability"]["all_tensors_finite"] is True
        and report["stability"]["save_reload_exact"] is True
        and report["stability"]["replay_exact"] is True
        and report["custody"]["single_generation_conserved"] is True
        and report["custody"]["intermediate_host_tensor_copy"] is False
        and report["custody"]["hot_step_python_or_numpy"] is False
        for report in native_reports
    )
    wall_green = conservative_speedup > 1.0
    return {
        "native_round_mean_milliseconds": native_means,
        "mlx_round_mean_milliseconds": mlx_means,
        "mean_speedup_mlx_over_native": mean_speedup,
        "conservative_speedup_mlx_over_native": conservative_speedup,
        "parity_green": parity_green,
        "native_mechanics_green": mechanics_green,
        "matched_joined_wall_gain_exceeds_uncertainty": wall_green,
        "selected": mechanics_green and parity_green and wall_green,
        "disposition": (
            "NATIVE_ZERO_COPY_TRIAD_STATION_SELECTED_PENDING_FULL_REMAINDER"
            if mechanics_green and parity_green and wall_green
            else "NATIVE_ZERO_COPY_TRIAD_NOT_SELECTED"
            if mechanics_green and parity_green
            else "INCONCLUSIVE_IMPLEMENTATION"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--native-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds < 2 or args.steps < 2 or args.warmup < 0:
        raise QualificationFault("qualification_bounds_invalid")
    build = build_native(args.source, args.native_binary)
    import mlx.core as mx

    control = build_mlx_control(mx)
    native_reports: list[dict[str, Any]] = []
    mlx_runs: list[dict[str, Any]] = []
    route_order: list[str] = []
    with tempfile.TemporaryDirectory(
        prefix="theseus_native_projection_qualification_"
    ) as temporary_text:
        temporary = Path(temporary_text)
        input_root = temporary / "frozen_inputs"
        write_frozen_inputs(input_root)
        for round_index in range(args.rounds):
            native_root = temporary / f"round_{round_index}_native"
            native_root.mkdir(parents=True)
            if round_index % 2 == 0:
                route_order.extend(["native", "mlx"])
                native_reports.append(
                    run_native(
                        args.native_binary,
                        root=native_root,
                        steps=args.steps,
                        warmup=args.warmup,
                        input_root=input_root,
                    )
                )
                mlx_runs.append(
                    run_mlx(
                        mx, control, steps=args.steps, warmup=args.warmup
                    )
                )
            else:
                route_order.extend(["mlx", "native"])
                mlx_runs.append(
                    run_mlx(
                        mx, control, steps=args.steps, warmup=args.warmup
                    )
                )
                native_reports.append(
                    run_native(
                        args.native_binary,
                        root=native_root,
                        steps=args.steps,
                        warmup=args.warmup,
                        input_root=input_root,
                    )
                )

        native = native_artifacts(native_reports[0])
        mlx = mlx_runs[0]
        station = mlx["first_station"]
        parity = {
            "output": compare(
                native["step1_output"],
                station["output"],
                tolerance=0.001,
            ),
            "input_gradient": compare(
                native["step1_dx"], station["dx"], tolerance=0.001
            ),
            "weight_gradient": compare(
                native["step1_dw"], station["dw"], tolerance=0.01
            ),
            "final_weight": compare(
                native["final_weight"],
                mlx["final_weight"],
                tolerance=0.0001,
            ),
            "final_first_moment": compare(
                native["final_first"],
                mlx["final_first"],
                tolerance=0.0001,
            ),
            "final_second_moment": compare(
                native["final_second"],
                mlx["final_second"],
                tolerance=0.0001,
            ),
        }
        decision = adjudicate(
            native_reports=native_reports,
            mlx_runs=mlx_runs,
            parity=parity,
        )
        report = {
            "policy": POLICY,
            "trigger_state": (
                "GREEN_STATION_SELECTED"
                if decision["selected"]
                else "RED_FALLBACK_TO_QUALIFIED_MLX"
                if decision["native_mechanics_green"]
                and decision["parity_green"]
                else "INCONCLUSIVE_IMPLEMENTATION"
            ),
            "disposition": decision["disposition"],
            "production_eligible": False,
            "shape": native_reports[0]["shape"],
            "build": build,
            "route_order": route_order,
            "rounds": args.rounds,
            "optimizer_steps_per_round": args.steps,
            "warmup_steps_per_round": args.warmup,
            "parity": parity,
            "decision": decision,
            "native_rounds": [
                {
                    "timing": native_report["timing"],
                    "stability": native_report["stability"],
                    "custody": native_report["custody"],
                    "execution": native_report["execution"],
                }
                for native_report in native_reports
            ],
            "mlx_rounds": [
                {
                    "timing": mlx_run["timing"],
                    "loss_prefix": mlx_run["loss_prefix"],
                }
                for mlx_run in mlx_runs
            ],
            "gates": {
                "full_station_parity": decision["parity_green"],
                "native_mechanics": decision["native_mechanics_green"],
                "matched_joined_wall_gain_exceeds_uncertainty": decision[
                    "matched_joined_wall_gain_exceeds_uncertainty"
                ],
                "sustained_resource_and_thermal_qualification": False,
                "real_metal_attention_pointer_loss_remainder": False,
                "sampler_and_objective_mass_conservation": False,
                "independent_gate_audit": False,
            },
            "claim_scope": (
                "One deterministic production-shape q_proj station. Selection "
                "would authorize only the next real-remainder integration gate, "
                "not checkpoint mutation or a full-training speedup claim."
            ),
            "canonical_checkpoint_mutated": False,
            "public_benchmark_rows_read": 0,
            "external_inference_calls": 0,
        }
    atomic_json(args.out, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
