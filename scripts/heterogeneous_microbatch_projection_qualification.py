#!/usr/bin/env python3
"""Qualify one exact two-shard heterogeneous q_proj update transaction.

This is a station-level mechanics gate.  One disjoint shard runs through the
native ANE/Accelerate gradient-only path while one runs through compiled MLX.
Both use one sealed FP32 weight generation and a global objective denominator;
the fail-closed contract joins them, clips once, and publishes one AdamW update.
"""

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
from typing import Any, Callable

import numpy as np

from heterogeneous_microbatch_contract import (
    GradientContribution,
    clip_and_adamw_once,
    gradient_schema,
    join_gradient_contributions,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_heterogeneous_microbatch_projection_qualification_v1"
NATIVE_SOURCE = ROOT / "native/ane_metal/ane_cpu_metal_projection_triad.m"
NATIVE_BINARY = Path("/private/tmp/theseus_heterogeneous_projection_shard")
ROWS = 2048
DIM = 512
ELEMENTS = ROWS * DIM
WEIGHT_ELEMENTS = DIM * DIM


class QualificationFault(ValueError):
    pass


def compare(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> dict[str, Any]:
    delta = np.abs(
        np.asarray(actual, dtype=np.float64)
        - np.asarray(expected, dtype=np.float64)
    )
    return {
        "tolerance": tolerance,
        "maximum_absolute_delta": float(delta.max(initial=0.0)),
        "mismatch_count": int(np.count_nonzero(delta > tolerance)),
        "all_finite": bool(np.all(np.isfinite(actual))),
    }


def build_native() -> None:
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
        str(NATIVE_SOURCE),
        "-o",
        str(NATIVE_BINARY),
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    if completed.returncode:
        raise QualificationFault(
            "native_build_failed:"
            + completed.stdout.decode("utf-8", errors="replace")[-2000:]
        )


def frozen_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2026072721)
    return {
        "weight": (rng.standard_normal((DIM, DIM)) * 0.03125).astype(np.float32),
        "x_a": (rng.standard_normal((ROWS, DIM)) * 0.125).astype(np.float32),
        "target_a": (rng.standard_normal((ROWS, DIM)) * 0.0625).astype(np.float32),
        "x_b": (rng.standard_normal((ROWS, DIM)) * 0.125).astype(np.float32),
        "target_b": (rng.standard_normal((ROWS, DIM)) * 0.0625).astype(np.float32),
    }


def build_mlx_gradient(mx: Any) -> Callable[..., tuple[Any, ...]]:
    @mx.compile
    def gradient(weight: Any, x: Any, target: Any) -> tuple[Any, ...]:
        output = x.astype(mx.float16) @ weight.astype(mx.float16)
        difference = output.astype(mx.float32) - target
        loss = (
            mx.array(0.5 / ELEMENTS, dtype=mx.float32)
            * mx.sum(mx.square(difference))
        )
        dy = difference * mx.array(1.0 / ELEMENTS, dtype=mx.float32)
        dx = dy.astype(mx.float16) @ weight.astype(mx.float16).T
        dw = x.astype(mx.float32).T @ dy
        return loss, output, dx, dw

    return gradient


def eval_gradient(
    mx: Any,
    gradient: Callable[..., tuple[Any, ...]],
    weight: Any,
    x: Any,
    target: Any,
) -> tuple[float, np.ndarray]:
    loss, _output, _dx, dw = gradient(weight, x, target)
    mx.eval(loss, dw)
    return float(loss.item()), np.asarray(dw, dtype=np.float32)


def write_native_inputs(root: Path, values: dict[str, np.ndarray]) -> Path:
    input_root = root / "native_inputs"
    input_root.mkdir()
    values["x_b"].tofile(input_root / "x_f32.bin")
    values["weight"].tofile(input_root / "weight_f32.bin")
    values["target_b"].tofile(input_root / "target_f32.bin")
    return input_root


def native_command(
    root: Path,
    input_root: Path,
    *,
    repetitions: int,
    warmup: int,
    ready: Path,
    go: Path,
) -> list[str]:
    return [
        str(NATIVE_BINARY),
        "--out",
        str(root / "native_report.json"),
        "--artifact-dir",
        str(root / "native_artifacts"),
        "--input-dir",
        str(input_root),
        "--steps",
        str(repetitions),
        "--warmup",
        str(warmup),
        "--gradient-only",
        "--ready-file",
        str(ready),
        "--go-file",
        str(go),
    ]


def run_concurrent_round(
    mx: Any,
    gradient: Callable[..., tuple[Any, ...]],
    arrays: dict[str, Any],
    values: dict[str, np.ndarray],
    *,
    repetitions: int,
    warmup: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="theseus_heterogeneous_projection_", dir="/private/tmp"
    ) as temporary:
        root = Path(temporary)
        ready = root / "native.ready"
        go = root / "go"
        input_root = write_native_inputs(root, values)
        native = subprocess.Popen(
            native_command(
                root,
                input_root,
                repetitions=repetitions,
                warmup=warmup,
                ready=ready,
                go=go,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 30.0
        while not ready.exists():
            if native.poll() is not None:
                output = native.communicate()[0] or b""
                raise QualificationFault(
                    "native_exited_before_ready:"
                    + output.decode("utf-8", errors="replace")[-2000:]
                )
            if time.monotonic() >= deadline:
                native.kill()
                raise QualificationFault("native_ready_timeout")
            time.sleep(0.001)
        go.write_text("go\n", encoding="utf-8")
        mlx_started = time.perf_counter()
        latest_loss = 0.0
        mlx_dw = None
        for _ in range(repetitions):
            latest_loss, mlx_dw = eval_gradient(
                mx,
                gradient,
                arrays["weight"],
                arrays["x_a"],
                arrays["target_a"],
            )
        mlx_milliseconds = (time.perf_counter() - mlx_started) * 1000.0
        output = native.communicate()[0] or b""
        if native.returncode:
            raise QualificationFault(
                f"native_failed_{native.returncode}:"
                + output.decode("utf-8", errors="replace")[-2000:]
            )
        report = json.loads((root / "native_report.json").read_text())
        if int((report["execution"]).get("optimizer_steps", -1)) != 0:
            raise QualificationFault("native_shard_performed_local_update")
        native_dw = np.fromfile(
            root / "native_artifacts/step1_dw_f32.bin", dtype=np.float32
        ).reshape(DIM, DIM)
        native_joined_mean = float(
            report["timing"]["station_means_milliseconds"]["joined"]
        )
        return {
            "mlx_loss": latest_loss,
            "mlx_dw": mlx_dw,
            "native_dw": native_dw,
            "mlx_total_milliseconds": mlx_milliseconds,
            "native_total_milliseconds": native_joined_mean * repetitions,
            "critical_path_milliseconds": max(
                mlx_milliseconds, native_joined_mean * repetitions
            ),
            "native_report": report,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds < 2 or args.warmup < 1 or args.repetitions < 2:
        raise ValueError("rounds >= 2, warmup >= 1, repetitions >= 2 required")
    import mlx.core as mx

    build_native()
    values = frozen_inputs()
    arrays = {name: mx.array(value) for name, value in values.items()}
    mx.eval(*arrays.values())
    gradient = build_mlx_gradient(mx)
    for _ in range(args.warmup):
        eval_gradient(
            mx, gradient, arrays["weight"], arrays["x_a"], arrays["target_a"]
        )
        eval_gradient(
            mx, gradient, arrays["weight"], arrays["x_b"], arrays["target_b"]
        )

    control_rounds: list[float] = []
    candidate_rounds: list[dict[str, Any]] = []
    control_dw = None
    for round_index in range(args.rounds):
        if round_index % 2 == 0:
            started = time.perf_counter()
            for _ in range(args.repetitions):
                _, dw_a = eval_gradient(
                    mx, gradient, arrays["weight"], arrays["x_a"], arrays["target_a"]
                )
                _, dw_b = eval_gradient(
                    mx, gradient, arrays["weight"], arrays["x_b"], arrays["target_b"]
                )
            control_rounds.append((time.perf_counter() - started) * 1000.0)
            control_dw = np.float32(0.5) * (dw_a + dw_b)
            candidate_rounds.append(
                run_concurrent_round(
                    mx,
                    gradient,
                    arrays,
                    values,
                    repetitions=args.repetitions,
                    warmup=args.warmup,
                )
            )
        else:
            candidate_rounds.append(
                run_concurrent_round(
                    mx,
                    gradient,
                    arrays,
                    values,
                    repetitions=args.repetitions,
                    warmup=args.warmup,
                )
            )
            started = time.perf_counter()
            for _ in range(args.repetitions):
                _, dw_a = eval_gradient(
                    mx, gradient, arrays["weight"], arrays["x_a"], arrays["target_a"]
                )
                _, dw_b = eval_gradient(
                    mx, gradient, arrays["weight"], arrays["x_b"], arrays["target_b"]
                )
            control_rounds.append((time.perf_counter() - started) * 1000.0)
            control_dw = np.float32(0.5) * (dw_a + dw_b)

    row_ids = [f"row{index}" for index in range(8)]
    row_masses = {row_id: 512.0 for row_id in row_ids}
    schema = gradient_schema({"weight": control_dw})
    latest = candidate_rounds[-1]
    joined_dw, join_receipt = join_gradient_contributions(
        [
            GradientContribution(
                shard_id="mlx_shard_a",
                engine="mlx_metal",
                generation=0,
                row_ids=tuple(row_ids[:4]),
                row_objective_masses=tuple(512.0 for _ in range(4)),
                global_objective_mass=4096.0,
                gradients={
                    "weight": np.float32(0.5) * latest["mlx_dw"]
                },
            ),
            GradientContribution(
                shard_id="ane_shard_b",
                engine="ane_accelerate",
                generation=0,
                row_ids=tuple(row_ids[4:]),
                row_objective_masses=tuple(512.0 for _ in range(4)),
                global_objective_mass=4096.0,
                gradients={
                    "weight": np.float32(0.5) * latest["native_dw"]
                },
            ),
        ],
        generation=0,
        expected_rows=row_ids,
        expected_row_objective_masses=row_masses,
        expected_schema=schema,
    )
    zeros = {"weight": np.zeros_like(values["weight"])}
    candidate_state = clip_and_adamw_once(
        {"weight": values["weight"]},
        zeros,
        zeros,
        joined_dw,
        learning_rate=3e-5,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        clip_norm=1.0,
    )
    control_state = clip_and_adamw_once(
        {"weight": values["weight"]},
        zeros,
        zeros,
        {"weight": control_dw},
        learning_rate=3e-5,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        clip_norm=1.0,
    )
    parity = {
        "combined_gradient": compare(
            joined_dw["weight"], control_dw, 0.01
        ),
        "updated_weight": compare(
            candidate_state[0]["weight"], control_state[0]["weight"], 1e-4
        ),
        "first_moment": compare(
            candidate_state[1]["weight"], control_state[1]["weight"], 1e-4
        ),
        "second_moment": compare(
            candidate_state[2]["weight"], control_state[2]["weight"], 1e-6
        ),
    }
    parity_green = all(
        item["mismatch_count"] == 0 and item["all_finite"]
        for item in parity.values()
    )
    candidate_totals = [
        float(row["critical_path_milliseconds"]) for row in candidate_rounds
    ]
    mean_speedup = statistics.fmean(control_rounds) / statistics.fmean(
        candidate_totals
    )
    conservative_speedup = min(control_rounds) / max(candidate_totals)
    report = {
        "policy": POLICY,
        "trigger_state": (
            "INCONCLUSIVE_IMPLEMENTATION_FULL_ANE_MODEL_PARITY_REQUIRED"
            if parity_green and conservative_speedup > 1.0
            else "RED_STATION_HETEROGENEOUS_SHARD_NOT_SELECTED"
        ),
        "disposition": (
            "STATION_MECHANICS_GREEN_FULL_MODEL_GATED"
            if parity_green and conservative_speedup > 1.0
            else "RETAIN_MLX_FOR_EXACT_STATION"
        ),
        "shape": {
            "logical_batch": 8,
            "shard_batch": 4,
            "sequence": 512,
            "hidden": 512,
            "projection": "decoder_self_attention_q_proj",
        },
        "rounds": args.rounds,
        "repetitions": args.repetitions,
        "timing": {
            "control_two_mlx_shards_total_milliseconds": control_rounds,
            "candidate_critical_path_total_milliseconds": candidate_totals,
            "mean_speedup_control_over_candidate": mean_speedup,
            "conservative_speedup_control_over_candidate": conservative_speedup,
        },
        "parity": parity,
        "join_receipt": join_receipt,
        "update_receipt": candidate_state[3],
        "gates": {
            "sampler_exact_disjoint_coverage": True,
            "objective_mass_conserved": True,
            "single_fp32_generation": True,
            "no_per_device_optimizer": True,
            "one_global_clip_and_adamw": True,
            "station_gradient_and_update_parity": parity_green,
            "matched_station_wall_gain": conservative_speedup > 1.0,
            "complete_ane_model_gradient_tree_parity": False,
            "no_python_or_numpy_gradient_bridge": False,
            "full_step_replay_resource_thermal_audit": False,
        },
        "production_eligible": False,
        "canonical_checkpoint_mutated": False,
        "public_benchmark_rows_read": 0,
        "external_inference_calls": 0,
        "claim_scope": (
            "One exact q_proj station with two disjoint sampler shards and one "
            "global update. This is not a transformer, full gradient tree, "
            "training-throughput, convergence, utility, or capability claim."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.out)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
