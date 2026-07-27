#!/usr/bin/env python3
"""Measure ANE-inspired projection fusion on the production-shaped MLX route.

The maderix/ANE runtime reduces dispatches by grouping Q/K/V projections and
the two SwiGLU input projections.  This probe tests the same algebra in MLX
without changing parameter names, shapes, checkpoint bytes, or optimizer
ownership: independently stored weights are concatenated inside the compiled
graph and receive gradients through that concatenation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


POLICY = "project_theseus_mlx_projection_fusion_probe_v1"
STATIONS = ("qkv_projection", "swiglu_gate_up")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", choices=STATIONS, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--head-count", type=int, default=8)
    parser.add_argument("--kv-head-count", type=int, default=2)
    parser.add_argument("--ff-size", type=int, default=1536)
    parser.add_argument("--layer-count", type=int, default=12)
    parser.add_argument("--reference-microbatch-seconds", type=float, default=0.56)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def relative_l2(reference: Any, candidate: Any, *, mx: Any) -> float:
    delta = (candidate.astype(mx.float32) - reference.astype(mx.float32)).reshape(-1)
    base = reference.astype(mx.float32).reshape(-1)
    numerator = float(mx.sqrt(mx.sum(mx.square(delta))).item())
    denominator = float(mx.sqrt(mx.sum(mx.square(base))).item())
    return numerator / max(denominator, 1e-30)


def maximum_absolute_delta(reference: Any, candidate: Any, *, mx: Any) -> float:
    return float(
        mx.max(
            mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32))
        ).item()
    )


def paired_timing(
    control: Callable[..., Any],
    candidate: Callable[..., Any],
    arguments: tuple[Any, ...],
    *,
    mx: Any,
    warmup: int,
    pairs: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        mx.eval(control(*arguments))
        mx.eval(candidate(*arguments))
    control_seconds: list[float] = []
    candidate_seconds: list[float] = []
    for pair in range(pairs):
        order = (
            (("control", control), ("candidate", candidate))
            if pair % 2 == 0
            else (("candidate", candidate), ("control", control))
        )
        for label, function in order:
            started = time.perf_counter()
            mx.eval(function(*arguments))
            duration = time.perf_counter() - started
            if label == "control":
                control_seconds.append(duration)
            else:
                candidate_seconds.append(duration)
    return control_seconds, candidate_seconds


def timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": round(min(values), 6),
        "median": round(statistics.median(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "maximum": round(max(values), 6),
    }


def projected_microbatch_fraction(
    control_median: float,
    candidate_median: float,
    *,
    layer_count: int,
    reference_microbatch_seconds: float,
) -> float:
    saved = max(0.0, control_median - candidate_median) * layer_count
    return saved / reference_microbatch_seconds


def qkv_probe(args: argparse.Namespace, *, mx: Any) -> dict[str, Any]:
    head_dim = args.hidden_size // args.head_count
    kv_width = args.kv_head_count * head_dim
    shape = (args.batch_size, args.sequence_length, args.hidden_size)
    mx.random.seed(2026072701)
    hidden = mx.random.normal(shape, dtype=mx.float32)
    query_weight = (
        mx.random.normal(
            (args.hidden_size, args.hidden_size), dtype=mx.float32
        )
        * 0.02
    )
    key_weight = (
        mx.random.normal((kv_width, args.hidden_size), dtype=mx.float32) * 0.02
    )
    value_weight = (
        mx.random.normal((kv_width, args.hidden_size), dtype=mx.float32) * 0.02
    )
    cotangent = mx.random.normal(
        (
            args.batch_size,
            args.sequence_length,
            args.hidden_size + 2 * kv_width,
        ),
        dtype=mx.float32,
    )
    mx.eval(hidden, query_weight, key_weight, value_weight, cotangent)

    def control_forward(x: Any, qw: Any, kw: Any, vw: Any) -> Any:
        return mx.concatenate(
            [
                mx.matmul(x, qw.T),
                mx.matmul(x, kw.T),
                mx.matmul(x, vw.T),
            ],
            axis=-1,
        )

    def candidate_forward(x: Any, qw: Any, kw: Any, vw: Any) -> Any:
        joined_weight = mx.concatenate([qw, kw, vw], axis=0)
        return mx.matmul(x, joined_weight.T)

    def control_objective(
        x: Any, qw: Any, kw: Any, vw: Any, cot: Any
    ) -> Any:
        output = control_forward(x, qw, kw, vw)
        return mx.sum(output * cot) / output.size

    def candidate_objective(
        x: Any, qw: Any, kw: Any, vw: Any, cot: Any
    ) -> Any:
        output = candidate_forward(x, qw, kw, vw)
        return mx.sum(output * cot) / output.size

    arguments = (hidden, query_weight, key_weight, value_weight, cotangent)
    control_compiled = mx.compile(
        mx.value_and_grad(control_objective, argnums=(0, 1, 2, 3))
    )
    candidate_compiled = mx.compile(
        mx.value_and_grad(candidate_objective, argnums=(0, 1, 2, 3))
    )
    control_output = control_forward(*arguments[:-1])
    candidate_output = candidate_forward(*arguments[:-1])
    control_result = control_compiled(*arguments)
    candidate_result = candidate_compiled(*arguments)
    mx.eval(control_output, candidate_output, control_result, candidate_result)
    control_seconds, candidate_seconds = paired_timing(
        control_compiled,
        candidate_compiled,
        arguments,
        mx=mx,
        warmup=args.warmup,
        pairs=args.pairs,
    )
    control_loss, control_gradients = control_result
    candidate_loss, candidate_gradients = candidate_result
    return comparison_report(
        args,
        mx=mx,
        control_output=control_output,
        candidate_output=candidate_output,
        control_loss=control_loss,
        candidate_loss=candidate_loss,
        control_gradients=control_gradients,
        candidate_gradients=candidate_gradients,
        control_seconds=control_seconds,
        candidate_seconds=candidate_seconds,
        retained_parameter_shapes=[
            list(query_weight.shape),
            list(key_weight.shape),
            list(value_weight.shape),
        ],
    )


def swiglu_probe(args: argparse.Namespace, *, mx: Any) -> dict[str, Any]:
    shape = (args.batch_size, args.sequence_length, args.hidden_size)
    mx.random.seed(2026072702)
    hidden = mx.random.normal(shape, dtype=mx.float32)
    gate_weight = (
        mx.random.normal(
            (args.ff_size, args.hidden_size), dtype=mx.float32
        )
        * 0.02
    )
    up_weight = (
        mx.random.normal(
            (args.ff_size, args.hidden_size), dtype=mx.float32
        )
        * 0.02
    )
    cotangent = mx.random.normal(
        (args.batch_size, args.sequence_length, args.ff_size),
        dtype=mx.float32,
    )
    mx.eval(hidden, gate_weight, up_weight, cotangent)

    def control_forward(x: Any, gate: Any, up: Any) -> Any:
        gate_value = mx.matmul(x, gate.T)
        return (gate_value * mx.sigmoid(gate_value)) * mx.matmul(x, up.T)

    def candidate_forward(x: Any, gate: Any, up: Any) -> Any:
        joined = mx.matmul(x, mx.concatenate([gate, up], axis=0).T)
        gate_value, up_value = mx.split(joined, 2, axis=-1)
        return (gate_value * mx.sigmoid(gate_value)) * up_value

    def control_objective(x: Any, gate: Any, up: Any, cot: Any) -> Any:
        output = control_forward(x, gate, up)
        return mx.sum(output * cot) / output.size

    def candidate_objective(x: Any, gate: Any, up: Any, cot: Any) -> Any:
        output = candidate_forward(x, gate, up)
        return mx.sum(output * cot) / output.size

    arguments = (hidden, gate_weight, up_weight, cotangent)
    control_compiled = mx.compile(
        mx.value_and_grad(control_objective, argnums=(0, 1, 2))
    )
    candidate_compiled = mx.compile(
        mx.value_and_grad(candidate_objective, argnums=(0, 1, 2))
    )
    control_output = control_forward(*arguments[:-1])
    candidate_output = candidate_forward(*arguments[:-1])
    control_result = control_compiled(*arguments)
    candidate_result = candidate_compiled(*arguments)
    mx.eval(control_output, candidate_output, control_result, candidate_result)
    control_seconds, candidate_seconds = paired_timing(
        control_compiled,
        candidate_compiled,
        arguments,
        mx=mx,
        warmup=args.warmup,
        pairs=args.pairs,
    )
    control_loss, control_gradients = control_result
    candidate_loss, candidate_gradients = candidate_result
    return comparison_report(
        args,
        mx=mx,
        control_output=control_output,
        candidate_output=candidate_output,
        control_loss=control_loss,
        candidate_loss=candidate_loss,
        control_gradients=control_gradients,
        candidate_gradients=candidate_gradients,
        control_seconds=control_seconds,
        candidate_seconds=candidate_seconds,
        retained_parameter_shapes=[
            list(gate_weight.shape),
            list(up_weight.shape),
        ],
    )


def comparison_report(
    args: argparse.Namespace,
    *,
    mx: Any,
    control_output: Any,
    candidate_output: Any,
    control_loss: Any,
    candidate_loss: Any,
    control_gradients: tuple[Any, ...],
    candidate_gradients: tuple[Any, ...],
    control_seconds: list[float],
    candidate_seconds: list[float],
    retained_parameter_shapes: list[list[int]],
) -> dict[str, Any]:
    control_median = statistics.median(control_seconds)
    candidate_median = statistics.median(candidate_seconds)
    speedup = control_median / max(candidate_median, 1e-30)
    projected_fraction = projected_microbatch_fraction(
        control_median,
        candidate_median,
        layer_count=args.layer_count,
        reference_microbatch_seconds=args.reference_microbatch_seconds,
    )
    gradient_maximums = [
        maximum_absolute_delta(reference, candidate, mx=mx)
        for reference, candidate in zip(
            control_gradients, candidate_gradients, strict=True
        )
    ]
    gradient_relative_l2 = [
        relative_l2(reference, candidate, mx=mx)
        for reference, candidate in zip(
            control_gradients, candidate_gradients, strict=True
        )
    ]
    return {
        "control_timing_seconds": timing_summary(control_seconds),
        "candidate_timing_seconds": timing_summary(candidate_seconds),
        "candidate_speedup": round(speedup, 6),
        "projected_full_microbatch_saved_fraction_upper_bound": round(
            projected_fraction, 6
        ),
        "projected_full_microbatch_saved_percent_upper_bound": round(
            100.0 * projected_fraction, 4
        ),
        "integrity": {
            "output_maximum_absolute_delta": maximum_absolute_delta(
                control_output, candidate_output, mx=mx
            ),
            "output_relative_l2": relative_l2(
                control_output, candidate_output, mx=mx
            ),
            "loss_absolute_delta": abs(
                float(control_loss.item()) - float(candidate_loss.item())
            ),
            "gradient_maximum_absolute_deltas": gradient_maximums,
            "gradient_relative_l2": gradient_relative_l2,
            "all_candidate_gradients_finite": all(
                bool(mx.all(mx.isfinite(value)).item())
                for value in candidate_gradients
            ),
            "parameter_names_and_shapes_can_remain_separate": True,
            "retained_parameter_shapes": retained_parameter_shapes,
        },
        "decision": {
            "station_candidate_faster": candidate_median < control_median,
            "requires_full_route_replay_before_adoption": True,
            "adopted_in_production": False,
        },
    }


def main() -> int:
    args = parse_args()
    positive = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "head_count": args.head_count,
        "kv_head_count": args.kv_head_count,
        "ff_size": args.ff_size,
        "layer_count": args.layer_count,
        "warmup": args.warmup,
        "pairs": args.pairs,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"all dimensions and repetitions must be positive: {positive}")
    if args.hidden_size % args.head_count or args.head_count % args.kv_head_count:
        raise ValueError("hidden/head and grouped-query dimensions must divide evenly")
    if args.reference_microbatch_seconds <= 0:
        raise ValueError("reference microbatch duration must be positive")
    if os.environ.get("THESEUS_GUARDED_ACCELERATOR_CHILD") != "1":
        raise RuntimeError("external host-resource watchdog is required")

    import mlx.core as mx

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    station_report = {
        "qkv_projection": qkv_probe,
        "swiglu_gate_up": swiglu_probe,
    }[args.station](args, mx=mx)
    report = {
        "policy": POLICY,
        "created_utc": datetime.now(UTC).isoformat(),
        "trigger_state": "GREEN",
        "station": args.station,
        "claim_scope": (
            "Synthetic production-shape MLX station comparison only. It does not "
            "establish full-step, sustained, convergence, utility, or capability gains."
        ),
        "source_concept": {
            "repository": "https://github.com/maderix/ANE",
            "audited_commit": "d91c9845c0784dec7753048954fc6d0e8411fe29",
            "concepts": [
                "fused QKV projection dispatch",
                "fused SwiGLU gate/up projection dispatch",
            ],
            "source_code_imported": False,
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mlx_version": importlib.metadata.version("mlx"),
        },
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "head_count": args.head_count,
            "kv_head_count": args.kv_head_count,
            "ff_size": args.ff_size,
            "layer_count": args.layer_count,
        },
        "execution": {
            "compute_dtype": "float32",
            "compiled": True,
            "warmup": args.warmup,
            "alternating_pairs": args.pairs,
            "externally_guarded": True,
        },
        **station_report,
        "mlx_memory": {
            "active_bytes": int(mx.get_active_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
            "peak_bytes": int(mx.get_peak_memory()),
        },
        "canonical_checkpoint_mutated": False,
        "public_benchmark_rows_read": 0,
        "external_inference_calls": 0,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
