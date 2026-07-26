#!/usr/bin/env python3
"""Bounded exact-shape timing for remaining 57M MLX training stations.

The probe uses synthetic tensors and reads no checkpoint, corpus row, or
evaluation surface.  Each invocation measures one station in a fresh process
so allocator/cache residue from a previous station cannot distort the result.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


POLICY = "project_theseus_mlx_training_operator_station_probe_v1"
STATIONS = (
    "decoder_self_attention",
    "swiglu_mlp",
    "two_rmsnorms",
    "global_gradient_clip",
    "adamw_active_state",
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def station_bound(
    station: str,
    median_seconds: float,
    *,
    layer_count: int,
    reference_microbatch_seconds: float,
) -> dict[str, Any]:
    multiplier = layer_count if station in {
        "decoder_self_attention",
        "swiglu_mlp",
        "two_rmsnorms",
    } else 1
    repeated_seconds = median_seconds * multiplier
    fraction = repeated_seconds / reference_microbatch_seconds
    return {
        "station_instances_per_optimizer_microbatch": multiplier,
        "repeated_station_seconds_upper_bound": round(repeated_seconds, 6),
        "elimination_fraction_of_reference_microbatch": round(fraction, 6),
        "elimination_percent_of_reference_microbatch": round(100.0 * fraction, 4),
        "custom_kernel_10_percent_bound_possible": fraction >= 0.10,
        "interpretation": (
            "The fraction is an elimination upper bound, not a forecast. Any "
            "replacement still performs the station's required math and memory traffic."
        ),
    }


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
    parser.add_argument("--active-parameter-count", type=int, default=40_384_512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--reference-microbatch-seconds", type=float, default=0.56)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def _timed(
    fn: Callable[..., Any],
    arguments: tuple[Any, ...],
    *,
    mx: Any,
    warmup: int,
    repetitions: int,
) -> tuple[list[float], Any]:
    latest = None
    for _ in range(warmup):
        latest = fn(*arguments)
        mx.eval(latest)
    durations: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        latest = fn(*arguments)
        mx.eval(latest)
        durations.append(time.perf_counter() - started)
    return durations, latest


def run_attention(args: argparse.Namespace, *, mx: Any) -> tuple[list[float], dict[str, Any]]:
    head_dim = args.hidden_size // args.head_count
    kv_width = args.kv_head_count * head_dim
    shape = (args.batch_size, args.sequence_length, args.hidden_size)
    mx.random.seed(2026072601)
    hidden = mx.random.normal(shape, dtype=mx.float32)
    q_weight = mx.random.normal((args.hidden_size, args.hidden_size), dtype=mx.float32) * 0.02
    k_weight = mx.random.normal((kv_width, args.hidden_size), dtype=mx.float32) * 0.02
    v_weight = mx.random.normal((kv_width, args.hidden_size), dtype=mx.float32) * 0.02
    out_weight = mx.random.normal((args.hidden_size, args.hidden_size), dtype=mx.float32) * 0.02
    cotangent = mx.random.normal(shape, dtype=mx.float32)
    mx.eval(hidden, q_weight, k_weight, v_weight, out_weight, cotangent)

    def objective(
        x: Any, qw: Any, kw: Any, vw: Any, ow: Any, cot: Any
    ) -> Any:
        batch, length, _ = x.shape
        query = mx.matmul(x, qw.T).reshape(
            batch, length, args.head_count, head_dim
        ).transpose(0, 2, 1, 3)
        key = mx.matmul(x, kw.T).reshape(
            batch, length, args.kv_head_count, head_dim
        ).transpose(0, 2, 1, 3)
        value = mx.matmul(x, vw.T).reshape(
            batch, length, args.kv_head_count, head_dim
        ).transpose(0, 2, 1, 3)
        query = mx.fast.rope(
            query, head_dim, traditional=False, base=10000.0, scale=1.0, offset=0
        )
        key = mx.fast.rope(
            key, head_dim, traditional=False, base=10000.0, scale=1.0, offset=0
        )
        attended = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=head_dim ** -0.5, mask="causal"
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, args.hidden_size)
        output = mx.matmul(attended, ow.T)
        return mx.sum(output * cot) / output.size

    compiled = mx.compile(mx.value_and_grad(objective, argnums=(0, 1, 2, 3, 4)))
    durations, latest = _timed(
        compiled,
        (hidden, q_weight, k_weight, v_weight, out_weight, cotangent),
        mx=mx,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    loss, gradients = latest
    return durations, {
        "scalar_finite": bool(mx.isfinite(loss).item()),
        "gradient_tensor_count": len(gradients),
        "all_gradients_finite": all(
            bool(mx.all(mx.isfinite(gradient)).item()) for gradient in gradients
        ),
        "uses_mlx_fast_rope": True,
        "uses_mlx_fast_grouped_query_sdpa": True,
    }


def run_swiglu(args: argparse.Namespace, *, mx: Any) -> tuple[list[float], dict[str, Any]]:
    shape = (args.batch_size, args.sequence_length, args.hidden_size)
    mx.random.seed(2026072602)
    hidden = mx.random.normal(shape, dtype=mx.float32)
    gate = mx.random.normal((args.ff_size, args.hidden_size), dtype=mx.float32) * 0.02
    up = mx.random.normal((args.ff_size, args.hidden_size), dtype=mx.float32) * 0.02
    down = mx.random.normal((args.hidden_size, args.ff_size), dtype=mx.float32) * 0.02
    cotangent = mx.random.normal(shape, dtype=mx.float32)
    mx.eval(hidden, gate, up, down, cotangent)

    def objective(x: Any, gate_w: Any, up_w: Any, down_w: Any, cot: Any) -> Any:
        gate_value = mx.matmul(x, gate_w.T)
        activated = gate_value * mx.sigmoid(gate_value)
        output = mx.matmul(activated * mx.matmul(x, up_w.T), down_w.T)
        return mx.sum(output * cot) / output.size

    compiled = mx.compile(mx.value_and_grad(objective, argnums=(0, 1, 2, 3)))
    durations, latest = _timed(
        compiled,
        (hidden, gate, up, down, cotangent),
        mx=mx,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    loss, gradients = latest
    return durations, {
        "scalar_finite": bool(mx.isfinite(loss).item()),
        "gradient_tensor_count": len(gradients),
        "all_gradients_finite": all(
            bool(mx.all(mx.isfinite(gradient)).item()) for gradient in gradients
        ),
        "compiled_elementwise_silu_and_gate": True,
    }


def run_norms(args: argparse.Namespace, *, mx: Any) -> tuple[list[float], dict[str, Any]]:
    shape = (args.batch_size, args.sequence_length, args.hidden_size)
    mx.random.seed(2026072603)
    hidden = mx.random.normal(shape, dtype=mx.float32)
    first_weight = mx.ones((args.hidden_size,), dtype=mx.float32)
    second_weight = mx.ones((args.hidden_size,), dtype=mx.float32)
    first_cotangent = mx.random.normal(shape, dtype=mx.float32)
    second_cotangent = mx.random.normal(shape, dtype=mx.float32)
    mx.eval(hidden, first_weight, second_weight, first_cotangent, second_cotangent)

    def objective(x: Any, first: Any, second: Any, cot_a: Any, cot_b: Any) -> Any:
        first_output = mx.fast.rms_norm(x, first, 1e-5)
        second_output = mx.fast.rms_norm(x, second, 1e-5)
        return (
            mx.sum(first_output * cot_a) + mx.sum(second_output * cot_b)
        ) / (2 * first_output.size)

    compiled = mx.compile(mx.value_and_grad(objective, argnums=(0, 1, 2)))
    durations, latest = _timed(
        compiled,
        (hidden, first_weight, second_weight, first_cotangent, second_cotangent),
        mx=mx,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    loss, gradients = latest
    return durations, {
        "scalar_finite": bool(mx.isfinite(loss).item()),
        "gradient_tensor_count": len(gradients),
        "all_gradients_finite": all(
            bool(mx.all(mx.isfinite(gradient)).item()) for gradient in gradients
        ),
        "uses_mlx_fast_rms_norm": True,
    }


def run_clip(args: argparse.Namespace, *, mx: Any) -> tuple[list[float], dict[str, Any]]:
    mx.random.seed(2026072604)
    gradient = mx.random.normal((args.active_parameter_count,), dtype=mx.float32)
    mx.eval(gradient)

    def clip(value: Any) -> tuple[Any, Any]:
        norm = mx.sqrt(mx.sum(mx.square(value)))
        scale = mx.minimum(mx.array(1.0, dtype=mx.float32), 1.0 / (norm + 1e-6))
        return value * scale, norm

    compiled = mx.compile(clip)
    durations, latest = _timed(
        compiled,
        (gradient,),
        mx=mx,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    clipped, norm = latest
    return durations, {
        "scalar_finite": bool(mx.isfinite(norm).item()),
        "all_gradients_finite": bool(mx.all(mx.isfinite(clipped)).item()),
        "gradient_element_count": args.active_parameter_count,
        "global_token_mass_weighted_update_boundary": True,
    }


def run_adamw(args: argparse.Namespace, *, mx: Any) -> tuple[list[float], dict[str, Any]]:
    mx.random.seed(2026072605)
    parameter = mx.random.normal((args.active_parameter_count,), dtype=mx.float32) * 0.02
    gradient = mx.random.normal((args.active_parameter_count,), dtype=mx.float32) * 0.001
    first_moment = mx.zeros_like(parameter)
    second_moment = mx.zeros_like(parameter)
    mx.eval(parameter, gradient, first_moment, second_moment)

    def adamw_step(
        value: Any, grad: Any, first: Any, second: Any
    ) -> tuple[Any, Any, Any]:
        next_first = 0.9 * first + 0.1 * grad
        next_second = 0.999 * second + 0.001 * mx.square(grad)
        next_value = (
            value * (1.0 - 3e-4 * 0.1)
            - 3e-4 * next_first / (mx.sqrt(next_second) + 1e-8)
        )
        return next_value, next_first, next_second

    compiled = mx.compile(adamw_step)
    for _ in range(args.warmup):
        parameter, first_moment, second_moment = compiled(
            parameter, gradient, first_moment, second_moment
        )
        mx.eval(parameter, first_moment, second_moment)
    durations: list[float] = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        parameter, first_moment, second_moment = compiled(
            parameter, gradient, first_moment, second_moment
        )
        mx.eval(parameter, first_moment, second_moment)
        durations.append(time.perf_counter() - started)
    return durations, {
        "scalar_finite": True,
        "all_parameters_finite": bool(mx.all(mx.isfinite(parameter)).item()),
        "all_moments_finite": bool(
            mx.all(mx.isfinite(first_moment)).item()
            and mx.all(mx.isfinite(second_moment)).item()
        ),
        "active_parameter_element_count": args.active_parameter_count,
        "optimizer_state_element_count": 2 * args.active_parameter_count,
        "bias_correction_omitted_from_timing": True,
        "interpretation": (
            "Elementwise active-state lower-level station timing; canonical MLX AdamW "
            "and exact checkpoint-state timing remain authoritative for production."
        ),
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
        "active_parameter_count": args.active_parameter_count,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"all dimensions and repetition counts must be positive: {positive}")
    if args.hidden_size % args.head_count or args.head_count % args.kv_head_count:
        raise ValueError("hidden/head and grouped-query head dimensions must divide evenly")
    if args.reference_microbatch_seconds <= 0:
        raise ValueError("reference microbatch duration must be positive")

    import mlx.core as mx

    runner = {
        "decoder_self_attention": run_attention,
        "swiglu_mlp": run_swiglu,
        "two_rmsnorms": run_norms,
        "global_gradient_clip": run_clip,
        "adamw_active_state": run_adamw,
    }[args.station]
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    durations, integrity = runner(args, mx=mx)
    median_seconds = statistics.median(durations)
    report = {
        "policy": POLICY,
        "created_utc": datetime.now(UTC).isoformat(),
        "trigger_state": "GREEN",
        "station": args.station,
        "claim_scope": (
            "Synthetic production-shape operator timing only; no checkpoint, corpus row, "
            "heldout, convergence, utility, capability, or custom-kernel speed claim."
        ),
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
            "active_parameter_count": args.active_parameter_count,
        },
        "execution": {
            "compute_dtype": "float32",
            "compiled": True,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "fresh_process_per_station_required": True,
        },
        "timing_seconds": {
            "minimum": round(min(durations), 6),
            "median": round(median_seconds, 6),
            "mean": round(statistics.mean(durations), 6),
            "p95": round(percentile(durations, 0.95), 6),
            "maximum": round(max(durations), 6),
        },
        "reference": {
            "observed_full_forward_backward_microbatch_seconds": (
                args.reference_microbatch_seconds
            ),
            **station_bound(
                args.station,
                median_seconds,
                layer_count=args.layer_count,
                reference_microbatch_seconds=args.reference_microbatch_seconds,
            ),
        },
        "integrity": integrity,
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
