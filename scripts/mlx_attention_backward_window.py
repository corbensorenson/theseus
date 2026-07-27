#!/usr/bin/env python3
"""Exact production-shape MLX attention backward window for ANE overlap.

The worker is synthetic and source-free.  It compiles the same GQA/RoPE/SDPA
forward/VJP station used by the bounded operator inventory, warms it, optionally
joins a filesystem barrier, and reports only the synchronized work interval.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any


POLICY = "project_theseus_mlx_attention_backward_window_v1"


def wait_for_go(ready_file: Path | None, go_file: Path | None) -> None:
    if ready_file is None and go_file is None:
        return
    if ready_file is None or go_file is None:
        raise ValueError("ready and go files must be supplied together")
    ready_file.write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + 30.0
    while not go_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("barrier_go_timeout")
        time.sleep(0.001)


def make_station(mx: Any, *, batch: int = 4, length: int = 512) -> tuple[Any, tuple[Any, ...]]:
    hidden_size = 512
    heads = 8
    kv_heads = 2
    head_dim = hidden_size // heads
    kv_width = kv_heads * head_dim
    shape = (batch, length, hidden_size)
    mx.random.seed(2026072717)
    hidden = mx.random.normal(shape, dtype=mx.float32)
    q_weight = mx.random.normal((hidden_size, hidden_size), dtype=mx.float32) * 0.02
    k_weight = mx.random.normal((kv_width, hidden_size), dtype=mx.float32) * 0.02
    v_weight = mx.random.normal((kv_width, hidden_size), dtype=mx.float32) * 0.02
    out_weight = mx.random.normal((hidden_size, hidden_size), dtype=mx.float32) * 0.02
    cotangent = mx.random.normal(shape, dtype=mx.float32)
    mx.eval(hidden, q_weight, k_weight, v_weight, out_weight, cotangent)

    def objective(
        x: Any, qw: Any, kw: Any, vw: Any, ow: Any, cot: Any
    ) -> Any:
        query = mx.matmul(x, qw.T).reshape(
            batch, length, heads, head_dim
        ).transpose(0, 2, 1, 3)
        key = mx.matmul(x, kw.T).reshape(
            batch, length, kv_heads, head_dim
        ).transpose(0, 2, 1, 3)
        value = mx.matmul(x, vw.T).reshape(
            batch, length, kv_heads, head_dim
        ).transpose(0, 2, 1, 3)
        query = mx.fast.rope(
            query, head_dim, traditional=False, base=10000.0, scale=1.0, offset=0
        )
        key = mx.fast.rope(
            key, head_dim, traditional=False, base=10000.0, scale=1.0, offset=0
        )
        attended = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=head_dim**-0.5, mask="causal"
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch, length, hidden_size
        )
        output = mx.matmul(attended, ow.T)
        return mx.sum(output * cot) / output.size

    compiled = mx.compile(
        mx.value_and_grad(objective, argnums=(0, 1, 2, 3, 4))
    )
    return compiled, (
        hidden,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        cotangent,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=24)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--go-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.repetitions < 2:
        raise ValueError("warmup >= 1 and repetitions >= 2 are required")
    import mlx.core as mx

    station, station_args = make_station(mx)
    for _ in range(args.warmup):
        latest = station(*station_args)
        mx.eval(latest)
    wait_for_go(args.ready_file, args.go_file)
    durations: list[float] = []
    latest = None
    barrier_started = time.perf_counter()
    for _ in range(args.repetitions):
        started = time.perf_counter()
        latest = station(*station_args)
        mx.eval(latest)
        durations.append((time.perf_counter() - started) * 1000.0)
    barrier_work_milliseconds = (time.perf_counter() - barrier_started) * 1000.0
    loss, gradients = latest
    report = {
        "policy": POLICY,
        "trigger_state": (
            "GREEN_ATTENTION_BACKWARD_WINDOW"
            if bool(mx.isfinite(loss).item())
            and all(
                bool(mx.all(mx.isfinite(gradient)).item())
                for gradient in gradients
            )
            else "RED_NONFINITE"
        ),
        "shape": {
            "batch": 4,
            "sequence": 512,
            "hidden": 512,
            "heads": 8,
            "kv_heads": 2,
        },
        "runtime": {
            "repetitions": args.repetitions,
            "mean_milliseconds": statistics.fmean(durations),
            "median_milliseconds": statistics.median(durations),
            "minimum_milliseconds": min(durations),
            "maximum_milliseconds": max(durations),
            "barrier_work_milliseconds": barrier_work_milliseconds,
        },
        "mechanics": {
            "compiled_value_and_grad": True,
            "mlx_fast_rope": True,
            "mlx_fast_grouped_query_sdpa": True,
            "loss_finite": bool(mx.isfinite(loss).item()),
            "all_gradients_finite": all(
                bool(mx.all(mx.isfinite(gradient)).item())
                for gradient in gradients
            ),
        },
        "canonical_checkpoint_mutated": False,
        "public_benchmark_rows_read": 0,
        "external_inference_calls": 0,
        "claim_scope": (
            "One exact production-shape self-attention forward/VJP window. "
            "It is independent overlap work, not a full decoder or optimizer step."
        ),
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["trigger_state"].startswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
