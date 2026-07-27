#!/usr/bin/env python3
"""Exact-shape MLX control for the Core ML state-weight projection probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


POLICY = "project_theseus_mlx_fp16_projection_control_v1"
ROWS = 4 * 512
INPUT_CHANNELS = 512
OUTPUT_CHANNELS = 768


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, int(len(ordered) * fraction)),
    )
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=64)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.repetitions < 2:
        raise SystemExit("warmup must be positive and repetitions at least two")

    import mlx.core as mx

    rng = np.random.default_rng(20260727)
    host_x = (
        rng.standard_normal((ROWS, INPUT_CHANNELS), dtype=np.float32) * 0.125
    ).astype(np.float16)
    host_weight = (
        rng.standard_normal(
            (INPUT_CHANNELS, OUTPUT_CHANNELS), dtype=np.float32
        )
        * 0.03125
    ).astype(np.float16)
    x = mx.array(host_x)
    weight = mx.array(host_weight)
    eager = x @ weight
    mx.eval(eager)
    compiled_projection = mx.compile(lambda left, right: left @ right)
    for _ in range(args.warmup):
        mx.eval(compiled_projection(x, weight))

    durations: list[float] = []
    output = None
    for _ in range(args.repetitions):
        started = time.perf_counter()
        output = compiled_projection(x, weight)
        mx.eval(output)
        durations.append(time.perf_counter() - started)
    if output is None:
        raise RuntimeError("compiled MLX projection produced no output")
    delta = np.abs(
        np.asarray(output, dtype=np.float16).astype(np.float32)
        - np.asarray(eager, dtype=np.float16).astype(np.float32)
    )
    mismatch_count = int(np.count_nonzero(delta > 0.001))
    mean_seconds = statistics.fmean(durations)
    report = {
        "policy": POLICY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_state": "GREEN" if mismatch_count == 0 else "RED",
        "claim_scope": (
            "Compiled MLX FP16 projection operator control only; no state "
            "update, backward, optimizer, transformer, or speedup claim."
        ),
        "hardware": {
            "machine": platform.machine(),
            "mlx_version": importlib.metadata.version("mlx"),
            "device": str(mx.default_device()),
            "device_info": dict(mx.device_info()),
        },
        "shape": {
            "logical_batch": 4,
            "sequence": 512,
            "rows": ROWS,
            "input_channels": INPUT_CHANNELS,
            "output_channels": OUTPUT_CHANNELS,
        },
        "runtime": {
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "mean_milliseconds": round(mean_seconds * 1000.0, 6),
            "median_milliseconds": round(
                statistics.median(durations) * 1000.0, 6
            ),
            "p95_milliseconds": round(
                percentile(durations, 0.95) * 1000.0, 6
            ),
            "rows_per_second": round(ROWS / mean_seconds, 6),
            "maximum_absolute_delta_vs_eager": float(
                delta.max(initial=0.0)
            ),
            "mismatch_count_vs_eager": mismatch_count,
        },
        "mlx_memory": {
            "active_bytes": int(mx.get_active_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
            "peak_bytes": int(mx.get_peak_memory()),
        },
        "resource_custody": {
            "public_benchmark_rows_read": 0,
            "external_inference_calls": 0,
            "canonical_checkpoint_mutated": False,
        },
    }
    if args.out is not None:
        atomic_json(args.out, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if mismatch_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
