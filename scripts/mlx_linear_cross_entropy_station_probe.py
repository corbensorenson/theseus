#!/usr/bin/env python3
"""Bounded synthetic timing for Theseus's vocabulary projection/loss station.

This probe intentionally does not load a checkpoint, training rows, or an
evaluation surface.  It measures the exact dense operation shape used by the
plain 57M pretraining microbatch so a custom fused-loss proposal can be ranked
before any production implementation is attempted.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=8195)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--loss-compute-dtype",
        choices=("model", "float32"),
        default="float32",
    )
    parser.add_argument(
        "--reference-microbatch-seconds",
        type=float,
        default=0.56,
        help="Observed full forward/backward microbatch time used only for an upper-bound ratio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    positive = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"all shape and repetition arguments must be positive: {positive}")
    if args.reference_microbatch_seconds <= 0:
        raise ValueError("reference microbatch duration must be positive")

    import mlx.core as mx
    import mlx.nn as nn

    compute_dtype = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[args.compute_dtype]
    mx.random.seed(20260726)
    hidden = mx.random.normal(
        (args.batch_size, args.sequence_length, args.hidden_size),
        dtype=compute_dtype,
    )
    classifier = mx.random.normal(
        (args.vocab_size, args.hidden_size),
        dtype=compute_dtype,
    ) * 0.02
    labels = mx.random.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.sequence_length),
        dtype=mx.int32,
    )
    mx.eval(hidden, classifier, labels)

    def dense_linear_cross_entropy(
        hidden_state: Any, classifier_weight: Any, target: Any
    ) -> Any:
        logits = mx.matmul(hidden_state, classifier_weight.T)
        if args.loss_compute_dtype == "float32":
            logits = logits.astype(mx.float32)
        return mx.mean(nn.losses.cross_entropy(logits, target))

    value_and_grad = mx.value_and_grad(
        dense_linear_cross_entropy,
        argnums=(0, 1),
    )
    compiled_value_and_grad = mx.compile(value_and_grad)

    for _ in range(args.warmup):
        loss, grads = compiled_value_and_grad(hidden, classifier, labels)
        mx.eval(loss, grads)

    durations: list[float] = []
    losses: list[float] = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        loss, grads = compiled_value_and_grad(hidden, classifier, labels)
        mx.eval(loss, grads)
        durations.append(time.perf_counter() - started)
        losses.append(float(loss.item()))

    median_seconds = statistics.median(durations)
    report = {
        "policy": "project_theseus_mlx_linear_cross_entropy_station_probe_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "claim_scope": (
            "Synthetic exact-shape station timing only; no checkpoint, training row, "
            "quality, convergence, or capability claim."
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
            "vocab_size": args.vocab_size,
            "supervised_token_count": args.batch_size * args.sequence_length,
            "materialized_logit_element_count": (
                args.batch_size * args.sequence_length * args.vocab_size
            ),
            "materialized_logit_bytes_float32": (
                args.batch_size
                * args.sequence_length
                * args.vocab_size
                * 4
            ),
        },
        "execution": {
            "compute_dtype": args.compute_dtype,
            "loss_compute_dtype": args.loss_compute_dtype,
            "compiled_outer_forward_and_vjp": True,
            "gradient_targets": ["hidden_state", "tied_classifier_weight"],
            "warmup": args.warmup,
            "repetitions": args.repetitions,
        },
        "timing_seconds": {
            "minimum": round(min(durations), 6),
            "median": round(median_seconds, 6),
            "mean": round(statistics.mean(durations), 6),
            "p95": round(percentile(durations, 0.95), 6),
            "maximum": round(max(durations), 6),
        },
        "reference": {
            "observed_full_microbatch_seconds": args.reference_microbatch_seconds,
            "station_fraction_of_full_microbatch_at_median": round(
                median_seconds / args.reference_microbatch_seconds, 6
            ),
            "interpretation": (
                "This ratio is an elimination upper bound, not a predicted fused-kernel "
                "speedup: any exact implementation still performs classifier dot products, "
                "log-sum-exp, and both gradients."
            ),
        },
        "finite": {
            "all_losses_finite": all(value == value for value in losses),
            "loss_minimum": round(min(losses), 8),
            "loss_maximum": round(max(losses), 8),
        },
        "canonical_checkpoint_mutated": False,
        "public_benchmark_rows_read": 0,
        "external_inference_calls": 0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
