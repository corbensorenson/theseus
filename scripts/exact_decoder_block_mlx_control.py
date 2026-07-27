#!/usr/bin/env python3
"""Matched compiled-MLX control for the exact native decoder-block join."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


POLICY = "project_theseus_exact_decoder_block_mlx_control_v1"
SEQUENCE = 128
DIM = 512
FF_DIM = 1536
QUERY_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 64
QUERY_GROUPS = QUERY_HEADS // KV_HEADS
AUTHORITY_MASS = float((SEQUENCE - 3) * DIM)
PARAMETER_COUNT = 3_015_680


class ControlFault(ValueError):
    pass


def half_round(value: np.ndarray) -> np.ndarray:
    return value.astype(np.float16).astype(np.float32)


def frozen_values() -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    positions = np.arange(SEQUENCE, dtype=np.float32)[:, None]
    channels = np.arange(DIM, dtype=np.float32)[None, :]
    hidden = (
        np.sin((positions * 17 + channels * 3) * 0.013) * 0.125
        + np.cos((positions * 5 + channels * 11) * 0.007) * 0.03125
    )
    hidden = half_round(hidden.astype(np.float32))
    norm1 = half_round(
        (1.0 + np.sin(np.arange(DIM, dtype=np.float32) * 0.019) * 0.01)
    )
    input_channels = np.arange(DIM, dtype=np.float32)[:, None]
    query_channels = np.arange(DIM, dtype=np.float32)[None, :]
    kv_channels = np.arange(KV_HEADS * HEAD_DIM, dtype=np.float32)[None, :]
    wq = half_round(
        np.sin((input_channels * 29 + query_channels * 7) * 0.003) * 0.03125
    )
    wk = half_round(
        np.cos((input_channels * 13 + kv_channels * 5) * 0.005) * 0.03125
    )
    wv = half_round(
        np.sin((input_channels * 23 + kv_channels * 3) * 0.004) * 0.03125
    )
    remainder = (
        np.sin((np.arange(2_621_952, dtype=np.int64) % 887) * 0.005) * 0.02
    ).astype(np.float32)
    out_end = DIM * DIM
    norm2_end = out_end + DIM
    gate_end = norm2_end + DIM * FF_DIM
    up_end = gate_end + DIM * FF_DIM
    out_weight = remainder[:out_end].reshape(DIM, DIM).copy()
    norm2 = (
        1.0 + np.sin(np.arange(DIM, dtype=np.float32) * 0.019) * 0.01
    ).astype(np.float32)
    gate = remainder[norm2_end:gate_end].reshape(DIM, FF_DIM).copy()
    up = remainder[gate_end:up_end].reshape(DIM, FF_DIM).copy()
    down = remainder[up_end:].reshape(FF_DIM, DIM).copy()
    flat_index = np.arange(SEQUENCE * DIM, dtype=np.int64)
    target = (
        np.sin((flat_index % 953).astype(np.float32) * 0.007) * 0.125
    ).reshape(SEQUENCE, DIM)
    mask = np.ones((SEQUENCE,), dtype=np.float32)
    mask[-3:] = 0.0
    parameters = [
        norm1, wq, wk, wv,
        out_weight, norm2, gate, up, down,
    ]
    if sum(value.size for value in parameters) != PARAMETER_COUNT:
        raise ControlFault("parameter_count_mismatch")
    return parameters, hidden, target.astype(np.float32), mask


def run_control(steps: int) -> dict[str, Any]:
    import mlx.core as mx

    parameters_np, hidden_np, target_np, mask_np = frozen_values()
    parameters = tuple(mx.array(value, dtype=mx.float32) for value in parameters_np)
    hidden = mx.array(hidden_np, dtype=mx.float32)
    target = mx.array(target_np, dtype=mx.float32)
    mask = mx.array(mask_np, dtype=mx.float32)
    half = mx.float16
    full = mx.float32
    half_dim = HEAD_DIM // 2
    inverse = 1.0 / (
        10000.0
        ** (np.arange(half_dim, dtype=np.float32) / float(half_dim))
    )
    angles = np.arange(SEQUENCE, dtype=np.float32)[:, None] * inverse[None, :]
    doubled = np.concatenate((angles, angles), axis=-1)
    rope_cos = mx.array(np.cos(doubled)[None, None], dtype=half)
    rope_sin = mx.array(np.sin(doubled)[None, None], dtype=half)
    causal = mx.triu(
        mx.full((SEQUENCE, SEQUENCE), -10000.0, dtype=half), k=1
    )

    def rms(x: Any, scale: Any, dtype: Any) -> Any:
        value = x.astype(dtype)
        weight = scale.astype(dtype)
        variance = mx.mean(value * value, axis=-1, keepdims=True)
        return value * mx.rsqrt(
            variance + mx.array(1.0e-5, dtype=dtype)
        ) * weight

    def rope(value: Any) -> Any:
        rotated = mx.concatenate(
            (-value[..., half_dim:], value[..., :half_dim]), axis=-1
        )
        return value * rope_cos + rotated * rope_sin

    def objective(params: tuple[Any, ...]) -> Any:
        (
            norm1, wq, wk, wv, out_weight,
            norm2, gate_weight, up_weight, down_weight,
        ) = params
        normalized1 = rms(hidden, norm1, half)
        query = mx.matmul(normalized1, wq.astype(half))
        key = mx.matmul(normalized1, wk.astype(half))
        value = mx.matmul(normalized1, wv.astype(half))
        query = rope(
            query.reshape(1, SEQUENCE, QUERY_HEADS, HEAD_DIM).transpose(
                0, 2, 1, 3
            )
        )
        key = rope(
            key.reshape(1, SEQUENCE, KV_HEADS, HEAD_DIM).transpose(
                0, 2, 1, 3
            )
        )
        value = value.reshape(1, SEQUENCE, KV_HEADS, HEAD_DIM).transpose(
            0, 2, 1, 3
        )
        grouped = query.reshape(
            1, KV_HEADS, QUERY_GROUPS, SEQUENCE, HEAD_DIM
        )
        scores = mx.matmul(
            grouped, mx.swapaxes(key[:, :, None], -1, -2)
        )
        scores = scores * mx.array(HEAD_DIM**-0.5, dtype=half)
        probabilities = mx.softmax(scores + causal, axis=-1)
        attended = mx.matmul(probabilities, value[:, :, None])
        attended = attended.reshape(1, QUERY_HEADS, SEQUENCE, HEAD_DIM)
        attended = attended.transpose(0, 2, 1, 3).reshape(
            SEQUENCE, DIM
        ).astype(full)
        after_attention = hidden + mx.matmul(attended, out_weight)
        normalized2 = rms(after_attention, norm2, full)
        gate = mx.matmul(normalized2, gate_weight)
        up = mx.matmul(normalized2, up_weight)
        activated = gate * mx.sigmoid(gate) * up
        output = after_attention + mx.matmul(activated, down_weight)
        difference = output - target
        return (
            mx.array(0.5, dtype=full)
            * mx.sum(mask[:, None] * difference * difference)
            / mx.array(AUTHORITY_MASS, dtype=full)
        )

    value_and_grad = mx.value_and_grad(objective)

    def update(
        params: tuple[Any, ...],
        first: tuple[Any, ...],
        second: tuple[Any, ...],
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], Any, Any]:
        loss, gradients = value_and_grad(params)
        square = mx.array(0.0, dtype=full)
        for gradient in gradients:
            square = square + mx.sum(gradient.astype(full) ** 2)
        norm = mx.sqrt(square)
        scale = mx.minimum(
            mx.array(1.0, dtype=full),
            mx.array(1.0, dtype=full)
            / mx.maximum(norm, mx.array(1.0e-12, dtype=full)),
        )
        next_params = []
        next_first = []
        next_second = []
        for parameter, m, v, gradient in zip(
            params, first, second, gradients, strict=True
        ):
            clipped = gradient.astype(full) * scale
            new_m = mx.array(0.9, dtype=full) * m + mx.array(
                0.1, dtype=full
            ) * clipped
            new_v = mx.array(0.999, dtype=full) * v + mx.array(
                0.001, dtype=full
            ) * clipped * clipped
            decayed = parameter * mx.array(1.0 - 3.0e-7, dtype=full)
            new_parameter = decayed - mx.array(3.0e-5, dtype=full) * new_m / (
                mx.sqrt(new_v) + mx.array(1.0e-8, dtype=full)
            )
            next_params.append(new_parameter)
            next_first.append(new_m)
            next_second.append(new_v)
        return (
            tuple(next_params),
            tuple(next_first),
            tuple(next_second),
            loss,
            norm,
        )

    compiled = mx.compile(update)
    zeros = tuple(mx.zeros_like(parameter) for parameter in parameters)
    warm = compiled(parameters, zeros, zeros)
    mx.eval(warm)

    def one() -> tuple[Any, ...]:
        result = compiled(parameters, zeros, zeros)
        mx.eval(result)
        return result

    first_started = time.perf_counter()
    first = one()
    first_ms = (time.perf_counter() - first_started) * 1000.0
    replay = one()
    first_digests = [
        hashlib.sha256(
            np.ascontiguousarray(np.asarray(value)).tobytes()
        ).hexdigest()
        for tree in first[:3]
        for value in tree
    ]
    replay_digests = [
        hashlib.sha256(
            np.ascontiguousarray(np.asarray(value)).tobytes()
        ).hexdigest()
        for tree in replay[:3]
        for value in tree
    ]
    replay_exact = (
        first_digests == replay_digests
        and float(first[3].item()) == float(replay[3].item())
        and float(first[4].item()) == float(replay[4].item())
    )
    current_params, current_first, current_second = parameters, zeros, zeros
    losses: list[float] = []
    norms: list[float] = []
    timings: list[float] = []
    finite = True
    for _ in range(steps):
        started = time.perf_counter()
        result = compiled(current_params, current_first, current_second)
        mx.eval(result)
        timings.append((time.perf_counter() - started) * 1000.0)
        current_params, current_first, current_second, loss, norm = result
        losses.append(float(loss.item()))
        norms.append(float(norm.item()))
        finite = finite and math.isfinite(losses[-1]) and math.isfinite(norms[-1])
    return {
        "policy": POLICY,
        "state": (
            "GREEN_MATCHED_COMPILED_MLX_CONTROL"
            if replay_exact and finite
            else "RED_MATCHED_COMPILED_MLX_CONTROL"
        ),
        "shape": {
            "batch": 1,
            "sequence": SEQUENCE,
            "d_model": DIM,
            "ff_dim": FF_DIM,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
        },
        "precision": {
            "attention_compute": "float16",
            "remainder_compute": "float32",
            "master_parameters": "float32",
            "optimizer_state": "float32",
        },
        "parameter_elements": PARAMETER_COUNT,
        "parameter_leaf_count": 9,
        "objective_authority_mass": AUTHORITY_MASS,
        "timing": {
            "first_milliseconds": first_ms,
            "mean_milliseconds": float(np.mean(timings)),
            "median_milliseconds": float(np.median(timings)),
            "minimum_milliseconds": float(np.min(timings)),
            "maximum_milliseconds": float(np.max(timings)),
            "steps": steps,
        },
        "first_loss": float(first[3].item()),
        "first_gradient_norm": float(first[4].item()),
        "final_loss": losses[-1],
        "final_gradient_norm": norms[-1],
        "gates": {
            "matched_shape": True,
            "matched_precision_split": True,
            "one_objective_mass_normalization": True,
            "one_global_norm_and_clip": True,
            "one_fp32_adamw_publication": True,
            "replay_exact": replay_exact,
            "sixty_four_step_finite": finite and steps >= 64,
            "initialization_numpy_only": True,
            "numpy_tensor_round_trip_in_timed_step": False,
            "production_eligible": False,
        },
        "capability_claim": "NONE_ENGINEERING_MLX_CONTROL_ONLY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 64:
        raise ControlFault("steps_must_cover_stability_gate")
    report = run_control(args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["state"].startswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
