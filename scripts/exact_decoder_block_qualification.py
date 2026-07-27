#!/usr/bin/env python3
"""Freeze and qualify the exact Project Theseus decoder-block numerical ABI.

This gate deliberately precedes the native ANE port.  It compares independent
MLX and PyTorch implementations at the production feature geometry, including
the scalar loss, input gradient, and all nine parameter gradients.  It does not
claim that a native ANE block exists or that a full Theseus step is accelerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/exact_decoder_block_qualification.json"
POLICY = "project_theseus_exact_decoder_block_qualification_v1"


class QualificationFault(ValueError):
    """Raised when the frozen ABI or an implementation violates authority."""


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy") != POLICY:
        raise QualificationFault("unexpected_policy")
    validate_parameter_schema(config)
    return config


def canonical_parameter_schema(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    shape = config["shape"]
    d_model = int(shape["d_model"])
    ff_dim = int(shape["ff_dim"])
    query_width = int(shape["query_heads"]) * int(shape["head_dim"])
    kv_width = int(shape["kv_heads"]) * int(shape["head_dim"])
    return {
        "attention_norm.weight": (d_model,),
        "attention.q_proj.weight": (query_width, d_model),
        "attention.k_proj.weight": (kv_width, d_model),
        "attention.v_proj.weight": (kv_width, d_model),
        "attention.out_proj.weight": (d_model, query_width),
        "ffn_norm.weight": (d_model,),
        "feed_forward.gate.weight": (ff_dim, d_model),
        "feed_forward.up.weight": (ff_dim, d_model),
        "feed_forward.down.weight": (d_model, ff_dim),
    }


def validate_parameter_schema(config: dict[str, Any]) -> None:
    shape = config["shape"]
    query_heads = int(shape["query_heads"])
    kv_heads = int(shape["kv_heads"])
    head_dim = int(shape["head_dim"])
    d_model = int(shape["d_model"])
    if query_heads % kv_heads:
        raise QualificationFault("query_heads_not_divisible_by_kv_heads")
    if query_heads * head_dim != d_model:
        raise QualificationFault("query_width_must_equal_d_model")
    expected = canonical_parameter_schema(config)
    recorded = {
        name: tuple(int(value) for value in dimensions)
        for name, dimensions in config["parameter_schema"].items()
    }
    if recorded != expected:
        raise QualificationFault("parameter_schema_mismatch")


def parameter_count(config: dict[str, Any]) -> int:
    return sum(math.prod(shape) for shape in canonical_parameter_schema(config).values())


def contiguous_gqa_owner(query_head: int, query_heads: int, kv_heads: int) -> int:
    if query_heads % kv_heads:
        raise QualificationFault("query_heads_not_divisible_by_kv_heads")
    if not 0 <= query_head < query_heads:
        raise QualificationFault("query_head_out_of_range")
    return query_head // (query_heads // kv_heads)


def split_half_rotate_numpy(value: np.ndarray) -> np.ndarray:
    if value.shape[-1] % 2:
        raise QualificationFault("rope_head_dim_must_be_even")
    half = value.shape[-1] // 2
    return np.concatenate((-value[..., half:], value[..., :half]), axis=-1)


def frozen_inputs(config: dict[str, Any]) -> dict[str, Any]:
    shape = config["shape"]
    batch = int(shape["batch"])
    sequence = int(shape["sequence"])
    d_model = int(shape["d_model"])
    rng = np.random.default_rng(2026072723)
    parameters: dict[str, np.ndarray] = {}
    for name, dimensions in canonical_parameter_schema(config).items():
        if name.endswith("_norm.weight") or name.endswith("norm.weight"):
            value = 1.0 + rng.standard_normal(dimensions) * 0.01
        else:
            value = rng.standard_normal(dimensions) * (1.0 / math.sqrt(dimensions[-1]))
        parameters[name] = value.astype(np.float32)
    mask = np.ones((batch, sequence), dtype=np.float32)
    if sequence >= 8:
        mask[:, -3:] = 0.0
    return {
        "parameters": parameters,
        "hidden": (rng.standard_normal((batch, sequence, d_model)) * 0.125).astype(
            np.float32
        ),
        "target": (rng.standard_normal((batch, sequence, d_model)) * 0.125).astype(
            np.float32
        ),
        "loss_mask": mask,
    }


def _rope_angles_numpy(sequence: int, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    inverse = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float32) / half))
    angles = np.arange(sequence, dtype=np.float32)[:, None] * inverse[None, :]
    doubled = np.concatenate((angles, angles), axis=-1)
    return np.cos(doubled)[None, None], np.sin(doubled)[None, None]


def _nested_parameters(
    parameters: dict[str, Any], leaf: Callable[[Any], Any]
) -> dict[str, Any]:
    return {
        "attention_norm": {"weight": leaf(parameters["attention_norm.weight"])},
        "attention": {
            "q_proj": {"weight": leaf(parameters["attention.q_proj.weight"])},
            "k_proj": {"weight": leaf(parameters["attention.k_proj.weight"])},
            "v_proj": {"weight": leaf(parameters["attention.v_proj.weight"])},
            "out_proj": {"weight": leaf(parameters["attention.out_proj.weight"])},
        },
        "ffn_norm": {"weight": leaf(parameters["ffn_norm.weight"])},
        "feed_forward": {
            "gate": {"weight": leaf(parameters["feed_forward.gate.weight"])},
            "up": {"weight": leaf(parameters["feed_forward.up.weight"])},
            "down": {"weight": leaf(parameters["feed_forward.down.weight"])},
        },
    }


def _flatten_tree(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_tree(value, name))
        else:
            flattened[name] = value
    return flattened


def torch_reference(
    config: dict[str, Any], values: dict[str, Any]
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    shape = config["shape"]
    batch = int(shape["batch"])
    sequence = int(shape["sequence"])
    d_model = int(shape["d_model"])
    ff_dim = int(shape["ff_dim"])
    query_heads = int(shape["query_heads"])
    kv_heads = int(shape["kv_heads"])
    head_dim = int(shape["head_dim"])
    query_groups = query_heads // kv_heads
    epsilon = float(config["semantics"]["rms_norm_epsilon"])

    params = _nested_parameters(
        values["parameters"],
        lambda array: torch.tensor(array, dtype=torch.float32, requires_grad=True),
    )
    hidden = torch.tensor(values["hidden"], dtype=torch.float32, requires_grad=True)
    target = torch.tensor(values["target"], dtype=torch.float32)
    loss_mask = torch.tensor(values["loss_mask"], dtype=torch.float32)

    def linear(x: Any, weight: Any) -> Any:
        return torch.matmul(x, weight.transpose(-1, -2))

    def rms_norm(x: Any, weight: Any) -> Any:
        variance = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + epsilon) * weight

    def rope(x: Any) -> Any:
        cos_np, sin_np = _rope_angles_numpy(sequence, head_dim)
        cos = torch.tensor(cos_np, dtype=torch.float32)
        sin = torch.tensor(sin_np, dtype=torch.float32)
        half = head_dim // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rotated * sin

    attention_input = rms_norm(hidden, params["attention_norm"]["weight"])
    query = linear(attention_input, params["attention"]["q_proj"]["weight"])
    key = linear(attention_input, params["attention"]["k_proj"]["weight"])
    value = linear(attention_input, params["attention"]["v_proj"]["weight"])
    query = rope(
        query.reshape(batch, sequence, query_heads, head_dim).transpose(1, 2)
    )
    key = rope(key.reshape(batch, sequence, kv_heads, head_dim).transpose(1, 2))
    value = value.reshape(batch, sequence, kv_heads, head_dim).transpose(1, 2)
    grouped_query = query.reshape(
        batch, kv_heads, query_groups, sequence, head_dim
    )
    scores = torch.matmul(grouped_query, key[:, :, None].transpose(-1, -2))
    scores = scores * (head_dim**-0.5)
    causal = torch.triu(
        torch.full((sequence, sequence), -1e9, dtype=torch.float32), diagonal=1
    )
    probabilities = torch.softmax(scores + causal, dim=-1)
    attended = torch.matmul(probabilities, value[:, :, None])
    attended = attended.reshape(batch, query_heads, sequence, head_dim)
    attended = attended.transpose(1, 2).reshape(batch, sequence, d_model)
    attention_output = linear(attended, params["attention"]["out_proj"]["weight"])
    after_attention = hidden + attention_output
    ffn_input = rms_norm(after_attention, params["ffn_norm"]["weight"])
    gate = linear(ffn_input, params["feed_forward"]["gate"]["weight"])
    up = linear(ffn_input, params["feed_forward"]["up"]["weight"])
    activated = torch.nn.functional.silu(gate) * up
    block_output = after_attention + linear(
        activated, params["feed_forward"]["down"]["weight"]
    )
    difference = block_output - target
    authority_mass = torch.sum(loss_mask) * d_model
    loss = 0.5 * torch.sum(loss_mask[..., None] * difference * difference)
    loss = loss / authority_mass
    loss.backward()
    flat_params = _flatten_tree(params)
    return {
        "output": block_output.detach().numpy(),
        "loss": float(loss.detach()),
        "input_gradient": hidden.grad.detach().numpy(),
        "parameter_gradients": {
            name: flat_params[name].grad.detach().numpy()
            for name in canonical_parameter_schema(config)
        },
        "authority_mass": float(authority_mass),
    }


def mlx_reference(
    config: dict[str, Any],
    values: dict[str, Any],
    *,
    compiled: bool,
    repetitions: int,
) -> dict[str, Any]:
    import mlx.core as mx

    shape = config["shape"]
    batch = int(shape["batch"])
    sequence = int(shape["sequence"])
    d_model = int(shape["d_model"])
    query_heads = int(shape["query_heads"])
    kv_heads = int(shape["kv_heads"])
    head_dim = int(shape["head_dim"])
    query_groups = query_heads // kv_heads
    epsilon = float(config["semantics"]["rms_norm_epsilon"])
    cos_np, sin_np = _rope_angles_numpy(sequence, head_dim)
    cos = mx.array(cos_np, dtype=mx.float32)
    sin = mx.array(sin_np, dtype=mx.float32)
    causal = mx.triu(
        mx.full((sequence, sequence), -1e9, dtype=mx.float32), k=1
    )
    params = _nested_parameters(values["parameters"], mx.array)
    hidden = mx.array(values["hidden"])
    target = mx.array(values["target"])
    loss_mask = mx.array(values["loss_mask"])

    def linear(x: Any, weight: Any) -> Any:
        return mx.matmul(x, mx.swapaxes(weight, -1, -2))

    def rms_norm(x: Any, weight: Any) -> Any:
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + mx.array(epsilon, dtype=mx.float32)) * weight

    def rope(x: Any) -> Any:
        half = head_dim // 2
        rotated = mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)
        return x * cos + rotated * sin

    def objective(
        parameters: dict[str, Any],
        input_hidden: Any,
        expected: Any,
        mask: Any,
    ) -> tuple[Any, Any, Any]:
        attention_input = rms_norm(
            input_hidden, parameters["attention_norm"]["weight"]
        )
        query = linear(
            attention_input, parameters["attention"]["q_proj"]["weight"]
        )
        key = linear(attention_input, parameters["attention"]["k_proj"]["weight"])
        value = linear(
            attention_input, parameters["attention"]["v_proj"]["weight"]
        )
        query = rope(
            query.reshape(batch, sequence, query_heads, head_dim).transpose(
                0, 2, 1, 3
            )
        )
        key = rope(
            key.reshape(batch, sequence, kv_heads, head_dim).transpose(
                0, 2, 1, 3
            )
        )
        value = value.reshape(batch, sequence, kv_heads, head_dim).transpose(
            0, 2, 1, 3
        )
        grouped_query = query.reshape(
            batch, kv_heads, query_groups, sequence, head_dim
        )
        scores = mx.matmul(grouped_query, mx.swapaxes(key[:, :, None], -1, -2))
        scores = scores * mx.array(head_dim**-0.5, dtype=mx.float32)
        probabilities = mx.softmax(scores + causal, axis=-1)
        attended = mx.matmul(probabilities, value[:, :, None])
        attended = attended.reshape(batch, query_heads, sequence, head_dim)
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch, sequence, d_model
        )
        attention_output = linear(
            attended, parameters["attention"]["out_proj"]["weight"]
        )
        after_attention = input_hidden + attention_output
        ffn_input = rms_norm(after_attention, parameters["ffn_norm"]["weight"])
        gate = linear(ffn_input, parameters["feed_forward"]["gate"]["weight"])
        up = linear(ffn_input, parameters["feed_forward"]["up"]["weight"])
        activated = (gate * mx.sigmoid(gate)) * up
        block_output = after_attention + linear(
            activated, parameters["feed_forward"]["down"]["weight"]
        )
        difference = block_output - expected
        authority_mass = mx.sum(mask) * mx.array(d_model, dtype=mx.float32)
        loss = (
            mx.array(0.5, dtype=mx.float32)
            * mx.sum(mask[..., None] * difference * difference)
            / authority_mass
        )
        return loss, block_output, authority_mass

    differentiated = mx.value_and_grad(objective, argnums=(0, 1))
    execute = mx.compile(differentiated) if compiled else differentiated
    result = execute(params, hidden, target, loss_mask)
    ((loss, output, authority_mass), (parameter_gradients, input_gradient)) = result
    mx.eval(
        loss,
        output,
        authority_mass,
        input_gradient,
        parameter_gradients,
    )
    started = time.perf_counter()
    for _ in range(repetitions):
        result = execute(params, hidden, target, loss_mask)
        ((loss, output, authority_mass), (parameter_gradients, input_gradient)) = (
            result
        )
        mx.eval(
            loss,
            output,
            authority_mass,
            input_gradient,
            parameter_gradients,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "output": np.asarray(output, dtype=np.float32),
        "loss": float(loss.item()),
        "input_gradient": np.asarray(input_gradient, dtype=np.float32),
        "parameter_gradients": {
            name: np.asarray(value, dtype=np.float32)
            for name, value in _flatten_tree(parameter_gradients).items()
        },
        "authority_mass": float(authority_mass.item()),
        "mean_milliseconds": elapsed_ms / repetitions,
    }


def comparison(
    actual: np.ndarray, expected: np.ndarray, tolerance: float
) -> dict[str, Any]:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        raise QualificationFault(f"shape_mismatch:{left.shape}:{right.shape}")
    delta = np.abs(left - right)
    scale = np.maximum(np.abs(right), 1e-12)
    return {
        "shape": list(left.shape),
        "tolerance": tolerance,
        "maximum_absolute_delta": float(delta.max(initial=0.0)),
        "mean_absolute_delta": float(delta.mean()) if delta.size else 0.0,
        "maximum_relative_delta": float((delta / scale).max(initial=0.0)),
        "mismatch_count": int(np.count_nonzero(delta > tolerance)),
        "all_finite": bool(np.all(np.isfinite(left))),
        "nonzero_fraction": float(np.count_nonzero(left) / left.size)
        if left.size
        else 1.0,
    }


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float32).tobytes()
    ).hexdigest()


def all_arrays(result: dict[str, Any]) -> Iterable[np.ndarray]:
    yield result["output"]
    yield result["input_gradient"]
    yield from result["parameter_gradients"].values()


def qualify(
    config: dict[str, Any],
    torch_result: dict[str, Any],
    mlx_result: dict[str, Any],
    replay_result: dict[str, Any],
) -> dict[str, Any]:
    gate = config["reference_gate"]
    output = comparison(
        mlx_result["output"],
        torch_result["output"],
        float(gate["output_absolute_tolerance"]),
    )
    input_gradient = comparison(
        mlx_result["input_gradient"],
        torch_result["input_gradient"],
        float(gate["input_gradient_absolute_tolerance"]),
    )
    parameter_gradients = {
        name: comparison(
            mlx_result["parameter_gradients"][name],
            torch_result["parameter_gradients"][name],
            float(gate["parameter_gradient_absolute_tolerance"]),
        )
        for name in canonical_parameter_schema(config)
    }
    loss_delta = abs(float(mlx_result["loss"]) - float(torch_result["loss"]))
    replay_digests = {
        "output": (
            array_digest(mlx_result["output"]),
            array_digest(replay_result["output"]),
        ),
        "input_gradient": (
            array_digest(mlx_result["input_gradient"]),
            array_digest(replay_result["input_gradient"]),
        ),
        **{
            f"parameter_gradient:{name}": (
                array_digest(mlx_result["parameter_gradients"][name]),
                array_digest(replay_result["parameter_gradients"][name]),
            )
            for name in canonical_parameter_schema(config)
        },
    }
    replay_exact = (
        float(mlx_result["loss"]) == float(replay_result["loss"])
        and all(left == right for left, right in replay_digests.values())
    )
    minimum_nonzero = float(gate["require_nonzero_gradient_fraction"])
    gradients_present = (
        set(mlx_result["parameter_gradients"])
        == set(canonical_parameter_schema(config))
        and input_gradient["nonzero_fraction"] >= minimum_nonzero
        and all(
            item["nonzero_fraction"] >= minimum_nonzero
            for item in parameter_gradients.values()
        )
    )
    numerical_parity = (
        output["mismatch_count"] == 0
        and input_gradient["mismatch_count"] == 0
        and all(item["mismatch_count"] == 0 for item in parameter_gradients.values())
        and loss_delta <= float(gate["loss_absolute_tolerance"])
    )
    finite = all(
        np.all(np.isfinite(value))
        for result in (torch_result, mlx_result)
        for value in all_arrays(result)
    ) and math.isfinite(float(mlx_result["loss"]))
    green = numerical_parity and gradients_present and finite and replay_exact
    return {
        "policy": POLICY,
        "state": (
            "EXACT_BLOCK_REFERENCE_GREEN_NATIVE_NOT_YET_EXECUTED"
            if green
            else "EXACT_BLOCK_REFERENCE_FAILED"
        ),
        "claim_scope": config["claim_scope"],
        "shape": config["shape"],
        "semantics": config["semantics"],
        "parameter_schema": config["parameter_schema"],
        "parameter_count": parameter_count(config),
        "objective": {
            "torch_authority_mass": torch_result["authority_mass"],
            "mlx_authority_mass": mlx_result["authority_mass"],
            "torch_loss": torch_result["loss"],
            "mlx_loss": mlx_result["loss"],
            "absolute_loss_delta": loss_delta,
        },
        "comparisons": {
            "output": output,
            "input_gradient": input_gradient,
            "parameter_gradients": parameter_gradients,
        },
        "replay": {
            "exact": replay_exact,
            "digests": {
                name: {"first": left, "second": right}
                for name, (left, right) in replay_digests.items()
            },
        },
        "timing": {
            "compiled_mlx_mean_milliseconds": mlx_result["mean_milliseconds"],
            "scope": "reference_harness_only_not_a_native_or_full_step_speed_claim",
        },
        "gates": {
            "independent_framework_numerical_parity": numerical_parity,
            "every_parameter_gradient_present": gradients_present,
            "all_finite": finite,
            "exact_replay": replay_exact,
            "native_ane_block_executed": False,
            "joined_wall_speedup_proven": False,
            "production_eligible": False,
        },
        "native_next": config["native_gate"],
        "source_boundary": config["source_boundary"],
        "capability_claim": "NONE_ENGINEERING_NUMERICAL_ABI_ONLY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    config = load_config(args.config)
    values = frozen_inputs(config)
    torch_result = torch_reference(config, values)
    mlx_result = mlx_reference(
        config, values, compiled=True, repetitions=args.repetitions
    )
    replay_result = mlx_reference(config, values, compiled=True, repetitions=1)
    report = qualify(config, torch_result, mlx_result, replay_result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["state"].endswith("NATIVE_NOT_YET_EXECUTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
