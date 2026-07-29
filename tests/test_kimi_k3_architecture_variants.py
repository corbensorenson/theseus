from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


if os.environ.get("THESEUS_GUARDED_ACCELERATOR_CHILD") != "1":
    pytest.skip(
        "accelerator test module requires scripts/host_resource_safety.py",
        allow_module_level=True,
    )

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mlx_utils


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from standard_causal_transformer_model import (
    CausalTransformerConfig,
    analytical_parameter_count,
    build_model,
)


def configs():
    common = {
        "vocab_size": 64,
        "d_model": 32,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "ff_dim": 64,
        "attention_policy": "causal",
    }
    return (
        CausalTransformerConfig(**common),
        CausalTransformerConfig(
            **common,
            attention_residual_mode="block",
            attention_residual_block_size=2,
        ),
    )


def situ_configs():
    common = {
        "vocab_size": 64,
        "d_model": 32,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "ff_dim": 64,
        "attention_policy": "causal",
    }
    return (
        CausalTransformerConfig(**common),
        CausalTransformerConfig(
            **common,
            feed_forward_activation="situ_glu",
            situ_glu_gate_beta=4.0,
            situ_glu_up_beta=25.0,
        ),
    )


def test_attnres_reduces_exactly_to_control_and_changes_function() -> None:
    control_config, candidate_config = configs()
    mx.random.seed(7)
    control = build_model(control_config, mx=mx, nn=nn)
    mx.random.seed(7)
    candidate = build_model(candidate_config, mx=mx, nn=nn)
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    control_logits, _cache = control(tokens)
    candidate.attention_residual_scale = 0.0
    reduced_logits, _cache = candidate(tokens)
    candidate.attention_residual_scale = 1.0
    active_logits, _cache = candidate(tokens)
    mx.eval(control_logits, reduced_logits, active_logits)
    assert (
        float(mx.max(mx.abs(control_logits - reduced_logits)).item())
        == 0.0
    )
    assert (
        float(mx.max(mx.abs(control_logits - active_logits)).item())
        > 0.0
    )


def test_attnres_parameter_accounting_and_gradients_are_complete() -> None:
    control_config, candidate_config = configs()
    assert (
        analytical_parameter_count(candidate_config)
        - analytical_parameter_count(control_config)
        == (candidate_config.num_layers + 1) * candidate_config.d_model
    )
    mx.random.seed(8)
    model = build_model(candidate_config, mx=mx, nn=nn)
    actual = sum(
        int(value.size)
        for _name, value in mlx_utils.tree_flatten(model.parameters())
    )
    assert actual == analytical_parameter_count(candidate_config)
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    def objective(local_model):
        logits, _cache = local_model(tokens)
        return mx.mean(mx.square(logits))

    loss, gradients = nn.value_and_grad(model, objective)(model)
    mx.eval(loss, gradients)
    flat = dict(mlx_utils.tree_flatten(gradients))
    query = flat["attention_residual_queries.weight"]
    assert float(mx.sum(mx.abs(query[0])).item()) == 0.0
    assert all(
        float(mx.sum(mx.abs(query[index])).item()) > 0.0
        for index in range(1, candidate_config.num_layers + 1)
    )
    for layer in range(candidate_config.num_layers):
        assert (
            sum(
                float(mx.sum(mx.abs(value)).item())
                for name, value in flat.items()
                if name.startswith(f"layers.{layer}.")
            )
            > 0.0
        )


def test_attnres_cached_decode_matches_full_decode() -> None:
    _control_config, candidate_config = configs()
    mx.random.seed(9)
    model = build_model(candidate_config, mx=mx, nn=nn)
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    full, _cache = model(tokens)
    cache = None
    pieces = []
    for index in range(int(tokens.shape[1])):
        logits, cache = model(tokens[:, index : index + 1], cache)
        pieces.append(logits)
    incremental = mx.concatenate(pieces, axis=1)
    mx.eval(full, incremental)
    assert (
        float(mx.max(mx.abs(full - incremental)).item()) < 2e-5
    )


def test_situ_glu_is_parameter_matched_source_faithful_and_bounded() -> None:
    control_config, candidate_config = situ_configs()
    assert analytical_parameter_count(candidate_config) == (
        analytical_parameter_count(control_config)
    )
    mx.random.seed(10)
    control = build_model(control_config, mx=mx, nn=nn)
    mx.random.seed(10)
    candidate = build_model(candidate_config, mx=mx, nn=nn)
    hidden = mx.linspace(-100.0, 100.0, 32).reshape(1, 1, 32)
    control_ffn = control.layers[0].feed_forward
    candidate_ffn = candidate.layers[0].feed_forward
    gate = candidate_ffn.gate(hidden)
    up = candidate_ffn.up(hidden)
    expected_product = (
        4.0
        * mx.tanh(gate / 4.0)
        * mx.sigmoid(gate)
        * 25.0
        * mx.tanh(up / 25.0)
    )
    actual = candidate_ffn(hidden)
    expected = candidate_ffn.down(expected_product)
    control_output = control_ffn(hidden)
    mx.eval(gate, up, actual, expected, control_output)
    assert (
        float(mx.max(mx.abs(actual - expected)).item()) <= 1e-5
    )
    assert float(mx.max(mx.abs(expected_product)).item()) <= 100.0
    assert (
        float(mx.max(mx.abs(actual - control_output)).item()) > 0.0
    )


def test_situ_glu_cached_decode_matches_full_decode() -> None:
    _control_config, candidate_config = situ_configs()
    mx.random.seed(11)
    model = build_model(candidate_config, mx=mx, nn=nn)
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    full, _cache = model(tokens)
    cache = None
    pieces = []
    for index in range(int(tokens.shape[1])):
        logits, cache = model(tokens[:, index : index + 1], cache)
        pieces.append(logits)
    incremental = mx.concatenate(pieces, axis=1)
    mx.eval(full, incremental)
    assert (
        float(mx.max(mx.abs(full - incremental)).item()) < 2e-5
    )
