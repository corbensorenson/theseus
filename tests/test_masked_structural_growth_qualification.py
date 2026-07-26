from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import masked_structural_growth_qualification as growth
from standard_causal_transformer_model import (
    CausalTransformerConfig,
    build_model,
)


def test_config_precommits_two_nested_function_preserving_schedules() -> None:
    config = growth.load_config(
        ROOT / "configs/masked_structural_growth_qualification.json"
    )

    assert len(config["growth_schedules"]) == 2
    assert config["training"]["steps"] == 128
    assert config["seeds"] == [20260722, 20260723, 20260724]
    for schedule in config["growth_schedules"]:
        stages = schedule["stages"]
        assert stages[-1]["full_layer_slots"] == [0, 1, 2, 3, 4, 5]
        assert stages[-1]["stop_step"] == 128
        assert all(
            set(left["full_layer_slots"]).issubset(right["full_layer_slots"])
            for left, right in zip(stages, stages[1:])
        )


def test_parameter_and_optimizer_path_maps_preserve_slots_and_zero_new_state() -> None:
    parameter_sources = growth.stage_parameter_sources(
        [
            "token_embedding.weight",
            "layers.0.attention.q_proj.weight",
            "layers.1.attention.q_proj.weight",
            "layers.2.attention.q_proj.weight",
            "final_norm.weight",
        ],
        target_slots=[0, 1, 3],
        prior_slots=[0, 3],
        new_layer_source_local_indices={1: 0},
    )
    assert parameter_sources == {
        "token_embedding.weight": ("prior", "token_embedding.weight"),
        "layers.0.attention.q_proj.weight": (
            "prior",
            "layers.0.attention.q_proj.weight",
        ),
        "layers.1.attention.q_proj.weight": (
            "stacked_prior",
            "layers.0.attention.q_proj.weight",
        ),
        "layers.2.attention.q_proj.weight": (
            "prior",
            "layers.1.attention.q_proj.weight",
        ),
        "final_norm.weight": ("prior", "final_norm.weight"),
    }
    optimizer_sources = growth.stage_optimizer_sources(
        [
            "step",
            "layers.0.attention.q_proj.weight.m",
            "layers.1.attention.q_proj.weight.m",
            "layers.2.attention.q_proj.weight.v",
        ],
        target_slots=[0, 1, 3],
        prior_slots=[0, 3],
    )
    assert optimizer_sources == {
        "step": "step",
        "layers.0.attention.q_proj.weight.m": (
            "layers.0.attention.q_proj.weight.m"
        ),
        "layers.1.attention.q_proj.weight.m": None,
        "layers.2.attention.q_proj.weight.v": (
            "layers.1.attention.q_proj.weight.v"
        ),
    }


def test_growth_masks_ramp_only_new_layers() -> None:
    schedule = {
        "mask_ramp_steps": 4,
        "stages": [
            {"stop_step": 2, "full_layer_slots": [0, 2]},
            {"stop_step": 8, "full_layer_slots": [0, 1, 2]},
        ],
    }

    assert growth.boundary_masks(schedule, 1) == (1.0, 0.0, 1.0)
    assert growth.stage_masks(schedule, 1, 3) == (1.0, 0.25, 1.0)
    assert growth.stage_masks(schedule, 1, 6) == (1.0, 1.0, 1.0)


def test_mlx_growth_boundary_is_exact_and_new_adam_moments_are_zero() -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    optim = pytest.importorskip("mlx.optimizers")
    mlx_utils = pytest.importorskip("mlx.utils")

    full_config = CausalTransformerConfig(
        vocab_size=64,
        d_model=16,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=32,
    )
    small_config = CausalTransformerConfig(
        vocab_size=64,
        d_model=16,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=32,
    )
    mx.random.seed(17)
    canonical = build_model(full_config, mx=mx, nn=nn)
    small = build_model(small_config, mx=mx, nn=nn)
    growth.update_model_for_stage(
        small,
        canonical_parameters=canonical.parameters(),
        target_slots=[0, 2],
        prior_model=None,
        prior_slots=None,
        new_layer_source_local_indices=None,
        mlx_utils=mlx_utils,
    )
    small_optimizer = optim.AdamW(learning_rate=1e-3)
    small_optimizer.init(small.trainable_parameters())

    def objective(model: object, tokens: object) -> object:
        logits, _cache = model(tokens)
        return mx.mean(logits * logits)

    loss, gradients = nn.value_and_grad(small, objective)(
        small, mx.array([[3, 5, 7]], dtype=mx.int32)
    )
    small_optimizer.update(small, gradients)
    mx.eval(small.parameters(), small_optimizer.state, loss)

    expanded = build_model(full_config, mx=mx, nn=nn)
    growth.update_model_for_stage(
        expanded,
        canonical_parameters=canonical.parameters(),
        target_slots=[0, 1, 2],
        prior_model=small,
        prior_slots=[0, 2],
        new_layer_source_local_indices={1: 0},
        mlx_utils=mlx_utils,
    )
    expanded_optimizer = optim.AdamW(learning_rate=1e-3)
    expanded_optimizer.init(expanded.trainable_parameters())
    inventory = growth.transfer_adamw_state(
        expanded_optimizer,
        small_optimizer,
        target_slots=[0, 1, 2],
        prior_slots=[0, 2],
        mlx_utils=mlx_utils,
    )
    mx.eval(expanded.parameters(), expanded_optimizer.state)

    tokens = mx.array([[2, 4, 6, 8]], dtype=mx.int32)
    small_logits, _cache = small(tokens)
    expanded.set_structural_growth_masks((1.0, 0.0, 1.0))
    expanded_logits, _cache = expanded(tokens)
    mx.eval(small_logits, expanded_logits)
    assert np.array_equal(np.asarray(small_logits), np.asarray(expanded_logits))
    assert inventory["new_zero_optimizer_leaves"] > 0
    flat_state = dict(mlx_utils.tree_flatten(expanded_optimizer.state))
    new_moments = [
        np.asarray(value)
        for name, value in flat_state.items()
        if name.startswith("layers.1.") and name.endswith((".m", ".v"))
    ]
    assert new_moments
    assert all(np.count_nonzero(value) == 0 for value in new_moments)
