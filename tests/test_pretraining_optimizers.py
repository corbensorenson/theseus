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
import mlx.optimizers as optim
import mlx.utils as mlx_utils


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pretraining_optimizers as candidate_optimizers


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 8, bias=False)
        self.output = nn.Linear(8, 2, bias=True)

    def __call__(self, value):
        return self.output(nn.silu(self.hidden(value)))


def loss_fn(model, inputs, targets):
    return mx.mean(mx.square(model(inputs) - targets))


def train(optimizer_id: str, steps: int = 48):
    mx.random.seed(20260722)
    model = TinyRegressor()
    optimizer = candidate_optimizers.build_optimizer(
        optimizer_id,
        learning_rate=0.02,
        weight_decay=0.0,
        warmup_steps=4,
        optim=optim,
        mx=mx,
    )
    inputs = mx.array(
        [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, -1.0, 2.0], [2.0, 1.0, 0.0, -1.0]],
        dtype=mx.float32,
    )
    targets = mx.array([[1.0, -1.0], [0.5, 0.25], [-0.5, 1.5]], dtype=mx.float32)
    value_and_grad = nn.value_and_grad(model, loss_fn)
    initial = float(loss_fn(model, inputs, targets).item())
    for _ in range(steps):
        loss, gradients = value_and_grad(model, inputs, targets)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
    final = float(loss_fn(model, inputs, targets).item())
    return model, optimizer, inputs, targets, initial, final


def test_all_optimizer_candidates_make_loss_progress() -> None:
    for optimizer_id in sorted(candidate_optimizers.OPTIMIZER_IDS):
        _model, _optimizer, _inputs, _targets, initial, final = train(optimizer_id)
        assert final < initial, (optimizer_id, initial, final)


def test_muon_routes_only_hidden_matrices() -> None:
    mx.random.seed(7)
    model = TinyRegressor()
    flat = dict(mlx_utils.tree_flatten(model.parameters()))
    assert candidate_optimizers.muon_hidden_matrix_filter(
        "hidden.weight", flat["hidden.weight"]
    )
    assert not candidate_optimizers.muon_hidden_matrix_filter(
        "output.weight", flat["output.weight"]
    )
    assert not candidate_optimizers.muon_hidden_matrix_filter(
        "output.bias", flat["output.bias"]
    )


def test_muon_owns_matrix_rate_without_inflating_adamw_fallback_rate() -> None:
    optimizer = candidate_optimizers.build_optimizer(
        "muon_mlx",
        learning_rate=3e-4,
        muon_learning_rate=0.02,
        weight_decay=0.01,
        optim=optim,
        mx=mx,
    )
    rates = [float(child.learning_rate.item()) for child in optimizer.optimizers]
    assert abs(rates[0] - 0.02) < 1e-8
    assert abs(rates[1] - 3e-4) < 1e-10


def test_schedule_free_eval_and_training_iterates_are_explicit_and_reversible() -> None:
    model, optimizer, inputs, _targets, _initial, _final = train(
        "schedule_free_adamw_mlx", steps=12
    )
    training_logits = model(inputs)
    optimizer.set_evaluation_iterate(model)
    evaluation_logits = model(inputs)
    optimizer.set_training_iterate(model)
    restored_logits = model(inputs)
    mx.eval(training_logits, evaluation_logits, restored_logits)
    assert float(mx.max(mx.abs(training_logits - evaluation_logits)).item()) > 0.0
    assert float(mx.max(mx.abs(training_logits - restored_logits)).item()) == 0.0


def test_schedule_free_optimizer_state_roundtrips_for_exact_next_update() -> None:
    model, optimizer, inputs, targets, _initial, _final = train(
        "schedule_free_adamw_mlx", steps=12
    )
    mx.random.seed(99)
    reloaded = TinyRegressor()
    reloaded.update(model.parameters())
    resumed = candidate_optimizers.build_optimizer(
        "schedule_free_adamw_mlx",
        learning_rate=0.02,
        weight_decay=0.0,
        warmup_steps=4,
        optim=optim,
        mx=mx,
    )
    resumed.state = mlx_utils.tree_unflatten(
        [(name, mx.array(value)) for name, value in mlx_utils.tree_flatten(optimizer.state)]
    )
    first_grad = nn.value_and_grad(model, loss_fn)
    second_grad = nn.value_and_grad(reloaded, loss_fn)
    first_loss, first_grads = first_grad(model, inputs, targets)
    second_loss, second_grads = second_grad(reloaded, inputs, targets)
    optimizer.update(model, first_grads)
    resumed.update(reloaded, second_grads)
    mx.eval(model.parameters(), reloaded.parameters(), first_loss, second_loss)
    for (left_name, left), (right_name, right) in zip(
        mlx_utils.tree_flatten(model.parameters()),
        mlx_utils.tree_flatten(reloaded.parameters()),
    ):
        assert left_name == right_name
        assert float(mx.max(mx.abs(left - right)).item()) == 0.0


def test_bfloat16_moment_adamw_keeps_fp32_weights_and_restarts_exactly() -> None:
    model, optimizer, inputs, targets, _initial, _final = train(
        "adamw_bfloat16_moments_mlx", steps=8
    )
    flat_state = dict(mlx_utils.tree_flatten(optimizer.state))
    moment_values = [
        value
        for name, value in flat_state.items()
        if name.endswith(".m") or name.endswith(".v")
    ]
    assert moment_values
    assert all(value.dtype == mx.bfloat16 for value in moment_values)
    assert all(
        value.dtype == mx.float32
        for _name, value in mlx_utils.tree_flatten(model.parameters())
    )
    assert (
        candidate_optimizers.optimizer_state_kind(optimizer)
        == "adamw_bfloat16_moments_fp32_transactional_update_math"
    )

    mx.random.seed(101)
    reloaded = TinyRegressor()
    reloaded.update(model.parameters())
    resumed = candidate_optimizers.build_optimizer(
        "adamw_bfloat16_moments_mlx",
        learning_rate=0.02,
        weight_decay=0.0,
        optim=optim,
        mx=mx,
    )
    resumed.state = mlx_utils.tree_unflatten(
        [(name, mx.array(value)) for name, value in mlx_utils.tree_flatten(optimizer.state)]
    )
    first_loss, first_grads = nn.value_and_grad(model, loss_fn)(
        model, inputs, targets
    )
    second_loss, second_grads = nn.value_and_grad(reloaded, loss_fn)(
        reloaded, inputs, targets
    )
    optimizer.update(model, first_grads)
    resumed.update(reloaded, second_grads)
    mx.eval(model.parameters(), reloaded.parameters(), first_loss, second_loss)
    for (left_name, left), (right_name, right) in zip(
        mlx_utils.tree_flatten(model.parameters()),
        mlx_utils.tree_flatten(reloaded.parameters()),
    ):
        assert left_name == right_name
        assert float(mx.max(mx.abs(left - right)).item()) == 0.0


def test_adafactor_factors_matrices_falls_back_for_vectors_and_restarts_exactly() -> None:
    model, optimizer, inputs, targets, _initial, _final = train(
        "adafactor_mlx", steps=8
    )
    flat_parameters = dict(mlx_utils.tree_flatten(model.parameters()))
    flat_state = dict(mlx_utils.tree_flatten(optimizer.state))
    matrix_name = "hidden.weight"
    vector_name = "output.bias"
    assert flat_state[f"{matrix_name}.exp_avg_sq_row"].shape == (
        flat_parameters[matrix_name].shape[0],
    )
    assert flat_state[f"{matrix_name}.exp_avg_sq_col"].shape == (
        flat_parameters[matrix_name].shape[1],
    )
    assert f"{matrix_name}.exp_avg_sq" not in flat_state
    assert flat_state[f"{vector_name}.exp_avg_sq"].shape == flat_parameters[
        vector_name
    ].shape
    assert f"{vector_name}.exp_avg_sq_row" not in flat_state
    assert (
        candidate_optimizers.optimizer_state_kind(optimizer)
        == "adafactor_factored_matrices_unfactored_vectors_scalars"
    )
    assert optimizer.parameter_scale_policy == "max_eps2_parameter_rms"
    assert optimizer.update_clip_threshold == 1.0

    mx.random.seed(102)
    reloaded = TinyRegressor()
    reloaded.update(model.parameters())
    resumed = candidate_optimizers.build_optimizer(
        "adafactor_mlx",
        learning_rate=0.02,
        weight_decay=0.0,
        optim=optim,
        mx=mx,
    )
    resumed.state = mlx_utils.tree_unflatten(
        [
            (name, mx.array(value))
            for name, value in mlx_utils.tree_flatten(optimizer.state)
        ]
    )
    resumed.init(reloaded.trainable_parameters())
    first_loss, first_grads = nn.value_and_grad(model, loss_fn)(
        model, inputs, targets
    )
    second_loss, second_grads = nn.value_and_grad(reloaded, loss_fn)(
        reloaded, inputs, targets
    )
    optimizer.update(model, first_grads)
    resumed.update(reloaded, second_grads)
    mx.eval(model.parameters(), reloaded.parameters(), first_loss, second_loss)
    for (left_name, left), (right_name, right) in zip(
        mlx_utils.tree_flatten(model.parameters()),
        mlx_utils.tree_flatten(reloaded.parameters()),
    ):
        assert left_name == right_name
        assert float(mx.max(mx.abs(left - right)).item()) == 0.0


def test_invalid_optimizer_contracts_fail_closed() -> None:
    try:
        candidate_optimizers.build_optimizer(
            "unknown", learning_rate=0.1, weight_decay=0.0, optim=optim, mx=mx
        )
    except candidate_optimizers.OptimizerContractFault as exc:
        assert "optimizer_unknown" in str(exc)
    else:
        raise AssertionError("unknown optimizer was accepted")
