from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from strict_generator_mlx_model import (  # noqa: E402
    MlxStrictGenerator,
    matched_dense_control_config,
    normalize_specialist_core_config,
    specialist_core_parameter_estimate,
)
from strict_generator_mlx_pretraining_probe import specialist_token_expert_map  # noqa: E402


def test_sparse_config_fails_closed_on_invalid_route() -> None:
    with pytest.raises(ValueError):
        normalize_specialist_core_config(
            {"enabled": True, "mode": "sparse_moe", "num_experts": 2, "top_k": 3}
        )
    with pytest.raises(ValueError):
        normalize_specialist_core_config({"enabled": True, "mode": "pretend_sparse"})


def test_matched_dense_control_tracks_sparse_active_expert_compute() -> None:
    sparse = {
        "enabled": True,
        "mode": "sparse_moe",
        "num_experts": 32,
        "top_k": 2,
        "expert_hidden_dim": 2048,
    }
    sparse_estimate = specialist_core_parameter_estimate(384, sparse)
    dense_estimate = specialist_core_parameter_estimate(384, matched_dense_control_config(sparse))
    assert sparse_estimate["specialist_active_parameter_fraction"] < 0.1
    relative_gap = abs(
        sparse_estimate["specialist_active_parameter_count_per_token"]
        - dense_estimate["specialist_active_parameter_count_per_token"]
    ) / dense_estimate["specialist_active_parameter_count_per_token"]
    assert relative_gap < 0.01
    assert sparse_estimate["specialist_total_parameter_count"] > dense_estimate["specialist_total_parameter_count"]


def test_router_supervision_map_covers_experts_without_runtime_labels() -> None:
    vocab = {f"NAME:private_token_{index}": index for index in range(128)}
    mapping, summary = specialist_token_expert_map(
        vocab,
        normalize_specialist_core_config(
            {
                "enabled": True,
                "mode": "sparse_moe",
                "num_experts": 8,
                "top_k": 2,
                "expert_hidden_dim": 32,
                "router_supervision_loss_weight": 0.2,
            }
        ),
    )
    assert len(mapping) == len(vocab)
    assert all(len(row) == 2 and row[0] != row[1] for row in mapping)
    assert summary["mapped_expert_count"] == 8
    assert summary["served_at_generation"] is False
    assert summary["uses_eval_tests_or_solutions"] is False


def test_sparse_mlx_model_routes_and_differentiates_selected_experts() -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    model = MlxStrictGenerator(
        source_vocab_size=32,
        target_vocab_size=40,
        max_source_len=8,
        max_target_len=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        semantic_slot_role_count=2,
        body_action_role_count=3,
        body_operand_role_count=4,
        body_state_event_role_count=5,
        body_executable_span_role_count=6,
        specialist_core={
            "enabled": True,
            "mode": "sparse_moe",
            "num_experts": 16,
            "top_k": 2,
            "expert_hidden_dim": 32,
            "router_aux_loss_weight": 0.01,
            "router_z_loss_weight": 0.001,
        },
        mx=mx,
        nn=nn,
    ).model
    src = mx.array([[1, 2, 3, 0], [4, 5, 6, 7]])
    tgt = mx.array([[1, 2, 3, 4, 5], [1, 6, 7, 8, 9]])

    def loss_fn(active_model, source, target):
        logits, router_loss = active_model.forward_with_router_loss(source, target[:, :-1])
        token_loss = mx.mean(nn.losses.cross_entropy(logits, target[:, 1:], reduction="none"))
        return token_loss + router_loss

    loss, gradients = nn.value_and_grad(model, loss_fn)(model, src, tgt)
    mx.eval(loss, gradients)
    route = model.specialist_route(src, tgt[:, :-1])
    mx.eval(route)
    assert model(src, tgt[:, :-1]).shape == (2, 4, 40)
    assert route["indices"].shape == (2, 4, 2)
    assert float(loss.item()) > 0.0
    assert len(route["indices"].tolist()) == 2
