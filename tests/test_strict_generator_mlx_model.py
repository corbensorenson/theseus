from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
from strict_generator_mlx_decode_eval import generate_candidates_mlx  # noqa: E402
from strict_generator_mlx_pretraining_probe import (  # noqa: E402
    active_parameter_accounting,
    model_data_exposure_summary,
    specialist_token_expert_map,
)


def _tiny_mlx_decode_model(*, sparse: bool = False):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    model = MlxStrictGenerator(
        source_vocab_size=32,
        target_vocab_size=40,
        max_source_len=8,
        max_target_len=10,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        semantic_slot_role_count=2,
        body_action_role_count=3,
        body_operand_role_count=4,
        body_state_event_role_count=5,
        body_executable_span_role_count=6,
        specialist_core=(
            {
                "enabled": True,
                "mode": "sparse_moe",
                "num_experts": 4,
                "top_k": 2,
                "expert_hidden_dim": 24,
            }
            if sparse
            else {"enabled": False}
        ),
        mx=mx,
        nn=nn,
    ).model
    model.eval()
    return mx, model


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


def test_data_exposure_never_counts_optimizer_repetition_as_unique_data() -> None:
    summary = model_data_exposure_summary(
        one_pass_source_token_positions=600,
        one_pass_target_token_positions=400,
        optimizer_token_positions=10_000,
        active_parameter_count=2_000,
    )
    assert summary["one_pass_total_token_positions"] == 1_000
    assert summary["one_pass_tokens_per_active_parameter"] == 0.5
    assert summary["optimizer_repetition_factor"] == 10.0
    assert summary["data_scale_state"] == "underdata"
    assert summary["optimizer_repetition_counted_as_unique_data"] is False


def test_active_parameter_accounting_uses_measured_core_and_selected_experts() -> None:
    summary = active_parameter_accounting(
        {
            "parameter_count": 10_000,
            "core_parameter_count": 8_000,
            "parameter_count_by_root": {"slot_role_router": 2_000},
        },
        {
            "specialist_total_parameter_count": 4_000,
            "specialist_active_parameter_count_per_token": 1_000,
        },
        active_optional_roots={"slot_role_router"},
    )
    assert summary["model_total_parameter_count"] == 10_000
    assert summary["shared_core_parameter_count_excluding_specialists"] == 4_000
    assert summary["model_active_parameter_count_per_token"] == 7_000


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


def test_factorized_auxiliary_heads_share_vocab_projection() -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    kwargs = dict(
        source_vocab_size=32,
        target_vocab_size=128,
        max_source_len=8,
        max_target_len=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        semantic_slot_role_count=6,
        body_action_role_count=3,
        body_operand_role_count=4,
        body_state_event_role_count=5,
        body_executable_span_role_count=6,
        specialist_core={"enabled": False},
        mx=mx,
        nn=nn,
    )
    legacy = MlxStrictGenerator(**kwargs, auxiliary_head_policy="legacy_materialized_v1").model
    factorized = MlxStrictGenerator(
        **kwargs, auxiliary_head_policy="shared_factorized_on_demand_v1"
    ).model
    factorized_without_slot = MlxStrictGenerator(
        **kwargs,
        auxiliary_head_policy="shared_factorized_on_demand_v1",
        semantic_slot_head=False,
    ).model
    tied = MlxStrictGenerator(
        **kwargs,
        auxiliary_head_policy="shared_factorized_on_demand_v1",
        semantic_slot_head=False,
        output_projection_policy="tied_target_embedding_v1",
    ).model
    legacy_count = sum(int(value.size) for _name, value in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(legacy.trainable_parameters()))
    factorized_count = sum(int(value.size) for _name, value in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(factorized.trainable_parameters()))
    factorized_without_slot_count = sum(
        int(value.size)
        for _name, value in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
            factorized_without_slot.trainable_parameters()
        )
    )
    tied_count = sum(
        int(value.size)
        for _name, value in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
            tied.trainable_parameters()
        )
    )
    src = mx.array([[1, 2, 3, 0]])
    logits = factorized.semantic_slot_logits(src)
    mx.eval(logits)
    assert logits.shape == (1, 6, 128)
    assert factorized_count < legacy_count
    assert factorized_without_slot_count < factorized_count
    tied_logits = tied(src, mx.array([[1, 2, 3]]))
    mx.eval(tied_logits)
    assert tied_logits.shape == (1, 3, 128)
    assert tied_count < factorized_without_slot_count


def test_incremental_decode_matches_full_prefix_for_every_materialized_head() -> None:
    mx, model = _tiny_mlx_decode_model()
    src = mx.array([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]], dtype=mx.int32)
    target = mx.array([[1, 8, 9, 10], [1, 11, 12, 13]], dtype=mx.int32)
    context = model.prepare_incremental_decode(src)
    cache = None
    for position in range(int(target.shape[1])):
        full = model.logits_bundle(src, target[:, : position + 1])
        incremental, cache = model.incremental_logits_bundle(
            target[:, position : position + 1],
            position=position,
            decode_context=context,
            self_cache=cache,
        )
        mx.eval(*list(full.values()), *list(incremental.values()))
        assert set(full) == set(incremental)
        for name in sorted(full):
            expected = np.asarray(full[name][:, -1, :])
            actual = np.asarray(incremental[name][:, -1, :])
            assert np.max(np.abs(expected - actual)) < 1e-5, name
            assert np.array_equal(np.argmax(expected, axis=-1), np.argmax(actual, axis=-1)), name
        assert cache is not None
        assert len(cache) == 2
        assert all(int(layer["keys"].shape[2]) == position + 1 for layer in cache)
        assert all(int(layer["values"].shape[2]) == position + 1 for layer in cache)


@pytest.mark.parametrize("sparse", [False, True])
def test_bundled_logits_match_legacy_individual_head_calls(sparse: bool) -> None:
    mx, model = _tiny_mlx_decode_model(sparse=sparse)
    src = mx.array([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=mx.int32)
    target = mx.array([[1, 8, 9], [1, 10, 11]], dtype=mx.int32)
    bundled = model.logits_bundle(src, target)
    legacy = {
        "token": model(src, target),
        "body_transition": model.body_transition_logits(src, target),
        "body_action": model.body_action_logits(src, target),
        "body_operand": model.body_operand_logits(src, target),
        "body_state_event": model.body_state_event_logits(src, target),
    }
    mx.eval(*list(bundled.values()), *list(legacy.values()))
    for name in sorted(legacy):
        assert np.max(np.abs(np.asarray(bundled[name]) - np.asarray(legacy[name]))) < 1e-6, name


def test_incremental_decode_rejects_stale_or_malformed_cache_state() -> None:
    mx, model = _tiny_mlx_decode_model()
    src = mx.array([[1, 2, 3, 0]], dtype=mx.int32)
    context = model.prepare_incremental_decode(src)
    token = mx.array([[1]], dtype=mx.int32)

    with pytest.raises(ValueError, match="position out of range"):
        model.incremental_logits_bundle(
            token,
            position=99,
            decode_context=context,
            self_cache=None,
        )

    stale_context = dict(context)
    stale_context["layer_count"] = 1
    with pytest.raises(ValueError, match="context layer count mismatch"):
        model.incremental_logits_bundle(
            token,
            position=0,
            decode_context=stale_context,
            self_cache=None,
        )

    _bundle, cache = model.incremental_logits_bundle(
        token,
        position=0,
        decode_context=context,
        self_cache=None,
    )
    with pytest.raises(ValueError, match="self-cache layer count mismatch"):
        model.incremental_logits_bundle(
            mx.array([[2]], dtype=mx.int32),
            position=1,
            decode_context=context,
            self_cache=cache[:1],
        )


def test_incremental_and_full_prefix_candidate_search_have_identical_token_identity() -> None:
    mx, model = _tiny_mlx_decode_model()
    target_vocab = {
        token: index
        for index, token in enumerate(
            [
                "<pad>",
                "<bos>",
                "<eos>",
                "return",
                "NAME:value",
                "NEWLINE",
                "NUMBER:1",
                "(",
                ")",
                ":",
                "INDENT",
                "DEDENT",
                "=",
                "+",
                "pass",
                "if",
                "else",
                "True",
                "False",
                "NAME:result",
                "NUMBER:0",
                "-",
                "*",
                "not",
                "and",
                "or",
                "in",
                "for",
                "while",
                "break",
                "continue",
                "None",
                "<unk>",
                "[",
                "]",
                "{",
                "}",
                ",",
                ".",
                "==",
            ]
        )
    }
    common = {
        "max_target_tokens": 8,
        "fanout_top_k": 2,
        "grammar_top_k": len(target_vocab),
        "decode_beam_width": 2,
        "decode_branching_factor": 2,
        "target_mode": "body_tokens",
        "body_token_decode_policy": "lightweight_python_v1",
        "source_texts": ["Return value."],
        "allowed_name_sets": [{"value"}],
        "input_type_hints_by_row": [{}],
        "source_condition_expectations": [{"enabled": False, "required_features": []}],
        "require_parameter_use": False,
        "require_nontrivial_return": False,
        "require_top_level_return": False,
        "use_semantic_plan_head_prefix": False,
        "prefer_source_plan_compatibility": False,
        "use_semantic_slot_head_prefix": False,
        "enable_learned_expression_token_bias": False,
        "use_body_transition_head": True,
        "body_transition_head_blend": 0.35,
        "use_body_action_head": True,
        "body_action_head_blend": 0.35,
        "use_body_operand_head": True,
        "body_operand_head_blend": 0.35,
        "use_body_state_event_head": True,
        "body_state_event_head_blend": 0.35,
        "prefer_learned_prefix_decision_adequacy": False,
        "prefer_source_condition_adequacy": False,
        "require_source_condition_adequacy": False,
        "block_shallow_loop_identity_update": False,
        "enable_loop_progress_guard": False,
        "enable_expression_closure_guard": False,
        "enable_expression_value_guard": False,
        "enable_semantic_operation_value_construction": False,
        "require_binding_prefix_groups": False,
        "mx": mx,
    }
    source_rows = [[1, 2, 3, 0, 0, 0, 0, 0]]
    incremental, incremental_diagnostics = generate_candidates_mlx(
        model,
        source_rows,
        target_vocab,
        decode_cache_mode="incremental",
        **common,
    )
    full, full_diagnostics = generate_candidates_mlx(
        model,
        source_rows,
        target_vocab,
        decode_cache_mode="full_prefix",
        **common,
    )
    def identity(rows):
        return [
            [(row["decoded_token_sha256"], row["decoded_tokens"], row["body"]) for row in task]
            for task in rows
        ]
    assert identity(incremental) == identity(full)
    for incremental_task, full_task in zip(incremental, full):
        for incremental_row, full_row in zip(incremental_task, full_task):
            assert abs(incremental_row["rank_score"] - full_row["rank_score"]) < 1e-5
    assert incremental_diagnostics[0]["decode_cache_receipt"]["source_encode_batch_calls"] == 1
    assert incremental_diagnostics[0]["decode_cache_receipt"]["full_prefix_recomputation_batch_calls"] == 0
    assert full_diagnostics[0]["decode_cache_receipt"]["full_prefix_recomputation_batch_calls"] > 0
