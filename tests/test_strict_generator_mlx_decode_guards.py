from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from strict_generator_mlx_decode_guards import (
    bare_builtin_type_value_argument_invalid,
    token_blocked_by_strict_decode_guard,
    unclosed_paren_indices,
)
import semantic_ir
from neural_seed_token_decoder_support import (
    build_target_vocab,
    forced_lightweight_python_token,
    target_tokens,
)
from neural_seed_decode_static_guard import decode_static_guard
from strict_generator_mlx_decode_eval import broad_private_decode_token_budget, mlx_token_choices
from strict_generator_mlx_pretraining_probe import body_start_span_mask_mlx, parameter_is_optional_auxiliary


def test_nested_isinstance_type_tuple_is_legal() -> None:
    prefix = ["NAME:if", "NAME:isinstance", "OP:(", "NAME:data", "OP:,", "OP:("]

    assert unclosed_paren_indices(["if", "isinstance", "(", "data", ",", "("]) == [2, 5]
    assert not bare_builtin_type_value_argument_invalid(
        ("if", "isinstance", "(", "data", ",", "("),
        "NAME:list",
    )
    assert not token_blocked_by_strict_decode_guard(
        prefix,
        "NAME:list",
        require_nontrivial_return=True,
        allowed_names={"data", "other"},
        input_type_hints={},
    )


def test_builtin_type_is_still_blocked_as_unrelated_value_argument() -> None:
    assert bare_builtin_type_value_argument_invalid(("return", "min", "("), "NAME:list")
    assert bare_builtin_type_value_argument_invalid(("answer", "=", "max", "(", "data", ","), "NAME:dict")


def test_isinstance_first_argument_does_not_gain_type_tuple_exception() -> None:
    prefix = ["NAME:return", "NAME:isinstance", "OP:("]

    assert token_blocked_by_strict_decode_guard(
        prefix,
        "NAME:list",
        require_nontrivial_return=True,
        allowed_names={"data"},
        input_type_hints={},
    )


def test_layout_helper_never_synthesizes_a_compound_clause_colon() -> None:
    inverse = {0: "OP::", 1: "NEWLINE:", 2: "INDENT:"}
    probabilities = [0.99, 0.005, 0.005]

    assert forced_lightweight_python_token(
        ["NAME:if", "NAME:isinstance", "OP:(", "NAME:data", "OP:)", "NAME:and", "NAME:data"],
        inverse,
        probabilities,
    ) is None
    assert forced_lightweight_python_token(
        ["NAME:if", "NAME:data"],
        inverse,
        probabilities,
    ) is None


def test_layout_helper_still_completes_post_colon_layout() -> None:
    inverse = {0: "OP::", 1: "NEWLINE:", 2: "INDENT:"}
    probabilities = [0.01, 0.89, 0.1]

    assert forced_lightweight_python_token(
        ["NAME:if", "NAME:data", "OP::"],
        inverse,
        probabilities,
    ) == (1, 0.89)


def test_visible_parameter_can_start_a_nontrivial_return_expression() -> None:
    assert not token_blocked_by_strict_decode_guard(
        ["NAME:return"],
        "NAME:data",
        require_nontrivial_return=True,
        allowed_names={"data"},
        input_type_hints={},
    )
    assert not decode_static_guard(
        "return data",
        allowed_names={"data"},
        require_parameter_use=True,
        require_nontrivial_return=True,
        require_top_level_return=True,
    )["passed"]
    assert decode_static_guard(
        "return data[0]",
        allowed_names={"data"},
        require_parameter_use=True,
        require_nontrivial_return=True,
        require_top_level_return=True,
    )["passed"]


def test_canonical_safe_head_target_is_reachable_token_by_token() -> None:
    body = "if isinstance(data, (list, tuple)) and data:\n    return data[0]\nreturn other"
    target_mode = semantic_ir.PLAN_BODY_TARGET_MODE
    tokens = target_tokens(body, target_mode=target_mode)
    vocab = build_target_vocab([body], max_vocab=512, target_mode=target_mode)
    inverse = {index: token for token, index in vocab.items()}
    full = [vocab["<bos>"], *[vocab[token] for token in tokens], vocab["<eos>"]]
    body_start = tokens.index("SLOT:BODY_START")

    for position in range(body_start + 1, len(tokens) + 1):
        target_id = full[position + 1] if position < len(tokens) else vocab["<eos>"]
        probabilities = np.full(len(vocab), 1e-12, dtype=np.float64)
        probabilities[target_id] = 1.0
        choices = mlx_token_choices(
            probabilities,
            inverse,
            vocab,
            full[: position + 1],
            eos_id=vocab["<eos>"],
            grammar_top_k=64,
            max_choices=2,
            token_policy="strict_body_token_legality_v1",
            target_mode=target_mode,
            allowed_names={"data", "other"},
            input_type_hints={},
            source_condition_expectation={},
            plan_prefix_choices=None,
            slot_prefix_probs=None,
            enable_learned_expression_token_bias=False,
            require_parameter_use=True,
            require_nontrivial_return=True,
            require_top_level_return=True,
            prefer_learned_prefix_decision_adequacy=False,
            prefer_source_condition_adequacy=False,
            block_shallow_loop_identity_update=False,
            enable_loop_progress_guard=False,
            enable_expression_closure_guard=False,
            enable_expression_value_guard=False,
            enable_semantic_operation_value_construction=False,
            require_binding_prefix_groups=False,
        )
        assert target_id in [token_id for token_id, _probability in choices], (
            position,
            inverse[target_id],
            [inverse[token_id] for token_id, _probability in choices],
        )


def test_parameter_role_classification_fails_unknown_tensors_into_core() -> None:
    assert parameter_is_optional_auxiliary("plan_router.weight")
    assert parameter_is_optional_auxiliary("body_state_event_router.bias")
    assert not parameter_is_optional_auxiliary("encoder.layers.0.attention.query_proj.weight")
    assert not parameter_is_optional_auxiliary("new_unclassified_head.weight")


def test_body_contrast_mask_is_bounded_after_body_start() -> None:
    target = np.array([[1, 8, 9, 42, 10, 11, 12, 2]], dtype=np.int32)
    target_out = target[:, 1:]
    mask = body_start_span_mask_mlx(target, target_out, 42, np, span_token_count=2)
    assert mask.tolist() == [[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]


def test_semantic_plan_prefix_does_not_consume_frozen_body_budget() -> None:
    broad = {"max_target_tokens": 112}

    assert broad_private_decode_token_budget(
        max_target=256,
        broad_config=broad,
        target_mode="body_tokens",
    ) == 112
    assert broad_private_decode_token_budget(
        max_target=256,
        broad_config=broad,
        target_mode=semantic_ir.PLAN_BODY_TARGET_MODE,
    ) == 112 + semantic_ir.PLAN_MAX_TOKENS + 1
