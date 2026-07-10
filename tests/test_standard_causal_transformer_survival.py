from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from standard_causal_transformer_model import CausalTransformerConfig, build_model
from standard_causal_transformer_survival import (
    beam_rank_score,
    completion_pool_target,
    assign_body_balanced_sampling_weights,
    normalized_sampling_probabilities,
    phase_target_positions,
    prune_complete_beams,
    render_visible_signature,
    semantic_stage_source,
    select_family_disjoint_eval,
    training_callable_signature,
    training_targets_complete,
    validate_config,
    visible_eval_source,
)
from blind_information_flow_audit import audit_source
from standard_causal_transformer_conditioning import deranged_source_arrays


def test_decoder_is_causal_and_cached_decode_matches_full_decode() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(11)
    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
    )
    model = build_model(config, mx=mx, nn=nn)
    prefix_a = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    prefix_b = mx.array([[1, 2, 3, 9]], dtype=mx.int32)
    logits_a, _ = model(prefix_a)
    logits_b, _ = model(prefix_b)
    mx.eval(logits_a, logits_b)
    assert bool(mx.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-5))

    prefill, cache = model(prefix_a[:, :3])
    cached, _ = model(prefix_a[:, 3:], cache)
    full, _ = model(prefix_a)
    mx.eval(prefill, cached, full)
    assert bool(mx.allclose(cached[:, -1], full[:, -1], atol=1e-4))


def test_visible_eval_source_does_not_read_solution_or_tests() -> None:
    row = {
        "prompt": "Return the input length.",
        "entry_point": "length",
        "solution_body": "SECRET_SOLUTION",
        "tests": "SECRET_TEST",
        "decoder_contract": {"visible_arg_count_hint": 1},
    }
    visible = visible_eval_source(row)
    assert visible == "Return the input length.\nsignature def length(data):"
    assert "SECRET" not in visible
    changed_hidden = {
        **row,
        "solution_body": "DIFFERENT_SECRET",
        "tests": "DIFFERENT_TEST",
        "concept_residual_label": "answer_identifying_family",
        "decoder_contract": {
            "visible_arg_count_hint": 1,
            "return_shape": "forbidden_hidden_shape",
            "required_constructs": ["forbidden_hidden_construct"],
        },
    }
    assert visible_eval_source(changed_hidden) == visible


def test_empty_body_is_rejected_instead_of_receiving_fallback() -> None:
    with pytest.raises(ValueError, match="empty body"):
        render_visible_signature("def solve(data):", "")


def test_family_disjoint_split_is_deterministic_and_nonempty() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    rows_a, families_a = select_family_disjoint_eval(config)
    rows_b, families_b = select_family_disjoint_eval(config)
    assert families_a == families_b
    assert len(families_a) == config["evaluation"]["holdout_family_count"]
    assert len(rows_a) == len(rows_b) == len(families_a) * config["evaluation"]["rows_per_family"]
    assert {row["concept_residual_label"] for row in rows_a} == set(families_a)
    assert len({(row["prompt"], row["solution_body"]) for row in rows_a}) == len(rows_a) == 24


def test_config_rejects_fallback_or_external_inference() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    validate_config(config)
    bad = json.loads(json.dumps(config))
    bad["boundaries"]["fallback_returns_allowed"] = True
    with pytest.raises(ValueError, match="fallback"):
        validate_config(bad)

    thin = json.loads(json.dumps(config))
    thin["evaluation"]["holdout_family_count"] = 6
    thin["evaluation"]["rows_per_family"] = 4
    with pytest.raises(ValueError, match="24 distinct families"):
        validate_config(thin)


def test_blind_audit_inspects_the_new_inference_surface(tmp_path: Path) -> None:
    violating = tmp_path / "violating_generator.py"
    violating.write_text(
        "def visible_eval_source(row):\n"
        "    return row.get('solution_body')\n",
        encoding="utf-8",
    )
    result = audit_source(violating)
    assert result["violation_count"] == 1
    assert result["violations"][0]["field"] == "solution_body"

    clean = audit_source(ROOT / "scripts" / "standard_causal_transformer_survival.py")
    assert clean["violation_count"] == 0


def test_source_derangement_keeps_targets_and_changes_sources() -> None:
    import numpy as np

    inputs = np.array(
        [
            [1, 10, 2, 20, 21, 0],
            [1, 11, 12, 2, 20, 22],
        ],
        dtype=np.int32,
    )
    labels = np.array(
        [
            [10, 2, 20, 21, 23, 0],
            [11, 12, 2, 20, 22, 23],
        ],
        dtype=np.int32,
    )
    mask = np.array(
        [
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    negative_inputs, negative_labels, negative_mask = deranged_source_arrays(inputs, labels, mask, seed=7)
    assert list(negative_inputs[0, :3]) == [1, 11, 12]
    assert 2 in negative_inputs[0]
    assert list(negative_labels[0][negative_mask[0] > 0]) == [21, 23]
    assert list(negative_labels[1][negative_mask[1] > 0]) == [22, 23]


def test_semantic_stage_source_rejects_placeholders_and_strips_metadata() -> None:
    placeholder, placeholder_audit = semantic_stage_source(
        {
            "function": "foo",
            "source_text": "Implement Python function foo.\nfoo\nvisible_intent_tags intent_list\nsignature def foo(data):",
        }
    )
    assert placeholder == ""
    assert placeholder_audit["placeholder"] is True

    semantic, semantic_audit = semantic_stage_source(
        {
            "function": "dedupe",
            "source_text": (
                "Return unique input values in their original order while preserving the first occurrence.\n"
                "dedupe\nvisible_intent_tags intent_collection\n"
                "prompt_operation_hints op_stable_dedup\nsignature def dedupe(data):"
            ),
        }
    )
    assert semantic.startswith("Return unique input values")
    assert "visible_intent_tags" not in semantic
    assert "prompt_operation_hints" not in semantic
    assert semantic.endswith("signature def dedupe(data):")
    assert semantic_audit == {"placeholder": False, "too_short": False, "metadata_tagged": True}


def test_finished_beam_pool_does_not_collapse_to_fanout_size() -> None:
    config = {"fanout": 4, "completion_pool_multiplier": 8}
    assert completion_pool_target(config) == 32
    rows = [
        {"tokens": [f"token_{index}"], "score": -float(index + 1)}
        for index in range(40)
    ]
    retained = prune_complete_beams(rows, limit=completion_pool_target(config), length_penalty=0.7)
    assert len(retained) == 32
    assert beam_rank_score(retained[0], 0.7) >= beam_rank_score(retained[-1], 0.7)


def test_private_callable_signature_repairs_stale_hint_from_executable_contract() -> None:
    signature, receipt = training_callable_signature(
        {
            "entry_point": "pair_sum",
            "solution_body": "return data + other",
            "tests": "assert pair_sum(2, 3) == 5\n",
            "decoder_contract": {"visible_arg_count_hint": 1},
        }
    )
    assert signature == "def pair_sum(data, other):"
    assert receipt == {"source": "private_tests", "arity": 2}


def test_family_disjoint_eval_freezes_normalized_callable_signature() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    rows, _families = select_family_disjoint_eval(config)
    lcs = next(row for row in rows if row["concept_residual_label"] == "bpg_lcs_length")
    assert lcs["callable_signature"].endswith("(data, other):")
    assert lcs["callable_signature_receipt"] == {"source": "private_tests", "arity": 2}
    assert visible_eval_source(lcs).endswith("signature " + lcs["callable_signature"])


def test_private_prompt_variants_share_fixed_body_sampling_mass() -> None:
    examples = [
        {"source": "licensed_function", "source_text": "licensed", "body": "return data"},
        {"source": "governed_private", "source_text": "first", "body": "return data + 1"},
        {"source": "governed_private", "source_text": "second", "body": "return data + 1"},
        {"source": "governed_private", "source_text": "third", "body": "return data - 1"},
    ]
    weighted, audit = assign_body_balanced_sampling_weights(examples, private_body_weight=16.0)
    assert weighted[0]["sampling_weight"] == 1.0
    assert weighted[1]["sampling_weight"] == weighted[2]["sampling_weight"] == 8.0
    assert weighted[3]["sampling_weight"] == 16.0
    assert audit["private_sampling_mass"] == 32.0
    probabilities = normalized_sampling_probabilities(
        np.array([row["sampling_weight"] for row in weighted]), len(weighted)
    )
    assert probabilities is not None
    assert float(probabilities.sum()) == pytest.approx(1.0)


def test_training_completion_uses_consumed_positions_not_estimated_steps() -> None:
    config = {
        "pretrain_target_token_positions": 100,
        "sft_target_token_positions": 50,
    }
    reports = [
        {"phase": "licensed_module_causal_pretraining", "target_positions_consumed": 100},
        {"phase": "prompt_signature_body_sft", "target_positions_consumed": 49},
    ]
    assert phase_target_positions(reports, "prompt_signature_body_sft") == 49
    assert training_targets_complete(reports, config) is False
    reports.append(
        {"phase": "prompt_signature_body_sft_continuation", "target_positions_consumed": 1}
    )
    assert training_targets_complete(reports, config) is True
