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
    batched_beam_advance,
    completion_pool_target,
    assign_body_balanced_sampling_weights,
    normalized_sampling_probabilities,
    phase_target_positions,
    prune_complete_beams,
    render_visible_signature,
    semantic_stage_source,
    select_family_disjoint_eval,
    select_preference_train_rows,
    stage_signature,
    training_callable_signature,
    training_targets_complete,
    validate_config,
    visible_eval_source,
)
from blind_information_flow_audit import audit_source
from standard_causal_transformer_conditioning import deranged_source_arrays
from code_lm_private_verifier import evaluate_all_private_candidates
from standard_causal_transformer_preference import (
    PreferenceArrays,
    build_preference_pairs,
    reward_removed_pairs,
    train_dpo,
)
from standard_causal_transformer_survival_gate import (
    audit_generation_mode_canary,
    audit_preference_canary,
)
from generation_mode_gate import audit_comparison, read_report_ref
from policy_optimization_gate import extract_behavior_metrics, summarize_behavior_evidence


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


def test_batched_beam_advance_matches_serial_cached_model_calls() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(13)
    config = CausalTransformerConfig(vocab_size=64, d_model=32, num_layers=2, num_heads=4, num_kv_heads=2, ff_dim=64)
    model = build_model(config, mx=mx, nn=nn)
    prefix = mx.array([[1, 2, 3]], dtype=mx.int32)
    logits, cache = model(prefix)
    mx.eval(logits, *[value for pair in cache for value in pair])
    beam = {"tokens": ["x"], "score": -0.5, "cache": cache, "logits": logits[0, -1]}
    specs = [
        {"beam": beam, "local_id": 4, "token": "a", "log_probability": -0.1},
        {"beam": beam, "local_id": 5, "token": "b", "log_probability": -0.2},
    ]
    batched = batched_beam_advance(model, specs, target_offset=10, mx=mx)
    for index, spec in enumerate(specs):
        serial_logits, serial_cache = model(
            mx.array([[10 + spec["local_id"]]], dtype=mx.int32), cache
        )
        mx.eval(serial_logits, *[value for pair in serial_cache for value in pair])
        assert bool(mx.allclose(batched[index]["logits"], serial_logits[0, -1], atol=1e-5))
        for (batch_key, batch_value), (serial_key, serial_value) in zip(
            batched[index]["cache"], serial_cache
        ):
            assert bool(mx.allclose(batch_key, serial_key, atol=1e-5))
            assert bool(mx.allclose(batch_value, serial_value, atol=1e-5))


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


def test_stage_signature_ignores_decode_only_knobs_but_tracks_staging_contract() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    baseline = stage_signature(config)
    decode_only = json.loads(json.dumps(config))
    decode_only["evaluation"]["beam_width"] += 4
    decode_only["preference"]["optimizer_steps"] += 1
    assert stage_signature(decode_only) == baseline

    staging_change = json.loads(json.dumps(config))
    staging_change["tokenization"]["max_source_tokens"] += 1
    assert stage_signature(staging_change) != baseline


def test_preference_rows_are_private_verifier_bearing_and_eval_disjoint() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    eval_rows, holdout_families = select_family_disjoint_eval(config)
    rows, audit = select_preference_train_rows(
        config,
        holdout_families=holdout_families,
        eval_prompt_hashes={__import__("hashlib").sha256(row["prompt"].encode()).hexdigest() for row in eval_rows},
        eval_body_hashes={__import__("hashlib").sha256(row["solution_body"].encode()).hexdigest() for row in eval_rows},
    )
    assert len(rows) == config["preference"]["max_train_tasks"]
    assert all(row["split"] == "eval" and row["tests"] and row["callable_signature"] for row in rows)
    assert not ({row["concept_residual_label"] for row in rows} & set(holdout_families))
    assert audit["train_eval_family_overlap_count"] == 0
    assert audit["train_eval_prompt_overlap_count"] == 0
    assert audit["train_eval_body_overlap_count"] == 0


def test_preference_pairs_require_exact_pass_and_hard_negative() -> None:
    tasks = [{"task_id": "task-a", "prompt": "p", "callable_signature": "def solve(data):"}]
    candidates = [
        {"task_id": "task-a", "candidate_sha256": "pass", "code": "def solve(data):\n    return len(data)"},
        {"task_id": "task-a", "candidate_sha256": "fail", "code": "def solve(data):\n    return data"},
    ]
    verifier = {
        "verification_attempt_labels": [
            {
                "task_id": "task-a",
                "phase": "private_eval",
                "candidate_sha256": "fail",
                "passed": False,
                "verification_reward": 0.7,
                "verification_stage": "runtime_loaded",
                "rank_score": -1.0,
                "semantic_ir_state": "READY",
            },
            {
                "task_id": "task-a",
                "phase": "private_eval",
                "candidate_sha256": "pass",
                "passed": True,
                "verification_reward": 1.0,
                "verification_stage": "intended_behavior_passed",
                "rank_score": -2.0,
                "semantic_ir_state": "READY",
            },
        ]
    }
    pairs, summary = build_preference_pairs(tasks, candidates, verifier, max_pairs=4, seed=3)
    assert len(pairs) == 1
    assert pairs[0]["chosen"]["candidate_sha256"] == "pass"
    assert pairs[0]["rejected"]["candidate_sha256"] == "fail"
    assert summary["selected_pair_count"] == summary["semantic_ir_ready_pair_count"] == 1


def test_reward_removed_control_has_identically_zero_pair_signal() -> None:
    pairs = [
        {"pair_id": str(index), "chosen": f"c{index}", "rejected": f"r{index}", "chosen_reward": 1.0, "rejected_reward": 0.0, "chosen_stage": "pass", "rejected_stage": "fail"}
        for index in range(4)
    ]
    control, summary = reward_removed_pairs(pairs, seed=9)
    assert summary == {
        "pair_count": 4,
        "zero_reward_pair_count": 4,
        "all_pair_margins_identically_zero": True,
        "verifier_direction_available_to_control": False,
    }
    assert all(row["chosen"] == row["rejected"] for row in control)
    assert all(row["chosen_reward"] == row["rejected_reward"] for row in control)


def test_tiny_dpo_update_increases_verifier_preference_margin() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(17)
    cfg = CausalTransformerConfig(vocab_size=32, d_model=16, num_layers=1, num_heads=4, num_kv_heads=2, ff_dim=32)
    reference = build_model(cfg, mx=mx, nn=nn)
    policy = build_model(cfg, mx=mx, nn=nn)
    policy.update(reference.parameters())
    arrays = PreferenceArrays(
        chosen_inputs=np.array([[1, 4, 5, 6], [1, 7, 8, 9]], dtype=np.int32),
        chosen_labels=np.array([[4, 5, 6, 10], [7, 8, 9, 11]], dtype=np.int32),
        chosen_mask=np.ones((2, 4), dtype=np.float32),
        rejected_inputs=np.array([[1, 4, 5, 12], [1, 7, 8, 13]], dtype=np.int32),
        rejected_labels=np.array([[4, 5, 12, 14], [7, 8, 13, 15]], dtype=np.int32),
        rejected_mask=np.ones((2, 4), dtype=np.float32),
    )
    report = train_dpo(
        policy,
        reference,
        arrays,
        optimizer_steps=6,
        batch_size=2,
        learning_rate=1e-3,
        beta=0.2,
        gradient_clip_norm=1.0,
        seed=5,
        mx=mx,
        nn=nn,
        optim=optim,
    )
    assert report["state"] == "TRAINED"
    assert report["preference_margin_delta"] > 0
    assert report["public_training_rows_written"] == report["external_inference_calls"] == 0


def test_all_candidate_preference_labeling_does_not_stop_after_first_pass() -> None:
    task = {
        "task_id": "private-pref",
        "entry_point": "solve",
        "category": "private",
        "concept_residual_label": "private_preference_test",
        "split": "eval",
        "tests": "assert solve([1, 2, 3]) == 3\n",
    }
    candidates = [
        {
            "task_id": "private-pref",
            "entry_point": "solve",
            "phase": "private_eval",
            "rank": 1,
            "candidate_sha256": "pass",
            "code": "def solve(data):\n    return len(data)",
        },
        {
            "task_id": "private-pref",
            "entry_point": "solve",
            "phase": "private_eval",
            "rank": 2,
            "candidate_sha256": "fail",
            "code": "def solve(data):\n    return data",
        },
    ]
    result = evaluate_all_private_candidates([task], candidates)
    labels = result["verification_attempt_labels"]
    assert len(labels) == 2
    assert {row["candidate_sha256"]: row["passed"] for row in labels} == {
        "pass": True,
        "fail": False,
    }
    assert result["uses_eval_tests_or_solutions_for_generation"] is False


def test_preference_gate_rejects_margin_only_behavior_regression_promotion() -> None:
    canary = {
        "state": "GREEN",
        "adoption_state": "NOT_ADOPTED",
        "reward_improves_behavior": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "preference_pair_summary": {"selected_pair_count": 1},
        "reward_present_training": {"preference_margin_delta": 1.8},
        "reward_removed_training": {"preference_margin_delta": 0.0},
        "base_heldout": {"passed_task_count": 1, "rank1_passed_task_count": 0},
        "reward_present_heldout": {"passed_task_count": 0, "rank1_passed_task_count": 0, "integrity_mismatch_count": 0},
        "reward_removed_heldout": {"passed_task_count": 1, "rank1_passed_task_count": 0, "integrity_mismatch_count": 0},
        "artifacts": {"reward_checkpoint": "checkpoints/reward.npz", "control_checkpoint": "checkpoints/control.npz"},
    }
    clean = audit_preference_canary(canary, ROOT / "checkpoints" / "canonical.npz")
    assert clean["hard_gaps"] == []
    assert clean["reward_behavior_delta"] == -1

    canary["adoption_state"] = "QUALIFIED_SHADOW"
    invalid = audit_preference_canary(canary, ROOT / "checkpoints" / "canonical.npz")
    assert "preference_adoption_state_mismatch" in {row["kind"] for row in invalid["hard_gaps"]}


def test_generation_mode_gate_recomputes_speed_and_nonregression() -> None:
    canary = {
        "state": "GREEN",
        "adoption_state": "BATCHED_DEFAULT",
        "candidate_manifest_equal": True,
        "behavior_non_regression": True,
        "integrity_non_regression": True,
        "generation_speedup": 2.5,
        "serial": {"generation_runtime_ms": 1000, "passed_task_count": 1, "rank1_passed_task_count": 0, "integrity_mismatch_count": 0},
        "batched": {"generation_runtime_ms": 400, "passed_task_count": 1, "rank1_passed_task_count": 0, "integrity_mismatch_count": 0},
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    audit = audit_generation_mode_canary(canary)
    assert audit["hard_gaps"] == []
    assert audit["speedup"] == 2.5

    canary["batched"]["passed_task_count"] = 0
    invalid = audit_generation_mode_canary(canary)
    kinds = {row["kind"] for row in invalid["hard_gaps"]}
    assert "generation_mode_behavior_decision_mismatch" in kinds
    assert "generation_mode_adoption_state_mismatch" in kinds


def test_policy_gate_treats_margin_only_preference_regression_as_negative_evidence() -> None:
    payload = {
        "preference_canary": {
            "adoption_state": "NOT_ADOPTED",
            "reward_improves_behavior": False,
            "preference_pair_summary": {"selected_pair_count": 1},
            "reward_present_training": {"preference_margin_delta": 1.8},
            "reward_removed_training": {"preference_margin_delta": 0.0},
            "base_heldout": {"passed_task_count": 1},
            "reward_present_heldout": {"passed_task_count": 0, "integrity_mismatch_count": 0},
            "reward_removed_heldout": {"passed_task_count": 1},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
    }
    metrics = extract_behavior_metrics(payload)
    evidence = [{"path": "synthetic.json", "present": True, "behavior_metrics": metrics}]
    summary = summarize_behavior_evidence(evidence)
    assert metrics["preference_reward_behavior_delta"] == -1
    assert summary["has_behavior_lift"] is False
    assert summary["has_loss_only_lift"] is True
    assert summary["behavior_regression_refs"] == ["synthetic.json"]


def test_generation_registry_reads_nested_same_report_modes_and_requires_parity(tmp_path: Path) -> None:
    report_path = tmp_path / "mode_report.json"
    report_path.write_text(
        json.dumps(
            {
                "trigger_state": "GREEN",
                "summary": {
                    "family_disjoint_eval_task_count": 24,
                    "public_training_rows": 0,
                    "external_inference_calls": 0,
                    "fallback_return_count": 0,
                },
                "generation_mode_canary": {
                    "candidate_manifest_equal": True,
                    "public_training_rows_written": 0,
                    "external_inference_calls": 0,
                    "serial": {
                        "generation_runtime_ms": 1000,
                        "candidate_count": 8,
                        "passed_task_count": 1,
                        "integrity_mismatch_count": 0,
                        "accepted_verified_output_per_second": 1.0,
                    },
                    "batched": {
                        "generation_runtime_ms": 400,
                        "candidate_count": 8,
                        "passed_task_count": 1,
                        "integrity_mismatch_count": 0,
                        "accepted_verified_output_per_second": 2.5,
                    },
                },
            }
        )
    )
    serial = read_report_ref(str(report_path), metric_path="generation_mode_canary.serial")
    batched = read_report_ref(str(report_path), metric_path="generation_mode_canary.batched")
    assert serial["metric_path_present"] and batched["metric_path_present"]
    assert serial["metrics"]["task_count"] == batched["metrics"]["task_count"] == 24
    assert serial["metrics"]["candidate_manifest_equal"] is True
    modes = {
        "serial": {"metrics": serial["metrics"]},
        "batched": {"metrics": batched["metrics"]},
    }
    comparison = audit_comparison(
        {"id": "same-report", "baseline_mode_id": "serial", "candidate_mode_id": "batched"},
        modes,
    )
    assert comparison["promotable"] is True

    modes["batched"]["metrics"]["candidate_manifest_equal"] = False
    assert audit_comparison(
        {"id": "no-parity", "baseline_mode_id": "serial", "candidate_mode_id": "batched"},
        modes,
    )["promotable"] is False
