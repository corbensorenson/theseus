from __future__ import annotations

import json
import hashlib
import subprocess
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
    EXECUTABLE_STATE_ROLES,
    beam_rank_score,
    batched_beam_advance,
    cache_arrays,
    completion_pool_target,
    canonical_model_signature,
    compare_target_mode_canaries,
    encode_model_source,
    executable_state_role_lookup,
    executable_state_token_roles,
    extend_target_vocab_for_mode,
    generation_prefix_complete,
    assign_body_balanced_sampling_weights,
    normalized_sampling_probabilities,
    phase_target_positions,
    prune_complete_beams,
    render_visible_signature,
    semantic_stage_source,
    select_family_disjoint_eval,
    select_preference_train_rows,
    standalone_sft_contract_decision,
    source_token_offset,
    stage_materialization_lock,
    stage_signature,
    target_token_offset,
    training_target_tokens,
    training_callable_signature,
    training_targets_complete,
    validate_config,
    visible_eval_source,
)
from blind_information_flow_audit import audit_source
from candidate_integrity import recompute_candidate_integrity
from standard_causal_transformer_conditioning import (
    deranged_source_arrays,
    inspect_bindings as inspect_conditioning_bindings,
    publish_completed_checkpoint,
    validate_config as validate_conditioning_config,
)
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
    audit_sft_contract_admission,
    audit_target_mode_comparison,
)
from generation_mode_gate import audit_comparison, read_report_ref
from policy_optimization_gate import extract_behavior_metrics, summarize_behavior_evidence
from code_lm_decoder_contracts import visible_arg_count_hint_for_task
from broad_private_generalization_ladder_v1 import row_from_template, template_bank


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
    mx.eval(logits, *cache_arrays(cache))
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
        mx.eval(serial_logits, *cache_arrays(serial_cache))
        assert bool(mx.allclose(batched[index]["logits"], serial_logits[0, -1], atol=1e-5))
        for (batch_key, batch_value), (serial_key, serial_value) in zip(
            batched[index]["cache"], serial_cache
        ):
            assert bool(mx.allclose(batch_key, serial_key, atol=1e-5))
            assert bool(mx.allclose(batch_value, serial_value, atol=1e-5))


def test_executable_state_role_lookup_has_matched_semantic_and_hash_density() -> None:
    base = {
        "model": {"state_memory_slots": len(EXECUTABLE_STATE_ROLES)},
        "tokenization": {"shared_source_target_vocabulary": False},
    }
    source_vocab = {"walk": 0, "return": 1, "value": 2}
    target_vocab = {"NAME:for": 0, "NAME:return": 1, "OP:+=": 2}
    semantic = executable_state_role_lookup(
        {**base, "model": {**base["model"], "state_memory_mode": "semantic_roles"}},
        source_vocab,
        target_vocab,
    )
    hashed = executable_state_role_lookup(
        {**base, "model": {**base["model"], "state_memory_mode": "hash_control"}},
        source_vocab,
        target_vocab,
    )
    assert semantic is not None and hashed is not None
    assert semantic.shape == hashed.shape
    assert np.array_equal(semantic.sum(axis=1), hashed.sum(axis=1))
    assert not np.array_equal(semantic, hashed)
    assert executable_state_token_roles("NAME:return") == ("return_closure",)
    assert "traversal" in executable_state_token_roles("walk")
    assert "state_update" in executable_state_token_roles("OP:+=")


def test_executable_state_memory_is_causal_and_cache_partition_invariant() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(29)
    slots = len(EXECUTABLE_STATE_ROLES)
    lookup = np.zeros((64, slots), dtype=np.float32)
    for token_id in range(64):
        lookup[token_id, token_id % slots] = 1.0
    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        state_memory_slots=slots,
        state_memory_chunk_size=2,
        state_memory_local_window=4,
        state_memory_mode="semantic_roles",
    )
    model = build_model(config, mx=mx, nn=nn, state_role_lookup=lookup)
    sequence_a = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    sequence_b = mx.array([[1, 2, 3, 4, 9]], dtype=mx.int32)
    logits_a, _ = model(sequence_a)
    logits_b, _ = model(sequence_b)
    mx.eval(logits_a, logits_b)
    assert bool(mx.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-5))

    prefix_logits, cache = model(sequence_a[:, :3])
    cached_logits, cached_state = model(sequence_a[:, 3:], cache)
    full_logits, full_state = model(sequence_a)
    mx.eval(prefix_logits, cached_logits, full_logits, *cache_arrays(cached_state), *cache_arrays(full_state))
    assert bool(mx.allclose(cached_logits, full_logits[:, 3:], atol=1e-4))
    for cached_layer, full_layer in zip(cached_state, full_state):
        for cached_value, full_value in zip(cached_layer, full_layer):
            assert bool(mx.allclose(cached_value, full_value, atol=1e-4))


def test_structured_and_hash_state_models_are_parameter_matched() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    slots = len(EXECUTABLE_STATE_ROLES)
    lookup = np.eye(slots, dtype=np.float32)[np.arange(64) % slots]
    common = dict(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        state_memory_slots=slots,
        state_memory_chunk_size=2,
        state_memory_local_window=4,
    )
    semantic = build_model(
        CausalTransformerConfig(**common, state_memory_mode="semantic_roles"),
        mx=mx,
        nn=nn,
        state_role_lookup=lookup,
    )
    control = build_model(
        CausalTransformerConfig(**common, state_memory_mode="hash_control"),
        mx=mx,
        nn=nn,
        state_role_lookup=lookup,
    )
    semantic_count = sum(value.size for _name, value in mlx_utils.tree_flatten(semantic.parameters()))
    control_count = sum(value.size for _name, value in mlx_utils.tree_flatten(control.parameters()))
    assert semantic_count == control_count


def test_visible_eval_source_does_not_read_solution_or_tests() -> None:
    row = {
        "prompt": "Return the input length.",
        "entry_point": "length",
        "solution_body": "SECRET_SOLUTION",
        "tests": "SECRET_TEST",
        "decoder_contract": {"visible_arg_count_hint": 1},
    }
    visible = visible_eval_source(row)
    assert visible == (
        "Return the input length.\n"
        "signature def solve(data=None, other=None, *extra):"
    )
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
    target_limit = config["tokenization"]["max_target_tokens"] - 2
    from neural_seed_token_decoder_support import body_tokens

    assert max(len(body_tokens(row["solution_body"])) for row in rows_a) <= target_limit
    assert config["evaluation"]["decode_max_target_tokens"] >= max(
        len(body_tokens(row["solution_body"])) for row in rows_a
    )


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

    missing_vocab_mode = json.loads(json.dumps(config))
    del missing_vocab_mode["tokenization"]["shared_source_target_vocabulary"]
    with pytest.raises(ValueError, match="explicitly boolean"):
        validate_config(missing_vocab_mode)

    task_named = json.loads(json.dumps(config))
    task_named["tokenization"]["canonical_model_signature_name"] = "task_name"
    with pytest.raises(ValueError, match="must remain solve"):
        validate_config(task_named)

    invalid_plan = json.loads(json.dumps(config))
    invalid_plan["tokenization"]["target_mode"] = "typed_semantic_ir_plan_body_tokens_v1"
    invalid_plan["tokenization"]["semantic_plan_max_tokens"] = 4
    with pytest.raises(ValueError, match="semantic plan token budget"):
        validate_config(invalid_plan)


def test_standard_transformer_plan_body_target_is_learned_and_closed_vocab() -> None:
    import semantic_ir
    from neural_seed_token_decoder_rendering import PLAN_BODY_START_TOKEN
    from neural_seed_token_decoder_support import body_tokens as tokenize_body

    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    config["tokenization"]["target_mode"] = semantic_ir.PLAN_BODY_TARGET_MODE
    config["tokenization"]["semantic_plan_max_tokens"] = 32
    body = "out = []\nfor value in data:\n    out.append(value + 1)\nreturn out"
    tokens = training_target_tokens(body, config)
    assert tokens[0] == semantic_ir.PLAN_BEGIN
    assert PLAN_BODY_START_TOKEN in tokens
    assert tokens[tokens.index(PLAN_BODY_START_TOKEN) + 1 :] == tokenize_body(body)
    assert generation_prefix_complete(tokens, target_mode=semantic_ir.PLAN_BODY_TARGET_MODE)

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "NAME:out": 4}
    summary = extend_target_vocab_for_mode(config, vocab)
    assert summary["target_independent_closed_protocol"] is True
    assert summary["added_token_count"] == len(semantic_ir.plan_protocol_tokens()) + 1
    assert set(semantic_ir.plan_protocol_tokens()) <= set(vocab)
    assert PLAN_BODY_START_TOKEN in vocab


def test_matched_target_mode_comparison_rejects_loss_only_plan_improvement() -> None:
    common_config = {
        "seed": 1,
        "sources": {"private": "x"},
        "model": {"d_model": 8},
        "training": {"pretrain_target_token_positions": 10, "sft_target_token_positions": 20},
        "evaluation": {"holdout_family_count": 24},
        "preference": {"max_train_tasks": 8},
        "boundaries": {"public_training_rows": 0},
        "tokenization": {
            "target_mode": "body_tokens",
            "semantic_plan_max_tokens": 0,
            "max_source_tokens": 32,
        },
    }

    def report(mode: str, loss: float, candidates: int, tasks: int, reward: float) -> dict:
        return {
            "stage": {"target_mode": mode},
            "architecture": {"parameter_count": 100},
            "training": {"complete": True, "eval_loss_after": loss},
            "summary": {"candidate_count": candidates, "candidate_task_count": tasks},
            "decode": {"runtime_ms": 10},
            "private_verifier": {
                "summary": {"passed_task_count": 0},
                "private_verification": {"mean_verification_reward": reward},
            },
            "runtime_ms": 20,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }

    plan_config = json.loads(json.dumps(common_config))
    plan_config["tokenization"]["target_mode"] = "typed_semantic_ir_plan_body_tokens_v1"
    plan_config["tokenization"]["semantic_plan_max_tokens"] = 32
    control_config = json.loads(json.dumps(common_config))
    clean_integrity = {"summary": {"integrity_mismatch_count": 0}}
    result = compare_target_mode_canaries(
        report("typed_semantic_ir_plan_body_tokens_v1", 1.6, 4, 2, 0.2),
        report("body_tokens", 1.8, 8, 4, 0.4),
        plan_config=plan_config,
        control_config=control_config,
        plan_integrity=clean_integrity,
        control_integrity=clean_integrity,
    )
    assert result["trigger_state"] == "GREEN"
    assert result["adoption_state"] == "NOT_ADOPTED"
    assert result["deltas"]["candidate_task_count"] == -2
    assert "no_verifier_pass_gain" in result["adoption_rejection_reasons"]


def test_matched_target_mode_comparison_rejects_non_target_config_drift() -> None:
    config = {
        "seed": 1,
        "sources": {"private": "x"},
        "tokenization": {"target_mode": "body_tokens", "max_source_tokens": 32},
        "model": {"d_model": 8},
        "training": {"pretrain_target_token_positions": 10},
        "evaluation": {"holdout_family_count": 24},
        "preference": {"max_train_tasks": 8},
        "boundaries": {"public_training_rows": 0},
    }
    plan_config = json.loads(json.dumps(config))
    plan_config["tokenization"]["target_mode"] = "typed_semantic_ir_plan_body_tokens_v1"
    control_config = json.loads(json.dumps(config))
    control_config["tokenization"]["max_source_tokens"] = 64
    plan_report = {
        "stage": {"target_mode": "typed_semantic_ir_plan_body_tokens_v1"},
        "architecture": {"parameter_count": 100},
        "training": {"complete": True, "eval_loss_after": 1.0},
        "summary": {"candidate_count": 1, "candidate_task_count": 1},
        "decode": {"runtime_ms": 1},
        "private_verifier": {
            "summary": {"passed_task_count": 1},
            "private_verification": {"mean_verification_reward": 1.0},
        },
        "runtime_ms": 1,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    control_report = json.loads(json.dumps(plan_report))
    control_report["stage"]["target_mode"] = "body_tokens"
    integrity = {"summary": {"integrity_mismatch_count": 0}}
    result = compare_target_mode_canaries(
        plan_report,
        control_report,
        plan_config=plan_config,
        control_config=control_config,
        plan_integrity=integrity,
        control_integrity=integrity,
    )
    assert result["trigger_state"] == "RED"
    assert result["matched_checks"]["tokenization_except_target_mode"] is False


def test_target_mode_gate_replays_artifact_bindings_and_adoption(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bound")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact_names = {
        "plan_report",
        "plan_config",
        "plan_candidates",
        "plan_checkpoint",
        "plan_integrity",
        "control_report",
        "control_config",
        "control_candidates",
        "control_checkpoint",
        "control_integrity",
    }
    comparison = {
        "policy": "project_theseus_standard_causal_target_mode_matched_comparison_v1",
        "trigger_state": "GREEN",
        "adoption_state": "NOT_ADOPTED",
        "adoption_rejection_reasons": ["no_verifier_pass_gain"],
        "boundaries_clean": True,
        "matched_checks": {"seed": True, "target_modes_expected": True},
        "plan": {
            "passed_task_count": 0,
            "candidate_task_count": 2,
            "mean_verification_reward": 0.2,
            "integrity_mismatch_count": 0,
        },
        "control": {
            "passed_task_count": 0,
            "candidate_task_count": 4,
            "mean_verification_reward": 0.4,
            "integrity_mismatch_count": 0,
        },
        "deltas": {"candidate_task_count": -2},
        "artifacts": {
            name: {"path": str(artifact), "sha256": digest, "bytes": 5}
            for name in artifact_names
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison), encoding="utf-8")
    audit = audit_target_mode_comparison(path)
    assert audit["state"] == "GREEN"
    assert audit["adoption_state"] == "NOT_ADOPTED"
    assert all(item["matches"] for item in audit["receipt"]["artifact_receipts"].values())

    comparison["adoption_state"] = "ADOPTED"
    path.write_text(json.dumps(comparison), encoding="utf-8")
    invalid = audit_target_mode_comparison(path)
    assert invalid["state"] == "RED"
    assert any(
        gap["kind"] == "target_mode_adoption_decision_mismatch"
        for gap in invalid["hard_gaps"]
    )


def test_sft_contract_admission_audit_retains_behavior_negative_result(tmp_path: Path) -> None:
    filtered_config = {
        "policy": "candidate",
        "model": {"d_model": 32},
        "training": {"positions": 100},
        "sft_contract_admission": {
            "require_self_contained_body": True,
            "private_sampling_probability_target": 0.25,
        },
    }
    control_config = {key: value for key, value in filtered_config.items() if key != "sft_contract_admission"}
    filtered_config_path = tmp_path / "filtered-config.json"
    control_config_path = tmp_path / "control-config.json"
    filtered_config_path.write_text(json.dumps(filtered_config), encoding="utf-8")
    control_config_path.write_text(json.dumps(control_config), encoding="utf-8")

    def report(config_path: Path, *, candidates: int, tasks: int, reward: float) -> dict[str, object]:
        return {
            "artifacts": {"config": str(config_path)},
            "stage": {
                "sft_contract_admission": {
                    "target_body_fields_added_to_model_source": 0,
                    "heldout_rows_read_by_filter": 0,
                }
            },
            "summary": {
                "model_only_passed_task_count": 0,
                "candidate_task_count": tasks,
                "candidate_count": candidates,
            },
            "private_verifier": {"private_verification": {"mean_verification_reward": reward}},
            "training": {"eval_loss_after": 1.0},
            "decode": {"runtime_ms": 100},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }

    filtered_report_path = tmp_path / "filtered-report.json"
    control_report_path = tmp_path / "control-report.json"
    integrity_path = tmp_path / "integrity.json"
    blind_path = tmp_path / "blind.json"
    filtered_report_path.write_text(
        json.dumps(report(filtered_config_path, candidates=3, tasks=2, reward=0.2)),
        encoding="utf-8",
    )
    control_report_path.write_text(
        json.dumps(report(control_config_path, candidates=5, tasks=4, reward=0.4)),
        encoding="utf-8",
    )
    integrity_path.write_text(
        json.dumps({"trigger_state": "GREEN", "summary": {"integrity_mismatch_count": 0}}),
        encoding="utf-8",
    )
    blind_path.write_text(
        json.dumps({"trigger_state": "GREEN", "summary": {"invalid_claim_count": 0}}),
        encoding="utf-8",
    )
    audit = audit_sft_contract_admission(
        report_path=filtered_report_path,
        integrity_path=integrity_path,
        blind_audit_path=blind_path,
        control_report_path=control_report_path,
    )
    assert audit["state"] == "GREEN"
    assert audit["adoption_state"] == "NOT_ADOPTED"
    assert audit["deltas"]["candidate_task_count"] == -2
    assert audit["receipt"]["matched_config_except_contract_admission"] is True


def test_sft_contract_admission_audit_is_not_run_when_local_evidence_is_absent(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    audit = audit_sft_contract_admission(
        report_path=missing,
        integrity_path=missing,
        blind_audit_path=missing,
        control_report_path=missing,
    )
    assert audit["state"] == "NOT_RUN"
    assert audit["hard_gaps"] == []


def test_candidate_integrity_recomputes_direct_plan_body_trace_instead_of_trusting_flags() -> None:
    import semantic_ir
    from neural_seed_token_decoder_rendering import PLAN_BODY_START_TOKEN
    from neural_seed_token_decoder_support import body_tokens as tokenize_body

    body = "out = []\nfor value in data:\n    out.append(value + 1)\nreturn out"
    code = "def solve(data):\n" + "\n".join(f"    {line}" for line in body.splitlines()) + "\n"
    tokens = [
        *semantic_ir.body_to_plan_tokens(body, max_tokens=32),
        PLAN_BODY_START_TOKEN,
        *tokenize_body(body),
    ]
    row = {
        "candidate_generation_mode": "direct_decoder_only_causal_semantic_plan_body_tokens",
        "candidate_source": "standard_causal_transformer_survival",
        "substrate_arm": "transformer_hybrid_survival",
        "code": code,
        "decoded_target_tokens": tokens,
        "decoded_token_sha256": hashlib.sha256(" ".join(tokens).encode()).hexdigest(),
        "provenance": {"candidate_family": "structural_adapter"},
    }
    verified = recompute_candidate_integrity(row)
    assert verified["recomputed_candidate_family"] == "transformer_hybrid"
    assert verified["direct_plan_body_trace"]["valid"] is True

    corrupted = json.loads(json.dumps(row))
    corrupted["decoded_target_tokens"][-2] = "NAME:wrong"
    corrupted["decoded_token_sha256"] = hashlib.sha256(
        " ".join(corrupted["decoded_target_tokens"]).encode()
    ).hexdigest()
    rejected = recompute_candidate_integrity(corrupted)
    assert rejected["recomputed_candidate_family"] == "structural_adapter"
    assert rejected["direct_plan_body_trace"]["valid"] is False
    assert "decoded_body_trace_code_mismatch" in rejected["direct_plan_body_trace"]["faults"]


def test_evaluate_only_requires_execute_before_any_report_write() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "standard_causal_transformer_survival.py"), "--evaluate-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--resume and --evaluate-only require --execute" in result.stderr


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
    assert list(negative_mask.sum(axis=1)) == list(mask.sum(axis=1))


def test_conditioning_config_forbids_canonical_report_overwrite() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_conditioning.json").read_text())
    validate_conditioning_config(config)
    unsafe = json.loads(json.dumps(config))
    unsafe["report"] = "reports/standard_causal_transformer_survival.json"
    with pytest.raises(ValueError, match="cannot overwrite"):
        validate_conditioning_config(unsafe)


def test_conditioning_preflight_rejects_stale_stage_and_incomplete_training(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_conditioning.json").read_text())
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    (stage_dir / "stage_arrays_v1.npz").write_bytes(b"not-loaded-by-preflight")
    (stage_dir / "stage_metadata_v1.json").write_text(json.dumps({"stage_signature": "stale"}))
    base_report = tmp_path / "base_report.json"
    base_report.write_text(
        json.dumps(
            {
                "artifacts": {
                    "config": config["base_config"],
                    "checkpoint": config["base_checkpoint"],
                    "stage_dir": str(stage_dir),
                },
                "stage": {"stage_signature": "stale"},
                "training": {"complete": False, "evaluation_only_replay": True},
            }
        )
    )
    config["stage_dir"] = str(stage_dir)
    config["stage_signature"] = "stale"
    config["base_report"] = str(base_report)
    config["report"] = str(tmp_path / "conditioning.json")
    bindings = inspect_conditioning_bindings(config)
    fault_codes = {row["code"] for row in bindings["binding_faults"]}
    blocker_codes = {row["code"] for row in bindings["training_blockers"]}
    assert "stage_logic_or_source_mismatch" in fault_codes
    assert "base_training_receipt_incomplete" in blocker_codes
    assert "base_training_receipt_is_evaluation_replay" in blocker_codes
    assert bindings["ready_for_measure"] is False
    assert bindings["ready_for_train"] is False


def test_conditioning_checkpoint_publication_is_atomic(tmp_path: Path) -> None:
    final = tmp_path / "checkpoint.npz"
    partial = tmp_path / "checkpoint.partial.npz"
    final.write_bytes(b"old-complete")
    partial.write_bytes(b"new-complete")
    assert final.read_bytes() == b"old-complete"
    publish_completed_checkpoint(partial, final)
    assert final.read_bytes() == b"new-complete"
    assert not partial.exists()


def test_conditioning_rejects_ambiguous_legacy_execute_flag() -> None:
    canonical = ROOT / "reports" / "standard_causal_transformer_survival.json"
    before = canonical.read_bytes()
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "standard_causal_transformer_conditioning.py"), "--execute"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --execute" in result.stderr
    assert canonical.read_bytes() == before


def test_conditioning_train_mode_fails_before_optimizer_without_complete_receipt(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_conditioning.json").read_text())
    config["conditioned_checkpoint_dir"] = str(tmp_path / "checkpoint")
    config["report"] = str(tmp_path / "report.json")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "standard_causal_transformer_conditioning.py"),
            "--config",
            str(config_path),
            "--mode",
            "train",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["trigger_state"] == "RED"
    assert {
        "base_training_receipt_incomplete",
        "base_training_receipt_is_evaluation_replay",
    } <= {row["code"] for row in report["typed_faults"]}
    assert not (tmp_path / "checkpoint").exists()


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
    assert semantic.endswith("signature def solve(data):")
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


def test_private_callable_signature_is_invariant_to_hidden_tests_body_and_decoder_metadata() -> None:
    base = {
        "entry_point": "pair_sum",
        "solution_body": "return data + other",
        "tests": "assert pair_sum(2, 3) == 5\n",
        "decoder_contract": {"visible_arg_count_hint": 2, "return_shape": "number"},
    }
    changed_hidden = {
        **base,
        "solution_body": "return None",
        "tests": "assert pair_sum(1) == 99\n",
        "decoder_contract": {"visible_arg_count_hint": 1, "return_shape": "dict"},
    }
    expected = (
        "def pair_sum(data=None, other=None, *extra):",
        {"source": "generic_prompt_only_interface", "arity": "variable"},
    )
    assert training_callable_signature(base) == expected
    assert training_callable_signature(changed_hidden) == expected


def test_explicit_callable_signature_is_preserved_without_hidden_derivation() -> None:
    signature, receipt = training_callable_signature(
        {
            "entry_point": "pair_sum",
            "callable_signature": "def pair_sum(left, right):",
            "tests": "assert pair_sum(1) == 99\n",
            "solution_body": "return None",
        }
    )
    assert signature == "def pair_sum(left, right):"
    assert receipt == {"source": "explicit_callable_signature", "arity": 2}


def test_model_signature_canonicalizes_name_but_preserves_declared_arguments() -> None:
    config = {"tokenization": {"canonical_model_signature_name": "solve"}}
    assert canonical_model_signature("def task_123(left, right, *extra):", config) == (
        "def solve(left, right, *extra):"
    )


def test_shared_source_encoding_uses_target_vocabulary_id_space() -> None:
    config = {"tokenization": {"shared_source_target_vocabulary": True}}
    source_vocab = {"Return": 0}
    target_vocab = json.loads(
        (ROOT / "checkpoints" / "strict_generator_mlx_dense_body_open_vocab" /
         "strict_generator_mlx_strict_generator_dense_right_sized_v1_vocab.json").read_text()
    )["target_vocab"]
    ids, receipt = encode_model_source("Return data", source_vocab, target_vocab, config)
    assert ids
    assert receipt["unknown_token_count"] == 0
    assert receipt["model_stream"] == "shared_source_target"
    assert source_token_offset(config, source_vocab) == 3
    assert target_token_offset(config, source_vocab) == 3


def test_split_source_encoding_uses_disjoint_source_embedding_segment() -> None:
    config = {"tokenization": {"shared_source_target_vocabulary": False}}
    source_vocab = {"Return": 0, "data": 1}
    target_vocab = {"<bos>": 0, "return": 1, "data": 2}
    ids, receipt = encode_model_source("Return data", source_vocab, target_vocab, config)
    assert ids == [0, 1]
    assert receipt["model_stream"] == "split_source_target"
    assert source_token_offset(config, source_vocab) == 3
    assert target_token_offset(config, source_vocab) == 5
    embedded_source_ids = [source_token_offset(config, source_vocab) + value for value in ids]
    assert max(embedded_source_ids) < target_token_offset(config, source_vocab)


def test_broad_private_rows_declare_prompt_visible_signature_from_template_contract() -> None:
    by_category = {row.category: row for row in template_bank()}
    one_arg = row_from_template(
        by_category["bpg_gcd_positive"], split="eval", task_index=1, variant=1
    )
    two_arg = row_from_template(
        by_category["bpg_lcs_length"], split="eval", task_index=2, variant=2
    )
    four_arg = row_from_template(
        by_category["bpg_shortest_hops"], split="eval", task_index=3, variant=3
    )
    assert one_arg["callable_signature"].endswith("(data):")
    assert two_arg["callable_signature"].endswith("(data, other):")
    assert four_arg["callable_signature"].endswith("(data, other, *extra):")
    assert all(
        row["provenance"]["callable_signature_source"]
        == "template_declared_argument_contract"
        for row in (one_arg, two_arg, four_arg)
    )


def test_decoder_interface_hint_ignores_private_tests_and_solution() -> None:
    row = {
        "category": "",
        "prompt": "Transform the input sequence.",
        "tests": "assert solve(1, 2, 3, 4) == 10",
        "solution_body": "return data + other + extra[0] + extra[1]",
        "public_benchmark": False,
    }
    assert visible_arg_count_hint_for_task(row) is None
    row["prompt"] = "Compare two strings and return the shorter one."
    assert visible_arg_count_hint_for_task(row) == 2


def test_blind_audit_rejects_hidden_test_derived_signature_helper(tmp_path: Path) -> None:
    source = tmp_path / "leaky_signature.py"
    source.write_text(
        "def training_callable_signature(row):\n"
        "    tests = row.get('tests')\n"
        "    return 'def solve(data):' if tests else 'def solve():'\n"
    )
    audit = audit_source(source)
    assert audit["violation_count"] == 1
    assert audit["violations"][0]["kind"] == "forbidden_field_in_inference_path"
    assert audit["violations"][0]["field"] == "tests"


def test_family_disjoint_eval_freezes_normalized_callable_signature() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    rows, _families = select_family_disjoint_eval(config)
    lcs = next(row for row in rows if row["concept_residual_label"] == "bpg_lcs_length")
    assert lcs["callable_signature"].endswith("(data, other):")
    assert lcs["callable_signature_receipt"] == {
        "source": "explicit_callable_signature",
        "arity": 2,
    }
    assert visible_eval_source(lcs).endswith("signature def solve(data, other):")


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


def test_private_sampling_mass_can_match_a_frozen_probability() -> None:
    examples = [
        {"source": "licensed_function", "source_text": "licensed-a", "body": "return data"},
        {"source": "licensed_function", "source_text": "licensed-b", "body": "return other"},
        {"source": "governed_private", "source_text": "private-a", "body": "return data + 1"},
        {"source": "governed_private", "source_text": "private-b", "body": "return data - 1"},
    ]
    weighted, audit = assign_body_balanced_sampling_weights(
        examples,
        private_body_weight=16.0,
        private_sampling_probability_target=0.25,
    )
    assert audit["configured_private_body_sampling_weight"] == 16.0
    assert audit["private_sampling_probability"] == pytest.approx(0.25)
    assert sum(row["sampling_weight"] for row in weighted[2:]) == pytest.approx(2 / 3)


def test_standalone_sft_contract_accepts_imports_and_lexical_closures() -> None:
    source = "Compute a rounded square root.\nsignature def solve(data):"
    body = (
        "import math\n"
        "def rounded(value):\n"
        "    return round(math.sqrt(value), 3)\n"
        "return rounded(data)"
    )
    decision = standalone_sft_contract_decision(source, body)
    assert decision["accepted"] is True
    assert decision["unresolved_names"] == []
    assert decision["target_derived_source_field_count"] == 0


def test_standalone_sft_contract_rejects_hidden_module_context() -> None:
    source = "Normalize numeric input.\nsignature def solve(data):"
    decision = standalone_sft_contract_decision(source, "return np.asarray(data)")
    assert decision["accepted"] is False
    assert decision["reject_reasons"] == ["unresolved_module_context"]
    assert decision["unresolved_names"] == ["np"]
    assert decision["model_source_unchanged"] is True


def test_standalone_sft_contract_requires_declared_signature() -> None:
    decision = standalone_sft_contract_decision("Compute a sum.", "return sum(data)")
    assert decision["accepted"] is False
    assert "declared_signature_missing" in decision["reject_reasons"]


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

    contract_change = json.loads(json.dumps(config))
    contract_change["sft_contract_admission"] = {
        "require_self_contained_body": True,
        "private_sampling_probability_target": 0.25,
    }
    assert stage_signature(contract_change) != baseline


def test_stage_signature_binds_each_admitted_training_source(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    source = tmp_path / "private.jsonl"
    source.write_text('{"prompt":"p","solution_body":"return data"}\n')
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"train_admitted_sources": [{"path": str(source)}]}))
    config["sources"]["training_admission"] = str(admission)
    baseline = stage_signature(config)
    source.write_text('{"prompt":"p","solution_body":"return data + 1"}\n')
    assert stage_signature(config) != baseline


def test_stage_materialization_lock_blocks_duplicate_writer_and_recovers_stale_owner(
    tmp_path: Path,
) -> None:
    with stage_materialization_lock(tmp_path, timeout_seconds=0.1):
        with pytest.raises(TimeoutError):
            with stage_materialization_lock(tmp_path, timeout_seconds=0.05):
                pass
    lock_path = tmp_path / ".materialize.lock"
    lock_path.write_text(json.dumps({"owner_token": "dead", "pid": 99999999, "created_epoch": 0}))
    with stage_materialization_lock(tmp_path, timeout_seconds=0.1):
        assert lock_path.exists()
    assert not lock_path.exists()


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

    canary["serial"]["passed_task_count"] = 0
    canary["batched"]["passed_task_count"] = 0
    canary["behavior_non_regression"] = True
    canary["adoption_state"] = "BATCHED_RUNTIME_ONLY"
    runtime_only = audit_generation_mode_canary(canary)
    assert runtime_only["hard_gaps"] == []
    assert runtime_only["adoption_state"] == "BATCHED_RUNTIME_ONLY"


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
