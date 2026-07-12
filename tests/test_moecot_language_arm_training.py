from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from moecot_language_arm_training import (  # noqa: E402
    ARM_IDS,
    audit_arm_views,
    audit_tokenizer_stage,
    generate_model_text,
    matched_decoder_only_config,
    materialize_target_supervision,
    range_view,
    serialization_valid_local_ids,
    train_target,
    validate_config,
    validate_resume,
)
from standard_causal_transformer_model import (  # noqa: E402
    CausalTransformerConfig,
    build_model,
    parameter_count,
)
from standard_causal_transformer_survival import causal_loss  # noqa: E402
from neural_seed_open_vocab import reserve_byte_fallback_tokens  # noqa: E402


def arm_views() -> dict:
    return {
        "policy": "project_theseus_moecot_canonical_arm_views_v1",
        "arms": {
            arm_id: {
                "row_ranges": [{"start": index * 2, "stop": index * 2 + 2}],
                "target_positions": 8,
                "independent_weights_required": True,
            }
            for index, arm_id in enumerate(ARM_IDS)
        },
        "mixed_dense_control": {"row_ranges": [{"start": 0, "stop": 10}]},
        "hidden_generalist_fallback": "forbidden",
    }


def tiny_config(tmp_path: Path) -> dict:
    return {
        "policy": "project_theseus_moecot_language_arm_training_v1",
        "seed": 7,
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "arm_model": {
            "d_model": 16,
            "num_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 1,
            "ff_dim": 32,
            "rope_base": 10000.0,
            "rms_norm_eps": 0.00001,
            "attention_policy": "causal",
            "source_target_separator_token_id": 2,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "min_learning_rate": 0.0001,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "checkpoint_every_steps": 1,
            "maximum_optimizer_repetitions": 4,
            "maximum_supervision_optimizer_repetitions": 32,
            "supervision_optimizer_repetitions": 4,
            "termination_loss_weight": 4.0,
            "byte_boundary_loss_weight": 2.0,
        },
        "comparison_contract": {"preregistered_before_training": True},
        "evaluation": {
            "policy": "project_theseus_moecot_direct_model_only_evaluation_v1",
            "beam_width": 2,
            "branching_factor": 2,
            "target_visible_to_generator": False,
            "templates_renderers_routers_tools_allowed": False,
        },
        "boundaries": {
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "templates_renderers_routers_tools_credit": 0,
            "hidden_generalist_fallback": "forbidden",
            "runtime_serving_allowed": False,
        },
    }


def test_arm_views_are_an_exact_non_overlapping_partition() -> None:
    accepted = audit_arm_views(arm_views(), 10)
    assert accepted["state"] == "GREEN"
    assert accepted["non_overlapping_complete_partition"] is True

    tampered = arm_views()
    tampered["arms"]["python"]["row_ranges"][0]["start"] = 1
    rejected = audit_arm_views(tampered, 10)
    assert rejected["state"] == "RED"
    assert any(gap.startswith("arm_range_gap_or_overlap:python") for gap in rejected["hard_gaps"])


def test_training_denies_legacy_stage_without_each_language_tokenizer_receipt() -> None:
    profiles = {
        "policy": "project_theseus_moecot_language_tokenizer_v1",
        "english_conversation_instruction": "exact_text_v1",
        "english_broad": "exact_text_v1",
        "python": "exact_text_v1",
        "javascript_typescript": "exact_text_v1",
        "html_css": "exact_text_v1",
        "rust": "exact_text_v1",
    }
    category_profiles = {
        f"{category}:{profile}": 1
        for category, profile in profiles.items()
        if category != "policy"
    }
    accepted = audit_tokenizer_stage(
        {"tokenization": {"canonical_language_profiles": profiles}},
        {
            "tokenizer_audit": {
                "category_profiles_by_selected_document": category_profiles,
                "roundtrip_failure_count": 0,
                "rejected_unknown_token_document_count": 1,
                "admitted_unknown_token_position_count": 0,
            }
        },
    )
    assert accepted["state"] == "GREEN"

    rejected = audit_tokenizer_stage(
        {"tokenization": {"canonical_language_profiles": profiles}}, {}
    )
    assert rejected["state"] == "RED"
    assert len(rejected["hard_gaps"]) == 6


def test_config_rejects_capability_credit_and_hidden_fallback(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    validate_config(config)
    config["boundaries"]["fallback_return_count"] = 1
    with pytest.raises(ValueError, match="no-cheat counters"):
        validate_config(config)
    config["boundaries"]["fallback_return_count"] = 0
    config["boundaries"]["hidden_generalist_fallback"] = "allowed"
    with pytest.raises(ValueError, match="hidden generalist"):
        validate_config(config)


def test_range_view_coalesces_adjacent_ranges_without_copy() -> None:
    array = np.arange(24).reshape(6, 4)
    view = range_view(array, [{"start": 1, "stop": 3}, {"start": 3, "stop": 5}])
    assert np.shares_memory(array, view)
    assert view.tolist() == array[1:5].tolist()


def test_exact_supervision_masks_only_target_and_never_truncates(tmp_path: Path) -> None:
    source_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "Fix": 4}
    target_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "done": 4}
    reserve_byte_fallback_tokens(source_vocab, max_vocab=270, stream="source")
    reserve_byte_fallback_tokens(target_vocab, max_vocab=270, stream="target")
    row = {
        "split": "private_train",
        "arm_id": "english",
        "public_benchmark": False,
        "prompt": "Fix this",
        "target": "done",
    }
    artifact = tmp_path / "train.jsonl"
    artifact.write_text(json.dumps(row) + "\n")
    import hashlib

    target = {
        "target_id": "english",
        "supervision_artifacts": {
            "private_train": {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "row_count": 1,
            }
        },
    }
    base = {
        "tokenization": {"max_sequence_tokens": 32, "shared_source_target_vocabulary": False}
    }
    training_config = {
        "training": {"termination_loss_weight": 4.0, "byte_boundary_loss_weight": 2.0}
    }
    stage = materialize_target_supervision(
        training_config,
        base,
        target,
        metadata={"source_vocab": source_vocab, "target_vocab": target_vocab},
    )
    separator = int(np.flatnonzero(stage.inputs[0] == 2)[0])
    supervised = np.flatnonzero(stage.mask[0])
    assert supervised[0] > separator
    assert stage.receipt["source_truncation_count"] == 0
    assert stage.receipt["target_truncation_count"] == 0
    assert stage.receipt["public_training_rows_written"] == 0
    assert stage.receipt["weighted_loss_positions"] > stage.receipt["target_positions"]

    base["tokenization"]["max_sequence_tokens"] = 4
    with pytest.raises(ValueError, match="requires truncation"):
        materialize_target_supervision(
            training_config,
            base,
            target,
            metadata={"source_vocab": source_vocab, "target_vocab": target_vocab},
        )


def test_generation_api_cannot_receive_hidden_target() -> None:
    parameters = inspect.signature(generate_model_text).parameters
    assert "prompt" in parameters
    assert "target" not in parameters
    assert "expected" not in parameters


def test_decoder_only_control_is_mechanically_parameter_matched() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    encoder_config = {
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "ff_dim": 64,
        "attention_policy": "encoder_decoder",
        "source_target_separator_token_id": 2,
        "source_encoder_layers": 1,
    }

    def count(model_config: dict) -> int:
        model = build_model(
            CausalTransformerConfig(vocab_size=64, **model_config), mx=mx, nn=nn
        )
        return int(parameter_count(model, mlx_utils))

    reference = count(encoder_config)
    control, observed = matched_decoder_only_config(
        reference, encoder_config, count=count
    )
    assert control["attention_policy"] == "prefix_lm"
    assert "source_encoder_layers" not in control
    assert abs(observed - reference) / reference <= 0.01


def test_target_only_loss_trains_source_encoder_and_cross_attention() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    model = build_model(
        CausalTransformerConfig(
            vocab_size=64,
            d_model=16,
            num_layers=1,
            num_heads=4,
            num_kv_heads=1,
            ff_dim=32,
            attention_policy="encoder_decoder",
            source_encoder_layers=1,
        ),
        mx=mx,
        nn=nn,
    )
    inputs = mx.array([[1, 10, 11, 2, 20, 21]], dtype=mx.int32)
    labels = mx.array([[10, 11, 2, 20, 21, 3]], dtype=mx.int32)
    target_only_mask = mx.array([[0, 0, 0, 0, 1, 1]], dtype=mx.float32)
    loss_and_grad = nn.value_and_grad(model, causal_loss)
    loss, gradients = loss_and_grad(
        model, inputs, labels, target_only_mask, mx, nn
    )
    mx.eval(loss, gradients)
    gradient_mass = {
        name: float(mx.sum(mx.abs(value)).item())
        for name, value in mlx_utils.tree_flatten(gradients)
    }
    assert sum(
        value for name, value in gradient_mass.items() if name.startswith("source_layers.")
    ) > 0.0
    assert sum(
        value for name, value in gradient_mass.items() if ".source_attention." in name
    ) > 0.0


def test_byte_span_grammar_never_forces_completion_or_allows_invalid_tokens() -> None:
    inverse = {
        0: "<pad>",
        1: "<unk>",
        2: "<bos>",
        3: "<eos>",
        4: "text",
        5: "<target_token_bytes>",
        6: "</target_token_bytes>",
        7: "<byte:61>",
    }
    outside = serialization_valid_local_ids([], inverse)
    assert 3 in outside and 4 in outside and 5 in outside
    assert 6 not in outside and 7 not in outside
    inside = serialization_valid_local_ids(["<target_token_bytes>"], inverse)
    assert set(inside) == {6, 7}


def test_resume_rejects_tampered_checkpoint_optimizer_and_plan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.npz"
    optimizer = tmp_path / "optimizer.safetensors"
    checkpoint.write_bytes(b"weights")
    optimizer.write_bytes(b"optimizer")
    import hashlib

    receipt = {
        "policy": "project_theseus_moecot_language_arm_training_receipt_v1",
        "target_id": "python",
        "plan_sha256": "plan-a",
        "stage_signature": "stage-a",
        "row_ranges": [{"start": 0, "stop": 2}],
        "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
        "optimizer_state_sha256": hashlib.sha256(b"optimizer").hexdigest(),
    }
    plan = {"plan_sha256": "plan-a", "stage": {"stage_signature": "stage-a"}}
    target = {"target_id": "python", "row_ranges": [{"start": 0, "stop": 2}]}
    validate_resume(receipt, plan, target, checkpoint, optimizer)

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint_identity_mismatch"):
        validate_resume(receipt, plan, target, checkpoint, optimizer)
    checkpoint.write_bytes(b"weights")
    plan["plan_sha256"] = "plan-b"
    with pytest.raises(ValueError, match="plan_identity_mismatch"):
        validate_resume(receipt, plan, target, checkpoint, optimizer)


def test_tiny_mlx_arm_writes_distinct_resumable_model_and_optimizer_state(tmp_path: Path) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    config = tiny_config(tmp_path)
    validate_config(config)
    model = build_model(
        CausalTransformerConfig(vocab_size=64, **config["arm_model"]),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
    )
    count = int(parameter_count(model, mlx_utils))
    inputs = np.asarray(
        [[1, 4, 5, 6, 7, 8, 9, 10], [1, 11, 12, 13, 14, 15, 16, 17]],
        dtype=np.int32,
    )
    labels = np.asarray(
        [[4, 5, 6, 7, 8, 9, 10, 2], [11, 12, 13, 14, 15, 16, 17, 2]],
        dtype=np.int32,
    )
    mask = np.ones_like(inputs, dtype=np.uint8)
    stage = SimpleNamespace(pretrain_inputs=inputs, pretrain_labels=labels, pretrain_mask=mask)
    checkpoint = tmp_path / "checkpoints" / "python" / "weights.npz"
    optimizer_path = checkpoint.parent / "optimizer.safetensors"
    receipt_path = checkpoint.parent / "training_receipt.json"
    target = {
        "target_id": "python",
        "role": "language_arm",
        "row_ranges": [{"start": 0, "stop": 2}],
        "unique_target_positions": 32,
        "model": config["arm_model"],
        "parameter_count": count,
        "checkpoint": str(checkpoint),
        "optimizer_state": str(optimizer_path),
        "receipt": str(receipt_path),
    }
    plan = {
        "plan_sha256": "a" * 64,
        "stage": {"stage_signature": "stage-a", "metadata_sha256": "b" * 64},
        "models": {"vocab_size": 64},
    }
    first = train_target(
        config,
        plan,
        target,
        stage=stage,
        max_steps=1,
        resume=False,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
    )
    assert first["optimizer_steps"] == 1
    assert first["complete"] is False
    assert checkpoint.is_file() and optimizer_path.is_file() and receipt_path.is_file()
    assert not checkpoint.with_name("weights.partial.npz").exists()
    assert first["checkpoint_sha256"] != first["optimizer_state_sha256"]
    assert first["public_training_rows_written"] == 0

    second = train_target(
        config,
        plan,
        target,
        stage=stage,
        max_steps=1,
        resume=True,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
    )
    assert second["resume"] is True
    assert second["optimizer_steps"] == 2
    assert second["optimizer_positions"] > first["optimizer_positions"]
    assert second["resume_base_checkpoint_sha256"] == first["checkpoint_sha256"]
    assert json.loads(receipt_path.read_text())["optimizer_state_sha256"] == second["optimizer_state_sha256"]
