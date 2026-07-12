from __future__ import annotations

import json
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
    range_view,
    train_target,
    validate_config,
    validate_resume,
)
from standard_causal_transformer_model import (  # noqa: E402
    CausalTransformerConfig,
    build_model,
    parameter_count,
)


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
        },
        "comparison_contract": {"preregistered_before_training": True},
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
