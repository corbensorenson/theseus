from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kerc_adequacy_canary import (  # noqa: E402
    classify_gates,
    compact_metrics,
    select_balanced_subset,
)
from kerc_checkpoint_schema import (  # noqa: E402
    migrate_legacy_checkpoint,
    rollback_checkpoint_contract,
    validate_checkpoint_contract,
)


def fixture_stage() -> SimpleNamespace:
    inputs = np.zeros((8, 8), dtype=np.int32)
    labels = np.zeros((8, 8), dtype=np.int32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    verifier_labels = np.ones((8, 4), dtype=np.float32)
    for objective in range(4):
        positive = objective * 2
        negative = positive + 1
        source = [1, 10 + objective, 50 + objective, 2]
        inputs[positive, :6] = [*source, 3, 20 + objective]
        inputs[negative, :6] = [*source, 3, 21 + objective]
        labels[positive, :6] = [10 + objective, 50 + objective, 2, 3, 20 + objective, 4]
        labels[negative, :6] = [10 + objective, 50 + objective, 2, 3, 21 + objective, 4]
        mask[positive, 4:6] = 1
        verifier_labels[negative, objective] = 0.0
    return SimpleNamespace(
        inputs=inputs,
        labels=labels,
        mask=mask,
        loss_mask=mask.astype(np.float32),
        sample_weights=np.ones((8,), dtype=np.float64),
        kerc_residual_labels=np.tile(np.arange(4, dtype=np.int32), (8, 1)),
        kerc_verifier_labels=verifier_labels,
    )


def test_compact_metrics_preserves_scalars_and_hides_diagnostic_vectors() -> None:
    compact = compact_metrics(
        {
            "mean_total_loss": 1.5,
            "mechanism_activity": {"stage_weights_maximum_absolute": 1.0},
            "expected_token_logits": [[1.0, 2.0]],
            "verifier_logits": [[3.0]],
        }
    )
    assert compact["mean_total_loss"] == 1.5
    assert compact["mechanism_activity"]["stage_weights_maximum_absolute"] == 1.0
    assert "expected_token_logits" not in compact
    assert compact["diagnostic_vectors_embedded"] is False


def test_balanced_subset_requires_exact_source_bound_corruption() -> None:
    subset, receipt = select_balanced_subset(
        fixture_stage(),
        task_token_ids=(10, 11, 12, 13),
        separator_id=2,
        positives_per_objective=1,
    )
    assert len(subset.inputs) == 8
    assert receipt["positive_row_count"] == 4
    assert receipt["verifier_only_row_count"] == 4
    assert all(len(indices) == 2 for indices in receipt["rows_by_objective"].values())


def test_balanced_subset_rejects_missing_verifier_pair() -> None:
    stage = fixture_stage()
    stage.inputs[1, 2] = 99
    try:
        select_balanced_subset(
            stage,
            task_token_ids=(10, 11, 12, 13),
            separator_id=2,
            positives_per_objective=1,
        )
    except ValueError as exc:
        assert "exact-source verifier pair" in str(exc)
    else:
        raise AssertionError("missing verifier pair was admitted")


def gate_fixture() -> dict:
    return {
        "acceptance": {
            "maximum_final_to_initial_loss_ratio": 0.8,
            "minimum_token_accuracy_gain": 0.01,
            "minimum_residual_informative_channel_count": 3,
            "minimum_residual_macro_balanced_accuracy": 0.75,
            "minimum_verifier_macro_balanced_accuracy": 0.75,
            "minimum_verifier_negative_recall": 0.5,
            "maximum_checkpoint_reload_logit_delta": 1e-6,
            "maximum_checkpoint_migration_logit_delta": 1e-6,
            "maximum_checkpoint_rollback_logit_delta": 1e-6,
            "maximum_resume_equivalence_logit_delta": 1e-5,
            "maximum_disabled_mechanism_activity": 0.0,
            "minimum_active_intervention_logit_delta": 1e-7,
            "maximum_inactive_residual_intervention_logit_delta": 1e-7,
            "minimum_target_tokens_per_second": 1.0,
        },
        "required_trained_ablations": [
            "without_stage_routing",
            "without_hierarchical_residual",
            "without_independent_verifier",
        ],
    }


def trained_ablation_fixture() -> dict:
    return {
        "without_stage_routing": {
            "after": {"mechanism_activity": {"stage_weights_maximum_absolute": 0.0}}
        },
        "without_hierarchical_residual": {
            "after": {"mechanism_activity": {"residual_logits_maximum_absolute": 0.0}}
        },
        "without_independent_verifier": {
            "after": {"mechanism_activity": {"verifier_logits_maximum_absolute": 0.0}}
        },
    }


def test_failed_learning_is_inconclusive_not_architecture_falsification() -> None:
    state, gates = classify_gates(
        gate_fixture(),
        before={"mean_total_loss": 10.0, "token_accuracy": 0.0},
        after={"mean_total_loss": 9.0, "token_accuracy": 0.0, "residual_informative_channel_count": 1, "residual_informative_macro_balanced_accuracy": 0.5, "verifier_macro_balanced_accuracy": 0.5, "verifier_minimum_negative_recall": 0.0},
        first_phase={"target_tokens_per_second": 2.0},
        second_phase={"target_tokens_per_second": 2.0},
        reload_delta=0.0,
        resume_delta=0.0,
        interventions={
            "trusted_stage_token_removed": {"delta_by_objective": {name: 1e-3 for name in ("a", "b")}},
            "hierarchical_residual_values_zeroed": {
                "delta_by_objective": {"surface_to_kernel_program_v1": 1e-3, "answer_packet_to_surface_v1": 1e-3, "surface_direct_control_v1": 0.0, "kernel_program_to_answer_packet_v1": 0.0},
                "expected_active_objectives": ["surface_to_kernel_program_v1", "answer_packet_to_surface_v1"],
                "expected_inactive_objectives": ["surface_direct_control_v1", "kernel_program_to_answer_packet_v1"],
            },
            "independent_verifier_classifier_zeroed": {"maximum_verifier_logit_delta": 1e-3},
        },
        migration_rejection=True,
        schema_migration_valid=True,
        schema_unknown_rejection=True,
        migration_logit_delta=0.0,
        rollback_logit_delta=0.0,
        trained_ablations=trained_ablation_fixture(),
        partial_file_count=0,
    )
    assert state == "INCONCLUSIVE_EXPERIMENT"
    assert not any(row["severity"] == "hard" and not row["passed"] for row in gates)


def test_lifecycle_failure_is_red() -> None:
    state, _gates = classify_gates(
        gate_fixture(),
        before={"mean_total_loss": 10.0, "token_accuracy": 0.0},
        after={"mean_total_loss": 1.0, "token_accuracy": 0.5, "residual_informative_channel_count": 3, "residual_informative_macro_balanced_accuracy": 1.0, "verifier_macro_balanced_accuracy": 1.0, "verifier_minimum_negative_recall": 1.0},
        first_phase={"target_tokens_per_second": 2.0},
        second_phase={"target_tokens_per_second": 2.0},
        reload_delta=0.0,
        resume_delta=0.0,
        interventions={
            "trusted_stage_token_removed": {"delta_by_objective": {"a": 1e-3}},
            "hierarchical_residual_values_zeroed": {
                "delta_by_objective": {"surface_to_kernel_program_v1": 1e-3, "answer_packet_to_surface_v1": 1e-3, "surface_direct_control_v1": 0.0, "kernel_program_to_answer_packet_v1": 0.0},
                "expected_active_objectives": ["surface_to_kernel_program_v1", "answer_packet_to_surface_v1"],
                "expected_inactive_objectives": ["surface_direct_control_v1", "kernel_program_to_answer_packet_v1"],
            },
            "independent_verifier_classifier_zeroed": {"maximum_verifier_logit_delta": 1e-3},
        },
        migration_rejection=False,
        schema_migration_valid=True,
        schema_unknown_rejection=True,
        migration_logit_delta=0.0,
        rollback_logit_delta=0.0,
        trained_ablations=trained_ablation_fixture(),
        partial_file_count=0,
    )
    assert state == "RED"


def checkpoint_binding() -> dict:
    return {
        "target_id": "english_kerc",
        "role": "kerc_english_candidate",
        "model_config_sha256": "model-sha",
        "plan_sha256": "plan-sha",
        "stage_signature": "stage-sha",
        "vocab_size": 32,
        "kernel_code_vocabulary_sha256": "codebook-sha",
    }


def test_checkpoint_schema_migrates_real_serialization_and_rolls_back(tmp_path: Path) -> None:
    import mlx.core as mx

    legacy_checkpoint = tmp_path / "legacy.npz"
    legacy_optimizer = tmp_path / "legacy_optimizer.safetensors"
    mx.savez(
        str(legacy_checkpoint),
        **{
            "embedding.weight": mx.arange(12, dtype=mx.float32).reshape(3, 4),
            "head.bias": mx.array([1.0, -1.0], dtype=mx.float32),
        },
    )
    mx.save_safetensors(
        str(legacy_optimizer),
        {"state.0": mx.array([0.25, 0.5], dtype=mx.float32)},
    )
    checkpoint = tmp_path / "current.safetensors"
    optimizer = tmp_path / "current_optimizer.safetensors"
    manifest_path = tmp_path / "manifest.json"
    manifest = migrate_legacy_checkpoint(
        legacy_checkpoint=legacy_checkpoint,
        legacy_optimizer=legacy_optimizer,
        checkpoint=checkpoint,
        optimizer=optimizer,
        manifest_path=manifest_path,
        binding=checkpoint_binding(),
    )
    validate_checkpoint_contract(
        manifest,
        checkpoint=checkpoint,
        optimizer=optimizer,
        binding=checkpoint_binding(),
    )
    rollback = rollback_checkpoint_contract(
        manifest,
        checkpoint=checkpoint,
        optimizer=optimizer,
        rollback_checkpoint=tmp_path / "rollback.npz",
        rollback_optimizer=tmp_path / "rollback_optimizer.safetensors",
        binding=checkpoint_binding(),
    )
    assert rollback["checkpoint_inventory_sha256"] == (
        manifest["source"]["checkpoint_inventory"]["inventory_sha256"]
    )
    assert rollback["optimizer_inventory_sha256"] == (
        manifest["source"]["optimizer_inventory"]["inventory_sha256"]
    )


def test_checkpoint_schema_rejects_binding_retarget(tmp_path: Path) -> None:
    import mlx.core as mx

    legacy_checkpoint = tmp_path / "legacy.npz"
    legacy_optimizer = tmp_path / "legacy_optimizer.safetensors"
    mx.savez(str(legacy_checkpoint), **{"weight": mx.ones((2, 2))})
    mx.save_safetensors(str(legacy_optimizer), {"state": mx.zeros((2,))})
    checkpoint = tmp_path / "current.safetensors"
    optimizer = tmp_path / "current_optimizer.safetensors"
    manifest = migrate_legacy_checkpoint(
        legacy_checkpoint=legacy_checkpoint,
        legacy_optimizer=legacy_optimizer,
        checkpoint=checkpoint,
        optimizer=optimizer,
        manifest_path=tmp_path / "manifest.json",
        binding=checkpoint_binding(),
    )
    retargeted = {**checkpoint_binding(), "plan_sha256": "different-plan"}
    try:
        validate_checkpoint_contract(
            manifest,
            checkpoint=checkpoint,
            optimizer=optimizer,
            binding=retargeted,
        )
    except ValueError as exc:
        assert "binding_mismatch" in str(exc)
    else:
        raise AssertionError("retargeted checkpoint binding was admitted")
