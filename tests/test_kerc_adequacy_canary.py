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
    expected_logit_delta_by_objective,
    nested_logit_equivalence,
    select_balanced_subset,
)
from kerc_checkpoint_schema import (  # noqa: E402
    migrate_legacy_checkpoint,
    rollback_checkpoint_contract,
    validate_checkpoint_contract,
)


def fixture_stage() -> SimpleNamespace:
    inputs = np.zeros((12, 8), dtype=np.int32)
    labels = np.zeros((12, 8), dtype=np.int32)
    mask = np.zeros((12, 8), dtype=np.uint8)
    verifier_labels = np.ones((12, 5), dtype=np.float32)
    objectives = [0, 1, 2, 3, 0]
    signatures = [
        [0, 0, 0, 0],
        [1, 1, 2, 3],
        [0, 1, 2, 3],
        [1, 1, 2, 3],
        [1, 0, 0, 3],
    ]
    coverage: list[tuple[str, ...]] = []
    for pair_index, objective in enumerate(objectives):
        positive = pair_index * 2
        negative = positive + 1
        source = [1, 10 + objective, 50 + pair_index, 2]
        inputs[positive, :6] = [*source, 3, 20 + pair_index]
        inputs[negative, :6] = [*source, 3, 30 + pair_index]
        labels[positive, :6] = [
            10 + objective,
            50 + pair_index,
            2,
            3,
            20 + pair_index,
            4,
        ]
        labels[negative, :6] = [
            10 + objective,
            50 + pair_index,
            2,
            3,
            30 + pair_index,
            4,
        ]
        mask[positive, 4:6] = 1
        verifier_labels[negative, pair_index] = 0.0
        base = [
            f"objective:{('surface_direct_control_v1', 'surface_to_kernel_program_v1', 'kernel_program_to_answer_packet_v1', 'answer_packet_to_surface_v1')[objective]}",
            "interaction:present" if signatures[pair_index][0] else "interaction:absent",
        ]
        base.extend(
            f"residual:{channel}:active"
            for channel, value in zip(
                ("interaction", "segment", "token", "exact"),
                signatures[pair_index],
            )
            if value
        )
        base.append(f"decision:{('ANSWER', 'CLARIFY', 'ABSTAIN', 'ANSWER', 'ANSWER')[pair_index]}")
        coverage.append((*base, "verifier:positive"))
        coverage.append(
            (
                *base,
                "verifier:negative:"
                + (
                    "semantic_consistency",
                    "protected_object_consistency",
                    "numeric_value_consistency",
                    "surface_fidelity",
                    "answer_decision_consistency",
                )[pair_index],
            )
        )
    for offset, strategy in enumerate(("context_withheld", "context_shuffled")):
        index = 10 + offset
        inputs[index, :6] = [1, 10, 80 + offset, 2, 3, 90 + offset]
        labels[index, :6] = [10, 80 + offset, 2, 3, 90 + offset, 4]
        verifier_labels[index] = np.asarray([0, 1, 1, 1, 0], dtype=np.float32)
        coverage.append(
            (
                "objective:surface_direct_control_v1",
                "interaction:absent",
                f"verifier:counterfactual:{strategy}",
            )
        )
    return SimpleNamespace(
        inputs=inputs,
        labels=labels,
        mask=mask,
        loss_mask=mask.astype(np.float32),
        sample_weights=np.ones((12,), dtype=np.float64),
        kerc_residual_labels=np.asarray(
            [value for signature in signatures for value in (signature, signature)]
            + [[1, 0, 0, 0], [1, 0, 0, 0]],
            dtype=np.int32,
        ),
        kerc_verifier_labels=verifier_labels,
        kerc_coverage_labels=tuple(coverage),
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


def test_resume_logit_equivalence_uses_absolute_and_relative_float_tolerance() -> None:
    equivalent = nested_logit_equivalence(
        [[2.0, -3.0]],
        [[2.000015, -3.00002]],
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    assert equivalent["equivalent"] is True
    assert equivalent["violation_count"] == 0

    divergent = nested_logit_equivalence(
        [[0.0, 1.0]],
        [[0.001, 1.0]],
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    assert divergent["equivalent"] is False
    assert divergent["violation_count"] == 1


def test_intervention_delta_supports_multiple_pairs_per_objective() -> None:
    baseline = {"expected_token_logits": [[1.0], [2.0], [3.0]]}
    changed = {"expected_token_logits": [[1.5], [9.0], [3.25]]}
    selection = {
        "row_indices": [10, 11, 12],
        "rows_by_objective": {
            "surface_direct_control_v1": [[10, 11], [12, 13]],
            "surface_to_kernel_program_v1": [[12, 14]],
        },
    }

    deltas = expected_logit_delta_by_objective(baseline, changed, selection)

    assert deltas["surface_direct_control_v1"] == 0.5
    assert deltas["surface_to_kernel_program_v1"] == 0.25


def test_balanced_subset_requires_exact_source_bound_corruption() -> None:
    subset, receipt = select_balanced_subset(
        fixture_stage(),
        task_token_ids=(10, 11, 12, 13),
        separator_id=2,
        positives_per_objective=1,
    )
    assert len(subset.inputs) == 12
    assert receipt["positive_row_count"] == 5
    assert receipt["verifier_only_row_count"] == 7
    assert sum(len(pairs) for pairs in receipt["rows_by_objective"].values()) == 5
    assert receipt["missing_required_coverage"] == []
    assert receipt["required_coverage_count"] == 21
    assert receipt["informative_residual_channels"] == [
        "interaction",
        "segment",
        "token",
        "exact",
    ]


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


def test_balanced_subset_rejects_missing_interaction_contrast() -> None:
    stage = fixture_stage()
    stage.kerc_residual_labels[:, 0] = 0
    try:
        select_balanced_subset(
            stage,
            task_token_ids=(10, 11, 12, 13),
            separator_id=2,
            positives_per_objective=1,
        )
    except ValueError as exc:
        assert "residual channel contrast" in str(exc)
    else:
        raise AssertionError("interaction-free adequacy subset was admitted")


def gate_fixture() -> dict:
    return {
        "acceptance": {
            "maximum_final_to_initial_loss_ratio": 0.8,
            "minimum_token_accuracy_gain": 0.01,
            "minimum_residual_informative_channel_count": 4,
            "minimum_residual_macro_balanced_accuracy": 0.75,
            "minimum_verifier_macro_balanced_accuracy": 0.75,
            "minimum_verifier_negative_recall": 0.5,
            "maximum_checkpoint_reload_logit_delta": 1e-6,
            "maximum_optimizer_state_reload_delta": 0.0,
            "maximum_checkpoint_migration_logit_delta": 1e-6,
            "maximum_checkpoint_rollback_logit_delta": 1e-6,
            "maximum_resume_equivalence_logit_delta": 1e-5,
            "maximum_resume_equivalence_relative_logit_delta": 1e-5,
            "maximum_disabled_mechanism_activity": 0.0,
            "minimum_active_intervention_logit_delta": 1e-7,
            "maximum_inactive_residual_intervention_logit_delta": 1e-7,
            "minimum_target_tokens_per_second": 1.0,
        },
        "required_trained_ablations": [
            "without_stage_routing",
            "without_hierarchical_residual",
            "without_interaction_residual",
            "without_independent_verifier",
        ],
        "required_interventions": [
            "trusted_stage_token_removed",
            "hierarchical_residual_values_zeroed",
            "interaction_residual_values_zeroed",
            "independent_verifier_classifier_zeroed",
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
        "without_interaction_residual": {
            "after": {
                "mechanism_activity": {
                    "interaction_residual_logits_maximum_absolute": 0.0
                }
            }
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
        resume_equivalence={"equivalent": True, "discrete_outcomes_preserved": True},
        optimizer_reload_delta=0.0,
        interventions={
            "trusted_stage_token_removed": {"delta_by_objective": {name: 1e-3 for name in ("a", "b")}},
            "hierarchical_residual_values_zeroed": {
                "delta_by_objective": {"surface_to_kernel_program_v1": 1e-3, "answer_packet_to_surface_v1": 1e-3, "surface_direct_control_v1": 0.0, "kernel_program_to_answer_packet_v1": 0.0},
                "expected_active_objectives": ["surface_to_kernel_program_v1", "answer_packet_to_surface_v1"],
                "expected_inactive_objectives": ["surface_direct_control_v1", "kernel_program_to_answer_packet_v1"],
            },
            "interaction_residual_values_zeroed": {
                "delta_by_objective": {"surface_to_kernel_program_v1": 1e-3, "answer_packet_to_surface_v1": 1e-3},
                "expected_active_objectives": ["surface_to_kernel_program_v1", "answer_packet_to_surface_v1"],
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
        resume_equivalence={"equivalent": True, "discrete_outcomes_preserved": True},
        optimizer_reload_delta=0.0,
        interventions={
            "trusted_stage_token_removed": {"delta_by_objective": {"a": 1e-3}},
            "hierarchical_residual_values_zeroed": {
                "delta_by_objective": {"surface_to_kernel_program_v1": 1e-3, "answer_packet_to_surface_v1": 1e-3, "surface_direct_control_v1": 0.0, "kernel_program_to_answer_packet_v1": 0.0},
                "expected_active_objectives": ["surface_to_kernel_program_v1", "answer_packet_to_surface_v1"],
                "expected_inactive_objectives": ["surface_direct_control_v1", "kernel_program_to_answer_packet_v1"],
            },
            "interaction_residual_values_zeroed": {
                "delta_by_objective": {"surface_to_kernel_program_v1": 1e-3, "answer_packet_to_surface_v1": 1e-3},
                "expected_active_objectives": ["surface_to_kernel_program_v1", "answer_packet_to_surface_v1"],
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
