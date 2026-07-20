from __future__ import annotations

import json
import inspect
import hashlib
import base64
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
    KERC_UNIT_CANDIDATE_FEATURE_DIM,
    RaggedRows,
    architecture_training_authority,
    audit_arm_views,
    audit_specialist_data_scaling,
    audit_tokenizer_stage,
    build_source_to_target_lookup,
    behavior_diagnostics,
    checkpoint_generation_paths,
    cleanup_progress_generation,
    ensure_shared_trunk_migration,
    generate_kerc_pipeline_text,
    generate_model_text,
    inspect_checkpoint_inventory,
    kerc_global_token_rows,
    kerc_serialization_valid_ids,
    kerc_unit_allocator_training_authority,
    matched_decoder_only_config,
    materialize_kerc_unit_allocator_row,
    materialize_target_supervision,
    model_accounting,
    pack_kerc_unit_allocator_batch,
    range_view,
    serialization_valid_local_ids,
    should_evaluate_target,
    target_contracts,
    target_copy_identity_ranges,
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
from neural_seed_open_vocab import (  # noqa: E402
    TARGET_BYTE_BEGIN,
    TARGET_BYTE_END,
    reserve_byte_fallback_tokens,
)
from moecot_source_conditioned_pretraining import (  # noqa: E402
    build_kerc_code_vocabulary,
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


def test_kerc_unit_allocator_rows_materialize_ragged_source_visible_features() -> None:
    candidates = [
        {
            "fidelity_index": index,
            "encoded_bits": 8 * (index + 1),
            "hard_blocked": index < 1,
            "distortion_lower": 0.25 / (index + 1),
            "distortion_upper": 0.5 / (index + 1),
            "dimension_losses": [None if offset % 2 else 0.0 for offset in range(13)],
            "executable_pass_fraction": 1.0,
        }
        for index in range(4)
    ]
    source_visible_candidates = [
        {
            "fidelity_index": index,
            "encoded_bits": 8 * (index + 1),
            "uncompressed_bits": 64,
            "structural_loss": 0.25 / (index + 1),
            "distortion_vector": [
                None if offset % 2 else 0.0 for offset in range(13)
            ],
            "k2_hard_blocked": index < 1,
            "payload_sha256": "sha256:" + str(index) * 64,
        }
        for index in range(4)
    ]
    row = materialize_kerc_unit_allocator_row(
        {
            "kerc_residual_unit_allocator_loss_enabled": True,
            "prompt": json.dumps(
                {
                    "program": {"tokens": ["KOP:FIXTURE"]},
                    "concept_capsules": {},
                    "protected_objects": {},
                    "residual": {"tokens": ["typed-unit"]},
                }
            ),
            "kerc_residual_unit_targets": [
                {
                    "unit_id": "ru:fixture",
                    "unit_kind": "token_residue",
                    "source_path": "/token/fixture",
                    "source_payload_wire_b64": base64.b64encode(b"typed-unit").decode(),
                    "source_visible_candidates": source_visible_candidates,
                    "maximum_structural_distortion": 0.5,
                    "candidates": candidates,
                    "selected_fidelity_index": 2,
                    "confidence_target": 0.75,
                    "allocator_loss_enabled": True,
                }
            ],
        }
    )
    assert row is not None
    assert row["candidate_features"].shape == (
        1,
        4,
        KERC_UNIT_CANDIDATE_FEATURE_DIM,
    )
    assert row["hard_block_mask"].tolist() == [[True, False, False, False]]
    packed = pack_kerc_unit_allocator_batch([row, None])
    assert packed is not None
    assert packed["byte_ids"].shape == (len(b"/token/fixture\x00typed-unit"),)
    assert packed["byte_offsets"].tolist() == [
        [[0, len(b"/token/fixture\x00typed-unit")]],
        [[0, 0]],
    ]
    assert packed["unit_mask"].tolist() == [[1.0], [0.0]]
    assert packed["loss_mask"].tolist() == [[1.0], [0.0]]
    assert packed["hard_block_mask"][1].all()


def test_kerc_unit_allocator_long_training_fails_closed_without_semantic_authority() -> None:
    config = json.loads(
        (ROOT / "configs" / "moecot_language_arm_training.json").read_text()
    )
    config["kerc_unit_allocator_qualification"] = "runtime/missing-qualification.json"
    authority = kerc_unit_allocator_training_authority(config)
    assert authority["authorized"] is False
    assert authority["gaps"]


def test_ragged_rows_isolates_long_sequences_without_dense_corpus_padding() -> None:
    rows = [
        np.arange(5, dtype=np.int32),
        np.arange(9_001, dtype=np.int32),
        np.arange(7, dtype=np.int32),
        np.arange(12_000, dtype=np.int32),
    ]
    ragged = RaggedRows(rows, dtype=np.int32, standard_width=8_192)

    order = ragged.length_bucketed_order(seed=17, probabilities=None)
    batches = ragged.batch_indices(order, maximum_batch_size=2)

    assert sorted(order) == [0, 1, 2, 3]
    assert sum(batches, []) == order
    assert sorted(len(batch) for batch in batches) == [1, 1, 2]
    for batch in batches:
        widths = [len(rows[index]) for index in batch]
        assert len(batch) == 1 or max(widths) <= 8_192
        materialized = ragged[batch]
        assert materialized.shape == (len(batch), max(widths))
        for row_index, source_index in enumerate(batch):
            assert np.array_equal(
                materialized[row_index, : len(rows[source_index])],
                rows[source_index],
            )

    assert ragged.physical_bytes == sum(row.nbytes for row in rows)
    assert ragged.physical_bytes < ragged.shape[0] * ragged.shape[1] * 4


def tiny_config(tmp_path: Path) -> dict:
    return {
        "policy": "project_theseus_moecot_language_arm_training_v1",
        "seed": 7,
        "architecture_training_authority": {
            "policy": "project_theseus_pre_training_architecture_authority_v1",
            "required_for_long_optimizer_runs": True,
            "pre_training_canary_max_steps": 8,
            "gate_command": [
                "python3",
                "scripts/roadmap_implementation_gate.py",
                "--gate",
                "--require-pre-training-ready",
            ],
        },
        "generation_architecture": {
            "contract": "configs/generation_architecture_contracts.json",
            "required_policy": "project_theseus_generation_architecture_contracts_v1",
            "base_mode": "autoregressive",
            "checkpoint_shaping_auxiliary": "mtp",
            "initial_loss_scale": 0.0,
        },
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "topology": {
            "policy": "project_theseus_moecot_shared_trunk_source_specialists_v2",
            "mode": "shared_trunk_language_experts",
            "expert_adapter_dim": 4,
            "expert_trainable_scope": "adapter_only",
            "shared_trunk_bootstrap": {
                "policy": "project_theseus_exact_shared_trunk_migration_v1",
                "checkpoint": "fixture",
                "checkpoint_sha256": "a" * 64,
                "optimizer_state": "fixture",
                "optimizer_state_sha256": "b" * 64,
                "receipt": "fixture",
                "receipt_sha256": "c" * 64,
            },
        },
        "shared_trunk_model": {
            "d_model": 16,
            "num_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 1,
            "ff_dim": 32,
            "rope_base": 10000.0,
            "rms_norm_eps": 0.00001,
            "attention_policy": "causal",
            "source_target_separator_token_id": 2,
            "mtp_future_offsets": [2, 3, 4],
            "mtp_low_rank": 1,
            "mtp_loss_weights": [0.3, 0.2, 0.1],
            "mtp_loss_scale": 0.0,
            "mtp_maximum_head_parameter_overhead_ratio": 0.25,
        },
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
            "mtp_future_offsets": [2, 3, 4],
            "mtp_low_rank": 1,
            "mtp_loss_weights": [0.3, 0.2, 0.1],
            "mtp_loss_scale": 0.0,
            "mtp_maximum_head_parameter_overhead_ratio": 0.25,
            "expert_adapter_dim": 4,
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
            "maximum_kernel_english_optimizer_repetitions": 2,
            "kernel_english_optimizer_repetitions": 1,
            "termination_loss_weight": 4.0,
            "byte_boundary_loss_weight": 2.0,
        },
        "comparison_contract": {
            "preregistered_before_training": True,
            "first_campaign_candidate_ids": [
                "shared_trunk",
                "english",
                "python",
                "javascript_typescript",
                "html_css",
                "rust",
                "dense_total_parameter",
                "dense_active_parameter",
                "english_surface_control",
                "english_kerc",
            ],
        },
        "kernel_english_training": {
            "policy": "project_theseus_moecot_kernel_english_stage_v1",
            "required": True,
            "objective_order": [
                "surface_direct_control_v1",
                "surface_to_kernel_program_v1",
                "kernel_program_to_answer_packet_v1",
                "answer_packet_to_surface_v1",
            ],
            "maximum_sequence_tokens": 16384,
            "sequence_buckets": {
                "policy": "project_theseus_kerc_exact_sequence_buckets_v1",
                "routing": "encoded_length_only_without_target_semantic_metadata",
                "buckets": [
                    {
                        "bucket_id": "standard_8k",
                        "maximum_sequence_tokens": 8192,
                        "maximum_batch_size": 2,
                    },
                    {
                        "bucket_id": "exact_high_fan_in_16k",
                        "maximum_sequence_tokens": 16384,
                        "maximum_batch_size": 1,
                    },
                ],
                "truncation_allowed": False,
                "row_drop_allowed": False,
                "long_bucket_capability_credit": False,
            },
            "batch_size": 1,
            "residual_auxiliary_weight": 0.25,
            "verifier_auxiliary_weight": 0.5,
            "code_vocabulary": {
                "policy": "project_theseus_kerc_dual_code_vocabulary_v1",
                "fit_split": "private_train",
                "kernel_max_vocab": 512,
                "pointer_max_vocab": 512,
                "surface_vocabulary_owner": "canonical_moecot_target_vocab",
                "byte_fallback_required": True,
                "dev_eval_vocabulary_fit_forbidden": True,
            },
        },
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


def test_training_authority_allows_bounded_canaries_but_gates_long_runs(
    tmp_path: Path,
) -> None:
    cfg = tiny_config(tmp_path)
    calls: list[list[str]] = []

    def denied_runner(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=2, stdout="not ready", stderr="")

    canary = architecture_training_authority(cfg, max_steps=8, runner=denied_runner)
    assert canary["trigger_state"] == "GREEN"
    assert canary["authority"] == "BOUNDED_ARCHITECTURE_CANARY"
    assert canary["long_optimizer_run_authorized"] is False
    assert calls == []

    long_run = architecture_training_authority(cfg, max_steps=0, runner=denied_runner)
    assert long_run["trigger_state"] == "RED"
    assert long_run["authority"] == "DENIED"
    assert calls == [
        [
            "python3",
            "scripts/roadmap_implementation_gate.py",
            "--gate",
            "--require-pre-training-ready",
        ]
    ]


def test_deferred_kerc_path_has_zero_optimizer_and_target_exposure(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    canonical = json.loads(
        (ROOT / "configs" / "moecot_language_arm_training.json").read_text()
    )
    cfg["kernel_english_training"] = canonical["kernel_english_training"]
    active = cfg["kernel_english_training"]["disposition"]
    cfg["kernel_english_training"]["disposition"] = {
        "policy": active["policy"],
        "state": "DEFERRED_FROM_FIRST_LONG_RUN",
        "deferral_scope": "full_kerc_candidate_pending_k4_through_k8",
        "evidence_scope": "decision_grade_k0_through_k3_with_explicit_remaining_gaps",
        "terminal_evidence_state": "INCONCLUSIVE_IMPLEMENTATION",
        "full_kerc_training_enabled": False,
        "general_kerc_falsification_claimed": False,
        "learned_capability_claimed": False,
        "first_campaign_topology_exposure": 0,
        "first_campaign_optimizer_repetitions": 0,
        "retained_mechanisms": active["retained_mechanisms"],
        "qualification_evidence": active["qualification_evidence"],
        "non_claims": active["non_claims"],
    }
    cfg["kernel_english_training"]["required"] = False
    cfg["kernel_english_training"]["records_by_split"] = {
        "private_train": 0,
        "private_dev": 0,
        "private_eval": 0,
    }
    cfg["training"]["kernel_english_optimizer_repetitions"] = 0
    cfg["comparison_contract"]["first_campaign_candidate_ids"] = canonical[
        "comparison_contract"
    ]["first_campaign_candidate_ids"]

    validate_config(cfg)

    targets = target_contracts(
        cfg,
        arm_views(),
        {
            "moecot_system": {
                "shared_trunk_model": cfg["shared_trunk_model"],
                "shared_trunk_parameter_count": 10,
                "arm_model": cfg["arm_model"],
                "arm_parameter_count": 12,
            },
            "dense_total_parameter": {"model": {}, "parameter_count": 14},
            "dense_active_parameter": {"model": {}, "parameter_count": 12},
            "canonical_vocab_size": 32,
        },
        "plan",
        supervision_audit={"artifacts": {}},
        source_conditioned_audit={"artifacts": {}},
        kernel_english_audit={"artifacts": {}, "learned_pipeline_contract": {}},
    )
    assert "english_kerc" not in targets
    assert "english_surface_control" not in targets

    tampered = json.loads(json.dumps(cfg))
    tampered["training"]["kernel_english_optimizer_repetitions"] = 1
    with pytest.raises(
        ValueError, match="retired KERC path must receive zero optimizer repetitions"
    ):
        validate_config(tampered)

    tampered = json.loads(json.dumps(cfg))
    tampered["comparison_contract"]["first_campaign_candidate_ids"].append(
        "english_kerc"
    )
    with pytest.raises(ValueError, match="first-campaign candidate inventory mismatch"):
        validate_config(tampered)

    tampered = json.loads(json.dumps(cfg))
    tampered["kernel_english_training"]["disposition"][
        "general_kerc_falsification_claimed"
    ] = True
    with pytest.raises(ValueError, match="KERC_TRAINING_DISPOSITION_INVALID"):
        validate_config(tampered)


def test_arm_views_are_an_exact_non_overlapping_partition() -> None:
    accepted = audit_arm_views(arm_views(), 10)
    assert accepted["state"] == "GREEN"
    assert accepted["non_overlapping_complete_partition"] is True

    tampered = arm_views()
    tampered["arms"]["python"]["row_ranges"][0]["start"] = 1
    rejected = audit_arm_views(tampered, 10)
    assert rejected["state"] == "RED"
    assert any(
        gap.startswith("arm_range_gap_or_overlap:python")
        for gap in rejected["hard_gaps"]
    )


def test_specialist_data_scaling_binds_each_parameter_owner() -> None:
    base = {
        "data_model_scaling_contract": {
            "planning_basis": {"minimum_unique_positions_per_active_parameter": 20.0}
        }
    }
    targets = {
        "shared_trunk": {"unique_target_positions": 2000},
        **{arm: {"unique_target_positions": 200} for arm in ARM_IDS},
    }
    models = {
        "moecot_system": {
            "shared_trunk_parameter_count": 100,
            "expert_parameter_count_per_arm": 10,
        }
    }
    accepted = audit_specialist_data_scaling(base, targets, models)
    assert accepted["state"] == "GREEN"
    assert all(row["meets_floor"] for row in accepted["rows"])

    targets["html_css"]["unique_target_positions"] = 199
    rejected = audit_specialist_data_scaling(base, targets, models)
    assert rejected["state"] == "RED"
    assert rejected["hard_gaps"] == [
        "specialist_unique_position_floor_not_met:html_css"
    ]


def test_progress_generation_paths_and_cleanup_are_step_scoped(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.npz"
    optimizer = tmp_path / "optimizer.safetensors"
    old_checkpoint, old_optimizer = checkpoint_generation_paths(
        checkpoint, optimizer, 500
    )
    new_checkpoint, new_optimizer = checkpoint_generation_paths(
        checkpoint, optimizer, 1000
    )
    for path in (
        checkpoint,
        optimizer,
        old_checkpoint,
        old_optimizer,
        new_checkpoint,
        new_optimizer,
    ):
        path.write_bytes(path.name.encode())

    cleanup_progress_generation(
        {
            "checkpoint": str(old_checkpoint),
            "optimizer_state": str(old_optimizer),
        },
        canonical_checkpoint=checkpoint,
        canonical_optimizer=optimizer,
        keep={new_checkpoint, new_optimizer},
    )

    assert not old_checkpoint.exists()
    assert not old_optimizer.exists()
    assert checkpoint.exists() and optimizer.exists()
    assert new_checkpoint.exists() and new_optimizer.exists()

    with pytest.raises(ValueError, match="step must be positive"):
        checkpoint_generation_paths(checkpoint, optimizer, 0)


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


def test_only_executable_compositions_receive_direct_evaluation() -> None:
    assert should_evaluate_target({"role": "language_expert"}) is True
    assert should_evaluate_target({"role": "dense_control"}) is True
    assert should_evaluate_target({"role": "english_surface_control"}) is True
    assert should_evaluate_target({"role": "kerc_english_candidate"}) is True
    assert should_evaluate_target({"role": "shared_trunk"}) is False
    assert should_evaluate_target({"role": ""}) is False


def test_shared_trunk_migration_is_exact_and_rejects_source_tampering(
    tmp_path: Path,
) -> None:
    import hashlib
    import mlx.core as mx
    import mlx.nn as nn

    model_config = tiny_config(tmp_path)["shared_trunk_model"]
    source_dir = tmp_path / "v6" / "shared_trunk"
    source_dir.mkdir(parents=True)
    source_checkpoint = source_dir / "weights.npz"
    source_optimizer = source_dir / "optimizer.safetensors"
    source_receipt_path = source_dir / "training_receipt.json"
    model = build_model(
        CausalTransformerConfig(vocab_size=64, **model_config), mx=mx, nn=nn
    )
    model.save_weights(str(source_checkpoint))
    source_optimizer.write_bytes(b"optimizer-state")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source_receipt = {
        "policy": "project_theseus_moecot_language_arm_training_receipt_v1",
        "target_id": "shared_trunk",
        "role": "shared_trunk",
        "plan_sha256": "old-plan",
        "stage_signature": "stage-migrate",
        "row_ranges": [{"start": 0, "stop": 2}],
        "complete": True,
        "optimizer_steps": 4,
        "optimizer_positions": 32,
        "checkpoint_sha256": digest(source_checkpoint),
        "optimizer_state_sha256": digest(source_optimizer),
        "capability_claim": "NOT_EVALUATED",
    }
    source_receipt_path.write_text(json.dumps(source_receipt))
    destination = tmp_path / "v7" / "shared_trunk"
    target = {
        "target_id": "shared_trunk",
        "role": "shared_trunk",
        "row_ranges": [{"start": 0, "stop": 2}],
        "model": model_config,
        "checkpoint": str(destination / "weights.npz"),
        "optimizer_state": str(destination / "optimizer.safetensors"),
        "receipt": str(destination / "training_receipt.json"),
    }
    config = tiny_config(tmp_path)
    config["topology"]["shared_trunk_bootstrap"] = {
        "policy": "project_theseus_exact_shared_trunk_migration_v1",
        "checkpoint": str(source_checkpoint),
        "checkpoint_sha256": digest(source_checkpoint),
        "optimizer_state": str(source_optimizer),
        "optimizer_state_sha256": digest(source_optimizer),
        "receipt": str(source_receipt_path),
        "receipt_sha256": digest(source_receipt_path),
    }
    plan = {
        "plan_sha256": "new-plan",
        "stage": {"stage_signature": "stage-migrate"},
        "models": {"vocab_size": 64},
        "targets": {"shared_trunk": target},
    }
    migrated = ensure_shared_trunk_migration(
        config,
        plan,
        metadata={"source_vocab": {"<pad>": 0}, "target_vocab": {"<pad>": 0}},
        base={"tokenization": {"shared_source_target_vocabulary": True}},
        mx=mx,
        nn=nn,
    )
    assert migrated["plan_sha256"] == "new-plan"
    assert migrated["migration"]["training_positions_added"] == 0
    assert digest(destination / "weights.npz") == digest(source_checkpoint)
    assert digest(destination / "optimizer.safetensors") == digest(source_optimizer)

    for path in destination.iterdir():
        path.unlink()
    source_checkpoint.write_bytes(source_checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="source checkpoint identity mismatch"):
        ensure_shared_trunk_migration(
            config,
            plan,
            metadata={"source_vocab": {"<pad>": 0}, "target_vocab": {"<pad>": 0}},
            base={"tokenization": {"shared_source_target_vocabulary": True}},
            mx=mx,
            nn=nn,
        )


def test_fresh_shared_trunk_initialization_is_seed_bound_and_expert_denied(
    tmp_path: Path,
) -> None:
    config = tiny_config(tmp_path)
    config["topology"].pop("shared_trunk_bootstrap")
    config["topology"]["policy"] = (
        "project_theseus_moecot_scaled_low_rank_specialists_v3"
    )
    config["topology"]["shared_trunk_initialization"] = {
        "policy": "project_theseus_seeded_fresh_trunk_initialization_v1",
        "seed": config["seed"],
        "reason": "isolated larger-shape test",
    }
    target_dir = tmp_path / "fresh" / "shared_trunk"
    plan = {
        "stage": {"stage_signature": "fresh-stage"},
        "targets": {
            "shared_trunk": {
                "checkpoint": str(target_dir / "weights.npz"),
                "optimizer_state": str(target_dir / "optimizer.safetensors"),
                "receipt": str(target_dir / "training_receipt.json"),
            }
        },
    }
    authorized = ensure_shared_trunk_migration(
        config,
        plan,
        metadata={},
        base={},
        mx=None,
        nn=None,
        require_existing=False,
    )
    assert authorized["state"] == "FRESH_INITIALIZATION_AUTHORIZED"
    assert authorized["training_positions_added"] == 0
    assert authorized["capability_credit"] == "NONE"

    with pytest.raises(ValueError, match="requires a completed fresh shared trunk"):
        ensure_shared_trunk_migration(
            config,
            plan,
            metadata={},
            base={},
            mx=None,
            nn=None,
            require_existing=True,
        )

    config["topology"]["shared_trunk_initialization"]["seed"] += 1
    with pytest.raises(ValueError, match="seed mismatch"):
        ensure_shared_trunk_migration(
            config,
            plan,
            metadata={},
            base={},
            mx=None,
            nn=None,
            require_existing=False,
        )


def test_behavior_diagnostics_are_evaluator_only_and_do_not_retain_text() -> None:
    prompt = (
        "Apply the requested change to this rust excerpt.\n\nRequest:\nRename it."
        "\n\nCurrent excerpt:\nlet old = 1;\n\n\n"
        "Return only the complete revised excerpt."
    )
    diagnostics = behavior_diagnostics(
        generated="let new = 1;\n",
        expected="let new = 1;\n",
        prompt=prompt,
    )
    assert diagnostics["target_sequence_similarity"] == 1.0
    assert diagnostics["source_excerpt_available"] is True
    assert diagnostics["source_sequence_similarity"] < 1.0
    assert diagnostics["raw_generated_text_retained"] is False
    assert "generated" not in diagnostics and "expected" not in diagnostics


def test_range_view_coalesces_adjacent_ranges_without_copy() -> None:
    array = np.arange(24).reshape(6, 4)
    view = range_view(array, [{"start": 1, "stop": 3}, {"start": 3, "stop": 5}])
    assert np.shares_memory(array, view)
    assert view.tolist() == array[1:5].tolist()


def test_exact_supervision_masks_only_target_and_never_truncates(
    tmp_path: Path,
) -> None:
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
        "tokenization": {
            "max_sequence_tokens": 32,
            "shared_source_target_vocabulary": False,
        }
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

    kerc_target = {
        **target,
        "target_id": "english_kerc",
        "role": "kerc_english_candidate",
    }
    ordinary_stage = materialize_target_supervision(
        training_config,
        base,
        kerc_target,
        metadata={"source_vocab": source_vocab, "target_vocab": target_vocab},
    )
    assert ordinary_stage.kerc_residual_labels is None
    assert ordinary_stage.kerc_residual_loss_mask is None
    assert ordinary_stage.kerc_verifier_labels is None
    assert ordinary_stage.receipt["dual_code_vocabulary_sha256"] == ""

    base["tokenization"]["max_sequence_tokens"] = 4
    widened = materialize_target_supervision(
        training_config,
        base,
        target,
        metadata={"source_vocab": source_vocab, "target_vocab": target_vocab},
        artifact_field="supervision_artifacts",
        receipt_policy="project_theseus_moecot_kernel_english_arrays_v1",
        maximum_sequence_tokens=32,
    )
    assert widened.receipt["sequence_width"] < 32
    assert widened.receipt["maximum_sequence_tokens_contract"] == 32
    assert widened.receipt["staged_padding_columns_elided"] > 0
    assert widened.receipt["sequence_width_source"] == "objective_override"
    with pytest.raises(ValueError, match="requires truncation"):
        materialize_target_supervision(
            training_config,
            base,
            target,
            metadata={"source_vocab": source_vocab, "target_vocab": target_vocab},
        )


def test_kerc_materialization_trains_verifier_negatives_without_generator_credit(
    tmp_path: Path,
) -> None:
    source_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    target_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    reserve_byte_fallback_tokens(source_vocab, max_vocab=270, stream="source")
    reserve_byte_fallback_tokens(target_vocab, max_vocab=270, stream="target")
    source_vocab["<KERC_TASK_SURFACE_TO_KERNEL>"] = len(source_vocab)
    row = {
        "split": "private_train",
        "arm_id": "english",
        "objective": "surface_to_kernel_program_v1",
        "public_benchmark": False,
        "prompt": "Compile this governed sentence.",
        "target": '{"program":"valid"}',
        "trusted_source_prefix_tokens": ["<KERC_TASK_SURFACE_TO_KERNEL>"],
        "trusted_prefix_authority": "internal_objective_route_only",
        "optimizer_sampling_weight": 0.25,
        "kerc_residual_channels": ["interaction", "segment", "token", "exact"],
        "kerc_residual_labels": [1, 0, 0, 3],
        "kerc_verifier_dimensions": [
            "semantic_consistency",
            "protected_object_consistency",
            "numeric_value_consistency",
            "surface_fidelity",
            "answer_decision_consistency",
        ],
        "kerc_verifier_positive_labels": [1, 1, 1, 1, 1],
        "kerc_answer_disposition": "ANSWER",
        "kerc_verifier_negative": {
            "target": '{"program":"corrupted"}',
            "target_sha256": "sha256:"
            + hashlib.sha256(b'{"program":"corrupted"}').hexdigest(),
            "labels": [0, 1, 1, 1, 1],
            "failed_dimension": "semantic_consistency",
            "generator_loss_enabled": False,
        },
        "kerc_context_counterfactuals": [
            {
                "strategy": strategy,
                "prompt": counter_prompt,
                "prompt_sha256": "sha256:"
                + hashlib.sha256(counter_prompt.encode()).hexdigest(),
                "target": '{"program":"valid"}',
                "target_sha256": "sha256:"
                + hashlib.sha256(b'{"program":"valid"}').hexdigest(),
                "labels": [0, 1, 1, 1, 0],
                "failed_dimensions": [
                    "semantic_consistency",
                    "answer_decision_consistency",
                ],
                "generator_loss_enabled": False,
                "unique_source_credit": 0,
                "candidate_generation_credit": 0,
            }
            for strategy, counter_prompt in (
                ("context_withheld", "Compile without document support."),
                ("context_shuffled", "Compile against unrelated support."),
            )
        ],
    }
    artifact = tmp_path / "kerc.jsonl"
    artifact.write_text(json.dumps(row) + "\n")
    code_vocabulary = build_kerc_code_vocabulary(
        [row],
        {"kernel_max_vocab": 512, "pointer_max_vocab": 512},
    )

    target = {
        "target_id": "english_kerc",
        "role": "kerc_english_candidate",
        "model": {
            "kerc_kernel_token_start": 600,
            "kerc_pointer_token_start": 1200,
        },
        "kernel_code_vocabulary": {"payload": code_vocabulary},
        "kernel_english_artifacts": {
            "private_train": {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "row_count": 1,
            }
        },
    }
    stage = materialize_target_supervision(
        {
            "training": {
                "termination_loss_weight": 4.0,
                "byte_boundary_loss_weight": 2.0,
            }
        },
        {"tokenization": {"max_sequence_tokens": 128}},
        target,
        metadata={"source_vocab": source_vocab, "target_vocab": target_vocab},
        artifact_field="kernel_english_artifacts",
        receipt_policy="project_theseus_moecot_kernel_english_arrays_v1",
        maximum_sequence_tokens=128,
        objective_filter=("surface_to_kernel_program_v1",),
    )

    assert stage.inputs.shape[0] == 4
    assert isinstance(stage.inputs, RaggedRows)
    assert stage.receipt["storage_layout"] == "ragged_rows_dynamic_batch_padding_v1"
    assert (
        stage.receipt["physical_array_bytes"]
        <= stage.receipt["dense_equivalent_array_bytes"]
    )
    assert int(stage.mask[0].sum()) > 0
    assert int(stage.mask[1].sum()) == 0
    assert int(stage.mask[2].sum()) == 0
    assert int(stage.mask[3].sum()) == 0
    assert stage.receipt["generator_training_row_count"] == 1
    assert stage.receipt["verifier_only_row_count"] == 3
    assert stage.kerc_residual_labels.tolist() == [[1, 0, 0, 3]] * 4
    assert stage.kerc_residual_loss_mask.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert stage.receipt["kerc_residual_supervision_row_count"] == 1
    assert stage.receipt["kerc_verifier_only_rows_receive_residual_loss"] is False
    assert stage.kerc_verifier_labels.tolist() == [
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
    ]
    assert stage.receipt["kerc_verifier_dimensions"][-1] == (
        "answer_decision_consistency"
    )
    assert stage.kerc_coverage_labels[0][-1] == "verifier:positive"
    assert stage.kerc_coverage_labels[1][-1] == (
        "verifier:negative:semantic_consistency"
    )
    assert stage.kerc_coverage_labels[2][-1] == (
        "verifier:counterfactual:context_withheld"
    )
    assert stage.kerc_coverage_labels[3][-1] == (
        "verifier:counterfactual:context_shuffled"
    )
    assert stage.sample_weights.tolist() == [0.25] * 4
    assert stage.receipt["sampling_weight_sum"] == 1.0
    assert stage.receipt["kerc_context_counterfactual_counts"] == {
        "context_withheld": 1,
        "context_shuffled": 1,
    }
    assert (
        stage.receipt["dual_code_vocabulary_sha256"]
        == code_vocabulary["contract_sha256"]
    )
    positive_ids = set(int(value) for value in stage.inputs[0])
    assert any(600 <= value < 1200 for value in positive_ids)
    assert any(value >= 1200 for value in positive_ids)

    order = stage.inputs.length_bucketed_order(seed=11, probabilities=None)
    assert sorted(order) == [0, 1, 2, 3]
    assert sum(stage.inputs.batch_indices(order, maximum_batch_size=2), []) == order

    row["kerc_context_counterfactuals"][0]["generator_loss_enabled"] = True
    artifact.write_text(json.dumps(row) + "\n")
    target["kernel_english_artifacts"]["private_train"]["sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="context counterfactual contract"):
        materialize_target_supervision(
            {
                "training": {
                    "termination_loss_weight": 4.0,
                    "byte_boundary_loss_weight": 2.0,
                }
            },
            {"tokenization": {"max_sequence_tokens": 128}},
            target,
            metadata={"source_vocab": source_vocab, "target_vocab": target_vocab},
            artifact_field="kernel_english_artifacts",
            receipt_policy="project_theseus_moecot_kernel_english_arrays_v1",
            maximum_sequence_tokens=128,
            objective_filter=("surface_to_kernel_program_v1",),
        )


def test_kerc_dual_vocab_is_charged_only_to_candidate_and_surface_control_is_matched() -> (
    None
):
    config = json.loads(
        (ROOT / "configs" / "moecot_language_arm_training.json").read_text()
    )
    kernel_cfg = config["kernel_english_training"]
    deferred = kernel_cfg["disposition"]
    kernel_cfg["required"] = True
    kernel_cfg["records_by_split"] = kernel_cfg[
        "deferred_candidate_records_by_split"
    ]
    kernel_cfg["disposition"] = {
        "policy": deferred["policy"],
        "state": "CANDIDATE_REQUIRED",
        "qualification_scope": "faithful_full_compiler_core_renderer_candidate",
        "basis": "adequacy_audit_reopened_after_toy_proxy",
        "full_kerc_training_enabled": True,
        "general_kerc_falsification_claimed": False,
        "learned_capability_claimed": False,
        "retained_mechanisms": [],
        "superseded_proxy_evidence": deferred["superseded_proxy_evidence"],
        "non_claims": deferred["non_claims"],
    }
    base = json.loads((ROOT / config["base_config"]).read_text())
    metadata = json.loads(
        (ROOT / config["stage_dir"] / "stage_metadata_v1.json").read_text()
    )
    models = model_accounting(config, base, metadata)
    kerc = models["english_kerc"]
    control = models["english_surface_control"]

    assert models["kerc_vocab_size"] == (
        models["canonical_vocab_size"]
        + config["kernel_english_training"]["code_vocabulary"]["kernel_max_vocab"]
        + config["kernel_english_training"]["code_vocabulary"]["pointer_max_vocab"]
    )
    assert kerc["vocab_size"] == models["kerc_vocab_size"]
    assert control["vocab_size"] == models["canonical_vocab_size"]
    assert abs(control["parameter_delta_vs_kerc"]) / kerc["parameter_count"] < 0.001
    model = kerc["model"]
    assert model["kerc_surface_token_end"] == models["canonical_vocab_size"]
    assert model["kerc_kernel_token_start"] == models["canonical_vocab_size"]
    assert model["kerc_pointer_token_end"] == models["kerc_vocab_size"]


def test_generation_api_cannot_receive_hidden_target() -> None:
    parameters = inspect.signature(generate_model_text).parameters
    assert "prompt" in parameters
    assert "target" not in parameters
    assert "expected" not in parameters
    pipeline_parameters = inspect.signature(generate_kerc_pipeline_text).parameters
    assert "prompt" in pipeline_parameters
    assert "expected" not in pipeline_parameters


def test_kerc_code_decoder_keeps_byte_fallback_inside_one_code_space() -> None:
    code_vocabulary = {
        "kernel_vocab": {
            "<pad>": 0,
            "<unk>": 1,
            TARGET_BYTE_BEGIN: 2,
            "<byte:41>": 3,
            TARGET_BYTE_END: 4,
            "PROGRAM": 5,
        },
        "pointer_vocab": {
            "<pad>": 0,
            "<unk>": 1,
            TARGET_BYTE_BEGIN: 2,
            "<byte:42>": 3,
            TARGET_BYTE_END: 4,
            "@E1": 5,
        },
    }
    rows = kerc_global_token_rows(
        code_vocabulary, kernel_offset=100, pointer_offset=200, pointer_end=300
    )
    open_ids = kerc_serialization_valid_ids([], rows, end_id=10)
    assert 10 in open_ids
    assert 102 in open_ids and 202 in open_ids
    kernel_byte_ids = kerc_serialization_valid_ids(
        [{"space": "V_K", "token": TARGET_BYTE_BEGIN}], rows, end_id=10
    )
    assert 103 in kernel_byte_ids and 104 in kernel_byte_ids
    assert 203 not in kernel_byte_ids and 204 not in kernel_byte_ids
    assert 10 not in kernel_byte_ids


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

    lookup = np.full(64, -1, dtype=np.int32)
    lookup[10] = 20
    lookup[11] = 21
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
            source_copy_mode="pointer_generator",
            source_copy_auxiliary_loss_weight=0.25,
        ),
        mx=mx,
        nn=nn,
        source_to_target_lookup=lookup,
    )
    inputs = mx.array([[1, 10, 11, 2, 20, 21]], dtype=mx.int32)
    labels = mx.array([[10, 11, 2, 20, 21, 3]], dtype=mx.int32)
    target_only_mask = mx.array([[0, 0, 0, 0, 1, 1]], dtype=mx.float32)
    loss_and_grad = nn.value_and_grad(model, causal_loss)
    loss, gradients = loss_and_grad(model, inputs, labels, target_only_mask, mx, nn)
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss.item()))
    gradient_mass = {
        name: float(mx.sum(mx.abs(value)).item())
        for name, value in mlx_utils.tree_flatten(gradients)
    }
    assert (
        sum(
            value
            for name, value in gradient_mass.items()
            if name.startswith("source_layers.")
        )
        > 0.0
    )
    assert (
        sum(
            value
            for name, value in gradient_mass.items()
            if ".source_attention." in name
        )
        > 0.0
    )
    assert (
        sum(value for name, value in gradient_mass.items() if name.startswith("copy_"))
        > 0.0
    )


def test_source_target_lookup_uses_exact_token_identity_only() -> None:
    base = {"tokenization": {"shared_source_target_vocabulary": False}}
    metadata = {
        "source_vocab": {"<pad>": 0, "same": 1, "source-only": 2},
        "target_vocab": {"<pad>": 0, "same": 1, "target-only": 2},
    }
    lookup = build_source_to_target_lookup(base, metadata)
    source_offset = 3
    target_offset = 6
    assert int(lookup[source_offset + 1]) == target_offset + 1
    assert int(lookup[source_offset + 2]) == -1

    structured = build_source_to_target_lookup(
        base,
        metadata,
        vocab_size=12,
        identity_ranges=((9, 12),),
    )
    assert structured[9:12].tolist() == [9, 10, 11]


def test_kerc_copy_identity_ranges_cover_all_shared_code_spaces() -> None:
    target = {
        "role": "kerc_english_candidate",
        "model": {
            "kerc_surface_token_start": 10,
            "kerc_surface_token_end": 20,
            "kerc_kernel_token_start": 20,
            "kerc_kernel_token_end": 30,
            "kerc_pointer_token_start": 30,
            "kerc_pointer_token_end": 40,
        },
    }

    assert target_copy_identity_ranges(target) == ((10, 20), (20, 30), (30, 40))


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


def test_tiny_mlx_arm_writes_distinct_resumable_model_and_optimizer_state(
    tmp_path: Path,
) -> None:
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
    stage = SimpleNamespace(
        pretrain_inputs=inputs, pretrain_labels=labels, pretrain_mask=mask
    )
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
    assert (
        json.loads(receipt_path.read_text())["optimizer_state_sha256"]
        == second["optimizer_state_sha256"]
    )


def test_source_and_kernel_phases_are_accounted_separately_before_sft(
    tmp_path: Path,
) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    config = tiny_config(tmp_path)
    model = build_model(
        CausalTransformerConfig(vocab_size=64, **config["arm_model"]), mx=mx, nn=nn
    )
    count = int(parameter_count(model, mlx_utils))
    base_inputs = np.asarray([[1, 4, 5, 6]], dtype=np.int32)
    base_stage = SimpleNamespace(
        pretrain_inputs=base_inputs,
        pretrain_labels=np.asarray([[4, 5, 6, 2]], dtype=np.int32),
        pretrain_mask=np.ones_like(base_inputs, dtype=np.uint8),
    )
    source_stage = SimpleNamespace(
        inputs=np.asarray([[1, 10, 2, 20], [1, 11, 2, 21]], dtype=np.int32),
        labels=np.asarray([[10, 2, 20, 3], [11, 2, 21, 3]], dtype=np.int32),
        mask=np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8),
        loss_mask=np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
        receipt={"policy": "project_theseus_moecot_source_conditioned_arrays_v1"},
    )
    kernel_stage = SimpleNamespace(
        inputs=np.asarray([[1, 12, 2, 22, 0, 0, 0, 0]], dtype=np.int32),
        labels=np.asarray([[12, 2, 22, 3, 0, 0, 0, 0]], dtype=np.int32),
        mask=np.asarray([[0, 0, 1, 1, 0, 0, 0, 0]], dtype=np.uint8),
        loss_mask=np.asarray([[0, 0, 1, 1, 0, 0, 0, 0]], dtype=np.float32),
        receipt={"policy": "project_theseus_moecot_kernel_english_arrays_v1"},
    )
    checkpoint = tmp_path / "checkpoints" / "english" / "weights.npz"
    target = {
        "target_id": "english",
        "role": "language_arm",
        "row_ranges": [{"start": 0, "stop": 1}],
        "unique_target_positions": 0,
        "model": config["arm_model"],
        "parameter_count": count,
        "checkpoint": str(checkpoint),
        "optimizer_state": str(checkpoint.parent / "optimizer.safetensors"),
        "receipt": str(checkpoint.parent / "training_receipt.json"),
    }
    plan = {
        "plan_sha256": "c" * 64,
        "stage": {"stage_signature": "stage-c", "metadata_sha256": "d" * 64},
        "models": {"vocab_size": 64},
    }
    result = train_target(
        config,
        plan,
        target,
        stage=base_stage,
        source_conditioned_stage=source_stage,
        kernel_english_stage=kernel_stage,
        max_steps=2,
        resume=False,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
    )
    assert result["pretrain_optimizer_positions"] == 0
    assert result["source_conditioned_optimizer_positions"] == 4
    assert result["kernel_english_optimizer_positions"] == 2
    assert result["supervision_optimizer_positions"] == 0
    assert result["phases"]["source_conditioned_pretraining"]["optimizer_steps"] == 1
    assert result["phases"]["kernel_english"]["optimizer_steps"] == 1
    assert result["phases"]["kernel_english"]["static_sequence_width"] == 8
    assert result["phases"]["kernel_english"]["maximum_dynamic_batch_width"] == 4
    assert result["phases"]["kernel_english"]["padded_positions_avoided"] == 4
    assert result["source_conditioned_stage"]["policy"].endswith("arrays_v1")
    assert result["kernel_english_stage"]["policy"].endswith("arrays_v1")

    canary_target = {
        **target,
        "checkpoint": str(tmp_path / "canary" / "weights.npz"),
        "optimizer_state": str(tmp_path / "canary" / "optimizer.safetensors"),
        "receipt": str(tmp_path / "canary" / "training_receipt.json"),
    }
    canary = train_target(
        config,
        plan,
        canary_target,
        stage=base_stage,
        source_conditioned_stage=source_stage,
        kernel_english_stage=kernel_stage,
        max_steps=1,
        resume=False,
        training_phase="kernel_english",
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
    )
    assert canary["pretrain_optimizer_positions"] == 0
    assert canary["source_conditioned_optimizer_positions"] == 0
    assert canary["kernel_english_optimizer_positions"] == 2
    assert canary["training_phase_selection"] == "kernel_english"
    assert canary["bounded_phase_canary"] is True
    assert canary["complete"] is False


def test_stale_bounded_canary_does_not_poison_fresh_plan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.npz"
    optimizer = tmp_path / "optimizer.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    optimizer.write_bytes(b"optimizer")
    receipt = tmp_path / "training_receipt.json"
    target = {
        "target_id": "english_kerc",
        "receipt": str(receipt),
        "checkpoint": str(checkpoint),
        "optimizer_state": str(optimizer),
        "row_ranges": [{"start": 0, "stop": 1}],
        "parameter_count": 1,
        "vocab_size": 8,
    }
    receipt.write_text(
        json.dumps(
            {
                "target_id": "english_kerc",
                "plan_sha256": "old-plan",
                "stage_signature": "stage",
                "row_ranges": target["row_ranges"],
                "parameter_count": 1,
                "vocab_size": 8,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "optimizer_state": str(optimizer),
                "optimizer_state_sha256": hashlib.sha256(
                    optimizer.read_bytes()
                ).hexdigest(),
                "optimizer_steps": 1,
                "optimizer_positions": 10,
                "complete": False,
                "bounded_phase_canary": True,
                "capability_claim": "NOT_EVALUATED",
            }
        )
    )

    inventory = inspect_checkpoint_inventory(
        {"english_kerc": target}, "new-plan", "stage"
    )
    assert inventory["state"] == "NOT_RUN"
    assert inventory["hard_gaps"] == []
    assert inventory["stale_canary_count"] == 1
    assert inventory["rows"][0]["state"] == "STALE_CANARY"


def test_shared_trunk_and_expert_checkpoint_ownership_are_separate(
    tmp_path: Path,
) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    config = tiny_config(tmp_path)
    trunk_model = build_model(
        CausalTransformerConfig(vocab_size=64, **config["shared_trunk_model"]),
        mx=mx,
        nn=nn,
    )
    expert_model = build_model(
        CausalTransformerConfig(vocab_size=64, **config["arm_model"]),
        mx=mx,
        nn=nn,
    )
    trunk_count = int(parameter_count(trunk_model, mlx_utils))
    expert_total = int(parameter_count(expert_model, mlx_utils))
    expert_delta = expert_total - trunk_count
    inputs = np.asarray([[1, 4, 5, 6], [1, 7, 8, 9]], dtype=np.int32)
    base_stage = SimpleNamespace(
        pretrain_inputs=inputs,
        pretrain_labels=np.asarray([[4, 5, 6, 2], [7, 8, 9, 2]], dtype=np.int32),
        pretrain_mask=np.ones_like(inputs, dtype=np.uint8),
    )
    root = tmp_path / "checkpoints"
    trunk_checkpoint = root / "shared_trunk" / "weights.npz"
    plan = {
        "plan_sha256": "e" * 64,
        "stage": {"stage_signature": "stage-e", "metadata_sha256": "f" * 64},
        "models": {
            "vocab_size": 64,
            "moecot_system": {"expert_parameter_count_per_arm": expert_delta},
        },
    }
    trunk_target = {
        "target_id": "shared_trunk",
        "role": "shared_trunk",
        "row_ranges": [{"start": 0, "stop": 2}],
        "unique_target_positions": 8,
        "model": config["shared_trunk_model"],
        "parameter_count": trunk_count,
        "checkpoint": str(trunk_checkpoint),
        "optimizer_state": str(trunk_checkpoint.parent / "optimizer.safetensors"),
        "receipt": str(trunk_checkpoint.parent / "training_receipt.json"),
    }
    trunk_result = train_target(
        config,
        plan,
        trunk_target,
        stage=base_stage,
        max_steps=1,
        resume=False,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
    )
    assert trunk_result["complete"] is True

    expert_checkpoint = root / "rust" / "expert_adapter.safetensors"
    source_stage = SimpleNamespace(
        inputs=np.asarray([[1, 10, 2, 20], [1, 11, 2, 21]], dtype=np.int32),
        labels=np.asarray([[10, 2, 20, 3], [11, 2, 21, 3]], dtype=np.int32),
        mask=np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8),
        loss_mask=np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
        receipt={"policy": "project_theseus_moecot_source_conditioned_arrays_v1"},
    )
    expert_target = {
        "target_id": "rust",
        "role": "language_expert",
        "row_ranges": [{"start": 0, "stop": 2}],
        "unique_target_positions": 0,
        "model": config["arm_model"],
        "parameter_count": expert_total,
        "checkpoint": str(expert_checkpoint),
        "shared_trunk_checkpoint": str(trunk_checkpoint),
        "optimizer_state": str(expert_checkpoint.parent / "optimizer.safetensors"),
        "receipt": str(expert_checkpoint.parent / "training_receipt.json"),
    }
    expert_result = train_target(
        config,
        plan,
        expert_target,
        stage=base_stage,
        source_conditioned_stage=source_stage,
        max_steps=1,
        resume=False,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
    )
    assert expert_result["trainable_parameter_count"] == expert_delta
    assert (
        expert_result["shared_trunk_checkpoint_sha256"]
        == trunk_result["checkpoint_sha256"]
    )
    adapter_keys = set(mx.load(str(expert_checkpoint)))
    assert adapter_keys
    assert all(".expert_adapter." in key for key in adapter_keys)
