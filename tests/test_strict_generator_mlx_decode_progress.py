from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from strict_generator_mlx_decode_reporting import (  # noqa: E402
    bind_decode_progress_split,
    checkpoint_evaluation_lineage,
    commit_decode_progress_batch,
    compose_transitive_checkpoint_training_lineage,
    family_disjoint_checkpoint_lineage_gate,
    initialize_decode_progress,
    resolve_checkpoint_paths_from_report,
)
from strict_generator_mlx_adaptation_selection import (  # noqa: E402
    SEMANTIC_CONSTRUCTION_REPAIR_PROFILES,
)
from strict_generator_mlx_private_adaptation import (  # noqa: E402
    semantic_profile_checkpoint_compatibility,
)


def _contract(value: str = "a") -> dict[str, object]:
    return {
        "policy": "strict_mlx_decode_run_contract_v1",
        "checkpoint_sha256": value * 64,
        "vocab_sha256": "b" * 64,
        "decode_cache_mode": "incremental",
    }


def test_progress_commit_is_atomic_private_and_resumable(tmp_path: Path) -> None:
    path = tmp_path / "decode.progress.json"
    progress = initialize_decode_progress(path, run_contract=_contract(), resume=False)
    split = bind_decode_progress_split(
        progress,
        path,
        split_name="family_disjoint",
        task_input_hashes=["task-a", "task-b"],
    )
    assert split["completed"] == {}

    commit_decode_progress_batch(
        progress,
        path,
        split_name="family_disjoint",
        records=[
            {
                "task_index": 0,
                "task_input_hash": "task-a",
                "decoded": [{"body": "return value"}],
                "diagnostic": {"accepted_candidate_rows": 1},
            }
        ],
        batch_receipt={"decode_cache_receipt": {"mode": "incremental"}},
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["splits"]["family_disjoint"]["completed"]["0"]["decoded"]
    assert persisted["public_training_rows"] == 0
    assert persisted["external_inference_calls"] == 0
    assert "prompt" not in json.dumps(persisted).lower()
    assert not list(tmp_path.glob("*.tmp"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    resumed = initialize_decode_progress(path, run_contract=_contract(), resume=True)
    assert resumed["resume_count"] == 1
    assert len(resumed["splits"]["family_disjoint"]["completed"]) == 1


def test_progress_rejects_stale_contract_and_task_inventory(tmp_path: Path) -> None:
    path = tmp_path / "decode.progress.json"
    progress = initialize_decode_progress(path, run_contract=_contract(), resume=False)
    bind_decode_progress_split(
        progress,
        path,
        split_name="family_disjoint",
        task_input_hashes=["task-a"],
    )
    with pytest.raises(ValueError, match="run contract mismatch"):
        initialize_decode_progress(path, run_contract=_contract("c"), resume=True)
    with pytest.raises(ValueError, match="task inventory mismatch"):
        bind_decode_progress_split(
            progress,
            path,
            split_name="family_disjoint",
            task_input_hashes=["task-other"],
        )


def test_progress_rejects_corruption_and_symlink_paths(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        initialize_decode_progress(corrupt, run_contract=_contract(), resume=True)

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symlinked"):
        initialize_decode_progress(symlink, run_contract=_contract(), resume=False)


def test_checkpoint_report_resolves_and_verifies_archived_sidecar(tmp_path: Path) -> None:
    logical = tmp_path / "checkpoint.npz"
    archived = tmp_path / "archive" / "checkpoint.npz"
    archived.parent.mkdir()
    archived.write_bytes(b"retained-checkpoint")
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"tokens": []}', encoding="utf-8")
    logical.with_name(logical.name + ".archive-pointer.json").write_text(
        json.dumps(
            {
                "policy": "project_theseus_archived_artifact_pointer_v1",
                "archive_path": str(archived),
                "original_path": str(logical),
            }
        ),
        encoding="utf-8",
    )
    report = {
        "budget": {
            "checkpoint": str(logical),
            "checkpoint_sha256": hashlib.sha256(archived.read_bytes()).hexdigest(),
            "vocab": str(vocab),
            "vocab_sha256": hashlib.sha256(vocab.read_bytes()).hexdigest(),
        }
    }
    resolved_checkpoint, resolved_vocab = resolve_checkpoint_paths_from_report(report)
    assert resolved_checkpoint == archived
    assert resolved_vocab == vocab


def test_checkpoint_report_rejects_corrupted_archived_payload(tmp_path: Path) -> None:
    logical = tmp_path / "checkpoint.npz"
    archived = tmp_path / "archive" / "checkpoint.npz"
    archived.parent.mkdir()
    archived.write_bytes(b"retained-checkpoint")
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"tokens": []}', encoding="utf-8")
    logical.with_name(logical.name + ".archive-pointer.json").write_text(
        json.dumps(
            {
                "policy": "project_theseus_archived_artifact_pointer_v1",
                "archive_path": str(archived),
            }
        ),
        encoding="utf-8",
    )
    report = {
        "budget": {
            "checkpoint": str(logical),
            "checkpoint_sha256": "0" * 64,
            "vocab": str(vocab),
        }
    }
    with pytest.raises(ValueError, match="checkpoint receipt SHA-256 mismatch"):
        resolve_checkpoint_paths_from_report(report)


def test_checkpoint_report_prefers_live_artifact_over_stale_sidecar(tmp_path: Path) -> None:
    logical = tmp_path / "checkpoint.npz"
    logical.write_bytes(b"current-checkpoint")
    archived = tmp_path / "archive.npz"
    archived.write_bytes(b"stale-checkpoint")
    logical.with_name(logical.name + ".archive-pointer.json").write_text(
        json.dumps(
            {
                "policy": "project_theseus_archived_artifact_pointer_v1",
                "archive_path": str(archived),
            }
        ),
        encoding="utf-8",
    )
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"tokens": []}', encoding="utf-8")
    report = {
        "budget": {
            "checkpoint": str(logical),
            "checkpoint_sha256": hashlib.sha256(logical.read_bytes()).hexdigest(),
            "vocab": str(vocab),
        }
    }
    resolved_checkpoint, _ = resolve_checkpoint_paths_from_report(report)
    assert resolved_checkpoint == logical


def test_checkpoint_lineage_rejects_private_residual_repair_family_claim() -> None:
    lineage = checkpoint_evaluation_lineage(
        {
            "policy": "adaptation",
            "trigger_state": "GREEN",
            "summary": {
                "family_disjoint_evidence": False,
                "family_disjoint_holdout_exclusion": {
                    "enabled": False,
                    "clean": True,
                    "family_disjoint_evidence": False,
                    "policy": "private_residual_repair_row_holdout_v1",
                },
            },
        },
        report_path="reports/checkpoint.json",
    )
    assert lineage["family_disjoint_claim_state"] == "DISALLOWED"
    assert not lineage["family_disjoint_evidence"]
    gate = family_disjoint_checkpoint_lineage_gate({"family_disjoint": {}}, lineage)
    assert not gate["passed"]
    assert gate["severity"] == "hard"


def test_checkpoint_lineage_requires_explicit_family_disjoint_receipt() -> None:
    lineage = checkpoint_evaluation_lineage(
        {"policy": "legacy_checkpoint", "summary": {}},
        report_path="reports/legacy.json",
    )
    assert lineage["family_disjoint_claim_state"] == "UNVERIFIED"
    assert not lineage["family_disjoint_evidence"]
    assert not family_disjoint_checkpoint_lineage_gate({"family_disjoint": {}}, lineage)["passed"]
    assert family_disjoint_checkpoint_lineage_gate({"private_train_replay": {}}, lineage)["passed"]


def test_checkpoint_lineage_accepts_clean_explicit_family_disjoint_receipt() -> None:
    lineage = checkpoint_evaluation_lineage(
        {
            "policy": "adaptation",
            "trigger_state": "GREEN",
            "summary": {
                "family_disjoint_evidence": True,
                "family_disjoint_holdout_exclusion": {
                    "enabled": True,
                    "clean": True,
                    "policy": "exclude_configured_family_disjoint_holdout_families_v1",
                },
                "checkpoint_training_lineage": {
                    "policy": "strict_generator_from_scratch_family_disjoint_lineage_v1",
                    "family_disjoint_evidence": True,
                    "model_initialized_from_scratch": True,
                },
            },
        }
    )
    assert lineage["family_disjoint_claim_state"] == "VERIFIED"
    assert lineage["family_disjoint_evidence"]
    assert family_disjoint_checkpoint_lineage_gate({"family_disjoint": {}}, lineage)["passed"]


def test_checkpoint_lineage_fails_closed_on_contradictory_receipt() -> None:
    lineage = checkpoint_evaluation_lineage(
        {
            "summary": {
                "family_disjoint_evidence": True,
                "family_disjoint_holdout_exclusion": {
                    "enabled": False,
                    "clean": True,
                    "policy": "private_residual_repair_row_holdout_v1",
                },
            }
        }
    )
    assert lineage["family_disjoint_claim_state"] == "DISALLOWED"
    assert not lineage["family_disjoint_evidence"]


def test_transitive_checkpoint_lineage_cannot_reset_bad_parent() -> None:
    local = {
        "enabled": True,
        "clean": True,
        "excluded_row_count": 12,
        "policy": "exclude_configured_family_disjoint_holdout_families_v1",
    }
    clean = compose_transitive_checkpoint_training_lineage(
        {"family_disjoint_evidence": True},
        local,
        private_residual_repair_split=False,
        injected_teacher_row_count=0,
    )
    assert clean["family_disjoint_evidence"]
    bad_parent = compose_transitive_checkpoint_training_lineage(
        {"family_disjoint_evidence": False},
        local,
        private_residual_repair_split=False,
        injected_teacher_row_count=0,
    )
    assert not bad_parent["family_disjoint_evidence"]
    teacher_contaminated = compose_transitive_checkpoint_training_lineage(
        {"family_disjoint_evidence": True},
        local,
        private_residual_repair_split=False,
        injected_teacher_row_count=1,
    )
    assert not teacher_contaminated["family_disjoint_evidence"]


def test_semantic_repair_profiles_cannot_weaken_data_split() -> None:
    for profile in SEMANTIC_CONSTRUCTION_REPAIR_PROFILES.values():
        assert "private_residual_repair_split" not in profile.get("overrides", {})


def test_semantic_profile_checkpoint_compatibility_fails_before_training() -> None:
    profile = {
        **SEMANTIC_CONSTRUCTION_REPAIR_PROFILES["strict_full_body_semantic_construction_v1"],
        "profile_id": "strict_full_body_semantic_construction_v1",
    }
    incompatible = semantic_profile_checkpoint_compatibility(
        profile,
        {
            "target_mode": "body_tokens",
            "target_vocab": {"<pad>": 0, "NAME:return": 1},
            "semantic_slot_head_materialized": False,
        },
    )
    assert not incompatible["compatible"]
    assert incompatible["optimizer_steps_before_rejection"] == 0
    assert set(incompatible["missing_components"]) == {
        "semantic_plan_label_space",
        "semantic_slot_head",
        "semantic_slot_label_space",
    }
    compatible = semantic_profile_checkpoint_compatibility(
        profile,
        {
            "target_mode": "plan_semantic_slots_body_tokens_v1",
            "target_vocab": {
                "<pad>": 0,
                "SLOT:PLAN_AST_RETURN_NUMBER": 1,
                "SLOT:RETURN_SHAPE_NUMBER": 2,
                "SLOT:BODY_START": 3,
            },
            "semantic_slot_head_materialized": True,
        },
    )
    assert compatible["compatible"]


def test_progress_rejects_wrong_task_hash_before_commit(tmp_path: Path) -> None:
    path = tmp_path / "decode.progress.json"
    progress = initialize_decode_progress(path, run_contract=_contract(), resume=False)
    bind_decode_progress_split(
        progress,
        path,
        split_name="family_disjoint",
        task_input_hashes=["task-a"],
    )
    with pytest.raises(ValueError, match="task hash mismatch"):
        commit_decode_progress_batch(
            progress,
            path,
            split_name="family_disjoint",
            records=[
                {
                    "task_index": 0,
                    "task_input_hash": "wrong",
                    "decoded": [],
                    "diagnostic": {},
                }
            ],
            batch_receipt={},
        )
