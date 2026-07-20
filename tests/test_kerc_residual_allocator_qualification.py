from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kerc_residual_allocator_qualification import (  # noqa: E402
    ablated_rows,
    allocator_input_shape,
    baseline_metrics,
    load_independent_eval_records,
    load_split_rows,
    select_rows,
    validate_config,
)
from moecot_language_arm_training import (  # noqa: E402
    KERC_UNIT_CANDIDATE_BASE_FEATURE_DIM,
    KERC_UNIT_CANDIDATE_FEATURE_DIM,
)


def allocator_row(identity: str, labels: tuple[int, ...]) -> dict:
    unit_count = len(labels)
    features = np.zeros((unit_count, 4, 18), dtype=np.float32)
    for unit_index in range(unit_count):
        features[unit_index, :, 0] = [0.1, 0.2, 0.3, 0.4]
    hard = np.zeros((unit_count, 4), dtype=bool)
    for index, label in enumerate(labels):
        hard[index, :label] = True
    return {
        "source_record_sha256": "sha256:" + identity * 64,
        "source_group": f"private:{identity}",
        "split": "private_train",
        "unit_ids": tuple(f"ru:{identity}:{index}" for index in range(unit_count)),
        "byte_rows": tuple(np.asarray([index + 1], dtype=np.int32) for index in range(unit_count)),
        "kind_ids": np.arange(unit_count, dtype=np.int32) % 5,
        "candidate_features": features,
        "hard_block_mask": hard,
        "labels": np.asarray(labels, dtype=np.int32),
        "confidence_targets": np.ones(unit_count, dtype=np.float32),
        "loss_mask": np.ones(unit_count, dtype=np.float32),
        "k2_labels": np.full(unit_count, 3, dtype=np.int32),
        "target_identity": "sha256:" + identity * 64,
    }


def test_selection_preserves_available_class_coverage_without_inventing_class_zero() -> None:
    rows = [
        allocator_row("1", (1, 2)),
        allocator_row("2", (2, 3)),
        allocator_row("3", (1, 3)),
    ]
    selected, receipt = select_rows(
        rows, maximum=3, minimum_per_class=1, seed=17
    )
    assert len(selected) == 3
    assert set(receipt["selected_class_counts"]) == {"1", "2", "3"}
    assert receipt["answer_text_visible"] is False
    assert receipt["model_outcomes_visible"] is False


def test_strong_baselines_obey_hard_constraints_and_are_reported_separately() -> None:
    train = [allocator_row("4", (1, 2, 3))]
    evaluation = [allocator_row("5", (1, 2, 3))]
    result = baseline_metrics(train, evaluation)
    assert result["presence_by_kind"]["hard_violation_count"] == 0
    assert result["source_visible_constrained_rate"]["accuracy"] == 1.0
    assert result["k2_structural_selection"]["accuracy"] < 1.0


def test_ragged_input_receipt_charges_actual_bytes_without_truncation() -> None:
    rows = [allocator_row("a", (1, 2)), allocator_row("b", (3,))]
    rows[0]["byte_rows"] = (
        np.asarray([1, 2, 3], dtype=np.int32),
        np.asarray([4], dtype=np.int32),
    )
    rows[1]["byte_rows"] = (np.asarray([5, 6], dtype=np.int32),)
    receipt = allocator_input_shape(rows, batch_records=2)
    assert receipt["actual_source_byte_count"] == 6
    assert receipt["naive_rectangular_byte_slots_at_configured_batch"] == 12
    assert receipt["source_byte_truncation_count"] == 0


def test_config_rejects_evaluator_feature_and_public_data_authority() -> None:
    config = json.loads(
        (ROOT / "configs" / "kerc_residual_allocator_qualification.json").read_text()
    )
    assert validate_config(copy.deepcopy(config)) == config
    for key, value in (
        ("public_training_rows_written", 1),
        ("evaluator_effect_features_visible_to_model", True),
    ):
        mutated = copy.deepcopy(config)
        mutated["boundaries"][key] = value
        with pytest.raises(ValueError):
            validate_config(mutated)


def test_loader_skips_fully_withheld_rows_but_rejects_suppressed_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private_eval.jsonl"
    withheld = {
        "objective": "kernel_program_to_answer_packet_v1",
        "split": "private_eval",
        "kerc_residual_unit_allocator_loss_enabled": False,
        "kerc_residual_unit_targets": [{"allocator_loss_enabled": False}],
    }
    path.write_text(json.dumps(withheld) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no authoritative rows"):
        load_split_rows(path, "private_eval")
    withheld["kerc_residual_unit_targets"][0]["allocator_loss_enabled"] = True
    path.write_text(json.dumps(withheld) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="authority flag suppresses"):
        load_split_rows(path, "private_eval")


def test_independent_evaluator_lookup_is_bound_to_record_not_unit_tuple(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidates.jsonl"
    records = [
        {
            "record_sha256": f"sha256:{token * 64}",
            "kernel_packet": {
                "residual": {
                    "unit_packet": {"units": [{"unit_id": "ru:shared"}]}
                }
            },
            "variant": token,
        }
        for token in ("a", "b")
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    selected = [
        {
            "source_record_sha256": "sha256:" + "b" * 64,
            "unit_ids": ("ru:shared",),
        }
    ]
    result = load_independent_eval_records(path, selected)
    assert set(result) == {"sha256:" + "b" * 64}
    assert result["sha256:" + "b" * 64]["variant"] == "b"


def test_source_relation_shuffle_is_matched_and_never_zero_fills() -> None:
    rows = [allocator_row(token, (2,)) for token in ("c", "d", "e")]
    for index, row in enumerate(rows, 1):
        expanded = np.zeros((1, 4, KERC_UNIT_CANDIDATE_FEATURE_DIM), dtype=np.float32)
        expanded[:, :, :KERC_UNIT_CANDIDATE_BASE_FEATURE_DIM] = row[
            "candidate_features"
        ]
        expanded[:, :, KERC_UNIT_CANDIDATE_BASE_FEATURE_DIM:] = float(index)
        row["candidate_features"] = expanded
        row["kind_ids"][:] = 2
    shuffled = ablated_rows(rows, mode="source_relation_shuffled", seed=31)
    observed = [
        float(row["candidate_features"][0, 0, KERC_UNIT_CANDIDATE_BASE_FEATURE_DIM])
        for row in shuffled
    ]
    assert sorted(observed) == [1.0, 2.0, 3.0]
    assert all(left != right for left, right in zip(observed, [1.0, 2.0, 3.0]))
