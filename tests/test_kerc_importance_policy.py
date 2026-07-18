from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kerc_importance_policy import (  # noqa: E402
    ImportancePolicyFault,
    evaluator_only_targets,
    fit_importance_policy,
    predict_importance,
    source_visible_features,
)


def record(index: int, split: str, *, protected: bool, complex_claim: bool) -> dict:
    source = (
        f'Person {index} said "Preserve identifier ID-{index}."'
        if protected
        else f"Summarize ordinary statement number {index} clearly."
    )
    objects = (
        {
            "@Q1": {
                "object_type": "QUOTE",
                "copy_policy": "EXACT",
            }
        }
        if protected
        else {}
    )
    claims = [
        {
            "claim_id": "claim-1",
            "polarity": "NEGATIVE" if complex_claim else "AFFIRMED",
            "modality": "POSSIBLE" if complex_claim else "ASSERTED",
            "attribution": {"speaker": "@Q1"} if protected else {},
        }
    ]
    if complex_claim:
        claims.append(
            {
                "claim_id": "claim-2",
                "polarity": "AFFIRMED",
                "modality": "ASSERTED",
                "attribution": {},
            }
        )
    return {
        "split": split,
        "source_text": source,
        "provenance": {"source_group": f"{split}-group-{index}"},
        "kernel_packet": {
            "program": {
                "nodes": [
                    {
                        "polarity": claims[0]["polarity"],
                        "modality": claims[0]["modality"],
                    }
                ]
            },
            "protected_objects": objects,
            "correction_lattice": {"corrections": []},
            "residual": {
                "segment_frame": {},
                "token_tags": (
                    [
                        {
                            "tag": "ENTITY:PERSON",
                            "source_span": [0, 6],
                        }
                    ]
                    if protected
                    else []
                ),
            },
        },
        "answer_packet": {
            "claims": claims,
            "decision": {"disposition": "ANSWER"},
            "required_terms": (["ID"] if protected else []),
            "required_caveats": (["Preserve exact identity"] if complex_claim else []),
        },
    }


def fixture_rows() -> list[dict]:
    rows = []
    for split in ("private_train", "private_dev", "private_eval"):
        for index in range(12):
            rows.append(
                record(
                    index,
                    split,
                    protected=index % 2 == 0,
                    complex_claim=index % 3 == 0,
                )
            )
    return rows


def test_importance_features_cannot_see_evaluator_only_answer_targets() -> None:
    row = record(1, "private_eval", protected=False, complex_claim=False)
    before = source_visible_features(row)
    changed = copy.deepcopy(row)
    changed["answer_packet"]["claims"].append(
        {
            "claim_id": "hidden-evaluator-change",
            "polarity": "NEGATIVE",
            "modality": "REQUIRED",
            "attribution": {},
        }
    )
    assert (source_visible_features(changed) == before).all()
    assert not (evaluator_only_targets(changed) == evaluator_only_targets(row)).all()


def test_importance_policy_fits_calibrates_and_binds_predictions() -> None:
    rows = fixture_rows()
    policy = fit_importance_policy(rows)
    assert policy["source_group_disjoint"] is True
    assert policy["fit_split"] == "private_train"
    assert policy["calibration_split"] == "private_dev"
    assert policy["final_evaluation_split"] == "private_eval"
    assert set(policy["metrics_by_split"]) == {
        "private_train",
        "private_dev",
        "private_eval",
    }
    receipt = predict_importance(rows[-1], policy)
    assert receipt["target_fields_visible_to_policy"] == []
    assert 0.0 <= receipt["allocation_importance"] <= 1.0
    tampered = copy.deepcopy(policy)
    tampered["weights_by_dimension"]["semantic_importance"][0] += 1.0
    with pytest.raises(ImportancePolicyFault, match="identity mismatch"):
        predict_importance(rows[-1], tampered)
