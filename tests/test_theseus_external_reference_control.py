from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_luna_reference_is_a_separate_measurement_only_factorial() -> None:
    policy = json.loads(
        (ROOT / "configs" / "theseus_external_reference_control.json").read_text(encoding="utf-8")
    )

    assert policy["state"] == "PROSPECTIVELY_DEFINED_TRANSPORT_NOT_BOUND"
    assert policy["reference_model"]["requested_model"] == "gpt-5.6-luna"
    assert policy["reference_model"]["reasoning_effort"] == "xhigh"
    assert policy["factorial"]["models"] == [
        "best_locally_qualified_model",
        "gpt-5.6-luna_xhigh_reference",
    ]
    assert policy["factorial"]["system_conditions"] == ["direct", "theseus_integrated"]
    assert policy["factorial"]["cross_model_rank_is_not_a_subsystem_causal_claim"] is True


def test_reference_cannot_serve_train_select_tasks_or_write_external_source() -> None:
    policy = json.loads(
        (ROOT / "configs" / "theseus_external_reference_control.json").read_text(encoding="utf-8")
    )
    gate = policy["admission_gate"]

    assert gate["raw_private_data_allowed"] is False
    assert gate["external_source_effects_allowed"] is False
    assert gate["training_row_admission_allowed"] is False
    assert gate["runtime_serving_allowed"] is False
    assert gate["automatic_book_support_promotion_allowed"] is False
    assert policy["matched_requirements"]["task_selection_completed_before_any_arm_opens"] is True
    assert policy["counters"]["accepted_training_rows"] == 0
    assert policy["counters"]["user_facing_serving_tokens"] == 0


def test_charter_and_roadmap_bind_the_measurement_only_exception() -> None:
    charter = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "roadmap.md").read_text(encoding="utf-8")

    assert "measurement-only reference control" in charter
    assert "mixed into local-model denominators" in charter
    assert "P3 hosted reference control" in roadmap
    assert "gpt-5.6-luna" in roadmap
