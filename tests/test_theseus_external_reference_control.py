from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_luna_reference_is_a_separate_measurement_only_factorial() -> None:
    policy = json.loads(
        (ROOT / "configs" / "theseus_external_reference_control.json").read_text(encoding="utf-8")
    )

    assert policy["state"] == "TRANSPORT_SOURCE_BOUND_OFFLINE_QUALIFIED_ZERO_CALLS"
    assert policy["reference_model"]["requested_model"] == "gpt-5.6-luna"
    assert policy["reference_model"]["reasoning_effort"] == "xhigh"
    assert policy["factorial"]["models"] == [
        "frozen_TMax_9B_MLX_8bit_33812d6cf04f88856f25eb828de4f3144a194560",
        "gpt-5.6-luna_xhigh_reference",
    ]
    assert policy["factorial"]["system_conditions"] == ["direct", "theseus_integrated"]
    assert policy["factorial"]["cross_model_rank_is_not_a_subsystem_causal_claim"] is True
    assert policy["transport"]["api"] == "responses"
    assert policy["transport"]["endpoint"] == "https://api.openai.com/v1/responses"
    assert policy["transport"]["max_output_tokens"] == "OMITTED_NO_PROJECT_QUALITY_CAP"
    assert policy["transport"]["store"] is False
    assert policy["transport"]["tools"] == []


def test_reference_cannot_serve_train_select_tasks_or_write_external_source() -> None:
    policy = json.loads(
        (ROOT / "configs" / "theseus_external_reference_control.json").read_text(encoding="utf-8")
    )
    gate = policy["admission_gate"]

    assert gate["raw_private_data_allowed"] is False
    assert gate["external_source_effects_allowed"] is False
    assert gate["training_row_admission_allowed"] is False
    assert gate["runtime_serving_allowed"] is False
    assert gate["task_selection_or_tuning_allowed"] is False
    assert gate["subsystem_or_prompt_tuning_allowed"] is False
    assert gate["local_model_selection_allowed"] is False
    assert gate["claim_queue_selection_allowed"] is False
    assert gate["local_denominator_membership_allowed"] is False
    assert gate["automatic_book_support_promotion_allowed"] is False
    assert policy["matched_requirements"]["task_selection_completed_before_any_arm_opens"] is True
    assert policy["matched_requirements"]["all_four_cells_sealed_before_any_candidate_call"] is True
    assert policy["activation_scope"]["missing_transport_disposition"].startswith(
        "Omit the reference cells"
    )
    assert policy["counters"]["reference_calls_made"] == 0
    assert policy["counters"]["accepted_training_rows"] == 0
    assert policy["counters"]["user_facing_serving_tokens"] == 0
    assert policy["activation_scope"]["reference_calls_authorized"] is False


def test_reference_completion_has_no_project_selected_quality_cap() -> None:
    policy = json.loads(
        (ROOT / "configs" / "theseus_external_reference_control.json").read_text(
            encoding="utf-8"
        )
    )
    completion = policy["completion_policy"]

    assert completion["normal_completion"].endswith("model_EOS")
    assert completion["project_selected_generated_token_quality_cap"] is None
    assert completion["boundary_hit_counts_as_model_or_mechanism_failure"] is False
    assert completion["boundary_hit_enters_treatment_effect_denominator"] is False


def test_luna_price_basis_and_physical_limits_are_explicit() -> None:
    policy = json.loads(
        (ROOT / "configs" / "theseus_external_reference_control.json").read_text(
            encoding="utf-8"
        )
    )
    price = policy["price_basis"]

    assert price["input_usd_per_million"] == 0.2
    assert price["cached_input_usd_per_million"] == 0.02
    assert price["output_usd_per_million"] == 1.2
    assert price["long_context_threshold_tokens"] == 272000
    assert price["physical_limits"]["context_window_tokens"] == 1050000
    assert price["physical_limits"]["maximum_output_tokens"] == 128000
    assert price["worst_case_uncapped_physical_call_usd"] == 0.6504


def test_charter_and_roadmap_bind_the_measurement_only_exception() -> None:
    charter = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "roadmap.md").read_text(encoding="utf-8")

    assert "measurement-only reference control" in charter
    assert "mixed into local-model denominators" in charter
    assert "P3 hosted reference control" in roadmap
    assert "gpt-5.6-luna" in roadmap


def test_charter_forbids_project_selected_quality_token_caps() -> None:
    charter = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    completion = json.loads(
        (ROOT / "configs" / "theseus_generation_completion_policy.json").read_text(
            encoding="utf-8"
        )
    )

    assert "must not use a project-selected generated-token" in charter
    assert "physical addressability boundary" in charter
    assert "actual tokens, time, verifier use" in charter
    assert completion["physical_safety_boundary"]["project_selected_token_cap"] is None
    assert completion["ceiling_hit_disposition"]["score_as_model_failure"] is False
    assert completion["ceiling_hit_disposition"]["include_in_treatment_effect_denominator"] is False
