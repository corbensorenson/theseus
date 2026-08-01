from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_d1_fresh_qualification_instrument as instrument  # noqa: E402


CONFIG = ROOT / "configs" / "theseus_d1_fresh_qualification_instrument.json"


def test_instrument_is_green_but_does_not_open_D1_before_survivor() -> None:
    report = instrument.build_report(CONFIG, disposition_override={})
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["activation_state"] == (
        "WAITING_FOR_GREEN_DECISION_RELEVANT_P4V2R2_SURVIVOR"
    )
    assert report["execution_authorized"] is False
    assert report["source_acquisition_authorized"] is False
    assert report["candidate_or_control_calls_authorized"] is False


def test_power_design_is_recomputed_not_a_convenience_task_cap() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    design = instrument.recompute_power_design(config["power_design"])
    assert design["required_discordant_pairs"] == 18
    assert design["required_treatment_wins_at_required_pairs"] == 13
    assert design["achieved_type_I_error"] <= 0.05
    assert design["achieved_power_at_minimum_worthwhile_effect"] >= 0.8
    assert design["design_derived_cohort_size"] == 44
    assert design["probability_of_reaching_required_pairs_at_discordance_floor"] >= 0.9
    assert design["matches_declared_design"] is True
    assert design["output_token_budget_involved"] is False


def test_generation_has_no_arbitrary_output_cap() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    completion = config["generation_completion"]
    assert completion["project_selected_quality_token_cap"] is None
    assert completion["normal_completion"] == ["parser_complete", "model_eos"]
    assert completion["sole_numeric_boundary"] == (
        "pinned_model_declared_context_window_minus_exact_prompt_tokens"
    )
    assert completion[
        "boundary_or_host_stop_is_model_mechanism_candidate_or_evaluator_failure"
    ] is False


def test_D1_language_scope_matches_the_exact_P4_implementation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["source_surface"]["programming_language_scope"] == ["Python"]
    assert "cross-language portability" in config["source_surface"][
        "language_scope_reason"
    ]


def test_green_survivor_opens_only_source_acquisition() -> None:
    disposition = {
        "trigger_state": "GREEN",
        "scientific_status": "P4V2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "consumption": {"eligible_for_D1": True},
        "decision_rule": {
            "survivor_effect_rule_passed": True,
            "effect_decision_authorized": True,
        },
    }
    report = instrument.build_report(CONFIG, disposition_override=disposition)
    assert report["trigger_state"] == "GREEN"
    assert report["activation_state"] == (
        "READY_FOR_AUTONOMOUS_D1_SOURCE_MEMBERSHIP_FREEZE"
    )
    assert report["source_acquisition_authorized"] is True
    assert report["candidate_or_control_calls_authorized"] is False
    assert report["execution_authorized"] is False


def test_historical_p4s_survivor_cannot_open_the_successor_d1_lane() -> None:
    disposition = {
        "trigger_state": "GREEN",
        "scientific_status": "P4S_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "consumption": {"eligible_for_D1": True},
        "decision_rule": {
            "survivor_effect_rule_passed": True,
            "effect_decision_authorized": True,
        },
    }
    report = instrument.build_report(CONFIG, disposition_override=disposition)
    assert report["source_acquisition_authorized"] is False
    assert report["survivor_checks"]["decision_relevant_survivor"] is False


def test_no_user_gate_rerun_or_cross_stage_authority() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["activation"]["user_or_operator_approval_required"] is False
    assert config["one_shot_authority"]["user_or_operator_gate"] is False
    assert config["one_shot_authority"]["rerun_consumed_identity_allowed"] is False
    assert config["decision_rule"]["automatic_book_support_promotion"] is False
    assert config["decision_rule"]["serving_training_D2_or_teacher_authority"] is False


def test_prior_repository_inventory_is_bound_for_disjoint_selection() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    repositories, faults = instrument.prior_repository_inventory(config)
    assert faults == []
    assert len(repositories) >= 47
    assert "jd/tenacity" in repositories
    assert "urllib3/urllib3" in repositories
    assert len(repositories) == len(set(repositories))
