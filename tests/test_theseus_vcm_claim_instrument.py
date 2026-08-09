from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_claim_instrument as instrument  # noqa: E402


def test_vcm_claim_instrument_audit_is_green_and_call_free() -> None:
    report = instrument.audit(instrument.DEFAULT_CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert report["active_claim_id"] == "virtual-context-abi.core"
    assert report["prospective_design_ready"] is True
    assert report["task_source_acquisition_opened"] is False
    assert report["candidate_generation_opened"] is False
    assert report["hidden_evaluation_opened"] is False
    assert report["faults"] == []
    assert all(value == 0 for value in report["counters"].values())


def test_vcm_packet_controls_cover_required_fail_closed_states() -> None:
    rows = instrument.packet_control_audit()
    assert {row["control"] for row in rows} == {
        "correct",
        "omitted",
        "stale",
        "tainted",
        "revoked",
        "wrong_scope",
        "over_compressed",
    }
    assert all(row["passed"] is True for row in rows)
    assert next(row for row in rows if row["control"] == "correct")["ready"] is True
    assert all(row["ready"] is False for row in rows if row["control"] != "correct")


def test_vcm_route_controls_bind_real_model_visible_content() -> None:
    config = p2a.read_json(instrument.DEFAULT_CONFIG)
    rows = instrument.route_control_audit(config)
    assert {row["control"] for row in rows} == {
        "correct_vcm",
        "omitted_materialization",
        "tainted_materialization",
        "shuffled_packet_changes_binding",
        "information_matched_plain_context",
        "no_added_context",
        "maximal_ungoverned_context",
        "raw_context_excluded_from_receipt",
    }
    assert all(row["passed"] is True for row in rows)


def test_vcm_claim_design_has_no_quality_cap_or_cross_stage_authority() -> None:
    config = p2a.read_json(instrument.DEFAULT_CONFIG)
    assert config["completion_policy"]["project_selected_quality_token_cap"] is None
    assert config["completion_policy"]["normal_completion"] == ["complete_artifact", "model_eos"]
    assert config["prospective_design"]["control_qualification_task_count"] == 9
    assert config["prospective_design"]["task_count"] == 53
    assert config["prospective_design"]["task_count_binding_rule"] == (
        "derive_from_predeclared_useful_effect_power_analysis_before_source_acquisition"
    )
    assert set(config["prospective_design"]["candidate_visible_fields"]) == {
        "natural_language_request",
        "callable_signature_when_present",
        "broad_parent_effect_root",
        "arm_specific_model_visible_context",
    }
    assert "allowed_effect_paths" not in config["prospective_design"]["candidate_visible_fields"]
    assert config["prospective_design"]["effect_boundary"] == {
        "broad_parent_effect_root": "repository",
        "same_root_for_every_arm": True,
        "target_derived_effect_paths_forbidden": True,
        "candidate_patch_scope_recomputed_independently": True,
    }
    assert config["context_resource_policy"]["information_matched_plain_context_budget"].startswith("exact same information")
    power = config["prospective_design"]["power_design"]
    assert power["minimum_useful_absolute_effect"] == 0.35
    power_at_53 = instrument.worst_case_exact_mcnemar_power(53, 0.35, 0.05)
    assert abs(power_at_53 - power["worst_case_power_at_53"]) < 1e-12
    assert power_at_53 >= 0.80
    assert instrument.worst_case_exact_mcnemar_power(52, 0.35, 0.05) < 0.80
    authority = config["authority"]
    assert authority["local_model_calls_authorized"] == 0
    assert authority["external_reference_calls_authorized"] == 0
    assert authority["teacher_calls_authorized"] is False
    assert authority["training_rows_authorized"] is False
    assert authority["D1_authorized"] is False
    assert authority["D2_authorized"] is False
    assert authority["book_support_promotion_authorized"] is False
    assert authority["user_or_operator_gate"] is False
