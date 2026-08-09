from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "roadmap_implementation_matrix.json"
HUMAN_STATE_PATHS = (
    ROOT / "README.md",
    ROOT / "roadmap.md",
    ROOT / "docs" / "PROJECT_STATE.md",
)


def active_claim() -> dict:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    return matrix["research_program_recenter"]["active_claim"]


def test_human_state_surfaces_match_machine_active_claim() -> None:
    active = active_claim()
    required_values = (
        active["claim_id"],
        active["phase"],
        active["state"],
        active["active_attempt_id"],
        active["current_blocker"],
        active["next_legal_action"],
    )

    for path in HUMAN_STATE_PATHS:
        text = path.read_text(encoding="utf-8")
        missing = [value for value in required_values if value not in text]
        assert not missing, f"{path.relative_to(ROOT)} missing active values: {missing}"


def test_active_status_is_specific_and_single_claim() -> None:
    active = active_claim()
    assert active["claim_id"] == "virtual-context-abi.core"
    assert active["phase"] == "K3_REAL_WORK_MATCHED_CANARY"
    assert active["selected_task_index"] == 26
    assert active["last_closed_task_index"] == 26

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count("active claim is") == 1
    assert (
        "The active claim is `cognitive-compilation-and-semantic-ir.core`" not in readme
    )


def test_decision_acceleration_contract_freezes_scope_and_evidence_growth() -> None:
    contract = active_claim()["decision_acceleration_contract"]
    book = contract["book_claim_binding"]
    information = contract["information_flow_contract"]
    scope = contract["scope_freeze"]
    k2 = contract["K2_completion_boundary"]
    baselines = contract["K3_baseline_contract"]
    screen = contract["K3_screen_and_execution_contract"]
    decision = contract["decision_rule"]
    evidence = contract["evidence_proportionality"]

    assert book["core_claim_id"] == "virtual-context-abi.core"
    assert book["core_claim_semantics_unchanged_from_pin"] is True
    assert book["support_state"] == "argument"
    assert [layer["id"] for layer in book["claim_layers"]] == [
        "L0_CONFORMANCE",
        "L1_INTEGRITY",
        "L2_MODEL_USE_AND_UTILITY",
        "L3_ECONOMICS",
        "L4_TRANSFER",
    ]
    assert information["candidate_store_source"] == "exact parent snapshot only"
    assert information["target_snapshot_visible_to_generation_or_ranking"] is False
    assert information["target_patch_visible_to_generation_or_ranking"] is False
    assert information["target_derived_allowed_effect_paths_authorized"] is False
    assert information["violation_disposition"].startswith("INVALID_INFORMATION_FLOW")

    assert scope["source_panel_task_count"] == 62
    assert scope["control_qualification_task_count"] == 9
    assert scope["claim_task_count"] == 53
    assert scope["new_task_or_repository_slots_authorized"] == 0
    assert scope["exact_consumed_surface_rerun_authorized"] is False
    assert k2["task26_is_final_bespoke_per_task_dependency_canary"] is True
    assert k2["remaining_dependency_closures_use_one_manifest_driven_generic_owner"]
    assert k2["generic_owner_replayed_closure_count"] == 6
    assert k2["generic_owner_replayed_managers"] == ["cargo", "npm", "pnpm", "uv"]
    assert k2["distinct_parent_target_dependency_closure_count"] == 58
    assert k2["closures_green_before_task26"] == 6
    assert k2["closures_green_after_task26"] == 6
    assert (
        k2["task26_terminal_disposition"]
        == "INCONCLUSIVE_INSTRUMENT_DEPENDENCY_POLICY_RISK_CLASS"
    )
    assert k2["locked_closures_remaining_after_task26"] == 51
    assert k2["shared_content_addressed_manager_stores_required"] is True
    assert k2["per_task_duplicate_package_cache_authorized"] is False
    assert (
        k2["per_task_script_config_test_or_report_families_after_task26_authorized"]
        is False
    )
    assert (
        "ordinary_direct_retrieval_same_parent_store_query_and_context_opportunity"
        in baselines["current_campaign_routes"]
    )
    assert baselines["target_derived_oracle_control_authorized"] is False
    assert (
        "information_matched_flat_direct_context"
        in baselines["claim_panel_required_local_arms"]
    )
    assert screen["screen_uses_local_model_only"] is True
    assert screen["screen_pass_count_threshold_authorized"] is False
    assert screen["candidate_sealed_before_hidden_scoring"] is True
    assert screen["seed_pseudoreplication_authorized"] is False
    assert (
        screen[
            "fixed_sequence_or_other_valid_multiplicity_control_must_be_frozen_before_outcomes"
        ]
        is True
    )
    assert screen["fixed_53_task_panel_may_not_expand_if_secondary_or_narrow_gate_is_underpowered"] is True
    assert "L0_conformance_green" in decision["full_candidate_requires"]
    assert "prospectively_powered_noninferiority_to_information_matched_flat_context" in (
        decision["narrow_governed_transport_candidate_requires"]
    )
    assert "INVALID_INFORMATION_FLOW" in decision["K3_terminal_outcomes"]
    assert evidence["new_dashboard_or_report_family_authorized"] is False
    assert evidence["evidence_cost_cannot_count_as_mechanism_progress"] is True


def test_active_work_packages_form_one_forward_dependency_graph() -> None:
    packages = active_claim()["decision_acceleration_contract"][
        "execution_work_packages"
    ]
    package_ids = [package["id"] for package in packages]

    assert len(package_ids) == len(set(package_ids))
    assert [package["id"] for package in packages if package["status"].startswith("ACTIVE")] == [
        "K3_01R_PARENT_PAGE_AND_MATCHED_HOST_ADEQUACY_REPAIR"
    ]

    seen: set[str] = set()
    for package in packages:
        assert set(package["depends_on"]).issubset(seen)
        seen.add(package["id"])

    assert package_ids.index("K3_03_REFERENCE_REBIND_OR_OMIT") < package_ids.index(
        "K3_04_POWERED_LOCAL_CLAIM_CAMPAIGN"
    )
    assert package_ids[-1] == "K5_01_BOOK_HANDOFF_AND_NEXT_RESIDUAL"


def test_mismatched_reference_claim_cannot_open_calls() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    governed_reference = matrix["research_program_recenter"]["openai_reference_control"]
    assert governed_reference["billable_api_inference_authorized"] is False
    assert "Codex-subscription" in governed_reference["required_access_path"]
    assert governed_reference["unverifiable_access_path_disposition"] == "PROSPECTIVE_OMISSION"

    reference = json.loads(
        (ROOT / "configs" / "theseus_external_reference_control.json").read_text(
            encoding="utf-8"
        )
    )
    activation = reference["activation_scope"]

    if activation["current_claim_id"] != active_claim()["claim_id"]:
        assert activation["current_claim_pool_open"] is False
        assert activation["reference_calls_authorized"] is False
