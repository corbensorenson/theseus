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
    assert active["phase"] == "K2_EVALUATOR_INSTRUMENT_QUALIFICATION"
    assert active["selected_task_index"] == 26
    assert active["last_closed_task_index"] == 30

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count("active claim is") == 1
    assert (
        "The active claim is `cognitive-compilation-and-semantic-ir.core`" not in readme
    )


def test_decision_acceleration_contract_freezes_scope_and_evidence_growth() -> None:
    contract = active_claim()["decision_acceleration_contract"]
    scope = contract["scope_freeze"]
    k2 = contract["K2_completion_boundary"]
    baselines = contract["K3_baseline_contract"]
    evidence = contract["evidence_proportionality"]

    assert scope["source_panel_task_count"] == 62
    assert scope["control_qualification_task_count"] == 9
    assert scope["claim_task_count"] == 53
    assert scope["new_task_or_repository_slots_authorized"] == 0
    assert k2["task26_is_final_bespoke_per_task_dependency_canary"] is True
    assert k2["remaining_dependency_closures_use_one_manifest_driven_generic_owner"]
    assert (
        k2["per_task_script_config_test_or_report_families_after_task26_authorized"]
        is False
    )
    assert (
        "ordinary_retrieval_same_store_and_budget"
        in baselines["required_simple_controls"]
    )
    assert (
        "poisoned_or_tainted_memory"
        in baselines["required_negative_and_mechanism_controls"]
    )
    assert evidence["new_dashboard_or_report_family_authorized"] is False
    assert evidence["evidence_cost_cannot_count_as_mechanism_progress"] is True
