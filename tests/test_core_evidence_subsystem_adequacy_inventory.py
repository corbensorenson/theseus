from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "core_evidence_subsystem_adequacy_inventory.py"
SPEC = importlib.util.spec_from_file_location("subsystem_inventory", SCRIPT)
assert SPEC and SPEC.loader
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


def test_checked_in_inventory_is_explicitly_historical_and_superseded() -> None:
    report = inventory.build_report(
        ROOT / "configs" / "core_evidence_subsystem_adequacy.json",
        run_tests=False,
    )
    owners = {row["owner_id"]: row for row in report["owners"]}
    assert report["trigger_state"] == (
        "HISTORICAL_SUPERSEDED_NO_CURRENT_AUTHORITY"
    )
    assert report["worker_identity"]["historical_identity_current"] is False
    assert report["worker_identity"]["state"] == (
        "HISTORICAL_SUPERSEDED_NO_CURRENT_AUTHORITY"
    )
    assert report["development_terminal_disposition"].endswith(
        "core_evidence_repository_stack_development_worker_v4_edit_commitment_disposition.json"
    )
    assert owners["verified_reuse"]["state"] == "INVENTORY_GREEN_TESTS_NOT_RUN"
    assert owners["authority_governance"]["state"] == "INVENTORY_GREEN_TESTS_NOT_RUN"
    assert report["development_dogfood"]["state"] == (
        "HISTORICAL_SUPERSEDED_NO_CURRENT_AUTHORITY"
    )
    assert report["development_dogfood"]["program_phase"] == (
        "HISTORICAL_PRE_P1_L0_INVENTORY_SUPERSEDED"
    )
    assert report["development_dogfood"]["latest_experiment_state"] == (
        "POSITIVE_SCOPED_FUNCTIONAL_RESCUE_TERMINAL_VERIFICATION_UNRESOLVED"
    )
    assert report["current_authority"]["state"] == (
        "NO_CURRENT_EXECUTION_OR_CLAIM_AUTHORITY"
    )
    assert report["current_authority"]["current_instrument"] == (
        "configs/theseus_p4s_cognitive_compilation_instrument.json"
    )
    assert report["development_dogfood"]["ready"] is False
    supersession = next(
        row
        for row in report["development_dogfood"]["findings"]
        if row["check"] == "historical_supersession_explicit"
    )
    assert supersession["passed"] is True
    assert report["E2_heldouts_opened"] == 0


def test_owner_audit_requires_live_eligible_role_and_entrypoint() -> None:
    owner = {
        "owner_id": "example",
        "implementation_ids": ["impl.example"],
        "required_role": "planning",
        "tests": ["tests/test_core_evidence_subsystem_adequacy_inventory.py"],
        "required_interventions": ["control"],
    }
    implementations = {
        "impl.example": {
            "id": "impl.example",
            "status": "retired",
            "role": "governance",
            "canonical_entrypoint": "scripts/core_evidence_subsystem_adequacy_inventory.py",
            "verification_command": "",
            "routing_eligibility": {"eligible": False},
        }
    }
    result = inventory.audit_owner(owner, implementations, run_tests=False)
    assert result["state"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert sum(not row["passed"] for row in result["findings"]) >= 4


def test_config_preserves_measurement_boundaries() -> None:
    config = json.loads(
        (ROOT / "configs" / "core_evidence_subsystem_adequacy.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["boundaries"]["external_inference"] == "forbidden"
    assert config["boundaries"]["teacher_calls"] == "forbidden"
    assert config["boundaries"]["E2_heldout_consumption"] == "forbidden"
    assert config["boundaries"]["D2_consumption"] == "forbidden"
    assert config["adequacy_rules"]["development_causal_smoke_required_after_mechanics"]
    dogfood = config["development_dogfood"]
    assert dogfood["task_policy"][
        "development_rows_eligible_for_fresh_qualification"
    ] is False
    assert dogfood["boundaries"]["external_inference"] == "forbidden"
    assert dogfood["boundaries"]["teacher_calls"] == "forbidden"
    assert dogfood["boundaries"]["E2_heldout_consumption"] == "forbidden"
    assert dogfood["boundaries"]["D2_consumption"] == "forbidden"
    assert dogfood["boundaries"]["automatic_user_facing_effects"] == 0
    assert dogfood["boundaries"]["learned_theseus_student_credit"] == 0
