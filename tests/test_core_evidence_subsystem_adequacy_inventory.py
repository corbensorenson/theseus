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


def test_checked_in_inventory_binds_registered_reuse_owner() -> None:
    report = inventory.build_report(
        ROOT / "configs" / "core_evidence_subsystem_adequacy.json",
        run_tests=False,
    )
    owners = {row["owner_id"]: row for row in report["owners"]}
    assert report["trigger_state"] == "RED_WORKER_REQUALIFICATION_REQUIRED"
    assert report["worker_identity"]["historical_identity_current"] is False
    assert report["worker_identity"]["state"] == (
        "DEVELOPMENT_SUCCESSOR_REQUALIFICATION_REQUIRED"
    )
    assert owners["verified_reuse"]["state"] == "INVENTORY_GREEN_TESTS_NOT_RUN"
    assert owners["authority_governance"]["state"] == "INVENTORY_GREEN_TESTS_NOT_RUN"
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
