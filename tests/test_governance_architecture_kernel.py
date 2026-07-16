from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import governance_rights_receipt_suite as governance  # noqa: E402
import policy_update_lease  # noqa: E402


def fixture_inputs() -> tuple[dict, dict, dict]:
    config = json.loads((ROOT / "configs" / "governance_rights_receipt_suite.json").read_text(encoding="utf-8"))
    assistant = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "effect_canary": {
            "proposer_id": "proposer",
            "observer_id": "observer",
            "evaluator_id": "evaluator",
            "effect_inventory": [{"effect_id": "effect"}],
            "rollback": {"complete": True, "residual_count": 0},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
    }
    evidence_store = {
        "assurance_evaluation_integrity": {
            "state": "GREEN",
            "assurance_case_records": [
                {"record_id": "case.live", "validation": {"valid": True}}
            ],
        }
    }
    return config["architecture_governance"], assistant, evidence_store


def test_architecture_governance_kernel_passes_and_rejects_all_negative_controls() -> None:
    policy, assistant, evidence_store = fixture_inputs()

    result = governance.build_architecture_governance_kernel(
        policy,
        assistant_report=assistant,
        evidence_store=evidence_store,
    )

    assert result["state"] == "GREEN"
    assert result["support_state"] == "synthetic-test-backed"
    assert len(result["expected_invalid_controls"]) == 16
    assert all(row["rejected"] for row in result["expected_invalid_controls"])


def test_oversight_commitment_and_exchange_validators_fail_closed() -> None:
    policy, assistant, evidence_store = fixture_inputs()
    result = governance.build_architecture_governance_kernel(
        policy,
        assistant_report=assistant,
        evidence_store=evidence_store,
    )
    oversight = json.loads(json.dumps(result["oversight_protocol_records"][0]))
    commitment = json.loads(json.dumps(result["capability_commitment_records"][0]))
    exchange = json.loads(json.dumps(result["inter_stack_exchange_records"][0]))

    oversight["roles"]["promotion_authority"] = oversight["roles"]["proposer"]
    commitment["observed_safeguards"][commitment["required_safeguards"][0]] = False
    exchange["nonce_seen_before"] = True

    assert not governance.validate_oversight_protocol(oversight)["valid"]
    assert not governance.validate_capability_commitment(commitment)["valid"]
    assert not governance.validate_inter_stack_exchange(exchange)["valid"]


def test_governance_owner_consumes_multi_target_policy_update_leases() -> None:
    report = policy_update_lease.run_reference_matrix()
    assert report["trigger_state"] == "GREEN"
    assert report["summary"]["target_count"] == 7
    assert report["summary"]["committed_target_count"] == 7
    assert report["summary"]["rollback_canary_exact"] is True
    assert report["summary"]["mutation_case_count"] == report["summary"]["mutation_passed_count"]
