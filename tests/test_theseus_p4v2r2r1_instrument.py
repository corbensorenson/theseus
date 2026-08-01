from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2r1_instrument as instrument  # noqa: E402


def test_recovery_instrument_is_exactly_bound_and_uncapped() -> None:
    value = p2a.read_json(instrument.OUT)
    report = instrument.audit(value)

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["runtime_attempt_namespace"] == "p4v2r2r1_attempt1"
    assert report["project_selected_quality_token_cap"] is None
    assert report["task_candidate_or_control_calls"] == 0


def test_recovery_instrument_requires_real_route_canary_and_per_call_receipts() -> None:
    value = p2a.read_json(instrument.OUT)

    assert value["route_canary_contract"]["real_frozen_model_call_required"] is True
    assert value["route_canary_contract"]["task_denominator_consumed"] == 0
    assert value["matched_arm_contract"]["immutable_backend_receipt_per_model_call"] is True
    assert value["matched_arm_contract"]["stop_immediately_on_any_route_receipt_failure"] is True
