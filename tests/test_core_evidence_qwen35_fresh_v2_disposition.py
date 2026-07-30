from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import core_evidence_qwen35_fresh_v2_disposition as disposition  # noqa: E402


def test_valid_failure_is_scoped_to_worker_control() -> None:
    report = disposition.build(
        disposition.EVALUATION,
        disposition.CANDIDATES,
        disposition.FREEZE,
        disposition.EVALUATOR_MANIFEST,
    )
    assert report["trigger_state"] == "GREEN"
    assert (
        report["disposition"]
        == "FAIL_QWEN35_WORKER_CONTROL_COMPETENCE"
    )
    assert report["qualification_passed"] is False
    assert report["observed"]["attempted"] == 3
    assert report["observed"]["useful"] == 0
    assert report["observed"]["unsafe"] == 1
    assert report["observed"]["timed_out"] == 3
    assert report["observed"]["rollback_verified"] == 3
    assert report["observed"]["causal_wall_counts"] == {
        "AUTHORITY_OR_INTEGRITY_VIOLATION": 1,
        "EDIT_SYNTHESIS_NO_PATCH": 2,
    }
    assert (
        report["terminal_effects"]["original_E2_heldout_remains_sealed"]
        is True
    )
    assert "VCM efficacy" in report["causal_diagnosis"]["not_tested"]
    assert report["counters"]["external_inference_calls"] == 0
