from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_campaign as campaign  # noqa: E402


def test_campaign_is_bound_and_has_no_partial_unsealed_receipts() -> None:
    report = campaign.audit_campaign()
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["complete_tasks"] + report["pending_tasks"] == 10
    assert report["model_calls_retained"] == report["complete_tasks"] * 6
    assert report["pool_seal_commit"] == "1aa756e2a83ade8a144dfa1ef309ca2934b50720"
    assert report["hosted_reference"]["calls"] == 0
