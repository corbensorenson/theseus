from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_campaign as campaign  # noqa: E402


def test_interrupted_v1_campaign_retains_partial_receipts_fail_closed() -> None:
    report = campaign.audit_campaign()
    assert report["trigger_state"] == "RED"
    assert report["faults"] == [
        "partial_unsealed_runtime_receipts:p4_03_fastapi_15515"
    ]
    assert report["complete_tasks"] == 2
    assert report["pending_tasks"] == 8
    assert report["complete_tasks"] + report["pending_tasks"] == 10
    assert report["model_calls_retained"] == report["complete_tasks"] * 6
    assert report["pool_seal_commit"] == "1aa756e2a83ade8a144dfa1ef309ca2934b50720"
    assert report["hosted_reference"]["calls"] == 0
