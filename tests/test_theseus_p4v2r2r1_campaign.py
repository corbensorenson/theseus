from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r1_campaign as campaign  # noqa: E402


def test_interrupted_recovery_campaign_retains_partial_receipts_fail_closed() -> None:
    report = campaign.audit_campaign()

    assert report["trigger_state"] == "RED"
    assert report["faults"] == [
        "instrument_audit_red",
        "partial_unsealed_runtime_receipts:p4v2r2_08_pylsp_715",
    ]
    assert report["complete_tasks"] == 6
    assert report["pending_tasks"] == 4
    assert report["model_calls_retained"] == 36
    assert report["physical_context_boundary_hits"] == 0
    assert report["project_selected_quality_token_cap"] is None


def test_recovery_campaign_has_fresh_result_and_runtime_namespace() -> None:
    pool = campaign.p2a.read_json(campaign.POOL)
    paths = [campaign.result_paths(row) for row in pool["tasks"]]

    assert len({row["run"] for row in paths}) == 10
    assert all("p4v2r2r1_attempt1" in row["run"].name for row in paths)
    assert campaign.RUNTIME_ATTEMPT_NAMESPACE == "p4v2r2r1_attempt1"
