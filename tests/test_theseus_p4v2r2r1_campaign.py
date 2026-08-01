from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r1_campaign as campaign  # noqa: E402


def test_recovery_campaign_preflight_is_green_and_unconsumed() -> None:
    report = campaign.audit_campaign()

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["complete_tasks"] == 0
    assert report["pending_tasks"] == 10
    assert report["model_calls_retained"] == 0
    assert report["physical_context_boundary_hits"] == 0
    assert report["project_selected_quality_token_cap"] is None


def test_recovery_campaign_has_fresh_result_and_runtime_namespace() -> None:
    pool = campaign.p2a.read_json(campaign.POOL)
    paths = [campaign.result_paths(row) for row in pool["tasks"]]

    assert len({row["run"] for row in paths}) == 10
    assert all("p4v2r2r1_attempt1" in row["run"].name for row in paths)
    assert campaign.RUNTIME_ATTEMPT_NAMESPACE == "p4v2r2r1_attempt1"
