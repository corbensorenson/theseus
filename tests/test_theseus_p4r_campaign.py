from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4r_campaign as campaign  # noqa: E402


def test_p4r_campaign_is_bound_and_completion_based() -> None:
    report = campaign.audit_campaign()

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["complete_tasks"] + report["pending_tasks"] == 10
    assert report["model_calls_retained"] == report["complete_tasks"] * 6
    assert report["parser_complete_calls"] + report["model_eos_calls"] == report["model_calls_retained"]
    assert report["safety_ceiling_hits"] == 0
    assert report["project_selected_quality_token_cap"] is None
    assert report["pool_sha256"] == campaign.POOL_SHA256
    assert report["pool_seal_commit"] == campaign.POOL_SEAL_COMMIT
    assert report["hosted_reference"]["calls"] == 0


def test_p4r_result_paths_are_campaign_specific() -> None:
    pool = campaign.p2a.read_json(campaign.POOL)
    paths = [campaign.result_paths(row) for row in pool["tasks"]]

    assert len({row["run"] for row in paths}) == 10
    assert len({row["evaluation"] for row in paths}) == 10
    assert all(row["run"].name.startswith("theseus_p4r_") for row in paths)
