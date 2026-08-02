from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r3_campaign as campaign  # noqa: E402


def test_repaired_campaign_is_green_pending_and_zero_call() -> None:
    report = campaign.audit_campaign()

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["complete_tasks"] == 0
    assert report["pending_tasks"] == 10
    assert report["model_calls_retained"] == 0
    assert report["physical_context_boundary_hits"] == 0
    assert report["project_selected_quality_token_cap"] is None
    assert report["prompt_continuity"][
        "project_selected_first_artifact_character_cap"
    ] is None


def test_result_and_runtime_namespaces_are_fresh() -> None:
    pool = campaign.p2a.read_json(campaign.POOL)
    row = pool["tasks"][0]
    paths = campaign.result_paths(row)

    assert "p4v2r2r3_attempt1" in paths["run"].name
    assert "p4v2r2r3_attempt1" in paths["evaluation"].name
    assert campaign.RUNTIME_ATTEMPT_NAMESPACE == "p4v2r2r4_attempt1"
