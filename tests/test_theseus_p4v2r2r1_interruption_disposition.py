from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r1_interruption_disposition as disposition  # noqa: E402


def test_interrupted_campaign_is_scoped_and_not_resumable() -> None:
    report = disposition.build_report(process_active_override=False)

    assert report["trigger_state"] == "GREEN"
    assert report["scientific_status"] == "INCONCLUSIVE_EXPERIMENT"
    assert report["faults"] == []
    assert report["denominators"]["complete_tasks"] == 6
    assert report["denominators"]["partially_consumed_tasks"] == 1
    assert report["denominators"]["candidate_unseen_tasks"] == 3
    assert report["denominators"]["learned_model_calls"] == 37
    assert report["denominators"]["physical_context_boundary_hits"] == 0
    assert report["denominators"]["project_selected_quality_token_cap"] is None
    assert report["interruption"]["same_denominator_resume_authorized"] is False
    assert report["next_stage"]["D1_eligible"] is False
    assert report["interim_observation"]["terminal_mechanism_decision_authorized"] is False


def test_complete_evaluations_replay_and_partial_route_is_green() -> None:
    report = disposition.build_report(process_active_override=False)

    assert len(report["completed_tasks"]) == 6
    assert all(row["blind_evaluator_replay_match"] for row in report["completed_tasks"])
    assert all(row["route_custody_green"] for row in report["completed_tasks"])
    assert report["partial_task"]["route_custody_green"] is True
    assert report["partial_task"]["runtime_receipts"] == 1
