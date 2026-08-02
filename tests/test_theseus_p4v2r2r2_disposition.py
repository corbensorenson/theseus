from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2r2_disposition as disposition  # noqa: E402


def test_source_pool_is_still_adequate_after_pre_generation_repair() -> None:
    pool = p2a.read_json(disposition.POOL)
    audit = disposition.audit_source_pool(pool)

    assert audit["passed"] is True
    assert audit["faults"] == []
    assert audit["task_count"] == 10
    assert audit["distinct_repository_count"] == 10
    assert audit["source_registry_audit"]["instrument"]["binding_mode"] == (
        "prospective_pre_generation_repair"
    )


def test_unconsumed_campaign_is_review_required_not_information_flow_failure() -> None:
    report = disposition.build_report()

    assert report["trigger_state"] == "RED"
    assert report["scientific_status"] == "P4V2R2R2_REVIEW_REQUIRED"
    assert "campaign_not_complete" in report["faults"]
    assert report["denominators"]["tasks"] == 0
    assert report["denominators"]["learned_model_calls"] == 0
    assert report["denominators"]["project_selected_quality_token_cap"] is None
    assert report["next_stage"]["D1_eligible"] is False


def test_terminal_classification_preserves_adequacy_and_D1_boundaries() -> None:
    assert disposition.classify_status(
        information_flow_green=True,
        boundary_hits=0,
        mechanics_floor=True,
        experiment_floor=True,
        survivor_rule=True,
    ) == "P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
    assert disposition.classify_status(
        information_flow_green=True,
        boundary_hits=0,
        mechanics_floor=False,
        experiment_floor=True,
        survivor_rule=False,
    ) == "INCONCLUSIVE_IMPLEMENTATION"
    assert disposition.next_stage("P4V2R2R2_ADEQUATE_NO_SURVIVOR")[
        "D1_eligible"
    ] is False
