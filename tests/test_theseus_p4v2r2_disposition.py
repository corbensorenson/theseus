from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4v2r2_disposition as disposition  # noqa: E402


def test_p4v2r2_pre_campaign_disposition_fails_closed_without_consumption() -> None:
    report = disposition.build_report()

    assert report["trigger_state"] == "RED"
    assert "campaign_not_complete" in report["faults"]
    assert report["scientific_status"] == "P4V2R2_REVIEW_REQUIRED"
    assert report["denominators"]["tasks"] == 0
    assert report["denominators"]["learned_model_calls"] == 0
    assert report["denominators"]["project_selected_quality_token_cap"] is None
    assert report["consumption"]["eligible_for_D1"] is False
    assert report["next_stage"]["D1_eligible"] is False


def test_incomplete_denominator_is_not_mislabeled_as_observed_leakage() -> None:
    report = disposition.build_report()

    assert report["adequacy"]["information_flow_green"] is False
    assert not any(
        fault.startswith(
            (
                "information_flow_invalid:",
                "candidate_integrity_recomputation_invalid:",
                "independent_evaluation_replay_fault:",
                "independent_evaluation_replay_mismatch:",
            )
        )
        for fault in report["faults"]
    )
    assert report["scientific_status"] == "P4V2R2_REVIEW_REQUIRED"


def test_p4v2r2_source_pool_audit_recomputes_license_and_disjointness() -> None:
    pool = disposition.p2a.read_json(disposition.POOL)
    audit = disposition.audit_source_pool(pool)

    assert audit["passed"] is True
    assert audit["faults"] == []
    assert audit["task_count"] == audit["distinct_repository_count"] == 10
    assert set(audit["license_spdx_ids"]) == {
        "BSD-3-Clause",
        "MIT",
        "MIT-CMU",
        "MPL-2.0",
    }
    assert audit["predecessor_repository_overlap"] == []


def test_p4v2r2_terminal_classification_is_exact_and_predeclared() -> None:
    classify = disposition.classify_status
    common = {
        "information_flow_green": True,
        "boundary_hits": 0,
        "mechanics_floor": True,
        "experiment_floor": True,
    }
    assert classify(**common, survivor_rule=True) == (
        "P4V2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
    )
    assert classify(**common, survivor_rule=False) == "P4V2R2_ADEQUATE_NO_SURVIVOR"
    assert classify(
        **{**common, "mechanics_floor": False}, survivor_rule=False
    ) == "INCONCLUSIVE_IMPLEMENTATION"
    assert classify(
        **{**common, "experiment_floor": False}, survivor_rule=False
    ) == "INCONCLUSIVE_EXPERIMENT"
    assert classify(
        **{**common, "information_flow_green": False}, survivor_rule=False
    ) == "INVALID_INFORMATION_FLOW"
    assert classify(
        **{**common, "boundary_hits": 1}, survivor_rule=False
    ) == "INSTRUMENT_INADEQUATE_GENERATION_BOUNDARY_HIT"


def test_p4v2r2_disposition_never_promotes_book_or_training_automatically() -> None:
    for status in (
        "P4V2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE",
        "P4V2R2_ADEQUATE_NO_SURVIVOR",
        "INCONCLUSIVE_IMPLEMENTATION",
        "INCONCLUSIVE_EXPERIMENT",
    ):
        next_stage = disposition.next_stage(status)
        assert next_stage["book_support_state_effect"] == "none"
        assert next_stage["D1_eligible"] is (
            status == "P4V2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
        )
