from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_campaign_controller import (  # noqa: E402
    ARM_IDS,
    SYSTEM_IDS,
    build_campaign_status,
    decide_review,
    validate_review_receipt,
    wilson_interval,
)


def review(candidate: str, *, passed: int, arm_passed: int, position: int = 100_000_000) -> dict:
    digest = "a" * 64
    checkpoint_digest = {
        "moecot_system": "b" * 64,
        "dense_active_parameter": "c" * 64,
        "dense_total_parameter": "d" * 64,
    }[candidate]
    return {
        "policy": "project_theseus_architecture_review_receipt_v1",
        "candidate_id": candidate,
        "review_optimizer_positions": position,
        "evidence": {
            "split": "private_dev",
            "source_disjoint": True,
            "direct_model_only": True,
            "confirmation_surface_consumed": False,
            "public_surface_consumed": False,
            "fallback_return_count": 0,
            "templates_renderers_routers_tools_credit": 0,
            "case_count": 500,
            "passed_count": passed,
            "by_arm": {
                arm: {"case_count": 100, "passed_count": arm_passed} for arm in ARM_IDS
            },
            "plan_sha256": digest,
            "stage_signature": digest,
            "checkpoint_sha256": checkpoint_digest,
            "evaluator_sha256": digest,
            "case_contract_sha256": digest,
            "visible_case_ids_sha256": digest,
            "verifier_budget_sha256": digest,
            "optimizer_positions": position,
            "accepted_verified_outputs_per_second": 0.1,
        },
    }


def test_current_contract_has_twenty_x_first_review_opportunity(tmp_path: Path) -> None:
    report = build_campaign_status(
        scale_config_path=ROOT / "configs/neural_seed_50m_scale_preregistration.json",
        training_config_path=ROOT / "configs/moecot_language_arm_training.json",
        review_dir=tmp_path / "missing",
    )
    assert report["trigger_state"] == "READY"
    assert report["first_review_budget_speedup_opportunity"] == 20.0
    assert report["target_speedup_met_by_contract"] is True
    assert report["target_speedup_empirically_proven"] is False
    assert report["reviews"][0]["decision"] == "WAIT_FOR_MATCHED_RECEIPTS"


def test_one_zero_behavior_pilot_does_not_retire_scale_rung() -> None:
    rows = [review(candidate, passed=0, arm_passed=0) for candidate in SYSTEM_IDS]
    result = decide_review(
        review_position=100_000_000,
        maximum_optimizer_positions=2_000_000_000,
        active_candidates=list(SYSTEM_IDS),
        rows=rows,
        prior_complete_reviews=0,
    )
    assert result["decision"] == "CONTINUE_MATCHED_NO_BEHAVIOR"
    assert result["architecture_falsification_claimed"] is False
    assert result["halted_candidate_ids"] == []


def test_two_zero_behavior_reviews_stop_only_exact_scale_rung() -> None:
    rows = [
        review(candidate, passed=0, arm_passed=0, position=250_000_000)
        for candidate in SYSTEM_IDS
    ]
    result = decide_review(
        review_position=250_000_000,
        maximum_optimizer_positions=2_000_000_000,
        active_candidates=list(SYSTEM_IDS),
        rows=rows,
        prior_complete_reviews=1,
    )
    assert result["decision"] == "STOP_SCALE_RUNG"
    assert result["architecture_falsification_claimed"] is False
    assert result["halted_candidate_ids"] == list(SYSTEM_IDS)


def test_single_review_does_not_prune_without_clear_weak_tail_dominance() -> None:
    rows = [
        review("moecot_system", passed=300, arm_passed=60),
        review("dense_active_parameter", passed=250, arm_passed=50),
        review("dense_total_parameter", passed=245, arm_passed=49),
    ]
    result = decide_review(
        review_position=100_000_000,
        maximum_optimizer_positions=2_000_000_000,
        active_candidates=list(SYSTEM_IDS),
        rows=rows,
        prior_complete_reviews=0,
    )
    assert result["decision"] == "CONTINUE_MATCHED"
    assert result["halted_candidate_ids"] == []


def test_single_review_can_make_scoped_engineering_halt_on_clear_dominance() -> None:
    rows = [
        review("moecot_system", passed=490, arm_passed=98),
        review("dense_active_parameter", passed=5, arm_passed=1),
        review("dense_total_parameter", passed=0, arm_passed=0),
    ]
    result = decide_review(
        review_position=100_000_000,
        maximum_optimizer_positions=2_000_000_000,
        active_candidates=list(SYSTEM_IDS),
        rows=rows,
        prior_complete_reviews=0,
    )
    assert result["decision"] == "HALT_DOMINATED"
    assert result["architecture_falsification_claimed"] is False
    assert result["halted_candidate_ids"] == [
        "dense_active_parameter",
        "dense_total_parameter",
    ]


def test_repeated_review_can_halt_clearly_dominated_candidates() -> None:
    rows = [
        review("moecot_system", passed=400, arm_passed=80, position=250_000_000),
        review("dense_active_parameter", passed=25, arm_passed=5, position=250_000_000),
        review("dense_total_parameter", passed=20, arm_passed=4, position=250_000_000),
    ]
    result = decide_review(
        review_position=250_000_000,
        maximum_optimizer_positions=2_000_000_000,
        active_candidates=list(SYSTEM_IDS),
        rows=rows,
        prior_complete_reviews=1,
    )
    assert result["decision"] == "HALT_DOMINATED"
    assert result["halted_candidate_ids"] == [
        "dense_active_parameter",
        "dense_total_parameter",
    ]


def test_receipt_rejects_confirmation_public_fallback_and_missing_arm() -> None:
    row = review("moecot_system", passed=10, arm_passed=2)
    broken = copy.deepcopy(row)
    broken["evidence"]["confirmation_surface_consumed"] = True
    broken["evidence"]["public_surface_consumed"] = True
    broken["evidence"]["fallback_return_count"] = 1
    del broken["evidence"]["by_arm"]["rust"]
    gaps = validate_review_receipt(broken)
    assert "confirmation_surface_consumed" in gaps
    assert "public_surface_consumed" in gaps
    assert "fallback_return_count_nonzero" in gaps
    assert "arm_coverage_incomplete" in gaps


def test_review_loader_rejects_unmatched_case_identity(tmp_path: Path) -> None:
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    for candidate in SYSTEM_IDS:
        row = review(candidate, passed=10, arm_passed=2)
        if candidate == "dense_total_parameter":
            row["evidence"]["visible_case_ids_sha256"] = "b" * 64
        (review_dir / f"{candidate}.json").write_text(json.dumps(row))
    report = build_campaign_status(
        scale_config_path=ROOT / "configs/neural_seed_50m_scale_preregistration.json",
        training_config_path=ROOT / "configs/moecot_language_arm_training.json",
        review_dir=review_dir,
    )
    assert report["reviews"][0]["decision"] == "REJECT_UNMATCHED_REVIEW"
    assert "unmatched_review_field:visible_case_ids_sha256" in report["reviews"][0]["hard_gaps"]
    assert report["trigger_state"] == "RED"
    assert report["next_action"]["kind"] == "repair_matched_review_contract"
    assert any(
        "unmatched_review_field:visible_case_ids_sha256" in gap
        for gap in report["hard_gaps"]
    )


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(17, 100)
    assert lower < 0.17 < upper
