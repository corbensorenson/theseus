from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_production_conformance as conformance  # noqa: E402


def test_production_conformance_is_green_and_non_claim() -> None:
    report = conformance.run_conformance()

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["coverage"]["consumed_evaluator_only_mechanics_fixtures"] == 10
    assert report["coverage"]["parse_lower_apply_visible_passes"] == 10
    assert report["coverage"]["exact_coordinate_round_trips"] == 10
    assert all(
        value == 10
        for value in report["coverage"]["corruption_rejections"].values()
    )
    assert all(
        value >= 1
        for value in report["coverage"]["operation_mechanics"].values()
    )
    assert report["counters"]["candidate_or_control_calls"] == 0
    assert report["counters"]["hidden_evaluator_calls"] == 0
    assert report["project_selected_quality_token_cap"] is None
