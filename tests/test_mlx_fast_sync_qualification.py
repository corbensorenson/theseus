from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mlx_fast_sync_qualification as qualification


def route() -> dict:
    return {
        "starting_checkpoint_sha256": "checkpoint",
        "starting_optimizer_state_sha256": "optimizer",
        "batch_index_sha256_prefix": ["a"],
        "optimizer_positions": 100,
        "data_cursor_start": {"batch_index": 0},
        "data_cursor_next": {"batch_index": 8},
        "final_loss": 1.5,
    }


def resource(swap: float = 0.0) -> dict:
    return {"passed": True, "maximum_swapout_growth_mib": swap}


def test_any_reversed_pair_blocks_selection_without_percentage_floor() -> None:
    report = qualification.disposition(
        speedup_ratios=[1.02, 1.03, 1.001, 0.97],
        model_comparison={"passed": True},
        current_control=route(),
        current_candidate=route(),
        control_resource=resource(128.0),
        candidate_resource=resource(64.0),
    )

    assert report["trigger_state"] == "INCONCLUSIVE_EXPERIMENT"
    assert report["selection"]["candidate_selected"] is False
    assert report["selection"]["arbitrary_percentage_hurdle"] is False
    assert report["gates"]["full_model_state_within_frozen_tolerance"] is True
    assert report["gates"]["candidate_wins_every_pair"] is False
    assert report["gates"]["zero_swap_growth"] is False


def test_consistent_free_gain_is_selected_without_minimum_percent() -> None:
    report = qualification.disposition(
        speedup_ratios=[1.001, 1.002, 1.003, 1.001],
        model_comparison={"passed": True},
        current_control=route(),
        current_candidate=route(),
        control_resource=resource(),
        candidate_resource=resource(),
    )

    assert report["trigger_state"] == "GREEN_SELECTED"
    assert report["selection"]["candidate_selected"] is True
