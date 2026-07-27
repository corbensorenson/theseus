from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import source_target_window_projection_qualification as qualification


def route(*, compact: bool, rate: float) -> dict:
    return {
        "training_phase": "source_conditioned_pretraining",
        "route_mode": "eager",
        "precision_mode": "float32",
        "optimizer_steps": 4,
        "starting_checkpoint_sha256": "checkpoint",
        "starting_optimizer_state_sha256": "optimizer",
        "batch_index_sha256_prefix": ["a", "b", "c", "d"],
        "optimizer_positions": 100,
        "data_cursor_start": {"batch_index": 0},
        "data_cursor_next": {"batch_index": 4},
        "compact_output_projection": compact,
        "post_first_positions_per_second": rate,
        "final_loss": 3.5,
        "rng_content": {"sha256": "rng"},
    }


def resource(*, swap: float = 0.0) -> dict:
    return {
        "passed": True,
        "maximum_swapout_growth_mib": swap,
        "maximum_inferred_unified_memory_mib": 2048.0,
        "minimum_reclaimable_available_mib": 1024.0,
    }


def comparison(*, passed: bool, delta: float) -> dict:
    return {
        "passed": passed,
        "maximum_absolute_delta": delta,
        "maximum_relative_l2_delta": delta,
        "tolerance_mismatch_names": [] if passed else ["weight"],
    }


def test_selector_has_no_arbitrary_percentage_hurdle() -> None:
    report = qualification.decide(
        control=route(compact=False, rate=100.0),
        control_replay=route(compact=False, rate=101.0),
        candidate=route(compact=True, rate=102.0),
        control_resource=resource(),
        control_replay_resource=resource(),
        candidate_resource=resource(),
        control_replay_model_comparison=comparison(
            passed=True, delta=1e-7
        ),
        candidate_model_comparison=comparison(
            passed=True, delta=2e-7
        ),
    )

    assert report["trigger_state"] == "GREEN_SELECTED"
    assert report["selection"]["candidate_selected"] is True
    assert report["selection"]["arbitrary_percentage_hurdle"] is False


def test_state_drift_blocks_candidate_without_falsifying_mechanism() -> None:
    report = qualification.decide(
        control=route(compact=False, rate=100.0),
        control_replay=route(compact=False, rate=99.0),
        candidate=route(compact=True, rate=106.0),
        control_resource=resource(),
        control_replay_resource=resource(),
        candidate_resource=resource(swap=64.0),
        control_replay_model_comparison=comparison(
            passed=True, delta=1e-7
        ),
        candidate_model_comparison=comparison(
            passed=False, delta=7e-6
        ),
    )

    assert report["trigger_state"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert report["selection"]["candidate_selected"] is False
    assert report["gates"]["candidate_beats_mean_control"] is True
    assert report["gates"]["candidate_model_within_frozen_tolerance"] is False
    assert report["gates"]["zero_swap_growth"] is False
    assert "Repair" in report["next_gate"]
