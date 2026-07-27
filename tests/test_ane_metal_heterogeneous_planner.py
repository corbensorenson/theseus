from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ane_metal_heterogeneous_planner as planner


def load_config() -> dict:
    return json.loads(
        (ROOT / "configs/ane_metal_heterogeneous_execution.json").read_text(
            encoding="utf-8"
        )
    )


def green_gates() -> dict[str, bool]:
    return {gate: True for gate in planner.REQUIRED_GATES}


def test_config_defaults_disabled_and_covers_full_split_sweep() -> None:
    config = load_config()
    planner.validate_config(config)

    assert config["canonical_enabled"] is False
    assert config["split_ratios"][0] == 0.0
    assert config["split_ratios"][-1] == 1.0
    assert config["fallback"]["timing"] == "before_checkpoint_mutation"


def test_unstable_compiler_blocks_even_fast_candidate() -> None:
    evidence = {
        "shape_id": "qkv_projection",
        "compiler_attempts": [
            {"compiled": True},
            {"compiled": False},
        ],
        "metal_control_seconds": [10.0, 10.1, 9.9],
        "split_candidates": [
            {
                "ane_output_fraction": 0.5,
                "concurrent_wall_seconds": [7.0, 7.1, 6.9],
                "concurrent_gpu_seconds": [6.8, 6.9, 6.7],
                "concurrent_ane_seconds": [6.5, 6.6, 6.4],
                "gpu_isolation_seconds": [7.0, 7.1, 7.2],
                "ane_isolation_seconds": [6.8, 6.9, 7.0],
                "join_and_fence_seconds": [0.1, 0.1, 0.1],
                "gates": green_gates(),
            }
        ],
    }

    report = planner.plan(load_config(), evidence)

    assert report["compiler"]["state"] == "M1_PRIVATE_COMPILER_UNSTABLE"
    assert report["selected"] is None
    assert report["checkpoint_mutation_authorized"] is False


def test_selection_requires_worst_candidate_to_beat_best_control() -> None:
    base = {
        "shape_id": "gate_up_projection",
        "compiler_attempts": [{"compiled": True}, {"compiled": True}],
        "metal_control_seconds": [10.0, 10.1, 9.9],
    }
    uncertain = {
        **base,
        "split_candidates": [
            {
                "ane_output_fraction": 0.625,
                "concurrent_wall_seconds": [9.7, 9.8, 10.0],
                "concurrent_gpu_seconds": [9.5, 9.6, 9.8],
                "concurrent_ane_seconds": [9.1, 9.2, 9.3],
                "gpu_isolation_seconds": [9.7, 9.8, 9.9],
                "ane_isolation_seconds": [9.3, 9.4, 9.5],
                "join_and_fence_seconds": [0.1, 0.1, 0.1],
                "gates": green_gates(),
            }
        ],
    }
    green = {
        **base,
        "split_candidates": [
            {
                "ane_output_fraction": 0.625,
                "concurrent_wall_seconds": [8.7, 8.8, 8.9],
                "concurrent_gpu_seconds": [8.5, 8.6, 8.7],
                "concurrent_ane_seconds": [8.1, 8.2, 8.3],
                "gpu_isolation_seconds": [8.6, 8.7, 8.8],
                "ane_isolation_seconds": [8.2, 8.3, 8.4],
                "join_and_fence_seconds": [0.1, 0.1, 0.1],
                "gates": green_gates(),
            }
        ],
    }

    assert planner.plan(load_config(), uncertain)["selected"] is None
    selected = planner.plan(load_config(), green)["selected"]
    assert selected is not None
    assert selected["ane_output_fraction"] == pytest.approx(0.625)
    assert selected["conservative_speedup"] > 1.0


def test_missing_integrity_gate_forces_fallback() -> None:
    gates = green_gates()
    gates["gradient_parity"] = False
    evidence = {
        "shape_id": "qkv_projection",
        "compiler_attempts": [{"compiled": True}, {"compiled": True}],
        "metal_control_seconds": [10.0, 10.0],
        "split_candidates": [
            {
                "ane_output_fraction": 0.5,
                "concurrent_wall_seconds": [5.0, 5.1],
                "concurrent_gpu_seconds": [4.8, 4.9],
                "concurrent_ane_seconds": [4.7, 4.8],
                "gpu_isolation_seconds": [4.8, 4.9],
                "ane_isolation_seconds": [4.7, 4.8],
                "join_and_fence_seconds": [0.1, 0.1],
                "gates": gates,
            }
        ],
    }

    report = planner.plan(load_config(), evidence)

    assert report["selected"] is None
    assert "gradient_parity" in report["candidates"][0]["missing_or_failed_gates"]


def test_no_timings_preserves_inconclusive_implementation() -> None:
    report = planner.plan(
        load_config(),
        {
            "shape_id": "qkv_projection",
            "compiler_attempts": [
                {"compiled": True},
                {"compiled": False},
                {"compiled": False},
            ],
        },
    )

    assert report["trigger_state"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert report["selected"] is None
    assert report["canonical_backend_changed"] is False


def test_current_m1_evidence_routes_past_visibility_to_persistent_partition() -> None:
    evidence = json.loads(
        (ROOT / "configs/ane_metal_m1_evidence_2026_07_27.json").read_text(
            encoding="utf-8"
        )
    )

    report = planner.plan(load_config(), evidence)

    assert report["compiler"]["repeatable"] is True
    assert (
        report["same_surface_bridge"]["state"]
        == "GREEN_CONCURRENT_SHARED_READ_VISIBILITY"
    )
    assert "zero_copy_same_surface_visibility" not in report["blockers"]
    assert "structure_aligned_persistent_partition_candidate" in report["blockers"]
    assert "dynamic_or_persistent_training_weight_update_path" in report["blockers"]
    assert report["checkpoint_mutation_authorized"] is False
