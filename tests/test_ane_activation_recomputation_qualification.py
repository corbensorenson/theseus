from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ane_activation_recomputation_qualification as qualification


def native_payload(milliseconds: float = 40.0) -> dict:
    return {
        "trigger_state": "GREEN_RECOMPUTE_MECHANICS",
        "runtime": {"barrier_work_milliseconds": milliseconds},
        "parity": {"mismatch_count": 0},
        "custody": {"intermediate_python_or_numpy_round_trip": False},
        "memory": {
            "discarded_gate_up_fp32_mib_per_layer": 24.0,
            "retained_layer_boundary_fp32_mib": 4.0,
            "maximum_twelve_layer_discarded_mib": 288.0,
        },
    }


def mlx_payload(milliseconds: float = 200.0) -> dict:
    return {
        "trigger_state": "GREEN_ATTENTION_BACKWARD_WINDOW",
        "runtime": {"barrier_work_milliseconds": milliseconds},
        "mechanics": {"all_gradients_finite": True},
    }


def test_adjudication_requires_conservative_joined_wall_win() -> None:
    result = qualification.adjudicate(
        [native_payload(), native_payload()],
        [mlx_payload(220.0), mlx_payload(210.0)],
        [
            {
                "native": native_payload(45.0),
                "mlx": mlx_payload(190.0),
                "critical_path_milliseconds": 190.0,
            },
            {
                "native": native_payload(44.0),
                "mlx": mlx_payload(195.0),
                "critical_path_milliseconds": 195.0,
            },
        ],
        repetitions=10,
    )
    assert result["ane_recompute_hidden_inside_mlx_window"] is True
    assert result["joined_wall_selected"] is True
    assert result["production_eligible"] is False


def test_adjudication_rejects_contention_slowdown() -> None:
    result = qualification.adjudicate(
        [native_payload(), native_payload()],
        [mlx_payload(180.0), mlx_payload(180.0)],
        [
            {
                "native": native_payload(50.0),
                "mlx": mlx_payload(220.0),
                "critical_path_milliseconds": 220.0,
            },
            {
                "native": native_payload(50.0),
                "mlx": mlx_payload(225.0),
                "critical_path_milliseconds": 225.0,
            },
        ],
        repetitions=10,
    )
    assert result["joined_wall_selected"] is False


def test_native_source_binds_exact_topology_and_no_hot_python_bridge() -> None:
    source = (
        ROOT / "native/ane_metal/ane_swiglu_activation_recompute.m"
    ).read_text(encoding="utf-8")
    assert "#define ROWS 2048" in source
    assert "#define FF_DIM 1536" in source
    assert "#define PROJECTION_COUNT 6" in source
    assert "intermediate_python_or_numpy_round_trip\\\":false" in source
    assert "EvaluateANE" in source


def test_checked_in_report_stays_narrow_if_present() -> None:
    path = ROOT / "reports/ane_activation_recomputation_m1.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["canonical_backend_changed"] is False
    assert report["canonical_checkpoint_mutated"] is False
    assert report["decision"]["production_eligible"] is False
