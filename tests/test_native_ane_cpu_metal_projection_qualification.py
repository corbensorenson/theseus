from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import native_ane_cpu_metal_projection_qualification as qualification


def test_frozen_tensors_are_exact_shape_dtype_and_replayable() -> None:
    first = qualification.deterministic_values()
    second = qualification.deterministic_values()
    assert [value.shape for value in first] == [
        (2048, 512),
        (512, 512),
        (2048, 512),
    ]
    assert all(value.dtype == np.float32 for value in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_compare_enforces_predeclared_tolerance() -> None:
    observed = np.asarray([0.0, 1.001, 1.2], dtype=np.float32)
    reference = np.asarray([0.0, 1.0, 1.0], dtype=np.float32)
    result = qualification.compare(observed, reference, tolerance=0.01)
    assert result["mismatch_count"] == 1
    assert result["maximum_absolute_delta"] == pytest.approx(0.2)


def test_adjudication_requires_parity_mechanics_and_conservative_wall_win() -> None:
    native = {
        "timing": {
            "summaries_milliseconds": {
                "joined": {"mean_milliseconds": 4.0}
            }
        },
        "stability": {
            "all_tensors_finite": True,
            "save_reload_exact": True,
            "replay_exact": True,
        },
        "custody": {
            "single_generation_conserved": True,
            "intermediate_host_tensor_copy": False,
            "hot_step_python_or_numpy": False,
        },
    }
    mlx = {"timing": {"mean_milliseconds": 5.0}}
    parity = {
        "output": {
            "mismatch_count": 0,
        }
    }
    selected = qualification.adjudicate(
        native_reports=[native, native],
        mlx_runs=[mlx, mlx],
        parity=parity,
    )
    assert selected["selected"] is True
    slower = qualification.adjudicate(
        native_reports=[
            {
                **native,
                "timing": {
                    "summaries_milliseconds": {
                        "joined": {"mean_milliseconds": 6.0}
                    }
                },
            }
        ]
        * 2,
        mlx_runs=[mlx, mlx],
        parity=parity,
    )
    assert slower["selected"] is False
    assert slower["disposition"] == "NATIVE_ZERO_COPY_TRIAD_NOT_SELECTED"


def test_native_source_has_direct_surface_and_single_update_contract() -> None:
    source = (
        ROOT / "native/ane_metal/ane_cpu_metal_projection_triad.m"
    ).read_text(encoding="utf-8")
    assert "EvaluateANE" in source
    assert "cblas_sgemm" in source
    assert "IOSurfaceRef" in source
    assert "setComputePipelineState:metal->adamw" in source
    assert "hot_step_python_or_numpy" in source
    assert "intermediate_host_tensor_copy" in source


def test_checked_in_hardware_report_is_narrow_fallback() -> None:
    report_path = (
        ROOT
        / "reports/native_ane_cpu_metal_projection_qualification_m1.json"
    )
    if not report_path.is_file():
        pytest.skip("M1 hardware report not checked in")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["trigger_state"] == "RED_FALLBACK_TO_QUALIFIED_MLX"
    assert report["decision"]["native_mechanics_green"] is True
    assert report["decision"]["parity_green"] is True
    assert report["decision"]["selected"] is False
    assert (
        report["decision"]["matched_joined_wall_gain_exceeds_uncertainty"]
        is False
    )
    assert report["canonical_checkpoint_mutated"] is False
