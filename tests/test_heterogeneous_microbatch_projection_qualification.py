from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import heterogeneous_microbatch_projection_qualification as qualification


def test_frozen_inputs_are_exact_and_disjoint() -> None:
    first = qualification.frozen_inputs()
    second = qualification.frozen_inputs()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["weight"].shape == (512, 512)
    assert first["x_a"].shape == (2048, 512)
    assert not np.array_equal(first["x_a"], first["x_b"])


def test_comparison_is_fail_closed() -> None:
    expected = np.zeros((2,), dtype=np.float32)
    actual = np.array([0.0, 0.2], dtype=np.float32)
    result = qualification.compare(actual, expected, 0.1)
    assert result["mismatch_count"] == 1
    assert result["all_finite"] is True


def test_checked_in_report_remains_station_scoped_if_present() -> None:
    path = ROOT / "reports/heterogeneous_microbatch_projection_m1.json"
    if not path.exists():
        return
    report = json.loads(path.read_text())
    assert report["production_eligible"] is False
    assert report["canonical_checkpoint_mutated"] is False
    assert (
        report["gates"]["complete_ane_model_gradient_tree_parity"] is False
    )
