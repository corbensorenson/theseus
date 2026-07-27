from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mlx_projection_fusion_probe",
    ROOT / "scripts" / "mlx_projection_fusion_probe.py",
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_projected_microbatch_fraction_counts_all_layers() -> None:
    observed = PROBE.projected_microbatch_fraction(
        0.010,
        0.006,
        layer_count=12,
        reference_microbatch_seconds=0.56,
    )
    assert abs(observed - (0.004 * 12 / 0.56)) < 1e-12


def test_timing_summary_uses_median_not_best_case() -> None:
    observed = PROBE.timing_summary([0.004, 0.002, 0.003])
    assert observed["minimum"] == 0.002
    assert observed["median"] == 0.003
    assert observed["maximum"] == 0.004
