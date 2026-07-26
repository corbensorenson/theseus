from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mlx_training_operator_station_probe",
    ROOT / "scripts" / "mlx_training_operator_station_probe.py",
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_layer_station_bound_expands_across_twelve_layers() -> None:
    bound = PROBE.station_bound(
        "swiglu_mlp",
        0.01,
        layer_count=12,
        reference_microbatch_seconds=0.56,
    )
    assert bound["station_instances_per_optimizer_microbatch"] == 12
    assert bound["repeated_station_seconds_upper_bound"] == 0.12
    assert bound["custom_kernel_10_percent_bound_possible"] is True


def test_update_station_bound_is_counted_once() -> None:
    bound = PROBE.station_bound(
        "global_gradient_clip",
        0.02,
        layer_count=12,
        reference_microbatch_seconds=0.56,
    )
    assert bound["station_instances_per_optimizer_microbatch"] == 1
    assert bound["elimination_fraction_of_reference_microbatch"] == 0.035714
    assert bound["custom_kernel_10_percent_bound_possible"] is False


def test_percentile_requires_observations() -> None:
    try:
        PROBE.percentile([], 0.95)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty observations must fail closed")
