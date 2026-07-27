from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import heterogeneous_microbatch_contract as contract


SCHEMA = {"weight": [[2, 2], "float32"]}
ROWS = ["row0", "row1", "row2", "row3"]
MASSES = {row_id: float(index + 1) for index, row_id in enumerate(ROWS)}


def contribution(
    shard_id: str,
    engine: str,
    rows: tuple[str, ...],
    value: float,
) -> contract.GradientContribution:
    return contract.GradientContribution(
        shard_id=shard_id,
        engine=engine,
        generation=7,
        row_ids=rows,
        row_objective_masses=tuple(MASSES[row_id] for row_id in rows),
        global_objective_mass=sum(MASSES.values()),
        gradients={"weight": np.full((2, 2), value, dtype=np.float32)},
    )


def test_join_preserves_exact_sampler_mass_and_one_accumulator() -> None:
    gradients, receipt = contract.join_gradient_contributions(
        [
            contribution("metal", "mlx_metal", ("row0", "row1"), 1.0),
            contribution("ane", "ane_accelerate", ("row2", "row3"), 2.0),
        ],
        generation=7,
        expected_rows=ROWS,
        expected_row_objective_masses=MASSES,
        expected_schema=SCHEMA,
    )
    np.testing.assert_array_equal(
        gradients["weight"], np.full((2, 2), 3.0, dtype=np.float32)
    )
    assert receipt["global_objective_mass"] == 10.0
    assert receipt["one_fp32_accumulator"] is True
    assert receipt["local_optimizer_steps"] == 0


def test_join_rejects_row_overlap() -> None:
    with pytest.raises(contract.ContractFault, match="sampler_row_overlap"):
        contract.join_gradient_contributions(
            [
                contribution("metal", "mlx_metal", ("row0", "row1"), 1.0),
                contribution("ane", "ane_accelerate", ("row1", "row3"), 2.0),
            ],
            generation=7,
            expected_rows=ROWS,
            expected_row_objective_masses=MASSES,
            expected_schema=SCHEMA,
        )


def test_join_rejects_stale_generation_and_local_optimizer() -> None:
    base = contribution("ane", "ane_accelerate", ("row2", "row3"), 2.0)
    for replacement, fault in (
        ({"generation": 6}, "stale_or_mixed_parameter_generation"),
        ({"local_optimizer_steps": 1}, "per_device_optimizer_update_forbidden"),
    ):
        altered = contract.GradientContribution(
            **{**base.__dict__, **replacement}
        )
        with pytest.raises(contract.ContractFault, match=fault):
            contract.join_gradient_contributions(
                [
                    contribution(
                        "metal", "mlx_metal", ("row0", "row1"), 1.0
                    ),
                    altered,
                ],
                generation=7,
                expected_rows=ROWS,
                expected_row_objective_masses=MASSES,
                expected_schema=SCHEMA,
            )


def test_clip_and_adamw_publishes_once() -> None:
    parameters = {"weight": np.ones((2, 2), dtype=np.float32)}
    zeros = {"weight": np.zeros((2, 2), dtype=np.float32)}
    gradients = {"weight": np.full((2, 2), 3.0, dtype=np.float32)}
    updated, first, second, receipt = contract.clip_and_adamw_once(
        parameters,
        zeros,
        zeros,
        gradients,
        learning_rate=3e-4,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=0.1,
        clip_norm=1.0,
    )
    assert receipt["global_clip_count"] == 1
    assert receipt["adamw_update_count"] == 1
    assert receipt["published_generation_increment"] == 1
    assert receipt["all_state_finite"] is True
    assert np.all(updated["weight"] < parameters["weight"])
    assert np.all(first["weight"] > 0)
    assert np.all(second["weight"] > 0)


def test_native_gradient_mode_forbids_local_update() -> None:
    source = (
        ROOT / "native/ane_metal/ane_cpu_metal_projection_triad.m"
    ).read_text(encoding="utf-8")
    assert "--gradient-only" in source
    assert "gradient_contribution_only" in source
    assert "local_optimizer_update_forbidden" in source
