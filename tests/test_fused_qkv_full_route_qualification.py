from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fused_qkv_full_route_qualification as qualification


CONFIG = ROOT / "configs/fused_qkv_full_route_qualification.json"


def test_config_binds_three_alternating_finite_pairs() -> None:
    config = qualification.load_config(CONFIG)
    assert [row["order"] for row in config["pairs"]] == [
        ["control", "candidate"],
        ["candidate", "control"],
        ["control", "candidate"],
    ]
    assert config["hard_boundaries"]["optimizer_steps_per_process"] == 8
    assert config["hard_boundaries"]["production_checkpoint_mutation"] is False
    assert config["hard_boundaries"]["public_training_rows"] == 0


def test_decision_rejects_inconsistent_and_numerically_drifting_route() -> None:
    result = qualification.decide(
        speedup_ratios=[0.98, 0.85, 1.11],
        loss_deltas=[1e-4, -2e-4, 3e-4],
        model_state_passed=False,
        optimizer_state_passed=False,
        rng_state_exact=True,
        matched_authority=True,
        resources_passed=True,
        maximum_loss_delta=2e-6,
    )
    assert result["selected"] is False
    assert result["candidate_win_count"] == 1
    assert result["gates"]["candidate_wins_every_pair"] is False
    assert result["gates"]["final_loss_within_frozen_tolerance"] is False
    assert result["gates"]["model_state_within_frozen_tolerance"] is False


def test_decision_requires_no_arbitrary_percentage_floor() -> None:
    result = qualification.decide(
        speedup_ratios=[1.001, 1.002, 1.003],
        loss_deltas=[0.0, 0.0, 0.0],
        model_state_passed=True,
        optimizer_state_passed=True,
        rng_state_exact=True,
        matched_authority=True,
        resources_passed=True,
        maximum_loss_delta=2e-6,
    )
    assert result["selected"] is True


def test_mutated_public_boundary_fails_closed(tmp_path: Path) -> None:
    import json

    config = json.loads(CONFIG.read_text())
    config["hard_boundaries"]["public_training_rows"] = 1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(
        qualification.FusedQKVQualificationFault,
        match="hard_boundary_nonzero",
    ):
        qualification.load_config(path)
