from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kimi_k3_architecture_qualification as qualification


CONFIG = ROOT / "configs/kimi_k3_attnres_qualification.json"
SITU_CONFIG = ROOT / "configs/kimi_k3_situ_glu_qualification.json"


def test_attnres_config_is_finite_source_disjoint_six_million_rung() -> None:
    config = qualification.load_config(CONFIG)
    assert config["variant"]["kind"] == "block_attention_residual"
    assert config["variant"]["block_size"] == 2
    assert config["training"]["steps"] == 128
    assert config["seeds"] == [20260722, 20260723, 20260724]
    preflight = qualification.preflight(CONFIG)
    assert preflight["trigger_state"] == "GREEN"
    assert preflight["source_overlap_count"] == 0
    assert preflight["parameter_count"]["counts"] == {
        "control": 6623232,
        "candidate": 6625024,
    }


def test_attnres_model_contract_refuses_in_place_control_migration() -> None:
    config = qualification.load_config(CONFIG)
    control = qualification.model_config(
        config, 8195, candidate=False
    )
    candidate = qualification.model_config(
        config, 8195, candidate=True
    )
    assert control.attention_residual_mode == "none"
    assert candidate.attention_residual_mode == "block"
    assert candidate.attention_residual_block_size == 2
    with pytest.raises(
        qualification.KimiK3ArchitectureFault,
        match="in_place_topology_migration_forbidden",
    ):
        qualification.assert_migration_compatible(control, candidate)


def test_situ_config_is_matched_and_refuses_swiglu_migration() -> None:
    config = qualification.load_config(SITU_CONFIG)
    assert config["variant"]["kind"] == "situ_glu"
    assert config["variant"]["gate_beta"] == 4.0
    assert config["variant"]["up_beta"] == 25.0
    assert config["training"]["steps"] == 128
    assert config["seeds"] == [20260722, 20260723, 20260724]
    preflight = qualification.preflight(SITU_CONFIG)
    assert preflight["trigger_state"] == "GREEN"
    assert preflight["source_overlap_count"] == 0
    assert preflight["parameter_count"]["counts"] == {
        "control": 6623232,
        "candidate": 6623232,
    }
    control = qualification.model_config(
        config, 8195, candidate=False
    )
    candidate = qualification.model_config(
        config, 8195, candidate=True
    )
    assert control.feed_forward_activation == "swiglu"
    assert candidate.feed_forward_activation == "situ_glu"
    assert candidate.situ_glu_gate_beta == 4.0
    assert candidate.situ_glu_up_beta == 25.0
    with pytest.raises(
        qualification.KimiK3ArchitectureFault,
        match="in_place_topology_migration_forbidden",
    ):
        qualification.assert_migration_compatible(control, candidate)


def test_attnres_adoption_requires_material_restart_repayment() -> None:
    config = qualification.load_config(CONFIG)
    arms = {arm: {"ntp_loss": 1.0} for arm in config["scoped_arms"]}
    runs = []
    for kind, final_loss, wall, quality_wall in (
        ("control", 1.0, 100.0, 80.0),
        ("candidate", 0.99, 90.0, 70.0),
    ):
        for seed in config["seeds"]:
            runs.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "joined_training_wall_seconds": wall,
                    "training_primary_step_seconds": wall * 0.8,
                    "peak_allocator_bytes": 100,
                    "curve": [
                        {
                            "step": 0,
                            "joined_wall_seconds": 0.0,
                            "heldout": {"ntp_loss": 2.0},
                        },
                        {
                            "step": 128,
                            "joined_wall_seconds": quality_wall,
                            "heldout": {"ntp_loss": final_loss},
                        },
                    ],
                    "final_heldout": {
                        "ntp_loss": final_loss,
                        "by_arm": arms,
                    },
                    "checkpoint_replay": {
                        "exact_model_reload": True,
                        "exact_optimizer_reload": True,
                        "next_update_numerically_equivalent": True,
                    },
                    "finite_gradients": True,
                }
            )
    comparison = qualification.compare(runs, config)
    assert comparison["disposition"] == "NOT_SELECTED_FIRST_CAMPAIGN"
    assert comparison["quality_route_gates"][
        "material_mean_loss_improvement"
    ] is False


def test_mutated_public_boundary_fails_closed() -> None:
    config = json.loads(CONFIG.read_text())
    config["hard_boundaries"]["public_training_rows"] = 1
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "bad.json"
        path.write_text(json.dumps(config))
        with pytest.raises(
            qualification.KimiK3ArchitectureFault,
            match="hard_boundary_nonzero",
        ):
            qualification.load_config(path)
