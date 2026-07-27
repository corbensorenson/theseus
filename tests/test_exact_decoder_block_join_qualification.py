from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/exact_decoder_block_join_qualification.py"
SPEC = importlib.util.spec_from_file_location(
    "exact_decoder_block_join_qualification", PATH
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def receipt() -> dict:
    gates = {
        name: True
        for name in (
            "one_process",
            "compile_once_ane_forward_backward",
            "generation_tagged_iosurface_forward_backward",
            "single_thread_fp32_accelerate_dw",
            "native_metal_remainder",
            "all_nine_parameter_leaves",
            "combined_hidden_gradient",
            "one_objective_mass_normalization",
            "one_global_norm_and_clip",
            "one_fp32_adamw_publication",
            "replay_exact",
            "all_finite",
            "sixty_four_step_finite",
        )
    }
    gates["matched_mlx_wall_control"] = False
    gates["production_eligible"] = False
    return {
        "policy": "project_theseus_exact_decoder_block_join_v1",
        "trigger_state": "GREEN",
        "mismatch_count": 0,
        "shape": {
            "batch": 1,
            "sequence": 128,
            "d_model": 512,
            "ff_dim": 1536,
            "query_heads": 8,
            "kv_heads": 2,
        },
        "parameter_generation": 0,
        "parameter_elements": 3_015_680,
        "parameter_leaf_count": 9,
        "objective_authority_mass": 64_000.0,
        "timing": {"joined_milliseconds": 12.0},
        "loss": 1.0,
        "global_gradient_norm": 2.0,
        "attention_nonzero_gradient_fraction": 1.0,
        "remainder_nonzero_gradient_fraction": 1.0,
        "minimum_leaf_nonzero_gradient_fraction": 1.0,
        "gates": gates,
    }


def test_validation_closes_complete_block_mechanics() -> None:
    report = module.validate(receipt())
    assert report["state"] == "GREEN_EXACT_DECODER_BLOCK_JOIN_MLX_CONTROL_OPEN"
    assert report["gates"]["complete_decoder_block_mechanics"] is True
    assert report["gates"]["matched_mlx_wall_control"] is False
    assert report["gates"]["production_eligible"] is False


def test_validation_fails_on_second_optimizer_authority() -> None:
    value = copy.deepcopy(receipt())
    value["gates"]["one_fp32_adamw_publication"] = False
    with pytest.raises(module.QualificationFault, match="missing_gate"):
        module.validate(value)


def test_validation_uses_frozen_gradient_coverage_gate() -> None:
    value = copy.deepcopy(receipt())
    value["minimum_leaf_nonzero_gradient_fraction"] = 0.949
    with pytest.raises(
        module.QualificationFault,
        match="gradient_coverage_below_frozen_gate",
    ):
        module.validate(value)
