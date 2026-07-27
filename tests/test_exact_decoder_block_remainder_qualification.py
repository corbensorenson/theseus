from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/exact_decoder_block_remainder_qualification.py"
SPEC = importlib.util.spec_from_file_location(
    "exact_decoder_block_remainder_qualification", MODULE_PATH
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def native_receipt() -> dict:
    comparisons = {
        name: {
            "tolerance": 1e-5,
            "maximum_absolute_delta": 1e-7,
            "mismatch_count": 0,
        }
        for name in (
            "swiglu_activation",
            "swiglu_gate_gradient",
            "swiglu_up_gradient",
        )
    }
    gates = {
        name: True
        for name in (
            "out_projection_and_unscaled_residual",
            "second_rmsnorm_forward_backward",
            "swiglu_forward_backward",
            "down_projection",
            "masked_scalar_loss",
            "all_five_parameter_leaves",
            "attended_and_direct_hidden_gradients",
            "one_global_clip",
            "one_fp32_adamw_update",
            "replay_exact",
            "sixty_four_step_finite",
        )
    }
    gates["complete_attention_join"] = False
    gates["production_eligible"] = False
    return {
        "policy": "project_theseus_exact_decoder_block_remainder_v1",
        "trigger_state": "GREEN",
        "mismatch_count": 0,
        "shape": {"rows": 128, "d_model": 512, "ff_dim": 1536},
        "parameter_generation": 0,
        "parameter_elements": 2_621_952,
        "objective_authority_mass": 64_000.0,
        "timing": {"mean_joined_64_milliseconds": 7.0},
        "comparisons": comparisons,
        "loss": 1.0,
        "gradient_norm": 2.0,
        "nonzero_gradient_fraction": 1.0,
        "gates": gates,
    }


def test_validation_closes_exact_remainder_only() -> None:
    report = module.validate(native_receipt())
    assert report["state"] == "GREEN_EXACT_BLOCK_REMAINDER_ATTENTION_JOIN_OPEN"
    assert report["gates"]["native_metal_elementwise_loss_reduction_update"] is True
    assert report["gates"]["all_nine_block_parameter_leaves"] is False
    assert report["gates"]["complete_decoder_block"] is False
    assert report["gates"]["production_eligible"] is False


def test_validation_fails_closed_on_missing_update_gate() -> None:
    receipt = copy.deepcopy(native_receipt())
    receipt["gates"]["one_fp32_adamw_update"] = False
    with pytest.raises(module.QualificationFault, match="missing_gate"):
        module.validate(receipt)


def test_validation_fails_closed_on_partial_gradient_tree() -> None:
    receipt = copy.deepcopy(native_receipt())
    receipt["nonzero_gradient_fraction"] = 0.99
    with pytest.raises(
        module.QualificationFault, match="incomplete_gradient_coverage"
    ):
        module.validate(receipt)
