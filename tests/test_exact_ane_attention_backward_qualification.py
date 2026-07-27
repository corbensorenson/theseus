from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/exact_ane_attention_backward_qualification.py"
SPEC = importlib.util.spec_from_file_location(
    "exact_ane_attention_backward_qualification", MODULE_PATH
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def native_receipt() -> dict:
    return {
        "policy": "project_theseus_exact_ane_attention_backward_v1",
        "trigger_state": "GREEN",
        "mismatch_count": 0,
        "shape": {
            "sequence": 128,
            "query_heads": 8,
            "kv_heads": 2,
            "head_dim": 64,
        },
        "parameter_generation": 0,
        "compile_milliseconds": 177.0,
        "mean_evaluation_milliseconds": 0.37,
        "mean_cpu_projection_gradient_milliseconds": 0.3,
        "comparisons": {
            name: {
                "tolerance": 0.002,
                "maximum_absolute_delta": 0.0003,
                "mismatch_count": 0,
            }
            for name in (
                "dq_rope",
                "dk_tiled_rope",
                "dv_tiled",
                "dq_inverse_split_half_rope",
                "dk_contiguous_reduce_inverse_split_half_rope",
                "dv_contiguous_reduce",
            )
        }
        | {
            "accelerate_projection_operator": {
                "tolerance": 0.0001,
                "maximum_absolute_delta": 1e-7,
                "mismatch_count": 0,
            }
        }
        | {
            name: {
                "tolerance_policy": "analytical_fp16_boundary_propagation",
                "maximum_allowed_delta": 0.05,
                "maximum_absolute_delta": 0.02,
                "mismatch_count": 0,
            }
            for name in (
                "q_proj_weight_gradient",
                "k_proj_weight_gradient",
                "v_proj_weight_gradient",
                "projection_input_gradient",
                "attention_norm_input_gradient",
                "attention_norm_scale_gradient",
            )
        },
    }


def test_validation_closes_attention_gradient_tree() -> None:
    report = module.validate(native_receipt())
    assert report["state"].endswith("BLOCK_REMAINDER_OPEN")
    assert report["gates"]["causal_softmax_backward"] is True
    assert report["gates"]["full_query_head_dq_dk_dv"] is True
    assert report["gates"]["contiguous_gqa_kv_reduction"] is True
    assert report["gates"]["inverse_split_half_rope"] is True
    assert report["gates"]["qkv_parameter_gradients"] is True
    assert report["gates"]["attention_rmsnorm_input_gradient"] is True
    assert report["gates"]["complete_attention_gradient_tree"] is True
    assert report["gates"]["production_eligible"] is False


def test_validation_fails_closed_on_gradient_mismatch() -> None:
    receipt = copy.deepcopy(native_receipt())
    receipt["comparisons"]["dv_tiled"]["mismatch_count"] = 1
    with pytest.raises(module.QualificationFault, match="comparison_failed:dv_tiled"):
        module.validate(receipt)


def test_validation_fails_closed_on_propagation_bound_violation() -> None:
    receipt = copy.deepcopy(native_receipt())
    receipt["comparisons"]["q_proj_weight_gradient"][
        "maximum_absolute_delta"
    ] = 0.06
    with pytest.raises(
        module.QualificationFault,
        match="comparison_failed:q_proj_weight_gradient",
    ):
        module.validate(receipt)
