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
        "compile_milliseconds": 177.0,
        "mean_evaluation_milliseconds": 0.37,
        "comparisons": {
            name: {
                "tolerance": 0.002,
                "maximum_absolute_delta": 0.0003,
                "mismatch_count": 0,
            }
            for name in ("dq_rope", "dk_tiled_rope", "dv_tiled")
        },
    }


def test_validation_keeps_gradient_closure_open() -> None:
    report = module.validate(native_receipt())
    assert report["state"].endswith("GRADIENT_CLOSURE_OPEN")
    assert report["gates"]["causal_softmax_backward"] is True
    assert report["gates"]["full_query_head_dq_dk_dv"] is True
    assert report["gates"]["qkv_parameter_gradients"] is False
    assert report["gates"]["complete_attention_gradient_tree"] is False
    assert report["gates"]["production_eligible"] is False


def test_validation_fails_closed_on_gradient_mismatch() -> None:
    receipt = copy.deepcopy(native_receipt())
    receipt["comparisons"]["dv_tiled"]["mismatch_count"] = 1
    with pytest.raises(module.QualificationFault, match="comparison_failed:dv_tiled"):
        module.validate(receipt)
