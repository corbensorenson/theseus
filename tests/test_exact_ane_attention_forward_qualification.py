from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/exact_ane_attention_forward_qualification.py"
SPEC = importlib.util.spec_from_file_location(
    "exact_ane_attention_forward_qualification", MODULE_PATH
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def stage_receipt(stage: int) -> dict:
    if stage < 4:
        return {
            "qualification_stage": stage,
            "trigger_state": "GREEN_RUNTIME_EXECUTION",
            "compile_milliseconds": 1.0 + stage,
            "mean_evaluation_milliseconds": 0.1 + stage,
        }
    return {
        "policy": "project_theseus_exact_ane_attention_forward_v1",
        "trigger_state": "GREEN",
        "mismatch_count": 0,
        "shape": {
            "sequence": 128,
            "d_model": 512,
            "query_heads": 8,
            "kv_heads": 2,
            "head_dim": 64,
        },
        "compile_milliseconds": 5.0,
        "mean_evaluation_milliseconds": 0.5,
        "comparisons": {
            name: {
                "tolerance": 0.01,
                "maximum_absolute_delta": 0.001,
                "mismatch_count": 0,
            }
            for name in (
                "attended",
                "query_rope",
                "key_rope",
                "value",
                "attention_norm",
            )
        },
    }


def unaligned_failure() -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=["probe"],
        returncode=5,
        stdout=(
            b"ANEProgramProcessRequestDirect() Failed with status=0x1d"
        ),
    )


def test_build_command_binds_stage_and_alignment() -> None:
    command = module.build_command(3, aligned=True)
    assert "-DQUALIFICATION_STAGE=3" in command
    assert not any("NORM_SCALE_SPAN" in item for item in command)
    unaligned = module.build_command(0, aligned=False)
    assert "-DNORM_SCALE_SPAN=1" in unaligned


def test_validation_preserves_forward_only_claim_boundary() -> None:
    report = module.validate(
        [stage_receipt(stage) for stage in module.STAGES],
        unaligned_failure(),
    )
    assert report["state"] == "GREEN_EXACT_ATTENTION_FORWARD_NATIVE_BACKWARD_OPEN"
    assert report["packed_surface"]["aligned_spatial_extent"] == 1024
    assert report["gates"]["causal_attention"] is True
    assert report["gates"]["input_gradient"] is False
    assert report["gates"]["complete_decoder_block"] is False
    assert report["gates"]["production_eligible"] is False


def test_validation_fails_on_missing_gradient_boundary_drift() -> None:
    receipts = [stage_receipt(stage) for stage in module.STAGES]
    receipts[-1]["comparisons"]["attended"]["mismatch_count"] = 1
    with pytest.raises(module.QualificationFault, match="comparison_failed:attended"):
        module.validate(receipts, unaligned_failure())
