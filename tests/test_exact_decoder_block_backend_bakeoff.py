from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/exact_decoder_block_backend_bakeoff.py"
SPEC = importlib.util.spec_from_file_location(
    "exact_decoder_block_backend_bakeoff", PATH
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


SHAPE = {
    "batch": 1,
    "sequence": 128,
    "d_model": 512,
    "ff_dim": 1536,
    "query_heads": 8,
    "kv_heads": 2,
}


def native(milliseconds: float) -> dict:
    return {
        "trigger_state": "GREEN",
        "shape": SHAPE,
        "parameter_elements": 3_015_680,
        "parameter_leaf_count": 9,
        "objective_authority_mass": 64_000.0,
        "timing": {"mean_joined_64_milliseconds": milliseconds},
        "loss": 1.0,
        "global_gradient_norm": 2.0,
        "gates": {
            "replay_exact": True,
            "all_finite": True,
            "sixty_four_step_finite": True,
            "one_fp32_adamw_publication": True,
        },
    }


def mlx(milliseconds: float) -> dict:
    return {
        "state": "GREEN_MATCHED_COMPILED_MLX_CONTROL",
        "shape": SHAPE,
        "parameter_elements": 3_015_680,
        "parameter_leaf_count": 9,
        "objective_authority_mass": 64_000.0,
        "timing": {"mean_milliseconds": milliseconds},
        "first_loss": 1.0,
        "first_gradient_norm": 2.0,
        "gates": {
            "replay_exact": True,
            "sixty_four_step_finite": True,
            "one_fp32_adamw_publication": True,
            "matched_precision_split": True,
        },
    }


def test_selector_retains_mlx_when_native_is_slower() -> None:
    report = module.decide(
        [native(10.0), native(11.0)],
        [mlx(5.0), mlx(5.5)],
    )
    assert report["state"] == "GREEN_MATCHED_BAKEOFF_RETAIN_MLX"
    assert report["selection"]["retained_backend"] == "compiled_mlx"
    assert report["canonical_backend_changed"] is False


def test_selector_requires_conservative_native_win() -> None:
    report = module.decide(
        [native(4.0), native(6.0)],
        [mlx(5.0), mlx(5.0)],
    )
    assert report["mean_control_over_candidate_speedup"] == 1.0
    assert report["selection"]["native_selected"] is False
