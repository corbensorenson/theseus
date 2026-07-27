from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import training_acceleration_final_selector as selector


def test_final_selector_binds_every_finite_candidate_disposition() -> None:
    config = json.loads(
        (
            ROOT / "configs/training_acceleration_final_selector.json"
        ).read_text(encoding="utf-8")
    )

    assert set(config["reports"]) == {
        "fp16_stability",
        "fp16_replay",
        "optimizer",
        "structural_growth",
        "progressive_sequence",
        "lazy_optimizer_state",
        "bounded_dispositions",
        "fast_sync_qualification",
        "target_window_qualification",
        "station_attention",
        "station_swiglu",
        "station_rmsnorm",
        "station_clip",
        "station_adamw",
    }
    assert config["selected_recipe"]["compute_dtype"] == "float32"
    assert config["selected_recipe"]["optimizer_id"] == "adamw_mlx"
    assert (
        config["selected_recipe"]["structural_growth_policy"]
        == "full_depth_from_step_zero"
    )
    assert (
        config["selected_recipe"]["sequence_length_policy"]
        == "fixed_target_width"
    )


def test_final_selector_is_green_against_current_authoritative_evidence() -> None:
    report = selector.execute(
        ROOT / "configs/training_acceleration_final_selector.json"
    )

    assert report["trigger_state"] == "GREEN"
    assert all(report["gates"].values())
    assert report["campaign_disposition"]["launch_recipe_changed_by_challenger"] is False
    assert set(report["candidate_dispositions"]) == {
        "fp16_fp32_master",
        "ademamix_mlx",
        "adam_mini_mlx",
        "masked_depth_conservative_v1",
        "masked_depth_paper_shaped_v1",
        "progressive_128_256_512",
        "lazy_inactive_optimizer_state",
        "fused_linear_cross_entropy",
        "exact_sequence_packing",
        "rust_training_hot_loop",
        "mlx_metal_fast_synch",
        "source_target_window_projection",
        "asynchronous_checkpoint_publication",
        "new_custom_metal_kernel",
        "mlx_sdpa_blocks_on_fp32",
        "apple_neural_accelerator_on_m1",
        "high_power_mode",
    }
    assert report["bounded_measurements"]["fast_sync_minimum_speedup"] < 1.0
    assert (
        report["bounded_measurements"][
            "target_window_projection_maximum_model_delta"
        ]
        > 5e-6
    )
    assert (
        report["campaign_disposition"]["kind"]
        == "FINITE_ACCELERATION_SELECTOR_RETAIN_CURRENT_ROUTE"
    )
