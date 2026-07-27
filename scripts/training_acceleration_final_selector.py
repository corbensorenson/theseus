#!/usr/bin/env python3
"""Bind the finite acceleration docket to one launch execution recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/training_acceleration_final_selector.json"
POLICY = "project_theseus_training_acceleration_final_selector_v1"


class FinalSelectorFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalSelectorFault(f"json_object_required:{relative(path)}")
    return value


def disposition_row(report: dict[str, Any], candidate: str) -> dict[str, Any]:
    return next(
        (
            row
            for row in report.get("deprioritized_or_closed") or []
            if isinstance(row, dict) and row.get("candidate") == candidate
        ),
        {},
    )


def execute(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("policy") != POLICY:
        raise FinalSelectorFault("config_policy_invalid")
    reports = {
        name: (resolve(path), read_json(resolve(path)))
        for name, path in config["reports"].items()
    }
    trainer_path = resolve(config["production_trainer_config"])
    trainer = read_json(trainer_path)
    execution = trainer["training"]["execution_policy"]
    pretraining = execution["pretraining"]
    optimizer = reports["optimizer"][1]
    structural = reports["structural_growth"][1]
    sequence = reports["progressive_sequence"][1]
    fp16 = reports["fp16_replay"][1]
    fp16_stability = reports["fp16_stability"][1]
    lazy = reports["lazy_optimizer_state"][1]
    bounded = reports["bounded_dispositions"][1]
    fast_sync = reports["fast_sync_qualification"][1]
    target_window = reports["target_window_qualification"][1]
    station_reports = {
        name: reports[f"station_{name}"][1]
        for name in ("attention", "swiglu", "rmsnorm", "clip", "adamw")
    }
    packing = disposition_row(bounded, "exact_sequence_packing")
    fused_loss = disposition_row(
        bounded, "cut_cross_entropy_custom_metal_now"
    )
    rust_loop = disposition_row(
        bounded, "rust_rewrite_of_python_training_loop"
    )
    gates = {
        "fp16_stability_measured": (
            fp16_stability.get("compute_dtype") == "float16"
            and int(fp16_stability.get("optimizer_steps") or 0) == 64
        ),
        "fp16_replay_not_adopted": (
            fp16.get("trajectory_repeatability_state") == "YELLOW"
            and fp16.get("trajectory_divergence_predates_checkpoint") is True
        ),
        "optimizer_reference_selected": (
            optimizer.get("trigger_state") == "GREEN"
            and optimizer["campaign_disposition"]["selected_optimizer"]
            == "adamw_mlx"
            and optimizer["width_transfer"]["trigger_state"] == "GREEN"
        ),
        "structural_control_selected": (
            structural.get("trigger_state") == "GREEN"
            and structural["campaign_disposition"][
                "selected_training_structure"
            ]
            == "full_depth_control"
        ),
        "sequence_control_selected": (
            sequence.get("trigger_state") == "GREEN"
            and sequence["campaign_disposition"]["selected_sequence_policy"]
            == "fixed_512_control"
        ),
        "lazy_state_trigger_failed": (
            lazy.get("trigger_state") == "GREEN"
            and lazy.get("implementation_disposition")
            == "NOT_IMPLEMENTED_MEMORY_TO_FASTER_MICROBATCH_TRIGGER_FAILED"
            and lazy["decision"]["production_route_changed"] is False
        ),
        "fused_loss_requires_ranked_residual_or_memory_trigger": (
            fused_loss.get("disposition")
            == "DO_NOT_IMPLEMENT_BEFORE_FULL_STATION_RANKING_OR_A_PROVEN_MICROBATCH_MEMORY_TRIGGER"
            and lazy["decision"]["production_route_changed"] is False
        ),
        "packing_not_semantics_free_on_current_lineage": (
            bounded["local_hot_path"]["training_position_occupancy"][
                "fraction"
            ]
            > 0.90
            and packing.get("disposition")
            == "LOW_PRIORITY_UNLESS_A_DIFFERENT_STAGE_SHOWS_SUBSTANTIALLY_LOWER_OCCUPANCY"
        ),
        "rust_host_rewrite_rejected_by_measured_amdahl_bound": (
            rust_loop.get("disposition")
            == "CLOSED_WITHOUT_A_NEW_CPU_PROFILE"
        ),
        "fast_sync_retained_control_on_reversed_pair_and_swap": (
            fast_sync.get("trigger_state") == "INCONCLUSIVE_EXPERIMENT"
            and fast_sync["selection"]["candidate_selected"] is False
            and fast_sync["selection"]["arbitrary_percentage_hurdle"] is False
            and fast_sync["gates"][
                "full_model_state_within_frozen_tolerance"
            ]
            is True
            and fast_sync["gates"]["candidate_wins_every_pair"] is False
        ),
        "target_window_projection_rejected_by_state_prerequisite": (
            target_window.get("trigger_state")
            == "INCONCLUSIVE_IMPLEMENTATION"
            and target_window["selection"]["candidate_selected"] is False
            and target_window["selection"]["arbitrary_percentage_hurdle"]
            is False
            and target_window["gates"][
                "independent_control_replay_within_frozen_tolerance"
            ]
            is True
            and target_window["gates"][
                "candidate_model_within_frozen_tolerance"
            ]
            is False
        ),
        "asynchronous_checkpoint_lacks_safe_snapshot_trigger": (
            float(
                bounded["local_hot_path"][
                    "checkpoint_fraction_of_device_time_at_32_steps"
                ]
            )
            > 0.0
            and lazy["decision"]["production_route_changed"] is False
        ),
        "custom_kernel_station_closed": (
            all(
                report.get("trigger_state") == "GREEN"
                for report in station_reports.values()
            )
            and station_reports["attention"]["integrity"][
                "uses_mlx_fast_grouped_query_sdpa"
            ]
            is True
            and station_reports["attention"]["integrity"][
                "uses_mlx_fast_rope"
            ]
            is True
            and station_reports["swiglu"]["integrity"][
                "compiled_elementwise_silu_and_gate"
            ]
            is True
        ),
        "production_precision_bound": (
            execution["compute_dtype"] == "float32"
            and execution["fp32_master"] is False
            and execution["token_loss_compute_dtype"] == "float32"
        ),
        "production_compiled_route_bound": (
            pretraining["training_step_mode"] == "compiled"
            and int(pretraining["compiled_microbatch_size"]) == 4
            and int(pretraining["compile_width_quantum"]) == 64
        ),
        "production_optimizer_bound": (
            trainer["training"]["optimizer_id"] == "adamw_mlx"
        ),
        "production_full_depth_bound": (
            "structural_growth" not in trainer["training"]
        ),
        "production_fixed_sequence_bound": (
            "progressive_sequence_length" not in trainer["training"]
        ),
        "ambient_swap_is_diagnostic": (
            trainer["host_resource_safety"]["swapout_growth_action"]
            == "report_only"
            and float(
                trainer["host_resource_safety"][
                    "minimum_available_during_run_mib"
                ]
            )
            >= 2048.0
        ),
    }
    selected = config["selected_recipe"]
    resolved_recipe = {
        "compute_dtype": execution["compute_dtype"],
        "fp32_master": execution["fp32_master"],
        "token_loss_compute_dtype": execution["token_loss_compute_dtype"],
        "training_step_mode": pretraining["training_step_mode"],
        "compiled_microbatch_size": int(
            pretraining["compiled_microbatch_size"]
        ),
        "compile_width_quantum": int(pretraining["compile_width_quantum"]),
        "optimizer_id": trainer["training"]["optimizer_id"],
        "structural_growth_policy": "full_depth_from_step_zero",
        "sequence_length_policy": "fixed_target_width",
        "training_rope_kernel": trainer["training"]["training_rope_kernel"],
        "swapout_growth_action": trainer["host_resource_safety"][
            "swapout_growth_action"
        ],
    }
    gates["selected_recipe_matches_config"] = resolved_recipe == selected
    report = {
        "policy": POLICY,
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "support_state": "SUPPORTED" if all(gates.values()) else "BLOCKED",
        "gates": gates,
        "selected_recipe": resolved_recipe,
        "candidate_dispositions": {
            "fp16_fp32_master": "NOT_SELECTED_REPLAY_PREREQUISITE_FAILED",
            "ademamix_mlx": optimizer["comparisons"]["ademamix_mlx"][
                "disposition"
            ],
            "adam_mini_mlx": optimizer["comparisons"]["adam_mini_mlx"][
                "disposition"
            ],
            "masked_depth_conservative_v1": structural["comparisons"][
                "masked_depth_conservative_v1"
            ]["disposition"],
            "masked_depth_paper_shaped_v1": structural["comparisons"][
                "masked_depth_paper_shaped_v1"
            ]["disposition"],
            "progressive_128_256_512": sequence["comparison"]["disposition"],
            "lazy_inactive_optimizer_state": lazy[
                "implementation_disposition"
            ],
            "fused_linear_cross_entropy": (
                "NOT_IMPLEMENTED_NO_FASTER_MICROBATCH_MEMORY_TRIGGER_AND_"
                "NATIVE_STATION_ALREADY_RANKED"
            ),
            "exact_sequence_packing": (
                "NOT_IMPLEMENTED_CURRENT_LINEAGE_REQUIRES_BLOCK_MASK_"
                "POSITION_AND_SAMPLER_PARITY_AT_94_73_PERCENT_OCCUPANCY"
            ),
            "rust_training_hot_loop": (
                "NOT_IMPLEMENTED_PROFILE_SHOWS_METAL_BOUND_HOST_LOOP_"
                "OUTSIDE_THE_CRITICAL_PATH"
            ),
            "mlx_metal_fast_synch": (
                "NOT_SELECTED_FOURTH_PAIR_REVERSED_AND_CURRENT_PAIR_SWAPPED_"
                "STATE_PARITY_GREEN_NO_PERCENTAGE_HURDLE"
            ),
            "source_target_window_projection": (
                "NOT_SELECTED_FROZEN_MODEL_STATE_TOLERANCE_FAILED_"
                "CONTROL_REPLAY_GREEN"
            ),
            "asynchronous_checkpoint_publication": (
                "NOT_IMPLEMENTED_IMMUTABLE_SNAPSHOT_MEMORY_AND_REPLAY_"
                "TRIGGER_ABSENT"
            ),
            "new_custom_metal_kernel": (
                "NOT_AUTHORIZED_NO_UNFUSED_RESIDUAL_IDENTIFIED_BY_"
                "BOUNDED_STATION_PROFILES"
            ),
            "mlx_sdpa_blocks_on_fp32": "NOT_APPLICABLE_TO_UPSTREAM_FP32_PATH",
            "apple_neural_accelerator_on_m1": "UNAVAILABLE_ON_THIS_HARDWARE",
            "high_power_mode": "UNAVAILABLE_ON_THIS_HARDWARE",
        },
        "bounded_measurements": {
            "fast_sync_speedup_ratios": fast_sync["timing"][
                "matched_pair_speedup_ratios"
            ],
            "fast_sync_geometric_mean_speedup": fast_sync["timing"][
                "geometric_mean_speedup"
            ],
            "fast_sync_minimum_speedup": fast_sync["timing"][
                "minimum_pair_speedup"
            ],
            "target_window_projection_speedup": target_window["timing"][
                "mean_control_over_candidate_speedup"
            ],
            "target_window_projection_maximum_model_delta": (
                target_window["numerical_authority"]["candidate_model"][
                    "maximum_absolute_delta"
                ]
            ),
            "fused_loss_elimination_upper_bound": bounded[
                "linear_cross_entropy_station"
            ]["fused_loss_elimination_upper_bound"],
            "packing_padding_fraction_upper_bound": packing[
                "local_padding_fraction_upper_bound"
            ],
            "rust_host_loop_upper_bound": rust_loop["local_upper_bound"],
            "checkpoint_fraction_at_32_steps": bounded["local_hot_path"][
                "checkpoint_fraction_of_device_time_at_32_steps"
            ],
        },
        "evidence": {
            name: {
                "path": relative(path),
                "sha256": sha256_file(path),
            }
            for name, (path, _report) in reports.items()
        }
        | {
            "production_trainer_config": {
                "path": relative(trainer_path),
                "sha256": sha256_file(trainer_path),
            }
        },
        "campaign_disposition": {
            "kind": "FINITE_ACCELERATION_SELECTOR_RETAIN_CURRENT_ROUTE",
            "launch_recipe_changed_by_challenger": False,
            "launch_recipe": resolved_recipe,
            "next_action": (
                "run the final sustained selected-route qualification, then "
                "return to the matched 57M campaign without another elective "
                "acceleration search"
            ),
        },
        "capability_claim": "NONE_EXECUTION_SELECTION_ONLY",
        "non_claims": [
            "Selection is local M1 MLX engineering evidence, not model capability or cross-hardware superiority.",
            "Candidate exclusions are scoped engineering dispositions, not scientific falsifications.",
        ],
    }
    report_path = resolve(config["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    report = execute(args.config.resolve())
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "selected_recipe": report["selected_recipe"],
                "failed_gates": [
                    name for name, passed in report["gates"].items() if not passed
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] == "GREEN" or not args.gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
