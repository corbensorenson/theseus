from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ane_metal_heterogeneous_planner as planner


def load_config() -> dict:
    return json.loads(
        (ROOT / "configs/ane_metal_heterogeneous_execution.json").read_text(
            encoding="utf-8"
        )
    )


def green_gates() -> dict[str, bool]:
    return {gate: True for gate in planner.REQUIRED_GATES}


def test_config_defaults_disabled_and_covers_full_split_sweep() -> None:
    config = load_config()
    planner.validate_config(config)

    assert config["canonical_enabled"] is False
    assert config["split_ratios"][0] == 0.0
    assert config["split_ratios"][-1] == 1.0
    assert config["fallback"]["timing"] == "before_checkpoint_mutation"
    assert set(config["dynamic_training_qualification_gates"]) == set(
        planner.DYNAMIC_SEMANTIC_GATES
    )
    assert set(config["three_engine_training_qualification_gates"]) == set(
        planner.THREE_ENGINE_TRAINING_GATES
    )
    assert set(config["exact_projection_triad_qualification_gates"]) == set(
        planner.EXACT_PROJECTION_TRIAD_GATES
    )


def test_unstable_compiler_blocks_even_fast_candidate() -> None:
    evidence = {
        "shape_id": "qkv_projection",
        "compiler_attempts": [
            {"compiled": True},
            {"compiled": False},
        ],
        "metal_control_seconds": [10.0, 10.1, 9.9],
        "split_candidates": [
            {
                "ane_output_fraction": 0.5,
                "concurrent_wall_seconds": [7.0, 7.1, 6.9],
                "concurrent_gpu_seconds": [6.8, 6.9, 6.7],
                "concurrent_ane_seconds": [6.5, 6.6, 6.4],
                "gpu_isolation_seconds": [7.0, 7.1, 7.2],
                "ane_isolation_seconds": [6.8, 6.9, 7.0],
                "join_and_fence_seconds": [0.1, 0.1, 0.1],
                "gates": green_gates(),
            }
        ],
    }

    report = planner.plan(load_config(), evidence)

    assert report["compiler"]["state"] == "M1_PRIVATE_COMPILER_UNSTABLE"
    assert report["selected"] is None
    assert report["checkpoint_mutation_authorized"] is False


def test_selection_requires_worst_candidate_to_beat_best_control() -> None:
    base = {
        "shape_id": "gate_up_projection",
        "compiler_attempts": [{"compiled": True}, {"compiled": True}],
        "metal_control_seconds": [10.0, 10.1, 9.9],
    }
    uncertain = {
        **base,
        "split_candidates": [
            {
                "ane_output_fraction": 0.625,
                "concurrent_wall_seconds": [9.7, 9.8, 10.0],
                "concurrent_gpu_seconds": [9.5, 9.6, 9.8],
                "concurrent_ane_seconds": [9.1, 9.2, 9.3],
                "gpu_isolation_seconds": [9.7, 9.8, 9.9],
                "ane_isolation_seconds": [9.3, 9.4, 9.5],
                "join_and_fence_seconds": [0.1, 0.1, 0.1],
                "gates": green_gates(),
            }
        ],
    }
    green = {
        **base,
        "split_candidates": [
            {
                "ane_output_fraction": 0.625,
                "concurrent_wall_seconds": [8.7, 8.8, 8.9],
                "concurrent_gpu_seconds": [8.5, 8.6, 8.7],
                "concurrent_ane_seconds": [8.1, 8.2, 8.3],
                "gpu_isolation_seconds": [8.6, 8.7, 8.8],
                "ane_isolation_seconds": [8.2, 8.3, 8.4],
                "join_and_fence_seconds": [0.1, 0.1, 0.1],
                "gates": green_gates(),
            }
        ],
    }

    assert planner.plan(load_config(), uncertain)["selected"] is None
    selected = planner.plan(load_config(), green)["selected"]
    assert selected is not None
    assert selected["ane_output_fraction"] == pytest.approx(0.625)
    assert selected["conservative_speedup"] > 1.0


def test_missing_integrity_gate_forces_fallback() -> None:
    gates = green_gates()
    gates["gradient_parity"] = False
    evidence = {
        "shape_id": "qkv_projection",
        "compiler_attempts": [{"compiled": True}, {"compiled": True}],
        "metal_control_seconds": [10.0, 10.0],
        "split_candidates": [
            {
                "ane_output_fraction": 0.5,
                "concurrent_wall_seconds": [5.0, 5.1],
                "concurrent_gpu_seconds": [4.8, 4.9],
                "concurrent_ane_seconds": [4.7, 4.8],
                "gpu_isolation_seconds": [4.8, 4.9],
                "ane_isolation_seconds": [4.7, 4.8],
                "join_and_fence_seconds": [0.1, 0.1],
                "gates": gates,
            }
        ],
    }

    report = planner.plan(load_config(), evidence)

    assert report["selected"] is None
    assert "gradient_parity" in report["candidates"][0]["missing_or_failed_gates"]


def test_no_timings_preserves_inconclusive_implementation() -> None:
    report = planner.plan(
        load_config(),
        {
            "shape_id": "qkv_projection",
            "compiler_attempts": [
                {"compiled": True},
                {"compiled": False},
                {"compiled": False},
            ],
        },
    )

    assert report["trigger_state"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert report["selected"] is None
    assert report["canonical_backend_changed"] is False


def test_current_m1_evidence_routes_past_visibility_to_persistent_partition() -> None:
    evidence = json.loads(
        (ROOT / "configs/ane_metal_m1_evidence_2026_07_27.json").read_text(
            encoding="utf-8"
        )
    )

    report = planner.plan(load_config(), evidence)

    assert report["compiler"]["repeatable"] is True
    assert (
        report["same_surface_bridge"]["state"]
        == "GREEN_CONCURRENT_SHARED_READ_VISIBILITY"
    )
    assert "zero_copy_same_surface_visibility" not in report["blockers"]
    assert (
        report["dynamic_training_path"]["compile_once_mutable_weight_transport"]
        is True
    )
    assert "dynamic_or_persistent_training_weight_update_path" not in report["blockers"]
    assert "split_half_rope_parity" not in report["blockers"]
    assert (
        report["dynamic_training_path"]["split_half_rope_probe"]["state"]
        == "GREEN_OPERATOR_PARITY"
    )
    assert "unscaled_residual_parity" in report["blockers"]
    assert "full_vocabulary_softmax_parity" in report["blockers"]
    assert (
        "optional_structure_aligned_partition_after_dynamic_route_disposition"
        in report["blockers"]
    )
    assert report["checkpoint_mutation_authorized"] is False


def test_dynamic_coexistence_is_reported_only_as_unmatched_mechanics_ceiling() -> None:
    evidence = {
        "shape_id": "theseus_width512_decoder_control",
        "compiler_attempts": [{"compiled": True}, {"compiled": True}],
        "dynamic_training_path": {
            "state": "GREEN_TRANSPORT_INCONCLUSIVE_SEMANTICS",
            "compile_once_mutable_weight_transport": True,
            "semantic_gates": {
                gate: False for gate in planner.DYNAMIC_SEMANTIC_GATES
            },
            "process_coexistence": {
                "actual_process_overlap_observed": True,
                "mlx_standalone_positions_per_second": [2820.379, 3285.661],
                "mlx_concurrent_positions_per_second": [2332.88, 2410.994],
                "ane_standalone_positions_per_second": [2298.025, 2588.473],
                "ane_concurrent_positions_per_second": [1968.474, 2150.357],
            },
        },
    }

    report = planner.plan(load_config(), evidence)
    dynamic = report["dynamic_training_path"]

    assert dynamic["production_eligible"] is False
    assert (
        dynamic["coexistence"]["unmatched_combined_vs_mlx_mechanics_ceiling"]
        == pytest.approx(1.451465, rel=1e-5)
    )
    assert "not an end-to-end training speedup" in dynamic["coexistence"]["claim_scope"]


def test_dynamic_route_cannot_select_without_every_semantic_gate() -> None:
    semantic_gates = {gate: True for gate in planner.DYNAMIC_SEMANTIC_GATES}
    semantic_gates["pointer_generator_parity"] = False
    evidence = {
        "shape_id": "theseus_width512_decoder_control",
        "compiler_attempts": [{"compiled": True}, {"compiled": True}],
        "dynamic_training_path": {
            "state": "GREEN_TRANSPORT_INCONCLUSIVE_SEMANTICS",
            "compile_once_mutable_weight_transport": True,
            "semantic_gates": semantic_gates,
            "process_coexistence": {
                "actual_process_overlap_observed": True,
                "mlx_standalone_positions_per_second": [3000.0, 3000.0],
                "mlx_concurrent_positions_per_second": [2400.0, 2400.0],
                "ane_standalone_positions_per_second": [2500.0, 2500.0],
                "ane_concurrent_positions_per_second": [2100.0, 2100.0],
            },
        },
    }

    report = planner.plan(load_config(), evidence)

    assert report["dynamic_training_path"]["production_eligible"] is False
    assert report["dynamic_training_path"]["failed_semantic_gates"] == [
        "pointer_generator_parity"
    ]
    assert "pointer_generator_parity" in report["blockers"]


def test_checked_in_rope_probe_and_receipt_bind_the_green_operator_gate() -> None:
    source = ROOT / "native/ane_metal/ane_split_half_rope_probe.m"
    receipt = json.loads(
        (ROOT / "reports/ane_split_half_rope_m1.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (ROOT / "configs/ane_metal_m1_evidence_2026_07_27.json").read_text(
            encoding="utf-8"
        )
    )
    dynamic = evidence["dynamic_training_path"]

    assert source.is_file()
    assert "#define HEADS 8" in source.read_text(encoding="utf-8")
    assert "#define SEQUENCE 512" in source.read_text(encoding="utf-8")
    assert receipt["trigger_state"] == "GREEN_OPERATOR_PARITY"
    assert receipt["mismatch_count"] == 0
    assert receipt["maximum_absolute_delta"] <= receipt["tolerance"]
    assert dynamic["semantic_gates"]["split_half_rope_parity"] is True
    assert dynamic["split_half_rope_probe"]["report"] == (
        "reports/ane_split_half_rope_m1.json"
    )


def test_current_m1_evidence_keeps_three_engine_route_mechanical_only() -> None:
    evidence = json.loads(
        (ROOT / "configs/ane_metal_m1_evidence_2026_07_27.json").read_text(
            encoding="utf-8"
        )
    )

    report = planner.plan(load_config(), evidence)
    triad = report["three_engine_scheduling"]

    assert triad["state_transport_green"] is True
    assert triad["cpu_weight_gradient_operator_green"] is True
    assert triad["mechanical_overlap_green"] is True
    assert triad["ane_projection_wall_ratio_over_mlx"] > 1.0
    assert triad["isolated_ane_projection_faster_than_mlx"] is False
    assert triad["always_on_three_engine_policy_selected"] is False
    assert triad["worst_concurrent_kernel_slowdown"] > 1.0
    assert triad["production_eligible"] is False
    assert "exact_ane_forward_parity" not in triad["failed_training_gates"]
    assert "exact_ane_input_gradient_parity" not in triad["failed_training_gates"]
    assert "one_authoritative_fp32_update" not in triad["failed_training_gates"]
    assert "exact_cpu_weight_gradient_parity" not in triad["failed_training_gates"]
    assert (
        "no_intermediate_python_or_numpy_round_trip"
        in triad["failed_training_gates"]
    )
    assert (
        "matched_joined_wall_gain_exceeds_uncertainty"
        in triad["failed_training_gates"]
    )


def test_three_engine_evidence_matches_bound_operator_receipts() -> None:
    evidence = json.loads(
        (ROOT / "configs/ane_metal_m1_evidence_2026_07_27.json").read_text(
            encoding="utf-8"
        )
    )
    coreml = json.loads(
        (ROOT / "reports/coreml_state_weight_m1.json").read_text(
            encoding="utf-8"
        )
    )
    mlx = json.loads(
        (ROOT / "reports/mlx_fp16_projection_control_m1.json").read_text(
            encoding="utf-8"
        )
    )
    coexistence = json.loads(
        (ROOT / "reports/cpu_gpu_ane_coexistence_m1.json").read_text(
            encoding="utf-8"
        )
    )

    public_state = evidence["public_coreml_state_weight_transport"]
    assert coreml["trigger_state"] == public_state["state"]
    assert coreml["runtime"]["mean_milliseconds"] == pytest.approx(
        public_state["mean_milliseconds"]
    )
    assert coreml["resource_custody"]["temporary_model_removed"] is True
    assert mlx["runtime"]["mean_milliseconds"] == pytest.approx(
        evidence["mlx_projection_control"]["mean_milliseconds"]
    )
    triad = evidence["three_engine_coexistence"]
    assert coexistence["trigger_state"] == triad["state"]
    assert coexistence["overlap_speedup_vs_serial_sum"] == pytest.approx(
        triad["overlap_speedup_vs_serial_sum"]
    )
    assert {
        label: coexistence["workers"][label]["kernel_slowdown"]
        for label in ("cpu", "gpu", "ane")
    } == pytest.approx(triad["worker_kernel_slowdowns"])


def test_exact_public_projection_triad_is_narrowly_rejected_and_redirected() -> None:
    evidence = json.loads(
        (ROOT / "configs/ane_metal_m1_evidence_2026_07_27.json").read_text(
            encoding="utf-8"
        )
    )
    hardware = json.loads(
        (ROOT / "reports/ane_cpu_metal_projection_triad_m1.json").read_text(
            encoding="utf-8"
        )
    )
    resource = json.loads(
        (
            ROOT
            / "reports/ane_cpu_metal_projection_triad_m1.host_resource_safety.json"
        ).read_text(encoding="utf-8")
    )

    report = planner.plan(load_config(), evidence)
    triad = report["exact_projection_triad"]

    assert triad["mechanics_green"] is True
    assert triad["public_bridge_selected"] is False
    assert triad["private_zero_copy_triad_is_immediate_next"] is False
    assert triad["mean_speedup_control_over_hybrid"] == pytest.approx(
        hardware["timing"]["mean_speedup_control_over_hybrid"]
    )
    assert triad["hybrid_wall_ratio_over_mlx"] > 2.0
    assert triad["resource_receipt"]["maximum_swapout_growth_mib"] == 0.0
    assert resource["maximum_swapout_growth_mib"] == 0.0
    assert resource["passed"] is True
    assert "matched_joined_wall_gain_exceeds_uncertainty" in triad["failed_gates"]
    assert "no_intermediate_python_or_numpy_round_trip" in triad["failed_gates"]


def test_native_zero_copy_triad_rejects_direct_offload() -> None:
    evidence = json.loads(
        (ROOT / "configs/ane_metal_m1_evidence_2026_07_27.json").read_text(
            encoding="utf-8"
        )
    )
    report = planner.plan(load_config(), evidence)
    native = report["native_zero_copy_projection_triad"]

    assert native["parity_and_mechanics_green"] is True
    assert native["direct_q_proj_selected"] is False
    assert native["activation_recomputation_is_immediate_next"] is False
    assert native["mean_speedup_mlx_over_native"] < 1.0
    assert native["conservative_speedup_mlx_over_native"] < 1.0
    assert native["resource_receipt"]["maximum_swapout_growth_mib"] == 0.0
    assert native["production_eligible"] is False


def test_recomputation_result_is_closed_by_whole_microbatch_station() -> None:
    evidence = json.loads(
        (ROOT / "configs/ane_metal_m1_evidence_2026_07_27.json").read_text(
            encoding="utf-8"
        )
    )
    report = planner.plan(load_config(), evidence)
    recompute = report["ane_activation_recomputation"]

    assert recompute["mechanics_green"] is True
    assert recompute["schedule_selected"] is False
    assert recompute["whole_microbatch_is_immediate_next"] is False
    assert recompute["mean_speedup_mlx_over_candidate"] < 1.0
    assert recompute["conservative_speedup_mlx_over_candidate"] < 1.0
    assert recompute["resource_receipt"]["maximum_swapout_growth_mib"] == 0.0
    assert recompute["production_eligible"] is False


def test_whole_microbatch_station_advances_to_exact_decoder_block() -> None:
    evidence = json.loads(
        (ROOT / "configs/ane_metal_m1_evidence_2026_07_27.json").read_text(
            encoding="utf-8"
        )
    )
    report = planner.plan(load_config(), evidence)
    station = report["heterogeneous_microbatch_projection"]

    assert station["station_authority_green"] is True
    assert station["full_model_ready"] is False
    assert station["exact_ane_decoder_block_is_immediate_next"] is True
    assert station["mean_speedup_control_over_candidate"] > 1.0
    assert station["conservative_speedup_control_over_candidate"] > 1.0
    assert station["authority"]["local_optimizer_steps"] == 0
    assert station["resource_receipt"]["maximum_swapout_growth_mib"] == 0.0
    assert station["production_eligible"] is False
