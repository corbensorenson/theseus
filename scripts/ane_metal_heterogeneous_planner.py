#!/usr/bin/env python3
"""Plan fail-closed ANE+Metal output-channel partition experiments.

This is deliberately not a device backend.  It converts repeated per-shape
measurements into a split decision while refusing to select a route unless the
worst measured candidate bound beats the best measured Metal-control bound and
every integrity gate is true.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


POLICY = "project_theseus_ane_metal_heterogeneous_plan_v1"
REQUIRED_GATES = (
    "compiler_repeatable",
    "output_parity",
    "loss_parity",
    "gradient_parity",
    "save_reload_replay",
    "no_host_round_trip",
    "zero_swap_growth",
    "resource_reserve_preserved",
    "thermal_sustainability",
    "independent_gate_audit",
)

DYNAMIC_SEMANTIC_GATES = (
    "gqa_forward_backward_parity",
    "split_half_rope_parity",
    "unscaled_residual_parity",
    "full_vocabulary_softmax_parity",
    "loss_mask_and_objective_mass_parity",
    "source_encoder_parity",
    "decoder_cross_attention_parity",
    "pointer_generator_parity",
    "auxiliary_objective_parity",
    "fp32_master_loss_scaling_stability",
    "single_synchronized_optimizer_update",
    "sampler_partition_conservation",
    "save_reload_replay",
    "production_batch_shape",
    "sustained_resource_and_thermal_qualification",
    "matched_joined_wall_gain_exceeds_uncertainty",
    "independent_gate_audit",
)

THREE_ENGINE_TRAINING_GATES = (
    "exact_ane_forward_parity",
    "exact_ane_input_gradient_parity",
    "exact_cpu_weight_gradient_parity",
    "one_authoritative_fp32_update",
    "sampler_and_objective_mass_conservation",
    "no_intermediate_python_or_numpy_round_trip",
    "save_reload_replay",
    "sixty_four_step_stability",
    "matched_joined_wall_gain_exceeds_uncertainty",
    "sustained_resource_and_thermal_qualification",
    "independent_gate_audit",
)

EXACT_PROJECTION_TRIAD_GATES = (
    "forward_and_dx_compute_plan_prefers_ane",
    "output_parity",
    "loss_parity",
    "input_gradient_parity",
    "weight_gradient_parity",
    "updated_weight_parity",
    "single_generation_conservation",
    "single_fp32_update",
    "save_reload_exact",
    "replay_exact",
    "sixty_four_step_finite",
    "matched_joined_wall_gain_exceeds_uncertainty",
    "zero_swap_growth",
    "thermal_sustainability",
    "no_intermediate_python_or_numpy_round_trip",
    "independent_gate_audit",
)


class PlanningFault(ValueError):
    """Raised when an experiment packet cannot support a valid decision."""


def _finite_positive(values: Iterable[float], label: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) < 2:
        raise PlanningFault(f"{label}_requires_at_least_two_repeats")
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise PlanningFault(f"{label}_must_be_finite_and_positive")
    return result


def timing_interval(values: Iterable[float]) -> dict[str, float | int]:
    """Return a conservative repeatability interval without distribution claims."""

    samples = _finite_positive(values, "timing")
    return {
        "count": len(samples),
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "maximum_seconds": max(samples),
        "mean_seconds": statistics.fmean(samples),
    }


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != "project_theseus_ane_metal_heterogeneous_execution_v1":
        raise PlanningFault("unexpected_config_policy")
    ratios = config.get("split_ratios")
    if not isinstance(ratios, list) or ratios != sorted(set(ratios)):
        raise PlanningFault("split_ratios_must_be_sorted_and_unique")
    if not ratios or ratios[0] != 0.0 or ratios[-1] != 1.0:
        raise PlanningFault("split_ratios_must_include_zero_and_one")
    if any(not 0.0 <= float(ratio) <= 1.0 for ratio in ratios):
        raise PlanningFault("split_ratio_out_of_range")
    if config.get("canonical_enabled") is not False:
        raise PlanningFault("experimental_backend_must_default_disabled")


def _positive_mean(values: Any, label: str) -> float | None:
    if values is None:
        return None
    samples = _finite_positive(values, label)
    return statistics.fmean(samples)


def _dynamic_training_record(evidence: dict[str, Any]) -> dict[str, Any]:
    dynamic = evidence.get("dynamic_training_path") or {}
    semantic_gates = dynamic.get("semantic_gates") or {}
    failed_semantic_gates = [
        gate for gate in DYNAMIC_SEMANTIC_GATES if semantic_gates.get(gate) is not True
    ]
    coexistence = dynamic.get("process_coexistence") or {}
    mlx_standalone = _positive_mean(
        coexistence.get("mlx_standalone_positions_per_second"),
        "dynamic_mlx_standalone_positions_per_second",
    )
    mlx_concurrent = _positive_mean(
        coexistence.get("mlx_concurrent_positions_per_second"),
        "dynamic_mlx_concurrent_positions_per_second",
    )
    ane_standalone = _positive_mean(
        coexistence.get("ane_standalone_positions_per_second"),
        "dynamic_ane_standalone_positions_per_second",
    )
    ane_concurrent = _positive_mean(
        coexistence.get("ane_concurrent_positions_per_second"),
        "dynamic_ane_concurrent_positions_per_second",
    )
    mechanics_ceiling = None
    combined_concurrent = None
    if mlx_concurrent is not None and ane_concurrent is not None:
        combined_concurrent = mlx_concurrent + ane_concurrent
    if combined_concurrent is not None and mlx_standalone is not None:
        mechanics_ceiling = combined_concurrent / mlx_standalone
    return {
        "state": dynamic.get("state", "NOT_MEASURED"),
        "compile_once_mutable_weight_transport": (
            dynamic.get("compile_once_mutable_weight_transport") is True
        ),
        "gqa_grouping_probe": dynamic.get("gqa_grouping_probe"),
        "split_half_rope_probe": dynamic.get("split_half_rope_probe"),
        "semantic_gates": semantic_gates,
        "failed_semantic_gates": failed_semantic_gates,
        "coexistence": {
            "actual_process_overlap_observed": (
                coexistence.get("actual_process_overlap_observed") is True
            ),
            "mlx_standalone_mean_positions_per_second": mlx_standalone,
            "mlx_concurrent_mean_positions_per_second": mlx_concurrent,
            "ane_standalone_mean_positions_per_second": ane_standalone,
            "ane_concurrent_mean_positions_per_second": ane_concurrent,
            "combined_concurrent_mechanics_positions_per_second": combined_concurrent,
            "unmatched_combined_vs_mlx_mechanics_ceiling": mechanics_ceiling,
            "claim_scope": (
                "The workloads are not yet semantically or computationally matched. "
                "This is a coexistence ceiling, not an end-to-end training speedup."
            ),
        },
        "production_eligible": (
            dynamic.get("compile_once_mutable_weight_transport") is True
            and not failed_semantic_gates
            and coexistence.get("actual_process_overlap_observed") is True
        ),
    }


def _finite_measurement(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    measured = float(value)
    if not math.isfinite(measured) or measured <= 0:
        raise PlanningFault(f"{key}_must_be_finite_and_positive")
    return measured


def _three_engine_record(evidence: dict[str, Any]) -> dict[str, Any]:
    public_state = evidence.get("public_coreml_state_weight_transport") or {}
    mlx_control = evidence.get("mlx_projection_control") or {}
    cpu_dw = evidence.get("cpu_accelerate_dw") or {}
    coexistence = evidence.get("three_engine_coexistence") or {}
    gates = evidence.get("three_engine_training_gates") or {}
    failed_gates = [
        gate for gate in THREE_ENGINE_TRAINING_GATES if gates.get(gate) is not True
    ]

    ane_mean = _finite_measurement(public_state, "mean_milliseconds")
    mlx_mean = _finite_measurement(mlx_control, "mean_milliseconds")
    cpu_dw_mean = _finite_measurement(cpu_dw, "mean_milliseconds")
    overlap_speedup = _finite_measurement(
        coexistence, "overlap_speedup_vs_serial_sum"
    )
    ane_over_mlx = (
        ane_mean / mlx_mean
        if ane_mean is not None and mlx_mean is not None
        else None
    )
    worker_slowdowns = coexistence.get("worker_kernel_slowdowns") or {}
    finite_slowdowns: dict[str, float] = {}
    for label, raw_value in worker_slowdowns.items():
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0:
            raise PlanningFault(f"{label}_kernel_slowdown_must_be_positive")
        finite_slowdowns[str(label)] = value
    worst_slowdown = max(finite_slowdowns.values(), default=None)

    state_transport_green = (
        public_state.get("state") == "GREEN_PUBLIC_ANE_STATE_WEIGHT_TRANSPORT"
        and public_state.get("state_update_visible_without_recompile") is True
        and public_state.get("matmul_prefers_ane") is True
        and int(public_state.get("output_mismatch_count", -1)) == 0
        and ane_mean is not None
        and mlx_mean is not None
    )
    cpu_dw_green = (
        cpu_dw.get("state") == "GREEN"
        and int(cpu_dw.get("mismatch_count", -1)) == 0
        and cpu_dw_mean is not None
    )
    mechanical_overlap_green = (
        coexistence.get("state") == "GREEN_THREE_ENGINE_MECHANICAL_OVERLAP"
        and coexistence.get("actual_overlap_observed") is True
        and overlap_speedup is not None
        and overlap_speedup > 1.0
        and set(finite_slowdowns) == {"cpu", "gpu", "ane"}
    )
    production_eligible = (
        state_transport_green
        and cpu_dw_green
        and mechanical_overlap_green
        and not failed_gates
    )
    return {
        "state_transport_green": state_transport_green,
        "cpu_weight_gradient_operator_green": cpu_dw_green,
        "mechanical_overlap_green": mechanical_overlap_green,
        "public_state_weight": public_state,
        "mlx_projection_control": mlx_control,
        "cpu_accelerate_weight_gradient": cpu_dw,
        "coexistence": coexistence,
        "ane_projection_wall_ratio_over_mlx": ane_over_mlx,
        "cpu_weight_gradient_mean_milliseconds": cpu_dw_mean,
        "isolated_ane_projection_faster_than_mlx": (
            ane_over_mlx < 1.0 if ane_over_mlx is not None else None
        ),
        "worst_concurrent_kernel_slowdown": worst_slowdown,
        "always_on_three_engine_policy_selected": False,
        "scheduler_policy": (
            "Use measured critical-path list scheduling with one weight generation. "
            "A device receives work only when joined wall improves; do not sum unmatched "
            "operator rates or force all engines on."
        ),
        "preferred_training_probe": (
            "ANE forward/input-gradient, single-thread Accelerate FP32 weight-gradient, "
            "and MLX/Metal attention, loss, pointer, reduction, and optimizer work joined "
            "into one sampler-exact FP32 update."
        ),
        "training_gates": gates,
        "failed_training_gates": failed_gates,
        "production_eligible": production_eligible,
        "claim_scope": (
            "Public state transport, isolated operator parity, and three-process "
            "coexistence mechanics only. This is not an end-to-end training speedup."
        ),
    }


def _exact_projection_triad_record(evidence: dict[str, Any]) -> dict[str, Any]:
    triad = evidence.get("exact_projection_triad") or {}
    gates = triad.get("gates") or {}
    failed_gates = [
        gate for gate in EXACT_PROJECTION_TRIAD_GATES if gates.get(gate) is not True
    ]
    hybrid_mean = _finite_measurement(triad, "hybrid_mean_milliseconds")
    control_mean = _finite_measurement(triad, "mlx_control_mean_milliseconds")
    measured_speedup = _finite_measurement(
        triad, "mean_speedup_control_over_hybrid"
    )
    if hybrid_mean is not None and control_mean is not None:
        derived_speedup = control_mean / hybrid_mean
        if measured_speedup is None:
            measured_speedup = derived_speedup
        elif not math.isclose(
            measured_speedup, derived_speedup, rel_tol=1e-6, abs_tol=1e-9
        ):
            raise PlanningFault("exact_projection_triad_speedup_inconsistent")
    mechanics_green = all(
        gates.get(gate) is True
        for gate in (
            "forward_and_dx_compute_plan_prefers_ane",
            "output_parity",
            "loss_parity",
            "input_gradient_parity",
            "weight_gradient_parity",
            "updated_weight_parity",
            "single_generation_conservation",
            "single_fp32_update",
            "save_reload_exact",
            "replay_exact",
            "sixty_four_step_finite",
            "zero_swap_growth",
        )
    )
    public_bridge_selected = (
        mechanics_green
        and not failed_gates
        and measured_speedup is not None
        and measured_speedup > 1.0
    )
    return {
        "state": triad.get("state", "NOT_MEASURED"),
        "disposition": triad.get("disposition", "NOT_MEASURED"),
        "shape": triad.get("shape"),
        "hybrid_mean_milliseconds": hybrid_mean,
        "mlx_control_mean_milliseconds": control_mean,
        "mean_speedup_control_over_hybrid": measured_speedup,
        "hybrid_wall_ratio_over_mlx": (
            hybrid_mean / control_mean
            if hybrid_mean is not None and control_mean is not None
            else None
        ),
        "maximum_absolute_delta_by_station": triad.get(
            "maximum_absolute_delta_by_station"
        ),
        "resource_receipt": triad.get("resource_receipt"),
        "gates": gates,
        "failed_gates": failed_gates,
        "mechanics_green": mechanics_green,
        "public_bridge_selected": public_bridge_selected,
        "private_zero_copy_triad_is_immediate_next": (
            mechanics_green
            and not public_bridge_selected
            and not bool(evidence.get("native_zero_copy_projection_triad"))
            and gates.get("matched_joined_wall_gain_exceeds_uncertainty") is False
            and gates.get("no_intermediate_python_or_numpy_round_trip") is False
        ),
        "production_eligible": public_bridge_selected,
        "claim_scope": (
            "One deterministic q_proj-shaped optimizer transaction only. A losing "
            "public Python/NumPy/Core ML bridge rejects that exact bridge, not ANE "
            "training or a native IOSurface implementation."
        ),
    }


def _native_zero_copy_projection_triad_record(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    triad = evidence.get("native_zero_copy_projection_triad") or {}
    native_means = [
        float(value)
        for value in triad.get("native_round_mean_milliseconds") or []
    ]
    mlx_means = [
        float(value)
        for value in triad.get("mlx_round_mean_milliseconds") or []
    ]
    if triad and (
        len(native_means) < 2
        or len(mlx_means) < 2
        or any(
            not math.isfinite(value) or value <= 0
            for value in native_means + mlx_means
        )
    ):
        raise PlanningFault("native_zero_copy_round_timings_invalid")
    mean_speedup = (
        statistics.fmean(mlx_means) / statistics.fmean(native_means)
        if native_means and mlx_means
        else None
    )
    conservative_speedup = (
        min(mlx_means) / max(native_means)
        if native_means and mlx_means
        else None
    )
    recorded_mean = _finite_measurement(
        triad, "mean_speedup_mlx_over_native"
    )
    recorded_conservative = _finite_measurement(
        triad, "conservative_speedup_mlx_over_native"
    )
    if mean_speedup is not None and (
        not math.isclose(mean_speedup, recorded_mean or -1, rel_tol=1e-6)
        or not math.isclose(
            conservative_speedup,
            recorded_conservative or -1,
            rel_tol=1e-6,
        )
    ):
        raise PlanningFault("native_zero_copy_speedup_inconsistent")
    gates = triad.get("gates") or {}
    custody = triad.get("custody") or {}
    parity_mechanics_green = (
        gates.get("full_station_parity") is True
        and gates.get("native_mechanics") is True
        and gates.get("resource_safety") is True
        and gates.get("zero_swap_growth") is True
        and custody.get("single_generation_conserved") is True
        and custody.get("one_fp32_gradient_accumulator") is True
        and custody.get("one_fp32_adamw_update_per_step") is True
        and custody.get("hot_step_python_or_numpy") is False
        and custody.get("intermediate_host_tensor_copy") is False
    )
    selected = (
        parity_mechanics_green
        and gates.get("matched_joined_wall_gain_exceeds_uncertainty") is True
        and conservative_speedup is not None
        and conservative_speedup > 1.0
    )
    return {
        "state": triad.get("state", "NOT_MEASURED"),
        "disposition": triad.get("disposition", "NOT_MEASURED"),
        "shape": triad.get("shape"),
        "native_round_mean_milliseconds": native_means,
        "mlx_round_mean_milliseconds": mlx_means,
        "mean_speedup_mlx_over_native": mean_speedup,
        "conservative_speedup_mlx_over_native": conservative_speedup,
        "parity_and_mechanics_green": parity_mechanics_green,
        "direct_q_proj_selected": selected,
        "activation_recomputation_is_immediate_next": (
            parity_mechanics_green
            and not selected
            and not bool(evidence.get("ane_activation_recomputation"))
        ),
        "resource_receipt": triad.get("resource_receipt"),
        "gates": gates,
        "production_eligible": False,
        "claim_scope": (
            "Direct q_proj offload at one exact shape only. A wall-time loss "
            "does not falsify ANE recomputation, whole-microbatch work, "
            "campaign concurrency, or inference."
        ),
    }


def _ane_activation_recomputation_record(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    record = evidence.get("ane_activation_recomputation") or {}
    controls = [
        float(value)
        for value in record.get("mlx_control_milliseconds_per_iteration") or []
    ]
    candidates = [
        float(value)
        for value in record.get(
            "concurrent_critical_path_milliseconds_per_iteration"
        )
        or []
    ]
    native = [
        float(value)
        for value in record.get(
            "native_recompute_milliseconds_per_iteration"
        )
        or []
    ]
    values = controls + candidates + native
    if record and (
        len(controls) < 2
        or len(candidates) < 2
        or len(native) < 2
        or any(not math.isfinite(value) or value <= 0 for value in values)
    ):
        raise PlanningFault("ane_recomputation_timings_invalid")
    mean_speedup = (
        statistics.fmean(controls) / statistics.fmean(candidates)
        if controls and candidates
        else None
    )
    conservative_speedup = (
        min(controls) / max(candidates)
        if controls and candidates
        else None
    )
    recorded_mean = _finite_measurement(
        record, "mean_speedup_mlx_over_candidate"
    )
    recorded_conservative = _finite_measurement(
        record, "conservative_speedup_mlx_over_candidate"
    )
    if mean_speedup is not None and (
        not math.isclose(mean_speedup, recorded_mean or -1, rel_tol=1e-6)
        or not math.isclose(
            conservative_speedup,
            recorded_conservative or -1,
            rel_tol=1e-6,
        )
    ):
        raise PlanningFault("ane_recomputation_speedup_inconsistent")
    gates = record.get("gates") or {}
    mechanics_green = (
        gates.get("native_mechanics") is True
        and gates.get(
            "ane_recompute_hidden_inside_independent_metal_window"
        )
        is True
        and gates.get("resource_safety") is True
        and gates.get("zero_swap_growth") is True
        and int(record.get("parity_mismatch_count_above_0_001", -1)) == 0
    )
    selected = (
        mechanics_green
        and gates.get("matched_joined_wall_gain_exceeds_uncertainty") is True
        and conservative_speedup is not None
        and conservative_speedup > 1.0
    )
    return {
        "state": record.get("state", "NOT_MEASURED"),
        "disposition": record.get("disposition", "NOT_MEASURED"),
        "shape": record.get("shape"),
        "native_recompute_milliseconds_per_iteration": native,
        "mlx_control_milliseconds_per_iteration": controls,
        "concurrent_critical_path_milliseconds_per_iteration": candidates,
        "mean_speedup_mlx_over_candidate": mean_speedup,
        "conservative_speedup_mlx_over_candidate": conservative_speedup,
        "mechanics_green": mechanics_green,
        "schedule_selected": selected,
        "whole_microbatch_is_immediate_next": mechanics_green and not selected,
        "memory": record.get("memory"),
        "resource_receipt": record.get("resource_receipt"),
        "gates": gates,
        "production_eligible": False,
        "claim_scope": (
            "One exact SwiGLU gate/up recomputation schedule against one "
            "independent attention-backward window. A joined-wall loss does "
            "not falsify other recomputation schedules, whole-microbatch "
            "parallelism, matched-arm concurrency, or inference."
        ),
    }


def _candidate_record(
    record: dict[str, Any], control: dict[str, float | int]
) -> dict[str, Any]:
    ratio = float(record["ane_output_fraction"])
    if not 0.0 < ratio < 1.0:
        raise PlanningFault("candidate_ratio_must_be_strictly_between_zero_and_one")
    concurrent = timing_interval(record["concurrent_wall_seconds"])
    gpu_latency = timing_interval(record["concurrent_gpu_seconds"])
    ane_latency = timing_interval(record["concurrent_ane_seconds"])
    gpu_isolation = timing_interval(record["gpu_isolation_seconds"])
    ane_isolation = timing_interval(record["ane_isolation_seconds"])
    overhead = timing_interval(record["join_and_fence_seconds"])

    control_best = float(control["minimum_seconds"])
    control_worst = float(control["maximum_seconds"])
    candidate_best = float(concurrent["minimum_seconds"])
    candidate_worst = float(concurrent["maximum_seconds"])
    optimistic_speedup = control_worst / candidate_best
    conservative_speedup = control_best / candidate_worst

    actual_overlap_observed = (
        float(concurrent["maximum_seconds"])
        < float(gpu_isolation["minimum_seconds"])
        + float(ane_isolation["minimum_seconds"])
    )
    gpu_latency_not_regressed = (
        float(gpu_latency["maximum_seconds"])
        <= float(gpu_isolation["maximum_seconds"])
    )
    gates = record.get("gates", {})
    missing = [gate for gate in REQUIRED_GATES if gates.get(gate) is not True]
    if not actual_overlap_observed:
        missing.append("actual_overlap_proven")
    if not gpu_latency_not_regressed:
        missing.append("gpu_latency_not_regressed_beyond_uncertainty")
    eligible = not missing and conservative_speedup > 1.0
    return {
        "ane_output_fraction": ratio,
        "metal_output_fraction": 1.0 - ratio,
        "concurrent_wall": concurrent,
        "concurrent_gpu": gpu_latency,
        "concurrent_ane": ane_latency,
        "gpu_isolation": gpu_isolation,
        "ane_isolation": ane_isolation,
        "join_and_fence": overhead,
        "actual_overlap_observed": actual_overlap_observed,
        "gpu_latency_not_regressed_beyond_uncertainty": gpu_latency_not_regressed,
        "optimistic_speedup": optimistic_speedup,
        "conservative_speedup": conservative_speedup,
        "gain_exceeds_observed_timing_uncertainty": conservative_speedup > 1.0,
        "missing_or_failed_gates": missing,
        "eligible": eligible,
    }


def plan(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    compiler_attempts = evidence.get("compiler_attempts", [])
    compiler_successes = sum(
        1 for attempt in compiler_attempts if attempt.get("compiled") is True
    )
    compiler_failures = sum(
        1 for attempt in compiler_attempts if attempt.get("compiled") is False
    )
    compiler_repeatable = (
        len(compiler_attempts) >= 2 and compiler_failures == 0
    )
    incompatible_mil_attempts = evidence.get("incompatible_mil_attempts", [])
    dynamic_training = _dynamic_training_record(evidence)
    three_engine = _three_engine_record(evidence)
    exact_projection_triad = _exact_projection_triad_record(evidence)
    native_zero_copy_projection_triad = (
        _native_zero_copy_projection_triad_record(evidence)
    )
    ane_activation_recomputation = (
        _ane_activation_recomputation_record(evidence)
    )

    report: dict[str, Any] = {
        "policy": POLICY,
        "config_policy": config["policy"],
        "shape_id": evidence.get("shape_id"),
        "compiler": {
            "attempt_count": len(compiler_attempts),
            "success_count": compiler_successes,
            "failure_count": compiler_failures,
            "repeatable": compiler_repeatable,
            "state": (
                "M1_STATIC_AND_DYNAMIC_SHAPED_COMPILER_REPEATABLE"
                if compiler_repeatable
                and dynamic_training["compile_once_mutable_weight_transport"]
                else "M1_STATIC_BAKED_COMPILER_REPEATABLE"
                if compiler_repeatable
                else "M1_PRIVATE_COMPILER_UNSTABLE"
                if compiler_successes and compiler_failures
                else "NOT_QUALIFIED"
            ),
        },
        "incompatible_mil_attempts": incompatible_mil_attempts,
        "weight_update_path": evidence.get("weight_update_path"),
        "dynamic_training_path": dynamic_training,
        "three_engine_scheduling": three_engine,
        "exact_projection_triad": exact_projection_triad,
        "native_zero_copy_projection_triad": (
            native_zero_copy_projection_triad
        ),
        "ane_activation_recomputation": ane_activation_recomputation,
        "same_surface_bridge": evidence.get("same_surface_bridge"),
        "canonical_backend_changed": False,
        "checkpoint_mutation_authorized": False,
        "public_benchmark_rows_read": 0,
        "external_inference_calls": 0,
    }

    metal_samples = evidence.get("metal_control_seconds", [])
    candidates = evidence.get("split_candidates", [])
    if not metal_samples or not candidates:
        split_state = evidence.get("split_linear_station", {}).get("state")
        dynamic_transport_green = (
            dynamic_training["compile_once_mutable_weight_transport"] is True
        )
        blockers = []
        if dynamic_transport_green:
            blockers.extend(dynamic_training["failed_semantic_gates"])
            blockers.append("mlx_lazy_graph_or_native_metal_gradient_integration")
        else:
            blockers.extend(
                [
                    (
                        "structure_aligned_persistent_partition_candidate"
                        if split_state == "MECHANICS_GREEN_PHYSICAL_JOIN_NOT_SELECTED"
                        else "measured_metal_control_and_joint_candidates"
                    ),
                    "mlx_lazy_graph_integration",
                ]
            )
        if evidence.get("same_surface_bridge", {}).get("state") != (
            "GREEN_CONCURRENT_SHARED_READ_VISIBILITY"
        ):
            blockers.insert(1, "zero_copy_same_surface_visibility")
        if not compiler_repeatable:
            blockers.insert(0, "repeatable_private_ane_compile")
        if (
            evidence.get("weight_update_path", {}).get("state") != "QUALIFIED"
            and not dynamic_transport_green
        ):
            blockers.append("dynamic_or_persistent_training_weight_update_path")
        if dynamic_transport_green and split_state == (
            "MECHANICS_GREEN_PHYSICAL_JOIN_NOT_SELECTED"
        ):
            blockers.append(
                "optional_structure_aligned_partition_after_dynamic_route_disposition"
            )
        if three_engine["state_transport_green"]:
            blockers.extend(
                gate
                for gate in three_engine["failed_training_gates"]
                if gate not in blockers
            )
        report.update(
            {
                "trigger_state": "INCONCLUSIVE_IMPLEMENTATION",
                "control": None,
                "candidates": [],
                "selected": None,
                "blockers": blockers,
            }
        )
        return report

    control = timing_interval(metal_samples)
    candidate_reports = [
        _candidate_record(record, control) for record in candidates
    ]
    eligible = [record for record in candidate_reports if record["eligible"]]
    selected = (
        max(eligible, key=lambda record: record["conservative_speedup"])
        if compiler_repeatable and eligible
        else None
    )
    report.update(
        {
            "trigger_state": (
                "GREEN_EXPERIMENTAL_SPLIT_SELECTED"
                if selected is not None
                else "RED_FALLBACK_TO_QUALIFIED_MLX_METAL"
            ),
            "control": control,
            "candidates": candidate_reports,
            "selected": selected,
            "blockers": (
                []
                if selected is not None
                else [
                    "repeatable_private_ane_compile"
                    if not compiler_repeatable
                    else "no_candidate_clears_integrity_and_uncertainty"
                ]
            ),
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ane_metal_heterogeneous_execution.json"),
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    report = plan(config, evidence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
