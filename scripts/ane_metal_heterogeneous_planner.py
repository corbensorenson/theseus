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
                "M1_STATIC_BAKED_COMPILER_REPEATABLE"
                if compiler_repeatable
                else "M1_PRIVATE_COMPILER_UNSTABLE"
                if compiler_successes and compiler_failures
                else "NOT_QUALIFIED"
            ),
        },
        "incompatible_mil_attempts": incompatible_mil_attempts,
        "weight_update_path": evidence.get("weight_update_path"),
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
        blockers = [
            (
                "structure_aligned_persistent_partition_candidate"
                if split_state == "MECHANICS_GREEN_PHYSICAL_JOIN_NOT_SELECTED"
                else "measured_metal_control_and_joint_candidates"
            ),
            "mlx_lazy_graph_integration",
        ]
        if evidence.get("same_surface_bridge", {}).get("state") != (
            "GREEN_CONCURRENT_SHARED_READ_VISIBILITY"
        ):
            blockers.insert(1, "zero_copy_same_surface_visibility")
        if not compiler_repeatable:
            blockers.insert(0, "repeatable_private_ane_compile")
        if evidence.get("weight_update_path", {}).get("state") != "QUALIFIED":
            blockers.append("dynamic_or_persistent_training_weight_update_path")
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
