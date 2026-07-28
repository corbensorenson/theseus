#!/usr/bin/env python3
"""Sustained scratch-only qualification for the selected MLX training route."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import fresh_process_pretraining_qualification as fresh
import moecot_language_arm_training as training
import neural_seed_training_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "selected_route_sustained_qualification.json"
POLICY = "project_theseus_selected_route_sustained_qualification_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "command": command,
            "error": type(exc).__name__,
        }
    text = "\n".join(
        value.strip() for value in (result.stdout, result.stderr) if value.strip()
    )
    return {
        "available": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "text": text,
    }


def machine_state() -> dict[str, Any]:
    power = command_output(["pmset", "-g", "batt"])
    power["on_ac_power"] = "Now drawing from 'AC Power'" in str(
        power.get("text") or ""
    )
    thermal = command_output(["pmset", "-g", "therm"])
    numeric_limits = [
        int(value)
        for value in re.findall(
            r"(?:CPU_Speed_Limit|GPU_Speed_Limit|Scheduler_Limit)\s*=\s*(\d+)",
            str(thermal.get("text") or ""),
        )
    ]
    thermal["performance_limits"] = numeric_limits
    thermal["warning_detected"] = any(value < 100 for value in numeric_limits)
    return {
        "captured_utc": training.now(),
        "power": power,
        "thermal": thermal,
    }


def selected_recipe_from_training(config: dict[str, Any]) -> dict[str, Any]:
    execution = dict((config.get("training") or {}).get("execution_policy") or {})
    pretraining = dict(execution.get("pretraining") or {})
    return {
        "compute_dtype": execution.get("compute_dtype"),
        "fp32_master": execution.get("fp32_master"),
        "token_loss_compute_dtype": execution.get("token_loss_compute_dtype"),
        "training_step_mode": pretraining.get("training_step_mode"),
        "compiled_microbatch_size": pretraining.get("compiled_microbatch_size"),
        "compile_width_quantum": pretraining.get("compile_width_quantum"),
        "optimizer_id": "adamw_mlx",
        "structural_growth_policy": "full_depth_from_step_zero",
        "sequence_length_policy": "fixed_target_width",
        "training_rope_kernel": "mlx_fast",
        "swapout_growth_action": (
            config.get("host_resource_safety") or {}
        ).get("swapout_growth_action"),
    }


def prepare_scratch_paths(
    target: dict[str, Any],
    scratch_root: Path,
    *,
    initialize: bool,
) -> dict[str, Path]:
    scratch_target = training.scratch_target_contract(target, scratch_root)
    if initialize:
        fresh.initialize_scratch(target, scratch_root)
    return fresh.target_paths(scratch_target)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != POLICY:
        raise ValueError("unexpected sustained qualification policy")
    if int(config.get("segment_optimizer_steps") or 0) != 64:
        raise ValueError("selected route must retain qualified 64-step segments")
    if int(config.get("replicates_per_thermal_window") or 0) < 3:
        raise ValueError(
            "a thermal window needs at least three replicates for a median "
            "and an observed uncertainty range"
        )
    if config.get("stopping_rule") != (
        "adjacent_replicated_window_uncertainty_overlap_or_clear_"
        "degradation_v1"
    ):
        raise ValueError("unexpected evidence-driven stopping rule")
    forbidden = {
        "minimum_contiguous_child_wall_seconds",
        "maximum_intersegment_wall_seconds",
        "minimum_last_to_first_joined_throughput_ratio",
    }
    present = sorted(forbidden.intersection(config))
    if present:
        raise ValueError(
            "arbitrary time or percentage gates are forbidden:"
            + ",".join(present)
        )
    for key in (
        "require_ac_power",
        "require_resource_availability",
        "require_exact_resume_each_segment",
        "require_canonical_lineage_unchanged",
        "system_swap_growth_is_diagnostic",
    ):
        if config.get(key) is not True:
            raise ValueError(f"{key} must remain true")


def summarize_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wall = sum(float(row["child_wall_seconds"]) for row in rows)
    device = sum(float(row["device_step_seconds"]) for row in rows)
    positions = sum(int(row["optimizer_position_delta"]) for row in rows)
    return {
        "segment_count": len(rows),
        "optimizer_positions": positions,
        "child_wall_seconds": round(wall, 6),
        "device_step_seconds": round(device, 6),
        "joined_positions_per_second": round(positions / wall, 6),
        "device_positions_per_second": round(positions / device, 6),
        "minimum_reclaimable_available_mib": min(
            float(row["host_resource_safety"]["minimum_reclaimable_available_mib"])
            for row in rows
        ),
        "maximum_process_rss_mib": max(
            float(row["host_resource_safety"]["maximum_process_rss_mib"])
            for row in rows
        ),
        "maximum_inferred_unified_memory_mib": max(
            float(row["host_resource_safety"]["maximum_inferred_unified_memory_mib"])
            for row in rows
        ),
        "maximum_swapout_growth_mib": max(
            float(row["host_resource_safety"]["maximum_swapout_growth_mib"])
            for row in rows
        ),
        "all_on_ac_power": all(
            bool((row["machine_state_after"].get("power") or {}).get("on_ac_power"))
            for row in rows
        ),
        "thermal_warning_observed": any(
            bool(
                (row["machine_state_after"].get("thermal") or {}).get(
                    "warning_detected"
                )
            )
            for row in rows
        ),
    }


def first_middle_last(
    rows: list[dict[str, Any]], window_size: int
) -> dict[str, Any]:
    count = min(len(rows), window_size)
    midpoint = len(rows) // 2
    middle_start = max(0, midpoint - count // 2)
    middle_start = min(middle_start, len(rows) - count)
    return {
        "first": summarize_window(rows[:count]),
        "middle": summarize_window(rows[middle_start : middle_start + count]),
        "last": summarize_window(rows[-count:]),
    }


def adjacent_window_stability(
    rows: list[dict[str, Any]], window_size: int
) -> dict[str, Any]:
    """Stop from replicated wall evidence, never from elapsed clock time."""

    required = 2 * window_size
    if len(rows) < required:
        return {
            "terminal": False,
            "state": "MORE_REPLICATES_REQUIRED",
            "required_segment_count": required,
            "observed_segment_count": len(rows),
        }
    previous_rows = rows[-required:-window_size]
    current_rows = rows[-window_size:]
    previous = [
        float(row["joined_positions_per_second"])
        for row in previous_rows
    ]
    current = [
        float(row["joined_positions_per_second"])
        for row in current_rows
    ]
    ratios = [
        candidate / control
        for candidate in current
        for control in previous
    ]
    uncertainty_low = min(ratios)
    uncertainty_high = max(ratios)
    uncertainty_overlap = uncertainty_low <= 1.0 <= uncertainty_high
    clearly_degraded = max(current) < min(previous)
    clearly_improved = min(current) > max(previous)
    thermal_warning = any(
        bool(
            (row["machine_state_after"].get("thermal") or {}).get(
                "warning_detected"
            )
        )
        for row in previous_rows + current_rows
    )
    terminal = thermal_warning or uncertainty_overlap or clearly_degraded
    state = (
        "THERMAL_WARNING_OBSERVED"
        if thermal_warning
        else "STABLE_WITHIN_OBSERVED_REPLICATE_UNCERTAINTY"
        if uncertainty_overlap
        else "CLEAR_DEGRADATION"
        if clearly_degraded
        else "CLEAR_IMPROVEMENT_CONTINUE_TO_PLATEAU"
        if clearly_improved
        else "MORE_REPLICATES_REQUIRED"
    )
    return {
        "terminal": terminal,
        "state": state,
        "required_segment_count": required,
        "observed_segment_count": len(rows),
        "window_size": window_size,
        "previous_joined_positions_per_second": previous,
        "current_joined_positions_per_second": current,
        "previous_median": statistics.median(previous),
        "current_median": statistics.median(current),
        "current_over_previous_median_ratio": (
            statistics.median(current) / statistics.median(previous)
        ),
        "replicate_ratio_uncertainty_interval": [
            uncertainty_low,
            uncertainty_high,
        ],
        "uncertainty_interval_contains_one": uncertainty_overlap,
        "clear_degradation": clearly_degraded,
        "clear_improvement": clearly_improved,
        "thermal_warning_observed": thermal_warning,
        "arbitrary_percentage_tolerance": None,
        "elapsed_time_requirement": None,
    }


def report_for(
    *,
    config_path: Path,
    config: dict[str, Any],
    selector_path: Path,
    selector: dict[str, Any],
    training_config_path: Path,
    training_config: dict[str, Any],
    plan: dict[str, Any],
    canonical_before: dict[str, Any],
    canonical_after: dict[str, Any],
    rows: list[dict[str, Any]],
    availability_checks: list[dict[str, Any]],
    interruption: dict[str, Any] | None,
) -> dict[str, Any]:
    joined_wall = sum(float(row["child_wall_seconds"]) for row in rows)
    window_size = int(config["replicates_per_thermal_window"])
    stability = adjacent_window_stability(rows, window_size)
    completed = bool(stability["terminal"])
    windows = (
        first_middle_last(rows, window_size)
        if len(rows) >= 2 * window_size
        else {}
    )
    canonical_unchanged = canonical_before == canonical_after
    recipe = selected_recipe_from_training(training_config)
    selector_recipe = selector.get("selected_recipe")
    exact_resume = all(row.get("resume_validation") == "GREEN" for row in rows)
    all_ac = all(
        bool((row["machine_state_after"].get("power") or {}).get("on_ac_power"))
        for row in rows
    )
    availability_paused = bool(
        interruption
        and interruption.get("fault") == "availability_paused"
    )
    hard_gaps: list[str] = []
    if interruption is not None and not availability_paused:
        hard_gaps.append("sustained_window_interrupted")
    if not completed and not availability_paused:
        hard_gaps.append("replicated_thermal_evidence_incomplete")
    if selector.get("trigger_state") != "GREEN" or selector_recipe != recipe:
        hard_gaps.append("selected_recipe_identity_mismatch")
    if not exact_resume:
        hard_gaps.append("exact_resume_validation_failed")
    # A resource-policy transition after an atomic segment is an availability
    # pause, not a training failure. The current invocation's rows remain
    # durable but are archived as a nonqualifying thermal window on resume.
    # A completed window may never hide a battery-powered segment.
    if config["require_ac_power"] and not all_ac and not availability_paused:
        hard_gaps.append("ac_power_requirement_failed")
    if config["require_canonical_lineage_unchanged"] and not canonical_unchanged:
        hard_gaps.append("canonical_lineage_mutated")
    if stability.get("clear_degradation") is True:
        hard_gaps.append("sustained_throughput_degradation_observed")
    if stability.get("thermal_warning_observed") is True:
        hard_gaps.append("thermal_warning_observed")
    trigger_state = (
        "PAUSED"
        if availability_paused and not hard_gaps
        else "GREEN"
        if not hard_gaps
        else "RED"
    )
    return {
        "policy": POLICY,
        "created_utc": training.now(),
        "trigger_state": trigger_state,
        "support_state": "SUPPORTED" if trigger_state == "GREEN" else "INCOMPLETE",
        "hard_gaps": sorted(set(hard_gaps)),
        "config": training.relative(config_path),
        "config_sha256": sha256_file(config_path),
        "training_config": {
            "path": training.relative(training_config_path),
            "sha256": sha256_file(training_config_path),
        },
        "selector": {
            "path": training.relative(selector_path),
            "sha256": sha256_file(selector_path),
            "trigger_state": selector.get("trigger_state"),
        },
        "plan_sha256": plan["plan_sha256"],
        "selected_recipe": recipe,
        "segment_optimizer_steps": int(config["segment_optimizer_steps"]),
        "successful_segment_count": len(rows),
        "observed_child_wall_seconds": round(joined_wall, 6),
        "elapsed_time_requirement": None,
        "arbitrary_percentage_tolerance": None,
        "stopping_rule": config["stopping_rule"],
        "thermal_stability": stability,
        "exact_resume_each_segment": exact_resume,
        "canonical_lineage_unchanged": canonical_unchanged,
        "canonical_before": canonical_before,
        "canonical_after": canonical_after,
        "all_segments_on_ac_power": all_ac,
        "first_middle_last": windows,
        "availability_checks": availability_checks,
        "interruption": interruption,
        "paused_window": (
            {
                "qualifying": False,
                "segment_count": len(rows),
                "reason": list(interruption.get("failed_gates") or []),
                "resume_disposition": (
                    "archive_window_and_continue_exact_scratch_lineage"
                ),
            }
            if availability_paused
            else None
        ),
        "segments": rows,
        "system_swap_growth_treatment": "DIAGNOSTIC_ONLY",
        "capability_claim": "NONE_SUSTAINED_EXECUTION_QUALIFICATION_ONLY",
    }


def execute(config_path: Path, out: Path) -> dict[str, Any]:
    config = read_json(config_path)
    validate_config(config)
    training_config_path = training.resolve(str(config["training_config"]))
    selector_path = training.resolve(str(config["selector_report"]))
    availability_path = training.resolve(str(config["availability_config"]))
    selector = read_json(selector_path)
    availability_policy = read_json(availability_path)
    campaign.validate_availability_policy(availability_policy)
    training_config, plan, target = fresh.canonical_contract(training_config_path)
    scratch_root = training.resolve(str(config["scratch_root"]))
    progress_path = scratch_root / "sustained_progress.json"
    segment_dir = scratch_root / "segments"
    canonical_paths = fresh.target_paths(target)
    availability_checks: list[dict[str, Any]] = []
    initial_availability = campaign.availability_state(availability_policy)
    availability_checks.append(initial_availability)
    if initial_availability["trigger_state"] != "GREEN":
        preserved_rows: list[dict[str, Any]] = []
        canonical_before = fresh.identities(canonical_paths)
        if progress_path.is_file():
            preserved_progress = read_json(progress_path)
            preserved_rows = list(preserved_progress.get("segments") or [])
            canonical_before = dict(
                preserved_progress.get("canonical_before") or canonical_before
            )
        interruption = {
            "fault": "availability_paused",
            "segment_index": len(preserved_rows) + 1,
            "failed_gates": initial_availability["failed_gates"],
        }
        report = report_for(
            config_path=config_path,
            config=config,
            selector_path=selector_path,
            selector=selector,
            training_config_path=training_config_path,
            training_config=training_config,
            plan=plan,
            canonical_before=canonical_before,
            canonical_after=fresh.identities(canonical_paths),
            rows=preserved_rows,
            availability_checks=availability_checks,
            interruption=interruption,
        )
        training.write_json_atomic(out, report)
        return report
    if progress_path.is_file():
        scratch_paths = prepare_scratch_paths(
            target,
            scratch_root,
            initialize=False,
        )
        progress = read_json(progress_path)
        rows = list(progress.get("segments") or [])
        canonical_before = dict(progress["canonical_before"])
        prior_qualified_peak = max(
            [
                float(
                    progress.get(
                        "qualified_peak_inferred_unified_memory_mib"
                    )
                    or 0.0
                )
            ]
            + [
                float(
                    row["host_resource_safety"].get(
                        "maximum_inferred_unified_memory_mib"
                    )
                    or 0.0
                )
                for row in rows
            ]
        )
        progress[
            "qualified_peak_inferred_unified_memory_mib"
        ] = prior_qualified_peak
        if not scratch_paths["receipt"].is_file():
            raise ValueError("sustained progress exists without scratch receipt")
        if rows:
            attempts = list(progress.get("prior_windows") or [])
            attempts.append(
                {
                    "archived_utc": training.now(),
                    "segment_count": len(rows),
                    "child_wall_seconds": round(
                        sum(float(row["child_wall_seconds"]) for row in rows),
                        6,
                    ),
                    "reason": {
                        "fault": "fresh_process_invocation_boundary",
                        "explanation": (
                            "Thermal windows never span an invocation boundary; "
                            "the exact scratch lineage may continue."
                        ),
                    },
                    "qualified_peak_inferred_unified_memory_mib": (
                        prior_qualified_peak
                    ),
                }
            )
            progress["prior_windows"] = attempts
            rows = []
    else:
        scratch_paths = prepare_scratch_paths(
            target,
            scratch_root,
            initialize=True,
        )
        canonical_before = fresh.identities(canonical_paths)
        rows = []
        progress = {
            "policy": POLICY,
            "canonical_before": canonical_before,
            "segments": rows,
            "prior_windows": [],
            "qualified_peak_inferred_unified_memory_mib": 0.0,
        }
    segment_dir.mkdir(parents=True, exist_ok=True)
    last_finished = time.time()
    interruption: dict[str, Any] | None = None
    while not adjacent_window_stability(
        rows, int(config["replicates_per_thermal_window"])
    )["terminal"]:
        availability = campaign.availability_state(availability_policy)
        availability_checks.append(availability)
        if availability["trigger_state"] != "GREEN":
            interruption = {
                "fault": "availability_paused",
                "segment_index": len(rows) + 1,
                "failed_gates": availability["failed_gates"],
            }
            progress.update(
                {
                    "segments": rows,
                    "last_segment_finished_epoch": last_finished,
                    "interruption": interruption,
                }
            )
            training.write_json_atomic(progress_path, progress)
            break
        index = len(rows) + 1
        prior_receipt = read_json(scratch_paths["receipt"])
        prior_positions = int(prior_receipt.get("optimizer_positions") or 0)
        state_before = machine_state()
        guard_config = copy.deepcopy(training_config)
        qualified_peak = max(
            [
                float(
                    progress.get(
                        "qualified_peak_inferred_unified_memory_mib"
                    )
                    or 0.0
                )
            ]
            + [
                float(
                    row["host_resource_safety"].get(
                        "maximum_inferred_unified_memory_mib"
                    )
                    or 0.0
                )
                for row in rows
            ]
        )
        if qualified_peak > 0.0:
            guard_config["host_resource_safety"][
                "qualified_peak_inferred_unified_memory_mib"
            ] = qualified_peak
        try:
            child, host = fresh.guarded_child(
                config=guard_config,
                config_path=training_config_path,
                scratch_root=scratch_root,
                steps=int(config["segment_optimizer_steps"]),
                out=segment_dir / f"segment-{index:04d}.json",
                durable_host_receipt=segment_dir
                / f"segment-{index:04d}.host_resource_safety.json",
                stage_id=f"sustained_segment_{index:04d}",
            )
        except RuntimeError as exc:
            interruption = {"fault": str(exc), "segment_index": index}
            progress.update(
                {
                    "segments": rows,
                    "last_segment_finished_epoch": last_finished,
                    "interruption": interruption,
                }
            )
            training.write_json_atomic(progress_path, progress)
            break
        state_after = machine_state()
        receipt = read_json(scratch_paths["receipt"])
        phase = dict((receipt.get("phases") or {}).get("pretraining") or {})
        position_delta = int(child["optimizer_positions"]) - prior_positions
        child_wall = float(child["wall_seconds"])
        device_seconds = float(child.get("device_step_seconds_total") or 0.0)
        if position_delta <= 0 or child_wall <= 0.0 or device_seconds <= 0.0:
            interruption = {
                "fault": "invalid_segment_progress_or_timing",
                "segment_index": index,
            }
            break
        rows.append(
            {
                "segment_index": index,
                "started_machine_state": state_before,
                "machine_state_after": state_after,
                "optimizer_position_start": prior_positions,
                "optimizer_position_end": int(child["optimizer_positions"]),
                "optimizer_position_delta": position_delta,
                "optimizer_steps": int(child["optimizer_steps"]),
                "child_wall_seconds": child_wall,
                "device_step_seconds": device_seconds,
                "joined_positions_per_second": round(
                    position_delta / child_wall, 6
                ),
                "device_positions_per_second": round(
                    position_delta / device_seconds, 6
                ),
                "final_loss": phase.get("final_loss"),
                "mean_loss": phase.get("mean_loss"),
                "resume_validation": child.get("resume_validation"),
                "checkpoint_sha256": child.get("checkpoint_sha256"),
                "optimizer_state_sha256": child.get(
                    "optimizer_state_sha256"
                ),
                "mlx_rng_state_sha256": child.get("mlx_rng_state_sha256"),
                "host_resource_safety": host,
            }
        )
        progress[
            "qualified_peak_inferred_unified_memory_mib"
        ] = max(
            qualified_peak,
            float(
                host.get("maximum_inferred_unified_memory_mib") or 0.0
            ),
        )
        last_finished = time.time()
        progress.update(
            {
                "policy": POLICY,
                "canonical_before": canonical_before,
                "segments": rows,
                "last_segment_finished_epoch": last_finished,
                "interruption": None,
            }
        )
        training.write_json_atomic(progress_path, progress)
    canonical_after = fresh.identities(canonical_paths)
    report = report_for(
        config_path=config_path,
        config=config,
        selector_path=selector_path,
        selector=selector,
        training_config_path=training_config_path,
        training_config=training_config,
        plan=plan,
        canonical_before=canonical_before,
        canonical_after=canonical_after,
        rows=rows,
        availability_checks=availability_checks,
        interruption=interruption,
    )
    training.write_json_atomic(out, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("sustained qualification requires --execute")
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    out = (
        Path(args.out).resolve()
        if args.out
        else training.resolve(str(config["report"]))
    )
    report = execute(config_path, out)
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "successful_segment_count": report[
                    "successful_segment_count"
                ],
                "observed_child_wall_seconds": report[
                    "observed_child_wall_seconds"
                ],
                "thermal_stability_state": report["thermal_stability"][
                    "state"
                ],
                "hard_gaps": report["hard_gaps"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
