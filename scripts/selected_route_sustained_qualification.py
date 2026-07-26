#!/usr/bin/env python3
"""Sustained scratch-only qualification for the selected MLX training route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
    if float(config.get("minimum_contiguous_child_wall_seconds") or 0.0) < 7200:
        raise ValueError("sustained qualification must cover at least two hours")
    if int(config.get("minimum_successful_segments") or 0) < 3:
        raise ValueError("first/middle/last qualification requires three segments")
    if not 0.0 < float(config.get("window_fraction") or 0.0) <= 1.0 / 3.0:
        raise ValueError("window fraction must be in (0, 1/3]")
    if not 0.0 < float(
        config.get("minimum_last_to_first_joined_throughput_ratio") or 0.0
    ) <= 1.0:
        raise ValueError("invalid sustained throughput retention gate")
    for key in (
        "require_ac_power",
        "require_user_presence_availability",
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


def first_middle_last(rows: list[dict[str, Any]], fraction: float) -> dict[str, Any]:
    count = max(1, math.ceil(len(rows) * fraction))
    midpoint = len(rows) // 2
    middle_start = max(0, midpoint - count // 2)
    middle_start = min(middle_start, len(rows) - count)
    return {
        "first": summarize_window(rows[:count]),
        "middle": summarize_window(rows[middle_start : middle_start + count]),
        "last": summarize_window(rows[-count:]),
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
    minimum_segments = int(config["minimum_successful_segments"])
    minimum_wall = float(config["minimum_contiguous_child_wall_seconds"])
    completed = len(rows) >= minimum_segments and joined_wall >= minimum_wall
    windows = (
        first_middle_last(rows, float(config["window_fraction"]))
        if len(rows) >= minimum_segments
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
    retention = (
        float((windows.get("last") or {}).get("joined_positions_per_second") or 0.0)
        / float((windows.get("first") or {}).get("joined_positions_per_second") or 1.0)
        if windows
        else 0.0
    )
    availability_paused = bool(
        interruption
        and interruption.get("fault") == "availability_paused"
    )
    hard_gaps: list[str] = []
    if interruption is not None and not availability_paused:
        hard_gaps.append("sustained_window_interrupted")
    if not completed and not availability_paused:
        hard_gaps.append("two_hour_contiguous_window_incomplete")
    if selector.get("trigger_state") != "GREEN" or selector_recipe != recipe:
        hard_gaps.append("selected_recipe_identity_mismatch")
    if not exact_resume:
        hard_gaps.append("exact_resume_validation_failed")
    if config["require_ac_power"] and not all_ac:
        hard_gaps.append("ac_power_requirement_failed")
    if config["require_canonical_lineage_unchanged"] and not canonical_unchanged:
        hard_gaps.append("canonical_lineage_mutated")
    if completed and retention < float(
        config["minimum_last_to_first_joined_throughput_ratio"]
    ):
        hard_gaps.append("sustained_throughput_retention_failed")
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
        "contiguous_child_wall_seconds": round(joined_wall, 6),
        "minimum_contiguous_child_wall_seconds": minimum_wall,
        "exact_resume_each_segment": exact_resume,
        "canonical_lineage_unchanged": canonical_unchanged,
        "canonical_before": canonical_before,
        "canonical_after": canonical_after,
        "all_segments_on_ac_power": all_ac,
        "first_middle_last": windows,
        "last_to_first_joined_throughput_ratio": round(retention, 6),
        "availability_checks": availability_checks,
        "interruption": interruption,
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
        canonical_before = fresh.identities(canonical_paths)
        interruption = {
            "fault": "availability_paused",
            "segment_index": 1,
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
            rows=[],
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
        if not scratch_paths["receipt"].is_file():
            raise ValueError("sustained progress exists without scratch receipt")
        prior_finished = float(progress.get("last_segment_finished_epoch") or 0.0)
        prior_interruption = progress.get("interruption")
        if rows and (
            prior_interruption is not None
            or prior_finished <= 0.0
            or time.time() - prior_finished
            > float(config["maximum_intersegment_wall_seconds"])
        ):
            attempts = list(progress.get("prior_windows") or [])
            attempts.append(
                {
                    "archived_utc": training.now(),
                    "segment_count": len(rows),
                    "child_wall_seconds": round(
                        sum(float(row["child_wall_seconds"]) for row in rows),
                        6,
                    ),
                    "reason": (
                        prior_interruption
                        or {"fault": "intersegment_wall_limit_exceeded"}
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
        }
    segment_dir.mkdir(parents=True, exist_ok=True)
    last_finished = time.time()
    interruption: dict[str, Any] | None = None
    while (
        sum(float(row["child_wall_seconds"]) for row in rows)
        < float(config["minimum_contiguous_child_wall_seconds"])
        or len(rows) < int(config["minimum_successful_segments"])
    ):
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
        if rows:
            gap = time.time() - last_finished
            if gap > float(config["maximum_intersegment_wall_seconds"]):
                interruption = {
                    "fault": "intersegment_wall_limit_exceeded",
                    "gap_seconds": round(gap, 6),
                }
                break
        index = len(rows) + 1
        prior_receipt = read_json(scratch_paths["receipt"])
        prior_positions = int(prior_receipt.get("optimizer_positions") or 0)
        state_before = machine_state()
        try:
            child, host = fresh.guarded_child(
                config=training_config,
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
                "contiguous_child_wall_seconds": report[
                    "contiguous_child_wall_seconds"
                ],
                "last_to_first_joined_throughput_ratio": report[
                    "last_to_first_joined_throughput_ratio"
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
