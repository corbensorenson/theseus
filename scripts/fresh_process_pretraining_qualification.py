#!/usr/bin/env python3
"""Qualify bounded fresh-process pretraining on the exact 57M lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import host_resource_safety  # noqa: E402
import moecot_language_arm_training as training  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/moecot_language_arm_training.json"
DEFAULT_OUT = ROOT / "reports/fresh_process_pretraining_qualification.json"
POLICY = "project_theseus_fresh_process_pretraining_qualification_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def canonical_contract(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = training.bind_scale_preregistration(
        training.read_json(config_path)
    )
    plan = training.build_plan(config, config_path=config_path)
    if plan.get("trigger_state") == "RED":
        raise RuntimeError(
            "training plan is not GREEN: "
            + ",".join(plan.get("hard_gaps") or [])
        )
    target = plan["targets"][training.SHARED_TRUNK_ID]
    return config, plan, target


def target_paths(target: dict[str, Any]) -> dict[str, Path]:
    receipt_path = training.resolve(str(target["receipt"]))
    receipt = training.read_json(receipt_path)
    paths = {
        "checkpoint": training.resolve(str(receipt["checkpoint"])),
        "optimizer_state": training.resolve(str(receipt["optimizer_state"])),
        "receipt": receipt_path,
    }
    if receipt.get("mlx_rng_state"):
        paths["mlx_rng_state"] = training.resolve(
            str(receipt["mlx_rng_state"])
        )
    return paths


def identities(paths: dict[str, Path]) -> dict[str, str]:
    return {key + "_sha256": sha256_file(path) for key, path in paths.items()}


def compare_safetensors(
    left: Path,
    right: Path,
    *,
    absolute_tolerance: float = 5e-6,
    relative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Independently compare saved arrays without importing MLX."""

    import gc

    import numpy as np
    from safetensors.numpy import load_file

    left_values = load_file(str(left))
    right_values = load_file(str(right))
    left_names = set(left_values)
    right_names = set(right_values)
    maximum_absolute_delta = 0.0
    maximum_relative_l2_delta = 0.0
    nonfinite_tensor_count = 0
    shape_mismatch_names = []
    tolerance_mismatch_names = []
    for name in sorted(left_names & right_names):
        left_value = np.asarray(left_values[name])
        right_value = np.asarray(right_values[name])
        if left_value.shape != right_value.shape:
            shape_mismatch_names.append(name)
            continue
        if not np.isfinite(left_value).all() or not np.isfinite(right_value).all():
            nonfinite_tensor_count += 1
            continue
        delta = np.asarray(left_value, dtype=np.float64) - np.asarray(
            right_value, dtype=np.float64
        )
        absolute_delta = float(np.max(np.abs(delta), initial=0.0))
        relative_l2 = float(
            np.linalg.norm(delta.ravel())
            / max(
                1e-12,
                np.linalg.norm(
                    np.asarray(left_value, dtype=np.float64).ravel()
                ),
            )
        )
        maximum_absolute_delta = max(
            maximum_absolute_delta, absolute_delta
        )
        maximum_relative_l2_delta = max(
            maximum_relative_l2_delta, relative_l2
        )
        if (
            absolute_delta > absolute_tolerance
            and relative_l2 > relative_tolerance
        ):
            tolerance_mismatch_names.append(name)
    result = {
        "tensor_names_exact": left_names == right_names,
        "left_only_names": sorted(left_names - right_names),
        "right_only_names": sorted(right_names - left_names),
        "shape_mismatch_names": shape_mismatch_names,
        "nonfinite_tensor_count": nonfinite_tensor_count,
        "maximum_absolute_delta": maximum_absolute_delta,
        "maximum_relative_l2_delta": maximum_relative_l2_delta,
        "absolute_tolerance": absolute_tolerance,
        "relative_l2_tolerance": relative_tolerance,
        "acceptance_semantics": (
            "per_tensor_absolute_or_relative_tolerance_v1"
        ),
        "tolerance_mismatch_names": tolerance_mismatch_names,
    }
    result["passed"] = bool(
        result["tensor_names_exact"]
        and not shape_mismatch_names
        and not nonfinite_tensor_count
        and not tolerance_mismatch_names
    )
    del left_values, right_values
    gc.collect()
    return result


def initialize_scratch(
    target: dict[str, Any],
    scratch_root: Path,
) -> dict[str, Any]:
    source_paths = target_paths(target)
    source_receipt = training.read_json(source_paths["receipt"])
    scratch_target = training.scratch_target_contract(target, scratch_root)
    scratch_paths = {
        "checkpoint": Path(str(scratch_target["checkpoint"])),
        "optimizer_state": Path(str(scratch_target["optimizer_state"])),
        "receipt": Path(str(scratch_target["receipt"])),
    }
    scratch_paths["mlx_rng_state"] = training.rng_state_path(
        scratch_paths["optimizer_state"]
    )
    for key in ("checkpoint", "optimizer_state"):
        clone_exact(source_paths[key], scratch_paths[key])
    if "mlx_rng_state" in source_paths:
        clone_exact(
            source_paths["mlx_rng_state"],
            scratch_paths["mlx_rng_state"],
        )
    scratch_receipt = dict(source_receipt)
    scratch_receipt.update(
        {
            "checkpoint": str(scratch_paths["checkpoint"]),
            "optimizer_state": str(scratch_paths["optimizer_state"]),
        }
    )
    if "mlx_rng_state" in source_paths:
        scratch_receipt["mlx_rng_state"] = str(
            scratch_paths["mlx_rng_state"]
        )
    else:
        scratch_receipt.pop("mlx_rng_state", None)
        scratch_receipt.pop("mlx_rng_state_sha256", None)
    training.write_json_atomic(scratch_paths["receipt"], scratch_receipt)
    return scratch_target


def clone_scratch_lineage(
    source_target: dict[str, Any],
    target: dict[str, Any],
    destination_root: Path,
) -> dict[str, Any]:
    """Clone one exact intermediate generation into an independent replay root."""

    source_paths = target_paths(source_target)
    source_receipt = training.read_json(source_paths["receipt"])
    destination_target = training.scratch_target_contract(
        target, destination_root
    )
    destination_paths = {
        "checkpoint": Path(str(destination_target["checkpoint"])),
        "optimizer_state": Path(str(destination_target["optimizer_state"])),
        "receipt": Path(str(destination_target["receipt"])),
    }
    destination_paths["mlx_rng_state"] = training.rng_state_path(
        destination_paths["optimizer_state"]
    )
    for key in ("checkpoint", "optimizer_state", "mlx_rng_state"):
        clone_exact(source_paths[key], destination_paths[key])
    destination_receipt = dict(source_receipt)
    destination_receipt.update(
        {
            "checkpoint": str(destination_paths["checkpoint"]),
            "optimizer_state": str(destination_paths["optimizer_state"]),
            "mlx_rng_state": str(destination_paths["mlx_rng_state"]),
        }
    )
    training.write_json_atomic(
        destination_paths["receipt"], destination_receipt
    )
    return destination_target


def run_child(
    *,
    config_path: Path,
    scratch_root: Path,
    steps: int,
    out: Path,
) -> int:
    config, plan, target = canonical_contract(config_path)
    scratch_target = training.scratch_target_contract(target, scratch_root)
    stage_dir = training.resolve(str(config["stage_dir"]))
    metadata = training.read_json(stage_dir / "stage_metadata_v1.json")
    canonical = metadata["summary"]["canonical_pretrain_stage"]
    stage = training.canonical_pretraining_execution_stage(
        stage_dir,
        canonical,
        active=True,
    )
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    started = time.perf_counter()
    receipt = training.train_target(
        config,
        plan,
        scratch_target,
        stage=stage,
        source_conditioned_stage=None,
        kernel_english_stage=None,
        supervision_stage=None,
        max_steps=steps,
        resume=True,
        training_phase="pretraining",
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
    )
    validation = training.validate_resume(
        receipt,
        plan,
        scratch_target,
        Path(str(receipt["checkpoint"])),
        Path(str(receipt["optimizer_state"])),
    )
    phase = dict((receipt.get("phases") or {}).get("pretraining") or {})
    result = {
        "policy": "project_theseus_fresh_process_pretraining_child_v1",
        "trigger_state": "GREEN",
        "requested_steps": steps,
        "optimizer_steps": int(phase.get("optimizer_steps") or 0),
        "optimizer_positions": int(receipt.get("optimizer_positions") or 0),
        "pretrain_optimizer_positions": int(
            receipt.get("pretrain_optimizer_positions") or 0
        ),
        "data_cursor_next": phase.get("data_cursor_next"),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "optimizer_state_sha256": receipt["optimizer_state_sha256"],
        "mlx_rng_state_sha256": receipt["mlx_rng_state_sha256"],
        "resume_validation": "GREEN",
        "plan_identity_migration": validation,
        "tokens_per_second": phase.get("tokens_per_second"),
        "device_step_seconds_total": phase.get(
            "device_step_seconds_total"
        ),
        "mlx_active_memory_bytes_maximum": phase.get(
            "mlx_active_memory_bytes_maximum"
        ),
        "mlx_cache_memory_bytes_maximum": phase.get(
            "mlx_cache_memory_bytes_maximum"
        ),
        "checkpoint_publication": receipt.get("checkpoint_publication"),
        "phase_boundary_cache_releases": receipt.get(
            "phase_boundary_cache_releases"
        ),
        "wall_seconds": round(time.perf_counter() - started, 6),
    }
    training.write_json_atomic(out, result)
    return 0


def guarded_child(
    *,
    config: dict[str, Any],
    config_path: Path,
    scratch_root: Path,
    steps: int,
    out: Path,
    durable_host_receipt: Path,
    stage_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        str(
            training.resolve(
                str(config["host_resource_safety"]["qualified_python"])
            )
        ),
        str(Path(__file__).resolve()),
        "--child",
        "--config",
        str(config_path),
        "--scratch-root",
        str(scratch_root),
        "--steps",
        str(steps),
        "--out",
        str(out),
    ]
    try:
        result = host_resource_safety.run_guarded(
            command,
            cwd=ROOT,
            policy=training.training_host_policy(config),
            env={
                "THESEUS_GUARDED_ACCELERATOR_CHILD": "1",
                "THESEUS_GUARDED_TRAINING_CHILD": "1",
            },
        )
    except host_resource_safety.HostResourceSafetyFault as exc:
        failure = {
            "policy": POLICY,
            "stage_id": stage_id,
            "passed": False,
            "fault": str(exc),
            "child_started": False,
            "command": command,
        }
        training.write_json_atomic(durable_host_receipt, failure)
        raise RuntimeError(
            "fresh-process child failed before launch: " + str(exc)
        ) from exc
    training.write_json_atomic(
        durable_host_receipt,
        {
            "policy": POLICY,
            "stage_id": stage_id,
            "passed": bool(result.receipt.get("passed")),
            "fault": result.receipt.get("fault"),
            "host_resource_safety": result.receipt,
        },
    )
    if not result.receipt.get("passed") or not out.is_file():
        raise RuntimeError(
            "fresh-process child failed: "
            + str(result.receipt.get("fault") or result.stderr[-1000:])
        )
    return training.read_json(out), result.receipt


def qualify(
    config_path: Path, *, durable_host_receipt: Path
) -> dict[str, Any]:
    config, plan, target = canonical_contract(config_path)
    segment_policy = dict(
        config["architecture_training_authority"]["fresh_process_segments"]
    )
    segment_steps = int(segment_policy["maximum_optimizer_steps"])
    canonical_paths = target_paths(target)
    canonical_before = identities(canonical_paths)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="theseus-fresh-process-", dir="/private/tmp"
    ) as directory:
        root = Path(directory)
        segmented_root = root / "segmented"
        replay_root = root / "replay"
        initialize_scratch(target, segmented_root)
        first, first_host = guarded_child(
            config=config,
            config_path=config_path,
            scratch_root=segmented_root,
            steps=segment_steps,
            out=root / "segment-1.json",
            durable_host_receipt=durable_host_receipt,
            stage_id="segment_1",
        )
        clone_scratch_lineage(
            training.scratch_target_contract(target, segmented_root),
            target,
            replay_root,
        )
        second, second_host = guarded_child(
            config=config,
            config_path=config_path,
            scratch_root=segmented_root,
            steps=segment_steps,
            out=root / "segment-2.json",
            durable_host_receipt=durable_host_receipt,
            stage_id="segment_2",
        )
        replay, replay_host = guarded_child(
            config=config,
            config_path=config_path,
            scratch_root=replay_root,
            steps=segment_steps,
            out=root / "segment-2-replay.json",
            durable_host_receipt=durable_host_receipt,
            stage_id="segment_2_independent_replay",
        )
        segment_rows = [first, second]
        host_rows = [first_host, second_host, replay_host]
        segmented_paths = target_paths(
            training.scratch_target_contract(target, segmented_root)
        )
        replay_paths = target_paths(
            training.scratch_target_contract(target, replay_root)
        )
        segmented_ids = identities(segmented_paths)
        replay_ids = identities(replay_paths)
        model_tensor_comparison = compare_safetensors(
            segmented_paths["checkpoint"],
            replay_paths["checkpoint"],
        )
        optimizer_tensor_comparison = compare_safetensors(
            segmented_paths["optimizer_state"],
            replay_paths["optimizer_state"],
        )
    canonical_after = identities(canonical_paths)
    byte_exact_artifact_parity = all(
        replay_ids[key] == segmented_ids[key]
        for key in (
            "checkpoint_sha256",
            "optimizer_state_sha256",
            "mlx_rng_state_sha256",
        )
    )
    numeric_parity = bool(
        model_tensor_comparison["passed"]
        and optimizer_tensor_comparison["passed"]
        and replay_ids["mlx_rng_state_sha256"]
        == segmented_ids["mlx_rng_state_sha256"]
    )
    contiguous = bool(
        segment_rows[0]["optimizer_positions"]
        < segment_rows[1]["optimizer_positions"]
        and replay["optimizer_positions"]
        == segment_rows[1]["optimizer_positions"]
        and replay["data_cursor_next"]
        == segment_rows[1]["data_cursor_next"]
    )
    zero_swap = all(
        float(row.get("maximum_swapout_growth_mib") or 0.0) == 0.0
        for row in host_rows
    )
    swapout_growth_action = str(
        (config.get("host_resource_safety") or {}).get(
            "swapout_growth_action"
        )
        or "fail_closed"
    )
    swap_growth_treatment = (
        "DIAGNOSTIC_ONLY"
        if swapout_growth_action == "report_only"
        else "HARD_GAP"
    )
    host_resource_guard_passed = all(
        row.get("passed") is True and not str(row.get("fault") or "")
        for row in host_rows
    )
    canonical_unchanged = canonical_before == canonical_after
    exact_resume = all(
        row.get("resume_validation") == "GREEN"
        for row in (*segment_rows, replay)
    )
    hard_gaps = []
    if not numeric_parity:
        hard_gaps.append("independent_segmented_replay_numeric_mismatch")
    if not contiguous:
        hard_gaps.append("segmented_cursor_or_position_discontinuity")
    if not zero_swap and swap_growth_treatment == "HARD_GAP":
        hard_gaps.append("swap_growth_observed")
    if not host_resource_guard_passed:
        hard_gaps.append("host_resource_guard_failed")
    if not canonical_unchanged:
        hard_gaps.append("canonical_lineage_mutated")
    if not exact_resume:
        hard_gaps.append("exact_resume_validation_failed")
    segmented_device_seconds = sum(
        float(row.get("device_step_seconds_total") or 0.0)
        for row in segment_rows
    )
    segmented_wall = sum(
        float(row.get("wall_seconds") or 0.0) for row in segment_rows
    )
    segmented_publication_seconds = sum(
        float((row.get("checkpoint_publication") or {}).get("total_seconds") or 0.0)
        for row in segment_rows
    )
    return {
        "policy": POLICY,
        "created_utc": training.now(),
        "trigger_state": "RED" if hard_gaps else "GREEN",
        "hard_gaps": hard_gaps,
        "training_config": {
            "path": training.relative(config_path),
            "sha256": sha256_file(config_path),
        },
        "plan_sha256": plan["plan_sha256"],
        "qualified_execution_policy": segment_policy,
        "contiguous_segment_count": len(segment_rows),
        "segment_optimizer_steps": segment_steps,
        "canonical_lineage_unchanged": canonical_unchanged,
        "exact_resume_validation": exact_resume,
        "independent_segmented_replay_byte_exact": (
            byte_exact_artifact_parity
        ),
        "independent_segmented_replay_numeric_parity": numeric_parity,
        "model_tensor_comparison": model_tensor_comparison,
        "optimizer_tensor_comparison": optimizer_tensor_comparison,
        "segmented_cursor_and_position_contiguous": contiguous,
        "zero_swap_growth": zero_swap,
        "swap_growth_treatment": swap_growth_treatment,
        "host_resource_guard_passed": host_resource_guard_passed,
        "fresh_process_segments": segment_rows,
        "second_segment_independent_replay": replay,
        "fresh_process_host_safety": host_rows,
        "segmented_child_wall_seconds": round(segmented_wall, 6),
        "segmented_device_step_seconds": round(
            segmented_device_seconds, 6
        ),
        "segmented_checkpoint_publication_seconds": round(
            segmented_publication_seconds, 6
        ),
        "non_device_segment_overhead_seconds": round(
            max(0.0, segmented_wall - segmented_device_seconds), 6
        ),
        "canonical_before": canonical_before,
        "canonical_after": canonical_after,
        "qualification_wall_seconds": round(
            time.perf_counter() - started, 6
        ),
        "capability_claim": "NONE_EXECUTION_AND_RESUME_QUALIFICATION_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scratch-root", default="", help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    out = Path(args.out).resolve()
    if args.child:
        if not args.scratch_root or args.steps <= 0:
            parser.error("child mode requires scratch root and positive steps")
        return run_child(
            config_path=config_path,
            scratch_root=Path(args.scratch_root).resolve(),
            steps=args.steps,
            out=out,
        )
    if not args.execute:
        parser.error("qualification requires --execute")
    report = qualify(
        config_path,
        durable_host_receipt=out.with_suffix(".host_resource_safety.json"),
    )
    training.write_json_atomic(out, report)
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "contiguous_segment_count": report[
                    "contiguous_segment_count"
                ],
                "independent_segmented_replay_numeric_parity": report[
                    "independent_segmented_replay_numeric_parity"
                ],
                "zero_swap_growth": report["zero_swap_growth"],
                "non_device_segment_overhead_seconds": report[
                    "non_device_segment_overhead_seconds"
                ],
                "hard_gaps": report["hard_gaps"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if report["trigger_state"] == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
