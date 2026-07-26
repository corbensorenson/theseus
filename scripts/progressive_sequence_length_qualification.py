#!/usr/bin/env python3
"""Matched-token MLX qualification for a 128 -> 256 -> 512 curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

import masked_structural_growth_qualification as growth
import mtp_matched_adequacy as corpus
import optimizer_matched_adequacy as optimizer_adequacy
from standard_causal_transformer_model import (
    analytical_parameter_breakdown,
    build_model,
)
from standard_causal_transformer_survival import model_vocab_size


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/progressive_sequence_length_qualification.json"
POLICY = "project_theseus_progressive_sequence_length_qualification_v1"


class ProgressiveLengthFault(ValueError):
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


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy") != POLICY:
        raise ProgressiveLengthFault("config_policy_invalid")
    steps = int(config["training"]["steps"])
    schedule = config.get("curriculum") or []
    if (
        [int(row["maximum_sequence_tokens"]) for row in schedule]
        != [128, 256, 512]
        or int(schedule[-1]["stop_step"]) != steps
        or any(
            int(left["stop_step"]) >= int(right["stop_step"])
            for left, right in zip(schedule, schedule[1:])
        )
    ):
        raise ProgressiveLengthFault("curriculum_128_256_512_invalid")
    if int(config["control_maximum_sequence_tokens"]) != 512:
        raise ProgressiveLengthFault("control_width_invalid")
    if not 0.0 < float(config["windowing"]["context_fraction"]) < 1.0:
        raise ProgressiveLengthFault("window_context_fraction_invalid")
    if len(set(int(seed) for seed in config.get("seeds") or [])) < 3:
        raise ProgressiveLengthFault("three_distinct_seeds_required")
    if steps <= 0 or steps > 192:
        raise ProgressiveLengthFault("step_budget_invalid")
    boundaries = config.get("hard_boundaries") or {}
    for field in (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "fallback_or_template_credit",
        "confirmation_surface_consumption",
    ):
        if boundaries.get(field) != 0:
            raise ProgressiveLengthFault(f"hard_boundary_nonzero:{field}")
    for field in (
        "production_checkpoint_mutation",
        "heldout_labels_visible_to_candidate",
        "selection_from_throughput_alone",
        "supervised_token_dropping",
    ):
        if boundaries.get(field) is not False:
            raise ProgressiveLengthFault(f"hard_boundary_boolean_invalid:{field}")
    return config


def target_position_count(row: dict[str, Any]) -> int:
    return max(
        0, len(row["sequence"]) - 1 - int(row["target_mask_start"])
    )


def window_supervised_row(
    row: dict[str, Any],
    maximum: int,
    *,
    context_fraction: float,
) -> list[dict[str, Any]]:
    """Split target positions exactly once while retaining bounded left context."""

    sequence = list(row["sequence"])
    prediction_stop = len(sequence) - 1
    target_start = int(row["target_mask_start"])
    if prediction_stop <= maximum:
        return [dict(row)]
    windows: list[dict[str, Any]] = []
    cursor = target_start
    context_budget = max(1, int(maximum * context_fraction))
    while cursor < prediction_stop:
        start = max(0, cursor - context_budget)
        target_capacity = maximum - (cursor - start)
        if target_capacity <= 0:
            raise ProgressiveLengthFault("window_target_capacity_zero")
        stop = min(prediction_stop, cursor + target_capacity)
        window = dict(row)
        window["row_id"] = f"{row['row_id']}:window:{maximum}:{cursor}:{stop}"
        window["sequence"] = sequence[start : stop + 1]
        window["target_mask_start"] = cursor - start
        window["window_source_identity"] = row["source_identity"]
        window["window_original_target_start"] = cursor
        window["window_original_target_stop"] = stop
        windows.append(window)
        cursor = stop
    if sum(target_position_count(window) for window in windows) != target_position_count(
        row
    ):
        raise ProgressiveLengthFault("window_supervision_conservation_failed")
    if any(len(window["sequence"]) - 1 > maximum for window in windows):
        raise ProgressiveLengthFault("window_width_exceeded")
    return windows


def window_logical_batch(
    rows: list[dict[str, Any]],
    maximum: int,
    *,
    context_fraction: float,
) -> list[dict[str, Any]]:
    return [
        window
        for row in rows
        for window in window_supervised_row(
            row, maximum, context_fraction=context_fraction
        )
    ]


def padding_row(index: int) -> dict[str, Any]:
    return {
        "arm_id": "padding",
        "row_id": f"padding:{index}",
        "source_identity": f"padding:{index}",
        "dataset_id": "none",
        "license_spdx": "NONE",
        "sequence": [0, 0],
        "target_mask_start": 1,
        "target_byte_count": 0,
    }


def pad_rows(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if len(rows) > size:
        raise ProgressiveLengthFault("compiled_batch_pad_target_too_small")
    return rows + [padding_row(index) for index in range(size - len(rows))]


def curriculum_width(config: dict[str, Any], step: int) -> int:
    for stage in config["curriculum"]:
        if step <= int(stage["stop_step"]):
            return int(stage["maximum_sequence_tokens"])
    raise ProgressiveLengthFault("curriculum_step_out_of_range")


def make_compiled_step(
    model: Any,
    optimizer: Any,
    *,
    mx: Any,
    nn: Any,
    optim: Any,
    mlx_utils: Any,
    clip: float,
) -> Any:
    def objective(local_model: Any, x: Any, y: Any, mask: Any) -> Any:
        logits, _cache = local_model(x)
        losses = nn.losses.cross_entropy(logits, y)
        denominator = mx.maximum(
            mx.sum(mask), mx.array(1.0, dtype=mx.float32)
        )
        return mx.sum(losses * mask) / denominator

    value_and_grad = nn.value_and_grad(model, objective)
    compiled_state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=compiled_state, outputs=compiled_state)
    def compiled_step(x: Any, y: Any, mask: Any) -> tuple[Any, Any, Any]:
        loss, gradients = value_and_grad(model, x, y, mask)
        gradient_l1 = sum(
            mx.sum(mx.abs(value))
            for _name, value in mlx_utils.tree_flatten(gradients)
        )
        gradients, gradient_norm = optim.clip_grad_norm(gradients, clip)
        optimizer.update(model, gradients)
        return loss, gradient_norm, gradient_l1

    return compiled_step


def batch_loss(
    model: Any, x: Any, y: Any, mask: Any, *, mx: Any, nn: Any
) -> Any:
    logits, _cache = model(x)
    losses = nn.losses.cross_entropy(logits, y)
    denominator = mx.maximum(
        mx.sum(mask), mx.array(1.0, dtype=mx.float32)
    )
    return mx.sum(losses * mask) / denominator


def evaluate(
    model: Any,
    rows: dict[str, list[dict[str, Any]]],
    maximum: int,
    *,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    return corpus._evaluate(model, rows, maximum, mx, nn)


def checkpoint_replay(
    *,
    model: Any,
    optimizer: Any,
    local_config: Any,
    x: Any,
    y: Any,
    mask: Any,
    checkpoint_root: Path,
    checkpoint_name: str,
    config: dict[str, Any],
    mx: Any,
    nn: Any,
    optim: Any,
    mlx_utils: Any,
    retain_primary_update: bool,
) -> tuple[dict[str, Any], float, float]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_root / f"{checkpoint_name}.model.npz"
    optimizer_path = checkpoint_root / f"{checkpoint_name}.optimizer.npz"
    model_names = growth.save_tree(
        model_path, model.parameters(), mx=mx, mlx_utils=mlx_utils
    )
    optimizer_names = growth.save_tree(
        optimizer_path, optimizer.state, mx=mx, mlx_utils=mlx_utils
    )
    checkpoint_sha256 = {
        "model": sha256_file(model_path),
        "optimizer": sha256_file(optimizer_path),
        "model_names": sha256_file(model_names),
        "optimizer_names": sha256_file(optimizer_names),
    }
    clone = build_model(local_config, mx=mx, nn=nn)
    clone.update(
        growth.load_tree(model_path, model_names, mx=mx, mlx_utils=mlx_utils)
    )
    clone_optimizer = growth.build_adamw(config, mx=mx, optim=optim)
    clone_optimizer.state = growth.load_tree(
        optimizer_path, optimizer_names, mx=mx, mlx_utils=mlx_utils
    )
    clone_optimizer.init(clone.trainable_parameters())
    mx.eval(clone.parameters(), clone_optimizer.state)
    exact_model = growth.tree_digest(
        model.parameters(), mlx_utils
    ) == growth.tree_digest(clone.parameters(), mlx_utils)
    exact_optimizer = growth.tree_digest(
        optimizer.state, mlx_utils
    ) == growth.tree_digest(clone_optimizer.state, mlx_utils)
    primary_step = make_compiled_step(
        model,
        optimizer,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
        clip=float(config["training"]["gradient_clip_norm"]),
    )
    clone_step = make_compiled_step(
        clone,
        clone_optimizer,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
        clip=float(config["training"]["gradient_clip_norm"]),
    )
    primary_started = time.perf_counter()
    primary_loss, primary_norm, primary_l1 = primary_step(x, y, mask)
    mx.eval(
        primary_loss,
        primary_norm,
        primary_l1,
        model.parameters(),
        optimizer.state,
    )
    primary_seconds = time.perf_counter() - primary_started
    audit_started = time.perf_counter()
    clone_loss, clone_norm, clone_l1 = clone_step(x, y, mask)
    mx.eval(
        clone_loss,
        clone_norm,
        clone_l1,
        clone.parameters(),
        clone_optimizer.state,
    )
    clone_seconds = time.perf_counter() - audit_started
    model_error = growth.tree_max_difference(
        model.parameters(), clone.parameters(), mlx_utils
    )
    optimizer_error = growth.tree_max_difference(
        optimizer.state, clone_optimizer.state, mlx_utils
    )
    tolerance = float(config["decision"]["maximum_next_update_absolute_error"])
    receipt = {
        "checkpoint_sha256": checkpoint_sha256,
        "exact_model_reload": exact_model,
        "exact_optimizer_reload": exact_optimizer,
        "next_loss_absolute_error": abs(
            float(primary_loss.item()) - float(clone_loss.item())
        ),
        "next_gradient_norm_absolute_error": abs(
            float(primary_norm.item()) - float(clone_norm.item())
        ),
        "next_model_max_absolute_error": model_error,
        "next_optimizer_max_absolute_error": optimizer_error,
        "maximum_allowed_next_update_absolute_error": tolerance,
        "next_update_numerically_equivalent": (
            model_error <= tolerance
            and optimizer_error <= tolerance
            and float(primary_loss.item()) == float(clone_loss.item())
            and float(primary_norm.item()) == float(clone_norm.item())
        ),
        "primary_loss": float(primary_loss.item()),
        "primary_gradient_norm": float(primary_norm.item()),
        "primary_gradient_l1": float(primary_l1.item()),
    }
    if not retain_primary_update:
        model.update(
            growth.load_tree(
                model_path, model_names, mx=mx, mlx_utils=mlx_utils
            )
        )
        optimizer.state = growth.load_tree(
            optimizer_path,
            optimizer_names,
            mx=mx,
            mlx_utils=mlx_utils,
        )
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)
    for path in (model_path, optimizer_path, model_names, optimizer_names):
        path.unlink(missing_ok=True)
    return (
        receipt,
        primary_seconds if retain_primary_update else 0.0,
        clone_seconds
        if retain_primary_update
        else primary_seconds + clone_seconds,
    )


def long_context_rows(
    rows: dict[str, list[dict[str, Any]]], threshold: int
) -> dict[str, list[dict[str, Any]]]:
    return {
        arm: [
            row for row in arm_rows if len(row["sequence"]) - 1 > threshold
        ]
        for arm, arm_rows in rows.items()
    }


def run_one(
    config: dict[str, Any],
    *,
    run_id: str,
    seed: int,
    logical_batches: list[list[dict[str, Any]]],
    pad_sizes: dict[int, int],
    heldout: dict[str, list[dict[str, Any]]],
    heldout_long: dict[str, list[dict[str, Any]]],
    vocabulary: int,
    scratch: Path,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    steps = int(config["training"]["steps"])
    context_fraction = float(config["windowing"]["context_fraction"])
    full_width = int(config["control_maximum_sequence_tokens"])
    local_config = growth.model_config(
        config, vocabulary, depth=int(config["model"]["num_layers"])
    )
    growth.reset_peak_memory(mx)
    mx.random.seed(int(seed))
    started = time.perf_counter()
    model = build_model(local_config, mx=mx, nn=nn)
    initial_sha256 = growth.tree_digest(model.parameters(), mlx_utils)
    optimizer = growth.build_adamw(config, mx=mx, optim=optim)
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    compiled_step = make_compiled_step(
        model,
        optimizer,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
        clip=float(config["training"]["gradient_clip_norm"]),
    )
    initial = evaluate(model, heldout, full_width, mx=mx, nn=nn)
    initial_long = evaluate(
        model, heldout_long, full_width, mx=mx, nn=nn
    )
    curve = [
        {
            "step": 0,
            "heldout": initial,
            "joined_wall_seconds": time.perf_counter() - started,
        }
    ]
    current_width = full_width if run_id == "fixed_512_control" else 128
    transition_receipts: list[dict[str, Any]] = []
    finite_gradients = True
    first_gradient_l1 = None
    positions = 0
    primary_step_seconds = 0.0
    audit_seconds = 0.0
    input_tokens = 0
    padded_input_tokens = 0
    window_rows = 0
    for step, logical_batch in enumerate(logical_batches, 1):
        width = (
            full_width
            if run_id == "fixed_512_control"
            else curriculum_width(config, step)
        )
        rows = window_logical_batch(
            logical_batch, width, context_fraction=context_fraction
        )
        unpadded_count = len(rows)
        rows = pad_rows(rows, pad_sizes[width])
        x_np, y_np, mask_np = corpus.make_batch(rows, width)
        if int(mask_np.sum()) != sum(
            target_position_count(row) for row in logical_batch
        ):
            raise ProgressiveLengthFault(
                f"step_supervision_conservation_failed:{run_id}:{seed}:{step}"
            )
        x, y, mask = mx.array(x_np), mx.array(y_np), mx.array(mask_np)
        transition = width != current_width
        if transition:
            old_rows = pad_rows(
                window_logical_batch(
                    logical_batch,
                    current_width,
                    context_fraction=context_fraction,
                ),
                pad_sizes[current_width],
            )
            old_x_np, old_y_np, old_mask_np = corpus.make_batch(
                old_rows, current_width
            )
            old_loss = batch_loss(
                model,
                mx.array(old_x_np),
                mx.array(old_y_np),
                mx.array(old_mask_np),
                mx=mx,
                nn=nn,
            )
            new_loss = batch_loss(model, x, y, mask, mx=mx, nn=nn)
            mx.eval(old_loss, new_loss)
            replay, replay_primary, replay_audit = checkpoint_replay(
                model=model,
                optimizer=optimizer,
                local_config=local_config,
                x=x,
                y=y,
                mask=mask,
                checkpoint_root=scratch / f"{run_id}_{seed}",
                checkpoint_name=f"transition_{step}_{width}",
                config=config,
                mx=mx,
                nn=nn,
                optim=optim,
                mlx_utils=mlx_utils,
                retain_primary_update=True,
            )
            transition_receipts.append(
                {
                    "step": step,
                    "prior_width": current_width,
                    "target_width": width,
                    "old_width_loss": float(old_loss.item()),
                    "new_width_loss": float(new_loss.item()),
                    "relative_loss_jump": (
                        float(new_loss.item()) - float(old_loss.item())
                    )
                    / max(float(old_loss.item()), 1e-12),
                    "checkpoint_replay": replay,
                }
            )
            primary_step_seconds += replay_primary
            audit_seconds += replay_audit
            loss_value = float(replay["primary_loss"])
            gradient_norm_value = float(replay["primary_gradient_norm"])
            gradient_l1_value = float(replay["primary_gradient_l1"])
            compiled_step = make_compiled_step(
                model,
                optimizer,
                mx=mx,
                nn=nn,
                optim=optim,
                mlx_utils=mlx_utils,
                clip=float(config["training"]["gradient_clip_norm"]),
            )
            current_width = width
        else:
            step_started = time.perf_counter()
            loss, gradient_norm, gradient_l1 = compiled_step(x, y, mask)
            mx.eval(
                loss,
                gradient_norm,
                gradient_l1,
                model.parameters(),
                optimizer.state,
            )
            primary_step_seconds += time.perf_counter() - step_started
            loss_value = float(loss.item())
            gradient_norm_value = float(gradient_norm.item())
            gradient_l1_value = float(gradient_l1.item())
        finite_gradients = (
            finite_gradients
            and math.isfinite(loss_value)
            and math.isfinite(gradient_norm_value)
            and math.isfinite(gradient_l1_value)
            and gradient_l1_value > 0.0
        )
        if first_gradient_l1 is None:
            first_gradient_l1 = gradient_l1_value
        positions += int(mask_np.sum())
        input_tokens += int(
            sum(
                len(row["sequence"]) - 1
                for row in rows[:unpadded_count]
            )
        )
        padded_input_tokens += int(x_np.size)
        window_rows += unpadded_count
        if (
            step % int(config["training"]["evaluation_interval_steps"]) == 0
            or step == steps
        ):
            heldout_result = evaluate(
                model, heldout, full_width, mx=mx, nn=nn
            )
            curve.append(
                {
                    "step": step,
                    "heldout": heldout_result,
                    "joined_wall_seconds": (
                        time.perf_counter() - started - audit_seconds
                    ),
                }
            )
    final = curve[-1]["heldout"]
    final_long = evaluate(
        model, heldout_long, full_width, mx=mx, nn=nn
    )
    final_rows = window_logical_batch(
        logical_batches[-1],
        current_width,
        context_fraction=context_fraction,
    )
    final_rows = pad_rows(final_rows, pad_sizes[current_width])
    final_x_np, final_y_np, final_mask_np = corpus.make_batch(
        final_rows, current_width
    )
    final_replay, _unused_primary, final_audit = checkpoint_replay(
        model=model,
        optimizer=optimizer,
        local_config=local_config,
        x=mx.array(final_x_np),
        y=mx.array(final_y_np),
        mask=mx.array(final_mask_np),
        checkpoint_root=scratch / f"{run_id}_{seed}",
        checkpoint_name="final",
        config=config,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
        retain_primary_update=False,
    )
    audit_seconds += final_audit
    total_wall = time.perf_counter() - started
    joined_wall = total_wall - audit_seconds
    return {
        "run_id": run_id,
        "kind": (
            "fixed_512_control"
            if run_id == "fixed_512_control"
            else "progressive_128_256_512"
        ),
        "seed": int(seed),
        "model": asdict(local_config),
        "initial_parameter_sha256": initial_sha256,
        "optimizer_id": "adamw_mlx",
        "optimizer_steps": steps,
        "optimizer_positions": positions,
        "input_tokens_including_repeated_context": input_tokens,
        "padded_input_tokens": padded_input_tokens,
        "window_row_count": window_rows,
        "initial_heldout": initial,
        "initial_long_context_heldout": initial_long,
        "final_heldout": final,
        "final_long_context_heldout": final_long,
        "curve": curve,
        "transition_receipts": transition_receipts,
        "final_checkpoint_replay": final_replay,
        "finite_gradients": finite_gradients,
        "first_gradient_l1": first_gradient_l1,
        "training_primary_step_seconds": primary_step_seconds,
        "joined_training_wall_seconds": joined_wall,
        "qualification_total_wall_seconds": total_wall,
        "excluded_audit_twin_seconds": audit_seconds,
        "peak_allocator_bytes": growth.peak_memory_bytes(mx),
        "capability_claim": "NONE_ENGINEERING_SELECTION_ONLY",
    }


def first_quality_point(
    curve: list[dict[str, Any]], threshold: float
) -> tuple[int, float] | None:
    for point in curve:
        if float(point["heldout"]["ntp_loss"]) <= threshold:
            return int(point["step"]), float(point["joined_wall_seconds"])
    return None


def compare(
    runs: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = {
        int(row["seed"]): row
        for row in runs
        if row["run_id"] == "fixed_512_control"
    }
    candidates = [
        row for row in runs if row["run_id"] == "progressive_128_256_512"
    ]
    paired = []
    for row in candidates:
        control = controls[int(row["seed"])]
        control_loss = float(control["final_heldout"]["ntp_loss"])
        candidate_loss = float(row["final_heldout"]["ntp_loss"])
        control_long = float(
            control["final_long_context_heldout"]["ntp_loss"]
        )
        candidate_long = float(
            row["final_long_context_heldout"]["ntp_loss"]
        )
        control_quality = first_quality_point(control["curve"], control_loss)
        candidate_quality = first_quality_point(row["curve"], control_loss)
        arm_regressions = {
            arm: (
                float(row["final_heldout"]["by_arm"][arm]["ntp_loss"])
                - float(control["final_heldout"]["by_arm"][arm]["ntp_loss"])
            )
            / max(
                float(control["final_heldout"]["by_arm"][arm]["ntp_loss"]),
                1e-12,
            )
            for arm in config["scoped_arms"]
        }
        paired.append(
            {
                "seed": int(row["seed"]),
                "final_loss_relative_regression": (
                    candidate_loss - control_loss
                )
                / max(control_loss, 1e-12),
                "long_context_loss_relative_regression": (
                    candidate_long - control_long
                )
                / max(control_long, 1e-12),
                "maximum_weak_arm_relative_loss_regression": max(
                    arm_regressions.values()
                ),
                "arm_relative_loss_regressions": arm_regressions,
                "joined_wall_time_ratio": float(
                    row["joined_training_wall_seconds"]
                )
                / max(
                    float(control["joined_training_wall_seconds"]), 1e-12
                ),
                "primary_step_time_ratio": float(
                    row["training_primary_step_seconds"]
                )
                / max(
                    float(control["training_primary_step_seconds"]), 1e-12
                ),
                "peak_allocator_ratio": float(row["peak_allocator_bytes"])
                / max(float(control["peak_allocator_bytes"]), 1.0),
                "optimizer_positions_equal": int(row["optimizer_positions"])
                == int(control["optimizer_positions"]),
                "candidate_quality_point": candidate_quality,
                "control_quality_point": control_quality,
                "time_to_control_final_quality_ratio": (
                    float(candidate_quality[1])
                    / max(float(control_quality[1]), 1e-12)
                    if candidate_quality is not None
                    and control_quality is not None
                    else None
                ),
            }
        )
    decision = config["decision"]
    time_ratios = [
        row["time_to_control_final_quality_ratio"] for row in paired
    ]
    gates = {
        "all_seeds": {row["seed"] for row in paired}
        == {int(seed) for seed in config["seeds"]},
        "matched_supervised_positions": all(
            row["optimizer_positions_equal"] for row in paired
        ),
        "joined_wall": statistics.fmean(
            row["joined_wall_time_ratio"] for row in paired
        )
        <= float(decision["maximum_mean_joined_wall_time_ratio"]),
        "time_to_quality": all(value is not None for value in time_ratios)
        and statistics.fmean(float(value) for value in time_ratios)
        <= float(
            decision["maximum_mean_time_to_control_final_quality_ratio"]
        ),
        "final_loss": statistics.fmean(
            row["final_loss_relative_regression"] for row in paired
        )
        <= float(decision["maximum_mean_final_loss_relative_regression"]),
        "long_context_loss": max(
            row["long_context_loss_relative_regression"] for row in paired
        )
        <= float(
            decision["maximum_long_context_loss_relative_regression"]
        ),
        "weak_arm": max(
            row["maximum_weak_arm_relative_loss_regression"]
            for row in paired
        )
        <= float(decision["maximum_weak_arm_relative_loss_regression"]),
        "transition_stability": max(
            transition["relative_loss_jump"]
            for row in candidates
            for transition in row["transition_receipts"]
        )
        <= float(decision["maximum_transition_relative_loss_jump"]),
        "checkpoint_replay": all(
            transition["checkpoint_replay"]["exact_model_reload"]
            and transition["checkpoint_replay"]["exact_optimizer_reload"]
            and transition["checkpoint_replay"][
                "next_update_numerically_equivalent"
            ]
            for row in candidates
            for transition in row["transition_receipts"]
        )
        and all(
            row["final_checkpoint_replay"]["exact_model_reload"]
            and row["final_checkpoint_replay"]["exact_optimizer_reload"]
            and row["final_checkpoint_replay"][
                "next_update_numerically_equivalent"
            ]
            for row in candidates
        ),
        "finite_gradients": all(row["finite_gradients"] for row in candidates),
    }
    adopted = all(gates.values())
    comparison = {
        "control_id": "fixed_512_control",
        "candidate_id": "progressive_128_256_512",
        "paired_runs": paired,
        "mean_joined_wall_time_ratio": statistics.fmean(
            row["joined_wall_time_ratio"] for row in paired
        ),
        "mean_primary_step_time_ratio": statistics.fmean(
            row["primary_step_time_ratio"] for row in paired
        ),
        "mean_final_loss_relative_regression": statistics.fmean(
            row["final_loss_relative_regression"] for row in paired
        ),
        "mean_time_to_control_final_quality_ratio": (
            statistics.fmean(float(value) for value in time_ratios)
            if all(value is not None for value in time_ratios)
            else None
        ),
        "gates": gates,
        "disposition": (
            "ADOPTED_FOR_TARGET_TRANSFER"
            if adopted
            else "NOT_SELECTED_FIRST_CAMPAIGN"
        ),
        "scientific_falsification_claimed": False,
    }
    disposition = {
        "selected_sequence_policy": (
            "progressive_128_256_512" if adopted else "fixed_512_control"
        ),
        "kind": (
            "PROGRESSIVE_SEQUENCE_SELECTED"
            if adopted
            else "FIXED_SEQUENCE_CONTROL_RETAINED"
        ),
        "scientific_falsification_claimed": False,
        "reentry_condition": (
            None
            if adopted
            else "a new prospective corpus/schedule rung that passes matched-token, long-context, weak-tail, replay, and joined-time gates"
        ),
    }
    return comparison, disposition


def prepare_journal(
    config_path: Path, scratch: Path
) -> tuple[Path, list[dict[str, Any]], bool]:
    contract_path = scratch / "run_journal.contract.json"
    journal_path = scratch / "run_journal.jsonl"
    expected = {
        "policy": "project_theseus_progressive_sequence_journal_v1",
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "model_implementation_sha256": sha256_file(
            ROOT / "scripts/standard_causal_transformer_model.py"
        ),
    }
    if contract_path.is_file() and journal_path.is_file():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed == expected:
            return (
                journal_path,
                [
                    json.loads(line)
                    for line in journal_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ],
                True,
            )
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    contract_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    journal_path.touch()
    return journal_path, [], False


def execute(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    metadata = json.loads(
        resolve(config["stage_metadata"]).read_text(encoding="utf-8")
    )
    base = json.loads(resolve(config["base_config"]).read_text(encoding="utf-8"))
    vocabulary = model_vocab_size(
        base, metadata["source_vocab"], metadata["target_vocab"]
    )
    train_rows = corpus.load_governed_rows(config, split="train")
    heldout = corpus.load_governed_rows(config, split="heldout")
    train_ids = optimizer_adequacy.source_sets(train_rows)
    heldout_ids = optimizer_adequacy.source_sets(heldout)
    if train_ids & heldout_ids:
        raise ProgressiveLengthFault("source_disjointness_failed")
    heldout_long = long_context_rows(
        heldout, int(config["long_context_threshold_tokens"])
    )
    if any(not rows for rows in heldout_long.values()):
        raise ProgressiveLengthFault("long_context_arm_coverage_missing")
    steps = int(config["training"]["steps"])
    logical_batches_by_seed = {
        int(seed): corpus.balanced_batches(
            train_rows, steps=steps, seed=int(seed)
        )
        for seed in config["seeds"]
    }
    context_fraction = float(config["windowing"]["context_fraction"])
    widths = [128, 256, 512]
    pad_sizes = {
        width: max(
            len(
                window_logical_batch(
                    batch, width, context_fraction=context_fraction
                )
            )
            for batches in logical_batches_by_seed.values()
            for batch in batches
        )
        for width in widths
    }
    supervision_audit = {
        str(seed): {
            str(width): {
                "logical_batch_count": len(batches),
                "maximum_window_rows_per_batch": pad_sizes[width],
                "all_batch_target_positions_preserved": all(
                    sum(
                        target_position_count(window)
                        for window in window_logical_batch(
                            batch,
                            width,
                            context_fraction=context_fraction,
                        )
                    )
                    == sum(target_position_count(row) for row in batch)
                    for batch in batches
                ),
            }
            for width in widths
        }
        for seed, batches in logical_batches_by_seed.items()
    }
    if not all(
        row["all_batch_target_positions_preserved"]
        for seed_row in supervision_audit.values()
        for row in seed_row.values()
    ):
        raise ProgressiveLengthFault("supervision_audit_failed")
    scratch = resolve(config["scratch_root"])
    journal_path, journal_rows, resumed = prepare_journal(
        config_path, scratch
    )
    index = {
        (str(row["run_id"]), int(row["seed"])): row for row in journal_rows
    }
    run_ids = ["fixed_512_control", "progressive_128_256_512"]
    runs = []
    run_order = {}
    for seed_index, seed in enumerate(config["seeds"]):
        order = run_ids if seed_index % 2 == 0 else list(reversed(run_ids))
        run_order[str(seed)] = order
        for run_id in order:
            key = (run_id, int(seed))
            row = index.get(key)
            if row is None:
                row = run_one(
                    config,
                    run_id=run_id,
                    seed=int(seed),
                    logical_batches=logical_batches_by_seed[int(seed)],
                    pad_sizes=pad_sizes,
                    heldout=heldout,
                    heldout_long=heldout_long,
                    vocabulary=vocabulary,
                    scratch=scratch,
                )
                with journal_path.open("a", encoding="utf-8") as handle:
                    handle.write(canonical(row) + "\n")
                    handle.flush()
                index[key] = row
            runs.append(row)
    for seed in config["seeds"]:
        identities = {
            row["initial_parameter_sha256"]
            for row in runs
            if int(row["seed"]) == int(seed)
        }
        if len(identities) != 1:
            raise ProgressiveLengthFault(
                f"matched_initialization_failed:{seed}"
            )
    comparison, disposition = compare(runs, config)
    local_config = growth.model_config(
        config, vocabulary, depth=int(config["model"]["num_layers"])
    )
    report = {
        "policy": POLICY,
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "model_implementation_sha256": sha256_file(
            ROOT / "scripts/standard_causal_transformer_model.py"
        ),
        "support_state": "SUPPORTED",
        "trigger_state": "GREEN",
        "model_parameter_count": sum(
            analytical_parameter_breakdown(local_config).values()
        ),
        "sequence_distribution": {
            arm: {
                "row_count": len(rows),
                "minimum_tokens": min(len(row["sequence"]) - 1 for row in rows),
                "median_tokens": statistics.median(
                    len(row["sequence"]) - 1 for row in rows
                ),
                "maximum_tokens": max(len(row["sequence"]) - 1 for row in rows),
            }
            for arm, rows in train_rows.items()
        },
        "source_disjointness": {
            "train_source_count": len(train_ids),
            "heldout_source_count": len(heldout_ids),
            "intersection_count": len(train_ids & heldout_ids),
            "passed": not bool(train_ids & heldout_ids),
        },
        "long_context_heldout_rows_by_arm": {
            arm: len(rows) for arm, rows in heldout_long.items()
        },
        "compiled_pad_sizes": pad_sizes,
        "supervision_conservation_audit": supervision_audit,
        "run_order_by_seed": run_order,
        "resumed_from_journal": resumed,
        "runs": runs,
        "comparison": comparison,
        "campaign_disposition": disposition,
        "hard_boundaries": config["hard_boundaries"],
        "external_source_basis": {
            "paper": "https://arxiv.org/abs/2310.00576",
            "local_adaptation": (
                "128 to 256 to 512 RoPE curriculum; each original supervised "
                "target position appears exactly once per logical batch at "
                "every width; source-conditioned rows retain bounded left "
                "context and use stable padded batch shapes"
            ),
        },
        "non_claims": [
            "No capability, full-campaign, architecture-superiority, AGI, or ASI claim.",
            "A failed curriculum is scoped to this data distribution, windowing policy, model, optimizer, budget, evaluator, and host.",
            "Engineering exclusion is not scientific falsification of GrowLength.",
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
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    report = execute(args.config.resolve())
    if args.summary_only:
        print(
            json.dumps(
                {
                    "trigger_state": report["trigger_state"],
                    "campaign_disposition": report["campaign_disposition"],
                    "comparison": report["comparison"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
