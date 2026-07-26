#!/usr/bin/env python3
"""Matched MLX qualification for function-preserving staged decoder growth.

This is an engineering selection rung, not a capability benchmark.  It trains
smaller depth-compatible subnetworks first, inserts canonically initialized
blocks behind zero residual masks, transfers AdamW state for every surviving
parameter, and compares joined time-to-private-heldout-quality with a matched
full-depth control.
"""

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

import mtp_matched_adequacy as corpus
import optimizer_matched_adequacy as optimizer_adequacy
import pretraining_optimizers
from standard_causal_transformer_model import (
    CausalTransformerConfig,
    analytical_parameter_breakdown,
    build_model,
)
from standard_causal_transformer_survival import (
    SOURCE_TARGET_SEPARATOR_ID,
    model_vocab_size,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/masked_structural_growth_qualification.json"
POLICY = "project_theseus_masked_structural_growth_qualification_v1"


class StructuralGrowthFault(ValueError):
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
        raise StructuralGrowthFault("config_policy_invalid")
    final_depth = int(config["model"]["num_layers"])
    steps = int(config["training"]["steps"])
    schedules = config.get("growth_schedules") or []
    if len(schedules) < 1:
        raise StructuralGrowthFault("growth_schedule_missing")
    ids: set[str] = set()
    for schedule in schedules:
        schedule_id = str(schedule.get("id") or "")
        if not schedule_id or schedule_id in ids:
            raise StructuralGrowthFault("growth_schedule_id_invalid")
        ids.add(schedule_id)
        stages = schedule.get("stages") or []
        if not stages or int(stages[-1]["stop_step"]) != steps:
            raise StructuralGrowthFault(
                f"growth_schedule_final_step_invalid:{schedule_id}"
            )
        prior_stop = 0
        prior_slots: list[int] = []
        for stage in stages:
            slots = [int(value) for value in stage.get("full_layer_slots") or []]
            stop = int(stage.get("stop_step") or 0)
            if (
                not slots
                or slots != sorted(set(slots))
                or min(slots) < 0
                or max(slots) >= final_depth
                or stop <= prior_stop
                or not set(prior_slots).issubset(slots)
            ):
                raise StructuralGrowthFault(
                    f"growth_schedule_stage_invalid:{schedule_id}"
                )
            stack_sources = {
                int(target): int(source)
                for target, source in (
                    stage.get("new_layer_source_local_indices") or {}
                ).items()
            }
            new_local_indices = {
                index
                for index, slot in enumerate(slots)
                if slot not in set(prior_slots)
            }
            if prior_slots and (
                set(stack_sources) != new_local_indices
                or any(
                    source < 0 or source >= len(prior_slots)
                    for source in stack_sources.values()
                )
            ):
                raise StructuralGrowthFault(
                    f"growth_schedule_stacking_invalid:{schedule_id}"
                )
            if not prior_slots and stack_sources:
                raise StructuralGrowthFault(
                    f"growth_schedule_initial_stacking_invalid:{schedule_id}"
                )
            prior_slots, prior_stop = slots, stop
        if prior_slots != list(range(final_depth)):
            raise StructuralGrowthFault(
                f"growth_schedule_final_depth_invalid:{schedule_id}"
            )
        if int(schedule.get("mask_ramp_steps") or 0) <= 0:
            raise StructuralGrowthFault(
                f"growth_schedule_mask_ramp_invalid:{schedule_id}"
            )
    if len(set(int(seed) for seed in config.get("seeds") or [])) < 3:
        raise StructuralGrowthFault("three_distinct_seeds_required")
    if steps <= 0 or steps > 192:
        raise StructuralGrowthFault("step_budget_invalid")
    if int(config["training"]["evaluation_interval_steps"]) <= 0:
        raise StructuralGrowthFault("evaluation_interval_invalid")
    if float(config["training"]["gradient_clip_norm"]) <= 0.0:
        raise StructuralGrowthFault("gradient_clip_invalid")
    boundaries = config.get("hard_boundaries") or {}
    for field in (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "fallback_or_template_credit",
        "confirmation_surface_consumption",
    ):
        if boundaries.get(field) != 0:
            raise StructuralGrowthFault(f"hard_boundary_nonzero:{field}")
    for field in (
        "production_checkpoint_mutation",
        "heldout_labels_visible_to_candidate",
        "selection_from_throughput_alone",
    ):
        if boundaries.get(field) is not False:
            raise StructuralGrowthFault(f"hard_boundary_boolean_invalid:{field}")
    return config


def model_config(
    config: dict[str, Any], vocab_size: int, *, depth: int
) -> CausalTransformerConfig:
    row = config["model"]
    return CausalTransformerConfig(
        vocab_size=vocab_size,
        d_model=int(row["d_model"]),
        num_layers=int(depth),
        num_heads=int(row["num_heads"]),
        num_kv_heads=int(row["num_kv_heads"]),
        ff_dim=int(row["ff_dim"]),
        attention_policy=str(row["attention_policy"]),
        source_target_separator_token_id=SOURCE_TARGET_SEPARATOR_ID,
    )


def split_layer_path(name: str) -> tuple[int, str] | None:
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "layers":
        return None
    try:
        return int(parts[1]), ".".join(parts[2:])
    except ValueError:
        return None


def stage_parameter_sources(
    target_names: list[str],
    *,
    target_slots: list[int],
    prior_slots: list[int] | None,
    new_layer_source_local_indices: dict[int, int] | None = None,
) -> dict[str, tuple[str, str]]:
    """Return target -> (source kind, source path) without touching MLX."""

    prior_index = {
        slot: index for index, slot in enumerate(prior_slots or [])
    }
    result: dict[str, tuple[str, str]] = {}
    for target_name in target_names:
        layer = split_layer_path(target_name)
        if layer is None:
            result[target_name] = (
                "prior" if prior_slots is not None else "canonical",
                target_name,
            )
            continue
        local_index, suffix = layer
        full_slot = target_slots[local_index]
        if full_slot in prior_index:
            result[target_name] = (
                "prior",
                f"layers.{prior_index[full_slot]}.{suffix}",
            )
        elif (
            prior_slots is not None
            and new_layer_source_local_indices is not None
            and local_index in new_layer_source_local_indices
        ):
            result[target_name] = (
                "stacked_prior",
                f"layers.{new_layer_source_local_indices[local_index]}.{suffix}",
            )
        else:
            result[target_name] = (
                "canonical",
                f"layers.{full_slot}.{suffix}",
            )
    return result


def stage_optimizer_sources(
    target_names: list[str],
    *,
    target_slots: list[int],
    prior_slots: list[int],
) -> dict[str, str | None]:
    """Map target AdamW state names to prior names; None means fresh zero state."""

    prior_index = {slot: index for index, slot in enumerate(prior_slots)}
    result: dict[str, str | None] = {}
    for target_name in target_names:
        layer = split_layer_path(target_name)
        if layer is None:
            result[target_name] = target_name
            continue
        local_index, suffix = layer
        full_slot = target_slots[local_index]
        result[target_name] = (
            f"layers.{prior_index[full_slot]}.{suffix}"
            if full_slot in prior_index
            else None
        )
    return result


def update_model_for_stage(
    model: Any,
    *,
    canonical_parameters: Any,
    target_slots: list[int],
    prior_model: Any | None,
    prior_slots: list[int] | None,
    new_layer_source_local_indices: dict[int, int] | None = None,
    mlx_utils: Any,
) -> dict[str, int]:
    canonical = dict(mlx_utils.tree_flatten(canonical_parameters))
    prior = (
        dict(mlx_utils.tree_flatten(prior_model.parameters()))
        if prior_model is not None
        else {}
    )
    target = dict(mlx_utils.tree_flatten(model.parameters()))
    sources = stage_parameter_sources(
        list(target),
        target_slots=target_slots,
        prior_slots=prior_slots,
        new_layer_source_local_indices=new_layer_source_local_indices,
    )
    output = []
    canonical_count = 0
    prior_count = 0
    for target_name in target:
        kind, source_name = sources[target_name]
        source = (
            prior if kind in {"prior", "stacked_prior"} else canonical
        )
        if source_name not in source:
            raise StructuralGrowthFault(
                f"stage_parameter_source_missing:{source_name}"
            )
        output.append((target_name, source[source_name]))
        canonical_count += kind == "canonical"
        prior_count += kind in {"prior", "stacked_prior"}
    model.update(mlx_utils.tree_unflatten(output))
    return {
        "canonical_parameter_leaves": canonical_count,
        "transferred_parameter_leaves": prior_count,
        "stacked_new_layer_parameter_leaves": sum(
            kind == "stacked_prior" for kind, _source in sources.values()
        ),
    }


def transfer_adamw_state(
    target_optimizer: Any,
    prior_optimizer: Any,
    *,
    target_slots: list[int],
    prior_slots: list[int],
    mlx_utils: Any,
) -> dict[str, int]:
    target = dict(mlx_utils.tree_flatten(target_optimizer.state))
    prior = dict(mlx_utils.tree_flatten(prior_optimizer.state))
    sources = stage_optimizer_sources(
        list(target), target_slots=target_slots, prior_slots=prior_slots
    )
    output = []
    copied = 0
    zero_retained = 0
    for target_name, initial_value in target.items():
        source_name = sources[target_name]
        if source_name is None:
            output.append((target_name, initial_value))
            zero_retained += 1
        else:
            if source_name not in prior:
                raise StructuralGrowthFault(
                    f"stage_optimizer_source_missing:{source_name}"
                )
            output.append((target_name, prior[source_name]))
            copied += 1
    target_optimizer.state = mlx_utils.tree_unflatten(output)
    return {
        "transferred_optimizer_leaves": copied,
        "new_zero_optimizer_leaves": zero_retained,
    }


def tree_digest(tree: Any, mlx_utils: Any) -> str:
    digest = hashlib.sha256()
    for name, value in mlx_utils.tree_flatten(tree):
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def tree_max_difference(left: Any, right: Any, mlx_utils: Any) -> float:
    left_flat = mlx_utils.tree_flatten(left)
    right_flat = mlx_utils.tree_flatten(right)
    if [name for name, _value in left_flat] != [
        name for name, _value in right_flat
    ]:
        return float("inf")
    return max(
        (
            float(
                np.max(
                    np.abs(
                        np.asarray(left_value, dtype=np.float64)
                        - np.asarray(right_value, dtype=np.float64)
                    )
                )
            )
            if int(left_value.size)
            else 0.0
        )
        for (_left_name, left_value), (_right_name, right_value) in zip(
            left_flat, right_flat
        )
    )


def save_tree(
    path: Path, tree: Any, *, mx: Any, mlx_utils: Any
) -> Path:
    flat = mlx_utils.tree_flatten(tree)
    mx.savez(
        str(path),
        **{f"a{index}": value for index, (_name, value) in enumerate(flat)},
    )
    names = path.with_suffix(".names.json")
    names.write_text(
        json.dumps([name for name, _value in flat]) + "\n", encoding="utf-8"
    )
    return names


def load_tree(path: Path, names: Path, *, mx: Any, mlx_utils: Any) -> Any:
    loaded = mx.load(str(path))
    paths = json.loads(names.read_text(encoding="utf-8"))
    return mlx_utils.tree_unflatten(
        [(name, loaded[f"a{index}"]) for index, name in enumerate(paths)]
    )


def build_adamw(config: dict[str, Any], *, mx: Any, optim: Any) -> Any:
    return pretraining_optimizers.build_optimizer(
        "adamw_mlx",
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        warmup_steps=0,
        optim=optim,
        mx=mx,
    )


def schedule_stage(
    schedule: dict[str, Any], global_step: int
) -> tuple[int, dict[str, Any]]:
    for index, stage in enumerate(schedule["stages"]):
        if global_step <= int(stage["stop_step"]):
            return index, stage
    raise StructuralGrowthFault("schedule_step_out_of_range")


def stage_masks(
    schedule: dict[str, Any], stage_index: int, global_step: int
) -> tuple[float, ...]:
    stage = schedule["stages"][stage_index]
    slots = [int(value) for value in stage["full_layer_slots"]]
    if stage_index == 0:
        return tuple(1.0 for _slot in slots)
    prior_slots = {
        int(value)
        for value in schedule["stages"][stage_index - 1]["full_layer_slots"]
    }
    transition_step = (
        int(schedule["stages"][stage_index - 1]["stop_step"]) + 1
    )
    ramp = int(schedule["mask_ramp_steps"])
    value = min(1.0, max(0.0, (global_step - transition_step + 1) / ramp))
    return tuple(1.0 if slot in prior_slots else value for slot in slots)


def boundary_masks(
    schedule: dict[str, Any], stage_index: int
) -> tuple[float, ...]:
    slots = [
        int(value)
        for value in schedule["stages"][stage_index]["full_layer_slots"]
    ]
    prior_slots = {
        int(value)
        for value in schedule["stages"][stage_index - 1]["full_layer_slots"]
    }
    return tuple(1.0 if slot in prior_slots else 0.0 for slot in slots)


def make_compiled_step(model: Any, optimizer: Any, *, mx: Any, nn: Any, optim: Any, mlx_utils: Any, clip: float) -> Any:
    def objective(
        local_model: Any, x: Any, y: Any, mask: Any, growth_masks: Any
    ) -> Any:
        logits, _cache = local_model(
            x, structural_growth_masks=growth_masks
        )
        losses = nn.losses.cross_entropy(logits, y)
        denominator = mx.maximum(
            mx.sum(mask), mx.array(1.0, dtype=mx.float32)
        )
        return mx.sum(losses * mask) / denominator

    value_and_grad = nn.value_and_grad(model, objective)
    compiled_state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=compiled_state, outputs=compiled_state)
    def compiled_step(
        x: Any, y: Any, mask: Any, growth_masks: Any
    ) -> tuple[Any, Any, Any]:
        loss, gradients = value_and_grad(model, x, y, mask, growth_masks)
        gradient_l1 = sum(
            mx.sum(mx.abs(value))
            for _name, value in mlx_utils.tree_flatten(gradients)
        )
        gradients, gradient_norm = optim.clip_grad_norm(gradients, clip)
        optimizer.update(model, gradients)
        return loss, gradient_norm, gradient_l1

    return compiled_step


def evaluate(
    model: Any,
    masks: tuple[float, ...],
    heldout: dict[str, list[dict[str, Any]]],
    maximum: int,
    *,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    model.set_structural_growth_masks(masks)
    result = corpus._evaluate(model, heldout, maximum, mx, nn)
    model.set_structural_growth_masks(tuple(1.0 for _value in masks))
    return result


def peak_memory_bytes(mx: Any) -> int | None:
    getter = getattr(mx, "get_peak_memory", None)
    if getter is None:
        metal = getattr(mx, "metal", None)
        getter = getattr(metal, "get_peak_memory", None)
    return int(getter()) if getter is not None else None


def reset_peak_memory(mx: Any) -> None:
    reset = getattr(mx, "reset_peak_memory", None)
    if reset is None:
        metal = getattr(mx, "metal", None)
        reset = getattr(metal, "reset_peak_memory", None)
    if reset is not None:
        reset()


def checkpoint_replay(
    *,
    model: Any,
    optimizer: Any,
    local_config: CausalTransformerConfig,
    masks: tuple[float, ...],
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
    retain_primary_update: bool = True,
) -> tuple[dict[str, Any], float, float]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_root / f"{checkpoint_name}.model.npz"
    optimizer_path = checkpoint_root / f"{checkpoint_name}.optimizer.npz"
    model_names = save_tree(
        model_path, model.parameters(), mx=mx, mlx_utils=mlx_utils
    )
    optimizer_names = save_tree(
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
        load_tree(model_path, model_names, mx=mx, mlx_utils=mlx_utils)
    )
    clone_optimizer = build_adamw(config, mx=mx, optim=optim)
    clone_optimizer.state = load_tree(
        optimizer_path, optimizer_names, mx=mx, mlx_utils=mlx_utils
    )
    clone_optimizer.init(clone.trainable_parameters())
    mx.eval(clone.parameters(), clone_optimizer.state)
    exact_model = tree_digest(model.parameters(), mlx_utils) == tree_digest(
        clone.parameters(), mlx_utils
    )
    exact_optimizer = tree_digest(
        optimizer.state, mlx_utils
    ) == tree_digest(clone_optimizer.state, mlx_utils)
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
    growth = mx.array(masks, dtype=mx.float32)
    primary_started = time.perf_counter()
    primary_loss, primary_norm, primary_l1 = primary_step(
        x, y, mask, growth
    )
    mx.eval(
        primary_loss,
        primary_norm,
        primary_l1,
        model.parameters(),
        optimizer.state,
    )
    primary_seconds = time.perf_counter() - primary_started
    audit_started = time.perf_counter()
    clone_loss, clone_norm, clone_l1 = clone_step(x, y, mask, growth)
    mx.eval(
        clone_loss,
        clone_norm,
        clone_l1,
        clone.parameters(),
        clone_optimizer.state,
    )
    audit_seconds = time.perf_counter() - audit_started
    next_model_error = tree_max_difference(
        model.parameters(), clone.parameters(), mlx_utils
    )
    next_optimizer_error = tree_max_difference(
        optimizer.state, clone_optimizer.state, mlx_utils
    )
    next_update_tolerance = float(
        config["decision"]["maximum_next_update_absolute_error"]
    )
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
        "next_gradient_l1_absolute_error": abs(
            float(primary_l1.item()) - float(clone_l1.item())
        ),
        "next_model_max_absolute_error": next_model_error,
        "next_optimizer_max_absolute_error": next_optimizer_error,
        "maximum_allowed_next_update_absolute_error": next_update_tolerance,
        "primary_loss": float(primary_loss.item()),
        "primary_gradient_norm": float(primary_norm.item()),
        "primary_gradient_l1": float(primary_l1.item()),
        "exact_next_update": (
            next_model_error == 0.0
            and next_optimizer_error == 0.0
            and float(primary_loss.item()) == float(clone_loss.item())
        ),
        "next_update_numerically_equivalent": (
            next_model_error <= next_update_tolerance
            and next_optimizer_error <= next_update_tolerance
            and float(primary_loss.item()) == float(clone_loss.item())
            and float(primary_norm.item()) == float(clone_norm.item())
        ),
    }
    if not retain_primary_update:
        model.update(
            load_tree(model_path, model_names, mx=mx, mlx_utils=mlx_utils)
        )
        optimizer.state = load_tree(
            optimizer_path,
            optimizer_names,
            mx=mx,
            mlx_utils=mlx_utils,
        )
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)
    for path in (
        model_path,
        optimizer_path,
        model_names,
        optimizer_names,
    ):
        path.unlink(missing_ok=True)
    return (
        receipt,
        primary_seconds if retain_primary_update else 0.0,
        (
            audit_seconds
            if retain_primary_update
            else primary_seconds + audit_seconds
        ),
    )


def run_one(
    config: dict[str, Any],
    *,
    run_id: str,
    schedule: dict[str, Any] | None,
    seed: int,
    train_rows: dict[str, list[dict[str, Any]]],
    heldout: dict[str, list[dict[str, Any]]],
    batches: list[list[dict[str, Any]]],
    vocabulary: int,
    scratch: Path,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    steps = int(config["training"]["steps"])
    maximum = int(config["supervision"]["maximum_sequence_tokens"])
    full_depth = int(config["model"]["num_layers"])
    reset_peak_memory(mx)
    mx.random.seed(int(seed))
    run_started = time.perf_counter()
    canonical_model = build_model(
        model_config(config, vocabulary, depth=full_depth), mx=mx, nn=nn
    )
    canonical_parameters = canonical_model.parameters()
    canonical_sha256 = tree_digest(canonical_parameters, mlx_utils)
    if schedule is None:
        model = canonical_model
        slots = list(range(full_depth))
    else:
        slots = [
            int(value)
            for value in schedule["stages"][0]["full_layer_slots"]
        ]
        model = build_model(
            model_config(config, vocabulary, depth=len(slots)), mx=mx, nn=nn
        )
        update_model_for_stage(
            model,
            canonical_parameters=canonical_parameters,
            target_slots=slots,
            prior_model=None,
            prior_slots=None,
            new_layer_source_local_indices=None,
            mlx_utils=mlx_utils,
        )
    optimizer = build_adamw(config, mx=mx, optim=optim)
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    current_stage = 0
    current_masks = tuple(1.0 for _slot in slots)
    compiled_step = make_compiled_step(
        model,
        optimizer,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
        clip=float(config["training"]["gradient_clip_norm"]),
    )
    initial = evaluate(
        model, current_masks, heldout, maximum, mx=mx, nn=nn
    )
    curve = [
        {
            "step": 0,
            "heldout": initial,
            "joined_wall_seconds": time.perf_counter() - run_started,
        }
    ]
    transitions: list[dict[str, Any]] = []
    finite_gradients = True
    first_gradient_l1 = None
    optimizer_positions = 0
    audit_seconds = 0.0
    primary_step_seconds = 0.0
    for global_step, batch in enumerate(batches, 1):
        x_np, y_np, mask_np = corpus.make_batch(batch, maximum)
        x, y, mask = mx.array(x_np), mx.array(y_np), mx.array(mask_np)
        transition = False
        if schedule is not None:
            target_stage, stage = schedule_stage(schedule, global_step)
            if target_stage != current_stage:
                transition = True
                prior_model, prior_optimizer, prior_slots = (
                    model,
                    optimizer,
                    slots,
                )
                slots = [
                    int(value) for value in stage["full_layer_slots"]
                ]
                model = build_model(
                    model_config(config, vocabulary, depth=len(slots)),
                    mx=mx,
                    nn=nn,
                )
                transfer_inventory = update_model_for_stage(
                    model,
                    canonical_parameters=canonical_parameters,
                    target_slots=slots,
                    prior_model=prior_model,
                    prior_slots=prior_slots,
                    new_layer_source_local_indices={
                        int(target): int(source)
                        for target, source in (
                            stage.get("new_layer_source_local_indices") or {}
                        ).items()
                    },
                    mlx_utils=mlx_utils,
                )
                optimizer = build_adamw(config, mx=mx, optim=optim)
                optimizer.init(model.trainable_parameters())
                optimizer_inventory = transfer_adamw_state(
                    optimizer,
                    prior_optimizer,
                    target_slots=slots,
                    prior_slots=prior_slots,
                    mlx_utils=mlx_utils,
                )
                mx.eval(model.parameters(), optimizer.state)
                boundary = boundary_masks(schedule, target_stage)
                prior_logits, _cache = prior_model(x)
                # Use the hard eager zero branch for the formal insertion
                # boundary.  A tensor expression ``x + 0 * delta`` is
                # mathematically identical but can alter Metal graph fusion
                # upstream by a few ulps; that is suitable for the continuous
                # ramp, not for the exact function-preservation claim.
                model.set_structural_growth_masks(boundary)
                next_logits, _cache = model(x)
                model.set_structural_growth_masks(
                    tuple(1.0 for _slot in slots)
                )
                prior_losses = nn.losses.cross_entropy(
                    prior_logits, y
                )
                next_losses = nn.losses.cross_entropy(next_logits, y)
                denominator = mx.maximum(
                    mx.sum(mask), mx.array(1.0, dtype=mx.float32)
                )
                prior_loss = mx.sum(prior_losses * mask) / denominator
                next_loss = mx.sum(next_losses * mask) / denominator
                mx.eval(prior_logits, next_logits, prior_loss, next_loss)
                output_error = float(
                    mx.max(mx.abs(prior_logits - next_logits)).item()
                )
                loss_error = abs(
                    float(prior_loss.item()) - float(next_loss.item())
                )
                current_stage = target_stage
                current_masks = stage_masks(
                    schedule, current_stage, global_step
                )
                replay, replay_primary_seconds, replay_audit_seconds = checkpoint_replay(
                    model=model,
                    optimizer=optimizer,
                    local_config=model_config(
                        config, vocabulary, depth=len(slots)
                    ),
                    masks=current_masks,
                    x=x,
                    y=y,
                    mask=mask,
                    checkpoint_root=scratch / f"{run_id}_{seed}",
                    checkpoint_name=f"transition_{global_step}",
                    config=config,
                    mx=mx,
                    nn=nn,
                    optim=optim,
                    mlx_utils=mlx_utils,
                )
                audit_seconds += replay_audit_seconds
                transitions.append(
                    {
                        "global_step": global_step,
                        "prior_full_layer_slots": prior_slots,
                        "target_full_layer_slots": slots,
                        "boundary_masks": boundary,
                        "first_update_masks": current_masks,
                        "boundary_output_max_absolute_error": output_error,
                        "boundary_loss_absolute_error": loss_error,
                        "function_preserving": output_error
                        <= float(
                            config["decision"][
                                "maximum_boundary_output_absolute_error"
                            ]
                        )
                        and loss_error
                        <= float(
                            config["decision"][
                                "maximum_boundary_loss_absolute_error"
                            ]
                        ),
                        "parameter_transfer": transfer_inventory,
                        "optimizer_transfer": optimizer_inventory,
                        "checkpoint_replay": replay,
                    }
                )
                primary_step_seconds += replay_primary_seconds
                compiled_step = make_compiled_step(
                    model,
                    optimizer,
                    mx=mx,
                    nn=nn,
                    optim=optim,
                    mlx_utils=mlx_utils,
                    clip=float(config["training"]["gradient_clip_norm"]),
                )
            else:
                current_masks = stage_masks(
                    schedule, current_stage, global_step
                )
        if transition:
            replay = transitions[-1]["checkpoint_replay"]
            loss_value = float(replay["primary_loss"])
            gradient_norm_value = float(replay["primary_gradient_norm"])
            gradient_l1_value = float(replay["primary_gradient_l1"])
        else:
            growth = mx.array(current_masks, dtype=mx.float32)
            step_started = time.perf_counter()
            loss, gradient_norm, gradient_l1 = compiled_step(
                x, y, mask, growth
            )
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
        finite_gradients = finite_gradients and math.isfinite(loss_value)
        finite_gradients = (
            finite_gradients
            and math.isfinite(gradient_norm_value)
            and gradient_l1_value is not None
            and math.isfinite(gradient_l1_value)
            and gradient_l1_value > 0.0
        )
        if first_gradient_l1 is None:
            first_gradient_l1 = gradient_l1_value
        optimizer_positions += int(mask_np.sum())
        if (
            global_step
            % int(config["training"]["evaluation_interval_steps"])
            == 0
            or global_step == steps
        ):
            heldout_result = evaluate(
                model,
                current_masks,
                heldout,
                maximum,
                mx=mx,
                nn=nn,
            )
            curve.append(
                {
                    "step": global_step,
                    "heldout": heldout_result,
                    "joined_wall_seconds": (
                        time.perf_counter() - run_started - audit_seconds
                    ),
                }
            )
    final_checkpoint_masks = current_masks
    final_x_np, final_y_np, final_mask_np = corpus.make_batch(
        batches[-1], maximum
    )
    final_replay, _final_primary_seconds, final_audit_seconds = checkpoint_replay(
        model=model,
        optimizer=optimizer,
        local_config=model_config(config, vocabulary, depth=len(slots)),
        masks=final_checkpoint_masks,
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
    audit_seconds += final_audit_seconds
    total_wall = time.perf_counter() - run_started
    joined_wall = total_wall - audit_seconds
    final = curve[-1]["heldout"]
    return {
        "run_id": run_id,
        "kind": "full_depth_control" if schedule is None else "masked_structural_growth",
        "schedule": schedule,
        "seed": int(seed),
        "model": asdict(model_config(config, vocabulary, depth=full_depth)),
        "canonical_initial_parameter_sha256": canonical_sha256,
        "optimizer_id": "adamw_mlx",
        "optimizer_steps": steps,
        "optimizer_positions": optimizer_positions,
        "initial_heldout": initial,
        "final_heldout": final,
        "curve": curve,
        "finite_gradients": finite_gradients,
        "first_gradient_l1": first_gradient_l1,
        "transitions": transitions,
        "final_checkpoint_replay": final_replay,
        "training_primary_step_seconds": primary_step_seconds,
        "joined_training_wall_seconds": joined_wall,
        "qualification_total_wall_seconds": total_wall,
        "excluded_audit_twin_seconds": audit_seconds,
        "peak_allocator_bytes": peak_memory_bytes(mx),
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
        if row["kind"] == "full_depth_control"
    }
    comparisons: dict[str, Any] = {}
    eligible: list[tuple[float, str]] = []
    for schedule in config["growth_schedules"]:
        schedule_id = schedule["id"]
        candidate_runs = [
            row for row in runs if row["run_id"] == schedule_id
        ]
        paired = []
        for row in candidate_runs:
            control = controls[int(row["seed"])]
            control_loss = float(control["final_heldout"]["ntp_loss"])
            candidate_loss = float(row["final_heldout"]["ntp_loss"])
            control_quality = first_quality_point(
                control["curve"], control_loss
            )
            candidate_quality = first_quality_point(
                row["curve"], control_loss
            )
            arm_regressions = {
                arm: (
                    float(row["final_heldout"]["by_arm"][arm]["ntp_loss"])
                    - float(control["final_heldout"]["by_arm"][arm]["ntp_loss"])
                )
                / max(
                    float(
                        control["final_heldout"]["by_arm"][arm]["ntp_loss"]
                    ),
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
                    "control_quality_point": control_quality,
                    "candidate_quality_point": candidate_quality,
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
            "joined_wall": statistics.fmean(
                row["joined_wall_time_ratio"] for row in paired
            )
            <= float(decision["maximum_mean_joined_wall_time_ratio"]),
            "final_loss": statistics.fmean(
                row["final_loss_relative_regression"] for row in paired
            )
            <= float(decision["maximum_mean_final_loss_relative_regression"]),
            "weak_arm": max(
                row["maximum_weak_arm_relative_loss_regression"]
                for row in paired
            )
            <= float(
                decision["maximum_weak_arm_relative_loss_regression"]
            ),
            "seed_quality": (
                sum(
                    row["final_loss_relative_regression"]
                    <= float(
                        decision[
                            "maximum_seed_final_loss_relative_regression"
                        ]
                    )
                    for row in paired
                )
                / len(paired)
            )
            >= float(decision["minimum_seed_quality_fraction"]),
            "time_to_quality": all(value is not None for value in time_ratios)
            and statistics.fmean(float(value) for value in time_ratios)
            <= float(
                decision[
                    "maximum_mean_time_to_control_final_quality_ratio"
                ]
            ),
            "function_preservation": all(
                transition["function_preserving"]
                for row in candidate_runs
                for transition in row["transitions"]
            ),
            "checkpoint_replay": all(
                transition["checkpoint_replay"]["exact_model_reload"]
                and transition["checkpoint_replay"]["exact_optimizer_reload"]
                and transition["checkpoint_replay"][
                    "next_update_numerically_equivalent"
                ]
                for row in candidate_runs
                for transition in row["transitions"]
            )
            and all(
                row["final_checkpoint_replay"]["exact_model_reload"]
                and row["final_checkpoint_replay"]["exact_optimizer_reload"]
                and row["final_checkpoint_replay"][
                    "next_update_numerically_equivalent"
                ]
                for row in candidate_runs
            ),
            "finite_gradients": all(
                row["finite_gradients"] for row in candidate_runs
            ),
        }
        adopted = all(gates.values())
        mean_time_ratio = statistics.fmean(
            row["joined_wall_time_ratio"] for row in paired
        )
        comparisons[schedule_id] = {
            "paired_runs": paired,
            "mean_joined_wall_time_ratio": mean_time_ratio,
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
        if adopted:
            eligible.append((mean_time_ratio, schedule_id))
    selected = min(eligible)[1] if eligible else "full_depth_control"
    return comparisons, {
        "selected_training_structure": selected,
        "kind": (
            "MASKED_STRUCTURAL_GROWTH_SELECTED"
            if eligible
            else "FULL_DEPTH_CONTROL_RETAINED"
        ),
        "scientific_falsification_claimed": False,
        "reentry_condition": (
            None
            if eligible
            else "a new prospective schedule or larger matched rung that passes the same function, replay, weak-tail, and joined-time gates"
        ),
    }


def journal_contract(config_path: Path) -> dict[str, Any]:
    return {
        "policy": "project_theseus_masked_structural_growth_journal_v1",
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "model_implementation_sha256": sha256_file(
            ROOT / "scripts/standard_causal_transformer_model.py"
        ),
    }


def prepare_journal(
    config_path: Path, scratch: Path
) -> tuple[Path, list[dict[str, Any]], bool]:
    contract_path = scratch / "run_journal.contract.json"
    journal_path = scratch / "run_journal.jsonl"
    expected = journal_contract(config_path)
    if contract_path.is_file() and journal_path.is_file():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed == expected:
            rows = [
                json.loads(line)
                for line in journal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return journal_path, rows, True
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
        raise StructuralGrowthFault("source_disjointness_failed")
    steps = int(config["training"]["steps"])
    batches_by_seed = {
        int(seed): corpus.balanced_batches(
            train_rows, steps=steps, seed=int(seed)
        )
        for seed in config["seeds"]
    }
    scratch = resolve(config["scratch_root"])
    journal_path, journal_rows, resumed = prepare_journal(
        config_path, scratch
    )
    index = {
        (str(row["run_id"]), int(row["seed"])): row for row in journal_rows
    }
    schedule_by_id = {
        schedule["id"]: schedule for schedule in config["growth_schedules"]
    }
    run_ids = ["full_depth_control", *schedule_by_id]
    orders = [
        run_ids[offset:] + run_ids[:offset]
        for offset in range(len(config["seeds"]))
    ]
    runs: list[dict[str, Any]] = []
    for seed, order in zip(config["seeds"], orders):
        for run_id in order:
            key = (run_id, int(seed))
            row = index.get(key)
            if row is None:
                row = run_one(
                    config,
                    run_id=run_id,
                    schedule=schedule_by_id.get(run_id),
                    seed=int(seed),
                    train_rows=train_rows,
                    heldout=heldout,
                    batches=batches_by_seed[int(seed)],
                    vocabulary=vocabulary,
                    scratch=scratch,
                )
                with journal_path.open("a", encoding="utf-8") as handle:
                    handle.write(canonical(row) + "\n")
                    handle.flush()
                index[key] = row
            runs.append(row)
    comparisons, disposition = compare(runs, config)
    final_depth_config = model_config(
        config, vocabulary, depth=int(config["model"]["num_layers"])
    )
    parameter_count = sum(
        analytical_parameter_breakdown(final_depth_config).values()
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
        "trigger_state": "GREEN"
        if disposition["selected_training_structure"]
        else "RED",
        "model_parameter_count": parameter_count,
        "source_disjointness": {
            "train_source_count": len(train_ids),
            "heldout_source_count": len(heldout_ids),
            "intersection_count": len(train_ids & heldout_ids),
            "passed": not bool(train_ids & heldout_ids),
        },
        "run_order_by_seed": {
            str(seed): order for seed, order in zip(config["seeds"], orders)
        },
        "resumed_from_journal": resumed,
        "runs": runs,
        "comparisons": comparisons,
        "campaign_disposition": disposition,
        "hard_boundaries": config["hard_boundaries"],
        "external_source_basis": {
            "paper": "https://arxiv.org/abs/2305.02869",
            "official_repository": "https://github.com/cofe-ai/MSG",
            "local_adaptation": (
                "fixed-width staged decoder depth; zero residual masks; "
                "official-style stacking initialization from trained layers; "
                "surviving AdamW moments copied and new moments zeroed"
            ),
        },
        "non_claims": [
            "No capability, full-campaign, architecture-superiority, AGI, or ASI claim.",
            "A failed schedule is scoped to this implementation, data, model, optimizer, budget, evaluator, and host.",
            "Engineering exclusion is not scientific falsification of masked structural growth.",
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
                    "comparisons": report["comparisons"],
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
