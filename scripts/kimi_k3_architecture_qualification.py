#!/usr/bin/env python3
"""Matched finite qualification for Kimi K3-derived topology candidates.

This owner supports only Block Attention Residuals and SiTU-GLU. It is a
private engineering selection surface, never a capability benchmark.
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

import masked_structural_growth_qualification as common
import mtp_matched_adequacy as corpus
import optimizer_matched_adequacy as optimizer_adequacy
import pretraining_candidate_canary
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
DEFAULT_CONFIG = ROOT / "configs/kimi_k3_attnres_qualification.json"
POLICY = "project_theseus_kimi_k3_architecture_qualification_v1"
VARIANTS = {"block_attention_residual", "situ_glu"}


class KimiK3ArchitectureFault(ValueError):
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


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy") != POLICY:
        raise KimiK3ArchitectureFault("config_policy_invalid")
    variant = config.get("variant") or {}
    if variant.get("kind") not in VARIANTS or not variant.get("id"):
        raise KimiK3ArchitectureFault("variant_invalid")
    if variant["kind"] == "block_attention_residual":
        block_size = int(variant.get("block_size") or 0)
        layers = int(config["model"]["num_layers"])
        if block_size <= 0 or block_size > layers:
            raise KimiK3ArchitectureFault("attnres_block_size_invalid")
        if float(variant.get("candidate_scale", -1.0)) != 1.0:
            raise KimiK3ArchitectureFault("attnres_candidate_scale_invalid")
        if float(variant.get("reduction_scale", -1.0)) != 0.0:
            raise KimiK3ArchitectureFault("attnres_reduction_scale_invalid")
    else:
        if (
            float(variant.get("gate_beta") or 0.0) != 4.0
            or float(variant.get("up_beta") or 0.0) != 25.0
        ):
            raise KimiK3ArchitectureFault("situ_glu_source_betas_invalid")
    seeds = [int(seed) for seed in config.get("seeds") or []]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise KimiK3ArchitectureFault("three_distinct_seeds_required")
    training = config.get("training") or {}
    if (
        int(training.get("steps") or 0) <= 0
        or int(training.get("steps") or 0) > 192
        or int(training.get("evaluation_interval_steps") or 0) <= 0
        or float(training.get("learning_rate") or 0.0) <= 0.0
        or float(training.get("gradient_clip_norm") or 0.0) <= 0.0
    ):
        raise KimiK3ArchitectureFault("training_contract_invalid")
    decision = config.get("decision") or {}
    for field in (
        "maximum_next_update_absolute_error",
        "maximum_weak_arm_relative_loss_regression",
        "maximum_mean_joined_wall_time_ratio",
        "maximum_mean_time_to_control_final_quality_ratio",
        "minimum_mean_final_loss_improvement",
        "minimum_seed_win_fraction",
    ):
        if float(decision.get(field, -1.0)) < 0.0:
            raise KimiK3ArchitectureFault(
                f"decision_contract_invalid:{field}"
            )
    boundaries = config.get("hard_boundaries") or {}
    for field in (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "fallback_or_template_credit",
        "confirmation_surface_consumption",
    ):
        if boundaries.get(field) != 0:
            raise KimiK3ArchitectureFault(f"hard_boundary_nonzero:{field}")
    for field in (
        "production_checkpoint_mutation",
        "heldout_labels_visible_to_candidate",
        "selection_from_throughput_alone",
        "in_place_live_checkpoint_migration",
    ):
        if boundaries.get(field) is not False:
            raise KimiK3ArchitectureFault(
                f"hard_boundary_boolean_invalid:{field}"
            )
    return config


def model_config(
    config: dict[str, Any], vocabulary: int, *, candidate: bool
) -> CausalTransformerConfig:
    row = config["model"]
    variant = config["variant"]
    return CausalTransformerConfig(
        vocab_size=int(vocabulary),
        d_model=int(row["d_model"]),
        num_layers=int(row["num_layers"]),
        num_heads=int(row["num_heads"]),
        num_kv_heads=int(row["num_kv_heads"]),
        ff_dim=int(row["ff_dim"]),
        attention_policy=str(row["attention_policy"]),
        source_target_separator_token_id=SOURCE_TARGET_SEPARATOR_ID,
        attention_residual_mode=(
            "block"
            if candidate
            and variant["kind"] == "block_attention_residual"
            else "none"
        ),
        attention_residual_block_size=(
            int(variant["block_size"])
            if candidate
            and variant["kind"] == "block_attention_residual"
            else 0
        ),
        feed_forward_activation=(
            "situ_glu"
            if candidate and variant["kind"] == "situ_glu"
            else "swiglu"
        ),
        situ_glu_gate_beta=float(variant.get("gate_beta", 4.0)),
        situ_glu_up_beta=float(variant.get("up_beta", 25.0)),
    )


def assert_migration_compatible(
    source: CausalTransformerConfig,
    target: CausalTransformerConfig,
) -> None:
    fields = (
        "attention_residual_mode",
        "attention_residual_block_size",
        "feed_forward_activation",
        "situ_glu_gate_beta",
        "situ_glu_up_beta",
    )
    changed = [
        field
        for field in fields
        if getattr(source, field) != getattr(target, field)
    ]
    if changed:
        raise KimiK3ArchitectureFault(
            "in_place_topology_migration_forbidden:" + ",".join(changed)
        )


def build_adamw(config: dict[str, Any], *, optim: Any) -> Any:
    return optim.AdamW(
        learning_rate=float(config["training"]["learning_rate"]),
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=float(config["training"]["weight_decay"]),
        bias_correction=False,
    )


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
    def objective(
        local_model: Any, x: Any, y: Any, mask: Any
    ) -> Any:
        logits, _cache = local_model(x)
        losses = nn.losses.cross_entropy(logits, y)
        denominator = mx.maximum(
            mx.sum(mask), mx.array(1.0, dtype=mx.float32)
        )
        return mx.sum(losses * mask) / denominator

    value_and_grad = nn.value_and_grad(model, objective)
    compiled_state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=compiled_state, outputs=compiled_state)
    def compiled_step(
        x: Any, y: Any, mask: Any
    ) -> tuple[Any, Any, Any]:
        loss, gradients = value_and_grad(model, x, y, mask)
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
    heldout: dict[str, list[dict[str, Any]]],
    maximum: int,
    *,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    return corpus._evaluate(model, heldout, maximum, mx, nn)


def flat_array_digest(rows: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in rows:
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def common_parameter_identity(
    control: Any, candidate: Any, mlx_utils: Any
) -> dict[str, Any]:
    left = dict(mlx_utils.tree_flatten(control.parameters()))
    right = dict(mlx_utils.tree_flatten(candidate.parameters()))
    shared = sorted(set(left) & set(right))
    mismatches = [
        name
        for name in shared
        if tuple(left[name].shape) != tuple(right[name].shape)
        or float(
            np.max(
                np.abs(
                    np.asarray(left[name], dtype=np.float64)
                    - np.asarray(right[name], dtype=np.float64)
                )
            )
        )
        != 0.0
    ]
    return {
        "shared_parameter_count": len(shared),
        "control_only_parameters": sorted(set(left) - set(right)),
        "candidate_only_parameters": sorted(set(right) - set(left)),
        "mismatched_shared_parameters": mismatches,
        "control_common_sha256": flat_array_digest(
            [(name, left[name]) for name in shared]
        ),
        "candidate_common_sha256": flat_array_digest(
            [(name, right[name]) for name in shared]
        ),
    }


def selection_common_parameter_digest(
    model: Any, *, variant: str, mlx_utils: Any
) -> str:
    rows = mlx_utils.tree_flatten(model.parameters())
    if variant == "block_attention_residual":
        rows = [
            (name, value)
            for name, value in rows
            if name != "attention_residual_queries.weight"
        ]
    return flat_array_digest(rows)


def mechanics_probe(
    config: dict[str, Any],
    *,
    vocabulary: int,
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    seed = int(config["seeds"][0])
    control_config = model_config(config, vocabulary, candidate=False)
    candidate_config = model_config(config, vocabulary, candidate=True)
    mx.random.seed(seed)
    control = build_model(control_config, mx=mx, nn=nn)
    mx.random.seed(seed)
    candidate = build_model(candidate_config, mx=mx, nn=nn)
    identity = common_parameter_identity(control, candidate, mlx_utils)
    maximum = int(config["supervision"]["maximum_sequence_tokens"])
    x_np, y_np, mask_np = corpus.make_batch(batch, maximum)
    x, y, mask = mx.array(x_np), mx.array(y_np), mx.array(mask_np)
    control_logits, _cache = control(x)
    candidate_logits, _cache = candidate(x)
    mx.eval(control_logits, candidate_logits)
    intervention_error = float(
        mx.max(mx.abs(candidate_logits - control_logits)).item()
    )

    def objective(local_model: Any) -> Any:
        logits, _cache = local_model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.sum(losses * mask) / mx.maximum(
            mx.sum(mask), mx.array(1.0, dtype=mx.float32)
        )

    loss, gradients = nn.value_and_grad(candidate, objective)(candidate)
    mx.eval(loss, gradients)
    flat_gradients = dict(mlx_utils.tree_flatten(gradients))
    layer_gradient_l1 = {
        str(index): sum(
            float(mx.sum(mx.abs(value)).item())
            for name, value in flat_gradients.items()
            if name.startswith(f"layers.{index}.")
        )
        for index in range(int(config["model"]["num_layers"]))
    }
    variant = config["variant"]["kind"]
    result: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "common_parameter_identity": identity,
        "common_parameters_exact": not identity[
            "mismatched_shared_parameters"
        ]
        and identity["control_common_sha256"]
        == identity["candidate_common_sha256"],
        "candidate_intervention_output_max_absolute_difference": (
            intervention_error
        ),
        "candidate_changes_function": intervention_error > 0.0,
        "candidate_loss": float(loss.item()),
        "layer_gradient_l1": layer_gradient_l1,
        "gradient_to_every_layer": all(
            value > 0.0 for value in layer_gradient_l1.values()
        ),
        "in_place_migration_refused": False,
    }
    try:
        assert_migration_compatible(control_config, candidate_config)
    except KimiK3ArchitectureFault:
        result["in_place_migration_refused"] = True

    if variant == "block_attention_residual":
        candidate.attention_residual_scale = float(
            config["variant"]["reduction_scale"]
        )
        reduced_logits, _cache = candidate(x)
        mx.eval(reduced_logits)
        reduction_error = float(
            mx.max(mx.abs(reduced_logits - control_logits)).item()
        )
        candidate.attention_residual_scale = float(
            config["variant"]["candidate_scale"]
        )
        query_gradient = flat_gradients[
            "attention_residual_queries.weight"
        ]
        query_rows = [
            float(mx.sum(mx.abs(query_gradient[index])).item())
            for index in range(int(query_gradient.shape[0]))
        ]
        result.update(
            {
                "reduction_to_control_output_max_absolute_error": (
                    reduction_error
                ),
                "reduction_to_control_exact": reduction_error == 0.0,
                "query_gradient_l1_by_index": query_rows,
                "gradient_to_every_nontrivial_query": all(
                    value > 0.0 for value in query_rows[1:]
                ),
                "first_query_single_source_zero_gradient_expected": (
                    query_rows[0] == 0.0
                ),
            }
        )
    else:
        probe = mx.linspace(-100.0, 100.0, 4097)
        gate_beta = float(config["variant"]["gate_beta"])
        up_beta = float(config["variant"]["up_beta"])
        situ = (
            gate_beta
            * mx.tanh(probe / gate_beta)
            * mx.sigmoid(probe)
            * up_beta
            * mx.tanh(probe / up_beta)
        )
        swiglu = nn.silu(probe) * probe
        near = mx.array([-1e-4, 1e-4], dtype=mx.float32)
        situ_near = (
            gate_beta
            * mx.tanh(near / gate_beta)
            * mx.sigmoid(near)
            * up_beta
            * mx.tanh(near / up_beta)
        )
        swiglu_near = nn.silu(near) * near
        mx.eval(situ, swiglu, situ_near, swiglu_near)
        result.update(
            {
                "formal_coordinate_bound": gate_beta * up_beta,
                "observed_maximum_absolute_scalar_response": float(
                    mx.max(mx.abs(situ)).item()
                ),
                "bound_respected": float(mx.max(mx.abs(situ)).item())
                <= gate_beta * up_beta,
                "near_origin_max_absolute_difference": float(
                    mx.max(mx.abs(situ_near - swiglu_near)).item()
                ),
                "large_input_swiglu_maximum": float(
                    mx.max(mx.abs(swiglu)).item()
                ),
            }
        )
    return result


def checkpoint_replay(
    *,
    config: dict[str, Any],
    candidate: bool,
    model: Any,
    optimizer: Any,
    x: Any,
    y: Any,
    mask: Any,
    vocabulary: int,
    checkpoint_root: Path,
    mx: Any,
    nn: Any,
    optim: Any,
    mlx_utils: Any,
) -> dict[str, Any]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_root / "model.npz"
    optimizer_path = checkpoint_root / "optimizer.npz"
    model.save_weights(str(model_path))
    names_path = optimizer_adequacy.save_optimizer_state(
        optimizer_path, optimizer, mx, mlx_utils
    )
    clone = build_model(
        model_config(config, vocabulary, candidate=candidate),
        mx=mx,
        nn=nn,
    )
    clone.load_weights(str(model_path))
    clone_optimizer = build_adamw(config, optim=optim)
    optimizer_adequacy.load_optimizer_state(
        optimizer_path,
        names_path,
        clone_optimizer,
        mx,
        mlx_utils,
    )
    optimizer_adequacy.bind_loaded_optimizer_state(
        clone_optimizer, clone.trainable_parameters()
    )
    exact_model = common.tree_digest(
        model.parameters(), mlx_utils
    ) == common.tree_digest(clone.parameters(), mlx_utils)
    exact_optimizer = common.tree_digest(
        optimizer.state, mlx_utils
    ) == common.tree_digest(clone_optimizer.state, mlx_utils)

    def objective(local_model: Any, local_x: Any, local_y: Any, local_mask: Any) -> Any:
        logits, _cache = local_model(local_x)
        losses = nn.losses.cross_entropy(logits, local_y)
        return mx.sum(losses * local_mask) / mx.maximum(
            mx.sum(local_mask), mx.array(1.0, dtype=mx.float32)
        )

    left_loss, left_gradients = nn.value_and_grad(model, objective)(
        model, x, y, mask
    )
    right_loss, right_gradients = nn.value_and_grad(clone, objective)(
        clone, x, y, mask
    )
    left_gradients, left_norm = optim.clip_grad_norm(
        left_gradients, float(config["training"]["gradient_clip_norm"])
    )
    right_gradients, right_norm = optim.clip_grad_norm(
        right_gradients, float(config["training"]["gradient_clip_norm"])
    )
    optimizer.update(model, left_gradients)
    clone_optimizer.update(clone, right_gradients)
    mx.eval(
        left_loss,
        right_loss,
        left_norm,
        right_norm,
        model.parameters(),
        clone.parameters(),
        optimizer.state,
        clone_optimizer.state,
    )
    model_error = common.tree_max_difference(
        model.parameters(), clone.parameters(), mlx_utils
    )
    optimizer_error = common.tree_max_difference(
        optimizer.state, clone_optimizer.state, mlx_utils
    )
    loss_error = abs(
        float(left_loss.item()) - float(right_loss.item())
    )
    gradient_norm_error = abs(
        float(left_norm.item()) - float(right_norm.item())
    )
    tolerance = float(
        config["decision"]["maximum_next_update_absolute_error"]
    )
    receipt = {
        "model_checkpoint": relative(model_path),
        "model_checkpoint_sha256": sha256_file(model_path),
        "optimizer_checkpoint": relative(optimizer_path),
        "optimizer_checkpoint_sha256": sha256_file(optimizer_path),
        "optimizer_names_sha256": sha256_file(names_path),
        "exact_model_reload": exact_model,
        "exact_optimizer_reload": exact_optimizer,
        "next_loss_absolute_error": loss_error,
        "next_gradient_norm_absolute_error": gradient_norm_error,
        "gradient_norm_comparison_role": (
            "diagnostic_only; next-update equivalence is judged from the "
            "resulting model and optimizer state"
        ),
        "next_model_max_absolute_error": model_error,
        "next_optimizer_max_absolute_error": optimizer_error,
        "maximum_allowed_next_update_absolute_error": tolerance,
        "next_update_numerically_equivalent": (
            model_error <= tolerance
            and optimizer_error <= tolerance
            and loss_error <= tolerance
        ),
    }
    return receipt


def run_one(
    config: dict[str, Any],
    *,
    candidate: bool,
    seed: int,
    train_rows: dict[str, list[dict[str, Any]]],
    heldout: dict[str, list[dict[str, Any]]],
    batches: list[list[dict[str, Any]]],
    vocabulary: int,
    scratch: Path,
    monitor: pretraining_candidate_canary.CandidateCanaryMonitor,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    run_id = config["variant"]["id"] if candidate else "control"
    maximum = int(config["supervision"]["maximum_sequence_tokens"])
    steps = int(config["training"]["steps"])
    local_config = model_config(config, vocabulary, candidate=candidate)
    common.reset_peak_memory(mx)
    mx.random.seed(int(seed))
    model = build_model(local_config, mx=mx, nn=nn)
    initial_parameter_sha256 = common.tree_digest(
        model.parameters(), mlx_utils
    )
    initial_common_parameter_sha256 = selection_common_parameter_digest(
        model,
        variant=config["variant"]["kind"],
        mlx_utils=mlx_utils,
    )
    optimizer = build_adamw(config, optim=optim)
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
    started = time.perf_counter()
    initial = evaluate(model, heldout, maximum, mx=mx, nn=nn)
    curve = [
        {
            "step": 0,
            "heldout": initial,
            "joined_wall_seconds": time.perf_counter() - started,
        }
    ]
    positions = 0
    finite = True
    first_gradient_l1 = None
    primary_step_seconds = 0.0
    last_batch = None
    for step, batch in enumerate(batches, 1):
        x_np, y_np, mask_np = corpus.make_batch(batch, maximum)
        x, y, mask = mx.array(x_np), mx.array(y_np), mx.array(mask_np)
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
        values = (
            float(loss.item()),
            float(gradient_norm.item()),
            float(gradient_l1.item()),
        )
        finite = finite and all(math.isfinite(value) for value in values)
        if first_gradient_l1 is None:
            first_gradient_l1 = values[2]
        positions += int(mask_np.sum())
        last_batch = (x, y, mask)
        monitor.check(f"{run_id}:{seed}", step)
        if (
            step % int(config["training"]["evaluation_interval_steps"])
            == 0
            or step == steps
        ):
            curve.append(
                {
                    "step": step,
                    "heldout": evaluate(
                        model, heldout, maximum, mx=mx, nn=nn
                    ),
                    "joined_wall_seconds": time.perf_counter() - started,
                }
            )
    joined_wall = time.perf_counter() - started
    if last_batch is None:
        raise KimiK3ArchitectureFault("training_batch_missing")
    replay_started = time.perf_counter()
    replay = checkpoint_replay(
        config=config,
        candidate=candidate,
        model=model,
        optimizer=optimizer,
        x=last_batch[0],
        y=last_batch[1],
        mask=last_batch[2],
        vocabulary=vocabulary,
        checkpoint_root=scratch / run_id / str(seed),
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
    )
    replay_seconds = time.perf_counter() - replay_started
    breakdown = analytical_parameter_breakdown(local_config)
    return {
        "run_id": run_id,
        "kind": "candidate" if candidate else "control",
        "seed": int(seed),
        "model": asdict(local_config),
        "parameter_breakdown": breakdown,
        "parameter_count": sum(breakdown.values()),
        "initial_parameter_sha256": initial_parameter_sha256,
        "initial_common_parameter_sha256": (
            initial_common_parameter_sha256
        ),
        "optimizer_id": "adamw_mlx",
        "optimizer_steps": steps,
        "optimizer_positions": positions,
        "initial_heldout": initial,
        "final_heldout": curve[-1]["heldout"],
        "curve": curve,
        "finite_gradients": finite,
        "first_gradient_l1": first_gradient_l1,
        "training_primary_step_seconds": primary_step_seconds,
        "joined_training_wall_seconds": joined_wall,
        "excluded_replay_audit_seconds": replay_seconds,
        "peak_allocator_bytes": common.peak_memory_bytes(mx),
        "checkpoint_replay": replay,
        "public_training_rows": 0,
        "public_evaluation_rows": 0,
        "external_inference_calls": 0,
        "fallback_or_template_credit": 0,
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
) -> dict[str, Any]:
    controls = {
        int(row["seed"]): row for row in runs if row["kind"] == "control"
    }
    candidates = [
        row for row in runs if row["kind"] == "candidate"
    ]
    paired = []
    for row in candidates:
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
                - float(
                    control["final_heldout"]["by_arm"][arm]["ntp_loss"]
                )
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
                "relative_loss_improvement": (
                    control_loss - candidate_loss
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
                "allocator_peak_ratio": float(
                    row["peak_allocator_bytes"] or 0
                )
                / max(float(control["peak_allocator_bytes"] or 1), 1.0),
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
    mean_gain = statistics.fmean(
        row["relative_loss_improvement"] for row in paired
    )
    time_to_quality = [
        row["time_to_control_final_quality_ratio"] for row in paired
    ]
    common_gates = {
        "all_seeds": {row["seed"] for row in paired}
        == {int(seed) for seed in config["seeds"]},
        "weak_arm": max(
            row["maximum_weak_arm_relative_loss_regression"]
            for row in paired
        )
        <= float(
            decision["maximum_weak_arm_relative_loss_regression"]
        ),
        "seed_wins": sum(
            row["relative_loss_improvement"] > 0.0 for row in paired
        )
        / len(paired)
        >= float(decision["minimum_seed_win_fraction"]),
        "joined_wall": statistics.fmean(
            row["joined_wall_time_ratio"] for row in paired
        )
        <= float(decision["maximum_mean_joined_wall_time_ratio"]),
        "checkpoint_replay": all(
            row["checkpoint_replay"]["exact_model_reload"]
            and row["checkpoint_replay"]["exact_optimizer_reload"]
            and row["checkpoint_replay"][
                "next_update_numerically_equivalent"
            ]
            for row in candidates
        ),
        "finite_gradients": all(
            row["finite_gradients"] for row in candidates
        ),
    }
    acceleration_route = {
        **common_gates,
        "time_to_quality": all(
            value is not None for value in time_to_quality
        )
        and statistics.fmean(
            float(value) for value in time_to_quality
        )
        <= float(
            decision[
                "maximum_mean_time_to_control_final_quality_ratio"
            ]
        ),
        "no_mean_loss_regression": mean_gain >= 0.0,
    }
    quality_route = {
        **common_gates,
        "material_mean_loss_improvement": mean_gain
        >= float(decision["minimum_mean_final_loss_improvement"]),
    }
    adopted = all(acceleration_route.values()) or all(
        quality_route.values()
    )
    return {
        "paired_runs": paired,
        "mean_relative_loss_improvement": mean_gain,
        "mean_joined_wall_time_ratio": statistics.fmean(
            row["joined_wall_time_ratio"] for row in paired
        ),
        "mean_primary_step_time_ratio": statistics.fmean(
            row["primary_step_time_ratio"] for row in paired
        ),
        "mean_allocator_peak_ratio": statistics.fmean(
            row["allocator_peak_ratio"] for row in paired
        ),
        "mean_time_to_control_final_quality_ratio": (
            statistics.fmean(
                float(value) for value in time_to_quality
            )
            if all(value is not None for value in time_to_quality)
            else None
        ),
        "acceleration_route_gates": acceleration_route,
        "quality_route_gates": quality_route,
        "disposition": (
            "ADOPTED_NEW_INCOMPATIBLE_LINEAGE"
            if adopted
            else "NOT_SELECTED_FIRST_CAMPAIGN"
        ),
        "selected_architecture": (
            config["variant"]["id"] if adopted else "control"
        ),
        "scientific_falsification_claimed": False,
    }


def journal_contract(config_path: Path) -> dict[str, Any]:
    return {
        "policy": "project_theseus_kimi_k3_architecture_journal_v1",
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
            return (
                journal_path,
                [
                    json.loads(line)
                    for line in journal_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
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


def append_journal(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        handle.flush()


def parameter_count_gate(
    config: dict[str, Any], vocabulary: int
) -> dict[str, Any]:
    control = analytical_parameter_breakdown(
        model_config(config, vocabulary, candidate=False)
    )
    candidate = analytical_parameter_breakdown(
        model_config(config, vocabulary, candidate=True)
    )
    declared = config["model_parameter_count_range"]
    counts = {
        "control": sum(control.values()),
        "candidate": sum(candidate.values()),
    }
    return {
        "counts": counts,
        "candidate_parameter_delta": counts["candidate"]
        - counts["control"],
        "candidate_parameter_ratio": counts["candidate"]
        / max(counts["control"], 1),
        "declared_range": declared,
        "passed": all(
            int(declared["minimum"])
            <= count
            <= int(declared["maximum"])
            for count in counts.values()
        ),
    }


def execute(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    metadata = json.loads(
        resolve(config["stage_metadata"]).read_text(encoding="utf-8")
    )
    base = json.loads(
        resolve(config["base_config"]).read_text(encoding="utf-8")
    )
    vocabulary = model_vocab_size(
        base, metadata["source_vocab"], metadata["target_vocab"]
    )
    count_gate = parameter_count_gate(config, vocabulary)
    if not count_gate["passed"]:
        raise KimiK3ArchitectureFault("parameter_count_out_of_range")
    train = corpus.load_governed_rows(config, split="train")
    heldout = corpus.load_governed_rows(config, split="heldout")
    train_ids = optimizer_adequacy.source_sets(train)
    heldout_ids = optimizer_adequacy.source_sets(heldout)
    if train_ids & heldout_ids:
        raise KimiK3ArchitectureFault("source_disjointness_failed")
    scratch = resolve(config["scratch_root"])
    journal_path, journal_rows, resumed = prepare_journal(
        config_path, scratch
    )
    index = {
        (row["kind"], int(row["seed"])): row for row in journal_rows
    }
    lease = pretraining_candidate_canary.candidate_lease(
        candidate_id=config["candidate_lease_id"],
        max_steps=int(config["training"]["steps"]),
        scratch_checkpoint_root=scratch,
        targets=["shared_trunk"],
        phase="pretraining",
        resume=False,
    )
    if not lease["authorized"]:
        raise KimiK3ArchitectureFault(
            "candidate_lease_denied:" + ",".join(lease["faults"])
        )
    monitor = pretraining_candidate_canary.CandidateCanaryMonitor(lease)
    mechanics_batch = corpus.balanced_batches(
        train, steps=1, seed=int(config["seeds"][0])
    )[0]
    mechanics = mechanics_probe(
        config, vocabulary=vocabulary, batch=mechanics_batch
    )
    runs = []
    for seed in config["seeds"]:
        batches = corpus.balanced_batches(
            train,
            steps=int(config["training"]["steps"]),
            seed=int(seed),
        )
        for candidate in (False, True):
            kind = "candidate" if candidate else "control"
            key = (kind, int(seed))
            row = index.get(key)
            if row is None:
                row = run_one(
                    config,
                    candidate=candidate,
                    seed=int(seed),
                    train_rows=train,
                    heldout=heldout,
                    batches=batches,
                    vocabulary=vocabulary,
                    scratch=scratch,
                    monitor=monitor,
                )
                append_journal(journal_path, row)
                index[key] = row
            runs.append(row)
    for seed in config["seeds"]:
        pair = [row for row in runs if int(row["seed"]) == int(seed)]
        if len({row["optimizer_positions"] for row in pair}) != 1:
            raise KimiK3ArchitectureFault(
                f"matched_optimizer_positions_failed:{seed}"
            )
        if len(
            {row["initial_common_parameter_sha256"] for row in pair}
        ) != 1:
            raise KimiK3ArchitectureFault(
                f"matched_common_initialization_failed:{seed}"
            )
    comparison = compare(runs, config)
    resource = monitor.finalize(runs)
    variant = config["variant"]["kind"]
    mechanics_gates = {
        "common_parameters_exact": mechanics["common_parameters_exact"],
        "candidate_changes_function": mechanics[
            "candidate_changes_function"
        ],
        "gradient_to_every_layer": mechanics["gradient_to_every_layer"],
        "in_place_migration_refused": mechanics[
            "in_place_migration_refused"
        ],
    }
    if variant == "block_attention_residual":
        mechanics_gates.update(
            {
                "reduction_to_control_exact": mechanics[
                    "reduction_to_control_exact"
                ],
                "gradient_to_every_nontrivial_query": mechanics[
                    "gradient_to_every_nontrivial_query"
                ],
                "first_query_single_source_zero_gradient_expected": mechanics[
                    "first_query_single_source_zero_gradient_expected"
                ],
            }
        )
    else:
        mechanics_gates.update(
            {
                "formal_bound_respected": mechanics["bound_respected"],
                "near_origin_match": mechanics[
                    "near_origin_max_absolute_difference"
                ]
                <= float(
                    config["decision"][
                        "maximum_near_origin_absolute_error"
                    ]
                ),
            }
        )
    gates = {
        "source_disjoint": not train_ids & heldout_ids,
        "matched_optimizer_positions": True,
        "matched_common_initialization": True,
        "parameter_count": count_gate["passed"],
        "mechanics": all(mechanics_gates.values()),
        "checkpoint_replay": all(
            row["checkpoint_replay"]["next_update_numerically_equivalent"]
            for row in runs
        ),
        "finite_gradients": all(
            row["finite_gradients"] for row in runs
        ),
        "resource_bounds": resource["passed"],
        "no_public_external_or_fallback": all(
            row["public_training_rows"] == 0
            and row["public_evaluation_rows"] == 0
            and row["external_inference_calls"] == 0
            and row["fallback_or_template_credit"] == 0
            for row in runs
        ),
    }
    report = {
        "policy": POLICY,
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "support_state": "private-source-disjoint-matched-topology-experiment",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation_identity": {
            "qualification_path": relative(Path(__file__)),
            "qualification_sha256": sha256_file(Path(__file__)),
            "model_path": "scripts/standard_causal_transformer_model.py",
            "model_sha256": sha256_file(
                ROOT / "scripts/standard_causal_transformer_model.py"
            ),
        },
        "variant": config["variant"],
        "candidate_lease": lease,
        "source_disjointness": {
            "train_count": len(train_ids),
            "heldout_count": len(heldout_ids),
            "overlap_count": len(train_ids & heldout_ids),
        },
        "parameter_count": count_gate,
        "mechanics": mechanics,
        "mechanics_gates": mechanics_gates,
        "run_journal": {
            "path": relative(journal_path),
            "resumed": resumed,
            "reused_run_count": len(journal_rows),
            "completed_run_count": len(runs),
        },
        "runs": runs,
        "comparison": comparison,
        "resource_receipt": resource,
        "gates": gates,
        "campaign_disposition": {
            "selected_architecture": comparison[
                "selected_architecture"
            ],
            "disposition": comparison["disposition"],
            "live_checkpoint_migration": "FORBIDDEN",
            "scientific_falsification_claimed": False,
        },
        "non_claims": [
            "private teacher-forced loss is not direct assistant utility",
            "a bounded first-campaign disposition is not broad architecture falsification",
            "no public, frozen functional, live teacher, or production checkpoint surface was used",
        ],
    }
    output = resolve(config["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def preflight(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    metadata = json.loads(
        resolve(config["stage_metadata"]).read_text(encoding="utf-8")
    )
    base = json.loads(
        resolve(config["base_config"]).read_text(encoding="utf-8")
    )
    vocabulary = model_vocab_size(
        base, metadata["source_vocab"], metadata["target_vocab"]
    )
    train = corpus.load_governed_rows(config, split="train")
    heldout = corpus.load_governed_rows(config, split="heldout")
    overlap = len(
        optimizer_adequacy.source_sets(train)
        & optimizer_adequacy.source_sets(heldout)
    )
    count_gate = parameter_count_gate(config, vocabulary)
    return {
        "policy": POLICY,
        "trigger_state": (
            "GREEN" if overlap == 0 and count_gate["passed"] else "RED"
        ),
        "variant": config["variant"],
        "seed_count": len(config["seeds"]),
        "step_count": int(config["training"]["steps"]),
        "source_overlap_count": overlap,
        "parameter_count": count_gate,
        "execution_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    report = execute(config_path) if args.execute else preflight(
        config_path
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
