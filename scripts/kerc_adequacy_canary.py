#!/usr/bin/env python3
"""Decision-grade mechanics adequacy harness for the faithful KERC candidate.

This harness deliberately does not estimate KERC utility. It checks whether the
registered implementation can learn a tiny source-bound subset, causally uses
its named modules, survives checkpoint/optimizer restart, and sustains multiple
MLX batches. A failed threshold is inconclusive evidence about the architecture,
not authority to retire it.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import resource
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from kerc_checkpoint_schema import (
    canonical_sha256 as checkpoint_contract_sha256,
    migrate_legacy_checkpoint,
    rollback_checkpoint_contract,
    validate_checkpoint_contract,
)
from kernel_english_protocol import (
    KERC_RESIDUAL_CHANNELS,
    KERC_VERIFIER_DIMENSIONS,
    TRAINING_OBJECTIVES,
)
from moecot_language_arm_training import (
    build_plan,
    build_source_to_target_lookup,
    materialize_target_supervision,
    publish_optimizer,
    sha256_file,
    validate_resume,
)
from standard_causal_transformer_model import CausalTransformerConfig, build_model
from standard_causal_transformer_survival import (
    balanced_binary_class_weights,
    causal_loss,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "kerc_adequacy_canary.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def active_width(inputs: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> int:
    active = np.any((inputs != 0) | (labels != 0) | (mask != 0), axis=0)
    indices = np.flatnonzero(active)
    return int(indices[-1] + 1) if len(indices) else 1


def source_identity(row: np.ndarray, separator_id: int) -> tuple[int, ...]:
    positions = np.flatnonzero(row == separator_id)
    stop = int(positions[0] + 1) if len(positions) else len(row)
    return tuple(int(value) for value in row[:stop])


def select_balanced_subset(
    stage: Any,
    *,
    task_token_ids: tuple[int, ...],
    separator_id: int,
    positives_per_objective: int,
) -> tuple[Any, dict[str, Any]]:
    """Select real shortest rows, then bind each to its exact verifier pair."""

    if len(task_token_ids) != len(TRAINING_OBJECTIVES):
        raise ValueError("KERC task-token inventory does not match objective inventory")
    positive = np.asarray(stage.mask).sum(axis=1) > 0
    identities = [source_identity(row, separator_id) for row in np.asarray(stage.inputs)]
    if positives_per_objective != 1:
        raise ValueError("balanced KERC adequacy selection currently requires one positive per objective")
    candidate_matrix: dict[str, dict[int, tuple[int, int]]] = {}
    for objective, token_id in zip(TRAINING_OBJECTIVES, task_token_ids):
        candidates = [
            index
            for index, row in enumerate(np.asarray(stage.inputs))
            if positive[index] and bool(np.any(row == int(token_id)))
        ]
        candidates.sort(
            key=lambda index: (
                active_width(
                    stage.inputs[index : index + 1],
                    stage.labels[index : index + 1],
                    stage.loss_mask[index : index + 1],
                ),
                canonical_sha256(stage.inputs[index].tolist()),
            )
        )
        by_dimension: dict[int, list[tuple[int, int]]] = {}
        for index in candidates:
            negative = index + 1
            if (
                negative >= len(identities)
                or positive[negative]
                or identities[negative] != identities[index]
                or not bool(np.any(stage.inputs[negative] == int(token_id)))
            ):
                raise ValueError(
                    f"KERC row does not have one exact-source verifier pair: {objective}"
                )
            dimensions = np.flatnonzero(
                np.asarray(stage.kerc_verifier_labels[negative]) < 0.5
            )
            if len(dimensions) != 1:
                raise ValueError(
                    f"KERC verifier pair must corrupt exactly one dimension: {objective}"
                )
            by_dimension.setdefault(int(dimensions[0]), []).append((index, negative))
        candidate_matrix[objective] = {
            dimension: min(
                rows,
                key=lambda pair: (
                    active_width(
                        stage.inputs[pair[0] : pair[0] + 1],
                        stage.labels[pair[0] : pair[0] + 1],
                        stage.loss_mask[pair[0] : pair[0] + 1],
                    ),
                    canonical_sha256(stage.inputs[pair[0]].tolist()),
                ),
            )
            for dimension, rows in by_dimension.items()
        }
    assignments = []
    for dimensions in itertools.permutations(range(len(KERC_VERIFIER_DIMENSIONS))):
        pairs = []
        for objective, dimension in zip(TRAINING_OBJECTIVES, dimensions):
            pair = candidate_matrix[objective].get(dimension)
            if pair is None:
                break
            pairs.append(pair)
        if len(pairs) == len(TRAINING_OBJECTIVES):
            assignments.append((dimensions, pairs))
    if not assignments:
        raise ValueError("KERC adequacy subset cannot cover every verifier dimension")
    dimensions, pairs = min(
        assignments,
        key=lambda assignment: (
            sum(
                active_width(
                    stage.inputs[pair[0] : pair[0] + 1],
                    stage.labels[pair[0] : pair[0] + 1],
                    stage.loss_mask[pair[0] : pair[0] + 1],
                )
                for pair in assignment[1]
            ),
            canonical_sha256(assignment),
        ),
    )
    selected_by_objective = {
        objective: list(pair) for objective, pair in zip(TRAINING_OBJECTIVES, pairs)
    }
    selected = [index for pair in pairs for index in pair]
    if len(selected) != len(set(selected)):
        raise ValueError("KERC adequacy subset contains duplicate rows")
    width = active_width(
        stage.inputs[selected], stage.labels[selected], stage.loss_mask[selected]
    )
    subset = SimpleNamespace(
        inputs=np.asarray(stage.inputs[selected, :width], dtype=np.int32),
        labels=np.asarray(stage.labels[selected, :width], dtype=np.int32),
        mask=np.asarray(stage.mask[selected, :width], dtype=np.uint8),
        loss_mask=np.asarray(stage.loss_mask[selected, :width], dtype=np.float32),
        sample_weights=np.asarray(stage.sample_weights[selected], dtype=np.float64),
        kerc_residual_labels=np.asarray(stage.kerc_residual_labels[selected], dtype=np.int32),
        kerc_verifier_labels=np.asarray(stage.kerc_verifier_labels[selected], dtype=np.float32),
    )
    return subset, {
        "row_indices": selected,
        "rows_by_objective": selected_by_objective,
        "verifier_dimension_by_objective": {
            objective: KERC_VERIFIER_DIMENSIONS[dimension]
            for objective, dimension in zip(TRAINING_OBJECTIVES, dimensions)
        },
        "row_count": len(selected),
        "positive_row_count": int(np.asarray(subset.mask).sum(axis=1).astype(bool).sum()),
        "verifier_only_row_count": int((np.asarray(subset.mask).sum(axis=1) == 0).sum()),
        "sequence_width": width,
        "content_sha256": canonical_sha256(
            {
                "inputs": subset.inputs.tolist(),
                "labels": subset.labels.tolist(),
                "mask": subset.mask.tolist(),
                "residual": subset.kerc_residual_labels.tolist(),
                "verifier": subset.kerc_verifier_labels.tolist(),
            }
        ),
    }


def build_faithful_model(
    target: dict[str, Any],
    *,
    copy_lookup: np.ndarray,
    mx: Any,
    nn: Any,
    ablations: dict[str, str] | None = None,
) -> Any:
    model_config = {**target["model"], **(ablations or {})}
    return build_model(
        CausalTransformerConfig(vocab_size=int(target["vocab_size"]), **model_config),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=copy_lookup,
    )


def row_arrays(stage: Any, index: int) -> tuple[np.ndarray, ...]:
    width = active_width(
        stage.inputs[index : index + 1],
        stage.labels[index : index + 1],
        stage.loss_mask[index : index + 1],
    )
    return (
        stage.inputs[index : index + 1, :width],
        stage.labels[index : index + 1, :width],
        stage.loss_mask[index : index + 1, :width],
        stage.kerc_residual_labels[index : index + 1],
        stage.kerc_verifier_labels[index : index + 1],
    )


def evaluate_model(model: Any, stage: Any, *, mx: Any, nn: Any) -> dict[str, Any]:
    total_losses: list[float] = []
    token_correct = 0
    token_total = 0
    residual_correct = 0
    residual_total = 0
    verifier_correct = 0
    verifier_total = 0
    expected_token_logits: list[list[float]] = []
    verifier_logits_rows: list[list[float]] = []
    residual_prediction_rows: list[list[int]] = []
    residual_target_rows: list[list[int]] = []
    verifier_prediction_rows: list[list[int]] = []
    verifier_target_rows: list[list[int]] = []
    mechanism_activity = {
        "stage_weights_maximum_absolute": 0.0,
        "residual_logits_maximum_absolute": 0.0,
        "verifier_logits_maximum_absolute": 0.0,
    }
    model.eval()
    for index in range(len(stage.inputs)):
        inputs, labels, loss_mask, residual, verifier = row_arrays(stage, index)
        x = mx.array(inputs, dtype=mx.int32)
        y = mx.array(labels, dtype=mx.int32)
        m = mx.array(loss_mask, dtype=mx.float32)
        residual_target = mx.array(residual, dtype=mx.int32)
        verifier_target = mx.array(verifier, dtype=mx.float32)
        loss = causal_loss(
            model,
            x,
            y,
            m,
            mx,
            nn,
            None,
            0.0,
            None,
            "binary_multilabel",
            0,
            (),
            residual_target,
            0.25,
            verifier_target,
            0.5,
        )
        logits, _cache, aux = model(x, return_training_aux=True)
        mx.eval(loss, logits, aux["kerc"]["residual_logits"], aux["kerc"]["verifier_logits"])
        total_losses.append(float(loss.item()))
        logits_np = np.asarray(logits)
        labels_np = np.asarray(labels)
        active = np.asarray(loss_mask) > 0
        if bool(active.any()):
            predictions = logits_np.argmax(axis=-1)
            token_correct += int((predictions[active] == labels_np[active]).sum())
            token_total += int(active.sum())
            target_logits = np.take_along_axis(logits_np, labels_np[..., None], axis=-1)[..., 0]
            expected_token_logits.append(target_logits[active].astype(float).tolist())
        else:
            expected_token_logits.append([])
        residual_logits = np.asarray(aux["kerc"]["residual_logits"])
        mechanism_activity["stage_weights_maximum_absolute"] = max(
            mechanism_activity["stage_weights_maximum_absolute"],
            float(np.max(np.abs(np.asarray(aux["kerc"]["stage_weights"])))),
        )
        mechanism_activity["residual_logits_maximum_absolute"] = max(
            mechanism_activity["residual_logits_maximum_absolute"],
            float(np.max(np.abs(residual_logits))),
        )
        residual_predictions = residual_logits.argmax(axis=-1)
        residual_prediction_rows.extend(residual_predictions.astype(int).tolist())
        residual_target_rows.extend(np.asarray(residual).astype(int).tolist())
        residual_correct += int((residual_predictions == residual).sum())
        residual_total += int(np.asarray(residual).size)
        verifier_logits = np.asarray(aux["kerc"]["verifier_logits"])
        mechanism_activity["verifier_logits_maximum_absolute"] = max(
            mechanism_activity["verifier_logits_maximum_absolute"],
            float(np.max(np.abs(verifier_logits))),
        )
        verifier_logits_rows.append(verifier_logits[0].astype(float).tolist())
        verifier_predictions = verifier_logits >= 0.0
        verifier_prediction_rows.extend(verifier_predictions.astype(int).tolist())
        verifier_target_rows.extend((np.asarray(verifier) >= 0.5).astype(int).tolist())
        verifier_correct += int((verifier_predictions == (verifier >= 0.5)).sum())
        verifier_total += int(np.asarray(verifier).size)
    residual_predictions_array = np.asarray(residual_prediction_rows, dtype=np.int32)
    residual_targets_array = np.asarray(residual_target_rows, dtype=np.int32)
    residual_channels: dict[str, Any] = {}
    for index, channel in enumerate(KERC_RESIDUAL_CHANNELS):
        target_values = residual_targets_array[:, index]
        predicted_values = residual_predictions_array[:, index]
        classes = sorted(int(value) for value in np.unique(target_values))
        recalls = [
            float(np.mean(predicted_values[target_values == value] == value))
            for value in classes
        ]
        residual_channels[channel] = {
            "observed_classes": classes,
            "informative": len(classes) > 1,
            "accuracy": float(np.mean(predicted_values == target_values)),
            "balanced_accuracy": float(np.mean(recalls)),
            "majority_baseline_accuracy": max(
                float(np.mean(target_values == value)) for value in classes
            ),
        }
    informative_residual = [
        row["balanced_accuracy"]
        for row in residual_channels.values()
        if row["informative"]
    ]
    verifier_predictions_array = np.asarray(verifier_prediction_rows, dtype=np.int32)
    verifier_targets_array = np.asarray(verifier_target_rows, dtype=np.int32)
    verifier_dimensions: dict[str, Any] = {}
    for index, dimension in enumerate(KERC_VERIFIER_DIMENSIONS):
        target_values = verifier_targets_array[:, index]
        predicted_values = verifier_predictions_array[:, index]
        positive = target_values == 1
        negative = target_values == 0
        positive_recall = float(np.mean(predicted_values[positive] == 1)) if bool(positive.any()) else None
        negative_recall = float(np.mean(predicted_values[negative] == 0)) if bool(negative.any()) else None
        verifier_dimensions[dimension] = {
            "positive_count": int(positive.sum()),
            "negative_count": int(negative.sum()),
            "positive_recall": positive_recall,
            "negative_recall": negative_recall,
            "balanced_accuracy": (
                (positive_recall + negative_recall) / 2.0
                if positive_recall is not None and negative_recall is not None
                else None
            ),
        }
    informative_verifier = [
        row["balanced_accuracy"]
        for row in verifier_dimensions.values()
        if row["balanced_accuracy"] is not None
    ]
    return {
        "mean_total_loss": float(np.mean(total_losses)),
        "token_accuracy": token_correct / max(1, token_total),
        "token_position_count": token_total,
        "residual_accuracy": residual_correct / max(1, residual_total),
        "residual_informative_channel_count": len(informative_residual),
        "residual_informative_macro_balanced_accuracy": (
            float(np.mean(informative_residual)) if informative_residual else None
        ),
        "residual_channels": residual_channels,
        "residual_label_count": residual_total,
        "verifier_bit_accuracy": verifier_correct / max(1, verifier_total),
        "verifier_macro_balanced_accuracy": (
            float(np.mean(informative_verifier)) if informative_verifier else None
        ),
        "verifier_minimum_negative_recall": min(
            (row["negative_recall"] for row in verifier_dimensions.values() if row["negative_recall"] is not None),
            default=None,
        ),
        "verifier_dimensions": verifier_dimensions,
        "verifier_label_count": verifier_total,
        "expected_token_logits": expected_token_logits,
        "verifier_logits": verifier_logits_rows,
        "mechanism_activity": mechanism_activity,
    }


def one_update(
    model: Any,
    optimizer: Any,
    loss_and_grad: Any,
    stage: Any,
    index: int,
    *,
    gradient_clip: float,
    verifier_positive_weights: Any,
    verifier_negative_weights: Any,
    mx: Any,
    nn: Any,
    optim: Any,
) -> tuple[float, int]:
    inputs, labels, loss_mask, residual, verifier = row_arrays(stage, index)
    x = mx.array(inputs, dtype=mx.int32)
    y = mx.array(labels, dtype=mx.int32)
    m = mx.array(loss_mask, dtype=mx.float32)
    loss, grads = loss_and_grad(
        model,
        x,
        y,
        m,
        mx,
        nn,
        None,
        0.0,
        None,
        "binary_multilabel",
        0,
        (),
        mx.array(residual, dtype=mx.int32),
        0.25,
        mx.array(verifier, dtype=mx.float32),
        0.5,
        verifier_positive_weights,
        verifier_negative_weights,
    )
    grads, norm = optim.clip_grad_norm(grads, gradient_clip)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss, norm)
    return float(loss.item()), int(np.asarray(loss_mask).astype(bool).sum())


def train_steps(
    model: Any,
    optimizer: Any,
    stage: Any,
    *,
    start_step: int,
    steps: int,
    gradient_clip: float,
    verifier_balance_maximum: float,
    mx: Any,
    nn: Any,
    optim: Any,
) -> dict[str, Any]:
    loss_and_grad = nn.value_and_grad(model, causal_loss)
    (
        verifier_positive_weights,
        verifier_negative_weights,
        verifier_class_weight_receipt,
    ) = balanced_binary_class_weights(
        stage.kerc_verifier_labels,
        maximum=float(verifier_balance_maximum),
        require_both_classes=True,
    )
    matrix_verifier_positive_weights = mx.array(
        verifier_positive_weights, dtype=mx.float32
    )
    matrix_verifier_negative_weights = mx.array(
        verifier_negative_weights, dtype=mx.float32
    )
    mx.eval(matrix_verifier_positive_weights, matrix_verifier_negative_weights)
    losses: list[float] = []
    target_positions = 0
    started = time.perf_counter()
    model.train()
    for offset in range(steps):
        index = (start_step + offset) % len(stage.inputs)
        loss, positions = one_update(
            model,
            optimizer,
            loss_and_grad,
            stage,
            index,
            gradient_clip=gradient_clip,
            verifier_positive_weights=matrix_verifier_positive_weights,
            verifier_negative_weights=matrix_verifier_negative_weights,
            mx=mx,
            nn=nn,
            optim=optim,
        )
        losses.append(loss)
        target_positions += positions
    elapsed = time.perf_counter() - started
    return {
        "optimizer_steps": steps,
        "target_positions": target_positions,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "wall_seconds": elapsed,
        "target_tokens_per_second": target_positions / max(elapsed, 1e-9),
        "verifier_class_weights": verifier_class_weight_receipt,
    }


def flatten_parameter_delta(left: Any, right: Any, *, mlx_utils: Any) -> float:
    left_rows = dict(mlx_utils.tree_flatten(left.parameters()))
    right_rows = dict(mlx_utils.tree_flatten(right.parameters()))
    if left_rows.keys() != right_rows.keys():
        raise ValueError("model parameter inventories differ after resume")
    return max(
        float(np.max(np.abs(np.asarray(left_rows[name]) - np.asarray(right_rows[name]))))
        for name in left_rows
    )


def nested_logit_delta(left: list[list[float]], right: list[list[float]]) -> float:
    values = [
        abs(a - b)
        for left_row, right_row in zip(left, right)
        for a, b in zip(left_row, right_row)
    ]
    return float(max(values or [0.0]))


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep scalar evidence and content identity, not bulky diagnostic vectors."""

    private = {
        "expected_token_logits": metrics.get("expected_token_logits") or [],
        "verifier_logits": metrics.get("verifier_logits") or [],
    }
    return {
        key: value
        for key, value in metrics.items()
        if key not in private
    } | {
        "diagnostic_vectors_sha256": canonical_sha256(private),
        "diagnostic_vectors_embedded": False,
    }


def train_mechanism_ablations(
    target: dict[str, Any],
    copy_lookup: np.ndarray,
    stage: Any,
    config: dict[str, Any],
    *,
    mx: Any,
    nn: Any,
    optim: Any,
) -> dict[str, Any]:
    """Train matched same-seed controls with one named KERC mechanism removed."""

    variants = {
        "without_stage_routing": {"kerc_stage_routing_ablation": "zero"},
        "without_hierarchical_residual": {"kerc_residual_ablation": "zero"},
        "without_independent_verifier": {"kerc_verifier_ablation": "zero"},
    }
    optimization = config["optimization"]
    steps = int(optimization["steps_before_resume"]) + int(
        optimization["steps_after_resume"]
    )
    receipts: dict[str, Any] = {}
    for name, ablation in variants.items():
        mx.random.seed(int(config["seed"]))
        model = build_faithful_model(
            target,
            copy_lookup=copy_lookup,
            mx=mx,
            nn=nn,
            ablations=ablation,
        )
        mx.eval(model.parameters())
        optimizer = optim.AdamW(
            learning_rate=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        before = evaluate_model(model, stage, mx=mx, nn=nn)
        training = train_steps(
            model,
            optimizer,
            stage,
            start_step=0,
            steps=steps,
            gradient_clip=float(optimization["gradient_clip_norm"]),
            verifier_balance_maximum=float(
                optimization["verifier_class_balance_maximum"]
            ),
            mx=mx,
            nn=nn,
            optim=optim,
        )
        after = evaluate_model(model, stage, mx=mx, nn=nn)
        receipts[name] = {
            "ablation": ablation,
            "seed": int(config["seed"]),
            "same_subset": True,
            "same_optimizer_steps": True,
            "same_optimizer_hyperparameters": True,
            "training": training,
            "before": compact_metrics(before),
            "after": compact_metrics(after),
            "capability_claim": "NONE_MECHANICS_ONLY",
            "negative_verdict_authority": "NONE",
        }
    return receipts


def expected_logit_delta_by_objective(
    baseline: dict[str, Any],
    changed: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, float]:
    position = {row_index: local for local, row_index in enumerate(selection["row_indices"])}
    return {
        objective: nested_logit_delta(
            [baseline["expected_token_logits"][position[indices[0]]]],
            [changed["expected_token_logits"][position[indices[0]]]],
        )
        for objective, indices in selection["rows_by_objective"].items()
    }


def intervention_receipts(
    checkpoint: Path,
    target: dict[str, Any],
    copy_lookup: np.ndarray,
    stage: Any,
    selection: dict[str, Any],
    baseline: dict[str, Any],
    *,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    def loaded() -> Any:
        model = build_faithful_model(target, copy_lookup=copy_lookup, mx=mx, nn=nn)
        model.load_weights(str(checkpoint), strict=True)
        mx.eval(model.parameters())
        return model

    route_model = loaded()
    route_inputs = np.asarray(stage.inputs).copy()
    for token_id in target["model"]["kerc_task_token_ids"]:
        route_inputs[route_inputs == int(token_id)] = 0
    route_stage = SimpleNamespace(**{**stage.__dict__, "inputs": route_inputs})
    route_metrics = evaluate_model(route_model, route_stage, mx=mx, nn=nn)

    residual_model = loaded()
    residual_model.kerc_residual_values.weight = mx.zeros_like(
        residual_model.kerc_residual_values.weight
    )
    mx.eval(residual_model.parameters())
    residual_metrics = evaluate_model(residual_model, stage, mx=mx, nn=nn)

    verifier_model = loaded()
    verifier_model.kerc_verifier_classifier.weight = mx.zeros_like(
        verifier_model.kerc_verifier_classifier.weight
    )
    verifier_model.kerc_verifier_classifier.bias = mx.zeros_like(
        verifier_model.kerc_verifier_classifier.bias
    )
    mx.eval(verifier_model.parameters())
    verifier_metrics = evaluate_model(verifier_model, stage, mx=mx, nn=nn)
    verifier_delta = nested_logit_delta(
        baseline["verifier_logits"], verifier_metrics["verifier_logits"]
    )
    return {
        "trusted_stage_token_removed": {
            "delta_by_objective": expected_logit_delta_by_objective(
                baseline, route_metrics, selection
            )
        },
        "hierarchical_residual_values_zeroed": {
            "delta_by_objective": expected_logit_delta_by_objective(
                baseline, residual_metrics, selection
            ),
            "expected_active_objectives": [
                "surface_to_kernel_program_v1",
                "answer_packet_to_surface_v1",
            ],
            "expected_inactive_objectives": [
                "surface_direct_control_v1",
                "kernel_program_to_answer_packet_v1",
            ],
        },
        "independent_verifier_classifier_zeroed": {
            "maximum_verifier_logit_delta": verifier_delta
        },
    }


def classify_gates(
    config: dict[str, Any],
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    first_phase: dict[str, Any],
    second_phase: dict[str, Any],
    reload_delta: float,
    resume_delta: float,
    interventions: dict[str, Any],
    migration_rejection: bool,
    schema_migration_valid: bool,
    schema_unknown_rejection: bool,
    migration_logit_delta: float,
    rollback_logit_delta: float,
    trained_ablations: dict[str, Any],
    partial_file_count: int,
) -> tuple[str, list[dict[str, Any]]]:
    accept = config["acceptance"]
    checks: list[tuple[str, bool, Any, str]] = []
    loss_ratio = after["mean_total_loss"] / max(before["mean_total_loss"], 1e-12)
    checks.append(("tiny_subset_loss_reduction", loss_ratio <= accept["maximum_final_to_initial_loss_ratio"], loss_ratio, "adequacy"))
    token_gain = after["token_accuracy"] - before["token_accuracy"]
    checks.append(("tiny_subset_token_accuracy_gain", token_gain >= accept["minimum_token_accuracy_gain"], token_gain, "adequacy"))
    residual_observed = {
        "informative_channel_count": after["residual_informative_channel_count"],
        "macro_balanced_accuracy": after["residual_informative_macro_balanced_accuracy"],
    }
    checks.append(("residual_allocator_learned", after["residual_informative_channel_count"] >= accept["minimum_residual_informative_channel_count"] and after["residual_informative_macro_balanced_accuracy"] is not None and after["residual_informative_macro_balanced_accuracy"] >= accept["minimum_residual_macro_balanced_accuracy"], residual_observed, "adequacy"))
    verifier_observed = {
        "macro_balanced_accuracy": after["verifier_macro_balanced_accuracy"],
        "minimum_negative_recall": after["verifier_minimum_negative_recall"],
    }
    checks.append(("verifier_learned", after["verifier_macro_balanced_accuracy"] is not None and after["verifier_macro_balanced_accuracy"] >= accept["minimum_verifier_macro_balanced_accuracy"] and after["verifier_minimum_negative_recall"] is not None and after["verifier_minimum_negative_recall"] >= accept["minimum_verifier_negative_recall"], verifier_observed, "adequacy"))
    checks.append(("checkpoint_reload_equivalent", reload_delta <= accept["maximum_checkpoint_reload_logit_delta"], reload_delta, "hard"))
    checks.append(("optimizer_resume_equivalent", resume_delta <= accept["maximum_resume_equivalence_logit_delta"], resume_delta, "hard"))
    checks.append(("resume_mismatch_rejected", migration_rejection, migration_rejection, "hard"))
    checks.append(("checkpoint_schema_migration_valid", schema_migration_valid, schema_migration_valid, "hard"))
    checks.append(("unknown_checkpoint_schema_rejected", schema_unknown_rejection, schema_unknown_rejection, "hard"))
    checks.append(("migration_behavior_equivalent", migration_logit_delta <= accept["maximum_checkpoint_migration_logit_delta"], migration_logit_delta, "hard"))
    checks.append(("rollback_behavior_equivalent", rollback_logit_delta <= accept["maximum_checkpoint_rollback_logit_delta"], rollback_logit_delta, "hard"))
    required_ablations = set(config["required_trained_ablations"])
    checks.append(("trained_mechanism_ablation_inventory", set(trained_ablations) == required_ablations, sorted(trained_ablations), "hard"))
    activity_limit = float(accept["maximum_disabled_mechanism_activity"])
    disabled_activity = {
        "without_stage_routing": trained_ablations.get("without_stage_routing", {}).get("after", {}).get("mechanism_activity", {}).get("stage_weights_maximum_absolute"),
        "without_hierarchical_residual": trained_ablations.get("without_hierarchical_residual", {}).get("after", {}).get("mechanism_activity", {}).get("residual_logits_maximum_absolute"),
        "without_independent_verifier": trained_ablations.get("without_independent_verifier", {}).get("after", {}).get("mechanism_activity", {}).get("verifier_logits_maximum_absolute"),
    }
    checks.append(("trained_ablations_remove_named_mechanism", all(value is not None and float(value) <= activity_limit for value in disabled_activity.values()), disabled_activity, "hard"))
    checks.append(("temporary_artifacts_cleaned", partial_file_count == 0, partial_file_count, "hard"))
    minimum_delta = float(accept["minimum_active_intervention_logit_delta"])
    route_deltas = interventions["trusted_stage_token_removed"]["delta_by_objective"]
    checks.append(("trusted_stage_route_is_causal", all(value > minimum_delta for value in route_deltas.values()), route_deltas, "adequacy"))
    residual = interventions["hierarchical_residual_values_zeroed"]
    checks.append(("residual_path_is_causal_when_active", all(residual["delta_by_objective"][name] > minimum_delta for name in residual["expected_active_objectives"]), residual["delta_by_objective"], "adequacy"))
    inactive_max = float(accept["maximum_inactive_residual_intervention_logit_delta"])
    checks.append(("residual_path_is_scoped_when_inactive", all(residual["delta_by_objective"][name] <= inactive_max for name in residual["expected_inactive_objectives"]), residual["delta_by_objective"], "hard"))
    checks.append(("independent_verifier_is_causal", interventions["independent_verifier_classifier_zeroed"]["maximum_verifier_logit_delta"] > minimum_delta, interventions["independent_verifier_classifier_zeroed"], "adequacy"))
    throughput = min(first_phase["target_tokens_per_second"], second_phase["target_tokens_per_second"])
    checks.append(("multi_batch_mlx_throughput", throughput >= accept["minimum_target_tokens_per_second"], throughput, "adequacy"))
    rows = [
        {"gate": name, "passed": bool(passed), "observed": observed, "severity": severity}
        for name, passed, observed, severity in checks
    ]
    if any(not row["passed"] and row["severity"] == "hard" for row in rows):
        return "RED", rows
    if any(not row["passed"] for row in rows):
        return "INCONCLUSIVE_EXPERIMENT", rows
    return "GREEN", rows


def run(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("policy") != "project_theseus_kerc_adequacy_canary_v1":
        raise ValueError("KERC adequacy policy mismatch")
    boundaries = config.get("boundaries") or {}
    if any(int(boundaries.get(key) or 0) for key in (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
    )):
        raise ValueError("KERC adequacy canary boundary is not zero")
    optimization = config["optimization"]
    total_steps = int(optimization["steps_before_resume"]) + int(optimization["steps_after_resume"])
    if total_steps > int(optimization["maximum_total_steps"]):
        raise ValueError("KERC adequacy update budget exceeds the frozen cap")

    training_path = resolve(config["training_config"])
    training = read_json(training_path)
    plan = build_plan(training, config_path=training_path)
    if plan["trigger_state"] != "GREEN":
        raise ValueError("canonical MoECOT plan is not GREEN")
    target = plan["targets"][config["target_id"]]
    base = read_json(resolve(training["base_config"]))
    metadata = read_json(resolve(training["stage_dir"]) / "stage_metadata_v1.json")
    stage = materialize_target_supervision(
        training,
        base,
        target,
        metadata=metadata,
        artifact_field="kernel_english_artifacts",
        receipt_policy="project_theseus_moecot_kernel_english_arrays_v1",
        maximum_sequence_tokens=int(training["kernel_english_training"]["maximum_sequence_tokens"]),
        objective_filter=tuple(target["kernel_english_objectives"]),
    )
    subset, selection = select_balanced_subset(
        stage,
        task_token_ids=tuple(target["model"]["kerc_task_token_ids"]),
        separator_id=int(target["model"]["source_target_separator_token_id"]),
        positives_per_objective=int(config["subset"]["positive_rows_per_objective"]),
    )
    if len(subset.inputs) > int(config["subset"]["maximum_rows"]):
        raise ValueError("KERC adequacy subset exceeds its frozen row cap")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    mx.random.seed(int(config["seed"]))
    copy_lookup = build_source_to_target_lookup(
        base, metadata, vocab_size=int(target["vocab_size"])
    )
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = output_root / "kerc_adequacy_weights.npz"
    optimizer_path = output_root / "kerc_adequacy_optimizer.safetensors"
    initial_checkpoint = output_root / "kerc_adequacy_initial_weights.npz"
    migrated_checkpoint = output_root / "kerc_adequacy_weights.v1.safetensors"
    migrated_optimizer = output_root / "kerc_adequacy_optimizer.v1.safetensors"
    migration_manifest_path = output_root / "kerc_adequacy_checkpoint_manifest.v1.json"
    rollback_checkpoint = output_root / "kerc_adequacy_weights.rollback.npz"
    rollback_optimizer = output_root / "kerc_adequacy_optimizer.rollback.safetensors"
    for stale in output_root.glob("*.partial*"):
        stale.unlink()

    model = build_faithful_model(target, copy_lookup=copy_lookup, mx=mx, nn=nn)
    mx.eval(model.parameters())
    model.save_weights(str(initial_checkpoint))
    optimizer = optim.AdamW(
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    before = evaluate_model(model, subset, mx=mx, nn=nn)
    rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    first_phase = train_steps(
        model,
        optimizer,
        subset,
        start_step=0,
        steps=int(optimization["steps_before_resume"]),
        gradient_clip=float(optimization["gradient_clip_norm"]),
        verifier_balance_maximum=float(
            optimization["verifier_class_balance_maximum"]
        ),
        mx=mx,
        nn=nn,
        optim=optim,
    )
    model.save_weights(str(checkpoint))
    publish_optimizer(mx, mlx_utils, optimizer, optimizer_path)
    checkpoint_metrics = evaluate_model(model, subset, mx=mx, nn=nn)

    reloaded = build_faithful_model(target, copy_lookup=copy_lookup, mx=mx, nn=nn)
    reloaded.load_weights(str(checkpoint), strict=True)
    reloaded_optimizer = optim.AdamW(
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    reloaded_optimizer.state = mlx_utils.tree_unflatten(
        list(mx.load(str(optimizer_path)).items())
    )
    mx.eval(reloaded.parameters(), reloaded_optimizer.state)
    reloaded_metrics = evaluate_model(reloaded, subset, mx=mx, nn=nn)
    reload_delta = nested_logit_delta(
        checkpoint_metrics["expected_token_logits"],
        reloaded_metrics["expected_token_logits"],
    )

    second_phase_left = train_steps(
        model,
        optimizer,
        subset,
        start_step=int(optimization["steps_before_resume"]),
        steps=int(optimization["steps_after_resume"]),
        gradient_clip=float(optimization["gradient_clip_norm"]),
        verifier_balance_maximum=float(
            optimization["verifier_class_balance_maximum"]
        ),
        mx=mx,
        nn=nn,
        optim=optim,
    )
    second_phase = train_steps(
        reloaded,
        reloaded_optimizer,
        subset,
        start_step=int(optimization["steps_before_resume"]),
        steps=int(optimization["steps_after_resume"]),
        gradient_clip=float(optimization["gradient_clip_norm"]),
        verifier_balance_maximum=float(
            optimization["verifier_class_balance_maximum"]
        ),
        mx=mx,
        nn=nn,
        optim=optim,
    )
    resume_parameter_delta = flatten_parameter_delta(model, reloaded, mlx_utils=mlx_utils)
    after = evaluate_model(reloaded, subset, mx=mx, nn=nn)
    reloaded.save_weights(str(checkpoint))
    publish_optimizer(mx, mlx_utils, reloaded_optimizer, optimizer_path)
    checkpoint_binding = {
        "target_id": target["target_id"],
        "role": target["role"],
        "model_config_sha256": canonical_sha256(target["model"]),
        "plan_sha256": plan["plan_sha256"],
        "stage_signature": plan["stage"]["stage_signature"],
        "vocab_size": target["vocab_size"],
        "kernel_code_vocabulary_sha256": target["kernel_code_vocabulary"]["payload"]["contract_sha256"],
    }
    migration_manifest = migrate_legacy_checkpoint(
        legacy_checkpoint=checkpoint,
        legacy_optimizer=optimizer_path,
        checkpoint=migrated_checkpoint,
        optimizer=migrated_optimizer,
        manifest_path=migration_manifest_path,
        binding=checkpoint_binding,
    )
    validate_checkpoint_contract(
        migration_manifest,
        checkpoint=migrated_checkpoint,
        optimizer=migrated_optimizer,
        binding=checkpoint_binding,
    )
    migrated_model = build_faithful_model(target, copy_lookup=copy_lookup, mx=mx, nn=nn)
    migrated_model.load_weights(str(migrated_checkpoint), strict=True)
    migrated_metrics = evaluate_model(migrated_model, subset, mx=mx, nn=nn)
    migration_logit_delta = nested_logit_delta(
        after["expected_token_logits"], migrated_metrics["expected_token_logits"]
    )
    rollback_receipt = rollback_checkpoint_contract(
        migration_manifest,
        checkpoint=migrated_checkpoint,
        optimizer=migrated_optimizer,
        rollback_checkpoint=rollback_checkpoint,
        rollback_optimizer=rollback_optimizer,
        binding=checkpoint_binding,
    )
    rollback_model = build_faithful_model(target, copy_lookup=copy_lookup, mx=mx, nn=nn)
    rollback_model.load_weights(str(rollback_checkpoint), strict=True)
    rollback_metrics = evaluate_model(rollback_model, subset, mx=mx, nn=nn)
    rollback_logit_delta = nested_logit_delta(
        after["expected_token_logits"], rollback_metrics["expected_token_logits"]
    )
    schema_unknown_rejection = False
    unknown_manifest = {**migration_manifest, "schema_version": 999}
    unknown_unsigned = dict(unknown_manifest)
    unknown_unsigned.pop("contract_sha256", None)
    unknown_manifest["contract_sha256"] = checkpoint_contract_sha256(unknown_unsigned)
    try:
        validate_checkpoint_contract(
            unknown_manifest,
            checkpoint=migrated_checkpoint,
            optimizer=migrated_optimizer,
            binding=checkpoint_binding,
        )
    except ValueError as exc:
        schema_unknown_rejection = "unsupported_schema_version" in str(exc)
    schema_migration_valid = (
        migration_manifest["source"]["checkpoint_inventory"]["inventory_sha256"]
        == migration_manifest["target"]["checkpoint_inventory"]["inventory_sha256"]
        and migration_manifest["source"]["optimizer_inventory"]["inventory_sha256"]
        == migration_manifest["target"]["optimizer_inventory"]["inventory_sha256"]
        and rollback_receipt["checkpoint_inventory_sha256"]
        == migration_manifest["source"]["checkpoint_inventory"]["inventory_sha256"]
        and rollback_receipt["optimizer_inventory_sha256"]
        == migration_manifest["source"]["optimizer_inventory"]["inventory_sha256"]
    )
    trained_ablations = train_mechanism_ablations(
        target,
        copy_lookup,
        subset,
        config,
        mx=mx,
        nn=nn,
        optim=optim,
    )
    interventions = intervention_receipts(
        checkpoint,
        target,
        copy_lookup,
        subset,
        selection,
        after,
        mx=mx,
        nn=nn,
    )

    valid_resume_receipt = {
        "policy": "project_theseus_moecot_language_arm_training_receipt_v1",
        "target_id": target["target_id"],
        "plan_sha256": plan["plan_sha256"],
        "stage_signature": plan["stage"]["stage_signature"],
        "row_ranges": target["row_ranges"],
        "vocab_size": target["vocab_size"],
        "kernel_code_vocabulary_sha256": target["kernel_code_vocabulary"]["payload"]["contract_sha256"],
        "checkpoint_schema_policy": target["checkpoint_schema_policy"],
        "checkpoint_schema": target["checkpoint_schema"],
        "checkpoint_schema_version": target["checkpoint_schema_version"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "optimizer_state_sha256": sha256_file(optimizer_path),
    }
    validate_resume(valid_resume_receipt, plan, target, checkpoint, optimizer_path)
    invalid_receipt = {**valid_resume_receipt, "kernel_code_vocabulary_sha256": "sha256:invalid"}
    migration_rejection = False
    try:
        validate_resume(invalid_receipt, plan, target, checkpoint, optimizer_path)
    except ValueError as exc:
        migration_rejection = "kernel_code_vocabulary_identity_mismatch" in str(exc)
    partial_files = list(output_root.glob("*.partial*"))
    trigger_state, gates = classify_gates(
        config,
        before=before,
        after=after,
        first_phase=first_phase,
        second_phase=second_phase,
        reload_delta=reload_delta,
        resume_delta=resume_parameter_delta,
        interventions=interventions,
        migration_rejection=migration_rejection,
        schema_migration_valid=schema_migration_valid,
        schema_unknown_rejection=schema_unknown_rejection,
        migration_logit_delta=migration_logit_delta,
        rollback_logit_delta=rollback_logit_delta,
        trained_ablations=trained_ablations,
        partial_file_count=len(partial_files),
    )
    rss_after = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "policy": config["policy"],
        "trigger_state": trigger_state,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation": relative(Path(__file__)),
        "implementation_sha256": sha256_file(Path(__file__)),
        "training_config": relative(training_path),
        "training_config_sha256": sha256_file(training_path),
        "plan_sha256": plan["plan_sha256"],
        "target_id": target["target_id"],
        "target_parameter_count": int(target["parameter_count"]),
        "stage_manifest_sha256": plan["kernel_english_training"]["manifest_sha256"],
        "code_vocabulary_sha256": target["kernel_code_vocabulary"]["payload"]["contract_sha256"],
        "selection": selection,
        "before": compact_metrics(before),
        "after": compact_metrics(after),
        "optimization": {
            "first_phase": first_phase,
            "continuous_second_phase": second_phase_left,
            "resumed_second_phase": second_phase,
            "checkpoint_reload_maximum_logit_delta": reload_delta,
            "resume_maximum_parameter_delta": resume_parameter_delta,
        },
        "interventions": interventions,
        "trained_mechanism_ablations": trained_ablations,
        "lifecycle": {
            "valid_resume_accepted": True,
            "mismatched_codebook_resume_rejected": migration_rejection,
            "schema_migration_valid": schema_migration_valid,
            "unknown_schema_rejected": schema_unknown_rejection,
            "migration_maximum_logit_delta": migration_logit_delta,
            "rollback_maximum_logit_delta": rollback_logit_delta,
            "migration_manifest": relative(migration_manifest_path),
            "migration_manifest_sha256": sha256_file(migration_manifest_path),
            "migration_contract_sha256": migration_manifest["contract_sha256"],
            "migration_source_schema": migration_manifest["source_schema"],
            "migration_target_schema": migration_manifest["target_schema"],
            "rollback_receipt": rollback_receipt,
            "partial_file_count": len(partial_files),
            "checkpoint": relative(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer_state": relative(optimizer_path),
            "optimizer_state_sha256": sha256_file(optimizer_path),
        },
        "resource": {
            "maximum_rss_before": rss_before,
            "maximum_rss_after": rss_after,
            "maximum_rss_delta": max(0, rss_after - rss_before),
            "backend": "mlx_metal",
            "batch_count": total_steps * 2,
        },
        "gates": gates,
        "hard_gaps": [
            row["gate"] for row in gates if not row["passed"] and row["severity"] == "hard"
        ],
        "inconclusive_gaps": [
            row["gate"] for row in gates if not row["passed"] and row["severity"] == "adequacy"
        ],
        "capability_claim": "NONE_MECHANICS_ONLY",
        "negative_verdict_authority": "NONE",
        **boundaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    if not args.execute:
        report = {
            "policy": config.get("policy"),
            "trigger_state": "PLANNED",
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "score_semantics": "mechanics adequacy only; no capability or negative-verdict authority",
            **(config.get("boundaries") or {}),
        }
    else:
        report = run(config_path)
    output = resolve(args.out or config["report"])
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
