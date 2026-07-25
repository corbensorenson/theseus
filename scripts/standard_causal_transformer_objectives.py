"""Losses and label preparation for the standard causal transformer.

This module is deliberately model-agnostic: MLX objects are supplied by callers so
the objective remains replayable under the canonical training runtime.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from typing import Any

import numpy as np


def mtp_curriculum_scale(
    step: int,
    *,
    warmup_steps: int,
    ramp_steps: int,
    maximum: float,
) -> float:
    """Return a monotonic NTP-to-MTP auxiliary-loss scale.

    The base next-token objective remains active throughout.  A zero-valued
    prefix lets the causal model establish ordinary token prediction before
    the future-token auxiliary receives optimizer authority.
    """

    if step < 0 or warmup_steps < 0 or ramp_steps <= 0:
        raise ValueError("MTP curriculum steps must be nonnegative with a positive ramp")
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("MTP curriculum maximum must be finite and nonnegative")
    if step < warmup_steps:
        return 0.0
    progress = min(1.0, (step - warmup_steps + 1) / float(ramp_steps))
    return float(maximum) * progress


def prepare_semantic_plan_labels(
    labels: np.ndarray | None,
    *,
    mode: str,
    seed: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Prepare the semantic or causally-invalid shuffled control labels."""

    if labels is None or mode == "none":
        return None, {
            "label_mode": "none",
            "row_count": 0,
            "fixed_point_count": 0,
            "label_sha256": "",
        }
    matrix = np.asarray(labels, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("semantic plan labels must be a non-empty rank-two matrix")
    if mode == "semantic":
        prepared = matrix
        fixed_points = len(matrix)
    elif mode == "shuffled":
        if len(matrix) < 2:
            raise ValueError("shuffled semantic plan control requires at least two rows")
        order = np.random.default_rng(seed).permutation(len(matrix))
        assignment = np.empty_like(order)
        assignment[order] = np.roll(order, 1)
        prepared = matrix[assignment]
        fixed_points = int(np.sum(assignment == np.arange(len(matrix))))
        if fixed_points:
            raise AssertionError("semantic plan shuffled control contains fixed points")
    elif mode == "dropout":
        prepared = np.zeros_like(matrix)
        fixed_points = int(np.sum(np.all(prepared == matrix, axis=1)))
    else:
        raise ValueError(f"unsupported semantic plan label mode: {mode}")
    return prepared, {
        "label_mode": mode,
        "row_count": len(prepared),
        "feature_count": int(prepared.shape[1]),
        "positive_label_count": int(prepared.sum()),
        "label_density": round(float(prepared.mean()), 8),
        "fixed_point_count": fixed_points,
        "label_sha256": hashlib.sha256(prepared.tobytes()).hexdigest(),
    }


def semantic_plan_positive_weights(
    labels: np.ndarray | None, *, maximum: float = 16.0
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Balance sparse plan obligations without changing row or feature mass."""

    if labels is None:
        return None, {"state": "NOT_APPLICABLE", "weight_sha256": ""}
    matrix = np.asarray(labels, dtype=np.float32)
    positives = matrix.sum(axis=0)
    negatives = len(matrix) - positives
    weights = np.ones((matrix.shape[1],), dtype=np.float32)
    observed = positives > 0
    weights[observed] = np.clip(
        negatives[observed] / positives[observed], 1.0, float(maximum)
    )
    return weights, {
        "state": "MEASURED",
        "feature_count": int(len(weights)),
        "unobserved_feature_count": int((~observed).sum()),
        "minimum": round(float(weights.min()), 8),
        "maximum": round(float(weights.max()), 8),
        "mean": round(float(weights.mean()), 8),
        "cap": float(maximum),
        "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
    }


def balanced_binary_class_weights(
    labels: np.ndarray,
    *,
    maximum: float = 16.0,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build feature-local inverse-frequency weights for a binary objective.

    KERC verifier corruptions are sparse by construction: a negative example
    invalidates one verifier dimension while the other dimensions remain true.
    A raw mean BCE therefore rewards an all-positive classifier.  These weights
    make each observed class contribute equal mass within each dimension.
    """

    matrix = np.asarray(labels, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] <= 0:
        raise ValueError("balanced binary labels must be a non-empty rank-two matrix")
    if not np.all((matrix == 0.0) | (matrix == 1.0)):
        raise ValueError("balanced binary labels must contain only zero or one")
    if maximum < 1.0 or not math.isfinite(maximum):
        raise ValueError("maximum binary class weight must be finite and at least one")
    positives = matrix.sum(axis=0)
    negatives = float(len(matrix)) - positives
    missing = (positives == 0.0) | (negatives == 0.0)
    if require_both_classes and bool(np.any(missing)):
        indices = np.flatnonzero(missing).tolist()
        raise ValueError(
            f"balanced binary objective requires both classes in features {indices}"
        )
    positive_weights = np.ones_like(positives, dtype=np.float32)
    negative_weights = np.ones_like(negatives, dtype=np.float32)
    observed = ~missing
    positive_weights[observed] = np.clip(
        float(len(matrix)) / (2.0 * positives[observed]),
        1.0 / float(maximum),
        float(maximum),
    )
    negative_weights[observed] = np.clip(
        float(len(matrix)) / (2.0 * negatives[observed]),
        1.0 / float(maximum),
        float(maximum),
    )
    inventory = np.stack((positive_weights, negative_weights), axis=0)
    return positive_weights, negative_weights, {
        "state": "MEASURED",
        "row_count": int(len(matrix)),
        "feature_count": int(matrix.shape[1]),
        "positive_counts": [int(value) for value in positives],
        "negative_counts": [int(value) for value in negatives],
        "missing_class_features": np.flatnonzero(missing).tolist(),
        "require_both_classes": bool(require_both_classes),
        "maximum_class_weight": float(maximum),
        "minimum_weight": round(float(inventory.min()), 8),
        "maximum_weight": round(float(inventory.max()), 8),
        "weight_sha256": hashlib.sha256(inventory.tobytes()).hexdigest(),
    }


def balanced_categorical_class_weights(
    labels: np.ndarray,
    *,
    class_count: int,
    maximum: float,
    require_two_classes_per_feature: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive per-channel inverse-frequency weights for categorical auxiliaries."""

    matrix = np.asarray(labels, dtype=np.int32)
    if matrix.ndim != 2 or not len(matrix) or class_count <= 1:
        raise ValueError("balanced categorical objective requires a nonempty label matrix")
    if np.any(matrix < 0) or np.any(matrix >= int(class_count)):
        raise ValueError("balanced categorical labels exceed the declared class inventory")
    if not np.isfinite(maximum) or maximum < 1.0:
        raise ValueError("maximum categorical class weight must be finite and at least one")
    counts = np.stack(
        [
            np.bincount(matrix[:, feature], minlength=class_count)
            for feature in range(matrix.shape[1])
        ],
        axis=0,
    ).astype(np.float32)
    observed = counts > 0.0
    observed_counts = observed.sum(axis=1)
    if require_two_classes_per_feature and bool(np.any(observed_counts < 2)):
        features = np.flatnonzero(observed_counts < 2).tolist()
        raise ValueError(
            f"balanced categorical objective requires two classes in features {features}"
        )
    weights = np.ones_like(counts, dtype=np.float32)
    for feature in range(matrix.shape[1]):
        active = observed[feature]
        weights[feature, active] = np.clip(
            float(len(matrix))
            / (float(observed_counts[feature]) * counts[feature, active]),
            1.0 / float(maximum),
            float(maximum),
        )
    return weights, {
        "state": "MEASURED",
        "row_count": int(len(matrix)),
        "feature_count": int(matrix.shape[1]),
        "class_count": int(class_count),
        "counts_by_feature_and_class": [
            [int(value) for value in row] for row in counts
        ],
        "observed_class_count_by_feature": [
            int(value) for value in observed_counts
        ],
        "require_two_classes_per_feature": bool(require_two_classes_per_feature),
        "maximum_class_weight": float(maximum),
        "minimum_weight": round(float(weights.min()), 8),
        "maximum_weight": round(float(weights.max()), 8),
        "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
    }


def causal_loss(
    model: Any,
    inputs: Any,
    labels: Any,
    mask: Any,
    mx: Any,
    nn: Any,
    plan_labels: Any | None = None,
    plan_weight: float = 0.0,
    plan_positive_weights: Any | None = None,
    plan_loss_mode: str = "binary_multilabel",
    plan_slot_count: int = 0,
    plan_factor_group_sizes: tuple[int, ...] = (),
    kerc_residual_labels: Any | None = None,
    kerc_residual_weight: float = 0.0,
    kerc_verifier_labels: Any | None = None,
    kerc_verifier_weight: float = 0.0,
    kerc_verifier_positive_weights: Any | None = None,
    kerc_verifier_negative_weights: Any | None = None,
    kerc_residual_class_weights: Any | None = None,
    kerc_residual_loss_mask: Any | None = None,
    kerc_decision_labels: Any | None = None,
    kerc_decision_weight: float = 0.0,
    kerc_decision_class_weights: Any | None = None,
    kerc_decision_loss_mask: Any | None = None,
    kerc_unit_residual_labels: Any | None = None,
    kerc_unit_residual_weight: float = 0.0,
    kerc_unit_residual_loss_mask: Any | None = None,
    kerc_unit_confidence_targets: Any | None = None,
    kerc_unit_byte_ids: Any | None = None,
    kerc_unit_byte_mask: Any | None = None,
    kerc_unit_byte_offsets: Any | None = None,
    kerc_unit_kind_ids: Any | None = None,
    kerc_unit_candidate_features: Any | None = None,
    kerc_unit_mask: Any | None = None,
    kerc_unit_hard_block_mask: Any | None = None,
    kerc_unit_class_weights: Any | None = None,
    source_conditioning: bool | None = None,
    prune_inactive_auxiliary_outputs: bool = True,
    prune_zero_authority_decoder: bool = True,
    normalization_mask: Any | None = None,
) -> Any:
    copy_aux = None
    copy_weight = float(getattr(model, "copy_auxiliary_loss_weight", 0.0))
    mtp_weight = float(getattr(model, "mtp_loss_scale", 0.0))
    needs_plan = plan_labels is not None and plan_weight > 0.0
    needs_kerc = (
        kerc_residual_labels is not None and kerc_residual_weight > 0.0
    ) or (
        kerc_unit_residual_labels is not None and kerc_unit_residual_weight > 0.0
    ) or (kerc_verifier_labels is not None and kerc_verifier_weight > 0.0) or (
        kerc_decision_labels is not None and kerc_decision_weight > 0.0
    )
    unit_conditioning_inputs = (
        kerc_unit_byte_ids,
        kerc_unit_kind_ids,
        kerc_unit_candidate_features,
        kerc_unit_mask,
        kerc_unit_hard_block_mask,
    )
    unit_conditioning_active = any(
        value is not None for value in unit_conditioning_inputs
    )
    if unit_conditioning_active and (
        not all(value is not None for value in unit_conditioning_inputs)
        or (kerc_unit_byte_mask is None) == (kerc_unit_byte_offsets is None)
    ):
        raise ValueError(
            "KERC generator unit conditioning requires a complete byte layout and unit inputs"
        )
    token_supervision_active = bool(mx.any(mask > 0.0))
    copy_loss_active = (
        token_supervision_active
        and copy_weight > 0.0
        and source_conditioning is not False
    )
    mtp_loss_active = token_supervision_active and mtp_weight > 0.0
    auxiliary_only = (
        prune_zero_authority_decoder
        and not token_supervision_active
        and needs_kerc
        and not needs_plan
    )
    auxiliary_outputs_required = (
        needs_plan
        or copy_loss_active
        or mtp_loss_active
        or needs_kerc
        or unit_conditioning_active
    )
    if not prune_inactive_auxiliary_outputs:
        auxiliary_outputs_required = (
            needs_plan or copy_weight > 0.0 or mtp_weight > 0.0 or needs_kerc
        )
    if auxiliary_outputs_required:
        logits, _cache, training_aux = model(
            inputs,
            source_conditioning=source_conditioning,
            return_training_aux=True,
            kerc_unit_byte_ids=kerc_unit_byte_ids,
            kerc_unit_byte_mask=kerc_unit_byte_mask,
            kerc_unit_byte_offsets=kerc_unit_byte_offsets,
            kerc_unit_kind_ids=kerc_unit_kind_ids,
            kerc_unit_candidate_features=kerc_unit_candidate_features,
            kerc_unit_mask=kerc_unit_mask,
            kerc_unit_hard_block_mask=kerc_unit_hard_block_mask,
            auxiliary_only=auxiliary_only,
            output_position_mask=(
                mask if getattr(model, "compact_output_projection", False) else None
            ),
        )
        plan_logits = training_aux.get("plan_logits")
        copy_aux = training_aux.get("copy_aux")
        mtp_logits = list(training_aux.get("mtp_logits") or [])
        kerc_aux = training_aux.get("kerc")
        if needs_plan and plan_logits is None:
            raise ValueError("semantic plan labels require an enabled learned plan head")
        if needs_kerc and not isinstance(kerc_aux, dict):
            raise ValueError("KERC labels require the faithful learned KERC modules")
    else:
        logits, _cache = model(
            inputs,
            source_conditioning=source_conditioning,
            output_position_mask=(
                mask if getattr(model, "compact_output_projection", False) else None
            ),
        )
        plan_logits = None
        mtp_logits = []
        kerc_aux = None
    output_width = int(logits.shape[1])
    output_start = int(
        (training_aux or {}).get(
            "output_position_start",
            int(labels.shape[1]) - output_width,
        )
    ) if auxiliary_outputs_required else int(labels.shape[1]) - output_width
    output_stop = int(
        (training_aux or {}).get(
            "output_position_stop",
            output_start + output_width,
        )
    ) if auxiliary_outputs_required else output_start + output_width
    if (
        output_width > int(labels.shape[1])
        or output_start < 0
        or output_stop > int(labels.shape[1])
        or output_stop - output_start != output_width
    ):
        raise ValueError("model output width exceeds label width")
    token_labels = labels[:, output_start:output_stop]
    token_mask = mask[:, output_start:output_stop]
    token_loss = nn.losses.cross_entropy(logits, token_labels)
    denominator_mask = normalization_mask if normalization_mask is not None else mask
    denominator = mx.maximum(
        mx.sum(denominator_mask), mx.array(1.0, dtype=mx.float32)
    )
    body_loss = mx.sum(token_loss * token_mask) / denominator
    if copy_aux is not None and copy_loss_active:
        body_loss = body_loss + copy_weight * pointer_generator_auxiliary_loss(
            copy_aux,
            token_labels,
            token_mask,
            mx,
            normalization_labels=labels if normalization_mask is not None else None,
            normalization_mask=normalization_mask,
        )
    if mtp_loss_active:
        body_loss = body_loss + mtp_weight * mtp_auxiliary_loss(
            mtp_logits,
            labels,
            mask,
            tuple(getattr(model, "mtp_future_offsets", ())),
            tuple(getattr(model, "mtp_loss_weights", ())),
            mx,
            nn,
        )
    if kerc_residual_labels is not None and kerc_residual_weight > 0.0:
        residual_logits = kerc_aux.get("residual_logits") if kerc_aux else None
        if residual_logits is None:
            raise ValueError("KERC residual labels require residual allocator logits")
        choices = int(residual_logits.shape[-1])
        residual_targets = kerc_residual_labels.astype(mx.int32)
        residual_element_loss = nn.losses.cross_entropy(
            residual_logits.reshape(-1, choices),
            residual_targets.reshape(-1),
        ).reshape(int(residual_logits.shape[0]), int(residual_logits.shape[1]))
        if kerc_residual_class_weights is not None:
            if tuple(kerc_residual_class_weights.shape) != (
                int(residual_logits.shape[1]),
                choices,
            ):
                raise ValueError("KERC residual class weights do not match logits")
            broadcast_weights = mx.broadcast_to(
                kerc_residual_class_weights[None, :, :],
                residual_logits.shape,
            )
            selected_weights = mx.take_along_axis(
                broadcast_weights,
                residual_targets[:, :, None],
                axis=-1,
            )[:, :, 0]
            residual_element_loss = residual_element_loss * selected_weights
        if kerc_residual_loss_mask is None:
            residual_authority = mx.ones(
                (int(residual_logits.shape[0]),), dtype=mx.float32
            )
        else:
            if tuple(kerc_residual_loss_mask.shape) != (
                int(residual_logits.shape[0]),
            ):
                raise ValueError("KERC residual loss mask must be one value per row")
            residual_authority = kerc_residual_loss_mask.astype(mx.float32)
        residual_denominator = mx.maximum(
            mx.sum(residual_authority) * int(residual_logits.shape[1]),
            mx.array(1.0, dtype=mx.float32),
        )
        residual_loss = mx.sum(
            residual_element_loss * residual_authority[:, None]
        ) / residual_denominator
        body_loss = body_loss + float(kerc_residual_weight) * residual_loss
    if kerc_unit_residual_labels is not None and kerc_unit_residual_weight > 0.0:
        unit_logits = kerc_aux.get("unit_residual_logits") if kerc_aux else None
        confidence_logits = kerc_aux.get("unit_confidence_logits") if kerc_aux else None
        if unit_logits is None or confidence_logits is None:
            raise ValueError("KERC per-unit labels require per-unit allocator logits")
        targets = kerc_unit_residual_labels.astype(mx.int32)
        choices = int(unit_logits.shape[-1])
        element_loss = nn.losses.cross_entropy(
            unit_logits.reshape(-1, choices), targets.reshape(-1)
        ).reshape(int(unit_logits.shape[0]), int(unit_logits.shape[1]))
        authority = (
            kerc_unit_residual_loss_mask.astype(mx.float32)
            if kerc_unit_residual_loss_mask is not None
            else kerc_unit_mask.astype(mx.float32)
        )
        if kerc_unit_class_weights is not None:
            if tuple(kerc_unit_class_weights.shape) != (choices,):
                raise ValueError("KERC per-unit class weights do not match choices")
            selected_weights = mx.take(
                kerc_unit_class_weights, targets.reshape(-1)
            ).reshape(targets.shape)
            element_loss = element_loss * selected_weights
        denominator = mx.maximum(
            mx.sum(authority), mx.array(1.0, dtype=mx.float32)
        )
        unit_loss = mx.sum(element_loss * authority) / denominator
        if kerc_unit_confidence_targets is not None:
            confidence_targets = kerc_unit_confidence_targets.astype(mx.float32)
            confidence_loss = (
                mx.maximum(confidence_logits, 0.0)
                - confidence_logits * confidence_targets
                + mx.log1p(mx.exp(-mx.abs(confidence_logits)))
            )
            unit_loss = unit_loss + 0.25 * (
                mx.sum(confidence_loss * authority) / denominator
            )
        body_loss = body_loss + float(kerc_unit_residual_weight) * unit_loss
    if kerc_verifier_labels is not None and kerc_verifier_weight > 0.0:
        verifier_logits = kerc_aux.get("verifier_logits") if kerc_aux else None
        if verifier_logits is None:
            raise ValueError("KERC verifier labels require full-sequence verifier logits")
        verifier_targets = kerc_verifier_labels.astype(mx.float32)
        verifier_element_loss = (
            mx.maximum(verifier_logits, 0.0)
            - verifier_logits * verifier_targets
            + mx.log1p(mx.exp(-mx.abs(verifier_logits)))
        )
        if (kerc_verifier_positive_weights is None) != (
            kerc_verifier_negative_weights is None
        ):
            raise ValueError(
                "KERC verifier positive and negative weights must be supplied together"
            )
        if kerc_verifier_positive_weights is not None:
            class_weights = (
                verifier_targets * kerc_verifier_positive_weights
                + (1.0 - verifier_targets) * kerc_verifier_negative_weights
            )
            verifier_element_loss = verifier_element_loss * class_weights
        verifier_loss = mx.mean(verifier_element_loss)
        body_loss = body_loss + float(kerc_verifier_weight) * verifier_loss
    if kerc_decision_labels is not None and kerc_decision_weight > 0.0:
        decision_logits = kerc_aux.get("decision_logits") if kerc_aux else None
        if decision_logits is None:
            raise ValueError("KERC decision labels require source-conditioned logits")
        decision_targets = kerc_decision_labels.astype(mx.int32)
        decision_element_loss = nn.losses.cross_entropy(
            decision_logits, decision_targets
        )
        if kerc_decision_class_weights is not None:
            if tuple(kerc_decision_class_weights.shape) != (
                int(decision_logits.shape[-1]),
            ):
                raise ValueError("KERC decision class weights do not match logits")
            decision_element_loss = decision_element_loss * mx.take(
                kerc_decision_class_weights, decision_targets
            )
        if kerc_decision_loss_mask is None:
            decision_authority = mx.ones(
                (int(decision_logits.shape[0]),), dtype=mx.float32
            )
        else:
            if tuple(kerc_decision_loss_mask.shape) != (
                int(decision_logits.shape[0]),
            ):
                raise ValueError("KERC decision loss mask must be one value per row")
            decision_authority = kerc_decision_loss_mask.astype(mx.float32)
        decision_loss = mx.sum(
            decision_element_loss * decision_authority
        ) / mx.maximum(mx.sum(decision_authority), 1.0)
        body_loss = body_loss + float(kerc_decision_weight) * decision_loss
    if plan_logits is None:
        return body_loss
    plan_targets = plan_labels.astype(mx.float32)
    if plan_loss_mode == "factorized_step_categorical":
        slots = int(plan_slot_count)
        groups = tuple(int(value) for value in plan_factor_group_sizes)
        if (
            slots <= 0
            or int(plan_logits.shape[-1]) % slots
            or len(groups) < 2
            or groups[0] != 1
            or sum(groups) != int(plan_logits.shape[-1]) // slots
        ):
            raise ValueError("factorized plan loss requires a complete grouped slot field")
        width = int(plan_logits.shape[-1]) // slots
        slot_logits = plan_logits.reshape(-1, width)
        slot_targets = plan_targets.reshape(-1, width)
        presence_logits = slot_logits[:, 0]
        presence_targets = slot_targets[:, 0]
        presence_loss = mx.mean(
            mx.maximum(presence_logits, 0.0)
            - presence_logits * presence_targets
            + mx.log1p(mx.exp(-mx.abs(presence_logits)))
        )
        active_mass = mx.maximum(mx.sum(presence_targets), 1.0)
        factor_losses = []
        offset = 1
        for group_width in groups[1:]:
            group_logits = slot_logits[:, offset : offset + group_width]
            group_targets = slot_targets[:, offset : offset + group_width]
            target_ids = mx.argmax(group_targets, axis=-1).astype(mx.int32)
            per_slot = nn.losses.cross_entropy(group_logits, target_ids)
            factor_losses.append(mx.sum(per_slot * presence_targets) / active_mass)
            offset += group_width
        plan_loss = presence_loss + mx.mean(mx.stack(factor_losses))
        return body_loss + float(plan_weight) * plan_loss
    if plan_loss_mode == "slot_categorical":
        slots = int(plan_slot_count)
        if slots <= 0 or int(plan_logits.shape[-1]) % slots:
            raise ValueError("slot-categorical plan loss requires an evenly divided slot field")
        width = int(plan_logits.shape[-1]) // slots
        slot_logits = plan_logits.reshape(-1, width)
        slot_targets = plan_targets.reshape(-1, width)
        active = mx.sum(slot_targets, axis=-1) > 0.5
        target_ids = mx.where(
            active,
            mx.argmax(slot_targets, axis=-1).astype(mx.int32) + 1,
            mx.zeros(active.shape, dtype=mx.int32),
        )
        empty_logits = mx.zeros((int(slot_logits.shape[0]), 1), dtype=slot_logits.dtype)
        categorical_logits = mx.concatenate([empty_logits, slot_logits], axis=-1)
        plan_loss = mx.mean(nn.losses.cross_entropy(categorical_logits, target_ids))
        return body_loss + float(plan_weight) * plan_loss
    if plan_loss_mode != "binary_multilabel":
        raise ValueError(f"unsupported semantic plan loss mode: {plan_loss_mode}")
    softplus_positive = mx.maximum(-plan_logits, 0.0) + mx.log1p(
        mx.exp(-mx.abs(plan_logits))
    )
    softplus_negative = mx.maximum(plan_logits, 0.0) + mx.log1p(
        mx.exp(-mx.abs(plan_logits))
    )
    positive_weights = (
        plan_positive_weights
        if plan_positive_weights is not None
        else mx.ones((plan_logits.shape[-1],), dtype=mx.float32)
    )
    plan_loss = mx.mean(
        (1.0 - plan_targets) * softplus_negative
        + plan_targets * positive_weights[None, :] * softplus_positive
    )
    return body_loss + float(plan_weight) * plan_loss


def checkpointed_causal_loss(model: Any, *args: Any, **kwargs: Any) -> Any:
    """Recompute the complete scalar objective during backward.

    Layer-only checkpointing cannot release forward intermediates created by the
    vocabulary, pointer-generator, or auxiliary objectives. Passing the complete
    trainable parameter tree to ``mx.checkpoint`` preserves parameter-gradient
    ownership while allowing those intermediates to be recomputed exactly.
    """

    if len(args) < 5:
        raise ValueError("checkpointed causal loss requires inputs, labels, mask, mx, and nn")
    mx = args[3]

    def recompute(parameters: Any) -> Any:
        model.update(parameters)
        return causal_loss(model, *args, **kwargs)

    return mx.checkpoint(recompute)(model.trainable_parameters())


def decomposed_checkpointed_causal_loss_and_grad(
    model: Any, *args: Any, **kwargs: Any
) -> tuple[Any, Any]:
    """Accumulate an exact KERC objective as serial checkpointed backwards.

    The ordinary complete objective is mathematically additive, but a single
    lazy MLX graph retains the token/copy path and every KERC auxiliary path at
    once. This route evaluates each active additive component independently,
    accumulates its parameter gradients in FP32, and releases the superseded
    graph before the next component. It is intentionally limited to KERC
    objectives without a semantic-plan head.
    """

    token_loss_position_chunk_size = int(
        kwargs.pop("token_loss_position_chunk_size", 0)
    )
    if token_loss_position_chunk_size < 0:
        raise ValueError("token-loss position chunk size cannot be negative")
    bound = inspect.signature(causal_loss).bind(model, *args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    mx = values["mx"]
    nn = values["nn"]
    if values["plan_labels"] is not None and float(values["plan_weight"]) > 0.0:
        raise ValueError(
            "decomposed KERC objective does not accept semantic-plan supervision"
        )
    call_base = {
        key: value
        for key, value in values.items()
        if key not in {"model", "inputs", "labels", "mask", "mx", "nn"}
    }
    auxiliary_groups = {
        "legacy_residual": (
            "kerc_residual_labels",
            "kerc_residual_weight",
            (
                "kerc_residual_class_weights",
                "kerc_residual_loss_mask",
            ),
        ),
        "unit_residual": (
            "kerc_unit_residual_labels",
            "kerc_unit_residual_weight",
            (
                "kerc_unit_residual_loss_mask",
                "kerc_unit_confidence_targets",
                "kerc_unit_byte_ids",
                "kerc_unit_byte_mask",
                "kerc_unit_byte_offsets",
                "kerc_unit_kind_ids",
                "kerc_unit_candidate_features",
                "kerc_unit_mask",
                "kerc_unit_hard_block_mask",
                "kerc_unit_class_weights",
            ),
        ),
        "verifier": (
            "kerc_verifier_labels",
            "kerc_verifier_weight",
            (
                "kerc_verifier_positive_weights",
                "kerc_verifier_negative_weights",
            ),
        ),
        "decision": (
            "kerc_decision_labels",
            "kerc_decision_weight",
            (
                "kerc_decision_class_weights",
                "kerc_decision_loss_mask",
            ),
        ),
    }
    auxiliary_fields = {
        field
        for label_field, weight_field, extra_fields in auxiliary_groups.values()
        for field in (label_field, weight_field, *extra_fields)
    }
    unit_conditioning_fields = (
        "kerc_unit_byte_ids",
        "kerc_unit_byte_mask",
        "kerc_unit_byte_offsets",
        "kerc_unit_kind_ids",
        "kerc_unit_candidate_features",
        "kerc_unit_mask",
        "kerc_unit_hard_block_mask",
    )

    def inactive_kwargs() -> dict[str, Any]:
        component = dict(call_base)
        for _name, (label_field, weight_field, extra_fields) in auxiliary_groups.items():
            component[label_field] = None
            component[weight_field] = 0.0
            for field in extra_fields:
                component[field] = None
        return component

    components: list[tuple[str, Any, dict[str, Any]]] = []
    token_mask = values["mask"]
    if bool(mx.any(token_mask > 0.0).item()):
        active_columns = mx.any(token_mask > 0.0, axis=0)
        active_indices = np.flatnonzero(np.asarray(active_columns))
        chunk_size = token_loss_position_chunk_size or len(active_indices)
        for chunk_index, start in enumerate(range(0, len(active_indices), chunk_size)):
            selected = active_indices[start : start + chunk_size]
            first = int(selected[0])
            last = int(selected[-1])
            if not np.array_equal(selected, np.arange(first, last + 1)):
                raise ValueError(
                    "token-loss position chunking requires contiguous supervision"
                )
            position_mask = (
                (mx.arange(int(token_mask.shape[1])) >= first)
                & (mx.arange(int(token_mask.shape[1])) <= last)
            )[None, :]
            component_mask = mx.where(
                position_mask,
                token_mask,
                mx.zeros(token_mask.shape, dtype=mx.float32),
            )
            token_component = inactive_kwargs()
            token_component["normalization_mask"] = token_mask
            for field in unit_conditioning_fields:
                token_component[field] = values[field]
            components.append(
                (f"token_and_copy_{chunk_index}", component_mask, token_component)
            )
    zero_mask = mx.zeros(token_mask.shape, dtype=mx.float32)
    for name, (label_field, weight_field, extra_fields) in auxiliary_groups.items():
        if (
            values[label_field] is None
            or float(values[weight_field]) <= 0.0
        ):
            continue
        component = inactive_kwargs()
        component[label_field] = values[label_field]
        component[weight_field] = values[weight_field]
        for field in extra_fields:
            component[field] = values[field]
        components.append((name, zero_mask, component))
    if not components:
        raise ValueError("decomposed KERC objective has no active component")
    if any(field not in call_base for field in auxiliary_fields):
        raise AssertionError("decomposed KERC objective field inventory drifted")

    value_and_grad = nn.value_and_grad(model, checkpointed_causal_loss)
    tree_map = __import__("mlx.utils", fromlist=["tree_map"]).tree_map
    accumulated = None
    total_loss = 0.0
    for _name, component_mask, component_kwargs in components:
        loss, gradients = value_and_grad(
            model,
            values["inputs"],
            values["labels"],
            component_mask,
            mx,
            nn,
            **component_kwargs,
        )
        mx.eval(loss, gradients)
        fp32_gradients = tree_map(
            lambda value: value.astype(mx.float32), gradients
        )
        accumulated = (
            fp32_gradients
            if accumulated is None
            else tree_map(
                lambda prior, value: prior + value,
                accumulated,
                fp32_gradients,
            )
        )
        mx.eval(accumulated)
        total_loss += float(loss.item())
        del gradients
        del fp32_gradients
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    return mx.array(total_loss, dtype=mx.float32), accumulated


def mtp_auxiliary_loss(
    logits_by_offset: list[Any],
    labels: Any,
    mask: Any,
    future_offsets: tuple[int, ...],
    loss_weights: tuple[float, ...],
    mx: Any,
    nn: Any,
) -> Any:
    """Score future-token heads against the same masked causal training stream."""

    if not logits_by_offset or len(logits_by_offset) != len(future_offsets):
        raise ValueError("MTP loss requires one logits tensor per future offset")
    if len(loss_weights) != len(future_offsets):
        raise ValueError("MTP loss weights must align with future offsets")
    total = mx.array(0.0, dtype=mx.float32)
    contributed = False
    sequence_length = int(labels.shape[1])
    for logits, offset, weight in zip(logits_by_offset, future_offsets, loss_weights):
        shift_from_next_token_labels = int(offset) - 1
        if shift_from_next_token_labels < 1 or shift_from_next_token_labels >= sequence_length:
            continue
        head_logits = logits[:, :-shift_from_next_token_labels, :]
        head_labels = labels[:, shift_from_next_token_labels:]
        head_mask = mask[:, shift_from_next_token_labels:]
        token_loss = nn.losses.cross_entropy(head_logits, head_labels)
        denominator = mx.maximum(
            mx.sum(head_mask), mx.array(1.0, dtype=mx.float32)
        )
        total = total + float(weight) * mx.sum(token_loss * head_mask) / denominator
        contributed = contributed or float(weight) > 0.0
    if not contributed:
        raise ValueError("MTP loss has no valid weighted future positions")
    return total


def pointer_generator_auxiliary_loss(
    copy_aux: dict[str, Any],
    labels: Any,
    mask: Any,
    mx: Any,
    *,
    normalization_labels: Any | None = None,
    normalization_mask: Any | None = None,
) -> Any:
    """Train pointer alignment and generation/copy gating on visible train labels."""

    pointer_scores = copy_aux["pointer_scores"]
    source_ids = copy_aux["source_copy_ids"]
    source_valid = copy_aux["source_copy_valid"]
    generator_gate = copy_aux["generator_gate"]
    matches = (
        source_ids[:, None, :] == labels[:, :, None]
    ) & source_valid[:, None, :]
    copyable = mx.any(matches, axis=-1)
    valid_scores = mx.where(
        source_valid[:, None, :],
        pointer_scores,
        mx.array(-1e9, dtype=mx.float32),
    )
    matching_scores = mx.where(
        matches,
        pointer_scores,
        mx.array(-1e9, dtype=mx.float32),
    )
    alignment_loss = -(
        mx.logsumexp(matching_scores, axis=-1)
        - mx.logsumexp(valid_scores, axis=-1)
    )
    supervised = mask.astype(mx.float32)
    copy_mask = supervised * copyable.astype(mx.float32)
    alignment_denominator = mx.sum(copy_mask)
    gate_denominator = mx.sum(supervised)
    if normalization_mask is not None:
        if normalization_labels is None:
            raise ValueError(
                "pointer normalization mask requires full normalization labels"
            )
        normalization_matches = (
            source_ids[:, None, :] == normalization_labels[:, :, None]
        ) & source_valid[:, None, :]
        normalization_copyable = mx.any(normalization_matches, axis=-1)
        normalization_supervised = normalization_mask.astype(mx.float32)
        alignment_denominator = mx.sum(
            normalization_supervised
            * normalization_copyable.astype(mx.float32)
        )
        gate_denominator = mx.sum(normalization_supervised)
    alignment = mx.sum(alignment_loss * copy_mask) / mx.maximum(
        alignment_denominator, 1.0
    )
    gate = mx.minimum(mx.maximum(generator_gate, 1e-6), 1.0 - 1e-6)
    generate_target = (~copyable).astype(mx.float32)
    gate_loss = -(
        generate_target * mx.log(gate)
        + (1.0 - generate_target) * mx.log(1.0 - gate)
    )
    gate_loss = mx.sum(gate_loss * supervised) / mx.maximum(
        gate_denominator, 1.0
    )
    return alignment + gate_loss
