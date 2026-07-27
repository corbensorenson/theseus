#!/usr/bin/env python3
"""Fail-closed authority contract for one heterogeneous sampler/update batch."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


POLICY = "project_theseus_heterogeneous_microbatch_contract_v1"
GLOBAL_NORMALIZATION = "global_objective_mass_denominator_v1"


class ContractFault(ValueError):
    """Raised when a shard cannot cross the authoritative update gate."""


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def gradient_schema(gradients: Mapping[str, np.ndarray]) -> dict[str, list[Any]]:
    return {
        name: [list(np.asarray(value).shape), str(np.asarray(value).dtype)]
        for name, value in sorted(gradients.items())
    }


@dataclass(frozen=True)
class GradientContribution:
    shard_id: str
    engine: str
    generation: int
    row_ids: tuple[str, ...]
    row_objective_masses: tuple[float, ...]
    global_objective_mass: float
    gradients: Mapping[str, np.ndarray]
    normalization: str = GLOBAL_NORMALIZATION
    local_optimizer_steps: int = 0


def _validate_contribution(
    contribution: GradientContribution,
    *,
    generation: int,
    expected_schema: Mapping[str, list[Any]],
    expected_global_mass: float,
) -> None:
    if not contribution.shard_id or not contribution.engine:
        raise ContractFault("shard_identity_missing")
    if contribution.generation != generation:
        raise ContractFault("stale_or_mixed_parameter_generation")
    if contribution.local_optimizer_steps:
        raise ContractFault("per_device_optimizer_update_forbidden")
    if contribution.normalization != GLOBAL_NORMALIZATION:
        raise ContractFault("local_or_unknown_gradient_normalization")
    if not math.isclose(
        contribution.global_objective_mass,
        expected_global_mass,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ContractFault("global_objective_mass_mismatch")
    if (
        not contribution.row_ids
        or len(contribution.row_ids) != len(contribution.row_objective_masses)
        or len(set(contribution.row_ids)) != len(contribution.row_ids)
    ):
        raise ContractFault("shard_row_or_mass_shape_invalid")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in contribution.row_objective_masses
    ):
        raise ContractFault("nonpositive_or_nonfinite_row_objective_mass")
    if gradient_schema(contribution.gradients) != dict(expected_schema):
        raise ContractFault("gradient_schema_mismatch")
    for value in contribution.gradients.values():
        array = np.asarray(value)
        if array.dtype != np.float32:
            raise ContractFault("gradient_contribution_not_fp32")
        if not bool(np.all(np.isfinite(array))):
            raise ContractFault("gradient_contribution_nonfinite")


def join_gradient_contributions(
    contributions: Sequence[GradientContribution],
    *,
    generation: int,
    expected_rows: Sequence[str],
    expected_row_objective_masses: Mapping[str, float],
    expected_schema: Mapping[str, list[Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate and sum globally normalized FP32 shard gradients exactly once."""

    if len(contributions) < 2:
        raise ContractFault("heterogeneous_join_requires_at_least_two_shards")
    if len(set(expected_rows)) != len(expected_rows) or not expected_rows:
        raise ContractFault("expected_sampler_rows_invalid")
    if set(expected_rows) != set(expected_row_objective_masses):
        raise ContractFault("expected_row_mass_authority_mismatch")
    expected_global_mass = math.fsum(
        float(expected_row_objective_masses[row_id])
        for row_id in expected_rows
    )
    if not math.isfinite(expected_global_mass) or expected_global_mass <= 0:
        raise ContractFault("expected_global_objective_mass_invalid")

    observed_rows: list[str] = []
    observed_mass_by_row: dict[str, float] = {}
    shard_ids: set[str] = set()
    engines: set[str] = set()
    accumulated = {
        name: np.zeros(tuple(shape_dtype[0]), dtype=np.float32)
        for name, shape_dtype in expected_schema.items()
    }
    for contribution in contributions:
        _validate_contribution(
            contribution,
            generation=generation,
            expected_schema=expected_schema,
            expected_global_mass=expected_global_mass,
        )
        if contribution.shard_id in shard_ids:
            raise ContractFault("duplicate_shard_identity")
        shard_ids.add(contribution.shard_id)
        engines.add(contribution.engine)
        for row_id, mass in zip(
            contribution.row_ids, contribution.row_objective_masses
        ):
            if row_id in observed_mass_by_row:
                raise ContractFault("sampler_row_overlap")
            observed_rows.append(row_id)
            observed_mass_by_row[row_id] = float(mass)
        for name, value in contribution.gradients.items():
            np.add(
                accumulated[name],
                np.asarray(value),
                out=accumulated[name],
                casting="no",
            )
    if observed_rows != list(expected_rows):
        if set(observed_rows) != set(expected_rows):
            raise ContractFault("sampler_row_coverage_mismatch")
        raise ContractFault("sampler_row_order_mismatch")
    for row_id in expected_rows:
        if not math.isclose(
            observed_mass_by_row[row_id],
            float(expected_row_objective_masses[row_id]),
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ContractFault("row_objective_mass_mismatch")
    if len(engines) < 2:
        raise ContractFault("heterogeneous_join_requires_distinct_engines")
    if any(not bool(np.all(np.isfinite(value))) for value in accumulated.values()):
        raise ContractFault("accumulated_gradient_nonfinite")
    receipt = {
        "policy": POLICY,
        "generation": generation,
        "shard_ids": sorted(shard_ids),
        "engines": sorted(engines),
        "sampler_rows": list(expected_rows),
        "sampler_row_count": len(expected_rows),
        "global_objective_mass": expected_global_mass,
        "normalization": GLOBAL_NORMALIZATION,
        "gradient_schema_digest": digest(expected_schema),
        "one_fp32_accumulator": True,
        "local_optimizer_steps": 0,
        "ready_for_single_global_clip_and_update": True,
    }
    return accumulated, receipt


def clip_and_adamw_once(
    parameters: Mapping[str, np.ndarray],
    first_moments: Mapping[str, np.ndarray],
    second_moments: Mapping[str, np.ndarray],
    gradients: Mapping[str, np.ndarray],
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    clip_norm: float,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Apply one global FP32 clip and one AdamW publication."""

    names = sorted(parameters)
    if (
        names != sorted(first_moments)
        or names != sorted(second_moments)
        or names != sorted(gradients)
    ):
        raise ContractFault("optimizer_tree_schema_mismatch")
    gradient_square_sum = math.fsum(
        float(
            np.sum(
                np.square(np.asarray(gradients[name], dtype=np.float64)),
                dtype=np.float64,
            )
        )
        for name in names
    )
    gradient_norm = math.sqrt(gradient_square_sum)
    if not math.isfinite(gradient_norm):
        raise ContractFault("global_gradient_norm_nonfinite")
    scale = min(1.0, float(clip_norm) / max(gradient_norm, 1e-6))
    next_parameters: dict[str, np.ndarray] = {}
    next_first: dict[str, np.ndarray] = {}
    next_second: dict[str, np.ndarray] = {}
    for name in names:
        parameter = np.asarray(parameters[name], dtype=np.float32)
        first = np.asarray(first_moments[name], dtype=np.float32)
        second = np.asarray(second_moments[name], dtype=np.float32)
        gradient = np.asarray(gradients[name], dtype=np.float32) * np.float32(
            scale
        )
        if not (
            parameter.shape == first.shape == second.shape == gradient.shape
        ):
            raise ContractFault("optimizer_leaf_shape_mismatch")
        next_m = np.float32(beta1) * first + np.float32(1.0 - beta1) * gradient
        next_v = (
            np.float32(beta2) * second
            + np.float32(1.0 - beta2) * gradient * gradient
        )
        decayed = parameter * np.float32(
            1.0 - learning_rate * weight_decay
        )
        next_parameter = decayed - np.float32(learning_rate) * next_m / (
            np.sqrt(next_v).astype(np.float32) + np.float32(epsilon)
        )
        next_parameters[name] = next_parameter.astype(np.float32, copy=False)
        next_first[name] = next_m.astype(np.float32, copy=False)
        next_second[name] = next_v.astype(np.float32, copy=False)
    receipt = {
        "policy": POLICY,
        "gradient_norm": gradient_norm,
        "clip_scale": scale,
        "global_clip_count": 1,
        "adamw_update_count": 1,
        "published_generation_increment": 1,
        "all_state_finite": all(
            bool(np.all(np.isfinite(value)))
            for tree in (next_parameters, next_first, next_second)
            for value in tree.values()
        ),
    }
    if not receipt["all_state_finite"]:
        raise ContractFault("published_optimizer_state_nonfinite")
    return next_parameters, next_first, next_second, receipt
