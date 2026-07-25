#!/usr/bin/env python3
"""Probe K5 overfit/termination on exact rows admitted to its bounded training set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import host_resource_safety
import kerc_k5_candidate_evaluator as candidate_evaluator
import kernel_english_protocol as kernel_protocol
import moecot_language_arm_training as training
import standard_causal_transformer_survival as survival


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_kerc_k5_stage_learnability_probe_v1"
PROBE_RESOURCE_RECEIPT = (
    "reports/rdc_kerc_k5_stage_learnability_stratified_target_position_complete_"
    "terminal4096_seed_20260722_teacher_forced.host_resource_safety.json"
)
PROBE_RESOURCE_RECEIPT_SHA256 = (
    "5ad0ac0bcb2c0dfddca791ccb4b5aaddca625060300361dc254994ebc5804f33"
)
GRADIENT_INTERFERENCE_RESOURCE_RECEIPT = (
    "reports/rdc_kerc_k5_stage_learnability_stratified_target_position_complete_"
    "terminal4096_seed_20260722_teacher_forced_gradient_interference."
    "host_resource_safety.json"
)
GRADIENT_INTERFERENCE_RESOURCE_RECEIPT_SHA256 = (
    "df34f200fb57c0e59ac75aac675bc44c6c01ea3fcb68453006daba13dbadb999"
)


def objective_balanced_exposure_batches(
    *,
    row_count: int,
    optimizer_steps: int,
    single_objective_warmup_steps: int,
    fixed_objective_index: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Reconstruct the exact objective batches admitted by the training policy."""

    if optimizer_steps < 0:
        raise ValueError("optimizer steps cannot be negative")
    schedule = training.kerc_overfit_batch_schedule(
        row_count=row_count,
        single_objective_warmup_steps=single_objective_warmup_steps,
        fixed_objective_index=fixed_objective_index,
    )
    if not schedule:
        return (tuple(range(row_count)),) * optimizer_steps
    return tuple(
        schedule[min(step, len(schedule) - 1)] for step in range(optimizer_steps)
    )


def resource_stress_prefix_replay(
    *,
    inputs: Any,
    mask: Any,
    progress_mask: Any,
    coverage_indices: tuple[int, ...] | list[int],
    enabled: bool,
    capacity: int,
) -> tuple[list[int], dict[str, Any]]:
    """Independently reconstruct the maximum-target and maximum-width prefix."""

    prefix = [int(index) for index in coverage_indices]
    inactive = {
        "policy": "project_theseus_target_and_width_resource_stress_prefix_v2",
        "active": False,
        "selected_index_sha256": "",
        "selected_indices_sha256": "",
        "selected_row_count": 0,
        "rows": [],
        "target_positions": 0,
        "active_width": 0,
        "already_in_coverage_prefix": False,
    }
    if not enabled:
        return prefix, inactive
    row_count = len(inputs)
    if len(mask) != row_count or len(progress_mask) != row_count:
        raise ValueError("resource-stress replay arrays are misaligned")
    if capacity < len(prefix):
        raise ValueError("resource-stress replay capacity is below coverage prefix")
    target_counts: list[int] = []
    active_widths: list[int] = []
    for index in range(row_count):
        input_row = np.asarray(inputs[index])
        mask_row = np.asarray(mask[index])
        progress_row = np.asarray(progress_mask[index])
        if not input_row.shape == mask_row.shape == progress_row.shape:
            raise ValueError("resource-stress replay row shapes are misaligned")
        active = np.flatnonzero(
            (input_row != 0) | (mask_row != 0) | (progress_row != 0)
        )
        target_counts.append(int(progress_row.sum()))
        active_widths.append(int(active[-1] + 1) if len(active) else 1)
    maximum_target_index = max(
        range(row_count),
        key=lambda index: (
            target_counts[index],
            active_widths[index],
            -index,
        ),
    )
    maximum_width_index = max(
        range(row_count),
        key=lambda index: (
            active_widths[index],
            target_counts[index],
            -index,
        ),
    )
    if target_counts[maximum_target_index] <= 0:
        raise ValueError("resource-stress replay requires a supervised row")
    stress_roles: dict[int, list[str]] = {}
    for index, role in (
        (maximum_target_index, "maximum_target_positions"),
        (maximum_width_index, "maximum_active_width"),
    ):
        stress_roles.setdefault(index, []).append(role)
    stress_rows = []
    for stress_index, roles in stress_roles.items():
        already_selected = stress_index in prefix
        if not already_selected and len(prefix) + 1 > capacity:
            raise ValueError(
                "resource-stress replay capacity cannot include coverage and stress rows"
            )
        if not already_selected:
            prefix.append(stress_index)
        stress_rows.append(
            {
                "roles": roles,
                "selected_index_sha256": hashlib.sha256(
                    np.asarray([stress_index], dtype=np.int64).tobytes()
                ).hexdigest(),
                "target_positions": target_counts[stress_index],
                "active_width": active_widths[stress_index],
                "already_in_coverage_prefix": already_selected,
            }
        )
    stress_indices = list(stress_roles)
    primary = stress_rows[stress_indices.index(maximum_target_index)]
    return prefix, {
        "policy": "project_theseus_target_and_width_resource_stress_prefix_v2",
        "active": True,
        "selected_index_sha256": primary["selected_index_sha256"],
        "selected_indices_sha256": hashlib.sha256(
            np.asarray(stress_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "selected_row_count": len(stress_rows),
        "rows": stress_rows,
        "target_positions": primary["target_positions"],
        "active_width": primary["active_width"],
        "already_in_coverage_prefix": primary["already_in_coverage_prefix"],
    }


def align_teacher_forced_predictions(
    predicted_ids: np.ndarray,
    expected_next: np.ndarray,
    *,
    supervised_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align predictions from either full-sequence or compact target-only execution."""

    predicted = np.asarray(predicted_ids, dtype=np.int64).reshape(-1)
    expected = np.asarray(expected_next, dtype=np.int64).reshape(-1)
    if supervised_start < 0 or supervised_start > len(expected):
        raise ValueError("K5 supervised start falls outside the expected sequence")
    supervised_expected = expected[supervised_start:]
    if len(predicted) == len(expected):
        supervised_predictions = predicted[supervised_start:]
    elif len(predicted) == len(supervised_expected):
        # Compact encoder-decoder execution projects only the target partition,
        # beginning at the target BOS input position.
        supervised_predictions = predicted
    else:
        raise ValueError(
            "K5 teacher-forced logits do not align with full or compact execution"
        )
    return supervised_predictions, supervised_expected


def gradient_parameter_group(path: str) -> str:
    if (
        "source_attention" in path
        or path.startswith("source_layers.")
        or path.startswith("source_final_norm.")
    ):
        return "source_conditioned_bridge"
    if "kerc_stage_adapters" in path or path.startswith("kerc_stage_embedding."):
        return "stage_conditioning"
    if path.startswith("kerc_kernel_output."):
        return "kernel_output"
    if path.startswith("kerc_surface_output."):
        return "surface_output"
    if "kerc_" in path:
        return "other_kerc"
    return "other_trainable"


def gradient_pair_metrics(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("gradient snapshots do not share an exact parameter inventory")
    totals: dict[str, dict[str, float | int]] = {}
    for name in sorted(left):
        left_value = np.asarray(left[name], dtype=np.float64)
        right_value = np.asarray(right[name], dtype=np.float64)
        if left_value.shape != right_value.shape:
            raise ValueError(f"gradient shape mismatch: {name}")
        group = gradient_parameter_group(name)
        for key in ("all_trainable", group):
            row = totals.setdefault(
                key,
                {"dot": 0.0, "left_square": 0.0, "right_square": 0.0, "elements": 0},
            )
            row["dot"] = float(row["dot"]) + float(
                np.sum(left_value * right_value, dtype=np.float64)
            )
            row["left_square"] = float(row["left_square"]) + float(
                np.sum(left_value * left_value, dtype=np.float64)
            )
            row["right_square"] = float(row["right_square"]) + float(
                np.sum(right_value * right_value, dtype=np.float64)
            )
            row["elements"] = int(row["elements"]) + int(left_value.size)
    result: dict[str, Any] = {}
    for group, row in sorted(totals.items()):
        left_norm = float(np.sqrt(float(row["left_square"])))
        right_norm = float(np.sqrt(float(row["right_square"])))
        denominator = left_norm * right_norm
        result[group] = {
            "cosine": round(float(row["dot"]) / denominator, 8) if denominator else None,
            "dot": round(float(row["dot"]), 8),
            "left_norm": round(left_norm, 8),
            "right_norm": round(right_norm, 8),
            "element_count": int(row["elements"]),
        }
    return result


def aggregate_teacher_forced_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate preselected row diagnostics without model-outcome row selection."""

    totals: dict[str, dict[str, Any]] = {}
    for row in rows:
        objective = str(row["objective"])
        target = totals.setdefault(
            objective,
            {
                "row_count": 0,
                "sampled_row_count": 0,
                "optimizer_step_count": 0,
                "correct": 0,
                "total": 0,
                "eos_top1_count": 0,
                "frequency": {
                    "singleton": {"correct": 0, "total": 0},
                    "repeated": {"correct": 0, "total": 0},
                },
            },
        )
        target["row_count"] += 1
        target["sampled_row_count"] += int(bool(row["sampled_training_row"]))
        target["optimizer_step_count"] += int(row["sampled_optimizer_step_count"])
        target["correct"] += int(row["teacher_forced_top1_correct"])
        target["total"] += int(row["teacher_forced_top1_total"])
        target["eos_top1_count"] += int(bool(row["teacher_forced_eos_top1"]))
        for frequency in ("singleton", "repeated"):
            observed = row["teacher_forced_accuracy_by_target_frequency"][frequency]
            target["frequency"][frequency]["correct"] += int(observed["correct"])
            target["frequency"][frequency]["total"] += int(observed["total"])
    result: dict[str, Any] = {}
    for objective, target in totals.items():
        target["top1_accuracy"] = round(
            target["correct"] / max(1, target["total"]), 8
        )
        target["eos_top1_rate"] = round(
            target["eos_top1_count"] / max(1, target["row_count"]), 8
        )
        for frequency in ("singleton", "repeated"):
            observed = target["frequency"][frequency]
            observed["accuracy"] = round(
                observed["correct"] / max(1, observed["total"]), 8
            )
        result[objective] = target
    return result


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def objective_exposure_projection(
    coverage_labels: tuple[tuple[str, ...], ...],
    training_row_ids: tuple[str, ...],
    exposure_counts: dict[str, int],
) -> dict[str, dict[str, Any]]:
    if len(coverage_labels) != len(training_row_ids):
        raise ValueError("K5 exposure projection rows are misaligned")
    grouped: dict[str, list[int]] = {}
    for labels, row_id in zip(coverage_labels, training_row_ids, strict=True):
        objectives = [
            label.split(":", 1)[1]
            for label in labels
            if label.startswith("objective:")
        ]
        if len(objectives) != 1:
            raise ValueError("K5 exposure projection row has no unique objective")
        grouped.setdefault(objectives[0], []).append(
            int(exposure_counts.get(str(row_id), 0))
        )
    return {
        objective: {
            "population_row_count": len(counts),
            "sampled_unique_row_count": sum(count > 0 for count in counts),
            "unsampled_row_count": sum(count == 0 for count in counts),
            "total_optimizer_updates": sum(counts),
            "minimum_row_exposures": min(counts),
            "maximum_row_exposures": max(counts),
            "mean_row_exposures": round(sum(counts) / len(counts), 8),
        }
        for objective, counts in sorted(grouped.items())
    }


def subset_supervision_stage_rows(
    stage: Any, indices: Any
) -> SimpleNamespace:
    """Reproduce the trainer's row projection for an independent probe."""

    selected = tuple(int(value) for value in np.asarray(indices).tolist())
    row_count = len(stage.inputs)
    if not selected or any(index < 0 or index >= row_count for index in selected):
        raise ValueError("K5 probe stage row projection is invalid")
    values: dict[str, Any] = {}
    for name, value in vars(stage).items():
        if isinstance(value, training.RaggedRows):
            values[name] = training.RaggedRows(
                [np.asarray(value[index]) for index in selected],
                dtype=value.dtype,
                standard_width=value.standard_width,
            )
        elif (
            isinstance(value, np.ndarray)
            and value.ndim
            and len(value) == row_count
        ):
            values[name] = np.asarray(value[list(selected)])
        elif isinstance(value, tuple) and len(value) == row_count:
            values[name] = tuple(value[index] for index in selected)
        elif name == "receipt":
            receipt = copy.deepcopy(value)
            receipt["probe_token_supervised_row_projection"] = {
                "policy": "project_theseus_kerc_probe_token_authority_scope_v1",
                "original_row_count": row_count,
                "selected_row_count": len(selected),
                "selected_indices_sha256": hashlib.sha256(
                    np.asarray(selected, dtype=np.int64).tobytes()
                ).hexdigest(),
            }
            values[name] = receipt
        else:
            values[name] = value
    return SimpleNamespace(**values)


def exact_training_row_panel_ids(report: dict[str, Any]) -> tuple[str, ...]:
    """Validate an explicit multi-row training diagnostic selection."""

    rows = report.get("rows")
    if (
        report.get("policy") != POLICY
        or report.get("qualification_state")
        != "TEACHER_FORCED_DIAGNOSTIC_ONLY"
        or not isinstance(rows, list)
        or len(rows) < 2
        or any(
            not isinstance(row, dict)
            or row.get("admitted_training_row") is not True
            or not str(row.get("row_id") or "")
            for row in rows
        )
        or len({str(row["row_id"]) for row in rows}) != len(rows)
        or int(report.get("public_benchmark_prompts_used") or 0) != 0
        or int(report.get("external_inference_calls") or 0) != 0
        or int(report.get("fallback_template_router_tool_credit") or 0) != 0
    ):
        raise ValueError(
            "K5 training-row panel report is not an exact admitted training panel"
        )
    return tuple(str(row["row_id"]) for row in rows)


def replay_training_stage_row_scope(
    stage: Any, execution_policy: dict[str, Any]
) -> tuple[Any, dict[str, int]]:
    """Mirror trainer row-selection order before checkpoint diagnostics."""

    overfit_rows_per_objective = int(
        execution_policy.get("kerc_overfit_rows_per_objective") or 0
    )
    if overfit_rows_per_objective:
        stage = training.select_kerc_overfit_stage(
            stage, rows_per_objective=overfit_rows_per_objective
        )
    observed_scope: dict[str, int] = {}
    if execution_policy.get("kerc_delta_stage_only") is not None:
        original_stage_row_count = len(stage.inputs)
        token_supervised_indices = training.token_supervised_row_indices(
            stage.loss_mask
        )
        if not len(token_supervised_indices):
            raise ValueError("K5 stage probe has no token-supervised rows")
        stage = subset_supervision_stage_rows(stage, token_supervised_indices)
        observed_scope = {
            "stage_only_token_supervised_row_count": len(stage.inputs),
            "stage_only_zero_token_authority_rows_excluded": (
                original_stage_row_count - len(stage.inputs)
            ),
        }
    return stage, observed_scope


def kernel_phase_replay_seed(
    result: dict[str, Any], phase: dict[str, Any], fallback_seed: int
) -> int:
    """Recover the exact seed at kernel-phase entry, including prior updates."""

    cursor = phase.get("data_cursor_start") or {}
    if cursor:
        if cursor.get("policy") != "project_theseus_training_data_cursor_v1":
            raise ValueError("K5 phase replay data-cursor policy mismatch")
        cursor_seed = int(cursor.get("seed") or 0)
        if cursor_seed <= 0:
            raise ValueError("K5 phase replay data-cursor seed is invalid")
        return cursor_seed
    effective_seed = int(
        result.get("effective_training_seed")
        or result.get("candidate_seed")
        or fallback_seed
    )
    cumulative_steps = int(result.get("optimizer_steps") or 0)
    phase_steps = int(phase.get("optimizer_steps") or 0)
    if cumulative_steps < phase_steps:
        raise ValueError("K5 cumulative optimizer steps precede phase steps")
    return effective_seed + cumulative_steps - phase_steps


def segmented_sampler_replay_contract(
    result: dict[str, Any],
    phase: dict[str, Any],
    execution_policy: dict[str, Any],
    *,
    source_result: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Recover cumulative phase authority from an exact fresh-process segment."""

    coverage = phase.get("coverage_first_sampling") or {}
    if (
        execution_policy.get("candidate_scratch_resume_policy")
        != "exact_fresh_process_segment_v1"
        or coverage.get("cumulative_across_fresh_process_segments") is not True
    ):
        return None
    source_steps = int(
        execution_policy.get("continuation_source_optimizer_steps") or 0
    )
    final_steps = int(result.get("optimizer_steps") or 0)
    cumulative_steps = final_steps - source_steps
    recorded_cumulative_positions = int(
        result.get("current_kernel_phase_optimizer_positions") or 0
    )
    source_current_positions = 0
    source_coverage_counts: dict[str, int] = {}
    source_plan_sha256 = ""
    if (
        source_result is not None
        and str(source_result.get("plan_sha256") or "")
        != str(result.get("plan_sha256") or "")
        and source_result.get(
            "current_kernel_phase_position_accounting_reset"
        )
        is True
    ):
        candidate_source_current_positions = int(
            source_result.get("current_kernel_phase_optimizer_positions")
            or 0
        )
        # Older continuation reports imported the source phase counter before
        # the destination reset. New reports reset it at import. Rebase only
        # when the recorded destination counter can actually contain the
        # source counter; otherwise subtracting it creates a false negative.
        if recorded_cumulative_positions > candidate_source_current_positions:
            source_plan_sha256 = str(source_result.get("plan_sha256") or "")
            source_current_positions = candidate_source_current_positions
            source_phase = (
                (source_result.get("phases") or {}).get("kernel_english") or {}
            )
            source_coverage_counts = {
                str(key): int(value)
                for key, value in (
                    (
                        source_phase.get("coverage_first_sampling") or {}
                    ).get("observed_label_counts")
                    or {}
                ).items()
            }
    cumulative_positions = (
        recorded_cumulative_positions - source_current_positions
    )
    segment_steps = int(phase.get("optimizer_steps") or 0)
    planning_capacity = int(coverage.get("planning_capacity") or 0)
    if (
        result.get("current_kernel_phase_position_accounting_reset") is not True
        or source_steps <= 0
        or cumulative_steps <= segment_steps
        or cumulative_positions <= int(phase.get("target_positions_consumed") or 0)
        or segment_steps <= 0
        or planning_capacity <= 0
    ):
        raise ValueError(
            "exact fresh-process segment lacks cumulative sampler authority"
        )
    contract: dict[str, Any] = {
        "optimizer_steps": cumulative_steps,
        "optimizer_positions": cumulative_positions,
        "planning_capacity": planning_capacity,
        "segment_steps": segment_steps,
    }
    if source_current_positions or source_coverage_counts:
        contract.update(
            {
                "source_plan_sha256": source_plan_sha256,
                "source_current_kernel_phase_optimizer_positions": (
                    source_current_positions
                ),
                "source_coverage_observed_label_counts": (
                    source_coverage_counts
                ),
            }
        )
    return contract


def dense_training_epoch_order(
    prior_order: list[int],
    *,
    seed: int,
    probabilities: np.ndarray | None,
    minimum_stratum_coverage: bool,
) -> list[int]:
    """Mirror train_phase after its indexed RaggedRows become a dense array."""

    if probabilities is None:
        order = list(prior_order)
        random.Random(seed).shuffle(order)
        return order
    return survival.stratified_low_variance_sampling_order(
        probabilities,
        row_count=len(prior_order),
        seed=seed,
        minimum_stratum_coverage=minimum_stratum_coverage,
    )


def exact_stage_teacher_forced_arrays(
    inputs: Any, labels: Any, progress_mask: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select teacher-forced targets from the exact staged transport arrays."""

    input_ids = np.asarray(inputs, dtype=np.int32)
    label_ids = np.asarray(labels, dtype=np.int32)
    authority = np.asarray(progress_mask, dtype=np.float32) > 0.0
    if (
        input_ids.ndim != 1
        or label_ids.shape != input_ids.shape
        or authority.shape != input_ids.shape
        or not bool(authority.any())
    ):
        raise ValueError("K5 exact staged teacher-forcing arrays are invalid")
    return input_ids, label_ids[authority], authority


def exact_coverage_planning_capacity(
    authoritative_phase: dict[str, Any],
    *,
    requested_steps: int,
    replay_step_limit: int,
    stage_row_count: int,
    segmented_contract: dict[str, Any] | None,
) -> int:
    """Preserve bounded population authority across segmented repeat steps."""

    coverage_receipt = authoritative_phase.get("coverage_first_sampling") or {}
    reported = int(coverage_receipt.get("planning_capacity") or 0)
    capacity = (
        int(segmented_contract["planning_capacity"])
        if segmented_contract is not None
        else reported or int(requested_steps)
    )
    required_capacity = min(int(replay_step_limit), int(stage_row_count))
    if capacity < required_capacity or capacity > int(stage_row_count):
        raise ValueError(
            "K5 coverage planning capacity falls outside the exact staged population"
        )
    return capacity


def align_exact_stage_predictions(
    predictions: Any, authority: np.ndarray
) -> np.ndarray:
    """Accept either full-sequence or compact target-only model execution."""

    values = np.asarray(predictions, dtype=np.int64)
    if values.ndim != 1 or authority.ndim != 1:
        raise ValueError("K5 staged prediction alignment requires vectors")
    if len(values) == len(authority):
        return values[authority]
    if len(values) == int(authority.sum()):
        return values
    raise ValueError(
        "K5 staged prediction alignment requires full or compact execution"
    )


def selected_training_rows(
    report: dict[str, Any],
    target: dict[str, Any],
    metadata: dict[str, Any],
    *,
    rows_per_objective: int = 1,
    authoritative_receipt_required: bool = True,
    projection_prior_optimizer_positions: int = 0,
    matched_row_ids: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= int(rows_per_objective) <= 16:
        raise ValueError("K5 probe rows per objective must be in [1, 16]")
    artifacts = target.get("kernel_english_artifacts") or {}
    artifact = artifacts.get("private_train") or {}
    path = resolve(str(artifact.get("path") or ""))
    if not path.is_file() or sha256(path) != str(artifact.get("sha256") or ""):
        raise ValueError("K5 private-train artifact identity mismatch")
    report_objective_order = tuple(
        ((report.get("kernel_english_training") or {}).get("learned_pipeline_contract") or {}).get(
            "objective_order"
        )
        or ()
    )
    config = read_json(resolve(report["config"]))
    lease = report.get("candidate_canary_lease") or {}
    execution_policy = lease.get("execution_policy") or {}
    stage_objective_filter = tuple(
        str(value)
        for value in (
            execution_policy.get("kerc_stage_objective_filter") or ()
        )
    )
    if stage_objective_filter and not set(stage_objective_filter).issubset(
        set(report_objective_order)
    ):
        raise ValueError("K5 stage objective filter is outside the report contract")
    objective_order = stage_objective_filter or report_objective_order
    maximum_rows = int(
        (lease.get("execution_policy") or {}).get(
            "kerc_bounded_source_row_limit", 0
        )
    ) or min(
        256,
        max(
            64,
            int(lease.get("requested_steps") or 0)
            * int((config.get("kernel_english_training") or {}).get("batch_size") or 1)
            * 2,
        ),
    )
    _source_count, selected = training.select_bounded_supervision_rows(
        path,
        split="private_train",
        objective_filter=objective_order,
        maximum_rows=maximum_rows,
    )
    base = read_json(resolve(report["base_config"]))
    eligible: list[dict[str, Any]] = []
    for source_index, row in selected:
        eligible.append(
            {
                **row,
                "source_row_id": str(row.get("row_id") or ""),
                "row_id": training.supervision_row_instance_id(
                    str(row.get("row_id") or ""),
                    artifact_key="private_train",
                    source_index=int(source_index),
                ),
            }
        )
    stage = training.materialize_target_supervision(
        config,
        base,
        target,
        metadata=metadata,
        artifact_field="kernel_english_artifacts",
        receipt_policy="project_theseus_moecot_kernel_english_arrays_v1",
        maximum_sequence_tokens=int(
            config["kernel_english_training"]["maximum_sequence_tokens"]
        ),
        objective_filter=objective_order,
        bounded_source_row_limit=maximum_rows,
    )
    stage = training.bound_supervision_stage_sequence_width(
        stage,
        maximum_sequence_tokens=int(execution_policy["maximum_training_sequence_tokens"]),
        maximum_supervised_sequence_tokens=int(
            execution_policy["maximum_supervised_training_sequence_tokens_by_target"][
                "english_kerc"
            ]
        ),
        required_kerc_coverage_labels=(
            tuple(
                str(value)
                for value in (
                    execution_policy.get("kerc_stage_required_coverage_labels")
                    or ()
                )
            )
            if stage_objective_filter
            else None
        ),
    )
    overfit_rows_per_objective = int(
        execution_policy.get("kerc_overfit_rows_per_objective") or 0
    )
    stage, observed_scope = replay_training_stage_row_scope(
        stage, execution_policy
    )
    admitted_ids = set(stage.training_row_ids)
    eligible = [
        row for row in eligible if str(row["row_id"]) in admitted_ids
    ]
    if execution_policy.get("kerc_delta_stage_only") is not None:
        if authoritative_receipt_required:
            phase = (
                (
                    candidate_evaluator.result_by_target(
                        report, "english_kerc"
                    ).get("phases")
                    or {}
                ).get("kernel_english")
                or {}
            )
            expected_scope = {
                key: int(phase.get(key) or 0) for key in observed_scope
            }
            if observed_scope != expected_scope:
                raise ValueError(
                    "K5 stage probe token-authority scope does not match "
                    "the authoritative training receipt"
                )
    objective_balanced_full_batch = bool(
        execution_policy.get("objective_balanced_full_batch")
    )
    if objective_balanced_full_batch:
        if rows_per_objective != 1:
            raise ValueError(
                "four-row K5 overfit probes require one row per objective"
            )
        result = candidate_evaluator.result_by_target(report, "english_kerc")
        phase = ((result.get("phases") or {}).get("kernel_english") or {})
        optimizer_steps = int(phase.get("optimizer_steps") or 0)
        optimizer_positions = int(
            result.get("kernel_english_optimizer_positions") or 0
        )
        single_objective_warmup_steps = int(
            execution_policy.get("kerc_overfit_single_objective_warmup_steps") or 0
        )
        fixed_objective_value = execution_policy.get(
            "kerc_overfit_fixed_objective_index"
        )
        fixed_objective_index = (
            int(fixed_objective_value)
            if fixed_objective_value is not None
            else None
        )
        exposure_batches = objective_balanced_exposure_batches(
            row_count=len(stage.inputs),
            optimizer_steps=optimizer_steps,
            single_objective_warmup_steps=single_objective_warmup_steps,
            fixed_objective_index=fixed_objective_index,
        )
        exposure_counts = Counter(
            int(index) for batch in exposure_batches for index in batch
        )
        expected_positions = sum(
            int(np.asarray(stage.mask[index]).sum()) * int(count)
            for index, count in exposure_counts.items()
        )
        if optimizer_steps <= 0:
            raise ValueError("objective-balanced K5 report has no optimizer steps")
        if optimizer_positions != expected_positions:
            raise ValueError(
                "objective-balanced K5 optimizer-position receipt does not match "
                "reconstructed objective exposure"
            )
        sampled_unique_row_count = len(exposure_counts)
        if int(phase.get("sampled_unique_row_count") or 0) != sampled_unique_row_count:
            raise ValueError(
                "objective-balanced K5 sampled-row receipt does not match reconstruction"
            )
        expected_schedule = training.kerc_overfit_batch_schedule(
            row_count=len(stage.inputs),
            single_objective_warmup_steps=single_objective_warmup_steps,
            fixed_objective_index=fixed_objective_index,
        )
        expected_schedule_sha256 = (
            hashlib.sha256(
                json.dumps(expected_schedule, separators=(",", ":")).encode()
            ).hexdigest()
            if expected_schedule
            else ""
        )
        schedule_receipt = {
            "batch_index_schedule_policy": phase.get("batch_index_schedule_policy"),
            "batch_index_schedule_length": int(
                phase.get("batch_index_schedule_length") or 0
            ),
            "batch_index_schedule_sha256": str(
                phase.get("batch_index_schedule_sha256") or ""
            ),
        }
        expected_schedule_receipt = {
            "batch_index_schedule_policy": (
                "finite_warmup_then_repeat_final_batch_v1"
                if expected_schedule
                else "ordinary_epoch_batches"
            ),
            "batch_index_schedule_length": len(expected_schedule),
            "batch_index_schedule_sha256": expected_schedule_sha256,
        }
        if schedule_receipt != expected_schedule_receipt:
            raise ValueError(
                "objective-balanced K5 batch schedule receipt does not match policy"
            )
        eligible_by_id = {str(row["row_id"]): row for row in eligible}
        chosen = []
        for objective_index, objective in enumerate(objective_order):
            objective_ids = [
                str(row_id)
                for row_id in stage.training_row_ids
                if str(row_id) in eligible_by_id
                and str(eligible_by_id[str(row_id)].get("objective") or "")
                == objective
            ]
            if len(objective_ids) != 1 or objective_ids[0] not in eligible_by_id:
                raise ValueError(
                    "objective-balanced K5 probe requires exactly one admitted row "
                    f"for objective {objective}"
                )
            chosen.append(
                {
                    **eligible_by_id[objective_ids[0]],
                    "sampled_optimizer_step_count": int(
                        exposure_counts.get(objective_index, 0)
                    ),
                }
            )
        return chosen, {
            "policy": "project_theseus_kerc_sampler_replay_v1",
            "execution": "objective_balanced_schedule_receipt_reconstruction",
            "stage_row_count": len(stage.inputs),
            "sampled_unique_row_count": sampled_unique_row_count,
            "sampled_row_coverage_rate": round(
                sampled_unique_row_count / max(1, len(stage.inputs)), 8
            ),
            "optimizer_steps": optimizer_steps,
            "optimizer_positions": optimizer_positions,
            "epochs_touched": int(phase.get("epochs_touched") or optimizer_steps),
            "per_row_optimizer_step_count": {
                str(stage.training_row_ids[index]): int(exposure_counts.get(index, 0))
                for index in range(len(stage.inputs))
            },
            **expected_schedule_receipt,
            "selection_uses_model_outcomes": False,
            "selection_uses_target_text": False,
        }
    sampling_weights = stage.sample_weights
    objective_sampling_receipt: dict[str, Any] | None = None
    if bool(execution_policy.get("kerc_objective_balanced_sampling")):
        sampling_weights, objective_sampling_receipt = (
            training.kerc_objective_balanced_sample_weights(
                stage,
                uniform_within_objective=bool(
                    execution_policy.get(
                        "kerc_uniform_within_objective_sampling", False
                    )
                ),
                objective_sampling_mass=(
                    execution_policy.get("kerc_objective_sampling_mass") or None
                ),
            )
        )
    probabilities = survival.normalized_sampling_probabilities(
        sampling_weights, len(stage.inputs)
    )
    row_costs = []
    for index in range(len(stage.inputs)):
        active = np.flatnonzero(
            (np.asarray(stage.inputs[index]) != 0)
            | (np.asarray(stage.mask[index]) != 0)
        )
        row_costs.append(int(active[-1] + 1) if len(active) else 1)
    authoritative_result = candidate_evaluator.result_by_target(
        report, "english_kerc"
    )
    authoritative_phase = (
        (authoritative_result.get("phases") or {}).get("kernel_english") or {}
    )
    source_result: dict[str, Any] | None = None
    source_report_value = str(
        execution_policy.get("continuation_source_report") or ""
    )
    if source_report_value:
        source_report_path = resolve(source_report_value)
        expected_source_report_sha256 = str(
            execution_policy.get("continuation_source_report_sha256") or ""
        )
        if (
            not source_report_path.is_file()
            or not expected_source_report_sha256
            or sha256(source_report_path) != expected_source_report_sha256
        ):
            raise ValueError(
                "K5 sampler source-report identity does not match execution policy"
            )
        source_result = candidate_evaluator.result_by_target(
            read_json(source_report_path), "english_kerc"
        )
    segmented_contract = segmented_sampler_replay_contract(
        authoritative_result,
        authoritative_phase,
        execution_policy,
        source_result=source_result,
    )
    replay_step_limit = (
        int(segmented_contract["optimizer_steps"])
        if segmented_contract is not None
        else int(lease["requested_steps"])
    )
    coverage_planning_capacity = exact_coverage_planning_capacity(
        authoritative_phase,
        requested_steps=int(lease["requested_steps"]),
        replay_step_limit=replay_step_limit,
        stage_row_count=len(stage.inputs),
        segmented_contract=segmented_contract,
    )
    coverage = survival.coverage_first_plan(
        stage.kerc_coverage_labels,
        (
            ()
            if overfit_rows_per_objective
            else tuple(
                str(value)
                for value in (
                    execution_policy.get("kerc_stage_required_coverage_labels")
                    or ()
                )
            )
            if stage_objective_filter
            else training.KERC_CANARY_REQUIRED_COVERAGE
        ),
        row_count=len(stage.inputs),
        capacity=coverage_planning_capacity,
        row_costs=tuple(row_costs),
    )
    coverage_prefix, resource_stress_receipt = resource_stress_prefix_replay(
        inputs=stage.inputs,
        mask=stage.mask,
        progress_mask=stage.mask,
        coverage_indices=coverage["selected_indices"],
        enabled=bool(execution_policy.get("kerc_resource_stress_prefix", False)),
        capacity=coverage_planning_capacity
        * int(execution_policy.get("batch_size") or 1),
    )
    target_positions = (
        int(segmented_contract["optimizer_positions"])
        if segmented_contract is not None
        else
        int(
            authoritative_result.get(
                "kernel_english_optimizer_target_positions"
            )
            or 0
        )
        if authoritative_receipt_required
        else max(
            0,
            int(stage.mask.sum())
            * int(execution_policy.get("kernel_optimizer_repetitions") or 1)
            - int(projection_prior_optimizer_positions),
        )
    )
    exposure_counts: dict[str, int] = {}
    sampled_indices: set[int] = set()
    sampled_index_prefix: list[int] = []
    batch_index_sha256_prefix: list[str] = []
    all_batch_index_sha256: list[str] = []
    replay_data_cursor_starts: list[dict[str, Any]] = []
    replay_data_cursor_next: list[dict[str, Any]] = []
    replay_batch_indices: list[tuple[int, ...]] = []
    replay_batch_positions: list[int] = []
    replay_coverage_counts = {
        str(label): 0 for label in coverage["required_labels"]
    }
    consumed = 0
    steps = 0
    epoch = 0
    replay_seed = kernel_phase_replay_seed(
        authoritative_result,
        authoritative_phase,
        int(lease["selected_seed"]),
    )
    persistent_order = list(range(len(stage.inputs)))
    batch_size = int(execution_policy.get("batch_size") or 1)
    while consumed < target_positions and steps < replay_step_limit:
        order = dense_training_epoch_order(
            (
                list(range(len(stage.inputs)))
                if segmented_contract is not None
                else persistent_order
            ),
            seed=replay_seed + epoch,
            probabilities=probabilities,
            minimum_stratum_coverage=bool(
                execution_policy.get(
                    "kerc_weighted_sampling_minimum_stratum_coverage", False
                )
            ),
        )
        if epoch == 0:
            order = survival.prepend_coverage_indices(order, coverage_prefix)
        persistent_order = list(order)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if consumed >= target_positions or steps >= replay_step_limit:
                break
            batch_sha256 = hashlib.sha256(
                np.asarray(indices, dtype=np.int64).tobytes()
            ).hexdigest()
            all_batch_index_sha256.append(batch_sha256)
            if len(batch_index_sha256_prefix) < 16:
                batch_index_sha256_prefix.append(batch_sha256)
            replay_data_cursor_starts.append(
                {
                    "policy": "project_theseus_training_data_cursor_v1",
                    "row_count": len(stage.inputs),
                    "batch_size": batch_size,
                    "seed": replay_seed,
                    "epoch": epoch,
                    "batch_index": start // batch_size,
                }
            )
            next_batch_index = start // batch_size + 1
            if next_batch_index >= math.ceil(len(order) / batch_size):
                replay_data_cursor_next.append(
                    {
                        "policy": "project_theseus_training_data_cursor_v1",
                        "row_count": len(stage.inputs),
                        "batch_size": batch_size,
                        "seed": replay_seed,
                        "epoch": epoch + 1,
                        "batch_index": 0,
                    }
                )
            else:
                replay_data_cursor_next.append(
                    {
                        "policy": "project_theseus_training_data_cursor_v1",
                        "row_count": len(stage.inputs),
                        "batch_size": batch_size,
                        "seed": replay_seed,
                        "epoch": epoch,
                        "batch_index": next_batch_index,
                    }
                )
            replay_batch_indices.append(tuple(int(index) for index in indices))
            batch_positions = int(np.asarray(stage.mask[indices]).sum())
            replay_batch_positions.append(batch_positions)
            for index in indices:
                sampled_indices.add(int(index))
                if len(sampled_index_prefix) < 16:
                    sampled_index_prefix.append(int(index))
                row_id = stage.training_row_ids[int(index)]
                exposure_counts[row_id] = exposure_counts.get(row_id, 0) + 1
                for label in stage.kerc_coverage_labels[int(index)]:
                    if label in replay_coverage_counts:
                        replay_coverage_counts[label] += 1
            consumed += batch_positions
            steps += 1
        epoch += 1
    expected_prefix_sha256 = hashlib.sha256(
        np.asarray(coverage_prefix, dtype=np.int64).tobytes()
    ).hexdigest()
    if authoritative_receipt_required:
        authoritative_coverage = (
            authoritative_phase.get("coverage_first_sampling") or {}
        )
        authoritative_stress = authoritative_coverage.get("resource_stress") or {}
        if authoritative_stress != resource_stress_receipt:
            raise ValueError(
                "K5 resource-stress sampler receipt does not match independent replay"
            )
        if (
            str(authoritative_coverage.get("selected_indices_sha256") or "")
            != expected_prefix_sha256
        ):
            raise ValueError(
                "K5 coverage/stress prefix identity does not match independent replay"
            )
        authoritative_batch_prefix = list(
            authoritative_phase.get("batch_index_sha256_prefix") or ()
        )
        if segmented_contract is not None:
            segment_steps = int(segmented_contract["segment_steps"])
            replay_contract = {
                "optimizer_steps": int(segmented_contract["optimizer_steps"]),
                "optimizer_positions": int(
                    segmented_contract["optimizer_positions"]
                ),
                "data_cursor_start": authoritative_phase.get(
                    "data_cursor_start"
                ),
                "data_cursor_next": authoritative_phase.get("data_cursor_next"),
                "segment_optimizer_positions": int(
                    authoritative_phase.get("target_positions_consumed") or 0
                ),
                "segment_sampled_unique_row_count": int(
                    authoritative_phase.get("sampled_unique_row_count") or 0
                ),
                "coverage_observed_label_counts": (
                    {
                        str(label): int(count)
                        - int(
                            (
                                segmented_contract.get(
                                    "source_coverage_observed_label_counts"
                                )
                                or {}
                            ).get(str(label), 0)
                        )
                        for label, count in (
                            authoritative_coverage.get(
                                "observed_label_counts"
                            )
                            or {}
                        ).items()
                    }
                ),
            }
            segment_indices = {
                index
                for batch in replay_batch_indices[-segment_steps:]
                for index in batch
            }
            replay_observed = {
                "optimizer_steps": steps,
                "optimizer_positions": consumed,
                "data_cursor_start": replay_data_cursor_starts[-segment_steps],
                "data_cursor_next": replay_data_cursor_next[-1],
                "segment_optimizer_positions": sum(
                    replay_batch_positions[-segment_steps:]
                ),
                "segment_sampled_unique_row_count": len(segment_indices),
                "coverage_observed_label_counts": replay_coverage_counts,
            }
            if (
                all_batch_index_sha256[-segment_steps:]
                != authoritative_batch_prefix
            ):
                raise ValueError(
                    "K5 segmented sampler batch suffix does not match "
                    "the authoritative final segment: "
                    f"expected={authoritative_batch_prefix} "
                    f"observed={all_batch_index_sha256[-segment_steps:]}"
                )
        else:
            if batch_index_sha256_prefix != authoritative_batch_prefix:
                raise ValueError(
                    "K5 sampler batch prefix does not match independent replay"
                )
            replay_contract = {
                "optimizer_steps": int(
                    authoritative_phase.get("optimizer_steps") or 0
                ),
                "optimizer_positions": int(
                    authoritative_phase.get("target_positions_consumed") or 0
                ),
                "sampled_unique_row_count": int(
                    authoritative_phase.get("sampled_unique_row_count") or 0
                ),
            }
            replay_observed = {
                "optimizer_steps": steps,
                "optimizer_positions": consumed,
                "sampled_unique_row_count": len(sampled_indices),
            }
        if replay_observed != replay_contract:
            raise ValueError(
                "K5 sampler replay does not match the authoritative training receipt: "
                f"expected={replay_contract} observed={replay_observed}"
            )
    if objective_sampling_receipt is not None and authoritative_receipt_required:
        authoritative_sampling = authoritative_phase.get("kerc_objective_sampling") or {}
        for key in (
            "policy",
            "row_count",
            "objective_row_counts",
            "objective_base_sampling_mass",
            "objective_balanced_sampling_mass",
            "weight_sha256",
        ):
            if objective_sampling_receipt.get(key) != authoritative_sampling.get(key):
                raise ValueError(
                    f"K5 objective-sampling replay mismatch: {key}"
                )
        authoritative_within_objective_policy = str(
            authoritative_sampling.get("within_objective_weight_policy")
            or "source_weight_preserving_legacy_v1"
        )
        if (
            objective_sampling_receipt.get("within_objective_weight_policy")
            != authoritative_within_objective_policy
        ):
            raise ValueError(
                "K5 objective-sampling replay mismatch: "
                "within_objective_weight_policy"
            )
    sampling_prefix = []
    for index in sampled_index_prefix:
        row_id = str(stage.training_row_ids[index])
        objective_labels = [
            label.split(":", 1)[1]
            for label in stage.kerc_coverage_labels[index]
            if label.startswith("objective:")
        ]
        if len(objective_labels) != 1:
            raise ValueError("K5 sampled row has no unique objective label")
        active = np.flatnonzero(
            (np.asarray(stage.inputs[index]) != 0)
            | (np.asarray(stage.mask[index]) != 0)
        )
        sampling_prefix.append(
            {
                "stage_index": int(index),
                "row_id": row_id,
                "objective": objective_labels[0],
                "execution_width": int(active[-1] + 1) if len(active) else 1,
                "target_positions": int(np.asarray(stage.mask[index]).sum()),
            }
        )
    chosen = []
    stage_index_by_id = {
        str(row_id): index
        for index, row_id in enumerate(stage.training_row_ids)
    }
    eligible_by_id = {str(row["row_id"]): row for row in eligible}
    if matched_row_ids:
        if (
            len(matched_row_ids) != rows_per_objective * len(objective_order)
            or len(set(matched_row_ids)) != len(matched_row_ids)
            or any(row_id not in eligible_by_id for row_id in matched_row_ids)
        ):
            raise ValueError(
                "matched K5 row selection is not an exact admitted row panel"
            )
        selected_rows_by_objective = {
            objective: [
                eligible_by_id[row_id]
                for row_id in matched_row_ids
                if str(eligible_by_id[row_id].get("objective") or "")
                == objective
            ]
            for objective in objective_order
        }
        if any(
            len(rows) != rows_per_objective
            for rows in selected_rows_by_objective.values()
        ):
            raise ValueError(
                "matched K5 row selection is not objective balanced"
            )
    else:
        selected_rows_by_objective = {}
        for objective in objective_order:
            rows = [
                row for row in eligible if row.get("objective") == objective
            ]
            if not rows:
                raise ValueError(
                    f"no envelope-admitted K5 row for objective {objective}"
                )
            selected_rows_by_objective[objective] = sorted(
                rows,
                key=lambda row: (
                    -int(exposure_counts.get(str(row["row_id"]), 0)),
                    str(row["row_id"]),
                ),
            )[:rows_per_objective]
    for objective in objective_order:
        selected_objective_rows = selected_rows_by_objective[objective]
        chosen.extend(
            {
                **selected_row,
                "sampled_optimizer_step_count": int(
                    exposure_counts.get(str(selected_row["row_id"]), 0)
                ),
                "sequence_tokens": int(
                    max(
                        np.flatnonzero(
                            (np.asarray(stage.inputs[
                                stage_index_by_id[str(selected_row["row_id"])]
                            ]) != 0)
                            | (np.asarray(stage.mask[
                                stage_index_by_id[str(selected_row["row_id"])]
                            ]) != 0)
                        ),
                        default=-1,
                    )
                    + 1
                ),
                "target_tokens": int(
                    np.asarray(
                        stage.mask[
                            stage_index_by_id[str(selected_row["row_id"])]
                        ]
                    ).sum()
                ),
                "_stage_inputs": np.asarray(
                    stage.inputs[
                        stage_index_by_id[str(selected_row["row_id"])]
                    ]
                ),
                "_stage_labels": np.asarray(
                    stage.labels[
                        stage_index_by_id[str(selected_row["row_id"])]
                    ]
                ),
                "_stage_progress_mask": np.asarray(
                    stage.mask[
                        stage_index_by_id[str(selected_row["row_id"])]
                    ]
                ),
                "_stage_loss_mask": np.asarray(
                    stage.loss_mask[
                        stage_index_by_id[str(selected_row["row_id"])]
                    ]
                ),
            }
            for selected_row in selected_objective_rows
        )
    objective_exposure = objective_exposure_projection(
        stage.kerc_coverage_labels,
        stage.training_row_ids,
        exposure_counts,
    )
    return chosen, {
        "policy": "project_theseus_kerc_sampler_replay_v1",
        "stage_row_count": len(stage.inputs),
        "sampled_unique_row_count": len(sampled_indices),
        "sampled_row_coverage_rate": round(
            len(sampled_indices) / max(1, len(stage.inputs)), 8
        ),
        "optimizer_steps": steps,
        "optimizer_positions": consumed,
        "replay_seed": replay_seed,
        "epochs_touched": epoch,
        "objective_sampling": objective_sampling_receipt,
        "semantic_target_position_counts_by_kind": dict(
            (stage.receipt or {}).get(
                "kerc_compiler_semantic_pointer_position_counts_by_kind"
            )
            or {}
        ),
        "semantic_target_position_count": int(
            (stage.receipt or {}).get(
                "kerc_compiler_semantic_pointer_position_count"
            )
            or 0
        ),
        "semantic_target_preweight_loss_mass_by_kind": dict(
            (stage.receipt or {}).get(
                "kerc_compiler_semantic_pointer_preweight_loss_mass_by_kind"
            )
            or {}
        ),
        "semantic_target_postweight_loss_mass_by_kind": dict(
            (stage.receipt or {}).get(
                "kerc_compiler_semantic_pointer_postweight_loss_mass_by_kind"
            )
            or {}
        ),
        "semantic_target_loss_weights_by_kind": dict(
            (stage.receipt or {}).get(
                "kerc_compiler_semantic_pointer_loss_weights_by_kind"
            )
            or {}
        ),
        "semantic_target_preweight_loss_histogram_by_kind": dict(
            (stage.receipt or {}).get(
                "kerc_compiler_semantic_pointer_preweight_loss_histogram_by_kind"
            )
            or {}
        ),
        "resource_stress": resource_stress_receipt,
        "execution_prefix_indices_sha256": expected_prefix_sha256,
        "sampling_prefix": sampling_prefix,
        "batch_index_sha256_prefix": batch_index_sha256_prefix,
        "authoritative_receipt_match": authoritative_receipt_required,
        "segmented_cumulative_replay": segmented_contract is not None,
        "matched_row_selection": bool(matched_row_ids),
        "matched_row_ids_sha256": (
            hashlib.sha256(
                json.dumps(
                    list(matched_row_ids), separators=(",", ":")
                ).encode()
            ).hexdigest()
            if matched_row_ids
            else ""
        ),
        "projection_only": not authoritative_receipt_required,
        "target_positions_requested": target_positions,
        "all_rows_sampled": len(sampled_indices) == len(stage.inputs),
        "unsampled_row_count": len(stage.inputs) - len(sampled_indices),
        "objective_exposure_projection": objective_exposure,
        "selection_uses_model_outcomes": False,
        "selection_uses_target_text": False,
        "minimum_stratum_coverage": bool(
            execution_policy.get(
                "kerc_weighted_sampling_minimum_stratum_coverage", False
            )
        ),
    }


def resolve_probe_checkpoint(
    result: dict[str, Any],
    *,
    diagnostic_checkpoint: str = "",
    diagnostic_checkpoint_sha256: str = "",
) -> tuple[Path, dict[str, Any]]:
    """Resolve either the receipt checkpoint or an explicit matched counterfactual."""

    if bool(diagnostic_checkpoint) != bool(diagnostic_checkpoint_sha256):
        raise ValueError(
            "diagnostic checkpoint path and SHA-256 are required together"
        )
    authoritative = resolve(str(result.get("checkpoint") or ""))
    authoritative_sha256 = str(result.get("checkpoint_sha256") or "")
    if (
        not authoritative.is_file()
        or sha256(authoritative) != authoritative_sha256
    ):
        raise ValueError("K5 stage-probe checkpoint identity mismatch")
    if not diagnostic_checkpoint:
        return authoritative, {
            "policy": "project_theseus_kerc_probe_receipt_checkpoint_v1",
            "matched_row_selection_source": "training_report",
            "counterfactual_checkpoint": False,
            "authoritative_training_checkpoint": (
                candidate_evaluator.source_artifact(authoritative)
            ),
        }
    selected = resolve(diagnostic_checkpoint)
    if (
        not selected.is_file()
        or sha256(selected) != diagnostic_checkpoint_sha256
    ):
        raise ValueError("K5 diagnostic checkpoint identity mismatch")
    return selected, {
        "policy": "project_theseus_kerc_matched_checkpoint_counterfactual_v1",
        "matched_row_selection_source": "training_report",
        "counterfactual_checkpoint": True,
        "authoritative_training_checkpoint": (
            candidate_evaluator.source_artifact(authoritative)
        ),
        "diagnostic_checkpoint": candidate_evaluator.source_artifact(selected),
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if not host_resource_safety.accelerator_child_authorized():
        raise ValueError("K5 stage probe requires the external host watchdog")
    import mlx.core as mx
    import mlx.nn as nn

    report_path = resolve(args.training_report)
    report = read_json(report_path)
    result = candidate_evaluator.result_by_target(report, "english_kerc")
    if int(result.get("candidate_seed") or 0) != int(args.seed):
        raise ValueError("K5 stage-probe seed mismatch")
    checkpoint, checkpoint_selection = resolve_probe_checkpoint(
        result,
        diagnostic_checkpoint=str(
            getattr(args, "diagnostic_checkpoint", "") or ""
        ),
        diagnostic_checkpoint_sha256=str(
            getattr(args, "diagnostic_checkpoint_sha256", "") or ""
        ),
    )
    metadata = read_json(resolve(str((report.get("stage") or {}).get("metadata") or "")))
    base = read_json(resolve(str(report.get("base_config") or "")))
    target = copy.deepcopy((report.get("targets") or {})["english_kerc"])
    execution_policy = (
        (report.get("candidate_canary_lease") or {}).get("execution_policy") or {}
    )
    source_vocab = dict(metadata["source_vocab"])
    target_vocab = dict(metadata["target_vocab"])
    model = training.build_model(
        training.CausalTransformerConfig(
            vocab_size=int(target["vocab_size"]), **target["model"]
        ),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=training.build_source_to_target_lookup(
            base,
            metadata,
            vocab_size=int(target["vocab_size"]),
            identity_ranges=training.target_copy_identity_ranges(target),
        ),
        gradient_checkpointing=False,
        attention_query_chunk_size=int(
            execution_policy.get("attention_query_chunk_size") or 0
        ),
        attention_key_chunk_size=int(
            execution_policy.get("attention_key_chunk_size") or 0
        ),
        compact_encoder_decoder_partitions=bool(
            execution_policy.get("compact_encoder_decoder_partitions", False)
        ),
        compact_partition_width_quantum=int(
            execution_policy.get("compact_partition_width_quantum") or 0
        ),
    )
    model.load_weights(str(checkpoint))
    evaluation_compute_dtype = str(
        getattr(args, "evaluation_compute_dtype", "authoritative_fp32")
    )
    if evaluation_compute_dtype == "bfloat16":
        model.set_dtype(mx.bfloat16)
    elif evaluation_compute_dtype != "authoritative_fp32":
        raise ValueError("unsupported K5 evaluation compute dtype")
    mx.eval(model.parameters())
    model.eval()
    config = training.bind_scale_preregistration(read_json(resolve(report["config"])))
    envelope = int(
        (((report.get("candidate_canary_lease") or {}).get("execution_policy") or {})
        .get("maximum_supervised_training_sequence_tokens_by_target", {})
        .get("english_kerc", 0))
    )
    code_vocabulary = (target.get("kernel_code_vocabulary") or {}).get("payload") or {}
    model_contract = target["model"]
    rows = []
    token_accuracy_counts: dict[int, dict[str, Any]] = {}
    token_label_lookup = {
        int(model_contract["kerc_kernel_token_start"]) + int(local_id): str(token)
        for token, local_id in (code_vocabulary.get("kernel_vocab") or {}).items()
    }
    token_label_lookup.update(
        {
            int(model_contract["kerc_pointer_token_start"]) + int(local_id): str(token)
            for token, local_id in (
                code_vocabulary.get("pointer_vocab") or {}
            ).items()
        }
    )
    token_label_lookup[int(model_contract["kerc_end_token_id"])] = "<KERC_END>"
    gradient_examples: list[dict[str, Any]] = []
    rows_per_objective = int(getattr(args, "rows_per_objective", 1))
    diagnostic_beam_width = int(getattr(args, "beam_width", 1))
    diagnostic_branching_factor = int(
        getattr(args, "branching_factor", 1)
    )
    online_transport_validator = bool(
        getattr(args, "online_transport_validator", False)
    )
    if args.gradient_interference and rows_per_objective != 1:
        raise ValueError(
            "gradient-interference diagnostics require one row per objective"
        )
    matched_row_report_path: Path | None = None
    matched_row_report_artifact: dict[str, Any] | None = None
    retained_row_report_path: Path | None = None
    retained_row_report_artifact: dict[str, Any] | None = None
    training_row_panel_report_path: Path | None = None
    training_row_panel_report_artifact: dict[str, Any] | None = None
    matched_row_ids: tuple[str, ...] = ()
    if str(getattr(args, "matched_row_report", "") or ""):
        matched_row_report_path = resolve(args.matched_row_report)
        matched_row_report = read_json(matched_row_report_path)
        matched_rows = matched_row_report.get("rows")
        if (
            matched_row_report.get("policy") != POLICY
            or matched_row_report.get("qualification_state")
            != "TEACHER_FORCED_DIAGNOSTIC_ONLY"
            or not isinstance(matched_rows, list)
            or not matched_rows
            or any(
                not isinstance(row, dict) or not str(row.get("row_id") or "")
                for row in matched_rows
            )
        ):
            raise ValueError(
                "matched K5 row report is not a teacher-forced probe"
            )
        matched_row_ids = tuple(str(row["row_id"]) for row in matched_rows)
        matched_row_report_artifact = candidate_evaluator.source_artifact(
            matched_row_report_path
        )
        checkpoint_selection["matched_row_selection_source"] = (
            "explicit_teacher_forced_probe_report"
        )
    if str(getattr(args, "retained_row_report", "") or ""):
        retained_row_report_path = resolve(args.retained_row_report)
        retained_row_report = read_json(retained_row_report_path)
        retained_rows = retained_row_report.get("rows")
        if (
            retained_row_report.get("policy") != POLICY
            or not isinstance(retained_rows, list)
            or len(retained_rows) != 1
            or not isinstance(retained_rows[0], dict)
            or retained_rows[0].get("admitted_training_row") is not True
            or not str(retained_rows[0].get("row_id") or "")
            or int(retained_row_report.get("public_benchmark_prompts_used") or 0)
            != 0
            or int(retained_row_report.get("external_inference_calls") or 0) != 0
            or int(
                retained_row_report.get("fallback_template_router_tool_credit") or 0
            )
            != 0
        ):
            raise ValueError(
                "retained K5 row report is not an exact training-row probe"
            )
        matched_row_ids = (str(retained_rows[0]["row_id"]),)
        retained_row_report_artifact = candidate_evaluator.source_artifact(
            retained_row_report_path
        )
        checkpoint_selection["matched_row_selection_source"] = (
            "explicit_retained_training_row_probe_report"
        )
    if str(getattr(args, "training_row_panel_report", "") or ""):
        training_row_panel_report_path = resolve(args.training_row_panel_report)
        training_row_panel_report = read_json(training_row_panel_report_path)
        matched_row_ids = exact_training_row_panel_ids(
            training_row_panel_report
        )
        training_row_panel_report_artifact = (
            candidate_evaluator.source_artifact(
                training_row_panel_report_path
            )
        )
        checkpoint_selection["matched_row_selection_source"] = (
            "explicit_admitted_training_row_panel_report"
        )
    selected_rows, sampling_replay = selected_training_rows(
        report,
        target,
        metadata,
        rows_per_objective=rows_per_objective,
        matched_row_ids=matched_row_ids,
    )
    for row in selected_rows:
        objective = str(row["objective"])
        structured_source = objective in training.KERC_STRUCTURED_SOURCE_OBJECTIVES
        prepared = training.prepare_model_text_prompt(
            str(row["prompt"]),
            source_vocab,
            target_vocab,
            base,
            max_source_tokens=int(config["kernel_english_training"]["maximum_sequence_tokens"]),
            trusted_source_prefix_tokens=(training.TRAINING_TASK_TAGS[objective],),
            structured_source_code_vocabulary=(code_vocabulary if structured_source else None),
            structured_source_kernel_offset=int(model_contract["kerc_kernel_token_start"]),
            structured_source_pointer_offset=int(model_contract["kerc_pointer_token_start"]),
        )
        if prepared.get("fault"):
            raise ValueError(f"K5 stage-probe prompt became invalid: {objective}")
        stage_inputs, supervised_expected, supervised_authority = (
            exact_stage_teacher_forced_arrays(
                row["_stage_inputs"],
                row["_stage_labels"],
                row["_stage_progress_mask"],
            )
        )
        teacher_logits, teacher_cache = model(
            mx.array(
                np.asarray([stage_inputs], dtype=np.int32),
                dtype=mx.int32,
            )
        )
        teacher_predictions = mx.argmax(teacher_logits[0], axis=-1)
        mx.eval(teacher_predictions, *training.cache_arrays(teacher_cache))
        predicted_ids = np.asarray(teacher_predictions, dtype=np.int64)
        supervised_predictions = align_exact_stage_predictions(
            predicted_ids, supervised_authority
        )
        gradient_mask = np.asarray(
            [row["_stage_loss_mask"]], dtype=np.float32
        )
        gradient_examples.append(
            {
                "objective": objective,
                "inputs": np.asarray([stage_inputs], dtype=np.int32),
                "labels": np.asarray([row["_stage_labels"]], dtype=np.int32),
                "mask": gradient_mask,
            }
        )
        teacher_correct = int(
            np.sum(supervised_predictions == supervised_expected)
        )
        teacher_total = int(len(supervised_expected))
        teacher_eos_top1 = bool(
            teacher_total and supervised_predictions[-1] == supervised_expected[-1]
        )
        token_regions = {
            "surface": (
                int(model_contract["kerc_surface_token_start"]),
                int(model_contract["kerc_surface_token_end"]),
            ),
            "kernel": (
                int(model_contract["kerc_kernel_token_start"]),
                int(model_contract["kerc_kernel_token_end"]),
            ),
            "pointer": (
                int(model_contract["kerc_pointer_token_start"]),
                int(model_contract["kerc_pointer_token_end"]),
            ),
        }
        end_mask = supervised_expected == int(model_contract["kerc_end_token_id"])
        region_masks = {
            name: (supervised_expected >= start) & (supervised_expected < end)
            & ~end_mask
            for name, (start, end) in token_regions.items()
        }
        region_masks["end"] = end_mask
        covered = np.zeros_like(supervised_expected, dtype=bool)
        for mask in region_masks.values():
            covered |= mask
        region_masks["other"] = ~covered
        accuracy_by_region = {}
        correctness = supervised_predictions == supervised_expected
        for name, mask in region_masks.items():
            total = int(np.sum(mask))
            correct = int(np.sum(correctness & mask))
            accuracy_by_region[name] = {
                "correct": correct,
                "total": total,
                "accuracy": round(correct / max(1, total), 8),
            }
        for expected_id, is_correct in zip(
            supervised_expected.tolist(), correctness.tolist()
        ):
            token_id = int(expected_id)
            token_row = token_accuracy_counts.setdefault(
                token_id,
                {
                    "expected_token_id": token_id,
                    "token": token_label_lookup.get(
                        token_id, f"<TOKEN:{token_id}>"
                    ),
                    "correct": 0,
                    "total": 0,
                },
            )
            token_row["total"] += 1
            token_row["correct"] += int(bool(is_correct))
        expected_frequencies = Counter(int(value) for value in supervised_expected)
        frequency_masks = {
            "singleton": np.asarray(
                [expected_frequencies[int(value)] == 1 for value in supervised_expected],
                dtype=bool,
            ),
            "repeated": np.asarray(
                [expected_frequencies[int(value)] > 1 for value in supervised_expected],
                dtype=bool,
            ),
        }
        accuracy_by_frequency = {}
        for name, mask in frequency_masks.items():
            total = int(np.sum(mask))
            correct = int(np.sum(correctness & mask))
            accuracy_by_frequency[name] = {
                "correct": correct,
                "total": total,
                "accuracy": round(correct / max(1, total), 8),
            }
        del teacher_logits, teacher_cache, teacher_predictions
        mx.clear_cache()
        teacher_forced_only = bool(args.teacher_forced_only)
        if teacher_forced_only:
            generated = ""
            generation = {
                "state": "NOT_RUN_TEACHER_FORCED_DIAGNOSTIC_ONLY",
                "stop_reason": None,
                "reason": "",
                "generated_token_count": 0,
            }
            syntax_valid = False
        elif objective in training.KERC_KERNEL_OBJECTIVES:
            generated, generation = training.generate_kerc_code_text(
                model,
                str(row["prompt"]),
                source_vocab,
                target_vocab,
                base,
                code_vocabulary=code_vocabulary,
                kernel_offset=int(model_contract["kerc_kernel_token_start"]),
                pointer_offset=int(model_contract["kerc_pointer_token_start"]),
                pointer_end=int(model_contract["kerc_pointer_token_end"]),
                max_tokens=envelope,
                max_source_tokens=int(config["kernel_english_training"]["maximum_sequence_tokens"]),
                beam_width=diagnostic_beam_width,
                branching_factor=diagnostic_branching_factor,
                length_penalty=float(config["evaluation"]["length_penalty"]),
                trusted_source_prefix_token=training.TRAINING_TASK_TAGS[objective],
                structured_source=objective in training.KERC_STRUCTURED_SOURCE_OBJECTIVES,
                completion_validator=(
                    lambda text: (
                        kernel_protocol.decode_learned_compiler_transport(
                            text
                        )
                    )
                    if online_transport_validator
                    and objective == "surface_to_kernel_program_v1"
                    else None
                ),
                online_completion_validation=(
                    online_transport_validator
                    and objective == "surface_to_kernel_program_v1"
                ),
                mx=mx,
            )
            try:
                json.loads(generated)
                syntax_valid = True
            except (TypeError, ValueError, json.JSONDecodeError):
                syntax_valid = False
        else:
            generated, generation = training.generate_model_text(
                model,
                str(row["prompt"]),
                source_vocab,
                target_vocab,
                base,
                max_tokens=envelope,
                max_source_tokens=int(config["kernel_english_training"]["maximum_sequence_tokens"]),
                beam_width=1,
                branching_factor=1,
                length_penalty=float(config["evaluation"]["length_penalty"]),
                trusted_source_prefix_tokens=(training.TRAINING_TASK_TAGS[objective],),
                structured_source_code_vocabulary=(
                    code_vocabulary
                    if objective in training.KERC_STRUCTURED_SOURCE_OBJECTIVES
                    else None
                ),
                structured_source_kernel_offset=int(model_contract["kerc_kernel_token_start"]),
                structured_source_pointer_offset=int(model_contract["kerc_pointer_token_start"]),
                mx=mx,
            )
            syntax_valid = bool(generated)
        compiler_transport_valid: bool | None = None
        compiler_transport_fault = ""
        compiler_transport_shape: dict[str, Any] | None = None
        if (
            not teacher_forced_only
            and objective == "surface_to_kernel_program_v1"
            and generated
        ):
            try:
                decoded_transport = json.loads(generated)
                compiler_transport_shape = (
                    kernel_protocol.learned_compiler_transport_shape_signature(
                        decoded_transport
                    )
                )
                kernel_protocol.materialize_learned_compiler_transport(
                    decoded_transport
                )
                compiler_transport_valid = True
            except kernel_protocol.KernelProtocolFault as exc:
                compiler_transport_valid = False
                compiler_transport_fault = exc.code
            except (TypeError, ValueError, json.JSONDecodeError):
                compiler_transport_valid = False
                compiler_transport_fault = (
                    "KERC_LEARNED_COMPILER_OUTPUT_INVALID"
                )
        rows.append(
            {
                "row_id": row["row_id"],
                "objective": objective,
                "admitted_training_row": True,
                "sampled_training_row": int(row["sampled_optimizer_step_count"]) > 0,
                "sampled_optimizer_step_count": int(
                    row["sampled_optimizer_step_count"]
                ),
                "sequence_tokens": int(row["sequence_tokens"]),
                "target_tokens": int(row["target_tokens"]),
                "teacher_forced_top1_correct": teacher_correct,
                "teacher_forced_top1_total": teacher_total,
                "teacher_forced_top1_accuracy": round(
                    teacher_correct / max(1, teacher_total), 8
                ),
                "teacher_forced_eos_top1": teacher_eos_top1,
                "teacher_forced_unique_expected_token_count": len(
                    expected_frequencies
                ),
                "teacher_forced_accuracy_by_token_region": accuracy_by_region,
                "teacher_forced_accuracy_by_target_frequency": accuracy_by_frequency,
                "generated_character_count": len(generated),
                "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                "exact_match": (
                    None
                    if teacher_forced_only
                    else generated == str(row["target"])
                ),
                "syntax_valid": (
                    None if teacher_forced_only else syntax_valid
                ),
                "compiler_transport_valid": compiler_transport_valid,
                "compiler_transport_fault": compiler_transport_fault,
                "compiler_transport_shape": compiler_transport_shape,
                "generation_state": generation.get("state"),
                "stop_reason": generation.get("stop_reason"),
                "fault_reason": generation.get("reason") or "",
                "generation_semantic_selection": (
                    generation.get("semantic_selection")
                    if online_transport_validator
                    else None
                ),
                "generated_token_count": int(generation.get("generated_token_count") or 0),
                "raw_generated_text_retained": False,
            }
        )
    gradient_interference = None
    if args.gradient_interference:
        from mlx.utils import tree_flatten

        execution_policy = (
            (report.get("candidate_canary_lease") or {}).get("execution_policy") or {}
        )
        model.freeze_to_kerc_delta(
            include_source_conditioned_bridge=bool(
                execution_policy.get("kerc_delta_include_source_conditioned_bridge")
            )
        )
        model.train()
        loss_and_grad = nn.value_and_grad(model, training.causal_loss)
        snapshots: dict[str, dict[str, np.ndarray]] = {}
        losses: dict[str, float] = {}
        assumed_separator_losses: dict[str, float] = {}
        for example in gradient_examples:
            objective = str(example["objective"])
            loss, gradients = loss_and_grad(
                model,
                mx.array(example["inputs"], dtype=mx.int32),
                mx.array(example["labels"], dtype=mx.int32),
                mx.array(example["mask"], dtype=mx.float32),
                mx,
                nn,
                source_conditioning=None,
            )
            mx.eval(loss, gradients)
            losses[objective] = round(float(loss.item()), 8)
            snapshots[objective] = {
                str(name): np.asarray(value, dtype=np.float32)
                for name, value in tree_flatten(gradients)
            }
            del gradients
            mx.clear_cache()
            assumed_loss, assumed_gradients = loss_and_grad(
                model,
                mx.array(example["inputs"], dtype=mx.int32),
                mx.array(example["labels"], dtype=mx.int32),
                mx.array(example["mask"], dtype=mx.float32),
                mx,
                nn,
                source_conditioning=True,
            )
            mx.eval(assumed_loss, assumed_gradients)
            assumed_separator_losses[objective] = round(
                float(assumed_loss.item()), 8
            )
            del assumed_gradients
            mx.clear_cache()
        objectives = [str(row["objective"]) for row in selected_rows]
        pairwise = []
        for left_index, left in enumerate(objectives):
            for right in objectives[left_index + 1 :]:
                pairwise.append(
                    {
                        "left_objective": left,
                        "right_objective": right,
                        "groups": gradient_pair_metrics(
                            snapshots[left], snapshots[right]
                        ),
                    }
                )
        gradient_interference = {
            "state": "MEASURED_TOKEN_AND_COPY_OBJECTIVE_DIAGNOSTIC",
            "trainable_scope": "frozen_warm_trunk_kerc_delta_with_configured_source_bridge",
            "objective_losses": losses,
            "objective_losses_assume_separator": assumed_separator_losses,
            "pairwise": pairwise,
            "parameter_inventory_exact_across_objectives": True,
            "includes_kerc_auxiliary_objectives": False,
            "claim_boundary": "Gradient cosine diagnoses this checkpoint and token/copy objective only; it is not KERC utility or causal architecture evidence.",
        }
        del snapshots
        model.eval()
        mx.clear_cache()
    exact_count = sum(row["exact_match"] is True for row in rows)
    syntax_count = sum(row["syntax_valid"] is True for row in rows)
    teacher_forced_summary = aggregate_teacher_forced_rows(rows)
    teacher_forced_token_diagnostics = []
    for token_row in token_accuracy_counts.values():
        correct = int(token_row["correct"])
        total = int(token_row["total"])
        teacher_forced_token_diagnostics.append(
            {
                **token_row,
                "error_count": total - correct,
                "accuracy": round(correct / max(1, total), 8),
            }
        )
    teacher_forced_token_diagnostics.sort(
        key=lambda row: (
            -int(row["error_count"]),
            -int(row["total"]),
            int(row["expected_token_id"]),
        )
    )
    payload = {
        "policy": POLICY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_state": "GREEN",
        "qualification_state": (
            "TEACHER_FORCED_DIAGNOSTIC_ONLY"
            if args.teacher_forced_only
            else "LEARNABILITY_SANITY_GREEN"
            if exact_count == len(rows)
            else "LEARNABILITY_SANITY_FAILED"
        ),
        "free_generation_evaluated": not bool(args.teacher_forced_only),
        "candidate_id": str(
            (report.get("candidate_canary_lease") or {}).get("candidate_id") or ""
        ),
        "seed": int(args.seed),
        "checkpoint": candidate_evaluator.source_artifact(checkpoint),
        "checkpoint_selection": checkpoint_selection,
        "evaluation_compute_dtype": evaluation_compute_dtype,
        "evaluation_execution_policy": {
            "attention_query_chunk_size": int(
                execution_policy.get("attention_query_chunk_size") or 0
            ),
            "attention_key_chunk_size": int(
                execution_policy.get("attention_key_chunk_size") or 0
            ),
            "compact_encoder_decoder_partitions": bool(
                execution_policy.get("compact_encoder_decoder_partitions", False)
            ),
            "beam_width": diagnostic_beam_width,
            "branching_factor": diagnostic_branching_factor,
            "online_transport_validator": online_transport_validator,
            "online_transport_validator_credit": (
                "ASSISTED_DIAGNOSTIC_ONLY"
                if online_transport_validator
                else "NONE"
            ),
        },
        "training_report": candidate_evaluator.source_artifact(report_path),
        "matched_row_report": matched_row_report_artifact,
        "retained_row_report": retained_row_report_artifact,
        "training_row_panel_report": training_row_panel_report_artifact,
        "candidate_supervised_sequence_envelope": envelope,
        "row_count": len(rows),
        "objective_count": len({row["objective"] for row in rows}),
        "sampling_replay": sampling_replay,
        "rows_per_objective": rows_per_objective,
        "teacher_forced_summary_by_objective": teacher_forced_summary,
        "teacher_forced_token_diagnostics_by_error": (
            teacher_forced_token_diagnostics
        ),
        "gradient_interference_evaluated": bool(args.gradient_interference),
        "gradient_interference": gradient_interference,
        "exact_match_count": exact_count,
        "syntax_valid_count": syntax_count,
        "rows": rows,
        "compiler_transport_diagnostic": {
            "evaluated_row_count": sum(
                row["compiler_transport_valid"] is not None for row in rows
            ),
            "valid_row_count": sum(
                row["compiler_transport_valid"] is True for row in rows
            ),
            "fault_counts": dict(
                sorted(
                    Counter(
                        str(row["compiler_transport_fault"])
                        for row in rows
                        if row["compiler_transport_fault"]
                    ).items()
                )
            ),
            "raw_generated_text_retained": False,
            "shape_signatures_retain_semantic_values": False,
            "online_transport_validator": online_transport_validator,
            "assisted_output_credit_required": online_transport_validator,
        },
        "generator_visible_fields": ["trusted_source_prefix_tokens", "prompt"],
        "target_visible_to_generator": False,
        "public_benchmark_prompts_used": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "capability_claim": "NONE_TRAINING_ROW_OVERFIT_DIAGNOSTIC_ONLY",
    }
    write_json(resolve(args.out), payload)
    return payload


def project_current_candidate_scheduler(args: argparse.Namespace) -> dict[str, Any]:
    report_path = resolve(args.training_report)
    report = read_json(report_path)
    source_result = candidate_evaluator.result_by_target(report, "english_kerc")
    if int(source_result.get("candidate_seed") or 0) != int(args.seed):
        raise ValueError("K5 scheduler projection seed mismatch")
    source_checkpoint = resolve(str(source_result.get("checkpoint") or ""))
    if (
        not source_checkpoint.is_file()
        or sha256(source_checkpoint) != source_result.get("checkpoint_sha256")
    ):
        raise ValueError("K5 scheduler projection checkpoint identity mismatch")
    candidate_contract = training.pretraining_candidate_canary.load_contract()
    candidate_id = str(
        (report.get("candidate_canary_lease") or {}).get("candidate_id") or ""
    )
    candidate = next(
        (
            row
            for row in candidate_contract["canaries"]
            if row["candidate_id"] == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError("K5 scheduler projection candidate is not registered")
    lease = training.pretraining_candidate_canary.candidate_lease(
        candidate_id=candidate_id,
        max_steps=int(candidate["max_steps"]),
        scratch_checkpoint_root=(
            ROOT
            / "runtime"
            / "t0a_canaries"
            / candidate_id
            / "scheduler_projection_only"
        ),
        targets=["english_kerc"],
        phase="kernel_english",
        resume=False,
        selected_seed=int(args.seed),
        contract=candidate_contract,
    )
    if lease.get("authorized") is not True:
        raise ValueError(
            "K5 scheduler projection candidate lease denied: "
            + ",".join(lease.get("faults") or [])
        )
    projected_report = copy.deepcopy(report)
    projected_report["candidate_canary_lease"] = lease
    metadata = read_json(
        resolve(str((report.get("stage") or {}).get("metadata") or ""))
    )
    target = copy.deepcopy((report.get("targets") or {})["english_kerc"])
    _selected, replay = selected_training_rows(
        projected_report,
        target,
        metadata,
        rows_per_objective=1,
        authoritative_receipt_required=False,
        projection_prior_optimizer_positions=int(
            source_result.get("kernel_english_optimizer_positions") or 0
        ),
    )
    objective_exposure = replay["objective_exposure_projection"]
    all_objectives_covered = all(
        int(row["unsampled_row_count"]) == 0
        and int(row["minimum_row_exposures"]) >= 1
        for row in objective_exposure.values()
    )
    coverage_complete = bool(replay["all_rows_sampled"]) and all_objectives_covered
    payload = {
        "policy": "project_theseus_kerc_k5_scheduler_projection_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_state": "GREEN" if coverage_complete else "RED",
        "qualification_state": (
            "SCHEDULER_PROJECTION_GREEN_NO_BEHAVIOR_CREDIT"
            if coverage_complete
            else "SCHEDULER_PROJECTION_RED"
        ),
        "candidate_id": candidate_id,
        "seed": int(args.seed),
        "candidate_lease_digest": lease["lease_digest"],
        "execution_policy": lease["execution_policy"],
        "source_training_report": candidate_evaluator.source_artifact(report_path),
        "source_checkpoint": candidate_evaluator.source_artifact(source_checkpoint),
        "source_optimizer_steps": int(source_result.get("optimizer_steps") or 0),
        "source_kernel_optimizer_positions": int(
            source_result.get("kernel_english_optimizer_positions") or 0
        ),
        "scheduler_projection": replay,
        "all_positive_rows_covered": coverage_complete,
        "selection_uses_model_outcomes": False,
        "selection_uses_target_text": False,
        "public_benchmark_prompts_used": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "capability_claim": "NONE_SCHEDULER_PROJECTION_ONLY",
    }
    write_json(resolve(args.out), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--scheduler-projection-only", action="store_true")
    parser.add_argument("--teacher-forced-only", action="store_true")
    parser.add_argument("--gradient-interference", action="store_true")
    parser.add_argument("--diagnostic-checkpoint", default="")
    parser.add_argument("--diagnostic-checkpoint-sha256", default="")
    parser.add_argument("--matched-row-report", default="")
    parser.add_argument("--retained-row-report", default="")
    parser.add_argument("--training-row-panel-report", default="")
    parser.add_argument("--rows-per-objective", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--branching-factor", type=int, default=1)
    parser.add_argument(
        "--online-transport-validator", action="store_true"
    )
    parser.add_argument(
        "--evaluation-compute-dtype",
        choices=("authoritative_fp32", "bfloat16"),
        default="authoritative_fp32",
    )
    args = parser.parse_args()
    if bool(args.diagnostic_checkpoint) != bool(
        args.diagnostic_checkpoint_sha256
    ):
        parser.error(
            "--diagnostic-checkpoint and --diagnostic-checkpoint-sha256 "
            "are required together"
        )
    if args.diagnostic_checkpoint and not args.teacher_forced_only:
        parser.error(
            "a matched diagnostic checkpoint is restricted to "
            "--teacher-forced-only evaluation"
        )
    if args.matched_row_report and not args.teacher_forced_only:
        parser.error(
            "an explicit matched row report is restricted to "
            "--teacher-forced-only evaluation"
        )
    if (
        bool(args.matched_row_report)
        + bool(args.retained_row_report)
        + bool(args.training_row_panel_report)
        > 1
    ):
        parser.error(
            "K5 row-selection reports are mutually exclusive"
        )
    if args.retained_row_report and (
        args.teacher_forced_only or args.rows_per_objective != 1
    ):
        parser.error(
            "an explicit retained row report requires free generation and "
            "--rows-per-objective 1"
        )
    if args.training_row_panel_report and (
        args.teacher_forced_only or args.rows_per_objective < 2
    ):
        parser.error(
            "an explicit training-row panel requires free generation and "
            "--rows-per-objective >= 2"
        )
    if not 1 <= args.beam_width <= 16:
        parser.error("--beam-width must be in [1, 16]")
    if not 1 <= args.branching_factor <= 16:
        parser.error("--branching-factor must be in [1, 16]")
    if args.teacher_forced_only and (
        args.beam_width != 1
        or args.branching_factor != 1
        or args.online_transport_validator
    ):
        parser.error(
            "teacher-forced-only diagnostics do not accept beam overrides"
        )
    if args.scheduler_projection_only:
        if (
            args.guarded
            or args.teacher_forced_only
            or args.gradient_interference
            or args.diagnostic_checkpoint
            or args.diagnostic_checkpoint_sha256
            or args.matched_row_report
            or args.rows_per_objective != 1
            or args.beam_width != 1
            or args.branching_factor != 1
            or args.online_transport_validator
            or args.evaluation_compute_dtype != "authoritative_fp32"
        ):
            parser.error(
                "--scheduler-projection-only cannot be combined with accelerator "
                "probe options"
            )
        result = project_current_candidate_scheduler(args)
        print(
            json.dumps(
                {
                    "qualification_state": result["qualification_state"],
                    "all_positive_rows_covered": result[
                        "all_positive_rows_covered"
                    ],
                    "optimizer_steps": result["scheduler_projection"][
                        "optimizer_steps"
                    ],
                    "optimizer_positions": result["scheduler_projection"][
                        "optimizer_positions"
                    ],
                },
                indent=2,
            )
        )
        return 0 if result["trigger_state"] == "GREEN" else 2
    if args.guarded:
        training_report = read_json(resolve(args.training_report))
        lease = training_report.get("candidate_canary_lease") or {}
        candidate_id = str(lease.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("K5 guarded probe requires a candidate-bound training report")
        candidate_contract = training.pretraining_candidate_canary.load_contract()
        candidate = next(
            (
                row
                for row in candidate_contract["canaries"]
                if row["candidate_id"] == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ValueError("K5 guarded probe candidate is not registered")
        training_config = read_json(
            resolve(str(training_report.get("config") or ""))
        )
        qualified_python = resolve(
            str(
                (training_config.get("host_resource_safety") or {}).get(
                    "qualified_python"
                )
                or ""
            )
        )
        if not qualified_python.is_file():
            raise ValueError("K5 guarded probe qualified Python is missing")
        command = [
            str(qualified_python),
            str(Path(__file__).resolve()),
            "--training-report",
            args.training_report,
            "--seed",
            str(args.seed),
            "--out",
            args.out,
        ]
        if args.teacher_forced_only:
            command.append("--teacher-forced-only")
        if args.diagnostic_checkpoint:
            command.extend(
                [
                    "--diagnostic-checkpoint",
                    args.diagnostic_checkpoint,
                    "--diagnostic-checkpoint-sha256",
                    args.diagnostic_checkpoint_sha256,
                ]
            )
        if args.matched_row_report:
            command.extend(
                ["--matched-row-report", args.matched_row_report]
            )
        if args.retained_row_report:
            command.extend(
                ["--retained-row-report", args.retained_row_report]
            )
        if args.training_row_panel_report:
            command.extend(
                [
                    "--training-row-panel-report",
                    args.training_row_panel_report,
                ]
            )
        if args.gradient_interference:
            command.append("--gradient-interference")
        if args.rows_per_objective != 1:
            command.extend(["--rows-per-objective", str(args.rows_per_objective)])
        if args.beam_width != 1:
            command.extend(["--beam-width", str(args.beam_width)])
        if args.branching_factor != 1:
            command.extend(
                ["--branching-factor", str(args.branching_factor)]
            )
        if args.online_transport_validator:
            command.append("--online-transport-validator")
        if args.evaluation_compute_dtype != "authoritative_fp32":
            command.extend(
                ["--evaluation-compute-dtype", args.evaluation_compute_dtype]
            )
        process = host_resource_safety.run_guarded(
            command,
            cwd=ROOT,
            policy=host_resource_safety.policy_from_mapping(
                candidate_evaluator.operation_specific_host_safety_mapping(
                    candidate_id=candidate_id,
                    candidate_contract=candidate_contract,
                    receipt_path=(
                        GRADIENT_INTERFERENCE_RESOURCE_RECEIPT
                        if args.gradient_interference
                        else PROBE_RESOURCE_RECEIPT
                    ),
                    receipt_sha256=(
                        GRADIENT_INTERFERENCE_RESOURCE_RECEIPT_SHA256
                        if args.gradient_interference
                        else PROBE_RESOURCE_RECEIPT_SHA256
                    ),
                    command_marker="kerc_k5_stage_learnability_probe.py",
                ),
                maximum_wall_seconds=float(candidate["max_wall_seconds"]),
            ),
            env={"THESEUS_GUARDED_ACCELERATOR_CHILD": "1"},
        )
        receipt_path = resolve(args.out).with_suffix(".host_resource_safety.json")
        write_json(receipt_path, process.receipt)
        if process.stdout:
            print(process.stdout[-4000:])
        if process.stderr:
            print(process.stderr[-4000:], file=sys.stderr)
        return 0 if process.receipt.get("passed") is True else 2
    result = execute(args)
    print(json.dumps({key: result[key] for key in ("qualification_state", "exact_match_count", "syntax_valid_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
