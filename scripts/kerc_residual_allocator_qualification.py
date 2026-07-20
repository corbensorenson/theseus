#!/usr/bin/env python3
"""Qualify the production KERC per-unit allocator on governed private rows.

This owner exercises the exact allocator embedded in the canonical KERC model.
It uses source-visible K2 features only, compares strong deterministic controls,
and treats missing semantic-panel evidence as inconclusive rather than as a KERC
failure.  Public benchmark payloads and answer text never enter this route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from moecot_language_arm_training import (
    build_plan,
    materialize_kerc_unit_allocator_row,
    pack_kerc_unit_allocator_batch,
)
from kerc_residual_allocator_evaluator import evaluate_allocator_predictions
from kerc_residual_interventions import TARGET_PRODUCER_ID
from standard_causal_transformer_model import CausalTransformerConfig, build_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "kerc_residual_allocator_qualification.json"
CORE_OBJECTIVE = "kernel_program_to_answer_packet_v1"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("policy") != "project_theseus_kerc_residual_allocator_qualification_v1":
        raise ValueError("KERC allocator qualification policy mismatch")
    if len(set(int(seed) for seed in config.get("seeds") or [])) < 3:
        raise ValueError("KERC allocator qualification requires at least three seeds")
    selection = config.get("selection") or {}
    expected_visible = [
        "source_record_sha256",
        "split",
        "unit_kind",
        "selected_fidelity_index",
    ]
    if (
        selection.get("selection_visible_fields") != expected_visible
        or selection.get("answer_text_visible") is not False
        or selection.get("model_outcomes_visible") is not False
    ):
        raise ValueError("KERC allocator selection boundary invalid")
    boundaries = config.get("boundaries") or {}
    zero_keys = (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
    )
    if any(int(boundaries.get(key) or 0) for key in zero_keys):
        raise ValueError("KERC allocator no-cheat boundary is nonzero")
    if boundaries.get("evaluator_effect_features_visible_to_model") is not False:
        raise ValueError("KERC allocator evaluator features must be hidden")
    return config


def _authoritative_labels(row: dict[str, Any]) -> list[int]:
    return [
        int(label)
        for label, authority in zip(row["labels"], row["loss_mask"])
        if float(authority) > 0.0
    ]


def load_split_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            view = json.loads(raw)
            if view.get("objective") != CORE_OBJECTIVE:
                continue
            if view.get("split") != split:
                raise ValueError(f"KERC allocator split mismatch at line {line_number}")
            if view.get("kerc_residual_unit_allocator_loss_enabled") is not True:
                if any(
                    bool(target.get("allocator_loss_enabled"))
                    for target in view.get("kerc_residual_unit_targets") or []
                ):
                    raise ValueError(
                        f"KERC allocator authority flag suppresses a target at line {line_number}"
                    )
                continue
            if view.get("generator_visible_fields") != [
                "trusted_source_prefix_tokens",
                "prompt",
            ]:
                raise ValueError(f"KERC allocator generator boundary invalid at line {line_number}")
            identity = str(view.get("source_record_sha256") or "")
            target_identity = canonical_sha256(view.get("kerc_residual_unit_targets") or [])
            if not identity:
                raise ValueError(f"KERC allocator record identity missing at line {line_number}")
            if identity in seen:
                if seen[identity] != target_identity:
                    raise ValueError(f"KERC allocator duplicate target mismatch: {identity}")
                continue
            materialized = materialize_kerc_unit_allocator_row(view)
            if materialized is None:
                raise ValueError(f"KERC allocator row did not materialize: {identity}")
            k2_labels = np.asarray(
                view.get("kerc_residual_unit_fidelity_labels") or [], dtype=np.int32
            )
            if tuple(k2_labels.shape) != tuple(materialized["labels"].shape):
                raise ValueError(f"KERC K2 label inventory mismatch: {identity}")
            materialized["source_record_sha256"] = identity
            materialized["source_group"] = str(view.get("source_group") or "")
            materialized["split"] = split
            materialized["k2_labels"] = k2_labels
            materialized["target_identity"] = target_identity
            rows.append(materialized)
            seen[identity] = target_identity
    if not rows:
        raise ValueError(f"KERC allocator split has no authoritative rows: {split}")
    return rows


def load_independent_eval_records(
    candidate_path: Path, rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    wanted = {str(row["source_record_sha256"]): row for row in rows}
    found: dict[str, dict[str, Any]] = {}
    with candidate_path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            record = json.loads(raw)
            packet = (record.get("kernel_packet") or {}).get("residual", {}).get(
                "unit_packet", {}
            )
            identity = str(record.get("record_sha256") or "")
            if identity in wanted:
                observed_units = tuple(
                    str(unit.get("unit_id") or "")
                    for unit in packet.get("units") or []
                )
                if observed_units != tuple(wanted[identity]["unit_ids"]):
                    raise ValueError(
                        "KERC independent evaluator unit inventory mismatch"
                    )
                if identity in found and canonical_sha256(found[identity]) != canonical_sha256(record):
                    raise ValueError("KERC independent evaluator record identity collision")
                found[identity] = record
    missing = set(wanted) - set(found)
    if missing:
        raise ValueError(f"KERC independent evaluator records missing: {len(missing)}")
    return found


def select_rows(
    rows: list[dict[str, Any]],
    *,
    maximum: int,
    minimum_per_class: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['source_record_sha256']}".encode()
        ).hexdigest(),
    )
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    counts: Counter[int] = Counter()
    for row in ordered:
        labels = set(_authoritative_labels(row))
        if any(counts[label] < minimum_per_class for label in labels):
            chosen.append(row)
            chosen_ids.add(str(row["source_record_sha256"]))
            counts.update(_authoritative_labels(row))
            if len(chosen) >= maximum:
                break
    for row in ordered:
        if len(chosen) >= maximum:
            break
        if str(row["source_record_sha256"]) not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(str(row["source_record_sha256"]))
            counts.update(_authoritative_labels(row))
    missing = [label for label in range(4) if counts[label] < minimum_per_class]
    # Class 0 may be intentionally withheld by the target producer.  It is not
    # fabricated merely to satisfy a balanced-looking receipt.
    missing_required = [label for label in missing if any(label in _authoritative_labels(row) for row in rows)]
    if missing_required:
        raise ValueError(f"KERC allocator selected split lacks classes: {missing_required}")
    return chosen, {
        "available_record_count": len(rows),
        "selected_record_count": len(chosen),
        "selected_authoritative_unit_count": sum(counts.values()),
        "selected_class_counts": {str(key): value for key, value in sorted(counts.items())},
        "selection_sha256": canonical_sha256(
            [row["source_record_sha256"] for row in chosen]
        ),
        "answer_text_visible": False,
        "model_outcomes_visible": False,
    }


def build_allocator_model(target: dict[str, Any], *, seed: int, mx: Any, nn: Any) -> Any:
    mx.random.seed(seed)
    vocab_size = int(target["vocab_size"])
    return build_model(
        CausalTransformerConfig(vocab_size=vocab_size, **target["model"]),
        mx=mx,
        nn=nn,
        source_to_target_lookup=np.arange(vocab_size, dtype=np.int32),
    )


def _mlx_batch(rows: list[dict[str, Any]], *, mx: Any) -> dict[str, Any]:
    packed = pack_kerc_unit_allocator_batch(rows)
    if packed is None:
        raise ValueError("KERC allocator batch is empty")
    return {
        key: mx.array(value)
        for key, value in packed.items()
    }


def allocator_loss(
    model: Any,
    batch: dict[str, Any],
    class_weights: Any,
    confidence_weight: float,
    *,
    mx: Any,
) -> Any:
    output = model.kerc_allocate_units(
        unit_byte_ids=batch["byte_ids"],
        unit_byte_mask=None,
        unit_byte_offsets=batch["byte_offsets"],
        unit_kind_ids=batch["kind_ids"],
        unit_candidate_features=batch["candidate_features"],
        unit_mask=batch["unit_mask"],
        unit_hard_block_mask=batch["hard_block_mask"],
        source_summary=None,
    )
    logits = output["logits"]
    labels = batch["labels"].astype(mx.int32)
    authority = batch["loss_mask"].astype(mx.float32)
    log_probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    selected = mx.take_along_axis(log_probabilities, labels[..., None], axis=-1)[..., 0]
    weights = mx.take(class_weights, labels) * authority
    classification = -mx.sum(selected * weights) / mx.maximum(mx.sum(weights), 1.0)
    confidence_logits = output["confidence_logits"]
    confidence_targets = batch["confidence_targets"].astype(mx.float32)
    confidence_loss = (
        mx.maximum(confidence_logits, 0.0)
        - confidence_logits * confidence_targets
        + mx.log1p(mx.exp(-mx.abs(confidence_logits)))
    )
    confidence_loss = mx.sum(confidence_loss * authority) / mx.maximum(
        mx.sum(authority), 1.0
    )
    return classification + float(confidence_weight) * confidence_loss


def class_weights(rows: Iterable[dict[str, Any]], maximum: float) -> np.ndarray:
    counts = Counter(label for row in rows for label in _authoritative_labels(row))
    active = [count for count in counts.values() if count]
    if len(active) < 2:
        raise ValueError("KERC allocator needs at least two authoritative classes")
    mean = sum(active) / len(active)
    return np.asarray(
        [min(maximum, mean / max(1, counts[index])) for index in range(4)],
        dtype=np.float32,
    )


def allocator_input_shape(
    rows: Iterable[dict[str, Any]], batch_records: int
) -> dict[str, Any]:
    rows = list(rows)
    byte_widths = [
        len(payload) for row in rows for payload in row["byte_rows"]
    ]
    unit_counts = [len(row["unit_ids"]) for row in rows]
    if not rows or not byte_widths:
        raise ValueError("KERC allocator input shape is empty")
    actual_bytes = sum(byte_widths)
    maximum_actual_bytes_per_record = max(
        sum(len(payload) for payload in row["byte_rows"]) for row in rows
    )
    maximum_units = max(unit_counts)
    maximum_bytes = max(byte_widths)
    rectangular_split_slots = len(rows) * maximum_units * maximum_bytes
    rectangular_batch_slots = int(batch_records) * maximum_units * maximum_bytes
    maximum_ragged_batch_slots = (
        int(batch_records) * maximum_actual_bytes_per_record
    )
    return {
        "record_count": len(rows),
        "unit_count": len(byte_widths),
        "actual_source_byte_count": actual_bytes,
        "maximum_units_per_record": maximum_units,
        "maximum_bytes_per_unit": maximum_bytes,
        "maximum_actual_bytes_per_record": maximum_actual_bytes_per_record,
        "naive_rectangular_byte_slots_over_selected_split": rectangular_split_slots,
        "naive_rectangular_byte_slots_at_configured_batch": rectangular_batch_slots,
        "maximum_ragged_byte_slots_at_configured_batch": maximum_ragged_batch_slots,
        "selected_split_ragged_to_naive_slot_ratio": round(
            actual_bytes / max(1, rectangular_split_slots), 8
        ),
        "maximum_batch_ragged_to_naive_slot_ratio": round(
            maximum_ragged_batch_slots / max(1, rectangular_batch_slots), 8
        ),
        "layout": "flat_exact_bytes_plus_per_unit_prefix_sum_offsets",
        "source_byte_truncation_count": 0,
    }


def _flatten_predictions(
    rows: list[dict[str, Any]],
    predictions: list[np.ndarray],
    confidences: list[np.ndarray],
) -> dict[str, Any]:
    labels: list[int] = []
    predicted: list[int] = []
    confidence_values: list[float] = []
    kinds: list[int] = []
    k2: list[int] = []
    contested: list[bool] = []
    hard_violations = 0
    for row, row_predictions, row_confidence in zip(rows, predictions, confidences):
        for index, authority in enumerate(row["loss_mask"]):
            if float(authority) <= 0.0:
                continue
            label = int(row["labels"][index])
            prediction = int(row_predictions[index])
            labels.append(label)
            predicted.append(prediction)
            confidence_values.append(float(row_confidence[index]))
            kinds.append(int(row["kind_ids"][index]))
            k2.append(int(row["k2_labels"][index]))
            contested.append(int((~row["hard_block_mask"][index]).sum()) > 1)
            hard_violations += int(bool(row["hard_block_mask"][index, prediction]))
    if not labels:
        raise ValueError("KERC allocator evaluation has no authoritative units")
    class_accuracy = {}
    for label in sorted(set(labels)):
        indices = [index for index, value in enumerate(labels) if value == label]
        class_accuracy[str(label)] = sum(predicted[index] == label for index in indices) / len(indices)
    accuracy = sum(left == right for left, right in zip(labels, predicted)) / len(labels)
    contested_indices = [index for index, value in enumerate(contested) if value]
    contested_accuracy = (
        sum(labels[index] == predicted[index] for index in contested_indices)
        / len(contested_indices)
        if contested_indices
        else 0.0
    )
    brier = sum(
        (confidence - float(left == right)) ** 2
        for left, right, confidence in zip(labels, predicted, confidence_values)
    ) / len(labels)
    return {
        "unit_count": len(labels),
        "accuracy": round(accuracy, 8),
        "contested_unit_count": len(contested_indices),
        "contested_accuracy": round(contested_accuracy, 8),
        "single_admissible_unit_count": len(labels) - len(contested_indices),
        "macro_class_accuracy": round(sum(class_accuracy.values()) / len(class_accuracy), 8),
        "class_accuracy": {key: round(value, 8) for key, value in class_accuracy.items()},
        "hard_violation_count": hard_violations,
        "confidence_brier": round(brier, 8),
        "labels": labels,
        "predictions": predicted,
        "kinds": kinds,
        "k2_labels": k2,
    }


def evaluate_model(model: Any, rows: list[dict[str, Any]], *, batch_size: int, mx: Any) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        batch = _mlx_batch(selected, mx=mx)
        output = model.kerc_allocate_units(
            unit_byte_ids=batch["byte_ids"],
            unit_byte_mask=None,
            unit_byte_offsets=batch["byte_offsets"],
            unit_kind_ids=batch["kind_ids"],
            unit_candidate_features=batch["candidate_features"],
            unit_mask=batch["unit_mask"],
            unit_hard_block_mask=batch["hard_block_mask"],
            source_summary=None,
        )
        probability = 1.0 / (1.0 + mx.exp(-output["confidence_logits"]))
        mx.eval(output["logits"], probability)
        logits = np.asarray(output["logits"])
        confidence = np.asarray(probability)
        for row_index, row in enumerate(selected):
            unit_count = len(row["unit_ids"])
            predictions.append(np.argmax(logits[row_index, :unit_count], axis=-1))
            confidences.append(confidence[row_index, :unit_count])
    return _flatten_predictions(rows, predictions, confidences)


def evaluation_logits(
    model: Any, rows: list[dict[str, Any]], *, batch_size: int, mx: Any
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        batch = _mlx_batch(selected, mx=mx)
        output = model.kerc_allocate_units(
            unit_byte_ids=batch["byte_ids"],
            unit_byte_mask=None,
            unit_byte_offsets=batch["byte_offsets"],
            unit_kind_ids=batch["kind_ids"],
            unit_candidate_features=batch["candidate_features"],
            unit_mask=batch["unit_mask"],
            unit_hard_block_mask=batch["hard_block_mask"],
            source_summary=None,
        )
        mx.eval(output["logits"])
        logits = np.asarray(output["logits"])
        for row_index, row in enumerate(selected):
            values.append(logits[row_index, : len(row["unit_ids"])])
    return np.concatenate(values, axis=0)


def independent_panel(
    model: Any,
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    *,
    batch_size: int,
    mx: Any,
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        batch = _mlx_batch(selected, mx=mx)
        output = model.kerc_allocate_units(
            unit_byte_ids=batch["byte_ids"],
            unit_byte_mask=None,
            unit_byte_offsets=batch["byte_offsets"],
            unit_kind_ids=batch["kind_ids"],
            unit_candidate_features=batch["candidate_features"],
            unit_mask=batch["unit_mask"],
            unit_hard_block_mask=batch["hard_block_mask"],
            source_summary=None,
        )
        probability = 1.0 / (1.0 + mx.exp(-output["confidence_logits"]))
        mx.eval(output["logits"], probability)
        logits = np.asarray(output["logits"])
        confidence = np.asarray(probability)
        for row_index, row in enumerate(selected):
            count = len(row["unit_ids"])
            predictions.append(np.argmax(logits[row_index, :count], axis=-1))
            confidences.append(confidence[row_index, :count])
    totals: Counter[str] = Counter()
    kinds: dict[str, Counter[str]] = defaultdict(Counter)
    receipt_ids = []
    for row, selected, confidence in zip(rows, predictions, confidences):
        identity = str(row["source_record_sha256"])
        record = records[identity]
        packet = record["kernel_packet"]
        residual = packet["residual"]
        receipt = evaluate_allocator_predictions(
            unit_packet=residual["unit_packet"],
            source_record_sha256=residual["unit_packet"]["source_record_sha256"],
            global_state=(record["hrl_state"].get("global") or {}),
            segment_residual=(residual.get("segment_frame") or {}),
            token_residuals=list(residual.get("token_tags") or []),
            concept_capsules=(packet.get("concept_capsules") or {}),
            exact_objects=(packet.get("protected_objects") or {}),
            source_family=str(record["provenance"]["source_group"]),
            predictions=[
                {
                    "unit_id": unit_id,
                    "selected_fidelity_index": int(action),
                    "confidence": float(probability),
                }
                for unit_id, action, probability in zip(
                    row["unit_ids"], selected, confidence
                )
            ],
            training_target_producer_id=TARGET_PRODUCER_ID,
        )
        summary = receipt["summary"]
        totals["record_count"] += 1
        totals["unit_count"] += int(summary["unit_count"])
        totals["hard_violation_count"] += int(summary["hard_violation_count"])
        totals["selected_encoded_bits"] += int(summary["selected_encoded_bits"])
        totals["independent_oracle_encoded_bits"] += int(
            summary["independent_oracle_encoded_bits"]
        )
        totals["rate_regret_bits"] += int(summary["rate_regret_bits"])
        for kind, counts in summary["by_unit_kind"].items():
            kinds[kind].update(counts)
        receipt_ids.append(receipt["receipt_sha256"])
    return {
        **dict(totals),
        "by_unit_kind": {kind: dict(counts) for kind, counts in sorted(kinds.items())},
        "receipt_ledger_sha256": canonical_sha256(receipt_ids),
        "target_producer_is_final_evaluator": False,
        "public_or_hidden_target_used": False,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def ablated_rows(
    rows: list[dict[str, Any]], *, mode: str, seed: int
) -> list[dict[str, Any]]:
    if mode not in {"content_shuffled", "candidate_features_zeroed", "unit_kind_zeroed"}:
        raise ValueError(f"unknown KERC allocator ablation: {mode}")
    result = []
    payloads = [row["byte_rows"] for row in rows]
    if mode == "content_shuffled":
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        if len(order) > 1 and all(index == value for index, value in enumerate(order)):
            order = order[1:] + order[:1]
    else:
        order = list(range(len(rows)))
    for index, row in enumerate(rows):
        copy_row = dict(row)
        if mode == "content_shuffled":
            donor = payloads[order[index]]
            if len(donor) == len(row["byte_rows"]):
                copy_row["byte_rows"] = tuple(
                    np.asarray(value, dtype=np.int32) for value in donor
                )
            else:
                copy_row["byte_rows"] = tuple(
                    np.full_like(value, 63, dtype=np.int32)
                    for value in row["byte_rows"]
                )
        elif mode == "candidate_features_zeroed":
            copy_row["candidate_features"] = np.zeros_like(
                row["candidate_features"], dtype=np.float32
            )
        else:
            copy_row["kind_ids"] = np.zeros_like(row["kind_ids"], dtype=np.int32)
        result.append(copy_row)
    return result


def baseline_metrics(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_kind: dict[int, Counter[int]] = defaultdict(Counter)
    for row in train_rows:
        for kind, label, authority in zip(row["kind_ids"], row["labels"], row["loss_mask"]):
            if float(authority) > 0.0:
                by_kind[int(kind)][int(label)] += 1
    majority_predictions: list[np.ndarray] = []
    rate_predictions: list[np.ndarray] = []
    k2_predictions: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    for row in eval_rows:
        majority = []
        rate = []
        for index, kind in enumerate(row["kind_ids"]):
            allowed = np.flatnonzero(~row["hard_block_mask"][index])
            if not len(allowed):
                raise ValueError("KERC allocator baseline has no admissible action")
            ranked = [label for label, _count in by_kind[int(kind)].most_common()]
            majority.append(next((label for label in ranked if label in allowed), int(allowed[-1])))
            encoded_rate = row["candidate_features"][index, :, 0]
            rate.append(int(allowed[np.argmin(encoded_rate[allowed])]))
        majority_predictions.append(np.asarray(majority, dtype=np.int32))
        rate_predictions.append(np.asarray(rate, dtype=np.int32))
        k2_predictions.append(np.asarray(row["k2_labels"], dtype=np.int32))
        confidence.append(np.ones(len(row["unit_ids"]), dtype=np.float32))
    return {
        "presence_by_kind": _flatten_predictions(eval_rows, majority_predictions, confidence),
        "source_visible_constrained_rate": _flatten_predictions(eval_rows, rate_predictions, confidence),
        "k2_structural_selection": _flatten_predictions(eval_rows, k2_predictions, confidence),
    }


def gradient_norm(gradients: Any, *, mx_utils: Any, mx: Any) -> float:
    total = 0.0
    for _name, value in mx_utils.tree_flatten(gradients):
        if value is None:
            continue
        mx.eval(value)
        total += float(mx.sum(value.astype(mx.float32) ** 2).item())
    return math.sqrt(total)


def train_seed(
    *,
    target: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    seed: int,
    config: dict[str, Any],
    checkpoint_path: Path | None,
    maximum_steps: int | None = None,
    independent_records: dict[str, dict[str, Any]] | None = None,
    mx: Any,
    nn: Any,
    optim: Any,
    mx_utils: Any,
) -> dict[str, Any]:
    optimization = config["optimization"]
    batch_size = int(optimization["batch_records"])
    model = build_allocator_model(target, seed=seed, mx=mx, nn=nn)
    optimizer = optim.AdamW(
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    weights = mx.array(
        class_weights(train_rows, float(optimization["class_balance_maximum"])),
        dtype=mx.float32,
    )

    def loss_fn(candidate: Any, batch: dict[str, Any], class_weight_values: Any) -> Any:
        return allocator_loss(
            candidate,
            batch,
            class_weight_values,
            float(optimization["confidence_loss_weight"]),
            mx=mx,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    order = list(range(len(train_rows)))
    rng = random.Random(seed)
    curve = []
    first_gradient_norm = 0.0
    first_applied_gradient_norm = 0.0
    started = time.perf_counter()
    model.train()
    step_cap = int(maximum_steps or optimization["maximum_steps"])
    for step in range(1, step_cap + 1):
        if (step - 1) % max(1, math.ceil(len(order) / batch_size)) == 0:
            rng.shuffle(order)
        start = ((step - 1) * batch_size) % len(order)
        indices = (order + order)[start : start + batch_size]
        batch = _mlx_batch([train_rows[index] for index in indices], mx=mx)
        loss, gradients = loss_and_grad(model, batch, weights)
        raw_gradient_norm = (
            gradient_norm(gradients, mx_utils=mx_utils, mx=mx)
            if step == 1
            else 0.0
        )
        gradients, _reported_gradient_norm = optim.clip_grad_norm(
            gradients, float(optimization["gradient_clip_norm"])
        )
        if step == 1:
            first_gradient_norm = raw_gradient_norm
            first_applied_gradient_norm = min(
                raw_gradient_norm, float(optimization["gradient_clip_norm"])
            )
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 1 or step % int(optimization["evaluation_every_steps"]) == 0:
            metric = evaluate_model(model, eval_rows, batch_size=batch_size, mx=mx)
            curve.append({"step": step, "loss": float(loss.item()), "eval_accuracy": metric["accuracy"]})
            model.train()
    evaluation = evaluate_model(model, eval_rows, batch_size=batch_size, mx=mx)
    clean_predictions = list(evaluation["predictions"])
    interventions = {}
    for mode in ("content_shuffled", "candidate_features_zeroed", "unit_kind_zeroed"):
        ablated = evaluate_model(
            model,
            ablated_rows(eval_rows, mode=mode, seed=seed),
            batch_size=batch_size,
            mx=mx,
        )
        interventions[mode] = {
            "accuracy": ablated["accuracy"],
            "prediction_change_count": sum(
                left != right
                for left, right in zip(clean_predictions, ablated["predictions"])
            ),
        }
    panel = (
        independent_panel(
            model,
            eval_rows,
            independent_records,
            batch_size=batch_size,
            mx=mx,
        )
        if independent_records is not None
        else None
    )
    reload_delta = None
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(checkpoint_path))
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        }
        fresh = build_allocator_model(target, seed=seed + 100000, mx=mx, nn=nn)
        fresh.load_weights(str(checkpoint_path), strict=True)
        replay_rows = eval_rows[: min(16, len(eval_rows))]
        left = evaluation_logits(model, replay_rows, batch_size=batch_size, mx=mx)
        right = evaluation_logits(fresh, replay_rows, batch_size=batch_size, mx=mx)
        reload_delta = float(np.max(np.abs(left - right)))
    return {
        "seed": seed,
        "steps": step_cap,
        "first_gradient_norm": first_gradient_norm,
        "first_applied_gradient_norm": first_applied_gradient_norm,
        "gradient_clip_norm": float(optimization["gradient_clip_norm"]),
        "curve": curve,
        "evaluation": {key: value for key, value in evaluation.items() if key not in {"labels", "predictions", "kinds", "k2_labels"}},
        "interventions": interventions,
        "independent_typed_panel": panel,
        "reload_maximum_logit_delta": reload_delta,
        "checkpoint": checkpoint,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    training_path = resolve(config["training_config"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    plan = build_plan(training, config_path=training_path)
    if plan.get("trigger_state") != "GREEN":
        raise ValueError("canonical MoECOT plan is not GREEN")
    target = plan["targets"][config["target_id"]]
    stage_root = resolve(config["stage_root"])
    stage_manifest_path = stage_root / "manifest.json"
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    intervention_summary = (
        (stage_manifest.get("materialization_execution") or {}).get(
            "unit_intervention_targets"
        )
        or {}
    )
    if (
        stage_manifest.get("trigger_state") != "GREEN"
        or stage_manifest.get("hard_gaps")
        or intervention_summary.get("target_producer") != TARGET_PRODUCER_ID
        or intervention_summary.get("target_producer_is_final_evaluator") is not False
        or intervention_summary.get("public_or_hidden_target_used") is not False
        or int(intervention_summary.get("external_inference_calls") or 0) != 0
        or int(intervention_summary.get("fallback_return_count") or 0) != 0
        or int(intervention_summary.get("allocator_authority_unit_count") or 0) <= 0
    ):
        raise ValueError("canonical KERC allocator stage evidence is not qualified")
    selection = config["selection"]
    rows = {
        split: load_split_rows(stage_root / f"{split}.jsonl", split)
        for split in ("private_train", "private_dev", "private_eval")
    }
    selected = {}
    selection_receipts = {}
    for split, maximum_key in (
        ("private_train", "maximum_train_records"),
        ("private_dev", "maximum_dev_records"),
        ("private_eval", "maximum_eval_records"),
    ):
        selected[split], selection_receipts[split] = select_rows(
            rows[split],
            maximum=int(selection[maximum_key]),
            minimum_per_class=int(selection["minimum_authoritative_units_per_class"]),
            seed=int(config["seeds"][0]),
        )
    split_identities = {
        split: {str(row["source_record_sha256"]) for row in split_rows}
        for split, split_rows in selected.items()
    }
    overlap = sum(
        len(split_identities[left] & split_identities[right])
        for left, right in (
            ("private_train", "private_dev"),
            ("private_train", "private_eval"),
            ("private_dev", "private_eval"),
        )
    )
    if overlap:
        raise ValueError(f"KERC allocator source split overlap: {overlap}")
    baselines = baseline_metrics(selected["private_train"], selected["private_eval"])
    independent_record_path = resolve(stage_manifest["source"]["path"])
    if file_sha256(independent_record_path) != stage_manifest["source"]["sha256"]:
        raise ValueError("KERC independent evaluator source identity mismatch")
    independent_records = load_independent_eval_records(
        independent_record_path, selected["private_eval"]
    )

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mx_utils

    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    overfit_count = min(
        int(config["optimization"]["overfit_records"]),
        len(selected["private_train"]),
    )
    overfit_rows, overfit_selection = select_rows(
        selected["private_train"],
        maximum=overfit_count,
        minimum_per_class=2,
        seed=int(config["seeds"][0]) + 50000,
    )
    overfit = train_seed(
        target=target,
        train_rows=overfit_rows,
        eval_rows=overfit_rows,
        seed=int(config["seeds"][0]) + 50000,
        config=config,
        checkpoint_path=None,
        maximum_steps=int(config["optimization"]["overfit_maximum_steps"]),
        independent_records=None,
        mx=mx,
        nn=nn,
        optim=optim,
        mx_utils=mx_utils,
    )
    for index, seed in enumerate(config["seeds"]):
        seed_reports.append(
            train_seed(
                target=target,
                train_rows=selected["private_train"],
                eval_rows=selected["private_eval"],
                seed=int(seed),
                config=config,
                checkpoint_path=(output_root / "allocator_weights.safetensors") if index == 0 else None,
                maximum_steps=None,
                independent_records=independent_records if index == 0 else None,
                mx=mx,
                nn=nn,
                optim=optim,
                mx_utils=mx_utils,
            )
        )
    acceptance = config["acceptance"]
    accuracies = [float(row["evaluation"]["accuracy"]) for row in seed_reports]
    contested_accuracies = [
        float(row["evaluation"]["contested_accuracy"]) for row in seed_reports
    ]
    mechanics_checks = {
        "seed_count": len(seed_reports) >= int(acceptance["minimum_seed_count"]),
        "nonzero_gradient": all(
            float(row["first_gradient_norm"])
            >= float(acceptance["minimum_nonzero_gradient_norm"])
            for row in seed_reports
        ),
        "hard_constraints": all(
            int(row["evaluation"]["hard_violation_count"])
            <= int(acceptance["maximum_hard_violation_count"])
            for row in seed_reports
        ),
        "independent_typed_panel": int(
            (seed_reports[0].get("independent_typed_panel") or {}).get(
                "hard_violation_count", -1
            )
        )
        <= int(acceptance["maximum_hard_violation_count"]),
        "checkpoint_reload": float(
            seed_reports[0]["reload_maximum_logit_delta"] or 0.0
        )
        <= float(acceptance["maximum_reload_logit_delta"]),
        "representative_subset_overfit": float(
            overfit["evaluation"]["contested_accuracy"]
        )
        >= float(acceptance["minimum_overfit_accuracy"]),
        "overfit_nonzero_gradient": float(overfit["first_gradient_norm"])
        >= float(acceptance["minimum_nonzero_gradient_norm"]),
        "source_disjoint": overlap == 0,
        "eval_class_count": len(
            selection_receipts["private_eval"]["selected_class_counts"]
        )
        >= int(acceptance["minimum_source_disjoint_eval_class_count"]),
    }
    presence = float(baselines["presence_by_kind"]["contested_accuracy"])
    learned_margin = min(contested_accuracies) - presence
    learned_generalization = learned_margin >= float(
        acceptance["minimum_presence_baseline_margin"]
    )
    constrained_rate = float(baselines["source_visible_constrained_rate"]["accuracy"])
    signal_diagnostics = {
        mode: {
            "all_seeds_change_predictions": all(
                int(row["interventions"][mode]["prediction_change_count"]) > 0
                for row in seed_reports
            ),
            "total_prediction_change_count": sum(
                int(row["interventions"][mode]["prediction_change_count"])
                for row in seed_reports
            ),
        }
        for mode in (
            "content_shuffled",
            "candidate_features_zeroed",
            "unit_kind_zeroed",
        )
    }
    independent_panel = seed_reports[0].get("independent_typed_panel") or {}
    causal_adequacy_checks = {
        "nontrivial_source_signal_use": all(
            any(
                int(row["interventions"][mode]["prediction_change_count"]) > 0
                for mode in signal_diagnostics
            )
            for row in seed_reports
        ),
        "deterministic_constraint_engine_not_target_oracle": constrained_rate < 1.0,
        "independent_evaluator_zero_rate_regret": int(
            independent_panel.get("rate_regret_bits") or 0
        )
        == 0,
        "independent_semantic_panel_complete": False,
    }
    report = {
        "policy": config["policy"],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "training_config_sha256": file_sha256(training_path),
        "stage_manifest": {
            "path": str(stage_manifest_path),
            "sha256": file_sha256(stage_manifest_path),
            "contract_sha256": stage_manifest["contract_sha256"],
            "unit_intervention_targets": intervention_summary,
        },
        "independent_evaluator_source": {
            "path": str(independent_record_path),
            "sha256": file_sha256(independent_record_path),
            "same_module_as_target_producer": False,
        },
        "target_id": config["target_id"],
        "model_config_sha256": canonical_sha256(target["model"]),
        "selection": selection_receipts,
        "allocator_input_shape": {
            split: allocator_input_shape(
                split_rows, int(config["optimization"]["batch_records"])
            )
            for split, split_rows in selected.items()
        },
        "source_split_overlap_count": overlap,
        "baselines": {
            name: {key: value for key, value in metric.items() if key not in {"labels", "predictions", "kinds", "k2_labels"}}
            for name, metric in baselines.items()
        },
        "overfit": {**overfit, "selection": overfit_selection},
        "seeds": seed_reports,
        "aggregate": {
            "minimum_eval_accuracy": min(accuracies),
            "mean_eval_accuracy": sum(accuracies) / len(accuracies),
            "maximum_eval_accuracy": max(accuracies),
            "minimum_contested_eval_accuracy": min(contested_accuracies),
            "mean_contested_eval_accuracy": sum(contested_accuracies)
            / len(contested_accuracies),
            "minimum_presence_baseline_margin": learned_margin,
            "constrained_rate_baseline_accuracy": constrained_rate,
        },
        "mechanics_checks": mechanics_checks,
        "mechanics_trigger_state": "GREEN" if all(mechanics_checks.values()) else "RED",
        "causal_signal_diagnostics": signal_diagnostics,
        "causal_adequacy_checks": causal_adequacy_checks,
        "causal_adequacy_trigger_state": (
            "GREEN" if all(causal_adequacy_checks.values()) else "RED"
        ),
        "learned_generalization_above_presence": learned_generalization,
        "learned_allocator_needed_beyond_constraint_engine": max(accuracies) > constrained_rate,
        "semantic_panel_complete": False,
        "canonical_long_training_authorized": False,
        "learned_allocator_claimed": False,
        "trigger_state": "YELLOW" if all(mechanics_checks.values()) else "RED",
        "disposition": (
            "INCONCLUSIVE_EXPERIMENT_DEGENERATE_TARGET_PENDING_SEMANTIC_PANEL"
            if all(mechanics_checks.values())
            else "INCONCLUSIVE_IMPLEMENTATION_REPAIR_OWNER"
        ),
        "recommendation": (
            "Retain the deterministic hard-constraint/rate engine; do not give the "
            "neural allocator long-training authority until source-dependent soft "
            "semantic consequences and an independent human/executable evaluator "
            "panel produce nondegenerate decisions."
        ),
        "utility_claimed": False,
        "scientific_negative_claimed": False,
        "retirement_authority": False,
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "evaluator_effect_features_visible_to_model": False,
    }
    report["receipt_sha256"] = canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = run(resolve(args.config))
    out = resolve(args.out) if args.out else resolve(
        json.loads(resolve(args.config).read_text(encoding="utf-8"))["report"]
    )
    write_json_atomic(out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["mechanics_trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
