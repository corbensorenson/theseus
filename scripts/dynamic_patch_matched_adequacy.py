#!/usr/bin/env python3
"""Matched, source-disjoint surface-codec adequacy for the T0A architecture docket.

The experiment compares the canonical open vocabulary with raw-byte, fixed,
visible-boundary, prefix-entropy, and learned prefix-boundary byte candidates.
All byte candidates share topology and initialization. Work matching is based on
frozen parameter-position accounting; raw-byte exposure is reported separately.
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
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import pretraining_candidate_canary
from dynamic_byte_patch_codec import (
    boundaries_from_probabilities,
    build_dynamic_patch_causal_candidate,
    patch_ids_from_boundaries,
)
from moecot_language_tokenizer import exact_text_tokens
from neural_seed_open_vocab import encode_tokens
from standard_causal_transformer_model import CausalTransformerConfig, build_model
from standard_causal_transformer_survival import (
    GLOBAL_BOS_ID,
    SOURCE_TARGET_SEPARATOR_ID,
    model_vocab_size,
    source_token_offset,
    target_token_offset,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dynamic_patch_matched_adequacy.json"
POLICY = "project_theseus_dynamic_patch_matched_adequacy_v1"
BYTE_FRAME_PREFIX = b"<|user|>\n"
BYTE_FRAME_SEPARATOR = b"\n<|assistant|>\n"


class DynamicPatchAdequacyFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy") != POLICY:
        raise DynamicPatchAdequacyFault("config_policy_invalid")
    expected = {
        "fixed_open_vocab_ar",
        "raw_byte_ar",
        "fixed_byte_patch_ar",
        "entropy_byte_patch_ar",
        "whitespace_byte_patch_control",
        "learned_dynamic_byte_patch_ar",
    }
    candidates = config.get("candidates") or []
    if {row.get("id") for row in candidates} != expected:
        raise DynamicPatchAdequacyFault("candidate_inventory_invalid")
    if tuple(config.get("scoped_arms") or ()) != (
        "english",
        "python",
        "javascript_typescript",
        "html_css",
        "rust",
    ):
        raise DynamicPatchAdequacyFault("scoped_arm_contract_invalid")
    boundaries = config.get("hard_boundaries") or {}
    for field in (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "fallback_or_template_credit",
        "confirmation_surface_consumption",
    ):
        if boundaries.get(field) != 0:
            raise DynamicPatchAdequacyFault(f"hard_boundary_nonzero:{field}")
    for field in (
        "production_checkpoint_mutation",
        "heldout_labels_visible_to_optimizer",
        "selection_from_compression_or_loss_alone",
    ):
        if boundaries.get(field) is not False:
            raise DynamicPatchAdequacyFault(f"hard_boundary_boolean_invalid:{field}")
    if int(config["training"]["steps"]) > 128:
        raise DynamicPatchAdequacyFault("candidate_step_budget_exceeded")
    return config


def _rank(identity: str, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}:{identity}".encode()).hexdigest()


def load_governed_rows(config: dict[str, Any], split: str) -> dict[str, list[dict[str, Any]]]:
    supervision = config["supervision"]
    expected_split = supervision["train_split"] if split == "train" else supervision["heldout_split"]
    limit = int(supervision["train_rows_per_arm"] if split == "train" else supervision["heldout_rows_per_arm"])
    maximum_bytes = int(supervision["maximum_raw_bytes"])
    maximum_tokens = int(supervision["maximum_open_vocab_tokens"])
    metadata = json.loads(resolve(config["stage_metadata"]).read_text(encoding="utf-8"))
    base = json.loads(resolve(config["base_config"]).read_text(encoding="utf-8"))
    source_vocab = metadata["source_vocab"]
    target_vocab = metadata["target_vocab"]
    source_offset = source_token_offset(base, source_vocab)
    target_offset = target_token_offset(base, source_vocab)
    selected: dict[str, list[dict[str, Any]]] = {}
    root = resolve(supervision["root"])
    for arm in config["scoped_arms"]:
        eligible = []
        path = root / expected_split / f"{arm}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if (
                    row.get("split") != expected_split
                    or row.get("arm_id") != arm
                    or row.get("public_benchmark") is not False
                    or row.get("public_benchmark_payload") is not False
                    or int(row.get("external_inference_calls") or 0)
                    or int(row.get("fallback_return_count") or 0)
                ):
                    raise DynamicPatchAdequacyFault(
                        f"governed_row_boundary_invalid:{arm}:{expected_split}:{line_number}"
                    )
                if not row.get("license_spdx") or not row.get("source_identity"):
                    raise DynamicPatchAdequacyFault(
                        f"governed_row_provenance_missing:{arm}:{expected_split}:{line_number}"
                    )
                prompt = str(row["prompt"])
                target = str(row["target"])
                prompt_bytes = prompt.encode("utf-8")
                target_bytes = target.encode("utf-8")
                payload = BYTE_FRAME_PREFIX + prompt_bytes + BYTE_FRAME_SEPARATOR + target_bytes
                target_byte_start = len(BYTE_FRAME_PREFIX) + len(prompt_bytes) + len(BYTE_FRAME_SEPARATOR)
                source_ids, source_receipt = encode_tokens(exact_text_tokens(prompt), source_vocab, stream="source")
                target_ids, target_receipt = encode_tokens(exact_text_tokens(target), target_vocab, stream="target")
                if source_receipt["unknown_token_count"] or target_receipt["unknown_token_count"]:
                    raise DynamicPatchAdequacyFault(
                        f"canonical_tokenization_unknown:{arm}:{expected_split}:{line_number}"
                    )
                sequence = [GLOBAL_BOS_ID]
                sequence.extend(source_offset + int(value) for value in source_ids)
                sequence.append(SOURCE_TARGET_SEPARATOR_ID)
                sequence.append(target_offset + int(target_vocab["<bos>"]))
                target_token_start = len(sequence)
                sequence.extend(target_offset + int(value) for value in target_ids)
                sequence.append(target_offset + int(target_vocab["<eos>"]))
                if len(payload) > maximum_bytes or len(sequence) - 1 > maximum_tokens:
                    continue
                eligible.append(
                    {
                        "arm_id": arm,
                        "row_id": str(row["row_id"]),
                        "source_identity": str(row["source_identity"]),
                        "dataset_id": str(row["dataset_id"]),
                        "license_spdx": str(row["license_spdx"]),
                        "payload": payload,
                        "target_byte_start": target_byte_start,
                        "target_byte_count": len(target_bytes),
                        "token_sequence": sequence,
                        "target_token_mask_start": target_token_start - 1,
                    }
                )
        namespace = f"{supervision['selection_namespace']}:{expected_split}:{arm}"
        eligible.sort(key=lambda row: _rank(row["source_identity"], namespace))
        if len(eligible) < limit:
            raise DynamicPatchAdequacyFault(
                f"source_disjoint_rows_insufficient:{arm}:{expected_split}:{len(eligible)}<{limit}"
            )
        selected[arm] = eligible[:limit]
    return selected


def source_disjoint_receipt(
    train: dict[str, list[dict[str, Any]]], heldout: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    train_ids = {row["source_identity"] for rows in train.values() for row in rows}
    heldout_ids = {row["source_identity"] for rows in heldout.values() for row in rows}
    overlap = train_ids & heldout_ids
    return {
        "train_source_count": len(train_ids),
        "heldout_source_count": len(heldout_ids),
        "cross_split_overlap_count": len(overlap),
        "by_arm": {
            arm: {
                "train_rows": len(train[arm]),
                "heldout_rows": len(heldout[arm]),
                "source_identity_overlap": len(
                    {row["source_identity"] for row in train[arm]}
                    & {row["source_identity"] for row in heldout[arm]}
                ),
            }
            for arm in train
        },
        "selection_digest": digest(
            {"train": sorted(train_ids), "heldout": sorted(heldout_ids)}
        ),
        "passed": not overlap,
    }


def fit_prefix_entropy(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    context = np.zeros(257, dtype=np.int64)
    pairs = np.zeros((257, 256), dtype=np.int64)
    for row in rows:
        previous = 256
        for value in row["payload"]:
            context[previous] += 1
            pairs[previous, value] += 1
            previous = value
    return {"context": context, "pairs": pairs}


def prefix_uncertainty(payload: bytes, entropy_model: dict[str, Any]) -> list[float]:
    context = entropy_model["context"]
    pairs = entropy_model["pairs"]
    previous = 256
    values = []
    for value in payload:
        probability = (float(pairs[previous, value]) + 1.0) / (float(context[previous]) + 256.0)
        values.append(min(1.0, max(0.0, -math.log(probability) / math.log(256.0))))
        previous = value
    return values


def boundary_targets(
    payload: bytes,
    source: str,
    *,
    uncertainty: list[float],
    fixed_width: int,
    entropy_threshold: float,
) -> list[float]:
    if source == "every_byte":
        return [1.0] * len(payload)
    if source == "fixed_width":
        return [1.0 if (index + 1) % fixed_width == 0 or index + 1 == len(payload) else 0.0 for index in range(len(payload))]
    if source in {"prefix_entropy", "learned_prefix_entropy_prediction"}:
        return [1.0 if value >= entropy_threshold or index + 1 == len(payload) else 0.0 for index, value in enumerate(uncertainty)]
    if source == "visible_whitespace_and_code_punctuation":
        punctuation = set(b" \t\r\n()[]{}:;,.+-=*/<>\"'`_")
        return [1.0 if value in punctuation or index + 1 == len(payload) else 0.0 for index, value in enumerate(payload)]
    raise DynamicPatchAdequacyFault(f"boundary_source_invalid:{source}")


def points_for_targets(targets: list[float], maximum_patch_bytes: int) -> tuple[int, ...]:
    return boundaries_from_probabilities(
        targets,
        threshold=0.5,
        max_patch_bytes=maximum_patch_bytes,
    )


def patch_inputs(
    rows: list[dict[str, Any]],
    source: str,
    entropy_model: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    model = config["model"]
    width = max(len(row["payload"]) for row in rows)
    ids = np.zeros((len(rows), width), dtype=np.int32)
    mask = np.zeros((len(rows), width), dtype=np.float32)
    target_mask = np.zeros((len(rows), width), dtype=np.float32)
    boundary_y = np.zeros((len(rows), width), dtype=np.float32)
    uncertainty_array = np.zeros((len(rows), width), dtype=np.float32)
    patch_ids = np.zeros((len(rows), width), dtype=np.int32)
    within = np.zeros((len(rows), width), dtype=np.int32)
    patch_counts = []
    for index, row in enumerate(rows):
        payload = row["payload"]
        length = len(payload)
        uncertainty = prefix_uncertainty(payload, entropy_model)
        targets = boundary_targets(
            payload,
            source,
            uncertainty=uncertainty,
            fixed_width=int(model["fixed_patch_bytes"]),
            entropy_threshold=float(model["entropy_boundary_threshold"]),
        )
        points = points_for_targets(targets, int(model["maximum_patch_bytes"]))
        assignments = patch_ids_from_boundaries(points, length)
        positions: list[int] = []
        for start, stop in zip(points, points[1:]):
            positions.extend(range(stop - start))
        ids[index, :length] = np.frombuffer(payload, dtype=np.uint8)
        mask[index, :length] = 1.0
        target_mask[index, int(row["target_byte_start"]):length] = 1.0
        boundary_y[index, :length] = np.asarray(targets, dtype=np.float32)
        uncertainty_array[index, :length] = np.asarray(uncertainty, dtype=np.float32)
        patch_ids[index, :length] = np.asarray(assignments, dtype=np.int32)
        within[index, :length] = np.asarray(positions, dtype=np.int32)
        patch_counts.append(len(points) - 1)
    return {
        "ids": ids,
        "mask": mask,
        "target_mask": target_mask,
        "boundary_targets": boundary_y,
        "uncertainty": uncertainty_array,
        "patch_ids": patch_ids,
        "within": within,
        "patch_counts": np.asarray(patch_counts, dtype=np.int32),
    }


def open_vocab_batch(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(len(row["token_sequence"]) - 1 for row in rows)
    x = np.zeros((len(rows), width), dtype=np.int32)
    y = np.zeros((len(rows), width), dtype=np.int32)
    mask = np.zeros((len(rows), width), dtype=np.float32)
    for index, row in enumerate(rows):
        sequence = np.asarray(row["token_sequence"], dtype=np.int32)
        length = len(sequence) - 1
        x[index, :length] = sequence[:-1]
        y[index, :length] = sequence[1:]
        mask[index, int(row["target_token_mask_start"]):length] = 1.0
    return x, y, mask


def balanced_step_rows(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    steps: int,
    seed: int,
    batch_schedule: list[int],
) -> list[list[dict[str, Any]]]:
    if len(batch_schedule) != steps or any(value <= 0 for value in batch_schedule):
        raise DynamicPatchAdequacyFault("batch_schedule_invalid")
    rng = np.random.default_rng(seed)
    orders = {arm: rng.permutation(len(rows)).tolist() for arm, rows in rows_by_arm.items()}
    cursors = {arm: 0 for arm in rows_by_arm}
    arms = list(rows_by_arm)
    batches = []
    for step in range(steps):
        arm = arms[step % len(arms)]
        rows = rows_by_arm[arm]
        order = orders[arm]
        batch = []
        for _ in range(batch_schedule[step]):
            cursor = cursors[arm]
            if cursor and cursor % len(order) == 0:
                order = rng.permutation(len(rows)).tolist()
                orders[arm] = order
            batch.append(rows[order[cursor % len(order)]])
            cursors[arm] = cursor + 1
        batches.append(batch)
    return batches


def fractional_batch_schedule(
    ratio: float, *, steps: int, maximum_batch_size: int
) -> list[int]:
    """Deterministically distribute floor/ceil batches without result feedback."""

    bounded = min(float(maximum_batch_size), max(1.0, float(ratio)))
    lower = int(math.floor(bounded))
    fraction = bounded - lower
    accumulator = 0.0
    result = []
    for _ in range(steps):
        value = lower
        accumulator += fraction
        if accumulator >= 1.0 - 1e-12 and lower < maximum_batch_size:
            value += 1
            accumulator -= 1.0
        result.append(value)
    return result


def build_open_model(config: dict[str, Any], vocab_size: int, mx: Any, nn: Any) -> Any:
    model = config["model"]
    d_model = int(model["open_vocab_d_model"])
    return build_model(
        CausalTransformerConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=int(model["num_layers"]),
            num_heads=int(model["open_vocab_heads"]),
            num_kv_heads=1,
            ff_dim=d_model * 2,
            attention_policy="prefix_lm",
            source_target_separator_token_id=SOURCE_TARGET_SEPARATOR_ID,
        ),
        mx=mx,
        nn=nn,
    )


def build_patch_model(config: dict[str, Any], mx: Any, nn: Any) -> Any:
    model = config["model"]
    return build_dynamic_patch_causal_candidate(
        d_model=int(model["patch_d_model"]),
        patch_hidden_dim=int(model["patch_hidden_dim"]),
        max_patches=int(config["supervision"]["maximum_raw_bytes"]),
        max_patch_bytes=int(model["maximum_patch_bytes"]),
        vocab_size=256,
        mx=mx,
        nn=nn,
    )


def parameter_accounting(model: Any, mlx_utils: Any, representation: str) -> dict[str, int]:
    flat = mlx_utils.tree_flatten(model.parameters())
    total = sum(int(value.size) for _name, value in flat)
    if representation == "canonical_open_vocab":
        return {"total": total, "core": total, "local": 0, "boundary": 0}
    boundary = sum(int(value.size) for name, value in flat if ".boundary_" in name)
    local = sum(
        int(value.size)
        for name, value in flat
        if any(part in name for part in ("byte_embedding", "within_patch_position", "local_previous_projection", "local_norm", "byte_decoder"))
    )
    return {"total": total, "core": total - local - boundary, "local": local, "boundary": boundary}


def estimated_work_for_rows(
    rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    accounting: dict[str, int],
    entropy_model: dict[str, Any],
    config: dict[str, Any],
) -> int:
    if candidate["representation"] == "canonical_open_vocab":
        return sum(accounting["core"] * (len(row["token_sequence"]) - 1) for row in rows)
    packed = patch_inputs(rows, candidate["boundary_source"] if candidate["boundary_source"] != "learned_prefix_entropy_prediction" else "prefix_entropy", entropy_model, config)
    byte_positions = int(packed["mask"].sum())
    patch_positions = int(packed["patch_counts"].sum())
    return accounting["core"] * patch_positions + (accounting["local"] + accounting["boundary"]) * byte_positions


def evaluate_open(model: Any, rows_by_arm: dict[str, list[dict[str, Any]]], mx: Any, nn: Any) -> dict[str, Any]:
    by_arm = {}
    loss_mass = 0.0
    token_count = 0
    correct = 0
    raw_bytes = 0
    for arm, rows in rows_by_arm.items():
        x_np, y_np, mask_np = open_vocab_batch(rows)
        logits, _cache = model(mx.array(x_np))
        losses = nn.losses.cross_entropy(logits, mx.array(y_np))
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(losses, predictions)
        local_mass = float((np.asarray(losses) * mask_np).sum())
        local_tokens = int(mask_np.sum())
        local_correct = int(((np.asarray(predictions) == y_np) * mask_np).sum())
        local_bytes = sum(int(row["target_byte_count"]) for row in rows)
        by_arm[arm] = {
            "loss_per_raw_target_byte": local_mass / max(1, local_bytes),
            "token_accuracy": local_correct / max(1, local_tokens),
            "model_positions": local_tokens,
            "raw_target_bytes": local_bytes,
            "functional_verifier_state": "NOT_AVAILABLE_IN_ADMITTED_CORPUS",
        }
        loss_mass += local_mass
        token_count += local_tokens
        correct += local_correct
        raw_bytes += local_bytes
    return {
        "loss_per_raw_target_byte": loss_mass / max(1, raw_bytes),
        "token_accuracy": correct / max(1, token_count),
        "model_positions": token_count,
        "raw_target_bytes": raw_bytes,
        "by_arm": by_arm,
        "functional_verifier_available": False,
        "capability_claim": "NOT_EVALUATED",
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def learned_eval_inputs(
    model: Any,
    rows: list[dict[str, Any]],
    entropy_model: dict[str, Any],
    config: dict[str, Any],
    mx: Any,
) -> dict[str, np.ndarray]:
    teacher = patch_inputs(rows, "prefix_entropy", entropy_model, config)
    output = model(
        mx.array(teacher["ids"]),
        mx.array(teacher["patch_ids"]),
        mx.array(teacher["within"]),
        mx.array(teacher["uncertainty"]),
        mx.array(teacher["mask"]),
    )
    mx.eval(output[1])
    probabilities = sigmoid(np.asarray(output[1]))
    model_cfg = config["model"]
    rebuilt = []
    for index, row in enumerate(rows):
        length = len(row["payload"])
        values = probabilities[index, :length].tolist()
        points = boundaries_from_probabilities(
            values,
            threshold=float(model_cfg["learned_boundary_threshold"]),
            max_patch_bytes=int(model_cfg["maximum_patch_bytes"]),
        )
        rebuilt.append((points, values))
    packed = dict(teacher)
    packed["patch_ids"] = np.zeros_like(teacher["patch_ids"])
    packed["within"] = np.zeros_like(teacher["within"])
    packed["patch_counts"] = np.zeros(len(rows), dtype=np.int32)
    for index, (row, (points, _values)) in enumerate(zip(rows, rebuilt)):
        length = len(row["payload"])
        assignments = patch_ids_from_boundaries(points, length)
        positions = []
        for start, stop in zip(points, points[1:]):
            positions.extend(range(stop - start))
        packed["patch_ids"][index, :length] = assignments
        packed["within"][index, :length] = positions
        packed["patch_counts"][index] = len(points) - 1
    return packed


def evaluate_patch(
    model: Any,
    rows_by_arm: dict[str, list[dict[str, Any]]],
    candidate: dict[str, Any],
    entropy_model: dict[str, Any],
    config: dict[str, Any],
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    by_arm = {}
    loss_mass = 0.0
    target_bytes = 0
    correct = 0
    patch_positions = 0
    raw_positions = 0
    for arm, rows in rows_by_arm.items():
        source = candidate["boundary_source"]
        packed = (
            learned_eval_inputs(model, rows, entropy_model, config, mx)
            if source == "learned_prefix_entropy_prediction"
            else patch_inputs(rows, source, entropy_model, config)
        )
        logits, _boundary, _patches = model(
            mx.array(packed["ids"]),
            mx.array(packed["patch_ids"]),
            mx.array(packed["within"]),
            mx.array(packed["uncertainty"]),
            mx.array(packed["mask"]),
        )
        losses = nn.losses.cross_entropy(logits, mx.array(packed["ids"]))
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(losses, predictions)
        local_mass = float((np.asarray(losses) * packed["target_mask"]).sum())
        local_bytes = int(packed["target_mask"].sum())
        local_correct = int(((np.asarray(predictions) == packed["ids"]) * packed["target_mask"]).sum())
        local_patches = int(packed["patch_counts"].sum())
        local_raw = int(packed["mask"].sum())
        by_arm[arm] = {
            "loss_per_raw_target_byte": local_mass / max(1, local_bytes),
            "byte_accuracy": local_correct / max(1, local_bytes),
            "patch_positions": local_patches,
            "raw_input_bytes": local_raw,
            "contraction_ratio": local_raw / max(1, local_patches),
            "functional_verifier_state": "NOT_AVAILABLE_IN_ADMITTED_CORPUS",
        }
        loss_mass += local_mass
        target_bytes += local_bytes
        correct += local_correct
        patch_positions += local_patches
        raw_positions += local_raw
    return {
        "loss_per_raw_target_byte": loss_mass / max(1, target_bytes),
        "byte_accuracy": correct / max(1, target_bytes),
        "patch_positions": patch_positions,
        "raw_input_bytes": raw_positions,
        "contraction_ratio": raw_positions / max(1, patch_positions),
        "raw_target_bytes": target_bytes,
        "by_arm": by_arm,
        "functional_verifier_available": False,
        "capability_claim": "NOT_EVALUATED",
    }


def run_candidate(
    config: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    train_rows: dict[str, list[dict[str, Any]]],
    heldout_rows: dict[str, list[dict[str, Any]]],
    entropy_model: dict[str, Any],
    batch_schedule: list[int],
    monitor: pretraining_candidate_canary.CandidateCanaryMonitor,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    metadata = json.loads(resolve(config["stage_metadata"]).read_text(encoding="utf-8"))
    base = json.loads(resolve(config["base_config"]).read_text(encoding="utf-8"))
    vocab_size = model_vocab_size(base, metadata["source_vocab"], metadata["target_vocab"])
    mx.random.seed(int(seed))
    if candidate["representation"] == "canonical_open_vocab":
        model = build_open_model(config, vocab_size, mx, nn)
    else:
        model = build_patch_model(config, mx, nn)
    accounting = parameter_accounting(model, mlx_utils, candidate["representation"])
    steps = int(config["training"]["steps"])
    batches = balanced_step_rows(
        train_rows,
        steps=steps,
        seed=seed,
        batch_schedule=batch_schedule,
    )
    optimizer = optim.AdamW(
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    if candidate["representation"] == "canonical_open_vocab":
        def objective(local_model: Any, x: Any, y: Any, mask: Any) -> Any:
            logits, _cache = local_model(x)
            losses = nn.losses.cross_entropy(logits, y)
            return mx.sum(losses * mask) / mx.maximum(mx.sum(mask), 1.0)
        value_and_grad = nn.value_and_grad(model, objective)
        initial = evaluate_open(model, heldout_rows, mx, nn)
    else:
        def objective(local_model: Any, ids: Any, patch_ids: Any, within: Any, uncertainty: Any, mask: Any, target_mask: Any, boundary_y: Any) -> Any:
            logits, boundary_logits, _patches = local_model(ids, patch_ids, within, uncertainty, mask)
            token_loss = mx.sum(nn.losses.cross_entropy(logits, ids) * target_mask) / mx.maximum(mx.sum(target_mask), 1.0)
            boundary_loss = mx.sum(nn.losses.binary_cross_entropy(boundary_logits, boundary_y, with_logits=True, reduction="none") * mask) / mx.maximum(mx.sum(mask), 1.0)
            return token_loss + float(config["training"]["boundary_loss_scale"]) * boundary_loss
        value_and_grad = nn.value_and_grad(model, objective)
        initial = evaluate_patch(model, heldout_rows, candidate, entropy_model, config, mx, nn)

    curve = [{"step": 0, "heldout": initial}]
    started = time.perf_counter()
    optimizer_positions = 0
    raw_byte_exposures = 0
    estimated_work = 0
    active_gradient = False
    boundary_gradient_l1 = 0.0
    for step, rows in enumerate(batches, 1):
        if candidate["representation"] == "canonical_open_vocab":
            x_np, y_np, mask_np = open_vocab_batch(rows)
            loss, gradients = value_and_grad(model, mx.array(x_np), mx.array(y_np), mx.array(mask_np))
            optimizer_positions += int(mask_np.sum())
            raw_byte_exposures += sum(len(row["payload"]) for row in rows)
        else:
            source = candidate["boundary_source"]
            training_source = "prefix_entropy" if source == "learned_prefix_entropy_prediction" else source
            packed = patch_inputs(rows, training_source, entropy_model, config)
            loss, gradients = value_and_grad(
                model,
                mx.array(packed["ids"]),
                mx.array(packed["patch_ids"]),
                mx.array(packed["within"]),
                mx.array(packed["uncertainty"]),
                mx.array(packed["mask"]),
                mx.array(packed["target_mask"]),
                mx.array(packed["boundary_targets"]),
            )
            optimizer_positions += int(packed["patch_counts"].sum())
            raw_byte_exposures += int(packed["mask"].sum())
        for name, value in mlx_utils.tree_flatten(gradients):
            amount = float(mx.sum(mx.abs(value)).item())
            active_gradient = active_gradient or amount > 0.0
            if ".boundary_" in name:
                boundary_gradient_l1 += amount
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        estimated_work += estimated_work_for_rows(rows, candidate, accounting, entropy_model, config)
        monitor.check(f"{candidate['id']}:{seed}:train", step)
        if step % int(config["training"]["evaluation_interval_steps"]) == 0 or step == steps:
            evaluation = (
                evaluate_open(model, heldout_rows, mx, nn)
                if candidate["representation"] == "canonical_open_vocab"
                else evaluate_patch(model, heldout_rows, candidate, entropy_model, config, mx, nn)
            )
            curve.append({"step": step, "heldout": evaluation})
    wall = time.perf_counter() - started
    checkpoint_dir = resolve(config["scratch_root"]) / candidate["id"] / str(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "weights.safetensors"
    model.save_weights(str(checkpoint))
    mx.random.seed(int(seed) + 999)
    reloaded = (
        build_open_model(config, vocab_size, mx, nn)
        if candidate["representation"] == "canonical_open_vocab"
        else build_patch_model(config, mx, nn)
    )
    reloaded.load_weights(str(checkpoint))
    replay = (
        evaluate_open(reloaded, heldout_rows, mx, nn)
        if candidate["representation"] == "canonical_open_vocab"
        else evaluate_patch(reloaded, heldout_rows, candidate, entropy_model, config, mx, nn)
    )
    final = curve[-1]["heldout"]
    reload_exact = canonical(final) == canonical(replay)
    if not reload_exact:
        raise DynamicPatchAdequacyFault(f"checkpoint_reload_not_exact:{candidate['id']}:{seed}")
    return {
        "candidate_id": candidate["id"],
        "representation": candidate["representation"],
        "boundary_source": candidate["boundary_source"],
        "seed": int(seed),
        "mean_batch_size": mean([float(value) for value in batch_schedule]),
        "batch_schedule": batch_schedule,
        "batch_schedule_digest": digest(batch_schedule),
        "optimizer_steps": steps,
        "optimizer_positions": optimizer_positions,
        "raw_byte_exposures": raw_byte_exposures,
        "estimated_parameter_position_work": estimated_work,
        "parameter_accounting": accounting,
        "active_gradient": active_gradient,
        "boundary_gradient_l1": boundary_gradient_l1,
        "initial_heldout": initial,
        "final_heldout": final,
        "learning_curve": curve,
        "training_wall_seconds": wall,
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_reload_exact": reload_exact,
    }


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def summarize(config: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate = {}
    for candidate in config["candidates"]:
        rows = [row for row in runs if row["candidate_id"] == candidate["id"]]
        by_candidate[candidate["id"]] = {
            "seeds": [row["seed"] for row in rows],
            "mean_final_loss_per_raw_target_byte": mean([float(row["final_heldout"]["loss_per_raw_target_byte"]) for row in rows]),
            "mean_training_wall_seconds": mean([float(row["training_wall_seconds"]) for row in rows]),
            "mean_estimated_parameter_position_work": mean([float(row["estimated_parameter_position_work"]) for row in rows]),
            "mean_raw_byte_exposures": mean([float(row["raw_byte_exposures"]) for row in rows]),
            "parameter_count": int(rows[0]["parameter_accounting"]["total"]),
            "mean_contraction_ratio": mean([float(row["final_heldout"].get("contraction_ratio") or 1.0) for row in rows]),
            "weak_arm_loss_per_raw_byte": max(mean([float(row["final_heldout"]["by_arm"][arm]["loss_per_raw_target_byte"]) for row in rows]) for arm in config["scoped_arms"]),
            "checkpoint_reload_exact": all(row["checkpoint_reload_exact"] for row in rows),
            "active_gradient": all(row["active_gradient"] for row in rows),
        }
    decision = config["prospective_decision"]
    dynamic_id = decision["dynamic_candidate"]
    dynamic = by_candidate[dynamic_id]
    controls = {key: value for key, value in by_candidate.items() if key != dynamic_id}
    best_control_id, best_control = min(
        controls.items(), key=lambda item: item[1]["mean_final_loss_per_raw_target_byte"]
    )
    paired = []
    dynamic_runs = {row["seed"]: row for row in runs if row["candidate_id"] == dynamic_id}
    control_runs = {row["seed"]: row for row in runs if row["candidate_id"] == best_control_id}
    for seed in config["seeds"]:
        candidate_row = dynamic_runs[int(seed)]
        control_row = control_runs[int(seed)]
        candidate_loss = float(candidate_row["final_heldout"]["loss_per_raw_target_byte"])
        control_loss = float(control_row["final_heldout"]["loss_per_raw_target_byte"])
        arm_regressions = []
        for arm in config["scoped_arms"]:
            candidate_arm = float(candidate_row["final_heldout"]["by_arm"][arm]["loss_per_raw_target_byte"])
            control_arm = float(control_row["final_heldout"]["by_arm"][arm]["loss_per_raw_target_byte"])
            arm_regressions.append((candidate_arm - control_arm) / max(control_arm, 1e-12))
        paired.append(
            {
                "seed": int(seed),
                "relative_loss_per_raw_byte_improvement": (control_loss - candidate_loss) / max(control_loss, 1e-12),
                "maximum_weak_arm_relative_loss_regression": max(arm_regressions),
                "estimated_work_ratio": float(candidate_row["estimated_parameter_position_work"]) / max(float(control_row["estimated_parameter_position_work"]), 1.0),
            }
        )
    improvements = [row["relative_loss_per_raw_byte_improvement"] for row in paired]
    weak = [row["maximum_weak_arm_relative_loss_regression"] for row in paired]
    work = [row["estimated_work_ratio"] for row in paired]
    parameter_ratio = dynamic["parameter_count"] / max(1, best_control["parameter_count"])
    gates = {
        "relative_loss_per_raw_byte": mean(improvements) >= float(decision["minimum_relative_loss_per_raw_byte_improvement"]),
        "weak_arm_regression": max(weak) <= float(decision["maximum_weak_arm_relative_loss_regression"]),
        "seed_win_fraction": sum(value > 0 for value in improvements) / len(improvements) >= float(decision["minimum_seed_win_fraction"]),
        "estimated_work": mean(work) <= float(decision["maximum_mean_work_ratio"]),
        "parameter_count": max(parameter_ratio, 1.0 / max(parameter_ratio, 1e-12)) <= float(decision["maximum_parameter_count_ratio"]),
        "functional_verifier": not bool(decision["functional_verifier_required_for_adoption"]),
        "checkpoint_reload": dynamic["checkpoint_reload_exact"],
        "gradient_flow": dynamic["active_gradient"],
    }
    adopted = all(gates.values())
    return {
        "by_candidate": by_candidate,
        "dynamic_comparison": {
            "dynamic_candidate": dynamic_id,
            "best_bounded_control": best_control_id,
            "paired_runs": paired,
            "mean_relative_loss_per_raw_byte_improvement": mean(improvements),
            "maximum_weak_arm_relative_loss_regression": max(weak),
            "seed_win_fraction": sum(value > 0 for value in improvements) / len(improvements),
            "mean_estimated_work_ratio": mean(work),
            "parameter_count_ratio": parameter_ratio,
            "gates": gates,
            "disposition": "ADOPTED" if adopted else "SCOPED_DISCOVERY_EXCLUDED_FROM_FIRST_CAMPAIGN",
            "scientific_falsification_claimed": False,
            "reason": (
                "all preregistered quality, weak-tail, seed, cost, parameter, replay, and functional gates passed"
                if adopted
                else "bounded codec evidence cannot support first-campaign adoption; missing functional evidence or a prospective gate failed"
            ),
            "reentry_condition": None if adopted else "source-disjoint verifier-bearing assistant/code rows at a larger useful model rung with matched raw data, parameters, work, optimizer opportunity, and direct generation",
        },
    }


def execute(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    scratch = resolve(config["scratch_root"])
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    lease = pretraining_candidate_canary.candidate_lease(
        candidate_id=config["candidate_lease_id"],
        max_steps=int(config["training"]["steps"]),
        scratch_checkpoint_root=scratch,
        targets=["shared_trunk"],
        phase="pretraining",
        resume=False,
    )
    if not lease["authorized"]:
        raise DynamicPatchAdequacyFault("candidate_lease_denied:" + ",".join(lease["faults"]))
    monitor = pretraining_candidate_canary.CandidateCanaryMonitor(lease)
    train_rows = load_governed_rows(config, "train")
    heldout_rows = load_governed_rows(config, "heldout")
    disjoint = source_disjoint_receipt(train_rows, heldout_rows)
    if not disjoint["passed"]:
        raise DynamicPatchAdequacyFault("source_disjoint_contract_failed")
    entropy_model = fit_prefix_entropy(row for rows in train_rows.values() for row in rows)

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    metadata = json.loads(resolve(config["stage_metadata"]).read_text(encoding="utf-8"))
    base = json.loads(resolve(config["base_config"]).read_text(encoding="utf-8"))
    vocab_size = model_vocab_size(base, metadata["source_vocab"], metadata["target_vocab"])
    accounting = {}
    average_work = {}
    sample_rows = [row for rows in train_rows.values() for row in rows]
    for candidate in config["candidates"]:
        model = build_open_model(config, vocab_size, mx, nn) if candidate["representation"] == "canonical_open_vocab" else build_patch_model(config, mx, nn)
        accounting[candidate["id"]] = parameter_accounting(model, mlx_utils, candidate["representation"])
        average_work[candidate["id"]] = estimated_work_for_rows(sample_rows, candidate, accounting[candidate["id"]], entropy_model, config) / len(sample_rows)
    reference = max(average_work.values())
    maximum_batch = int(config["training"]["maximum_compute_matched_batch_size"])
    batch_schedules = {
        candidate["id"]: fractional_batch_schedule(
            reference / max(average_work[candidate["id"]], 1.0),
            steps=int(config["training"]["steps"]),
            maximum_batch_size=maximum_batch,
        )
        for candidate in config["candidates"]
    }
    runs = []
    for candidate in config["candidates"]:
        for seed in config["seeds"]:
            runs.append(
                run_candidate(
                    config,
                    candidate,
                    int(seed),
                    train_rows,
                    heldout_rows,
                    entropy_model,
                    batch_schedules[candidate["id"]],
                    monitor,
                )
            )
    resource_receipt = monitor.finalize(runs)
    summary = summarize(config, runs)
    hard_boundaries = dict(config["hard_boundaries"])
    report = {
        "policy": POLICY,
        "schema_version": "1.0.0",
        "trigger_state": "GREEN" if resource_receipt["passed"] and disjoint["passed"] and all(row["checkpoint_reload_exact"] for row in runs) else "RED",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate_contract": config["candidate_contract"],
        "candidate_lease": lease,
        "source_disjoint": disjoint,
        "work_matching": {
            "method": "same parameter scale and optimizer steps; batch size frozen from parameter-position estimate",
            "average_single_row_work": average_work,
            "batch_schedules": batch_schedules,
            "mean_batch_sizes": {
                key: mean([float(value) for value in schedule])
                for key, schedule in batch_schedules.items()
            },
            "parameter_accounting": accounting,
            "raw_byte_exposure_reported_separately": True,
        },
        "runs": runs,
        "summary": summary,
        "resource_receipt": resource_receipt,
        "hard_boundaries": hard_boundaries,
        "candidate_integrity": {
            "hidden_target_metadata_visible": False,
            "public_or_confirmation_surface_consumed": False,
            "deterministic_or_template_credit": 0,
            "learned_boundary_inputs": "current_and_prior_generated_bytes_plus_train_fitted_prefix_entropy_only",
            "functional_claim": "NOT_EVALUATED",
        },
    }
    resolve(config["report"]).parent.mkdir(parents=True, exist_ok=True)
    resolve(config["report"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    train = load_governed_rows(config, "train")
    heldout = load_governed_rows(config, "heldout")
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if source_disjoint_receipt(train, heldout)["passed"] else "RED",
        "candidate_count": len(config["candidates"]),
        "seed_count": len(config["seeds"]),
        "train_rows": sum(map(len, train.values())),
        "heldout_rows": sum(map(len, heldout.values())),
        "source_disjoint": source_disjoint_receipt(train, heldout),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = execute(resolve(args.config)) if args.execute else preflight(resolve(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("trigger_state") == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
