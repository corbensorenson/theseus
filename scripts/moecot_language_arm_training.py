#!/usr/bin/env python3
"""Train a shared MoECOT trunk, language experts, and matched dense controls.

The runtime consumes the immutable canonical stage produced by the standard
transformer corpus path. It does not build another corpus, route answers, or
turn training loss into a capability claim.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import difflib
import hashlib
import heapq
import json
import math
import os
import random
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

import host_resource_safety
import pretraining_candidate_canary
import pretraining_optimizers


KERC_UNIT_KIND_IDS = {
    "interaction_entry": 0,
    "segment_frame": 1,
    "token_residue": 2,
    "concept_realization": 3,
    "exact_object": 4,
}
KERC_UNIT_CANDIDATE_BASE_FEATURE_DIM = 18
KERC_UNIT_SOURCE_RELATION_FEATURE_DIM = 64
KERC_UNIT_CANDIDATE_FEATURE_DIM = (
    KERC_UNIT_CANDIDATE_BASE_FEATURE_DIM + KERC_UNIT_SOURCE_RELATION_FEATURE_DIM
)
_KERC_RELATION_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.:-]{1,}|-?\d+(?:\.\d+)?"
)


def training_operation(args: argparse.Namespace) -> str:
    if bool(getattr(args, "migrate_shared_trunk_checkpoint_format", False)):
        return "checkpoint_migration"
    if bool(getattr(args, "evaluate_progress", False)):
        return "evaluation"
    if bool(getattr(args, "execute", False)):
        return "training"
    return ""


def training_host_policy(
    config: dict[str, Any],
) -> host_resource_safety.HostSafetyPolicy:
    contract = config.get("host_resource_safety") or {}
    required = {
        "policy",
        "required_for_training",
        "required_for_evaluation",
        "required_for_checkpoint_migration",
        "candidate_canaries_use_freeze_shard_runner",
        "qualified_python",
        "receipt_directory",
        "maximum_process_memory_mib",
        "minimum_available_before_launch_mib",
        "minimum_available_during_run_mib",
        "maximum_swapout_growth_mib",
        "maximum_wall_seconds",
        "poll_interval_seconds",
        "terminate_grace_seconds",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError("host resource safety contract incomplete: " + ",".join(missing))
    receipt_directory = Path(str(contract["receipt_directory"]))
    qualified_python = resolve(str(contract["qualified_python"]))
    if (
        contract["policy"] != host_resource_safety.POLICY
        or any(
            contract[key] is not True
            for key in (
                "required_for_training",
                "required_for_evaluation",
                "required_for_checkpoint_migration",
                "candidate_canaries_use_freeze_shard_runner",
            )
        )
        or receipt_directory.is_absolute()
        or not receipt_directory.parts
        or receipt_directory.parts[0] != "reports"
        or ".." in receipt_directory.parts
        or not qualified_python.is_file()
    ):
        raise ValueError("host resource safety contract invalid")
    policy = host_resource_safety.HostSafetyPolicy(
        max_process_memory_mib=float(contract["maximum_process_memory_mib"]),
        minimum_available_before_launch_mib=float(
            contract["minimum_available_before_launch_mib"]
        ),
        minimum_available_during_run_mib=float(
            contract["minimum_available_during_run_mib"]
        ),
        maximum_swapout_growth_mib=float(contract["maximum_swapout_growth_mib"]),
        maximum_wall_seconds=float(contract["maximum_wall_seconds"]),
        poll_interval_seconds=float(contract["poll_interval_seconds"]),
        terminate_grace_seconds=float(contract["terminate_grace_seconds"]),
        swapout_growth_action=str(
            contract.get("swapout_growth_action") or "hard_stop"
        ),
    )
    policy.validate(physical_memory_mib=host_resource_safety.physical_memory_mib())
    return policy


def launch_guarded_training(
    config: dict[str, Any], operation: str, *, candidate_id: str = ""
) -> int:
    policy = training_host_policy(config)
    if candidate_id:
        contract = pretraining_candidate_canary.load_contract()
        candidate = next(
            (
                row
                for row in contract["canaries"]
                if row["candidate_id"] == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ValueError("guarded candidate is not registered")
        policy = host_resource_safety.policy_from_mapping(
            pretraining_candidate_canary.candidate_host_safety_mapping(
                candidate_id, contract
            ),
            maximum_wall_seconds=float(candidate["max_wall_seconds"]),
        )
    child_args = list(sys.argv[1:])
    child_args.remove("--guarded")
    command = [
        str(resolve(str(config["host_resource_safety"]["qualified_python"]))),
        relative(Path(__file__).resolve()),
        *child_args,
    ]
    try:
        result = host_resource_safety.run_guarded(
            command,
            cwd=ROOT,
            policy=policy,
            env={
                "THESEUS_GUARDED_ACCELERATOR_CHILD": "1",
                "THESEUS_GUARDED_TRAINING_CHILD": "1",
            },
        )
        receipt = {**result.receipt, "operation": operation}
    except host_resource_safety.HostResourceSafetyFault as exc:
        result = None
        receipt = {
            "policy": host_resource_safety.POLICY,
            "operation": operation,
            "command": command,
            "passed": False,
            "child_started": False,
            "terminated_by_guard": False,
            "fault": str(exc),
            "returncode": None,
            "limits": asdict(policy),
        }
    receipt_path = resolve(
        str(
            Path(config["host_resource_safety"]["receipt_directory"])
            / (
                f"{operation}-{candidate_id}-latest.json"
                if candidate_id
                else f"{operation}-latest.json"
            )
        )
    )
    write_json_atomic(receipt_path, receipt)
    durable_receipt_path = guarded_output_receipt_path(child_args)
    if durable_receipt_path is not None:
        write_json_atomic(durable_receipt_path, receipt)
    if result is not None and result.stdout:
        print(result.stdout[-4000:], end="" if result.stdout.endswith("\n") else "\n")
    if result is not None and result.stderr:
        print(result.stderr[-4000:], end="" if result.stderr.endswith("\n") else "\n")
    print(
        json.dumps(
            {
                "passed": receipt["passed"],
                "fault": receipt["fault"],
                "returncode": receipt["returncode"],
                "receipt": relative(receipt_path),
                "durable_receipt": (
                    relative(durable_receipt_path)
                    if durable_receipt_path is not None
                    else ""
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["passed"] else 2


def guarded_output_receipt_path(child_args: list[str]) -> Path | None:
    """Derive one durable watchdog receipt beside an explicit JSON report."""

    if "--out" not in child_args:
        return None
    index = child_args.index("--out")
    if index + 1 >= len(child_args) or not str(child_args[index + 1]).strip():
        raise ValueError("guarded --out requires a report path")
    report = resolve(child_args[index + 1])
    return report.with_name(report.stem + ".host_resource_safety.json")


def _kerc_relation_tokens(value: str | bytes) -> tuple[str, ...]:
    text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
    tokens: set[str] = set()
    for raw in _KERC_RELATION_TOKEN_RE.findall(text):
        token = raw.casefold()
        token = re.sub(r"[0-9a-f]{12,}", "<id>", token)
        token = re.sub(r"\d+", "<n>", token)
        for part in re.split(r"[.:-]+", token):
            if len(part) >= 2:
                tokens.add(part)
    return tuple(sorted(tokens))


def _signed_hash_sketch(tokens: tuple[str, ...], width: int, namespace: str) -> np.ndarray:
    result = np.zeros(width, dtype=np.float32)
    for token in tokens:
        digest = hashlib.sha256(f"{namespace}:{token}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % width
        result[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 0.0 else result


def kerc_unit_source_relation_features(
    *, prompt: str, source_path: str, payload: bytes
) -> np.ndarray:
    """Encode source-only unit-to-task relations without target/evaluator access."""

    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError as exc:
        raise ValueError("KERC allocator prompt must be canonical structured JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("KERC allocator prompt must contain a structured source object")
    # The residual inventory itself would make every unit trivially overlap.  Task
    # relevance is measured against the source-side program and governed objects.
    context = {
        key: parsed.get(key)
        for key in (
            "program",
            "concept_capsules",
            "protected_objects",
            "source_character_length",
        )
        if key in parsed
    }
    context_text = json.dumps(
        context, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    unit_tokens = _kerc_relation_tokens(payload)
    path_tokens = _kerc_relation_tokens(source_path)
    context_tokens = _kerc_relation_tokens(context_text)
    unit_set = set(unit_tokens)
    path_set = set(path_tokens)
    context_set = set(context_tokens)
    overlap = tuple(sorted(unit_set & context_set))
    path_overlap = tuple(sorted(path_set & context_set))
    payload_size = max(1, len(payload))
    scalars = np.asarray(
        [
            min(1.0, math.log1p(len(payload)) / 12.0),
            min(1.0, math.log1p(len(context_text.encode("utf-8"))) / 14.0),
            min(1.0, math.log1p(len(source_path)) / 8.0),
            sum(32 <= value < 127 for value in payload) / payload_size,
            min(1.0, len(unit_tokens) / 64.0),
            min(1.0, len(context_tokens) / 512.0),
            min(1.0, len(overlap) / 16.0),
            len(overlap) / max(1, len(unit_tokens)),
            min(1.0, len(path_overlap) / 8.0),
            len(path_overlap) / max(1, len(path_tokens)),
            sum(48 <= value <= 57 for value in payload) / payload_size,
            sum(65 <= value <= 90 for value in payload) / payload_size,
            sum(value in b"{}[],:" for value in payload) / payload_size,
            float(bool(parsed.get("protected_objects"))),
            float(bool(parsed.get("concept_capsules"))),
            float(bool((parsed.get("program") or {}).get("tokens"))),
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            scalars,
            _signed_hash_sketch(unit_tokens, 16, "unit"),
            _signed_hash_sketch(context_tokens, 16, "context"),
            _signed_hash_sketch(
                tuple(sorted(set(overlap) | {f"path:{value}" for value in path_tokens})),
                16,
                "relation",
            ),
        ]
    )
    if features.shape != (KERC_UNIT_SOURCE_RELATION_FEATURE_DIM,) or not np.isfinite(
        features
    ).all():
        raise ValueError("invalid KERC source-relation feature vector")
    return features.astype(np.float32)


def materialize_kerc_unit_allocator_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Decode bounded K3 source-visible unit supervision without evaluator fields."""

    if row.get("kerc_residual_unit_allocator_loss_enabled") is not True:
        return None
    targets = list(row.get("kerc_residual_unit_targets") or [])
    if not targets:
        raise ValueError("enabled KERC per-unit allocation row has no unit targets")
    byte_rows: list[np.ndarray] = []
    kind_ids: list[int] = []
    candidate_features: list[np.ndarray] = []
    hard_masks: list[np.ndarray] = []
    labels: list[int] = []
    confidence: list[float] = []
    authority: list[float] = []
    unit_ids: list[str] = []
    prompt = str(row.get("prompt") or "")
    if not prompt:
        raise ValueError("KERC per-unit allocation row has no source prompt")
    for target in targets:
        unit_id = str(target.get("unit_id") or "")
        kind = str(target.get("unit_kind") or "")
        source_path = str(target.get("source_path") or "")
        try:
            payload = base64.b64decode(
                str(target.get("source_payload_wire_b64") or ""), validate=True
            )
        except ValueError as exc:
            raise ValueError(f"invalid KERC unit payload encoding: {unit_id}") from exc
        if not unit_id or kind not in KERC_UNIT_KIND_IDS or not source_path or not payload:
            raise ValueError(f"invalid KERC unit identity or payload: {unit_id}")
        candidates = list(target.get("candidates") or [])
        source_visible = list(target.get("source_visible_candidates") or [])
        if len(candidates) != 4 or len(source_visible) != 4:
            raise ValueError(f"KERC unit requires four candidate actions: {unit_id}")
        maximum_bits = max(
            1, max(int(candidate.get("encoded_bits") or 0) for candidate in source_visible)
        )
        maximum_uncompressed = max(
            1,
            max(
                int(candidate.get("uncompressed_bits") or 0)
                for candidate in source_visible
            ),
        )
        maximum_distortion = float(target.get("maximum_structural_distortion") or 0.0)
        relation_features = kerc_unit_source_relation_features(
            prompt=prompt, source_path=source_path, payload=payload
        )
        features = []
        hard = []
        for index, (candidate, visible) in enumerate(zip(candidates, source_visible)):
            distortion = list(visible.get("distortion_vector") or [])
            encoded_bits = int(visible.get("encoded_bits") or 0)
            uncompressed_bits = int(visible.get("uncompressed_bits") or 0)
            if (
                len(distortion) != 13
                or int(candidate.get("fidelity_index", -1)) != index
                or int(visible.get("fidelity_index", -1)) != index
                or encoded_bits != int(candidate.get("encoded_bits") or 0)
            ):
                raise ValueError(f"invalid KERC unit candidate features: {unit_id}:{index}")
            feature = [
                encoded_bits / maximum_bits,
                uncompressed_bits / maximum_uncompressed,
                encoded_bits / max(1, uncompressed_bits),
                float(visible.get("structural_loss") or 0.0),
                maximum_distortion,
                *[-1.0 if value is None else float(value) for value in distortion],
            ]
            if len(feature) != KERC_UNIT_CANDIDATE_BASE_FEATURE_DIM:
                raise ValueError(f"invalid KERC unit base feature vector: {unit_id}:{index}")
            feature.extend(float(value) for value in relation_features)
            if len(feature) != KERC_UNIT_CANDIDATE_FEATURE_DIM or not np.isfinite(feature).all():
                raise ValueError(f"invalid KERC unit feature vector: {unit_id}:{index}")
            features.append(feature)
            hard.append(bool(candidate.get("hard_blocked")))
        selected = int(target.get("selected_fidelity_index", -1))
        if selected not in range(4) or hard[selected]:
            raise ValueError(f"invalid KERC unit target choice: {unit_id}")
        source_visible_bytes = source_path.encode("utf-8") + b"\x00" + payload
        byte_rows.append(
            np.frombuffer(source_visible_bytes, dtype=np.uint8).astype(np.int32)
        )
        kind_ids.append(KERC_UNIT_KIND_IDS[kind])
        candidate_features.append(np.asarray(features, dtype=np.float32))
        hard_masks.append(np.asarray(hard, dtype=bool))
        labels.append(selected)
        confidence.append(float(target.get("confidence_target") or 0.0))
        authority.append(float(bool(target.get("allocator_loss_enabled"))))
        unit_ids.append(unit_id)
    if not any(authority):
        raise ValueError("KERC per-unit allocation row has no authoritative target")
    return {
        "unit_ids": tuple(unit_ids),
        "byte_rows": tuple(byte_rows),
        "kind_ids": np.asarray(kind_ids, dtype=np.int32),
        "candidate_features": np.asarray(candidate_features, dtype=np.float32),
        "hard_block_mask": np.asarray(hard_masks, dtype=bool),
        "labels": np.asarray(labels, dtype=np.int32),
        "confidence_targets": np.asarray(confidence, dtype=np.float32),
        "loss_mask": np.asarray(authority, dtype=np.float32),
    }


def without_kerc_unit_loss(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    result["loss_mask"] = np.zeros_like(row["loss_mask"], dtype=np.float32)
    return result


def pack_kerc_unit_allocator_batch(
    rows: list[dict[str, Any] | None],
) -> dict[str, np.ndarray] | None:
    active = [row for row in rows if row is not None]
    if not active:
        return None
    maximum_units = max(len(row["unit_ids"]) for row in active)
    batch = len(rows)
    flat_byte_rows: list[np.ndarray] = []
    byte_offsets = np.zeros((batch, maximum_units, 2), dtype=np.int64)
    kind_ids = np.zeros((batch, maximum_units), dtype=np.int32)
    features = np.zeros(
        (batch, maximum_units, 4, KERC_UNIT_CANDIDATE_FEATURE_DIM),
        dtype=np.float32,
    )
    hard = np.ones((batch, maximum_units, 4), dtype=bool)
    labels = np.zeros((batch, maximum_units), dtype=np.int32)
    confidence = np.zeros((batch, maximum_units), dtype=np.float32)
    unit_mask = np.zeros((batch, maximum_units), dtype=np.float32)
    loss_mask = np.zeros((batch, maximum_units), dtype=np.float32)
    byte_cursor = 0
    for batch_index, row in enumerate(rows):
        if row is None:
            continue
        count = len(row["unit_ids"])
        kind_ids[batch_index, :count] = row["kind_ids"]
        features[batch_index, :count] = row["candidate_features"]
        hard[batch_index, :count] = row["hard_block_mask"]
        labels[batch_index, :count] = row["labels"]
        confidence[batch_index, :count] = row["confidence_targets"]
        unit_mask[batch_index, :count] = 1.0
        loss_mask[batch_index, :count] = row["loss_mask"]
        for unit_index, payload in enumerate(row["byte_rows"]):
            start = byte_cursor
            flat_byte_rows.append(np.asarray(payload, dtype=np.int32))
            byte_cursor += len(payload)
            byte_offsets[batch_index, unit_index] = (start, byte_cursor)
    if np.any((unit_mask == 1.0) & hard.all(axis=-1)):
        raise ValueError("KERC per-unit batch contains a unit with no admissible action")
    byte_ids = np.concatenate(flat_byte_rows)
    return {
        "byte_ids": byte_ids,
        "byte_offsets": byte_offsets,
        "kind_ids": kind_ids,
        "candidate_features": features,
        "hard_block_mask": hard,
        "labels": labels,
        "confidence_targets": confidence,
        "unit_mask": unit_mask,
        "loss_mask": loss_mask,
    }

from kerc_checkpoint_schema import CURRENT_SCHEMA, CURRENT_SCHEMA_VERSION, POLICY as KERC_CHECKPOINT_POLICY
from kerc_concept_registry import ConceptRegistry
from kernel_english_protocol import (
    ANSWER_DISPOSITION_ORDER,
    KERC_HIERARCHICAL_COMPILER_POLICY,
    KERC_COMPILER_SEMANTIC_TARGET_KINDS,
    KERC_VERIFIER_DIMENSIONS,
    KERNEL_VERSION,
    LEARNED_COMPILER_COMPACT_TRANSPORT_POLICY,
    LEARNED_COMPILER_COMPACT_TRANSPORT_VERSION,
    LEARNED_COMPILER_SEMANTIC_POINTER_TRANSPORT_POLICY,
    LEARNED_COMPILER_SEMANTIC_POINTER_TRANSPORT_VERSION,
    LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_POLICY,
    LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION,
    LEARNED_ANSWER_TRANSPORT_POLICY,
    LEARNED_PROGRAM_TRANSPORT_POLICY,
    KernelProtocolFault,
    TRAINING_TASK_TAGS,
    compact_learned_compiler_transport_text,
    execute_learned_pipeline,
    learned_compiler_transport_required_continuation_token_indices,
    learned_compiler_transport_semantic_pointer_token_indices_by_kind,
    parse_learned_answer_output,
    parse_learned_compiler_output,
    validate_training_disposition,
)
from standard_causal_transformer_model import (
    CausalTransformerConfig,
    analytical_parameter_count,
    analytical_trainable_parameter_count,
    build_model,
    parameter_count,
)
from standard_causal_transformer_corpus import load_pretrain_memmaps, pretrain_array_paths
from standard_causal_transformer_survival import (
    GLOBAL_BOS_ID,
    SOURCE_TARGET_SEPARATOR_ID,
    batched_beam_advance as advance_beams_batched,
    build_schedule,
    cache_arrays,
    causal_loss,
    checkpointed_causal_loss,
    decomposed_checkpointed_causal_loss_and_grad,
    evaluate_loss,
    model_vocab_size,
    required_steps,
    serial_beam_advance as advance_beams_serial,
    source_token_offset,
    target_token_offset,
    stratified_low_variance_sampling_order,
    train_phase,
)
from moecot_language_tokenizer import exact_text_tokens
from moecot_source_conditioned_pretraining import (
    KERC_KERNEL_OBJECTIVES,
    KERC_SEQUENCE_BUCKET_POLICY,
    KERC_STRUCTURED_SOURCE_OBJECTIVES,
    decode_kerc_global_target,
    encode_kerc_global_target,
    encode_kerc_global_target_with_logical_ranges,
    kerc_code_tokens,
    kerc_surface_tokens,
)
from neural_seed_open_vocab import (
    MAX_TOKEN_BYTES,
    TARGET_BYTE_BEGIN,
    TARGET_BYTE_END,
    active_target_span,
    byte_token_bytes,
    decode_target_tokens,
    encode_tokens,
    is_byte_token,
)
import vcm_semantic_memory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
DEFAULT_CHECKPOINT_FORMAT_MIGRATION_REPORT = (
    ROOT / "reports" / "shared_trunk_checkpoint_format_migration.json"
)
ARM_IDS = ("english", "python", "javascript_typescript", "html_css", "rust")


def kerc_unit_allocator_training_authority(config: dict[str, Any]) -> dict[str, Any]:
    """Admit K3 loss to long training only after decision-grade qualification."""

    configured = str(config.get("kerc_unit_allocator_qualification") or "")
    gaps: list[str] = []
    qualification_config_path = ROOT / configured if configured else Path()
    if not configured or not qualification_config_path.is_file():
        gaps.append("qualification_config_missing")
        qualification_config: dict[str, Any] = {}
    else:
        qualification_config = json.loads(
            qualification_config_path.read_text(encoding="utf-8")
        )
    report_value = str(qualification_config.get("report") or "")
    report_path = ROOT / report_value if report_value else Path()
    if not report_value or not report_path.is_file():
        gaps.append("qualification_report_missing")
        report: dict[str, Any] = {}
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    config_path = ROOT / "configs" / "moecot_language_arm_training.json"
    checks = {
        "qualification_config_bound": bool(qualification_config)
        and report.get("config_sha256") == sha256_file(qualification_config_path),
        "training_config_bound": bool(report)
        and report.get("training_config_sha256") == sha256_file(config_path),
        "mechanics_green": report.get("mechanics_trigger_state") == "GREEN",
        "causal_adequacy_green": report.get("causal_adequacy_trigger_state")
        == "GREEN",
        "semantic_panel_complete": report.get("semantic_panel_complete") is True,
        "canonical_long_training_authorized": report.get(
            "canonical_long_training_authorized"
        )
        is True,
        "no_public_training_rows": int(report.get("public_training_rows_written") or 0)
        == 0,
        "no_external_inference": int(report.get("external_inference_calls") or 0)
        == 0,
        "no_fallback": int(report.get("fallback_return_count") or 0) == 0,
    }
    gaps.extend(key for key, passed in checks.items() if not passed)
    return {
        "authorized": not gaps,
        "checks": checks,
        "gaps": sorted(set(gaps)),
        "qualification_config": configured,
        "qualification_report": report_value,
        "qualification_receipt_sha256": str(report.get("receipt_sha256") or ""),
    }


class RaggedRows:
    """Immutable row store that pads only the mini-batch selected by the trainer."""

    def __init__(
        self,
        rows: list[np.ndarray],
        *,
        dtype: Any,
        standard_width: int = 8192,
    ) -> None:
        self._rows = tuple(np.asarray(row, dtype=dtype) for row in rows)
        self.dtype = np.dtype(dtype)
        self.standard_width = int(standard_width)
        self.shape = (
            len(self._rows),
            max((int(row.shape[0]) for row in self._rows), default=1),
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: Any) -> np.ndarray:
        if isinstance(index, (int, np.integer)):
            return self._rows[int(index)]
        indices = [int(value) for value in index]
        width = max((len(self._rows[value]) for value in indices), default=1)
        batch = np.zeros((len(indices), width), dtype=self.dtype)
        for row_index, source_index in enumerate(indices):
            row = self._rows[source_index]
            batch[row_index, : len(row)] = row
        return batch

    def sum(self, axis: int | None = None) -> Any:
        if axis is None:
            return sum((row.sum() for row in self._rows), start=0)
        if axis == 1:
            return np.asarray([row.sum() for row in self._rows])
        raise ValueError(f"RaggedRows only supports axis=None or axis=1, got {axis}")

    def length_bucketed_order(
        self,
        *,
        seed: int,
        probabilities: np.ndarray | None,
        minimum_stratum_coverage: bool = False,
    ) -> list[int]:
        if probabilities is None:
            sampled = list(range(len(self._rows)))
            random.Random(seed).shuffle(sampled)
        else:
            sampled = stratified_low_variance_sampling_order(
                probabilities,
                row_count=len(self._rows),
                seed=seed,
                minimum_stratum_coverage=minimum_stratum_coverage,
            )
        buckets: dict[int, list[int]] = {}
        for index in sampled:
            width = len(self._rows[int(index)])
            bucket = 0 if width <= self.standard_width else 1
            buckets.setdefault(bucket, []).append(int(index))
        bucket_order = sorted(buckets)
        random.Random(seed ^ 0x4B455243).shuffle(bucket_order)
        return [index for bucket in bucket_order for index in buckets[bucket]]

    def batch_indices(
        self, order: list[int], *, maximum_batch_size: int
    ) -> list[list[int]]:
        batches: list[list[int]] = []
        index = 0
        while index < len(order):
            width = len(self._rows[order[index]])
            size = 1 if width > self.standard_width else maximum_batch_size
            batch = order[index : index + size]
            if any(
                (len(self._rows[row]) > self.standard_width) !=
                (width > self.standard_width)
                for row in batch
            ):
                batch = [order[index]]
            batches.append(batch)
            index += len(batch)
        return batches

    @property
    def physical_bytes(self) -> int:
        return sum(int(row.nbytes) for row in self._rows)


def token_supervised_row_indices(loss_mask: Any) -> np.ndarray:
    """Return only rows with generator-token training authority."""

    return np.asarray(
        [
            index
            for index in range(len(loss_mask))
            if bool(np.asarray(loss_mask[index]).any())
        ],
        dtype=np.int64,
    )


def bound_supervision_stage_sequence_width(
    stage: Any,
    *,
    maximum_sequence_tokens: int,
    maximum_supervised_sequence_tokens: int | None = None,
    required_kerc_coverage_labels: tuple[str, ...] | None = None,
) -> Any:
    """Retain the widest empirically qualified rows without hiding exclusions."""

    if maximum_sequence_tokens <= 0:
        raise ValueError("training sequence bound must be positive")
    row_count = len(stage.inputs)
    def active_width(index: int) -> int:
        fields = [
            np.asarray(getattr(stage, name)[index])
            for name in ("inputs", "labels", "mask", "loss_mask")
            if getattr(stage, name, None) is not None
        ]
        active = np.flatnonzero(
            np.logical_or.reduce([value != 0 for value in fields])
        )
        return int(active[-1] + 1) if len(active) else 1

    widths = [active_width(index) for index in range(row_count)]
    supervised = [
        bool(np.asarray(stage.loss_mask[index]).sum() > 0)
        for index in range(row_count)
    ]
    supervised_limit = int(
        maximum_supervised_sequence_tokens or maximum_sequence_tokens
    )
    if supervised_limit <= 0 or supervised_limit > maximum_sequence_tokens:
        raise ValueError(
            "supervised training sequence bound must be positive and no wider than the global bound"
        )
    selected = [
        index for index, width in enumerate(widths)
        if width <= maximum_sequence_tokens
        and (not supervised[index] or width <= supervised_limit)
    ]
    if not selected:
        raise ValueError("training sequence bound excludes every staged row")
    coverage_rows = getattr(stage, "kerc_coverage_labels", None)
    retained_labels = (
        {label for index in selected for label in coverage_rows[index]}
        if coverage_rows is not None
        else set()
    )
    required_coverage = set(
        required_kerc_coverage_labels
        if required_kerc_coverage_labels is not None
        else KERC_CANARY_REQUIRED_COVERAGE
    )
    if coverage_rows is not None:
        missing = sorted(required_coverage - retained_labels)
        if missing:
            raise ValueError(
                "training sequence bound loses KERC coverage: " + ",".join(missing)
            )
    values: dict[str, Any] = {}
    for name, value in vars(stage).items():
        if name == "receipt":
            receipt = copy.deepcopy(value)
            receipt["training_sequence_resource_selection"] = {
                "policy": "project_theseus_empirically_qualified_training_width_v1",
                "maximum_sequence_tokens": int(maximum_sequence_tokens),
                "maximum_supervised_sequence_tokens": supervised_limit,
                "qualification_basis": (
                    "matched_guarded_full_objective_sequence_envelope"
                ),
                "original_row_count": row_count,
                "selected_row_count": len(selected),
                "excluded_row_count": row_count - len(selected),
                "original_maximum_sequence_tokens": max(widths),
                "selected_maximum_sequence_tokens": max(widths[index] for index in selected),
                "selected_maximum_supervised_sequence_tokens": max(
                    [
                        widths[index]
                        for index in selected
                        if supervised[index]
                    ]
                    or [0]
                ),
                "excluded_supervised_row_count": sum(
                    supervised[index] and index not in set(selected)
                    for index in range(row_count)
                ),
                "excluded_supervised_minimum_sequence_tokens": min(
                    [
                        widths[index]
                        for index in range(row_count)
                        if supervised[index] and index not in set(selected)
                    ]
                    or [0]
                ),
                "excluded_minimum_sequence_tokens": min(
                    [width for width in widths if width > maximum_sequence_tokens]
                    or [0]
                ),
                "selected_indices_sha256": hashlib.sha256(
                    np.asarray(selected, dtype=np.int64).tobytes()
                ).hexdigest(),
                "required_kerc_coverage_preserved": (
                    coverage_rows is None
                    or required_coverage.issubset(retained_labels)
                ),
                "required_kerc_coverage_labels": sorted(required_coverage),
                "longer_row_disposition": (
                    "K7_MAXIMUM_STRESS_REMAINS_ACTIVE_BLOCKER_NOT_CAPABILITY_NEGATIVE"
                ),
            }
            values[name] = receipt
        elif isinstance(value, RaggedRows):
            values[name] = RaggedRows(
                [np.asarray(value[index]) for index in selected],
                dtype=value.dtype,
                standard_width=value.standard_width,
            )
        elif isinstance(value, np.ndarray) and len(value) == row_count:
            values[name] = np.asarray(value[selected])
        elif isinstance(value, tuple) and len(value) == row_count:
            values[name] = tuple(value[index] for index in selected)
        else:
            values[name] = value
    return SimpleNamespace(**values)


def kerc_overfit_batch_schedule(
    *,
    row_count: int,
    single_objective_warmup_steps: int,
    fixed_objective_index: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return a hard-stage-first warmup followed by the joint objective batch."""

    if single_objective_warmup_steps < 0:
        raise ValueError("KERC single-objective warmup steps cannot be negative")
    if fixed_objective_index is not None:
        if single_objective_warmup_steps:
            raise ValueError("KERC fixed-objective and warmup schedules are exclusive")
        if row_count != 4 or fixed_objective_index not in range(row_count):
            raise ValueError("KERC fixed-objective schedule requires an index in [0, 3]")
        return ((int(fixed_objective_index),),)
    if not single_objective_warmup_steps:
        return ()
    if row_count != 4:
        raise ValueError("KERC overfit curriculum requires exactly four objective rows")
    hard_stage_first = (1, 2, 3, 0)
    return tuple(
        (hard_stage_first[index % len(hard_stage_first)],)
        for index in range(single_objective_warmup_steps)
    ) + (tuple(range(row_count)),)


def kerc_objective_balanced_sample_weights(
    stage: Any,
    *,
    uniform_within_objective: bool = False,
    objective_sampling_mass: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Give retained KERC objectives explicit optimizer-sampling mass.

    The legacy policy retained per-row source weights inside each objective.
    That preserves the aggregate objective mass, but on a long run it can
    recycle a small equal-weight stratum while leaving other admitted rows
    unseen.  The coverage-safe policy makes rows uniform inside an objective;
    smooth weighted round-robin then exhausts every objective-local row cycle
    before repeating it.  A content-bound objective-mass policy may allocate
    the post-coverage residual toward measured weak objectives without
    sacrificing the positive-row coverage floor.
    """

    row_count = len(stage.inputs)
    labels_by_row = tuple(getattr(stage, "kerc_coverage_labels", ()) or ())
    base_weights = np.asarray(
        getattr(stage, "sample_weights", np.ones(row_count)), dtype=np.float64
    )
    if (
        row_count <= 0
        or len(labels_by_row) != row_count
        or len(base_weights) != row_count
        or np.any(base_weights <= 0.0)
    ):
        raise ValueError("KERC objective balancing requires positive aligned rows")
    objectives: list[str] = []
    for labels in labels_by_row:
        matched = [
            label.split(":", 1)[1]
            for label in labels
            if label.startswith("objective:")
        ]
        if len(matched) != 1 or matched[0] not in TRAINING_TASK_TAGS:
            raise ValueError("KERC rows require exactly one registered objective label")
        objectives.append(matched[0])
    missing = sorted(set(TRAINING_TASK_TAGS) - set(objectives))
    if missing:
        raise ValueError(f"KERC objective balancing is missing objectives: {missing}")
    row_counts = {objective: objectives.count(objective) for objective in TRAINING_TASK_TAGS}
    base_mass = {
        objective: float(
            sum(
                base_weights[index]
                for index, observed in enumerate(objectives)
                if observed == objective
            )
        )
        for objective in TRAINING_TASK_TAGS
    }
    if objective_sampling_mass is None:
        target_mass = {objective: 1.0 for objective in TRAINING_TASK_TAGS}
    else:
        if set(objective_sampling_mass) != set(TRAINING_TASK_TAGS):
            raise ValueError(
                "KERC objective sampling mass must name every registered objective"
            )
        raw_target_mass = {
            objective: float(objective_sampling_mass[objective])
            for objective in TRAINING_TASK_TAGS
        }
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in raw_target_mass.values()
        ):
            raise ValueError("KERC objective sampling mass must be finite and positive")
        target_mass_total = float(sum(raw_target_mass.values()))
        target_mass = {
            objective: value / target_mass_total
            for objective, value in raw_target_mass.items()
        }
    weights = np.asarray(
        [
            (
                target_mass[objective] / float(row_counts[objective])
                if uniform_within_objective
                else (
                    target_mass[objective]
                    * float(base_weights[index])
                    / base_mass[objective]
                )
            )
            for index, objective in enumerate(objectives)
        ],
        dtype=np.float64,
    )
    balanced_mass = {
        objective: float(
            sum(
                weights[index]
                for index, observed in enumerate(objectives)
                if observed == objective
            )
        )
        for objective in TRAINING_TASK_TAGS
    }
    return weights, {
        "policy": (
            "project_theseus_kerc_objective_residual_allocation_v3"
            if objective_sampling_mass is not None
            else (
                "project_theseus_kerc_objective_balanced_sampling_v2"
                if uniform_within_objective
                else "project_theseus_kerc_objective_balanced_sampling_v1"
            )
        ),
        "active": True,
        "row_count": row_count,
        "objective_row_counts": row_counts,
        "objective_base_sampling_mass": base_mass,
        "objective_balanced_sampling_mass": balanced_mass,
        "objective_target_sampling_mass": target_mass,
        "objective_mass_policy": (
            "checkpoint_diagnostic_residual_allocation_after_full_row_coverage_v1"
            if objective_sampling_mass is not None
            else "equal_objective_mass_v1"
        ),
        "sampling_policy": "stratified_smooth_weighted_round_robin_v1",
        "within_objective_weight_policy": (
            "uniform_without_replacement_cycle_v1"
            if uniform_within_objective
            else "source_weight_preserving_legacy_v1"
        ),
        "replacement_sampling_after_coverage_prefix": False,
        "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
    }


def select_kerc_overfit_stage(stage: Any, *, rows_per_objective: int) -> Any:
    """Select a tiny generator-only stage for training-row learnability sanity."""

    if rows_per_objective <= 0 or not getattr(stage, "training_row_ids", None):
        raise ValueError("KERC overfit selection requires aligned training row identities")
    selected: list[int] = []
    selected_by_objective: dict[str, int] = {}
    available_objectives = [
        objective
        for objective in TRAINING_TASK_TAGS
        if any(
            f"objective:{objective}" in labels
            for labels in stage.kerc_coverage_labels
        )
    ]
    if not available_objectives:
        raise ValueError("KERC overfit stage has no generator objective")
    for objective in available_objectives:
        label = f"objective:{objective}"
        candidates = [
            index
            for index, (row_id, labels) in enumerate(
                zip(stage.training_row_ids, stage.kerc_coverage_labels)
            )
            if label in labels
            and ":verifier_negative" not in row_id
            and ":counterfactual:" not in row_id
            and bool(np.asarray(stage.mask[index]).any())
        ]
        if len(candidates) < rows_per_objective:
            raise ValueError(f"KERC overfit stage lacks generator rows for {objective}")
        ranked = sorted(
            candidates,
            key=lambda index: (
                len(np.asarray(stage.inputs[index])),
                str(stage.training_row_ids[index]),
            ),
        )[:rows_per_objective]
        selected.extend(ranked)
        selected_by_objective[objective] = len(ranked)
    values: dict[str, Any] = {}
    for name, value in vars(stage).items():
        if name == "receipt":
            receipt = copy.deepcopy(value)
            receipt.update(
                {
                    "row_count": len(selected),
                    "generator_training_row_count": len(selected),
                    "verifier_only_row_count": 0,
                    "target_positions": int(
                        sum(np.asarray(stage.mask[index]).sum() for index in selected)
                    ),
                    "weighted_loss_positions": float(
                        sum(
                            np.asarray(stage.loss_mask[index]).sum()
                            for index in selected
                        )
                    ),
                    "sampling_weight_sum": float(
                        sum(float(stage.sample_weights[index]) for index in selected)
                    ),
                    "overfit_diagnostic": {
                        "policy": (
                            "project_theseus_kerc_four_objective_overfit_v1"
                            if len(available_objectives) == 4
                            else "project_theseus_kerc_filtered_objective_overfit_v1"
                        ),
                        "active": True,
                        "rows_per_objective": int(rows_per_objective),
                        "selected_by_objective": selected_by_objective,
                        "selected_row_ids_sha256": hashlib.sha256(
                            "\n".join(
                                str(stage.training_row_ids[index])
                                for index in selected
                            ).encode()
                        ).hexdigest(),
                        "selection_uses_model_outcomes": False,
                        "capability_claim": "NONE_TRAINING_ROW_LEARNABILITY_ONLY",
                    },
                }
            )
            values[name] = receipt
        elif isinstance(value, RaggedRows):
            values[name] = RaggedRows(
                [np.asarray(value[index]) for index in selected],
                dtype=value.dtype,
                standard_width=value.standard_width,
            )
        elif isinstance(value, np.ndarray) and len(value) == len(stage.inputs):
            values[name] = np.asarray(value[selected])
        elif isinstance(value, tuple) and len(value) == len(stage.inputs):
            values[name] = tuple(value[index] for index in selected)
        else:
            values[name] = value
    return SimpleNamespace(**values)

SHARED_TRUNK_ID = "shared_trunk"
CONTROL_IDS = ("dense_total_parameter", "dense_active_parameter")
KERC_ENGLISH_ID = "english_kerc"
SURFACE_ENGLISH_CONTROL_ID = "english_surface_control"
ENGLISH_COMPARISON_IDS = (SURFACE_ENGLISH_CONTROL_ID, KERC_ENGLISH_ID)
KERC_CANARY_REQUIRED_COVERAGE = (
    *(f"objective:{objective}" for objective in TRAINING_TASK_TAGS),
    "decision:ANSWER",
    "decision:CLARIFY",
    "decision:ABSTAIN",
    "interaction:present",
    "interaction:absent",
    "residual:interaction:active",
    "residual:segment:active",
    "residual:token:active",
    "residual:exact:active",
    "verifier:positive",
    *(f"verifier:negative:{dimension}" for dimension in KERC_VERIFIER_DIMENSIONS),
    "verifier:counterfactual:context_withheld",
    "verifier:counterfactual:context_shuffled",
)


def bounded_kerc_coverage_required(target: dict[str, Any], stage: Any) -> bool:
    """Require set-cover batching for every bounded KERC lease, not only tiny canaries."""

    return (
        str(target.get("role") or "") == "kerc_english_candidate"
        and stage is not None
        and bool(
            ((stage.receipt or {}).get("bounded_selection") or {}).get("active")
        )
        and not bool(
            ((stage.receipt or {}).get("overfit_diagnostic") or {}).get("active")
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--guarded",
        action="store_true",
        help="Launch a heavy operation under the canonical external host watchdog.",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=[SHARED_TRUNK_ID, *ARM_IDS, *CONTROL_IDS, *ENGLISH_COMPARISON_IDS],
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--campaign-segment",
        action="store_true",
        help=(
            "Run one qualified fresh-process shared-trunk pretraining segment. "
            "Requires exact resume and a positive bounded --max-steps."
        ),
    )
    parser.add_argument(
        "--architecture-candidate-id",
        default="",
        help="Candidate-specific T0A canary id; required for bounded pre-freeze optimizer work.",
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=0,
        help="Bind one preregistered seed for an isolated canonical candidate run.",
    )
    parser.add_argument(
        "--candidate-initialization-state",
        default="",
        help=(
            "Optional candidate-only JSON custody state used to align a matched "
            "pair across separately watchdog-isolated target processes."
        ),
    )
    parser.add_argument(
        "--optimizer-id",
        choices=sorted(pretraining_optimizers.OPTIMIZER_IDS),
        default="",
        help="Optimizer implementation for an optimizer_adequacy scratch canary.",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "pretraining", "source_conditioned_pretraining", "kernel_english", "supervision"),
        default="all",
        help="Run the full ordered curriculum or one canonical phase for a bounded mechanics canary.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--candidate-continuation-report",
        default="",
        help=(
            "Import one content-bound terminal candidate receipt into a fresh "
            "scratch namespace for a governed one-shot continuation. This is "
            "not general candidate scratch resume authority."
        ),
    )
    parser.add_argument(
        "--scratch-checkpoint-root",
        default="",
        help=(
            "Write a bounded non-resumable canary outside the registered checkpoint "
            "lineage. Requires --execute, --architecture-candidate-id, and a positive "
            "candidate-authorized --max-steps."
        ),
    )
    parser.add_argument(
        "--evaluate-progress",
        action="store_true",
        help="Measure source-disjoint private-development loss for an incomplete checkpoint.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="",
        help="Optional earlier checkpoint for a matched learning-curve comparison.",
    )
    parser.add_argument(
        "--migrate-shared-trunk-checkpoint-format",
        action="store_true",
        help=(
            "Atomically migrate the registered shared-trunk model from its qualified "
            "legacy container to the configured checkpoint format without training."
        ),
    )
    args = parser.parse_args()
    heavy_operation = training_operation(args)
    if args.guarded and not heavy_operation:
        parser.error("--guarded requires training, evaluation, or checkpoint migration")
    if args.migrate_shared_trunk_checkpoint_format and (
        args.execute
        or args.evaluate_progress
        or args.resume
        or args.target
        or args.max_steps
        or args.scratch_checkpoint_root
        or args.baseline_checkpoint
        or args.phase != "all"
    ):
        parser.error(
            "--migrate-shared-trunk-checkpoint-format cannot be combined with "
            "training, evaluation, target, phase, or scratch options"
        )
    if args.evaluate_progress and args.execute:
        parser.error("--evaluate-progress and --execute are mutually exclusive")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if (args.execute or args.evaluate_progress) and not args.target:
        parser.error("execution or progress evaluation requires at least one explicit --target")
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    if args.campaign_segment and (
        not args.execute
        or not args.resume
        or args.max_steps <= 0
        or args.architecture_candidate_id
        or args.scratch_checkpoint_root
        or args.evaluate_progress
        or args.phase != "pretraining"
        or set(args.target or []) != {SHARED_TRUNK_ID}
    ):
        parser.error(
            "--campaign-segment requires resumed shared_trunk pretraining "
            "execution with a positive --max-steps and no candidate scratch"
        )
    if bool(args.scratch_checkpoint_root) != bool(args.architecture_candidate_id):
        parser.error(
            "--scratch-checkpoint-root and --architecture-candidate-id are required together"
        )
    if bool(args.scratch_checkpoint_root) != bool(args.candidate_seed):
        parser.error(
            "--candidate-seed is required exactly when candidate scratch training is requested"
        )
    if args.candidate_initialization_state and not args.architecture_candidate_id:
        parser.error(
            "--candidate-initialization-state requires an architecture candidate"
        )
    segmented_candidate_resume = bool(
        args.resume
        and args.architecture_candidate_id == "rdc_kerc_k5_adequacy"
    )
    if args.candidate_continuation_report and (
        not args.execute
        or (args.resume and not segmented_candidate_resume)
        or not args.scratch_checkpoint_root
        or args.architecture_candidate_id
        not in {"rdc_kerc_k5_adequacy", "rdc_kerc_k5_overfit"}
        or args.candidate_seed <= 0
        or args.phase != "kernel_english"
        or set(args.target or []) != {KERC_ENGLISH_ID}
    ):
        parser.error(
            "--candidate-continuation-report requires a fresh or governed segmented-resume "
            "rdc_kerc_k5 adequacy/overfit scratch execution for only english_kerc, "
            "one bound seed, and the kernel_english phase"
        )
    if args.scratch_checkpoint_root and (
        not args.execute
        or (args.resume and not segmented_candidate_resume)
        or args.max_steps <= 0
    ):
        parser.error(
            "candidate scratch training requires fresh execution or the governed "
            "K5 segmented-resume path with positive --max-steps"
        )
    if args.optimizer_id and (
        args.architecture_candidate_id != "optimizer_adequacy"
        or not args.scratch_checkpoint_root
    ):
        parser.error(
            "--optimizer-id is restricted to an optimizer_adequacy scratch canary"
        )

    config_path = resolve(args.config)
    raw_config = read_json(config_path)
    if heavy_operation:
        accelerator_authorized = host_resource_safety.accelerator_child_authorized()
        training_authorized = (
            os.environ.get("THESEUS_GUARDED_TRAINING_CHILD") == "1"
        )
        if args.guarded:
            if accelerator_authorized or training_authorized:
                parser.error("nested guarded training launch is forbidden")
            return launch_guarded_training(
                raw_config,
                heavy_operation,
                candidate_id=args.architecture_candidate_id,
            )
        if args.architecture_candidate_id:
            authorized = accelerator_authorized
        else:
            authorized = accelerator_authorized and training_authorized
        if not authorized:
            parser.error(
                "heavy MLX execution requires --guarded or the replacement freeze shard runner"
            )
    config = bind_scale_preregistration(raw_config)
    preissued_candidate_authority = None
    if args.execute and args.max_steps > 0:
        preissued_candidate_authority = architecture_training_authority(
            config,
            max_steps=args.max_steps,
            candidate_id=args.architecture_candidate_id,
            scratch_checkpoint_root=args.scratch_checkpoint_root,
            targets=list(dict.fromkeys(args.target or [])),
            phase=args.phase,
            resume=args.resume,
            candidate_seed=args.candidate_seed,
            campaign_segment=args.campaign_segment,
        )
        candidate_lease = (preissued_candidate_authority or {}).get(
            "candidate_lease"
        ) or {}
        execution_policy = candidate_lease.get("execution_policy") or {}
        if (
            execution_policy.get("require_external_watchdog") is True
            and os.environ.get("THESEUS_GUARDED_ACCELERATOR_CHILD") != "1"
        ):
            parser.error(
                "this accelerator canary must be launched by the Theseus host "
                "resource watchdog; direct execution is denied"
            )
    config = bind_candidate_canary_overlay(
        config,
        (preissued_candidate_authority or {}).get("candidate_lease"),
    )
    plan = build_plan(
        config,
        config_path=config_path,
        candidate_lease=(preissued_candidate_authority or {}).get("candidate_lease"),
    )
    if plan["trigger_state"] == "RED":
        write_json(resolve(args.out or config["report"]), plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 2
    unavailable_targets = [
        target_id
        for target_id in list(dict.fromkeys(args.target or []))
        if target_id not in (plan.get("targets") or {})
    ]
    if unavailable_targets:
        parser.error(
            "targets are not executable in the current campaign: "
            + ",".join(unavailable_targets)
        )
    if args.migrate_shared_trunk_checkpoint_format:
        report = migrate_shared_trunk_checkpoint_format(config, plan)
        write_json(
            resolve(args.out) if args.out else DEFAULT_CHECKPOINT_FORMAT_MIGRATION_REPORT,
            report,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    report = plan
    if args.evaluate_progress:
        report = evaluate_training_progress(
            config,
            plan,
            targets=list(dict.fromkeys(args.target or [])),
            baseline_checkpoint=args.baseline_checkpoint,
        )
    elif args.execute:
        authority = preissued_candidate_authority or architecture_training_authority(
            config,
            max_steps=args.max_steps,
            candidate_id=args.architecture_candidate_id,
            scratch_checkpoint_root=args.scratch_checkpoint_root,
            targets=list(dict.fromkeys(args.target or [])),
            phase=args.phase,
            resume=args.resume,
            candidate_seed=args.candidate_seed,
            campaign_segment=args.campaign_segment,
        )
        if authority["trigger_state"] != "GREEN":
            report = {
                **plan,
                "trigger_state": "RED",
                "hard_gaps": list(plan.get("hard_gaps") or [])
                + ["pre_training_architecture_authority_denied"],
                "architecture_training_authority": authority,
            }
            write_json(resolve(args.out or config["report"]), report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 2
        report = execute_targets(
            config,
            plan,
            config_path=resolve(args.config),
            targets=list(dict.fromkeys(args.target or [])),
            max_steps=args.max_steps,
            resume=args.resume,
            training_phase=args.phase,
            scratch_checkpoint_root=(
                resolve(args.scratch_checkpoint_root)
                if args.scratch_checkpoint_root
                else None
            ),
            candidate_lease=(authority.get("candidate_lease") or None),
            optimizer_id=args.optimizer_id,
            candidate_initialization_state_path=(
                resolve(args.candidate_initialization_state)
                if args.candidate_initialization_state
                else None
            ),
            candidate_continuation_report_path=(
                resolve(args.candidate_continuation_report)
                if args.candidate_continuation_report
                else None
            ),
        )
    write_json(resolve(args.out or config["report"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def architecture_training_authority(
    config: dict[str, Any],
    *,
    max_steps: int,
    candidate_id: str = "",
    scratch_checkpoint_root: str | Path = "",
    targets: list[str] | None = None,
    phase: str = "all",
    resume: bool = False,
    candidate_seed: int = 0,
    campaign_segment: bool = False,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Permit content-bound candidate canaries, but gate long optimizer spend."""

    cfg = config.get("architecture_training_authority")
    if not isinstance(cfg, dict) or cfg.get("policy") != (
        "project_theseus_pre_training_architecture_authority_v1"
    ):
        return {
            "policy": "project_theseus_pre_training_architecture_authority_v1",
            "trigger_state": "RED",
            "authority": "DENIED",
            "reason": "architecture_training_authority_contract_missing",
        }
    canary_cap = int(cfg.get("pre_training_canary_max_steps") or 0)
    if max_steps > 0:
        if campaign_segment:
            segment = dict(cfg.get("fresh_process_segments") or {})
            report_path = resolve(
                str(segment.get("qualification_report") or "")
            )
            report = (
                read_json(report_path)
                if report_path.is_file()
                else {}
            )
            required_segments = int(
                segment.get("minimum_qualified_contiguous_segments") or 0
            )
            segment_valid = bool(
                segment.get("policy")
                == "project_theseus_bounded_fresh_process_pretraining_v1"
                and segment.get("target_id") == SHARED_TRUNK_ID
                and segment.get("phase") == "pretraining"
                and int(segment.get("maximum_optimizer_steps") or 0) > 0
                and segment.get("compute_dtype") == "float32"
                and segment.get("fp32_master") is False
                and int(segment.get("compiled_microbatch_size") or 0) == 4
                and segment.get("resume_required") is True
                and segment.get("require_external_watchdog") is True
            )
            request_valid = bool(
                not candidate_id
                and not scratch_checkpoint_root
                and list(targets or []) == [SHARED_TRUNK_ID]
                and phase == "pretraining"
                and resume
                and 0 < max_steps
                <= int(segment.get("maximum_optimizer_steps") or 0)
            )
            qualification_valid = bool(
                report.get("policy")
                == "project_theseus_fresh_process_pretraining_qualification_v1"
                and report.get("trigger_state") == "GREEN"
                and int(report.get("contiguous_segment_count") or 0)
                >= required_segments
                and report.get("qualified_execution_policy") == segment
                and report.get("canonical_lineage_unchanged") is True
                and report.get("exact_resume_validation") is True
                and report.get(
                    "independent_segmented_replay_numeric_parity"
                )
                is True
                and report.get("zero_swap_growth") is True
            )
            authorized = segment_valid and request_valid and qualification_valid
            return {
                "policy": cfg["policy"],
                "trigger_state": "GREEN" if authorized else "RED",
                "authority": (
                    "QUALIFIED_FRESH_PROCESS_CAMPAIGN_SEGMENT"
                    if authorized
                    else "DENIED"
                ),
                "maximum_steps": max_steps,
                "long_optimizer_run_authorized": bool(authorized),
                "fresh_process_segment_policy": segment,
                "qualification_report": relative(report_path),
                "qualification_report_sha256": (
                    sha256_file(report_path) if report_path.is_file() else ""
                ),
                "reason": (
                    ""
                    if authorized
                    else "fresh_process_segment_qualification_or_request_invalid"
                ),
            }
        if not candidate_id:
            return {
                "policy": cfg["policy"],
                "trigger_state": "RED",
                "authority": "DENIED",
                "reason": "candidate_specific_canary_lease_required",
                "legacy_generic_canary_cap": canary_cap,
            }
        lease = pretraining_candidate_canary.candidate_lease(
            candidate_id=candidate_id,
            max_steps=max_steps,
            scratch_checkpoint_root=scratch_checkpoint_root,
            targets=list(targets or []),
            phase=phase,
            resume=resume,
            selected_seed=(candidate_seed if candidate_seed > 0 else None),
        )
        return {
            "policy": cfg["policy"],
            "trigger_state": "GREEN" if lease["authorized"] else "RED",
            "authority": (
                "CANDIDATE_SPECIFIC_ARCHITECTURE_CANARY"
                if lease["authorized"]
                else "DENIED"
            ),
            "maximum_steps": max_steps,
            "legacy_generic_canary_cap": canary_cap,
            "long_optimizer_run_authorized": False,
            "candidate_lease": lease,
            "reason": "" if lease["authorized"] else "candidate_specific_canary_lease_denied",
        }
    command = [str(value) for value in cfg.get("gate_command") or []]
    if cfg.get("required_for_long_optimizer_runs") is not True or not command:
        return {
            "policy": cfg["policy"],
            "trigger_state": "RED",
            "authority": "DENIED",
            "reason": "long_optimizer_gate_contract_invalid",
        }
    completed = runner(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "policy": cfg["policy"],
        "trigger_state": "GREEN" if completed.returncode == 0 else "RED",
        "authority": (
            "ARCHITECTURE_FREEZE_GREEN"
            if completed.returncode == 0
            else "DENIED"
        ),
        "maximum_steps": max_steps,
        "canary_cap": canary_cap,
        "long_optimizer_run_authorized": completed.returncode == 0,
        "gate_command": command,
        "gate_exit_code": int(completed.returncode),
        "gate_output_tail": (completed.stdout or completed.stderr or "")[-2000:],
    }


def bind_scale_preregistration(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve one preregistered model owner into the executable training config."""

    reference = config.get("scale_preregistration")
    if not isinstance(reference, dict):
        return config
    prereg_path = resolve(str(reference.get("config") or ""))
    if not prereg_path.is_file():
        raise ValueError("scale preregistration config is missing")
    prereg = read_json(prereg_path)
    required_policy = str(reference.get("required_policy") or "")
    candidate = prereg.get("candidate") if isinstance(prereg.get("candidate"), dict) else {}
    if prereg.get("policy") != required_policy:
        raise ValueError("scale preregistration policy mismatch")
    if candidate.get("id") != reference.get("candidate_id"):
        raise ValueError("scale preregistration candidate mismatch")
    if candidate.get("expert_trainable_scope") != (
        (config.get("topology") or {}).get("expert_trainable_scope")
    ):
        raise ValueError("scale preregistration expert scope mismatch")

    bound = copy.deepcopy(config)
    for key in ("shared_trunk_model", "arm_model"):
        declared = bound.get(key)
        selected = candidate.get(key)
        if not isinstance(selected, dict):
            raise ValueError(f"scale preregistration is missing {key}")
        if declared is not None and declared != selected:
            raise ValueError(f"duplicate executable {key} disagrees with preregistration")
        bound[key] = copy.deepcopy(selected)
    topology = bound.get("topology") or {}
    for field in ("expert_adapter_dim", "source_expert_adapter_dim"):
        selected = int((candidate.get("arm_model") or {}).get(field) or 0)
        if int(topology.get(field) or 0) != selected:
            raise ValueError(f"topology {field} disagrees with preregistration")

    # A deferred KERC model shares the selected trunk shape but retains its own
    # explicitly registered heads. It receives no first-campaign optimizer credit.
    if isinstance(bound.get("kerc_english_model"), dict):
        kerc_only = {
            key: value
            for key, value in bound["kerc_english_model"].items()
            if key not in bound["shared_trunk_model"]
        }
        bound["kerc_english_model"] = {
            **copy.deepcopy(bound["shared_trunk_model"]),
            **kerc_only,
        }
    bound["_resolved_scale_preregistration"] = {
        "config": relative(prereg_path),
        "config_sha256": sha256_file(prereg_path),
        "candidate_id": str(candidate["id"]),
    }
    return bound


def bind_candidate_canary_overlay(
    config: dict[str, Any], candidate_lease: dict[str, Any] | None
) -> dict[str, Any]:
    """Activate a deferred candidate only inside its bounded scratch lease.

    KERC was removed from the first long campaign after K0-K3, but its frozen
    source-disjoint stage remains bound to the earlier faithful-candidate
    contract. Reconstructing that exact contract in memory lets K4-K8 canaries
    use the canonical trainer without mutating the production disposition or
    weakening the stage-manifest identity check.
    """

    if candidate_lease is None or candidate_lease.get("candidate_id") not in {
        "rdc_kerc_adequacy",
        "rdc_kerc_k5_adequacy",
        "rdc_kerc_k5_overfit",
    }:
        return config
    pretraining_candidate_canary.validate_lease(candidate_lease)
    existing = config.get("_candidate_canary_overlay") or {}
    if existing.get("lease_digest") == candidate_lease.get("lease_digest"):
        return config
    candidate_targets = list(candidate_lease.get("targets") or [])
    if (
        not candidate_targets
        or not set(candidate_targets).issubset(set(ENGLISH_COMPARISON_IDS))
        or candidate_lease.get("phase") != "kernel_english"
    ):
        raise ValueError(
            "RDC/KERC candidate overlay requires a KERC matched-comparison target"
        )

    bound = copy.deepcopy(config)
    kernel = bound["kernel_english_training"]
    records = dict(kernel.get("deferred_candidate_records_by_split") or {})
    if not records or any(int(value) <= 0 for value in records.values()):
        raise ValueError("RDC/KERC candidate overlay requires frozen candidate records")
    prior = dict(kernel.get("disposition") or {})
    kernel["required"] = True
    kernel["records_by_split"] = records
    kernel.pop("deferred_candidate_records_by_split", None)
    kernel["disposition"] = {
        "policy": "project_theseus_kerc_pretraining_disposition_v1",
        "state": "CANDIDATE_REQUIRED",
        "qualification_scope": "faithful_full_compiler_core_renderer_candidate",
        "basis": "adequacy_audit_reopened_after_toy_proxy",
        "full_kerc_training_enabled": True,
        "general_kerc_falsification_claimed": False,
        "learned_capability_claimed": False,
        "retained_mechanisms": [],
        "superseded_proxy_evidence": copy.deepcopy(
            prior.get("superseded_proxy_evidence") or {}
        ),
        "non_claims": list(prior.get("non_claims") or []),
    }
    execution_policy = dict(candidate_lease.get("execution_policy") or {})
    compiler_transport_override = execution_policy.get("kerc_compiler_transport")
    if compiler_transport_override is not None:
        if (
            candidate_lease.get("candidate_id")
            not in {"rdc_kerc_k5_adequacy", "rdc_kerc_k5_overfit"}
            or compiler_transport_override
            != {
                "policy": LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_POLICY,
                "version": LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION,
                "source_authority": "generator_visible_prompt.source_surface_only",
                "materializer_generation_credit": 0,
                "materializer_capability_credit": 0,
            }
        ):
            raise ValueError("candidate compiler transport override is invalid")
        bound["kerc_compiler_transport"] = copy.deepcopy(
            compiler_transport_override
        )
    kernel_repetitions = int(
        execution_policy.get("kernel_optimizer_repetitions") or 1
    )
    if kernel_repetitions <= 0 or kernel_repetitions > int(
        bound["training"]["maximum_kernel_english_optimizer_repetitions"]
    ):
        raise ValueError("candidate kernel optimizer repetitions exceed the frozen cap")
    bound["training"]["kernel_english_optimizer_repetitions"] = kernel_repetitions
    candidate_decode_envelopes = dict(
        execution_policy.get(
            "maximum_supervised_training_sequence_tokens_by_target"
        )
        or {}
    )
    if KERC_ENGLISH_ID in candidate_decode_envelopes:
        bound["evaluation"]["kerc_decode_max_target_tokens"] = min(
            int(bound["evaluation"]["kerc_decode_max_target_tokens"]),
            int(candidate_decode_envelopes[KERC_ENGLISH_ID]),
        )
    comparison = bound["comparison_contract"]
    comparison["first_campaign_candidate_ids"] = [
        SHARED_TRUNK_ID,
        *ARM_IDS,
        *CONTROL_IDS,
        *ENGLISH_COMPARISON_IDS,
    ]
    comparison["required_views"] = [
        *comparison.get("required_views", []),
        *comparison.pop("deferred_kerc_views", []),
    ]
    comparison["required_metrics"] = [
        *comparison.get("required_metrics", []),
        *comparison.pop("deferred_kerc_metrics", []),
    ]
    bound["_candidate_canary_overlay"] = {
        "policy": "project_theseus_candidate_config_overlay_v1",
        "candidate_id": "rdc_kerc_adequacy",
        "lease_digest": candidate_lease["lease_digest"],
        "production_disposition_mutated": False,
        "stage_identity_relaxed": False,
        "execution_policy": execution_policy,
    }
    return bound


def build_plan(
    config: dict[str, Any],
    *,
    config_path: Path,
    candidate_lease: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = bind_candidate_canary_overlay(config, candidate_lease)
    config = bind_scale_preregistration(config)
    gaps: list[str] = []
    validate_config(config)
    base_path = resolve(str(config["base_config"]))
    base = read_json(base_path)
    scale_audit = audit_scale_preregistration(config)
    scale_gaps = list(scale_audit["hard_gaps"])
    candidate_scale_deviations: list[str] = []
    if candidate_lease is not None:
        pretraining_candidate_canary.validate_lease(candidate_lease)
        if candidate_lease.get("authorized") is not True:
            raise ValueError("candidate plan requires an authorized lease")
        candidate_scale_deviations = [
            gap
            for gap in scale_gaps
            if gap.startswith("scale_preregistration_input_stale:")
            or gap == "scale_preregistration_report_config_identity_mismatch"
        ]
        scale_gaps = [
            gap for gap in scale_gaps if gap not in candidate_scale_deviations
        ]
        scale_audit["production_hard_gaps_acknowledged_by_candidate_lease"] = (
            candidate_scale_deviations
        )
        scale_audit["candidate_lease_digest"] = candidate_lease["lease_digest"]
        scale_audit["state"] = "CANDIDATE_SCRATCH_ONLY"
    gaps.extend(scale_gaps)
    stage_dir = resolve(str(config["stage_dir"]))
    metadata_path = stage_dir / "stage_metadata_v1.json"
    if not metadata_path.is_file():
        gaps.append("canonical_stage_metadata_missing")
        metadata: dict[str, Any] = {}
    else:
        metadata = read_json(metadata_path)
    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    canonical = (
        summary.get("canonical_pretrain_stage")
        if isinstance(summary.get("canonical_pretrain_stage"), dict)
        else {}
    )
    scale_stage_audit = audit_scale_stage_contract(
        config, base, canonical, scale_audit=scale_audit
    )
    scale_audit.update(scale_stage_audit)
    gaps.extend(scale_stage_audit["hard_gaps"])
    arm_views = canonical.get("arm_views") if isinstance(canonical.get("arm_views"), dict) else {}
    range_audit = audit_arm_views(arm_views, int(canonical.get("window_count") or 0))
    gaps.extend(range_audit["hard_gaps"])
    tokenizer_audit = audit_tokenizer_stage(base, canonical)
    gaps.extend(tokenizer_audit["hard_gaps"])
    supervision_audit = audit_supervision_stage(config, config_path=config_path)
    gaps.extend(supervision_audit["hard_gaps"])
    source_conditioned_audit = audit_source_conditioned_stage(config)
    gaps.extend(source_conditioned_audit["hard_gaps"])
    kernel_english_audit = audit_kernel_english_stage(config)
    gaps.extend(kernel_english_audit["hard_gaps"])
    stage_arrays = canonical.get("array_artifacts") if isinstance(canonical.get("array_artifacts"), dict) else {}
    for key, row in stage_arrays.items():
        path = resolve(str(row.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(row.get("sha256") or ""):
            gaps.append(f"canonical_stage_array_identity_mismatch:{key}")

    models: dict[str, Any] = {}
    if metadata:
        models = model_accounting(config, base, metadata)
        scale_model_audit = audit_scale_model_accounting(config, models, scale_audit)
        scale_audit.update(scale_model_audit)
        gaps.extend(scale_model_audit["hard_gaps"])
        dense_total = int(models["dense_total_parameter"]["parameter_count"])
        arm_total = int(models["moecot_system"]["total_parameter_count"])
        delta = abs(arm_total - dense_total) / max(1, dense_total)
        models["moecot_system"]["total_parameter_delta_vs_dense_total"] = round(delta, 8)
        if delta > 0.10:
            gaps.append("moecot_total_parameters_outside_preregistered_tolerance")
        active_reference = int(models["moecot_system"]["active_parameter_count_per_request"])
        active_delta = abs(
            int(models["dense_active_parameter"]["parameter_count"])
            - active_reference
        ) / max(1, active_reference)
        models["dense_active_parameter"]["parameter_delta_fraction"] = round(
            active_delta, 8
        )
        if active_delta > 0.01:
            gaps.append("active_parameter_control_mismatch")
    plan_identity = plan_sha256(
        config,
        metadata,
        models,
        supervision_audit,
        source_conditioned_audit,
        kernel_english_audit,
        scale_audit,
    )
    targets = target_contracts(
        config,
        arm_views,
        models,
        plan_identity,
        supervision_audit=supervision_audit,
        source_conditioned_audit=source_conditioned_audit,
        kernel_english_audit=kernel_english_audit,
    )
    specialist_scaling = audit_specialist_data_scaling(
        base,
        targets,
        models,
    )
    gaps.extend(specialist_scaling["hard_gaps"])
    candidate_exposure_deviations: list[str] = []
    for target_id, target in targets.items():
        if target.get("optimizer_repetition_ceiling_ready") is not True:
            gap = f"optimizer_repetition_ceiling_exceeded:{target_id}"
            if candidate_lease is not None and target_id in ENGLISH_COMPARISON_IDS:
                candidate_exposure_deviations.append(gap)
            else:
                gaps.append(gap)
    checkpoint_inventory = inspect_checkpoint_inventory(
        targets,
        plan_identity,
        summary.get("stage_signature"),
        plan_identity_contract=config.get("plan_identity") or {},
    )
    checkpoint_gaps = list(checkpoint_inventory["hard_gaps"])
    candidate_checkpoint_deviations: list[str] = []
    if candidate_lease is not None:
        candidate_checkpoint_deviations = checkpoint_gaps
        checkpoint_inventory[
            "production_hard_gaps_acknowledged_by_candidate_lease"
        ] = candidate_checkpoint_deviations
        checkpoint_inventory["candidate_scratch_only"] = True
        checkpoint_gaps = []
    gaps.extend(checkpoint_gaps)
    return {
        "policy": "project_theseus_moecot_language_arm_training_plan_v1",
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "mode": (
            "candidate_scratch_plan"
            if candidate_lease is not None
            else "preregistered_plan"
        ),
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "base_config": relative(base_path),
        "base_config_sha256": sha256_file(base_path),
        "stage": {
            "path": relative(stage_dir),
            "metadata": relative(metadata_path),
            "metadata_sha256": sha256_file(metadata_path) if metadata_path.is_file() else "",
            "stage_signature": summary.get("stage_signature"),
            "array_artifacts": stage_arrays,
            "arm_view_policy": arm_views.get("policy"),
            "range_audit": range_audit,
            "tokenizer_audit": tokenizer_audit,
        },
        "models": models,
        "scale_preregistration": scale_audit,
        "supervision": supervision_audit,
        "source_conditioned_pretraining": source_conditioned_audit,
        "kernel_english_training": kernel_english_audit,
        "targets": targets,
        "specialist_data_scaling": specialist_scaling,
        "checkpoint_inventory": checkpoint_inventory,
        "comparison_contract": config["comparison_contract"],
        "execution_policy": copy.deepcopy(
            (config.get("training") or {}).get("execution_policy") or {}
        ),
        "plan_identity": config.get("plan_identity") or {},
        "candidate_canary_context": {
            "active": candidate_lease is not None,
            "lease_digest": (
                str(candidate_lease.get("lease_digest") or "")
                if candidate_lease is not None
                else ""
            ),
            "production_scale_deviations": candidate_scale_deviations,
            "production_checkpoint_deviations": candidate_checkpoint_deviations,
            "production_optimizer_exposure_deviations": (
                candidate_exposure_deviations
            ),
            "production_training_authorized": False,
            "capability_claim_authorized": False,
            "config_overlay": config.get("_candidate_canary_overlay") or {},
        },
        "training_implementation_closure": training_implementation_closure(config),
        "plan_sha256": plan_identity,
        "hard_gaps": sorted(set(gaps)),
        "non_claims": [
            "plan and checkpoint smoke are not learned capability",
            "training loss is not direct answer utility",
            "routing success is not answer success",
            "neither accounting view may be selected after results are known",
        ],
        **no_cheat(config),
    }


def audit_scale_preregistration(config: dict[str, Any]) -> dict[str, Any]:
    reference = config.get("scale_preregistration")
    if not isinstance(reference, dict):
        return {
            "state": "NOT_REQUIRED",
            "candidate_id": "",
            "hard_gaps": [],
        }
    prereg_path = resolve(str(reference.get("config") or ""))
    report_path = resolve(str(reference.get("report") or ""))
    gaps: list[str] = []
    prereg = read_json(prereg_path) if prereg_path.is_file() else {}
    report = read_json(report_path) if report_path.is_file() else {}
    candidate_id = str(reference.get("candidate_id") or "")
    if not prereg:
        gaps.append("scale_preregistration_config_missing")
    if not report:
        gaps.append("scale_preregistration_report_missing")
    if prereg.get("policy") != reference.get("required_policy"):
        gaps.append("scale_preregistration_policy_mismatch")
    if (prereg.get("candidate") or {}).get("id") != candidate_id:
        gaps.append("scale_preregistration_candidate_mismatch")
    if report:
        if report.get("policy") != reference.get("required_policy"):
            gaps.append("scale_preregistration_report_policy_mismatch")
        if report.get("training_authorized") is not True:
            gaps.append("scale_preregistration_training_not_authorized")
        if report.get("proposal_state") != "AUTHORIZED_FOR_FROZEN_TRAINING_PLAN":
            gaps.append("scale_preregistration_proposal_not_authorized")
        config_ref = report.get("config") if isinstance(report.get("config"), dict) else {}
        if (
            config_ref.get("path") != relative(prereg_path)
            or config_ref.get("sha256") != sha256_file(prereg_path)
        ):
            gaps.append("scale_preregistration_report_config_identity_mismatch")
        if (report.get("architecture") or {}).get("candidate_id") != candidate_id:
            gaps.append("scale_preregistration_report_candidate_mismatch")
        for input_id, artifact in (report.get("input_artifacts") or {}).items():
            if not isinstance(artifact, dict):
                gaps.append(f"scale_preregistration_input_invalid:{input_id}")
                continue
            artifact_path = resolve(str(artifact.get("path") or ""))
            if (
                not artifact_path.is_file()
                or sha256_file(artifact_path) != str(artifact.get("sha256") or "")
            ):
                gaps.append(f"scale_preregistration_input_stale:{input_id}")
        capacity_artifact = (report.get("input_artifacts") or {}).get(
            "canonical_capacity_report"
        ) or {}
        capacity_path = resolve(str(capacity_artifact.get("path") or ""))
        if capacity_path.is_file():
            from neural_seed_50m_scale_preregistration import (
                canonical_capacity as replay_canonical_capacity,
            )

            if not replay_canonical_capacity(read_json(capacity_path))["receipt_valid"]:
                gaps.append(
                    "scale_preregistration_capacity_receipt_no_longer_replays"
                )
    evaluation_path = resolve(str(reference.get("evaluation_freeze") or ""))
    evaluation = read_json(evaluation_path) if evaluation_path.is_file() else {}
    if not evaluation:
        gaps.append("fresh_functional_evaluation_freeze_missing")
    elif (
        evaluation.get("policy")
        != "project_theseus_private_functional_utility_freeze_v2"
        or evaluation.get("immutable") is not True
        or evaluation.get("evaluation_state") != "NOT_EVALUATED"
        or evaluation.get("candidate_id") != candidate_id
        or evaluation.get("source_disjoint") is not True
        or int(evaluation.get("consumed_case_count") or 0) != 0
    ):
        gaps.append("fresh_functional_evaluation_freeze_invalid")
    checkpoint_root = resolve(str(config.get("checkpoint_root") or ""))
    if checkpoint_root.name != candidate_id:
        gaps.append("checkpoint_namespace_not_bound_to_scale_candidate")
    return {
        "policy": "project_theseus_executable_scale_binding_v1",
        "state": "GREEN" if not gaps else "RED",
        "candidate_id": candidate_id,
        "config": relative(prereg_path),
        "config_sha256": sha256_file(prereg_path) if prereg_path.is_file() else "",
        "report": relative(report_path),
        "report_sha256": sha256_file(report_path) if report_path.is_file() else "",
        "architecture": report.get("architecture") or {},
        "data_support": report.get("data_support") or {},
        "heldout_utility_contract": report.get("heldout_utility_contract") or {},
        "evaluation_freeze": relative(evaluation_path),
        "evaluation_freeze_sha256": (
            sha256_file(evaluation_path) if evaluation_path.is_file() else ""
        ),
        "evaluation_freeze_semantic_sha256": (
            evaluation_freeze_semantic_sha256(evaluation) if evaluation else ""
        ),
        "hard_gaps": gaps,
    }


def audit_scale_stage_contract(
    config: dict[str, Any],
    base: dict[str, Any],
    canonical: dict[str, Any],
    *,
    scale_audit: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(config.get("scale_preregistration"), dict):
        return {"stage_contract_state": "NOT_REQUIRED", "hard_gaps": []}
    gaps: list[str] = []
    architecture = scale_audit.get("architecture") or {}
    active_parameters = int(architecture.get("active_parameter_count_per_request") or 0)
    minimum_ratio = float(
        ((read_json(resolve(str(config["scale_preregistration"]["config"]))).get("scaling_contract") or {}).get(
            "minimum_unique_positions_per_active_parameter"
        ) or 0.0)
    )
    required_positions = int(math.ceil(active_parameters * minimum_ratio))
    staged_positions = int(canonical.get("materialized_positions") or 0)
    selected_rung = (base.get("data_model_scaling_contract") or {}).get("selected_rung") or {}
    if selected_rung.get("id") != config["scale_preregistration"].get("candidate_id"):
        gaps.append("base_scale_rung_not_bound_to_preregistered_candidate")
    if int(selected_rung.get("active_parameter_count") or 0) != active_parameters:
        gaps.append("base_scale_parameter_count_mismatch")
    if required_positions <= 0 or staged_positions < required_positions:
        gaps.append("staged_unique_position_floor_not_met_for_scale_candidate")
    return {
        "stage_contract_state": "GREEN" if not gaps else "RED",
        "required_unique_positions": required_positions,
        "staged_unique_positions": staged_positions,
        "hard_gaps": gaps,
    }


def audit_scale_model_accounting(
    config: dict[str, Any], models: dict[str, Any], scale_audit: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(config.get("scale_preregistration"), dict):
        return {"model_accounting_state": "NOT_REQUIRED", "hard_gaps": []}
    expected = scale_audit.get("architecture") or {}
    observed = models.get("moecot_system") or {}
    gaps: list[str] = []
    comparisons = {
        "shared_trunk_parameter_count": (
            int(expected.get("shared_trunk_parameter_count") or 0),
            int(observed.get("shared_trunk_parameter_count") or 0),
        ),
        "expert_parameter_count_per_arm": (
            int(expected.get("expert_parameter_count_per_arm") or 0),
            int(observed.get("expert_parameter_count_per_arm") or 0),
        ),
        "active_parameter_count_per_request": (
            int(expected.get("active_parameter_count_per_request") or 0),
            int(observed.get("active_parameter_count_per_request") or 0),
        ),
        "total_parameter_count": (
            int(expected.get("total_parameter_count") or 0),
            int(observed.get("total_parameter_count") or 0),
        ),
        "dense_active_parameter_count": (
            int((expected.get("dense_active_parameter") or {}).get("parameter_count") or 0),
            int((models.get("dense_active_parameter") or {}).get("parameter_count") or 0),
        ),
        "dense_total_parameter_count": (
            int((expected.get("dense_total_parameter") or {}).get("parameter_count") or 0),
            int((models.get("dense_total_parameter") or {}).get("parameter_count") or 0),
        ),
    }
    for field, (wanted, actual) in comparisons.items():
        if wanted <= 0 or wanted != actual:
            gaps.append(f"scale_model_accounting_mismatch:{field}")
    return {
        "model_accounting_state": "GREEN" if not gaps else "RED",
        "parameter_comparisons": {
            field: {"expected": wanted, "observed": actual}
            for field, (wanted, actual) in comparisons.items()
        },
        "hard_gaps": gaps,
    }


def audit_specialist_data_scaling(
    base: dict[str, Any],
    targets: dict[str, Any],
    models: dict[str, Any],
) -> dict[str, Any]:
    """Bind every trained parameter owner to enough unique model-visible data."""

    ratio = float(
        ((base.get("data_model_scaling_contract") or {}).get("planning_basis") or {}).get(
            "minimum_unique_positions_per_active_parameter"
        )
        or 0.0
    )
    expert_parameters = int(
        ((models.get("moecot_system") or {}).get("expert_parameter_count_per_arm"))
        or 0
    )
    trunk_parameters = int(
        ((models.get("moecot_system") or {}).get("shared_trunk_parameter_count"))
        or 0
    )
    rows: list[dict[str, Any]] = []
    gaps: list[str] = []
    for target_id in (SHARED_TRUNK_ID, *ARM_IDS):
        parameters = trunk_parameters if target_id == SHARED_TRUNK_ID else expert_parameters
        positions = int((targets.get(target_id) or {}).get("unique_target_positions") or 0)
        required = int(np.ceil(parameters * ratio)) if parameters and ratio else 0
        row = {
            "target_id": target_id,
            "owned_parameter_count": parameters,
            "unique_model_visible_positions": positions,
            "minimum_required_positions": required,
            "positions_per_owned_parameter": round(positions / max(1, parameters), 6),
            "meets_floor": bool(parameters > 0 and positions >= required),
        }
        if not row["meets_floor"]:
            gaps.append(f"specialist_unique_position_floor_not_met:{target_id}")
        rows.append(row)
    return {
        "policy": "project_theseus_moecot_specialist_data_scaling_v1",
        "minimum_unique_positions_per_owned_parameter": ratio,
        "state": "GREEN" if not gaps else "RED",
        "rows": rows,
        "hard_gaps": gaps,
        "optimizer_repetition_counted_as_unique_data": False,
        "capability_credit": "NONE",
    }
def inspect_checkpoint_inventory(
    targets: dict[str, Any],
    plan_identity: str,
    stage_signature: Any,
    *,
    plan_identity_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    gaps = []
    checkpoint_hashes: set[str] = set()
    optimizer_hashes: set[str] = set()
    stale_canary_count = 0
    for target_id, target in targets.items():
        receipt_path = resolve(str(target["receipt"]))
        if not receipt_path.is_file():
            rows.append({"target_id": target_id, "state": "NOT_RUN"})
            continue
        receipt = read_json(receipt_path)
        checkpoint = resolve(str(receipt.get("checkpoint") or target["checkpoint"]))
        optimizer = resolve(
            str(receipt.get("optimizer_state") or target["optimizer_state"])
        )
        faults = []
        try:
            validate_resume(
                receipt,
                {
                    "plan_sha256": plan_identity,
                    "stage": {"stage_signature": stage_signature},
                    "plan_identity": plan_identity_contract or {},
                },
                target,
                checkpoint,
                optimizer,
            )
        except ValueError as exc:
            faults.append(str(exc))
        checkpoint_hash = str(receipt.get("checkpoint_sha256") or "")
        optimizer_hash = str(receipt.get("optimizer_state_sha256") or "")
        if checkpoint_hash in checkpoint_hashes:
            faults.append("checkpoint_digest_not_distinct")
        if optimizer_hash in optimizer_hashes:
            faults.append("optimizer_digest_not_distinct")
        checkpoint_hashes.add(checkpoint_hash)
        optimizer_hashes.add(optimizer_hash)
        stale_canary = bool(faults) and (
            receipt.get("bounded_phase_canary") is True
            and receipt.get("complete") is False
            and receipt.get("capability_claim") == "NOT_EVALUATED"
        )
        if stale_canary:
            stale_canary_count += 1
        elif faults:
            gaps.extend(f"checkpoint_inventory:{target_id}:{fault}" for fault in faults)
        rows.append(
            {
                "target_id": target_id,
                "state": (
                    "GREEN" if not faults else "STALE_CANARY" if stale_canary else "RED"
                ),
                "optimizer_steps": int(receipt.get("optimizer_steps") or 0),
                "optimizer_positions": int(receipt.get("optimizer_positions") or 0),
                "complete": bool(receipt.get("complete")),
                "checkpoint_sha256": checkpoint_hash,
                "optimizer_state_sha256": optimizer_hash,
                "capability_claim": receipt.get("capability_claim"),
                "faults": faults,
            }
        )
    completed_smokes = sum(
        row.get("state") == "GREEN" and int(row.get("optimizer_steps") or 0) > 0 for row in rows
    )
    return {
        "state": "GREEN" if completed_smokes == len(targets) and not gaps else (
            "RED" if gaps else "NOT_RUN"
        ),
        "target_count": len(targets),
        "valid_smoke_count": completed_smokes,
        "distinct_checkpoint_digest_count": len(checkpoint_hashes),
        "distinct_optimizer_digest_count": len(optimizer_hashes),
        "all_targets_smoke_ready": completed_smokes == len(targets) and not gaps,
        "stale_canary_count": stale_canary_count,
        "rows": rows,
        "hard_gaps": gaps,
        "capability_claim": "NOT_EVALUATED",
    }


def model_accounting(
    config: dict[str, Any], base: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    config = bind_scale_preregistration(config)
    canonical_vocab_size = model_vocab_size(
        base,
        dict(metadata.get("source_vocab") or {}),
        dict(metadata.get("target_vocab") or {}),
    )
    kernel_disposition = validate_training_disposition(
        config["kernel_english_training"]
    )
    kerc_enabled = kernel_disposition.get("full_kerc_training_enabled") is True
    code_contract = config["kernel_english_training"]["code_vocabulary"]
    kernel_capacity = int(code_contract["kernel_max_vocab"]) if kerc_enabled else 0
    pointer_capacity = int(code_contract["pointer_max_vocab"]) if kerc_enabled else 0
    kerc_vocab_size = canonical_vocab_size + kernel_capacity + pointer_capacity
    def count(
        model_config: dict[str, Any], *, vocab_size: int = canonical_vocab_size
    ) -> int:
        return analytical_parameter_count(
            CausalTransformerConfig(vocab_size=vocab_size, **model_config)
        )

    trunk_count = count(config["shared_trunk_model"])
    arm_config = CausalTransformerConfig(
        vocab_size=canonical_vocab_size, **config["arm_model"]
    )
    arm_count = analytical_parameter_count(arm_config)
    expert_scope = str(config["topology"]["expert_trainable_scope"])
    expert_count = analytical_trainable_parameter_count(arm_config, expert_scope)
    if expert_count <= 0:
        raise ValueError("language expert must add parameters to the shared trunk")
    system_total = trunk_count + expert_count * len(ARM_IDS)
    dense_active_model, dense_active_count = matched_decoder_only_config(
        arm_count, config["arm_model"], count=count
    )
    dense_total_model, dense_total_count = matched_decoder_only_config(
        system_total, config["arm_model"], count=count
    )
    result = {
        "moecot_system": {
            "topology": config["topology"],
            "shared_trunk_model": config["shared_trunk_model"],
            "shared_trunk_parameter_count": trunk_count,
            "arm_model": config["arm_model"],
            "arm_parameter_count": arm_count,
            "expert_parameter_count_per_arm": expert_count,
            "expert_trainable_scope": expert_scope,
            "arm_count": len(ARM_IDS),
            "total_parameter_count": system_total,
            "active_parameter_count_per_request": arm_count,
            "router_parameter_count": 0,
            "router_accounting_state": "EXCLUDED_UNTIL_LANGUAGE_ROUTER_IS_TRAINED",
        },
        "dense_total_parameter": {
            "model": dense_total_model,
            "parameter_count": dense_total_count,
            "active_parameter_count_per_request": dense_total_count,
            "parameter_delta_vs_moecot_total": dense_total_count
            - system_total,
            "architecture": "decoder_only_prefix_lm_control",
        },
        "dense_active_parameter": {
            "model": dense_active_model,
            "parameter_count": dense_active_count,
            "active_parameter_count_per_request": dense_active_count,
            "parameter_delta_vs_active_arm": dense_active_count - arm_count,
            "architecture": "decoder_only_prefix_lm_control",
        },
        "vocab_size": canonical_vocab_size,
        "canonical_vocab_size": canonical_vocab_size,
        "kerc_vocab_size": kerc_vocab_size,
    }
    if not kerc_enabled:
        result["deferred_architecture_candidates"] = {
            KERC_ENGLISH_ID: {
                "state": "DEFERRED_FROM_FIRST_LONG_RUN",
                "topology_exposure": 0,
                "optimizer_repetitions": 0,
                "terminal_evidence_state": "INCONCLUSIVE_IMPLEMENTATION",
            }
        }
        return result

    source_vocab = dict(metadata.get("source_vocab") or {})
    source_offset = source_token_offset(base, source_vocab)
    missing_kerc_tokens = [
        token for token in TRAINING_TASK_TAGS.values() if token not in source_vocab
    ]
    if missing_kerc_tokens:
        raise ValueError(
            "KERC trusted task tokens missing from canonical vocabulary: "
            + ",".join(missing_kerc_tokens)
        )
    kerc_model = dict(config["kerc_english_model"])
    kerc_model["kerc_task_token_ids"] = [
        source_offset + int(source_vocab[TRAINING_TASK_TAGS[objective]])
        for objective in TRAINING_TASK_TAGS
    ]
    kerc_model["kerc_verifier_output_dim"] = len(KERC_VERIFIER_DIMENSIONS)
    kerc_model["kerc_decision_output_dim"] = len(ANSWER_DISPOSITION_ORDER)
    canonical_target_start = target_token_offset(base, source_vocab)
    kerc_model.update(
        {
            "kerc_surface_token_start": canonical_target_start,
            "kerc_surface_token_end": canonical_vocab_size,
            "kerc_kernel_token_start": canonical_vocab_size,
            "kerc_kernel_token_end": canonical_vocab_size + kernel_capacity,
            "kerc_pointer_token_start": canonical_vocab_size + kernel_capacity,
            "kerc_pointer_token_end": kerc_vocab_size,
            "kerc_end_token_id": canonical_target_start
            + int((metadata.get("target_vocab") or {})["<eos>"]),
        }
    )
    kerc_count = count(kerc_model, vocab_size=kerc_vocab_size)

    def surface_count_fn(model: dict[str, Any]) -> int:
        return count(model, vocab_size=canonical_vocab_size)

    surface_model, surface_count = matched_encoder_decoder_config(
        kerc_count,
        config["shared_trunk_model"],
        count=surface_count_fn,
    )
    result[KERC_ENGLISH_ID] = {
        "model": kerc_model,
        "parameter_count": kerc_count,
        "active_parameter_count_per_request": kerc_count,
        "architecture": "kerc_modular_shared_trunk_candidate",
        "vocab_size": kerc_vocab_size,
        "code_vocabulary_capacity": {
            "kernel": kernel_capacity,
            "pointer": pointer_capacity,
        },
    }
    result[SURFACE_ENGLISH_CONTROL_ID] = {
        "model": surface_model,
        "parameter_count": surface_count,
        "active_parameter_count_per_request": surface_count,
        "parameter_delta_vs_kerc": surface_count - kerc_count,
        "architecture": "matched_surface_encoder_decoder_control",
        "vocab_size": canonical_vocab_size,
    }
    return result


def matched_decoder_only_config(
    reference_parameters: int,
    seed: dict[str, Any],
    *,
    count: Any,
) -> tuple[dict[str, Any], int]:
    """Mechanically width-match a prefix-LM control without copying the encoder."""

    candidate = dict(seed)
    candidate["attention_policy"] = "prefix_lm"
    candidate.pop("source_encoder_layers", None)
    candidate.pop("source_copy_mode", None)
    candidate.pop("source_copy_auxiliary_loss_weight", None)
    candidate.pop("expert_adapter_dim", None)
    candidate.pop("source_expert_adapter_dim", None)
    candidate["ff_dim"] = 1
    low_count = int(count(candidate))
    candidate["ff_dim"] = 2
    slope = int(count(candidate)) - low_count
    if slope <= 0:
        raise ValueError("decoder-only parameter matching requires positive FF slope")
    estimated = max(1, round(1 + (reference_parameters - low_count) / slope))
    choices: list[tuple[int, int, dict[str, Any]]] = []
    for width in range(max(1, estimated - 3), estimated + 4):
        model = {**candidate, "ff_dim": width}
        observed = int(count(model))
        choices.append((abs(observed - reference_parameters), observed, model))
    _delta, observed, selected = min(choices, key=lambda row: (row[0], row[1]))
    return selected, observed


def matched_encoder_decoder_config(
    reference_parameters: int,
    seed: dict[str, Any],
    *,
    count: Any,
) -> tuple[dict[str, Any], int]:
    """Width-match a conventional surface model to the full KERC system."""

    candidate = dict(seed)
    for key in (
        "kerc_task_token_ids",
        "kerc_stage_adapter_dim",
        "kerc_reasoner_output_delta_dim",
        "kerc_residual_choice_count",
        "kerc_residual_bottleneck_dim",
        "kerc_residual_unit_kind_count",
        "kerc_residual_unit_feature_dim",
        "kerc_residual_unit_byte_vocab_size",
        "kerc_verifier_dim",
        "kerc_verifier_output_dim",
        "kerc_decision_bottleneck_dim",
        "kerc_decision_output_dim",
    ):
        candidate.pop(key, None)
    candidate["ff_dim"] = 1
    low_count = int(count(candidate))
    candidate["ff_dim"] = 2
    slope = int(count(candidate)) - low_count
    if slope <= 0:
        raise ValueError("surface-control parameter matching requires positive FF slope")
    estimated = max(1, round(1 + (reference_parameters - low_count) / slope))
    choices = []
    for width in range(max(1, estimated - 3), estimated + 4):
        model = {**candidate, "ff_dim": width}
        observed = int(count(model))
        choices.append((abs(observed - reference_parameters), observed, model))
    _delta, observed, selected = min(choices, key=lambda row: (row[0], row[1]))
    return selected, observed


def build_source_to_target_lookup(
    base: dict[str, Any],
    metadata: dict[str, Any],
    *,
    vocab_size: int | None = None,
    identity_ranges: tuple[tuple[int, int], ...] = (),
) -> np.ndarray:
    """Map source identities and explicitly shared structured IDs for copying."""

    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    vocab_size = int(
        vocab_size or model_vocab_size(base, source_vocab, target_vocab)
    )
    lookup = np.full(vocab_size, -1, dtype=np.int32)
    source_offset = source_token_offset(base, source_vocab)
    target_offset = target_token_offset(base, source_vocab)
    for token, source_id in source_vocab.items():
        target_id = target_vocab.get(token)
        if target_id is not None:
            lookup[source_offset + int(source_id)] = target_offset + int(target_id)
    for start, end in identity_ranges:
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > vocab_size:
            raise ValueError(f"copy identity range is outside the model vocabulary: {start}:{end}")
        lookup[start:end] = np.arange(start, end, dtype=np.int32)
    return lookup


def target_copy_identity_ranges(target: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    """Return global code spaces that are valid on both sides of a KERC stage."""

    model = target.get("model") or {}
    if str(target.get("role") or "") != "kerc_english_candidate":
        return ()
    ranges = (
        (int(model["kerc_surface_token_start"]), int(model["kerc_surface_token_end"])),
        (int(model["kerc_kernel_token_start"]), int(model["kerc_kernel_token_end"])),
        (int(model["kerc_pointer_token_start"]), int(model["kerc_pointer_token_end"])),
    )
    if any(left_end != right_start for (_left_start, left_end), (right_start, _right_end) in zip(ranges, ranges[1:])):
        raise ValueError("KERC copy identity ranges must be contiguous and non-overlapping")
    return ranges


def target_contracts(
    config: dict[str, Any],
    arm_views: dict[str, Any],
    models: dict[str, Any],
    plan_identity: str,
    *,
    supervision_audit: dict[str, Any],
    source_conditioned_audit: dict[str, Any],
    kernel_english_audit: dict[str, Any],
) -> dict[str, Any]:
    root = resolve(str(config["checkpoint_root"]))
    targets: dict[str, Any] = {}
    scale_reference = config.get("scale_preregistration")
    if isinstance(scale_reference, dict):
        prereg = read_json(resolve(str(scale_reference["config"])))
        scaling = prereg.get("scaling_contract") or {}
        optimizer_ratio = float(
            scaling.get("minimum_optimizer_positions_per_active_parameter") or 0.0
        )
        maximum_repetitions = float(
            scaling.get("maximum_optimizer_repetition_factor") or 0.0
        )
    else:
        optimizer_ratio = 1.0
        maximum_repetitions = float(
            (config.get("training") or {}).get("maximum_optimizer_repetitions")
            or 1.0
        )
    kernel_cfg = config.get("kernel_english_training") or {}
    kernel_disposition = validate_training_disposition(kernel_cfg)
    english_comparison_ids = (
        ENGLISH_COMPARISON_IDS
        if kernel_disposition.get("full_kerc_training_enabled") is True
        else ()
    )
    for target in (
        SHARED_TRUNK_ID,
        *ARM_IDS,
        *CONTROL_IDS,
        *english_comparison_ids,
    ):
        if target == SHARED_TRUNK_ID:
            view = arm_views.get("mixed_dense_control") or {}
            model_key = "moecot_system"
            model = (models.get(model_key) or {}).get("shared_trunk_model") or config[
                "shared_trunk_model"
            ]
            parameter_count_value = int(
                (models.get(model_key) or {}).get("shared_trunk_parameter_count") or 0
            )
            role = "shared_trunk"
            owned_parameter_count = parameter_count_value
        elif target in ARM_IDS:
            view = (arm_views.get("arms") or {}).get(target) or {}
            model_key = "moecot_system"
            model = (models.get(model_key) or {}).get("arm_model") or config["arm_model"]
            parameter_count_value = int((models.get(model_key) or {}).get("arm_parameter_count") or 0)
            role = "language_expert"
            owned_parameter_count = int(
                (models.get(model_key) or {}).get("expert_parameter_count_per_arm")
                or 0
            )
        elif target in CONTROL_IDS:
            view = arm_views.get("mixed_dense_control") or {}
            model = (models.get(target) or {}).get("model") or {}
            parameter_count_value = int((models.get(target) or {}).get("parameter_count") or 0)
            role = "dense_control"
            owned_parameter_count = parameter_count_value
        else:
            view = (arm_views.get("arms") or {}).get("english") or {}
            model = (models.get(target) or {}).get("model") or {}
            parameter_count_value = int(
                (models.get(target) or {}).get("parameter_count") or 0
            )
            role = (
                "kerc_english_candidate"
                if target == KERC_ENGLISH_ID
                else "english_surface_control"
            )
            owned_parameter_count = parameter_count_value
        directory = root / target
        checkpoint_name = (
            "expert_delta.safetensors" if target in ARM_IDS else "weights.safetensors"
        )
        unique_target_positions = int(view.get("target_positions") or 0)
        exposure = target_optimizer_exposure(
            owned_parameter_count=owned_parameter_count,
            unique_target_positions=unique_target_positions,
            minimum_optimizer_ratio=optimizer_ratio,
            maximum_repetitions=maximum_repetitions,
        )
        targets[target] = {
            "target_id": target,
            "role": role,
            "expert_trainable_scope": (
                str(config["topology"]["expert_trainable_scope"])
                if target in ARM_IDS
                else ""
            ),
            "row_ranges": list(view.get("row_ranges") or []),
            "row_count": sum(int(row["stop"]) - int(row["start"]) for row in view.get("row_ranges") or []),
            "unique_target_positions": unique_target_positions,
            "owned_parameter_count": owned_parameter_count,
            "minimum_optimizer_positions": exposure[
                "minimum_optimizer_positions"
            ],
            "optimizer_target_positions": exposure["optimizer_target_positions"],
            "optimizer_repetition_factor": exposure["optimizer_repetition_factor"],
            "maximum_optimizer_repetition_factor": maximum_repetitions,
            "optimizer_repetition_ceiling_ready": exposure[
                "optimizer_repetition_ceiling_ready"
            ],
            "optimizer_repetition_counted_as_unique_data": False,
            "model": model,
            "vocab_size": int(
                (models.get(target) or {}).get("vocab_size")
                or models.get("canonical_vocab_size")
                or models.get("vocab_size")
                or 0
            ),
            "parameter_count": parameter_count_value,
            "estimated_parameter_token_product": owned_parameter_count
            * exposure["optimizer_target_positions"],
            "checkpoint": relative(
                directory / checkpoint_name
            ),
            "checkpoint_schema_policy": (
                KERC_CHECKPOINT_POLICY if target == KERC_ENGLISH_ID else ""
            ),
            "checkpoint_schema": CURRENT_SCHEMA if target == KERC_ENGLISH_ID else "",
            "checkpoint_schema_version": (
                CURRENT_SCHEMA_VERSION if target == KERC_ENGLISH_ID else 0
            ),
            "shared_trunk_checkpoint": (
                relative(root / SHARED_TRUNK_ID / "weights.safetensors")
                if target in (*ARM_IDS, KERC_ENGLISH_ID)
                else ""
            ),
            "optimizer_state": relative(directory / "optimizer.safetensors"),
            "receipt": relative(directory / "training_receipt.json"),
            "plan_sha256": plan_identity,
            "supervision_artifacts": (
                {
                    split: supervision_audit["artifacts"].get(f"{target}:{split}")
                    for split in ("private_train", "private_dev", "private_eval")
                }
                if target in ARM_IDS
                else {
                    f"{arm}:{split}": supervision_audit["artifacts"].get(f"{arm}:{split}")
                    for arm in ARM_IDS
                    for split in ("private_train", "private_dev", "private_eval")
                }
                if target not in ENGLISH_COMPARISON_IDS
                else {
                    split: supervision_audit["artifacts"].get(f"english:{split}")
                    for split in ("private_train", "private_dev", "private_eval")
                }
            ),
            "source_conditioned_artifacts": (
                {}
                if target in ENGLISH_COMPARISON_IDS
                else
                {
                    "private_train": source_conditioned_audit["artifacts"].get(target)
                }
                if target in ARM_IDS
                and source_conditioned_audit["artifacts"].get(target)
                else {
                    f"{arm}:private_train": source_conditioned_audit["artifacts"].get(arm)
                    for arm in ARM_IDS
                    if source_conditioned_audit["artifacts"].get(arm)
                }
                if target not in ARM_IDS
                else {}
            ),
            "kernel_english_artifacts": (
                {
                    "private_train": kernel_english_audit["artifacts"].get(
                        "english:private_train"
                    )
                }
                if target == "english"
                and kernel_english_audit["artifacts"].get("english:private_train")
                else {
                    "private_train": kernel_english_audit["artifacts"].get(
                        "english:private_train"
                    )
                }
                if target in ENGLISH_COMPARISON_IDS
                and kernel_english_audit["artifacts"].get("english:private_train")
                else {
                    "english:private_train": kernel_english_audit["artifacts"].get(
                        "english:private_train"
                    )
                }
                if (target == SHARED_TRUNK_ID or target in CONTROL_IDS)
                and kernel_english_audit["artifacts"].get("english:private_train")
                else {}
            ),
            "kernel_english_objectives": (
                list(TRAINING_TASK_TAGS)
                if target == KERC_ENGLISH_ID
                else ["surface_direct_control_v1"]
                if target == SURFACE_ENGLISH_CONTROL_ID
                else []
            ),
            "kernel_code_vocabulary": (
                kernel_english_audit.get("code_vocabulary") or {}
                if target == KERC_ENGLISH_ID
                else {}
            ),
            "kerc_compiler_transport": (
                copy.deepcopy(config.get("kerc_compiler_transport") or {})
                if target == KERC_ENGLISH_ID
                else {}
            ),
        }
    return targets


def target_optimizer_exposure(
    *,
    owned_parameter_count: int,
    unique_target_positions: int,
    minimum_optimizer_ratio: float,
    maximum_repetitions: float,
) -> dict[str, Any]:
    """Predeclare target-specific optimizer exposure without inventing data."""

    minimum = int(math.ceil(owned_parameter_count * minimum_optimizer_ratio))
    optimizer_positions = max(unique_target_positions, minimum)
    repetition = optimizer_positions / max(1, unique_target_positions)
    return {
        "minimum_optimizer_positions": minimum,
        "optimizer_target_positions": optimizer_positions,
        "optimizer_repetition_factor": round(repetition, 8),
        "optimizer_repetition_ceiling_ready": bool(
            unique_target_positions > 0 and repetition <= maximum_repetitions
        ),
        "optimizer_repetition_counted_as_unique_data": False,
    }


def audit_arm_views(arm_views: dict[str, Any], window_count: int) -> dict[str, Any]:
    gaps: list[str] = []
    arms = arm_views.get("arms") if isinstance(arm_views.get("arms"), dict) else {}
    if tuple(arms) != ARM_IDS:
        gaps.append("canonical_arm_set_or_order_mismatch")
    occupied: list[tuple[int, int, str]] = []
    for arm_id in ARM_IDS:
        view = arms.get(arm_id) if isinstance(arms.get(arm_id), dict) else {}
        if view.get("independent_weights_required") is not True:
            gaps.append(f"independent_weights_not_required:{arm_id}")
        for row in view.get("row_ranges") or []:
            start, stop = int(row.get("start") or 0), int(row.get("stop") or 0)
            if start < 0 or stop <= start or stop > window_count:
                gaps.append(f"invalid_row_range:{arm_id}")
            occupied.append((start, stop, arm_id))
    occupied.sort()
    cursor = 0
    for start, stop, arm_id in occupied:
        if start != cursor:
            gaps.append(f"arm_range_gap_or_overlap:{arm_id}:{cursor}:{start}")
        cursor = max(cursor, stop)
    if cursor != window_count:
        gaps.append("arm_ranges_do_not_cover_stage")
    control = arm_views.get("mixed_dense_control") or {}
    if control.get("row_ranges") != [{"start": 0, "stop": window_count}]:
        gaps.append("dense_control_not_exact_full_stage")
    if arm_views.get("hidden_generalist_fallback") != "forbidden":
        gaps.append("hidden_generalist_fallback_not_forbidden")
    return {
        "state": "GREEN" if not gaps else "RED",
        "hard_gaps": gaps,
        "window_count": window_count,
        "covered_rows": cursor,
        "non_overlapping_complete_partition": not gaps,
    }


def audit_tokenizer_stage(base: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    expected = (base.get("tokenization") or {}).get("canonical_language_profiles") or {}
    observed = canonical.get("tokenizer_audit") if isinstance(canonical.get("tokenizer_audit"), dict) else {}
    category_profiles = (
        observed.get("category_profiles_by_selected_document")
        if isinstance(observed.get("category_profiles_by_selected_document"), dict)
        else {}
    )
    gaps: list[str] = []
    if expected.get("policy") != "project_theseus_moecot_language_tokenizer_v1":
        gaps.append("canonical_language_tokenizer_policy_missing")
    for category in (
        "english_conversation_instruction",
        "english_broad",
        "python",
        "javascript_typescript",
        "html_css",
        "rust",
    ):
        profile = str(expected.get(category) or "")
        if not profile:
            gaps.append(f"canonical_language_tokenizer_profile_missing:{category}")
        elif int(category_profiles.get(f"{category}:{profile}") or 0) <= 0:
            gaps.append(f"canonical_stage_tokenizer_profile_unproven:{category}:{profile}")
    if int(observed.get("roundtrip_failure_count") or 0):
        gaps.append("canonical_stage_tokenizer_roundtrip_failure")
    if int(observed.get("admitted_unknown_token_position_count") or 0):
        gaps.append("canonical_stage_admitted_unknown_token_position")
    return {
        "state": "GREEN" if not gaps else "RED",
        "policy": expected.get("policy"),
        "expected_profiles": {
            category: expected.get(category)
            for category in (
                "english_conversation_instruction",
                "english_broad",
                "python",
                "javascript_typescript",
                "html_css",
                "rust",
            )
        },
        "observed": observed,
        "hard_gaps": gaps,
        "failure_behavior": "deny_training_until_stage_is_rebuilt",
    }


def audit_supervision_stage(
    config: dict[str, Any], *, config_path: Path
) -> dict[str, Any]:
    cfg = config.get("supervision") if isinstance(config.get("supervision"), dict) else {}
    root = resolve(str(cfg.get("stage_root") or ""))
    manifest_path = root / "manifest.json"
    gaps: list[str] = []
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    if not manifest:
        gaps.append("moecot_supervision_manifest_missing")
    if manifest.get("policy") != "project_theseus_moecot_language_supervision_v1":
        gaps.append("moecot_supervision_manifest_policy_mismatch")
    if manifest.get("trigger_state") != "GREEN":
        gaps.append("moecot_supervision_manifest_not_green")
    expected_supervision_contract = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("contract_sha256") != expected_supervision_contract:
        gaps.append("moecot_supervision_contract_identity_mismatch")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for arm in ARM_IDS:
        for split, wanted in (
            ("private_train", int((cfg.get("train_rows_by_arm") or {}).get(arm) or 0)),
            ("private_dev", int((cfg.get("development_rows_by_arm") or {}).get(arm) or 0)),
            ("private_eval", int((cfg.get("heldout_rows_by_arm") or {}).get(arm) or 0)),
        ):
            key = f"{arm}:{split}"
            row = artifacts.get(key) if isinstance(artifacts.get(key), dict) else {}
            path = resolve(str(row.get("path") or ""))
            if not path.is_file() or sha256_file(path) != str(row.get("sha256") or ""):
                gaps.append(f"moecot_supervision_artifact_identity_mismatch:{key}")
            if int(row.get("row_count") or 0) != wanted:
                gaps.append(f"moecot_supervision_row_count_mismatch:{key}")
    overlap = manifest.get("split_overlap_audit") if isinstance(manifest.get("split_overlap_audit"), dict) else {}
    if int(overlap.get("prompt_overlap_count") or 0):
        gaps.append("moecot_supervision_prompt_overlap")
    if int(overlap.get("target_overlap_count") or 0):
        gaps.append("moecot_supervision_target_overlap")
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(manifest.get(key) or 0):
            gaps.append(f"moecot_supervision_nonzero_boundary:{key}")
    return {
        "state": "GREEN" if not gaps else "RED",
        "manifest": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else "",
        "artifacts": artifacts,
        "row_counts": manifest.get("row_counts") or {},
        "split_overlap_audit": overlap,
        "source_receipts": manifest.get("source_receipts") or [],
        "hard_gaps": gaps,
        "score_semantics": "frozen supervision provenance and split integrity only",
    }


def audit_source_conditioned_stage(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("source_conditioned_pretraining")
    cfg = cfg if isinstance(cfg, dict) else {}
    root = resolve(str(cfg.get("stage_root") or ""))
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    gaps: list[str] = []
    if manifest.get("policy") != "project_theseus_moecot_source_conditioned_pretraining_v1":
        gaps.append("source_conditioned_manifest_policy_mismatch")
    if manifest.get("trigger_state") != "GREEN":
        gaps.append("source_conditioned_manifest_not_green")
    expected_contract = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("contract_sha256") != expected_contract:
        gaps.append("source_conditioned_contract_identity_mismatch")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for arm, wanted in (cfg.get("rows_by_arm") or {}).items():
        if int(wanted) <= 0:
            continue
        row = artifacts.get(arm) if isinstance(artifacts.get(arm), dict) else {}
        path = resolve(str(row.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(row.get("sha256") or ""):
            gaps.append(f"source_conditioned_artifact_identity_mismatch:{arm}")
        if int(row.get("row_count") or 0) != int(wanted):
            gaps.append(f"source_conditioned_row_count_mismatch:{arm}")
    for key in (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
    ):
        if int(manifest.get(key) or 0):
            gaps.append(f"source_conditioned_nonzero_boundary:{key}")
    return {
        "state": "GREEN" if not gaps else "RED",
        "manifest": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else "",
        "artifacts": artifacts,
        "copy_coverage_by_arm": manifest.get("copy_coverage_by_arm") or {},
        "corruption": manifest.get("corruption") or {},
        "hard_gaps": gaps,
        "score_semantics": "source-conditioned objective readiness only",
    }


def audit_kernel_english_stage(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("kernel_english_training")
    cfg = cfg if isinstance(cfg, dict) else {}
    disposition = validate_training_disposition(cfg)
    if disposition.get("full_kerc_training_enabled") is not True:
        return {
            "state": "DEFERRED_FROM_FIRST_LONG_RUN",
            "manifest": "",
            "manifest_sha256": "",
            "artifacts": {},
            "code_vocabulary": {},
            "learned_pipeline_contract": {},
            "architecture_disposition": disposition,
            "full_kerc_training_enabled": False,
            "retained_mechanisms": list(
                disposition.get("retained_mechanisms") or []
            ),
            "selected_record_count_by_split": {
                split: 0 for split in (cfg.get("records_by_split") or {})
            },
            "compiled_view_count_by_objective": {},
            "unique_raw_source_count": 0,
            "derived_view_unique_data_credit": 0,
            "split_overlap_audit": {
                "source_group_overlap_count": 0,
                "raw_source_overlap_count": 0,
                "content_bound_disjoint": True,
                "hard_gaps": [],
            },
            "hard_gaps": [],
            "score_semantics": (
                "bounded pre-training architecture disposition; full KERC receives "
                "zero optimizer exposure"
            ),
        }
    root = resolve(str(cfg.get("stage_root") or ""))
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    gaps: list[str] = []
    if not manifest:
        return {
            "state": "RED",
            "manifest": relative(manifest_path),
            "manifest_sha256": "",
            "artifacts": {},
            "code_vocabulary": {},
            "learned_pipeline_contract": {},
            "selected_record_count_by_split": {},
            "compiled_view_count_by_objective": {},
            "unique_raw_source_count": 0,
            "derived_view_unique_data_credit": 0,
            "split_overlap_audit": {},
            "hard_gaps": sorted(set([*gaps, "kernel_english_manifest_missing"])),
            "score_semantics": "KERC objective/checkpoint readiness only; not learned capability",
        }
    if manifest.get("policy") != "project_theseus_moecot_kernel_english_stage_v1":
        gaps.append("kernel_english_manifest_policy_mismatch")
    if manifest.get("trigger_state") != "GREEN":
        gaps.append("kernel_english_manifest_not_green")
    expected_contract = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("contract_sha256") != expected_contract:
        gaps.append("kernel_english_contract_identity_mismatch")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    compiled_by_objective = manifest.get("compiled_view_count_by_objective") or {}
    compiled_by_split = manifest.get("compiled_view_count_by_split_and_objective") or {}
    expected_view_count = sum(int(value) for value in compiled_by_objective.values())
    if set(compiled_by_objective) != set(cfg.get("objective_order") or ()):
        gaps.append("kernel_english_compiled_objective_inventory_mismatch")
    for split, wanted in (cfg.get("records_by_split") or {}).items():
        key = f"english:{split}"
        row = artifacts.get(key) if isinstance(artifacts.get(key), dict) else {}
        path = resolve(str(row.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(row.get("sha256") or ""):
            gaps.append(f"kernel_english_artifact_identity_mismatch:{key}")
        if int(row.get("unique_record_count") or 0) != int(wanted):
            gaps.append(f"kernel_english_record_count_mismatch:{key}")
        expected_split_views = sum(
            int(value) for value in (compiled_by_split.get(split) or {}).values()
        )
        if not expected_split_views or int(row.get("row_count") or 0) != expected_split_views:
            gaps.append(f"kernel_english_view_count_mismatch:{key}")
    if sum(
        sum(int(value) for value in (compiled_by_split.get(split) or {}).values())
        for split in (cfg.get("records_by_split") or {})
    ) != expected_view_count:
        gaps.append("kernel_english_compiled_view_accounting_mismatch")
    overlap = manifest.get("split_overlap_audit") or {}
    if overlap.get("content_bound_disjoint") is not True:
        gaps.append("kernel_english_split_overlap")
    if int(manifest.get("derived_view_unique_data_credit") or 0):
        gaps.append("kernel_english_derived_view_unique_credit_nonzero")
    if int(manifest.get("verifier_corruption_count") or 0) != expected_view_count:
        gaps.append("kernel_english_verifier_corruption_count_mismatch")
    if manifest.get("verifier_corruptions_receive_generator_loss") is not False:
        gaps.append("kernel_english_verifier_corruption_generator_credit")
    code_vocabulary = manifest.get("code_vocabulary") or {}
    code_path = resolve(str(code_vocabulary.get("path") or ""))
    if (
        not code_path.is_file()
        or sha256_file(code_path) != str(code_vocabulary.get("sha256") or "")
    ):
        gaps.append("kernel_english_code_vocabulary_identity_mismatch")
        code_vocabulary_payload: dict[str, Any] = {}
    else:
        code_vocabulary_payload = read_json(code_path)
        if (
            code_vocabulary_payload.get("policy")
            != "project_theseus_kerc_dual_code_vocabulary_v1"
            or code_vocabulary_payload.get("contract_sha256")
            != code_vocabulary.get("contract_sha256")
            or code_vocabulary_payload.get("fit_split") != "private_train"
            or int(code_vocabulary_payload.get("dev_eval_vocabulary_fit_count") or 0)
            or int(
                code_vocabulary_payload.get(
                    "verifier_corruption_vocabulary_fit_count"
                )
                or 0
            )
        ):
            gaps.append("kernel_english_code_vocabulary_contract_mismatch")
        else:
            unsigned_codebook = {
                key: value
                for key, value in code_vocabulary_payload.items()
                if key != "contract_sha256"
            }
            observed_codebook_hash = "sha256:" + hashlib.sha256(
                json.dumps(
                    unsigned_codebook, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            configured_codebook = cfg.get("code_vocabulary") or {}
            if (
                observed_codebook_hash
                != code_vocabulary_payload.get("contract_sha256")
                or int(code_vocabulary_payload.get("kernel_max_vocab") or 0)
                != int(configured_codebook.get("kernel_max_vocab") or 0)
                or int(code_vocabulary_payload.get("pointer_max_vocab") or 0)
                != int(configured_codebook.get("pointer_max_vocab") or 0)
            ):
                gaps.append("kernel_english_code_vocabulary_content_mismatch")
    for key in (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
        "template_credit",
        "deterministic_renderer_credit",
        "candidate_generation_credit",
    ):
        if int(manifest.get(key) or 0):
            gaps.append(f"kernel_english_nonzero_boundary:{key}")
    return {
        "state": "GREEN" if not gaps else "RED",
        "manifest": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else "",
        "artifacts": artifacts,
        "code_vocabulary": {
            **code_vocabulary,
            "payload": code_vocabulary_payload,
        },
        "learned_pipeline_contract": manifest.get("learned_pipeline_contract") or {},
        "selected_record_count_by_split": manifest.get("selected_record_count_by_split") or {},
        "compiled_view_count_by_objective": manifest.get("compiled_view_count_by_objective") or {},
        "unique_raw_source_count": int(manifest.get("unique_raw_source_count") or 0),
        "derived_view_unique_data_credit": int(
            manifest.get("derived_view_unique_data_credit") or 0
        ),
        "split_overlap_audit": overlap,
        "hard_gaps": sorted(set(gaps)),
        "score_semantics": "KERC objective/checkpoint readiness only; not learned capability",
    }


def safetensors_payload_index(
    path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], int]:
    """Read and validate the small, non-executable safetensors index."""

    with path.open("rb") as handle:
        encoded_length = handle.read(8)
        if len(encoded_length) != 8:
            raise ValueError("optimizer safetensors header is truncated")
        header_length = int(struct.unpack("<Q", encoded_length)[0])
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise ValueError("optimizer safetensors header length is invalid")
        encoded_header = handle.read(header_length)
    try:
        header = json.loads(encoded_header.rstrip(b" ").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("optimizer safetensors header is invalid") from error
    if not isinstance(header, dict):
        raise ValueError("optimizer safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("optimizer safetensors metadata must be an object")
    payload_bytes = path.stat().st_size - 8 - header_length
    intervals: list[tuple[int, int, str]] = []
    for name, row in header.items():
        if (
            not isinstance(name, str)
            or not isinstance(row, dict)
            or not isinstance(row.get("dtype"), str)
            or not isinstance(row.get("shape"), list)
            or not isinstance(row.get("data_offsets"), list)
            or len(row["data_offsets"]) != 2
        ):
            raise ValueError("optimizer safetensors tensor index is invalid")
        start, stop = (int(value) for value in row["data_offsets"])
        if start < 0 or stop < start or stop > payload_bytes:
            raise ValueError("optimizer safetensors tensor offsets are invalid")
        intervals.append((start, stop, name))
    ordered = sorted(intervals)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("optimizer safetensors tensor payloads overlap")
    return metadata, header, 8 + header_length


def safetensors_raw_tensor_sha256(
    path: Path,
    row: dict[str, Any],
    *,
    payload_offset: int,
) -> str:
    start, stop = (int(value) for value in row["data_offsets"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(payload_offset + start)
        remaining = stop - start
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("optimizer safetensors tensor payload is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def project_kerc_compiler_optimizer_state(
    source: Path,
    destination: Path,
    *,
    include_stage_embedding: bool = True,
) -> dict[str, Any]:
    """Project exact AdamW custody to the diagnosed compiler trainable scope.

    This is a byte-preserving projection, not an optimizer reset. It retains the
    global schedule/step scalars and both persistent moments for every parameter
    that ``freeze_to_kerc_stage(1)`` exposes, while rejecting an incomplete or
    structurally surprising source state.
    """

    if destination.exists():
        raise ValueError("projected optimizer destination must be fresh")
    metadata, index, source_payload_offset = safetensors_payload_index(source)
    global_names = {"learning_rate", "step"}

    def compiler_parameter(name: str) -> bool:
        return (
            (
                include_stage_embedding
                and name.startswith("kerc_stage_embedding.")
            )
            or name.startswith("kerc_stage_adapters.1.")
            or ".kerc_decoder_stage_adapters.1." in name
            or name.startswith("kerc_kernel_output.")
        )

    selected_names = sorted(
        name
        for name in index
        if name in global_names
        or (
            (name.endswith(".m") or name.endswith(".v"))
            and compiler_parameter(name.rsplit(".", 1)[0])
        )
    )
    if not global_names.issubset(selected_names):
        raise ValueError("compiler optimizer projection is missing global state")
    selected_parameters = {
        name.rsplit(".", 1)[0]
        for name in selected_names
        if name not in global_names
    }
    if (
        not selected_parameters
        or (
            include_stage_embedding
            and not any(
                name.startswith("kerc_stage_embedding.")
                for name in selected_parameters
            )
        )
        or (
            not include_stage_embedding
            and any(
                name.startswith("kerc_stage_embedding.")
                for name in selected_parameters
            )
        )
        or not any(name.startswith("kerc_stage_adapters.1.") for name in selected_parameters)
        or not any(name.startswith("kerc_kernel_output.") for name in selected_parameters)
        or any(
            f"{parameter}.{moment}" not in index
            for parameter in selected_parameters
            for moment in ("m", "v")
        )
    ):
        raise ValueError("compiler optimizer projection source scope is incomplete")
    unexpected_scalars = sorted(
        name
        for name in index
        if not (name.endswith(".m") or name.endswith(".v"))
        and name not in global_names
    )
    if unexpected_scalars:
        raise ValueError(
            "compiler optimizer projection found unsupported global state: "
            + ",".join(unexpected_scalars)
        )

    projected_index: dict[str, dict[str, Any]] = {}
    cursor = 0
    for name in selected_names:
        source_row = index[name]
        start, stop = (int(value) for value in source_row["data_offsets"])
        size = stop - start
        projected_index[name] = {
            "dtype": source_row["dtype"],
            "shape": source_row["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    projected_header: dict[str, Any] = {
        "__metadata__": {
            "policy": "project_theseus_exact_kerc_stage_optimizer_projection_v1",
            "source_policy": str(metadata.get("policy") or ""),
            "stage_index": "1",
        },
        **projected_index,
    }
    encoded_header = json.dumps(
        projected_header, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    encoded_header += b" " * (-len(encoded_header) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as output:
            output.write(struct.pack("<Q", len(encoded_header)))
            output.write(encoded_header)
            for name in selected_names:
                start, stop = (int(value) for value in index[name]["data_offsets"])
                source_handle.seek(source_payload_offset + start)
                remaining = stop - start
                while remaining:
                    chunk = source_handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(
                            "optimizer safetensors tensor payload is truncated"
                        )
                    output.write(chunk)
                    remaining -= len(chunk)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    projected_metadata, projected_rows, projected_payload_offset = (
        safetensors_payload_index(destination)
    )
    if (
        set(projected_rows) != set(selected_names)
        or projected_metadata.get("policy")
        != "project_theseus_exact_kerc_stage_optimizer_projection_v1"
    ):
        raise ValueError("projected compiler optimizer state failed structural replay")
    tensor_sha256 = {
        name: safetensors_raw_tensor_sha256(
            source, index[name], payload_offset=source_payload_offset
        )
        for name in selected_names
    }
    mismatches = [
        name
        for name in selected_names
        if tensor_sha256[name]
        != safetensors_raw_tensor_sha256(
            destination,
            projected_rows[name],
            payload_offset=projected_payload_offset,
        )
    ]
    if mismatches:
        raise ValueError(
            "projected compiler optimizer tensor identity mismatch: "
            + ",".join(mismatches)
        )
    return {
        "policy": "project_theseus_exact_kerc_stage_optimizer_projection_v1",
        "stage_index": 1,
        "scope": "compiler",
        "stage_embedding_included": bool(include_stage_embedding),
        "source_optimizer_state": relative(source),
        "source_optimizer_state_sha256": sha256_file(source),
        "source_tensor_count": len(index),
        "source_tensor_payload_bytes": sum(
            int(row["data_offsets"][1]) - int(row["data_offsets"][0])
            for row in index.values()
        ),
        "projected_optimizer_state": relative(destination),
        "projected_optimizer_state_sha256": sha256_file(destination),
        "projected_tensor_count": len(projected_rows),
        "projected_tensor_payload_bytes": cursor,
        "selected_parameter_count": len(selected_parameters),
        "selected_tensor_names": selected_names,
        "selected_raw_tensor_sha256": tensor_sha256,
        "source_tensor_values_mutated": False,
        "optimizer_step_reset": False,
        "learning_rate_reset": False,
        "independent_projection_replay": "GREEN",
    }


def merge_kerc_compiler_delta_checkpoint(
    source: Path,
    delta: Path,
    destination: Path,
) -> dict[str, Any]:
    """Merge an exact compiler-output delta into the FP32 source lineage."""

    if destination.exists():
        raise ValueError("merged checkpoint destination must be fresh")
    _source_metadata, source_index, source_payload = (
        safetensors_payload_index(source)
    )
    _delta_metadata, delta_index, delta_payload = safetensors_payload_index(
        delta
    )

    def compiler_parameter(name: str) -> bool:
        return (
            name.startswith("kerc_stage_embedding.")
            or name.startswith("kerc_stage_adapters.1.")
            or ".kerc_decoder_stage_adapters.1." in name
            or name.startswith("kerc_kernel_output.")
        )

    full_compiler_names = {
        name for name in source_index if compiler_parameter(name)
    }
    output_only_names = {
        name
        for name in full_compiler_names
        if not name.startswith("kerc_stage_embedding.")
    }
    if (
        len(full_compiler_names) != 5
        or len(output_only_names) != 4
        or frozenset(delta_index)
        not in {frozenset(full_compiler_names), frozenset(output_only_names)}
    ):
        raise ValueError("compiler delta checkpoint scope is not exact")
    expected_delta_names = set(delta_index)
    incompatible = sorted(
        name
        for name in expected_delta_names
        if delta_index[name]["dtype"] != source_index[name]["dtype"]
        or delta_index[name]["dtype"] != "F32"
        or delta_index[name]["shape"] != source_index[name]["shape"]
    )
    if incompatible:
        raise ValueError(
            "compiler delta tensor dtype or shape mismatch:"
            + ",".join(incompatible)
        )

    merged_index: dict[str, dict[str, Any]] = {}
    cursor = 0
    for name in sorted(source_index):
        row = delta_index[name] if name in delta_index else source_index[name]
        start, stop = (int(value) for value in row["data_offsets"])
        size = stop - start
        merged_index[name] = {
            "dtype": row["dtype"],
            "shape": row["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    header: dict[str, Any] = {
        "__metadata__": {
            "policy": "project_theseus_kerc_compiler_delta_merge_v1",
            "source_checkpoint_sha256": sha256_file(source),
            "compiler_delta_sha256": sha256_file(delta),
        },
        **merged_index,
    }
    encoded_header = json.dumps(
        header, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    encoded_header += b" " * (-len(encoded_header) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with temporary.open("xb") as output:
            output.write(struct.pack("<Q", len(encoded_header)))
            output.write(encoded_header)
            for name in sorted(source_index):
                selected_delta = name in delta_index
                path = delta if selected_delta else source
                payload_offset = (
                    delta_payload if selected_delta else source_payload
                )
                row = (
                    delta_index[name]
                    if selected_delta
                    else source_index[name]
                )
                start, stop = (int(value) for value in row["data_offsets"])
                with path.open("rb") as input_handle:
                    input_handle.seek(payload_offset + start)
                    remaining = stop - start
                    while remaining:
                        chunk = input_handle.read(
                            min(1024 * 1024, remaining)
                        )
                        if not chunk:
                            raise ValueError(
                                "checkpoint tensor payload is truncated"
                            )
                        output.write(chunk)
                        remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    _merged_metadata, observed_index, observed_payload = (
        safetensors_payload_index(destination)
    )
    if set(observed_index) != set(source_index):
        raise ValueError("merged checkpoint tensor identity changed")
    selected_delta_exact = all(
        safetensors_raw_tensor_sha256(
            destination,
            observed_index[name],
            payload_offset=observed_payload,
        )
        == safetensors_raw_tensor_sha256(
            delta,
            delta_index[name],
            payload_offset=delta_payload,
        )
        for name in delta_index
    )
    frozen_source_exact = all(
        safetensors_raw_tensor_sha256(
            destination,
            observed_index[name],
            payload_offset=observed_payload,
        )
        == safetensors_raw_tensor_sha256(
            source,
            source_index[name],
            payload_offset=source_payload,
        )
        for name in source_index
        if name not in delta_index
    )
    if not selected_delta_exact or not frozen_source_exact:
        raise ValueError("merged checkpoint independent tensor replay failed")
    return {
        "policy": "project_theseus_kerc_compiler_delta_merge_v1",
        "source_checkpoint": relative(source),
        "source_checkpoint_sha256": sha256_file(source),
        "compiler_delta": relative(delta),
        "compiler_delta_sha256": sha256_file(delta),
        "merged_checkpoint": relative(destination),
        "merged_checkpoint_sha256": sha256_file(destination),
        "source_tensor_count": len(source_index),
        "compiler_delta_tensor_count": len(delta_index),
        "stage_embedding_included": any(
            name.startswith("kerc_stage_embedding.")
            for name in delta_index
        ),
        "frozen_source_tensor_count": len(source_index) - len(delta_index),
        "compiler_delta_tensor_names": sorted(delta_index),
        "selected_delta_tensors_exact": selected_delta_exact,
        "frozen_source_tensors_exact": frozen_source_exact,
        "canonical_fp32_tensor_dtype_preserved": True,
        "independent_merge_replay": "GREEN",
        "source_checkpoint_mutated": False,
        "delta_checkpoint_mutated": False,
        "capability_claim": "NONE_CHECKPOINT_CUSTODY_ONLY",
    }


def initialize_candidate_continuation_receipt(
    report_path: Path,
    *,
    target: dict[str, Any],
    candidate_lease: dict[str, Any],
) -> dict[str, Any]:
    """Import one exact candidate generation into a fresh scratch namespace.

    The source tensors remain immutable and are loaded by their content hashes.
    Only the small receipt is copied.  The next committed generation is written
    under the new scratch target, so the terminal source run is never mutated.
    """

    execution_policy = dict(candidate_lease.get("execution_policy") or {})
    required_policy_fields = (
        "continuation_source_report",
        "continuation_source_report_sha256",
        "continuation_source_plan_sha256",
        "continuation_source_checkpoint_sha256",
        "continuation_source_optimizer_state_sha256",
        "continuation_source_mlx_rng_state_sha256",
        "continuation_source_optimizer_steps",
        "continuation_source_optimizer_positions",
        "continuation_reset_data_cursor_phase",
        "continuation_reset_data_cursor_seed",
        "continuation_reset_phase_position_accounting",
        "continuation_learning_rate",
        "continuation_min_learning_rate",
        "continuation_warmup_steps",
    )
    missing = [
        key
        for key in required_policy_fields
        if execution_policy.get(key) in (None, "")
    ]
    if missing:
        raise ValueError(
            "candidate continuation policy is incomplete: " + ",".join(missing)
        )
    source_report = resolve(str(execution_policy["continuation_source_report"]))
    if (
        report_path.resolve() != source_report.resolve()
        or not source_report.is_file()
        or sha256_file(source_report)
        != str(execution_policy["continuation_source_report_sha256"])
    ):
        raise ValueError("candidate continuation report identity mismatch")
    report = read_json(source_report)
    source_lease = report.get("candidate_canary_lease") or {}
    results = report.get("results") or []
    selected_seed = int(candidate_lease.get("selected_seed") or 0)
    if (
        report.get("policy")
        != "project_theseus_moecot_language_arm_training_plan_v1"
        or report.get("mode") != "training_execution"
        or report.get("trigger_state") != "GREEN"
        or report.get("hard_gaps")
        or report.get("executed_targets") != [KERC_ENGLISH_ID]
        or len(results) != 1
        or source_lease.get("candidate_id") != "rdc_kerc_k5_adequacy"
        or int(source_lease.get("selected_seed") or 0) != selected_seed
    ):
        raise ValueError("candidate continuation source report is not admissible")
    receipt = copy.deepcopy(results[0])
    expected_identities = {
        "plan_sha256": str(execution_policy["continuation_source_plan_sha256"]),
        "checkpoint_sha256": str(
            execution_policy["continuation_source_checkpoint_sha256"]
        ),
        "optimizer_state_sha256": str(
            execution_policy["continuation_source_optimizer_state_sha256"]
        ),
        "mlx_rng_state_sha256": str(
            execution_policy["continuation_source_mlx_rng_state_sha256"]
        ),
        "optimizer_steps": int(
            execution_policy["continuation_source_optimizer_steps"]
        ),
        "optimizer_positions": int(
            execution_policy["continuation_source_optimizer_positions"]
        ),
    }
    if (
        receipt.get("policy")
        != "project_theseus_moecot_language_arm_training_receipt_v1"
        or receipt.get("target_id") != target.get("target_id")
        or int(receipt.get("candidate_seed") or 0) != selected_seed
        or any(receipt.get(key) != value for key, value in expected_identities.items())
    ):
        raise ValueError("candidate continuation source receipt identity mismatch")
    source_artifacts = {
        "checkpoint": (
            resolve(str(receipt.get("checkpoint") or "")),
            str(receipt.get("checkpoint_sha256") or ""),
        ),
        "optimizer_state": (
            resolve(str(receipt.get("optimizer_state") or "")),
            str(receipt.get("optimizer_state_sha256") or ""),
        ),
        "mlx_rng_state": (
            resolve(str(receipt.get("mlx_rng_state") or "")),
            str(receipt.get("mlx_rng_state_sha256") or ""),
        ),
    }
    artifact_faults = [
        name
        for name, (path, expected_sha256) in source_artifacts.items()
        if not path.is_file() or sha256_file(path) != expected_sha256
    ]
    if artifact_faults:
        raise ValueError(
            "candidate continuation artifact identity mismatch: "
            + ",".join(artifact_faults)
        )
    receipt_path = resolve(str(target["receipt"]))
    destination_optimizer = resolve(str(target["optimizer_state"]))
    destination_artifacts = [
        receipt_path,
        resolve(str(target["checkpoint"])),
        destination_optimizer,
    ]
    if any(path.exists() for path in destination_artifacts):
        raise ValueError("candidate continuation requires a fresh scratch target")
    optimizer_state_projection = None
    kerc_stage_only = execution_policy.get("kerc_delta_stage_only")
    if kerc_stage_only is not None:
        if int(kerc_stage_only) != 1:
            raise ValueError(
                "candidate continuation optimizer projection currently supports "
                "only the diagnosed KERC compiler stage"
            )
        if execution_policy.get("continuation_optimizer_state_projection_policy") != (
            "project_theseus_exact_kerc_stage_optimizer_projection_v1"
        ):
            raise ValueError(
                "KERC stage-only continuation requires exact optimizer-state projection"
            )
        optimizer_state_projection = project_kerc_compiler_optimizer_state(
            source_artifacts["optimizer_state"][0],
            destination_optimizer,
            include_stage_embedding=bool(
                execution_policy.get(
                    "kerc_stage_train_stage_embedding", True
                )
            ),
        )
        receipt["optimizer_state"] = relative(destination_optimizer)
        receipt["optimizer_state_sha256"] = optimizer_state_projection[
            "projected_optimizer_state_sha256"
        ]
        receipt["optimizer_state_projection"] = optimizer_state_projection
    import_receipt = {
        "policy": "project_theseus_exact_candidate_continuation_import_v1",
        "source_report": relative(source_report),
        "source_report_sha256": sha256_file(source_report),
        "source_plan_sha256": receipt["plan_sha256"],
        "source_checkpoint": relative(source_artifacts["checkpoint"][0]),
        "source_checkpoint_sha256": receipt["checkpoint_sha256"],
        "source_optimizer_state": relative(source_artifacts["optimizer_state"][0]),
        "source_optimizer_state_sha256": receipt["optimizer_state_sha256"],
        "source_mlx_rng_state": relative(source_artifacts["mlx_rng_state"][0]),
        "source_mlx_rng_state_sha256": receipt["mlx_rng_state_sha256"],
        "source_optimizer_steps": int(receipt["optimizer_steps"]),
        "source_optimizer_positions": int(receipt["optimizer_positions"]),
        "reset_phase_position_accounting": str(
            execution_policy["continuation_reset_phase_position_accounting"]
        ),
        "selected_seed": selected_seed,
        "destination_receipt": relative(receipt_path),
        "source_generation_mutated": False,
        "registered_lineage_mutated": False,
    }
    inherited_current_kernel_positions = int(
        receipt.get("current_kernel_phase_optimizer_positions") or 0
    )
    inherited_coverage_counts = dict(
        (
            (
                (receipt.get("phases") or {}).get("kernel_english")
                or {}
            ).get("coverage_first_sampling")
            or {}
        ).get("observed_label_counts")
        or {}
    )
    import_receipt["inherited_current_kernel_phase_optimizer_positions"] = (
        inherited_current_kernel_positions
    )
    import_receipt["inherited_coverage_observed_label_counts"] = (
        inherited_coverage_counts
    )
    # This is a fresh candidate continuation, not another one-step segment in
    # the source objective. The destination phase and coverage ledgers begin at
    # zero; later fresh-process resumes accumulate from the first destination
    # update.
    receipt["current_kernel_phase_optimizer_positions"] = 0
    receipt["current_kernel_phase_position_accounting_reset"] = False
    if optimizer_state_projection is not None:
        import_receipt["source_optimizer_state_sha256"] = expected_identities[
            "optimizer_state_sha256"
        ]
        import_receipt["optimizer_state_projection"] = optimizer_state_projection
    receipt["candidate_continuation_import"] = import_receipt
    write_json_atomic(receipt_path, receipt)
    return import_receipt


def execute_targets(
    config: dict[str, Any],
    plan: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
    targets: list[str],
    max_steps: int,
    resume: bool,
    training_phase: str = "all",
    scratch_checkpoint_root: Path | None = None,
    candidate_lease: dict[str, Any] | None = None,
    optimizer_id: str = "",
    candidate_initialization_state_path: Path | None = None,
    candidate_continuation_report_path: Path | None = None,
) -> dict[str, Any]:
    if candidate_lease is not None and candidate_lease.get("selected_seed") is None:
        raise ValueError("canonical candidate execution requires one bound preregistered seed")
    if candidate_continuation_report_path is not None and (
        candidate_lease is None
        or candidate_lease.get("candidate_id")
        not in {"rdc_kerc_k5_adequacy", "rdc_kerc_k5_overfit"}
        or scratch_checkpoint_root is None
        or (
            resume
            and (
                candidate_lease.get("execution_policy") or {}
            ).get("candidate_scratch_resume_policy")
            != "exact_fresh_process_segment_v1"
        )
        or training_phase != "kernel_english"
        or set(targets) != {KERC_ENGLISH_ID}
    ):
        raise ValueError("candidate continuation authority is invalid")

    stage_dir = resolve(str(config["stage_dir"]))
    metadata = read_json(stage_dir / "stage_metadata_v1.json")
    base = read_json(resolve(str(config["base_config"])))
    if any(target_id == SHARED_TRUNK_ID or target_id in ARM_IDS for target_id in targets):
        import mlx.core as mx
        import mlx.nn as nn

        ensure_shared_trunk_migration(
            config,
            plan,
            metadata=metadata,
            base=base,
            mx=mx,
            nn=nn,
            require_existing=any(target_id in ARM_IDS for target_id in targets),
        )
    canonical = metadata["summary"]["canonical_pretrain_stage"]
    stage = canonical_pretraining_execution_stage(
        stage_dir,
        canonical,
        active=training_phase in {"all", "pretraining"},
    )
    defer_ordinary_auxiliary_stages = training_phase == "all"
    auxiliary_cache_paths: dict[str, dict[str, Path]] = {}
    if defer_ordinary_auxiliary_stages:
        for target_id in targets:
            target = plan["targets"][target_id]
            if str(target.get("role") or "") == "kerc_english_candidate":
                continue
            target_cache_paths: dict[str, Path] = {}
            for artifact_field in (
                "source_conditioned_artifacts",
                "supervision_artifacts",
            ):
                if not (target.get(artifact_field) or {}):
                    continue
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "moecot_auxiliary_stage_cache.py"),
                        "--config",
                        str(config_path),
                        "--target",
                        target_id,
                        "--artifact-field",
                        artifact_field,
                    ],
                    cwd=ROOT,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                receipt_policy = (
                    "project_theseus_moecot_source_conditioned_arrays_v1"
                    if artifact_field == "source_conditioned_artifacts"
                    else "project_theseus_moecot_exact_supervision_arrays_v1"
                )
                target_cache_paths[artifact_field] = (
                    auxiliary_stage_cache_path(
                        config,
                        base,
                        target,
                        metadata=metadata,
                        artifact_field=artifact_field,
                        receipt_policy=receipt_policy,
                    )
                )
            auxiliary_cache_paths[target_id] = target_cache_paths
    supervision_stages = {
        target_id: (
            defer_target_supervision(
                config,
                base,
                plan["targets"][target_id],
                metadata=metadata,
                cache_path=auxiliary_cache_paths[target_id][
                    "supervision_artifacts"
                ],
            )
            if defer_ordinary_auxiliary_stages
            and str(plan["targets"][target_id].get("role") or "")
            != "kerc_english_candidate"
            else materialize_target_supervision(
                config,
                base,
                plan["targets"][target_id],
                metadata=metadata,
            )
        )
        if training_phase in {"all", "supervision"}
        else None
        for target_id in targets
    }
    source_conditioned_stages = {
        target_id: (
            defer_target_supervision(
                config,
                base,
                plan["targets"][target_id],
                metadata=metadata,
                artifact_field="source_conditioned_artifacts",
                receipt_policy=(
                    "project_theseus_moecot_source_conditioned_arrays_v1"
                ),
                cache_path=auxiliary_cache_paths[target_id][
                    "source_conditioned_artifacts"
                ],
            )
            if defer_ordinary_auxiliary_stages
            and str(plan["targets"][target_id].get("role") or "")
            != "kerc_english_candidate"
            else materialize_target_supervision(
                config,
                base,
                plan["targets"][target_id],
                metadata=metadata,
                artifact_field="source_conditioned_artifacts",
                receipt_policy=(
                    "project_theseus_moecot_source_conditioned_arrays_v1"
                ),
            )
        )
        if training_phase in {"all", "source_conditioned_pretraining"}
        and (plan["targets"][target_id].get("source_conditioned_artifacts") or {})
        else None
        for target_id in targets
    }
    kerc_canary_row_limit = 0
    if candidate_lease is not None and candidate_lease.get("candidate_id") in {
        "rdc_kerc_adequacy",
        "rdc_kerc_k5_adequacy",
        "rdc_kerc_k5_overfit",
    }:
        kerc_batch = int(
            (config.get("kernel_english_training") or {}).get("batch_size") or 1
        )
        configured_canary_row_limit = int(
            (candidate_lease.get("execution_policy") or {}).get(
                "kerc_bounded_source_row_limit", 0
            )
        )
        kerc_canary_row_limit = (
            configured_canary_row_limit
            if configured_canary_row_limit
            else min(
                256,
                max(64, int(candidate_lease["requested_steps"]) * kerc_batch * 2),
            )
        )
        if not 4 <= kerc_canary_row_limit <= 4096:
            raise ValueError("KERC bounded source row limit must be in [4, 4096]")
    candidate_execution_policy = dict(
        (candidate_lease or {}).get("execution_policy") or {}
    )
    kerc_stage_objective_filter = tuple(
        str(value)
        for value in (
            candidate_execution_policy.get("kerc_stage_objective_filter") or ()
        )
    )
    kerc_stage_required_coverage_labels = tuple(
        str(value)
        for value in (
            candidate_execution_policy.get(
                "kerc_stage_required_coverage_labels"
            )
            or ()
        )
    )
    kerc_stage_excluded_coverage_labels = dict(
        candidate_execution_policy.get(
            "kerc_stage_excluded_coverage_labels"
        )
        or {}
    )
    if kerc_stage_objective_filter:
        stage_learnability_intervention = bool(
            candidate_execution_policy.get(
                "kerc_stage_learnability_intervention", False
            )
        )
        minimum_coverage_rows = int(
            candidate_execution_policy.get(
                "kerc_stage_minimum_coverage_rows", 0
            )
        )
        coverage_multiplier = int(
            candidate_execution_policy.get("kerc_stage_coverage_multiplier", 0)
        )
        bounded_source_rows = int(
            candidate_execution_policy.get("kerc_bounded_source_row_limit", 0)
        )
        if (
            set(targets) != {KERC_ENGLISH_ID}
            or not set(kerc_stage_objective_filter).issubset(TRAINING_TASK_TAGS)
            or candidate_execution_policy.get("kerc_objective_balanced_sampling")
            is True
            or not stage_learnability_intervention
            or minimum_coverage_rows <= 0
            or coverage_multiplier <= 0
            or bounded_source_rows
            != minimum_coverage_rows * coverage_multiplier
            or not kerc_stage_required_coverage_labels
            or not set(kerc_stage_required_coverage_labels).issubset(
                KERC_CANARY_REQUIRED_COVERAGE
            )
            or {
                label
                for label in kerc_stage_required_coverage_labels
                if label.startswith("objective:")
            }
            != {
                f"objective:{objective}"
                for objective in kerc_stage_objective_filter
            }
            or set(kerc_stage_required_coverage_labels).intersection(
                kerc_stage_excluded_coverage_labels
            )
            or set(kerc_stage_excluded_coverage_labels)
            != {"decision:ABSTAIN"}
            or any(
                not str(reason).startswith("K7_LONG_SEQUENCE_STRESS_")
                for reason in kerc_stage_excluded_coverage_labels.values()
            )
        ):
            raise ValueError(
                "KERC stage objective filtering requires one KERC target, known "
                "objectives, objective balancing disabled, and a declared "
                "coverage-multiple learnability intervention"
            )
    kernel_english_stages = {
        target_id: materialize_target_supervision(
            config,
            base,
            plan["targets"][target_id],
            metadata=metadata,
            artifact_field="kernel_english_artifacts",
            receipt_policy="project_theseus_moecot_kernel_english_arrays_v1",
            maximum_sequence_tokens=int(
                config["kernel_english_training"]["maximum_sequence_tokens"]
            ),
            objective_filter=tuple(
                kerc_stage_objective_filter
                if target_id == KERC_ENGLISH_ID and kerc_stage_objective_filter
                else plan["targets"][target_id].get("kernel_english_objectives") or ()
            ),
            bounded_source_row_limit=kerc_canary_row_limit,
        )
        if training_phase in {"all", "kernel_english"}
        and (plan["targets"][target_id].get("kernel_english_artifacts") or {})
        else None
        for target_id in targets
    }
    if kerc_stage_objective_filter:
        stage_receipt = (
            kernel_english_stages[KERC_ENGLISH_ID].receipt
            if kernel_english_stages.get(KERC_ENGLISH_ID) is not None
            else {}
        )
        selected_source_rows = sum(
            int(row.get("selected_row_count") or 0)
            for row in stage_receipt.get("artifacts") or []
        )
        if selected_source_rows != int(
            candidate_execution_policy["kerc_bounded_source_row_limit"]
        ):
            raise ValueError(
                "KERC stage learnability intervention did not materialize its "
                "declared source-row budget"
            )
    maximum_candidate_sequence_tokens = int(
        ((candidate_lease or {}).get("execution_policy") or {}).get(
            "maximum_training_sequence_tokens"
        )
        or 0
    )
    candidate_supervised_sequence_by_target = dict(
        ((candidate_lease or {}).get("execution_policy") or {}).get(
            "maximum_supervised_training_sequence_tokens_by_target"
        )
        or {}
    )
    if maximum_candidate_sequence_tokens:
        kernel_english_stages = {
            target_id: (
                bound_supervision_stage_sequence_width(
                    stage,
                    maximum_sequence_tokens=maximum_candidate_sequence_tokens,
                    maximum_supervised_sequence_tokens=(
                        int(
                            candidate_supervised_sequence_by_target.get(
                                target_id, maximum_candidate_sequence_tokens
                            )
                        )
                    ),
                    required_kerc_coverage_labels=(
                        kerc_stage_required_coverage_labels
                        if target_id == KERC_ENGLISH_ID
                        and kerc_stage_objective_filter
                        else None
                    ),
                )
                if stage is not None
                else None
            )
            for target_id, stage in kernel_english_stages.items()
        }
    overfit_rows_per_objective = int(
        ((candidate_lease or {}).get("execution_policy") or {}).get(
            "kerc_overfit_rows_per_objective"
        )
        or 0
    )
    if overfit_rows_per_objective:
        if (
            ((candidate_lease or {}).get("execution_policy") or {}).get(
                "kerc_overfit_generator_rows_only"
            )
            is not True
            or set(targets) != {KERC_ENGLISH_ID}
        ):
            raise ValueError("KERC overfit policy requires one generator-only KERC target")
        kernel_english_stages[KERC_ENGLISH_ID] = select_kerc_overfit_stage(
            kernel_english_stages[KERC_ENGLISH_ID],
            rows_per_objective=overfit_rows_per_objective,
        )
    # Candidate stage construction is CPU-only and can transiently retain
    # source rows before the bounded RaggedRows projection replaces them.
    # Initialize Metal only after that temporary residency has been collected.
    import gc

    gc.collect()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    monitor = (
        pretraining_candidate_canary.CandidateCanaryMonitor(candidate_lease)
        if candidate_lease is not None
        else None
    )
    candidate_initialization_state = None
    if candidate_lease is not None:
        expected_execution_policy = dict(
            candidate_lease.get("execution_policy") or {}
        )
        if (
            candidate_initialization_state_path is not None
            and candidate_initialization_state_path.is_file()
        ):
            candidate_initialization_state = read_json(
                candidate_initialization_state_path
            )
            if candidate_initialization_state.get(
                "execution_policy"
            ) != expected_execution_policy:
                raise ValueError(
                    "candidate initialization execution policy changed across isolated targets"
                )
            if int(candidate_initialization_state.get("seed") or 0) != int(
                candidate_lease.get("selected_seed") or 0
            ):
                raise ValueError(
                    "candidate initialization seed changed across isolated targets"
                )
        else:
            candidate_initialization_state = {
                "execution_policy": expected_execution_policy
            }
    results = []
    for target_id in targets:
        target = (
            scratch_target_contract(
                plan["targets"][target_id], scratch_checkpoint_root
            )
            if scratch_checkpoint_root is not None
            else plan["targets"][target_id]
        )
        continuation_import = None
        target_resume = resume
        if candidate_continuation_report_path is not None:
            if resume:
                execution_policy = dict(
                    (candidate_lease or {}).get("execution_policy") or {}
                )
                prior_segment_receipt = read_json(resolve(str(target["receipt"])))
                if (
                    execution_policy.get("candidate_scratch_resume_policy")
                    != "exact_fresh_process_segment_v1"
                    or prior_segment_receipt.get(
                        "current_kernel_phase_position_accounting_reset"
                    )
                    is not True
                    or prior_segment_receipt.get("target_id") != target_id
                    or int(prior_segment_receipt.get("candidate_seed") or 0)
                    != int((candidate_lease or {}).get("selected_seed") or 0)
                    or candidate_continuation_report_path.resolve()
                    != resolve(
                        str(execution_policy["continuation_source_report"])
                    ).resolve()
                    or sha256_file(candidate_continuation_report_path)
                    != str(
                        execution_policy[
                            "continuation_source_report_sha256"
                        ]
                    )
                ):
                    raise ValueError(
                        "candidate segmented continuation identity mismatch"
                    )
                continuation_import = {
                    "policy": (
                        "project_theseus_exact_candidate_continuation_import_v1"
                    ),
                    "source_checkpoint_sha256": str(
                        execution_policy[
                            "continuation_source_checkpoint_sha256"
                        ]
                    ),
                    "source_optimizer_state_sha256": str(
                        execution_policy[
                            "continuation_source_optimizer_state_sha256"
                        ]
                    ),
                    "source_mlx_rng_state_sha256": str(
                        execution_policy[
                            "continuation_source_mlx_rng_state_sha256"
                        ]
                    ),
                    "segmented_resume": True,
                }
            else:
                continuation_import = initialize_candidate_continuation_receipt(
                    candidate_continuation_report_path,
                    target=target,
                    candidate_lease=candidate_lease or {},
                )
            if candidate_initialization_state is None:
                raise ValueError("candidate continuation state is missing")
            candidate_initialization_state["candidate_continuation"] = (
                continuation_import
            )
            target_resume = True
        result = train_target(
            config,
            plan,
            target,
            stage=stage,
            source_conditioned_stage=source_conditioned_stages[target_id],
            kernel_english_stage=kernel_english_stages[target_id],
            supervision_stage=supervision_stages[target_id],
            max_steps=max_steps,
            resume=target_resume,
            training_phase=training_phase,
            mx=mx,
            nn=nn,
            optim=optim,
            mlx_utils=mlx_utils,
            step_boundary_callback=(monitor.check if monitor is not None else None),
            optimizer_id=optimizer_id,
            candidate_seed=int(
                (candidate_lease or {}).get("selected_seed") or 0
            ),
            candidate_initialization_state=candidate_initialization_state,
        )
        if (
            candidate_initialization_state is not None
            and candidate_initialization_state_path is not None
        ):
            write_json_atomic(
                candidate_initialization_state_path,
                candidate_initialization_state,
            )
        candidate_behavior_rows = terminal_candidate_behavior_rows(
            candidate_lease
        )
        result["candidate_behavior_evaluation_disposition"] = {
            "policy": "project_theseus_terminal_candidate_behavior_evaluation_v1",
            "requested_rows": int(
                (candidate_lease or {}).get("behavior_eval_rows") or 0
            ),
            "executed_rows": candidate_behavior_rows,
            "terminal_candidate_run": candidate_behavior_rows > 0,
            "nonterminal_resource_preflight": bool(candidate_lease)
            and candidate_behavior_rows == 0,
            "terminal_behavior_evaluation_required": bool(candidate_lease),
        }
        if should_evaluate_target(target) and (
            result.get("complete") or candidate_behavior_rows > 0
        ):
            result["evaluation"] = evaluate_target(
                config,
                base,
                candidate_bound_evaluation_plan(plan, candidate_lease),
                target,
                metadata=metadata,
                mx=mx,
                nn=nn,
                maximum_rows=candidate_behavior_rows,
            )
        results.append(result)
    if candidate_initialization_state is not None and len(targets) == 2:
        if not candidate_initialization_state.get("aligned_target_id"):
            raise ValueError("candidate pair did not complete common initialization alignment")
    gaps = [
        f"{row['target_id']}:{gap}"
        for row in results
        for gap in row.get("hard_gaps") or []
    ]
    refreshed_inventory = inspect_checkpoint_inventory(
        plan["targets"], plan["plan_sha256"], plan["stage"]["stage_signature"]
    )
    canary_resource_receipt = monitor.finalize(results) if monitor is not None else None
    if canary_resource_receipt and not canary_resource_receipt["passed"]:
        gaps.extend(canary_resource_receipt["faults"])
    return {
        **plan,
        "checkpoint_inventory": refreshed_inventory,
        "candidate_canary_lease": candidate_lease,
        "candidate_canary_resource_receipt": canary_resource_receipt,
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "mode": "training_execution",
        "executed_targets": targets,
        "scratch_execution": {
            "enabled": scratch_checkpoint_root is not None,
            "checkpoint_root": (
                relative(scratch_checkpoint_root)
                if scratch_checkpoint_root is not None
                else ""
            ),
            "registered_lineage_mutated": False,
            "resumable": bool(
                scratch_checkpoint_root is not None
                and (
                    (candidate_lease or {}).get("execution_policy") or {}
                ).get("candidate_scratch_resume_policy")
                == "exact_fresh_process_segment_v1"
            )
            if scratch_checkpoint_root is not None
            else resume,
            "one_shot_candidate_continuation": (
                candidate_continuation_report_path is not None
            ),
            "candidate_continuation_report": (
                relative(candidate_continuation_report_path)
                if candidate_continuation_report_path is not None
                else ""
            ),
        },
        "results": results,
        "hard_gaps": gaps,
        "all_requested_targets_complete": bool(results)
        and all(row.get("complete") for row in results),
        **no_cheat(config),
    }


def scratch_target_contract(target: dict[str, Any], root: Path) -> dict[str, Any]:
    """Redirect one bounded canary without changing the registered plan identity."""

    scratch = copy.deepcopy(target)
    directory = root / str(target["target_id"])
    checkpoint_suffix = Path(str(target["checkpoint"])).suffix or ".npz"
    checkpoint_name = "weights" + checkpoint_suffix
    scratch.update(
        {
            "checkpoint": str(directory / checkpoint_name),
            "optimizer_state": str(directory / "optimizer.safetensors"),
            "receipt": str(directory / "training_receipt.json"),
            "scratch_canary": True,
            "registered_checkpoint": str(target["checkpoint"]),
            "registered_optimizer_state": str(target["optimizer_state"]),
            "registered_receipt": str(target["receipt"]),
        }
    )
    return scratch


def evaluate_training_progress(
    config: dict[str, Any],
    plan: dict[str, Any],
    *,
    targets: list[str],
    baseline_checkpoint: str = "",
) -> dict[str, Any]:
    """Measure private-development loss without requiring campaign completion."""

    import mlx.core as mx
    import mlx.nn as nn

    stage_dir = resolve(str(config["stage_dir"]))
    metadata = read_json(stage_dir / "stage_metadata_v1.json")
    base = read_json(resolve(str(config["base_config"])))
    results = []
    gaps = []
    for target_id in targets:
        target = copy.deepcopy(plan["targets"][target_id])
        receipt_path = resolve(str(target["receipt"]))
        if not receipt_path.is_file():
            gaps.append(f"{target_id}:training_receipt_missing")
            continue
        receipt = read_json(receipt_path)
        current_checkpoint = resolve(str(receipt.get("checkpoint") or ""))
        checkpoints = [("current", current_checkpoint)]
        if baseline_checkpoint:
            checkpoints.insert(0, ("baseline", resolve(baseline_checkpoint)))
        checkpoint_reports = []
        for label, checkpoint in checkpoints:
            if not checkpoint.is_file():
                gaps.append(f"{target_id}:{label}_checkpoint_missing")
                continue
            trained_vocab_size = int(
                target.get("vocab_size") or plan["models"]["vocab_size"]
            )
            model = build_model(
                CausalTransformerConfig(
                    vocab_size=trained_vocab_size,
                    **target["model"],
                ),
                mx=mx,
                nn=nn,
                state_role_lookup=None,
                source_to_target_lookup=build_source_to_target_lookup(
                    base,
                    metadata,
                    vocab_size=trained_vocab_size,
                    identity_ranges=target_copy_identity_ranges(target),
                ),
            )
            if target.get("role") == "language_expert":
                shared = resolve(str(target.get("shared_trunk_checkpoint") or ""))
                if not shared.is_file():
                    raise ValueError(
                        "expert progress evaluation requires shared trunk checkpoint"
                    )
                model.load_weights(str(shared), strict=False)
                model.load_weights(str(checkpoint), strict=False)
            else:
                model.load_weights(str(checkpoint))
            mx.eval(model.parameters())
            by_arm = {}
            weighted_loss = 0.0
            weighted_positions = 0
            for arm_id in ARM_IDS:
                arm_target = copy.deepcopy(target)
                artifacts = arm_target.get("supervision_artifacts") or {}
                arm_target["supervision_artifacts"] = {
                    key: value
                    for key, value in artifacts.items()
                    if key == f"{arm_id}:private_dev"
                }
                if not arm_target["supervision_artifacts"]:
                    continue
                development = materialize_target_supervision(
                    config,
                    base,
                    arm_target,
                    metadata=metadata,
                    split="private_dev",
                )
                started = time.perf_counter()
                loss = evaluate_loss(
                    model,
                    development.inputs,
                    development.labels,
                    development.loss_mask,
                    batch_size=int(config["training"]["batch_size"]),
                    mx=mx,
                    nn=nn,
                )
                positions = int(development.mask.sum())
                by_arm[arm_id] = {
                    "teacher_forced_loss": loss,
                    "row_count": len(development.inputs),
                    "target_positions": positions,
                    "wall_seconds": round(time.perf_counter() - started, 6),
                }
                weighted_loss += loss * positions
                weighted_positions += positions
            checkpoint_reports.append(
                {
                    "label": label,
                    "checkpoint": relative(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "optimizer_steps": (
                        int(receipt.get("optimizer_steps") or 0)
                        if label == "current"
                        else None
                    ),
                    "optimizer_positions": (
                        int(receipt.get("optimizer_positions") or 0)
                        if label == "current"
                        else None
                    ),
                    "teacher_forced_loss": round(
                        weighted_loss / max(1, weighted_positions), 6
                    ),
                    "target_positions": weighted_positions,
                    "by_arm": by_arm,
                }
            )
            del model
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        comparison = None
        if len(checkpoint_reports) == 2:
            baseline, current = checkpoint_reports
            baseline_loss = float(baseline["teacher_forced_loss"])
            current_loss = float(current["teacher_forced_loss"])
            arm_deltas = {
                arm_id: round(
                    float(current["by_arm"][arm_id]["teacher_forced_loss"])
                    - float(baseline["by_arm"][arm_id]["teacher_forced_loss"]),
                    6,
                )
                for arm_id in sorted(set(baseline["by_arm"]) & set(current["by_arm"]))
            }
            comparison = {
                "absolute_loss_delta": round(current_loss - baseline_loss, 6),
                "relative_loss_reduction": round(
                    (baseline_loss - current_loss) / max(1e-12, baseline_loss),
                    8,
                ),
                "improved": current_loss < baseline_loss,
                "regressed_arms": sorted(
                    arm_id for arm_id, delta in arm_deltas.items() if delta > 0.0
                ),
                "loss_delta_by_arm": arm_deltas,
            }
        results.append(
            {
                "target_id": target_id,
                "receipt": relative(receipt_path),
                "receipt_complete": bool(receipt.get("complete")),
                "checkpoints": checkpoint_reports,
                "comparison": comparison,
            }
        )
    return {
        **plan,
        "created_utc": now(),
        "mode": "private_development_learning_curve",
        "trigger_state": "RED" if gaps else "GREEN",
        "results": results,
        "hard_gaps": gaps,
        "evaluation_split": "private_dev",
        "confirmation_split_consumed": False,
        "public_calibration_consumed": False,
        "capability_claim": "NOT_EVALUATED",
        "score_semantics": (
            "Teacher-forced source-disjoint private-development learning signal only; "
            "not direct generation utility or a promotion claim."
        ),
        **no_cheat(config),
    }


def ensure_shared_trunk_migration(
    config: dict[str, Any],
    plan: dict[str, Any],
    *,
    metadata: dict[str, Any],
    base: dict[str, Any],
    mx: Any,
    nn: Any,
    require_existing: bool = True,
) -> dict[str, Any]:
    """Validate, migrate, or authorize fresh initialization for the shared trunk."""

    target = plan["targets"][SHARED_TRUNK_ID]
    checkpoint = resolve(str(target["checkpoint"]))
    optimizer = resolve(str(target["optimizer_state"]))
    receipt_path = resolve(str(target["receipt"]))
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        committed_checkpoint = resolve(str(receipt.get("checkpoint") or checkpoint))
        committed_optimizer = resolve(
            str(receipt.get("optimizer_state") or optimizer)
        )
        validate_resume(
            receipt,
            plan,
            target,
            committed_checkpoint,
            committed_optimizer,
        )
        return receipt
    if any(path.exists() for path in (checkpoint, optimizer, receipt_path)):
        raise ValueError("partial shared trunk migration state requires operator cleanup")

    topology = config["topology"]
    initialization = topology.get("shared_trunk_initialization") or {}
    if initialization.get("policy") == "project_theseus_seeded_fresh_trunk_initialization_v1":
        if int(initialization.get("seed") or -1) != int(config["seed"]):
            raise ValueError("fresh shared trunk seed mismatch")
        if require_existing:
            raise ValueError("language expert requires a completed fresh shared trunk")
        return {
            "policy": initialization["policy"],
            "state": "FRESH_INITIALIZATION_AUTHORIZED",
            "seed": int(config["seed"]),
            "training_positions_added": 0,
            "capability_credit": "NONE",
        }

    bootstrap = topology.get("shared_trunk_bootstrap") or initialization
    if bootstrap.get("policy") != "project_theseus_exact_shared_trunk_migration_v1":
        raise ValueError("unsupported shared trunk initialization policy")
    source_checkpoint = resolve(str(bootstrap["checkpoint"]))
    source_optimizer = resolve(str(bootstrap["optimizer_state"]))
    source_receipt_path = resolve(str(bootstrap["receipt"]))
    for path, expected, label in (
        (source_checkpoint, bootstrap["checkpoint_sha256"], "checkpoint"),
        (source_optimizer, bootstrap["optimizer_state_sha256"], "optimizer"),
        (source_receipt_path, bootstrap["receipt_sha256"], "receipt"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"shared trunk migration source {label} identity mismatch")
    source_receipt = read_json(source_receipt_path)
    if not bool(source_receipt.get("complete")):
        raise ValueError("shared trunk migration source is incomplete")
    if source_receipt.get("checkpoint_sha256") != bootstrap["checkpoint_sha256"]:
        raise ValueError("shared trunk source receipt checkpoint mismatch")
    if source_receipt.get("optimizer_state_sha256") != bootstrap["optimizer_state_sha256"]:
        raise ValueError("shared trunk source receipt optimizer mismatch")
    if source_receipt.get("stage_signature") != plan["stage"]["stage_signature"]:
        raise ValueError("shared trunk migration stage identity mismatch")

    target_vocab_size = int(
        target.get("vocab_size") or plan["models"]["vocab_size"]
    )
    copy_lookup = build_source_to_target_lookup(
        base,
        metadata,
        vocab_size=target_vocab_size,
        identity_ranges=target_copy_identity_ranges(target),
    )
    model = build_model(
        CausalTransformerConfig(
            vocab_size=target_vocab_size, **target["model"]
        ),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=copy_lookup,
    )
    model.load_weights(str(source_checkpoint), strict=True)
    mx.eval(model.parameters())

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_copy(source_checkpoint, checkpoint)
    atomic_copy(source_optimizer, optimizer)
    receipt = {
        **source_receipt,
        "created_utc": now(),
        "target_id": SHARED_TRUNK_ID,
        "role": "shared_trunk",
        "plan_sha256": plan["plan_sha256"],
        "row_ranges": target["row_ranges"],
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "optimizer_state": relative(optimizer),
        "optimizer_state_sha256": sha256_file(optimizer),
        "resume": False,
        "resume_base_checkpoint_sha256": "",
        "migration": {
            "policy": bootstrap["policy"],
            "source_checkpoint": relative(source_checkpoint),
            "source_checkpoint_sha256": bootstrap["checkpoint_sha256"],
            "source_optimizer_state": relative(source_optimizer),
            "source_optimizer_state_sha256": bootstrap["optimizer_state_sha256"],
            "source_receipt": relative(source_receipt_path),
            "source_receipt_sha256": bootstrap["receipt_sha256"],
            "strict_model_load_proved": True,
            "model_config_sha256": hashlib.sha256(
                json.dumps(
                    target["model"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "training_positions_added": 0,
            "capability_credit": "NONE",
        },
        "capability_claim": "NOT_EVALUATED",
        "hard_gaps": [],
    }
    write_json_atomic(receipt_path, receipt)
    validate_resume(receipt, plan, target, checkpoint, optimizer)
    return receipt


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def should_evaluate_target(target: dict[str, Any]) -> bool:
    """Only executable model compositions receive direct behavior evaluation."""

    role = str(target.get("role") or "")
    return role in {
        "language_expert",
        "dense_control",
        "english_surface_control",
        "kerc_english_candidate",
    }


def supervision_row_instance_id(
    row_id: str, *, artifact_key: str, source_index: int
) -> str:
    """Disambiguate repeated semantic row ids without using answer metadata."""

    if source_index < 0 or not artifact_key:
        raise ValueError("supervision row instance identity requires source custody")
    return (
        f"{row_id or artifact_key}:artifact:{artifact_key}:source_index:{source_index}"
    )


@dataclass(frozen=True)
class DeferredSupervisionStage:
    """Carry exact planning mass without retaining auxiliary arrays."""

    planning_row_count: int
    artifact_field: str
    receipt_policy: str
    factory: Callable[[], Any]

    def materialize(self) -> Any:
        stage = self.factory()
        if len(stage.inputs) != self.planning_row_count:
            raise ValueError(
                "deferred supervision row count changed between planning and "
                f"materialization: {len(stage.inputs)} != {self.planning_row_count}"
            )
        return stage


def auxiliary_stage_cache_contract(
    config: dict[str, Any],
    base: dict[str, Any],
    target: dict[str, Any],
    *,
    metadata: dict[str, Any],
    artifact_field: str,
    receipt_policy: str,
    split: str,
) -> dict[str, Any]:
    """Content-bind an ordinary dense auxiliary-array cache."""

    artifacts = target.get(artifact_field) or {}
    selected = {
        str(key): artifact
        for key, artifact in artifacts.items()
        if key == split or str(key).endswith(f":{split}")
    }
    training = config.get("training") or {}
    return {
        "policy": "project_theseus_auxiliary_stage_memmap_cache_v1",
        "implementation_sha256": sha256_file(Path(__file__)),
        "target_id": str(target["target_id"]),
        "target_role": str(target.get("role") or ""),
        "artifact_field": artifact_field,
        "receipt_policy": receipt_policy,
        "split": split,
        "artifacts": selected,
        "tokenization": base.get("tokenization") or {},
        "source_vocab_sha256": hashlib.sha256(
            json.dumps(
                metadata.get("source_vocab") or {},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "target_vocab_sha256": hashlib.sha256(
            json.dumps(
                metadata.get("target_vocab") or {},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "loss_weights": {
            key: training.get(key)
            for key in (
                "termination_loss_weight",
                "byte_boundary_loss_weight",
                "kerc_compiler_schema_continuation_loss_weight",
                "kerc_compiler_semantic_pointer_loss_weight",
                "kerc_compiler_semantic_pointer_loss_weights_by_kind",
            )
        },
    }


def auxiliary_stage_cache_path(
    config: dict[str, Any],
    base: dict[str, Any],
    target: dict[str, Any],
    *,
    metadata: dict[str, Any],
    artifact_field: str,
    receipt_policy: str,
    split: str = "private_train",
) -> Path:
    contract = auxiliary_stage_cache_contract(
        config,
        base,
        target,
        metadata=metadata,
        artifact_field=artifact_field,
        receipt_policy=receipt_policy,
        split=split,
    )
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target["target_id"]))
    return (
        ROOT
        / "runtime"
        / "moecot_auxiliary_stage_cache_v1"
        / safe_target
        / artifact_field
        / digest
    )


def load_auxiliary_stage_cache(
    cache_path: Path,
    *,
    expected_contract: dict[str, Any],
) -> Any:
    """Validate and open cached arrays as read-only NumPy memmaps."""

    manifest_path = cache_path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"auxiliary stage cache manifest missing: {cache_path}")
    manifest = read_json(manifest_path)
    contract_sha256 = hashlib.sha256(
        json.dumps(
            expected_contract, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if (
        manifest.get("policy")
        != "project_theseus_auxiliary_stage_memmap_cache_v1"
        or manifest.get("contract") != expected_contract
        or manifest.get("contract_sha256") != contract_sha256
    ):
        raise ValueError("auxiliary stage cache contract mismatch")
    arrays: dict[str, Any] = {}
    for name in ("inputs", "labels", "mask", "loss_mask", "sample_weights"):
        artifact = (manifest.get("arrays") or {}).get(name) or {}
        path = cache_path / str(artifact.get("file") or "")
        if (
            not path.is_file()
            or sha256_file(path) != str(artifact.get("sha256") or "")
        ):
            raise ValueError(f"auxiliary stage cache array mismatch: {name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            list(value.shape) != list(artifact.get("shape") or [])
            or str(value.dtype) != str(artifact.get("dtype") or "")
        ):
            raise ValueError(f"auxiliary stage cache array schema mismatch: {name}")
        arrays[name] = value
    return SimpleNamespace(
        **arrays,
        kerc_residual_labels=None,
        kerc_residual_loss_mask=None,
        kerc_unit_allocator_rows=None,
        kerc_verifier_labels=None,
        kerc_decision_labels=None,
        kerc_decision_loss_mask=None,
        kerc_coverage_labels=None,
        training_row_ids=(),
        receipt=dict(manifest["receipt"]),
        cache_manifest={
            "path": relative(manifest_path),
            "sha256": sha256_file(manifest_path),
            "contract_sha256": contract_sha256,
        },
    )


def write_auxiliary_stage_cache(
    config: dict[str, Any],
    base: dict[str, Any],
    target: dict[str, Any],
    *,
    metadata: dict[str, Any],
    artifact_field: str,
    receipt_policy: str,
    split: str = "private_train",
) -> Path:
    """Materialize one cache in a short-lived CPU-only producer process."""

    contract = auxiliary_stage_cache_contract(
        config,
        base,
        target,
        metadata=metadata,
        artifact_field=artifact_field,
        receipt_policy=receipt_policy,
        split=split,
    )
    cache_path = auxiliary_stage_cache_path(
        config,
        base,
        target,
        metadata=metadata,
        artifact_field=artifact_field,
        receipt_policy=receipt_policy,
        split=split,
    )
    if cache_path.is_dir():
        load_auxiliary_stage_cache(
            cache_path, expected_contract=contract
        )
        return cache_path
    stage = materialize_target_supervision(
        config,
        base,
        target,
        metadata=metadata,
        artifact_field=artifact_field,
        receipt_policy=receipt_policy,
        split=split,
    )
    temporary = cache_path.with_name(
        f".{cache_path.name}.partial-{os.getpid()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    array_artifacts: dict[str, Any] = {}
    try:
        for name in (
            "inputs",
            "labels",
            "mask",
            "loss_mask",
            "sample_weights",
        ):
            path = temporary / f"{name}.npy"
            with path.open("wb") as handle:
                np.save(handle, np.asarray(getattr(stage, name)), allow_pickle=False)
            value = np.asarray(getattr(stage, name))
            array_artifacts[name] = {
                "file": path.name,
                "sha256": sha256_file(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": int(path.stat().st_size),
            }
        manifest = {
            "policy": "project_theseus_auxiliary_stage_memmap_cache_v1",
            "created_utc": now(),
            "contract": contract,
            "contract_sha256": hashlib.sha256(
                json.dumps(
                    contract, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "arrays": array_artifacts,
            "receipt": stage.receipt,
            "training_row_ids_sha256": hashlib.sha256(
                "\n".join(stage.training_row_ids).encode()
            ).hexdigest(),
        }
        write_json(temporary / "manifest.json", manifest)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(temporary, cache_path)
        except OSError:
            if not cache_path.is_dir():
                raise
            shutil.rmtree(temporary)
        load_auxiliary_stage_cache(
            cache_path, expected_contract=contract
        )
        return cache_path
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def defer_target_supervision(
    config: dict[str, Any],
    base: dict[str, Any],
    target: dict[str, Any],
    *,
    metadata: dict[str, Any],
    artifact_field: str = "supervision_artifacts",
    receipt_policy: str = "project_theseus_moecot_exact_supervision_arrays_v1",
    split: str = "private_train",
    cache_path: Path | None = None,
) -> DeferredSupervisionStage:
    """Bind one ordinary auxiliary stage while deferring its dense arrays."""

    if str(target.get("role") or "") == "kerc_english_candidate":
        raise ValueError("KERC auxiliary stages require eager audited materialization")
    artifacts = target.get(artifact_field) or {}
    selected = [
        (key, artifact)
        for key, artifact in artifacts.items()
        if key == split or str(key).endswith(f":{split}")
    ]
    if not selected:
        raise ValueError(
            f"target has no frozen {artifact_field} train artifact: "
            f"{target['target_id']}"
        )
    planning_row_count = sum(
        int((artifact or {}).get("row_count") or 0)
        for _key, artifact in selected
    )
    if planning_row_count <= 0:
        raise ValueError("deferred supervision requires positive frozen row count")
    cache_contract = auxiliary_stage_cache_contract(
        config,
        base,
        target,
        metadata=metadata,
        artifact_field=artifact_field,
        receipt_policy=receipt_policy,
        split=split,
    )
    return DeferredSupervisionStage(
        planning_row_count=planning_row_count,
        artifact_field=artifact_field,
        receipt_policy=receipt_policy,
        factory=(
            lambda: load_auxiliary_stage_cache(
                cache_path, expected_contract=cache_contract
            )
            if cache_path is not None
            else materialize_target_supervision(
                config,
                base,
                target,
                metadata=metadata,
                artifact_field=artifact_field,
                receipt_policy=receipt_policy,
                split=split,
            )
        ),
    )


def materialize_target_supervision(
    config: dict[str, Any],
    base: dict[str, Any],
    target: dict[str, Any],
    *,
    metadata: dict[str, Any],
    artifact_field: str = "supervision_artifacts",
    receipt_policy: str = "project_theseus_moecot_exact_supervision_arrays_v1",
    maximum_sequence_tokens: int | None = None,
    objective_filter: tuple[str, ...] = (),
    split: str = "private_train",
    bounded_source_row_limit: int = 0,
) -> Any:
    """Encode one frozen private split without truncation or hidden-field routing."""

    if split not in {"private_train", "private_dev", "private_eval"}:
        raise ValueError(f"unsupported private supervision split: {split}")

    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    if not source_vocab or not target_vocab:
        raise ValueError("canonical stage metadata is missing exact vocabularies")
    source_offset = source_token_offset(base, source_vocab)
    target_offset = target_token_offset(base, source_vocab)
    max_sequence = int(
        maximum_sequence_tokens
        or (base.get("tokenization") or {}).get("max_sequence_tokens")
        or 0
    )
    artifacts = target.get(artifact_field) or {}
    selected = [
        (key, row)
        for key, row in artifacts.items()
        if key == split or str(key).endswith(f":{split}")
    ]
    if not selected:
        raise ValueError(
            f"target has no frozen {artifact_field} train artifact: {target['target_id']}"
        )

    # Freeze token rows into compact arrays immediately. Keeping thousands of
    # Python integer lists alive until the post-pass leaves allocator arenas
    # resident during MLX execution and materially raises unified-memory pressure.
    sequences: list[np.ndarray] = []
    training_row_ids: list[str] = []
    mask_starts: list[int] = []
    generator_loss_enabled: list[bool] = []
    compiler_schema_continuation_indices: list[tuple[int, ...]] = []
    compiler_semantic_pointer_indices_by_kind: list[
        dict[str, tuple[int, ...]]
    ] = []
    compiler_semantic_pointer_position_counts_by_kind = {
        kind: 0 for kind in KERC_COMPILER_SEMANTIC_TARGET_KINDS
    }
    sampling_weights: list[float] = []
    kerc_residual_rows: list[list[int]] = []
    kerc_residual_loss_enabled: list[bool] = []
    kerc_unit_allocator_rows: list[dict[str, Any] | None] = []
    kerc_verifier_rows: list[list[int]] = []
    kerc_decision_rows: list[int] = []
    kerc_decision_loss_enabled: list[bool] = []
    kerc_coverage_rows: list[tuple[str, ...]] = []
    kerc_model = str(target.get("role") or "") == "kerc_english_candidate"
    kerc_mode = kerc_model and artifact_field == "kernel_english_artifacts"
    code_vocabulary = (
        ((target.get("kernel_code_vocabulary") or {}).get("payload") or {})
        if kerc_mode
        else {}
    )
    if kerc_mode and code_vocabulary.get("policy") != (
        "project_theseus_kerc_dual_code_vocabulary_v1"
    ):
        raise ValueError("KERC target requires its content-bound dual-code vocabulary")
    kernel_offset = int((target.get("model") or {}).get("kerc_kernel_token_start") or 0)
    pointer_offset = int((target.get("model") or {}).get("kerc_pointer_token_start") or 0)
    pointer_end = int(
        (target.get("model") or {}).get("kerc_pointer_token_end")
        or (
            pointer_offset
            + max(
                (int(value) for value in (code_vocabulary.get("pointer_vocab") or {}).values()),
                default=-1,
            )
            + 1
        )
    )
    code_token_rows = (
        kerc_global_token_rows(
            code_vocabulary,
            kernel_offset=kernel_offset,
            pointer_offset=pointer_offset,
            pointer_end=pointer_end,
        )
        if kerc_mode
        else {}
    )
    ground_truth_json_grammar_audited_count = 0
    compact_compiler_transport_generator_rows = 0
    compact_compiler_transport_legacy_target_tokens = 0
    compact_compiler_transport_encoded_target_tokens = 0
    compact_compiler_transport_allocator_values_removed = 0
    compiler_transport = (
        target.get("kerc_compiler_transport")
        if isinstance(target.get("kerc_compiler_transport"), dict)
        else {}
    )
    compiler_transport_version = int(
        compiler_transport.get("version")
        or LEARNED_COMPILER_COMPACT_TRANSPORT_VERSION
    )
    if kerc_mode and compiler_transport_version not in {
        LEARNED_COMPILER_COMPACT_TRANSPORT_VERSION,
        LEARNED_COMPILER_SEMANTIC_POINTER_TRANSPORT_VERSION,
        LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION,
    }:
        raise ValueError(
            f"unsupported KERC compiler transport version: {compiler_transport_version}"
        )
    row_hashes: list[str] = []
    artifact_receipts: list[dict[str, Any]] = []
    context_counterfactual_counts = {
        "context_withheld": 0,
        "context_shuffled": 0,
    }
    for key, artifact in selected:
        if not isinstance(artifact, dict):
            raise ValueError(f"invalid supervision artifact contract: {key}")
        path = resolve(str(artifact.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(artifact.get("sha256") or ""):
            raise ValueError(f"supervision artifact identity mismatch: {key}")
        observed_rows = 0
        source_rows = 0
        bounded_row_references: list[tuple[int, str, int]] | None = None
        if bounded_source_row_limit:
            source_rows, bounded_row_references = (
                select_bounded_supervision_row_references(
                    path,
                    split=split,
                    objective_filter=objective_filter,
                    maximum_rows=bounded_source_row_limit,
                )
            )
        with path.open(encoding="utf-8") as handle:
            row_iterator = (
                iter_bounded_supervision_rows(path, bounded_row_references)
                if bounded_row_references is not None
                else (
                    (index, json.loads(line))
                    for index, line in enumerate(handle)
                )
            )
            for source_index, row in row_iterator:
                if bounded_row_references is None:
                    source_rows += 1
                if row.get("split") != split or row.get("public_benchmark") is not False:
                    raise ValueError(f"invalid supervision boundary: {key}:{source_index}")
                if objective_filter and str(row.get("objective") or "") not in objective_filter:
                    continue
                prompt = str(row.get("prompt") or "")
                answer = str(row.get("target") or "")
                objective = str(row.get("objective") or "")
                row_instance_id = supervision_row_instance_id(
                    str(row.get("row_id") or ""),
                    artifact_key=str(key),
                    source_index=int(source_index),
                )
                structured_source = (
                    kerc_mode and objective in KERC_STRUCTURED_SOURCE_OBJECTIVES
                )
                if structured_source:
                    source_body_ids, source_receipt = encode_kerc_global_target(
                        prompt,
                        code_vocabulary=code_vocabulary,
                        kernel_offset=kernel_offset,
                        pointer_offset=pointer_offset,
                    )
                else:
                    source_body_ids, source_receipt = encode_tokens(
                        kerc_surface_tokens(prompt)
                        if kerc_mode
                        else exact_text_tokens(prompt),
                        source_vocab,
                        stream="source",
                    )
                trusted_prefix = list(row.get("trusted_source_prefix_tokens") or [])
                if trusted_prefix:
                    if (
                        len(trusted_prefix) != 1
                        or trusted_prefix[0] not in source_vocab
                        or row.get("trusted_prefix_authority")
                        != "internal_objective_route_only"
                    ):
                        raise ValueError(f"invalid trusted source-prefix contract: {key}")
                trusted_source_ids = [
                    source_token_offset(base, source_vocab)
                    + int(source_vocab[token])
                    for token in trusted_prefix
                ]
                source_ids = [
                    *trusted_source_ids,
                    *(
                        source_body_ids
                        if structured_source
                        else [
                            source_token_offset(base, source_vocab) + int(value)
                            for value in source_body_ids
                        ]
                    ),
                ]
                kernel_objective = (
                    kerc_mode
                    and str(row.get("objective") or "") in KERC_KERNEL_OBJECTIVES
                )
                compiler_source_surface: str | None = None
                if (
                    kerc_mode
                    and objective == "surface_to_kernel_program_v1"
                    and compiler_transport_version
                    == LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION
                ):
                    try:
                        compiler_prompt = json.loads(prompt)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid KERC compiler prompt: {key}:{source_index}"
                        ) from exc
                    compiler_source_surface = str(
                        compiler_prompt.get("source_surface") or ""
                    )
                    if (
                        not compiler_source_surface
                        or compiler_prompt.get("source_character_length")
                        != len(compiler_source_surface)
                    ):
                        raise ValueError(
                            f"invalid KERC compiler source contract: {key}:{source_index}"
                        )
                encoded_answer = (
                    compact_learned_compiler_transport_text(
                        answer,
                        transport_version=compiler_transport_version,
                        source=compiler_source_surface,
                    )
                    if kerc_mode and objective == "surface_to_kernel_program_v1"
                    else answer
                )
                compiler_continuation_indices: tuple[int, ...] = ()
                semantic_indices_by_kind: dict[str, tuple[int, ...]] = {
                    kind: () for kind in KERC_COMPILER_SEMANTIC_TARGET_KINDS
                }
                if kernel_objective:
                    if objective == "surface_to_kernel_program_v1":
                        (
                            target_ids,
                            target_receipt,
                            logical_token_ranges,
                        ) = encode_kerc_global_target_with_logical_ranges(
                            encoded_answer,
                            code_vocabulary=code_vocabulary,
                            kernel_offset=kernel_offset,
                            pointer_offset=pointer_offset,
                        )
                        code_tokens = [
                            str(token) for token in kerc_code_tokens(encoded_answer)
                        ]
                        logical_continuation_indices = (
                            learned_compiler_transport_required_continuation_token_indices(
                                code_tokens,
                                source=compiler_source_surface,
                            )
                        )
                        compiler_continuation_indices = tuple(
                            encoded_index
                            for logical_index in logical_continuation_indices
                            for encoded_index in range(
                                logical_token_ranges[logical_index][0],
                                logical_token_ranges[logical_index][1],
                            )
                        )
                        logical_semantic_pointer_indices_by_kind = (
                            learned_compiler_transport_semantic_pointer_token_indices_by_kind(
                                code_tokens,
                                source=compiler_source_surface,
                            )
                        )
                        semantic_indices_by_kind = {
                            kind: tuple(
                                encoded_index
                                for logical_index in (
                                    logical_semantic_pointer_indices_by_kind[kind]
                                )
                                for encoded_index in range(
                                    logical_token_ranges[logical_index][0],
                                    logical_token_ranges[logical_index][1],
                                )
                            )
                            for kind in KERC_COMPILER_SEMANTIC_TARGET_KINDS
                        }
                    else:
                        target_ids, target_receipt = encode_kerc_global_target(
                            encoded_answer,
                            code_vocabulary=code_vocabulary,
                            kernel_offset=kernel_offset,
                            pointer_offset=pointer_offset,
                        )
                    if objective == "surface_to_kernel_program_v1":
                        legacy_target_ids, legacy_target_receipt = (
                            encode_kerc_global_target(
                                answer,
                                code_vocabulary=code_vocabulary,
                                kernel_offset=kernel_offset,
                                pointer_offset=pointer_offset,
                            )
                        )
                        if int(
                            legacy_target_receipt.get("unknown_token_count") or 0
                        ):
                            raise ValueError(
                                f"legacy KERC compiler target became unrepresentable: {key}"
                            )
                        compact_compiler_transport_generator_rows += 1
                        compact_compiler_transport_legacy_target_tokens += len(
                            legacy_target_ids
                        )
                        compact_compiler_transport_encoded_target_tokens += len(
                            target_ids
                        )
                        if (
                            compiler_transport_version
                            in {
                                LEARNED_COMPILER_SEMANTIC_POINTER_TRANSPORT_VERSION,
                                LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION,
                            }
                        ):
                            compact_compiler_transport_allocator_values_removed += len(
                                (
                                    json.loads(answer).get("residual") or {}
                                ).get("unit_fidelity")
                                or []
                            )
                else:
                    target_ids, target_receipt = encode_tokens(
                        kerc_surface_tokens(answer), target_vocab, stream="target"
                    )
                if int(source_receipt.get("unknown_token_count") or 0) or int(
                    target_receipt.get("unknown_token_count") or 0
                ):
                    raise ValueError(f"frozen supervision row became unrepresentable: {key}")
                if kernel_objective:
                    ground_truth_json_grammar_audited_count += 1
                    if not kerc_ground_truth_serialization_valid(
                        target_ids, code_token_rows
                    ):
                        raise ValueError(
                            "KERC ground-truth target is outside the inference JSON grammar: "
                            f"{key}:{source_index}"
                        )
                sequence = [GLOBAL_BOS_ID]
                sequence.extend(int(value) for value in source_ids)
                sequence.append(SOURCE_TARGET_SEPARATOR_ID)
                sequence.append(target_offset + int(target_vocab["<bos>"]))
                target_start = len(sequence)
                sequence.extend(
                    int(value) if kernel_objective else target_offset + int(value)
                    for value in target_ids
                )
                sequence.append(target_offset + int(target_vocab["<eos>"]))
                if len(sequence) > max_sequence + 1:
                    raise ValueError(f"frozen supervision row requires truncation: {key}")
                sequences.append(np.asarray(sequence, dtype=np.int32))
                training_row_ids.append(row_instance_id)
                mask_starts.append(target_start - 1)
                generator_loss_enabled.append(True)
                compiler_schema_continuation_indices.append(
                    compiler_continuation_indices
                    if kerc_mode and objective == "surface_to_kernel_program_v1"
                    else ()
                )
                compiler_semantic_pointer_indices_by_kind.append(
                    semantic_indices_by_kind
                    if kerc_mode and objective == "surface_to_kernel_program_v1"
                    else {}
                )
                sampling_weight = float(row.get("optimizer_sampling_weight", 1.0))
                if not 0.0 < sampling_weight <= 1.0:
                    raise ValueError(f"invalid supervision sampling weight: {key}:{source_rows - 1}")
                sampling_weights.append(sampling_weight)
                if kerc_mode:
                    unit_allocator_row = materialize_kerc_unit_allocator_row(row)
                    residual = list(row.get("kerc_residual_labels") or [])
                    residual_channels = list(row.get("kerc_residual_channels") or [])
                    verifier_dimensions = list(row.get("kerc_verifier_dimensions") or [])
                    positive = list(row.get("kerc_verifier_positive_labels") or [])
                    negative = (
                        row.get("kerc_verifier_negative")
                        if isinstance(row.get("kerc_verifier_negative"), dict)
                        else {}
                    )
                    negative_labels = list(negative.get("labels") or [])
                    disposition = answer_disposition_from_training_row(row)
                    if (
                        len(residual) != 4
                        or residual_channels != ["interaction", "segment", "token", "exact"]
                        or any(isinstance(value, bool) or not isinstance(value, int) or value not in range(4) for value in residual)
                        or verifier_dimensions != list(KERC_VERIFIER_DIMENSIONS)
                        or positive != [1] * len(KERC_VERIFIER_DIMENSIONS)
                        or len(negative_labels) != len(KERC_VERIFIER_DIMENSIONS)
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value not in (0, 1)
                            for value in negative_labels
                        )
                        or negative_labels.count(0) != 1
                        or negative.get("generator_loss_enabled") is not False
                        or disposition not in ANSWER_DISPOSITION_ORDER
                    ):
                        raise ValueError(f"invalid KERC auxiliary supervision: {key}:{source_rows - 1}")
                    negative_answer = str(negative.get("target") or "")
                    encoded_negative_answer = (
                        compact_learned_compiler_transport_text(
                            negative_answer,
                            transport_version=compiler_transport_version,
                            source=compiler_source_surface,
                        )
                        if objective == "surface_to_kernel_program_v1"
                        else negative_answer
                    )
                    if kernel_objective:
                        negative_ids, negative_receipt = encode_kerc_global_target(
                            encoded_negative_answer,
                            code_vocabulary=code_vocabulary,
                            kernel_offset=kernel_offset,
                            pointer_offset=pointer_offset,
                        )
                    else:
                        negative_ids, negative_receipt = encode_tokens(
                            kerc_surface_tokens(negative_answer),
                            target_vocab,
                            stream="target",
                        )
                    if (
                        not negative_answer
                        or int(negative_receipt.get("unknown_token_count") or 0)
                        or negative_answer == answer
                        or str(negative.get("target_sha256") or "")
                        != "sha256:"
                        + hashlib.sha256(negative_answer.encode("utf-8")).hexdigest()
                    ):
                        raise ValueError(f"invalid KERC verifier corruption: {key}:{source_rows - 1}")
                    negative_sequence = [GLOBAL_BOS_ID]
                    negative_sequence.extend(int(value) for value in source_ids)
                    negative_sequence.append(SOURCE_TARGET_SEPARATOR_ID)
                    negative_sequence.append(target_offset + int(target_vocab["<bos>"]))
                    negative_start = len(negative_sequence)
                    negative_sequence.extend(
                        int(value)
                        if kernel_objective
                        else target_offset + int(value)
                        for value in negative_ids
                    )
                    negative_sequence.append(target_offset + int(target_vocab["<eos>"]))
                    if len(negative_sequence) > max_sequence + 1:
                        raise ValueError(
                            f"KERC verifier corruption requires truncation: {key}"
                        )
                    kerc_residual_rows.append([int(value) for value in residual])
                    kerc_residual_loss_enabled.append(True)
                    kerc_unit_allocator_rows.append(unit_allocator_row)
                    kerc_verifier_rows.append([1] * len(KERC_VERIFIER_DIMENSIONS))
                    kerc_decision_rows.append(
                        ANSWER_DISPOSITION_ORDER.index(disposition)
                    )
                    kerc_decision_loss_enabled.append(True)
                    base_coverage = kerc_training_coverage_labels(row, residual)
                    kerc_coverage_rows.append((*base_coverage, "verifier:positive"))
                    sequences.append(np.asarray(negative_sequence, dtype=np.int32))
                    training_row_ids.append(
                        row_instance_id + ":verifier_negative"
                    )
                    mask_starts.append(negative_start - 1)
                    generator_loss_enabled.append(False)
                    compiler_schema_continuation_indices.append(())
                    compiler_semantic_pointer_indices_by_kind.append({})
                    sampling_weights.append(sampling_weight)
                    kerc_residual_rows.append([int(value) for value in residual])
                    kerc_residual_loss_enabled.append(False)
                    kerc_unit_allocator_rows.append(without_kerc_unit_loss(unit_allocator_row))
                    kerc_verifier_rows.append([int(value) for value in negative_labels])
                    kerc_decision_rows.append(
                        ANSWER_DISPOSITION_ORDER.index(disposition)
                    )
                    kerc_decision_loss_enabled.append(False)
                    failed_dimension = str(negative.get("failed_dimension") or "")
                    if failed_dimension not in KERC_VERIFIER_DIMENSIONS:
                        raise ValueError(
                            f"invalid KERC verifier failed dimension: {key}:{source_rows - 1}"
                        )
                    kerc_coverage_rows.append(
                        (*base_coverage, f"verifier:negative:{failed_dimension}")
                    )
                    row_hashes.append(
                        hashlib.sha256(
                            (
                                json.dumps(trusted_prefix, separators=(",", ":"))
                                + "\0"
                                + prompt
                                + "\0VERIFIER_ONLY\0"
                                + negative_answer
                                + "\0"
                                + json.dumps(negative_labels, separators=(",", ":"))
                            ).encode()
                        ).hexdigest()
                    )
                    for counterfactual in row.get("kerc_context_counterfactuals") or []:
                        if not isinstance(counterfactual, dict):
                            raise ValueError(
                                f"invalid KERC context counterfactual: {key}:{source_rows - 1}"
                            )
                        strategy = str(counterfactual.get("strategy") or "")
                        counter_prompt = str(counterfactual.get("prompt") or "")
                        counter_answer = str(counterfactual.get("target") or "")
                        counter_labels = list(counterfactual.get("labels") or [])
                        failed_dimensions = list(
                            counterfactual.get("failed_dimensions") or []
                        )
                        expected_failed_dimensions = [
                            KERC_VERIFIER_DIMENSIONS[index]
                            for index, value in enumerate(counter_labels)
                            if value == 0
                        ]
                        if (
                            strategy not in context_counterfactual_counts
                            or not counter_prompt
                            or counter_prompt == prompt
                            or not counter_answer
                            or counterfactual.get("generator_loss_enabled") is not False
                            or int(counterfactual.get("unique_source_credit") or 0)
                            or int(counterfactual.get("candidate_generation_credit") or 0)
                            or len(counter_labels) != len(KERC_VERIFIER_DIMENSIONS)
                            or any(
                                isinstance(value, bool)
                                or not isinstance(value, int)
                                or value not in (0, 1)
                                for value in counter_labels
                            )
                            or counter_labels.count(0) != 2
                            or failed_dimensions != expected_failed_dimensions
                            or failed_dimensions
                            != [
                                "semantic_consistency",
                                "answer_decision_consistency",
                            ]
                            or str(counterfactual.get("prompt_sha256") or "")
                            != "sha256:"
                            + hashlib.sha256(counter_prompt.encode()).hexdigest()
                            or str(counterfactual.get("target_sha256") or "")
                            != "sha256:"
                            + hashlib.sha256(counter_answer.encode()).hexdigest()
                        ):
                            raise ValueError(
                                "invalid KERC context counterfactual contract: "
                                f"{key}:{source_rows - 1}:{strategy}"
                            )
                        if structured_source:
                            counter_source_body_ids, counter_source_receipt = (
                                encode_kerc_global_target(
                                    counter_prompt,
                                    code_vocabulary=code_vocabulary,
                                    kernel_offset=kernel_offset,
                                    pointer_offset=pointer_offset,
                                )
                            )
                        else:
                            counter_source_body_ids, counter_source_receipt = (
                                encode_tokens(
                                    kerc_surface_tokens(counter_prompt),
                                    source_vocab,
                                    stream="source",
                                )
                            )
                        counter_source_ids = [
                            *trusted_source_ids,
                            *(
                                counter_source_body_ids
                                if structured_source
                                else [
                                    source_offset + int(value)
                                    for value in counter_source_body_ids
                                ]
                            ),
                        ]
                        if kernel_objective:
                            encoded_counter_answer = (
                                compact_learned_compiler_transport_text(
                                    counter_answer,
                                    transport_version=compiler_transport_version,
                                    source=(
                                        str(
                                            (
                                                json.loads(counter_prompt)
                                                if objective
                                                == "surface_to_kernel_program_v1"
                                                else {}
                                            ).get("source_surface")
                                            or ""
                                        )
                                        if objective
                                        == "surface_to_kernel_program_v1"
                                        and compiler_transport_version
                                        == LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION
                                        else None
                                    ),
                                )
                                if objective == "surface_to_kernel_program_v1"
                                else counter_answer
                            )
                            counter_target_ids, counter_target_receipt = (
                                encode_kerc_global_target(
                                    encoded_counter_answer,
                                    code_vocabulary=code_vocabulary,
                                    kernel_offset=kernel_offset,
                                    pointer_offset=pointer_offset,
                                )
                            )
                        else:
                            counter_target_ids, counter_target_receipt = encode_tokens(
                                kerc_surface_tokens(counter_answer),
                                target_vocab,
                                stream="target",
                            )
                        if int(counter_source_receipt.get("unknown_token_count") or 0) or int(
                            counter_target_receipt.get("unknown_token_count") or 0
                        ):
                            raise ValueError(
                                "unrepresentable KERC context counterfactual: "
                                f"{key}:{source_rows - 1}:{strategy}"
                            )
                        counter_sequence = [GLOBAL_BOS_ID]
                        counter_sequence.extend(int(value) for value in counter_source_ids)
                        counter_sequence.append(SOURCE_TARGET_SEPARATOR_ID)
                        counter_sequence.append(target_offset + int(target_vocab["<bos>"]))
                        counter_start = len(counter_sequence)
                        counter_sequence.extend(
                            int(value)
                            if kernel_objective
                            else target_offset + int(value)
                            for value in counter_target_ids
                        )
                        counter_sequence.append(
                            target_offset + int(target_vocab["<eos>"])
                        )
                        if len(counter_sequence) > max_sequence + 1:
                            raise ValueError(
                                "KERC context counterfactual requires truncation: "
                                f"{key}:{strategy}"
                            )
                        sequences.append(np.asarray(counter_sequence, dtype=np.int32))
                        training_row_ids.append(
                            row_instance_id + f":counterfactual:{strategy}"
                        )
                        mask_starts.append(counter_start - 1)
                        generator_loss_enabled.append(False)
                        compiler_schema_continuation_indices.append(())
                        compiler_semantic_pointer_indices_by_kind.append({})
                        sampling_weights.append(sampling_weight)
                        kerc_residual_rows.append([int(value) for value in residual])
                        kerc_residual_loss_enabled.append(False)
                        kerc_unit_allocator_rows.append(
                            without_kerc_unit_loss(unit_allocator_row)
                        )
                        kerc_verifier_rows.append(
                            [int(value) for value in counter_labels]
                        )
                        kerc_decision_rows.append(
                            ANSWER_DISPOSITION_ORDER.index(disposition)
                        )
                        kerc_decision_loss_enabled.append(False)
                        kerc_coverage_rows.append(
                            (
                                *base_coverage,
                                f"verifier:counterfactual:{strategy}",
                            )
                        )
                        context_counterfactual_counts[strategy] += 1
                        row_hashes.append(
                            hashlib.sha256(
                                (
                                    json.dumps(
                                        trusted_prefix, separators=(",", ":")
                                    )
                                    + "\0CONTEXT_COUNTERFACTUAL\0"
                                    + strategy
                                    + "\0"
                                    + counter_prompt
                                    + "\0"
                                    + counter_answer
                                    + "\0"
                                    + json.dumps(
                                        counter_labels, separators=(",", ":")
                                    )
                                ).encode()
                            ).hexdigest()
                        )
                row_hashes.append(
                    hashlib.sha256(
                        (
                            json.dumps(trusted_prefix, separators=(",", ":"))
                            + "\0"
                            + prompt
                            + "\0"
                            + answer
                        ).encode()
                    ).hexdigest()
                )
                observed_rows += 1
        if source_rows != int(artifact.get("row_count") or 0):
            raise ValueError(f"supervision row count changed: {key}")
        artifact_receipts.append(
            {
                "key": key,
                "path": relative(path),
                "sha256": str(artifact["sha256"]),
                "row_count": source_rows,
                "selected_row_count": observed_rows,
                "bounded_source_row_limit": int(bounded_source_row_limit),
            }
        )

    materialized_width = max((len(sequence) - 1 for sequence in sequences), default=1)
    if materialized_width > max_sequence:
        raise ValueError("materialized supervision width exceeds its sequence contract")
    termination_id = target_offset + int(target_vocab["<eos>"])
    byte_begin_id = target_offset + int(target_vocab[TARGET_BYTE_BEGIN])
    byte_end_id = target_offset + int(target_vocab[TARGET_BYTE_END])
    code_boundary_ids: list[int] = []
    if kerc_mode:
        for vocab, offset in (
            (code_vocabulary.get("kernel_vocab") or {}, kernel_offset),
            (code_vocabulary.get("pointer_vocab") or {}, pointer_offset),
        ):
            for token in (TARGET_BYTE_BEGIN, TARGET_BYTE_END):
                if token not in vocab:
                    raise ValueError("KERC code vocabulary is missing byte boundaries")
                code_boundary_ids.append(offset + int(vocab[token]))
    input_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    loss_rows: list[np.ndarray] = []
    compiler_schema_continuation_loss_weight = float(
        config["training"]["kerc_compiler_schema_continuation_loss_weight"]
    )
    compiler_semantic_pointer_loss_weight = float(
        config["training"].get("kerc_compiler_semantic_pointer_loss_weight", 1.0)
    )
    configured_semantic_weights = config["training"].get(
        "kerc_compiler_semantic_pointer_loss_weights_by_kind"
    )
    compiler_semantic_pointer_loss_weights_by_kind = {
        kind: float(
            (configured_semantic_weights or {}).get(
                kind, compiler_semantic_pointer_loss_weight
            )
        )
        for kind in KERC_COMPILER_SEMANTIC_TARGET_KINDS
    }
    compiler_semantic_pointer_preweight_loss_mass_by_kind = {
        kind: 0.0 for kind in KERC_COMPILER_SEMANTIC_TARGET_KINDS
    }
    compiler_semantic_pointer_postweight_loss_mass_by_kind = {
        kind: 0.0 for kind in KERC_COMPILER_SEMANTIC_TARGET_KINDS
    }
    compiler_semantic_pointer_preweight_loss_histogram_by_kind = {
        kind: Counter() for kind in KERC_COMPILER_SEMANTIC_TARGET_KINDS
    }
    compiler_schema_continuation_position_count = 0
    compiler_schema_continuation_preweight_loss_mass = 0.0
    compiler_schema_continuation_postweight_loss_mass = 0.0
    compiler_schema_continuation_preweight_loss_histogram: Counter[str] = Counter()
    compiler_semantic_pointer_position_count = 0
    for (
        sequence,
        mask_start,
        generator_enabled,
        continuation_indices,
        semantic_pointer_indices_by_kind,
    ) in zip(
        sequences,
        mask_starts,
        generator_loss_enabled,
        compiler_schema_continuation_indices,
        compiler_semantic_pointer_indices_by_kind,
        strict=True,
    ):
        row_inputs = np.asarray(sequence[:-1], dtype=np.int32)
        row_labels = np.asarray(sequence[1:], dtype=np.int32)
        row_mask = np.zeros(len(row_inputs), dtype=np.uint8)
        if generator_enabled:
            row_mask[mask_start:] = 1
        row_loss = row_mask.astype(np.float32)
        row_loss[(row_mask == 1) & (row_labels == termination_id)] = float(
            config["training"]["termination_loss_weight"]
        )
        row_loss[
            (row_mask == 1)
            & ((row_labels == byte_begin_id) | (row_labels == byte_end_id))
        ] = float(config["training"]["byte_boundary_loss_weight"])
        if code_boundary_ids:
            row_loss[(row_mask == 1) & np.isin(row_labels, code_boundary_ids)] = float(
                config["training"]["byte_boundary_loss_weight"]
            )
        for continuation_index in continuation_indices:
            loss_index = mask_start + int(continuation_index)
            if (
                not generator_enabled
                or loss_index < 0
                or loss_index >= len(row_loss)
                or not row_mask[loss_index]
            ):
                raise ValueError(
                    "KERC compiler schema continuation loss position is invalid"
                )
            compiler_schema_continuation_preweight_loss_mass += float(
                row_loss[loss_index]
            )
            compiler_schema_continuation_preweight_loss_histogram[
                str(float(row_loss[loss_index]))
            ] += 1
            row_loss[loss_index] = max(
                float(row_loss[loss_index]),
                compiler_schema_continuation_loss_weight,
            )
            compiler_schema_continuation_postweight_loss_mass += float(
                row_loss[loss_index]
            )
            compiler_schema_continuation_position_count += 1
        for kind in KERC_COMPILER_SEMANTIC_TARGET_KINDS:
            kind_weight = compiler_semantic_pointer_loss_weights_by_kind[kind]
            for semantic_pointer_index in (
                semantic_pointer_indices_by_kind.get(kind) or ()
            ):
                loss_index = mask_start + int(semantic_pointer_index)
                if (
                    not generator_enabled
                    or loss_index < 0
                    or loss_index >= len(row_loss)
                    or not row_mask[loss_index]
                ):
                    raise ValueError(
                        "KERC compiler semantic-pointer loss position is invalid"
                    )
                compiler_semantic_pointer_preweight_loss_mass_by_kind[
                    kind
                ] += float(row_loss[loss_index])
                compiler_semantic_pointer_preweight_loss_histogram_by_kind[
                    kind
                ][str(float(row_loss[loss_index]))] += 1
                row_loss[loss_index] = max(
                    float(row_loss[loss_index]),
                    kind_weight,
                )
                compiler_semantic_pointer_postweight_loss_mass_by_kind[
                    kind
                ] += float(row_loss[loss_index])
                compiler_semantic_pointer_position_counts_by_kind[kind] += 1
                compiler_semantic_pointer_position_count += 1
        input_rows.append(row_inputs)
        label_rows.append(row_labels)
        mask_rows.append(row_mask)
        loss_rows.append(row_loss)
    if kerc_mode:
        inputs = RaggedRows(input_rows, dtype=np.int32)
        labels = RaggedRows(label_rows, dtype=np.int32)
        mask = RaggedRows(mask_rows, dtype=np.uint8)
        loss_mask = RaggedRows(loss_rows, dtype=np.float32)
    else:
        inputs = np.zeros((len(sequences), materialized_width), dtype=np.int32)
        labels = np.zeros((len(sequences), materialized_width), dtype=np.int32)
        mask = np.zeros((len(sequences), materialized_width), dtype=np.uint8)
        loss_mask = np.zeros((len(sequences), materialized_width), dtype=np.float32)
        for index, (row_inputs, row_labels, row_mask, row_loss) in enumerate(
            zip(input_rows, label_rows, mask_rows, loss_rows)
        ):
            width = len(row_inputs)
            inputs[index, :width] = row_inputs
            labels[index, :width] = row_labels
            mask[index, :width] = row_mask
            loss_mask[index, :width] = row_loss
    receipt = {
        "policy": receipt_policy,
        "target_id": target["target_id"],
        "artifacts": artifact_receipts,
        "row_count": len(sequences),
        "generator_training_row_count": sum(generator_loss_enabled),
        "verifier_only_row_count": len(generator_loss_enabled) - sum(generator_loss_enabled),
        "target_positions": int(mask.sum()),
        "weighted_loss_positions": float(loss_mask.sum()),
        "sampling_weight_sum": float(sum(sampling_weights)),
        "sampling_weight_minimum": float(min(sampling_weights or [1.0])),
        "sampling_weight_maximum": float(max(sampling_weights or [1.0])),
        "termination_loss_weight": float(config["training"]["termination_loss_weight"]),
        "byte_boundary_loss_weight": float(config["training"]["byte_boundary_loss_weight"]),
        "kerc_compiler_schema_continuation_loss_weight": (
            compiler_schema_continuation_loss_weight
        ),
        "kerc_compiler_schema_continuation_position_count": (
            compiler_schema_continuation_position_count
        ),
        "kerc_compiler_schema_continuation_preweight_loss_mass": (
            compiler_schema_continuation_preweight_loss_mass
        ),
        "kerc_compiler_schema_continuation_postweight_loss_mass": (
            compiler_schema_continuation_postweight_loss_mass
        ),
        "kerc_compiler_schema_continuation_preweight_loss_histogram": dict(
            sorted(
                compiler_schema_continuation_preweight_loss_histogram.items()
            )
        ),
        "kerc_compiler_schema_continuation_semantic_values_added": 0,
        "kerc_compiler_schema_continuation_position_mapping": (
            "logical_atom_to_exact_encoded_half_open_range_v1"
        ),
        "kerc_compiler_semantic_pointer_loss_weight": (
            compiler_semantic_pointer_loss_weight
        ),
        "kerc_compiler_semantic_pointer_loss_weights_by_kind": dict(
            compiler_semantic_pointer_loss_weights_by_kind
        ),
        "kerc_compiler_semantic_pointer_position_count": (
            compiler_semantic_pointer_position_count
        ),
        "kerc_compiler_semantic_pointer_position_counts_by_kind": dict(
            compiler_semantic_pointer_position_counts_by_kind
        ),
        "kerc_compiler_semantic_pointer_kind_policy": (
            "project_theseus_kerc_compiler_semantic_target_kinds_v1"
        ),
        "kerc_compiler_semantic_pointer_preweight_loss_mass_by_kind": dict(
            compiler_semantic_pointer_preweight_loss_mass_by_kind
        ),
        "kerc_compiler_semantic_pointer_postweight_loss_mass_by_kind": dict(
            compiler_semantic_pointer_postweight_loss_mass_by_kind
        ),
        "kerc_compiler_semantic_pointer_preweight_loss_histogram_by_kind": {
            kind: dict(
                sorted(
                    compiler_semantic_pointer_preweight_loss_histogram_by_kind[
                        kind
                    ].items()
                )
            )
            for kind in KERC_COMPILER_SEMANTIC_TARGET_KINDS
        },
        "kerc_compiler_semantic_pointer_position_mapping": (
            "existing_target_atom_to_exact_encoded_half_open_range_v1"
        ),
        "kerc_compiler_semantic_pointer_values_added": 0,
        "sequence_width": materialized_width,
        "maximum_sequence_tokens_contract": max_sequence,
        "staged_padding_columns_elided": max_sequence - materialized_width,
        "sequence_width_source": (
            "objective_override" if maximum_sequence_tokens is not None else "base_stage"
        ),
        "storage_layout": (
            "ragged_rows_shared_shifted_token_storage_v2"
            if kerc_mode
            else "dense_rows_v1"
        ),
        "physical_array_bytes": (
            sum(int(sequence.nbytes) for sequence in sequences)
            + mask.physical_bytes
            + loss_mask.physical_bytes
            if kerc_mode
            else int(inputs.nbytes + labels.nbytes + mask.nbytes + loss_mask.nbytes)
        ),
        "dense_equivalent_array_bytes": int(
            len(sequences)
            * materialized_width
            * (
                np.dtype(np.int32).itemsize * 2
                + np.dtype(np.uint8).itemsize
                + np.dtype(np.float32).itemsize
            )
        ),
        "content_digest": hashlib.sha256("\n".join(row_hashes).encode()).hexdigest(),
        "bounded_selection": {
            "policy": "project_theseus_content_ranked_objective_stratified_canary_v1",
            "active": bool(bounded_source_row_limit),
            "maximum_source_rows_per_artifact": int(bounded_source_row_limit),
            "full_artifact_identity_and_boundary_scan_required": True,
            "capability_claim": "NONE" if bounded_source_row_limit else "NOT_APPLICABLE",
        },
        "generator_visible_fields": ["trusted_source_prefix_tokens", "prompt"],
        "trusted_source_prefix_injected_separately_from_raw_text": True,
        "evaluator_only_fields": ["target", "target_sha256", "source_identity"],
        "source_truncation_count": 0,
        "target_truncation_count": 0,
        "objective_filter": list(objective_filter),
        "dual_code_vocabulary_sha256": (
            code_vocabulary.get("contract_sha256") if kerc_mode else ""
        ),
        "kernel_target_token_offset": kernel_offset if kerc_mode else 0,
        "pointer_target_token_offset": pointer_offset if kerc_mode else 0,
        "dual_code_byte_boundary_ids": code_boundary_ids,
        "ground_truth_json_grammar_audited_count": (
            ground_truth_json_grammar_audited_count if kerc_mode else 0
        ),
        "ground_truth_json_grammar_rejected_count": 0,
        "compact_compiler_transport": (
            {
                "policy": (
                    LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_POLICY
                    if compiler_transport_version
                    == LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION
                    else (
                        LEARNED_COMPILER_SEMANTIC_POINTER_TRANSPORT_POLICY
                        if compiler_transport_version
                        == LEARNED_COMPILER_SEMANTIC_POINTER_TRANSPORT_VERSION
                        else LEARNED_COMPILER_COMPACT_TRANSPORT_POLICY
                    )
                ),
                "version": compiler_transport_version,
                "migration_boundary": (
                    "frozen_legacy_semantic_artifact_to_training_and_runtime_wire"
                ),
                "generator_row_count": compact_compiler_transport_generator_rows,
                "legacy_target_token_count": (
                    compact_compiler_transport_legacy_target_tokens
                ),
                "encoded_target_token_count": (
                    compact_compiler_transport_encoded_target_tokens
                ),
                "target_tokens_elided": (
                    compact_compiler_transport_legacy_target_tokens
                    - compact_compiler_transport_encoded_target_tokens
                ),
                "exact_reconstruction_required": True,
                "learned_semantic_values_elided": 0,
                "allocator_owned_unit_fidelity_rows_removed": (
                    compact_compiler_transport_allocator_values_removed
                ),
                "allocator_attachment_generation_credit": 0,
                "compiler_allocator_ownership_enforced": (
                    compiler_transport_version
                    in {
                        LEARNED_COMPILER_SEMANTIC_POINTER_TRANSPORT_VERSION,
                        LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION,
                    }
                ),
                "source_span_materialization_generation_credit": 0,
                "source_span_materialization_capability_credit": 0,
                "source_span_pointer_requires_prompt_source_and_declared_node_span": (
                    compiler_transport_version
                    == LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION
                ),
                "target_metadata_visible_to_generator": False,
                "deterministic_generation_credit": 0,
            }
            if kerc_mode
            else {}
        ),
        "kerc_verifier_dimensions": (
            list(KERC_VERIFIER_DIMENSIONS) if kerc_mode else []
        ),
        "kerc_context_counterfactual_counts": (
            context_counterfactual_counts if kerc_mode else {}
        ),
        "kerc_context_counterfactuals_receive_generator_loss": False,
        "kerc_verifier_only_rows_receive_residual_loss": False,
        "kerc_verifier_only_rows_receive_decision_loss": False,
        "kerc_residual_supervision_row_count": (
            sum(kerc_residual_loss_enabled) if kerc_mode else 0
        ),
        "kerc_per_unit_allocator_supervision_row_count": (
            sum(
                row is not None and bool(np.asarray(row["loss_mask"]).any())
                for row in kerc_unit_allocator_rows
            )
            if kerc_mode
            else 0
        ),
        "kerc_per_unit_allocator_supervised_unit_count": (
            sum(
                int(np.asarray(row["loss_mask"]).sum())
                for row in kerc_unit_allocator_rows
                if row is not None
            )
            if kerc_mode
            else 0
        ),
        "legacy_four_channel_allocator_training_authority": not any(
            row is not None and bool(np.asarray(row["loss_mask"]).any())
            for row in kerc_unit_allocator_rows
        )
        if kerc_mode
        else False,
        "kerc_context_counterfactuals_receive_unique_source_credit": 0,
        "kerc_context_counterfactuals_receive_candidate_generation_credit": 0,
        "canary_coverage_catalog": (
            {
                label: sum(label in labels for labels in kerc_coverage_rows)
                for label in KERC_CANARY_REQUIRED_COVERAGE
            }
            if kerc_mode
            else {}
        ),
        "canary_coverage_labels_are_model_inputs": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return SimpleNamespace(
        inputs=inputs,
        labels=labels,
        mask=mask,
        loss_mask=loss_mask,
        sample_weights=np.asarray(sampling_weights, dtype=np.float64),
        kerc_residual_labels=(
            np.asarray(kerc_residual_rows, dtype=np.int32) if kerc_mode else None
        ),
        kerc_residual_loss_mask=(
            np.asarray(kerc_residual_loss_enabled, dtype=np.float32)
            if kerc_mode
            else None
        ),
        kerc_unit_allocator_rows=(
            tuple(kerc_unit_allocator_rows) if kerc_mode else None
        ),
        kerc_verifier_labels=(
            np.asarray(kerc_verifier_rows, dtype=np.float32) if kerc_mode else None
        ),
        kerc_decision_labels=(
            np.asarray(kerc_decision_rows, dtype=np.int32) if kerc_mode else None
        ),
        kerc_decision_loss_mask=(
            np.asarray(kerc_decision_loss_enabled, dtype=np.float32)
            if kerc_mode
            else None
        ),
        kerc_coverage_labels=(tuple(kerc_coverage_rows) if kerc_mode else None),
        training_row_ids=tuple(training_row_ids),
        receipt=receipt,
    )


def select_bounded_supervision_rows(
    path: Path,
    *,
    split: str,
    objective_filter: tuple[str, ...],
    maximum_rows: int,
) -> tuple[int, list[tuple[int, dict[str, Any]]]]:
    """Compatibility wrapper returning selected rows after an offset-only scan."""

    source_rows, references = select_bounded_supervision_row_references(
        path,
        split=split,
        objective_filter=objective_filter,
        maximum_rows=maximum_rows,
    )
    return source_rows, list(iter_bounded_supervision_rows(path, references))


def iter_bounded_supervision_rows(
    path: Path,
    references: list[tuple[int, str, int]],
) -> Any:
    """Yield selected JSONL rows one at a time in deterministic row-id order."""

    with path.open("rb") as handle:
        for source_index, expected_row_id, byte_offset in references:
            handle.seek(byte_offset)
            line = handle.readline()
            if not line:
                raise ValueError(
                    f"bounded supervision row offset vanished: {path}:{source_index}"
                )
            row = json.loads(line)
            if str(row.get("row_id") or "") != expected_row_id:
                raise ValueError(
                    f"bounded supervision row identity changed: {path}:{source_index}"
                )
            yield source_index, row


def select_bounded_supervision_row_references(
    path: Path,
    *,
    split: str,
    objective_filter: tuple[str, ...],
    maximum_rows: int,
) -> tuple[int, list[tuple[int, str, int]]]:
    """Select rows using byte-offset metadata while validating the full artifact."""

    objectives = tuple(dict.fromkeys(objective_filter))
    if maximum_rows <= 0 or not objectives or maximum_rows < len(objectives):
        raise ValueError("bounded supervision selection requires one row per objective")
    heaps: dict[str, list[tuple[int, str, int, int]]] = {
        objective: [] for objective in objectives
    }
    coverage_best: dict[str, tuple[int, str, int, int]] = {}
    observed_residual_labels: set[str] = set()
    observed_decision_labels: set[str] = set()
    observed_verifier_labels: set[str] = set()
    observed_context_counterfactual_labels: set[str] = set()
    source_rows = 0
    with path.open("rb") as handle:
        while True:
            byte_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            source_index = source_rows
            row = json.loads(line)
            source_rows += 1
            if row.get("split") != split or row.get("public_benchmark") is not False:
                raise ValueError(
                    f"invalid bounded supervision boundary: {path}:{source_index}"
                )
            objective = str(row.get("objective") or "")
            if objective not in heaps:
                continue
            row_id = str(row.get("row_id") or "")
            if not row_id:
                raise ValueError(f"bounded supervision row lacks row_id: {path}:{source_index}")
            rank = int.from_bytes(
                hashlib.sha256(
                    b"project_theseus_kerc_canary_row_v1\0" + row_id.encode()
                ).digest(),
                "big",
            )
            entry = (-rank, row_id, source_index, byte_offset)
            heap = heaps[objective]
            heapq.heappush(heap, entry)
            if len(heap) > maximum_rows:
                heapq.heappop(heap)
            residual = list(row.get("kerc_residual_labels") or [])
            labels = {
                f"objective:{objective}",
                f"decision:{row.get('kerc_answer_disposition')}",
                "interaction:present"
                if residual and int(residual[0]) > 0
                else "interaction:absent",
            }
            observed_decision_labels.add(
                f"decision:{row.get('kerc_answer_disposition')}"
            )
            failed = str(
                ((row.get("kerc_verifier_negative") or {}).get("failed_dimension"))
                or ""
            )
            if failed:
                verifier_label = f"verifier:{failed}"
                labels.add(verifier_label)
                observed_verifier_labels.add(verifier_label)
            for counterfactual in row.get("kerc_context_counterfactuals") or []:
                strategy = str(counterfactual.get("strategy") or "")
                if strategy:
                    counterfactual_label = f"verifier:counterfactual:{strategy}"
                    labels.add(counterfactual_label)
                    observed_context_counterfactual_labels.add(
                        counterfactual_label
                    )
            if len(residual) == 4:
                for channel, value in zip(
                    ("interaction", "segment", "token", "exact"), residual
                ):
                    label = f"residual:{channel}:{int(value)}"
                    labels.add(label)
                    observed_residual_labels.add(label)
            positive_rank_entry = (rank, row_id, source_index, byte_offset)
            for label in labels:
                prior = coverage_best.get(label)
                if prior is None or positive_rank_entry[:2] < prior[:2]:
                    coverage_best[label] = positive_rank_entry
    required_labels = {
        *(f"objective:{objective}" for objective in objectives),
        *observed_decision_labels,
        *observed_verifier_labels,
        "interaction:present",
        "interaction:absent",
        *observed_context_counterfactual_labels,
        *observed_residual_labels,
    }
    expected_decisions = {"decision:ANSWER", "decision:CLARIFY", "decision:ABSTAIN"}
    if not expected_decisions.issubset(observed_decision_labels):
        raise ValueError("bounded supervision source lacks required decision classes")
    missing = sorted(required_labels.difference(coverage_best))
    if missing:
        raise ValueError("bounded supervision coverage missing: " + ",".join(missing))
    chosen: dict[str, tuple[int, str, int]] = {}
    for label in sorted(required_labels):
        _rank, row_id, source_index, byte_offset = coverage_best[label]
        chosen[row_id] = (source_index, row_id, byte_offset)
    if len(chosen) > maximum_rows:
        raise ValueError("bounded supervision limit cannot cover required KERC labels")
    fill = sorted(
        (
            (-negative_rank, row_id, source_index, byte_offset)
            for heap in heaps.values()
            for negative_rank, row_id, source_index, byte_offset in heap
        ),
        key=lambda entry: (entry[0], entry[1]),
    )
    for _rank, row_id, source_index, byte_offset in fill:
        chosen.setdefault(row_id, (source_index, row_id, byte_offset))
        if len(chosen) >= maximum_rows:
            break
    selected = list(chosen.values())
    selected.sort(key=lambda item: item[1])
    return source_rows, selected


def kerc_training_coverage_labels(
    row: dict[str, Any], residual: list[int]
) -> tuple[str, ...]:
    """Classify training-only mechanics coverage without changing model-visible text."""

    labels = [f"objective:{str(row.get('objective') or '')}"]
    labels.append("interaction:present" if residual[0] > 0 else "interaction:absent")
    for channel, value in zip(("interaction", "segment", "token", "exact"), residual):
        if value > 0:
            labels.append(f"residual:{channel}:active")
    disposition = answer_disposition_from_training_row(row)
    if disposition:
        labels.append(f"decision:{disposition}")
    return tuple(labels)


def answer_disposition_from_training_row(row: dict[str, Any]) -> str:
    """Read a supervised decision label for sampler accounting, never generation."""

    explicit = str(row.get("kerc_answer_disposition") or "")
    if explicit in ANSWER_DISPOSITION_ORDER:
        return explicit

    def visit(value: Any) -> str:
        if isinstance(value, dict):
            decision = value.get("decision")
            if isinstance(decision, dict):
                disposition = str(decision.get("disposition") or "")
                if disposition in {"ANSWER", "PARTIAL", "CLARIFY", "ABSTAIN"}:
                    return disposition
            for child in value.values():
                found = visit(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = visit(child)
                if found:
                    return found
        return ""

    for field in ("target", "prompt"):
        try:
            found = visit(json.loads(str(row.get(field) or "")))
        except (TypeError, ValueError, json.JSONDecodeError):
            found = ""
        if found:
            return found
    return ""


def candidate_evaluation_execution_policy(plan: dict[str, Any]) -> dict[str, Any]:
    """Bind evaluation to the graph-shaping policy that produced a candidate."""

    candidate = dict(
        ((plan.get("candidate_canary_lease") or {}).get("execution_policy") or {})
    )
    return {
        "attention_query_chunk_size": int(
            candidate.get("attention_query_chunk_size") or 0
        ),
        "attention_key_chunk_size": int(
            candidate.get("attention_key_chunk_size") or 0
        ),
        "compact_encoder_decoder_partitions": bool(
            candidate.get("compact_encoder_decoder_partitions", False)
        ),
        "compact_partition_width_quantum": int(
            candidate.get("compact_partition_width_quantum") or 0
        ),
        "gradient_checkpointing": False,
    }


def candidate_bound_evaluation_plan(
    plan: dict[str, Any], candidate_lease: dict[str, Any] | None
) -> dict[str, Any]:
    """Carry graph-shaping candidate authority into embedded evaluation."""

    return {**plan, "candidate_canary_lease": candidate_lease}


def terminal_candidate_behavior_rows(
    candidate_lease: dict[str, Any] | None,
) -> int:
    """Run the costly heldout decoder only for a candidate's terminal rung."""

    if not candidate_lease or candidate_lease.get("candidate_id") not in {
        "rdc_kerc_adequacy",
        "rdc_kerc_k5_adequacy",
        "rdc_kerc_k5_overfit",
    }:
        return 0
    requested_steps = int(candidate_lease.get("requested_steps") or 0)
    maximum_steps = int(
        ((candidate_lease.get("budgets") or {}).get("max_steps")) or 0
    )
    if requested_steps <= 0 or maximum_steps <= 0:
        raise ValueError("candidate behavior evaluation requires positive step budgets")
    return (
        int(candidate_lease.get("behavior_eval_rows") or 0)
        if requested_steps == maximum_steps
        else 0
    )


def evaluate_target(
    config: dict[str, Any],
    base: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    *,
    metadata: dict[str, Any],
    mx: Any,
    nn: Any,
    split: str = "private_dev",
    maximum_rows: int = 0,
    selected_row_ids: tuple[str, ...] = (),
    selection_namespace: str = "",
    selection_contract_sha256: str = "",
) -> dict[str, Any]:
    """Evaluate frozen rows while keeping answers outside the generation call."""

    output = resolve(str(target["receipt"])).with_name(
        f"evaluation_{split}_receipt.json"
    )
    checkpoint = resolve(str(target["checkpoint"]))
    checkpoint_sha256 = sha256_file(checkpoint)
    evaluation_contract_sha256 = hashlib.sha256(
        json.dumps(
            config["evaluation"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    progress_policy = (
        "project_theseus_transactional_model_evaluation_progress_v1"
    )
    progress_identity = {
        "target_id": target["target_id"],
        "split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "maximum_rows_per_artifact": int(maximum_rows),
    }
    if selected_row_ids:
        if maximum_rows != len(selected_row_ids):
            raise ValueError(
                "frozen evaluation selection must match the maximum row count"
            )
        if (
            len(set(selected_row_ids)) != len(selected_row_ids)
            or not selection_namespace
            or len(selection_contract_sha256) != 64
        ):
            raise ValueError("frozen evaluation selection contract is invalid")
        progress_identity.update(
            {
                "selection_namespace": selection_namespace,
                "selection_contract_sha256": selection_contract_sha256,
            }
        )
    resumed_rows: list[dict[str, Any]] = []
    if output.is_file():
        prior_evaluation = read_json(output)
        identity_matches = all(
            prior_evaluation.get(key) == value
            for key, value in progress_identity.items()
        )
        if not identity_matches:
            raise ValueError(
                "evaluation progress identity mismatch; use a fresh output namespace"
            )
        if prior_evaluation.get("complete") is True:
            return {
                **prior_evaluation,
                "rows": {
                    "path": relative(output),
                    "embedded_row_count": int(
                        prior_evaluation.get("row_count") or 0
                    ),
                },
            }
        if prior_evaluation.get("policy") != progress_policy:
            raise ValueError("evaluation progress policy mismatch")
        resumed_rows = list(prior_evaluation.get("rows") or [])
        if len(
            {
                str(row.get("row_id") or "")
                for row in resumed_rows
            }
        ) != len(resumed_rows):
            raise ValueError("evaluation progress contains duplicate row identities")

    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    evaluated_vocab_size = int(
        target.get("vocab_size") or plan["models"]["vocab_size"]
    )
    evaluation_execution_policy = candidate_evaluation_execution_policy(plan)
    model = build_model(
        CausalTransformerConfig(
            vocab_size=evaluated_vocab_size, **target["model"]
        ),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=build_source_to_target_lookup(
            base,
            metadata,
            vocab_size=evaluated_vocab_size,
            identity_ranges=target_copy_identity_ranges(target),
        ),
        gradient_checkpointing=evaluation_execution_policy[
            "gradient_checkpointing"
        ],
        attention_query_chunk_size=evaluation_execution_policy[
            "attention_query_chunk_size"
        ],
        attention_key_chunk_size=evaluation_execution_policy[
            "attention_key_chunk_size"
        ],
        compact_encoder_decoder_partitions=evaluation_execution_policy[
            "compact_encoder_decoder_partitions"
        ],
        compact_partition_width_quantum=evaluation_execution_policy[
            "compact_partition_width_quantum"
        ],
    )
    if target.get("role") == "language_expert":
        shared = resolve(str(target.get("shared_trunk_checkpoint") or ""))
        if not shared.is_file():
            raise ValueError("expert evaluation requires shared trunk checkpoint")
        model.load_weights(str(shared), strict=False)
        model.load_weights(str(checkpoint), strict=False)
    else:
        model.load_weights(str(checkpoint))
    mx.eval(model.parameters())
    model.eval()
    artifacts = target.get("supervision_artifacts") or {}
    selected = [
        (key, row)
        for key, row in artifacts.items()
        if key == split or str(key).endswith(f":{split}")
    ]
    rows: list[dict[str, Any]] = list(resumed_rows)
    completed_row_ids = {
        str(row.get("row_id") or "") for row in resumed_rows
    }
    evaluation_artifacts: list[dict[str, Any]] = []
    for key, artifact in selected:
        path = resolve(str((artifact or {}).get("path") or ""))
        if not path.is_file() or sha256_file(path) != str((artifact or {}).get("sha256") or ""):
            raise ValueError(f"evaluation artifact identity mismatch: {key}")
        evaluation_artifacts.append(
            {
                "key": key,
                "path": relative(path),
                "sha256": str(artifact["sha256"]),
                "row_count": int(artifact["row_count"]),
            }
        )
        with path.open(encoding="utf-8") as handle:
            source_rows = [json.loads(line) for line in handle]
        if selected_row_ids:
            rows_by_id = {
                str(row.get("row_id") or ""): row for row in source_rows
            }
            if len(rows_by_id) != len(source_rows):
                raise ValueError(f"evaluation artifact has duplicate row identities: {key}")
            missing = [
                row_id for row_id in selected_row_ids if row_id not in rows_by_id
            ]
            if missing:
                raise ValueError(
                    "frozen evaluation selection row identity missing: "
                    + ",".join(missing)
                )
            source_rows = [rows_by_id[row_id] for row_id in selected_row_ids]
        elif maximum_rows > 0:
            source_rows = sorted(
                source_rows,
                key=lambda row: hashlib.sha256(
                    (
                        "t0a_private_rdc_kerc_source_disjoint_v1:"
                        + str(row.get("row_id") or row.get("source_identity") or "")
                    ).encode()
                ).hexdigest(),
            )[:maximum_rows]
        for row in source_rows:
                if row.get("split") != split or row.get("public_benchmark") is not False:
                    raise ValueError(f"invalid evaluation boundary: {key}")
                row_id = str(row.get("row_id") or "")
                if not row_id:
                    raise ValueError(f"evaluation row lacks row_id: {key}")
                if row_id in completed_row_ids:
                    continue
                if target.get("role") == "kerc_english_candidate":
                    generated, generation = generate_kerc_pipeline_text(
                        model,
                        str(row.get("prompt") or ""),
                        source_vocab,
                        target_vocab,
                        base,
                        target=target,
                        max_tokens=int(
                            config["evaluation"]["kerc_decode_max_target_tokens"]
                        ),
                        max_source_tokens=int(
                            config["kernel_english_training"]["maximum_sequence_tokens"]
                        ),
                        beam_width=int(config["evaluation"]["kerc_beam_width"]),
                        branching_factor=int(
                            config["evaluation"]["kerc_branching_factor"]
                        ),
                        length_penalty=float(
                            config["evaluation"]["length_penalty"]
                        ),
                        interaction_id=f"kerc-eval:{split}:{row.get('row_id')}",
                        mx=mx,
                    )
                else:
                    generated, generation = generate_model_text(
                        model,
                        str(row.get("prompt") or ""),
                        source_vocab,
                        target_vocab,
                        base,
                        max_tokens=int(
                            config["evaluation"]["decode_max_target_tokens"]
                        ),
                        max_source_tokens=int(
                            config["supervision"]["maximum_source_encoded_tokens"]
                        ),
                        beam_width=int(config["evaluation"]["beam_width"]),
                        branching_factor=int(
                            config["evaluation"]["branching_factor"]
                        ),
                        length_penalty=float(
                            config["evaluation"]["length_penalty"]
                        ),
                        mx=mx,
                    )
                expected = str(row.get("target") or "")
                arm_id = str(row.get("arm_id") or "")
                diagnostics = behavior_diagnostics(
                    generated=generated,
                    expected=expected,
                    prompt=str(row.get("prompt") or ""),
                )
                rows.append(
                    {
                        "row_id": row.get("row_id"),
                        "arm_id": arm_id,
                        "prompt_sha256": row.get("prompt_sha256"),
                        "expected_sha256": row.get("target_sha256"),
                        "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                        "exact_match": (
                            generated == expected
                            and generation.get("state") == "GREEN"
                            and generation.get("stop_reason") == "eos"
                        ),
                        "nonempty": bool(generated),
                        "behavior_diagnostics": diagnostics,
                        "syntax": syntax_diagnostic(generated, arm_id),
                        "generation": generation,
                    }
                )
                completed_row_ids.add(row_id)
                write_json_atomic(
                    output,
                    {
                        "policy": progress_policy,
                        "created_utc": now(),
                        "complete": False,
                        **progress_identity,
                        "row_count": len(rows),
                        "completed_row_ids": sorted(completed_row_ids),
                        "rows": rows,
                        "generator_visible_fields": ["prompt"],
                        "evaluator_only_fields": [
                            "target",
                            "target_sha256",
                            "source_identity",
                        ],
                        "target_visible_to_generator": False,
                        "public_training_rows_written": 0,
                        "external_inference_calls": 0,
                        "fallback_return_count": 0,
                        "capability_claim": "NONE_INCOMPLETE_EVALUATION_PROGRESS",
                    },
                )
    by_arm: dict[str, Any] = {}
    for arm_id in ARM_IDS:
        arm_rows = [row for row in rows if row["arm_id"] == arm_id]
        if arm_rows:
            by_arm[arm_id] = evaluation_summary(arm_rows)
    report = {
        "policy": config["evaluation"]["policy"],
        "created_utc": now(),
        "trigger_state": "GREEN",
        "target_id": target["target_id"],
        "split": split,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "evaluation_artifacts": evaluation_artifacts,
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "maximum_rows_per_artifact": int(maximum_rows),
        "complete": True,
        "resumed_row_count": len(resumed_rows),
        "row_count": len(rows),
        "bounded_candidate_evaluation": {
            "active": maximum_rows > 0,
            "maximum_rows_per_artifact": int(maximum_rows),
            "selection_namespace": (
                selection_namespace
                if selected_row_ids
                else "t0a_private_rdc_kerc_source_disjoint_v1"
                if maximum_rows > 0
                else ""
            ),
            "selection_contract_sha256": (
                selection_contract_sha256 if selected_row_ids else ""
            ),
            "explicit_frozen_row_selection": bool(selected_row_ids),
            "selection_uses_model_outcomes": False,
            "selection_uses_answer_text": False,
        },
        "evaluation_execution_policy": evaluation_execution_policy,
        "summary": evaluation_summary(rows),
        "by_arm": by_arm,
        "rows": rows,
        "generator_visible_fields": ["prompt"],
        "evaluator_only_fields": ["target", "target_sha256", "source_identity"],
        "target_visible_to_generator": False,
        "candidate_family": (
            "learned_kerc_compiler_core_renderer_roundtrip"
            if target.get("role") == "kerc_english_candidate"
            else "direct_autoregressive_model_text"
        ),
        "templates_renderers_routers_tools_credit": 0,
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "capability_claim": (
            "PRIVATE_DEVELOPMENT_DIAGNOSTIC"
            if split == "private_dev"
            else "PRIVATE_FROZEN_CONFIRMATION_ONLY"
        ),
    }
    write_json_atomic(output, report)
    return {**report, "rows": {"path": relative(output), "embedded_row_count": len(rows)}}


def validate_required_compiler_transport_version(
    parsed: dict[str, Any],
    *,
    required_version: int,
) -> None:
    """Reject a valid legacy wire result when the bound target requires v3."""

    observed_version = int(parsed.get("compiler_transport_version") or 0)
    if observed_version != int(required_version):
        raise KernelProtocolFault(
            "KERC_COMPILER_TRANSPORT_VERSION_MISMATCH",
            f"required={int(required_version)} observed={observed_version}",
            path="compiler_output",
        )


def generate_kerc_pipeline_text(
    model: Any,
    prompt: str,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    base: dict[str, Any],
    *,
    target: dict[str, Any],
    max_tokens: int,
    max_source_tokens: int,
    beam_width: int,
    branching_factor: int,
    length_penalty: float,
    interaction_id: str,
    mx: Any,
) -> tuple[str, dict[str, Any]]:
    """Run the actual learned KERC chain; reject any broken intermediate."""

    code_vocabulary = (
        ((target.get("kernel_code_vocabulary") or {}).get("payload") or {})
    )
    model_contract = target.get("model") or {}
    compiler_transport = (
        target.get("kerc_compiler_transport")
        if isinstance(target.get("kerc_compiler_transport"), dict)
        else {}
    )
    required_compiler_transport_version = int(
        compiler_transport.get("version")
        or LEARNED_COMPILER_COMPACT_TRANSPORT_VERSION
    )
    if code_vocabulary.get("policy") != "project_theseus_kerc_dual_code_vocabulary_v1":
        return "", generation_fault("kerc_code_vocabulary_missing")
    if required_compiler_transport_version not in {
        LEARNED_COMPILER_COMPACT_TRANSPORT_VERSION,
        LEARNED_COMPILER_SEMANTIC_POINTER_TRANSPORT_VERSION,
        LEARNED_COMPILER_SOURCE_SPAN_TRANSPORT_VERSION,
    }:
        return "", generation_fault("kerc_compiler_transport_version_invalid")
    hrl_state = vcm_semantic_memory.create_hierarchical_residual_state(
        interaction_id,
        scope={
            "user": "local-evaluation",
            "project": "theseus",
            "conversation": interaction_id,
            "privacy": "private_local",
        },
    )

    def execute_stage(objective: str, stage_prompt: str) -> tuple[str, dict[str, Any]]:
        if objective in KERC_KERNEL_OBJECTIVES:
            def validate_completion(candidate_text: str) -> None:
                if objective == "surface_to_kernel_program_v1":
                    compiler_prompt = json.loads(stage_prompt)
                    surface = str(compiler_prompt.get("source_surface") or "")
                    parsed = parse_learned_compiler_output(
                        candidate_text,
                        protected_objects={},
                        concept_capsules={},
                        source_character_length=len(surface),
                        source=surface,
                        hrl_state=hrl_state,
                        concept_resolver=(
                            concept_registry.resolve
                            if concept_registry is not None
                            else None
                        ),
                    )
                    validate_required_compiler_transport_version(
                        parsed,
                        required_version=required_compiler_transport_version,
                    )
                else:
                    parse_learned_answer_output(candidate_text)

            return generate_kerc_code_text(
                model,
                stage_prompt,
                source_vocab,
                target_vocab,
                base,
                code_vocabulary=code_vocabulary,
                kernel_offset=int(model_contract["kerc_kernel_token_start"]),
                pointer_offset=int(model_contract["kerc_pointer_token_start"]),
                pointer_end=int(model_contract["kerc_pointer_token_end"]),
                max_tokens=max_tokens,
                max_source_tokens=max_source_tokens,
                beam_width=beam_width,
                branching_factor=branching_factor,
                length_penalty=length_penalty,
                trusted_source_prefix_token=TRAINING_TASK_TAGS[objective],
                structured_source=(
                    objective in KERC_STRUCTURED_SOURCE_OBJECTIVES
                ),
                completion_validator=validate_completion,
                completion_prefix_validator=(
                    lambda rows: kerc_protocol_constant_prefix_valid(
                        rows, objective=objective
                    )
                ),
                mx=mx,
            )
        if objective == "answer_packet_to_surface_v1":
            return generate_model_text(
                model,
                stage_prompt,
                source_vocab,
                target_vocab,
                base,
                max_tokens=max_tokens,
                max_source_tokens=max_source_tokens,
                beam_width=beam_width,
                branching_factor=branching_factor,
                length_penalty=length_penalty,
                trusted_source_prefix_tokens=(TRAINING_TASK_TAGS[objective],),
                structured_source_code_vocabulary=code_vocabulary,
                structured_source_kernel_offset=int(
                    model_contract["kerc_kernel_token_start"]
                ),
                structured_source_pointer_offset=int(
                    model_contract["kerc_pointer_token_start"]
                ),
                mx=mx,
            )
        return "", generation_fault("kerc_objective_not_routeable")

    concept_registry: ConceptRegistry | None = None
    concept_registry_fault = ""
    try:
        try:
            concept_registry = ConceptRegistry()
        except (OSError, sqlite3.Error, ValueError) as exc:
            concept_registry_fault = str(exc)
        text, receipt = execute_learned_pipeline(
            prompt,
            hrl_state=hrl_state,
            stage_executor=execute_stage,
            concept_resolver=(
                concept_registry.resolve if concept_registry is not None else None
            ),
        )
    except KernelProtocolFault as exc:
        return "", {
            **generation_fault(exc.code),
            "policy": "project_theseus_kerc_learned_pipeline_execution_v1",
            "fault": exc.record(),
            "direct_surface_route_used": False,
            "concept_registry_available": concept_registry is not None,
            "concept_registry_fault": concept_registry_fault,
        }
    finally:
        if concept_registry is not None:
            concept_registry.close()
    return text, {
        **receipt,
        "decoder": "learned_kerc_five_stage_roundtrip_v1",
        "target_visible_to_generator": False,
        "byte_serialization_valid": True,
        "stop_reason": "validated_roundtrip",
        "concept_registry_available": concept_registry is not None,
        "concept_registry_fault": concept_registry_fault,
    }


def select_semantically_valid_completion(
    decoded_complete: list[dict[str, Any]],
    completion_validator: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the highest-ranked independently valid learned proposal."""

    if not decoded_complete:
        raise ValueError("semantic completion selection requires candidates")
    selected = decoded_complete[0]
    receipt: dict[str, Any] = {
        "validator_active": completion_validator is not None,
        "complete_candidate_count": len(decoded_complete),
        "rejected_candidate_count": 0,
        "rejection_counts": {},
        "selected_rank": int(selected["rank"]),
        "selected_semantically_valid": None,
        "output_repaired_or_rewritten": False,
    }
    if completion_validator is None:
        return selected, receipt
    rejection_counts: Counter[str] = Counter()
    rejection_detail_classes: Counter[str] = Counter()
    valid_candidate: dict[str, Any] | None = None
    for candidate in decoded_complete:
        try:
            completion_validator(str(candidate["decoded_text"]))
        except KernelProtocolFault as exc:
            rejection_counts[exc.code] += 1
            detail_class = (
                "missing"
                if exc.detail in {"None", "", "NoneType"}
                else "present_invalid"
            )
            rejection_detail_classes[f"{exc.code}:{detail_class}"] += 1
            continue
        valid_candidate = candidate
        break
    receipt["rejected_candidate_count"] = int(sum(rejection_counts.values()))
    receipt["rejection_counts"] = dict(sorted(rejection_counts.items()))
    receipt["rejection_detail_classes"] = dict(
        sorted(rejection_detail_classes.items())
    )
    receipt["selected_semantically_valid"] = valid_candidate is not None
    if valid_candidate is not None:
        selected = valid_candidate
    receipt["selected_rank"] = int(selected["rank"])
    return selected, receipt


def kerc_decoded_prefix(rows: list[dict[str, str]]) -> str | None:
    """Decode completed transport spans into the JSON prefix seen by validators."""

    pieces: list[str] = []
    active_space = ""
    byte_tokens: list[str] = []
    for row in rows:
        token = row["token"]
        if token == TARGET_BYTE_BEGIN:
            if active_space:
                return None
            active_space = row["space"]
            byte_tokens = [token]
        elif token == TARGET_BYTE_END:
            if row["space"] != active_space:
                return None
            decoded, receipt = decode_target_tokens([*byte_tokens, token])
            if receipt.get("state") != "READY":
                return None
            pieces.extend(str(value) for value in decoded)
            active_space = ""
            byte_tokens = []
        elif active_space:
            if row["space"] != active_space or not is_byte_token(token):
                return None
            byte_tokens.append(token)
        else:
            pieces.append(token)
    return "".join(pieces)


def kerc_protocol_constant_prefix_valid(
    rows: list[dict[str, str]], *, objective: str
) -> bool:
    """Constrain only immutable KERC ABI constants, never semantic values."""

    prefix = kerc_decoded_prefix(rows)
    if prefix is None:
        return False
    if objective == "surface_to_kernel_program_v1" and prefix.lstrip().startswith("["):
        compact_prefix = prefix.lstrip()
        return compact_prefix.startswith(
            f"[{LEARNED_COMPILER_COMPACT_TRANSPORT_VERSION},"
        )
    versions = re.findall(r'"kernel_version"\s*:\s*"([^"\\]*)"', prefix)
    if any(value != KERNEL_VERSION for value in versions):
        return False
    policies = re.findall(r'"policy"\s*:\s*"([^"\\]*)"', prefix)
    allowed_policies = (
        {KERC_HIERARCHICAL_COMPILER_POLICY, LEARNED_PROGRAM_TRANSPORT_POLICY}
        if objective == "surface_to_kernel_program_v1"
        else {LEARNED_ANSWER_TRANSPORT_POLICY}
    )
    return all(value in allowed_policies for value in policies)


def generate_kerc_code_text(
    model: Any,
    prompt: str,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    base: dict[str, Any],
    *,
    code_vocabulary: dict[str, Any],
    kernel_offset: int,
    pointer_offset: int,
    pointer_end: int,
    max_tokens: int,
    max_source_tokens: int,
    beam_width: int,
    branching_factor: int,
    length_penalty: float,
    trusted_source_prefix_token: str,
    structured_source: bool,
    batched_beam_advance: bool = True,
    device_logit_filter: bool = True,
    preprune_beam_expansions: bool = True,
    completion_validator: Callable[[str], None] | None = None,
    completion_prefix_validator: Callable[[list[dict[str, str]]], bool] | None = None,
    online_completion_validation: bool = False,
    mx: Any,
) -> tuple[str, dict[str, Any]]:
    """Decode one grammar-serialized KERC code object in disjoint V_K/V_P."""

    acceleration = generation_acceleration_receipt(
        batched_beam_advance=batched_beam_advance,
        device_logit_filter=device_logit_filter,
        preprune_beam_expansions=preprune_beam_expansions,
    )

    if structured_source:
        source_ids, source_receipt = encode_kerc_global_target(
            prompt,
            code_vocabulary=code_vocabulary,
            kernel_offset=kernel_offset,
            pointer_offset=pointer_offset,
        )
    else:
        source_ids, source_receipt = encode_tokens(
            kerc_surface_tokens(prompt), source_vocab, stream="source"
        )
    if int(source_receipt.get("unknown_token_count") or 0):
        return "", {**generation_fault("source_unrepresentable"), **acceleration}
    if trusted_source_prefix_token not in source_vocab:
        return "", {
            **generation_fault("trusted_source_prefix_unrepresentable"),
            **acceleration,
        }
    source_offset = source_token_offset(base, source_vocab)
    source_ids = [
        source_offset + int(source_vocab[trusted_source_prefix_token]),
        *(
            source_ids
            if structured_source
            else [source_offset + int(value) for value in source_ids]
        ),
    ]
    if len(source_ids) > max_source_tokens:
        return "", {**generation_fault("source_requires_truncation"), **acceleration}
    target_offset = target_token_offset(base, source_vocab)
    end_id = target_offset + int(target_vocab["<eos>"])
    prompt_ids = [GLOBAL_BOS_ID]
    prompt_ids.extend(int(value) for value in source_ids)
    prompt_ids.append(SOURCE_TARGET_SEPARATOR_ID)
    prompt_ids.append(target_offset + int(target_vocab["<bos>"]))
    logits, cache = model(mx.array([prompt_ids], dtype=mx.int32))
    mx.eval(logits, *cache_arrays(cache))
    token_rows = kerc_global_token_rows(
        code_vocabulary,
        kernel_offset=kernel_offset,
        pointer_offset=pointer_offset,
        pointer_end=pointer_end,
    )
    if online_completion_validation and completion_validator is None:
        raise ValueError(
            "online completion validation requires an independent validator"
        )
    beams = [
        {
            "ids": [],
            "tokens": [],
            "score": 0.0,
            "logits": logits[0, -1],
            "cache": cache,
        }
    ]
    complete: list[dict[str, Any]] = []
    online_rejection_counts: Counter[str] = Counter()
    online_rejection_detail_classes: Counter[str] = Counter()
    online_complete_candidate_count = 0
    for _ in range(max_tokens):
        expansions: list[dict[str, Any]] = []
        for beam in beams:
            allowed = kerc_serialization_valid_ids(
                beam["tokens"], token_rows, end_id=end_id
            )
            if completion_prefix_validator is not None:
                filtered: list[int] = []
                for token_id in allowed:
                    if token_id == end_id:
                        filtered.append(token_id)
                        continue
                    row = token_rows[token_id]
                    token = row["token"]
                    if (
                        token != TARGET_BYTE_END
                        and '"' not in token
                    ) or completion_prefix_validator(
                        [*beam["tokens"], row]
                    ):
                        filtered.append(token_id)
                allowed = filtered
            if not allowed:
                continue
            ranked = rank_global_allowed_logits(
                beam["logits"],
                allowed,
                branching_factor=branching_factor,
                device_filter=device_logit_filter,
                mx=mx,
            )
            for token_id, log_probability in ranked:
                score = float(beam["score"]) + log_probability
                if token_id == end_id:
                    candidate = {
                        "ids": list(beam["ids"]),
                        "tokens": list(beam["tokens"]),
                        "score": score,
                    }
                    if online_completion_validation:
                        online_complete_candidate_count += 1
                        candidate_text, candidate_receipt = (
                            decode_kerc_global_target(
                                list(candidate["ids"]),
                                code_vocabulary=code_vocabulary,
                                kernel_offset=kernel_offset,
                                pointer_offset=pointer_offset,
                            )
                        )
                        if candidate_receipt.get("state") != "READY":
                            online_rejection_counts[
                                "KERC_CODE_DECODE_FAULT"
                            ] += 1
                            online_rejection_detail_classes[
                                "KERC_CODE_DECODE_FAULT:present_invalid"
                            ] += 1
                            continue
                        try:
                            completion_validator(str(candidate_text))
                        except KernelProtocolFault as exc:
                            online_rejection_counts[exc.code] += 1
                            detail_class = (
                                "missing"
                                if exc.detail in {"None", "", "NoneType"}
                                else "present_invalid"
                            )
                            online_rejection_detail_classes[
                                f"{exc.code}:{detail_class}"
                            ] += 1
                            continue
                        candidate["decoded_text"] = candidate_text
                        candidate["decode_receipt"] = candidate_receipt
                    complete.append(candidate)
                    continue
                row = token_rows[token_id]
                expansions.append(
                    {
                        "beam": beam,
                        "global_id": token_id,
                        "token": row,
                        "log_probability": log_probability,
                    }
                )
        if preprune_beam_expansions:
            expansions = prune_text_expansion_specs(
                expansions,
                limit=beam_width,
                length_penalty=length_penalty,
            )
        expansions = (
            advance_beams_batched(model, expansions, target_offset=0, mx=mx)
            if batched_beam_advance
            else advance_beams_serial(model, expansions, target_offset=0, mx=mx)
        )
        beams = sorted(
            expansions, key=lambda row: beam_score(row, length_penalty), reverse=True
        )[: max(1, beam_width)]
        complete = sorted(
            complete, key=lambda row: beam_score(row, length_penalty), reverse=True
        )[: max(1, beam_width)]
        if not beams or (
            complete
            and len(complete) >= beam_width
            and beam_score(complete[0], length_penalty)
            >= beam_score(beams[0], length_penalty)
        ):
            break
    semantic_selection: dict[str, Any] = {
        "validator_active": completion_validator is not None,
        "complete_candidate_count": len(complete),
        "rejected_candidate_count": 0,
        "rejection_counts": {},
        "selected_rank": None,
        "selected_semantically_valid": None,
        "output_repaired_or_rewritten": False,
        "online_completion_validation": bool(
            online_completion_validation
        ),
        "rejected_candidates_terminated_search": False,
        "assisted_output_credit_required": bool(
            online_completion_validation
        ),
    }
    if online_completion_validation:
        semantic_selection["complete_candidate_count"] = int(
            online_complete_candidate_count
        )
        semantic_selection["rejected_candidate_count"] = int(
            sum(online_rejection_counts.values())
        )
        semantic_selection["rejection_counts"] = dict(
            sorted(online_rejection_counts.items())
        )
        semantic_selection["rejection_detail_classes"] = dict(
            sorted(online_rejection_detail_classes.items())
        )
    decoded = ""
    decode_receipt: dict[str, Any] = {}
    if complete:
        ranked_complete = sorted(
            complete,
            key=lambda row: beam_score(row, length_penalty),
            reverse=True,
        )
        decoded_complete: list[dict[str, Any]] = []
        for rank, candidate in enumerate(ranked_complete, start=1):
            candidate_text = candidate.get("decoded_text")
            candidate_receipt = candidate.get("decode_receipt")
            if candidate_text is None or candidate_receipt is None:
                candidate_text, candidate_receipt = decode_kerc_global_target(
                    list(candidate["ids"]),
                    code_vocabulary=code_vocabulary,
                    kernel_offset=kernel_offset,
                    pointer_offset=pointer_offset,
                )
            if candidate_receipt.get("state") != "READY":
                continue
            decoded_complete.append(
                {
                    **candidate,
                    "decoded_text": candidate_text,
                    "decode_receipt": candidate_receipt,
                    "rank": rank,
                }
            )
        if not decoded_complete:
            return "", {
                **generation_fault("kerc_code_decode_fault"),
                **acceleration,
                "semantic_selection": semantic_selection,
            }
        selected, semantic_selection = select_semantically_valid_completion(
            decoded_complete, completion_validator
        )
        if online_completion_validation:
            semantic_selection["complete_candidate_count"] = int(
                online_complete_candidate_count
            )
            semantic_selection["rejected_candidate_count"] = (
                int(semantic_selection["rejected_candidate_count"])
                + int(sum(online_rejection_counts.values()))
            )
            combined_rejections = Counter(
                semantic_selection.get("rejection_counts") or {}
            )
            combined_rejections.update(online_rejection_counts)
            semantic_selection["rejection_counts"] = dict(
                sorted(combined_rejections.items())
            )
            combined_detail = Counter(
                semantic_selection.get("rejection_detail_classes") or {}
            )
            combined_detail.update(online_rejection_detail_classes)
            semantic_selection["rejection_detail_classes"] = dict(
                sorted(combined_detail.items())
            )
            semantic_selection["online_completion_validation"] = True
            semantic_selection["rejected_candidates_terminated_search"] = (
                False
            )
            semantic_selection["assisted_output_credit_required"] = True
        if (
            completion_validator is not None
            and semantic_selection["selected_semantically_valid"] is not True
        ):
            return "", {
                **generation_fault("no_semantically_valid_completion"),
                **acceleration,
                "semantic_selection": semantic_selection,
            }
        decoded = str(selected["decoded_text"])
        decode_receipt = dict(selected["decode_receipt"])
        stop_reason = "eos"
    elif beams:
        selected = max(beams, key=lambda row: beam_score(row, length_penalty))
        stop_reason = "max_tokens"
        decoded, decode_receipt = decode_kerc_global_target(
            list(selected["ids"]),
            code_vocabulary=code_vocabulary,
            kernel_offset=kernel_offset,
            pointer_offset=pointer_offset,
        )
    elif online_completion_validation and online_rejection_counts:
        semantic_selection["complete_candidate_count"] = int(
            online_complete_candidate_count
        )
        semantic_selection["rejected_candidate_count"] = int(
            sum(online_rejection_counts.values())
        )
        semantic_selection["rejection_counts"] = dict(
            sorted(online_rejection_counts.items())
        )
        semantic_selection["rejection_detail_classes"] = dict(
            sorted(online_rejection_detail_classes.items())
        )
        return "", {
            **generation_fault("no_semantically_valid_completion"),
            **acceleration,
            "semantic_selection": semantic_selection,
        }
    else:
        return "", {
            **generation_fault("no_serialization_valid_sequence"),
            **acceleration,
        }
    if decode_receipt.get("state") != "READY":
        return "", {
            **generation_fault("kerc_code_decode_fault"),
            **acceleration,
            "decode_receipt": decode_receipt,
        }
    generated_token_sha256 = hashlib.sha256(
        json.dumps(selected["ids"], separators=(",", ":")).encode()
    ).hexdigest()
    if stop_reason == "max_tokens":
        return "", {
            **generation_fault("kerc_code_incomplete_at_budget"),
            **acceleration,
            "semantic_selection": semantic_selection,
            "stop_reason": stop_reason,
            "generated_token_count": len(selected["ids"]),
            "generated_token_sha256": generated_token_sha256,
            "byte_serialization_valid": True,
            "json_prefix_complete": False,
            "target_visible_to_generator": False,
            "trusted_source_prefix_tokens": [trusted_source_prefix_token],
            "fallback_return_count": 0,
        }
    return decoded, {
        "state": "GREEN",
        "decoder": "beam_kerc_dual_code_serialization_v1",
        **acceleration,
        "semantic_selection": semantic_selection,
        "beam_width": int(beam_width),
        "branching_factor": int(branching_factor),
        "stop_reason": stop_reason,
        "generated_token_count": len(selected["ids"]),
        "generated_token_sha256": generated_token_sha256,
        "byte_serialization_valid": True,
        "json_prefix_complete": True,
        "target_visible_to_generator": False,
        "trusted_source_prefix_tokens": [trusted_source_prefix_token],
        "fallback_return_count": 0,
    }


def rank_global_allowed_logits(
    logits: Any,
    allowed_ids: list[int],
    *,
    branching_factor: int,
    device_filter: bool,
    mx: Any,
) -> list[tuple[int, float]]:
    """Rank sparse global token ids without copying the full vocabulary to host."""

    if not allowed_ids:
        return []
    limit = min(len(allowed_ids), max(1, int(branching_factor)))
    if not device_filter:
        values = np.asarray(logits).astype(np.float64)
        allowed_values = np.asarray(
            [values[token_id] for token_id in allowed_ids], dtype=np.float64
        )
        maximum = float(allowed_values.max())
        normalizer = maximum + float(np.log(np.exp(allowed_values - maximum).sum()))
        ranked = sorted(
            allowed_ids,
            key=lambda token_id: float(values[token_id]),
            reverse=True,
        )[:limit]
        return [
            (token_id, float(values[token_id]) - normalizer) for token_id in ranked
        ]
    device_ids = mx.array(allowed_ids, dtype=mx.int32)
    allowed_logits = mx.take(logits, device_ids, axis=0)
    log_normalizer = mx.logsumexp(allowed_logits, axis=0)
    selected_positions = mx.argsort(allowed_logits, axis=0)[-limit:]
    selected_logits = mx.take(allowed_logits, selected_positions, axis=0)
    mx.eval(selected_positions, selected_logits, log_normalizer)
    positions = np.asarray(selected_positions, dtype=np.int64)[::-1]
    values = np.asarray(selected_logits, dtype=np.float64)[::-1]
    normalizer = float(log_normalizer.item())
    return [
        (int(allowed_ids[int(position)]), float(value) - normalizer)
        for position, value in zip(positions, values)
    ]


def kerc_global_token_rows(
    code_vocabulary: dict[str, Any],
    *,
    kernel_offset: int,
    pointer_offset: int,
    pointer_end: int,
) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for space, key, offset, end in (
        ("V_K", "kernel_vocab", kernel_offset, pointer_offset),
        ("V_P", "pointer_vocab", pointer_offset, pointer_end),
    ):
        for token, local_id in (code_vocabulary.get(key) or {}).items():
            global_id = offset + int(local_id)
            if not offset <= global_id < end or global_id in rows:
                raise ValueError("KERC code vocabulary exceeds its assigned range")
            rows[global_id] = {"space": space, "token": str(token)}
    return rows


def kerc_ground_truth_serialization_valid(
    token_ids: list[int], token_rows: dict[int, dict[str, str]]
) -> bool:
    """Audit that one supervised KERC target is reachable by the decoder."""

    state: KercJsonStreamState | None = KercJsonStreamState()
    active_space = ""
    byte_tokens: list[str] = []
    byte_payload_length = 0
    for token_id in token_ids:
        row = token_rows.get(int(token_id))
        if row is None:
            return False
        token = row["token"]
        if token == TARGET_BYTE_BEGIN:
            if active_space or not kerc_json_state_accepts_atom(state):
                return False
            active_space = row["space"]
            byte_tokens = [token]
            byte_payload_length = 0
        elif token == TARGET_BYTE_END:
            if row["space"] != active_space:
                return False
            decoded, receipt = decode_target_tokens([*byte_tokens, token])
            if receipt.get("state") != "READY":
                return False
            state = kerc_json_token_transition(state, "".join(decoded))
            if state is None:
                return False
            active_space = ""
            byte_tokens = []
            byte_payload_length = 0
        elif active_space:
            if row["space"] != active_space or not is_byte_token(token):
                return False
            byte_payload_length += len(byte_token_bytes(token))
            if byte_payload_length > MAX_TOKEN_BYTES:
                return False
            byte_tokens.append(token)
        else:
            state = kerc_json_token_transition(state, token)
            if state is None:
                return False
    return not active_space and kerc_json_state_complete(state)


def kerc_serialization_valid_ids(
    generated: list[dict[str, str]],
    token_rows: dict[int, dict[str, str]],
    *,
    end_id: int,
) -> list[int]:
    """Return byte-safe IDs that also preserve canonical JSON syntax.

    KERC code targets are JSON.  The earlier decoder constrained only byte-span
    transport, so malformed punctuation remained eligible and dominated learned
    beams.  This independent pushdown state rejects invalid JSON transitions
    without supplying any semantic field value or hidden answer information.
    """

    active_space = ""
    byte_tokens: list[str] = []
    byte_payload_length = 0
    json_state: KercJsonStreamState | None = KercJsonStreamState()
    for row in generated:
        token = row["token"]
        if token == TARGET_BYTE_BEGIN:
            if active_space:
                return []
            active_space = row["space"]
            byte_tokens = [token]
            byte_payload_length = 0
        elif token == TARGET_BYTE_END:
            if row["space"] != active_space:
                return []
            byte_tokens.append(token)
            decoded, receipt = decode_target_tokens(byte_tokens)
            if receipt.get("state") != "READY":
                return []
            json_state = kerc_json_token_transition(
                json_state, "".join(decoded)
            )
            if json_state is None:
                return []
            active_space = ""
            byte_tokens = []
            byte_payload_length = 0
        elif active_space and (
            row["space"] != active_space or not is_byte_token(token)
        ):
            return []
        elif active_space:
            byte_payload_length += len(byte_token_bytes(token))
            if byte_payload_length > MAX_TOKEN_BYTES:
                return []
            byte_tokens.append(token)
        else:
            json_state = kerc_json_token_transition(json_state, token)
            if json_state is None:
                return []
    allowed: list[int] = []
    for token_id, row in token_rows.items():
        token = row["token"]
        if active_space:
            if row["space"] != active_space:
                continue
            if is_byte_token(token) and byte_payload_length + len(
                byte_token_bytes(token)
            ) <= MAX_TOKEN_BYTES:
                allowed.append(token_id)
            elif token == TARGET_BYTE_END:
                decoded, receipt = decode_target_tokens(
                    [*byte_tokens, TARGET_BYTE_END]
                )
                if receipt.get("state") == "READY" and kerc_json_token_transition(
                    json_state, "".join(decoded)
                ) is not None:
                    allowed.append(token_id)
        elif token == TARGET_BYTE_BEGIN or (
            token not in {"<pad>", "<unk>", "<bos>", "<eos>", TARGET_BYTE_END}
            and not is_byte_token(token)
        ):
            if token == TARGET_BYTE_BEGIN:
                if kerc_json_state_accepts_atom(json_state):
                    allowed.append(token_id)
            elif kerc_json_token_transition(json_state, token) is not None:
                allowed.append(token_id)
    if not active_space and kerc_json_state_complete(json_state):
        allowed.append(end_id)
    return allowed


def kerc_json_state_accepts_atom(
    state: KercJsonStreamState | None,
) -> bool:
    # A byte span is a transport fragment, not necessarily one complete JSON
    # atom.  It may therefore begin at any unfinished valid prefix (including
    # between two fragments of one long string, number, or literal).
    return state is not None and not kerc_json_state_complete(state)


def kerc_json_state_complete(
    state: KercJsonStreamState | None,
) -> bool:
    if state is None or state.root_phase != "done" or state.stack:
        return False
    if state.mode == "structural":
        return True
    return state.mode == "number" and state.number_state in {
        "zero",
        "integer",
        "fraction",
        "exponent_digits",
    }


@dataclass(frozen=True)
class KercJsonStreamState:
    """Incremental JSON-prefix state across arbitrary KERC token fragments."""

    root_phase: str = "value"
    stack: tuple[tuple[str, str], ...] = ()
    mode: str = "structural"
    string_role: str = ""
    unicode_remaining: int = 0
    literal_remaining: str = ""
    number_state: str = ""


def kerc_json_token_transition(
    state: KercJsonStreamState | None,
    token: str,
) -> KercJsonStreamState | None:
    """Advance an arbitrary JSON text fragment through a deterministic PDA.

    KERC's lossless tokenizer may split one long JSON string across adjacent
    byte spans.  Treating every closed span as a complete JSON atom made valid
    supervised sequences impossible to decode.  This state machine retains
    lexical state across span boundaries while still constraining syntax only;
    it supplies no field names, values, or answer-derived metadata.
    """

    if state is None:
        return None
    root_phase = state.root_phase
    stack = [list(frame) for frame in state.stack]
    mode = state.mode
    string_role = state.string_role
    unicode_remaining = state.unicode_remaining
    literal_remaining = state.literal_remaining
    number_state = state.number_state

    def snapshot() -> KercJsonStreamState:
        return KercJsonStreamState(
            root_phase=root_phase,
            stack=tuple((str(kind), str(phase)) for kind, phase in stack),
            mode=mode,
            string_role=string_role,
            unicode_remaining=unicode_remaining,
            literal_remaining=literal_remaining,
            number_state=number_state,
        )

    def begin_value() -> bool:
        nonlocal root_phase
        if stack:
            kind, phase = stack[-1]
            if (kind == "object" and phase != "value") or (
                kind == "array" and phase not in {"value", "value_or_end"}
            ):
                return False
            stack[-1][1] = "comma_or_end"
            return True
        if root_phase != "value":
            return False
        root_phase = "done"
        return True

    def number_accepting() -> bool:
        return number_state in {
            "zero",
            "integer",
            "fraction",
            "exponent_digits",
        }

    index = 0
    while index < len(token):
        character = token[index]
        if mode == "string":
            if character == '"':
                mode = "structural"
                if string_role == "key":
                    if not stack or stack[-1][0] != "object" or stack[-1][1] not in {
                        "key",
                        "key_or_end",
                    }:
                        return None
                    stack[-1][1] = "colon"
                string_role = ""
            elif character == "\\":
                mode = "string_escape"
            elif ord(character) < 0x20:
                return None
            index += 1
            continue
        if mode == "string_escape":
            if character == "u":
                mode = "string_unicode"
                unicode_remaining = 4
            elif character in '"\\/bfnrt':
                mode = "string"
            else:
                return None
            index += 1
            continue
        if mode == "string_unicode":
            if character not in "0123456789abcdefABCDEF":
                return None
            unicode_remaining -= 1
            if unicode_remaining == 0:
                mode = "string"
            index += 1
            continue
        if mode == "literal":
            if not literal_remaining or character != literal_remaining[0]:
                return None
            literal_remaining = literal_remaining[1:]
            if not literal_remaining:
                mode = "structural"
            index += 1
            continue
        if mode == "number":
            if number_state == "minus":
                if character == "0":
                    number_state = "zero"
                elif character in "123456789":
                    number_state = "integer"
                else:
                    return None
                index += 1
                continue
            if number_state == "zero":
                if character == ".":
                    number_state = "decimal_point"
                    index += 1
                    continue
                if character in "eE":
                    number_state = "exponent"
                    index += 1
                    continue
            elif number_state == "integer":
                if character.isdigit():
                    index += 1
                    continue
                if character == ".":
                    number_state = "decimal_point"
                    index += 1
                    continue
                if character in "eE":
                    number_state = "exponent"
                    index += 1
                    continue
            elif number_state == "decimal_point":
                if character.isdigit():
                    number_state = "fraction"
                    index += 1
                    continue
                return None
            elif number_state == "fraction":
                if character.isdigit():
                    index += 1
                    continue
                if character in "eE":
                    number_state = "exponent"
                    index += 1
                    continue
            elif number_state == "exponent":
                if character in "+-":
                    number_state = "exponent_sign"
                    index += 1
                    continue
                if character.isdigit():
                    number_state = "exponent_digits"
                    index += 1
                    continue
                return None
            elif number_state == "exponent_sign":
                if character.isdigit():
                    number_state = "exponent_digits"
                    index += 1
                    continue
                return None
            elif number_state == "exponent_digits":
                if character.isdigit():
                    index += 1
                    continue
            if not number_accepting():
                return None
            mode = "structural"
            number_state = ""
            # Reprocess the delimiter under the structural state.
            continue

        if character in " \t\r\n":
            index += 1
            continue
        if not stack:
            if root_phase == "done":
                return None
            phase = "value"
            kind = "root"
        else:
            kind, phase = stack[-1]
        if kind == "object" and phase in {"key", "key_or_end"}:
            if character == '"':
                mode = "string"
                string_role = "key"
                index += 1
                continue
            if phase == "key_or_end" and character == "}":
                stack.pop()
                index += 1
                continue
            return None
        if kind == "object" and phase == "colon":
            if character != ":":
                return None
            stack[-1][1] = "value"
            index += 1
            continue
        if phase == "comma_or_end":
            if character == ",":
                stack[-1][1] = "key" if kind == "object" else "value"
                index += 1
                continue
            if (kind == "object" and character == "}") or (
                kind == "array" and character == "]"
            ):
                stack.pop()
                index += 1
                continue
            return None
        if kind == "array" and phase == "value_or_end" and character == "]":
            stack.pop()
            index += 1
            continue
        if phase not in {"value", "value_or_end"}:
            return None
        if character == "{":
            if not begin_value():
                return None
            stack.append(["object", "key_or_end"])
        elif character == "[":
            if not begin_value():
                return None
            stack.append(["array", "value_or_end"])
        elif character == '"':
            if not begin_value():
                return None
            mode = "string"
            string_role = "value"
        elif character in "tfn":
            if not begin_value():
                return None
            mode = "literal"
            literal_remaining = {"t": "rue", "f": "alse", "n": "ull"}[character]
        elif character == "-" or character.isdigit():
            if not begin_value():
                return None
            mode = "number"
            number_state = (
                "minus"
                if character == "-"
                else "zero"
                if character == "0"
                else "integer"
            )
        else:
            return None
        index += 1
    return snapshot()


def prepare_model_text_prompt(
    prompt: str,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    base: dict[str, Any],
    *,
    max_source_tokens: int,
    trusted_source_prefix_tokens: tuple[str, ...] = (),
    structured_source_code_vocabulary: dict[str, Any] | None = None,
    structured_source_kernel_offset: int = 0,
    structured_source_pointer_offset: int = 0,
) -> dict[str, Any]:
    """Compile visible prompt text into one target-generation prefix."""

    structured_source = bool(structured_source_code_vocabulary)
    if structured_source:
        source_ids, source_receipt = encode_kerc_global_target(
            prompt,
            code_vocabulary=structured_source_code_vocabulary or {},
            kernel_offset=structured_source_kernel_offset,
            pointer_offset=structured_source_pointer_offset,
        )
    else:
        prompt_tokens = (
            kerc_surface_tokens(prompt)
            if any(
                str(token).startswith("<KERC_TASK_")
                for token in trusted_source_prefix_tokens
            )
            else exact_text_tokens(prompt)
        )
        source_ids, source_receipt = encode_tokens(
            prompt_tokens, source_vocab, stream="source"
        )
    if int(source_receipt.get("unknown_token_count") or 0):
        return {"fault": "source_unrepresentable"}
    if any(token not in source_vocab for token in trusted_source_prefix_tokens):
        return {"fault": "trusted_source_prefix_unrepresentable"}
    if len(trusted_source_prefix_tokens) > 1:
        return {"fault": "trusted_source_prefix_ambiguous"}
    source_offset = source_token_offset(base, source_vocab)
    source_ids = [
        *(
            source_offset + int(source_vocab[token])
            for token in trusted_source_prefix_tokens
        ),
        *(
            source_ids
            if structured_source
            else [source_offset + int(value) for value in source_ids]
        ),
    ]
    if len(source_ids) > max_source_tokens:
        return {"fault": "source_requires_truncation"}
    target_offset = target_token_offset(base, source_vocab)
    prompt_ids = [GLOBAL_BOS_ID]
    prompt_ids.extend(int(value) for value in source_ids)
    prompt_ids.append(SOURCE_TARGET_SEPARATOR_ID)
    prompt_ids.append(target_offset + int(target_vocab["<bos>"]))
    prefix_key = hashlib.sha256(
        np.asarray(prompt_ids, dtype=np.int32).tobytes()
    ).hexdigest()
    return {
        "prompt_ids": prompt_ids,
        "prefix_key": prefix_key,
        "target_offset": target_offset,
    }


def generate_model_text(
    model: Any,
    prompt: str,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    base: dict[str, Any],
    *,
    max_tokens: int,
    max_source_tokens: int,
    beam_width: int,
    branching_factor: int,
    length_penalty: float,
    trusted_source_prefix_tokens: tuple[str, ...] = (),
    structured_source_code_vocabulary: dict[str, Any] | None = None,
    structured_source_kernel_offset: int = 0,
    structured_source_pointer_offset: int = 0,
    batched_beam_advance: bool = True,
    device_logit_filter: bool = True,
    preprune_beam_expansions: bool = True,
    prompt_prefix_cache: Any | None = None,
    mx: Any,
) -> tuple[str, dict[str, Any]]:
    """Generate from prompt only; the grammar constrains byte serialization, not meaning."""

    acceleration = generation_acceleration_receipt(
        batched_beam_advance=batched_beam_advance,
        device_logit_filter=device_logit_filter,
        preprune_beam_expansions=preprune_beam_expansions,
    )
    prepared = prepare_model_text_prompt(
        prompt,
        source_vocab,
        target_vocab,
        base,
        max_source_tokens=max_source_tokens,
        trusted_source_prefix_tokens=trusted_source_prefix_tokens,
        structured_source_code_vocabulary=structured_source_code_vocabulary,
        structured_source_kernel_offset=structured_source_kernel_offset,
        structured_source_pointer_offset=structured_source_pointer_offset,
    )
    if prepared.get("fault"):
        return "", {
            **generation_fault(str(prepared["fault"])),
            **acceleration,
        }
    prompt_ids = list(prepared["prompt_ids"])
    prefix_key = str(prepared["prefix_key"])
    target_offset = int(prepared["target_offset"])
    prefill_started = time.perf_counter()
    cached_prefix = (
        prompt_prefix_cache.get(prefix_key)
        if prompt_prefix_cache is not None
        else None
    )
    if cached_prefix is None:
        logits, cache = model(mx.array([prompt_ids], dtype=mx.int32))
        mx.eval(logits, *cache_arrays(cache))
        if prompt_prefix_cache is not None:
            prompt_prefix_cache.put(prefix_key, (logits, cache))
        prefix_cache_state = "MISS" if prompt_prefix_cache is not None else "DISABLED"
    else:
        logits, cache = cached_prefix
        mx.eval(logits, *cache_arrays(cache))
        prefix_cache_state = "HIT"
    acceleration.update(
        {
            "prompt_prefix_cache_state": prefix_cache_state,
            "prompt_prefix_sha256": prefix_key,
            "prompt_prefix_token_count": len(prompt_ids),
            "prompt_prefill_seconds": round(
                time.perf_counter() - prefill_started, 6
            ),
        }
    )
    inverse = {int(value): str(token) for token, value in target_vocab.items()}
    serialization_states = serialization_allowed_local_ids(inverse)
    beams = [
        {"tokens": [], "score": 0.0, "logits": logits[0, -1], "cache": cache}
    ]
    complete: list[dict[str, Any]] = []
    serialization_complete: list[dict[str, Any]] = []
    for _ in range(max_tokens):
        expansion_specs: list[dict[str, Any]] = []
        for beam in beams:
            allowed = serialization_states[
                bool(active_target_span(beam["tokens"])["active"])
            ]
            if not allowed:
                continue
            ranked = rank_allowed_logits(
                beam["logits"],
                allowed,
                id_offset=target_offset,
                branching_factor=branching_factor,
                device_filter=device_logit_filter,
                mx=mx,
            )
            for local_id, log_probability in ranked:
                token = inverse[local_id]
                score = float(beam["score"]) + log_probability
                if token == "<eos>":
                    complete.append({"tokens": list(beam["tokens"]), "score": score})
                    continue
                expansion_specs.append(
                    {
                        "beam": beam,
                        "local_id": local_id,
                        "token": token,
                        "log_probability": log_probability,
                    }
                )
        if preprune_beam_expansions:
            expansion_specs = prune_text_expansion_specs(
                expansion_specs,
                limit=beam_width,
                length_penalty=length_penalty,
            )
        expansions = (
            advance_beams_batched(
                model,
                expansion_specs,
                target_offset=target_offset,
                mx=mx,
            )
            if batched_beam_advance
            else advance_beams_serial(
                model,
                expansion_specs,
                target_offset=target_offset,
                mx=mx,
            )
        )
        beams = sorted(
            expansions,
            key=lambda row: beam_score(row, length_penalty),
            reverse=True,
        )[: max(1, beam_width)]
        serialization_complete.extend(
            {"tokens": list(row["tokens"]), "score": float(row["score"])}
            for row in beams
            if not bool(active_target_span(row["tokens"])["active"])
        )
        serialization_complete = sorted(
            serialization_complete,
            key=lambda row: beam_score(row, length_penalty),
            reverse=True,
        )[: max(1, beam_width)]
        complete = sorted(
            complete,
            key=lambda row: beam_score(row, length_penalty),
            reverse=True,
        )[: max(1, beam_width)]
        if not beams or (
            complete
            and len(complete) >= beam_width
            and beam_score(complete[0], length_penalty)
            >= beam_score(beams[0], length_penalty)
        ):
            break
    return finalize_model_text_generation(
        beams=beams,
        complete=complete,
        serialization_complete=serialization_complete,
        length_penalty=length_penalty,
        beam_width=beam_width,
        branching_factor=branching_factor,
        acceleration=acceleration,
        trusted_source_prefix_tokens=trusted_source_prefix_tokens,
    )


def finalize_model_text_generation(
    *,
    beams: list[dict[str, Any]],
    complete: list[dict[str, Any]],
    serialization_complete: list[dict[str, Any]],
    length_penalty: float,
    beam_width: int,
    branching_factor: int,
    acceleration: dict[str, Any],
    trusted_source_prefix_tokens: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    """Finalize one exact-text beam search without changing its ranking contract."""

    if complete:
        selected = max(complete, key=lambda row: beam_score(row, length_penalty))
        generated_tokens = list(selected["tokens"])
        stop_reason = "eos"
    elif serialization_complete:
        selected = max(
            serialization_complete,
            key=lambda row: beam_score(row, length_penalty),
        )
        generated_tokens = list(selected["tokens"])
        stop_reason = "max_tokens_serialization_complete"
    else:
        return "", {
            **generation_fault("no_serialization_valid_sequence"),
            **acceleration,
        }
    decoded, decode_receipt = decode_target_tokens(generated_tokens)
    if decode_receipt.get("state") != "READY":
        return "", {
            **generation_fault("byte_serialization_fault"),
            **acceleration,
            "decode_receipt": decode_receipt,
        }
    text = "".join(decoded)
    return text, {
        "state": "GREEN",
        "decoder": "beam_exact_text_with_byte_span_grammar_v1",
        **acceleration,
        "beam_width": int(beam_width),
        "branching_factor": int(branching_factor),
        "stop_reason": stop_reason,
        "generated_token_count": len(generated_tokens),
        "generated_token_sha256": hashlib.sha256(
            "\n".join(generated_tokens).encode()
        ).hexdigest(),
        "byte_serialization_valid": True,
        "target_visible_to_generator": False,
        "trusted_source_prefix_tokens": list(trusted_source_prefix_tokens),
        "fallback_return_count": 0,
    }


def generate_model_text_batch(
    model: Any,
    prompts: list[str],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    base: dict[str, Any],
    *,
    max_tokens: int,
    max_source_tokens: int,
    beam_width: int,
    branching_factor: int,
    length_penalty: float,
    trusted_source_prefix_tokens: tuple[str, ...] = (),
    batched_beam_advance: bool = True,
    device_logit_filter: bool = True,
    preprune_beam_expansions: bool = True,
    prompt_prefix_cache: Any | None = None,
    mx: Any,
) -> list[tuple[str, dict[str, Any]]]:
    """Generate independent requests with shape-safe cross-request MLX batching."""

    if not prompts:
        return []
    batch_started = time.perf_counter()
    inverse = {int(value): str(token) for token, value in target_vocab.items()}
    serialization_states = serialization_allowed_local_ids(inverse)
    shared_target_offset = target_token_offset(base, source_vocab)
    records: list[dict[str, Any]] = []
    results: list[tuple[str, dict[str, Any]] | None] = [None] * len(prompts)
    prefill_buckets: dict[int, list[dict[str, Any]]] = {}
    shared_forward_count = 0
    shared_request_indices: set[int] = set()
    model_forward_count = 0
    maximum_forward_request_count = 1
    first_decode_forward_seconds: float | None = None

    for request_index, prompt in enumerate(prompts):
        acceleration = generation_acceleration_receipt(
            batched_beam_advance=batched_beam_advance,
            device_logit_filter=device_logit_filter,
            preprune_beam_expansions=preprune_beam_expansions,
        )
        prepared = prepare_model_text_prompt(
            prompt,
            source_vocab,
            target_vocab,
            base,
            max_source_tokens=max_source_tokens,
            trusted_source_prefix_tokens=trusted_source_prefix_tokens,
        )
        if prepared.get("fault"):
            results[request_index] = (
                "",
                {
                    **generation_fault(str(prepared["fault"])),
                    **acceleration,
                    "cross_request_batching": "prompt_length_bucketed_v1",
                    "cross_request_batch_state": "REJECTED_BEFORE_PREFILL",
                    "cross_request_batch_size": len(prompts),
                },
            )
            continue
        record = {
            "request_index": request_index,
            "prompt_ids": list(prepared["prompt_ids"]),
            "prefix_key": str(prepared["prefix_key"]),
            "target_offset": int(prepared["target_offset"]),
            "acceleration": acceleration,
            "beams": [],
            "complete": [],
            "serialization_complete": [],
            "active": True,
            "prefill_bucket_size": 1,
        }
        cached_prefix = (
            prompt_prefix_cache.get(record["prefix_key"])
            if prompt_prefix_cache is not None
            else None
        )
        if cached_prefix is not None:
            logits, cache = cached_prefix
            mx.eval(logits, *cache_arrays(cache))
            record["logits"] = logits
            record["cache"] = cache
            record["prefix_cache_state"] = "HIT"
            records.append(record)
        else:
            record["prefix_cache_state"] = (
                "MISS" if prompt_prefix_cache is not None else "DISABLED"
            )
            prefill_buckets.setdefault(len(record["prompt_ids"]), []).append(record)
            records.append(record)

    for bucket in prefill_buckets.values():
        prefill_started = time.perf_counter()
        batch_ids = mx.array(
            [record["prompt_ids"] for record in bucket], dtype=mx.int32
        )
        logits, cache = model(batch_ids)
        mx.eval(logits, *cache_arrays(cache))
        elapsed = time.perf_counter() - prefill_started
        model_forward_count += 1
        maximum_forward_request_count = max(maximum_forward_request_count, len(bucket))
        if len(bucket) > 1:
            shared_forward_count += 1
            shared_request_indices.update(
                int(record["request_index"]) for record in bucket
            )
        for index, record in enumerate(bucket):
            request_logits = logits[index : index + 1]
            request_cache = [
                tuple(value[index : index + 1] for value in layer_cache)
                for layer_cache in cache
            ]
            record["logits"] = request_logits
            record["cache"] = request_cache
            record["prefill_seconds"] = elapsed
            record["prefill_bucket_size"] = len(bucket)
            if prompt_prefix_cache is not None:
                prompt_prefix_cache.put(
                    record["prefix_key"], (request_logits, request_cache)
                )

    for record in records:
        logits = record["logits"]
        cache = record["cache"]
        record["beams"] = [
            {
                "tokens": [],
                "score": 0.0,
                "logits": logits[0, -1],
                "cache": cache,
            }
        ]
        record["acceleration"].update(
            {
                "prompt_prefix_cache_state": record["prefix_cache_state"],
                "prompt_prefix_sha256": record["prefix_key"],
                "prompt_prefix_token_count": len(record["prompt_ids"]),
                "prompt_prefill_seconds": round(
                    float(record.get("prefill_seconds") or 0.0), 6
                ),
            }
        )

    for _decode_step in range(max_tokens):
        forward_buckets: dict[int, list[dict[str, Any]]] = {}
        active_records = [record for record in records if record["active"]]
        if not active_records:
            break
        for record in active_records:
            expansion_specs: list[dict[str, Any]] = []
            for beam in record["beams"]:
                allowed = serialization_states[
                    bool(active_target_span(beam["tokens"])["active"])
                ]
                if not allowed:
                    continue
                ranked = rank_allowed_logits(
                    beam["logits"],
                    allowed,
                    id_offset=record["target_offset"],
                    branching_factor=branching_factor,
                    device_filter=device_logit_filter,
                    mx=mx,
                )
                for local_id, log_probability in ranked:
                    token = inverse[local_id]
                    score = float(beam["score"]) + log_probability
                    if token == "<eos>":
                        record["complete"].append(
                            {"tokens": list(beam["tokens"]), "score": score}
                        )
                        continue
                    expansion_specs.append(
                        {
                            "beam": beam,
                            "local_id": local_id,
                            "token": token,
                            "log_probability": log_probability,
                            "request_index": record["request_index"],
                        }
                    )
            if preprune_beam_expansions:
                expansion_specs = prune_text_expansion_specs(
                    expansion_specs,
                    limit=beam_width,
                    length_penalty=length_penalty,
                )
            record["pending_expansion_count"] = len(expansion_specs)
            if expansion_specs:
                forward_buckets.setdefault(
                    len(record["prompt_ids"]), []
                ).extend(expansion_specs)
            else:
                record["beams"] = []

        advanced_by_request: dict[int, list[dict[str, Any]]] = {
            int(record["request_index"]): [] for record in active_records
        }
        for expansion_specs in forward_buckets.values():
            request_count = len(
                {int(spec["request_index"]) for spec in expansion_specs}
            )
            maximum_forward_request_count = max(
                maximum_forward_request_count, request_count
            )
            if request_count > 1:
                shared_forward_count += 1
                shared_request_indices.update(
                    int(spec["request_index"]) for spec in expansion_specs
                )
            model_forward_count += 1
            advanced = (
                advance_beams_batched(
                    model,
                    expansion_specs,
                    target_offset=shared_target_offset,
                    mx=mx,
                )
                if batched_beam_advance
                else advance_beams_serial(
                    model,
                    expansion_specs,
                    target_offset=shared_target_offset,
                    mx=mx,
                )
            )
            if first_decode_forward_seconds is None:
                first_decode_forward_seconds = time.perf_counter() - batch_started
            for spec, row in zip(expansion_specs, advanced):
                advanced_by_request[int(spec["request_index"])].append(row)

        for record in active_records:
            request_index = int(record["request_index"])
            record["beams"] = sorted(
                advanced_by_request[request_index],
                key=lambda row: beam_score(row, length_penalty),
                reverse=True,
            )[: max(1, beam_width)]
            record["serialization_complete"].extend(
                {"tokens": list(row["tokens"]), "score": float(row["score"])}
                for row in record["beams"]
                if not bool(active_target_span(row["tokens"])["active"])
            )
            record["serialization_complete"] = sorted(
                record["serialization_complete"],
                key=lambda row: beam_score(row, length_penalty),
                reverse=True,
            )[: max(1, beam_width)]
            record["complete"] = sorted(
                record["complete"],
                key=lambda row: beam_score(row, length_penalty),
                reverse=True,
            )[: max(1, beam_width)]
            if not record["beams"] or (
                record["complete"]
                and len(record["complete"]) >= beam_width
                and beam_score(record["complete"][0], length_penalty)
                >= beam_score(record["beams"][0], length_penalty)
            ):
                record["active"] = False

    batch_seconds = time.perf_counter() - batch_started
    for record in records:
        request_index = int(record["request_index"])
        acceleration = {
            **record["acceleration"],
            "cross_request_batching": "prompt_length_bucketed_v1",
            "cross_request_batch_state": (
                "BATCHED"
                if request_index in shared_request_indices
                else "NO_COMPATIBLE_PEER"
            ),
            "cross_request_batch_size": len(prompts),
            "cross_request_prefill_bucket_size": record["prefill_bucket_size"],
            "cross_request_shared_forward_count": shared_forward_count,
            "cross_request_model_forward_count": model_forward_count,
            "cross_request_maximum_forward_request_count": maximum_forward_request_count,
            "cross_request_batch_seconds": round(batch_seconds, 6),
            "cross_request_first_decode_forward_seconds": (
                round(first_decode_forward_seconds, 6)
                if first_decode_forward_seconds is not None
                else None
            ),
        }
        results[request_index] = finalize_model_text_generation(
            beams=record["beams"],
            complete=record["complete"],
            serialization_complete=record["serialization_complete"],
            length_penalty=length_penalty,
            beam_width=beam_width,
            branching_factor=branching_factor,
            acceleration=acceleration,
            trusted_source_prefix_tokens=trusted_source_prefix_tokens,
        )
    if any(result is None for result in results):
        raise RuntimeError("cross-request generation lost a request result")
    return [result for result in results if result is not None]


def generation_acceleration_receipt(
    *,
    batched_beam_advance: bool,
    device_logit_filter: bool,
    preprune_beam_expansions: bool,
) -> dict[str, Any]:
    """Expose the decode route even when generation rejects its output."""

    return {
        "beam_advance": (
            "mlx_batched_per_token_v1"
            if batched_beam_advance
            else "mlx_serial_per_expansion_reference_v1"
        ),
        "logit_filter": (
            "mlx_allowed_ids_device_topk_v1"
            if device_logit_filter
            else "numpy_target_vocab_reference_v1"
        ),
        "preprune_beam_expansions": bool(preprune_beam_expansions),
    }


def serialization_valid_local_ids(
    generated_tokens: list[str], inverse: dict[int, str]
) -> list[int]:
    active = bool(active_target_span(generated_tokens)["active"])
    return serialization_allowed_local_ids(inverse)[active]


def serialization_allowed_local_ids(
    inverse: dict[int, str],
) -> dict[bool, list[int]]:
    """Compile both byte-serialization grammar states once per request."""

    outside: list[int] = []
    inside: list[int] = []
    for local_id, token in inverse.items():
        if is_byte_token(token) or token == TARGET_BYTE_END:
            inside.append(local_id)
        if token == "<eos>" or token == TARGET_BYTE_BEGIN or (
            token not in {"<pad>", "<unk>", "<bos>", TARGET_BYTE_END}
            and not is_byte_token(token)
        ):
            outside.append(local_id)
    return {False: outside, True: inside}


def rank_allowed_logits(
    logits: Any,
    allowed_ids: list[int],
    *,
    id_offset: int,
    branching_factor: int,
    device_filter: bool,
    mx: Any,
) -> list[tuple[int, float]]:
    """Rank an admissible subset without transferring the full vocabulary."""

    if not allowed_ids:
        return []
    limit = min(len(allowed_ids), max(1, int(branching_factor)))
    if not device_filter:
        values = np.asarray(logits[id_offset:]).astype(np.float64)
        allowed_values = np.asarray(
            [values[token_id] for token_id in allowed_ids], dtype=np.float64
        )
        maximum = float(allowed_values.max())
        normalizer = maximum + float(
            np.log(np.exp(allowed_values - maximum).sum())
        )
        ranked = sorted(
            allowed_ids,
            key=lambda token_id: float(values[token_id]),
            reverse=True,
        )[:limit]
        return [
            (token_id, float(values[token_id]) - normalizer)
            for token_id in ranked
        ]

    local_ids = mx.array(allowed_ids, dtype=mx.int32)
    global_ids = local_ids + int(id_offset)
    allowed_logits = mx.take(logits, global_ids, axis=0)
    log_normalizer = mx.logsumexp(allowed_logits, axis=0)
    selected_positions = mx.argsort(allowed_logits, axis=0)[-limit:]
    selected_logits = mx.take(allowed_logits, selected_positions, axis=0)
    mx.eval(selected_positions, selected_logits, log_normalizer)
    positions = np.asarray(selected_positions, dtype=np.int64)[::-1]
    values = np.asarray(selected_logits, dtype=np.float64)[::-1]
    normalizer = float(log_normalizer.item())
    return [
        (int(allowed_ids[int(position)]), float(value) - normalizer)
        for position, value in zip(positions, values)
    ]


def prune_text_expansion_specs(
    specs: list[dict[str, Any]],
    *,
    limit: int,
    length_penalty: float,
) -> list[dict[str, Any]]:
    """Prune by the exact score used after advance, before paying for logits."""

    unique: dict[tuple[str, ...], tuple[dict[str, Any], float]] = {}
    for spec in specs:
        beam = spec["beam"]
        tokens = tuple(
            json.dumps(token, sort_keys=True, separators=(",", ":"))
            if isinstance(token, (dict, list))
            else str(token)
            for token in (*beam["tokens"], spec["token"])
        )
        score = float(beam["score"]) + float(spec["log_probability"])
        rank = score / (max(1, len(tokens)) ** max(0.0, float(length_penalty)))
        prior = unique.get(tokens)
        if prior is None or rank > prior[1]:
            unique[tokens] = (spec, rank)
    return [
        row[0]
        for row in sorted(
            unique.values(), key=lambda row: row[1], reverse=True
        )[: max(1, int(limit))]
    ]


def beam_score(row: dict[str, Any], length_penalty: float) -> float:
    length = max(1, len(row.get("tokens") or []))
    return float(row.get("score") or 0.0) / (length ** max(0.0, length_penalty))


def generation_fault(reason: str) -> dict[str, Any]:
    return {
        "state": "FAULT",
        "reason": reason,
        "target_visible_to_generator": False,
        "failure_behavior": "reject_without_fallback",
        "fallback_return_count": 0,
    }


def syntax_diagnostic(text: str, arm_id: str) -> dict[str, Any]:
    if arm_id == "python":
        try:
            ast.parse(text)
        except SyntaxError as exc:
            return {"state": "INVALID", "checker": "python_ast", "detail": str(exc)[:200]}
        return {"state": "VALID", "checker": "python_ast"}
    return {
        "state": "NOT_CLAIMED",
        "checker": "none",
        "reason": "language-native parser not yet bound into this evaluation contract",
    }


def behavior_diagnostics(*, generated: str, expected: str, prompt: str) -> dict[str, Any]:
    """Evaluator-only failure telemetry; none of these values enter generation."""

    source_excerpt = ""
    marker = "\nCurrent excerpt:\n"
    terminator = "\n\n\nReturn only the complete revised excerpt."
    if marker in prompt:
        source_excerpt = prompt.split(marker, 1)[1]
        if terminator in source_excerpt:
            source_excerpt = source_excerpt.split(terminator, 1)[0]
    generated_lines = [line for line in generated.splitlines() if line.strip()]
    return {
        "generated_character_count": len(generated),
        "expected_character_count": len(expected),
        "target_length_ratio": round(len(generated) / max(1, len(expected)), 8),
        "target_sequence_similarity": round(
            difflib.SequenceMatcher(None, generated, expected, autojunk=False).ratio(), 8
        ),
        "source_excerpt_available": bool(source_excerpt),
        "source_sequence_similarity": round(
            difflib.SequenceMatcher(
                None, generated, source_excerpt, autojunk=False
            ).ratio(),
            8,
        )
        if source_excerpt
        else None,
        "nonempty_line_count": len(generated_lines),
        "unique_nonempty_line_ratio": round(
            len(set(generated_lines)) / max(1, len(generated_lines)), 8
        ),
        "raw_generated_text_retained": False,
    }


def evaluation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(bool(row.get("exact_match")) for row in rows)
    nonempty = sum(bool(row.get("nonempty")) for row in rows)
    serialization_valid = sum(
        bool((row.get("generation") or {}).get("byte_serialization_valid")) for row in rows
    )
    syntax_valid = sum((row.get("syntax") or {}).get("state") == "VALID" for row in rows)
    syntax_checked = sum(
        (row.get("syntax") or {}).get("state") in {"VALID", "INVALID"} for row in rows
    )
    similarities = [
        float((row.get("behavior_diagnostics") or {}).get("target_sequence_similarity") or 0.0)
        for row in rows
    ]
    source_similarities = [
        float(value)
        for row in rows
        if (
            value := (row.get("behavior_diagnostics") or {}).get(
                "source_sequence_similarity"
            )
        )
        is not None
    ]
    length_ratios = [
        float((row.get("behavior_diagnostics") or {}).get("target_length_ratio") or 0.0)
        for row in rows
    ]
    return {
        "row_count": total,
        "exact_match_count": exact,
        "exact_target_match_rate": round(exact / max(1, total), 8),
        "nonempty_count": nonempty,
        "nonempty_rate": round(nonempty / max(1, total), 8),
        "byte_serialization_valid_count": serialization_valid,
        "byte_serialization_valid_rate": round(serialization_valid / max(1, total), 8),
        "syntax_checked_count": syntax_checked,
        "syntax_valid_count": syntax_valid,
        "syntax_valid_rate_when_checked": round(syntax_valid / max(1, syntax_checked), 8),
        "mean_target_sequence_similarity": round(sum(similarities) / max(1, total), 8),
        "mean_source_sequence_similarity": round(
            sum(source_similarities) / max(1, len(source_similarities)), 8
        ),
        "mean_target_length_ratio": round(sum(length_ratios) / max(1, total), 8),
        "raw_generated_text_retained": False,
    }


def _candidate_tensor_manifest(mapping: dict[str, np.ndarray]) -> dict[str, Any]:
    manifest = tensor_mapping_manifest(mapping)
    return {
        "sha256": manifest["sha256"],
        "tensor_count": manifest["tensor_count"],
        "element_count": manifest["element_count"],
        "payload_bytes": manifest["payload_bytes"],
    }


def canonical_tensor_dtype(value: Any) -> str:
    """Normalize equivalent NumPy and MLX dtype spellings for custody checks."""

    return str(getattr(value, "dtype", value)).rsplit(".", 1)[-1]


def exact_candidate_continuation_initialization_receipt(
    state: dict[str, Any] | None,
    *,
    target_id: str,
    seed: int,
) -> dict[str, Any] | None:
    """Describe the exact-checkpoint initialization path without materializing a random reference.

    Matched scratch candidates need a common random named-tensor reference.
    A governed one-shot continuation does not: its already validated model,
    optimizer, and RNG artifacts replace the fresh model before any optimizer
    step. Writing hundreds of random reference tensors in that case adds disk,
    memory, and swap pressure without affecting the resumed state.
    """

    continuation = dict((state or {}).get("candidate_continuation") or {})
    if not continuation:
        return None
    if continuation.get("policy") != (
        "project_theseus_exact_candidate_continuation_import_v1"
    ):
        raise ValueError("candidate continuation import policy mismatch")
    required = (
        "source_checkpoint_sha256",
        "source_optimizer_state_sha256",
        "source_mlx_rng_state_sha256",
    )
    if any(not str(continuation.get(key) or "") for key in required):
        raise ValueError("candidate continuation exact state identity is incomplete")
    return {
        "policy": "project_theseus_exact_candidate_continuation_initialization_v1",
        "state": "EXACT_CHECKPOINT_IMPORT_REPLACES_RANDOM_INITIALIZATION",
        "target_id": target_id,
        "seed": int(seed),
        "source_checkpoint_sha256": continuation["source_checkpoint_sha256"],
        "source_optimizer_state_sha256": continuation[
            "source_optimizer_state_sha256"
        ],
        "source_mlx_rng_state_sha256": continuation[
            "source_mlx_rng_state_sha256"
        ],
        "common_random_reference_required": False,
        "common_initialization_files_written": 0,
        "optimizer_step_before_exact_state_load": False,
    }


def load_kerc_stage_selective_compute_checkpoint(
    model: Any,
    checkpoint: Path,
    *,
    stage_index: int,
    include_stage_embedding: bool = True,
    mx: Any,
    mlx_utils: Any,
) -> dict[str, Any]:
    """Load one diagnostic checkpoint with exact FP32 stage trainables.

    Frozen tensors are materialized as BF16 to avoid retaining a second full
    FP32 model. The selected stage tensors remain the source checkpoint's exact
    FP32 values. This is a bounded compute representation, not a canonical
    checkpoint migration; any behavior-positive delta must be merged back into
    the independently verified FP32 source lineage before promotion.
    """

    if stage_index != 1:
        raise ValueError(
            "selective compute checkpoint loading currently supports KERC stage 1"
        )
    if not checkpoint.is_file():
        raise ValueError("selective compute source checkpoint is missing")
    destination = dict(mlx_utils.tree_flatten(model.parameters()))
    selected = dict(mlx_utils.tree_flatten(model.trainable_parameters()))
    if not destination or not selected:
        raise ValueError("selective compute model scope is empty")
    _metadata, source_index, payload_offset = safetensors_payload_index(
        checkpoint
    )
    if set(source_index) != set(destination):
        missing = sorted(set(destination) - set(source_index))
        unexpected = sorted(set(source_index) - set(destination))
        raise ValueError(
            "selective compute checkpoint tensor identity mismatch:"
            f"missing={','.join(missing[:8])};unexpected={','.join(unexpected[:8])}"
        )
    shape_faults = sorted(
        name
        for name, row in source_index.items()
        if row.get("dtype") != "F32"
        or tuple(int(value) for value in row["shape"])
        != tuple(destination[name].shape)
    )
    if shape_faults:
        raise ValueError(
            "selective compute checkpoint tensor dtype or shape mismatch:"
            + ",".join(shape_faults[:8])
        )
    selected_names = set(selected)
    expected_selected_count = 5 if include_stage_embedding else 4
    if (
        len(selected_names) != expected_selected_count
        or (
            include_stage_embedding
            and not any(
                name.startswith("kerc_stage_embedding.")
                for name in selected_names
            )
        )
        or (
            not include_stage_embedding
            and any(
                name.startswith("kerc_stage_embedding.")
                for name in selected_names
            )
        )
        or not any(name.startswith("kerc_stage_adapters.1.") for name in selected_names)
        or not any(name.startswith("kerc_kernel_output.") for name in selected_names)
    ):
        raise ValueError("selective compute compiler trainable scope is not exact")
    selected_source_raw_sha256 = {
        name: safetensors_raw_tensor_sha256(
            checkpoint,
            source_index[name],
            payload_offset=payload_offset,
        )
        for name in sorted(selected_names)
    }

    batch: list[tuple[str, Any]] = []
    batch_bytes = 0
    for name in sorted(source_index):
        row = source_index[name]
        start, _stop = (int(value) for value in row["data_offsets"])
        source_view = np.memmap(
            checkpoint,
            dtype=np.float32,
            mode="r",
            offset=payload_offset + start,
            shape=tuple(int(value) for value in row["shape"]),
            order="C",
        )
        value = mx.array(
            source_view,
            dtype=(
                mx.float32
                if name in selected_names
                else mx.bfloat16
            ),
        )
        batch.append((name, value))
        batch_bytes += int(value.size) * (4 if name in selected_names else 2)
        if batch_bytes >= 16 * 1024 * 1024:
            model.load_weights(batch, strict=False)
            mx.eval(*[tensor for _name, tensor in batch])
            batch = []
            batch_bytes = 0
    if batch:
        model.load_weights(batch, strict=False)
        mx.eval(*[tensor for _name, tensor in batch])

    loaded = dict(mlx_utils.tree_flatten(model.parameters()))
    selected_loaded_raw_sha256: dict[str, str] = {}
    selected_exact = True
    for name in sorted(selected_names):
        row = source_index[name]
        start, _stop = (int(value) for value in row["data_offsets"])
        source_view = np.memmap(
            checkpoint,
            dtype=np.float32,
            mode="r",
            offset=payload_offset + start,
            shape=tuple(int(value) for value in row["shape"]),
            order="C",
        )
        loaded_array = np.ascontiguousarray(np.asarray(loaded[name]))
        loaded_sha256 = hashlib.sha256(
            memoryview(loaded_array).cast("B")
        ).hexdigest()
        selected_loaded_raw_sha256[name] = loaded_sha256
        selected_exact = bool(
            selected_exact
            and loaded_sha256 == selected_source_raw_sha256[name]
            and np.array_equal(source_view, loaded_array)
        )
        del loaded_array, source_view
    selected_fp32 = all(
        canonical_tensor_dtype(loaded[name]) == "float32"
        for name in selected_names
    )
    frozen_bfloat16 = all(
        canonical_tensor_dtype(value) == "bfloat16"
        for name, value in loaded.items()
        if name not in selected_names
    )
    if not selected_exact or not selected_fp32 or not frozen_bfloat16:
        raise ValueError("selective compute checkpoint dtype or custody check failed")
    selected_source_manifest_sha256 = hashlib.sha256(
        json.dumps(
            selected_source_raw_sha256,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    selected_loaded_manifest_sha256 = hashlib.sha256(
        json.dumps(
            selected_loaded_raw_sha256,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "policy": "project_theseus_kerc_stage_selective_compute_checkpoint_v1",
        "stage_index": stage_index,
        "scope": "compiler",
        "stage_embedding_included": bool(include_stage_embedding),
        "source_checkpoint": relative(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_tensor_count": len(source_index),
        "selected_tensor_names": sorted(selected_names),
        "selected_tensor_count": len(selected_names),
        "selected_source_raw_tensor_sha256": selected_source_raw_sha256,
        "selected_loaded_raw_tensor_sha256": selected_loaded_raw_sha256,
        "selected_source_manifest_sha256": selected_source_manifest_sha256,
        "selected_loaded_manifest_sha256": selected_loaded_manifest_sha256,
        "selected_fp32_exact": selected_exact,
        "frozen_parameter_dtype": "bfloat16",
        "frozen_tensor_count": len(source_index) - len(selected_names),
        "canonical_checkpoint_migration": False,
        "merge_into_verified_fp32_source_required_for_promotion": True,
        "optimizer_step_before_custody_check": False,
        "capability_claim": "NONE_DIAGNOSTIC_COMPUTE_REPRESENTATION",
    }


def apply_kerc_stage_selective_delta_checkpoint(
    model: Any,
    checkpoint: Path,
    *,
    mx: Any,
    mlx_utils: Any,
) -> dict[str, Any]:
    """Overlay one exact FP32 compiler delta on a verified source-backed model."""

    if not checkpoint.is_file():
        raise ValueError("selective compute delta checkpoint is missing")
    selected = dict(mlx_utils.tree_flatten(model.trainable_parameters()))
    delta = dict(mx.load(str(checkpoint)))
    if set(delta) != set(selected):
        raise ValueError("selective compute delta tensor identity mismatch")
    faults = sorted(
        name
        for name, value in delta.items()
        if canonical_tensor_dtype(value) != "float32"
        or tuple(value.shape) != tuple(selected[name].shape)
    )
    if faults:
        raise ValueError(
            "selective compute delta dtype or shape mismatch:"
            + ",".join(faults)
        )
    model.load_weights(list(delta.items()), strict=False)
    mx.eval(model.trainable_parameters())
    loaded = dict(mlx_utils.tree_flatten(model.trainable_parameters()))
    mismatches = sorted(
        name
        for name in delta
        if not np.array_equal(np.asarray(delta[name]), np.asarray(loaded[name]))
    )
    if mismatches:
        raise ValueError(
            "selective compute delta load custody mismatch:"
            + ",".join(mismatches)
        )
    raw_sha256 = {
        name: hashlib.sha256(
            memoryview(np.ascontiguousarray(np.asarray(value))).cast("B")
        ).hexdigest()
        for name, value in sorted(delta.items())
    }
    return {
        "policy": "project_theseus_exact_kerc_stage_delta_overlay_v1",
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tensor_count": len(delta),
        "tensor_names": sorted(delta),
        "raw_tensor_sha256": raw_sha256,
        "loaded_exact": True,
    }


def align_candidate_common_initialization(
    model: Any,
    *,
    state: dict[str, Any] | None,
    target_id: str,
    seed: int,
    receipt_path: Path,
    mx: Any,
    mlx_utils: Any,
) -> dict[str, Any] | None:
    """Align common candidate tensors without hiding candidate-only parameters.

    A shared PRNG seed is insufficient when two architectures consume a different
    number of random draws during construction.  The first candidate is therefore
    the named-tensor reference.  Later candidates copy only tensors with the same
    name, shape, and dtype.  Candidate-specific tensors remain at their original
    initialization and are content-checked before and after alignment.
    """

    if state is None:
        return None
    if seed <= 0:
        raise ValueError("candidate common initialization requires a positive seed")
    flat = dict(mlx_utils.tree_flatten(model.parameters()))
    if not flat:
        raise ValueError("candidate model has no parameters to align")
    reference = state.get("reference")
    if reference is None:
        reference_root = receipt_path.parent.parent / "candidate_common_initialization"
        reference_root.mkdir(parents=True, exist_ok=True)
        reference_files: dict[str, dict[str, Any]] = {}
        for name, value in sorted(flat.items()):
            array = np.array(value)
            filename = hashlib.sha256(name.encode("utf-8")).hexdigest() + ".npy"
            path = reference_root / filename
            np.save(path, array, allow_pickle=False)
            reference_files[name] = {
                "path": str(path),
                "shape": list(array.shape),
                "dtype": canonical_tensor_dtype(array),
                "nbytes": int(array.nbytes),
                "sha256": sha256_file(path),
            }
            del array
        receipt: dict[str, Any] = {
            "policy": "project_theseus_candidate_common_initialization_v1",
            "role": "reference",
            "target_id": target_id,
            "seed": int(seed),
            "exact_alignment": None,
            "common_tensor_manifest": None,
            "architecture_specific_tensor_manifest": None,
            "architecture_specific_tensors_unchanged": True,
            "comparison_state": "PENDING_MATCHED_TARGET",
        }
        state.update(
            {
                "seed": int(seed),
                "reference_target_id": target_id,
                "reference": reference_files,
                "reference_storage": "disk_backed_npy_per_tensor_v1",
                "reference_receipt": receipt,
                "reference_receipt_path": str(receipt_path),
            }
        )
        return receipt
    if int(state.get("seed") or 0) != int(seed):
        raise ValueError("candidate common initialization seed changed within pair")
    if state.get("aligned_target_id"):
        raise ValueError("candidate common initialization supports exactly one matched pair")

    common_names = sorted(
        name
        for name, value in flat.items()
        if name in reference
        and tuple(reference[name]["shape"]) == tuple(value.shape)
        and canonical_tensor_dtype(reference[name]["dtype"])
        == canonical_tensor_dtype(value)
    )
    if not common_names:
        raise ValueError("candidate pair has no common named tensors to align")
    unique_before = {
        name: value for name, value in flat.items() if name not in common_names
    }
    unique_before_manifest = _candidate_tensor_manifest(unique_before)
    batch: list[tuple[str, Any]] = []
    batch_bytes = 0
    for name in common_names:
        metadata = reference[name]
        path = Path(str(metadata["path"]))
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError("candidate initialization reference tensor drifted")
        batch.append((name, mx.array(np.load(path, mmap_mode="r"))))
        batch_bytes += int(metadata["nbytes"])
        if batch_bytes >= 32 * 1024 * 1024:
            model.load_weights(batch, strict=False)
            mx.eval(model.parameters())
            batch = []
            batch_bytes = 0
    if batch:
        model.load_weights(batch, strict=False)
        mx.eval(model.parameters())
    aligned = dict(mlx_utils.tree_flatten(model.parameters()))
    common_after = {name: aligned[name] for name in common_names}
    reference_common = {
        name: np.load(Path(str(reference[name]["path"])), mmap_mode="r")
        for name in common_names
    }
    reference_manifest = _candidate_tensor_manifest(reference_common)
    aligned_manifest = _candidate_tensor_manifest(common_after)
    unique_after = {
        name: value for name, value in aligned.items() if name not in common_names
    }
    unique_after_manifest = _candidate_tensor_manifest(unique_after)
    exact = (
        reference_manifest == aligned_manifest
        and all(
            np.array_equal(reference_common[name], common_after[name])
            for name in common_names
        )
    )
    unique_unchanged = unique_before_manifest == unique_after_manifest
    if not exact or not unique_unchanged:
        raise ValueError("candidate common initialization integrity failure")

    reference_unique = {
        name: np.load(Path(str(value["path"])), mmap_mode="r")
        for name, value in reference.items()
        if name not in common_names
    }
    reference_receipt = state["reference_receipt"]
    reference_receipt.update(
        {
            "exact_alignment": True,
            "common_tensor_manifest": reference_manifest,
            "architecture_specific_tensor_manifest": _candidate_tensor_manifest(
                reference_unique
            ),
            "architecture_specific_tensors_unchanged": True,
            "comparison_state": "EXACT_COMMON_SUBSPACE_ALIGNED",
            "matched_target_id": target_id,
        }
    )
    receipt = {
        "policy": "project_theseus_candidate_common_initialization_v1",
        "role": "aligned",
        "target_id": target_id,
        "seed": int(seed),
        "exact_alignment": True,
        "common_tensor_manifest": aligned_manifest,
        "architecture_specific_tensor_manifest": unique_after_manifest,
        "architecture_specific_tensors_unchanged": unique_unchanged,
        "comparison_state": "EXACT_COMMON_SUBSPACE_ALIGNED",
        "reference_target_id": state["reference_target_id"],
    }
    state["aligned_target_id"] = target_id
    state["aligned_receipt"] = receipt

    # The first target has already published its transactional training receipt.
    # Advance only its initialization subreceipt after the second target proves
    # the common subset; all other immutable checkpoint fields remain unchanged.
    reference_path = Path(str(state["reference_receipt_path"]))
    if reference_path.is_file():
        durable = read_json(reference_path)
        durable["candidate_initialization"] = reference_receipt
        write_json_atomic(reference_path, durable)
    return receipt


def initialize_kerc_from_shared_trunk(
    model: Any,
    *,
    checkpoint: Path,
    receipt_path: Path,
    mx: Any,
    mlx_utils: Any,
) -> dict[str, Any]:
    """Warm-start the exact KERC/common subspace from the registered trunk.

    KERC extends the canonical vocabulary and adds architecture-specific heads,
    so the shared token embedding is prefix-expanded while every shape-identical
    trunk tensor is copied exactly.  New KERC rows and modules retain their
    seeded initialization.
    """

    if not checkpoint.is_file() or not receipt_path.is_file():
        raise ValueError("KERC shared-trunk initialization artifacts are missing")
    receipt = read_json(receipt_path)
    checkpoint_sha256 = sha256_file(checkpoint)
    if (
        checkpoint_sha256 != str(receipt.get("checkpoint_sha256") or "")
        or int(receipt.get("optimizer_steps") or 0) < 3000
        or receipt.get("trigger_state") != "GREEN"
        or int(receipt.get("public_training_rows_written") or 0)
        or int(receipt.get("external_inference_calls") or 0)
        or int(receipt.get("fallback_return_count") or 0)
    ):
        raise ValueError("KERC shared-trunk progress checkpoint custody failed")
    source = dict(mx.load(str(checkpoint)))
    destination = dict(mlx_utils.tree_flatten(model.parameters()))
    missing = sorted(set(source) - set(destination))
    if missing:
        raise ValueError(
            "KERC model is missing shared-trunk tensors: " + ",".join(missing[:8])
        )
    exact: list[tuple[str, Any]] = []
    expanded: list[tuple[str, Any]] = []
    incompatible: list[str] = []
    for name, value in source.items():
        target = destination[name]
        if tuple(value.shape) == tuple(target.shape):
            exact.append((name, value))
            continue
        if (
            (
                name == "token_embedding.weight"
                or name.startswith("mtp_output_heads.")
            )
            and len(value.shape) == len(target.shape) == 2
            and int(value.shape[0]) < int(target.shape[0])
            and int(value.shape[1]) == int(target.shape[1])
        ):
            expanded.append(
                (
                    name,
                    mx.concatenate(
                        [value, target[int(value.shape[0]) :]], axis=0
                    ),
                )
            )
            continue
        incompatible.append(name)
    required_expansions = {
        "token_embedding.weight",
        "mtp_output_heads.0.weight",
        "mtp_output_heads.1.weight",
        "mtp_output_heads.2.weight",
    }
    if incompatible or {name for name, _value in expanded} != required_expansions:
        raise ValueError(
            "KERC shared-trunk tensor compatibility failure:"
            + ",".join(incompatible[:8])
        )
    model.load_weights([*exact, *expanded], strict=False)
    mx.eval(model.parameters())
    warmed = dict(mlx_utils.tree_flatten(model.parameters()))
    embedding = warmed.get("token_embedding.weight")
    output_head_names = (
        "kerc_kernel_output.weight",
        "kerc_surface_output.weight",
    )
    if embedding is None or any(
        name not in warmed or tuple(warmed[name].shape) != tuple(embedding.shape)
        for name in output_head_names
    ):
        raise ValueError("KERC output heads cannot inherit the warm token embedding")
    model.load_weights(
        [(name, embedding) for name in output_head_names], strict=False
    )
    mx.eval(model.parameters())
    initialized = dict(mlx_utils.tree_flatten(model.parameters()))
    if any(
        not np.array_equal(initialized[name], initialized["token_embedding.weight"])
        for name in output_head_names
    ):
        raise ValueError("KERC warm output-head initialization is not exact")
    shared_elements = sum(int(value.size) for _name, value in exact) + sum(
        int(source[name].size) for name, _value in expanded
    )
    return {
        "policy": "registered_shared_trunk_progress_checkpoint_common_subspace_v1",
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "receipt": relative(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "source_optimizer_steps": int(receipt["optimizer_steps"]),
        "source_receipt_complete": bool(receipt.get("complete")),
        "source_receipt_state": receipt.get("trigger_state"),
        "exact_shape_tensor_count": len(exact),
        "prefix_expanded_tensor_names": [name for name, _value in expanded],
        "source_tensor_count": len(source),
        "shared_element_count": shared_elements,
        "new_kerc_non_output_tensor_initialization_preserved": True,
        "kerc_output_head_initialization": {
            "policy": "warm_token_embedding_exact_copy_v1",
            "head_names": list(output_head_names),
            "head_count": len(output_head_names),
            "canonical_rows_inherit_shared_trunk": True,
            "kerc_only_rows_inherit_seeded_token_embeddings": True,
            "heads_remain_independently_trainable": True,
            "answer_metadata_used": False,
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "capability_claim": "NONE_PROGRESS_CHECKPOINT_INITIALIZATION_ONLY",
    }


def kerc_segmented_delta_resume_required(receipt: dict[str, Any]) -> bool:
    """Distinguish a five-tensor delta generation from its full merged image."""

    selective = dict(receipt.get("selective_compute_checkpoint") or {})
    return bool(
        receipt.get("current_kernel_phase_position_accounting_reset")
        and selective.get("source_checkpoint")
        and receipt.get("checkpoint_representation")
        != "full_fp32_source_with_exact_compiler_delta_merge_v1"
    )


def train_target(
    config: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    *,
    stage: Any,
    max_steps: int,
    resume: bool,
    training_phase: str = "all",
    mx: Any,
    nn: Any,
    optim: Any,
    mlx_utils: Any,
    source_conditioned_stage: Any | None = None,
    kernel_english_stage: Any | None = None,
    supervision_stage: Any | None = None,
    step_boundary_callback: Any = None,
    optimizer_id: str = "",
    candidate_seed: int = 0,
    candidate_initialization_state: dict[str, Any] | None = None,
    qualification_phase_step_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    active_phases = {
        "pretraining",
        "source_conditioned_pretraining",
        "kernel_english",
        "supervision",
    }
    if training_phase != "all":
        if training_phase not in active_phases:
            raise ValueError(f"unknown training phase: {training_phase}")
        active_phases = {training_phase}
    phase_step_limits = {
        str(key): int(value)
        for key, value in (
            qualification_phase_step_limits or {}
        ).items()
    }
    if (
        any(key not in active_phases for key in phase_step_limits)
        or any(value < 0 for value in phase_step_limits.values())
    ):
        raise ValueError(
            "qualification phase step limits require active phases and nonnegative values"
        )

    def bounded_phase_steps(phase: str, available: int) -> int:
        return min(
            int(available),
            int(phase_step_limits.get(phase, available)),
        )
    target_id = str(target["target_id"])
    effective_seed = (
        int(candidate_seed)
        if candidate_seed > 0
        else int(config["seed"]) + stable_int(target_id)
    )
    candidate_execution_policy = dict(
        (candidate_initialization_state or {}).get("execution_policy") or {}
    )
    candidate_continuation = dict(
        (candidate_initialization_state or {}).get("candidate_continuation") or {}
    )
    exact_continuation_initialization = (
        exact_candidate_continuation_initialization_receipt(
            candidate_initialization_state,
            target_id=target_id,
            seed=effective_seed,
        )
    )
    candidate_cache_limit_bytes = int(
        candidate_execution_policy.get("mlx_cache_limit_mib") or 0
    ) * 1024 * 1024
    if candidate_cache_limit_bytes:
        mx.set_cache_limit(candidate_cache_limit_bytes)
        mx.clear_cache()
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
    trained_vocab_size = int(
        target.get("vocab_size") or plan["models"]["vocab_size"]
    )
    if "pretraining" in active_phases:
        inputs = range_view(stage.pretrain_inputs, target["row_ranges"])
        labels = range_view(stage.pretrain_labels, target["row_ranges"])
        mask = range_view(stage.pretrain_mask, target["row_ranges"])
    else:
        inputs = np.empty((0, 1), dtype=np.int32)
        labels = np.empty((0, 1), dtype=np.int32)
        mask = np.empty((0, 1), dtype=np.uint8)
    copy_lookup = None
    if str((target.get("model") or {}).get("source_copy_mode") or "none") != "none":
        copy_lookup = build_source_to_target_lookup(
            read_json(resolve(str(config["base_config"]))),
            read_json(resolve(str(config["stage_dir"])) / "stage_metadata_v1.json"),
            vocab_size=trained_vocab_size,
            identity_ranges=target_copy_identity_ranges(target),
        )
    if candidate_seed > 0:
        random.seed(effective_seed)
        mx.random.seed(effective_seed)
    model_config = CausalTransformerConfig(
        vocab_size=trained_vocab_size, **target["model"]
    )
    selective_fp32_trainables = bool(
        candidate_execution_policy.get("selective_fp32_trainables", False)
    )
    kerc_stage_only_value = candidate_execution_policy.get(
        "kerc_delta_stage_only"
    )
    kerc_stage_only = (
        int(kerc_stage_only_value) if kerc_stage_only_value is not None else None
    )
    kerc_stage_train_stage_embedding = bool(
        candidate_execution_policy.get(
            "kerc_stage_train_stage_embedding", True
        )
    )
    kerc_stage_detach_frozen_trunk = bool(
        candidate_execution_policy.get(
            "kerc_stage_detach_frozen_trunk", False
        )
    )
    gradient_checkpointing_active = bool(
        candidate_execution_policy.get("gradient_checkpointing", False)
    )
    parameter_initialization_dtype = str(
        candidate_execution_policy.get("parameter_initialization_dtype")
        or "float32"
    )
    if selective_fp32_trainables and (
        not candidate_continuation
        or candidate_execution_policy.get("compute_dtype") != "bfloat16"
        or candidate_execution_policy.get("fp32_master") is not False
        or parameter_initialization_dtype != "bfloat16"
        or candidate_execution_policy.get(
            "exact_checkpoint_placeholder_initialization"
        )
        is not True
        or candidate_execution_policy.get("freeze_warm_trunk_train_kerc_delta")
        is not True
        or int(candidate_execution_policy.get("kerc_delta_stage_only") or -1)
        != 1
        or (
            (kerc_stage_train_stage_embedding, kerc_stage_detach_frozen_trunk)
            not in {(True, False), (False, True)}
        )
        or candidate_execution_policy.get(
            "continuation_optimizer_state_projection_policy"
        )
        != "project_theseus_exact_kerc_stage_optimizer_projection_v1"
    ):
        raise ValueError(
            "selective FP32 trainables require the exact stage-1 BF16 "
            "continuation policy"
        )

    def record_selective_initialization_stage(stage_name: str) -> None:
        if not selective_fp32_trainables:
            return
        diagnostic_receipt_path = resolve(str(target["receipt"]))
        if not diagnostic_receipt_path.is_file():
            raise ValueError(
                "selective compute continuation receipt is missing"
            )
        diagnostic_receipt = read_json(diagnostic_receipt_path)
        diagnostic_receipt["selective_compute_initialization_stage"] = (
            stage_name
        )
        diagnostic_receipt[
            "selective_compute_initialization_stage_optimizer_step"
        ] = False
        write_json_atomic(diagnostic_receipt_path, diagnostic_receipt)

    model = build_model(
        model_config,
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=copy_lookup,
        rope_kernel=str(
            config["training"].get("training_rope_kernel")
            or "manual_reference"
        ),
        gradient_checkpointing=gradient_checkpointing_active,
        attention_query_chunk_size=int(
            candidate_execution_policy.get("attention_query_chunk_size") or 0
        ),
        attention_key_chunk_size=int(
            candidate_execution_policy.get("attention_key_chunk_size") or 0
        ),
        compact_encoder_decoder_partitions=bool(
            candidate_execution_policy.get(
                "compact_encoder_decoder_partitions", False
            )
        ),
        compact_partition_width_quantum=int(
            candidate_execution_policy.get(
                "compact_partition_width_quantum", 0
            )
        ),
        parameter_initialization_dtype=parameter_initialization_dtype,
        exact_checkpoint_placeholder_initialization=bool(
            candidate_execution_policy.get(
                "exact_checkpoint_placeholder_initialization", False
            )
        ),
    )
    record_selective_initialization_stage("BF16_MODEL_CONSTRUCTED")
    shared_trunk_initialization = None
    if (
        target.get("role") == "kerc_english_candidate"
        and candidate_execution_policy.get("initialization_policy")
        == "registered_shared_trunk_progress_checkpoint_common_subspace_v1"
        and not candidate_continuation
    ):
        shared_checkpoint = resolve(str(target.get("shared_trunk_checkpoint") or ""))
        shared_trunk_initialization = initialize_kerc_from_shared_trunk(
            model,
            checkpoint=shared_checkpoint,
            receipt_path=shared_checkpoint.parent / "training_receipt.json",
            mx=mx,
            mlx_utils=mlx_utils,
        )
    candidate_initialization_receipt = (
        exact_continuation_initialization
        if exact_continuation_initialization is not None
        else align_candidate_common_initialization(
            model,
            state=candidate_initialization_state,
            target_id=target_id,
            seed=effective_seed,
            receipt_path=resolve(str(target["receipt"])),
            mx=mx,
            mlx_utils=mlx_utils,
        )
    )
    if shared_trunk_initialization is not None:
        candidate_initialization_receipt = {
            **candidate_initialization_receipt,
            "shared_trunk_initialization": shared_trunk_initialization,
        }
    kerc_delta_scope = bool(
        candidate_execution_policy.get("freeze_warm_trunk_train_kerc_delta", False)
    )
    kerc_source_bridge = bool(
        candidate_execution_policy.get(
            "kerc_delta_include_source_conditioned_bridge", False
        )
    )
    if kerc_stage_only is not None:
        if not kerc_delta_scope or target.get("role") != "kerc_english_candidate":
            raise ValueError("KERC stage-only scope requires frozen KERC delta training")
        model.freeze_to_kerc_stage(
            kerc_stage_only,
            include_stage_embedding=kerc_stage_train_stage_embedding,
            detach_frozen_trunk=kerc_stage_detach_frozen_trunk,
        )
    elif kerc_delta_scope:
        if target.get("role") != "kerc_english_candidate":
            raise ValueError("KERC delta scope requires the KERC English candidate")
        model.freeze_to_kerc_delta(
            include_source_conditioned_bridge=kerc_source_bridge
        )
    record_selective_initialization_stage("COMPILER_TRAINABLE_SCOPE_BOUND")
    expert_mode = target.get("role") == "language_expert"
    expert_scope = ""
    shared_trunk_checkpoint = resolve(str(target.get("shared_trunk_checkpoint") or ""))
    shared_trunk_checkpoint_sha256 = ""
    if expert_mode:
        if not shared_trunk_checkpoint.is_file():
            raise ValueError("language expert requires a completed shared trunk checkpoint")
        shared_receipt_path = shared_trunk_checkpoint.parent / "training_receipt.json"
        shared_receipt = read_json(shared_receipt_path)
        if not bool(shared_receipt.get("complete")):
            raise ValueError("language expert requires a complete shared trunk receipt")
        shared_trunk_checkpoint_sha256 = sha256_file(shared_trunk_checkpoint)
        if shared_trunk_checkpoint_sha256 != shared_receipt.get("checkpoint_sha256"):
            raise ValueError("shared trunk checkpoint identity mismatch")
        model.load_weights(str(shared_trunk_checkpoint), strict=False)
        expert_scope = str(
            target.get("expert_trainable_scope")
            or config["topology"]["expert_trainable_scope"]
        )
        model.freeze_to_language_expert(expert_scope)
    authoritative_model = model
    master_model = None
    production_execution_policy = (
        dict((config.get("training") or {}).get("execution_policy") or {})
        if candidate_initialization_state is None
        else {}
    )
    compute_execution_policy = (
        candidate_execution_policy
        if candidate_initialization_state is not None
        else production_execution_policy
    )
    compute_dtype_name = str(
        compute_execution_policy.get("compute_dtype") or "float32"
    )
    if compute_execution_policy.get("fp32_master") is True:
        if compute_dtype_name != "bfloat16":
            raise ValueError("FP32 master requires bfloat16 execution")
        master_model = authoritative_model
        model = build_model(
            model_config,
            mx=mx,
            nn=nn,
            state_role_lookup=None,
            source_to_target_lookup=copy_lookup,
            rope_kernel=str(
                config["training"].get("training_rope_kernel")
                or "manual_reference"
            ),
            gradient_checkpointing=gradient_checkpointing_active,
            attention_query_chunk_size=int(
                candidate_execution_policy.get("attention_query_chunk_size") or 0
            ),
            attention_key_chunk_size=int(
                candidate_execution_policy.get("attention_key_chunk_size") or 0
            ),
            compact_encoder_decoder_partitions=bool(
                candidate_execution_policy.get(
                    "compact_encoder_decoder_partitions", False
                )
            ),
            compact_partition_width_quantum=int(
                candidate_execution_policy.get(
                    "compact_partition_width_quantum", 0
                )
            ),
        )
        model.load_weights(
            list(mlx_utils.tree_flatten(master_model.parameters())), strict=True
        )
        if kerc_stage_only is not None:
            model.freeze_to_kerc_stage(
                kerc_stage_only,
                include_stage_embedding=kerc_stage_train_stage_embedding,
                detach_frozen_trunk=kerc_stage_detach_frozen_trunk,
            )
        elif kerc_delta_scope:
            model.freeze_to_kerc_delta(
                include_source_conditioned_bridge=kerc_source_bridge
            )
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters(), master_model.parameters())
    observed_parameters = int(parameter_count(authoritative_model, mlx_utils))
    if observed_parameters != int(target["parameter_count"]):
        raise ValueError("target model parameter identity changed after preregistration")
    trainable_parameters = int(
        sum(
            value.size
            for _name, value in mlx_utils.tree_flatten(
                authoritative_model.trainable_parameters()
            )
        )
    )
    if expert_mode and trainable_parameters != int(
        plan["models"]["moecot_system"]["expert_parameter_count_per_arm"]
    ):
        raise ValueError("expert trainable parameter ownership mismatch")
    checkpoint = resolve(str(target["checkpoint"]))
    optimizer_path = resolve(str(target["optimizer_state"]))
    receipt_path = resolve(str(target["receipt"]))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    optimizer_target_positions = int(
        target.get("optimizer_target_positions")
        or target.get("unique_target_positions")
        or 0
    )
    optimizer_repetition_factor = float(
        target.get("optimizer_repetition_factor")
        or (
            optimizer_target_positions
            / max(1, int(target.get("unique_target_positions") or 0))
        )
    )
    planned_steps = (
        required_steps(
            mask,
            int(training["batch_size"]),
            optimizer_target_positions,
        )
        if "pretraining" in active_phases
        else 0
    )
    deferred_supervision = isinstance(
        supervision_stage, DeferredSupervisionStage
    )
    deferred_source_conditioned = isinstance(
        source_conditioned_stage, DeferredSupervisionStage
    )
    supervision_planning_rows = (
        int(supervision_stage.planning_row_count)
        if deferred_supervision
        else len(supervision_stage.mask)
        if supervision_stage is not None
        else 0
    )
    source_planning_rows = (
        int(source_conditioned_stage.planning_row_count)
        if deferred_source_conditioned
        else len(source_conditioned_stage.mask)
        if source_conditioned_stage is not None
        else 0
    )
    unique_sft_positions = (
        int(supervision_stage.mask.sum())
        if supervision_stage is not None and not deferred_supervision
        else 0
    )
    sft_repetitions = int(training.get("supervision_optimizer_repetitions") or 1)
    sft_positions = unique_sft_positions * sft_repetitions
    sft_planned_steps = (
        math.ceil(
            supervision_planning_rows
            * sft_repetitions
            / int(training["batch_size"])
        )
        if deferred_supervision
        else required_steps(
            supervision_stage.mask,
            int(training["batch_size"]),
            sft_positions,
        )
        if sft_positions
        else 0
    )
    unique_source_positions = (
        int(source_conditioned_stage.mask.sum())
        if source_conditioned_stage is not None
        and not deferred_source_conditioned
        else 0
    )
    source_repetitions = int(training.get("source_conditioned_optimizer_repetitions") or 1)
    source_positions = unique_source_positions * source_repetitions
    source_planned_steps = (
        math.ceil(
            source_planning_rows
            * source_repetitions
            / int(training["batch_size"])
        )
        if deferred_source_conditioned
        else required_steps(
            source_conditioned_stage.mask,
            int(training["batch_size"]),
            source_positions,
        )
        if source_positions
        else 0
    )
    unique_kernel_positions = (
        int(kernel_english_stage.mask.sum())
        if kernel_english_stage is not None
        else 0
    )
    kernel_repetitions = int(training.get("kernel_english_optimizer_repetitions") or 1)
    kernel_positions = unique_kernel_positions * kernel_repetitions
    kernel_batch_size = int(
        candidate_execution_policy.get("batch_size")
        or (config.get("kernel_english_training") or {}).get("batch_size")
        or training["batch_size"]
    )
    kernel_planned_steps = (
        required_steps(
            kernel_english_stage.mask,
            kernel_batch_size,
            kernel_positions,
        )
        if kernel_positions
        else 0
    )
    schedule_training = dict(training)
    if candidate_continuation:
        if (
            candidate_continuation.get("policy")
            != "project_theseus_exact_candidate_continuation_import_v1"
        ):
            raise ValueError("candidate continuation import policy mismatch")
        schedule_training.update(
            {
                "learning_rate": float(
                    candidate_execution_policy["continuation_learning_rate"]
                ),
                "min_learning_rate": float(
                    candidate_execution_policy["continuation_min_learning_rate"]
                ),
                "warmup_steps": int(
                    candidate_execution_policy["continuation_warmup_steps"]
                ),
            }
        )
    schedule_override_keys = (
        "learning_rate",
        "min_learning_rate",
        "warmup_steps",
    )
    if (
        not candidate_continuation
        and any(key in candidate_execution_policy for key in schedule_override_keys)
    ):
        if not all(key in candidate_execution_policy for key in schedule_override_keys):
            raise ValueError(
                "candidate optimizer schedule override requires learning rate, floor, and warmup"
            )
        schedule_training.update(
            {
                "learning_rate": float(candidate_execution_policy["learning_rate"]),
                "min_learning_rate": float(
                    candidate_execution_policy["min_learning_rate"]
                ),
                "warmup_steps": int(candidate_execution_policy["warmup_steps"]),
            }
        )
    if (
        schedule_training["learning_rate"] <= 0
        or schedule_training["min_learning_rate"] <= 0
        or schedule_training["min_learning_rate"]
        > schedule_training["learning_rate"]
        or schedule_training["warmup_steps"] < 0
    ):
        raise ValueError("candidate optimizer schedule override is invalid")
    schedule = build_schedule(
        optim,
        mx,
        schedule_training,
        planned_steps
        + source_planned_steps
        + kernel_planned_steps
        + sft_planned_steps
        + 128,
    )
    selected_optimizer_id = str(
        candidate_execution_policy.get("optimizer_id")
        or optimizer_id
        or training.get("optimizer_id")
        or "adamw_mlx"
    )
    optimizer_learning_rate = (
        float(schedule_training["learning_rate"])
        if selected_optimizer_id in {"muon_mlx", "schedule_free_adamw_mlx"}
        else schedule
    )
    optimizer = pretraining_optimizers.build_optimizer(
        selected_optimizer_id,
        learning_rate=optimizer_learning_rate,
        weight_decay=float(training["weight_decay"]),
        warmup_steps=int(schedule_training.get("warmup_steps") or 0),
        optim=optim,
        mx=mx,
    )
    # Candidate-specific optimizer mechanics use the stable eager route.
    # Ordinary campaign targets use the phase-specific execution policy that
    # was qualified against the immutable trunk.
    optimizer_training_step_mode = (
        "eager"
        if optimizer_id or candidate_execution_policy.get("optimizer_id")
        else str(
            (
                production_execution_policy.get("pretraining") or {}
            ).get("training_step_mode")
            or "auto"
        )
    )
    prior_steps = 0
    prior_pretrain_positions = 0
    prior_source_positions = 0
    prior_kernel_positions = 0
    prior_sft_positions = 0
    prior_checkpoint_hash = ""
    resumed = False
    prior_receipt: dict[str, Any] = {}
    mlx_rng_restored = False
    resume_plan_identity_migration: dict[str, Any] | None = None
    selective_compute_checkpoint: dict[str, Any] | None = None
    if resume and not receipt_path.is_file():
        orphaned_state = [
            relative(path)
            for path in (checkpoint, optimizer_path)
            if path.is_file()
        ]
        if orphaned_state:
            raise ValueError(
                "resume receipt missing for existing campaign state: "
                + ", ".join(orphaned_state)
            )
    if resume and receipt_path.is_file():
        prior = read_json(receipt_path)
        prior_receipt = prior
        resume_checkpoint = resolve(str(prior.get("checkpoint") or checkpoint))
        resume_optimizer = resolve(
            str(prior.get("optimizer_state") or optimizer_path)
        )
        resume_plan_identity_migration = validate_resume(
            prior,
            plan,
            target,
            resume_checkpoint,
            resume_optimizer,
        )
        resume_model = master_model if master_model is not None else model
        if selective_fp32_trainables:
            prior_selective = dict(
                prior.get("selective_compute_checkpoint") or {}
            )
            segmented_delta_resume = kerc_segmented_delta_resume_required(
                prior
            )
            source_backing_checkpoint = (
                resolve(str(prior_selective["source_checkpoint"]))
                if segmented_delta_resume
                else resume_checkpoint
            )
            selective_compute_checkpoint = (
                load_kerc_stage_selective_compute_checkpoint(
                    resume_model,
                    source_backing_checkpoint,
                    stage_index=int(kerc_stage_only or -1),
                    include_stage_embedding=kerc_stage_train_stage_embedding,
                    mx=mx,
                    mlx_utils=mlx_utils,
                )
            )
            if segmented_delta_resume:
                selective_compute_checkpoint["resume_delta_overlay"] = (
                    apply_kerc_stage_selective_delta_checkpoint(
                        resume_model,
                        resume_checkpoint,
                        mx=mx,
                        mlx_utils=mlx_utils,
                    )
                )
                selective_compute_checkpoint[
                    "selected_loaded_raw_tensor_sha256"
                ] = selective_compute_checkpoint["resume_delta_overlay"][
                    "raw_tensor_sha256"
                ]
                selective_compute_checkpoint[
                    "selected_loaded_manifest_sha256"
                ] = hashlib.sha256(
                    json.dumps(
                        selective_compute_checkpoint[
                            "selected_loaded_raw_tensor_sha256"
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            prior["selective_compute_checkpoint"] = (
                selective_compute_checkpoint
            )
            prior["selective_compute_checkpoint_state"] = (
                "LOADED_AND_VERIFIED_BEFORE_OPTIMIZER_STEP"
            )
            write_json_atomic(receipt_path, prior)
        else:
            resume_model.load_weights(
                str(resume_checkpoint), strict=not expert_mode
            )
        if master_model is not None:
            model.update(
                mlx_utils.tree_map(
                    lambda value: value.astype(mx.bfloat16),
                    master_model.trainable_parameters(),
                )
            )
        flat_optimizer_state = mx.load(str(resume_optimizer))
        optimizer_state_projection = prior.get("optimizer_state_projection")
        if optimizer_state_projection is not None:
            trainable_names = {
                name
                for name, _value in mlx_utils.tree_flatten(
                    authoritative_model.trainable_parameters()
                )
            }
            expected_optimizer_names = {"learning_rate", "step"} | {
                f"{name}.{moment}"
                for name in trainable_names
                for moment in ("m", "v")
            }
            observed_optimizer_names = set(flat_optimizer_state)
            if (
                optimizer_state_projection.get("policy")
                != "project_theseus_exact_kerc_stage_optimizer_projection_v1"
                or kerc_stage_only != 1
                or candidate_execution_policy.get(
                    "continuation_optimizer_state_projection_policy"
                )
                != "project_theseus_exact_kerc_stage_optimizer_projection_v1"
                or str(optimizer_state_projection.get(
                    "projected_optimizer_state_sha256"
                ) or "")
                != sha256_file(resume_optimizer)
                or int(
                    optimizer_state_projection.get("selected_parameter_count")
                    or 0
                )
                != len(trainable_names)
                or observed_optimizer_names != expected_optimizer_names
            ):
                raise ValueError(
                    "resume denied: projected optimizer state does not exactly "
                    "match the compiler trainable scope"
                )
        optimizer.state = mlx_utils.tree_unflatten(
            list(flat_optimizer_state.items())
        )
        if hasattr(optimizer, "set_training_iterate"):
            optimizer.set_training_iterate(
                master_model if master_model is not None else model
            )
        prior_rng_path = resolve(str(prior.get("mlx_rng_state") or ""))
        if prior.get("mlx_rng_state"):
            if (
                not prior_rng_path.is_file()
                or sha256_file(prior_rng_path)
                != str(prior.get("mlx_rng_state_sha256") or "")
            ):
                raise ValueError("resume denied: mlx_rng_state_identity_mismatch")
            flat_rng = mx.load(str(prior_rng_path))
            expected_names = [f"state.{index}" for index in range(len(mx.random.state))]
            if sorted(flat_rng) != expected_names:
                raise ValueError("resume denied: mlx_rng_state_shape_mismatch")
            mx.random.state = [flat_rng[name] for name in expected_names]
            mlx_rng_restored = True
        mx.eval(
            model.parameters(),
            (
                master_model.parameters()
                if master_model is not None
                else model.parameters()
            ),
            optimizer.state,
        )
        prior_steps = int(prior.get("optimizer_steps") or 0)
        prior_pretrain_positions = int(prior.get("pretrain_optimizer_positions") or 0)
        prior_source_positions = int(
            prior.get("source_conditioned_optimizer_positions") or 0
        )
        prior_kernel_positions = int(
            prior.get("kernel_english_optimizer_positions") or 0
        )
        prior_sft_positions = int(prior.get("supervision_optimizer_positions") or 0)
        prior_checkpoint_hash = sha256_file(resume_checkpoint)
        resumed = True
    continuation_source_kernel_positions = 0
    kernel_phase_prior_positions = prior_kernel_positions
    if candidate_continuation:
        reset_phase_position_accounting = str(
            candidate_execution_policy.get(
                "continuation_reset_phase_position_accounting"
            )
            or ""
        )
        if reset_phase_position_accounting != "kernel_english":
            raise ValueError(
                "candidate continuation must explicitly reset only kernel "
                "phase-position accounting"
            )
        continuation_segment_resume = bool(
            prior_receipt.get(
                "current_kernel_phase_position_accounting_reset"
            )
        )
        continuation_source_kernel_positions = int(
            prior_receipt.get(
                "continuation_source_kernel_english_optimizer_positions",
                prior_kernel_positions,
            )
            or prior_kernel_positions
        )
        kernel_phase_prior_positions = (
            int(
                prior_receipt.get(
                    "current_kernel_phase_optimizer_positions", 0
                )
                or 0
            )
            if continuation_segment_resume
            else 0
        )
    else:
        continuation_segment_resume = False
    if deferred_source_conditioned:
        prior_unique_source_positions = int(
            prior_receipt.get(
                "unique_source_conditioned_target_positions", 0
            )
            or 0
        )
        if prior_unique_source_positions:
            unique_source_positions = prior_unique_source_positions
            source_positions = (
                unique_source_positions * source_repetitions
            )
    if deferred_supervision:
        prior_unique_sft_positions = int(
            prior_receipt.get("unique_supervision_target_positions", 0)
            or 0
        )
        if prior_unique_sft_positions:
            unique_sft_positions = prior_unique_sft_positions
            sft_positions = unique_sft_positions * sft_repetitions
    remaining_positions = (
        max(0, optimizer_target_positions - prior_pretrain_positions)
        if "pretraining" in active_phases
        else 0
    )
    remaining_sft_positions = (
        max(0, sft_positions - prior_sft_positions)
        if "supervision" in active_phases
        else 0
    )
    remaining_source_positions = (
        max(0, source_positions - prior_source_positions)
        if "source_conditioned_pretraining" in active_phases
        else 0
    )
    remaining_kernel_positions = (
        max(0, kernel_positions - kernel_phase_prior_positions)
        if "kernel_english" in active_phases
        else 0
    )
    allowed_steps = (
        max_steps
        if max_steps
        else planned_steps
        + source_planned_steps
        + kernel_planned_steps
        + sft_planned_steps
        + 128
    )
    temporary_checkpoint = checkpoint.with_name(
        checkpoint.stem + ".partial" + checkpoint.suffix
    )
    heartbeat = checkpoint.parent / "training_heartbeat.json"
    started = time.perf_counter()
    completed_positions = {
        "pretrain": prior_pretrain_positions,
        "source": prior_source_positions,
        "kernel": prior_kernel_positions,
        "supervision": prior_sft_positions,
    }

    def phase_resume_state(
        phase_key: str, default_seed: int
    ) -> tuple[int, dict[str, Any] | None]:
        return resume_phase_data_state(
            prior_receipt,
            resume_plan_identity_migration,
            target_id=target_id,
            phase_key=phase_key,
            default_seed=default_seed,
        )

    def commit_progress_checkpoint(progress: dict[str, Any]) -> None:
        phase = str(progress["phase"])
        positions = dict(completed_positions)
        if "kernel_english" in phase:
            positions["kernel"] = prior_kernel_positions + int(
                progress["target_positions_consumed"]
            )
        elif "source_conditioned_pretraining" in phase:
            positions["source"] = prior_source_positions + int(
                progress["target_positions_consumed"]
            )
        elif "supervision" in phase:
            positions["supervision"] = prior_sft_positions + int(
                progress["target_positions_consumed"]
            )
        else:
            positions["pretrain"] = prior_pretrain_positions + int(
                progress["target_positions_consumed"]
            )
        global_step = int(progress["global_step"])
        generation_checkpoint, generation_optimizer = checkpoint_generation_paths(
            checkpoint,
            optimizer_path,
            global_step,
        )
        generation_rng = rng_state_path(generation_optimizer)
        previous = read_json(receipt_path) if receipt_path.is_file() else {}
        publication = publish_checkpoint_pair(
            authoritative_model,
            generation_checkpoint,
            generation_checkpoint.with_name(
                generation_checkpoint.stem + ".partial" + generation_checkpoint.suffix
            ),
            optimizer,
            generation_optimizer,
            mx=mx,
            mlx_utils=mlx_utils,
            trainable_only=expert_mode or selective_fp32_trainables,
            rng_path=generation_rng,
        )
        progress_receipt = {
            "policy": "project_theseus_moecot_language_arm_training_receipt_v1",
            "created_utc": now(),
            "trigger_state": "GREEN",
            "target_id": target_id,
            "role": target["role"],
            "optimizer_id": selected_optimizer_id,
            "candidate_seed": int(candidate_seed),
            "effective_training_seed": int(effective_seed),
            "candidate_initialization": candidate_initialization_receipt,
            "candidate_execution_policy": candidate_execution_policy,
            "selective_compute_checkpoint": selective_compute_checkpoint,
            "optimizer_state_kind": pretraining_optimizers.optimizer_state_kind(
                optimizer
            ),
            "plan_sha256": plan["plan_sha256"],
            "stage_signature": plan["stage"]["stage_signature"],
            "stage_metadata_sha256": plan["stage"]["metadata_sha256"],
            "row_ranges": target["row_ranges"],
            "parameter_count": observed_parameters,
            "vocab_size": trained_vocab_size,
            "kernel_code_vocabulary_sha256": str(
                (((target.get("kernel_code_vocabulary") or {}).get("payload") or {}).get(
                    "contract_sha256"
                )
                or "")
            ),
            "checkpoint_schema_policy": str(target.get("checkpoint_schema_policy") or ""),
            "checkpoint_schema": str(target.get("checkpoint_schema") or ""),
            "checkpoint_schema_version": int(target.get("checkpoint_schema_version") or 0),
            "trainable_parameter_count": trainable_parameters,
            "expert_trainable_scope": expert_scope if expert_mode else "",
            "shared_trunk_checkpoint": (
                relative(shared_trunk_checkpoint) if expert_mode else ""
            ),
            "shared_trunk_checkpoint_sha256": shared_trunk_checkpoint_sha256,
            "optimizer_steps": global_step,
            "optimizer_positions": sum(positions.values()),
            "pretrain_optimizer_positions": positions["pretrain"],
            "source_conditioned_optimizer_positions": positions["source"],
            "kernel_english_optimizer_positions": positions["kernel"],
            "continuation_source_kernel_english_optimizer_positions": (
                continuation_source_kernel_positions
                if candidate_continuation
                else 0
            ),
            "current_kernel_phase_optimizer_positions": (
                kernel_phase_prior_positions
                + int(progress["target_positions_consumed"])
                if "kernel_english" in phase
                else kernel_phase_prior_positions
            ),
            "current_kernel_phase_position_accounting_reset": bool(
                candidate_continuation
            ),
            "supervision_optimizer_positions": positions["supervision"],
            "unique_target_positions": int(target["unique_target_positions"]),
            "optimizer_target_positions": optimizer_target_positions,
            "checkpoint": relative(generation_checkpoint),
            "checkpoint_sha256": publication["checkpoint_sha256"],
            "optimizer_state": relative(generation_optimizer),
            "optimizer_state_sha256": publication["optimizer_state_sha256"],
            "mlx_rng_state": relative(generation_rng),
            "mlx_rng_state_sha256": publication["mlx_rng_state_sha256"],
            "checkpoint_publication": publication,
            "checkpoint_representation": (
                "kerc_compiler_fp32_delta_over_content_bound_source_v1"
                if selective_fp32_trainables
                else "language_expert_trainable_delta_v1"
                if expert_mode
                else "full_model_v1"
            ),
            "complete": False,
            "transactional_progress": progress,
            "resume_base_checkpoint_sha256": prior_checkpoint_hash,
            "resume_plan_identity_migration": resume_plan_identity_migration,
            "capability_claim": "NOT_EVALUATED",
            "hard_gaps": [],
            **no_cheat(config),
        }
        write_json_atomic(receipt_path, progress_receipt)
        cleanup_progress_generation(
            previous,
            canonical_checkpoint=checkpoint,
            canonical_optimizer=optimizer_path,
            canonical_rng=rng_state_path(optimizer_path),
            keep={generation_checkpoint, generation_optimizer, generation_rng},
            preserve=bool(
                candidate_execution_policy.get(
                    "retain_segment_checkpoint_generations", False
                )
            ),
        )

    random.seed(effective_seed + prior_steps)
    if not mlx_rng_restored:
        mx.random.seed(effective_seed + prior_steps)
    objective_gradient_checkpointing_active = bool(
        candidate_execution_policy.get("objective_gradient_checkpointing", False)
    )
    objective_gradient_decomposition_active = bool(
        candidate_execution_policy.get("objective_gradient_decomposition", False)
    )
    if (
        objective_gradient_decomposition_active
        and not objective_gradient_checkpointing_active
    ):
        raise ValueError(
            "objective gradient decomposition requires complete-objective checkpointing"
        )
    configured_token_loss_position_chunk_size = int(
        candidate_execution_policy.get("token_loss_position_chunk_size", 0)
    )
    token_loss_position_chunk_size = (
        configured_token_loss_position_chunk_size
        if objective_gradient_decomposition_active
        else 0
    )
    if (
        configured_token_loss_position_chunk_size
        and not objective_gradient_decomposition_active
    ):
        raise ValueError(
            "token-loss position chunking requires objective gradient decomposition"
        )
    base_causal_loss = (
        partial(causal_loss, token_loss_compute_dtype="float32")
        if (
            compute_dtype_name == "bfloat16"
            and str(
                compute_execution_policy.get("token_loss_compute_dtype")
                or "model"
            )
            == "float32"
        )
        else causal_loss
    )
    loss_and_grad = (
        partial(
            decomposed_checkpointed_causal_loss_and_grad,
            token_loss_position_chunk_size=token_loss_position_chunk_size,
        )
        if objective_gradient_decomposition_active
        else nn.value_and_grad(
            model,
            checkpointed_causal_loss
            if objective_gradient_checkpointing_active
            else base_causal_loss,
        )
    )
    pretraining_execution = dict(
        production_execution_policy.get("pretraining") or {}
    )
    source_execution = dict(
        production_execution_policy.get(
            "source_conditioned_pretraining"
        )
        or {}
    )
    supervision_execution = dict(
        production_execution_policy.get("supervision") or {}
    )
    phase_boundary_cache_releases: dict[str, Any] = {}

    def release_phase_boundary_cache(boundary: str) -> None:
        release_started = time.perf_counter()
        mx.synchronize()
        cache_before = (
            int(mx.get_cache_memory())
            if hasattr(mx, "get_cache_memory")
            else None
        )
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        cache_after = (
            int(mx.get_cache_memory())
            if hasattr(mx, "get_cache_memory")
            else None
        )
        import gc

        gc.collect()
        phase_boundary_cache_releases[boundary] = {
            "policy": "optimizer_safe_synchronized_mlx_cache_release_v1",
            "seconds": round(time.perf_counter() - release_started, 6),
            "cache_memory_bytes_before": cache_before,
            "cache_memory_bytes_after": cache_after,
        }

    pretrain_seed, pretrain_cursor = phase_resume_state(
        "pretraining", effective_seed + prior_steps
    )
    pretrain_allowed_steps = bounded_phase_steps(
        "pretraining", allowed_steps
    )
    pretrain_phase = train_phase(
        model,
        optimizer,
        loss_and_grad,
        inputs,
        labels,
        mask,
        progress_mask=mask,
        ordered_plan_loss_weight=1.0,
        sample_weights=None,
        plan_labels=None,
        plan_label_mode="none",
        plan_auxiliary_weight=0.0,
        plan_shuffle_seed=0,
        plan_loss_mode="binary_multilabel",
        plan_slot_count=0,
        plan_factor_group_sizes=(),
        phase_name=f"moecot_pretraining:{target_id}",
        target_positions=remaining_positions,
        batch_size=int(training["batch_size"]),
        gradient_clip=float(training["gradient_clip_norm"]),
        seed=pretrain_seed,
        resume_data_cursor=pretrain_cursor,
        max_steps=pretrain_allowed_steps,
        checkpoint=temporary_checkpoint,
        checkpoint_every=max(1, int(training["checkpoint_every_steps"])),
        heartbeat=heartbeat,
        global_step_offset=prior_steps,
        heartbeat_position_offset=prior_pretrain_positions,
        heartbeat_position_target_total=optimizer_target_positions,
        mx=mx,
        optim=optim,
        checkpoint_callback=commit_progress_checkpoint,
        source_conditioning=False,
        step_boundary_callback=step_boundary_callback,
        training_step_mode=str(
            pretraining_execution.get("training_step_mode")
            or optimizer_training_step_mode
        ),
        compiled_microbatch_size=int(
            pretraining_execution.get("compiled_microbatch_size") or 4
        ),
        compile_width_quantum=int(
            pretraining_execution.get("compile_width_quantum") or 64
        ),
        materialize_compiled_state_after_update=bool(
            pretraining_execution.get(
                "materialize_compiled_state_after_update", False
            )
        ),
        eager_gradient_accumulation_microbatch_size=int(
            pretraining_execution.get(
                "eager_gradient_accumulation_microbatch_size"
            )
            or 0
        ),
        master_model=master_model,
        compute_dtype_name=compute_dtype_name,
    )
    completed_positions["pretrain"] = prior_pretrain_positions + int(
        pretrain_phase["target_positions_consumed"]
    )
    used_steps = int(pretrain_phase["optimizer_steps"])
    if (
        "source_conditioned_pretraining" in active_phases
        and used_steps < allowed_steps
    ):
        release_phase_boundary_cache(
            "pretraining_to_source_conditioned_pretraining"
        )
    auxiliary_stage_materialization: dict[str, Any] = {}
    source_conditioned_stage_receipt = (
        source_conditioned_stage.receipt
        if source_conditioned_stage is not None
        and not deferred_source_conditioned
        else None
    )
    supervision_stage_receipt = (
        supervision_stage.receipt
        if supervision_stage is not None and not deferred_supervision
        else None
    )
    if (
        deferred_source_conditioned
        and "source_conditioned_pretraining" in active_phases
        and used_steps < allowed_steps
        and bounded_phase_steps(
            "source_conditioned_pretraining", allowed_steps - used_steps
        )
        > 0
        and (remaining_source_positions > 0 or source_positions == 0)
    ):
        materialize_started = time.perf_counter()
        source_conditioned_stage = source_conditioned_stage.materialize()
        auxiliary_stage_materialization[
            "source_conditioned_pretraining"
        ] = {
            "policy": "deferred_until_phase_boundary_v1",
            "seconds": round(time.perf_counter() - materialize_started, 6),
            "row_count": len(source_conditioned_stage.inputs),
            "physical_array_bytes": int(
                source_conditioned_stage.receipt.get(
                    "physical_array_bytes", 0
                )
                or 0
            ),
        }
        source_conditioned_stage_receipt = (
            source_conditioned_stage.receipt
        )
        unique_source_positions = int(
            source_conditioned_stage.mask.sum()
        )
        source_positions = unique_source_positions * source_repetitions
        remaining_source_positions = max(
            0, source_positions - prior_source_positions
        )
    source_conditioned_phase = {
        "phase": f"moecot_source_conditioned_pretraining:{target_id}",
        "optimizer_steps": 0,
        "target_positions_consumed": 0,
        "target_positions_requested": remaining_source_positions,
        "mean_loss": None,
        "final_loss": None,
    }
    if (
        source_conditioned_stage is not None
        and remaining_source_positions > 0
        and used_steps < allowed_steps
    ):
        source_seed, source_cursor = phase_resume_state(
            "source_conditioned_pretraining",
            effective_seed + prior_steps + used_steps,
        )
        source_conditioned_phase = train_phase(
            model,
            optimizer,
            loss_and_grad,
            source_conditioned_stage.inputs,
            source_conditioned_stage.labels,
            source_conditioned_stage.loss_mask,
            progress_mask=source_conditioned_stage.mask,
            ordered_plan_loss_weight=1.0,
            sample_weights=None,
            plan_labels=None,
            plan_label_mode="none",
            plan_auxiliary_weight=0.0,
            plan_shuffle_seed=0,
            plan_loss_mode="binary_multilabel",
            plan_slot_count=0,
            plan_factor_group_sizes=(),
            phase_name=f"moecot_source_conditioned_pretraining:{target_id}",
            target_positions=remaining_source_positions,
            batch_size=int(training["batch_size"]),
            gradient_clip=float(training["gradient_clip_norm"]),
            seed=source_seed,
            resume_data_cursor=source_cursor,
            max_steps=bounded_phase_steps(
                "source_conditioned_pretraining",
                allowed_steps - used_steps,
            ),
            checkpoint=temporary_checkpoint,
            checkpoint_every=max(1, int(training["checkpoint_every_steps"])),
            heartbeat=heartbeat,
            global_step_offset=prior_steps + used_steps,
            heartbeat_position_offset=prior_source_positions,
            heartbeat_position_target_total=source_positions,
            mx=mx,
            optim=optim,
            checkpoint_callback=commit_progress_checkpoint,
            step_boundary_callback=step_boundary_callback,
            source_to_target_lookup=copy_lookup,
            training_step_mode=str(
                source_execution.get("training_step_mode")
                or optimizer_training_step_mode
            ),
            compiled_microbatch_size=int(
                source_execution.get("compiled_microbatch_size") or 4
            ),
            compile_width_quantum=int(
                source_execution.get("compile_width_quantum") or 64
            ),
            eager_gradient_accumulation_microbatch_size=int(
                source_execution.get(
                    "eager_gradient_accumulation_microbatch_size"
                )
                or 0
            ),
            master_model=master_model,
            compute_dtype_name=compute_dtype_name,
        )
        used_steps += int(source_conditioned_phase["optimizer_steps"])
        completed_positions["source"] = prior_source_positions + int(
            source_conditioned_phase["target_positions_consumed"]
        )
    if deferred_source_conditioned and not isinstance(
        source_conditioned_stage, DeferredSupervisionStage
    ):
        source_conditioned_stage = None
        release_phase_boundary_cache(
            "source_conditioned_pretraining_to_next_phase"
        )
        auxiliary_stage_materialization[
            "source_conditioned_pretraining"
        ]["released_before_next_phase"] = True
    kernel_english_phase = {
        "phase": f"moecot_kernel_english:{target_id}",
        "optimizer_steps": 0,
        "target_positions_consumed": 0,
        "target_positions_requested": remaining_kernel_positions,
        "mean_loss": None,
        "final_loss": None,
    }
    unit_allocator_rows_present = False
    unit_allocator_active = False
    unit_allocator_authority = kerc_unit_allocator_training_authority(config)
    if (
        kernel_english_stage is not None
        and remaining_kernel_positions > 0
        and used_steps < allowed_steps
    ):
        kernel_seed, kernel_cursor = phase_resume_state(
            "kernel_english",
            effective_seed + prior_steps + used_steps,
        )
        kerc_target = str(target.get("role") or "") == "kerc_english_candidate"
        generator_only_overfit = bool(
            ((kernel_english_stage.receipt or {}).get("overfit_diagnostic") or {}).get(
                "active"
            )
        )
        if generator_only_overfit and candidate_execution_policy.get(
            "kerc_overfit_generator_rows_only"
        ) is not True:
            raise ValueError("KERC overfit stage requires generator-only candidate authority")
        kerc_target_only_overfit_loss = bool(
            candidate_execution_policy.get("kerc_overfit_target_only_loss", False)
        )
        if kerc_target_only_overfit_loss and not generator_only_overfit:
            raise ValueError("KERC target-only loss requires the generator overfit stage")
        objective_balanced_full_batch = bool(
            candidate_execution_policy.get("objective_balanced_full_batch", False)
        )
        if objective_balanced_full_batch and (
            not generator_only_overfit
            or kernel_batch_size != len(kernel_english_stage.inputs)
            or len(kernel_english_stage.inputs) != 4
        ):
            raise ValueError(
                "objective-balanced KERC overfit requires one four-objective full batch"
            )
        kerc_single_objective_warmup_steps = int(
            candidate_execution_policy.get(
                "kerc_overfit_single_objective_warmup_steps", 0
            )
        )
        fixed_objective_value = candidate_execution_policy.get(
            "kerc_overfit_fixed_objective_index"
        )
        kerc_fixed_objective_index = (
            int(fixed_objective_value)
            if fixed_objective_value is not None
            else None
        )
        if kerc_single_objective_warmup_steps and not (
            generator_only_overfit and objective_balanced_full_batch
        ):
            raise ValueError(
                "KERC single-objective warmup requires the four-objective overfit stage"
            )
        if kerc_fixed_objective_index is not None and not (
            generator_only_overfit and objective_balanced_full_batch
        ):
            raise ValueError(
                "KERC fixed-objective training requires the four-objective overfit stage"
            )
        kerc_batch_index_schedule = kerc_overfit_batch_schedule(
            row_count=len(kernel_english_stage.inputs),
            single_objective_warmup_steps=kerc_single_objective_warmup_steps,
            fixed_objective_index=kerc_fixed_objective_index,
        )
        kerc_objective_balancing_active = bool(
            candidate_execution_policy.get("kerc_objective_balanced_sampling", False)
        )
        if kerc_objective_balancing_active:
            if not kerc_target or generator_only_overfit:
                raise ValueError(
                    "KERC objective-balanced sampling requires the adequacy stage"
                )
            kernel_sample_weights, kerc_objective_sampling_receipt = (
                kerc_objective_balanced_sample_weights(
                    kernel_english_stage,
                    uniform_within_objective=bool(
                        candidate_execution_policy.get(
                            "kerc_uniform_within_objective_sampling", False
                        )
                    ),
                    objective_sampling_mass=(
                        candidate_execution_policy.get(
                            "kerc_objective_sampling_mass"
                        )
                        or None
                    ),
                )
            )
        else:
            kernel_sample_weights = getattr(
                kernel_english_stage, "sample_weights", None
            )
            kerc_objective_sampling_receipt = {
                "policy": "project_theseus_kerc_objective_balanced_sampling_v1",
                "active": False,
            }
        kernel_training_indices = np.arange(
            len(kernel_english_stage.inputs), dtype=np.int64
        )
        stage_only_missing_coverage: list[str] = []
        if kerc_stage_only is not None:
            kernel_training_indices = token_supervised_row_indices(
                kernel_english_stage.loss_mask
            )
            if not len(kernel_training_indices):
                raise ValueError(
                    "KERC stage-only training has no token-supervised rows"
                )
            retained_coverage = {
                label
                for index in kernel_training_indices
                for label in kernel_english_stage.kerc_coverage_labels[
                    int(index)
                ]
            }
            required_stage_coverage = set(
                candidate_execution_policy.get(
                    "kerc_stage_required_coverage_labels"
                )
                or ()
            )
            stage_only_missing_coverage = sorted(
                required_stage_coverage - retained_coverage
            )
            if stage_only_missing_coverage and not generator_only_overfit:
                raise ValueError(
                    "KERC stage-only token-supervised rows lose required "
                    "coverage: "
                    + ",".join(stage_only_missing_coverage)
                )
        kernel_training_inputs = kernel_english_stage.inputs[
            kernel_training_indices
        ]
        kernel_training_labels = kernel_english_stage.labels[
            kernel_training_indices
        ]
        kernel_training_loss_mask = (
            kernel_english_stage.mask[kernel_training_indices]
            if kerc_target_only_overfit_loss
            else kernel_english_stage.loss_mask[kernel_training_indices]
        )
        kernel_training_progress_mask = kernel_english_stage.mask[
            kernel_training_indices
        ]
        kernel_training_coverage_labels = tuple(
            kernel_english_stage.kerc_coverage_labels[int(index)]
            for index in kernel_training_indices
        )
        if kernel_sample_weights is not None:
            kernel_sample_weights = np.asarray(kernel_sample_weights)[
                kernel_training_indices
            ]
        unit_allocator_rows = (
            getattr(kernel_english_stage, "kerc_unit_allocator_rows", None)
            if kerc_target
            and not generator_only_overfit
            and kerc_stage_only is None
            else None
        )
        unit_allocator_rows_present = bool(
            unit_allocator_rows
            and any(
                row is not None and bool(np.asarray(row["loss_mask"]).any())
                for row in unit_allocator_rows
            )
        )
        unit_allocator_active = (
            unit_allocator_rows_present
            and unit_allocator_authority["authorized"] is True
        )
        kernel_english_phase = train_phase(
            model,
            optimizer,
            loss_and_grad,
            kernel_training_inputs,
            kernel_training_labels,
            kernel_training_loss_mask,
            progress_mask=kernel_training_progress_mask,
            ordered_plan_loss_weight=1.0,
            sample_weights=(
                None
                if objective_balanced_full_batch
                else kernel_sample_weights
            ),
            plan_labels=None,
            plan_label_mode="none",
            plan_auxiliary_weight=0.0,
            plan_shuffle_seed=0,
            plan_loss_mode="binary_multilabel",
            plan_slot_count=0,
            plan_factor_group_sizes=(),
            kerc_residual_labels=(
                kernel_english_stage.kerc_residual_labels
                if kerc_target
                and not unit_allocator_rows_present
                and not generator_only_overfit
                and kerc_stage_only is None
                else None
            ),
            kerc_residual_weight=(
                float(config["kernel_english_training"]["residual_auxiliary_weight"])
                if kerc_target
                and not unit_allocator_rows_present
                and not generator_only_overfit
                and kerc_stage_only is None
                else 0.0
            ),
            kerc_residual_loss_mask=(
                kernel_english_stage.kerc_residual_loss_mask
                if kerc_target
                and not unit_allocator_rows_present
                and not generator_only_overfit
                and kerc_stage_only is None
                else None
            ),
            kerc_unit_allocator_rows=(
                unit_allocator_rows if unit_allocator_active else None
            ),
            kerc_unit_batch_packer=(
                pack_kerc_unit_allocator_batch if unit_allocator_active else None
            ),
            kerc_unit_residual_weight=(
                float(
                    config["kernel_english_training"][
                        "unit_residual_auxiliary_weight"
                    ]
                )
                if unit_allocator_active
                else 0.0
            ),
            kerc_verifier_labels=(
                kernel_english_stage.kerc_verifier_labels
                if str(target.get("role") or "") == "kerc_english_candidate"
                and not generator_only_overfit
                and kerc_stage_only is None
                else None
            ),
            kerc_verifier_weight=(
                float(config["kernel_english_training"]["verifier_auxiliary_weight"])
                if str(target.get("role") or "") == "kerc_english_candidate"
                and not generator_only_overfit
                and kerc_stage_only is None
                else 0.0
            ),
            kerc_verifier_balance_maximum=float(
                (config.get("kernel_english_training") or {}).get(
                    "verifier_class_balance_maximum", 16.0
                )
            ),
            kerc_verifier_require_both_classes=bool(
                (config.get("kernel_english_training") or {}).get(
                    "verifier_require_both_classes", True
                )
            ),
            kerc_decision_labels=(
                kernel_english_stage.kerc_decision_labels
                if str(target.get("role") or "") == "kerc_english_candidate"
                and not generator_only_overfit
                and kerc_stage_only is None
                else None
            ),
            kerc_decision_weight=(
                float(config["kernel_english_training"]["decision_auxiliary_weight"])
                if str(target.get("role") or "") == "kerc_english_candidate"
                and not generator_only_overfit
                and kerc_stage_only is None
                else 0.0
            ),
            kerc_decision_class_count=len(ANSWER_DISPOSITION_ORDER),
            kerc_decision_balance_maximum=float(
                (config.get("kernel_english_training") or {}).get(
                    "decision_class_balance_maximum", 16.0
                )
            ),
            kerc_decision_require_two_classes=True,
            kerc_decision_loss_mask=(
                kernel_english_stage.kerc_decision_loss_mask
                if str(target.get("role") or "") == "kerc_english_candidate"
                and not generator_only_overfit
                and kerc_stage_only is None
                else None
            ),
            coverage_labels=(
                kernel_training_coverage_labels
                if bounded_kerc_coverage_required(target, kernel_english_stage)
                else None
            ),
            required_coverage_labels=(
                ()
                if generator_only_overfit
                else
                tuple(
                    candidate_execution_policy.get(
                        "kerc_stage_required_coverage_labels"
                    )
                    or ()
                )
                if kerc_stage_only is not None
                else KERC_CANARY_REQUIRED_COVERAGE
                if bounded_kerc_coverage_required(target, kernel_english_stage)
                else ()
            ),
            phase_name=f"moecot_kernel_english:{target_id}",
            target_positions=remaining_kernel_positions,
            batch_size=kernel_batch_size,
            gradient_clip=float(training["gradient_clip_norm"]),
            seed=kernel_seed,
            resume_data_cursor=kernel_cursor,
            max_steps=bounded_phase_steps(
                "kernel_english",
                allowed_steps - used_steps,
            ),
            checkpoint=temporary_checkpoint,
            checkpoint_every=max(1, int(training["checkpoint_every_steps"])),
            heartbeat=heartbeat,
            global_step_offset=prior_steps + used_steps,
            heartbeat_position_offset=kernel_phase_prior_positions,
            heartbeat_position_target_total=kernel_positions,
            mx=mx,
            optim=optim,
            checkpoint_callback=commit_progress_checkpoint,
            step_boundary_callback=step_boundary_callback,
            source_to_target_lookup=copy_lookup,
            training_step_mode=optimizer_training_step_mode,
            clear_device_cache_before_step=bool(
                candidate_execution_policy.get("clear_mlx_cache_before_step", False)
            ),
            clear_device_cache_after_backward=bool(
                candidate_execution_policy.get(
                    "clear_mlx_cache_after_backward", False
                )
            ),
            clear_device_cache_after_step=bool(
                candidate_execution_policy.get(
                    "clear_mlx_cache_after_step", False
                )
            ),
            transactional_eager_step=bool(
                candidate_execution_policy.get("transactional_eager_step", False)
            ),
            optimizer_state_offload_path=(
                temporary_checkpoint.with_name(
                    "optimizer_between_step_offload.npz"
                )
                if candidate_execution_policy.get(
                    "optimizer_state_offload_between_steps", False
                )
                else None
            ),
            optimizer_state_offload_minimum_target_positions=int(
                candidate_execution_policy.get(
                    "optimizer_state_offload_minimum_target_positions", 0
                )
            ),
            master_model=master_model,
            compute_dtype_name=compute_dtype_name,
            sequence_balanced_token_loss=bool(
                candidate_execution_policy.get(
                    "sequence_balanced_token_loss", False
                )
            ),
            target_token_frequency_balance_power=float(
                candidate_execution_policy.get(
                    "target_token_frequency_balance_power", 0.0
                )
            ),
            weighted_sampling_minimum_stratum_coverage=bool(
                candidate_execution_policy.get(
                    "kerc_weighted_sampling_minimum_stratum_coverage", False
                )
            ),
            eager_gradient_accumulation_microbatch_size=int(
                candidate_execution_policy.get(
                    "eager_gradient_accumulation_microbatch_size", 0
                )
            ),
            eager_execution_width_quantum=int(
                candidate_execution_policy.get(
                    "compact_partition_width_quantum", 0
                )
            ),
            batch_index_schedule=kerc_batch_index_schedule,
            resource_stress_prefix=bool(
                candidate_execution_policy.get(
                    "kerc_resource_stress_prefix", False
                )
            ),
            prior_coverage_observed_counts=dict(
                (
                    (
                        (prior_receipt.get("phases") or {}).get(
                            "kernel_english"
                        )
                        or {}
                    ).get("coverage_first_sampling")
                    or {}
                ).get("observed_label_counts")
                or {}
            )
            if continuation_segment_resume
            else {},
            allow_incomplete_required_coverage=bool(
                int(
                    candidate_execution_policy.get(
                        "fresh_process_step_segment", 0
                    )
                    or 0
                )
            ),
        )
        kernel_english_phase["kerc_overfit_loss_scope"] = (
            "target_positions_only"
            if kerc_target_only_overfit_loss
            else "canonical_loss_mask"
        )
        kernel_english_phase["objective_gradient_checkpointing_active"] = bool(
            objective_gradient_checkpointing_active
        )
        kernel_english_phase["objective_gradient_decomposition_active"] = bool(
            objective_gradient_decomposition_active
        )
        kernel_english_phase["token_loss_position_chunk_size"] = int(
            token_loss_position_chunk_size
        )
        kernel_english_phase["gradient_checkpointing_active"] = bool(
            gradient_checkpointing_active
        )
        kernel_english_phase["kerc_stage_only"] = kerc_stage_only
        kernel_english_phase["kerc_stage_train_stage_embedding"] = bool(
            kerc_stage_train_stage_embedding
        )
        kernel_english_phase["kerc_stage_detach_frozen_trunk"] = bool(
            kerc_stage_detach_frozen_trunk
        )
        kernel_english_phase[
            "stage_only_token_supervised_row_count"
        ] = int(len(kernel_training_indices))
        kernel_english_phase[
            "stage_only_zero_token_authority_rows_excluded"
        ] = int(
            len(kernel_english_stage.inputs) - len(kernel_training_indices)
        )
        kernel_english_phase["stage_only_missing_coverage_labels"] = (
            stage_only_missing_coverage
        )
        kernel_english_phase[
            "overfit_population_coverage_gate_applied_before_row_selection"
        ] = bool(generator_only_overfit and stage_only_missing_coverage)
        kernel_english_phase["stage_only_auxiliary_losses_disabled"] = bool(
            kerc_stage_only is not None
        )
        kernel_english_phase["kerc_objective_sampling"] = (
            kerc_objective_sampling_receipt
        )
        used_steps += int(kernel_english_phase["optimizer_steps"])
        completed_positions["kernel"] = prior_kernel_positions + int(
            kernel_english_phase["target_positions_consumed"]
        )
    if (
        deferred_supervision
        and "supervision" in active_phases
        and used_steps < allowed_steps
        and bounded_phase_steps("supervision", allowed_steps - used_steps) > 0
        and (remaining_sft_positions > 0 or sft_positions == 0)
    ):
        materialize_started = time.perf_counter()
        supervision_stage = supervision_stage.materialize()
        auxiliary_stage_materialization["supervision"] = {
            "policy": "deferred_until_phase_boundary_v1",
            "seconds": round(time.perf_counter() - materialize_started, 6),
            "row_count": len(supervision_stage.inputs),
            "physical_array_bytes": int(
                supervision_stage.receipt.get("physical_array_bytes", 0)
                or 0
            ),
        }
        supervision_stage_receipt = supervision_stage.receipt
        unique_sft_positions = int(supervision_stage.mask.sum())
        sft_positions = unique_sft_positions * sft_repetitions
        remaining_sft_positions = max(
            0, sft_positions - prior_sft_positions
        )
    supervision_phase = {
        "phase": f"moecot_supervision:{target_id}",
        "optimizer_steps": 0,
        "target_positions_consumed": 0,
        "target_positions_requested": remaining_sft_positions,
        "mean_loss": None,
        "final_loss": None,
    }
    if supervision_stage is not None and remaining_sft_positions > 0 and used_steps < allowed_steps:
        supervision_seed, supervision_cursor = phase_resume_state(
            "supervision",
            effective_seed + prior_steps + used_steps,
        )
        supervision_phase = train_phase(
            model,
            optimizer,
            loss_and_grad,
            supervision_stage.inputs,
            supervision_stage.labels,
            supervision_stage.loss_mask,
            progress_mask=supervision_stage.mask,
            ordered_plan_loss_weight=1.0,
            sample_weights=None,
            plan_labels=None,
            plan_label_mode="none",
            plan_auxiliary_weight=0.0,
            plan_shuffle_seed=0,
            plan_loss_mode="binary_multilabel",
            plan_slot_count=0,
            plan_factor_group_sizes=(),
            phase_name=f"moecot_supervision:{target_id}",
            target_positions=remaining_sft_positions,
            batch_size=int(training["batch_size"]),
            gradient_clip=float(training["gradient_clip_norm"]),
            seed=supervision_seed,
            resume_data_cursor=supervision_cursor,
            max_steps=bounded_phase_steps(
                "supervision",
                allowed_steps - used_steps,
            ),
            checkpoint=temporary_checkpoint,
            checkpoint_every=max(1, int(training["checkpoint_every_steps"])),
            heartbeat=heartbeat,
            global_step_offset=prior_steps + used_steps,
            heartbeat_position_offset=prior_sft_positions,
            heartbeat_position_target_total=sft_positions,
            mx=mx,
            optim=optim,
            checkpoint_callback=commit_progress_checkpoint,
            step_boundary_callback=step_boundary_callback,
            source_to_target_lookup=copy_lookup,
            training_step_mode=str(
                supervision_execution.get("training_step_mode")
                or optimizer_training_step_mode
            ),
            compiled_microbatch_size=int(
                supervision_execution.get("compiled_microbatch_size") or 4
            ),
            compile_width_quantum=int(
                supervision_execution.get("compile_width_quantum") or 64
            ),
            eager_gradient_accumulation_microbatch_size=int(
                supervision_execution.get(
                    "eager_gradient_accumulation_microbatch_size"
                )
                or 0
            ),
            master_model=master_model,
            compute_dtype_name=compute_dtype_name,
        )
    if deferred_supervision and not isinstance(
        supervision_stage, DeferredSupervisionStage
    ):
        supervision_stage = None
        release_phase_boundary_cache(
            "supervision_to_checkpoint_publication"
        )
        auxiliary_stage_materialization[
            "supervision"
        ]["released_before_checkpoint_publication"] = True
    if active_phases == {"pretraining"}:
        release_phase_boundary_cache(
            "pretraining_to_checkpoint_publication"
        )
    total_steps = prior_steps + used_steps + int(supervision_phase["optimizer_steps"])
    final_checkpoint, final_optimizer = checkpoint_generation_paths(
        checkpoint, optimizer_path, total_steps
    )
    final_rng = rng_state_path(final_optimizer)
    publication = publish_checkpoint_pair(
        authoritative_model,
        final_checkpoint,
        final_checkpoint.with_name(
            final_checkpoint.stem + ".partial" + final_checkpoint.suffix
        ),
        optimizer,
        final_optimizer,
        mx=mx,
        mlx_utils=mlx_utils,
        trainable_only=expert_mode or selective_fp32_trainables,
        rng_path=final_rng,
    )
    publish_generation_alias(final_checkpoint, checkpoint)
    publish_generation_alias(final_optimizer, optimizer_path)
    publish_generation_alias(final_rng, rng_state_path(optimizer_path))
    publication["canonical_aliases"] = {
        "checkpoint": relative(checkpoint),
        "optimizer_state": relative(optimizer_path),
        "mlx_rng_state": relative(rng_state_path(optimizer_path)),
        "mechanism": "atomic_hard_link_replace",
    }
    total_pretrain_positions = prior_pretrain_positions + int(
        pretrain_phase["target_positions_consumed"]
    )
    total_sft_positions = prior_sft_positions + int(
        supervision_phase["target_positions_consumed"]
    )
    total_source_positions = prior_source_positions + int(
        source_conditioned_phase["target_positions_consumed"]
    )
    total_kernel_positions = prior_kernel_positions + int(
        kernel_english_phase["target_positions_consumed"]
    )
    kernel_phase_positions = kernel_phase_prior_positions + int(
        kernel_english_phase["target_positions_consumed"]
    )
    total_positions = (
        total_pretrain_positions
        + total_source_positions
        + total_kernel_positions
        + total_sft_positions
    )
    receipt = {
        "policy": "project_theseus_moecot_language_arm_training_receipt_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "target_id": target_id,
        "role": target["role"],
        "optimizer_id": selected_optimizer_id,
        "candidate_seed": int(candidate_seed),
        "effective_training_seed": int(effective_seed),
        "candidate_initialization": candidate_initialization_receipt,
        "candidate_execution_policy": candidate_execution_policy,
        "selective_compute_checkpoint": selective_compute_checkpoint,
        "optimizer_state_kind": pretraining_optimizers.optimizer_state_kind(
            optimizer
        ),
        "plan_sha256": plan["plan_sha256"],
        "stage_signature": plan["stage"]["stage_signature"],
        "stage_metadata_sha256": plan["stage"]["metadata_sha256"],
        "row_ranges": target["row_ranges"],
        "parameter_count": observed_parameters,
        "vocab_size": trained_vocab_size,
        "kernel_code_vocabulary_sha256": str(
            (((target.get("kernel_code_vocabulary") or {}).get("payload") or {}).get(
                "contract_sha256"
            )
            or "")
        ),
        "checkpoint_schema_policy": str(target.get("checkpoint_schema_policy") or ""),
        "checkpoint_schema": str(target.get("checkpoint_schema") or ""),
        "checkpoint_schema_version": int(target.get("checkpoint_schema_version") or 0),
        "trainable_parameter_count": trainable_parameters,
        "expert_trainable_scope": (
            expert_scope if expert_mode else ""
        ),
        "shared_trunk_checkpoint": (
            relative(shared_trunk_checkpoint) if expert_mode else ""
        ),
        "shared_trunk_checkpoint_sha256": shared_trunk_checkpoint_sha256,
        "optimizer_steps": total_steps,
        "optimizer_positions": total_positions,
        "pretrain_optimizer_positions": total_pretrain_positions,
        "source_conditioned_optimizer_positions": total_source_positions,
        "kernel_english_optimizer_positions": total_kernel_positions,
        "continuation_source_kernel_english_optimizer_positions": (
            continuation_source_kernel_positions
            if candidate_continuation
            else 0
        ),
        "current_kernel_phase_optimizer_positions": kernel_phase_positions,
        "current_kernel_phase_position_accounting_reset": bool(
            candidate_continuation
        ),
        "supervision_optimizer_positions": total_sft_positions,
        "unique_target_positions": int(target["unique_target_positions"]),
        "optimizer_target_positions": optimizer_target_positions,
        "optimizer_repetition_factor": optimizer_repetition_factor,
        "unique_source_conditioned_target_positions": unique_source_positions,
        "source_conditioned_optimizer_target_positions": source_positions,
        "source_conditioned_optimizer_repetitions": source_repetitions,
        "unique_kernel_english_target_positions": unique_kernel_positions,
        "kernel_english_optimizer_target_positions": kernel_positions,
        "kernel_english_optimizer_repetitions": kernel_repetitions,
        "kerc_unit_allocator_rows_present": unit_allocator_rows_present,
        "kerc_unit_allocator_training_active": unit_allocator_active,
        "kerc_unit_allocator_training_authority": unit_allocator_authority,
        "unique_supervision_target_positions": unique_sft_positions,
        "supervision_optimizer_target_positions": sft_positions,
        "supervision_optimizer_repetitions": sft_repetitions,
        "complete": (
            total_pretrain_positions >= optimizer_target_positions
            and total_source_positions >= source_positions
            and kernel_phase_positions >= kernel_positions
            and total_sft_positions >= sft_positions
            and not (
                deferred_source_conditioned
                and source_conditioned_stage_receipt is None
                and source_planning_rows > 0
            )
            and not (
                deferred_supervision
                and supervision_stage_receipt is None
                and supervision_planning_rows > 0
            )
        ),
        "checkpoint": relative(final_checkpoint),
        "checkpoint_sha256": publication["checkpoint_sha256"],
        "optimizer_state": relative(final_optimizer),
        "optimizer_state_sha256": publication["optimizer_state_sha256"],
        "mlx_rng_state": relative(final_rng),
        "mlx_rng_state_sha256": publication["mlx_rng_state_sha256"],
        "checkpoint_publication": publication,
        "checkpoint_representation": (
            "kerc_compiler_fp32_delta_over_content_bound_source_v1"
            if selective_fp32_trainables
            else "language_expert_trainable_delta_v1"
            if expert_mode
            else "full_model_v1"
        ),
        "immutable_generation_publication": True,
        "resume_requested": resume,
        "resume": resumed,
        "training_phase_selection": training_phase,
        "bounded_phase_canary": (
            training_phase != "all" or bool(phase_step_limits)
        ),
        "qualification_phase_step_limits": phase_step_limits,
        "auxiliary_stage_residency": {
            "policy": "deferred_until_phase_boundary_then_released_v1",
            "source_conditioned_planning_row_count": source_planning_rows,
            "supervision_planning_row_count": supervision_planning_rows,
            "materialization": auxiliary_stage_materialization,
        },
        "phase_boundary_cache_releases": phase_boundary_cache_releases,
        "resume_base_checkpoint_sha256": prior_checkpoint_hash,
        "resume_plan_identity_migration": resume_plan_identity_migration,
        "phases": {
            "pretraining": pretrain_phase,
            "source_conditioned_pretraining": source_conditioned_phase,
            "kernel_english": kernel_english_phase,
            "supervision": supervision_phase,
        },
        "source_conditioned_stage": (
            source_conditioned_stage_receipt
        ),
        "kernel_english_stage": (
            kernel_english_stage.receipt if kernel_english_stage is not None else None
        ),
        "supervision_stage": (
            supervision_stage_receipt
        ),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "energy_joules": None,
        "energy_measurement_state": "NOT_AVAILABLE_FROM_MLX_RUNTIME",
        "capability_claim": "NOT_EVALUATED",
        "hard_gaps": [],
        **no_cheat(config),
    }
    previous_receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    write_json_atomic(receipt_path, receipt)
    cleanup_progress_generation(
        previous_receipt,
        canonical_checkpoint=checkpoint,
        canonical_optimizer=optimizer_path,
        canonical_rng=rng_state_path(optimizer_path),
        keep={final_checkpoint, final_optimizer, final_rng},
        preserve=bool(
            candidate_execution_policy.get(
                "retain_segment_checkpoint_generations", False
            )
        ),
    )
    return receipt


def canonical_pretraining_execution_stage(
    stage_dir: Path,
    canonical: dict[str, Any],
    *,
    active: bool,
) -> SimpleNamespace:
    """Load canonical pretraining arrays only for an active pretraining phase."""

    if not active:
        return SimpleNamespace(
            pretrain_inputs=np.empty((0, 1), dtype=np.int32),
            pretrain_labels=np.empty((0, 1), dtype=np.int32),
            pretrain_mask=np.empty((0, 1), dtype=np.uint8),
            loaded=False,
            phase_inactive=True,
        )
    shape = (
        int(canonical["window_count"]),
        int(canonical["max_sequence_tokens"]),
    )
    arrays = load_pretrain_memmaps(
        pretrain_array_paths(stage_dir),
        shape,
        expected=canonical["array_artifacts"],
    )
    return SimpleNamespace(
        pretrain_inputs=arrays[0],
        pretrain_labels=arrays[1],
        pretrain_mask=arrays[2],
        loaded=True,
        phase_inactive=False,
    )


def range_view(array: np.ndarray, ranges: list[dict[str, int]]) -> np.ndarray:
    normalized = [(int(row["start"]), int(row["stop"])) for row in ranges]
    if not normalized:
        raise ValueError("training target has no stage ranges")
    if all(normalized[index][1] == normalized[index + 1][0] for index in range(len(normalized) - 1)):
        return array[normalized[0][0] : normalized[-1][1]]
    return np.concatenate([array[start:stop] for start, stop in normalized], axis=0)


def tensor_mapping_manifest(mapping: dict[str, Any]) -> dict[str, Any]:
    """Content-bind tensor names, shapes, dtypes, and bytes across file formats."""

    digest = hashlib.sha256()
    total_elements = 0
    total_bytes = 0
    for name in sorted(mapping):
        array = np.asarray(mapping[name])
        descriptor = json.dumps(
            {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))
        total_elements += int(array.size)
        total_bytes += int(array.nbytes)
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": len(mapping),
        "element_count": total_elements,
        "payload_bytes": total_bytes,
    }


def migrate_shared_trunk_checkpoint_format(
    config: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Advance the shared-trunk receipt only after exact cross-format replay."""

    import mlx.core as mx

    contract = config.get("checkpoint_format_migration") or {}
    if contract.get("policy") != "project_theseus_checkpoint_format_migration_v1":
        raise ValueError("checkpoint format migration contract is missing")
    target = (plan.get("targets") or {}).get(SHARED_TRUNK_ID)
    if not isinstance(target, dict):
        raise ValueError("shared-trunk target is missing from the training plan")
    receipt_path = resolve(str(target["receipt"]))
    if not receipt_path.is_file():
        raise ValueError("shared-trunk receipt is missing")
    receipt = read_json(receipt_path)
    source_checkpoint = resolve(str(receipt.get("checkpoint") or ""))
    target_checkpoint = resolve(str(target["checkpoint"]))
    optimizer = resolve(str(receipt.get("optimizer_state") or target["optimizer_state"]))
    if target_checkpoint.suffix != str(contract.get("target_suffix") or ""):
        raise ValueError("configured shared-trunk checkpoint does not use the target format")

    if source_checkpoint.resolve() == target_checkpoint.resolve():
        validate_resume(receipt, plan, target, target_checkpoint, optimizer)
        migration = receipt.get("checkpoint_format_migration") or {}
        if (
            migration.get("policy") != contract["policy"]
            or migration.get("exact_tensor_parity") is not True
        ):
            raise ValueError("registered target-format receipt lacks exact migration evidence")
        legacy = resolve(str(migration.get("source_checkpoint") or ""))
        legacy_removed = not legacy.exists() if legacy != target_checkpoint else True
        return {
            "policy": contract["policy"],
            "trigger_state": "GREEN",
            "migration_state": "ALREADY_COMMITTED",
            "target_id": SHARED_TRUNK_ID,
            "checkpoint": relative(target_checkpoint),
            "checkpoint_sha256": sha256_file(target_checkpoint),
            "optimizer_state": relative(optimizer),
            "optimizer_state_sha256": sha256_file(optimizer),
            "tensor_manifest": migration.get("tensor_manifest"),
            "legacy_source_removed": legacy_removed,
            "training_positions_added": 0,
            "capability_claim": "NONE",
            "hard_gaps": [] if legacy_removed else ["legacy_source_cleanup_pending"],
        }

    if source_checkpoint.suffix != str(contract.get("source_suffix") or ""):
        raise ValueError("registered shared-trunk checkpoint is not the qualified source format")
    if not source_checkpoint.is_file() or not optimizer.is_file():
        raise ValueError("registered shared-trunk model or optimizer is missing")
    plan_migration = validate_resume(
        receipt, plan, target, source_checkpoint, optimizer
    )

    qualification_path = resolve(str(contract.get("qualification_report") or ""))
    if not qualification_path.is_file():
        raise ValueError("checkpoint format qualification report is missing")
    qualification = read_json(qualification_path)
    storage = qualification.get("checkpoint_storage") or {}
    if (
        storage.get("policy") != "project_theseus_checkpoint_format_qualification_v1"
        or storage.get("state") != "GREEN"
        or storage.get("exact_tensor_parity") is not True
        or storage.get("adoption_recommendation") != "QUALIFIED_FOR_CONTROLLED_MIGRATION"
        or storage.get("source_checkpoint") != relative(source_checkpoint)
        or float(storage.get("safetensors_load_speedup") or 0.0)
        < float(contract.get("minimum_qualified_load_speedup") or 0.0)
    ):
        raise ValueError("checkpoint format qualification does not authorize migration")

    source_sha256 = sha256_file(source_checkpoint)
    source_bytes = source_checkpoint.stat().st_size
    source = mx.load(str(source_checkpoint))
    mx.eval(*source.values())
    source_manifest = tensor_mapping_manifest(source)
    if source_manifest != storage.get("source_tensor_manifest"):
        raise ValueError("live source tensor manifest does not match qualification evidence")

    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_checkpoint.with_name(
        target_checkpoint.stem + ".partial" + target_checkpoint.suffix
    )
    temporary.unlink(missing_ok=True)
    started = time.perf_counter()
    mx.save_safetensors(
        str(temporary),
        source,
        metadata={"policy": "project_theseus_model_checkpoint_v1"},
    )
    converted = mx.load(str(temporary))
    mx.eval(*converted.values())
    converted_manifest = tensor_mapping_manifest(converted)
    if converted_manifest != source_manifest:
        temporary.unlink(missing_ok=True)
        raise ValueError("converted checkpoint failed exact tensor replay")
    os.replace(temporary, target_checkpoint)
    target_sha256 = sha256_file(target_checkpoint)
    target_bytes = target_checkpoint.stat().st_size
    migration = {
        "policy": contract["policy"],
        "created_utc": now(),
        "source_checkpoint": relative(source_checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_bytes": source_bytes,
        "source_format": source_checkpoint.suffix.lstrip("."),
        "target_checkpoint": relative(target_checkpoint),
        "target_checkpoint_sha256": target_sha256,
        "target_checkpoint_bytes": target_bytes,
        "target_format": target_checkpoint.suffix.lstrip("."),
        "tensor_manifest": source_manifest,
        "exact_tensor_parity": True,
        "optimizer_state_unchanged": True,
        "optimizer_state_sha256": sha256_file(optimizer),
        "optimizer_steps": int(receipt.get("optimizer_steps") or 0),
        "optimizer_positions": int(receipt.get("optimizer_positions") or 0),
        "training_positions_added": 0,
        "qualification_report": relative(qualification_path),
        "qualification_report_sha256": sha256_file(qualification_path),
        "qualified_load_speedup": float(storage["safetensors_load_speedup"]),
        "publication_seconds": round(time.perf_counter() - started, 6),
        "atomic_checkpoint_replacement": True,
        "atomic_receipt_replacement": True,
        "capability_claim": "NONE",
    }
    migrated_receipt = copy.deepcopy(receipt)
    migrated_receipt.update(
        {
            "checkpoint": relative(target_checkpoint),
            "checkpoint_sha256": target_sha256,
            "plan_sha256": plan["plan_sha256"],
            "resume_plan_identity_migration": plan_migration,
            "checkpoint_format_migration": migration,
        }
    )
    write_json_atomic(receipt_path, migrated_receipt)
    validate_resume(migrated_receipt, plan, target, target_checkpoint, optimizer)
    source_checkpoint.unlink()
    return {
        "policy": contract["policy"],
        "trigger_state": "GREEN",
        "migration_state": "COMMITTED",
        "target_id": SHARED_TRUNK_ID,
        **migration,
        "legacy_source_removed": not source_checkpoint.exists(),
        "storage_bytes_reclaimed": max(0, source_bytes - target_bytes),
        "hard_gaps": [],
    }


def publish_model(
    model: Any,
    checkpoint: Path,
    temporary: Path,
    *,
    mx: Any,
    mlx_utils: Any,
    trainable_only: bool,
) -> None:
    temporary.unlink(missing_ok=True)
    if trainable_only:
        weights = {
            name: value
            for name, value in mlx_utils.tree_flatten(model.trainable_parameters())
        }
        mx.save_safetensors(
            str(temporary),
            weights,
            metadata={"policy": "moecot_language_expert_delta_v2"},
        )
    else:
        model.save_weights(str(temporary))
    if not temporary.is_file():
        raise ValueError("MLX model checkpoint publication failed")
    os.replace(temporary, checkpoint)


def publish_optimizer(mx: Any, mlx_utils: Any, optimizer: Any, path: Path) -> None:
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    temporary.unlink(missing_ok=True)
    flat = {name: value for name, value in mlx_utils.tree_flatten(optimizer.state)}
    mx.save_safetensors(str(temporary), flat, metadata={"policy": "moecot_optimizer_state_v1"})
    os.replace(temporary, path)


def rng_state_path(optimizer_path: Path) -> Path:
    return optimizer_path.with_name(
        optimizer_path.stem + ".mlx-rng" + optimizer_path.suffix
    )


def publish_mlx_rng_state(mx: Any, path: Path) -> None:
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    temporary.unlink(missing_ok=True)
    mx.save_safetensors(
        str(temporary),
        {f"state.{index}": value for index, value in enumerate(mx.random.state)},
        metadata={"policy": "project_theseus_mlx_rng_state_v1"},
    )
    os.replace(temporary, path)


def publish_generation_alias(generation: Path, alias: Path) -> None:
    """Atomically point a compatibility filename at an immutable generation."""

    temporary = alias.with_name(alias.name + ".alias-partial")
    temporary.unlink(missing_ok=True)
    os.link(generation, temporary)
    os.replace(temporary, alias)


def publish_checkpoint_pair(
    model: Any,
    checkpoint: Path,
    temporary_checkpoint: Path,
    optimizer: Any,
    optimizer_path: Path,
    *,
    mx: Any,
    mlx_utils: Any,
    trainable_only: bool,
    rng_path: Path | None = None,
) -> dict[str, Any]:
    """Publish model and optimizer atomically per file with measured durable costs."""

    started = time.perf_counter()
    evaluation_iterate = hasattr(optimizer, "set_evaluation_iterate")
    if evaluation_iterate:
        optimizer.set_evaluation_iterate(model)
    try:
        publish_model(
            model,
            checkpoint,
            temporary_checkpoint,
            mx=mx,
            mlx_utils=mlx_utils,
            trainable_only=trainable_only,
        )
    finally:
        if evaluation_iterate:
            optimizer.set_training_iterate(model)
    model_seconds = time.perf_counter() - started
    optimizer_started = time.perf_counter()
    publish_optimizer(mx, mlx_utils, optimizer, optimizer_path)
    optimizer_seconds = time.perf_counter() - optimizer_started
    rng_seconds = 0.0
    if rng_path is not None:
        rng_started = time.perf_counter()
        publish_mlx_rng_state(mx, rng_path)
        rng_seconds = time.perf_counter() - rng_started
    hash_started = time.perf_counter()
    checkpoint_sha256 = sha256_file(checkpoint)
    optimizer_state_sha256 = sha256_file(optimizer_path)
    mlx_rng_state_sha256 = sha256_file(rng_path) if rng_path is not None else ""
    hash_seconds = time.perf_counter() - hash_started
    return {
        "policy": "project_theseus_checkpoint_publication_timing_v1",
        "model_serialization_seconds": round(model_seconds, 6),
        "optimizer_serialization_seconds": round(optimizer_seconds, 6),
        "mlx_rng_serialization_seconds": round(rng_seconds, 6),
        "content_hash_seconds": round(hash_seconds, 6),
        "total_seconds": round(time.perf_counter() - started, 6),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "optimizer_state_bytes": optimizer_path.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "optimizer_state_sha256": optimizer_state_sha256,
        "mlx_rng_state_bytes": rng_path.stat().st_size if rng_path is not None else 0,
        "mlx_rng_state_sha256": mlx_rng_state_sha256,
        "atomic_file_replacement": True,
        "published_model_iterate": "evaluation_x" if evaluation_iterate else "training",
        "training_iterate_restored_after_publication": True,
        "background_serialization": False,
    }


def checkpoint_generation_paths(
    checkpoint: Path, optimizer: Path, global_step: int
) -> tuple[Path, Path]:
    if global_step <= 0:
        raise ValueError("checkpoint generation step must be positive")
    suffix = f".step-{global_step:08d}"
    return (
        checkpoint.with_name(checkpoint.stem + suffix + checkpoint.suffix),
        optimizer.with_name(optimizer.stem + suffix + optimizer.suffix),
    )


def cleanup_progress_generation(
    receipt: dict[str, Any],
    *,
    canonical_checkpoint: Path,
    canonical_optimizer: Path,
    canonical_rng: Path | None = None,
    keep: set[Path] | None = None,
    preserve: bool = False,
) -> None:
    """Delete only superseded step generations after a newer receipt commits."""

    if preserve:
        return
    retained = {path.resolve() for path in (keep or set())}
    canonical_rng = canonical_rng or rng_state_path(canonical_optimizer)
    for key, canonical in (
        ("checkpoint", canonical_checkpoint),
        ("optimizer_state", canonical_optimizer),
        ("mlx_rng_state", canonical_rng),
    ):
        value = str(receipt.get(key) or "")
        if not value:
            continue
        candidate = resolve(value)
        if key == "mlx_rng_state":
            prefix = canonical_optimizer.stem + ".step-"
            suffix_matches = candidate.name.endswith(
                ".mlx-rng" + canonical_optimizer.suffix
            )
        else:
            prefix = canonical.stem + ".step-"
            suffix_matches = candidate.suffix == canonical.suffix
        if (
            candidate.resolve() not in retained
            and candidate.parent.resolve() == canonical.parent.resolve()
            and candidate.name.startswith(prefix)
            and suffix_matches
        ):
            candidate.unlink(missing_ok=True)


def resume_phase_data_state(
    prior_receipt: dict[str, Any],
    plan_migration: dict[str, Any] | None,
    *,
    target_id: str,
    phase_key: str,
    default_seed: int,
) -> tuple[int, dict[str, Any] | None]:
    """Resolve an exact phase cursor, including explicit sampler migrations."""

    if (
        plan_migration
        and plan_migration.get("reset_data_cursor_phase") == phase_key
    ):
        return (
            int(plan_migration.get("reset_data_cursor_seed", default_seed)),
            None,
        )
    phase_name = {
        "pretraining": f"moecot_pretraining:{target_id}",
        "source_conditioned_pretraining": (
            f"moecot_source_conditioned_pretraining:{target_id}"
        ),
        "kernel_english": f"moecot_kernel_english:{target_id}",
        "supervision": f"moecot_supervision:{target_id}",
    }[phase_key]
    transactional = prior_receipt.get("transactional_progress") or {}
    phase_receipt = (prior_receipt.get("phases") or {}).get(phase_key) or {}
    cursor = None
    if transactional.get("phase") == phase_name:
        cursor = transactional.get("data_cursor")
    if cursor is None:
        cursor = phase_receipt.get("data_cursor_next")
    if not isinstance(cursor, dict):
        return default_seed, None
    return int(cursor.get("seed", default_seed)), cursor


def validate_resume(
    receipt: dict[str, Any], plan: dict[str, Any], target: dict[str, Any], checkpoint: Path, optimizer: Path
) -> dict[str, Any] | None:
    faults = []
    plan_migration: dict[str, Any] | None = None
    if receipt.get("policy") != "project_theseus_moecot_language_arm_training_receipt_v1":
        faults.append("receipt_policy_mismatch")
    if receipt.get("target_id") != target["target_id"]:
        faults.append("target_identity_mismatch")
    if receipt.get("plan_sha256") != plan["plan_sha256"]:
        plan_migration = accepted_plan_identity_migration(receipt, plan, target)
        if plan_migration is None:
            faults.append("plan_identity_mismatch")
    if receipt.get("stage_signature") != plan["stage"]["stage_signature"]:
        faults.append("stage_identity_mismatch")
    if receipt.get("row_ranges") != target["row_ranges"]:
        faults.append("stage_range_mismatch")
    if target.get("vocab_size") is not None and int(receipt.get("vocab_size") or 0) != int(
        target["vocab_size"]
    ):
        faults.append("vocab_size_mismatch")
    expected_codebook = str(
        (((target.get("kernel_code_vocabulary") or {}).get("payload") or {}).get(
            "contract_sha256"
        )
        or "")
    )
    if expected_codebook and receipt.get("kernel_code_vocabulary_sha256") != expected_codebook:
        faults.append("kernel_code_vocabulary_identity_mismatch")
    if target.get("role") == "kerc_english_candidate":
        if receipt.get("checkpoint_schema_policy") != target.get("checkpoint_schema_policy"):
            faults.append("kerc_checkpoint_schema_policy_mismatch")
        if receipt.get("checkpoint_schema") != target.get("checkpoint_schema"):
            faults.append("kerc_checkpoint_schema_mismatch")
        if int(receipt.get("checkpoint_schema_version") or -1) != int(
            target.get("checkpoint_schema_version") or 0
        ):
            faults.append("kerc_checkpoint_schema_version_mismatch")
    if target.get("role") == "language_expert":
        shared = resolve(str(target.get("shared_trunk_checkpoint") or ""))
        if (
            not shared.is_file()
            or sha256_file(shared)
            != receipt.get("shared_trunk_checkpoint_sha256")
        ):
            faults.append("shared_trunk_checkpoint_identity_mismatch")
    if not checkpoint.is_file() or sha256_file(checkpoint) != receipt.get("checkpoint_sha256"):
        faults.append("checkpoint_identity_mismatch")
    if not optimizer.is_file() or sha256_file(optimizer) != receipt.get("optimizer_state_sha256"):
        faults.append("optimizer_identity_mismatch")
    if faults:
        raise ValueError("resume denied: " + ",".join(faults))
    return plan_migration


def accepted_plan_identity_migration(
    receipt: dict[str, Any], plan: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any] | None:
    contract = plan.get("plan_identity")
    if not isinstance(contract, dict) or contract.get("policy") not in {
        "project_theseus_semantic_training_plan_identity_v2",
        "project_theseus_semantic_training_plan_identity_v3",
    }:
        return None
    for row in contract.get("legacy_migrations") or []:
        if not isinstance(row, dict):
            continue
        if (
            row.get("target_id") == target.get("target_id")
            and row.get("legacy_plan_sha256") == receipt.get("plan_sha256")
            and migration_receipt_identity_matches(row, receipt)
            and row.get("required_current_plan_sha256") == plan.get("plan_sha256")
            and row.get("required_stage_signature") == receipt.get("stage_signature")
            and row.get("required_stage_signature")
            == (plan.get("stage") or {}).get("stage_signature")
        ):
            return {
                "policy": contract["policy"],
                "migration_id": row.get("migration_id"),
                "legacy_plan_sha256": row.get("legacy_plan_sha256"),
                "current_plan_sha256": plan.get("plan_sha256"),
                "legacy_scale_report_sha256": row.get(
                    "legacy_scale_report_sha256"
                ),
                "legacy_checkpoint_sha256": row.get("legacy_checkpoint_sha256"),
                "legacy_optimizer_state_sha256": row.get(
                    "legacy_optimizer_state_sha256"
                ),
                "legacy_optimizer_steps": row.get("legacy_optimizer_steps"),
                "legacy_optimizer_positions": row.get(
                    "legacy_optimizer_positions"
                ),
                "evidence": row.get("evidence"),
                "reason": row.get("reason"),
                "reset_data_cursor_phase": row.get("reset_data_cursor_phase"),
                "reset_data_cursor_seed": row.get("reset_data_cursor_seed"),
            }
    return None


def migration_receipt_identity_matches(
    migration: dict[str, Any], receipt: dict[str, Any]
) -> bool:
    fields = (
        ("legacy_checkpoint_sha256", "checkpoint_sha256"),
        ("legacy_optimizer_state_sha256", "optimizer_state_sha256"),
        ("legacy_optimizer_steps", "optimizer_steps"),
        ("legacy_optimizer_positions", "optimizer_positions"),
    )
    for migration_field, receipt_field in fields:
        expected = migration.get(migration_field)
        if expected is None or expected == receipt.get(receipt_field):
            continue
        if migration_field == "legacy_optimizer_state_sha256":
            projection = receipt.get("optimizer_state_projection") or {}
            if (
                projection.get("policy")
                == "project_theseus_exact_kerc_stage_optimizer_projection_v1"
                and projection.get("source_optimizer_state_sha256") == expected
                and projection.get("projected_optimizer_state_sha256")
                == receipt.get(receipt_field)
                and projection.get("source_tensor_values_mutated") is False
                and projection.get("optimizer_step_reset") is False
                and projection.get("learning_rate_reset") is False
                and projection.get("independent_projection_replay") == "GREEN"
            ):
                continue
        return False
    return True


def evaluation_freeze_semantic_sha256(evaluation: dict[str, Any]) -> str:
    """Bind evaluation behavior while excluding timestamps and state snapshots."""

    semantic_fields = (
        "policy",
        "candidate_id",
        "candidate_packet_sha256",
        "case_contract_sha256",
        "case_count",
        "cases_by_arm",
        "compiler_sha256",
        "case_compiler_sha256",
        "generation_wrapper_sha256",
        "verifier_sha256",
        "local_english_rater_config_sha256",
        "local_english_rater_implementation_sha256",
        "toolchain_identity_sha256",
        "consumption_policy_sha256",
        "consumption_registry",
        "source_disjoint",
        "public_training_rows_written",
        "external_inference_calls",
        "templates_renderers_routers_tools_credit",
    )
    payload = {key: evaluation.get(key) for key in semantic_fields}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def training_implementation_closure(config: dict[str, Any]) -> list[dict[str, str]]:
    contract = config.get("plan_identity") or {}
    paths = contract.get("implementation_closure") or []
    if not isinstance(paths, list) or not paths:
        return []
    rows = []
    seen = set()
    for declared in paths:
        path = resolve(str(declared))
        canonical = relative(path)
        if canonical in seen:
            raise ValueError(f"duplicate training implementation closure path: {canonical}")
        if not path.is_file():
            raise ValueError(f"training implementation closure path is missing: {canonical}")
        seen.add(canonical)
        rows.append({"path": canonical, "sha256": sha256_file(path)})
    return sorted(rows, key=lambda row: row["path"])


def plan_sha256(
    config: dict[str, Any],
    metadata: dict[str, Any],
    models: dict[str, Any],
    supervision: dict[str, Any],
    source_conditioned: dict[str, Any],
    kernel_english: dict[str, Any],
    scale_preregistration: dict[str, Any],
) -> str:
    training_artifacts = {
        key: value
        for key, value in (supervision.get("artifacts") or {}).items()
        if str(key).endswith(":private_train")
    }
    payload = {
        "training_contract": {
            key: config.get(key)
            for key in (
                "policy",
                "seed",
                "topology",
                "shared_trunk_model",
                "arm_model",
                "controls",
                "training",
                "boundaries",
            )
        },
        "plan_identity_policy": (config.get("plan_identity") or {}).get("policy"),
        "training_implementation_closure": training_implementation_closure(config),
        "stage_signature": (metadata.get("summary") or {}).get("stage_signature"),
        "arm_views": ((metadata.get("summary") or {}).get("canonical_pretrain_stage") or {}).get("arm_views"),
        "models": models,
        "supervision_training_artifacts": training_artifacts,
        "source_conditioned_training_artifacts": source_conditioned.get("artifacts")
        or {},
        "kernel_english_training_artifacts": kernel_english.get("artifacts") or {},
        "kernel_english_learned_pipeline_contract": kernel_english.get(
            "learned_pipeline_contract"
        )
        or {},
        "scale_preregistration": {
            key: scale_preregistration.get(key)
            for key in (
                "candidate_id",
                "config_sha256",
                "evaluation_freeze_semantic_sha256",
                "required_unique_positions",
                "staged_unique_positions",
            )
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != "project_theseus_moecot_language_arm_training_v1":
        raise ValueError("unexpected MoECOT training policy")
    training_host_policy(config)
    optimizer_id = str((config.get("training") or {}).get("optimizer_id") or "adamw_mlx")
    if optimizer_id not in pretraining_optimizers.OPTIMIZER_IDS:
        raise ValueError("training optimizer is not registered")
    authority = config.get("architecture_training_authority") or {}
    if authority.get("policy") != "project_theseus_pre_training_architecture_authority_v1":
        raise ValueError("pre-training architecture authority contract is required")
    if authority.get("required_for_long_optimizer_runs") is not True:
        raise ValueError("long optimizer runs must require architecture readiness")
    if int(authority.get("pre_training_canary_max_steps") or 0) != 8:
        raise ValueError("legacy pre-training canary cap must remain recorded at eight steps")
    canary_contract = resolve(str(authority.get("candidate_canary_contract") or ""))
    if (
        not canary_contract.is_file()
        or authority.get("candidate_canary_policy")
        != "project_theseus_pretraining_architecture_candidates_v1"
        or authority.get("generic_canary_authority") != "denied"
    ):
        raise ValueError("candidate-specific architecture canary contract is required")
    loaded_canary_contract = pretraining_candidate_canary.load_contract(canary_contract)
    if loaded_canary_contract.get("policy") != authority.get("candidate_canary_policy"):
        raise ValueError("candidate-specific architecture canary policy mismatch")
    fresh_segments = dict(authority.get("fresh_process_segments") or {})
    if (
        fresh_segments.get("policy")
        != "project_theseus_bounded_fresh_process_pretraining_v1"
        or fresh_segments.get("target_id") != SHARED_TRUNK_ID
        or fresh_segments.get("phase") != "pretraining"
        or int(fresh_segments.get("maximum_optimizer_steps") or 0) != 64
        or fresh_segments.get("compute_dtype") != "float32"
        or fresh_segments.get("fp32_master") is not False
        or int(fresh_segments.get("compiled_microbatch_size") or 0) != 4
        or fresh_segments.get("resume_required") is not True
        or fresh_segments.get("require_external_watchdog") is not True
        or int(
            fresh_segments.get("minimum_qualified_contiguous_segments") or 0
        )
        < 2
        or not str(fresh_segments.get("qualification_report") or "")
    ):
        raise ValueError("fresh-process pretraining segment contract is invalid")
    if [str(value) for value in authority.get("gate_command") or []] != [
        "python3",
        "scripts/roadmap_implementation_gate.py",
        "--gate",
        "--require-pre-training-ready",
    ]:
        raise ValueError("architecture readiness gate command mismatch")
    format_migration = config.get("checkpoint_format_migration") or {}
    if (
        format_migration.get("policy")
        != "project_theseus_checkpoint_format_migration_v1"
        or format_migration.get("source_suffix") != ".npz"
        or format_migration.get("target_suffix") != ".safetensors"
        or float(format_migration.get("minimum_qualified_load_speedup") or 0.0) < 1.2
        or not str(format_migration.get("qualification_report") or "")
    ):
        raise ValueError("checkpoint format migration contract is invalid")
    identity = config.get("plan_identity") or {}
    if identity.get("policy") == "project_theseus_semantic_training_plan_identity_v3":
        closure = training_implementation_closure(config)
        required = {
            "scripts/moecot_language_arm_training.py",
            "scripts/standard_causal_transformer_model.py",
            "scripts/standard_causal_transformer_survival.py",
        }
        observed = {row["path"] for row in closure}
        missing = sorted(required - observed)
        if missing:
            raise ValueError(
                "training implementation closure is incomplete: " + ",".join(missing)
            )
    generation = config.get("generation_architecture") or {}
    contract_path = resolve(str(generation.get("contract") or ""))
    if not contract_path.is_file():
        raise ValueError("generation architecture contract is required")
    generation_contract = read_json(contract_path)
    if (
        generation_contract.get("policy") != generation.get("required_policy")
        or generation_contract.get("first_campaign_base") != generation.get("base_mode")
        or generation.get("checkpoint_shaping_auxiliary") != "mtp"
        or float(generation.get("initial_loss_scale", -1.0)) != 0.0
    ):
        raise ValueError("generation architecture selection does not match its contract")
    mtp = dict((generation_contract.get("modes") or {}).get("mtp") or {})
    shape = dict(generation_contract.get("mtp_shape_contract") or {})
    expected_mtp = {
        "mtp_future_offsets": list(shape.get("future_offsets") or []),
        "mtp_low_rank": int(mtp.get("low_rank") or 0),
        "mtp_loss_weights": list(mtp.get("loss_weights") or []),
        "mtp_loss_scale": 0.0,
        "mtp_maximum_head_parameter_overhead_ratio": float(
            shape.get("maximum_parameter_overhead_ratio") or 0.0
        ),
    }
    for model_id in ("shared_trunk_model", "arm_model", "kerc_english_model"):
        if model_id not in config:
            continue
        model = config.get(model_id) or {}
        if {key: model.get(key) for key in expected_mtp} != expected_mtp:
            raise ValueError(f"{model_id} does not consume the frozen MTP contract")
    comparison = config.get("comparison_contract") or {}
    if comparison.get("preregistered_before_training") is not True:
        raise ValueError("comparison contract must be preregistered")
    topology = config.get("topology") or {}
    if topology.get("policy") not in {
        "project_theseus_moecot_shared_trunk_source_specialists_v2",
        "project_theseus_moecot_scaled_low_rank_specialists_v3",
    } or topology.get("mode") != "shared_trunk_language_experts":
        raise ValueError("unexpected MoECOT shared-trunk topology")
    arm_model = dict(config.get("arm_model") or {})
    expert_dim = int(arm_model.pop("expert_adapter_dim", 0))
    source_expert_dim = int(arm_model.pop("source_expert_adapter_dim", 0))
    if arm_model != dict(config.get("shared_trunk_model") or {}):
        raise ValueError("language expert model must exactly extend the shared trunk")
    if expert_dim != int(topology.get("expert_adapter_dim") or 0) or expert_dim <= 0:
        raise ValueError("language expert dimension must match the topology contract")
    if source_expert_dim != int(topology.get("source_expert_adapter_dim") or 0):
        raise ValueError("source expert dimension must match the topology contract")
    kerc_model = dict(config.get("kerc_english_model") or {})
    if kerc_model:
        kerc_dimensions = {
            key: int(kerc_model.pop(key, 0))
            for key in (
                "kerc_stage_adapter_dim",
                "kerc_decoder_stage_adapter_dim",
                "kerc_reasoner_output_delta_dim",
                "kerc_residual_choice_count",
                "kerc_residual_bottleneck_dim",
                "kerc_residual_unit_kind_count",
                "kerc_residual_unit_feature_dim",
                "kerc_residual_unit_byte_vocab_size",
                "kerc_verifier_dim",
                "kerc_verifier_output_dim",
                "kerc_decision_bottleneck_dim",
                "kerc_decision_output_dim",
            )
        }
        if kerc_model != dict(config.get("shared_trunk_model") or {}):
            raise ValueError("KERC English model must exactly extend the shared trunk")
        if (
            kerc_dimensions["kerc_stage_adapter_dim"] <= 0
            or kerc_dimensions["kerc_residual_choice_count"] < 4
            or kerc_dimensions["kerc_residual_bottleneck_dim"] <= 0
            or kerc_dimensions["kerc_residual_unit_kind_count"] < 5
            or kerc_dimensions["kerc_residual_unit_feature_dim"] <= 0
            or kerc_dimensions["kerc_residual_unit_byte_vocab_size"] != 257
            or kerc_dimensions["kerc_verifier_dim"] <= 0
            or kerc_dimensions["kerc_verifier_output_dim"]
            != len(KERC_VERIFIER_DIMENSIONS)
            or kerc_dimensions["kerc_decision_bottleneck_dim"] <= 0
            or kerc_dimensions["kerc_decision_output_dim"]
            != len(ANSWER_DISPOSITION_ORDER)
        ):
            raise ValueError("KERC English learned module dimensions are incomplete")
    if topology.get("expert_trainable_scope") not in {
        "adapter_only",
        "source_conditioned_delta",
        "low_rank_source_adapters",
    }:
        raise ValueError("unsupported language expert trainable scope")
    initialization = topology.get("shared_trunk_initialization") or {}
    bootstrap = topology.get("shared_trunk_bootstrap") or initialization
    if bootstrap.get("policy") == "project_theseus_exact_shared_trunk_migration_v1":
        for key in (
            "checkpoint",
            "checkpoint_sha256",
            "optimizer_state",
            "optimizer_state_sha256",
            "receipt",
            "receipt_sha256",
        ):
            if not bootstrap.get(key):
                raise ValueError(f"shared trunk migration missing {key}")
    elif initialization.get("policy") == "project_theseus_seeded_fresh_trunk_initialization_v1":
        if int(initialization.get("seed") or -1) != int(config.get("seed") or -2):
            raise ValueError("fresh shared trunk initialization seed mismatch")
        if not str(initialization.get("reason") or "").strip():
            raise ValueError("fresh shared trunk initialization requires a reason")
    else:
        raise ValueError("shared trunk initialization contract is required")
    boundaries = config.get("boundaries") or {}
    if any(int(boundaries.get(key) or 0) for key in (
        "public_training_rows_written", "external_inference_calls", "fallback_return_count",
        "templates_renderers_routers_tools_credit",
    )):
        raise ValueError("MoECOT training no-cheat counters must remain zero")
    if boundaries.get("hidden_generalist_fallback") != "forbidden":
        raise ValueError("hidden generalist fallback must remain forbidden")
    evaluation = config.get("evaluation") or {}
    if evaluation.get("policy") != "project_theseus_moecot_direct_model_only_evaluation_v1":
        raise ValueError("unexpected MoECOT evaluation policy")
    if not 1 <= int(evaluation.get("beam_width") or 0) <= 16:
        raise ValueError("evaluation beam width must be bounded")
    if not 1 <= int(evaluation.get("branching_factor") or 0) <= 16:
        raise ValueError("evaluation branching factor must be bounded")
    kernel_training = config.get("kernel_english_training") or {}
    kerc_decode_tokens = int(evaluation.get("kerc_decode_max_target_tokens") or 0)
    kerc_training_active = bool(kernel_training.get("required")) or bool(
        (kernel_training.get("disposition") or {}).get("full_kerc_training_enabled")
    )
    if (
        (kerc_training_active and kerc_decode_tokens < 1)
        or kerc_decode_tokens
        > int(kernel_training.get("maximum_sequence_tokens") or 0)
    ):
        raise ValueError(
            "KERC evaluation decode envelope must fit the registered KERC sequence envelope"
        )
    kerc_beam_width = int(evaluation.get("kerc_beam_width") or 0)
    if (kerc_training_active and kerc_beam_width < 1) or kerc_beam_width > 16:
        raise ValueError("KERC evaluation beam width must be bounded")
    kerc_branching_factor = int(evaluation.get("kerc_branching_factor") or 0)
    if (
        (kerc_training_active and kerc_branching_factor < 1)
        or kerc_branching_factor > 16
    ):
        raise ValueError("KERC evaluation branching factor must be bounded")
    if evaluation.get("target_visible_to_generator") is not False:
        raise ValueError("evaluation target must remain hidden from generation")
    if evaluation.get("templates_renderers_routers_tools_allowed") is not False:
        raise ValueError("assisted generation is forbidden in model-only evaluation")
    training = config.get("training") or {}
    repetitions = int(training.get("supervision_optimizer_repetitions") or 0)
    if not 1 <= repetitions <= int(
        training.get("maximum_supervision_optimizer_repetitions") or 0
    ):
        raise ValueError("supervision repetition must remain within the frozen maximum")
    source_repetitions = int(
        training.get("source_conditioned_optimizer_repetitions") or 1
    )
    if not 1 <= source_repetitions <= int(
        training.get("maximum_source_conditioned_optimizer_repetitions") or 1
    ):
        raise ValueError(
            "source-conditioned repetition must remain within the frozen maximum"
        )
    kernel_cfg = config.get("kernel_english_training") or {}
    if kernel_cfg.get("policy") != "project_theseus_moecot_kernel_english_stage_v1":
        raise ValueError("KERC training contract is required")
    kernel_disposition = validate_training_disposition(kernel_cfg)
    kernel_enabled = kernel_disposition.get("full_kerc_training_enabled") is True
    expected_first_campaign = (
        SHARED_TRUNK_ID,
        *ARM_IDS,
        *CONTROL_IDS,
        *(ENGLISH_COMPARISON_IDS if kernel_enabled else ()),
    )
    if tuple(comparison.get("first_campaign_candidate_ids") or ()) != tuple(
        expected_first_campaign
    ):
        raise ValueError("first-campaign candidate inventory mismatch")
    if tuple(kernel_cfg.get("objective_order") or ()) != (
        "surface_direct_control_v1",
        "surface_to_kernel_program_v1",
        "kernel_program_to_answer_packet_v1",
        "answer_packet_to_surface_v1",
    ):
        raise ValueError("KERC objective identity/order mismatch")
    kernel_repetitions = int(training.get("kernel_english_optimizer_repetitions") or 0)
    maximum_kernel_repetitions = int(
        training.get("maximum_kernel_english_optimizer_repetitions") or 0
    )
    if kernel_enabled:
        if not 1 <= kernel_repetitions <= maximum_kernel_repetitions:
            raise ValueError("KERC repetition must remain within the frozen maximum")
    elif kernel_repetitions != 0:
        raise ValueError("retired KERC path must receive zero optimizer repetitions")
    if not 1 <= int(kernel_cfg.get("batch_size") or 0) <= int(training["batch_size"]):
        raise ValueError("KERC batch size must be positive and no larger than the base batch")
    if int(kernel_cfg.get("maximum_sequence_tokens") or 0) <= 0:
        raise ValueError("KERC sequence budget must be positive")
    sequence_buckets = kernel_cfg.get("sequence_buckets") or {}
    bucket_rows = sequence_buckets.get("buckets") or []
    if (
        sequence_buckets.get("policy") != KERC_SEQUENCE_BUCKET_POLICY
        or sequence_buckets.get("routing")
        != "encoded_length_only_without_target_semantic_metadata"
        or [row.get("bucket_id") for row in bucket_rows]
        != ["standard_8k", "exact_high_fan_in_16k"]
        or [int(row.get("maximum_sequence_tokens") or 0) for row in bucket_rows]
        != [8192, int(kernel_cfg["maximum_sequence_tokens"])]
        or [int(row.get("maximum_batch_size") or 0) for row in bucket_rows]
        != [2, 1]
        or sequence_buckets.get("truncation_allowed") is not False
        or sequence_buckets.get("row_drop_allowed") is not False
        or sequence_buckets.get("long_bucket_capability_credit") is not False
    ):
        raise ValueError("KERC sequence-bucket contract is incomplete")
    for key in (
        "residual_auxiliary_weight",
        "unit_residual_auxiliary_weight",
        "verifier_auxiliary_weight",
    ):
        value = float(
            kernel_cfg.get(
                key,
                kernel_cfg.get("residual_auxiliary_weight", 0.0),
            )
            or 0.0
        )
        if not 0.0 < value <= 1.0:
            raise ValueError(f"KERC {key} must be positive and no greater than one")
    code_vocabulary = kernel_cfg.get("code_vocabulary") or {}
    if (
        code_vocabulary.get("policy")
        != "project_theseus_kerc_dual_code_vocabulary_v1"
        or code_vocabulary.get("fit_split") != "private_train"
        or code_vocabulary.get("surface_vocabulary_owner")
        != "canonical_moecot_target_vocab"
        or code_vocabulary.get("byte_fallback_required") is not True
        or code_vocabulary.get("dev_eval_vocabulary_fit_forbidden") is not True
        or int(code_vocabulary.get("kernel_max_vocab") or 0) < 512
        or int(code_vocabulary.get("pointer_max_vocab") or 0) < 512
    ):
        raise ValueError("KERC dual-code vocabulary contract is incomplete")
    if not 1.0 <= float(training.get("termination_loss_weight") or 0.0) <= 8.0:
        raise ValueError("termination loss weight must remain bounded")
    if not 1.0 <= float(training.get("byte_boundary_loss_weight") or 0.0) <= 8.0:
        raise ValueError("byte-boundary loss weight must remain bounded")
    if not 1.0 <= float(
        training.get("kerc_compiler_schema_continuation_loss_weight") or 0.0
    ) <= 8.0:
        raise ValueError(
            "KERC compiler schema-continuation loss weight must remain bounded"
        )
    if not 1.0 <= float(
        training.get("kerc_compiler_semantic_pointer_loss_weight", 1.0)
    ) <= 8.0:
        raise ValueError(
            "KERC compiler semantic-pointer loss weight must remain bounded"
        )
    semantic_weights_by_kind = training.get(
        "kerc_compiler_semantic_pointer_loss_weights_by_kind"
    )
    if semantic_weights_by_kind is not None and (
        not isinstance(semantic_weights_by_kind, dict)
        or set(semantic_weights_by_kind)
        != set(KERC_COMPILER_SEMANTIC_TARGET_KINDS)
        or any(
            not 1.0 <= float(semantic_weights_by_kind[kind]) <= 8.0
            for kind in KERC_COMPILER_SEMANTIC_TARGET_KINDS
        )
    ):
        raise ValueError(
            "KERC compiler semantic-pointer kind weights must be exact and bounded"
        )


def no_cheat(config: dict[str, Any]) -> dict[str, Any]:
    return {**config["boundaries"], "score_semantics": "training provenance only; direct verifier behavior is evaluated separately"}


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    write_json(temporary, payload)
    os.replace(temporary, path)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
