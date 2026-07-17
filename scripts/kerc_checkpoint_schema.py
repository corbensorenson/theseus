#!/usr/bin/env python3
"""Versioned, content-bound checkpoint migration for the KERC English candidate.

The first governed KERC canaries used MLX's unversioned NPZ weight format.  Long
training uses the v1 safetensors contract below.  Migration changes the actual
serialization format, audits every tensor, binds model/data/codebook identities,
and supports a lossless rollback rehearsal.  It grants no capability credit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


POLICY = "project_theseus_kerc_checkpoint_schema_v1"
CURRENT_SCHEMA_VERSION = 1
LEGACY_SCHEMA = "mlx_unversioned_npz_v0"
CURRENT_SCHEMA = "mlx_safetensors_v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy(value: Any) -> np.ndarray:
    return np.asarray(value)


def tensor_inventory(tensors: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_elements = 0
    for name in sorted(tensors):
        array = _numpy(tensors[name])
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"non-numeric checkpoint tensor: {name}")
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"non-finite checkpoint tensor: {name}")
        contiguous = np.ascontiguousarray(array)
        rows.append(
            {
                "name": name,
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "element_count": int(contiguous.size),
                "content_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
            }
        )
        total_elements += int(contiguous.size)
    if not rows:
        raise ValueError("checkpoint tensor inventory is empty")
    return {
        "tensor_count": len(rows),
        "element_count": total_elements,
        "inventory_sha256": canonical_sha256(rows),
        "tensors": rows,
    }


def _atomic_save_safetensors(
    path: Path, tensors: Mapping[str, Any], *, metadata: Mapping[str, str]
) -> None:
    import mlx.core as mx

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}.safetensors")
    temporary.unlink(missing_ok=True)
    mx.save_safetensors(str(temporary), dict(tensors), metadata=dict(metadata))
    os.replace(temporary, path)


def _atomic_save_npz(path: Path, tensors: Mapping[str, Any]) -> None:
    import mlx.core as mx

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}.npz")
    temporary.unlink(missing_ok=True)
    mx.savez(str(temporary), **dict(tensors))
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    import mlx.core as mx

    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(mx.load(str(path)))


def _binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "target_id",
        "role",
        "model_config_sha256",
        "plan_sha256",
        "stage_signature",
        "vocab_size",
        "kernel_code_vocabulary_sha256",
    )
    normalized = {key: binding.get(key) for key in required}
    missing = [key for key, value in normalized.items() if value in (None, "")]
    if missing:
        raise ValueError("checkpoint binding is incomplete: " + ",".join(missing))
    if normalized["role"] != "kerc_english_candidate":
        raise ValueError("checkpoint binding is not the KERC English candidate")
    normalized["vocab_size"] = int(normalized["vocab_size"])
    return normalized


def migrate_legacy_checkpoint(
    *,
    legacy_checkpoint: Path,
    legacy_optimizer: Path,
    checkpoint: Path,
    optimizer: Path,
    manifest_path: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Migrate real v0 NPZ weights to the canonical v1 safetensors contract."""

    if legacy_checkpoint.suffix != ".npz":
        raise ValueError("legacy KERC checkpoint must be NPZ")
    if checkpoint.suffix != ".safetensors" or optimizer.suffix != ".safetensors":
        raise ValueError("KERC v1 checkpoint and optimizer must be safetensors")
    normalized_binding = _binding(binding)
    weights = _load(legacy_checkpoint)
    optimizer_state = _load(legacy_optimizer)
    source_weights = tensor_inventory(weights)
    source_optimizer = tensor_inventory(optimizer_state)
    metadata = {
        "policy": POLICY,
        "schema": CURRENT_SCHEMA,
        "schema_version": str(CURRENT_SCHEMA_VERSION),
        "binding_sha256": canonical_sha256(normalized_binding),
    }
    _atomic_save_safetensors(checkpoint, weights, metadata=metadata)
    _atomic_save_safetensors(
        optimizer,
        optimizer_state,
        metadata={**metadata, "artifact_role": "optimizer_state"},
    )
    target_weights = tensor_inventory(_load(checkpoint))
    target_optimizer = tensor_inventory(_load(optimizer))
    if source_weights["inventory_sha256"] != target_weights["inventory_sha256"]:
        raise ValueError("KERC checkpoint migration changed model tensors")
    if source_optimizer["inventory_sha256"] != target_optimizer["inventory_sha256"]:
        raise ValueError("KERC checkpoint migration changed optimizer tensors")
    manifest: dict[str, Any] = {
        "policy": POLICY,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "source_schema": LEGACY_SCHEMA,
        "target_schema": CURRENT_SCHEMA,
        "binding": normalized_binding,
        "binding_sha256": canonical_sha256(normalized_binding),
        "migration": {
            "transform": "npz_to_safetensors_full_tensor_inventory_v1",
            "training_positions_added": 0,
            "capability_credit": "NONE",
        },
        "source": {
            "checkpoint_sha256": file_sha256(legacy_checkpoint),
            "optimizer_sha256": file_sha256(legacy_optimizer),
            "checkpoint_inventory": source_weights,
            "optimizer_inventory": source_optimizer,
        },
        "target": {
            "checkpoint_sha256": file_sha256(checkpoint),
            "optimizer_sha256": file_sha256(optimizer),
            "checkpoint_inventory": target_weights,
            "optimizer_inventory": target_optimizer,
        },
    }
    manifest["contract_sha256"] = canonical_sha256(manifest)
    _atomic_write_json(manifest_path, manifest)
    return manifest


def validate_checkpoint_contract(
    manifest: Mapping[str, Any],
    *,
    checkpoint: Path,
    optimizer: Path,
    binding: Mapping[str, Any],
) -> None:
    faults: list[str] = []
    if manifest.get("policy") != POLICY:
        faults.append("policy_mismatch")
    if int(manifest.get("schema_version") or -1) != CURRENT_SCHEMA_VERSION:
        faults.append("unsupported_schema_version")
    if manifest.get("target_schema") != CURRENT_SCHEMA:
        faults.append("target_schema_mismatch")
    normalized_binding = _binding(binding)
    if manifest.get("binding") != normalized_binding:
        faults.append("binding_mismatch")
    if manifest.get("binding_sha256") != canonical_sha256(normalized_binding):
        faults.append("binding_digest_mismatch")
    unsigned = dict(manifest)
    observed_contract = unsigned.pop("contract_sha256", "")
    if observed_contract != canonical_sha256(unsigned):
        faults.append("contract_digest_mismatch")
    target = manifest.get("target") or {}
    if not checkpoint.is_file() or target.get("checkpoint_sha256") != file_sha256(checkpoint):
        faults.append("checkpoint_identity_mismatch")
    if not optimizer.is_file() or target.get("optimizer_sha256") != file_sha256(optimizer):
        faults.append("optimizer_identity_mismatch")
    if not faults:
        if (target.get("checkpoint_inventory") or {}).get("inventory_sha256") != tensor_inventory(
            _load(checkpoint)
        )["inventory_sha256"]:
            faults.append("checkpoint_inventory_mismatch")
        if (target.get("optimizer_inventory") or {}).get("inventory_sha256") != tensor_inventory(
            _load(optimizer)
        )["inventory_sha256"]:
            faults.append("optimizer_inventory_mismatch")
    if faults:
        raise ValueError("KERC checkpoint contract rejected: " + ",".join(faults))


def rollback_checkpoint_contract(
    manifest: Mapping[str, Any],
    *,
    checkpoint: Path,
    optimizer: Path,
    rollback_checkpoint: Path,
    rollback_optimizer: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehearse lossless rollback to the prior unversioned serialization regime."""

    validate_checkpoint_contract(
        manifest, checkpoint=checkpoint, optimizer=optimizer, binding=binding
    )
    weights = _load(checkpoint)
    optimizer_state = _load(optimizer)
    _atomic_save_npz(rollback_checkpoint, weights)
    _atomic_save_safetensors(
        rollback_optimizer,
        optimizer_state,
        metadata={"policy": "project_theseus_kerc_checkpoint_rollback_v1"},
    )
    weight_inventory = tensor_inventory(_load(rollback_checkpoint))
    optimizer_inventory = tensor_inventory(_load(rollback_optimizer))
    source = manifest["source"]
    if weight_inventory["inventory_sha256"] != source["checkpoint_inventory"]["inventory_sha256"]:
        raise ValueError("KERC rollback changed model tensors")
    if optimizer_inventory["inventory_sha256"] != source["optimizer_inventory"]["inventory_sha256"]:
        raise ValueError("KERC rollback changed optimizer tensors")
    return {
        "policy": "project_theseus_kerc_checkpoint_rollback_v1",
        "source_contract_sha256": manifest["contract_sha256"],
        "rollback_schema": LEGACY_SCHEMA,
        "checkpoint_sha256": file_sha256(rollback_checkpoint),
        "optimizer_sha256": file_sha256(rollback_optimizer),
        "checkpoint_inventory_sha256": weight_inventory["inventory_sha256"],
        "optimizer_inventory_sha256": optimizer_inventory["inventory_sha256"],
        "training_positions_added": 0,
        "capability_credit": "NONE",
    }
