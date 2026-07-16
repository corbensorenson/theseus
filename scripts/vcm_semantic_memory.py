"""Durable semantic-memory projection for the canonical VCM implementation.

This module does not create a second memory authority. It deterministically
projects VCM pages, graph edges, and usage events into stable semantic objects,
typed relations, lifecycle decisions, and bounded snapshots that are embedded
in the existing VCM graph/report artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


ONTOLOGY_VERSION = "1.0.0"
OBJECT_TYPES = {
    "architecture_spec",
    "checkpoint",
    "evidence",
    "policy",
    "procedure",
    "scoped_preference",
    "task_state",
    "tool_output",
    "unknown",
}
RELATION_TYPES = {
    "contradicts": {"temporal": True, "transitive": False},
    "depends_on": {"temporal": True, "transitive": True},
    "derived_from": {"temporal": True, "transitive": True},
    "invalidates": {"temporal": True, "transitive": True},
    "rejected_because": {"temporal": True, "transitive": False},
    "supports": {"temporal": True, "transitive": False},
    "supersedes": {"temporal": True, "transitive": True},
}
PROTECTED_TYPES = {"architecture_spec", "policy", "procedure", "task_state"}
HRL_VERSION = "HRL-1.0"
HRL_LIFECYCLE_STATES = {"ACTIVE", "CHECKPOINTED", "CLOSED"}
HRL_AUTHORITIES = {"compiler": 0, "document": 1, "tool": 2, "user": 3, "system": 4}
HRL_GLOBAL_BUCKETS = {"terminology", "aliases", "style", "units", "formatting", "fidelity", "security_labels"}


class HRLStateFault(ValueError):
    """Typed, fail-closed Hierarchical Residual Ledger fault."""

    def __init__(self, code: str, detail: str, *, path: str = "") -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.path = path

    def record(self) -> dict[str, Any]:
        return {
            "fault_type": self.code,
            "detail": self.detail,
            "path": self.path,
            "failure_behavior": "reject_without_approximation_or_cross_scope_reuse",
        }


def create_hierarchical_residual_state(
    interaction_id: str,
    *,
    scope: dict[str, Any],
    language: str = "en",
) -> dict[str, Any]:
    """Create VCM-owned interaction residual state with no implicit user sharing."""

    if not interaction_id:
        raise HRLStateFault("VCM_HRL_INTERACTION_ID_MISSING", "interaction_id is required")
    normalized_scope = _validate_hrl_scope(scope)
    state = {
        "record_type": "vcm_hierarchical_residual_state",
        "hrl_version": HRL_VERSION,
        "interaction_id": interaction_id,
        "sequence": 0,
        "lifecycle_state": "ACTIVE",
        "parent_state_hash": None,
        "scope": normalized_scope,
        "global": {
            "language": language,
            "terminology": {},
            "aliases": {},
            "style": {},
            "units": {},
            "formatting": {},
            "fidelity": {},
            "security_labels": {},
        },
        "segments": {},
        "token_residuals": {},
        "exact_object_refs": {},
        "update_log": [],
        "checkpoint_parent_hashes": [],
        "tombstones": [],
        "model_parameter_storage": False,
        "cross_user_reuse_allowed": False,
    }
    state["state_hash"] = hierarchical_residual_state_hash(state)
    return state


def hierarchical_residual_state_hash(state: dict[str, Any]) -> str:
    payload = copy.deepcopy(state)
    payload.pop("state_hash", None)
    return _digest(payload)


def validate_hierarchical_residual_state(
    state: dict[str, Any],
    *,
    expected_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(state, dict) or state.get("hrl_version") != HRL_VERSION:
        raise HRLStateFault(
            "VCM_HRL_VERSION_INCOMPATIBLE",
            str(state.get("hrl_version") if isinstance(state, dict) else None),
            path="state.hrl_version",
        )
    if state.get("lifecycle_state") not in HRL_LIFECYCLE_STATES:
        raise HRLStateFault("VCM_HRL_LIFECYCLE_INVALID", str(state.get("lifecycle_state")), path="state.lifecycle_state")
    scope = _validate_hrl_scope(state.get("scope") if isinstance(state.get("scope"), dict) else {})
    if expected_scope is not None:
        requested = _validate_hrl_scope(expected_scope)
        for field in ("user", "project", "organization", "conversation"):
            existing = scope.get(field)
            incoming = requested.get(field)
            if existing is not None and incoming != existing:
                raise HRLStateFault(
                    "VCM_HRL_SCOPE_MISMATCH",
                    f"{field}: state={existing} request={incoming}",
                    path=f"state.scope.{field}",
                )
    observed_hash = str(state.get("state_hash") or "")
    expected_hash = hierarchical_residual_state_hash(state)
    if observed_hash != expected_hash:
        raise HRLStateFault(
            "VCM_HRL_STATE_HASH_INVALID",
            f"observed={observed_hash} expected={expected_hash}",
            path="state.state_hash",
        )
    return {
        "state": "READY",
        "state_hash": observed_hash,
        "scope_hash": _digest(scope),
        "sequence": int(state.get("sequence") or 0),
        "lifecycle_state": state["lifecycle_state"],
        "cross_user_reuse_allowed": False,
        "fallback_return_count": 0,
    }


def apply_hierarchical_residual_delta(
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    *,
    expected_state_hash: str,
    actor_authority: str,
    actor_id: str,
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply an append-only, authority-aware HRL delta transactionally."""

    validate_hierarchical_residual_state(state)
    if state["state_hash"] != expected_state_hash:
        raise HRLStateFault(
            "VCM_HRL_STATE_DESYNCHRONIZED",
            f"expected={expected_state_hash} local={state['state_hash']}",
            path="expected_state_hash",
        )
    if actor_authority not in HRL_AUTHORITIES or not actor_id:
        raise HRLStateFault("VCM_HRL_ACTOR_INVALID", f"{actor_authority}:{actor_id}", path="actor")
    if not isinstance(provenance, dict) or not provenance:
        raise HRLStateFault("VCM_HRL_PROVENANCE_MISSING", "delta provenance is required", path="provenance")
    if not isinstance(operations, list) or not operations:
        raise HRLStateFault("VCM_HRL_DELTA_EMPTY", "at least one operation is required", path="operations")
    if state["lifecycle_state"] == "CLOSED":
        raise HRLStateFault("VCM_HRL_STATE_CLOSED", "closed state is immutable", path="state.lifecycle_state")

    working = copy.deepcopy(state)
    parent_hash = working["state_hash"]
    applied = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise HRLStateFault("VCM_HRL_OPERATION_INVALID", str(operation), path=f"operations[{index}]")
        applied.append(
            _apply_hrl_operation(
                working,
                operation,
                actor_authority=actor_authority,
                actor_id=actor_id,
                provenance=provenance,
                path=f"operations[{index}]",
            )
        )
    working["sequence"] = int(working.get("sequence") or 0) + 1
    working["parent_state_hash"] = parent_hash
    delta_core = {
        "record_type": "vcm_hierarchical_residual_delta",
        "hrl_version": HRL_VERSION,
        "interaction_id": working["interaction_id"],
        "sequence": working["sequence"],
        "parent_state_hash": parent_hash,
        "operations": copy.deepcopy(operations),
        "applied": applied,
        "actor_authority": actor_authority,
        "actor_id": actor_id,
        "provenance": copy.deepcopy(provenance),
    }
    delta_core["delta_id"] = _stable_id("hrldelta", _canonical(delta_core))
    working["update_log"].append(
        {
            "delta_id": delta_core["delta_id"],
            "sequence": working["sequence"],
            "parent_state_hash": parent_hash,
            "operation_count": len(applied),
            "actor_authority": actor_authority,
            "provenance_hash": _digest(provenance),
        }
    )
    working.pop("state_hash", None)
    working["state_hash"] = hierarchical_residual_state_hash(working)
    delta_core["result_state_hash"] = working["state_hash"]
    delta_core["delta_sha256"] = _digest(delta_core)
    delta_core["failure_behavior"] = "transaction_rejected_without_partial_commit"
    delta_core["fallback_return_count"] = 0
    validate_hierarchical_residual_state(working)
    return working, delta_core


def replay_hierarchical_residual_deltas(
    initial_state: dict[str, Any],
    deltas: list[dict[str, Any]],
) -> dict[str, Any]:
    state = copy.deepcopy(initial_state)
    validate_hierarchical_residual_state(state)
    for index, delta in enumerate(deltas):
        if not isinstance(delta, dict):
            raise HRLStateFault("VCM_HRL_DELTA_INVALID", str(delta), path=f"deltas[{index}]")
        if delta.get("parent_state_hash") != state.get("state_hash"):
            raise HRLStateFault("VCM_HRL_REPLAY_PARENT_MISMATCH", str(delta.get("parent_state_hash")), path=f"deltas[{index}]")
        state, replayed = apply_hierarchical_residual_delta(
            state,
            copy.deepcopy(delta.get("operations") or []),
            expected_state_hash=str(delta["parent_state_hash"]),
            actor_authority=str(delta.get("actor_authority") or ""),
            actor_id=str(delta.get("actor_id") or ""),
            provenance=copy.deepcopy(delta.get("provenance") or {}),
        )
        if replayed["delta_id"] != delta.get("delta_id") or state["state_hash"] != delta.get("result_state_hash"):
            raise HRLStateFault("VCM_HRL_REPLAY_DIGEST_MISMATCH", str(delta.get("delta_id")), path=f"deltas[{index}]")
    return {
        "state": state,
        "delta_count": len(deltas),
        "state_digest_match": True,
        "deterministic_replay": True,
        "fallback_return_count": 0,
    }


def migrate_hierarchical_residual_state(
    state: dict[str, Any],
    *,
    target_version: str = HRL_VERSION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministically migrate the only admitted legacy schema to HRL-1.0."""

    source_version = str(state.get("hrl_version") or "") if isinstance(state, dict) else ""
    if target_version != HRL_VERSION:
        raise HRLStateFault("VCM_HRL_MIGRATION_TARGET_UNSUPPORTED", target_version, path="target_version")
    if source_version == HRL_VERSION:
        validate_hierarchical_residual_state(state)
        migrated = copy.deepcopy(state)
        return migrated, {
            "mode": "identity",
            "from_version": source_version,
            "to_version": target_version,
            "state_hash_preserved": True,
            "semantic_fields_preserved": True,
        }
    if source_version != "HRL-0.9":
        raise HRLStateFault("VCM_HRL_MIGRATION_SOURCE_UNSUPPORTED", source_version, path="state.hrl_version")
    migrated = copy.deepcopy(state)
    migrated["hrl_version"] = HRL_VERSION
    migrated.setdefault("record_type", "vcm_hierarchical_residual_state")
    migrated.setdefault("sequence", 0)
    migrated.setdefault("lifecycle_state", "ACTIVE")
    migrated.setdefault("parent_state_hash", None)
    migrated["scope"] = _validate_hrl_scope(migrated.get("scope") if isinstance(migrated.get("scope"), dict) else {})
    global_state = migrated.setdefault("global", {})
    global_state.setdefault("language", "en")
    for bucket in HRL_GLOBAL_BUCKETS:
        global_state.setdefault(bucket, {})
    migrated.setdefault("segments", {})
    migrated.setdefault("token_residuals", {})
    migrated.setdefault("exact_object_refs", {})
    migrated.setdefault("update_log", [])
    migrated.setdefault("checkpoint_parent_hashes", [])
    migrated.setdefault("tombstones", [])
    migrated["model_parameter_storage"] = False
    migrated["cross_user_reuse_allowed"] = False
    migrated.pop("state_hash", None)
    migrated["state_hash"] = hierarchical_residual_state_hash(migrated)
    validate_hierarchical_residual_state(migrated)
    receipt = {
        "mode": "deterministic_additive_projection",
        "from_version": source_version,
        "to_version": target_version,
        "semantic_fields_preserved": True,
        "cross_user_reuse_widened": False,
        "result_state_hash": migrated["state_hash"],
        "rollback": "retain content-bound HRL-0.9 checkpoint; no in-place overwrite",
    }
    receipt["migration_sha256"] = _digest(receipt)
    return migrated, receipt


def _apply_hrl_operation(
    state: dict[str, Any],
    operation: dict[str, Any],
    *,
    actor_authority: str,
    actor_id: str,
    provenance: dict[str, Any],
    path: str,
) -> dict[str, Any]:
    op = str(operation.get("op") or "")
    authority_rank = HRL_AUTHORITIES[actor_authority]
    common = {
        "authority": actor_authority,
        "authority_rank": authority_rank,
        "actor_id": actor_id,
        "provenance_hash": _digest(provenance),
        "privacy": str(operation.get("privacy") or "interaction_private"),
        "expiry": operation.get("expiry"),
    }
    if op in {"DEFINE", "SET_STYLE", "BIND_ALIAS", "LOCK_TERM"}:
        bucket_by_op = {
            "DEFINE": "terminology",
            "SET_STYLE": "style",
            "BIND_ALIAS": "aliases",
            "LOCK_TERM": "terminology",
        }
        bucket = bucket_by_op[op]
        key = str(operation.get("key") or "")
        if not key or "value" not in operation:
            raise HRLStateFault("VCM_HRL_GLOBAL_OPERATION_INVALID", _canonical(operation), path=path)
        if actor_authority == "document":
            raise HRLStateFault("VCM_HRL_DOCUMENT_GLOBAL_MUTATION_FORBIDDEN", op, path=path)
        if op == "LOCK_TERM" and actor_authority not in {"user", "system"}:
            raise HRLStateFault("VCM_HRL_LOCK_AUTHORITY_INSUFFICIENT", actor_authority, path=path)
        target = state["global"][bucket]
        _require_hrl_precedence(target.get(key), authority_rank, path)
        target[key] = {
            "value": copy.deepcopy(operation["value"]),
            "locked": op == "LOCK_TERM" or bool(operation.get("locked")),
            **common,
        }
        return {"op": op, "path": f"global.{bucket}.{key}", "status": "applied"}
    if op == "OVERRIDE":
        segment_id = str(operation.get("segment_id") or "")
        key = str(operation.get("key") or "")
        if not segment_id or not key or "value" not in operation:
            raise HRLStateFault("VCM_HRL_SEGMENT_OVERRIDE_INVALID", _canonical(operation), path=path)
        segment = state["segments"].setdefault(segment_id, {"entries": {}})
        _require_hrl_precedence(segment["entries"].get(key), authority_rank, path)
        segment["entries"][key] = {"value": copy.deepcopy(operation["value"]), "locked": bool(operation.get("locked")), **common}
        segment["local_state_hash"] = _digest(segment["entries"])
        return {"op": op, "path": f"segments.{segment_id}.{key}", "status": "applied"}
    if op == "SET_TOKEN_RESIDUAL":
        node_id = str(operation.get("kernel_node") or "")
        if not re.fullmatch(r"k[0-9]+", node_id) or str(operation.get("exactness") or "") not in {"semantic", "faithful", "lexical", "exact"}:
            raise HRLStateFault("VCM_HRL_TOKEN_RESIDUAL_INVALID", _canonical(operation), path=path)
        state["token_residuals"][node_id] = {
            "realization_ref": operation.get("realization_ref"),
            "morphology": copy.deepcopy(operation.get("morphology") or {}),
            "exactness": operation["exactness"],
            "source_alignment": copy.deepcopy(operation.get("source_alignment")),
            "confidence": float(operation.get("confidence", 1.0)),
            **common,
        }
        return {"op": op, "path": f"token_residuals.{node_id}", "status": "applied"}
    if op == "REGISTER_EXACT_REF":
        handle = str(operation.get("handle") or "")
        content_ref = str(operation.get("content_ref") or "")
        if not re.fullmatch(r"@[EQNDKX][0-9]+", handle) or not re.fullmatch(r"sha256:[0-9a-f]{64}", content_ref):
            raise HRLStateFault("VCM_HRL_EXACT_REF_INVALID", _canonical(operation), path=path)
        state["exact_object_refs"][handle] = {
            "content_ref": content_ref,
            "copy_policy": str(operation.get("copy_policy") or "EXACT"),
            "access_policy": str(operation.get("access_policy") or "task_scoped_least_privilege"),
            **common,
        }
        return {"op": op, "path": f"exact_object_refs.{handle}", "status": "applied"}
    if op == "EVICT":
        target_path = str(operation.get("path") or "")
        removed = _evict_hrl_path(state, target_path, authority_rank, path)
        state["tombstones"].append({"path": target_path, "prior_value_hash": _digest(removed), **common})
        return {"op": op, "path": target_path, "status": "applied"}
    if op == "RESET":
        reset_scope = str(operation.get("scope") or "")
        if actor_authority not in {"user", "system"}:
            raise HRLStateFault("VCM_HRL_RESET_AUTHORITY_INSUFFICIENT", actor_authority, path=path)
        if reset_scope == "interaction":
            for bucket in HRL_GLOBAL_BUCKETS:
                state["global"][bucket] = {}
            state["segments"] = {}
            state["token_residuals"] = {}
            state["exact_object_refs"] = {}
        elif reset_scope.startswith("segment:"):
            state["segments"].pop(reset_scope.split(":", 1)[1], None)
        else:
            raise HRLStateFault("VCM_HRL_RESET_SCOPE_INVALID", reset_scope, path=path)
        state["tombstones"].append({"path": reset_scope, "prior_value_hash": None, **common})
        return {"op": op, "path": reset_scope, "status": "applied"}
    if op == "CHECKPOINT":
        state["checkpoint_parent_hashes"].append(state["state_hash"])
        state["lifecycle_state"] = "CHECKPOINTED"
        return {"op": op, "path": "checkpoint", "status": "applied"}
    if op == "CLOSE":
        if actor_authority not in {"user", "system"}:
            raise HRLStateFault("VCM_HRL_CLOSE_AUTHORITY_INSUFFICIENT", actor_authority, path=path)
        state["lifecycle_state"] = "CLOSED"
        return {"op": op, "path": "lifecycle_state", "status": "applied"}
    raise HRLStateFault("VCM_HRL_OPERATION_UNKNOWN", op, path=path)


def _require_hrl_precedence(existing: Any, incoming_rank: int, path: str) -> None:
    if not isinstance(existing, dict):
        return
    existing_rank = int(existing.get("authority_rank") or 0)
    if existing.get("locked") and incoming_rank <= existing_rank:
        raise HRLStateFault("VCM_HRL_LOCKED_ENTRY_OVERRIDE_FORBIDDEN", str(existing_rank), path=path)
    if incoming_rank < existing_rank:
        raise HRLStateFault("VCM_HRL_AUTHORITY_PRECEDENCE_VIOLATION", f"incoming={incoming_rank} existing={existing_rank}", path=path)


def _evict_hrl_path(state: dict[str, Any], target_path: str, authority_rank: int, fault_path: str) -> Any:
    parts = target_path.split(".")
    if len(parts) != 3 or parts[0] != "global" or parts[1] not in HRL_GLOBAL_BUCKETS:
        raise HRLStateFault("VCM_HRL_EVICT_PATH_INVALID", target_path, path=fault_path)
    bucket = state["global"][parts[1]]
    if parts[2] not in bucket:
        raise HRLStateFault("VCM_HRL_EVICT_TARGET_MISSING", target_path, path=fault_path)
    _require_hrl_precedence(bucket[parts[2]], authority_rank, fault_path)
    return bucket.pop(parts[2])


def _validate_hrl_scope(scope: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(scope, dict):
        raise HRLStateFault("VCM_HRL_SCOPE_INVALID", str(scope), path="scope")
    conversation = str(scope.get("conversation") or "")
    if not conversation:
        raise HRLStateFault("VCM_HRL_CONVERSATION_SCOPE_MISSING", "conversation is required", path="scope.conversation")
    return {
        "user": str(scope["user"]) if scope.get("user") is not None else None,
        "project": str(scope["project"]) if scope.get("project") is not None else None,
        "organization": str(scope["organization"]) if scope.get("organization") is not None else None,
        "conversation": conversation,
        "expiry": scope.get("expiry"),
        "privacy": str(scope.get("privacy") or "private_local"),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return "sha256:" + sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}:{sha256(value.encode('utf-8', errors='replace')).hexdigest()[:24]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", value.lower())


def _object_root(address: str) -> str:
    return re.sub(r"@v[^/]+$", "", address)


def _semantic_id(address: str) -> str:
    return _stable_id("semobj", _object_root(address))


def _page_text(page: dict[str, Any]) -> str:
    reps = page.get("representations") if isinstance(page.get("representations"), dict) else {}
    pieces = [
        str(page.get("title") or ""),
        str(page.get("type") or ""),
        str(page.get("execution_class") or ""),
        str(page.get("address") or ""),
    ]
    for level in ("L1", "L2", "L3"):
        rep = reps.get(level) if isinstance(reps.get(level), dict) else {}
        pieces.append(str(rep.get("materialized_text") or ""))
    source = page.get("source") if isinstance(page.get("source"), dict) else {}
    pieces.append(str(source.get("source_path") or ""))
    return " ".join(pieces)


def _usage_counts(usage_events: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in usage_events:
        if not isinstance(row, dict):
            continue
        for key in ("artifact", "address", "source_path"):
            value = str(row.get(key) or "")
            if value:
                counts[value] += 1
    return counts


def _lifecycle_state(page: dict[str, Any], invalidated: set[str]) -> str:
    address = str(page.get("address") or "")
    status = str(page.get("status") or "active")
    if address in invalidated:
        return "retracted"
    if status in {"quarantined", "rejected"}:
        return "quarantined"
    if status in {"deleted", "tombstoned", "revoked"}:
        return "retracted"
    if status in {"stale", "superseded"}:
        return "superseded"
    return "active"


def _consolidation_tier(page: dict[str, Any], lifecycle: str, usage_count: int) -> str:
    page_type = str(page.get("type") or "unknown")
    if lifecycle in {"quarantined", "retracted", "superseded"}:
        return "cold"
    if page_type in PROTECTED_TYPES or usage_count > 0:
        return "hot"
    return "warm"


def _lexical_vector(text: str, limit: int = 48) -> dict[str, int]:
    counts = Counter(token for token in _tokens(text) if len(token) > 1)
    return dict(sorted(counts.most_common(limit)))


def _object_record(
    page: dict[str, Any],
    *,
    invalidated: set[str],
    usage_counts: Counter[str],
    prior_objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    address = str(page.get("address") or "")
    object_id = _semantic_id(address)
    source = page.get("source") if isinstance(page.get("source"), dict) else {}
    source_path = str(source.get("source_path") or "")
    lifecycle = _lifecycle_state(page, invalidated)
    usage_count = usage_counts[address] + usage_counts[source_path]
    tier = _consolidation_tier(page, lifecycle, usage_count)
    object_type = str(page.get("type") or "unknown")
    if object_type not in OBJECT_TYPES:
        object_type = "unknown"
    revision = {
        "address": address,
        "immutable_version": page.get("immutable_version"),
        "content_hash": page.get("content_hash"),
        "source_hash": source.get("source_hash"),
        "observed_utc": page.get("created_utc") or _now(),
    }
    prior = prior_objects.get(object_id, {})
    prior_revisions = prior.get("revision_history") if isinstance(prior.get("revision_history"), list) else []
    revisions = [row for row in prior_revisions if isinstance(row, dict) and row.get("content_hash") != revision["content_hash"]]
    revisions.append(revision)
    revisions = revisions[-16:]
    vector = _lexical_vector(_page_text(page))
    return {
        "record_type": "semantic_object_record",
        "semantic_object_id": object_id,
        "stable_namespace": _object_root(address),
        "current_address": address,
        "object_type": object_type,
        "execution_class": page.get("execution_class"),
        "lifecycle_state": lifecycle,
        "consolidation_tier": tier,
        "usage_count": usage_count,
        "current_revision": revision,
        "revision_history": revisions,
        "provenance": {
            "source_path": source_path,
            "source_role": source.get("source_role"),
            "source_hash": source.get("source_hash"),
            "taints": list(page.get("taints") or []),
        },
        "retrieval_vector": vector,
        "retrieval_vector_hash": _digest(vector),
        "payload_policy": {
            "materialization": "vcm_page_by_address",
            "compaction": "metadata_and_provenance_only_when_cold",
            "forgetting_semantics": "retrieval_suppression_and_payload_tombstone_only",
            "parametric_unlearning_claimed": False,
        },
    }


def _relation_records(
    graph: dict[str, Any],
    object_by_address: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in graph.get("edges", []) if isinstance(graph.get("edges"), list) else []:
        if not isinstance(raw, dict):
            continue
        relation_type = str(raw.get("type") or "")
        if relation_type not in RELATION_TYPES:
            continue
        source_ref = str(raw.get("from") or "")
        target_ref = str(raw.get("to") or "")
        if not source_ref or not target_ref:
            rejected.append({"reason": "missing_endpoint", "edge": raw})
            continue
        source_id = object_by_address.get(source_ref, _stable_id("external", source_ref))
        target_id = object_by_address.get(target_ref, _stable_id("external", target_ref))
        temporal = {
            "valid_from_utc": raw.get("valid_from_utc") or raw.get("created_utc"),
            "valid_to_utc": raw.get("valid_to_utc"),
            "observed_utc": raw.get("created_utc") or _now(),
        }
        body = {
            "source_object_id": source_id,
            "target_object_id": target_id,
            "source_ref": source_ref,
            "target_ref": target_ref,
            "relation_type": relation_type,
            "endpoint_scope": "internal" if source_ref in object_by_address and target_ref in object_by_address else "external_reference",
            "temporal": temporal,
            "provenance_kind": raw.get("provenance") or raw.get("reason") or "vcm_graph_edge",
        }
        body["relation_id"] = _stable_id("semrel", _canonical(body))
        body["record_type"] = "semantic_relation_record"
        records.append(body)
    unique = {str(row["relation_id"]): row for row in records}
    return sorted(unique.values(), key=lambda row: str(row["relation_id"])), rejected


def _migration_record(previous: dict[str, Any]) -> dict[str, Any]:
    prior_ontology = previous.get("ontology") if isinstance(previous.get("ontology"), dict) else {}
    from_version = str(prior_ontology.get("version") or "none")
    changes = [] if from_version == ONTOLOGY_VERSION else [
        "stable semantic object identity",
        "typed temporal relation records",
        "explicit lifecycle and consolidation tiers",
        "hybrid sparse-vector and graph retrieval",
    ]
    return {
        "record_type": "ontology_migration_record",
        "migration_id": _stable_id("ontmig", f"{from_version}->{ONTOLOGY_VERSION}"),
        "from_version": from_version,
        "to_version": ONTOLOGY_VERSION,
        "mode": "identity" if not changes else "additive_projection",
        "changes": changes,
        "destructive": False,
        "preserves_object_ids": True,
        "preserves_revision_history": True,
        "rollback": "reload prior graph artifact; source VCM pages remain authoritative",
    }


def _consolidation_records(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in objects:
        tier = str(row.get("consolidation_tier") or "warm")
        lifecycle = str(row.get("lifecycle_state") or "active")
        if lifecycle == "retracted":
            action = "suppress_retrieval_and_tombstone_cached_payload"
        elif tier == "cold":
            action = "retain_identity_provenance_and_compact_materialization"
        elif tier == "hot":
            action = "retain_index_and_bounded_materialization"
        else:
            action = "retain_sparse_index_and_load_payload_on_demand"
        records.append(
            {
                "record_type": "memory_consolidation_record",
                "semantic_object_id": row["semantic_object_id"],
                "from_tier": None,
                "to_tier": tier,
                "action": action,
                "reason": f"lifecycle={lifecycle};usage={row.get('usage_count', 0)};type={row.get('object_type')}",
                "provenance_preserved": True,
                "parametric_unlearning_claimed": False,
            }
        )
    return records


def derive_lifecycle_operations(
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for row in objects:
        object_id = str(row.get("semantic_object_id") or "")
        lifecycle = str(row.get("lifecycle_state") or "active")
        if lifecycle == "retracted":
            operations.append({"operation": "retract", "target_object_id": object_id, "reason": "vcm_invalidation_closure"})
        if str(row.get("consolidation_tier") or "") == "cold":
            operations.append({"operation": "compact", "target_object_id": object_id, "reason": f"cold_lifecycle_{lifecycle}"})
    for row in relations:
        if row.get("relation_type") != "supersedes" or row.get("endpoint_scope") != "internal":
            continue
        operations.append(
            {
                "operation": "supersede",
                "target_object_id": row.get("target_object_id"),
                "successor_object_id": row.get("source_object_id"),
                "reason": "typed_supersedes_relation",
            }
        )
    unique = {_digest(row): row for row in operations}
    return [unique[key] for key in sorted(unique)]


def apply_lifecycle_transactions(
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    operations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row.get("semantic_object_id") or ""): json.loads(_canonical(row)) for row in objects}
    relation_rows = [json.loads(_canonical(row)) for row in relations]
    transactions: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        kind = str(operation.get("operation") or "")
        target_id = str(operation.get("target_object_id") or "")
        target = by_id.get(target_id)
        touched_ids = [target_id] if target_id else []
        failure = ""
        if kind not in {"compact", "merge", "retract", "supersede"}:
            failure = "unsupported_lifecycle_operation"
        elif target is None:
            failure = "target_semantic_object_missing"
        elif kind == "compact" and target.get("consolidation_tier") != "cold":
            failure = "compaction_requires_cold_tier"
        before = {object_id: by_id.get(object_id) for object_id in touched_ids if object_id in by_id}
        added_relations: list[dict[str, Any]] = []
        if not failure and target is not None:
            if kind == "retract":
                target["lifecycle_state"] = "retracted"
                target["consolidation_tier"] = "cold"
                target["retrieval_vector"] = {}
                target["payload_policy"]["materialization"] = "tombstoned"
                target["payload_policy"]["erasure_closure"] = [
                    "semantic_retrieval_vector",
                    "bounded_snapshot_membership",
                    "compiled_context_eligibility",
                    "runtime_cache_materialization",
                ]
            elif kind == "compact":
                target["retrieval_vector"] = {}
                target["payload_policy"]["materialization"] = "compacted_metadata_and_provenance_only"
                target["payload_policy"]["physical_payload_present_in_semantic_state"] = False
                target["payload_policy"]["compaction_receipt_required"] = True
            elif kind == "supersede":
                successor_id = str(operation.get("successor_object_id") or "")
                successor = by_id.get(successor_id)
                if successor is None:
                    failure = "successor_semantic_object_missing"
                elif successor_id == target_id:
                    failure = "self_supersession_forbidden"
                else:
                    touched_ids.append(successor_id)
                    before[successor_id] = json.loads(_canonical(successor))
                    target["lifecycle_state"] = "superseded"
                    target["consolidation_tier"] = "cold"
                    target["superseded_by_object_id"] = successor_id
                    added_relations.append(
                        _lifecycle_relation(successor_id, target_id, "supersedes", str(operation.get("reason") or "lifecycle_transaction"))
                    )
            elif kind == "merge":
                source_ids = sorted({str(value) for value in operation.get("source_object_ids", []) if value})
                missing = [value for value in source_ids if value not in by_id]
                if missing:
                    failure = f"merge_source_missing:{','.join(missing)}"
                elif target_id in source_ids:
                    failure = "merge_target_cannot_be_source"
                else:
                    for source_id in source_ids:
                        source = by_id[source_id]
                        touched_ids.append(source_id)
                        before[source_id] = json.loads(_canonical(source))
                        source["lifecycle_state"] = "superseded"
                        source["consolidation_tier"] = "cold"
                        source["merged_into_object_id"] = target_id
                        added_relations.append(
                            _lifecycle_relation(target_id, source_id, "derived_from", str(operation.get("reason") or "lifecycle_merge"))
                        )
                    target["merge_source_object_ids"] = source_ids
                    target["merge_provenance"] = [by_id[source_id].get("provenance") for source_id in source_ids]
        if failure:
            for object_id, snapshot in before.items():
                by_id[object_id] = snapshot
            added_relations = []
        else:
            relation_rows.extend(added_relations)
        after = {object_id: by_id.get(object_id) for object_id in touched_ids if object_id in by_id}
        transaction_basis = {"index": index, "operation": operation, "before": before, "after": after, "failure": failure}
        transactions.append(
            {
                "record_type": "semantic_memory_lifecycle_transaction",
                "transaction_id": _stable_id("semtxn", _canonical(transaction_basis)),
                "operation": kind,
                "read_set": sorted(before),
                "write_set": [] if failure else sorted(after),
                "before_hash": _digest(before),
                "after_hash": _digest(after),
                "status": "rejected" if failure else "committed",
                "typed_failure": failure or None,
                "reason": operation.get("reason"),
                "provenance_preserved": all(bool((row or {}).get("provenance")) for row in after.values()),
                "rollback": {
                    "mode": "restore_touched_semantic_objects",
                    "before_object_hashes": {
                        object_id: _digest(snapshot) for object_id, snapshot in before.items()
                    },
                },
            }
        )
    unique_relations = {str(row.get("relation_id") or _digest(row)): row for row in relation_rows}
    return (
        sorted(by_id.values(), key=lambda row: str(row.get("semantic_object_id") or "")),
        sorted(unique_relations.values(), key=lambda row: str(row.get("relation_id") or "")),
        transactions,
    )


def _lifecycle_relation(source_id: str, target_id: str, relation_type: str, reason: str) -> dict[str, Any]:
    body = {
        "source_object_id": source_id,
        "target_object_id": target_id,
        "source_ref": source_id,
        "target_ref": target_id,
        "relation_type": relation_type,
        "endpoint_scope": "internal",
        "temporal": {"valid_from_utc": None, "valid_to_utc": None, "observed_utc": _now()},
        "provenance_kind": reason,
    }
    return {
        "record_type": "semantic_relation_record",
        "relation_id": _stable_id("semrel", _canonical(body)),
        **body,
    }


def query_semantic_memory(
    semantic_memory: dict[str, Any],
    query: str,
    *,
    limit: int = 20,
    include_suppressed: bool = False,
) -> list[dict[str, Any]]:
    objects = [row for row in semantic_memory.get("objects", []) if isinstance(row, dict)]
    query_terms = Counter(_tokens(query))
    if not query_terms:
        return []
    document_frequency: Counter[str] = Counter()
    for row in objects:
        for term in set((row.get("retrieval_vector") or {}).keys()):
            document_frequency[term] += 1
    relation_degree: Counter[str] = Counter()
    for rel in semantic_memory.get("relations", []):
        if isinstance(rel, dict):
            relation_degree[str(rel.get("source_object_id") or "")] += 1
            relation_degree[str(rel.get("target_object_id") or "")] += 1
    total = max(1, len(objects))
    scored = []
    for row in objects:
        lifecycle = str(row.get("lifecycle_state") or "active")
        if not include_suppressed and lifecycle in {"quarantined", "retracted"}:
            continue
        vector = row.get("retrieval_vector") if isinstance(row.get("retrieval_vector"), dict) else {}
        length = max(1, sum(int(v) for v in vector.values()))
        lexical = 0.0
        matched = []
        for term, qtf in query_terms.items():
            tf = int(vector.get(term) or 0)
            if not tf:
                continue
            matched.append(term)
            idf = math.log(1.0 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            lexical += qtf * idf * ((tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / 64.0)))
        object_id = str(row.get("semantic_object_id") or "")
        graph_score = min(1.0, relation_degree[object_id] / 8.0)
        authority_bonus = 0.2 if row.get("object_type") in PROTECTED_TYPES else 0.0
        score = lexical + graph_score * 0.25 + authority_bonus
        if matched:
            scored.append(
                {
                    "semantic_object_id": object_id,
                    "address": row.get("current_address"),
                    "object_type": row.get("object_type"),
                    "lifecycle_state": lifecycle,
                    "consolidation_tier": row.get("consolidation_tier"),
                    "score": round(score, 8),
                    "score_components": {
                        "sparse_bm25": round(lexical, 8),
                        "graph_degree": round(graph_score * 0.25, 8),
                        "authority_bonus": authority_bonus,
                    },
                    "matched_terms": sorted(matched),
                }
            )
    return sorted(scored, key=lambda row: (-float(row["score"]), str(row["semantic_object_id"])))[: max(1, limit)]


def _bounded_snapshot(
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    task: str,
    limit: int = 32,
) -> dict[str, Any]:
    by_id = {str(row["semantic_object_id"]): row for row in objects}
    roots = [
        str(row["semantic_object_id"])
        for row in objects
        if row.get("lifecycle_state") == "active" and row.get("object_type") in PROTECTED_TYPES
    ][:8]
    neighbors: dict[str, list[str]] = defaultdict(list)
    for row in relations:
        source = str(row.get("source_object_id") or "")
        target = str(row.get("target_object_id") or "")
        if source in by_id and target in by_id:
            neighbors[source].append(target)
            neighbors[target].append(source)
    selected: list[str] = []
    queue: deque[str] = deque(roots)
    while queue and len(selected) < limit:
        object_id = queue.popleft()
        if object_id in selected or object_id not in by_id:
            continue
        selected.append(object_id)
        queue.extend(sorted(neighbors.get(object_id, [])))
    if len(selected) < limit:
        for row in objects:
            object_id = str(row["semantic_object_id"])
            if object_id not in selected and row.get("lifecycle_state") == "active":
                selected.append(object_id)
                if len(selected) >= limit:
                    break
    selected_set = set(selected)
    selected_relations = [
        str(row["relation_id"])
        for row in relations
        if row.get("source_object_id") in selected_set and row.get("target_object_id") in selected_set
    ]
    body = {
        "task": task,
        "ontology_version": ONTOLOGY_VERSION,
        "object_ids": selected,
        "relation_ids": selected_relations,
        "limit": limit,
        "truncated": len(objects) > len(selected),
    }
    return {
        "record_type": "certified_bounded_semantic_snapshot",
        "snapshot_id": _stable_id("semsnap", _canonical(body)),
        **body,
        "certificate": {
            "content_hash": _digest(body),
            "source_object_count": len(objects),
            "selected_object_count": len(selected),
            "suppressed_lifecycle_states": ["quarantined", "retracted"],
            "deterministic_ordering": True,
        },
    }


def restart_replay_receipt(semantic_memory: dict[str, Any], *, query: str = "current project policy") -> dict[str, Any]:
    durable = {
        key: semantic_memory.get(key)
        for key in ("ontology", "objects", "relations", "consolidation_records", "bounded_snapshot")
    }
    before_digest = _digest(durable)
    reloaded = json.loads(_canonical(durable))
    after_digest = _digest(reloaded)
    before_results = query_semantic_memory(semantic_memory, query, limit=8)
    replay_state = dict(semantic_memory)
    replay_state.update(reloaded)
    after_results = query_semantic_memory(replay_state, query, limit=8)
    return {
        "record_type": "memory_restart_replay_receipt",
        "serialization": "canonical_json_round_trip",
        "before_digest": before_digest,
        "after_digest": after_digest,
        "state_digest_match": before_digest == after_digest,
        "query": query,
        "query_result_ids_before": [row["semantic_object_id"] for row in before_results],
        "query_result_ids_after": [row["semantic_object_id"] for row in after_results],
        "query_replay_match": before_results == after_results,
    }


def build_semantic_memory(
    pages: list[dict[str, Any]],
    graph: dict[str, Any],
    *,
    usage_events: list[dict[str, Any]] | None = None,
    previous: dict[str, Any] | None = None,
    task: str = "",
) -> dict[str, Any]:
    previous = previous if isinstance(previous, dict) else {}
    previous_objects = {
        str(row.get("semantic_object_id") or ""): row
        for row in previous.get("objects", [])
        if isinstance(row, dict) and row.get("semantic_object_id")
    }
    invalidation = graph.get("invalidation") if isinstance(graph.get("invalidation"), dict) else {}
    invalidated = {str(value) for value in invalidation.get("invalidated_addresses", [])}
    usage_counts = _usage_counts(usage_events or [])
    objects = [
        _object_record(page, invalidated=invalidated, usage_counts=usage_counts, prior_objects=previous_objects)
        for page in sorted(pages, key=lambda row: str(row.get("address") or ""))
    ]
    object_by_address = {str(row["current_address"]): str(row["semantic_object_id"]) for row in objects}
    relations, rejected_relations = _relation_records(graph, object_by_address)
    lifecycle_operations = derive_lifecycle_operations(objects, relations)
    objects, relations, lifecycle_transactions = apply_lifecycle_transactions(objects, relations, lifecycle_operations)
    ontology = {
        "record_type": "semantic_memory_ontology",
        "version": ONTOLOGY_VERSION,
        "object_types": sorted(OBJECT_TYPES),
        "relation_types": RELATION_TYPES,
        "lifecycle_states": ["active", "quarantined", "retracted", "superseded"],
        "consolidation_tiers": ["hot", "warm", "cold"],
        "unknown_types_fail_closed": True,
    }
    semantic_memory = {
        "policy": "project_theseus_vcm_semantic_memory_v1",
        "created_utc": _now(),
        "ontology": ontology,
        "ontology_migrations": [_migration_record(previous)],
        "objects": objects,
        "relations": relations,
        "rejected_relations": rejected_relations,
        "lifecycle_transactions": lifecycle_transactions,
        "consolidation_records": _consolidation_records(objects),
        "bounded_snapshot": _bounded_snapshot(objects, relations, task=task),
        "claims": {
            "stable_semantic_identity": True,
            "typed_temporal_relations": True,
            "ontology_migration": True,
            "graph_sparse_vector_hybrid_retrieval": True,
            "provenance_preserving_compaction": True,
            "transactional_merge_supersession_retraction_compaction": True,
            "parametric_unlearning": False,
            "dense_embedding_retrieval": False,
        },
        "external_inference_calls": 0,
    }
    semantic_memory["restart_replay"] = restart_replay_receipt(semantic_memory)
    semantic_memory["state_digest"] = _digest(
        {key: semantic_memory[key] for key in ("ontology", "objects", "relations", "consolidation_records", "bounded_snapshot")}
    )
    return semantic_memory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["replay", "migrate-replay"])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--pages", default="")
    parser.add_argument("--task", default="persisted semantic memory migration")
    parser.add_argument("--query", default="current project policy")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    payload = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    semantic_memory = payload.get("semantic_memory") if isinstance(payload.get("semantic_memory"), dict) else payload
    migration = None
    if args.command == "migrate-replay":
        if not args.pages:
            parser.error("--pages is required for migrate-replay")
        page_text = Path(args.pages).read_text(encoding="utf-8")
        if page_text.lstrip().startswith("["):
            pages = json.loads(page_text)
        else:
            pages = [json.loads(line) for line in page_text.splitlines() if line.strip()]
        base_graph = {key: value for key, value in payload.items() if key != "semantic_memory"}
        semantic_memory = build_semantic_memory(
            [row for row in pages if isinstance(row, dict)],
            base_graph,
            previous=semantic_memory,
            task=args.task,
        )
        migration = semantic_memory.get("ontology_migrations", [None])[0]
    receipt = restart_replay_receipt(semantic_memory, query=args.query)
    receipt["process_replay"] = True
    receipt["ontology_version"] = (semantic_memory.get("ontology") or {}).get("version")
    receipt["object_count"] = len(semantic_memory.get("objects") or [])
    receipt["migration"] = migration
    if args.out:
        Path(args.out).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["state_digest_match"] and receipt["query_replay_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
