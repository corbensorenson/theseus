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

from kerc_residual_economics import (
    ResidualEconomicsFault,
    validate_promotion_economics,
)


ONTOLOGY_VERSION = "1.1.0"
QCSA_VERSION = "QCSA-VCM-1.0"
DEFAULT_QCSA_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "vcm_semantic_addressing.json"
)
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
SEMANTIC_IDENTITY_KINDS = {
    "occurrence",
    "type",
    "instance",
    "proposition",
    "expression",
    "memory",
    "tool",
    "policy",
    "capability",
    "artifact",
    "obligation",
}
QCSA_AUTHORITY_RANK = {
    "none": 0,
    "read_context": 1,
    "propose_route": 2,
    "request_effect": 3,
    "release_effect": 4,
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


class QCSAFault(ValueError):
    """Typed, fail-closed semantic-addressing contract fault."""

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
            "failure_behavior": "reject_without_identity_retarget_or_authority_widening",
        }


def load_qcsa_config(path: Path = DEFAULT_QCSA_CONFIG) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QCSAFault("VCM_QCSA_CONFIG_UNREADABLE", str(exc), path=str(path)) from exc
    return validate_qcsa_config(payload)


def validate_qcsa_config(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("policy") != "project_theseus_vcm_qcsa_integration_v1":
        raise QCSAFault("VCM_QCSA_POLICY_INVALID", str(payload.get("policy")), path="policy")
    if payload.get("state") != "RETAIN_BOUNDED_MECHANISMS_RETIRE_FULL_QCSA_FIRST_RUN":
        raise QCSAFault("VCM_QCSA_DISPOSITION_INVALID", str(payload.get("state")), path="state")
    if payload.get("architecture_scope") != "existing_vcm_scf_viea_owners_only":
        raise QCSAFault("VCM_QCSA_SIDECAR_SCOPE_INVALID", str(payload.get("architecture_scope")), path="architecture_scope")
    retained = tuple(payload.get("retained_mechanisms") or ())
    expected_retained = (
        "stable_soid_address_physical_route_indirection",
        "plural_authoritative_atlas_facets",
        "semantic_address_certificate_with_residual_and_authority_fields",
        "explicit_atlas_migration_compatibility_and_exact_rollback",
        "semantic_resolution_separate_from_effect_authority",
    )
    if retained != expected_retained:
        raise QCSAFault("VCM_QCSA_RETAINED_MECHANISMS_INVALID", _canonical(retained), path="retained_mechanisms")
    if tuple(payload.get("retired_from_first_long_run") or ()) != (
        "full_qcsa_composed_training_objective",
        "adaptive_active_question_policy",
    ):
        raise QCSAFault("VCM_QCSA_RETIRED_MECHANISMS_INVALID", _canonical(payload.get("retired_from_first_long_run")), path="retired_from_first_long_run")
    atlas = payload.get("atlas") if isinstance(payload.get("atlas"), dict) else {}
    facets = atlas.get("facets") if isinstance(atlas.get("facets"), list) else []
    facet_ids = [str(row.get("facet_id") or "") for row in facets if isinstance(row, dict)]
    if len(facet_ids) < 3 or len(set(facet_ids)) != len(facet_ids) or any(not value for value in facet_ids):
        raise QCSAFault("VCM_QCSA_ATLAS_FACETS_INVALID", _canonical(facet_ids), path="atlas.facets")
    certificate = payload.get("certificate") if isinstance(payload.get("certificate"), dict) else {}
    allowed = set(certificate.get("allowed_uses") or [])
    prohibited = set(certificate.get("prohibited_uses") or [])
    if not allowed or allowed.intersection(prohibited):
        raise QCSAFault("VCM_QCSA_CERTIFICATE_USE_POLICY_INVALID", _canonical(certificate), path="certificate")
    if certificate.get("authority_ceiling") != "propose_route":
        raise QCSAFault("VCM_QCSA_AUTHORITY_CEILING_INVALID", str(certificate.get("authority_ceiling")), path="certificate.authority_ceiling")
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    hashes = evidence.get("artifact_sha256") if isinstance(evidence.get("artifact_sha256"), dict) else {}
    expected_hashes = {
        "whitepaper", "package_manifest", "implementation_result", "frozen_evaluation_result",
        "evaluation_dispositions", "vertical_result", "implementation_validator",
        "evaluation_validator", "vertical_validator",
    }
    if set(hashes) != expected_hashes or any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in hashes.values()):
        raise QCSAFault("VCM_QCSA_EVIDENCE_HASH_SET_INVALID", _canonical(hashes), path="evidence.artifact_sha256")
    denominators = evidence.get("denominators")
    if denominators != {
        "implementation_lanes": 12,
        "heldout_cases": 60,
        "systems": 13,
        "seeds": 3,
        "prediction_records": 2340,
        "vertical_stages": 13,
        "vertical_adversarial_paths": 10,
    }:
        raise QCSAFault("VCM_QCSA_EVIDENCE_DENOMINATOR_INVALID", _canonical(denominators), path="evidence.denominators")
    measurements = evidence.get("measurements") if isinstance(evidence.get("measurements"), dict) else {}
    if not (
        float(measurements.get("qcsa_task_decision_accuracy", -1.0))
        == float(measurements.get("best_baseline_task_decision_accuracy", -2.0))
        == 1.0
        and float(measurements.get("operation_ratio", 0.0)) > 1.5
        and float(measurements.get("no_active_questions_object_accuracy", -1.0)) == 1.0
        and float(measurements.get("no_active_questions_task_accuracy", -1.0)) == 1.0
        and int(measurements.get("no_certificate_authority_unsafe_release_count", -1)) == 9
        and float(measurements.get("full_migration_compatibility", -1.0)) == 1.0
    ):
        raise QCSAFault("VCM_QCSA_EVIDENCE_MEASUREMENTS_INVALID", _canonical(measurements), path="evidence.measurements")
    replay = evidence.get("replay_state") if isinstance(evidence.get("replay_state"), dict) else {}
    if replay.get("evaluation_byte_replay") != "RED_ONE_MICRO_ROUNDING_DRIFT" or (replay.get("rounding_drift") or {}).get("decision_or_gate_changed") is not False:
        raise QCSAFault("VCM_QCSA_REPLAY_BOUNDARY_INVALID", _canonical(replay), path="evidence.replay_state")
    return copy.deepcopy(payload)


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
    if op in {
        "DEFINE",
        "SET_STYLE",
        "BIND_ALIAS",
        "LOCK_TERM",
        "PROMOTE_SHARED_RESIDUAL",
    }:
        bucket_by_op = {
            "DEFINE": "terminology",
            "SET_STYLE": "style",
            "BIND_ALIAS": "aliases",
            "LOCK_TERM": "terminology",
            "PROMOTE_SHARED_RESIDUAL": "terminology",
        }
        bucket = bucket_by_op[op]
        key = str(operation.get("key") or "")
        if not key or "value" not in operation:
            raise HRLStateFault("VCM_HRL_GLOBAL_OPERATION_INVALID", _canonical(operation), path=path)
        if actor_authority == "document":
            raise HRLStateFault("VCM_HRL_DOCUMENT_GLOBAL_MUTATION_FORBIDDEN", op, path=path)
        promotion_receipt = None
        if op == "PROMOTE_SHARED_RESIDUAL":
            try:
                promotion_receipt = validate_promotion_economics(
                    operation.get("economics")
                    if isinstance(operation.get("economics"), dict)
                    else {}
                )
            except ResidualEconomicsFault as exc:
                raise HRLStateFault(exc.code, exc.detail, path=path) from exc
            if promotion_receipt["should_promote"] is not True:
                raise HRLStateFault(
                    "VCM_HRL_RESIDUAL_PROMOTION_BEFORE_BREAK_EVEN",
                    str(promotion_receipt["minimum_uses_strict_break_even"]),
                    path=path,
                )
        if op == "LOCK_TERM" and actor_authority not in {"user", "system"}:
            raise HRLStateFault("VCM_HRL_LOCK_AUTHORITY_INSUFFICIENT", actor_authority, path=path)
        target = state["global"][bucket]
        _require_hrl_precedence(target.get(key), authority_rank, path)
        target[key] = {
            "value": copy.deepcopy(operation["value"]),
            "locked": op == "LOCK_TERM" or bool(operation.get("locked")),
            **(
                {"promotion_economics": copy.deepcopy(promotion_receipt)}
                if promotion_receipt is not None
                else {}
            ),
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


def _page_source(page: dict[str, Any]) -> dict[str, Any]:
    source = page.get("source")
    if isinstance(source, dict) and source.get("source_path"):
        return source
    authoritative = page.get("authoritative_sources")
    if isinstance(authoritative, list):
        for row in authoritative:
            if isinstance(row, dict) and row.get("source_path"):
                return row
    return {}


def _identity_kind(page: dict[str, Any]) -> str:
    metadata = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
    explicit = str(metadata.get("semantic_identity_kind") or "")
    if explicit:
        if explicit not in SEMANTIC_IDENTITY_KINDS:
            raise QCSAFault("VCM_QCSA_IDENTITY_KIND_INVALID", explicit, path="page.metadata.semantic_identity_kind")
        return explicit
    page_type = str(page.get("type") or "unknown")
    return {
        "architecture_spec": "artifact",
        "checkpoint": "artifact",
        "evidence": "proposition",
        "policy": "policy",
        "procedure": "obligation",
        "scoped_preference": "proposition",
        "task_state": "memory",
        "tool_output": "artifact",
    }.get(page_type, "occurrence")


def _semantic_identity(page: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metadata = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
    explicit_soid = str(metadata.get("semantic_object_id") or "")
    if explicit_soid:
        if not re.fullmatch(r"soid:[0-9a-f]{24,64}", explicit_soid):
            raise QCSAFault("VCM_QCSA_SOID_INVALID", explicit_soid, path="page.metadata.semantic_object_id")
        return explicit_soid, {
            "kind": "explicit_soid",
            "basis_digest": _digest(explicit_soid),
            "address_independent": True,
        }
    explicit_key = str(metadata.get("semantic_identity_key") or "")
    packet_id = str(metadata.get("packet_id") or "")
    usage_id = str(
        metadata.get("usage_event_id")
        or metadata.get("usage_id")
        or metadata.get("event_id")
        or ""
    )
    source_kind = str(metadata.get("source_kind") or "")
    source = _page_source(page)
    source_path = str(source.get("source_path") or "")
    if explicit_key:
        basis_kind, basis = "explicit_identity_key", explicit_key
    elif source_kind == "context_packet" and packet_id:
        basis_kind, basis = "context_packet_id", packet_id
    elif source_kind == "dogfood_usage_event" and usage_id:
        basis_kind, basis = "usage_event_id", usage_id
    elif source_path:
        basis_kind, basis = "authoritative_source_path", source_path
    else:
        basis_kind, basis = "legacy_address_root", _object_root(str(page.get("address") or ""))
    return _stable_id("soid", f"project-theseus\n{basis_kind}\n{basis}"), {
        "kind": basis_kind,
        "basis_digest": _digest(basis),
        "address_independent": basis_kind != "legacy_address_root",
    }


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
    source = _page_source(page)
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
    object_id, identity_basis = _semantic_identity(page)
    source = _page_source(page)
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
    if not prior:
        address_root = _object_root(address)
        prior = next(
            (
                row
                for row in prior_objects.values()
                if str(row.get("stable_namespace") or "") == address_root
                or any(
                    _object_root(str(revision_row.get("address") or "")) == address_root
                    for revision_row in row.get("revision_history", [])
                    if isinstance(revision_row, dict)
                )
            ),
            {},
        )
    prior_revisions = prior.get("revision_history") if isinstance(prior.get("revision_history"), list) else []
    revisions = [row for row in prior_revisions if isinstance(row, dict) and row.get("content_hash") != revision["content_hash"]]
    revisions.append(revision)
    revisions = revisions[-16:]
    vector = _lexical_vector(_page_text(page))
    return {
        "record_type": "semantic_object_record",
        "semantic_object_id": object_id,
        "identity_kind": _identity_kind(page),
        "identity_basis": identity_basis,
        "legacy_semantic_object_ids": sorted(
            {
                str(prior.get("semantic_object_id") or ""),
                *[str(value) for value in prior.get("legacy_semantic_object_ids", [])],
            }
            - {"", object_id}
        ),
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


def build_semantic_identity_registry(
    objects: list[dict[str, Any]],
    *,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous = previous if isinstance(previous, dict) else {}
    bindings: list[dict[str, Any]] = []
    seen_addresses: dict[str, str] = {}
    seen_soids: set[str] = set()
    legacy_lineage: list[dict[str, Any]] = []
    for row in objects:
        soid = str(row.get("semantic_object_id") or "")
        address = str(row.get("current_address") or "")
        if not re.fullmatch(r"soid:[0-9a-f]{24,64}", soid):
            raise QCSAFault("VCM_QCSA_SOID_INVALID", soid, path="objects.semantic_object_id")
        if not address:
            raise QCSAFault("VCM_QCSA_ADDRESS_MISSING", soid, path="objects.current_address")
        if address in seen_addresses:
            raise QCSAFault(
                "VCM_QCSA_ADDRESS_COLLISION",
                f"{address};owners={seen_addresses[address]},{soid}",
                path="identity_registry.address_bindings",
            )
        seen_addresses[address] = soid
        seen_soids.add(soid)
        bindings.append(
            {
                "address": address,
                "soid": soid,
                "identity_kind": row.get("identity_kind"),
                "lifecycle_state": row.get("lifecycle_state"),
                "binding_digest": _digest({"address": address, "soid": soid}),
            }
        )
        for legacy in row.get("legacy_semantic_object_ids", []):
            legacy_lineage.append(
                {
                    "from_id": str(legacy),
                    "to_soid": soid,
                    "mode": "explicit_qcsa_identity_projection",
                }
            )
    previous_bindings = {
        str(row.get("address") or ""): str(row.get("soid") or "")
        for row in previous.get("address_bindings", [])
        if isinstance(row, dict)
    }
    retargets = []
    for address, old in previous_bindings.items():
        current = seen_addresses.get(address)
        if not current or not old or old == current:
            continue
        retargets.append({"address": address, "previous_soid": old, "current_soid": current})
    if retargets:
        raise QCSAFault("VCM_QCSA_SILENT_ADDRESS_RETARGET", _canonical(retargets), path="identity_registry.address_bindings")
    body = {
        "policy": "project_theseus_vcm_semantic_identity_registry_v1",
        "qcsa_version": QCSA_VERSION,
        "object_count": len(seen_soids),
        "address_bindings": sorted(bindings, key=lambda row: (str(row["address"]), str(row["soid"]))),
        "legacy_lineage": sorted(legacy_lineage, key=lambda row: (row["from_id"], row["to_soid"])),
        "approved_identity_migrations": [],
        "identity_is_separate_from_address": all(
            bool((row.get("identity_basis") or {}).get("address_independent"))
            for row in objects
        ),
        "similarity_or_neighborhood_establishes_identity": False,
    }
    body["registry_digest"] = _digest(body)
    return body


def _atlas_path(object_row: dict[str, Any], facet_id: str, epoch: str) -> list[str]:
    provenance = object_row.get("provenance") if isinstance(object_row.get("provenance"), dict) else {}
    if facet_id == "task_retrieval":
        raw = [
            str(object_row.get("object_type") or "unknown"),
            str(object_row.get("execution_class") or "unknown"),
            *sorted((object_row.get("retrieval_vector") or {}).keys())[:2],
        ]
    elif facet_id == "evidence_authority":
        raw = [
            str(provenance.get("source_role") or "unknown"),
            "protected" if object_row.get("object_type") in PROTECTED_TYPES else "ordinary",
            str(object_row.get("identity_kind") or "occurrence"),
        ]
    elif facet_id == "lifecycle_storage":
        raw = [
            str(object_row.get("lifecycle_state") or "unknown"),
            str(object_row.get("consolidation_tier") or "unknown"),
            str((object_row.get("payload_policy") or {}).get("materialization") or "unknown"),
        ]
    else:
        raise QCSAFault("VCM_QCSA_FACET_UNKNOWN", facet_id, path="atlas.facets")
    path: list[str] = []
    prefix: list[str] = []
    for value in [item for item in raw if item]:
        prefix.append(value)
        path.append(f"sva:{epoch}:{facet_id}:{sha256(_canonical(prefix).encode('utf-8')).hexdigest()[:16]}")
    if not path:
        raise QCSAFault("VCM_QCSA_EMPTY_ADDRESS_PATH", str(object_row.get("semantic_object_id")), path=f"atlas.paths.{facet_id}")
    return path


def build_semantic_address_atlas(
    objects: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    epoch: str | None = None,
) -> dict[str, Any]:
    config = validate_qcsa_config(config)
    atlas_config = config["atlas"]
    authoritative_epoch = str(epoch or atlas_config["authoritative_epoch"])
    if not authoritative_epoch:
        raise QCSAFault("VCM_QCSA_ATLAS_EPOCH_MISSING", "", path="atlas.authoritative_epoch")
    facet_rows = copy.deepcopy(atlas_config["facets"])
    paths: dict[str, dict[str, list[str]]] = {}
    for row in objects:
        soid = str(row["semantic_object_id"])
        paths[soid] = {
            str(facet["facet_id"]): _atlas_path(row, str(facet["facet_id"]), authoritative_epoch)
            for facet in facet_rows
        }
    body = {
        "record_type": "vcm_plural_semantic_address_atlas",
        "policy": "project_theseus_vcm_plural_atlas_v1",
        "qcsa_version": QCSA_VERSION,
        "epoch_id": authoritative_epoch,
        "authority_state": "authoritative",
        "candidate_epochs_may_route": False,
        "immutable": True,
        "facets": facet_rows,
        "paths": paths,
        "object_count": len(paths),
        "residuals": [
            "paths_are_deterministic_local_projections_not_learned_semantic_quality",
            "natural_task_advantage_unestablished",
        ],
    }
    body["codebook_digest"] = _digest(
        {key: body[key] for key in ("epoch_id", "facets", "paths")}
    )
    return body


def issue_semantic_address_certificate(
    object_row: dict[str, Any],
    atlas: dict[str, Any],
    config: dict[str, Any],
    *,
    task: str,
    consumer: str = "vcm_context_compiler",
) -> dict[str, Any]:
    config = validate_qcsa_config(config)
    soid = str(object_row.get("semantic_object_id") or "")
    paths = (atlas.get("paths") or {}).get(soid)
    if not isinstance(paths, dict) or not paths:
        raise QCSAFault("VCM_QCSA_CERTIFICATE_PATHS_MISSING", soid, path="atlas.paths")
    certificate_config = config["certificate"]
    weighted_paths = [
        {
            "facet_id": facet_id,
            "path": path,
            "weight": round(1.0 / len(paths), 8),
        }
        for facet_id, path in sorted(paths.items())
    ]
    body = {
        "record_type": "semantic_address_certificate",
        "policy": "project_theseus_vcm_semantic_address_certificate_v1",
        "qcsa_version": QCSA_VERSION,
        "soid": soid,
        "identity_kind": object_row.get("identity_kind"),
        "occurrence_or_expression": {
            "address": object_row.get("current_address"),
            "content_hash": (object_row.get("current_revision") or {}).get("content_hash"),
        },
        "context": {"task_digest": _digest(task), "scope": "local_private_vcm"},
        "task": task,
        "consumer": consumer,
        "atlas_epoch": atlas.get("epoch_id"),
        "weighted_top_k_paths": weighted_paths,
        "confidence": 1.0,
        "confidence_semantics": "exact_registry_binding_not_semantic_correctness",
        "entropy": 0.0,
        "boundary_state": "exact_registered_object",
        "adequacy_termination": "sufficient_for_address_resolution_only",
        "cross_facet_consistency": len(weighted_paths) == len(paths),
        "provenance": copy.deepcopy(object_row.get("provenance") or {}),
        "groundings": [
            {
                "address": object_row.get("current_address"),
                "content_hash": (object_row.get("current_revision") or {}).get("content_hash"),
            }
        ],
        "residuals": [
            "natural_semantic_adequacy_unestablished",
            "learned_addressing_unestablished",
            "production_route_quality_unestablished",
        ],
        "allowed_uses": list(certificate_config["allowed_uses"]),
        "prohibited_uses": list(certificate_config["prohibited_uses"]),
        "authority_ceiling": certificate_config["authority_ceiling"],
        "validity": {
            "lifecycle_state": object_row.get("lifecycle_state"),
            "expires_utc": None,
            "revalidation_triggers": list(certificate_config["revalidation_triggers"]),
        },
        "migration": {
            "compatible_epoch": atlas.get("epoch_id"),
            "requires_explicit_receipt_for_new_epoch": True,
            "silent_soid_retarget_forbidden": True,
        },
        "effect_authority_granted": False,
    }
    body["certificate_digest"] = _digest(body)
    body["certificate_id"] = _stable_id("sac", body["certificate_digest"])
    return body


def verify_semantic_address_certificate(
    certificate: dict[str, Any],
    semantic_memory: dict[str, Any],
    *,
    requested_use: str,
    requested_authority: str,
    expected_task: str | None = None,
    expected_consumer: str | None = None,
) -> dict[str, Any]:
    if certificate.get("policy") != "project_theseus_vcm_semantic_address_certificate_v1":
        raise QCSAFault("VCM_QCSA_CERTIFICATE_POLICY_INVALID", str(certificate.get("policy")), path="certificate.policy")
    unsigned = copy.deepcopy(certificate)
    observed_digest = str(unsigned.pop("certificate_digest", ""))
    unsigned.pop("certificate_id", None)
    expected_digest = _digest(unsigned)
    if observed_digest != expected_digest:
        raise QCSAFault("VCM_QCSA_CERTIFICATE_DIGEST_INVALID", observed_digest, path="certificate.certificate_digest")
    if certificate.get("certificate_id") != _stable_id("sac", observed_digest):
        raise QCSAFault("VCM_QCSA_CERTIFICATE_ID_INVALID", str(certificate.get("certificate_id")), path="certificate.certificate_id")
    atlas = semantic_memory.get("semantic_address_atlas") if isinstance(semantic_memory.get("semantic_address_atlas"), dict) else {}
    if certificate.get("atlas_epoch") != atlas.get("epoch_id") or atlas.get("authority_state") != "authoritative":
        raise QCSAFault("VCM_QCSA_CERTIFICATE_EPOCH_STALE", str(certificate.get("atlas_epoch")), path="certificate.atlas_epoch")
    expected_codebook_digest = _digest(
        {key: atlas.get(key) for key in ("epoch_id", "facets", "paths")}
    )
    if atlas.get("codebook_digest") != expected_codebook_digest:
        raise QCSAFault("VCM_QCSA_ATLAS_DIGEST_INVALID", str(atlas.get("codebook_digest")), path="atlas.codebook_digest")
    soid = str(certificate.get("soid") or "")
    objects = {str(row.get("semantic_object_id") or ""): row for row in semantic_memory.get("objects", []) if isinstance(row, dict)}
    if soid not in objects:
        raise QCSAFault("VCM_QCSA_CERTIFICATE_SOID_UNKNOWN", soid, path="certificate.soid")
    object_row = objects[soid]
    occurrence = certificate.get("occurrence_or_expression") if isinstance(certificate.get("occurrence_or_expression"), dict) else {}
    current_revision = object_row.get("current_revision") if isinstance(object_row.get("current_revision"), dict) else {}
    if occurrence != {
        "address": object_row.get("current_address"),
        "content_hash": current_revision.get("content_hash"),
    }:
        raise QCSAFault("VCM_QCSA_CERTIFICATE_OBJECT_STATE_STALE", soid, path="certificate.occurrence_or_expression")
    validity = certificate.get("validity") if isinstance(certificate.get("validity"), dict) else {}
    if validity.get("lifecycle_state") != object_row.get("lifecycle_state"):
        raise QCSAFault("VCM_QCSA_CERTIFICATE_LIFECYCLE_STALE", soid, path="certificate.validity.lifecycle_state")
    atlas_paths = (atlas.get("paths") or {}).get(soid)
    expected_weighted_paths = [
        {
            "facet_id": facet_id,
            "path": path,
            "weight": round(1.0 / len(atlas_paths), 8),
        }
        for facet_id, path in sorted(atlas_paths.items())
    ] if isinstance(atlas_paths, dict) and atlas_paths else []
    if certificate.get("weighted_top_k_paths") != expected_weighted_paths:
        raise QCSAFault("VCM_QCSA_CERTIFICATE_PATHS_STALE", soid, path="certificate.weighted_top_k_paths")
    if requested_use not in set(certificate.get("allowed_uses") or []) or requested_use in set(certificate.get("prohibited_uses") or []):
        raise QCSAFault("VCM_QCSA_CERTIFICATE_USE_DENIED", requested_use, path="requested_use")
    if requested_authority not in QCSA_AUTHORITY_RANK:
        raise QCSAFault("VCM_QCSA_AUTHORITY_UNKNOWN", requested_authority, path="requested_authority")
    ceiling = str(certificate.get("authority_ceiling") or "none")
    if ceiling not in QCSA_AUTHORITY_RANK or QCSA_AUTHORITY_RANK[requested_authority] > QCSA_AUTHORITY_RANK[ceiling]:
        raise QCSAFault("VCM_QCSA_AUTHORITY_CEILING_EXCEEDED", f"requested={requested_authority};ceiling={ceiling}", path="requested_authority")
    if expected_task is not None and certificate.get("task") != expected_task:
        raise QCSAFault("VCM_QCSA_CERTIFICATE_TASK_MISMATCH", str(certificate.get("task")), path="certificate.task")
    if expected_consumer is not None and certificate.get("consumer") != expected_consumer:
        raise QCSAFault("VCM_QCSA_CERTIFICATE_CONSUMER_MISMATCH", str(certificate.get("consumer")), path="certificate.consumer")
    if not certificate.get("residuals") or certificate.get("effect_authority_granted") is not False:
        raise QCSAFault("VCM_QCSA_CERTIFICATE_BOUNDARY_INVALID", soid, path="certificate")
    return {
        "state": "VERIFIED_FOR_REQUESTED_USE",
        "soid": soid,
        "requested_use": requested_use,
        "requested_authority": requested_authority,
        "effect_authority_granted": False,
        "certificate_digest": observed_digest,
    }


def translate_semantic_address(
    semantic_memory: dict[str, Any],
    certificate: dict[str, Any],
    *,
    requested_use: str = "route_proposal",
    requested_authority: str = "propose_route",
    preferred_representation: str = "L2",
    expected_task: str | None = None,
    expected_consumer: str = "vcm_context_compiler",
) -> dict[str, Any]:
    if preferred_representation not in {"L0", "L1", "L2", "L3", "L4", "L5"}:
        raise QCSAFault("VCM_QCSA_REPRESENTATION_INVALID", preferred_representation, path="preferred_representation")
    verified = verify_semantic_address_certificate(
        certificate,
        semantic_memory,
        requested_use=requested_use,
        requested_authority=requested_authority,
        expected_task=expected_task,
        expected_consumer=expected_consumer,
    )
    objects = {str(row.get("semantic_object_id") or ""): row for row in semantic_memory.get("objects", []) if isinstance(row, dict)}
    row = objects[verified["soid"]]
    if row.get("lifecycle_state") != "active":
        raise QCSAFault("VCM_QCSA_ROUTE_LIFECYCLE_DENIED", str(row.get("lifecycle_state")), path="object.lifecycle_state")
    route = {
        "record_type": "vcm_semantic_to_physical_route",
        "policy": "project_theseus_vcm_semantic_to_physical_route_v1",
        "soid": verified["soid"],
        "semantic_address_epoch": certificate["atlas_epoch"],
        "physical_address": row.get("current_address"),
        "representation": preferred_representation,
        "requested_use": requested_use,
        "authority_request": requested_authority,
        "authority_ceiling": certificate["authority_ceiling"],
        "effect_authority_granted": False,
        "requires_separate_scf_effect_authorization": True,
        "certificate_digest": verified["certificate_digest"],
    }
    route["route_digest"] = _digest(route)
    return route


def _qcsa_state_digest(state: dict[str, Any]) -> str:
    return _digest(
        {
            key: state.get(key)
            for key in (
                "ontology", "objects", "relations", "identity_registry",
                "semantic_address_atlas", "semantic_address_certificates",
            )
        }
    )


def migrate_semantic_atlas(
    semantic_memory: dict[str, Any],
    *,
    target_epoch: str,
    changes: list[dict[str, Any]],
    inventory: dict[str, list[str]],
    shadow_passed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_inventory = {"descendants", "caches", "backups", "receipts"}
    if set(inventory) != required_inventory or any(not isinstance(inventory[key], list) or not inventory[key] for key in required_inventory):
        raise QCSAFault("VCM_QCSA_MIGRATION_INVENTORY_INCOMPLETE", _canonical(inventory), path="inventory")
    source_atlas = semantic_memory.get("semantic_address_atlas") if isinstance(semantic_memory.get("semantic_address_atlas"), dict) else {}
    source_epoch = str(source_atlas.get("epoch_id") or "")
    if not source_epoch or source_atlas.get("authority_state") != "authoritative" or not target_epoch or target_epoch == source_epoch:
        raise QCSAFault("VCM_QCSA_MIGRATION_EPOCH_INVALID", f"{source_epoch}->{target_epoch}", path="target_epoch")
    if not shadow_passed:
        raise QCSAFault("VCM_QCSA_MIGRATION_SHADOW_FAILED", target_epoch, path="shadow_passed")
    before = copy.deepcopy(semantic_memory)
    working = copy.deepcopy(semantic_memory)
    by_id = {str(row.get("semantic_object_id") or ""): row for row in working.get("objects", []) if isinstance(row, dict)}
    address_owner = {str(row.get("current_address") or ""): soid for soid, row in by_id.items()}
    compatibility: list[dict[str, Any]] = []
    typed_failures: list[dict[str, Any]] = []
    for index, change in enumerate(changes):
        mode = str(change.get("mode") or "readdress")
        if mode == "fail":
            typed_failures.append({**copy.deepcopy(change), "status": "typed_failure"})
            continue
        if mode != "readdress":
            raise QCSAFault("VCM_QCSA_MIGRATION_MODE_UNSUPPORTED", mode, path=f"changes[{index}].mode")
        soid = str(change.get("soid") or "")
        row = by_id.get(soid)
        if row is None:
            raise QCSAFault("VCM_QCSA_MIGRATION_SOID_UNKNOWN", soid, path=f"changes[{index}].soid")
        old_address = str(row.get("current_address") or "")
        new_address = str(change.get("new_address") or "")
        if not new_address or (new_address in address_owner and address_owner[new_address] != soid):
            raise QCSAFault("VCM_QCSA_MIGRATION_ADDRESS_COLLISION", new_address, path=f"changes[{index}].new_address")
        if change.get("new_soid") not in {None, "", soid}:
            raise QCSAFault("VCM_QCSA_SILENT_MIGRATION_RETARGET", str(change.get("new_soid")), path=f"changes[{index}].new_soid")
        row["current_address"] = new_address
        row["stable_namespace"] = _object_root(new_address)
        row.setdefault("revision_history", []).append(
            {
                **copy.deepcopy(row.get("current_revision") or {}),
                "address": new_address,
                "migration_epoch": target_epoch,
            }
        )
        row["current_revision"] = copy.deepcopy(row["revision_history"][-1])
        address_owner.pop(old_address, None)
        address_owner[new_address] = soid
        compatibility.append(
            {"soid": soid, "old_address": old_address, "new_address": new_address, "status": "compatible_same_soid"}
        )
    config = load_qcsa_config()
    prior_identity_context = copy.deepcopy(semantic_memory.get("identity_registry") or {})
    prior_identity_context["objects"] = copy.deepcopy(semantic_memory.get("objects") or [])
    working["identity_registry"] = build_semantic_identity_registry(
        working["objects"], previous=prior_identity_context
    )
    working["semantic_address_atlas"] = build_semantic_address_atlas(
        working["objects"], config, epoch=target_epoch
    )
    task = str(working.get("task") or "")
    working["semantic_address_certificates"] = [
        issue_semantic_address_certificate(row, working["semantic_address_atlas"], config, task=task)
        for row in working["objects"]
    ]
    before_digest = _qcsa_state_digest(before)
    after_digest = _qcsa_state_digest(working)
    receipt = {
        "record_type": "vcm_semantic_atlas_migration_receipt",
        "policy": "project_theseus_vcm_semantic_atlas_migration_v1",
        "source_epoch": source_epoch,
        "target_epoch": target_epoch,
        "compatibility": compatibility,
        "typed_failures": typed_failures,
        "shadow_passed": True,
        "inventory": copy.deepcopy(inventory),
        "before_state_digest": before_digest,
        "after_state_digest": after_digest,
        "same_soid_preserved": all(row["soid"] in by_id for row in compatibility),
        "rollback_state": before,
        "rollback_state_digest": before_digest,
        "effect_authority_granted": False,
    }
    receipt["receipt_digest"] = _digest({key: value for key, value in receipt.items() if key != "rollback_state"})
    return working, receipt


def rollback_semantic_atlas(
    migrated: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _qcsa_state_digest(migrated) != receipt.get("after_state_digest"):
        raise QCSAFault("VCM_QCSA_ROLLBACK_AFTER_STATE_MISMATCH", str(receipt.get("after_state_digest")), path="receipt.after_state_digest")
    restored = copy.deepcopy(receipt.get("rollback_state"))
    if not isinstance(restored, dict) or _qcsa_state_digest(restored) != receipt.get("rollback_state_digest"):
        raise QCSAFault("VCM_QCSA_ROLLBACK_STATE_INVALID", str(receipt.get("rollback_state_digest")), path="receipt.rollback_state")
    return restored, {
        "state": "ROLLED_BACK_EXACTLY",
        "restored_epoch": (restored.get("semantic_address_atlas") or {}).get("epoch_id"),
        "restored_state_digest": _qcsa_state_digest(restored),
        "matches_pre_migration": True,
        "effect_authority_granted": False,
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


def _migration_record(previous: dict[str, Any], objects: list[dict[str, Any]]) -> dict[str, Any]:
    prior_ontology = previous.get("ontology") if isinstance(previous.get("ontology"), dict) else {}
    from_version = str(prior_ontology.get("version") or "none")
    changes = [] if from_version == ONTOLOGY_VERSION else [
        "opaque semantic object identity independent of semantic address",
        "typed temporal relation records",
        "explicit lifecycle and consolidation tiers",
        "hybrid sparse-vector and graph retrieval",
        "plural authoritative semantic-address facets",
        "consumer-bound semantic address certificates",
        "authority-safe semantic-to-physical route proposals",
    ]
    prior_ids = {
        str(row.get("semantic_object_id") or "")
        for row in previous.get("objects", [])
        if isinstance(row, dict) and row.get("semantic_object_id")
    }
    current_ids = {str(row.get("semantic_object_id") or "") for row in objects}
    legacy_ids = {
        str(value)
        for row in objects
        for value in row.get("legacy_semantic_object_ids", [])
    }
    return {
        "record_type": "ontology_migration_record",
        "migration_id": _stable_id("ontmig", f"{from_version}->{ONTOLOGY_VERSION}"),
        "from_version": from_version,
        "to_version": ONTOLOGY_VERSION,
        "mode": "identity" if not changes else "additive_projection",
        "changes": changes,
        "destructive": False,
        "preserves_object_ids": prior_ids.issubset(current_ids),
        "preserves_identity_lineage": prior_ids.issubset(current_ids | legacy_ids),
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
        for key in (
            "ontology", "objects", "relations", "consolidation_records",
            "bounded_snapshot", "identity_registry", "semantic_address_atlas",
            "semantic_address_certificates", "qcsa_integration",
        )
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
    qcsa_config = load_qcsa_config()
    input_addresses = [str(page.get("address") or "") for page in pages]
    address_counts = Counter(input_addresses)
    duplicate_addresses = sorted(address for address, count in address_counts.items() if address and count > 1)
    if duplicate_addresses:
        raise QCSAFault(
            "VCM_QCSA_ADDRESS_COLLISION",
            _canonical(duplicate_addresses),
            path="pages.address",
        )
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
    prior_identity_context = copy.deepcopy(previous.get("identity_registry") or {})
    prior_identity_context["objects"] = copy.deepcopy(previous.get("objects") or [])
    identity_registry = build_semantic_identity_registry(
        objects,
        previous=prior_identity_context,
    )
    semantic_address_atlas = build_semantic_address_atlas(objects, qcsa_config)
    semantic_address_certificates = [
        issue_semantic_address_certificate(row, semantic_address_atlas, qcsa_config, task=task)
        for row in objects
    ]
    ontology = {
        "record_type": "semantic_memory_ontology",
        "version": ONTOLOGY_VERSION,
        "object_types": sorted(OBJECT_TYPES),
        "relation_types": RELATION_TYPES,
        "lifecycle_states": ["active", "quarantined", "retracted", "superseded"],
        "consolidation_tiers": ["hot", "warm", "cold"],
        "unknown_types_fail_closed": True,
        "semantic_identity_kinds": sorted(SEMANTIC_IDENTITY_KINDS),
        "identity_address_physical_route_are_distinct": True,
    }
    semantic_memory = {
        "policy": "project_theseus_vcm_semantic_memory_v1",
        "created_utc": _now(),
        "task": task,
        "ontology": ontology,
        "ontology_migrations": [_migration_record(previous, objects)],
        "objects": objects,
        "relations": relations,
        "rejected_relations": rejected_relations,
        "lifecycle_transactions": lifecycle_transactions,
        "consolidation_records": _consolidation_records(objects),
        "bounded_snapshot": _bounded_snapshot(objects, relations, task=task),
        "identity_registry": identity_registry,
        "semantic_address_atlas": semantic_address_atlas,
        "semantic_address_certificates": semantic_address_certificates,
        "claims": {
            "stable_semantic_identity": True,
            "identity_address_physical_route_indirection": True,
            "plural_authoritative_atlas_facets": True,
            "semantic_address_certificates": True,
            "semantic_resolution_grants_effect_authority": False,
            "adaptive_active_question_policy": False,
            "full_qcsa_matched_advantage": False,
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
    certificate_verifications = [
        verify_semantic_address_certificate(
            certificate,
            semantic_memory,
            requested_use="context_retrieval",
            requested_authority="read_context",
            expected_task=task,
            expected_consumer="vcm_context_compiler",
        )
        for certificate in semantic_address_certificates
    ]
    route_probe = None
    authority_denial = {"state": "NOT_APPLICABLE_EMPTY_MEMORY"}
    routeable_soids = {
        str(row.get("semantic_object_id") or "")
        for row in objects
        if row.get("lifecycle_state") == "active"
    }
    route_probe_certificate = next(
        (
            certificate
            for certificate in semantic_address_certificates
            if certificate.get("soid") in routeable_soids
        ),
        None,
    )
    if route_probe_certificate is not None:
        route_probe = translate_semantic_address(
            semantic_memory,
            route_probe_certificate,
            requested_use="route_proposal",
            requested_authority="propose_route",
        )
        try:
            translate_semantic_address(
                semantic_memory,
                route_probe_certificate,
                requested_use="route_proposal",
                requested_authority="release_effect",
            )
        except QCSAFault as exc:
            authority_denial = exc.record()
        else:
            raise QCSAFault("VCM_QCSA_EFFECT_AUTHORITY_ESCAPE", "release_effect was accepted", path="qcsa_integration.authority_denial")
    semantic_memory["qcsa_integration"] = {
        "policy": qcsa_config["policy"],
        "state": "GREEN",
        "architecture_scope": qcsa_config["architecture_scope"],
        "retained_mechanisms": copy.deepcopy(qcsa_config["retained_mechanisms"]),
        "retired_from_first_long_run": copy.deepcopy(qcsa_config["retired_from_first_long_run"]),
        "full_qcsa_training_objective_exposure": 0,
        "adaptive_active_question_policy_state": "RETIRED_FROM_FIRST_LONG_RUN",
        "certificate_verification_count": len(certificate_verifications),
        "certificate_count": len(semantic_address_certificates),
        "address_independent_identity_count": sum(
            1 for row in objects if (row.get("identity_basis") or {}).get("address_independent") is True
        ),
        "legacy_address_root_identity_count": sum(
            1 for row in objects if (row.get("identity_basis") or {}).get("address_independent") is not True
        ),
        "route_probe": route_probe,
        "authority_denial_probe": authority_denial,
        "evidence": copy.deepcopy(qcsa_config["evidence"]),
        "non_claims": copy.deepcopy(qcsa_config["non_claims"]),
    }
    semantic_memory["restart_replay"] = restart_replay_receipt(semantic_memory)
    semantic_memory["state_digest"] = _digest(
        {
            key: semantic_memory[key]
            for key in (
                "ontology", "objects", "relations", "consolidation_records",
                "bounded_snapshot", "identity_registry", "semantic_address_atlas",
                "semantic_address_certificates", "qcsa_integration",
            )
        }
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
