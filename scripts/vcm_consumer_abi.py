#!/usr/bin/env python3
"""Shared fail-closed VCM consumer ABI.

The VCM governor owns source resolution. This module owns the stable contract
that every runtime consumer must apply after resolution: representation
certification, copy-on-write transaction boundaries, leases, adequacy, taint,
deletion, contradiction, and authority checks. Payload text is never embedded.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOVERNOR = ROOT / "reports" / "vcm_context_governor.json"
DEFAULT_SEMANTIC_INDEX = ROOT / "reports" / "virtual_context_memory_index.json"
ABI_VERSION = "project_theseus_vcm_consumer_abi_v1"
REQUIRED_GOVERNOR_STATES = {
    "mission_brief_status": "ready",
    "deletion_closure_status": "closed",
    "scif_status": "ready",
    "context_abi_fixture_status": "ready",
    "context_resolver_status": "ready",
    "representation_certificate_status": "ready",
    "snapshot_branch_status": "ready",
}
DEFAULT_DENIED_TAINTS = {
    "public_benchmark_payload",
    "public_calibration_payload",
    "raw_private_user_text",
    "runtime_external_inference",
    "revoked",
    "deleted",
}


def governor_receipt(
    consumer_id: str,
    path: str | Path = DEFAULT_GOVERNOR,
) -> dict[str, Any]:
    governor_path = resolve(path)
    payload = read_json(governor_path)
    summary = as_dict(payload.get("summary"))
    faults: list[str] = []
    if not governor_path.exists():
        faults.append("VCM_GOVERNOR_MISSING")
    if payload.get("trigger_state") != "GREEN":
        faults.append("VCM_GOVERNOR_NOT_GREEN")
    if int(summary.get("hard_gap_count") or 0) != 0:
        faults.append("VCM_GOVERNOR_HARD_GAPS")
    for key, expected in REQUIRED_GOVERNOR_STATES.items():
        if summary.get(key) != expected:
            faults.append(f"VCM_GOVERNOR_{key.upper()}_NOT_{str(expected).upper()}")
    count_pairs = (
        ("context_abi_fixture_passed_count", "context_abi_fixture_count"),
        ("context_resolver_passed_count", "context_resolver_request_count"),
        ("representation_certificate_passed_count", "representation_certificate_count"),
        ("snapshot_branch_passed_count", "snapshot_branch_count"),
    )
    for passed_key, total_key in count_pairs:
        passed = int(summary.get(passed_key) or 0)
        total = int(summary.get(total_key) or 0)
        if total <= 0 or passed != total:
            faults.append(f"VCM_GOVERNOR_{passed_key.upper()}_INCOMPLETE")
    if int(summary.get("deletion_closure_fault_count") or 0) != 0:
        faults.append("VCM_DELETION_CLOSURE_FAULT")
    receipt_basis = {
        "consumer_id": consumer_id,
        "path": rel(governor_path),
        "created_utc": payload.get("created_utc"),
        "content_hash": file_sha256(governor_path),
        "summary": summary,
    }
    return {
        "record_type": "vcm_context_governor_receipt",
        "receipt_id": stable_id("vcm_governor_receipt", receipt_basis),
        "consumer_id": consumer_id,
        "path": rel(governor_path),
        "content_hash": receipt_basis["content_hash"],
        "created_utc": payload.get("created_utc"),
        "trigger_state": payload.get("trigger_state"),
        "summary": summary,
        "ready": not faults,
        "typed_faults": sorted(set(faults)),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }


def build_consumer_packet(
    *,
    consumer_id: str,
    purpose: str,
    read_set: list[str],
    write_set: list[str],
    authority_ceiling: list[str],
    permitted_uses: list[str],
    governor_path: str | Path = DEFAULT_GOVERNOR,
    semantic_index_path: str | Path = DEFAULT_SEMANTIC_INDEX,
    context_refs: list[dict[str, Any]] | None = None,
    materialized_authority_labels: list[str] | None = None,
    taint_labels: list[str] | None = None,
    denied_taints: list[str] | None = None,
    deletion_obligations: list[str] | None = None,
    contradiction_refs: list[str] | None = None,
    compression_loss: float = 0.0,
    max_compression_loss: float = 0.35,
    lease_seconds: int = 900,
    audit_refs: list[str] | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    issued_at = now_utc or datetime.now(timezone.utc)
    receipt = governor_receipt(consumer_id, governor_path)
    semantic_index_state = semantic_index_receipt_for(semantic_index_path)
    references = normalized_context_refs(context_refs or [], read_set, semantic_index_state)
    semantic_index_receipt = {
        key: value for key, value in semantic_index_state.items() if key != "lookup"
    }
    authority = unique(authority_ceiling)
    materialized_authority = unique(materialized_authority_labels or authority)
    taints = unique(
        (taint_labels or [])
        + [taint for row in references for taint in as_list(row.get("taint_labels"))]
    )
    denied = set(denied_taints or DEFAULT_DENIED_TAINTS)
    deletions = unique(deletion_obligations or [])
    contradictions = unique(
        (contradiction_refs or [])
        + [ref for row in references for ref in as_list(row.get("contradiction_refs"))]
    )
    faults = list(receipt.get("typed_faults") or [])
    faults.extend(reference_faults(references, issued_at))
    if set(taints).intersection(denied):
        faults.append("CONTEXT_TAINT_DENIED")
    if contradictions:
        faults.append("CONTEXT_CONTRADICTION_UNRESOLVED")
    if float(compression_loss) > float(max_compression_loss):
        faults.append("CONTEXT_OVER_COMPRESSED")
    if not authority:
        faults.append("CONTEXT_AUTHORITY_CEILING_MISSING")
    if not set(materialized_authority).issubset(set(authority)):
        faults.append("CONTEXT_AUTHORITY_WIDENING")
    if set(read_set).intersection(set(write_set)):
        faults.append("CONTEXT_SOURCE_MUTATION_ATTEMPT")
    if lease_seconds <= 0:
        faults.append("CONTEXT_LEASE_INVALID")
    faults = sorted(set(faults))
    packet_id = stable_id(
        "vcm_consumer_packet",
        consumer_id,
        purpose,
        receipt.get("receipt_id"),
        semantic_index_receipt,
        references,
        read_set,
        write_set,
        authority,
        taints,
        deletions,
        contradictions,
        compression_loss,
    )
    snapshot_id = f"snapshot://theseus/vcm/{receipt.get('content_hash') or 'missing'}"
    ready = not faults
    representation_certificate = {
        "certificate_id": stable_id("vcm_representation_certificate", packet_id),
        "consumer_id": consumer_id,
        "packet_id": packet_id,
        "source_refs": references,
        "semantic_index_receipt": semantic_index_receipt,
        "omissions": [
            "raw_payload_text_not_embedded",
            "raw_prompt_not_embedded",
            "context_packet_contains_refs_hashes_and_policy_only",
        ],
        "loss_contract": {
            "representation_contract": "consumer_context_refs_hashes_and_governance_v1",
            "payload_boundary": "references_and_hashes_only",
            "compression_loss": float(compression_loss),
            "max_compression_loss": float(max_compression_loss),
            "summary_can_raise_authority": False,
            "taint_and_omissions_preserved": True,
        },
        "permitted_uses": unique(permitted_uses),
        "authority_ceiling": authority,
        "materialized_authority_labels": materialized_authority,
        "consumer_policy": {
            "fail_closed_on_missing": True,
            "fail_closed_on_stale": True,
            "fail_closed_on_tainted_for_training": True,
            "fail_closed_on_contradiction": True,
            "fail_closed_on_over_compression": True,
            "best_effort_materialization_allowed": False,
            "authority_widening_allowed": False,
            "public_benchmark_payload_training_allowed": False,
            "raw_private_text_materialization_allowed": False,
        },
        "taint_labels": taints,
        "deletion_obligations": deletions,
        "typed_faults": faults,
        "audit_refs": unique((audit_refs or []) + [rel(governor_path), rel(semantic_index_path)]),
        **zero_counter_fields(),
    }
    snapshot_branch = {
        "branch_id": stable_id("vcm_snapshot_branch", packet_id),
        "packet_id": packet_id,
        "consumer_id": consumer_id,
        "parent_snapshot_id": snapshot_id,
        "branch_snapshot_id": f"{snapshot_id}/branch/{packet_id}",
        "operation": "consumer_context_materialization",
        "copy_on_write": True,
        "source_mutation_allowed": False,
        "read_set": unique(read_set),
        "write_set": unique(write_set),
        "branch_policy": "copy_on_write_fail_closed_no_best_effort",
        "taint_labels": taints,
        "propagated_taint_labels": taints,
        "deletion_obligations": deletions,
        "contradiction_refs": contradictions,
        "closure_state": "closed" if ready else "typed_fault",
        "faults": faults,
        "replay_boundary": "references_hashes_policy_and_typed_faults_only",
        "audit_refs": unique((audit_refs or []) + [rel(governor_path), rel(semantic_index_path)]),
        **zero_counter_fields(),
    }
    lease = {
        "record_type": "context_lease_receipt",
        "record_id": stable_id("vcm_context_lease", packet_id),
        "lease_id": stable_id("vcm_context_lease", packet_id),
        "packet_id": packet_id,
        "consumer_id": consumer_id,
        "issued_utc": utc_text(issued_at),
        "expires_utc": utc_text(issued_at + timedelta(seconds=max(0, lease_seconds))),
        "lease_seconds": lease_seconds,
        "lease_state": "active" if ready else "denied",
        "authority_ceiling": authority,
        "revocation_refs": deletions,
        "faults": faults,
        **zero_counter_fields(),
    }
    records = consumer_records(
        packet_id=packet_id,
        consumer_id=consumer_id,
        purpose=purpose,
        receipt=receipt,
        certificate=representation_certificate,
        branch=snapshot_branch,
        lease=lease,
        ready=ready,
        faults=faults,
    )
    packet = {
        "policy": ABI_VERSION,
        "packet_id": packet_id,
        "consumer_id": consumer_id,
        "purpose": purpose,
        "ready": ready,
        "adequacy_state": f"governed_sufficient_for_{purpose}" if ready else "typed_context_fault",
        "governor_receipt": receipt,
        "representation_certificate": representation_certificate,
        "snapshot_branch": snapshot_branch,
        "context_lease_receipt": lease,
        "typed_faults": faults,
        "records": records,
        "non_claims": [
            "VCM context adequacy is not learned-generation evidence.",
            "Semantic context transactions are not native KV/prefix-cache parity evidence.",
            "A ready packet proves only the declared consumer context boundary.",
        ],
        **zero_counter_fields(),
    }
    packet["validation"] = validate_consumer_packet(packet)
    if not packet["validation"]["passed"]:
        packet["ready"] = False
        packet["typed_faults"] = sorted(set(packet["typed_faults"] + packet["validation"]["faults"]))
        packet["adequacy_state"] = "typed_context_fault"
    return packet


def validate_consumer_packet(packet: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    certificate = as_dict(packet.get("representation_certificate"))
    branch = as_dict(packet.get("snapshot_branch"))
    lease = as_dict(packet.get("context_lease_receipt"))
    receipt = as_dict(packet.get("governor_receipt"))
    if packet.get("policy") != ABI_VERSION:
        faults.append("CONTEXT_ABI_VERSION_INVALID")
    if not receipt.get("ready"):
        faults.append("VCM_GOVERNOR_RECEIPT_NOT_READY")
    for key in ("source_refs", "omissions", "permitted_uses", "authority_ceiling"):
        if not as_list(certificate.get(key)):
            faults.append(f"REPRESENTATION_CERTIFICATE_{key.upper()}_MISSING")
    semantic_index_receipt = as_dict(certificate.get("semantic_index_receipt"))
    if semantic_index_receipt.get("ready") is not True or not semantic_index_receipt.get("content_hash"):
        faults.append("REPRESENTATION_CERTIFICATE_SEMANTIC_INDEX_INVALID")
    policy = as_dict(certificate.get("consumer_policy"))
    required_true = (
        "fail_closed_on_missing",
        "fail_closed_on_stale",
        "fail_closed_on_contradiction",
        "fail_closed_on_over_compression",
    )
    if any(policy.get(key) is not True for key in required_true):
        faults.append("REPRESENTATION_CERTIFICATE_FAIL_CLOSED_POLICY_INVALID")
    if policy.get("best_effort_materialization_allowed") is not False:
        faults.append("REPRESENTATION_CERTIFICATE_BEST_EFFORT_INVALID")
    if policy.get("authority_widening_allowed") is not False:
        faults.append("REPRESENTATION_CERTIFICATE_AUTHORITY_POLICY_INVALID")
    ceiling = set(str(x) for x in as_list(certificate.get("authority_ceiling")))
    actual = set(str(x) for x in as_list(certificate.get("materialized_authority_labels")))
    if not actual.issubset(ceiling):
        faults.append("CONTEXT_AUTHORITY_WIDENING")
    if as_dict(certificate.get("loss_contract")).get("summary_can_raise_authority") is not False:
        faults.append("REPRESENTATION_CERTIFICATE_LOSS_CONTRACT_INVALID")
    if branch.get("copy_on_write") is not True or branch.get("source_mutation_allowed") is not False:
        faults.append("SNAPSHOT_BRANCH_COPY_ON_WRITE_INVALID")
    if set(as_list(branch.get("read_set"))).intersection(set(as_list(branch.get("write_set")))):
        faults.append("CONTEXT_SOURCE_MUTATION_ATTEMPT")
    if set(as_list(branch.get("taint_labels"))) != set(as_list(branch.get("propagated_taint_labels"))):
        faults.append("SNAPSHOT_BRANCH_TAINT_DROPPED")
    if lease.get("lease_state") not in {"active", "denied"} or int(lease.get("lease_seconds") or 0) <= 0:
        faults.append("CONTEXT_LEASE_INVALID")
    if any(int(packet.get(key) or 0) != 0 for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")):
        faults.append("CONTEXT_NO_CHEAT_COUNTER_INVALID")
    if packet.get("raw_prompt_stored") is not False or packet.get("raw_private_text_stored") is not False:
        faults.append("CONTEXT_RAW_TEXT_BOUNDARY_INVALID")
    return {"passed": not faults, "faults": sorted(set(faults))}


def compact_consumer_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Return an audit receipt without duplicating the full context packet."""
    certificate = as_dict(packet.get("representation_certificate"))
    branch = as_dict(packet.get("snapshot_branch"))
    lease = as_dict(packet.get("context_lease_receipt"))
    governor = as_dict(packet.get("governor_receipt"))
    source_refs = as_list(certificate.get("source_refs"))
    return {
        "policy": packet.get("policy"),
        "packet_id": packet.get("packet_id"),
        "consumer_id": packet.get("consumer_id"),
        "purpose": packet.get("purpose"),
        "ready": packet.get("ready") is True,
        "adequacy_state": packet.get("adequacy_state"),
        "typed_faults": as_list(packet.get("typed_faults")),
        "validation": as_dict(packet.get("validation")),
        "governor_receipt_id": governor.get("receipt_id"),
        "governor_content_hash": governor.get("content_hash"),
        "representation_certificate_id": certificate.get("certificate_id"),
        "semantic_index_path": as_dict(certificate.get("semantic_index_receipt")).get("path"),
        "semantic_index_content_hash": as_dict(certificate.get("semantic_index_receipt")).get("content_hash"),
        "source_ref_count": len(source_refs),
        "source_ref_digest": stable_id("vcm_source_refs", source_refs),
        "omissions": as_list(certificate.get("omissions")),
        "permitted_uses": as_list(certificate.get("permitted_uses")),
        "authority_ceiling": as_list(certificate.get("authority_ceiling")),
        "taint_labels": as_list(certificate.get("taint_labels")),
        "deletion_obligations": as_list(certificate.get("deletion_obligations")),
        "snapshot_branch_id": branch.get("branch_id"),
        "parent_snapshot_id": branch.get("parent_snapshot_id"),
        "branch_snapshot_id": branch.get("branch_snapshot_id"),
        "read_write_set_digest": stable_id(
            "vcm_read_write_sets",
            as_list(branch.get("read_set")),
            as_list(branch.get("write_set")),
        ),
        "closure_state": branch.get("closure_state"),
        "lease_id": lease.get("lease_id"),
        "lease_state": lease.get("lease_state"),
        "lease_expires_utc": lease.get("expires_utc"),
        "record_types": sorted(
            {
                str(row.get("record_type"))
                for row in as_list(packet.get("records"))
                if isinstance(row, dict) and row.get("record_type")
            }
        ),
        "full_packet_evidence_required": True,
        **zero_counter_fields(),
    }


def consumer_records(
    *,
    packet_id: str,
    consumer_id: str,
    purpose: str,
    receipt: dict[str, Any],
    certificate: dict[str, Any],
    branch: dict[str, Any],
    lease: dict[str, Any],
    ready: bool,
    faults: list[str],
) -> list[dict[str, Any]]:
    support = "SUPPORTED" if ready else "BLOCKED"
    common = {
        "consumer_id": consumer_id,
        "packet_id": packet_id,
        "support_state": support,
        "vcm_context_governor_receipt_id": receipt.get("receipt_id"),
        **zero_counter_fields(),
    }
    records = [
        {
            **common,
            "record_type": "context_abi_record",
            "record_id": stable_id("consumer_context_abi", packet_id),
            "abi_version": ABI_VERSION,
            "certificate_id": certificate.get("certificate_id"),
            "snapshot_branch_id": branch.get("branch_id"),
            "lease_id": lease.get("lease_id"),
            "admission_state": "certified" if ready else "blocked",
            "adequacy_state": f"governed_sufficient_for_{purpose}" if ready else "typed_context_fault",
            "fault_state": "none" if ready else ",".join(faults),
        },
        {
            **common,
            "record_type": "context_transaction",
            "record_id": stable_id("consumer_context_transaction", packet_id),
            "transaction_id": stable_id("consumer_context_transaction", packet_id),
            **{key: branch.get(key) for key in (
                "operation", "parent_snapshot_id", "branch_snapshot_id", "read_set", "write_set",
                "branch_policy", "taint_labels", "propagated_taint_labels", "deletion_obligations",
                "contradiction_refs", "closure_state", "faults", "replay_boundary", "audit_refs",
            )},
        },
        {
            **common,
            "record_type": "context_adequacy",
            "record_id": stable_id("consumer_context_adequacy", packet_id),
            "adequacy_id": stable_id("consumer_context_adequacy", packet_id),
            "context_transaction_id": stable_id("consumer_context_transaction", packet_id),
            "adequacy_state": f"governed_sufficient_for_{purpose}" if ready else "typed_context_fault",
            "fail_closed": not ready,
            "residual_risks": faults,
            "source_refs": certificate.get("source_refs"),
            "omissions": certificate.get("omissions"),
        },
        lease,
        {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": stable_id("consumer_context_authority", packet_id),
            "authority_scope": certificate.get("authority_ceiling"),
            "allowed_effects": certificate.get("permitted_uses"),
            "denied_effects": ["authority_widening", "raw_payload_copy", "runtime_external_inference"],
        },
        {
            **common,
            "record_type": "failure_boundary",
            "record_id": stable_id("consumer_context_failure", packet_id),
            "failure_id": stable_id("consumer_context_failure", packet_id),
            "terminal": not ready,
            "structured_non_solved": not ready,
            "typed_faults": faults,
        },
    ]
    if faults:
        records.append(
            {
                **common,
                "record_type": "typed_context_fault",
                "record_id": stable_id("typed_context_fault", packet_id),
                "faults": faults,
                "terminal": True,
                "fallback_return_used": False,
            }
        )
    return records


def semantic_index_receipt_for(path: str | Path) -> dict[str, Any]:
    index_path = resolve(path)
    payload = read_json(index_path)
    pages = [row for row in as_list(payload.get("pages")) if isinstance(row, dict)]
    lookup: dict[str, dict[str, Any]] = {}
    for page in pages:
        address = str(page.get("address") or "")
        if address:
            lookup[address] = page
        for alias in as_list(page.get("aliases")):
            if alias:
                lookup[str(alias)] = page
    return {
        "path": rel(index_path),
        "content_hash": file_sha256(index_path),
        "policy": payload.get("policy"),
        "page_count": len(pages),
        "ready": bool(index_path.exists() and pages),
        "lookup": lookup,
    }


def normalized_context_refs(
    context_refs: list[dict[str, Any]],
    read_set: list[str],
    semantic_index_receipt: dict[str, Any],
) -> list[dict[str, Any]]:
    by_ref: dict[str, dict[str, Any]] = {}
    semantic_lookup = as_dict(semantic_index_receipt.get("lookup"))
    for source in context_refs:
        if not isinstance(source, dict):
            continue
        row = dict(source)
        key = str(row.get("ref") or row.get("path") or "")
        if row.get("kind") == "semantic_address":
            indexed = as_dict(semantic_lookup.get(key))
            source_path = str(indexed.get("source_path") or "")
            source_file = resolve(source_path) if source_path else None
            row.update(
                {
                    "exists": bool(indexed),
                    "resolved_from_index": bool(indexed),
                    "canonical_address": indexed.get("address"),
                    "source_path": source_path,
                    "source_sha256": file_sha256(source_file) if source_file else "",
                    "status": indexed.get("status"),
                    "model_visible": indexed.get("model_visible") is True,
                    "taint_labels": unique(
                        as_list(indexed.get("taints")) + as_list(row.get("taint_labels"))
                    ),
                    "contradiction_refs": unique(
                        as_list(indexed.get("contradiction_refs"))
                        + as_list(row.get("contradiction_refs"))
                    ),
                }
            )
            if str(indexed.get("status") or "").lower() in {"revoked", "deleted", "retired"}:
                row["revoked"] = True
        if key not in by_ref:
            by_ref[key] = row
            continue
        current = by_ref[key]
        current["taint_labels"] = unique(as_list(current.get("taint_labels")) + as_list(row.get("taint_labels")))
        current["contradiction_refs"] = unique(
            as_list(current.get("contradiction_refs")) + as_list(row.get("contradiction_refs"))
        )
        current["required"] = current.get("required") is not False or row.get("required") is not False
        current["exists"] = bool(current.get("exists")) or bool(row.get("exists"))
        if not current.get("sha256") and row.get("sha256"):
            current["sha256"] = row.get("sha256")
    rows = list(by_ref.values())
    known = set(by_ref)
    for item in read_set:
        if item in known:
            continue
        path = resolve(item)
        rows.append(
            {
                "kind": "artifact_ref",
                "ref": item,
                "required": True,
                "exists": path.exists(),
                "sha256": file_sha256(path),
                "taint_labels": [],
                "contradiction_refs": [],
            }
        )
    return rows


def reference_faults(references: list[dict[str, Any]], now_utc: datetime) -> list[str]:
    faults: list[str] = []
    for row in references:
        required = row.get("required") is not False
        ref = str(row.get("ref") or row.get("path") or "")
        exists = row.get("exists")
        if exists is None and ref:
            exists = resolve(ref).exists()
        if required and (not ref or not exists):
            faults.append("CONTEXT_REQUIRED_MISSING")
        if row.get("kind") == "semantic_address" and row.get("resolved_from_index") is not True:
            faults.append("CONTEXT_SEMANTIC_REF_UNRESOLVED")
        if required and row.get("kind") == "semantic_address" and row.get("model_visible") is not True:
            faults.append("CONTEXT_SEMANTIC_REF_NOT_MODEL_VISIBLE")
        if row.get("stale") is True:
            faults.append("CONTEXT_REQUIRED_STALE")
        max_age_seconds = int(row.get("max_age_seconds") or 0)
        created = parse_utc(str(row.get("created_utc") or ""))
        if max_age_seconds > 0 and created and (now_utc - created).total_seconds() > max_age_seconds:
            faults.append("CONTEXT_REQUIRED_STALE")
        if as_list(row.get("contradiction_refs")):
            faults.append("CONTEXT_CONTRADICTION_UNRESOLVED")
        if row.get("revoked") is True or row.get("deleted") is True:
            faults.append("CONTEXT_REVOKED_OR_DELETED")
    return faults


def zero_counter_fields() -> dict[str, Any]:
    return {
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }


def stable_id(*parts: Any) -> str:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = resolve(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except ValueError:
        return str(value)


def unique(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def parse_utc(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
