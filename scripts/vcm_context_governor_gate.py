#!/usr/bin/env python3
"""VCM context adequacy and verification-bandwidth gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "vcm_context_governor.json"
DEFAULT_REPORT = ROOT / "reports" / "vcm_context_governor.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "vcm_context_governor.md"
DEFAULT_BRIEF = ROOT / "reports" / "vcm_mission_brief.json"
DEFAULT_CLOSURE = ROOT / "reports" / "vcm_deletion_closure.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--mission-brief-out", default=rel(DEFAULT_BRIEF))
    parser.add_argument("--deletion-closure-out", default=rel(DEFAULT_CLOSURE))
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(config_path, config, started)
    write_json(resolve(args.out), report)
    write_json(resolve(args.mission_brief_out), report["mission_brief"])
    write_json(resolve(args.deletion_closure_out), report["deletion_closure"])
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(config_path: Path, config: dict[str, Any], started: float) -> dict[str, Any]:
    chunks = [score_chunk(row) for row in list_dicts(config.get("context_chunks"))]
    chunk_by_id = {row["id"]: row for row in chunks}
    mission_brief = build_mission_brief(dict_value(config.get("mission_brief_fixture")), chunk_by_id)
    scif = audit_scif(dict_value(config.get("digital_scif_fixture")))
    closure = audit_deletion_closure(dict_value(config.get("deletion_closure_fixture")))
    context_abi = audit_context_abi_fixtures(list_dicts(config.get("context_abi_fixtures")))
    context_resolver = audit_context_resolver_conformance(
        list_dicts(config.get("semantic_address_catalog")),
        list_dicts(config.get("context_resolver_requests")),
    )
    representation_certificates = audit_representation_certificates(context_resolver["requests"])
    snapshot_branches = audit_snapshot_branch_ledger(context_resolver["requests"])
    boundaries = audit_boundaries(dict_value(config.get("hard_boundaries")))
    hard_gaps = []
    warnings = []
    for row in chunks:
        hard_gaps.extend(row["hard_gaps"])
        warnings.extend(row["warnings"])
    hard_gaps.extend(mission_brief["hard_gaps"])
    warnings.extend(mission_brief["warnings"])
    hard_gaps.extend(scif["hard_gaps"])
    warnings.extend(scif["warnings"])
    hard_gaps.extend(closure["hard_gaps"])
    warnings.extend(closure["warnings"])
    hard_gaps.extend(context_abi["hard_gaps"])
    warnings.extend(context_abi["warnings"])
    hard_gaps.extend(context_resolver["hard_gaps"])
    warnings.extend(context_resolver["warnings"])
    hard_gaps.extend(representation_certificates["hard_gaps"])
    warnings.extend(representation_certificates["warnings"])
    hard_gaps.extend(snapshot_branches["hard_gaps"])
    warnings.extend(snapshot_branches["warnings"])
    hard_gaps.extend([gate for gate in boundaries if gate["severity"] == "hard" and not gate["passed"]])
    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"
    summary = {
        "config": rel(config_path),
        "chunk_count": len(chunks),
        "pinned_chunk_count": sum(1 for row in chunks if row["eviction_state"] == "pinned"),
        "fail_closed_chunk_count": sum(1 for row in chunks if row["context_decision"] == "fail_closed"),
        "mission_brief_status": mission_brief["status"],
        "mission_brief_omission_count": len(mission_brief["omissions"]),
        "mission_brief_compression_loss": mission_brief["compression_loss"],
        "scif_status": scif["status"],
        "deletion_closure_status": closure["status"],
        "deletion_closure_fault_count": closure["closure_fault_count"],
        "context_abi_fixture_status": context_abi["status"],
        "context_abi_fixture_count": len(context_abi["fixtures"]),
        "context_abi_fixture_passed_count": context_abi["passed_count"],
        "context_abi_required_scenario_count": len(context_abi["required_scenarios"]),
        "context_abi_viea_record_count": len(context_abi["viea_context_abi_records"]),
        "context_resolver_status": context_resolver["status"],
        "context_resolver_request_count": len(context_resolver["requests"]),
        "context_resolver_passed_count": context_resolver["passed_count"],
        "context_resolver_required_scenario_count": len(context_resolver["required_scenarios"]),
        "context_resolver_viea_record_count": len(context_resolver["viea_context_resolver_records"]),
        "context_resolver_materialized_count": sum(1 for row in context_resolver["requests"] if row.get("decision") == "materialize"),
        "context_resolver_typed_fault_count": sum(1 for row in context_resolver["requests"] if row.get("decision") == "typed_fault"),
        "representation_certificate_status": representation_certificates["status"],
        "representation_certificate_count": len(representation_certificates["certificates"]),
        "representation_certificate_passed_count": representation_certificates["passed_count"],
        "representation_certificate_expected_invalid_rejected_count": representation_certificates["expected_invalid_rejected_count"],
        "representation_certificate_viea_record_count": len(representation_certificates["viea_representation_certificate_records"]),
        "snapshot_branch_status": snapshot_branches["status"],
        "snapshot_branch_count": len(snapshot_branches["branches"]),
        "snapshot_branch_passed_count": snapshot_branches["passed_count"],
        "snapshot_branch_expected_invalid_rejected_count": snapshot_branches["expected_invalid_rejected_count"],
        "snapshot_branch_typed_fault_count": sum(1 for row in snapshot_branches["branches"] if row.get("faults")),
        "snapshot_branch_viea_record_count": len(snapshot_branches["viea_snapshot_branch_records"]),
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
    }
    return {
        "policy": "project_theseus_vcm_context_governor_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "boundary_gates": boundaries,
        "chunks": chunks,
        "mission_brief": mission_brief,
        "digital_scif": scif,
        "deletion_closure": closure,
        "context_abi": context_abi,
        "context_resolver": context_resolver,
        "representation_certificates": representation_certificates,
        "snapshot_branch_ledger": snapshot_branches,
        "viea_context_abi_records": context_abi["viea_context_abi_records"],
        "viea_context_resolver_records": context_resolver["viea_context_resolver_records"],
        "viea_vcm_representation_certificate_records": representation_certificates["viea_representation_certificate_records"],
        "viea_vcm_snapshot_branch_records": snapshot_branches["viea_snapshot_branch_records"],
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rules": {
            "adequacy": "A run can fail closed when required context is missing, stale, tainted, overspecified, or beyond budget.",
            "compression": "Mission briefs must record omissions, compression loss, authority limits, and source chunk ids.",
            "taint": "Untrusted text cannot self-pin or become instructions through summary compression.",
            "deletion": "Deleted or revoked material must close derived summaries/caches/training rows or emit a closure fault.",
            "context_abi": "Context requests must resolve through stable address/version/mount/snapshot/representation fields and emit typed faults instead of best-effort materialization when unresolved, inadequate, denied, or lease-expired.",
            "context_resolver": "Deployed resolver conformance must resolve real semantic addresses to local artifact refs and hashes, or emit typed faults without payload leakage.",
            "representation_certificate": "Every materialized or faulted context packet must carry source refs, omissions, loss contract, permitted uses, authority ceiling, consumer policy, and a proof that summaries cannot raise source authority.",
            "snapshot_branch": "Every deployed resolver request must emit a copy-on-write branch ledger with read/write sets, taint propagation, deletion obligations, contradiction refs, closure state, and typed faults.",
            "runtime": "VCM context governance does not claim native KV/prefix-cache parity.",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def audit_boundaries(boundaries: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gate("larger_context_not_substitute", boundaries.get("larger_context_substitutes_for_verified_context") is False, "hard", boundaries.get("larger_context_substitutes_for_verified_context")),
        gate("summary_must_preserve_taint", boundaries.get("compressed_summary_may_drop_taint_or_omissions") is False, "hard", boundaries.get("compressed_summary_may_drop_taint_or_omissions")),
        gate("untrusted_context_cannot_self_pin", boundaries.get("untrusted_context_can_self_pin") is False, "hard", boundaries.get("untrusted_context_can_self_pin")),
        gate("speculative_prefetch_not_visible", boundaries.get("speculative_prefetch_may_influence_generation") is False, "hard", boundaries.get("speculative_prefetch_may_influence_generation")),
        gate("deleted_material_cannot_remain_silent", boundaries.get("deleted_material_can_remain_silent") is False, "hard", boundaries.get("deleted_material_can_remain_silent")),
        gate("runtime_external_inference_forbidden", boundaries.get("runtime_external_inference_allowed") is False, "hard", boundaries.get("runtime_external_inference_allowed")),
    ]


def score_chunk(row: dict[str, Any]) -> dict[str, Any]:
    chunk_id = str(row.get("id") or "<missing-id>")
    relevance = bounded(row.get("relevance"))
    entropy = bounded(row.get("entropy"))
    criticality = bounded(row.get("criticality"))
    goal_similarity = bounded(row.get("goal_similarity"))
    compression_loss = bounded(row.get("compression_loss"))
    taint = str(row.get("taint") or "unknown")
    clearance = str(row.get("clearance") or "unknown")
    eviction_state = str(row.get("eviction_state") or "unknown")
    value_score = round((0.35 * relevance) + (0.3 * criticality) + (0.25 * goal_similarity) + (0.1 * (1.0 - entropy)), 6)
    hard_gaps = []
    warnings = []
    if "untrusted" in taint and eviction_state == "pinned":
        hard_gaps.append(item_gap(chunk_id, "untrusted_self_pin_attempt", {"taint": taint, "eviction_state": eviction_state}))
    if compression_loss > 0.5 and criticality >= 0.7:
        hard_gaps.append(item_gap(chunk_id, "critical_context_compression_loss_too_high", {"compression_loss": compression_loss}))
    if clearance not in {"authorized", "summary_only", "evidence_ref_only"}:
        warnings.append(item_gap(chunk_id, "unknown_clearance", {"clearance": clearance}, severity="warning"))
    decision = "include"
    if "untrusted" in taint and clearance != "summary_only":
        decision = "fail_closed"
    elif "stale" in taint and clearance == "evidence_ref_only":
        decision = "reference_only"
    elif value_score < 0.35 and eviction_state != "pinned":
        decision = "evict"
    return {
        "id": chunk_id,
        "type": str(row.get("type") or ""),
        "taint": taint,
        "clearance": clearance,
        "eviction_state": eviction_state,
        "value_score": value_score,
        "compression_loss": compression_loss,
        "context_decision": decision,
        "summary": str(row.get("summary") or ""),
        "omissions": list_values(row.get("omissions")),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def build_mission_brief(fixture: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required_ids = [str(x) for x in list_values(fixture.get("required_chunks"))]
    optional_ids = [str(x) for x in list_values(fixture.get("optional_chunks"))]
    missing_required = [chunk_id for chunk_id in required_ids if chunk_id not in chunks]
    selected = [chunks[chunk_id] for chunk_id in required_ids + optional_ids if chunk_id in chunks and chunks[chunk_id]["context_decision"] in {"include", "reference_only"}]
    omissions = []
    for chunk in selected:
        omissions.extend([{"chunk_id": chunk["id"], "omission": item} for item in chunk["omissions"]])
    compression_loss = round(sum(float(chunk["compression_loss"]) for chunk in selected) / max(1, len(selected)), 6)
    authority_limits = [
        "public benchmark payloads remain calibration-only",
        "untrusted external notes are summary-only",
        "stale reports are evidence refs, not current routing authority",
        "runtime external inference is forbidden",
    ]
    text = " ".join(chunk["summary"] for chunk in selected)
    hard_gaps = []
    warnings = []
    if missing_required and fixture.get("must_fail_closed_if_missing_required") is True:
        hard_gaps.append(item_gap(str(fixture.get("id") or "mission_brief"), "missing_required_context", {"missing": missing_required}))
    if fixture.get("must_record_omissions") is True and not omissions:
        hard_gaps.append(item_gap(str(fixture.get("id") or "mission_brief"), "omissions_not_recorded", {}))
    if fixture.get("must_record_authority_limits") is True and not authority_limits:
        hard_gaps.append(item_gap(str(fixture.get("id") or "mission_brief"), "authority_limits_not_recorded", {}))
    status = "fail_closed" if hard_gaps else "ready"
    return {
        "id": str(fixture.get("id") or ""),
        "status": status,
        "objective": str(fixture.get("objective") or ""),
        "selected_chunk_ids": [chunk["id"] for chunk in selected],
        "missing_required_chunk_ids": missing_required,
        "brief": text,
        "omissions": omissions,
        "authority_limits": authority_limits,
        "compression_loss": compression_loss,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def audit_scif(fixture: dict[str, Any]) -> dict[str, Any]:
    hard_gaps = []
    warnings = []
    if fixture.get("raw_sensitive_context_visible_to_generator") is not False:
        hard_gaps.append(item_gap(str(fixture.get("id") or "scif"), "raw_sensitive_context_visible", fixture))
    if not list_values(fixture.get("handles")):
        hard_gaps.append(item_gap(str(fixture.get("id") or "scif"), "handles_missing", {}))
    if fixture.get("residual_leak_risk_recorded") is not True:
        hard_gaps.append(item_gap(str(fixture.get("id") or "scif"), "residual_leak_risk_not_recorded", {}))
    return {
        "id": str(fixture.get("id") or ""),
        "status": "ready" if not hard_gaps else "fail_closed",
        "handles": list_values(fixture.get("handles")),
        "sanitized_outputs": list_values(fixture.get("sanitized_outputs")),
        "zeroize_policy": str(fixture.get("zeroize_policy") or ""),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def audit_deletion_closure(fixture: dict[str, Any]) -> dict[str, Any]:
    closed_states = {"sanitized", "deleted", "not_materialized", "not_admitted", "tombstoned"}
    descendants = list_dicts(fixture.get("derived_artifacts"))
    open_descendants = [row for row in descendants if str(row.get("closure_state") or "") not in closed_states]
    hard_gaps = []
    warnings = []
    if open_descendants and fixture.get("fault_on_open_descendant") is True:
        hard_gaps.append(item_gap(str(fixture.get("id") or "deletion"), "open_descendants_after_revocation", {"open": open_descendants}))
    if not descendants:
        hard_gaps.append(item_gap(str(fixture.get("id") or "deletion"), "derived_artifacts_missing", {}))
    return {
        "id": str(fixture.get("id") or ""),
        "status": "closed" if not hard_gaps else "closure_fault",
        "revoked_material": str(fixture.get("revoked_material") or ""),
        "descendant_count": len(descendants),
        "closure_fault_count": len(open_descendants),
        "derived_artifacts": descendants,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


REQUIRED_ABI_SCENARIOS = {
    "valid_leased_materialization",
    "mandatory_miss_typed_fault",
    "inadequate_verification_support",
    "mount_policy_denied",
    "lease_expired_reuse_blocked",
}
REQUIRED_ABI_FIELDS = {
    "request_id",
    "task_id",
    "semantic_address",
    "version",
    "mount",
    "snapshot_id",
    "representation_contract",
    "authority_labels",
    "admission_state",
    "adequacy_state",
    "fault_state",
    "materialization_ref",
    "audit_refs",
    "lease_state",
    "expected_decision",
}


def audit_context_abi_fixtures(fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    audited = []
    records: list[dict[str, Any]] = []
    seen_scenarios = {str(row.get("scenario") or "") for row in fixtures}
    missing_scenarios = sorted(REQUIRED_ABI_SCENARIOS - seen_scenarios)
    if missing_scenarios:
        hard_gaps.append(item_gap("context_abi", "required_scenarios_missing", {"missing": missing_scenarios}))

    for row in fixtures:
        scenario = str(row.get("scenario") or "<missing-scenario>")
        item_id = str(row.get("request_id") or scenario)
        item_gaps: list[dict[str, Any]] = []
        missing_fields = sorted(field for field in REQUIRED_ABI_FIELDS if field not in row)
        if missing_fields:
            item_gaps.append(item_gap(item_id, "context_abi_required_fields_missing", {"missing": missing_fields}))
        requested_authority = set(str(x) for x in list_values(row.get("requested_authority_labels")))
        materialized_authority = set(str(x) for x in list_values(row.get("materialized_authority_labels")))
        if materialized_authority and not materialized_authority.issubset(requested_authority):
            item_gaps.append(
                item_gap(
                    item_id,
                    "context_abi_authority_widened",
                    {
                        "requested_authority_labels": sorted(requested_authority),
                        "materialized_authority_labels": sorted(materialized_authority),
                    },
                )
            )
        materialization = str(row.get("materialization_ref") or "")
        fault = str(row.get("fault_state") or "")
        admission = str(row.get("admission_state") or "")
        adequacy = str(row.get("adequacy_state") or "")
        lease = str(row.get("lease_state") or "")
        expected = str(row.get("expected_decision") or "")
        best_effort = bool(row.get("best_effort_materialization"))

        if scenario == "valid_leased_materialization":
            if admission != "admitted" or adequacy != "adequate" or lease != "active" or not materialization or fault not in {"", "none"}:
                item_gaps.append(
                    item_gap(
                        item_id,
                        "valid_materialization_contract_failed",
                        {
                            "admission_state": admission,
                            "adequacy_state": adequacy,
                            "lease_state": lease,
                            "materialization_ref": materialization,
                            "fault_state": fault,
                        },
                    )
                )
        elif scenario == "mandatory_miss_typed_fault":
            if fault != "mandatory_miss" or materialization or expected != "typed_fault":
                item_gaps.append(item_gap(item_id, "mandatory_miss_not_typed_fault", {"fault_state": fault, "materialization_ref": materialization, "expected_decision": expected}))
        elif scenario == "inadequate_verification_support":
            if adequacy not in {"inadequate_for_verification", "inadequate"} or fault != "inadequate_verification_support" or expected != "reject_for_verification":
                item_gaps.append(item_gap(item_id, "verification_inadequacy_not_blocked", {"adequacy_state": adequacy, "fault_state": fault, "expected_decision": expected}))
        elif scenario == "mount_policy_denied":
            if admission != "denied" or fault != "mount_policy_denied" or materialization:
                item_gaps.append(item_gap(item_id, "mount_denial_not_fail_closed", {"admission_state": admission, "fault_state": fault, "materialization_ref": materialization}))
        elif scenario == "lease_expired_reuse_blocked":
            if lease != "expired" or fault != "lease_expired" or expected != "block_reuse" or materialization:
                item_gaps.append(item_gap(item_id, "expired_lease_reuse_not_blocked", {"lease_state": lease, "fault_state": fault, "expected_decision": expected, "materialization_ref": materialization}))
        else:
            warnings.append(item_gap(item_id, "unknown_context_abi_scenario", {"scenario": scenario}, severity="warning"))

        if best_effort:
            item_gaps.append(item_gap(item_id, "best_effort_materialization_forbidden", {"scenario": scenario}))
        if not list_values(row.get("audit_refs")):
            item_gaps.append(item_gap(item_id, "context_abi_audit_refs_missing", {"scenario": scenario}))

        hard_gaps.extend(item_gaps)
        passed = not item_gaps
        audited.append(
            {
                **row,
                "passed": passed,
                "hard_gaps": item_gaps,
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            }
        )
        records.extend(context_abi_records(row, passed))

    status = "ready" if not hard_gaps else "fail_closed"
    return {
        "policy": "project_theseus_context_abi_fixture_gate_v1",
        "status": status,
        "required_scenarios": sorted(REQUIRED_ABI_SCENARIOS),
        "seen_scenarios": sorted(seen_scenarios),
        "passed_count": sum(1 for row in audited if row.get("passed")),
        "fixtures": audited,
        "viea_context_abi_records": records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "non_claims": [
            "Context ABI fixtures validate resolver/fault protocol semantics only.",
            "Passing these fixtures is not a VCM benchmark score, native KV-cache parity claim, or learned-generation claim.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def context_abi_records(row: dict[str, Any], passed: bool) -> list[dict[str, Any]]:
    request_id = str(row.get("request_id") or row.get("scenario") or stable_id(row))
    scenario = str(row.get("scenario") or "")
    support_state = "SUPPORTED" if passed else "BLOCKED"
    common = {
        "task_kind": "vcm_context_abi_fixture",
        "target": "vcm_context_governor",
        "request_id": request_id,
        "scenario": scenario,
        "support_state": support_state,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }
    content_hash = stable_hash(row)
    return [
        {
            **common,
            "record_type": "context_abi_record",
            "record_id": stable_id("context_abi_record", request_id, content_hash),
            "semantic_address": row.get("semantic_address"),
            "version": row.get("version"),
            "mount": row.get("mount"),
            "snapshot_id": row.get("snapshot_id"),
            "representation_contract": row.get("representation_contract"),
            "authority_scope": ",".join(str(x) for x in list_values(row.get("authority_labels"))),
            "admission_state": row.get("admission_state"),
            "adequacy_state": row.get("adequacy_state"),
            "fault_state": row.get("fault_state"),
            "materialization_ref": row.get("materialization_ref"),
            "lease_state": row.get("lease_state"),
            "content_hash": content_hash,
        },
        {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": stable_id("context_abi_authority", request_id),
            "authority_scope": ",".join(str(x) for x in list_values(row.get("materialized_authority_labels"))),
            "allowed_effects": ["read_context_mount", "emit_context_receipt"],
            "denied_effects": ["authority_widening", "best_effort_materialization", "runtime_external_inference"],
        },
        {
            **common,
            "record_type": "context_transaction",
            "record_id": stable_id("context_abi_transaction", request_id),
            "evidence_ref": "reports/vcm_context_governor.json",
            "content_hash": content_hash,
        },
        {
            **common,
            "record_type": "context_adequacy",
            "record_id": stable_id("context_abi_adequacy", request_id),
            "state": row.get("adequacy_state"),
            "evidence_ref": "reports/vcm_context_governor.json",
        },
        {
            **common,
            "record_type": "failure_boundary",
            "record_id": stable_id("context_abi_failure", request_id),
            "failure_id": stable_id("context_abi_fault", request_id),
            "blocked_reason": row.get("fault_state") or "none",
            "terminal": bool(passed and str(row.get("fault_state") or "") in {"", "none"}),
            "structured_non_solved": str(row.get("fault_state") or "") not in {"", "none"},
        },
        {
            **common,
            "record_type": "artifact_graph_record",
            "record_id": stable_id("context_abi_artifact", request_id),
            "artifact_ref": "reports/vcm_context_governor.json",
            "evidence_ref": "configs/vcm_context_governor.json",
            "content_hash": content_hash,
        },
        {
            **common,
            "record_type": "claim_record",
            "record_id": stable_id("context_abi_claim", request_id),
            "claim_id": stable_id("context_abi_claim", scenario),
            "evidence_ref": "reports/vcm_context_governor.json",
            "learned_generation_claim_allowed": False,
        },
        {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": stable_id("context_abi_evidence", request_id),
            "previous_support_state": "UNREVIEWED",
            "current_support_state": support_state,
            "evidence_ref": "reports/vcm_context_governor.json",
        },
    ]


REQUIRED_RESOLVER_SCENARIOS = {
    "real_artifact_materialization",
    "metadata_only_private_trace",
    "mandatory_miss_typed_fault",
    "mount_policy_denied",
    "lease_expired_reuse_blocked",
    "inadequate_verification_support",
}
REQUIRED_RESOLVER_FIELDS = {
    "request_id",
    "scenario",
    "task_id",
    "semantic_address",
    "mount",
    "snapshot_id",
    "representation_contract",
    "requested_authority_labels",
    "expected_decision",
}


def audit_context_resolver_conformance(catalog_rows: list[dict[str, Any]], request_rows: list[dict[str, Any]]) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    catalog: dict[str, dict[str, Any]] = {}
    catalog_reports: list[dict[str, Any]] = []
    for row in catalog_rows:
        address = str(row.get("semantic_address") or "")
        if not address:
            hard_gaps.append(item_gap("context_resolver_catalog", "semantic_address_missing", row))
            continue
        if address in catalog:
            hard_gaps.append(item_gap(address, "duplicate_semantic_address", {"semantic_address": address}))
        catalog[address] = row
        source_path = str(row.get("source_path") or "")
        resolved = resolve(source_path) if source_path else None
        source_exists = bool(resolved and resolved.exists())
        digest = sha256_file(resolved) if source_exists and resolved is not None else ""
        catalog_reports.append(
            {
                "semantic_address": address,
                "source_path": source_path,
                "source_exists": source_exists,
                "source_sha256": digest,
                "mount": row.get("mount"),
                "lease_state": row.get("lease_state"),
                "admission_policy": row.get("admission_policy"),
                "support_state": row.get("support_state"),
                "raw_payload_stored": False,
            }
        )

    seen_scenarios = {str(row.get("scenario") or "") for row in request_rows}
    missing_scenarios = sorted(REQUIRED_RESOLVER_SCENARIOS - seen_scenarios)
    if missing_scenarios:
        hard_gaps.append(item_gap("context_resolver", "required_scenarios_missing", {"missing": missing_scenarios}))

    audited: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for row in request_rows:
        item_id = str(row.get("request_id") or row.get("semantic_address") or "context_resolver_request")
        item_gaps: list[dict[str, Any]] = []
        missing_fields = sorted(field for field in REQUIRED_RESOLVER_FIELDS if empty_value(row.get(field)))
        if missing_fields:
            item_gaps.append(item_gap(item_id, "context_resolver_required_fields_missing", {"missing": missing_fields}))

        address = str(row.get("semantic_address") or "")
        entry = catalog.get(address)
        resolved = resolve_context_request(row, entry)
        expected = str(row.get("expected_decision") or "")
        if expected and resolved["decision"] != expected:
            item_gaps.append(
                item_gap(
                    item_id,
                    "context_resolver_decision_mismatch",
                    {"expected_decision": expected, "actual_decision": resolved["decision"], "fault_state": resolved["fault_state"]},
                )
            )
        requested_authority = set(str(x) for x in list_values(row.get("requested_authority_labels")))
        materialized_authority = set(str(x) for x in list_values(resolved.get("materialized_authority_labels")))
        if materialized_authority and not materialized_authority.issubset(requested_authority):
            item_gaps.append(
                item_gap(
                    item_id,
                    "context_resolver_authority_widened",
                    {
                        "requested_authority_labels": sorted(requested_authority),
                        "materialized_authority_labels": sorted(materialized_authority),
                    },
                )
            )
        if resolved["decision"] == "materialize" and not resolved["materialization_ref"]:
            item_gaps.append(item_gap(item_id, "context_resolver_materialization_ref_missing", {"semantic_address": address}))
        if resolved["decision"] != "materialize" and resolved["materialization_ref"]:
            item_gaps.append(item_gap(item_id, "context_resolver_typed_fault_with_materialization_ref", {"semantic_address": address, "materialization_ref": resolved["materialization_ref"]}))
        if bool(row.get("best_effort_materialization")):
            item_gaps.append(item_gap(item_id, "best_effort_materialization_forbidden", {"semantic_address": address}))
        if not list_values(row.get("audit_refs")):
            item_gaps.append(item_gap(item_id, "context_resolver_audit_refs_missing", {"semantic_address": address}))

        hard_gaps.extend(item_gaps)
        passed = not item_gaps
        audited_row = {
            **row,
            **resolved,
            "passed": passed,
            "hard_gaps": item_gaps,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "raw_payload_stored": False,
            "raw_private_text_stored": False,
        }
        audited.append(audited_row)
        records.extend(context_resolver_records(audited_row, passed))

    return {
        "policy": "project_theseus_context_resolver_conformance_gate_v1",
        "status": "ready" if not hard_gaps else "fail_closed",
        "required_scenarios": sorted(REQUIRED_RESOLVER_SCENARIOS),
        "seen_scenarios": sorted(seen_scenarios),
        "catalog": catalog_reports,
        "passed_count": sum(1 for row in audited if row.get("passed")),
        "requests": audited,
        "viea_context_resolver_records": records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "non_claims": [
            "Context resolver conformance resolves real local artifact addresses and typed faults only.",
            "Resolved refs are payload hashes and paths, not raw private text or public benchmark payload training data.",
            "Passing this gate is not a VCM benchmark score, native KV-cache parity claim, or learned-generation claim.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def resolve_context_request(row: dict[str, Any], entry: dict[str, Any] | None) -> dict[str, Any]:
    address = str(row.get("semantic_address") or "")
    if entry is None:
        return resolver_fault(row, "mandatory_miss", "missing", "not_issued", "typed_fault", "")

    admission_policy = str(entry.get("admission_policy") or "admit")
    lease_state = str(entry.get("lease_state") or "active")
    source_path = str(entry.get("source_path") or "")
    source = resolve(source_path) if source_path else None
    source_exists = bool(source and source.exists())
    required_support = str(row.get("required_support_state") or "")
    support_state = str(entry.get("support_state") or "")
    representation = str(row.get("representation_contract") or "")
    available_representations = {str(x) for x in list_values(entry.get("representation_contracts"))}

    if admission_policy == "deny" or "public_benchmark/payload" in address:
        return resolver_fault(row, "mount_policy_denied", "unsafe", "not_issued", "typed_fault", "")
    if lease_state == "expired":
        return resolver_fault(row, "lease_expired", "stale", "expired", "typed_fault", "")
    if not source_exists:
        return resolver_fault(row, "mandatory_miss", "missing", "not_issued", "typed_fault", "")
    if representation and available_representations and representation not in available_representations:
        return resolver_fault(row, "representation_unavailable", "inadequate", lease_state, "typed_fault", "")
    if required_support and support_state and required_support != support_state:
        return resolver_fault(row, "inadequate_verification_support", "inadequate_for_verification", lease_state, "typed_fault", "")

    source_sha256 = sha256_file(source) if source is not None else ""
    return {
        "decision": "materialize",
        "admission_state": "admitted",
        "adequacy_state": "adequate",
        "fault_state": "none",
        "lease_state": lease_state,
        "materialization_ref": rel(source) if source is not None else "",
        "materialization_sha256": source_sha256,
        "materialization_bytes": source.stat().st_size if source is not None else 0,
        "materialized_authority_labels": list_values(entry.get("authority_labels")),
        "source_exists": source_exists,
        "catalog_support_state": support_state,
        "catalog_snapshot_id": entry.get("snapshot_id"),
        "catalog_version": entry.get("version") or ("sha256:" + source_sha256 if source_sha256 else ""),
    }


def resolver_fault(row: dict[str, Any], fault_state: str, adequacy_state: str, lease_state: str, decision: str, materialization_ref: str) -> dict[str, Any]:
    return {
        "decision": decision,
        "admission_state": "denied" if fault_state in {"mount_policy_denied", "lease_expired"} else "unresolved",
        "adequacy_state": adequacy_state,
        "fault_state": fault_state,
        "lease_state": lease_state,
        "materialization_ref": materialization_ref,
        "materialization_sha256": "",
        "materialization_bytes": 0,
        "materialized_authority_labels": [],
        "source_exists": False,
        "catalog_support_state": "",
        "catalog_snapshot_id": "",
        "catalog_version": "",
    }


def context_resolver_records(row: dict[str, Any], passed: bool) -> list[dict[str, Any]]:
    request_id = str(row.get("request_id") or row.get("scenario") or stable_id(row))
    scenario = str(row.get("scenario") or "")
    support_state = "SUPPORTED" if passed else "BLOCKED"
    content_hash = stable_hash(
        {
            "request_id": request_id,
            "semantic_address": row.get("semantic_address"),
            "decision": row.get("decision"),
            "fault_state": row.get("fault_state"),
            "materialization_ref": row.get("materialization_ref"),
            "materialization_sha256": row.get("materialization_sha256"),
            "lease_state": row.get("lease_state"),
        }
    )
    common = {
        "task_kind": "vcm_context_resolver_conformance",
        "target": "vcm_context_governor",
        "request_id": request_id,
        "scenario": scenario,
        "support_state": support_state,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }
    return [
        {
            **common,
            "record_type": "context_abi_record",
            "record_id": stable_id("context_resolver_abi", request_id, content_hash),
            "semantic_address": row.get("semantic_address"),
            "version": row.get("catalog_version") or row.get("version"),
            "mount": row.get("mount"),
            "snapshot_id": row.get("catalog_snapshot_id") or row.get("snapshot_id"),
            "representation_contract": row.get("representation_contract"),
            "authority_scope": ",".join(str(x) for x in list_values(row.get("materialized_authority_labels"))),
            "admission_state": row.get("admission_state"),
            "adequacy_state": row.get("adequacy_state"),
            "fault_state": row.get("fault_state"),
            "materialization_ref": row.get("materialization_ref"),
            "lease_state": row.get("lease_state"),
            "content_hash": content_hash,
        },
        {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": stable_id("context_resolver_authority", request_id),
            "authority_scope": ",".join(str(x) for x in list_values(row.get("materialized_authority_labels"))),
            "allowed_effects": ["read_context_mount", "emit_context_receipt", "hash_materialization_ref"],
            "denied_effects": ["payload_copy", "authority_widening", "best_effort_materialization", "runtime_external_inference"],
        },
        {
            **common,
            "record_type": "context_transaction",
            "record_id": stable_id("context_resolver_transaction", request_id),
            "transaction_id": stable_id("context_resolver_transaction", request_id),
            "operation": "resolve_semantic_address",
            "snapshot_id": row.get("catalog_snapshot_id") or row.get("snapshot_id"),
            "mounts": [row.get("mount")],
            "read_set": [row.get("semantic_address"), row.get("materialization_ref")],
            "write_set": [],
            "branch_policy": "fail_closed_no_best_effort",
            "taint_labels": list_values(row.get("taint_labels")),
            "deletion_obligations": list_values(row.get("deletion_obligations")),
            "declassification_refs": [],
            "derivative_refs": [row.get("materialization_sha256")] if row.get("materialization_sha256") else [],
            "contradiction_refs": [],
            "materialization_state": row.get("decision"),
            "closure_state": "closed" if row.get("decision") == "materialize" else "typed_fault",
            "faults": [] if row.get("fault_state") in {"", "none"} else [row.get("fault_state")],
            "audit_refs": list_values(row.get("audit_refs")) + ["reports/vcm_context_governor.json"],
            "replay_boundary": "path_and_hash_only_no_payload",
            "non_claims": [
                "resolver transaction does not store raw payload text",
                "resolver transaction is not learned-generation evidence",
            ],
            "evidence_ref": "reports/vcm_context_governor.json",
            "content_hash": content_hash,
        },
        {
            **common,
            "record_type": "context_adequacy",
            "record_id": stable_id("context_resolver_adequacy", request_id),
            "adequacy_id": stable_id("context_resolver_adequacy", request_id, row.get("adequacy_state")),
            "state": row.get("adequacy_state"),
            "adequacy_state": row.get("adequacy_state"),
            "context_transaction_id": stable_id("context_resolver_transaction", request_id),
            "evidence_ref": "reports/vcm_context_governor.json",
        },
        {
            **common,
            "record_type": "failure_boundary",
            "record_id": stable_id("context_resolver_failure", request_id),
            "failure_id": stable_id("context_resolver_fault", request_id),
            "blocked_reason": row.get("fault_state") or "none",
            "terminal": bool(passed and str(row.get("fault_state") or "") in {"", "none"}),
            "structured_non_solved": str(row.get("fault_state") or "") not in {"", "none"},
        },
        {
            **common,
            "record_type": "artifact_graph_record",
            "record_id": stable_id("context_resolver_artifact", request_id),
            "artifact_id": stable_id("context_resolver_artifact", request_id),
            "artifact_ref": row.get("materialization_ref") or "reports/vcm_context_governor.json",
            "evidence_ref": "configs/vcm_context_governor.json",
            "content_hash": content_hash,
        },
        {
            **common,
            "record_type": "claim_record",
            "record_id": stable_id("context_resolver_claim", request_id),
            "claim_id": stable_id("context_resolver_claim", scenario),
            "evidence_ref": "reports/vcm_context_governor.json",
            "learned_generation_claim_allowed": False,
        },
        {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": stable_id("context_resolver_evidence", request_id),
            "previous_support_state": "UNREVIEWED",
            "current_support_state": support_state,
            "evidence_ref": "reports/vcm_context_governor.json",
        },
    ]


def audit_representation_certificates(requests: list[dict[str, Any]]) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    certificates = []
    records: list[dict[str, Any]] = []
    for row in requests:
        certificate = build_representation_certificate(row)
        item_gaps = validate_representation_certificate(certificate)
        passed = not item_gaps
        hard_gaps.extend(item_gaps)
        audited = {**certificate, "passed": passed, "hard_gaps": item_gaps}
        certificates.append(audited)
        records.extend(representation_certificate_records(audited, passed))

    if not certificates:
        hard_gaps.append(item_gap("representation_certificate", "certificates_missing", {}))

    invalid_controls = [
        invalid_certificate_control(
            "authority_widening_rejected",
            {**certificates[0], "materialized_authority_labels": list_values(certificates[0].get("authority_ceiling")) + ["runtime_external_inference"]}
            if certificates
            else {},
        ),
        invalid_certificate_control("missing_source_refs_rejected", {**certificates[0], "source_refs": []} if certificates else {}),
        invalid_certificate_control("missing_authority_ceiling_rejected", {**certificates[0], "authority_ceiling": []} if certificates else {}),
        invalid_certificate_control(
            "best_effort_consumer_policy_rejected",
            {
                **certificates[0],
                "consumer_policy": {**dict_value(certificates[0].get("consumer_policy")), "best_effort_materialization_allowed": True},
            }
            if certificates
            else {},
        ),
    ]
    rejected_count = sum(1 for row in invalid_controls if row["rejected"])
    if rejected_count != len(invalid_controls):
        hard_gaps.append(
            item_gap(
                "representation_certificate",
                "expected_invalid_certificate_control_not_rejected",
                {"expected": len(invalid_controls), "rejected": rejected_count},
            )
        )

    return {
        "policy": "project_theseus_vcm_representation_certificate_gate_v1",
        "status": "ready" if not hard_gaps else "fail_closed",
        "certificate_count": len(certificates),
        "passed_count": sum(1 for row in certificates if row.get("passed")),
        "expected_invalid_controls": invalid_controls,
        "expected_invalid_rejected_count": rejected_count,
        "certificates": certificates,
        "viea_representation_certificate_records": records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "non_claims": [
            "Representation certificates constrain context meaning and authority only.",
            "Representation certificates are not native KV-cache parity evidence.",
            "Representation certificates are not learned-generation evidence.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_representation_certificate(row: dict[str, Any]) -> dict[str, Any]:
    request_id = str(row.get("request_id") or stable_id(row))
    decision = str(row.get("decision") or "")
    fault_state = str(row.get("fault_state") or "")
    materialization_ref = str(row.get("materialization_ref") or "")
    materialization_sha256 = str(row.get("materialization_sha256") or "")
    source_refs = [
        {
            "kind": "semantic_address",
            "ref": str(row.get("semantic_address") or ""),
            "snapshot_id": str(row.get("catalog_snapshot_id") or row.get("snapshot_id") or ""),
        }
    ]
    if materialization_ref:
        source_refs.append({"kind": "materialized_artifact", "ref": materialization_ref, "sha256": materialization_sha256})
    for ref in list_values(row.get("audit_refs")):
        source_refs.append({"kind": "audit_ref", "ref": str(ref)})

    omissions = ["raw_payload_text_not_embedded", "context_packet_contains_refs_and_hashes_only"]
    if row.get("raw_private_text_stored") is False:
        omissions.append("raw_private_text_not_staged")
    if decision != "materialize":
        omissions.append(f"payload_not_materialized_due_to_{fault_state or 'typed_fault'}")
    if list_values(row.get("deletion_obligations")):
        omissions.append("deletion_obligations_preserved")

    authority_ceiling = list_values(row.get("requested_authority_labels"))
    materialized_authority = list_values(row.get("materialized_authority_labels"))
    representation_contract = str(row.get("representation_contract") or "")
    permitted_uses = ["typed_fault_handling", "audit_replay"]
    if decision == "materialize":
        permitted_uses = ["context_read", "evidence_pointer", "audit_replay"]
        if "metadata_only" in representation_contract:
            permitted_uses.append("metadata_accounting")
        if "summary_only" in representation_contract:
            permitted_uses.append("summary_context_only")

    return {
        "certificate_id": stable_id("vcm_representation_certificate", request_id),
        "request_id": request_id,
        "scenario": row.get("scenario"),
        "semantic_address": row.get("semantic_address"),
        "snapshot_id": row.get("catalog_snapshot_id") or row.get("snapshot_id"),
        "version": row.get("catalog_version") or row.get("version"),
        "mount": row.get("mount"),
        "source_refs": source_refs,
        "omissions": omissions,
        "loss_contract": {
            "representation_contract": representation_contract,
            "payload_boundary": "path_and_hash_refs_only_no_raw_payload",
            "summary_can_raise_authority": False,
            "compression_must_preserve_taint_and_omissions": True,
            "fault_state": fault_state,
            "materialization_state": decision,
        },
        "permitted_uses": permitted_uses,
        "authority_ceiling": authority_ceiling,
        "materialized_authority_labels": materialized_authority,
        "consumer_policy": {
            "fail_closed_on_missing": True,
            "fail_closed_on_stale": True,
            "fail_closed_on_tainted_for_training": True,
            "best_effort_materialization_allowed": False,
            "authority_widening_allowed": False,
            "public_benchmark_payload_training_allowed": False,
            "raw_private_text_materialization_allowed": False,
        },
        "typed_fault": fault_state if fault_state not in {"", "none"} else "",
        "audit_refs": list_values(row.get("audit_refs")) + ["reports/vcm_context_governor.json"],
        "taint_labels": list_values(row.get("taint_labels")),
        "deletion_obligations": list_values(row.get("deletion_obligations")),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_private_text_stored": False,
        "raw_prompt_stored": False,
        "non_claims": [
            "certificate constrains representation and authority only",
            "not learned-generation evidence",
            "not native KV-cache parity evidence",
        ],
    }


def validate_representation_certificate(row: dict[str, Any]) -> list[dict[str, Any]]:
    item_id = str(row.get("certificate_id") or row.get("request_id") or "representation_certificate")
    gaps: list[dict[str, Any]] = []
    if not list_values(row.get("source_refs")):
        gaps.append(item_gap(item_id, "representation_certificate_source_refs_missing", {}))
    if not list_values(row.get("omissions")):
        gaps.append(item_gap(item_id, "representation_certificate_omissions_missing", {}))
    if not dict_value(row.get("loss_contract")):
        gaps.append(item_gap(item_id, "representation_certificate_loss_contract_missing", {}))
    if not list_values(row.get("permitted_uses")):
        gaps.append(item_gap(item_id, "representation_certificate_permitted_uses_missing", {}))
    authority_ceiling = set(str(x) for x in list_values(row.get("authority_ceiling")))
    materialized_authority = set(str(x) for x in list_values(row.get("materialized_authority_labels")))
    if not authority_ceiling:
        gaps.append(item_gap(item_id, "representation_certificate_authority_ceiling_missing", {}))
    if materialized_authority and not materialized_authority.issubset(authority_ceiling):
        gaps.append(
            item_gap(
                item_id,
                "representation_certificate_authority_widened",
                {
                    "authority_ceiling": sorted(authority_ceiling),
                    "materialized_authority_labels": sorted(materialized_authority),
                },
            )
        )
    policy = dict_value(row.get("consumer_policy"))
    required_policy = {
        "fail_closed_on_missing": True,
        "best_effort_materialization_allowed": False,
        "authority_widening_allowed": False,
        "public_benchmark_payload_training_allowed": False,
        "raw_private_text_materialization_allowed": False,
    }
    for key, expected in required_policy.items():
        if policy.get(key) is not expected:
            gaps.append(item_gap(item_id, f"representation_certificate_policy_{key}_invalid", {"expected": expected, "actual": policy.get(key)}))
    if dict_value(row.get("loss_contract")).get("summary_can_raise_authority") is not False:
        gaps.append(item_gap(item_id, "representation_certificate_summary_can_raise_authority", {}))
    if any(int_or(row.get(key)) != 0 for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")):
        gaps.append(item_gap(item_id, "representation_certificate_no_cheat_counter_fault", {}))
    if row.get("raw_private_text_stored") is not False or row.get("raw_prompt_stored") is not False:
        gaps.append(item_gap(item_id, "representation_certificate_raw_text_stored", {}))
    return gaps


def invalid_certificate_control(control: str, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "control": control,
        "rejected": bool(validate_representation_certificate(candidate)),
        "candidate_certificate_id": candidate.get("certificate_id"),
    }


def representation_certificate_records(row: dict[str, Any], passed: bool) -> list[dict[str, Any]]:
    request_id = str(row.get("request_id") or row.get("certificate_id") or stable_id(row))
    support_state = "SUPPORTED" if passed else "BLOCKED"
    content_hash = stable_hash(row)
    common = {
        "task_kind": "vcm_representation_certificate",
        "target": "vcm_context_governor",
        "request_id": request_id,
        "scenario": row.get("scenario"),
        "support_state": support_state,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }
    return [
        {
            **common,
            "record_type": "context_abi_record",
            "record_id": stable_id("representation_certificate_abi", request_id, content_hash),
            "certificate_id": row.get("certificate_id"),
            "semantic_address": row.get("semantic_address"),
            "version": row.get("version"),
            "mount": row.get("mount"),
            "snapshot_id": row.get("snapshot_id"),
            "representation_contract": dict_value(row.get("loss_contract")).get("representation_contract"),
            "authority_scope": ",".join(str(x) for x in list_values(row.get("authority_ceiling"))),
            "admission_state": "certified" if passed else "blocked",
            "adequacy_state": "certificate_ready" if passed else "certificate_fault",
            "fault_state": row.get("typed_fault") or "none",
            "materialization_ref": next((src.get("ref") for src in list_dicts(row.get("source_refs")) if src.get("kind") == "materialized_artifact"), ""),
            "lease_state": "certificate_only",
            "content_hash": content_hash,
        },
        {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": stable_id("representation_certificate_authority", request_id),
            "authority_scope": ",".join(str(x) for x in list_values(row.get("authority_ceiling"))),
            "allowed_effects": ["read_context_refs", "emit_representation_certificate"],
            "denied_effects": ["authority_widening", "raw_payload_copy", "runtime_external_inference"],
        },
        {
            **common,
            "record_type": "claim_record",
            "record_id": stable_id("representation_certificate_claim", request_id),
            "claim_id": stable_id("representation_certificate_claim", row.get("scenario")),
            "evidence_ref": "reports/vcm_context_governor.json",
            "learned_generation_claim_allowed": False,
        },
        {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": stable_id("representation_certificate_evidence", request_id),
            "previous_support_state": "UNREVIEWED",
            "current_support_state": support_state,
            "evidence_ref": "reports/vcm_context_governor.json",
        },
    ]


def audit_snapshot_branch_ledger(requests: list[dict[str, Any]]) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    branches = []
    records: list[dict[str, Any]] = []
    for row in requests:
        branch = build_snapshot_branch(row)
        item_gaps = validate_snapshot_branch(branch)
        passed = not item_gaps
        hard_gaps.extend(item_gaps)
        audited = {**branch, "passed": passed, "hard_gaps": item_gaps}
        branches.append(audited)
        records.extend(snapshot_branch_records(audited, passed))

    if not branches:
        hard_gaps.append(item_gap("snapshot_branch", "snapshot_branches_missing", {}))

    invalid_controls = [
        invalid_snapshot_branch_control(
            "taint_drop_rejected",
            {**branches[0], "propagated_taint_labels": []} if branches else {},
        ),
        invalid_snapshot_branch_control(
            "source_mutation_rejected",
            {**branches[0], "write_set": list_values(branches[0].get("read_set")), "source_mutation_allowed": True} if branches else {},
        ),
        invalid_snapshot_branch_control(
            "public_payload_materialization_rejected",
            {
                **next((row for row in branches if "public_benchmark" in str(row.get("semantic_address"))), branches[0] if branches else {}),
                "materialization_state": "materialize",
                "faults": [],
            }
            if branches
            else {},
        ),
        invalid_snapshot_branch_control(
            "mandatory_miss_without_fault_rejected",
            {
                **next((row for row in branches if row.get("scenario") == "mandatory_miss_typed_fault"), branches[0] if branches else {}),
                "faults": [],
                "materialization_state": "typed_fault",
            }
            if branches
            else {},
        ),
    ]
    rejected_count = sum(1 for row in invalid_controls if row["rejected"])
    if rejected_count != len(invalid_controls):
        hard_gaps.append(
            item_gap(
                "snapshot_branch",
                "expected_invalid_snapshot_branch_control_not_rejected",
                {"expected": len(invalid_controls), "rejected": rejected_count},
            )
        )

    return {
        "policy": "project_theseus_vcm_snapshot_branch_ledger_v1",
        "status": "ready" if not hard_gaps else "fail_closed",
        "branch_count": len(branches),
        "passed_count": sum(1 for row in branches if row.get("passed")),
        "expected_invalid_controls": invalid_controls,
        "expected_invalid_rejected_count": rejected_count,
        "branches": branches,
        "viea_snapshot_branch_records": records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "non_claims": [
            "Snapshot branches prove context transaction discipline only.",
            "Snapshot branches do not prove native KV-cache parity.",
            "Snapshot branches do not credit learned generation.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_snapshot_branch(row: dict[str, Any]) -> dict[str, Any]:
    request_id = str(row.get("request_id") or stable_id(row))
    parent_snapshot = str(row.get("catalog_snapshot_id") or row.get("snapshot_id") or "snapshot://unknown")
    materialization_ref = str(row.get("materialization_ref") or "")
    fault_state = str(row.get("fault_state") or "")
    faults = [] if fault_state in {"", "none"} else [fault_state]
    taints = list_values(row.get("taint_labels"))
    return {
        "branch_id": stable_id("vcm_snapshot_branch", request_id),
        "request_id": request_id,
        "scenario": row.get("scenario"),
        "semantic_address": row.get("semantic_address"),
        "parent_snapshot_id": parent_snapshot,
        "branch_snapshot_id": f"{parent_snapshot}/branch/{stable_id(request_id, row.get('decision'), fault_state)}",
        "operation": "resolve_semantic_address",
        "copy_on_write": True,
        "source_mutation_allowed": False,
        "read_set": [item for item in [row.get("semantic_address"), materialization_ref] if item],
        "write_set": [],
        "mounts": [row.get("mount")],
        "branch_policy": "copy_on_write_fail_closed_no_best_effort",
        "taint_labels": taints,
        "propagated_taint_labels": taints,
        "deletion_obligations": list_values(row.get("deletion_obligations")),
        "contradiction_refs": [],
        "declassification_refs": [],
        "derivative_refs": [row.get("materialization_sha256")] if row.get("materialization_sha256") else [],
        "materialization_state": row.get("decision"),
        "closure_state": "closed" if row.get("decision") == "materialize" else "typed_fault",
        "faults": faults,
        "audit_refs": list_values(row.get("audit_refs")) + ["reports/vcm_context_governor.json"],
        "replay_boundary": "path_and_hash_only_no_payload",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_private_text_stored": False,
        "raw_prompt_stored": False,
        "non_claims": [
            "copy-on-write branch records context discipline only",
            "not learned-generation evidence",
            "not native KV-cache parity evidence",
        ],
    }


def validate_snapshot_branch(row: dict[str, Any]) -> list[dict[str, Any]]:
    item_id = str(row.get("branch_id") or row.get("request_id") or "snapshot_branch")
    gaps: list[dict[str, Any]] = []
    if not row.get("parent_snapshot_id") or not row.get("branch_snapshot_id"):
        gaps.append(item_gap(item_id, "snapshot_branch_ids_missing", {}))
    if row.get("copy_on_write") is not True:
        gaps.append(item_gap(item_id, "snapshot_branch_copy_on_write_missing", {}))
    if row.get("source_mutation_allowed") is not False:
        gaps.append(item_gap(item_id, "snapshot_branch_source_mutation_allowed", {}))
    read_set = {str(x) for x in list_values(row.get("read_set"))}
    write_set = {str(x) for x in list_values(row.get("write_set"))}
    if not read_set:
        gaps.append(item_gap(item_id, "snapshot_branch_read_set_missing", {}))
    if write_set.intersection(read_set):
        gaps.append(item_gap(item_id, "snapshot_branch_mutates_source_read_set", {"intersection": sorted(write_set.intersection(read_set))}))
    taints = {str(x) for x in list_values(row.get("taint_labels"))}
    propagated = {str(x) for x in list_values(row.get("propagated_taint_labels"))}
    if taints and not taints.issubset(propagated):
        gaps.append(item_gap(item_id, "snapshot_branch_taint_not_propagated", {"taint_labels": sorted(taints), "propagated": sorted(propagated)}))
    materialization_state = str(row.get("materialization_state") or "")
    faults = list_values(row.get("faults"))
    if materialization_state != "materialize" and not faults:
        gaps.append(item_gap(item_id, "snapshot_branch_typed_fault_missing", {"materialization_state": materialization_state}))
    if "public_benchmark" in str(row.get("semantic_address") or "") and materialization_state == "materialize":
        gaps.append(item_gap(item_id, "snapshot_branch_public_payload_materialized", {"semantic_address": row.get("semantic_address")}))
    if str(row.get("closure_state") or "") not in {"closed", "typed_fault"}:
        gaps.append(item_gap(item_id, "snapshot_branch_closure_state_invalid", {"closure_state": row.get("closure_state")}))
    if any(int_or(row.get(key)) != 0 for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")):
        gaps.append(item_gap(item_id, "snapshot_branch_no_cheat_counter_fault", {}))
    if row.get("raw_private_text_stored") is not False or row.get("raw_prompt_stored") is not False:
        gaps.append(item_gap(item_id, "snapshot_branch_raw_text_stored", {}))
    return gaps


def invalid_snapshot_branch_control(control: str, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "control": control,
        "rejected": bool(validate_snapshot_branch(candidate)),
        "candidate_branch_id": candidate.get("branch_id"),
    }


def snapshot_branch_records(row: dict[str, Any], passed: bool) -> list[dict[str, Any]]:
    request_id = str(row.get("request_id") or row.get("branch_id") or stable_id(row))
    support_state = "SUPPORTED" if passed else "BLOCKED"
    content_hash = stable_hash(row)
    common = {
        "task_kind": "vcm_snapshot_branch",
        "target": "vcm_context_governor",
        "request_id": request_id,
        "scenario": row.get("scenario"),
        "support_state": support_state,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }
    return [
        {
            **common,
            "record_type": "context_transaction",
            "record_id": stable_id("snapshot_branch_transaction", request_id, content_hash),
            "transaction_id": row.get("branch_id"),
            "operation": row.get("operation"),
            "snapshot_id": row.get("branch_snapshot_id"),
            "parent_snapshot_id": row.get("parent_snapshot_id"),
            "mounts": row.get("mounts"),
            "read_set": row.get("read_set"),
            "write_set": row.get("write_set"),
            "branch_policy": row.get("branch_policy"),
            "taint_labels": row.get("propagated_taint_labels"),
            "deletion_obligations": row.get("deletion_obligations"),
            "declassification_refs": row.get("declassification_refs"),
            "derivative_refs": row.get("derivative_refs"),
            "contradiction_refs": row.get("contradiction_refs"),
            "materialization_state": row.get("materialization_state"),
            "closure_state": row.get("closure_state"),
            "faults": row.get("faults"),
            "audit_refs": row.get("audit_refs"),
            "replay_boundary": row.get("replay_boundary"),
            "non_claims": row.get("non_claims"),
            "content_hash": content_hash,
        },
        {
            **common,
            "record_type": "context_adequacy",
            "record_id": stable_id("snapshot_branch_adequacy", request_id),
            "adequacy_id": stable_id("snapshot_branch_adequacy", request_id, row.get("closure_state")),
            "state": "adequate" if row.get("closure_state") == "closed" else "typed_fault",
            "adequacy_state": "adequate" if row.get("closure_state") == "closed" else "typed_fault",
            "context_transaction_id": row.get("branch_id"),
            "evidence_ref": "reports/vcm_context_governor.json",
        },
        {
            **common,
            "record_type": "failure_boundary",
            "record_id": stable_id("snapshot_branch_failure", request_id),
            "failure_id": stable_id("snapshot_branch_fault", request_id),
            "blocked_reason": ",".join(str(x) for x in list_values(row.get("faults"))) or "none",
            "terminal": bool(passed),
            "structured_non_solved": bool(list_values(row.get("faults"))),
        },
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VCM Context Governor",
        "",
        f"- trigger_state: `{report['trigger_state']}`",
        f"- chunks: `{report['summary']['chunk_count']}` pinned `{report['summary']['pinned_chunk_count']}` fail_closed `{report['summary']['fail_closed_chunk_count']}`",
        f"- mission brief: `{report['summary']['mission_brief_status']}` omissions `{report['summary']['mission_brief_omission_count']}` compression_loss `{report['summary']['mission_brief_compression_loss']}`",
        f"- SCIF: `{report['summary']['scif_status']}`",
        f"- deletion closure: `{report['summary']['deletion_closure_status']}` faults `{report['summary']['deletion_closure_fault_count']}`",
        f"- context ABI fixtures: `{report['summary']['context_abi_fixture_status']}` passed `{report['summary']['context_abi_fixture_passed_count']}/{report['summary']['context_abi_fixture_count']}` records `{report['summary']['context_abi_viea_record_count']}`",
        f"- deployed resolver: `{report['summary']['context_resolver_status']}` passed `{report['summary']['context_resolver_passed_count']}/{report['summary']['context_resolver_request_count']}` records `{report['summary']['context_resolver_viea_record_count']}` materialized `{report['summary']['context_resolver_materialized_count']}` typed_faults `{report['summary']['context_resolver_typed_fault_count']}`",
        f"- hard gaps: `{report['summary']['hard_gap_count']}` warnings: `{report['summary']['warning_count']}`",
        "",
        "## Chunk Decisions",
        "",
    ]
    for chunk in report["chunks"]:
        lines.append(f"- `{chunk['id']}` decision=`{chunk['context_decision']}` value=`{chunk['value_score']}` taint=`{chunk['taint']}` clearance=`{chunk['clearance']}`")
    lines.extend(["", "## Hard Gaps", ""])
    if report["hard_gaps"]:
        for item in report["hard_gaps"]:
            lines.append(f"- `{item['id']}` `{item['kind']}`: `{json.dumps(item['evidence'], sort_keys=True)}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Rules", ""])
    for key, value in report["rules"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report["policy"],
        "created_utc": report["created_utc"],
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"],
        "warnings": report["warnings"],
    }


def item_gap(item_id: str, kind: str, evidence: dict[str, Any], severity: str = "hard") -> dict[str, Any]:
    return {"id": item_id, "kind": kind, "severity": severity, "evidence": evidence}


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"id": name, "kind": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def bounded(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if math.isnan(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def stable_id(*parts: Any) -> str:
    return hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def list_values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def empty_value(value: Any) -> bool:
    return value in (None, "", [], {})


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
