#!/usr/bin/env python3
"""Integrity kernel for the canonical Theseus report evidence store.

This module contains the proof/claim mechanics that are orthogonal to SQLite
storage: epistemic TCB records, public-safe evidence packs, material-claim
dependency revision, and adversarial receipt replay.  It is imported by
``report_evidence_store.py``; it is not an alternate report lane.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
NO_CHEAT_KEYS = (
    "public_training_rows_written",
    "external_inference_calls",
    "fallback_return_count",
)
POSITIVE_SUPPORT_STATES = {
    "prototype-backed",
    "synthetic-test-backed",
    "empirical-test-backed",
    "replayable-reference-backed",
    "SUPPORTED",
}
CLAIM_RECORD_TYPES = {
    "claim_record",
    "claim_ledger_entry",
    "proof_carrying_claim",
    "capability_claim_disposition",
}


def citeable_green_report_paths(
    registry: dict[str, Any],
    roadmap_matrix: dict[str, Any],
) -> list[Path]:
    """Resolve the reports the live registry/roadmap actually cites as gates."""

    refs: set[str] = set()
    for contract in list_dicts(registry.get("route_evidence_contracts")):
        for requirement in list_dicts(contract.get("requirements")):
            add_report_ref(refs, requirement.get("path"))
    for implementation in list_dicts(registry.get("implementations")):
        for value in list_values(implementation.get("evidence_outputs")):
            add_report_ref(refs, value)
    for phase in list_dicts(roadmap_matrix.get("phases")):
        for value in list_values(phase.get("current_evidence")):
            add_report_ref(refs, value)

    paths: list[Path] = []
    for ref in sorted(refs):
        path = resolve(ref)
        payload = read_json(path, {})
        if path.is_file() and is_green_gate(payload):
            paths.append(path)
    return paths


def command_index(
    registry: dict[str, Any],
    roadmap_matrix: dict[str, Any],
) -> dict[str, list[str]]:
    index: dict[str, set[str]] = {}
    for implementation in list_dicts(registry.get("implementations")):
        command = str(implementation.get("verification_command") or "").strip()
        if not command:
            continue
        for output in list_values(implementation.get("evidence_outputs")):
            ref = normalize_report_ref(output)
            if ref:
                index.setdefault(ref, set()).add(command)
    for phase in list_dicts(roadmap_matrix.get("phases")):
        commands = [str(value) for value in list_values(phase.get("required_gates")) if str(value).strip()]
        commands.extend(
            str(value)
            for value in list_values(phase.get("integration_smoke"))
            if str(value).lstrip().startswith(("python", "cargo", "bash"))
        )
        for output in list_values(phase.get("current_evidence")):
            ref = normalize_report_ref(output)
            if ref:
                index.setdefault(ref, set()).update(commands)
    return {key: sorted(values) for key, values in index.items()}


def build_standard_evidence_packs(
    paths: Iterable[Path],
    *,
    commands: dict[str, list[str]],
) -> list[dict[str, Any]]:
    packs: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=lambda item: rel(item)):
        payload = read_json(path, {})
        if not is_green_gate(payload):
            continue
        raw = canonical_bytes(payload)
        source_ref = rel(path)
        non_claims = normalized_strings(payload.get("non_claims") or payload.get("non_claim"))
        if not non_claims:
            non_claims = [
                "A GREEN gate supports only the named gate contract; it is not an unscoped capability claim.",
                "This evidence pack contains digests and compact receipts, not private report payloads.",
            ]
        support_state = explicit_support_state(payload)
        source_counters = top_level_no_cheat_counters(payload)
        pack = {
            "record_type": "public_safe_evidence_pack",
            "pack_id": stable_id("evidence_pack", source_ref, hashlib.sha256(raw).hexdigest()),
            "source_path": source_ref,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_policy": str(payload.get("policy") or ""),
            "trigger_state": str(payload.get("trigger_state") or "GREEN").upper(),
            "support_state": support_state,
            "commands": commands.get(source_ref, []),
            "baseline_receipt": compact_named_sections(payload, ("baseline", "control", "comparison")),
            "negative_control_receipt": compact_named_sections(
                payload,
                ("negative", "adversarial", "expected_invalid", "trap", "control"),
            ),
            "residual_receipt": compact_named_sections(
                payload,
                ("residual", "hard_gap", "warning", "blocker", "failure"),
            ),
            "non_claims": non_claims,
            "claim_boundaries": non_claims,
            "private_payload_copied": False,
            "source_activity_counters": source_counters,
            "public_safe": int(source_counters["public_training_rows_written"]) == 0,
            "content_projection": "digest_commands_counts_and_claim_boundaries_only",
            **zero_no_cheat_counters(),
        }
        pack["validation_state"] = "GREEN" if validate_evidence_pack(pack) else "RED"
        packs.append(pack)
    return packs


def validate_evidence_pack(pack: dict[str, Any]) -> bool:
    required = {
        "pack_id",
        "source_path",
        "source_sha256",
        "trigger_state",
        "support_state",
        "commands",
        "baseline_receipt",
        "negative_control_receipt",
        "residual_receipt",
        "non_claims",
    }
    if not required.issubset(pack):
        return False
    path = resolve(str(pack.get("source_path") or ""))
    payload = read_json(path, {})
    if not path.is_file() or not payload:
        return False
    if hashlib.sha256(canonical_bytes(payload)).hexdigest() != str(pack.get("source_sha256") or ""):
        return False
    if str(payload.get("trigger_state") or "").upper() != str(pack.get("trigger_state") or "").upper():
        return False
    if str(pack.get("trigger_state") or "").upper() != "GREEN":
        return False
    if not pack.get("public_safe") or pack.get("private_payload_copied"):
        return False
    if not list_values(pack.get("non_claims")):
        return False
    return all(int_or(pack.get(key)) == 0 for key in NO_CHEAT_KEYS)


def build_epistemic_tcb(*, rotation_epoch: str | None = None) -> dict[str, Any]:
    epoch = rotation_epoch or datetime.now(timezone.utc).strftime("%G-W%V")
    roots = [
        trust_root("evidence_ledger", "scripts/report_evidence_store.py", "append-only run ledger and digest writer"),
        trust_root("receipt_integrity", "scripts/report_evidence_integrity.py", "receipt replay and claim dependency auditor"),
        trust_root("blind_flow", "scripts/blind_information_flow_audit.py", "blind information-flow auditor"),
        trust_root("candidate_integrity", "scripts/candidate_integrity.py", "independent candidate provenance auditor"),
        trust_root("private_verifier", "scripts/code_lm_private_verifier.py", "private functional verifier harness"),
        trust_root("spine_verifier", "scripts/viea_spine_record_gate.py", "cross-component VIEA record verifier"),
        trust_root("retention_replay", "scripts/theseus_artifact_retention_replay_gate.py", "archive/pointer exact replay verifier"),
        trust_root("scf_kernel", "scripts/scf_reference_kernel_v1.py", "route authority and stable-field verifier"),
    ]
    root_by_id = {row["root_id"]: row for row in roots}
    assignments = [
        rotated_assignment(epoch, "evidence_ledger", ["spine_verifier", "retention_replay"], root_by_id),
        rotated_assignment(epoch, "private_verifier", ["candidate_integrity", "blind_flow"], root_by_id),
        rotated_assignment(epoch, "candidate_integrity", ["blind_flow", "spine_verifier"], root_by_id),
        rotated_assignment(epoch, "spine_verifier", ["scf_kernel", "receipt_integrity"], root_by_id),
        rotated_assignment(epoch, "retention_replay", ["receipt_integrity", "spine_verifier"], root_by_id),
    ]
    errors = validate_tcb(roots, assignments)
    return {
        "record_type": "epistemic_trusted_computing_base",
        "policy": "project_theseus_epistemic_tcb_v1",
        "state": "GREEN" if not errors else "RED",
        "support_state": "synthetic-test-backed" if not errors else "unsupported",
        "rotation_epoch": epoch,
        "trust_roots": roots,
        "audit_assignments": assignments,
        "validation_errors": errors,
        "trust_boundary": {
            "model_and_candidate_outputs_are_untrusted": True,
            "self_declared_pass_or_provenance_flags_are_untrusted": True,
            "digest_writer_is_not_its_own_only_auditor": True,
            "high_consequence_verifiers_have_primary_and_shadow_auditors": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "non_claims": [
            "The TCB inventory names trusted mechanisms; it does not prove those mechanisms bug-free.",
            "Rotated independent replay reduces common-mode error but does not establish formal verification.",
        ],
        **zero_no_cheat_counters(),
    }


def validate_tcb(roots: list[dict[str, Any]], assignments: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [str(row.get("root_id") or "") for row in roots]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_trust_root_id")
    known = set(ids)
    for row in roots:
        if not row.get("exists") or not row.get("sha256"):
            errors.append(f"missing_trust_root:{row.get('root_id')}")
    for assignment in assignments:
        subject = str(assignment.get("subject_root_id") or "")
        auditors = [str(value) for value in list_values(assignment.get("auditor_root_ids"))]
        if subject not in known or any(auditor not in known for auditor in auditors):
            errors.append(f"unknown_audit_root:{subject}")
        if subject in auditors:
            errors.append(f"self_audit:{subject}")
        if len(set(auditors)) < 2:
            errors.append(f"insufficient_independent_auditors:{subject}")
    return sorted(set(errors))


def build_material_claim_revision(
    paths: Iterable[Path],
    evidence_packs: list[dict[str, Any]],
    *,
    stored_claim_versions: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    payloads = [(path, read_json(path, {})) for path in sorted(set(paths), key=lambda item: rel(item))]
    claims: dict[str, dict[str, Any]] = {}
    duplicate_claim_ids: set[str] = set()
    for row in stored_claim_versions:
        merge_claim_projection(claims, duplicate_claim_ids, dict(row))
    for path, payload in payloads:
        for row in extract_material_claims(payload, source_path=rel(path)):
            merge_claim_projection(claims, duplicate_claim_ids, row)

    dependency_targets = set(claims)
    for claim in claims.values():
        dependency_targets.update(normalized_strings(claim.get("evidence_refs")))
    string_index: dict[str, set[str]] = {}
    for path, payload in payloads:
        for value in walk_strings(payload):
            if value in dependency_targets:
                string_index.setdefault(value, set()).add(rel(path))
    pack_paths = {str(pack.get("source_path") or ""): pack for pack in evidence_packs}
    revisions: list[dict[str, Any]] = []
    dependency_edges: list[dict[str, str]] = []
    for claim_id, claim in sorted(claims.items()):
        refs = normalized_strings(claim.get("evidence_refs"))
        dependents = set(string_index.get(claim_id, set()))
        for ref in refs:
            dependents.update(string_index.get(ref, set()))
        dependents.difference_update(normalized_strings(claim.get("source_paths")))
        for surface in sorted(dependents):
            dependency_edges.append({"claim_id": claim_id, "dependent_surface": surface})
        failed_refs = [ref for ref in refs if evidence_ref_state(ref, pack_paths) in {"missing", "red"}]
        if failed_refs and str(claim.get("support_state") or "") in POSITIVE_SUPPORT_STATES:
            revisions.append(
                {
                    "record_type": "claim_revision_transition",
                    "transition_id": stable_id("claim_revision", claim_id, failed_refs),
                    "transition_type": "downgrade",
                    "claim_id": claim_id,
                    "from_state": claim.get("support_state"),
                    "to_state": "unsupported",
                    "reason": "one_or_more_evidence_dependencies_are_missing_or_red",
                    "failed_evidence_refs": failed_refs,
                    "dependent_surface_refs": sorted(dependents),
                    "revision_history_policy": "append_only_never_overwrite",
                    "non_claims": ["dependency invalidation only", "not new capability evidence"],
                    **zero_no_cheat_counters(),
                }
            )
    cycles = claim_dependency_cycles(claims)
    errors = []
    if duplicate_claim_ids:
        errors.append("conflicting_duplicate_claim_ids")
    if cycles:
        errors.append("claim_dependency_cycles")
    family_records: list[dict[str, Any]] = []
    all_source_paths = sorted(
        {
            source_path
            for claim in claims.values()
            for source_path in normalized_strings(claim.get("source_paths"))
        }
    )
    for source_path in all_source_paths:
        source_claims = [
            claim
            for claim in claims.values()
            if source_path in normalized_strings(claim.get("source_paths"))
        ]
        source_ids = {claim["claim_id"] for claim in source_claims}
        family_records.append(
            {
                "source_path": source_path,
                "claim_count": len(source_claims),
                "dependent_surface_edge_count": sum(1 for edge in dependency_edges if edge["claim_id"] in source_ids),
                "invalidation_count": sum(1 for row in revisions if row["claim_id"] in source_ids),
            }
        )
    full_index_digest = hashlib.sha256(
        canonical_bytes(
            {
                "claims": list(claims.values()),
                "dependency_edges": dependency_edges,
                "revision_transitions": revisions,
            }
        )
    ).hexdigest()
    return {
        "record_type": "material_claim_revision_index",
        "policy": "project_theseus_material_claim_revision_v1",
        "state": "GREEN" if not errors else "RED",
        "support_state": "synthetic-test-backed" if not errors else "unsupported",
        "material_claim_count": len(claims),
        "claim_emitting_run_family_count": len(all_source_paths),
        "dependent_surface_edge_count": len(dependency_edges),
        "dependent_invalidation_count": len(revisions),
        "duplicate_claim_ids": sorted(duplicate_claim_ids),
        "dependency_cycles": cycles,
        "validation_errors": errors,
        "claim_index_sha256": full_index_digest,
        "claim_family_records": family_records,
        "claim_samples": list(claims.values())[:100],
        "dependency_edge_samples": dependency_edges[:500],
        "revision_transitions": revisions,
        "revision_history_policy": "append_only_never_overwrite",
        "non_claims": [
            "Dependency discovery records claim impact; it does not upgrade the underlying claim.",
            "Missing or RED dependencies can only downgrade support, never synthesize support.",
        ],
        **zero_no_cheat_counters(),
    }


def extract_material_claims(payload: Any, *, source_path: str) -> list[dict[str, Any]]:
    claims: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for raw in walk_dicts(payload):
        if not looks_like_claim(raw):
            continue
        claim_id = str(raw.get("claim_id") or raw.get("record_id") or stable_id("claim", source_path, raw))
        merge_claim_projection(
            claims,
            duplicates,
            normalize_claim(claim_id, raw, source_path=source_path),
        )
    return list(claims.values())


def merge_claim_projection(
    claims: dict[str, dict[str, Any]],
    conflicting_ids: set[str],
    row: dict[str, Any],
) -> None:
    claim_id = str(row.get("claim_id") or "")
    if not claim_id:
        return
    if claim_id not in claims:
        claims[claim_id] = row
        return
    existing = claims[claim_id]
    existing_text = str(existing.get("claim") or "")
    row_text = str(row.get("claim") or "")
    if existing_text and row_text and existing_text != row_text:
        conflicting_ids.add(claim_id)
        return
    existing["claim"] = existing_text or row_text
    existing["source_paths"] = sorted(
        set(normalized_strings(existing.get("source_paths")))
        | set(normalized_strings(row.get("source_paths")))
    )
    existing["evidence_refs"] = sorted(
        set(normalized_strings(existing.get("evidence_refs")))
        | set(normalized_strings(row.get("evidence_refs")))
    )
    existing["claim_dependency_refs"] = sorted(
        set(normalized_strings(existing.get("claim_dependency_refs")))
        | set(normalized_strings(row.get("claim_dependency_refs")))
    )
    existing["support_state"] = weaker_support_state(
        str(existing.get("support_state") or "argument"),
        str(row.get("support_state") or "argument"),
    )


def audit_receipt_faithfulness(
    evidence_packs: list[dict[str, Any]],
    claim_revision: dict[str, Any],
    tcb: dict[str, Any],
    *,
    sample_size: int = 64,
    seed: int = 1407,
) -> dict[str, Any]:
    ordered = sorted(evidence_packs, key=lambda row: str(row.get("source_sha256") or ""))
    rng = random.Random(int(seed))
    sample = rng.sample(ordered, min(max(0, int(sample_size)), len(ordered))) if ordered else []
    replay_checks = [
        {
            "pack_id": pack.get("pack_id"),
            "source_path": pack.get("source_path"),
            "passed": validate_evidence_pack(pack),
        }
        for pack in sample
    ]
    base = dict(sample[0]) if sample else {}
    traps = [
        trap("digest_mismatch", validate_evidence_pack, {**base, "source_sha256": "0" * 64}),
        trap("missing_source", validate_evidence_pack, {**base, "source_path": "reports/does_not_exist.json"}),
        trap("trigger_mismatch", validate_evidence_pack, {**base, "trigger_state": "RED"}),
        trap("private_payload_copy", validate_evidence_pack, {**base, "private_payload_copied": True}),
        trap("no_cheat_counter_fault", validate_evidence_pack, {**base, "external_inference_calls": 1}),
    ] if base else []
    bad_roots = list_dicts(tcb.get("trust_roots"))
    bad_assignments = list_dicts(tcb.get("audit_assignments"))
    if bad_assignments:
        forged = dict(bad_assignments[0])
        forged["auditor_root_ids"] = [forged.get("subject_root_id"), forged.get("subject_root_id")]
        traps.append({"trap": "tcb_self_audit", "rejected": bool(validate_tcb(bad_roots, [forged]))})
    forged_claims = {
        "a": {"claim_id": "a", "claim_dependency_refs": ["b"]},
        "b": {"claim_id": "b", "claim_dependency_refs": ["a"]},
    }
    traps.append({"trap": "claim_dependency_cycle", "rejected": bool(claim_dependency_cycles(forged_claims))})
    passed_replays = sum(1 for row in replay_checks if row["passed"])
    rejected_traps = sum(1 for row in traps if row.get("rejected"))
    state = "GREEN" if passed_replays == len(replay_checks) and rejected_traps == len(traps) and tcb.get("state") == "GREEN" and claim_revision.get("state") == "GREEN" else "RED"
    return {
        "record_type": "receipt_faithfulness_adversarial_audit",
        "policy": "project_theseus_receipt_faithfulness_audit_v1",
        "state": state,
        "support_state": "synthetic-test-backed" if state == "GREEN" else "unsupported",
        "random_seed": int(seed),
        "eligible_pack_count": len(ordered),
        "randomized_deep_replay_count": len(replay_checks),
        "passed_deep_replay_count": passed_replays,
        "trap_fixture_count": len(traps),
        "rejected_trap_fixture_count": rejected_traps,
        "cross_component_checks": [
            "gate payload trigger_state equals compact pack trigger_state",
            "canonical source digest equals evidence-pack digest",
            "material claims retain evidence dependencies and dependent surfaces",
            "TCB subject is never its own sole auditor",
            "no-cheat counters stay zero in exported receipts",
        ],
        "replay_checks": replay_checks,
        "trap_fixtures": traps,
        "non_claims": [
            "Adversarial receipt replay tests evidence plumbing, not model capability.",
            "Randomized sampling is deterministic and does not replace complete source-specific verification.",
        ],
        **zero_no_cheat_counters(),
    }


def trust_root(root_id: str, path_value: str, role: str) -> dict[str, Any]:
    path = resolve(path_value)
    return {
        "root_id": root_id,
        "path": rel(path),
        "role": role,
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else "",
        "authority": "verify_and_record_only_no_runtime_capability_widening",
    }


def rotated_assignment(
    epoch: str,
    subject: str,
    auditors: list[str],
    roots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    offset = int(hashlib.sha256(f"{epoch}:{subject}".encode("utf-8")).hexdigest()[:8], 16) % len(auditors)
    ordered = auditors[offset:] + auditors[:offset]
    return {
        "subject_root_id": subject,
        "auditor_root_ids": ordered,
        "auditor_digests": [roots.get(auditor, {}).get("sha256", "") for auditor in ordered],
        "rotation_epoch": epoch,
        "audit_policy": "primary_plus_shadow_independent_replay",
    }


def normalize_claim(claim_id: str, row: dict[str, Any], *, source_path: str) -> dict[str, Any]:
    support = str(row.get("support_state") or row.get("status") or "argument")
    refs = normalized_strings(row.get("evidence_refs") or row.get("evidence_ref"))
    dependency_refs = normalized_strings(
        row.get("claim_dependency_refs") or row.get("depends_on_claim_ids") or row.get("parent_claim_ids")
    )
    return {
        "claim_id": claim_id,
        "claim": str(row.get("claim") or row.get("statement") or row.get("description") or "")[:500],
        "support_state": support,
        "evidence_refs": refs,
        "claim_dependency_refs": dependency_refs,
        "source_paths": [source_path],
        "non_claims": normalized_strings(row.get("non_claims") or row.get("non_claim")),
    }


def claim_dependency_cycles(claims: dict[str, dict[str, Any]]) -> list[list[str]]:
    graph = {
        claim_id: [ref for ref in normalized_strings(row.get("claim_dependency_refs")) if ref in claims]
        for claim_id, row in claims.items()
    }
    cycles: set[tuple[str, ...]] = set()
    visiting: list[str] = []
    active: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            start = visiting.index(node)
            cycle = visiting[start:] + [node]
            cycles.add(tuple(cycle))
            return
        if node in done:
            return
        active.add(node)
        visiting.append(node)
        for child in graph.get(node, []):
            visit(child)
        visiting.pop()
        active.remove(node)
        done.add(node)

    for node in sorted(graph):
        visit(node)
    return [list(cycle) for cycle in sorted(cycles)]


def weaker_support_state(left: str, right: str) -> str:
    rank = {
        "unsupported": 0,
        "argument": 1,
        "prototype-backed": 2,
        "synthetic-test-backed": 3,
        "empirical-test-backed": 4,
        "replayable-reference-backed": 5,
        "SUPPORTED": 3,
    }
    return min((left, right), key=lambda value: rank.get(value, 0))


def evidence_ref_state(ref: str, packs: dict[str, dict[str, Any]]) -> str:
    normalized = normalize_report_ref(ref)
    pack = packs.get(normalized)
    if pack:
        return "green" if pack.get("validation_state") == "GREEN" else "red"
    if normalized.startswith("reports/"):
        payload = read_json(resolve(normalized), {})
        if not payload:
            return "missing"
        return "green" if is_green_gate(payload) else "red"
    return "external_or_nonreport"


def looks_like_claim(row: dict[str, Any]) -> bool:
    record_type = str(row.get("record_type") or "")
    if record_type in CLAIM_RECORD_TYPES:
        return True
    return bool(row.get("claim_id") and (row.get("claim") or row.get("statement")) and (row.get("support_state") or row.get("status")))


def is_green_gate(payload: Any) -> bool:
    return isinstance(payload, dict) and str(payload.get("trigger_state") or "").upper() == "GREEN"


def explicit_support_state(payload: dict[str, Any]) -> str:
    candidates = [payload.get("support_state")]
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    candidates.extend(value for key, value in summary.items() if str(key).endswith("support_state"))
    for value in candidates:
        text = str(value or "")
        if text:
            return text
    return "argument"


def top_level_no_cheat_counters(payload: dict[str, Any]) -> dict[str, int]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    aliases = {
        "public_training_rows_written": ("public_training_rows_written", "public_training_rows"),
        "external_inference_calls": ("external_inference_calls",),
        "fallback_return_count": ("fallback_return_count", "fallback_count"),
    }
    return {
        target: max([int_or(payload.get(key)) for key in keys] + [int_or(summary.get(key)) for key in keys])
        for target, keys in aliases.items()
    }


def compact_named_sections(payload: dict[str, Any], needles: tuple[str, ...]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > 6 or len(matches) >= 32:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                lowered = str(key).lower()
                if any(needle in lowered for needle in needles):
                    matches.append({"path": child_path, **shape_summary(child)})
                visit(child, child_path, depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value[:64]):
                visit(child, f"{path}[{index}]", depth + 1)

    visit(payload, "", 0)
    return {
        "matched_section_count": len(matches),
        "sections": matches,
        "absence_is_explicit": not matches,
    }


def shape_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"value_type": "object", "field_count": len(value), "fields": sorted(map(str, value.keys()))[:20]}
    if isinstance(value, list):
        return {"value_type": "array", "item_count": len(value)}
    return {"value_type": type(value).__name__, "present": value is not None}


def walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_strings(child)


def normalized_strings(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return sorted({str(item) for item in values if str(item).strip()})


def add_report_ref(refs: set[str], value: Any) -> None:
    ref = normalize_report_ref(value)
    if ref:
        refs.add(ref)


def normalize_report_ref(value: Any) -> str:
    text = str(value or "").strip()
    if not text.startswith("reports/") or any(char in text for char in "*?[]"):
        return ""
    return text


def trap(name: str, validator: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    return {"trap": name, "rejected": not bool(validator(candidate))}


def zero_no_cheat_counters() -> dict[str, int]:
    return {key: 0 for key in NO_CHEAT_KEYS}


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in list_values(value) if isinstance(item, dict)]


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def stable_id(*parts: Any) -> str:
    return hashlib.sha256(canonical_bytes(parts)).hexdigest()[:24]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)
