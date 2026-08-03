#!/usr/bin/env python3
"""Independently admit the repaired 62-task English VCM source panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_vcm_source_panel_audit as v1audit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_panel_audit_v2.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_source_panel_audit_v2.json"
POLICY = "project_theseus_vcm_source_panel_audit_v2"
ROLES = {"parent_source", "target_source", "parent_verifier", "target_verifier"}
LANGUAGE_REPLACEMENT_INDICES = {1, 12, 19, 48, 51, 56}
ZERO_ACTIVITY_KEYS = v1audit.ZERO_ACTIVITY_KEYS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = audit(p2a.resolve(args.config))
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    authority = p2a.mapping(config.get("authority"))
    if not authority or any(value is not False for value in authority.values()):
        faults.append("audit_authority_boundary_invalid")
    loaded: dict[str, dict[str, Any]] = {}
    for name, raw_binding in p2a.mapping(config.get("reports")).items():
        binding = p2a.mapping(raw_binding)
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"report_binding_invalid:{name}")
            loaded[name] = {}
        else:
            loaded[name] = p2a.read_json(path)
    selection = loaded.get("metadata_selection", {})
    materialization = loaded.get("failed_materialization", {})
    replacement_28 = loaded.get("replacement_28", {})
    prior_audit = loaded.get("prior_panel_audit", {})
    language_replacements = loaded.get("language_replacements_v2", {})
    validate_terminal_reports(selection, materialization, replacement_28, prior_audit, language_replacements, faults)

    selected = p2a.dicts(selection.get("selected_source_identities"))
    original_rows = p2a.dicts(materialization.get("rows"))
    row_28 = p2a.mapping(replacement_28.get("replacement_materialization"))
    language_rows = {integer(row.get("index")): row for row in p2a.dicts(language_replacements.get("replacement_rows"))}
    if set(language_rows) != LANGUAGE_REPLACEMENT_INDICES:
        faults.append("language_replacement_index_set_invalid")
    if len(selected) != 62 or len(original_rows) != 62 or [integer(row.get("index")) for row in original_rows] != list(range(1, 63)):
        faults.append("original_panel_cardinality_invalid")

    rows: list[dict[str, Any]] = []
    for index in range(1, 63):
        original = original_rows[index - 1] if len(original_rows) >= index else {}
        metadata = selected[index - 1] if len(selected) >= index else {}
        for field in v1audit.IDENTITY_FIELDS:
            if original.get(field) != metadata.get(field):
                faults.append(f"task_{index:02d}:original_metadata_identity_mismatch:{field}")
        if index == 28:
            row = row_28
            if original.get("faults") != ["selected_verifier_bytes_unchanged"]:
                faults.append("task_28:original_failure_not_preserved")
        elif index in LANGUAGE_REPLACEMENT_INDICES:
            row = language_rows.get(index, {})
            if row.get("replacement_for_title_sha256") != original.get("natural_language_request_sha256"):
                faults.append(f"task_{index:02d}:superseded_title_binding_invalid")
        else:
            row = original
            if row.get("faults") != []:
                faults.append(f"task_{index:02d}:unexpected_original_fault")
        if integer(row.get("index")) != index:
            faults.append(f"task_{index:02d}:assembled_index_invalid")
        if row.get("panel") != metadata.get("panel") or row.get("query_language") != metadata.get("query_language"):
            faults.append(f"task_{index:02d}:slot_panel_or_language_changed")
        title = str(row.get("natural_language_request") or "")
        if hashlib.sha256(title.encode()).hexdigest() != row.get("natural_language_request_sha256"):
            faults.append(f"task_{index:02d}:natural_language_request_hash_invalid")
        rows.append(row)

    language_receipts = {integer(row.get("index")): row for row in p2a.dicts(language_replacements.get("language_classification_receipts"))}
    if set(language_receipts) != LANGUAGE_REPLACEMENT_INDICES:
        faults.append("language_receipt_index_set_invalid")
    for index in sorted(LANGUAGE_REPLACEMENT_INDICES):
        row = rows[index - 1]
        receipt = language_receipts.get(index, {})
        if receipt.get("repository") != row.get("repository") or receipt.get("title_sha256") != row.get("natural_language_request_sha256") or receipt.get("dominant_language") != "en" or receipt.get("accepted_english") is not True or receipt.get("forbidden_unicode_scripts") != []:
            faults.append(f"task_{index:02d}:english_classification_receipt_invalid")

    old_non_english = [integer(row.get("index")) for row in p2a.dicts(prior_audit.get("replacement_slots_required"))]
    if old_non_english != sorted(LANGUAGE_REPLACEMENT_INDICES) or integer(prior_audit.get("english_eligible_task_count")) != 56:
        faults.append("prior_english_review_binding_invalid")
    title_items = [{"index": integer(row.get("index")), "title_sha256": str(row.get("natural_language_request_sha256") or "")} for row in rows]
    title_set_sha256 = v1audit.stable_hash(title_items)
    expected = p2a.mapping(config.get("expected_panel"))
    if title_set_sha256 != expected.get("title_set_sha256"):
        faults.append("assembled_title_set_binding_invalid")

    repositories = [str(row.get("repository") or "") for row in rows]
    if len(set(repositories)) != 62 or any(not repository for repository in repositories):
        faults.append("assembled_repository_uniqueness_invalid")
    prior = set(v1audit.tracked_prior_repositories(config_path))
    original_repositories = {str(row.get("repository") or "") for row in selected}
    replacement_repositories = {rows[index - 1]["repository"] for index in LANGUAGE_REPLACEMENT_INDICES | {28}}
    if replacement_repositories & original_repositories or replacement_repositories & prior:
        faults.append("replacement_repository_disjointness_invalid")

    archive_paths: list[str] = []
    member_receipts = 0
    total_member_bytes = 0
    source_difference_count = 0
    verifier_difference_count = 0
    for row in rows:
        index = integer(row.get("index"))
        archives = p2a.mapping(row.get("archives"))
        if set(archives) != ROLES or row.get("faults") != []:
            faults.append(f"task_{index:02d}:archive_set_invalid")
            continue
        hashes: dict[str, dict[str, str]] = {}
        for role in sorted(ROLES):
            receipt = p2a.mapping(archives.get(role))
            path = p2a.resolve(str(receipt.get("path") or ""))
            archive_faults, member_hashes, member_bytes = v1audit.audit_archive(path, receipt)
            faults.extend(f"task_{index:02d}:{role}:{fault}" for fault in archive_faults)
            expected_root = f"vcm-claim-{index:02d}"
            if receipt.get("root") != expected_root or path.name != f"{expected_root}-{role}.tar.gz":
                faults.append(f"task_{index:02d}:{role}:archive_identity_invalid")
            license_path = str((row.get("parent_license_path") if role.startswith("parent_") else row.get("target_license_path")) or "")
            if license_path not in member_hashes:
                faults.append(f"task_{index:02d}:{role}:license_member_missing")
            hashes[role] = member_hashes
            archive_paths.append(p2a.rel(path))
            member_receipts += len(member_hashes)
            total_member_bytes += member_bytes
        if v1audit.selected_paths_changed(row, hashes, "source"):
            source_difference_count += 1
        else:
            faults.append(f"task_{index:02d}:selected_source_bytes_unchanged")
        if v1audit.selected_paths_changed(row, hashes, "verifier"):
            verifier_difference_count += 1
        else:
            faults.append(f"task_{index:02d}:selected_verifier_bytes_unchanged")

    if len(archive_paths) != integer(expected.get("archive_count")) or len(set(archive_paths)) != len(archive_paths):
        faults.append("archive_path_cardinality_invalid")
    if member_receipts != integer(expected.get("member_receipt_count")):
        faults.append("member_receipt_count_invalid")
    if total_member_bytes != integer(expected.get("total_member_bytes")):
        faults.append("total_member_bytes_invalid")
    if source_difference_count != 62 or verifier_difference_count != 62:
        faults.append("source_or_verifier_difference_count_invalid")
    superseded_paths = [p2a.mapping(receipt).get("path") for row in original_rows if integer(row.get("index")) in LANGUAGE_REPLACEMENT_INDICES for receipt in p2a.mapping(row.get("archives")).values()]
    if len(superseded_paths) != integer(expected.get("superseded_archive_count")) or any(path in set(archive_paths) for path in superseded_paths):
        faults.append("superseded_archive_disposition_invalid")

    actual_quotas = Counter((str(row.get("panel") or ""), str(row.get("query_language") or "")) for row in rows)
    expected_quotas = {(str(row.get("panel") or ""), str(row.get("query_language") or "")): integer(row.get("count")) for row in p2a.dicts(expected.get("quotas"))}
    if dict(actual_quotas) != expected_quotas:
        faults.append("panel_language_quotas_invalid")
    for name in ("failed_materialization", "replacement_28", "language_replacements_v2"):
        counters = p2a.mapping(loaded.get(name, {}).get("counters"))
        for key in ZERO_ACTIVITY_KEYS:
            if integer(counters.get(key)) != 0:
                faults.append(f"{name}_forbidden_activity:{key}")

    admitted = not faults
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if admitted else "RED",
        "state": "SIXTY_TWO_ENGLISH_SOURCE_TASKS_ADMITTED_BEFORE_EVALUATOR_EXECUTION" if admitted else "SOURCE_PANEL_V2_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "source_panel_admitted": admitted,
        "assembled_task_count": len(rows),
        "unique_repository_count": len(set(repositories)),
        "english_eligible_task_count": 62 if admitted else 0,
        "archive_receipt_count": len(archive_paths),
        "member_receipt_count": member_receipts,
        "total_member_bytes": total_member_bytes,
        "selected_source_difference_count": source_difference_count,
        "selected_verifier_difference_count": verifier_difference_count,
        "superseded_non_english_archive_count_preserved_but_ignored": len(superseded_paths),
        "task_28_replacement_bound": True,
        "six_language_replacements_bound": set(language_rows) == LANGUAGE_REPLACEMENT_INDICES,
        "title_set_sha256": title_set_sha256,
        "panel_language_quotas": [{"panel": panel, "query_language": language, "count": count} for (panel, language), count in sorted(actual_quotas.items())],
        "assembled_rows": rows,
        "candidate_packet_materialization_opened": False,
        "parent_target_or_evaluator_executions": 0,
        "local_model_calls": 0,
        "external_reference_calls": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "config": artifact(config_path),
        "reports": {name: artifact(p2a.resolve(str(p2a.mapping(binding).get("path") or ""))) for name, binding in p2a.mapping(config.get("reports")).items()},
        "maximum_inference": config.get("maximum_inference"),
    }


def validate_terminal_reports(selection: dict[str, Any], materialization: dict[str, Any], replacement_28: dict[str, Any], prior_audit: dict[str, Any], language_replacements: dict[str, Any], faults: list[str]) -> None:
    if selection.get("trigger_state") != "GREEN" or len(p2a.dicts(selection.get("selected_source_identities"))) != 62:
        faults.append("metadata_selection_terminal_invalid")
    if materialization.get("trigger_state") != "RED" or materialization.get("faults") != ["task_28:selected_verifier_bytes_unchanged"]:
        faults.append("materialization_terminal_invalid")
    if replacement_28.get("trigger_state") != "GREEN" or replacement_28.get("replacement_admitted") is not True:
        faults.append("replacement_28_terminal_invalid")
    if prior_audit.get("trigger_state") != "RED" or prior_audit.get("state") != "SOURCE_PANEL_LANGUAGE_REPLACEMENTS_REQUIRED" or prior_audit.get("archive_integrity_green") is not True:
        faults.append("prior_panel_audit_terminal_invalid")
    if language_replacements.get("trigger_state") != "GREEN" or language_replacements.get("state") != "SIX_ENGLISH_SOURCE_REPLACEMENTS_BOUND" or language_replacements.get("replacement_set_admitted") is not True or language_replacements.get("checkpoint_artifact_hash_verified_final") is not True:
        faults.append("language_replacement_terminal_invalid")
    checkpoint = p2a.mapping(language_replacements.get("checkpoint"))
    path = p2a.resolve(str(checkpoint.get("path") or ""))
    if not path.is_file() or p2a.sha256_file(path) != str(checkpoint.get("sha256") or ""):
        faults.append("language_replacement_checkpoint_invalid")
    if p2a.mapping(language_replacements.get("attempted_request_role_accounting")).get("sum_equals_checkpoint_logical_requests") is not True:
        faults.append("language_replacement_role_accounting_invalid")


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "source_panel_admitted", "assembled_task_count",
        "unique_repository_count", "english_eligible_task_count", "archive_receipt_count",
        "member_receipt_count", "selected_source_difference_count",
        "selected_verifier_difference_count", "superseded_non_english_archive_count_preserved_but_ignored",
        "candidate_packet_materialization_opened", "parent_target_or_evaluator_executions",
        "local_model_calls", "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
