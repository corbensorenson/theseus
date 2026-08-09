#!/usr/bin/env python3
"""Role-separately audit the three-row VCM adequacy replacement transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_content_language_replacements as content  # noqa: E402
import theseus_vcm_source_panel_audit as panel_audit  # noqa: E402

POLICY = "project_theseus_vcm_three_row_adequacy_replacements_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_three_row_adequacy_replacements.json"
ROLES = {"parent_source", "target_source", "parent_verifier", "target_verifier"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    config = p2a.read_json(path)
    report = audit(path)
    p2a.write_json(p2a.resolve(args.out or config["audit_report"]), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != "project_theseus_vcm_three_row_adequacy_replacements_v1":
        faults.append("producer_policy_invalid")
    producer_path = p2a.resolve(str(config.get("report") or ""))
    panel_path = p2a.resolve(str(config.get("source_panel") or ""))
    producer = p2a.read_json(producer_path) if producer_path.is_file() else {}
    panel = p2a.read_json(panel_path) if panel_path.is_file() else {}
    if producer.get("trigger_state") != "GREEN" or producer.get("replacement_set_admitted") is not True:
        faults.append("producer_not_admitted")
    if producer.get("policy") != "project_theseus_vcm_three_row_adequacy_replacements_v1":
        faults.append("producer_report_policy_invalid")
    if producer.get("config", {}).get("sha256") != p2a.sha256_file(config_path):
        faults.append("producer_config_binding_invalid")
    if panel.get("trigger_state") != "GREEN" or panel.get("source_panel_admitted") is not True:
        faults.append("source_panel_invalid")

    indices = [int(row["index"]) for row in config.get("replacement_slots", [])]
    rows = {int(row.get("index") or 0): row for row in p2a.dicts(producer.get("replacement_rows"))}
    old_rows = {int(row.get("index") or 0): row for row in p2a.dicts(panel.get("assembled_rows"))}
    if indices != [12, 13, 35] or set(rows) != set(indices):
        faults.append("replacement_row_set_invalid")
    current_repositories = {str(row.get("repository") or "") for row in old_rows.values()}
    repositories = [str(rows.get(index, {}).get("repository") or "") for index in indices]
    if len(set(repositories)) != 3 or any(not repository for repository in repositories):
        faults.append("replacement_repository_uniqueness_invalid")
    if set(repositories) & current_repositories:
        faults.append("replacement_repository_not_source_disjoint")

    title_receipts = {int(row.get("index") or 0): row for row in p2a.dicts(producer.get("language_classification_receipts"))}
    content_receipts = {int(row.get("index") or 0): row for row in p2a.dicts(producer.get("content_language_receipts"))}
    if set(title_receipts) != set(indices) or set(content_receipts) != set(indices):
        faults.append("language_receipt_set_invalid")
    ranges = [(row["name"], int(row["start"], 16), int(row["end"], 16)) for row in config["forbidden_unicode_scripts"]]
    binary = set(config["binary_extensions"])
    archive_count = member_count = total_bytes = source_changes = verifier_changes = 0
    audited_rows = []
    for slot in config.get("replacement_slots", []):
        index = int(slot["index"]); row = rows.get(index, {}); old = old_rows.get(index, {})
        if row.get("panel") != slot.get("panel") or row.get("query_language") != slot.get("query_language"):
            faults.append(f"task_{index}:slot_shape_invalid")
        if row.get("replacement_for_title_sha256") != slot.get("rejected_title_sha256") or slot.get("rejected_title_sha256") != old.get("natural_language_request_sha256"):
            faults.append(f"task_{index}:replaced_title_binding_invalid")
        title = str(row.get("natural_language_request") or "")
        if hashlib.sha256(title.encode()).hexdigest() != row.get("natural_language_request_sha256"):
            faults.append(f"task_{index}:title_hash_invalid")
        title_receipt = title_receipts.get(index, {})
        if (
            title_receipt.get("repository") != row.get("repository")
            or title_receipt.get("title_sha256") != row.get("natural_language_request_sha256")
            or title_receipt.get("dominant_language") != "en"
            or title_receipt.get("accepted_english") is not True
            or title_receipt.get("forbidden_unicode_scripts") != []
        ):
            faults.append(f"task_{index}:title_language_receipt_invalid")
        content_receipt = content_receipts.get(index, {})
        violations = content.selected_content_violations(row, ranges, binary) if row else [{"missing": True}]
        if (
            content_receipt.get("repository") != row.get("repository")
            or content_receipt.get("selected_content_english_scope_passed") is not True
            or content_receipt.get("violations") != []
            or violations
        ):
            faults.append(f"task_{index}:selected_content_language_invalid")
        archives = p2a.mapping(row.get("archives")); hashes: dict[str, dict[str, str]] = {}
        if set(archives) != ROLES or row.get("faults") != []:
            faults.append(f"task_{index}:archive_set_invalid")
        for role in sorted(ROLES):
            receipt = p2a.mapping(archives.get(role)); path = p2a.resolve(str(receipt.get("path") or ""))
            archive_faults, member_hashes, member_bytes = panel_audit.audit_archive(path, receipt)
            faults.extend(f"task_{index}:{role}:{fault}" for fault in archive_faults)
            hashes[role] = member_hashes; archive_count += 1; member_count += len(member_hashes); total_bytes += member_bytes
            license_path = str(row.get("parent_license_path") if role.startswith("parent_") else row.get("target_license_path") or "")
            if license_path not in member_hashes:
                faults.append(f"task_{index}:{role}:license_missing")
        source_changed = panel_audit.selected_paths_changed(row, hashes, "source")
        verifier_changed = panel_audit.selected_paths_changed(row, hashes, "verifier")
        source_changes += int(source_changed); verifier_changes += int(verifier_changed)
        if not source_changed: faults.append(f"task_{index}:source_unchanged")
        if not verifier_changed: faults.append(f"task_{index}:verifier_unchanged")
        audited_rows.append({"index": index, "repository": row.get("repository"), "panel": row.get("panel"), "query_language": row.get("query_language"), "source_changed": source_changed, "verifier_changed": verifier_changed, "archive_count": len(archives)})

    counters = p2a.mapping(producer.get("counters"))
    forbidden_counter_keys = ("parent_target_or_evaluator_executions", "local_model_calls", "external_reference_calls", "teacher_calls", "training_rows_admitted", "D1_cases_consumed", "D2_cases_consumed")
    if any(counters.get(key) != 0 for key in forbidden_counter_keys):
        faults.append("forbidden_execution_counter_nonzero")
    if producer.get("qualified_rows_rerun") is not False or producer.get("frozen_qualified_indices") != [16, 25, 56]:
        faults.append("qualified_row_freeze_invalid")
    authority = p2a.mapping(config.get("audit_authority"))
    if not authority or any(value is not False for value in authority.values()):
        faults.append("audit_authority_boundary_invalid")
    admitted = not faults
    return {
        "policy": POLICY,
        "audit_kind": "role-separated_rederivation",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if admitted else "RED",
        "state": "THREE_ROW_ADEQUACY_REPLACEMENTS_ROLE_SEPARATELY_REDERIVED" if admitted else "THREE_ROW_ADEQUACY_REPLACEMENT_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "replacement_set_admitted": admitted,
        "replacement_indices": indices,
        "audited_rows": audited_rows,
        "unique_replacement_repository_count": len(set(repositories)),
        "source_disjoint_from_current_panel": not bool(set(repositories) & current_repositories),
        "archive_receipt_count": archive_count,
        "member_receipt_count": member_count,
        "total_member_bytes": total_bytes,
        "selected_source_difference_count": source_changes,
        "selected_verifier_difference_count": verifier_changes,
        "qualified_rows_rerun": False,
        "frozen_qualified_indices": [16, 25, 56],
        "parent_target_or_evaluator_executions": 0,
        "local_model_calls": 0,
        "external_reference_calls": 0,
        "producer_report": {"path": p2a.rel(producer_path), "sha256": p2a.sha256_file(producer_path)} if producer_path.is_file() else {},
        "config": {"path": p2a.rel(config_path), "sha256": p2a.sha256_file(config_path)},
        "maximum_inference": config.get("audit_maximum_inference"),
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "replacement_set_admitted", "replacement_indices",
        "unique_replacement_repository_count", "source_disjoint_from_current_panel",
        "archive_receipt_count", "selected_source_difference_count", "selected_verifier_difference_count",
        "qualified_rows_rerun", "parent_target_or_evaluator_executions", "local_model_calls",
        "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
