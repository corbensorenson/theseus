#!/usr/bin/env python3
"""Assemble and role-separately audit the VCM source panel after adequacy replacements."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_content_language_replacements as content  # noqa: E402
import theseus_vcm_source_panel_audit as v1  # noqa: E402

POLICY = "project_theseus_vcm_source_panel_audit_v4"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_panel_audit_v4.json"
ROLES = {"parent_source", "target_source", "parent_verifier", "target_verifier"}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG)); parser.add_argument("--out", default="")
    args = parser.parse_args(); path = p2a.resolve(args.config); config = p2a.read_json(path); report = audit(path)
    p2a.write_json(p2a.resolve(args.out or config["report"]), report); print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path); faults: list[str] = []; loaded: dict[str, dict[str, Any]] = {}
    if config.get("policy") != POLICY: faults.append("policy_invalid")
    for binding in p2a.dicts(config.get("owner_bindings")):
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != binding.get("sha256"): faults.append(f"owner_binding_invalid:{binding.get('path')}")
    for name, binding in p2a.mapping(config.get("reports")).items():
        record = p2a.mapping(binding); path = p2a.resolve(str(record.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != record.get("sha256"):
            faults.append(f"report_binding_invalid:{name}"); loaded[name] = {}
        else: loaded[name] = p2a.read_json(path)
    prior = loaded.get("source_panel_v3", {}); replacements = loaded.get("adequacy_replacements", {}); replacement_audit = loaded.get("adequacy_replacement_audit", {})
    indices = [int(value) for value in config.get("replacement_indices", [])]
    if prior.get("trigger_state") != "GREEN" or prior.get("source_panel_admitted") is not True or prior.get("assembled_task_count") != 62: faults.append("prior_panel_invalid")
    if replacements.get("trigger_state") != "GREEN" or replacements.get("replacement_set_admitted") is not True: faults.append("replacement_producer_invalid")
    if replacement_audit.get("trigger_state") != "GREEN" or replacement_audit.get("replacement_set_admitted") is not True: faults.append("replacement_audit_invalid")
    old_rows = p2a.dicts(prior.get("assembled_rows")); new = {int(row.get("index") or 0): row for row in p2a.dicts(replacements.get("replacement_rows"))}
    if indices != [12, 13, 35] or set(new) != set(indices) or len(old_rows) != 62: faults.append("replacement_cardinality_invalid")
    rows = []
    for index in range(1, 63):
        old = old_rows[index - 1] if len(old_rows) >= index else {}; row = new.get(index, old)
        if index in new:
            if row.get("panel") != old.get("panel") or row.get("query_language") != old.get("query_language") or row.get("replacement_for_title_sha256") != old.get("natural_language_request_sha256"):
                faults.append(f"task_{index}:replacement_binding_invalid")
        elif row != old: faults.append(f"task_{index}:unapproved_substitution")
        if int(row.get("index") or 0) != index: faults.append(f"task_{index}:index_invalid")
        title = str(row.get("natural_language_request") or "")
        if hashlib.sha256(title.encode()).hexdigest() != row.get("natural_language_request_sha256"): faults.append(f"task_{index}:title_hash_invalid")
        rows.append(row)
    repositories = [str(row.get("repository") or "") for row in rows]
    if len(set(repositories)) != 62 or any(not value for value in repositories): faults.append("repository_uniqueness_invalid")
    if {new[index].get("repository") for index in indices} & {row.get("repository") for row in old_rows}: faults.append("replacement_repository_not_disjoint")
    title_receipts = {int(row.get("index") or 0): row for row in p2a.dicts(replacements.get("language_classification_receipts"))}
    content_receipts = {int(row.get("index") or 0): row for row in p2a.dicts(replacements.get("content_language_receipts"))}
    if set(title_receipts) != set(indices) or set(content_receipts) != set(indices): faults.append("replacement_language_receipts_invalid")
    ranges = [(row["name"], int(row["start"], 16), int(row["end"], 16)) for row in config["forbidden_unicode_scripts"]]; binary = set(config["binary_extensions"])
    archives = members = total_bytes = source_changes = verifier_changes = 0; paths: list[str] = []; violations: list[dict[str, Any]] = []
    for row in rows:
        index = int(row.get("index") or 0); receipts = p2a.mapping(row.get("archives")); hashes: dict[str, dict[str, str]] = {}
        if set(receipts) != ROLES or row.get("faults") != []: faults.append(f"task_{index}:archive_set_invalid"); continue
        for role in sorted(ROLES):
            receipt = p2a.mapping(receipts[role]); path = p2a.resolve(str(receipt.get("path") or "")); archive_faults, member_hashes, member_bytes = v1.audit_archive(path, receipt)
            faults.extend(f"task_{index}:{role}:{fault}" for fault in archive_faults); hashes[role] = member_hashes; paths.append(p2a.rel(path)); archives += 1; members += len(member_hashes); total_bytes += member_bytes
            license_path = str(row.get("parent_license_path") if role.startswith("parent_") else row.get("target_license_path") or "")
            if license_path not in member_hashes: faults.append(f"task_{index}:{role}:license_missing")
        source_changed = v1.selected_paths_changed(row, hashes, "source"); verifier_changed = v1.selected_paths_changed(row, hashes, "verifier")
        source_changes += int(source_changed); verifier_changes += int(verifier_changed)
        if not source_changed: faults.append(f"task_{index}:source_unchanged")
        if not verifier_changed: faults.append(f"task_{index}:verifier_unchanged")
        found = content.selected_content_violations(row, ranges, binary)
        if found: violations.append({"index": index, "repository": row.get("repository"), "violations": found}); faults.append(f"task_{index}:selected_content_language_invalid")
    quotas = Counter((str(row.get("panel") or ""), str(row.get("query_language") or "")) for row in rows)
    expected = {(row["panel"], row["query_language"]): int(row["count"]) for row in config["expected_quotas"]}
    if dict(quotas) != expected: faults.append("panel_language_quotas_invalid")
    if archives != 248 or len(set(paths)) != 248: faults.append("archive_cardinality_invalid")
    if source_changes != 62 or verifier_changes != 62: faults.append("difference_count_invalid")
    authority = p2a.mapping(config.get("authority"))
    if not authority or any(value is not False for value in authority.values()): faults.append("authority_boundary_invalid")
    admitted = not faults
    return {"policy": POLICY, "audit_kind": "role-separated_rederivation", "created_utc": p2a.now(), "trigger_state": "GREEN" if admitted else "RED", "state": "SIXTY_TWO_TASK_PANEL_WITH_ADEQUACY_REPLACEMENTS_ADMITTED" if admitted else "SOURCE_PANEL_V4_AUDIT_FAILED", "faults": sorted(set(faults)), "source_panel_admitted": admitted, "assembled_task_count": len(rows), "unique_repository_count": len(set(repositories)), "english_title_and_selected_content_task_count": 62 if admitted else 0, "archive_receipt_count": archives, "member_receipt_count": members, "total_member_bytes": total_bytes, "selected_source_difference_count": source_changes, "selected_verifier_difference_count": verifier_changes, "replacement_indices": indices, "preserved_row_count": 59, "content_violations": violations, "panel_language_quotas": [{"panel": panel, "query_language": language, "count": count} for (panel, language), count in sorted(quotas.items())], "assembled_rows": rows, "candidate_packet_materialization_opened": False, "parent_target_or_evaluator_executions": 0, "local_model_calls": 0, "external_reference_calls": 0, "D1_cases_consumed": 0, "D2_cases_consumed": 0, "config": {"path": p2a.rel(config_path), "sha256": p2a.sha256_file(config_path)}, "reports": config.get("reports"), "maximum_inference": config.get("maximum_inference")}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in ("trigger_state", "state", "source_panel_admitted", "assembled_task_count", "unique_repository_count", "archive_receipt_count", "member_receipt_count", "selected_source_difference_count", "selected_verifier_difference_count", "replacement_indices", "preserved_row_count", "content_violations", "parent_target_or_evaluator_executions", "local_model_calls", "external_reference_calls", "faults")}


if __name__ == "__main__": raise SystemExit(main())
