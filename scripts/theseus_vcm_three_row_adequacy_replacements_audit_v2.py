#!/usr/bin/env python3
"""Repair only the sealed three-row audit's nonexistent-counter mechanics bug."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_source_materialization as source  # noqa: E402
import theseus_vcm_three_row_adequacy_replacements_audit as v1  # noqa: E402

POLICY = "project_theseus_vcm_three_row_adequacy_replacements_audit_v2"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_three_row_adequacy_replacements_audit_v2.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    config = p2a.read_json(config_path)
    report = audit(config_path)
    p2a.write_json(p2a.resolve(args.out or config["report"]), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    for binding in p2a.dicts(config.get("source_bindings")):
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != binding.get("sha256"):
            faults.append(f"source_binding_invalid:{binding.get('id')}")

    producer_config_path = p2a.resolve(str(config.get("producer_config") or ""))
    producer_config = p2a.read_json(producer_config_path) if producer_config_path.is_file() else {}
    producer_path = p2a.resolve(str(producer_config.get("report") or ""))
    producer = p2a.read_json(producer_path) if producer_path.is_file() else {}
    v1_report_path = p2a.resolve(str(config.get("v1_audit_report") or ""))
    sealed_v1 = p2a.read_json(v1_report_path) if v1_report_path.is_file() else {}
    live_v1 = v1.audit(producer_config_path) if producer_config_path.is_file() else {}

    if producer.get("trigger_state") != "GREEN" or producer.get("replacement_set_admitted") is not True:
        faults.append("producer_not_admitted")
    if sealed_v1.get("trigger_state") != "RED" or sealed_v1.get("faults") != ["forbidden_execution_counter_nonzero"]:
        faults.append("v1_bug_trigger_invalid")
    comparable = set(sealed_v1) - {"created_utc"}
    if any(sealed_v1.get(key) != live_v1.get(key) for key in comparable):
        faults.append("v1_rederivation_drift")
    if live_v1.get("faults") != ["forbidden_execution_counter_nonzero"]:
        faults.append("v1_fault_not_isolated")

    counters = p2a.mapping(producer.get("counters"))
    expected_schema = set(source.zero_counters()) | {"public_metadata_selection_requests", "local_language_scope_classification_calls"}
    if set(counters) != expected_schema:
        faults.append("producer_counter_schema_invalid")
    forbidden_keys = {
        "D1_cases_consumed", "D2_cases_consumed", "candidate_or_control_calls",
        "external_inference_calls", "local_model_calls", "parent_target_or_evaluator_executions",
        "teacher_calls", "training_rows_written",
    }
    if any(counters.get(key) != 0 for key in forbidden_keys):
        faults.append("real_forbidden_execution_counter_nonzero")
    allowed_nonzero = {
        "public_metadata_selection_requests", "public_metadata_title_requests",
        "public_source_content_requests", "source_archives_materialized",
        "source_bytes_materialized", "local_language_scope_classification_calls",
    }
    if any(value != 0 for key, value in counters.items() if key not in forbidden_keys | allowed_nonzero):
        faults.append("undeclared_nonzero_counter")
    if counters.get("source_archives_materialized") != 12:
        faults.append("archive_counter_invalid")

    v1_fields = {
        "replacement_indices": [12, 13, 35], "unique_replacement_repository_count": 3,
        "source_disjoint_from_current_panel": True, "archive_receipt_count": 12,
        "selected_source_difference_count": 3, "selected_verifier_difference_count": 3,
        "qualified_rows_rerun": False, "frozen_qualified_indices": [16, 25, 56],
        "parent_target_or_evaluator_executions": 0, "local_model_calls": 0,
        "external_reference_calls": 0,
    }
    for key, expected in v1_fields.items():
        if live_v1.get(key) != expected:
            faults.append(f"v1_integrity_field_invalid:{key}")
    authority = p2a.mapping(config.get("authority"))
    if not authority or any(value is not False for value in authority.values()):
        faults.append("authority_boundary_invalid")

    green = not faults
    return {
        "policy": POLICY,
        "audit_kind": "role-separated_rederivation",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if green else "RED",
        "state": "THREE_ROW_ADEQUACY_REPLACEMENTS_AUDIT_COUNTER_REPAIR_GREEN" if green else "THREE_ROW_ADEQUACY_REPLACEMENTS_AUDIT_V2_FAILED",
        "faults": sorted(set(faults)),
        "replacement_set_admitted": green,
        "replacement_indices": live_v1.get("replacement_indices"),
        "unique_replacement_repository_count": live_v1.get("unique_replacement_repository_count"),
        "source_disjoint_from_current_panel": live_v1.get("source_disjoint_from_current_panel"),
        "archive_receipt_count": live_v1.get("archive_receipt_count"),
        "member_receipt_count": live_v1.get("member_receipt_count"),
        "total_member_bytes": live_v1.get("total_member_bytes"),
        "selected_source_difference_count": live_v1.get("selected_source_difference_count"),
        "selected_verifier_difference_count": live_v1.get("selected_verifier_difference_count"),
        "qualified_rows_rerun": False,
        "frozen_qualified_indices": [16, 25, 56],
        "producer_counter_schema": sorted(counters),
        "forbidden_counter_keys_checked": sorted(forbidden_keys),
        "parent_target_or_evaluator_executions": 0,
        "local_model_calls": 0,
        "external_reference_calls": 0,
        "producer_report": artifact(producer_path),
        "v1_audit_report": artifact(v1_report_path),
        "config": artifact(config_path),
        "maximum_inference": config.get("maximum_inference"),
    }


def artifact(path: Path) -> dict[str, Any]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)} if path.is_file() else {"path": p2a.rel(path), "sha256": ""}


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
