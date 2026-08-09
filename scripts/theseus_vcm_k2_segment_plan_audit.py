#!/usr/bin/env python3
"""Role-separated audit for the generic 62-row K2.05 segment plan."""
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

POLICY = "project_theseus_vcm_k2_segment_plan_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_k2_segment_plan.json"
SAFE_ROW_KEYS = {
    "natural_language_request", "natural_language_request_sha256", "parent_archive",
    "parent_archive_sha256", "parent_archive_root", "parent_revision", "license_spdx",
    "sanitization_report", "sanitization_report_sha256",
}
FORBIDDEN_KEYS = {
    "allowed_effect_paths", "answer", "answer_family", "category", "expected",
    "hidden_tests", "pull_request", "repository", "repository_identity",
    "required_constructs", "return_shape", "selected_source_paths",
    "selected_verifier_paths", "solution", "solution_body", "solution_expr",
    "source_task_id", "target_diff", "target_patch", "target_snapshot", "tests",
    "type_family",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    result = audit(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg.get("audit_report") or "")), result)
    print(json.dumps({key: result.get(key) for key in ("trigger_state", "state", "faults", "audited_task_count", "segment_counts", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if result["trigger_state"] == "GREEN" else 2


def audit(path: Path = DEFAULT_CONFIG, *, report: dict[str, Any] | None = None, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("audit_policy") != POLICY:
        faults.append("audit_policy_invalid")
    audit_owner = p2a.resolve(str(cfg.get("audit_owner") or ""))
    producer = p2a.resolve(str(cfg.get("owner") or ""))
    if audit_owner != Path(__file__).resolve() or p2a.sha256_file(audit_owner) != cfg.get("audit_owner_sha256"):
        faults.append("audit_owner_binding_invalid")
    if not producer.is_file() or p2a.sha256_file(producer) != cfg.get("owner_sha256"):
        faults.append("producer_owner_binding_invalid")
    payloads: dict[str, dict[str, Any]] = {}
    for binding in p2a.dicts(cfg.get("sources")):
        source_id = str(binding.get("id") or "")
        source = p2a.resolve(str(binding.get("path") or ""))
        if not source_id or source_id in payloads or not source.is_file() or p2a.sha256_file(source) != binding.get("sha256"):
            faults.append(f"source_binding_invalid:{source_id}")
            payloads[source_id] = {}
        else:
            payloads[source_id] = p2a.read_json(source)
    producer_report = report if report is not None else p2a.read_json(p2a.resolve(str(cfg.get("report") or "")))
    safe_manifest = manifest if manifest is not None else p2a.read_json(p2a.resolve(str(cfg.get("manifest_out") or "")))
    if producer_report.get("trigger_state") != "GREEN" or producer_report.get("panel_admitted") is not False:
        faults.append("producer_state_invalid")
    if safe_manifest.get("policy") != "project_theseus_vcm_parent_only_batch_manifest_v1":
        faults.append("manifest_policy_invalid")
    if safe_manifest.get("broad_parent_effect_root") != "repository":
        faults.append("broad_parent_effect_root_invalid")
    if set(p2a.strings(safe_manifest.get("candidate_visible_fields"))) != {
        "natural_language_request", "callable_signature_when_present",
        "broad_parent_effect_root", "arm_specific_model_visible_context",
    }:
        faults.append("candidate_visible_field_contract_invalid")

    panel = indexed(payloads.get("source_panel", {}).get("assembled_rows"), "index")
    closures = indexed(payloads.get("repository_closures", {}).get("tasks"), "campaign_index")
    runners = indexed(payloads.get("runner_inventory", {}).get("rows"), "index")
    classes = indexed(payloads.get("dependency_classes", {}).get("rows"), "index")
    plans = indexed(payloads.get("dependency_plan", {}).get("rows"), "index")
    rows = p2a.dicts(safe_manifest.get("rows"))
    schedule = p2a.dicts(producer_report.get("schedule"))
    expected_count = int(cfg.get("expected_task_count") or 0)
    if len(rows) != expected_count or len(schedule) != expected_count:
        faults.append("manifest_or_schedule_count_invalid")
    segment_counts = {name: 0 for name in ("static_no_project_lock", "immutable_resolution_required", "locked_closure")}
    for index, actual in enumerate(rows, start=1):
        if set(actual) != SAFE_ROW_KEYS or recursive_forbidden(actual):
            faults.append(f"unsafe_manifest_fields:{index}")
        panel_row = panel.get(index, {})
        closure_row = closures.get(index, {})
        repositories = {str(row.get("repository") or "") for row in (panel_row, closure_row, runners.get(index, {}), classes.get(index, {}), plans.get(index, {}))}
        if len(repositories) != 1 or "" in repositories:
            faults.append(f"source_identity_alignment_invalid:{index}")
        parent = next((row for row in p2a.dicts(closure_row.get("artifacts")) if row.get("label") == "parent"), {})
        expected = {
            "natural_language_request": panel_row.get("natural_language_request"),
            "natural_language_request_sha256": panel_row.get("natural_language_request_sha256"),
            "parent_archive": parent.get("normalized"),
            "parent_archive_sha256": parent.get("normalized_sha256"),
            "parent_archive_root": parent.get("source_archive_root"),
            "parent_revision": parent.get("revision"),
            "license_spdx": panel_row.get("license_spdx"),
            "sanitization_report": parent.get("sanitization_report"),
            "sanitization_report_sha256": parent.get("sanitization_report_sha256"),
        }
        if actual != expected:
            faults.append(f"parent_manifest_rederivation_failed:{index}")
        if p2a.sha256_text(str(actual.get("natural_language_request") or "")) != actual.get("natural_language_request_sha256"):
            faults.append(f"natural_request_digest_invalid:{index}")
        if str(actual.get("parent_revision") or "") != str(panel_row.get("base_revision") or ""):
            faults.append(f"parent_revision_invalid:{index}")
        if not str(actual.get("parent_archive") or "").endswith("_parent.tar.gz") or "target" in str(actual.get("parent_archive") or "").lower():
            faults.append(f"non_parent_archive_in_manifest:{index}")
        if not str(actual.get("sanitization_report") or "").endswith("_parent.json") or "target" in str(actual.get("sanitization_report") or "").lower():
            faults.append(f"non_parent_sanitization_in_manifest:{index}")
        scheduled = schedule[index - 1] if index <= len(schedule) else {}
        manager = str(plans.get(index, {}).get("manager") or "")
        segment = "static_no_project_lock" if manager == "trusted_runtime_or_harness" else "immutable_resolution_required" if manager == "resolver_required" else "locked_closure"
        if scheduled.get("panel_index") != index or scheduled.get("segment") != segment or scheduled.get("panel_admission") != "withheld_until_all_62_rows_complete":
            faults.append(f"segment_schedule_invalid:{index}")
        segment_counts[segment] += 1
    if segment_counts != p2a.mapping(cfg.get("expected_segment_counts")):
        faults.append("segment_count_rederivation_failed")
    if producer_report.get("parent_only_manifest_projection_sha256") != digest_json(safe_manifest):
        faults.append("manifest_projection_digest_invalid")
    for key in ("network_or_dependency_execution_performed", "repository_runner_executions", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls"):
        if producer_report.get(key) not in (False, 0):
            faults.append(f"producer_counter_invalid:{key}")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "K2_05_TARGET_FREE_SEGMENT_PLAN_ROLE_SEPARATELY_REDERIVED" if not faults else "K2_05_SEGMENT_PLAN_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)},
        "producer_report": {"path": p2a.rel(p2a.resolve(str(cfg.get("report") or ""))), "sha256": p2a.sha256_file(p2a.resolve(str(cfg.get("report") or "")))},
        "parent_only_manifest": {"path": p2a.rel(p2a.resolve(str(cfg.get("manifest_out") or ""))), "sha256": p2a.sha256_file(p2a.resolve(str(cfg.get("manifest_out") or ""))), "projection_sha256": digest_json(safe_manifest)},
        "audited_task_count": len(rows),
        "segment_counts": segment_counts,
        "candidate_visible_field_count": len(p2a.strings(safe_manifest.get("candidate_visible_fields"))),
        "target_derived_selector_input_count": 0 if not faults else None,
        "panel_admitted": False,
        "audit_kind": "role-separated rederivation",
        "network_or_dependency_execution_performed": False,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": cfg.get("audit_maximum_inference"),
    }


def indexed(raw: Any, key: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in p2a.dicts(raw):
        index = int(row.get(key) or 0)
        if index > 0 and index not in result:
            result[index] = row
    return result


def recursive_forbidden(value: Any) -> list[str]:
    hits: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in FORBIDDEN_KEYS:
                hits.add(str(key))
            hits.update(recursive_forbidden(child))
    elif isinstance(value, list):
        for child in value:
            hits.update(recursive_forbidden(child))
    return sorted(hits)


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
