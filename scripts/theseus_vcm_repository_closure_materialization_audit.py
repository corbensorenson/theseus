#!/usr/bin/env python3
"""Role-separately audit generic VCM repository closure materialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_repository_closure_materialization as owner  # noqa: E402

POLICY = "project_theseus_vcm_repository_closure_materialization_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_repository_closure_materialization_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG)); parser.add_argument("--out", default="")
    args = parser.parse_args(); path = p2a.resolve(args.config); config = p2a.read_json(path); report = audit(path)
    p2a.write_json(p2a.resolve(args.out or config["report"]), report); print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path); faults: list[str] = []
    if config.get("policy") != POLICY: faults.append("policy_invalid")
    for binding in p2a.dicts(config.get("bindings")):
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != binding.get("sha256"): faults.append(f"binding_invalid:{binding.get('id')}")
    panel_path = p2a.resolve(str(config.get("source_panel") or "")); producer_path = p2a.resolve(str(config.get("producer_report") or "")); predecessor_path = p2a.resolve(str(config.get("predecessor_report") or ""))
    panel = p2a.read_json(panel_path) if panel_path.is_file() else {}; producer = p2a.read_json(producer_path) if producer_path.is_file() else {}; predecessor = p2a.read_json(predecessor_path) if predecessor_path.is_file() else {}
    if panel.get("trigger_state") != "GREEN" or panel.get("source_panel_admitted") is not True: faults.append("source_panel_invalid")
    if producer.get("trigger_state") != "GREEN" or producer.get("archive_artifacts") != 124 or producer.get("network_fetches") != 6: faults.append("producer_invalid")
    if predecessor.get("trigger_state") != "GREEN" or predecessor.get("archive_artifacts") != 124: faults.append("predecessor_invalid")
    registry = owner.transform_panel(panel); registry_faults = owner.d1.audit_registry(registry) if registry else ["registry_missing"]
    faults.extend(f"registry:{fault}" for fault in registry_faults)
    tasks = {int(row.get("campaign_index") or 0): row for row in p2a.dicts(producer.get("tasks"))}; expected = {int(row.get("campaign_index") or 0): row for row in registry.get("tasks", [])}
    predecessor_tasks = {int(row.get("campaign_index") or 0): row for row in p2a.dicts(predecessor.get("tasks"))}
    if set(tasks) != set(range(1, 63)) or set(expected) != set(tasks): faults.append("task_index_set_invalid")
    replayed = replacements = artifact_count = member_requirement_count = 0; audited_rows = []
    replacement_indices = set(int(value) for value in config.get("replacement_indices", []))
    for index in range(1, 63):
        task = expected.get(index, {}); row = tasks.get(index, {}); artifacts = p2a.dicts(row.get("artifacts")); row_faults: list[str] = []
        if row.get("repository") != task.get("repository") or row.get("selection_digest") != task.get("selection_digest"): row_faults.append("task_identity_invalid")
        if {str(item.get("label")) for item in artifacts} != {"parent", "target"}: row_faults.append("artifact_labels_invalid")
        for receipt in artifacts:
            path = p2a.resolve(str(receipt.get("normalized") or "")); sanitation_path = p2a.resolve(str(receipt.get("sanitization_report") or "")); sanitation = p2a.read_json(sanitation_path) if sanitation_path.is_file() else {}
            if not path.is_file() or p2a.sha256_file(path) != receipt.get("normalized_sha256"): row_faults.append(f"{receipt.get('label')}:archive_identity_invalid")
            if not sanitation_path.is_file() or p2a.sha256_file(sanitation_path) != receipt.get("sanitization_report_sha256"): row_faults.append(f"{receipt.get('label')}:sanitization_identity_invalid")
            row_faults.extend(f"{receipt.get('label')}:{fault}" for fault in owner.audit_closure_artifact(task, receipt, path, sanitation))
            if receipt.get("upstream_retained") is not False: row_faults.append(f"{receipt.get('label')}:upstream_retention_invalid")
            artifact_count += 1; member_requirement_count += len(receipt.get("required_relative_members", []))
        prior = predecessor_tasks.get(index, {})
        if index in replacement_indices:
            replacements += 1
            if row.get("repository") == prior.get("repository"): row_faults.append("replacement_repository_unchanged")
        else:
            replayed += 1
            prior_artifacts = {item.get("label"): item.get("normalized_sha256") for item in p2a.dicts(prior.get("artifacts"))}
            current_artifacts = {item.get("label"): item.get("normalized_sha256") for item in artifacts}
            if row.get("repository") != prior.get("repository") or current_artifacts != prior_artifacts: row_faults.append("unchanged_closure_replay_drift")
        faults.extend(f"task_{index}:{fault}" for fault in row_faults); audited_rows.append({"index": index, "repository": row.get("repository"), "artifact_count": len(artifacts), "faults": row_faults})
    if replayed != 59 or replacements != 3 or artifact_count != 124: faults.append("replay_or_replacement_denominator_invalid")
    authority = p2a.mapping(config.get("authority"))
    if not authority or any(value is not False for value in authority.values()): faults.append("authority_boundary_invalid")
    green = not faults
    return {"policy": POLICY, "audit_kind": "role-separated_rederivation", "created_utc": p2a.now(), "trigger_state": "GREEN" if green else "RED", "state": "REVISED_62_ROW_REPOSITORY_CLOSURES_ROLE_SEPARATELY_REDERIVED" if green else "REPOSITORY_CLOSURE_AUDIT_FAILED", "faults": sorted(set(faults)), "task_count": len(tasks), "archive_artifact_count": artifact_count, "required_member_receipt_count": member_requirement_count, "replayed_unchanged_task_count": replayed, "replacement_task_count": replacements, "replacement_indices": sorted(replacement_indices), "audited_rows": audited_rows, "network_calls": 0, "parent_target_or_evaluator_executions": 0, "local_model_calls": 0, "external_reference_calls": 0, "config": artifact(config_path), "producer_report": artifact(producer_path), "predecessor_report": artifact(predecessor_path), "maximum_inference": config.get("maximum_inference")}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)} if path.is_file() else {"path": p2a.rel(path), "sha256": ""}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in ("trigger_state", "state", "task_count", "archive_artifact_count", "required_member_receipt_count", "replayed_unchanged_task_count", "replacement_task_count", "replacement_indices", "network_calls", "parent_target_or_evaluator_executions", "local_model_calls", "external_reference_calls", "faults")}


if __name__ == "__main__": raise SystemExit(main())
