#!/usr/bin/env python3
"""Materialize all frozen parent-only VCM stores from one audited safe manifest."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_parent_only_materializer as base  # noqa: E402

POLICY = "project_theseus_vcm_parent_store_batch_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_parent_store_batch.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--store-out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report, store = materialize(path)
    store_path = p2a.resolve(args.store_out or str(cfg.get("store_out") or ""))
    report_path = p2a.resolve(args.out or str(cfg.get("report") or ""))
    p2a.write_json(store_path, store)
    report["store_artifact"] = {"path": p2a.rel(store_path), "sha256": p2a.sha256_file(store_path), "bytes": store_path.stat().st_size}
    p2a.write_json(report_path, report)
    print(json.dumps({key: report.get(key) for key in ("trigger_state", "state", "faults", "row_count", "regular_file_count", "text_page_count", "resource_preflight", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def materialize(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    validate_binding(cfg, "owner", "owner_sha256", Path(__file__).resolve(), faults)
    validate_binding(cfg, "base_materializer", "base_materializer_sha256", Path(base.__file__).resolve(), faults)
    if cfg.get("policy") != POLICY or cfg.get("broad_parent_effect_root") != "repository":
        faults.append("batch_policy_or_effect_root_invalid")
    manifest, rows = load_manifest(cfg, faults)
    resource = resource_preflight(rows, cfg, faults)
    store_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for binding in rows:
        built, row_faults = base.build_row(binding, path, cfg)
        faults.extend(row_faults)
        request_id = str(built.get("request_id") or "")
        if not request_id or request_id in seen:
            faults.append("request_id_invalid_or_duplicate")
        seen.add(request_id)
        store_rows.append(p2a.mapping(built.get("store_row")))
        report_rows.append(p2a.mapping(built.get("report_row")))
    expected = int(cfg.get("expected_row_count") or 0)
    if len(report_rows) != expected:
        faults.append("row_count_invalid")
    ready = not faults
    store = {
        "policy": base.STORE_POLICY,
        "created_utc": p2a.now(),
        "source_boundary": "exact_immutable_parent_archives_only_from_role_audited_target_free_manifest",
        "content_storage": "archive_backed_no_duplicate_payload",
        "selector_policy": base.SELECTOR_POLICY,
        "manifest_sha256": p2a.mapping(cfg.get("manifest")).get("sha256"),
        "rows": store_rows,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
    }
    report = {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if ready else "RED",
        "state": "K2_05_ALL_62_PARENT_STORES_MATERIALIZED" if ready else "K2_05_PARENT_STORE_BATCH_FAILED",
        "faults": sorted(set(faults)),
        "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)},
        "manifest": {"path": p2a.mapping(cfg.get("manifest")).get("path"), "sha256": p2a.mapping(cfg.get("manifest")).get("sha256"), "policy": manifest.get("policy")},
        "row_count": len(report_rows),
        "regular_file_count": sum(int(row.get("regular_file_count") or 0) for row in report_rows),
        "text_page_count": sum(int(row.get("text_page_count") or 0) for row in report_rows),
        "rows": report_rows,
        "resource_preflight": resource,
        "information_flow": {"target_archive_read": False, "target_diff_read": False, "target_selected_path_read": False, "allowed_effect_paths_present": False, "complete_parent_text_frontier_retained_without_selection_cap": True},
        "panel_admitted": False,
        "network_or_dependency_execution_performed": False,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": cfg.get("maximum_inference"),
    }
    return report, store


def load_manifest(cfg: dict[str, Any], faults: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binding = p2a.mapping(cfg.get("manifest"))
    path = p2a.resolve(str(binding.get("path") or ""))
    if not path.is_file() or p2a.sha256_file(path) != binding.get("sha256"):
        faults.append("manifest_binding_invalid")
        return {}, []
    manifest = p2a.read_json(path)
    if manifest.get("policy") != binding.get("required_policy") or manifest.get("broad_parent_effect_root") != "repository":
        faults.append("manifest_policy_or_effect_root_invalid")
    audit_binding = p2a.mapping(cfg.get("manifest_role_audit"))
    audit_path = p2a.resolve(str(audit_binding.get("path") or ""))
    if not audit_path.is_file() or p2a.sha256_file(audit_path) != audit_binding.get("sha256"):
        faults.append("manifest_role_audit_binding_invalid")
    else:
        audit = p2a.read_json(audit_path)
        if audit.get("trigger_state") != "GREEN" or audit.get("state") != audit_binding.get("required_state") or audit.get("parent_only_manifest", {}).get("sha256") != binding.get("sha256"):
            faults.append("manifest_role_audit_state_invalid")
    return manifest, p2a.dicts(manifest.get("rows"))


def resource_preflight(rows: list[dict[str, Any]], cfg: dict[str, Any], faults: list[str]) -> dict[str, Any]:
    archive_bytes = sum(p2a.resolve(str(row.get("parent_archive") or "")).stat().st_size for row in rows if p2a.resolve(str(row.get("parent_archive") or "")).is_file())
    contract = p2a.mapping(cfg.get("resource_contract"))
    reserve = int(contract.get("host_reserve_bytes") or 0)
    multiplier = int(contract.get("archive_bytes_to_output_upper_multiplier") or 0)
    projected = archive_bytes * multiplier
    free = shutil.disk_usage(ROOT).free
    safe = max(0, free - reserve)
    ready = reserve > 0 and multiplier > 0 and projected <= safe
    if not ready:
        faults.append("parent_store_batch_resource_preflight_closed")
    return {"parent_archive_count": len(rows), "parent_archive_bytes": archive_bytes, "projected_output_upper_bytes": projected, "host_free_bytes": free, "host_reserve_bytes": reserve, "safe_incremental_headroom_bytes": safe, "execution_gate": "OPEN" if ready else "CLOSED_HOST_STORAGE", "content_payload_duplication": False}


def validate_binding(cfg: dict[str, Any], path_key: str, hash_key: str, expected: Path, faults: list[str]) -> None:
    actual = p2a.resolve(str(cfg.get(path_key) or ""))
    if actual != expected or not actual.is_file() or p2a.sha256_file(actual) != cfg.get(hash_key):
        faults.append(f"{path_key}_binding_invalid")


if __name__ == "__main__":
    raise SystemExit(main())
