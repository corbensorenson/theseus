#!/usr/bin/env python3
"""Independently rederive the first retained VCM dependency closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa: E402

POLICY = "project_theseus_vcm_dependency_closure_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_dependency_closure_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = audit(config_path)
    p2a.write_json(p2a.resolve(args.out or p2a.read_json(config_path)["report"]), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    owner = p2a.resolve(str(config.get("owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != str(config.get("owner_sha256") or ""):
        faults.append("owner_binding_invalid")
    artifacts: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for name, raw in p2a.mapping(config.get("artifacts")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        paths[name] = path
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"artifact_binding_invalid:{name}")
            artifacts[name] = {}
        else:
            artifacts[name] = p2a.read_json(path)
    canary = artifacts.get("canary_report", {})
    canary_config = artifacts.get("canary_config", {})
    sandbox = artifacts.get("trusted_build_canaries", {})
    if canary.get("trigger_state") != "GREEN" or canary.get("state") != "V2_TASK_3_DEPENDENCY_CACHE_ACQUIRED_AND_OFFLINE_REPLAY_QUALIFIED":
        faults.append("canary_report_not_green")
    if canary_config.get("policy") != "project_theseus_vcm_dependency_prefetch_canary_v2":
        faults.append("canary_config_invalid")
    if sandbox.get("trigger_state") != "GREEN" or "network_denial_and_write_confinement" not in p2a.strings(sandbox.get("qualified_scopes")):
        faults.append("trusted_sandbox_network_denial_not_green")
    authority = p2a.mapping(config.get("authority"))
    for key, value in authority.items():
        if value is not (key == "static_audit_authorized"):
            faults.append(f"authority_invalid:{key}")

    receipts = p2a.mapping(canary.get("receipts"))
    online = p2a.mapping(receipts.get("online_acquisition"))
    offline = p2a.mapping(receipts.get("offline_replay"))
    command_contract = p2a.mapping(canary_config.get("commands"))
    online_tail = p2a.strings(command_contract.get("online_acquisition_args"))
    offline_tail = p2a.strings(command_contract.get("offline_replay_args"))
    command_checks = {
        "online_returncode_zero": online.get("returncode") == 0,
        "offline_returncode_zero": offline.get("returncode") == 0,
        "online_boundary_clear": online.get("boundary_hit") is False,
        "offline_boundary_clear": offline.get("boundary_hit") is False,
        "online_lifecycle_flag": online.get("lifecycle_scripts_disabled_by_exact_cli_flag") is True,
        "offline_lifecycle_flag": offline.get("lifecycle_scripts_disabled_by_exact_cli_flag") is True,
        "online_network_enabled_phase": online.get("network_denied") is False,
        "offline_network_denied_phase": offline.get("network_denied") is True,
        "online_args_exact": p2a.strings(online.get("command"))[2:-2] == online_tail,
        "offline_args_exact": p2a.strings(offline.get("command"))[2:-2] == offline_tail,
        "online_cache_flag_exact": p2a.strings(online.get("command"))[-2] == "--cache",
        "offline_cache_flag_exact": p2a.strings(offline.get("command"))[-2] == "--cache",
    }
    faults.extend(f"command_check_failed:{key}" for key, passed in command_checks.items() if not passed)

    store = p2a.resolve(str(config.get("retained_store") or ""))
    cache_receipt, cache_faults = base.inspect_cache(store, canary_config) if store.is_dir() else ({}, ["retained_store_absent"])
    faults.extend(cache_faults)
    reported_cache = p2a.mapping(receipts.get("retained_store"))
    cache_checks = {
        "exact_two_content_blobs": cache_receipt.get("content_blob_count") == 2,
        "all_lock_integrities_matched": cache_receipt.get("matched_lock_integrity_count") == cache_receipt.get("expected_lock_integrity_count") == 2,
        "no_missing_lock_integrities": cache_receipt.get("missing_lock_integrity_packages") == [],
        "cache_bytes_match_report": cache_receipt.get("total_cache_bytes") == reported_cache.get("total_cache_bytes"),
        "cache_files_match_report": cache_receipt.get("total_cache_file_count") == reported_cache.get("total_cache_file_count"),
        "content_identity_matches_report": cache_receipt.get("content_blobs_sha256") == reported_cache.get("content_blobs_sha256"),
    }
    faults.extend(f"cache_check_failed:{key}" for key, passed in cache_checks.items() if not passed)

    target = p2a.mapping(p2a.mapping(canary_config.get("archives")).get("target"))
    source_identity = archive_tree_identity(p2a.resolve(str(target.get("path") or "")), str(target.get("archive_root") or ""))
    before_source = p2a.mapping(receipts.get("repository_source_before"))
    after_source = p2a.mapping(receipts.get("repository_source_after"))
    source_checks = {
        "before_after_identical": before_source == after_source,
        "archive_rederivation_matches_before": source_identity == before_source,
        "archive_rederivation_matches_after": source_identity == after_source,
    }
    faults.extend(f"source_check_failed:{key}" for key, passed in source_checks.items() if not passed)

    expected_dependencies = p2a.mapping(p2a.mapping(canary_config.get("task")).get("expected_dependencies"))
    online_dependencies = p2a.mapping(receipts.get("online_installed_dependencies"))
    offline_dependencies = p2a.mapping(receipts.get("offline_installed_dependencies"))
    dependency_checks = {
        "online_faults_empty": online_dependencies.get("faults") == [],
        "offline_faults_empty": offline_dependencies.get("faults") == [],
        "online_expected_exact": online_dependencies.get("expected") == expected_dependencies,
        "offline_expected_exact": offline_dependencies.get("expected") == expected_dependencies,
        "online_offline_observed_exact": online_dependencies.get("observed") == offline_dependencies.get("observed"),
        "observed_versions_exact": all(
            p2a.mapping(p2a.mapping(online_dependencies.get("observed")).get(name)).get("version") == version
            for name, version in expected_dependencies.items()
        ),
    }
    faults.extend(f"dependency_check_failed:{key}" for key, passed in dependency_checks.items() if not passed)

    limits = p2a.mapping(canary_config.get("limits"))
    minimum_free = int(limits.get("minimum_free_bytes_after_execution") or 0)
    current_free = shutil.disk_usage(ROOT).free
    storage_checks = {
        "run_free_before_above_reserve": int(receipts.get("free_bytes_before") or 0) >= minimum_free,
        "run_free_after_above_reserve": int(receipts.get("free_bytes_after") or 0) >= minimum_free,
        "current_free_above_reserve": current_free >= minimum_free,
        "cache_below_retention_ceiling": int(cache_receipt.get("total_cache_bytes") or 0) <= int(limits.get("maximum_retained_bytes") or 0),
    }
    faults.extend(f"storage_check_failed:{key}" for key, passed in storage_checks.items() if not passed)

    zero_counters = {
        "repository_runner_executions": canary.get("repository_runner_executions"),
        "parent_target_or_evaluator_executions": canary.get("parent_target_or_evaluator_executions"),
        "candidate_or_control_calls": canary.get("candidate_or_control_calls"),
        "external_reference_calls": canary.get("external_reference_calls"),
    }
    if any(value != 0 for value in zero_counters.values()):
        faults.append("downstream_zero_counter_invalid")
    observations = {
        "task_index": 3,
        "content_blob_count": cache_receipt.get("content_blob_count"),
        "matched_lock_integrity_count": cache_receipt.get("matched_lock_integrity_count"),
        "expected_lock_integrity_count": cache_receipt.get("expected_lock_integrity_count"),
        "retained_cache_bytes": cache_receipt.get("total_cache_bytes"),
        "retained_cache_file_count": cache_receipt.get("total_cache_file_count"),
        "source_file_count": source_identity.get("file_count"),
        "source_bytes": source_identity.get("bytes"),
        "current_free_bytes": current_free,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
    }
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "TASK_3_DEPENDENCY_CLOSURE_INDEPENDENTLY_REDERIVED" if not faults else "TASK_3_DEPENDENCY_CLOSURE_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "config": base.identity(config_path),
        "artifacts": {name: base.identity(path) for name, path in paths.items()},
        "retained_store": p2a.rel(store),
        "command_checks": command_checks,
        "cache_checks": cache_checks,
        "source_checks": source_checks,
        "dependency_checks": dependency_checks,
        "storage_checks": storage_checks,
        "rederived_cache": cache_receipt,
        "rederived_source": source_identity,
        "observations": observations,
        "static_audit_only": True,
        "network_or_dependency_execution_performed": False,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def archive_tree_identity(archive: Path, archive_root: str) -> dict[str, Any]:
    rows = []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile() or not member.name.startswith(archive_root + "/"):
                continue
            relative = member.name[len(archive_root) + 1:]
            extracted = handle.extractfile(member)
            payload = extracted.read() if extracted else b""
            rows.append({"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    # The canary's filesystem walk uses pathlib component ordering rather than
    # raw string ordering (for example, `current/` precedes `current.json`).
    rows.sort(key=lambda row: PurePosixPath(row["path"]))
    return {"file_count": len(rows), "bytes": sum(row["bytes"] for row in rows), "identity_sha256": base.digest_json(rows)}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "observations", "static_audit_only",
        "network_or_dependency_execution_performed", "repository_runner_executions",
        "parent_target_or_evaluator_executions", "candidate_or_control_calls",
        "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
