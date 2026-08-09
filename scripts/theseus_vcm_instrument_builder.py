#!/usr/bin/env python3
"""Build one row-oriented VCM instrument ledger from manifest-bound evidence."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa: E402
import theseus_vcm_dependency_prefetch_canary_v2 as bounded  # noqa: E402

POLICY = "project_theseus_vcm_instrument_builder_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_instrument_builder.json"
RESOURCE_FIELDS = (
    "projected_download_bytes",
    "projected_installed_bytes",
    "projected_peak_temporary_bytes",
    "shared_store_deduplicated_bytes",
    "host_free_bytes",
    "host_reserve_bytes",
    "projected_wall_time",
    "untrusted_build_risk_class",
)
HOST_METADATA_NAMES = frozenset({".DS_Store"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute-risks", action="store_true")
    parser.add_argument("--execute-typescript-repair", action="store_true")
    parser.add_argument("--preflight-batch", action="store_true")
    parser.add_argument("--compile-segment-plan", action="store_true")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    if args.compile_segment_plan:
        cfg = p2a.read_json(path)
        result, manifest = compile_segment_plan(path)
        manifest_path = p2a.resolve(str(cfg.get("manifest_out") or ""))
        p2a.write_json(manifest_path, manifest)
        result["parent_only_manifest_artifact"] = base.identity(manifest_path)
        p2a.write_json(p2a.resolve(args.out or str(cfg.get("report") or "")), result)
        print(json.dumps({key: result.get(key) for key in ("trigger_state", "state", "faults", "task_count", "segment_counts", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
        return 0 if result["trigger_state"] == "GREEN" else 2
    result = preflight_batch(path) if args.preflight_batch else execute_typescript_repair(path) if args.execute_typescript_repair else execute_risks(path) if args.execute_risks else build(path)
    p2a.write_json(p2a.resolve(args.out or p2a.read_json(path)["report"]), result)
    print(json.dumps(summary(result), indent=2, sort_keys=True))
    return 0 if result["trigger_state"] == "GREEN" else 2


def build(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != POLICY:
        faults.append("policy_invalid")
    owner = p2a.resolve(str(cfg.get("owner") or ""))
    if (
        owner != Path(__file__).resolve()
        or not owner.is_file()
        or p2a.sha256_file(owner) != cfg.get("owner_sha256")
    ):
        faults.append("owner_binding_invalid")
    allowed = {
        "static_evidence_replay_authorized",
        "four_risk_canary_executions_authorized",
        "network_denied_replays_authorized",
        "untrusted_parent_typescript_transpilation_authorized",
        "untrusted_parent_rust_compilation_authorized",
        "shared_store_retention_authorized",
        "batch_resource_preflight_authorized",
    }
    for key, value in p2a.mapping(cfg.get("authority")).items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")
    sandbox = p2a.mapping(p2a.mapping(cfg.get("tools")).get("sandbox_exec"))
    sandbox_path = p2a.resolve(str(sandbox.get("path") or ""))
    if not sandbox_path.is_file() or p2a.sha256_file(sandbox_path) != sandbox.get("sha256"):
        faults.append("sandbox_tool_binding_invalid")

    schedule_binding = p2a.mapping(cfg.get("schedule"))
    schedule_path = p2a.resolve(str(schedule_binding.get("path") or ""))
    if (
        not schedule_path.is_file()
        or p2a.sha256_file(schedule_path) != schedule_binding.get("sha256")
    ):
        faults.append("schedule_binding_invalid")
        schedule = {}
    else:
        schedule = {
            int(row["schedule_ordinal"]): row
            for row in p2a.dicts(p2a.read_json(schedule_path).get("schedule"))
        }

    store = p2a.mapping(cfg.get("store_contract"))
    expected_roots = {
        manager: f"runtime/vcm_evaluator/dependency_store/shared/{manager}"
        for manager in ("npm", "pnpm", "cargo", "uv", "bun", "yarn")
    }
    if p2a.mapping(store.get("manager_roots")) != expected_roots:
        faults.append("shared_manager_roots_invalid")
    if store.get("per_task_duplicate_package_cache_authorized") is not False:
        faults.append("duplicate_cache_policy_invalid")
    if store.get("installed_environments_are_disposable") is not True:
        faults.append("disposable_environment_policy_invalid")

    rows: list[dict[str, Any]] = []
    closure_ids: set[str] = set()
    for raw in p2a.dicts(cfg.get("rows")):
        closure_id = str(raw.get("closure_id") or "")
        if not closure_id or closure_id in closure_ids:
            faults.append(f"closure_id_invalid:{closure_id}")
        closure_ids.add(closure_id)
        ordinal = int(raw.get("schedule_ordinal") or 0)
        scheduled = p2a.mapping(schedule.get(ordinal))
        if (
            scheduled.get("index") != raw.get("task_index")
            or scheduled.get("manager") != raw.get("manager")
        ):
            faults.append(f"schedule_row_mismatch:{closure_id}")
        audit_binding = p2a.mapping(raw.get("audit"))
        audit_path = p2a.resolve(str(audit_binding.get("path") or ""))
        if (
            not audit_path.is_file()
            or p2a.sha256_file(audit_path) != audit_binding.get("sha256")
        ):
            faults.append(f"audit_binding_invalid:{closure_id}")
            audit = {}
        else:
            audit = p2a.read_json(audit_path)
        if (
            audit.get("trigger_state") != "GREEN"
            or audit.get("state") != raw.get("required_audit_state")
            or audit.get("static_audit_only") is not True
            or audit.get("network_or_dependency_execution_performed") is not False
        ):
            faults.append(f"audit_state_invalid:{closure_id}")
        for key in (
            "repository_runner_executions",
            "parent_target_or_evaluator_executions",
            "candidate_or_control_calls",
            "external_reference_calls",
        ):
            if audit.get(key) != 0:
                faults.append(f"downstream_counter_invalid:{closure_id}:{key}")

        projection = p2a.mapping(raw.get("projection"))
        receipt = {
            "closure_id": closure_id,
            "task_index": raw.get("task_index"),
            "schedule_ordinal": ordinal,
            "manager": raw.get("manager"),
            "side": raw.get("side"),
            "state": "QUALIFIED_EXISTING_CLOSURE_REPLAY",
            "audit": base.identity(audit_path) if audit_path.is_file() else {},
            "audit_state": audit.get("state"),
            "dependency_denominator_kind": raw.get("dependency_denominator_kind"),
            "dependency_denominator": project(audit, projection.get("dependency_denominator")),
            "retained_store_bytes": project(audit, projection.get("retained_store_bytes")),
            "retained_store_file_count": project(audit, projection.get("retained_store_file_count")),
            "retained_store_identity_sha256": project(audit, projection.get("retained_store_identity_sha256")),
            "retained_store_identity_scope": raw.get("retained_store_identity_scope"),
            "source_bytes": project(audit, projection.get("source_bytes")),
            "source_file_count": project(audit, projection.get("source_file_count")),
            "source_identity_sha256": project(audit, projection.get("source_identity_sha256")),
            "forward_shared_store_root": expected_roots.get(str(raw.get("manager"))),
            "historical_store_topology_reused_for_forward_execution": False,
            "network_or_dependency_execution_performed": False,
            "repository_runner_executions": 0,
            "parent_target_or_evaluator_executions": 0,
            "candidate_or_control_calls": 0,
            "external_reference_calls": 0,
        }
        required_receipt = (
            "dependency_denominator",
            "retained_store_bytes",
            "retained_store_file_count",
            "retained_store_identity_sha256",
            "source_bytes",
            "source_file_count",
            "source_identity_sha256",
        )
        for key in required_receipt:
            if receipt.get(key) in (None, ""):
                faults.append(f"projection_missing:{closure_id}:{key}")
        rows.append(receipt)

    if len(rows) != int(cfg.get("expected_replayed_closure_count") or -1):
        faults.append("replayed_closure_count_invalid")
    managers = sorted({str(row.get("manager")) for row in rows})
    if managers != p2a.strings(cfg.get("expected_replayed_managers")):
        faults.append("manager_coverage_invalid")

    host_free = shutil.disk_usage(ROOT).free
    resource = {
        "state": "STATIC_REPLAY_ONLY_K2_03_PROJECTION_PENDING",
        "projected_download_bytes": 0,
        "projected_installed_bytes": 0,
        "projected_peak_temporary_bytes": 0,
        "shared_store_deduplicated_bytes": None,
        "host_free_bytes": host_free,
        "host_reserve_bytes": int(p2a.mapping(cfg.get("resource_contract")).get("host_reserve_bytes") or 0),
        "projected_wall_time": 0,
        "untrusted_build_risk_class": "none_static_replay",
    }
    if set(resource) - {"state"} != set(RESOURCE_FIELDS):
        faults.append("resource_schema_invalid")
    if host_free < resource["host_reserve_bytes"]:
        faults.append("host_reserve_boundary_hit")
    risk_plan = validate_risk_plan(cfg, schedule, host_free, faults)
    if p2a.strings(p2a.mapping(cfg.get("risk_resume")).get("content_identity_excluded_host_metadata_names")) != sorted(HOST_METADATA_NAMES):
        faults.append("host_metadata_exclusion_contract_invalid")
    if p2a.mapping(cfg.get("risk_resume")).get("bun_original_cache_root") != "/private/tmp/theseus-vcm-generic-risks-p3xg3b97/bun-cache":
        faults.append("bun_original_cache_root_binding_invalid")

    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "GENERIC_VCM_INSTRUMENT_EXISTING_CLOSURES_REPLAYED" if not faults else "GENERIC_VCM_INSTRUMENT_BUILDER_FAILED",
        "faults": sorted(set(faults)),
        "config": base.identity(path),
        "schedule": base.identity(schedule_path) if schedule_path.is_file() else {},
        "replayed_closure_count": len(rows),
        "replayed_managers": managers,
        "rows": rows,
        "store_contract": store,
        "resource_preflight": resource,
        "risk_canary_plan": risk_plan,
        "risk_resume": cfg.get("risk_resume", {}),
        "prior_risk_attempts": cfg.get("prior_risk_attempts", []),
        "static_evidence_replay_only": True,
        "network_or_dependency_execution_performed": False,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": cfg.get("maximum_inference"),
    }


def preflight_batch(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Project the full K2.05 batch without fetching, installing, or executing."""
    before = build(path)
    cfg = p2a.read_json(path)
    plan = p2a.mapping(cfg.get("batch_preflight"))
    faults = list(p2a.strings(before.get("faults")))
    if plan.get("state") != "PROSPECTIVE_K2_05_BATCH_RESOURCE_PREFLIGHT_ZERO_EXECUTION":
        faults.append("batch_preflight_identity_invalid")
    bindings: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for raw in p2a.dicts(plan.get("inputs")):
        name = str(raw.get("id") or "")
        source = p2a.resolve(str(raw.get("path") or ""))
        if not name or name in bindings:
            faults.append(f"batch_input_id_invalid:{name}")
            continue
        bindings[name] = raw
        if not source.is_file() or p2a.sha256_file(source) != raw.get("sha256"):
            faults.append(f"batch_input_binding_invalid:{name}")
            payloads[name] = {}
        else:
            payloads[name] = p2a.read_json(source)

    schedule_rows = p2a.dicts(payloads.get("dependency_plan", {}).get("rows"))
    runner_rows = p2a.dicts(payloads.get("runner_inventory", {}).get("rows"))
    closure_rows = p2a.dicts(payloads.get("repository_closures", {}).get("tasks"))
    materializer_audit = payloads.get("parent_materializer_audit", {})
    expected = int(plan.get("expected_task_count") or 0)
    if [len(schedule_rows), len(runner_rows), len(closure_rows)] != [expected, expected, expected]:
        faults.append("batch_task_denominator_invalid")
    if materializer_audit.get("trigger_state") != "GREEN":
        faults.append("representative_parent_materializer_not_green")

    coefficients = p2a.mapping(plan.get("observed_manager_bytes_per_locked_entry"))
    manager_counts: dict[str, int] = {}
    manager_entries: dict[str, int] = {}
    manager_projection: dict[str, dict[str, Any]] = {}
    for row in schedule_rows:
        manager = str(row.get("manager") or "")
        manager_counts[manager] = manager_counts.get(manager, 0) + 1
        manager_entries[manager] = manager_entries.get(manager, 0) + int(row.get("estimated_locked_package_count") or 0)
    projected_store_upper = 0
    largest_row_store_upper = 0
    for manager in ("npm", "pnpm", "cargo", "uv", "bun", "yarn"):
        per_entry = int(coefficients.get(manager) or 0)
        if per_entry <= 0:
            faults.append(f"manager_projection_basis_invalid:{manager}")
        entries = manager_entries.get(manager, 0)
        store_upper = entries * per_entry
        projected_store_upper += store_upper
        manager_row_max = max(
            (int(row.get("estimated_locked_package_count") or 0) for row in schedule_rows if row.get("manager") == manager),
            default=0,
        ) * per_entry
        largest_row_store_upper = max(largest_row_store_upper, manager_row_max)
        manager_projection[manager] = {
            "task_count": manager_counts.get(manager, 0),
            "locked_entry_count": entries,
            "observed_upper_bytes_per_entry": per_entry,
            "shared_store_upper_without_cross_lock_deduplication_bytes": store_upper,
            "largest_single_row_store_upper_bytes": manager_row_max,
        }
    retained_roots = p2a.mapping(p2a.mapping(cfg.get("store_contract")).get("manager_roots"))
    retained_shared_bytes = sum(directory_bytes(p2a.resolve(value)) for value in retained_roots.values())
    projected_new_download_upper = max(0, projected_store_upper - retained_shared_bytes)
    install_multiplier = float(plan.get("serial_disposable_install_multiplier") or 0.0)
    projected_peak_install = int(largest_row_store_upper * install_multiplier)
    projected_peak_temp = int(plan.get("maximum_serial_temporary_bytes") or 0)
    host_free = shutil.disk_usage(ROOT).free
    reserve = int(p2a.mapping(cfg.get("resource_contract")).get("host_reserve_bytes") or 0)
    safe_headroom = max(0, host_free - reserve)
    required_incremental_peak = projected_new_download_upper + projected_peak_install + projected_peak_temp
    deficit = max(0, required_incremental_peak - safe_headroom)
    execution_ready = not faults and deficit == 0
    resource = {
        "projected_download_bytes": projected_new_download_upper,
        "projected_installed_bytes": projected_peak_install,
        "projected_peak_temporary_bytes": projected_peak_temp,
        "shared_store_deduplicated_bytes": {
            "observed_retained_lower_bound": retained_shared_bytes,
            "upper_bound_without_cross_lock_deduplication": projected_store_upper,
            "exact_future_deduplication_unknown_before_acquisition": True,
        },
        "host_free_bytes": host_free,
        "host_reserve_bytes": reserve,
        "projected_wall_time": int(plan.get("maximum_batch_wall_seconds") or 0),
        "untrusted_build_risk_class": "serialized_prequalified_ecosystem_classes_only",
    }
    if set(resource) != set(RESOURCE_FIELDS):
        faults.append("batch_resource_schema_invalid")
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "K2_05_BATCH_PREFLIGHT_GREEN_EXECUTION_READY" if execution_ready else "K2_05_BATCH_PREFLIGHT_GREEN_EXECUTION_BLOCKED_HOST_STORAGE" if not faults else "K2_05_BATCH_PREFLIGHT_INVALID",
        "faults": sorted(set(faults)),
        "batch_preflight": {
            "task_count": expected,
            "locked_task_count": sum(manager_counts.get(manager, 0) for manager in ("npm", "pnpm", "cargo", "uv", "bun", "yarn")),
            "static_task_count": manager_counts.get("trusted_runtime_or_harness", 0),
            "immutable_resolution_task_count": manager_counts.get("resolver_required", 0),
            "locked_entry_count": sum(manager_entries.values()),
            "manager_projection": manager_projection,
            "resource_projection": resource,
            "safe_incremental_headroom_bytes": safe_headroom,
            "required_incremental_peak_bytes": required_incremental_peak,
            "host_storage_deficit_bytes": deficit,
            "execution_ready": execution_ready,
            "execution_gate": "OPEN" if execution_ready else "CLOSED_HOST_STORAGE",
            "disposition": "READY_FOR_SERIAL_BATCH" if execution_ready else "INCONCLUSIVE_EXPERIMENT_HOST_STORAGE_PREFLIGHT",
            "no_dedupe_upper_bound_is_safety_projection_not_expected_spend": True,
        },
        "static_evidence_replay_only": True,
        "network_or_dependency_execution_performed": False,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": plan.get("maximum_inference"),
    }


def compile_segment_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile one target-free parent manifest and segmented K2.05 schedule."""
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != "project_theseus_vcm_k2_segment_plan_v1":
        faults.append("segment_policy_invalid")
    owner = p2a.resolve(str(cfg.get("owner") or ""))
    if owner != Path(__file__).resolve() or p2a.sha256_file(owner) != cfg.get("owner_sha256"):
        faults.append("segment_owner_binding_invalid")
    payloads: dict[str, dict[str, Any]] = {}
    for binding in p2a.dicts(cfg.get("sources")):
        source_id = str(binding.get("id") or "")
        source = p2a.resolve(str(binding.get("path") or ""))
        if not source_id or source_id in payloads:
            faults.append(f"segment_source_id_invalid:{source_id}")
            continue
        if not source.is_file() or p2a.sha256_file(source) != binding.get("sha256"):
            faults.append(f"segment_source_binding_invalid:{source_id}")
            payloads[source_id] = {}
        else:
            payloads[source_id] = p2a.read_json(source)
    panel_rows = index_rows(payloads.get("source_panel", {}).get("assembled_rows"), "index")
    closure_rows = index_rows(payloads.get("repository_closures", {}).get("tasks"), "campaign_index")
    runner_rows = index_rows(payloads.get("runner_inventory", {}).get("rows"), "index")
    class_rows = index_rows(payloads.get("dependency_classes", {}).get("rows"), "index")
    plan_rows = index_rows(payloads.get("dependency_plan", {}).get("rows"), "index")
    expected = int(cfg.get("expected_task_count") or 0)
    if any(len(rows) != expected for rows in (panel_rows, closure_rows, runner_rows, class_rows, plan_rows)):
        faults.append("segment_source_denominator_invalid")

    safe_rows: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    for index in range(1, expected + 1):
        panel = panel_rows.get(index, {})
        closure = closure_rows.get(index, {})
        runner = runner_rows.get(index, {})
        classification = class_rows.get(index, {})
        dependency = plan_rows.get(index, {})
        repositories = {str(row.get("repository") or "") for row in (panel, closure, runner, classification, dependency)}
        if len(repositories) != 1 or "" in repositories:
            faults.append(f"segment_repository_alignment_invalid:{index}")
        parent = next((row for row in p2a.dicts(closure.get("artifacts")) if row.get("label") == "parent"), {})
        if (
            str(parent.get("revision") or "") != str(panel.get("base_revision") or "")
            or not str(parent.get("normalized") or "").endswith("_parent.tar.gz")
            or "target" in str(parent.get("normalized") or "").lower()
        ):
            faults.append(f"segment_parent_alignment_invalid:{index}")
        request = str(panel.get("natural_language_request") or "")
        request_sha = str(panel.get("natural_language_request_sha256") or "")
        if p2a.sha256_text(request) != request_sha:
            faults.append(f"segment_request_hash_invalid:{index}")
        request_id = base.digest_json({"request_sha256": request_sha, "parent_archive_sha256": parent.get("normalized_sha256"), "policy": "project_theseus_vcm_parent_only_batch_manifest_v1"})
        if request_id in request_ids:
            faults.append(f"segment_request_id_duplicate:{index}")
        request_ids.add(request_id)
        safe_rows.append({
            "natural_language_request": request,
            "natural_language_request_sha256": request_sha,
            "parent_archive": parent.get("normalized"),
            "parent_archive_sha256": parent.get("normalized_sha256"),
            "parent_archive_root": parent.get("source_archive_root"),
            "parent_revision": parent.get("revision"),
            "license_spdx": panel.get("license_spdx"),
            "sanitization_report": parent.get("sanitization_report"),
            "sanitization_report_sha256": parent.get("sanitization_report_sha256"),
        })
        manager = str(dependency.get("manager") or "")
        segment = "static_no_project_lock" if manager == "trusted_runtime_or_harness" else "immutable_resolution_required" if manager == "resolver_required" else "locked_closure"
        schedule.append({
            "panel_index": index,
            "request_id": request_id,
            "segment": segment,
            "manager": manager,
            "dependency_class": dependency.get("dependency_class"),
            "evaluator_execution_ready": classification.get("evaluator_execution_ready") is True,
            "parent_store_materialization_ready": True,
            "panel_admission": "withheld_until_all_62_rows_complete",
        })
    segment_counts = {name: sum(row["segment"] == name for row in schedule) for name in ("static_no_project_lock", "immutable_resolution_required", "locked_closure")}
    expected_counts = p2a.mapping(cfg.get("expected_segment_counts"))
    if segment_counts != expected_counts:
        faults.append("segment_counts_invalid")
    manifest = {
        "policy": "project_theseus_vcm_parent_only_batch_manifest_v1",
        "created_utc": p2a.now(),
        "source_boundary": "exact_parent_snapshots_and_authoritative_v3_natural_requests_only",
        "broad_parent_effect_root": "repository",
        "candidate_visible_fields": ["natural_language_request", "callable_signature_when_present", "broad_parent_effect_root", "arm_specific_model_visible_context"],
        "rows": safe_rows,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
    }
    return ({
        "policy": "project_theseus_vcm_k2_segment_plan_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "K2_05_TARGET_FREE_SEGMENT_PLAN_COMPILED" if not faults else "K2_05_SEGMENT_PLAN_FAILED",
        "faults": sorted(set(faults)),
        "config": base.identity(path),
        "task_count": len(schedule),
        "segment_counts": segment_counts,
        "schedule": schedule,
        "parent_only_manifest_projection_sha256": base.digest_json(manifest),
        "candidate_packet_materialization_opened": False,
        "panel_admitted": False,
        "partial_panel_admission_forbidden": True,
        "target_archive_read_by_candidate_path": False,
        "target_derived_selector_input_present": False,
        "network_or_dependency_execution_performed": False,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": cfg.get("maximum_inference"),
    }, manifest)


def index_rows(raw: Any, key: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in p2a.dicts(raw):
        index = int(row.get(key) or 0)
        if index > 0 and index not in rows:
            rows[index] = row
    return rows


def directory_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and item.name not in HOST_METADATA_NAMES)


def validate_risk_plan(
    cfg: dict[str, Any], schedule: dict[int, dict[str, Any]], host_free: int, faults: list[str]
) -> dict[str, Any]:
    plan = p2a.mapping(cfg.get("risk_canary_plan"))
    if plan.get("state") != "PROSPECTIVELY_SEALED_GENERIC_RISK_EXECUTOR_V7_NARROW_TYPESCRIPT_MECHANICS" or plan.get("campaign_id") != "k2_03_generic_ecosystem_risk_canaries_v7":
        faults.append("risk_campaign_identity_invalid")
    rows = p2a.dicts(plan.get("rows"))
    if [row.get("risk_class") for row in rows] != [
        "bun_real_lock_install",
        "yarn_real_lock_install",
        "typescript_parent_repository_transpilation",
        "rust_parent_repository_untrusted_compilation",
    ]:
        faults.append("risk_class_order_or_coverage_invalid")
    observed = []
    for row in rows:
        risk_id = str(row.get("risk_id") or "")
        ordinal = int(row.get("schedule_ordinal") or 0)
        scheduled = p2a.mapping(schedule.get(ordinal))
        if scheduled.get("index") != row.get("task_index") or scheduled.get("manager") != row.get("manager"):
            faults.append(f"risk_schedule_binding_invalid:{risk_id}")
        archive = p2a.mapping(row.get("parent_archive"))
        archive_path = p2a.resolve(str(archive.get("path") or ""))
        if not archive_path.is_file() or p2a.sha256_file(archive_path) != archive.get("sha256"):
            faults.append(f"risk_archive_binding_invalid:{risk_id}")
        tool = p2a.mapping(row.get("tool"))
        tool_path = p2a.resolve(str(tool.get("path") or ""))
        if not tool_path.is_file() or p2a.sha256_file(tool_path) != tool.get("sha256"):
            faults.append(f"risk_tool_binding_invalid:{risk_id}")
        if row.get("support_tool"):
            support = p2a.mapping(row.get("support_tool")); support_path = p2a.resolve(str(support.get("path") or ""))
            if not support_path.is_file() or p2a.sha256_file(support_path) != support.get("sha256"):
                faults.append(f"risk_support_tool_binding_invalid:{risk_id}")
        limits = p2a.mapping(row.get("resource_projection"))
        for key in ("projected_download_bytes", "projected_installed_bytes", "projected_peak_temporary_bytes", "projected_wall_time_seconds", "maximum_process_group_rss_mib"):
            if not isinstance(limits.get(key), int) or int(limits.get(key) or 0) < 0:
                faults.append(f"risk_resource_projection_invalid:{risk_id}:{key}")
        projected_peak = int(limits.get("projected_peak_temporary_bytes") or 0)
        reserve = int(p2a.mapping(cfg.get("resource_contract")).get("host_reserve_bytes") or 0)
        if host_free - projected_peak < reserve:
            faults.append(f"risk_host_reserve_projection_hit:{risk_id}")
        observed.append({
            "risk_id": risk_id,
            "risk_class": row.get("risk_class"),
            "task_index": row.get("task_index"),
            "manager": row.get("manager"),
            "parent_archive": base.identity(archive_path) if archive_path.is_file() else {},
            "tool": base.identity(tool_path) if tool_path.is_file() else {},
            "command": p2a.strings(row.get("command")),
            "resource_projection": limits,
            "execution_authorized": True,
        })
    if plan.get("execution_order") != "serialized_exact_list_order":
        faults.append("risk_execution_order_invalid")
    return {
        "state": plan.get("state"),
        "campaign_id": plan.get("campaign_id"),
        "row_count": len(observed),
        "rows": observed,
        "host_free_bytes": host_free,
        "host_reserve_bytes": int(p2a.mapping(cfg.get("resource_contract")).get("host_reserve_bytes") or 0),
        "execution_order": plan.get("execution_order"),
        "execution_authorized": True,
    }


def execute_risks(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    before = build(path)
    if before["trigger_state"] != "GREEN":
        return before
    cfg = p2a.read_json(path)
    rows = p2a.dicts(p2a.mapping(cfg["risk_canary_plan"]).get("rows"))
    reserve = int(p2a.mapping(cfg["resource_contract"])["host_reserve_bytes"])
    stores = {name: p2a.resolve(f"runtime/vcm_evaluator/dependency_store/shared/{name}") for name in ("bun", "yarn")}
    resume = p2a.mapping(cfg.get("risk_resume"))
    expected_bun = p2a.mapping(resume.get("qualified_bun_store"))
    observed_bun = tree_receipt(stores["bun"])
    expected_yarn = p2a.mapping(resume.get("qualified_yarn_store"))
    observed_yarn = tree_receipt(stores["yarn"])
    if (
        not stores["bun"].is_dir()
        or observed_bun.get("identity_sha256") != expected_bun.get("identity_sha256")
        or observed_bun.get("file_count") != expected_bun.get("file_count")
        or observed_bun.get("bytes") != expected_bun.get("bytes")
    ):
        return finish_risks(before, ["qualified_bun_resume_store_identity_invalid"], {"observed_bun_store": observed_bun}, False)
    if (
        not stores["yarn"].is_dir()
        or observed_yarn.get("identity_sha256") != expected_yarn.get("identity_sha256")
        or observed_yarn.get("file_count") != expected_yarn.get("file_count")
        or observed_yarn.get("bytes") != expected_yarn.get("bytes")
    ):
        return finish_risks(before, ["qualified_yarn_resume_store_identity_invalid"], {"observed_yarn_store": observed_yarn}, False)
    free_before = shutil.disk_usage(ROOT).free
    if free_before < reserve + max(int(p2a.mapping(row["resource_projection"])["projected_peak_temporary_bytes"]) for row in rows):
        return finish_risks(before, ["host_reserve_preflight_boundary_hit"], {"free_bytes_before": free_before}, False)
    faults: list[str] = []
    receipts: dict[str, Any] = {
        "free_bytes_before": free_before,
        "rows": [],
        "resumed_from_commit": resume.get("source_commit"),
        "qualified_bun_store_before": observed_bun,
        "qualified_yarn_store_before": observed_yarn,
        "bun_network_acquisition_repeated": False,
        "yarn_network_acquisition_repeated": False,
    }
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-generic-risks-", dir="/private/tmp") as raw:
        work = Path(raw).resolve()
        repos: dict[int, Path] = {}
        for task_index in (61, 4, 36):
            row = next(row for row in rows if row["task_index"] == task_index)
            repo = work / f"task-{task_index}" / "repository"
            repo.parent.mkdir(parents=True)
            archive = p2a.mapping(row["parent_archive"])
            extraction, errs = base.safe_extract_repository(p2a.resolve(str(archive["path"])), repo, str(row["archive_root"]))
            faults.extend(f"task-{task_index}:{err}" for err in errs)
            receipts[f"task_{task_index}_extraction"] = extraction
            repos[task_index] = repo
        if not faults:
            bun_row = rows[0]
            bun_cache = work / "bun-cache"; bun_home = work / "bun-home"; bun_tmp = work / "bun-tmp"
            rebased, rebase_faults = copy_bun_store_for_replay(
                stores["bun"],
                bun_cache,
                Path(str(resume.get("bun_original_cache_root") or "")),
            )
            receipts["bun_disposable_symlink_rebase"] = rebased
            faults.extend(rebase_faults)
            bun_home.mkdir(); bun_tmp.mkdir()
            bun_path = str(p2a.resolve(str(p2a.mapping(bun_row["tool"])["path"])))
            bun_env = minimal_env(bun_home, bun_tmp, f"{Path(bun_path).parent}:/usr/bin:/bin")
            bun_cmd = [bun_path, *p2a.strings(bun_row["command"])[:-1], str(bun_cache)]
            if not faults:
                faults.extend(run_offline_replay("bun_resume", [*bun_cmd, "--offline"], repos[61], work, bun_env, cfg, receipts))
        if not faults:
            yarn_row = rows[1]
            yarn_cache = work / "yarn-cache"; yarn_home = work / "yarn-home"; yarn_tmp = work / "yarn-tmp"
            shutil.copytree(stores["yarn"], yarn_cache, symlinks=True, ignore=shutil.ignore_patterns(*HOST_METADATA_NAMES)); yarn_home.mkdir(); yarn_tmp.mkdir()
            yarn_env = minimal_env(yarn_home, yarn_tmp, "/usr/bin:/bin")
            node = str(p2a.resolve(str(p2a.mapping(yarn_row["tool"])["path"])))
            args = p2a.strings(yarn_row["command"]); yarn_js = str(p2a.resolve(args[0])); base_args = [node, yarn_js, *args[1:-1], str(yarn_cache)]
            cwd = repos[4] / str(yarn_row.get("working_directory") or ".")
            faults.extend(run_offline_replay("yarn_resume", [*base_args, "--offline"], cwd, work, yarn_env, cfg, receipts))
        if not faults:
            ts_row = rows[2]; bun = str(p2a.resolve(str(p2a.mapping(ts_row["tool"])["path"])))
            receipt = bounded.run_sandboxed([bun, *p2a.strings(ts_row["command"])], repos[61], work, bun_env, cfg, network_denied=True)
            stdout_path = work / "offline.stdout"
            receipt["stdout_head"] = stdout_path.read_text(encoding="utf-8", errors="replace")[:2000] if stdout_path.is_file() else ""
            receipts["typescript_transpilation"] = receipt
            if base.command_failed(receipt): faults.append("typescript_transpilation_failed")
        if not [fault for fault in faults if fault != "typescript_transpilation_failed"]:
            rust_row = rows[3]; cargo = str(p2a.resolve(str(p2a.mapping(rust_row["tool"])["path"])))
            cargo_home = work / "cargo-home"; shutil.copytree(p2a.resolve("runtime/vcm_evaluator/dependency_store/cargo/task-36"), cargo_home)
            rust_home = work / "rust-home"; rust_tmp = work / "rust-tmp"; rust_home.mkdir(); rust_tmp.mkdir()
            rust_env = minimal_env(rust_home, rust_tmp, f"{Path(cargo).parent}:/usr/bin:/bin"); rust_env["CARGO_HOME"] = str(cargo_home)
            receipt = bounded.run_sandboxed([cargo, *p2a.strings(rust_row["command"])], repos[36], work, rust_env, cfg, network_denied=True)
            receipts["rust_compilation"] = receipt
            if base.command_failed(receipt): faults.append("rust_compilation_failed")
        receipts["free_bytes_during"] = shutil.disk_usage(ROOT).free
        if receipts["free_bytes_during"] < reserve: faults.append("host_reserve_postflight_boundary_hit")
    receipts["stores"] = {name: tree_receipt(store) for name, store in stores.items() if store.exists()}
    receipts["qualified_bun_store_after"] = tree_receipt(stores["bun"])
    receipts["qualified_yarn_store_after"] = tree_receipt(stores["yarn"])
    if receipts["qualified_bun_store_after"] != observed_bun:
        faults.append("qualified_bun_store_mutated_during_resume")
    if receipts["qualified_yarn_store_after"] != observed_yarn:
        faults.append("qualified_yarn_store_mutated_during_resume")
    receipts["free_bytes_after"] = shutil.disk_usage(ROOT).free
    return finish_risks(before, faults, receipts, True)


def execute_typescript_repair(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    before = build(path)
    if before["trigger_state"] != "GREEN":
        return before
    cfg = p2a.read_json(path)
    rows = p2a.dicts(p2a.mapping(cfg["risk_canary_plan"]).get("rows"))
    reserve = int(p2a.mapping(cfg["resource_contract"])["host_reserve_bytes"])
    store = p2a.resolve("runtime/vcm_evaluator/dependency_store/shared/bun")
    resume = p2a.mapping(cfg.get("risk_resume"))
    expected = p2a.mapping(resume.get("qualified_bun_store"))
    observed = tree_receipt(store)
    faults: list[str] = []
    if (
        not store.is_dir()
        or observed.get("identity_sha256") != expected.get("identity_sha256")
        or observed.get("file_count") != expected.get("file_count")
        or observed.get("bytes") != expected.get("bytes")
    ):
        return finish_typescript_repair(before, ["qualified_bun_resume_store_identity_invalid"], {"observed_bun_store": observed}, False)
    free_before = shutil.disk_usage(ROOT).free
    if free_before < reserve + 4294967296:
        return finish_typescript_repair(before, ["host_reserve_preflight_boundary_hit"], {"free_bytes_before": free_before}, False)
    receipts: dict[str, Any] = {"free_bytes_before": free_before, "qualified_bun_store_before": observed}
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-typescript-repair-", dir="/private/tmp") as raw:
        work = Path(raw).resolve()
        ts_row = rows[2]
        repo = work / "task-61" / "repository"
        repo.parent.mkdir(parents=True)
        archive = p2a.mapping(ts_row["parent_archive"])
        extraction, errs = base.safe_extract_repository(p2a.resolve(str(archive["path"])), repo, str(ts_row["archive_root"]))
        faults.extend(errs)
        receipts["task_61_extraction"] = extraction
        bun_cache = work / "bun-cache"
        rebased, rebase_faults = copy_bun_store_for_replay(store, bun_cache, Path(str(resume.get("bun_original_cache_root") or "")))
        receipts["bun_disposable_symlink_rebase"] = rebased
        faults.extend(rebase_faults)
        bun_home = work / "bun-home"; bun_tmp = work / "bun-tmp"; bun_home.mkdir(); bun_tmp.mkdir()
        bun = str(p2a.resolve(str(p2a.mapping(ts_row["tool"])["path"])))
        bun_env = minimal_env(bun_home, bun_tmp, f"{Path(bun).parent}:/usr/bin:/bin")
        install = [bun, "install", "--frozen-lockfile", "--ignore-scripts", "--cache-dir", str(bun_cache), "--offline"]
        if not faults:
            faults.extend(run_offline_replay("bun_resume", install, repo, work, bun_env, cfg, receipts))
        source_before = base.tree_identity(repo, excluded_roots={"node_modules"})
        if not faults:
            receipt = bounded.run_sandboxed([bun, *p2a.strings(ts_row["command"])], repo, work, bun_env, cfg, network_denied=True)
            stdout_path = work / "offline.stdout"
            receipt["stdout_head"] = stdout_path.read_text(encoding="utf-8", errors="replace")[:2000] if stdout_path.is_file() else ""
            receipts["typescript_narrow_mechanics"] = receipt
            if base.command_failed(receipt):
                faults.append("typescript_narrow_mechanics_failed")
        receipts["source_before"] = source_before
        receipts["source_after"] = base.tree_identity(repo, excluded_roots={"node_modules"})
        if receipts["source_before"] != receipts["source_after"]:
            faults.append("typescript_narrow_mechanics_source_mutated")
        receipts["free_bytes_during"] = shutil.disk_usage(ROOT).free
        if receipts["free_bytes_during"] < reserve:
            faults.append("host_reserve_postflight_boundary_hit")
    receipts["qualified_bun_store_after"] = tree_receipt(store)
    if receipts["qualified_bun_store_after"] != observed:
        faults.append("qualified_bun_store_mutated_during_resume")
    receipts["free_bytes_after"] = shutil.disk_usage(ROOT).free
    return finish_typescript_repair(before, faults, receipts, True)


def finish_typescript_repair(before: dict[str, Any], faults: list[str], receipts: dict[str, Any], executed: bool) -> dict[str, Any]:
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if executed and not faults else "RED",
        "state": "K2_03_NARROW_REAL_PARENT_TYPESCRIPT_MECHANICS_QUALIFIED" if executed and not faults else "K2_03_TYPESCRIPT_MECHANICS_REPAIR_FAILED",
        "faults": sorted(set(faults)),
        "typescript_repair_execution_performed": executed,
        "typescript_repair_receipts": receipts,
        "network_or_dependency_execution_performed": executed,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
    }


def run_install_pair(name: str, online: list[str], offline: list[str], cwd: Path, work: Path, env: dict[str, str], cfg: dict[str, Any], receipts: dict[str, Any]) -> list[str]:
    faults=[]; before=base.tree_identity(cwd, excluded_roots={"node_modules"})
    on=bounded.run_sandboxed(online,cwd,work,env,cfg,network_denied=False);receipts[f"{name}_online"]=on
    if base.command_failed(on): return [f"{name}_online_install_failed"]
    if (cwd/"node_modules").exists(): shutil.rmtree(cwd/"node_modules")
    off=bounded.run_sandboxed(offline,cwd,work,env,cfg,network_denied=True);receipts[f"{name}_offline"]=off
    if base.command_failed(off): faults.append(f"{name}_offline_replay_failed")
    after=base.tree_identity(cwd, excluded_roots={"node_modules"});receipts[f"{name}_source_before"]=before;receipts[f"{name}_source_after"]=after
    if before!=after:faults.append(f"{name}_source_mutated_outside_node_modules")
    return faults


def run_offline_replay(name: str, command: list[str], cwd: Path, work: Path, env: dict[str, str], cfg: dict[str, Any], receipts: dict[str, Any]) -> list[str]:
    before = base.tree_identity(cwd, excluded_roots={"node_modules"})
    receipt = bounded.run_sandboxed(command, cwd, work, env, cfg, network_denied=True)
    receipts[f"{name}_offline"] = receipt
    faults = [f"{name}_offline_replay_failed"] if base.command_failed(receipt) else []
    after = base.tree_identity(cwd, excluded_roots={"node_modules"})
    receipts[f"{name}_source_before"] = before
    receipts[f"{name}_source_after"] = after
    if before != after:
        faults.append(f"{name}_source_mutated_outside_node_modules")
    return faults


def copy_bun_store_for_replay(source: Path, destination: Path, original_root: Path) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns(*HOST_METADATA_NAMES))
    except shutil.Error as exc:
        return {"transformed_link_count": 0, "copy_error_count": len(exc.args[0]) if exc.args else 1}, ["bun_disposable_store_copy_failed"]
    for link in sorted(path for path in destination.rglob("*") if path.is_symlink()):
        original_target = Path(os.readlink(link))
        try:
            relative_target = original_target.relative_to(original_root)
        except ValueError:
            faults.append(f"bun_symlink_target_outside_original_cache:{link.relative_to(destination).as_posix()}")
            continue
        source_target = source / relative_target
        if not source_target.exists() or source_target.is_symlink():
            faults.append(f"bun_symlink_mapped_target_invalid:{link.relative_to(destination).as_posix()}")
            continue
        disposable_target = destination / relative_target
        rebased_target = os.path.relpath(disposable_target, start=link.parent)
        link.unlink()
        os.symlink(rebased_target, link)
        rows.append({
            "path": link.relative_to(destination).as_posix(),
            "original_target": original_target.as_posix(),
            "rebased_target": rebased_target,
            "mapped_target": relative_target.as_posix(),
        })
    broken_after = sorted(path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_symlink() and not path.exists())
    if broken_after:
        faults.append("bun_disposable_store_broken_symlinks_after_rebase")
    return {
        "original_cache_root": original_root.as_posix(),
        "transformed_link_count": len(rows),
        "transformations_sha256": base.digest_json(rows),
        "broken_link_count_after": len(broken_after),
        "retained_source_mutated": False,
    }, faults


def minimal_env(home: Path, tmp: Path, path: str) -> dict[str,str]:
    return {"HOME":str(home),"TMPDIR":str(tmp),"PATH":path,"CI":"1","NO_COLOR":"1"}


def tree_receipt(root: Path) -> dict[str,Any]:
    all_files=[p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()] if root.exists() else []
    metadata=[p for p in all_files if p.name in HOST_METADATA_NAMES]
    files=[p for p in all_files if p.name not in HOST_METADATA_NAMES]
    return {
        "path":p2a.rel(root),
        "file_count":len(files),
        "bytes":sum(p.stat().st_size for p in files),
        "identity_sha256":base.digest_json([{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":p2a.sha256_file(p)} for p in files]),
        "excluded_host_metadata":[{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":p2a.sha256_file(p)} for p in metadata],
    }


def finish_risks(before: dict[str,Any], faults: list[str], receipts: dict[str,Any], executed: bool) -> dict[str,Any]:
    return {**before,"created_utc":p2a.now(),"trigger_state":"GREEN" if executed and not faults else "RED","state":"K2_03_FOUR_ECOSYSTEM_RISK_CANARIES_QUALIFIED" if executed and not faults else "K2_03_RISK_CANARIES_FAILED","faults":sorted(set(faults)),"risk_execution_performed":executed,"risk_receipts":receipts,"network_or_dependency_execution_performed":executed,"candidate_or_control_calls":0,"external_reference_calls":0}


def project(document: dict[str, Any], raw_path: Any) -> Any:
    value: Any = document
    for part in str(raw_path or "").split("."):
        if not part or not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "trigger_state",
        "state",
        "replayed_closure_count",
        "replayed_managers",
        "resource_preflight",
        "batch_preflight",
        "static_evidence_replay_only",
        "network_or_dependency_execution_performed",
        "repository_runner_executions",
        "parent_target_or_evaluator_executions",
        "candidate_or_control_calls",
        "external_reference_calls",
        "faults",
    )
    return {key: result.get(key) for key in keys}


if __name__ == "__main__":
    raise SystemExit(main())
