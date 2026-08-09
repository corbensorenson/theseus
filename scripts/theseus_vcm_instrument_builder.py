#!/usr/bin/env python3
"""Build one row-oriented VCM instrument ledger from manifest-bound evidence."""
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
import theseus_vcm_dependency_prefetch_canary as base  # noqa: E402

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    result = build(path)
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
    allowed = {"static_evidence_replay_authorized"}
    for key, value in p2a.mapping(cfg.get("authority")).items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")

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
        for manager in ("npm", "pnpm", "cargo", "uv")
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
        "static_evidence_replay_only": True,
        "network_or_dependency_execution_performed": False,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": cfg.get("maximum_inference"),
    }


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
