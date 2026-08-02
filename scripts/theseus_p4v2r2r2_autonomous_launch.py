#!/usr/bin/env python3
"""Machine-authorize and lease the sealed all-new P4 campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import neural_seed_training_campaign as resource_owner  # noqa: E402
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2_autonomous_launch as host  # noqa: E402
import theseus_p4v2r2r2_campaign as campaign  # noqa: E402


POLICY = "project_theseus_p4v2r2r2_autonomous_launch_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_p4v2r2r2_autonomous_launch.json"
BOUND_CAMPAIGN_COMMIT = "aea5dac6a5e4ebd25391c554afb0a0c2957890c4"


def validate_config(config: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("kind") != "machine_predicate_exclusive_one_shot_lease":
        faults.append("authority_kind_invalid")
    required_true = (
        "require_external_power_physically_connected",
        "require_battery_not_discharging",
        "require_measured_runtime_memory_available",
        "require_derived_disk_envelope_available",
        "require_no_competing_accelerator_job",
        "require_metal_accelerator_usable",
    )
    required_false = (
        "user_or_operator_approval_required",
        "project_selected_quality_token_cap_allowed",
        "physical_boundary_is_negative_evidence",
        "external_inference_authorized",
        "serving_authorized",
        "training_row_admission_authorized",
        "D1_authorized",
        "D2_authorized",
        "book_support_promotion_authorized",
    )
    if any(authority.get(key) is not True for key in required_true):
        faults.append("required_machine_predicate_missing")
    if any(authority.get(key) is not False for key in required_false):
        faults.append("forbidden_authority_present")
    if int(p2a.mapping(config.get("resource_derivation")).get("disk_output_call_count") or 0) != 60:
        faults.append("disk_call_denominator_invalid")
    runtime = p2a.mapping(config.get("qualified_runtime"))
    if runtime != {"python_version": [3, 12, 5], "mlx_version": "0.32.0", "mlx_lm_version": "0.31.3"}:
        faults.append("qualified_runtime_identity_invalid")
    return sorted(set(faults))


def audit_bindings(config: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    owners: list[dict[str, Any]] = []
    for path_key, hash_key in (
        ("campaign", "campaign_sha256"),
        ("task_pool", "task_pool_sha256"),
        ("instrument", "instrument_sha256"),
        ("route_canary", "route_canary_sha256"),
        ("runtime_preflight", "runtime_preflight_sha256"),
        ("python", "python_sha256"),
        ("accelerator_inventory", "accelerator_inventory_sha256"),
    ):
        path = p2a.resolve(str(config.get(path_key) or ""))
        expected = str(config.get(hash_key) or "")
        observed = p2a.sha256_file(path)
        passed = path.is_file() and len(expected) == 64 and observed == expected
        if not passed:
            faults.append(f"binding_invalid:{path_key}")
        owners.append(
            {
                "owner": path_key,
                "path": p2a.rel(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": passed,
            }
        )
    if config.get("campaign_commit") != BOUND_CAMPAIGN_COMMIT:
        faults.append("campaign_commit_invalid")
    return {"passed": not faults, "faults": sorted(set(faults)), "owners": owners}


def preflight(
    config: dict[str, Any],
    *,
    config_path: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    override = overrides or {}
    bindings = audit_bindings(config)
    campaign_audit = override.get("campaign") or campaign.audit_campaign()
    power = override.get("power") or host.power_status()
    memory = override.get("memory") or host.memory_status(config)
    disk = override.get("disk") or host.disk_status(config)
    runtime = override.get("runtime") or host.runtime_status(config)
    metal = override.get("metal") or host.metal_status(config)
    jobs = override.get("jobs")
    if jobs is None:
        jobs = resource_owner.active_accelerator_jobs(
            p2a.strings(config.get("exclusive_accelerator_process_patterns"))
        )
    lease_path = p2a.resolve(str(config.get("active_lease") or ""))
    lease_exists = bool(override.get("lease_exists", lease_path.exists()))
    config_faults = validate_config(config)
    gates = {
        "config_valid": not config_faults,
        "sealed_bindings_exact": bindings["passed"],
        "campaign_audit_green": campaign_audit.get("trigger_state") == "GREEN",
        "campaign_has_pending_tasks": int(campaign_audit.get("pending_tasks") or 0) > 0,
        "external_power_physically_connected": power.get("external_connected") is True,
        "battery_not_discharging": power.get("discharging") is False,
        "measured_runtime_memory_available": memory.get("passed") is True,
        "derived_disk_envelope_available": disk.get("passed") is True,
        "qualified_runtime_executable_bound": runtime.get("passed") is True,
        "metal_accelerator_usable": metal.get("passed") is True,
        "no_competing_accelerator_job": not jobs,
        "exclusive_lease_available": not lease_exists,
    }
    failed = [key for key, passed in gates.items() if not passed]
    faults = sorted(set(config_faults + p2a.strings(bindings.get("faults"))))
    state = "GREEN" if not failed and not faults else "RED" if faults else "PAUSED"
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": state,
        "launch_authorized": state == "GREEN",
        "config": host.source_identity(config_path),
        "gates": gates,
        "failed_gates": failed,
        "faults": faults,
        "sealed_bindings": bindings,
        "campaign_audit": campaign_audit,
        "power": power,
        "memory": memory,
        "disk": disk,
        "runtime": runtime,
        "metal": metal,
        "competing_accelerator_jobs": jobs,
        "active_lease": p2a.rel(lease_path),
        "project_selected_quality_token_cap": None,
        "maximum_inference": "Authorizes exactly one local sealed campaign under an exclusive machine lease; no D1, D2, training, serving, hosted inference, or book promotion.",
    }


def execute_once(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path=config_path)
    if before["trigger_state"] != "GREEN":
        return before
    lease_path = p2a.resolve(str(config["active_lease"]))
    archive_dir = p2a.resolve(str(config["lease_archive_directory"]))
    archive_dir.mkdir(parents=True, exist_ok=True)
    lease_id = uuid.uuid4().hex
    lease = {
        "policy": POLICY,
        "lease_id": lease_id,
        "state": "RUNNING",
        "created_utc": p2a.now(),
    }
    try:
        host.write_json_exclusive(lease_path, lease)
    except FileExistsError:
        report = preflight(config, config_path=config_path)
        report["trigger_state"] = "PAUSED"
        report["launch_authorized"] = False
        return report
    completed: subprocess.CompletedProcess[str] | None = None
    error = ""
    try:
        completed = subprocess.run(
            [str(config["python"]), str(p2a.resolve(str(config["campaign"])))],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - lease retains terminal error.
        error = f"{type(exc).__name__}: {exc}"[:2000]
    final = campaign.audit_campaign()
    complete = (
        completed is not None
        and completed.returncode == 0
        and final.get("trigger_state") == "GREEN"
        and final.get("complete_tasks") == 10
    )
    lease.update(
        {
            "state": "COMPLETED" if complete else "STOPPED_RETAIN_EVIDENCE",
            "completed_utc": p2a.now(),
            "child_returncode": completed.returncode if completed else None,
            "error": error,
        }
    )
    p2a.write_json(lease_path, lease)
    archive_path = archive_dir / f"{lease_id}.json"
    os.replace(lease_path, archive_path)
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if complete else "RED",
        "launch_authorized": True,
        "campaign_complete": complete,
        "preflight": before,
        "lease": host.source_identity(archive_path),
        "child_returncode": completed.returncode if completed else None,
        "child_stdout_tail": completed.stdout[-2000:] if completed else "",
        "child_stderr_tail": completed.stderr[-2000:] if completed else "",
        "error": error,
        "final_campaign_audit": final,
        "project_selected_quality_token_cap": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    config = p2a.read_json(config_path)
    report = (
        execute_once(config, config_path=config_path)
        if args.execute
        else preflight(config, config_path=config_path)
    )
    p2a.write_json(p2a.resolve(str(config["report"])), report)
    campaign_audit = p2a.mapping(
        report.get("final_campaign_audit") or report.get("campaign_audit")
    )
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "launch_authorized": report["launch_authorized"],
                "complete_tasks": campaign_audit.get("complete_tasks"),
                "pending_tasks": campaign_audit.get("pending_tasks"),
                "failed_gates": report.get("failed_gates", []),
                "faults": report.get("faults", []),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
