#!/usr/bin/env python3
"""Machine-authorize the sealed P4-v2r2 campaign without a user gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import host_resource_safety as host_safety  # noqa: E402
import neural_seed_training_campaign as resource_owner  # noqa: E402
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2_campaign as campaign  # noqa: E402


POLICY = "project_theseus_p4v2r2_autonomous_launch_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_p4v2r2_autonomous_launch.json"
CONTEXT_TOKENS = 262_144
CAMPAIGN_COMMIT = "a38d7b968e3cf76f305616df2e7cfce9f4c837ea"
DISPOSITION_COMMIT = "e2fdb69ced76a4625e5ad86ddf74e37792aa989c"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    config = p2a.read_json(config_path)
    report = execute_once(config, config_path=config_path) if args.execute else preflight(
        config, config_path=config_path
    )
    out = p2a.resolve(args.out or str(config["report"]))
    p2a.write_json(out, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def preflight(
    config: dict[str, Any],
    *,
    config_path: Path,
    power_override: dict[str, Any] | None = None,
    memory_override: dict[str, Any] | None = None,
    disk_override: dict[str, Any] | None = None,
    jobs_override: list[dict[str, Any]] | None = None,
    campaign_override: dict[str, Any] | None = None,
    lease_exists_override: bool | None = None,
) -> dict[str, Any]:
    config_faults = validate_config(config)
    bindings = audit_bindings(config)
    power = power_override or power_status()
    memory = memory_override or memory_status(config)
    disk = disk_override or disk_status(config)
    jobs = jobs_override
    if jobs is None:
        jobs = resource_owner.active_accelerator_jobs(
            p2a.strings(config.get("exclusive_accelerator_process_patterns"))
        )
    campaign_audit = campaign_override or campaign.audit_campaign()
    lease_path = p2a.resolve(str(config.get("active_lease") or ""))
    lease_exists = (
        lease_path.exists()
        if lease_exists_override is None
        else bool(lease_exists_override)
    )
    gates = {
        "config_valid": not config_faults,
        "sealed_bindings_exact": bindings["passed"],
        "campaign_audit_green": campaign_audit.get("trigger_state") == "GREEN",
        "campaign_has_pending_tasks": int(campaign_audit.get("pending_tasks") or 0) > 0,
        "external_power_physically_connected": power.get("external_connected") is True,
        "battery_not_discharging": power.get("discharging") is False,
        "measured_runtime_memory_available": memory.get("passed") is True,
        "derived_disk_envelope_available": disk.get("passed") is True,
        "no_competing_accelerator_job": not jobs,
        "exclusive_lease_available": not lease_exists,
    }
    failed = [name for name, passed in gates.items() if not passed]
    faults = sorted(set(config_faults + p2a.strings(bindings.get("faults"))))
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not failed and not faults else "PAUSED",
        "launch_authorized": not failed and not faults,
        "config": source_identity(config_path),
        "gates": gates,
        "failed_gates": failed,
        "faults": faults,
        "sealed_bindings": bindings,
        "campaign_audit": campaign_audit,
        "power": power,
        "memory": memory,
        "disk": disk,
        "competing_accelerator_jobs": jobs,
        "active_lease": p2a.rel(lease_path),
        "authority": p2a.mapping(config.get("authority")),
        "maximum_inference": (
            "A GREEN state authorizes exactly the already-sealed local P4-v2r2 "
            "campaign under an exclusive lease. It grants no D1, D2, serving, "
            "training, hosted inference, book-support, or rerun authority."
        ),
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
        "preflight_sha256": p2a.stable_hash(before),
        "campaign_command": [
            str(config["python"]),
            str(p2a.resolve(str(config["campaign"]))),
        ],
    }
    try:
        write_json_exclusive(lease_path, lease)
    except FileExistsError:
        raced = preflight(config, config_path=config_path)
        raced["trigger_state"] = "PAUSED"
        raced["launch_authorized"] = False
        raced["lease_acquisition_race"] = True
        return raced

    completed: subprocess.CompletedProcess[str] | None = None
    error = ""
    try:
        completed = subprocess.run(
            lease["campaign_command"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:2000]
    final_audit = campaign.audit_campaign()
    complete = (
        completed is not None
        and completed.returncode == 0
        and final_audit.get("trigger_state") == "GREEN"
        and int(final_audit.get("complete_tasks") or 0) == 10
        and int(final_audit.get("pending_tasks") or 0) == 0
    )
    lease.update(
        {
            "state": "COMPLETED" if complete else "STOPPED_RETAIN_EVIDENCE",
            "completed_utc": p2a.now(),
            "child_returncode": completed.returncode if completed else None,
            "stdout_tail": completed.stdout[-2000:] if completed else "",
            "stderr_tail": completed.stderr[-2000:] if completed else "",
            "error": error,
            "final_campaign_audit_sha256": p2a.stable_hash(final_audit),
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
        "lease": source_identity(archive_path),
        "child_returncode": completed.returncode if completed else None,
        "child_stdout_tail": completed.stdout[-2000:] if completed else "",
        "child_stderr_tail": completed.stderr[-2000:] if completed else "",
        "error": error,
        "final_campaign_audit": final_audit,
        "physical_stop_is_negative_evidence": False,
        "project_selected_quality_token_cap": None,
    }


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
    if int(
        p2a.mapping(config.get("resource_derivation")).get(
            "disk_output_call_count"
        )
        or 0
    ) != 60:
        faults.append("disk_call_denominator_invalid")
    return sorted(set(faults))


def audit_bindings(config: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    rows = []
    for path_key, hash_key in (
        ("campaign", "campaign_sha256"),
        ("disposition", "disposition_sha256"),
        ("task_pool", "task_pool_sha256"),
        ("instrument", "instrument_sha256"),
        ("runtime_preflight", "runtime_preflight_sha256"),
    ):
        path = p2a.resolve(str(config.get(path_key) or ""))
        expected = str(config.get(hash_key) or "")
        observed = p2a.sha256_file(path)
        passed = path.is_file() and len(expected) == 64 and observed == expected
        if not passed:
            faults.append(f"binding_invalid:{path_key}")
        rows.append(
            {
                "owner": path_key,
                "path": p2a.rel(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": passed,
            }
        )
    if config.get("campaign_commit") != CAMPAIGN_COMMIT:
        faults.append("campaign_commit_invalid")
    if config.get("disposition_commit") != DISPOSITION_COMMIT:
        faults.append("disposition_commit_invalid")
    return {"passed": not faults, "faults": sorted(set(faults)), "owners": rows}


def power_status() -> dict[str, Any]:
    pmset = subprocess.run(
        ["/usr/bin/pmset", "-g", "batt"],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    ioreg = subprocess.run(
        ["/usr/sbin/ioreg", "-rn", "AppleSmartBattery", "-w0"],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    pmset_text = (pmset.stdout + pmset.stderr).strip()
    ioreg_text = (ioreg.stdout + ioreg.stderr).strip()
    external_values = re.findall(
        r'"(?:AppleRaw)?ExternalConnected"\s*=\s*(Yes|No)', ioreg_text
    )
    external = bool(external_values) and any(value == "Yes" for value in external_values)
    discharging = "discharging" in pmset_text.lower()
    return {
        "available": pmset.returncode == 0 and ioreg.returncode == 0 and bool(external_values),
        "external_connected": external,
        "discharging": discharging,
        "battery_percent": parse_battery_percent(pmset_text),
        "pmset_summary": " ".join(pmset_text.split())[:300],
        "ioreg_external_values": external_values,
    }


def parse_battery_percent(text: str) -> int | None:
    match = re.search(r"\b([0-9]{1,3})%;", text)
    return int(match.group(1)) if match else None


def memory_status(config: dict[str, Any]) -> dict[str, Any]:
    snapshot = host_safety.host_memory_snapshot()
    runtime = p2a.read_json(p2a.resolve(str(config["runtime_preflight"])))
    required = float(p2a.mapping(runtime.get("runtime")).get("peak_rss_mib") or 0.0)
    available = float(snapshot.reclaimable_available_mib)
    return {
        "passed": required > 0 and available >= required,
        "required_reclaimable_mib": required,
        "required_source": "bound_runtime_preflight_peak_rss_mib",
        "observed_reclaimable_mib": round(available, 3),
        "physical_memory_mib": round(snapshot.physical_memory_mib, 3),
    }


def disk_status(config: dict[str, Any]) -> dict[str, Any]:
    derivation = p2a.mapping(config.get("resource_derivation"))
    source_root = p2a.resolve(str(config["source_fixture_root"]))
    source_bytes = sum(path.stat().st_size for path in source_root.rglob("*") if path.is_file())
    output_bytes = (
        int(derivation["disk_output_call_count"])
        * CONTEXT_TOKENS
        * int(derivation["disk_output_bytes_per_context_token"])
    )
    required = source_bytes * int(derivation["disk_source_copy_count"]) + output_bytes
    free = shutil.disk_usage(ROOT).free
    return {
        "passed": free >= required,
        "free_bytes": free,
        "required_bytes": required,
        "source_fixture_bytes": source_bytes,
        "maximum_retained_output_bytes": output_bytes,
        "derivation": derivation,
    }


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def source_identity(path: Path) -> dict[str, Any]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    campaign_audit = p2a.mapping(
        report.get("final_campaign_audit") or report.get("campaign_audit")
    )
    return {
        "trigger_state": report.get("trigger_state"),
        "launch_authorized": report.get("launch_authorized"),
        "complete_tasks": campaign_audit.get("complete_tasks"),
        "pending_tasks": campaign_audit.get("pending_tasks"),
        "failed_gates": report.get("failed_gates", []),
        "faults": report.get("faults", []),
    }


if __name__ == "__main__":
    raise SystemExit(main())
