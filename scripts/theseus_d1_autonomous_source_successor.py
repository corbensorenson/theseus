#!/usr/bin/env python3
"""Autonomously open and freeze the D1 source cohort after an exact P4 survivor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_d1_fresh_qualification_instrument as instrument  # noqa: E402
import theseus_d1_online_metadata_acquisition as acquisition  # noqa: E402
import theseus_d1_source_selection as selection  # noqa: E402


POLICY = "project_theseus_d1_autonomous_source_successor_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_d1_autonomous_source_successor.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-until-terminal", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    out = resolve(args.out or str(config["report"]))
    if args.wait_until_terminal:
        if not args.execute:
            parser.error("--wait-until-terminal requires --execute")
        if args.poll_seconds <= 0 or args.poll_seconds > 60:
            parser.error("--poll-seconds must be in (0, 60]")
        report = wait_until_terminal(
            config,
            config_path=config_path,
            out=out,
            poll_seconds=args.poll_seconds,
        )
    else:
        report = (
            execute_once(config, config_path=config_path)
            if args.execute
            else preflight(config, config_path=config_path)
        )
        instrument.write_json(out, report)
        print(json.dumps(summary(report), indent=2, sort_keys=True), flush=True)
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def wait_until_terminal(
    config: dict[str, Any],
    *,
    config_path: Path,
    out: Path,
    poll_seconds: float,
) -> dict[str, Any]:
    while True:
        report = preflight(config, config_path=config_path)
        instrument.write_json(out, report)
        print(json.dumps(summary(report), sort_keys=True), flush=True)
        if report.get("terminal") is True or report.get("trigger_state") == "RED":
            return report
        if report.get("execution_authorized") is True:
            report = execute_once(config, config_path=config_path)
            instrument.write_json(out, report)
            print(json.dumps(summary(report), sort_keys=True), flush=True)
            if report.get("terminal") is True or report.get("trigger_state") == "RED":
                return report
        time.sleep(poll_seconds)


def preflight(
    config: dict[str, Any],
    *,
    config_path: Path,
    disposition_override: dict[str, Any] | None = None,
    ledger_override: dict[str, Any] | None = None,
    registry_override: dict[str, Any] | None = None,
    now_override: datetime | None = None,
    lease_exists_override: bool | None = None,
) -> dict[str, Any]:
    faults = validate_config(config)
    binding_audit = audit_bindings(config)
    faults.extend(strings(binding_audit.get("faults")))
    disposition_path = resolve(str(config.get("p4_terminal_disposition") or ""))
    disposition = (
        disposition_override
        if disposition_override is not None
        else read_optional(disposition_path)
    )
    status = str(disposition.get("scientific_status") or "")
    p4_green = disposition.get("trigger_state") == "GREEN"
    survivor = p4_green and status == config.get("p4_survivor_status")
    non_survivor = p4_green and status in set(
        strings(config.get("p4_terminal_non_survivor_statuses"))
    )
    lease_path = resolve(str(config.get("active_lease") or ""))
    lease_exists = (
        lease_path.exists()
        if lease_exists_override is None
        else bool(lease_exists_override)
    )
    base = {
        "policy": POLICY,
        "created_utc": now(now_override),
        "trigger_state": "RED" if faults else "PAUSED",
        "activation_state": "CONTRACT_INVALID" if faults else "WAITING_FOR_TERMINAL_P4V2R2R2",
        "terminal": False,
        "execution_authorized": False,
        "next_action": "none",
        "faults": sorted(set(faults)),
        "config": source_identity(config_path),
        "binding_audit": binding_audit,
        "p4_terminal_disposition": input_identity(
            disposition_path, disposition, disposition_override
        ),
        "network_calls": 0,
        "archive_fetches": 0,
        "parent_target_oracle_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "active_lease": relative(lease_path),
        "lease_available": not lease_exists,
        "authority": mapping(config.get("authority")),
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }
    if faults:
        return base
    if non_survivor:
        base.update(
            {
                "trigger_state": "GREEN",
                "activation_state": "CLOSED_P4V2R2R2_NON_SURVIVOR",
                "terminal": True,
            }
        )
        return base
    if not survivor:
        if disposition and p4_green:
            base["trigger_state"] = "RED"
            base["activation_state"] = "UNRECOGNIZED_TERMINAL_P4V2R2R2_STATUS"
            base["faults"] = ["unrecognized_terminal_P4V2R2R2_status"]
        return base

    registry_path = resolve(str(config.get("source_registry") or ""))
    registry = (
        registry_override
        if registry_override is not None
        else read_optional(registry_path)
    )
    if registry:
        registry_faults = audit_registry(registry)
        base["source_registry"] = input_identity(
            registry_path, registry, registry_override
        )
        base["source_registry_audit"] = {
            "passed": not registry_faults,
            "faults": registry_faults,
            "task_count": len(dictionaries(registry.get("tasks"))),
        }
        if registry_faults:
            base["trigger_state"] = "RED"
            base["activation_state"] = "FROZEN_D1_SOURCE_REGISTRY_INVALID"
            base["faults"] = registry_faults
        else:
            base["trigger_state"] = "GREEN"
            base["activation_state"] = "D1_SOURCE_REGISTRY_FROZEN"
            base["terminal"] = True
        return base

    selection_config_path = resolve(str(config["source_selection_config"]))
    selection_config = read_json(selection_config_path)
    ledger_path = resolve(str(config.get("metadata_ledger") or ""))
    ledger = ledger_override if ledger_override is not None else read_optional(ledger_path)
    selection_report = selection.build_report(
        selection_config_path,
        disposition_override=disposition_override,
        ledger_override=ledger if ledger else None,
    )
    base["selection_preflight"] = selection_report
    if selection_report.get("registry_ready") is True:
        base.update(
            {
                "trigger_state": "PAUSED" if lease_exists else "GREEN",
                "activation_state": (
                    "WAITING_FOR_EXCLUSIVE_SOURCE_LEASE"
                    if lease_exists
                    else "D1_SOURCE_REGISTRY_FREEZE_READY"
                ),
                "execution_authorized": not lease_exists,
                "next_action": "freeze_registry",
            }
        )
        return base

    acquisition_preflight = acquisition.preflight(
        selection_config_path,
        disposition_override=disposition_override,
        now_override=now_override,
    )
    base["metadata_acquisition_preflight"] = acquisition_preflight
    if acquisition_preflight.get("network_acquisition_authorized") is not True:
        base["activation_state"] = "WAITING_FOR_COMPLETE_POST_SNAPSHOT_UTC_INTERVAL"
        return base
    newest_interval_end = max(
        (
            str(row.get("end_utc") or "")
            for row in dictionaries(acquisition_preflight.get("complete_intervals"))
        ),
        default="",
    )
    acquired_interval_end = str(ledger.get("frame_end_utc") or "") if ledger else ""
    base["metadata_frame_custody"] = {
        "ledger_present": bool(ledger),
        "acquired_interval_end_utc": acquired_interval_end,
        "newest_complete_interval_end_utc": newest_interval_end,
        "new_complete_interval_available": (
            bool(newest_interval_end) and newest_interval_end > acquired_interval_end
        ),
    }
    if ledger and newest_interval_end <= acquired_interval_end:
        base["activation_state"] = "WAITING_FOR_NEW_COMPLETE_UTC_INTERVAL"
        return base
    base.update(
        {
            "trigger_state": "PAUSED" if lease_exists else "GREEN",
            "activation_state": (
                "WAITING_FOR_EXCLUSIVE_SOURCE_LEASE"
                if lease_exists
                else "D1_METADATA_ACQUISITION_READY"
            ),
            "execution_authorized": not lease_exists,
            "next_action": "acquire_metadata_then_freeze_if_complete",
        }
    )
    return base


def execute_once(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path=config_path)
    if before.get("execution_authorized") is not True:
        return before
    lease_path = resolve(str(config["active_lease"]))
    lease_id = uuid.uuid4().hex
    lease = {
        "policy": POLICY,
        "lease_id": lease_id,
        "state": "RUNNING",
        "created_utc": now(),
        "action": before.get("next_action"),
        "preflight_sha256": stable_hash(before),
    }
    try:
        write_json_exclusive(lease_path, lease)
    except FileExistsError:
        raced = preflight(config, config_path=config_path)
        raced["trigger_state"] = "PAUSED"
        raced["execution_authorized"] = False
        raced["activation_state"] = "LEASE_ACQUISITION_RACE"
        return raced

    commands: list[list[str]] = []
    if before.get("next_action") == "acquire_metadata_then_freeze_if_complete":
        commands.append(
            [str(config["python"]), str(resolve(str(config["metadata_acquisition"]))), "--execute"]
        )
    commands.append(
        [str(config["python"]), str(resolve(str(config["source_selection"]))), "--freeze"]
    )
    receipts: list[dict[str, Any]] = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        receipts.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
        if completed.returncode != 0:
            break
    final = preflight(config, config_path=config_path)
    lease.update(
        {
            "state": "COMPLETED" if final.get("terminal") is True else "PAUSED_RETAIN_CUSTODY",
            "completed_utc": now(),
            "receipts": receipts,
            "final_activation_state": final.get("activation_state"),
        }
    )
    instrument.write_json(lease_path, lease)
    archive_dir = resolve(str(config["lease_archive_directory"]))
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{lease_id}.json"
    os.replace(lease_path, archive_path)
    final["lease"] = source_identity(archive_path)
    final["execution_receipts"] = receipts
    return final


def validate_config(config: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    if config.get("state") != "BOUND_BEFORE_P4V2R2R2_TERMINAL_EVIDENCE":
        faults.append("state_invalid")
    authority = mapping(config.get("authority"))
    if authority.get("kind") != "machine_predicate_exclusive_exactly_once_source_lease":
        faults.append("authority_kind_invalid")
    required_true = (
        "network_metadata_only_after_exact_p4_survivor",
        "freeze_design_derived_initial_source_frame_exactly_once",
    )
    required_false = (
        "user_or_operator_approval_required",
        "repeat_acquisition_without_new_complete_utc_interval",
        "archive_fetch_authorized",
        "parent_target_oracle_or_evaluator_execution_authorized",
        "candidate_or_control_calls_authorized",
        "external_inference_authorized",
        "teacher_calls_authorized",
        "training_rows_authorized",
        "serving_authorized",
        "D2_authorized",
        "book_support_promotion_authorized",
        "project_selected_quality_token_cap_allowed",
    )
    if any(authority.get(key) is not True for key in required_true):
        faults.append("required_authority_boundary_missing")
    if any(authority.get(key) is not False for key in required_false):
        faults.append("forbidden_authority_present")
    if authority.get("wait_deadline_seconds") is not None:
        faults.append("arbitrary_wait_deadline_present")
    return sorted(set(faults))


def audit_bindings(config: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    rows = []
    for path_key, digest_key in (
        ("instrument", "instrument_sha256"),
        ("source_selection_config", "source_selection_config_sha256"),
        ("metadata_acquisition", "metadata_acquisition_sha256"),
        ("source_selection", "source_selection_sha256"),
    ):
        path = resolve(str(config.get(path_key) or ""))
        expected = str(config.get(digest_key) or "")
        observed = sha256_file(path)
        passed = path.is_file() and len(expected) == 64 and observed == expected
        if not passed:
            faults.append(f"binding_invalid:{path_key}")
        rows.append(
            {
                "owner": path_key,
                "path": relative(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": passed,
            }
        )
    return {"passed": not faults, "faults": faults, "owners": rows}


def audit_registry(registry: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if registry.get("policy") != selection.REGISTRY_POLICY:
        faults.append("source_registry_policy_invalid")
    if registry.get("state") != (
        "FIXED_BEFORE_ARCHIVE_FETCH_PARENT_TARGET_ORACLE_EVALUATOR_OR_CANDIDATE_EXECUTION"
    ):
        faults.append("source_registry_state_invalid")
    tasks = dictionaries(registry.get("tasks"))
    declared_count = int(registry.get("task_count") or 0)
    if declared_count < 44 or len(tasks) != declared_count:
        faults.append("source_registry_task_count_invalid")
    repositories = [str(row.get("repository") or "").lower() for row in tasks]
    if len(set(repositories)) != declared_count:
        faults.append("source_registry_repositories_not_distinct")
    if [int(row.get("campaign_index") or 0) for row in tasks] != list(
        range(1, declared_count + 1)
    ):
        faults.append("source_registry_campaign_indexes_invalid")
    boundaries = mapping(registry.get("boundaries"))
    for key in (
        "archive_fetches",
        "parent_target_oracle_or_evaluator_executions",
        "candidate_or_control_calls",
        "external_inference_calls",
        "teacher_calls",
        "training_rows_written",
    ):
        if int(boundaries.get(key) or 0) != 0:
            faults.append(f"source_registry_boundary_nonzero:{key}")
    if registry.get("replacement_after_membership_freeze") is not False:
        faults.append("source_registry_replacement_allowed")
    return sorted(set(faults))


def input_identity(
    path: Path, value: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    if not value:
        return {"path": relative(path), "present": False, "sha256": ""}
    return {
        "path": relative(path),
        "present": True,
        "sha256": stable_hash(value) if override is not None else sha256_file(path),
    }


def source_identity(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def read_optional(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value or [] if isinstance(row, dict)]


def strings(value: Any) -> list[str]:
    return [str(row) for row in value or [] if isinstance(row, str)]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def now(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "activation_state": report.get("activation_state"),
        "terminal": report.get("terminal"),
        "execution_authorized": report.get("execution_authorized"),
        "next_action": report.get("next_action"),
        "faults": report.get("faults", []),
    }


if __name__ == "__main__":
    raise SystemExit(main())
