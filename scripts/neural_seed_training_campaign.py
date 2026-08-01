#!/usr/bin/env python3
"""Drive canonical neural-seed pretraining through qualified fresh processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import moecot_language_arm_training as training  # noqa: E402
import host_resource_safety  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/moecot_language_arm_training.json"
DEFAULT_OUT = ROOT / "reports/neural_seed_training_campaign.json"
DEFAULT_AVAILABILITY_CONFIG = (
    ROOT / "configs" / "neural_seed_training_availability.json"
)
POLICY = "project_theseus_fresh_process_training_campaign_v1"
AVAILABILITY_POLICY = "project_theseus_resource_aware_training_segments_v2"
LINEAGE_POLICY = "project_theseus_append_only_training_segment_lineage_v1"
LINEAGE_ANCHOR_POLICY = "project_theseus_t1_prospective_lineage_anchor_v1"


def validate_availability_policy(policy: dict[str, Any]) -> None:
    if policy.get("policy") != AVAILABILITY_POLICY:
        raise ValueError("unexpected training availability policy")
    if policy.get("enabled") is not True:
        raise ValueError("training availability scheduler must remain enabled")
    if "launch_windows" in policy:
        raise ValueError("clock-based launch windows are forbidden")
    if "minimum_disk_free_gib" in policy:
        raise ValueError("arbitrary fixed disk floors are forbidden")
    disk = policy.get("disk_reserve") or {}
    if (
        disk.get("policy") != "two_complete_checkpoint_transactions_v1"
        or int(disk.get("complete_transactions_required") or 0) < 2
        or not str(disk.get("training_config") or "")
    ):
        raise ValueError(
            "disk reserve must derive from two complete checkpoint transactions"
        )
    lineage = policy.get("lineage_custody") or {}
    if (
        lineage.get("policy") != LINEAGE_POLICY
        or not str(lineage.get("anchor") or "")
        or not str(lineage.get("ledger_directory") or "")
        or any(
            lineage.get(key) is not True
            for key in (
                "archive_before_and_after_receipts",
                "archive_child_and_host_guard_receipts",
                "require_contiguous_identity_before_launch",
                "manifest_written_last",
            )
        )
    ):
        raise ValueError("append-only segment lineage custody is required")
    behavior = policy.get("segment_behavior") or {}
    if not all(
        behavior.get(key) is True
        for key in (
            "never_suspend_in_flight_metal_graph",
            "reevaluate_after_every_transactional_segment",
            "stop_launching_when_gate_closes",
            "atomic_checkpoint_before_yield",
        )
    ):
        raise ValueError("transactional segment behavior must remain fail closed")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_identity(
    receipt: dict[str, Any],
    *,
    receipt_sha256: str | None = None,
) -> dict[str, Any]:
    required = (
        "optimizer_steps",
        "optimizer_positions",
        "pretrain_optimizer_positions",
        "checkpoint_sha256",
        "optimizer_state_sha256",
        "mlx_rng_state_sha256",
    )
    missing = [key for key in required if receipt.get(key) is None]
    if missing:
        raise ValueError(
            "training receipt identity incomplete:" + ",".join(missing)
        )
    checkpoint_path = training.resolve(str(receipt.get("checkpoint") or ""))
    receipt_path = checkpoint_path.parent / "training_receipt.json"
    if receipt_sha256 is None:
        receipt_sha256 = receipt.get("receipt_sha256")
        if receipt_path.is_file():
            receipt_sha256 = file_sha256(receipt_path)
    return {
        "optimizer_steps": int(receipt["optimizer_steps"]),
        "optimizer_positions": int(receipt["optimizer_positions"]),
        "pretrain_optimizer_positions": int(
            receipt["pretrain_optimizer_positions"]
        ),
        "checkpoint_sha256": str(receipt["checkpoint_sha256"]),
        "optimizer_state_sha256": str(receipt["optimizer_state_sha256"]),
        "mlx_rng_state_sha256": str(receipt["mlx_rng_state_sha256"]),
        "receipt_sha256": str(receipt_sha256 or ""),
    }


def load_lineage_anchor(policy: dict[str, Any]) -> dict[str, Any]:
    lineage = dict(policy["lineage_custody"])
    anchor_path = training.resolve(str(lineage["anchor"]))
    anchor = training.read_json(anchor_path)
    if (
        anchor.get("policy") != LINEAGE_ANCHOR_POLICY
        or anchor.get("campaign_id")
        != "moecot_mlx_57m_active_preregistered_v1"
        or anchor.get("target_id") != training.SHARED_TRUNK_ID
        or not isinstance(anchor.get("anchor_identity"), dict)
        or (anchor.get("pre_anchor_custody") or {}).get(
            "full_segment_predecessor_chain_available"
        )
        is not False
    ):
        raise ValueError("T1 lineage anchor is invalid")
    return anchor


def lineage_state(
    policy: dict[str, Any], current_receipt: dict[str, Any]
) -> dict[str, Any]:
    anchor = load_lineage_anchor(policy)
    expected = dict(anchor["anchor_identity"])
    ledger_root = training.resolve(
        str(policy["lineage_custody"]["ledger_directory"])
    )
    manifests = sorted(ledger_root.glob("step-*_to_*/manifest.json"))
    previous_steps = int(expected["optimizer_steps"])
    for path in manifests:
        row = training.read_json(path)
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(
                "append-only training lineage artifacts missing:"
                + training.relative(path)
            )
        archived: dict[str, Path] = {}
        for name in (
            "before_receipt",
            "after_receipt",
            "child_report",
            "host_resource_safety",
        ):
            artifact = artifacts.get(name)
            artifact_path = training.resolve(
                str((artifact or {}).get("path") or "")
            )
            if (
                not isinstance(artifact, dict)
                or not artifact_path.is_file()
                or file_sha256(artifact_path) != artifact.get("sha256")
            ):
                raise ValueError(
                    "append-only training lineage artifact invalid:"
                    + name
                    + ":"
                    + training.relative(path)
                )
            archived[name] = artifact_path
        archived_before = training.read_json(archived["before_receipt"])
        archived_after = training.read_json(archived["after_receipt"])
        before_identity = receipt_identity(
            archived_before,
            receipt_sha256=str(artifacts["before_receipt"]["sha256"]),
        )
        after_identity = receipt_identity(
            archived_after,
            receipt_sha256=str(artifacts["after_receipt"]["sha256"]),
        )
        if (
            row.get("policy") != LINEAGE_POLICY
            or row.get("before_identity") != before_identity
            or row.get("after_identity") != after_identity
            or before_identity != expected
            or int(after_identity.get("optimizer_steps") or 0)
            <= previous_steps
        ):
            raise ValueError(
                "append-only training lineage is noncontiguous:"
                + training.relative(path)
            )
        expected = after_identity
        previous_steps = int(expected["optimizer_steps"])
    current = receipt_identity(current_receipt)
    if current != expected:
        raise ValueError(
            "live training receipt does not match the append-only lineage head"
        )
    return {
        "policy": LINEAGE_POLICY,
        "anchor": training.relative(
            training.resolve(str(policy["lineage_custody"]["anchor"]))
        ),
        "ledger_directory": training.relative(ledger_root),
        "manifest_count": len(manifests),
        "head_identity": expected,
        "pre_anchor_full_chain_available": False,
        "trigger_state": "GREEN",
    }


def archive_segment_lineage(
    policy: dict[str, Any],
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    command: list[str],
    returncode: int,
    segment_out: Path,
) -> str:
    before_steps = int(before["optimizer_steps"])
    after_steps = int(after["optimizer_steps"])
    if after_steps <= before_steps:
        raise ValueError("a lineage segment must advance optimizer steps")
    ledger_root = training.resolve(
        str(policy["lineage_custody"]["ledger_directory"])
    )
    segment_dir = ledger_root / (
        f"step-{before_steps:08d}_to_{after_steps:08d}"
    )
    manifest_path = segment_dir / "manifest.json"
    if manifest_path.exists():
        raise ValueError("append-only lineage segment already exists")
    segment_dir.mkdir(parents=True, exist_ok=False)
    before_path = segment_dir / "before_receipt.json"
    after_path = segment_dir / "after_receipt.json"
    training.write_json_atomic(before_path, before)
    training.write_json_atomic(after_path, after)
    before_receipt_sha256 = file_sha256(before_path)
    after_receipt_sha256 = file_sha256(after_path)
    before_identity = receipt_identity(
        before,
        receipt_sha256=before_receipt_sha256,
    )
    after_identity = receipt_identity(
        after,
        receipt_sha256=after_receipt_sha256,
    )
    artifacts = {
        "before_receipt": {
            "path": training.relative(before_path),
            "sha256": before_receipt_sha256,
        },
        "after_receipt": {
            "path": training.relative(after_path),
            "sha256": after_receipt_sha256,
        },
    }
    host_path = segment_out.with_suffix(".host_resource_safety.json")
    for name, path in (
        ("child_report", segment_out),
        ("host_resource_safety", host_path),
    ):
        if not path.is_file():
            raise ValueError(f"segment lineage artifact missing:{name}")
        destination = segment_dir / path.name
        shutil.copy2(path, destination)
        artifacts[name] = {
            "path": training.relative(destination),
            "sha256": file_sha256(destination),
        }
    manifest = {
        "policy": LINEAGE_POLICY,
        "created_utc": training.now(),
        "campaign_id": "moecot_mlx_57m_active_preregistered_v1",
        "target_id": training.SHARED_TRUNK_ID,
        "before_identity": before_identity,
        "after_identity": after_identity,
        "command": command,
        "returncode": int(returncode),
        "artifacts": artifacts,
        "capability_claim": "NOT_EVALUATED",
        "support_state_effect": "none",
    }
    training.write_json_atomic(manifest_path, manifest)
    return training.relative(manifest_path)


def mac_power_state() -> tuple[bool | None, bool | None, dict[str, str]]:
    battery = subprocess.run(
        ["pmset", "-g", "batt"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    custom = subprocess.run(
        ["pmset", "-g", "custom"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    on_ac = (
        "Now drawing from 'AC Power'" in battery.stdout
        if battery.returncode == 0
        else None
    )
    ac_section = custom.stdout.split("AC Power:", 1)[-1]
    match = re.search(r"\blowpowermode\s+([01])\b", ac_section)
    low_power = bool(int(match.group(1))) if match else None
    return on_ac, low_power, {
        "battery": (battery.stdout + battery.stderr).strip(),
        "custom": (custom.stdout + custom.stderr).strip(),
    }


def active_accelerator_jobs(patterns: list[str]) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,stat=,command="],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [
            {
                "telemetry_fault": "process_inventory_unavailable",
                "error_type": type(exc).__name__,
            }
        ]
    if result.returncode != 0:
        return [{"telemetry_fault": "process_inventory_unavailable"}]
    own_pid = os.getpid()
    inventory: list[tuple[int, int, str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) != 4:
            continue
        pid, ppid, status, command = int(parts[0]), int(parts[1]), parts[2], parts[3]
        inventory.append((pid, ppid, status, command))
    parents = {pid: ppid for pid, ppid, _status, _command in inventory}
    own_process_tree = {own_pid}
    cursor = own_pid
    while cursor in parents and parents[cursor] > 0:
        cursor = parents[cursor]
        if cursor in own_process_tree:
            break
        own_process_tree.add(cursor)
    rows = []
    for pid, ppid, status, command in inventory:
        if pid in own_process_tree:
            continue
        # Darwin marks a process with E after it has entered irreversible exit
        # teardown. Zombies/dead processes likewise cannot dispatch new work.
        # They are retained by the OS/parent for cleanup but are not live
        # accelerator competitors; positive device usability is checked by the
        # launcher's independent Metal canary before execution.
        if "E" in status or status.startswith(("Z", "X")):
            continue
        matched = [pattern for pattern in patterns if pattern in command]
        if matched:
            rows.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "status": status,
                    "matched_patterns": matched,
                    "command": command[:500],
                }
            )
    return rows


def evaluate_availability(
    policy: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    gates = {
        "ac_power": (
            snapshot.get("on_ac_power") is True
            if policy["require_ac_power"]
            else True
        ),
        "low_power_mode_off": (
            snapshot.get("low_power_mode") is False
            if policy["require_low_power_mode_off"]
            else True
        ),
        "disk_reserve": (
            snapshot.get("checkpoint_transaction_requirement_available")
            is True
            and int(snapshot["disk_free_bytes"])
            >= int(snapshot["disk_required_bytes"])
        ),
        "no_interactive_accelerator_job": not bool(
            snapshot["active_accelerator_jobs"]
        ),
        "yield_control_absent": not bool(snapshot["yield_requested"]),
    }
    return {
        "policy": AVAILABILITY_POLICY,
        "captured_utc": training.now(),
        "trigger_state": "GREEN" if all(gates.values()) else "PAUSED",
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "snapshot": snapshot,
    }


def availability_state(policy: dict[str, Any]) -> dict[str, Any]:
    validate_availability_policy(policy)
    on_ac, low_power_mode, power_receipt = mac_power_state()
    memory = host_resource_safety.host_memory_snapshot()
    disk = shutil.disk_usage(ROOT)
    try:
        disk_requirement = checkpoint_transaction_requirement(policy)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        disk_requirement = {
            "available": False,
            "error": type(exc).__name__,
            "transaction_bytes": 0,
            "required_bytes": 0,
            "complete_transactions_required": int(
                (policy.get("disk_reserve") or {}).get(
                    "complete_transactions_required"
                )
                or 0
            ),
            "files": [],
        }
    yield_path = training.resolve(
        str(policy["yield_after_segment_control"])
    )
    snapshot = {
        "on_ac_power": on_ac,
        "low_power_mode": low_power_mode,
        "power_receipt": power_receipt,
        "disk_free_gib": round(disk.free / (1024**3), 6),
        "disk_free_bytes": int(disk.free),
        "disk_required_bytes": int(disk_requirement["required_bytes"]),
        "checkpoint_transaction_requirement_available": bool(
            disk_requirement["available"]
        ),
        "checkpoint_transaction_requirement": disk_requirement,
        "reclaimable_available_mib": round(
            memory.reclaimable_available_mib, 3
        ),
        "swapouts_mib": round(memory.swapouts_mib, 3),
        "active_accelerator_jobs": active_accelerator_jobs(
            list(policy["interactive_accelerator_process_patterns"])
        ),
        "yield_requested": yield_path.is_file(),
        "yield_control": training.relative(yield_path),
    }
    return evaluate_availability(policy, snapshot)


def checkpoint_transaction_requirement(
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Derive disk headroom from the live atomic checkpoint transaction."""

    validate_availability_policy(policy)
    reserve = dict(policy["disk_reserve"])
    training_config_path = training.resolve(str(reserve["training_config"]))
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    checkpoint_root = training.resolve(str(training_config["checkpoint_root"]))
    receipt_path = checkpoint_root / "shared_trunk" / "training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    transaction_paths = [
        training.resolve(str(receipt[key]))
        for key in ("checkpoint", "optimizer_state", "mlx_rng_state")
    ] + [receipt_path]
    if any(not path.is_file() for path in transaction_paths):
        raise ValueError("checkpoint transaction source file missing")
    files = [
        {
            "path": training.relative(path),
            "bytes": path.stat().st_size,
        }
        for path in transaction_paths
    ]
    transaction_bytes = sum(int(item["bytes"]) for item in files)
    copies = int(reserve["complete_transactions_required"])
    return {
        "available": True,
        "policy": reserve["policy"],
        "training_config": training.relative(training_config_path),
        "complete_transactions_required": copies,
        "transaction_bytes": transaction_bytes,
        "required_bytes": transaction_bytes * copies,
        "files": files,
    }


def campaign_state(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = training.bind_scale_preregistration(
        training.read_json(config_path)
    )
    plan = training.build_plan(config, config_path=config_path)
    if plan.get("trigger_state") == "RED":
        raise RuntimeError(
            "canonical plan is not GREEN: "
            + ",".join(plan.get("hard_gaps") or [])
        )
    target = plan["targets"][training.SHARED_TRUNK_ID]
    receipt_path = training.resolve(str(target["receipt"]))
    receipt = (
        training.read_json(receipt_path)
        if receipt_path.is_file()
        else {}
    )
    return config, plan, target, receipt


def estimate(
    config: dict[str, Any],
    target: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    policy = dict(
        config["architecture_training_authority"]["fresh_process_segments"]
    )
    qualification_path = training.resolve(
        str(policy["qualification_report"])
    )
    qualification = (
        training.read_json(qualification_path)
        if qualification_path.is_file()
        else {}
    )
    rows = list(qualification.get("fresh_process_segments") or [])
    device_positions = sum(
        int(
            row.get("pretrain_optimizer_positions")
            or row.get("optimizer_positions")
            or 0
        )
        for row in rows[-1:]
    ) - (
        int(
            rows[0].get("pretrain_optimizer_positions")
            or rows[0].get("optimizer_positions")
            or 0
        )
        if len(rows) > 1
        else 0
    )
    device_seconds = sum(
        float(row.get("device_step_seconds_total") or 0.0)
        for row in rows[1:] if len(rows) > 1
    )
    measured_pps = (
        device_positions / device_seconds
        if device_positions > 0 and device_seconds > 0
        else 0.0
    )
    completed = int(receipt.get("pretrain_optimizer_positions") or 0)
    target_positions = int(target["optimizer_target_positions"])
    remaining = max(0, target_positions - completed)
    seconds = remaining / measured_pps if measured_pps > 0 else None
    return {
        "completed_pretrain_optimizer_positions": completed,
        "target_pretrain_optimizer_positions": target_positions,
        "remaining_pretrain_optimizer_positions": remaining,
        "measured_device_positions_per_second": round(measured_pps, 3),
        "estimated_remaining_device_seconds": (
            round(seconds, 3) if seconds is not None else None
        ),
        "estimated_remaining_device_days": (
            round(seconds / 86400.0, 3)
            if seconds is not None
            else None
        ),
        "qualification_report": training.relative(qualification_path),
        "qualification_trigger_state": qualification.get("trigger_state"),
    }


def run_campaign(
    *,
    config_path: Path,
    out: Path,
    max_segments: int,
    availability_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if availability_policy is None:
        availability_policy = training.read_json(
            DEFAULT_AVAILABILITY_CONFIG
        )
    validate_availability_policy(availability_policy)
    config, plan, target, receipt = campaign_state(config_path)
    lineage = lineage_state(availability_policy, receipt)
    policy = dict(
        config["architecture_training_authority"]["fresh_process_segments"]
    )
    steps = int(policy["maximum_optimizer_steps"])
    authority = training.architecture_training_authority(
        config,
        max_steps=steps,
        targets=[training.SHARED_TRUNK_ID],
        phase="pretraining",
        resume=True,
        campaign_segment=True,
    )
    if authority.get("trigger_state") != "GREEN":
        raise RuntimeError(
            "fresh-process segment authority denied: "
            + str(authority.get("reason") or "")
        )
    segment_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    paused_reason = ""
    started = time.perf_counter()
    while True:
        _, _, target, before = campaign_state(config_path)
        lineage = lineage_state(availability_policy, before)
        remaining = max(
            0,
            int(target["optimizer_target_positions"])
            - int(before.get("pretrain_optimizer_positions") or 0),
        )
        if remaining == 0 or (
            max_segments > 0 and len(segment_rows) >= max_segments
        ):
            break
        availability = availability_state(availability_policy)
        availability_rows.append(availability)
        if availability["trigger_state"] != "GREEN":
            paused_reason = ",".join(availability["failed_gates"])
            break
        segment_out = out.with_name(out.stem + ".segment-latest.json")
        command = [
            str(
                training.resolve(
                    str(config["host_resource_safety"]["qualified_python"])
                )
            ),
            str(ROOT / "scripts" / "moecot_language_arm_training.py"),
            "--config",
            str(config_path),
            "--out",
            str(segment_out),
            "--guarded",
            "--execute",
            "--target",
            training.SHARED_TRUNK_ID,
            "--phase",
            "pretraining",
            "--resume",
            "--max-steps",
            str(steps),
            "--campaign-segment",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        _, _, _, after = campaign_state(config_path)
        before_steps = int(before.get("optimizer_steps") or 0)
        after_steps = int(after.get("optimizer_steps") or 0)
        lineage_manifest = ""
        if completed.returncode == 0 and after_steps > before_steps:
            lineage_manifest = archive_segment_lineage(
                availability_policy,
                before=before,
                after=after,
                command=command,
                returncode=completed.returncode,
                segment_out=segment_out,
            )
            lineage = lineage_state(availability_policy, after)
        row = {
            "segment_index": len(segment_rows) + 1,
            "command": command,
            "returncode": completed.returncode,
            "optimizer_steps_before": before_steps,
            "optimizer_steps_after": after_steps,
            "optimizer_step_delta": after_steps - before_steps,
            "pretrain_positions_before": int(
                before.get("pretrain_optimizer_positions") or 0
            ),
            "pretrain_positions_after": int(
                after.get("pretrain_optimizer_positions") or 0
            ),
            "checkpoint_sha256_after": after.get("checkpoint_sha256"),
            "optimizer_state_sha256_after": after.get(
                "optimizer_state_sha256"
            ),
            "mlx_rng_state_sha256_after": after.get(
                "mlx_rng_state_sha256"
            ),
            "lineage_manifest": lineage_manifest,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }
        segment_rows.append(row)
        if (
            completed.returncode != 0
            or row["optimizer_step_delta"] <= 0
            or row["optimizer_step_delta"] > steps
        ):
            break
        interim = {
            "policy": POLICY,
            "created_utc": training.now(),
            "trigger_state": "RUNNING",
            "plan_sha256": plan["plan_sha256"],
            "segment_policy": policy,
            "segments": segment_rows,
            "availability": availability_rows,
            "lineage": lineage,
            "progress": estimate(config, target, after),
        }
        training.write_json_atomic(out, interim)
    _, final_plan, final_target, final_receipt = campaign_state(config_path)
    failed = bool(
        segment_rows
        and (
            segment_rows[-1]["returncode"] != 0
            or segment_rows[-1]["optimizer_step_delta"] <= 0
            or segment_rows[-1]["optimizer_step_delta"] > steps
        )
    )
    trigger_state = "RED" if failed else "PAUSED" if paused_reason else "GREEN"
    return {
        "policy": POLICY,
        "created_utc": training.now(),
        "trigger_state": trigger_state,
        "plan_sha256": final_plan["plan_sha256"],
        "segment_policy": policy,
        "segment_authority": authority,
        "segments_executed": len(segment_rows),
        "segments": segment_rows,
        "availability_policy": availability_policy,
        "availability": availability_rows,
        "lineage": lineage_state(availability_policy, final_receipt),
        "paused_reason": paused_reason,
        "progress": estimate(config, final_target, final_receipt),
        "pretraining_complete": (
            int(final_receipt.get("pretrain_optimizer_positions") or 0)
            >= int(final_target["optimizer_target_positions"])
        ),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "capability_claim": "NOT_EVALUATED",
        "hard_gaps": (
            ["fresh_process_segment_failed"] if failed else []
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Zero runs until pretraining completion; positive values bound this invocation.",
    )
    parser.add_argument(
        "--availability-config",
        default=str(DEFAULT_AVAILABILITY_CONFIG),
    )
    args = parser.parse_args()
    if args.max_segments < 0:
        parser.error("--max-segments cannot be negative")
    config_path = Path(args.config).resolve()
    out = Path(args.out).resolve()
    availability_path = Path(args.availability_config).resolve()
    availability_policy = training.read_json(availability_path)
    validate_availability_policy(availability_policy)
    if args.execute:
        report = run_campaign(
            config_path=config_path,
            out=out,
            max_segments=args.max_segments,
            availability_policy=availability_policy,
        )
    else:
        config, plan, target, receipt = campaign_state(config_path)
        report = {
            "policy": POLICY,
            "created_utc": training.now(),
            "trigger_state": "STATUS_ONLY",
            "plan_sha256": plan["plan_sha256"],
            "progress": estimate(config, target, receipt),
            "availability": availability_state(availability_policy),
            "lineage": lineage_state(availability_policy, receipt),
            "capability_claim": "NOT_EVALUATED",
        }
    training.write_json_atomic(out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report.get("trigger_state") == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
