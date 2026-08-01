#!/usr/bin/env python3
"""Machine-authorize exactly one transactional neural-seed training segment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import moecot_language_arm_training as training  # noqa: E402
import neural_seed_campaign_controller as review_controller  # noqa: E402
import neural_seed_training_campaign as campaign  # noqa: E402
import pre_long_run_replacement_freeze as replacement_freeze  # noqa: E402


POLICY = "project_theseus_neural_seed_autonomous_one_shot_launch_v1"
DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_autonomous_launch_controller.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    out = resolve(args.out or str(config["report"]))
    report = execute_one_shot(config, config_path=config_path) if args.execute else preflight(
        config, config_path=config_path
    )
    training.write_json_atomic(out, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def preflight(
    config: dict[str, Any], *, config_path: Path,
    source_state_override: dict[str, Any] | None = None,
    process_jobs_override: list[dict[str, Any]] | None = None,
    package_override: dict[str, Any] | None = None,
    independent_override: dict[str, Any] | None = None,
    scale_override: dict[str, Any] | None = None,
    availability_override: dict[str, Any] | None = None,
    review_override: dict[str, Any] | None = None,
    source_binding_override: bool | None = None,
) -> dict[str, Any]:
    validate_config(config)
    package_path = resolve(str(config["replacement_freeze"]))
    independent_path = resolve(str(config["independent_readiness_audit"]))
    scale_path = resolve(str(config["scale_preregistration"]))
    training_path = resolve(str(config["training_config"]))
    availability_path = resolve(str(config["availability_config"]))
    review_config_path = resolve(str(config["review_config"]))
    yield_path = resolve(str(config["yield_control"]))
    lease_path = resolve(str(config["active_lease"]))

    package = package_override or read_json(package_path)
    independent = independent_override or read_json(independent_path)
    scale = scale_override or read_json(scale_path)
    source = source_state_override or source_state()
    process_jobs = process_jobs_override
    if process_jobs is None:
        process_jobs = campaign.active_accelerator_jobs(
            list(config["exclusive_accelerator_process_patterns"])
        )
    availability_policy = read_json(availability_path)
    availability = availability_override or campaign.availability_state(
        availability_policy
    )
    prospective_snapshot = dict(availability.get("snapshot") or {})
    prospective_snapshot["active_accelerator_jobs"] = process_jobs
    prospective_snapshot["yield_requested"] = yield_path.is_file()
    prospective_availability = campaign.evaluate_availability(
        availability_policy, prospective_snapshot
    )
    if review_override is None:
        review_config = read_json(review_config_path)
        review = review_controller.build_campaign_status(
            scale_config_path=resolve("configs/neural_seed_50m_scale_preregistration.json"),
            training_config_path=training_path,
            review_dir=resolve(str(review_config["review_directory"])),
        )
    else:
        review = review_override

    package_identity_valid = bool(
        package.get("package_identity")
        and replacement_freeze.verify_package_identity(package)
    )
    gates = {
        "replacement_freeze_green": (
            package.get("trigger_state") == "GREEN"
            and not package.get("failed_gates")
            and package_identity_valid
        ),
        "independent_readiness_green": (
            independent.get("trigger_state") == "GREEN"
            and not independent.get("failed_audits")
        ),
        "scale_preregistration_green": (
            scale.get("trigger_state") == "GREEN"
            and scale.get("contract_state") == "GREEN"
            and scale.get("proposal_state") == "AUTHORIZED_FOR_FROZEN_TRAINING_PLAN"
            and scale.get("training_authorized") is True
        ),
        "source_clean_and_exactly_bound": (
            source_is_bound_to_package(config, package=package, source=source)
            if source_binding_override is None
            else source_binding_override
        ),
        "review_controller_ready": review.get("trigger_state") in {"READY", "GREEN"},
        "emergency_yield_absent": not yield_path.is_file(),
        "no_active_lease": not lease_path.exists(),
        "no_competing_accelerator_job": not process_jobs,
        "prospective_resource_gate_green": (
            prospective_availability.get("trigger_state") == "GREEN"
        ),
        "D2_unconsumed": (
            int((scale.get("boundaries") or {}).get("D2_cases_consumed") or 0) == 0
            and (package.get("functional_surface") or {}).get("consumed_case_count", 0) == 0
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "policy": POLICY,
        "created_utc": training.now(),
        "trigger_state": "GREEN" if not failed else "PAUSED",
        "launch_authorized": not failed,
        "config": artifact(config_path),
        "gates": gates,
        "failed_gates": failed,
        "source_binding": source,
        "replacement_freeze": artifact(package_path),
        "replacement_freeze_package_identity_valid": package_identity_valid,
        "independent_readiness_audit": artifact(independent_path),
        "scale_preregistration": artifact(scale_path),
        "review_controller_state": review.get("trigger_state"),
        "current_availability": availability,
        "prospective_availability_under_exclusive_lease": prospective_availability,
        "competing_accelerator_jobs": process_jobs,
        "yield_control": relative(yield_path),
        "active_lease": relative(lease_path),
        "authority": config["authority"],
        "maximum_inference": (
            "A GREEN preflight authorizes only one bounded transactional training "
            "segment. It does not authorize D2 evaluation, public calibration, "
            "serving, external inference, architecture promotion, or book support."
        ),
    }


def execute_one_shot(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path=config_path)
    if before["trigger_state"] != "GREEN":
        return before
    lease_id = uuid.uuid4().hex
    yield_path = resolve(str(config["yield_control"]))
    lease_path = resolve(str(config["active_lease"]))
    archive_dir = resolve(str(config["lease_archive_directory"]))
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{lease_id}.json"
    rollback_root = resolve(str(config["rollback_staging_root"]))
    rollback_dir = rollback_root / lease_id
    child_report = resolve(str(config["child_report"]))
    training_config = resolve(str(config["training_config"]))
    availability_config = resolve(str(config["availability_config"]))
    raw_training = read_json(training_config)
    python_path = resolve(str(raw_training["host_resource_safety"]["qualified_python"]))
    command = [
        str(python_path),
        str(resolve(str(config["one_shot_command"]["script"]))),
        "--config", str(training_config),
        "--availability-config", str(availability_config),
        "--out", str(child_report),
        "--execute",
        "--max-segments", str(config["one_shot_command"]["max_segments"]),
    ]
    lease = {
        "policy": POLICY,
        "lease_id": lease_id,
        "created_utc": training.now(),
        "state": "PREPARED",
        "source_binding": before["source_binding"],
        "preflight_sha256": stable_hash(before),
        "command": command,
        "yield_control_unchanged": False,
        "checkpoint_rollback_state": "NOT_REQUIRED",
    }
    try:
        write_json_exclusive(lease_path, lease)
    except FileExistsError:
        raced = preflight(config, config_path=config_path)
        raced["trigger_state"] = "PAUSED"
        raced["launch_authorized"] = False
        if "no_active_lease" not in raced["failed_gates"]:
            raced["failed_gates"].append("no_active_lease")
        raced["lease_acquisition_race"] = True
        return raced

    rollback: dict[str, Any] | None = None
    completed: subprocess.CompletedProcess[str] | None = None
    error = ""
    yield_control_unchanged = False
    yield_present_before = yield_path.is_file()
    manifests_before: set[Path] = set()
    try:
        manifests_before = lineage_manifest_paths(config)
        rollback = snapshot_checkpoint_transaction(config, rollback_dir)
        lease["state"] = "RUNNING"
        training.write_json_atomic(lease_path, lease)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        yield_control_unchanged = yield_path.is_file() == yield_present_before
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:2000]
        yield_control_unchanged = yield_path.is_file() == yield_present_before

    try:
        child = read_json(child_report) if child_report.is_file() else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        child = {}
        error = append_error(error, f"child_receipt_invalid:{type(exc).__name__}:{exc}")
    segment_rows = list(child.get("segments") or [])
    try:
        manifests_after = lineage_manifest_paths(config)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        manifests_after = set(manifests_before)
        error = append_error(error, f"lineage_inventory_failed:{type(exc).__name__}:{exc}")
    new_manifests = manifests_after - manifests_before
    lineage_verified = verify_committed_lineage(
        config,
        training_config=training_config,
        segment_rows=segment_rows,
        new_manifests=new_manifests,
    )
    valid_segment = bool(
        completed is not None
        and completed.returncode == 0
        and child.get("trigger_state") == "GREEN"
        and int(child.get("segments_executed") or 0) == 1
        and len(segment_rows) == 1
        and 0 < int(segment_rows[0].get("optimizer_step_delta") or 0)
        <= int(config["one_shot_command"]["expected_maximum_optimizer_steps"])
        and lineage_verified
        and yield_control_unchanged
    )
    rollback_state = "NOT_REQUIRED"
    rollback_faults: list[str] = []
    if not valid_segment:
        manifest_advanced = bool(new_manifests)
        if manifest_advanced:
            rollback_state = "DENIED_APPEND_ONLY_LINEAGE_ALREADY_COMMITTED"
            rollback_faults.append("post_commit_validation_failed_manual_custody_review")
        elif rollback is not None:
            try:
                rollback_state, rollback_faults = restore_checkpoint_transaction(rollback)
            except Exception as exc:  # noqa: BLE001
                rollback_state = "ROLLBACK_FAILED"
                rollback_faults.append(
                    f"rollback_exception:{type(exc).__name__}:{exc}"[:2000]
                )
        else:
            rollback_state = "SNAPSHOT_NOT_CREATED_NO_TRAINING_STARTED"
    try:
        cleanup_rollback_staging(rollback_dir, allowed_root=rollback_root)
    except Exception as exc:  # noqa: BLE001
        rollback_faults.append(
            f"rollback_staging_cleanup_failed:{type(exc).__name__}:{exc}"[:2000]
        )
        if rollback_state not in {
            "DENIED_APPEND_ONLY_LINEAGE_ALREADY_COMMITTED",
            "ROLLBACK_FAILED",
        }:
            rollback_state = "ROLLBACK_FAILED"
    if valid_segment:
        lease_state = "COMMITTED"
    elif rollback_state == "DENIED_APPEND_ONLY_LINEAGE_ALREADY_COMMITTED":
        lease_state = "FAILED_REQUIRES_CUSTODY_REVIEW"
    elif rollback_state == "ROLLBACK_FAILED":
        lease_state = "FAILED_ROLLBACK_FAULT"
    elif rollback_state == "SNAPSHOT_NOT_CREATED_NO_TRAINING_STARTED":
        lease_state = "FAILED_BEFORE_SEGMENT"
    else:
        lease_state = "FAILED_ROLLED_BACK"
    lease.update({
        "state": lease_state,
        "completed_utc": training.now(),
        "child_returncode": completed.returncode if completed is not None else None,
        "child_report": artifact(child_report),
        "yield_control_unchanged": yield_control_unchanged,
        "checkpoint_rollback_state": rollback_state,
        "rollback_faults": rollback_faults,
        "new_lineage_manifests": sorted(relative(path) for path in new_manifests),
        "lineage_verified": lineage_verified,
        "error": error,
        "stdout_tail": completed.stdout[-1000:] if completed is not None else "",
        "stderr_tail": completed.stderr[-1000:] if completed is not None else "",
    })
    training.write_json_atomic(lease_path, lease)
    os.replace(lease_path, archive_path)
    return {
        "policy": POLICY,
        "created_utc": training.now(),
        "trigger_state": "GREEN" if valid_segment else "RED",
        "launch_authorized": before["launch_authorized"],
        "segment_committed": valid_segment,
        "lease": artifact(archive_path),
        "preflight": before,
        "child_report": artifact(child_report),
        "child_summary": {
            "trigger_state": child.get("trigger_state"),
            "segments_executed": child.get("segments_executed"),
            "pretraining_complete": child.get("pretraining_complete"),
            "progress": child.get("progress"),
        },
        "yield_control_unchanged": yield_control_unchanged,
        "checkpoint_rollback_state": rollback_state,
        "rollback_faults": rollback_faults,
        "new_lineage_manifests": sorted(relative(path) for path in new_manifests),
        "lineage_verified": lineage_verified,
        "error": error,
        "D2_evaluation_authorized": False,
        "capability_claim": "NOT_EVALUATED",
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create a lease without overwriting a concurrent controller's lease."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def lineage_manifest_paths(config: dict[str, Any]) -> set[Path]:
    policy = read_json(resolve(str(config["availability_config"])))
    ledger = resolve(str(policy["lineage_custody"]["ledger_directory"]))
    return {path.resolve() for path in ledger.glob("step-*_to_*/manifest.json")}


def verify_committed_lineage(
    config: dict[str, Any],
    *,
    training_config: Path,
    segment_rows: list[dict[str, Any]],
    new_manifests: set[Path],
) -> bool:
    if len(segment_rows) != 1 or len(new_manifests) != 1:
        return False
    declared = resolve(str(segment_rows[0].get("lineage_manifest") or "")).resolve()
    if declared not in new_manifests or not declared.is_file():
        return False
    try:
        policy = read_json(resolve(str(config["availability_config"])))
        _, _, _, current_receipt = campaign.campaign_state(training_config)
        lineage = campaign.lineage_state(policy, current_receipt)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError):
        return False
    return lineage.get("trigger_state") == "GREEN"


def snapshot_checkpoint_transaction(
    config: dict[str, Any], destination: Path,
) -> dict[str, Any]:
    policy = read_json(resolve(str(config["availability_config"])))
    requirement = campaign.checkpoint_transaction_requirement(policy)
    if requirement.get("available") is not True:
        raise RuntimeError("checkpoint transaction unavailable for rollback snapshot")
    destination.mkdir(parents=True, exist_ok=False)
    files: list[dict[str, Any]] = []
    for index, row in enumerate(requirement["files"], 1):
        source = resolve(str(row["path"]))
        target = destination / f"{index:02d}-{source.name}"
        shutil.copy2(source, target)
        files.append({
            "source": str(source),
            "backup": str(target),
            "sha256": sha256_file(source),
        })
    return {"directory": str(destination), "files": files}


def restore_checkpoint_transaction(snapshot: dict[str, Any]) -> tuple[str, list[str]]:
    faults: list[str] = []
    for row in snapshot.get("files") or []:
        source = Path(str(row["source"]))
        backup = Path(str(row["backup"]))
        expected = str(row["sha256"])
        if not backup.is_file() or sha256_file(backup) != expected:
            faults.append(f"rollback_backup_invalid:{backup.name}")
            continue
        temporary = source.with_name(source.name + ".autonomous-rollback-tmp")
        shutil.copy2(backup, temporary)
        os.replace(temporary, source)
        if sha256_file(source) != expected:
            faults.append(f"rollback_restore_mismatch:{source.name}")
    return (
        "RESTORED_EXACT_PRESEGMENT_TRANSACTION" if not faults else "ROLLBACK_FAILED",
        faults,
    )


def cleanup_rollback_staging(path: Path, *, allowed_root: Path) -> None:
    root = allowed_root
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("rollback staging path escaped its fixed root") from exc
    if path.is_dir():
        shutil.rmtree(path)


def source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    ready = all(row.returncode == 0 for row in (commit, branch, status))
    return {
        "commit": commit.stdout.strip() if ready else "",
        "branch": branch.stdout.strip() if ready else "",
        "clean_at_generation": ready and not dirty,
        "dirty_path_count": len(dirty),
        "dirty_paths": dirty[:200],
    }


def source_is_bound_to_package(
    config: dict[str, Any], *, package: dict[str, Any], source: dict[str, Any]
) -> bool:
    """Accept a clean evidence-only descendant of the clean freeze source commit."""

    package_source = dict(package.get("source_binding") or {})
    current_commit = str(source.get("commit") or "")
    bound_commit = str(package_source.get("commit") or "")
    if (
        source.get("clean_at_generation") is not True
        or package_source.get("clean_at_generation") is not True
        or not current_commit
        or not bound_commit
    ):
        return False
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", bound_commit, current_commit],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if ancestry.returncode != 0:
        return False
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{bound_commit}..{current_commit}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if changed.returncode != 0:
        return False
    changed_paths = {line.strip() for line in changed.stdout.splitlines() if line.strip()}
    allowed_paths = {str(path) for path in config["post_binding_evidence_paths"]}
    if not changed_paths.issubset(allowed_paths):
        return False
    package_artifacts = dict(package.get("source_artifacts") or {})
    if not package_artifacts:
        return False
    for row in package_artifacts.values():
        path = resolve(str(row.get("path") or ""))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            return False
    return True


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != POLICY:
        raise ValueError("autonomous launch policy mismatch")
    authority = dict(config.get("authority") or {})
    required_true = (
        "reevaluate_before_every_segment",
        "never_remove_or_modify_yield_control",
        "require_clean_source_bound_to_replacement_freeze",
        "require_no_competing_accelerator_job",
        "require_atomic_checkpoint_rollback_snapshot",
        "require_append_only_lineage_commit",
    )
    if authority.get("kind") != "machine_predicate_lease":
        raise ValueError("launch authority is not a machine-predicate lease")
    if authority.get("user_or_operator_approval_required") is not False:
        raise ValueError("user or operator gate is forbidden")
    if authority.get("removes_hold_permanently") is not False:
        raise ValueError("controller may not permanently remove the hold")
    if any(authority.get(key) is not True for key in required_true):
        raise ValueError("autonomous launch safety predicate missing")
    for key in (
        "D2_evaluation_authorized",
        "public_calibration_authorized",
        "external_inference_authorized",
    ):
        if authority.get(key) is not False:
            raise ValueError(f"forbidden authority enabled:{key}")
    command = dict(config.get("one_shot_command") or {})
    if command.get("script") != "scripts/neural_seed_training_campaign.py":
        raise ValueError("one-shot command is not fixed")
    if int(command.get("max_segments") or 0) != 1:
        raise ValueError("one-shot command must run exactly one segment")
    if int(command.get("expected_maximum_optimizer_steps") or 0) != 64:
        raise ValueError("one-shot segment step contract mismatch")
    allowed_evidence = config.get("post_binding_evidence_paths")
    if allowed_evidence != ["reports/pre_long_run_replacement_freeze.json"]:
        raise ValueError("post-binding evidence path contract mismatch")
    if resolve(str(config.get("rollback_staging_root") or "")) != resolve(
        "runtime/rollback/neural_seed_autonomous_launch"
    ):
        raise ValueError("rollback staging root is not the fixed project root")


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "launch_authorized": report.get("launch_authorized"),
        "segment_committed": report.get("segment_committed"),
        "failed_gates": report.get("failed_gates") or (
            (report.get("preflight") or {}).get("failed_gates")
        ),
        "yield_control_unchanged": report.get("yield_control_unchanged"),
        "checkpoint_rollback_state": report.get("checkpoint_rollback_state"),
    }


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def append_error(current: str, addition: str) -> str:
    return (current + (";" if current else "") + addition)[:2000]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "sha256": sha256_file(path) if path.is_file() else "",
        "exists": path.is_file(),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
