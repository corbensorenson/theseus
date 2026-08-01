#!/usr/bin/env python3
"""Materialize exact D1 parent/target archives after the source registry freezes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_d1_autonomous_source_successor as successor  # noqa: E402
import theseus_p2b_sanitize_archive as sanitizer  # noqa: E402


POLICY = "project_theseus_d1_source_materialization_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_d1_source_materialization.json"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
STATUSES = {"added", "modified", "removed", "renamed", "copied", "changed"}
LICENSE_NAMES = {
    "copying",
    "copying.md",
    "copying.txt",
    "license",
    "license.md",
    "license.rst",
    "license.txt",
    "unlicense",
    "unlicense.txt",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--wait-until-terminal", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    out = resolve(args.out or str(config["report"]))
    if args.wait_until_terminal:
        if not args.fetch:
            parser.error("--wait-until-terminal requires --fetch")
        report = wait_until_terminal(config, config_path=config_path, out=out)
    elif args.fetch:
        report = execute_once(config, config_path=config_path)
        write_json(out, report)
    else:
        report = preflight(config, config_path=config_path)
        write_json(out, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True), flush=True)
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def wait_until_terminal(
    config: dict[str, Any], *, config_path: Path, out: Path
) -> dict[str, Any]:
    poll_seconds = float(config.get("poll_seconds") or 60)
    while True:
        report = preflight(config, config_path=config_path)
        write_json(out, report)
        print(json.dumps(summary(report), sort_keys=True), flush=True)
        if report.get("trigger_state") == "RED" or report.get("terminal") is True:
            return report
        if report.get("execution_authorized") is True:
            report = execute_once(config, config_path=config_path)
            write_json(out, report)
            print(json.dumps(summary(report), sort_keys=True), flush=True)
            if report.get("trigger_state") == "RED" or report.get("terminal") is True:
                return report
        time.sleep(poll_seconds)


def preflight(
    config: dict[str, Any],
    *,
    config_path: Path,
    registry_override: dict[str, Any] | None = None,
    lease_exists_override: bool | None = None,
) -> dict[str, Any]:
    faults = validate_config(config)
    binding_audit = audit_bindings(config)
    faults.extend(strings(binding_audit.get("faults")))
    registry_path = resolve(str(config.get("source_registry") or ""))
    registry = (
        registry_override
        if registry_override is not None
        else read_json(registry_path) if registry_path.is_file() else {}
    )
    lease_path = resolve(str(config.get("active_lease") or ""))
    lease_exists = (
        lease_path.exists()
        if lease_exists_override is None
        else bool(lease_exists_override)
    )
    report = {
        "policy": POLICY,
        "created_utc": successor.now(),
        "trigger_state": "RED" if faults else "PAUSED",
        "activation_state": "CONTRACT_INVALID" if faults else "WAITING_FOR_FROZEN_D1_SOURCE_REGISTRY",
        "terminal": False,
        "execution_authorized": False,
        "faults": sorted(set(faults)),
        "config": source_identity(config_path),
        "binding_audit": binding_audit,
        "source_registry": input_identity(registry_path, registry, registry_override),
        "lease_available": not lease_exists,
        "network_fetches": 0,
        "archive_artifacts": 0,
        "parent_target_executions": 0,
        "oracle_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "authority": mapping(config.get("authority")),
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }
    if faults or not registry:
        return report
    registry_faults = audit_registry(registry)
    report["registry_audit"] = {
        "passed": not registry_faults,
        "faults": registry_faults,
        "task_count": len(dictionaries(registry.get("tasks"))),
    }
    if registry_faults:
        report["trigger_state"] = "RED"
        report["activation_state"] = "FROZEN_D1_SOURCE_REGISTRY_INVALID"
        report["faults"] = registry_faults
        return report
    existing = audit_existing_materialization(config, registry)
    report["existing_materialization"] = existing
    if existing.get("complete") is True:
        report["trigger_state"] = "GREEN"
        report["activation_state"] = "D1_SOURCE_ARCHIVES_MATERIALIZED"
        report["terminal"] = True
        report["archive_artifacts"] = existing.get("artifact_count")
        return report
    report["trigger_state"] = "PAUSED" if lease_exists else "GREEN"
    report["activation_state"] = (
        "WAITING_FOR_EXCLUSIVE_MATERIALIZATION_LEASE"
        if lease_exists
        else "D1_SOURCE_ARCHIVE_FETCH_READY"
    )
    report["execution_authorized"] = not lease_exists
    return report


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
        "created_utc": successor.now(),
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
    registry = read_json(resolve(str(config["source_registry"])))
    try:
        result = materialize(config, registry, downloader=download)
    except (OSError, tarfile.TarError, urllib.error.URLError, ValueError) as exc:
        result = {
            **before,
            "trigger_state": "PAUSED",
            "activation_state": "RETRYABLE_SOURCE_TRANSPORT_OR_ARCHIVE_ERROR",
            "execution_authorized": False,
            "error": f"{type(exc).__name__}:{exc}"[:4000],
        }
    lease.update(
        {
            "state": "COMPLETED" if result.get("terminal") is True else "PAUSED_RETAIN_ARTIFACTS",
            "completed_utc": successor.now(),
            "result_sha256": stable_hash(result),
        }
    )
    write_json(lease_path, lease)
    archive_dir = resolve(str(config["lease_archive_directory"]))
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{lease_id}.json"
    os.replace(lease_path, archive_path)
    result["lease"] = source_identity(archive_path)
    return result


def materialize(
    config: dict[str, Any],
    registry: dict[str, Any],
    *,
    downloader: Callable[[str, Path], None],
) -> dict[str, Any]:
    faults = audit_registry(registry)
    rows: list[dict[str, Any]] = []
    network_fetches = 0
    archive_root = resolve(str(config["archive_root"]))
    report_root = resolve(str(config["sanitization_report_root"]))
    for task in dictionaries(registry.get("tasks")):
        task_row = {
            "campaign_index": task.get("campaign_index"),
            "repository": task.get("repository"),
            "selection_digest": task.get("selection_digest"),
            "artifacts": [],
        }
        for plan in artifact_plan(task, config):
            upstream = archive_root / plan["upstream_name"]
            normalized = archive_root / plan["normalized_name"]
            sanitization_path = report_root / plan["sanitization_name"]
            if not upstream.is_file():
                downloader(plan["url"], upstream)
                network_fetches += 1
            sanitization = sanitizer.sanitize(upstream, normalized)
            write_json(sanitization_path, sanitization)
            artifact_faults = audit_materialized_artifact(
                task, plan, normalized, sanitization
            )
            faults.extend(artifact_faults)
            task_row["artifacts"].append(
                {
                    **plan,
                    "upstream": relative(upstream),
                    "upstream_sha256": sha256_file(upstream),
                    "normalized": relative(normalized),
                    "normalized_sha256": sha256_file(normalized),
                    "sanitization_report": relative(sanitization_path),
                    "sanitization_report_sha256": sha256_file(sanitization_path),
                    "source_archive_root": sanitization.get("source_archive_root"),
                    "faults": artifact_faults,
                }
            )
        rows.append(task_row)
    artifact_count = sum(len(row["artifacts"]) for row in rows)
    if len(rows) != 44:
        faults.append("materialized_task_count_invalid")
    if artifact_count != 88:
        faults.append("materialized_archive_count_invalid")
    terminal = bool(faults) or artifact_count == 88
    report = {
        "policy": POLICY,
        "created_utc": successor.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "activation_state": (
            "D1_SOURCE_ARCHIVES_MATERIALIZED"
            if not faults
            else "D1_SOURCE_MATERIALIZATION_TERMINAL_FAILURE_RETAIN_TASKS"
        ),
        "terminal": terminal,
        "execution_authorized": False,
        "faults": sorted(set(faults)),
        "source_registry": source_identity(resolve(str(config["source_registry"]))),
        "network_fetches": network_fetches,
        "task_count": len(rows),
        "archive_artifacts": artifact_count,
        "tasks": rows,
        "postfreeze_task_replacement_allowed": False,
        "parent_target_executions": 0,
        "oracle_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }
    write_json(resolve(str(config["report"])), report)
    return report


def validate_config(config: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    if config.get("state") != "BOUND_BEFORE_D1_SOURCE_REGISTRY_EXISTS":
        faults.append("state_invalid")
    if config.get("allowed_transport_origin") != "https://codeload.github.com":
        faults.append("transport_origin_invalid")
    if not (0 < float(config.get("poll_seconds") or 0) <= 60):
        faults.append("poll_interval_invalid")
    authority = mapping(config.get("authority"))
    required_true = (
        "network_source_archives_only_after_registry_freeze",
        "exact_parent_and_target_revisions_only",
    )
    required_false = (
        "user_or_operator_approval_required",
        "postfreeze_task_replacement_allowed",
        "parent_target_execution_authorized",
        "oracle_or_evaluator_execution_authorized",
        "candidate_or_control_calls_authorized",
        "external_inference_authorized",
        "teacher_calls_authorized",
        "training_rows_authorized",
        "serving_authorized",
        "D2_authorized",
        "book_support_promotion_authorized",
    )
    if authority.get("kind") != "exclusive_idempotent_frozen_registry_materialization_lease":
        faults.append("authority_kind_invalid")
    if any(authority.get(key) is not True for key in required_true):
        faults.append("required_authority_boundary_missing")
    if any(authority.get(key) is not False for key in required_false):
        faults.append("forbidden_authority_present")
    return sorted(set(faults))


def audit_bindings(config: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    rows = []
    for path_key, hash_key in (
        ("source_successor_config", "source_successor_config_sha256"),
        ("sanitizer", "sanitizer_sha256"),
    ):
        path = resolve(str(config.get(path_key) or ""))
        expected = str(config.get(hash_key) or "")
        observed = sha256_file(path)
        passed = path.is_file() and len(expected) == 64 and observed == expected
        if not passed:
            faults.append(f"binding_invalid:{path_key}")
        rows.append({"path": relative(path), "expected": expected, "observed": observed, "passed": passed})
    return {"passed": not faults, "faults": faults, "owners": rows}


def audit_registry(registry: dict[str, Any]) -> list[str]:
    faults = successor.audit_registry(registry)
    if registry.get("campaign_id") != "theseus_d1_cognitive_compilation_fresh_qualification_v1":
        faults.append("source_registry_campaign_invalid")
    if registry.get("claim_id") != "cognitive-compilation-and-semantic-ir.core":
        faults.append("source_registry_claim_invalid")
    for index, task in enumerate(dictionaries(registry.get("tasks")), 1):
        repository = str(task.get("repository") or "")
        parent = str(task.get("parent_revision") or "")
        target = str(task.get("target_revision") or "")
        files = dictionaries(task.get("changed_files"))
        if not REPOSITORY.fullmatch(repository):
            faults.append(f"repository_invalid:{index}")
        if not SHA40.fullmatch(parent) or not SHA40.fullmatch(target) or parent == target:
            faults.append(f"revision_invalid:{index}")
        if not files or sorted(str(row.get("filename") or "") for row in files) != sorted(
            strings(task.get("changed_paths"))
        ):
            faults.append(f"changed_file_inventory_invalid:{index}")
        for row in files:
            if str(row.get("status") or "") not in STATUSES:
                faults.append(f"changed_file_status_invalid:{index}")
            if not safe_relative_path(str(row.get("filename") or "")):
                faults.append(f"changed_file_path_invalid:{index}")
            previous = str(row.get("previous_filename") or "")
            if previous and not safe_relative_path(previous):
                faults.append(f"changed_previous_path_invalid:{index}")
    return sorted(set(faults))


def artifact_plan(task: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    repository = str(task["repository"])
    slug = safe_slug(repository)
    index = int(task["campaign_index"])
    origin = str(config["allowed_transport_origin"])
    plans = []
    for label in ("parent", "target"):
        revision = str(task[f"{label}_revision"])
        prefix = f"{index:02d}_{slug}_{label}"
        plans.append(
            {
                "label": label,
                "revision": revision,
                "url": f"{origin}/{repository}/tar.gz/{revision}",
                "upstream_name": f"{prefix}_upstream.tar.gz",
                "normalized_name": f"{prefix}.tar.gz",
                "sanitization_name": f"{prefix}.json",
                "required_relative_members": required_changed_members(task, label),
            }
        )
    return plans


def required_changed_members(task: dict[str, Any], label: str) -> list[str]:
    required: set[str] = set()
    for row in dictionaries(task.get("changed_files")):
        status = str(row.get("status") or "")
        filename = str(row.get("filename") or "")
        previous = str(row.get("previous_filename") or "")
        if label == "parent" and status not in {"added"}:
            required.add(previous if status in {"renamed", "copied"} and previous else filename)
        if label == "target" and status not in {"removed"}:
            required.add(filename)
    return sorted(required)


def audit_materialized_artifact(
    task: dict[str, Any],
    plan: dict[str, Any],
    archive: Path,
    sanitization: dict[str, Any],
) -> list[str]:
    faults: list[str] = []
    if sanitization.get("trigger_state") != "GREEN":
        faults.append("archive_sanitization_red")
    root = str(sanitization.get("source_archive_root") or "")
    if not root or not root.endswith(str(plan["revision"])):
        faults.append("archive_root_revision_mismatch")
    names = archive_names(archive)
    required = [f"{root}/{path}" for path in plan["required_relative_members"]]
    missing = [path for path in required if path not in names]
    if missing:
        faults.append("required_changed_members_missing:" + stable_hash(missing))
    if not any(is_root_license(name, root) for name in names):
        faults.append("root_license_file_missing")
    if any(not sanitizer.safe_member_name(name) for name in names):
        faults.append("normalized_archive_unsafe_member_path")
    return faults


def audit_existing_materialization(config: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    report_path = resolve(str(config["report"]))
    if not report_path.is_file():
        return {"complete": False, "artifact_count": 0, "faults": []}
    report = read_json(report_path)
    faults: list[str] = []
    if report.get("policy") != POLICY or report.get("trigger_state") != "GREEN":
        faults.append("existing_report_not_green")
    registry_identity = mapping(report.get("source_registry"))
    current_registry = resolve(str(config["source_registry"]))
    if registry_identity.get("sha256") != sha256_file(current_registry):
        faults.append("existing_report_registry_mismatch")
    artifact_count = int(report.get("archive_artifacts") or 0)
    for task in dictionaries(report.get("tasks")):
        for artifact in dictionaries(task.get("artifacts")):
            for path_key, hash_key in (
                ("upstream", "upstream_sha256"),
                ("normalized", "normalized_sha256"),
                ("sanitization_report", "sanitization_report_sha256"),
            ):
                path = resolve(str(artifact.get(path_key) or ""))
                if not path.is_file() or sha256_file(path) != artifact.get(hash_key):
                    faults.append(f"existing_artifact_invalid:{path_key}")
    return {"complete": not faults and artifact_count == 88, "artifact_count": artifact_count, "faults": sorted(set(faults))}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Project-Theseus-D1-Source/1"})
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 -- exact constructed codeload origin.
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)


def archive_names(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as handle:
        return {member.name.rstrip("/") for member in handle.getmembers()}


def is_root_license(name: str, root: str) -> bool:
    path = PurePosixPath(name)
    return len(path.parts) == 2 and path.parts[0] == root and path.parts[1].lower() in LICENSE_NAMES


def safe_relative_path(value: str) -> bool:
    return sanitizer.safe_member_name(value) and not value.endswith("/")


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def input_identity(path: Path, value: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    return {"path": relative(path), "present": bool(value), "sha256": stable_hash(value) if override is not None and value else sha256_file(path)}


def source_identity(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in ("trigger_state", "activation_state", "terminal", "execution_authorized", "network_fetches", "archive_artifacts", "faults")}


if __name__ == "__main__":
    raise SystemExit(main())
