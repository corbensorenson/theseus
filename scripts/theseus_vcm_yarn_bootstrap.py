#!/usr/bin/env python3
"""Acquire an exact Yarn Classic runtime for the sole Yarn-locked VCM task."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import ssl
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_yarn_bootstrap_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_yarn_bootstrap.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    config = p2a.read_json(config_path)
    report = acquire(config, config_path) if args.acquire else preflight(config, config_path)
    p2a.write_json(p2a.resolve(args.out or config["report"]), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    faults = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    owner = p2a.resolve(str(config.get("owner") or ""))
    if owner != Path(__file__).resolve() or p2a.sha256_file(owner) != str(config.get("owner_sha256") or ""):
        faults.append("owner_binding_invalid")
    ca = Path(str(config.get("ca_bundle") or ""))
    if not ca.is_file() or p2a.sha256_file(ca) != str(config.get("ca_bundle_sha256") or ""):
        faults.append("ca_bundle_identity_invalid")
    artifact = p2a.mapping(config.get("artifact"))
    if artifact.get("version") != "1.22.22" or not str(artifact.get("url") or "").startswith("https://registry.npmjs.org/yarn/-/"):
        faults.append("artifact_identity_invalid")
    if len(str(artifact.get("sha1") or "")) != 40 or not str(artifact.get("sha512_base64") or ""):
        faults.append("artifact_digest_invalid")
    authority = p2a.mapping(config.get("authority"))
    allowed = {"exact_artifact_download_authorized", "safe_extraction_authorized"}
    for key, value in authority.items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")
    target = p2a.resolve(str(config.get("target") or ""))
    if not str(target).startswith(str((ROOT / "runtime" / "vcm_evaluator" / "toolchains").resolve())):
        faults.append("target_outside_vcm_toolchains")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED" if faults else "PAUSED",
        "state": "CONTRACT_INVALID" if faults else "READY_FOR_EXACT_YARN_ACQUISITION",
        "faults": sorted(set(faults)),
        "config": artifact_identity(config_path),
        "acquisition_executed": False,
        "network_requests": 0,
        "tool_execution_performed": False,
        "repository_execution_authorized": False,
        "dependency_installation_authorized": False,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def acquire(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path)
    if before["trigger_state"] == "RED":
        return before
    target = p2a.resolve(str(config["target"]))
    if target.exists():
        receipt, faults = inspect_target(target, config)
        return finish(before, config, receipt, faults, requests=0, reused=True)
    artifact = p2a.mapping(config["artifact"])
    maximum = int(config.get("maximum_archive_bytes") or 0)
    minimum_free = int(config.get("minimum_free_bytes_after_acquisition") or 0)
    free_before = shutil.disk_usage(target.parent if target.parent.exists() else ROOT).free
    faults = []
    receipt = {}
    requests = 0
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-yarn-", dir="/private/tmp") as raw:
        staging = Path(raw)
        archive_path = staging / "yarn.tgz"
        request = urllib.request.Request(str(artifact["url"]), headers={"User-Agent": "Project-Theseus-VCM-Yarn-Bootstrap/1"})
        context = ssl.create_default_context(cafile=str(config["ca_bundle"]))
        requests = 1
        with urllib.request.urlopen(request, context=context, timeout=float(config.get("timeout_seconds") or 60)) as response, archive_path.open("wb") as out:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    faults.append("archive_physical_boundary_hit")
                    break
                out.write(chunk)
        if faults:
            return finish(before, config, receipt, faults, requests=requests, reused=False, free_before=free_before)
        payload = archive_path.read_bytes()
        observed_sha1 = hashlib.sha1(payload).hexdigest()  # noqa: S324 - registry identity receipt
        observed_sha512 = base64.b64encode(hashlib.sha512(payload).digest()).decode()
        if observed_sha1 != artifact["sha1"]:
            faults.append("archive_sha1_mismatch")
        if observed_sha512 != artifact["sha512_base64"]:
            faults.append("archive_sha512_mismatch")
        extracted = staging / "extracted"
        extracted.mkdir()
        member_receipts, extraction_faults = safe_extract(archive_path, extracted)
        faults.extend(extraction_faults)
        package = extracted / "package"
        if not faults:
            target.parent.mkdir(parents=True, exist_ok=True)
            projected_free = shutil.disk_usage(target.parent).free - directory_bytes(package)
            if projected_free < minimum_free:
                faults.append("free_space_reserve_boundary_hit")
        if not faults:
            os.replace(package, target)
        receipt = {
            "url": artifact["url"],
            "archive_bytes": len(payload),
            "archive_sha1": observed_sha1,
            "archive_sha512_base64": observed_sha512,
            "member_receipts": member_receipts,
        }
    target_receipt, target_faults = inspect_target(target, config) if target.exists() else ({}, ["target_not_materialized"])
    faults.extend(target_faults)
    receipt["target"] = target_receipt
    return finish(before, config, receipt, faults, requests=requests, reused=False, free_before=free_before)


def safe_extract(archive: Path, destination: Path) -> tuple[list[dict[str, Any]], list[str]]:
    receipts = []
    faults = []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "package":
                faults.append(f"unsafe_member_path:{member.name}")
                continue
            output = destination.joinpath(*path.parts)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                faults.append(f"non_regular_member:{member.name}")
                continue
            extracted = handle.extractfile(member)
            payload = extracted.read() if extracted else b""
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
            output.chmod(0o755 if path.name in {"yarn.js", "yarnpkg.js"} else 0o644)
            receipts.append({"path": path.as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    return sorted(receipts, key=lambda row: row["path"]), sorted(set(faults))


def inspect_target(target: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults = []
    package_json = target / "package.json"
    yarn_js = target / "bin" / "yarn.js"
    if not package_json.is_file() or not yarn_js.is_file():
        faults.append("required_yarn_files_absent")
        metadata = {}
    else:
        try:
            metadata = json.loads(package_json.read_text())
        except json.JSONDecodeError:
            metadata = {}
            faults.append("package_json_invalid")
    if metadata.get("name") != "yarn" or metadata.get("version") != p2a.mapping(config["artifact"]).get("version"):
        faults.append("package_identity_invalid")
    files = []
    if target.is_dir():
        for path in sorted(target.rglob("*")):
            if path.is_file() and not path.is_symlink():
                files.append({"path": path.relative_to(target).as_posix(), "bytes": path.stat().st_size, "sha256": p2a.sha256_file(path)})
    return {
        "path": p2a.rel(target),
        "version": metadata.get("version"),
        "yarn_js": artifact_identity(yarn_js),
        "file_count": len(files),
        "files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "bytes": sum(row["bytes"] for row in files),
    }, sorted(set(faults))


def finish(before: dict[str, Any], config: dict[str, Any], receipt: dict[str, Any], faults: list[str], *, requests: int, reused: bool, free_before: int | None = None) -> dict[str, Any]:
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "EXACT_YARN_RUNTIME_MATERIALIZED_NOT_EXECUTED" if not faults else "YARN_ACQUISITION_FAILED",
        "faults": sorted(set(faults)),
        "acquisition_executed": not reused,
        "existing_target_reused": reused,
        "network_requests": requests,
        "free_bytes_before": free_before,
        "receipt": receipt,
        "tool_execution_performed": False,
        "repository_execution_authorized": False,
        "dependency_installation_authorized": False,
    }


def directory_bytes(path: Path) -> int:
    return sum(row.stat().st_size for row in path.rglob("*") if row.is_file())


def artifact_identity(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "acquisition_executed", "existing_target_reused",
        "network_requests", "tool_execution_performed", "parent_target_or_evaluator_executions",
        "candidate_or_control_calls", "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
