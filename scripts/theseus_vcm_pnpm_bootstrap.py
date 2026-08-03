#!/usr/bin/env python3
"""Acquire exact pnpm 10.32.1 for the second scheduled VCM lock task."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_yarn_bootstrap as common  # noqa: E402

POLICY = "project_theseus_vcm_pnpm_bootstrap_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_pnpm_bootstrap.json"


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
    if config.get("state") != "PROSPECTIVE_EXACT_PNPM_10_32_1_FOR_VCM_TASK_7_ONLY":
        faults.append("state_invalid")
    owner = p2a.resolve(str(config.get("owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != str(config.get("owner_sha256") or ""):
        faults.append("owner_binding_invalid")
    artifact = p2a.mapping(config.get("artifact"))
    if artifact.get("name") != "pnpm" or artifact.get("version") != "10.32.1" or artifact.get("url") != "https://registry.npmjs.org/pnpm/-/pnpm-10.32.1.tgz":
        faults.append("artifact_identity_invalid")
    if len(str(artifact.get("sha1") or "")) != 40 or not str(artifact.get("sha512_base64") or ""):
        faults.append("artifact_digest_invalid")
    ca = Path(str(config.get("ca_bundle") or ""))
    if not ca.is_file() or p2a.sha256_file(ca) != str(config.get("ca_bundle_sha256") or ""):
        faults.append("ca_bundle_invalid")
    node = p2a.mapping(config.get("node"))
    node_path = p2a.resolve(str(node.get("path") or ""))
    if not node_path.is_file() or p2a.sha256_file(node_path) != str(node.get("sha256") or "") or node.get("version") != "22.20.0":
        faults.append("node_binding_invalid")
    target = p2a.resolve(str(config.get("target") or ""))
    root = (ROOT / "runtime" / "vcm_evaluator" / "toolchains").resolve()
    if target.parent != root or target.name != "pnpm-10.32.1":
        faults.append("target_invalid")
    authority = p2a.mapping(config.get("authority"))
    allowed = {"exact_artifact_download_authorized", "safe_extraction_authorized", "exact_version_probe_authorized"}
    for key, value in authority.items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED" if faults else "PAUSED",
        "state": "CONTRACT_INVALID" if faults else "READY_FOR_EXACT_PNPM_10_32_1_ACQUISITION",
        "faults": sorted(set(faults)),
        "config": identity(config_path),
        "acquisition_executed": False,
        "network_requests": 0,
        "tool_version_probes": 0,
        "dependency_installations": 0,
        "repository_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def acquire(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path)
    if before["trigger_state"] == "RED":
        return before
    target = p2a.resolve(str(config["target"]))
    if target.exists():
        receipt, faults = inspect_target(target, config, probe=True)
        return finish(before, receipt, faults, requests=0, reused=True)
    artifact = p2a.mapping(config["artifact"])
    maximum = int(config["maximum_archive_bytes"])
    minimum_free = int(config["minimum_free_bytes_after_acquisition"])
    target.parent.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(target.parent).free
    faults: list[str] = []
    receipt: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-pnpm-", dir="/private/tmp") as raw:
        staging = Path(raw)
        archive = staging / "pnpm.tgz"
        request = urllib.request.Request(str(artifact["url"]), headers={"User-Agent": "Project-Theseus-VCM-pnpm-Bootstrap/1"})
        context = ssl.create_default_context(cafile=str(config["ca_bundle"]))
        with urllib.request.urlopen(request, context=context, timeout=float(config.get("timeout_seconds") or 120)) as response, archive.open("wb") as out:
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
        payload = archive.read_bytes() if archive.is_file() else b""
        sha1 = hashlib.sha1(payload).hexdigest()  # noqa: S324 - registry identity
        sha512 = base64.b64encode(hashlib.sha512(payload).digest()).decode()
        if sha1 != artifact["sha1"]:
            faults.append("archive_sha1_mismatch")
        if sha512 != artifact["sha512_base64"]:
            faults.append("archive_sha512_mismatch")
        extracted = staging / "extracted"
        extracted.mkdir()
        members, extraction_faults = common.safe_extract(archive, extracted) if not faults else ([], [])
        faults.extend(extraction_faults)
        package = extracted / "package"
        if not faults and shutil.disk_usage(target.parent).free - common.directory_bytes(package) < minimum_free:
            faults.append("free_space_reserve_boundary_hit")
        if not faults:
            os.replace(package, target)
        receipt = {
            "url": artifact["url"], "archive_bytes": len(payload),
            "archive_sha1": sha1, "archive_sha512_base64": sha512,
            "member_count": len(members), "member_receipts_sha256": digest_json(members),
            "free_bytes_before": free_before,
        }
    target_receipt, target_faults = inspect_target(target, config, probe=True) if target.exists() else ({}, ["target_not_materialized"])
    faults.extend(target_faults)
    receipt["target"] = target_receipt
    receipt["free_bytes_after"] = shutil.disk_usage(target.parent).free
    return finish(before, receipt, faults, requests=1, reused=False)


def inspect_target(target: Path, config: dict[str, Any], *, probe: bool) -> tuple[dict[str, Any], list[str]]:
    faults = []
    package_json = target / "package.json"
    pnpm_cli = target / "bin" / "pnpm.cjs"
    try:
        metadata = json.loads(package_json.read_text())
    except (OSError, json.JSONDecodeError):
        metadata = {}
        faults.append("package_json_invalid")
    if metadata.get("name") != "pnpm" or metadata.get("version") != "10.32.1" or not pnpm_cli.is_file():
        faults.append("package_identity_invalid")
    probe_receipt: dict[str, Any] = {}
    if probe and pnpm_cli.is_file():
        node = str(p2a.resolve(str(p2a.mapping(config["node"])["path"])))
        completed = subprocess.run([node, str(pnpm_cli), "--version"], cwd=target, env={"HOME": str(target), "PATH": "/usr/bin:/bin", "NO_COLOR": "1"}, capture_output=True, check=False, timeout=15)
        probe_receipt = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.decode("utf-8", "replace").strip(),
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        }
        if completed.returncode != 0 or probe_receipt["stdout"] != "10.32.1":
            faults.append("version_probe_invalid")
    files = [path for path in sorted(target.rglob("*")) if path.is_file() and not path.is_symlink()]
    return {
        "path": p2a.rel(target), "version": metadata.get("version"),
        "pnpm_cli": identity(pnpm_cli), "version_probe": probe_receipt,
        "file_count": len(files), "bytes": sum(path.stat().st_size for path in files),
        "files_identity_sha256": digest_json([{"path": path.relative_to(target).as_posix(), "bytes": path.stat().st_size, "sha256": p2a.sha256_file(path)} for path in files]),
    }, sorted(set(faults))


def finish(before: dict[str, Any], receipt: dict[str, Any], faults: list[str], *, requests: int, reused: bool) -> dict[str, Any]:
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "EXACT_PNPM_10_32_1_MATERIALIZED_AND_VERSION_QUALIFIED" if not faults else "PNPM_10_32_1_ACQUISITION_FAILED",
        "faults": sorted(set(faults)), "acquisition_executed": not reused,
        "existing_target_reused": reused, "network_requests": requests,
        "tool_version_probes": 1 if receipt else 0, "receipt": receipt,
        "dependency_installations": 0, "repository_executions": 0,
    }


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "acquisition_executed", "existing_target_reused",
        "network_requests", "tool_version_probes", "dependency_installations",
        "repository_executions", "candidate_or_control_calls", "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
