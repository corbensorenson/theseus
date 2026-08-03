#!/usr/bin/env python3
"""Acquire the exact Node runtime required by the first VCM dependency canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_node_bootstrap_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_node_bootstrap.json"


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
    if config.get("state") != "PROSPECTIVE_EXACT_NODE_RUNTIME_FOR_VCM_TASK_3_DEPENDENCY_CANARY_ONLY":
        faults.append("state_invalid")
    owner = p2a.resolve(str(config.get("owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != str(config.get("owner_sha256") or ""):
        faults.append("owner_binding_invalid")
    ca = Path(str(config.get("ca_bundle") or ""))
    if not ca.is_file() or p2a.sha256_file(ca) != str(config.get("ca_bundle_sha256") or ""):
        faults.append("ca_bundle_identity_invalid")
    artifact = p2a.mapping(config.get("artifact"))
    if artifact.get("version") != "22.20.0" or artifact.get("npm_version") != "10.9.3":
        faults.append("artifact_version_invalid")
    if artifact.get("archive_name") != "node-v22.20.0-darwin-arm64.tar.gz":
        faults.append("artifact_name_invalid")
    if artifact.get("url") != "https://nodejs.org/dist/v22.20.0/node-v22.20.0-darwin-arm64.tar.gz":
        faults.append("artifact_url_invalid")
    if len(str(artifact.get("sha256") or "")) != 64:
        faults.append("artifact_digest_invalid")
    checksum = p2a.mapping(config.get("checksum_source"))
    if checksum.get("url") != "https://nodejs.org/dist/v22.20.0/SHASUMS256.txt" or len(str(checksum.get("sha256") or "")) != 64:
        faults.append("checksum_source_invalid")
    target = p2a.resolve(str(config.get("target") or ""))
    toolchain_root = (ROOT / "runtime" / "vcm_evaluator" / "toolchains").resolve()
    if target.parent != toolchain_root or target.name != "node-v22.20.0-darwin-arm64":
        faults.append("target_invalid")
    if int(config.get("maximum_archive_bytes") or 0) <= 0 or int(config.get("minimum_free_bytes_after_acquisition") or 0) <= 0:
        faults.append("physical_policy_invalid")
    authority = p2a.mapping(config.get("authority"))
    allowed = {"exact_artifact_download_authorized", "safe_extraction_authorized", "exact_version_probe_authorized"}
    for key, value in authority.items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED" if faults else "PAUSED",
        "state": "CONTRACT_INVALID" if faults else "READY_FOR_EXACT_NODE_RUNTIME_ACQUISITION",
        "faults": sorted(set(faults)),
        "config": identity(config_path),
        "acquisition_executed": False,
        "network_requests": 0,
        "tool_version_probes": 0,
        "dependency_installations": 0,
        "repository_executions": 0,
        "parent_target_or_evaluator_executions": 0,
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
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-node-", dir="/private/tmp") as raw:
        staging = Path(raw)
        archive_path = staging / str(artifact["archive_name"])
        request = urllib.request.Request(str(artifact["url"]), headers={"User-Agent": "Project-Theseus-VCM-Node-Bootstrap/1"})
        context = ssl.create_default_context(cafile=str(config["ca_bundle"]))
        with urllib.request.urlopen(request, context=context, timeout=float(config.get("timeout_seconds") or 120)) as response, archive_path.open("wb") as out:
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
        archive_sha256 = p2a.sha256_file(archive_path) if archive_path.is_file() else ""
        if archive_sha256 != artifact["sha256"]:
            faults.append("archive_sha256_mismatch")
        extracted = staging / "extracted"
        extracted.mkdir()
        members, extraction_faults = safe_extract(archive_path, extracted, str(artifact["archive_root"])) if not faults else ([], [])
        faults.extend(extraction_faults)
        package = extracted / str(artifact["archive_root"])
        if not faults:
            projected_free = shutil.disk_usage(target.parent).free - directory_bytes(package)
            if projected_free < minimum_free:
                faults.append("free_space_reserve_boundary_hit")
        if not faults:
            os.replace(package, target)
        receipt = {
            "url": artifact["url"],
            "archive_bytes": archive_path.stat().st_size if archive_path.is_file() else 0,
            "archive_sha256": archive_sha256,
            "member_count": len(members),
            "member_receipts_sha256": digest_json(members),
            "free_bytes_before": free_before,
        }
    target_receipt, target_faults = inspect_target(target, config, probe=True) if target.exists() else ({}, ["target_not_materialized"])
    faults.extend(target_faults)
    receipt["target"] = target_receipt
    receipt["free_bytes_after"] = shutil.disk_usage(target.parent).free
    return finish(before, receipt, faults, requests=1, reused=False)


def safe_extract(archive: Path, destination: Path, archive_root: str) -> tuple[list[dict[str, Any]], list[str]]:
    receipts: list[dict[str, Any]] = []
    faults: list[str] = []
    symlinks: list[tuple[Path, str, str]] = []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            normalized = normalize_member(member.name, archive_root)
            if normalized is None:
                faults.append(f"unsafe_member_path:{member.name}")
                continue
            output = destination.joinpath(*PurePosixPath(normalized).parts)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if member.isfile():
                extracted = handle.extractfile(member)
                payload = extracted.read() if extracted else b""
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(payload)
                output.chmod(0o755 if member.mode & 0o111 else 0o644)
                receipts.append({"path": normalized, "kind": "file", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
                continue
            if member.issym() and safe_symlink_target(normalized, member.linkname, archive_root):
                symlinks.append((output, member.linkname, normalized))
                continue
            faults.append(f"non_regular_or_unsafe_link_member:{member.name}")
    if not faults:
        for output, linkname, normalized in symlinks:
            output.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(linkname, output)
            receipts.append({"path": normalized, "kind": "symlink", "target": linkname})
    return sorted(receipts, key=lambda row: row["path"]), sorted(set(faults))


def normalize_member(name: str, archive_root: str) -> str | None:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or path.parts[0] != archive_root or ".." in path.parts:
        return None
    return path.as_posix()


def safe_symlink_target(member_name: str, linkname: str, archive_root: str) -> bool:
    target = PurePosixPath(linkname)
    if target.is_absolute():
        return False
    parts: list[str] = list(PurePosixPath(member_name).parent.parts)
    for part in target.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return False
            parts.pop()
        else:
            parts.append(part)
    return bool(parts) and parts[0] == archive_root


def inspect_target(target: Path, config: dict[str, Any], *, probe: bool) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    artifact = p2a.mapping(config["artifact"])
    node = target / "bin" / "node"
    npm_cli = target / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
    npm_package = target / "lib" / "node_modules" / "npm" / "package.json"
    if not node.is_file() or not npm_cli.is_file() or not npm_package.is_file():
        faults.append("required_runtime_files_absent")
        npm_meta = {}
    else:
        try:
            npm_meta = json.loads(npm_package.read_text())
        except json.JSONDecodeError:
            npm_meta = {}
            faults.append("npm_package_json_invalid")
    if npm_meta.get("name") != "npm" or npm_meta.get("version") != artifact.get("npm_version"):
        faults.append("npm_static_identity_invalid")
    probes: dict[str, Any] = {}
    if probe and node.is_file() and npm_cli.is_file():
        probes["node"] = version_probe([str(node), "--version"], target)
        probes["npm"] = version_probe([str(node), str(npm_cli), "--version"], target)
        if probes["node"].get("stdout") != f"v{artifact['version']}" or probes["node"].get("returncode") != 0:
            faults.append("node_version_probe_invalid")
        if probes["npm"].get("stdout") != artifact["npm_version"] or probes["npm"].get("returncode") != 0:
            faults.append("npm_version_probe_invalid")
    files = [path for path in sorted(target.rglob("*")) if path.is_file() and not path.is_symlink()]
    symlinks = [path for path in sorted(target.rglob("*")) if path.is_symlink()]
    return {
        "path": p2a.rel(target),
        "node": identity(node),
        "npm_cli": identity(npm_cli),
        "npm_version": npm_meta.get("version"),
        "file_count": len(files),
        "symlink_count": len(symlinks),
        "bytes": sum(path.stat().st_size for path in files),
        "files_identity_sha256": digest_json([{"path": path.relative_to(target).as_posix(), "bytes": path.stat().st_size, "sha256": p2a.sha256_file(path)} for path in files]),
        "symlinks_identity_sha256": digest_json([{"path": path.relative_to(target).as_posix(), "target": os.readlink(path)} for path in symlinks]),
        "version_probes": probes,
    }, sorted(set(faults))


def version_probe(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, env={"HOME": str(cwd), "PATH": "/usr/bin:/bin", "NO_COLOR": "1"}, capture_output=True, check=False, timeout=15)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.decode("utf-8", "replace").strip(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }


def finish(before: dict[str, Any], receipt: dict[str, Any], faults: list[str], *, requests: int, reused: bool) -> dict[str, Any]:
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "EXACT_NODE_RUNTIME_MATERIALIZED_AND_VERSION_QUALIFIED" if not faults else "NODE_RUNTIME_ACQUISITION_FAILED",
        "faults": sorted(set(faults)),
        "acquisition_executed": not reused,
        "existing_target_reused": reused,
        "network_requests": requests,
        "tool_version_probes": 2 if receipt else 0,
        "receipt": receipt,
        "dependency_installations": 0,
        "repository_executions": 0,
    }


def directory_bytes(path: Path) -> int:
    return sum(row.stat().st_size for row in path.rglob("*") if row.is_file() and not row.is_symlink())


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "acquisition_executed", "existing_target_reused",
        "network_requests", "tool_version_probes", "dependency_installations",
        "repository_executions", "parent_target_or_evaluator_executions",
        "candidate_or_control_calls", "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
