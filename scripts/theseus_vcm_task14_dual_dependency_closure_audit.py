#!/usr/bin/env python3
"""Independently rederive both retained uv closures for VCM task 14."""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import shutil
import sys
import tarfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa: E402

POLICY = "project_theseus_vcm_task14_dual_dependency_closure_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_task14_dual_dependency_closure_audit.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    path = p2a.resolve(args.config)
    result = audit(path)
    p2a.write_json(p2a.resolve(args.out or p2a.read_json(path)["report"]), result)
    print(json.dumps(summary(result), indent=2, sort_keys=True))
    return 0 if result["trigger_state"] == "GREEN" else 2


def audit(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != POLICY:
        faults.append("policy_invalid")
    owner = p2a.resolve(str(cfg.get("owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != cfg.get("owner_sha256"):
        faults.append("owner_binding_invalid")
    artifacts: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, Path] = {}
    for name, raw in p2a.mapping(cfg.get("artifacts")).items():
        binding = p2a.mapping(raw)
        artifact = p2a.resolve(str(binding.get("path") or ""))
        artifact_paths[name] = artifact
        if not artifact.is_file() or p2a.sha256_file(artifact) != binding.get("sha256"):
            faults.append(f"artifact_binding_invalid:{name}")
            artifacts[name] = {}
        else:
            artifacts[name] = p2a.read_json(artifact)
    canary_cfg = artifacts.get("canary_config", {})
    canary = artifacts.get("canary_report", {})
    sandbox = artifacts.get("trusted_build_canaries", {})
    if canary_cfg.get("policy") != "project_theseus_vcm_task14_dual_dependency_canary_v1":
        faults.append("canary_config_invalid")
    if canary.get("trigger_state") != "GREEN" or canary.get("state") != "TASK_14_SEPARATE_PARENT_TARGET_UV_CLOSURES_ACQUIRED_AND_OFFLINE_REPLAY_QUALIFIED":
        faults.append("canary_not_green")
    if sandbox.get("trigger_state") != "GREEN" or "network_denial_and_write_confinement" not in p2a.strings(sandbox.get("qualified_scopes")):
        faults.append("trusted_sandbox_invalid")
    for key, value in p2a.mapping(cfg.get("authority")).items():
        if value is not (key == "static_audit_authorized"):
            faults.append(f"authority_invalid:{key}")

    receipts = p2a.mapping(canary.get("receipts"))
    side_reports = p2a.mapping(receipts.get("sides"))
    observations: dict[str, Any] = {}
    side_checks: dict[str, Any] = {}
    for label in ("parent", "target"):
        side_cfg = p2a.mapping(p2a.mapping(canary_cfg.get("sides")).get(label))
        side_report = p2a.mapping(side_reports.get(label))
        archive = p2a.resolve(str(side_cfg.get("archive") or ""))
        root = str(side_cfg.get("archive_root") or "")
        lock = lock_packages(archive, root)
        lock_identity = digest(lock)
        source = archive_tree_identity(archive, root)
        store = p2a.resolve(str(side_cfg.get("retained_store") or ""))
        store_receipt = tree_identity(store)
        distributions = cached_distributions(store)
        locked_pairs = {(normal(str(row.get("name") or "")), str(row.get("version") or "")) for row in lock}
        observed_pairs = {(row["name"], row["version"]) for row in distributions}
        reported_store = p2a.mapping(side_report.get("retained_store"))
        reported_online = p2a.mapping(side_report.get("online_environment"))
        reported_offline = p2a.mapping(side_report.get("offline_environment"))
        online = p2a.mapping(side_report.get("online_sync"))
        offline = p2a.mapping(side_report.get("offline_replay"))
        online_args = p2a.strings(p2a.mapping(canary_cfg.get("commands")).get("online_sync_args"))
        offline_args = p2a.strings(p2a.mapping(canary_cfg.get("commands")).get("offline_replay_args"))
        tools = p2a.mapping(canary_cfg.get("tools"))
        uv = str(p2a.resolve(str(p2a.mapping(tools.get("uv")).get("path") or "")))
        python = str(p2a.resolve(str(p2a.mapping(tools.get("python")).get("path") or "")))
        checks = {
            "lock_package_count_exact": len(lock) == int(side_cfg.get("lock_package_count") or -1),
            "lock_identity_exact": lock_identity == side_cfg.get("lock_artifact_identity_sha256"),
            "source_before_after_identical": p2a.mapping(side_report.get("repository_source_before")) == p2a.mapping(side_report.get("repository_source_after")),
            "archive_matches_source_receipt": source == p2a.mapping(side_report.get("repository_source_before")),
            "store_file_count_matches": store_receipt.get("file_count") == reported_store.get("file_count"),
            "store_bytes_match": store_receipt.get("bytes") == reported_store.get("bytes"),
            "store_identity_matches": store_receipt.get("identity_sha256") == reported_store.get("identity_sha256"),
            "cached_distribution_count_exact": len(distributions) == 12,
            "cached_distributions_are_locked": observed_pairs <= locked_pairs,
            "required_runtime_dependencies_present": {"httpx", "pydantic", "websockets"} <= {row["name"] for row in distributions},
            "project_not_cached_as_distribution": "scrapebadger" not in {row["name"] for row in distributions},
            "online_offline_distributions_match": reported_online.get("distributions") == reported_offline.get("distributions"),
            "cached_metadata_matches_report": distributions == reported_offline.get("distributions"),
            "online_command_exact": command_exact(online, uv, online_args, python, label),
            "offline_command_exact": command_exact(offline, uv, offline_args, python, label),
            "online_phase_green": online.get("returncode") == 0 and online.get("boundary_hit") is False and online.get("network_denied") is False,
            "offline_phase_green": offline.get("returncode") == 0 and offline.get("boundary_hit") is False and offline.get("network_denied") is True,
            "no_sdist_payload": no_sdist_payload(store),
        }
        faults.extend(f"{label}:check_failed:{key}" for key, value in checks.items() if not value)
        side_checks[label] = checks
        observations[label] = {
            "lock_package_count": len(lock),
            "lock_artifact_identity_sha256": lock_identity,
            "cached_distribution_count": len(distributions),
            "retained_store_file_count": store_receipt.get("file_count"),
            "retained_store_bytes": store_receipt.get("bytes"),
            "retained_store_identity_sha256": store_receipt.get("identity_sha256"),
            "source_file_count": source.get("file_count"),
            "source_bytes": source.get("bytes"),
            "source_identity_sha256": source.get("identity_sha256"),
        }

    limits = p2a.mapping(canary_cfg.get("limits"))
    current_free = shutil.disk_usage(ROOT).free
    storage_checks = {
        "run_before_above_reserve": int(receipts.get("free_bytes_before") or 0) >= int(limits.get("minimum_free_bytes_after_execution") or 0),
        "run_after_above_reserve": int(receipts.get("free_bytes_after") or 0) >= int(limits.get("minimum_free_bytes_after_execution") or 0),
        "current_above_reserve": current_free >= int(limits.get("minimum_free_bytes_after_execution") or 0),
        "reported_total_matches_sides": int(receipts.get("retained_total_bytes") or -1) == sum(int(observations[x]["retained_store_bytes"] or 0) for x in ("parent", "target")),
        "total_below_ceiling": int(receipts.get("retained_total_bytes") or 0) <= int(limits.get("maximum_total_retained_bytes") or 0),
        "each_side_below_ceiling": all(int(observations[x]["retained_store_bytes"] or 0) <= int(limits.get("maximum_retained_bytes_per_side") or 0) for x in ("parent", "target")),
    }
    faults.extend(f"storage_check_failed:{key}" for key, value in storage_checks.items() if not value)
    zero_keys = ("source_build_executions", "project_installations", "repository_runner_executions", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls")
    zero = {key: canary.get(key) for key in zero_keys}
    if any(value != 0 for value in zero.values()):
        faults.append("downstream_zero_counter_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "TASK_14_DUAL_UV_DEPENDENCY_CLOSURES_INDEPENDENTLY_REDERIVED" if not faults else "TASK_14_DUAL_UV_DEPENDENCY_CLOSURE_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "config": base.identity(path),
        "artifacts": {name: base.identity(artifact) for name, artifact in artifact_paths.items()},
        "side_checks": side_checks,
        "storage_checks": storage_checks,
        "observations": {**observations, "current_free_bytes": current_free},
        "static_audit_only": True,
        "network_or_dependency_execution_performed": False,
        **{key: 0 for key in zero_keys},
        "maximum_inference": cfg.get("maximum_inference"),
    }


def command_exact(receipt: dict[str, Any], uv: str, args: list[str], python: str, label: str) -> bool:
    command = p2a.strings(receipt.get("command"))
    prefix = [uv, *args, "--python", python, "--cache-dir"]
    return command[: len(prefix)] == prefix and len(command) == len(prefix) + 1 and command[-1].endswith(f"/{label}/cache")


def lock_packages(archive: Path, root: str) -> list[dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as handle:
        entry = handle.extractfile(f"{root}/uv.lock")
        value = tomllib.loads((entry.read() if entry else b"").decode())
    return sorted(
        [{"name": row.get("name"), "version": row.get("version"), "source": row.get("source"), "sdist": row.get("sdist"), "wheels": row.get("wheels", [])} for row in value.get("package", [])],
        key=lambda row: (str(row["name"]), str(row["version"])),
    )


def archive_tree_identity(archive: Path, root: str) -> dict[str, Any]:
    rows = []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.isfile() and member.name.startswith(root + "/"):
                entry = handle.extractfile(member)
                content = entry.read() if entry else b""
                rows.append({"path": member.name[len(root) + 1 :], "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    rows.sort(key=lambda row: PurePosixPath(row["path"]))
    return {"file_count": len(rows), "bytes": sum(row["bytes"] for row in rows), "identity_sha256": base.digest_json(rows)}


def tree_identity(root: Path) -> dict[str, Any]:
    files = [path for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink()] if root.exists() else []
    rows = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": p2a.sha256_file(path)} for path in files]
    return {"file_count": len(rows), "bytes": sum(row["bytes"] for row in rows), "identity_sha256": base.digest_json(rows)}


def cached_distributions(store: Path) -> list[dict[str, str]]:
    parser = email.parser.Parser()
    rows = []
    for metadata in sorted((store / "archive-v0").glob("*/*.dist-info/METADATA")):
        parsed = parser.parsestr(metadata.read_text(errors="replace"))
        rows.append({"name": normal(str(parsed.get("Name") or "")), "version": str(parsed.get("Version") or ""), "metadata_sha256": p2a.sha256_file(metadata)})
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def no_sdist_payload(store: Path) -> bool:
    root = store / "sdists-v9"
    return not any(path.is_file() and path.name not in {".git", ".gitignore"} for path in root.rglob("*")) if root.exists() else True


def normal(value: str) -> str:
    return value.lower().replace("_", "-")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("trigger_state", "state", "observations", "static_audit_only", "network_or_dependency_execution_performed", "source_build_executions", "project_installations", "repository_runner_executions", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls", "faults")
    return {key: result.get(key) for key in keys}


if __name__ == "__main__":
    raise SystemExit(main())
