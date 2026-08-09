#!/usr/bin/env python3
"""Independently rederive task 26's wheel-only uv instrument wall."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa: E402

POLICY = "project_theseus_vcm_task26_dependency_closure_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_task26_dependency_closure_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
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
    if (
        owner != Path(__file__).resolve()
        or not owner.is_file()
        or p2a.sha256_file(owner) != cfg.get("owner_sha256")
    ):
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
    if canary_cfg.get("policy") != "project_theseus_vcm_task26_dependency_canary_v1":
        faults.append("canary_config_invalid")
    if canary.get("trigger_state") != "RED" or canary.get("state") != "TASK_26_DEPENDENCY_CANARY_FAILED":
        faults.append("canary_terminal_state_invalid")

    for key, value in p2a.mapping(cfg.get("authority")).items():
        if value is not (key == "static_audit_authorized"):
            faults.append(f"authority_invalid:{key}")

    target = p2a.mapping(p2a.mapping(canary_cfg.get("archives")).get("target"))
    archive = p2a.resolve(str(target.get("path") or ""))
    archive_root = str(target.get("archive_root") or "")
    packages = lock_packages(archive, archive_root)
    package = next(
        (
            row
            for row in packages
            if normal(str(row.get("name") or "")) == "proxy-tools"
            and str(row.get("version") or "") == "0.1.0"
        ),
        {},
    )
    task = p2a.mapping(canary_cfg.get("task"))
    lock_checks = {
        "lock_package_count_exact": len(packages) == int(task.get("lock_package_count") or -1),
        "lock_identity_exact": digest(packages) == task.get("lock_artifact_identity_sha256"),
        "proxy_tools_0_1_0_present": bool(package),
        "proxy_tools_has_sdist": bool(package.get("sdist")),
        "proxy_tools_has_no_wheel": package.get("wheels") == [],
    }
    faults.extend(f"lock_check_failed:{key}" for key, value in lock_checks.items() if not value)

    receipts = p2a.mapping(canary.get("receipts"))
    online = p2a.mapping(receipts.get("online_sync"))
    tools = p2a.mapping(canary_cfg.get("tools"))
    uv = str(p2a.resolve(str(p2a.mapping(tools.get("uv")).get("path") or "")))
    python = str(p2a.resolve(str(p2a.mapping(tools.get("python")).get("path") or "")))
    online_args = p2a.strings(p2a.mapping(canary_cfg.get("commands")).get("online_sync_args"))
    command = p2a.strings(online.get("command"))
    prefix = [uv, *online_args, "--python", python, "--cache-dir"]
    stderr = str(online.get("stderr_head") or "")
    command_checks = {
        "online_command_exact": command[: len(prefix)] == prefix and len(command) == len(prefix) + 1,
        "online_returncode_is_uv_resolution_failure": online.get("returncode") == 2,
        "host_boundary_not_hit": online.get("boundary_hit") is False,
        "network_phase_was_enabled": online.get("network_denied") is False,
        "failure_names_exact_locked_distribution": "proxy-tools==0.1.0" in stderr,
        "failure_names_wheel_only_policy": "marked as `--no-build`" in stderr,
        "failure_names_missing_binary_distribution": "has no binary" in stderr,
        "stderr_receipt_matches": hashlib.sha256(stderr.encode()).hexdigest() == online.get("stderr_sha256"),
    }
    faults.extend(f"command_check_failed:{key}" for key, value in command_checks.items() if not value)

    source = archive_tree_identity(archive, archive_root)
    before = p2a.mapping(receipts.get("repository_source_before"))
    after = p2a.mapping(receipts.get("repository_source_after"))
    source_checks = {
        "source_before_after_identical": before == after,
        "archive_matches_source_before": source == before,
        "archive_matches_source_after": source == after,
    }
    faults.extend(f"source_check_failed:{key}" for key, value in source_checks.items() if not value)

    retained_store = p2a.resolve(str(canary_cfg.get("retained_store") or ""))
    boundary_checks = {
        "offline_replay_not_claimed": canary.get("network_denied_offline_replays") == 0,
        "retained_store_not_promoted": not retained_store.exists(),
        "source_builds_remained_forbidden": canary.get("source_build_executions") == 0,
        "project_install_remained_forbidden": canary.get("project_installations") == 0,
    }
    faults.extend(f"boundary_check_failed:{key}" for key, value in boundary_checks.items() if not value)
    zero_keys = (
        "repository_runner_executions",
        "parent_target_or_evaluator_executions",
        "candidate_or_control_calls",
        "external_reference_calls",
    )
    zero = {key: canary.get(key) for key in zero_keys}
    if any(value != 0 for value in zero.values()):
        faults.append("downstream_zero_counter_invalid")

    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "TASK_26_WHEEL_ONLY_INSTRUMENT_WALL_INDEPENDENTLY_REDERIVED" if not faults else "TASK_26_INSTRUMENT_WALL_AUDIT_FAILED",
        "disposition": "INCONCLUSIVE_INSTRUMENT_DEPENDENCY_POLICY_RISK_CLASS" if not faults else "AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "config": base.identity(path),
        "artifacts": {name: base.identity(artifact) for name, artifact in artifact_paths.items()},
        "lock_checks": lock_checks,
        "command_checks": command_checks,
        "source_checks": source_checks,
        "boundary_checks": boundary_checks,
        "observations": {
            "task_index": 26,
            "lock_package_count": len(packages),
            "blocking_distribution": "proxy-tools==0.1.0",
            "blocking_distribution_has_sdist": bool(package.get("sdist")),
            "blocking_distribution_wheel_count": len(package.get("wheels", [])),
            "canary_returncode": online.get("returncode"),
            "canary_duration_ms": online.get("duration_ms"),
            "source_file_count": source.get("file_count"),
            "source_bytes": source.get("bytes"),
            "source_identity_sha256": source.get("identity_sha256"),
        },
        "static_audit_only": True,
        "network_or_dependency_execution_performed": False,
        "source_build_executions": 0,
        "project_installations": 0,
        **{key: 0 for key in zero_keys},
        "maximum_inference": cfg.get("maximum_inference"),
    }


def lock_packages(archive: Path, root: str) -> list[dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as handle:
        entry = handle.extractfile(f"{root}/uv.lock")
        value = tomllib.loads((entry.read() if entry else b"").decode())
    return sorted(
        [
            {
                "name": row.get("name"),
                "version": row.get("version"),
                "source": row.get("source"),
                "sdist": row.get("sdist"),
                "wheels": row.get("wheels", []),
            }
            for row in value.get("package", [])
        ],
        key=lambda row: (str(row["name"]), str(row["version"])),
    )


def archive_tree_identity(archive: Path, root: str) -> dict[str, Any]:
    rows = []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.isfile() and member.name.startswith(root + "/"):
                entry = handle.extractfile(member)
                content = entry.read() if entry else b""
                rows.append(
                    {
                        "path": member.name[len(root) + 1 :],
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
    rows.sort(key=lambda row: PurePosixPath(row["path"]))
    return {
        "file_count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "identity_sha256": base.digest_json(rows),
    }


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normal(value: str) -> str:
    return value.lower().replace("_", "-")


def summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "trigger_state",
        "state",
        "disposition",
        "observations",
        "static_audit_only",
        "network_or_dependency_execution_performed",
        "source_build_executions",
        "project_installations",
        "repository_runner_executions",
        "parent_target_or_evaluator_executions",
        "candidate_or_control_calls",
        "external_reference_calls",
        "faults",
    )
    return {key: result.get(key) for key in keys}


if __name__ == "__main__":
    raise SystemExit(main())
