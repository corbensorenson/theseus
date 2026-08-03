#!/usr/bin/env python3
"""Acquire and offline-replay exactly one frozen VCM evaluator dependency lock."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import host_resource_safety as host_safety  # noqa: E402
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_dependency_prefetch_canary_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_dependency_prefetch_canary.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    config = p2a.read_json(config_path)
    report = execute(config, config_path) if args.execute else preflight(config, config_path)
    p2a.write_json(p2a.resolve(args.out or config["report"]), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    if config.get("state") != "PROSPECTIVE_SINGLE_TASK_3_DEPENDENCY_ACQUISITION_AND_OFFLINE_REPLAY":
        faults.append("state_invalid")
    owner = p2a.resolve(str(config.get("owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != str(config.get("owner_sha256") or ""):
        faults.append("owner_binding_invalid")
    reports: dict[str, Any] = {}
    for name, raw in p2a.mapping(config.get("reports")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"report_binding_invalid:{name}")
            reports[name] = {}
        else:
            reports[name] = p2a.read_json(path)
    plan = reports.get("prefetch_plan", {})
    node_report = reports.get("node_runtime", {})
    schedule = p2a.dicts(plan.get("schedule"))
    if plan.get("trigger_state") != "GREEN" or not schedule or schedule[0].get("index") != 3 or schedule[0].get("manager") != "npm":
        faults.append("prefetch_plan_first_row_invalid")
    if node_report.get("trigger_state") != "GREEN" or node_report.get("state") != "EXACT_NODE_RUNTIME_MATERIALIZED_AND_VERSION_QUALIFIED":
        faults.append("node_runtime_not_green")
    task = p2a.mapping(config.get("task"))
    if task.get("index") != 3 or task.get("repository") != "QoderAI/better-harness" or task.get("manager") != "npm":
        faults.append("task_binding_invalid")
    archives = p2a.mapping(config.get("archives"))
    observed_identities: dict[str, Any] = {}
    for label in ("parent", "target"):
        binding = p2a.mapping(archives.get(label))
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"archive_binding_invalid:{label}")
            continue
        identities, archive_faults = archive_lock_identities(path, str(binding.get("archive_root") or ""), config)
        faults.extend(f"{label}:{fault}" for fault in archive_faults)
        observed_identities[label] = identities
    if observed_identities.get("parent") != observed_identities.get("target"):
        faults.append("parent_target_dependency_identity_mismatch")
    for name, raw in p2a.mapping(config.get("tools")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"tool_binding_invalid:{name}")
    commands = p2a.mapping(config.get("commands"))
    online = p2a.strings(commands.get("online_acquisition_args"))
    offline = p2a.strings(commands.get("offline_replay_args"))
    required = {"ci", "--ignore-scripts", "--no-audit", "--no-fund"}
    if not required.issubset(set(online)) or offline != [*online, "--offline"]:
        faults.append("command_contract_invalid")
    authority = p2a.mapping(config.get("authority"))
    allowed = {
        "temporary_normalized_archive_extraction_authorized",
        "single_network_dependency_acquisition_authorized",
        "untrusted_dependency_archive_extraction_authorized",
        "lifecycle_script_disabled_dependency_installation_authorized",
        "network_denied_offline_replay_authorized",
        "content_addressed_cache_retention_authorized",
    }
    for key, value in authority.items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")
    limits = p2a.mapping(config.get("limits"))
    for key in ("minimum_free_bytes_after_execution", "maximum_retained_bytes", "wall_seconds_per_command", "cpu_seconds_per_command", "output_mib", "open_files", "user_processes", "max_process_group_rss_mib", "terminate_grace_seconds"):
        if float(limits.get(key) or 0) <= 0:
            faults.append(f"limit_invalid:{key}")
    store = p2a.resolve(str(config.get("retained_store") or ""))
    expected_parent = (ROOT / "runtime" / "vcm_evaluator" / "dependency_store" / "npm").resolve()
    if store.parent != expected_parent or store.name != "task-03":
        faults.append("retained_store_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED" if faults else "PAUSED",
        "state": "CONTRACT_INVALID" if faults else "READY_FOR_SINGLE_TASK_3_DEPENDENCY_CANARY",
        "faults": sorted(set(faults)),
        "config": identity(config_path),
        "observed_parent_target_dependency_identities": observed_identities,
        "execution_performed": False,
        "network_enabled_dependency_installations": 0,
        "network_denied_dependency_installations": 0,
        "dependency_installations": 0,
        "network_request_count_exact": None,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def execute(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path)
    if before["trigger_state"] == "RED":
        return before
    store = p2a.resolve(str(config["retained_store"]))
    if store.exists():
        return finish(before, ["retained_store_already_exists"], {}, executed=False)
    limits = p2a.mapping(config["limits"])
    free_before = shutil.disk_usage(ROOT).free
    if free_before < int(limits["minimum_free_bytes_after_execution"]):
        return finish(before, ["free_space_reserve_preflight_boundary_hit"], {"free_bytes_before": free_before}, executed=False)
    faults: list[str] = []
    receipts: dict[str, Any] = {"free_bytes_before": free_before}
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-dependency-canary-", dir="/private/tmp") as raw:
        work = Path(raw).resolve()
        repo = work / "repository"
        cache = work / "cache"
        home = work / "home"
        tmp = work / "tmp"
        cache.mkdir(); home.mkdir(); tmp.mkdir()
        target_binding = p2a.mapping(p2a.mapping(config["archives"])["target"])
        extraction_receipt, extraction_faults = safe_extract_repository(
            p2a.resolve(str(target_binding["path"])), repo, str(target_binding["archive_root"])
        )
        faults.extend(extraction_faults)
        receipts["repository_extraction"] = extraction_receipt
        source_before = tree_identity(repo, excluded_roots={"node_modules"}) if not faults else {}
        commands = p2a.mapping(config["commands"])
        base_args = p2a.strings(commands["online_acquisition_args"])
        online_command = npm_command(config, base_args, cache)
        env = {
            "HOME": str(home), "TMPDIR": str(tmp), "PATH": "/usr/bin:/bin",
            "CI": "1", "NO_COLOR": "1", "npm_config_update_notifier": "false",
        }
        if not faults:
            receipts["online_acquisition"] = run_sandboxed(online_command, repo, work, env, config, network_denied=False)
            if command_failed(receipts["online_acquisition"]):
                faults.append("online_dependency_acquisition_failed")
        online_modules = inspect_installed_dependencies(repo, config) if not faults else {}
        receipts["online_installed_dependencies"] = online_modules
        if not faults and online_modules.get("faults"):
            faults.extend(f"online:{fault}" for fault in online_modules["faults"])
        if (repo / "node_modules").exists():
            shutil.rmtree(repo / "node_modules")
        offline_command = npm_command(config, p2a.strings(commands["offline_replay_args"]), cache)
        if not faults:
            receipts["offline_replay"] = run_sandboxed(offline_command, repo, work, env, config, network_denied=True)
            if command_failed(receipts["offline_replay"]):
                faults.append("offline_dependency_replay_failed")
        offline_modules = inspect_installed_dependencies(repo, config) if not faults else {}
        receipts["offline_installed_dependencies"] = offline_modules
        if not faults and offline_modules.get("faults"):
            faults.extend(f"offline:{fault}" for fault in offline_modules["faults"])
        source_after = tree_identity(repo, excluded_roots={"node_modules"}) if repo.exists() else {}
        receipts["repository_source_before"] = source_before
        receipts["repository_source_after"] = source_after
        if source_before != source_after:
            faults.append("repository_source_mutated_outside_node_modules")
        cache_receipt, cache_faults = inspect_cache(cache, config)
        receipts["cache"] = cache_receipt
        faults.extend(cache_faults)
        retained_bytes = directory_bytes(cache)
        if retained_bytes > int(limits["maximum_retained_bytes"]):
            faults.append("retained_store_size_boundary_hit")
        projected_free = shutil.disk_usage(ROOT).free - retained_bytes
        if projected_free < int(limits["minimum_free_bytes_after_execution"]):
            faults.append("free_space_reserve_postflight_boundary_hit")
        if not faults:
            store.parent.mkdir(parents=True, exist_ok=True)
            os.replace(cache, store)
    receipts["retained_store"] = inspect_retained_store(store, config) if store.exists() else {}
    receipts["free_bytes_after"] = shutil.disk_usage(ROOT).free
    return finish(before, faults, receipts, executed=True)


def run_sandboxed(command: list[str], cwd: Path, work: Path, env: dict[str, str], config: dict[str, Any], *, network_denied: bool) -> dict[str, Any]:
    limits = p2a.mapping(config["limits"])
    backend = str(p2a.resolve(str(p2a.mapping(p2a.mapping(config["tools"])["sandbox_exec"])["path"])))
    profile = sandbox_profile(work, network_denied)
    stdout_path = work / ("offline.stdout" if network_denied else "online.stdout")
    stderr_path = work / ("offline.stderr" if network_denied else "online.stderr")

    def set_limits() -> None:
        mib = 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (int(limits["cpu_seconds_per_command"]),) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (int(limits["output_mib"]) * mib,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (int(limits["open_files"]),) * 2)
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (int(limits["user_processes"]),) * 2)

    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [backend, "-p", profile, *command], cwd=cwd, env=env, stdout=stdout, stderr=stderr,
            start_new_session=True, preexec_fn=set_limits,
        )
        boundary = ""
        telemetry_fault = ""
        maximum_rss = 0.0
        deadline = started + float(limits["wall_seconds_per_command"])
        while process.poll() is None:
            if time.monotonic() >= deadline:
                boundary = "wall_boundary_hit"; break
            try:
                maximum_rss = max(maximum_rss, host_safety.process_rss_mib(process.pid))
            except Exception as exc:
                telemetry_fault = f"{type(exc).__name__}:{exc}"[:1000]
                boundary = "rss_telemetry_unavailable"; break
            if maximum_rss > float(limits["max_process_group_rss_mib"]):
                boundary = "rss_boundary_hit"; break
            time.sleep(0.1)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=float(limits["terminate_grace_seconds"]))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
        else:
            returncode = int(process.returncode or 0)
    maximum = int(limits["output_mib"]) * 1024 * 1024
    stdout_bytes = stdout_path.read_bytes()
    stderr_bytes = stderr_path.read_bytes()
    if (len(stdout_bytes) >= maximum or len(stderr_bytes) >= maximum) and not boundary:
        boundary = "output_boundary_hit"
    return {
        "command": command,
        "returncode": returncode,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "network_denied": network_denied,
        "boundary_hit": bool(boundary),
        "boundary_reason": boundary,
        "rss_telemetry_fault": telemetry_fault,
        "maximum_process_group_rss_mib": round(maximum_rss, 3),
        "stdout_bytes": len(stdout_bytes),
        "stderr_bytes": len(stderr_bytes),
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "stderr_head": stderr_bytes.decode("utf-8", "replace")[:2000],
        "lifecycle_scripts_disabled_by_exact_cli_flag": "--ignore-scripts" in command,
    }


def sandbox_profile(work: Path, network_denied: bool) -> str:
    lines = ["(version 1)", "(allow default)"]
    if network_denied:
        lines.append("(deny network*)")
    lines.extend([
        f'(deny file-write* (require-not (subpath "{work}")))',
        '(allow file-write* (literal "/dev/null"))',
    ])
    return "\n".join(lines)


def npm_command(config: dict[str, Any], args: list[str], cache: Path) -> list[str]:
    tools = p2a.mapping(config["tools"])
    node = str(p2a.resolve(str(p2a.mapping(tools["node"])["path"])))
    npm_cli = str(p2a.resolve(str(p2a.mapping(tools["npm_cli"])["path"])))
    return [node, npm_cli, *args, "--cache", str(cache)]


def archive_lock_identities(archive: Path, archive_root: str, config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    required = p2a.mapping(p2a.mapping(config["task"])["required_files"])
    observed: dict[str, Any] = {}
    faults: list[str] = []
    with tarfile.open(archive, "r:gz") as handle:
        members = {member.name: member for member in handle.getmembers()}
        for label, binding_raw in required.items():
            binding = p2a.mapping(binding_raw)
            relative = str(binding.get("path") or "")
            member = members.get(f"{archive_root}/{relative}")
            if member is None or not member.isfile():
                faults.append(f"required_file_absent:{label}")
                continue
            extracted = handle.extractfile(member)
            payload = extracted.read() if extracted else b""
            sha256 = hashlib.sha256(payload).hexdigest()
            observed[label] = {"path": relative, "bytes": len(payload), "sha256": sha256}
            if sha256 != str(binding.get("sha256") or ""):
                faults.append(f"required_file_digest_mismatch:{label}")
    return observed, sorted(set(faults))


def safe_extract_repository(archive: Path, destination: Path, archive_root: str) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    receipts: list[dict[str, Any]] = []
    destination.mkdir()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or not path.parts or path.parts[0] != archive_root or ".." in path.parts:
                faults.append(f"unsafe_member_path:{member.name}")
                continue
            relative_parts = path.parts[1:]
            if not relative_parts:
                continue
            output = destination.joinpath(*relative_parts)
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
            output.chmod(0o755 if member.mode & 0o111 else 0o644)
            receipts.append({"path": PurePosixPath(*relative_parts).as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    return {
        "regular_file_count": len(receipts),
        "regular_file_bytes": sum(row["bytes"] for row in receipts),
        "member_receipts_sha256": digest_json(sorted(receipts, key=lambda row: row["path"])),
    }, sorted(set(faults))


def expected_lock_integrities(config: dict[str, Any]) -> dict[str, str]:
    target = p2a.mapping(p2a.mapping(config["archives"])["target"])
    archive = p2a.resolve(str(target["path"]))
    root = str(target["archive_root"])
    lock_path = str(p2a.mapping(p2a.mapping(config["task"])["required_files"])["lock"]["path"])
    with tarfile.open(archive, "r:gz") as handle:
        member = handle.getmember(f"{root}/{lock_path}")
        extracted = handle.extractfile(member)
        lock = json.loads((extracted.read() if extracted else b"{}").decode())
    result: dict[str, str] = {}
    for name, row in p2a.mapping(lock.get("packages")).items():
        if not name or not isinstance(row, dict) or not str(row.get("integrity") or "").startswith("sha512-"):
            continue
        digest = base64.b64decode(str(row["integrity"])[7:]).hex()
        result[name] = digest
    return result


def inspect_cache(cache: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    blobs = []
    content = cache / "_cacache" / "content-v2" / "sha512"
    for path in sorted(content.rglob("*")) if content.is_dir() else []:
        if not path.is_file():
            continue
        digest = hashlib.sha512(path.read_bytes()).hexdigest()
        blobs.append({"path": path.relative_to(cache).as_posix(), "bytes": path.stat().st_size, "sha512": digest})
    expected = expected_lock_integrities(config)
    observed = {row["sha512"] for row in blobs}
    missing = sorted(name for name, digest in expected.items() if digest not in observed)
    if missing:
        faults.append("lock_integrity_artifacts_missing:" + ",".join(missing))
    return {
        "content_blob_count": len(blobs),
        "content_blob_bytes": sum(row["bytes"] for row in blobs),
        "content_blobs_sha256": digest_json(blobs),
        "expected_lock_integrity_count": len(expected),
        "matched_lock_integrity_count": len(expected) - len(missing),
        "missing_lock_integrity_packages": missing,
        "total_cache_bytes": directory_bytes(cache),
        "total_cache_file_count": sum(1 for path in cache.rglob("*") if path.is_file()),
    }, sorted(set(faults))


def inspect_installed_dependencies(repo: Path, config: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    expected = p2a.mapping(p2a.mapping(config["task"])["expected_dependencies"])
    observed: dict[str, Any] = {}
    for name, version in expected.items():
        path = repo / "node_modules" / Path(*name.split("/")) / "package.json"
        try:
            meta = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            faults.append(f"installed_package_absent_or_invalid:{name}")
            continue
        observed[name] = {"version": meta.get("version"), "package_json_sha256": p2a.sha256_file(path)}
        if meta.get("version") != version:
            faults.append(f"installed_package_version_mismatch:{name}")
    return {"expected": expected, "observed": observed, "faults": sorted(set(faults))}


def tree_identity(root: Path, *, excluded_roots: set[str]) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in excluded_roots:
            continue
        if path.is_file() and not path.is_symlink():
            rows.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": p2a.sha256_file(path)})
    return {"file_count": len(rows), "bytes": sum(row["bytes"] for row in rows), "identity_sha256": digest_json(rows)}


def inspect_retained_store(store: Path, config: dict[str, Any]) -> dict[str, Any]:
    cache, faults = inspect_cache(store, config)
    return {"path": p2a.rel(store), **cache, "faults": faults}


def command_failed(receipt: dict[str, Any]) -> bool:
    return receipt.get("returncode") != 0 or receipt.get("boundary_hit") is True


def finish(before: dict[str, Any], faults: list[str], receipts: dict[str, Any], *, executed: bool) -> dict[str, Any]:
    online_count = int("online_acquisition" in receipts)
    offline_count = int("offline_replay" in receipts)
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if executed and not faults else "RED",
        "state": "TASK_3_DEPENDENCY_CACHE_ACQUIRED_AND_OFFLINE_REPLAY_QUALIFIED" if executed and not faults else "TASK_3_DEPENDENCY_CANARY_FAILED",
        "faults": sorted(set(faults)),
        "execution_performed": executed,
        "network_enabled_dependency_installations": online_count,
        "network_denied_dependency_installations": offline_count,
        "dependency_installations": online_count + offline_count,
        "network_request_count_exact": None,
        "receipts": receipts,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
    }


def directory_bytes(path: Path) -> int:
    return sum(row.stat().st_size for row in path.rglob("*") if row.is_file()) if path.exists() else 0


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "execution_performed",
        "network_enabled_dependency_installations", "network_denied_dependency_installations",
        "dependency_installations", "network_request_count_exact", "repository_runner_executions",
        "parent_target_or_evaluator_executions", "candidate_or_control_calls",
        "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
