#!/usr/bin/env python3
"""Repair only VCM dependency-canary output/file-size boundary coupling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import host_resource_safety as host_safety  # noqa: E402
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa: E402

POLICY = "project_theseus_vcm_dependency_prefetch_canary_v2"
STATE = "PROSPECTIVE_V2_TASK_3_DEPENDENCY_CANARY_SEPARATE_OUTPUT_AND_DEPENDENCY_FILE_LIMITS"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_dependency_prefetch_canary_v2.json"


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
    print(json.dumps(base.summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    inherited = copy.deepcopy(config)
    inherited["policy"] = base.POLICY
    inherited["state"] = "PROSPECTIVE_SINGLE_TASK_3_DEPENDENCY_ACQUISITION_AND_OFFLINE_REPLAY"
    inherited["owner"] = p2a.rel(Path(base.__file__).resolve())
    inherited["owner_sha256"] = p2a.sha256_file(Path(base.__file__).resolve())
    report = base.preflight(inherited, config_path)
    faults = list(p2a.strings(report.get("faults")))
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    if config.get("state") != STATE:
        faults.append("state_invalid")
    owner = p2a.resolve(str(config.get("owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != str(config.get("owner_sha256") or ""):
        faults.append("owner_binding_invalid")
    predecessor_binding = p2a.mapping(config.get("predecessor_report"))
    predecessor_path = p2a.resolve(str(predecessor_binding.get("path") or ""))
    if not predecessor_path.is_file() or p2a.sha256_file(predecessor_path) != str(predecessor_binding.get("sha256") or ""):
        faults.append("predecessor_report_binding_invalid")
        predecessor = {}
    else:
        predecessor = p2a.read_json(predecessor_path)
    if predecessor.get("trigger_state") != "RED" or predecessor.get("faults") != ["online_dependency_acquisition_failed"]:
        faults.append("predecessor_disposition_invalid")
    online = p2a.mapping(p2a.mapping(predecessor.get("receipts")).get("online_acquisition"))
    if online.get("returncode") != 1 or "SIGXFSZ" not in str(online.get("stderr_head") or ""):
        faults.append("predecessor_fsize_fault_not_bound")
    limits = p2a.mapping(config.get("limits"))
    output_mib = int(limits.get("output_mib") or 0)
    file_mib = int(limits.get("maximum_single_written_file_mib") or 0)
    retained_mib = int(limits.get("maximum_retained_bytes") or 0) // (1024 * 1024)
    if output_mib != 8 or file_mib <= output_mib or file_mib != retained_mib:
        faults.append("separated_file_boundary_invalid")
    return {
        **report,
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED" if faults else "PAUSED",
        "state": "CONTRACT_INVALID" if faults else "READY_FOR_V2_TASK_3_DEPENDENCY_CANARY",
        "faults": sorted(set(faults)),
        "config": base.identity(config_path),
        "predecessor_report": base.identity(predecessor_path),
        "boundary_repair": {
            "captured_output_mib": output_mib,
            "maximum_single_written_file_mib": file_mib,
            "maximum_retained_bytes": int(limits.get("maximum_retained_bytes") or 0),
            "captured_output_monitored_independently": True,
        },
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
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-dependency-canary-v2-", dir="/private/tmp") as raw:
        work = Path(raw).resolve()
        repo = work / "repository"
        cache = work / "cache"
        home = work / "home"
        tmp = work / "tmp"
        cache.mkdir(); home.mkdir(); tmp.mkdir()
        target_binding = p2a.mapping(p2a.mapping(config["archives"])["target"])
        extraction_receipt, extraction_faults = base.safe_extract_repository(
            p2a.resolve(str(target_binding["path"])), repo, str(target_binding["archive_root"])
        )
        faults.extend(extraction_faults)
        receipts["repository_extraction"] = extraction_receipt
        source_before = base.tree_identity(repo, excluded_roots={"node_modules"}) if not faults else {}
        commands = p2a.mapping(config["commands"])
        env = {
            "HOME": str(home), "TMPDIR": str(tmp), "PATH": "/usr/bin:/bin",
            "CI": "1", "NO_COLOR": "1", "npm_config_update_notifier": "false",
        }
        online_command = base.npm_command(config, p2a.strings(commands["online_acquisition_args"]), cache)
        if not faults:
            receipts["online_acquisition"] = run_sandboxed(online_command, repo, work, env, config, network_denied=False)
            if base.command_failed(receipts["online_acquisition"]):
                faults.append("online_dependency_acquisition_failed")
        receipts["online_installed_dependencies"] = base.inspect_installed_dependencies(repo, config) if not faults else {}
        if not faults and receipts["online_installed_dependencies"].get("faults"):
            faults.extend(f"online:{fault}" for fault in receipts["online_installed_dependencies"]["faults"])
        if (repo / "node_modules").exists():
            shutil.rmtree(repo / "node_modules")
        offline_command = base.npm_command(config, p2a.strings(commands["offline_replay_args"]), cache)
        if not faults:
            receipts["offline_replay"] = run_sandboxed(offline_command, repo, work, env, config, network_denied=True)
            if base.command_failed(receipts["offline_replay"]):
                faults.append("offline_dependency_replay_failed")
        receipts["offline_installed_dependencies"] = base.inspect_installed_dependencies(repo, config) if not faults else {}
        if not faults and receipts["offline_installed_dependencies"].get("faults"):
            faults.extend(f"offline:{fault}" for fault in receipts["offline_installed_dependencies"]["faults"])
        source_after = base.tree_identity(repo, excluded_roots={"node_modules"}) if repo.exists() else {}
        receipts["repository_source_before"] = source_before
        receipts["repository_source_after"] = source_after
        if source_before != source_after:
            faults.append("repository_source_mutated_outside_node_modules")
        cache_receipt, cache_faults = base.inspect_cache(cache, config)
        receipts["cache"] = cache_receipt
        faults.extend(cache_faults)
        retained_bytes = base.directory_bytes(cache)
        if retained_bytes > int(limits["maximum_retained_bytes"]):
            faults.append("retained_store_size_boundary_hit")
        if shutil.disk_usage(ROOT).free - retained_bytes < int(limits["minimum_free_bytes_after_execution"]):
            faults.append("free_space_reserve_postflight_boundary_hit")
        if not faults:
            store.parent.mkdir(parents=True, exist_ok=True)
            os.replace(cache, store)
    receipts["retained_store"] = base.inspect_retained_store(store, config) if store.exists() else {}
    receipts["free_bytes_after"] = shutil.disk_usage(ROOT).free
    return finish(before, faults, receipts, executed=True)


def run_sandboxed(command: list[str], cwd: Path, work: Path, env: dict[str, str], config: dict[str, Any], *, network_denied: bool) -> dict[str, Any]:
    limits = p2a.mapping(config["limits"])
    backend = str(p2a.resolve(str(p2a.mapping(p2a.mapping(config["tools"])["sandbox_exec"])["path"])))
    profile = base.sandbox_profile(work, network_denied)
    stdout_path = work / ("offline.stdout" if network_denied else "online.stdout")
    stderr_path = work / ("offline.stderr" if network_denied else "online.stderr")

    def set_limits() -> None:
        mib = 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (int(limits["cpu_seconds_per_command"]),) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (int(limits["maximum_single_written_file_mib"]) * mib,) * 2)
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
        maximum_output = int(limits["output_mib"]) * 1024 * 1024
        deadline = started + float(limits["wall_seconds_per_command"])
        while process.poll() is None:
            if time.monotonic() >= deadline:
                boundary = "wall_boundary_hit"; break
            if stdout_path.stat().st_size > maximum_output or stderr_path.stat().st_size > maximum_output:
                boundary = "captured_output_boundary_hit"; break
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
    stdout_bytes = stdout_path.read_bytes()
    stderr_bytes = stderr_path.read_bytes()
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
        "captured_output_mib_limit": int(limits["output_mib"]),
        "single_written_file_mib_limit": int(limits["maximum_single_written_file_mib"]),
    }


def finish(before: dict[str, Any], faults: list[str], receipts: dict[str, Any], *, executed: bool) -> dict[str, Any]:
    report = base.finish(before, faults, receipts, executed=executed)
    report["policy"] = POLICY
    report["state"] = "V2_TASK_3_DEPENDENCY_CACHE_ACQUIRED_AND_OFFLINE_REPLAY_QUALIFIED" if report["trigger_state"] == "GREEN" else "V2_TASK_3_DEPENDENCY_CANARY_FAILED"
    return report


if __name__ == "__main__":
    raise SystemExit(main())
