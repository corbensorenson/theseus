#!/usr/bin/env python3
"""Acquire and offline-replay only task 7's exact pnpm dependency lock."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa: E402
import theseus_vcm_dependency_prefetch_canary_v2 as bounded  # noqa: E402

POLICY = "project_theseus_vcm_task7_dependency_canary_v1"
STATE = "PROSPECTIVE_TASK_7_EXACT_PNPM_DEPENDENCY_ACQUISITION_AND_OFFLINE_REPLAY"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_task7_dependency_canary.json"


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
    if config.get("policy") != POLICY or config.get("state") != STATE:
        faults.append("policy_or_state_invalid")
    owner = p2a.resolve(str(config.get("owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != str(config.get("owner_sha256") or ""):
        faults.append("owner_binding_invalid")
    reports = {}
    for name, raw in p2a.mapping(config.get("reports")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"report_binding_invalid:{name}")
            reports[name] = {}
        else:
            reports[name] = p2a.read_json(path)
    schedule = p2a.dicts(reports.get("prefetch_plan", {}).get("schedule"))
    row = next((item for item in schedule if item.get("schedule_ordinal") == 2), {})
    compatibility_rows = p2a.dicts(reports.get("compatibility_v2", {}).get("rows"))
    compatibility = next((item for item in compatibility_rows if item.get("index") == 7), {})
    if row.get("index") != 7 or row.get("manager") != "pnpm" or row.get("governing_lock", {}).get("sha256") != "ca962f0d490977a3d1bb6b36d05a5f4c8305f984066fb3135c816b206cb54776":
        faults.append("schedule_binding_invalid")
    if compatibility.get("state") != "COMPATIBLE_DECLARED_REQUIREMENTS" or "node22_20_pnpm10_32_1" not in compatibility.get("compatible_profile_ids", []):
        faults.append("compatibility_binding_invalid")
    if reports.get("node_runtime", {}).get("trigger_state") != "GREEN" or reports.get("pnpm_runtime", {}).get("trigger_state") != "GREEN":
        faults.append("runtime_report_invalid")
    task = p2a.mapping(config.get("task"))
    if task.get("index") != 7 or task.get("repository") != "moshcoder/moshcode" or task.get("manager") != "pnpm":
        faults.append("task_binding_invalid")
    observed = {}
    for label, raw in p2a.mapping(config.get("archives")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"archive_binding_invalid:{label}")
            continue
        identities, archive_faults = base.archive_lock_identities(path, str(binding.get("archive_root") or ""), config)
        faults.extend(f"{label}:{fault}" for fault in archive_faults)
        observed[label] = identities
    if observed.get("parent") != observed.get("target"):
        faults.append("parent_target_dependency_identity_mismatch")
    for name, raw in p2a.mapping(config.get("tools")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"tool_binding_invalid:{name}")
    commands = p2a.mapping(config.get("commands"))
    online = p2a.strings(commands.get("online_acquisition_args"))
    offline = p2a.strings(commands.get("offline_replay_args"))
    if online != ["install", "--frozen-lockfile", "--ignore-scripts", "--reporter=silent"] or offline != [*online, "--offline"]:
        faults.append("command_contract_invalid")
    authority = p2a.mapping(config.get("authority"))
    allowed = {"temporary_normalized_archive_extraction_authorized", "single_network_dependency_acquisition_authorized", "untrusted_dependency_archive_extraction_authorized", "lifecycle_script_disabled_dependency_installation_authorized", "network_denied_offline_replay_authorized", "content_addressed_cache_retention_authorized"}
    for key, value in authority.items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")
    limits = p2a.mapping(config.get("limits"))
    for key in ("minimum_free_bytes_after_execution", "maximum_retained_bytes", "maximum_single_written_file_mib", "wall_seconds_per_command", "cpu_seconds_per_command", "output_mib", "open_files", "user_processes", "max_process_group_rss_mib", "terminate_grace_seconds"):
        if float(limits.get(key) or 0) <= 0:
            faults.append(f"limit_invalid:{key}")
    store = p2a.resolve(str(config.get("retained_store") or ""))
    if store.parent != (ROOT / "runtime/vcm_evaluator/dependency_store/pnpm").resolve() or store.name != "task-07":
        faults.append("retained_store_invalid")
    return {
        "policy": POLICY, "created_utc": p2a.now(),
        "trigger_state": "RED" if faults else "PAUSED",
        "state": "CONTRACT_INVALID" if faults else "READY_FOR_TASK_7_EXACT_PNPM_DEPENDENCY_CANARY",
        "faults": sorted(set(faults)), "config": base.identity(config_path),
        "observed_parent_target_dependency_identities": observed,
        "execution_performed": False, "network_enabled_dependency_installations": 0,
        "network_denied_dependency_installations": 0, "dependency_installations": 0,
        "repository_runner_executions": 0, "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0, "external_reference_calls": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def execute(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path)
    if before["trigger_state"] == "RED":
        return before
    store = p2a.resolve(str(config["retained_store"]))
    if store.exists():
        return finish(before, ["retained_store_already_exists"], {}, False)
    limits = p2a.mapping(config["limits"])
    free_before = shutil.disk_usage(ROOT).free
    if free_before < int(limits["minimum_free_bytes_after_execution"]):
        return finish(before, ["free_space_reserve_preflight_boundary_hit"], {"free_bytes_before": free_before}, False)
    faults: list[str] = []
    receipts: dict[str, Any] = {"free_bytes_before": free_before}
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-task7-deps-", dir="/private/tmp") as raw:
        work = Path(raw).resolve(); repo = work / "repository"; cache = work / "store"
        home = work / "home"; tmp = work / "tmp"
        cache.mkdir(); home.mkdir(); tmp.mkdir()
        target = p2a.mapping(p2a.mapping(config["archives"])["target"])
        receipts["repository_extraction"], extraction_faults = base.safe_extract_repository(p2a.resolve(str(target["path"])), repo, str(target["archive_root"]))
        faults.extend(extraction_faults)
        source_before = base.tree_identity(repo, excluded_roots={"node_modules"}) if not faults else {}
        env = {"HOME": str(home), "TMPDIR": str(tmp), "PATH": "/usr/bin:/bin", "CI": "1", "NO_COLOR": "1", "PNPM_HOME": str(home)}
        commands = p2a.mapping(config["commands"])
        online = pnpm_command(config, p2a.strings(commands["online_acquisition_args"]), cache)
        if not faults:
            receipts["online_acquisition"] = bounded.run_sandboxed(online, repo, work, env, config, network_denied=False)
            if base.command_failed(receipts["online_acquisition"]): faults.append("online_dependency_acquisition_failed")
        receipts["online_installed_dependencies"] = inspect_installed(repo, config) if not faults else {}
        if not faults: faults.extend(f"online:{fault}" for fault in receipts["online_installed_dependencies"]["faults"])
        if (repo / "node_modules").exists(): shutil.rmtree(repo / "node_modules")
        offline = pnpm_command(config, p2a.strings(commands["offline_replay_args"]), cache)
        if not faults:
            receipts["offline_replay"] = bounded.run_sandboxed(offline, repo, work, env, config, network_denied=True)
            if base.command_failed(receipts["offline_replay"]): faults.append("offline_dependency_replay_failed")
        receipts["offline_installed_dependencies"] = inspect_installed(repo, config) if not faults else {}
        if not faults: faults.extend(f"offline:{fault}" for fault in receipts["offline_installed_dependencies"]["faults"])
        source_after = base.tree_identity(repo, excluded_roots={"node_modules"}) if repo.exists() else {}
        receipts["repository_source_before"] = source_before; receipts["repository_source_after"] = source_after
        if source_before != source_after: faults.append("repository_source_mutated_outside_node_modules")
        receipts["store"] = store_identity(cache, config)
        retained_bytes = base.directory_bytes(cache)
        if retained_bytes > int(limits["maximum_retained_bytes"]): faults.append("retained_store_size_boundary_hit")
        if shutil.disk_usage(ROOT).free - retained_bytes < int(limits["minimum_free_bytes_after_execution"]): faults.append("free_space_reserve_postflight_boundary_hit")
        if not faults:
            store.parent.mkdir(parents=True, exist_ok=True); os.replace(cache, store)
    receipts["retained_store"] = store_identity(store, config) if store.exists() else {}
    receipts["free_bytes_after"] = shutil.disk_usage(ROOT).free
    return finish(before, faults, receipts, True)


def pnpm_command(config: dict[str, Any], args: list[str], store: Path) -> list[str]:
    tools = p2a.mapping(config["tools"])
    return [str(p2a.resolve(str(p2a.mapping(tools["node"])["path"]))), str(p2a.resolve(str(p2a.mapping(tools["pnpm_cli"])["path"]))), *args, "--store-dir", str(store)]


def inspect_installed(repo: Path, config: dict[str, Any]) -> dict[str, Any]:
    faults = []; observed = {}
    for name, version in p2a.mapping(p2a.mapping(config["task"])["expected_dependencies"]).items():
        path = repo / "node_modules" / Path(*name.split("/")) / "package.json"
        try: metadata = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError): metadata = {}; faults.append(f"installed_package_absent_or_invalid:{name}")
        observed[name] = {"version": metadata.get("version"), "package_json_sha256": p2a.sha256_file(path) if path.is_file() else ""}
        if metadata.get("version") != version: faults.append(f"installed_package_version_mismatch:{name}")
    return {"observed": observed, "faults": sorted(set(faults))}


def store_identity(store: Path, config: dict[str, Any]) -> dict[str, Any]:
    rows = [{"path": p.relative_to(store).as_posix(), "bytes": p.stat().st_size, "sha256": p2a.sha256_file(p)} for p in sorted(store.rglob("*")) if p.is_file() and not p.is_symlink()] if store.exists() else []
    task = p2a.mapping(config["task"])
    return {"path": p2a.rel(store), "file_count": len(rows), "bytes": sum(r["bytes"] for r in rows), "files_identity_sha256": base.digest_json(rows), "bound_lock_integrity_sha512_hex": task.get("expected_lock_integrity_sha512_hex")}


def finish(before: dict[str, Any], faults: list[str], receipts: dict[str, Any], executed: bool) -> dict[str, Any]:
    online = int("online_acquisition" in receipts); offline = int("offline_replay" in receipts)
    return {**before, "created_utc": p2a.now(), "trigger_state": "GREEN" if executed and not faults else "RED", "state": "TASK_7_EXACT_PNPM_DEPENDENCY_CACHE_ACQUIRED_AND_OFFLINE_REPLAY_QUALIFIED" if executed and not faults else "TASK_7_DEPENDENCY_CANARY_FAILED", "faults": sorted(set(faults)), "execution_performed": executed, "network_enabled_dependency_installations": online, "network_denied_dependency_installations": offline, "dependency_installations": online + offline, "receipts": receipts}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in ("trigger_state", "state", "execution_performed", "network_enabled_dependency_installations", "network_denied_dependency_installations", "dependency_installations", "repository_runner_executions", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls", "faults")}


if __name__ == "__main__":
    raise SystemExit(main())
