#!/usr/bin/env python3
"""Materialize and offline-replay the six immutable evaluator environments."""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_immutable_resolution_segment as resolver  # noqa: E402
import theseus_vcm_six_row_environment_preflight as fit_owner  # noqa: E402

POLICY = "project_theseus_vcm_six_row_environment_materializer_v1"
STATE = "PROSPECTIVE_K2_05_SIX_ROW_SHARED_STORE_DISPOSABLE_ENVIRONMENT_V2_RESUME"
DEFAULT_CONFIG = ROOT / "configs/theseus_vcm_six_row_environment_materializer.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); path = p2a.resolve(args.config); cfg = p2a.read_json(path)
    report = execute(path) if args.execute else preflight_report(path)
    p2a.write_json(p2a.resolve(args.out or cfg["report"]), report)
    print(json.dumps({k: report.get(k) for k in ("trigger_state","state","faults","execution_performed","task_count","qualified_task_count","inconclusive_task_count","network_enabled_materializations","network_denied_replays","package_installations","candidate_or_control_calls","external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    cfg = p2a.read_json(path); faults: list[str] = []
    if cfg.get("policy") != POLICY or cfg.get("state") != STATE: faults.append("policy_or_state_invalid")
    for key, expected in (("owner", Path(__file__).resolve()), ("audit_owner", ROOT / "scripts/theseus_vcm_six_row_environment_materializer_audit.py")):
        owner = p2a.resolve(str(cfg.get(key) or ""))
        if owner != expected.resolve() or not owner.is_file() or p2a.sha256_file(owner) != cfg.get(f"{key}_sha256"): faults.append(f"{key}_binding_invalid")
    sources: dict[str, dict[str, Any]] = {}
    for name, raw in p2a.mapping(cfg.get("sources")).items():
        binding = p2a.mapping(raw); source = p2a.resolve(str(binding.get("path") or ""))
        if not source.is_file() or p2a.sha256_file(source) != binding.get("sha256"): faults.append(f"source_binding_invalid:{name}"); sources[name] = {}
        else: sources[name] = p2a.read_json(source)
    if sources.get("fit", {}).get("execution_ready") is not True or sources.get("fit_audit", {}).get("trigger_state") != "GREEN": faults.append("fit_predecessor_invalid")
    if sources.get("resolution", {}).get("qualified_task_count") != 6 or sources.get("resolution_audit", {}).get("trigger_state") != "GREEN": faults.append("resolution_predecessor_invalid")
    predecessor = sources.get("materializer_v1", {}); predecessor_audit = sources.get("materializer_audit_v1", {})
    if predecessor.get("trigger_state") != "GREEN" or predecessor.get("qualified_task_count") != 4 or predecessor.get("inconclusive_task_count") != 2 or predecessor_audit.get("trigger_state") != "GREEN": faults.append("materializer_predecessor_invalid")
    resolver_cfg, resolver_bound, resolver_faults = resolver.preflight(ROOT / "configs/theseus_vcm_immutable_resolution_segment.json")
    faults.extend(f"resolver:{fault}" for fault in resolver_faults)
    tools: dict[str, str] = {}
    for name, raw in p2a.mapping(cfg.get("tools")).items():
        binding = p2a.mapping(raw); tool = p2a.resolve(str(binding.get("path") or ""))
        if not tool.is_file() or p2a.sha256_file(tool) != binding.get("sha256"): faults.append(f"tool_binding_invalid:{name}")
        tools[name] = str(tool)
    report_rows = {int(row.get("index") or 0): row for row in p2a.dicts(sources.get("resolution", {}).get("rows"))}
    bound_rows: dict[int, dict[str, Any]] = {}
    rows = p2a.dicts(cfg.get("rows"))
    if [row.get("index") for row in rows] != resolver.EXPECTED_INDICES: faults.append("row_denominator_invalid")
    for row in rows:
        index = int(row.get("index") or 0); lock = p2a.resolve(str(row.get("lock") or "")); receipt = p2a.mapping(p2a.mapping(report_rows.get(index, {}).get("receipt")).get("lock"))
        if not lock.is_file() or p2a.sha256_file(lock) != row.get("sha256") or receipt.get("sha256") != row.get("sha256") or receipt.get("package_count") != row.get("package_count"): faults.append(f"lock_binding_invalid:{index}")
        original = p2a.mapping(p2a.mapping(resolver_bound.get("rows")).get(index))
        if row.get("manager") != p2a.mapping(original.get("row")).get("manager"): faults.append(f"manager_binding_invalid:{index}")
        if row.get("manager") == "uv" and row.get("python_tool") not in tools: faults.append(f"python_tool_invalid:{index}")
        bound_rows[index] = {"config": row, "resolver": original, "lock": lock}
    allowed = {"serialized_network_dependency_materialization_authorized","network_denied_offline_replay_authorized","wheel_only_python_installation_authorized","cargo_fetch_without_build_authorized","shared_store_retention_authorized","disposable_environment_authorized"}
    for key, value in p2a.mapping(cfg.get("authority")).items():
        if value is not (key in allowed): faults.append(f"authority_invalid:{key}")
    store = p2a.resolve(str(cfg.get("store") or ""))
    if store != (ROOT / "runtime/vcm_evaluator/dependency_store/six-row-environments-v1").resolve(): faults.append("store_binding_invalid")
    if not store.is_dir() or tree_identity(store) != predecessor.get("retained_shared_store"): faults.append("predecessor_store_binding_invalid")
    return cfg, {"sources": sources, "tools": tools, "rows": bound_rows, "resolver_cfg": resolver_cfg, "store": store}, sorted(set(faults))


def preflight_report(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, _, faults = preflight(path); return finish(cfg, path, faults, [], False, {})


def execute(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, bound, faults = preflight(path); store: Path = bound.get("store", Path("/invalid"))
    if faults: return finish(cfg, path, faults, [], False, {})
    fit = fit_owner.evaluate(ROOT / "configs/theseus_vcm_six_row_environment_preflight.json")
    if fit.get("execution_ready") is not True: return finish(cfg, path, ["current_fit_boundary_closed"], [], False, {})
    limits = p2a.mapping(cfg["limits"]); rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-six-env-", dir="/private/tmp") as raw:
        batch = Path(raw).resolve(); cache = store; previous = {int(row.get("index") or 0): row for row in p2a.dicts(bound["sources"]["materializer_v1"].get("rows"))}
        for index in resolver.EXPECTED_INDICES:
            item = bound["rows"][index]; row = p2a.mapping(item["config"])
            if previous.get(index, {}).get("disposition") == "ENVIRONMENT_MATERIALIZATION_QUALIFIED":
                rows.append({"index":index,"repository":row["repository"],"manager":row["manager"],"lock":{"path":p2a.rel(item["lock"]),"sha256":row["sha256"],"package_count":row["package_count"]},"receipts":{"predecessor_reuse":True,"predecessor_report_sha256":p2a.mapping(cfg["sources"])["materializer_v1"]["sha256"]},"faults":[],"disposition":"ENVIRONMENT_MATERIALIZATION_QUALIFIED_REUSED_FROM_SEALED_PREDECESSOR"})
                continue
            rows.append(materialize_row(cfg, bound, row, item, batch, cache))
            if shutil.disk_usage(ROOT).free < int(limits["minimum_free_bytes_after_execution"]): faults.append("free_space_reserve_postflight_boundary_hit"); break
            if tree_identity(cache)["bytes"] > int(limits["maximum_shared_store_bytes"]): faults.append("shared_store_size_boundary_hit"); break
    if len(rows) != 6: faults.append("row_denominator_not_completed")
    return finish(cfg, path, faults, rows, True, tree_identity(store))


def materialize_row(cfg: dict[str, Any], bound: dict[str, Any], row: dict[str, Any], item: dict[str, Any], batch: Path, cache: Path) -> dict[str, Any]:
    index = int(row["index"]); work = batch / f"task-{index:02d}"; work.mkdir(); receipts: dict[str, Any] = {}; faults: list[str] = []
    env = {"HOME": str(work / "home"), "TMPDIR": str(work / "tmp"), "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "CI": "1", "NO_COLOR": "1", "UV_NO_CONFIG": "1", "UV_PYTHON_DOWNLOADS": "never"}
    Path(env["HOME"]).mkdir(); Path(env["TMPDIR"]).mkdir()
    if row["manager"] == "uv":
        venv = work / "venv"; python = bound["tools"][row["python_tool"]]
        venv_cmd = [bound["tools"]["uv"], "venv", str(venv), "--python", python, "--no-python-downloads", "--no-config"]
        receipts["online_venv"] = run_sandboxed(venv_cmd, work, batch, env, cfg, True)
        sync = [bound["tools"]["uv"], "pip", "sync", str(item["lock"]), "--python", str(venv / "bin/python"), "--require-hashes", "--no-build", "--index-strategy", "first-index", "--default-index", "https://pypi.org/simple", "--cache-dir", str(cache / "uv"), "--link-mode", "copy", "--no-config", "--no-progress", "--color", "never"]
        if row.get("find_links"): sync.extend(["--find-links", str(p2a.resolve(str(row["find_links"])))])
        if not failed(receipts["online_venv"]): receipts["online_sync"] = run_sandboxed(sync, work, batch, env, cfg, False)
        if failed(receipts.get("online_sync", {})): faults.append("online_wheel_only_sync_failed")
        if not faults: receipts["online_environment"] = inspect_python_environment(venv, item["lock"]); faults.extend(f"online:{v}" for v in receipts["online_environment"]["faults"])
        if venv.exists(): shutil.rmtree(venv)
        if not faults:
            receipts["offline_venv"] = run_sandboxed(venv_cmd, work, batch, env, cfg, True)
            offline = [*sync, "--offline"]; receipts["offline_sync"] = run_sandboxed(offline, work, batch, env, cfg, True)
            if failed(receipts["offline_venv"]) or failed(receipts["offline_sync"]): faults.append("network_denied_replay_failed")
        if not faults: receipts["offline_environment"] = inspect_python_environment(venv, item["lock"]); faults.extend(f"offline:{v}" for v in receipts["offline_environment"]["faults"])
        if not faults and receipts["online_environment"] != receipts["offline_environment"]: faults.append("online_offline_environment_mismatch")
        if venv.exists(): shutil.rmtree(venv)
    else:
        source = p2a.mapping(item["resolver"]); repository, extract_faults = resolver.extract_regular_archive(source["archive"], work / "repository")
        faults.extend(extract_faults); sealed_lock = repository / "Cargo.lock"
        if not faults: shutil.copyfile(item["lock"], sealed_lock)
        env["CARGO_HOME"] = str(cache / "cargo"); env["CARGO_NET_GIT_FETCH_WITH_CLI"] = "false"; env["CARGO_REGISTRIES_CRATES_IO_PROTOCOL"] = "sparse"; env["PATH"] = f'{Path(bound["tools"]["cargo"]).parent}:/usr/bin:/bin'
        command = [bound["tools"]["cargo"], "fetch", "--locked", "--manifest-path", str(repository / "Cargo.toml"), "--config", "net.git-fetch-with-cli=false"]
        if not faults: receipts["online_fetch"] = run_sandboxed(command, repository, batch, env, cfg, False)
        if failed(receipts.get("online_fetch", {})): faults.append("online_cargo_fetch_failed")
        if not faults: receipts["offline_fetch"] = run_sandboxed([*command, "--offline"], repository, batch, env, cfg, True)
        if failed(receipts.get("offline_fetch", {})): faults.append("network_denied_cargo_replay_failed")
        if sealed_lock.is_file() and p2a.sha256_file(sealed_lock) != row["sha256"]: faults.append("cargo_lock_mutated")
    disposition = "ENVIRONMENT_MATERIALIZATION_QUALIFIED" if not faults else ("INCONCLUSIVE_EXPERIMENT_HOST_RESOURCE_BOUNDARY" if any(p2a.mapping(v).get("boundary_hit") for v in receipts.values()) else "INCONCLUSIVE_IMPLEMENTATION_ENVIRONMENT_MATERIALIZATION")
    return {"index": index, "repository": row["repository"], "manager": row["manager"], "lock": {"path": p2a.rel(item["lock"]), "sha256": row["sha256"], "package_count": row["package_count"]}, "receipts": receipts, "faults": sorted(set(faults)), "disposition": disposition}


def run_sandboxed(command: list[str], cwd: Path, batch: Path, env: dict[str, str], cfg: dict[str, Any], network_denied: bool) -> dict[str, Any]:
    cache_root = p2a.resolve(str(cfg["store"])); profile = "\n".join(["(version 1)", "(allow default)", *( ["(deny network*)"] if network_denied else []), f'(deny file-write* (require-not (subpath "{batch}")))', f'(allow file-write* (subpath "{cache_root}"))', '(allow file-write* (literal "/dev/null"))'])
    full = [str(p2a.resolve(str(p2a.mapping(cfg["tools"])["sandbox_exec"]["path"]))), "-p", profile, *command]; started = time.monotonic(); boundary = ""
    try: done = subprocess.run(full, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=int(p2a.mapping(cfg["limits"])["wall_seconds_per_command"]), check=False)
    except subprocess.TimeoutExpired as exc: done = subprocess.CompletedProcess(full, 124, exc.stdout or b"", exc.stderr or b""); boundary = "wall_timeout"
    stdout = done.stdout or b""; stderr = done.stderr or b""
    return {"command": full, "returncode": done.returncode, "duration_ms": round((time.monotonic()-started)*1000,3), "network_denied": network_denied, "boundary_hit": bool(boundary), "boundary_reason": boundary, "stdout": stdout.decode("utf-8","replace"), "stderr": stderr.decode("utf-8","replace"), "stdout_bytes": len(stdout), "stderr_bytes": len(stderr), "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest(), "stdout_complete": True, "stderr_complete": True, "project_selected_output_cap": None}


def inspect_python_environment(venv: Path, lock: Path) -> dict[str, Any]:
    expected = {(normalize_name(name), version) for name, version in re.findall(r"^([A-Za-z0-9_.-]+)==([^\\\s]+)", lock.read_text(), re.MULTILINE)}; rows=[]; faults=[]; parser=email.parser.Parser()
    sites=list(venv.glob("lib/python*/site-packages"))
    if len(sites)!=1: return {"faults":["site_packages_denominator_invalid"],"distributions":[]}
    for metadata in sites[0].glob("*.dist-info/METADATA"):
        parsed=parser.parsestr(metadata.read_text(errors="replace")); row={"name":normalize_name(str(parsed.get("Name") or "")),"version":str(parsed.get("Version") or "")}; rows.append(row)
        if (row["name"],row["version"]) not in expected:faults.append(f"installed_distribution_not_locked:{row['name']}@{row['version']}")
    return {"faults":sorted(set(faults)),"distributions":sorted(rows,key=lambda v:(v["name"],v["version"]))}


def tree_identity(root: Path) -> dict[str, Any]:
    rows=[]; total=0
    if root.exists():
        for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink() and p.name != ".DS_Store"): size=path.stat().st_size; total+=size; rows.append({"path":path.relative_to(root).as_posix(),"bytes":size,"sha256":p2a.sha256_file(path)})
    return {"path":p2a.rel(root),"file_count":len(rows),"bytes":total,"identity_sha256":hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


def failed(receipt: dict[str, Any]) -> bool: return not receipt or receipt.get("returncode") != 0 or receipt.get("boundary_hit") is True


def normalize_name(value: str) -> str: return re.sub(r"[-_.]+", "-", value).lower()


def finish(cfg: dict[str, Any], path: Path, faults: list[str], rows: list[dict[str, Any]], executed: bool, store: dict[str, Any]) -> dict[str, Any]:
    qualified=sum(str(row.get("disposition") or "").startswith("ENVIRONMENT_MATERIALIZATION_QUALIFIED") for row in rows); online=sum("online_sync" in p2a.mapping(r.get("receipts")) or "online_fetch" in p2a.mapping(r.get("receipts")) for r in rows); offline=sum("offline_sync" in p2a.mapping(r.get("receipts")) or "offline_fetch" in p2a.mapping(r.get("receipts")) for r in rows)
    return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"RED" if faults else ("GREEN" if executed else "PAUSED"),"state":"K2_05_SIX_ROW_ENVIRONMENTS_MATERIALIZED_WITH_SCOPED_DISPOSITIONS" if executed and not faults else ("READY_FOR_SIX_ROW_ENVIRONMENT_MATERIALIZATION" if not faults else "K2_05_SIX_ROW_ENVIRONMENT_MATERIALIZATION_INVALID"),"faults":sorted(set(faults)),"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"execution_performed":executed,"task_count":len(rows),"qualified_task_count":qualified,"inconclusive_task_count":len(rows)-qualified,"rows":rows,"retained_shared_store":store,"network_enabled_materializations":online,"network_denied_replays":offline,"package_installations":online+offline,"source_build_executions":0,"project_installations":0,"repository_runner_executions":0,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"teacher_calls":0,"panel_admitted":False,"partial_panel_admission_forbidden":True,"maximum_inference":cfg.get("maximum_inference")}


if __name__ == "__main__": raise SystemExit(main())
