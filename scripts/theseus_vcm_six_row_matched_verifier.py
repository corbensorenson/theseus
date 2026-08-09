#!/usr/bin/env python3
"""Execute the prospectively sealed six-row common-evaluator panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_six_row_matched_verifier_v1"
STATE = "PROSPECTIVE_K2_05_SIX_ROW_MATCHED_PARENT_TARGET_VERIFIERS_V1"
DEFAULT_CONFIG = ROOT / "configs/theseus_vcm_six_row_matched_verifier.json"
EXPECTED_INDICES = [12, 13, 16, 25, 35, 56]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report = execute(path) if args.execute else preflight_report(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg.get("report") or "")), report)
    print(json.dumps({key: report.get(key) for key in (
        "trigger_state", "state", "faults", "execution_performed", "task_count",
        "qualified_task_count", "inconclusive_task_count", "parent_target_or_evaluator_executions",
        "package_installations", "candidate_or_control_calls", "external_reference_calls",
    )}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def preflight(path: Path = DEFAULT_CONFIG, *, verify_store: bool = True) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != POLICY or cfg.get("state") != STATE:
        faults.append("policy_or_state_invalid")
    for key, expected in (
        ("owner", Path(__file__).resolve()),
        ("audit_owner", ROOT / "scripts/theseus_vcm_six_row_matched_verifier_audit.py"),
    ):
        owner = p2a.resolve(str(cfg.get(key) or ""))
        if owner != expected.resolve() or not owner.is_file() or p2a.sha256_file(owner) != cfg.get(f"{key}_sha256"):
            faults.append(f"{key}_binding_invalid")

    sources: dict[str, dict[str, Any]] = {}
    for name, raw in p2a.mapping(cfg.get("sources")).items():
        binding = p2a.mapping(raw)
        source = p2a.resolve(str(binding.get("path") or ""))
        if not source.is_file() or p2a.sha256_file(source) != binding.get("sha256"):
            faults.append(f"source_binding_invalid:{name}")
            sources[name] = {}
        else:
            sources[name] = p2a.read_json(source)
    environment = sources.get("environment", {})
    environment_audit = sources.get("environment_audit", {})
    if environment.get("trigger_state") != "GREEN" or environment.get("qualified_task_count") != 6:
        faults.append("environment_predecessor_invalid")
    if environment_audit.get("trigger_state") != "GREEN" or environment_audit.get("qualified_task_count") != 6:
        faults.append("environment_audit_predecessor_invalid")

    tools: dict[str, Path] = {}
    for name, raw in p2a.mapping(cfg.get("tools")).items():
        binding = p2a.mapping(raw)
        tool = p2a.resolve(str(binding.get("path") or ""))
        if not tool.is_file() or p2a.sha256_file(tool) != binding.get("sha256"):
            faults.append(f"tool_binding_invalid:{name}")
        tools[name] = tool

    store = p2a.resolve(str(cfg.get("store") or ""))
    expected_store = p2a.mapping(environment.get("retained_shared_store"))
    if not store.is_dir() or p2a.rel(store) != expected_store.get("path"):
        faults.append("store_path_invalid")
    elif verify_store and tree_identity(store) != expected_store:
        faults.append("store_identity_invalid")

    rows = p2a.dicts(cfg.get("rows"))
    indices = [int(row.get("index") or 0) for row in rows]
    if indices != EXPECTED_INDICES or int(cfg.get("expected_task_count") or 0) != len(EXPECTED_INDICES):
        faults.append("row_denominator_invalid")
    bound_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row.get("index") or 0)
        manager = str(row.get("manager") or "")
        if manager not in {"uv", "cargo"}:
            faults.append(f"manager_invalid:{index}")
        if manager == "uv" and str(row.get("python_tool") or "") not in tools:
            faults.append(f"python_tool_invalid:{index}")
        lock = p2a.resolve(str(row.get("lock") or ""))
        if not lock.is_file() or p2a.sha256_file(lock) != row.get("lock_sha256"):
            faults.append(f"lock_binding_invalid:{index}")
        archives: dict[str, Path] = {}
        for side in ("parent", "target"):
            archive_binding = p2a.mapping(row.get(f"{side}_archive"))
            archive = p2a.resolve(str(archive_binding.get("path") or ""))
            if not archive.is_file() or p2a.sha256_file(archive) != archive_binding.get("sha256"):
                faults.append(f"archive_binding_invalid:{index}:{side}")
            elif archive_root(archive) != archive_binding.get("root"):
                faults.append(f"archive_root_invalid:{index}:{side}")
            archives[side] = archive
        observed_changes = archive_changes(archives.get("parent"), archives.get("target")) if all(path.is_file() for path in archives.values()) else []
        declared_changes = sorted(p2a.strings(row.get("target_changed_paths")))
        evaluators = sorted(p2a.strings(row.get("common_evaluator_paths")))
        if observed_changes != declared_changes:
            faults.append(f"target_change_partition_invalid:{index}")
        if not evaluators or not set(evaluators).issubset(declared_changes):
            faults.append(f"common_evaluator_partition_invalid:{index}")
        if set(evaluators) & set(p2a.strings(row.get("forbidden_transplant_paths"))):
            faults.append(f"forbidden_transplant_overlap:{index}")
        evaluator_hashes = p2a.mapping(row.get("common_evaluator_sha256"))
        for evaluator in evaluators:
            payload = archive_member(archives.get("target"), evaluator)
            if payload is None or hashlib.sha256(payload).hexdigest() != evaluator_hashes.get(evaluator):
                faults.append(f"common_evaluator_binding_invalid:{index}:{evaluator}")
        support_hashes = p2a.mapping(row.get("unchanged_harness_sha256"))
        for support, expected_sha in support_hashes.items():
            parent_payload = archive_member(archives.get("parent"), support)
            target_payload = archive_member(archives.get("target"), support)
            if parent_payload is None or parent_payload != target_payload or hashlib.sha256(parent_payload).hexdigest() != expected_sha:
                faults.append(f"unchanged_harness_binding_invalid:{index}:{support}")
        bound_rows[index] = {"config": row, "lock": lock, "archives": archives}

    authority = p2a.mapping(cfg.get("authority"))
    expected_true = {
        "offline_exact_lock_replay_authorized", "hidden_target_evaluator_transplant_authorized",
        "serial_parent_target_verifier_execution_authorized", "disposable_copy_on_write_cache_clone_authorized",
        "untrusted_rust_compile_authorized", "complete_diagnostics_authorized",
    }
    for key, value in authority.items():
        if value is not (key in expected_true):
            faults.append(f"authority_invalid:{key}")
    return cfg, {"sources": sources, "tools": tools, "store": store, "rows": bound_rows}, sorted(set(faults))


def preflight_report(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, bound, faults = preflight(path, verify_store=False)
    store = tree_identity(bound["store"]) if not faults and bound.get("store") else {}
    if not faults and store != p2a.mapping(bound["sources"]["environment"].get("retained_shared_store")):
        faults.append("store_identity_invalid")
    return finish(cfg, path, faults, [], False, store, store, 0, 0)


def execute(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, bound, faults = preflight(path, verify_store=False)
    store_before = tree_identity(bound["store"]) if not faults else {}
    if not faults and store_before != p2a.mapping(bound["sources"]["environment"].get("retained_shared_store")):
        faults.append("store_identity_invalid")
    if faults:
        return finish(cfg, path, faults, [], False, store_before, store_before, 0, 0)
    limits = p2a.mapping(cfg.get("limits"))
    reserve = int(limits.get("minimum_free_bytes_after_execution") or 0)
    peak = int(limits.get("required_incremental_peak_bytes") or 0)
    if shutil.disk_usage(ROOT).free - peak < reserve:
        return finish(cfg, path, ["current_fit_boundary_closed"], [], False, store_before, store_before, 0, 0)

    rows: list[dict[str, Any]] = []
    executions = 0
    package_installs = 0
    cache_clone_receipts: dict[str, dict[str, Any]] = {}
    resource_boundary_closed = False
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-six-verifier-", dir="/private/tmp") as raw:
        batch = Path(raw).resolve()
        cache_clones: dict[str, Path] = {}
        for manager in ("uv", "cargo"):
            source = bound["store"] / manager
            destination = batch / "cache" / manager
            destination.parent.mkdir(parents=True, exist_ok=True)
            cache_clone_receipts[manager] = clone_tree(bound["tools"]["cp"], source, destination)
            if cache_clone_receipts[manager].get("returncode") != 0:
                faults.append(f"disposable_cache_clone_failed:{manager}")
            cache_clones[manager] = destination
        if not faults:
            for index in EXPECTED_INDICES:
                if resource_boundary_closed or shutil.disk_usage(ROOT).free < reserve:
                    rows.append(not_executed_row(bound["rows"][index]["config"], "INCONCLUSIVE_EXPERIMENT_HOST_RESOURCE_BOUNDARY"))
                    resource_boundary_closed = True
                    continue
                row, row_executions, row_installs = execute_row(cfg, bound, bound["rows"][index], batch, cache_clones)
                rows.append(row)
                executions += row_executions
                package_installs += row_installs
                if shutil.disk_usage(ROOT).free < reserve:
                    resource_boundary_closed = True
        if len(rows) < len(EXPECTED_INDICES):
            completed = {int(row.get("index") or 0) for row in rows}
            for index in EXPECTED_INDICES:
                if index not in completed:
                    rows.append(not_executed_row(bound["rows"][index]["config"], "INCONCLUSIVE_EXPERIMENT_HOST_RESOURCE_BOUNDARY"))
    store_after = tree_identity(bound["store"])
    if store_after != store_before:
        faults.append("retained_store_mutated")
    report = finish(cfg, path, faults, rows, True, store_before, store_after, executions, package_installs)
    report["cache_clone_receipts"] = cache_clone_receipts
    report["host_resource_boundary_closed"] = resource_boundary_closed
    return report


def execute_row(cfg: dict[str, Any], bound: dict[str, Any], item: dict[str, Any], batch: Path, caches: dict[str, Path]) -> tuple[dict[str, Any], int, int]:
    row = p2a.mapping(item["config"])
    index = int(row["index"])
    task = batch / f"task-{index:02d}"
    task.mkdir()
    roots: dict[str, Path] = {}
    works: dict[str, Path] = {}
    row_faults: list[str] = []
    for side in ("parent", "target"):
        works[side] = task / side
        roots[side], extract_faults = extract_regular_archive(item["archives"][side], works[side] / "source")
        row_faults.extend(f"{side}:{fault}" for fault in extract_faults)
    evaluator_receipts: list[dict[str, Any]] = []
    for evaluator in p2a.strings(row.get("common_evaluator_paths")):
        payload = archive_member(item["archives"]["target"], evaluator)
        if payload is None:
            row_faults.append(f"target_evaluator_missing:{evaluator}")
            continue
        expected_sha = p2a.mapping(row.get("common_evaluator_sha256")).get(evaluator)
        for side in ("parent", "target"):
            destination = roots[side] / evaluator
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            destination.chmod(0o644)
            evaluator_receipts.append({"side": side, "path": evaluator, "sha256": p2a.sha256_file(destination), "expected_sha256": expected_sha})
    environment: dict[str, Any] = {}
    sides: dict[str, dict[str, Any]] = {}
    executions = 0
    package_installs = 0
    if not row_faults and row["manager"] == "uv":
        environment, environment_faults = create_python_environment(cfg, bound, item, task, caches["uv"])
        row_faults.extend(environment_faults)
        package_installs = 1 if environment.get("sync") else 0
        if not row_faults:
            python = Path(str(environment["venv"])) / "bin" / "python"
            for side in ("parent", "target"):
                receipt = run_verifier(cfg, bound, row, side, roots[side], works[side], python, caches["uv"])
                sides[side] = receipt
                executions += 1
    elif not row_faults:
        environment, environment_faults = prepare_cargo_environment(item, roots, caches["cargo"])
        row_faults.extend(environment_faults)
        if not row_faults:
            for side in ("parent", "target"):
                receipt = run_verifier(cfg, bound, row, side, roots[side], works[side], bound["tools"]["cargo"], caches["cargo"])
                sides[side] = receipt
                executions += 1
                target_dir = works[side] / "cargo-target"
                if target_dir.exists():
                    shutil.rmtree(target_dir)
    disposition = derive_disposition(sides, row_faults)
    result = {
        "index": index,
        "repository": row.get("repository"),
        "manager": row.get("manager"),
        "lock": {"path": p2a.rel(item["lock"]), "sha256": p2a.sha256_file(item["lock"])},
        "archives": {side: {"path": p2a.rel(item["archives"][side]), "sha256": p2a.sha256_file(item["archives"][side])} for side in ("parent", "target")},
        "common_evaluator_paths": p2a.strings(row.get("common_evaluator_paths")),
        "common_evaluator_receipts": evaluator_receipts,
        "target_production_or_nontransplanted_path_count": len(p2a.strings(row.get("target_changed_paths"))) - len(p2a.strings(row.get("common_evaluator_paths"))),
        "forbidden_transplant_paths": p2a.strings(row.get("forbidden_transplant_paths")),
        "environment": environment,
        "parent": sides.get("parent", {}),
        "target": sides.get("target", {}),
        "faults": sorted(set(row_faults)),
        "disposition": disposition,
    }
    return result, executions, package_installs


def create_python_environment(cfg: dict[str, Any], bound: dict[str, Any], item: dict[str, Any], task: Path, cache: Path) -> tuple[dict[str, Any], list[str]]:
    row = p2a.mapping(item["config"])
    venv = task / "venv"
    env_root = task / "environment"
    env_root.mkdir()
    env = base_env(env_root)
    env["UV_CACHE_DIR"] = str(cache)
    uv = bound["tools"]["uv"]
    python = bound["tools"][str(row["python_tool"])]
    venv_command = [str(uv), "venv", str(venv), "--python", str(python), "--no-python-downloads", "--no-config"]
    venv_receipt = run_sandboxed(venv_command, task, task, [task, cache], env, cfg, "environment")
    sync_command = [str(uv), "pip", "sync", str(item["lock"]), "--python", str(venv / "bin/python"), "--require-hashes", "--no-build", "--offline", "--cache-dir", str(cache), "--link-mode", "copy", "--no-config", "--no-progress", "--color", "never"]
    if row.get("find_links"):
        sync_command.extend(["--find-links", str(p2a.resolve(str(row["find_links"])))])
    sync_receipt = run_sandboxed(sync_command, task, task, [task, cache], env, cfg, "environment") if receipt_ok(venv_receipt) else {}
    faults: list[str] = []
    if not receipt_ok(venv_receipt):
        faults.append("offline_venv_creation_failed")
    if not receipt_ok(sync_receipt):
        faults.append("offline_exact_lock_sync_failed")
    return {"venv": str(venv), "venv_command": venv_receipt, "sync": sync_receipt, "network_denied": True, "project_installation": False}, faults


def prepare_cargo_environment(item: dict[str, Any], roots: dict[str, Path], cache: Path) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    receipts: dict[str, Any] = {"cargo_home": str(cache), "network_denied": True, "project_installation": False}
    for side, root in roots.items():
        destination = root / "Cargo.lock"
        shutil.copyfile(item["lock"], destination)
        observed = p2a.sha256_file(destination)
        receipts[f"{side}_lock_sha256"] = observed
        if observed != p2a.mapping(item["config"]).get("lock_sha256"):
            faults.append(f"sealed_cargo_lock_mismatch:{side}")
    return receipts, faults


def run_verifier(cfg: dict[str, Any], bound: dict[str, Any], row: dict[str, Any], side: str, root: Path, work: Path, executable: Path, cache: Path) -> dict[str, Any]:
    cwd = root / str(row.get("working_directory") or ".")
    env = base_env(work)
    if row["manager"] == "uv":
        venv_bin = executable.parent
        env["PATH"] = f"{venv_bin}:/usr/bin:/bin"
        env["PYTHONPATH"] = os.pathsep.join([str(cwd), str(root)])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["QT_QPA_PLATFORM"] = "offscreen"
        command = [str(executable), *p2a.strings(row.get("arguments"))]
        writable = [work]
    else:
        env["PATH"] = f"{bound['tools']['cargo'].parent}:/usr/bin:/bin"
        env["CARGO_HOME"] = str(cache)
        env["CARGO_NET_OFFLINE"] = "true"
        env["CARGO_NET_GIT_FETCH_WITH_CLI"] = "false"
        env["RUSTC"] = str(bound["tools"]["rustc"])
        env["CARGO_TARGET_DIR"] = str(work / "cargo-target")
        command = [str(executable), *p2a.strings(row.get("arguments"))]
        writable = [work, cache]
    receipt = run_sandboxed(command, cwd, work, writable, env, cfg, "cargo" if row["manager"] == "cargo" else "python")
    receipt.update({
        "side": side,
        "manager": row["manager"],
        "declared_arguments": p2a.strings(row.get("arguments")),
        "working_directory": str(row.get("working_directory") or "."),
        "common_evaluator_paths": p2a.strings(row.get("common_evaluator_paths")),
    })
    return receipt


def run_sandboxed(command: list[str], cwd: Path, work: Path, writable: list[Path], env: dict[str, str], cfg: dict[str, Any], limit_class: str) -> dict[str, Any]:
    sandbox = p2a.resolve(str(p2a.mapping(cfg.get("tools")).get("sandbox_exec", {}).get("path") or ""))
    profile_lines = ["(version 1)", "(allow default)", "(deny network*)", "(deny mach-lookup)", "(deny file-write*)", '(allow file-write* (literal "/dev/null"))']
    for path in writable:
        profile_lines.append(f'(allow file-write* (subpath "{path.resolve()}"))')
    profile = "\n".join(profile_lines)
    full = [str(sandbox), "-p", profile, *command]
    stdout_path = work / f"stdout-{time.monotonic_ns()}.bin"
    stderr_path = work / f"stderr-{time.monotonic_ns()}.bin"
    timeout = int(p2a.mapping(cfg.get("limits")).get(f"{limit_class}_wall_seconds") or 1200)
    started = time.monotonic()
    boundary = ""
    work.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(full, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
            boundary = "host_safety_wall_timeout"
    stdout_bytes = stdout_path.read_bytes()
    stderr_bytes = stderr_path.read_bytes()
    return {
        "command": full,
        "returncode": returncode,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "network_denied": True,
        "boundary_hit": bool(boundary),
        "boundary_reason": boundary,
        "stdout": stdout_bytes.decode("utf-8", errors="replace"),
        "stderr": stderr_bytes.decode("utf-8", errors="replace"),
        "stdout_bytes": len(stdout_bytes),
        "stderr_bytes": len(stderr_bytes),
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "stdout_complete": True,
        "stderr_complete": True,
        "sandbox_profile_sha256": hashlib.sha256(profile.encode()).hexdigest(),
        "project_selected_output_cap": None,
    }


def clone_tree(cp: Path, source: Path, destination: Path) -> dict[str, Any]:
    command = [str(cp), "-cR", str(source), str(destination)]
    started = time.monotonic()
    done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stdout, stderr = done.stdout or b"", done.stderr or b""
    return {
        "command": command, "returncode": done.returncode, "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "source": p2a.rel(source), "destination_kind": "disposable_copy_on_write_clone",
        "stdout": stdout.decode("utf-8", "replace"), "stderr": stderr.decode("utf-8", "replace"),
        "stdout_bytes": len(stdout), "stderr_bytes": len(stderr),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "network_denied_by_absence_of_network_operation": True, "project_selected_output_cap": None,
    }


def derive_disposition(sides: dict[str, dict[str, Any]], faults: list[str]) -> str:
    if faults:
        return "INCONCLUSIVE_IMPLEMENTATION_MATCHED_VERIFIER_MECHANICS"
    if set(sides) != {"parent", "target"}:
        return "INCONCLUSIVE_IMPLEMENTATION_MATCHED_VERIFIER_MECHANICS"
    if any(p2a.mapping(receipt).get("boundary_hit") for receipt in sides.values()):
        return "INCONCLUSIVE_EXPERIMENT_HOST_RESOURCE_BOUNDARY"
    parent_rc = sides["parent"].get("returncode")
    target_rc = sides["target"].get("returncode")
    if parent_rc not in (None, 0) and target_rc == 0:
        return "QUALIFIED_COMMON_EVALUATOR_PARENT_FAIL_TARGET_PASS"
    if parent_rc == 0 and target_rc == 0:
        return "INCONCLUSIVE_EXPERIMENT_EVALUATOR_NOT_DISCRIMINATIVE"
    return "INCONCLUSIVE_IMPLEMENTATION_MATCHED_VERIFIER_MECHANICS"


def not_executed_row(row: dict[str, Any], disposition: str) -> dict[str, Any]:
    return {"index": int(row.get("index") or 0), "repository": row.get("repository"), "manager": row.get("manager"), "parent": {}, "target": {}, "faults": ["not_executed_host_resource_boundary"], "disposition": disposition}


def extract_regular_archive(archive: Path, destination: Path) -> tuple[Path, list[str]]:
    faults: list[str] = []
    roots: set[str] = set()
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or any(part in {"", ".", ".."} for part in parts):
                faults.append("unsafe_member_path")
                continue
            roots.add(parts[0])
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(member.mode & 0o777)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = handle.extractfile(member)
                if extracted is None:
                    faults.append("unreadable_regular_member")
                else:
                    target.write_bytes(extracted.read())
                    target.chmod(member.mode & 0o777)
            else:
                faults.append("non_regular_member")
    if len(roots) != 1:
        faults.append("archive_root_count_invalid")
    return destination / next(iter(roots), ""), sorted(set(faults))


def archive_root(archive: Path) -> str:
    roots: set[str] = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            parts = PurePosixPath(member.name).parts
            if parts:
                roots.add(parts[0])
    return next(iter(roots)) if len(roots) == 1 else ""


def archive_member(archive: Path | None, relative: str) -> bytes | None:
    if archive is None or not archive.is_file():
        return None
    with tarfile.open(archive, "r:gz") as handle:
        members = [member for member in handle.getmembers() if member.isfile() and member.name.endswith(f"/{relative}")]
        if len(members) != 1:
            return None
        extracted = handle.extractfile(members[0])
        return extracted.read() if extracted is not None else None


def archive_changes(parent: Path | None, target: Path | None) -> list[str]:
    if parent is None or target is None:
        return []
    def identity(archive: Path) -> dict[str, str]:
        rows: dict[str, str] = {}
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle:
                parts = PurePosixPath(member.name).parts
                if member.isfile() and len(parts) > 1:
                    extracted = handle.extractfile(member)
                    if extracted is not None:
                        rows[PurePosixPath(*parts[1:]).as_posix()] = hashlib.sha256(extracted.read()).hexdigest()
        return rows
    left, right = identity(parent), identity(target)
    return sorted(path for path in set(left) | set(right) if left.get(path) != right.get(path))


def tree_identity(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total = 0
    if root.exists():
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and not candidate.is_symlink() and candidate.name != ".DS_Store"):
            size = path.stat().st_size
            total += size
            rows.append({"path": path.relative_to(root).as_posix(), "bytes": size, "sha256": p2a.sha256_file(path)})
    identity = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"path": p2a.rel(root), "file_count": len(rows), "bytes": total, "identity_sha256": identity}


def base_env(work: Path) -> dict[str, str]:
    home, tmp = work / "home", work / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    return {"HOME": str(home), "TMPDIR": str(tmp), "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "CI": "1", "NO_COLOR": "1"}


def receipt_ok(receipt: dict[str, Any]) -> bool:
    return bool(receipt) and receipt.get("returncode") == 0 and receipt.get("boundary_hit") is not True


def finish(cfg: dict[str, Any], path: Path, faults: list[str], rows: list[dict[str, Any]], executed: bool, store_before: dict[str, Any], store_after: dict[str, Any], executions: int, package_installs: int) -> dict[str, Any]:
    qualified = sum(row.get("disposition") == "QUALIFIED_COMMON_EVALUATOR_PARENT_FAIL_TARGET_PASS" for row in rows)
    completed = len(rows) == int(cfg.get("expected_task_count") or 0)
    valid = not faults and (not executed or completed)
    state = "K2_05_SIX_ROW_MATCHED_VERIFIERS_EXECUTED_WITH_SCOPED_DISPOSITIONS" if executed and valid else ("READY_FOR_SIX_ROW_MATCHED_VERIFIER_EXECUTION" if not executed and valid else "K2_05_SIX_ROW_MATCHED_VERIFIER_INVALID")
    return {
        "policy": POLICY, "created_utc": p2a.now(), "trigger_state": "GREEN" if executed and valid else ("PAUSED" if not executed and valid else "RED"),
        "state": state, "faults": sorted(set(faults)), "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)},
        "execution_performed": executed, "task_count": len(rows), "qualified_task_count": qualified,
        "inconclusive_task_count": len(rows) - qualified, "rows": rows, "retained_store_before": store_before,
        "retained_store_after": store_after, "retained_store_unchanged": store_before == store_after,
        "package_installations": package_installs,
        "source_build_executions": sum(1 for row in rows if row.get("manager") == "cargo" for side in ("parent", "target") if p2a.mapping(row.get(side))),
        "project_installations": 0,
        "repository_runner_executions": executions, "parent_target_or_evaluator_executions": executions,
        "network_enabled_calls": 0, "candidate_or_control_calls": 0, "external_reference_calls": 0, "teacher_calls": 0,
        "panel_admitted": False, "partial_panel_admission_forbidden": True, "target_production_transplant_count": 0,
        "project_selected_output_cap": None, "maximum_inference": cfg.get("maximum_inference"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
