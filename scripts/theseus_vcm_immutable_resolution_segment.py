#!/usr/bin/env python3
"""Resolve the six prospectively sealed immutable evaluator dependency closures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_immutable_resolution_segment_v1"
STATE = "PROSPECTIVE_K2_05_SIX_IMMUTABLE_RESOLUTION_CLOSURES_PREQUALIFIED_WHEEL_V4"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_immutable_resolution_segment.json"
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
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def preflight(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != POLICY or cfg.get("state") != STATE:
        faults.append("policy_or_state_invalid")
    for key, expected in (("owner", Path(__file__).resolve()),):
        binding = p2a.resolve(str(cfg.get(key) or ""))
        if binding != expected or not binding.is_file() or p2a.sha256_file(binding) != cfg.get(f"{key}_sha256"):
            faults.append(f"{key}_binding_invalid")
    audit_owner = p2a.resolve(str(cfg.get("audit_owner") or ""))
    if not audit_owner.is_file() or p2a.sha256_file(audit_owner) != cfg.get("audit_owner_sha256"):
        faults.append("audit_owner_binding_invalid")

    sources: dict[str, dict[str, Any]] = {}
    for raw in p2a.dicts(cfg.get("sources")):
        source_id = str(raw.get("id") or "")
        source = p2a.resolve(str(raw.get("path") or ""))
        if not source_id or not source.is_file() or p2a.sha256_file(source) != raw.get("sha256"):
            faults.append(f"source_binding_invalid:{source_id}")
            sources[source_id] = {}
        else:
            sources[source_id] = p2a.read_json(source)

    predecessor: dict[str, dict[str, Any]] = {}
    for name, raw in p2a.mapping(cfg.get("predecessor")).items():
        binding = p2a.mapping(raw)
        predecessor_path = p2a.resolve(str(binding.get("path") or ""))
        if not predecessor_path.is_file() or p2a.sha256_file(predecessor_path) != binding.get("sha256"):
            faults.append(f"predecessor_binding_invalid:{name}")
            predecessor[name] = {}
        else:
            predecessor[name] = p2a.read_json(predecessor_path)
    previous_report = predecessor.get("producer_report", {})
    previous_audit = predecessor.get("audit_report", {})
    if previous_report.get("trigger_state") != "GREEN" or previous_report.get("qualified_task_count") != 5 or previous_report.get("inconclusive_task_count") != 1:
        faults.append("predecessor_producer_state_invalid")
    if previous_audit.get("trigger_state") != "GREEN" or p2a.strings(previous_audit.get("faults")) or previous_audit.get("qualified_task_count") != 5 or previous_audit.get("inconclusive_task_count") != 1:
        faults.append("predecessor_audit_state_invalid")
    acquisition = sources.get("python314_toolchain_acquisition", {})
    interpreter = p2a.mapping(acquisition.get("interpreter"))
    python314 = p2a.mapping(p2a.mapping(cfg.get("tools")).get("python_3_14"))
    if acquisition.get("trigger_state") != "GREEN" or interpreter.get("path") != python314.get("path") or interpreter.get("sha256") != python314.get("sha256") or interpreter.get("version") != "3.14.2":
        faults.append("python314_acquisition_binding_invalid")
    build = sources.get("sandbox_wheel_build", {})
    build_audit = sources.get("sandbox_wheel_build_audit", {})
    retained_wheel = p2a.mapping(p2a.mapping(build.get("receipt")).get("retained_wheel"))
    local_wheel = p2a.mapping(cfg.get("local_wheel"))
    local_wheel_path = p2a.resolve(str(local_wheel.get("path") or ""))
    if build.get("trigger_state") != "GREEN" or build_audit.get("trigger_state") != "GREEN" or retained_wheel.get("path") != local_wheel.get("path") or retained_wheel.get("sha256") != local_wheel.get("sha256") or not local_wheel_path.is_file() or p2a.sha256_file(local_wheel_path) != local_wheel.get("sha256"):
        faults.append("prequalified_local_wheel_binding_invalid")

    tools: dict[str, str] = {}
    for name, raw in p2a.mapping(cfg.get("tools")).items():
        binding = p2a.mapping(raw)
        tool = p2a.resolve(str(binding.get("path") or ""))
        if not tool.is_file() or p2a.sha256_file(tool) != binding.get("sha256"):
            faults.append(f"tool_binding_invalid:{name}")
        tools[name] = str(tool)

    closures = {int(row.get("campaign_index") or 0): row for row in p2a.dicts(sources.get("repository_closures", {}).get("tasks"))}
    inventory = {int(row.get("index") or 0): row for row in p2a.dicts(sources.get("runner_inventory", {}).get("rows"))}
    classes = {int(row.get("index") or 0): row for row in p2a.dicts(sources.get("dependency_classes", {}).get("rows"))}
    rows = p2a.dicts(cfg.get("rows"))
    indices = [int(row.get("index") or 0) for row in rows]
    if indices != EXPECTED_INDICES or len(set(indices)) != len(EXPECTED_INDICES):
        faults.append("immutable_resolution_denominator_invalid")
    bound_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row.get("index") or 0)
        closure, runner, classification = closures.get(index, {}), inventory.get(index, {}), classes.get(index, {})
        repositories = {str(item.get("repository") or "") for item in (row, closure, runner, classification)}
        if len(repositories) != 1 or "" in repositories:
            faults.append(f"source_alignment_invalid:{index}")
        if classification.get("dependency_class") != "IMMUTABLE_RESOLUTION_REQUIRED" or classification.get("immutable_resolution_required_before_execution") is not True:
            faults.append(f"dependency_class_invalid:{index}")
        expected_manifests = sorted(p2a.dicts(classification.get("relevant_manifest_receipts")), key=lambda item: str(item.get("path")))
        configured_manifests = sorted(p2a.dicts(row.get("manifest_receipts")), key=lambda item: str(item.get("path")))
        if configured_manifests != expected_manifests:
            faults.append(f"manifest_receipts_invalid:{index}")
        if p2a.strings(row.get("selected_verifier_paths")) != p2a.strings(classification.get("selected_verifier_paths")):
            faults.append(f"selected_verifier_paths_invalid:{index}")
        target = next((item for item in p2a.dicts(closure.get("artifacts")) if item.get("label") == "target"), {})
        archive = p2a.resolve(str(target.get("normalized") or ""))
        if not archive.is_file() or p2a.sha256_file(archive) != target.get("normalized_sha256"):
            faults.append(f"target_archive_binding_invalid:{index}")
        else:
            observed, archive_faults = archive_manifest_receipts(archive, str(target.get("source_archive_root") or ""), configured_manifests)
            faults.extend(f"task_{index}:{fault}" for fault in archive_faults)
            if observed != configured_manifests:
                faults.append(f"archive_manifest_identity_invalid:{index}")
        manager = str(row.get("manager") or "")
        language = str(classification.get("query_language") or "")
        if (manager, language) not in {("uv", "Python"), ("cargo", "Rust")}:
            faults.append(f"manager_language_invalid:{index}")
        if manager == "uv" and str(row.get("python_tool") or "") not in tools:
            faults.append(f"python_tool_binding_invalid:{index}")
        faults.extend(f"task_{index}:{fault}" for fault in static_input_safety(row, archive, str(target.get("source_archive_root") or "")))
        bound_rows[index] = {"row": row, "closure": closure, "runner": runner, "classification": classification, "target": target, "archive": archive}

    authority = p2a.mapping(cfg.get("authority"))
    allowed = {"target_archive_extraction_authorized", "registry_metadata_resolution_authorized", "immutable_lock_write_authorized", "shared_resolver_cache_authorized"}
    for key, value in authority.items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")
    output_dir = p2a.resolve(str(cfg.get("output_directory") or ""))
    cache_root = p2a.resolve(str(cfg.get("shared_cache_root") or ""))
    if output_dir.parent != (ROOT / "reports").resolve() or output_dir.name != "theseus_vcm_immutable_resolution_locks":
        faults.append("output_directory_invalid")
    if cache_root != (ROOT / "runtime/vcm_evaluator/dependency_store/immutable-resolution").resolve():
        faults.append("shared_cache_root_invalid")
    return cfg, {"sources": sources, "tools": tools, "rows": bound_rows, "predecessor": predecessor}, sorted(set(faults))


def preflight_report(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, bound, faults = preflight(path)
    return finish(cfg, path, [], faults, execution_performed=False, cache_root=None, free_before=shutil.disk_usage(ROOT).free)


def execute(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, bound, faults = preflight(path)
    if faults:
        return finish(cfg, path, [], faults, execution_performed=False, cache_root=None, free_before=shutil.disk_usage(ROOT).free)
    limits = p2a.mapping(cfg.get("limits"))
    reserve = int(limits.get("minimum_free_bytes_after_execution") or 0)
    max_cache = int(limits.get("maximum_shared_cache_bytes") or 0)
    free_before = shutil.disk_usage(ROOT).free
    cache_root = p2a.resolve(str(cfg.get("shared_cache_root") or ""))
    output_dir = p2a.resolve(str(cfg.get("output_directory") or ""))
    if free_before - max_cache < reserve:
        return finish(cfg, path, [], ["free_space_reserve_preflight_boundary_hit"], execution_performed=False, cache_root=cache_root, free_before=free_before)
    cache_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-immutable-resolution-", dir="/private/tmp") as raw:
        temp_root = Path(raw).resolve()
        for index in EXPECTED_INDICES:
            item = bound["rows"][index]
            row = p2a.mapping(item["row"])
            work = temp_root / f"task-{index:02d}"
            root, extract_faults = extract_regular_archive(item["archive"], work)
            if extract_faults:
                results.append(scoped_failure(row, item, "INCONCLUSIVE_IMPLEMENTATION_ARCHIVE_EXTRACTION", extract_faults))
                continue
            output = output_dir / str(row.get("output_name") or "")
            reused = reusable_predecessor_lock(bound, index, output)
            if reused:
                results.append({"index": index, "repository": row.get("repository"), "manager": row.get("manager"), "target_archive": p2a.rel(item["archive"]), "target_archive_sha256": p2a.sha256_file(item["archive"]), "command": [], "receipt": {"lock": reused, "predecessor_reuse": True, "predecessor_producer_report_sha256": p2a.mapping(cfg.get("predecessor")).get("producer_report", {}).get("sha256")}, "faults": [], "disposition": "RESOLUTION_QUALIFIED_IMMUTABLE_LOCK_REUSED_FROM_SEALED_PREDECESSOR"})
                continue
            if output.exists():
                results.append(scoped_failure(row, item, "INCONCLUSIVE_EXPERIMENT_IMMUTABLE_OUTPUT_ALREADY_EXISTS", ["immutable_output_already_exists"]))
                continue
            command, env, generated = resolution_command(cfg, bound, row, root, cache_root, temp_root)
            source_before = tree_identity(root, excluded={"Cargo.lock"})
            receipt = run(command, root, env, limits)
            source_after = tree_identity(root, excluded={"Cargo.lock"})
            receipt.update({"source_before": source_before, "source_after": source_after})
            row_faults: list[str] = []
            if source_before != source_after:
                row_faults.append("source_mutated_outside_generated_lock")
            if receipt.get("boundary_hit"):
                disposition = "INCONCLUSIVE_EXPERIMENT_HOST_RESOURCE_BOUNDARY"
            elif receipt.get("returncode") != 0:
                disposition = "INCONCLUSIVE_EXPERIMENT_DEPENDENCY_RESOLUTION"
            elif not generated.is_file():
                disposition = "INCONCLUSIVE_IMPLEMENTATION_LOCK_OUTPUT_MISSING"
                row_faults.append("generated_lock_missing")
            else:
                lock_receipt, validation_faults = validate_lock(str(row.get("manager") or ""), generated)
                row_faults.extend(validation_faults)
                if validation_faults:
                    disposition = "INCONCLUSIVE_IMPLEMENTATION_LOCK_VALIDATION"
                else:
                    shutil.copyfile(generated, output)
                    lock_receipt.update({"path": p2a.rel(output), "sha256": p2a.sha256_file(output), "bytes": output.stat().st_size})
                    disposition = "RESOLUTION_QUALIFIED_IMMUTABLE_LOCK"
                    receipt["lock"] = lock_receipt
            results.append({"index": index, "repository": row.get("repository"), "manager": row.get("manager"), "target_archive": p2a.rel(item["archive"]), "target_archive_sha256": p2a.sha256_file(item["archive"]), "command": command, "receipt": receipt, "faults": sorted(set(row_faults)), "disposition": disposition})
            if tree_identity(cache_root).get("bytes", 0) > max_cache or shutil.disk_usage(ROOT).free < reserve:
                faults.append("shared_cache_or_free_space_postflight_boundary_hit")
                break
    if len(results) != len(EXPECTED_INDICES) and not faults:
        faults.append("immutable_resolution_denominator_not_completed")
    return finish(cfg, path, results, faults, execution_performed=True, cache_root=cache_root, free_before=free_before)


def resolution_command(cfg: dict[str, Any], bound: dict[str, Any], row: dict[str, Any], root: Path, cache_root: Path, temp_root: Path) -> tuple[list[str], dict[str, str], Path]:
    index = int(row["index"])
    tools = bound["tools"]
    home = temp_root / f"home-{index:02d}"
    tmp = temp_root / f"tmp-{index:02d}"
    home.mkdir(); tmp.mkdir()
    env = {"HOME": str(home), "TMPDIR": str(tmp), "PATH": "/usr/bin:/bin", "CI": "1", "NO_COLOR": "1", "LANG": "C", "LC_ALL": "C"}
    if row.get("manager") == "uv":
        generated = temp_root / f"task-{index:02d}-requirements.lock"
        python = tools[str(row.get("python_tool") or "")]
        command = [tools["uv"], "pip", "compile", *[str(root / value) for value in p2a.strings(row.get("inputs"))]]
        for extra in p2a.strings(row.get("extras")):
            command.extend(["--extra", extra])
        if row.get("find_links"):
            command.extend(["--find-links", str(p2a.resolve(str(row.get("find_links"))))])
        command.extend(["--python", python, "--python-platform", "aarch64-apple-darwin", "--generate-hashes", "--no-build", "--index-strategy", "first-index", "--default-index", "https://pypi.org/simple", "--cache-dir", str(cache_root / "uv"), "--output-file", str(generated), "--color", "never", "--no-progress"])
        env.update({"UV_PYTHON": python, "UV_PYTHON_DOWNLOADS": "never", "UV_NO_CONFIG": "1"})
    else:
        generated = root / "Cargo.lock"
        command = [tools["cargo"], "generate-lockfile", "--manifest-path", str(root / "Cargo.toml"), "--config", "net.git-fetch-with-cli=false"]
        env.update({"CARGO_HOME": str(cache_root / "cargo"), "CARGO_NET_GIT_FETCH_WITH_CLI": "false", "CARGO_REGISTRIES_CRATES_IO_PROTOCOL": "sparse"})
    return command, env, generated


def run(command: list[str], cwd: Path, env: dict[str, str], limits: dict[str, Any]) -> dict[str, Any]:
    def setup() -> None:
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (int(limits["cpu_seconds_per_command"]), int(limits["cpu_seconds_per_command"])))
        resource.setrlimit(resource.RLIMIT_NOFILE, (int(limits["open_files"]), int(limits["open_files"])))
        resource.setrlimit(resource.RLIMIT_NPROC, (int(limits["user_processes"]), int(limits["user_processes"])))
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=False, preexec_fn=setup)
    boundary_hit = False
    try:
        stdout, stderr = process.communicate(timeout=int(limits["wall_seconds_per_command"]))
    except subprocess.TimeoutExpired:
        boundary_hit = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=int(limits["terminate_grace_seconds"]))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    return {"returncode": process.returncode, "duration_ms": round((time.monotonic() - started) * 1000, 3), "boundary_hit": boundary_hit, "boundary_reason": "wall_timeout" if boundary_hit else "", "stdout": stdout.decode("utf-8", "replace"), "stderr": stderr.decode("utf-8", "replace"), "stdout_bytes": len(stdout), "stderr_bytes": len(stderr), "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest(), "stdout_complete": True, "stderr_complete": True, "project_selected_output_cap": None}


def static_input_safety(row: dict[str, Any], archive: Path, root: str) -> list[str]:
    faults: list[str] = []
    if not archive.is_file():
        return ["archive_absent_for_input_safety"]
    with tarfile.open(archive, "r:gz") as handle:
        if row.get("manager") == "uv":
            for input_path in p2a.strings(row.get("inputs")):
                payload = read_archive_member(handle, root, input_path).decode("utf-8", "replace")
                if input_path.endswith(".toml"):
                    document = tomllib.loads(payload)
                    project = p2a.mapping(document.get("project"))
                    dependency_values = [*p2a.strings(project.get("dependencies"))]
                    for values in p2a.mapping(project.get("optional-dependencies")).values():
                        dependency_values.extend(p2a.strings(values))
                    unsafe = any(re.search(r"(?i)(?:\s@\s|git\+|https?://|file://)", value) for value in dependency_values)
                else:
                    unsafe = bool(re.search(r"(?im)^\s*(--?(?:index-url|extra-index-url|find-links)|-r\s|--requirement\s|--editable\s|-e\s)", payload) or re.search(r"(?i)(?:\s@\s|git\+|https?://|file://)", payload))
                if unsafe:
                    faults.append(f"non_registry_or_nested_python_input:{input_path}")
        else:
            for member in handle.getmembers():
                rel = relative_member(member.name, root)
                if member.isfile() and rel.endswith("Cargo.toml"):
                    extracted = handle.extractfile(member)
                    payload = (extracted.read() if extracted else b"").decode("utf-8", "replace")
                    if re.search(r"(?m)\b(?:git|registry)\s*=", payload):
                        faults.append(f"non_default_registry_cargo_input:{rel}")
    return sorted(set(faults))


def archive_manifest_receipts(archive: Path, root: str, expected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    faults: list[str] = []
    with tarfile.open(archive, "r:gz") as handle:
        for item in expected:
            path = str(item.get("path") or "")
            try:
                payload = read_archive_member(handle, root, path)
            except (KeyError, ValueError):
                faults.append(f"manifest_missing:{path}")
                continue
            rows.append({"path": path, "sha256": hashlib.sha256(payload).hexdigest()})
    return sorted(rows, key=lambda item: item["path"]), faults


def read_archive_member(handle: tarfile.TarFile, root: str, relative: str) -> bytes:
    member = handle.getmember(f"{root}/{relative}")
    extracted = handle.extractfile(member)
    if extracted is None:
        raise ValueError(relative)
    return extracted.read()


def relative_member(name: str, root: str) -> str:
    prefix = root.rstrip("/") + "/"
    return name[len(prefix):] if name.startswith(prefix) else name


def extract_regular_archive(archive: Path, destination: Path) -> tuple[Path, list[str]]:
    faults: list[str] = []
    destination.mkdir(parents=True, exist_ok=True)
    roots: set[str] = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                faults.append("unsafe_archive_member")
                continue
            roots.add(pure.parts[0])
            if not (member.isfile() or member.isdir()):
                continue
            target = destination.joinpath(*pure.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = handle.extractfile(member)
                if extracted is None:
                    faults.append("archive_member_unreadable")
                    continue
                target.write_bytes(extracted.read())
                target.chmod(member.mode & 0o777)
    if len(roots) != 1:
        faults.append("archive_root_ambiguous")
    root = destination / next(iter(roots), "")
    return root, sorted(set(faults))


def validate_lock(manager: str, path: Path) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    package_count = 0
    if manager == "cargo":
        try:
            document = tomllib.loads(path.read_text())
            package_count = len(document.get("package", []))
            if document.get("version") not in {3, 4} or package_count < 1:
                faults.append("cargo_lock_structure_invalid")
        except (OSError, tomllib.TOMLDecodeError):
            faults.append("cargo_lock_parse_failed")
    else:
        payload = path.read_text(errors="replace")
        package_count = len(re.findall(r"(?m)^[A-Za-z0-9_.-]+==[^\s\\]+", payload))
        hashes = len(re.findall(r"--hash=sha256:[0-9a-f]{64}", payload))
        if package_count < 1 or hashes < package_count:
            faults.append("hashed_python_lock_structure_invalid")
    return {"manager": manager, "package_count": package_count}, faults


def reusable_predecessor_lock(bound: dict[str, Any], index: int, output: Path) -> dict[str, Any]:
    previous = p2a.mapping(p2a.mapping(bound.get("predecessor")).get("producer_report"))
    row = next((item for item in p2a.dicts(previous.get("rows")) if int(item.get("index") or 0) == index), {})
    if not str(row.get("disposition") or "").startswith("RESOLUTION_QUALIFIED_IMMUTABLE_LOCK"):
        return {}
    lock = p2a.mapping(p2a.mapping(row.get("receipt")).get("lock"))
    lock_path = p2a.resolve(str(lock.get("path") or ""))
    if lock_path != output or not lock_path.is_file() or p2a.sha256_file(lock_path) != lock.get("sha256") or lock_path.stat().st_size != lock.get("bytes"):
        return {}
    return dict(lock)


def tree_identity(root: Path, *, excluded: set[str] | None = None) -> dict[str, Any]:
    excluded = excluded or set()
    rows = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.relative_to(root).as_posix() not in excluded:
                rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": p2a.sha256_file(path)})
    return {"file_count": len(rows), "bytes": sum(item["bytes"] for item in rows), "identity_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def scoped_failure(row: dict[str, Any], item: dict[str, Any], disposition: str, faults: list[str]) -> dict[str, Any]:
    return {"index": row.get("index"), "repository": row.get("repository"), "manager": row.get("manager"), "target_archive": p2a.rel(item["archive"]), "target_archive_sha256": p2a.sha256_file(item["archive"]), "command": [], "receipt": {}, "faults": sorted(set(faults)), "disposition": disposition}


def finish(cfg: dict[str, Any], path: Path, rows: list[dict[str, Any]], faults: list[str], *, execution_performed: bool, cache_root: Path | None, free_before: int) -> dict[str, Any]:
    qualified = sum(str(row.get("disposition") or "").startswith("RESOLUTION_QUALIFIED_IMMUTABLE_LOCK") for row in rows)
    return {"policy": POLICY, "created_utc": p2a.now(), "trigger_state": "RED" if faults else ("GREEN" if execution_performed else "PAUSED"), "state": "K2_05_IMMUTABLE_RESOLUTION_SEGMENT_EXECUTED_WITH_SCOPED_DISPOSITIONS" if execution_performed and not faults else ("READY_FOR_SIX_IMMUTABLE_RESOLUTION_CLOSURES" if not faults else "K2_05_IMMUTABLE_RESOLUTION_SEGMENT_INVALID"), "faults": sorted(set(faults)), "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}, "execution_performed": execution_performed, "task_count": len(rows), "qualified_task_count": qualified, "inconclusive_task_count": len(rows) - qualified, "rows": rows, "cache": tree_identity(cache_root) if cache_root else {}, "free_bytes_before": free_before, "free_bytes_after": shutil.disk_usage(ROOT).free, "panel_admitted": False, "partial_panel_admission_forbidden": True, "package_installations": 0, "source_build_executions": 0, "repository_runner_executions": 0, "parent_target_or_evaluator_executions": 0, "candidate_or_control_calls": 0, "external_reference_calls": 0, "teacher_calls": 0, "maximum_inference": cfg.get("maximum_inference")}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in ("trigger_state", "state", "faults", "execution_performed", "task_count", "qualified_task_count", "inconclusive_task_count", "package_installations", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls")}


if __name__ == "__main__":
    raise SystemExit(main())
