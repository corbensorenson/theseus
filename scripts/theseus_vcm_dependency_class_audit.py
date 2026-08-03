#!/usr/bin/env python3
"""Classify VCM evaluator dependency closure without executing repositories."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import tarfile
import warnings
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_dependency_class_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_dependency_class_audit.json"
PYTHON_SUFFIXES = {".py"}
JS_SUFFIXES = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}
PYTHON_HARNESS_MODULES = {"pytest", "_pytest"}
JS_IMPORT = re.compile(
    r"(?:\bfrom\s+|\bimport\s*\(|\brequire\s*\()['\"]([^'\"]+)['\"]"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = audit(config_path)
    output = args.out or str(p2a.read_json(config_path).get("report") or "")
    p2a.write_json(p2a.resolve(output), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    runner_binding = p2a.mapping(config.get("runner_inventory"))
    runner_path = p2a.resolve(str(runner_binding.get("path") or ""))
    if not runner_path.is_file() or p2a.sha256_file(runner_path) != str(runner_binding.get("sha256") or ""):
        faults.append("runner_inventory_binding_invalid")
        runner = {}
    else:
        runner = p2a.read_json(runner_path)
    if runner.get("trigger_state") != "GREEN" or runner.get("observations", {}).get("tasks_with_any_independent_runner_receipt") != 62:
        faults.append("runner_inventory_not_green")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("normalized_archive_static_read_authorized") is not True or any(
        value is not False
        for key, value in authority.items()
        if key != "normalized_archive_static_read_authorized"
    ):
        faults.append("authority_boundary_invalid")

    rows = []
    for prior in p2a.dicts(runner.get("rows")):
        index = integer(prior.get("index"))
        archive_binding = p2a.mapping(prior.get("target_archive"))
        archive = p2a.resolve(str(archive_binding.get("path") or ""))
        if not archive.is_file() or p2a.sha256_file(archive) != str(archive_binding.get("sha256") or ""):
            faults.append(f"target_archive_binding_invalid:{index}")
            members = {}
        else:
            members = read_members(archive)
        locks = p2a.dicts(prior.get("selected_path_relevant_locks"))
        language = str(prior.get("query_language") or "")
        verifier_paths = p2a.strings(prior.get("selected_verifier_paths"))
        source_paths = p2a.strings(prior.get("selected_source_paths"))
        commands = [str(row.get("command") or "") for row in p2a.dicts(prior.get("independent_runner_receipts"))]
        harness = sorted({
            "pytest" for command in commands if re.search(r"(?:^|\s)(?:python\s+-m\s+)?pytest(?:\s|$)", command)
        })
        closure = static_evaluator_closure(language, verifier_paths, source_paths, members)
        external = sorted(set(p2a.strings(closure.get("external_dependencies"))) - set(harness))
        if locks:
            dependency_class = "EXACT_LOCK_RECEIPT_PRESENT"
            immutable_resolution_required = False
            lock_not_required = False
        elif external or closure.get("unsupported_static_language"):
            dependency_class = "IMMUTABLE_RESOLUTION_REQUIRED"
            immutable_resolution_required = True
            lock_not_required = False
        else:
            dependency_class = "LOCK_NOT_REQUIRED_FOR_STATIC_EVALUATOR_CLOSURE"
            immutable_resolution_required = False
            lock_not_required = True
        rows.append({
            "index": index,
            "repository": prior.get("repository"),
            "query_language": language,
            "selected_verifier_paths": verifier_paths,
            "runner_commands": commands,
            "relevant_manifest_receipts": p2a.dicts(prior.get("selected_path_relevant_manifests")),
            "relevant_lock_receipts": locks,
            "trusted_evaluator_harness_requirements": harness,
            "static_evaluator_closure": closure,
            "external_dependencies_excluding_harness": external,
            "dependency_class": dependency_class,
            "lock_not_required_for_scoped_evaluator": lock_not_required,
            "immutable_resolution_required_before_execution": immutable_resolution_required,
            "dependency_resolution_performed": False,
            "repository_execution_performed": False,
            "evaluator_execution_ready": False,
        })
    observations = {
        "task_count": len(rows),
        "tasks_with_exact_lock_receipt": sum(row["dependency_class"] == "EXACT_LOCK_RECEIPT_PRESENT" for row in rows),
        "tasks_lock_not_required_for_static_evaluator_closure": sum(row["lock_not_required_for_scoped_evaluator"] for row in rows),
        "tasks_requiring_immutable_resolution": sum(row["immutable_resolution_required_before_execution"] for row in rows),
        "tasks_dependency_classified": len(rows),
        "tasks_evaluator_execution_ready": 0,
        "dependency_resolutions": 0,
        "parent_target_or_evaluator_executions": 0,
    }
    if len(rows) != 62:
        faults.append("task_count_invalid")
    expected = p2a.mapping(config.get("expected_observations"))
    for key, value in expected.items():
        if observations.get(key) != value:
            faults.append(f"expected_observation_mismatch:{key}")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "STATIC_EVALUATOR_DEPENDENCY_CLASSES_BOUND" if not faults else "DEPENDENCY_CLASS_AUDIT_INVALID",
        "faults": sorted(set(faults)),
        "config": artifact(config_path),
        "runner_inventory": artifact(runner_path),
        "observations": observations,
        "rows": rows,
        "classification_policy": {
            "scope": "transitive static imports reachable from selected executable verifier roots plus source files explicitly named by those verifier bytes",
            "exact_lock_receipt_is_not_installed_dependency_proof": True,
            "pytest_is_evaluator_harness_not_project_lock": True,
            "unresolved_third_party_import_requires_immutable_resolution": True,
            "static_stdlib_or_node_builtin_closure_can_waive_project_lock": True,
            "dynamic_runtime_behavior_is_not_qualified": True,
            "dependency_resolution_or_installation": False,
            "repository_execution": False,
        },
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def static_evaluator_closure(
    language: str, verifier_paths: list[str], source_paths: list[str], members: dict[str, bytes]
) -> dict[str, Any]:
    if language == "Python":
        return python_closure(verifier_paths, source_paths, members)
    if language in {"JavaScript", "TypeScript"}:
        return javascript_closure(verifier_paths, source_paths, members)
    if language == "Rust":
        return rust_closure(verifier_paths, members)
    return {
        "visited_files": [], "visited_file_receipts": [], "external_dependencies": [],
        "harness_dependencies": [], "unresolved_local_imports": [],
        "unsupported_static_language": True,
    }


def python_closure(verifier_paths: list[str], source_paths: list[str], members: dict[str, bytes]) -> dict[str, Any]:
    module_map: dict[str, set[str]] = {}
    for path in members:
        if not path.endswith(".py"):
            continue
        parts = list(PurePosixPath(path).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        variants = [parts]
        for marker in ("src", "lib"):
            if marker in parts:
                variants.append(parts[parts.index(marker) + 1 :])
        original_parts = list(PurePosixPath(path).with_suffix("").parts)
        for start in range(len(original_parts) - 1):
            package_init = PurePosixPath(*original_parts[: start + 1], "__init__.py").as_posix()
            if package_init in members:
                variants.append(original_parts[start:])
        for variant in variants:
            if variant:
                module_map.setdefault(".".join(variant), set()).add(path)
    roots = [path for path in verifier_paths if path.endswith(".py") and path in members]
    verifier_text = "\n".join(members[path].decode("utf-8", "replace") for path in roots)
    for path in source_paths:
        if path.endswith(".py") and path in members and source_path_explicitly_named(path, verifier_text):
            roots.append(path)
    queue = deque(sorted(set(roots)))
    visited: set[str] = set()
    external: set[str] = set()
    harness: set[str] = set()
    unresolved: set[str] = set()
    parse_faults: list[str] = []
    while queue:
        path = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        text = members[path].decode("utf-8", errors="replace")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text, filename=path)
        except (SyntaxError, ValueError):
            parse_faults.append(path)
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    names = relative_python_imports(path, node)
                elif node.module:
                    names = [node.module]
                    names.extend(f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*")
            for name in names:
                top = name.split(".", 1)[0]
                if top in sys.stdlib_module_names or top in {"__future__"}:
                    continue
                if top in PYTHON_HARNESS_MODULES:
                    harness.add(top)
                    continue
                local = resolve_python_module(name, module_map, current_path=path, members=members)
                if local:
                    queue.extend(sorted(local - visited))
                elif name.startswith("."):
                    unresolved.add(name)
                else:
                    external.add(top)
    return closure_payload(visited, members, external, harness, unresolved, parse_faults)


def relative_python_imports(path: str, node: ast.ImportFrom) -> list[str]:
    parts = list(PurePosixPath(path).parent.parts)
    trim = max(0, node.level - 1)
    base = parts[: len(parts) - trim] if trim else parts
    if node.module:
        base.extend(node.module.split("."))
    names = [".".join(base)] if base else []
    names.extend(".".join([*base, alias.name]) for alias in node.names if alias.name != "*")
    return [name for name in names if name]


def resolve_python_module(
    name: str, module_map: dict[str, set[str]], *, current_path: str, members: dict[str, bytes]
) -> set[str]:
    if name in module_map:
        return set(module_map[name])
    sibling_base = PurePosixPath(current_path).parent.joinpath(*name.split("."))
    for candidate in (
        sibling_base.with_suffix(".py").as_posix(),
        sibling_base.joinpath("__init__.py").as_posix(),
    ):
        if candidate in members:
            return {candidate}
    parts = name.split(".")
    while len(parts) > 1:
        parts.pop()
        candidate = ".".join(parts)
        if candidate in module_map:
            return set(module_map[candidate])
    return set()


def javascript_closure(verifier_paths: list[str], source_paths: list[str], members: dict[str, bytes]) -> dict[str, Any]:
    roots = [path for path in verifier_paths if PurePosixPath(path).suffix.lower() in JS_SUFFIXES and path in members]
    verifier_text = "\n".join(members[path].decode("utf-8", "replace") for path in roots)
    for path in source_paths:
        if PurePosixPath(path).suffix.lower() in JS_SUFFIXES and path in members and source_path_explicitly_named(path, verifier_text):
            roots.append(path)
    queue = deque(sorted(set(roots)))
    visited: set[str] = set()
    external: set[str] = set()
    unresolved: set[str] = set()
    while queue:
        path = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        text = members[path].decode("utf-8", errors="replace")
        for name in JS_IMPORT.findall(text):
            if name.startswith("node:"):
                continue
            if name.startswith(("./", "../")):
                resolved = resolve_js_relative(path, name, members)
                if resolved:
                    queue.append(resolved)
                else:
                    unresolved.add(f"{path}:{name}")
            else:
                external.add(name.split("/", 1)[0] if not name.startswith("@") else "/".join(name.split("/")[:2]))
    return closure_payload(visited, members, external, set(), unresolved, [])


def resolve_js_relative(path: str, name: str, members: dict[str, bytes]) -> str:
    base = PurePosixPath(path).parent.joinpath(name)
    candidates = [base.as_posix()]
    if not base.suffix:
        candidates.extend(base.with_suffix(suffix).as_posix() for suffix in JS_SUFFIXES)
        candidates.extend(base.joinpath("index" + suffix).as_posix() for suffix in JS_SUFFIXES)
    for candidate in candidates:
        normalized = PurePosixPath(candidate).as_posix()
        while normalized.startswith("../"):
            normalized = normalized[3:]
        if normalized in members:
            return normalized
    return ""


def rust_closure(verifier_paths: list[str], members: dict[str, bytes]) -> dict[str, Any]:
    manifests = []
    external = set()
    for path, payload in members.items():
        if PurePosixPath(path).name.lower() != "cargo.toml":
            continue
        text = payload.decode("utf-8", errors="replace")
        if any(marker in text for marker in ("[dependencies]", "[dev-dependencies]", "[workspace.dependencies]", ".dependencies]")):
            manifests.append(path)
            external.add("cargo_manifest_dependency_graph")
    visited = {path for path in verifier_paths if path.endswith(".rs") and path in members}
    return closure_payload(visited, members, external, set(), set(), []) | {
        "dependency_manifests_with_dependency_sections": sorted(manifests)
    }


def source_path_explicitly_named(path: str, text: str) -> bool:
    parts = PurePosixPath(path).parts
    return path in text or PurePosixPath(path).name in text or (
        len(parts) >= 2 and all(part in text for part in parts[-2:])
    )


def closure_payload(
    visited: set[str], members: dict[str, bytes], external: set[str], harness: set[str],
    unresolved: set[str], parse_faults: list[str]
) -> dict[str, Any]:
    return {
        "visited_files": sorted(visited),
        "visited_file_receipts": [
            {"path": path, "sha256": hashlib.sha256(members[path]).hexdigest()} for path in sorted(visited)
        ],
        "external_dependencies": sorted(external),
        "harness_dependencies": sorted(harness),
        "unresolved_local_imports": sorted(unresolved),
        "parse_faults": sorted(parse_faults),
        "unsupported_static_language": False,
    }


def read_members(path: Path) -> dict[str, bytes]:
    result = {}
    with tarfile.open(path, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) <= 1:
                continue
            relative = PurePosixPath(*parts[1:]).as_posix()
            extracted = handle.extractfile(member)
            result[relative] = extracted.read() if extracted else b""
    return result


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "observations", "parent_target_or_evaluator_executions",
        "candidate_or_control_calls", "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
