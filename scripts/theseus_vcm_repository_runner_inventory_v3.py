#!/usr/bin/env python3
"""Bind selected-verifier runtime receipts for the repaired VCM source panel.

This successor does not execute repositories or resolve dependencies.  It
retains independently declared project runners from the preliminary inventory
and closes only the residual cases where the selected verifier itself provides
an unambiguous standard runtime convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_repository_runner_inventory_v3"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_repository_runner_inventory_v3.json"
MANIFESTS = {
    "package.json", "pyproject.toml", "setup.py", "setup.cfg", "cargo.toml",
    "requirements.txt", "pipfile",
}
LOCKS = {
    "package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock",
    "bun.lock", "bun.lockb", "uv.lock", "poetry.lock", "pdm.lock",
    "pipfile.lock", "cargo.lock",
}
PYTEST_RUNNABLE = re.compile(
    r"(?:^|\s)(?:/[^\s]+/)?python(?:3)?\s+-m\s+pytest\s+([^\s`]+)", re.I
)
THIRD_PARTY_JS_IMPORT = re.compile(
    r"(?:from\s+|import\s*\()['\"](?!node:|\.{1,2}/|/)([^'\"]+)['\"]"
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
    reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, Path] = {}
    for name, raw in p2a.mapping(config.get("reports")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        report_paths[name] = path
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"report_binding_invalid:{name}")
            reports[name] = {}
        else:
            reports[name] = p2a.read_json(path)
    inventory = reports.get("preliminary_inventory", {})
    panel = reports.get("source_panel", {})
    if inventory.get("trigger_state") != "GREEN" or inventory.get("observations", {}).get("task_count") != 62:
        faults.append("preliminary_inventory_not_green")
    if panel.get("trigger_state") != "GREEN" or panel.get("source_panel_admitted") is not True:
        faults.append("source_panel_not_admitted")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("normalized_archive_static_read_authorized") is not True or any(
        value is not False
        for key, value in authority.items()
        if key != "normalized_archive_static_read_authorized"
    ):
        faults.append("authority_boundary_invalid")

    panel_by_index = {
        integer(row.get("index")): row for row in p2a.dicts(panel.get("assembled_rows"))
    }
    rows = []
    for prior in p2a.dicts(inventory.get("rows")):
        index = integer(prior.get("index"))
        selected = p2a.mapping(panel_by_index.get(index))
        if str(prior.get("repository") or "") != str(selected.get("repository") or ""):
            faults.append(f"panel_inventory_identity_mismatch:{index}")
        target = p2a.mapping(prior.get("target"))
        archive = p2a.resolve(str(target.get("archive") or ""))
        if not archive.is_file() or p2a.sha256_file(archive) != str(target.get("archive_sha256") or ""):
            faults.append(f"target_archive_binding_invalid:{index}")
            members: dict[str, bytes] = {}
        else:
            members = read_members(archive)
        selected_paths = sorted(set(
            p2a.strings(selected.get("selected_source_paths"))
            + p2a.strings(selected.get("selected_verifier_paths"))
        ))
        relevant_manifests = relevant_members(members, selected_paths, MANIFESTS, requirements=True)
        relevant_locks = relevant_members(members, selected_paths, LOCKS)
        inherited = p2a.dicts(target.get("runner_receipts"))
        derived = []
        # The successor is intentionally narrow: independently declared
        # repository runners remain authoritative.  Runtime conventions are
        # considered only for the four residual tasks, so parent-only golden
        # artifacts in already-covered tasks cannot become false target gaps.
        if not inherited:
            for verifier_path in p2a.strings(selected.get("selected_verifier_paths")):
                payload = members.get(verifier_path)
                if payload is None:
                    continue
                candidate = derive_selected_verifier_receipt(verifier_path, payload)
                if candidate:
                    derived.append(candidate)
        receipts = unique_receipts([*inherited, *derived])
        rows.append({
            "index": index,
            "repository": prior.get("repository"),
            "query_language": selected.get("query_language"),
            "target_archive": artifact(archive),
            "selected_source_paths": p2a.strings(selected.get("selected_source_paths")),
            "selected_verifier_paths": p2a.strings(selected.get("selected_verifier_paths")),
            "root_manifests": p2a.strings(target.get("root_manifests")),
            "root_locks": p2a.strings(target.get("root_locks")),
            "selected_path_relevant_manifests": relevant_manifests,
            "selected_path_relevant_locks": relevant_locks,
            "inherited_independent_runner_receipts": inherited,
            "selected_verifier_runtime_receipts": derived,
            "independent_runner_receipts": receipts,
            "independent_runner_receipt_present": bool(receipts),
            "dependency_resolution_performed": False,
            "execution_performed": False,
        })
    observations = {
        "task_count": len(rows),
        "tasks_with_root_manifest": sum(bool(row["root_manifests"]) for row in rows),
        "tasks_with_selected_path_relevant_manifest": sum(bool(row["selected_path_relevant_manifests"]) for row in rows),
        "tasks_with_root_lock": sum(bool(row["root_locks"]) for row in rows),
        "tasks_with_selected_path_relevant_lock": sum(bool(row["selected_path_relevant_locks"]) for row in rows),
        "tasks_with_inherited_independent_runner_receipt": sum(bool(row["inherited_independent_runner_receipts"]) for row in rows),
        "tasks_with_selected_verifier_runtime_receipt": sum(bool(row["selected_verifier_runtime_receipts"]) for row in rows),
        "tasks_with_any_independent_runner_receipt": sum(row["independent_runner_receipt_present"] for row in rows),
        "tasks_without_independent_runner_receipt": sum(not row["independent_runner_receipt_present"] for row in rows),
        "dependency_resolutions": 0,
        "parent_target_or_evaluator_executions": 0,
    }
    if len(rows) != 62:
        faults.append("task_count_invalid")
    expected = p2a.mapping(config.get("expected_observations"))
    for key, value in expected.items():
        if observations.get(key) != value:
            faults.append(f"expected_observation_mismatch:{key}")
    residual_indices = [row["index"] for row in rows if not row["independent_runner_receipt_present"]]
    if residual_indices:
        faults.append("independent_runner_receipt_residual")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "SELECTED_VERIFIER_RUNNER_RECEIPTS_COMPLETE" if not faults else "RUNNER_RECEIPT_BINDING_INVALID",
        "faults": sorted(set(faults)),
        "config": artifact(config_path),
        "reports": {name: artifact(path) for name, path in report_paths.items()},
        "observations": observations,
        "residual_indices": residual_indices,
        "rows": rows,
        "derivation_policy": {
            "preliminary_independently_declared_runners_retained": True,
            "python_unittest_selected_file_convention": True,
            "selected_file_embedded_pytest_command": True,
            "self_contained_node_builtin_mjs_convention": True,
            "hand_authored_repository_specific_command": False,
            "dependency_resolution_or_installation": False,
            "repository_execution": False,
        },
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def read_members(path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            relative = strip_root(member.name)
            if not relative:
                continue
            extracted = handle.extractfile(member)
            result[relative] = extracted.read() if extracted else b""
    return result


def relevant_members(
    members: dict[str, bytes], selected_paths: list[str], names: set[str], *, requirements: bool = False
) -> list[dict[str, str]]:
    rows = []
    for path, payload in members.items():
        basename = PurePosixPath(path).name.lower()
        if basename not in names and not (requirements and basename.startswith("requirements")):
            continue
        directory = PurePosixPath(path).parent.as_posix()
        if directory == ".":
            relevant = True
        else:
            relevant = any(candidate == directory or candidate.startswith(directory + "/") for candidate in selected_paths)
        if relevant:
            rows.append({"path": path, "sha256": hashlib.sha256(payload).hexdigest()})
    return sorted(rows, key=lambda row: row["path"])


def derive_selected_verifier_receipt(path: str, payload: bytes) -> dict[str, str] | None:
    text = payload.decode("utf-8", errors="replace")
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".py" and re.search(r"(?:^|\n)\s*(?:import\s+unittest|from\s+unittest\s+import\s+)", text):
        return runtime_receipt(
            "python_unittest_selected_file", path, ".", f"python -m unittest {path}", payload
        )
    if suffix == ".py":
        for match in PYTEST_RUNNABLE.finditer(text):
            declared_path = match.group(1).rstrip(".,;:)")
            if path == declared_path or path.endswith("/" + declared_path):
                prefix = path[: -(len(declared_path) + 1)] if path != declared_path else "."
                return runtime_receipt(
                    "selected_file_embedded_pytest_command",
                    path,
                    prefix or ".",
                    f"python -m pytest {declared_path}",
                    payload,
                )
    if suffix in {".mjs", ".js", ".cjs"}:
        imports_node = bool(re.search(r"(?:from\s+|import\s*\()['\"]node:", text))
        self_checks = "throw new Error" in text or "process.exitCode" in text
        if imports_node and self_checks and not THIRD_PARTY_JS_IMPORT.search(text):
            return runtime_receipt(
                "self_contained_node_builtin_selected_file", path, ".", f"node {path}", payload
            )
    return None


def runtime_receipt(kind: str, path: str, working_directory: str, command: str, payload: bytes) -> dict[str, str]:
    row = {
        "kind": kind,
        "source_path": path,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "working_directory": working_directory,
        "command": command,
        "derivation": "selected_verifier_runtime_convention",
    }
    row["receipt_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return row


def unique_receipts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        {str(row.get("receipt_sha256")): row for row in rows if row.get("receipt_sha256")}.values(),
        key=lambda row: (str(row.get("kind")), str(row.get("source_path")), str(row.get("command"))),
    )


def strip_root(name: str) -> str:
    parts = PurePosixPath(name).parts
    return PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else ""


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "observations", "residual_indices",
        "parent_target_or_evaluator_executions", "candidate_or_control_calls",
        "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
