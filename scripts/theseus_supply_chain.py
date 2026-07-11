"""AI bill of materials and derivative invalidation for the Theseus registry."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(_canonical(value).encode("utf-8"))


def _artifact_id(kind: str, identity: str) -> str:
    return f"aibom:{kind}:{hashlib.sha256(identity.encode('utf-8', errors='replace')).hexdigest()[:24]}"


def _file_identity(root: Path, raw_path: str, *, requested_by: str, kind: str) -> dict[str, Any]:
    path = Path(raw_path)
    path = path if path.is_absolute() else root / path
    exists = path.exists()
    is_file = exists and path.is_file()
    observed_hash = ""
    size = None
    if is_file:
        try:
            payload = path.read_bytes()
            observed_hash = _digest_bytes(payload)
            size = len(payload)
        except OSError:
            exists = False
    identity = str(path.relative_to(root)).replace("\\", "/") if path.is_relative_to(root) else str(path)
    if exists:
        identity_state = "observed"
    elif kind == "derived_evidence":
        identity_state = "not_materialized"
    else:
        identity_state = "missing"
    return {
        "record_type": "aibom_artifact",
        "artifact_id": _artifact_id(kind, identity),
        "artifact_kind": kind,
        "requested_identity": {
            "locator": raw_path,
            "requested_by": requested_by,
            "constraint": "exact_registry_locator",
        },
        "resolved_identity": {
            "locator": identity,
            "exists": exists,
            "kind": "file" if is_file else ("directory" if exists else "missing"),
        },
        "observed_identity": {
            "locator": identity,
            "sha256": observed_hash,
            "bytes": size,
        },
        "identity_state": identity_state,
        "signature_state": "not_configured",
        "advisory_state": "not_checked_offline",
        "license_state": "registry_or_source_metadata_required",
    }


def _python_imports(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _python_dependency(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    origin = str(spec.origin or "") if spec else ""
    version = "stdlib_or_unknown"
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass
    observed_hash = ""
    if origin and Path(origin).is_file():
        try:
            observed_hash = _digest_bytes(Path(origin).read_bytes())
        except OSError:
            pass
    return {
        "record_type": "aibom_artifact",
        "artifact_id": _artifact_id("python_dependency", name),
        "artifact_kind": "python_dependency",
        "requested_identity": {"module": name, "constraint": "imported_by_registered_source"},
        "resolved_identity": {"module": name, "version": version, "origin": origin},
        "observed_identity": {"module": name, "version": version, "origin": origin, "origin_sha256": observed_hash},
        "identity_state": "observed" if spec else "unavailable_current_environment",
        "signature_state": "not_configured",
        "advisory_state": "not_checked_offline",
        "license_state": "package_metadata_required",
    }


def _surface_bundle(root: Path, surface_id: str, rows: list[dict[str, Any]], artifact_type: str) -> dict[str, Any]:
    members = []
    for row in sorted(rows, key=lambda item: str(item.get("path") or "")):
        raw = str(row.get("path") or "")
        path = root / raw
        content_hash = ""
        if row.get("kind") == "file" and path.is_file():
            try:
                content_hash = _digest_bytes(path.read_bytes())
            except OSError:
                content_hash = "unreadable"
        members.append(
            {
                "path": raw,
                "kind": row.get("kind"),
                "bytes": row.get("bytes"),
                "content_hash": content_hash,
            }
        )
    bundle_identity = _digest(members)
    return {
        "record_type": "aibom_artifact",
        "artifact_id": _artifact_id("surface_bundle", surface_id),
        "artifact_kind": "surface_bundle",
        "artifact_domain": artifact_type,
        "requested_identity": {"surface_id": surface_id, "constraint": "all_registry_owned_members"},
        "resolved_identity": {"surface_id": surface_id, "member_count": len(members)},
        "observed_identity": {
            "surface_id": surface_id,
            "member_count": len(members),
            "member_content_merkle_root": bundle_identity,
            "total_bytes": sum(int(row.get("bytes") or 0) for row in members),
        },
        "identity_state": "observed",
        "signature_state": "not_configured",
        "advisory_state": "not_checked_offline",
        "license_state": "member_metadata_and_project_license",
    }


def descendant_invalidation(
    artifact_ids: set[str],
    dependencies: list[dict[str, str]],
    invalid_roots: set[str],
) -> dict[str, Any]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for edge in dependencies:
        parent = str(edge.get("dependency_artifact_id") or "")
        child = str(edge.get("dependent_artifact_id") or "")
        if parent and child:
            reverse[parent].add(child)
    invalidated = set(invalid_roots)
    queue: deque[str] = deque(sorted(invalid_roots))
    while queue:
        current = queue.popleft()
        for child in sorted(reverse.get(current, set())):
            if child not in invalidated:
                invalidated.add(child)
                queue.append(child)
    unknown = sorted(invalidated - artifact_ids)
    return {
        "record_type": "derivative_invalidation_record",
        "invalid_roots": sorted(invalid_roots),
        "invalidated_artifact_ids": sorted(invalidated & artifact_ids),
        "unknown_artifact_ids": unknown,
        "closure_complete": not unknown,
    }


def build_aibom(root: Path, policy: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    dependencies: list[dict[str, str]] = []
    implementation_artifacts: dict[str, str] = {}
    entries_by_surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if isinstance(entry, dict) and entry.get("surface_id"):
            entries_by_surface[str(entry["surface_id"])].append(entry)
    surface_bundle_ids: dict[str, str] = {}
    for surface in policy.get("surfaces", []) if isinstance(policy.get("surfaces"), list) else []:
        if not isinstance(surface, dict) or not surface.get("id"):
            continue
        surface_id = str(surface["id"])
        row = _surface_bundle(
            root,
            surface_id,
            entries_by_surface.get(surface_id, []),
            str(surface.get("artifact_type") or "unknown"),
        )
        artifacts[row["artifact_id"]] = row
        surface_bundle_ids[surface_id] = str(row["artifact_id"])
    for implementation in policy.get("implementations", []) if isinstance(policy.get("implementations"), list) else []:
        if not isinstance(implementation, dict):
            continue
        impl_id = str(implementation.get("id") or "")
        if not impl_id:
            continue
        impl_artifact_id = _artifact_id("implementation", impl_id)
        implementation_artifacts[impl_id] = impl_artifact_id
        artifacts[impl_artifact_id] = {
            "record_type": "aibom_artifact",
            "artifact_id": impl_artifact_id,
            "artifact_kind": "implementation",
            "requested_identity": {"implementation_id": impl_id, "constraint": "registry_exact_id"},
            "resolved_identity": {
                "implementation_id": impl_id,
                "abstraction_id": implementation.get("abstraction_id"),
                "backend": implementation.get("backend"),
                "status": implementation.get("status"),
            },
            "observed_identity": {
                "implementation_id": impl_id,
                "canonical_entrypoint": implementation.get("canonical_entrypoint"),
                "registry_payload_sha256": _digest(implementation),
            },
            "identity_state": "observed",
            "signature_state": "not_configured",
            "advisory_state": "not_checked_offline",
            "license_state": "project_license",
        }
        surface_id = str(implementation.get("surface_id") or "")
        if surface_id in surface_bundle_ids:
            dependencies.append(
                {
                    "dependency_artifact_id": surface_bundle_ids[surface_id],
                    "dependent_artifact_id": impl_artifact_id,
                    "relation": "surface_content_realizes_implementation",
                }
            )
        paths = [implementation.get("canonical_entrypoint"), *(implementation.get("dependencies") or [])]
        for raw in sorted({str(value) for value in paths if value}):
            suffix = Path(raw).suffix.lower()
            kind = "configuration" if suffix in {".json", ".toml", ".yaml", ".yml"} else "code"
            row = _file_identity(root, raw, requested_by=impl_id, kind=kind)
            artifacts[row["artifact_id"]] = row
            dependencies.append(
                {
                    "dependency_artifact_id": row["artifact_id"],
                    "dependent_artifact_id": impl_artifact_id,
                    "relation": "required_by_implementation",
                }
            )
        for raw in implementation.get("evidence_outputs", []) or []:
            row = _file_identity(root, str(raw), requested_by=impl_id, kind="derived_evidence")
            artifacts[row["artifact_id"]] = row
            dependencies.append(
                {
                    "dependency_artifact_id": impl_artifact_id,
                    "dependent_artifact_id": row["artifact_id"],
                    "relation": "derives_evidence",
                }
            )
    registered_python = {
        root / str(row.get("path"))
        for row in entries
        if row.get("kind") == "file" and str(row.get("path") or "").endswith(".py")
    }
    imports = set()
    for path in registered_python:
        imports.update(_python_imports(path))
    for name in sorted(imports):
        row = _python_dependency(name)
        artifacts[row["artifact_id"]] = row
    runtime = {
        "record_type": "aibom_artifact",
        "artifact_id": _artifact_id("runtime", "local-python-platform"),
        "artifact_kind": "runtime_hardware_profile",
        "requested_identity": {"profile": "local_registry_materialization"},
        "resolved_identity": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
        "observed_identity": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "identity_state": "observed",
        "signature_state": "not_configured",
        "advisory_state": "not_checked_offline",
        "license_state": "not_applicable",
    }
    artifacts[runtime["artifact_id"]] = runtime
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    missing = {artifact_id for artifact_id, row in artifacts.items() if row.get("identity_state") == "missing"}
    invalidation = descendant_invalidation(set(artifacts), dependencies, missing)
    artifact_rows = sorted(artifacts.values(), key=lambda row: str(row["artifact_id"]))
    identity_artifacts = [
        row
        for row in artifact_rows
        if row.get("artifact_kind") != "derived_evidence"
        and not (row.get("artifact_kind") == "surface_bundle" and row.get("artifact_domain") == "generated_artifact")
    ]
    identity_artifact_ids = {str(row["artifact_id"]) for row in identity_artifacts}
    identity_dependencies = [
        row
        for row in dependencies
        if row.get("dependency_artifact_id") in identity_artifact_ids
        and row.get("dependent_artifact_id") in identity_artifact_ids
    ]
    build_basis = {
        "git_revision": revision,
        "policy_sha256": _digest(policy),
        "artifact_identities": [
            {"artifact_id": row["artifact_id"], "observed_identity": row["observed_identity"]}
            for row in identity_artifacts
        ],
        "dependencies": sorted(identity_dependencies, key=lambda row: _canonical(row)),
    }
    observed_count = sum(row.get("identity_state") == "observed" for row in artifact_rows)
    unavailable_count = sum(row.get("identity_state") == "unavailable_current_environment" for row in artifact_rows)
    not_materialized_count = sum(row.get("identity_state") == "not_materialized" for row in artifact_rows)
    domain_counts: dict[str, int] = defaultdict(int)
    for row in artifact_rows:
        domain_counts[str(row.get("artifact_domain") or row.get("artifact_kind") or "unknown")] += 1
    return {
        "policy": "project_theseus_aibom_v1",
        "record_type": "ai_bill_of_materials",
        "schema_version": "1.0.0",
        "git_revision": revision,
        "build_identity": _digest(build_basis),
        "artifacts": artifact_rows,
        "dependencies": sorted(dependencies, key=lambda row: _canonical(row)),
        "derivative_invalidation": invalidation,
        "summary": {
            "artifact_count": len(artifact_rows),
            "observed_identity_count": observed_count,
            "missing_identity_count": len(missing),
            "unavailable_current_environment_count": unavailable_count,
            "not_materialized_evidence_count": not_materialized_count,
            "dependency_edge_count": len(dependencies),
            "invalidated_artifact_count": len(invalidation["invalidated_artifact_ids"]),
            "requested_resolved_observed_distinct": True,
            "artifact_domain_counts": dict(sorted(domain_counts.items())),
            "build_identity_artifact_count": len(identity_artifacts),
            "build_identity_excludes_self_referential_derived_evidence": True,
        },
        "attestation": {
            "kind": "local_content_bound_materialization",
            "build_identity": _digest(build_basis),
            "cryptographic_signature": "not_configured",
            "reproducible_second_build_match": "not_run",
            "advisory_snapshot": "not_checked_offline",
            "release_admission_claimed": False,
        },
        "claims": {
            "requested_resolved_observed_identity": True,
            "content_bound_local_materialization": True,
            "derivative_invalidation_closure": invalidation["closure_complete"],
            "signed_supply_chain": False,
            "advisory_freshness": False,
            "reproducible_build": False,
            "weight_custody": False,
        },
        "external_inference_calls": 0,
    }
