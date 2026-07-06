"""Artifact indexing, payload, and bundle helpers for the Theseus Hive node."""

from __future__ import annotations

import base64
import hashlib
import heapq
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hive_security
import viea_spine_records
from hive_node_common import ROOT, get_path, now, read_json
from hive_node_federation import join_token, shared_secret
from hive_node_identity import load_identity


def artifact_index(policy: dict[str, Any], *, limit: int = 200) -> dict[str, Any]:
    identity = load_identity(policy)
    max_items = max(1, min(limit, int(get_path(policy, ["artifact_sync", "max_index_items"], 500))))
    roots = artifact_roots(policy)
    rows: list[dict[str, Any]] = []
    candidates, scan = recent_artifact_candidates(policy, roots, max_items=max_items)
    citation = viea_spine_records.materialized_artifact_citation("hive_artifact_index", artifact_path="reports/hive_artifact_index")
    for path, stat in candidates:
        artifact_id = hashlib.sha256(artifact_rel_path(policy, path).encode("utf-8")).hexdigest()[:24]
        rows.append(
            {
                "artifact_id": artifact_id,
                "node_id": identity.get("node_id"),
                "node_name": identity.get("node_name"),
                "path": artifact_rel_path(policy, path),
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": sha256_file(path),
                "kind": classify_artifact_path(path),
                "viea_artifact_citation_id": citation.get("citation_id"),
            }
        )
    rows.sort(key=lambda row: str(row.get("modified_utc") or ""), reverse=True)
    return {
        "ok": True,
        "policy": "project_theseus_hive_artifact_index_v0",
        "created_utc": now(),
        "node_id": identity.get("node_id"),
        "artifacts": rows[:max_items],
        "scan": scan,
        "viea_artifact_citation": citation,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def recent_artifact_candidates(policy: dict[str, Any], roots: list[Path], *, max_items: int) -> tuple[list[tuple[Path, Any]], dict[str, Any]]:
    """Return recent candidate artifacts without hashing the whole reports tree."""
    max_scan_files = int(get_path(policy, ["artifact_sync", "max_index_scan_files"], 5000))
    max_scan_files = max(max_items, max_scan_files)
    heap: list[tuple[float, int, Path, Any]] = []
    seen: set[str] = set()
    scanned = 0
    matched = 0
    truncated = False
    sequence = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if scanned >= max_scan_files:
                truncated = True
                break
            try:
                if not path.is_file():
                    continue
                scanned += 1
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                if not indexable_artifact_path(policy, path):
                    continue
                if not safe_artifact_path(policy, path):
                    continue
                stat = path.stat()
            except OSError:
                continue
            matched += 1
            sequence += 1
            item = (float(stat.st_mtime), sequence, path, stat)
            if len(heap) < max_items:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        if truncated:
            break
    recent = heapq.nlargest(max_items, heap, key=lambda item: (item[0], item[1]))
    return [(path, stat) for _mtime, _seq, path, stat in recent], {
        "strategy": "recent_first_bounded_scan_then_sha256_returned_files",
        "max_items": max_items,
        "max_scan_files": max_scan_files,
        "scanned_files": scanned,
        "matched_files": matched,
        "truncated": truncated,
    }

def read_artifact_payload(policy: dict[str, Any], rel_path: str) -> dict[str, Any]:
    path = resolve_artifact_path(policy, rel_path)
    if not path:
        return {"ok": False, "error": "artifact_path_not_allowed", "path": rel_path}
    if not path.exists() or not path.is_file():
        return {"ok": False, "error": "artifact_missing", "path": rel_path}
    max_bytes = int(get_path(policy, ["artifact_sync", "max_artifact_bytes"], 10 * 1024 * 1024))
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {"ok": False, "error": "artifact_read_failed", "message": str(exc), "path": rel_path}
    if len(data) > max_bytes:
        return {"ok": False, "error": "artifact_too_large", "size_bytes": len(data), "max_bytes": max_bytes, "path": rel_path}
    body = {
        "ok": True,
        "policy": "project_theseus_hive_artifact_payload_v0",
        "created_utc": now(),
        "path": artifact_rel_path(policy, path),
        "kind": classify_artifact_path(path),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "encoding": "base64",
        "content_b64": base64.b64encode(data).decode("ascii"),
        "viea_artifact_citation": viea_spine_records.materialized_artifact_citation(
            "hive_artifact_payload",
            artifact_path=artifact_rel_path(policy, path),
        ),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    body["artifact_signature"] = hive_security.sign_artifact(
        {key: body[key] for key in ["path", "kind", "size_bytes", "sha256"]},
        secret=join_token(policy) or shared_secret(policy),
        signer=str(load_identity(policy).get("node_id") or ""),
    )
    return body

def materialize_output_artifacts(policy: dict[str, Any], report: dict[str, Any], command: list[str]) -> list[dict[str, Any]]:
    payload = report.get("payload") if isinstance(report.get("payload"), dict) else {}
    requested = get_path(report, ["job", "output_artifacts"], [])
    if not isinstance(requested, list):
        requested = payload.get("output_artifacts") if isinstance(payload.get("output_artifacts"), list) else []
    actual_out = command_output_path(command)
    artifacts: list[dict[str, Any]] = []
    for artifact in requested:
        if not isinstance(artifact, dict):
            continue
        target = resolve_artifact_path(policy, str(artifact.get("path") or ""))
        if not target:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if actual_out and actual_out.exists() and actual_out.resolve() != target.resolve():
            try:
                shutil.copy2(actual_out, target)
            except OSError:
                pass
        if target.exists() and target.is_file():
            artifacts.append(artifact_metadata(policy, target, artifact_type=str(artifact.get("type") or "")))
    if actual_out and actual_out.exists() and safe_artifact_path(policy, actual_out):
        meta = artifact_metadata(policy, actual_out, artifact_type="command_report")
        if all(row.get("path") != meta.get("path") for row in artifacts):
            artifacts.append(meta)
        artifacts.extend(related_artifacts_from_report(policy, actual_out))
    return artifacts

def related_artifacts_from_report(policy: dict[str, Any], report_path: Path) -> list[dict[str, Any]]:
    report = read_json(report_path, {})
    related: list[str] = []
    for key_path in [
        ["telemetry", "model_path"],
        ["telemetry", "child_report_path"],
        ["child_report_path"],
    ]:
        value = get_path(report, key_path, "")
        if value:
            related.append(str(value))
    out: list[dict[str, Any]] = []
    for rel in related:
        path = resolve_artifact_path(policy, rel)
        if path and path.exists() and path.is_file():
            out.append(artifact_metadata(policy, path, artifact_type=classify_artifact_path(path)))
    return out

def bundle_result_artifacts(policy: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    max_total = int(get_path(policy, ["artifact_sync", "max_relay_bundle_bytes"], 2 * 1024 * 1024))
    total = 0
    bundled = []
    for artifact in result.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        payload = read_artifact_payload(policy, str(artifact.get("path") or ""))
        if not payload.get("ok"):
            continue
        size = int(payload.get("size_bytes") or 0)
        if total + size > max_total:
            continue
        total += size
        bundled.append(payload)
    return {
        **result,
        "artifact_bundle": {
            "policy": "project_theseus_hive_artifact_bundle_v0",
            "created_utc": now(),
            "total_bytes": total,
            "artifacts": bundled,
        },
    }

def command_output_path(command: list[str]) -> Path | None:
    for idx, part in enumerate(command[:-1]):
        if part == "--out":
            return (ROOT / command[idx + 1]).resolve()
    return None

def artifact_metadata(policy: dict[str, Any], path: Path, *, artifact_type: str = "") -> dict[str, Any]:
    stat = path.stat()
    artifact_path = artifact_rel_path(policy, path)
    citation = viea_spine_records.materialized_artifact_citation("hive_artifact_metadata", artifact_path=artifact_path)
    return {
        "path": artifact_path,
        "kind": artifact_type or classify_artifact_path(path),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "viea_artifact_citation_id": citation.get("citation_id"),
        "viea_artifact_citation": citation,
    }

def artifact_roots(policy: dict[str, Any]) -> list[Path]:
    configured = get_path(policy, ["artifact_sync", "allowed_roots"], ["reports/hive_chunks", "reports", "checkpoints/hive_promoted"])
    roots = []
    for item in configured if isinstance(configured, list) else []:
        roots.append(ROOT / str(item))
    return roots

def artifact_rel_path(policy: dict[str, Any], path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        pass
    for root in artifact_roots(policy):
        root_resolved = root.resolve()
        try:
            rel = resolved.relative_to(root_resolved)
        except ValueError:
            continue
        try:
            prefix = root.relative_to(ROOT)
        except ValueError:
            prefix = Path(root.name)
        return str(prefix / rel).replace("\\", "/")
    return str(path).replace("\\", "/")

def resolve_artifact_path(policy: dict[str, Any], rel_path: str) -> Path | None:
    if not rel_path or Path(rel_path).is_absolute() or ".." in Path(rel_path).parts:
        return None
    path = ROOT / rel_path
    return path if safe_artifact_path(policy, path) else None

def safe_artifact_path(policy: dict[str, Any], path: Path) -> bool:
    resolved = path.resolve()
    for root in artifact_roots(policy):
        root_resolved = root.resolve()
        try:
            if resolved.is_relative_to(root_resolved):
                return True
        except AttributeError:
            if str(resolved).startswith(str(root_resolved)):
                return True
    return False

def classify_artifact_path(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".dmg", ".pkg", ".msi", ".exe", ".appimage", ".deb", ".rpm", ".zip", ".tar.gz")):
        return "installer_package"
    if name.endswith((".ps1", ".cmd", ".sh", ".command")) and "dist" in str(path).replace("\\", "/").split("/"):
        return "installer_launcher"
    if name.endswith(".model.json") or "model" in name:
        return "model_json"
    if "chunk" in str(path).lower():
        return "worker_report"
    if "checkpoint" in str(path).lower():
        return "checkpoint_manifest"
    return "json_report"

def indexable_artifact_path(policy: dict[str, Any], path: Path) -> bool:
    suffixes = get_path(policy, ["artifact_sync", "index_file_suffixes"], [".json"])
    if not isinstance(suffixes, list):
        suffixes = [".json"]
    name = path.name.lower()
    rel = artifact_rel_path(policy, path).lower()
    if name.endswith(".artifact.json") or rel.startswith("reports/hive_artifact_inbox/"):
        return False
    return any(name.endswith(str(suffix).lower()) or rel.endswith(str(suffix).lower()) for suffix in suffixes)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
