#!/usr/bin/env python3
"""Materialize exact VCM parent/head repository closures without executing them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import sys
import tempfile
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_d1_source_materialization as d1  # noqa: E402

POLICY = "project_theseus_vcm_repository_closure_materialization_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_repository_closure_materialization.json"


class HostStorageBoundary(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = preflight(config, config_path)
    if args.execute and report["execution_authorized"] is True:
        report = execute(config, config_path)
    out = resolve(args.out or config["report"])
    write_json(out, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config: dict[str, Any], config_path: Path, panel_override: dict[str, Any] | None = None) -> dict[str, Any]:
    faults = validate_config(config)
    for row in config.get("bindings", []):
        path = resolve(str(row.get("path") or ""))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            faults.append(f"binding_invalid:{row.get('id')}")
    panel_path = resolve(str(config.get("source_panel") or ""))
    panel = panel_override if panel_override is not None else read_json(panel_path) if panel_path.is_file() else {}
    if panel.get("trigger_state") != "GREEN" or panel.get("source_panel_admitted") is not True:
        faults.append("source_panel_not_admitted")
    registry = transform_panel(panel)
    faults.extend(d1.audit_registry(registry) if registry else ["transformed_registry_missing"])
    storage = storage_receipt(config)
    if storage["available_bytes"] <= storage["minimum_free_bytes_after_download"]:
        faults.append("host_storage_reserve_unavailable")
    return {
        "policy": POLICY,
        "created_utc": d1.successor.now(),
        "trigger_state": "RED" if faults else "GREEN",
        "state": "CONTRACT_INVALID" if faults else "EXACT_REPOSITORY_CLOSURE_FETCH_READY",
        "faults": sorted(set(faults)),
        "execution_authorized": not faults,
        "config": identity(config_path),
        "source_panel": identity(panel_path) if panel_override is None else {"path": relative(panel_path), "sha256": d1.stable_hash(panel)},
        "transformed_registry_sha256": d1.stable_hash(registry) if registry else "",
        "task_count": len(registry.get("tasks", [])),
        "planned_archive_count": len(registry.get("tasks", [])) * 2,
        "storage": storage,
        "archive_artifacts": 0,
        "network_fetches": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def execute(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path)
    if before["execution_authorized"] is not True:
        return before
    panel = read_json(resolve(config["source_panel"]))
    registry = transform_panel(panel)
    transport = StorageSafeDownloader(config)
    adapted = {
        **config,
        "policy": d1.POLICY,
        "state": "BOUND_BEFORE_D1_SOURCE_REGISTRY_EXISTS",
        "allowed_transport_origin": "https://codeload.github.com",
    }
    try:
        result = materialize_closures(adapted, registry, transport)
    except HostStorageBoundary as exc:
        return {
            **before,
            "trigger_state": "PAUSED",
            "state": "HOST_STORAGE_PHYSICAL_BOUNDARY_HIT_RETAIN_COMPLETED_ARCHIVES",
            "faults": [str(exc)],
            "execution_authorized": False,
            "network_fetches": transport.completed_downloads,
            "downloaded_bytes": transport.completed_bytes,
            "physical_boundary_hit": True,
        }
    except (OSError, tarfile.TarError, urllib.error.URLError, ValueError) as exc:
        return {
            **before,
            "trigger_state": "PAUSED",
            "state": "RETRYABLE_TRANSPORT_OR_ARCHIVE_ERROR_RETAIN_COMPLETED_DERIVATIVES",
            "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
            "execution_authorized": False,
            "network_fetches": transport.completed_downloads,
            "downloaded_bytes": transport.completed_bytes,
            "physical_boundary_hit": False,
        }
    faults = list(result.get("faults", []))
    return {
        **before,
        "created_utc": d1.successor.now(),
        "trigger_state": result.get("trigger_state"),
        "state": "EXACT_PARENT_HEAD_REPOSITORY_CLOSURES_MATERIALIZED" if not faults else "REPOSITORY_CLOSURE_MATERIALIZATION_INVALID",
        "faults": faults,
        "execution_authorized": False,
        "archive_artifacts": result.get("archive_artifacts", 0),
        "network_fetches": result.get("network_fetches", 0),
        "downloaded_bytes": transport.completed_bytes,
        "tasks": result.get("tasks", []),
        "physical_boundary_hit": False,
        "parent_target_or_evaluator_executions": 0,
    }


def materialize_closures(config: dict[str, Any], registry: dict[str, Any], downloader: Callable[[str, Path], None]) -> dict[str, Any]:
    faults = d1.audit_registry(registry)
    rows = []
    archive_root = resolve(config["archive_root"])
    report_root = resolve(config["sanitization_report_root"])
    reserve = int(config["physical_storage_policy"]["minimum_free_bytes_after_download"])
    network_fetches = 0
    for task in registry["tasks"]:
        task_row = {"campaign_index": task["campaign_index"], "repository": task["repository"], "selection_digest": task["selection_digest"], "artifacts": []}
        for plan in d1.artifact_plan(task, config):
            upstream = archive_root / plan["upstream_name"]
            normalized = archive_root / plan["normalized_name"]
            sanitation_path = report_root / plan["sanitization_name"]
            if normalized.is_file() and sanitation_path.is_file():
                sanitation = read_json(sanitation_path)
                artifact_faults = d1.audit_materialized_artifact(task, plan, normalized, sanitation)
                upstream_sha = str(sanitation.get("input", {}).get("sha256") or "")
            else:
                if not upstream.is_file():
                    downloader(plan["url"], upstream); network_fetches += 1
                upstream_sha = sha256_file(upstream)
                if shutil.disk_usage(archive_root).free - upstream.stat().st_size < reserve:
                    raise HostStorageBoundary("normalization_free_space_reserve_boundary_hit")
                sanitation = d1.sanitizer.sanitize(upstream, normalized)
                write_json(sanitation_path, sanitation)
                artifact_faults = d1.audit_materialized_artifact(task, plan, normalized, sanitation)
                if artifact_faults:
                    faults.extend(artifact_faults)
                else:
                    upstream.unlink(missing_ok=True)
            task_row["artifacts"].append({
                **plan,
                "upstream_retained": upstream.is_file(),
                "upstream_sha256": upstream_sha,
                "normalized": relative(normalized),
                "normalized_sha256": sha256_file(normalized),
                "sanitization_report": relative(sanitation_path),
                "sanitization_report_sha256": sha256_file(sanitation_path),
                "source_archive_root": sanitation.get("source_archive_root"),
                "faults": artifact_faults,
            })
        rows.append(task_row)
    count = sum(len(row["artifacts"]) for row in rows)
    if count != len(registry["tasks"]) * 2: faults.append("materialized_archive_count_invalid")
    return {"trigger_state": "GREEN" if not faults else "RED", "faults": sorted(set(faults)), "archive_artifacts": count, "network_fetches": network_fetches, "tasks": rows}


def transform_panel(panel: dict[str, Any]) -> dict[str, Any]:
    tasks = []
    for row in panel.get("assembled_rows", []):
        archives = row.get("archives", {})
        parent = member_paths(archives, "parent_source") | member_paths(archives, "parent_verifier")
        target = member_paths(archives, "target_source") | member_paths(archives, "target_verifier")
        selected = sorted(set(row.get("selected_source_paths", [])) | set(row.get("selected_verifier_paths", [])))
        changed = []
        for path in selected:
            in_parent, in_target = path in parent, path in target
            status = "modified" if in_parent and in_target else "added" if in_target else "removed" if in_parent else "changed"
            changed.append({"filename": path, "status": status, "previous_filename": ""})
        tasks.append({
            "campaign_index": int(row.get("index") or 0),
            "repository": row.get("repository"),
            "parent_revision": row.get("base_revision"),
            "target_revision": row.get("head_revision"),
            "selection_digest": row.get("opaque_source_id"),
            "changed_paths": selected,
            "changed_files": changed,
        })
    return {
        "policy": "project_theseus_d1_online_source_registry_v1",
        "state": "FIXED_BEFORE_ARCHIVE_FETCH_PARENT_TARGET_ORACLE_EVALUATOR_OR_CANDIDATE_EXECUTION",
        "campaign_id": "theseus_d1_cognitive_compilation_fresh_qualification_v1",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "task_count": len(tasks),
        "tasks": tasks,
        "boundaries": {key: 0 for key in ("archive_fetches", "parent_target_oracle_or_evaluator_executions", "candidate_or_control_calls", "external_inference_calls", "teacher_calls", "training_rows_written")},
        "replacement_after_membership_freeze": False,
    }


def member_paths(archives: dict[str, Any], role: str) -> set[str]:
    return {str(row.get("path") or "") for row in archives.get(role, {}).get("members", []) if str(row.get("path") or "").lower() not in d1.LICENSE_NAMES}


class StorageSafeDownloader:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.completed_bytes = 0
        self.completed_downloads = 0

    def __call__(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        maximum = int(self.config["physical_storage_policy"]["maximum_upstream_archive_bytes"])
        reserve = int(self.config["physical_storage_policy"]["minimum_free_bytes_after_download"])
        request = urllib.request.Request(url, headers={"User-Agent": "Project-Theseus-VCM-Closure/1"})
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            size = 0
            try:
                tls = self.config["tls_ca_bundle"]
                context = ssl.create_default_context(cafile=str(tls["path"]))
                with urllib.request.urlopen(request, timeout=180, context=context) as response:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > maximum:
                            raise HostStorageBoundary("single_archive_physical_boundary_hit")
                        if shutil.disk_usage(destination.parent).free - len(chunk) < reserve:
                            raise HostStorageBoundary("host_free_space_reserve_boundary_hit")
                        handle.write(chunk)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, destination)
        self.completed_bytes += size
        self.completed_downloads += 1


def validate_config(config: dict[str, Any]) -> list[str]:
    faults = []
    if config.get("policy") != POLICY: faults.append("policy_invalid")
    if config.get("state") != "PROSPECTIVE_EXACT_REPOSITORY_CLOSURE_BEFORE_UNTRUSTED_EXECUTION": faults.append("state_invalid")
    authority = config.get("authority", {})
    for key, value in authority.items():
        expected = key == "exact_public_source_archive_retrieval_authorized"
        if value is not expected: faults.append(f"authority_invalid:{key}")
    storage = config.get("physical_storage_policy", {})
    if int(storage.get("minimum_free_bytes_after_download") or 0) < 8 * 1024**3: faults.append("storage_reserve_too_small")
    if int(storage.get("maximum_upstream_archive_bytes") or 0) <= 0: faults.append("archive_boundary_invalid")
    tls = config.get("tls_ca_bundle", {})
    tls_path = Path(str(tls.get("path") or ""))
    if not tls_path.is_file() or sha256_file(tls_path) != tls.get("sha256"): faults.append("tls_ca_bundle_identity_invalid")
    return faults


def storage_receipt(config: dict[str, Any]) -> dict[str, int]:
    root = resolve(config["archive_root"]); root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    return {"available_bytes": usage.free, "total_bytes": usage.total, **{k:int(v) for k,v in config["physical_storage_policy"].items()}}


def resolve(value: str | Path) -> Path: p=Path(value); return p if p.is_absolute() else ROOT/p
def relative(path: Path) -> str:
    try: return path.resolve().relative_to(ROOT).as_posix()
    except ValueError: return str(path)
def read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text())
def write_json(path: Path, value: dict[str, Any]) -> None: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def sha256_file(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
def identity(path: Path) -> dict[str,str]: return {"path":relative(path),"sha256":sha256_file(path)}
def summary(r: dict[str,Any]) -> dict[str,Any]: return {k:r.get(k) for k in ("trigger_state","state","execution_authorized","task_count","planned_archive_count","archive_artifacts","network_fetches","physical_boundary_hit","faults")}

if __name__ == "__main__": raise SystemExit(main())
