#!/usr/bin/env python3
"""Acquire and statically classify exact sdist-only evaluator dependencies."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_untrusted_build_preflight_v1"
STATE = "PROSPECTIVE_EXACT_SDIST_STATIC_RISK_CLASSIFICATION_V1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_untrusted_build_preflight.json"
DANGEROUS_TOKENS = {"subprocess", "popen", "system", "socket", "urlopen", "requests", "curl", "wget", "ctypes", "eval", "exec"}


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
    print(json.dumps({key: report.get(key) for key in ("trigger_state", "state", "faults", "execution_performed", "risk_class", "network_downloads", "source_build_executions", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def preflight(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != POLICY or cfg.get("state") != STATE:
        faults.append("policy_or_state_invalid")
    for key, expected in (("owner", Path(__file__).resolve()), ("audit_owner", ROOT / "scripts/theseus_vcm_untrusted_build_preflight_audit.py")):
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
    producer = sources.get("immutable_resolution_v3", {})
    audit = sources.get("immutable_resolution_audit_v3", {})
    task13 = next((row for row in p2a.dicts(producer.get("rows")) if int(row.get("index") or 0) == 13), {})
    if producer.get("trigger_state") != "GREEN" or producer.get("qualified_task_count") != 5 or task13.get("disposition") != "INCONCLUSIVE_EXPERIMENT_DEPENDENCY_RESOLUTION" or "mock-open==1.4.0" not in str(p2a.mapping(task13.get("receipt")).get("stderr") or ""):
        faults.append("task13_sdist_wall_binding_invalid")
    if audit.get("trigger_state") != "GREEN" or audit.get("qualified_task_count") != 5 or audit.get("inconclusive_task_count") != 1:
        faults.append("immutable_resolution_audit_binding_invalid")
    package = p2a.mapping(cfg.get("package"))
    if package.get("name") != "mock-open" or package.get("version") != "1.4.0" or package.get("sha256") != "c3ecb6b8c32a5899a4f5bf4495083b598b520c698bba00e1ce2ace6e9c239100" or not str(package.get("url") or "").startswith("https://files.pythonhosted.org/"):
        faults.append("package_binding_invalid")
    store = p2a.resolve(str(cfg.get("retained_sdist") or ""))
    if store != (ROOT / "runtime/vcm_evaluator/dependency_store/sdist/mock-open-1.4.0.tar.gz").resolve():
        faults.append("retained_sdist_path_invalid")
    allowed = {"exact_pypi_sdist_download_authorized", "static_archive_inspection_authorized", "content_addressed_sdist_retention_authorized"}
    for key, value in p2a.mapping(cfg.get("authority")).items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")
    return cfg, {"sources": sources, "task13": task13, "store": store}, sorted(set(faults))


def preflight_report(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, _, faults = preflight(path)
    return finish(cfg, path, faults, {}, execution=False, downloads=0)


def execute(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, bound, faults = preflight(path)
    if faults:
        return finish(cfg, path, faults, {}, execution=False, downloads=0)
    limits = p2a.mapping(cfg.get("limits"))
    if shutil.disk_usage(ROOT).free - int(limits.get("maximum_download_bytes") or 0) < int(limits.get("minimum_free_bytes_after_execution") or 0):
        return finish(cfg, path, ["free_space_reserve_preflight_boundary_hit"], {}, execution=False, downloads=0)
    store: Path = bound["store"]
    if store.exists():
        return finish(cfg, path, ["retained_sdist_already_exists"], {}, execution=False, downloads=0)
    package = p2a.mapping(cfg.get("package"))
    receipt: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-sdist-preflight-", dir="/private/tmp") as raw:
        downloaded = Path(raw) / "package.tar.gz"
        request = urllib.request.Request(str(package["url"]), headers={"User-Agent": "Project-Theseus-VCM-Instrument/1"})
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(request, timeout=int(limits["network_timeout_seconds"])) as response, downloaded.open("wb") as handle:
            while chunk := response.read(65536):
                size += len(chunk)
                if size > int(limits["maximum_download_bytes"]):
                    faults.append("sdist_download_size_boundary_hit")
                    break
                digest.update(chunk)
                handle.write(chunk)
        receipt["download"] = {"url": package["url"], "bytes": size, "sha256": digest.hexdigest()}
        if digest.hexdigest() != package["sha256"]:
            faults.append("sdist_sha256_mismatch")
        if not faults:
            inspection, inspection_faults = inspect_sdist(downloaded, limits)
            receipt["inspection"] = inspection
            faults.extend(inspection_faults)
        if not faults:
            store.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, store)
    receipt["retained_sdist"] = {"path": p2a.rel(store), "bytes": store.stat().st_size, "sha256": p2a.sha256_file(store)} if store.is_file() else {}
    return finish(cfg, path, faults, receipt, execution=True, downloads=1)


def inspect_sdist(path: Path, limits: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    rows: list[dict[str, Any]] = []
    root_names: set[str] = set()
    setup_payload = b""
    pyproject_payload = b""
    license_paths: list[str] = []
    with tarfile.open(path, "r:gz") as handle:
        for member in handle.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                faults.append("unsafe_archive_member_path")
                continue
            root_names.add(pure.parts[0])
            if not (member.isfile() or member.isdir()):
                faults.append(f"unsupported_archive_member_type:{member.name}")
                continue
            if not member.isfile():
                continue
            if member.size > int(limits["maximum_single_member_bytes"]):
                faults.append(f"single_member_size_boundary_hit:{member.name}")
                continue
            extracted = handle.extractfile(member)
            payload = extracted.read() if extracted else b""
            relative = PurePosixPath(*pure.parts[1:]).as_posix()
            rows.append({"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            if relative == "setup.py":
                setup_payload = payload
            elif relative == "pyproject.toml":
                pyproject_payload = payload
            if PurePosixPath(relative).name.lower().startswith(("license", "copying")):
                license_paths.append(relative)
    if len(root_names) != 1:
        faults.append("archive_root_ambiguous")
    if len(rows) > int(limits["maximum_member_count"]) or sum(row["bytes"] for row in rows) > int(limits["maximum_uncompressed_bytes"]):
        faults.append("archive_expansion_boundary_hit")
    analysis = analyze_setup(setup_payload)
    if not setup_payload:
        faults.append("setup_py_missing")
    if analysis["parse_fault"]:
        faults.append("setup_py_parse_failed")
    if analysis["dangerous_tokens"]:
        faults.append("setup_py_static_dangerous_token")
    if not license_paths:
        faults.append("license_file_missing")
    risk_class = "LOW_COMPLEXITY_LEGACY_SETUP_PY_ELIGIBLE_FOR_NETWORK_DENIED_SANDBOX_CANARY" if not faults and not pyproject_payload else "INCONCLUSIVE_UNTRUSTED_BUILD_RISK"
    inspection = {"archive_root_count": len(root_names), "regular_file_count": len(rows), "regular_file_bytes": sum(row["bytes"] for row in rows), "member_receipts": sorted(rows, key=lambda row: row["path"]), "member_receipts_sha256": hashlib.sha256(json.dumps(sorted(rows, key=lambda row: row["path"]), sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "setup_py": analysis, "pyproject_present": bool(pyproject_payload), "license_paths": sorted(license_paths), "risk_class": risk_class}
    return inspection, sorted(set(faults))


def analyze_setup(payload: bytes) -> dict[str, Any]:
    text = payload.decode("utf-8", "replace")
    imports: set[str] = set()
    calls: list[str] = []
    parse_fault = ""
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(str(node.module or ""))
            elif isinstance(node, ast.Call):
                calls.append(call_name(node.func))
    except SyntaxError as exc:
        parse_fault = f"{exc.__class__.__name__}:{exc.lineno}"
    lowered = text.lower()
    dangerous = sorted(token for token in DANGEROUS_TOKENS if token in lowered)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "imports": sorted(imports), "calls": sorted(set(filter(None, calls))), "dangerous_tokens": dangerous, "parse_fault": parse_fault}


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{call_name(node.value)}.{node.attr}".strip(".")
    return ""


def finish(cfg: dict[str, Any], path: Path, faults: list[str], receipt: dict[str, Any], *, execution: bool, downloads: int) -> dict[str, Any]:
    risk = str(p2a.mapping(receipt.get("inspection")).get("risk_class") or "")
    return {"policy": POLICY, "created_utc": p2a.now(), "trigger_state": "RED" if faults else ("GREEN" if execution else "PAUSED"), "state": "EXACT_SDIST_STATIC_PREFLIGHT_QUALIFIED" if execution and not faults else ("READY_FOR_EXACT_SDIST_STATIC_PREFLIGHT" if not faults else "EXACT_SDIST_STATIC_PREFLIGHT_FAILED"), "faults": sorted(set(faults)), "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}, "execution_performed": execution, "risk_class": risk, "receipt": receipt, "free_bytes_after": shutil.disk_usage(ROOT).free, "network_downloads": downloads, "source_build_executions": 0, "package_installations": 0, "repository_runner_executions": 0, "parent_target_or_evaluator_executions": 0, "candidate_or_control_calls": 0, "external_reference_calls": 0, "teacher_calls": 0, "next_authorized_boundary": "prospectively_seal_network_denied_sandbox_wheel_build_canary" if risk.startswith("LOW_COMPLEXITY") and not faults else "none", "maximum_inference": cfg.get("maximum_inference")}


if __name__ == "__main__":
    raise SystemExit(main())
