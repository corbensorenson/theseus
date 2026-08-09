#!/usr/bin/env python3
"""Build one prequalified exact sdist inside a network-denied macOS sandbox."""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_sandbox_wheel_build_canary_v1"
STATE = "PROSPECTIVE_NETWORK_DENIED_EXACT_SDIST_WHEEL_BUILD_V1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_sandbox_wheel_build_canary.json"


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
    print(json.dumps({key: report.get(key) for key in ("trigger_state", "state", "faults", "execution_performed", "source_build_executions", "network_denied_builds", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def preflight(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != POLICY or cfg.get("state") != STATE:
        faults.append("policy_or_state_invalid")
    for key, expected in (("owner", Path(__file__).resolve()), ("audit_owner", ROOT / "scripts/theseus_vcm_sandbox_wheel_build_canary_audit.py")):
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
    preflight_report = sources.get("sdist_preflight", {})
    preflight_audit = sources.get("sdist_preflight_audit", {})
    expected_risk = "LOW_COMPLEXITY_LEGACY_SETUP_PY_ELIGIBLE_FOR_NETWORK_DENIED_SANDBOX_CANARY"
    if preflight_report.get("trigger_state") != "GREEN" or preflight_report.get("risk_class") != expected_risk or preflight_report.get("next_authorized_boundary") != "prospectively_seal_network_denied_sandbox_wheel_build_canary":
        faults.append("sdist_preflight_not_eligible")
    if preflight_audit.get("trigger_state") != "GREEN" or preflight_audit.get("risk_class") != expected_risk:
        faults.append("sdist_preflight_audit_not_green")
    tools: dict[str, str] = {}
    for name, raw in p2a.mapping(cfg.get("tools")).items():
        binding = p2a.mapping(raw)
        tool = p2a.resolve(str(binding.get("path") or ""))
        if not tool.is_file() or p2a.sha256_file(tool) != binding.get("sha256"):
            faults.append(f"tool_binding_invalid:{name}")
        tools[name] = str(tool)
    sdist = p2a.resolve(str(cfg.get("sdist") or ""))
    if not sdist.is_file() or p2a.sha256_file(sdist) != cfg.get("sdist_sha256"):
        faults.append("sdist_binding_invalid")
    wheel_store = p2a.resolve(str(cfg.get("wheel_store") or ""))
    if wheel_store != (ROOT / "runtime/vcm_evaluator/dependency_store/wheels").resolve():
        faults.append("wheel_store_invalid")
    allowed = {"disposable_build_tool_environment_authorized", "network_build_tool_wheel_sync_authorized", "single_exact_sdist_build_authorized", "network_denied_sandbox_build_authorized", "built_wheel_retention_authorized"}
    for key, value in p2a.mapping(cfg.get("authority")).items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")
    return cfg, {"sources": sources, "tools": tools, "sdist": sdist, "wheel_store": wheel_store}, sorted(set(faults))


def preflight_report(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, _, faults = preflight(path)
    return finish(cfg, path, faults, {}, execution=False)


def execute(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg, bound, faults = preflight(path)
    if faults:
        return finish(cfg, path, faults, {}, execution=False)
    limits = p2a.mapping(cfg.get("limits"))
    if shutil.disk_usage(ROOT).free - int(limits["maximum_temporary_bytes"]) < int(limits["minimum_free_bytes_after_execution"]):
        return finish(cfg, path, ["free_space_reserve_preflight_boundary_hit"], {}, execution=False)
    receipt: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-wheel-build-", dir="/private/tmp") as raw:
        work = Path(raw).resolve(); venv = work / "venv"; cache = work / "cache"; out = work / "out"; home = work / "home"; tmp = work / "tmp"
        for directory in (cache, out, home, tmp): directory.mkdir()
        env = {"HOME": str(home), "TMPDIR": str(tmp), "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "CI": "1", "NO_COLOR": "1", "UV_PYTHON_DOWNLOADS": "never", "UV_NO_CONFIG": "1"}
        uv = bound["tools"]["uv"]; python = bound["tools"]["python"]; sandbox = bound["tools"]["sandbox_exec"]
        receipt["venv"] = run([uv, "venv", str(venv), "--python", python, "--no-python-downloads", "--no-config"], work, env, limits)
        if receipt["venv"]["returncode"] != 0: faults.append("build_tool_venv_failed")
        tool_args = [uv, "pip", "install", "--python", str(venv / "bin/python"), "--no-build", "--default-index", "https://pypi.org/simple", "--cache-dir", str(cache), "--no-config", "--no-progress", "--color", "never", *p2a.strings(cfg.get("build_tool_requirements"))]
        if not faults:
            receipt["build_tool_sync"] = run(tool_args, work, env, limits)
            if receipt["build_tool_sync"]["returncode"] != 0: faults.append("build_tool_sync_failed")
        profile = "\n".join(["(version 1)", "(allow default)", "(deny network*)", f'(deny file-write* (require-not (subpath "{work}")))', '(allow file-write* (literal "/dev/null"))'])
        build_args = [sandbox, "-p", profile, uv, "build", str(bound["sdist"]), "--wheel", "--out-dir", str(out), "--python", str(venv / "bin/python"), "--no-build-isolation", "--offline", "--cache-dir", str(cache), "--no-config", "--no-progress", "--color", "never"]
        if not faults:
            receipt["sandbox_build"] = run(build_args, work, env, limits)
            if receipt["sandbox_build"]["returncode"] != 0: faults.append("network_denied_sandbox_build_failed")
        wheels = sorted(out.glob("*.whl"))
        if not faults and len(wheels) != 1: faults.append("built_wheel_denominator_invalid")
        if not faults:
            wheel_receipt, wheel_faults = inspect_wheel(wheels[0])
            receipt["wheel"] = wheel_receipt
            faults.extend(wheel_faults)
        if not faults:
            store: Path = bound["wheel_store"]; store.mkdir(parents=True, exist_ok=True); target = store / wheels[0].name
            if target.exists(): faults.append("retained_wheel_already_exists")
            else: shutil.copyfile(wheels[0], target); receipt["retained_wheel"] = {"path": p2a.rel(target), "bytes": target.stat().st_size, "sha256": p2a.sha256_file(target)}
    return finish(cfg, path, faults, receipt, execution=True)


def run(command: list[str], cwd: Path, env: dict[str, str], limits: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=int(limits["wall_seconds_per_command"]), check=False)
        boundary = False; reason = ""
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(command, 124, exc.stdout or b"", exc.stderr or b""); boundary = True; reason = "wall_timeout"
    stdout = completed.stdout or b""; stderr = completed.stderr or b""
    return {"command": command, "returncode": completed.returncode, "duration_ms": round((time.monotonic()-started)*1000,3), "boundary_hit": boundary, "boundary_reason": reason, "stdout": stdout.decode("utf-8","replace"), "stderr": stderr.decode("utf-8","replace"), "stdout_bytes": len(stdout), "stderr_bytes": len(stderr), "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest(), "stdout_complete": True, "stderr_complete": True, "project_selected_output_cap": None}


def inspect_wheel(path: Path) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []; rows = []; metadata = b""
    with zipfile.ZipFile(path) as wheel:
        for info in wheel.infolist():
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts: faults.append("unsafe_wheel_member")
            payload = wheel.read(info)
            rows.append({"path": info.filename, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            if info.filename.endswith(".dist-info/METADATA"): metadata = payload
            if info.filename.endswith((".so", ".dylib", ".dll", ".exe")): faults.append("native_or_executable_payload_present")
    parsed = email.parser.Parser().parsestr(metadata.decode("utf-8", "replace")) if metadata else None
    name = str(parsed.get("Name") or "") if parsed else ""; version = str(parsed.get("Version") or "") if parsed else ""
    if name.lower().replace("_","-") != "mock-open" or version != "1.4.0": faults.append("wheel_metadata_identity_invalid")
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": p2a.sha256_file(path), "member_count": len(rows), "member_receipts_sha256": hashlib.sha256(json.dumps(sorted(rows,key=lambda r:r["path"]),sort_keys=True,separators=(",",":")).encode()).hexdigest(), "metadata_name": name, "metadata_version": version, "pure_python": not any(row["path"].endswith((".so",".dylib",".dll",".exe")) for row in rows)}, sorted(set(faults))


def finish(cfg: dict[str, Any], path: Path, faults: list[str], receipt: dict[str, Any], *, execution: bool) -> dict[str, Any]:
    built = 1 if receipt.get("sandbox_build") else 0
    return {"policy": POLICY, "created_utc": p2a.now(), "trigger_state": "RED" if faults else ("GREEN" if execution else "PAUSED"), "state": "NETWORK_DENIED_EXACT_SDIST_WHEEL_BUILD_QUALIFIED" if execution and not faults else ("READY_FOR_NETWORK_DENIED_EXACT_SDIST_WHEEL_BUILD" if not faults else "NETWORK_DENIED_EXACT_SDIST_WHEEL_BUILD_FAILED"), "faults": sorted(set(faults)), "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}, "execution_performed": execution, "receipt": receipt, "source_build_executions": built, "network_denied_builds": built, "build_tool_package_installations": 2 if receipt.get("build_tool_sync") else 0, "evaluator_package_installations": 0, "repository_runner_executions": 0, "parent_target_or_evaluator_executions": 0, "candidate_or_control_calls": 0, "external_reference_calls": 0, "teacher_calls": 0, "next_authorized_boundary": "repair_task13_immutable_resolution_with_exact_retained_wheel" if execution and not faults else "none", "maximum_inference": cfg.get("maximum_inference")}


if __name__ == "__main__": raise SystemExit(main())
