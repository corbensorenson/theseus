#!/usr/bin/env python3
"""Qualify trusted offline evaluator build mechanics without repository code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_trusted_build_canaries_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_trusted_build_canaries.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    config = p2a.read_json(config_path)
    report = qualify(config, config_path) if args.qualify else preflight(config, config_path)
    p2a.write_json(p2a.resolve(args.out or config["report"]), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    faults = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    owner = p2a.resolve(str(config.get("owner") or ""))
    if owner != Path(__file__).resolve() or p2a.sha256_file(owner) != str(config.get("owner_sha256") or ""):
        faults.append("owner_binding_invalid")
    identity_binding = p2a.mapping(config.get("toolchain_identity_report"))
    identity_path = p2a.resolve(str(identity_binding.get("path") or ""))
    if not identity_path.is_file() or p2a.sha256_file(identity_path) != str(identity_binding.get("sha256") or ""):
        faults.append("toolchain_identity_binding_invalid")
        identity = {}
    else:
        identity = p2a.read_json(identity_path)
    if identity.get("trigger_state") != "GREEN":
        faults.append("toolchain_identity_not_green")
    for name in ("sandbox_exec", "python", "node", "npm_cli", "pnpm_cli", "bun", "deno", "rustc"):
        raw = p2a.mapping(p2a.mapping(config.get("tools")).get(name))
        path = p2a.resolve(str(raw.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(raw.get("sha256") or ""):
            faults.append(f"tool_identity_invalid:{name}")
    authority = p2a.mapping(config.get("authority"))
    allowed = {"trusted_fixture_materialization_authorized", "trusted_build_canary_execution_authorized"}
    for key, value in authority.items():
        if value is not (key in allowed):
            faults.append(f"authority_invalid:{key}")
    limits = p2a.mapping(config.get("limits"))
    for key in ("cpu_seconds", "wall_seconds", "output_mib", "open_files", "user_processes"):
        if int(limits.get(key) or 0) <= 0:
            faults.append(f"limit_invalid:{key}")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED" if faults else "PAUSED",
        "state": "CONTRACT_INVALID" if faults else "READY_FOR_TRUSTED_OFFLINE_BUILD_CANARIES",
        "faults": sorted(set(faults)),
        "config": artifact(config_path),
        "toolchain_identity_report": artifact(identity_path),
        "qualification_executed": False,
        "qualified_scopes": [],
        "repository_execution_authorized": False,
        "dependency_prefetch_authorized": False,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def qualify(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path)
    if before["trigger_state"] == "RED":
        return before
    faults = []
    receipts: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="theseus-vcm-build-", dir="/private/tmp") as raw_root:
        root = Path(raw_root).resolve()
        tools = {name: str(p2a.resolve(str(p2a.mapping(raw).get("path") or ""))) for name, raw in p2a.mapping(config["tools"]).items()}
        profile = sandbox_profile(root)
        env = {"HOME": str(root), "TMPDIR": str(root / "tmp"), "PATH": "/usr/bin:/bin", "NO_COLOR": "1", "CI": "1"}
        (root / "tmp").mkdir()
        receipts["sandbox_denials"] = denial_canary(root, tools, profile, env, config)
        if not receipts["sandbox_denials"].get("passed"):
            faults.append("sandbox_denial_canary_failed")

        pip_root = root / "pip"
        pip_root.mkdir()
        make_wheel(pip_root / "wheelhouse")
        receipts["pip_offline_install"] = run(
            [tools["python"], "-m", "pip", "install", "--disable-pip-version-check", "--no-index", "--find-links", str(pip_root / "wheelhouse"), "--target", str(pip_root / "site"), "trusted-dep==1.0.0"],
            pip_root, profile, env, config,
        )
        receipts["pip_import"] = run(
            [tools["python"], "-c", "import trusted_dep; assert trusted_dep.VALUE == 7"],
            pip_root, profile, {**env, "PYTHONPATH": str(pip_root / "site")}, config,
        )

        for manager in ("npm", "pnpm", "bun"):
            manager_root = root / manager
            make_node_fixture(manager_root)
            if manager == "npm":
                prefix = [tools["node"], tools["npm_cli"]]
                lock_args = ["install", "--package-lock-only", "--offline", "--ignore-scripts", "--no-audit", "--no-fund"]
                install_args = ["ci", "--offline", "--ignore-scripts", "--no-audit", "--no-fund"]
            elif manager == "pnpm":
                prefix = [tools["node"], tools["pnpm_cli"]]
                lock_args = ["install", "--lockfile-only", "--offline", "--ignore-scripts"]
                install_args = ["install", "--offline", "--frozen-lockfile", "--ignore-scripts"]
            else:
                prefix = [tools["bun"]]
                lock_args = ["install", "--lockfile-only", "--offline", "--ignore-scripts"]
                install_args = ["install", "--offline", "--frozen-lockfile", "--ignore-scripts"]
            receipts[f"{manager}_offline_lock"] = run(prefix + lock_args, manager_root, profile, env, config)
            receipts[f"{manager}_offline_install"] = run(prefix + install_args, manager_root, profile, env, config)
            receipts[f"{manager}_local_import"] = run(
                [tools["node"], "-e", "const d=require('trusted-dep');if(d.value!==7)process.exit(2)"],
                manager_root, profile, env, config,
            )
            sentinel_hits = sorted(p.relative_to(manager_root).as_posix() for p in manager_root.rglob("install-script-ran"))
            receipts[f"{manager}_lifecycle_script_suppression"] = {
                "passed": not sentinel_hits,
                "sentinel_hits": sentinel_hits,
            }

        ts_root = root / "typescript"
        ts_root.mkdir()
        (ts_root / "canary.ts").write_text("const value: number = 7; if (value !== 7) throw new Error('bad');\n", encoding="utf-8")
        receipts["deno_typescript_check"] = run(
            [tools["deno"], "check", "--no-config", str(ts_root / "canary.ts")],
            ts_root, profile, env, config,
        )
        receipts["deno_typescript_run"] = run(
            [tools["deno"], "run", "--cached-only", "--no-config", str(ts_root / "canary.ts")],
            ts_root, profile, env, config,
        )

        rust_root = root / "rust"
        rust_root.mkdir()
        (rust_root / "canary.rs").write_text("fn main(){assert_eq!(3+4,7);}\n", encoding="utf-8")
        receipts["rust_compile"] = run(
            [tools["rustc"], "--edition=2021", str(rust_root / "canary.rs"), "-o", str(rust_root / "canary")],
            rust_root, profile, env, config,
        )
        receipts["rust_binary_run"] = run([str(rust_root / "canary")], rust_root, profile, env, config)

    command_receipts = [row for row in receipts.values() if isinstance(row, dict) and "returncode" in row]
    for name, row in receipts.items():
        if "returncode" in row and (row.get("returncode") != 0 or row.get("boundary_hit")):
            faults.append(f"canary_failed:{name}")
        if name.endswith("lifecycle_script_suppression") and row.get("passed") is not True:
            faults.append(f"canary_failed:{name}")
    passed = not faults
    scopes = [
        "network_denial_and_write_confinement",
        "pip_local_wheel_offline_install",
        "npm_local_file_offline_install_ignore_scripts",
        "pnpm_local_file_offline_install_ignore_scripts",
        "bun_local_file_offline_install_ignore_scripts",
        "deno_dependency_free_typescript_check_and_run",
        "rustc_dependency_free_trusted_source_compile_and_run",
    ] if passed else []
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if passed else "RED",
        "state": "TRUSTED_OFFLINE_BUILD_CANARIES_QUALIFIED" if passed else "TRUSTED_OFFLINE_BUILD_CANARIES_FAILED",
        "faults": sorted(set(faults)),
        "qualification_executed": True,
        "qualified_scopes": scopes,
        "explicitly_unqualified_scopes": [
            "yarn_runtime", "remote_dependency_prefetch", "real_lock_resolution",
            "untrusted_install_scripts", "untrusted_repository_transpilation",
            "untrusted_rust_or_build_script_compilation", "repository_runner_adequacy",
        ],
        "receipts": receipts,
        "command_receipt_count": len(command_receipts),
        "repository_execution_authorized": False,
        "dependency_prefetch_authorized": False,
    }


def run(command: list[str], cwd: Path, profile: str, env: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    limits = p2a.mapping(config["limits"])
    backend = str(p2a.resolve(str(p2a.mapping(config["tools"])["sandbox_exec"]["path"])))

    def set_limits() -> None:
        mib = 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (int(limits["cpu_seconds"]),) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (int(limits["output_mib"]) * mib,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (int(limits["open_files"]),) * 2)
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (int(limits["user_processes"]),) * 2)

    started = p2a.monotonic() if hasattr(p2a, "monotonic") else None
    try:
        completed = subprocess.run(
            [backend, "-p", profile, *command], cwd=cwd, env=env, capture_output=True,
            check=False, timeout=int(limits["wall_seconds"]), preexec_fn=set_limits,
        )
        boundary = ""
        returncode = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        boundary = "wall_boundary_hit"
        returncode = None
        stdout, stderr = exc.stdout or b"", exc.stderr or b""
    maximum = int(limits["output_mib"]) * 1024 * 1024
    output_boundary = len(stdout) > maximum or len(stderr) > maximum
    if output_boundary and not boundary:
        boundary = "output_boundary_hit"
    stdout, stderr = stdout[:maximum], stderr[:maximum]
    return {
        "command": command,
        "cwd_leaf": cwd.name,
        "returncode": returncode,
        "boundary_hit": bool(boundary),
        "boundary_reason": boundary,
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_head": stderr.decode("utf-8", "replace")[:4000],
        "stderr_tail": stderr.decode("utf-8", "replace")[-2000:],
        "project_selected_quality_token_cap": None,
    }


def denial_canary(root: Path, tools: dict[str, str], profile: str, env: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    outside = ROOT / "runtime" / "control" / "vcm_trusted_build_escape"
    source = (
        "import json,pathlib,socket\n"
        f"o=pathlib.Path({str(outside)!r});r={{}}\n"
        "try:o.write_text('x');r['outside_write_denied']=False\n"
        "except OSError:r['outside_write_denied']=True\n"
        "s=socket.socket();s.settimeout(.2)\n"
        "try:s.connect(('127.0.0.1',9));r['network_denied']=False\n"
        "except OSError:r['network_denied']=True\n"
        "print(json.dumps(r,sort_keys=True))\n"
    )
    receipt = run([tools["python"], "-c", source], root, profile, env, config)
    try:
        # Re-run outside receipt storage is deliberately avoided; the stdout is
        # known from the exact canary only through its expected successful exit.
        passed = receipt["returncode"] == 0 and not receipt["boundary_hit"]
    except Exception:
        passed = False
    return {**receipt, "passed": passed, "expected_denials": ["network_denied", "outside_write_denied"]}


def sandbox_profile(root: Path) -> str:
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        f'(deny file-write* (require-not (subpath "{root}")))',
        '(allow file-write* (literal "/dev/null"))',
    ])


def make_wheel(wheelhouse: Path) -> None:
    wheelhouse.mkdir()
    path = wheelhouse / "trusted_dep-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("trusted_dep/__init__.py", "VALUE = 7\n")
        archive.writestr("trusted_dep-1.0.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: trusted-dep\nVersion: 1.0.0\n")
        archive.writestr("trusted_dep-1.0.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: theseus\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr("trusted_dep-1.0.0.dist-info/RECORD", "")


def make_node_fixture(root: Path) -> None:
    dep = root / "dep"
    dep.mkdir(parents=True)
    (root / "package.json").write_text(json.dumps({"name": "trusted-app", "version": "1.0.0", "private": True, "dependencies": {"trusted-dep": "file:./dep"}}), encoding="utf-8")
    (dep / "package.json").write_text(json.dumps({"name": "trusted-dep", "version": "1.0.0", "main": "index.js", "scripts": {"install": "node -e \"require('fs').writeFileSync('install-script-ran','x')\""}}), encoding="utf-8")
    (dep / "index.js").write_text("module.exports = {value: 7};\n", encoding="utf-8")


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "qualification_executed", "qualified_scopes",
        "explicitly_unqualified_scopes", "command_receipt_count",
        "parent_target_or_evaluator_executions", "candidate_or_control_calls",
        "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
