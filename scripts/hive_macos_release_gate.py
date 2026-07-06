"""macOS installer release-candidate gate for Project Theseus Hive.

This script turns the manual Intel/Apple-Silicon rollout checklist into a
repeatable report. It can rebuild package artifacts, verify/publish the private
Hive update catalog, reinstall this Mac as a local canary, and then check the
live service/update/menu-bar endpoints that spare Macs need before rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import plistlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError

import theseus_runtime


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIST_MACOS = ROOT / "dist" / "macos"
DEFAULT_API_URL = "http://127.0.0.1:8791"
MACOS_RUNTIME_REPORTS = Path.home() / "Library" / "Application Support" / "Project Theseus Hive" / "runtime" / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Project Theseus Hive macOS release-candidate gates.")
    parser.add_argument("--execute", action="store_true", help="Rebuild, verify, publish, install local canary, and converge local update state.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-version-publish", action="store_true")
    parser.add_argument("--skip-local-install", action="store_true")
    parser.add_argument("--skip-local-converge", action="store_true")
    parser.add_argument("--skip-mlx-work-proof", action="store_true", help="Skip bounded MLX worker/CLI proof; Apple-Silicon private canary will not be fully proven.")
    parser.add_argument("--skip-deps", action="store_true", help="Use only for fast diagnostics; real Apple-Silicon canary should install deps.")
    parser.add_argument("--no-require-mlx", action="store_true", help="Do not require MLX during local Apple-Silicon canary install.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--coordinator-url", action="append", default=[])
    parser.add_argument("--peer-url", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--out", default="reports/hive_macos_release_gate.json")
    parser.add_argument("--markdown-out", default="reports/hive_macos_release_gate.md")
    args = parser.parse_args()

    report = run_gate(args)
    write_json(resolve(args.out), report)
    if args.markdown_out:
        resolve(args.markdown_out).write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 2


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    gates: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    platform_info = platform_report()

    if platform.system() != "Darwin":
        gates.append(gate("macos_host", False, "hard", "macOS packaging must run on macOS.", {}))
        return finish_report(args, gates, actions, platform_info, started)

    source_before = git_status()
    gates.append(gate("source_clean_before_release", not source_before.get("dirty"), "hard", "Git working tree is clean before release actions.", source_before))
    runtime_doctor = theseus_runtime.runtime_doctor_report(write_report=True)
    action_python = str(get_path(runtime_doctor, ["preferred_runtime", "python"], "") or sys.executable)
    gates.append(
        gate(
            "mac_runtime_doctor_not_red",
            runtime_doctor.get("state") != "RED",
            "hard",
            "Mac runtime doctor must not be RED; source, installed app, LaunchAgents, and MLX runtime must be explicitly identified.",
            compact_runtime_doctor(runtime_doctor),
        )
    )

    if args.execute and not args.skip_build:
        env = dict(os.environ)
        env["THESEUS_MACOS_BUILD_DMG"] = "1"
        actions.append(run_action("build_macos_artifacts", ["./scripts/package_theseus_macos.sh"], timeout=3600, env=env))

    artifacts = inspect_macos_artifacts()
    gates.extend(artifact_gates(artifacts))

    if args.execute and not args.skip_version_publish:
        actions.append(run_action("verify_hive_version", [action_python, "scripts/hive_version_manager.py", "verify", "--out", "reports/hive_verified_version.json"], timeout=300))
        actions.append(run_action("publish_hive_catalog", [action_python, "scripts/hive_version_manager.py", "publish", "--out", "reports/hive_update_catalog.json"], timeout=300))
        actions.append(run_action("refresh_installer_artifacts", [action_python, "scripts/hive_version_manager.py", "installer-artifacts", "--out", "reports/hive_installer_artifacts.json"], timeout=300))
        actions.append(run_action("refresh_hive_version_status", [action_python, "scripts/hive_version_manager.py", "status", "--out", "reports/hive_version_status.json"], timeout=300))

    version = inspect_version_state()
    gates.extend(version_gates(version))

    mirror = inspect_runtime_catalog_mirror()
    gates.append(gate("live_runtime_catalog_mirror", mirror.get("ok"), "hard", "Published catalog and installer reports are mirrored where the LaunchAgent-backed node serves them.", mirror))

    if args.execute and not args.skip_local_install:
        install_command = ["./scripts/install_theseus_hive_macos.sh", "--auto-update-soft", "--install-service", "--enable-service", "--start"]
        if args.skip_deps:
            install_command.append("--skip-deps")
        if platform_info.get("is_apple_silicon") and not args.no_require_mlx:
            install_command.append("--require-mlx")
        actions.append(run_action("install_local_macos_canary", install_command, timeout=3600))
        wait_for_api(args.api_url, timeout_seconds=30)

    local = inspect_local_canary(args)
    gates.extend(local_canary_gates(local, platform_info))

    mlx_work = run_mlx_work_proof(args, python=action_python, platform_info=platform_info)
    gates.extend(mlx_work_gates(mlx_work, platform_info))

    if args.execute and not args.skip_local_converge:
        converge_api_url = str(local.get("api_url") or args.api_url or DEFAULT_API_URL).rstrip("/")
        actions.append(
            run_action(
                "converge_local_soft_update",
                [
                    action_python,
                    "scripts/hive_version_manager.py",
                    "converge",
                    "--execute",
                    "--peer-url",
                    converge_api_url,
                    "--timeout-seconds",
                    str(max(1, int(args.timeout))),
                    "--out",
                    "reports/hive_version_convergence.json",
                ],
                timeout=180,
            )
        )

    convergence = read_json(REPORTS / "hive_version_convergence.json", {})
    gates.append(
        gate(
            "local_soft_update_convergence",
            local_soft_convergence_ok(convergence, str(local.get("api_url") or args.api_url or DEFAULT_API_URL)),
            "hard",
            "Local node can consume the private catalog and apply safe soft update metadata.",
            compact_convergence(convergence),
        )
    )

    network = run_network_doctor(args, python=action_python)
    gates.append(gate("network_doctor_not_red", network.get("state") != "RED", "fleet", "Hive network doctor must not be RED before trusting multi-node Windows/Mac convergence.", network_summary(network)))

    source_after = git_status()
    gates.append(gate("source_clean_after_release", not source_after.get("dirty"), "hard", "Generated reports/dist artifacts should stay out of source control.", source_after))
    gates.append(gate("physical_intel_canary", False, "physical", "Run this same gate on one Intel Mac before broad Intel rollout.", {"required_command": "python3 scripts/hive_macos_release_gate.py --execute --skip-build --skip-version-publish"}))
    return finish_report(args, gates, actions, platform_info, started, artifacts=artifacts, version=version, local=local, network=network, runtime=runtime_doctor, mlx_work=mlx_work)


def inspect_macos_artifacts() -> dict[str, Any]:
    manifest = read_json(DIST_MACOS / "hive-installer-artifacts.json", {})
    artifacts: dict[str, Any] = {
        "manifest": manifest,
        "paths": {},
        "app": inspect_app(DIST_MACOS / "ProjectTheseusHive.app"),
        "codesign": run_capture(["codesign", "-dv", "--verbose=4", str(DIST_MACOS / "ProjectTheseusHive.app")], timeout=30, merge_stderr=True),
        "spctl": run_capture(["spctl", "-a", "-vv", "-t", "exec", str(DIST_MACOS / "ProjectTheseusHive.app")], timeout=30, merge_stderr=True),
    }
    for name in ["ProjectTheseusHive.app", "ProjectTheseusHive.zip", "ProjectTheseusHive.pkg", "ProjectTheseusHive.dmg"]:
        path = DIST_MACOS / name
        artifacts["paths"][name] = file_row(path)
    artifacts["wrapper"] = inspect_wrapper(DIST_MACOS / "ProjectTheseusHive.app" / "Contents" / "MacOS" / "ProjectTheseusHive")
    artifacts["menu_bar_helper"] = inspect_binary(DIST_MACOS / "ProjectTheseusHive.app" / "Contents" / "Resources" / "payload" / "packaging" / "macos" / "TheseusHiveMenuBar")
    spctl_text = str(artifacts["spctl"].get("stderr_tail") or artifacts["spctl"].get("stdout_tail") or "")
    codesign_text = str(artifacts["codesign"].get("stderr_tail") or artifacts["codesign"].get("stdout_tail") or "")
    artifacts["notarized"] = "accepted" in spctl_text.lower()
    artifacts["ad_hoc_signed"] = "Signature=adhoc" in codesign_text
    return artifacts


def artifact_gates(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = artifacts.get("manifest") if isinstance(artifacts.get("manifest"), dict) else {}
    compat = manifest.get("macos_compatibility") if isinstance(manifest.get("macos_compatibility"), dict) else {}
    paths = artifacts.get("paths") if isinstance(artifacts.get("paths"), dict) else {}
    target_arches = set(compat.get("target_architectures") or [])
    menu_arches = set(compat.get("menu_bar_helper_architectures") or [])
    wrapper = artifacts.get("wrapper") if isinstance(artifacts.get("wrapper"), dict) else {}
    return [
        gate("macos_app_exists", bool(paths.get("ProjectTheseusHive.app", {}).get("exists")), "hard", "ProjectTheseusHive.app exists.", paths.get("ProjectTheseusHive.app", {})),
        gate("macos_dmg_exists", bool(paths.get("ProjectTheseusHive.dmg", {}).get("exists")), "hard", "ProjectTheseusHive.dmg exists.", paths.get("ProjectTheseusHive.dmg", {})),
        gate("macos_pkg_exists", bool(paths.get("ProjectTheseusHive.pkg", {}).get("exists")), "hard", "ProjectTheseusHive.pkg exists.", paths.get("ProjectTheseusHive.pkg", {})),
        gate("macos_zip_exists", bool(paths.get("ProjectTheseusHive.zip", {}).get("exists")), "hard", "ProjectTheseusHive.zip exists.", paths.get("ProjectTheseusHive.zip", {})),
        gate("macos_minimum_10_15", str(compat.get("minimum_system_version") or "") == "10.15", "hard", "Installer declares macOS 10.15 minimum.", compat),
        gate("macos_universal_intent", {"x86_64", "arm64"}.issubset(target_arches), "hard", "Artifact manifest targets both Intel and Apple Silicon.", compat),
        gate("menu_bar_universal_or_rebuildable", {"x86_64", "arm64"}.issubset(menu_arches), "hard", "Menu bar helper is universal in the package manifest.", compat),
        gate("dmg_installer_uses_launchagent_start", bool(wrapper.get("auto_update_soft") and wrapper.get("install_service") and wrapper.get("enable_service") and wrapper.get("bounded_api_poll") and not wrapper.get("default_foreground_start")), "hard", "DMG app wrapper enables LaunchAgents and safe soft updates without a foreground start that can hold the installer window open.", wrapper),
        gate("codesign_private_canary_ok", bool(artifacts.get("ad_hoc_signed") or artifacts.get("notarized")), "hard", "App is at least ad-hoc signed for private canary testing.", {"ad_hoc_signed": artifacts.get("ad_hoc_signed"), "notarized": artifacts.get("notarized")}),
        gate("developer_id_notarization", bool(artifacts.get("notarized")), "public", "Developer ID signing/notarization is required before broad non-technical distribution.", {"spctl": artifacts.get("spctl", {})}),
    ]


def inspect_version_state() -> dict[str, Any]:
    status = read_json(REPORTS / "hive_version_status.json", {})
    verified = read_json(REPORTS / "hive_verified_version.json", {})
    publish = read_json(REPORTS / "hive_version_publish.json", {})
    catalog = read_json(REPORTS / "hive_update_catalog.json", {})
    return {
        "status": status,
        "verified": verified,
        "publish": publish,
        "catalog": catalog,
    }


def version_gates(version: dict[str, Any]) -> list[dict[str, Any]]:
    status = version.get("status") if isinstance(version.get("status"), dict) else {}
    verified = version.get("verified") if isinstance(version.get("verified"), dict) else {}
    publish = version.get("publish") if isinstance(version.get("publish"), dict) else {}
    catalog = version.get("catalog") if isinstance(version.get("catalog"), dict) else {}
    local_version = get_path(status, ["local", "version_id"], "")
    verified_version = str(verified.get("version_id") or "")
    local_commit = get_path(status, ["local", "git", "commit"], "")
    verified_commit = get_path(verified, ["git", "commit"], "")
    return [
        gate("verified_version_current", bool(verified.get("ok") and local_version and local_version == verified_version), "hard", "Verified version matches the local source version.", {"local_version": local_version, "verified_version": verified_version}),
        gate("verified_commit_current", bool(local_commit and local_commit == verified_commit), "hard", "Verified manifest is for the current commit.", {"local_commit": local_commit, "verified_commit": verified_commit}),
        gate("private_update_catalog_published", bool(catalog.get("ok") and catalog.get("offers")), "hard", "Private Hive update catalog has at least one safe soft offer.", {"ok": catalog.get("ok"), "offer_count": len(catalog.get("offers") or [])}),
        gate("publish_report_ok", bool(publish.get("ok")), "hard", "Publish report succeeded.", publish),
        gate("status_matches_verified_target", bool(get_path(status, ["convergence", "local_matches_target"], False)), "hard", "Version status sees local node at the verified target.", get_path(status, ["convergence"], {})),
    ]


def inspect_runtime_catalog_mirror() -> dict[str, Any]:
    rows = {}
    for name in ["hive_update_catalog.json", "hive_verified_version.json", "hive_installer_artifacts.json"]:
        source = REPORTS / name
        runtime = MACOS_RUNTIME_REPORTS / name
        if name == "hive_installer_artifacts.json":
            source_json = read_json(source, {})
            runtime_json = read_json(runtime, {})
            match = installer_artifacts_semantic_match(source_json, runtime_json)
        else:
            match = source.exists() and runtime.exists() and sha256_file(source) == sha256_file(runtime)
        rows[name] = {
            "source": file_row(source),
            "runtime": file_row(runtime),
            "sha256_match": source.exists() and runtime.exists() and sha256_file(source) == sha256_file(runtime),
            "semantic_match": match,
        }
    return {
        "ok": all(row["semantic_match"] for row in rows.values()),
        "runtime_reports": str(MACOS_RUNTIME_REPORTS),
        "files": rows,
    }


def installer_artifacts_semantic_match(source: Any, runtime: Any) -> bool:
    if not isinstance(source, dict) or not isinstance(runtime, dict):
        return False
    source_installers = macos_installer_identity(source)
    runtime_installers = macos_installer_identity(runtime)
    return bool(source_installers) and source_installers == runtime_installers


def macos_installer_identity(report: dict[str, Any]) -> list[tuple[str, str, int]]:
    rows = report.get("artifacts") if isinstance(report.get("artifacts"), list) else []
    identity = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = str(row.get("path") or "")
        if not path.startswith("dist/macos/ProjectTheseusHive."):
            continue
        if not any(path.endswith(suffix) for suffix in [".dmg", ".pkg", ".zip"]):
            continue
        identity.append((path, str(row.get("sha256") or ""), int(row.get("size_bytes") or 0)))
    return sorted(identity)


def inspect_local_canary(args: argparse.Namespace) -> dict[str, Any]:
    api_url = choose_api_url(str(args.api_url or DEFAULT_API_URL).rstrip("/"), timeout=float(args.timeout))
    headers = auth_headers()
    probe_python, probe_cwd = installed_probe_runtime()
    local = {
        "api_url": api_url,
        "launchagents": inspect_launchagents(),
        "installed_apps": inspect_installed_apps(),
        "endpoints": {
            "status": fetch_url(api_url + "/api/hive/status", timeout=args.timeout),
            "auth_status": fetch_url(api_url + "/api/hive/auth/status", timeout=args.timeout, headers=headers),
            "mobile": fetch_url(api_url + "/mobile", timeout=args.timeout, expect_json=False),
            "update_catalog": fetch_url(api_url + "/api/hive/update-catalog", timeout=args.timeout),
            "update_status": fetch_url(api_url + "/api/hive/update-status", timeout=args.timeout, headers=headers),
            "installer_artifacts": fetch_url(api_url + "/api/hive/installer-artifacts", timeout=args.timeout, headers=headers),
        },
        "probe_runtime": {"python": str(probe_python), "cwd": str(probe_cwd)},
        "probe": run_json([str(probe_python), "scripts/hive_node.py", "probe", "--out", "reports/hive_status.json"], timeout=180, cwd=probe_cwd),
    }
    return local


def local_canary_gates(local: dict[str, Any], platform_info: dict[str, Any]) -> list[dict[str, Any]]:
    endpoints = local.get("endpoints") if isinstance(local.get("endpoints"), dict) else {}
    launchagents = local.get("launchagents") if isinstance(local.get("launchagents"), dict) else {}
    probe = local.get("probe") if isinstance(local.get("probe"), dict) else {}
    mlx = get_path(probe, ["resources", "mlx"], {})
    is_apple_silicon = bool(platform_info.get("is_apple_silicon"))
    mlx_ok = bool(mlx.get("available")) if is_apple_silicon else not bool(mlx.get("available"))
    return [
        gate("local_hive_launchagent_loaded", bool(get_path(launchagents, ["local.project-theseus.hive", "loaded"], False)), "hard", "Hive LaunchAgent is loaded.", launchagents.get("local.project-theseus.hive", {})),
        gate("local_update_launchagent_loaded", bool(get_path(launchagents, ["local.project-theseus.update", "loaded"], False)), "hard", "Update LaunchAgent is loaded.", launchagents.get("local.project-theseus.update", {})),
        gate("local_menubar_launchagent_loaded", bool(get_path(launchagents, ["local.project-theseus.hive-menubar", "loaded"], False)), "hard", "Menu bar LaunchAgent is loaded.", launchagents.get("local.project-theseus.hive-menubar", {})),
        gate("local_status_endpoint", bool(get_path(endpoints, ["status", "ok"], False)), "hard", "Local Hive status endpoint responds.", endpoints.get("status", {})),
        gate(
            "local_auth_status_endpoint",
            bool(
                get_path(endpoints, ["auth_status", "ok"], False)
                and get_path(endpoints, ["auth_status", "json", "authenticated"], False)
                and get_path(endpoints, ["auth_status", "json", "security", "tokens_printed"], True) is False
            ),
            "hard",
            "Local authenticated Hive status endpoint responds without printing tokens.",
            endpoints.get("auth_status", {}),
        ),
        gate("local_mobile_endpoint", bool(get_path(endpoints, ["mobile", "ok"], False)), "hard", "Local /mobile operator page loads.", endpoints.get("mobile", {})),
        gate("local_update_catalog_endpoint", bool(get_path(endpoints, ["update_catalog", "ok"], False) and get_path(endpoints, ["update_catalog", "json", "offers"], [])), "hard", "Local node exposes the private update catalog.", endpoints.get("update_catalog", {})),
        gate("local_installer_artifacts_endpoint", bool(get_path(endpoints, ["installer_artifacts", "ok"], False)), "hard", "Local node exposes installer artifacts for artifact sync.", endpoints.get("installer_artifacts", {})),
        gate("mlx_capability_matches_arch", mlx_ok, "hard" if is_apple_silicon else "hard", "Apple Silicon should advertise MLX after canary install; Intel should not.", {"is_apple_silicon": is_apple_silicon, "mlx": mlx}),
    ]


def run_mlx_work_proof(args: argparse.Namespace, *, python: str, platform_info: dict[str, Any]) -> dict[str, Any]:
    if args.skip_mlx_work_proof:
        return {
            "ok": not bool(platform_info.get("is_apple_silicon")),
            "state": "SKIPPED",
            "skipped": True,
            "message": "Bounded MLX work proof skipped by flag.",
        }
    command = [
        python,
        "scripts/macos_mlx_work_proof.py",
        "--timeout-seconds",
        str(max(60, int(args.timeout))),
        "--out",
        "reports/macos_mlx_work_proof.json",
        "--markdown-out",
        "reports/macos_mlx_work_proof.md",
    ]
    result = run_json(command, timeout=max(180, int(args.timeout) * 20))
    if result:
        return result
    return read_json(REPORTS / "macos_mlx_work_proof.json", {})


def mlx_work_gates(mlx_work: dict[str, Any], platform_info: dict[str, Any]) -> list[dict[str, Any]]:
    is_apple_silicon = bool(platform_info.get("is_apple_silicon"))
    is_intel = bool(platform_info.get("is_intel_mac"))
    summary = mlx_work.get("summary") if isinstance(mlx_work.get("summary"), dict) else {}
    if is_intel:
        return [
            gate(
                "macos_mlx_work_proof_not_required_on_intel",
                True,
                "hard",
                "Intel Macs are CPU/storage/operator nodes and should not require MLX work proof.",
                compact_mlx_work(mlx_work),
            )
        ]
    return [
        gate(
            "macos_mlx_work_proof_ok",
            bool(mlx_work.get("ok")) if is_apple_silicon else True,
            "hard",
            "Apple Silicon can run bounded MLX worker chunks and command bridges.",
            compact_mlx_work(mlx_work),
        ),
        gate(
            "macos_mlx_worker_chunks_ok",
            bool(summary.get("worker_smoke_count") == summary.get("worker_smoke_ok_count") and summary.get("worker_smoke_count", 0) >= 3) if is_apple_silicon else True,
            "hard",
            "Registered Hive MLX eval/training/rollout worker chunks complete and emit receipts.",
            compact_mlx_rows(mlx_work.get("worker_smokes")),
        ),
        gate(
            "macos_mlx_cli_bridges_ok",
            bool(summary.get("cli_smoke_count") == summary.get("cli_smoke_ok_count") and summary.get("cli_smoke_count", 0) >= 4) if is_apple_silicon else True,
            "hard",
            "Mac MLX command bridges for CUDA-style training lanes complete bounded smokes.",
            compact_mlx_rows(mlx_work.get("cli_smokes")),
        ),
    ]


def compact_mlx_work(mlx_work: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": mlx_work.get("ok"),
        "state": mlx_work.get("state"),
        "summary": mlx_work.get("summary") if isinstance(mlx_work.get("summary"), dict) else {},
        "mlx": {
            "available": get_path(mlx_work, ["mlx", "available"], False),
            "preferred_runtime": get_path(mlx_work, ["mlx", "preferred_runtime"], ""),
            "preferred_python": get_path(mlx_work, ["mlx", "preferred_python"], ""),
        },
        "parity_audit": mlx_work.get("parity_audit") if isinstance(mlx_work.get("parity_audit"), dict) else {},
        "next_actions": mlx_work.get("next_actions") or [],
        "skipped": bool(mlx_work.get("skipped")),
    }


def compact_mlx_rows(rows: Any) -> list[dict[str, Any]]:
    compact = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "name": row.get("name"),
                "task_kind": row.get("task_kind"),
                "command_name": row.get("command_name"),
                "parity_for": row.get("parity_for"),
                "ok": row.get("ok"),
                "backend": row.get("backend"),
                "report_path": row.get("report_path"),
                "metrics": row.get("metrics"),
                "error": row.get("error"),
            }
        )
    return compact


def inspect_launchagents() -> dict[str, Any]:
    labels = ["local.project-theseus.hive", "local.project-theseus.hive-menubar", "local.project-theseus.update"]
    uid = os.getuid()
    rows: dict[str, Any] = {}
    for label in labels:
        result = run_capture(["launchctl", "print", f"gui/{uid}/{label}"], timeout=20)
        fallback = {} if result.get("ok") else run_capture(["launchctl", "list", label], timeout=20)
        rows[label] = {
            "loaded": bool(result.get("ok") or fallback.get("ok")),
            "print": compact_capture(result),
            "list": compact_capture(fallback) if fallback else {},
            "plist": file_row(Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"),
        }
    return rows


def inspect_installed_apps() -> dict[str, Any]:
    rows = {}
    for base in [Path("/Applications"), Path.home() / "Applications"]:
        for name in ["Project Theseus Hive.app", "Project Theseus Setup.app", "Project Theseus Doctor.app"]:
            path = base / name
            if path.exists():
                rows[str(path)] = inspect_app(path)
    return rows


def run_network_doctor(args: argparse.Namespace, *, python: str) -> dict[str, Any]:
    command = [python, "scripts/hive_network_doctor.py", "--timeout", str(args.timeout), "--out", "reports/hive_network_doctor.json", "--markdown-out", "reports/hive_network_doctor.md"]
    for url in args.coordinator_url:
        command.extend(["--coordinator-url", url])
    for url in args.peer_url:
        command.extend(["--peer-url", url])
    result = run_json(command, timeout=120)
    if result:
        return result
    return read_json(REPORTS / "hive_network_doctor.json", {})


def compact_runtime_doctor(report: dict[str, Any]) -> dict[str, Any]:
    preferred = report.get("preferred_runtime") if isinstance(report.get("preferred_runtime"), dict) else {}
    return {
        "state": report.get("state"),
        "preferred_runtime": {
            "name": preferred.get("name"),
            "python": preferred.get("python"),
            "mlx_available": get_path(preferred, ["mlx", "available"], False),
        },
        "blockers": report.get("blockers") or [],
        "warnings": report.get("warnings") or [],
        "false_negatives": report.get("false_negatives") or [],
        "runtime_roots": report.get("runtime_roots") or {},
        "runtime_state": {
            "warnings": get_path(report, ["runtime_state", "warnings"], []),
            "source_verified_version": get_path(report, ["runtime_state", "contexts", "source_checkout", "verified_version", "version_id"], ""),
            "source_catalog_version": get_path(report, ["runtime_state", "contexts", "source_checkout", "update_catalog", "latest_version_id"], ""),
            "installed_verified_version": get_path(report, ["runtime_state", "contexts", "installed_app", "verified_version", "version_id"], ""),
            "installed_catalog_version": get_path(report, ["runtime_state", "contexts", "installed_app", "update_catalog", "latest_version_id"], ""),
            "installed_status_local_version": get_path(report, ["runtime_state", "contexts", "installed_app", "version_status", "local_version_id"], ""),
            "installed_checkin_applied_checkpoint": get_path(report, ["runtime_state", "contexts", "installed_app", "update_checkin", "applied_checkpoint_id"], ""),
        },
        "next_actions": report.get("next_actions") or [],
    }


def finish_report(
    args: argparse.Namespace,
    gates: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    platform_info: dict[str, Any],
    started: float,
    **sections: Any,
) -> dict[str, Any]:
    hard = [row for row in gates if row.get("severity") == "hard"]
    public = [row for row in gates if row.get("severity") == "public"]
    fleet = [row for row in gates if row.get("severity") == "fleet"]
    physical = [row for row in gates if row.get("severity") == "physical"]
    private_canary_ready = all(row.get("ok") for row in hard)
    public_distribution_ready = private_canary_ready and all(row.get("ok") for row in public)
    fleet_rollout_ready = private_canary_ready and all(row.get("ok") for row in fleet) and all(row.get("ok") for row in physical)
    failed = [row for row in gates if not row.get("ok")]
    report = {
        "ok": private_canary_ready,
        "policy": "project_theseus_macos_release_gate_v1",
        "created_utc": now(),
        "execute": bool(args.execute),
        "platform": platform_info,
        "private_canary_ready": private_canary_ready,
        "public_distribution_ready": public_distribution_ready,
        "fleet_rollout_ready": fleet_rollout_ready,
        "summary": {
            "gate_count": len(gates),
            "failed_count": len(failed),
            "hard_failed": [row["name"] for row in failed if row.get("severity") == "hard"],
            "public_pending": [row["name"] for row in failed if row.get("severity") == "public"],
            "fleet_pending": [row["name"] for row in failed if row.get("severity") == "fleet"],
            "physical_pending": [row["name"] for row in failed if row.get("severity") == "physical"],
        },
        "gates": gates,
        "actions": actions,
        "sections": sections,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "next_actions": next_actions(private_canary_ready, public_distribution_ready, fleet_rollout_ready, failed),
        "external_inference_calls": 0,
    }
    return report


def next_actions(private_ready: bool, public_ready: bool, fleet_ready: bool, failed: list[dict[str, Any]]) -> list[str]:
    actions = []
    failed_names = {row.get("name") for row in failed}
    if not private_ready:
        actions.append("Fix failed hard gates before installing this DMG on another Mac.")
    if "mlx_capability_matches_arch" in failed_names:
        actions.append("On Apple Silicon, rerun the canary without --skip-deps and inspect reports/macos_dependency_bootstrap.json.")
    if any(name in failed_names for name in {"macos_mlx_work_proof_ok", "macos_mlx_worker_chunks_ok", "macos_mlx_cli_bridges_ok"}):
        actions.append("Inspect reports/macos_mlx_work_proof.json and reports/macos_mlx_work_proof/*.json for failed Apple MLX worker or command bridge evidence.")
    if "network_doctor_not_red" in failed_names:
        actions.append("Fix LAN/firewall/coordinator reachability before trusting Windows/Mac convergence.")
    if "physical_intel_canary" in failed_names:
        actions.append("Run the gate on one Intel Mac before broad Intel rollout.")
    if not public_ready:
        actions.append("Developer ID signing and notarization are still required for easy installs outside your own machines.")
    if private_ready and not fleet_ready:
        actions.append("Private Apple-Silicon canary is usable; hold broad fleet rollout until fleet/physical gates pass.")
    return actions


def gate(name: str, ok: bool, severity: str, message: str, evidence: Any) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "severity": severity,
        "message": message,
        "evidence": evidence,
    }


def run_action(name: str, command: list[str], *, timeout: int, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    result = run_capture(command, timeout=timeout, env=env)
    return {
        "name": name,
        "command": command,
        **result,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def run_capture(command: list[str], *, timeout: int, env: dict[str, str] | None = None, merge_stderr: bool = False) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if merge_stderr and stderr and not stdout:
        stdout = stderr
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def run_json(command: list[str], *, timeout: int, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=cwd or ROOT, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc), "command": command}
    text = result.stdout or "{}"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict) and payload:
        payload.setdefault("ok", result.returncode == 0)
        payload.setdefault("returncode", result.returncode)
        if result.returncode != 0:
            payload.setdefault("stderr_tail", result.stderr[-4000:])
        return payload
    if result.returncode != 0:
        return {
            "ok": False,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "command": command,
        }
    return {"ok": True, "stdout": text}


def installed_probe_runtime() -> tuple[Path, Path]:
    install_root = Path.home() / "Library" / "Application Support" / "Project Theseus Hive" / "app" / "current"
    installed_python = install_root / ".venv-puffer" / "bin" / "python"
    if installed_python.exists() and (install_root / "scripts" / "hive_node.py").exists():
        return installed_python, install_root
    return Path(sys.executable), ROOT


def fetch_url(url: str, *, timeout: float, headers: dict[str, str] | None = None, expect_json: bool = True) -> dict[str, Any]:
    request = urlrequest.Request(url, headers=headers or {"User-Agent": "ProjectTheseusMacGate/0.1"})
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:  # noqa: S310 - private local Hive endpoint.
            raw = response.read(4 * 1024 * 1024)
            status = response.status
            content_type = response.headers.get("Content-Type", "")
    except (OSError, URLError, TimeoutError) as exc:
        return {"ok": False, "url": url, "error": str(exc)}
    text = raw.decode("utf-8", errors="replace")
    row: dict[str, Any] = {"ok": 200 <= status < 300, "url": url, "status": status, "content_type": content_type, "bytes": len(raw)}
    if expect_json:
        try:
            row["json"] = json.loads(text)
        except json.JSONDecodeError:
            row["ok"] = False
            row["error"] = "non_json_response"
            row["body_tail"] = text[-400:]
    else:
        row["body_contains_hive"] = "Hive" in text or "Theseus" in text
        row["ok"] = bool(row["ok"] and row["body_contains_hive"])
    return row


def wait_for_api(api_url: str, *, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    url = api_url.rstrip("/") + "/api/hive/status"
    while time.time() < deadline:
        if fetch_url(url, timeout=1.0).get("ok"):
            return True
        time.sleep(1)
    return False


def choose_api_url(preferred: str, *, timeout: float) -> str:
    candidates = [preferred]
    for path in [REPORTS / "hive_status.json", MACOS_RUNTIME_REPORTS / "hive_status.json"]:
        status = read_json(path, {})
        if isinstance(status, dict):
            if status.get("api_url"):
                candidates.append(str(status["api_url"]))
            listen_host = str(status.get("listen_host") or "")
            if listen_host:
                candidates.append(f"http://{listen_host}:8791")
    candidates.append(DEFAULT_API_URL)
    seen: set[str] = set()
    for candidate in candidates:
        url = str(candidate or "").rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        if fetch_url(url + "/api/hive/status", timeout=timeout).get("ok"):
            return url
    return preferred.rstrip("/")


def auth_headers() -> dict[str, str]:
    headers = {"User-Agent": "ProjectTheseusMacGate/0.1"}
    token = hive_secret()
    if token:
        headers["X-Theseus-Hive-Secret"] = token
    return headers


def hive_secret() -> str:
    if os.environ.get("THESEUS_HIVE_SECRET"):
        return str(os.environ["THESEUS_HIVE_SECRET"])
    for path in [ROOT / "configs" / "hive_join.local.json", Path.home() / "Library" / "Application Support" / "Project Theseus Hive" / "app" / "current" / "configs" / "hive_join.local.json"]:
        data = read_json(path, {})
        if isinstance(data, dict) and data.get("join_token"):
            return str(data.get("join_token") or "")
    return ""


def inspect_wrapper(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return {
        "path": str(path),
        "exists": path.exists(),
        "auto_update_soft": "--auto-update-soft" in text,
        "install_service": "--install-service" in text,
        "enable_service": "--enable-service" in text,
        "start_option_available": "--start" in text,
        "default_foreground_start": "DEFAULT_ARGS=\"${THESEUS_MACOS_INSTALLER_ARGS:---auto-update-soft --install-service --enable-service --start}\"" in text,
        "bounded_api_poll": "Checking local Hive API" in text,
        "overridable": "THESEUS_MACOS_INSTALLER_ARGS" in text,
    }


def inspect_app(path: Path) -> dict[str, Any]:
    info_path = path / "Contents" / "Info.plist"
    info: dict[str, Any] = {}
    if info_path.exists():
        try:
            with info_path.open("rb") as handle:
                info = plistlib.load(handle)
        except (OSError, plistlib.InvalidFileException):
            info = {}
    executable = path / "Contents" / "MacOS" / str(info.get("CFBundleExecutable") or path.stem)
    return {
        "path": str(path),
        "exists": path.exists(),
        "bundle_identifier": info.get("CFBundleIdentifier"),
        "bundle_name": info.get("CFBundleName"),
        "minimum_system_version": info.get("LSMinimumSystemVersion"),
        "lsui_element": bool(info.get("LSUIElement")),
        "executable": inspect_binary(executable),
    }


def inspect_binary(path: Path) -> dict[str, Any]:
    row = file_row(path)
    if not path.exists():
        return row
    head = b""
    try:
        head = path.read_bytes()[:2]
    except OSError:
        pass
    row["script"] = head == b"#!"
    lipo = run_capture(["lipo", "-archs", str(path)], timeout=20)
    row["lipo"] = compact_capture(lipo)
    if lipo.get("ok"):
        row["architectures"] = str(lipo.get("stdout_tail") or "").split()
    return row


def file_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path) if path.is_file() else "",
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def platform_report() -> dict[str, Any]:
    machine = platform.machine()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": machine,
        "python": platform.python_version(),
        "is_apple_silicon": platform.system() == "Darwin" and machine.lower() in {"arm64", "aarch64"},
        "is_intel_mac": platform.system() == "Darwin" and machine.lower() in {"x86_64", "amd64"},
    }


def git_status() -> dict[str, Any]:
    branch = run_text(["git", "branch", "--show-current"]).strip()
    commit = run_text(["git", "rev-parse", "HEAD"]).strip()
    porcelain = run_text(["git", "status", "--porcelain"])
    dirty_paths = [line[3:].strip() if len(line) > 3 else line.strip() for line in porcelain.splitlines() if line.strip()]
    return {"branch": branch, "commit": commit, "short_commit": commit[:12], "dirty": bool(dirty_paths), "dirty_paths": dirty_paths[:80]}


def run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def compact_capture(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in ["ok", "returncode", "error", "message", "stdout_tail", "stderr_tail"] if result.get(key) not in {None, ""}}


def compact_convergence(convergence: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": convergence.get("ok"),
        "execute": convergence.get("execute"),
        "catalog_url": convergence.get("catalog_url"),
        "target_count": convergence.get("target_count"),
        "updated_or_current": convergence.get("updated_or_current"),
        "nodes": [
            {
                "peer_url": row.get("peer_url"),
                "ok": row.get("ok"),
                "status": row.get("status"),
                "hard_update_available": row.get("hard_update_available"),
                "after": row.get("after"),
                "error": row.get("error"),
            }
            for row in convergence.get("nodes", [])[:10]
            if isinstance(row, dict)
        ],
    }


def local_soft_convergence_ok(convergence: dict[str, Any], api_url: str) -> bool:
    if not isinstance(convergence, dict) or not convergence:
        return False
    if bool(convergence.get("ok")):
        return True
    local_urls = {DEFAULT_API_URL.rstrip("/"), str(api_url or "").rstrip("/")}
    for row in convergence.get("nodes", []) if isinstance(convergence.get("nodes"), list) else []:
        if not isinstance(row, dict):
            continue
        peer_url = str(row.get("peer_url") or "").rstrip("/")
        platform_system = str(get_path(row, ["platform", "system"], "") or "")
        is_local = peer_url in local_urls or (peer_url.startswith("http://127.") and ":8791" in peer_url) or platform_system == "Darwin"
        if not is_local:
            continue
        after = row.get("after") if isinstance(row.get("after"), dict) else {}
        status = str(row.get("status") or get_path(after, ["status"], "") or "")
        update_clear = after.get("update_available") is False or after.get("soft_update_available") is False
        if bool(row.get("ok")) and status in {"current", "soft_installed", "updated"} and update_clear:
            return True
        if bool(after.get("ok")) and update_clear:
            return True
    return False


def network_summary(network: dict[str, Any]) -> dict[str, Any]:
    findings = network.get("findings") if isinstance(network.get("findings"), list) else []
    return {
        "ok": network.get("ok"),
        "state": network.get("state"),
        "coordinator": network.get("coordinator"),
        "summary": network.get("summary"),
        "findings": [
            {
                "severity": row.get("severity"),
                "id": row.get("id") or row.get("code"),
                "message": row.get("message") or row.get("title"),
                "fix": row.get("fix") or "; ".join(str(item) for item in row.get("fixes") or []),
            }
            for row in findings[:12]
            if isinstance(row, dict)
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Project Theseus Hive macOS Release Gate",
        "",
        f"- Created UTC: `{report.get('created_utc')}`",
        f"- Private canary ready: `{report.get('private_canary_ready')}`",
        f"- Public distribution ready: `{report.get('public_distribution_ready')}`",
        f"- Fleet rollout ready: `{report.get('fleet_rollout_ready')}`",
        "",
        "## Failed/Pending Gates",
    ]
    failed = [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("ok")]
    if not failed:
        lines.append("- None")
    for row in failed:
        lines.append(f"- `{row.get('severity')}` `{row.get('name')}`: {row.get('message')}")
    lines.extend(["", "## Next Actions"])
    for item in report.get("next_actions", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Artifacts"])
    paths = get_path(report, ["sections", "artifacts", "paths"], {})
    if isinstance(paths, dict):
        for name, info in paths.items():
            if isinstance(info, dict) and info.get("exists"):
                lines.append(f"- `{name}`: `{info.get('path')}` sha256 `{info.get('sha256')}`")
    return "\n".join(lines) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
