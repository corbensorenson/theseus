"""macOS canary, training preflight, role, and install readiness gates.

This is the high-level Mac lane for Project Theseus Hive. It deliberately
reuses the existing runtime doctor, node registry, training orchestrator,
bootstrap bundle, version manager, and release gate helpers instead of creating
another control plane.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
DIST_MACOS = ROOT / "dist" / "macos"
POLICY_PATH = CONFIGS / "hive_policy.json"
ROLE_CONFIG_PATH = CONFIGS / "hive_node_roles.local.json"
DEFAULT_API_URL = "http://127.0.0.1:8791"

sys.path.insert(0, str(ROOT / "scripts"))
import hive_bootstrap  # noqa: E402
import hive_macos_release_gate  # noqa: E402
import hive_node_registry  # noqa: E402
import hive_node_resources  # noqa: E402
import hive_training_orchestrator  # noqa: E402
import hive_version_manager  # noqa: E402
import theseus_runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run macOS-specific Project Theseus Hive readiness gates.")
    sub = parser.add_subparsers(dest="command")

    roles = sub.add_parser("roles", help="Assign and verify this Mac's Hive roles.")
    add_common_out(roles, "reports/macos_role_assignment.json")
    roles.add_argument("--write-local-config", action="store_true")

    preflight = sub.add_parser("training-preflight", help="Gate long Mac training and optionally queue one local worker proof.")
    add_common_out(preflight, "reports/macos_training_preflight.json")
    preflight.add_argument("--execute", action="store_true")
    preflight.add_argument("--profile", default="smoke")
    preflight.add_argument("--timeout", type=float, default=90.0)
    preflight.add_argument("--min-disk-gib", type=float, default=25.0)
    preflight.add_argument("--min-battery-percent", type=float, default=45.0)
    preflight.add_argument("--allow-battery-smoke", action="store_true")
    preflight.add_argument("--offline", action="store_true")

    dmg = sub.add_parser("dmg-readiness", help="Build/check DMG, package, version catalog, and installed-app update readiness.")
    add_common_out(dmg, "reports/macos_dmg_readiness_gate.json")
    dmg.add_argument("--execute", action="store_true")
    dmg.add_argument("--skip-build", action="store_true")
    dmg.add_argument("--skip-version-publish", action="store_true")
    dmg.add_argument("--api-url", default=DEFAULT_API_URL)

    join = sub.add_parser("join-bundle", help="Write one-click Mac/iPhone/Watch join artifacts.")
    add_common_out(join, "reports/macos_join_bundle_status.json")
    join.add_argument("--bundle-out", default="dist/macos/ProjectTheseusHive.join.json")
    join.add_argument("--qr-out", default="dist/macos/ProjectTheseusHive.join.svg")
    join.add_argument("--no-token", action="store_true")
    join.add_argument("--coordinator-url", action="append", default=[])
    join.add_argument("--relay-url", action="append", default=[])
    join.add_argument("--operator-token-scope", default="")

    app_status = sub.add_parser("app-status", help="Report installed-app status lines: Joined, Running, Update OK, Training Ready.")
    add_common_out(app_status, "reports/macos_app_install_status.json")
    app_status.add_argument("--api-url", default=DEFAULT_API_URL)
    app_status.add_argument("--text", action="store_true")

    canary = sub.add_parser("canary", help="Run Mac role, app, training, DMG, and join-bundle gates together.")
    add_common_out(canary, "reports/macos_canary.json")
    canary.add_argument("--execute", action="store_true")
    canary.add_argument("--write-local-config", action="store_true")
    canary.add_argument("--skip-build", action="store_true")
    canary.add_argument("--skip-version-publish", action="store_true")
    canary.add_argument("--skip-training-execute", action="store_true")
    canary.add_argument("--write-join-bundle", action="store_true")
    canary.add_argument("--api-url", default=DEFAULT_API_URL)

    args = parser.parse_args()
    policy = read_json(POLICY_PATH, {})
    if args.command in {None, "canary"}:
        report = canary_report(policy, args)
    elif args.command == "roles":
        report = role_assignment_report(policy, write_local=bool(args.write_local_config))
    elif args.command == "training-preflight":
        report = training_preflight_report(policy, args)
    elif args.command == "dmg-readiness":
        report = dmg_readiness_report(policy, args)
    elif args.command == "join-bundle":
        report = join_bundle_report(policy, args)
    elif args.command == "app-status":
        report = app_status_report(policy, api_url=str(args.api_url or DEFAULT_API_URL))
        if args.text:
            print(app_status_text(report))
    else:
        parser.print_help()
        return 2

    out = resolve(str(getattr(args, "out", "") or "reports/macos_canary.json"))
    write_json(out, report)
    if str(getattr(args, "markdown_out", "") or ""):
        resolve(str(args.markdown_out)).write_text(markdown_report(report), encoding="utf-8")
    if not bool(getattr(args, "text", False)):
        print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def add_common_out(parser: argparse.ArgumentParser, default_out: str) -> None:
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--markdown-out", default="")


def canary_report(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    role = role_assignment_report(policy, write_local=bool(getattr(args, "write_local_config", False) or getattr(args, "execute", False)))
    preflight_args = argparse.Namespace(
        execute=bool(getattr(args, "execute", False)) and not bool(getattr(args, "skip_training_execute", False)),
        profile="smoke",
        timeout=90.0,
        min_disk_gib=25.0,
        min_battery_percent=45.0,
        allow_battery_smoke=True,
        offline=True,
    )
    training = training_preflight_report(policy, preflight_args)
    dmg_args = argparse.Namespace(
        execute=bool(getattr(args, "execute", False)),
        skip_build=bool(getattr(args, "skip_build", False)),
        skip_version_publish=bool(getattr(args, "skip_version_publish", False)),
        api_url=str(getattr(args, "api_url", "") or DEFAULT_API_URL),
    )
    dmg = dmg_readiness_report(policy, dmg_args)
    app = app_status_report(policy, api_url=str(getattr(args, "api_url", "") or DEFAULT_API_URL))
    join = {}
    if bool(getattr(args, "write_join_bundle", False) or getattr(args, "execute", False)):
        join_args = argparse.Namespace(
            bundle_out="dist/macos/ProjectTheseusHive.join.json",
            qr_out="dist/macos/ProjectTheseusHive.join.svg",
            no_token=False,
            coordinator_url=[],
            relay_url=[],
            operator_token_scope="",
        )
        join = join_bundle_report(policy, join_args)

    gates = []
    gates.extend(prefix_gates("roles", role.get("gates", [])))
    gates.extend(prefix_gates("training", training.get("gates", [])))
    gates.extend(prefix_gates("dmg", dmg.get("gates", [])))
    gates.extend(prefix_gates("app", app.get("gates", [])))
    if join:
        gates.extend(prefix_gates("join", join.get("gates", [])))
    return finish(
        "project_theseus_macos_canary_v0",
        gates,
        {
            "platform": platform_report(),
            "roles": role,
            "training_preflight": training,
            "dmg_readiness": dmg,
            "app_status": app,
            "join_bundle": join,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "next_commands": [
                "theseus mac training-preflight --execute --offline --allow-battery-smoke",
                "theseus mac dmg-readiness --execute",
                "theseus mac join-bundle",
            ],
        },
    )


def role_assignment_report(policy: dict[str, Any], *, write_local: bool = False) -> dict[str, Any]:
    resources = hive_node_resources.probe_resources(policy)
    caps = hive_node_resources.classify_capabilities(resources, policy)
    capability_ids = sorted({str(cap.get("id") or "") for cap in caps if cap.get("id")})
    status = fetch_json(DEFAULT_API_URL.rstrip("/") + "/api/hive/status", timeout=1.5) or read_json(REPORTS / "hive_status.json", {})
    live_caps = sorted({str(row.get("id") or "") for row in status.get("capabilities", []) if isinstance(row, dict)})
    runtime = theseus_runtime.runtime_doctor_report(write_report=True)
    runtime_mlx = runtime_mlx_available(runtime)
    platform_info = platform_report()
    effective_capability_ids = sorted(set(capability_ids) | set(live_caps) | ({"mlx_apple", "apple_mlx"} if runtime_mlx and platform_info["is_apple_silicon"] else set()))
    roles = assigned_roles(policy, platform_info, effective_capability_ids)
    expected = role_expectations(platform_info, effective_capability_ids)
    config = {
        "policy": "project_theseus_hive_node_roles_local_v0",
        "created_utc": now(),
        "roles": roles,
        "platform": platform_info,
        "capability_ids": effective_capability_ids,
        "active_shell_capability_ids": capability_ids,
        "live_capability_ids": live_caps,
        "expected": expected,
        "notes": "Local ignored role assignment. Capabilities remain probed from hardware/runtime and must not be faked.",
    }
    if write_local:
        write_json(ROLE_CONFIG_PATH, config)
    gates = [
        gate("macos_host", platform_info["system"] == "Darwin", "hard", "Mac role assignment must run on macOS.", platform_info),
        gate("cpu_worker_present", "cpu_worker" in effective_capability_ids, "hard", "Every Mac must advertise CPU worker capability.", effective_capability_ids),
        gate("intel_does_not_advertise_mlx", not platform_info["is_intel"] or not ({"mlx_apple", "apple_mlx", "mlx_cuda"} & set(effective_capability_ids)), "hard", "Intel Macs must not advertise MLX accelerator roles.", effective_capability_ids),
        gate("apple_silicon_mlx_backed_by_runtime", not platform_info["is_apple_silicon"] or not ({"mlx_apple", "apple_mlx"} & set(effective_capability_ids)) or runtime_mlx or bool({"mlx_apple", "apple_mlx"} & set(live_caps)), "hard", "Apple Silicon MLX role must be backed by source/installed runtime or live Hive status.", {"active_shell_mlx": resources.get("mlx", {}), "runtime": compact_runtime(runtime), "live_capability_ids": live_caps}),
        gate("live_role_config_written", not write_local or ROLE_CONFIG_PATH.exists(), "hard", "Local role config is written when requested.", {"path": rel(ROLE_CONFIG_PATH)}),
    ]
    return finish(
        "project_theseus_macos_role_assignment_v0",
        gates,
        {
            "platform": platform_info,
            "roles": roles,
            "expected": expected,
            "resources": compact_resources(resources),
            "runtime": compact_runtime(runtime),
            "capability_ids": effective_capability_ids,
            "active_shell_capability_ids": capability_ids,
            "live_capability_ids": live_caps,
            "local_config_path": rel(ROLE_CONFIG_PATH),
            "local_config_written": bool(write_local and ROLE_CONFIG_PATH.exists()),
        },
    )


def training_preflight_report(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    platform_info = platform_report()
    runtime = theseus_runtime.runtime_doctor_report(write_report=True)
    resources = hive_node_resources.probe_resources(policy)
    role = role_assignment_report(policy, write_local=False)
    gates = training_resource_gates(runtime, resources, platform_info, args, role)
    execute = bool(getattr(args, "execute", False))
    execution: dict[str, Any] = {}
    if execute:
        execution = run_training_worker_canary(policy, args, platform_info)
        gates.extend(training_execution_gates(execution, platform_info))
    else:
        gates.append(gate("worker_canary_execute_requested", False, "advisory", "Run with --execute to queue and verify one local worker report.", {}))
    long_training_allowed = not any(row["ok"] is False and row["severity"] == "hard" for row in gates if row["name"].startswith("resource_") or row["name"].startswith("runtime_"))
    if get_path(resources, ["power", "on_ac_power"], True) is False:
        long_training_allowed = False
    return finish(
        "project_theseus_macos_training_preflight_v0",
        gates,
        {
            "platform": platform_info,
            "state": gate_state(gates),
            "long_training_allowed": long_training_allowed,
            "bounded_smoke_allowed": not any(row["ok"] is False and row["severity"] == "hard" for row in gates),
            "runtime": compact_runtime(runtime),
            "resources": compact_resources(resources),
            "role_assignment": role.get("summary", {}),
            "execution": execution,
            "offline_contract": {
                "artifact_sync": "forbidden_in_offline_preflight",
                "teacher": "forbidden",
                "external_inference": "forbidden",
                "remote_peer_queueing": "disabled_when_offline",
            },
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
    )


def training_resource_gates(runtime: dict[str, Any], resources: dict[str, Any], platform_info: dict[str, Any], args: argparse.Namespace, role: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    disk_free = to_float(get_path(resources, ["disk", "free_gib"], 0.0))
    min_disk = float(getattr(args, "min_disk_gib", 25.0) or 25.0)
    power = resources.get("power") if isinstance(resources.get("power"), dict) else {}
    battery = to_float(power.get("battery_percent"))
    min_battery = float(getattr(args, "min_battery_percent", 45.0) or 45.0)
    allow_battery_smoke = bool(getattr(args, "allow_battery_smoke", False))
    battery_smoke_floor = min(min_battery, 20.0) if allow_battery_smoke else min_battery
    battery_ok = (
        power.get("on_ac_power") is True
        or battery is None
        or battery >= min_battery
        or (allow_battery_smoke and battery >= battery_smoke_floor)
    )
    thermal = resources.get("thermal") if isinstance(resources.get("thermal"), dict) else {}
    mlx = resources.get("mlx") if isinstance(resources.get("mlx"), dict) else {}
    role_caps = set((role or {}).get("capability_ids") or get_path(role or {}, ["summary", "capability_ids"], []) or [])
    runtime_mlx = runtime_mlx_available(runtime)
    mlx_available = bool(mlx.get("available")) or runtime_mlx or bool({"mlx_apple", "apple_mlx"} & role_caps)
    mlx_evidence = {
        "active_shell_mlx": mlx,
        "runtime": compact_runtime(runtime),
        "role_capability_ids": sorted(role_caps),
        "available": mlx_available,
    }
    gates = [
        gate("runtime_doctor_not_red", runtime.get("state") != "RED", "hard", "Runtime doctor must not be RED before training.", compact_runtime(runtime)),
        gate("resource_disk_floor", disk_free >= min_disk, "hard", f"Training needs at least {min_disk:g} GiB free disk/cache space.", resources.get("disk", {})),
        gate("resource_thermal_not_hot", str(thermal.get("state") or "nominal") not in {"hot", "throttled"}, "hard", "Training is rejected while thermal state is hot/throttled.", thermal),
        gate(
            "resource_battery_floor",
            battery_ok,
            "hard",
            (
                f"Battery must be on AC or >= {min_battery:g}% for training; "
                f"--allow-battery-smoke permits bounded canaries at >= {battery_smoke_floor:g}%."
            ),
            {**power, "allow_battery_smoke": allow_battery_smoke, "battery_smoke_floor": battery_smoke_floor},
        ),
        gate("resource_long_training_ac_power", power.get("on_ac_power") is True, "advisory", "Long training should run on AC power; battery mode is only for bounded smoke/canary work.", power),
    ]
    if platform_info["is_apple_silicon"]:
        gates.append(gate("resource_apple_mlx_available", mlx_available, "hard", "Apple Silicon training preflight requires usable MLX in the installed/preferred Hive runtime.", mlx_evidence))
    if platform_info["is_intel"]:
        gates.append(gate("resource_intel_no_mlx", not mlx_available, "hard", "Intel Mac must not claim MLX; it joins as CPU/storage/operator.", mlx_evidence))
    return gates


def run_training_worker_canary(policy: dict[str, Any], args: argparse.Namespace, platform_info: dict[str, Any]) -> dict[str, Any]:
    if platform_info["is_intel"]:
        return run_intel_cpu_canary(policy, args)
    round_id = "mac-preflight-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = hive_training_orchestrator.orchestrate(
        policy,
        profile=str(getattr(args, "profile", "") or "smoke"),
        run_id="mac-training-preflight",
        round_id=round_id,
        execute=True,
        sync=False,
        max_jobs=1,
        allow_wan=False,
        local_only=True,
    )
    paths = expected_worker_report_paths(report)
    worker = poll_worker_reports(paths, timeout=float(getattr(args, "timeout", 90.0) or 90.0), expected_chunk_ids=expected_chunk_ids(report))
    return {
        "kind": "apple_silicon_mlx_worker_canary",
        "ok": bool(report.get("ok") and worker.get("ok")),
        "orchestrator": compact_orchestrator(report),
        "expected_worker_reports": [rel(path) for path in paths],
        "worker_report": worker,
        "artifact_sync_empty": not bool(report.get("artifact_sync")),
        "teacher_used": bool(get_path(worker, ["payload", "teacher_used"], False)),
        "external_inference_calls": int(get_path(worker, ["payload", "external_inference_calls"], 0) or 0),
    }


def run_intel_cpu_canary(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    # Intel Macs are not accelerator workers. The CPU canary verifies the node can
    # accept a registered light task without advertising unsupported MLX.
    payload = {"job_id": "mac-intel-cpu-canary-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "force_requeue": True}
    body = json.dumps({"kind": "resource_probe", "payload": payload}).encode("utf-8")
    result = fetch_post_json(DEFAULT_API_URL.rstrip() + "/api/hive/tasks", body, timeout=5.0)
    return {
        "kind": "intel_cpu_operator_canary",
        "ok": bool(result.get("ok")),
        "task": result,
        "artifact_sync_empty": True,
        "teacher_used": False,
        "external_inference_calls": 0,
    }


def training_execution_gates(execution: dict[str, Any], platform_info: dict[str, Any]) -> list[dict[str, Any]]:
    gates = [
        gate("worker_canary_queued", bool(get_path(execution, ["orchestrator", "queued_task_kinds"]) or get_path(execution, ["task", "ok"])), "hard", "Preflight must queue a registered local task.", execution),
        gate("worker_canary_no_artifact_sync", bool(execution.get("artifact_sync_empty")), "hard", "Offline Mac preflight must not run Hive artifact sync.", execution),
        gate("worker_canary_no_teacher", not bool(execution.get("teacher_used")) and int(execution.get("external_inference_calls") or 0) == 0, "hard", "Worker canary must not use teacher or external inference.", execution),
    ]
    if platform_info["is_apple_silicon"]:
        gates.extend(
            [
                gate("mlx_worker_chunk_queued", "mlx_training_chunk" in set(get_path(execution, ["orchestrator", "queued_task_kinds"], [])), "hard", "Apple Silicon canary must queue an MLX training chunk.", execution.get("orchestrator", {})),
                gate("mlx_worker_report_written", bool(get_path(execution, ["worker_report", "ok"], False)), "hard", "Apple Silicon canary must write the expected MLX worker report.", execution.get("worker_report", {})),
            ]
        )
    if platform_info["is_intel"]:
        gates.append(gate("intel_cpu_canary_ok", bool(execution.get("ok")), "hard", "Intel Mac canary must accept a CPU/operator task.", execution))
    return gates


def dmg_readiness_report(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    actions: list[dict[str, Any]] = []
    if bool(getattr(args, "execute", False)) and not bool(getattr(args, "skip_build", False)):
        env = dict(os.environ)
        env["THESEUS_MACOS_BUILD_DMG"] = "1"
        actions.append(run_action("build_macos_artifacts", ["./scripts/package_theseus_macos.sh"], timeout=3600, env=env))
    if bool(getattr(args, "execute", False)) and not bool(getattr(args, "skip_version_publish", False)):
        py = preferred_python()
        actions.extend(
            [
                run_action("verify_hive_version", [py, "scripts/hive_version_manager.py", "verify", "--out", "reports/hive_verified_version.json"], timeout=300),
                run_action("publish_hive_catalog", [py, "scripts/hive_version_manager.py", "publish", "--out", "reports/hive_update_catalog.json"], timeout=300),
                run_action("refresh_installer_artifacts", [py, "scripts/hive_version_manager.py", "installer-artifacts", "--out", "reports/hive_installer_artifacts.json"], timeout=300),
                run_action("refresh_hive_version_status", [py, "scripts/hive_version_manager.py", "status", "--out", "reports/hive_version_status.json"], timeout=300),
            ]
        )

    artifacts = hive_macos_release_gate.inspect_macos_artifacts()
    version = hive_macos_release_gate.inspect_version_state()
    app = app_status_report(policy, api_url=str(getattr(args, "api_url", "") or DEFAULT_API_URL))
    current_commit = git_commit()
    verified_commit = str(get_path(version, ["verified", "git", "commit"], "") or "")
    gates = []
    gates.extend(hive_macos_release_gate.artifact_gates(artifacts))
    gates.extend(hive_macos_release_gate.version_gates(version))
    gates.extend(
        [
            gate("verified_commit_matches_current", bool(current_commit and verified_commit == current_commit), "hard", "Published verified version must match this source commit.", {"current_commit": current_commit, "verified_commit": verified_commit}),
            gate("dmg_update_catalog_served_by_app", bool(get_path(app, ["summary", "update_ok"], False)), "hard", "Installed app/Hive node must serve update catalog without Codex.", app.get("summary", {})),
            gate("dmg_installer_artifacts_served_by_app", bool(get_path(app, ["summary", "installer_artifacts_ok"], False)), "hard", "Installed app/Hive node must serve installer artifacts without Codex.", app.get("summary", {})),
        ]
    )
    for action in actions:
        gates.append(gate("action_" + safe_id(action["name"]), bool(action.get("ok")), "hard", f"Action {action['name']} must succeed.", action))
    return finish(
        "project_theseus_macos_dmg_readiness_gate_v0",
        gates,
        {
            "platform": platform_report(),
            "current_commit": current_commit,
            "actions": actions,
            "artifacts": compact_artifacts(artifacts),
            "version": compact_version(version),
            "app_status": app.get("summary", {}),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
    )


def join_bundle_report(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    bundle_args = argparse.Namespace(
        out=str(getattr(args, "bundle_out", "") or "dist/macos/ProjectTheseusHive.join.json"),
        qr_out=str(getattr(args, "qr_out", "") or "dist/macos/ProjectTheseusHive.join.svg"),
        no_token=bool(getattr(args, "no_token", False)),
        coordinator_url=list(getattr(args, "coordinator_url", []) or []),
        relay_url=list(getattr(args, "relay_url", []) or []),
        operator_token_scope=str(getattr(args, "operator_token_scope", "") or ""),
    )
    report = hive_bootstrap.write_bootstrap_bundle(policy, bundle_args)
    command_path = write_macos_join_command(bundle_args.out) if report.get("ok") else ""
    gates = [
        gate("join_bundle_written", bool(report.get("ok") and Path(ROOT / bundle_args.out).exists()), "hard", "Token-scoped Mac join bundle is written.", report),
        gate("join_bundle_no_codex_required", bool(report.get("ok")), "hard", "Join bundle can be imported by installed app/CLI without Codex.", report),
        gate("join_bundle_command_written", bool(command_path and (ROOT / command_path).exists()), "hard", "A double-click command is available for non-Codex Mac join.", {"command_path": command_path}),
    ]
    return finish(
        "project_theseus_macos_join_bundle_v0",
        gates,
        {
            "bundle_report": report,
            "bundle_path": bundle_args.out,
            "qr_path": bundle_args.qr_out,
            "mac_join_command": command_path,
            "security": {
                "token_bearing": not bool(bundle_args.no_token),
                "treat_as_password": not bool(bundle_args.no_token),
                "arbitrary_shell": False,
            },
            "install_contract": {
                "other_mac": "Install ProjectTheseusHive.dmg/pkg, then double-click the join command or import ProjectTheseusHive.join.json in the app/CLI.",
                "iphone_watch": "Scan/open the generated QR/profile in the native iPhone app; Watch receives the profile from iPhone.",
            },
        },
    )


def app_status_report(policy: dict[str, Any], *, api_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    join_cfg = read_json(CONFIGS / "hive_join.local.json", {})
    base_url = api_url.rstrip("/")
    status = fetch_json(base_url + "/api/hive/status", timeout=5.0)
    update = fetch_json(base_url + "/api/hive/update-catalog", timeout=3.0)
    installer = fetch_json(base_url + "/api/hive/installer-artifacts", timeout=3.0)
    training = read_json(REPORTS / "macos_training_preflight.json", {})
    role = read_json(REPORTS / "macos_role_assignment.json", {})
    joined = bool(join_cfg.get("hive_id") or (status or {}).get("hive_id"))
    running = bool(status and status.get("ok", True) is not False)
    update_ok = bool(update and update.get("ok", True) is not False)
    installer_ok = bool(installer and installer.get("ok", True) is not False)
    role_caps = set(role.get("capability_ids") or get_path(role, ["summary", "capability_ids"], []) or [])
    live_caps = {str(row.get("id") or "") for row in status.get("capabilities", []) if isinstance(row, dict)}
    live_roles = {str(item) for item in status.get("roles", []) if str(item)}
    readiness_caps = role_caps | live_caps
    readiness_roles = set(get_path(role, ["roles"], []) or []) | live_roles
    training_ready = bool(
        training.get("bounded_smoke_allowed")
        or get_path(training, ["summary", "bounded_smoke_allowed"], False)
        or {"mlx_apple", "apple_mlx", "cpu_worker"} & readiness_caps
        or {"mlx_training", "mlx_eval", "cpu_worker", "operator", "storage"} & readiness_roles
    )
    gates = [
        gate("app_joined", joined, "hard", "Installed app has joined or can identify a Hive.", compact_join_config(join_cfg)),
        gate("app_running", running, "hard", "Local Hive app/API is running.", {"api_url": api_url, "status": compact_status(status or {})}),
        gate("app_update_ok", update_ok, "hard", "Local app/API serves the update catalog.", compact_endpoint(update or {})),
        gate("app_installer_artifacts_ok", installer_ok, "hard", "Local app/API serves installer artifacts.", compact_endpoint(installer or {})),
        gate("app_training_ready", training_ready, "hard", "Local app has a training/operator role ready for this Mac class.", {"training": training.get("summary", {}), "role": role.get("summary", {}), "live_capability_ids": sorted(live_caps), "live_roles": sorted(live_roles)}),
    ]
    summary = {
        "joined": joined,
        "running": running,
        "update_ok": update_ok,
        "installer_artifacts_ok": installer_ok,
        "training_ready": training_ready,
        "api_url": api_url,
    }
    return finish(
        "project_theseus_macos_app_status_v0",
        gates,
        {
            "summary": summary,
            "status_lines": app_status_lines(summary),
            "join_config_present": bool(join_cfg),
            "status": compact_status(status or {}),
            "update_catalog": compact_endpoint(update or {}),
            "installer_artifacts": compact_endpoint(installer or {}),
        },
    )


def assigned_roles(policy: dict[str, Any], platform_info: dict[str, Any], capability_ids: list[str]) -> list[str]:
    common = ["operator", "storage", "checkpoint_chat", "artifact_sync", "update_client"]
    if platform_info["is_apple_silicon"] and {"mlx_apple", "apple_mlx"} & set(capability_ids):
        return ["mlx_training", "mlx_eval", "mlx_rollout", *common]
    if platform_info["is_intel"]:
        return ["cpu_worker", "lightweight_orchestration", *common]
    if "nvidia_cuda" in capability_ids:
        return ["cuda_training", "cuda_eval", "cuda_rollout", *common]
    return ["cpu_worker", *common]


def role_expectations(platform_info: dict[str, Any], capability_ids: list[str]) -> dict[str, Any]:
    caps = set(capability_ids)
    return {
        "apple_silicon": "mlx_training/mlx_eval/operator/storage when MLX is available; CPU/operator otherwise",
        "intel": "cpu_worker/storage/operator/lightweight_orchestration; never MLX",
        "this_mac_class": "apple_silicon" if platform_info["is_apple_silicon"] else "intel" if platform_info["is_intel"] else platform_info["machine"],
        "advertises_mlx": bool({"mlx_apple", "apple_mlx", "mlx_cuda"} & caps),
    }


def runtime_mlx_available(runtime: dict[str, Any]) -> bool:
    preferred = runtime.get("preferred_runtime") if isinstance(runtime.get("preferred_runtime"), dict) else {}
    if get_path(preferred, ["mlx", "available"], False):
        return True
    rows = runtime.get("python_runtimes") if isinstance(runtime.get("python_runtimes"), list) else []
    return any(get_path(row, ["mlx", "available"], False) for row in rows if isinstance(row, dict))


def expected_worker_report_paths(report: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    jobs = get_path(report, ["plan", "jobs"], [])
    for job in jobs if isinstance(jobs, list) else []:
        for artifact in get_path(job, ["payload", "output_artifacts"], []) if isinstance(get_path(job, ["payload", "output_artifacts"], []), list) else []:
            path = str(artifact.get("path") or "")
            if path:
                out.extend(candidate_report_paths(path))
        task_kind = str(job.get("task_kind") or "")
        if task_kind:
            out.extend(candidate_report_paths(f"reports/hive_chunks/{task_kind}_last.json"))
    return list(dict.fromkeys(out))


def candidate_report_paths(path_text: str) -> list[Path]:
    path = Path(path_text)
    out = [ROOT / path]
    parts = path.parts
    if parts and parts[0] == "reports":
        suffix = Path(*parts[1:]) if len(parts) > 1 else Path()
        for root in report_search_roots():
            out.append(root / suffix)
    return unique_paths(out)


def report_search_roots() -> list[Path]:
    roots = [REPORTS]
    try:
        roots.extend(hive_training_orchestrator.report_roots())
    except Exception:
        pass
    roots.extend(
        [
            Path.home() / "Library" / "Application Support" / "Project Theseus Hive" / "runtime" / "reports",
            Path.home() / "Library" / "Application Support" / "ProjectTheseus" / "runtime" / "reports",
        ]
    )
    return unique_paths(roots)


def unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.expanduser().resolve())
        except OSError:
            key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(path.expanduser())
    return out


def expected_chunk_ids(report: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    jobs = get_path(report, ["plan", "jobs"], [])
    for job in jobs if isinstance(jobs, list) else []:
        chunk_id = str(get_path(job, ["payload", "chunk_id"], "") or "")
        if chunk_id:
            out.add(chunk_id)
        for artifact in get_path(job, ["payload", "output_artifacts"], []) if isinstance(get_path(job, ["payload", "output_artifacts"], []), list) else []:
            path = Path(str(artifact.get("path") or ""))
            if path.suffix == ".json" and path.stem:
                out.add(path.stem)
    return out


def poll_worker_reports(paths: list[Path], *, timeout: float, expected_chunk_ids: set[str] | None = None) -> dict[str, Any]:
    deadline = time.time() + max(1.0, timeout)
    expected = set(expected_chunk_ids or set())
    while time.time() < deadline:
        for path in paths:
            payload = read_json(path, {})
            if payload and payload.get("policy") == "project_theseus_hive_worker_chunk_v0":
                chunk_id = str(payload.get("chunk_id") or "")
                if expected and chunk_id not in expected:
                    continue
                return {"ok": bool(payload.get("ok")), "path": rel(path), "payload": compact_worker(payload)}
        time.sleep(1.0)
    return {"ok": False, "error": "worker_report_timeout", "waited_seconds": timeout, "paths": [rel(path) for path in paths], "expected_chunk_ids": sorted(expected)}


def compact_worker(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": payload.get("ok"),
        "kind": payload.get("kind"),
        "backend": payload.get("backend"),
        "chunk_id": payload.get("chunk_id"),
        "profile": payload.get("profile"),
        "runtime_ms": payload.get("runtime_ms"),
        "teacher_used": payload.get("teacher_used"),
        "external_inference_calls": payload.get("external_inference_calls"),
        "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
        "work_receipt": payload.get("work_receipt") if isinstance(payload.get("work_receipt"), dict) else {},
    }


def compact_orchestrator(report: dict[str, Any]) -> dict[str, Any]:
    jobs = get_path(report, ["plan", "jobs"], [])
    execution = report.get("execution") if isinstance(report.get("execution"), list) else []
    return {
        "ok": report.get("ok"),
        "mode": report.get("mode"),
        "run_id": report.get("run_id"),
        "round_id": report.get("round_id"),
        "profile": report.get("profile"),
        "node_count": report.get("node_count"),
        "queued_task_kinds": [str(job.get("task_kind") or "") for job in jobs if isinstance(job, dict)],
        "execution": [{"ok": row.get("ok"), "job_id": row.get("job_id"), "task_kind": row.get("task_kind")} for row in execution],
        "artifact_sync": report.get("artifact_sync"),
        "fleet_readiness": report.get("fleet_readiness"),
    }


def write_macos_join_command(bundle_path: str) -> str:
    out = DIST_MACOS / "Join Project Theseus Hive.command"
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle_abs = (ROOT / bundle_path).resolve()
    script = f"""#!/usr/bin/env sh
set -eu
cd "{ROOT}"
if command -v theseus >/dev/null 2>&1; then
  exec theseus join --invite "{bundle_abs}" --start
fi
exec "{preferred_python()}" scripts/theseus_cli.py join --invite "{bundle_abs}" --start
"""
    out.write_text(script, encoding="utf-8")
    out.chmod(0o755)
    return rel(out)


def prefix_gates(prefix: str, gates: list[Any]) -> list[dict[str, Any]]:
    out = []
    for row in gates:
        if not isinstance(row, dict):
            continue
        copied = dict(row)
        copied["name"] = prefix + "_" + str(copied.get("name") or "gate")
        out.append(copied)
    return out


def gate(name: str, ok: bool, severity: str, message: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "severity": severity, "message": message, "evidence": evidence}


def finish(policy_name: str, gates: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    hard_failures = [row for row in gates if row.get("severity") == "hard" and not row.get("ok")]
    public_failures = [row for row in gates if row.get("severity") == "public" and not row.get("ok")]
    state = "RED" if hard_failures else "YELLOW" if public_failures or any(not row.get("ok") for row in gates) else "GREEN"
    return {
        "ok": not hard_failures,
        "policy": policy_name,
        "created_utc": now(),
        "state": state,
        "summary": {
            **payload.get("summary", {}),
            "gate_count": len(gates),
            "hard_failures": len(hard_failures),
            "public_or_advisory_failures": len([row for row in gates if row not in hard_failures and not row.get("ok")]),
        },
        "gates": gates,
        **payload,
    }


def gate_state(gates: list[dict[str, Any]]) -> str:
    if any(row.get("severity") == "hard" and not row.get("ok") for row in gates):
        return "RED"
    if any(not row.get("ok") for row in gates):
        return "YELLOW"
    return "GREEN"


def platform_report() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine().lower()
    return {
        "system": system,
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "is_macos": system == "Darwin",
        "is_apple_silicon": system == "Darwin" and machine in {"arm64", "aarch64"},
        "is_intel": system == "Darwin" and machine in {"x86_64", "amd64"},
    }


def compact_runtime(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": report.get("state"),
        "blockers": report.get("blockers"),
        "warnings": report.get("warnings"),
        "preferred_runtime": report.get("preferred_runtime"),
        "launchagents": report.get("launchagents"),
    }


def compact_resources(resources: dict[str, Any]) -> dict[str, Any]:
    return {
        "cpu": resources.get("cpu"),
        "disk": resources.get("disk"),
        "power": resources.get("power"),
        "thermal": resources.get("thermal"),
        "mlx": resources.get("mlx"),
        "nvidia": resources.get("nvidia"),
    }


def compact_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "paths": artifacts.get("paths"),
        "wrapper": artifacts.get("wrapper"),
        "notarized": artifacts.get("notarized"),
        "ad_hoc_signed": artifacts.get("ad_hoc_signed"),
        "compatibility": get_path(artifacts, ["manifest", "macos_compatibility"], {}),
    }


def compact_version(version: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": get_path(version, ["status", "convergence"], {}),
        "verified_version": get_path(version, ["verified", "version_id"], ""),
        "verified_commit": get_path(version, ["verified", "git", "commit"], ""),
        "catalog_ok": get_path(version, ["catalog", "ok"], False),
        "offer_count": len(get_path(version, ["catalog", "offers"], []) or []),
    }


def compact_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": status.get("ok", True) if status else False,
        "node_id": status.get("node_id"),
        "node_name": status.get("node_name"),
        "hive_id": status.get("hive_id"),
        "roles": status.get("roles"),
        "capabilities": [row.get("id") for row in status.get("capabilities", []) if isinstance(row, dict)],
    }


def compact_join_config(join_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": join_cfg.get("policy"),
        "hive_id": join_cfg.get("hive_id"),
        "hive_name": join_cfg.get("hive_name"),
        "tier": join_cfg.get("tier"),
        "coordinator_url": join_cfg.get("coordinator_url"),
        "relay_url": join_cfg.get("relay_url"),
        "join_token_configured": bool(join_cfg.get("join_token")),
        "token_printed": False,
    }


def compact_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": payload.get("ok", True) if payload else False,
        "policy": payload.get("policy"),
        "created_utc": payload.get("created_utc"),
        "keys": sorted(payload.keys())[:20] if isinstance(payload, dict) else [],
    }


def app_status_lines(summary: dict[str, Any]) -> list[str]:
    def line(label: str, ok: bool) -> str:
        return f"{label}: {'OK' if ok else 'Needs attention'}"

    return [
        line("Joined", bool(summary.get("joined"))),
        line("Running", bool(summary.get("running"))),
        line("Update OK", bool(summary.get("update_ok"))),
        line("Installer Artifacts", bool(summary.get("installer_artifacts_ok"))),
        line("Training Ready", bool(summary.get("training_ready"))),
    ]


def app_status_text(report: dict[str, Any]) -> str:
    lines = list(report.get("status_lines") or app_status_lines(report.get("summary", {})))
    lines.append(f"Report: {rel(REPORTS / 'macos_app_install_status.json')}")
    return "\n".join(lines)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('policy', 'macos report')}",
        "",
        f"- State: {report.get('state')}",
        f"- OK: {report.get('ok')}",
        f"- Created: {report.get('created_utc')}",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []) if isinstance(report.get("gates"), list) else []:
        mark = "PASS" if row.get("ok") else "FAIL"
        lines.append(f"- {mark} `{row.get('name')}` ({row.get('severity')}): {row.get('message')}")
    lines.append("")
    return "\n".join(lines)


def run_action(name: str, command: list[str], *, timeout: int, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, env=env)
        return {
            "name": name,
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "command": command,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "ok": False,
            "returncode": 124,
            "command": command,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }


def preferred_python() -> str:
    rows = theseus_runtime.python_runtime_checks()
    preferred = theseus_runtime.preferred_python_runtime(rows)
    return str(preferred.get("python") or sys.executable)


def fetch_json(url: str, *, timeout: float = 2.0) -> dict[str, Any]:
    try:
        with urlrequest.urlopen(url, timeout=timeout) as response:  # noqa: S310 - local/private Hive endpoint.
            raw = response.read().decode("utf-8")
    except (OSError, URLError):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def fetch_post_json(url: str, body: bytes, *, timeout: float = 5.0) -> dict[str, Any]:
    req = urlrequest.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local/private Hive endpoint.
            raw = response.read().decode("utf-8")
    except (OSError, URLError) as exc:
        return {"ok": False, "error": "post_failed", "message": str(exc), "url": url}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_response", "body": raw[:500]}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_response"}


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def safe_id(value: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value).strip()).strip("-._").lower()
    return slug or "id"


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return default
        if cur is None:
            return default
    return cur


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
