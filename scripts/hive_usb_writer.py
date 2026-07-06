"""Universal Project Theseus Hive USB installer writer.

This is the single source for portable Hive installer USB/zip bundles. It
creates one clean payload plus small platform launchers for Windows, macOS, and
Linux. The platform launchers copy the payload to the local machine first, then
run the OS-specific installer from that local app directory.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hive_profiles


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "dist" / "universal-usb" / "ProjectTheseusUniversalUSB"

EXCLUDE_DIR_NAMES = {
    ".attd_tmp",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "checkpoints",
    "dist",
    "games",
    "node_modules",
    "reports",
    "target",
    "tmp",
    "updates",
    "vendor",
}
EXCLUDE_DIR_PREFIXES = (".venv",)
HEAVY_DATA_DIRS = {
    "benchmarks/requests",
    "benchmarks/responses",
    "benchmarks/snapshots",
    "data/.cache",
    "data/external_benchmark_candidates",
    "data/local_roms",
    "data/public_benchmarks",
    "data/rom_manifests",
    "data/synthetic",
}
SECRET_PATTERNS = (
    ".env",
    ".env.*",
    "*.key",
    "*.local.json",
    "*.p12",
    "*.pem",
    "*.pfx",
    "*.secret.json",
)
LARGE_BINARY_PATTERNS = (
    "*.bin",
    "*.chd",
    "*.ckpt",
    "*.cue",
    "*.gb",
    "*.gba",
    "*.gbc",
    "*.iso",
    "*.n64",
    "*.nds",
    "*.nes",
    "*.onnx",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.smc",
    "*.z64",
)
NORMALIZE_SUFFIXES = {
    ".cmd",
    ".command",
    ".css",
    ".desktop",
    ".html",
    ".js",
    ".json",
    ".md",
    ".plist",
    ".ps1",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}


def main(argv: list[str] | None = None, *, default_out: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a universal Project Theseus Hive installer USB bundle.")
    sub = parser.add_subparsers(dest="command")

    list_cmd = sub.add_parser("list", help="List candidate removable USB targets.")
    list_cmd.add_argument("--json", action="store_true")

    write_cmd = sub.add_parser("write", help="Build a bundle and optionally write it to a USB drive.")
    add_write_args(write_cmd, default_out or DEFAULT_OUT)

    add_write_args(parser, default_out or DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.command == "list":
        report = {"ok": True, "policy": "project_theseus_usb_targets_v0", "created_utc": now(), "targets": list_usb_drives()}
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "write":
        pass
    report = build_bundle(
        out=Path(args.out),
        coordinator_url=args.coordinator_url,
        invite=Path(args.invite) if args.invite else None,
        expires_days=int(args.expires_days or 30),
        include_heavy_data=bool(args.include_heavy_data),
        zip_bundle=not bool(args.no_zip),
        force=bool(args.force),
        hive_mode=str(args.hive_mode),
        new_hive_name=str(args.new_hive_name or ""),
        new_hive_tier=str(args.new_hive_tier),
        activate_new_hive=not bool(args.no_activate_new_hive),
        public_gateway_url=str(args.public_gateway_url or ""),
        public_mode=str(args.public_mode),
        public_worker_name=str(args.public_worker_name or ""),
        target=Path(args.target) if args.target else None,
        confirm_label=str(args.confirm_label or ""),
        yes=bool(args.yes),
        usb_label=str(args.usb_label or ""),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 2


def add_write_args(parser: argparse.ArgumentParser, default_out: Path) -> None:
    parser.add_argument("--out", default=str(default_out), help="Output bundle directory.")
    parser.add_argument("--target", default="", help="USB root to write, for example E:\\ or /Volumes/THESEUS.")
    parser.add_argument("--confirm-label", default="", help="Required USB volume label confirmation unless --yes is used.")
    parser.add_argument("--yes", action="store_true", help="Confirm the selected target without matching the label.")
    parser.add_argument("--usb-label", default="THESEUS_HIVE", help="Best-effort volume label to apply after writing.")
    parser.add_argument("--coordinator-url", default="", help="Coordinator URL, for example http://10.0.0.147:8791.")
    parser.add_argument("--invite", default="", help="Existing invite JSON to embed. Defaults to a fresh invite for the active local Hive.")
    parser.add_argument("--hive-mode", choices=["current", "new", "public"], default="current")
    parser.add_argument("--new-hive-name", default="Project Theseus Hive USB")
    parser.add_argument("--new-hive-tier", choices=["private", "friends_family", "company"], default="private")
    parser.add_argument("--no-activate-new-hive", action="store_true", help="Do not activate the new Hive on this coordinator after creating it.")
    parser.add_argument("--public-gateway-url", default="")
    parser.add_argument("--public-mode", choices=["off", "idle", "always"], default="idle")
    parser.add_argument("--public-worker-name", default="")
    parser.add_argument("--expires-days", type=int, default=30)
    parser.add_argument("--include-heavy-data", action="store_true", help="Include large benchmark caches and external data candidates.")
    parser.add_argument("--no-zip", action="store_true", help="Only create the directory bundle.")
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it already exists.")
    parser.add_argument("--dry-run", action="store_true")


def build_bundle(
    *,
    out: Path = DEFAULT_OUT,
    coordinator_url: str = "",
    invite: Path | None = None,
    expires_days: int = 30,
    include_heavy_data: bool = False,
    zip_bundle: bool = True,
    force: bool = False,
    hive_mode: str = "current",
    new_hive_name: str = "Project Theseus Hive USB",
    new_hive_tier: str = "private",
    activate_new_hive: bool = True,
    public_gateway_url: str = "",
    public_mode: str = "idle",
    public_worker_name: str = "",
    target: Path | None = None,
    confirm_label: str = "",
    yes: bool = False,
    usb_label: str = "THESEUS_HIVE",
    dry_run: bool = False,
) -> dict[str, Any]:
    out = out.resolve()
    usb_preflight: dict[str, Any] = {}
    if target:
        target = normalize_target_root(target)
        usb_preflight = validate_usb_target(target, confirm_label=confirm_label, yes=yes)
        if not usb_preflight.get("ok"):
            return usb_preflight
    if out.exists():
        if not force:
            return {"ok": False, "error": "output_exists", "bundle_dir": str(out), "next_action": "Re-run with --force or choose a different --out."}
        if not safe_remove_target(out):
            return {"ok": False, "error": "unsafe_output_path", "bundle_dir": str(out)}
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    coordinator_url = coordinator_url or detect_coordinator_url()
    credential_plan = resolve_credential_plan(
        hive_mode=hive_mode,
        coordinator_url=coordinator_url,
        invite=invite,
        expires_days=expires_days,
        new_hive_name=new_hive_name,
        new_hive_tier=new_hive_tier,
        activate_new_hive=activate_new_hive,
        public_gateway_url=public_gateway_url,
        public_mode=public_mode,
        public_worker_name=public_worker_name,
    )
    if not credential_plan.get("ok"):
        return credential_plan
    payload_dir = out / "payload"
    invite_dir = out / "invite"
    invite_dir.mkdir(parents=True, exist_ok=True)
    invite_path = invite_dir / "theseus-hive-invite.json"

    copy_report = copy_payload(payload_dir, include_heavy_data=include_heavy_data)
    if credential_plan.get("invite_path"):
        shutil.copy2(Path(str(credential_plan["invite_path"])), invite_path)
    else:
        write_json(invite_path, credential_plan["invite"])

    write_install_config(out, credential_plan, coordinator_url)
    write_launchers(out, credential_plan, coordinator_url)
    manifest = write_bundle_manifest(out, payload_dir, invite_path, coordinator_url, copy_report, credential_plan)
    zip_path = ""
    if zip_bundle:
        zip_path = str(create_zip(out))
    usb_report: dict[str, Any] = {}
    if target:
        usb_report = write_to_usb(
            bundle=out,
            target=target,
            confirm_label=confirm_label,
            yes=yes,
            usb_label=usb_label,
            dry_run=dry_run,
            preflight=usb_preflight,
        )
        if not usb_report.get("ok"):
            return {**usb_report, "bundle_dir": str(out), "zip_path": zip_path}

    report = {
        "ok": True,
        "policy": "project_theseus_universal_hive_usb_writer_v0",
        "created_utc": now(),
        "bundle_dir": str(out),
        "zip_path": zip_path,
        "coordinator_url": coordinator_url,
        "invite_path": str(invite_path),
        "hive_mode": credential_plan.get("hive_mode"),
        "credential_source": credential_plan.get("credential_source"),
        "payload_files": manifest["payload_file_count"],
        "payload_bytes": manifest["payload_bytes"],
        "skipped_paths": copy_report["skipped_count"],
        "launchers": {
            "windows": "Install Project Theseus Hive.cmd",
            "macos": "Install Project Theseus Hive.command",
            "linux": "sh install-from-usb.sh",
        },
        "usb_write": usb_report,
        "security_note": "The invite in this bundle contains a Hive join token. Treat the bundle like a password.",
    }
    write_json(out / "writer-report.json", report)
    return report


def resolve_credential_plan(
    *,
    hive_mode: str,
    coordinator_url: str,
    invite: Path | None,
    expires_days: int,
    new_hive_name: str,
    new_hive_tier: str,
    activate_new_hive: bool,
    public_gateway_url: str,
    public_mode: str,
    public_worker_name: str,
) -> dict[str, Any]:
    hive_mode = "public" if hive_mode == "public_worker" else hive_mode
    if hive_mode == "public":
        public_invite = {
            "policy": "project_theseus_public_worker_usb_invite_v0",
            "created_utc": now(),
            "hive_id": "theseus-public",
            "hive_name": "Project Theseus Public Worker",
            "tier": "public",
            "relay_url": public_gateway_url,
            "coordinator_url": coordinator_url,
            "join_token": "",
            "public_contribution": {
                "mode": public_mode,
                "gateway_url": public_gateway_url,
                "worker_name": public_worker_name,
                "explicit_opt_in": public_mode != "off",
            },
        }
        return {
            "ok": True,
            "hive_mode": "public",
            "credential_source": "public_worker_no_private_token",
            "invite": public_invite,
            "public_mode": public_mode,
            "public_gateway_url": public_gateway_url,
            "public_worker_name": public_worker_name,
            "private_invite_enabled": False,
        }
    if invite:
        loaded = read_json(invite, {})
        if not isinstance(loaded, dict) or not loaded.get("hive_id"):
            return {"ok": False, "error": "invalid_invite", "invite": str(invite)}
        return {
            "ok": True,
            "hive_mode": "current",
            "credential_source": "explicit_invite",
            "invite_path": str(invite),
            "invite": loaded,
            "private_invite_enabled": True,
            "public_mode": "off",
            "public_gateway_url": "",
            "public_worker_name": "",
        }
    if hive_mode == "new":
        profile_report = hive_profiles.create_profile(
            name=new_hive_name or "Project Theseus Hive USB",
            tier=new_hive_tier,
            hive_id="",
            relay_url="",
            join_token="",
            mode="lan",
            activate=activate_new_hive,
        )
        if not profile_report.get("ok"):
            return profile_report
        private_profile = {}
        profiles = hive_profiles.load_profiles_private().get("profiles") or []
        new_public = profile_report.get("profile") if isinstance(profile_report.get("profile"), dict) else {}
        for profile in profiles:
            if isinstance(profile, dict) and profile.get("profile_id") == new_public.get("profile_id"):
                private_profile = profile
                break
        if not private_profile:
            return {"ok": False, "error": "new_hive_profile_not_found"}
        invite_payload = create_invite_payload(
            coordinator_url=coordinator_url,
            expires_days=expires_days,
            hive_id=str(private_profile.get("hive_id") or ""),
            join_token=str(private_profile.get("join_token") or ""),
            name=str(private_profile.get("name") or new_hive_name),
            tier=str(private_profile.get("tier") or "private"),
        )
        return {
            "ok": True,
            "hive_mode": "new",
            "credential_source": "new_local_hive_profile",
            "new_hive_activated": bool(activate_new_hive),
            "profile": hive_profiles.public_profile(private_profile),
            "invite": invite_payload,
            "private_invite_enabled": True,
            "public_mode": "off",
            "public_gateway_url": "",
            "public_worker_name": "",
        }
    invite_payload = create_invite_payload(coordinator_url=coordinator_url, expires_days=expires_days)
    return {
        "ok": True,
        "hive_mode": "current",
        "credential_source": "active_hive",
        "invite": invite_payload,
        "private_invite_enabled": True,
        "public_mode": "off",
        "public_gateway_url": "",
        "public_worker_name": "",
    }


def copy_payload(dst: Path, *, include_heavy_data: bool) -> dict[str, Any]:
    stats: dict[str, Any] = {"copied_count": 0, "copied_bytes": 0, "skipped_count": 0, "skipped": []}

    def skip(path: Path, rel: str) -> str:
        name = path.name
        if name in EXCLUDE_DIR_NAMES:
            return "generated_or_local_dir"
        if any(name.startswith(prefix) for prefix in EXCLUDE_DIR_PREFIXES):
            return "local_virtualenv"
        rel_posix = rel.replace("\\", "/")
        if not include_heavy_data and any(rel_posix == item or rel_posix.startswith(item + "/") for item in HEAVY_DATA_DIRS):
            return "heavy_data_cache"
        if path.is_file() and any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_PATTERNS):
            return "local_secret"
        if path.is_file() and any(fnmatch.fnmatch(name.lower(), pattern) for pattern in LARGE_BINARY_PATTERNS):
            return "large_binary"
        if is_reparse_point(path):
            return "reparse_point"
        return ""

    def record_skip(rel: str, reason: str) -> None:
        stats["skipped_count"] += 1
        if len(stats["skipped"]) < 500:
            stats["skipped"].append({"path": rel, "reason": reason})

    def walk(src_dir: Path, dst_dir: Path, rel_dir: str = "") -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for child in sorted(src_dir.iterdir(), key=lambda p: p.name.lower()):
            rel = f"{rel_dir}/{child.name}" if rel_dir else child.name
            reason = skip(child, rel)
            if reason:
                record_skip(rel, reason)
                continue
            target = dst_dir / child.name
            if child.is_dir():
                walk(child, target, rel)
            elif child.is_file():
                copy_file(child, target)
                stats["copied_count"] += 1
                stats["copied_bytes"] += target.stat().st_size

    walk(ROOT, dst)
    return stats


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in NORMALIZE_SUFFIXES:
        data = src.read_bytes()
        if b"\0" not in data:
            dst.write_bytes(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        else:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)
    if src.suffix.lower() in {".sh", ".command"}:
        make_executable(dst)


def create_invite_payload(
    *,
    coordinator_url: str,
    expires_days: int,
    hive_id: str = "",
    join_token: str = "",
    name: str = "",
    tier: str = "private",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-invite-") as tmp:
        invite_path = Path(tmp) / "invite.json"
        create_invite_file(
            invite_path,
            coordinator_url=coordinator_url,
            expires_days=expires_days,
            hive_id=hive_id,
            join_token=join_token,
            name=name,
            tier=tier,
        )
        return read_json(invite_path, {})


def create_invite_file(
    invite_path: Path,
    *,
    coordinator_url: str,
    expires_days: int,
    hive_id: str = "",
    join_token: str = "",
    name: str = "",
    tier: str = "private",
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "hive_invite.py"),
        "create",
        "--tier",
        tier,
        "--coordinator-url",
        coordinator_url,
        "--expires-days",
        str(max(1, expires_days)),
        "--out",
        str(invite_path),
    ]
    if hive_id:
        cmd.extend(["--hive-id", hive_id])
    if join_token:
        cmd.extend(["--join-token", join_token])
    if name:
        cmd.extend(["--name", name])
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to create Hive invite. Make sure this install is registered and has an active Hive profile.\n"
            + completed.stdout
            + completed.stderr
        )


def write_install_config(bundle: Path, plan: dict[str, Any], coordinator_url: str) -> None:
    config = {
        "policy": "project_theseus_usb_install_config_v0",
        "created_utc": now(),
        "hive_mode": plan.get("hive_mode"),
        "credential_source": plan.get("credential_source"),
        "coordinator_url": coordinator_url,
        "private_invite_enabled": bool(plan.get("private_invite_enabled")),
        "public_contribution": {
            "mode": plan.get("public_mode") or "off",
            "gateway_url": plan.get("public_gateway_url") or "",
            "worker_name": plan.get("public_worker_name") or "",
            "allow": (plan.get("public_mode") or "off") != "off",
        },
        "autorun_note": "Modern Windows, macOS, and Linux intentionally block silent USB autorun. This key provides root launchers for one-click install/start.",
    }
    write_json(bundle / "theseus-usb-installer.json", config)


def write_launchers(bundle: Path, plan: dict[str, Any], coordinator_url: str) -> None:
    write_posix_launchers(bundle, plan, coordinator_url)
    write_windows_launchers(bundle, plan, coordinator_url)
    write_autorun(bundle)
    write_readme(bundle, plan, coordinator_url)


def write_posix_launchers(bundle: Path, plan: dict[str, Any], coordinator_url: str) -> None:
    install_sh = bundle / "install-from-usb.sh"
    command_file = bundle / "Install Project Theseus Hive.command"
    app_exec = bundle / "Install Project Theseus Hive.app" / "Contents" / "MacOS" / "Install Project Theseus Hive"
    app_plist = bundle / "Install Project Theseus Hive.app" / "Contents" / "Info.plist"
    hive_mode = str(plan.get("hive_mode") or "current")
    public_mode = sh_single_quote(str(plan.get("public_mode") or "off"))
    public_gateway = sh_single_quote(str(plan.get("public_gateway_url") or ""))
    public_worker = sh_single_quote(str(plan.get("public_worker_name") or ""))
    private_invite = "1" if plan.get("private_invite_enabled") else "0"

    write_lf(
        install_sh,
        f"""#!/usr/bin/env sh
set -eu
BUNDLE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
INVITE="$BUNDLE_ROOT/invite/theseus-hive-invite.json"
PAYLOAD="$BUNDLE_ROOT/payload"
COORDINATOR_URL="{coordinator_url}"
HIVE_MODE="{hive_mode}"
PRIVATE_INVITE="{private_invite}"
PUBLIC_MODE={public_mode}
PUBLIC_GATEWAY_URL={public_gateway}
PUBLIC_WORKER_NAME={public_worker}
case "$(uname -s)" in
  Darwin)
    if [ "$HIVE_MODE" = "public" ]; then
      exec "$PAYLOAD/scripts/install_theseus_hive_macos.sh" --source "$PAYLOAD" --coordinator-url "$COORDINATOR_URL" --public-mode "$PUBLIC_MODE" --public-gateway-url "$PUBLIC_GATEWAY_URL" --public-worker-name "$PUBLIC_WORKER_NAME" --allow-public --auto-update-soft --install-service --enable-service --start "$@"
    else
      exec "$PAYLOAD/scripts/install_theseus_hive_macos.sh" --source "$PAYLOAD" --invite "$INVITE" --coordinator-url "$COORDINATOR_URL" --auto-update-soft --install-service --enable-service --start "$@"
    fi
    ;;
  Linux)
    if [ "$HIVE_MODE" = "public" ]; then
      exec "$PAYLOAD/scripts/install_theseus_hive_linux.sh" --source "$PAYLOAD" --coordinator-url "$COORDINATOR_URL" --public-mode "$PUBLIC_MODE" --public-gateway-url "$PUBLIC_GATEWAY_URL" --public-worker-name "$PUBLIC_WORKER_NAME" --allow-public --auto-update-soft --install-service --enable-service --start "$@"
    else
      exec "$PAYLOAD/scripts/install_theseus_hive_linux.sh" --source "$PAYLOAD" --invite "$INVITE" --coordinator-url "$COORDINATOR_URL" --auto-update-soft --install-service --enable-service --start "$@"
    fi
    ;;
  *)
    printf '%s\\n' "Unsupported OS for this launcher. On Windows, double-click Install Project Theseus Hive.cmd."
    exit 2
    ;;
esac
""",
    )
    make_executable(install_sh)
    write_lf(
        command_file,
        """#!/usr/bin/env sh
DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$DIR/install-from-usb.sh" "$@"
""",
    )
    make_executable(command_file)
    write_lf(
        app_exec,
        """#!/usr/bin/env sh
APP_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
exec "$APP_ROOT/install-from-usb.sh" "$@"
""",
    )
    make_executable(app_exec)
    write_lf(
        app_plist,
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>Install Project Theseus Hive</string>
  <key>CFBundleIdentifier</key>
  <string>local.project-theseus.usb-installer</string>
  <key>CFBundleName</key>
  <string>Install Project Theseus Hive</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
</dict>
</plist>
""",
    )
    desktop = bundle / "Install Project Theseus Hive.desktop"
    write_lf(
        desktop,
        """[Desktop Entry]
Type=Application
Name=Install Project Theseus Hive
Comment=Install and join Project Theseus Hive
Exec=sh install-from-usb.sh
Terminal=true
Categories=Utility;Development;
""",
    )


def write_windows_launchers(bundle: Path, plan: dict[str, Any], coordinator_url: str) -> None:
    cmd = bundle / "Install Project Theseus Hive.cmd"
    ps1 = bundle / "Install Project Theseus Hive.ps1"
    write_lf(
        cmd,
        """@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install Project Theseus Hive.ps1" %*
if errorlevel 1 pause
""",
    )
    template = WINDOWS_INSTALLER_TEMPLATE
    replacements = {
        "__COORDINATOR_URL__": ps_single_quote(coordinator_url),
        "__HIVE_MODE__": ps_single_quote(str(plan.get("hive_mode") or "current")),
        "__PRIVATE_INVITE__": "$true" if plan.get("private_invite_enabled") else "$false",
        "__PUBLIC_MODE__": ps_single_quote(str(plan.get("public_mode") or "off")),
        "__PUBLIC_GATEWAY_URL__": ps_single_quote(str(plan.get("public_gateway_url") or "")),
        "__PUBLIC_WORKER_NAME__": ps_single_quote(str(plan.get("public_worker_name") or "")),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    write_lf(ps1, template)


def write_autorun(bundle: Path) -> None:
    write_lf(
        bundle / "autorun.inf",
        """[AutoRun]
label=Project Theseus Hive
action=Install Project Theseus Hive
open=Install Project Theseus Hive.cmd
shell\\install=&Install Project Theseus Hive
shell\\install\\command=Install Project Theseus Hive.cmd
""",
    )


WINDOWS_INSTALLER_TEMPLATE = r'''param(
  [string]$InstallRoot = "",
  [string]$RuntimeRoot = "",
  [switch]$NoStart,
  [switch]$NoTray
)

$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Payload = Join-Path $BundleRoot "payload"
$Invite = Join-Path $BundleRoot "invite\theseus-hive-invite.json"
$CoordinatorUrl = __COORDINATOR_URL__
$HiveMode = __HIVE_MODE__
$PrivateInvite = __PRIVATE_INVITE__
$PublicMode = __PUBLIC_MODE__
$PublicGatewayUrl = __PUBLIC_GATEWAY_URL__
$PublicWorkerName = __PUBLIC_WORKER_NAME__

if (-not (Test-Path $Payload)) {
  throw "Missing payload folder: $Payload"
}
if ($PrivateInvite -and -not (Test-Path $Invite)) {
  throw "Missing Hive invite: $Invite"
}
if (-not $InstallRoot) {
  $InstallRoot = Join-Path $env:LOCALAPPDATA "Project Theseus Hive\app\current"
}
if ($InstallRoot.Length -lt 12) {
  throw "Refusing unsafe install root: $InstallRoot"
}
$parent = Split-Path -Parent $InstallRoot
New-Item -ItemType Directory -Force -Path $parent | Out-Null
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null

$robocopyArgs = @(
  $Payload,
  $InstallRoot,
  "/MIR",
  "/XD", ".git", ".attd_tmp", ".venv-puffer", "target", "reports", "checkpoints", "dist", "games", "node_modules", "tmp", "updates", "vendor", "__pycache__",
  "/XF", "*.pyc", "*.pyo"
)
& robocopy @robocopyArgs | Out-Host
if ($LASTEXITCODE -ge 8) {
  throw "robocopy failed with exit code $LASTEXITCODE"
}

Set-Location $InstallRoot
$installArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", "scripts\install_theseus_hive.ps1",
  "-CoordinatorUrl", $CoordinatorUrl,
  "-CreateDesktopShortcut",
  "-InstallTray",
  "-AutoUpdateSoft",
  "-InstallScheduledTask"
)
if ($NoTray) {
  $installArgs = $installArgs | Where-Object { $_ -ne "-InstallTray" }
}
if ($PrivateInvite) { $installArgs += @("-Invite", $Invite) }
if ($HiveMode -eq "public") {
  $installArgs += @("-PublicMode", $PublicMode, "-PublicGatewayUrl", $PublicGatewayUrl, "-PublicWorkerName", $PublicWorkerName, "-AllowPublic")
}
if ($RuntimeRoot) { $installArgs += @("-RuntimeRoot", $RuntimeRoot) }
if (-not $NoStart) { $installArgs += "-StartNow" }
if ((-not $NoStart) -and (-not $NoTray)) { $installArgs += "-StartTray" }
& powershell @installArgs

Write-Host ""
Write-Host "Project Theseus payload installed at: $InstallRoot"
Write-Host "Project Theseus coordinator: $CoordinatorUrl"
Write-Host "Project Theseus USB mode: $HiveMode"
if (-not $NoTray) { Write-Host "Project Theseus Hive tray installed. Use the tray icon near the clock to open dashboard/chat." }
'''


def write_readme(bundle: Path, plan: dict[str, Any], coordinator_url: str) -> None:
    mode = str(plan.get("hive_mode") or "current")
    credential = str(plan.get("credential_source") or "")
    write_lf(
        bundle / "README-USB.txt",
        f"""Project Theseus Universal Hive USB Installer

Coordinator: {coordinator_url}
Mode: {mode}
Credential source: {credential}

Windows:
  Double-click Install Project Theseus Hive.cmd

macOS:
  Unzip this bundle on the Mac, then double-click Install Project Theseus Hive.command
  CLI fallback: sh install-from-usb.sh

Linux:
  Run: sh install-from-usb.sh

If macOS blocks the unsigned private installer, use Terminal from this folder:
  xattr -dr com.apple.quarantine .
  sh install-from-usb.sh

What the installer does:
- Copies the payload to local app storage before launching services
- Stores mutable runtime data outside the USB drive
- Applies the embedded Hive invite
- Starts the local Hive node and points it at the coordinator
- Creates OS-appropriate app/desktop shortcuts when supported

Security note:
Private Hive bundles include invite/theseus-hive-invite.json with a Hive join token.
Treat this USB/zip like a password for your private Hive. Public worker bundles do
not carry a private Hive token.

Autorun note:
Modern Windows, macOS, and Linux intentionally block silent USB autorun. This key
puts the correct installer launchers at the USB root so install/start is one
click or one command on each OS.
""",
    )
    # Keep the old Mac filename for users who already learned it.
    shutil.copy2(bundle / "README-USB.txt", bundle / "README-MAC.txt")


def write_bundle_manifest(
    bundle: Path,
    payload: Path,
    invite: Path,
    coordinator_url: str,
    copy_report: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(payload.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(payload).as_posix()
        size = path.stat().st_size
        total += size
        files.append({"path": rel, "size": size, "sha256": sha256_file(path)})
    payload_manifest = {
        "policy": "project_theseus_payload_manifest_v0",
        "created_utc": now(),
        "file_count": len(files),
        "payload_bytes": total,
        "files": files,
    }
    write_json(bundle / "payload-manifest.json", payload_manifest)
    bundle_manifest = {
        "policy": "project_theseus_universal_hive_usb_manifest_v0",
        "created_utc": now(),
        "coordinator_url": coordinator_url,
        "hive_mode": plan.get("hive_mode"),
        "credential_source": plan.get("credential_source"),
        "private_invite_enabled": bool(plan.get("private_invite_enabled")),
        "invite_path": invite.relative_to(bundle).as_posix(),
        "payload_file_count": len(files),
        "payload_bytes": total,
        "copy_report": copy_report,
    }
    write_json(bundle / "bundle-manifest.json", bundle_manifest)
    return bundle_manifest


def list_usb_drives() -> list[dict[str, Any]]:
    if os.name == "nt":
        return list_windows_usb_drives()
    if sys.platform == "darwin":
        return list_posix_mounts(["/Volumes"])
    user = os.environ.get("USER") or ""
    bases = ["/media", "/mnt"]
    if user:
        bases.extend([f"/media/{user}", f"/run/media/{user}"])
    return list_posix_mounts(bases)


def list_windows_usb_drives() -> list[dict[str, Any]]:
    ps = r"""
$rows = Get-CimInstance Win32_LogicalDisk |
  Where-Object { $_.DriveType -eq 2 } |
  Select-Object DeviceID,VolumeName,FileSystem,Size,FreeSpace
$rows | ConvertTo-Json -Compress
"""
    result = subprocess.run(["powershell", "-NoProfile", "-Command", ps], text=True, capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    raw = read_json_text(result.stdout.strip(), [])
    rows = raw if isinstance(raw, list) else [raw]
    targets: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        root = str(row.get("DeviceID") or "")
        if root and not root.endswith("\\"):
            root += "\\"
        size = int(row.get("Size") or 0)
        free = int(row.get("FreeSpace") or 0)
        targets.append(
            {
                "path": root,
                "label": str(row.get("VolumeName") or ""),
                "filesystem": str(row.get("FileSystem") or ""),
                "size_bytes": size,
                "free_bytes": free,
                "size_gib": round(size / (1024**3), 2) if size else 0,
                "free_gib": round(free / (1024**3), 2) if free else 0,
                "removable": True,
            }
        )
    return targets


def list_posix_mounts(bases: list[str]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for base in bases:
        root = Path(base).expanduser()
        if not root.exists() or not root.is_dir():
            continue
        candidates = list(root.iterdir()) if root.name not in {"mnt", "media"} else [root, *list(root.iterdir())]
        for item in candidates:
            if not item.exists() or not item.is_dir():
                continue
            try:
                resolved = str(item.resolve())
                usage = shutil.disk_usage(item)
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            targets.append(
                {
                    "path": str(item),
                    "label": item.name,
                    "filesystem": "",
                    "size_bytes": usage.total,
                    "free_bytes": usage.free,
                    "size_gib": round(usage.total / (1024**3), 2),
                    "free_gib": round(usage.free / (1024**3), 2),
                    "removable": True,
                }
            )
    return targets


def write_to_usb(
    *,
    bundle: Path,
    target: Path,
    confirm_label: str,
    yes: bool,
    usb_label: str,
    dry_run: bool,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = normalize_target_root(target)
    validation = preflight or validate_usb_target(target, confirm_label=confirm_label, yes=yes)
    if not validation.get("ok"):
        return validation
    paths = theseus_usb_paths(target)
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "target": str(target),
            "target_label": validation.get("label"),
            "would_replace": [str(path) for path in paths if path.exists()],
        }
    for path in paths:
        remove_within(path, target)
    copy_tree_contents(bundle, target)
    if usb_label:
        set_volume_label(target, usb_label)
    write_json(
        target / "theseus-usb-write-report.json",
        {
            "ok": True,
            "policy": "project_theseus_usb_write_report_v0",
            "created_utc": now(),
            "target": str(target),
            "previous_label": validation.get("label"),
            "requested_label": usb_label,
            "source_bundle": str(bundle),
        },
    )
    return {
        "ok": True,
        "target": str(target),
        "target_label": validation.get("label"),
        "requested_label": usb_label,
        "launchers": {
            "windows": str(target / "Install Project Theseus Hive.cmd"),
            "macos": str(target / "Install Project Theseus Hive.command"),
            "linux": str(target / "install-from-usb.sh"),
        },
    }


def validate_usb_target(target: Path, *, confirm_label: str, yes: bool) -> dict[str, Any]:
    if not target.exists() or not target.is_dir():
        return {"ok": False, "error": "target_not_found", "target": str(target)}
    if is_dangerous_target(target):
        return {"ok": False, "error": "unsafe_target", "target": str(target)}
    info = usb_info_for_target(target)
    if not info.get("removable"):
        return {
            "ok": False,
            "error": "target_not_listed_as_removable",
            "target": str(target),
            "next_action": "Use theseus usb list and select a removable target.",
        }
    label = str(info.get("label") or "")
    if not yes and not confirm_label:
        return {
            "ok": False,
            "error": "confirmation_required",
            "target": str(target),
            "label": label,
            "next_action": "Re-run with --confirm-label matching the USB label, or --yes.",
        }
    if confirm_label and label.lower() != confirm_label.lower():
        return {"ok": False, "error": "label_mismatch", "target": str(target), "label": label, "confirm_label": confirm_label}
    return {"ok": True, **info}


def usb_info_for_target(target: Path) -> dict[str, Any]:
    target_resolved = str(target.resolve()).rstrip("\\/")
    for row in list_usb_drives():
        row_path = str(row.get("path") or "").rstrip("\\/")
        try:
            if row_path and str(Path(row_path).resolve()).rstrip("\\/").lower() == target_resolved.lower():
                return row
        except OSError:
            if row_path.lower() == target_resolved.lower():
                return row
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        usage = None
    return {
        "path": str(target),
        "label": target.name,
        "filesystem": "",
        "size_bytes": usage.total if usage else 0,
        "free_bytes": usage.free if usage else 0,
        "removable": False,
    }


def normalize_target_root(target: Path) -> Path:
    text = str(target)
    if os.name == "nt" and len(text) == 2 and text[1] == ":":
        return Path(text + "\\")
    return target.resolve()


def is_dangerous_target(target: Path) -> bool:
    try:
        resolved = target.resolve()
        home = Path.home().resolve()
        return resolved in {Path(resolved.anchor), home, ROOT.resolve(), (ROOT / "dist").resolve()}
    except OSError:
        return True


def theseus_usb_paths(target: Path) -> list[Path]:
    names = [
        "payload",
        "invite",
        "Install Project Theseus Hive.app",
        "Install Project Theseus Hive.cmd",
        "Install Project Theseus Hive.command",
        "Install Project Theseus Hive.desktop",
        "Install Project Theseus Hive.ps1",
        "README-MAC.txt",
        "README-USB.txt",
        "autorun.inf",
        "bundle-manifest.json",
        "payload-manifest.json",
        "theseus-usb-installer.json",
        "writer-report.json",
        "theseus-usb-write-report.json",
        "install-from-usb.sh",
    ]
    return [target / name for name in names]


def remove_within(path: Path, root: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved == root_resolved or root_resolved not in resolved.parents:
        raise RuntimeError(f"Refusing to remove unsafe path: {path}")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_tree_contents(src: Path, dst: Path) -> None:
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def set_volume_label(target: Path, label: str) -> None:
    if not label:
        return
    if os.name == "nt":
        drive = str(target.drive or str(target)[:2]).rstrip(":")
        if drive:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", f"Set-Volume -DriveLetter {drive} -NewFileSystemLabel {ps_single_quote(label)}"],
                text=True,
                capture_output=True,
                check=False,
            )
    elif sys.platform == "darwin":
        subprocess.run(["diskutil", "rename", str(target), label], text=True, capture_output=True, check=False)


def create_zip(bundle: Path) -> Path:
    zip_path = bundle.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    base = bundle.parent
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(bundle.rglob("*")):
            rel = path.relative_to(base).as_posix()
            if path.is_dir():
                info = zipfile.ZipInfo(rel.rstrip("/") + "/")
                info.external_attr = (0o755 << 16) | 0x10
                zf.writestr(info, b"")
                continue
            info = zipfile.ZipInfo(rel)
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if should_zip_executable(path) else 0o644
            info.external_attr = mode << 16
            info.date_time = datetime.now().timetuple()[:6]
            with path.open("rb") as handle:
                zf.writestr(info, handle.read())
    return zip_path


def detect_coordinator_url() -> str:
    for rel in ("reports/hive_status.json", "reports/hive_status_probe_for_review.json"):
        data = read_json(ROOT / rel, {})
        if isinstance(data, dict):
            url = str(data.get("api_url") or data.get("node", {}).get("api_url") or "")
            if url:
                return url
    return f"http://{local_ip_guess()}:8791"


def local_ip_guess() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


def safe_remove_target(path: Path) -> bool:
    try:
        resolved = path.resolve()
        allowed = (ROOT / "dist").resolve()
        return resolved == allowed or allowed in resolved.parents
    except OSError:
        return False


def should_zip_executable(path: Path) -> bool:
    if path.suffix.lower() in {".sh", ".command"}:
        return True
    parts = path.parts
    return len(parts) >= 2 and parts[-2] == "MacOS"


def make_executable(path: Path) -> None:
    try:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_json_text(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def is_reparse_point(path: Path) -> bool:
    if os.name != "nt":
        return path.is_symlink()
    try:
        return bool(path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return path.is_symlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ps_single_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sh_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
