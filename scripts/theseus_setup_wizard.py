"""Click-friendly Project Theseus Hive setup wizard.

This is the no-terminal onboarding surface. It serves a local-only web wizard
that creates hives, joins hives from invite files or pasted JSON, switches
profiles, starts the local node/relay, and generates phone join QR codes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import hive_profiles
import hive_remote_access
import hive_usb_writer
import license_manager
import public_hive_contributor
import theseus_runtime
import update_manager
from theseus_qr import qr_svg


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
POLICY_PATH = CONFIGS / "hive_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler())
    url = f"http://{args.host}:{args.port}"
    print(f"Project Theseus Setup Wizard: {url}")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def make_handler() -> type[BaseHTTPRequestHandler]:
    class WizardHandler(BaseHTTPRequestHandler):
        server_version = "ProjectTheseusSetupWizard/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                return self.html(INDEX_HTML)
            if parsed.path == "/api/status":
                return self.json(status_payload())
            if parsed.path == "/api/usb/list":
                return self.json({"ok": True, "policy": "project_theseus_usb_targets_v0", "created_utc": now(), "targets": hive_usb_writer.list_usb_drives()})
            if parsed.path == "/api/qr":
                query = parse_qs(parsed.query)
                text = first(query, "text")
                try:
                    svg = qr_svg(text)
                except Exception as exc:
                    return self.json({"ok": False, "error": str(exc)}, status=400)
                data = svg.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            body = self.read_json()
            if parsed.path == "/api/create-hive":
                return self.json(create_hive(body))
            if parsed.path == "/api/join-hive":
                return self.json(join_hive(body))
            if parsed.path == "/api/switch-hive":
                return self.json(switch_hive(body))
            if parsed.path == "/api/start-services":
                return self.json(start_services(body))
            if parsed.path == "/api/upgrade-relay":
                return self.json(upgrade_active_hive_to_relay(body))
            if parsed.path == "/api/remote/configure-relay":
                return self.json(configure_remote_relay(body))
            if parsed.path == "/api/remote/wireguard-guide":
                return self.json(write_remote_wireguard_guide(body))
            if parsed.path == "/api/usb/write":
                return self.json(write_usb_bundle(body))
            if parsed.path == "/api/public-contribution":
                return self.json(configure_public_contribution(body))
            if parsed.path == "/api/public-contribution/poll-once":
                return self.json(public_hive_contributor.poll_once(read_json(POLICY_PATH, {})))
            if parsed.path == "/api/update/configure":
                return self.json(configure_updates(body))
            if parsed.path == "/api/update/check":
                return self.json(check_updates(body))
            if parsed.path == "/api/update/apply":
                return self.json(apply_update(body))
            if parsed.path == "/api/license/register":
                return self.json(register_license(body))
            if parsed.path == "/api/license/import":
                return self.json(import_license(body))
            if parsed.path == "/api/license/request":
                return self.json(license_request(body))
            return self.send_error(HTTPStatus.NOT_FOUND)

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}

        def html(self, content: str) -> None:
            data = content.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def json(self, payload: Any, status: int = 200) -> None:
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return WizardHandler


def status_payload() -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    profiles = hive_profiles.load_profiles()
    active_private = hive_profiles.active_profile()
    active_public = hive_profiles.public_profile(active_private) if active_private else {}
    return {
        "ok": True,
        "policy": "project_theseus_setup_wizard_status_v0",
        "created_utc": now(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.executable,
        },
        "local_ips": local_ips(),
        "hive_policy": {
            "tiers": policy.get("federation", {}).get("tiers", {}),
            "default_relay_port": policy.get("relay", {}).get("server_port", 8793),
            "public_status": policy.get("federation", {}).get("tiers", {}).get("public", {}).get("status", "disabled"),
        },
        "profiles": profiles,
        "active_profile": active_public,
        "active_invite": invite_from_profile(active_private, include_token=True) if active_private else {},
        "license": license_manager.status_report(write_report=True),
        "reports": {
            "hive_status": read_json(REPORTS / "hive_status.json", {}),
            "hive_peers": read_json(REPORTS / "hive_peers.json", {}),
            "hive_scheduler": read_json(REPORTS / "hive_scheduler.json", {}),
            "hive_relay": read_json(REPORTS / "hive_relay_status.json", {}),
        },
        "public_contribution": public_hive_contributor.status_report(policy),
        "updates": update_manager.status_report(write_report=True),
        "remote_access": hive_remote_access.status_report(policy=policy, write_report=True),
        "runtime_paths": theseus_runtime.runtime_report(create=False, write_report=True),
        "usb_writer": {
            "default_out": str((ROOT / "dist" / "universal-usb" / "ProjectTheseusUniversalUSB").relative_to(ROOT)).replace("\\", "/"),
            "default_coordinator_url": default_coordinator_url(),
            "targets": hive_usb_writer.list_usb_drives(),
        },
    }


def create_hive(body: dict[str, Any]) -> dict[str, Any]:
    name = str(body.get("name") or "My Project Theseus Hive").strip()
    tier = str(body.get("tier") or "private")
    mode = str(body.get("mode") or "lan")
    if tier == "public":
        return public_disabled()
    relay_url = ""
    if mode == "relay":
        relay_url = str(body.get("relay_url") or default_relay_url()).strip()
    report = hive_profiles.create_profile(
        name=name,
        tier=tier,
        hive_id=str(body.get("hive_id") or ""),
        relay_url=relay_url,
        join_token=str(body.get("join_token") or ""),
        mode=mode,
        activate=True,
    )
    if not report.get("ok"):
        return report
    active = hive_profiles.active_profile()
    invite = invite_from_profile(active, include_token=True)
    invite_path = REPORTS / f"hive_invite_{active.get('profile_id', 'active')}.json"
    write_json(invite_path, invite)
    if body.get("start_services", True):
        start_services({"start_relay": mode == "relay", "restart": body.get("restart_services", True)})
    return {
        "ok": True,
        "profile": hive_profiles.public_profile(active),
        "invite": invite,
        "invite_path": rel(invite_path),
        "phone_url": phone_join_url(active),
        "qr_url": "/api/qr?text=" + quote(phone_join_url(active), safe=""),
        "message": "Hive created. Add phones with the QR/link or add computers with the invite file.",
    }


def join_hive(body: dict[str, Any]) -> dict[str, Any]:
    invite = body.get("invite") if isinstance(body.get("invite"), dict) else {}
    invite_text = str(body.get("invite_text") or "").strip()
    if invite_text:
        try:
            invite = json.loads(invite_text)
        except json.JSONDecodeError:
            return {"ok": False, "error": "invite_json_invalid"}
    if not invite:
        invite = {
            "hive_id": str(body.get("hive_id") or ""),
            "hive_name": str(body.get("name") or body.get("hive_id") or "Joined Hive"),
            "tier": str(body.get("tier") or "private"),
            "relay_url": str(body.get("relay_url") or ""),
            "coordinator_url": str(body.get("coordinator_url") or ""),
            "join_token": str(body.get("join_token") or ""),
        }
    report = hive_profiles.add_invite_profile(invite, name=str(body.get("name") or ""), activate=True)
    if not report.get("ok"):
        return report
    if body.get("start_services", True):
        start_services({"restart": body.get("restart_services", True)})
    return {
        "ok": True,
        "profile": report.get("profile"),
        "message": "Hive joined and selected.",
    }


def switch_hive(body: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(body.get("profile_id") or "")
    report = hive_profiles.switch_profile(profile_id)
    if report.get("ok") and body.get("restart_services", False):
        service_report = start_services({"restart": True})
        report["service_report"] = service_report
    return report


def upgrade_active_hive_to_relay(body: dict[str, Any]) -> dict[str, Any]:
    active = hive_profiles.active_profile()
    if not active:
        return {"ok": False, "error": "no_active_hive"}
    if active.get("tier") == "public":
        return public_disabled()
    profiles = hive_profiles.load_profiles_private()
    relay_url = str(body.get("relay_url") or default_relay_url()).strip()
    for profile in profiles.get("profiles") or []:
        if profile.get("profile_id") == active.get("profile_id"):
            profile["relay_url"] = relay_url
            profile["mode"] = "relay"
            profile["last_used_utc"] = now()
            break
    profiles["active_profile_id"] = active.get("profile_id")
    write_json(hive_profiles.PROFILES_PATH, profiles)
    hive_profiles.switch_profile(str(active.get("profile_id")))
    active = hive_profiles.active_profile()
    invite = invite_from_profile(active, include_token=True)
    invite_path = REPORTS / f"hive_invite_{active.get('profile_id', 'active')}.json"
    write_json(invite_path, invite)
    service_report = start_services({"start_relay": True, "restart": body.get("restart_services", True)})
    return {
        "ok": True,
        "profile": hive_profiles.public_profile(active),
        "invite": invite,
        "invite_path": rel(invite_path),
        "phone_url": phone_join_url(active),
        "qr_url": "/api/qr?text=" + quote(phone_join_url(active), safe=""),
        "service_report": service_report,
        "message": "This hive now has relay details for multi-network/workshop/company joining.",
    }


def configure_remote_relay(body: dict[str, Any]) -> dict[str, Any]:
    report = hive_remote_access.configure_relay_url(
        str(body.get("relay_url") or ""),
        out=str(body.get("out") or "reports/hive_invite_remote_private.json"),
        policy=read_json(POLICY_PATH, {}),
    )
    if report.get("ok") and body.get("start_services", True):
        report["service_report"] = start_services(
            {
                "start_relay": True,
                "restart": body.get("restart_services", True),
            }
        )
    return report


def write_remote_wireguard_guide(body: dict[str, Any]) -> dict[str, Any]:
    return hive_remote_access.write_wireguard_guide(
        out=str(body.get("out") or "reports/hive_wireguard_free_setup.md"),
        policy=read_json(POLICY_PATH, {}),
    )


def configure_public_contribution(body: dict[str, Any]) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    class Args:
        mode = str(body.get("mode") or "off")
        gateway_url = str(body.get("gateway_url") or "")
        worker_name = str(body.get("worker_name") or "")
        allow = bool(body.get("allow"))

    report = public_hive_contributor.configure_contribution(policy, Args())
    return report


def register_license(body: dict[str, Any]) -> dict[str, Any]:
    class Args:
        name = str(body.get("name") or "")
        email = str(body.get("email") or "")
        organization = str(body.get("organization") or "")
        usage = str(body.get("usage") or "personal_homelab")
        seats = int(body.get("seats") or 1)
        commercial = bool(body.get("commercial"))
        accept_terms = bool(body.get("accept_terms"))

    return license_manager.register_install(license_manager.read_json(license_manager.POLICY_PATH, {}), Args())


def import_license(body: dict[str, Any]) -> dict[str, Any]:
    return license_manager.import_license(
        license_manager.read_json(license_manager.POLICY_PATH, {}),
        file_path=str(body.get("file") or ""),
        raw=str(body.get("license_json") or ""),
    )


def license_request(body: dict[str, Any]) -> dict[str, Any]:
    features = body.get("features") if isinstance(body.get("features"), list) else []
    return license_manager.license_request(license_manager.read_json(license_manager.POLICY_PATH, {}), [str(item) for item in features])


def write_usb_bundle(body: dict[str, Any]) -> dict[str, Any]:
    out = resolve_path(str(body.get("out") or "dist/universal-usb/ProjectTheseusUniversalUSB"))
    invite = str(body.get("invite") or "")
    try:
        return hive_usb_writer.build_bundle(
            out=out,
            coordinator_url=str(body.get("coordinator_url") or default_coordinator_url()),
            invite=resolve_path(invite) if invite else None,
            expires_days=int(body.get("expires_days") or 30),
            include_heavy_data=bool(body.get("include_heavy_data")),
            zip_bundle=not bool(body.get("no_zip")),
            force=bool(body.get("force", True)),
            hive_mode=str(body.get("hive_mode") or "current"),
            new_hive_name=str(body.get("new_hive_name") or "Project Theseus Hive USB"),
            new_hive_tier=str(body.get("new_hive_tier") or "private"),
            activate_new_hive=not bool(body.get("no_activate_new_hive")),
            public_gateway_url=str(body.get("public_gateway_url") or ""),
            public_mode=str(body.get("public_mode") or "idle"),
            public_worker_name=str(body.get("public_worker_name") or ""),
            target=resolve_path(str(body.get("target") or "")) if body.get("target") else None,
            confirm_label=str(body.get("confirm_label") or ""),
            yes=bool(body.get("yes")),
            usb_label=str(body.get("usb_label") or "THESEUS_HIVE"),
            dry_run=bool(body.get("dry_run")),
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def configure_updates(body: dict[str, Any]) -> dict[str, Any]:
    class Args:
        mode = str(body.get("mode") or "")
        channel = str(body.get("channel") or "")
        track = str(body.get("track") or "")
        catalog_url = str(body.get("catalog_url") or "")
        check_on_start = bool(body.get("check_on_start"))
        no_check_on_start = bool(body.get("no_check_on_start"))
        auto_install_soft = bool(body.get("auto_install_soft"))
        no_auto_install_soft = bool(body.get("no_auto_install_soft"))
        auto_install_hard = bool(body.get("auto_install_hard"))
        no_auto_install_hard = bool(body.get("no_auto_install_hard"))
        allow_prerelease = bool(body.get("allow_prerelease"))
        no_allow_prerelease = bool(body.get("no_allow_prerelease"))

    return update_manager.configure_client(update_manager.read_json(update_manager.POLICY_PATH, {}), Args())


def check_updates(body: dict[str, Any]) -> dict[str, Any]:
    class Args:
        catalog_url = str(body.get("catalog_url") or "")
        update_id = str(body.get("update_id") or "")
        apply = bool(body.get("apply"))
        if_enabled_on_start = bool(body.get("if_enabled_on_start"))
        respect_interval = bool(body.get("respect_interval"))

    return update_manager.check_for_updates(update_manager.read_json(update_manager.POLICY_PATH, {}), Args())


def apply_update(body: dict[str, Any]) -> dict[str, Any]:
    mode = str(body.get("mode") or "soft")
    if mode == "hard" and not body.get("allow_hard"):
        return {"ok": False, "error": "hard_update_requires_explicit_confirmation"}

    class Args:
        mode = str(body.get("mode") or "soft")
        execute = bool(body.get("execute", True))
        allow_hard = bool(body.get("allow_hard"))
        restart = bool(body.get("restart"))
        offer = str(body.get("offer") or "")

    return update_manager.apply_update(update_manager.read_json(update_manager.POLICY_PATH, {}), Args())


def start_services(body: dict[str, Any]) -> dict[str, Any]:
    active = hive_profiles.active_profile()
    dashboard_port = int(body.get("dashboard_port") or 8787)
    hive_port = int(body.get("hive_port") or 8791)
    relay_port = int(body.get("relay_port") or 8793)
    start_relay = bool(body.get("start_relay") or (active and active.get("mode") == "relay"))
    restart = bool(body.get("restart"))
    python = python_executable()
    started: list[dict[str, Any]] = []
    if restart:
        stop_known_processes()
        time.sleep(0.3)
    dashboard_process = existing_service_processes(["sparkstream_dashboard.py", "hive_operator_dashboard.py"])
    hive_process = existing_service_processes(["hive_node.py"])
    relay_process = existing_service_processes(["hive_relay.py"])
    if dashboard_process:
        started.append(reused_service("dashboard", dashboard_process))
    elif not port_open("127.0.0.1", dashboard_port):
        started.append(spawn([python, "scripts/sparkstream_dashboard.py", "--host", "127.0.0.1", "--port", str(dashboard_port)], "dashboard"))
    if hive_process:
        started.append(reused_service("hive_node", hive_process))
    elif not port_open("127.0.0.1", hive_port):
        args = [python, "scripts/hive_node.py", "daemon", "--port", str(hive_port)]
        if active.get("relay_url"):
            args += ["--relay-url", str(active.get("relay_url"))]
        if active.get("hive_id"):
            args += ["--hive-id", str(active.get("hive_id"))]
        started.append(spawn(args, "hive_node", env_for_profile(active)))
    if start_relay and relay_process:
        started.append(reused_service("hive_relay", relay_process))
    elif start_relay and not port_open("127.0.0.1", relay_port):
        started.append(spawn([python, "scripts/hive_relay.py", "--port", str(relay_port)], "hive_relay", env_for_profile(active)))
    subprocess.run(
        [python, "scripts/hive_scheduler.py", "--out", "reports/hive_scheduler.json", "--worker-chunks"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=theseus_runtime.runtime_env(),
    )
    update_check = startup_update_check()
    return {
        "ok": True,
        "started": started,
        "dashboard_url": f"http://127.0.0.1:{dashboard_port}",
        "hive_status_url": f"http://127.0.0.1:{hive_port}/api/hive/status",
        "relay_mobile_url": f"http://127.0.0.1:{relay_port}/mobile" if start_relay else "",
        "update_check": update_check,
    }


def startup_update_check() -> dict[str, Any]:
    try:
        class Args:
            catalog_url = ""
            update_id = ""
            apply = False
            if_enabled_on_start = True
            respect_interval = True

        return update_manager.check_for_updates(update_manager.read_json(update_manager.POLICY_PATH, {}), Args())
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def invite_from_profile(profile: dict[str, Any], *, include_token: bool) -> dict[str, Any]:
    if not profile:
        return {}
    expires = datetime.now(timezone.utc) + timedelta(days=30)
    roaming = hive_remote_access.mobile_roaming_profile(profile, policy=read_json(POLICY_PATH, {}), include_token=include_token)
    invite = {
        "policy": "project_theseus_hive_invite_v0",
        "created_utc": now(),
        "expires_utc": expires.isoformat(),
        "hive_id": profile.get("hive_id"),
        "hive_name": profile.get("name"),
        "tier": profile.get("tier"),
        "relay_url": profile.get("relay_url"),
        "coordinator_url": roaming.get("coordinator_url", default_coordinator_url()),
        "coordinator_urls": roaming.get("coordinator_urls", []),
        "node_urls": roaming.get("node_urls", []),
        "relay_urls": roaming.get("relay_urls", []),
        "operator_urls": roaming.get("operator_urls", []),
        "roaming": roaming.get("roaming", {}),
        "ios_app_url": roaming.get("ios_app_url", "") if include_token else "",
        "allowed_task_scope": allowed_task_scope(str(profile.get("tier") or "private")),
        "install": {
            "windows": ["Double-click Project Theseus Setup, choose Join Hive, and paste/import this invite."],
            "mac_linux": ["Open Project Theseus Setup, choose Join Hive, and paste/import this invite."],
            "phone": [phone_join_url(profile) or "Open the relay mobile URL and enter the invite token."],
        },
        "security_notes": [
            "Treat this invite like a password.",
            "Remote workers can only run registered bounded Hive task kinds.",
        ],
    }
    if include_token:
        invite["join_token"] = profile.get("join_token")
    return invite


def allowed_task_scope(tier: str) -> list[str]:
    policy = read_json(POLICY_PATH, {})
    return list(policy.get("relay", {}).get("allowed_remote_task_kinds_by_tier", {}).get(tier, []))


def phone_join_url(profile: dict[str, Any]) -> str:
    relay = str(profile.get("relay_url") or "").rstrip("/")
    if not relay:
        return ""
    hive_id = quote(str(profile.get("hive_id") or ""), safe="")
    token = quote(str(profile.get("join_token") or ""), safe="")
    return f"{relay}/m?h={hive_id}&t={token}"


def default_relay_url() -> str:
    return f"http://{best_local_ip()}:{read_json(POLICY_PATH, {}).get('relay', {}).get('server_port', 8793)}"


def default_coordinator_url() -> str:
    status = read_json(REPORTS / "hive_status.json", {})
    if isinstance(status, dict) and status.get("api_url"):
        return str(status.get("api_url"))
    return f"http://{best_local_ip()}:8791"


def public_disabled() -> dict[str, Any]:
    return {
        "ok": False,
        "error": "public_hive_disabled",
        "message": "Public Hive is a visible target but disabled until signed workers, sandboxing, reputation, abuse controls, and legal terms exist.",
    }


def env_for_profile(profile: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    if profile.get("join_token"):
        env["THESEUS_HIVE_SECRET"] = str(profile.get("join_token"))
    if profile.get("hive_id"):
        env["THESEUS_HIVE_ID"] = str(profile.get("hive_id"))
    if profile.get("relay_url"):
        env["THESEUS_HIVE_RELAY_URL"] = str(profile.get("relay_url"))
    if profile.get("tier"):
        env["THESEUS_HIVE_TIER"] = str(profile.get("tier"))
    return env


def spawn(command: list[str], label: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    spawn_env = (env or os.environ.copy()).copy()
    spawn_env.update(theseus_runtime.runtime_env())
    kwargs: dict[str, Any] = {"cwd": ROOT, "env": spawn_env}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
    return {"label": label, "pid": proc.pid, "command": command[:3] + ["..."] if len(command) > 3 else command}


def reused_service(label: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
    first = matches[0] if matches else {}
    return {
        "label": label,
        "pid": int(first.get("pid") or 0) or None,
        "reused_existing": True,
        "duplicate_existing_count": max(0, len(matches) - 1),
        "command_preview": str(first.get("command_line") or "")[:240],
    }


def existing_service_processes(needles: list[str]) -> list[dict[str, Any]]:
    lowered_needles = [needle.lower().replace("\\", "/") for needle in needles]
    rows = process_snapshot()
    matches = []
    for row in rows:
        pid = int(row.get("pid") or 0)
        command_line = str(row.get("command_line") or "")
        lowered = command_line.lower().replace("\\", "/")
        if pid == os.getpid() or ("powershell" in lowered and "-command" in lowered):
            continue
        if any(needle in lowered for needle in lowered_needles):
            matches.append({"pid": pid, "command_line": command_line})
    matches.sort(key=lambda row: int(row.get("pid") or 0), reverse=True)
    return matches


def process_snapshot() -> list[dict[str, Any]]:
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 2",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                capture_output=True,
                timeout=15,
            )
        except Exception:
            return []
        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            return []
        items = payload if isinstance(payload, list) else [payload]
        return [
            {"pid": int(item.get("ProcessId") or 0), "command_line": str(item.get("CommandLine") or "")}
            for item in items
            if isinstance(item, dict) and item.get("CommandLine")
        ]
    try:
        result = subprocess.run(["ps", "-eo", "pid=,args="], cwd=ROOT, check=False, text=True, capture_output=True, timeout=15)
    except Exception:
        return []
    rows = []
    for line in (result.stdout or "").splitlines():
        pid, _, command_line = line.strip().partition(" ")
        if pid.isdigit() and command_line:
            rows.append({"pid": int(pid), "command_line": command_line})
    return rows


def stop_known_processes(force: bool = False) -> list[dict[str, Any]]:
    if platform.system() != "Windows":
        return []
    patterns = [
        r"sparkstream_dashboard\.py",
        r"hive_node\.py",
        r"hive_relay\.py",
        r"theseus_setup_wizard\.py",
    ]
    if force:
        patterns.extend(
            [
                r"sparkstream_daemon\.py",
                r"autonomy_cycle\.py",
                r"code_lm_closure\.py",
                r"legacy_port_mechanisms\.py",
                r"symliquid-cli\.exe",
            ]
        )
    pattern_literal = "|".join(patterns).replace("'", "''")
    ps = f"""
$pattern = '{pattern_literal}'
$rows = Get-CimInstance Win32_Process |
  Where-Object {{ $_.CommandLine -and $_.CommandLine -match $pattern }} |
  Select-Object ProcessId,Name,CommandLine
foreach ($row in $rows) {{
  Stop-Process -Id $row.ProcessId -Force -ErrorAction SilentlyContinue
}}
$rows | ConvertTo-Json -Compress
"""
    result = subprocess.run(["powershell", "-NoProfile", "-Command", ps], cwd=ROOT, check=False, text=True, capture_output=True)
    if not result.stdout.strip():
        return []
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return [{"error": "stop_report_parse_failed", "stdout": result.stdout.strip()}]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def local_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        ips.update(info[4][0] for info in infos)
    except OSError:
        pass
    best = best_local_ip()
    if best:
        ips.add(best)
    return sorted(ips)


def best_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def python_executable() -> str:
    local = ROOT / ".venv-puffer" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    return str(local) if local.exists() else sys.executable


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_path(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def first(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key) or []
    return values[0] if values else ""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project Theseus Setup</title>
  <style>
    :root { color-scheme: dark; --bg:#111418; --panel:#1b2026; --line:#343d47; --text:#eef3f8; --muted:#a8b2bd; --accent:#57b8ff; --good:#79d279; --warn:#f2c66d; --bad:#ef7d7d; }
    * { box-sizing:border-box; }
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }
    main { max-width:1080px; margin:0 auto; padding:24px; display:grid; gap:18px; }
    header { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
    h1 { margin:0; font-size:32px; }
    h2 { margin:0 0 10px; font-size:20px; }
    p { color:var(--muted); line-height:1.45; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; display:grid; gap:10px; }
    .choice { cursor:pointer; min-height:150px; }
    .choice:hover, .selected { border-color:var(--accent); }
    label { color:var(--muted); font-size:13px; display:grid; gap:6px; }
    input, textarea, select, button { width:100%; border:1px solid var(--line); border-radius:8px; padding:10px; font:inherit; background:#12161b; color:var(--text); }
    button { cursor:pointer; background:#24303a; }
    button.primary { background:#155f93; border-color:#2588ca; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .row { display:flex; gap:10px; flex-wrap:wrap; }
    .row > * { flex:1 1 160px; }
    .pill { display:inline-flex; align-items:center; gap:6px; padding:4px 8px; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:12px; }
    .good { color:var(--good); } .warn { color:var(--warn); } .bad { color:var(--bad); }
    pre { margin:0; white-space:pre-wrap; overflow-wrap:anywhere; max-height:320px; overflow:auto; background:#0b0e11; border:1px solid var(--line); border-radius:8px; padding:12px; color:#d8e0e8; }
    .qr { background:white; padding:10px; width:max-content; max-width:100%; }
    .qr img { display:block; max-width:240px; width:100%; height:auto; }
    .hidden { display:none; }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Project Theseus Setup</h1>
      <p>Set up a personal Hive, join a trusted Hive, switch between Hives, or add your phone. No terminal needed.</p>
    </div>
    <button id="refresh">Refresh</button>
  </header>

  <section class="grid">
    <div class="card choice selected" data-mode="create">
      <h2>Create a Hive</h2>
      <p>First computer in a home, workshop, or company Hive. Creates invite details for more machines.</p>
    </div>
    <div class="card choice" data-mode="join">
      <h2>Join a Hive</h2>
      <p>Use an invite file, pasted invite text, or join details from another trusted computer.</p>
    </div>
    <div class="card choice" data-mode="switch">
      <h2>Switch Hives</h2>
      <p>Move this computer between home, work, friends, and future public Hives.</p>
    </div>
  </section>

  <section class="card">
    <h2>Registration / License</h2>
    <p>Register this install before creating or joining hives. Homelab and research use is free under the small-hive cap; company and public-gateway use require a signed license.</p>
    <div id="license-status"></div>
    <div class="row">
      <label>Name<input id="license-name" placeholder="Your name"></label>
      <label>Email<input id="license-email" placeholder="you@example.com"></label>
      <label>Organization<input id="license-org" placeholder="Home, lab, or company"></label>
    </div>
    <div class="row">
      <label>Usage
        <select id="license-usage">
          <option value="personal_homelab">Personal homelab</option>
          <option value="research">Research</option>
          <option value="startup_free">Startup/free under 12 seats</option>
          <option value="company">Company / commercial</option>
          <option value="public_operator">Public Hive operator</option>
        </select>
      </label>
      <label>Seats<input id="license-seats" value="1"></label>
      <label><input id="license-commercial" type="checkbox"> commercial use</label>
    </div>
    <label><input id="license-terms" type="checkbox"> I accept the current Project Theseus terms for this install.</label>
    <div class="row">
      <button id="license-register">Register Install</button>
      <button id="license-request">Create License Request</button>
    </div>
    <textarea id="license-import-text" rows="5" placeholder="Paste signed license JSON here"></textarea>
    <button id="license-import">Import Signed License</button>
  </section>

  <section class="card">
    <h2>Public Contribution</h2>
    <p>Let this machine lend idle compute to the larger Theseus Hive while keeping private Hive data, local files, ROMs, and teacher access sealed off.</p>
    <div class="row">
      <label>Mode
        <select id="public-mode">
          <option value="off">Off</option>
          <option value="idle">Idle only</option>
          <option value="always">Always when resource gates are green</option>
        </select>
      </label>
      <label>Public gateway URL<input id="public-gateway" placeholder="https://public.theseus-hive.example"></label>
      <label>Worker name<input id="public-worker-name" placeholder="This computer"></label>
    </div>
    <label><input id="public-allow" type="checkbox"> I understand this donates bounded idle compute to public Theseus training, without sharing private Hive data.</label>
    <div class="row">
      <button id="public-save">Save Public Contribution</button>
      <button id="public-poll">Check Public Queue Once</button>
    </div>
    <div id="public-status"></div>
  </section>

  <section class="card">
    <h2>Free Remote Access</h2>
    <p>Use same-LAN/hotspot, self-hosted WireGuard, or the built-in authenticated relay. Paid mesh services are optional, not required. Do not forward the node API port directly to the internet.</p>
    <div id="remote-status"></div>
    <div class="row">
      <label>Relay URL<input id="remote-relay-url" placeholder="http://10.87.0.1:8793"></label>
      <button id="remote-configure">Configure Relay</button>
      <button id="remote-guide">Write WireGuard Guide</button>
    </div>
  </section>

  <section class="card">
    <h2>Updates</h2>
    <div id="update-status"></div>
    <div class="row">
      <label>Update mode
        <select id="update-mode">
          <option value="notify">Notify me first</option>
          <option value="manual">Manual only</option>
          <option value="auto_soft">Auto-install safe model updates</option>
          <option value="auto_safe">Auto-install safe updates</option>
        </select>
      </label>
      <label>Channel
        <select id="update-channel">
          <option value="community">Community</option>
          <option value="public">Public</option>
          <option value="company">Company</option>
        </select>
      </label>
      <label>Track
        <select id="update-track">
          <option value="stable">Stable</option>
          <option value="beta">Beta</option>
          <option value="dev">Dev</option>
        </select>
      </label>
    </div>
    <label>Catalog URL<input id="update-catalog" placeholder="Official catalog URL or private Hive catalog"></label>
    <div class="row">
      <label>Available update<select id="update-offer"></select></label>
      <label><input id="update-check-start" type="checkbox"> check when Project Theseus starts</label>
      <label><input id="update-auto-soft" type="checkbox"> safe auto-install</label>
      <label><input id="update-auto-hard" type="checkbox"> app/source auto-install</label>
      <label><input id="update-prerelease" type="checkbox"> allow prerelease</label>
    </div>
    <div class="row">
      <button id="update-save">Save Update Settings</button>
      <button id="update-check">Check Now</button>
      <button id="update-install" class="primary">Install Safe Update</button>
    </div>
  </section>

  <section id="create-panel" class="card">
    <h2>Create a Hive</h2>
    <div class="row">
      <label>Hive name<input id="create-name" value="My Project Theseus Hive"></label>
      <label>Hive type
        <select id="create-tier">
          <option value="private">Private: my machines</option>
          <option value="friends_family">Semi-private: friends and family</option>
          <option value="company">Company: paid license required</option>
          <option value="public">Public: anyone can join later</option>
        </select>
      </label>
    </div>
    <div class="row">
      <label>Connection style
        <select id="create-mode">
          <option value="lan">Same network only</option>
          <option value="relay">Multiple networks / phones / workshop</option>
        </select>
      </label>
      <label>Relay URL<input id="create-relay" placeholder="auto-filled for this computer"></label>
    </div>
    <button id="create-hive" class="primary">Create and Start</button>
  </section>

  <section id="join-panel" class="card hidden">
    <h2>Join a Hive</h2>
    <p>Paste the invite JSON, or fill in the details manually. Treat invite tokens like passwords.</p>
    <textarea id="join-invite" rows="8" placeholder="Paste invite JSON here"></textarea>
    <div class="row">
      <label>Hive ID<input id="join-hive-id"></label>
      <label>Invite token<input id="join-token" type="password"></label>
      <label>Relay URL<input id="join-relay"></label>
    </div>
    <div class="row">
      <label>Display name<input id="join-name" placeholder="Home Hive"></label>
      <label>Type<select id="join-tier"><option value="private">Private</option><option value="friends_family">Semi-private</option><option value="company">Company</option></select></label>
    </div>
    <button id="join-hive" class="primary">Join and Start</button>
  </section>

  <section id="switch-panel" class="card hidden">
    <h2>Switch Hives</h2>
    <div id="profiles"></div>
  </section>

  <section class="grid">
    <div class="card">
      <h2>Current Hive</h2>
      <div id="current"></div>
      <div class="row">
        <button id="start-services">Start App</button>
        <button id="upgrade-relay">Enable Phones / Multi-network</button>
      </div>
    </div>
    <div class="card">
      <h2>Add a Phone</h2>
      <p>Scan this from iPhone or Android after relay/multi-network mode is enabled.</p>
      <div id="phone"></div>
    </div>
  </section>

  <section class="card">
    <h2>Installer USB</h2>
    <div class="row">
      <label>USB target<select id="usb-target"></select></label>
      <label>Confirm label<input id="usb-confirm" placeholder="Select a USB first"></label>
      <label>New USB label<input id="usb-label" value="THESEUS_HIVE"></label>
    </div>
    <div class="row">
      <label>Output folder<input id="usb-out" value="dist/universal-usb/ProjectTheseusUniversalUSB"></label>
      <label>Coordinator URL<input id="usb-coordinator" placeholder="http://this-computer:8791"></label>
      <label>Invite days<input id="usb-expires" value="30"></label>
    </div>
    <div class="row">
      <label>Hive mode
        <select id="usb-hive-mode">
          <option value="current">Use current Hive</option>
          <option value="new">Create new private Hive</option>
          <option value="public">Public worker</option>
        </select>
      </label>
      <label>New Hive name<input id="usb-new-hive-name" value="Project Theseus Hive USB"></label>
      <label>New Hive type
        <select id="usb-new-hive-tier">
          <option value="private">Private</option>
          <option value="friends_family">Friends/family</option>
          <option value="company">Company</option>
        </select>
      </label>
    </div>
    <div class="row">
      <label>Public mode
        <select id="usb-public-mode">
          <option value="idle">Idle</option>
          <option value="always">Always</option>
          <option value="off">Off</option>
        </select>
      </label>
      <label>Public gateway URL<input id="usb-public-gateway"></label>
      <label>Public worker name<input id="usb-public-worker"></label>
    </div>
    <div class="row">
      <label><input id="usb-force" type="checkbox" checked> replace existing bundle</label>
      <label><input id="usb-yes" type="checkbox"> I picked the correct USB</label>
      <label><input id="usb-heavy" type="checkbox"> include heavy data</label>
      <label><input id="usb-nozip" type="checkbox"> folder only</label>
    </div>
    <button id="usb-write" class="primary">Write Universal Installer</button>
    <div id="usb-status"></div>
  </section>

  <section class="card">
    <h2>Output</h2>
    <pre id="out">Ready.</pre>
  </section>
</main>
<script>
let state = {};
let selectedMode = "create";
const $ = (id) => document.getElementById(id);
async function api(path, body) {
  const res = await fetch(path, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(body || {}) });
  const data = await res.json();
  $("out").textContent = JSON.stringify(data, null, 2);
  await refresh();
  return data;
}
async function refresh() {
  state = await (await fetch("/api/status")).json();
  const ip = (state.local_ips || []).find((x) => !x.startsWith("127.")) || "127.0.0.1";
  if (!$("create-relay").value) $("create-relay").placeholder = `http://${ip}:8793`;
  const usb = state.usb_writer || {};
  if ($("usb-out") && !$("usb-out").value) $("usb-out").value = usb.default_out || "dist/universal-usb/ProjectTheseusUniversalUSB";
  if ($("usb-coordinator") && !$("usb-coordinator").value) $("usb-coordinator").value = usb.default_coordinator_url || `http://${ip}:8791`;
  renderUsbTargets(usb.targets || []);
  render();
}
function render() {
  const active = state.active_profile || {};
  const license = state.license || {};
  const feature = license.feature_summary || {};
  const entitlement = license.entitlement || {};
  $("license-status").innerHTML = `
    <p><span class="pill ${license.registration_complete ? "good" : "warn"}">${license.registration_complete ? "Registered" : "Registration required"}</span>
    <span class="pill">${escapeHtml(entitlement.tier || "unregistered")}</span>
    <span class="pill">${escapeHtml(entitlement.source || "")}</span></p>
    <p>${escapeHtml(license.next_action || "")}</p>
    <p>Nodes ${escapeHtml((license.hive || {}).node_count || 1)} / ${escapeHtml(entitlement.node_limit || 1)} - company ${feature.can_create_company_hive ? "allowed" : "locked"} - public gateway ${feature.can_operate_public_gateway ? "allowed" : "locked"}</p>
  `;
  $("current").innerHTML = active.profile_id ? `
    <div class="pill good">Active</div>
    <p><strong>${escapeHtml(active.name || active.hive_id)}</strong></p>
    <p>${escapeHtml(active.tier || "")} / ${escapeHtml(active.mode || "lan")}</p>
    <p>${escapeHtml(active.relay_url || "Same-network LAN only")}</p>
  ` : `<p>No active Hive yet. Create one or join one.</p>`;
  const invite = state.active_invite || {};
  const phoneUrl = invite.install && invite.install.phone && invite.install.phone[0] || "";
  const iosAppUrl = invite.ios_app_url || "";
  const qrUrl = iosAppUrl || (phoneUrl.startsWith("http") ? phoneUrl : "");
  if (phoneUrl && phoneUrl.startsWith("http")) {
    $("phone").innerHTML = `
      ${qrUrl ? `<div class="qr"><img alt="Phone join QR" src="/api/qr?text=${encodeURIComponent(qrUrl)}"></div>` : ""}
      ${iosAppUrl ? `<p><strong>iPhone app</strong><br><input readonly value="${escapeAttr(iosAppUrl)}"></p>` : ""}
      <p><strong>Web fallback</strong><br><input readonly value="${escapeAttr(phoneUrl)}"></p>
    `;
  } else {
    $("phone").innerHTML = iosAppUrl ? `
      <div class="qr"><img alt="iPhone app join QR" src="/api/qr?text=${encodeURIComponent(iosAppUrl)}"></div>
      <p><strong>iPhone app</strong><br><input readonly value="${escapeAttr(iosAppUrl)}"></p>
      <p class="warn">For cellular access, configure a private tunnel or HTTPS relay endpoint.</p>
    ` : `<p class="warn">Enable phones / multi-network to create a relay phone link.</p>`;
  }
  const profiles = ((state.profiles || {}).profiles || []);
  $("profiles").innerHTML = profiles.length ? profiles.map((p) => `
    <div class="card">
      <strong>${escapeHtml(p.name || p.hive_id)}</strong>
      <span class="pill">${escapeHtml(p.tier)}</span>
      <span class="pill">${escapeHtml(p.mode || "lan")}</span>
      <p>${escapeHtml(p.relay_url || "LAN only")}</p>
      <button data-switch="${escapeAttr(p.profile_id)}">Switch to this Hive</button>
    </div>
  `).join("") : "<p>No saved Hives yet.</p>";
  document.querySelectorAll("[data-switch]").forEach((btn) => btn.onclick = () => api("/api/switch-hive", {profile_id: btn.dataset.switch, restart_services: true}));
  const pub = state.public_contribution || {};
  if ($("public-mode")) $("public-mode").value = pub.mode || "off";
  if ($("public-gateway") && !$("public-gateway").value) $("public-gateway").value = pub.gateway_url || "";
  if ($("public-worker-name") && !$("public-worker-name").value) $("public-worker-name").value = (pub.worker || {}).worker_name || "";
  if ($("public-allow")) $("public-allow").checked = !!pub.explicit_opt_in;
  $("public-status").innerHTML = `
    <p><span class="pill ${pub.can_connect_now ? "good" : "warn"}">${pub.can_connect_now ? "Ready when tasks exist" : "Waiting"}</span></p>
    <p>${escapeHtml(pub.next_action || "")}</p>
    <pre>${escapeHtml(JSON.stringify((pub.gates || []).map((g) => ({gate:g.name, ok:g.ok, detail:g.detail})), null, 2))}</pre>
  `;
  renderRemoteAccess();
  const usb = state.usb_writer || {};
  $("usb-status").innerHTML = `
    <p><span class="pill">${escapeHtml(usb.default_coordinator_url || "")}</span></p>
  `;
  renderUpdates();
}
function renderRemoteAccess() {
  const remote = state.remote_access || {};
  const relay = remote.relay || {};
  const local = remote.local || {};
  const cost = remote.cost_policy || {};
  const actions = remote.next_actions || [];
  if ($("remote-relay-url") && !$("remote-relay-url").value) $("remote-relay-url").value = relay.url || "";
  $("remote-status").innerHTML = `
    <p>
      <span class="pill ${relay.configured ? "good" : "warn"}">${relay.configured ? "Relay configured" : "LAN/hotspot only"}</span>
      <span class="pill ${local.relay_live ? "good" : "warn"}">${local.relay_live ? "Relay running" : "Relay stopped"}</span>
      <span class="pill ${local.wireguard_tools_installed ? "good" : ""}">WireGuard tools ${local.wireguard_tools_installed ? "found" : "optional"}</span>
      <span class="pill">${cost.paid_dependency_required ? "Paid dependency" : "No paid dependency"}</span>
    </p>
    <p>${escapeHtml(relay.url || "Configure a relay URL after same-LAN, hotspot, or a private tunnel is available.")}</p>
    <pre>${escapeHtml(JSON.stringify({scope: relay.scope || "not_configured", warnings: relay.warnings || [], next_actions: actions}, null, 2))}</pre>
  `;
}
function renderUpdates() {
  const updates = state.updates || {};
  const client = updates.client || {};
  const offer = updates.current_offer || {};
  const installed = updates.installed || {};
  const checkin = updates.last_checkin || {};
  if ($("update-mode")) $("update-mode").value = client.mode || "notify";
  if ($("update-channel")) $("update-channel").value = client.channel || "community";
  if ($("update-track")) $("update-track").value = client.track || "stable";
  if ($("update-catalog")) $("update-catalog").value = client.catalog_url || "";
  if ($("update-check-start")) $("update-check-start").checked = !!client.check_on_start;
  if ($("update-auto-soft")) $("update-auto-soft").checked = !!client.auto_install_soft;
  if ($("update-auto-hard")) $("update-auto-hard").checked = !!client.auto_install_hard;
  if ($("update-prerelease")) $("update-prerelease").checked = !!client.allow_prerelease;
  const offers = updates.catalog_offers || [];
  const selected = offer.update_id || "";
  $("update-offer").innerHTML = `<option value="">Latest compatible update</option>` + offers.map((o) => {
    const label = `${o.update_id || "update"} ${o.checkpoint_id || ""} ${o.published_utc || ""}`;
    return `<option value="${escapeAttr(o.update_id || "")}">${escapeHtml(label)}</option>`;
  }).join("");
  if (selected) $("update-offer").value = selected;
  $("update-status").innerHTML = `
    <p>
      <span class="pill ${updates.update_available ? "warn" : "good"}">${updates.update_available ? "Update available" : "Current or unchecked"}</span>
      <span class="pill">${escapeHtml(client.mode || "notify")}</span>
      <span class="pill">${escapeHtml(checkin.catalog_source || "local/offline")}</span>
    </p>
    <p>${escapeHtml(updates.next_action || "")}</p>
    <p>${escapeHtml(offer.headline || installed.headline || "No update selected yet.")}</p>
    <p>Last check: ${escapeHtml(checkin.created_utc || "never")} / catalog ${checkin.catalog_ok ? "ok" : "not confirmed"}</p>
  `;
}
function renderUsbTargets(targets) {
  const select = $("usb-target");
  if (!select) return;
  const current = select.value;
  select.innerHTML = `<option value="">Build zip only</option>` + targets.map((t) => {
    const label = `${t.path} ${t.label || "NO_LABEL"} ${t.free_gib || 0} GiB free`;
    return `<option value="${escapeAttr(t.path)}" data-label="${escapeAttr(t.label || "")}">${escapeHtml(label)}</option>`;
  }).join("");
  if (current) select.value = current;
  select.onchange = () => {
    const opt = select.options[select.selectedIndex];
    $("usb-confirm").value = opt ? (opt.dataset.label || "") : "";
  };
  if (!$("usb-confirm").value && select.selectedIndex >= 0) {
    const opt = select.options[select.selectedIndex];
    $("usb-confirm").value = opt ? (opt.dataset.label || "") : "";
  }
}
document.querySelectorAll(".choice").forEach((card) => card.onclick = () => {
  selectedMode = card.dataset.mode;
  document.querySelectorAll(".choice").forEach((x) => x.classList.toggle("selected", x === card));
  ["create","join","switch"].forEach((name) => $(`${name}-panel`).classList.toggle("hidden", name !== selectedMode));
});
$("refresh").onclick = refresh;
$("license-register").onclick = () => api("/api/license/register", {name:$("license-name").value, email:$("license-email").value, organization:$("license-org").value, usage:$("license-usage").value, seats:Number($("license-seats").value || 1), commercial:$("license-commercial").checked, accept_terms:$("license-terms").checked});
$("license-request").onclick = () => api("/api/license/request", {features:["company_hive","distributed_worker_chunks"]});
$("license-import").onclick = () => api("/api/license/import", {license_json:$("license-import-text").value});
$("create-hive").onclick = () => api("/api/create-hive", {name:$("create-name").value, tier:$("create-tier").value, mode:$("create-mode").value, relay_url:$("create-relay").value, start_services:true});
$("join-hive").onclick = () => api("/api/join-hive", {invite_text:$("join-invite").value, hive_id:$("join-hive-id").value, join_token:$("join-token").value, relay_url:$("join-relay").value, name:$("join-name").value, tier:$("join-tier").value, start_services:true});
$("start-services").onclick = () => api("/api/start-services", {});
$("upgrade-relay").onclick = () => api("/api/upgrade-relay", {});
$("public-save").onclick = () => api("/api/public-contribution", {mode:$("public-mode").value, gateway_url:$("public-gateway").value, worker_name:$("public-worker-name").value, allow:$("public-allow").checked});
$("public-poll").onclick = () => api("/api/public-contribution/poll-once", {});
$("remote-configure").onclick = () => api("/api/remote/configure-relay", {relay_url:$("remote-relay-url").value, start_services:true, restart_services:true});
$("remote-guide").onclick = () => api("/api/remote/wireguard-guide", {});
$("update-save").onclick = () => api("/api/update/configure", {mode:$("update-mode").value, channel:$("update-channel").value, track:$("update-track").value, catalog_url:$("update-catalog").value, check_on_start:$("update-check-start").checked, no_check_on_start:!$("update-check-start").checked, auto_install_soft:$("update-auto-soft").checked, no_auto_install_soft:!$("update-auto-soft").checked, auto_install_hard:$("update-auto-hard").checked, no_auto_install_hard:!$("update-auto-hard").checked, allow_prerelease:$("update-prerelease").checked, no_allow_prerelease:!$("update-prerelease").checked});
$("update-check").onclick = () => api("/api/update/check", {catalog_url:$("update-catalog").value, update_id:$("update-offer").value});
$("update-install").onclick = () => api("/api/update/apply", {mode:"soft", execute:true});
$("usb-write").onclick = () => api("/api/usb/write", {out:$("usb-out").value, target:$("usb-target").value, confirm_label:$("usb-confirm").value, yes:$("usb-yes").checked, usb_label:$("usb-label").value, coordinator_url:$("usb-coordinator").value, hive_mode:$("usb-hive-mode").value, new_hive_name:$("usb-new-hive-name").value, new_hive_tier:$("usb-new-hive-tier").value, public_mode:$("usb-public-mode").value, public_gateway_url:$("usb-public-gateway").value, public_worker_name:$("usb-public-worker").value, expires_days:Number($("usb-expires").value || 30), force:$("usb-force").checked, include_heavy_data:$("usb-heavy").checked, no_zip:$("usb-nozip").checked});
function escapeHtml(value){ return String(value || "").replace(/[&<>"']/g, (ch) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch])); }
function escapeAttr(value){ return escapeHtml(value).replace(/`/g, "&#96;"); }
refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
