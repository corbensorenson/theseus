"""Governed remote-control session handoff for Project Theseus Hive.

This module deliberately does not implement a custom screen/input streamer.
Remote desktop is latency-sensitive and security-sensitive, so the Hive should
broker audited session metadata and launch mature transports: RDP, VNC/macOS
Screen Sharing, RustDesk, Sunshine/Moonlight, or whatever provider a node has
explicitly configured.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
POLICY_PATH = CONFIGS / "hive_policy.json"

sys.path.insert(0, str(ROOT / "scripts"))
import hive_profiles  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Project Theseus Hive remote-control broker.")
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Detect local remote-control provider readiness.")
    status.add_argument("--out", default="reports/hive_remote_control_status.json")

    request = sub.add_parser("request", help="Create an audited remote-control session handoff.")
    request.add_argument("--target-node", default="local")
    request.add_argument("--provider", default="auto")
    request.add_argument("--mode", choices=["view", "control"], default="control")
    request.add_argument("--duration-minutes", type=int, default=60)
    request.add_argument("--out", default="reports/hive_remote_control_session.json")

    launch = sub.add_parser("launch", help="Launch a local desktop client for a target host or URL.")
    launch.add_argument("--provider", choices=["rdp", "vnc", "screen_sharing", "rustdesk", "sunshine_moonlight"], default="rdp")
    launch.add_argument("--host", default="")
    launch.add_argument("--target-url", default="")
    launch.add_argument("--rustdesk-id", default="")
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--out", default="reports/hive_remote_control_launch.json")

    args = parser.parse_args()
    policy = read_json(POLICY_PATH, {})
    if args.command in {None, "status"}:
        report = status_report(policy=policy, write_report=False)
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    if args.command == "request":
        report = request_session(
            policy=policy,
            payload={
                "target_node_id": args.target_node,
                "provider": args.provider,
                "mode": args.mode,
                "duration_minutes": args.duration_minutes,
            },
        )
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    if args.command == "launch":
        report = launch_client(
            provider=args.provider,
            host=args.host,
            target_url=args.target_url,
            rustdesk_id=args.rustdesk_id,
            execute=bool(args.execute),
        )
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    parser.print_help()
    return 2


def status_report(*, policy: dict[str, Any] | None = None, write_report: bool = True) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    config = local_config(policy)
    hosts = candidate_hosts(policy)
    providers = detect_providers(policy, config, hosts)
    ready = [provider for provider in providers if provider.get("ready")]
    report = {
        "ok": True,
        "policy": "project_theseus_hive_remote_control_status_v0",
        "created_utc": now(),
        "enabled": bool(get_path(policy, ["remote_control", "enabled"], True)),
        "hive_id": active_hive_id(),
        "local": {
            "system": platform.system(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "candidate_hosts": hosts,
        },
        "providers": providers,
        "provider_count": len(providers),
        "ready_provider_count": len(ready),
        "preferred_provider_id": best_provider_id(providers),
        "operator_links": operator_links(providers, hosts[0] if hosts else ""),
        "security": security_summary(policy),
        "next_actions": next_actions(providers, policy),
    }
    if write_report:
        write_json(ROOT / str(get_path(policy, ["remote_control", "status_path"], "reports/hive_remote_control_status.json")), report)
    return report


def request_session(
    *,
    policy: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    peers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    payload = payload or {}
    if not bool(get_path(policy, ["remote_control", "enabled"], True)):
        return {"ok": False, "error": "remote_control_disabled"}
    tier = active_hive_tier()
    allowed_tiers = set(str(item) for item in get_path(policy, ["remote_control", "allowed_tiers"], ["private", "company"]))
    if tier and tier not in allowed_tiers:
        return {"ok": False, "error": "remote_control_not_allowed_for_tier", "tier": tier, "allowed_tiers": sorted(allowed_tiers)}

    target_node_id = str(payload.get("target_node_id") or "local")
    mode = str(payload.get("mode") or "control")
    provider_request = str(payload.get("provider") or "auto")
    duration_minutes = max(5, min(int(payload.get("duration_minutes") or 60), int(get_path(policy, ["remote_control", "max_session_minutes"], 240))))
    target = target_node(policy, target_node_id, peers=peers)
    if not target:
        return {"ok": False, "error": "target_node_not_found", "target_node_id": target_node_id}

    providers = target_providers(target)
    provider = choose_provider(providers, provider_request)
    if not provider:
        return {
            "ok": False,
            "error": "no_remote_control_provider_for_target",
            "target": public_target(target),
            "provider_request": provider_request,
            "providers": providers,
        }

    host = target_host(target)
    connect = connection_handoff(provider, host)
    if not provider.get("ready"):
        return {
            "ok": False,
            "error": "target_remote_control_provider_not_ready",
            "target": public_target(target),
            "provider": provider,
            "connect": connect,
            "next_actions": [
                "Enable or configure the selected remote-control host on the target node.",
                "For cross-platform phone control, RustDesk with a private/self-hosted relay is usually the easiest first provider.",
                "For Windows RDP, enable Remote Desktop and use LAN/WireGuard/private tunnel rather than exposing TCP 3389.",
            ],
        }
    session = {
        "session_id": f"hive_rc_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "created_utc": now(),
        "expires_utc": (datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)).isoformat(),
        "mode": mode,
        "provider": provider,
        "target": public_target(target),
        "connect": connect,
        "transport_contract": {
            "metadata_broker": "Hive API and relay can carry authenticated session metadata.",
            "pixel_input_transport": "Use the provider transport directly: same LAN, private VPN, or that provider's self-hosted relay.",
            "public_port_policy": get_path(policy, ["remote_control", "public_port_policy"], "never_expose_raw_desktop_ports"),
        },
        "operator_requirements": [
            "Operator device must have a compatible remote desktop client installed.",
            "Target host must already have the chosen remote-control server enabled.",
            "Use LAN, private tunnel, or self-hosted provider relay for offsite control.",
        ],
        "side_effect_class": "interactive_remote_control",
        "audit": {
            "requested_provider": provider_request,
            "approved_by": "private_hive_secret_or_loopback",
            "ledger_path": str(get_path(policy, ["remote_control", "ledger_path"], "reports/hive_remote_control_ledger.jsonl")),
        },
    }
    report = {
        "ok": True,
        "policy": "project_theseus_hive_remote_control_session_v0",
        "created_utc": now(),
        "session": session,
    }
    ledger_path = ROOT / str(get_path(policy, ["remote_control", "ledger_path"], "reports/hive_remote_control_ledger.jsonl"))
    append_jsonl(ledger_path, report)
    write_json(ROOT / str(get_path(policy, ["remote_control", "session_report_path"], "reports/hive_remote_control_session.json")), report)
    return report


def launch_client(*, provider: str, host: str, target_url: str, rustdesk_id: str, execute: bool) -> dict[str, Any]:
    handoff = connection_handoff({"id": provider, "rustdesk_id": rustdesk_id}, host)
    url = target_url or str(handoff.get("connect_url") or "")
    command = list(handoff.get("desktop_command") or [])
    report = {
        "ok": True,
        "policy": "project_theseus_hive_remote_control_launch_v0",
        "created_utc": now(),
        "provider": provider,
        "host": host,
        "target_url": url,
        "desktop_command": command,
        "executed": bool(execute),
    }
    if not execute:
        report["next_action"] = "Rerun with --execute to open the local client."
        return report
    try:
        if command:
            subprocess.Popen(command, cwd=ROOT)  # noqa: S603 - operator-requested local client launch.
        elif url:
            open_url(url)
        else:
            return {**report, "ok": False, "error": "no_launch_target"}
    except OSError as exc:
        return {**report, "ok": False, "error": "launch_failed", "message": str(exc)}
    return report


def detect_providers(policy: dict[str, Any], config: dict[str, Any], hosts: list[str]) -> list[dict[str, Any]]:
    allowed = set(str(item) for item in get_path(policy, ["remote_control", "allowed_providers"], []))
    if not allowed:
        allowed = {"rustdesk", "rdp", "vnc", "screen_sharing", "sunshine_moonlight"}
    rows = [
        detect_rustdesk(config),
        detect_rdp(config),
        detect_screen_sharing(config),
        detect_vnc(config),
        detect_sunshine(config),
    ]
    providers = [row for row in rows if row.get("id") in allowed]
    for provider in providers:
        provider["connect"] = connection_handoff(provider, hosts[0] if hosts else "")
    return providers


def detect_rustdesk(config: dict[str, Any]) -> dict[str, Any]:
    exe = first_existing(
        [
            shutil.which("rustdesk"),
            shutil.which("rustdesk.exe"),
            os.environ.get("RUSTDESK_EXE", ""),
            r"C:\Program Files\RustDesk\rustdesk.exe",
            r"C:\Program Files (x86)\RustDesk\rustdesk.exe",
            "/Applications/RustDesk.app",
            "/usr/bin/rustdesk",
            "/usr/local/bin/rustdesk",
        ]
    )
    provider_cfg = config.get("providers", {}).get("rustdesk", {}) if isinstance(config.get("providers"), dict) else {}
    rustdesk_id = str(provider_cfg.get("id") or config.get("rustdesk_id") or "")
    return {
        "id": "rustdesk",
        "label": "RustDesk",
        "installed": bool(exe),
        "configured": bool(rustdesk_id),
        "ready": bool(exe or rustdesk_id),
        "role": "cross_platform_unattended_remote_desktop",
        "executable": exe,
        "rustdesk_id": rustdesk_id,
        "strength": "Best all-around phone/desktop takeover path when each node has RustDesk and a self-hosted/private relay.",
        "notes": [
            "Works across Windows, macOS, Linux, Android, and iOS with the RustDesk client.",
            "Put each node's RustDesk ID in configs/hive_remote_control.local.json for one-tap handoff.",
        ],
    }


def detect_rdp(config: dict[str, Any]) -> dict[str, Any]:
    is_windows = platform.system() == "Windows"
    client = first_existing([shutil.which("mstsc"), shutil.which("mstsc.exe"), r"C:\Windows\System32\mstsc.exe"])
    enabled = windows_rdp_enabled() if is_windows else False
    service = windows_service_running("TermService") if is_windows else False
    provider_cfg = config.get("providers", {}).get("rdp", {}) if isinstance(config.get("providers"), dict) else {}
    return {
        "id": "rdp",
        "label": "Microsoft Remote Desktop",
        "installed": bool(client),
        "configured": bool(enabled or provider_cfg.get("enabled")),
        "ready": bool(is_windows and enabled and service),
        "role": "windows_host_or_desktop_client",
        "executable": client,
        "host_enabled": enabled,
        "service_running": service,
        "default_port": 3389,
        "strength": "Excellent for Windows PCs over LAN/VPN; iOS and Android clients are mature.",
        "notes": [
            "Do not expose TCP 3389 directly to the internet.",
            "Use WireGuard/private tunnel or an RD Gateway for offsite access.",
        ],
    }


def detect_screen_sharing(config: dict[str, Any]) -> dict[str, Any]:
    is_mac = platform.system() == "Darwin"
    app = "/System/Library/CoreServices/Applications/Screen Sharing.app"
    host_enabled = mac_service_loaded("com.apple.screensharing") if is_mac else False
    return {
        "id": "screen_sharing",
        "label": "macOS Screen Sharing",
        "installed": bool(is_mac and Path(app).exists()),
        "configured": bool(host_enabled),
        "ready": bool(is_mac and host_enabled),
        "role": "macos_vnc_screen_sharing",
        "app": app if is_mac else "",
        "default_port": 5900,
        "strength": "Native for Mac-to-Mac and VNC-compatible clients over LAN/VPN.",
        "notes": ["Use `vnc://HOST` from Mac, or a VNC client from phone/Windows/Linux."],
    }


def detect_vnc(config: dict[str, Any]) -> dict[str, Any]:
    client = first_existing([shutil.which("vncviewer"), shutil.which("vncviewer.exe"), shutil.which("xtigervncviewer")])
    local_port = port_open("127.0.0.1", 5900)
    provider_cfg = config.get("providers", {}).get("vnc", {}) if isinstance(config.get("providers"), dict) else {}
    return {
        "id": "vnc",
        "label": "VNC",
        "installed": bool(client),
        "configured": bool(local_port or provider_cfg.get("enabled")),
        "ready": bool(local_port or provider_cfg.get("enabled")),
        "role": "generic_vnc",
        "executable": client,
        "default_port": 5900,
        "strength": "Simple cross-platform fallback when a VNC server is explicitly enabled.",
        "notes": ["VNC should stay behind LAN/VPN unless wrapped by a hardened gateway."],
    }


def detect_sunshine(config: dict[str, Any]) -> dict[str, Any]:
    exe = first_existing([shutil.which("sunshine"), shutil.which("sunshine.exe"), r"C:\Program Files\Sunshine\sunshine.exe"])
    local_port = port_open("127.0.0.1", 47990)
    provider_cfg = config.get("providers", {}).get("sunshine_moonlight", {}) if isinstance(config.get("providers"), dict) else {}
    return {
        "id": "sunshine_moonlight",
        "label": "Sunshine / Moonlight",
        "installed": bool(exe),
        "configured": bool(local_port or provider_cfg.get("enabled")),
        "ready": bool(local_port or provider_cfg.get("enabled")),
        "role": "low_latency_streaming_control",
        "executable": exe,
        "default_port": 47990,
        "strength": "Best for low-latency GPU/workshop control once paired.",
        "notes": ["Pair Moonlight clients with Sunshine per node; keep it private/VPN unless deliberately hardened."],
    }


def connection_handoff(provider: dict[str, Any], host: str) -> dict[str, Any]:
    provider_id = str(provider.get("id") or "")
    host = clean_host(host)
    if provider_id == "rdp":
        target = host or "HOST"
        return {
            "connect_url": f"rdp://full%20address=s:{quote(target, safe='')}:3389",
            "desktop_command": ["mstsc.exe", f"/v:{target}"] if platform.system() == "Windows" else [],
            "phone_client_hint": "Open with Microsoft Remote Desktop or another RDP client.",
        }
    if provider_id in {"vnc", "screen_sharing"}:
        target = host or "HOST"
        return {
            "connect_url": f"vnc://{target}",
            "desktop_command": ["open", f"vnc://{target}"] if platform.system() == "Darwin" else [],
            "phone_client_hint": "Open with a VNC/Screen Sharing client.",
        }
    if provider_id == "rustdesk":
        rustdesk_id = str(provider.get("rustdesk_id") or "")
        connect_url = f"rustdesk://{quote(rustdesk_id, safe='')}" if rustdesk_id else ""
        return {
            "connect_url": connect_url,
            "desktop_command": [],
            "phone_client_hint": "Open RustDesk and connect to the target ID.",
            "rustdesk_id": rustdesk_id,
        }
    if provider_id == "sunshine_moonlight":
        target = host or "HOST"
        return {
            "connect_url": f"moonlight://{target}",
            "desktop_command": [],
            "phone_client_hint": "Open Moonlight and connect to the paired Sunshine host.",
        }
    return {"connect_url": "", "desktop_command": [], "phone_client_hint": "Install a compatible remote-control client."}


def target_node(policy: dict[str, Any], target_node_id: str, *, peers: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    local_status = local_node_for_target(policy)
    if target_node_id in {"", "local", str(local_status.get("node_id") or "")}:
        return local_status
    rows = peers if peers is not None else read_peer_rows()
    expected_hive = active_hive_id()
    for peer in rows:
        if str(peer.get("node_id") or "") != target_node_id:
            continue
        if expected_hive and str(peer.get("hive_id") or expected_hive) != expected_hive:
            continue
        return peer
    return None


def local_node_for_target(policy: dict[str, Any]) -> dict[str, Any]:
    identity = read_json(ROOT / str(get_path(policy, ["node", "identity_path"], "reports/hive_node_identity.json")), {})
    status = read_json(ROOT / str(get_path(policy, ["node", "status_path"], "reports/hive_status.json")), {})
    rc = status_report(policy=policy, write_report=False)
    local_host = (rc.get("local", {}).get("candidate_hosts") or ["127.0.0.1"])[0]
    return {
        "node_id": identity.get("node_id") or status.get("node_id") or "local",
        "node_name": identity.get("node_name") or status.get("node_name") or socket.gethostname(),
        "hostname": socket.gethostname(),
        "api_url": status.get("api_url") or f"http://{local_host}:8791",
        "hive_id": active_hive_id(),
        "platform": status.get("platform") or {"system": platform.system(), "machine": platform.machine()},
        "remote_control": rc,
        "is_local": True,
    }


def target_providers(target: dict[str, Any]) -> list[dict[str, Any]]:
    rc = target.get("remote_control") if isinstance(target.get("remote_control"), dict) else {}
    providers = rc.get("providers") if isinstance(rc.get("providers"), list) else []
    return [provider for provider in providers if isinstance(provider, dict)]


def choose_provider(providers: list[dict[str, Any]], requested: str) -> dict[str, Any] | None:
    if requested and requested != "auto":
        for provider in providers:
            if provider.get("id") == requested:
                return provider
        return None
    ready = [provider for provider in providers if provider.get("ready")]
    if ready:
        return sorted(ready, key=provider_rank)[0]
    potential = [provider for provider in providers if provider.get("installed") or provider.get("configured")]
    if potential:
        return sorted(potential, key=provider_rank)[0]
    return providers[0] if providers else None


def provider_rank(provider: dict[str, Any]) -> int:
    order = {"rustdesk": 10, "rdp": 20, "screen_sharing": 25, "vnc": 30, "sunshine_moonlight": 40}
    penalty = 0 if provider.get("ready") else 100
    return order.get(str(provider.get("id") or ""), 90) + penalty


def target_host(target: dict[str, Any]) -> str:
    rc = target.get("remote_control") if isinstance(target.get("remote_control"), dict) else {}
    hosts = get_path(rc, ["local", "candidate_hosts"], [])
    if isinstance(hosts, list):
        for host in hosts:
            cleaned = clean_host(str(host))
            if cleaned and cleaned not in {"127.0.0.1", "::1", "localhost"}:
                return cleaned
    api = str(target.get("api_url") or "")
    parsed = urlparse(api if "://" in api else f"http://{api}")
    return clean_host(parsed.hostname or str(target.get("hostname") or ""))


def public_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": target.get("node_id"),
        "node_name": target.get("node_name"),
        "hostname": target.get("hostname"),
        "api_url": target.get("api_url"),
        "platform": target.get("platform") or {},
        "is_local": bool(target.get("is_local")),
    }


def operator_links(providers: list[dict[str, Any]], host: str) -> list[dict[str, Any]]:
    links = []
    for provider in providers:
        handoff = connection_handoff(provider, host)
        links.append(
            {
                "provider_id": provider.get("id"),
                "label": provider.get("label"),
                "ready": bool(provider.get("ready")),
                "connect_url": handoff.get("connect_url"),
                "phone_client_hint": handoff.get("phone_client_hint"),
            }
        )
    return links


def local_config(policy: dict[str, Any]) -> dict[str, Any]:
    return read_json(ROOT / str(get_path(policy, ["remote_control", "config_path"], "configs/hive_remote_control.local.json")), {})


def read_peer_rows() -> list[dict[str, Any]]:
    peers_report = read_json(REPORTS / "hive_peers.json", {})
    rows = peers_report.get("peers") if isinstance(peers_report, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def candidate_hosts(policy: dict[str, Any]) -> list[str]:
    configured = local_config(policy).get("candidate_hosts")
    rows = [str(item) for item in configured] if isinstance(configured, list) else []
    status = read_json(ROOT / str(get_path(policy, ["node", "status_path"], "reports/hive_status.json")), {})
    if status.get("listen_host"):
        rows.append(str(status.get("listen_host")))
    if status.get("api_url"):
        parsed = urlparse(str(status.get("api_url")))
        if parsed.hostname:
            rows.append(parsed.hostname)
    rows.extend(local_ips())
    return unique_hosts([host for host in rows if clean_host(host)])


def local_ips() -> list[str]:
    rows: list[str] = []
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
        for info in infos:
            ip = info[4][0]
            if ip not in rows:
                rows.append(ip)
    except OSError:
        pass
    best = best_local_ip()
    if best not in rows:
        rows.insert(0, best)
    return rows or ["127.0.0.1"]


def best_local_ip() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def unique_hosts(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        host = clean_host(value)
        if host and host not in seen:
            seen.add(host)
            out.append(host)
    return out


def clean_host(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if "://" in value:
        parsed = urlparse(value)
        return parsed.hostname or ""
    return value.strip("[]").split("/", 1)[0].split(":", 1)[0]


def security_summary(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "trust_boundary": get_path(policy, ["security", "trust_boundary"], "trusted_local_network_private_tunnel_or_authenticated_relay_only"),
        "requires_hive_secret": True,
        "allowed_tiers": get_path(policy, ["remote_control", "allowed_tiers"], ["private", "company"]),
        "public_port_policy": get_path(policy, ["remote_control", "public_port_policy"], "never_expose_raw_desktop_ports"),
        "raw_hive_streaming": "not_implemented_by_design",
    }


def next_actions(providers: list[dict[str, Any]], policy: dict[str, Any]) -> list[str]:
    ready = [provider for provider in providers if provider.get("ready")]
    if ready:
        return [
            "Use theseus control request --target-node NODE to create an audited session handoff.",
            "Use LAN/private tunnel/provider relay for the actual screen/input transport.",
        ]
    system = platform.system()
    if system == "Windows":
        return [
            "Install/configure RustDesk for easiest phone takeover, or enable Windows Remote Desktop for LAN/VPN RDP.",
            "Do not expose TCP 3389 directly to the internet; use WireGuard/private tunnel.",
        ]
    if system == "Darwin":
        return [
            "Enable Screen Sharing or install/configure RustDesk, then refresh Hive status.",
            "Use vnc://HOST on Mac or a VNC/RustDesk client on phone.",
        ]
    return [
        "Install/configure RustDesk, xrdp, VNC, or Sunshine on this Linux node.",
        "Keep remote desktop transports behind LAN, WireGuard/private tunnel, or a hardened self-hosted relay.",
    ]


def active_hive_id() -> str:
    active = hive_profiles.active_profile()
    return str(active.get("hive_id") or os.environ.get("THESEUS_HIVE_ID") or "local") if active else os.environ.get("THESEUS_HIVE_ID", "local")


def active_hive_tier() -> str:
    active = hive_profiles.active_profile()
    return str(active.get("tier") or os.environ.get("THESEUS_HIVE_TIER") or "private") if active else os.environ.get("THESEUS_HIVE_TIER", "private")


def best_provider_id(providers: list[dict[str, Any]]) -> str:
    provider = choose_provider(providers, "auto")
    return str(provider.get("id") or "") if provider else ""


def windows_rdp_enabled() -> bool:
    try:
        result = subprocess.run(
            ["reg", "query", r"HKLM\SYSTEM\CurrentControlSet\Control\Terminal Server", "/v", "fDenyTSConnections"],
            text=True,
            capture_output=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return "0x0" in result.stdout.lower()


def windows_service_running(name: str) -> bool:
    try:
        result = subprocess.run(["sc", "query", name], text=True, capture_output=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "RUNNING" in result.stdout


def mac_service_loaded(label: str) -> bool:
    try:
        result = subprocess.run(["launchctl", "print", f"system/{label}"], text=True, capture_output=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def first_existing(values: list[str | None]) -> str:
    for value in values:
        if not value:
            continue
        path = Path(value)
        if path.exists():
            return str(path)
        if shutil.which(value):
            return str(shutil.which(value))
    return ""


def open_url(url: str) -> None:
    if platform.system() == "Windows":
        os.startfile(url)  # type: ignore[attr-defined]  # noqa: S606 - operator-requested URL handoff.
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", url], cwd=ROOT)
    else:
        subprocess.Popen(["xdg-open", url], cwd=ROOT)


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
