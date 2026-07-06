"""No-cost remote access planning for Project Theseus Hive.

This module keeps the default remote path free and self-hosted. It does not
make paid mesh services a dependency. The preferred modes are:

1. same LAN or phone hotspot for a small travel Hive;
2. self-hosted WireGuard/private tunnel for home, workshop, and travel;
3. the built-in authenticated relay, exposed only through a private tunnel or
   an HTTPS reverse proxy;
4. raw router forwarding only as an explicit last resort, and never for the
   node API port.
"""

from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import platform
import shutil
import socket
import subprocess
import sys
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
    parser = argparse.ArgumentParser(description="Plan and configure free Project Theseus Hive remote access.")
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Show current no-cost remote access posture.")
    status.add_argument("--out", default="reports/hive_remote_access_status.json")

    configure = sub.add_parser("configure-relay", help="Set the active Hive relay URL and write a remote invite.")
    configure.add_argument("--relay-url", default="")
    configure.add_argument("--out", default="reports/hive_invite_remote_private.json")
    configure.add_argument("--start", action="store_true")
    configure.add_argument("--restart", action="store_true")

    guide = sub.add_parser("wireguard-guide", help="Write a free self-hosted WireGuard setup guide.")
    guide.add_argument("--out", default="reports/hive_wireguard_free_setup.md")

    mobile = sub.add_parser("mobile-profile", help="Write an iPhone roaming connection profile for the active Hive.")
    mobile.add_argument("--out", default="reports/hive_mobile_roaming_profile.json")
    mobile.add_argument("--no-token", action="store_true")

    args = parser.parse_args()
    policy = read_json(POLICY_PATH, {})
    if args.command in {None, "status"}:
        report = status_report(policy=policy, write_report=False)
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    if args.command == "configure-relay":
        report = configure_relay_url(args.relay_url, out=args.out, policy=policy)
        if report.get("ok") and args.start:
            report["service_report"] = start_relay_service(restart=bool(args.restart))
        write_json(REPORTS / "hive_remote_access_configure.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    if args.command == "wireguard-guide":
        report = write_wireguard_guide(out=args.out, policy=policy)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    if args.command == "mobile-profile":
        report = write_mobile_roaming_profile(out=args.out, include_token=not bool(args.no_token), policy=policy)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    parser.print_help()
    return 2


def status_report(*, policy: dict[str, Any] | None = None, write_report: bool = True) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    active = hive_profiles.active_profile()
    relay_port = int(get_path(policy, ["relay", "server_port"], 8793))
    relay_url = str(active.get("relay_url") or "")
    relay = analyze_url(relay_url, default_port=relay_port)
    local_hive_live = port_open("127.0.0.1", int(get_path(policy, ["node", "http_port"], 8791)))
    local_relay_live = port_open("127.0.0.1", relay_port)
    report = {
        "ok": True,
        "policy": "project_theseus_hive_remote_access_status_v0",
        "created_utc": now(),
        "cost_policy": {
            "paid_dependency_required": False,
            "default_paid_mesh_vendor": "",
            "preferred_free_modes": [
                "same_lan_or_phone_hotspot",
                "self_hosted_wireguard_private_tunnel",
                "built_in_authenticated_relay_over_private_tunnel_or_https",
            ],
        },
        "active_hive": hive_profiles.public_profile(active) if active else {},
        "local": {
            "system": platform.system(),
            "ips": local_ips(),
            "hive_api_live": local_hive_live,
            "relay_live": local_relay_live,
            "relay_port": relay_port,
            "wireguard_tools_installed": bool(shutil.which("wg")),
        },
        "relay": relay,
        "mobile_roaming": mobile_roaming_summary(active, policy=policy, include_token=False),
        "router_rules": router_rules(policy, relay),
        "travel_hive": {
            "works_without_home_internet": True,
            "description": "Laptop(s) plus phone can form a local Hive on the same Wi-Fi/hotspot, then reconnect to home/workshop when the tunnel or relay is available.",
        },
        "latency_policy": {
            "interactive_tasks": "prefer local, same LAN, or private tunnel nodes",
            "training_chunks": "prefer CUDA/MLX accelerators, but split into checkpointed chunks before sending over high-latency links",
            "relay_tasks": "treat relay-polled work as higher latency and suitable for bounded asynchronous tasks",
        },
        "next_actions": next_actions(active, relay, local_relay_live),
    }
    if write_report:
        write_json(REPORTS / "hive_remote_access_status.json", report)
    return report


def configure_relay_url(relay_url: str, *, out: str = "reports/hive_invite_remote_private.json", policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    active = hive_profiles.active_profile()
    if not active:
        return {
            "ok": False,
            "error": "no_active_hive",
            "message": "Create or join a private Hive before configuring remote access.",
        }
    if active.get("tier") == "public":
        return {"ok": False, "error": "public_hive_disabled"}
    relay_port = int(get_path(policy, ["relay", "server_port"], 8793))
    relay_url = normalize_relay_url(relay_url or default_relay_url(policy), relay_port)
    relay = analyze_url(relay_url, default_port=relay_port)
    store = hive_profiles.load_profiles_private()
    updated: dict[str, Any] = {}
    for profile in store.get("profiles") or []:
        if profile.get("profile_id") == active.get("profile_id"):
            profile["relay_url"] = relay_url
            profile["mode"] = "relay"
            profile["last_used_utc"] = now()
            updated = profile
            break
    if not updated:
        return {"ok": False, "error": "active_profile_not_found"}
    store["active_profile_id"] = updated.get("profile_id")
    hive_profiles.write_json(hive_profiles.PROFILES_PATH, store)
    hive_profiles.write_join_config(updated)
    invite = invite_from_profile(updated, include_token=True, include_token_in_phone_url=False)
    invite.update(mobile_roaming_profile(updated, policy=policy, include_token=True))
    invite_path = ROOT / out
    write_json(invite_path, invite)
    report = {
        "ok": True,
        "policy": "project_theseus_hive_remote_relay_configure_v0",
        "created_utc": now(),
        "profile": hive_profiles.public_profile(updated),
        "relay": relay,
        "invite_path": rel(invite_path),
        "phone_url": phone_join_url(updated, include_token=False),
        "ios_app_url_written_to_invite": True,
        "security_notes": [
            "No paid mesh service is required.",
            "Forward the relay port only if you cannot use WireGuard/private tunnel.",
            "Do not forward the Hive node API port 8791 directly to the public internet.",
            "Use HTTPS for any relay URL that is reachable from the public internet.",
        ],
        "next_commands": [
            "python3 scripts/theseus_cli.py start --relay --restart",
            f"python3 scripts/hive_remote_access.py wireguard-guide --out reports/hive_wireguard_free_setup.md",
        ],
    }
    write_json(REPORTS / "hive_remote_access_configure.json", report)
    return report


def write_mobile_roaming_profile(
    *,
    out: str = "reports/hive_mobile_roaming_profile.json",
    include_token: bool,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    active = hive_profiles.active_profile()
    if not active:
        return {"ok": False, "error": "no_active_hive"}
    payload = mobile_roaming_profile(active, policy=policy, include_token=include_token)
    path = ROOT / out
    write_json(path, payload)
    report = {
        "ok": True,
        "policy": "project_theseus_mobile_roaming_profile_write_v0",
        "created_utc": now(),
        "path": rel(path),
        "hive_id": active.get("hive_id"),
        "endpoint_count": len(payload.get("roaming", {}).get("endpoints", [])),
        "token_included": bool(include_token),
        "ios_app_url_written_to_profile": bool(include_token),
        "security_notes": [
            "The token-bearing profile and iPhone app URL are passwords. Share them only with your own devices.",
            "The app will try local LAN, private tunnel, and relay endpoints automatically.",
        ],
    }
    write_json(REPORTS / "hive_mobile_roaming_profile_report.json", report)
    return report


def write_wireguard_guide(*, out: str = "reports/hive_wireguard_free_setup.md", policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    active = hive_profiles.active_profile()
    relay_port = int(get_path(policy, ["relay", "server_port"], 8793))
    node_port = int(get_path(policy, ["node", "http_port"], 8791))
    coordinator_vpn_ip = "10.87.0.1"
    relay_url = f"http://{coordinator_vpn_ip}:{relay_port}"
    node_url = f"http://{coordinator_vpn_ip}:{node_port}"
    path = ROOT / out
    text = f"""# Free Remote Access For Project Theseus Hive

This path uses self-hosted WireGuard plus the built-in authenticated Hive relay.
It does not require Tailscale, ZeroTier, a paid VPN, or a paid cloud machine.

## Recommended Topology

- Home coordinator: always-on Mac, Windows PC, Linux box, router, or NAS.
- Workshop nodes: join the same WireGuard network when away from home LAN.
- Travel Hive: laptop(s) plus iPhone on the same Wi-Fi/hotspot; reconnect to
  home/workshop over WireGuard when available.

## Ports

- Forward only WireGuard UDP, commonly `51820/udp`, to the WireGuard endpoint.
- Do not forward Hive node API `{node_port}/tcp` directly to the internet.
- If you expose the built-in relay publicly, put HTTPS authentication in front
  of `{relay_port}/tcp`; direct public HTTP is not recommended.

## Suggested WireGuard Address Plan

- Home coordinator: `{coordinator_vpn_ip}/24`
- Main Windows node: `10.87.0.2/32`
- Workshop coordinator: `10.87.0.20/32`
- Travel laptop: `10.87.0.40/32`
- iPhone: `10.87.0.50/32`

After the tunnel is up, configure Theseus remote relay mode:

```bash
python3 scripts/hive_remote_access.py configure-relay --relay-url {relay_url} --start
```

Use these URLs from iPhone or travel laptops while connected to WireGuard:

- Native iPhone app node URL: `{node_url}`
- Relay operator page: `{relay_url}/mobile`

## Install Notes

WireGuard is free/open-source and cross-platform. Install the official app on
iPhone, macOS, Windows, or Linux, then import the peer config generated by your
WireGuard endpoint/router. If your router already supports WireGuard, use that
as the endpoint so the home coordinator does not need to run the VPN service.

## Active Hive

- Hive name: {active.get("name", "") if active else ""}
- Hive id: {active.get("hive_id", "") if active else ""}
- Current relay URL: {active.get("relay_url", "") if active else ""}

Keep the Hive invite token out of screenshots, logs, and public URLs.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    report = {
        "ok": True,
        "policy": "project_theseus_wireguard_free_setup_guide_v0",
        "created_utc": now(),
        "path": rel(path),
        "wireguard_tools_installed": bool(shutil.which("wg")),
        "relay_url_after_tunnel": relay_url,
        "iphone_node_url_after_tunnel": node_url,
    }
    write_json(REPORTS / "hive_wireguard_free_setup_report.json", report)
    return report


def invite_from_profile(profile: dict[str, Any], *, include_token: bool, include_token_in_phone_url: bool) -> dict[str, Any]:
    expires = datetime.now(timezone.utc) + timedelta(days=30)
    policy = read_json(POLICY_PATH, {})
    roaming = mobile_roaming_profile(profile, policy=policy, include_token=include_token)
    invite: dict[str, Any] = {
        "policy": "project_theseus_hive_invite_v0",
        "created_utc": now(),
        "expires_utc": expires.isoformat(),
        "hive_id": profile.get("hive_id"),
        "hive_name": profile.get("name"),
        "tier": profile.get("tier"),
        "relay_url": profile.get("relay_url"),
        "coordinator_url": roaming.get("coordinator_url", ""),
        "coordinator_urls": roaming.get("coordinator_urls", []),
        "node_urls": roaming.get("node_urls", []),
        "relay_urls": roaming.get("relay_urls", []),
        "operator_urls": roaming.get("operator_urls", []),
        "roaming": roaming.get("roaming", {}),
        "ios_app_url": roaming.get("ios_app_url", "") if include_token else "",
        "allowed_task_scope": allowed_task_scope(str(profile.get("tier") or "private")),
        "install": {
            "windows": ["Install Project Theseus, choose Join Hive, and import this invite."],
            "mac_linux": ["Install Project Theseus, choose Join Hive, and import this invite."],
            "phone": [phone_join_url(profile, include_token=include_token_in_phone_url) or "Open the relay mobile URL and enter the invite token."],
        },
        "security_notes": [
            "Treat this invite like a password.",
            "Remote workers can only run registered bounded Hive task kinds.",
            "Prefer WireGuard/private tunnel. Use HTTPS before exposing any relay URL publicly.",
        ],
    }
    if include_token:
        invite["join_token"] = profile.get("join_token")
    return invite


def mobile_roaming_profile(profile: dict[str, Any], *, policy: dict[str, Any], include_token: bool) -> dict[str, Any]:
    summary = mobile_roaming_summary(profile, policy=policy, include_token=include_token)
    endpoint_urls = unique_urls(list(summary.get("coordinator_urls") or []) + list(summary.get("node_urls") or []) + list(summary.get("relay_urls") or []))
    payload = {
        "policy": "project_theseus_hive_mobile_roaming_profile_v1",
        "created_utc": now(),
        "hive_id": profile.get("hive_id"),
        "hive_name": profile.get("name"),
        "tier": profile.get("tier"),
        "coordinator_url": summary.get("coordinator_url", ""),
        "coordinator_urls": summary.get("coordinator_urls", []),
        "node_urls": summary.get("node_urls", []),
        "relay_url": profile.get("relay_url"),
        "relay_urls": summary.get("relay_urls", []),
        "operator_urls": summary.get("operator_urls", []),
        "bonjour": summary.get("roaming", {}).get("bonjour", {}),
        "handover": summary.get("roaming", {}).get("handover", {}),
        "roaming": summary.get("roaming", {}),
        "update_catalog_url": endpoint_url(endpoint_urls[0], "api/hive/update-catalog") if endpoint_urls else "",
        "update_catalog_urls": [endpoint_url(url, "api/hive/update-catalog") for url in endpoint_urls],
        "installer_artifacts_urls": [endpoint_url(url, "api/hive/installer-artifacts") for url in endpoint_urls],
        "device_policy": {
            "token_model": "per_user_operator_tokens_preferred_machine_join_token_allowed_for_owner_devices",
            "revocation": "use theseus hive revoke-user USER_ID for phones; rotate the machine invite if a join token is leaked",
            "no_codex_required": True,
            "paid_dependency_required": False,
        },
    }
    if include_token:
        payload["join_token"] = profile.get("join_token")
        payload["operator_token"] = profile.get("join_token")
        payload["ios_app_url"] = mobile_app_join_url(profile, policy=policy, include_token=True)
    return payload


def mobile_roaming_summary(profile: dict[str, Any], *, policy: dict[str, Any], include_token: bool) -> dict[str, Any]:
    if not profile:
        return {
            "configured": False,
            "coordinator_url": "",
            "coordinator_urls": [],
            "node_urls": [],
            "relay_urls": [],
            "operator_urls": [],
            "roaming": {"strategy": "configure_hive_first", "endpoints": []},
        }
    node_port = int(get_path(policy, ["node", "http_port"], 8791))
    status = read_json(REPORTS / "hive_status.json", {})
    raw_node_urls = [
        str(profile.get("coordinator_url") or ""),
        str(status.get("api_url") or ""),
        f"http://{best_local_ip()}:{node_port}",
    ]
    bonjour_urls = local_bonjour_endpoint_urls(status, node_port=node_port)
    raw_node_urls.extend(bonjour_urls)
    for url in profile.get("coordinator_urls") or []:
        raw_node_urls.append(str(url))
    for url in profile.get("node_urls") or []:
        raw_node_urls.append(str(url))
    raw_node_urls.extend(known_peer_endpoint_urls(kinds={"node"}))
    for ip in local_ips():
        raw_node_urls.append(f"http://{ip}:{node_port}")
    all_node_urls = unique_urls(raw_node_urls)
    node_urls = [url for url in all_node_urls if analyze_url(url, default_port=node_port).get("scope") != "loopback"] or all_node_urls
    raw_relay_urls = [str(profile.get("relay_url") or "")]
    for url in profile.get("relay_urls") or []:
        raw_relay_urls.append(str(url))
    raw_relay_urls.extend(known_peer_endpoint_urls(kinds={"relay"}))
    relay_urls = unique_urls(raw_relay_urls)
    endpoints: list[dict[str, Any]] = []
    for url in node_urls:
        analyzed = analyze_url(url, default_port=node_port)
        endpoints.append(
            {
                "kind": "node",
                "url": url,
                "operator_url": operator_url(url),
                "scope": analyzed.get("scope"),
                "transport": endpoint_transport("node", analyzed),
                "priority": endpoint_priority("node", analyzed),
                "works_when": works_when("node", analyzed),
                "health": endpoint_health(url, default_port=node_port),
                "security": endpoint_security("node", analyzed),
                "source": endpoint_source("node", url, bonjour_urls=bonjour_urls),
            }
        )
    relay_port = int(get_path(policy, ["relay", "server_port"], 8793))
    for url in relay_urls:
        analyzed = analyze_url(url, default_port=relay_port)
        endpoints.append(
            {
                "kind": "relay",
                "url": url,
                "operator_url": operator_url(url),
                "scope": analyzed.get("scope"),
                "transport": endpoint_transport("relay", analyzed),
                "priority": endpoint_priority("relay", analyzed),
                "works_when": works_when("relay", analyzed),
                "health": endpoint_health(url, default_port=relay_port),
                "security": endpoint_security("relay", analyzed),
                "source": endpoint_source("relay", url, bonjour_urls=bonjour_urls),
            }
        )
    endpoints = sorted(endpoints, key=lambda item: int(item.get("priority") or 999))
    coordinator_urls = unique_urls([item.get("url", "") for item in endpoints])
    operator_urls = unique_nonempty_strings([str(item.get("operator_url", "")) for item in endpoints])
    summary = {
        "configured": bool(coordinator_urls),
        "coordinator_url": coordinator_urls[0] if coordinator_urls else "",
        "coordinator_urls": coordinator_urls,
        "node_urls": node_urls,
        "relay_urls": relay_urls,
        "operator_urls": operator_urls,
        "roaming": {
            "strategy": "iphone_try_last_good_then_lan_then_private_tunnel_then_https_relay",
            "endpoint_selection": {
                "order": ["last_good", "bonjour_local_hostname", "same_lan_private_ip", "private_tunnel", "https_relay"],
                "failover_timeout_seconds": 2,
                "persist_last_good_endpoint": True,
                "parallel_probe": True,
                "demote_after_failures": 2,
                "background_reprobe_seconds": 30,
            },
            "bonjour": bonjour_roaming_summary(policy, bonjour_urls=bonjour_urls, node_port=node_port),
            "handover": roaming_handover_policy(),
            "endpoints": endpoints,
            "requires_public_path_for_cellular": not any(analyze_url(url, default_port=relay_port).get("internet_safe") for url in relay_urls),
            "no_paid_dependency_required": True,
            "security": {
                "raw_public_node_api_allowed": False,
                "raw_public_8791_forbidden": True,
                "public_paths_require_https": True,
                "per_user_tokens": "supported by theseus hive add-user and revoke-user",
                "device_revocation_command": "theseus hive revoke-user <user_id>",
            },
            "update_catalog_urls": [endpoint_url(url, "api/hive/update-catalog") for url in coordinator_urls],
            "installer_artifacts_urls": [endpoint_url(url, "api/hive/installer-artifacts") for url in coordinator_urls],
            "notes": [
                "The iPhone app stores all endpoint candidates and picks the first reachable one.",
                "Cellular/offsite access still needs one reachable path: self-hosted WireGuard, an HTTPS relay, or a router rule to the relay port.",
                "Never expose 8791 directly to the public internet.",
            ],
        },
    }
    if include_token:
        summary["ios_app_url"] = mobile_app_join_url(profile, policy=policy, include_token=True)
    return summary


def known_peer_endpoint_urls(*, kinds: set[str]) -> list[str]:
    urls: list[str] = []
    reports = [
        read_json(REPORTS / "hive_peers.json", {}),
        read_json(REPORTS / "hive_peer_registry.json", {}),
    ]
    for report in reports:
        if not isinstance(report, dict):
            continue
        rows: list[dict[str, Any]] = []
        for key in ["local_node", "peers", "known_peers", "stale_peers"]:
            value = report.get(key)
            if isinstance(value, dict):
                rows.append(value)
            elif isinstance(value, list):
                rows.extend(row for row in value if isinstance(row, dict))
        for row in rows:
            if "node" in kinds and row.get("api_url"):
                urls.append(str(row.get("api_url") or ""))
            if "relay" in kinds and row.get("relay_url"):
                urls.append(str(row.get("relay_url") or ""))
    return unique_urls(urls)


def endpoint_url(base_url: str, path: str) -> str:
    normalized = normalize_endpoint_url(base_url)
    if not normalized:
        return ""
    return normalized.rstrip("/") + "/" + path.strip("/")


def mobile_app_join_url(profile: dict[str, Any], *, policy: dict[str, Any], include_token: bool) -> str:
    payload = mobile_roaming_profile_payload_for_url(profile, policy=policy, include_token=include_token)
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"theseushive://join?profile={encoded}"


def mobile_roaming_profile_payload_for_url(profile: dict[str, Any], *, policy: dict[str, Any], include_token: bool) -> dict[str, Any]:
    summary = mobile_roaming_summary_without_app_url(profile, policy=policy)
    compact_endpoints = [
        {"kind": endpoint.get("kind"), "url": endpoint.get("url"), "transport": endpoint.get("transport"), "priority": endpoint.get("priority")}
        for endpoint in summary.get("roaming", {}).get("endpoints", [])
        if endpoint.get("url")
    ]
    payload = {
        "policy": "project_theseus_hive_mobile_roaming_profile_v0",
        "hive_id": profile.get("hive_id"),
        "hive_name": profile.get("name"),
        "tier": profile.get("tier"),
        "coordinator_url": summary.get("coordinator_url", ""),
        "coordinator_urls": summary.get("coordinator_urls", []),
        "node_urls": summary.get("node_urls", []),
        "relay_url": profile.get("relay_url"),
        "relay_urls": summary.get("relay_urls", []),
        "operator_urls": summary.get("operator_urls", []),
        "roaming": {
            "strategy": "iphone_try_last_good_then_lan_then_private_tunnel_then_https_relay",
            "endpoint_selection": {
                "order": ["last_good", "bonjour_local_hostname", "same_lan_private_ip", "private_tunnel", "https_relay"],
                "failover_timeout_seconds": 2,
                "parallel_probe": True,
            },
            "bonjour": summary.get("roaming", {}).get("bonjour", {}),
            "handover": summary.get("roaming", {}).get("handover", {}),
            "endpoints": compact_endpoints,
        },
    }
    if include_token:
        payload["join_token"] = profile.get("join_token")
    return payload


def mobile_roaming_summary_without_app_url(profile: dict[str, Any], *, policy: dict[str, Any]) -> dict[str, Any]:
    return mobile_roaming_summary(profile, policy=policy, include_token=False)


def local_bonjour_endpoint_urls(status: dict[str, Any], *, node_port: int) -> list[str]:
    hosts: list[str] = []
    for value in [status.get("hostname"), socket.gethostname(), socket.getfqdn()]:
        host = str(value or "").strip().strip(".")
        lowered = host.lower()
        if not host or lowered in {"localhost", "ip6-localhost"} or lowered.endswith(".arpa"):
            continue
        hosts.append(host)
        if "." not in host:
            hosts.append(f"{host}.local")
    return unique_urls([f"http://{host}:{node_port}" for host in hosts])


def bonjour_roaming_summary(policy: dict[str, Any], *, bonjour_urls: list[str], node_port: int) -> dict[str, Any]:
    return {
        "enabled": bool(get_path(policy, ["discovery", "bonjour_enabled"], True)),
        "service_type": str(get_path(policy, ["discovery", "bonjour_service_type"], "_theseus-hive._tcp")),
        "domain": str(get_path(policy, ["discovery", "bonjour_domain"], "local.")),
        "node_port": node_port,
        "candidate_urls": bonjour_urls,
        "use_when": "same_lan_or_phone_hotspot_macos_ios",
        "signed_txt_records": True,
        "notes": "Mac nodes advertise signed Bonjour records; native clients should prefer .local endpoints when nearby and fall back without user prompts.",
    }


def roaming_handover_policy() -> dict[str, Any]:
    return {
        "last_good_first": True,
        "parallel_probe": True,
        "probe_timeout_seconds": 2,
        "relay_probe_timeout_seconds": 5,
        "demote_endpoint_after_failures": 2,
        "background_reprobe_seconds": 30,
        "promote_lower_latency_endpoint": True,
        "latency_buckets": {
            "interactive_preferred_ms": 120,
            "same_lan_expected_ms": 30,
            "private_tunnel_expected_ms": 150,
            "relay_expected_ms": 500,
        },
        "task_routing": {
            "chat_and_operator": "prefer last_good same-LAN or private tunnel endpoint",
            "training_chunks": "prefer accelerator capability and lease length over latency after coordination is complete",
            "storage_preview": "prefer same-LAN node for thumbnails; use relay/tunnel for offsite browsing",
        },
    }


def unique_urls(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = normalize_endpoint_url(str(value or ""))
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def unique_nonempty_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = value.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_endpoint_url(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.hostname:
        return ""
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}".rstrip("/")


def operator_url(base_url: str) -> str:
    normalized = normalize_endpoint_url(base_url)
    return normalized.rstrip("/") + "/mobile" if normalized else ""


def endpoint_source(kind: str, url: str, *, bonjour_urls: list[str]) -> str:
    normalized = normalize_endpoint_url(url)
    if kind == "node" and normalized in set(bonjour_urls):
        return "bonjour_local_hostname"
    analyzed = analyze_url(url, default_port=8791 if kind == "node" else 8793)
    scope = analyzed.get("scope")
    if kind == "relay":
        return "relay"
    if scope == "private_ip":
        return "same_lan_private_ip"
    if scope == "private_dns":
        return "private_dns"
    if scope == "loopback":
        return "loopback"
    return "configured_or_learned"


def endpoint_priority(kind: str, analyzed: dict[str, Any]) -> int:
    scope = analyzed.get("scope")
    host = str(analyzed.get("host") or "").lower()
    if kind == "node" and scope == "private_dns" and host.endswith(".local"):
        return 5
    if kind == "node" and scope in {"private_ip", "private_dns"}:
        return 10
    if kind == "relay" and scope in {"private_ip", "private_dns"}:
        return 20
    if kind == "relay" and analyzed.get("internet_safe"):
        return 30
    if kind == "node" and scope == "loopback":
        return 90
    return 80


def works_when(kind: str, analyzed: dict[str, Any]) -> str:
    host = str(analyzed.get("host") or "").lower()
    if kind == "node" and analyzed.get("scope") == "private_dns" and host.endswith(".local"):
        return "same_lan_or_phone_hotspot_with_bonjour"
    if analyzed.get("internet_safe"):
        return "cellular_or_any_wifi_with_https"
    if analyzed.get("scope") in {"private_ip", "private_dns"}:
        return "same_lan_hotspot_or_private_tunnel"
    if kind == "relay":
        return "reachable_authenticated_relay"
    return "reachable_private_node"


def endpoint_transport(kind: str, analyzed: dict[str, Any]) -> str:
    host = str(analyzed.get("host") or "").lower()
    if kind == "node" and analyzed.get("scope") == "private_dns" and host.endswith(".local"):
        return "macos_bonjour_mdns"
    if analyzed.get("internet_safe"):
        return "https_public_relay"
    if analyzed.get("scope") == "loopback":
        return "local_only"
    if analyzed.get("scope") in {"private_ip", "private_dns"}:
        return "same_lan_hotspot_or_private_tunnel"
    if kind == "relay":
        return "authenticated_relay_requires_tls_before_public_use"
    return "private_node_api"


def endpoint_security(kind: str, analyzed: dict[str, Any]) -> dict[str, Any]:
    warnings = list(analyzed.get("warnings") or [])
    if kind == "node" and analyzed.get("scope") in {"public_ip", "public_dns", "unknown_dns"}:
        warnings.append("node_api_must_not_be_public")
    return {
        "internet_safe": bool(analyzed.get("internet_safe")),
        "private_tunnel_safe": bool(analyzed.get("private_tunnel_safe")),
        "raw_public_node_api_allowed": False,
        "warnings": sorted(set(str(item) for item in warnings if item)),
    }


def endpoint_health(url: str, *, default_port: int) -> dict[str, Any]:
    analyzed = analyze_url(url, default_port=default_port)
    host = str(analyzed.get("host") or "")
    port = int(analyzed.get("port") or default_port)
    if not host:
        return {"checked": False, "reachable": False, "error": "missing_host"}
    return {
        "checked": True,
        "reachable": port_open(host, port, timeout=0.3),
        "host": host,
        "port": port,
        "scope": analyzed.get("scope"),
    }


def phone_join_url(profile: dict[str, Any], *, include_token: bool) -> str:
    relay = str(profile.get("relay_url") or "").rstrip("/")
    if not relay:
        return ""
    hive_id = quote(str(profile.get("hive_id") or ""), safe="")
    url = f"{relay}/m?h={hive_id}"
    if include_token:
        token = quote(str(profile.get("join_token") or ""), safe="")
        url += f"&t={token}"
    return url


def analyze_url(url: str, *, default_port: int) -> dict[str, Any]:
    if not url:
        return {
            "configured": False,
            "url": "",
            "scope": "not_configured",
            "internet_safe": False,
            "warnings": ["No relay URL configured; this Hive is same-network only unless a private tunnel is active."],
        }
    parsed = urlparse(url)
    host = parsed.hostname or ""
    scheme = parsed.scheme or ""
    scope = host_scope(host)
    publicish = scope in {"public_ip", "public_dns", "unknown_dns"}
    warnings: list[str] = []
    if publicish and scheme != "https":
        warnings.append("Public or unknown relay URL is not HTTPS. Use a TLS reverse proxy before internet exposure.")
    if scope in {"loopback", "private_ip", "private_dns"}:
        warnings.append("Relay URL is private-scope; it works over LAN, hotspot, or WireGuard/private tunnel, not the open internet.")
    if parsed.port == 8791:
        warnings.append("Relay URL points at the Hive node API port. Use the relay port instead, usually 8793.")
    return {
        "configured": True,
        "url": url,
        "scheme": scheme,
        "host": host,
        "port": parsed.port or (443 if scheme == "https" else default_port),
        "scope": scope,
        "internet_safe": bool(publicish and scheme == "https"),
        "private_tunnel_safe": scope in {"loopback", "private_ip", "private_dns"},
        "warnings": warnings,
    }


def normalize_relay_url(value: str, default_port: int) -> str:
    value = value.strip()
    if not value:
        return default_relay_url(read_json(POLICY_PATH, {}))
    if "://" not in value:
        host = value.split("/", 1)[0].split(":", 1)[0].strip("[]")
        scheme = "http" if host_scope(host) in {"loopback", "private_ip", "private_dns"} else "https"
        value = f"{scheme}://{value}"
    parsed = urlparse(value)
    if parsed.scheme == "http" and parsed.port is None and parsed.hostname:
        return f"{value.rstrip('/')}:{default_port}"
    return value.rstrip("/")


def router_rules(policy: dict[str, Any], relay: dict[str, Any]) -> dict[str, Any]:
    relay_port = int(get_path(policy, ["relay", "server_port"], 8793))
    return {
        "preferred": "No router forwarding if using WireGuard/private tunnel or same LAN.",
        "wireguard_optional": "If this home router is the WireGuard endpoint, forward UDP 51820 to the router or WireGuard host.",
        "relay_last_resort": f"If not using WireGuard and you accept the risk, forward TCP {relay_port} to this coordinator's TCP {relay_port} behind HTTPS/auth.",
        "never_forward": [
            f"TCP {get_path(policy, ['node', 'http_port'], 8791)} Hive node API",
            f"TCP {get_path(policy, ['node', 'dashboard_port'], 8787)} local dashboard",
        ],
        "relay_url_scope": relay.get("scope"),
    }


def next_actions(active: dict[str, Any], relay: dict[str, Any], local_relay_live: bool) -> list[str]:
    if not active:
        return ["Create a private Hive first, then configure remote relay mode."]
    if not relay.get("configured"):
        return ["Run: python3 scripts/hive_remote_access.py configure-relay --start"]
    actions: list[str] = []
    if not local_relay_live:
        actions.append("Start the relay: python3 scripts/theseus_cli.py start --relay")
    if relay.get("private_tunnel_safe"):
        actions.append("Use WireGuard/private tunnel or the same LAN/hotspot before opening the relay URL from phone/travel nodes.")
    if relay.get("internet_safe"):
        actions.append("Relay URL is HTTPS/public-scope; use invite token authentication and keep remote task kinds bounded.")
    if relay.get("warnings"):
        actions.extend(str(item) for item in relay.get("warnings") or [])
    return actions or ["Remote access posture looks ready."]


def start_relay_service(*, restart: bool) -> dict[str, Any]:
    command = [sys.executable, "scripts/theseus_cli.py", "start", "--relay"]
    if restart:
        command.append("--restart")
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    payload = parse_json(result.stdout.strip(), {})
    if isinstance(payload, dict) and payload:
        payload.setdefault("ok", result.returncode == 0)
        return payload
    return {"ok": result.returncode == 0, "returncode": result.returncode, "stderr_tail": result.stderr[-2000:]}


def default_relay_url(policy: dict[str, Any]) -> str:
    port = int(get_path(policy, ["relay", "server_port"], 8793))
    return f"http://{best_local_ip()}:{port}"


def host_scope(host: str) -> str:
    if not host:
        return "unknown"
    lowered = host.lower()
    if lowered in {"localhost", "ip6-localhost"}:
        return "loopback"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if lowered.endswith(".local") or lowered.endswith(".lan") or lowered.endswith(".home"):
            return "private_dns"
        return "unknown_dns"
    if ip.is_loopback:
        return "loopback"
    if ip.is_private or ip.is_link_local:
        return "private_ip"
    return "public_ip"


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


def allowed_task_scope(tier: str) -> list[str]:
    policy = read_json(POLICY_PATH, {})
    return list(get_path(policy, ["relay", "allowed_remote_task_kinds_by_tier", tier], []))


def port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def parse_json(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


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
