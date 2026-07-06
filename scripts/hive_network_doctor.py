"""Hive network doctor for real multi-node and roaming access checks.

This is intentionally non-mutating. It checks the local node, configured
coordinator/relay URLs, stale peers, and public exposure posture, then writes
operator-facing fixes that do not reveal Hive tokens.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
POLICY_PATH = CONFIGS / "hive_policy.json"
DEFAULT_OUT = REPORTS / "hive_network_doctor.json"
DEFAULT_MARKDOWN = REPORTS / "hive_network_doctor.md"

sys.path.insert(0, str(ROOT / "scripts"))
import hive_profiles  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Project Theseus Hive network reachability and roaming posture.")
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--retries", type=int, default=2, help="Maximum probe attempts per endpoint. Use at least 2 to detect retry-recovered flapping.")
    parser.add_argument("--retry-delay", type=float, default=0.5, help="Seconds to wait between failed endpoint probe attempts.")
    parser.add_argument("--coordinator-url", action="append", default=[])
    parser.add_argument("--peer-url", action="append", default=[])
    args = parser.parse_args()

    policy = read_json(resolve(args.policy), {})
    report = doctor_report(
        policy=policy,
        timeout=float(args.timeout),
        retries=int(args.retries),
        retry_delay=float(args.retry_delay),
        coordinator_urls=[str(item) for item in args.coordinator_url or []],
        peer_urls=[str(item) for item in args.peer_url or []],
        write_report=True,
        out=resolve(args.out),
        markdown_out=resolve(args.markdown_out),
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("state") in {"GREEN", "YELLOW"} else 2


def doctor_report(
    *,
    policy: dict[str, Any] | None = None,
    timeout: float = 8.0,
    retries: int = 2,
    retry_delay: float = 0.5,
    coordinator_urls: list[str] | None = None,
    peer_urls: list[str] | None = None,
    write_report: bool = True,
    out: Path | None = None,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    node_port = int(get_path(policy, ["node", "http_port"], 8791))
    relay_port = int(get_path(policy, ["relay", "server_port"], 8793))
    secret = hive_secret(policy)
    join = active_join_config(policy)
    active_profile = hive_profiles.public_profile(hive_profiles.active_profile()) if hive_profiles.active_profile() else {}
    status_report = read_json(ROOT / str(get_path(policy, ["node", "status_path"], "reports/hive_status.json")), {})
    peers_report = latest_peers_report(policy, timeout=timeout, secret=secret)
    endpoint_specs = collect_endpoint_specs(
        policy,
        node_port=node_port,
        relay_port=relay_port,
        join=join,
        active_profile=active_profile,
        status_report=status_report,
        peers_report=peers_report,
        coordinator_urls=coordinator_urls or [],
        peer_urls=peer_urls or [],
    )
    checks = [
        check_endpoint(spec, timeout=timeout, retries=retries, retry_delay=retry_delay, secret=secret, node_port=node_port)
        for spec in endpoint_specs
    ]
    peers = collect_peers(peers_report)
    stale_peers = stale_peer_rows(policy, peers)
    findings = build_findings(policy, checks, stale_peers, join, relay_port=relay_port, node_port=node_port)
    state = overall_state(findings, checks)
    summary = summarize(checks, stale_peers, findings)
    report = {
        "ok": state in {"GREEN", "YELLOW"},
        "policy": "project_theseus_hive_network_doctor_v0",
        "created_utc": now(),
        "state": state,
        "summary": summary,
        "probe_policy": {
            "timeout_seconds": timeout,
            "max_attempts": max(1, retries),
            "retry_delay_seconds": retry_delay,
            "flapping_definition": "endpoint eventually verified after a failed attempt, or peer registry already marks it flapping",
        },
        "local": {
            "system": platform.system(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "local_ips": local_ips(),
            "node_port": node_port,
            "relay_port": relay_port,
            "shared_secret_configured": bool(secret),
        },
        "active_hive": {
            "hive_id": join.get("hive_id") or active_profile.get("hive_id"),
            "hive_name": join.get("hive_name") or active_profile.get("name"),
            "tier": join.get("tier") or active_profile.get("tier"),
            "coordinator_url": join.get("coordinator_url") or first_url(endpoint_specs, "coordinator"),
            "relay_url": join.get("relay_url") or active_profile.get("relay_url"),
        },
        "endpoints": checks,
        "stale_peers": stale_peers,
        "findings": findings,
        "next_actions": next_actions(findings),
        "exact_fixes": exact_fixes(findings, node_port=node_port, relay_port=relay_port),
        "roaming": roaming_posture(checks),
        "security_guardrails": {
            "do_not_forward_node_api_port": node_port,
            "public_mobile_access": "use HTTPS relay or a private WireGuard tunnel; do not expose raw node API",
            "remote_execution": "registered bounded Hive task kinds only",
            "tokens": "not printed by this report",
        },
    }
    if write_report:
        write_json(out or DEFAULT_OUT, report)
        (markdown_out or DEFAULT_MARKDOWN).parent.mkdir(parents=True, exist_ok=True)
        (markdown_out or DEFAULT_MARKDOWN).write_text(render_markdown(report), encoding="utf-8")
    return report


def collect_endpoint_specs(
    policy: dict[str, Any],
    *,
    node_port: int,
    relay_port: int,
    join: dict[str, Any],
    active_profile: dict[str, Any],
    status_report: dict[str, Any],
    peers_report: dict[str, Any],
    coordinator_urls: list[str],
    peer_urls: list[str],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    add_endpoint(specs, "local_loopback", f"http://127.0.0.1:{node_port}", "This Mac local Hive API", node_port)
    for ip in local_ips():
        if ip.startswith("127."):
            continue
        add_endpoint(specs, "local_lan_candidate", f"http://{ip}:{node_port}", "LAN URL other devices should use for this Mac", node_port)
    if status_report.get("api_url"):
        add_endpoint(specs, "local_advertised", str(status_report.get("api_url")), "API URL from latest local probe", node_port)

    for url in coordinator_urls:
        add_endpoint(specs, "coordinator", url, "Coordinator URL supplied to this doctor", node_port)
    for key in ["coordinator_url", "relay_url"]:
        if join.get(key):
            add_endpoint(specs, "coordinator" if key == "coordinator_url" else "relay", str(join.get(key)), f"Active join config {key}", node_port if key == "coordinator_url" else relay_port)
    for key in ["coordinator_urls", "node_urls", "relay_urls", "operator_urls"]:
        values = join.get(key) if isinstance(join.get(key), list) else []
        for value in values:
            kind = "relay" if "relay" in key else "coordinator"
            add_endpoint(specs, kind, str(value), f"Active join config {key}", relay_port if kind == "relay" else node_port)
    if active_profile.get("relay_url"):
        add_endpoint(specs, "relay", str(active_profile.get("relay_url")), "Active Hive profile relay URL", relay_port)

    remote_status = read_json(REPORTS / "hive_remote_access_status.json", {})
    mobile = remote_status.get("mobile_roaming") if isinstance(remote_status.get("mobile_roaming"), dict) else {}
    roaming = mobile.get("roaming") if isinstance(mobile.get("roaming"), dict) else {}
    for endpoint in roaming.get("endpoints") or []:
        if isinstance(endpoint, dict) and endpoint.get("url"):
            kind = str(endpoint.get("kind") or "roaming")
            add_endpoint(specs, kind, str(endpoint.get("url")), "Last mobile roaming profile endpoint", relay_port if kind == "relay" else node_port)

    for peer in collect_peers(peers_report):
        if peer.get("api_url"):
            kind = "local_peer" if peer.get("is_local") else "peer"
            add_endpoint(specs, kind, str(peer.get("api_url")), f"Peer registry: {peer.get('node_name') or peer.get('node_id')}", node_port, peer=peer)
        if peer.get("relay_url"):
            add_endpoint(specs, "relay", str(peer.get("relay_url")), f"Peer relay: {peer.get('node_name') or peer.get('node_id')}", relay_port, peer=peer)
    for url in peer_urls:
        add_endpoint(specs, "peer", url, "Peer URL supplied to this doctor", node_port)
    return dedupe_specs(specs)


def add_endpoint(
    specs: list[dict[str, Any]],
    kind: str,
    raw_url: str,
    label: str,
    default_port: int,
    *,
    peer: dict[str, Any] | None = None,
) -> None:
    url = normalize_endpoint_url(raw_url, default_port=default_port)
    if not url:
        return
    specs.append({"kind": kind, "url": url, "label": label, "default_port": default_port, "peer": public_peer(peer or {})})


def dedupe_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for spec in specs:
        url = str(spec.get("url") or "")
        if not url:
            continue
        existing = merged.setdefault(url, {**spec, "kinds": [], "labels": []})
        kind = str(spec.get("kind") or "")
        label = str(spec.get("label") or "")
        if kind and kind not in existing["kinds"]:
            existing["kinds"].append(kind)
        if label and label not in existing["labels"]:
            existing["labels"].append(label)
        if not existing.get("peer") and spec.get("peer"):
            existing["peer"] = spec.get("peer")
    return list(merged.values())


def check_endpoint(
    spec: dict[str, Any],
    *,
    timeout: float,
    retries: int,
    retry_delay: float,
    secret: str,
    node_port: int,
) -> dict[str, Any]:
    url = str(spec.get("url") or "")
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else int(spec.get("default_port") or node_port))
    scope = host_scope(host)
    started = time.perf_counter()
    attempts = probe_endpoint_attempts(url, host, port, timeout=timeout, max_attempts=retries, retry_delay=retry_delay, secret=secret)
    best = best_probe_attempt(attempts)
    tcp = best.get("tcp") if isinstance(best.get("tcp"), dict) else {"ok": False, "error": "missing_host" if not host else "probe_not_run"}
    status = best.get("status") if isinstance(best.get("status"), dict) else {}
    operator = best.get("operator") if isinstance(best.get("operator"), dict) else {}
    security = endpoint_security(parsed, scope, node_port=node_port)
    latency_ms = int((time.perf_counter() - started) * 1000)
    status_ok = bool(status.get("ok")) or bool(status.get("node_id"))
    operator_ok = bool(operator.get("ok")) if operator else False
    shared_secret = shared_secret_probe_state(operator, authenticated=operator_ok)
    peer = spec.get("peer") if isinstance(spec.get("peer"), dict) else {}
    peer_seen_recently = bool(peer.get("online") or peer.get("discovery_state") in {"reachable", "flapping", "discovered", "blocked"})
    peer_outbound_verified = bool(peer.get("reachable"))
    outbound_ok = bool(status_ok or operator_ok)
    failed_attempts = [row for row in attempts if not endpoint_attempt_verified(row)]
    successful_attempts = [row for row in attempts if endpoint_attempt_verified(row)]
    retry_recovered = bool(successful_attempts and failed_attempts)
    registry_flapping = bool(peer.get("flapping") or peer.get("discovery_state") == "flapping")
    flapping = bool(retry_recovered or registry_flapping)
    if outbound_ok:
        directionality = "outbound_verified"
    elif peer_seen_recently:
        directionality = "inbound_seen_outbound_blocked"
    else:
        directionality = "not_seen"
    return {
        "kind": spec.get("kind"),
        "kinds": spec.get("kinds") or [spec.get("kind")],
        "url": url,
        "labels": spec.get("labels") or [spec.get("label")],
        "host": host,
        "port": port,
        "scheme": parsed.scheme,
        "scope": scope,
        "tcp_ok": bool(tcp.get("ok")),
        "http_status_ok": status_ok,
        "auth_ok": operator_ok,
        "operator_auth_ok": operator_ok,
        "auth_status_ok": operator_ok,
        "auth_probe_path": operator.get("probe_path", "") if isinstance(operator, dict) else "",
        "auth_probe_policy": operator.get("policy", "") if isinstance(operator, dict) else "",
        "latency_ms": latency_ms,
        "node_id": status.get("node_id") if isinstance(status, dict) else "",
        "node_name": status.get("node_name") if isinstance(status, dict) else "",
        "hive_id": status.get("hive_id") if isinstance(status, dict) else "",
        "platform": status.get("platform") if isinstance(status.get("platform"), dict) else {},
        "tcp_error": tcp.get("error", ""),
        "http_error": status.get("error", "") if isinstance(status, dict) else "",
        "operator_error": operator.get("error", "") if isinstance(operator, dict) else "",
        "auth_error": operator.get("error", "") if isinstance(operator, dict) else "",
        "attempt_count": len(attempts),
        "attempt_success_count": len(successful_attempts),
        "attempt_failure_count": len(failed_attempts),
        "retry_recovered": retry_recovered,
        "flapping": flapping,
        "registry_flapping": registry_flapping,
        "probe_attempts": public_probe_attempts(attempts),
        "latency": summarize_attempt_latency(attempts, total_ms=latency_ms),
        "authenticated_secret_proof": bool(shared_secret.get("authenticated_secret_proof")),
        "remote_shared_secret_configured": shared_secret.get("remote_shared_secret_configured"),
        "remote_shared_secret_state": shared_secret.get("remote_shared_secret_state"),
        "remote_shared_secret_source": shared_secret.get("remote_shared_secret_source"),
        "security": security,
        "peer": peer,
        "directionality": directionality,
        "peer_registry_state": peer.get("discovery_state", ""),
        "peer_seen_recently": peer_seen_recently,
        "peer_outbound_verified": peer_outbound_verified,
        "peer_last_seen_age_seconds": age_seconds(str(peer.get("last_seen_utc") or "")),
        "peer_outbound_age_seconds": age_seconds(str(peer.get("last_outbound_verified_utc") or "")),
    }


def probe_endpoint_attempts(
    url: str,
    host: str,
    port: int,
    *,
    timeout: float,
    max_attempts: int,
    retry_delay: float,
    secret: str,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    attempt_count = max(1, int(max_attempts or 1))
    for attempt_index in range(1, attempt_count + 1):
        started = time.perf_counter()
        tcp = tcp_check(host, port, timeout=timeout) if host else {"ok": False, "error": "missing_host"}
        status: dict[str, Any] = {}
        operator: dict[str, Any] = {}
        if tcp.get("ok"):
            status = http_json(url.rstrip("/") + "/api/hive/status", timeout=timeout, headers={})
            if secret:
                operator = authenticated_status(url, timeout=timeout, secret=secret)
        attempt = {
            "attempt": attempt_index,
            "tcp": tcp,
            "status": status,
            "operator": operator,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
        attempts.append(attempt)
        if endpoint_attempt_verified(attempt):
            break
        if attempt_index < attempt_count and retry_delay > 0:
            time.sleep(retry_delay)
    return attempts


def endpoint_attempt_verified(attempt: dict[str, Any]) -> bool:
    status = attempt.get("status") if isinstance(attempt.get("status"), dict) else {}
    operator = attempt.get("operator") if isinstance(attempt.get("operator"), dict) else {}
    return bool(status.get("ok") or status.get("node_id") or operator.get("ok"))


def best_probe_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    for attempt in attempts:
        if endpoint_attempt_verified(attempt):
            return attempt
    return attempts[-1] if attempts else {}


def public_probe_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        tcp = attempt.get("tcp") if isinstance(attempt.get("tcp"), dict) else {}
        status = attempt.get("status") if isinstance(attempt.get("status"), dict) else {}
        operator = attempt.get("operator") if isinstance(attempt.get("operator"), dict) else {}
        rows.append(
            {
                "attempt": attempt.get("attempt"),
                "latency_ms": attempt.get("latency_ms"),
                "tcp_ok": bool(tcp.get("ok")),
                "tcp_error": tcp.get("error", ""),
                "http_status_ok": bool(status.get("ok") or status.get("node_id")),
                "http_error": status.get("error", ""),
                "auth_ok": bool(operator.get("ok")),
                "auth_error": operator.get("error", ""),
                "verified": endpoint_attempt_verified(attempt),
            }
        )
    return rows


def summarize_attempt_latency(attempts: list[dict[str, Any]], *, total_ms: int) -> dict[str, Any]:
    samples = [int(row.get("latency_ms") or 0) for row in attempts if row.get("latency_ms") is not None]
    success_samples = [int(row.get("latency_ms") or 0) for row in attempts if endpoint_attempt_verified(row)]
    return {
        "total_ms": total_ms,
        "min_attempt_ms": min(samples) if samples else None,
        "max_attempt_ms": max(samples) if samples else None,
        "successful_attempt_ms": min(success_samples) if success_samples else None,
    }


def shared_secret_probe_state(operator: dict[str, Any], *, authenticated: bool) -> dict[str, Any]:
    """Classify authenticated probe evidence without treating old peers as failures.

    Newer peers expose `security.shared_secret_configured` on authenticated
    status. Older peers may authenticate successfully without that field. A
    successful authenticated probe is still a proof that this node has the
    correct Hive secret for the endpoint; the missing remote field is reporting
    ambiguity, not an auth failure.
    """

    if not isinstance(operator, dict) or not authenticated:
        return {
            "authenticated_secret_proof": False,
            "remote_shared_secret_configured": None,
            "remote_shared_secret_state": "not_authenticated",
            "remote_shared_secret_source": "not_authenticated",
        }
    remote = get_path(operator, ["security", "shared_secret_configured"], None)
    if remote is True:
        state = "reported_configured"
        source = "security.shared_secret_configured"
    elif remote is False:
        state = "reported_not_configured"
        source = "security.shared_secret_configured"
    else:
        state = "authenticated_but_remote_field_unreported"
        source = "auth_probe_success_without_remote_field"
    return {
        "authenticated_secret_proof": True,
        "remote_shared_secret_configured": remote,
        "remote_shared_secret_state": state,
        "remote_shared_secret_source": source,
    }


def authenticated_status(url: str, *, timeout: float, secret: str) -> dict[str, Any]:
    headers = {"X-Theseus-Hive-Secret": secret}
    auth = http_json(url.rstrip("/") + "/api/hive/auth/status", timeout=timeout, headers=headers)
    if auth.get("ok"):
        auth["probe_path"] = "/api/hive/auth/status"
        return auth
    fallback = http_json(url.rstrip("/") + "/api/hive/operator/status", timeout=timeout, headers=headers)
    fallback["probe_path"] = "/api/hive/operator/status"
    fallback["fallback_reason"] = auth.get("error") or auth.get("message") or "auth_status_unavailable"
    return fallback


def endpoint_security(parsed: Any, scope: str, *, node_port: int) -> dict[str, Any]:
    publicish = scope in {"public_ip", "public_dns", "unknown_dns"}
    warnings: list[str] = []
    if publicish and parsed.scheme != "https":
        warnings.append("public_endpoint_without_https")
    if publicish and (parsed.port or (443 if parsed.scheme == "https" else 80)) == node_port:
        warnings.append("public_node_api_port_exposed")
    if parsed.scheme == "http" and publicish:
        warnings.append("raw_http_public_path")
    return {
        "publicish": publicish,
        "private_or_tunnel": scope in {"loopback", "private_ip", "private_dns"},
        "internet_safe": bool(publicish and parsed.scheme == "https" and (parsed.port or 443) != node_port),
        "warnings": warnings,
    }


def build_findings(
    policy: dict[str, Any],
    checks: list[dict[str, Any]],
    stale_peers: list[dict[str, Any]],
    join: dict[str, Any],
    *,
    relay_port: int,
    node_port: int,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    local = endpoints_with_kind(checks, "local_loopback")
    if not any(endpoint_verified(row) for row in local):
        findings.append(finding("RED", "local_hive_api_down", "Local Hive API is not reachable on loopback.", ["Start or repair the Mac Hive service."]))

    lan = endpoints_with_kind(checks, "local_lan_candidate") + endpoints_with_kind(checks, "local_advertised")
    if any(row.get("tcp_ok") for row in local) and lan and not any(row.get("tcp_ok") for row in lan if row.get("scope") != "loopback"):
        findings.append(
            finding(
                "YELLOW",
                "mac_lan_url_not_listening",
                "This Mac is reachable on loopback but not on any LAN candidate URL.",
                ["Restart the Hive daemon bound to 0.0.0.0 and check macOS firewall/local network permission."],
            )
        )

    coordinator = endpoints_with_kind(checks, "coordinator")
    flapping_coordinator = [row for row in coordinator if row.get("flapping")]
    if join.get("coordinator_url") and not any(endpoint_verified(row) for row in coordinator):
        findings.append(
            finding(
                "RED",
                "coordinator_unreachable",
                "Configured Hive coordinator is not reachable from this Mac.",
                [
                    "On Windows, confirm the Hive node is running and Windows Firewall allows TCP 8791 on the Private profile.",
                    "Confirm the coordinator IP did not change, then update the Mac invite/profile if needed.",
                    "Check Wi-Fi client isolation or VLAN rules between this Mac and Windows.",
                ],
            )
        )
    elif flapping_coordinator:
        findings.append(
            finding(
                "RED",
                "coordinator_flapping",
                "Configured Hive coordinator is reachable only after retries or is marked flapping by the peer registry.",
                [
                    "Treat distributed work as not ready until repeated probes are stable.",
                    "On Windows, reserve or confirm the LAN IP and repair the inbound firewall on the Private network profile.",
                    "Check Windows sleep, VPN changes, Wi-Fi roaming, or client-isolation rules that interrupt TCP 8791.",
                ],
            )
        )
    elif coordinator and any(row.get("tcp_ok") for row in coordinator) and not any(row.get("operator_auth_ok") for row in coordinator):
        findings.append(
            finding(
                "YELLOW",
                "coordinator_auth_not_verified",
                "Coordinator port answers but authenticated Hive status was not verified.",
                ["Confirm this Mac has the correct active Hive invite token and restart the Hive service."],
            )
        )

    peers = endpoints_with_kind(checks, "peer")
    remote_peers = [row for row in peers if is_remote_peer_endpoint(row)]
    if remote_peers and not any(endpoint_verified(row) for row in remote_peers):
        findings.append(
            finding(
                "RED",
                "registered_peers_unreachable",
                "Peer registry contains remote nodes but none answer live status.",
                ["Refresh discovery, wake sleeping peers, and run this doctor again before queueing distributed work."],
            )
        )
    inbound_only = [row for row in remote_peers if row.get("directionality") == "inbound_seen_outbound_blocked"]
    flapping_remote_peers = [row for row in remote_peers if row.get("flapping")]
    if inbound_only:
        findings.append(
            finding(
                "RED",
                "peer_inbound_only_outbound_blocked",
                "At least one peer can report into this node, but this node cannot call back to that peer.",
                [
                    "Open the peer firewall for the Hive node API on the private LAN profile.",
                    "Confirm both machines are on the same subnet and no client-isolation rule blocks peer-to-peer traffic.",
                    "Use a private tunnel/relay path for higher-latency work if direct LAN peer calls are blocked.",
                ],
            )
        )
    if flapping_remote_peers:
        findings.append(
            finding(
                "RED",
                "peer_flapping",
                "At least one remote peer is intermittently reachable or registry-marked flapping.",
                [
                    "Do not queue CUDA/MLX cross-node training until this peer has stable outbound verification.",
                    "Run the network doctor twice several minutes apart and confirm the same peer is reachable without retry recovery.",
                    "Repair firewall, sleep, IP churn, VPN, or Wi-Fi roaming issues on the flapping peer.",
                ],
            )
        )
    if stale_peers:
        findings.append(
            finding(
                "YELLOW",
                "stale_peers_present",
                f"{len(stale_peers)} peer record(s) are stale or missing a fresh heartbeat.",
                ["Run discovery/probe on each node or let stale peers age out before planning overnight work."],
            )
        )

    public_risks = [
        row
        for row in checks
        if any(item in {"public_node_api_port_exposed", "raw_http_public_path"} for item in get_path(row, ["security", "warnings"], []))
    ]
    if public_risks:
        findings.append(
            finding(
                "RED",
                "unsafe_public_exposure",
                "At least one endpoint appears to expose raw HTTP or the node API on a public path.",
                ["Remove public forwarding for 8791 and use HTTPS relay or WireGuard/private tunnel instead."],
            )
        )

    relay = endpoints_with_kind(checks, "relay")
    has_internet_safe_relay = any(get_path(row, ["security", "internet_safe"], False) for row in relay)
    has_private_path = any(row.get("scope") in {"private_ip", "private_dns"} for row in checks)
    if not has_internet_safe_relay:
        findings.append(
            finding(
                "YELLOW",
                "cellular_roaming_path_missing",
                "No HTTPS public relay is configured for walking/cellular access.",
                [
                    "Use self-hosted WireGuard for free private roaming, or expose only the relay behind HTTPS.",
                    f"Do not forward TCP {node_port}; if you must use a router rule, forward only the authenticated relay path on {relay_port} behind HTTPS.",
                ],
            )
        )
    if not has_private_path:
        findings.append(
            finding(
                "YELLOW",
                "no_private_lan_or_tunnel_path_seen",
                "No private LAN or tunnel endpoint was visible in the profile.",
                ["Join the same LAN/hotspot or configure a private tunnel endpoint before expecting low-latency chat/control."],
            )
        )
    return findings


def finding(severity: str, code: str, title: str, fixes: list[str]) -> dict[str, Any]:
    return {"severity": severity, "code": code, "title": title, "fixes": fixes}


def overall_state(findings: list[dict[str, Any]], checks: list[dict[str, Any]]) -> str:
    if any(row.get("severity") == "RED" for row in findings):
        return "RED"
    if any(row.get("severity") == "YELLOW" for row in findings):
        return "YELLOW"
    if not checks:
        return "RED"
    return "GREEN"


def summarize(checks: list[dict[str, Any]], stale_peers: list[dict[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    remote_peers = [row for row in endpoints_with_kind(checks, "peer") if is_remote_peer_endpoint(row)]
    finding_codes = [str(row.get("code") or "") for row in findings if row.get("code")]
    red_codes = [str(row.get("code") or "") for row in findings if row.get("severity") == "RED" and row.get("code")]
    yellow_codes = [str(row.get("code") or "") for row in findings if row.get("severity") == "YELLOW" and row.get("code")]
    return {
        "endpoint_count": len(checks),
        "tcp_reachable_count": sum(1 for row in checks if row.get("tcp_ok")),
        "http_status_ok_count": sum(1 for row in checks if row.get("http_status_ok")),
        "endpoint_verified_count": sum(1 for row in checks if endpoint_verified(row)),
        "auth_ok_count": sum(1 for row in checks if row.get("auth_ok")),
        "operator_auth_ok_count": sum(1 for row in checks if row.get("operator_auth_ok")),
        "auth_status_ok_count": sum(1 for row in checks if row.get("auth_status_ok")),
        "auth_secret_proof_count": sum(1 for row in checks if row.get("authenticated_secret_proof")),
        "retry_recovered_count": sum(1 for row in checks if row.get("retry_recovered")),
        "flapping_endpoint_count": sum(1 for row in checks if row.get("flapping")),
        "remote_shared_secret_reported_configured_count": sum(1 for row in checks if row.get("remote_shared_secret_state") == "reported_configured"),
        "remote_shared_secret_unreported_authenticated_count": sum(1 for row in checks if row.get("remote_shared_secret_state") == "authenticated_but_remote_field_unreported"),
        "coordinator_reachable": any(endpoint_verified(row) for row in endpoints_with_kind(checks, "coordinator")),
        "remote_peer_reachable_count": sum(1 for row in remote_peers if endpoint_verified(row)),
        "remote_peer_flapping_count": sum(1 for row in remote_peers if row.get("flapping")),
        "remote_peer_inbound_only_count": sum(1 for row in remote_peers if row.get("directionality") == "inbound_seen_outbound_blocked"),
        "stale_peer_count": len(stale_peers),
        "red_findings": sum(1 for row in findings if row.get("severity") == "RED"),
        "yellow_findings": sum(1 for row in findings if row.get("severity") == "YELLOW"),
        "finding_codes": finding_codes,
        "red_finding_codes": red_codes,
        "yellow_finding_codes": yellow_codes,
    }


def next_actions(findings: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for row in findings:
        for fix in row.get("fixes") or []:
            if fix not in actions:
                actions.append(str(fix))
    return actions[:10] or ["Network posture looks ready for bounded multi-node Hive work."]


def exact_fixes(findings: list[dict[str, Any]], *, node_port: int, relay_port: int) -> dict[str, Any]:
    codes = {str(row.get("code") or "") for row in findings}
    fixes: dict[str, Any] = {}
    if "local_hive_api_down" in codes or "mac_lan_url_not_listening" in codes:
        fixes["mac_service"] = [
            "chmod +x scripts/start_theseus_hive.sh scripts/install_theseus_hive_macos.sh",
            "./scripts/start_theseus_hive.sh",
            "launchctl list | grep -i theseus",
            f"curl http://127.0.0.1:{node_port}/api/hive/status",
        ]
    if "coordinator_unreachable" in codes or "registered_peers_unreachable" in codes or "coordinator_flapping" in codes or "peer_flapping" in codes:
        fixes["windows_coordinator"] = [
            r'powershell -ExecutionPolicy Bypass -File scripts\start_theseus_hive.ps1',
            f'powershell -Command "New-NetFirewallRule -DisplayName \\"Project Theseus Hive {node_port}\\" -Direction Inbound -Action Allow -Protocol TCP -LocalPort {node_port} -Profile Private"',
            'powershell -Command "Get-NetConnectionProfile | Set-NetConnectionProfile -NetworkCategory Private"',
            "powershell -Command \"Get-NetConnectionProfile\"",
            "powershell -Command \"Get-NetFirewallRule -DisplayName '*Theseus*' | Format-Table DisplayName, Enabled, Direction, Action, Profile\"",
            "ipconfig",
            f"curl http://WINDOWS-LAN-IP:{node_port}/api/hive/status",
        ]
        fixes["network_checks"] = [
            "Put Mac and Windows on the same LAN/VLAN or phone hotspot.",
            "Disable guest Wi-Fi/client isolation for Hive nodes.",
            "If the Windows IP changed, re-export or update the Mac Hive invite/profile coordinator_url.",
            "If the peer flaps, reserve the Windows DHCP address or use a stable private tunnel hostname in the Hive profile.",
        ]
    if "peer_inbound_only_outbound_blocked" in codes:
        fixes["directionality"] = [
            f"From this Mac: curl http://WINDOWS-LAN-IP:{node_port}/api/hive/status",
            f"On Windows: Test-NetConnection -ComputerName MAC-LAN-IP -Port {node_port}",
            f"On Windows: Test-NetConnection -ComputerName 127.0.0.1 -Port {node_port}",
            "If Windows can reach Mac but Mac cannot reach Windows, repair the Windows inbound firewall/private network profile first.",
        ]
    if "cellular_roaming_path_missing" in codes:
        fixes["free_roaming"] = [
            "python3 scripts/hive_remote_access.py wireguard-guide --out reports/hive_wireguard_free_setup.md",
            f"Use WireGuard/private tunnel first. If using router forwarding, do not forward {node_port}; expose only a relay path on {relay_port} behind HTTPS/auth.",
            "python3 scripts/hive_remote_access.py mobile-profile --out reports/hive_mobile_roaming_profile.json",
        ]
    if "unsafe_public_exposure" in codes:
        fixes["public_exposure"] = [
            f"Remove any router rule forwarding TCP {node_port}.",
            "Put public access behind HTTPS and invite/user token authentication.",
            "Keep remote task policy limited to configs/hive_policy.json registered task kinds.",
        ]
    return fixes


def roaming_posture(checks: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = []
    for row in checks:
        kind = "private_tunnel_or_lan"
        if get_path(row, ["security", "internet_safe"], False):
            kind = "https_public_relay"
        elif row.get("scope") == "loopback":
            kind = "local_only"
        endpoints.append(
            {
                "url": row.get("url"),
                "transport": kind,
                "reachable_now": bool(row.get("http_status_ok") or row.get("tcp_ok")),
                "scope": row.get("scope"),
                "priority": 10 if kind == "private_tunnel_or_lan" else (20 if kind == "https_public_relay" else 90),
            }
        )
    return {
        "strategy": "last_good_then_lan_hotspot_then_private_tunnel_then_https_relay",
        "failover_timeout_seconds": 3,
        "endpoints": sorted(endpoints, key=lambda row: int(row.get("priority") or 999)),
        "raw_public_node_api_allowed": False,
    }


def endpoint_verified(row: dict[str, Any]) -> bool:
    return bool(row.get("http_status_ok") or row.get("auth_ok") or row.get("auth_status_ok"))


def latest_peers_report(policy: dict[str, Any], *, timeout: float, secret: str) -> dict[str, Any]:
    port = int(get_path(policy, ["node", "http_port"], 8791))
    headers = {"X-Theseus-Hive-Secret": secret} if secret else {}
    live = http_json(f"http://127.0.0.1:{port}/api/hive/peers", timeout=timeout, headers=headers)
    if live and not live.get("error"):
        return live
    return read_json(ROOT / str(get_path(policy, ["node", "peers_path"], "reports/hive_peers.json")), {})


def collect_peers(peers_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(peer: dict[str, Any], *, is_local: bool) -> None:
        key = str(peer.get("node_id") or peer.get("api_url") or "")
        if not key or key in seen:
            return
        seen.add(key)
        rows.append({**peer, "is_local": is_local})

    local = peers_report.get("local_node") if isinstance(peers_report.get("local_node"), dict) else {}
    if local:
        add(local, is_local=True)
    for key in ["peers", "known_peers", "stale_peers"]:
        for peer in peers_report.get(key) or []:
            if isinstance(peer, dict):
                add(peer, is_local=False)
    return rows


def stale_peer_rows(policy: dict[str, Any], peers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stale_after = int(get_path(policy, ["node", "stale_after_seconds"], get_path(policy, ["relay", "node_stale_after_seconds"], 90)))
    out: list[dict[str, Any]] = []
    for peer in peers:
        if peer.get("is_local") or peer_is_local_url(peer):
            continue
        stamp = str(peer.get("last_seen_utc") or peer.get("updated_utc") or peer.get("created_utc") or "")
        age = age_seconds(stamp)
        if age is None:
            out.append({**public_peer(peer), "reason": "missing_last_seen_utc"})
        elif age > stale_after:
            out.append({**public_peer(peer), "reason": "stale_heartbeat", "age_seconds": int(age), "stale_after_seconds": stale_after})
    return out


def is_remote_peer_endpoint(row: dict[str, Any]) -> bool:
    if row.get("scope") == "loopback":
        return False
    peer = row.get("peer") if isinstance(row.get("peer"), dict) else {}
    if peer.get("is_local"):
        return False
    host = str(row.get("host") or "")
    return host not in set(local_ips())


def peer_is_local_url(peer: dict[str, Any]) -> bool:
    parsed = urlparse(str(peer.get("api_url") or ""))
    host = parsed.hostname or ""
    return host in set(local_ips()) or host in {"127.0.0.1", "localhost"}


def active_join_config(policy: dict[str, Any]) -> dict[str, Any]:
    join_path = ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json"))
    join = read_json(join_path, {})
    return join if isinstance(join, dict) else {}


def hive_secret(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["security", "shared_secret_env"], "THESEUS_HIVE_SECRET"))
    value = os.environ.get(env_name, "")
    if value:
        return value
    join = active_join_config(policy)
    if join.get("join_token"):
        return str(join.get("join_token") or "")
    private = hive_profiles.active_profile()
    return str(private.get("join_token") or "") if private else ""


def http_json(url: str, *, timeout: float, headers: dict[str, str]) -> dict[str, Any]:
    req = urlrequest.Request(url, headers=headers, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:  # noqa: S310 - user-configured private Hive endpoint.
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        return {"ok": False, "error": f"http_{exc.code}", "url": url}
    except (URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": str(exc), "url": url}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_response", "body": raw[:200], "url": url}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_json", "url": url}


def tcp_check(host: str, port: int, *, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"ok": True, "latency_ms": int((time.perf_counter() - started) * 1000)}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


def normalize_endpoint_url(value: str, *, default_port: int) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.hostname:
        return ""
    path = parsed.path.rstrip("/")
    if path in {"/mobile", "/operator", "/m"}:
        path = ""
    port = parsed.port
    if port is None and parsed.scheme == "http":
        port = default_port
    if port is None and parsed.scheme == "https":
        port = 443
    port_part = f":{port}" if port and not (parsed.scheme == "https" and port == 443) else ""
    return f"{parsed.scheme}://{parsed.hostname}{port_part}{path}".rstrip("/")


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
        if "." in lowered:
            return "public_dns"
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
    if best and best not in rows:
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


def endpoints_with_kind(checks: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [row for row in checks if kind in set(str(item) for item in row.get("kinds") or [row.get("kind")])]


def first_url(specs: list[dict[str, Any]], kind: str) -> str:
    for spec in specs:
        if kind in set(str(item) for item in spec.get("kinds") or [spec.get("kind")]):
            return str(spec.get("url") or "")
    return ""


def public_peer(peer: dict[str, Any]) -> dict[str, Any]:
    if not peer or not (peer.get("node_id") or peer.get("node_name") or peer.get("api_url")):
        return {}
    return {
        "node_id": peer.get("node_id"),
        "node_name": peer.get("node_name"),
        "api_url": peer.get("api_url"),
        "relay_url": peer.get("relay_url"),
        "last_seen_utc": peer.get("last_seen_utc"),
        "created_utc": peer.get("created_utc"),
        "platform": peer.get("platform") if isinstance(peer.get("platform"), dict) else {},
        "is_local": bool(peer.get("is_local")),
        "trusted": peer.get("trusted"),
        "online": peer.get("online"),
        "reachable": peer.get("reachable"),
        "blocked": peer.get("blocked"),
        "discovery_state": peer.get("discovery_state"),
        "trust_state": peer.get("trust_state"),
        "last_outbound_verified_utc": peer.get("last_outbound_verified_utc"),
        "last_failure_utc": peer.get("last_failure_utc"),
        "reachability": peer.get("reachability") if isinstance(peer.get("reachability"), dict) else {},
    }


def age_seconds(stamp: str) -> float | None:
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hive Network Doctor",
        "",
        f"- State: **{report.get('state')}**",
        f"- Created: `{report.get('created_utc')}`",
        f"- Endpoints: {get_path(report, ['summary', 'endpoint_count'], 0)}",
        f"- HTTP reachable: {get_path(report, ['summary', 'http_status_ok_count'], 0)}",
        f"- Retry-recovered endpoints: {get_path(report, ['summary', 'retry_recovered_count'], 0)}",
        f"- Flapping endpoints: {get_path(report, ['summary', 'flapping_endpoint_count'], 0)}",
        f"- Coordinator reachable: {get_path(report, ['summary', 'coordinator_reachable'], False)}",
        f"- Remote peers reachable: {get_path(report, ['summary', 'remote_peer_reachable_count'], 0)}",
        "",
        "## Findings",
        "",
    ]
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    if findings:
        for row in findings:
            lines.append(f"- **{row.get('severity')} {row.get('code')}**: {row.get('title')}")
            for fix in row.get("fixes") or []:
                lines.append(f"  - Fix: {fix}")
    else:
        lines.append("- No blocking findings.")
    lines.extend(["", "## Endpoints", ""])
    for row in report.get("endpoints") or []:
        bits = [
            "tcp=ok" if row.get("tcp_ok") else f"tcp={row.get('tcp_error') or 'fail'}",
            "http=ok" if row.get("http_status_ok") else f"http={row.get('http_error') or 'fail'}",
            f"auth=ok:{row.get('auth_probe_path') or 'unknown'}" if row.get("auth_ok") else "auth=not_verified",
            f"secret={row.get('remote_shared_secret_state') or 'unknown'}",
            f"attempts={row.get('attempt_success_count', 0)}/{row.get('attempt_count', 0)}",
            "flapping=yes" if row.get("flapping") else "flapping=no",
        ]
        lines.append(f"- `{row.get('url')}` ({row.get('scope')}, {', '.join(str(item) for item in row.get('kinds') or [])}) - {', '.join(bits)}")
    lines.extend(["", "## Next Actions", ""])
    for item in report.get("next_actions") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
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
