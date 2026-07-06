"""Peer discovery and registry helpers for the Project Theseus Hive node."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError
from urllib.parse import urlparse

from hive_node_common import append_jsonl, display_path, event, get_path, now, read_json, report_path, task_ledger_path, write_json
from hive_node_federation import hive_id, hive_tier, join_token, shared_secret, unique_nonempty
from hive_node_identity import load_identity


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PEERS_LOCK = threading.Lock()
PEERS: dict[str, dict[str, Any]] = {}
PEERS_LOADED = False


def announce_multicast(policy: dict[str, Any], status: dict[str, Any]) -> None:
    discovery = policy.get("discovery") or {}
    group = str(discovery.get("multicast_group") or "239.255.87.87")
    port = int(discovery.get("multicast_port") or 8789)
    payload = signed_discovery_payload(policy, status)
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, int(discovery.get("ttl") or 1))
        sock.sendto(raw, (group, port))
    except OSError:
        pass
    finally:
        sock.close()


def listen_multicast(policy: dict[str, Any], stop_event: threading.Event | None = None) -> None:
    discovery = policy.get("discovery") or {}
    group = str(discovery.get("multicast_group") or "239.255.87.87")
    port = int(discovery.get("multicast_port") or 8789)
    stop = stop_event or threading.Event()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))
        mreq = socket.inet_aton(group) + socket.inet_aton("0.0.0.0")
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(1.0)
        while not stop.is_set():
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("prefix") != discovery.get("announce_prefix", "project-theseus-hive"):
                continue
            if str(payload.get("hive_id") or "") and str(payload.get("hive_id") or "") != hive_id(policy):
                continue
            peer = payload.get("peer") if isinstance(payload.get("peer"), dict) else {}
            trusted = verify_discovery_payload(policy, payload) or unsigned_multicast_trusted(policy, payload)
            source = "multicast_signed" if trusted else "multicast_unsigned"
            accept_peer(policy, peer, source=source, trusted=trusted)
    except OSError:
        return
    finally:
        sock.close()


def bonjour_enabled(policy: dict[str, Any]) -> bool:
    return bool(get_path(policy, ["discovery", "bonjour_enabled"], True)) and bool(shutil.which("dns-sd"))


def bonjour_service_type(policy: dict[str, Any]) -> str:
    return str(get_path(policy, ["discovery", "bonjour_service_type"], "_theseus-hive._tcp"))


def bonjour_domain(policy: dict[str, Any]) -> str:
    domain = str(get_path(policy, ["discovery", "bonjour_domain"], "local.")).strip() or "local."
    return domain if domain.endswith(".") else domain + "."


def bonjour_register_command(policy: dict[str, Any], status: dict[str, Any], port: int) -> list[str]:
    dns_sd = shutil.which("dns-sd")
    if not dns_sd:
        return []
    service_name = safe_bonjour_name(f"Theseus Hive {status.get('node_name') or status.get('node_id')}")
    return [
        dns_sd,
        "-R",
        service_name,
        bonjour_service_type(policy),
        bonjour_domain(policy),
        str(int(port)),
        *bonjour_txt_records(policy, status),
    ]


def bonjour_txt_records(policy: dict[str, Any], status: dict[str, Any]) -> list[str]:
    caps = ",".join(str(row.get("id") or "") for row in status.get("capabilities", [])[:10] if isinstance(row, dict) and row.get("id"))
    roles = ",".join(str(item) for item in status.get("roles", [])[:8])
    payload = {
        "txtvers": "1",
        "hive_id": str(status.get("hive_id") or ""),
        "node_id": str(status.get("node_id") or ""),
        "node_name": str(status.get("node_name") or "")[:64],
        "api_url": str(status.get("api_url") or "")[:180],
        "relay_url": str(status.get("relay_url") or "")[:180],
        "roles": roles[:180],
        "caps": caps[:180],
    }
    sig = bonjour_signature(policy, payload)
    if sig:
        payload["sig"] = sig
        payload["sig_alg"] = "hmac-sha256"
    return [f"{key}={value}" for key, value in payload.items() if value]


def bonjour_signature(policy: dict[str, Any], txt: dict[str, Any]) -> str:
    token = join_token(policy) or shared_secret(policy)
    if not token:
        return ""
    signed = {
        "hive_id": txt.get("hive_id"),
        "node_id": txt.get("node_id"),
        "api_url": txt.get("api_url"),
    }
    raw = json.dumps(signed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(token.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def verify_bonjour_txt(policy: dict[str, Any], txt: dict[str, Any]) -> bool:
    token = join_token(policy) or shared_secret(policy)
    if not token:
        return False
    signature = str(txt.get("sig") or "")
    if not signature:
        return False
    expected = bonjour_signature(policy, txt)
    return bool(expected and hmac.compare_digest(signature, expected))


def scan_bonjour(policy: dict[str, Any], *, seconds: float) -> dict[str, Any]:
    dns_sd = shutil.which("dns-sd")
    if not dns_sd or not bool(get_path(policy, ["discovery", "bonjour_enabled"], True)):
        return {
            "ok": False,
            "policy": "project_theseus_hive_bonjour_scan_v0",
            "created_utc": now(),
            "available": False,
            "reason": "dns_sd_not_available_or_disabled",
            "peers": [],
        }
    browse = run_dns_sd_capture([dns_sd, "-B", bonjour_service_type(policy), bonjour_domain(policy)], timeout=max(0.5, seconds))
    instances = parse_bonjour_browse(browse.get("stdout", ""), policy)
    peers: list[dict[str, Any]] = []
    for instance in instances[:32]:
        resolved = run_dns_sd_capture([dns_sd, "-L", instance, bonjour_service_type(policy), bonjour_domain(policy)], timeout=2.0)
        peer = peer_from_bonjour_resolve(instance, resolved.get("stdout", ""), policy)
        if peer:
            peers.append(peer)
    report = {
        "ok": True,
        "policy": "project_theseus_hive_bonjour_scan_v0",
        "created_utc": now(),
        "available": True,
        "service_type": bonjour_service_type(policy),
        "domain": bonjour_domain(policy),
        "instance_count": len(instances),
        "peer_count": len(peers),
        "peers": peers,
    }
    write_json(REPORTS / "hive_bonjour_discovery.json", report)
    return report


def run_dns_sd_capture(command: list[str], *, timeout: float) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {"ok": result.returncode == 0, "stdout": result.stdout or "", "stderr": result.stderr or "", "returncode": result.returncode}
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {"ok": True, "timeout": True, "stdout": stdout, "stderr": stderr}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}


def parse_bonjour_browse(stdout: str, policy: dict[str, Any]) -> list[str]:
    service_type = bonjour_service_type(policy)
    out: list[str] = []
    for line in stdout.splitlines():
        if " Add " not in f" {line} " or service_type not in line:
            continue
        instance = line.split(service_type, 1)[-1].strip(" .\t")
        if instance and instance not in out:
            out.append(instance)
    return out


def peer_from_bonjour_resolve(instance: str, stdout: str, policy: dict[str, Any]) -> dict[str, Any]:
    host = ""
    port = 0
    txt: dict[str, str] = {}
    for line in stdout.splitlines():
        match = re.search(r"can be reached at ([^:]+):(\d+)", line)
        if match:
            host = match.group(1).rstrip(".")
            port = int(match.group(2))
        for key, value in re.findall(r"([A-Za-z0-9_.-]+)=([^ \t\r\n]+)", line):
            txt[key] = value.strip('"')
    api_url = txt.get("api_url") or (f"http://{host}:{port}" if host and port else "")
    node_id = txt.get("node_id") or ""
    if not node_id or not api_url:
        return {}
    return {
        "node_id": node_id,
        "node_name": txt.get("node_name") or instance,
        "api_url": api_url,
        "relay_url": txt.get("relay_url") or "",
        "hive_id": txt.get("hive_id") or hive_id(policy),
        "federation_tier": hive_tier(policy),
        "capabilities": [{"id": item, "score": 0.0, "detail": "bonjour_txt"} for item in (txt.get("caps") or "").split(",") if item],
        "roles": [item for item in (txt.get("roles") or "").split(",") if item],
        "bonjour": {
            "instance": instance,
            "host": host,
            "port": port,
            "signed": verify_bonjour_txt(policy, txt),
            "service_type": bonjour_service_type(policy),
        },
    }


def accept_bonjour_peers(policy: dict[str, Any], report: dict[str, Any]) -> None:
    peers = report.get("peers") if isinstance(report.get("peers"), list) else []
    for peer in peers:
        if not isinstance(peer, dict):
            continue
        signed = bool(get_path(peer, ["bonjour", "signed"], False))
        if join_token(policy) or shared_secret(policy):
            trusted = signed
            source = "bonjour_signed" if signed else "bonjour_unsigned"
        else:
            trusted = bool(get_path(policy, ["discovery", "trust_unsigned_multicast_without_secret"], True))
            source = "bonjour_unsigned"
        accept_peer(policy, peer, source=source, trusted=trusted)


def safe_bonjour_name(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {" ", "-", "_", "."} else "-" for ch in value).strip()
    clean = " ".join(clean.split())
    return (clean or "Theseus Hive Node")[:60]


def signed_discovery_payload(policy: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    discovery = policy.get("discovery") or {}
    payload = {
        "prefix": discovery.get("announce_prefix", "project-theseus-hive"),
        "hive_id": hive_id(policy),
        "created_utc": now(),
        "nonce": uuid.uuid4().hex[:16],
        "peer": peer_from_status(status),
    }
    signature = discovery_signature(policy, payload)
    if signature:
        payload["signature"] = signature
        payload["signature_alg"] = "hmac-sha256"
    return payload


def verify_discovery_payload(policy: dict[str, Any], payload: dict[str, Any]) -> bool:
    token = join_token(policy) or shared_secret(policy)
    signature = str(payload.get("signature") or "")
    if not token or not signature:
        return False
    expected = discovery_signature(policy, {key: value for key, value in payload.items() if key != "signature"})
    return bool(expected and hmac.compare_digest(signature, expected))


def unsigned_multicast_trusted(policy: dict[str, Any], payload: dict[str, Any]) -> bool:
    if join_token(policy) or shared_secret(policy):
        return False
    if bool(get_path(policy, ["discovery", "require_signed_multicast"], False)):
        return False
    if not isinstance(payload.get("peer"), dict):
        return False
    return bool(get_path(policy, ["discovery", "trust_unsigned_multicast_without_secret"], True))


def discovery_signature(policy: dict[str, Any], payload: dict[str, Any]) -> str:
    token = join_token(policy) or shared_secret(policy)
    if not token:
        return ""
    signed = {
        "prefix": payload.get("prefix"),
        "hive_id": payload.get("hive_id"),
        "created_utc": payload.get("created_utc"),
        "nonce": payload.get("nonce"),
        "peer": payload.get("peer") if isinstance(payload.get("peer"), dict) else {},
    }
    raw = json.dumps(signed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(token.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def accept_heartbeat_response(policy: dict[str, Any], payload: dict[str, Any], *, source: str) -> None:
    if not isinstance(payload, dict):
        return
    peer = payload.get("peer") if isinstance(payload.get("peer"), dict) else {}
    if peer:
        accept_peer(policy, peer, source=source, trusted=True)
    for row in payload.get("peers") or []:
        if isinstance(row, dict):
            accept_peer(policy, row, source=source + "_peer", trusted=True)


def ensure_peer_registry_loaded(policy: dict[str, Any]) -> None:
    global PEERS_LOADED
    with PEERS_LOCK:
        if PEERS_LOADED:
            return
        PEERS_LOADED = True
    registry = read_json(peer_registry_path(policy), {})
    rows: list[dict[str, Any]] = []
    if isinstance(registry.get("known_peers"), list):
        rows.extend(row for row in registry.get("known_peers") if isinstance(row, dict))
    legacy = read_json(report_path(policy, "peers_path", "reports/hive_peers.json", ""), {})
    for key in ["peers", "known_peers"]:
        for row in legacy.get(key) or []:
            if isinstance(row, dict):
                rows.append(row)
    expected_hive = hive_id(policy)
    identity = load_identity(policy)
    self_id = str(identity.get("node_id") or "")
    self_name = str(identity.get("node_name") or "")
    loaded: dict[str, dict[str, Any]] = {}
    for row in rows:
        node_id = str(row.get("node_id") or "")
        if not node_id:
            continue
        if stale_self_peer(row, self_id=self_id, self_name=self_name):
            continue
        peer_hive = str(row.get("hive_id") or expected_hive)
        if expected_hive and expected_hive != "local" and peer_hive and peer_hive != expected_hive:
            continue
        existing = loaded.get(node_id, {})
        merged = {**existing, **sanitize_peer_for_registry(row)}
        merged["node_id"] = node_id
        merged["hive_id"] = peer_hive
        merged["first_seen_utc"] = existing.get("first_seen_utc") or merged.get("first_seen_utc") or merged.get("last_seen_utc") or now()
        merged["trusted"] = bool(existing.get("trusted") or row.get("trusted") or peer_is_trusted(row))
        merged["trust_state"] = "trusted" if merged["trusted"] else "unverified"
        merged["discovery_sources"] = unique_nonempty(
            [str(item) for item in existing.get("discovery_sources", []) if item]
            + [str(item) for item in row.get("discovery_sources", []) if item]
        )
        loaded[node_id] = merged
    if loaded:
        with PEERS_LOCK:
            for node_id, row in loaded.items():
                PEERS.setdefault(node_id, row)


def stale_self_peer(peer: dict[str, Any], *, self_id: str, self_name: str) -> bool:
    node_id = str(peer.get("node_id") or "")
    if not node_id:
        return False
    if node_id == self_id:
        return True
    if self_name and str(peer.get("node_name") or "") == self_name:
        return True
    parsed = urlparse(str(peer.get("api_url") or ""))
    host = parsed.hostname or ""
    return host in set(local_address_candidates())


def local_address_candidates() -> list[str]:
    rows = ["127.0.0.1", "localhost", socket.gethostname()]
    local_ip = find_local_ip()
    if local_ip:
        rows.append(local_ip)
    try:
        rows.append(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    return unique_nonempty([str(item) for item in rows if item])


def persist_peer_registry(policy: dict[str, Any]) -> None:
    stale_after = float(get_path(policy, ["node", "stale_after_seconds"], 45))
    retention = float(get_path(policy, ["discovery", "peer_retention_seconds"], 604800))
    now_mono = time.monotonic()
    with PEERS_LOCK:
        retained: dict[str, dict[str, Any]] = {}
        rows = []
        for node_id, peer in PEERS.items():
            if peer_should_retain(peer, retention_seconds=retention, now_mono=now_mono):
                retained[node_id] = peer
                rows.append(public_peer_row(peer, stale_after=stale_after, now_mono=now_mono, include_private_state=True))
        if len(retained) != len(PEERS):
            PEERS.clear()
            PEERS.update(retained)
    rows = sorted(rows, key=lambda peer: str(peer.get("last_seen_utc") or ""), reverse=True)
    max_peers = int(get_path(policy, ["discovery", "max_known_peers"], 256))
    payload = {
        "policy": "project_theseus_hive_peer_registry_v1",
        "created_utc": now(),
        "hive_id": hive_id(policy),
        "known_peer_count": len(rows[:max_peers]),
        "known_peers": rows[:max_peers],
    }
    write_json(peer_registry_path(policy), payload)


def peer_registry_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["discovery", "peer_registry_path"], "reports/hive_peer_registry.json"))


def sanitize_peer_for_registry(peer: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in peer.items() if key != "last_seen_monotonic" and key != "last_failure_monotonic"}


def trusted_peer_sources() -> set[str]:
    return {
        "coordinator_status",
        "coordinator_heartbeat",
        "coordinator_heartbeat_peer",
        "relay",
        "peer_probe",
        "peer_probe_peer",
        "multicast_signed",
        "bonjour_signed",
    }


def outbound_verified_peer_sources() -> set[str]:
    return {
        "coordinator_status",
        "coordinator_heartbeat",
        "peer_probe",
    }


def source_is_trusted(source: str) -> bool:
    if source in trusted_peer_sources():
        return True
    return source.startswith("heartbeat:")


def source_is_outbound_verified(source: str) -> bool:
    return source in outbound_verified_peer_sources()


def peer_is_trusted(peer: dict[str, Any]) -> bool:
    if bool(peer.get("trusted")):
        return True
    sources = {str(item) for item in peer.get("discovery_sources", []) if item}
    return any(source_is_trusted(source) for source in sources)


def peer_age_seconds(peer: dict[str, Any], *, now_mono: float) -> float | None:
    stamp = peer.get("last_seen_monotonic")
    if isinstance(stamp, (int, float)) and stamp > 0:
        return max(0.0, now_mono - float(stamp))
    return timestamp_age_seconds(str(peer.get("last_seen_utc") or peer.get("updated_utc") or peer.get("created_utc") or ""))


def peer_outbound_age_seconds(peer: dict[str, Any], *, now_mono: float) -> float | None:
    stamp = peer.get("last_outbound_verified_monotonic")
    if isinstance(stamp, (int, float)) and stamp > 0:
        return max(0.0, now_mono - float(stamp))
    return timestamp_age_seconds(str(peer.get("last_outbound_verified_utc") or ""))


def peer_failure_is_newer(peer: dict[str, Any]) -> bool:
    failure_mono = peer.get("last_failure_monotonic")
    outbound_mono = peer.get("last_outbound_verified_monotonic")
    if isinstance(failure_mono, (int, float)) and isinstance(outbound_mono, (int, float)):
        return float(failure_mono) >= float(outbound_mono)
    failure_age = timestamp_age_seconds(str(peer.get("last_failure_utc") or ""))
    outbound_age = timestamp_age_seconds(str(peer.get("last_outbound_verified_utc") or ""))
    if failure_age is None:
        return False
    if outbound_age is None:
        return True
    return failure_age <= outbound_age


def peer_is_seen(peer: dict[str, Any], *, stale_after: float, now_mono: float) -> bool:
    age = peer_age_seconds(peer, now_mono=now_mono)
    return age is not None and age <= stale_after


def peer_is_outbound_reachable(peer: dict[str, Any], *, stale_after: float, now_mono: float) -> bool:
    if peer_failure_is_newer(peer) and int(peer.get("consecutive_failures") or 0) > 0:
        return False
    age = peer_outbound_age_seconds(peer, now_mono=now_mono)
    return age is not None and age <= stale_after


def peer_should_retain(peer: dict[str, Any], *, retention_seconds: float, now_mono: float) -> bool:
    if retention_seconds <= 0:
        return True
    age = peer_age_seconds(peer, now_mono=now_mono)
    failure_age = timestamp_age_seconds(str(peer.get("last_failure_utc") or ""))
    if age is None:
        return True
    if age <= retention_seconds:
        return True
    if failure_age is not None and failure_age <= retention_seconds:
        return True
    return False


def public_peer_row(
    peer: dict[str, Any],
    *,
    stale_after: float,
    now_mono: float,
    include_private_state: bool = False,
) -> dict[str, Any]:
    age = peer_age_seconds(peer, now_mono=now_mono)
    outbound_age = peer_outbound_age_seconds(peer, now_mono=now_mono)
    failure_age = timestamp_age_seconds(str(peer.get("last_failure_utc") or ""))
    trusted = peer_is_trusted(peer)
    online = peer_is_seen(peer, stale_after=stale_after, now_mono=now_mono)
    outbound_reachable = trusted and peer_is_outbound_reachable(peer, stale_after=stale_after, now_mono=now_mono)
    blocked = bool(peer_failure_is_newer(peer) and int(peer.get("consecutive_failures") or 0) > 0)
    recent_failure = failure_age is not None and failure_age <= max(stale_after, 300.0)
    flapping = bool(outbound_reachable and recent_failure)
    row = sanitize_peer_for_registry(peer)
    row["trusted"] = trusted
    row["trust_state"] = "trusted" if trusted else "unverified"
    if flapping:
        discovery_state = "flapping"
    elif outbound_reachable:
        discovery_state = "reachable"
    elif blocked:
        discovery_state = "blocked"
    elif age is not None and age <= stale_after:
        discovery_state = "discovered"
    elif age is not None:
        discovery_state = "stale"
    else:
        discovery_state = "unknown"
    row["discovery_state"] = discovery_state
    row["reachable"] = bool(outbound_reachable)
    row["blocked"] = blocked
    row["flapping"] = flapping
    row["online"] = bool(online and trusted)
    row["age_seconds"] = int(age) if age is not None else None
    row["outbound_age_seconds"] = int(outbound_age) if outbound_age is not None else None
    row["failure_age_seconds"] = int(failure_age) if failure_age is not None else None
    row["stale_after_seconds"] = int(stale_after)
    row["discovery_sources"] = unique_nonempty([str(item) for item in row.get("discovery_sources", []) if item])
    row["consecutive_failures"] = int(row.get("consecutive_failures") or 0)
    row["reachability"] = {
        "state": discovery_state,
        "seen_recently": bool(online and trusted),
        "outbound_verified": bool(outbound_reachable),
        "last_seen_utc": row.get("last_seen_utc"),
        "last_outbound_verified_utc": row.get("last_outbound_verified_utc"),
        "last_failure_utc": row.get("last_failure_utc"),
        "last_error": row.get("last_error", "") if include_private_state else "",
    }
    if not include_private_state:
        row.pop("last_error", None)
    return row


def mark_peer_failure(policy: dict[str, Any], node_id: str, error: str) -> None:
    if not node_id:
        return
    now_utc = now()
    with PEERS_LOCK:
        peer = PEERS.get(node_id)
        if not peer:
            return
        peer["last_failure_utc"] = now_utc
        peer["last_failure_monotonic"] = time.monotonic()
        peer["last_error"] = str(error)[:300]
        peer["consecutive_failures"] = int(peer.get("consecutive_failures") or 0) + 1
    persist_peer_registry(policy)


def mark_peer_url_failure(policy: dict[str, Any], api_url: str, error: str) -> None:
    normalized = api_url.rstrip("/")
    node_id = ""
    with PEERS_LOCK:
        for peer in PEERS.values():
            if str(peer.get("api_url") or "").rstrip("/") == normalized:
                node_id = str(peer.get("node_id") or "")
                break
    if node_id:
        mark_peer_failure(policy, node_id, error)


def timestamp_age_seconds(stamp: str) -> float | None:
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())


def accept_peer(policy: dict[str, Any], peer: dict[str, Any], *, source: str = "unknown", trusted: bool = False) -> dict[str, Any]:
    ensure_peer_registry_loaded(policy)
    if not isinstance(peer, dict) or not peer.get("node_id"):
        return {"ok": False, "error": "invalid_peer"}
    self_id = load_identity(policy)["node_id"]
    if peer.get("node_id") == self_id:
        return {"ok": True, "status": "ignored_self"}
    expected_hive = hive_id(policy)
    peer_hive = str(peer.get("hive_id") or expected_hive)
    if expected_hive and expected_hive != "local" and peer_hive and peer_hive != expected_hive:
        return {"ok": False, "error": "peer_hive_mismatch", "peer_hive_id": peer_hive, "expected_hive_id": expected_hive}
    api_url = str(peer.get("api_url") or "")
    if api_url and not api_url.startswith(("http://", "https://")):
        return {"ok": False, "error": "peer_api_url_unsupported", "api_url": api_url}
    node_id = str(peer["node_id"])
    now_utc = now()
    source = str(source or "unknown")
    outbound_verified = source_is_outbound_verified(source)
    with PEERS_LOCK:
        existing = PEERS.get(node_id, {})
        sources = unique_nonempty(
            [str(item) for item in existing.get("discovery_sources", []) if item]
            + [source]
        )
        merged = {**existing, **peer}
        merged["node_id"] = node_id
        merged["hive_id"] = peer_hive
        merged["first_seen_utc"] = existing.get("first_seen_utc") or now_utc
        merged["last_seen_utc"] = now_utc
        merged["last_seen_monotonic"] = time.monotonic()
        merged["last_contact_direction"] = "outbound" if outbound_verified else ("inbound" if source.startswith("heartbeat:") else "discovery")
        merged["discovery_sources"] = sources
        merged["trusted"] = bool(existing.get("trusted") or trusted or source_is_trusted(source))
        merged["trust_state"] = "trusted" if merged["trusted"] else "unverified"
        if merged["trusted"] and outbound_verified:
            merged["last_verified_utc"] = now_utc
            merged["last_outbound_verified_utc"] = now_utc
            merged["last_outbound_verified_monotonic"] = time.monotonic()
            merged["consecutive_failures"] = 0
            merged.pop("last_error", None)
        PEERS[node_id] = merged
        trusted_state = bool(merged.get("trusted"))
    persist_peer_registry(policy)
    return {"ok": True, "status": "accepted", "node_id": node_id, "trusted": trusted_state}


def current_peers(policy: dict[str, Any]) -> list[dict[str, Any]]:
    ensure_peer_registry_loaded(policy)
    stale_after = float(get_path(policy, ["node", "stale_after_seconds"], 45))
    now_mono = time.monotonic()
    with PEERS_LOCK:
        return [
            public_peer_row(peer, stale_after=stale_after, now_mono=now_mono)
            for peer in PEERS.values()
            if peer_is_seen(peer, stale_after=stale_after, now_mono=now_mono) and peer_is_trusted(peer)
        ]


def known_peers(policy: dict[str, Any]) -> list[dict[str, Any]]:
    ensure_peer_registry_loaded(policy)
    stale_after = float(get_path(policy, ["node", "stale_after_seconds"], 45))
    now_mono = time.monotonic()
    with PEERS_LOCK:
        return [
            public_peer_row(peer, stale_after=stale_after, now_mono=now_mono)
            for peer in PEERS.values()
        ]


def probe_known_peers(policy: dict[str, Any], status: dict[str, Any]) -> None:
    if not bool(get_path(policy, ["discovery", "probe_known_peers"], True)):
        return
    secret = join_token(policy) or shared_secret(policy)
    if not secret:
        return
    rows = known_peers(policy)
    for peer in rows:
        if not peer.get("trusted"):
            continue
        api_url = str(peer.get("api_url") or "").rstrip("/")
        if not api_url or api_url.startswith("http://127.0.0.1:"):
            continue
        if str(peer.get("node_id") or "") == str(status.get("node_id") or ""):
            continue
        try:
            body = json.dumps({"peer": peer_from_status(status)}).encode("utf-8")
            req = urlrequest.Request(
                api_url + "/api/hive/heartbeat",
                data=body,
                headers={"Content-Type": "application/json", "X-Theseus-Hive-Secret": secret},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=float(get_path(policy, ["discovery", "peer_probe_timeout_seconds"], 3))) as response:  # noqa: S310 - private trusted Hive peer.
                payload = json.loads(response.read().decode("utf-8"))
            accept_heartbeat_response(policy, payload, source="peer_probe")
        except (OSError, URLError, json.JSONDecodeError) as exc:
            mark_peer_failure(policy, str(peer.get("node_id") or ""), str(exc))


def peer_from_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": status.get("node_id"),
        "node_name": status.get("node_name"),
        "hostname": status.get("hostname"),
        "api_url": status.get("api_url"),
        "dashboard_url": status.get("dashboard_url"),
        "hive_id": status.get("hive_id"),
        "federation_tier": status.get("federation_tier"),
        "relay_url": status.get("relay_url"),
        "roles": status.get("roles") or [],
        "platform": status.get("platform"),
        "capabilities": status.get("capabilities"),
        "resources": {
            "cpu": get_path(status, ["resources", "cpu"], {}),
            "memory": get_path(status, ["resources", "memory"], {}),
            "nvidia": get_path(status, ["resources", "nvidia"], {}),
            "mlx": get_path(status, ["resources", "mlx"], {}),
            "disk": get_path(status, ["resources", "disk"], {}),
            "power": get_path(status, ["resources", "power"], {}),
            "thermal": get_path(status, ["resources", "thermal"], {}),
        },
        "slots": status.get("slots") or [],
        "storage": status.get("storage") or {},
        "remote_control": status.get("remote_control") or {},
        "voice_following": status.get("voice_following") or {},
        "runtime_paths": status.get("runtime_paths") or {},
        "updates": status.get("updates") or {},
        "created_utc": status.get("created_utc"),
    }


def peer_state_counts(peers: list[dict[str, Any]]) -> dict[str, int]:
    states = {"reachable": 0, "flapping": 0, "discovered": 0, "stale": 0, "blocked": 0, "unknown": 0, "unverified": 0}
    for peer in peers:
        state = str(peer.get("discovery_state") or "unknown")
        if not peer.get("trusted"):
            states["unverified"] += 1
        states[state if state in states else "unknown"] += 1
    return states


def find_local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
        finally:
            sock.close()
    except OSError:
        return "127.0.0.1"
