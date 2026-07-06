"""One-step bootstrap artifacts for Project Theseus Hive nodes and phones.

The bootstrap bundle is the durable object a non-Codex installer, phone, Watch
relay, spare Mac, Windows PC, or Linux/RPi node can import to join the Hive.
It contains endpoint candidates and update/catalog URLs, and includes the join
token only when explicitly requested.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
POLICY_PATH = CONFIGS / "hive_policy.json"

sys.path.insert(0, str(ROOT / "scripts"))
import hive_profiles  # noqa: E402
import hive_remote_access  # noqa: E402
from theseus_qr import qr_svg  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a Project Theseus Hive bootstrap join bundle.")
    parser.add_argument("--out", default="reports/hive_join_bundle.json")
    parser.add_argument("--qr-out", default="reports/hive_join_profile_qr.svg")
    parser.add_argument("--no-token", action="store_true")
    parser.add_argument("--coordinator-url", action="append", default=[])
    parser.add_argument("--relay-url", action="append", default=[])
    parser.add_argument("--operator-token-scope", default="")
    args = parser.parse_args()

    policy = read_json(POLICY_PATH, {})
    report = write_bootstrap_bundle(policy=policy, args=args)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 2


def write_bootstrap_bundle(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    active = hive_profiles.active_profile()
    if not active:
        return {"ok": False, "error": "no_active_hive", "message": "Create or join a Hive before writing a bootstrap bundle."}
    include_token = not bool(getattr(args, "no_token", False))
    bundle = bootstrap_bundle(
        active,
        policy=policy,
        include_token=include_token,
        extra_coordinator_urls=[str(item) for item in getattr(args, "coordinator_url", []) or []],
        extra_relay_urls=[str(item) for item in getattr(args, "relay_url", []) or []],
        operator_token_scope=str(getattr(args, "operator_token_scope", "") or ""),
    )
    out = resolve(str(args.out or "reports/hive_join_bundle.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    qr_path = resolve(str(args.qr_out or "reports/hive_join_profile_qr.svg"))
    qr_error = ""
    if include_token:
        qr_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            qr_path.write_text(qr_svg(str(bundle.get("qr_join_url") or bundle.get("ios_app_url") or "")), encoding="utf-8")
        except ValueError as exc:
            qr_error = str(exc)
            qr_path = qr_path.with_suffix(".txt")
            qr_path.write_text(str(bundle.get("ios_app_url") or bundle.get("qr_join_url") or ""), encoding="utf-8")
    report = {
        "ok": True,
        "policy": "project_theseus_hive_bootstrap_bundle_write_v0",
        "created_utc": now(),
        "bundle_path": rel(out),
        "qr_path": rel(qr_path) if include_token else "",
        "qr_error": qr_error,
        "hive_id": active.get("hive_id"),
        "hive_name": active.get("name"),
        "endpoint_count": len(bundle.get("roaming", {}).get("endpoints", [])),
        "update_catalog_url_count": len(bundle.get("update_catalog_urls", [])),
        "token_included": include_token,
        "ios_app_url_written": bool(include_token),
        "qr_join_url_written": bool(bundle.get("qr_join_url")),
        "next_commands": [
            f"theseus join --invite {rel(out)}",
            "Open the QR SVG from a trusted screen to import the iPhone profile.",
            "Install the macOS DMG/PKG or Windows installer, then import this bundle; Codex is not required on the target node.",
        ],
        "security_notes": [
            "Token-bearing bundles and QR codes are passwords.",
            "Use per-user operator tokens for family phones when possible.",
            "The bundle never enables arbitrary shell remote execution.",
        ],
    }
    write_json(REPORTS / "hive_join_bundle_report.json", report)
    return report


def bootstrap_bundle(
    profile: dict[str, Any],
    *,
    policy: dict[str, Any],
    include_token: bool,
    extra_coordinator_urls: list[str],
    extra_relay_urls: list[str],
    operator_token_scope: str,
) -> dict[str, Any]:
    enriched = dict(profile)
    merge_profile_urls(enriched, "coordinator_urls", extra_coordinator_urls)
    merge_profile_urls(enriched, "relay_urls", extra_relay_urls)
    roaming = hive_remote_access.mobile_roaming_profile(enriched, policy=policy, include_token=include_token)
    endpoint_urls = unique_urls(
        list(roaming.get("coordinator_urls") or [])
        + list(roaming.get("node_urls") or [])
        + list(roaming.get("relay_urls") or [])
        + live_peer_urls()
    )
    update_catalog_urls = [url.rstrip("/") + "/api/hive/update-catalog" for url in endpoint_urls]
    installer_artifacts_urls = [url.rstrip("/") + "/api/hive/installer-artifacts" for url in endpoint_urls]
    bundle: dict[str, Any] = {
        "policy": "project_theseus_hive_join_bundle_v1",
        "created_utc": now(),
        "hive_id": profile.get("hive_id"),
        "hive_name": profile.get("name"),
        "tier": profile.get("tier"),
        "coordinator_url": roaming.get("coordinator_url", ""),
        "coordinator_urls": roaming.get("coordinator_urls", []),
        "node_urls": roaming.get("node_urls", []),
        "relay_url": profile.get("relay_url"),
        "relay_urls": roaming.get("relay_urls", []),
        "operator_urls": roaming.get("operator_urls", []),
        "roaming": roaming.get("roaming", {}),
        "update_catalog_url": update_catalog_urls[0] if update_catalog_urls else "",
        "update_catalog_urls": update_catalog_urls,
        "installer_artifacts_urls": installer_artifacts_urls,
        "token_scope": token_scope(profile, policy=policy, operator_token_scope=operator_token_scope),
        "node_identity": local_identity_summary(),
        "install": install_contract(profile, bundle_path_hint="BOOTSTRAP_BUNDLE.json"),
        "security": {
            "remote_tasks": "registered bounded Hive task kinds only",
            "arbitrary_shell": False,
            "public_gateway": False,
            "router_forwarding_required": False,
            "recommended_remote_access": "same_lan_hotspot_or_self_hosted_wireguard_or_https_relay",
        },
        "no_codex_required": True,
    }
    if include_token:
        bundle["join_token"] = profile.get("join_token")
        bundle["ios_app_url"] = mobile_app_join_url(bundle)
        bundle["qr_join_url"] = short_mobile_join_url(bundle)
    return bundle


def merge_profile_urls(profile: dict[str, Any], key: str, values: list[str]) -> None:
    rows = profile.get(key) if isinstance(profile.get(key), list) else []
    profile[key] = unique_urls([str(item) for item in rows] + values)


def live_peer_urls() -> list[str]:
    peers = read_json(REPORTS / "hive_peers.json", {})
    rows = []
    for key in ["local_node", "peers", "known_peers"]:
        value = peers.get(key) if isinstance(peers, dict) else None
        if isinstance(value, dict):
            rows.append(value)
        elif isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    urls: list[str] = []
    for row in rows:
        urls.append(str(row.get("api_url") or ""))
        urls.append(str(row.get("relay_url") or ""))
    return urls


def local_identity_summary() -> dict[str, Any]:
    identity = read_json(REPORTS / "hive_node_identity.json", {})
    status = read_json(REPORTS / "hive_status.json", {})
    return {
        "node_id": identity.get("node_id") or status.get("node_id"),
        "node_name": identity.get("node_name") or status.get("node_name"),
        "hostname": socket.gethostname(),
        "api_url": status.get("api_url"),
    }


def token_scope(profile: dict[str, Any], *, policy: dict[str, Any], operator_token_scope: str) -> dict[str, Any]:
    tier = str(profile.get("tier") or "private")
    allowed = hive_remote_access.allowed_task_scope(tier)
    if operator_token_scope:
        requested = {item.strip() for item in operator_token_scope.split(",") if item.strip()}
        allowed = [item for item in allowed if item in requested]
    return {
        "tier": tier,
        "remote_task_kinds": allowed,
        "can_request_worker_chunks": tier in {"private", "company"},
        "can_request_teacher": False,
        "can_request_arbitrary_shell": False,
        "quotas": get_path(policy, ["security", "quotas"], {}),
    }


def install_contract(profile: dict[str, Any], *, bundle_path_hint: str) -> dict[str, list[str]]:
    return {
        "macos": [
            "Open ProjectTheseusHive.dmg and launch ProjectTheseusHive.app, or install ProjectTheseusHive.pkg.",
            f"theseus join --invite {bundle_path_hint} --start",
        ],
        "windows": [
            "Run the Project Theseus Windows installer, then choose Join Hive and import the bundle.",
            f"theseus join --invite {bundle_path_hint} --start",
        ],
        "linux_rpi": [
            "Install Python 3, clone or unpack the release payload, then import the bundle.",
            f"python3 scripts/theseus_cli.py join --invite {bundle_path_hint} --start",
        ],
        "iphone_watch": [
            "Open the QR/import URL in the native iPhone app.",
            "The Watch receives the private operator profile from the iPhone through WatchConnectivity.",
        ],
    }


def mobile_app_join_url(bundle: dict[str, Any]) -> str:
    raw = json.dumps(compact_mobile_payload(bundle), separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"theseushive://join?profile={encoded}"


def short_mobile_join_url(bundle: dict[str, Any]) -> str:
    endpoint = shortest_endpoint(bundle)
    token = str(bundle.get("join_token") or "")
    hive_id = str(bundle.get("hive_id") or "")
    if not endpoint or not token:
        return ""
    base = f"theseushive://join?u={quote(endpoint, safe=':.[]')}&t={quote(token, safe='')}"
    with_hive = base + f"&h={quote(hive_id, safe='')}"
    return with_hive if hive_id and len(with_hive.encode("utf-8")) <= 108 else base


def shortest_endpoint(bundle: dict[str, Any]) -> str:
    candidates = [
        str(bundle.get("coordinator_url") or ""),
        *[str(item) for item in bundle.get("node_urls", []) if item],
        *[str(item) for item in bundle.get("coordinator_urls", []) if item],
        *[str(item) for item in bundle.get("relay_urls", []) if item],
    ]
    stripped = [strip_http_for_short_url(item) for item in candidates if item]
    return sorted(stripped, key=lambda item: len(item.encode("utf-8")))[0] if stripped else ""


def strip_http_for_short_url(value: str) -> str:
    parsed = urlparse(value if "://" in value else "http://" + value)
    if not parsed.hostname:
        return value
    port = f":{parsed.port}" if parsed.port else ""
    if parsed.scheme == "http":
        return f"{parsed.hostname}{port}"
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def first_url(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if item:
                return str(item)
    return ""


def compact_mobile_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    endpoints = []
    for endpoint in get_path(bundle, ["roaming", "endpoints"], []) if isinstance(get_path(bundle, ["roaming", "endpoints"], []), list) else []:
        if not isinstance(endpoint, dict) or not endpoint.get("url"):
            continue
        endpoints.append(
            {
                "kind": endpoint.get("kind"),
                "url": endpoint.get("url"),
                "transport": endpoint.get("transport"),
                "priority": endpoint.get("priority"),
            }
        )
    return {
        "policy": "project_theseus_hive_mobile_roaming_profile_v0",
        "hive_id": bundle.get("hive_id"),
        "hive_name": bundle.get("hive_name"),
        "tier": bundle.get("tier"),
        "coordinator_url": bundle.get("coordinator_url", ""),
        "coordinator_urls": bundle.get("coordinator_urls", []),
        "node_urls": bundle.get("node_urls", []),
        "relay_url": bundle.get("relay_url", ""),
        "relay_urls": bundle.get("relay_urls", []),
        "operator_urls": bundle.get("operator_urls", []),
        "roaming": {
            "strategy": "iphone_try_last_good_then_lan_then_private_tunnel_then_https_relay",
            "endpoints": endpoints[:8],
        },
        "join_token": bundle.get("join_token", ""),
    }


def unique_urls(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = normalize_url(str(value or ""))
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def normalize_url(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("theseushive://"):
        return ""
    if "://" not in value:
        value = "http://" + value
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.hostname:
        return ""
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}".rstrip("/")


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
