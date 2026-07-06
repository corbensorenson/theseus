"""Cross-node storage helper calls for the Hive operator API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError
from urllib.parse import urlencode

import hive_storage
from hive_node_common import now
from hive_node_federation import hive_id, join_token, shared_secret
from hive_node_identity import load_identity
from hive_node_peer_registry import current_peers, find_local_ip


def storage_peer_browse(policy: dict[str, Any], *, node_id: str, share_id: str, rel_path: str, limit: int) -> dict[str, Any]:
    if is_local_storage_node(policy, node_id):
        result = hive_storage.browse_share(policy=policy, share_id=share_id, rel_path=rel_path, limit=limit)
        return storage_with_node(result, local_storage_peer(policy))
    peer = storage_peer_by_node_id(policy, node_id)
    if not peer:
        return {"ok": False, "error": "storage_peer_not_found", "node_id": node_id}
    result = fetch_peer_storage_json(
        policy,
        peer,
        "/api/hive/storage/browse",
        {"share_id": share_id, "path": rel_path, "limit": str(limit)},
    )
    return storage_with_node(result, peer)


def storage_peer_file_payload(policy: dict[str, Any], *, node_id: str, share_id: str, rel_path: str) -> dict[str, Any]:
    if is_local_storage_node(policy, node_id):
        result = hive_storage.read_file_payload(policy=policy, share_id=share_id, rel_path=rel_path)
        return storage_with_node(result, local_storage_peer(policy))
    peer = storage_peer_by_node_id(policy, node_id)
    if not peer:
        return {"ok": False, "error": "storage_peer_not_found", "node_id": node_id}
    result = fetch_peer_storage_json(
        policy,
        peer,
        "/api/hive/storage/file",
        {"share_id": share_id, "path": rel_path},
    )
    return storage_with_node(result, peer)


def storage_peer_file_bytes(policy: dict[str, Any], *, node_id: str, share_id: str, rel_path: str) -> dict[str, Any]:
    if is_local_storage_node(policy, node_id):
        result = hive_storage.read_file_bytes(policy=policy, share_id=share_id, rel_path=rel_path)
        return storage_with_node(result, local_storage_peer(policy))
    peer = storage_peer_by_node_id(policy, node_id)
    if not peer:
        return {"ok": False, "error": "storage_peer_not_found", "node_id": node_id}
    result = fetch_peer_storage_bytes(
        policy,
        peer,
        "/api/hive/storage/file",
        {"share_id": share_id, "path": rel_path, "raw": "1"},
    )
    return storage_with_node(result, peer)


def local_storage_peer(policy: dict[str, Any]) -> dict[str, Any]:
    identity = load_identity(policy)
    port = int(((policy.get("node") or {}).get("http_port")) or 8791)
    return {
        "node_id": identity.get("node_id"),
        "node_name": identity.get("node_name"),
        "api_url": f"http://{find_local_ip()}:{port}",
        "is_local": True,
    }


def is_local_storage_node(policy: dict[str, Any], node_id: str) -> bool:
    normalized = str(node_id or "local")
    return normalized in {"", "local", str(load_identity(policy).get("node_id") or "")}


def storage_peer_by_node_id(policy: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    expected_hive = hive_id(policy)
    for peer in current_peers(policy):
        if str(peer.get("node_id") or "") != str(node_id or ""):
            continue
        if str(peer.get("hive_id") or expected_hive) != expected_hive:
            return None
        if not str(peer.get("api_url") or "").startswith(("http://", "https://")):
            return None
        return peer
    return None


def storage_with_node(result: dict[str, Any], peer: dict[str, Any]) -> dict[str, Any]:
    return {
        **result,
        "node": {
            "node_id": peer.get("node_id"),
            "node_name": peer.get("node_name"),
            "api_url": peer.get("api_url"),
            "is_local": bool(peer.get("is_local")),
        },
    }


def fetch_peer_storage_json(policy: dict[str, Any], peer: dict[str, Any], endpoint: str, query: dict[str, str]) -> dict[str, Any]:
    url = str(peer.get("api_url") or "").rstrip("/") + endpoint + "?" + urlencode(query)
    req = urlrequest.Request(url, headers={"Accept": "application/json"}, method="GET")
    secret = join_token(policy) or shared_secret(policy)
    if secret:
        req.add_header("X-Theseus-Hive-Secret", secret)
    try:
        with urlrequest.urlopen(req, timeout=30) as response:  # noqa: S310 - private Hive peer selected from trusted peer table.
            raw = response.read().decode("utf-8")
    except URLError as exc:
        return {"ok": False, "error": "peer_storage_request_failed", "message": str(exc), "peer_url": peer.get("api_url")}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "peer_storage_non_json_response", "body": raw[:300], "peer_url": peer.get("api_url")}
    if not isinstance(value, dict):
        return {"ok": False, "error": "peer_storage_unexpected_response", "peer_url": peer.get("api_url")}
    return value


def fetch_peer_storage_bytes(policy: dict[str, Any], peer: dict[str, Any], endpoint: str, query: dict[str, str]) -> dict[str, Any]:
    url = str(peer.get("api_url") or "").rstrip("/") + endpoint + "?" + urlencode(query)
    req = urlrequest.Request(url, headers={"Accept": "application/octet-stream"}, method="GET")
    secret = join_token(policy) or shared_secret(policy)
    if secret:
        req.add_header("X-Theseus-Hive-Secret", secret)
    try:
        with urlrequest.urlopen(req, timeout=45) as response:  # noqa: S310 - private Hive peer selected from trusted peer table.
            data = response.read()
            content_type = str(response.headers.get("Content-Type") or "application/octet-stream")
    except URLError as exc:
        return {"ok": False, "error": "peer_storage_file_failed", "message": str(exc), "peer_url": peer.get("api_url")}
    return {
        "ok": True,
        "policy": "project_theseus_hive_storage_peer_file_v0",
        "created_utc": now(),
        "content_type": content_type,
        "name": Path(str(query.get("path") or "hive-file")).name or "hive-file",
        "size_bytes": len(data),
        "content_bytes": data,
    }
