"""Relay and coordinator sync helpers for the Hive node runtime."""

from __future__ import annotations

import json
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError
from urllib.parse import urlencode

from hive_node_artifacts import bundle_result_artifacts
from hive_node_common import append_jsonl, event, get_path, task_ledger_path
from hive_node_federation import coordinator_urls, hive_id, hive_tier, join_token, relay_url, shared_secret
from hive_node_identity import load_identity
from hive_node_peer_registry import accept_heartbeat_response, accept_peer, mark_peer_url_failure, peer_from_status


def register_with_relay(policy: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    url = relay_url(policy)
    if not url:
        return {"ok": False, "error": "relay_not_configured"}
    payload = {
        "hive_id": hive_id(policy),
        "join_token": join_token(policy),
        "tier": hive_tier(policy),
        "peer": peer_from_status(status),
    }
    return relay_request(policy, "POST", f"{url.rstrip('/')}/api/hive/relay/register", payload)


def sync_static_coordinator(policy: dict[str, Any], status: dict[str, Any]) -> None:
    urls = coordinator_urls(policy)
    timeout = float(get_path(policy, ["discovery", "coordinator_sync_timeout_seconds"], 8))
    for url in urls:
        base = url.rstrip("/")
        local_api = str(status.get("api_url") or "").rstrip("/")
        if not base or base == local_api or base.startswith("http://127.0.0.1:"):
            continue
        remote_peer: dict[str, Any] = {}
        try:
            with urlrequest.urlopen(base + "/api/hive/status", timeout=timeout) as response:  # noqa: S310 - private Hive coordinator.
                remote = json.loads(response.read().decode("utf-8"))
            if isinstance(remote, dict):
                remote_peer = peer_from_status(remote)
        except (OSError, URLError, json.JSONDecodeError) as exc:
            mark_peer_url_failure(policy, base, str(exc))
        try:
            body = json.dumps({"peer": peer_from_status(status)}).encode("utf-8")
            req = urlrequest.Request(
                base + "/api/hive/heartbeat",
                data=body,
                headers={"Content-Type": "application/json", "X-Theseus-Hive-Secret": join_token(policy) or shared_secret(policy)},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=timeout) as response:  # noqa: S310 - private Hive coordinator.
                heartbeat = json.loads(response.read().decode("utf-8"))
            if remote_peer:
                accept_peer(policy, remote_peer, source="coordinator_heartbeat", trusted=True)
            accept_heartbeat_response(policy, heartbeat, source="coordinator_heartbeat")
        except (OSError, URLError, json.JSONDecodeError) as exc:
            mark_peer_url_failure(policy, base, str(exc))


def fetch_relay_peers(policy: dict[str, Any]) -> list[dict[str, Any]]:
    url = relay_url(policy)
    if not url:
        return []
    query = urlencode({"hive_id": hive_id(policy)})
    result = relay_request(policy, "GET", f"{url.rstrip('/')}/api/hive/relay/peers?{query}", None)
    peers = result.get("peers") if isinstance(result, dict) else []
    return peers if isinstance(peers, list) else []


def poll_relay_tasks(policy: dict[str, Any], status: dict[str, Any]) -> list[dict[str, Any]]:
    url = relay_url(policy)
    if not url:
        return []
    query = urlencode({"hive_id": hive_id(policy), "node_id": status.get("node_id")})
    result = relay_request(policy, "GET", f"{url.rstrip('/')}/api/hive/relay/tasks/poll?{query}", None)
    tasks = result.get("tasks") if isinstance(result, dict) else []
    return tasks if isinstance(tasks, list) else []


def post_relay_result(policy: dict[str, Any], task: dict[str, Any], result: dict[str, Any]) -> None:
    url = str(task.get("relay_url") or relay_url(policy))
    if not url:
        return
    relay_request(
        policy,
        "POST",
        f"{url.rstrip('/')}/api/hive/relay/tasks/result",
        {
            "hive_id": task.get("hive_id") or hive_id(policy),
            "join_token": join_token(policy),
            "node_id": load_identity(policy)["node_id"],
            "result": bundle_result_artifacts(policy, result),
        },
    )


def relay_request(policy: dict[str, Any], method: str, url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    data = json.dumps(payload or {}).encode("utf-8") if method == "POST" else None
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    token = join_token(policy)
    if token:
        req.add_header("X-Theseus-Hive-Secret", token)
    try:
        with urlrequest.urlopen(req, timeout=10) as response:  # noqa: S310 - user-configured private relay endpoint.
            raw = response.read().decode("utf-8")
    except URLError as exc:
        append_jsonl(task_ledger_path(policy), event("relay_error", {"url": url, "error": str(exc)}))
        return {"ok": False, "error": str(exc)}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_relay_response"}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_relay_response"}
