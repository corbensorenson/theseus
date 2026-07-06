"""Authenticated polling relay for Project Theseus Hive.

LAN multicast does not cross home/workshop/friend networks. This relay provides
a simple rendezvous and task mailbox that nodes can poll from behind NAT. It is
not a public open compute market; public mode is disabled by policy until the
project has signed workers, sandboxing, reputation, and abuse controls.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "hive_policy.json"
STATE_LOCK = threading.Lock()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--state", default="")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    host = args.host or str(get_path(policy, ["relay", "server_host"], "0.0.0.0"))
    port = int(args.port or get_path(policy, ["relay", "server_port"], 8793))
    state_path = ROOT / (args.state or str(get_path(policy, ["relay", "state_path"], "reports/hive_relay_state.json")))
    write_json(
        ROOT / str(get_path(policy, ["relay", "status_path"], "reports/hive_relay_status.json")),
        {
            "policy": "project_theseus_hive_relay_status_v0",
            "created_utc": now(),
            "host": host,
            "port": port,
            "state_path": str(state_path.relative_to(ROOT)).replace("\\", "/"),
            "mobile_operator_ui": bool(get_path(policy, ["relay", "mobile_operator_ui"], True)),
        },
    )
    server = ThreadingHTTPServer((host, port), make_handler(policy, state_path))
    print(f"Project Theseus Hive relay: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def make_handler(policy: dict[str, Any], state_path: Path) -> type[BaseHTTPRequestHandler]:
    class RelayHandler(BaseHTTPRequestHandler):
        server_version = "ProjectTheseusHiveRelay/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/mobile", "/m"}:
                return self.serve_mobile()
            if parsed.path == "/api/hive/relay/status":
                return self.send_json(public_status(policy, state_path))
            if parsed.path == "/api/hive/relay/peers":
                ok, reason, hive_id = self.authorize(parsed)
                if not ok:
                    return self.send_json({"ok": False, "error": reason}, status=403)
                state = load_state(state_path)
                prune_state(policy, state)
                return self.send_json({"ok": True, "hive_id": hive_id, "peers": list((state.get("hives", {}).get(hive_id, {}).get("nodes") or {}).values())})
            if parsed.path == "/api/hive/relay/results":
                ok, reason, hive_id = self.authorize(parsed)
                if not ok:
                    return self.send_json({"ok": False, "error": reason}, status=403)
                query = parse_qs(parsed.query)
                limit = int(first(query, "limit") or 25)
                state = load_state(state_path)
                results = (state.get("hives", {}).get(hive_id, {}).get("results") or [])[-max(1, min(limit, 100)):]
                return self.send_json({"ok": True, "hive_id": hive_id, "results": results})
            if parsed.path == "/api/hive/relay/tasks/poll":
                ok, reason, hive_id = self.authorize(parsed)
                if not ok:
                    return self.send_json({"ok": False, "error": reason}, status=403)
                query = parse_qs(parsed.query)
                node_id = first(query, "node_id")
                if not node_id:
                    return self.send_json({"ok": False, "error": "node_id_required"}, status=400)
                tasks = poll_tasks(policy, state_path, hive_id, node_id)
                return self.send_json({"ok": True, "hive_id": hive_id, "node_id": node_id, "tasks": tasks})
            return self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            payload = self.read_json_body()
            if parsed.path == "/api/hive/relay/register":
                hive_id = str(payload.get("hive_id") or "")
                token = self.headers.get("X-Theseus-Hive-Secret", "") or str(payload.get("join_token") or "")
                ok, reason = authorize_token(policy, hive_id, token)
                if not ok:
                    return self.send_json({"ok": False, "error": reason}, status=403)
                peer = payload.get("peer") if isinstance(payload.get("peer"), dict) else {}
                tier = str(payload.get("tier") or "private")
                result = register_node(policy, state_path, hive_id, peer, tier)
                return self.send_json(result)
            if parsed.path == "/api/hive/relay/tasks":
                hive_id = str(payload.get("hive_id") or "")
                token = self.headers.get("X-Theseus-Hive-Secret", "") or str(payload.get("join_token") or "")
                ok, reason = authorize_token(policy, hive_id, token)
                if not ok:
                    return self.send_json({"ok": False, "error": reason}, status=403)
                result = enqueue_relay_task(policy, state_path, hive_id, payload)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/relay/tasks/result":
                hive_id = str(payload.get("hive_id") or "")
                token = self.headers.get("X-Theseus-Hive-Secret", "") or str(payload.get("join_token") or "")
                ok, reason = authorize_token(policy, hive_id, token)
                if not ok:
                    return self.send_json({"ok": False, "error": reason}, status=403)
                result = record_task_result(state_path, hive_id, payload)
                return self.send_json(result)
            if parsed.path == "/api/hive/relay/chat":
                hive_id = str(payload.get("hive_id") or "")
                token = self.headers.get("X-Theseus-Hive-Secret", "") or str(payload.get("join_token") or "")
                ok, reason = authorize_token(policy, hive_id, token)
                if not ok:
                    return self.send_json({"ok": False, "error": reason}, status=403)
                result = enqueue_mobile_chat(policy, state_path, hive_id, payload)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            return self.send_error(HTTPStatus.NOT_FOUND)

        def authorize(self, parsed: Any) -> tuple[bool, str, str]:
            query = parse_qs(parsed.query)
            hive_id = first(query, "hive_id")
            token = self.headers.get("X-Theseus-Hive-Secret", "") or first(query, "token")
            ok, reason = authorize_token(policy, hive_id, token)
            return ok, reason, hive_id

        def serve_mobile(self) -> None:
            data = mobile_html().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            if len(raw.encode("utf-8")) > int(get_path(policy, ["relay", "max_payload_bytes"], 65536)):
                return {}
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}

        def send_json(self, payload: Any, status: int = 200) -> None:
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return RelayHandler


def public_status(policy: dict[str, Any], state_path: Path) -> dict[str, Any]:
    state = load_state(state_path)
    prune_state(policy, state)
    hives = state.get("hives") if isinstance(state.get("hives"), dict) else {}
    return {
        "policy": "project_theseus_hive_relay_public_status_v0",
        "created_utc": now(),
        "hive_count": len(hives),
        "node_count": sum(len((hive.get("nodes") or {})) for hive in hives.values() if isinstance(hive, dict)),
        "pending_task_count": sum(len(hive.get("tasks") or []) for hive in hives.values() if isinstance(hive, dict)),
        "mobile_operator_ui": bool(get_path(policy, ["relay", "mobile_operator_ui"], True)),
    }


def register_node(policy: dict[str, Any], state_path: Path, hive_id: str, peer: dict[str, Any], tier: str) -> dict[str, Any]:
    if not hive_id:
        return {"ok": False, "error": "hive_id_required"}
    if tier == "public":
        return {"ok": False, "error": "public_hive_disabled"}
    if not peer.get("node_id"):
        return {"ok": False, "error": "node_id_required"}
    with STATE_LOCK:
        state = load_state(state_path)
        hive = state.setdefault("hives", {}).setdefault(hive_id, {"nodes": {}, "tasks": [], "results": []})
        peer["tier"] = tier
        peer["last_seen_utc"] = now()
        hive["nodes"][str(peer["node_id"])] = peer
        write_json(state_path, state)
    return {"ok": True, "hive_id": hive_id, "node_id": peer.get("node_id"), "peer_count": len(hive["nodes"])}


def enqueue_relay_task(policy: dict[str, Any], state_path: Path, hive_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    tier = str(payload.get("tier") or "private")
    kind = str(payload.get("kind") or "")
    target_node_id = str(payload.get("target_node_id") or "")
    allowed = set(get_path(policy, ["relay", "allowed_remote_task_kinds_by_tier", tier], []))
    if not hive_id:
        return {"ok": False, "error": "hive_id_required"}
    if kind not in allowed:
        return {"ok": False, "error": "task_not_allowed_for_tier", "kind": kind, "tier": tier}
    if kind in set(get_path(policy, ["security", "forbidden_remote_task_kinds"], [])):
        return {"ok": False, "error": "forbidden_task_kind", "kind": kind}
    task = {
        "task_id": f"relay_task_{int(time.time() * 1000)}_{secrets.token_hex(4)}",
        "kind": kind,
        "target_node_id": target_node_id,
        "payload": payload.get("task_payload") if isinstance(payload.get("task_payload"), dict) else {},
        "tier": tier,
        "status": "queued",
        "created_utc": now(),
    }
    with STATE_LOCK:
        state = load_state(state_path)
        hive = state.setdefault("hives", {}).setdefault(hive_id, {"nodes": {}, "tasks": [], "results": []})
        hive.setdefault("tasks", []).append(task)
        write_json(state_path, state)
    return {"ok": True, "hive_id": hive_id, "task": task}


def enqueue_mobile_chat(policy: dict[str, Any], state_path: Path, hive_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = load_state(state_path)
    hive = state.get("hives", {}).get(hive_id, {})
    nodes = list((hive.get("nodes") or {}).values())
    gateway = best_chat_gateway(nodes)
    if not gateway:
        return {"ok": False, "error": "no_chat_gateway_available"}
    return enqueue_relay_task(
        policy,
        state_path,
        hive_id,
        {
            "tier": str(payload.get("tier") or "private"),
            "kind": str(get_path(policy, ["relay", "chat_task_kind"], "checkpoint_chat")),
            "target_node_id": gateway.get("node_id"),
            "task_payload": {
                "checkpoint_id": str(payload.get("checkpoint_id") or "live"),
                "prompt": str(payload.get("prompt") or "Summarize Project Theseus status."),
            },
        },
    )


def poll_tasks(policy: dict[str, Any], state_path: Path, hive_id: str, node_id: str) -> list[dict[str, Any]]:
    now_ts = time.time()
    ttl = int(get_path(policy, ["relay", "task_ttl_seconds"], 3600))
    claimed: list[dict[str, Any]] = []
    with STATE_LOCK:
        state = load_state(state_path)
        hive = state.setdefault("hives", {}).setdefault(hive_id, {"nodes": {}, "tasks": [], "results": []})
        kept = []
        for task in hive.get("tasks", []):
            created = parse_time(task.get("created_utc"))
            if created and now_ts - created > ttl:
                task["status"] = "expired"
                hive.setdefault("results", []).append(task)
                continue
            target = str(task.get("target_node_id") or "")
            if task.get("status") == "queued" and (not target or target == node_id):
                task["status"] = "claimed"
                task["claimed_by"] = node_id
                task["claimed_utc"] = now()
                claimed.append(task)
            else:
                kept.append(task)
        hive["tasks"] = kept
        write_json(state_path, state)
    return claimed


def record_task_result(state_path: Path, hive_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if not result.get("task_id"):
        return {"ok": False, "error": "task_id_required"}
    result["reported_utc"] = now()
    result["reported_by"] = payload.get("node_id")
    with STATE_LOCK:
        state = load_state(state_path)
        hive = state.setdefault("hives", {}).setdefault(hive_id, {"nodes": {}, "tasks": [], "results": []})
        hive.setdefault("results", []).append(result)
        hive["results"] = hive["results"][-200:]
        write_json(state_path, state)
    return {"ok": True, "task_id": result.get("task_id")}


def prune_state(policy: dict[str, Any], state: dict[str, Any]) -> None:
    stale_after = int(get_path(policy, ["relay", "node_stale_after_seconds"], 90))
    now_ts = time.time()
    for hive in (state.get("hives") or {}).values():
        if not isinstance(hive, dict):
            continue
        nodes = hive.get("nodes") if isinstance(hive.get("nodes"), dict) else {}
        for node_id, node in list(nodes.items()):
            seen = parse_time(node.get("last_seen_utc"))
            if seen and now_ts - seen > stale_after:
                nodes.pop(node_id, None)


def best_chat_gateway(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    def score(node: dict[str, Any]) -> float:
        total = 0.0
        for cap in node.get("capabilities") or []:
            if cap.get("id") == "checkpoint_chat_gateway":
                total += 1.0
            total += float(cap.get("score") or 0.0) * 0.1
        return total

    candidates = [node for node in nodes if any(cap.get("id") == "checkpoint_chat_gateway" for cap in node.get("capabilities") or [])]
    return max(candidates, key=score) if candidates else {}


def authorize_token(policy: dict[str, Any], hive_id: str, token: str) -> tuple[bool, str]:
    if not hive_id:
        return False, "hive_id_required"
    revocations = read_json(ROOT / str(get_path(policy, ["security", "revocations_path"], "configs/hive_revocations.local.json")), {})
    revoked = set(str(item) for item in revocations.get("revoked_subjects", []) if item)
    revoked.update(str(item) for item in revocations.get("revoked_invite_ids", []) if item)
    if hive_id in revoked or token in revoked:
        return False, "hive_or_token_revoked"
    expected = [value for value in [os.environ.get("THESEUS_HIVE_SECRET", ""), local_join_token(hive_id)] if value]
    if token and token in expected:
        return True, "secret_ok"
    return False, "valid_hive_secret_required"


def local_join_token(hive_id: str) -> str:
    path = ROOT / "configs" / "hive_join.local.json"
    if not path.exists():
        return ""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    if not isinstance(value, dict) or str(value.get("hive_id") or "") != hive_id:
        return ""
    return str(value.get("join_token") or "")


def mobile_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project Theseus Hive</title>
  <style>
    body { margin:0; font-family: system-ui, sans-serif; background:#101214; color:#e8edf2; }
    main { max-width:720px; margin:0 auto; padding:18px; display:grid; gap:12px; }
    input, textarea, button { width:100%; box-sizing:border-box; border:1px solid #303942; border-radius:8px; background:#1f252b; color:#e8edf2; padding:10px; font:inherit; }
    button { cursor:pointer; }
    pre { white-space:pre-wrap; overflow-wrap:anywhere; background:#0c0e10; border:1px solid #303942; border-radius:8px; padding:10px; }
    .muted { color:#98a4af; font-size:13px; }
  </style>
</head>
<body>
<main>
  <h1>Project Theseus Hive</h1>
  <p class="muted">Mobile operator client. Use a private/friends invite token.</p>
  <input id="hive" placeholder="hive id">
  <input id="token" placeholder="invite token / password" type="password">
  <button id="status">Status</button>
  <textarea id="prompt" rows="4" placeholder="Ask the live/checkpoint system..."></textarea>
  <button id="chat">Send Chat Task</button>
  <pre id="out">Ready.</pre>
</main>
<script>
const out = document.getElementById('out');
const hive = document.getElementById('hive');
const token = document.getElementById('token');
const params = new URLSearchParams(location.search);
hive.value = params.get('hive_id') || params.get('h') || localStorage.hive_id || '';
token.value = params.get('token') || params.get('t') || localStorage.hive_token || '';
function save(){ localStorage.hive_id=hive.value; localStorage.hive_token=token.value; }
async function api(url, options={}) {
  save();
  options.headers = Object.assign({'Content-Type':'application/json','X-Theseus-Hive-Secret': token.value}, options.headers || {});
  const res = await fetch(url, options);
  const text = await res.text();
  try { return JSON.parse(text); } catch { return {ok:false, body:text}; }
}
document.getElementById('status').onclick = async () => {
  const data = await api('/api/hive/relay/peers?hive_id=' + encodeURIComponent(hive.value));
  out.textContent = JSON.stringify(data, null, 2);
};
document.getElementById('chat').onclick = async () => {
  const data = await api('/api/hive/relay/chat', {method:'POST', body: JSON.stringify({hive_id:hive.value, join_token:token.value, prompt:document.getElementById('prompt').value})});
  out.textContent = JSON.stringify(data, null, 2);
  const taskId = data && data.task && data.task.task_id;
  if (taskId) {
    for (let i = 0; i < 20; i++) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const results = await api('/api/hive/relay/results?hive_id=' + encodeURIComponent(hive.value) + '&limit=20');
      const match = (results.results || []).find((row) => row.task_id === taskId);
      if (match) {
        out.textContent = JSON.stringify(match, null, 2);
        break;
      }
    }
  }
};
</script>
</body>
</html>"""


def first(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key) or []
    return values[0] if values else ""


def parse_time(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"policy": "project_theseus_hive_relay_state_v0", "hives": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"policy": "project_theseus_hive_relay_state_v0", "hives": {}}
    return value if isinstance(value, dict) else {"policy": "project_theseus_hive_relay_state_v0", "hives": {}}


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
