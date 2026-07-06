"""Project Theseus Hive node runtime.

This is the cross-platform runtime an eventual Windows/macOS app installer runs
in the background. It keeps a small local HTTP API, advertises node capability
on the LAN by multicast, records peer status, and executes only registered safe
task kinds from configs/hive_policy.json.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import queue
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import license_manager
import compute_market
import hive_security
import openai_compat_server
import theseus_runtime
import update_manager
import hive_version_manager
import hive_storage
import hive_network_doctor
import hive_remote_control
import hive_rented_compute
import hive_voice_following
import hive_spatial
import hive_users
from hive_node_peer_registry import accept_bonjour_peers, accept_heartbeat_response, accept_peer, announce_multicast, bonjour_enabled, bonjour_register_command, current_peers, ensure_peer_registry_loaded, known_peers, listen_multicast, mark_peer_url_failure, peer_from_status, peer_registry_path, peer_state_counts, probe_known_peers, scan_bonjour
from hive_node_operator_runtime import operator_solo_learning_summary, operator_training_summary, operator_utilization_summary
from hive_node_operator_targets import node_can_run_task, operator_accelerator_summary, operator_targets
from hive_node_slots import acquire_slot, init_slot_state, local_task_support, release_slot, slots_snapshot
from hive_node_relay import fetch_relay_peers, poll_relay_tasks, post_relay_result, register_with_relay, sync_static_coordinator
from hive_node_storage_peer import storage_peer_browse, storage_peer_file_bytes, storage_peer_file_payload
from hive_node_update_api import hive_version_catalog, startup_update_check, update_apply_from_payload, update_check_from_payload, update_configure_from_payload, update_policy
from hive_node_operator_profile import compact_ios_profile_url, compact_operator_profile, ios_profile_url, operator_revocation_hint, qr_svg_for_url
from hive_node_resources import classify_capabilities, probe_resources, resource_slots, worker_thread_count
from hive_node_notifications import acknowledge_operator_notifications, operator_network_summary, operator_notification_summary, operator_notifications
from hive_node_compact_reports import compact_hive_version, compact_license, compact_market, compact_operator_overnight, compact_promoted, compact_remote_control, compact_rented_compute, compact_runtime, compact_storage, compact_training_execution, compact_update_status, compact_utilization
from hive_node_identity import load_identity
from hive_node_federation import apply_runtime_join_overrides, coordinator_urls, hive_id, hive_tier, join_token, relay_url, shared_secret, unique_nonempty
from hive_node_common import append_jsonl, command_available, display_path, event, get_path, is_loopback, now, parse_json_arg, read_json, read_jsonl_tail, remove_if_exists, report_path, task_ledger_path, to_float, write_json, write_text
from hive_node_artifacts import artifact_index, bundle_result_artifacts, materialize_output_artifacts, read_artifact_payload
from hive_node_operator_ui import mobile_operator_html, operator_icon_png, operator_webmanifest
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib import request as urlrequest
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "hive_policy.json"
REPORTS = ROOT / "reports"
TASKS: "queue.Queue[dict[str, Any]]" = queue.Queue()
STOP = threading.Event()
STATUS_CACHE_LOCK = threading.Lock()
STATUS_CACHE: dict[int, tuple[float, dict[str, Any]]] = {}
OPERATOR_STATUS_CACHE_LOCK = threading.Lock()
OPERATOR_STATUS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    probe = sub.add_parser("probe")
    probe.add_argument("--out", default="")
    probe.add_argument("--peers-out", default="")

    discover = sub.add_parser("discover")
    discover.add_argument("--seconds", type=float, default=5.0)
    discover.add_argument("--out", default="")

    daemon = sub.add_parser("daemon")
    daemon.add_argument("--host", default="")
    daemon.add_argument("--port", type=int, default=0)
    daemon.add_argument("--no-discovery", action="store_true")
    daemon.add_argument("--no-worker", action="store_true")
    daemon.add_argument("--relay-url", default="")
    daemon.add_argument("--hive-id", default="")
    daemon.add_argument("--tier", default="")
    daemon.add_argument("--out", default="")

    submit = sub.add_parser("submit")
    submit.add_argument("--peer-url", required=True)
    submit.add_argument("--kind", required=True)
    submit.add_argument("--payload-json", default="{}")
    submit.add_argument("--out", default="")

    operator = sub.add_parser("operator-status")
    operator.add_argument("--out", default="reports/hive_operator_status.json")

    audit = sub.add_parser("operator-governance-audit")
    audit.add_argument("--out", default="reports/hive_operator_governance_audit.json")
    audit.add_argument("--request-kind", default="audit_export")
    audit.add_argument("--artifact-ref", action="append", default=[])

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command == "probe" or args.command is None:
        status = build_status(policy, http_port=0)
        write_json(report_path(policy, "status_path", "reports/hive_status.json", args.out), status)
        write_json(report_path(policy, "peers_path", "reports/hive_peers.json", args.peers_out), peers_report(policy))
        print(json.dumps(status, indent=2))
        return 0
    if args.command == "discover":
        status = build_status(policy, http_port=0)
        run_discovery(policy, status, seconds=args.seconds)
        report = peers_report(policy)
        write_json(report_path(policy, "peers_path", "reports/hive_peers.json", args.out), report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "daemon":
        return run_daemon(args, policy)
    if args.command == "submit":
        payload = parse_json_arg(args.payload_json)
        result = submit_task(policy, args.peer_url, args.kind, payload)
        if args.out:
            write_json(ROOT / args.out, result)
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 2
    if args.command == "operator-status":
        auth = {"ok": True, "token_kind": "loopback", "user_id": "local_cli", "role": "owner"}
        report = operator_status(policy, auth)
        if args.out:
            write_json(ROOT / args.out, report)
        summary = {
            "ok": True,
            "report": str((ROOT / args.out).relative_to(ROOT)).replace("\\", "/") if args.out else "reports/hive_operator_status.json",
            "teacher_governance": report.get("teacher_governance", {}),
        }
        print(json.dumps(summary, indent=2))
        return 0 if get_path(report, ["teacher_governance", "operator_visible"], False) else 2
    if args.command == "operator-governance-audit":
        auth = {"ok": True, "token_kind": "loopback", "user_id": "local_cli", "role": "owner"}
        payload = {"request_kind": args.request_kind, "artifact_refs": args.artifact_ref}
        report = operator_governance_audit(policy, payload, auth, out_path=ROOT / args.out)
        print(json.dumps({"ok": report.get("ok"), "trigger_state": report.get("trigger_state"), "summary": report.get("summary")}, indent=2))
        return 0 if report.get("ok") and report.get("trigger_state") in {"GREEN", "YELLOW"} else 2
    parser.print_help()
    return 2


def run_daemon(args: argparse.Namespace, policy: dict[str, Any]) -> int:
    apply_runtime_join_overrides(policy, args)
    host = args.host or str(get_path(policy, ["node", "http_host"], "0.0.0.0"))
    port = int(args.port or get_path(policy, ["node", "http_port"], 8791))
    ensure_peer_registry_loaded(policy)
    status = build_status(policy, http_port=port)
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(report_path(policy, "status_path", "reports/hive_status.json", args.out), status)
    append_jsonl(task_ledger_path(policy), event("daemon_start", {"node_id": status["node_id"], "api_url": status["api_url"]}))
    threading.Thread(target=startup_update_check, daemon=True).start()

    threads: list[threading.Thread] = []
    if not args.no_discovery and get_path(policy, ["discovery", "enabled"], True):
        threads.append(threading.Thread(target=discovery_loop, args=(policy, status), daemon=True))
        if bonjour_enabled(policy):
            threads.append(threading.Thread(target=bonjour_advertise_loop, args=(policy, port), daemon=True))
    if relay_url(policy):
        threads.append(threading.Thread(target=relay_loop, args=(policy, port), daemon=True))
    if not args.no_worker:
        init_slot_state(status)
        for idx in range(worker_thread_count(status, policy)):
            threads.append(threading.Thread(target=worker_loop, args=(policy, idx), daemon=True))
    for thread in threads:
        thread.start()

    server = ThreadingHTTPServer((host, port), make_handler(policy))
    print(f"Project Theseus Hive node: http://{status['listen_host']}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        STOP.set()
        server.server_close()
        append_jsonl(task_ledger_path(policy), event("daemon_stop", {"node_id": status["node_id"]}))
    return 0


def make_handler(policy: dict[str, Any]) -> type[BaseHTTPRequestHandler]:
    class HiveHandler(BaseHTTPRequestHandler):
        server_version = "ProjectTheseusHive/0.1"

        def auth(self, action: str = "status", task_kind: str = "", query: str = "") -> dict[str, Any]:
            return hive_users.authorize(
                policy,
                self.client_address[0],
                self.headers.get("X-Theseus-Hive-Secret", ""),
                query,
                action=action,
                task_kind=task_kind,
            )

        def deny_auth(self, auth: dict[str, Any]) -> None:
            self.send_json({"ok": False, "error": auth.get("error") or auth.get("reason") or "access_denied"}, status=403)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/mobile", "/operator", "/m"}:
                return self.send_text(mobile_operator_html(), content_type="text/html; charset=utf-8")
            if parsed.path == "/operator.webmanifest":
                return self.send_text(operator_webmanifest(), content_type="application/manifest+json; charset=utf-8")
            if parsed.path == "/api/hive/health":
                return self.send_json(build_health(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791))))
            if parsed.path in {"/operator-icon-180.png", "/operator-icon-1024.png"}:
                icon = operator_icon_png(parsed.path)
                if not icon:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                return self.send_bytes(icon, content_type="image/png", filename=parsed.path.rsplit("/", 1)[-1])
            if parsed.path in {"/", "/api/hive/status", "/api/hive/node"}:
                return self.send_json(cached_build_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791))))
            if parsed.path == "/api/hive/vcm/status":
                auth = self.auth("operator_status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(operator_vcm_summary())
            if parsed.path == "/api/hive/auth/status":
                auth = self.auth("status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(build_auth_status(policy, auth, http_port=int(get_path(policy, ["node", "http_port"], 8791))))
            if parsed.path == "/api/hive/operator/status":
                auth = self.auth("operator_status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(cached_operator_status(policy, auth))
            if parsed.path == "/api/hive/operator/roaming-profile":
                auth = self.auth("operator_status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                query = parse_qs(parsed.query)
                include_token = str((query.get("include_token") or query.get("with_token") or ["0"])[0]).lower() in {"1", "true", "yes"}
                provided_token = hive_users.token_from_request(self.headers.get("X-Theseus-Hive-Secret", ""), parsed.query)
                return self.send_json(operator_roaming_profile(policy, auth, provided_token=provided_token, include_token=include_token))
            if parsed.path == "/api/hive/operator/notifications":
                auth = self.auth("operator_status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                query = parse_qs(parsed.query)
                limit = int(to_float((query.get("limit") or ["30"])[0]) or 30)
                since = (query.get("since") or [""])[0]
                return self.send_json(operator_notifications(policy, auth, limit=limit, since=since, write_report=True))
            if parsed.path == "/api/hive/operator/governance-audit":
                auth = self.auth("operator_status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                query = parse_qs(parsed.query)
                artifact_refs = [str(item) for item in query.get("artifact_ref", []) if str(item or "")]
                payload = {
                    "request_kind": (query.get("request_kind") or ["audit_export"])[0],
                    "artifact_refs": artifact_refs,
                    "http_method": "GET",
                }
                return self.send_json(operator_governance_audit(policy, payload, auth))
            if parsed.path == "/api/hive/network-doctor":
                auth = self.auth("operator_status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                query = parse_qs(parsed.query)
                timeout = to_float((query.get("timeout") or ["8.0"])[0]) or 8.0
                return self.send_json(
                    hive_network_doctor.doctor_report(
                        policy=policy,
                        timeout=max(0.25, min(10.0, float(timeout))),
                        write_report=True,
                    )
                )
            if parsed.path == "/api/hive/update-catalog":
                hive_catalog = hive_version_catalog()
                return self.send_json(hive_catalog if hive_catalog.get("offers") else update_manager.public_catalog(update_policy()))
            if parsed.path == "/api/hive/version":
                return self.send_json(hive_version_manager.status_report(write_report=True))
            if parsed.path == "/api/hive/installer-artifacts":
                auth = self.auth("artifact_sync", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(hive_version_manager.installer_artifacts_report(write_report=True))
            if parsed.path == "/api/hive/update-status":
                auth = self.auth("status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(update_manager.status_report(policy=update_policy(), write_report=True))
            if parsed.path == "/api/hive/peers":
                return self.send_json(peers_report(policy))
            if parsed.path == "/api/hive/tasks":
                return self.send_json(tasks_report(policy))
            if parsed.path == "/api/hive/artifacts":
                auth = self.auth("artifact_sync", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                query = parse_qs(parsed.query)
                limit = int((query.get("limit") or ["200"])[0])
                return self.send_json(artifact_index(policy, limit=limit))
            if parsed.path == "/api/hive/artifact":
                auth = self.auth("artifact_sync", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                query = parse_qs(parsed.query)
                rel_path = (query.get("path") or [""])[0]
                return self.send_json(read_artifact_payload(policy, rel_path), status=200)
            if parsed.path == "/api/hive/storage/status":
                auth = self.auth("storage", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(hive_storage.status_report(policy=policy, write_report=True))
            if parsed.path == "/api/hive/storage/index":
                auth = self.auth("storage", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                query = parse_qs(parsed.query)
                limit = int((query.get("limit") or ["500"])[0])
                return self.send_json(hive_storage.index_report(policy=policy, limit=limit, write_report=True))
            if parsed.path == "/api/hive/storage/browse":
                auth = self.auth("storage", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                query = parse_qs(parsed.query)
                share_id = (query.get("share_id") or [""])[0]
                if not hive_users.storage_share_allowed(auth, share_id):
                    return self.send_json({"ok": False, "error": "storage_share_scope_denied", "share_id": share_id}, status=403)
                rel_path = (query.get("path") or [""])[0]
                limit = int((query.get("limit") or ["200"])[0])
                result = hive_storage.browse_share(policy=policy, share_id=share_id, rel_path=rel_path, limit=limit)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/storage/file":
                query = parse_qs(parsed.query)
                share_id = (query.get("share_id") or [""])[0]
                rel_path = (query.get("path") or [""])[0]
                raw = (query.get("raw") or query.get("download") or [""])[0] in {"1", "true", "yes"}
                auth = self.auth("storage_file" if raw else "storage", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                if not hive_users.storage_share_allowed(auth, share_id):
                    return self.send_json({"ok": False, "error": "storage_share_scope_denied", "share_id": share_id}, status=403)
                if raw:
                    result = hive_storage.read_file_bytes(policy=policy, share_id=share_id, rel_path=rel_path)
                    if not result.get("ok"):
                        return self.send_json(result, status=400)
                    return self.send_bytes(
                        result.pop("content_bytes"),
                        content_type=str(result.get("content_type") or "application/octet-stream"),
                        filename=str(result.get("name") or "hive-file"),
                        attachment=(query.get("download") or [""])[0] in {"1", "true", "yes"},
                    )
                result = hive_storage.read_file_payload(policy=policy, share_id=share_id, rel_path=rel_path)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/storage/peer/browse":
                auth = self.auth("storage", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                query = parse_qs(parsed.query)
                node_id = (query.get("node_id") or ["local"])[0]
                share_id = (query.get("share_id") or [""])[0]
                if not hive_users.storage_share_allowed(auth, share_id):
                    return self.send_json({"ok": False, "error": "storage_share_scope_denied", "share_id": share_id}, status=403)
                rel_path = (query.get("path") or [""])[0]
                limit = int((query.get("limit") or ["200"])[0])
                result = storage_peer_browse(policy, node_id=node_id, share_id=share_id, rel_path=rel_path, limit=limit)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/storage/peer/file":
                query = parse_qs(parsed.query)
                node_id = (query.get("node_id") or ["local"])[0]
                share_id = (query.get("share_id") or [""])[0]
                rel_path = (query.get("path") or [""])[0]
                raw = (query.get("raw") or query.get("download") or [""])[0] in {"1", "true", "yes"}
                auth = self.auth("storage_file" if raw else "storage", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                if not hive_users.storage_share_allowed(auth, share_id):
                    return self.send_json({"ok": False, "error": "storage_share_scope_denied", "share_id": share_id}, status=403)
                if raw:
                    result = storage_peer_file_bytes(policy, node_id=node_id, share_id=share_id, rel_path=rel_path)
                    if not result.get("ok"):
                        return self.send_json(result, status=400)
                    return self.send_bytes(
                        result.pop("content_bytes"),
                        content_type=str(result.get("content_type") or "application/octet-stream"),
                        filename=str(result.get("name") or "hive-file"),
                        attachment=(query.get("download") or [""])[0] in {"1", "true", "yes"},
                    )
                result = storage_peer_file_payload(policy, node_id=node_id, share_id=share_id, rel_path=rel_path)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/remote-control/status":
                auth = self.auth("status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(hive_remote_control.status_report(policy=policy, write_report=True))
            if parsed.path == "/api/hive/voice/status":
                auth = self.auth("status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(hive_voice_following.status_report(policy=policy, write_report=True))
            if parsed.path == "/api/hive/voice/route":
                auth = self.auth("status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                status = cached_build_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791)))
                return self.send_json(hive_voice_following.route_decision(policy=policy, status=status, peers=current_peers(policy), write_report=True))
            if parsed.path == "/api/hive/spatial/status":
                auth = self.auth("operator_status", query=parsed.query)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                status = cached_build_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791)))
                return self.send_json(hive_spatial.status_report(policy=policy, status=status, peers=current_peers(policy), auth_context=auth, write_report=True))
            return self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            payload = self.read_json_body()
            if parsed.path == "/api/hive/tasks":
                kind = str(payload.get("kind") or "")
                auth = self.auth("task", task_kind=kind)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                task_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
                result = enqueue_task(policy, kind, task_payload, source=f"http:{self.client_address[0]}")
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/operator/chat":
                auth = self.auth("chat", task_kind="checkpoint_chat")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(operator_chat(policy, payload, auth), status=200)
            if parsed.path == "/api/hive/operator/assistant-feedback":
                auth = self.auth("chat", task_kind="checkpoint_chat")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                result = operator_assistant_feedback(policy, payload, auth)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/operator/task":
                kind = str(payload.get("kind") or "")
                auth = self.auth("task", task_kind=kind)
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                result = operator_submit_task(policy, payload, auth)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/operator/notifications/ack":
                auth = self.auth("operator_status")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                result = acknowledge_operator_notifications(policy, payload, auth)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/operator/governance-audit":
                auth = self.auth("operator_status")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                result = operator_governance_audit(policy, payload, auth)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/operator/utilization":
                auth = self.auth("task", task_kind="utilization_sweep")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                result = operator_utilization_control(policy, payload, auth)
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/remote-control/session":
                auth = self.auth("remote_control")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                result = hive_remote_control.request_session(policy=policy, payload=payload, peers=current_peers(policy))
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/voice/presence":
                auth = self.auth("voice_control")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                result = hive_voice_following.presence_update(
                    policy=policy,
                    payload=payload,
                    requester={"source": "http", "client": self.client_address[0]},
                    write_report=True,
                )
                return self.send_json(result, status=200 if result.get("ok") else 400)
            if parsed.path == "/api/hive/heartbeat":
                auth = self.auth("status")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                peer = payload.get("peer") if isinstance(payload.get("peer"), dict) else payload
                accepted = accept_peer(policy, peer, source=f"heartbeat:{self.client_address[0]}", trusted=True)
                return self.send_json(
                    {
                        "ok": bool(accepted.get("ok", True)),
                        "accepted": accepted,
                        "peer": peer_from_status(cached_build_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791)))),
                        "peer_count": len(current_peers(policy)),
                        "peers": current_peers(policy),
                        "known_peer_count": len(known_peers(policy)),
                    }
                )
            if parsed.path == "/api/hive/update/configure":
                auth = self.auth("update_apply")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(update_configure_from_payload(payload))
            if parsed.path == "/api/hive/update/check":
                auth = self.auth("status")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                return self.send_json(update_check_from_payload(payload))
            if parsed.path in {"/api/hive/update/apply", "/api/hive/update/apply-soft"}:
                auth = self.auth("update_apply")
                if not auth.get("ok"):
                    return self.deny_auth(auth)
                mode = str(payload.get("mode") or "soft")
                if parsed.path == "/api/hive/update/apply-soft":
                    mode = "soft"
                if mode == "hard" and not is_loopback(self.client_address[0]):
                    return self.send_json({"ok": False, "error": "hard_update_requires_loopback"}, status=403)
                return self.send_json(update_apply_from_payload(payload, mode=mode))
            return self.send_error(HTTPStatus.NOT_FOUND)

        def read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
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

        def send_text(self, payload: str, *, content_type: str = "text/plain; charset=utf-8", status: int = 200) -> None:
            data = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_bytes(self, payload: bytes, *, content_type: str, filename: str, attachment: bool = False, status: int = 200) -> None:
            safe_name = filename.replace('"', "").replace("\r", "").replace("\n", "") or "hive-file"
            disposition = "attachment" if attachment else "inline"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Disposition", f'{disposition}; filename="{safe_name}"')
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return HiveHandler


def build_health(policy: dict[str, Any], *, http_port: int) -> dict[str, Any]:
    identity = load_identity(policy)
    local_ip = find_local_ip()
    port = http_port or int(get_path(policy, ["node", "http_port"], 8791))
    return {
        "ok": True,
        "policy": "project_theseus_hive_node_health_v1",
        "created_utc": now(),
        "node_id": identity["node_id"],
        "node_name": identity["node_name"],
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "api_url": f"http://{local_ip}:{port}",
        "hive_id": hive_id(policy),
        "shared_secret_configured": bool(join_token(policy) or shared_secret(policy)),
        "stale_after_seconds": int(get_path(policy, ["node", "stale_after_seconds"], 45)),
    }


def build_auth_status(policy: dict[str, Any], auth_context: dict[str, Any], *, http_port: int) -> dict[str, Any]:
    identity = load_identity(policy)
    local_ip = find_local_ip()
    port = http_port or int(get_path(policy, ["node", "http_port"], 8791))
    return {
        "ok": True,
        "policy": "project_theseus_hive_auth_status_v0",
        "created_utc": now(),
        "node_id": identity["node_id"],
        "node_name": identity["node_name"],
        "hostname": socket.gethostname(),
        "api_url": f"http://{local_ip}:{port}",
        "hive_id": hive_id(policy),
        "federation_tier": hive_tier(policy),
        "authenticated": True,
        "access": hive_users.user_summary(auth_context),
        "security": {
            "remote_tasks_require_secret": get_path(policy, ["security", "requires_shared_secret_for_remote_tasks"], True),
            "shared_secret_configured": bool(join_token(policy) or shared_secret(policy)),
            "multi_user_enabled": bool(get_path(policy, ["multi_user", "enabled"], True)),
            "auth_probe": "cheap_status_auth",
            "tokens_printed": False,
        },
    }


def build_status(policy: dict[str, Any], *, http_port: int) -> dict[str, Any]:
    identity = load_identity(policy)
    local_ip = find_local_ip()
    port = http_port or int(get_path(policy, ["node", "http_port"], 8791))
    resources = probe_resources(policy)
    capabilities = classify_capabilities(resources, policy)
    slots = resource_slots(resources, policy)
    voice_status = hive_voice_following.status_report(policy=policy, write_report=False)
    local_role_assignment = local_node_role_assignment(policy)
    status = {
        "policy": "project_theseus_hive_node_status_v0",
        "created_utc": now(),
        "node_id": identity["node_id"],
        "node_name": identity["node_name"],
        "node_public_key": identity.get("node_public_key", ""),
        "hostname": socket.gethostname(),
        "project_root": str(ROOT),
        "pid": os.getpid(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "listen_host": local_ip,
        "api_url": f"http://{local_ip}:{port}",
        "dashboard_url": f"http://{local_ip}:{int(get_path(policy, ['node', 'dashboard_port'], 8787))}",
        "hive_id": hive_id(policy),
        "federation_tier": hive_tier(policy),
        "relay_url": relay_url(policy),
        "roles": local_role_assignment.get("roles") or get_path(policy, ["node", "default_role"], ["worker"]),
        "role_assignment": local_role_assignment,
        "resources": resources,
        "storage": compact_storage(hive_storage.status_report(policy=policy, write_report=False)),
        "remote_control": compact_remote_control(hive_remote_control.status_report(policy=policy, write_report=False)),
        "rented_compute": compact_rented_compute(hive_rented_compute.status_report(policy=policy, write_report=False)),
        "utilization": compact_utilization(read_json(ROOT / "reports" / "hive_utilization_manager.json", {})),
        "voice_following": hive_voice_following.compact_status(voice_status),
        "capabilities": capabilities,
        "slots": slots,
        "runtime_paths": compact_runtime(theseus_runtime.runtime_report(create=False, write_report=True)),
        "license": compact_license(license_manager.status_report(write_report=True)),
        "compute_market": compact_market(compute_market.status_report(write_report=True)),
        "updates": compact_update_status(update_manager.status_report(policy=update_policy(), write_report=False)),
        "hive_version": compact_hive_version(hive_version_manager.status_report(write_report=False)),
        "virtual_context_memory": operator_vcm_summary(),
        "security": {
            "remote_tasks_require_secret": get_path(policy, ["security", "requires_shared_secret_for_remote_tasks"], True),
            "shared_secret_configured": bool(join_token(policy) or shared_secret(policy)),
            "multi_user_enabled": bool(get_path(policy, ["multi_user", "enabled"], True)),
            "user_count": int(hive_users.list_users(policy).get("user_count") or 0),
            "trust_boundary": get_path(policy, ["security", "trust_boundary"], "trusted_local_network_only"),
        },
        "task_kinds": sorted((policy.get("task_kinds") or {}).keys()),
        "stale_after_seconds": int(get_path(policy, ["node", "stale_after_seconds"], 45)),
    }
    status["spatial"] = hive_spatial.compact_status(hive_spatial.status_report(policy=policy, status=status, peers=[], write_report=False))
    return status


def local_node_role_assignment(policy: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / "configs" / "hive_node_roles.local.json"
    local = read_json(path, {})
    roles = local.get("roles") if isinstance(local.get("roles"), list) else []
    roles = [str(item) for item in roles if str(item).strip()]
    if roles:
        return {
            "policy": local.get("policy") or "project_theseus_hive_node_roles_local_v0",
            "source": display_path(path),
            "roles": roles,
            "created_utc": local.get("created_utc"),
            "expected": local.get("expected") if isinstance(local.get("expected"), dict) else {},
            "hardware_truth": "roles are local scheduler hints only; capabilities and slots remain probed from hardware/runtime",
        }
    return {
        "policy": "project_theseus_hive_node_roles_default_v0",
        "source": "configs/hive_policy.json",
        "roles": get_path(policy, ["node", "default_role"], ["worker"]),
        "hardware_truth": "roles are local scheduler hints only; capabilities and slots remain probed from hardware/runtime",
    }


def cached_build_status(policy: dict[str, Any], *, http_port: int, max_age_seconds: float | None = None) -> dict[str, Any]:
    ttl = float(max_age_seconds if max_age_seconds is not None else get_path(policy, ["node", "status_cache_seconds"], 3.0))
    if ttl <= 0:
        return build_status(policy, http_port=http_port)
    now_mono = time.monotonic()
    with STATUS_CACHE_LOCK:
        cached = STATUS_CACHE.get(http_port)
        if cached and now_mono - cached[0] <= ttl:
            return copy.deepcopy(cached[1])
    status = build_status(policy, http_port=http_port)
    with STATUS_CACHE_LOCK:
        STATUS_CACHE[http_port] = (time.monotonic(), copy.deepcopy(status))
    return status


def operator_fast_status(policy: dict[str, Any], *, http_port: int) -> dict[str, Any]:
    """Return a node status suitable for phone/operator refreshes.

    Full status probing can take several seconds on macOS because it touches
    power, runtime, update, license, and accelerator probes. The daemon writes a
    fresh status snapshot at startup and status probes can refresh it explicitly,
    so the operator surface should use a bounded-age snapshot before doing a
    synchronous full probe.
    """

    cache_ttl = float(get_path(policy, ["node", "operator_status_cache_status_seconds"], 30.0))
    now_mono = time.monotonic()
    with STATUS_CACHE_LOCK:
        cached = STATUS_CACHE.get(http_port)
        if cached and now_mono - cached[0] <= cache_ttl:
            status = copy.deepcopy(cached[1])
            status["_operator_status_source"] = {
                "kind": "memory_cache",
                "age_seconds": round(now_mono - cached[0], 3),
                "max_age_seconds": cache_ttl,
            }
            return status

    snapshot_path = report_path(policy, "status_path", "reports/hive_status.json", "")
    snapshot_max_age = float(get_path(policy, ["node", "operator_status_snapshot_max_age_seconds"], 300.0))
    snapshot = read_json(snapshot_path, {})
    snapshot_age = utc_age_seconds(str(snapshot.get("created_utc") or ""))
    if (
        snapshot.get("policy") == "project_theseus_hive_node_status_v0"
        and snapshot_age is not None
        and snapshot_age <= snapshot_max_age
    ):
        with STATUS_CACHE_LOCK:
            STATUS_CACHE[http_port] = (time.monotonic(), copy.deepcopy(snapshot))
        status = copy.deepcopy(snapshot)
        status["_operator_status_source"] = {
            "kind": "fresh_report_snapshot",
            "path": display_path(snapshot_path),
            "age_seconds": round(snapshot_age, 3),
            "max_age_seconds": snapshot_max_age,
        }
        return status

    status = cached_build_status(policy, http_port=http_port, max_age_seconds=cache_ttl)
    status["_operator_status_source"] = {
        "kind": "live_full_probe",
        "snapshot_path": display_path(snapshot_path),
        "snapshot_age_seconds": round(snapshot_age, 3) if snapshot_age is not None else None,
        "snapshot_max_age_seconds": snapshot_max_age,
    }
    return status


def utc_age_seconds(value: str) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())














































def discovery_loop(policy: dict[str, Any], status: dict[str, Any]) -> None:
    interval = max(15.0, float(get_path(policy, ["node", "heartbeat_interval_seconds"], 5)))
    probe_interval = max(60.0, float(get_path(policy, ["discovery", "known_peer_probe_interval_seconds"], 30)))
    last_probe = 0.0
    ensure_peer_registry_loaded(policy)
    listener = threading.Thread(target=listen_multicast, args=(policy,), daemon=True)
    listener.start()
    while not STOP.is_set():
        status = build_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791)))
        announce_multicast(policy, status)
        sync_static_coordinator(policy, status)
        if time.monotonic() - last_probe >= probe_interval:
            accept_bonjour_peers(policy, scan_bonjour(policy, seconds=float(get_path(policy, ["discovery", "bonjour_browse_seconds"], 3))))
            probe_known_peers(policy, status)
            last_probe = time.monotonic()
        write_json(report_path(policy, "peers_path", "reports/hive_peers.json", ""), peers_report(policy, local_status=status))
        write_json(report_path(policy, "status_path", "reports/hive_status.json", ""), status)
        STOP.wait(interval)


def relay_loop(policy: dict[str, Any], port: int) -> None:
    interval = max(15.0, float(get_path(policy, ["relay", "poll_interval_seconds"], get_path(policy, ["node", "heartbeat_interval_seconds"], 5))))
    ensure_peer_registry_loaded(policy)
    while not STOP.is_set():
        status = build_status(policy, http_port=port)
        register_with_relay(policy, status)
        for peer in fetch_relay_peers(policy):
            accept_peer(policy, peer, source="relay", trusted=True)
        for task in poll_relay_tasks(policy, status):
            enqueue_relay_task(policy, task)
        write_json(report_path(policy, "peers_path", "reports/hive_peers.json", ""), peers_report(policy, local_status=status))
        STOP.wait(interval)


def run_discovery(policy: dict[str, Any], status: dict[str, Any], *, seconds: float) -> None:
    ensure_peer_registry_loaded(policy)
    listener = threading.Thread(target=listen_multicast, args=(policy,), daemon=True)
    listener.start()
    deadline = time.time() + seconds
    while time.time() < deadline:
        announce_multicast(policy, status)
        sync_static_coordinator(policy, status)
        accept_bonjour_peers(policy, scan_bonjour(policy, seconds=min(3.0, max(0.5, deadline - time.time()))))
        probe_known_peers(policy, status)
        time.sleep(min(1.0, max(0.1, deadline - time.time())))


def bonjour_advertise_loop(policy: dict[str, Any], port: int) -> None:
    refresh = max(30.0, float(get_path(policy, ["discovery", "bonjour_advertise_refresh_seconds"], 300)))
    while not STOP.is_set():
        status = build_status(policy, http_port=port)
        command = bonjour_register_command(policy, status, port)
        if not command:
            STOP.wait(refresh)
            continue
        proc: subprocess.Popen[Any] | None = None
        try:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + refresh
            while not STOP.is_set() and time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                STOP.wait(1.0)
        except OSError as exc:
            append_jsonl(task_ledger_path(policy), event("bonjour_advertise_failed", {"error": str(exc)}))
            STOP.wait(refresh)
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()


def peers_report(policy: dict[str, Any], *, local_status: dict[str, Any] | None = None) -> dict[str, Any]:
    peers = current_peers(policy)
    all_known = known_peers(policy)
    stale = [peer for peer in all_known if peer.get("trusted") and not peer.get("online")]
    untrusted = [peer for peer in all_known if not peer.get("trusted")]
    registry = peer_registry_path(policy)
    return {
        "policy": "project_theseus_hive_peers_v0",
        "created_utc": now(),
        "local_node": peer_from_status(local_status or build_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791)))),
        "peer_count": len(peers),
        "peers": sorted(peers, key=lambda peer: str(peer.get("node_name") or peer.get("node_id"))),
        "known_peer_count": len(all_known),
        "known_peers": sorted(all_known, key=lambda peer: str(peer.get("node_name") or peer.get("node_id"))),
        "discovery_summary": peer_state_counts(all_known),
        "stale_peer_count": len(stale),
        "stale_peers": sorted(stale, key=lambda peer: str(peer.get("node_name") or peer.get("node_id"))),
        "untrusted_peer_count": len(untrusted),
        "untrusted_peers": sorted(untrusted, key=lambda peer: str(peer.get("node_name") or peer.get("node_id"))),
        "registry_path": display_path(registry),
        "stale_after_seconds": int(get_path(policy, ["node", "stale_after_seconds"], 45)),
    }


def enqueue_task(policy: dict[str, Any], kind: str, payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    task_defs = policy.get("task_kinds") if isinstance(policy.get("task_kinds"), dict) else {}
    if kind not in task_defs:
        return {"ok": False, "error": "unknown_task_kind", "kind": kind}
    if kind in set(get_path(policy, ["security", "forbidden_remote_task_kinds"], [])):
        return {"ok": False, "error": "forbidden_task_kind", "kind": kind}
    if not remote_task_allowed(policy, kind, source):
        return {"ok": False, "error": "task_kind_not_allowed_for_hive_tier", "kind": kind, "tier": hive_tier(policy)}
    license_check = task_license_check(kind, source)
    if not license_check.get("allowed"):
        return {"ok": False, "error": "license_required", "kind": kind, "license": license_check}
    support = local_task_support(policy, kind, payload)
    if not support.get("ok"):
        return {
            "ok": False,
            "error": "node_lacks_task_slot",
            "kind": kind,
            "support": support,
        }
    identity = load_identity(policy)
    job = job_model(policy, kind, payload)
    security = hive_security.authorize_task_payload(
        policy,
        kind=kind,
        payload=payload,
        source=source,
        hive_id=hive_id(policy),
        join_token=join_token(policy) or shared_secret(policy),
        local_node_id=str(identity.get("node_id") or ""),
    )
    if not security.get("ok"):
        return {"ok": False, "error": "security_check_failed", "kind": kind, "security": security}
    duplicate = duplicate_job(policy, str(job.get("job_id") or ""), payload)
    if duplicate:
        return {"ok": True, "duplicate": True, "task": duplicate}
    task = {
        "task_id": f"hive_task_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "kind": kind,
        "payload": payload,
        "job": job,
        "security": security,
        "source": source,
        "status": "queued",
        "created_utc": now(),
    }
    TASKS.put(task)
    append_jsonl(ROOT / str(get_path(policy, ["node", "task_queue_path"], "reports/hive_task_queue.jsonl")), task)
    return {"ok": True, "task": task}


def enqueue_relay_task(policy: dict[str, Any], relay_task: dict[str, Any]) -> dict[str, Any]:
    task_defs = policy.get("task_kinds") if isinstance(policy.get("task_kinds"), dict) else {}
    kind = str(relay_task.get("kind") or "")
    if kind not in task_defs:
        return {"ok": False, "error": "unknown_task_kind", "kind": kind}
    if kind in set(get_path(policy, ["security", "forbidden_remote_task_kinds"], [])):
        return {"ok": False, "error": "forbidden_task_kind", "kind": kind}
    if not remote_task_allowed(policy, kind, "relay"):
        return {"ok": False, "error": "task_kind_not_allowed_for_hive_tier", "kind": kind, "tier": hive_tier(policy)}
    license_check = task_license_check(kind, "relay")
    if not license_check.get("allowed"):
        return {"ok": False, "error": "license_required", "kind": kind, "license": license_check}
    payload = relay_task.get("payload") if isinstance(relay_task.get("payload"), dict) else {}
    support = local_task_support(policy, kind, payload)
    if not support.get("ok"):
        return {
            "ok": False,
            "error": "node_lacks_task_slot",
            "kind": kind,
            "support": support,
        }
    identity = load_identity(policy)
    job = job_model(policy, kind, payload)
    security = hive_security.authorize_task_payload(
        policy,
        kind=kind,
        payload=payload,
        source="relay",
        hive_id=hive_id(policy),
        join_token=join_token(policy) or shared_secret(policy),
        local_node_id=str(identity.get("node_id") or ""),
    )
    if not security.get("ok"):
        return {"ok": False, "error": "security_check_failed", "kind": kind, "security": security}
    duplicate = duplicate_job(policy, str(job.get("job_id") or ""), payload)
    if duplicate:
        return {"ok": True, "duplicate": True, "task": duplicate}
    task = {
        "task_id": relay_task.get("task_id") or f"relay_task_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "kind": kind,
        "payload": payload,
        "job": job,
        "security": security,
        "source": "relay",
        "relay_url": relay_url(policy),
        "hive_id": hive_id(policy),
        "status": "queued",
        "created_utc": now(),
    }
    TASKS.put(task)
    append_jsonl(ROOT / str(get_path(policy, ["node", "task_queue_path"], "reports/hive_task_queue.jsonl")), task)
    return {"ok": True, "task": task}


def job_model(policy: dict[str, Any], kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    jobs = policy.get("jobs") if isinstance(policy.get("jobs"), dict) else {}
    defaults = jobs.get("defaults") if isinstance(jobs.get("defaults"), dict) else {}
    by_kind = jobs.get("defaults_by_task_kind") if isinstance(jobs.get("defaults_by_task_kind"), dict) else {}
    kind_defaults = by_kind.get(kind) if isinstance(by_kind.get(kind), dict) else {}
    lease_seconds = int(payload.get("lease_seconds") or kind_defaults.get("lease_seconds") or defaults.get("lease_seconds") or 1800)
    max_retries = int(payload.get("max_retries") or kind_defaults.get("max_retries") or defaults.get("max_retries") or 0)
    return {
        "policy": "project_theseus_hive_job_v0",
        "job_id": str(payload.get("job_id") or payload.get("chunk_id") or f"{kind}_{uuid.uuid4().hex[:8]}"),
        "arm_id": str(payload.get("arm_id") or kind_defaults.get("arm_id") or default_arm_for_task(kind)),
        "task_kind": kind,
        "backend_requirements": payload.get("backend_requirements")
        if isinstance(payload.get("backend_requirements"), list)
        else default_backend_requirements(kind),
        "input_artifacts": payload.get("input_artifacts") if isinstance(payload.get("input_artifacts"), list) else [],
        "output_artifacts": payload.get("output_artifacts") if isinstance(payload.get("output_artifacts"), list) else default_output_artifacts(kind, payload),
        "merge_policy": str(payload.get("merge_policy") or kind_defaults.get("merge_policy") or defaults.get("merge_policy") or "append_report_then_gate"),
        "resource_budget": payload.get("resource_budget") if isinstance(payload.get("resource_budget"), dict) else kind_defaults.get("resource_budget", {}),
        "priority": int(payload.get("priority") or kind_defaults.get("priority") or defaults.get("priority") or 50),
        "lease_seconds": lease_seconds,
        "lease_expires_utc": utc_after_seconds(lease_seconds),
        "attempt": int(payload.get("attempt") or 0),
        "max_retries": max_retries,
    }


def default_arm_for_task(kind: str) -> str:
    if kind.startswith("cuda_"):
        return "rust_cuda_systems_arm"
    if kind.startswith("mlx_"):
        return "apple_mlx_worker_arm" if "apple" in kind else "mlx_worker_arm"
    if kind in {"capability_refresh", "readiness_check"}:
        return "governance_arm"
    return "hive_control_arm"


def default_backend_requirements(kind: str) -> list[str]:
    if kind.startswith("cuda_"):
        return ["nvidia_cuda", "rust_cuda"]
    if kind.startswith("mlx_"):
        return ["mlx_apple_or_mlx_cuda"]
    return ["cpu_worker"]


def default_output_artifacts(kind: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    chunk_id = str(payload.get("chunk_id") or payload.get("job_id") or kind)
    if kind.endswith("_chunk"):
        return [{"type": "worker_report", "path": f"reports/hive_chunks/{chunk_id}.json"}]
    return [{"type": "task_report", "path": str(payload.get("out") or "")}]


def duplicate_job(policy: dict[str, Any], job_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not job_id or payload.get("force_requeue") is True:
        return None
    with TASKS.mutex:
        for task in list(TASKS.queue):
            job = task.get("job") if isinstance(task.get("job"), dict) else {}
            if str(job.get("job_id") or "") == job_id:
                return {**task, "duplicate_status": "already_queued"}
    for path in [
        ROOT / str(get_path(policy, ["node", "task_queue_path"], "reports/hive_task_queue.jsonl")),
        ROOT / str(get_path(policy, ["node", "job_ledger_path"], "reports/hive_job_ledger.jsonl")),
    ]:
        for row in reversed(read_jsonl_tail(path, limit=500)):
            job = row.get("job") if isinstance(row.get("job"), dict) else {}
            if str(job.get("job_id") or "") != job_id:
                continue
            status = str(row.get("status") or "")
            if status in {"failed", "timeout"} and payload.get("retry_failed") is True:
                return None
            return {**row, "duplicate_status": status or "already_seen"}
    return None


def utc_after_seconds(seconds: int) -> str:
    return datetime.fromtimestamp(time.time() + max(1, seconds), tz=timezone.utc).isoformat()


def should_retry(task: dict[str, Any], result: dict[str, Any]) -> bool:
    if result.get("status") == "completed":
        return False
    job = task.get("job") if isinstance(task.get("job"), dict) else {}
    attempt = int(job.get("attempt") or 0)
    max_retries = int(job.get("max_retries") or 0)
    return attempt < max_retries


def retry_envelope(task: dict[str, Any]) -> dict[str, Any]:
    retry_task = json.loads(json.dumps(task))
    job = retry_task.get("job") if isinstance(retry_task.get("job"), dict) else {}
    attempt = int(job.get("attempt") or 0) + 1
    job["attempt"] = attempt
    job["lease_expires_utc"] = utc_after_seconds(int(job.get("lease_seconds") or 300))
    retry_task["job"] = job
    retry_task["status"] = "retry_queued"
    retry_task["retry_queued_utc"] = now()
    return retry_task


def worker_loop(policy: dict[str, Any], worker_index: int) -> None:
    while not STOP.is_set():
        try:
            task = TASKS.get(timeout=1.0)
        except queue.Empty:
            continue
        slot = acquire_slot(policy, str(task.get("kind") or ""), task.get("payload") if isinstance(task.get("payload"), dict) else {})
        if not slot:
            TASKS.put(task)
            TASKS.task_done()
            STOP.wait(0.5)
            continue
        try:
            result = run_task(policy, task)
        except Exception as exc:  # noqa: BLE001 - keep worker pool alive and ledgered.
            result = {
                **task,
                "status": "failed",
                "error": type(exc).__name__,
                "message": str(exc),
                "started_utc": now(),
                "finished_utc": now(),
                "returncode": 1,
            }
        finally:
            release_slot(str(slot.get("slot_id") or ""))
        result["worker_index"] = worker_index
        result["slot"] = {"slot_id": slot.get("slot_id"), "slot_type": slot.get("slot_type")}
        append_jsonl(task_ledger_path(policy), result)
        append_jsonl(ROOT / str(get_path(policy, ["node", "job_ledger_path"], "reports/hive_job_ledger.jsonl")), result)
        if task.get("source") == "relay":
            post_relay_result(policy, task, result)
        if should_retry(task, result):
            retry_task = retry_envelope(task)
            TASKS.put(retry_task)
            append_jsonl(ROOT / str(get_path(policy, ["node", "task_queue_path"], "reports/hive_task_queue.jsonl")), retry_task)
        TASKS.task_done()


def run_task(policy: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    task_defs = policy.get("task_kinds") if isinstance(policy.get("task_kinds"), dict) else {}
    task_def = task_defs.get(task.get("kind"), {})
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    command = render_command(task_def.get("command") or [], payload)
    timeout = int(task_def.get("timeout_seconds") or 300)
    started = time.perf_counter()
    started_utc = now()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, env=theseus_runtime.runtime_env())
        report = {
            **task,
            "status": "completed" if result.returncode == 0 else "failed",
            "started_utc": started_utc,
            "finished_utc": now(),
            "command": command,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        report["artifacts"] = materialize_output_artifacts(policy, report, command)
        report["artifact_signature"] = hive_security.sign_artifact(
            compact_result_for_signature(report),
            secret=join_token(policy) or shared_secret(policy),
            signer=str(load_identity(policy).get("node_id") or ""),
        )
        return report
    except subprocess.TimeoutExpired as exc:
        report = {
            **task,
            "status": "timeout",
            "started_utc": started_utc,
            "finished_utc": now(),
            "command": command,
            "returncode": 124,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
        report["artifact_signature"] = hive_security.sign_artifact(
            compact_result_for_signature(report),
            secret=join_token(policy) or shared_secret(policy),
            signer=str(load_identity(policy).get("node_id") or ""),
        )
        return report
    except OSError as exc:
        report = {
            **task,
            "status": "failed",
            "started_utc": started_utc,
            "finished_utc": now(),
            "command": command,
            "returncode": 127,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
        report["artifact_signature"] = hive_security.sign_artifact(
            compact_result_for_signature(report),
            secret=join_token(policy) or shared_secret(policy),
            signer=str(load_identity(policy).get("node_id") or ""),
        )
        return report


def compact_result_for_signature(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": report.get("task_id"),
        "kind": report.get("kind"),
        "job": report.get("job"),
        "status": report.get("status"),
        "returncode": report.get("returncode"),
        "runtime_ms": report.get("runtime_ms"),
        "stdout_tail_hash": hive_security.sha256_json(report.get("stdout_tail", "")),
        "stderr_tail_hash": hive_security.sha256_json(report.get("stderr_tail", "")),
    }


def render_command(template: list[Any], payload: dict[str, Any]) -> list[str]:
    values = {
        "python": sys.executable,
        "checkpoint_id": str(payload.get("checkpoint_id") or "live"),
        "prompt": str(payload.get("prompt") or "Summarize current Project Theseus status."),
        "session_id": str(payload.get("session_id") or "operator_default"),
        "feedback": str(payload.get("feedback") or "completed"),
        "intent": str(payload.get("intent") or "auto"),
        "surface": str(payload.get("source") or payload.get("surface") or "hive_operator"),
        "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }
    rendered = []
    for item in template:
        text = str(item)
        for key, value in values.items():
            text = text.replace("{" + key + "}", value)
        rendered.append(text)
    return rendered


def remote_task_allowed(policy: dict[str, Any], kind: str, source: str) -> bool:
    if is_local_task_source(source):
        return True
    allowed_by_tier = get_path(policy, ["relay", "allowed_remote_task_kinds_by_tier"], {})
    if not isinstance(allowed_by_tier, dict):
        return True
    allowed = allowed_by_tier.get(hive_tier(policy))
    if not isinstance(allowed, list):
        return False
    return kind in set(str(item) for item in allowed)


def task_license_check(kind: str, source: str) -> dict[str, Any]:
    low_local = kind in {"resource_probe", "capability_refresh", "readiness_check", "checkpoint_chat", "compute_market_status", "storage_status", "storage_index", "voice_following_status", "spatial_status", "network_doctor", "update_status", "hive_version_status", "hive_version_converge"} and is_local_task_source(source)
    if low_local:
        return {"allowed": True, "feature": "local_status"}
    feature = license_manager.feature_for_task_kind(kind)
    report = license_manager.check_feature(feature, write_report=True)
    return {
        "allowed": bool(report.get("allowed")),
        "feature": feature,
        "tier": get_path(report, ["entitlement", "tier"], ""),
        "source": get_path(report, ["entitlement", "source"], ""),
        "next_action": report.get("next_action"),
        "feature_check": report.get("feature_check", {}),
    }


def operator_status(policy: dict[str, Any], auth_context: dict[str, Any] | None = None) -> dict[str, Any]:
    status = operator_fast_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791)))
    peers = current_peers(policy)
    tasks = tasks_report(policy)
    allowed_task_kinds = operator_allowed_task_kinds(policy, auth_context)
    report = {
        "ok": True,
        "policy": "project_theseus_hive_operator_status_v0",
        "created_utc": now(),
        "access": hive_users.user_summary(auth_context),
        "hive": {
            "hive_id": hive_id(policy),
            "tier": hive_tier(policy),
            "local_node": peer_from_status(status),
            "local_status_source": status.get("_operator_status_source") or {},
            "peer_count": len(peers),
            "peers": peers,
            "targets": operator_targets(policy, status, peers),
            "allowed_task_kinds": allowed_task_kinds,
        },
        "roaming": operator_roaming_summary(policy, status, peers),
        "network": operator_network_summary(),
        "users": hive_users.list_users(policy) if hive_users.action_allowed(auth_context or {}, "manage_users") else {"enabled": bool(get_path(policy, ["multi_user", "enabled"], True)), "manageable": False},
        "storage": operator_storage_summary(policy, status, peers) if hive_users.action_allowed(auth_context or {}, "storage") else {"enabled": False, "access_denied": True, "reason": "user_role_cannot_view_storage"},
        "remote_control": operator_remote_control_summary(policy, status, peers) if hive_users.action_allowed(auth_context or {}, "remote_control") else {"enabled": bool(get_path(policy, ["remote_control", "enabled"], True)), "access_denied": True, "reason": "user_role_cannot_remote_control"},
        "rented_compute": status.get("rented_compute") or {},
        "utilization": operator_utilization_summary(status),
        "voice_following": hive_voice_following.operator_summary(policy, status, peers) if hive_users.action_allowed(auth_context or {}, "voice_control") else {"enabled": bool(get_path(policy, ["voice_following", "enabled"], True)), "access_denied": True, "reason": "user_role_cannot_control_voice"},
        "spatial": hive_spatial.operator_summary(policy, status, peers, auth_context),
        "accelerators": operator_accelerator_summary(policy, status, peers, allowed_task_kinds),
        "training": operator_training_summary(policy),
        "teacher_governance": operator_teacher_governance_summary(),
        "governance_audit": operator_governance_audit_summary(),
        "solo_learning": operator_solo_learning_summary(),
        "notifications": operator_notification_summary(policy, auth_context),
        "learning": operator_learning_summary(),
        "benchmarks": operator_benchmark_summary(),
        "games": operator_game_summary(),
        "autonomy": operator_autonomy_summary(),
        "assistant": operator_assistant_summary(),
        "memory": {
            "virtual_context_memory": operator_vcm_summary(),
        },
        "updates": {
            "candidate": compact_update_status(update_manager.status_report(policy=update_policy(), write_report=False)),
            "hive_version": compact_hive_version(hive_version_manager.status_report(write_report=False)),
        },
        "tasks": {
            "queued_in_memory": tasks.get("queued_in_memory"),
            "slots": tasks.get("slots"),
            "recent_queued": (tasks.get("recent_queued") or [])[-10:],
            "recent_results": (tasks.get("recent_results") or [])[-10:],
        },
        "controls": {
            "chat_endpoint": "/api/hive/operator/chat",
            "assistant_feedback_endpoint": "/api/hive/operator/assistant-feedback",
            "task_endpoint": "/api/hive/operator/task",
            "notifications_endpoint": "/api/hive/operator/notifications",
            "notifications_ack_endpoint": "/api/hive/operator/notifications/ack",
            "governance_audit_endpoint": "/api/hive/operator/governance-audit",
            "spatial_status_endpoint": "/api/hive/spatial/status",
            "allowed_side_effects": "registered bounded Hive task kinds only",
        },
        "external_inference_calls": 0,
    }
    write_json(ROOT / "reports" / "hive_operator_status.json", report)
    return report


def operator_teacher_governance_summary() -> dict[str, Any]:
    share_report = read_json(ROOT / "reports" / "teacher_share_ledger_summary.json", {})
    rights_report = read_json(ROOT / "reports" / "governance_rights_receipt_suite.json", {})
    gate_report = read_json(ROOT / "reports" / "teacher_distillation_gate.json", {})
    share = share_report.get("summary") if isinstance(share_report.get("summary"), dict) else {}
    rights = rights_report.get("summary") if isinstance(rights_report.get("summary"), dict) else {}
    gate = gate_report.get("summary") if isinstance(gate_report.get("summary"), dict) else {}
    daily_buckets = int(share.get("daily_trend_bucket_count") or 0)
    runtime_external_calls = int(share.get("runtime_external_inference_calls") or 0)
    public_training_rows = int(share.get("public_training_rows_written") or 0)
    fallback_returns = int(rights.get("fallback_return_count") or share.get("fallback_return_count") or 0)
    teacher_rows = int(share.get("teacher_accepted_rows") or 0)
    verified_self_rows = int(share.get("verified_self_generated_rows") or 0)
    accepted_rows = int(share.get("accepted_training_rows") or 0)
    rights_hard_gaps = int(rights.get("hard_gap_count") or 0)
    rights_fixture_count = int(rights.get("fixture_count") or 0)
    rights_passed_count = int(rights.get("passed_fixture_count") or 0)
    no_cheat_clean = (
        bool(share.get("no_cheat_clean", False))
        and rights_hard_gaps == 0
        and runtime_external_calls == 0
        and public_training_rows == 0
        and fallback_returns == 0
    )
    metric_ready = bool(share.get("metric_ready")) and bool(share.get("ledger_present"))
    rights_ready = rights_fixture_count > 0 and rights_passed_count == rights_fixture_count and rights_hard_gaps == 0
    trigger_state = "GREEN" if metric_ready and rights_ready and no_cheat_clean else "YELLOW"
    return {
        "enabled": True,
        "operator_visible": True,
        "trigger_state": trigger_state,
        "teacher_share_report": "reports/teacher_share_ledger_summary.json",
        "governance_rights_report": "reports/governance_rights_receipt_suite.json",
        "teacher_distillation_gate_report": "reports/teacher_distillation_gate.json",
        "teacher_share_ledger_state": share_report.get("trigger_state"),
        "governance_rights_state": rights_report.get("trigger_state"),
        "teacher_distillation_gate_state": share.get("distillation_gate_state") or gate_report.get("trigger_state"),
        "distillation_allowed": bool(share.get("distillation_allowed", gate.get("distillation_allowed", False))),
        "metric_ready": metric_ready,
        "ledger_present": bool(share.get("ledger_present")),
        "ledger_row_count": share.get("ledger_row_count"),
        "accepted_training_rows": accepted_rows,
        "teacher_accepted_rows": teacher_rows,
        "accepted_rows_from_teacher_ledger": share.get("accepted_rows_from_teacher_ledger"),
        "accepted_non_teacher_training_rows": share.get("accepted_non_teacher_training_rows"),
        "verified_self_generated_rows": verified_self_rows,
        "teacher_proposal_rows": share.get("teacher_proposal_rows"),
        "teacher_rejected_rows": share.get("teacher_rejected_rows"),
        "teacher_share_of_accepted_training_rows": share.get("teacher_share_of_accepted_training_rows"),
        "teacher_share_cap": share.get("teacher_share_cap"),
        "teacher_share_within_cap": bool(share.get("teacher_share_within_cap", False)),
        "teacher_share_graduation_target": share.get("teacher_share_graduation_target"),
        "teacher_share_target_trend": share.get("teacher_share_target_trend"),
        "daily_trend_bucket_count": daily_buckets,
        "trend_state": "multi_cycle_ready" if daily_buckets >= 2 else "single_bucket_baseline",
        "governance_right_fixture_count": rights_fixture_count,
        "passed_governance_right_fixture_count": rights_passed_count,
        "constitutional_fixture_count": rights.get("constitutional_fixture_count"),
        "passed_constitutional_fixture_count": rights.get("passed_constitutional_fixture_count"),
        "governance_right_record_count": rights.get("governance_right_record_count"),
        "failure_boundary_record_count": rights.get("failure_boundary_record_count"),
        "constitutional_predicate_record_count": rights.get("constitutional_predicate_record_count"),
        "artifact_graph_record_count": rights.get("artifact_graph_record_count"),
        "claim_record_count": rights.get("claim_record_count"),
        "evidence_transition_record_count": rights.get("evidence_transition_record_count"),
        "hard_gap_count": rights_hard_gaps,
        "warning_count": rights.get("warning_count"),
        "runtime_external_inference_calls": runtime_external_calls,
        "public_training_rows_written": public_training_rows,
        "fallback_return_count": fallback_returns,
        "runtime_external_tokens_forbidden": True,
        "teacher_rows_runtime_serving_allowed": False,
        "public_benchmark_training_allowed": False,
        "no_cheat_clean": no_cheat_clean,
        "note": "Operator visibility only; externally generated tokens remain forbidden at runtime.",
    }


def operator_governance_audit_summary() -> dict[str, Any]:
    report = read_json(ROOT / "reports" / "hive_operator_governance_audit.json", {})
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "enabled": True,
        "endpoint": "/api/hive/operator/governance-audit",
        "latest_report": "reports/hive_operator_governance_audit.json" if report else "",
        "trigger_state": report.get("trigger_state") if report else "",
        "request_id": report.get("request_id", ""),
        "request_kind": report.get("request_kind", ""),
        "artifact_count": summary.get("artifact_count", 0),
        "payload_citation_applicable_count": summary.get("payload_citation_applicable_count", 0),
        "payload_citation_ok_count": summary.get("payload_citation_ok_count", 0),
        "payload_citation_fault_count": summary.get("payload_citation_fault_count", 0),
        "governance_right_record_count": summary.get("governance_right_record_count", 0),
        "claim_record_count": summary.get("claim_record_count", 0),
        "no_cheat_clean": bool(summary.get("no_cheat_clean", False)),
        "public_training_rows_written": int(summary.get("public_training_rows_written") or 0),
        "runtime_external_inference_calls": int(summary.get("runtime_external_inference_calls") or 0),
        "fallback_return_count": int(summary.get("fallback_return_count") or 0),
        "non_claim": "Operator audit/export receipt evidence only; not learned-generation or institutional-governance proof.",
    }


def operator_governance_audit(
    policy: dict[str, Any],
    payload: dict[str, Any] | None,
    auth_context: dict[str, Any] | None,
    *,
    out_path: Path | None = None,
) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    created = now()
    request_kind = str(payload.get("request_kind") or "audit_export")
    user = hive_users.user_summary(auth_context)
    request_id = str(payload.get("request_id") or governance_stable_id("operator_governance_audit", created, request_kind, user.get("user_id") or user.get("token_kind") or "operator"))
    artifact_refs = governance_audit_artifact_refs(payload)
    artifacts = [operator_governance_artifact_citation(policy, ref) for ref in artifact_refs]
    payload_applicable_count = sum(1 for row in artifacts if row.get("payload_citation_applicable"))
    payload_ok_count = sum(1 for row in artifacts if row.get("payload_citation_ok"))
    missing_count = sum(1 for row in artifacts if not row.get("exists"))
    payload_fault_count = payload_applicable_count - payload_ok_count
    counter = {
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "runtime_external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    records = operator_governance_audit_records(
        request_id=request_id,
        request_kind=request_kind,
        created_utc=created,
        artifacts=artifacts,
        user=user,
        payload_ok_count=payload_ok_count,
        payload_applicable_count=payload_applicable_count,
        missing_count=missing_count,
    )
    trigger_state = "GREEN" if artifacts and missing_count == 0 and payload_applicable_count > 0 and payload_fault_count == 0 else "YELLOW"
    report = {
        "ok": True,
        "policy": "project_theseus_hive_operator_governance_audit_v1",
        "created_utc": created,
        "trigger_state": trigger_state,
        "request_id": request_id,
        "request_kind": request_kind,
        "access": user,
        "endpoint": "/api/hive/operator/governance-audit",
        "artifact_index_endpoint": "/api/hive/artifacts",
        "artifact_payload_endpoint": "/api/hive/artifact",
        "artifact_refs": artifacts,
        "summary": {
            "artifact_count": len(artifacts),
            "artifact_missing_count": missing_count,
            "payload_citation_applicable_count": payload_applicable_count,
            "payload_citation_ok_count": payload_ok_count,
            "payload_citation_not_applicable_count": len(artifacts) - payload_applicable_count,
            "payload_citation_fault_count": payload_fault_count,
            "governance_right_record_count": len(records["governance_right_records"]),
            "authority_use_receipt_count": len(records["authority_use_receipts"]),
            "failure_boundary_record_count": len(records["failure_boundary_records"]),
            "artifact_graph_record_count": len(records["artifact_graph_records"]),
            "claim_record_count": len(records["claim_records"]),
            "evidence_transition_record_count": len(records["evidence_transition_records"]),
            "no_cheat_clean": True,
            **counter,
        },
        **records,
        "rights": {
            "audit": "granted_as_manifest_and_artifact_citations",
            "portable_export": "granted_as_manifest_without_raw_private_text_or_secrets",
            "redaction": "raw private text, secrets, and benchmark payloads are represented by handles and hashes only",
            "appeal_path": "configs/project_steward.json",
            "preservation_obligation": "retain this receipt plus cited artifacts until steward review or retention manifest supersedes them",
        },
        "non_claims": [
            "This operator response proves a local audit/export receipt path, not institutional governance or legal compliance.",
            "It exposes artifact refs, hashes, and citations, not raw private text, secrets, public benchmark payloads, or hidden tests.",
            "It is not learned-generation evidence and cannot support model promotion.",
            "Trusted-peer artifact endpoint validation remains separate until a trusted peer is reachable.",
        ],
        **counter,
    }
    write_json(out_path or (ROOT / "reports" / "hive_operator_governance_audit.json"), report)
    return report


def governance_audit_artifact_refs(payload: dict[str, Any]) -> list[str]:
    defaults = [
        "AGENTS.md",
        "roadmap.md",
        "docs/PROJECT_STATE.md",
        "docs/PROJECT_REGISTRY.md",
        "configs/project_manifest_registry.json",
        "configs/project_steward.json",
        "configs/roadmap_implementation_matrix.json",
        "reports/roadmap_implementation_gate.json",
        "reports/viea_spine_record_gate.json",
        "reports/viea_spine_materialized_view.json",
        "reports/module_definition_of_done.json",
        "reports/book_to_theseus_crosswalk.json",
        "reports/governance_rights_receipt_suite.json",
    ]
    requested = payload.get("artifact_refs") if isinstance(payload.get("artifact_refs"), list) else []
    requested += payload.get("artifact_ref") if isinstance(payload.get("artifact_ref"), list) else []
    requested_refs = [str(item) for item in requested if str(item or "")]
    return unique_nonempty(defaults + requested_refs)[:40]


def operator_governance_artifact_citation(policy: dict[str, Any], artifact_ref: str) -> dict[str, Any]:
    clean_ref = str(artifact_ref or "").lstrip("/")
    path = (ROOT / clean_ref).resolve()
    allowed = str(path).startswith(str(ROOT.resolve())) and path.is_file()
    payload = read_artifact_payload(policy, clean_ref) if allowed else {"ok": False, "error": "artifact_path_not_allowed"}
    payload_applicable = bool(payload.get("ok")) or str(payload.get("error") or "") not in {"artifact_path_not_allowed"}
    stat_size = path.stat().st_size if allowed else 0
    sha = payload.get("sha256") if payload.get("ok") else (sha256_file(path) if allowed else "")
    return {
        "artifact_ref": clean_ref,
        "exists": bool(allowed),
        "size_bytes": stat_size,
        "sha256": sha,
        "artifact_endpoint": f"/api/hive/artifact?path={quote(clean_ref)}" if payload_applicable else "",
        "payload_citation_applicable": payload_applicable,
        "payload_citation_ok": bool(payload.get("ok")),
        "payload_error": payload.get("error", "") if payload_applicable else "source_ref_not_artifact_endpoint",
        "payload_policy": payload.get("policy", ""),
        "payload_viea_artifact_citation": payload.get("viea_artifact_citation") if payload.get("ok") else {},
    }


def operator_governance_audit_records(
    *,
    request_id: str,
    request_kind: str,
    created_utc: str,
    artifacts: list[dict[str, Any]],
    user: dict[str, Any],
    payload_ok_count: int,
    payload_applicable_count: int,
    missing_count: int,
) -> dict[str, list[dict[str, Any]]]:
    support_state = "SUPPORTED" if artifacts and missing_count == 0 and payload_applicable_count > 0 and payload_ok_count == payload_applicable_count else "PARTIAL"
    base = {
        "request_id": request_id,
        "request_kind": request_kind,
        "created_utc": created_utc,
        "support_state": support_state,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "runtime_external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
        "learned_generation_claim_allowed": False,
    }
    right_id = governance_stable_id("governance_right", request_id, request_kind)
    authority_id = governance_stable_id("authority_use", request_id, user.get("role") or "")
    governance_right_records = [
        {
            **base,
            "record_id": right_id,
            "record_type": "governance_right_record",
            "right_id": right_id,
            "right_type": "operator_audit_export",
            "holder": user.get("user_id") or user.get("token_kind") or "operator",
            "scope": "project_theseus_local_operator_state",
            "expected_decision": "grant_manifest_with_redactions",
            "material_available": bool(artifacts),
            "material_withheld": True,
            "withheld_material": ["raw private text", "secrets", "public benchmark payloads", "hidden tests"],
            "access_path": "reports/hive_operator_governance_audit.json",
            "appeal_path": "configs/project_steward.json",
            "preservation_obligation": "retain audit/export receipt and artifact citations until steward review or retention manifest supersedes them",
        }
    ]
    authority_use_receipts = [
        {
            **base,
            "record_id": authority_id,
            "record_type": "authority_use_receipt",
            "authority_id": authority_id,
            "actor": user.get("user_id") or user.get("token_kind") or "operator",
            "role": user.get("role") or "",
            "action": "operator_governance_audit",
            "authority_ceiling": "operator_status_read_only_audit_export",
            "side_effects": ["write reports/hive_operator_governance_audit.json"],
            "forbidden_effects": ["runtime external inference", "public benchmark training rows", "raw private text export", "secret export"],
        }
    ]
    failure_boundary_records = [
        {
            **base,
            "record_id": governance_stable_id("failure_boundary", request_id, "redaction"),
            "record_type": "failure_boundary",
            "failure_id": governance_stable_id("operator_audit_redaction_boundary", request_id),
            "blocked_reason": "raw_private_text_secrets_and_public_benchmark_payloads_are_not_exported",
            "terminal": False,
            "structured_non_solved": True,
            "containment_action": "return manifest, artifact refs, hashes, endpoint citations, and appeal path",
        }
    ]
    artifact_graph_records = []
    evidence_transition_records = []
    for item in artifacts:
        artifact_id = governance_stable_id("artifact_graph", request_id, item.get("artifact_ref"))
        artifact_graph_records.append(
            {
                **base,
                "record_id": artifact_id,
                "record_type": "artifact_graph_record",
                "artifact_ref": item.get("artifact_ref"),
                "artifact_endpoint": item.get("artifact_endpoint"),
                "content_hash": item.get("sha256"),
                "payload_citation_applicable": item.get("payload_citation_applicable"),
                "payload_citation_ok": item.get("payload_citation_ok"),
                "evidence_ref": "reports/hive_operator_governance_audit.json",
                "replay_limit": "local_artifact_endpoint_requires_operator_or_artifact_sync_scope",
                "non_claim": "artifact citation, not raw artifact export",
            }
        )
        evidence_transition_records.append(
            {
                **base,
                "record_id": governance_stable_id("evidence_transition", request_id, item.get("artifact_ref")),
                "record_type": "evidence_transition_record",
                "previous_support_state": "REQUESTED",
                "current_support_state": "SUPPORTED" if item.get("payload_citation_ok") else "PARTIAL",
                "evidence_ref": item.get("artifact_ref"),
            }
        )
    claim_records = [
        {
            **base,
            "record_id": governance_stable_id("claim", request_id),
            "record_type": "claim_record",
            "claim_id": governance_stable_id("operator_governance_audit_claim", request_id),
            "claim": "local_operator_audit_export_receipt_materialized",
            "evidence_ref": "reports/hive_operator_governance_audit.json",
            "support_state": support_state,
            "non_claim": "does not prove model capability, institutional governance, legal compliance, or peer artifact reachability",
        }
    ]
    return {
        "governance_right_records": governance_right_records,
        "authority_use_receipts": authority_use_receipts,
        "failure_boundary_records": failure_boundary_records,
        "artifact_graph_records": artifact_graph_records,
        "claim_records": claim_records,
        "evidence_transition_records": evidence_transition_records,
    }


def governance_stable_id(*parts: Any) -> str:
    return hive_security.sha256_json(parts)[:24]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_operator_status(policy: dict[str, Any], auth_context: dict[str, Any] | None = None) -> dict[str, Any]:
    ttl = float(get_path(policy, ["node", "operator_status_cache_seconds"], 5.0))
    if ttl <= 0:
        return operator_status(policy, auth_context)
    access_key = json.dumps(hive_users.user_summary(auth_context), sort_keys=True, separators=(",", ":"))
    now_mono = time.monotonic()
    with OPERATOR_STATUS_CACHE_LOCK:
        cached = OPERATOR_STATUS_CACHE.get(access_key)
        if cached and now_mono - cached[0] <= ttl:
            return copy.deepcopy(cached[1])
    report = operator_status(policy, auth_context)
    with OPERATOR_STATUS_CACHE_LOCK:
        OPERATOR_STATUS_CACHE[access_key] = (time.monotonic(), copy.deepcopy(report))
    return report


def operator_roaming_summary(policy: dict[str, Any], status: dict[str, Any], peers: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [peer_from_status(status)] + peers
    node_port = int(get_path(policy, ["node", "http_port"], 8791))
    bonjour_urls = operator_local_bonjour_urls(status, node_port=node_port)
    node_urls = unique_nonempty(bonjour_urls + [str(row.get("api_url") or "") for row in rows])
    relay_urls = unique_nonempty([str(row.get("relay_url") or "") for row in rows])
    endpoints = [
        {
            "kind": "node",
            "url": url,
            "operator_url": url.rstrip("/") + "/mobile",
            "source": operator_endpoint_source(url, bonjour_urls=bonjour_urls),
            "priority": operator_endpoint_priority("node", url, bonjour_urls=bonjour_urls),
        }
        for url in node_urls
    ] + [
        {
            "kind": "relay",
            "url": url,
            "operator_url": url.rstrip("/") + "/mobile",
            "source": "relay",
            "priority": operator_endpoint_priority("relay", url, bonjour_urls=bonjour_urls),
        }
        for url in relay_urls
    ]
    endpoints = sorted(endpoints, key=lambda row: int(row.get("priority") or 999))
    node_urls = unique_nonempty([str(row.get("url") or "") for row in endpoints if row.get("kind") == "node"])
    relay_urls = unique_nonempty([str(row.get("url") or "") for row in endpoints if row.get("kind") == "relay"])
    return {
        "policy": "project_theseus_hive_operator_roaming_v1",
        "strategy": "iphone_try_last_good_then_lan_then_private_tunnel_then_https_relay",
        "node_urls": node_urls,
        "relay_urls": relay_urls,
        "operator_urls": [url.rstrip("/") + "/mobile" for url in node_urls + relay_urls],
        "endpoints": endpoints,
        "endpoint_selection": operator_endpoint_selection_policy(),
        "bonjour": operator_bonjour_roaming_summary(policy, bonjour_urls=bonjour_urls, node_port=node_port),
        "handover": operator_roaming_handover_policy(),
        "update_catalog_urls": [url.rstrip("/") + "/api/hive/update-catalog" for url in node_urls + relay_urls],
        "installer_artifacts_urls": [url.rstrip("/") + "/api/hive/installer-artifacts" for url in node_urls + relay_urls],
        "security": {
            "raw_public_8791_forbidden": True,
            "public_access": "https_relay_or_private_tunnel",
            "paid_dependency_required": False,
        },
    }


def operator_local_bonjour_urls(status: dict[str, Any], *, node_port: int) -> list[str]:
    hosts: list[str] = []
    for value in [status.get("hostname"), socket.gethostname(), socket.getfqdn()]:
        host = str(value or "").strip().strip(".")
        lowered = host.lower()
        if not host or lowered in {"localhost", "ip6-localhost"} or lowered.endswith(".arpa"):
            continue
        hosts.append(host)
        if "." not in host:
            hosts.append(f"{host}.local")
    return unique_nonempty([f"http://{host}:{node_port}" for host in hosts])


def operator_endpoint_source(url: str, *, bonjour_urls: list[str]) -> str:
    normalized = url.rstrip("/")
    if normalized in {candidate.rstrip("/") for candidate in bonjour_urls}:
        return "bonjour_local_hostname"
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.endswith(".local"):
        return "bonjour_local_hostname"
    if host.endswith((".lan", ".home")):
        return "private_dns"
    if is_loopback(host):
        return "loopback"
    return "configured_or_discovered"


def operator_endpoint_priority(kind: str, url: str, *, bonjour_urls: list[str]) -> int:
    source = operator_endpoint_source(url, bonjour_urls=bonjour_urls)
    if kind == "node" and source == "bonjour_local_hostname":
        return 5
    if kind == "node":
        return 10
    if kind == "relay":
        return 30
    return 80


def operator_endpoint_selection_policy() -> dict[str, Any]:
    return {
        "order": ["last_good", "bonjour_local_hostname", "same_lan_private_ip", "private_tunnel", "https_relay"],
        "failover_timeout_seconds": 2,
        "persist_last_good_endpoint": True,
        "parallel_probe": True,
        "demote_after_failures": 2,
        "background_reprobe_seconds": 30,
    }


def operator_bonjour_roaming_summary(policy: dict[str, Any], *, bonjour_urls: list[str], node_port: int) -> dict[str, Any]:
    return {
        "enabled": bool(get_path(policy, ["discovery", "bonjour_enabled"], True)),
        "service_type": str(get_path(policy, ["discovery", "bonjour_service_type"], "_theseus-hive._tcp")),
        "domain": str(get_path(policy, ["discovery", "bonjour_domain"], "local.")),
        "node_port": node_port,
        "candidate_urls": bonjour_urls,
        "use_when": "same_lan_or_phone_hotspot_macos_ios",
        "signed_txt_records": True,
    }


def operator_roaming_handover_policy() -> dict[str, Any]:
    return {
        "last_good_first": True,
        "parallel_probe": True,
        "probe_timeout_seconds": 2,
        "relay_probe_timeout_seconds": 5,
        "demote_endpoint_after_failures": 2,
        "background_reprobe_seconds": 30,
        "promote_lower_latency_endpoint": True,
        "task_routing": {
            "chat_and_operator": "prefer last-good same-LAN or private tunnel endpoint",
            "training_chunks": "prefer accelerator capability and lease length over latency after coordination is complete",
            "storage_preview": "prefer same-LAN node for thumbnails; use relay/tunnel for offsite browsing",
        },
    }


def operator_roaming_profile(
    policy: dict[str, Any],
    auth_context: dict[str, Any],
    *,
    provided_token: str,
    include_token: bool,
) -> dict[str, Any]:
    status = build_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791)))
    peers = current_peers(policy)
    roaming = operator_roaming_summary(policy, status, peers)
    endpoint_urls = unique_nonempty(list(roaming.get("node_urls") or []) + list(roaming.get("relay_urls") or []))
    payload: dict[str, Any] = {
        "ok": True,
        "policy": "project_theseus_hive_operator_roaming_profile_v1",
        "created_utc": now(),
        "hive_id": hive_id(policy),
        "hive_name": get_path(status, ["hive_name"], hive_id(policy)),
        "tier": hive_tier(policy),
        "access": hive_users.user_summary(auth_context),
        "coordinator_url": endpoint_urls[0] if endpoint_urls else "",
        "coordinator_urls": endpoint_urls,
        "node_urls": roaming.get("node_urls") or [],
        "relay_urls": roaming.get("relay_urls") or [],
        "operator_urls": roaming.get("operator_urls") or [],
        "roaming": {
            **roaming,
            "endpoint_selection": operator_endpoint_selection_policy(),
        },
        "update_catalog_urls": roaming.get("update_catalog_urls") or [],
        "installer_artifacts_urls": roaming.get("installer_artifacts_urls") or [],
        "revocation": operator_revocation_hint(auth_context),
        "security": {
            "raw_public_node_api_allowed": False,
            "raw_public_8791_forbidden": True,
            "remote_execution": "registered_bounded_hive_task_kinds_only",
            "token_echoed_only_when_requested": bool(include_token),
        },
        "no_codex_required": True,
    }
    if include_token and provided_token:
        payload["join_token"] = provided_token
        payload["operator_token"] = provided_token
        payload["ios_app_url"] = ios_profile_url(payload)
        payload["qr_join_url"] = compact_ios_profile_url(payload)
        payload["qr_svg"] = qr_svg_for_url(str(payload.get("qr_join_url") or payload.get("ios_app_url") or ""))
    return payload














































def operator_storage_summary(policy: dict[str, Any], status: dict[str, Any], peers: list[dict[str, Any]]) -> dict[str, Any]:
    local = hive_storage.operator_summary(policy)
    rows = [peer_from_status(status)] + peers
    nodes = []
    for row in rows:
        storage = row.get("storage") if isinstance(row.get("storage"), dict) else {}
        if not storage:
            continue
        nodes.append(
            {
                "node_id": row.get("node_id"),
                "node_name": row.get("node_name"),
                "api_url": row.get("api_url"),
                "is_local": row.get("node_id") == status.get("node_id"),
                "share_count": storage.get("share_count", 0),
                "shares": storage.get("shares") or [],
            }
        )
    return {
        **local,
        "nodes": nodes,
        "node_count": len(nodes),
    }


def operator_remote_control_summary(policy: dict[str, Any], status: dict[str, Any], peers: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(peer_from_status(status), is_local=True)] + [dict(peer, is_local=False) for peer in peers]
    nodes = []
    provider_ids: set[str] = set()
    ready_count = 0
    for row in rows:
        remote_control = row.get("remote_control") if isinstance(row.get("remote_control"), dict) else {}
        providers = remote_control.get("providers") if isinstance(remote_control.get("providers"), list) else []
        compact_providers = []
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            provider_id = str(provider.get("id") or "")
            if provider_id:
                provider_ids.add(provider_id)
            if provider.get("ready"):
                ready_count += 1
            compact_providers.append(
                {
                    "id": provider_id,
                    "label": provider.get("label"),
                    "ready": bool(provider.get("ready")),
                    "installed": bool(provider.get("installed")),
                    "configured": bool(provider.get("configured")),
                    "connect": provider.get("connect") or {},
                }
            )
        nodes.append(
            {
                "node_id": row.get("node_id"),
                "node_name": row.get("node_name"),
                "api_url": row.get("api_url"),
                "is_local": bool(row.get("is_local")),
                "platform": row.get("platform") or {},
                "preferred_provider_id": remote_control.get("preferred_provider_id"),
                "ready_provider_count": remote_control.get("ready_provider_count"),
                "candidate_hosts": get_path(remote_control, ["local", "candidate_hosts"], []),
                "providers": compact_providers,
            }
        )
    return {
        "policy": "project_theseus_hive_operator_remote_control_v0",
        "enabled": bool(get_path(policy, ["remote_control", "enabled"], True)),
        "node_count": len(nodes),
        "ready_provider_count": ready_count,
        "provider_ids": sorted(provider_ids),
        "nodes": nodes,
        "session_endpoint": "/api/hive/remote-control/session",
        "status_endpoint": "/api/hive/remote-control/status",
        "transport_note": "Hive brokers authenticated session metadata; the screen/input stream uses RDP, VNC, RustDesk, Sunshine/Moonlight, LAN, VPN, or provider relay.",
    }


def operator_utilization_control(policy: dict[str, Any], payload: dict[str, Any], auth_context: dict[str, Any] | None = None) -> dict[str, Any]:
    if "utilization_sweep" not in operator_allowed_task_kinds(policy, auth_context):
        return {"ok": False, "error": "utilization_control_not_allowed"}
    action = str(payload.get("action") or "").strip().lower()
    pause_flag = ROOT / "reports" / "hive_utilization_pause.flag"
    stop_flag = ROOT / "reports" / "hive_utilization_stop.flag"
    if action == "pause":
        write_text(pause_flag, now() + "\n")
        return {"ok": True, "action": action, "pause_flag": str(pause_flag.relative_to(ROOT))}
    if action == "resume":
        remove_if_exists(pause_flag)
        return {"ok": True, "action": action, "pause_flag_cleared": True}
    if action == "stop":
        write_text(stop_flag, now() + "\n")
        return {"ok": True, "action": action, "stop_flag": str(stop_flag.relative_to(ROOT))}
    if action in {"clear_stop", "start"}:
        remove_if_exists(stop_flag)
        return {"ok": True, "action": action, "stop_flag_cleared": True}
    if action in {"sweep", "execute_sweep"}:
        task_payload = {
            "source": "mobile_operator",
            "force_requeue": True,
            "job_id": f"operator_utilization_sweep_{int(time.time())}",
        }
        return enqueue_task(policy, "utilization_sweep", task_payload, source="local")
    return {"ok": False, "error": "unknown_utilization_action", "action": action}






def operator_allowed_task_kinds(policy: dict[str, Any], auth_context: dict[str, Any] | None = None) -> list[str]:
    task_defs = policy.get("task_kinds") if isinstance(policy.get("task_kinds"), dict) else {}
    allowed_by_tier = get_path(policy, ["relay", "allowed_remote_task_kinds_by_tier", hive_tier(policy)], [])
    if not isinstance(allowed_by_tier, list):
        allowed_by_tier = []
    forbidden = set(str(item) for item in get_path(policy, ["security", "forbidden_remote_task_kinds"], []))
    allowed = [str(kind) for kind in allowed_by_tier if str(kind) in task_defs and str(kind) not in forbidden]
    fallback = [
        "resource_probe",
        "capability_refresh",
        "readiness_check",
        "checkpoint_chat",
        "compute_market_status",
        "update_status",
        "hive_version_status",
        "hive_version_converge",
    ]
    if not allowed:
        allowed = [kind for kind in fallback if kind in task_defs and kind not in forbidden]
    return hive_users.filter_task_kinds(auth_context, sorted(set(allowed)))


def operator_chat_uses_local_assistant(policy: dict[str, Any], payload: dict[str, Any]) -> bool:
    target = str(payload.get("target_node_id") or "auto").strip().lower()
    route = str(payload.get("route") or payload.get("chat_route") or payload.get("mode") or "").strip().lower()
    if route in {"queue", "task", "remote", "checkpoint_task"}:
        return False
    if route in {"assistant", "local_assistant", "canonical_assistant"}:
        return True
    if target in {"", "auto", "best", "local"}:
        return True
    local = peer_from_status(cached_build_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791))))
    local_ids = {
        str(local.get("node_id") or "").strip().lower(),
        str(local.get("node_name") or "").strip().lower(),
    }
    return bool(target and target in local_ids)


def operator_chat(policy: dict[str, Any], payload: dict[str, Any], auth_context: dict[str, Any] | None = None) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt_required"}
    if operator_chat_uses_local_assistant(policy, payload):
        return operator_chat_local_assistant(policy, payload, prompt, auth_context)
    task_payload = {
        "checkpoint_id": str(payload.get("checkpoint_id") or "live"),
        "prompt": prompt,
        "session_id": str(payload.get("session_id") or f"operator_{payload.get('target_node_id') or 'auto'}"),
        "source": "mobile_operator",
    }
    result = operator_submit_task(
        policy,
        {
            "kind": "checkpoint_chat",
            "target_node_id": str(payload.get("target_node_id") or "auto"),
            "task_payload": task_payload,
        },
        auth_context,
    )
    result["operator_action"] = "chat"
    return result


def operator_chat_local_assistant(
    policy: dict[str, Any],
    payload: dict[str, Any],
    prompt: str,
    auth_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed_task_kinds = operator_allowed_task_kinds(policy, auth_context)
    if "checkpoint_chat" not in allowed_task_kinds:
        return {
            "ok": False,
            "error": "checkpoint_chat_not_operator_allowed",
            "allowed_task_kinds": allowed_task_kinds,
            "access": hive_users.user_summary(auth_context),
        }
    session_id = str(payload.get("session_id") or "operator_local")
    checkpoint_id = str(payload.get("checkpoint_id") or "live")
    intent = str(payload.get("intent") or "auto")
    if intent not in {"auto", "chat", "code", "tool", "planning"}:
        intent = "auto"
    feedback = str(payload.get("feedback") or "completed")
    safe_session = safe_report_slug(session_id) or f"operator_{uuid.uuid4().hex[:8]}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = ROOT / "reports" / f"hive_operator_assistant_{safe_session}_{stamp}.json"
    markdown = ROOT / "reports" / f"hive_operator_assistant_{safe_session}_{stamp}.md"
    events = ROOT / "reports" / f"hive_operator_assistant_{safe_session}_events.jsonl"
    timeout = int(get_path(policy, ["operator", "assistant_runtime_timeout_seconds"], 360))
    command = [
        sys.executable,
        "scripts/theseus_assistant_runtime.py",
        "--prompt",
        prompt,
        "--checkpoint-id",
        checkpoint_id,
        "--session-id",
        session_id,
        "--surface",
        "hive_operator",
        "--intent",
        intent,
        "--feedback",
        feedback,
        "--out",
        rel_path(report),
        "--markdown-out",
        rel_path(markdown),
        "--events-out",
        rel_path(events),
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=theseus_runtime.runtime_env(),
        )
        runtime_report = read_json(report, {})
        compact = compact_operator_assistant_runtime(
            runtime_report,
            report_path=report,
            markdown=markdown,
            events=events,
            returncode=result.returncode,
            runtime_ms=int((time.perf_counter() - started) * 1000),
        )
        write_json(ROOT / "reports" / "hive_operator_assistant_latest.json", compact)
        return {
            "ok": result.returncode == 0 and runtime_report.get("trigger_state") in {"GREEN", "YELLOW"},
            "operator_action": "chat",
            "chat_mode": "canonical_local_assistant",
            "queued": False,
            "target": local_operator_target(policy),
            "assistant_runtime": compact,
            "stdout_tail": "" if result.returncode == 0 else result.stdout[-2000:],
            "stderr_tail": "" if result.returncode == 0 else result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        compact = {
            "trigger_state": "RED",
            "status": "timeout",
            "returncode": 124,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "report": rel_path(report),
            "markdown": rel_path(markdown),
            "events": rel_path(events),
            "summary": {
                "intent": intent,
                "session_id": session_id,
                "checkpoint_id": checkpoint_id,
                "feedback": feedback,
                "public_training_rows_written": 0,
                "runtime_external_inference_calls": 0,
                "fallback_return_count": 0,
            },
        }
        write_json(ROOT / "reports" / "hive_operator_assistant_latest.json", compact)
        return {
            "ok": False,
            "error": "assistant_runtime_timeout",
            "operator_action": "chat",
            "chat_mode": "canonical_local_assistant",
            "queued": False,
            "target": local_operator_target(policy),
            "assistant_runtime": compact,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }


def operator_assistant_feedback(
    policy: dict[str, Any],
    payload: dict[str, Any],
    auth_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed_task_kinds = operator_allowed_task_kinds(policy, auth_context)
    if "checkpoint_chat" not in allowed_task_kinds:
        return {
            "ok": False,
            "error": "checkpoint_chat_not_operator_allowed",
            "allowed_task_kinds": allowed_task_kinds,
            "access": hive_users.user_summary(auth_context),
        }
    outcome = str(payload.get("outcome") or payload.get("feedback") or "").strip().lower()
    if outcome not in {"accepted", "missed", "ignored", "corrected", "completed"}:
        return {"ok": False, "error": "invalid_feedback_outcome", "allowed": ["accepted", "missed", "ignored", "corrected", "completed"]}
    latest = read_json(ROOT / "reports" / "hive_operator_assistant_latest.json", {})
    latest_summary = latest.get("summary") if isinstance(latest.get("summary"), dict) else {}
    session_id = str(payload.get("session_id") or latest_summary.get("session_id") or "operator_local")
    safe_session = safe_report_slug(session_id) or f"operator_{uuid.uuid4().hex[:8]}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    event_out = ROOT / "reports" / f"hive_operator_feedback_event_{safe_session}_{stamp}.json"
    event_md = ROOT / "reports" / f"hive_operator_feedback_event_{safe_session}_{stamp}.md"
    bridge_out = ROOT / "reports" / f"hive_operator_feedback_training_bridge_{safe_session}_{stamp}.json"
    bridge_md = ROOT / "reports" / f"hive_operator_feedback_training_bridge_{safe_session}_{stamp}.md"
    artifact_refs = [
        str(payload.get("artifact_ref") or ""),
        str(latest.get("report") or ""),
        str(latest.get("events") or ""),
    ]
    artifact_refs = [item for item in dict.fromkeys(artifact_refs) if item]
    intent_summary = (
        f"hive_operator_feedback session={safe_session} "
        f"intent={latest_summary.get('intent') or 'unknown'} "
        f"latest_state={latest.get('trigger_state') or 'unknown'}"
    )
    command = [
        sys.executable,
        "scripts/dogfood_trace_event.py",
        "--surface",
        "hive_operator",
        "--assistant-lane",
        str(latest_summary.get("assistant_lane") or "chat_checkpoint"),
        "--outcome",
        outcome,
        "--intent-summary-redacted",
        intent_summary,
        "--duration-ms",
        "0",
        "--out",
        rel_path(event_out),
        "--markdown-out",
        rel_path(event_md),
        "--execute",
    ]
    for ref in artifact_refs:
        command.extend(["--artifact-ref", ref])
    started = time.perf_counter()
    event_result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120, env=theseus_runtime.runtime_env())
    event_report = read_json(event_out, {})
    bridge_result = {"returncode": None, "stdout_tail": "", "stderr_tail": "", "runtime_ms": 0}
    bridge_report: dict[str, Any] = {}
    if event_result.returncode == 0 and event_report.get("event_written") is True:
        bridge_cmd = [
            sys.executable,
            "scripts/dogfood_trace_training_bridge.py",
            "--out",
            rel_path(bridge_out),
            "--markdown-out",
            rel_path(bridge_md),
            "--execute",
        ]
        bridge_started = time.perf_counter()
        proc = subprocess.run(bridge_cmd, cwd=ROOT, text=True, capture_output=True, timeout=180, env=theseus_runtime.runtime_env())
        bridge_result = {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "runtime_ms": int((time.perf_counter() - bridge_started) * 1000),
        }
        bridge_report = read_json(bridge_out, {})
    compact = {
        "ok": event_result.returncode == 0 and event_report.get("event_written") is True,
        "operator_action": "assistant_feedback",
        "outcome": outcome,
        "session_id": session_id,
        "event_written": event_report.get("event_written"),
        "event_report": rel_path(event_out),
        "event_markdown": rel_path(event_md),
        "training_bridge_report": rel_path(bridge_out) if bridge_report else "",
        "training_bridge_state": bridge_report.get("trigger_state"),
        "training_rows_written": get_path(bridge_report, ["summary", "training_rows_written"], 0),
        "artifact_refs": artifact_refs,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "event_stdout_tail": "" if event_result.returncode == 0 else event_result.stdout[-2000:],
        "event_stderr_tail": "" if event_result.returncode == 0 else event_result.stderr[-2000:],
        "bridge_result": bridge_result,
    }
    write_json(ROOT / "reports" / "hive_operator_feedback_latest.json", compact)
    return compact


def compact_operator_assistant_runtime(
    runtime_report: dict[str, Any],
    *,
    report_path: Path,
    markdown: Path,
    events: Path,
    returncode: int,
    runtime_ms: int,
) -> dict[str, Any]:
    summary = runtime_report.get("summary") if isinstance(runtime_report.get("summary"), dict) else {}
    assistant_text = str(runtime_report.get("assistant_text") or "")
    return {
        "trigger_state": runtime_report.get("trigger_state") or "MISSING",
        "returncode": returncode,
        "runtime_ms": runtime_ms,
        "report": rel_path(report_path),
        "markdown": rel_path(markdown),
        "events": rel_path(events),
        "summary": {
            "intent": summary.get("intent"),
            "session_id": summary.get("session_id"),
            "checkpoint_id": summary.get("checkpoint_id"),
            "assistant_lane": summary.get("assistant_lane"),
            "feedback": summary.get("feedback"),
            "vcm_context_ready": summary.get("vcm_context_ready"),
            "vcm_selected_page_count": summary.get("vcm_selected_page_count"),
            "checkpoint_history_turns_loaded": summary.get("checkpoint_history_turns_loaded"),
            "code_private_probe_state": summary.get("code_private_probe_state"),
            "code_private_probe_selected_pass_rate": summary.get("code_private_probe_selected_pass_rate"),
            "tool_evidence_state": summary.get("tool_evidence_state"),
            "tool_evidence_result_count": summary.get("tool_evidence_result_count"),
            "tool_evidence_tool_on_solve_rate": summary.get("tool_evidence_tool_on_solve_rate"),
            "dogfood_event_written": summary.get("dogfood_event_written"),
            "dogfood_training_rows_written": summary.get("dogfood_training_rows_written"),
            "latest_public_run": summary.get("latest_public_run"),
            "latest_public_cards": summary.get("latest_public_cards"),
            "latest_public_score": summary.get("latest_public_score"),
            "latest_public_task_count": summary.get("latest_public_task_count"),
            "latest_public_measurement_kind": summary.get("latest_public_measurement_kind"),
            "latest_public_dominant_residual": summary.get("latest_public_dominant_residual"),
            "public_training_rows_written": summary.get("public_training_rows_written"),
            "runtime_external_inference_calls": summary.get("runtime_external_inference_calls"),
            "fallback_return_count": summary.get("fallback_return_count"),
        },
        "assistant_text": assistant_text[:6000],
        "assistant_text_truncated": len(assistant_text) > 6000,
        "vcm_context_packet": runtime_report.get("vcm_context_packet") if isinstance(runtime_report.get("vcm_context_packet"), dict) else {},
        "code_route": runtime_report.get("code_route") if isinstance(runtime_report.get("code_route"), dict) else {},
        "code_private_probe": compact_named_report(runtime_report.get("code_private_probe")),
        "tool_evidence": compact_named_report(runtime_report.get("tool_evidence")),
        "plan_context": runtime_report.get("plan_context") if isinstance(runtime_report.get("plan_context"), dict) else {},
        "teacher_policy": runtime_report.get("teacher_policy") if isinstance(runtime_report.get("teacher_policy"), dict) else {},
        "benchmark_status": runtime_report.get("benchmark_status") if isinstance(runtime_report.get("benchmark_status"), dict) else {},
        "public_benchmark_boundary": runtime_report.get("public_benchmark_boundary") if isinstance(runtime_report.get("public_benchmark_boundary"), dict) else {},
    }


def compact_named_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "active": value.get("active"),
        "trigger_state": value.get("trigger_state"),
        "report": value.get("report"),
        "markdown": value.get("markdown"),
        "trace": value.get("trace"),
        "summary": value.get("summary") if isinstance(value.get("summary"), dict) else {},
    }


def local_operator_target(policy: dict[str, Any]) -> dict[str, Any]:
    status = cached_build_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791)))
    local = peer_from_status(status)
    return {key: local.get(key) for key in ["node_id", "node_name", "api_url"]} | {"is_local": True}


def operator_assistant_summary() -> dict[str, Any]:
    state = read_json(ROOT / "reports" / "theseus_assistant_state_report.json", {})
    e2e = read_json(ROOT / "reports" / "theseus_assistant_e2e.json", {})
    runtime = read_json(ROOT / "reports" / "theseus_assistant_runtime.json", {})
    latest = read_json(ROOT / "reports" / "hive_operator_assistant_latest.json", {})
    latest_feedback = read_json(ROOT / "reports" / "hive_operator_feedback_latest.json", {})
    benchmark_measurement = read_json(ROOT / "reports" / "theseus_benchmark_measurement.json", {})
    state_summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
    e2e_summary = e2e.get("summary") if isinstance(e2e.get("summary"), dict) else {}
    runtime_summary = runtime.get("summary") if isinstance(runtime.get("summary"), dict) else {}
    latest_summary = latest.get("summary") if isinstance(latest.get("summary"), dict) else {}
    benchmark_summary = (
        benchmark_measurement.get("summary")
        if isinstance(benchmark_measurement.get("summary"), dict)
        else {}
    )
    return {
        "enabled": True,
        "canonical_runtime": "scripts/theseus_assistant_runtime.py",
        "chat_endpoint": "/api/hive/operator/chat",
        "assistant_feedback_endpoint": "/api/hive/operator/assistant-feedback",
        "local_chat_mode": "canonical_local_assistant",
        "remote_chat_mode": "bounded_checkpoint_chat_task",
        "state": state.get("trigger_state"),
        "e2e_state": e2e.get("trigger_state"),
        "runtime_state": runtime.get("trigger_state"),
        "assistant_cases": state_summary.get("assistant_cases") or f"{e2e_summary.get('passed_case_count')}/{e2e_summary.get('case_count')}",
        "vcm_ready": state_summary.get("runtime_vcm_ready") if state_summary else runtime_summary.get("vcm_context_ready"),
        "tool_evidence_state": state_summary.get("assistant_tool_evidence_state") or e2e_summary.get("tool_evidence_state"),
        "tool_evidence_result_count": state_summary.get("assistant_tool_evidence_result_count") or e2e_summary.get("tool_evidence_result_count"),
        "private_code_probe_state": state_summary.get("private_code_probe_state") or runtime_summary.get("code_private_probe_state"),
        "private_code_probe_selected_pass_rate": state_summary.get("private_code_probe_selected_pass_rate") or runtime_summary.get("code_private_probe_selected_pass_rate"),
        "dogfood_trainable_events": state_summary.get("dogfood_trainable_events"),
        "benchmark": {
            "latest_public_run": state_summary.get("latest_public_surface")
            or latest_summary.get("latest_public_run")
            or e2e_summary.get("latest_public_run"),
            "latest_public_cards": benchmark_summary.get("effective_cards")
            if isinstance(benchmark_summary.get("effective_cards"), list)
            else latest_summary.get("latest_public_cards"),
            "latest_public_score": state_summary.get("latest_public_score")
            or latest_summary.get("latest_public_score")
            or e2e_summary.get("latest_public_score"),
            "latest_public_task_count": state_summary.get("latest_public_task_count")
            or latest_summary.get("latest_public_task_count")
            or e2e_summary.get("latest_public_task_count"),
            "latest_public_measurement_kind": state_summary.get("benchmark_measurement_kind")
            or latest_summary.get("latest_public_measurement_kind")
            or e2e_summary.get("latest_public_measurement_kind"),
            "latest_public_dominant_residual": state_summary.get("public_residual_failure_counts")
            or latest_summary.get("latest_public_dominant_residual")
            or e2e_summary.get("latest_public_dominant_residual"),
            "public_training_rows_written": latest_summary.get("public_training_rows_written", 0),
            "runtime_external_inference_calls": latest_summary.get("runtime_external_inference_calls", 0),
            "fallback_return_count": latest_summary.get("fallback_return_count", 0),
        },
        "latest": {
            "trigger_state": latest.get("trigger_state"),
            "intent": latest_summary.get("intent"),
            "session_id": latest_summary.get("session_id"),
            "feedback": latest_summary.get("feedback"),
            "vcm_context_ready": latest_summary.get("vcm_context_ready"),
            "checkpoint_history_turns_loaded": latest_summary.get("checkpoint_history_turns_loaded"),
            "dogfood_event_written": latest_summary.get("dogfood_event_written"),
            "latest_public_run": latest_summary.get("latest_public_run"),
            "latest_public_score": latest_summary.get("latest_public_score"),
            "latest_public_task_count": latest_summary.get("latest_public_task_count"),
            "latest_public_measurement_kind": latest_summary.get("latest_public_measurement_kind"),
            "report": latest.get("report"),
        },
        "teacher": {
            "gate_state": get_path(latest, ["teacher_policy", "gate_state"], state_summary.get("teacher_distillation_gate_state")),
            "distillation_allowed": get_path(latest, ["teacher_policy", "distillation_allowed"], state_summary.get("teacher_distillation_allowed")),
            "teacher_accepted_row_share": get_path(latest, ["teacher_policy", "teacher_accepted_row_share"], state_summary.get("teacher_accepted_row_share")),
            "runtime_external_tokens_forbidden": get_path(latest, ["teacher_policy", "runtime_external_tokens_forbidden"], state_summary.get("teacher_runtime_external_tokens_forbidden")),
            "teacher_apply_mode_forbidden": get_path(latest, ["teacher_policy", "teacher_apply_mode_forbidden"], None),
        },
        "latest_feedback": {
            "ok": latest_feedback.get("ok"),
            "outcome": latest_feedback.get("outcome"),
            "event_written": latest_feedback.get("event_written"),
            "training_bridge_state": latest_feedback.get("training_bridge_state"),
            "training_rows_written": latest_feedback.get("training_rows_written"),
            "event_report": latest_feedback.get("event_report"),
        },
        "public_boundary": {
            "benchmarks_may_be_run_for_measurement": True,
            "train_on_public_prompts_tests_solutions_traces_or_scores": False,
            "runtime_external_inference_calls": 0,
        },
    }


def safe_report_slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:80]


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def operator_submit_task(policy: dict[str, Any], payload: dict[str, Any], auth_context: dict[str, Any] | None = None) -> dict[str, Any]:
    kind = str(payload.get("kind") or "")
    if not kind:
        return {"ok": False, "error": "kind_required"}
    allowed_task_kinds = operator_allowed_task_kinds(policy, auth_context)
    if kind not in allowed_task_kinds:
        return {
            "ok": False,
            "error": "task_kind_not_operator_allowed",
            "kind": kind,
            "allowed_task_kinds": allowed_task_kinds,
            "access": hive_users.user_summary(auth_context),
        }
    task_payload = payload.get("task_payload") if isinstance(payload.get("task_payload"), dict) else {}
    target = select_operator_target(policy, kind, str(payload.get("target_node_id") or "auto"))
    if not target.get("ok"):
        return target
    if target.get("is_local"):
        result = enqueue_task(policy, kind, task_payload, source="local")
    else:
        result = submit_task(policy, str(target.get("api_url") or ""), kind, task_payload)
    result["operator_action"] = "task_submit"
    result["target"] = {key: target.get(key) for key in ["node_id", "node_name", "api_url", "is_local"]}
    return result


def select_operator_target(policy: dict[str, Any], kind: str, target_node_id: str) -> dict[str, Any]:
    status = build_status(policy, http_port=int(get_path(policy, ["node", "http_port"], 8791)))
    local = peer_from_status(status)
    local["is_local"] = True
    nodes = [local] + [dict(peer, is_local=False) for peer in current_peers(policy)]
    if target_node_id and target_node_id not in {"auto", "best", "local"}:
        for node in nodes:
            if str(node.get("node_id") or "") == target_node_id:
                if not node.get("api_url"):
                    return {"ok": False, "error": "target_missing_api_url", "node_id": target_node_id}
                if not node_can_run_task(node, kind):
                    return {
                        "ok": False,
                        "error": "target_lacks_task_slot",
                        "kind": kind,
                        "node_id": target_node_id,
                        "node_name": node.get("node_name"),
                        "available_slots": node.get("slots") or [],
                        "capabilities": node.get("capabilities") or [],
                    }
                return {"ok": True, **node}
        return {"ok": False, "error": "target_node_not_found", "node_id": target_node_id}
    if target_node_id == "local":
        if not node_can_run_task(local, kind):
            return {
                "ok": False,
                "error": "target_lacks_task_slot",
                "kind": kind,
                "node_id": local.get("node_id"),
                "node_name": local.get("node_name"),
                "available_slots": local.get("slots") or [],
                "capabilities": local.get("capabilities") or [],
            }
        return {"ok": True, **local}

    def score(node: dict[str, Any]) -> float:
        total = 0.0
        if node.get("is_local"):
            total += 0.1
        for slot in node.get("slots") or []:
            if kind in (slot.get("task_kinds") or []):
                total += 2.0
                if slot.get("available"):
                    total += 1.0
        for cap in node.get("capabilities") or []:
            cap_id = str(cap.get("id") or "")
            if kind == "checkpoint_chat" and cap_id == "checkpoint_chat_gateway":
                total += 3.0
            if kind.startswith("cuda_") and "cuda" in cap_id:
                total += 2.0
            if kind.startswith("mlx_") and "mlx" in cap_id:
                total += 2.0
            total += float(cap.get("score") or 0.0) * 0.05
        return total

    viable = [node for node in nodes if node.get("api_url") and node_can_run_task(node, kind)]
    if not viable:
        return {
            "ok": False,
            "error": "no_operator_targets_for_task",
            "kind": kind,
            "node_count": len(nodes),
            "nodes": [
                {
                    "node_id": node.get("node_id"),
                    "node_name": node.get("node_name"),
                    "api_url": node.get("api_url"),
                    "slots": node.get("slots") or [],
                    "accelerators": node_accelerator_ids(node),
                }
                for node in nodes
            ],
        }
    best = max(viable, key=score)
    return {"ok": True, **best}


def operator_learning_summary() -> dict[str, Any]:
    scoreboard = read_json(ROOT / "reports" / "learning_scoreboard.json", {})
    broad = read_json(ROOT / "reports" / "broad_transfer_matrix.json", {})
    graduation = read_json(ROOT / "reports" / "real_code_benchmark_graduation.json", {})
    public_transfer = get_path(scoreboard, ["public_transfer"], {})
    broad_public = get_path(scoreboard, ["broad_public_transfer"], {})
    return {
        "scoreboard_state": scoreboard.get("state") or scoreboard.get("trigger_state"),
        "public_pass_rate": public_transfer.get("real_public_task_pass_rate") or broad_public.get("real_public_task_pass_rate"),
        "broad_pass_rate": get_path(broad, ["summary", "real_public_task_pass_rate"], broad.get("real_public_task_pass_rate")),
        "floor": get_path(broad, ["summary", "floor"], get_path(graduation, ["thresholds", "floor"], 0.70)),
        "promotion_allowed": graduation.get("promotion_allowed"),
        "student_evidence": graduation.get("student_evidence") or graduation.get("evidence_level"),
    }


def operator_benchmark_summary() -> dict[str, Any]:
    ledger = read_json(ROOT / "reports" / "benchmark_ledger.json", {})
    broad = read_json(ROOT / "reports" / "broad_transfer_matrix.json", {})
    cards = broad.get("cards") if isinstance(broad.get("cards"), list) else broad.get("matrix")
    if not isinstance(cards, list):
        cards = []
    return {
        "frontier": read_json(ROOT / "reports" / "frontier_policy_status.json", {}),
        "benchmarks": (ledger.get("benchmarks") or ledger.get("rows") or [])[:24] if isinstance(ledger, dict) else [],
        "broad_cards": cards[:12],
        "summary": broad.get("summary", {}) if isinstance(broad, dict) else {},
    }


def operator_game_summary() -> dict[str, Any]:
    rl = read_json(ROOT / "reports" / "rl_benchmark_registry.json", {})
    minecraft = read_json(ROOT / "reports" / "minecraft_runtime_probe.json", {})
    game_assets = read_json(ROOT / "reports" / "game_asset_inventory.json", {})
    return {
        "rl_summary": rl.get("summary", {}) if isinstance(rl, dict) else {},
        "rl_cards": (rl.get("cards") or rl.get("registry") or [])[:12] if isinstance(rl, dict) else [],
        "minecraft": {
            "ok": minecraft.get("ok"),
            "status": minecraft.get("status") or minecraft.get("trigger_state"),
            "next_action": minecraft.get("next_action"),
        },
        "assets": game_assets.get("summary", {}) if isinstance(game_assets, dict) else {},
    }


def operator_autonomy_summary() -> dict[str, Any]:
    vacation = read_json(ROOT / "reports" / "vacation_mode_supervisor.json", {})
    daemon = read_json(ROOT / "reports" / "sparkstream_status.json", {})
    viea = read_json(ROOT / "reports" / "viea_autonomy_spine.json", {})
    actions = read_json(ROOT / "reports" / "feedback_action_queue.json", {})
    return {
        "vacation_mode": {
            "ok": vacation.get("ok"),
            "state": vacation.get("state") or vacation.get("trigger_state"),
            "progress_contract": vacation.get("progress_contract"),
        },
        "daemon": {
            "running": daemon.get("running"),
            "status": daemon.get("status") or daemon.get("trigger_state"),
        },
        "viea": {
            "ok": viea.get("ok"),
            "state": viea.get("state") or viea.get("trigger_state"),
        },
        "next_actions": (actions.get("actions") or actions.get("queue") or [])[:8] if isinstance(actions, dict) else [],
    }


def operator_vcm_summary() -> dict[str, Any]:
    probe = read_json(ROOT / "reports" / "virtual_context_memory_probe.json", {})
    bench = read_json(ROOT / "reports" / "virtual_context_memory_bench.json", {})
    compiled = read_json(ROOT / "reports" / "virtual_context_compiled_context.json", {})
    graph = read_json(ROOT / "reports" / "virtual_context_memory_graph.json", {})
    snapshots = read_json(ROOT / "reports" / "virtual_context_memory_snapshots.json", {})
    training = read_json(ROOT / "reports" / "virtual_context_memory_training_admission.json", {})
    consumer_audit = read_json(ROOT / "reports" / "virtual_context_memory_consumer_audit.json", {})
    context_recovery = read_json(ROOT / "reports" / "vcm_context_recovery_benchmark.json", {})
    summary = get_path(probe, ["summary"], {})
    faults = compiled.get("semantic_page_faults") if isinstance(compiled.get("semantic_page_faults"), list) else []
    fault_counts: dict[str, int] = {}
    for row in faults:
        if isinstance(row, dict):
            kind = str(row.get("fault_type") or "unknown")
            fault_counts[kind] = fault_counts.get(kind, 0) + 1
    conflicts = [row for row in graph.get("edges", []) if isinstance(row, dict) and row.get("type") in {"contradicts", "supersedes", "invalidates"}]
    repairs = []
    if probe.get("trigger_state") != "GREEN":
        repairs.append({"priority": "high", "action": "refresh_virtual_context_memory", "reason": "VCM probe is not green."})
    for kind, count in sorted(fault_counts.items()):
        if kind == "capacity_fault":
            repairs.append({"priority": "medium", "action": "rebalance_context_budget", "reason": f"{count} pages faulted under capacity pressure."})
        elif kind == "deletion_fault":
            repairs.append({"priority": "high", "action": "respect_tombstone_closure", "reason": f"{count} deleted/invalidated pages were blocked."})
        else:
            repairs.append({"priority": "medium", "action": "inspect_vcm_faults", "reason": f"{count} {kind} faults are active."})
    if conflicts:
        repairs.append({"priority": "medium", "action": "inspect_vcm_graph_conflicts", "reason": f"{len(conflicts)} conflict/supersession/invalidation edges are present."})
    if consumer_audit.get("trigger_state") and consumer_audit.get("trigger_state") != "GREEN":
        repairs.append({"priority": "medium", "action": "review_vcm_consumer_audit", "reason": "High-value memory consumers are not fully VCM-integrated."})
    if context_recovery.get("trigger_state") and context_recovery.get("trigger_state") != "GREEN":
        repairs.append({"priority": "high", "action": "run_vcm_context_recovery_benchmark", "reason": "VCM context recovery is not green."})
    elif not context_recovery.get("trigger_state"):
        repairs.append({"priority": "medium", "action": "run_vcm_context_recovery_benchmark", "reason": "VCM context recovery has not been measured yet."})
    return {
        "policy": "project_theseus_operator_vcm_summary_v1",
        "state": probe.get("trigger_state") or "MISSING",
        "probe_created_utc": probe.get("created_utc"),
        "page_count": summary.get("semantic_pages"),
        "event_count": summary.get("event_count"),
        "graph_edge_count": summary.get("graph_edge_count"),
        "fault_count": len(faults),
        "fault_counts": fault_counts,
        "graph_conflict_count": len(conflicts),
        "latest_snapshot": snapshots.get("active_snapshot"),
        "bench_state": summary.get("vcm_bench_state"),
        "bench_policy": bench.get("policy"),
        "context_recovery_state": context_recovery.get("trigger_state") or "MISSING",
        "context_recovery_policy": context_recovery.get("policy"),
        "context_recovery_vcm_accuracy": get_path(context_recovery, ["summary", "vcm_answer_accuracy"], None),
        "context_recovery_best_baseline_accuracy": get_path(context_recovery, ["summary", "best_baseline_answer_accuracy"], None),
        "context_recovery_best_baseline_system": get_path(context_recovery, ["summary", "best_baseline_system"], ""),
        "training_admission_state": training.get("trigger_state"),
        "consumer_audit_state": consumer_audit.get("trigger_state"),
        "packet_only_consumer_count": get_path(consumer_audit, ["summary", "packet_only_consumer_count"], 0),
        "direct_only_consumer_count": get_path(consumer_audit, ["summary", "direct_only_consumer_count"], 0),
        "recommended_repairs": repairs[:8],
        "query_endpoint": "/api/hive/vcm/status",
        "external_inference_calls": int(summary.get("external_inference_calls") or 0),
    }








def is_local_task_source(source: str) -> bool:
    if not source or source == "local":
        return True
    if source.startswith("http:"):
        host = source.split(":", 1)[1]
        return is_loopback(host)
    return False


def tasks_report(policy: dict[str, Any]) -> dict[str, Any]:
    ledger = read_jsonl_tail(task_ledger_path(policy), 50)
    queued = read_jsonl_tail(ROOT / str(get_path(policy, ["node", "task_queue_path"], "reports/hive_task_queue.jsonl")), 50)
    return {
        "policy": "project_theseus_hive_tasks_v0",
        "created_utc": now(),
        "queued_in_memory": TASKS.qsize(),
        "slots": slots_snapshot(),
        "recent_queued": queued,
        "recent_results": ledger,
    }






























def authorize_task(policy: dict[str, Any], client_host: str, provided_secret: str, kind: str = "") -> tuple[bool, str]:
    if get_path(policy, ["security", "allow_loopback_tasks_without_secret"], True) and is_loopback(client_host):
        return True, "loopback"
    if not get_path(policy, ["security", "requires_shared_secret_for_remote_tasks"], True):
        return True, "secret_not_required"
    auth = hive_users.authorize(policy, client_host, provided_secret, action="task", task_kind=kind, allow_loopback=False)
    if auth.get("ok"):
        return True, str(auth.get("reason") or "token_ok")
    return False, str(auth.get("error") or "shared_secret_required_for_remote_task")


def authorize_secret_or_loopback(policy: dict[str, Any], client_host: str, provided_secret: str, query: str = "") -> tuple[bool, str]:
    if is_loopback(client_host):
        return True, "loopback"
    auth = hive_users.authorize(policy, client_host, provided_secret, query, action="status", allow_loopback=False)
    if auth.get("ok"):
        return True, str(auth.get("reason") or "token_ok")
    return False, str(auth.get("error") or "shared_secret_required")


def submit_task(policy: dict[str, Any], peer_url: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps({"kind": kind, "payload": payload}).encode("utf-8")
    req = urlrequest.Request(
        peer_url.rstrip("/") + "/api/hive/tasks",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    secret = join_token(policy) or shared_secret(policy)
    if secret:
        req.add_header("X-Theseus-Hive-Secret", secret)
    timeout = float(get_path(policy, ["node", "task_submit_timeout_seconds"], 30))
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local/private hive endpoint.
            raw = response.read().decode("utf-8")
    except URLError as exc:
        return {"ok": False, "error": str(exc), "peer_url": peer_url, "kind": kind}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_response", "body": raw[:500]}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_response"}






































def find_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()




































if __name__ == "__main__":
    raise SystemExit(main())
