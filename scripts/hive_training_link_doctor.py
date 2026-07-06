"""Report whether Hive nodes can exchange bounded training tasks.

This is a non-mutating fleet doctor. It checks the local Hive status plus the
latest peer registry and scheduler output, then answers the practical question:
can a Mac send CUDA work to this Windows/NVIDIA node, and can Windows send MLX
work back to a Mac worker?
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "hive_training_link_doctor.json"
DEFAULT_MARKDOWN = REPORTS / "hive_training_link_doctor.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh local Hive probe and scheduler reports first.")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    refresh_results = refresh_inputs() if args.refresh else []
    local_status = read_json(REPORTS / "hive_status.json", {})
    peer_report = read_json(REPORTS / "hive_peers.json", {})
    live_status = fetch_json("http://127.0.0.1:8791/api/hive/status", {})
    live_peers = fetch_json("http://127.0.0.1:8791/api/hive/peers", {})
    if live_status:
        local_status = live_status
    if live_peers:
        peer_report = live_peers
    scheduler = read_json(REPORTS / "hive_scheduler.json", {})
    policy = read_json(ROOT / "configs" / "hive_policy.json", {})

    nodes = collect_nodes(local_status, peer_report, policy)
    report = build_report(nodes, scheduler, policy, refresh_results=refresh_results)
    out_path = resolve_report_path(args.out)
    md_path = resolve_report_path(args.markdown_out)
    write_json(out_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("state") in {"GREEN", "YELLOW"} else 2


def refresh_inputs() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    live_status = fetch_json("http://127.0.0.1:8791/api/hive/status", {})
    live_peers = fetch_json("http://127.0.0.1:8791/api/hive/peers", {})
    commands: list[list[str]] = []
    if live_status:
        write_json(REPORTS / "hive_status.json", live_status)
        if live_peers:
            write_json(REPORTS / "hive_peers.json", live_peers)
        results.append(
            {
                "command": ["GET", "http://127.0.0.1:8791/api/hive/status"],
                "ok": True,
                "returncode": 0,
                "started_utc": now(),
                "stdout_tail": "captured live daemon status before scheduler refresh",
                "stderr_tail": "",
            }
        )
    else:
        commands.append([sys.executable, "scripts/hive_node.py", "probe"])
    commands.append([sys.executable, "scripts/hive_scheduler.py", "--worker-chunks", "--out", "reports/hive_scheduler.json"])
    for command in commands:
        started = datetime.now(timezone.utc)
        try:
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
            results.append(
                {
                    "command": command,
                    "ok": result.returncode == 0,
                    "returncode": result.returncode,
                    "started_utc": started.isoformat(),
                    "stdout_tail": result.stdout[-2000:],
                    "stderr_tail": result.stderr[-2000:],
                }
            )
        except subprocess.TimeoutExpired as exc:
            results.append(
                {
                    "command": command,
                    "ok": False,
                    "returncode": 124,
                    "started_utc": started.isoformat(),
                    "stdout_tail": exc.stdout[-2000:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": exc.stderr[-2000:] if isinstance(exc.stderr, str) else "",
                }
            )
    return results


def build_report(
    nodes: list[dict[str, Any]],
    scheduler: dict[str, Any],
    policy: dict[str, Any],
    *,
    refresh_results: list[dict[str, Any]],
) -> dict[str, Any]:
    local_node = next((node for node in nodes if node.get("is_local")), {})
    cuda_nodes = [node for node in nodes if can_run(node, "cuda_eval_chunk")]
    mlx_nodes = [node for node in nodes if can_run(node, "mlx_eval_chunk")]
    windows_cuda = [node for node in cuda_nodes if node_system(node) == "Windows"]
    mac_mlx = [node for node in mlx_nodes if node_system(node) == "Darwin"]
    remote_cuda = [node for node in cuda_nodes if not node.get("is_local")]
    remote_mlx = [node for node in mlx_nodes if not node.get("is_local")]
    allowed = set(allowed_task_kinds(policy))
    scheduler_placements = scheduler.get("placements") if isinstance(scheduler.get("placements"), list) else []
    link_checks = {
        "local_can_run_cuda": can_run(local_node, "cuda_eval_chunk"),
        "local_can_run_mlx": can_run(local_node, "mlx_eval_chunk"),
        "local_remote_tasks_require_secret": bool(get_path(local_node, ["security", "remote_tasks_require_secret"], True)),
        "local_shared_secret_configured": bool(get_path(local_node, ["security", "shared_secret_configured"], False)),
        "scheduler_shared_secret_configured": bool(get_path(scheduler, ["safety", "shared_secret_configured"], False)),
        "worker_receivers_shared_secret_configured": worker_receivers_have_secrets(nodes),
        "worker_receivers_missing_secret": worker_receivers_missing_secret(nodes),
        "worker_receivers_secret_unknown": worker_receivers_secret_unknown(nodes),
        "windows_cuda_receiver_ready": bool(windows_cuda),
        "mac_mlx_receiver_ready": bool(mac_mlx),
        "remote_cuda_target_ready": bool(remote_cuda),
        "remote_mlx_target_ready": bool(remote_mlx),
        "scheduler_has_cuda_worker_placement": any(str(row.get("task_kind") or "").startswith("cuda_") for row in scheduler_placements),
        "scheduler_has_mlx_worker_placement": any(str(row.get("task_kind") or "").startswith("mlx_") for row in scheduler_placements),
        "cuda_tasks_allowed": bool({"cuda_eval_chunk", "cuda_training_chunk", "cuda_rollout_chunk"} & allowed),
        "mlx_tasks_allowed": bool({"mlx_eval_chunk", "mlx_training_chunk", "mlx_rollout_chunk"} & allowed),
    }
    missing = missing_items(nodes, link_checks, allowed)
    state = "GREEN" if not missing else ("YELLOW" if nodes and (cuda_nodes or mlx_nodes) else "RED")
    return {
        "ok": state in {"GREEN", "YELLOW"},
        "policy": "project_theseus_hive_training_link_doctor_v0",
        "created_utc": now(),
        "state": state,
        "summary": {
            "node_count": len(nodes),
            "cuda_node_count": len(cuda_nodes),
            "mlx_node_count": len(mlx_nodes),
            "windows_cuda_node_count": len(windows_cuda),
            "mac_mlx_node_count": len(mac_mlx),
            "bidirectional_training_ready": not missing,
            "local_platform": node_system(local_node) or platform.system(),
        },
        "link_checks": link_checks,
        "nodes": [compact_node(node) for node in nodes],
        "recommended_commands": recommended_commands(cuda_nodes, mlx_nodes),
        "missing": missing,
        "next_actions": next_actions(missing),
        "refresh_results": refresh_results,
    }


def missing_items(nodes: list[dict[str, Any]], checks: dict[str, Any], allowed: set[str]) -> list[str]:
    missing: list[str] = []
    if not nodes:
        return ["no_hive_status_or_peers"]
    if not checks["windows_cuda_receiver_ready"]:
        missing.append("no_windows_cuda_receiver")
    if not checks["mac_mlx_receiver_ready"]:
        missing.append("no_mac_mlx_receiver")
    if "cuda_eval_chunk" not in allowed:
        missing.append("cuda_worker_tasks_not_allowed_by_policy")
    if "mlx_eval_chunk" not in allowed:
        missing.append("mlx_worker_tasks_not_allowed_by_policy")
    if checks.get("local_remote_tasks_require_secret") and not checks.get("local_shared_secret_configured"):
        missing.append("shared_secret_not_configured")
    if not checks.get("worker_receivers_shared_secret_configured"):
        if checks.get("worker_receivers_missing_secret"):
            missing.append("worker_receiver_secret_not_configured")
        if checks.get("worker_receivers_secret_unknown"):
            missing.append("worker_receiver_secret_unknown")
    return missing


def next_actions(missing: list[str]) -> list[str]:
    actions: list[str] = []
    if "no_windows_cuda_receiver" in missing:
        actions.append("Start the Hive daemon on the Windows/NVIDIA node and run `theseus cuda doctor --refresh`.")
    if "no_mac_mlx_receiver" in missing:
        actions.append("On the Mac, install/start the Hive node, install MLX, then run `./bin/theseus.sh device list` or `python scripts/hive_node.py probe`.")
    if "cuda_worker_tasks_not_allowed_by_policy" in missing or "mlx_worker_tasks_not_allowed_by_policy" in missing:
        actions.append("Check `configs/hive_policy.json` for the private/friends allowed remote task kinds.")
    if "shared_secret_not_configured" in missing:
        actions.append("Set the same `THESEUS_HIVE_SECRET` or active Hive invite token on both nodes before sending remote tasks.")
    if "worker_receiver_secret_not_configured" in missing:
        actions.append("Restart or rejoin worker nodes that report `shared_secret_configured=false` before queueing cross-node training.")
    if "worker_receiver_secret_unknown" in missing:
        actions.append("Run `theseus hive network-doctor --timeout 8` and refresh this doctor; the peer did not return authenticated security details reliably enough to prove worker-task readiness.")
    if not actions:
        actions.append("Run `theseus schedule --execute --worker-chunks` to submit bounded worker chunks, then sync reports with `python scripts/hive_artifact_sync.py --peer-url <peer> --limit 50`.")
    return actions


def recommended_commands(cuda_nodes: list[dict[str, Any]], mlx_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    cuda_target = first_remote_or_any(cuda_nodes)
    mlx_target = first_remote_or_any(mlx_nodes)
    return {
        "send_cuda_to_windows_or_linux": command_for(cuda_target, "cuda_eval_chunk", "manual_cuda_eval_from_peer"),
        "send_mlx_to_mac": command_for(mlx_target, "mlx_eval_chunk", "manual_mlx_eval_from_peer"),
        "run_scheduler_chunks": "theseus schedule --execute --worker-chunks",
        "sync_artifacts": "python scripts/hive_artifact_sync.py --peer-url <peer-api-url> --limit 50",
    }


def command_for(node: dict[str, Any], kind: str, chunk_id: str) -> str:
    api_url = str(node.get("api_url") or "<peer-api-url>")
    payload = json.dumps({"profile": "smoke", "chunk_id": chunk_id}, separators=(",", ":"))
    if platform.system() == "Windows":
        escaped = payload.replace('"', '\\"')
        return f'theseus do {kind} --peer-url {api_url} --payload-json "{escaped}"'
    return f"./bin/theseus.sh do {kind} --peer-url {api_url} --payload-json '{payload}'"


def first_remote_or_any(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    for node in nodes:
        if not node.get("is_local"):
            return node
    return nodes[0] if nodes else {}


def collect_nodes(local_status: dict[str, Any], peer_report: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    local = peer_report.get("local_node") if isinstance(peer_report.get("local_node"), dict) else {}
    if not local and local_status:
        local = peer_from_status(local_status)
    elif local and local_status:
        local = {**local, "security": local_status.get("security") or local.get("security") or {}}
    if local:
        nodes.append({**local, "is_local": True})
    for peer in peer_report.get("peers") or []:
        if isinstance(peer, dict):
            nodes.append({**peer, "is_local": False})
    return enrich_nodes(dedupe_nodes(nodes), policy)


def enrich_nodes(nodes: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    secret = hive_secret(policy)
    headers = {"X-Theseus-Hive-Secret": secret} if secret else {}
    for node in nodes:
        api_url = str(node.get("api_url") or "").rstrip("/")
        status: dict[str, Any] = {}
        auth_probe: dict[str, Any] = {}
        auth_ok = False
        if api_url:
            status = fetch_json(api_url + "/api/hive/status", {}, headers=headers, timeout=8.0)
            if secret:
                auth_probe = authenticated_probe(api_url, headers=headers, timeout=8.0)
                auth_ok = bool(auth_probe.get("ok")) if isinstance(auth_probe, dict) else False
            if isinstance(status, dict) and status.get("node_id"):
                security = effective_security(status, node, auth_probe)
                node = {
                    **node,
                    "node_id": status.get("node_id") or node.get("node_id"),
                    "node_name": status.get("node_name") or node.get("node_name"),
                    "api_url": status.get("api_url") or node.get("api_url"),
                    "platform": status.get("platform") or node.get("platform"),
                    "capabilities": status.get("capabilities") or node.get("capabilities") or [],
                    "resources": status.get("resources") or node.get("resources") or {},
                    "slots": status.get("slots") or node.get("slots") or [],
                    "security": security,
                    "security_observed": bool(security),
                    "auth_status_verified": auth_ok,
                    "auth_probe_path": auth_probe.get("probe_path", "") if isinstance(auth_probe, dict) else "",
                    "auth_probe_policy": auth_probe.get("policy", "") if isinstance(auth_probe, dict) else "",
                    "operator_auth_verified": auth_ok,
                    "reachability": {
                        "status_ok": True,
                        "auth_status_ok": auth_ok,
                        "auth_probe_path": auth_probe.get("probe_path", "") if isinstance(auth_probe, dict) else "",
                        "operator_auth_ok": auth_ok,
                    },
                }
            elif auth_ok:
                hive = auth_probe.get("hive") if isinstance(auth_probe.get("hive"), dict) else {}
                local_node = hive.get("local_node") if isinstance(hive.get("local_node"), dict) else {}
                auth_node = local_node or auth_probe
                auth_security = auth_probe.get("security") if isinstance(auth_probe.get("security"), dict) else {}
                node = {
                    **node,
                    "node_id": auth_node.get("node_id") or node.get("node_id"),
                    "node_name": auth_node.get("node_name") or node.get("node_name"),
                    "api_url": auth_node.get("api_url") or node.get("api_url"),
                    "platform": auth_node.get("platform") or node.get("platform"),
                    "capabilities": auth_node.get("capabilities") or node.get("capabilities") or [],
                    "resources": auth_node.get("resources") or node.get("resources") or {},
                    "slots": auth_node.get("slots") or node.get("slots") or [],
                    "security": auth_security or {
                        "remote_tasks_require_secret": True,
                        "shared_secret_configured": True,
                        "source": auth_probe.get("probe_path") or "auth_status_verified",
                    },
                    "security_observed": True,
                    "auth_status_verified": True,
                    "auth_probe_path": auth_probe.get("probe_path", ""),
                    "auth_probe_policy": auth_probe.get("policy", ""),
                    "operator_auth_verified": True,
                    "reachability": {"status_ok": False, "auth_status_ok": True, "auth_probe_path": auth_probe.get("probe_path", ""), "operator_auth_ok": True},
                }
            elif api_url:
                node = {
                    **node,
                    "security_observed": bool(node.get("security")),
                    "auth_status_verified": False,
                    "auth_probe_path": auth_probe.get("probe_path", "") if isinstance(auth_probe, dict) else "",
                    "auth_probe_policy": auth_probe.get("policy", "") if isinstance(auth_probe, dict) else "",
                    "operator_auth_verified": False,
                    "reachability": {"status_ok": False, "auth_status_ok": False, "auth_probe_path": auth_probe.get("probe_path", "") if isinstance(auth_probe, dict) else "", "operator_auth_ok": False},
                }
        out.append(node)
    return out


def authenticated_probe(api_url: str, *, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    auth = fetch_json(api_url.rstrip("/") + "/api/hive/auth/status", {}, headers=headers, timeout=timeout)
    if isinstance(auth, dict) and auth.get("ok"):
        auth["probe_path"] = "/api/hive/auth/status"
        return auth
    fallback = fetch_json(api_url.rstrip("/") + "/api/hive/operator/status", {}, headers=headers, timeout=timeout)
    if isinstance(fallback, dict):
        fallback["probe_path"] = "/api/hive/operator/status"
        fallback["fallback_reason"] = auth.get("error") or auth.get("message") or "auth_status_unavailable" if isinstance(auth, dict) else "auth_status_unavailable"
        return fallback
    return {}


def effective_security(status: dict[str, Any], node: dict[str, Any], auth_probe: dict[str, Any]) -> dict[str, Any]:
    security = status.get("security") if isinstance(status.get("security"), dict) else {}
    if not security:
        security = node.get("security") if isinstance(node.get("security"), dict) else {}
    auth_security = auth_probe.get("security") if isinstance(auth_probe.get("security"), dict) else {}
    if auth_security:
        return {
            **security,
            **auth_security,
            "source": auth_probe.get("probe_path") or auth_security.get("source") or "auth_status_verified",
        }
    if security:
        return security
    if auth_probe.get("ok"):
        return {
            "remote_tasks_require_secret": True,
            "shared_secret_configured": True,
            "source": auth_probe.get("probe_path") or "auth_status_verified",
        }
    return {}


def peer_from_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": status.get("node_id"),
        "node_name": status.get("node_name"),
        "hostname": status.get("hostname"),
        "api_url": status.get("api_url"),
        "platform": status.get("platform"),
        "capabilities": status.get("capabilities") or [],
        "resources": status.get("resources") or {},
        "security": status.get("security") or {},
        "slots": status.get("slots") or [],
    }


def dedupe_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for node in nodes:
        key = str(node.get("node_id") or node.get("api_url") or node.get("node_name") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(node)
    return out


def compact_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_name": node.get("node_name"),
        "api_url": node.get("api_url"),
        "is_local": bool(node.get("is_local")),
        "platform": node.get("platform"),
        "security": node.get("security") or {},
        "security_observed": bool(node.get("security_observed")),
        "auth_status_verified": bool(node.get("auth_status_verified")),
        "auth_probe_path": node.get("auth_probe_path") or get_path(node, ["reachability", "auth_probe_path"], ""),
        "auth_probe_policy": node.get("auth_probe_policy"),
        "operator_auth_verified": bool(node.get("operator_auth_verified")),
        "reachability": node.get("reachability") or {},
        "accelerators": accelerator_ids(node),
        "task_kinds": sorted({str(kind) for slot in node.get("slots") or [] for kind in (slot.get("task_kinds") or [])}),
        "can_run": {
            "cuda_eval_chunk": can_run(node, "cuda_eval_chunk"),
            "cuda_training_chunk": can_run(node, "cuda_training_chunk"),
            "cuda_rollout_chunk": can_run(node, "cuda_rollout_chunk"),
            "mlx_eval_chunk": can_run(node, "mlx_eval_chunk"),
            "mlx_training_chunk": can_run(node, "mlx_training_chunk"),
            "mlx_rollout_chunk": can_run(node, "mlx_rollout_chunk"),
        },
    }


def can_run(node: dict[str, Any], kind: str) -> bool:
    if not node:
        return False
    for slot in node.get("slots") or []:
        if kind in set(str(item) for item in (slot.get("task_kinds") or [])):
            return True
    accelerators = set(accelerator_ids(node))
    if kind.startswith("cuda_"):
        return "nvidia_cuda" in accelerators
    if kind.startswith("mlx_"):
        return bool({"mlx_apple", "apple_mlx", "mlx_cuda"} & accelerators)
    return False


def worker_receivers_missing_secret(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes:
        if not (can_run(node, "cuda_eval_chunk") or can_run(node, "mlx_eval_chunk")):
            continue
        if not node.get("security_observed") and not node.get("auth_status_verified") and not node.get("operator_auth_verified"):
            continue
        if not get_path(node, ["security", "remote_tasks_require_secret"], True):
            continue
        if get_path(node, ["security", "shared_secret_configured"], False):
            continue
        out.append({"node_id": node.get("node_id"), "node_name": node.get("node_name"), "api_url": node.get("api_url")})
    return out


def worker_receivers_secret_unknown(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes:
        if not (can_run(node, "cuda_eval_chunk") or can_run(node, "mlx_eval_chunk")):
            continue
        if node.get("security_observed") or node.get("auth_status_verified") or node.get("operator_auth_verified"):
            continue
        out.append({"node_id": node.get("node_id"), "node_name": node.get("node_name"), "api_url": node.get("api_url")})
    return out


def worker_receivers_have_secrets(nodes: list[dict[str, Any]]) -> bool:
    return not worker_receivers_missing_secret(nodes) and not worker_receivers_secret_unknown(nodes)


def hive_secret(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["security", "shared_secret_env"], "THESEUS_HIVE_SECRET"))
    if os.environ.get(env_name):
        return str(os.environ.get(env_name) or "")
    join = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    if isinstance(join, dict) and join.get("join_token"):
        return str(join.get("join_token") or "")
    profiles = read_json(ROOT / str(get_path(policy, ["federation", "profiles_path"], "configs/hive_profiles.local.json")), {})
    active = str(profiles.get("active_profile_id") or "") if isinstance(profiles, dict) else ""
    for profile in profiles.get("profiles", []) if isinstance(profiles.get("profiles"), list) else []:
        if isinstance(profile, dict) and (not active or profile.get("profile_id") == active):
            token = str(profile.get("join_token") or "")
            if token:
                return token
    return ""


def accelerator_ids(node: dict[str, Any]) -> list[str]:
    caps = {str(cap.get("id") or "") for cap in node.get("capabilities") or [] if isinstance(cap, dict)}
    resources = node.get("resources") if isinstance(node.get("resources"), dict) else {}
    mlx = resources.get("mlx") if isinstance(resources.get("mlx"), dict) else {}
    nvidia = resources.get("nvidia") if isinstance(resources.get("nvidia"), dict) else {}
    backend_ids = {str(item) for item in mlx.get("backend_ids") or []} if isinstance(mlx, dict) else set()
    out = caps | backend_ids
    if isinstance(nvidia, dict) and nvidia.get("available"):
        out.add("nvidia_cuda")
    if isinstance(mlx, dict) and mlx.get("available") and mlx.get("platform_is_macos"):
        out.add("apple_mlx")
        out.add("mlx_apple")
    return sorted(item for item in out if item)


def allowed_task_kinds(policy: dict[str, Any]) -> list[str]:
    task_defs = set((policy.get("task_kinds") or {}).keys()) if isinstance(policy.get("task_kinds"), dict) else set()
    tier = str((policy.get("federation") or {}).get("tier") or "private")
    allowed = get_path(policy, ["relay", "allowed_remote_task_kinds_by_tier", tier], [])
    forbidden = set(str(item) for item in get_path(policy, ["security", "forbidden_remote_task_kinds"], []))
    if not isinstance(allowed, list):
        return []
    return sorted(str(kind) for kind in allowed if str(kind) in task_defs and str(kind) not in forbidden)


def node_system(node: dict[str, Any]) -> str:
    platform_info = node.get("platform") if isinstance(node.get("platform"), dict) else {}
    return str(platform_info.get("system") or "")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hive Training Link Doctor",
        "",
        f"- State: **{report.get('state')}**",
        f"- Created: `{report.get('created_utc')}`",
        f"- Nodes: `{get_path(report, ['summary', 'node_count'], 0)}`",
        f"- Windows/NVIDIA receivers: `{get_path(report, ['summary', 'windows_cuda_node_count'], 0)}`",
        f"- Mac/MLX receivers: `{get_path(report, ['summary', 'mac_mlx_node_count'], 0)}`",
        f"- Bidirectional training ready: `{get_path(report, ['summary', 'bidirectional_training_ready'], False)}`",
        "",
        "## Nodes",
        "",
        "| Node | Local | Platform | Accelerators | Auth Probe | CUDA | MLX |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for node in report.get("nodes") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(node.get("node_name") or node.get("node_id") or ""),
                    str(bool(node.get("is_local"))),
                    str(get_path(node, ["platform", "system"], "")),
                    ", ".join(node.get("accelerators") or []),
                    str(node.get("auth_probe_path") or "not_verified"),
                    str(get_path(node, ["can_run", "cuda_eval_chunk"], False)),
                    str(get_path(node, ["can_run", "mlx_eval_chunk"], False)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions") or []:
        lines.append(f"- {action}")
    lines.extend(["", "## Commands", ""])
    commands = report.get("recommended_commands") if isinstance(report.get("recommended_commands"), dict) else {}
    for name, command in commands.items():
        lines.append(f"- `{name}`: `{command}`")
    return "\n".join(lines) + "\n"


def get_path(obj: Any, path: list[Any], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def fetch_json(url: str, default: Any, *, headers: dict[str, str] | None = None, timeout: float = 5.0) -> Any:
    request = urlrequest.Request(url, headers=headers or {})
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:  # noqa: S310 - trusted local/private Hive API.
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_report_path(value: str) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
