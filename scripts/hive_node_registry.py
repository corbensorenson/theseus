"""Authoritative Hive node registry.

The registry is the shared node view used by scheduling, utilization, board
assignment, fleet readiness, and artifact sync. It intentionally normalizes
the same facts everywhere: trust, reachability, capabilities, resource
blockers, training blockers, and best-node summaries.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import platform
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_POLICY = ROOT / "configs" / "hive_policy.json"
DEFAULT_OUT = REPORTS / "hive_node_registry.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    policy = read_json(resolve(args.policy), {})
    registry = build_registry(policy)
    write_json(resolve(args.out), registry)
    print(json.dumps(registry, indent=2))
    return 0 if registry.get("ok") else 2


def build_registry(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy if isinstance(policy, dict) else read_json(DEFAULT_POLICY, {})
    sources: list[dict[str, Any]] = []
    nodes: dict[str, dict[str, Any]] = {}

    if bool(get_path(policy, ["node", "registry_operator_status_enabled"], False)):
        operator = live_operator_status(policy)
        if operator.get("ok"):
            sources.append({"source": "live_operator_status", "ok": True})
            for node in nodes_from_operator(operator):
                merge_node(nodes, node, source="live_operator_status")
        else:
            sources.append({"source": "live_operator_status", "ok": False, "error": operator.get("error")})
    else:
        sources.append({"source": "live_operator_status", "ok": None, "skipped": True, "reason": "scheduler_registry_uses_lightweight_status_and_peers"})

    status = read_json(ROOT / str(get_path(policy, ["node", "status_path"], "reports/hive_status.json")), {})
    file_peers = read_json(ROOT / str(get_path(policy, ["node", "peers_path"], "reports/hive_peers.json")), {})
    if status:
        merge_node(nodes, {**peer_from_status(status), "is_local": True}, source="hive_status")
    local_peer = file_peers.get("local_node") if isinstance(file_peers.get("local_node"), dict) else {}
    if local_peer:
        merge_node(nodes, {**local_peer, "is_local": True}, source="hive_peers.local_node")
    merge_peers_report(nodes, file_peers, source="hive_peers")
    sources.append({"source": "hive_status_and_peers", "ok": bool(status or file_peers), "peer_count": len(file_peers.get("peers") or []) if isinstance(file_peers, dict) else 0})

    live_peers = live_peers_report(policy)
    if live_peers.get("ok") or live_peers.get("peers") or live_peers.get("known_peers") or live_peers.get("stale_peers"):
        local_live = live_peers.get("local_node") if isinstance(live_peers.get("local_node"), dict) else {}
        if local_live:
            merge_node(nodes, {**local_live, "is_local": True}, source="live_peers.local_node")
        merge_peers_report(nodes, live_peers, source="live_peers")
        sources.append(
            {
                "source": "live_peers",
                "ok": bool(live_peers.get("ok", True)),
                "peer_count": len(live_peers.get("peers") or []) if isinstance(live_peers, dict) else 0,
                "known_peer_count": len(live_peers.get("known_peers") or []) if isinstance(live_peers, dict) else 0,
                "stale_peer_count": len(live_peers.get("stale_peers") or []) if isinstance(live_peers, dict) else 0,
            }
        )

    utilization = read_json(REPORTS / "hive_utilization_manager.json", {})
    for row in utilization.get("nodes", []) if isinstance(utilization.get("nodes"), list) else []:
        if isinstance(row, dict):
            merge_node(nodes, sanitize_utilization_node(row), source="hive_utilization_manager")
    if utilization:
        sources.append({"source": "hive_utilization_manager", "ok": True, "node_count": len(utilization.get("nodes") or [])})

    normalized = [normalize_node(policy, row) for row in nodes.values() if row.get("node_id")]
    normalized.sort(key=lambda row: (not bool(row.get("is_local")), str(row.get("node_name") or row.get("node_id"))))
    network_doctor = network_doctor_alignment(policy)
    summary = summarize(policy, normalized, network_doctor)
    registry = {
        "ok": True,
        "policy": "project_theseus_hive_node_registry_v1",
        "created_utc": now(),
        "hive_id": hive_id(policy),
        "shared_secret_configured": bool(hive_secret(policy)),
        "sources": sources,
        "network_doctor": network_doctor,
        "summary": summary,
        "nodes": normalized,
        "score_semantics": "authoritative local Hive node view for scheduling and unattended readiness; not capability evidence",
        "external_inference_calls": 0,
    }
    return registry


def load_or_build(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = read_json(DEFAULT_OUT, {})
    if existing.get("policy") == "project_theseus_hive_node_registry_v1" and existing.get("nodes"):
        return existing
    return build_registry(policy)


def nodes_for_scheduler(policy: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return build_registry(policy).get("nodes", [])


def live_operator_status(policy: dict[str, Any]) -> dict[str, Any]:
    port = int(get_path(policy, ["node", "http_port"], 8791) or 8791)
    timeout = float(get_path(policy, ["node", "registry_operator_status_timeout_seconds"], 1.0) or 1.0)
    headers = secret_headers(policy)
    return fetch_json(f"http://127.0.0.1:{port}/api/hive/operator/status", headers=headers, timeout=max(0.25, min(10.0, timeout)))


def nodes_from_operator(operator: dict[str, Any]) -> list[dict[str, Any]]:
    hive = operator.get("hive") if isinstance(operator.get("hive"), dict) else {}
    out = []
    local = hive.get("local_node") if isinstance(hive.get("local_node"), dict) else {}
    if local:
        out.append({**local, "is_local": True})
    for peer in hive.get("peers", []) if isinstance(hive.get("peers"), list) else []:
        if isinstance(peer, dict):
            out.append({**peer, "is_local": False})
    return out


def live_peers_report(policy: dict[str, Any]) -> dict[str, Any]:
    port = int(get_path(policy, ["node", "http_port"], 8791) or 8791)
    headers = secret_headers(policy)
    return fetch_json(f"http://127.0.0.1:{port}/api/hive/peers", headers=headers, timeout=4.0)


def merge_peers_report(nodes: dict[str, dict[str, Any]], peers: dict[str, Any], *, source: str) -> None:
    for key in ["peers", "known_peers", "stale_peers"]:
        for peer in peers.get(key, []) if isinstance(peers.get(key), list) else []:
            if isinstance(peer, dict):
                merge_node(nodes, {**peer, "is_local": False}, source=f"{source}.{key}")


def sanitize_utilization_node(node: dict[str, Any]) -> dict[str, Any]:
    """Keep utilization capacity facts without making old reachability sticky."""

    out = dict(node)
    for key in [
        "reachable",
        "blocked",
        "flapping",
        "online",
        "discovery_state",
        "last_seen_utc",
        "last_outbound_verified_utc",
        "last_failure_utc",
        "last_error",
        "age_seconds",
        "outbound_age_seconds",
        "failure_age_seconds",
        "consecutive_failures",
        "reachability",
    ]:
        out.pop(key, None)
    return out


def merge_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any], *, source: str) -> None:
    node_id = str(node.get("node_id") or "")
    if not node_id:
        return
    existing = nodes.get(node_id, {})
    merged = deep_merge(existing, node)
    for url_key in ("api_url", "dashboard_url"):
        merged[url_key] = prefer_non_loopback_url(existing.get(url_key), node.get(url_key), merged.get(url_key))
    sources = list(merged.get("registry_sources") or [])
    if source not in sources:
        sources.append(source)
    merged["registry_sources"] = sources
    if node.get("is_local") is True:
        merged["is_local"] = True
    elif "is_local" not in merged:
        merged["is_local"] = False
    nodes[node_id] = merged


def prefer_non_loopback_url(existing: Any, incoming: Any, merged: Any) -> Any:
    existing_s = str(existing or "")
    incoming_s = str(incoming or "")
    merged_s = str(merged or "")
    if is_non_loopback_url(existing_s) and not is_non_loopback_url(incoming_s):
        return existing_s
    if is_non_loopback_url(incoming_s) and not is_non_loopback_url(existing_s):
        return incoming_s
    return merged_s or incoming_s or existing_s


def is_non_loopback_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    host = parsed.hostname or ""
    return bool(host and host_scope(host) != "loopback")


def deep_merge(left: Any, right: Any) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        out = dict(left)
        for key, value in right.items():
            if value in (None, "", [], {}):
                continue
            out[key] = deep_merge(out.get(key), value)
        return out
    return right if right not in (None, "", [], {}) else left


def normalize_node(policy: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    capabilities = normalize_capabilities(node.get("capabilities") or [])
    resources = node.get("resources") if isinstance(node.get("resources"), dict) else {}
    slots = normalize_slots(node.get("slots") if isinstance(node.get("slots"), list) else [], capabilities, policy)
    network = node_network_profile(node)
    reachability = reachability_state(policy, node, network)
    trust = trust_state(policy, node, network)
    storage_profile = runtime_storage_profile(policy, node)
    reachability_blockers = reachability_blockers_for_node(node, reachability)
    resource_blockers = resource_blockers_for_node(policy, node, storage_profile)
    training_blockers = training_blockers_for_node(policy, node, resource_blockers, storage_profile)
    version_blockers = version_blockers_for_node(node)
    training_blockers = sorted(set(training_blockers + version_blockers + reachability_blockers))
    resource_blockers = sorted(set(resource_blockers + reachability_blockers))
    capability_ids = [str(cap.get("id") or "") for cap in capabilities if cap.get("id")]
    accelerator_ids = [item for item in capability_ids if item in {"nvidia_cuda", "rust_cuda", "mlx_apple", "apple_mlx", "mlx_cuda"}]
    return {
        "node_id": node.get("node_id"),
        "node_name": node.get("node_name"),
        "hostname": node.get("hostname"),
        "api_url": node.get("api_url"),
        "dashboard_url": node.get("dashboard_url"),
        "hive_id": node.get("hive_id") or hive_id(policy),
        "federation_tier": node.get("federation_tier"),
        "platform": node.get("platform") or platform_summary(),
        "is_local": bool(node.get("is_local")),
        "capabilities": capabilities,
        "capability_ids": capability_ids,
        "accelerator_ids": accelerator_ids,
        "resources": resources,
        "slots": slots,
        "runtime_paths": node.get("runtime_paths") or {},
        "training_storage": storage_profile,
        "updates": node.get("updates") or {},
        "tasks": node.get("tasks") if isinstance(node.get("tasks"), dict) else {},
        "storage": node.get("storage") or {},
        "remote_control": node.get("remote_control") or {},
        "network": network,
        "reachability": reachability,
        "trust": trust,
        "resource_blockers": resource_blockers,
        "training_blockers": training_blockers,
        "light_task_allowed": bool(trust.get("trusted") and reachability.get("usable_for_light_tasks") and not version_blockers and not severe_resource_blocked(resource_blockers)),
        "training_allowed": bool(trust.get("trusted") and reachability.get("usable_for_training") and not training_blockers and has_training_slot(slots)),
        "last_seen_utc": node.get("last_seen_utc") or node.get("created_utc"),
        "registry_sources": node.get("registry_sources") or [],
    }


def summarize(policy: dict[str, Any], nodes: list[dict[str, Any]], network_doctor: dict[str, Any] | None = None) -> dict[str, Any]:
    trusted = [node for node in nodes if get_path(node, ["trust", "trusted"], False)]
    light = [node for node in trusted if node.get("light_task_allowed")]
    training = [node for node in trusted if node.get("training_allowed")]
    remote = [node for node in nodes if not node.get("is_local")]
    reachable_remote = [node for node in remote if get_path(node, ["reachability", "outbound_verified"], False)]
    blocked_remote = [node for node in remote if not get_path(node, ["reachability", "usable_for_light_tasks"], False)]
    network_doctor = network_doctor if isinstance(network_doctor, dict) else {}
    caps: dict[str, int] = {}
    idle_slots: dict[str, int] = {}
    blockers: dict[str, int] = {}
    for node in nodes:
        for cap in node.get("capability_ids") or []:
            caps[cap] = caps.get(cap, 0) + 1
        for slot in node.get("slots") or []:
            if slot.get("available"):
                slot_type = str(slot.get("slot_type") or "")
                idle_slots[slot_type] = idle_slots.get(slot_type, 0) + int(slot.get("capacity") or 1)
        for blocker in (node.get("resource_blockers") or []) + (node.get("training_blockers") or []):
            blockers[str(blocker)] = blockers.get(str(blocker), 0) + 1
    best_training = best_for_capability(training, ["nvidia_cuda", "mlx_apple", "apple_mlx", "cpu_worker"]) or {}
    best_cuda = best_for_capability(training, ["nvidia_cuda"], strict=True) or {}
    best_mlx = best_for_capability(training, ["mlx_apple", "apple_mlx", "mlx_cuda"], strict=True) or {}
    best_inference = best_for_capability(light, ["apple_mlx", "mlx_apple", "nvidia_cuda", "cpu_worker"]) or {}
    best_cpu = best_for_capability(light, ["cpu_worker"]) or {}
    doctor_state = str(network_doctor.get("state") or "MISSING")
    doctor_blocks_remote = doctor_state in {"RED", "MISSING", "STALE", "UNKNOWN"}
    distributed_ready = bool(len(training) > 1 and reachable_remote and not doctor_blocks_remote)
    mixed_cuda_mlx_ready = bool(distributed_ready and best_cuda and best_mlx)
    return {
        "node_count": len(nodes),
        "trusted_node_count": len(trusted),
        "remote_node_count": len(remote),
        "remote_outbound_verified_count": len(reachable_remote),
        "remote_blocked_or_stale_count": len(blocked_remote),
        "training_eligible_node_count": len(training),
        "light_eligible_node_count": len(light),
        "capabilities": caps,
        "idle_slots": idle_slots,
        "blockers": blockers,
        "best_training_node": best_training.get("node_name"),
        "best_cuda_node": best_cuda.get("node_name"),
        "best_mlx_node": best_mlx.get("node_name"),
        "best_inference_node": best_inference.get("node_name"),
        "best_cpu_node": best_cpu.get("node_name"),
        "remote_task_trust": "ready" if hive_secret(policy) and reachable_remote else "trusted_but_unreachable" if hive_secret(policy) and remote else "local_only_or_no_remote_peer",
        "network_doctor_state": doctor_state,
        "network_doctor_fresh": bool(network_doctor.get("fresh")),
        "network_doctor_age_seconds": network_doctor.get("age_seconds"),
        "network_doctor_finding_codes": network_doctor.get("finding_codes") if isinstance(network_doctor.get("finding_codes"), list) else [],
        "network_doctor_red_finding_codes": network_doctor.get("red_finding_codes") if isinstance(network_doctor.get("red_finding_codes"), list) else [],
        "distributed_training_ready": distributed_ready,
        "mixed_cuda_mlx_training_ready": mixed_cuda_mlx_ready,
        "remote_cuda_live_ready": bool(best_cuda and any(node.get("node_id") == best_cuda.get("node_id") and not node.get("is_local") for node in training)),
        "local_mlx_ready": bool(any(node.get("is_local") and {"mlx_apple", "apple_mlx", "mlx_cuda"} & set(node.get("capability_ids") or []) for node in training)),
    }


def network_doctor_alignment(policy: dict[str, Any]) -> dict[str, Any]:
    path = REPORTS / "hive_network_doctor.json"
    report = read_json(path, {})
    if not isinstance(report, dict) or not report:
        return {
            "ok": False,
            "state": "MISSING",
            "fresh": False,
            "report": "reports/hive_network_doctor.json",
            "reason": "network_doctor_report_missing",
            "finding_codes": ["network_doctor_report_missing"],
            "red_finding_codes": ["network_doctor_report_missing"],
            "yellow_finding_codes": [],
        }
    created = str(report.get("created_utc") or "")
    age = timestamp_age_seconds(created)
    stale_after = float(get_path(policy, ["node", "stale_after_seconds"], 45) or 45)
    freshness_seconds = max(180.0, stale_after * 4.0)
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    codes = [str(row.get("code") or row.get("id") or "") for row in findings if isinstance(row, dict) and (row.get("code") or row.get("id"))]
    red_codes = [
        str(row.get("code") or row.get("id") or "")
        for row in findings
        if isinstance(row, dict) and str(row.get("severity") or "").upper() == "RED" and (row.get("code") or row.get("id"))
    ]
    yellow_codes = [
        str(row.get("code") or row.get("id") or "")
        for row in findings
        if isinstance(row, dict) and str(row.get("severity") or "").upper() == "YELLOW" and (row.get("code") or row.get("id"))
    ]
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    for key, target in [("finding_codes", codes), ("red_finding_codes", red_codes), ("yellow_finding_codes", yellow_codes)]:
        if isinstance(summary.get(key), list):
            target[:] = [str(item) for item in summary.get(key) if item]
    state = str(report.get("state") or "UNKNOWN")
    fresh = bool(age is not None and age <= freshness_seconds)
    return {
        "ok": bool(report.get("ok")),
        "state": "STALE" if not fresh else state,
        "reported_state": state,
        "fresh": fresh,
        "created_utc": created,
        "age_seconds": int(age) if age is not None else None,
        "freshness_seconds": int(freshness_seconds),
        "report": "reports/hive_network_doctor.json",
        "summary": {
            "coordinator_reachable": summary.get("coordinator_reachable"),
            "remote_peer_reachable_count": summary.get("remote_peer_reachable_count"),
            "remote_peer_inbound_only_count": summary.get("remote_peer_inbound_only_count"),
            "stale_peer_count": summary.get("stale_peer_count"),
            "red_findings": summary.get("red_findings"),
            "yellow_findings": summary.get("yellow_findings"),
            "auth_secret_proof_count": summary.get("auth_secret_proof_count"),
            "remote_shared_secret_unreported_authenticated_count": summary.get("remote_shared_secret_unreported_authenticated_count"),
        },
        "finding_codes": codes,
        "red_finding_codes": red_codes,
        "yellow_finding_codes": yellow_codes,
        "interpretation": "use this doctor state as fleet readiness evidence; scheduler still performs live checks before queueing work",
    }


def trust_state(policy: dict[str, Any], node: dict[str, Any], network: dict[str, Any]) -> dict[str, Any]:
    expected_hive = hive_id(policy)
    node_hive = str(node.get("hive_id") or expected_hive)
    same_hive = not expected_hive or node_hive == expected_hive
    api_url = str(node.get("api_url") or "")
    secret_ok = bool(hive_secret(policy))
    private_network = str(network.get("class") or "") in {"local", "lan_or_private_tunnel"}
    trusted = bool(same_hive and api_url and (node.get("is_local") or secret_ok) and private_network)
    reasons = []
    if not same_hive:
        reasons.append("hive_id_mismatch")
    if not api_url:
        reasons.append("missing_api_url")
    if not secret_ok and not node.get("is_local"):
        reasons.append("remote_secret_missing")
    if not private_network:
        reasons.append("network_not_private_or_tunnel")
    return {
        "trusted": trusted,
        "same_hive": same_hive,
        "shared_secret_configured": secret_ok,
        "private_network": private_network,
        "reasons": reasons,
    }


def resource_blockers_for_node(policy: dict[str, Any], node: dict[str, Any], storage_profile: dict[str, Any] | None = None) -> list[str]:
    cfg = get_path(policy, ["hive_utilization"], {})
    resources = node.get("resources") if isinstance(node.get("resources"), dict) else {}
    storage_profile = storage_profile or runtime_storage_profile(policy, node)
    blockers: list[str] = []
    disk_free = number(get_path(resources, ["disk", "free_gib"], None))
    min_disk = float(get_path(cfg, ["min_disk_free_gib"], 5) or 5)
    if disk_free is not None and disk_free < min_disk and not storage_profile.get("source_disk_floor_suppressed"):
        blockers.append("disk_below_utilization_floor")
    memory_load = number(get_path(resources, ["memory", "load_percent"], None))
    max_mem = float(get_path(cfg, ["max_memory_load_percent"], 92) or 92)
    if memory_load is not None and memory_load > max_mem:
        blockers.append("memory_load_above_floor")
    return blockers


def training_blockers_for_node(
    policy: dict[str, Any],
    node: dict[str, Any],
    base: list[str],
    storage_profile: dict[str, Any] | None = None,
) -> list[str]:
    cfg = get_path(policy, ["hive_utilization"], {})
    blockers = list(base)
    resources = node.get("resources") if isinstance(node.get("resources"), dict) else {}
    storage_profile = storage_profile or runtime_storage_profile(policy, node)
    disk_free = number(get_path(resources, ["disk", "free_gib"], None))
    min_training = float(get_path(cfg, ["min_training_disk_free_gib"], 10) or 10)
    if disk_free is not None and disk_free < min_training and not storage_profile.get("source_disk_floor_suppressed"):
        blockers.append("disk_below_training_floor")
    for gpu in get_path(resources, ["nvidia", "gpus"], []) or []:
        if not isinstance(gpu, dict):
            continue
        total = number(gpu.get("memory_total_mib")) or 0
        used = number(gpu.get("memory_used_mib")) or 0
        mem_pct = used / total * 100.0 if total else 0.0
        if (number(gpu.get("utilization_gpu_percent")) or 0.0) > float(get_path(cfg, ["max_gpu_utilization_percent_to_enqueue"], 85) or 85):
            blockers.append("gpu_busy")
        if mem_pct > float(get_path(cfg, ["max_gpu_memory_used_percent_to_enqueue"], 88) or 88):
            blockers.append("gpu_vram_busy")
    return blockers


def runtime_storage_profile(policy: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """Classify whether low source-drive space should block training.

    Windows nodes keep source code on C: but generated artifacts, reports,
    checkpoints, caches, and Cargo targets may be junctioned to D:. In that
    setup, low C: free space should remain visible as a warning but should not
    idle CUDA training slots whose heavy writes are already redirected.
    """

    cfg = get_path(policy, ["hive_utilization"], {})
    min_disk = float(get_path(cfg, ["min_disk_free_gib"], 5) or 5)
    min_training = float(get_path(cfg, ["min_training_disk_free_gib"], 10) or 10)
    resources = node.get("resources") if isinstance(node.get("resources"), dict) else {}
    source_root = str(get_path(resources, ["disk", "root"], "") or "")
    source_free = number(get_path(resources, ["disk", "free_gib"], None))
    runtime_paths = node.get("runtime_paths") if isinstance(node.get("runtime_paths"), dict) else {}
    required = ["runtime_root", "data_dir", "cache_dir", "reports_dir", "checkpoints_dir", "cargo_target_dir"]
    present_paths = [str(runtime_paths.get(key) or "") for key in required if runtime_paths.get(key)]
    missing = [key for key in required if not runtime_paths.get(key)]
    anchors = sorted({path_anchor(path) for path in present_paths if path_anchor(path)})
    source_anchor = path_anchor(source_root)
    redirected = bool(present_paths and not missing and len(anchors) == 1 and source_anchor and anchors[0] != source_anchor)
    runtime_free = disk_free_gib_for_existing_path(present_paths[0]) if present_paths else None
    runtime_has_room = bool(runtime_free is not None and runtime_free >= max(min_disk, min_training))
    source_disk_low = bool(source_free is not None and source_free < max(min_disk, min_training))
    suppressed = bool(redirected and runtime_has_room and source_disk_low)
    return {
        "source_root": source_root,
        "source_anchor": source_anchor,
        "source_free_gib": source_free,
        "source_disk_low": source_disk_low,
        "runtime_paths_present": len(present_paths),
        "runtime_paths_missing": missing,
        "runtime_anchor": anchors[0] if len(anchors) == 1 else None,
        "runtime_free_gib": runtime_free,
        "runtime_has_room": runtime_has_room,
        "generated_writes_redirected": redirected,
        "source_disk_floor_suppressed": suppressed,
        "reason": (
            "source_disk_low_but_generated_training_writes_redirected"
            if suppressed
            else "source_disk_floor_enforced"
        ),
    }


def path_anchor(raw: str) -> str:
    value = str(raw or "").strip()
    if len(value) >= 2 and value[1] == ":":
        return value[:2].upper()
    if value.startswith("/"):
        return "/"
    return ""


def disk_free_gib_for_existing_path(raw: str) -> float | None:
    if not raw:
        return None
    path = Path(raw)
    probe = path
    for _ in range(8):
        if probe.exists():
            try:
                usage = shutil.disk_usage(str(probe))
                return round(usage.free / (1024**3), 2)
            except OSError:
                return None
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return None


def version_blockers_for_node(node: dict[str, Any]) -> list[str]:
    updates = node.get("updates") if isinstance(node.get("updates"), dict) else {}
    if updates.get("hard_update_available") or updates.get("restart_required"):
        return ["version_drift_hard_or_restart_required"]
    return []


def severe_resource_blocked(blockers: list[str]) -> bool:
    return any(item in {"memory_load_above_floor", "peer_blocked", "peer_stale", "peer_unreachable", "peer_unverified_outbound"} for item in blockers)


def reachability_state(policy: dict[str, Any], node: dict[str, Any], network: dict[str, Any]) -> dict[str, Any]:
    """Normalize peer registry reachability into scheduler-facing state.

    Trust and private IP membership are not enough to schedule work. Remote
    work needs a fresh outbound proof from this node to that peer; inbound
    heartbeat/discovery alone is useful presence evidence, but not a usable
    execution path.
    """

    if node.get("is_local"):
        return {
            "state": "local",
            "outbound_verified": True,
            "seen_recently": True,
            "usable_for_light_tasks": True,
            "usable_for_training": True,
            "reason": "local_node",
        }
    reach = node.get("reachability") if isinstance(node.get("reachability"), dict) else {}
    discovery_state = str(node.get("discovery_state") or reach.get("state") or "").strip().lower()
    if not discovery_state:
        discovery_state = "unknown"
    failure_newer = peer_failure_is_newer(node)
    blocked = bool(node.get("blocked") or discovery_state == "blocked" or (failure_newer and int(node.get("consecutive_failures") or 0) > 0))
    flapping = bool(node.get("flapping") or discovery_state == "flapping")
    reachable = bool(node.get("reachable") or reach.get("outbound_verified") or discovery_state in {"reachable", "flapping"})
    seen_recently = bool(node.get("online") or reach.get("seen_recently") or discovery_state in {"reachable", "flapping", "discovered", "blocked"})
    outbound_age = timestamp_age_seconds(str(node.get("last_outbound_verified_utc") or ""))
    if outbound_age is None:
        outbound_age = number(node.get("outbound_age_seconds"))
    last_seen_age = timestamp_age_seconds(str(node.get("last_seen_utc") or ""))
    if last_seen_age is None:
        last_seen_age = number(node.get("age_seconds"))
    stale_after = float(get_path(policy, ["node", "stale_after_seconds"], 45) or 45)
    if outbound_age is not None and outbound_age > stale_after:
        reachable = False
        if discovery_state in {"reachable", "flapping"}:
            discovery_state = "stale"
    if last_seen_age is not None and last_seen_age > stale_after and not reachable:
        seen_recently = False
        if discovery_state in {"discovered", "unknown"}:
            discovery_state = "stale"
    state = "reachable" if reachable and not flapping else "flapping" if reachable and flapping else discovery_state
    if blocked:
        state = "blocked"
    usable_light = bool(reachable and not blocked)
    usable_training = bool(reachable and not blocked and not flapping)
    reason = "outbound_verified"
    if blocked:
        reason = "last_failure_newer_than_outbound_verification"
    elif flapping:
        reason = "recent_failure_after_outbound_success"
    elif not reachable and seen_recently:
        reason = "inbound_or_discovery_only_no_outbound_path"
    elif not reachable:
        reason = "no_recent_outbound_verification"
    return {
        "state": state,
        "outbound_verified": reachable,
        "seen_recently": seen_recently,
        "blocked": blocked,
        "flapping": flapping,
        "usable_for_light_tasks": usable_light,
        "usable_for_training": usable_training,
        "reason": reason,
        "network_class": network.get("class"),
        "last_seen_utc": node.get("last_seen_utc"),
        "last_outbound_verified_utc": node.get("last_outbound_verified_utc"),
        "last_failure_utc": node.get("last_failure_utc"),
        "last_seen_age_seconds": int(last_seen_age) if last_seen_age is not None else None,
        "outbound_age_seconds": int(outbound_age) if outbound_age is not None else None,
        "consecutive_failures": int(node.get("consecutive_failures") or 0),
    }


def reachability_blockers_for_node(node: dict[str, Any], reachability: dict[str, Any]) -> list[str]:
    if node.get("is_local"):
        return []
    state = str(reachability.get("state") or "")
    if reachability.get("usable_for_training"):
        return []
    if state == "blocked":
        return ["peer_blocked"]
    if state == "flapping":
        return ["peer_flapping"]
    if state == "stale":
        return ["peer_stale"]
    if reachability.get("seen_recently") and not reachability.get("outbound_verified"):
        return ["peer_unverified_outbound"]
    return ["peer_unreachable"]


def peer_failure_is_newer(node: dict[str, Any]) -> bool:
    failure_age = timestamp_age_seconds(str(node.get("last_failure_utc") or ""))
    outbound_age = timestamp_age_seconds(str(node.get("last_outbound_verified_utc") or ""))
    if failure_age is None:
        return False
    if outbound_age is None:
        return True
    return failure_age <= outbound_age


def timestamp_age_seconds(stamp: str) -> float | None:
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())


def normalize_capabilities(raw: list[Any]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for item in raw:
        if isinstance(item, str):
            cap = {"id": item, "score": 0.5, "detail": ""}
        elif isinstance(item, dict):
            cap = dict(item)
        else:
            continue
        cap_id = str(cap.get("id") or "")
        if not cap_id or cap_id in seen:
            continue
        seen.add(cap_id)
        out.append(cap)
    return out


def normalize_slots(raw: list[Any], capabilities: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    slots = [dict(slot) for slot in raw if isinstance(slot, dict)]
    cap_ids = {str(cap.get("id") or "") for cap in capabilities}
    if not slots and "cpu_worker" in cap_ids:
        slots.append({"slot_id": "cpu:general", "slot_type": "cpu", "capacity": 1, "running": 0, "available": True})
    if not any(str(slot.get("slot_type") or "").startswith("cuda") for slot in slots) and "nvidia_cuda" in cap_ids:
        slots.append({"slot_id": "cuda:0", "slot_type": "cuda", "capacity": 1, "running": 0, "available": True})
    if not any("mlx" in str(slot.get("slot_type") or "") for slot in slots) and ({"mlx_apple", "apple_mlx", "mlx_cuda"} & cap_ids):
        slots.append({"slot_id": "mlx:apple:0", "slot_type": "mlx_apple", "capacity": 1, "running": 0, "available": True})
    for slot in slots:
        slot["capacity"] = max(1, int(slot.get("capacity") or 1))
        slot["running"] = max(0, int(slot.get("running") or 0))
        slot["available"] = bool(slot.get("available", slot["running"] < slot["capacity"]))
        slot["task_kinds"] = merged_slot_task_kinds(policy, str(slot.get("slot_type") or ""), slot.get("task_kinds"))
    return slots


def merged_slot_task_kinds(policy: dict[str, Any], slot_type: str, existing: Any) -> list[str]:
    out = [str(item) for item in existing if item] if isinstance(existing, list) else []
    key = slot_policy_key(slot_type)
    policy_kinds = get_path(policy, ["resource_slots", "task_kinds_by_slot", key], [])
    if isinstance(policy_kinds, list):
        for item in policy_kinds:
            value = str(item or "")
            if value and value not in out:
                out.append(value)
    return out


def slot_policy_key(slot_type: str) -> str:
    if slot_type in {"mlx", "mlx_apple", "mlx_cuda"} or "mlx" in slot_type:
        return "mlx"
    if slot_type.startswith("cuda"):
        return "cuda"
    return "cpu"


def has_training_slot(slots: list[dict[str, Any]]) -> bool:
    return any(str(slot.get("slot_type") or "") in {"cuda", "mlx", "mlx_apple", "mlx_cuda"} and slot.get("available") for slot in slots)


def best_for_capability(nodes: list[dict[str, Any]], capability_order: list[str], strict: bool = False) -> dict[str, Any]:
    for capability in capability_order:
        matching = [node for node in nodes if capability in set(node.get("capability_ids") or [])]
        if matching:
            return max(matching, key=node_score)
    if strict:
        return {}
    return max(nodes, key=node_score) if nodes else {}


def node_score(node: dict[str, Any]) -> float:
    score = sum(float(cap.get("score") or 0.0) for cap in node.get("capabilities") or [])
    free = number(get_path(node, ["resources", "disk", "free_gib"], None))
    if free is not None:
        score += min(0.2, free / 1000.0)
    if node.get("is_local"):
        score += 0.1
    latency = float(get_path(node, ["network", "estimated_latency_ms"], 0) or 0)
    score -= min(0.25, latency / 1000.0)
    return score


def peer_from_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": status.get("node_id"),
        "node_name": status.get("node_name"),
        "hostname": status.get("hostname"),
        "api_url": status.get("api_url"),
        "dashboard_url": status.get("dashboard_url"),
        "hive_id": status.get("hive_id"),
        "federation_tier": status.get("federation_tier"),
        "platform": status.get("platform") or {},
        "capabilities": status.get("capabilities") or [],
        "resources": status.get("resources") or {},
        "slots": status.get("slots") or [],
        "runtime_paths": status.get("runtime_paths") or {},
        "updates": status.get("updates") or {},
        "storage": status.get("storage") or {},
        "remote_control": status.get("remote_control") or {},
    }


def node_network_profile(node: dict[str, Any]) -> dict[str, Any]:
    parsed = urlparse(str(node.get("api_url") or ""))
    host = parsed.hostname or ""
    scope = host_scope(host)
    if scope == "loopback":
        return {"class": "local", "host_scope": scope, "estimated_latency_ms": 0, "task_fit": "interactive_and_training"}
    if scope in {"private_ip", "private_dns"}:
        return {"class": "lan_or_private_tunnel", "host_scope": scope, "estimated_latency_ms": 8, "task_fit": "interactive_and_training"}
    if scope in {"public_ip", "public_dns", "unknown_dns"}:
        return {"class": "wan", "host_scope": scope, "estimated_latency_ms": 80, "task_fit": "bounded_async_chunks"}
    return {"class": "relay_or_unknown", "host_scope": scope, "estimated_latency_ms": 120, "task_fit": "bounded_async_chunks"}


def host_scope(host: str) -> str:
    if not host:
        return "unknown"
    lowered = host.lower()
    if lowered in {"localhost", "ip6-localhost"}:
        return "loopback"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if lowered.endswith((".local", ".lan", ".home")):
            return "private_dns"
        return "public_dns" if "." in lowered else "unknown_dns"
    if ip.is_loopback:
        return "loopback"
    if ip.is_private or ip.is_link_local:
        return "private_ip"
    return "public_ip"


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
        if isinstance(profile, dict) and (not active or profile.get("profile_id") == active) and profile.get("join_token"):
            return str(profile.get("join_token") or "")
    return ""


def hive_id(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["federation", "hive_id_env"], "THESEUS_HIVE_ID"))
    if os.environ.get(env_name):
        return str(os.environ.get(env_name) or "")
    join = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    return str(join.get("hive_id") or get_path(policy, ["federation", "default_hive_id"], "local"))


def secret_headers(policy: dict[str, Any]) -> dict[str, str]:
    secret = hive_secret(policy)
    return {"X-Theseus-Hive-Secret": secret} if secret else {}


def fetch_json(url: str, *, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    try:
        req = urlrequest.Request(url, headers=headers)
        with urlrequest.urlopen(req, timeout=timeout) as response:  # noqa: S310 - private/local Hive endpoints only.
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc), "url": url}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_json", "url": url}


def platform_summary() -> dict[str, str]:
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version()}


def number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
