"""Operator target and accelerator summaries for Hive nodes."""

from __future__ import annotations

from typing import Any

from hive_node_common import get_path
from hive_node_peer_registry import peer_from_status


def operator_targets(policy: dict[str, Any], status: dict[str, Any], peers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del policy
    local = peer_from_status(status)
    local["is_local"] = True
    rows = [local]
    for peer in peers:
        row = dict(peer)
        row["is_local"] = False
        rows.append(row)
    return [
        {
            "node_id": row.get("node_id"),
            "node_name": row.get("node_name"),
            "api_url": row.get("api_url"),
            "dashboard_url": row.get("dashboard_url"),
            "is_local": row.get("is_local", False),
            "discovery_state": row.get("discovery_state", "reachable" if row.get("is_local") else ""),
            "reachable": row.get("reachable", True if row.get("is_local") else None),
            "blocked": row.get("blocked", False),
            "reachability": row.get("reachability") if isinstance(row.get("reachability"), dict) else {},
            "platform": row.get("platform", {}),
            "capability_ids": node_capability_ids(row),
            "accelerator_ids": node_accelerator_ids(row),
            "mlx_ready": node_has_apple_mlx(row),
            "storage_share_count": get_path(row, ["storage", "share_count"], 0),
            "remote_control_ready_provider_count": get_path(row, ["remote_control", "ready_provider_count"], 0),
            "remote_control_preferred_provider_id": get_path(row, ["remote_control", "preferred_provider_id"], ""),
            "voice_room_name": get_path(row, ["voice_following", "room_name"], ""),
            "voice_ready_to_listen": get_path(row, ["voice_following", "ready_to_listen"], False),
            "voice_ready_to_respond": get_path(row, ["voice_following", "ready_to_respond"], False),
            "task_kinds": sorted({str(kind) for slot in row.get("slots") or [] for kind in (slot.get("task_kinds") or []) if kind}),
        }
        for row in rows
    ]


def node_capability_ids(node: dict[str, Any]) -> list[str]:
    return [str(cap.get("id") or "") for cap in node.get("capabilities") or [] if isinstance(cap, dict)]


def node_has_capability(node: dict[str, Any], capability: str) -> bool:
    return capability in set(node_capability_ids(node))


def node_accelerator_ids(node: dict[str, Any]) -> list[str]:
    caps = set(node_capability_ids(node))
    mlx = get_path(node, ["resources", "mlx"], {})
    backend_ids = {str(item) for item in mlx.get("backend_ids") or []} if isinstance(mlx, dict) else set()
    accelerators = sorted((caps | backend_ids) & {"mlx_apple", "apple_mlx", "mlx_cuda", "nvidia_cuda"})
    return accelerators


def node_has_apple_mlx(node: dict[str, Any]) -> bool:
    accelerators = set(node_accelerator_ids(node))
    mlx = get_path(node, ["resources", "mlx"], {})
    return (
        "mlx_apple" in accelerators
        or "apple_mlx" in accelerators
        or (isinstance(mlx, dict) and bool(mlx.get("available")) and bool(mlx.get("platform_is_macos")))
    )


def node_has_nvidia_cuda(node: dict[str, Any]) -> bool:
    accelerators = set(node_accelerator_ids(node))
    nvidia = get_path(node, ["resources", "nvidia"], {})
    return "nvidia_cuda" in accelerators or (isinstance(nvidia, dict) and bool(nvidia.get("available")))


def node_can_run_task(node: dict[str, Any], kind: str) -> bool:
    slots = node.get("slots") if isinstance(node.get("slots"), list) else []
    for slot in slots:
        task_kinds = set(str(item) for item in (slot.get("task_kinds") or []))
        if kind in task_kinds:
            return True
    if kind.startswith("cuda_"):
        return node_has_nvidia_cuda(node)
    if kind.startswith("mlx_"):
        return node_has_apple_mlx(node) or "mlx_cuda" in set(node_accelerator_ids(node))
    if kind == "checkpoint_chat":
        return node_has_capability(node, "checkpoint_chat_gateway") or any("checkpoint_chat" in (slot.get("task_kinds") or []) for slot in slots)
    return any(kind in (slot.get("task_kinds") or []) for slot in slots)


def operator_accelerator_summary(
    policy: dict[str, Any],
    status: dict[str, Any],
    peers: list[dict[str, Any]],
    allowed_task_kinds: list[str] | None = None,
) -> dict[str, Any]:
    local = peer_from_status(status)
    local["is_local"] = True
    nodes = [local] + [dict(peer, is_local=False) for peer in peers]
    mlx_nodes = [node for node in nodes if node_has_apple_mlx(node)]
    cuda_nodes = [node for node in nodes if node_has_nvidia_cuda(node)]
    allowed = set(allowed_task_kinds or [])
    mlx_task_kinds = [kind for kind in ["mlx_eval_chunk", "mlx_training_chunk", "mlx_rollout_chunk"] if kind in allowed]
    cuda_task_kinds = [kind for kind in ["cuda_eval_chunk", "cuda_training_chunk", "cuda_rollout_chunk"] if kind in allowed]
    best_mlx = mlx_nodes[0] if mlx_nodes else {}
    best_cuda = cuda_nodes[0] if cuda_nodes else {}
    local_mlx = node_has_apple_mlx(local)
    local_cuda = node_has_nvidia_cuda(local)
    mlx = get_path(local, ["resources", "mlx"], {})
    return {
        "apple_mlx": {
            "available": bool(mlx_nodes),
            "local_ready": local_mlx,
            "node_count": len(mlx_nodes),
            "best_node_id": best_mlx.get("node_id"),
            "best_node_name": best_mlx.get("node_name"),
            "module": mlx.get("module") if isinstance(mlx, dict) else "mlx.core",
            "backend_ids": sorted({item for node in mlx_nodes for item in node_accelerator_ids(node) if "mlx" in item or item == "apple_mlx"}),
            "task_kinds": mlx_task_kinds,
            "queue_target": "local" if local_mlx else ("auto" if mlx_nodes else ""),
            "recommended": bool(mlx_nodes and mlx_task_kinds),
        },
        "nvidia_cuda": {
            "available": bool(cuda_nodes),
            "local_ready": local_cuda,
            "node_count": len(cuda_nodes),
            "best_node_id": best_cuda.get("node_id"),
            "best_node_name": best_cuda.get("node_name"),
            "backend_ids": sorted({item for node in cuda_nodes for item in node_accelerator_ids(node) if "cuda" in item or item == "nvidia_cuda"}),
            "task_kinds": cuda_task_kinds,
            "queue_target": "local" if local_cuda else ("auto" if cuda_nodes else ""),
            "recommended": bool(cuda_nodes and cuda_task_kinds),
        },
        "policy": {
            "external_inference": get_path(policy, ["worker_chunks", "external_inference"], "forbidden"),
            "teacher_use": get_path(policy, ["worker_chunks", "teacher_use"], "forbidden"),
            "network_access": get_path(policy, ["worker_chunks", "network_access"], "none"),
        },
    }
