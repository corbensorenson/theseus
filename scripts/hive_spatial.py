"""Privacy-preserving spatial Hive operator scene contract.

The spatial layer is an operator surface, not a camera/audio ingest pipeline.
It aggregates explicit room placement, node capability summaries, voice route,
storage share summaries, remote-control handoff readiness, and work state into
a compact scene that native visionOS, Quest/OpenXR, iPhone, and web clients can
render without receiving raw room scans, passthrough frames, microphone audio,
or arbitrary filesystem access.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hive_remote_control
import hive_storage
import hive_users
import hive_voice_following


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
POLICY_PATH = CONFIGS / "hive_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Project Theseus Hive spatial operator scene contract.")
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Write the privacy-preserving spatial scene contract.")
    status.add_argument("--out", default="reports/hive_spatial_status.json")

    configure = sub.add_parser("configure-node", help="Set this node's room/zone/position for spatial operator clients.")
    configure.add_argument("--room-id", default="")
    configure.add_argument("--room-name", default="")
    configure.add_argument("--zone", default="")
    configure.add_argument("--x", type=float, default=None)
    configure.add_argument("--y", type=float, default=None)
    configure.add_argument("--z", type=float, default=None)
    configure.add_argument("--yaw", type=float, default=None)
    display = configure.add_mutually_exclusive_group()
    display.add_argument("--display", dest="display", action="store_true")
    display.add_argument("--no-display", dest="display", action="store_false")
    configure.set_defaults(display=None)
    configure.add_argument("--surface", action="append", default=[], help="Operator surface tag, e.g. macos_menu_bar, visionos, quest.")
    configure.add_argument("--device", action="append", default=[], help="Nearby explicit device tag for spatial summaries.")
    configure.add_argument("--out", default="reports/hive_spatial_status.json")

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command in {None, "status"}:
        report = status_report(policy=policy, write_report=True)
        write_json(ROOT / args.out, report)
    elif args.command == "configure-node":
        report = configure_node(
            policy=policy,
            room_id=args.room_id,
            room_name=args.room_name,
            zone=args.zone,
            x=args.x,
            y=args.y,
            z=args.z,
            yaw_degrees=args.yaw,
            display=args.display,
            surfaces=args.surface,
            nearby_devices=args.device,
        )
        if args.out:
            write_json(ROOT / args.out, status_report(policy=policy, write_report=True))
    else:
        parser.print_help()
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def status_report(
    *,
    policy: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
    peers: list[dict[str, Any]] | None = None,
    auth_context: dict[str, Any] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    peers = [peer for peer in peers or [] if isinstance(peer, dict)]
    config = spatial_config(policy)
    local_voice = hive_voice_following.status_report(policy=policy, write_report=False)
    local_storage = hive_storage.status_report(policy=policy, write_report=False)
    local_remote = hive_remote_control.status_report(policy=policy, write_report=False)
    local_status = local_status_for_scene(policy, status, local_voice, local_storage, local_remote)
    route = hive_voice_following.route_decision(policy=policy, status=local_status, peers=peers, write_report=write_report)
    include_storage = hive_users.action_allowed(auth_context or owner_context(), "storage")
    include_remote = hive_users.action_allowed(auth_context or owner_context(), "remote_control")
    local_node = scene_node_from_status(
        policy=policy,
        config=config,
        node=local_status,
        is_local=True,
        local_storage=local_storage,
        local_remote=local_remote,
        include_storage=include_storage,
        include_remote=include_remote,
    )
    nodes = [local_node]
    for peer in peers:
        nodes.append(
            scene_node_from_status(
                policy=policy,
                config=config,
                node=peer,
                is_local=False,
                local_storage=None,
                local_remote=None,
                include_storage=include_storage,
                include_remote=include_remote,
            )
        )
    nodes = dedupe_nodes(nodes)
    rooms = rooms_from_nodes(nodes)
    report = {
        "ok": True,
        "policy": "project_theseus_hive_spatial_status_v0",
        "created_utc": now(),
        "enabled": bool(get_path(policy, ["spatial", "enabled"], True)) and bool(config.get("enabled", True)),
        "hive_id": local_status.get("hive_id") or "",
        "scene": {
            "default_units": get_path(policy, ["spatial", "default_units"], "meters"),
            "coordinate_space": get_path(policy, ["spatial", "coordinate_space"], "local_room_manual"),
            "map_raw_capture_retained": False,
            "client_role": "operator_surface",
            "recommended_render": "rooms_as_zones_nodes_as_capability_anchors",
        },
        "rooms": rooms,
        "nodes": nodes,
        "node_count": len(nodes),
        "room_count": len(rooms),
        "active_voice_route": public_route(route),
        "nearby_storage": nearby_storage(nodes),
        "nearby_devices": nearby_devices(nodes, config),
        "work": work_scene(nodes),
        "operator_contract": operator_contract(),
        "security": privacy_summary(policy),
        "next_actions": next_actions(config, local_node, route),
    }
    if write_report:
        write_json(ROOT / str(get_path(policy, ["spatial", "status_path"], "reports/hive_spatial_status.json")), report)
    return report


def configure_node(
    *,
    policy: dict[str, Any] | None = None,
    room_id: str = "",
    room_name: str = "",
    zone: str = "",
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    yaw_degrees: float | None = None,
    display: bool | None = None,
    surfaces: list[str] | None = None,
    nearby_devices: list[str] | None = None,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    config = spatial_config(policy)
    room = config.get("room") if isinstance(config.get("room"), dict) else {}
    if room_id:
        room["room_id"] = safe_id(room_id)
    if room_name:
        room["name"] = safe_label(room_name, 80)
    if zone:
        room["zone"] = safe_label(zone, 80)
    config["room"] = room

    position = config.get("position") if isinstance(config.get("position"), dict) else {}
    if any(value is not None for value in [x, y, z, yaw_degrees]):
        if x is not None:
            position["x"] = round(float(x), 4)
        if y is not None:
            position["y"] = round(float(y), 4)
        if z is not None:
            position["z"] = round(float(z), 4)
        if yaw_degrees is not None:
            position["yaw_degrees"] = round(float(yaw_degrees), 4)
        position["configured"] = True
        position.setdefault("coordinate_space", get_path(policy, ["spatial", "coordinate_space"], "local_room_manual"))
    config["position"] = position

    capabilities = config.get("capabilities") if isinstance(config.get("capabilities"), dict) else {}
    if display is not None:
        capabilities["display"] = bool(display)
    if surfaces:
        capabilities["operator_surfaces"] = unique_clean([*(capabilities.get("operator_surfaces") or []), *surfaces])
    config["capabilities"] = capabilities
    if nearby_devices:
        config["nearby_devices"] = unique_clean([*(config.get("nearby_devices") or []), *nearby_devices])
    write_spatial_config(policy, config)
    return {
        "ok": True,
        "policy": "project_theseus_hive_spatial_configure_v0",
        "created_utc": now(),
        "config_path": rel(spatial_config_path(policy)),
        "config": normalized_spatial_config(config),
        "next_command": "theseus spatial status",
    }


def operator_summary(
    policy: dict[str, Any],
    status: dict[str, Any],
    peers: list[dict[str, Any]],
    auth_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return status_report(policy=policy, status=status, peers=peers, auth_context=auth_context, write_report=True)


def compact_status(report: dict[str, Any]) -> dict[str, Any]:
    route = report.get("active_voice_route") if isinstance(report.get("active_voice_route"), dict) else {}
    nodes = report.get("nodes") if isinstance(report.get("nodes"), list) else []
    local = next((node for node in nodes if isinstance(node, dict) and node.get("is_local")), {})
    return {
        "enabled": bool(report.get("enabled")),
        "room_count": int(report.get("room_count") or 0),
        "node_count": int(report.get("node_count") or 0),
        "local_room_id": get_path(local, ["room", "room_id"], ""),
        "local_room_name": get_path(local, ["room", "name"], ""),
        "active_room_name": route.get("active_room_name") or "",
        "voice_route_state": route.get("state") or "unknown",
        "storage_anchor_count": len(report.get("nearby_storage") or []),
        "privacy": get_path(report, ["security", "raw_spatial_data_policy"], "summaries_only"),
    }


def local_status_for_scene(
    policy: dict[str, Any],
    status: dict[str, Any] | None,
    voice_status: dict[str, Any],
    storage_status: dict[str, Any],
    remote_status: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(status, dict) and status.get("node_id"):
        local = dict(status)
    else:
        identity = local_identity(policy)
        local = {
            "node_id": identity.get("node_id"),
            "node_name": identity.get("node_name"),
            "hostname": socket.gethostname(),
            "api_url": "",
            "hive_id": active_hive_id(policy),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
        }
    local.setdefault("voice_following", hive_voice_following.compact_status(voice_status))
    local.setdefault("storage", compact_storage(storage_status))
    local.setdefault("remote_control", compact_remote(remote_status))
    return local


def scene_node_from_status(
    *,
    policy: dict[str, Any],
    config: dict[str, Any],
    node: dict[str, Any],
    is_local: bool,
    local_storage: dict[str, Any] | None,
    local_remote: dict[str, Any] | None,
    include_storage: bool,
    include_remote: bool,
) -> dict[str, Any]:
    spatial = node.get("spatial") if isinstance(node.get("spatial"), dict) else {}
    room = room_for_node(config, node, spatial, is_local=is_local)
    voice = node.get("voice_following") if isinstance(node.get("voice_following"), dict) else {}
    storage = local_storage if is_local and local_storage else node.get("storage") if isinstance(node.get("storage"), dict) else {}
    remote = local_remote if is_local and local_remote else node.get("remote_control") if isinstance(node.get("remote_control"), dict) else {}
    capabilities = node_capabilities(policy, config, node, voice, storage, remote, is_local=is_local)
    return {
        "node_id": node.get("node_id") or get_path(spatial, ["node", "node_id"], ""),
        "node_name": node.get("node_name") or get_path(spatial, ["node", "node_name"], ""),
        "hostname": node.get("hostname") or "",
        "api_url": node.get("api_url") or "",
        "is_local": bool(is_local),
        "reachable": bool(is_local or node.get("reachable") or node.get("discovery_state") == "reachable"),
        "discovery_state": "local" if is_local else str(node.get("discovery_state") or ("reachable" if node.get("reachable") else "discovered")),
        "platform": compact_platform(node.get("platform") if isinstance(node.get("platform"), dict) else {}),
        "room": room,
        "position": position_for_node(config, spatial, is_local=is_local),
        "capabilities": capabilities,
        "voice": compact_voice(voice),
        "storage": public_storage(storage, include_storage=include_storage),
        "remote_control": public_remote(remote, include_remote=include_remote),
        "work": node_work(node),
        "links": node_links(node),
    }


def room_for_node(config: dict[str, Any], node: dict[str, Any], spatial: dict[str, Any], *, is_local: bool) -> dict[str, Any]:
    spatial_room = spatial.get("room") if isinstance(spatial.get("room"), dict) else {}
    local_room = config.get("room") if is_local and isinstance(config.get("room"), dict) else {}
    voice = node.get("voice_following") if isinstance(node.get("voice_following"), dict) else {}
    room_id = str(local_room.get("room_id") or spatial_room.get("room_id") or voice.get("room_id") or "unassigned")
    name = str(local_room.get("name") or spatial_room.get("name") or voice.get("room_name") or node.get("node_name") or socket.gethostname())
    zone = str(local_room.get("zone") or spatial_room.get("zone") or "default")
    return {
        "room_id": safe_id(room_id),
        "name": safe_label(name, 80),
        "zone": safe_label(zone, 80),
    }


def position_for_node(config: dict[str, Any], spatial: dict[str, Any], *, is_local: bool) -> dict[str, Any]:
    source = config.get("position") if is_local and isinstance(config.get("position"), dict) else {}
    if not source and isinstance(spatial.get("position"), dict):
        source = spatial.get("position") or {}
    configured = bool(source.get("configured"))
    return {
        "configured": configured,
        "x": float(source.get("x") or 0.0) if configured else None,
        "y": float(source.get("y") or 0.0) if configured else None,
        "z": float(source.get("z") or 0.0) if configured else None,
        "yaw_degrees": float(source.get("yaw_degrees") or 0.0) if configured else None,
        "coordinate_space": str(source.get("coordinate_space") or "local_room_manual"),
    }


def node_capabilities(
    policy: dict[str, Any],
    config: dict[str, Any],
    node: dict[str, Any],
    voice: dict[str, Any],
    storage: dict[str, Any],
    remote: dict[str, Any],
    *,
    is_local: bool,
) -> dict[str, Any]:
    local_caps = config.get("capabilities") if is_local and isinstance(config.get("capabilities"), dict) else {}
    ids = capability_ids(node)
    accelerators = accelerator_ids(node, ids)
    display_default = platform.system() in {"Darwin", "Windows"} or bool(os.environ.get("DISPLAY"))
    display = bool(local_caps.get("display", display_default)) if is_local else bool(get_path(node, ["spatial", "capabilities", "display"], False))
    surfaces = unique_clean([str(item) for item in local_caps.get("operator_surfaces") or []]) if is_local else unique_clean(get_path(node, ["spatial", "capabilities", "operator_surfaces"], []))
    return {
        "microphone": bool(voice.get("microphone_available") or voice.get("ready_to_listen")),
        "speaker": bool(voice.get("speaker_available") or voice.get("ready_to_respond")),
        "display": display,
        "spatial_operator": bool(surfaces),
        "operator_surfaces": surfaces,
        "storage": int(storage.get("share_count") or 0) > 0,
        "remote_control": int(remote.get("ready_provider_count") or 0) > 0,
        "accelerators": accelerators,
        "can_train_mlx": any(item in accelerators for item in ["mlx_apple", "apple_mlx"]),
        "can_train_cuda": any("cuda" in item for item in accelerators),
        "operator_only_spatial_client": bool(get_path(policy, ["spatial", "operator_only"], True)),
    }


def rooms_from_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rooms: dict[str, dict[str, Any]] = {}
    for node in nodes:
        room = node.get("room") if isinstance(node.get("room"), dict) else {}
        room_id = str(room.get("room_id") or "unassigned")
        current = rooms.setdefault(
            room_id,
            {
                "room_id": room_id,
                "name": room.get("name") or room_id,
                "zone": room.get("zone") or "default",
                "node_count": 0,
                "speaker_count": 0,
                "microphone_count": 0,
                "storage_anchor_count": 0,
                "work_anchor_count": 0,
            },
        )
        current["node_count"] += 1
        if get_path(node, ["capabilities", "speaker"], False):
            current["speaker_count"] += 1
        if get_path(node, ["capabilities", "microphone"], False):
            current["microphone_count"] += 1
        if get_path(node, ["capabilities", "storage"], False):
            current["storage_anchor_count"] += 1
        if get_path(node, ["work", "active"], False):
            current["work_anchor_count"] += 1
    return sorted(rooms.values(), key=lambda item: (str(item.get("zone")), str(item.get("name"))))


def nearby_storage(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in nodes:
        storage = node.get("storage") if isinstance(node.get("storage"), dict) else {}
        for share in storage.get("shares") or []:
            if not isinstance(share, dict):
                continue
            rows.append(
                {
                    "node_id": node.get("node_id"),
                    "node_name": node.get("node_name"),
                    "room_id": get_path(node, ["room", "room_id"], ""),
                    "room_name": get_path(node, ["room", "name"], ""),
                    "share_id": share.get("share_id"),
                    "name": share.get("name"),
                    "kind": share.get("kind"),
                    "tags": share.get("tags") or [],
                    "accessible": share.get("accessible"),
                }
            )
    return rows[:48]


def nearby_devices(nodes: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in config.get("nearby_devices") or []:
        rows.append({"kind": "manual", "label": safe_label(str(label), 100), "source": "local_spatial_config"})
    for node in nodes:
        for device in get_path(node, ["storage", "device_extensions"], []):
            if isinstance(device, dict):
                rows.append(
                    {
                        "kind": safe_label(str(device.get("kind") or "device"), 40),
                        "label": safe_label(str(device.get("name") or device.get("path") or "device"), 100),
                        "node_id": node.get("node_id"),
                        "room_id": get_path(node, ["room", "room_id"], ""),
                        "source": "hive_storage_extension",
                    }
                )
    return rows[:64]


def work_scene(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    active = [node for node in nodes if get_path(node, ["work", "active"], False)]
    return {
        "active_node_count": len(active),
        "training_node_count": sum(1 for node in nodes if get_path(node, ["work", "training_capable"], False)),
        "active_nodes": [
            {
                "node_id": node.get("node_id"),
                "node_name": node.get("node_name"),
                "room_id": get_path(node, ["room", "room_id"], ""),
                "summary": get_path(node, ["work", "summary"], ""),
            }
            for node in active[:24]
        ],
    }


def public_route(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": route.get("policy") or "project_theseus_hive_voice_route_v0",
        "created_utc": route.get("created_utc") or "",
        "state": route.get("state") or "unknown",
        "active_room_id": route.get("active_room_id") or "",
        "active_room_name": route.get("active_room_name") or "",
        "confidence": route.get("confidence") or 0.0,
        "listen_node": route.get("listen_node") if isinstance(route.get("listen_node"), dict) else {},
        "respond_node": route.get("respond_node") if isinstance(route.get("respond_node"), dict) else {},
        "security": route.get("security") if isinstance(route.get("security"), dict) else {},
    }


def operator_contract() -> dict[str, Any]:
    return {
        "policy": "project_theseus_hive_spatial_operator_contract_v0",
        "native_clients": {
            "visionos": "SwiftUI_RealityKit_client_should_import_operator_roaming_profile_then_render_spatial_status",
            "quest": "Unity_or_native_OpenXR_client_should_call_same_authenticated_JSON_contract",
            "iphone_watch": "profile_token_flow_shared_with_existing_native_shells",
        },
        "endpoints": {
            "spatial_status": "GET /api/hive/spatial/status",
            "operator_status": "GET /api/hive/operator/status",
            "voice_route": "GET /api/hive/voice/route",
            "storage_browse": "GET /api/hive/storage/peer/browse",
            "remote_control_session": "POST /api/hive/remote-control/session",
            "chat": "POST /api/hive/operator/chat",
            "task": "POST /api/hive/operator/task",
            "roaming_profile": "GET /api/hive/operator/roaming-profile",
        },
        "rendering_guidance": [
            "Render rooms/zones from summarized coordinates only.",
            "Render nodes as capability anchors with voice/storage/work badges.",
            "Use storage endpoints for explicit file previews, not arbitrary filesystem browsing.",
            "Use remote-control handoff endpoints only to launch governed sessions.",
        ],
    }


def privacy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_spatial_data_policy": get_path(policy, ["spatial", "raw_spatial_data_policy"], "never_relay_camera_passthrough_or_room_mesh_by_default"),
        "raw_camera_passthrough_retained": False,
        "room_mesh_retained": False,
        "raw_audio_retained": False,
        "transcripts_required_for_routing": False,
        "summary_fields_only": ["room", "zone", "capability", "presence_score", "work_state", "storage_share_summary"],
        "arbitrary_filesystem": False,
        "arbitrary_remote_shell": False,
        "spatial_devices_as_training_workers": False,
        "operator_role": "high_bandwidth_control_surface_first",
    }


def next_actions(config: dict[str, Any], local_node: dict[str, Any], route: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    room_id = get_path(local_node, ["room", "room_id"], "unassigned")
    if room_id == "unassigned":
        actions.append("Set this node's room: theseus spatial configure-node --room-id kitchen --room-name Kitchen")
    if not get_path(local_node, ["position", "configured"], False):
        actions.append("Optionally place this node in room coordinates: theseus spatial configure-node --x 0 --y 0 --z 0 --yaw 0")
    if not get_path(local_node, ["capabilities", "microphone"], False) or not get_path(local_node, ["capabilities", "speaker"], False):
        actions.append("Configure voice-following mic/speaker role with: theseus voice configure-room --microphone --speaker")
    if not get_path(local_node, ["capabilities", "storage"], False):
        actions.append("Add explicit storage anchors with: theseus storage add-share --path /Volumes/NAS/Photos --name Photos --tag photos")
    if route.get("state") != "active":
        actions.append("Smoke voice route with: theseus voice presence --score 0.8 --source smoke")
    if not config.get("nearby_devices"):
        actions.append("Add manual spatial device tags with: theseus spatial configure-node --device nas:photos --device rpi:kitchen-sensor")
    return actions[:6]


def spatial_config(policy: dict[str, Any]) -> dict[str, Any]:
    path = spatial_config_path(policy)
    if path.exists():
        value = read_json(path, {})
        if isinstance(value, dict):
            return normalized_spatial_config(value)
    return normalized_spatial_config({})


def normalized_spatial_config(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("policy", "project_theseus_hive_spatial_local_v0")
    config.setdefault("enabled", True)
    room = config.get("room") if isinstance(config.get("room"), dict) else {}
    room.setdefault("room_id", "unassigned")
    room.setdefault("name", socket.gethostname())
    room.setdefault("zone", "default")
    config["room"] = room
    position = config.get("position") if isinstance(config.get("position"), dict) else {}
    position.setdefault("configured", False)
    position.setdefault("coordinate_space", "local_room_manual")
    config["position"] = position
    capabilities = config.get("capabilities") if isinstance(config.get("capabilities"), dict) else {}
    capabilities.setdefault("display", platform.system() in {"Darwin", "Windows"} or bool(os.environ.get("DISPLAY")))
    capabilities.setdefault("operator_surfaces", [])
    config["capabilities"] = capabilities
    config.setdefault("nearby_devices", [])
    return config


def write_spatial_config(policy: dict[str, Any], config: dict[str, Any]) -> None:
    config["updated_utc"] = now()
    write_json(spatial_config_path(policy), normalized_spatial_config(config))


def spatial_config_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["spatial", "config_path"], "configs/hive_spatial.local.json"))


def public_storage(storage: dict[str, Any], *, include_storage: bool) -> dict[str, Any]:
    if not include_storage:
        return {"enabled": bool(storage.get("enabled", True)), "access_denied": True}
    shares = storage.get("shares") if isinstance(storage.get("shares"), list) else []
    return {
        "enabled": bool(storage.get("enabled", True)),
        "share_count": int(storage.get("share_count") or 0),
        "shares": [
            {
                "share_id": share.get("share_id"),
                "name": share.get("name"),
                "kind": share.get("kind"),
                "tags": share.get("tags") or [],
                "accessible": share.get("accessible"),
            }
            for share in shares[:24]
            if isinstance(share, dict) and share.get("enabled", True)
        ],
        "device_extensions": storage.get("device_extensions") if isinstance(storage.get("device_extensions"), list) else [],
        "security": {
            "summaries_only": True,
            "file_listing_requires_storage_endpoint": True,
        },
    }


def public_remote(remote: dict[str, Any], *, include_remote: bool) -> dict[str, Any]:
    if not include_remote:
        return {"enabled": bool(remote.get("enabled", True)), "access_denied": True}
    providers = remote.get("providers") if isinstance(remote.get("providers"), list) else []
    return {
        "enabled": bool(remote.get("enabled", True)),
        "ready_provider_count": int(remote.get("ready_provider_count") or 0),
        "preferred_provider_id": remote.get("preferred_provider_id") or "",
        "providers": [
            {
                "id": provider.get("id"),
                "ready": bool(provider.get("ready")),
                "launches_external_client": bool(provider.get("launches_external_client", True)),
            }
            for provider in providers[:12]
            if isinstance(provider, dict)
        ],
    }


def compact_storage(storage: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(storage.get("enabled", True)),
        "share_count": int(storage.get("share_count") or 0),
        "shares": storage.get("shares") if isinstance(storage.get("shares"), list) else [],
        "device_extensions": storage.get("device_extensions") if isinstance(storage.get("device_extensions"), list) else [],
    }


def compact_remote(remote: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(remote.get("enabled", True)),
        "ready_provider_count": int(remote.get("ready_provider_count") or 0),
        "preferred_provider_id": remote.get("preferred_provider_id") or "",
        "providers": remote.get("providers") if isinstance(remote.get("providers"), list) else [],
    }


def compact_voice(voice: dict[str, Any]) -> dict[str, Any]:
    return {
        "room_id": voice.get("room_id") or "",
        "room_name": voice.get("room_name") or "",
        "ready_to_listen": bool(voice.get("ready_to_listen")),
        "ready_to_respond": bool(voice.get("ready_to_respond")),
        "microphone_available": bool(voice.get("microphone_available")),
        "speaker_available": bool(voice.get("speaker_available")),
        "presence": voice.get("presence") if isinstance(voice.get("presence"), dict) else {},
        "privacy": voice.get("privacy") or "never_relay_raw_audio_by_default",
    }


def node_work(node: dict[str, Any]) -> dict[str, Any]:
    slots = node.get("slots") if isinstance(node.get("slots"), list) else []
    busy_slots = [slot for slot in slots if isinstance(slot, dict) and slot.get("status") in {"busy", "running"}]
    idle_slots = [slot for slot in slots if isinstance(slot, dict) and slot.get("status") in {"idle", "available"}]
    task_kinds = sorted({str(kind) for slot in slots if isinstance(slot, dict) for kind in slot.get("task_kinds", [])})
    utilization = node.get("utilization") if isinstance(node.get("utilization"), dict) else {}
    active = bool(busy_slots or utilization.get("trigger_state") in {"planned", "executed", "active"})
    training_capable = any(kind in task_kinds for kind in ["mlx_training_chunk", "mlx_rollout_chunk", "cuda_training_chunk", "cuda_rollout_chunk", "training_smoke"])
    return {
        "active": active,
        "summary": utilization.get("trigger_state") or ("busy" if busy_slots else "idle" if idle_slots else "unknown"),
        "idle_slots": len(idle_slots),
        "busy_slots": len(busy_slots),
        "task_kinds": task_kinds[:24],
        "training_capable": training_capable,
        "hive_version": get_path(node, ["hive_version", "local_version_id"], ""),
    }


def node_links(node: dict[str, Any]) -> dict[str, Any]:
    api_url = str(node.get("api_url") or "")
    return {
        "api_url": api_url,
        "mobile_url": api_url.rstrip("/") + "/mobile" if api_url else "",
        "storage_browse_endpoint": "/api/hive/storage/peer/browse",
        "remote_control_endpoint": "/api/hive/remote-control/session",
    }


def capability_ids(node: dict[str, Any]) -> list[str]:
    raw = node.get("capabilities")
    values: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                values.append(str(item.get("id") or ""))
            else:
                values.append(str(item or ""))
    return unique_clean(values)


def accelerator_ids(node: dict[str, Any], ids: list[str]) -> list[str]:
    accelerators = [item for item in ids if any(token in item for token in ["mlx", "cuda", "gpu", "metal"])]
    resources = node.get("resources") if isinstance(node.get("resources"), dict) else {}
    for item in resources.get("accelerators") or []:
        if isinstance(item, dict):
            accelerators.append(str(item.get("id") or item.get("kind") or item.get("name") or ""))
        else:
            accelerators.append(str(item or ""))
    return unique_clean(accelerators)


def compact_platform(platform_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": platform_report.get("system") or platform.system(),
        "machine": platform_report.get("machine") or "",
        "release": platform_report.get("release") or "",
    }


def dedupe_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("node_id") or node.get("api_url") or len(out))
        if node_id in seen:
            continue
        seen.add(node_id)
        out.append(node)
    return out


def local_identity(policy: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(get_path(policy, ["node", "identity_path"], "reports/hive_node_identity.json"))
    value = read_json(path, {})
    if isinstance(value, dict) and value.get("node_id"):
        return {
            "node_id": value.get("node_id"),
            "node_name": value.get("node_name") or f"{socket.gethostname()}-{platform.system().lower()}",
        }
    return {
        "node_id": f"unregistered-{safe_id(socket.gethostname())}",
        "node_name": f"{socket.gethostname()}-{platform.system().lower()}",
    }


def active_hive_id(policy: dict[str, Any]) -> str:
    for path in [
        ["hive", "hive_id"],
        ["node", "hive_id"],
    ]:
        value = get_path(policy, path, "")
        if value:
            return str(value)
    profile = read_json(ROOT / "configs" / "hive_profile.local.json", {})
    return str(profile.get("hive_id") or "")


def owner_context() -> dict[str, Any]:
    return {"ok": True, "authenticated": True, "role": "owner", "token_kind": "loopback"}


def unique_clean(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text[:120])
    return out


def safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip().lower()).strip(".-")
    return text[:64] or "unassigned"


def safe_label(value: str, limit: int) -> str:
    return re.sub(r"[^A-Za-z0-9_.: /-]+", "_", str(value).strip())[:limit] or "unknown"


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
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for item in path:
        if not isinstance(cur, dict) or item not in cur:
            return default
        cur = cur[item]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
