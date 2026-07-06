"""Privacy-preserving Hive voice-following route substrate.

This module does not perform speech recognition, speech synthesis, wake-word
detection, or raw audio relay. It records room/device capability, accepts tiny
voice-presence scores from local listeners, and chooses which trusted Hive node
should listen/respond as a user moves between rooms.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
POLICY_PATH = CONFIGS / "hive_policy.json"

_AUDIO_CACHE: dict[str, Any] = {"created_monotonic": 0.0, "report": {}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Project Theseus Hive voice-following status and routing.")
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Show local voice-following capability and privacy posture.")
    status.add_argument("--out", default="reports/hive_voice_following_status.json")

    configure = sub.add_parser("configure-room", help="Set this node's room and opt-in mic/speaker role.")
    configure.add_argument("--room-id", default="")
    configure.add_argument("--room-name", default="")
    mic = configure.add_mutually_exclusive_group()
    mic.add_argument("--microphone", dest="microphone", action="store_true")
    mic.add_argument("--no-microphone", dest="microphone", action="store_false")
    configure.set_defaults(microphone=None)
    speaker = configure.add_mutually_exclusive_group()
    speaker.add_argument("--speaker", dest="speaker", action="store_true")
    speaker.add_argument("--no-speaker", dest="speaker", action="store_false")
    configure.set_defaults(speaker=None)
    configure.add_argument("--priority", type=int, default=-1)
    configure.add_argument("--out", default="reports/hive_voice_following_status.json")

    presence = sub.add_parser("presence", help="Record a local voice-presence score without storing audio.")
    presence.add_argument("--score", type=float, required=True)
    presence.add_argument("--source", default="manual")
    presence.add_argument("--room-id", default="")
    presence.add_argument("--room-name", default="")
    presence.add_argument("--rms-db", type=float, default=None)
    presence.add_argument("--direction", type=float, default=None)
    presence.add_argument("--out", default="reports/hive_voice_presence_last.json")

    route = sub.add_parser("route", help="Choose the current listen/respond route across visible nodes.")
    route.add_argument("--out", default="reports/hive_voice_following_route.json")

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command in {None, "status"}:
        report = status_report(policy=policy, write_report=False)
        write_json(ROOT / args.out, report)
    elif args.command == "configure-room":
        report = configure_room(
            policy=policy,
            room_id=args.room_id,
            room_name=args.room_name,
            microphone=args.microphone,
            speaker=args.speaker,
            priority=args.priority,
        )
        if args.out:
            write_json(ROOT / args.out, status_report(policy=policy, write_report=False))
    elif args.command == "presence":
        report = presence_update(
            policy=policy,
            payload={
                "score": args.score,
                "source": args.source,
                "room_id": args.room_id,
                "room_name": args.room_name,
                "rms_db": args.rms_db,
                "direction_degrees": args.direction,
            },
            requester={"source": "cli"},
            write_report=True,
        )
        if args.out:
            write_json(ROOT / args.out, report)
    elif args.command == "route":
        report = route_decision(policy=policy, write_report=True)
        if args.out:
            write_json(ROOT / args.out, report)
    else:
        parser.print_help()
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def status_report(*, policy: dict[str, Any] | None = None, write_report: bool = True) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    config = voice_config(policy)
    identity = local_identity(policy)
    devices = detect_audio_devices_cached(policy)
    state = voice_state(policy)
    latest = latest_for_node(policy, str(identity.get("node_id") or ""))
    room = room_config(config)
    mic_cfg = config.get("microphone") if isinstance(config.get("microphone"), dict) else {}
    speaker_cfg = config.get("speaker") if isinstance(config.get("speaker"), dict) else {}
    enabled = bool(get_path(policy, ["voice_following", "enabled"], True)) and bool(config.get("enabled", True))
    ready_to_listen = enabled and bool(mic_cfg.get("enabled")) and bool(devices.get("microphone_available"))
    ready_to_respond = enabled and bool(speaker_cfg.get("enabled", True)) and bool(devices.get("speaker_available"))
    presence = public_presence(latest)
    local_route = route_from_rows(
        policy,
        rows=[
            {
                "node_id": identity.get("node_id"),
                "node_name": identity.get("node_name"),
                "api_url": "",
                "is_local": True,
                "voice_known": True,
                "room_id": room.get("room_id"),
                "room_name": room.get("name"),
                "ready_to_listen": ready_to_listen,
                "ready_to_respond": ready_to_respond,
                "microphone_available": bool(devices.get("microphone_available")),
                "speaker_available": bool(devices.get("speaker_available")),
                "speaker_priority": int(speaker_cfg.get("priority") or config.get("priority") or 50),
                "presence": presence,
                "privacy": get_path(policy, ["voice_following", "raw_audio_policy"], "never_relay_raw_audio_by_default"),
            }
        ],
        state=state,
    )
    report = {
        "ok": True,
        "policy": "project_theseus_hive_voice_following_status_v0",
        "created_utc": now(),
        "enabled": enabled,
        "node": {
            "node_id": identity.get("node_id"),
            "node_name": identity.get("node_name"),
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "room": room,
        "microphone": {
            "enabled": bool(mic_cfg.get("enabled")),
            "available": bool(devices.get("microphone_available")),
            "ready": ready_to_listen,
            "device_names": devices.get("microphones") or [],
            "source": mic_cfg.get("source") or "local_presence_detector",
        },
        "speaker": {
            "enabled": bool(speaker_cfg.get("enabled", True)),
            "available": bool(devices.get("speaker_available")),
            "ready": ready_to_respond,
            "device_names": devices.get("speakers") or [],
            "priority": int(speaker_cfg.get("priority") or config.get("priority") or 50),
        },
        "presence": presence,
        "route": local_route,
        "security": privacy_summary(policy),
        "next_actions": next_actions(ready_to_listen, ready_to_respond, room, devices),
    }
    if write_report:
        write_json(ROOT / str(get_path(policy, ["voice_following", "status_path"], "reports/hive_voice_following_status.json")), report)
    return report


def configure_room(
    *,
    policy: dict[str, Any],
    room_id: str,
    room_name: str,
    microphone: bool | None,
    speaker: bool | None,
    priority: int,
) -> dict[str, Any]:
    config = voice_config(policy)
    room = config.setdefault("room", {})
    if not isinstance(room, dict):
        room = {}
        config["room"] = room
    if room_id:
        room["room_id"] = safe_id(room_id)
    if room_name:
        room["name"] = room_name.strip()[:80]
    if not room.get("room_id"):
        room["room_id"] = safe_id(room.get("name") or socket.gethostname())
    if not room.get("name"):
        room["name"] = room["room_id"]
    mic_cfg = config.setdefault("microphone", {})
    if not isinstance(mic_cfg, dict):
        mic_cfg = {}
        config["microphone"] = mic_cfg
    if microphone is not None:
        mic_cfg["enabled"] = bool(microphone)
    speaker_cfg = config.setdefault("speaker", {})
    if not isinstance(speaker_cfg, dict):
        speaker_cfg = {}
        config["speaker"] = speaker_cfg
    if speaker is not None:
        speaker_cfg["enabled"] = bool(speaker)
    if priority >= 0:
        speaker_cfg["priority"] = max(0, min(priority, 100))
        config["priority"] = speaker_cfg["priority"]
    write_voice_config(policy, config)
    report = {
        "ok": True,
        "policy": "project_theseus_hive_voice_room_configured_v0",
        "created_utc": now(),
        "config_path": rel(voice_config_path(policy)),
        "room": room_config(config),
        "microphone_enabled": bool(mic_cfg.get("enabled")),
        "speaker_enabled": bool(speaker_cfg.get("enabled", True)),
        "speaker_priority": int(speaker_cfg.get("priority") or config.get("priority") or 50),
    }
    write_json(REPORTS / "hive_voice_following_configure_last.json", report)
    return report


def presence_update(
    *,
    policy: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    requester: dict[str, Any] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    payload = payload or {}
    if not bool(get_path(policy, ["voice_following", "enabled"], True)):
        return {"ok": False, "error": "voice_following_disabled"}
    forbidden = {"audio", "audio_b64", "pcm", "wav", "samples", "raw_audio", "transcript"}
    if forbidden & set(payload):
        return {
            "ok": False,
            "error": "raw_audio_or_transcript_not_accepted",
            "forbidden_fields": sorted(forbidden & set(payload)),
        }
    config = voice_config(policy)
    identity = local_identity(policy)
    room = room_config(config)
    room_id = safe_id(str(payload.get("room_id") or room.get("room_id") or "unassigned"))
    room_name = str(payload.get("room_name") or room.get("name") or room_id)[:80]
    score = clamp_float(payload.get("score"), 0.0, 1.0)
    event = {
        "policy": "project_theseus_hive_voice_presence_event_v0",
        "created_utc": now(),
        "node_id": identity.get("node_id"),
        "node_name": identity.get("node_name"),
        "hostname": socket.gethostname(),
        "room_id": room_id,
        "room_name": room_name,
        "score": score,
        "source": safe_label(str(payload.get("source") or "manual"), 60),
        "requester": requester or {},
        "audio_retained": False,
        "raw_audio_relayed": False,
    }
    rms = optional_float(payload.get("rms_db"))
    if rms is not None:
        event["rms_db"] = round(rms, 3)
    direction = optional_float(payload.get("direction_degrees") or payload.get("direction"))
    if direction is not None:
        event["direction_degrees"] = round(direction % 360.0, 3)
    state = voice_state(policy)
    latest = state.setdefault("latest_by_node", {})
    if not isinstance(latest, dict):
        latest = {}
        state["latest_by_node"] = latest
    latest[str(identity.get("node_id") or socket.gethostname())] = event
    state["updated_utc"] = now()
    state["route"] = route_from_rows(policy, rows=rows_from_status(policy=policy, status=None, peers=[]), state=state)
    write_voice_state(policy, state)
    report = {
        "ok": True,
        "policy": "project_theseus_hive_voice_presence_recorded_v0",
        "created_utc": now(),
        "presence": public_presence(event),
        "route": state.get("route") if isinstance(state.get("route"), dict) else {},
        "security": privacy_summary(policy),
    }
    if write_report:
        append_jsonl(ROOT / str(get_path(policy, ["voice_following", "presence_ledger_path"], "reports/hive_voice_presence_ledger.jsonl")), event)
        write_json(REPORTS / "hive_voice_presence_last.json", report)
    return report


def route_decision(
    *,
    policy: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
    peers: list[dict[str, Any]] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    state = voice_state(policy)
    rows = rows_from_status(policy=policy, status=status, peers=peers or [])
    route = route_from_rows(policy, rows=rows, state=state)
    state["route"] = route
    state["updated_utc"] = now()
    if write_report:
        write_voice_state(policy, state)
        write_json(ROOT / str(get_path(policy, ["voice_following", "route_report_path"], "reports/hive_voice_following_route.json")), route)
    return route


def operator_summary(policy: dict[str, Any], status: dict[str, Any], peers: list[dict[str, Any]]) -> dict[str, Any]:
    route = route_decision(policy=policy, status=status, peers=peers, write_report=True)
    return {
        "policy": "project_theseus_hive_operator_voice_following_v0",
        "enabled": bool(get_path(policy, ["voice_following", "enabled"], True)),
        "route": route,
        "nodes": route.get("nodes") or [],
        "node_count": len(route.get("nodes") or []),
        "ready_listener_count": sum(1 for row in route.get("nodes") or [] if row.get("ready_to_listen")),
        "ready_speaker_count": sum(1 for row in route.get("nodes") or [] if row.get("ready_to_respond")),
        "status_endpoint": "/api/hive/voice/status",
        "route_endpoint": "/api/hive/voice/route",
        "presence_endpoint": "/api/hive/voice/presence",
        "security": privacy_summary(policy),
    }


def compact_status(report: dict[str, Any]) -> dict[str, Any]:
    presence = report.get("presence") if isinstance(report.get("presence"), dict) else {}
    room = report.get("room") if isinstance(report.get("room"), dict) else {}
    microphone = report.get("microphone") if isinstance(report.get("microphone"), dict) else {}
    speaker = report.get("speaker") if isinstance(report.get("speaker"), dict) else {}
    node = report.get("node") if isinstance(report.get("node"), dict) else {}
    return {
        "enabled": bool(report.get("enabled")),
        "node_id": node.get("node_id"),
        "node_name": node.get("node_name"),
        "room_id": room.get("room_id"),
        "room_name": room.get("name"),
        "ready_to_listen": bool(microphone.get("ready")),
        "ready_to_respond": bool(speaker.get("ready")),
        "microphone_available": bool(microphone.get("available")),
        "speaker_available": bool(speaker.get("available")),
        "speaker_priority": speaker.get("priority"),
        "presence": presence,
        "route": report.get("route") if isinstance(report.get("route"), dict) else {},
        "privacy": get_path(report, ["security", "raw_audio_policy"], "never_relay_raw_audio_by_default"),
    }


def rows_from_status(
    *,
    policy: dict[str, Any],
    status: dict[str, Any] | None,
    peers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    local_status = status if isinstance(status, dict) and status.get("node_id") else {}
    if not local_status:
        local_voice = status_report(policy=policy, write_report=False)
        identity = local_identity(policy)
        local_status = {
            "node_id": identity.get("node_id"),
            "node_name": identity.get("node_name"),
            "api_url": "",
            "voice_following": compact_status(local_voice),
            "platform": {"system": platform.system(), "machine": platform.machine()},
        }
    elif "voice_following" not in local_status:
        local_status = dict(local_status)
        local_status["voice_following"] = compact_status(status_report(policy=policy, write_report=False))
    rows.append(public_voice_row(local_status, is_local=True))
    for peer in peers:
        if isinstance(peer, dict):
            row = public_voice_row(peer, is_local=False)
            if row.get("voice_known"):
                rows.append(row)
    return rows


def route_from_rows(policy: dict[str, Any], *, rows: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    max_age = float(get_path(policy, ["voice_following", "max_presence_age_seconds"], 20))
    min_score = float(get_path(policy, ["voice_following", "min_presence_score"], 0.2))
    now_ts = time.time()
    latest_by_node = state.get("latest_by_node") if isinstance(state.get("latest_by_node"), dict) else {}
    candidates = []
    for row in rows:
        presence = row.get("presence") if isinstance(row.get("presence"), dict) else {}
        latest = latest_by_node.get(str(row.get("node_id") or "")) if isinstance(latest_by_node, dict) else {}
        if isinstance(latest, dict) and latest:
            latest_public = public_presence(latest)
            if not presence or (presence_age_seconds(latest_public, now_ts=now_ts) or 999999.0) <= (presence_age_seconds(presence, now_ts=now_ts) or 999999.0):
                presence = latest_public
                row["presence"] = presence
        score = optional_float(presence.get("score")) or 0.0
        age = presence_age_seconds(presence, now_ts=now_ts)
        row["presence_age_seconds"] = age
        if score >= min_score and (age is None or age <= max_age):
            candidates.append(row)
    active = max(
        candidates,
        key=lambda item: (
            optional_float(get_path(item, ["presence", "score"], 0.0)) or 0.0,
            -1.0 * float(item.get("presence_age_seconds") or 0.0),
            1 if item.get("is_local") else 0,
        ),
        default={},
    )
    speaker_nodes = [row for row in rows if row.get("ready_to_respond")]
    same_room_speakers = [
        row
        for row in speaker_nodes
        if active and row.get("room_id") and row.get("room_id") == active.get("room_id")
    ]
    pool = same_room_speakers or speaker_nodes
    speaker = max(
        pool,
        key=lambda item: (
            int(item.get("speaker_priority") or 50),
            1 if item.get("node_id") == active.get("node_id") else 0,
            1 if item.get("is_local") else 0,
        ),
        default={},
    )
    confidence = optional_float(get_path(active, ["presence", "score"], 0.0)) or 0.0
    if not active:
        route_state = "idle_no_recent_presence"
    elif not speaker:
        route_state = "heard_no_ready_speaker"
    else:
        route_state = "active"
    return {
        "ok": True,
        "policy": "project_theseus_hive_voice_route_v0",
        "created_utc": now(),
        "state": route_state,
        "active_room_id": active.get("room_id") if active else "",
        "active_room_name": active.get("room_name") if active else "",
        "confidence": round(confidence, 4),
        "listen_node": public_route_node(active),
        "respond_node": public_route_node(speaker),
        "nodes": rows,
        "decision": {
            "min_presence_score": min_score,
            "max_presence_age_seconds": max_age,
            "speaker_rule": get_path(policy, ["voice_following", "speaker_selection"], "nearest_recent_presence_with_ready_speaker"),
            "same_room_speaker_preferred": bool(same_room_speakers),
        },
        "security": privacy_summary(policy),
    }


def public_voice_row(node: dict[str, Any], *, is_local: bool) -> dict[str, Any]:
    voice = node.get("voice_following") if isinstance(node.get("voice_following"), dict) else {}
    presence = voice.get("presence") if isinstance(voice.get("presence"), dict) else {}
    return {
        "node_id": node.get("node_id") or voice.get("node_id"),
        "node_name": node.get("node_name") or voice.get("node_name"),
        "api_url": node.get("api_url") or "",
        "is_local": bool(is_local),
        "voice_known": bool(voice),
        "room_id": voice.get("room_id"),
        "room_name": voice.get("room_name"),
        "ready_to_listen": bool(voice.get("ready_to_listen")),
        "ready_to_respond": bool(voice.get("ready_to_respond")),
        "microphone_available": bool(voice.get("microphone_available")),
        "speaker_available": bool(voice.get("speaker_available")),
        "speaker_priority": voice.get("speaker_priority"),
        "presence": presence,
        "privacy": voice.get("privacy") or "never_relay_raw_audio_by_default",
    }


def public_route_node(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "node_id": row.get("node_id"),
        "node_name": row.get("node_name"),
        "room_id": row.get("room_id"),
        "room_name": row.get("room_name"),
        "api_url": row.get("api_url") or "",
        "is_local": bool(row.get("is_local")),
        "presence": row.get("presence") if isinstance(row.get("presence"), dict) else {},
    }


def public_presence(event: dict[str, Any]) -> dict[str, Any]:
    if not event:
        return {}
    return {
        "created_utc": event.get("created_utc"),
        "node_id": event.get("node_id"),
        "node_name": event.get("node_name"),
        "room_id": event.get("room_id"),
        "room_name": event.get("room_name"),
        "score": event.get("score"),
        "source": event.get("source"),
        "rms_db": event.get("rms_db"),
        "direction_degrees": event.get("direction_degrees"),
        "audio_retained": False,
        "raw_audio_relayed": False,
        "age_seconds": presence_age_seconds(event),
    }


def presence_age_seconds(event: dict[str, Any], *, now_ts: float | None = None) -> float | None:
    created = str(event.get("created_utc") or "")
    if not created:
        return None
    try:
        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError:
        return None
    created_ts = created_dt.timestamp()
    return round((now_ts if now_ts is not None else time.time()) - created_ts, 3)


def latest_for_node(policy: dict[str, Any], node_id: str) -> dict[str, Any]:
    state = voice_state(policy)
    latest = state.get("latest_by_node") if isinstance(state.get("latest_by_node"), dict) else {}
    value = latest.get(node_id) if isinstance(latest, dict) else {}
    return value if isinstance(value, dict) else {}


def voice_config(policy: dict[str, Any]) -> dict[str, Any]:
    path = voice_config_path(policy)
    if path.exists():
        value = read_json(path, {})
        if isinstance(value, dict):
            return normalized_voice_config(value)
    return normalized_voice_config({})


def normalized_voice_config(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("policy", "project_theseus_hive_voice_following_local_v0")
    config.setdefault("enabled", True)
    room = config.get("room") if isinstance(config.get("room"), dict) else {}
    room.setdefault("room_id", "unassigned")
    room.setdefault("name", socket.gethostname())
    config["room"] = room
    mic = config.get("microphone") if isinstance(config.get("microphone"), dict) else {}
    mic.setdefault("enabled", False)
    mic.setdefault("source", "local_presence_detector")
    config["microphone"] = mic
    speaker = config.get("speaker") if isinstance(config.get("speaker"), dict) else {}
    speaker.setdefault("enabled", True)
    speaker.setdefault("priority", int(config.get("priority") or 50))
    config["speaker"] = speaker
    config.setdefault("priority", int(speaker.get("priority") or 50))
    return config


def write_voice_config(policy: dict[str, Any], config: dict[str, Any]) -> None:
    config["updated_utc"] = now()
    write_json(voice_config_path(policy), normalized_voice_config(config))


def voice_config_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["voice_following", "config_path"], "configs/hive_voice_following.local.json"))


def voice_state(policy: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(get_path(policy, ["voice_following", "state_path"], "reports/hive_voice_following_state.json"))
    value = read_json(path, {})
    if not isinstance(value, dict):
        value = {}
    value.setdefault("policy", "project_theseus_hive_voice_following_state_v0")
    value.setdefault("latest_by_node", {})
    return value


def write_voice_state(policy: dict[str, Any], state: dict[str, Any]) -> None:
    path = ROOT / str(get_path(policy, ["voice_following", "state_path"], "reports/hive_voice_following_state.json"))
    current = read_json(path, {})
    if isinstance(current, dict):
        state["latest_by_node"] = merge_latest_by_node(
            current.get("latest_by_node") if isinstance(current.get("latest_by_node"), dict) else {},
            state.get("latest_by_node") if isinstance(state.get("latest_by_node"), dict) else {},
        )
    write_json(path, state)


def merge_latest_by_node(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in [left, right]:
        for node_id, event in source.items():
            if not isinstance(event, dict):
                continue
            existing = merged.get(str(node_id))
            if not isinstance(existing, dict) or event_timestamp(event) >= event_timestamp(existing):
                merged[str(node_id)] = event
    return merged


def event_timestamp(event: dict[str, Any]) -> float:
    try:
        return datetime.fromisoformat(str(event.get("created_utc") or "").replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def room_config(config: dict[str, Any]) -> dict[str, Any]:
    room = config.get("room") if isinstance(config.get("room"), dict) else {}
    room_id = safe_id(str(room.get("room_id") or room.get("name") or "unassigned"))
    return {
        "room_id": room_id,
        "name": str(room.get("name") or room_id)[:80],
        "zone": str(room.get("zone") or "")[:80],
    }


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


def detect_audio_devices_cached(policy: dict[str, Any]) -> dict[str, Any]:
    ttl = float(get_path(policy, ["voice_following", "audio_probe_cache_seconds"], 30))
    now_mono = time.monotonic()
    if _AUDIO_CACHE.get("report") and now_mono - float(_AUDIO_CACHE.get("created_monotonic") or 0.0) < ttl:
        return dict(_AUDIO_CACHE["report"])
    report = detect_audio_devices()
    _AUDIO_CACHE["created_monotonic"] = now_mono
    _AUDIO_CACHE["report"] = report
    return dict(report)


def detect_audio_devices() -> dict[str, Any]:
    system = platform.system()
    if system == "Darwin":
        return detect_audio_devices_macos()
    if system == "Windows":
        return detect_audio_devices_windows()
    return detect_audio_devices_linux()


def detect_audio_devices_macos() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType", "-json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=6,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return audio_device_report([], [], error=str(exc))
    if result.returncode != 0:
        return audio_device_report([], [], error=result.stderr.strip() or "system_profiler_failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return audio_device_report([], [], error="system_profiler_non_json")
    microphones: list[str] = []
    speakers: list[str] = []
    for item in iter_dicts(payload):
        name = str(item.get("_name") or item.get("coreaudio_input_source") or item.get("coreaudio_output_source") or "").strip()
        if not name:
            continue
        if item.get("coreaudio_device_input") or "input" in " ".join(str(key).lower() for key in item):
            microphones.append(name)
        if item.get("coreaudio_device_output") or item.get("coreaudio_default_audio_output_device") or item.get("coreaudio_default_audio_system_device"):
            speakers.append(name)
    return audio_device_report(microphones, speakers)


def detect_audio_devices_windows() -> dict[str, Any]:
    if not shutil.which("powershell"):
        return audio_device_report([], [], error="powershell_not_found")
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-PnpDevice -Class AudioEndpoint | Select-Object FriendlyName,Status | ConvertTo-Json -Compress",
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return audio_device_report([], [], error=str(exc))
    if result.returncode != 0:
        return audio_device_report([], [], error=result.stderr.strip() or "audio_endpoint_probe_failed")
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        payload = []
    rows = payload if isinstance(payload, list) else [payload]
    names = [str(row.get("FriendlyName") or "") for row in rows if isinstance(row, dict) and row.get("Status") == "OK"]
    microphones = [name for name in names if any(token in name.lower() for token in ["microphone", "mic", "input"])]
    speakers = [name for name in names if any(token in name.lower() for token in ["speaker", "headphone", "output", "audio"])]
    return audio_device_report(microphones, speakers)


def detect_audio_devices_linux() -> dict[str, Any]:
    microphones: list[str] = []
    speakers: list[str] = []
    if shutil.which("pactl"):
        microphones.extend(command_lines(["pactl", "list", "sources", "short"], timeout=3))
        speakers.extend(command_lines(["pactl", "list", "sinks", "short"], timeout=3))
    if not microphones and shutil.which("arecord"):
        microphones.extend(command_lines(["arecord", "-l"], timeout=3))
    if not speakers and shutil.which("aplay"):
        speakers.extend(command_lines(["aplay", "-l"], timeout=3))
    return audio_device_report(microphones[:12], speakers[:12])


def audio_device_report(microphones: list[str], speakers: list[str], *, error: str = "") -> dict[str, Any]:
    microphones = unique_clean(microphones)
    speakers = unique_clean(speakers)
    return {
        "microphone_available": bool(microphones),
        "speaker_available": bool(speakers),
        "microphones": microphones[:8],
        "speakers": speakers[:8],
        "error": error,
    }


def command_lines(command: list[str], *, timeout: int) -> list[str]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def iter_dicts(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        rows.append(value)
        for child in value.values():
            rows.extend(iter_dicts(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(iter_dicts(child))
    return rows


def privacy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_audio_policy": get_path(policy, ["voice_following", "raw_audio_policy"], "never_relay_raw_audio_by_default"),
        "presence_events_only": True,
        "raw_audio_retention": False,
        "transcript_required_for_routing": False,
        "external_provider_stt_tts": "forbidden",
        "remote_execution": "registered_task_kinds_only",
    }


def next_actions(ready_to_listen: bool, ready_to_respond: bool, room: dict[str, Any], devices: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if room.get("room_id") == "unassigned":
        actions.append("Run: theseus voice configure-room --room-id kitchen --room-name Kitchen --microphone --speaker")
    if devices.get("microphone_available") and not ready_to_listen:
        actions.append("Opt in this room mic with: theseus voice configure-room --microphone")
    if not devices.get("microphone_available"):
        actions.append("Attach or enable a local microphone before using this node as a listener.")
    if not ready_to_respond:
        actions.append("Attach or enable a local speaker before using this node as a response target.")
    if not actions:
        actions.append("Post local presence from a native detector or smoke test with: theseus voice presence --score 0.8 --source smoke")
    return actions


def unique_clean(values: list[str]) -> list[str]:
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
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower()).strip(".-")
    return text[:64] or "unassigned"


def safe_label(value: str, limit: int) -> str:
    return re.sub(r"[^A-Za-z0-9_.: -]+", "_", value.strip())[:limit] or "unknown"


def clamp_float(value: Any, low: float, high: float) -> float:
    number = optional_float(value)
    if number is None:
        return low
    return round(max(low, min(high, number)), 4)


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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
