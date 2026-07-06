"""Operator roaming profile helpers for the Theseus Hive node."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import quote

from hive_node_common import get_path


def operator_revocation_hint(auth_context: dict[str, Any]) -> dict[str, Any]:
    token_kind = str(auth_context.get("token_kind") or "")
    user_id = str(auth_context.get("user_id") or "")
    if token_kind == "user_token" and user_id:
        return {
            "kind": "user_token",
            "user_id": user_id,
            "command": f"theseus hive revoke-user {user_id}",
            "lost_device_action": "revoke_this_user_or_rotate_their_token",
        }
    return {
        "kind": token_kind or "legacy_owner",
        "command": "rotate the private Hive invite token and reinstall/update trusted nodes",
        "lost_device_action": "legacy owner tokens are broad; prefer per-user phone tokens for daily mobile use",
    }

def ios_profile_url(profile: dict[str, Any]) -> str:
    raw = json.dumps(compact_operator_profile(profile), separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return "theseushive://join?profile=" + encoded

def compact_ios_profile_url(profile: dict[str, Any]) -> str:
    endpoints = [str(item) for item in profile.get("coordinator_urls") or [] if item]
    token = str(profile.get("operator_token") or profile.get("join_token") or "")
    if not endpoints or not token:
        return ""
    endpoint = strip_http_for_compact_profile(sorted(endpoints, key=lambda item: len(item))[0])
    base = f"theseushive://join?u={quote(endpoint, safe=':.[]')}&t={quote(token, safe='-_.~')}"
    hive = str(profile.get("hive_id") or "")
    with_hive = base + f"&h={quote(hive, safe='-_.~')}" if hive else base
    return with_hive if len(with_hive.encode("utf-8")) <= 108 else base

def compact_operator_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": "project_theseus_hive_operator_roaming_profile_v1",
        "hive_id": profile.get("hive_id"),
        "hive_name": profile.get("hive_name"),
        "tier": profile.get("tier"),
        "coordinator_url": profile.get("coordinator_url"),
        "coordinator_urls": profile.get("coordinator_urls") or [],
        "node_urls": profile.get("node_urls") or [],
        "relay_urls": profile.get("relay_urls") or [],
        "operator_urls": profile.get("operator_urls") or [],
        "bonjour": get_path(profile, ["roaming", "bonjour"], profile.get("bonjour") or {}),
        "handover": get_path(profile, ["roaming", "handover"], profile.get("handover") or {}),
        "join_token": profile.get("operator_token") or profile.get("join_token"),
        "operator_token": profile.get("operator_token") or profile.get("join_token"),
        "roaming": {
            "strategy": get_path(profile, ["roaming", "strategy"], "iphone_try_last_good_then_lan_then_private_tunnel_then_https_relay"),
            "endpoint_selection": get_path(profile, ["roaming", "endpoint_selection"], {}),
            "bonjour": get_path(profile, ["roaming", "bonjour"], profile.get("bonjour") or {}),
            "handover": get_path(profile, ["roaming", "handover"], profile.get("handover") or {}),
            "endpoints": [
                {
                    "kind": row.get("kind"),
                    "url": row.get("url"),
                    "operator_url": row.get("operator_url"),
                    "transport": row.get("transport"),
                    "priority": row.get("priority"),
                    "source": row.get("source"),
                }
                for row in get_path(profile, ["roaming", "endpoints"], [])
                if isinstance(row, dict) and row.get("url")
            ],
        },
    }

def strip_http_for_compact_profile(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if text.startswith("http://"):
        return text[len("http://") :]
    return text

def qr_svg_for_url(value: str) -> str:
    if not value:
        return ""
    try:
        from theseus_qr import qr_svg

        return qr_svg(value)
    except Exception:
        return ""
