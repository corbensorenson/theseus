"""Stable node identity helpers for the Theseus Hive node."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import uuid
from pathlib import Path
from typing import Any

from hive_node_common import ROOT, get_path, now, read_json, write_json


def load_identity(policy: dict[str, Any]) -> dict[str, Any]:
    node_name = f"{socket.gethostname()}-{platform.system().lower()}"
    paths = identity_candidate_paths(policy)
    configured_path = configured_identity_path(policy)
    stable_path = machine_identity_path()
    for path in paths:
        value = read_identity_file(path)
        if not value:
            continue
        if value.get("node_name") and value.get("node_name") != node_name:
            continue
        value.setdefault("node_name", node_name)
        if not value.get("node_signing_secret"):
            value["node_signing_secret"] = uuid.uuid4().hex + uuid.uuid4().hex
            value["node_public_key"] = hashlib.sha256(str(value["node_signing_secret"]).encode("utf-8")).hexdigest()
        value = reconcile_identity_with_registration(value, path)
        mirror_identity(value, [stable_path, configured_path])
        return value
    registered_node_id = registered_node_id_for_name(node_name)
    identity = {
        "policy": "project_theseus_hive_node_identity_v0",
        "node_id": registered_node_id or f"theseus-{uuid.uuid4().hex}",
        "node_name": node_name,
        "node_signing_secret": uuid.uuid4().hex + uuid.uuid4().hex,
        "created_utc": now(),
    }
    identity["node_public_key"] = hashlib.sha256(str(identity["node_signing_secret"]).encode("utf-8")).hexdigest()
    mirror_identity(identity, [stable_path, configured_path])
    return identity

def configured_identity_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["node", "identity_path"], "reports/hive_node_identity.json"))

def machine_identity_path() -> Path:
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return home / "Library" / "Application Support" / "Project Theseus Hive" / "hive_node_identity.json"
    if system == "Windows":
        base = Path(os.environ.get("APPDATA") or (home / "AppData" / "Roaming"))
        return base / "Project Theseus Hive" / "hive_node_identity.json"
    return home / ".config" / "project-theseus-hive" / "hive_node_identity.json"

def installed_app_identity_candidates() -> list[Path]:
    if platform.system() != "Darwin":
        return []
    home = Path.home()
    return [
        home / "Library" / "Application Support" / "Project Theseus Hive" / "app" / "current" / "reports" / "hive_node_identity.json",
        home / "Library" / "Application Support" / "ProjectTheseus" / "runtime" / "reports" / "hive_node_identity.json",
    ]

def identity_candidate_paths(policy: dict[str, Any]) -> list[Path]:
    paths = [machine_identity_path(), *installed_app_identity_candidates(), configured_identity_path(policy)]
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key and key not in seen:
            seen.add(key)
            out.append(path)
    return out

def read_identity_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) and value.get("node_id") else {}

def mirror_identity(identity: dict[str, Any], paths: list[Path]) -> None:
    for path in paths:
        try:
            existing = read_identity_file(path)
            if existing.get("node_id") == identity.get("node_id") and existing.get("node_signing_secret"):
                continue
            write_json(path, identity)
        except OSError:
            continue

def reconcile_identity_with_registration(identity: dict[str, Any], path: Path) -> dict[str, Any]:
    node_name = str(identity.get("node_name") or f"{socket.gethostname()}-{platform.system().lower()}")
    registered_node_id = registered_node_id_for_name(node_name)
    if not registered_node_id or identity.get("node_id") == registered_node_id:
        return identity
    aliases = identity.get("node_aliases")
    if not isinstance(aliases, list):
        aliases = []
    previous = str(identity.get("node_id") or "")
    if previous and previous not in aliases:
        aliases.append(previous)
    identity["node_id"] = registered_node_id
    identity["node_aliases"] = aliases
    identity["registration_reconciled_utc"] = now()
    write_json(path, identity)
    return identity

def registered_node_id_for_name(node_name: str) -> str:
    path = ROOT / "configs" / "theseus_registration.local.json"
    if not path.exists():
        return ""
    try:
        registration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    nodes = registration.get("registered_nodes") if isinstance(registration, dict) else []
    if not isinstance(nodes, list):
        return ""
    matching = [node for node in nodes if isinstance(node, dict) and str(node.get("node_name") or "") == node_name and node.get("node_id")]
    if matching:
        return str(matching[0].get("node_id") or "")
    if len(nodes) == 1 and isinstance(nodes[0], dict):
        return str(nodes[0].get("node_id") or "")
    return ""
