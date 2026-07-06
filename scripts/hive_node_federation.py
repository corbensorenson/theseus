"""Federation and join-token helpers for the Theseus Hive node."""

from __future__ import annotations

import argparse
import os
from typing import Any
from urllib.parse import urlparse

from hive_node_common import ROOT, get_path, read_json


def unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = value.strip().rstrip("/")
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out

def apply_runtime_join_overrides(policy: dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "relay_url", ""):
        os.environ[str(get_path(policy, ["federation", "relay_url_env"], "THESEUS_HIVE_RELAY_URL"))] = args.relay_url
    if getattr(args, "hive_id", ""):
        os.environ[str(get_path(policy, ["federation", "hive_id_env"], "THESEUS_HIVE_ID"))] = args.hive_id
    if getattr(args, "tier", ""):
        os.environ["THESEUS_HIVE_TIER"] = args.tier

def join_config(policy: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json"))
    return read_json(path, {})

def hive_id(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["federation", "hive_id_env"], "THESEUS_HIVE_ID"))
    cfg = join_config(policy)
    return os.environ.get(env_name) or str(cfg.get("hive_id") or "local")

def hive_tier(policy: dict[str, Any]) -> str:
    cfg = join_config(policy)
    return os.environ.get("THESEUS_HIVE_TIER") or str(cfg.get("tier") or get_path(policy, ["federation", "default_tier"], "private"))

def relay_url(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["federation", "relay_url_env"], "THESEUS_HIVE_RELAY_URL"))
    cfg = join_config(policy)
    return os.environ.get(env_name) or str(cfg.get("relay_url") or "")

def coordinator_url(policy: dict[str, Any]) -> str:
    urls = coordinator_urls(policy)
    return urls[0] if urls else ""

def coordinator_urls(policy: dict[str, Any]) -> list[str]:
    cfg = join_config(policy)
    values: list[str] = [os.environ.get("THESEUS_HIVE_COORDINATOR_URL", ""), str(cfg.get("coordinator_url") or "")]
    for key in ["coordinator_urls", "node_urls", "operator_urls"]:
        rows = cfg.get(key) if isinstance(cfg.get(key), list) else []
        values.extend(str(item) for item in rows)
    profiles = read_json(ROOT / str(get_path(policy, ["federation", "profiles_path"], "configs/hive_profiles.local.json")), {})
    active = str(profiles.get("active_profile_id") or "") if isinstance(profiles, dict) else ""
    for profile in profiles.get("profiles", []) if isinstance(profiles.get("profiles"), list) else []:
        if not isinstance(profile, dict) or (active and profile.get("profile_id") != active):
            continue
        values.append(str(profile.get("coordinator_url") or ""))
        for key in ["coordinator_urls", "node_urls", "operator_urls"]:
            rows = profile.get(key) if isinstance(profile.get(key), list) else []
            values.extend(str(item) for item in rows)
    remote = read_json(ROOT / "reports" / "hive_remote_access_status.json", {})
    mobile = remote.get("mobile_roaming") if isinstance(remote.get("mobile_roaming"), dict) else {}
    roaming = mobile.get("roaming") if isinstance(mobile.get("roaming"), dict) else {}
    values.append(str(roaming.get("coordinator_url") or ""))
    for key in ["coordinator_urls", "node_urls", "operator_urls"]:
        rows = roaming.get(key) if isinstance(roaming.get(key), list) else []
        values.extend(str(item) for item in rows)
    return unique_nonempty([normalize_node_base_url(value) for value in values])

def normalize_node_base_url(value: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        return ""
    candidate = raw if "://" in raw else f"http://{raw}"
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        return raw
    path = parsed.path.rstrip("/")
    if path in {"/mobile", "/operator", "/m"} or path.startswith("/api/hive"):
        path = ""
    return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")

def join_token(policy: dict[str, Any]) -> str:
    cfg = join_config(policy)
    return shared_secret(policy) or str(cfg.get("join_token") or "")

def shared_secret(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["security", "shared_secret_env"], "THESEUS_HIVE_SECRET"))
    env_value = os.environ.get(env_name, "")
    if env_value:
        return env_value
    join_cfg = join_config(policy)
    if isinstance(join_cfg, dict) and join_cfg.get("join_token"):
        return str(join_cfg.get("join_token") or "")
    profiles = read_json(ROOT / str(get_path(policy, ["federation", "profiles_path"], "configs/hive_profiles.local.json")), {})
    active = str(profiles.get("active_profile_id") or "") if isinstance(profiles, dict) else ""
    for profile in profiles.get("profiles", []) if isinstance(profiles.get("profiles"), list) else []:
        if isinstance(profile, dict) and (not active or profile.get("profile_id") == active):
            token = str(profile.get("join_token") or "")
            if token:
                return token
    return ""
