"""Manage local Project Theseus Hive profiles.

Profiles make it easy for one install to switch between a home hive, a work
hive, a friend/company hive, or future public hive without hand-editing env vars.
The profile store is ignored by git because it contains join tokens.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "hive_policy.json"
PROFILES_PATH = ROOT / "configs" / "hive_profiles.local.json"
JOIN_PATH = ROOT / "configs" / "hive_join.local.json"

sys.path.insert(0, str(ROOT / "scripts"))
import license_manager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    create = sub.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--tier", choices=["private", "friends_family", "company", "public"], default="private")
    create.add_argument("--hive-id", default="")
    create.add_argument("--relay-url", default="")
    create.add_argument("--join-token", default="")
    create.add_argument("--mode", choices=["lan", "relay"], default="lan")
    create.add_argument("--activate", action="store_true")
    create.add_argument("--out", default="")

    add = sub.add_parser("add-invite")
    add.add_argument("--invite", required=True)
    add.add_argument("--name", default="")
    add.add_argument("--activate", action="store_true")
    add.add_argument("--out", default="")

    switch = sub.add_parser("switch")
    switch.add_argument("--profile-id", required=True)
    switch.add_argument("--out", default="")

    sub.add_parser("list").add_argument("--out", default="")

    args = parser.parse_args()
    if args.command == "create":
        report = create_profile(
            name=args.name,
            tier=args.tier,
            hive_id=args.hive_id,
            relay_url=args.relay_url,
            join_token=args.join_token,
            mode=args.mode,
            activate=args.activate,
        )
    elif args.command == "add-invite":
        invite = read_json(ROOT / args.invite, {})
        report = add_invite_profile(invite, name=args.name, activate=args.activate)
    elif args.command == "switch":
        report = switch_profile(args.profile_id)
    elif args.command == "list":
        report = load_profiles()
    else:
        parser.print_help()
        return 2
    if getattr(args, "out", ""):
        write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def create_profile(
    *,
    name: str,
    tier: str,
    hive_id: str,
    relay_url: str,
    join_token: str,
    mode: str,
    activate: bool,
    coordinator_url: str = "",
    coordinator_urls: list[str] | None = None,
    node_urls: list[str] | None = None,
    relay_urls: list[str] | None = None,
    operator_urls: list[str] | None = None,
) -> dict[str, Any]:
    if tier == "public":
        return {
            "ok": False,
            "error": "public_hive_disabled",
            "message": "Public Hive is visible in the wizard but disabled until signed workers, sandboxing, reputation, and abuse controls exist.",
        }
    license_check = license_manager.check_feature(license_manager.feature_for_hive_tier(tier), context={"requested_tier": tier})
    if not license_check.get("allowed"):
        return {
            "ok": False,
            "error": "license_required",
            "tier": tier,
            "feature": license_manager.feature_for_hive_tier(tier),
            "license": compact_license_check(license_check),
            "message": license_check.get("next_action") or "Register this install or import a license before creating this hive.",
        }
    profile = {
        "profile_id": f"hive-{secrets.token_hex(6)}",
        "name": name,
        "tier": tier,
        "hive_id": hive_id or f"theseus-{secrets.token_hex(6)}",
        "relay_url": relay_url,
        "coordinator_url": coordinator_url,
        "coordinator_urls": coordinator_urls or [],
        "node_urls": node_urls or [],
        "relay_urls": relay_urls or [],
        "operator_urls": operator_urls or [],
        "join_token": join_token or secrets.token_urlsafe(32),
        "mode": mode,
        "created_utc": now(),
        "last_used_utc": now() if activate else "",
    }
    store = load_profiles_private()
    store.setdefault("profiles", []).append(profile)
    if activate:
        store["active_profile_id"] = profile["profile_id"]
        write_join_config(profile)
    write_json(PROFILES_PATH, store)
    return {"ok": True, "profile": public_profile(profile), "active": activate}


def add_invite_profile(invite: dict[str, Any], *, name: str, activate: bool) -> dict[str, Any]:
    missing = [key for key in ["hive_id", "join_token", "tier"] if not invite.get(key)]
    if missing:
        return {"ok": False, "error": "invalid_invite", "missing": missing}
    return create_profile(
        name=name or str(invite.get("hive_name") or invite.get("hive_id")),
        tier=str(invite.get("tier") or "private"),
        hive_id=str(invite.get("hive_id") or ""),
        relay_url=str(invite.get("relay_url") or ""),
        join_token=str(invite.get("join_token") or ""),
        mode="relay" if invite.get("relay_url") else "lan",
        activate=activate,
        coordinator_url=str(invite.get("coordinator_url") or ""),
        coordinator_urls=invite.get("coordinator_urls") if isinstance(invite.get("coordinator_urls"), list) else [],
        node_urls=invite.get("node_urls") if isinstance(invite.get("node_urls"), list) else [],
        relay_urls=invite.get("relay_urls") if isinstance(invite.get("relay_urls"), list) else [],
        operator_urls=invite.get("operator_urls") if isinstance(invite.get("operator_urls"), list) else [],
    )


def switch_profile(profile_id: str) -> dict[str, Any]:
    store = load_profiles_private()
    profiles = store.get("profiles") if isinstance(store.get("profiles"), list) else []
    for profile in profiles:
        if profile.get("profile_id") == profile_id:
            profile["last_used_utc"] = now()
            store["active_profile_id"] = profile_id
            write_join_config(profile)
            write_json(PROFILES_PATH, store)
            return {"ok": True, "active_profile": public_profile(profile)}
    return {"ok": False, "error": "profile_not_found", "profile_id": profile_id}


def active_profile() -> dict[str, Any]:
    store = load_profiles_private()
    active_id = store.get("active_profile_id")
    for profile in store.get("profiles") or []:
        if profile.get("profile_id") == active_id:
            return profile
    return {}


def load_profiles() -> dict[str, Any]:
    value = load_profiles_private()
    if not isinstance(value, dict):
        value = {}
    value.setdefault("ok", True)
    value.setdefault("policy", "project_theseus_hive_profiles_v0")
    value.setdefault("active_profile_id", "")
    value.setdefault("profiles", [])
    public = dict(value)
    public["profiles"] = [public_profile(profile) for profile in value.get("profiles") or []]
    return public


def load_profiles_private() -> dict[str, Any]:
    value = read_json(PROFILES_PATH, {}) if PROFILES_PATH.exists() else {}
    if isinstance(value, dict) and value.get("profiles"):
        value.setdefault("ok", True)
        value.setdefault("policy", "project_theseus_hive_profiles_v0")
        return value
    fallback = profile_from_join_config()
    if fallback:
        return {
            "ok": True,
            "policy": "project_theseus_hive_profiles_v0",
            "active_profile_id": fallback.get("profile_id"),
            "profiles": [fallback],
            "source": "legacy_join_config",
        }
    if isinstance(value, dict):
        value.setdefault("ok", True)
        value.setdefault("policy", "project_theseus_hive_profiles_v0")
        value.setdefault("active_profile_id", "")
        value.setdefault("profiles", [])
        return value
    return {
        "ok": True,
        "policy": "project_theseus_hive_profiles_v0",
        "active_profile_id": "",
        "profiles": [],
    }


def profile_from_join_config() -> dict[str, Any]:
    join = read_json(JOIN_PATH, {})
    if not isinstance(join, dict):
        return {}
    hive_id = str(join.get("hive_id") or "")
    join_token = str(join.get("join_token") or "")
    tier = str(join.get("tier") or "private")
    if not hive_id or not join_token:
        return {}
    relay_url = str(join.get("relay_url") or "")
    profile_id = str(join.get("profile_id") or f"hive-joined-{safe_id(hive_id)}")
    created = str(join.get("created_utc") or file_mtime_utc(JOIN_PATH) or now())
    return {
        "profile_id": profile_id,
        "name": str(join.get("hive_name") or hive_id),
        "tier": tier,
        "hive_id": hive_id,
        "relay_url": relay_url,
        "coordinator_url": str(join.get("coordinator_url") or ""),
        "coordinator_urls": join.get("coordinator_urls") if isinstance(join.get("coordinator_urls"), list) else [],
        "node_urls": join.get("node_urls") if isinstance(join.get("node_urls"), list) else [],
        "relay_urls": join.get("relay_urls") if isinstance(join.get("relay_urls"), list) else [],
        "operator_urls": join.get("operator_urls") if isinstance(join.get("operator_urls"), list) else [],
        "join_token": join_token,
        "mode": str(join.get("mode") or ("relay" if relay_url else "lan")),
        "created_utc": created,
        "last_used_utc": created,
    }


def write_join_config(profile: dict[str, Any]) -> None:
    payload = {
        "policy": "project_theseus_hive_join_config_v0",
        "created_utc": now(),
        "profile_id": profile.get("profile_id"),
        "hive_id": profile.get("hive_id"),
        "hive_name": profile.get("name"),
        "tier": profile.get("tier"),
        "relay_url": profile.get("relay_url"),
        "coordinator_url": profile.get("coordinator_url"),
        "coordinator_urls": profile.get("coordinator_urls") if isinstance(profile.get("coordinator_urls"), list) else [],
        "node_urls": profile.get("node_urls") if isinstance(profile.get("node_urls"), list) else [],
        "relay_urls": profile.get("relay_urls") if isinstance(profile.get("relay_urls"), list) else [],
        "operator_urls": profile.get("operator_urls") if isinstance(profile.get("operator_urls"), list) else [],
        "join_token": profile.get("join_token"),
        "mode": profile.get("mode"),
    }
    write_json(JOIN_PATH, payload)


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    clean = dict(profile)
    token = str(clean.pop("join_token", "") or "")
    clean["join_token_configured"] = bool(token)
    clean["join_token_preview"] = ""
    return clean


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)[-24:] or secrets.token_hex(6)


def file_mtime_utc(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return ""


def compact_license_check(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed": report.get("allowed"),
        "registration_complete": report.get("registration_complete"),
        "tier": get_path(report, ["entitlement", "tier"], ""),
        "source": get_path(report, ["entitlement", "source"], ""),
        "feature_check": report.get("feature_check", {}),
    }


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


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
