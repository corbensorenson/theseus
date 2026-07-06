"""Create and apply Project Theseus Hive invite bundles.

Invite bundles are the human-friendly join object for home/workshop machines,
phones, friends/family machines, and later company hives. They intentionally
write secrets only to ignored files or reports.
"""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "hive_policy.json"
DEFAULT_OUT = ROOT / "reports" / "hive_invite_last.json"

sys.path.insert(0, str(ROOT / "scripts"))
import license_manager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    create = sub.add_parser("create")
    create.add_argument("--tier", choices=["private", "friends_family", "company", "public"], default="private")
    create.add_argument("--hive-id", default="")
    create.add_argument("--join-token", default="")
    create.add_argument("--new-token", action="store_true")
    create.add_argument("--relay-url", default="")
    create.add_argument("--coordinator-url", default="")
    create.add_argument("--name", default="")
    create.add_argument("--expires-days", type=int, default=30)
    create.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    create.add_argument("--write-local-config", action="store_true")

    apply = sub.add_parser("apply")
    apply.add_argument("--invite", required=True)
    apply.add_argument("--out", default="reports/hive_join_apply_last.json")
    apply.add_argument("--write-local-config", action="store_true")

    configure = sub.add_parser("configure-local")
    configure.add_argument("--hive-id", default="")
    configure.add_argument("--hive-name", default="")
    configure.add_argument("--tier", default="")
    configure.add_argument("--relay-url", default="")
    configure.add_argument("--coordinator-url", default="")
    configure.add_argument("--join-token", default="")
    configure.add_argument("--out", default="reports/hive_join_configure_last.json")

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command == "create":
        invite = create_invite(policy, args)
        write_json(ROOT / args.out, invite)
        if args.write_local_config and invite.get("ok", True):
            write_join_config(policy, invite)
        print(json.dumps(invite, indent=2))
        return 0 if invite.get("ok", True) else 2
    if args.command == "apply":
        invite = read_json(ROOT / args.invite, {})
        report = apply_invite(policy, invite, write_local=bool(args.write_local_config))
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    if args.command == "configure-local":
        report = configure_local_join(policy, args)
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    parser.print_help()
    return 2


def create_invite(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    license_check = license_manager.check_feature(license_manager.feature_for_hive_tier(args.tier), context={"requested_tier": args.tier})
    if not license_check.get("allowed"):
        return {
            "ok": False,
            "error": "license_required",
            "tier": args.tier,
            "feature": license_manager.feature_for_hive_tier(args.tier),
            "message": license_check.get("next_action") or "Register this install or import a license before creating this invite.",
        }
    active = active_hive_config(policy)
    hive_id = args.hive_id or str(active.get("hive_id") or "") or f"theseus-{secrets.token_hex(6)}"
    relay_url = args.relay_url or str(active.get("relay_url") or "")
    coordinator_url = args.coordinator_url or str(active.get("coordinator_url") or "") or f"http://{local_host_guess()}:{get_path(policy, ['node', 'http_port'], 8791)}"
    if args.join_token:
        token = args.join_token
        token_source = "argument"
    elif not args.new_token and active.get("join_token"):
        token = str(active.get("join_token"))
        token_source = "active_hive"
    else:
        token = secrets.token_urlsafe(32)
        token_source = "new"
    invite_id = f"invite-{secrets.token_hex(8)}"
    expires = datetime.now(timezone.utc) + timedelta(days=max(1, int(args.expires_days)))
    tier_policy = get_path(policy, ["federation", "tiers", args.tier], {})
    return {
        "policy": "project_theseus_hive_invite_v0",
        "created_utc": now(),
        "expires_utc": expires.isoformat(),
        "invite_id": invite_id,
        "hive_id": hive_id,
        "hive_name": args.name or str(active.get("hive_name") or active.get("name") or "") or hive_id,
        "tier": args.tier,
        "relay_url": relay_url,
        "coordinator_url": coordinator_url,
        "join_token": token,
        "join_token_source": token_source,
        "shared_secret_env": get_path(policy, ["security", "shared_secret_env"], "THESEUS_HIVE_SECRET"),
        "hive_id_env": get_path(policy, ["federation", "hive_id_env"], "THESEUS_HIVE_ID"),
        "relay_url_env": get_path(policy, ["federation", "relay_url_env"], "THESEUS_HIVE_RELAY_URL"),
        "allowed_task_scope": get_path(policy, ["relay", "allowed_remote_task_kinds_by_tier", args.tier], []),
        "scopes": {
          "remote_task_kinds": get_path(policy, ["relay", "allowed_remote_task_kinds_by_tier", args.tier], []),
          "can_request_worker_chunks": args.tier in {"private", "company"},
          "can_request_teacher": False,
          "can_request_arbitrary_shell": False
        },
        "quotas": get_path(policy, ["security", "quotas"], {}),
        "revocation_subjects": [
          invite_id,
          hive_id
        ],
        "tier_policy": tier_policy,
        "install": {
            "windows": [
                f"powershell -ExecutionPolicy Bypass -File scripts\\install_theseus_hive.ps1 -Invite INVITE_FILE.json -AutoUpdateSoft -InstallScheduledTask -StartNow",
                f"$env:THESEUS_HIVE_SECRET='{token}'",
                f"$env:THESEUS_HIVE_ID='{hive_id}'",
                f"$env:THESEUS_HIVE_RELAY_URL='{relay_url}'",
                "powershell -ExecutionPolicy Bypass -File scripts\\start_theseus_hive.ps1"
            ],
            "mac_linux": [
                "./scripts/install_theseus_hive_macos.sh --invite INVITE_FILE.json --coordinator-url " + coordinator_url + " --auto-update-soft --install-service",
                "./scripts/install_theseus_hive_linux.sh --invite INVITE_FILE.json --auto-update-soft --install-service",
                f"export THESEUS_HIVE_SECRET='{token}'",
                f"export THESEUS_HIVE_ID='{hive_id}'",
                f"export THESEUS_HIVE_RELAY_URL='{relay_url}'",
                "./scripts/start_theseus_hive.sh"
            ],
            "phone": [
                f"Open {relay_url}/mobile?hive_id={hive_id}",
                "Use the invite token as the password when prompted."
            ]
        },
        "security_notes": [
            "Treat this invite like a password.",
            "Use private/friends_family for real work. public is design-only until sandbox/reputation/abuse controls exist.",
            "Remote workers can only run registered task kinds, not arbitrary shell."
        ],
    }


def apply_invite(policy: dict[str, Any], invite: dict[str, Any], *, write_local: bool) -> dict[str, Any]:
    if invite.get("policy") == "project_theseus_hive_user_invite_v0" or invite.get("invite_kind") == "operator_user":
        return {
            "ok": False,
            "error": "operator_user_invite_not_node_join_invite",
            "message": "This invite is for a person/phone operator token. Use a machine Hive invite to join a computer as a node.",
        }
    required = ["hive_id", "join_token", "tier"]
    missing = [key for key in required if not invite.get(key)]
    if missing:
        return {"ok": False, "error": "invalid_invite", "missing": missing}
    if invite.get("tier") == "public":
        return {
            "ok": False,
            "error": "public_hive_disabled",
            "message": "Public hive mode is design-only until signed workers, sandboxing, reputation, and abuse controls exist.",
        }
    license_check = license_manager.check_feature(license_manager.feature_for_hive_tier(str(invite.get("tier") or "private")), context={"requested_tier": str(invite.get("tier") or "private")})
    if not license_check.get("allowed"):
        return {
            "ok": False,
            "error": "license_required",
            "tier": invite.get("tier"),
            "feature": license_manager.feature_for_hive_tier(str(invite.get("tier") or "private")),
            "message": license_check.get("next_action") or "Register this install or import a license before applying this invite.",
        }
    if write_local:
        write_join_config(policy, invite)
    return {
        "ok": True,
        "policy": "project_theseus_hive_join_apply_v0",
        "created_utc": now(),
        "hive_id": invite.get("hive_id"),
        "tier": invite.get("tier"),
        "relay_url": invite.get("relay_url"),
        "coordinator_url": invite.get("coordinator_url"),
        "local_config_written": write_local,
        "local_config_path": str((ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json"))).relative_to(ROOT)).replace("\\", "/"),
        "next_command": "powershell -ExecutionPolicy Bypass -File scripts\\start_theseus_hive.ps1",
    }


def write_join_config(policy: dict[str, Any], invite: dict[str, Any]) -> None:
    path = ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json"))
    payload = {
        "policy": "project_theseus_hive_join_config_v0",
        "created_utc": now(),
        "invite_id": invite.get("invite_id"),
        "hive_id": invite.get("hive_id"),
        "hive_name": invite.get("hive_name"),
        "tier": invite.get("tier"),
        "relay_url": invite.get("relay_url"),
        "coordinator_url": invite.get("coordinator_url"),
        "join_token": invite.get("join_token"),
        "allowed_task_scope": invite.get("allowed_task_scope", []),
        "quotas": invite.get("quotas", {}),
    }
    write_json(path, payload)


def active_hive_config(policy: dict[str, Any]) -> dict[str, Any]:
    join_path = ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json"))
    join = read_json(join_path, {})
    profiles_path = ROOT / str(get_path(policy, ["federation", "profiles_path"], "configs/hive_profiles.local.json"))
    profiles = read_json(profiles_path, {})
    active_profile: dict[str, Any] = {}
    if isinstance(profiles, dict):
        active_id = str(profiles.get("active_profile_id") or "")
        rows = profiles.get("profiles") if isinstance(profiles.get("profiles"), list) else []
        for row in rows:
            if isinstance(row, dict) and str(row.get("profile_id") or "") == active_id:
                active_profile = row
                break
    merged: dict[str, Any] = {}
    if isinstance(active_profile, dict):
        merged.update(active_profile)
    if isinstance(join, dict):
        merged.update(join)
    return merged


def configure_local_join(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    path = ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json"))
    current = read_json(path, {})
    payload = {
        **(current if isinstance(current, dict) else {}),
        "policy": "project_theseus_hive_join_config_v0",
        "updated_utc": now(),
    }
    for key in ["hive_id", "hive_name", "tier", "relay_url", "coordinator_url", "join_token"]:
        value = getattr(args, key, "")
        if value:
            payload[key] = value
    payload.setdefault("tier", "private")
    write_json(path, payload)
    return {
        "ok": True,
        "policy": "project_theseus_hive_join_configure_v0",
        "created_utc": now(),
        "local_config_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "hive_id": payload.get("hive_id", ""),
        "relay_url": payload.get("relay_url", ""),
        "coordinator_url": payload.get("coordinator_url", ""),
        "join_token_configured": bool(payload.get("join_token")),
    }


def local_host_guess() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
