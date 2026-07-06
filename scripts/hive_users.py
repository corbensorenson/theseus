"""Multi-user access control for Project Theseus Hive.

This module intentionally separates people/operator access from machine join
secrets. Node-to-node trust can keep using the Hive join token, while phones
and family laptops can use per-user operator tokens with narrower scopes.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "hive_policy.json"
DEFAULT_USERS_PATH = ROOT / "configs" / "hive_users.local.json"
REPORTS = ROOT / "reports"

DEFAULT_ROLES: dict[str, dict[str, Any]] = {
    "owner": {
        "label": "Owner",
        "remote_task_kinds": ["*"],
        "can_view_status": True,
        "can_view_storage": True,
        "can_download_files": True,
        "can_remote_control": True,
        "can_request_worker_chunks": True,
        "can_manage_updates": True,
        "can_manage_users": True,
        "can_voice_control": True,
        "can_sync_artifacts": True,
    },
    "operator": {
        "label": "Operator",
        "remote_task_kinds": [
            "resource_probe",
            "capability_refresh",
            "readiness_check",
            "checkpoint_chat",
            "compute_market_status",
            "storage_status",
            "storage_index",
            "voice_following_status",
            "hive_version_status",
            "training_orchestrate",
            "training_smoke",
            "cuda_eval_chunk",
            "mlx_eval_chunk",
        ],
        "can_view_status": True,
        "can_view_storage": True,
        "can_download_files": True,
        "can_remote_control": True,
        "can_request_worker_chunks": True,
        "can_manage_updates": False,
        "can_manage_users": False,
        "can_voice_control": True,
        "can_sync_artifacts": True,
    },
    "member": {
        "label": "Family Member",
        "remote_task_kinds": [
            "resource_probe",
            "readiness_check",
            "checkpoint_chat",
            "storage_status",
            "storage_index",
            "voice_following_status",
        ],
        "can_view_status": True,
        "can_view_storage": True,
        "can_download_files": True,
        "can_remote_control": False,
        "can_request_worker_chunks": False,
        "can_manage_updates": False,
        "can_manage_users": False,
        "can_voice_control": True,
        "can_sync_artifacts": False,
    },
    "child": {
        "label": "Child",
        "remote_task_kinds": [
            "resource_probe",
            "readiness_check",
            "checkpoint_chat",
            "voice_following_status",
        ],
        "can_view_status": True,
        "can_view_storage": False,
        "can_download_files": False,
        "can_remote_control": False,
        "can_request_worker_chunks": False,
        "can_manage_updates": False,
        "can_manage_users": False,
        "can_voice_control": True,
        "can_sync_artifacts": False,
    },
    "guest": {
        "label": "Guest",
        "remote_task_kinds": [
            "resource_probe",
            "checkpoint_chat",
        ],
        "can_view_status": True,
        "can_view_storage": False,
        "can_download_files": False,
        "can_remote_control": False,
        "can_request_worker_chunks": False,
        "can_manage_updates": False,
        "can_manage_users": False,
        "can_voice_control": False,
        "can_sync_artifacts": False,
    },
}

WORKER_TASK_PREFIXES = ("cuda_", "mlx_")
WORKER_TASK_KINDS = {"training_smoke", "training_orchestrate"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    add = sub.add_parser("add")
    add.add_argument("--user-id", default="")
    add.add_argument("--name", required=True)
    add.add_argument("--role", choices=sorted(DEFAULT_ROLES), default="member")
    add.add_argument("--device-label", default="")
    add.add_argument("--expires-days", type=int, default=0)
    add.add_argument("--token", default="")
    add.add_argument("--replace", action="store_true")
    add.add_argument("--out", default="")

    sub.add_parser("list")

    revoke = sub.add_parser("revoke")
    revoke.add_argument("user_id")
    revoke.add_argument("--out", default="reports/hive_user_revoke_last.json")

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command == "add":
        report = create_user(policy, args)
        if args.out:
            write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    if args.command == "list":
        report = list_users(policy)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    if args.command == "revoke":
        report = revoke_user(policy, args.user_id)
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    parser.print_help()
    return 2


def create_user(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = load_user_config(policy)
    role = str(args.role or "member")
    roles = roles_config(policy)
    if role not in roles:
        return {"ok": False, "error": "unknown_role", "role": role, "available_roles": sorted(roles)}
    user_id = sanitize_id(args.user_id or args.name)
    users = config.setdefault("users", [])
    existing = next((row for row in users if isinstance(row, dict) and row.get("user_id") == user_id), None)
    if existing and not args.replace:
        return {"ok": False, "error": "user_exists", "user_id": user_id, "next_action": "rerun with --replace to rotate this user's token"}

    token = str(args.token or secrets.token_urlsafe(32))
    expires_utc = ""
    if int(args.expires_days or 0) > 0:
        expires_utc = (datetime.now(timezone.utc) + timedelta(days=int(args.expires_days))).isoformat()
    row = {
        "user_id": user_id,
        "display_name": str(args.name),
        "role": role,
        "enabled": True,
        "token_sha256": token_digest(token),
        "token_preview": token_preview(token),
        "device_labels": unique_nonempty([str(args.device_label or "")]),
        "created_utc": existing.get("created_utc") if isinstance(existing, dict) else now(),
        "updated_utc": now(),
        "expires_utc": expires_utc,
        "scopes": {},
    }
    if existing:
        users[users.index(existing)] = row
    else:
        users.append(row)
    config["updated_utc"] = now()
    write_user_config(policy, config)

    invite = user_invite(policy, row, token)
    invite_path = REPORTS / f"hive_user_invite_{user_id}.json"
    write_json(invite_path, invite)
    return {
        "ok": True,
        "policy": "project_theseus_hive_user_create_v0",
        "created_utc": now(),
        "user": public_user(row),
        "operator_token": token,
        "token_preview": token_preview(token),
        "invite_path": str(invite_path.relative_to(ROOT)).replace("\\", "/"),
        "invite": invite,
        "next_actions": [
            "Import the invite JSON in the iPhone app, or paste the operator token into /mobile.",
            "Use `theseus hive users` to inspect users and `theseus hive revoke-user USER_ID` to disable access.",
        ],
    }


def list_users(policy: dict[str, Any]) -> dict[str, Any]:
    config = load_user_config(policy)
    users = [public_user(row) for row in config.get("users", []) if isinstance(row, dict)]
    return {
        "ok": True,
        "policy": "project_theseus_hive_users_list_v0",
        "created_utc": now(),
        "config_path": str(user_config_path(policy).relative_to(ROOT)).replace("\\", "/"),
        "user_count": len(users),
        "roles": public_roles(policy),
        "users": users,
    }


def revoke_user(policy: dict[str, Any], user_id: str) -> dict[str, Any]:
    config = load_user_config(policy)
    target = sanitize_id(user_id)
    for row in config.get("users", []):
        if isinstance(row, dict) and row.get("user_id") == target:
            row["enabled"] = False
            row["revoked_utc"] = now()
            config["updated_utc"] = now()
            write_user_config(policy, config)
            return {"ok": True, "policy": "project_theseus_hive_user_revoke_v0", "created_utc": now(), "user": public_user(row)}
    return {"ok": False, "error": "user_not_found", "user_id": target}


def authenticate(policy: dict[str, Any], client_host: str, provided_token: str = "", query: str = "", *, allow_loopback: bool = True) -> dict[str, Any]:
    if allow_loopback and is_loopback(client_host):
        return auth_context(policy, "local", "Local User", "owner", "loopback", "loopback")
    token = token_from_request(provided_token, query)
    if not token:
        return {"ok": False, "error": "operator_token_required"}

    for expected in legacy_hive_tokens(policy):
        if expected and hmac.compare_digest(token, expected):
            return auth_context(policy, "legacy-owner", "Hive Owner", "owner", "legacy_hive_secret", "secret_ok")

    digest = token_digest(token)
    for row in load_user_config(policy).get("users", []):
        if not isinstance(row, dict):
            continue
        if not row.get("enabled", True):
            continue
        if row.get("expires_utc") and is_expired(str(row.get("expires_utc") or "")):
            continue
        if hmac.compare_digest(str(row.get("token_sha256") or ""), digest):
            ctx = auth_context(
                policy,
                str(row.get("user_id") or ""),
                str(row.get("display_name") or row.get("user_id") or "Hive User"),
                str(row.get("role") or "member"),
                "user_token",
                "user_token_ok",
            )
            if isinstance(row.get("scopes"), dict):
                ctx["scopes"] = merge_scopes(ctx.get("scopes") if isinstance(ctx.get("scopes"), dict) else {}, row["scopes"])
            ctx["device_labels"] = row.get("device_labels") if isinstance(row.get("device_labels"), list) else []
            return ctx
    return {"ok": False, "error": "operator_token_rejected"}


def authorize(
    policy: dict[str, Any],
    client_host: str,
    provided_token: str = "",
    query: str = "",
    *,
    action: str = "status",
    task_kind: str = "",
    allow_loopback: bool = True,
) -> dict[str, Any]:
    ctx = authenticate(policy, client_host, provided_token, query, allow_loopback=allow_loopback)
    if not ctx.get("ok"):
        return ctx
    if not action_allowed(ctx, action, task_kind=task_kind):
        ctx = dict(ctx)
        ctx.update({"ok": False, "error": "user_scope_denied", "action": action, "task_kind": task_kind})
    return ctx


def action_allowed(ctx: dict[str, Any], action: str, *, task_kind: str = "") -> bool:
    if ctx.get("token_kind") in {"loopback", "legacy_hive_secret"}:
        return True
    scopes = ctx.get("scopes") if isinstance(ctx.get("scopes"), dict) else {}
    if action in {"status", "operator_status"}:
        return bool(scopes.get("can_view_status", True))
    if action in {"chat"}:
        return task_allowed(ctx, "checkpoint_chat")
    if action in {"task"}:
        return task_allowed(ctx, task_kind)
    if action in {"storage", "storage_file"}:
        if not scopes.get("can_view_storage", False):
            return False
        if action == "storage_file":
            return bool(scopes.get("can_download_files", False))
        return True
    if action == "remote_control":
        return bool(scopes.get("can_remote_control", False))
    if action == "update_apply":
        return bool(scopes.get("can_manage_updates", False))
    if action == "voice_control":
        return bool(scopes.get("can_voice_control", False))
    if action == "artifact_sync":
        return bool(scopes.get("can_sync_artifacts", False))
    if action == "manage_users":
        return bool(scopes.get("can_manage_users", False))
    return bool(scopes.get("can_view_status", False))


def task_allowed(ctx: dict[str, Any], task_kind: str) -> bool:
    if not task_kind:
        return False
    if ctx.get("token_kind") in {"loopback", "legacy_hive_secret"}:
        return True
    scopes = ctx.get("scopes") if isinstance(ctx.get("scopes"), dict) else {}
    allowed = scopes.get("remote_task_kinds") if isinstance(scopes.get("remote_task_kinds"), list) else []
    normalized = {str(item) for item in allowed}
    if "*" not in normalized and task_kind not in normalized:
        return False
    if is_worker_task(task_kind) and not scopes.get("can_request_worker_chunks", False):
        return False
    return True


def filter_task_kinds(ctx: dict[str, Any] | None, task_kinds: list[str]) -> list[str]:
    if not ctx or ctx.get("token_kind") in {"loopback", "legacy_hive_secret"}:
        return sorted(set(task_kinds))
    return sorted({kind for kind in task_kinds if task_allowed(ctx, kind)})


def user_summary(ctx: dict[str, Any] | None) -> dict[str, Any]:
    if not ctx:
        return {"authenticated": False}
    scopes = ctx.get("scopes") if isinstance(ctx.get("scopes"), dict) else {}
    return {
        "authenticated": bool(ctx.get("ok", True)),
        "user_id": ctx.get("user_id"),
        "display_name": ctx.get("display_name"),
        "role": ctx.get("role"),
        "token_kind": ctx.get("token_kind"),
        "reason": ctx.get("reason"),
        "capabilities": {
            "can_view_storage": bool(scopes.get("can_view_storage", False)),
            "can_download_files": bool(scopes.get("can_download_files", False)),
            "can_remote_control": bool(scopes.get("can_remote_control", False)),
            "can_request_worker_chunks": bool(scopes.get("can_request_worker_chunks", False)),
            "can_manage_updates": bool(scopes.get("can_manage_updates", False)),
            "can_manage_users": bool(scopes.get("can_manage_users", False)),
            "can_voice_control": bool(scopes.get("can_voice_control", False)),
            "can_sync_artifacts": bool(scopes.get("can_sync_artifacts", False)),
        },
    }


def storage_share_allowed(ctx: dict[str, Any] | None, share_id: str) -> bool:
    if not ctx or ctx.get("token_kind") in {"loopback", "legacy_hive_secret"}:
        return True
    scopes = ctx.get("scopes") if isinstance(ctx.get("scopes"), dict) else {}
    allowlist = scopes.get("storage_share_allowlist")
    if not isinstance(allowlist, list) or not allowlist:
        return True
    return str(share_id or "") in {str(item) for item in allowlist}


def auth_context(policy: dict[str, Any], user_id: str, display_name: str, role: str, token_kind: str, reason: str) -> dict[str, Any]:
    scopes = role_scopes(policy, role)
    return {
        "ok": True,
        "reason": reason,
        "user_id": user_id,
        "display_name": display_name,
        "role": role,
        "token_kind": token_kind,
        "scopes": scopes,
    }


def role_scopes(policy: dict[str, Any], role: str) -> dict[str, Any]:
    roles = roles_config(policy)
    base = DEFAULT_ROLES.get(role, DEFAULT_ROLES["member"])
    configured = roles.get(role, {}) if isinstance(roles.get(role), dict) else {}
    return merge_scopes(base, configured)


def roles_config(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    configured = get_path(policy, ["multi_user", "roles"], {})
    roles = {key: dict(value) for key, value in DEFAULT_ROLES.items()}
    if isinstance(configured, dict):
        for key, value in configured.items():
            if isinstance(value, dict):
                roles[str(key)] = merge_scopes(roles.get(str(key), {}), value)
    return roles


def public_roles(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        role: {
            "label": scopes.get("label", role),
            "remote_task_kinds": scopes.get("remote_task_kinds", []),
            "can_view_storage": bool(scopes.get("can_view_storage", False)),
            "can_download_files": bool(scopes.get("can_download_files", False)),
            "can_remote_control": bool(scopes.get("can_remote_control", False)),
            "can_request_worker_chunks": bool(scopes.get("can_request_worker_chunks", False)),
            "can_manage_updates": bool(scopes.get("can_manage_updates", False)),
            "can_manage_users": bool(scopes.get("can_manage_users", False)),
            "can_voice_control": bool(scopes.get("can_voice_control", False)),
            "can_sync_artifacts": bool(scopes.get("can_sync_artifacts", False)),
        }
        for role, scopes in roles_config(policy).items()
    }


def merge_scopes(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = value
    return merged


def load_user_config(policy: dict[str, Any]) -> dict[str, Any]:
    path = user_config_path(policy)
    data = read_json(path, {})
    if not isinstance(data, dict) or data.get("policy") != "project_theseus_hive_users_v0":
        data = {
            "policy": "project_theseus_hive_users_v0",
            "created_utc": now(),
            "updated_utc": now(),
            "hive_id": active_hive_id(policy),
            "users": [],
        }
    data.setdefault("users", [])
    return data


def write_user_config(policy: dict[str, Any], payload: dict[str, Any]) -> None:
    write_json(user_config_path(policy), payload)


def user_config_path(policy: dict[str, Any]) -> Path:
    rel = str(get_path(policy, ["multi_user", "users_config_path"], "configs/hive_users.local.json"))
    return ROOT / rel


def user_invite(policy: dict[str, Any], row: dict[str, Any], token: str) -> dict[str, Any]:
    join = active_hive_config(policy)
    hive = active_hive_id(policy)
    coordinator = str(join.get("coordinator_url") or f"http://{local_host_guess()}:{get_path(policy, ['node', 'http_port'], 8791)}")
    relay = str(join.get("relay_url") or "")
    return {
        "policy": "project_theseus_hive_user_invite_v0",
        "invite_kind": "operator_user",
        "created_utc": now(),
        "hive_id": hive,
        "hive_name": join.get("hive_name") or hive,
        "tier": join.get("tier") or get_path(policy, ["federation", "default_tier"], "private"),
        "coordinator_url": coordinator,
        "relay_url": relay,
        "join_token": token,
        "operator_token": token,
        "user": public_user(row),
        "scopes": user_summary(auth_context(policy, str(row.get("user_id") or ""), str(row.get("display_name") or ""), str(row.get("role") or "member"), "user_token", "invite")).get("capabilities"),
        "security_notes": [
            "This is a user/operator invite, not a machine join invite.",
            "Use it on phones or trusted personal devices; revoke the user if a device is lost.",
            "User tokens can only use the role's allowed registered Hive actions.",
        ],
    }


def public_user(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": row.get("user_id"),
        "display_name": row.get("display_name"),
        "role": row.get("role"),
        "enabled": bool(row.get("enabled", True)),
        "token_configured": bool(row.get("token_sha256")),
        "token_preview": row.get("token_preview", ""),
        "device_labels": row.get("device_labels") if isinstance(row.get("device_labels"), list) else [],
        "created_utc": row.get("created_utc"),
        "updated_utc": row.get("updated_utc"),
        "expires_utc": row.get("expires_utc", ""),
        "revoked_utc": row.get("revoked_utc", ""),
    }


def token_from_request(provided_token: str, query: str) -> str:
    token = str(provided_token or "").strip()
    if token:
        return token
    if query:
        parsed = parse_qs(query)
        for key in ("token", "t", "operator_token", "access_token"):
            value = (parsed.get(key) or [""])[0]
            if value:
                return str(value)
    return ""


def legacy_hive_tokens(policy: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    env_name = str(get_path(policy, ["security", "shared_secret_env"], "THESEUS_HIVE_SECRET"))
    tokens.append(os.environ.get(env_name, ""))
    join = active_hive_config(policy)
    tokens.append(str(join.get("join_token") or ""))
    profiles = read_json(ROOT / str(get_path(policy, ["federation", "profiles_path"], "configs/hive_profiles.local.json")), {})
    if isinstance(profiles, dict):
        active = str(profiles.get("active_profile_id") or "")
        rows = profiles.get("profiles") if isinstance(profiles.get("profiles"), list) else []
        for row in rows:
            if isinstance(row, dict) and (not active or row.get("profile_id") == active):
                tokens.append(str(row.get("join_token") or ""))
    return unique_nonempty(tokens)


def active_hive_config(policy: dict[str, Any]) -> dict[str, Any]:
    join_path = ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json"))
    join = read_json(join_path, {})
    return join if isinstance(join, dict) else {}


def active_hive_id(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["federation", "hive_id_env"], "THESEUS_HIVE_ID"))
    return os.environ.get(env_name) or str(active_hive_config(policy).get("hive_id") or "local")


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_preview(token: str) -> str:
    return f"...{token[-8:]}" if token else ""


def is_worker_task(kind: str) -> bool:
    return kind in WORKER_TASK_KINDS or kind.startswith(WORKER_TASK_PREFIXES)


def is_expired(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed < datetime.now(timezone.utc)


def sanitize_id(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    clean = "-".join(part for part in clean.split("-") if part)
    return clean[:48] or f"user-{secrets.token_hex(4)}"


def unique_nonempty(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def local_host_guess() -> str:
    try:
        sock = __import__("socket").socket(__import__("socket").AF_INET, __import__("socket").SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
        finally:
            sock.close()
    except OSError:
        return "127.0.0.1"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_path(obj: Any, path: list[Any], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"} or host.startswith("127.")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
