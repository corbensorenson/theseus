"""Bounded Hive storage extensions.

Hive storage is intentionally opt-in. A node may expose named read-only roots
such as a NAS mount, Photos folder, project export folder, or Raspberry Pi
sensor dump directory. Remote clients can browse and pull files only from those
configured roots; arbitrary filesystem access remains out of scope.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import platform
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
POLICY_PATH = CONFIGS / "hive_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage bounded Project Theseus Hive storage shares.")
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Show configured shares and mount candidates.")
    status.add_argument("--out", default="reports/hive_storage_status.json")

    index = sub.add_parser("index", help="Write a bounded index for configured shares.")
    index.add_argument("--out", default="reports/hive_storage_index.json")
    index.add_argument("--limit", type=int, default=500)

    add = sub.add_parser("add-share", help="Expose a local folder or mounted NAS path as a read-only Hive share.")
    add.add_argument("--path", required=True)
    add.add_argument("--name", default="")
    add.add_argument("--share-id", default="")
    add.add_argument("--tag", action="append", default=[])
    add.add_argument("--writable", action="store_true")

    remove = sub.add_parser("remove-share", help="Disable a configured Hive storage share.")
    remove.add_argument("share_id")

    browse_cmd = sub.add_parser("browse", help="List a configured share directory.")
    browse_cmd.add_argument("--share-id", required=True)
    browse_cmd.add_argument("--path", default="")
    browse_cmd.add_argument("--limit", type=int, default=200)
    browse_cmd.add_argument("--out", default="")

    read_cmd = sub.add_parser("read", help="Read one configured-share file as base64 JSON.")
    read_cmd.add_argument("--share-id", required=True)
    read_cmd.add_argument("--path", required=True)
    read_cmd.add_argument("--out", default="")

    pull = sub.add_parser("pull", help="Pull a file from a peer's Hive storage share into the local inbox.")
    pull.add_argument("--peer-url", required=True)
    pull.add_argument("--share-id", required=True)
    pull.add_argument("--path", required=True)
    pull.add_argument("--out", default="")

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command in {None, "status"}:
        report = status_report(policy=policy, write_report=False)
        write_json(ROOT / args.out, report)
    elif args.command == "index":
        report = index_report(policy=policy, limit=args.limit, write_report=False)
        write_json(ROOT / args.out, report)
    elif args.command == "add-share":
        report = add_share(policy=policy, path=args.path, name=args.name, share_id=args.share_id, tags=args.tag, writable=bool(args.writable))
    elif args.command == "remove-share":
        report = remove_share(policy=policy, share_id=args.share_id)
    elif args.command == "browse":
        report = browse_share(policy=policy, share_id=args.share_id, rel_path=args.path, limit=args.limit)
        if args.out:
            write_json(ROOT / args.out, report)
    elif args.command == "read":
        report = read_file_payload(policy=policy, share_id=args.share_id, rel_path=args.path)
        if args.out:
            write_json(ROOT / args.out, redacted_payload_for_report(report))
    elif args.command == "pull":
        report = pull_file(policy=policy, peer_url=args.peer_url, share_id=args.share_id, rel_path=args.path, out=args.out)
    else:
        parser.print_help()
        return 2
    print(json.dumps(redacted_payload_for_report(report), indent=2))
    return 0 if report.get("ok", True) else 2


def status_report(*, policy: dict[str, Any] | None = None, write_report: bool = True) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    config = storage_config(policy)
    shares = [public_share(policy, share) for share in configured_shares(policy, config, include_disabled=True)]
    active = [share for share in shares if share.get("enabled") and share.get("accessible")]
    report = {
        "ok": True,
        "policy": "project_theseus_hive_storage_status_v0",
        "created_utc": now(),
        "node": {
            "hostname": socket.gethostname(),
            "system": platform.system(),
        },
        "enabled": bool(get_path(policy, ["storage", "enabled"], True)),
        "share_count": len(active),
        "shares": shares,
        "mount_candidates": mount_candidates(policy),
        "device_extensions": config.get("device_extensions") if isinstance(config.get("device_extensions"), list) else [],
        "limits": storage_limits(policy),
        "security": {
            "default_mode": "read_only",
            "exposes_only_configured_shares": True,
            "symlinks_outside_share_blocked": True,
            "arbitrary_filesystem": False,
        },
        "next_actions": storage_next_actions(active),
    }
    if write_report:
        write_json(ROOT / str(get_path(policy, ["storage", "status_path"], "reports/hive_storage_status.json")), report)
    return report


def operator_summary(policy: dict[str, Any]) -> dict[str, Any]:
    status = status_report(policy=policy, write_report=False)
    shares = [
        {
            "share_id": share.get("share_id"),
            "name": share.get("name"),
            "kind": share.get("kind"),
            "tags": share.get("tags") or [],
            "accessible": share.get("accessible"),
        }
        for share in status.get("shares", [])
        if share.get("enabled")
    ]
    return {
        "policy": "project_theseus_hive_operator_storage_v0",
        "enabled": status.get("enabled"),
        "share_count": status.get("share_count"),
        "shares": shares[:24],
        "limits": status.get("limits"),
        "security": status.get("security"),
    }


def index_report(*, policy: dict[str, Any] | None = None, limit: int = 500, write_report: bool = True) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    per_share_limit = max(1, min(limit, int(get_path(policy, ["storage", "max_index_items"], 500))))
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for share in configured_shares(policy, storage_config(policy)):
        result = browse_share(policy=policy, share_id=str(share.get("share_id") or ""), rel_path="", limit=per_share_limit)
        if not result.get("ok"):
            errors.append({"share_id": share.get("share_id"), "error": result.get("error")})
            continue
        rows.extend(result.get("entries") or [])
    report = {
        "ok": not errors,
        "policy": "project_theseus_hive_storage_index_v0",
        "created_utc": now(),
        "item_count": len(rows),
        "items": rows[:per_share_limit],
        "errors": errors,
    }
    if write_report:
        write_json(ROOT / str(get_path(policy, ["storage", "index_path"], "reports/hive_storage_index.json")), report)
    return report


def add_share(
    *,
    policy: dict[str, Any],
    path: str,
    name: str = "",
    share_id: str = "",
    tags: list[str] | None = None,
    writable: bool = False,
) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return {"ok": False, "error": "share_path_missing_or_not_directory", "path": str(root)}
    blocked = sensitive_path_reason(root)
    if blocked:
        return {"ok": False, "error": "share_path_blocked", "path": str(root), "reason": blocked}
    config = storage_config(policy)
    shares = config.setdefault("shares", [])
    if not isinstance(shares, list):
        shares = []
        config["shares"] = shares
    share_id = safe_id(share_id or name or root.name)
    row = {
        "share_id": share_id,
        "name": name or root.name or share_id,
        "path": str(root),
        "mode": "read_write_dropbox" if writable else "read_only",
        "enabled": True,
        "kind": classify_share_path(root),
        "tags": sorted({safe_tag(tag) for tag in (tags or []) if safe_tag(tag)}),
        "added_utc": now(),
    }
    replaced = False
    for idx, existing in enumerate(shares):
        if isinstance(existing, dict) and existing.get("share_id") == share_id:
            row["added_utc"] = existing.get("added_utc") or row["added_utc"]
            shares[idx] = {**existing, **row}
            replaced = True
            break
    if not replaced:
        shares.append(row)
    write_storage_config(policy, config)
    report = {
        "ok": True,
        "policy": "project_theseus_hive_storage_share_configured_v0",
        "created_utc": now(),
        "share": public_share(policy, row),
        "replaced": replaced,
        "security_note": "Only this named root is exposed, and it is read-only unless writable mode was explicitly requested.",
    }
    write_json(REPORTS / "hive_storage_configure_last.json", report)
    return report


def remove_share(*, policy: dict[str, Any], share_id: str) -> dict[str, Any]:
    config = storage_config(policy)
    shares = config.setdefault("shares", [])
    found = False
    for share in shares if isinstance(shares, list) else []:
        if isinstance(share, dict) and str(share.get("share_id") or "") == share_id:
            share["enabled"] = False
            share["disabled_utc"] = now()
            found = True
    if found:
        write_storage_config(policy, config)
    return {"ok": found, "policy": "project_theseus_hive_storage_share_removed_v0", "created_utc": now(), "share_id": share_id, "error": "" if found else "share_not_found"}


def browse_share(*, policy: dict[str, Any], share_id: str, rel_path: str = "", limit: int = 200) -> dict[str, Any]:
    share = share_by_id(policy, share_id)
    if not share:
        return {"ok": False, "error": "share_not_found", "share_id": share_id}
    target = resolve_share_path(share, rel_path)
    if not target:
        return {"ok": False, "error": "path_not_allowed", "share_id": share_id, "path": rel_path}
    if not target.exists():
        return {"ok": False, "error": "path_missing", "share_id": share_id, "path": rel_path}
    if not target.is_dir():
        return {"ok": False, "error": "path_not_directory", "share_id": share_id, "path": rel_path}
    max_items = max(1, min(limit, int(get_path(policy, ["storage", "max_browse_items"], 500))))
    entries = []
    try:
        children = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError as exc:
        return {"ok": False, "error": "browse_failed", "message": str(exc), "share_id": share_id, "path": rel_path}
    for child in children:
        if len(entries) >= max_items:
            break
        if not include_path(policy, child):
            continue
        meta = file_metadata(policy, share, child)
        if meta:
            entries.append(meta)
    return {
        "ok": True,
        "policy": "project_theseus_hive_storage_browse_v0",
        "created_utc": now(),
        "share": public_share(policy, share),
        "path": clean_rel_path(rel_path),
        "entry_count": len(entries),
        "entries": entries,
    }


def read_file_payload(*, policy: dict[str, Any], share_id: str, rel_path: str) -> dict[str, Any]:
    file_result = read_file_bytes(policy=policy, share_id=share_id, rel_path=rel_path)
    if not file_result.get("ok"):
        return file_result
    data = file_result.pop("content_bytes")
    return {
        **file_result,
        "encoding": "base64",
        "content_b64": base64.b64encode(data).decode("ascii"),
    }


def read_file_bytes(*, policy: dict[str, Any], share_id: str, rel_path: str) -> dict[str, Any]:
    share = share_by_id(policy, share_id)
    if not share:
        return {"ok": False, "error": "share_not_found", "share_id": share_id}
    target = resolve_share_path(share, rel_path)
    if not target:
        return {"ok": False, "error": "path_not_allowed", "share_id": share_id, "path": rel_path}
    if not target.exists() or not target.is_file():
        return {"ok": False, "error": "file_missing", "share_id": share_id, "path": rel_path}
    if not include_path(policy, target):
        return {"ok": False, "error": "file_hidden_or_blocked", "share_id": share_id, "path": rel_path}
    try:
        stat = target.stat()
    except OSError as exc:
        return {"ok": False, "error": "stat_failed", "message": str(exc), "share_id": share_id, "path": rel_path}
    max_bytes = int(get_path(policy, ["storage", "max_file_bytes"], 104_857_600))
    if stat.st_size > max_bytes:
        return {"ok": False, "error": "file_too_large", "size_bytes": stat.st_size, "max_bytes": max_bytes, "share_id": share_id, "path": rel_path}
    try:
        data = target.read_bytes()
    except OSError as exc:
        return {"ok": False, "error": "read_failed", "message": str(exc), "share_id": share_id, "path": rel_path}
    return {
        "ok": True,
        "policy": "project_theseus_hive_storage_file_v0",
        "created_utc": now(),
        "share_id": share_id,
        "path": share_rel_path(share, target),
        "name": target.name,
        "kind": classify_file(target),
        "content_type": mimetypes.guess_type(target.name)[0] or "application/octet-stream",
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "content_bytes": data,
    }


def pull_file(*, policy: dict[str, Any], peer_url: str, share_id: str, rel_path: str, out: str = "") -> dict[str, Any]:
    headers = {}
    secret = hive_secret(policy)
    if secret:
        headers["X-Theseus-Hive-Secret"] = secret
    url = peer_url.rstrip("/") + "/api/hive/storage/file?" + urlencode({"share_id": share_id, "path": rel_path})
    payload = fetch_json(url, headers=headers)
    if not payload.get("ok"):
        return {"ok": False, "error": "peer_storage_file_failed", "peer_url": peer_url, "detail": payload}
    try:
        data = base64.b64decode(str(payload.get("content_b64") or ""), validate=True)
    except Exception as exc:  # noqa: BLE001 - remote payload should be ledgered.
        return {"ok": False, "error": "base64_decode_failed", "message": str(exc)}
    actual_sha = hashlib.sha256(data).hexdigest()
    if payload.get("sha256") and actual_sha != payload.get("sha256"):
        return {"ok": False, "error": "sha256_mismatch", "expected": payload.get("sha256"), "actual": actual_sha}
    dest = resolve_pull_destination(policy, out, peer_url, share_id, str(payload.get("path") or rel_path))
    if not dest:
        return {"ok": False, "error": "destination_not_allowed", "out": out}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    report = {
        "ok": True,
        "policy": "project_theseus_hive_storage_pull_v0",
        "created_utc": now(),
        "peer_url": peer_url,
        "share_id": share_id,
        "remote_path": payload.get("path") or rel_path,
        "local_path": rel_to_root(dest),
        "size_bytes": len(data),
        "sha256": actual_sha,
    }
    append_jsonl(ROOT / str(get_path(policy, ["storage", "transfer_ledger_path"], "reports/hive_storage_transfer_ledger.jsonl")), report)
    write_json(REPORTS / "hive_storage_pull_last.json", report)
    return report


def storage_config(policy: dict[str, Any]) -> dict[str, Any]:
    path = storage_config_path(policy)
    if not path.exists():
        return {"policy": "project_theseus_hive_storage_local_v0", "shares": [], "device_extensions": []}
    value = read_json(path, {})
    if not isinstance(value, dict):
        return {"policy": "project_theseus_hive_storage_local_v0", "shares": [], "device_extensions": []}
    value.setdefault("policy", "project_theseus_hive_storage_local_v0")
    value.setdefault("shares", [])
    value.setdefault("device_extensions", [])
    return value


def write_storage_config(policy: dict[str, Any], config: dict[str, Any]) -> None:
    path = storage_config_path(policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    config["updated_utc"] = now()
    write_json(path, config)


def storage_config_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["storage", "config_path"], "configs/hive_storage.local.json"))


def configured_shares(policy: dict[str, Any], config: dict[str, Any], *, include_disabled: bool = False) -> list[dict[str, Any]]:
    if not get_path(policy, ["storage", "enabled"], True):
        return []
    rows = []
    for share in config.get("shares") or []:
        if not isinstance(share, dict):
            continue
        if not include_disabled and share.get("enabled") is False:
            continue
        if not share.get("share_id") or not share.get("path"):
            continue
        rows.append(share)
    return rows


def share_by_id(policy: dict[str, Any], share_id: str) -> dict[str, Any] | None:
    for share in configured_shares(policy, storage_config(policy)):
        if str(share.get("share_id") or "") == share_id:
            return share
    return None


def public_share(policy: dict[str, Any], share: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(share.get("path") or "")).expanduser()
    try:
        resolved = root.resolve()
    except OSError:
        resolved = root
    accessible = resolved.exists() and resolved.is_dir()
    blocked = sensitive_path_reason(resolved)
    return {
        "share_id": share.get("share_id"),
        "name": share.get("name") or share.get("share_id"),
        "kind": share.get("kind") or classify_share_path(resolved),
        "mode": share.get("mode") or "read_only",
        "enabled": share.get("enabled") is not False,
        "accessible": accessible and not blocked,
        "path": str(resolved),
        "blocked_reason": blocked,
        "tags": share.get("tags") if isinstance(share.get("tags"), list) else [],
        "added_utc": share.get("added_utc"),
    }


def resolve_share_path(share: dict[str, Any], rel_path: str) -> Path | None:
    root = Path(str(share.get("path") or "")).expanduser().resolve()
    if sensitive_path_reason(root):
        return None
    rel = clean_rel_path(rel_path)
    try:
        target = (root / rel).resolve() if rel else root
    except OSError:
        return None
    if not path_is_relative_to(target, root):
        return None
    return target


def clean_rel_path(value: str) -> str:
    text = str(value or "").replace("\\", "/").strip("/")
    parts = [part for part in text.split("/") if part and part not in {".", ".."}]
    return "/".join(parts)


def share_rel_path(share: dict[str, Any], path: Path) -> str:
    root = Path(str(share.get("path") or "")).expanduser().resolve()
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


def file_metadata(policy: dict[str, Any], share: dict[str, Any], path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    rel_path = share_rel_path(share, path)
    is_dir = path.is_dir()
    return {
        "share_id": share.get("share_id"),
        "path": rel_path,
        "name": path.name,
        "is_dir": is_dir,
        "kind": "directory" if is_dir else classify_file(path),
        "size_bytes": None if is_dir else stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "content_type": "" if is_dir else (mimetypes.guess_type(path.name)[0] or "application/octet-stream"),
        "downloadable": (not is_dir) and stat.st_size <= int(get_path(policy, ["storage", "max_file_bytes"], 104_857_600)),
    }


def include_path(policy: dict[str, Any], path: Path) -> bool:
    if get_path(policy, ["storage", "include_hidden"], False):
        return True
    return not any(part.startswith(".") for part in path.parts[-3:] if part)


def sensitive_path_reason(path: Path) -> str:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser()
    home = Path.home().resolve()
    system = platform.system()
    if str(resolved) == str(home):
        return "whole_home_directory_blocked"
    if str(resolved) in {"/", "/Users", "/System", "/Library", "/Applications"}:
        return "system_root_blocked"
    if system == "Windows" and str(resolved).rstrip("\\/").endswith(":"):
        return "drive_root_blocked"
    lower_parts = [part.lower() for part in resolved.parts]
    joined = "/".join(lower_parts)
    sensitive_markers = {
        ".ssh",
        ".gnupg",
        ".aws",
        ".azure",
        ".kube",
        ".docker",
        "keychains",
        "cookies",
        "login data",
    }
    if any(part in sensitive_markers for part in lower_parts):
        return "credential_or_profile_directory_blocked"
    browser_markers = [
        "library/application support/google/chrome",
        "library/application support/firefox",
        "library/application support/brave",
        "appdata/local/google/chrome",
        "appdata/roaming/mozilla/firefox",
    ]
    if any(marker in joined for marker in browser_markers):
        return "browser_profile_directory_blocked"
    return ""


def mount_candidates(policy: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    system = platform.system()
    home = Path.home()
    for name in ["Pictures", "Documents", "Downloads", "Desktop"]:
        candidate = home / name
        if candidate.exists() and candidate.is_dir():
            candidates.append(candidate)
    if system == "Darwin":
        volumes = Path("/Volumes")
        if volumes.exists():
            candidates.extend(path for path in volumes.iterdir() if path.is_dir() and path.name != "Macintosh HD")
    elif system == "Windows":
        for drive in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            candidate = Path(f"{drive}:/")
            if candidate.exists():
                candidates.append(candidate)
    else:
        for base in [Path("/mnt"), Path("/media"), Path("/srv")]:
            if base.exists():
                candidates.extend(path for path in base.iterdir() if path.is_dir())
    seen: set[str] = set()
    rows = []
    for path in candidates:
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.append(
            {
                "name": path.name or resolved,
                "path": resolved,
                "kind": classify_share_path(path),
                "suggested_share_id": safe_id(path.name or resolved),
                "configured": any(str(share.get("path") or "") == resolved for share in configured_shares(policy, storage_config(policy), include_disabled=True)),
            }
        )
    return rows[: int(get_path(policy, ["storage", "max_mount_candidates"], 64))]


def classify_share_path(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    if "/volumes/" in text or "/mnt/" in text or "/media/" in text or re.match(r"^[a-z]:/$", text):
        return "mounted_volume_or_nas"
    if "picture" in text or "photo" in text:
        return "photos"
    if "sensor" in text or "camera" in text:
        return "sensor_data"
    return "folder"


def classify_file(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or ""
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("audio/"):
        return "audio"
    if mime in {"application/pdf"}:
        return "document"
    if path.suffix.lower() in {".json", ".jsonl", ".txt", ".md", ".csv", ".tsv", ".log"}:
        return "text"
    return "file"


def resolve_pull_destination(policy: dict[str, Any], out: str, peer_url: str, share_id: str, rel_path: str) -> Path | None:
    if out:
        dest = (ROOT / out).expanduser().resolve() if not Path(out).expanduser().is_absolute() else Path(out).expanduser().resolve()
        allowed_root = ROOT / str(get_path(policy, ["storage", "inbox_path"], "reports/hive_storage_inbox"))
        if not path_is_relative_to(dest, ROOT.resolve()) and not path_is_relative_to(dest, allowed_root.resolve()):
            return None
        return dest
    inbox = ROOT / str(get_path(policy, ["storage", "inbox_path"], "reports/hive_storage_inbox"))
    peer = safe_id(peer_url.replace("://", "_"))
    return (inbox / peer / safe_id(share_id) / clean_rel_path(rel_path)).resolve()


def storage_limits(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_browse_items": int(get_path(policy, ["storage", "max_browse_items"], 500)),
        "max_index_items": int(get_path(policy, ["storage", "max_index_items"], 500)),
        "max_file_bytes": int(get_path(policy, ["storage", "max_file_bytes"], 104_857_600)),
    }


def storage_next_actions(active_shares: list[dict[str, Any]]) -> list[str]:
    if not active_shares:
        return [
            "Add an explicit read-only share, for example: theseus storage add-share --path /Volumes/NAS/Photos --name Photos --tag photos",
            "Use WireGuard/private tunnel or a protected Hive relay path before browsing shares from outside the LAN.",
        ]
    return [
        "Browse locally: theseus storage browse --share-id SHARE",
        "Pull from a peer: theseus storage pull --peer-url http://NODE:8791 --share-id SHARE --path relative/file.jpg",
    ]


def hive_secret(policy: dict[str, Any]) -> str:
    join = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    return str(join.get("join_token") or os.environ.get(str(get_path(policy, ["security", "shared_secret_env"], "THESEUS_HIVE_SECRET")), ""))


def fetch_json(url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = urlrequest.Request(url, headers=headers or {}, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=30) as response:  # noqa: S310 - user-configured private Hive endpoint.
            raw = response.read().decode("utf-8")
    except URLError as exc:
        return {"ok": False, "error": str(exc), "url": url}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_response", "body": raw[:300], "url": url}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_response", "url": url}


def safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower()).strip(".-")
    return text[:64] or "share"


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip().lower())[:40]


def redacted_payload_for_report(report: dict[str, Any]) -> dict[str, Any]:
    if "content_b64" not in report:
        return report
    return {key: value for key, value in report.items() if key != "content_b64"} | {"content_b64_redacted": True}


def path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
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


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def get_path(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
