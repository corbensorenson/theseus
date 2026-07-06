"""Stage private user-supplied ROM assets into ignored local-ROM storage.

This is an inbox processor for `games/` and other local collection folders. It
never downloads ROMs. It copies or extracts only currently supported GB/GBC/GBA
files into `data/local_roms/<system>/`, deduplicates by SHA-256, and inventories
inactive future assets such as NDS, CHD, and N64 images without activating them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "local_rom_policy.json"
DEFAULT_OUT = ROOT / "reports" / "local_rom_staging_report.json"
DEFAULT_DEST = ROOT / "data" / "local_roms"
DEFAULT_INVENTORY = ROOT / "reports" / "game_asset_inventory.json"

ACTIVE_EXTENSIONS = {".gb", ".gbc", ".gba"}
INACTIVE_EXTENSIONS = {".nds", ".z64", ".n64", ".chd", ".cdi", ".cue", ".iso"}
ARCHIVE_EXTENSIONS = {".zip", ".7z"}
SKIP_PREFIXES = ("._",)
SKIP_NAMES = {".DS_Store", "Thumbs.db"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--dest", default=str(DEFAULT_DEST.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--inventory-out", default=str(DEFAULT_INVENTORY.relative_to(ROOT)))
    parser.add_argument("--max-archives", type=int, default=64)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    source_roots = configured_roots(policy, args.source_root)
    dest = resolve_path(args.dest)
    active_exts = {str(item).lower() for item in policy.get("allowed_extensions") or ACTIVE_EXTENSIONS}
    active_exts &= ACTIVE_EXTENSIONS
    existing_hashes = existing_active_hashes(dest)
    staged: list[dict[str, Any]] = []
    inactive: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for root in source_roots:
        if not root.exists():
            skipped.append({"path": safe_path(root), "reason": "source_root_missing"})
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or should_skip(path):
                continue
            suffix = path.suffix.lower()
            if suffix in active_exts:
                staged.append(stage_file(path, root, dest, existing_hashes, execute=args.execute, source_archive=""))
            elif suffix in ARCHIVE_EXTENSIONS and len(archives) < args.max_archives:
                result = inspect_archive(path, root, dest, active_exts, existing_hashes, execute=args.execute)
                archives.append(result["archive"])
                staged.extend(result["staged"])
                inactive.extend(result["inactive"])
                skipped.extend(result["skipped"])
            elif suffix in INACTIVE_EXTENSIONS:
                inactive.append(asset_record(path, root, "inactive_future_asset"))
            else:
                skipped.append({"path": safe_path(path), "reason": f"unsupported_extension:{suffix or '<none>'}"})

    report = {
        "policy": "sparkstream_local_rom_asset_staging_v0",
        "created_utc": now(),
        "execute": args.execute,
        "source_policy": str(Path(args.policy).as_posix()),
        "source_roots": [root_record(root) for root in source_roots],
        "destination": safe_path(dest),
        "active_extensions": sorted(active_exts),
        "archive_extensions": sorted(ARCHIVE_EXTENSIONS),
        "staged": staged,
        "archives": archives,
        "inactive_assets": inactive,
        "skipped": skipped[:200],
        "summary": {
            "staged_count": len([row for row in staged if row.get("status") == "staged"]),
            "already_present_count": len([row for row in staged if row.get("status") == "already_present"]),
            "active_rom_records": len(staged),
            "unique_active_roms": len({str(row.get("sha256")) for row in staged if row.get("sha256")}),
            "archive_count": len(archives),
            "inactive_asset_count": len(inactive),
            "skip_count": len(skipped),
            "execute_required_for_changes": not args.execute,
            "autonomous_downloads": 0,
        },
        "next_actions": next_actions(staged, inactive, args.execute),
    }
    inventory = {
        "policy": "sparkstream_game_asset_inventory_v0",
        "created_utc": now(),
        "active_roms": staged,
        "inactive_assets": inactive,
        "archives": archives,
        "summary": report["summary"],
    }
    write_json(resolve_path(args.out), report)
    write_json(resolve_path(args.inventory_out), inventory)
    print(json.dumps(report, indent=2))
    return 0


def configured_roots(policy: dict[str, Any], cli_roots: list[str]) -> list[Path]:
    roots: list[Path] = []
    for item in policy.get("default_roots") or []:
        roots.append(resolve_path(str(item)))
    for item in cli_roots:
        roots.append(resolve_path(item))
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def inspect_archive(
    archive: Path,
    root: Path,
    dest: Path,
    active_exts: set[str],
    existing_hashes: set[str],
    *,
    execute: bool,
) -> dict[str, Any]:
    archive_row = asset_record(archive, root, "archive")
    staged: list[dict[str, Any]] = []
    inactive: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    list_result = list_archive(archive)
    archive_row["list_ok"] = list_result["ok"]
    archive_row["list_tool"] = list_result["tool"]
    archive_row["members"] = list_result["members"][:80]
    if not list_result["ok"]:
        archive_row["error"] = list_result.get("error")
        return {"archive": archive_row, "staged": staged, "inactive": inactive, "skipped": skipped}

    wanted_members = [
        member
        for member in list_result["members"]
        if Path(member).suffix.lower() in (active_exts | INACTIVE_EXTENSIONS)
        and not should_skip(Path(member))
    ]
    archive_row["wanted_members"] = wanted_members
    active_members = [member for member in wanted_members if Path(member).suffix.lower() in active_exts]
    if not execute or not wanted_members or not active_members:
        for member in wanted_members:
            suffix = Path(member).suffix.lower()
            row = {
                "path": safe_path(archive),
                "member": member,
                "extension": suffix,
                "status": "planned_extract" if suffix in active_exts else "inactive_archive_member",
            }
            if suffix in active_exts:
                staged.append(row)
            else:
                inactive.append(row)
        return {"archive": archive_row, "staged": staged, "inactive": inactive, "skipped": skipped}

    with tempfile.TemporaryDirectory(prefix="sparkstream_rom_extract_") as temp:
        temp_path = Path(temp)
        extract_result = extract_archive(archive, temp_path)
        archive_row["extract_ok"] = extract_result["ok"]
        archive_row["extract_tool"] = extract_result["tool"]
        if not extract_result["ok"]:
            archive_row["error"] = extract_result.get("error")
            return {"archive": archive_row, "staged": staged, "inactive": inactive, "skipped": skipped}
        for extracted in sorted(temp_path.rglob("*")):
            if not extracted.is_file() or should_skip(extracted):
                continue
            suffix = extracted.suffix.lower()
            if suffix in active_exts:
                staged.append(
                    stage_file(
                        extracted,
                        temp_path,
                        dest,
                        existing_hashes,
                        execute=True,
                        source_archive=safe_path(archive),
                    )
                )
            elif suffix in INACTIVE_EXTENSIONS:
                row = asset_record(extracted, temp_path, "inactive_extracted_member")
                row["source_archive"] = safe_path(archive)
                inactive.append(row)
    return {"archive": archive_row, "staged": staged, "inactive": inactive, "skipped": skipped}


def list_archive(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".zip":
        try:
            import zipfile

            with zipfile.ZipFile(path) as archive:
                return {"ok": True, "tool": "python_zipfile", "members": archive.namelist()}
        except Exception as exc:  # pragma: no cover - diagnostic path
            return {"ok": False, "tool": "python_zipfile", "members": [], "error": str(exc)}
    if path.suffix.lower() == ".7z":
        try:
            import py7zr  # type: ignore

            with py7zr.SevenZipFile(path, mode="r") as archive:
                return {"ok": True, "tool": "py7zr", "members": archive.getnames()}
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover - diagnostic path
            return {"ok": False, "tool": "py7zr", "members": [], "error": str(exc)}
    return run_archive_tool(["tar", "-tf", str(path)], tool="tar")


def extract_archive(path: Path, dest: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".zip":
        try:
            import zipfile

            with zipfile.ZipFile(path) as archive:
                archive.extractall(dest)
            return {"ok": True, "tool": "python_zipfile"}
        except Exception as exc:  # pragma: no cover - diagnostic path
            return {"ok": False, "tool": "python_zipfile", "error": str(exc)}
    if path.suffix.lower() == ".7z":
        try:
            import py7zr  # type: ignore

            with py7zr.SevenZipFile(path, mode="r") as archive:
                archive.extractall(dest)
            return {"ok": True, "tool": "py7zr"}
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover - diagnostic path
            return {"ok": False, "tool": "py7zr", "error": str(exc)}
    return run_archive_tool(["tar", "-xf", str(path), "-C", str(dest)], tool="tar")


def run_archive_tool(command: list[str], *, tool: str) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"ok": False, "tool": tool, "members": [], "error": str(exc)}
    members = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "ok": result.returncode == 0,
        "tool": tool,
        "members": members,
        "error": result.stderr.strip() if result.returncode != 0 else "",
    }


def stage_file(
    path: Path,
    root: Path,
    dest: Path,
    existing_hashes: set[str],
    *,
    execute: bool,
    source_archive: str,
) -> dict[str, Any]:
    digest = sha256_file(path)
    system = system_for_suffix(path.suffix)
    target = unique_target(dest / system, path.name, digest)
    row = asset_record(path, root, "staged" if execute else "planned_stage")
    row.update(
        {
            "system": system,
            "sha256": digest,
            "target": safe_path(target),
            "source_archive": source_archive,
        }
    )
    if digest in existing_hashes:
        row["status"] = "already_present"
        return row
    if execute:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        existing_hashes.add(digest)
        row["bytes_written"] = target.stat().st_size
    return row


def existing_active_hashes(dest: Path) -> set[str]:
    hashes: set[str] = set()
    if not dest.exists():
        return hashes
    for path in dest.rglob("*"):
        if path.is_file() and path.suffix.lower() in ACTIVE_EXTENSIONS:
            hashes.add(sha256_file(path))
    return hashes


def asset_record(path: Path, root: Path, status: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    return {
        "display_name": path.name,
        "extension": suffix,
        "path": safe_path(path),
        "root": safe_path(root),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "system": system_for_suffix(suffix),
        "status": status,
    }


def next_actions(staged: list[dict[str, Any]], inactive: list[dict[str, Any]], execute: bool) -> list[str]:
    active = [row for row in staged if row.get("status") in {"staged", "already_present", "planned_stage"}]
    actions: list[str] = []
    if not execute:
        actions.append("Rerun with --execute to copy/extract active GB/GBC/GBA assets into data/local_roms.")
    if any("emerald" in str(row.get("display_name", "")).lower() for row in active):
        actions.append("Pokemon Emerald is available; prioritize PyGBA/mGBA wrapper smoke for the first GBA RL benchmark.")
    if any("firered" in str(row.get("display_name", "")).lower() for row in active):
        actions.append("Pokemon FireRed is available as a second GBA progression/navigation benchmark candidate.")
    if inactive:
        actions.append("Inactive NDS/N64/CHD assets are inventoried only; add adapters later before using them as frontiers.")
    actions.append("Keep ROM/game files ignored; only wrapper code, manifests, and smoke reports should become tracked.")
    return actions


def unique_target(directory: Path, name: str, digest: str) -> Path:
    safe = sanitize_name(name)
    target = directory / safe
    if not target.exists():
        return target
    return directory / f"{Path(safe).stem}.{digest[:8]}{Path(safe).suffix}"


def sanitize_name(name: str) -> str:
    return "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in name).strip() or "rom"


def should_skip(path: Path) -> bool:
    name = path.name
    return name in SKIP_NAMES or name.startswith(SKIP_PREFIXES)


def root_record(root: Path) -> dict[str, Any]:
    return {"path": safe_path(root), "exists": root.exists()}


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def safe_path(path: Path) -> str:
    try:
        resolved = path.resolve()
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return f"<external>/{path.name}"


def system_for_suffix(suffix: str) -> str:
    text = suffix.lower()
    if text == ".gba":
        return "gba"
    if text == ".gbc":
        return "gbc"
    if text == ".gb":
        return "gb"
    if text == ".nds":
        return "nds"
    if text in {".z64", ".n64"}:
        return "n64"
    if text in {".chd", ".cdi", ".cue", ".iso"}:
        return "disc_image"
    return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
