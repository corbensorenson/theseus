"""Private local ROM inventory for user-supplied RL assets.

This script never downloads ROMs. It inventories ROM files that the user has
placed in configured local directories, writes an ignored report, and recommends
adapter paths that can become RL frontiers after wrapper smoke tests pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "local_rom_policy.json"
DEFAULT_OUT = ROOT / "reports" / "local_rom_registry.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--rom-root", action="append", default=[])
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    roots = configured_roots(policy, args.rom_root)
    rows = scan_roms(roots, policy, max_files=args.max_files)
    recommendations = recommend_profiles(rows, policy)
    report = {
        "policy": "sparkstream_local_rom_registry_v0",
        "created_utc": now(),
        "source_policy": str(Path(args.policy).as_posix()),
        "rights_attestation": policy.get("rights_attestation"),
        "autonomous_rom_download": "forbidden",
        "git_tracking": "rom_files_must_remain_ignored",
        "roots": [root_record(root) for root in roots],
        "roms": rows,
        "recommendations": recommendations,
        "summary": summary(rows, recommendations),
        "next_actions": next_actions(rows, recommendations),
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def configured_roots(policy: dict[str, Any], cli_roots: list[str]) -> list[Path]:
    roots: list[Path] = []
    for item in policy.get("default_roots") or []:
        roots.append(resolve_root(str(item)))
    env_var = str(policy.get("environment_variable") or "SPARKSTREAM_ROM_ROOTS")
    for item in split_env_roots(os.environ.get(env_var, "")):
        roots.append(resolve_root(item))
    for item in cli_roots:
        roots.append(resolve_root(item))
    seen: set[str] = set()
    deduped = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def split_env_roots(value: str) -> list[str]:
    if not value.strip():
        return []
    roots: list[str] = []
    for chunk in value.split(os.pathsep):
        text = chunk.strip().strip('"')
        if text:
            roots.append(text)
    return roots


def resolve_root(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def scan_roms(roots: list[Path], policy: dict[str, Any], *, max_files: int) -> list[dict[str, Any]]:
    suffixes = {str(item).lower() for item in policy.get("allowed_extensions") or [".gb", ".gbc", ".gba"]}
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if len(rows) >= max_files:
                break
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            rows.append(rom_record(path, root))
    return rows


def rom_record(path: Path, root: Path) -> dict[str, Any]:
    system = system_for_suffix(path.suffix)
    return {
        "id": "rom_" + sha256_file(path)[:16],
        "display_name": path.stem,
        "system": system,
        "extension": path.suffix.lower(),
        "path": safe_path(path),
        "root": safe_path(root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": "available_user_supplied_private_rom",
        "training_use": "allowed_after_wrapper_smoke_and_user_rights_attestation",
        "git_tracking": "ignored_required",
        "created_utc": now(),
    }


def recommend_profiles(roms: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for profile in policy.get("priority_rom_profiles") or []:
        matches = matching_roms(roms, profile)
        recommendations.append(
            {
                "profile_id": profile.get("profile_id"),
                "priority": profile.get("priority", "medium"),
                "system": profile.get("system"),
                "adapter": profile.get("adapter"),
                "why": profile.get("why"),
                "matched_rom_count": len(matches),
                "matched_roms": [
                    {
                        "id": row.get("id"),
                        "display_name": row.get("display_name"),
                        "system": row.get("system"),
                        "path": row.get("path"),
                    }
                    for row in matches[:8]
                ],
                "next_step": next_step_for_profile(profile, matches),
            }
        )
    return sorted(
        recommendations,
        key=lambda item: (
            0 if item.get("matched_rom_count") else 1,
            {"high": 0, "medium": 1, "low": 2}.get(str(item.get("priority")), 3),
            str(item.get("profile_id")),
        ),
    )


def matching_roms(roms: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    keywords = [str(item).lower() for item in profile.get("filename_keywords") or []]
    profile_system = str(profile.get("system") or "")
    matches = []
    for row in roms:
        name = str(row.get("display_name") or "").lower().replace("_", " ").replace("-", " ")
        system = str(row.get("system") or "")
        if profile_system == "gb_gbc" and system not in {"gb", "gbc"}:
            continue
        if profile_system not in {"", "gb_gbc"} and system != profile_system:
            continue
        if any(keyword in name for keyword in keywords):
            matches.append(row)
    return matches


def next_step_for_profile(profile: dict[str, Any], matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "No matching local ROM found yet. If you own one, place it under data/local_roms, set SPARKSTREAM_ROM_ROOTS, or pass --rom-root."
    adapter = profile.get("adapter")
    if adapter == "pygba_mgba":
        return "Install/build mGBA Python bindings, run PyGBA smoke, then add shaped reward frontier."
    if adapter in {"pyboy_or_gymboy", "pyboy_or_stable_retro"}:
        return "Run PyBoy/Gymboy smoke with deterministic seed, then define score/progress reward and reset policy."
    return "Create a custom wrapper with documented memory addresses, reward, reset, and deterministic smoke."


def summary(roms: list[dict[str, Any]], recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    by_system: dict[str, int] = {}
    for row in roms:
        system = str(row.get("system") or "unknown")
        by_system[system] = by_system.get(system, 0) + 1
    matched_profiles = sum(1 for item in recommendations if item.get("matched_rom_count"))
    return {
        "rom_count": len(roms),
        "by_system": dict(sorted(by_system.items())),
        "matched_priority_profiles": matched_profiles,
        "ready_for_wrapper_smoke": matched_profiles > 0,
        "autonomous_rom_downloads": 0,
    }


def next_actions(roms: list[dict[str, Any]], recommendations: list[dict[str, Any]]) -> list[str]:
    if not roms:
        return [
            "Place ethically obtained ROMs under data/local_roms, set SPARKSTREAM_ROM_ROOTS to your private collection path, or run local_rom_registry.py with --rom-root.",
            "Start with Pokemon Emerald for GBA/PyGBA or Tetris/Pokemon Red/Blue/Yellow for GB/GBC PyBoy/Gymboy wrappers.",
        ]
    actions = []
    top = next((item for item in recommendations if item.get("matched_rom_count")), None)
    if top:
        actions.append(f"Build wrapper smoke for {top.get('profile_id')} using adapter {top.get('adapter')}.")
    actions.append("Keep ROM files ignored and promote only wrapper smoke/regression reports into git-tracked code.")
    return actions


def root_record(root: Path) -> dict[str, Any]:
    return {
        "path": safe_path(root),
        "exists": root.exists(),
    }


def safe_path(path: Path) -> str:
    try:
        resolved = path.resolve()
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        # External collection paths are private machine state. Keep only enough
        # structure to diagnose which configured root was used.
        return f"<external>/{path.name}"


def system_for_suffix(suffix: str) -> str:
    text = suffix.lower()
    if text == ".gba":
        return "gba"
    if text == ".gbc":
        return "gbc"
    if text == ".gb":
        return "gb"
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
