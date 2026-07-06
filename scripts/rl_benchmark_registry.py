"""RL benchmark registry and license-gated source discovery.

SparkStream can grow an RL frontier from local toy tasks, Puffer/Ocean-style
environments, and later emulators or homebrew game assets. This script keeps
that growth bounded: it inventories local assets, discovers public candidates
when allowed, and marks anything ROM-like as pending license audit unless an
explicit permissive/open license signal is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "rl_benchmark_registry.json"
DEFAULT_ADAPTER_FACTORY = ROOT / "reports" / "benchmark_adapter_factory.json"
ALLOWED_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "mpl-2.0",
    "lgpl-2.1",
    "lgpl-3.0",
    "gpl-2.0",
    "gpl-3.0",
    "cc0-1.0",
    "unlicense",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--refresh-local", action="store_true")
    parser.add_argument("--allow-network-discovery", action="store_true")
    parser.add_argument("--allow-network-import", action="store_true")
    parser.add_argument("--import-approved", action="store_true")
    parser.add_argument("--discover-query", default="")
    parser.add_argument("--discover-limit", type=int, default=10)
    parser.add_argument("--max-imports", type=int, default=2)
    args = parser.parse_args()

    registry = read_json(ROOT / args.out) or base_registry()
    registry["updated_utc"] = now()

    if args.refresh_local or not registry.get("local_rl_inventory"):
        registry["local_rl_inventory"] = local_rl_inventory()

    smoke_frontiers = adapter_smoke_frontiers()
    registry["adapter_smoke_frontiers"] = smoke_frontiers
    registry["recommended_frontier"] = merge_recommended_frontiers(
        recommended_frontier(registry.get("local_rl_inventory") or {}),
        smoke_frontiers.get("rl_cards", []),
    )

    if args.discover_query:
        registry.setdefault("network_discovery_attempts", [])
        if not args.allow_network_discovery:
            registry["network_discovery_attempts"].append(
                {
                    "query": args.discover_query,
                    "status": "blocked_policy_requires_allow_network_discovery",
                    "created_utc": now(),
                }
            )
        else:
            discovery = discover_github_sources(args.discover_query, args.discover_limit)
            registry["network_discovery_attempts"].append(discovery)
            registry.setdefault("discovered_candidates", []).extend(discovery.get("candidates", []))

    if args.import_approved:
        registry.setdefault("network_import_attempts", [])
        if not args.allow_network_import:
            registry["network_import_attempts"].append(
                {
                    "status": "blocked_policy_requires_allow_network_import",
                    "created_utc": now(),
                    "max_imports": args.max_imports,
                }
            )
        else:
            registry["network_import_attempts"].append(import_approved_sources(registry, args.max_imports))

    registry["summary"] = summary(registry)
    write_json(ROOT / args.out, registry)
    print(json.dumps(registry, indent=2))
    return 0


def base_registry() -> dict[str, Any]:
    return {
        "policy": "sparkstream_rl_benchmark_registry_v0",
        "created_utc": now(),
        "updated_utc": now(),
        "license_policy": {
            "allowed_open_source_licenses": sorted(ALLOWED_LICENSES),
            "commercial_rom_downloads": "forbidden_without_explicit_rights",
            "gameboy_mode": "emulator_and_homebrew_only_until_user_provides_licensed_roms",
            "autonomous_import": "discover_and_queue; clone/download only after license audit",
        },
        "local_rl_inventory": {},
        "discovered_candidates": [],
        "network_discovery_attempts": [],
        "recommended_frontier": [],
        "adapter_smoke_frontiers": {},
        "summary": {},
    }


def local_rl_inventory() -> dict[str, Any]:
    puffer_roots = [
        ("vendored_pufferlib4", ROOT / "vendor" / "pufferlib"),
        ("public_benchmark_cache", ROOT / "data" / "public_benchmarks" / "pufferlib"),
    ]
    envs = []
    for source, puffer in puffer_roots:
        ocean = puffer / "ocean"
        if not ocean.exists():
            continue
        for child in sorted(ocean.iterdir()):
            if not child.is_dir():
                continue
            source_files = list(child.glob("*.c")) + list(child.glob("*.h"))
            envs.append(
                {
                    "name": f"puffer_ocean_{child.name}",
                    "kind": "local_puffer_ocean_env",
                    "path": rel(child),
                    "source": source,
                    "has_binding": (child / "binding.c").exists(),
                    "source_files": len(source_files),
                    "status": "available_local" if source_files else "incomplete",
                    "risk": "low",
                    "license_signal": rel(puffer / "LICENSE") if (puffer / "LICENSE").exists() else "",
                    "capability_target": "fast reset/step rollouts, sparse reward, legal action interfaces, and policy/value traces",
                }
            )
    built_in = [
        {
            "name": "symliquid_gridworld",
            "kind": "local_synthetic_rl",
            "path": "crates/symliquid-core/src/tasks/active_gridworld.rs",
            "status": "available_local",
            "risk": "low",
            "license_signal": "project_local",
        },
        {
            "name": "symliquid_active_classification",
            "kind": "local_active_inference_task",
            "path": "crates/symliquid-core/src/tasks/active_classification.rs",
            "status": "available_local",
            "risk": "low",
            "license_signal": "project_local",
        },
        {
            "name": "local_chess_rl",
            "kind": "local_board_game_rl",
            "path": "scripts/board_game_rl_benchmark.py",
            "status": "available_local" if (ROOT / "scripts" / "board_game_rl_benchmark.py").exists() else "missing_runner",
            "risk": "low",
            "license_signal": "project_local_runner_plus_python_chess_gpl_3_0_plus_dependency",
            "capability_target": "long-horizon tactics, legal-action masking, sparse reward, and self-play Elo trend evidence",
        },
        {
            "name": "local_go_rl",
            "kind": "local_board_game_rl",
            "path": "scripts/board_game_rl_benchmark.py",
            "status": "available_local" if (ROOT / "scripts" / "board_game_rl_benchmark.py").exists() else "missing_runner",
            "risk": "low",
            "license_signal": "project_local_rules_engine",
            "capability_target": "capture tactics, territory shaping, pass/endgame handling, sparse reward, and self-play Elo trend evidence",
        },
    ]
    emulators = detect_emulator_candidates()
    local_roms = read_json(ROOT / "reports" / "local_rom_registry.json")
    return {
        "updated_utc": now(),
        "local_envs": built_in + envs,
        "emulator_candidates": emulators,
        "local_user_roms": {
            "summary": local_roms.get("summary", {}),
            "recommendations": (local_roms.get("recommendations") or [])[:12],
            "policy": local_roms.get("policy"),
            "autonomous_rom_download": local_roms.get("autonomous_rom_download", "forbidden"),
        },
        "counts": {
            "local_envs": len(built_in) + len(envs),
            "puffer_ocean_envs": len(envs),
            "emulator_candidates": len(emulators),
            "local_user_roms": get_path(local_roms, ["summary", "rom_count"], 0),
        },
    }


def detect_emulator_candidates() -> list[dict[str, Any]]:
    roots = [ROOT / "data" / "public_benchmarks", ROOT / "data" / "external_benchmark_candidates"]
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for suffix in ("*.gb", "*.gbc", "*.gba"):
            for path in root.rglob(suffix):
                rows.append(
                    {
                        "path": rel(path),
                        "bytes": path.stat().st_size,
                        "status": "quarantined_pending_license_audit",
                        "note": "ROM-like asset detected. Do not train or benchmark until license is explicit.",
                    }
                )
    return rows


def recommended_frontier(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    envs = inventory.get("local_envs") or []
    recs = []
    for name in [
        "local_chess_rl",
        "local_go_rl",
        "puffer_ocean_snake",
        "puffer_ocean_breakout",
        "puffer_ocean_connect4",
        "symliquid_gridworld",
    ]:
        match = next((env for env in envs if env.get("name") == name), None)
        if match:
            recs.append(
                {
                    "name": match["name"],
                    "priority": "high" if ("puffer" in match["name"] or "local_" in match["name"]) else "medium",
                    "status": match["status"],
                    "next_step": "add smoke eval, then promote to regression only after stable seeded score and runtime budget",
                }
            )
    if not recs:
        recs.append(
            {
                "name": "rl_frontier_missing",
                "priority": "high",
                "next_step": "fetch or implement a licensed/open local RL environment before long RL ratcheting",
            }
        )
    local_rom_profiles = get_path(inventory, ["local_user_roms", "recommendations"], [])
    for item in local_rom_profiles:
        if not isinstance(item, dict) or not item.get("matched_rom_count"):
            continue
        recs.insert(
            0,
            {
                "name": f"local_rom_{item.get('profile_id')}",
                "priority": item.get("priority", "medium"),
                "status": "available_user_supplied_private_rom",
                "next_step": item.get("next_step"),
                "adapter": item.get("adapter"),
                "matched_rom_count": item.get("matched_rom_count"),
            },
        )
    return recs


def adapter_smoke_frontiers() -> dict[str, Any]:
    report = read_json(DEFAULT_ADAPTER_FACTORY)
    cards = report.get("cards") if isinstance(report.get("cards"), list) else []
    smoke_passed = [
        compact_adapter_card(card)
        for card in cards
        if isinstance(card, dict) and card.get("status") == "adapter_smoke_passed"
    ]
    smoke_passed = [card for card in smoke_passed if card]
    rl_cards = [
        card
        for card in smoke_passed
        if "rl" in str(card.get("category", "")).lower()
        or str(card.get("runner_family", "")).lower() in {"rl_local", "emulator_rl_local"}
    ]
    drone_cards = [
        card
        for card in smoke_passed
        if str(card.get("category", "")).startswith("drone_")
        or str(card.get("runner_family", "")).startswith("drone_")
    ]
    minecraft_cards = [
        card
        for card in smoke_passed
        if str(card.get("category", "")) == "minecraft_rl_environment"
        or str(card.get("runner_family", "")) == "minecraft_rl_local"
    ]
    by_category: dict[str, int] = {}
    for card in smoke_passed:
        category = str(card.get("category") or "unknown")
        by_category[category] = by_category.get(category, 0) + 1
    priority_order = {"high": 0, "medium": 1, "low": 2}
    smoke_passed.sort(key=lambda item: (priority_order.get(str(item.get("priority")), 9), item.get("id", "")))
    rl_cards.sort(key=lambda item: (priority_order.get(str(item.get("priority")), 9), item.get("id", "")))
    drone_cards.sort(key=lambda item: (priority_order.get(str(item.get("priority")), 9), item.get("id", "")))
    minecraft_cards.sort(key=lambda item: (priority_order.get(str(item.get("priority")), 9), item.get("id", "")))
    return {
        "source": rel(DEFAULT_ADAPTER_FACTORY) if DEFAULT_ADAPTER_FACTORY.exists() else "",
        "updated_utc": now(),
        "summary": {
            "smoke_passed_cards": len(smoke_passed),
            "smoke_passed_rl_cards": len(rl_cards),
            "smoke_passed_drone_cards": len(drone_cards),
            "smoke_passed_minecraft_cards": len(minecraft_cards),
            "by_category": by_category,
        },
        "rl_cards": rl_cards[:16],
        "drone_cards": drone_cards[:16],
        "minecraft_cards": minecraft_cards[:16],
        "cards": smoke_passed[:36],
    }


def compact_adapter_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": card.get("id"),
        "name": card.get("name"),
        "category": card.get("category"),
        "priority": card.get("priority"),
        "status": card.get("status"),
        "adapter_type": card.get("adapter_type"),
        "runner_family": card.get("runner_family"),
        "runtime_tier": card.get("runtime_tier"),
        "risk_tier": card.get("risk_tier"),
        "source": card.get("resource_pantry_path") or card.get("staged_path") or card.get("url") or "",
        "next_step": "run bounded local eval/training smoke, then promote to frontier or regression through the ratchet gate",
        "external_inference_calls": int(card.get("external_inference_calls") or 0),
    }


def merge_recommended_frontiers(
    local_recs: list[dict[str, Any]],
    smoke_rl_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(local_recs)
    seen = {str(item.get("name") or item.get("id") or "") for item in merged}
    for card in smoke_rl_cards:
        name = str(card.get("id") or "")
        if not name or name in seen:
            continue
        merged.append(
            {
                "name": name,
                "priority": card.get("priority", "medium"),
                "status": "adapter_smoke_passed",
                "adapter_type": card.get("adapter_type"),
                "runner_family": card.get("runner_family"),
                "source": card.get("source"),
                "next_step": card.get("next_step"),
            }
        )
        seen.add(name)
    return merged[:16]


def discover_github_sources(query: str, limit: int) -> dict[str, Any]:
    capped = max(1, min(limit, 25))
    encoded = urllib.parse.urlencode({"q": query, "per_page": str(capped)})
    url = f"https://api.github.com/search/repositories?{encoded}"
    request = urllib.request.Request(url, headers={"User-Agent": "SparkStreamRLBenchmarkRegistry/0.1"})
    attempt = {
        "query": query,
        "limit": capped,
        "created_utc": now(),
        "source": "github_repository_search",
        "status": "ok",
        "candidates": [],
    }
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read(4 * 1024 * 1024).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - record failure and keep daemon alive.
        attempt["status"] = "failed"
        attempt["error"] = str(exc)
        return attempt
    for row in payload.get("items", [])[:capped]:
        license_info = row.get("license") or {}
        spdx = str(license_info.get("spdx_id") or "").lower()
        rom_like = looks_rom_related(row)
        audit = "approved_open_license_pending_import" if spdx in ALLOWED_LICENSES else "pending_license_audit"
        if rom_like and spdx not in ALLOWED_LICENSES:
            audit = "rom_like_pending_explicit_rights"
        attempt["candidates"].append(
            {
                "name": row.get("full_name"),
                "url": row.get("html_url"),
                "clone_url": row.get("clone_url"),
                "description": row.get("description"),
                "stars": row.get("stargazers_count"),
                "updated_at": row.get("updated_at"),
                "license_spdx": spdx or "unknown",
                "audit_status": audit,
                "rom_like": rom_like,
            }
        )
    return attempt


def import_approved_sources(registry: dict[str, Any], max_imports: int) -> dict[str, Any]:
    capped = max(0, min(max_imports, 10))
    imported = registry.setdefault("imported_sources", [])
    imported_names = {
        item.get("name")
        for item in imported
        if isinstance(item, dict) and item.get("status") in {"imported_pending_integration", "already_imported"}
    }
    candidates = [
        item
        for item in registry.get("discovered_candidates", [])
        if isinstance(item, dict)
        and item.get("audit_status") == "approved_open_license_pending_import"
        and not item.get("rom_like")
        and item.get("name")
        and item.get("name") not in imported_names
    ]
    attempt = {
        "created_utc": now(),
        "status": "ok",
        "requested": capped,
        "eligible_candidates": len(candidates),
        "imports": [],
    }
    for candidate in candidates[:capped]:
        result = import_github_candidate(candidate)
        attempt["imports"].append(result)
        imported.append(result)
    if capped == 0:
        attempt["status"] = "no_imports_requested"
    elif not attempt["imports"]:
        attempt["status"] = "no_eligible_approved_sources"
    return attempt


def import_github_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    name = str(candidate.get("name") or "")
    safe_name = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in name)[:120]
    target_dir = ROOT / "data" / "external_benchmark_candidates" / "rl_sources"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_name}.zip"
    if target.exists():
        return {
            "name": name,
            "url": candidate.get("url"),
            "license_spdx": candidate.get("license_spdx"),
            "path": rel(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "status": "already_imported",
            "created_utc": now(),
        }
    repo = urllib.parse.quote(name, safe="/")
    url = f"https://api.github.com/repos/{repo}/zipball"
    try:
        bytes_written, digest = download_capped(url, target, 100 * 1024 * 1024)
        return {
            "name": name,
            "url": candidate.get("url"),
            "download_url": url,
            "license_spdx": candidate.get("license_spdx"),
            "path": rel(target),
            "bytes": bytes_written,
            "sha256": digest,
            "status": "imported_pending_integration",
            "note": "Open-license source archive staged only. Integrate as a benchmark after adapter/eval audit.",
            "created_utc": now(),
        }
    except Exception as exc:  # noqa: BLE001 - record and keep daemon alive.
        target.unlink(missing_ok=True)
        return {
            "name": name,
            "url": candidate.get("url"),
            "download_url": url,
            "license_spdx": candidate.get("license_spdx"),
            "status": "import_failed",
            "error": str(exc),
            "created_utc": now(),
        }


def download_capped(url: str, target: Path, max_bytes: int) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "SparkStreamRLBenchmarkRegistry/0.1"})
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(f"download_exceeded_cap:{max_bytes}")
            digest.update(chunk)
            handle.write(chunk)
    return total, digest.hexdigest()


def looks_rom_related(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "")
        for key in ["name", "full_name", "description", "topics"]
    ).lower()
    return any(token in text for token in ["gameboy", "game boy", "gb rom", "gbc", "gba", "rom"])


def summary(registry: dict[str, Any]) -> dict[str, Any]:
    inventory = registry.get("local_rl_inventory") or {}
    candidates = registry.get("discovered_candidates") or []
    approved = [
        item for item in candidates
        if isinstance(item, dict) and item.get("audit_status") == "approved_open_license_pending_import"
    ]
    return {
        "local_envs": get_path(inventory, ["counts", "local_envs"], 0),
        "puffer_ocean_envs": get_path(inventory, ["counts", "puffer_ocean_envs"], 0),
        "emulator_assets_pending_audit": get_path(inventory, ["counts", "emulator_candidates"], 0),
        "local_user_roms": get_path(inventory, ["counts", "local_user_roms"], 0),
        "local_rom_profiles_ready": get_path(inventory, ["local_user_roms", "summary", "matched_priority_profiles"], 0),
        "adapter_smoke_passed_cards": get_path(
            registry, ["adapter_smoke_frontiers", "summary", "smoke_passed_cards"], 0
        ),
        "adapter_smoke_passed_rl_cards": get_path(
            registry, ["adapter_smoke_frontiers", "summary", "smoke_passed_rl_cards"], 0
        ),
        "adapter_smoke_passed_drone_cards": get_path(
            registry, ["adapter_smoke_frontiers", "summary", "smoke_passed_drone_cards"], 0
        ),
        "discovered_candidates": len(candidates),
        "approved_open_license_candidates": len(approved),
        "imported_sources": len(registry.get("imported_sources") or []),
        "recommended_frontiers": len(registry.get("recommended_frontier") or []),
    }


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
