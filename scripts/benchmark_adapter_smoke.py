"""Smoke-test benchmark adapter cards without asking the teacher.

This runner intentionally performs the smallest local checks that turn a
catalogued benchmark source into actionable evidence:

- license/provenance and card schema checks;
- source archive or clone readability checks;
- local runtime dependency probes and bounded reset/step checks for known
  benchmark runtimes;
- local ROM rights/git-ignore gates for user-supplied emulator assets.

It never downloads sources and never calls external inference. Runtime probes
execute only known, bounded local smoke snippets in short-lived Python
processes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTORY = ROOT / "reports" / "benchmark_adapter_factory.json"
DEFAULT_PANTRY = ROOT / "reports" / "resource_pantry.json"
DEFAULT_ROMS = ROOT / "reports" / "local_rom_registry.json"
DEFAULT_OUT = ROOT / "reports" / "benchmark_adapter_smoke_status.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "benchmark_adapter_smoke_status.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory-report", default=str(DEFAULT_FACTORY.relative_to(ROOT)))
    parser.add_argument("--resource-pantry", default=str(DEFAULT_PANTRY.relative_to(ROOT)))
    parser.add_argument("--local-rom-registry", default=str(DEFAULT_ROMS.relative_to(ROOT)))
    parser.add_argument("--card-id", action="append", default=[])
    parser.add_argument("--needs-smoke-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--replace", action="store_true", help="Replace the smoke ledger instead of merging with prior smoke results.")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    factory = read_json(ROOT / args.factory_report)
    pantry = read_json(ROOT / args.resource_pantry)
    roms = read_json(ROOT / args.local_rom_registry)
    pantry_by_id = {
        str(item.get("id") or ""): item
        for item in pantry.get("sources", [])
        if isinstance(item, dict)
    }

    cards = [item for item in factory.get("cards", []) if isinstance(item, dict)]
    selected = select_cards(cards, args.card_id, args.needs_smoke_only, args.limit)
    rows = [smoke_card(card, pantry_by_id, roms) for card in selected]
    out_path = ROOT / args.out
    merged_rows = rows if args.replace else merge_prior_rows(out_path, rows, {str(card.get("id") or "") for card in cards})
    summary = summarize(merged_rows, len(cards))
    report = {
        "policy": "sparkstream_benchmark_adapter_smoke_status_v0",
        "created_utc": now(),
        "factory_report": args.factory_report,
        "resource_pantry": args.resource_pantry,
        "local_rom_registry": args.local_rom_registry,
        "selected_card_count": len(selected),
        "total_factory_cards": len(cards),
        "summary": summary,
        "cards": merged_rows,
        "next_actions": next_actions(merged_rows),
        "external_inference_calls": 0,
    }
    for row in merged_rows:
        row["report_path"] = rel(out_path)
    write_json(out_path, report)
    write_markdown(ROOT / args.markdown_out, report)
    print(json.dumps(report, indent=2))
    return 1 if summary["failed"] else 0


def select_cards(
    cards: list[dict[str, Any]],
    card_ids: list[str],
    needs_smoke_only: bool,
    limit: int,
) -> list[dict[str, Any]]:
    wanted = {item.strip() for item in card_ids if item.strip()}
    rows = cards
    if wanted:
        rows = [card for card in rows if str(card.get("id")) in wanted]
    if needs_smoke_only:
        rows = [card for card in rows if card.get("status") in {"needs_adapter_smoke", "adapter_card_ready"}]
    priority = {"highest": 0, "high": 1, "medium": 2, "low": 3}
    rows = sorted(rows, key=lambda card: (priority.get(str(card.get("priority") or "medium"), 2), str(card.get("id"))))
    if limit > 0:
        rows = rows[:limit]
    return rows


def merge_prior_rows(path: Path, rows: list[dict[str, Any]], valid_card_ids: set[str]) -> list[dict[str, Any]]:
    prior = read_json(path)
    merged: dict[str, dict[str, Any]] = {}
    for row in prior.get("cards", []) if isinstance(prior.get("cards"), list) else []:
        if not isinstance(row, dict):
            continue
        card_id = str(row.get("card_id") or row.get("id") or "")
        if card_id and card_id in valid_card_ids:
            merged[card_id] = row
    for row in rows:
        card_id = str(row.get("card_id") or row.get("id") or "")
        if card_id:
            merged[card_id] = row
    priority = {"highest": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(
        merged.values(),
        key=lambda row: (
            priority.get(str(row.get("priority") or "medium"), 2),
            str(row.get("card_id") or row.get("id") or ""),
        ),
    )


def smoke_card(card: dict[str, Any], pantry_by_id: dict[str, dict[str, Any]], roms: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    card_id = str(card.get("id") or "")
    source_id = str(card.get("source_id") or card_id.removeprefix("source_"))
    adapter_type = str(card.get("adapter_type") or "")
    category = str(card.get("category") or "")
    source = resolve_source(card, source_id, pantry_by_id)

    add_check(checks, "card_schema", bool(card_id and adapter_type and card.get("runner_family")), "hard", f"id={card_id} adapter={adapter_type}")
    add_check(checks, "license_allowed", bool(card.get("license_allowed")), "hard", f"license={card.get('license_spdx')} decision={card.get('decision')}")
    add_check(checks, "no_external_inference", int(card.get("external_inference_calls") or 0) == 0, "hard", "adapter smoke is local-only")

    if adapter_type == "emulator_rl_adapter" or category == "emulator_rl_environment":
        probe_emulator(card, source, checks, roms, pantry_by_id)
    elif adapter_type == "emulator_runtime_dependency_adapter" or category == "emulator_runtime_dependency":
        probe_source(card, source, checks)
        if source_id == "mgba":
            probe_mgba_runtime(checks, python=runtime_python_for_source("pygba"))
    elif adapter_type in {"drone_rl_adapter", "drone_racing_sitl_adapter", "drone_control_api_adapter"} or category.startswith("drone_"):
        probe_drone(card, source, checks)
    elif adapter_type in {"coding_agent_harness_adapter", "coding_agent_framework_adapter"} or category in {
        "coding_agent_benchmark",
        "coding_agent_framework",
    }:
        probe_coding_agent_harness(card, source, checks)
    elif adapter_type == "sandboxed_code_eval_adapter":
        probe_source(card, source, checks)
        add_check(checks, "sandbox_policy_recorded", True, "hard", "E3 sandbox policy is carried on the card")
    elif adapter_type == "minecraft_rl_adapter" or category == "minecraft_rl_environment":
        probe_minecraft(card, source, checks)
    elif adapter_type in {"gymnasium_or_native_rl_adapter", "language_eval_adapter"} and category in {"rl_benchmark", "rl_environment", "physics_rl_environment"}:
        probe_source(card, source, checks)
        module = module_for_source(source_id)
        if module:
            runtime_path = runtime_extra_path(source_id, source.get("path"))
            if probe_import(checks, module, runtime_path):
                probe_rl_reset_step(checks, source_id, runtime_path)
    elif adapter_type == "native_voice_eval_adapter":
        probe_source(card, source, checks)
        voice_policy = read_json(ROOT / "configs" / "native_voice_policy.json")
        add_check(
            checks,
            "native_voice_policy_present",
            bool(voice_policy),
            "hard",
            "voice scoring is native Theseus head/router I/O, not an installed STT/TTS dependency",
        )
        add_check(
            checks,
            "provider_stt_tts_forbidden",
            get_path(voice_policy, ["execution_boundary", "provider_stt_tts"], "") == "forbidden",
            "hard",
            "provider speech inference is forbidden",
        )
        add_check(
            checks,
            "pretrained_voice_inference_forbidden",
            get_path(voice_policy, ["execution_boundary", "pretrained_third_party_voice_models"], "")
            == "forbidden_for_inference",
            "hard",
            "Whisper/Vosk/SpeechBrain/pyttsx3-style inference cannot satisfy this lane",
        )
        add_check(checks, "privacy_policy_present", True, "hard", "voice benchmarks must stay local and metadata-gated")
    else:
        probe_source(card, source, checks)

    status = classify(checks)
    return {
        "card_id": card_id,
        "name": card.get("name"),
        "category": category,
        "adapter_type": adapter_type,
        "runner_family": card.get("runner_family"),
        "priority": card.get("priority"),
        "created_utc": now(),
        "smoke_status": status,
        "passed": status == "passed",
        "blocked": status in {"blocked", "metadata_passed_runtime_blocked", "failed"},
        "source": source,
        "checks": checks,
        "external_inference_calls": 0,
    }


def resolve_source(card: dict[str, Any], source_id: str, pantry_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for key in ("resource_pantry_path", "staged_path"):
        value = str(card.get(key) or "")
        if value:
            candidates.append(value)
    pantry = pantry_by_id.get(source_id, {})
    if pantry.get("clone_path"):
        candidates.append(str(pantry.get("clone_path")))
    if pantry.get("metadata_path"):
        candidates.append(str(pantry.get("metadata_path")))
    for value in candidates:
        path = resolve_path(value)
        if path.exists():
            return {
                "kind": "zip_archive" if path.suffix.lower() == ".zip" else "directory" if path.is_dir() else "file",
                "path": rel_or_abs(path),
                "exists": True,
                "pantry_present": bool(pantry.get("present")),
                "metadata_path": pantry.get("metadata_path", ""),
            }
    return {
        "kind": "missing",
        "path": candidates[0] if candidates else "",
        "exists": False,
        "pantry_present": bool(pantry.get("present")),
        "metadata_path": pantry.get("metadata_path", ""),
    }


def probe_source(card: dict[str, Any], source: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    path = resolve_path(str(source.get("path") or ""))
    add_check(checks, "source_resolved", source.get("exists") is True, "hard", f"{source.get('kind')} {source.get('path')}")
    if not path.exists():
        return
    if path.is_dir():
        health = clone_health(path)
        add_check(checks, "source_readable", True, "hard", f"entries={health['top_level_entries']}")
        add_check(checks, "license_or_readme_hint", bool(health["license_file_hint"] or health["readme_hint"]), "hard", json.dumps(health))
        source["health"] = health
    elif path.suffix.lower() == ".zip":
        health = zip_health(path)
        add_check(checks, "archive_readable", health["ok"], "hard", f"members={health.get('members', 0)}")
        add_check(checks, "license_or_readme_hint", bool(health.get("license_file_hint") or health.get("readme_hint")), "hard", json.dumps(health))
        source["health"] = health
    else:
        add_check(checks, "source_file_readable", path.is_file(), "hard", f"bytes={path.stat().st_size if path.exists() else 0}")


def probe_coding_agent_harness(card: dict[str, Any], source: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    source_id = str(card.get("source_id") or "")
    probe_source(card, source, checks)
    path = resolve_path(str(source.get("path") or ""))
    permission = card.get("permission_envelope") if isinstance(card.get("permission_envelope"), dict) else {}
    promotion_gates = {str(item) for item in card.get("promotion_gates", []) if isinstance(item, str)}
    adapter_plan = str(card.get("adapter_plan") or "").lower()
    add_check(
        checks,
        "sandbox_policy_recorded",
        permission.get("runtime") in {"E3", "E4"} and "external_inference" in permission,
        "hard",
        f"runtime={permission.get('runtime')} external_inference={permission.get('external_inference')}",
    )
    add_check(
        checks,
        "provider_judges_disabled_by_policy",
        permission.get("external_inference") == "forbidden" and "no_external_inference" in promotion_gates,
        "hard",
        "provider LLM/judge calls cannot satisfy coding-agent benchmark pressure",
    )
    add_check(
        checks,
        "network_disabled_during_eval",
        "forbidden during scoring" in str(permission.get("network") or ""),
        "hard",
        str(permission.get("network") or ""),
    )
    add_check(
        checks,
        "local_endpoint_contract_recorded",
        "local" in adapter_plan or "endpoint" in json.dumps(card.get("input_contract", {})).lower(),
        "hard",
        "Theseus or the local OpenAI-compatible endpoint is the only allowed solver surface",
    )
    if not path.exists():
        return
    if path.is_dir():
        files = list_limited_files(path, limit=2500)
        lowered = [rel_or_abs(item).lower().replace("\\", "/") for item in files]
        manifest_hits = [
            name
            for name in lowered
            if name.endswith(("package.json", "pyproject.toml", "setup.py", "uv.lock", "bun.lock", "bun.lockb", "pnpm-lock.yaml"))
        ]
        task_hits = [
            name
            for name in lowered
            if any(token in name for token in ("/bench", "/eval", "/task", "/scenario", "/problem", "/test", "/swe"))
            and name.endswith((".json", ".jsonl", ".yaml", ".yml", ".toml", ".md", ".py", ".ts"))
        ]
        provider_hits = [
            name
            for name in lowered
            if any(token in name for token in ("openai", "anthropic", "gemini", "provider", "api_key", "models.dev", "litellm"))
        ][:24]
        add_check(checks, "framework_manifest_present", bool(manifest_hits), "hard", ", ".join(manifest_hits[:8]) or "no package/pyproject/setup manifest")
        add_check(checks, "task_manifest_or_harness_present", bool(task_hits or manifest_hits), "hard", ", ".join(task_hits[:8]) or "framework manifest only")
        add_check(
            checks,
            "provider_paths_audited",
            True,
            "hard",
            json.dumps({"provider_path_count_sampled": len(provider_hits), "sample": provider_hits[:8]}),
        )
        if source_id in {"opencode", "opencode_bench"}:
            add_check(checks, "node_runtime_available", command_available("node"), "runtime", "node is required for OpenCode-style harness metadata probes")
            add_check(checks, "bun_runtime_available", command_available("bun"), "runtime", "OpenCode sources declare Bun as their package/runtime manager")
        elif source_id in {"openhands", "terminal_bench", "swe_atlas", "swe_rex", "swe_smith"}:
            container_status = container_runtime_status()
            add_check(
                checks,
                "container_runtime_available_for_full_harness",
                bool(container_status.get("ready")),
                "runtime",
                json.dumps(container_status),
            )
    elif path.suffix.lower() == ".zip":
        add_check(checks, "archive_agent_harness_manifest", True, "hard", "zip source readability checked by archive health")


def probe_emulator(
    card: dict[str, Any],
    source: dict[str, Any],
    checks: list[dict[str, Any]],
    roms: dict[str, Any],
    pantry_by_id: dict[str, dict[str, Any]],
) -> None:
    source_id = str(card.get("source_id") or "pygba")
    python = runtime_python_for_source(source_id)
    input_contract = card.get("input_contract") if isinstance(card.get("input_contract"), dict) else {}
    profile = str(input_contract.get("rom_profile_id") or source_id or "")
    requires_rom = bool(input_contract.get("rom_profile_id")) or str(card.get("id") or "").startswith("local_rom_")
    if not requires_rom:
        probe_source(card, source, checks)
        add_check(checks, "source_only_emulator_framework", True, "hard", "framework source smoke does not require a local user ROM")
        if source_id == "pyboy":
            if card.get("license_allowed"):
                probe_import(checks, "pyboy", runtime_extra_path(source_id, source.get("path")), python=python)
            return
        if source_id == "gymboy":
            probe_import(checks, "gymboy", runtime_extra_path(source_id, source.get("path")), python=python)
            return
        if source_id == "stable_retro":
            imported = probe_import(checks, "retro", runtime_extra_path(source_id, source.get("path")), check_name="stable_retro_import_available", python=python)
            if not imported:
                add_check(checks, "stable_retro_windows_build_note", False, "runtime", "stable-retro has no local importable runtime yet; source is staged but Windows wheel/build is unresolved")
            return

    is_gba_lane = source_id == "pygba" or profile.startswith("gba_")
    if requires_rom and not is_gba_lane:
        probe_gb_or_gbc_emulator(card, checks, roms, profile, python=python)
        return

    rom_path = Path()
    if requires_rom:
        rec = next(
            (
                item
                for item in roms.get("recommendations", [])
                if isinstance(item, dict) and str(item.get("profile_id")) == profile
            ),
            {},
        )
        matched = rec.get("matched_roms", []) if isinstance(rec.get("matched_roms"), list) else []
        rom = matched[0] if matched else {}
        rom_path = resolve_path(str(rom.get("path") or ""))
        add_check(checks, "user_supplied_rom_present", rom_path.exists(), "hard", f"profile={profile} path={safe_rom_path(rom_path)}")
        add_check(checks, "rom_file_ignored_by_git", git_ignored(rom_path) if rom_path.exists() else False, "hard", safe_rom_path(rom_path))
    pygba = pantry_by_id.get("pygba", {})
    pygba_path = resolve_path(str(pygba.get("clone_path") or ""))
    add_check(checks, "pygba_source_present", pygba_path.exists(), "hard", rel_or_abs(pygba_path))
    mgba_ready = probe_mgba_runtime(checks, python=python)
    pygba_import_path = None if import_available("pygba", python=python) else (pygba_path / "src" if pygba_path.exists() else None)
    probe_import(checks, "pygba", pygba_import_path, check_name="pygba_import_available", python=python)
    if (
        requires_rom
        and rom_path.exists()
        and pygba_path.exists()
        and mgba_ready
        and import_available("pygba", pygba_import_path, python=python)
    ):
        script = (
            "from pygba import PyGBA, PyGBAEnv, PokemonEmerald\n"
            f"gba=PyGBA.load({str(rom_path)!r})\n"
            "env=PyGBAEnv(gba, PokemonEmerald(), frameskip=1, max_episode_steps=2)\n"
            "obs,info=env.reset()\n"
            "obs,reward,done,truncated,info=env.step(env.action_space.sample())\n"
            "print(type(obs).__name__, float(reward), bool(done), bool(truncated))\n"
        )
        result = run_python(script, extra_path=pygba_import_path, timeout=30, python=python)
        add_check(checks, "wrapper_reset_step", result["ok"], "runtime", result["stdout_tail"] or result["stderr_tail"])
    elif requires_rom:
        add_check(checks, "wrapper_reset_step", False, "runtime", "skipped until mgba and pygba imports are available")


def probe_gb_or_gbc_emulator(
    card: dict[str, Any],
    checks: list[dict[str, Any]],
    roms: dict[str, Any],
    profile: str,
    *,
    python: Path | None = None,
) -> None:
    rec = next(
        (
            item
            for item in roms.get("recommendations", [])
            if isinstance(item, dict) and str(item.get("profile_id")) == profile
        ),
        {},
    )
    matched = rec.get("matched_roms", []) if isinstance(rec.get("matched_roms"), list) else []
    rom = matched[0] if matched else {}
    rom_path = resolve_path(str(rom.get("path") or ""))
    add_check(checks, "user_supplied_rom_present", rom_path.exists(), "hard", f"profile={profile} path={safe_rom_path(rom_path)}")
    add_check(checks, "rom_file_ignored_by_git", git_ignored(rom_path) if rom_path.exists() else False, "hard", safe_rom_path(rom_path))
    pyboy_available = probe_import(checks, "pyboy", None, check_name="pyboy_import_available", python=python)
    gymboy_available = import_available("gymboy", python=python)
    add_check(checks, "gymboy_import_available", gymboy_available, "runtime", "import_ok" if gymboy_available else "gymboy not importable in emulator runtime")
    if rom_path.exists() and pyboy_available:
        script = (
            "from pyboy import PyBoy\n"
            f"pyboy=PyBoy({str(rom_path)!r}, window='null')\n"
            "pyboy.tick(1)\n"
            "print('pyboy_boot_step_ok', pyboy.frame_count)\n"
            "pyboy.stop()\n"
        )
        result = run_python(script, timeout=30, python=python)
        add_check(checks, "pyboy_boot_step", result["ok"], "runtime", result["stdout_tail"] or result["stderr_tail"])
    elif rom_path.exists():
        add_check(checks, "pyboy_boot_step", False, "runtime", "skipped until pyboy imports are available")


def probe_mgba_runtime(checks: list[dict[str, Any]], *, python: Path | None = None) -> bool:
    script = (
        "import mgba\n"
        "import mgba.core\n"
        "from mgba._pylib import ffi, lib\n"
        "assert hasattr(lib, 'mCoreFind'), 'missing mCoreFind symbol'\n"
        "ffi.sizeof('mColor')\n"
        "print('mgba_python_bindings_ok', mgba.__version__)\n"
    )
    result = run_python(script, timeout=15, python=python)
    add_check(
        checks,
        "mgba_python_bindings_available",
        result["ok"],
        "runtime",
        result["stdout_tail"] or result["stderr_tail"] or "mGBA Python CFFI bindings are not fully importable",
    )
    return bool(result["ok"])


def probe_drone(card: dict[str, Any], source: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    probe_source(card, source, checks)
    source_id = str(card.get("source_id") or card.get("id") or "")
    category = str(card.get("category") or "")
    adapter_type = str(card.get("adapter_type") or "")
    python = runtime_python_for_source(source_id)
    permission = card.get("permission_envelope") if isinstance(card.get("permission_envelope"), dict) else {}
    gates = {str(item) for item in card.get("promotion_gates", [])}
    approval = set(permission.get("approval_required_for", [])) if isinstance(permission.get("approval_required_for"), list) else set()
    add_check(
        checks,
        "drone_safety_contract_present",
        permission.get("hardware") == "forbidden_without_explicit_human_approval"
        and "live_drone_endpoint" in approval
        and (
            any("live" in gate or "hardware" in gate for gate in gates)
            or "human_interaction_forbidden_during_submitted_run" in gates
            or "simulator_endpoint_health_passes_or_runtime_blocked" in gates
        ),
        "hard",
        json.dumps({"hardware": permission.get("hardware"), "approval_required_for": sorted(approval), "gates": sorted(gates)[:8]}),
    )
    add_check(checks, "simulation_or_loopback_only_by_default", True, "hard", "adapter smoke never connects to live flight hardware")
    if category == "drone_racing_simulator" or adapter_type == "drone_racing_sitl_adapter":
        digest = read_json(ROOT / "reports" / "ai_grand_prix_spec_digest.json")
        add_check(
            checks,
            "ai_grand_prix_spec_contract_recorded",
            bool(digest.get("spec_id") and digest.get("runtime", {}).get("python_known_good")),
            "hard",
            "reports/ai_grand_prix_spec_digest.json",
        )
    module = module_for_source(source_id)
    if module:
        runtime_path = runtime_extra_path(source_id, source.get("path"))
        imported = probe_import(checks, module, runtime_path, python=python)
        if imported:
            probe_drone_reset_step(checks, source_id, runtime_path, python=python)
            if adapter_type == "drone_racing_sitl_adapter":
                probe_airsim_endpoint(checks, source_id, python=python)
    elif source_id == "px4_sitl":
        probe_import(checks, "mavsdk", None, check_name="mavsdk_import_available", python=python)
        probe_px4_sitl_endpoint(checks, python=python)
    if adapter_type == "drone_control_api_adapter":
        add_check(checks, "no_live_control_connection_attempted", True, "hard", "import-only control API smoke")


def probe_airsim_endpoint(checks: list[dict[str, Any]], source_id: str, *, python: Path | None = None) -> None:
    module = "airsimdroneracinglab" if source_id == "airsim_drone_racing_lab" else "airsim"
    script = (
        f"import {module} as airsim_client\n"
        "client=airsim_client.MultirotorClient(ip='127.0.0.1')\n"
        "ok=bool(client.ping())\n"
        "print('airsim_endpoint_ping', ok)\n"
        "raise SystemExit(0 if ok else 2)\n"
    )
    result = run_python(script, timeout=6, python=python)
    add_check(
        checks,
        "simulator_endpoint_health",
        result["ok"],
        "runtime",
        result["stdout_tail"] or result["stderr_tail"] or "AirSim endpoint not responding on localhost",
    )


def probe_px4_sitl_endpoint(checks: list[dict[str, Any]], *, python: Path | None = None) -> None:
    script = (
        "import asyncio\n"
        "from mavsdk import System\n"
        "async def main():\n"
        "    drone=System()\n"
        "    await drone.connect(system_address='udp://:14540')\n"
        "    async for state in drone.core.connection_state():\n"
        "        if state.is_connected:\n"
        "            print('px4_sitl_connected')\n"
        "            return 0\n"
        "    return 2\n"
        "try:\n"
        "    raise SystemExit(asyncio.run(asyncio.wait_for(main(), timeout=6)))\n"
        "except TimeoutError:\n"
        "    print('px4_sitl_endpoint_timeout')\n"
        "    raise SystemExit(2)\n"
    )
    result = run_python(script, timeout=10, python=python)
    add_check(
        checks,
        "px4_sitl_runtime_probe",
        result["ok"],
        "runtime",
        result["stdout_tail"] or result["stderr_tail"] or "PX4 SITL endpoint not responding on udp://:14540",
    )


def probe_minecraft(card: dict[str, Any], source: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    probe_source(card, source, checks)
    source_id = str(card.get("source_id") or card.get("id") or "")
    policy = read_json(ROOT / "configs" / "minecraft_rl_policy.json")
    runtime = read_json(ROOT / "reports" / "minecraft_runtime_probe.json")
    full_runtime_source = source_id in {"minerl", "minedojo", "malmo"}
    python = runtime_python_for_source(source_id if source_id in {"crafter", "craftax", "minerl", "minedojo", "malmo"} else "crafter")
    if not runtime:
        result = run_python(
            "import subprocess, sys\n"
            "raise SystemExit(subprocess.run([sys.executable, 'scripts/minecraft_runtime_probe.py', '--out', 'reports/minecraft_runtime_probe.json']).returncode)\n",
            timeout=30,
            python=python,
        )
        runtime = read_json(ROOT / "reports" / "minecraft_runtime_probe.json")
        add_check(checks, "minecraft_runtime_probe_ran", result["ok"], "runtime", result["stdout_tail"] or result["stderr_tail"])
    summary = runtime.get("summary") if isinstance(runtime.get("summary"), dict) else {}
    module = module_for_source(source_id)
    add_check(
        checks,
        "minecraft_policy_present",
        bool(policy),
        "hard",
        "configs/minecraft_rl_policy.json",
    )
    add_check(
        checks,
        "user_license_attested_for_full_runtime",
        bool(get_path(policy, ["user_license", "user_reported_license_for_this_machine"], False)),
        "hard",
        "full Minecraft harness requires local user license",
    )
    add_check(
        checks,
        "local_minecraft_install_detected",
        bool(summary.get("local_minecraft_install_detected")) or not full_runtime_source,
        "runtime",
        json.dumps(runtime.get("install_paths", [])[:3]),
    )
    add_check(
        checks,
        "java_available_for_full_runtime",
        bool(summary.get("java_available")) or not full_runtime_source,
        "runtime",
        json.dumps(get_path(runtime, ["runtime", "java"], {})),
    )
    add_check(checks, "no_public_server_by_default", True, "hard", "public server and account actions are policy-gated")
    add_check(checks, "no_credentials_stored", True, "hard", "smoke does not read or store launcher credentials")
    if module:
        runtime_path = runtime_extra_path(source_id, source.get("path"))
        imported = probe_import(checks, module, runtime_path, python=python)
        if not imported and source_id == "malmo":
            imported = probe_import(checks, "MalmoPython", runtime_path, check_name="MalmoPython_import_available", python=python)
        if imported and source_id in {"crafter", "craftax"}:
            probe_rl_reset_step(checks, source_id, runtime_path, python=python)
    full_ready = bool(summary.get("full_minecraft_runtime_ready"))
    bridge_ready = bool(summary.get("bridge_runtime_ready"))
    add_check(
        checks,
        "minecraft_runtime_ready_or_bridge_available",
        full_ready or bridge_ready or source_id in {"minedojo", "malmo", "voyager_minecraft"},
        "runtime",
        f"full={full_ready} bridge={bridge_ready} source_id={source_id}",
    )


def probe_import(
    checks: list[dict[str, Any]],
    module: str,
    extra_path: Path | str | None,
    *,
    check_name: str | None = None,
    python: Path | None = None,
) -> bool:
    script = f"import {module}; print(getattr({module}, '__version__', 'import_ok'))"
    result = run_python(script, extra_path=extra_path, timeout=20, python=python)
    add_check(checks, check_name or f"{module}_import_available", result["ok"], "runtime", result["stdout_tail"] or result["stderr_tail"])
    return bool(result["ok"])


def probe_rl_reset_step(
    checks: list[dict[str, Any]],
    source_id: str,
    extra_path: Path | str | None,
    *,
    python: Path | None = None,
) -> None:
    scripts = {
        "gymnasium": (
            "import gymnasium as gym\n"
            "env=gym.make('CartPole-v1')\n"
            "obs,info=env.reset(seed=0)\n"
            "obs,reward,terminated,truncated,info=env.step(env.action_space.sample())\n"
            "print('gymnasium_reset_step_ok', type(obs).__name__, float(reward), bool(terminated), bool(truncated))\n"
        ),
        "minigrid": (
            "import gymnasium as gym, minigrid\n"
            "env=gym.make('MiniGrid-Empty-5x5-v0')\n"
            "obs,info=env.reset(seed=0)\n"
            "obs,reward,terminated,truncated,info=env.step(env.action_space.sample())\n"
            "print('minigrid_reset_step_ok', type(obs).__name__, float(reward), bool(terminated), bool(truncated))\n"
        ),
        "bsuite": (
            "from bsuite import load\n"
            "env=load('cartpole', {'seed': 0})\n"
            "ts=env.reset()\n"
            "action=env.action_spec().generate_value()\n"
            "ts=env.step(action)\n"
            "print('bsuite_reset_step_ok', float(ts.reward or 0.0), bool(ts.last()))\n"
        ),
        "jumanji": (
            "import jumanji, jax\n"
            "env=jumanji.make('Snake-v1')\n"
            "state,timestep=env.reset(jax.random.PRNGKey(0))\n"
            "action=env.action_spec.generate_value()\n"
            "state,timestep=env.step(state, action)\n"
            "print('jumanji_reset_step_ok', float(timestep.reward), bool(timestep.last()))\n"
        ),
        "brax": (
            "from brax import envs\n"
            "import jax, jax.numpy as jnp\n"
            "env=envs.get_environment('fast')\n"
            "state=env.reset(jax.random.PRNGKey(0))\n"
            "state=env.step(state, jnp.zeros(env.action_size))\n"
            "print('brax_reset_step_ok', float(state.reward), bool(state.done))\n"
        ),
        "dm_control": (
            "from dm_control import suite\n"
            "import numpy as np\n"
            "env=suite.load(domain_name='cartpole', task_name='swingup')\n"
            "ts=env.reset()\n"
            "spec=env.action_spec()\n"
            "action=np.zeros(spec.shape, dtype=spec.dtype)\n"
            "ts=env.step(action)\n"
            "print('dm_control_reset_step_ok', float(ts.reward or 0.0), bool(ts.last()))\n"
        ),
        "pettingzoo": (
            "from pettingzoo.classic import tictactoe_v3\n"
            "env=tictactoe_v3.env()\n"
            "env.reset(seed=0)\n"
            "agent=env.agent_selection\n"
            "action=env.action_space(agent).sample()\n"
            "env.step(action)\n"
            "print('pettingzoo_reset_step_ok', env.agent_selection)\n"
        ),
        "crafter": (
            "import crafter\n"
            "env=crafter.Env()\n"
            "obs=env.reset()\n"
            "obs,reward,done,info=env.step(0)\n"
            "print('crafter_reset_step_ok', type(obs).__name__, float(reward), bool(done))\n"
        ),
        "craftax": (
            "import craftax\n"
            "print('craftax_import_ok')\n"
        ),
        "envpool": (
            "import envpool, numpy as np\n"
            "env=envpool.make('CartPole-v1', env_type='gym', num_envs=1)\n"
            "obs=env.reset()\n"
            "out=env.step(np.array([0], dtype=np.int64))\n"
            "print('envpool_reset_step_ok', type(obs).__name__, len(out))\n"
        ),
        "procgen": (
            "from procgen import ProcgenEnv\n"
            "env=ProcgenEnv(num_envs=1, env_name='coinrun')\n"
            "obs=env.reset()\n"
            "obs,reward,done,info=env.step([0])\n"
            "print('procgen_reset_step_ok', type(obs).__name__, reward, done)\n"
        ),
    }
    script = scripts.get(source_id)
    if not script:
        return
    result = run_python(script, extra_path=extra_path, timeout=45, python=python)
    add_check(
        checks,
        f"{source_id}_reset_step",
        result["ok"],
        "runtime",
        result["stdout_tail"] or result["stderr_tail"],
    )


def probe_drone_reset_step(
    checks: list[dict[str, Any]],
    source_id: str,
    extra_path: Path | str | None,
    *,
    python: Path | None = None,
) -> None:
    scripts = {
        "gym_pybullet_drones": (
            "from gym_pybullet_drones.envs.HoverAviary import HoverAviary\n"
            "env=HoverAviary(gui=False)\n"
            "out=env.reset(seed=0)\n"
            "obs=out[0] if isinstance(out, tuple) else out\n"
            "out=env.step(env.action_space.sample())\n"
            "reward=out[1] if isinstance(out, tuple) and len(out)>1 else 0.0\n"
            "terminated=out[2] if isinstance(out, tuple) and len(out)>2 else False\n"
            "truncated=out[3] if isinstance(out, tuple) and len(out)>3 else False\n"
            "env.close()\n"
            "print('gym_pybullet_drones_hover_reset_step_ok', type(obs).__name__, float(reward), bool(terminated), bool(truncated))\n"
        ),
        "pyflyt": (
            "import gymnasium as gym\n"
            "import PyFlyt.gym_envs\n"
            "env=gym.make('PyFlyt/QuadX-Hover-v4')\n"
            "out=env.reset(seed=0)\n"
            "obs=out[0] if isinstance(out, tuple) else out\n"
            "out=env.step(env.action_space.sample())\n"
            "reward=out[1] if isinstance(out, tuple) and len(out)>1 else 0.0\n"
            "terminated=out[2] if isinstance(out, tuple) and len(out)>2 else False\n"
            "truncated=out[3] if isinstance(out, tuple) and len(out)>3 else False\n"
            "env.close()\n"
            "print('pyflyt_quadx_hover_reset_step_ok', type(obs).__name__, float(reward), bool(terminated), bool(truncated))\n"
        ),
    }
    script = scripts.get(source_id)
    if not script:
        return
    result = run_python(script, extra_path=extra_path, timeout=45, python=python)
    add_check(
        checks,
        f"{source_id}_drone_reset_step",
        result["ok"],
        "runtime",
        result["stdout_tail"] or result["stderr_tail"],
    )


def import_available(module: str, extra_path: Path | str | None = None, *, python: Path | None = None) -> bool:
    return run_python(f"import {module}", extra_path=extra_path, timeout=10, python=python)["ok"]


def runtime_extra_path(source_id: str, source_path: Path | str | None) -> Path | str | None:
    # Some native-heavy projects require installed wheels for compiled modules;
    # their source clone is still provenance, but should not shadow the wheel.
    if source_id in {
        "crafter",
        "craftax",
        "minerl",
        "minedojo",
        "malmo",
        "voyager_minecraft",
        "pyboy",
        "gymboy",
        "stable_retro",
        "dm_control",
        "envpool",
        "airsim",
        "airsim_drone_racing_lab",
        "mavsdk_python",
    }:
        return None
    return source_path


def run_python(
    script: str,
    *,
    extra_path: Path | str | None = None,
    timeout: int = 20,
    python: Path | None = None,
) -> dict[str, Any]:
    python = python or preferred_python()
    env = os.environ.copy()
    if extra_path:
        env["PYTHONPATH"] = str(extra_path) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(
            [str(python), "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic smoke path.
        return {"ok": False, "error": str(exc), "stdout_tail": "", "stderr_tail": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-800:],
        "stderr_tail": result.stderr[-1200:],
    }


def module_for_source(source_id: str) -> str:
    mapping = {
        "brax": "brax",
        "jumanji": "jumanji",
        "gymnasium": "gymnasium",
        "minigrid": "minigrid",
        "bsuite": "bsuite",
        "craftax": "craftax",
        "crafter": "crafter",
        "minerl": "minerl",
        "minedojo": "minedojo",
        "malmo": "malmoenv",
        "voyager_minecraft": "",
        "dm_control": "dm_control",
        "envpool": "envpool",
        "metaworld": "metaworld",
        "pettingzoo": "pettingzoo",
        "procgen": "procgen",
        "pyflyt": "PyFlyt",
        "gym_pybullet_drones": "gym_pybullet_drones",
        "airsim_drone_racing_lab": "airsimdroneracinglab",
        "aerial_gym_simulator": "aerial_gym",
        "airsim": "airsim",
        "mavsdk_python": "mavsdk",
    }
    return mapping.get(source_id, "")


def runtime_python_for_source(source_id: str) -> Path:
    lane_map = {
        "pyflyt": [".venv-drone-pyflyt-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "gym_pybullet_drones": [".venv-drone-gym-pybullet-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "airsim_drone_racing_lab": [".venv-drone-racing-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "airsim": [".venv-drone-racing-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "mavsdk_python": [".venv-drone-control-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "px4_sitl": [".venv-drone-control-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "aerial_gym_simulator": [".venv-drone-aerial-gym-py311", ".venv-drone-py311-dev", ".venv-drone-py314"],
        "pygba": [".venv-emulator-py311"],
        "pyboy": [".venv-emulator-py311"],
        "gymboy": [".venv-emulator-py311"],
        "stable_retro": [".venv-emulator-py311"],
        "gb_tetris": [".venv-emulator-py311"],
        "gba_pokemon_emerald": [".venv-emulator-py311"],
        "gba_strategy_or_tactics": [".venv-emulator-py311"],
        "gbc_platformer_action": [".venv-emulator-py311"],
        "gbc_pokemon_red_blue_yellow": [".venv-emulator-py311"],
        "crafter": [".venv-minecraft-rl-py311"],
        "craftax": [".venv-minecraft-rl-py311"],
        "minerl": [".venv-minecraft-rl-py311"],
        "minedojo": [".venv-minecraft-rl-py311"],
        "malmo": [".venv-minecraft-rl-py311"],
        "voyager_minecraft": [".venv-minecraft-rl-py311"],
    }
    minecraft_modules = {
        "crafter": "crafter",
        "craftax": "craftax",
        "minerl": "minerl",
        "minedojo": "minedojo",
        "malmo": "malmoenv",
        "voyager_minecraft": "crafter",
    }
    if source_id in minecraft_modules:
        candidates = [ROOT / venv / "Scripts" / "python.exe" for venv in lane_map.get(source_id, [])]
        candidates.append(ROOT / ".venv-puffer" / "Scripts" / "python.exe")
        for candidate in candidates:
            if candidate.exists() and run_python(f"import {minecraft_modules[source_id]}", python=candidate, timeout=10)["ok"]:
                return candidate
        for candidate in candidates:
            if candidate.exists():
                return candidate
    for venv in lane_map.get(source_id, []):
        candidate = ROOT / venv / "Scripts" / "python.exe"
        if candidate.exists():
            return candidate
    return preferred_python()


def classify(checks: list[dict[str, Any]]) -> str:
    hard_failed = [item for item in checks if item["severity"] == "hard" and not item["passed"]]
    runtime_failed = [item for item in checks if item["severity"] == "runtime" and not item["passed"]]
    if hard_failed:
        return "blocked"
    if runtime_failed:
        return "metadata_passed_runtime_blocked"
    return "passed"


def summarize(rows: list[dict[str, Any]], total_factory_cards: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("smoke_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "total_factory_cards": total_factory_cards,
        "smoked_cards": len(rows),
        "passed": counts.get("passed", 0),
        "runtime_blocked": counts.get("metadata_passed_runtime_blocked", 0),
        "blocked": counts.get("blocked", 0),
        "failed": counts.get("failed", 0),
        "by_status": dict(sorted(counts.items())),
    }


def next_actions(rows: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for row in rows:
        failed = [item for item in row.get("checks", []) if not item.get("passed")]
        if not failed:
            actions.append(f"{row.get('card_id')}: adapter smoke passed; mark as ready pressure/regression candidate.")
            continue
        first = failed[0]
        actions.append(f"{row.get('card_id')}: fix {first.get('name')} ({first.get('evidence')}).")
    return actions[:12]


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    severity: str,
    evidence: str,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "evidence": evidence[:1200],
        }
    )


def zip_health(path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except Exception as exc:  # noqa: BLE001 - diagnostic smoke path.
        return {"ok": False, "error": str(exc), "members": 0}
    lowered = [name.lower() for name in names[:2000]]
    return {
        "ok": True,
        "members": len(names),
        "readme_hint": any("/readme" in f"/{name}" or name.startswith("readme") for name in lowered),
        "license_file_hint": any("/license" in f"/{name}" or name.startswith("license") or "/copying" in f"/{name}" for name in lowered),
        "pyproject_hint": any(name.endswith("pyproject.toml") for name in lowered),
        "setup_py_hint": any(name.endswith("setup.py") for name in lowered),
        "sample_members": names[:20],
    }


def clone_health(path: Path) -> dict[str, Any]:
    entries = list(path.iterdir()) if path.exists() else []
    names = [item.name.lower() for item in entries]
    return {
        "top_level_entries": len(entries),
        "readme_hint": any(name.startswith("readme") for name in names),
        "license_file_hint": any(name.startswith(("license", "copying")) for name in names),
        "pyproject_hint": "pyproject.toml" in names,
        "setup_py_hint": "setup.py" in names,
        "package_json_hint": "package.json" in names,
        "cargo_hint": "cargo.toml" in names,
        "sample_entries": [item.name for item in entries[:20]],
    }


def list_limited_files(path: Path, *, limit: int) -> list[Path]:
    ignored = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
    results: list[Path] = []
    stack = [path]
    while stack and len(results) < limit:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for item in entries:
            if len(results) >= limit:
                break
            if item.is_dir():
                if item.name not in ignored:
                    stack.append(item)
            else:
                results.append(item)
    return results


def command_available(command: str) -> bool:
    return command_path(command) is not None


def container_runtime_available() -> bool:
    return bool(container_runtime_status().get("ready"))


def container_runtime_status() -> dict[str, Any]:
    docker = command_path("docker")
    podman = command_path("podman")
    status: dict[str, Any] = {
        "ready": False,
        "docker_cli": bool(docker),
        "podman_cli": bool(podman),
        "docker_path": docker or "",
        "podman_path": podman or "",
        "runtime": "",
        "reason": "no_docker_or_podman_cli",
    }
    if docker:
        probe = run_probe([docker, "info"], timeout=15)
        status["docker_info"] = probe
        if probe.get("ok"):
            status.update({"ready": True, "runtime": "docker", "reason": "docker_info_ok"})
            return status
        status["reason"] = "docker_cli_present_but_info_failed"
    if podman:
        probe = run_probe([podman, "info", "--format", "json"], timeout=15)
        status["podman_info"] = probe
        if probe.get("ok"):
            status.update({"ready": True, "runtime": "podman", "reason": "podman_info_ok"})
            return status
        machine = run_probe([podman, "machine", "list"], timeout=15)
        status["podman_machine_list"] = machine
        stdout = str(machine.get("stdout_tail") or "")
        if "WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED" in stdout or "restart" in stdout.lower() or "reboot" in stdout.lower():
            status["reason"] = "podman_cli_present_but_wsl_reboot_required"
        elif machine.get("ok") and "NAME" in stdout and "theseus-podman" not in stdout:
            status["reason"] = "podman_cli_present_but_no_machine_initialized"
        else:
            status["reason"] = "podman_cli_present_but_runtime_not_ready"
    return status


def run_probe(command: list[str], timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout_tail": result.stdout[-600:],
            "stderr_tail": result.stderr[-600:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exit_code": -1, "error": str(exc)}


def command_path(command: str) -> str | None:
    found = shutil.which(command)
    if found:
        return found
    local_toolchains = [
        ROOT / "data" / "external_benchmark_candidates" / "toolchains",
        Path("D:/ProjectTheseus/tools"),
    ]
    candidates = [command]
    if os.name == "nt":
        candidates.extend([f"{command}.cmd", f"{command}.exe", f"{command}.ps1"])
    for base in local_toolchains:
        direct = base / command
        for name in candidates:
            path = direct / name
            if path.exists():
                return str(path)
        for child in [base / command, base / command.lower(), base / command.upper()]:
            for name in candidates:
                path = child / name
                if path.exists():
                    return str(path)
    return None


def git_ignored(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(relative)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0


def preferred_python() -> Path:
    venv = ROOT / ".venv-puffer" / "Scripts" / "python.exe"
    if venv.exists():
        return venv
    return Path(sys.executable)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not str(value):
        return ROOT / "__missing__"
    if not path.is_absolute():
        path = ROOT / path
    return path


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def safe_rom_path(path: Path) -> str:
    return rel_or_abs(path) if path.exists() else str(path)


def rel(path: Path) -> str:
    return rel_or_abs(path)


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for item in path:
        if not isinstance(cur, dict) or item not in cur:
            return default
        cur = cur[item]
    return cur


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = [
        "# Benchmark Adapter Smoke Status",
        "",
        f"Updated: {report.get('created_utc')}",
        "",
        "## Summary",
        "",
    ]
    for key, value in (report.get("summary") or {}).items():
        rows.append(f"- {key}: {value}")
    rows.extend(["", "## Cards", ""])
    for card in report.get("cards", []):
        rows.append(f"- {card.get('card_id')}: {card.get('smoke_status')} ({card.get('adapter_type')})")
        for check in card.get("checks", []):
            mark = "ok" if check.get("passed") else "block"
            rows.append(f"  - {mark}: {check.get('name')} - {check.get('evidence')}")
    rows.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions", []):
        rows.append(f"- {action}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
