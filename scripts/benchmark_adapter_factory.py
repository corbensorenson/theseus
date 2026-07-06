"""Create benchmark adapter cards from governed source and ROM registries.

The factory does not download data or run new benchmarks. It turns catalogued
sources into explicit benchmark cards with loader/scorer smoke plans, license
gates, runtime tiers, and regression policy so the autonomy loop can move to
new pressure surfaces without bespoke teacher intervention.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "benchmark_adapter_factory.json"
DEFAULT_CATALOG = ROOT / "reports" / "online_source_catalog_report.json"
DEFAULT_CURRICULUM = ROOT / "reports" / "benchmaxx_curriculum.json"
DEFAULT_ROMS = ROOT / "reports" / "local_rom_registry.json"
DEFAULT_PANTRY = ROOT / "reports" / "resource_pantry.json"
DEFAULT_SMOKE = ROOT / "reports" / "benchmark_adapter_smoke_status.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--catalog-report", default=str(DEFAULT_CATALOG.relative_to(ROOT)))
    parser.add_argument("--curriculum", default=str(DEFAULT_CURRICULUM.relative_to(ROOT)))
    parser.add_argument("--local-rom-registry", default=str(DEFAULT_ROMS.relative_to(ROOT)))
    parser.add_argument("--resource-pantry", default=str(DEFAULT_PANTRY.relative_to(ROOT)))
    parser.add_argument("--smoke-report", default=str(DEFAULT_SMOKE.relative_to(ROOT)))
    parser.add_argument("--write-cards", action="store_true")
    parser.add_argument("--out", default="reports/benchmark_adapter_factory.json")
    parser.add_argument("--markdown-out", default="reports/benchmark_adapter_factory.md")
    args = parser.parse_args()

    config = read_json(ROOT / args.config)
    catalog = read_json(ROOT / args.catalog_report)
    curriculum = read_json(ROOT / args.curriculum)
    rom_registry = read_json(ROOT / args.local_rom_registry)
    pantry = read_json(ROOT / args.resource_pantry)
    smoke_report = read_json(ROOT / args.smoke_report)
    card_root = ROOT / str(config.get("card_root", "benchmarks/cards"))

    smoke_by_id = smoke_status_by_card(smoke_report)
    source_cards, excluded_card_ids = build_source_cards(config, catalog, curriculum, pantry, smoke_by_id)
    rom_cards = build_rom_cards(config, rom_registry, smoke_by_id)
    cards = source_cards + rom_cards
    priority_order = {"highest": 0, "high": 1, "medium": 2, "low": 3}
    cards.sort(key=lambda card: (priority_order.get(str(card.get("priority")), 9), card["id"]))

    written: list[str] = []
    deleted_excluded_cards: list[str] = []
    if args.write_cards:
        card_root.mkdir(parents=True, exist_ok=True)
        for card in cards:
            if not card.get("write_card", True):
                continue
            path = card_root / f"{safe_id(card['id'])}.json"
            file_card = dict(card)
            file_card.pop("last_smoke", None)
            write_json(path, file_card)
            written.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        if get_path(config, ["card_write_policy", "delete_excluded_cards"], False):
            for card_id in sorted(set(excluded_card_ids)):
                path = card_root / f"{safe_id(card_id)}.json"
                if path.exists():
                    path.unlink()
                    deleted_excluded_cards.append(str(path.relative_to(ROOT)).replace("\\", "/"))

    summary = {
        "sources_seen": len(catalog.get("sources", [])) if isinstance(catalog.get("sources"), list) else 0,
        "excluded_sources": len(excluded_card_ids),
        "cards": len(cards),
        "ready_cards": len([card for card in cards if card.get("status") in {"adapter_card_ready", "adapter_smoke_passed"}]),
        "smoke_passed": count_where(cards, "status", "adapter_smoke_passed"),
        "needs_smoke": count_where(cards, "status", "needs_adapter_smoke"),
        "blocked": len([card for card in cards if str(card.get("status", "")).startswith("blocked")]),
        "emulator_cards": len([card for card in cards if card.get("adapter_type") == "emulator_rl_adapter"]),
        "written_cards": len(written),
        "deleted_excluded_cards": len(deleted_excluded_cards),
    }
    report = {
        "policy": "sparkstream_benchmark_adapter_factory_report_v0",
        "created_utc": now(),
        "config": args.config,
        "catalog_report": args.catalog_report,
        "curriculum": args.curriculum,
        "local_rom_registry": args.local_rom_registry,
        "summary": summary,
        "cards": cards,
        "excluded_card_ids": sorted(set(excluded_card_ids)),
        "written_cards": written,
        "deleted_excluded_cards": deleted_excluded_cards,
        "next_actions": next_actions(cards),
        "resource_pantry": args.resource_pantry,
        "smoke_report": args.smoke_report,
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    write_markdown(ROOT / args.markdown_out, report)
    print(json.dumps(report, indent=2))
    return 0


def build_source_cards(
    config: dict[str, Any],
    catalog: dict[str, Any],
    curriculum: dict[str, Any],
    pantry: dict[str, Any],
    smoke_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    templates = config.get("adapter_templates") or {}
    overrides = config.get("source_overrides") or {}
    allowed = set(get_path(config, ["license_gate", "allowed_decisions"], []))
    pantry_by_id = {
        str(item.get("id") or ""): item
        for item in pantry.get("sources", [])
        if isinstance(item, dict)
    }
    cards: list[dict[str, Any]] = []
    excluded_card_ids: list[str] = []
    current_stage = get_path(curriculum, ["summary", "current_stage_id"], "")
    next_family = get_path(curriculum, ["summary", "next_frontier_family"], "")
    for source in catalog.get("sources", []) if isinstance(catalog.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("id") or source.get("name") or "source")
        card_id = f"source_{safe_id(source_id)}"
        pantry_row = pantry_by_id.get(source_id, {})
        override = overrides.get(source_id, {})
        category = canonical_category(str(override.get("category") or source.get("category") or "language_benchmark_framework"))
        template = templates.get(category) or templates.get("language_benchmark_framework", {})
        decision = str(source.get("decision") or "")
        if source_excluded_from_active_cards(config, source, decision):
            excluded_card_ids.append(card_id)
            continue
        license_allowed = bool(source.get("license_allowed") or decision in allowed)
        source_staged_path = existing_local_path(source.get("staged_path")) if license_allowed else ""
        pantry_clone_path = existing_local_path(pantry_row.get("clone_path")) if pantry_row.get("present") else ""
        staged_path = source_staged_path or pantry_clone_path
        source_staged = bool(source.get("staged")) and bool(source_staged_path)
        pantry_present = bool(pantry_row.get("present")) and bool(pantry_clone_path)
        imported = source_staged or pantry_present or decision in allowed
        if not license_allowed:
            status = "blocked_pending_license"
        elif not imported:
            status = "blocked_pending_source_stage"
        else:
            status = "needs_adapter_smoke"
        priority = str(override.get("priority") or source.get("priority") or "medium")
        card = {
            "schema": "sparkstream_benchmark_card_v0",
            "id": card_id,
            "source_id": source_id,
            "name": source.get("name") or source_id,
            "category": category,
            "priority": priority,
            "status": status,
            "url": source.get("url"),
            "license_spdx": source.get("license_spdx"),
            "license_allowed": license_allowed,
            "decision": decision,
            "staged": source_staged or pantry_present,
            "staged_path": staged_path,
            "resource_pantry_path": pantry_clone_path,
            "adapter_type": template.get("adapter_type"),
            "runner_family": template.get("runner_family"),
            "runtime_tier": template.get("runtime_tier", "E2"),
            "risk_tier": template.get("risk_tier", "low"),
            "capability_target": capability_target(category),
            "current_curriculum_stage": current_stage,
            "next_frontier_family": next_family,
            "input_contract": input_contract(category),
            "output_contract": output_contract(category),
            "smoke_steps": source.get("smoke_plan") or template.get("smoke_steps", []),
            "promotion_gates": template.get("promotion_gates", []),
            "contamination_policy": contamination_policy(category),
            "regression_policy": "If mastered, demote to regression suite and preserve any failures in residual escrow.",
            "permission_envelope": permission_envelope({**template, "category": category}),
            "teacher_role": "Only diagnose adapter/source bugs after local smoke fails; do not score or solve benchmark items.",
            "external_inference_calls": 0,
        }
        apply_smoke_status(card, smoke_by_id)
        cards.append(card)
    return cards, excluded_card_ids


def source_excluded_from_active_cards(config: dict[str, Any], source: dict[str, Any], decision: str) -> bool:
    exclude_decisions = set(str(item) for item in get_path(config, ["license_gate", "exclude_decisions"], []))
    if decision in exclude_decisions:
        return True
    if not get_path(config, ["license_gate", "exclude_non_permissive_or_uncleared_sources"], False):
        return False
    if not bool(source.get("license_allowed")):
        return True
    return str(source.get("import_policy") or "") == "queue_only"


def build_rom_cards(
    config: dict[str, Any],
    registry: dict[str, Any],
    smoke_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    templates = config.get("adapter_templates") or {}
    template = templates.get("emulator_rl_environment", {})
    cards: list[dict[str, Any]] = []
    recommendations = registry.get("recommendations", []) if isinstance(registry.get("recommendations"), list) else []
    for rec in recommendations:
        if not isinstance(rec, dict):
            continue
        matched = rec.get("matched_rom_count", 0) or 0
        profile_id = str(rec.get("profile_id") or "rom_profile")
        status = "needs_adapter_smoke" if matched else "blocked_missing_user_supplied_rom"
        card = {
            "schema": "sparkstream_benchmark_card_v0",
            "id": f"local_rom_{safe_id(profile_id)}",
            "source_id": profile_id,
            "name": f"Local ROM RL profile: {profile_id}",
            "category": "emulator_rl_environment",
            "priority": rec.get("priority", "medium"),
            "status": status,
            "url": "",
            "license_spdx": "user-supplied-private-asset",
            "license_allowed": bool(matched),
            "decision": "local_user_supplied_private_rom" if matched else "awaiting_user_asset",
            "staged": bool(matched),
            "staged_path": "",
            "adapter_type": template.get("adapter_type", "emulator_rl_adapter"),
            "runner_family": template.get("runner_family", "emulator_rl_local"),
            "runtime_tier": template.get("runtime_tier", "E3"),
            "risk_tier": template.get("risk_tier", "medium"),
            "capability_target": "pixel/control RL, exploration, planning, and delayed reward using user-owned local game assets.",
            "input_contract": {
                "rom_profile_id": profile_id,
                "rom_resolution": "runtime resolves ignored ROM path from reports/local_rom_registry.json",
                "actions": "adapter-defined discrete controls",
            },
            "output_contract": {
                "score": "episodic_reward_or_task_progress",
                "trace": "eventized emulator steps",
                "residuals": "stalls, deaths, loops, invalid actions, wrapper failures",
            },
            "smoke_steps": template.get("smoke_steps", []),
            "promotion_gates": template.get("promotion_gates", []),
            "contamination_policy": "No ROM bytes or hashes are tracked in cards; only ignored local registry resolves private assets.",
            "regression_policy": "Promote stable wrapper tasks to regression after deterministic smoke and shaped reward validation.",
            "permission_envelope": permission_envelope({**template, "category": "emulator_rl_environment"}),
            "recommended_adapter": rec.get("adapter"),
            "next_step": rec.get("next_step"),
            "external_inference_calls": 0,
        }
        apply_smoke_status(card, smoke_by_id)
        cards.append(card)
    return cards


def smoke_status_by_card(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("cards", []) if isinstance(report.get("cards"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        card_id = str(row.get("card_id") or row.get("id") or "")
        if card_id:
            out[card_id] = row
    return out


def apply_smoke_status(card: dict[str, Any], smoke_by_id: dict[str, dict[str, Any]]) -> None:
    smoke = smoke_by_id.get(str(card.get("id") or ""))
    if not smoke:
        return
    status = str(smoke.get("smoke_status") or "")
    card["last_smoke"] = {
        "status": status,
        "created_utc": smoke.get("created_utc"),
        "report_path": smoke.get("report_path"),
        "passed": smoke.get("passed"),
        "blocked": smoke.get("blocked"),
    }
    if status == "passed":
        card["status"] = "adapter_smoke_passed"
    elif status == "metadata_passed_runtime_blocked":
        card["status"] = "blocked_runtime_dependency"
    elif status in {"failed", "blocked"}:
        card["status"] = "blocked_adapter_smoke_failed"


def capability_target(category: str) -> str:
    mapping = {
        "training_data": "governed data ingestion, leakage-safe sampling, and residual-targeted learning pressure",
        "rl_benchmark": "reinforcement learning diagnostics for memory, exploration, and credit assignment",
        "rl_environment": "open-ended RL control and planning",
        "emulator_rl_environment": "game-based RL control and long-horizon planning",
        "emulator_runtime_dependency": "native emulator runtime support for user-supplied local game RL assets",
        "minecraft_rl_environment": "licensed local Minecraft and Minecraft-like open-world RL, crafting, navigation, and player-coop pressure",
        "drone_rl_environment": "simulation-first drone control, hover, waypoint, perception, and racing policy pressure",
        "drone_racing_simulator": "AI Grand Prix-style drone racing SITL, vision, telemetry, and MAVLink control pressure",
        "drone_control_api": "governed MAVLink/MAVSDK/PX4 control interfaces with simulator-first safety gates",
        "coding_benchmark": "local coding, repair, and unit-test reasoning",
        "coding_agent_benchmark": "repo-level coding-agent evaluation, issue-to-patch behavior, terminal/tool use, and self-debugging pressure",
        "coding_agent_framework": "local coding-agent harness integration without provider inference",
        "tool_use_benchmark": "function/tool-call fidelity, stateful tool use, and argument/schema robustness",
        "tool_dialogue_agent_benchmark": "multi-turn policy-constrained tool use with user/environment simulation",
        "web_agent_benchmark": "browser/web task planning in self-hosted sandboxes",
        "desktop_agent_benchmark": "desktop action planning in VM/sandboxed environments",
        "voice_benchmark": "native local speech and audio interaction",
        "long_context_memory_benchmark": "VCM context recovery, long-context retrieval/reasoning, temporal/session memory, evidence grounding, and stale/deletion rejection",
    }
    return mapping.get(category, "language, reasoning, and evaluation pressure")


def canonical_category(category: str) -> str:
    aliases = {
        "physics_rl_environment": "rl_environment",
        "robotics_rl_environment": "rl_environment",
        "multiagent_rl_environment": "rl_environment",
        "rl_runtime_acceleration": "rl_environment",
        "minecraft_environment": "minecraft_rl_environment",
        "minecraft_benchmark": "minecraft_rl_environment",
        "minecraft_like_rl_environment": "minecraft_rl_environment",
        "uav_rl_environment": "drone_rl_environment",
        "quadrotor_rl_environment": "drone_rl_environment",
        "drone_simulator": "drone_racing_simulator",
        "uav_simulator": "drone_racing_simulator",
        "mavlink_control_api": "drone_control_api",
        "voice_benchmark_framework": "voice_benchmark",
        "voice_benchmark_data": "voice_benchmark",
        "voice_training_data": "training_data",
        "tool_dialogue_benchmark": "tool_dialogue_agent_benchmark",
        "tool_agent_benchmark": "tool_dialogue_agent_benchmark",
        "function_calling_benchmark": "tool_use_benchmark",
        "browser_agent_benchmark": "web_agent_benchmark",
        "long_context_benchmark": "long_context_memory_benchmark",
        "context_recovery_benchmark": "long_context_memory_benchmark",
        "memory_benchmark": "long_context_memory_benchmark",
    }
    return aliases.get(category, category)


def input_contract(category: str) -> dict[str, Any]:
    if category == "training_data":
        return {"metadata": "dataset card", "sample": "tiny governed stream", "eval_overlap": "required"}
    if category in {"rl_benchmark", "rl_environment"}:
        return {"env_id": "local environment id", "seed": "deterministic seed", "policy": "local learner only"}
    if category == "emulator_runtime_dependency":
        return {
            "source": "staged native/runtime source",
            "python_runtime": "isolated emulator runtime",
            "symbols": "required CFFI imports and core symbols",
        }
    if category == "minecraft_rl_environment":
        return {
            "env_id": "local Minecraft/Minecraft-like environment id",
            "seed": "deterministic seed",
            "policy": "local Theseus learner/controller only",
            "world": "disposable local world or open-source bridge world",
            "license": "user-owned local Minecraft runtime required for full Minecraft harnesses",
        }
    if category == "drone_rl_environment":
        return {
            "env_id": "simulation-only drone environment id",
            "seed": "deterministic seed",
            "policy": "local learner/controller only",
            "safety": "no live hardware endpoint",
        }
    if category == "drone_racing_simulator":
        return {
            "sim_endpoint": "localhost/UDP SITL endpoint",
            "vision_stream": "30 Hz packetized camera stream when simulator is running",
            "telemetry": "MAVLink attitude/IMU/time messages",
            "control": "MAVLink setpoint commands under rate limits",
        }
    if category == "drone_control_api":
        return {
            "endpoint": "simulator or loopback endpoint",
            "commands": "arm/disarm/takeoff/position/attitude setpoints",
            "approval": "required for any live hardware target",
        }
    if category == "coding_benchmark":
        return {"prompt": "task statement", "tests": "local unit tests", "sandbox": "required"}
    if category == "coding_agent_benchmark":
        return {
            "repo": "sandboxed repository or task fixture",
            "issue": "task statement or bug report",
            "tests": "local verification command",
            "agent": "local Theseus/OpenAI-compatible endpoint only",
        }
    if category == "coding_agent_framework":
        return {
            "harness": "local agent framework source",
            "endpoint": "local Theseus/OpenAI-compatible endpoint contract",
            "sandbox": "required for any repo mutation",
            "providers": "external provider calls disabled during scoring",
        }
    if category == "tool_use_benchmark":
        return {
            "prompt_or_dialogue": "public calibration item or private holdout",
            "tool_schema": "available function/tool definitions",
            "state": "local synthetic/tool simulator state when applicable",
            "agent": "local Theseus/OpenAI-compatible endpoint only",
        }
    if category == "tool_dialogue_agent_benchmark":
        return {
            "user_goal": "multi-turn task objective",
            "policy": "domain constraints and allowed actions",
            "tools": "local API/tool simulator only",
            "agent": "local Theseus/OpenAI-compatible endpoint only",
        }
    if category == "long_context_memory_benchmark":
        return {
            "metadata": "public benchmark card or private analogue manifest",
            "context": "long local/private semantic pages, sessions, or document chunks",
            "query": "context recovery, multi-hop, temporal update, or abstention question",
            "evidence": "required page ids and provenance, never public answer templates for training",
            "memory_system": "VCM or baseline memory resolver under equal context budget",
        }
    return {"examples": "local benchmark items or metadata", "model": "local checkpoint or SymLiquid scorer"}


def output_contract(category: str) -> dict[str, Any]:
    if category == "training_data":
        return {"accepted_rows": "count", "rejections": "reasons", "leakage": "overlap report"}
    if category in {"rl_benchmark", "rl_environment", "emulator_rl_environment"}:
        return {"reward": "episode score", "steps": "count", "events": "event trace", "residuals": "failure clusters"}
    if category == "emulator_runtime_dependency":
        return {"runtime_ready": "bool", "symbols": "available or missing symbols", "residuals": "native build/import failures"}
    if category == "minecraft_rl_environment":
        return {
            "reward": "episode/goal score",
            "steps": "environment steps",
            "inventory": "inventory and crafting events",
            "world_events": "blocks, mobs, damage, navigation, goal progress",
            "residuals": "navigation, survival, crafting, recovery, and player-instruction failures",
        }
    if category in {"drone_rl_environment", "drone_racing_simulator"}:
        return {
            "reward": "episode/race score",
            "gates": "gates passed or waypoint progress",
            "safety_events": "collisions, boundary hits, failsafe triggers",
            "latency": "control and vision timing",
            "residuals": "failure clusters",
        }
    if category == "drone_control_api":
        return {"connection": "sim-only connection health", "rate": "command/heartbeat timing", "safety": "approval/failsafe status"}
    if category == "coding_benchmark":
        return {"pass_rate": "tests passed", "runtime": "cost", "residuals": "failed tasks"}
    if category == "coding_agent_benchmark":
        return {
            "pass_rate": "repo tasks passed",
            "patches": "accepted or rejected local diffs",
            "tool_trace": "terminal/file/action trace packets",
            "residuals": "failed issue, test, planning, and tool-use clusters",
        }
    if category == "coding_agent_framework":
        return {
            "harness_ready": "bool",
            "local_endpoint_ready": "bool",
            "forbidden_provider_paths": "audited provider/API-key paths",
            "residuals": "runtime and sandbox integration blockers",
        }
    if category == "tool_use_benchmark":
        return {
            "call_accuracy": "schema/argument/function selection score",
            "trace": "tool-call and validation packets",
            "state_errors": "state dependency or canonicalization failures",
            "residuals": "tool selection, argument, format, and policy clusters",
        }
    if category == "tool_dialogue_agent_benchmark":
        return {
            "task_success": "completed user objective under policy",
            "policy_adherence": "violations or safe refusals",
            "tool_trace": "ordered local API/tool calls",
            "residuals": "dialogue state, tool, policy, and recovery clusters",
        }
    if category == "long_context_memory_benchmark":
        return {
            "answer_accuracy": "exact or verifier-scored response correctness",
            "evidence_precision_recall": "selected evidence page quality under budget",
            "abstention": "unknown, stale, and deleted-memory rejection behavior",
            "latency": "local retrieval and graph traversal cost",
            "residuals": "missed retrieval, stale page use, bad update handling, or insufficient evidence",
        }
    return {"score": "accuracy or task metric", "residuals": "failed cases", "cost": "runtime and memory"}


def contamination_policy(category: str) -> str:
    if category == "training_data":
        return "Run exact-pair and sentence-level overlap checks against every private/frontier eval before training use."
    if category in {
        "language_benchmark_framework",
        "language_benchmark_suite",
        "coding_benchmark",
        "coding_agent_benchmark",
        "coding_agent_framework",
        "tool_use_benchmark",
        "tool_dialogue_agent_benchmark",
        "long_context_memory_benchmark",
    }:
        if category == "long_context_memory_benchmark":
            return "Public long-context/memory benchmarks are calibration metadata only. Training pressure must come from private analogues, dogfood traces, or governed teacher rows; never train on public prompts, contexts, answers, traces, or templates."
        return "Public benchmark is calibration/diagnostic unless private holdouts or live variants are generated."
    if category.startswith("drone_"):
        return "Treat public simulator tasks as calibration until private tracks, held-out seeds, and race-specific course variants are generated."
    if category == "minecraft_rl_environment":
        return "Treat public Minecraft tasks as calibration. Keep local disposable worlds and held-out seeds separate from any training traces."
    return "Record source provenance and keep generated frontiers separate from training samples."


def permission_envelope(template: dict[str, Any]) -> dict[str, Any]:
    tier = template.get("runtime_tier", "E2")
    category = str(template.get("category") or "")
    side_effects = ["write_reports", "write_adapter_cards"]
    approval_required = ["production", "financial", "legal", "security_sensitive", "destructive"]
    hardware = "not_applicable"
    if category == "minecraft_rl_environment":
        side_effects.extend(["write_disposable_world_metadata", "write_gameplay_trace_logs"])
        approval_required.extend(["public_server_access", "account_or_credential_access", "persistent_world_mutation"])
    if category.startswith("drone_"):
        side_effects.extend(["open_simulator_udp_loopback", "write_drone_event_logs"])
        approval_required.extend(["live_drone_endpoint", "arm_or_takeoff_command", "hardware_in_loop"])
        hardware = "forbidden_without_explicit_human_approval"
    if category in {"coding_agent_benchmark", "coding_agent_framework"}:
        side_effects.extend(["write_sandbox_repo_copies", "write_patch_artifacts", "write_tool_trace_logs"])
        approval_required.extend(["provider_api_key_use", "network_during_scoring", "host_repo_mutation"])
    if category in {"tool_use_benchmark", "tool_dialogue_agent_benchmark"}:
        side_effects.extend(["write_tool_trace_logs", "write_policy_simulator_reports"])
        approval_required.extend(["provider_api_key_use", "network_during_scoring", "live_customer_or_account_data"])
    if category == "long_context_memory_benchmark":
        side_effects.extend(["write_vcm_benchmark_reports", "write_private_context_recovery_residuals"])
        approval_required.extend(["public_calibration_unlock", "bulk_dataset_download", "public_data_training_use"])
    return {
        "memory": "benchmark-local metadata and task context only",
        "tools": "local loaders, scorers, sandboxes, and report writers",
        "runtime": tier,
        "side_effects": side_effects,
        "network": "forbidden during scoring; catalog import only through license-gated source tools",
        "external_inference": "forbidden",
        "hardware": hardware,
        "approval_required_for": sorted(set(approval_required)),
    }


def next_actions(cards: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for card in cards:
        status = str(card.get("status"))
        if status == "needs_adapter_smoke":
            actions.append(f"Build/run smoke for {card.get('id')} using {card.get('adapter_type')}.")
        elif status == "adapter_card_ready":
            actions.append(f"Queue {card.get('id')} for smallest local smoke before frontier use.")
        if len(actions) >= 12:
            break
    if not actions:
        actions.append("No unsmoked adapter cards remain; use smoke-passed cards as pressure runners and keep excluded license/terms-unclear sources out of active rotation.")
    return actions


def count_where(rows: list[dict[str, Any]], key: str, value: str) -> int:
    return len([row for row in rows if row.get(key) == value])


def existing_local_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    candidate = path if path.is_absolute() else ROOT / path
    if not candidate.exists():
        return ""
    try:
        return str(candidate.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(candidate)


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")[:96] or "item"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = [
        "# Benchmark Adapter Factory",
        "",
        f"Updated: {report.get('created_utc')}",
        "",
        "## Summary",
        "",
    ]
    for key, value in (report.get("summary") or {}).items():
        rows.append(f"- {key}: {value}")
    rows.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions", []):
        rows.append(f"- {action}")
    rows.extend(["", "## Cards", ""])
    for card in report.get("cards", [])[:40]:
        rows.append(
            f"- {card.get('id')}: {card.get('status')} ({card.get('adapter_type')}, {card.get('priority')})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
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
