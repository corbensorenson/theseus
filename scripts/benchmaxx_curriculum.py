"""Build the live Benchmaxx curriculum report.

The curriculum is the long-range course map for SparkStream/RMI. It answers:

- what capability stage the system is currently attacking;
- what benchmarks and source candidates define later stages;
- what is locked as regression, active as frontier, ready next, or blocked;
- when the teacher may be used.

It does not run model inference and does not fetch data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmaxx_curriculum_io import (
    get_path,
    now,
    number,
    read_json,
    render_markdown,
    write_json,
    write_text,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "benchmaxx_curriculum.json"
DEFAULT_OUT = ROOT / "reports" / "benchmaxx_curriculum.json"
DEFAULT_MARKDOWN_OUT = ROOT / "reports" / "benchmaxx_curriculum.md"
DEFAULT_CODING_PRESSURE_CARD_ORDER = [
    "source_bigcodebench",
    "source_evalplus",
    "source_human_eval",
    "source_mbpp",
    "source_livecodebench",
    "multistream_code_repair_pressure",
    "synthetic_code_repair_mutation",
    "synthetic_cross_arm_code_memory",
    "synthetic_cross_arm_tool_safety",
    "synthetic_cross_domain_code_rl_trace",
    "source_opencode",
    "source_swe_bench",
    "source_swe_agent",
    "source_mini_swe_agent",
    "source_codeclash",
    "source_swe_polybench",
    "source_swe_gen",
    "old_registry_benchmark_corben_coding_agent_native_v1",
    "old_registry_benchmark_corben_task_execution_native_v1",
    "old_registry_benchmark_corben_shortcut_resistant_compositional_v1",
]
PUBLIC_CODE_CALIBRATION_CARDS = {
    "source_bigcodebench",
    "source_evalplus",
    "source_human_eval",
    "source_mbpp",
    "source_livecodebench",
    "source_opencode",
    "source_opencode_bench",
    "source_swe_bench",
    "source_swe_agent",
    "source_mini_swe_agent",
    "source_codeclash",
    "source_swe_polybench",
    "source_swe_gen",
}
PUBLIC_CODE_TASK_CARDS = {
    "source_evalplus",
    "source_human_eval",
    "source_mbpp",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    config = read_json(ROOT / args.config, {})
    reports = load_reports()
    stages = build_stages(config, reports)
    current = current_stage(stages, config)
    next_frontier = next_frontier_plan(current, stages, reports, config)
    payload = {
        "policy": "sparkstream_benchmaxx_curriculum_report_v0",
        "created_utc": now(),
        "config": str(Path(args.config)).replace("\\", "/"),
        "summary": summary(stages, current, next_frontier),
        "teacher_role": config.get("teacher_role", {}),
        "global_rules": config.get("global_rules", {}),
        "current_stage": current,
        "next_frontier": next_frontier,
        "ordered_course": stages,
        "near_term_queue": near_term_queue(stages, current),
        "course_manifesto": [
            "Do not rerun saturated benchmarks as frontier pressure.",
            "Graduate at mastery, escrow the tail, and move to harder capability surfaces.",
            "Use the teacher to audit and diagnose walls, not to solve benchmark items.",
            "Grow toward native voice, web, desktop, and tool-use capability through local benchmarks.",
            "When same-family code rotation cannot clear public transfer, interleave broader transfer surfaces and return with artifacts loaded.",
        ],
    }
    write_json(ROOT / args.out, payload)
    write_text(ROOT / args.markdown_out, render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0


def load_reports() -> dict[str, Any]:
    reports = ROOT / "reports"
    return {
        "architecture_gate": read_json(reports / "architecture_gate_report.json", {}),
        "adapter_factory": read_json(reports / "benchmark_adapter_factory.json", {}),
        "ai_grand_prix_spec": read_json(reports / "ai_grand_prix_spec_digest.json", {}),
        "benchmark_ledger": read_json(reports / "benchmark_ledger.json", []),
        "candidate_gate": read_json(reports / "candidate_promotion_gate.json", {}),
        "capability_matrix": read_json(reports / "capability_matrix.json", {}),
        "code_residual_forge": read_json(reports / "code_residual_forge.json", {}),
        "code_frontier_rotation": read_json(reports / "code_frontier_rotation.json", {}),
        "code_transfer_artifacts": read_json(reports / "code_transfer_artifacts.json", {}),
        "real_code_benchmark_graduation": read_json(reports / "real_code_benchmark_graduation.json", {}),
        "broad_transfer_matrix": read_json(reports / "broad_transfer_matrix.json", {}),
        "external_inference_audit": read_json(reports / "external_inference_audit.json", {}),
        "frontier_policy": read_json(reports / "frontier_policy_status.json", {}),
        "local_rom_registry": read_json(reports / "local_rom_registry.json", {}),
        "minecraft_runtime_probe": read_json(reports / "minecraft_runtime_probe.json", {}),
        "online_source_catalog": read_json(reports / "online_source_catalog_report.json", {}),
        "python_runtime_compatibility": read_json(reports / "python_runtime_compatibility.json", {}),
        "residual_escrow": read_json(reports / "residual_escrow.json", {}),
        "rl_registry": read_json(reports / "rl_benchmark_registry.json", {}),
        "synthetic_data": read_json(reports / "synthetic_data_curator.json", {}),
        "synthetic_benchmark_factory": read_json(reports / "synthetic_benchmark_factory.json", {}),
        "multi_stream_trace_factory": read_json(reports / "multi_stream_trace_factory.json", {}),
        "old_project_registry_port": read_json(reports / "old_project_registry_port.json", {}),
        "open_conversation_pantry": read_json(reports / "open_conversation_training_pantry.json", {}),
        "multi_turn_conversation": read_json(reports / "multi_turn_conversation_benchmark.json", {}),
        "tool_registry": read_json(reports / "tool_registry.json", {}),
        "transfer_eval_suite": read_json(reports / "transfer_eval_suite.json", {}),
    }


def build_stages(config: dict[str, Any], reports: dict[str, Any]) -> list[dict[str, Any]]:
    source_index = source_index_from_reports(reports)
    ledger = reports.get("benchmark_ledger") if isinstance(reports.get("benchmark_ledger"), list) else []
    stages: list[dict[str, Any]] = []
    prior_locked = True
    for raw_stage in config.get("stages", []):
        if not isinstance(raw_stage, dict):
            continue
        matched = matching_benchmarks(raw_stage, ledger)
        source_rows = [source_index.get(str(source_id), source_stub(source_id)) for source_id in raw_stage.get("source_ids", [])]
        source_summary = summarize_sources(source_rows)
        status, blockers = stage_status(raw_stage, matched, source_summary, reports, prior_locked)
        score = stage_score(status, matched, source_summary, reports, raw_stage)
        stage = {
            "id": raw_stage.get("id"),
            "level": raw_stage.get("level"),
            "title": raw_stage.get("title"),
            "status": status,
            "readiness_score": score,
            "capability_goal": raw_stage.get("capability_goal"),
            "promotion_gate": raw_stage.get("promotion_gate"),
            "next_frontier_family": raw_stage.get("next_frontier_family"),
            "teacher_policy": raw_stage.get("teacher_policy"),
            "benchmark_patterns": raw_stage.get("benchmark_patterns", []),
            "matched_benchmarks": matched,
            "sources": source_rows,
            "source_summary": source_summary,
            "blockers": blockers,
            "recommended_rom_profiles": raw_stage.get("recommended_rom_profiles", []),
            "next_action": stage_next_action(raw_stage, status, matched, source_summary, reports, blockers),
        }
        stages.append(stage)
        prior_locked = prior_locked and status == "locked_regression"
    return stages


def source_index_from_reports(reports: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = reports.get("online_source_catalog") if isinstance(reports.get("online_source_catalog"), dict) else {}
    rows: dict[str, dict[str, Any]] = {}
    for row in catalog.get("sources", []):
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("id") or "")
        if not source_id:
            continue
        rows[source_id] = {
            "id": source_id,
            "name": row.get("name"),
            "category": row.get("category"),
            "decision": row.get("decision"),
            "import_policy": row.get("import_policy"),
            "license_spdx": row.get("license_spdx"),
            "staged": bool(row.get("staged")),
            "risk": row.get("risk"),
            "url": row.get("url"),
        }
    return rows


def source_stub(source_id: Any) -> dict[str, Any]:
    return {
        "id": str(source_id),
        "name": str(source_id),
        "decision": "not_in_online_source_catalog",
        "staged": False,
        "license_spdx": "unknown",
    }


def summarize_sources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "approved": len([row for row in rows if row.get("decision") == "approved_for_catalog_import"]),
        "staged": len([row for row in rows if row.get("staged")]),
        "blocked": len([row for row in rows if "blocked" in str(row.get("decision") or "")]),
        "queued_or_missing": len(
            [
                row
                for row in rows
                if row.get("decision") in {"queued_only", "not_in_online_source_catalog"}
            ]
        ),
    }


def matching_benchmarks(stage: dict[str, Any], ledger: list[Any]) -> list[dict[str, Any]]:
    patterns = [str(item).lower() for item in stage.get("benchmark_patterns", [])]
    rows: list[dict[str, Any]] = []
    for item in ledger:
        if not isinstance(item, dict):
            continue
        haystack = " ".join(
            str(item.get(key) or "")
            for key in [
                "benchmark_name",
                "benchmark_type",
                "capability_measured",
                "best_report",
            ]
        ).lower()
        if any(pattern and pattern in haystack for pattern in patterns):
            rows.append(
                {
                    "benchmark_name": item.get("benchmark_name"),
                    "lifecycle": item.get("lifecycle"),
                    "score": item.get("score"),
                    "residual": item.get("residual"),
                    "wall_type": item.get("wall_type"),
                    "current_threshold": get_path(item, ["graduation_policy", "current_threshold"], None),
                    "best_report": item.get("best_report"),
                }
            )
    return rows


def stage_status(
    stage: dict[str, Any],
    matched: list[dict[str, Any]],
    source_summary: dict[str, Any],
    reports: dict[str, Any],
    prior_locked: bool,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    stage_id = str(stage.get("id") or "")
    if not prior_locked:
        return "future", ["prior stage not locked yet"]
    if active_count(matched):
        return "active_frontier", []
    if locked_by_reports(stage_id, matched, reports):
        return "locked_regression", []
    if stage.get("requires_local_rom"):
        rom_ready = bool(get_path(reports, ["local_rom_registry", "summary", "ready_for_wrapper_smoke"], False))
        if not rom_ready:
            blockers.append("no matching user-supplied local ROM profile is ready for wrapper smoke")
            return "blocked_waiting_asset", blockers
    if stage.get("requires_local_minecraft_license"):
        install_ready = bool(get_path(reports, ["minecraft_runtime_probe", "summary", "local_minecraft_install_detected"], False))
        bridge_ready = bool(get_path(reports, ["minecraft_runtime_probe", "summary", "bridge_runtime_ready"], False))
        if not install_ready and not bridge_ready and source_summary.get("approved", 0) == 0:
            blockers.append("no local Minecraft runtime or open-source Minecraft-like bridge is ready")
            return "blocked_waiting_asset", blockers
    if source_summary.get("approved", 0) or source_summary.get("staged", 0):
        return "ready_next", []
    if source_summary.get("blocked", 0) or source_summary.get("queued_or_missing", 0):
        blockers.append("source candidates need license, metadata, or adapter audit")
        return "blocked_waiting_asset", blockers
    return "ready_next" if not stage.get("source_ids") else "future", blockers


def locked_by_reports(stage_id: str, matched: list[dict[str, Any]], reports: dict[str, Any]) -> bool:
    if stage_id == "substrate_bootstrap":
        architecture_ready = bool(
            get_path(reports, ["architecture_gate", "green"], False)
            or get_path(reports, ["architecture_gate", "ready_for_heavy_training"], False)
            or get_path(reports, ["architecture_gate", "status"], "") == "ready_for_heavy_training"
        )
        matched_regression = bool(matched) and regression_count(matched) == len(matched)
        return architecture_ready or matched_regression
    if stage_id == "language_grammar_core":
        names = {str(row.get("benchmark_name") or "") for row in matched if row.get("lifecycle") == "regression"}
        return "babylm_local_probe" in names and "babylm_mutated_holdout" in names
    if stage_id == "residual_synthetic_compression":
        synthetic_ready = bool(get_path(reports, ["synthetic_data", "training_ready"], False))
        escrow_clusters = int(number(get_path(reports, ["residual_escrow", "summary", "cluster_count"], 0)))
        return synthetic_ready and escrow_clusters > 0
    return bool(matched) and regression_count(matched) == len(matched)


def stage_score(
    status: str,
    matched: list[dict[str, Any]],
    source_summary: dict[str, Any],
    reports: dict[str, Any],
    stage: dict[str, Any],
) -> float:
    score = {
        "locked_regression": 1.0,
        "active_frontier": 0.72,
        "ready_next": 0.52,
        "blocked_waiting_asset": 0.35,
        "future": 0.12,
    }.get(status, 0.0)
    if matched:
        scores = [number(row.get("score")) for row in matched if isinstance(row.get("score"), (int, float))]
        if scores:
            score = max(score, min(0.95, sum(scores) / len(scores)))
    if source_summary.get("staged"):
        score += 0.04
    if stage.get("requires_local_rom") and not get_path(reports, ["local_rom_registry", "summary", "ready_for_wrapper_smoke"], False):
        score = min(score, 0.4)
    if stage.get("requires_local_minecraft_license"):
        install_ready = bool(get_path(reports, ["minecraft_runtime_probe", "summary", "local_minecraft_install_detected"], False))
        bridge_ready = bool(get_path(reports, ["minecraft_runtime_probe", "summary", "bridge_runtime_ready"], False))
        if install_ready or bridge_ready:
            score += 0.06
        else:
            score = min(score, 0.45)
    return round(max(0.0, min(1.0, score)), 4)


def stage_next_action(
    stage: dict[str, Any],
    status: str,
    matched: list[dict[str, Any]],
    source_summary: dict[str, Any],
    reports: dict[str, Any],
    blockers: list[str],
) -> str:
    if status == "locked_regression":
        return "Keep as regression and continue to the next capability stage."
    if status == "active_frontier":
        frontier = next((row for row in matched if row.get("lifecycle") == "frontier"), matched[0] if matched else {})
        return f"Improve active frontier {frontier.get('benchmark_name')} until mastery; escrow residuals."
    if stage.get("requires_local_rom") and blockers:
        return "Point SPARKSTREAM_ROM_ROOTS at your private collection or place owned ROMs under data/local_roms, then run wrapper smoke."
    if stage.get("requires_local_minecraft_license") and blockers:
        return "Run the Minecraft runtime probe and stage Crafter/Craftax bridge cards or MineDojo/Malmo full-runtime cards under the local license policy."
    if source_summary.get("approved") or source_summary.get("staged"):
        return "Build the smallest adapter smoke, then create a benchmark card and add it as diagnostic/frontier."
    if status == "blocked_waiting_asset":
        return "; ".join(blockers) if blockers else "Resolve source/license/asset blockers before use."
    return "Wait for prior stages to graduate, but keep metadata staged and adapters planned."


def current_stage(stages: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    focused = focus_stage(stages, config)
    if focused:
        return compact_stage(focused)
    for stage in stages:
        if stage.get("status") != "locked_regression":
            return compact_stage(stage)
    return compact_stage(stages[-1]) if stages else {}


def focus_stage(stages: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    focus = config.get("frontier_focus") if isinstance(config.get("frontier_focus"), dict) else {}
    mode = str(focus.get("mode") or "")
    if mode not in {"programming_first", "conversation_first_temporarily"}:
        return {}
    preferred = [str(item) for item in focus.get("prefer_stage_ids", [])]
    if not preferred:
        return {}
    by_id = {str(stage.get("id") or ""): stage for stage in stages}
    for stage_id in preferred:
        stage = by_id.get(stage_id)
        if not stage:
            continue
        if stage.get("status") in {"active_frontier", "ready_next", "future"}:
            if mode == "conversation_first_temporarily" and stage.get("status") == "future":
                focused = dict(stage)
                focused["status"] = "ready_next"
                focused["blockers"] = []
                focused["next_action"] = (
                    "Temporarily run the English multi-turn conversation/personality lane before returning to code."
                )
                return focused
            return stage
    return {}


def compact_stage(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": stage.get("id"),
        "level": stage.get("level"),
        "title": stage.get("title"),
        "status": stage.get("status"),
        "readiness_score": stage.get("readiness_score"),
        "next_frontier_family": stage.get("next_frontier_family"),
        "next_action": stage.get("next_action"),
        "teacher_policy": stage.get("teacher_policy"),
    }


def next_frontier_plan(
    current: dict[str, Any],
    stages: list[dict[str, Any]],
    reports: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    stage_id = str(current.get("id") or "")
    family = str(current.get("next_frontier_family") or "")
    plan: dict[str, Any] = {
        "stage_id": stage_id,
        "family": family,
        "status": current.get("status"),
        "action": current.get("next_action"),
        "teacher_role": current.get("teacher_policy"),
        "profile_hint": "inner_loop",
    }
    if family == "babylm_mutated":
        plan.update({"runnable_now": True, "runner_family": "babylm_mutated"})
    elif family == "rl_local":
        recs = get_path(reports, ["rl_registry", "recommended_frontier"], [])
        plan.update(
            {
                "runnable_now": bool(recs),
                "runner_family": "rl_local",
                "recommended_env": get_path(recs[0], ["name"], "") if isinstance(recs, list) and recs else "",
            }
        )
    elif family == "emulator_rl":
        emulator_ready = smoke_passed_adapter_cards(reports, category="emulator_rl_environment")
        emulator_blocked = blocked_adapter_cards(reports, category="emulator_rl_environment")
        rl_fallbacks = get_path(reports, ["rl_registry", "adapter_smoke_frontiers", "rl_cards"], [])
        rom_ready = bool(get_path(reports, ["local_rom_registry", "summary", "ready_for_wrapper_smoke"], False))
        if emulator_ready:
            plan.update(
                {
                    "runnable_now": True,
                    "runner_family": "emulator_rl_adapter_smoke",
                    "recommended_env": get_path(emulator_ready[0], ["id"], ""),
                    "local_rom_summary": get_path(reports, ["local_rom_registry", "summary"], {}),
                    "adapter_smoke": {
                        "emulator_ready": len(emulator_ready),
                        "emulator_blocked": len(emulator_blocked),
                    },
                }
            )
        elif isinstance(rl_fallbacks, list) and rl_fallbacks:
            plan.update(
                {
                    "runnable_now": True,
                    "runner_family": "rl_local",
                    "recommended_env": get_path(rl_fallbacks[0], ["id"], ""),
                    "fallback_family": "rl_local",
                    "fallback_reason": (
                        "Emulator ROM metadata is present, but wrapper smoke is blocked; "
                        "continue ratchet pressure on smoke-passed local RL cards."
                    ),
                    "local_rom_summary": get_path(reports, ["local_rom_registry", "summary"], {}),
                    "adapter_smoke": {
                        "emulator_ready": 0,
                        "emulator_blocked": len(emulator_blocked),
                        "rl_smoke_passed": len(rl_fallbacks),
                    },
                }
            )
        else:
            blocker = "local ROM profile unavailable"
            if rom_ready and emulator_blocked:
                blocker = "emulator wrapper runtime dependency blocked"
            plan.update(
                {
                    "runnable_now": False,
                    "runner_family": "emulator_rl_adapter_smoke",
                    "blocked_reason": blocker,
                    "local_rom_summary": get_path(reports, ["local_rom_registry", "summary"], {}),
                    "adapter_smoke": {
                        "emulator_ready": 0,
                        "emulator_blocked": len(emulator_blocked),
                    },
                }
            )
    elif family == "minecraft_rl":
        minecraft_ready = smoke_passed_adapter_cards(reports, category="minecraft_rl_environment")
        minecraft_blocked = blocked_adapter_cards(reports, category="minecraft_rl_environment")
        minecraft_needs_smoke = adapter_cards_by_status(reports, "needs_adapter_smoke", category="minecraft_rl_environment")
        runtime_summary = get_path(reports, ["minecraft_runtime_probe", "summary"], {})
        full_ready = bool(get_path(runtime_summary, ["full_minecraft_runtime_ready"], False))
        bridge_ready = bool(get_path(runtime_summary, ["bridge_runtime_ready"], False))
        install_ready = bool(get_path(runtime_summary, ["local_minecraft_install_detected"], False))
        preferred = preferred_card(
            minecraft_ready,
            ["source_crafter", "source_craftax", "source_minedojo", "source_malmo", "source_voyager_minecraft", "source_minerl"],
        )
        if preferred and (full_ready or bridge_ready or preferred.get("id") in {"source_crafter", "source_craftax"}):
            plan.update(
                {
                    "runnable_now": True,
                    "runner_family": "minecraft_rl_local",
                    "recommended_env": preferred.get("id", ""),
                    "runtime_summary": runtime_summary,
                    "adapter_smoke": {
                        "minecraft_ready": len(minecraft_ready),
                        "minecraft_blocked": len(minecraft_blocked),
                        "minecraft_needs_smoke": len(minecraft_needs_smoke),
                    },
                }
            )
        elif minecraft_needs_smoke:
            plan.update(
                {
                    "runnable_now": True,
                    "runner_family": "benchmark_adapter_smoke",
                    "recommended_env": get_path(minecraft_needs_smoke[0], ["id"], ""),
                    "action": "Run bounded Minecraft adapter smoke before promoting a Minecraft/Open-World source to pressure.",
                    "runtime_summary": runtime_summary,
                    "adapter_smoke": {
                        "minecraft_ready": len(minecraft_ready),
                        "minecraft_blocked": len(minecraft_blocked),
                        "minecraft_needs_smoke": len(minecraft_needs_smoke),
                    },
                }
            )
        else:
            plan.update(
                {
                    "runnable_now": False,
                    "runner_family": "minecraft_adapter_required",
                    "blocked_reason": (
                        "Minecraft runtime/cards are not ready"
                        if install_ready
                        else "local Minecraft install not detected and no bridge smoke passed"
                    ),
                    "runtime_summary": runtime_summary,
                    "adapter_smoke": {
                        "minecraft_ready": len(minecraft_ready),
                        "minecraft_blocked": len(minecraft_blocked),
                        "minecraft_needs_smoke": len(minecraft_needs_smoke),
                    },
                }
            )
    elif family == "drone_rl":
        drone_categories = {"drone_rl_environment", "drone_racing_simulator", "drone_control_api"}
        drone_ready = smoke_passed_adapter_cards(reports, categories=drone_categories)
        drone_blocked = blocked_adapter_cards(reports, categories=drone_categories)
        drone_needs_smoke = adapter_cards_by_status(reports, "needs_adapter_smoke", categories=drone_categories)
        spec_ready = bool(get_path(reports, ["ai_grand_prix_spec", "summary", "contract_recorded"], False))
        python_checked = bool(get_path(reports, ["python_runtime_compatibility", "summary"], {}))
        if drone_ready:
            plan.update(
                {
                    "runnable_now": True,
                    "runner_family": "drone_rl_local",
                    "recommended_env": get_path(drone_ready[0], ["id"], ""),
                    "adapter_smoke": {
                        "drone_ready": len(drone_ready),
                        "drone_blocked": len(drone_blocked),
                        "drone_needs_smoke": len(drone_needs_smoke),
                    },
                    "spec_ready": spec_ready,
                    "python_runtime_checked": python_checked,
                }
            )
        elif drone_needs_smoke:
            plan.update(
                {
                    "runnable_now": True,
                    "runner_family": "benchmark_adapter_smoke",
                    "recommended_env": get_path(drone_needs_smoke[0], ["id"], ""),
                    "action": "Run bounded drone adapter smoke before promoting any drone simulator to frontier.",
                    "adapter_smoke": {
                        "drone_ready": 0,
                        "drone_blocked": len(drone_blocked),
                        "drone_needs_smoke": len(drone_needs_smoke),
                    },
                    "spec_ready": spec_ready,
                    "python_runtime_checked": python_checked,
                }
            )
        else:
            blocker = "drone sources are not staged or adapter cards are blocked"
            if drone_blocked:
                blocker = "drone adapter smoke passed governance but is runtime-blocked; create the Python/simulator/SITL runtime"
            plan.update(
                {
                    "runnable_now": False,
                    "runner_family": "drone_adapter_required",
                    "blocked_reason": blocker,
                    "adapter_smoke": {
                        "drone_ready": 0,
                        "drone_blocked": len(drone_blocked),
                        "drone_needs_smoke": 0,
                    },
                    "spec_ready": spec_ready,
                    "python_runtime_checked": python_checked,
                }
            )
    elif family == "coding_local_sandbox":
        coding_categories = {
            "coding_benchmark",
            "coding_agent_benchmark",
            "coding_agent_framework",
            "synthetic_coding_benchmark",
            "synthetic_benchmark",
            "multi_stream_coding_benchmark",
        }
        coding_ready = smoke_passed_adapter_cards(reports, categories=coding_categories)
        coding_runnable = [card for card in coding_ready if coding_source_material_present(card)]
        coding_setup_required = [card for card in coding_ready if card not in coding_runnable]
        coding_blocked = blocked_adapter_cards(reports, categories=coding_categories)
        coding_needs_smoke = adapter_cards_by_status(reports, "needs_adapter_smoke", categories=coding_categories)
        rotation = select_same_family_pressure_card(
            coding_runnable,
            reports,
            config,
            family="coding_local_sandbox",
            default_order=DEFAULT_CODING_PRESSURE_CARD_ORDER,
        )
        transfer_interleave = select_transfer_interleave(
            reports,
            config,
            base_family="coding_local_sandbox",
            rotation=rotation,
        )
        by_id = {str(card.get("id") or ""): card for card in coding_runnable}
        preferred_id = str(rotation.get("selected_card_id") or "")
        preferred = by_id.get(preferred_id) or preferred_card(coding_runnable, coding_pressure_card_order(config))
        if preferred:
            common = {
                "adapter_smoke": {
                    "coding_ready": len(coding_ready),
                    "coding_runnable": len(coding_runnable),
                    "coding_source_setup_required": len(coding_setup_required),
                    "coding_source_setup_required_ids": [card.get("id") for card in coding_setup_required[:16]],
                    "coding_blocked": len(coding_blocked),
                    "coding_needs_smoke": len(coding_needs_smoke),
                },
                "programming_first": True,
                "same_family_rotation": rotation,
                "transfer_interleave": transfer_interleave,
                "real_code_graduation": real_code_graduation_context(reports),
            }
            if transfer_interleave.get("apply"):
                plan.update(
                    {
                        "base_family": family,
                        "family": transfer_interleave.get("family") or family,
                        "runnable_now": True,
                        "runner_family": transfer_interleave.get("runner_family") or "transfer_eval_local",
                        "recommended_env": transfer_interleave.get("recommended_env") or "transfer_eval_suite",
                        "action": transfer_interleave.get("action"),
                        **common,
                    }
                )
            else:
                plan.update(
                    {
                        "runnable_now": True,
                        "runner_family": "coding_local_sandbox",
                        "recommended_env": preferred.get("id", ""),
                        **common,
                    }
                )
        elif coding_needs_smoke:
            plan.update(
                {
                    "runnable_now": True,
                    "runner_family": "benchmark_adapter_smoke",
                    "recommended_env": get_path(coding_needs_smoke[0], ["id"], ""),
                    "action": "Run bounded coding adapter smoke before promoting a programming source to pressure.",
                    "adapter_smoke": {
                        "coding_ready": len(coding_ready),
                        "coding_runnable": 0,
                        "coding_source_setup_required": len(coding_setup_required),
                        "coding_source_setup_required_ids": [card.get("id") for card in coding_setup_required[:16]],
                        "coding_blocked": len(coding_blocked),
                        "coding_needs_smoke": len(coding_needs_smoke),
                    },
                    "programming_first": True,
                }
            )
        else:
            plan.update(
                {
                    "runnable_now": False,
                    "runner_family": "coding_adapter_required",
                    "blocked_reason": "coding benchmark sources are not staged or adapter cards are blocked",
                    "adapter_smoke": {
                        "coding_ready": len(coding_ready),
                        "coding_runnable": 0,
                        "coding_source_setup_required": len(coding_setup_required),
                        "coding_source_setup_required_ids": [card.get("id") for card in coding_setup_required[:16]],
                        "coding_blocked": len(coding_blocked),
                        "coding_needs_smoke": 0,
                    },
                    "programming_first": True,
                }
            )
    elif family == "tool_agent":
        conversation = reports.get("multi_turn_conversation") if isinstance(reports.get("multi_turn_conversation"), dict) else {}
        pantry = reports.get("open_conversation_pantry") if isinstance(reports.get("open_conversation_pantry"), dict) else {}
        plan.update(
            {
                "runnable_now": True,
                "runner_family": "conversation_multiturn_local",
                "recommended_env": "multi_turn_conversation_benchmark",
                "conversation_first": True,
                "conversation_summary": {
                    "benchmark_passed": conversation.get("passed"),
                    "benchmark_accuracy": get_path(conversation, ["summary", "accuracy"], None),
                    "benchmark_turns": get_path(conversation, ["summary", "turn_count"], None),
                    "personality_ready_turns": get_path(
                        conversation, ["summary", "personality_context_ready_turns"], None
                    ),
                    "open_conversation_train_rows": get_path(pantry, ["summary", "private_train_rows"], 0),
                    "open_conversation_sts_rows": get_path(pantry, ["summary", "sts_rows"], 0),
                },
                "action": (
                    "Run the multi-turn conversation benchmark and personality runtime audit; use open conversation "
                    "SFT/STS rows as private pressure while keeping public benchmarks calibration-only."
                ),
            }
        )
    else:
        plan.update({"runnable_now": False, "runner_family": "adapter_required"})
    return plan


def smoke_passed_adapter_cards(
    reports: dict[str, Any],
    *,
    category: str = "",
    categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    return adapter_cards_by_status(reports, "adapter_smoke_passed", category=category, categories=categories)


def blocked_adapter_cards(
    reports: dict[str, Any],
    *,
    category: str = "",
    categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    cards = []
    for card in adapter_cards(reports):
        if category and card.get("category") != category:
            continue
        if categories and card.get("category") not in categories:
            continue
        if str(card.get("status") or "").startswith("blocked"):
            cards.append(card)
    return cards


def adapter_cards_by_status(
    reports: dict[str, Any],
    status: str,
    *,
    category: str = "",
    categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    cards = []
    for card in adapter_cards(reports):
        if card.get("status") != status:
            continue
        if category and card.get("category") != category:
            continue
        if categories and card.get("category") not in categories:
            continue
        cards.append(card)
    return cards


def adapter_cards(reports: dict[str, Any]) -> list[dict[str, Any]]:
    rows = get_path(reports, ["adapter_factory", "cards"], [])
    cards = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    synthetic_rows = get_path(reports, ["synthetic_benchmark_factory", "cards"], [])
    if isinstance(synthetic_rows, list):
        cards.extend(row for row in synthetic_rows if isinstance(row, dict))
    multi_stream_rows = get_path(reports, ["multi_stream_trace_factory", "cards"], [])
    if isinstance(multi_stream_rows, list):
        cards.extend(row for row in multi_stream_rows if isinstance(row, dict))
    old_registry_rows = get_path(reports, ["old_project_registry_port", "cards"], [])
    if isinstance(old_registry_rows, list):
        cards.extend(row for row in old_registry_rows if isinstance(row, dict))
    return cards


def coding_source_material_present(card: dict[str, Any]) -> bool:
    if card.get("license_allowed") is False:
        return False
    if card.get("decision") not in {
        None,
        "",
        "approved_for_catalog_import",
        "synthetic_local_generated",
        "multi_stream_local_generated",
        "old_project_registry_ported",
    }:
        return False
    path = resolve_curriculum_path(str(card.get("resource_pantry_path") or card.get("staged_path") or ""))
    if not path.exists():
        return False
    if path.is_file():
        return True
    try:
        return any(path.iterdir())
    except OSError:
        return False


def resolve_curriculum_path(value: str | Path) -> Path:
    if not str(value):
        return ROOT / "__missing__"
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def preferred_card(cards: list[dict[str, Any]], ordered_ids: list[str]) -> dict[str, Any]:
    by_id = {str(card.get("id") or ""): card for card in cards}
    for card_id in ordered_ids:
        if card_id in by_id:
            return by_id[card_id]
    return cards[0] if cards else {}


def coding_pressure_card_order(config: dict[str, Any]) -> list[str]:
    rotation = config.get("same_family_rotation") if isinstance(config.get("same_family_rotation"), dict) else {}
    family = rotation.get("coding_local_sandbox") if isinstance(rotation.get("coding_local_sandbox"), dict) else {}
    configured = family.get("ordered_ids") if isinstance(family.get("ordered_ids"), list) else []
    ordered = [str(item) for item in configured if str(item)]
    return ordered or list(DEFAULT_CODING_PRESSURE_CARD_ORDER)


def select_same_family_pressure_card(
    cards: list[dict[str, Any]],
    reports: dict[str, Any],
    config: dict[str, Any],
    *,
    family: str,
    default_order: list[str],
) -> dict[str, Any]:
    rotation_cfg = config.get("same_family_rotation") if isinstance(config.get("same_family_rotation"), dict) else {}
    family_cfg = rotation_cfg.get(family) if isinstance(rotation_cfg.get(family), dict) else {}
    enabled = bool(rotation_cfg.get("enabled", True))
    ordered_ids = [str(item) for item in family_cfg.get("ordered_ids", default_order) if str(item)]
    if not ordered_ids:
        ordered_ids = list(default_order)
    by_id = {str(card.get("id") or ""): card for card in cards if str(card.get("id") or "")}
    ready_order = [card_id for card_id in ordered_ids if card_id in by_id]
    ready_order.extend([card_id for card_id in by_id if card_id not in ready_order])
    if not ready_order:
        return {
            "enabled": enabled,
            "family": family,
            "selected_card_id": "",
            "reason": "no_smoke_passed_cards",
            "ready_card_count": 0,
        }

    attempts_before_rotate = max(1, int(number(rotation_cfg.get("below_floor_attempts_before_rotate", 4))))
    stalled_cycles_before_rotate = max(1, int(number(rotation_cfg.get("stalled_cycles_before_rotate", 2))))
    floor_default = float(number(rotation_cfg.get("floor_threshold", 0.70)))
    current_id = current_pressure_card_id(reports, ready_order) or ready_order[0]
    if current_id not in ready_order:
        current_id = ready_order[0]
    current_row = ledger_row_for_card(reports, current_id, family=family)
    current_state = card_frontier_state(current_id, current_row, floor_default=floor_default)
    forge_hint = code_forge_rotation_hint(reports, family=family, current_id=current_id, ready_order=ready_order)
    public_transfer_stall = real_code_public_transfer_stall(
        reports,
        current_id=current_id,
        current_state=current_state,
        forge_hint=forge_hint,
        floor_default=floor_default,
        attempts_before_rotate=attempts_before_rotate,
        stalled_cycles_before_rotate=stalled_cycles_before_rotate,
    )
    below_floor_stalled = bool(
        current_state.get("below_floor")
        and (
            int(current_state.get("attempt_count") or 0) >= attempts_before_rotate
            or int(current_state.get("stalled_cycles") or 0) >= stalled_cycles_before_rotate
        )
    )
    promotion_blocked_for_current = candidate_gate_blocks_current_card(reports, current_id)
    public_code_ready_card = real_code_graduation_ready_card_id(reports, ready_order, floor_default=floor_default)
    public_code_pressure_card = real_code_calibration_pressure_card_id(reports, ready_order, floor_default=floor_default)
    broad_code_pressure_card = broad_public_code_pressure_card_id(
        reports,
        ready_order,
        floor_default=floor_default,
        skip_card=current_id if public_transfer_stall.get("stalled") else "",
    )
    current_public_card_needs_calibration = bool(
        public_transfer_stall.get("eligible_public_calibration")
        and not public_transfer_stall.get("current_card_in_public_report")
        and current_id in PUBLIC_CODE_CALIBRATION_CARDS
    )
    selected_id = current_id
    reason = "continue_current_card"
    if enabled and broad_code_pressure_card:
        selected_id = broad_code_pressure_card
        reason = (
            "continue_broad_public_code_transfer_wall"
            if selected_id == current_id
            else "rotate_to_broad_public_code_transfer_wall"
        )
    elif enabled and public_code_pressure_card and is_private_code_pressure_card(current_id):
        selected_id = public_code_pressure_card
        reason = "rotate_private_pressure_to_public_code_calibration_frontier"
    elif enabled and current_state.get("lifecycle") == "regression" and promotion_blocked_for_current and current_public_card_needs_calibration:
        reason = "continue_current_public_code_card_until_calibration_runs"
    elif enabled and current_state.get("lifecycle") == "regression" and promotion_blocked_for_current and public_code_pressure_card and public_code_pressure_card != current_id:
        selected_id = public_code_pressure_card
        reason = "rotate_regression_card_to_public_code_calibration_wall"
    elif enabled and public_code_ready_card and public_code_ready_card != current_id:
        selected_id = public_code_ready_card
        reason = "rotate_to_public_code_graduation_ready_card"
    elif enabled and current_state.get("lifecycle") == "regression" and promotion_blocked_for_current and public_transfer_stall.get("stalled") and len(ready_order) > 1:
        selected_id = next_public_or_source_code_card_id(
            ready_order,
            current_id,
            reports,
            family=family,
            floor_default=floor_default,
        ) or current_id
        reason = (
            "rotate_public_code_transfer_stalled_card"
            if selected_id != current_id
            else "public_code_transfer_stalled_but_no_ready_alternate"
        )
    elif enabled and current_state.get("lifecycle") == "regression" and promotion_blocked_for_current:
        reason = "continue_current_card_until_public_code_graduation_gate_passes"
    elif enabled and current_state.get("lifecycle") == "regression" and len(ready_order) > 1:
        if family == "coding_local_sandbox":
            selected_id = next_public_or_source_code_card_id(
                ready_order,
                current_id,
                reports,
                family=family,
                floor_default=floor_default,
            ) or current_id
            reason = (
                "rotate_promoted_regression_card_to_public_source_code"
                if selected_id != current_id
                else "all_public_source_code_cards_already_regression"
            )
        else:
            selected_id = next_open_card_id(ready_order, current_id, reports, family=family) or current_id
            reason = "rotate_promoted_regression_card" if selected_id != current_id else "all_ready_cards_already_regression"
    elif enabled and forge_hint.get("decision") == "rotate":
        hinted = str(forge_hint.get("selected_card_id") or "")
        selected_id = hinted if hinted in ready_order else current_id
        reason = "code_residual_forge_rotate" if selected_id != current_id else "code_residual_forge_hint_no_ready_target"
    elif enabled and below_floor_stalled and len(ready_order) > 1:
        selected_id = next_open_card_id(ready_order, current_id, reports, family=family) or current_id
        reason = "rotate_below_floor_stalled_card" if selected_id != current_id else "all_ready_cards_already_rotated"
    elif current_row and current_state.get("above_floor"):
        reason = "continue_current_card_above_floor"
    elif current_row:
        reason = "continue_current_card_until_rotation_threshold"
    else:
        reason = "start_first_smoke_passed_card"

    return_queue = rotation_queue_after(ready_order, selected_id)
    selected_state = card_frontier_state(
        selected_id,
        ledger_row_for_card(reports, selected_id, family=family),
        floor_default=floor_default,
    )
    return {
        "enabled": enabled,
        "family": family,
        "selected_card_id": selected_id,
        "reason": reason,
        "current_card_id": current_id,
        "current_card": current_state,
        "selected_card": selected_state,
        "ready_card_count": len(ready_order),
        "ready_order": ready_order[:20],
        "return_queue": return_queue[:8],
        "below_floor_attempts_before_rotate": attempts_before_rotate,
        "stalled_cycles_before_rotate": stalled_cycles_before_rotate,
        "floor_threshold": floor_default,
        "code_residual_forge_hint": forge_hint,
        "public_code_ready_card_id": public_code_ready_card,
        "public_code_calibration_pressure_card_id": public_code_pressure_card,
        "broad_public_code_pressure_card_id": broad_code_pressure_card,
        "public_code_transfer_stall": public_transfer_stall,
        "frontier_truth": {
            "synthetic_pressure_role": "private_training_or_probe_only",
            "promotion_facing_frontier_required": "real_public_code_calibration",
            "current_card_is_private_pressure": is_private_code_pressure_card(current_id),
            "selected_card_is_private_pressure": is_private_code_pressure_card(selected_id),
        },
    }


def candidate_gate_blocks_current_card(reports: dict[str, Any], current_id: str) -> bool:
    gate = reports.get("candidate_gate") if isinstance(reports.get("candidate_gate"), dict) else {}
    if gate.get("promote") is not False:
        return False
    active = str(get_path(gate, ["artifacts", "active_frontier"], ""))
    if current_id and current_id not in active:
        return False
    failed = {
        str(item.get("gate") or "")
        for item in gate.get("checks", [])
        if isinstance(item, dict) and not item.get("passed")
    }
    return bool(failed & public_code_blocking_gates())


def public_code_blocking_gates() -> set[str]:
    return {
        "real_code_benchmark_graduation_ready",
        "broad_public_code_transfer_ready",
    }


def real_code_graduation_ready_card_id(reports: dict[str, Any], ready_order: list[str], *, floor_default: float) -> str:
    report = reports.get("real_code_benchmark_graduation") if isinstance(reports.get("real_code_benchmark_graduation"), dict) else {}
    if report.get("policy") != "project_theseus_real_code_benchmark_graduation_v1":
        return ""
    if report.get("trigger_state") != "GREEN":
        return ""
    if report.get("candidate_source") not in {
        "local_theseus_student_checkpoint",
        "student_learning_checkpoint_v1",
        "student_neural_checkpoint_v1",
        "student_token_generator_checkpoint_v1",
        "student_code_lm_checkpoint_v1",
    }:
        return ""
    if report.get("public_benchmark_score_claim") not in {
        "student_checkpoint_public_task_calibration_only",
        "student_learning_checkpoint_public_task_calibration_only",
        "student_neural_checkpoint_public_task_calibration_only",
        "student_token_generator_checkpoint_public_task_calibration_only",
        "student_code_lm_checkpoint_public_task_calibration_only",
    }:
        return ""
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if int(summary.get("public_task_count") or 0) <= 0:
        return ""
    if float(number(summary.get("real_public_task_pass_rate"))) < floor_default:
        return ""
    if not bool(summary.get("student_candidate_benchmark_integrity_valid")):
        return ""
    if not bool(summary.get("token_level_code_generation_learned")):
        return ""
    if int(summary.get("benchmark_promotion_eligible_candidate_count") or 0) <= 0:
        return ""
    if int(summary.get("template_like_candidate_count") or 0) != 0:
        return ""
    if int(summary.get("loop_closure_candidate_count") or 0) != 0:
        return ""
    if int(summary.get("task_level_regressions_vs_single_stream") or 0) != 0:
        return ""
    for card_id in report.get("cards", []) if isinstance(report.get("cards"), list) else []:
        value = str(card_id or "")
        if value in ready_order:
            return value
    return ""


def real_code_calibration_pressure_card_id(reports: dict[str, Any], ready_order: list[str], *, floor_default: float) -> str:
    report = reports.get("real_code_benchmark_graduation") if isinstance(reports.get("real_code_benchmark_graduation"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    public_count = int(number(summary.get("public_task_count")))
    public_rate = float(number(summary.get("real_public_task_pass_rate")))
    if public_count > 0 and public_rate >= floor_default:
        return ""
    for card_id in report.get("cards", []) if isinstance(report.get("cards"), list) else []:
        value = str(card_id or "")
        if value in ready_order and value in PUBLIC_CODE_TASK_CARDS and public_count > 0:
            return value
    for value in ready_order:
        if value in PUBLIC_CODE_TASK_CARDS:
            return value
    for value in ready_order:
        if value in PUBLIC_CODE_CALIBRATION_CARDS:
            return value
    return ""


def broad_public_code_pressure_card_id(
    reports: dict[str, Any],
    ready_order: list[str],
    *,
    floor_default: float,
    skip_card: str = "",
) -> str:
    """Select the next public code card from the broad transfer matrix.

    The latest single real-code report can be clean while the broader matrix is
    still below floor or missing clean evidence. Use the matrix to keep
    unattended runs rotating through the true broad-transfer wall.
    """

    matrix = reports.get("broad_transfer_matrix") if isinstance(reports.get("broad_transfer_matrix"), dict) else {}
    if matrix.get("policy") != "project_theseus_broad_transfer_matrix_v1":
        return ""
    if matrix.get("trigger_state") == "GREEN":
        return ""
    summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
    min_tasks = int(number(summary.get("min_public_tasks_per_promotion_card")) or 32)
    rows = {str(row.get("card_id") or ""): row for row in matrix.get("rows", []) if isinstance(row, dict)}
    below_floor = {str(item) for item in summary.get("cards_below_floor", []) if str(item)}
    no_clean = {str(item) for item in summary.get("no_clean_student_evidence_cards", []) if str(item)}
    coverage_warning = {str(item) for item in summary.get("coverage_warning_cards", []) if str(item)}
    loader_only = {str(item) for item in summary.get("loader_only_cards", []) if str(item)}

    ordered = [card_id for card_id in ready_order if card_id != skip_card]

    for card_id in ordered:
        row = rows.get(card_id, {})
        if card_id in below_floor and card_id in PUBLIC_CODE_TASK_CARDS and int(row.get("public_task_count") or 0) >= min_tasks:
            return card_id
    for card_id in ordered:
        row = rows.get(card_id, {})
        if card_id in no_clean and card_id in PUBLIC_CODE_TASK_CARDS and int(row.get("public_task_count") or 0) >= min_tasks:
            return card_id
    for card_id in ordered:
        row = rows.get(card_id, {})
        if card_id in no_clean and card_id in PUBLIC_CODE_TASK_CARDS and int(row.get("public_task_count") or 0) > 0:
            return card_id
    for card_id in ordered:
        if card_id in coverage_warning and card_id in PUBLIC_CODE_TASK_CARDS:
            return card_id
    for card_id in ordered:
        if card_id in loader_only:
            return card_id
    return ""


def real_code_public_transfer_stall(
    reports: dict[str, Any],
    *,
    current_id: str,
    current_state: dict[str, Any],
    forge_hint: dict[str, Any],
    floor_default: float,
    attempts_before_rotate: int,
    stalled_cycles_before_rotate: int,
) -> dict[str, Any]:
    """Treat the public transfer floor as the real code-family stall signal.

    A pressure card can be above its private/local floor while the learned code
    generator still fails promotion-facing public calibration. Overnight runs
    should rotate to sibling code cards in that case instead of staying on one
    public-shaped surface indefinitely.
    """

    broad = reports.get("broad_transfer_matrix") if isinstance(reports.get("broad_transfer_matrix"), dict) else {}
    broad_summary = broad.get("summary") if isinstance(broad.get("summary"), dict) else {}
    broad_rows = [row for row in broad.get("rows", []) if isinstance(row, dict)]
    broad_row = next((row for row in broad_rows if str(row.get("card_id") or "") == current_id), {})

    report = reports.get("real_code_benchmark_graduation") if isinstance(reports.get("real_code_benchmark_graduation"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if broad_row:
        public_count = int(number(broad_row.get("public_task_count")))
        public_rate = float(number(broad_row.get("multi_stream_pass_rate")))
        floor_value = number(broad.get("required_public_task_floor"))
        broad_evidence_used = True
    elif broad_summary:
        public_count = int(number(broad_summary.get("real_public_task_count")))
        public_rate = float(number(broad_summary.get("real_public_pass_rate")))
        floor_value = number(broad.get("required_public_task_floor"))
        broad_evidence_used = True
    else:
        public_count = int(number(summary.get("public_task_count")))
        public_rate = float(number(summary.get("real_public_task_pass_rate")))
        floor_value = number(summary.get("required_public_task_floor"))
        broad_evidence_used = False
    floor = float(floor_value if floor_value > 0 else floor_default)
    cards = [str(card_id or "") for card_id in report.get("cards", []) if str(card_id or "")]
    current_in_report = bool(broad_row or (not cards or current_id in cards))
    frontier_pressure = reports.get("frontier_policy") if isinstance(reports.get("frontier_policy"), dict) else {}
    pressure_attempts = int(number(get_path(frontier_pressure, ["frontier_pressure", "active_frontier_attempt_count"], 0)))
    observed_attempts = max(
        int(current_state.get("attempt_count") or 0),
        int(number(forge_hint.get("active_attempt_count"))),
        pressure_attempts,
    )
    observed_stalled_cycles = int(current_state.get("stalled_cycles") or 0)
    below_public_floor = bool(public_count > 0 and public_rate < floor)
    broad_eligible = bool(
        broad_evidence_used
        and broad.get("policy") == "project_theseus_broad_transfer_matrix_v1"
        and broad.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(broad_summary.get("no_cheat_violation_count") or 0) == 0
        and not broad_summary.get("no_clean_student_evidence_cards")
        and (not broad_row or bool(broad_row.get("clean_student_evidence_available")))
        and (not broad_row or bool(broad_row.get("no_cheat_valid")))
    )
    single_report_eligible = bool(
        report.get("policy") == "project_theseus_real_code_benchmark_graduation_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and report.get("candidate_source")
        in {
            "local_theseus_student_checkpoint",
            "student_learning_checkpoint_v1",
            "student_neural_checkpoint_v1",
            "student_token_generator_checkpoint_v1",
            "student_code_lm_checkpoint_v1",
        }
        and report.get("public_benchmark_score_claim")
        in {
            "student_checkpoint_public_task_calibration_only",
            "student_learning_checkpoint_public_task_calibration_only",
            "student_neural_checkpoint_public_task_calibration_only",
            "student_token_generator_checkpoint_public_task_calibration_only",
            "student_code_lm_checkpoint_public_task_calibration_only",
        }
        and int(report.get("external_inference_calls") or 0) == 0
        and bool(summary.get("student_candidate_benchmark_integrity_valid"))
        and bool(summary.get("token_level_code_generation_learned"))
        and int(summary.get("template_like_candidate_count") or 0) == 0
        and int(summary.get("loop_closure_candidate_count") or 0) == 0
    )
    eligible = bool(broad_eligible or single_report_eligible)
    stalled = bool(
        eligible
        and current_in_report
        and below_public_floor
        and (
            observed_attempts >= max(1, attempts_before_rotate)
            or observed_stalled_cycles >= max(1, stalled_cycles_before_rotate)
        )
    )
    return {
        "stalled": stalled,
        "eligible_public_calibration": eligible,
        "current_card_in_public_report": current_in_report,
        "broad_matrix_evidence_used": broad_evidence_used,
        "public_task_count": public_count,
        "real_public_task_pass_rate": round(public_rate, 6),
        "required_floor": round(floor, 6),
        "floor_gap": round(max(0.0, floor - public_rate), 6),
        "observed_attempts": observed_attempts,
        "attempts_before_rotate": attempts_before_rotate,
        "observed_stalled_cycles": observed_stalled_cycles,
        "stalled_cycles_before_rotate": stalled_cycles_before_rotate,
        "reason": (
            "public_transfer_below_floor_past_rotation_threshold"
            if stalled
            else "public_transfer_rotation_not_due"
        ),
    }


def is_private_code_pressure_card(card_id: str) -> bool:
    if not card_id:
        return False
    if card_id in PUBLIC_CODE_CALIBRATION_CARDS:
        return False
    if card_id.startswith("source_"):
        return False
    return True


def code_forge_rotation_hint(
    reports: dict[str, Any],
    *,
    family: str,
    current_id: str,
    ready_order: list[str],
) -> dict[str, Any]:
    if family != "coding_local_sandbox":
        return {}
    forge = reports.get("code_residual_forge") if isinstance(reports.get("code_residual_forge"), dict) else {}
    rotation = forge.get("rotation") if isinstance(forge.get("rotation"), dict) else {}
    if not rotation:
        rotation = reports.get("code_frontier_rotation") if isinstance(reports.get("code_frontier_rotation"), dict) else {}
    if rotation.get("policy") != "project_theseus_code_frontier_rotation_hint_v1":
        return {}
    if rotation.get("family") != family:
        return {}
    if str(rotation.get("current_card_id") or "") != current_id:
        return {
            "decision": "ignore",
            "reason": "hint_current_card_mismatch",
            "hint_current_card_id": rotation.get("current_card_id"),
            "current_card_id": current_id,
        }
    selected = str(rotation.get("selected_card_id") or "")
    if selected and selected not in ready_order:
        return {
            "decision": "ignore",
            "reason": "hint_selected_card_not_ready",
            "selected_card_id": selected,
        }
    return {
        "decision": str(rotation.get("decision") or ""),
        "reason": str(rotation.get("reason") or ""),
        "selected_card_id": selected,
        "active_cluster_count": rotation.get("active_cluster_count"),
        "active_attempt_count": rotation.get("active_attempt_count"),
    }


def real_code_graduation_context(reports: dict[str, Any]) -> dict[str, Any]:
    report = reports.get("real_code_benchmark_graduation") if isinstance(reports.get("real_code_benchmark_graduation"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "public_task_count": summary.get("public_task_count"),
        "loader_regression_case_count": summary.get("loader_regression_case_count"),
        "total_case_count": summary.get("total_case_count"),
        "single_stream_pass_rate": summary.get("single_stream_pass_rate"),
        "multi_stream_pass_rate": summary.get("multi_stream_pass_rate"),
        "pass_rate_delta": summary.get("pass_rate_delta"),
        "task_level_regressions_vs_single_stream": summary.get("task_level_regressions_vs_single_stream"),
        "student_candidate_benchmark_integrity_valid": summary.get("student_candidate_benchmark_integrity_valid"),
        "token_level_code_generation_learned": summary.get("token_level_code_generation_learned"),
        "template_like_candidate_count": summary.get("template_like_candidate_count"),
        "loop_closure_candidate_count": summary.get("loop_closure_candidate_count"),
        "benchmark_promotion_eligible_candidate_count": summary.get("benchmark_promotion_eligible_candidate_count"),
        "public_benchmark_score_claim": report.get("public_benchmark_score_claim"),
        "promotion_allowed": report.get("promotion_allowed"),
    }


def current_pressure_card_id(reports: dict[str, Any], ready_order: list[str]) -> str:
    code_rotation = reports.get("code_frontier_rotation") if isinstance(reports.get("code_frontier_rotation"), dict) else {}
    if code_rotation.get("policy") == "project_theseus_code_frontier_rotation_hint_v1":
        current = str(code_rotation.get("current_card_id") or "")
        if current in ready_order:
            return current
    frontier = reports.get("frontier_policy") if isinstance(reports.get("frontier_policy"), dict) else {}
    candidates = [
        str(frontier.get("pressure_card_id") or ""),
        str(get_path(frontier, ["frontier_pressure", "next_pressure_card_id"], "") or ""),
        str(get_path(frontier, ["frontier_pressure", "curriculum_recommended_env"], "") or ""),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return ready_order[0] if ready_order else ""


def ledger_row_for_card(reports: dict[str, Any], card_id: str, *, family: str) -> dict[str, Any]:
    ledger = reports.get("benchmark_ledger") if isinstance(reports.get("benchmark_ledger"), list) else []
    matches = []
    for row in ledger:
        if not isinstance(row, dict):
            continue
        if benchmark_family(row) != family:
            continue
        haystack = f"{row.get('benchmark_name') or ''} {row.get('best_report') or ''}"
        if card_id and card_id in haystack:
            matches.append(row)
    if not matches:
        return {}
    return max(matches, key=lambda row: int(get_path(row, ["graduation_policy", "attempt_count"], 0) or 0))


def card_frontier_state(card_id: str, row: dict[str, Any], *, floor_default: float) -> dict[str, Any]:
    if not row:
        return {
            "card_id": card_id,
            "benchmark_name": "",
            "lifecycle": "unseen",
            "score": None,
            "floor": floor_default,
            "attempt_count": 0,
            "stalled_cycles": 0,
            "below_floor": False,
            "above_floor": False,
        }
    policy = row.get("graduation_policy") if isinstance(row.get("graduation_policy"), dict) else {}
    score = float(number(row.get("score")))
    floor = float(number(policy.get("floor_threshold") if policy.get("floor_threshold") is not None else floor_default))
    return {
        "card_id": card_id,
        "benchmark_name": row.get("benchmark_name"),
        "lifecycle": row.get("lifecycle"),
        "score": round(score, 4),
        "floor": round(floor, 4),
        "attempt_count": int(policy.get("attempt_count") or 0),
        "stalled_cycles": int(policy.get("stalled_cycles") or 0),
        "current_threshold": policy.get("current_threshold"),
        "below_floor": score < floor,
        "above_floor": score >= floor,
    }


def benchmark_family(row: dict[str, Any]) -> str:
    name = str(row.get("benchmark_name") or "")
    best_report = str(row.get("best_report") or "")
    if name.startswith("coding_") or any(card in best_report for card in DEFAULT_CODING_PRESSURE_CARD_ORDER):
        return "coding_local_sandbox"
    if name.startswith("drone_rl_") or name.startswith("drone_control_"):
        return "drone_rl"
    if name.startswith("minecraft_rl_") or name.startswith("minecraft_"):
        return "minecraft_rl"
    if name.startswith("web_agent_"):
        return "web_agent_local"
    if name.startswith("transfer_") or name.startswith("asi_transfer"):
        return "transfer_eval"
    return ""


def next_open_card_id(ready_order: list[str], current_id: str, reports: dict[str, Any], *, family: str) -> str:
    for card_id in rotation_queue_after(ready_order, current_id):
        row = ledger_row_for_card(reports, card_id, family=family)
        if row.get("lifecycle") != "regression":
            return card_id
    return ""


def next_public_or_source_code_card_id(
    ready_order: list[str],
    current_id: str,
    reports: dict[str, Any],
    *,
    family: str,
    floor_default: float,
) -> str:
    candidates = [
        card_id
        for card_id in rotation_queue_after(ready_order, current_id)
        if card_id in PUBLIC_CODE_CALIBRATION_CARDS or card_id.startswith("source_")
    ]
    for allow_regression in (False, True):
        for card_id in candidates:
            if is_private_code_pressure_card(card_id):
                continue
            state = card_frontier_state(
                card_id,
                ledger_row_for_card(reports, card_id, family=family),
                floor_default=floor_default,
            )
            if allow_regression or state.get("lifecycle") != "regression":
                return card_id
    return ""


def rotation_queue_after(ready_order: list[str], current_id: str) -> list[str]:
    if current_id not in ready_order:
        return [card_id for card_id in ready_order if card_id != current_id]
    index = ready_order.index(current_id)
    rotated = ready_order[index + 1 :] + ready_order[:index]
    return [card_id for card_id in rotated if card_id != current_id]


def select_transfer_interleave(
    reports: dict[str, Any],
    config: dict[str, Any],
    *,
    base_family: str,
    rotation: dict[str, Any],
) -> dict[str, Any]:
    cfg = config.get("transfer_interleave_rotation") if isinstance(config.get("transfer_interleave_rotation"), dict) else {}
    enabled = bool(cfg.get("enabled", False))
    stall = rotation.get("public_code_transfer_stall") if isinstance(rotation.get("public_code_transfer_stall"), dict) else {}
    threshold = max(1, int(number(cfg.get("active_attempts_before_cross_family", 6))))
    forced_threshold = max(threshold, int(number(cfg.get("max_same_family_attempts_before_forced_interleave", threshold * 2)) or threshold * 2))
    min_gap = float(number(cfg.get("minimum_public_floor_gap", 0.001)))
    observed_attempts = int(number(stall.get("observed_attempts")))
    floor_gap = float(number(stall.get("floor_gap")))
    selected_id = str(rotation.get("selected_card_id") or "")
    current_id = str(rotation.get("current_card_id") or "")
    reason = str(rotation.get("reason") or "")
    ready_order = [str(item) for item in rotation.get("ready_order", []) if str(item)]
    force_cross_family = observed_attempts >= forced_threshold
    same_family_is_moving = bool(
        cfg.get("same_family_first", True)
        and not force_cross_family
        and selected_id
        and current_id
        and selected_id != current_id
        and reason.startswith("rotate_")
    )
    attempted_public_cards = attempted_public_code_cards(reports, ready_order, family=base_family)
    candidate_blocked = candidate_gate_has_any_failed_gate(reports, public_code_blocking_gates())
    due = bool(
        enabled
        and candidate_blocked
        and not same_family_is_moving
        and observed_attempts >= threshold
        and floor_gap >= min_gap
        and len(attempted_public_cards) >= int(number(cfg.get("minimum_attempted_public_cards_before_cross_family", 3)) or 3)
    )
    base = {
        "enabled": enabled,
        "apply": False,
        "base_family": base_family,
        "return_family": base_family,
        "return_card_id": selected_id or current_id,
        "same_family_first": bool(cfg.get("same_family_first", True)),
        "same_family_is_moving": same_family_is_moving,
        "force_cross_family": force_cross_family,
        "observed_attempts": observed_attempts,
        "active_attempts_before_cross_family": threshold,
        "max_same_family_attempts_before_forced_interleave": forced_threshold,
        "floor_gap": round(floor_gap, 6),
        "minimum_public_floor_gap": min_gap,
        "attempted_public_cards": attempted_public_cards,
        "candidate_blocked_on_public_code": candidate_blocked,
        "teacher_escalation": {
            "role": cfg.get("teacher_role", "architecture_guidance_only_no_benchmark_answers"),
            "after_interleave_attempts": int(number(cfg.get("teacher_after_interleave_attempts", 2)) or 2),
            "currently_due": bool(
                candidate_blocked
                and observed_attempts >= threshold * (int(number(cfg.get("teacher_after_interleave_attempts", 2)) or 2) + 1)
            ),
            "constraint": "teacher diagnoses architecture/residual walls only; no benchmark answers or hidden tests",
        },
        "promotion_semantics": cfg.get(
            "promotion_semantics",
            "interleave scores are transfer pressure only; original promotion gates still apply",
        ),
    }
    if not enabled:
        return {**base, "reason": "transfer_interleave_disabled"}
    if same_family_is_moving:
        return {**base, "reason": "same_family_rotation_precedes_transfer_interleave"}
    if not candidate_blocked:
        return {**base, "reason": "public_code_promotion_gate_not_blocking"}
    if observed_attempts < threshold:
        return {**base, "reason": "below_cross_family_attempt_threshold"}
    if floor_gap < min_gap:
        return {**base, "reason": "public_code_floor_gap_too_small"}
    if len(attempted_public_cards) < int(number(cfg.get("minimum_attempted_public_cards_before_cross_family", 3)) or 3):
        return {**base, "reason": "same_family_public_code_coverage_not_exhausted"}

    cycle = cfg.get("cycle") if isinstance(cfg.get("cycle"), list) else []
    entries = [entry for entry in cycle if isinstance(entry, dict)]
    if not entries:
        return {**base, "reason": "no_transfer_interleave_cycle_configured"}
    start = (observed_attempts // threshold) % len(entries)
    for offset in range(len(entries)):
        entry = entries[(start + offset) % len(entries)]
        resolved = resolve_transfer_interleave_entry(entry, reports)
        if not resolved.get("runnable_now"):
            continue
        return {
            **base,
            **resolved,
            "apply": True,
            "reason": "cross_family_transfer_interleave_due",
            "action": (
                f"Interleave {resolved.get('family')} pressure via {resolved.get('recommended_env')} "
                f"because public code transfer remains {floor_gap:.3f} below floor after "
                f"{observed_attempts} attempts; export/load transfer artifacts and return to {selected_id or current_id}."
            ),
            "verification": [
                "original code frontier remains promotion-blocked until public code floor clears",
                "interleave run must export residuals or transfer artifacts",
                "next code-family run must report whether transfer artifacts were consumed and changed behavior",
                "no public benchmark answers, hidden tests, or teacher solutions enter training",
            ],
        }
    return {**base, "reason": "no_runnable_transfer_interleave_target"}


def resolve_transfer_interleave_entry(entry: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    family = str(entry.get("family") or "")
    runner = str(entry.get("runner_family") or "")
    requested_env = str(entry.get("recommended_env") or "")
    reason = str(entry.get("reason") or "")
    if family == "rl_local":
        recs = get_path(reports, ["rl_registry", "recommended_frontier"], [])
        env = requested_env if requested_env and requested_env != "auto" else ""
        if not env and isinstance(recs, list) and recs:
            env = str(get_path(recs[0], ["name"], "") or "")
        return {
            "family": "rl_local",
            "runner_family": runner or "rl_local",
            "recommended_env": env,
            "runnable_now": bool(env),
            "interleave_reason": reason,
        }
    if family == "transfer_eval":
        tasks = get_path(reports, ["transfer_eval_suite", "tasks"], [])
        return {
            "family": "transfer_eval",
            "runner_family": runner or "transfer_eval_local",
            "recommended_env": requested_env or "transfer_eval_suite",
            "runnable_now": bool(tasks),
            "interleave_reason": reason,
        }
    if family == "conversation_multiturn":
        conversation = reports.get("multi_turn_conversation") if isinstance(reports.get("multi_turn_conversation"), dict) else {}
        return {
            "family": "conversation_multiturn",
            "runner_family": runner or "conversation_multiturn_local",
            "recommended_env": requested_env or "multi_turn_conversation_benchmark",
            "runnable_now": True,
            "interleave_reason": reason,
            "conversation_summary": {
                "passed": conversation.get("passed"),
                "accuracy": get_path(conversation, ["summary", "accuracy"], None),
                "turn_count": get_path(conversation, ["summary", "turn_count"], None),
            },
        }
    if family == "web_agent_local":
        ready = smoke_passed_adapter_cards(reports, categories={"web_agent_benchmark", "web_agent_framework"})
        selected = preferred_card(ready, [requested_env] if requested_env and requested_env != "auto" else ["source_webarena", "source_browsergym"])
        return {
            "family": "web_agent_local",
            "runner_family": runner or "web_agent_local",
            "recommended_env": selected.get("id", ""),
            "runnable_now": bool(selected),
            "interleave_reason": reason,
        }
    if family == "drone_rl":
        ready = smoke_passed_adapter_cards(
            reports,
            categories={"drone_rl_environment", "drone_racing_simulator", "drone_control_api"},
        )
        selected = preferred_card(
            ready,
            [requested_env] if requested_env and requested_env != "auto" else ["source_pyflyt", "source_gym_pybullet_drones", "source_mavsdk_python"],
        )
        return {
            "family": "drone_rl",
            "runner_family": runner or "drone_rl_local",
            "recommended_env": selected.get("id", ""),
            "runnable_now": bool(selected),
            "interleave_reason": reason,
        }
    return {
        "family": family,
        "runner_family": runner,
        "recommended_env": requested_env,
        "runnable_now": False,
        "interleave_reason": reason or "unsupported_interleave_family",
    }


def attempted_public_code_cards(reports: dict[str, Any], ready_order: list[str], *, family: str) -> list[str]:
    attempted: list[str] = []
    for card_id in ready_order:
        if card_id not in PUBLIC_CODE_CALIBRATION_CARDS:
            continue
        if ledger_row_for_card(reports, card_id, family=family):
            attempted.append(card_id)
    return attempted


def candidate_gate_has_failed_gate(reports: dict[str, Any], gate_name: str) -> bool:
    return candidate_gate_has_any_failed_gate(reports, {gate_name})


def candidate_gate_has_any_failed_gate(reports: dict[str, Any], gate_names: set[str]) -> bool:
    gate = reports.get("candidate_gate") if isinstance(reports.get("candidate_gate"), dict) else {}
    failed = {
        str(item.get("gate") or "")
        for item in gate.get("checks", [])
        if isinstance(item, dict) and not item.get("passed")
    }
    return bool(failed & gate_names)


def near_term_queue(stages: list[dict[str, Any]], current: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        stage
        for stage in stages
        if stage.get("status") in {"active_frontier", "ready_next", "blocked_waiting_asset"}
    ]
    if current and current.get("id") and all(row.get("id") != current.get("id") for row in rows):
        rows.insert(0, current)
    return [
        {
            "id": row.get("id"),
            "level": row.get("level"),
            "status": row.get("status"),
            "title": row.get("title"),
            "next_action": row.get("next_action"),
            "blockers": row.get("blockers", []),
        }
        for row in rows[:6]
    ]


def summary(stages: list[dict[str, Any]], current: dict[str, Any], next_frontier: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for stage in stages:
        status = str(stage.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "stage_count": len(stages),
        "status_counts": dict(sorted(counts.items())),
        "locked_stages": counts.get("locked_regression", 0),
        "active_stages": counts.get("active_frontier", 0),
        "ready_stages": counts.get("ready_next", 0),
        "blocked_stages": counts.get("blocked_waiting_asset", 0),
        "future_stages": counts.get("future", 0),
        "current_stage_id": current.get("id"),
        "current_stage_title": current.get("title"),
        "current_stage_status": current.get("status"),
        "next_frontier_family": next_frontier.get("family"),
        "next_frontier_runnable_now": next_frontier.get("runnable_now"),
        "teacher_default_role": "audit_and_propose_only",
    }


def active_count(rows: list[dict[str, Any]]) -> int:
    return len([row for row in rows if row.get("lifecycle") == "frontier"])


def regression_count(rows: list[dict[str, Any]]) -> int:
    return len([row for row in rows if row.get("lifecycle") == "regression"])


if __name__ == "__main__":
    raise SystemExit(main())
