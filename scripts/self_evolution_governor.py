"""Govern the self-evolving SparkStream/RMI loop.

This report is the connective tissue between teacher proposals, local evidence,
benchmark adapter generation, architecture experiment queues, and guarded
source edits. It decides whether the system can keep improving locally, whether
the teacher should propose, and whether the teacher may apply a patch through
the guarded branch-and-gate lane.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "self_evolution_policy.json"
sys.path.insert(0, str(ROOT / "scripts"))
import license_manager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/self_evolution_governance.json")
    parser.add_argument("--markdown-out", default="reports/self_evolution_governance.md")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    state = load_state()
    state["license_status"] = license_manager.status_report(write_report=True)
    state["teacher_license"] = license_manager.check_feature("teacher_bootstrap", write_report=True)
    git_state = git_status()
    lanes = build_lanes(policy, state, git_state)
    report = {
        "policy": "sparkstream_self_evolution_governance_v0",
        "created_utc": now(),
        "policy_file": args.policy,
        "enabled": bool(policy.get("default_enabled", True)),
        "small_model_principle": policy.get("small_model_principle", {}),
        "state": summarize_state(state),
        "git_state": git_state,
        "lanes": lanes,
        "teacher_apply": teacher_apply_decision(policy, state, git_state, lanes),
        "next_actions": next_actions(lanes),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    write_markdown(ROOT / args.markdown_out, report)
    print(json.dumps(report, indent=2))
    return 0


def load_state() -> dict[str, Any]:
    reports = ROOT / "reports"
    return {
        "candidate": read_json(reports / "candidate_promotion_gate.json"),
        "benchmark_ledger": read_json(reports / "benchmark_ledger.json"),
        "residual_escrow": read_json(reports / "residual_escrow.json"),
        "resource_governor": read_json(reports / "resource_governor.json"),
        "autonomy_launch_readiness": read_json(reports / "autonomy_launch_readiness.json"),
        "arm_lifecycle": read_json(reports / "arm_lifecycle_governance.json"),
        "benchmaxx": read_json(reports / "benchmaxx_curriculum.json"),
        "adapter_factory": read_json(reports / "benchmark_adapter_factory.json"),
        "architecture_experiments": read_json(reports / "architecture_experiment_governance.json"),
        "model_growth_gate": read_json(reports / "model_growth_gate.json"),
        "loop_closure": read_json(reports / "loop_closure_harvester.json"),
        "legacy_concepts": read_json(reports / "legacy_project_concept_audit.json"),
        "legacy_ports": read_json(reports / "legacy_port_mechanisms.json"),
        "legacy_port_runtime_enforcement": read_json(reports / "legacy_port_runtime_enforcement.json"),
        "coherence_delirium_gate": read_json(reports / "coherence_delirium_gate.json"),
        "whitecell_threat_memory": read_json(reports / "whitecell_threat_memory.json"),
        "attd": read_json(reports / "attd_report.json"),
        "attd_packets": read_json(reports / "attd_maintenance_packets.json"),
        "external_inference_audit": read_json(reports / "external_inference_audit.json"),
        "license_status": read_json(reports / "license_status.json"),
        "teacher": read_json(reports / "teacher_oracle_last.json"),
    }


def build_lanes(policy: dict[str, Any], state: dict[str, Any], git_state: dict[str, Any]) -> list[dict[str, Any]]:
    failed = set(failed_gates(state.get("candidate") or {}))
    frontier = active_frontier(
        state.get("benchmark_ledger"),
        preferred_family=preferred_curriculum_family(state),
        preferred_card_id=str(get_path(state, ["benchmaxx", "next_frontier", "recommended_env"], "") or ""),
    )
    ready_adapter_cards = get_path(state, ["adapter_factory", "summary", "ready_cards"], 0) or 0
    needs_adapter_smoke = get_path(state, ["adapter_factory", "summary", "needs_smoke"], 0) or 0
    architecture_allowed = bool(get_path(state, ["architecture_experiments", "architecture_change_allowed"], False))
    loop_ready = get_path(state, ["loop_closure", "summary", "ready_for_tool_synthesis"], 0) or 0
    attd_state = str(get_path(state, ["attd", "trigger_state"], "MISSING"))
    attd_blocks_growth = attd_state in {"MISSING", "RED"}
    attd_packet_count = int(get_path(state, ["attd_packets", "packet_count"], 0) or 0)
    attd_teacher_needed = attd_teacher_trigger(policy, state)
    legacy_summary = get_path(state, ["legacy_concepts", "summary"], {}) or {}
    legacy_queue = get_path(state, ["legacy_concepts", "top_port_queue"], []) or []
    legacy_top = legacy_queue[0] if legacy_queue and isinstance(legacy_queue[0], dict) else {}
    legacy_p0_open = int(legacy_summary.get("p0_open", 0) or 0)
    legacy_port_summary = get_path(state, ["legacy_ports", "summary"], {}) or {}
    legacy_port_red = int(legacy_port_summary.get("red", 0) or 0)
    legacy_port_yellow = int(legacy_port_summary.get("yellow_or_degraded", 0) or 0)
    runtime_enforcement = state.get("legacy_port_runtime_enforcement") or {}
    runtime_summary = runtime_enforcement.get("summary") or {}
    coherence_gate = state.get("coherence_delirium_gate") or {}
    lanes = [
        {
            "id": "attd_repo_health_governance",
            "status": "missing_report" if attd_state == "MISSING" else ("blocked_growth" if attd_blocks_growth else ("maintenance_pressure" if attd_state == "YELLOW" else "green")),
            "purpose": "Deterministically prevent autonomous growth from turning source evolution into structural debt.",
            "trigger_state": attd_state,
            "attd_score": get_path(state, ["attd", "attd_score"], None),
            "maintenance_packets": attd_packet_count,
            "next_step": (
                "Run ATTD maintenance packets before feature, adapter, or architecture growth."
                if attd_blocks_growth
                else "Keep ATTD analyzer in every autonomy cycle and let packets guide cleanup."
            ),
        },
        {
            "id": "legacy_concept_port_queue",
            "status": "active" if legacy_p0_open else ("tracked" if legacy_summary else "missing_report"),
            "purpose": "Preserve useful predecessor-project concepts as concrete Theseus ports with acceptance gates.",
            "trigger_state": legacy_summary.get("trigger_state"),
            "p0_open": legacy_p0_open,
            "top_candidate": legacy_top.get("id"),
            "next_step": legacy_top.get("port_goal")
            or "Run scripts/legacy_project_concept_audit.py and keep old-project concepts mapped to current bottlenecks.",
        },
        {
            "id": "legacy_port_mechanisms",
            "status": "blocked" if legacy_port_red else ("watch" if legacy_port_yellow else "ready"),
            "purpose": "Keep predecessor-project ideas materialized as live reports that can drive teacher patches and local fixes.",
            "red": legacy_port_red,
            "yellow_or_degraded": legacy_port_yellow,
            "top_blocker": legacy_port_summary.get("top_blocker"),
            "next_step": (
                f"Fix {legacy_port_summary.get('top_blocker')} before long autonomy."
                if legacy_port_summary.get("top_blocker")
                else "Use reports/legacy_port_mechanisms.json as concrete implementation pressure."
            ),
        },
        {
            "id": "coherence_delirium_governance",
            "status": "ready" if coherence_gate.get("allows_self_edit") else "blocked",
            "purpose": "Prevent high-delirium runtime states from self-editing, expanding capability, or promoting candidates.",
            "trigger_state": coherence_gate.get("trigger_state"),
            "source_trigger_state": coherence_gate.get("source_trigger_state"),
            "coherence_score": coherence_gate.get("coherence_score"),
            "delirium_score": coherence_gate.get("delirium_score"),
            "allows_self_edit": coherence_gate.get("allows_self_edit"),
            "allows_capability_expansion": coherence_gate.get("allows_capability_expansion"),
            "blockers": coherence_gate.get("blockers", []),
            "next_step": (
                "Pause teacher apply and repair coherence/delirium blockers before self-evolution."
                if not coherence_gate.get("allows_self_edit")
                else "Keep coherence/delirium gate in teacher apply evidence."
            ),
        },
        {
            "id": "taskspell_effect_replay_enforcement",
            "status": "ready" if runtime_enforcement.get("ready_for_bounded_autonomy") else "blocked",
            "purpose": "Force every training/self-evolution/runtime action to carry a TaskSpell lock, effect record, replay refs, and verification record.",
            "trigger_state": runtime_summary.get("trigger_state"),
            "ready_for_bounded_autonomy": runtime_enforcement.get("ready_for_bounded_autonomy"),
            "ready_for_long_autonomy": runtime_enforcement.get("ready_for_long_autonomy"),
            "ready_for_self_evolution": runtime_enforcement.get("ready_for_self_evolution"),
            "effect_records": runtime_summary.get("effect_records"),
            "blockers": runtime_enforcement.get("blockers", []),
            "next_step": (
                "Clear runtime enforcement blockers before teacher apply; TaskSpell effect/replay remains mandatory."
                if not runtime_enforcement.get("ready_for_self_evolution")
                else "Attach the runtime enforcement report to teacher evidence packets and checkpoint decisions."
            ),
        },
        {
            "id": "guarded_teacher_self_edit",
            "status": "ready" if architecture_allowed or attd_teacher_needed else "waiting_for_architecture_wall_or_gates",
            "purpose": "Teacher can implement small source changes on an isolated branch, then local tests and regressions decide.",
            "blocked_by": blocked_self_edit(policy, state, git_state, architecture_allowed, attd_teacher_needed),
            "next_step": (
                "Run scripts/teacher_self_edit_runner.py with reason=attd_maintenance to consume ATTD packets."
                if attd_teacher_needed
                else "Run scripts/teacher_self_edit_runner.py when architecture experiment governance says a patch is justified."
            ),
        },
        {
            "id": "benchmark_adapter_factory",
            "status": "ready" if ready_adapter_cards or needs_adapter_smoke else "discover_more_sources",
            "purpose": "Convert source catalog and local ROM profiles into benchmark cards, loaders, scorers, smokes, and regression policies.",
            "ready_cards": ready_adapter_cards,
            "needs_smoke": needs_adapter_smoke,
            "next_step": "Build the smallest adapter smoke for the highest-priority ready card.",
        },
        {
            "id": "architecture_experiment_search",
            "status": "ready" if state.get("architecture_experiments") else "report_missing",
            "purpose": "Run controlled zero-param-first experiments, compare variants, preserve winners, and retire losers.",
            "recommended_experiment": get_path(state, ["architecture_experiments", "recommended_next_experiment", "id"], None),
            "architecture_change_allowed": architecture_allowed,
            "next_step": get_path(state, ["architecture_experiments", "recommended_next_experiment", "hypothesis"], "Refresh architecture experiment governance."),
        },
        {
            "id": "aggressive_loop_closure",
            "status": "ready" if loop_ready else "collect_more_recurrence",
            "purpose": "Compile repeated successful workflows into verified tools before adding model capacity.",
            "ready_tool_candidates": loop_ready,
            "next_step": "Convert high-recurrence low-risk candidates into tool registry entries with schema checks.",
        },
        {
            "id": "open_ended_transfer_curriculum",
            "status": "active",
            "purpose": "Move through language, synthetic compression, RL, coding, web, desktop, voice, and self-repair stages.",
            "current_stage": get_path(state, ["benchmaxx", "summary", "current_stage_id"], None),
            "next_frontier_family": get_path(state, ["benchmaxx", "summary", "next_frontier_family"], None),
            "frontier": frontier.get("benchmark_name"),
            "failed_candidate_gates": sorted(failed),
            "next_step": "Continue active frontier pressure until mastery; then rotate rather than rerun saturated surfaces.",
        },
        {
            "id": "compute_efficiency",
            "status": "green" if get_path(state, ["resource_governor", "decision", "can_run_requested_profile"], True) else "throttled",
            "purpose": "Keep hot loops in Rust/CUDA and spend the RTX 2060 Super where it produces measured progress.",
            "efficiency_score": get_path(state, ["resource_governor", "efficiency", "score"], None),
            "recommended_profile": get_path(state, ["resource_governor", "decision", "recommended_profile"], None),
            "next_step": "Prefer smoke -> inner_loop -> candidate; grow compute only after metrics justify it.",
        },
    ]
    return lanes


def teacher_apply_decision(
    policy: dict[str, Any],
    state: dict[str, Any],
    git_state: dict[str, Any],
    lanes: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = policy.get("guarded_self_edit") or {}
    lane = next((item for item in lanes if item.get("id") == "guarded_teacher_self_edit"), {})
    failed = set(failed_gates(state.get("candidate") or {}))
    frontier = active_frontier(
        state.get("benchmark_ledger"),
        preferred_family=preferred_curriculum_family(state),
        preferred_card_id=str(get_path(state, ["benchmaxx", "next_frontier", "recommended_env"], "") or ""),
    )
    trigger_gates = set(cfg.get("trigger_failed_gates", []))
    trigger_walls = set(cfg.get("trigger_wall_types", []))
    recommended_experiment = get_path(state, ["architecture_experiments", "recommended_next_experiment"], {})
    recommended_needs_teacher = bool(
        isinstance(recommended_experiment, dict)
        and (
            recommended_experiment.get("teacher_needed")
            or str(recommended_experiment.get("kind", "")).startswith("minimal_architecture")
        )
    )
    architecture_trigger = bool((failed & trigger_gates) or frontier.get("wall_type") in trigger_walls)
    attd_trigger = attd_teacher_trigger(policy, state)
    model_growth_allowed = bool(get_path(state, ["model_growth_gate", "model_growth_allowed"], False))
    trigger = bool(attd_trigger or architecture_trigger)
    clean_required = bool(cfg.get("requires_clean_worktree", True))
    dirty = bool(git_state.get("dirty"))
    auto_checkpoint_dirty = bool(cfg.get("auto_commit_dirty_worktree", False))
    dirty_blocks = bool(clean_required and dirty and not auto_checkpoint_dirty)
    attd_state = str(get_path(state, ["attd", "trigger_state"], "MISSING"))
    maintenance_reason = str(
        get_path(
            state,
            ["attd", "governance", "teacher_self_edit_exception_reason"],
            get_path(policy, ["attd_maintenance", "teacher_reason"], "attd_maintenance"),
        )
        or get_path(policy, ["attd_maintenance", "teacher_reason"], "attd_maintenance")
    )
    teacher_reason = maintenance_reason if attd_trigger else "architecture_wall"
    attd_allows_teacher = bool(get_path(state, ["attd", "governance", "allows_teacher_self_edit"], False))
    attd_allows_for_reason = bool(attd_allows_teacher or (attd_trigger and teacher_reason == maintenance_reason))
    whitecell_blockers = active_whitecell_blockers(state)
    coherence_allows_self_edit = bool(get_path(state, ["coherence_delirium_gate", "allows_self_edit"], False))
    runtime_allows_self_evolution = bool(
        get_path(state, ["legacy_port_runtime_enforcement", "ready_for_self_evolution"], True)
    )
    allowed = bool(
        cfg.get("enabled", True)
        and cfg.get("auto_apply_enabled", False)
        and trigger
        and (attd_trigger or recommended_needs_teacher)
        and (attd_trigger or model_growth_allowed)
        and lane.get("status") == "ready"
        and not dirty_blocks
        and attd_allows_for_reason
        and not whitecell_blockers
        and coherence_allows_self_edit
        and runtime_allows_self_evolution
        and bool(get_path(state, ["teacher_license", "allowed"], False))
        and get_path(state, ["external_inference_audit", "ok"], True)
    )
    blockers: list[str] = []
    if not cfg.get("enabled", True):
        blockers.append("self_edit_disabled")
    if not cfg.get("auto_apply_enabled", False):
        blockers.append("auto_apply_disabled")
    if not trigger:
        blockers.append("no_architecture_or_gate_trigger")
    if not attd_trigger and not recommended_needs_teacher:
        blockers.append("lower_cost_local_experiment_pending")
    if not attd_trigger and not model_growth_allowed:
        blockers.append("model_growth_gate_blocks_architecture_change")
    if lane.get("status") != "ready":
        blockers.append(str(lane.get("status")))
    if dirty_blocks:
        blockers.append("dirty_worktree_requires_branch_handoff_or_clean_commit")
    if not attd_allows_for_reason:
        blockers.append("attd_red_requires_maintenance_before_teacher_self_edit")
    for blocker in whitecell_blockers:
        blockers.append(f"whitecell_block_and_escalate:{blocker}")
    if not coherence_allows_self_edit:
        blockers.append("coherence_delirium_blocks_self_edit")
    if not runtime_allows_self_evolution:
        blockers.append("legacy_port_runtime_enforcement_blocks_self_evolution")
    if not get_path(state, ["teacher_license", "allowed"], False):
        blockers.append("license_blocks_teacher_bootstrap")
    if not get_path(state, ["external_inference_audit", "ok"], True):
        blockers.append("external_inference_audit_failed")
    return {
        "auto_apply_enabled": bool(cfg.get("auto_apply_enabled", False)),
        "allowed_now": allowed,
        "mode": cfg.get("teacher_mode", "apply"),
        "branch_prefix": cfg.get("branch_prefix"),
        "dirty_worktree_checkpoint_policy": {
            "requires_clean_worktree": clean_required,
            "dirty": dirty,
            "auto_commit_dirty_worktree": auto_checkpoint_dirty,
            "dirty_blocks_teacher_apply": dirty_blocks,
            "auto_commit_message": cfg.get("auto_commit_message"),
        },
        "license": {
            "teacher_bootstrap_allowed": get_path(state, ["teacher_license", "allowed"], False),
            "tier": get_path(state, ["license_status", "entitlement", "tier"], None),
            "source": get_path(state, ["license_status", "entitlement", "source"], None),
            "next_action": get_path(state, ["license_status", "next_action"], None),
        },
        "whitecell": {
            "trigger_state": get_path(state, ["whitecell_threat_memory", "trigger_state"], None),
            "active_block_and_escalate": whitecell_blockers,
            "local_only": get_path(state, ["whitecell_threat_memory", "local_only"], None),
        },
        "coherence_delirium": {
            "trigger_state": get_path(state, ["coherence_delirium_gate", "trigger_state"], None),
            "source_trigger_state": get_path(state, ["coherence_delirium_gate", "source_trigger_state"], None),
            "coherence_score": get_path(state, ["coherence_delirium_gate", "coherence_score"], None),
            "delirium_score": get_path(state, ["coherence_delirium_gate", "delirium_score"], None),
            "allows_self_edit": coherence_allows_self_edit,
            "blockers": get_path(state, ["coherence_delirium_gate", "blockers"], []),
        },
        "legacy_port_runtime_enforcement": {
            "trigger_state": get_path(state, ["legacy_port_runtime_enforcement", "summary", "trigger_state"], None),
            "ready_for_bounded_autonomy": get_path(state, ["legacy_port_runtime_enforcement", "ready_for_bounded_autonomy"], None),
            "ready_for_long_autonomy": get_path(state, ["legacy_port_runtime_enforcement", "ready_for_long_autonomy"], None),
            "ready_for_self_evolution": runtime_allows_self_evolution,
            "blockers": get_path(state, ["legacy_port_runtime_enforcement", "blockers"], []),
        },
        "triggered_by": {
            "failed_gates": sorted(failed & trigger_gates),
            "wall_type": frontier.get("wall_type") if architecture_trigger and frontier.get("wall_type") in trigger_walls else None,
            "recommended_experiment": recommended_experiment.get("id") if isinstance(recommended_experiment, dict) else None,
            "recommended_teacher_needed": recommended_needs_teacher,
            "attd_trigger_state": attd_state,
            "attd_maintenance_packets": get_path(state, ["attd_packets", "packet_count"], 0),
            "model_growth_allowed": model_growth_allowed,
            "primary_reason": teacher_reason,
        },
        "blockers": blockers,
        "runner_command": [
            "python",
            "scripts/teacher_self_edit_runner.py",
            "--execute",
            "--allow-teacher",
            "--reason",
            teacher_reason,
            "--out",
            "reports/teacher_self_edit_last.json",
        ],
    }


def attd_teacher_trigger(policy: dict[str, Any], state: dict[str, Any]) -> bool:
    cfg = policy.get("attd_maintenance") or {}
    if not cfg.get("enabled", True):
        return False
    attd_state = str(get_path(state, ["attd", "trigger_state"], "MISSING"))
    packet_count = int(get_path(state, ["attd_packets", "packet_count"], 0) or 0)
    if packet_count <= 0:
        return False
    if attd_state == "RED":
        return bool(cfg.get("auto_apply_on_red", True))
    if attd_state == "YELLOW":
        return bool(cfg.get("auto_apply_on_yellow", True))
    return False


def blocked_self_edit(
    policy: dict[str, Any],
    state: dict[str, Any],
    git_state: dict[str, Any],
    architecture_allowed: bool,
    attd_teacher_needed: bool,
) -> list[str]:
    blockers: list[str] = []
    cfg = policy.get("guarded_self_edit") or {}
    if not cfg.get("enabled", True):
        blockers.append("disabled_by_policy")
    if not architecture_allowed and not attd_teacher_needed:
        blockers.append("architecture_ladder_not_yet_at_patch")
    if (
        cfg.get("requires_clean_worktree", True)
        and git_state.get("dirty")
        and not cfg.get("auto_commit_dirty_worktree", False)
    ):
        blockers.append("dirty_worktree")
    if not get_path(state, ["coherence_delirium_gate", "allows_self_edit"], False):
        blockers.append("coherence_delirium")
    if not get_path(state, ["legacy_port_runtime_enforcement", "ready_for_self_evolution"], True):
        blockers.append("legacy_port_runtime_enforcement")
    return blockers


def active_whitecell_blockers(state: dict[str, Any]) -> list[str]:
    patterns = get_path(state, ["whitecell_threat_memory", "threat_patterns"], [])
    if not isinstance(patterns, list):
        return []
    blockers = []
    for row in patterns:
        if not isinstance(row, dict):
            continue
        if row.get("active") and row.get("action") == "block_and_escalate":
            blockers.append(str(row.get("pattern_id") or "unknown_whitecell_pattern"))
    return sorted(set(blockers))


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    candidate = state.get("candidate") or {}
    frontier = active_frontier(
        state.get("benchmark_ledger"),
        preferred_family=preferred_curriculum_family(state),
        preferred_card_id=str(get_path(state, ["benchmaxx", "next_frontier", "recommended_env"], "") or ""),
    )
    return {
        "candidate_promote": candidate.get("promote"),
        "candidate_gate": f"{candidate.get('passed')}/{candidate.get('total')}",
        "failed_candidate_gates": failed_gates(candidate),
        "frontier": frontier.get("benchmark_name"),
        "frontier_score": frontier.get("score"),
        "frontier_wall_type": frontier.get("wall_type"),
        "residual_clusters": get_path(state, ["residual_escrow", "summary", "cluster_count"], None),
        "launch_ready": get_path(state, ["autonomy_launch_readiness", "ready_for_autonomous_training"], None),
        "teacher_enabled_ready": get_path(state, ["autonomy_launch_readiness", "ready_for_teacher_enabled_run"], None),
        "adapter_cards": get_path(state, ["adapter_factory", "summary", "cards"], None),
        "architecture_change_allowed": get_path(state, ["architecture_experiments", "architecture_change_allowed"], None),
        "model_growth_allowed": get_path(state, ["model_growth_gate", "model_growth_allowed"], None),
        "model_growth_blockers": get_path(state, ["model_growth_gate", "hard_blockers"], []) + get_path(state, ["model_growth_gate", "missing_evidence"], []),
        "loop_tool_candidates": get_path(state, ["loop_closure", "summary", "candidates"], None),
        "legacy_concept_trigger_state": get_path(state, ["legacy_concepts", "summary", "trigger_state"], None),
        "legacy_p0_open": get_path(state, ["legacy_concepts", "summary", "p0_open"], None),
        "legacy_top_candidate": get_path(state, ["legacy_concepts", "summary", "top_candidate"], None),
        "legacy_port_red": get_path(state, ["legacy_ports", "summary", "red"], None),
        "legacy_port_yellow_or_degraded": get_path(state, ["legacy_ports", "summary", "yellow_or_degraded"], None),
        "legacy_port_top_blocker": get_path(state, ["legacy_ports", "summary", "top_blocker"], None),
        "legacy_port_runtime_enforcement": {
            "trigger_state": get_path(state, ["legacy_port_runtime_enforcement", "summary", "trigger_state"], None),
            "ready_for_bounded_autonomy": get_path(state, ["legacy_port_runtime_enforcement", "ready_for_bounded_autonomy"], None),
            "ready_for_long_autonomy": get_path(state, ["legacy_port_runtime_enforcement", "ready_for_long_autonomy"], None),
            "ready_for_self_evolution": get_path(state, ["legacy_port_runtime_enforcement", "ready_for_self_evolution"], None),
            "blockers": get_path(state, ["legacy_port_runtime_enforcement", "blockers"], []),
        },
        "coherence_delirium_gate": {
            "trigger_state": get_path(state, ["coherence_delirium_gate", "trigger_state"], None),
            "coherence_score": get_path(state, ["coherence_delirium_gate", "coherence_score"], None),
            "delirium_score": get_path(state, ["coherence_delirium_gate", "delirium_score"], None),
            "allows_self_edit": get_path(state, ["coherence_delirium_gate", "allows_self_edit"], None),
            "allows_capability_expansion": get_path(state, ["coherence_delirium_gate", "allows_capability_expansion"], None),
        },
        "whitecell_trigger_state": get_path(state, ["whitecell_threat_memory", "trigger_state"], None),
        "whitecell_active_blockers": active_whitecell_blockers(state),
        "attd_trigger_state": get_path(state, ["attd", "trigger_state"], None),
        "attd_score": get_path(state, ["attd", "attd_score"], None),
        "attd_maintenance_packets": get_path(state, ["attd_packets", "packet_count"], None),
    }


def next_actions(lanes: list[dict[str, Any]]) -> list[str]:
    rows = []
    for lane in lanes:
        step = lane.get("next_step")
        if step:
            rows.append(f"{lane.get('id')}: {step}")
    return rows


def git_status() -> dict[str, Any]:
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc), "dirty": True}
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "available": branch.returncode == 0 and status.returncode == 0,
        "branch": branch.stdout.strip(),
        "dirty": bool(lines),
        "porcelain_count": len(lines),
        "porcelain_sample": lines[:30],
    }


def active_frontier(ledger: Any, *, preferred_family: str = "", preferred_card_id: str = "") -> dict[str, Any]:
    if not isinstance(ledger, list):
        return {}
    frontiers = [row for row in ledger if isinstance(row, dict) and row.get("lifecycle") == "frontier"]
    if not frontiers:
        return {}
    preferred = [
        row
        for row in frontiers
        if row_frontier_family(row) == preferred_family
        and (not preferred_card_id or preferred_card_id in str(row.get("best_report") or row.get("benchmark_name") or ""))
    ]
    if preferred:
        return max(preferred, key=lambda row: float(row.get("residual") or 0.0))
    preferred = [row for row in frontiers if row_frontier_family(row) == preferred_family]
    if preferred:
        return max(preferred, key=lambda row: float(row.get("residual") or 0.0))
    return max(frontiers, key=lambda row: float(row.get("residual") or 0.0))


def preferred_curriculum_family(state: dict[str, Any]) -> str:
    runner = str(get_path(state, ["benchmaxx", "next_frontier", "runner_family"], "") or "")
    family = str(get_path(state, ["benchmaxx", "next_frontier", "family"], "") or "")
    mapped = {
        "minecraft_rl_local": "minecraft_rl",
        "drone_rl_local": "drone_rl",
        "coding_local_sandbox": "coding_local_sandbox",
        "web_agent_local": "web_agent_local",
        "transfer_eval_local": "transfer_eval",
        "rl_local": "rl_local",
        "babylm_mutated": "babylm_mutated",
    }
    return mapped.get(runner, family)


def row_frontier_family(row: dict[str, Any]) -> str:
    name = str(row.get("benchmark_name") or "")
    benchmark_type = str(row.get("benchmark_type") or "")
    best_report = str(row.get("best_report") or "")
    if name == "babylm_mutated_holdout" or "mutated" in benchmark_type:
        return "babylm_mutated"
    if name.startswith("ocean-"):
        return "rl_local"
    if name.startswith("minecraft_rl_") or "minecraft" in name:
        return "minecraft_rl"
    if name.startswith("drone_rl_") or name.startswith("drone_control_"):
        return "drone_rl"
    if name.startswith("coding_"):
        return "coding_local_sandbox"
    if name.startswith("web_agent_"):
        return "web_agent_local"
    if name.startswith("transfer_") or name.startswith("asi_transfer"):
        return "transfer_eval"
    if any(source in best_report for source in ["source_crafter", "source_craftax", "source_minerl", "source_minedojo", "source_malmo", "source_voyager_minecraft"]):
        return "minecraft_rl"
    if "source_gym_pybullet" in best_report or "source_pyflyt" in best_report or "source_mavsdk" in best_report:
        return "drone_rl"
    if "source_bigcodebench" in best_report:
        return "coding_local_sandbox"
    if "source_webarena" in best_report:
        return "web_agent_local"
    return ""


def failed_gates(candidate: dict[str, Any]) -> list[str]:
    return [str(item.get("gate")) for item in candidate.get("checks", []) if isinstance(item, dict) and not item.get("passed")]


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = [
        "# Self-Evolution Governance",
        "",
        f"Updated: {report.get('created_utc')}",
        "",
        "## State",
        "",
    ]
    for key, value in (report.get("state") or {}).items():
        rows.append(f"- {key}: {value}")
    rows.extend(["", "## Teacher Apply", ""])
    teacher = report.get("teacher_apply") or {}
    rows.append(f"- allowed_now: {teacher.get('allowed_now')}")
    rows.append(f"- blockers: {', '.join(teacher.get('blockers') or []) or 'none'}")
    rows.extend(["", "## Lanes", ""])
    for lane in report.get("lanes", []):
        rows.append(f"- {lane.get('id')}: {lane.get('status')} - {lane.get('next_step')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


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
