"""Support helpers for SparkStream autonomy cycle decisions and reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def goal_for_decision(reason: str, frontier: dict[str, Any], pressure: dict[str, Any]) -> str:
    if pressure.get("candidate_profile_required"):
        return (
            "Run the candidate evidence profile once on a fresh BabyLM frontier so promotion gates "
            "can close automatically instead of rerunning saturated frontiers indefinitely."
        )
    if pressure.get("needs_fresh_frontier"):
        if pressure.get("next_frontier_family") == "rl_local":
            return (
                f"Run local RL frontier {pressure.get('next_rl_frontier_env')} "
                f"seed{pressure.get('next_rl_frontier_seed')}; preserve saturated BabyLM/RL surfaces as regressions "
                "and keep residual escrow active."
            )
        if pressure.get("next_frontier_family") in {
            "minecraft_rl",
            "drone_rl",
            "coding_local_sandbox",
            "web_agent_local",
            "transfer_eval",
        }:
            return (
                f"Run local pressure card {pressure.get('next_pressure_card_id') or pressure.get('next_frontier_family')} "
                f"for {pressure.get('next_frontier_family')}; preserve saturated surfaces as regressions "
                "and keep residual escrow active."
            )
        if pressure.get("curriculum_adapter_required"):
            return (
                f"Build the smallest local adapter smoke for curriculum frontier "
                f"{pressure.get('next_frontier_family')}; do not rerun saturated surfaces as frontier pressure."
            )
        return (
            f"Create and attack fresh mutated BabyLM seed{pressure.get('next_mutated_babylm_seed')} "
            "as the next frontier; preserve all prior saturated surfaces as regressions and keep residual escrow active."
        )
    if pressure.get("curriculum_adapter_required"):
        return (
            f"Prepare the next curriculum lane {pressure.get('next_frontier_family')} by staging a local adapter, "
            "benchmark card, and smoke report before training resumes."
        )
    if frontier:
        return f"Improve active frontier {frontier.get('benchmark_name')} efficiently, then escrow residuals and graduate on mastery."
    return "Find or generate a harder licensed frontier benchmark before doing more training."

def should_call_teacher(policy: dict[str, Any], state: dict[str, Any], decision: dict[str, Any]) -> bool:
    if decision.get("reason") in {
        "repair_preflight_before_training",
        "candidate_profile_evidence_required",
        "continue_active_frontier_pressure",
        "curriculum_runnable_frontier_override",
        "active_frontier_below_floor_continue_or_bridge",
    }:
        # These are ordinary local ratchet states. They need local repair,
        # profiling, rotation, or more pressure before spending teacher budget.
        return False
    if decision.get("reason") in {
        "frontier_exhausted_rotate_to_fresh_frontier",
        "candidate_promoted_rotate_to_fresh_frontier",
        "promotion_rotation_request_pending",
    }:
        # Routine ratchet rotation is local machinery. Candidate-promotion
        # gates may still be red, but they should not spend teacher budget
        # unless a later cycle exposes a real wall, throttle, or safety issue.
        return False
    if decision.get("reason") == "curriculum_adapter_required_before_training":
        return False
    pressure = decision.get("frontier_pressure") or {}
    if pressure.get("architecture_upgrade_due"):
        return True
    if pressure.get("frontier_exhausted") and not pressure.get("candidate_profile_required"):
        return True
    failed = set(decision.get("failed_candidate_gates") or [])
    trigger_gates = set(get_path(policy, ["teacher_escalation", "trigger_gates"], []))
    if failed & trigger_gates and pressure.get("architecture_upgrade_due"):
        return True
    frontier = decision.get("frontier") or {}
    wall = frontier.get("wall_type")
    if wall in set(get_path(policy, ["teacher_escalation", "trigger_wall_types"], [])):
        score = frontier.get("score")
        floor = get_path(frontier, ["graduation_policy", "floor_threshold"], None)
        attempt_count = int(get_path(frontier, ["graduation_policy", "attempt_count"], 0) or 0)
        threshold = int(get_path(policy, ["teacher_escalation", "frontier_wall_attempt_threshold"], 3))
        if isinstance(score, (int, float)) and isinstance(floor, (int, float)) and score < floor and attempt_count >= max(1, threshold):
            return True
    preflight = state.get("preflight") or {}
    arm_governance = state.get("arm_lifecycle_governance") or {}
    if (
        get_path(arm_governance, ["teacher_escalation", "recommended"], False)
        and decision.get("reason")
        not in {
            "candidate_profile_evidence_required",
            "continue_active_frontier_pressure",
            "frontier_exhausted_rotate_to_fresh_frontier",
            "candidate_promoted_rotate_to_fresh_frontier",
        }
    ):
        return True
    return False

def teacher_evidence(state: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    candidate = state.get("candidate_gate") or {}
    frontier = decision.get("frontier") or {}
    pressure = decision.get("frontier_pressure") or {}
    evidence = [
        f"decision_reason={decision.get('reason')}",
        f"failed_candidate_gates={','.join(decision.get('failed_candidate_gates') or [])}",
        f"frontier={frontier.get('benchmark_name')} score={frontier.get('score')} floor={get_path(frontier, ['graduation_policy', 'floor_threshold'], None)} wall={frontier.get('wall_type')}",
        f"frontier_pressure=frontiers:{pressure.get('frontier_count')} regressions:{pressure.get('regression_count')} exhausted:{pressure.get('frontier_exhausted')} candidate_required:{pressure.get('candidate_profile_required')} next_seed:{pressure.get('next_mutated_babylm_seed')}",
        f"architecture_upgrade_due={pressure.get('architecture_upgrade_due')} below_floor_stagnation={pressure.get('below_floor_stagnation')}",
        f"candidate_promote={candidate.get('promote')} passed={candidate.get('passed')}/{candidate.get('total')}",
        f"residual_delta={candidate.get('residual_delta')}",
        f"arm_lifecycle_ready={get_path(state, ['arm_lifecycle_governance', 'ready_for_long_autonomy'], None)} proposals={get_path(state, ['arm_lifecycle_governance', 'summary', 'proposal_count'], None)} teacher_reason={get_path(state, ['arm_lifecycle_governance', 'teacher_escalation', 'reason'], None)}",
        f"benchmaxx_current={get_path(state, ['benchmaxx_curriculum', 'summary', 'current_stage_id'], None)} next={get_path(state, ['benchmaxx_curriculum', 'summary', 'next_frontier_family'], None)} blocked={get_path(state, ['benchmaxx_curriculum', 'summary', 'blocked_stages'], None)}",
        f"adapter_factory_cards={get_path(state, ['benchmark_adapter_factory', 'summary', 'cards'], None)} needs_smoke={get_path(state, ['benchmark_adapter_factory', 'summary', 'needs_smoke'], None)} blocked={get_path(state, ['benchmark_adapter_factory', 'summary', 'blocked'], None)}",
        f"architecture_experiment={get_path(state, ['architecture_experiment_governance', 'recommended_next_experiment', 'id'], None)} architecture_change_allowed={get_path(state, ['architecture_experiment_governance', 'architecture_change_allowed'], None)}",
        f"architecture_runner_status={get_path(state, ['architecture_experiment_runner', 'status'], None)} selected={get_path(state, ['architecture_experiment_runner', 'selected'], [])[:1]}",
        f"autoresearch_audit={get_path(state, ['autoresearch_gap_audit', 'summary', 'trigger_state'], None)} gaps={get_path(state, ['autoresearch_gap_audit', 'summary', 'gap_count'], None)} needs_baseline={get_path(state, ['autoresearch_gap_audit', 'summary', 'needs_baseline'], None)}",
        f"legacy_concepts={get_path(state, ['legacy_concepts', 'summary', 'trigger_state'], None)} p0_open={get_path(state, ['legacy_concepts', 'summary', 'p0_open'], None)} top={get_path(state, ['legacy_concepts', 'summary', 'top_candidate'], None)}",
        f"legacy_runtime_enforcement={get_path(state, ['legacy_port_runtime_enforcement', 'summary', 'trigger_state'], None)} bounded={get_path(state, ['legacy_port_runtime_enforcement', 'ready_for_bounded_autonomy'], None)} long={get_path(state, ['legacy_port_runtime_enforcement', 'ready_for_long_autonomy'], None)} blockers={get_path(state, ['legacy_port_runtime_enforcement', 'blockers'], [])}",
        f"loop_closure_candidates={get_path(state, ['loop_closure_harvester', 'summary', 'candidates'], None)} ready_tools={get_path(state, ['loop_closure_harvester', 'summary', 'ready_for_tool_synthesis'], None)}",
        f"arm_suckers_ready={get_path(state, ['arm_sucker_registry', 'summary', 'ready_suckers'], None)} blocked={get_path(state, ['arm_sucker_registry', 'summary', 'blocked_suckers'], None)} top={get_path(state, ['arm_sucker_registry', 'summary', 'top_ready_suckers'], [])[:3]}",
        f"arm_transfer_family={get_path(state, ['arm_transfer_plan', 'summary', 'frontier_family'], None)} ready_edges={get_path(state, ['arm_transfer_plan', 'summary', 'ready_edges'], None)} blocked_edges={get_path(state, ['arm_transfer_plan', 'summary', 'blocked_edges'], None)}",
        f"arm_transfer_artifacts={get_path(state, ['arm_transfer_artifacts', 'summary', 'artifacts'], None)}",
        f"self_evolution_apply_allowed={get_path(state, ['self_evolution_governance', 'teacher_apply', 'allowed_now'], None)} self_evolution_blockers={get_path(state, ['self_evolution_governance', 'teacher_apply', 'blockers'], [])}",
        f"teacher_self_edit_proof_status={get_path(state, ['teacher_self_edit_proof', 'status'], None)} success_rate={get_path(state, ['teacher_self_edit_proof', 'summary', 'success_rate'], None)}",
        f"promotion_closure_status={get_path(state, ['promotion_closure', 'status'], None)} rotation_request={get_path(state, ['frontier_rotation_request', 'request_id'], None)} consumed={get_path(state, ['frontier_rotation_request', 'consumed_utc'], None)}",
        f"attd_state={get_path(state, ['attd', 'trigger_state'], None)} attd_score={get_path(state, ['attd', 'attd_score'], None)} packets={get_path(state, ['attd_maintenance_packets', 'packet_count'], None)}",
        f"coherence_delirium={get_path(state, ['coherence_delirium_gate', 'trigger_state'], None)} coherence={get_path(state, ['coherence_delirium_gate', 'coherence_score'], None)} delirium={get_path(state, ['coherence_delirium_gate', 'delirium_score'], None)} allows_long_autonomy={get_path(state, ['coherence_delirium_gate', 'allows_long_autonomy'], None)}",
    ]
    return evidence

def build_self_improvement_queue(
    state: dict[str, Any], decision: dict[str, Any], teacher_result: dict[str, Any] | None
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    external_audit = state.get("external_inference_audit") or {}
    if external_audit and not external_audit.get("ok", False):
        items.append(
            {
                "priority": "critical",
                "kind": "external_inference_boundary",
                "title": "External inference boundary violation",
                "suggested_action": "stop ratcheting until every non-teacher inference path is removed or routed through teacher_oracle.py",
                "evidence": external_audit.get("summary"),
            }
        )
    viea_spine = state.get("viea_autonomy_spine") or {}
    viea_action_executor = state.get("viea_action_executor") or {}
    feedback_queue = state.get("feedback_action_queue") or {}
    broad_queue = state.get("broad_transfer_action_queue") or {}
    repo_repair = state.get("repo_repair_main_curriculum") or {}
    if viea_spine:
        summary = viea_spine.get("summary") if isinstance(viea_spine.get("summary"), dict) else {}
        items.append(
            {
                "priority": "high" if viea_spine.get("trigger_state") == "YELLOW" else "medium",
                "kind": "viea_autonomy_spine",
                "title": f"VIEA spine {viea_spine.get('trigger_state')} actions={summary.get('feedback_action_count')}",
                "suggested_action": "Execute the feedback action queue while preserving public benchmarks as calibration-only.",
                "evidence": {
                    "artifact_kernel_objects": summary.get("artifact_kernel_objects"),
                    "broad_transfer_floor_gap": summary.get("broad_transfer_floor_gap"),
                    "repo_repair_task_count": summary.get("repo_repair_task_count"),
                    "teacher_architect_experiments": summary.get("teacher_architect_experiments"),
                },
            }
        )
    if viea_action_executor:
        summary = viea_action_executor.get("summary") if isinstance(viea_action_executor.get("summary"), dict) else {}
        items.append(
            {
                "priority": "high" if int(summary.get("ready_action_count") or 0) else "medium",
                "kind": "viea_action_executor",
                "title": f"VIEA action executor ready={summary.get('ready_action_count')} completed={summary.get('completed_total')}",
                "suggested_action": "Run approved queue actions with step budgets, resume ledger, and public-calibration-only gates.",
                "evidence": summary,
            }
        )
    for source_name, queue_payload in [
        ("feedback_action_queue", feedback_queue),
        ("broad_transfer_action_queue", broad_queue),
    ]:
        actions = queue_payload.get("actions") if isinstance(queue_payload.get("actions"), list) else []
        for queue_action in actions[:8]:
            if not isinstance(queue_action, dict):
                continue
            items.append(
                {
                    "priority": queue_action.get("priority", "medium"),
                    "kind": f"viea_{queue_action.get('kind') or source_name}",
                    "title": queue_action.get("title") or "VIEA queued action",
                    "suggested_action": queue_action.get("suggested_action"),
                    "evidence": {
                        "source": source_name,
                        "command": queue_action.get("command", []),
                        "public_data_rule": queue_action.get("public_data_rule"),
                    },
                }
            )
    if repo_repair:
        items.append(
            {
                "priority": "high",
                "kind": "private_repo_repair_main_curriculum",
                "title": f"Repo repair central curriculum tasks={get_path(repo_repair, ['summary', 'task_count'], 0)}",
                "suggested_action": "Use inspect-patch-test-repair private traces as the main long-horizon programming pressure.",
                "evidence": repo_repair.get("summary"),
            }
        )
    attd = state.get("attd") or {}
    attd_packets = state.get("attd_maintenance_packets") or {}
    if attd:
        items.append(
            {
                "priority": "critical" if attd.get("trigger_state") == "RED" else ("high" if attd.get("trigger_state") == "YELLOW" else "medium"),
                "kind": "attd_repo_health",
                "title": f"ATTD {attd.get('trigger_state')} score={attd.get('attd_score')}",
                "suggested_action": (
                    "Run maintenance packets before further autonomous growth."
                    if attd.get("trigger_state") == "RED"
                    else "Keep ATTD packets in the maintenance queue while frontier work continues."
                ),
                "evidence": {
                    "components": attd.get("components"),
                    "packets": attd_packets.get("packet_count"),
                    "governance": attd.get("governance"),
                },
            }
        )
    coherence_gate = state.get("coherence_delirium_gate") or {}
    if coherence_gate:
        items.append(
            {
                "priority": (
                    "critical"
                    if not coherence_gate.get("allows_long_autonomy")
                    else ("high" if not coherence_gate.get("allows_candidate_promotion") else "medium")
                ),
                "kind": "coherence_delirium_gate",
                "title": f"Coherence/delirium {coherence_gate.get('trigger_state')} delirium={coherence_gate.get('delirium_score')}",
                "suggested_action": (
                    "Pause long autonomy and repair coherence blockers before risky capability work."
                    if not coherence_gate.get("allows_long_autonomy")
                    else "Keep promotion and capability expansion blocked until the candidate coherence threshold clears."
                    if not coherence_gate.get("allows_candidate_promotion")
                    else "Carry coherence gate evidence into launch and self-evolution packets."
                ),
                "evidence": {
                    "coherence_score": coherence_gate.get("coherence_score"),
                    "delirium_score": coherence_gate.get("delirium_score"),
                    "blockers": coherence_gate.get("blockers", []),
                    "candidate_blockers": coherence_gate.get("candidate_blockers", []),
                },
            }
        )
    frontier = decision.get("frontier") or {}
    pressure = decision.get("frontier_pressure") or {}
    if pressure.get("frontier_exhausted"):
        items.append(
            {
                "priority": "high",
                "kind": "frontier_rotation",
                "title": "Tracked frontier exhausted",
                "reason": decision.get("reason"),
                "suggested_action": (
                    f"rotate to mutated BabyLM seed{pressure.get('next_mutated_babylm_seed')}, "
                    "import a licensed benchmark candidate, or activate an RL smoke frontier"
                ),
            }
        )
    promotion_closure = state.get("promotion_closure") or {}
    if promotion_closure:
        items.append(
            {
                "priority": "high" if promotion_closure.get("status") == "promotion_closed" else "low",
                "kind": "promotion_closure",
                "title": f"Promotion closure: {promotion_closure.get('status')}",
                "suggested_action": "checkpoint, back up, mark regression, and start the requested harder frontier",
                "evidence": {
                    "accepted": get_path(promotion_closure, ["accepted_candidate", "benchmark_name"], None),
                    "rotation": get_path(promotion_closure, ["rotation_request", "pressure_card_id"], None),
                },
            }
        )
    if frontier:
        items.append(
            {
                "priority": "high",
                "kind": "frontier",
                "title": f"Improve {frontier.get('benchmark_name')}",
                "reason": decision.get("reason"),
                "suggested_action": frontier.get("recommended_intervention"),
            }
        )
    curriculum = state.get("benchmaxx_curriculum") or {}
    current_stage = curriculum.get("current_stage") if isinstance(curriculum.get("current_stage"), dict) else {}
    next_frontier = curriculum.get("next_frontier") if isinstance(curriculum.get("next_frontier"), dict) else {}
    if current_stage or next_frontier:
        items.append(
            {
                "priority": "high",
                "kind": "benchmaxx_curriculum",
                "title": f"Current course stage: {current_stage.get('id') or 'unknown'}",
                "suggested_action": next_frontier.get("action") or current_stage.get("next_action"),
                "evidence": {
                    "current_status": current_stage.get("status"),
                    "next_frontier_family": next_frontier.get("family"),
                    "next_stage": next_frontier.get("stage_id"),
                    "teacher_policy": current_stage.get("teacher_policy"),
                },
            }
        )
    adapter_factory = state.get("benchmark_adapter_factory") or {}
    if adapter_factory:
        items.append(
            {
                "priority": "high"
                if get_path(adapter_factory, ["summary", "needs_smoke"], 0)
                else "medium",
                "kind": "benchmark_adapter_factory",
                "title": (
                    f"Adapter cards={get_path(adapter_factory, ['summary', 'cards'], 0)} "
                    f"needs_smoke={get_path(adapter_factory, ['summary', 'needs_smoke'], 0)}"
                ),
                "suggested_action": "; ".join(adapter_factory.get("next_actions", [])[:3]),
            }
        )
    pantry_unblocker = state.get("benchmark_pantry_unblocker") or {}
    if pantry_unblocker:
        items.append(
            {
                "priority": "high"
                if get_path(pantry_unblocker, ["summary", "autonomous_safe_actions"], 0)
                else "medium",
                "kind": "benchmark_pantry_unblocker",
                "title": (
                    f"Blocked adapters={get_path(pantry_unblocker, ['summary', 'blocked_cards'], 0)} "
                    f"safe_actions={get_path(pantry_unblocker, ['summary', 'autonomous_safe_actions'], 0)}"
                ),
                "suggested_action": "; ".join(pantry_unblocker.get("next_actions", [])[:3]),
                "evidence": pantry_unblocker.get("summary"),
            }
        )
    pantry = state.get("resource_pantry") or {}
    if pantry:
        pantry_summary = pantry.get("summary") or {}
        items.append(
            {
                "priority": "high"
                if pantry_summary.get("failed_actions", 0)
                else ("medium" if pantry_summary.get("clone_allowed", 0) != pantry_summary.get("present_clones", 0) else "low"),
                "kind": "resource_pantry",
                "title": (
                    f"Resource pantry clones={pantry_summary.get('present_clones', 0)}/"
                    f"{pantry_summary.get('clone_allowed', 0)} adapter_ready={pantry_summary.get('adapter_ready', 0)}"
                ),
                "suggested_action": "; ".join((pantry.get("next_actions") or [])[:3]),
                "evidence": {
                    "storage_selected": pantry_summary.get("storage_selected"),
                    "clone_root": pantry_summary.get("clone_root"),
                    "blocked": pantry_summary.get("blocked"),
                    "failed_actions": pantry_summary.get("failed_actions"),
                },
            }
        )
    legacy_concepts = state.get("legacy_concepts") or {}
    if legacy_concepts:
        legacy_summary = legacy_concepts.get("summary") or {}
        queue = legacy_concepts.get("top_port_queue") or []
        top = queue[0] if queue and isinstance(queue[0], dict) else {}
        p0_open = int(legacy_summary.get("p0_open", 0) or 0)
        items.append(
            {
                "priority": "high" if p0_open else "medium",
                "kind": "legacy_concept_port_queue",
                "title": (
                    f"Legacy concepts {legacy_summary.get('trigger_state')} "
                    f"p0_open={p0_open} top={top.get('id')}"
                ),
                "suggested_action": top.get("port_goal")
                or "keep predecessor-project concept queue refreshed and only port concepts with concrete Theseus gates",
                "evidence": {
                    "projects_present": legacy_summary.get("projects_present"),
                    "missing_evidence_count": legacy_summary.get("missing_evidence_count"),
                    "candidate": top,
                },
            }
        )
    legacy_ports = state.get("legacy_port_mechanisms") or {}
    if legacy_ports:
        legacy_port_summary = legacy_ports.get("summary") or {}
        items.append(
            {
                "priority": "high" if legacy_port_summary.get("red") else ("medium" if legacy_port_summary.get("yellow_or_degraded") else "low"),
                "kind": "legacy_port_mechanisms",
                "title": (
                    f"Legacy ports materialized={legacy_port_summary.get('mechanisms')} "
                    f"red={legacy_port_summary.get('red')} yellow={legacy_port_summary.get('yellow_or_degraded')}"
                ),
                "suggested_action": (
                    f"Fix {legacy_port_summary.get('top_blocker')} before long autonomy."
                    if legacy_port_summary.get("top_blocker")
                    else "Use mechanism reports as teacher-ready patch targets."
                ),
                "evidence": legacy_port_summary,
            }
        )
    legacy_runtime_gate = state.get("legacy_runtime_governance_gate") or {}
    if legacy_runtime_gate:
        summary = legacy_runtime_gate.get("summary") or {}
        items.append(
            {
                "priority": "high" if legacy_runtime_gate.get("trigger_state") == "RED" else "medium",
                "kind": "legacy_runtime_governance_gate",
                "title": (
                    f"Runtime governance {legacy_runtime_gate.get('trigger_state')} "
                    f"warnings={summary.get('warning_count')} failed={len(summary.get('failed_gates') or [])}"
                ),
                "suggested_action": "clear proxy-truth/coherence warnings before candidate promotion; keep TaskSpell and PlanForge locks active for teacher work",
                "evidence": summary,
            }
        )
    runtime_enforcement = state.get("legacy_port_runtime_enforcement") or {}
    if runtime_enforcement:
        summary = runtime_enforcement.get("summary") or {}
        items.append(
            {
                "priority": (
                    "critical"
                    if summary.get("trigger_state") == "RED"
                    else ("high" if not runtime_enforcement.get("ready_for_long_autonomy") else "medium")
                ),
                "kind": "legacy_port_runtime_enforcement",
                "title": (
                    f"Legacy runtime enforcement {summary.get('trigger_state')} "
                    f"blockers={summary.get('blocker_count')}"
                ),
                "suggested_action": (
                    "Clear hotpath/ATTD blockers before long autonomy; keep TaskSpell effect/replay and PlanForge contracts attached to every action."
                    if not runtime_enforcement.get("ready_for_long_autonomy")
                    else "Use runtime enforcement ledgers as the blessed execution evidence layer."
                ),
                "evidence": summary,
            }
        )
    legacy_training = state.get("legacy_training_source_audit") or {}
    if legacy_training:
        summary = legacy_training.get("summary") or {}
        items.append(
            {
                "priority": "high" if legacy_training.get("trigger_state") == "RED" else "medium",
                "kind": "legacy_training_source_admission",
                "title": (
                    f"Legacy training sources {legacy_training.get('trigger_state')} "
                    f"serious={summary.get('serious_training_ready')} ready={summary.get('ready_local_verified')}"
                ),
                "suggested_action": "use only the admission-plan primary candidates for tiny dry-runs; keep benchmark answers quarantined",
                "evidence": summary,
            }
        )
    legacy_training_sample = state.get("legacy_training_source_sample") or {}
    if legacy_training_sample:
        summary = legacy_training_sample.get("summary") or {}
        items.append(
            {
                "priority": "high" if legacy_training_sample.get("trigger_state") == "RED" else "medium",
                "kind": "legacy_training_tiny_sample",
                "title": (
                    f"Legacy tiny sample {legacy_training_sample.get('trigger_state')} "
                    f"rows={summary.get('sample_rows')} lanes={len(summary.get('lane_counts') or {})}"
                ),
                "suggested_action": "use this bounded sample for a local dry-run before any full legacy source training",
                "evidence": summary,
            }
        )
    legacy_rl_envs = state.get("legacy_rl_environment_admission") or {}
    if legacy_rl_envs:
        summary = legacy_rl_envs.get("summary") or {}
        items.append(
            {
                "priority": "high" if legacy_rl_envs.get("trigger_state") == "RED" else "medium",
                "kind": "legacy_rl_environment_admission",
                "title": (
                    f"Legacy RL envs {legacy_rl_envs.get('trigger_state')} "
                    f"p0={summary.get('p0_smoke_lane')} envs={summary.get('environments')}"
                ),
                "suggested_action": "start with P0 manifest-ready environments, then install dependencies and emit seeded smoke replays",
                "evidence": summary,
            }
        )
    legacy_rl_smoke = state.get("legacy_rl_smoke_plan") or {}
    if legacy_rl_smoke:
        summary = legacy_rl_smoke.get("summary") or {}
        items.append(
            {
                "priority": "high" if legacy_rl_smoke.get("trigger_state") == "RED" else "medium",
                "kind": "legacy_rl_smoke_plan",
                "title": (
                    f"Legacy RL smoke plan {legacy_rl_smoke.get('trigger_state')} "
                    f"ready={summary.get('ready_for_seeded_smoke')} pending={summary.get('pending_dependency')} "
                    f"source_present_pending_install={summary.get('source_present_pending_install')} "
                    f"runner_pending_adapter={summary.get('runner_pending_adapter')}"
                ),
                "suggested_action": "install or map missing P0 environment dependencies, then run seeded smoke receipts",
                "evidence": summary,
            }
        )
    trace_capsules = state.get("trace_fabric_capsule_admission") or {}
    if trace_capsules:
        summary = trace_capsules.get("summary") or {}
        items.append(
            {
                "priority": "high" if trace_capsules.get("trigger_state") == "RED" else "medium",
                "kind": "trace_fabric_capsule_admission",
                "title": (
                    f"Trace capsules {trace_capsules.get('trigger_state')} "
                    f"accepted={summary.get('accepted_metadata_only')} quarantined={summary.get('quarantined')}"
                ),
                "suggested_action": "export only accepted metadata-only capsules through a row materializer with holdout/redaction checks",
                "evidence": summary,
            }
        )
    trace_materialized = state.get("trace_fabric_capsule_materialization") or {}
    if trace_materialized:
        summary = trace_materialized.get("summary") or {}
        items.append(
            {
                "priority": "high" if trace_materialized.get("trigger_state") == "RED" else "medium",
                "kind": "trace_fabric_capsule_materialization",
                "title": (
                    f"Trace rows {trace_materialized.get('trigger_state')} "
                    f"rows={summary.get('materialized_rows')} raw={summary.get('raw_payload_rows')}"
                ),
                "suggested_action": "feed only the materialized governance rows into tiny dry-runs; refresh admission before full trace-derived training",
                "evidence": summary,
            }
        )
    adapter_plan = state.get("legacy_adapter_bank_training_plan") or {}
    if adapter_plan:
        summary = adapter_plan.get("summary") or {}
        items.append(
            {
                "priority": "high" if adapter_plan.get("trigger_state") == "RED" else "medium",
                "kind": "legacy_adapter_bank_training_plan",
                "title": (
                    f"Adapter bank plan {adapter_plan.get('trigger_state')} "
                    f"lanes={summary.get('source_lane_count')} selected={len(summary.get('selected_adapters') or [])}"
                ),
                "suggested_action": "run zero-parameter lane dry-runs first; keep adapter weights eval-only until ablations and regression floors pass",
                "evidence": summary,
            }
        )
    active_inference = state.get("legacy_active_inference_pilot") or {}
    if active_inference:
        summary = active_inference.get("summary") or {}
        items.append(
            {
                "priority": "high" if active_inference.get("trigger_state") == "RED" else "medium",
                "kind": "legacy_active_inference_pilot",
                "title": (
                    f"Active inference pilot {active_inference.get('trigger_state')} "
                    f"error={summary.get('mean_prediction_error')} updates={summary.get('accepted_belief_updates')}"
                ),
                "suggested_action": "use prediction-error and expected-free-energy traces as governed world-model training signals before scaling to real adapters",
                "evidence": summary,
            }
        )
    architecture_experiments = state.get("architecture_experiment_governance") or {}
    if architecture_experiments:
        recommended = architecture_experiments.get("recommended_next_experiment") or {}
        items.append(
            {
                "priority": "high" if architecture_experiments.get("architecture_change_allowed") else "medium",
                "kind": "architecture_experiment",
                "title": f"Next experiment: {recommended.get('id')}",
                "suggested_action": recommended.get("hypothesis")
                or "refresh architecture experiment governance",
                "architecture_change_allowed": architecture_experiments.get("architecture_change_allowed"),
            }
        )
    architecture_runner = state.get("architecture_experiment_runner") or {}
    if architecture_runner:
        items.append(
            {
                "priority": "high" if architecture_runner.get("status") == "failed" else "medium",
                "kind": "architecture_experiment_runner",
                "title": f"Architecture runner {architecture_runner.get('status')}",
                "suggested_action": "keep matched experiment ledger current and retain winners only after gates hold",
                "evidence": architecture_runner.get("score_delta"),
            }
        )
    autoresearch_audit = state.get("autoresearch_gap_audit") or {}
    if autoresearch_audit:
        items.append(
            {
                "priority": "high"
                if get_path(autoresearch_audit, ["summary", "trigger_state"], "GREEN") == "RED"
                else "medium",
                "kind": "autoresearch_gap_audit",
                "title": (
                    f"Autoresearch audit {get_path(autoresearch_audit, ['summary', 'trigger_state'], '--')} "
                    f"gaps={get_path(autoresearch_audit, ['summary', 'gap_count'], 0)}"
                ),
                "suggested_action": "; ".join(autoresearch_audit.get("recommendations", [])[:3]),
            }
        )
    loop_closure = state.get("loop_closure_harvester") or {}
    if loop_closure:
        items.append(
            {
                "priority": "high"
                if get_path(loop_closure, ["summary", "ready_for_tool_synthesis"], 0)
                else "medium",
                "kind": "loop_closure_harvester",
                "title": (
                    f"Tool candidates={get_path(loop_closure, ['summary', 'candidates'], 0)} "
                    f"ready={get_path(loop_closure, ['summary', 'ready_for_tool_synthesis'], 0)}"
                ),
                "suggested_action": "; ".join(loop_closure.get("next_actions", [])[:3]),
            }
        )
    promoter = state.get("loop_closure_tool_promoter") or {}
    if promoter:
        items.append(
            {
                "priority": "high" if promoter.get("promoted") else "medium",
                "kind": "loop_closure_tool_promoter",
                "title": f"Promoted tools={len(promoter.get('promoted', []))}",
                "suggested_action": "use promoted adapter, pressure, checkpoint, and transfer tools before teacher escalation",
            }
        )
    transfer = state.get("transfer_eval_suite") or {}
    if transfer:
        items.append(
            {
                "priority": "high",
                "kind": "asi_transfer_eval",
                "title": f"ASI transfer suite score={get_path(transfer, ['summary', 'accuracy'], None)}",
                "suggested_action": "keep code repair, tool use, web, context, RL, self-debugging, and voice surfaces in the frontier mix",
                "residuals": transfer.get("residuals", [])[:7],
            }
        )
    arm_transfer = state.get("arm_transfer_plan") or {}
    arm_suckers = state.get("arm_sucker_registry") or {}
    if arm_suckers:
        summary = arm_suckers.get("summary") if isinstance(arm_suckers.get("summary"), dict) else {}
        items.append(
            {
                "priority": "high" if summary.get("blocked_suckers", 0) else "medium",
                "kind": "arm_sucker_transfer_hierarchy",
                "title": (
                    f"Arm-sucker hierarchy ready={summary.get('ready_suckers', 0)} "
                    f"blocked={summary.get('blocked_suckers', 0)}"
                ),
                "suggested_action": "load high-transfer parent arms before low-transfer suckers; promote suckers only after repeated sibling transfer",
                "evidence": arm_suckers.get("routing_contracts", [])[:5],
            }
        )
    if arm_transfer:
        summary = arm_transfer.get("summary") if isinstance(arm_transfer.get("summary"), dict) else {}
        items.append(
            {
                "priority": "high" if summary.get("blocked_edges", 0) else "medium",
                "kind": "arm_transfer_plan",
                "title": (
                    f"Arm transfer {summary.get('frontier_family') or 'general'} "
                    f"ready={summary.get('ready_edges', 0)} blocked={summary.get('blocked_edges', 0)}"
                ),
                "suggested_action": "; ".join(str(item) for item in (arm_transfer.get("next_actions") or [])[:3]),
                "evidence": arm_transfer.get("transfer_plan", [])[:4],
            }
        )
    transfer_artifacts = state.get("arm_transfer_artifacts") or {}
    if transfer_artifacts:
        items.append(
            {
                "priority": "medium",
                "kind": "arm_transfer_artifacts",
                "title": f"Transfer artifacts={get_path(transfer_artifacts, ['summary', 'artifacts'], 0)}",
                "suggested_action": "load transfer artifacts in matching arms before adding parameters",
                "evidence": transfer_artifacts.get("artifacts", [])[:5],
            }
        )
    teacher_proof = state.get("teacher_self_edit_proof") or {}
    if teacher_proof:
        items.append(
            {
                "priority": "high" if teacher_proof.get("status") in {"blocked", "needs_real_success_cycles"} else "medium",
                "kind": "teacher_self_edit_proof",
                "title": f"Teacher self-edit proof {teacher_proof.get('status')}",
                "suggested_action": "; ".join(teacher_proof.get("next_actions", [])[:3]),
                "evidence": teacher_proof.get("summary"),
            }
        )
    growth_gate = state.get("model_growth_gate") or {}
    if growth_gate:
        items.append(
            {
                "priority": "high" if not growth_gate.get("model_growth_allowed") else "medium",
                "kind": "model_growth_gate",
                "title": f"Model growth allowed={growth_gate.get('model_growth_allowed')}",
                "suggested_action": growth_gate.get("next_action"),
                "blockers": growth_gate.get("hard_blockers", []) + growth_gate.get("missing_evidence", []),
            }
        )
    self_evolution = state.get("self_evolution_governance") or {}
    if self_evolution:
        items.append(
            {
                "priority": "high"
                if get_path(self_evolution, ["teacher_apply", "allowed_now"], False)
                else "medium",
                "kind": "self_evolution_governance",
                "title": f"Teacher self-edit allowed={get_path(self_evolution, ['teacher_apply', 'allowed_now'], False)}",
                "suggested_action": "; ".join(self_evolution.get("next_actions", [])[:3]),
                "blockers": get_path(self_evolution, ["teacher_apply", "blockers"], []),
            }
        )
    for rec in get_path(state, ["benchmark_seeker", "recommendations"], []):
        items.append(
            {
                "priority": rec.get("priority", "medium"),
                "kind": rec.get("kind", "benchmark_seeker"),
                "title": rec.get("benchmark") or rec.get("kind"),
                "suggested_action": rec.get("action"),
            }
        )
    for source in get_path(state, ["online_source_catalog", "benchmark_candidates"], []):
        items.append(
            {
                "priority": source.get("priority", "medium"),
                "kind": "online_benchmark_candidate",
                "title": source.get("name") or source.get("id"),
                "suggested_action": (
                    f"{source.get('decision')}: stage source/archive only, then add adapter smoke before benchmark use"
                ),
            }
        )
    for source in get_path(state, ["online_source_catalog", "training_data_candidates"], []):
        items.append(
            {
                "priority": source.get("priority", "medium"),
                "kind": "training_data_candidate",
                "title": source.get("name") or source.get("id"),
                "suggested_action": (
                    f"{source.get('decision')}: keep metadata-only until sampling, dedupe, leakage, and quality gates pass"
                ),
            }
        )
    sampler = state.get("training_data_sampler") or {}
    if sampler:
        items.append(
            {
                "priority": "high" if sampler.get("training_use_allowed") else "medium",
                "kind": "training_data_sampler",
                "title": f"External samples ready={sampler.get('training_use_allowed')} rows={get_path(sampler, ['summary', 'sample_rows'], 0)} pairs={get_path(sampler, ['summary', 'pairwise_rows'], 0)}",
                "suggested_action": "use governed tiny samples only through low-ratio pairwise distillation; require teacher/human review before bulk use",
            }
        )
    for source in get_path(state, ["knowledge_sources", "sources"], []):
        if isinstance(source, dict) and not source.get("training_use_allowed", False):
            items.append(
                {
                    "priority": "medium",
                    "kind": "knowledge_source",
                    "title": source.get("name"),
                    "suggested_action": "targeted lookup only until license, robots, provenance, and approval gates pass",
                }
            )
    data_summary = get_path(state, ["training_data_inventory", "summary"], {})
    if data_summary:
        items.append(
            {
                "priority": "medium",
                "kind": "data_inventory",
                "title": f"Track {data_summary.get('files', 0)} training/eval assets",
                "suggested_action": "keep data inventory fresh and audit new external sources before training use",
            }
        )
    rom_summary = get_path(state, ["local_rom_registry", "summary"], {})
    if rom_summary:
        items.append(
            {
                "priority": "high" if rom_summary.get("ready_for_wrapper_smoke") else "medium",
                "kind": "local_rom_growth_lane",
                "title": f"Local ROMs={rom_summary.get('rom_count', 0)} profiles_ready={rom_summary.get('matched_priority_profiles', 0)}",
                "suggested_action": "use only user-supplied ignored ROMs; build deterministic wrapper smoke before adding emulator RL frontiers",
            }
        )
    staging_summary = get_path(state, ["local_rom_staging", "summary"], {})
    if staging_summary:
        items.append(
            {
                "priority": "high" if staging_summary.get("active_rom_records") else "medium",
                "kind": "local_rom_asset_staging",
                "title": f"Staged unique active ROMs={staging_summary.get('unique_active_roms', staging_summary.get('active_rom_records', 0))} inactive assets={staging_summary.get('inactive_asset_count', 0)}",
                "suggested_action": "prioritize Pokemon Emerald/PyGBA wrapper smoke; keep NDS/N64/disc assets inactive until adapters exist",
            }
        )
    synthetic = state.get("synthetic_data") or {}
    if synthetic:
        items.append(
            {
                "priority": "high" if synthetic.get("training_ready") else "medium",
                "kind": "synthetic_data",
                "title": f"Synthetic curator ready={synthetic.get('training_ready')} accepted={synthetic.get('accepted_count', 0)}",
                "suggested_action": "use residual-targeted synthetic blend only while ratio, leakage, provenance, and public/private delta gates stay green",
            }
        )
    for rec in get_path(state, ["rl_benchmark_registry", "recommended_frontier"], []):
        items.append(
            {
                "priority": rec.get("priority", "medium"),
                "kind": "rl_frontier",
                "title": rec.get("name"),
                "suggested_action": rec.get("next_step"),
            }
        )
    response = (teacher_result or {}).get("response_json")
    if isinstance(response, dict):
        items.append(
            {
                "priority": "high",
                "kind": "teacher_recommendation",
                "title": response.get("diagnosis", "Teacher recommendation"),
                "suggested_action": response.get("recommended_intervention"),
                "verification_steps": response.get("verification_steps"),
            }
        )
    resource = state.get("resource_governor") or {}
    if resource:
        items.append(
            {
                "priority": "medium" if get_path(resource, ["decision", "can_run_requested_profile"], True) else "high",
                "kind": "resource_governor",
                "title": f"Efficiency score {get_path(resource, ['efficiency', 'score'], 0.0)}",
                "suggested_action": "respect resource envelope before training; prefer Rust/CUDA hot loops and smaller profiles when throttled",
            }
        )
    hive = state.get("hive_scheduler") or {}
    if hive:
        summary = hive.get("summary") or {}
        items.append(
            {
                "priority": "medium" if summary.get("remote_placements") else "low",
                "kind": "hive_scheduler",
                "title": f"Hive nodes={summary.get('nodes', 0)} remote placements={summary.get('remote_placements', 0)}",
                "suggested_action": "run stronger peers for CUDA/MLX work and route weak clients through authorized checkpoint/chat gateways",
            }
        )
    performance = state.get("performance_optimizer") or {}
    if performance:
        perf_summary = performance.get("summary") if isinstance(performance.get("summary"), dict) else {}
        items.append(
            {
                "priority": "high"
                if performance.get("trigger_state") == "RED"
                else ("medium" if performance.get("trigger_state") == "YELLOW" else "low"),
                "kind": "performance_optimizer",
                "title": (
                    f"Performance {performance.get('trigger_state', 'UNKNOWN')} "
                    f"backend={perf_summary.get('preferred_training_backend')} "
                    f"chunks={perf_summary.get('recent_ok_worker_chunks', 0)}/"
                    f"{perf_summary.get('recent_worker_chunks', 0)}"
                ),
                "suggested_action": "; ".join(
                    str(row.get("action"))
                    for row in (performance.get("recommendations") or [])[:3]
                    if isinstance(row, dict)
                ),
                "evidence": {
                    "score": performance.get("score"),
                    "preferred_training_backend": perf_summary.get("preferred_training_backend"),
                    "preferred_inference_backend": perf_summary.get("preferred_inference_backend"),
                    "bottlenecks": [
                        row.get("id")
                        for row in (performance.get("bottlenecks") or [])[:5]
                        if isinstance(row, dict)
                    ],
                },
            }
        )
    goal = state.get("autonomous_goal") or {}
    if goal:
        items.append(
            {
                "priority": "medium",
                "kind": "autonomous_goal_trace",
                "title": "Latest router-mediated goal",
                "suggested_action": f"arms={','.join(goal.get('selected_arms') or [])}",
            }
        )
    arm_governance = state.get("arm_lifecycle_governance") or {}
    for proposal in arm_governance.get("proposals", [])[:12]:
        items.append(
            {
                "priority": proposal.get("priority", "medium"),
                "kind": f"arm_lifecycle:{proposal.get('kind', 'proposal')}",
                "title": proposal.get("arm_name") or ",".join(proposal.get("arm_names") or []) or "Arm lifecycle proposal",
                "suggested_action": proposal.get("action"),
                "requires_teacher": proposal.get("requires_teacher", False),
            }
        )
    context_packets = state.get("context_packets") or {}
    if context_packets:
        items.append(
            {
                "priority": "medium",
                "kind": "context_memory",
                "title": f"Context packets active={get_path(context_packets, ['summary', 'active_packet_count'], 0)} drop={get_path(context_packets, ['summary', 'drop_candidate_count'], 0)}",
                "suggested_action": "preserve high-importance conclusions and summaries; keep raw low-importance logs out of active context",
            }
        )
    capability_matrix = state.get("capability_matrix") or {}
    for gap in get_path(capability_matrix, ["summary", "top_gaps"], [])[:8]:
        items.append(
            {
                "priority": "medium",
                "kind": "capability_gap",
                "title": gap.get("capability_id"),
                "suggested_action": gap.get("gap"),
            }
        )
    return {
        "policy": "sparkstream_self_improvement_queue_v0",
        "updated_utc": now(),
        "items": items[:50],
    }

def active_frontier(ledger: Any, *, preferred_family: str = "", preferred_card_id: str = "") -> dict[str, Any]:
    if not isinstance(ledger, list):
        return {}
    frontiers = [
        row
        for row in ledger
        if isinstance(row, dict) and row.get("lifecycle") == "frontier"
    ]
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

def row_frontier_family(row: dict[str, Any]) -> str:
    name = str(row.get("benchmark_name") or "")
    benchmark_type = str(row.get("benchmark_type") or "")
    best_report = str(row.get("best_report") or "")
    if name == "babylm_mutated_holdout" or "mutated" in benchmark_type:
        return "babylm_mutated"
    if name.startswith("ocean-"):
        return "rl_local"
    if name.startswith("minecraft_rl_") or name.startswith("minecraft_"):
        return "minecraft_rl"
    if name.startswith("drone_rl_"):
        return "drone_rl"
    if name.startswith("drone_control_"):
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
    if any(
        source in best_report
        for source in [
            "source_bigcodebench",
            "source_evalplus",
            "source_human_eval",
            "source_mbpp",
            "source_livecodebench",
            "source_opencode",
            "source_swe_bench",
            "source_swe_agent",
            "source_mini_swe_agent",
            "source_codeclash",
            "source_swe_polybench",
            "source_swe_gen",
        ]
    ):
        return "coding_local_sandbox"
    if "source_webarena" in best_report:
        return "web_agent_local"
    return ""

def preferred_curriculum_family(state: dict[str, Any]) -> str:
    runner = str(get_path(state, ["benchmaxx_curriculum", "next_frontier", "runner_family"], "") or "")
    family = str(get_path(state, ["benchmaxx_curriculum", "next_frontier", "family"], "") or "")
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

def failed_gates(candidate: dict[str, Any]) -> list[str]:
    return [
        item.get("gate")
        for item in candidate.get("checks", [])
        if isinstance(item, dict) and not item.get("passed")
    ]

def compact_ledger_entry(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "cycle_id": report.get("cycle_id"),
        "created_utc": report.get("created_utc"),
        "profile": report.get("profile"),
        "execute": report.get("execute"),
        "ok": report.get("ok"),
        "decision": report.get("decision"),
        "teacher_needed": report.get("teacher_needed"),
        "teacher_used": report.get("teacher_used"),
        "command_count": len(report.get("commands", [])),
        "checkpoint": get_path(read_json(ROOT / "reports" / "checkpoint_last.json"), ["checkpoint_id"], None),
    }

def compact_observation(state: dict[str, Any]) -> dict[str, Any]:
    candidate = state.get("candidate_gate") or {}
    preflight = state.get("preflight") or {}
    seeker = state.get("benchmark_seeker") or {}
    checkpoint = state.get("checkpoint_registry") or {}
    checkpoint_backup = state.get("checkpoint_backup") or {}
    return {
        "preflight": {
            "heavy_training_allowed": preflight.get("heavy_training_allowed"),
            "passed": preflight.get("passed"),
            "total": preflight.get("total"),
            "blocker_count": preflight.get("blocker_count"),
            "warning_count": preflight.get("warning_count"),
        },
        "candidate_gate": {
            "promote": candidate.get("promote"),
            "passed": candidate.get("passed"),
            "total": candidate.get("total"),
            "failed_gates": failed_gates(candidate),
            "scores": candidate.get("scores"),
            "residual_delta": candidate.get("residual_delta"),
        },
        "promotion_closure": {
            "status": get_path(state, ["promotion_closure", "status"], None),
            "accepted_benchmark": get_path(state, ["promotion_closure", "accepted_candidate", "benchmark_name"], None),
            "rotation_family": get_path(state, ["promotion_closure", "rotation_request", "frontier_family"], None),
            "rotation_card": get_path(state, ["promotion_closure", "rotation_request", "pressure_card_id"], None),
            "accepted_count": len(get_path(state, ["accepted_candidate_registry", "accepted_candidates"], [])),
        },
        "frontier": active_frontier(
            state.get("benchmark_ledger"),
            preferred_family=preferred_curriculum_family(state),
            preferred_card_id=str(get_path(state, ["benchmaxx_curriculum", "next_frontier", "recommended_env"], "") or ""),
        ),
        "residual_summary": get_path(state, ["residual_escrow", "summary"], {}),
        "benchmark_recommendations": (seeker.get("recommendations") or [])[:10],
        "knowledge_sources": [
            {
                "name": source.get("name"),
                "status": source.get("status"),
                "training_use_allowed": source.get("training_use_allowed"),
            }
            for source in get_path(state, ["knowledge_sources", "sources"], [])[:10]
            if isinstance(source, dict)
        ],
        "online_source_catalog": {
            "sources": get_path(state, ["online_source_catalog", "summary", "sources"], None),
            "approved_for_catalog_import": get_path(
                state, ["online_source_catalog", "summary", "approved_for_catalog_import"], None
            ),
            "imported_or_present": get_path(
                state, ["online_source_catalog", "summary", "imported_or_present"], None
            ),
            "training_data_candidates": get_path(
                state, ["online_source_catalog", "summary", "training_data_candidates"], None
            ),
            "benchmark_candidates": get_path(
                state, ["online_source_catalog", "summary", "benchmark_candidates"], None
            ),
        },
        "resource_pantry": {
            "storage_selected": get_path(state, ["resource_pantry", "summary", "storage_selected"], None),
            "clone_root": get_path(state, ["resource_pantry", "summary", "clone_root"], None),
            "clone_allowed": get_path(state, ["resource_pantry", "summary", "clone_allowed"], None),
            "present_clones": get_path(state, ["resource_pantry", "summary", "present_clones"], None),
            "adapter_ready": get_path(state, ["resource_pantry", "summary", "adapter_ready"], None),
            "metadata_only": get_path(state, ["resource_pantry", "summary", "metadata_only"], None),
            "blocked": get_path(state, ["resource_pantry", "summary", "blocked"], None),
            "failed_actions": get_path(state, ["resource_pantry", "summary", "failed_actions"], None),
        },
        "legacy_concepts": {
            "trigger_state": get_path(state, ["legacy_concepts", "summary", "trigger_state"], None),
            "projects_present": get_path(state, ["legacy_concepts", "summary", "projects_present"], None),
            "port_candidates": get_path(state, ["legacy_concepts", "summary", "port_candidates"], None),
            "p0_open": get_path(state, ["legacy_concepts", "summary", "p0_open"], None),
            "top_candidate": get_path(state, ["legacy_concepts", "summary", "top_candidate"], None),
            "missing_evidence_count": get_path(state, ["legacy_concepts", "summary", "missing_evidence_count"], None),
        },
        "legacy_port_mechanisms": {
            "mechanisms": get_path(state, ["legacy_port_mechanisms", "summary", "mechanisms"], None),
            "red": get_path(state, ["legacy_port_mechanisms", "summary", "red"], None),
            "yellow_or_degraded": get_path(state, ["legacy_port_mechanisms", "summary", "yellow_or_degraded"], None),
            "top_blocker": get_path(state, ["legacy_port_mechanisms", "summary", "top_blocker"], None),
            "states": get_path(state, ["legacy_port_mechanisms", "summary", "states"], {}),
        },
        "legacy_runtime_governance": {
            "trigger_state": get_path(state, ["legacy_runtime_governance_gate", "trigger_state"], None),
            "ready_for_teacher_work": get_path(state, ["legacy_runtime_governance_gate", "ready_for_teacher_work"], None),
            "ready_for_candidate_promotion": get_path(state, ["legacy_runtime_governance_gate", "ready_for_candidate_promotion"], None),
            "warning_count": get_path(state, ["legacy_runtime_governance_gate", "summary", "warning_count"], None),
            "failed_gates": get_path(state, ["legacy_runtime_governance_gate", "summary", "failed_gates"], []),
        },
        "legacy_admissions": {
            "training_sources": {
                "trigger_state": get_path(state, ["legacy_training_source_audit", "trigger_state"], None),
                "ready_local_verified": get_path(state, ["legacy_training_source_audit", "summary", "ready_local_verified"], None),
                "serious_training_ready": get_path(state, ["legacy_training_source_audit", "summary", "serious_training_ready"], None),
                "hash_mismatches": get_path(state, ["legacy_training_source_audit", "summary", "hash_mismatches"], None),
            },
            "training_sample": {
                "trigger_state": get_path(state, ["legacy_training_source_sample", "trigger_state"], None),
                "sample_rows": get_path(state, ["legacy_training_source_sample", "summary", "sample_rows"], None),
                "lane_counts": get_path(state, ["legacy_training_source_sample", "summary", "lane_counts"], {}),
            },
            "rl_environments": {
                "trigger_state": get_path(state, ["legacy_rl_environment_admission", "trigger_state"], None),
                "environments": get_path(state, ["legacy_rl_environment_admission", "summary", "environments"], None),
                "p0_smoke_lane": get_path(state, ["legacy_rl_environment_admission", "summary", "p0_smoke_lane"], None),
                "hardware_gated_envs": get_path(state, ["legacy_rl_environment_admission", "summary", "hardware_gated_envs"], None),
            },
            "rl_smoke_plan": {
                "trigger_state": get_path(state, ["legacy_rl_smoke_plan", "trigger_state"], None),
                "planned_envs": get_path(state, ["legacy_rl_smoke_plan", "summary", "planned_envs"], None),
                "ready_for_seeded_smoke": get_path(state, ["legacy_rl_smoke_plan", "summary", "ready_for_seeded_smoke"], None),
                "pending_dependency": get_path(state, ["legacy_rl_smoke_plan", "summary", "pending_dependency"], None),
                "source_present_pending_install": get_path(state, ["legacy_rl_smoke_plan", "summary", "source_present_pending_install"], None),
                "runner_pending_adapter": get_path(state, ["legacy_rl_smoke_plan", "summary", "runner_pending_adapter"], None),
            },
            "trace_capsules": {
                "trigger_state": get_path(state, ["trace_fabric_capsule_admission", "trigger_state"], None),
                "accepted_metadata_only": get_path(state, ["trace_fabric_capsule_admission", "summary", "accepted_metadata_only"], None),
                "quarantined": get_path(state, ["trace_fabric_capsule_admission", "summary", "quarantined"], None),
                "materialized_rows": get_path(state, ["trace_fabric_capsule_materialization", "summary", "materialized_rows"], None),
                "raw_payload_rows": get_path(state, ["trace_fabric_capsule_materialization", "summary", "raw_payload_rows"], None),
            },
            "adapter_bank_training_plan": {
                "trigger_state": get_path(state, ["legacy_adapter_bank_training_plan", "trigger_state"], None),
                "ready_for_zero_param_dry_run": get_path(state, ["legacy_adapter_bank_training_plan", "ready_for_zero_param_dry_run"], None),
                "ready_for_adapter_activation": get_path(state, ["legacy_adapter_bank_training_plan", "ready_for_adapter_activation"], None),
                "plan_rows": get_path(state, ["legacy_adapter_bank_training_plan", "summary", "plan_rows"], None),
                "selected_adapters": get_path(state, ["legacy_adapter_bank_training_plan", "summary", "selected_adapters"], []),
                "zero_param_lanes": get_path(state, ["legacy_adapter_bank_training_plan", "summary", "zero_param_lanes"], []),
            },
            "active_inference_pilot": {
                "trigger_state": get_path(state, ["legacy_active_inference_pilot", "trigger_state"], None),
                "ready_for_world_model_training_signal": get_path(state, ["legacy_active_inference_pilot", "ready_for_world_model_training_signal"], None),
                "mean_prediction_error": get_path(state, ["legacy_active_inference_pilot", "summary", "mean_prediction_error"], None),
                "action_rankings": get_path(state, ["legacy_active_inference_pilot", "summary", "action_rankings"], None),
                "accepted_belief_updates": get_path(state, ["legacy_active_inference_pilot", "summary", "accepted_belief_updates"], None),
                "replay_id": get_path(state, ["legacy_active_inference_pilot", "replay_id"], None),
            },
        },
        "checkpoint_count": len(checkpoint.get("checkpoints", [])),
        "checkpoint_backup": {
            "status": checkpoint_backup.get("status"),
            "ok": checkpoint_backup.get("ok"),
            "checkpoint_id": checkpoint_backup.get("checkpoint_id"),
            "candidate_promote": checkpoint_backup.get("candidate_promote"),
        },
        "updates": {
            "available": get_path(state, ["update_status", "update_available"], None),
            "soft": get_path(state, ["update_status", "soft_update_available"], None),
            "hard": get_path(state, ["update_status", "hard_update_available"], None),
            "restart_required": get_path(state, ["update_status", "restart_required"], None),
            "offer_update_id": get_path(state, ["update_offer", "update_id"], None),
            "offer_checkpoint_id": get_path(state, ["update_offer", "checkpoint_id"], None),
            "installed_update_id": get_path(state, ["update_status", "installed", "active_update_id"], None),
            "headline": get_path(state, ["update_status", "current_offer", "headline"], None),
        },
        "data_summary": get_path(state, ["training_data_inventory", "summary"], {}),
        "personality": {
            "core_status": get_path(state, ["personality_core", "status"], None),
            "context_status": get_path(state, ["personality_context", "status"], None),
            "selected_cards": get_path(state, ["personality_context", "summary", "selected_cards"], None),
            "hard_invariants": get_path(state, ["personality_context", "summary", "hard_safety_invariants"], None),
            "drift_passed": get_path(state, ["personality_drift_eval", "passed"], None),
            "drift_score": get_path(state, ["personality_drift_eval", "summary", "average_score"], None),
            "belief_governance_status": get_path(state, ["belief_update_governance", "status"], None),
        },
        "local_rom_summary": get_path(state, ["local_rom_registry", "summary"], {}),
        "local_rom_staging": {
            "active_rom_records": get_path(state, ["local_rom_staging", "summary", "active_rom_records"], None),
            "unique_active_roms": get_path(state, ["local_rom_staging", "summary", "unique_active_roms"], None),
            "staged_count": get_path(state, ["local_rom_staging", "summary", "staged_count"], None),
            "already_present_count": get_path(state, ["local_rom_staging", "summary", "already_present_count"], None),
            "inactive_asset_count": get_path(state, ["local_rom_staging", "summary", "inactive_asset_count"], None),
        },
        "synthetic_data": {
            "training_ready": get_path(state, ["synthetic_data", "training_ready"], None),
            "accepted_count": get_path(state, ["synthetic_data", "accepted_count"], None),
            "blend_synthetic_ratio": get_path(state, ["synthetic_data", "blend_synthetic_ratio"], None),
            "verification_ok": get_path(state, ["synthetic_data", "verification", "ok"], None),
        },
        "synthetic_benchmark_factory": {
            "trigger_state": get_path(state, ["synthetic_benchmark_factory", "trigger_state"], None),
            "cards": get_path(state, ["synthetic_benchmark_factory", "summary", "cards"], None),
            "ready_cards": get_path(state, ["synthetic_benchmark_factory", "summary", "ready_cards"], None),
            "case_count": get_path(state, ["synthetic_benchmark_factory", "summary", "case_count"], None),
            "cross_arm_case_count": get_path(state, ["synthetic_benchmark_factory", "summary", "cross_arm_case_count"], None),
        },
        "multi_stream": {
            "factory_trigger_state": get_path(state, ["multi_stream_trace_factory", "trigger_state"], None),
            "case_count": get_path(state, ["multi_stream_trace_factory", "summary", "case_count"], None),
            "latest_score": get_path(state, ["multi_stream_code_pressure", "score"], None),
            "multi_stream_pass_rate": get_path(state, ["multi_stream_code_pressure", "summary", "multi_stream_pass_rate"], None),
            "single_stream_pass_rate": get_path(state, ["multi_stream_code_pressure", "summary", "single_stream_transfer_pass_rate"], None),
            "monitorability_score": get_path(state, ["multi_stream_monitorability_probe", "summary", "monitorability_score"], None),
            "candidate_gate_trigger_state": get_path(state, ["multi_stream_candidate_gate", "trigger_state"], None),
        },
        "rl_summary": get_path(state, ["rl_benchmark_registry", "summary"], {}),
        "resource_summary": {
            "can_run": get_path(state, ["resource_governor", "decision", "can_run_requested_profile"], None),
            "recommended_profile": get_path(state, ["resource_governor", "decision", "recommended_profile"], None),
            "efficiency_score": get_path(state, ["resource_governor", "efficiency", "score"], None),
            "throttle_reasons": get_path(state, ["resource_governor", "decision", "throttle_reasons"], []),
        },
        "performance_optimizer": {
            "trigger_state": get_path(state, ["performance_optimizer", "trigger_state"], None),
            "score": get_path(state, ["performance_optimizer", "score"], None),
            "preferred_training_backend": get_path(
                state, ["performance_optimizer", "summary", "preferred_training_backend"], None
            ),
            "preferred_inference_backend": get_path(
                state, ["performance_optimizer", "summary", "preferred_inference_backend"], None
            ),
            "recent_ok_worker_chunks": get_path(
                state, ["performance_optimizer", "summary", "recent_ok_worker_chunks"], None
            ),
            "bottlenecks": [
                row.get("id")
                for row in (get_path(state, ["performance_optimizer", "bottlenecks"], []) or [])[:5]
                if isinstance(row, dict)
            ],
        },
        "hive": {
            "node_id": get_path(state, ["hive_status", "node_id"], None),
            "node_name": get_path(state, ["hive_status", "node_name"], None),
            "peer_count": get_path(state, ["hive_peers", "peer_count"], None),
            "scheduler_nodes": get_path(state, ["hive_scheduler", "summary", "nodes"], None),
            "remote_placements": get_path(state, ["hive_scheduler", "summary", "remote_placements"], None),
            "real_worker_chunks": get_path(state, ["hive_scheduler", "summary", "real_worker_chunks"], None),
            "best_training_node": get_path(state, ["hive_scheduler", "summary", "best_training_node"], None),
            "best_cuda_node": get_path(state, ["hive_scheduler", "summary", "best_cuda_node"], None),
            "best_mlx_node": get_path(state, ["hive_scheduler", "summary", "best_mlx_node"], None),
            "best_inference_node": get_path(state, ["hive_scheduler", "summary", "best_inference_node"], None),
            "public_contribution_enabled": get_path(state, ["public_hive_contribution", "enabled"], None),
            "public_contribution_ready": get_path(state, ["public_hive_contribution", "can_connect_now"], None),
        },
        "license": {
            "registration_complete": get_path(state, ["license_status", "registration_complete"], None),
            "tier": get_path(state, ["license_status", "entitlement", "tier"], None),
            "source": get_path(state, ["license_status", "entitlement", "source"], None),
            "paid": get_path(state, ["license_status", "entitlement", "paid"], None),
            "worker_chunks_allowed": get_path(state, ["license_status", "feature_summary", "can_run_worker_chunks"], None),
            "company_hive_allowed": get_path(state, ["license_status", "feature_summary", "can_create_company_hive"], None),
        },
        "autonomous_goal_summary": {
            "selected_arms": get_path(state, ["autonomous_goal", "selected_arms"], []),
            "teacher_needed": get_path(state, ["autonomous_goal", "teacher_needed"], None),
            "ok": get_path(state, ["autonomous_goal", "outcome", "ok"], None),
        },
        "arm_lifecycle_summary": {
            "ready": get_path(state, ["arm_lifecycle_governance", "ready_for_long_autonomy"], None),
            "proposals": get_path(state, ["arm_lifecycle_governance", "summary", "proposal_count"], None),
            "split_candidates": get_path(state, ["arm_lifecycle_governance", "summary", "split_candidates"], None),
            "unknown_selected_arms": get_path(state, ["arm_lifecycle_governance", "summary", "unknown_selected_arm_count"], None),
            "teacher_recommended": get_path(state, ["arm_lifecycle_governance", "teacher_escalation", "recommended"], None),
        },
        "arm_sucker_registry": {
            "cores": get_path(state, ["arm_sucker_registry", "summary", "core_count"], None),
            "suckers": get_path(state, ["arm_sucker_registry", "summary", "sucker_count"], None),
            "ready_suckers": get_path(state, ["arm_sucker_registry", "summary", "ready_suckers"], None),
            "blocked_suckers": get_path(state, ["arm_sucker_registry", "summary", "blocked_suckers"], None),
            "average_maturity": get_path(state, ["arm_sucker_registry", "summary", "average_sucker_maturity"], None),
            "top_ready_suckers": get_path(state, ["arm_sucker_registry", "summary", "top_ready_suckers"], []),
        },
        "launch_readiness": {
            "ready_for_autonomous_training": get_path(state, ["autonomy_launch_readiness", "ready_for_autonomous_training"], None),
            "ready_for_teacher_enabled_run": get_path(state, ["autonomy_launch_readiness", "ready_for_teacher_enabled_run"], None),
            "ready_for_candidate_promotion": get_path(state, ["autonomy_launch_readiness", "ready_for_candidate_promotion"], None),
            "blockers": get_path(state, ["autonomy_launch_readiness", "blocker_failures"], []),
            "warnings": get_path(state, ["autonomy_launch_readiness", "warning_failures"], []),
        },
        "legacy_port_runtime_enforcement": {
            "trigger_state": get_path(state, ["legacy_port_runtime_enforcement", "summary", "trigger_state"], None),
            "ready_for_bounded_autonomy": get_path(state, ["legacy_port_runtime_enforcement", "ready_for_bounded_autonomy"], None),
            "ready_for_long_autonomy": get_path(state, ["legacy_port_runtime_enforcement", "ready_for_long_autonomy"], None),
            "ready_for_self_evolution": get_path(state, ["legacy_port_runtime_enforcement", "ready_for_self_evolution"], None),
            "blockers": get_path(state, ["legacy_port_runtime_enforcement", "blockers"], []),
        },
        "context_packets": {
            "packet_count": get_path(state, ["context_packets", "summary", "packet_count"], None),
            "active_packet_count": get_path(state, ["context_packets", "summary", "active_packet_count"], None),
            "summary_packet_count": get_path(state, ["context_packets", "summary", "summary_packet_count"], None),
            "drop_candidate_count": get_path(state, ["context_packets", "summary", "drop_candidate_count"], None),
            "top_score": get_path(state, ["context_packets", "summary", "top_score"], None),
        },
        "capability_matrix": {
            "capabilities": get_path(state, ["capability_matrix", "summary", "capabilities"], None),
            "average_maturity": get_path(state, ["capability_matrix", "summary", "average_maturity"], None),
            "ready_or_active": get_path(state, ["capability_matrix", "summary", "ready_or_active"], None),
            "partial_or_blocked": get_path(state, ["capability_matrix", "summary", "partial_or_blocked"], None),
            "behind_market_count": get_path(state, ["capability_matrix", "summary", "behind_market_count"], None),
            "differentiated_count": get_path(state, ["capability_matrix", "summary", "differentiated_count"], None),
        },
        "benchmaxx_curriculum": {
            "current_stage_id": get_path(state, ["benchmaxx_curriculum", "summary", "current_stage_id"], None),
            "current_stage_status": get_path(state, ["benchmaxx_curriculum", "summary", "current_stage_status"], None),
            "next_frontier_family": get_path(state, ["benchmaxx_curriculum", "summary", "next_frontier_family"], None),
            "locked_stages": get_path(state, ["benchmaxx_curriculum", "summary", "locked_stages"], None),
            "active_stages": get_path(state, ["benchmaxx_curriculum", "summary", "active_stages"], None),
            "blocked_stages": get_path(state, ["benchmaxx_curriculum", "summary", "blocked_stages"], None),
            "near_term_queue_count": len(get_path(state, ["benchmaxx_curriculum", "near_term_queue"], [])),
        },
        "benchmark_adapter_factory": {
            "cards": get_path(state, ["benchmark_adapter_factory", "summary", "cards"], None),
            "ready_cards": get_path(state, ["benchmark_adapter_factory", "summary", "ready_cards"], None),
            "needs_smoke": get_path(state, ["benchmark_adapter_factory", "summary", "needs_smoke"], None),
            "blocked": get_path(state, ["benchmark_adapter_factory", "summary", "blocked"], None),
            "written_cards": get_path(state, ["benchmark_adapter_factory", "summary", "written_cards"], None),
        },
        "benchmark_pantry_unblocker": {
            "blocked_cards": get_path(state, ["benchmark_pantry_unblocker", "summary", "blocked_cards"], None),
            "autonomous_safe_actions": get_path(
                state, ["benchmark_pantry_unblocker", "summary", "autonomous_safe_actions"], None
            ),
            "runtime_dependency_work": get_path(
                state, ["benchmark_pantry_unblocker", "summary", "runtime_dependency_work"], None
            ),
            "waiting_on_private_assets": get_path(
                state, ["benchmark_pantry_unblocker", "summary", "waiting_on_private_assets"], None
            ),
        },
        "architecture_experiment_governance": {
            "architecture_change_allowed": get_path(state, ["architecture_experiment_governance", "architecture_change_allowed"], None),
            "recommended_next_experiment": get_path(state, ["architecture_experiment_governance", "recommended_next_experiment", "id"], None),
            "recommended_status": get_path(state, ["architecture_experiment_governance", "recommended_next_experiment", "status"], None),
        },
        "architecture_experiment_runner": {
            "status": get_path(state, ["architecture_experiment_runner", "status"], None),
            "selected": [
                row.get("id")
                for row in get_path(state, ["architecture_experiment_runner", "selected"], [])[:3]
                if isinstance(row, dict)
            ],
            "score_delta": get_path(state, ["architecture_experiment_runner", "score_delta"], {}),
        },
        "autoresearch_gap_audit": {
            "trigger_state": get_path(state, ["autoresearch_gap_audit", "summary", "trigger_state"], None),
            "gap_count": get_path(state, ["autoresearch_gap_audit", "summary", "gap_count"], None),
            "ledger_entries": get_path(state, ["autoresearch_gap_audit", "summary", "ledger_entries"], None),
            "needs_baseline": get_path(state, ["autoresearch_gap_audit", "summary", "needs_baseline"], None),
        },
        "loop_closure_harvester": {
            "candidates": get_path(state, ["loop_closure_harvester", "summary", "candidates"], None),
            "ready_for_tool_synthesis": get_path(state, ["loop_closure_harvester", "summary", "ready_for_tool_synthesis"], None),
        },
        "loop_closure_tool_promoter": {
            "promoted": len(get_path(state, ["loop_closure_tool_promoter", "promoted"], [])),
            "after_tools": get_path(state, ["loop_closure_tool_promoter", "after_tools"], None),
        },
        "transfer_eval_suite": {
            "score": get_path(state, ["transfer_eval_suite", "summary", "accuracy"], None),
            "task_count": get_path(state, ["transfer_eval_suite", "summary", "task_count"], None),
            "frontier": get_path(state, ["transfer_eval_suite", "summary", "frontier"], None),
        },
        "native_voice_training_manifest": {
            "status": get_path(state, ["native_voice_training_manifest", "summary", "status"], None),
            "sources": get_path(state, ["native_voice_training_manifest", "summary", "sources"], None),
            "stt_sources": get_path(state, ["native_voice_training_manifest", "summary", "stt_sources"], None),
            "tts_sources": get_path(state, ["native_voice_training_manifest", "summary", "tts_sources"], None),
            "tiny_audio_clips": get_path(state, ["native_voice_training_manifest", "summary", "tiny_audio_clips"], None),
            "ready_for_native_training": get_path(
                state, ["native_voice_training_manifest", "summary", "ready_for_native_training"], None
            ),
        },
        "arm_transfer_plan": {
            "frontier_family": get_path(state, ["arm_transfer_plan", "summary", "frontier_family"], None),
            "transfer_edges": get_path(state, ["arm_transfer_plan", "summary", "transfer_edges"], None),
            "ready_edges": get_path(state, ["arm_transfer_plan", "summary", "ready_edges"], None),
            "blocked_edges": get_path(state, ["arm_transfer_plan", "summary", "blocked_edges"], None),
            "sucker_edges": get_path(state, ["arm_transfer_plan", "summary", "sucker_edges"], None),
            "target_suckers": get_path(state, ["arm_transfer_plan", "summary", "target_suckers"], []),
            "next_actions": get_path(state, ["arm_transfer_plan", "next_actions"], [])[:4],
        },
        "arm_transfer_artifacts": {
            "artifacts": get_path(state, ["arm_transfer_artifacts", "summary", "artifacts"], None),
            "frontier_family": get_path(state, ["arm_transfer_artifacts", "summary", "frontier_family"], None),
        },
        "model_growth_gate": {
            "model_growth_allowed": get_path(state, ["model_growth_gate", "model_growth_allowed"], None),
            "hard_blockers": get_path(state, ["model_growth_gate", "hard_blockers"], []),
            "missing_evidence": get_path(state, ["model_growth_gate", "missing_evidence"], []),
        },
        "self_evolution_governance": {
            "teacher_apply_allowed": get_path(state, ["self_evolution_governance", "teacher_apply", "allowed_now"], None),
            "teacher_apply_blockers": get_path(state, ["self_evolution_governance", "teacher_apply", "blockers"], []),
            "lane_count": len(get_path(state, ["self_evolution_governance", "lanes"], [])),
        },
        "teacher_self_edit_proof": {
            "status": get_path(state, ["teacher_self_edit_proof", "status"], None),
            "success_rate": get_path(state, ["teacher_self_edit_proof", "summary", "success_rate"], None),
            "recent_successes": get_path(state, ["teacher_self_edit_proof", "summary", "recent_successes"], None),
        },
        "attd": {
            "trigger_state": get_path(state, ["attd", "trigger_state"], None),
            "attd_score": get_path(state, ["attd", "attd_score"], None),
            "components": get_path(state, ["attd", "components"], {}),
            "maintenance_packets": get_path(state, ["attd_maintenance_packets", "packet_count"], None),
            "allows_long_autonomy": get_path(state, ["attd", "governance", "allows_long_autonomy"], None),
            "allows_architecture_change": get_path(state, ["attd", "governance", "allows_architecture_change"], None),
            "allows_adapter_card_writes": get_path(state, ["attd", "governance", "allows_adapter_card_writes"], None),
        },
        "coherence_delirium_gate": {
            "trigger_state": get_path(state, ["coherence_delirium_gate", "trigger_state"], None),
            "source_trigger_state": get_path(state, ["coherence_delirium_gate", "source_trigger_state"], None),
            "coherence_score": get_path(state, ["coherence_delirium_gate", "coherence_score"], None),
            "delirium_score": get_path(state, ["coherence_delirium_gate", "delirium_score"], None),
            "allows_long_autonomy": get_path(state, ["coherence_delirium_gate", "allows_long_autonomy"], None),
            "allows_candidate_promotion": get_path(state, ["coherence_delirium_gate", "allows_candidate_promotion"], None),
            "allows_self_edit": get_path(state, ["coherence_delirium_gate", "allows_self_edit"], None),
            "allows_capability_expansion": get_path(state, ["coherence_delirium_gate", "allows_capability_expansion"], None),
            "blockers": get_path(state, ["coherence_delirium_gate", "blockers"], []),
            "candidate_blockers": get_path(state, ["coherence_delirium_gate", "candidate_blockers"], []),
        },
        "external_inference_audit": {
            "ok": get_path(state, ["external_inference_audit", "ok"], None),
            "teacher_only_invariant": get_path(state, ["external_inference_audit", "teacher_only_invariant"], None),
            "summary": get_path(state, ["external_inference_audit", "summary"], {}),
        },
    }
