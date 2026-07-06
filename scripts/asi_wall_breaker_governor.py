#!/usr/bin/env python3
"""Executable wall map for Project Theseus.

This report is intentionally stricter than the operator scorecards. It turns
the known "ASI-seed" blockers into machine-readable gates, routing hints, and
safe next commands. It does not train, does not call external inference, and
does not unlock public calibration by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from theseus_archive_resolver import resolve_archived_path  # noqa: E402

REPORTS = ROOT / "reports"
PUBLIC_CALIBRATION_OPERATOR_LOCK = REPORTS / "public_calibration_operator_lock.flag"
PUBLIC_CODE_FLOOR = 0.70
PUBLIC_COVERAGE_FLOOR = 0.60
PUBLIC_NO_ADMISSIBLE_MAX = 0.25
PRIVATE_LEARNED_PASS_FLOOR = 0.70
PRIVATE_NO_ADMISSIBLE_MAX = 0.25
SIGNIFICANT_ARCH_DELTA = 0.01
EDGE_OBLIGATION_V1_BASELINE = 0.050314
BODY_EXEC_V1_BASELINE = 0.140625


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/asi_wall_breaker_governor.json")
    parser.add_argument("--markdown-out", default="reports/asi_wall_breaker_governor.md")
    args = parser.parse_args()

    state = observe()
    gates = global_gates(state)
    walls = build_walls(state, gates)
    hard_blockers = [row for row in walls if row.get("severity") == "hard" and row.get("status") != "cleared"]
    soft_blockers = [row for row in walls if row.get("severity") != "hard" and row.get("status") != "cleared"]
    next_action = choose_next_action(state, gates, walls)
    payload = {
        "policy": "project_theseus_asi_wall_breaker_governor_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_blockers else "YELLOW",
        "summary": {
            "wall_count": len(walls),
            "hard_blocker_count": len(hard_blockers),
            "soft_blocker_count": len(soft_blockers),
            "cleared_count": sum(1 for row in walls if row.get("status") == "cleared"),
            "public_calibration_allowed": gates["public_calibration_allowed"]["passed"],
            "public_calibration_operator_locked": bool(
                gates["public_calibration_allowed"]["evidence"].get("operator_lock_active")
            ),
            "model_growth_allowed": gates["model_growth_allowed"]["passed"],
            "candidate_promotion_allowed": gates["candidate_promotion_allowed"]["passed"],
            "next_primary_action": next_action["label"],
            "next_primary_command": next_action.get("command") or [],
            "closed_loop_residual_ratchet_decision": object_field(
                state.get("closed_loop_residual_ratchet"), "summary"
            ).get("decision"),
            "closed_loop_residual_ratchet_reason": object_field(
                state.get("closed_loop_residual_ratchet"), "summary"
            ).get("decision_reason"),
            "north_star": "observe -> act -> verify -> store evidence -> learn -> transfer -> route better next time",
            "external_inference_calls": 0,
        },
        "gates": gates,
        "walls": walls,
        "recommended_next_actions": recommended_actions(walls, next_action),
        "routing_hints": routing_hints(walls, gates),
        "safety_policy": {
            "public_benchmarks": "calibration-only; no public tests or solutions become training rows",
            "teacher": "proposal-only architecture experiments unless explicitly approved otherwise",
            "model_growth": "blocked until public transfer, coherence, promotion, and architecture-delta gates clear",
            "unsafe_actions": "no bulk downloads, uncertain-license data, commercial ROM fetching, public gateway operation, destructive actions, or teacher apply mode without explicit approval",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0


def observe() -> dict[str, Any]:
    return {
        "broad": read_json(REPORTS / "broad_transfer_matrix.json", {}),
        "preflight": read_json(REPORTS / "code_lm_closure_public_contract_preflight_seed23_32.json", {}),
        "decoder_gate": read_json(REPORTS / "decoder_v2_private_ablation_gate.json", {}),
        "transfer_proof": read_json(REPORTS / "private_public_transfer_proof.json", {}),
        "transfer_residual_packet": read_json(REPORTS / "private_public_transfer_residual_packet.json", {}),
        "candidate_replay_contract": read_json(REPORTS / "private_candidate_replay_contract_audit_v1.json", {}),
        "full_body_contract_transfer_recovery": read_json(REPORTS / "full_body_contract_transfer_recovery_v1.json", {}),
        "private_full_body_repair_runtime_readiness": read_json(
            REPORTS / "private_full_body_repair_runtime_readiness_v1.json", {}
        ),
        "broad_floor_recovery": read_json(REPORTS / "broad_public_code_transfer_floor_recovery.json", {}),
        "broad_residual_reader": read_json(REPORTS / "broad_transfer_residual_reader.json", {}),
        "broad_edge_v10_decoder_gate": read_json(
            REPORTS / "decoder_v2_private_ablation_gate_broad_floor_edge_contract_v10.json", {}
        ),
        "broad_edge_v10_transfer_proof": read_json(
            REPORTS / "private_public_transfer_proof_broad_floor_edge_contract_v10.json", {}
        ),
        "broad_post_v10_ablation": read_json(
            REPORTS / "broad_transfer_residual_decoder_ablation_after_v10_demote.json", {}
        ),
        "broad_edge_type_scale64_ablation": read_json(
            REPORTS / "broad_transfer_residual_decoder_ablation_edge_type_scale64.json", {}
        ),
        "edge_full_body_bridge_v2": read_json(
            REPORTS / "edge_full_body_contract_bridge_v2_private_edge_obligation_gate.json", {}
        ),
        "edge_full_body_bridge_v2_baseline": read_json(
            REPORTS / "edge_full_body_contract_bridge_v2_private_edge_obligation_gate_baseline.json", {}
        ),
        "edge_full_body_bridge_v2_public_verdict": read_json(
            REPORTS / "real_code_benchmark_graduation_edge_full_body_contract_bridge_v2_public_verdict.json", {}
        ),
        "edge_full_body_bridge_v2_public_verdict_residual_packet": read_json(
            REPORTS / "public_transfer_residual_packet_edge_full_body_contract_bridge_v2_public_verdict.json", {}
        ),
        "closed_loop_residual_ratchet": read_json(REPORTS / "closed_loop_residual_ratchet.json", {}),
        "private_closure": read_json(REPORTS / "code_lm_closure_private_pressure_private.json", {}),
        "train_once_closure": read_json(REPORTS / "code_lm_closure_private_pressure_private_recovery_train_once_fanout_v1.json", {}),
        "execution_shape": latest_execution_shape_candidate_coverage_report(),
        "scheduler": read_json(REPORTS / "high_transfer_curriculum_scheduler.json", {}),
        "architecture_results": read_json(REPORTS / "architecture_experiment_results.json", {}),
        "architecture_governance": read_json(REPORTS / "architecture_experiment_governance.json", {}),
        "report_store": read_json(REPORTS / "report_evidence_store.json", {}),
        "cross_domain": read_json(REPORTS / "cross_domain_sts_capsules.json", {}),
        "sts_causal_decoder_ablation": read_json(REPORTS / "sts_causal_decoder_ablation.json", {}),
        "agent_lane_transfer_gate": read_json(REPORTS / "agent_lane_transfer_gate.json", {}),
        "symliquid": read_json(REPORTS / "symliquid_state_engine.json", {}),
        "model_growth": read_json(REPORTS / "model_growth_gate.json", {}),
        "watchdog": read_json(REPORTS / "autonomy_watchdog.json", {}),
        "public_calibration_operator_lock": public_calibration_operator_lock_state(),
        "service_hygiene": read_json(REPORTS / "service_process_hygiene.json", {}),
        "conversation_v4": read_json(REPORTS / "high_transfer_multi_turn_conversation_hard_v4.json", {}),
        "tool_use": read_json(REPORTS / "high_transfer_long_horizon_tool_use.json", {}),
        "board_policy": read_json(REPORTS / "board_game_learned_policy.json", {}),
        "pufferlib": read_json(REPORTS / "pufferlib4_rl_lane.json", read_json(REPORTS / "pufferlib4_capability_probe.json", {})),
        "candidate_gate": read_json(REPORTS / "candidate_promotion_gate.json", {}),
        "coherence_gate": read_json(REPORTS / "coherence_delirium_gate.json", {}),
        "a_plus": read_json(REPORTS / "a_plus_operating_scorecard.json", {}),
        "system_efficiency": read_json(REPORTS / "system_efficiency_audit.json", {}),
        "attd": read_json(REPORTS / "attd_report.json", {}),
        "attd_packets": read_json(REPORTS / "attd_maintenance_packets.json", {}),
        "complexity": complexity_snapshot(),
    }


def global_gates(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    broad_summary = object_field(state["broad"], "summary")
    decoder_summary = object_field(state["decoder_gate"], "summary")
    execution_summary = object_field(state["execution_shape"], "summary")
    preflight_summary = object_field(state["preflight"], "summary")
    private_summary = object_field(state["private_closure"], "summary")
    train_once_summary = object_field(state["train_once_closure"], "summary")

    preflight_ok = bool(
        state["preflight"].get("trigger_state") == "GREEN"
        and int(number(preflight_summary.get("varargs_task_count")) or 0) == 0
        and int(number(preflight_summary.get("weak_required_construct_count")) or 0) == 0
        and int(number(preflight_summary.get("weak_full_body_count")) or 0) == 0
        and not preflight_summary.get("hard_blockers")
    )
    learned_pass = first_number(execution_summary.get("learned_token_decoder_pass_rate"))
    learned_no_admissible = first_number(execution_summary.get("learned_token_decoder_no_admissible_candidate_rate"))
    learned_no_admissible = 1.0 if learned_no_admissible is None else learned_no_admissible
    diagnostic_template_count = first_number(execution_summary.get("diagnostic_template_candidate_count"))
    diagnostic_template_count = 0.0 if diagnostic_template_count is None else diagnostic_template_count
    private_coverage_ok = bool(
        state["execution_shape"].get("trigger_state") == "GREEN"
        and float(learned_pass or 0.0) >= PRIVATE_LEARNED_PASS_FLOOR
        and float(learned_no_admissible) <= PRIVATE_NO_ADMISSIBLE_MAX
        and int(diagnostic_template_count) == 0
        and bool(execution_summary.get("learned_token_public_gate_ready", True))
    )
    public_eligible = first_number(decoder_summary.get("public_eligible_task_coverage")) or 0.0
    public_actual = first_number(decoder_summary.get("public_actual_token_task_coverage")) or 0.0
    public_no_admissible = first_number(decoder_summary.get("public_no_admissible_task_rate"))
    public_no_admissible = 1.0 if public_no_admissible is None else public_no_admissible
    public_receiver_ok = bool(
        bool(state["decoder_gate"].get("ready_for_public_calibration"))
        and float(public_eligible) >= PUBLIC_COVERAGE_FLOOR
        and float(public_actual) >= PUBLIC_COVERAGE_FLOOR
        and float(public_no_admissible) <= PUBLIC_NO_ADMISSIBLE_MAX
    )
    transfer_contract = current_private_transfer_contract_evidence(state)
    transfer_proof_ok = bool(transfer_contract.get("passed"))
    edge_bridge = edge_full_body_bridge_v2_evidence(state)
    edge_bridge_ok = bool(edge_bridge.get("passed"))
    edge_bridge_public_verdict = edge_full_body_bridge_v2_public_verdict_evidence(state)
    coverage_mtime = report_mtime(state["execution_shape"])
    closure_mtime = path_mtime(REPORTS / "code_lm_closure_private_pressure_private.json")
    train_once_closure_path = REPORTS / "code_lm_closure_private_pressure_private_recovery_train_once_fanout_v1.json"
    train_once_mtime = path_mtime(train_once_closure_path)
    gate_mtime = path_mtime(REPORTS / "decoder_v2_private_ablation_gate.json")
    closure_after_decoder_patch = bool(coverage_mtime and closure_mtime and closure_mtime >= coverage_mtime)
    gate_after_closure = bool(gate_mtime and closure_mtime and gate_mtime >= closure_mtime)
    private_closure_run_status = state["private_closure"].get("run_status") or private_summary.get("run_status")
    legacy_fresh_private_chain = bool(private_closure_run_status == "completed" and closure_after_decoder_patch and gate_after_closure)
    train_once_run_status = state["train_once_closure"].get("run_status") or train_once_summary.get("run_status")
    train_once_completed = bool(
        state["train_once_closure"].get("trigger_state") == "GREEN"
        and train_once_run_status == "completed"
        and train_once_summary.get("train_once_checkpoint_fanout") is True
        and train_once_summary.get("repeated_training_per_candidate_shard") is False
    )
    decoder_latest_closure = str(decoder_summary.get("latest_closure") or "").replace("\\", "/")
    decoder_points_to_train_once = decoder_latest_closure.endswith(
        "reports/code_lm_closure_private_pressure_private_recovery_train_once_fanout_v1.json"
    )
    train_once_after_decoder_patch = bool(coverage_mtime and train_once_mtime and train_once_mtime >= coverage_mtime)
    gate_after_train_once = bool(gate_mtime and train_once_mtime and gate_mtime >= train_once_mtime)
    train_once_fresh_private_chain = bool(
        train_once_completed and decoder_points_to_train_once and train_once_after_decoder_patch and gate_after_train_once
    )
    decoder_latest_closure_path = resolve(decoder_latest_closure) if decoder_latest_closure else Path()
    decoder_latest_report = read_json(decoder_latest_closure_path, {}) if decoder_latest_closure else {}
    decoder_latest_summary = object_field(decoder_latest_report, "summary")
    decoder_latest_mtime = path_mtime(decoder_latest_closure_path) if decoder_latest_closure else 0.0
    decoder_latest_completed = bool(
        decoder_latest_report.get("trigger_state") == "GREEN"
        and decoder_latest_report.get("run_status") == "completed"
        and decoder_latest_summary.get("train_once_checkpoint_fanout") is True
        and decoder_latest_summary.get("repeated_training_per_candidate_shard") is False
    )
    decoder_latest_after_decoder_patch = bool(
        coverage_mtime and decoder_latest_mtime and decoder_latest_mtime >= coverage_mtime
    )
    gate_after_decoder_latest = bool(gate_mtime and decoder_latest_mtime and gate_mtime >= decoder_latest_mtime)
    decoder_latest_fresh_private_chain = bool(
        decoder_latest_completed and decoder_latest_after_decoder_patch and gate_after_decoder_latest
    )
    fresh_private_chain = bool(
        legacy_fresh_private_chain or train_once_fresh_private_chain or decoder_latest_fresh_private_chain
    )
    public_pass_rate = first_number(broad_summary.get("real_public_pass_rate"), broad_summary.get("aggregate_pass_rate")) or 0.0
    architecture_delta = architecture_delta_evidence(state["architecture_results"])
    architecture_delta_ok = bool(architecture_delta["has_significant_delta"])
    sts_causal_summary = object_field(state["sts_causal_decoder_ablation"], "summary")
    sts_causal_ok = bool(
        state["sts_causal_decoder_ablation"].get("trigger_state") == "GREEN"
        and sts_causal_summary.get("same_seed_non_sts_comparator_present") is True
    )
    operator_lock_active = bool(state["public_calibration_operator_lock"].get("active"))
    agent_lane_transfer_ok = state["agent_lane_transfer_gate"].get("trigger_state") == "GREEN"
    coherence_ok = state["coherence_gate"].get("trigger_state") == "GREEN"
    candidate_promotion_ok = bool(state["candidate_gate"].get("promote")) and coherence_ok

    gates = {
        "contract_preflight_green": gate(preflight_ok, preflight_summary),
        "private_candidate_coverage_green": gate(private_coverage_ok, execution_summary),
        "fresh_private_closure_after_decoder_patch": gate(
            fresh_private_chain,
            {
                "private_closure_run_status": private_closure_run_status,
                "coverage_report": state["execution_shape"].get("source_report_path"),
                "coverage_mtime": coverage_mtime,
                "closure_mtime": closure_mtime,
                "decoder_gate_mtime": gate_mtime,
                "closure_after_decoder_patch": closure_after_decoder_patch,
                "gate_after_closure": gate_after_closure,
                "legacy_fresh_private_chain": legacy_fresh_private_chain,
                "train_once_closure": str(train_once_closure_path).replace("\\", "/"),
                "train_once_run_status": train_once_run_status,
                "train_once_trigger_state": state["train_once_closure"].get("trigger_state"),
                "train_once_mtime": train_once_mtime,
                "train_once_checkpoint_fanout": train_once_summary.get("train_once_checkpoint_fanout"),
                "train_once_repeated_training_per_candidate_shard": train_once_summary.get("repeated_training_per_candidate_shard"),
                "decoder_points_to_train_once": decoder_points_to_train_once,
                "train_once_after_decoder_patch": train_once_after_decoder_patch,
                "gate_after_train_once": gate_after_train_once,
                "train_once_fresh_private_chain": train_once_fresh_private_chain,
                "decoder_latest_closure": decoder_latest_closure,
                "decoder_latest_completed": decoder_latest_completed,
                "decoder_latest_mtime": decoder_latest_mtime,
                "decoder_latest_after_decoder_patch": decoder_latest_after_decoder_patch,
                "gate_after_decoder_latest": gate_after_decoder_latest,
                "decoder_latest_fresh_private_chain": decoder_latest_fresh_private_chain,
            },
        ),
        "public_receiver_candidate_coverage_green": gate(public_receiver_ok, decoder_summary),
        "private_public_transfer_proof_green": gate(
            transfer_proof_ok,
            transfer_contract,
        ),
        "edge_full_body_contract_bridge_v2_green": gate(edge_bridge_ok, edge_bridge),
        "edge_full_body_contract_bridge_v2_public_verdict": gate(
            bool(edge_bridge_public_verdict.get("present")),
            edge_bridge_public_verdict,
        ),
        "broad_public_transfer_floor": gate(public_pass_rate >= PUBLIC_CODE_FLOOR, {"pass_rate": public_pass_rate, "floor": PUBLIC_CODE_FLOOR}),
        "architecture_experiment_delta_evidence": gate(architecture_delta_ok, architecture_delta),
        "sts_causal_decoder_evidence": gate(sts_causal_ok, sts_causal_evidence(state)),
        "agent_lane_transfer_evidence": gate(
            agent_lane_transfer_ok, object_field(state["agent_lane_transfer_gate"], "summary")
        ),
        "report_causality_evidence": gate(report_causality_ok(state), report_causality_evidence(state)),
        "model_growth_allowed": gate(bool(state["model_growth"].get("model_growth_allowed")), model_growth_evidence(state)),
        "candidate_promotion_allowed": gate(candidate_promotion_ok, promotion_evidence(state)),
    }
    public_calibration_prereqs_ok = bool(
        preflight_ok and private_coverage_ok and fresh_private_chain and public_receiver_ok and transfer_proof_ok
    )
    gates["public_calibration_allowed"] = gate(
        bool(public_calibration_prereqs_ok and not operator_lock_active),
        {
            "rule": "GREEN contract preflight -> GREEN private candidate coverage -> fresh private closure after latest decoder patch -> decoder_v2 gate ready -> current private transfer contract",
            "technical_prerequisites_green": public_calibration_prereqs_ok,
            "operator_lock_active": operator_lock_active,
            "operator_lock": state["public_calibration_operator_lock"],
            "contract_preflight_green": preflight_ok,
            "private_candidate_coverage_green": private_coverage_ok,
            "fresh_private_closure_after_decoder_patch": fresh_private_chain,
            "public_receiver_candidate_coverage_green": public_receiver_ok,
            "private_public_transfer_proof_green": transfer_proof_ok,
            "edge_full_body_contract_bridge_v2_green": edge_bridge_ok,
            "spent_edge_full_body_contract_bridge_v2_public_verdict": edge_bridge_public_verdict,
        },
    )
    return gates


def build_walls(state: dict[str, Any], gates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    broad_summary = object_field(state["broad"], "summary")
    decoder_summary = object_field(state["decoder_gate"], "summary")
    execution_summary = object_field(state["execution_shape"], "summary")
    conversation_summary = object_field(state["conversation_v4"], "summary")
    puffer_summary = object_field(state["pufferlib"], "summary")
    service_summary = object_field(state["service_hygiene"], "summary")
    watchdog_summary = object_field(state["watchdog"], "summary")
    complexity = state["complexity"]
    system_efficiency_summary = object_field(state["system_efficiency"], "summary")
    attd_alignment_summary = get_path(state["system_efficiency"], ["attd_alignment", "summary"], {})
    attd_alignment_summary = attd_alignment_summary if isinstance(attd_alignment_summary, dict) else {}
    attd_components = state["attd"].get("components") if isinstance(state["attd"].get("components"), dict) else {}
    efficiency_red = state["system_efficiency"].get("trigger_state") == "RED"
    hard_maintainability_hotspots = int(number(system_efficiency_summary.get("hard_maintainability_hotspot_count")) or 0)
    attd_runtime_overlap_count = int(number(system_efficiency_summary.get("attd_runtime_overlap_count")) or 0)
    loop_bottleneck_count = int(number(system_efficiency_summary.get("loop_bottleneck_count")) or 0)
    iteration_speed_wall_cleared = (
        loop_bottleneck_count == 0
        and hard_maintainability_hotspots == 0
        and attd_runtime_overlap_count == 0
    )
    transfer_green = bool(gates["private_public_transfer_proof_green"]["passed"])
    residual_next = {} if transfer_green else transfer_residual_next_action(state)
    code_pass = first_number(broad_summary.get("real_public_pass_rate"), broad_summary.get("aggregate_pass_rate")) or 0.0
    private_pass = first_number(execution_summary.get("learned_token_decoder_pass_rate")) or 0.0
    private_no_admissible = first_number(execution_summary.get("learned_token_decoder_no_admissible_candidate_rate"))
    private_no_admissible = 1.0 if private_no_admissible is None else private_no_admissible
    transfer_gap_title = (
        "Private replay/full-body transfer contract is green; public transfer remains unspent until a governed calibration."
        if transfer_green
        else "Private learned-token recovery exists, but receiver transfer is not proven."
    )

    return [
        wall(
            "code_public_transfer_cap",
            "Code public transfer is still below the external floor.",
            "cleared" if code_pass >= PUBLIC_CODE_FLOOR else "blocked",
            "hard",
            {
                "real_public_pass_rate": code_pass,
                "floor": PUBLIC_CODE_FLOOR,
                "below_floor_cards": broad_summary.get("below_floor_cards") or broad_summary.get("cards_below_floor") or [],
                "public_receiver_ready": gates["public_receiver_candidate_coverage_green"]["passed"],
                "private_public_transfer_proof_green": transfer_green,
                "edge_full_body_contract_bridge_v2_green": gates["edge_full_body_contract_bridge_v2_green"]["passed"],
                "edge_full_body_contract_bridge_v2": gates["edge_full_body_contract_bridge_v2_green"]["evidence"],
                "closed_loop_residual_ratchet": closed_loop_ratchet_evidence(state),
                "public_calibration_operator_locked": bool(
                    gates["public_calibration_allowed"]["evidence"].get("operator_lock_active")
                ),
                "floor_recovery": floor_recovery_evidence(state),
            },
            "Receiver transfer gates are green; keep public calibration operator-locked until intentionally unlocked for exactly one bounded 4-card calibration.",
        ),
        wall(
            "private_to_public_transfer_gap",
            transfer_gap_title,
            "cleared"
            if transfer_green
            else "ready_for_fresh_private_chain"
            if gates["private_candidate_coverage_green"]["passed"]
            else "blocked",
            "hard",
            {
                "private_learned_token_pass_rate": private_pass,
                "private_no_admissible_rate": private_no_admissible,
                "public_actual_token_task_coverage": decoder_summary.get("public_actual_token_task_coverage"),
                "public_no_admissible_task_rate": decoder_summary.get("public_no_admissible_task_rate"),
                "private_public_transfer_proof_green": gates["private_public_transfer_proof_green"]["passed"],
                "transfer_proof": gates["private_public_transfer_proof_green"]["evidence"],
                "transfer_residual_packet": residual_next.get("evidence"),
            },
            "Private-to-public receiver transfer proof is GREEN; no more bridge work is required before the operator decides whether to run one bounded public calibration."
            if transfer_green
            else residual_next.get("label")
            or "Run fresh private_pressure_private_closure with the patched decoder, regenerate decoder_v2_private_ablation_gate, then require private_public_transfer_proof before public calibration.",
        ),
        wall(
            "learner_substrate_too_small",
            "The learner must prove parser/AST-constrained generation transfers, not just pass local private coverage.",
            "partial",
            "hard",
            {
                "private_candidate_coverage_green": gates["private_candidate_coverage_green"]["passed"],
                "fresh_private_closure_after_decoder_patch": gates["fresh_private_closure_after_decoder_patch"]["passed"],
                "public_receiver_candidate_coverage_green": gates["public_receiver_candidate_coverage_green"]["passed"],
                "source_report": state["execution_shape"].get("source_report_path"),
            },
            "Keep parser/AST-constrained learned generation as the only valid decoder path; score it through private closure and ablation before public.",
        ),
        wall(
            "architecture_experiment_loop_weak",
            "Architecture experiments are safe but have not shown enough capability delta.",
            "cleared" if gates["architecture_experiment_delta_evidence"]["passed"] else "blocked",
            "hard",
            gates["architecture_experiment_delta_evidence"]["evidence"],
            "Require residual cluster -> patch/architecture change -> private held-out delta >= 0.01 or explicit downstream transfer lift -> rollback/promotion.",
        ),
        wall(
            "sts_symliquid_not_causal_enough",
            "STS/SymLiquid must prove same-seed causal effect on decoder candidates and downstream policy.",
            "cleared" if gates["sts_causal_decoder_evidence"]["passed"] else "blocked",
            "hard",
            gates["sts_causal_decoder_evidence"]["evidence"],
            "Run a fresh patched private closure, then compare baseline vs contract-guided vs STS-conditioned vs AST/interface-repair candidates with the same seed.",
        ),
        wall(
            "reports_can_overclaim",
            "Reports must be control signals with named consumers and A/B effects, not morale artifacts.",
            "cleared" if gates["report_causality_evidence"]["passed"] else "blocked",
            "soft",
            gates["report_causality_evidence"]["evidence"],
            "Every new GREEN report must name a consumer and one measured decision or outcome delta.",
        ),
        wall(
            "model_growth_blocked",
            "Model growth is correctly blocked until transfer and architecture-delta evidence clear.",
            "controlled" if not gates["model_growth_allowed"]["passed"] else "cleared",
            "soft",
            gates["model_growth_allowed"]["evidence"],
            "Keep growth blocked; do not add capacity until cheaper transfer mechanisms have real public/private deltas.",
        ),
        wall(
            "autonomy_noisy",
            "Autonomy is alive but still noisy: YELLOW watchdogs, repeated frontier streaks, or duplicate launchers.",
            "cleared" if state["watchdog"].get("trigger_state") == "GREEN" and int(number(watchdog_summary.get("same_frontier_streak")) or 0) < 12 else "needs_reliability_pass",
            "soft",
            {
                "watchdog_state": state["watchdog"].get("trigger_state"),
                "same_frontier_streak": watchdog_summary.get("same_frontier_streak"),
                "recent_cycle_failures": watchdog_summary.get("recent_cycle_failures"),
                "duplicate_service_count": service_summary.get("duplicate_service_count"),
                "missing_required_service_count": service_summary.get("missing_required_service_count"),
            },
            "Use heartbeat/progress files for long loops and demote same-frontier streaks unless a useful signal appears.",
        ),
        wall(
            "non_code_lanes_not_frontier_grade",
            "Non-code lanes are active, but several still need harder evidence and transfer deltas.",
            "cleared" if non_code_frontier_ready(state) and gates["agent_lane_transfer_evidence"]["passed"] else "needs_frontier_upgrade",
            "soft",
            {
                "agent_lane_transfer_gate": gates["agent_lane_transfer_evidence"]["evidence"],
                "conversation_v4_accuracy": conversation_summary.get("accuracy"),
                "conversation_v4_graduated": conversation_summary.get("graduated"),
                "tool_use_case_count": get_path(state["tool_use"], ["summary", "case_count"], 0),
                "board_policy_rows": get_path(state["board_policy"], ["summary", "policy_train_row_count"], 0),
                "puffer_native_backend_ready": puffer_summary.get("native_backend_ready") or puffer_summary.get("native_backend_ok"),
                "puffer_policy_learning_evidence": puffer_summary.get("native_policy_learning_evidence"),
            },
            "Graduate conversation v4, make tool-use less synthetic, and require RL/game policies to produce transferable STS/control deltas.",
        ),
        wall(
            "codebase_complexity_drag",
            "Large modules still slow architecture evolution even after the 20k-line split.",
            "cleared" if complexity["max_source_lines"] <= 4500 else "managed_debt",
            "soft",
            {
                **complexity,
                "system_efficiency_maintainability_score": system_efficiency_summary.get("maintainability_score"),
                "hard_maintainability_hotspot_count": hard_maintainability_hotspots,
                "attd_score": state["attd"].get("attd_score"),
                "attd_components": attd_components,
            },
            "Continue extracting decoder core, verifier, ranking, STS conditioning, ingestion, execution repair, and report writing behind stable interfaces.",
        ),
        wall(
            "iteration_speed_and_assembly_debt",
            "ASI requires fast, safe iteration; slow opaque loops plus high assembly debt block compounding.",
            "cleared"
            if iteration_speed_wall_cleared
            else "blocked"
            if efficiency_red or hard_maintainability_hotspots
            else "needs_monitoring"
            if state["system_efficiency"].get("trigger_state") == "YELLOW"
            else "cleared",
            "hard" if efficiency_red and hard_maintainability_hotspots else "soft",
            {
                "system_efficiency_trigger_state": state["system_efficiency"].get("trigger_state"),
                "top_loop_bottleneck": system_efficiency_summary.get("top_loop_bottleneck"),
                "loop_bottleneck_count": loop_bottleneck_count,
                "maintainability_score": system_efficiency_summary.get("maintainability_score"),
                "hard_maintainability_hotspot_count": hard_maintainability_hotspots,
                "attd_trigger_state": state["attd"].get("trigger_state"),
                "attd_score": state["attd"].get("attd_score"),
                "attd_top_component": system_efficiency_summary.get("attd_top_component") or attd_alignment_summary.get("top_component"),
                "attd_runtime_overlap_count": attd_runtime_overlap_count,
                "attd_packet_count": system_efficiency_summary.get("attd_packet_count"),
                "architecture_cleanup_queue_count": system_efficiency_summary.get("architecture_cleanup_queue_count"),
                "top_architecture_cleanup_item": system_efficiency_summary.get("top_architecture_cleanup_item"),
            },
            "Runtime worker-chunk speed wall is closed; keep the planner fresh and move attention to architecture-delta, coherence, and broad-transfer blockers."
            if iteration_speed_wall_cleared
            else "Prioritize ATTD/runtime-overlap maintenance packets: split the slowest Code LM and benchmark hot paths into bounded modules with phase timing, staged verification, and stable interfaces.",
        ),
        wall(
            "promotion_and_coherence_blocked",
            "Candidate promotion is blocked by remaining transfer/profile gates."
            if state["coherence_gate"].get("trigger_state") == "GREEN"
            else "Candidate promotion and coherence gates still block model promotion.",
            "cleared" if gates["candidate_promotion_allowed"]["passed"] else "blocked",
            "hard",
            gates["candidate_promotion_allowed"]["evidence"],
            "Coherence is GREEN; keep promotion blocked until broad public transfer, maturity, and candidate evidence clear."
            if state["coherence_gate"].get("trigger_state") == "GREEN"
            else "Keep promotion blocked until coherence is GREEN and broad public transfer plus candidate evidence clear together.",
        ),
    ]


def transfer_residual_next_action(state: dict[str, Any]) -> dict[str, Any]:
    packet = state.get("transfer_residual_packet") if isinstance(state.get("transfer_residual_packet"), dict) else {}
    summary = object_field(packet, "summary")
    patch = object_field(packet, "next_source_level_decoder_patch")
    patch_id = str(summary.get("next_source_patch_id") or patch.get("id") or "")
    if packet.get("trigger_state") != "GREEN" or not patch_id:
        return {}
    mechanism = str(summary.get("next_source_patch_mechanism") or patch.get("mechanism") or "")
    private_acceptance = str(summary.get("private_eval_acceptance") or patch.get("private_eval_acceptance") or "")
    label = (
        f"Implement {patch_id}; public calibration stays locked until transfer proof shows "
        "public eligible coverage lift >= 0.03 without public tests or solutions."
    )
    return {
        "label": label,
        "command": [],
        "evidence": {
            "path": "reports/private_public_transfer_residual_packet.json",
            "next_source_patch_id": patch_id,
            "mechanism": mechanism,
            "private_eval_acceptance": private_acceptance,
            "failed_transfer_gates": summary.get("failed_transfer_gates") or [],
            "private_receiver_inventory_ready": summary.get("private_receiver_inventory_ready"),
        },
    }


def choose_next_action(state: dict[str, Any], gates: dict[str, dict[str, Any]], walls: list[dict[str, Any]]) -> dict[str, Any]:
    iteration_wall = next((row for row in walls if row.get("id") == "iteration_speed_and_assembly_debt"), {})
    iteration_loop_bottlenecks = int(number(get_path(iteration_wall, ["evidence", "loop_bottleneck_count"], 0)) or 0)
    if iteration_wall.get("status") == "blocked":
        return {
            "label": "Pause long training and clear ATTD/runtime-overlap architecture debt before more closure work.",
            "command": [
                "python",
                "scripts/system_efficiency_audit.py",
                "--out",
                "reports/system_efficiency_audit.json",
                "--markdown-out",
                "reports/system_efficiency_audit.md",
            ],
        }
    ratchet_action = closed_loop_ratchet_next_action(state)
    if ratchet_action:
        return ratchet_action
    if not gates["contract_preflight_green"]["passed"]:
        return {
            "label": "Run bounded public contract preflight only; public calibration remains locked.",
            "command": [
                "python",
                "scripts/code_lm_closure.py",
                "--public-cards",
                "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                "--seed",
                "23",
                "--max-public-cases-per-card",
                "32",
                "--private-count",
                "20",
                "--preflight-only",
                "--allow-concurrent",
                "--private-curriculum-out",
                "reports/code_lm_preflight_private_curriculum_seed23_32.jsonl",
                "--public-task-manifest-out",
                "reports/code_lm_public_tasks_preflight_seed23_32.jsonl",
                "--out",
                "reports/code_lm_closure_public_contract_preflight_seed23_32.json",
                "--lock-path",
                "reports/code_lm_closure_public_contract_preflight_seed23_32.lock",
            ],
        }
    if not gates["private_candidate_coverage_green"]["passed"]:
        return {
            "label": "Run bounded private learned-token candidate coverage gate; no public calibration.",
            "command": [
                "python",
                "scripts/execution_shape_private_ablation.py",
                "--max-eval-tasks",
                "32",
                "--out",
                "reports/execution_shape_candidate_coverage_current.json",
            ],
        }
    if not gates["fresh_private_closure_after_decoder_patch"]["passed"]:
        return {
            "label": "Run bounded chunked private recovery closure with patched decoder and all private pressure rows.",
            "command": [
                "python",
                "scripts/code_transfer_bounded_recovery_chain.py",
                "--execute",
            ],
        }
    if not gates["public_receiver_candidate_coverage_green"]["passed"]:
        if "decoder_fingerprint_current" in failed_gate_names(state["decoder_gate"]):
            return {
                "label": (
                    "Refresh four-card train-once fanout from the reusable checkpoint against current decoder source; "
                    "do not retrain or public-calibrate."
                ),
                "command": [
                    "python",
                    "scripts/code_lm_train_once_fanout.py",
                    "--execute",
                    "--slug",
                    "private_pressure_private_recovery_train_once_fanout_v1",
                    "--refresh-fanout-only",
                    "--full-fanout-refresh",
                    "--public-cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--max-public-cases-per-card",
                    "32",
                    "--private-count",
                    "320",
                    "--epochs",
                    "4",
                    "--candidates-per-task",
                    "8",
                    "--max-high-transfer-private-train",
                    "4800",
                    "--max-rust-work-steps",
                    "3000000",
                    "--rust-timeout-seconds",
                    "5400",
                    "--fanout-timeout-seconds",
                    "21600",
                    "--out",
                    "reports/code_lm_train_once_fanout.json",
                    "--markdown-out",
                    "reports/code_lm_train_once_fanout.md",
                ],
            }
        return {
            "label": "Run decoder_v2_private_ablation_gate to refresh receiver candidate coverage.",
            "command": ["python", "scripts/decoder_v2_private_ablation_gate.py"],
        }
    if not gates["private_public_transfer_proof_green"]["passed"]:
        return transfer_residual_next_action(state) or {
            "label": "Run private_public_transfer_proof; public calibration stays locked until same-seed receiver delta is proven.",
            "command": ["python", "scripts/private_public_transfer_proof.py"],
        }
    if not gates["sts_causal_decoder_evidence"]["passed"]:
        return {
            "label": "Run same-seed STS causal decoder ablation before any promotion claim.",
            "command": ["python", "scripts/sts_causal_decoder_ablation.py"],
        }
    if not gates["broad_public_transfer_floor"]["passed"]:
        if (
            gates["edge_full_body_contract_bridge_v2_green"]["passed"]
            and gates["public_receiver_candidate_coverage_green"]["passed"]
            and gates["private_public_transfer_proof_green"]["passed"]
        ):
            public_verdict = gates["edge_full_body_contract_bridge_v2_public_verdict"]["evidence"]
            if public_verdict.get("spent_failed"):
                packet = state.get("edge_full_body_bridge_v2_public_verdict_residual_packet")
                packet_green = isinstance(packet, dict) and packet.get("trigger_state") == "GREEN"
                if packet_green:
                    return {
                        "label": (
                            "The spent public verdict residual packet is GREEN; keep public calibration locked "
                            "and consume the paired edge/local-adapter decoder patch through private-only "
                            "validation or the next private closure before any future public request."
                        ),
                        "command": [],
                        "evidence": {
                            **public_verdict,
                            "residual_packet": "reports/public_transfer_residual_packet_edge_full_body_contract_bridge_v2_public_verdict.json",
                            "private_patch_focus": [
                                "edge_case",
                                "local_code_generation_adapter_needed",
                                "external_dependency_missing",
                                "interface_fidelity",
                                "return_shape_contract",
                                "no_admissible_reduction",
                            ],
                            "public_calibration": "locked_no_rerun",
                        },
                    }
                return {
                    "label": (
                        "The one authorized edge/full-body bridge public verdict is spent and failed the floor; "
                        "keep public calibration locked and convert exact residual families into private-only "
                        "edge/interface/admissibility architecture work."
                    ),
                    "command": [
                        "python",
                        "scripts/public_transfer_residual_packet.py",
                        "--real-code-verdict",
                        "reports/real_code_benchmark_graduation_edge_full_body_contract_bridge_v2_public_verdict.json",
                        "--out",
                        "reports/public_transfer_residual_packet_edge_full_body_contract_bridge_v2_public_verdict.json",
                        "--markdown-out",
                        "reports/public_transfer_residual_packet_edge_full_body_contract_bridge_v2_public_verdict.md",
                        "--prompt-out",
                        "reports/teacher_public_transfer_residual_prompt_edge_full_body_contract_bridge_v2_public_verdict.md",
                    ],
                    "evidence": public_verdict,
                }
            if gates["public_calibration_allowed"]["passed"]:
                return {
                    "label": "Allow exactly one bounded public 4-card calibration from GREEN private bridge evidence, then lock public reruns again.",
                    "command": scheduler_command(state, "receiver_calibration")
                    or ["python", "scripts/broad_code_calibration_scheduler.py", "--execute-once"],
                }
            return {
                "label": (
                    "Private edge/full-body bridge v2 is GREEN and receiver transfer gates are GREEN; "
                    "no spent failed public verdict is present, so keep public calibration operator-locked until "
                    "an explicit one-run 4-card calibration is approved."
                ),
                "command": [],
                "evidence": {
                    "edge_full_body_contract_bridge_v2": gates["edge_full_body_contract_bridge_v2_green"]["evidence"],
                    "decoder_gate_ready": gates["public_receiver_candidate_coverage_green"]["passed"],
                    "transfer_proof_ready": gates["private_public_transfer_proof_green"]["passed"],
                    "public_calibration_operator_locked": bool(
                        gates["public_calibration_allowed"]["evidence"].get("operator_lock_active")
                    ),
                    "public_calibration": "proposal_only_locked",
                },
            }
        recovery = floor_recovery_evidence(state)
        if not recovery.get("remaining_gap_explained"):
            return {
                "label": "Build private-only broad public code-transfer floor recovery evidence before another public calibration attempt.",
                "command": [
                    "python",
                    "scripts/broad_public_code_transfer_floor_recovery.py",
                    "--execute-ablation",
                ],
            }
        residual_action = broad_residual_reader_next_action(state)
        if residual_action:
            return residual_action
    if gates["public_calibration_allowed"]["evidence"].get("operator_lock_active"):
        if iteration_loop_bottlenecks <= 0 and not gates["architecture_experiment_delta_evidence"]["passed"]:
            return {
                "label": "Runtime speed wall is closed under operator lock; next target is a causal architecture experiment with measured private-heldout or transfer delta.",
                "command": ["python", "scripts/causal_architecture_delta_loop.py", "--execute-ablation"],
            }
        return {
            "label": "Keep public calibration operator-locked; optimize Code LM fanout/ranker/verifier speed and broad-transfer residuals.",
            "command": ["python", "scripts/system_efficiency_audit.py"],
        }
    if gates["public_calibration_allowed"]["passed"]:
        return {
            "label": "Allow exactly one bounded public 4-card calibration, then lock public reruns again.",
            "command": scheduler_command(state, "receiver_calibration") or ["python", "scripts/broad_code_calibration_scheduler.py", "--execute-once"],
        }
    return {
        "label": "Run the causal architecture delta loop with exact residual cluster; do not rerun public benchmarks.",
        "command": ["python", "scripts/causal_architecture_delta_loop.py", "--execute-ablation"],
    }


def closed_loop_ratchet_next_action(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("closed_loop_residual_ratchet")
    if not isinstance(report, dict) or not report:
        return {}
    summary = object_field(report, "summary")
    decision = object_field(report, "decision")
    kind = str(summary.get("decision") or decision.get("kind") or "")
    if kind not in {"promote", "rollback", "retry_private", "stop_blocker"}:
        return {}
    if report.get("trigger_state") not in {"GREEN", "YELLOW"}:
        return {}
    reason = str(summary.get("decision_reason") or decision.get("reason") or "closed-loop residual ratchet decision")
    command_value = decision.get("command")
    command = [str(part) for part in command_value] if isinstance(command_value, list) else []
    label_prefix = {
        "promote": "Closed-loop residual ratchet says promote",
        "rollback": "Closed-loop residual ratchet says rollback",
        "retry_private": "Closed-loop residual ratchet says retry privately",
        "stop_blocker": "Closed-loop residual ratchet stop blocker",
    }[kind]
    return {
        "label": f"{label_prefix}: {reason}",
        "command": command,
        "evidence": {
            "path": "reports/closed_loop_residual_ratchet.json",
            "trigger_state": report.get("trigger_state"),
            "decision": kind,
            "broad_public_pass_rate": summary.get("broad_public_pass_rate"),
            "public_floor": summary.get("public_floor"),
            "private_repair_pressure_rows": summary.get("private_repair_pressure_rows"),
            "same_seed_private_semantic_lift": summary.get("same_seed_private_semantic_lift"),
            "decoder_gate_ready": summary.get("decoder_gate_ready"),
            "private_public_transfer_ready": summary.get("private_public_transfer_ready"),
            "operator_lock_active": summary.get("operator_lock_active"),
            "same_frontier_churn_demoted": summary.get("same_frontier_churn_demoted"),
            "decision_evidence": object_field(decision, "evidence"),
        },
    }


def closed_loop_ratchet_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("closed_loop_residual_ratchet")
    if not isinstance(report, dict) or not report:
        return {"present": False}
    summary = object_field(report, "summary")
    decision = object_field(report, "decision")
    return {
        "present": True,
        "path": "reports/closed_loop_residual_ratchet.json",
        "trigger_state": report.get("trigger_state"),
        "decision": summary.get("decision") or decision.get("kind"),
        "decision_reason": summary.get("decision_reason") or decision.get("reason"),
        "broad_public_pass_rate": summary.get("broad_public_pass_rate"),
        "public_floor": summary.get("public_floor"),
        "dominant_residuals": summary.get("dominant_residuals"),
        "repair_item_count": summary.get("repair_item_count"),
        "private_repair_pressure_rows": summary.get("private_repair_pressure_rows"),
        "same_seed_private_semantic_lift": summary.get("same_seed_private_semantic_lift"),
        "same_frontier_churn_demoted": summary.get("same_frontier_churn_demoted"),
    }


def recommended_actions(walls: list[dict[str, Any]], next_action: dict[str, Any]) -> list[str]:
    actions = [str(next_action["label"])]
    for row in walls:
        if row.get("status") != "cleared":
            actions.append(str(row.get("next_action")))
    seen = set()
    deduped = []
    for action in actions:
        if action and action not in seen:
            seen.add(action)
            deduped.append(action)
    return deduped[:10]


def routing_hints(walls: list[dict[str, Any]], gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    weights = {
        "fresh_private_code_closure": 1.0,
        "decoder_v2_private_ablation_gate": 1.0,
        "one_public_calibration": 0.0,
        "teacher_architecture_experiment": 1.0,
        "coherence_repair": 1.0,
        "autonomy_reliability": 1.0,
        "frontier_non_code_lanes": 1.0,
        "architecture_decomposition": 1.0,
        "runtime_bottleneck_optimization": 1.0,
        "attd_modularization": 1.0,
    }
    iteration_wall = next((row for row in walls if row.get("id") == "iteration_speed_and_assembly_debt"), {})
    iteration_loop_bottlenecks = int(number(get_path(iteration_wall, ["evidence", "loop_bottleneck_count"], 0)) or 0)
    if iteration_wall.get("status") == "blocked":
        weights["runtime_bottleneck_optimization"] = 10.0
        weights["attd_modularization"] = 10.0
        weights["fresh_private_code_closure"] = 0.2
        weights["one_public_calibration"] = 0.0
    if gates["private_candidate_coverage_green"]["passed"] and not gates["fresh_private_closure_after_decoder_patch"]["passed"]:
        weights["fresh_private_code_closure"] = max(weights["fresh_private_code_closure"], 10.0 if iteration_wall.get("status") != "blocked" else 0.2)
    if gates["fresh_private_closure_after_decoder_patch"]["passed"] and not gates["public_receiver_candidate_coverage_green"]["passed"]:
        weights["decoder_v2_private_ablation_gate"] = 9.0
    operator_lock_active = bool(gates["public_calibration_allowed"]["evidence"].get("operator_lock_active"))
    if gates["public_calibration_allowed"]["passed"] and not operator_lock_active:
        weights["one_public_calibration"] = 7.0
    if operator_lock_active:
        weights["one_public_calibration"] = 0.0
        if iteration_loop_bottlenecks > 0:
            weights["runtime_bottleneck_optimization"] = max(weights["runtime_bottleneck_optimization"], 8.0)
        weights["architecture_decomposition"] = max(weights["architecture_decomposition"], 5.0)
    if not gates["architecture_experiment_delta_evidence"]["passed"]:
        weights["teacher_architecture_experiment"] = 6.0
    if not gates["sts_causal_decoder_evidence"]["passed"]:
        weights["decoder_v2_private_ablation_gate"] = max(weights["decoder_v2_private_ablation_gate"], 5.0)
    if not gates["agent_lane_transfer_evidence"]["passed"]:
        weights["frontier_non_code_lanes"] = 5.0
    if not gates["candidate_promotion_allowed"]["passed"]:
        weights["coherence_repair"] = 5.0
    if any(row.get("id") == "autonomy_noisy" and row.get("status") != "cleared" for row in walls):
        weights["autonomy_reliability"] = 4.0
    return {
        "action_kind_weights": weights,
        "public_calibration_locked": operator_lock_active or not gates["public_calibration_allowed"]["passed"],
        "public_calibration_operator_locked": operator_lock_active,
        "model_growth_locked": not gates["model_growth_allowed"]["passed"],
        "promotion_locked": not gates["candidate_promotion_allowed"]["passed"],
    }


def floor_recovery_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("broad_floor_recovery")
    if not isinstance(report, dict):
        report = {}
    summary = object_field(report, "summary")
    return {
        "path": "reports/broad_public_code_transfer_floor_recovery.json",
        "trigger_state": report.get("trigger_state"),
        "status": report.get("status"),
        "remaining_gap_explained": bool(summary.get("remaining_gap_explained")),
        "public_transfer_floor_cleared": bool(summary.get("public_transfer_floor_cleared")),
        "weak_cards": summary.get("weak_cards") if isinstance(summary.get("weak_cards"), list) else [],
        "dominant_residual_families": summary.get("dominant_residual_families"),
        "fresh_calibration_residual_families": summary.get("fresh_calibration_residual_families"),
        "next_blocker": summary.get("next_blocker"),
        "private_pressure_row_count": summary.get("private_pressure_row_count"),
        "same_seed_private_semantic_lift": summary.get("same_seed_private_semantic_lift"),
        "same_seed_ablation_status": summary.get("same_seed_ablation_status"),
        "public_tests_used": summary.get("public_tests_used"),
        "public_solutions_used": summary.get("public_solutions_used"),
        "score_semantics": "private-only residual recovery evidence; not public calibration or promotion evidence",
    }


def current_private_transfer_contract_evidence(state: dict[str, Any]) -> dict[str, Any]:
    """Return the current private transfer proof without spending public calibration.

    The legacy transfer proof remains valid when present, but the active
    practical lane now proves the bridge through replayable private candidates,
    full-body contract recovery, and the private repair/runtime readiness
    packet. This evidence is a technical prerequisite only; it does not unlock
    public calibration while the operator lock is active.
    """

    legacy = state.get("transfer_proof") if isinstance(state.get("transfer_proof"), dict) else {}
    legacy_summary = object_field(legacy, "summary")
    legacy_ok = bool(
        legacy.get("trigger_state") == "GREEN"
        and (legacy.get("ready_for_public_calibration") is True or legacy_summary.get("ready_for_public_calibration") is True)
    )

    replay = state.get("candidate_replay_contract") if isinstance(state.get("candidate_replay_contract"), dict) else {}
    replay_summary = object_field(replay, "summary")
    replay_ok = bool(
        replay.get("trigger_state") == "GREEN"
        and float(number(replay_summary.get("selected_intended_behavior_pass_rate")) or 0.0) >= 1.0
        and int(number(replay_summary.get("unexplained_no_candidate_count")) or 0) == 0
        and int(number(replay_summary.get("fallback_return_candidate_count")) or 0) == 0
    )

    recovery = (
        state.get("full_body_contract_transfer_recovery")
        if isinstance(state.get("full_body_contract_transfer_recovery"), dict)
        else {}
    )
    recovery_summary = object_field(recovery, "summary")
    required_rows = int(number(recovery_summary.get("required_readiness_eval_rows")) or 0)
    private_rows = int(number(recovery_summary.get("private_eval_rows")) or 0)
    selected_pass = float(number(recovery_summary.get("full_contract_selected_pass_rate")) or 0.0)
    selected_lift = float(number(recovery_summary.get("selected_pass_delta_full_minus_minimal")) or 0.0)
    recovery_ok = bool(
        recovery.get("trigger_state") == "GREEN"
        and recovery_summary.get("ready_for_future_governed_public_calibration") is True
        and selected_pass >= 0.95
        and selected_lift > 0.0
        and (required_rows == 0 or private_rows >= required_rows)
        and int(number(recovery_summary.get("fallback_return_count")) or 0) == 0
        and int(number(recovery_summary.get("template_like_candidate_count")) or 0) == 0
        and int(number(recovery_summary.get("public_training_rows")) or 0) == 0
        and int(number(recovery_summary.get("external_inference_calls")) or 0) == 0
    )

    readiness = (
        state.get("private_full_body_repair_runtime_readiness")
        if isinstance(state.get("private_full_body_repair_runtime_readiness"), dict)
        else {}
    )
    readiness_summary = object_field(readiness, "summary")
    readiness_ok = bool(
        readiness.get("trigger_state") == "GREEN"
        and int(number(readiness_summary.get("hard_failure_count")) or 0) == 0
        and readiness_summary.get("no_public_calibration_run") is True
        and int(number(readiness_summary.get("public_training_rows_written")) or 0) == 0
        and int(number(readiness_summary.get("external_inference_calls")) or 0) == 0
        and int(number(readiness_summary.get("fallback_return_count")) or 0) == 0
    )

    current_ok = bool(replay_ok and recovery_ok and readiness_ok)
    return {
        "passed": bool(legacy_ok or current_ok),
        "source": "legacy_private_public_transfer_proof" if legacy_ok else "current_replay_full_body_repair_readiness",
        "legacy_private_public_transfer_proof": {
            "trigger_state": legacy.get("trigger_state"),
            "ready_for_public_calibration": legacy.get("ready_for_public_calibration"),
            "summary_ready_for_public_calibration": legacy_summary.get("ready_for_public_calibration"),
            "passed": legacy_ok,
        },
        "current_structural_full_body_contract": {
            "passed": current_ok,
            "candidate_replay_green": replay_ok,
            "full_body_recovery_green": recovery_ok,
            "private_runtime_readiness_green": readiness_ok,
            "candidate_replay": {
                "path": "reports/private_candidate_replay_contract_audit_v1.json",
                "trigger_state": replay.get("trigger_state"),
                "task_count": replay_summary.get("task_count"),
                "candidate_row_count": replay_summary.get("candidate_row_count"),
                "selected_intended_behavior_pass_rate": replay_summary.get("selected_intended_behavior_pass_rate"),
                "pass_if_any_rate": replay_summary.get("pass_if_any_rate"),
                "unexplained_no_candidate_count": replay_summary.get("unexplained_no_candidate_count"),
                "fallback_return_candidate_count": replay_summary.get("fallback_return_candidate_count"),
            },
            "full_body_recovery": {
                "path": "reports/full_body_contract_transfer_recovery_v1.json",
                "trigger_state": recovery.get("trigger_state"),
                "private_eval_rows": recovery_summary.get("private_eval_rows"),
                "required_readiness_eval_rows": recovery_summary.get("required_readiness_eval_rows"),
                "full_contract_selected_pass_rate": recovery_summary.get("full_contract_selected_pass_rate"),
                "minimal_contract_selected_pass_rate": recovery_summary.get("minimal_contract_selected_pass_rate"),
                "selected_pass_delta_full_minus_minimal": recovery_summary.get("selected_pass_delta_full_minus_minimal"),
                "readiness_decision": recovery_summary.get("readiness_decision"),
                "public_training_rows": recovery_summary.get("public_training_rows"),
                "fallback_return_count": recovery_summary.get("fallback_return_count"),
            },
            "private_repair_runtime_readiness": {
                "path": "reports/private_full_body_repair_runtime_readiness_v1.json",
                "trigger_state": readiness.get("trigger_state"),
                "hard_failure_count": readiness_summary.get("hard_failure_count"),
                "no_public_calibration_run": readiness_summary.get("no_public_calibration_run"),
                "public_training_rows_written": readiness_summary.get("public_training_rows_written"),
                "external_inference_calls": readiness_summary.get("external_inference_calls"),
                "fallback_return_count": readiness_summary.get("fallback_return_count"),
                "recommendation": readiness_summary.get("recommendation"),
            },
        },
        "public_calibration_spent": False,
        "score_semantics": "private-only replay and repair readiness; not a public benchmark score or public calibration unlock",
    }


def edge_full_body_bridge_v2_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("edge_full_body_bridge_v2")
    if not isinstance(report, dict):
        report = {}
    baseline = state.get("edge_full_body_bridge_v2_baseline")
    if not isinstance(baseline, dict):
        baseline = {}
    summary = object_field(report, "summary")
    baseline_summary = object_field(baseline, "summary")
    edge_rate = first_number(summary.get("edge_obligation_ok"), summary.get("private_pass_rate")) or 0.0
    body_rate = first_number(summary.get("body_exec_pass_rate"), summary.get("body_exec_ok")) or 0.0
    same_seed_edge_baseline = first_number(
        baseline_summary.get("edge_obligation_ok"),
        baseline_summary.get("private_pass_rate"),
    )
    same_seed_body_baseline = first_number(
        baseline_summary.get("body_exec_pass_rate"),
        baseline_summary.get("body_exec_ok"),
    )
    leakage_count = int(number(summary.get("leakage_violation_count")) or 0)
    template_count = int(number(summary.get("template_like_candidate_count")) or 0)
    wrapper_count = int(number(summary.get("wrapper_like_candidate_count")) or 0)
    candidate_count = int(number(summary.get("candidate_count")) or 0)
    heldout_count = int(number(summary.get("heldout_private_task_count")) or 0)
    token_level_valid = first_number(summary.get("token_level_student_generation_valid")) or 0.0
    passed = bool(
        report.get("trigger_state") == "GREEN"
        and report.get("ready_for_public_calibration") is True
        and candidate_count > 0
        and heldout_count >= 64
        and float(token_level_valid) > 0.0
        and float(edge_rate) > EDGE_OBLIGATION_V1_BASELINE
        and float(body_rate) > BODY_EXEC_V1_BASELINE
        and leakage_count == 0
        and template_count == 0
        and wrapper_count == 0
    )
    return {
        "passed": passed,
        "path": "reports/edge_full_body_contract_bridge_v2_private_edge_obligation_gate.json",
        "baseline_path": "reports/edge_full_body_contract_bridge_v2_private_edge_obligation_gate_baseline.json",
        "trigger_state": report.get("trigger_state"),
        "ready_for_public_calibration": report.get("ready_for_public_calibration"),
        "edge_obligation_ok": edge_rate,
        "edge_obligation_v1_baseline": EDGE_OBLIGATION_V1_BASELINE,
        "edge_obligation_lift_vs_v1_baseline": round(float(edge_rate) - EDGE_OBLIGATION_V1_BASELINE, 6),
        "body_exec_pass_rate": body_rate,
        "body_exec_v1_baseline": BODY_EXEC_V1_BASELINE,
        "body_exec_lift_vs_v1_baseline": round(float(body_rate) - BODY_EXEC_V1_BASELINE, 6),
        "same_seed_baseline_edge_obligation_ok": same_seed_edge_baseline,
        "same_seed_baseline_body_exec_pass_rate": same_seed_body_baseline,
        "same_seed_edge_obligation_lift": (
            round(float(edge_rate) - float(same_seed_edge_baseline), 6)
            if same_seed_edge_baseline is not None
            else None
        ),
        "same_seed_body_exec_lift": (
            round(float(body_rate) - float(same_seed_body_baseline), 6)
            if same_seed_body_baseline is not None
            else None
        ),
        "candidate_count": candidate_count,
        "heldout_private_task_count": heldout_count,
        "token_level_student_generation_valid": token_level_valid,
        "return_shape_ok": summary.get("return_shape_ok"),
        "type_admissible_ok": summary.get("type_admissible_ok"),
        "branch_loop_obligation_ok": summary.get("branch_loop_obligation_ok"),
        "leakage_violation_count": leakage_count,
        "template_like_candidate_count": template_count,
        "wrapper_like_candidate_count": wrapper_count,
        "score_semantics": (
            "private-only edge/full-body bridge gate; permits only a later operator-approved bounded public "
            "calibration proposal, not public training, model growth, or promotion"
        ),
    }


def edge_full_body_bridge_v2_public_verdict_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("edge_full_body_bridge_v2_public_verdict")
    if not isinstance(report, dict):
        report = {}
    summary = object_field(report, "summary")
    residuals = report.get("residuals") if isinstance(report.get("residuals"), list) else []
    residual_counts: dict[str, int] = {}
    card_family_counts: dict[str, dict[str, int]] = {}
    for row in residuals:
        if not isinstance(row, dict):
            continue
        family = str(row.get("type") or "unknown")
        card = str(row.get("card_id") or "unknown")
        residual_counts[family] = residual_counts.get(family, 0) + 1
        card_counts = card_family_counts.setdefault(card, {})
        card_counts[family] = card_counts.get(family, 0) + 1
    pass_rate = first_number(summary.get("real_public_task_pass_rate"), report.get("score")) or 0.0
    template_count = int(number(summary.get("template_like_candidate_count")) or 0)
    loop_count = int(number(summary.get("loop_closure_candidate_count")) or 0)
    external_calls = int(number(report.get("external_inference_calls")) or 0)
    integrity_valid = bool(summary.get("student_candidate_benchmark_integrity_valid"))
    present = bool(report.get("policy") == "project_theseus_real_code_benchmark_graduation_v1")
    clean = bool(
        present
        and report.get("trigger_state") == "GREEN"
        and template_count == 0
        and loop_count == 0
        and external_calls == 0
        and integrity_valid
    )
    cleared_floor = bool(clean and float(pass_rate) >= PUBLIC_CODE_FLOOR)
    return {
        "present": present,
        "path": "reports/real_code_benchmark_graduation_edge_full_body_contract_bridge_v2_public_verdict.json",
        "created_utc": report.get("created_utc"),
        "trigger_state": report.get("trigger_state"),
        "pass_rate": pass_rate,
        "floor": PUBLIC_CODE_FLOOR,
        "cleared_floor": cleared_floor,
        "spent_failed": bool(clean and not cleared_floor),
        "clean_integrity": clean,
        "template_like_candidate_count": template_count,
        "loop_closure_candidate_count": loop_count,
        "external_inference_calls": external_calls,
        "student_candidate_benchmark_integrity_valid": integrity_valid,
        "public_task_count": int(number(summary.get("public_task_count")) or 0),
        "residual_count": len(residuals),
        "residual_family_counts": dict(sorted(residual_counts.items(), key=lambda item: (-item[1], item[0]))),
        "card_residual_family_counts": {
            card: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            for card, counts in sorted(card_family_counts.items())
        },
        "score_semantics": (
            "single spent operator-authorized calibration-only public verdict; failed result must route to "
            "private-only architecture work and does not unlock another public calibration"
        ),
    }


def broad_floor_recovery_ready_for_fresh_closure(state: dict[str, Any]) -> bool:
    report = state.get("broad_floor_recovery")
    if not isinstance(report, dict) or not report:
        return False
    summary = object_field(report, "summary")
    private_rows = int(number(summary.get("private_pressure_row_count")) or 0)
    rows_with_tests = int(number(summary.get("private_pressure_rows_with_behavior_tests")) or 0)
    semantic_lift = first_number(summary.get("same_seed_private_semantic_lift")) or 0.0
    recovery_mtime = path_mtime(REPORTS / "broad_public_code_transfer_floor_recovery.json")
    current_train_once_mtime = path_mtime(
        REPORTS / "code_lm_closure_private_pressure_private_recovery_train_once_fanout_v1.json"
    )
    latest_consuming_closure_mtime = max(
        current_train_once_mtime,
        latest_broad_floor_semantic_closure_mtime(),
    )
    return bool(
        report.get("trigger_state") == "GREEN"
        and summary.get("remaining_gap_explained") is True
        and summary.get("public_calibration_run") is False
        and summary.get("public_tests_used") is False
        and summary.get("public_solutions_used") is False
        and private_rows > 0
        and rows_with_tests >= private_rows
        and float(semantic_lift) >= 0.03
        and recovery_mtime > latest_consuming_closure_mtime
    )


def broad_floor_semantic_closure_versions() -> list[tuple[int, Path, float]]:
    prefix = "code_lm_closure_private_pressure_private_recovery_broad_floor"
    versions: list[tuple[int, Path, float]] = []
    for path in REPORTS.glob(f"{prefix}*_v*.json"):
        try:
            version = int(path.stem.rsplit("_v", 1)[1])
        except (IndexError, ValueError):
            continue
        versions.append((version, path, path_mtime(path)))
    return versions


def latest_broad_floor_semantic_closure_mtime() -> float:
    versions = broad_floor_semantic_closure_versions()
    return max((mtime for _, _, mtime in versions), default=0.0)


def latest_broad_floor_semantic_checkpoint_path() -> Path:
    prefix = "student_code_lm_checkpoint_private_pressure_private_recovery_broad_floor"
    versions: list[tuple[int, Path, float]] = []
    for path in REPORTS.glob(f"{prefix}*_v*.json"):
        try:
            version = int(path.stem.rsplit("_v", 1)[1])
        except (IndexError, ValueError):
            continue
        versions.append((version, path, path_mtime(path)))
    if versions:
        return resolve_archived_path(max(versions, key=lambda row: (row[0], row[2]))[1])
    return resolve_archived_path(
        REPORTS / "student_code_lm_checkpoint_private_pressure_private_recovery_train_once_fanout_v1.json"
    )


def command_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def next_broad_floor_semantic_slug() -> str:
    versions = broad_floor_semantic_closure_versions()
    next_version = max((version for version, _, _ in versions), default=10) + 1
    return f"private_pressure_private_recovery_broad_floor_semantic_v{next_version}"


def broad_residual_reader_next_action(state: dict[str, Any]) -> dict[str, Any]:
    reader = state.get("broad_residual_reader") if isinstance(state.get("broad_residual_reader"), dict) else {}
    summary = object_field(reader, "summary")
    recommendation = object_field(reader, "recommendation")
    if (
        reader.get("trigger_state") != "GREEN"
        or summary.get("wall_type") != "model_quality_wall"
        or summary.get("dominant_residual") not in {"edge_case", "edge_contract"}
    ):
        return {}
    v10_outcome = broad_edge_v10_outcome(state)
    if v10_outcome.get("attempted") and not v10_outcome.get("ready_for_public_calibration"):
        post_v10_ablation = broad_post_v10_ablation_outcome(state)
        if post_v10_ablation.get("attempted"):
            scale_ablation = broad_edge_type_scale64_ablation_outcome(state)
            if scale_ablation.get("validated_private_semantic_lift"):
                if broad_floor_recovery_ready_for_fresh_closure(state):
                    semantic_slug = next_broad_floor_semantic_slug()
                    semantic_report_suffix = semantic_slug.replace("private_pressure_private_recovery_", "")
                    checkpoint_path = latest_broad_floor_semantic_checkpoint_path()
                    return {
                        "label": (
                            "Run one fresh private-only train-once CUDA closure that consumes the new broad-floor "
                            "private pressure rows, then rerun decoder_v2_private_ablation_gate and "
                            "private_public_transfer_proof. Keep public calibration locked."
                        ),
                        "command": [
                            "python",
                            "scripts/code_lm_train_once_fanout.py",
                            "--execute",
                            "--slug",
                            semantic_slug,
                            "--public-cards",
                            "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                            "--max-public-cases-per-card",
                            "32",
                            "--private-count",
                            "480",
                            "--epochs",
                            "4",
                            "--candidates-per-task",
                            "8",
                            "--max-high-transfer-private-train",
                            "6400",
                            "--max-rust-work-steps",
                            "4000000",
                            "--rust-timeout-seconds",
                            "7200",
                            "--fanout-timeout-seconds",
                            "21600",
                            "--out",
                            f"reports/code_lm_train_once_fanout_{semantic_report_suffix}.json",
                            "--markdown-out",
                            f"reports/code_lm_train_once_fanout_{semantic_report_suffix}.md",
                        ],
                        "evidence": {
                            "v10_outcome": v10_outcome,
                            "post_v10_ablation": post_v10_ablation,
                            "scale_ablation": scale_ablation,
                            "floor_recovery": floor_recovery_evidence(state),
                            "public_calibration": "locked",
                            "score_semantics": (
                                "private train-once closure over clean broad-floor pressure rows; public tasks are "
                                "candidate fanout metadata only, not public tests or solutions"
                            ),
                        },
                    }
                return {
                    "label": (
                        "Edge/type residual-router patch has scaled private semantic lift and downstream transfer "
                        "gates are GREEN; stop looping on recovery-row generation and run a private-only "
                        "same-seed residual-family ablation against the current checkpoint, especially edge "
                        "contract, local adapter, and external dependency handling."
                    ),
                    "command": [
                        "python",
                        "scripts/broad_transfer_residual_decoder_ablation.py",
                        "--task-limit",
                        "96",
                        "--candidates-per-task",
                        "8",
                        "--checkpoint",
                        command_path(latest_broad_floor_semantic_checkpoint_path()),
                        "--out",
                        "reports/broad_transfer_residual_decoder_ablation_next_residual_families.json",
                        "--markdown-out",
                        "reports/broad_transfer_residual_decoder_ablation_next_residual_families.md",
                    ],
                    "evidence": {
                        "v10_outcome": v10_outcome,
                        "post_v10_ablation": post_v10_ablation,
                        "scale_ablation": scale_ablation,
                        "latest_broad_floor_checkpoint": command_path(
                            latest_broad_floor_semantic_checkpoint_path()
                        ),
                        "public_calibration": "locked",
                        "score_semantics": (
                            "private-only same-seed residual-family ablation after the consumed source patch; "
                            "no public tests, public solutions, or copied public bodies"
                        ),
                    },
                }
            return {
                "label": (
                    "Patch the residual decoder router for the remaining private edge-case and type-handling "
                    "no-admissible/semantic failures; then rerun an expanded private-only same-seed ablation."
                ),
                "command": [
                    "python",
                    "scripts/broad_transfer_residual_decoder_ablation.py",
                    "--task-limit",
                    "64",
                    "--candidates-per-task",
                    "6",
                    "--checkpoint",
                    command_path(latest_broad_floor_semantic_checkpoint_path()),
                    "--out",
                    "reports/broad_transfer_residual_decoder_ablation_edge_type_scale64.json",
                    "--markdown-out",
                    "reports/broad_transfer_residual_decoder_ablation_edge_type_scale64.md",
                ],
                "evidence": {
                    "v10_outcome": v10_outcome,
                    "post_v10_ablation": post_v10_ablation,
                    "public_calibration": "locked",
                    "score_semantics": "expanded private same-seed A/B after source patch; no public tests or solutions",
                },
            }
        return {
            "label": (
                "Demote the completed v10 edge-contract closure as diagnostic-only; run a private-only same-seed "
                "adapter/runtime/interface residual decoder ablation before any more broad-floor training."
            ),
            "command": [
                "python",
                "scripts/broad_transfer_residual_decoder_ablation.py",
                "--task-limit",
                "16",
                "--candidates-per-task",
                "4",
                "--checkpoint",
                command_path(latest_broad_floor_semantic_checkpoint_path()),
                "--out",
                "reports/broad_transfer_residual_decoder_ablation_after_v10_demote.json",
                "--markdown-out",
                "reports/broad_transfer_residual_decoder_ablation_after_v10_demote.md",
            ],
            "evidence": {
                "v10_outcome": v10_outcome,
                "next_blocker": floor_recovery_evidence(state).get("next_blocker"),
                "public_calibration": "locked",
                "score_semantics": "private same-seed A/B only; no public tests, public solutions, or template bodies",
            },
        }
    return {
        "label": (
            "Run a private-only CUDA train-once edge/intended-behavior closure from the broad-floor recovery rows; "
            "do not public-calibrate."
        ),
        "command": [
            "python",
            "scripts/code_lm_train_once_fanout.py",
            "--execute",
            "--slug",
            "private_pressure_private_recovery_broad_floor_edge_contract_v10",
            "--public-cards",
            "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
            "--max-public-cases-per-card",
            "32",
            "--private-count",
            "480",
            "--epochs",
            "4",
            "--candidates-per-task",
            "8",
            "--max-high-transfer-private-train",
            "6400",
            "--max-rust-work-steps",
            "4000000",
            "--rust-timeout-seconds",
            "7200",
            "--fanout-timeout-seconds",
            "21600",
            "--out",
            "reports/code_lm_train_once_fanout_broad_floor_edge_contract_v10.json",
            "--markdown-out",
            "reports/code_lm_train_once_fanout_broad_floor_edge_contract_v10.md",
        ],
        "evidence": {
            "path": "reports/broad_transfer_residual_reader.json",
            "dominant_residual": summary.get("dominant_residual"),
            "dominant_count": summary.get("dominant_count"),
            "broad_public_pass_rate": summary.get("broad_public_pass_rate"),
            "public_floor": summary.get("public_floor"),
            "private_pressure_row_count": summary.get("private_pressure_row_count"),
            "same_seed_private_semantic_lift": summary.get("same_seed_private_semantic_lift"),
            "decoder_patch": recommendation.get("decoder_patch"),
            "public_calibration": "locked",
        },
    }


def broad_edge_type_scale64_ablation_outcome(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("broad_edge_type_scale64_ablation")
    if not isinstance(report, dict) or not report:
        return {"attempted": False}
    delta = object_field(report, "delta")
    patched = object_field(report, "patched")
    patched_summary = object_field(patched, "summary")
    semantic_family_deltas = object_field(delta, "semantic_task_family_deltas")
    gates = report.get("gates") if isinstance(report.get("gates"), list) else []
    gate_map = {
        str(row.get("name")): bool(row.get("passed"))
        for row in gates
        if isinstance(row, dict) and row.get("name")
    }
    semantic_lift = first_number(delta.get("semantic_test_passed_task_rate_delta")) or 0.0
    no_admissible_count = int(number(patched_summary.get("no_admissible_task_count")) or 0)
    target_family_deltas: dict[str, float] = {}
    for family in ("algorithm_choice", "edge_case", "local_code_generation_adapter_needed", "type_handling"):
        family_semantic = object_field(semantic_family_deltas, family)
        target_family_deltas[family] = float(
            first_number(family_semantic.get("semantic_passed_task_rate_delta")) or 0.0
        )
    edge_type_lift = bool(target_family_deltas.get("edge_case", 0.0) > 0.0 and target_family_deltas.get("type_handling", 0.0) > 0.0)
    validated = bool(
        gate_map.get("private_only")
        and gate_map.get("no_public_candidates_emitted")
        and gate_map.get("private_semantic_correctness_lift")
        and gate_map.get("target_families_have_semantic_delta")
        and gate_map.get("patched_no_admissible_zero")
        and gate_map.get("no_fanout_speed_regression")
        and semantic_lift > 0.0
        and no_admissible_count == 0
        and edge_type_lift
    )
    return {
        "attempted": True,
        "trigger_state": report.get("trigger_state") or report.get("status"),
        "path": "reports/broad_transfer_residual_decoder_ablation_edge_type_scale64.json",
        "validated_private_semantic_lift": validated,
        "semantic_test_passed_task_rate_delta": semantic_lift,
        "patched_no_admissible_task_count": no_admissible_count,
        "target_family_semantic_deltas": target_family_deltas,
        "failed_gates": [name for name, passed in gate_map.items() if not passed],
        "public_candidates_emitted": not gate_map.get("no_public_candidates_emitted", False),
        "score_semantics": "private same-seed scale validation only; not public calibration or promotion evidence",
    }


def broad_post_v10_ablation_outcome(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("broad_post_v10_ablation")
    if not isinstance(report, dict) or not report:
        return {"attempted": False}
    delta = object_field(report, "delta")
    patched = object_field(report, "patched")
    patched_summary = object_field(patched, "summary")
    semantic_family_deltas = object_field(delta, "semantic_task_family_deltas")
    task_family_metrics = object_field(patched_summary, "task_family_metrics")
    no_admissible_count = int(number(patched_summary.get("no_admissible_task_count")) or 0)
    weak_families = []
    for family in ("edge_case", "type_handling", "local_code_generation_adapter_needed", "algorithm_choice"):
        family_semantic = object_field(semantic_family_deltas, family)
        family_metrics = object_field(task_family_metrics, family)
        semantic_delta = first_number(family_semantic.get("semantic_passed_task_rate_delta"))
        no_admissible_rate = first_number(family_metrics.get("no_admissible_rate")) or 0.0
        if (semantic_delta is not None and float(semantic_delta) <= 0.0) or float(no_admissible_rate) > 0.0:
            weak_families.append(
                {
                    "family": family,
                    "semantic_passed_task_rate_delta": semantic_delta,
                    "patched_no_admissible_rate": no_admissible_rate,
                    "patched_passed_task_rate": family_metrics.get("passed_task_rate"),
                    "patched_private_receiver_eligible_task_rate": family_metrics.get(
                        "private_receiver_eligible_task_rate"
                    ),
                }
            )
    return {
        "attempted": True,
        "trigger_state": report.get("trigger_state"),
        "path": "reports/broad_transfer_residual_decoder_ablation_after_v10_demote.json",
        "passed_task_rate_delta": delta.get("passed_task_rate_delta"),
        "private_receiver_eligible_task_rate_delta": delta.get("private_receiver_eligible_task_rate_delta"),
        "no_admissible_rate_delta": delta.get("no_admissible_rate_delta"),
        "semantic_test_passed_task_rate_delta": delta.get("semantic_test_passed_task_rate_delta"),
        "patched_no_admissible_task_count": no_admissible_count,
        "weak_families": weak_families,
        "ready_for_scale_validation": bool(
            report.get("trigger_state") == "GREEN" and no_admissible_count == 0 and not weak_families
        ),
    }


def broad_edge_v10_outcome(state: dict[str, Any]) -> dict[str, Any]:
    gate_report = state.get("broad_edge_v10_decoder_gate")
    proof_report = state.get("broad_edge_v10_transfer_proof")
    if not isinstance(gate_report, dict):
        gate_report = {}
    if not isinstance(proof_report, dict):
        proof_report = {}
    gate_summary = object_field(gate_report, "summary")
    proof_current = object_field(proof_report, "current")
    proof_deltas = object_field(proof_report, "deltas")
    canonical_summary = object_field(
        state.get("decoder_gate") if isinstance(state.get("decoder_gate"), dict) else {},
        "summary",
    )
    attempted = bool(gate_report or proof_report)
    ready = bool(
        gate_report.get("ready_for_public_calibration") is True
        and proof_report.get("ready_for_public_calibration") is True
    )
    metric_deltas = decoder_metric_deltas(
        canonical_summary,
        gate_summary,
        [
            "public_actual_token_task_coverage",
            "public_eligible_task_coverage",
            "public_no_admissible_task_rate",
            "public_program_synthesis_promotion_ready_rate",
            "public_candidate_count",
            "contract_guided_candidate_count",
            "sts_conditioned_candidate_count",
        ],
    )
    regressed_vs_canonical = bool(
        (metric_deltas.get("public_actual_token_task_coverage_delta") or 0.0) < 0.0
        or (metric_deltas.get("public_eligible_task_coverage_delta") or 0.0) < 0.0
        or (metric_deltas.get("public_no_admissible_task_rate_delta") or 0.0) > 0.0
        or (metric_deltas.get("public_program_synthesis_promotion_ready_rate_delta") or 0.0) < 0.0
    )
    return {
        "attempted": attempted,
        "trigger_state": gate_report.get("trigger_state"),
        "transfer_trigger_state": proof_report.get("trigger_state"),
        "ready_for_public_calibration": ready,
        "decoder_gate_ready": bool(gate_report.get("ready_for_public_calibration")),
        "transfer_proof_ready": bool(proof_report.get("ready_for_public_calibration")),
        "latest_closure": gate_summary.get("latest_closure") or proof_current.get("latest_closure"),
        "v10_public_actual_token_task_coverage": gate_summary.get("public_actual_token_task_coverage"),
        "v10_public_eligible_task_coverage": gate_summary.get("public_eligible_task_coverage"),
        "v10_public_no_admissible_task_rate": gate_summary.get("public_no_admissible_task_rate"),
        "v10_public_program_synthesis_promotion_ready_rate": gate_summary.get(
            "public_program_synthesis_promotion_ready_rate"
        ),
        "v10_public_no_admissible_top_reasons": gate_summary.get("public_no_admissible_top_reasons"),
        "metric_deltas_vs_current_canonical_decoder_gate": metric_deltas,
        "regressed_vs_current_canonical_decoder_gate": regressed_vs_canonical,
        "failed_transfer_gates": sorted(failed_gate_names(proof_report)),
        "proof_deltas_vs_historical_baseline": proof_deltas,
    }


def decoder_metric_deltas(
    baseline_summary: dict[str, Any], current_summary: dict[str, Any], keys: list[str]
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in keys:
        baseline = first_number(baseline_summary.get(key))
        current = first_number(current_summary.get(key))
        if baseline is None or current is None:
            continue
        deltas[f"{key}_delta"] = round(float(current) - float(baseline), 6)
    return deltas


def wall(wall_id: str, title: str, status: str, severity: str, evidence: Any, next_action: str) -> dict[str, Any]:
    return {
        "id": wall_id,
        "title": title,
        "status": status,
        "severity": severity,
        "evidence": evidence,
        "next_action": next_action,
    }


def gate(passed: bool, evidence: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "evidence": evidence}


def failed_gate_names(report: dict[str, Any]) -> set[str]:
    rows = report.get("gates") if isinstance(report, dict) else []
    names: set[str] = set()
    if not isinstance(rows, list):
        return names
    for row in rows:
        if not isinstance(row, dict) or row.get("passed", True):
            continue
        name = row.get("name") or row.get("gate")
        if name:
            names.add(str(name))
    return names


def scheduler_command(state: dict[str, Any], concept: str) -> list[str]:
    tasks = state.get("scheduler", {}).get("tasks") if isinstance(state.get("scheduler"), dict) else []
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
            if payload.get("concept") == concept and isinstance(task.get("command"), list):
                return [str(part) for part in task["command"]]
    return []


def architecture_delta_evidence(report: dict[str, Any]) -> dict[str, Any]:
    deltas = []
    for key in ("summary", "delta_evidence", "promotion_decision", "results"):
        value = report.get(key)
        collect_numeric_deltas(value, deltas)
    best = max((abs(row["value"]) for row in deltas), default=0.0)
    return {
        "status": report.get("status"),
        "best_abs_delta": round(best, 6),
        "delta_threshold": SIGNIFICANT_ARCH_DELTA,
        "has_significant_delta": best >= SIGNIFICANT_ARCH_DELTA,
        "sample_deltas": deltas[:8],
    }


def collect_numeric_deltas(value: Any, out: list[dict[str, Any]], prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "delta" in str(key).lower() or "lift" in str(key).lower() or "improv" in str(key).lower():
                number_value = first_number(child)
                if number_value is not None:
                    out.append({"path": path, "value": float(number_value)})
            collect_numeric_deltas(child, out, path)
    elif isinstance(value, list):
        for idx, child in enumerate(value[:80]):
            collect_numeric_deltas(child, out, f"{prefix}[{idx}]")


def report_causality_ok(state: dict[str, Any]) -> bool:
    causal = get_path(state["cross_domain"], ["summary", "causal_transfer"], {})
    sts_causal = sts_causal_evidence(state)
    return bool(
        isinstance(causal, dict)
        and causal.get("measured_transfer_effect")
        and float(number(causal.get("sts_pass_rate_delta")) or 0.0) > 0.0
        and sts_causal.get("same_seed_non_sts_comparator_present") is True
        and state["sts_causal_decoder_ablation"].get("trigger_state") == "GREEN"
        and get_path(state["symliquid"], ["summary", "strongest_action_kind"], "")
    )


def report_causality_evidence(state: dict[str, Any]) -> dict[str, Any]:
    causal = get_path(state["cross_domain"], ["summary", "causal_transfer"], {})
    sts_causal = sts_causal_evidence(state)
    return {
        "report_store_runs": get_path(state["report_store"], ["summary", "stored_run_count"], None),
        "cross_domain_measured_transfer_effect": causal.get("measured_transfer_effect") if isinstance(causal, dict) else None,
        "sts_pass_rate_delta": causal.get("sts_pass_rate_delta") if isinstance(causal, dict) else None,
        "sts_causal_decoder_ablation": sts_causal,
        "named_consumers": get_path(state["cross_domain"], ["summary", "transfer_target_counts"], {}),
        "symliquid_strongest_action_kind": get_path(state["symliquid"], ["summary", "strongest_action_kind"], ""),
    }


def sts_causal_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state["sts_causal_decoder_ablation"]
    summary = object_field(report, "summary")
    return {
        "trigger_state": report.get("trigger_state"),
        "same_seed_non_sts_comparator_present": summary.get("same_seed_non_sts_comparator_present"),
        "sts_public_eligible_coverage_delta": summary.get("sts_public_eligible_coverage_delta"),
        "sts_public_pass_rate_delta": summary.get("sts_public_pass_rate_delta"),
        "sts_contract_public_task_coverage": summary.get("sts_contract_public_task_coverage"),
        "decoder_public_no_admissible_task_rate": summary.get("decoder_public_no_admissible_task_rate"),
        "ready_for_architecture_promotion": report.get("ready_for_architecture_promotion"),
    }


def model_growth_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state["model_growth"]
    return {
        "model_growth_allowed": report.get("model_growth_allowed"),
        "hard_blockers": report.get("hard_blockers"),
        "missing_evidence": report.get("missing_evidence"),
        "next_action": report.get("next_action"),
    }


def promotion_evidence(state: dict[str, Any]) -> dict[str, Any]:
    coherence_gate = state["coherence_gate"]
    candidate = state["candidate_gate"]
    failed_from_checks = [
        row.get("gate")
        for row in candidate.get("checks", [])
        if isinstance(row, dict) and not row.get("passed")
    ]
    return {
        "candidate_promote": candidate.get("promote"),
        "candidate_passed_count": candidate.get("passed"),
        "candidate_total_count": candidate.get("total"),
        "candidate_failed_gates": candidate.get("failed_gates") or failed_from_checks,
        "coherence_trigger_state": coherence_gate.get("trigger_state"),
        "coherence_source_trigger_state": coherence_gate.get("source_trigger_state"),
        "coherence_score": coherence_gate.get("coherence_score"),
        "delirium_score": coherence_gate.get("delirium_score"),
        "coherence_allows_candidate_promotion": coherence_gate.get("allows_candidate_promotion"),
        "coherence_candidate_blockers": coherence_gate.get("candidate_blockers", []),
    }


def non_code_frontier_ready(state: dict[str, Any]) -> bool:
    conversation_summary = object_field(state["conversation_v4"], "summary")
    puffer_summary = object_field(state["pufferlib"], "summary")
    return bool(
        conversation_summary.get("graduated")
        and int(number(get_path(state["tool_use"], ["summary", "case_count"], 0)) or 0) >= 64
        and int(number(get_path(state["board_policy"], ["summary", "policy_train_row_count"], 0)) or 0) >= 512
        and (puffer_summary.get("native_policy_learning_evidence") or puffer_summary.get("policy_train_row_count"))
    )


def latest_execution_shape_candidate_coverage_report() -> dict[str, Any]:
    candidates: list[tuple[tuple[int, float, float, float], dict[str, Any]]] = []
    for path in REPORTS.glob("execution_shape_candidate_coverage*.json"):
        if path.name.endswith("_rust.json") or path.name.endswith("_checkpoint.json"):
            continue
        report = read_json(path, {})
        if not isinstance(report, dict) or not report:
            continue
        report.setdefault("source_report_path", rel(path))
        report.setdefault("source_report_mtime", path_mtime(path))
        summary = object_field(report, "summary")
        trigger_bonus = 1 if report.get("trigger_state") == "GREEN" else 0
        pass_rate = first_number(summary.get("learned_token_decoder_pass_rate")) or 0.0
        no_admissible = first_number(summary.get("learned_token_decoder_no_admissible_candidate_rate"))
        no_admissible = 1.0 if no_admissible is None else no_admissible
        candidates.append(((trigger_bonus, pass_rate, -no_admissible, path_mtime(path)), report))
    if not candidates:
        smoke = read_json(REPORTS / "execution_shape_private_ablation_smoke.json", {})
        if isinstance(smoke, dict):
            smoke.setdefault("source_report_path", rel(REPORTS / "execution_shape_private_ablation_smoke.json"))
            smoke.setdefault("source_report_mtime", path_mtime(REPORTS / "execution_shape_private_ablation_smoke.json"))
        return smoke
    return max(candidates, key=lambda item: item[0])[1]


def complexity_snapshot() -> dict[str, Any]:
    files = [
        ROOT / "crates/symliquid-cli/src/code_lm_closure/part_00.rs",
        ROOT / "crates/symliquid-cli/src/code_lm_closure/part_01.rs",
        ROOT / "crates/symliquid-cli/src/code_lm_closure/part_02.rs",
        ROOT / "crates/symliquid-cli/src/code_lm_closure/part_03.rs",
        ROOT / "scripts/autonomy_cycle.py",
        ROOT / "scripts/hive_work_board_executor.py",
        ROOT / "scripts/vacation_mode_supervisor.py",
    ]
    rows = []
    for path in files:
        rows.append({"path": rel(path), "lines": count_lines(path)})
    max_lines = max((row["lines"] for row in rows), default=0)
    return {
        "max_source_lines": max_lines,
        "target_max_source_lines": 4500,
        "largest_files": sorted(rows, key=lambda row: row["lines"], reverse=True)[:6],
    }


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def report_mtime(report: dict[str, Any]) -> float:
    value = report.get("source_report_mtime")
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def path_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def object_field(value: Any, key: str) -> dict[str, Any]:
    field = value.get(key) if isinstance(value, dict) else {}
    return field if isinstance(field, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def first_number(*values: Any) -> float | None:
    for value in values:
        out = number(value)
        if out is not None:
            return out
    return None


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def public_calibration_operator_lock_state() -> dict[str, Any]:
    active = PUBLIC_CALIBRATION_OPERATOR_LOCK.exists()
    reason = ""
    if active:
        try:
            reason = PUBLIC_CALIBRATION_OPERATOR_LOCK.read_text(encoding="utf-8").strip()
        except OSError:
            reason = "operator lock file exists but could not be read"
    return {
        "active": active,
        "path": str(PUBLIC_CALIBRATION_OPERATOR_LOCK.relative_to(ROOT)).replace("\\", "/"),
        "reason": reason,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Theseus ASI Wall Breaker Governor",
        "",
        f"- State: `{payload.get('trigger_state')}`",
        f"- Hard blockers: `{summary.get('hard_blocker_count')}`",
        f"- Public calibration allowed: `{summary.get('public_calibration_allowed')}`",
        f"- Model growth allowed: `{summary.get('model_growth_allowed')}`",
        f"- Candidate promotion allowed: `{summary.get('candidate_promotion_allowed')}`",
        f"- Next: {summary.get('next_primary_action')}",
        "",
        "## Walls",
        "",
    ]
    for row in payload.get("walls", []):
        lines.append(f"- `{row.get('status')}` `{row.get('severity')}` {row.get('id')}: {row.get('title')}")
    lines.extend(["", "## Recommended Next Actions", ""])
    for action in payload.get("recommended_next_actions", []):
        lines.append(f"- {action}")
    command = summary.get("next_primary_command") or []
    if command:
        lines.extend(["", "## Next Command", "", "```powershell", " ".join(str(part) for part in command), "```"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
