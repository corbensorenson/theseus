#!/usr/bin/env python3
"""Private-only generalization governor for the current Theseus wall.

The current bottleneck is not another local score loop. Private transfer lanes
are green, while the latest already-spent public calibration is still low and
operator-locked. This report consolidates that state, preserves the public
calibration boundary, and emits the next safe private/autonomy queue items.

It never runs public calibration, never unlocks the operator lock, and never
uses public tests or solutions as training data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_FLOOR = 0.70
SEMANTIC_ALIAS_MIN_ROWS = 1008
NOVEL_COMPOSITION_MIN_ROWS = 1008
RESIDUAL_FRONTIER_MIN_ROWS = 672
RESIDUAL_FRONTIER_MIN_SPECS = 16
RESIDUAL_FRONTIER_ONLY_MIN_PASS_RATE = 0.70
CONTRACT_BLIND_MIN_ROWS = 240

DEFAULT_OUT = REPORTS / "theseus_generalization_governor_v1.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_generalization_governor_v1.md"
DEFAULT_QUEUE = REPORTS / "theseus_generalization_governor_v1_queue.jsonl"

PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"
PUBLIC_CALIBRATION = REPORTS / "real_code_benchmark_graduation_wide_public_seed23_5x32_interface_floor_v1.json"
PUBLIC_MATRIX = REPORTS / "broad_transfer_matrix_wide_public_seed23_5x32_interface_floor_v1.json"
PUBLIC_TRACES = REPORTS / "real_code_benchmark_traces_wide_public_seed23_5x32_interface_floor_v1.jsonl"
PUBLIC_RESIDUAL = REPORTS / "public_code_transfer_residual_report_wide_public_seed23_5x32_interface_floor_v1.json"
POST_DISTILLATION_READINESS = REPORTS / "post_distillation_public_transfer_readiness_v1.json"
READINESS_PACKET = REPORTS / "public_calibration_readiness_packet.json"
OPERATOR_DRY_RUN = REPORTS / "operator_bounded_public_calibration_dry_run.json"
OPERATOR_EXECUTE = REPORTS / "operator_bounded_public_calibration_execute.json"
OPERATOR_APPROVAL = REPORTS / "public_calibration_operator_approval_post_v4_seed23_5x32.json"

BROAD_PRIVATE = REPORTS / "broad_private_generalization_unattended_v1.json"
BROAD_PRIVATE_LEARNED = REPORTS / "broad_private_learned_distillation_gate_v1.json"
SEMANTIC_ALIAS_GATE = REPORTS / "broad_private_semantic_alias_gate_v1.json"
NOVEL_COMPOSITION_GATE = REPORTS / "broad_private_novel_composition_gate_v1.json"
V4_SCORE = REPORTS / "public_safe_broad_transfer_maturity_v4_score.json"
V4_LEARNED = REPORTS / "public_safe_broad_transfer_maturity_v4_learned_distillation_gate.json"
POST_V4_AUTOPILOT = REPORTS / "post_v4_generalization_autopilot_v1.json"
POST_V4_LEARNED = REPORTS / "post_v4_private_shadow_transfer_v1_smoke160_learned_distillation_gate.json"
POST_V4_SCALING = REPORTS / "post_v4_generalization_autopilot_v1_scaling_profile.json"
POST_V4_QUEUE = REPORTS / "post_v4_generalization_autopilot_v1_queue.jsonl"
V5_REPORT = REPORTS / "private_ecology_generalization_v5.json"
V5_REFRESH = REPORTS / "private_ecology_generalization_v5_refresh.json"
V5_FULL_SCORE = REPORTS / "private_ecology_generalization_v5_full480_score.json"
V5_FULL_LEARNED = REPORTS / "private_ecology_generalization_v5_full480_learned_distillation_gate.json"
V5_QUEUE = REPORTS / "private_ecology_generalization_v5_queue.jsonl"
UNSEEN_TRANSFER_CHALLENGE = REPORTS / "private_unseen_transfer_challenge_v1.json"
UNSEEN_TRANSFER_LEARNED = REPORTS / "private_unseen_transfer_challenge_v1_learned_distillation_gate.json"
ARCH_GUIDANCE_POST_V4 = REPORTS / "architecture_guidance_loop_post_v4_generalization.json"
ARCH_EXPERIMENT_GOVERNANCE = REPORTS / "architecture_experiment_governance.json"
TEACHER_PREFLIGHT = REPORTS / "full_training_teacher_preflight.json"
CAUSAL_ARCHITECTURE_DELTA = REPORTS / "causal_architecture_delta_loop.json"
STUDENT_FIRST_AUDIT = REPORTS / "student_first_evidence_audit.json"
RESIDUAL_RATCHET = REPORTS / "private_residual_self_improvement_ratchet_v1.json"
RESIDUAL_FRONTIER = REPORTS / "private_residual_frontier_v1.json"
CONTRACT_BLIND_TRANSFER = REPORTS / "private_contract_blind_transfer_v1.json"
CONTRACT_BLIND_LEARNED = REPORTS / "private_contract_blind_transfer_v1_learned_distillation_gate.json"
AGENT_LANE_TRANSFER = REPORTS / "agent_lane_transfer_gate.json"

FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS = [
    REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json",
    REPORTS / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl",
    REPORTS / "student_code_candidates_post_v4_seed23_5x32.jsonl",
    REPORTS / "operator_bounded_public_calibration_post_v4_seed23_5x32.json",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--queue-out", default=rel(DEFAULT_QUEUE))
    parser.add_argument("--stale-seconds", type=int, default=72 * 3600)
    args = parser.parse_args()

    report = build_report(stale_seconds=max(3600, int(args.stale_seconds)), queue_out=resolve(args.queue_out))
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.queue_out), report.get("queue", []))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(*, stale_seconds: int, queue_out: Path) -> dict[str, Any]:
    state = observe()
    evidence = summarize_evidence(state, stale_seconds=stale_seconds)
    gates = build_gates(state, evidence)
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failed = [row for row in gates if row["severity"] == "warning" and not row["passed"]]
    transfer_blocked = bool(
        evidence["public"]["pass_rate"] < PUBLIC_FLOOR
        or evidence["public"]["cards_below_floor"]
        or evidence["readiness"]["operator_lock_active"]
    )
    private_maturity_blocked = bool(
        not evidence["private"]["v4_learned_maturity"]["ready"]
        or not evidence["private"]["post_v4_shadow_learned_maturity"]["ready"]
        or not evidence["private"]["v5_learned_maturity"]["ready"]
        or not evidence["private"]["unseen_transfer_learned_maturity"]["ready"]
        or not evidence["private"]["contract_blind_learned_maturity"]["ready"]
    )
    trigger_state = "RED" if hard_failed else "YELLOW" if warning_failed or transfer_blocked else "GREEN"
    queue = build_queue(evidence, gates)
    next_primary = queue[0] if queue else {}
    return {
        "policy": "project_theseus_generalization_governor_v1",
        "created_utc": now(),
        "ok": not hard_failed,
        "trigger_state": trigger_state,
        "wall": {
            "name": "broad_generalization_transfer",
            "status": "blocked" if transfer_blocked else "cleared",
            "short_reading": (
                "Public transfer remains below floor, and older private learned lanes are not maturity-green under structural/train-novelty checks."
                if transfer_blocked and private_maturity_blocked
                else (
                    "Private learned evidence is green, but the public wide score is still below floor."
                    if transfer_blocked
                    else "Public transfer floor is no longer the active blocker."
                )
            ),
        },
        "summary": {
            "public_pass_rate": evidence["public"]["pass_rate"],
            "public_floor": PUBLIC_FLOOR,
            "public_task_count": evidence["public"]["task_count"],
            "cards_below_floor": evidence["public"]["cards_below_floor"],
            "public_calibration_allowed": evidence["readiness"]["public_calibration_allowed"],
            "operator_lock_active": evidence["readiness"]["operator_lock_active"],
            "forbidden_post_v4_public_artifact_count": len(evidence["readiness"]["forbidden_post_v4_public_artifacts_present"]),
            "post_v4_public_artifacts_approved": evidence["readiness"]["post_v4_public_artifact_state"]["allowed"],
            "private_broad_green": evidence["private"]["broad_green"],
            "broad_strict_novel_learned_only_pass_rate": evidence["private"]["broad_strict_novel_learned_only_pass_rate"],
            "broad_strict_novel_learned_only_pass_count": evidence["private"]["broad_strict_novel_learned_only_pass_count"],
            "broad_exact_train_body_memory_pass_count": evidence["private"]["broad_exact_train_body_memory_pass_count"],
            "broad_learned_pass_normalized_ast_unique_count": evidence["private"]["broad_learned_pass_normalized_ast_unique_count"],
            "broad_learned_pass_ast_shape_count": evidence["private"]["broad_learned_pass_ast_shape_count"],
            "broad_learned_pass_top_duplicate_rate": evidence["private"]["broad_learned_pass_top_duplicate_rate"],
            "broad_learned_control_structure_ready": evidence["private"]["broad_learned_control_structure_ready"],
            "broad_learned_train_ast_overlap_rate": evidence["private"]["broad_learned_train_ast_overlap_rate"],
            "broad_learned_train_body_overlap_rate": evidence["private"]["broad_learned_train_body_overlap_rate"],
            "broad_learned_train_novelty_ready": evidence["private"]["broad_learned_train_novelty_ready"],
            "semantic_alias_green": evidence["private"]["semantic_alias_green"],
            "semantic_alias_learned_only_pass_rate": evidence["private"]["semantic_alias_learned_only_pass_rate"],
            "semantic_alias_inferred_token_pass_count": evidence["private"]["semantic_alias_inferred_token_pass_count"],
            "novel_composition_green": evidence["private"]["novel_composition_green"],
            "novel_composition_only_pass_rate": evidence["private"]["novel_composition_only_pass_rate"],
            "novel_composition_token_pass_count": evidence["private"]["novel_composition_token_pass_count"],
            "v4_learned_green": evidence["private"]["v4_learned_green"],
            "v4_learned_maturity_ready": evidence["private"]["v4_learned_maturity"]["ready"],
            "v4_learned_train_body_overlap_rate": evidence["private"]["v4_learned_maturity"]["train_body_overlap_rate"],
            "post_v4_shadow_cap_reached": evidence["private"]["post_v4_shadow_cap_reached"],
            "post_v4_shadow_learned_maturity_ready": evidence["private"]["post_v4_shadow_learned_maturity"]["ready"],
            "post_v4_shadow_train_body_overlap_rate": evidence["private"]["post_v4_shadow_learned_maturity"]["train_body_overlap_rate"],
            "v5_private_ecology_green": evidence["private"]["v5_green"],
            "v5_learned_maturity_ready": evidence["private"]["v5_learned_maturity"]["ready"],
            "v5_train_body_overlap_rate": evidence["private"]["v5_learned_maturity"]["train_body_overlap_rate"],
            "unseen_transfer_challenge_green": evidence["private"]["unseen_transfer_challenge_green"],
            "unseen_transfer_learned_maturity_ready": evidence["private"]["unseen_transfer_learned_maturity"]["ready"],
            "unseen_transfer_train_body_overlap_rate": evidence["private"]["unseen_transfer_learned_maturity"]["train_body_overlap_rate"],
            "unseen_transfer_challenge_pass_rate": evidence["private"]["unseen_transfer_challenge_pass_rate"],
            "unseen_transfer_challenge_learned_only_pass_rate": evidence["private"]["unseen_transfer_challenge_learned_only_pass_rate"],
            "unseen_transfer_challenge_rows": evidence["private"]["unseen_transfer_challenge_rows"],
            "unseen_transfer_exact_semantic_replay_count": evidence["private"]["unseen_transfer_exact_semantic_replay_count"],
            "residual_frontier_green": evidence["private"]["residual_frontier_green"],
            "residual_frontier_pass_rate": evidence["private"]["residual_frontier_pass_rate"],
            "residual_frontier_only_pass_rate": evidence["private"]["residual_frontier_only_pass_rate"],
            "residual_frontier_token_pass_count": evidence["private"]["residual_frontier_token_pass_count"],
            "residual_frontier_rows": evidence["private"]["residual_frontier_rows"],
            "residual_frontier_spec_count": evidence["private"]["residual_frontier_spec_count"],
            "contract_blind_transfer_green": evidence["private"]["contract_blind_transfer_green"],
            "contract_blind_transfer_rows": evidence["private"]["contract_blind_transfer_rows"],
            "contract_blind_transfer_pass_rate": evidence["private"]["contract_blind_transfer_pass_rate"],
            "contract_blind_transfer_strict_learned_only_pass_rate": evidence["private"]["contract_blind_transfer_strict_learned_only_pass_rate"],
            "contract_blind_transfer_learned_token_pass_count": evidence["private"]["contract_blind_transfer_learned_token_pass_count"],
            "contract_blind_transfer_semantic_names_withheld_count": evidence["private"]["contract_blind_transfer_semantic_names_withheld_count"],
            "contract_blind_learned_maturity_ready": evidence["private"]["contract_blind_learned_maturity"]["ready"],
            "contract_blind_train_body_overlap_rate": evidence["private"]["contract_blind_learned_maturity"]["train_body_overlap_rate"],
            "contract_blind_pass_ast_shape_count": evidence["private"]["contract_blind_learned_maturity"]["pass_ast_shape_count"],
            "contract_blind_pass_normalized_ast_unique_count": evidence["private"]["contract_blind_learned_maturity"]["pass_normalized_ast_unique_count"],
            "contract_blind_pass_top_duplicate_rate": evidence["private"]["contract_blind_learned_maturity"]["pass_top_duplicate_rate"],
            "architecture_guidance_green": evidence["downstream"]["architecture_guidance_green"],
            "architecture_experiment_recommended": evidence["downstream"]["architecture_experiment_recommended"],
            "teacher_preflight_ok": evidence["downstream"]["teacher_preflight_ok"],
            "causal_architecture_delta_green": evidence["downstream"]["causal_architecture_delta_green"],
            "causal_best_target_delta": evidence["downstream"]["causal_best_target_delta"],
            "causal_private_semantic_test_delta": evidence["downstream"]["causal_private_semantic_test_delta"],
            "causal_private_semantic_positive_family_count": evidence["downstream"]["causal_private_semantic_positive_family_count"],
            "causal_private_semantic_regressed_family_count": evidence["downstream"]["causal_private_semantic_regressed_family_count"],
            "student_first_audit_clear": evidence["downstream"]["student_first_audit_clear"],
            "residual_ratchet_green": evidence["downstream"]["residual_ratchet_green"],
            "residual_ratchet_decision": evidence["downstream"]["residual_ratchet_decision"],
            "residual_ratchet_top_action": evidence["downstream"]["residual_ratchet_top_action"],
            "agent_lane_core_transfer_ready": evidence["downstream"]["agent_lane_core_transfer_ready"],
            "agent_lane_breadth_extension_ready": evidence["downstream"]["agent_lane_breadth_extension_ready"],
            "agent_lane_terminal_tool_use_cases": evidence["downstream"]["agent_lane_terminal_tool_use_cases"],
            "agent_lane_cross_domain_capsule_count": evidence["downstream"]["agent_lane_cross_domain_capsule_count"],
            "agent_lane_sts_named_consumer_effect": evidence["downstream"]["agent_lane_sts_named_consumer_effect"],
            "agent_lane_remaining_blockers": evidence["downstream"]["agent_lane_remaining_blockers"],
            "prototype_pass_count_total": evidence["private"]["prototype_pass_count_total"],
            "learned_token_pass_count_total": evidence["private"]["learned_token_pass_count_total"],
            "external_inference_calls": evidence["safety"]["external_inference_calls"],
            "hard_failed_gate_count": len(hard_failed),
            "warning_failed_gate_count": len(warning_failed),
            "queue": rel(queue_out),
            "queue_item_count": len(queue),
            "next_primary_action": next_primary.get("kind"),
            "next_primary_command": next_primary.get("command") or [],
            "score_semantics": "governor report only; not promotion evidence and not public calibration",
        },
        "evidence": evidence,
        "gates": gates,
        "queue": queue,
        "next_actions": [row["title"] for row in queue[:6]],
        "rules": {
            "public_calibration": "Do not execute public calibration from this governor. Use the guarded operator runner only after explicit approval.",
            "training": "Train only from private curricula and private eval feedback; never from public prompts, tests, solutions, traces, or score labels.",
            "teacher": "Teacher may be used only for proposal-only architecture guidance through governed teacher paths.",
            "promotion": "Private learned-token success is necessary but not sufficient; promotion remains blocked until broad clean public transfer clears the floor honestly.",
        },
        "external_inference_calls": 0,
    }


def observe() -> dict[str, Any]:
    return {
        "public_calibration": read_json(PUBLIC_CALIBRATION, {}),
        "public_matrix": read_json(PUBLIC_MATRIX, {}),
        "public_residual": read_json(PUBLIC_RESIDUAL, {}),
        "post_distillation_readiness": read_json(POST_DISTILLATION_READINESS, {}),
        "readiness_packet": read_json(READINESS_PACKET, {}),
        "operator_dry_run": read_json(OPERATOR_DRY_RUN, {}),
        "broad_private": read_json(BROAD_PRIVATE, {}),
        "broad_private_learned": read_json(BROAD_PRIVATE_LEARNED, {}),
        "semantic_alias_gate": read_json(SEMANTIC_ALIAS_GATE, {}),
        "novel_composition_gate": read_json(NOVEL_COMPOSITION_GATE, {}),
        "v4_score": read_json(V4_SCORE, {}),
        "v4_learned": read_json(V4_LEARNED, {}),
        "post_v4_autopilot": read_json(POST_V4_AUTOPILOT, {}),
        "post_v4_learned": read_json(POST_V4_LEARNED, {}),
        "post_v4_scaling": read_json(POST_V4_SCALING, {}),
        "v5_report": read_json(V5_REPORT, {}),
        "v5_refresh": read_json(V5_REFRESH, {}),
        "v5_full_score": read_json(V5_FULL_SCORE, {}),
        "v5_full_learned": read_json(V5_FULL_LEARNED, {}),
        "unseen_transfer_challenge": read_json(UNSEEN_TRANSFER_CHALLENGE, {}),
        "unseen_transfer_learned": read_json(UNSEEN_TRANSFER_LEARNED, {}),
        "architecture_guidance": read_json(ARCH_GUIDANCE_POST_V4, {}),
        "architecture_experiment_governance": read_json(ARCH_EXPERIMENT_GOVERNANCE, {}),
        "teacher_preflight": read_json(TEACHER_PREFLIGHT, {}),
        "causal_architecture_delta": read_json(CAUSAL_ARCHITECTURE_DELTA, {}),
        "student_first_audit": read_json(STUDENT_FIRST_AUDIT, {}),
        "residual_ratchet": read_json(RESIDUAL_RATCHET, {}),
        "residual_frontier": read_json(RESIDUAL_FRONTIER, {}),
        "contract_blind_transfer": read_json(CONTRACT_BLIND_TRANSFER, {}),
        "contract_blind_learned": read_json(CONTRACT_BLIND_LEARNED, {}),
        "agent_lane_transfer": read_json(AGENT_LANE_TRANSFER, {}),
    }


def summarize_evidence(state: dict[str, Any], *, stale_seconds: int) -> dict[str, Any]:
    public_summary = object_field(state["public_calibration"], "summary")
    matrix_summary = object_field(state["public_matrix"], "summary")
    readiness = state["post_distillation_readiness"]
    readiness_summary = object_field(readiness, "summary")
    packet = state["readiness_packet"]
    packet_summary = object_field(packet, "summary")
    dry_run_summary = object_field(state["operator_dry_run"], "summary")

    public_pass_rate = first_number(
        public_summary.get("real_public_task_pass_rate"),
        public_summary.get("multi_stream_pass_rate"),
        matrix_summary.get("real_public_pass_rate"),
        readiness_summary.get("public_pass_rate"),
        0.0,
    )
    public_task_count = int(
        first_number(public_summary.get("public_task_count"), matrix_summary.get("real_public_task_count"), readiness_summary.get("public_task_count"), 0)
    )
    cards_below_floor = list_field(matrix_summary.get("cards_below_floor") or readiness_summary.get("cards_below_floor"))

    broad_summary = object_field(state["broad_private"], "summary")
    broad_learned_summary = object_field(state["broad_private_learned"], "summary")
    broad_learned_structure = object_field(broad_learned_summary, "learned_structural_inventory")
    broad_learned_novelty = object_field(broad_learned_summary, "learned_train_novelty_inventory")
    broad_strict_learned_pass_rate = first_number(broad_learned_summary.get("strict_novel_learned_only_pass_rate"), 0.0)
    broad_strict_learned_pass_count = int(first_number(broad_learned_summary.get("strict_novel_learned_only_pass_count"), 0))
    broad_exact_body_memory_pass_count = int(first_number(broad_learned_summary.get("exact_train_body_memory_pass_count"), 999))
    alias_summary = object_field(state["semantic_alias_gate"], "summary")
    composition_summary = object_field(state["novel_composition_gate"], "summary")
    v4_score_summary = object_field(state["v4_score"], "summary")
    v4_learned_summary = object_field(state["v4_learned"], "summary")
    v4_learned_maturity = learned_maturity(v4_learned_summary)
    post_v4_summary = object_field(state["post_v4_autopilot"], "summary")
    post_v4_learned_summary = object_field(state["post_v4_learned"], "summary")
    post_v4_learned_maturity = learned_maturity(post_v4_learned_summary)
    post_v4_scaling_summary = object_field(state["post_v4_scaling"], "summary")
    v5_refresh_summary = object_field(state["v5_refresh"], "summary")
    v5_score_summary = object_field(state["v5_full_score"], "summary")
    v5_learned_summary = object_field(state["v5_full_learned"], "summary")
    v5_learned_maturity = learned_maturity(v5_learned_summary)
    unseen_summary = object_field(state["unseen_transfer_challenge"], "summary")
    unseen_learned_summary = object_field(state["unseen_transfer_learned"], "summary")
    unseen_learned_maturity = learned_maturity(unseen_learned_summary)
    architecture_guidance = state["architecture_guidance"]
    architecture_experiment = state["architecture_experiment_governance"]
    teacher_preflight = state["teacher_preflight"]
    teacher_summary = object_field(teacher_preflight, "summary")
    causal_delta = state["causal_architecture_delta"]
    causal_summary = object_field(causal_delta, "summary")
    student_first = state["student_first_audit"]
    student_first_summary = object_field(student_first, "summary")
    residual_ratchet = state["residual_ratchet"]
    ratchet_summary = object_field(residual_ratchet, "summary")
    residual_frontier = state["residual_frontier"]
    frontier_summary = object_field(residual_frontier, "summary")
    contract_blind = state["contract_blind_transfer"]
    contract_blind_summary = object_field(contract_blind, "summary")
    contract_blind_learned = state["contract_blind_learned"]
    contract_blind_learned_summary = object_field(contract_blind_learned, "summary")
    contract_blind_learned_maturity = learned_maturity(contract_blind_learned_summary)
    agent_lane = state["agent_lane_transfer"]
    agent_summary = object_field(agent_lane, "summary")
    agent_repo = object_field(agent_summary, "repo_repair")
    agent_tool = object_field(agent_summary, "terminal_tool_use")
    agent_sts = object_field(agent_summary, "sts_consumption")
    agent_conversation = object_field(agent_summary, "conversation")
    agent_puffer = object_field(agent_summary, "pufferlib_rl")

    post_v4_public_state = post_v4_public_artifact_state()
    post_v4_present = list(post_v4_public_state.get("present_artifacts") or [])
    forbidden_present = [] if post_v4_public_state.get("allowed") else post_v4_present
    stale = {
        "post_distillation_readiness": stale_report(POST_DISTILLATION_READINESS, stale_seconds),
        "readiness_packet": stale_report(READINESS_PACKET, stale_seconds),
        "post_v4_autopilot": stale_report(POST_V4_AUTOPILOT, stale_seconds),
        "post_v4_learned": stale_report(POST_V4_LEARNED, stale_seconds),
        "semantic_alias_gate": stale_report(SEMANTIC_ALIAS_GATE, stale_seconds),
        "novel_composition_gate": stale_report(NOVEL_COMPOSITION_GATE, stale_seconds),
        "v4_learned": stale_report(V4_LEARNED, stale_seconds),
        "v5_refresh": stale_report(V5_REFRESH, stale_seconds),
        "v5_full_score": stale_report(V5_FULL_SCORE, stale_seconds),
        "v5_full_learned": stale_report(V5_FULL_LEARNED, stale_seconds),
        "unseen_transfer_challenge": stale_report(UNSEEN_TRANSFER_CHALLENGE, stale_seconds),
        "unseen_transfer_learned": stale_report(UNSEEN_TRANSFER_LEARNED, stale_seconds),
        "architecture_guidance": stale_report(ARCH_GUIDANCE_POST_V4, stale_seconds),
        "architecture_experiment_governance": stale_report(ARCH_EXPERIMENT_GOVERNANCE, stale_seconds),
        "teacher_preflight": stale_report(TEACHER_PREFLIGHT, stale_seconds),
        "causal_architecture_delta": stale_report(CAUSAL_ARCHITECTURE_DELTA, stale_seconds),
        "student_first_audit": stale_report(STUDENT_FIRST_AUDIT, stale_seconds),
        "residual_ratchet": stale_report(RESIDUAL_RATCHET, stale_seconds),
        "residual_frontier": stale_report(RESIDUAL_FRONTIER, stale_seconds),
        "contract_blind_transfer": stale_report(CONTRACT_BLIND_TRANSFER, stale_seconds),
        "contract_blind_learned": stale_report(CONTRACT_BLIND_LEARNED, stale_seconds),
        "agent_lane_transfer": stale_report(AGENT_LANE_TRANSFER, stale_seconds),
    }
    external_calls = sum(
        top_external_calls(report)
        for report in [
            readiness,
            packet,
            state["operator_dry_run"],
            state["broad_private"],
            state["broad_private_learned"],
            state["semantic_alias_gate"],
            state["novel_composition_gate"],
            state["v4_score"],
            state["v4_learned"],
            state["post_v4_autopilot"],
            state["post_v4_learned"],
            state["v5_report"],
            state["v5_refresh"],
            state["v5_full_score"],
            state["v5_full_learned"],
            state["unseen_transfer_challenge"],
            state["unseen_transfer_learned"],
            architecture_guidance,
            architecture_experiment,
            teacher_preflight,
            causal_delta,
            student_first,
            residual_ratchet,
            residual_frontier,
            contract_blind,
            contract_blind_learned,
            agent_lane,
        ]
    )
    broad_green = bool(
        state["broad_private"].get("trigger_state") == "GREEN"
        and first_number(broad_summary.get("heldout_pass_rate"), 0.0) >= 1.0
        and state["broad_private_learned"].get("trigger_state") == "GREEN"
        and first_number(broad_learned_summary.get("learned_only_pass_rate"), 0.0) >= 1.0
        and broad_strict_learned_pass_rate >= PUBLIC_FLOOR
        and broad_exact_body_memory_pass_count == 0
        and int(first_number(broad_learned_summary.get("prototype_pass_count"), 999)) == 0
        and int(first_number(broad_learned_structure.get("pass_normalized_ast_unique_count"), 0))
        >= int(first_number(broad_learned_structure.get("min_pass_normalized_ast_unique_count"), 999))
        and int(first_number(broad_learned_structure.get("pass_ast_shape_count"), 0))
        >= int(first_number(broad_learned_structure.get("min_pass_ast_shape_count"), 999))
        and first_number(broad_learned_structure.get("pass_top_normalized_ast_duplicate_rate"), 1.0)
        <= first_number(broad_learned_structure.get("max_pass_top_duplicate_rate"), 0.0)
        and broad_learned_structure.get("control_structure_coverage_ready") is True
        and broad_learned_novelty.get("novelty_ready") is True
    )
    alias_green = bool(
        state["semantic_alias_gate"].get("trigger_state") == "GREEN"
        and int(first_number(alias_summary.get("alias_row_count"), 0)) >= SEMANTIC_ALIAS_MIN_ROWS
        and first_number(alias_summary.get("pass_rate"), 0.0) >= 1.0
        and first_number(alias_summary.get("learned_only_pass_rate"), 0.0) >= 1.0
        and int(first_number(alias_summary.get("semantic_alias_inferred_token_pass_count"), 0)) >= SEMANTIC_ALIAS_MIN_ROWS
        and int(first_number(alias_summary.get("diagnostic_adapter_pass_count"), 999)) == 0
        and int(first_number(alias_summary.get("prototype_pass_count"), 999)) == 0
    )
    composition_green = bool(
        state["novel_composition_gate"].get("trigger_state") == "GREEN"
        and int(first_number(composition_summary.get("row_count"), 0)) >= NOVEL_COMPOSITION_MIN_ROWS
        and first_number(composition_summary.get("pass_rate"), 0.0) >= 1.0
        and first_number(composition_summary.get("composition_only_pass_rate"), 0.0) >= 1.0
        and int(first_number(composition_summary.get("composition_token_pass_count"), 0)) >= NOVEL_COMPOSITION_MIN_ROWS
        and int(first_number(composition_summary.get("diagnostic_adapter_pass_count"), 999)) == 0
        and int(first_number(composition_summary.get("prototype_pass_count"), 999)) == 0
    )
    v4_green = bool(
        state["v4_score"].get("trigger_state") == "GREEN"
        and first_number(v4_score_summary.get("pass_rate"), 0.0) >= 1.0
        and state["v4_learned"].get("trigger_state") == "GREEN"
        and first_number(v4_learned_summary.get("learned_only_pass_rate"), 0.0) >= 1.0
        and int(first_number(v4_learned_summary.get("prototype_pass_count"), 999)) == 0
        and v4_learned_maturity["ready"] is True
    )
    post_v4_green = bool(
        state["post_v4_autopilot"].get("trigger_state") == "GREEN"
        and first_number(post_v4_summary.get("pass_rate"), 0.0) >= 1.0
        and first_number(post_v4_summary.get("learned_only_pass_rate"), 0.0) >= 1.0
        and int(first_number(post_v4_summary.get("prototype_pass_count"), 999)) == 0
        and bool(get_path(state["post_v4_autopilot"], ["summary", "scale_efficiency", "scale_cap_reached"], False))
        and state["post_v4_learned"].get("trigger_state") == "GREEN"
        and first_number(post_v4_learned_summary.get("learned_only_pass_rate"), 0.0) >= 1.0
        and int(first_number(post_v4_learned_summary.get("prototype_pass_count"), 999)) == 0
        and post_v4_learned_maturity["ready"] is True
    )
    v5_green = bool(
        state["v5_report"].get("trigger_state") == "GREEN"
        and state["v5_refresh"].get("trigger_state") == "GREEN"
        and v5_refresh_summary.get("completion_evidence_status") == "private_ecology_v5_learned_refresh_ready"
        and get_path(state["v5_refresh"], ["summary", "freshness", "fresh"], False) is True
        and state["v5_full_score"].get("trigger_state") == "GREEN"
        and first_number(v5_score_summary.get("pass_rate"), 0.0) >= 1.0
        and state["v5_full_learned"].get("trigger_state") == "GREEN"
        and first_number(v5_learned_summary.get("learned_only_pass_rate"), 0.0) >= 1.0
        and int(first_number(v5_learned_summary.get("prototype_pass_count"), 999)) == 0
        and v5_learned_maturity["ready"] is True
    )
    unseen_green = bool(
        state["unseen_transfer_challenge"].get("trigger_state") == "GREEN"
        and unseen_summary.get("completion_evidence_status") == "private_unseen_transfer_challenge_ready"
        and int(first_number(unseen_summary.get("challenge_row_count"), 0)) >= 120
        and int(first_number(unseen_summary.get("exact_semantic_key_replay_count"), 999)) == 0
        and first_number(unseen_summary.get("pass_rate"), 0.0) >= 0.70
        and first_number(unseen_summary.get("learned_only_pass_rate"), 0.0) >= 0.70
        and int(first_number(unseen_summary.get("prototype_pass_count"), 999)) == 0
        and get_path(state["unseen_transfer_challenge"], ["summary", "freshness", "fresh"], False) is True
        and state["unseen_transfer_learned"].get("trigger_state") == "GREEN"
        and first_number(unseen_learned_summary.get("learned_only_pass_rate"), 0.0) >= 0.70
        and int(first_number(unseen_learned_summary.get("prototype_pass_count"), 999)) == 0
        and unseen_learned_maturity["ready"] is True
    )
    frontier_green = bool(
        residual_frontier.get("trigger_state") == "GREEN"
        and frontier_summary.get("completion_evidence_status") == "private_residual_frontier_ready"
        and int(first_number(frontier_summary.get("row_count"), 0)) >= RESIDUAL_FRONTIER_MIN_ROWS
        and int(first_number(frontier_summary.get("frontier_spec_count"), 0)) >= RESIDUAL_FRONTIER_MIN_SPECS
        and first_number(frontier_summary.get("pass_rate"), 0.0) >= 1.0
        and first_number(frontier_summary.get("frontier_only_pass_rate"), 0.0) >= RESIDUAL_FRONTIER_ONLY_MIN_PASS_RATE
        and int(first_number(frontier_summary.get("frontier_token_pass_count"), 0)) > 0
        and int(first_number(frontier_summary.get("diagnostic_adapter_pass_count"), 999)) == 0
        and int(first_number(frontier_summary.get("prototype_pass_count"), 999)) == 0
        and int(first_number(frontier_summary.get("external_inference_calls"), 0)) == 0
        and int(first_number(frontier_summary.get("hard_failed_gate_count"), 999)) == 0
    )
    contract_blind_green = bool(
        contract_blind.get("trigger_state") == "GREEN"
        and contract_blind_summary.get("completion_evidence_status") == "private_contract_blind_transfer_ready"
        and int(first_number(contract_blind_summary.get("row_count"), 0)) >= CONTRACT_BLIND_MIN_ROWS
        and first_number(contract_blind_summary.get("pass_rate"), 0.0) >= 0.70
        and first_number(contract_blind_summary.get("strict_novel_learned_only_pass_rate"), 0.0) >= 0.70
        and first_number(contract_blind_summary.get("control_pass_rate"), 1.0) < first_number(contract_blind_summary.get("strict_novel_learned_only_pass_rate"), 0.0)
        and int(first_number(contract_blind_summary.get("semantic_names_withheld_count"), 0)) >= int(first_number(contract_blind_summary.get("row_count"), 0))
        and int(first_number(contract_blind_summary.get("prototype_pass_count"), 999)) == 0
        and int(first_number(contract_blind_summary.get("diagnostic_adapter_pass_count"), 999)) == 0
        and int(first_number(contract_blind_summary.get("body_memory_replay_candidate_rows"), 999)) == 0
        and int(first_number(contract_blind_learned_summary.get("learned_only_task_count"), 0)) >= CONTRACT_BLIND_MIN_ROWS
        and int(first_number(contract_blind_learned_summary.get("learned_token_pass_count"), 0)) >= int(first_number(contract_blind_summary.get("learned_token_pass_count"), 0))
        and int(first_number(contract_blind_learned_summary.get("exact_train_body_memory_pass_count"), 999)) == 0
        and int(first_number(object_field(contract_blind_learned_summary, "candidate_inventory").get("exact_train_body_memory_candidate_rows"), 999)) == 0
        and int(first_number(object_field(contract_blind_learned_summary, "pass_inventory").get("diagnostic_adapter_pass_count"), 999)) == 0
        and contract_blind_learned_maturity["ready"] is True
        and int(first_number(contract_blind_summary.get("external_inference_calls"), 0)) == 0
    )
    learned_counts = [
        broad_strict_learned_pass_count,
        int(first_number(alias_summary.get("semantic_alias_inferred_token_pass_count"), 0)),
        int(first_number(composition_summary.get("composition_token_pass_count"), 0)),
        int(first_number(v4_learned_summary.get("learned_token_pass_count"), 0)),
        int(first_number(post_v4_learned_summary.get("learned_token_pass_count"), 0)),
        int(first_number(v5_learned_summary.get("learned_token_pass_count"), 0)),
        int(first_number(unseen_learned_summary.get("learned_token_pass_count"), 0)),
        int(first_number(frontier_summary.get("frontier_token_pass_count"), 0)),
        int(first_number(contract_blind_summary.get("learned_token_pass_count"), 0)),
    ]
    prototype_counts = [
        int(first_number(broad_learned_summary.get("prototype_pass_count"), 0)),
        int(first_number(alias_summary.get("prototype_pass_count"), 0)),
        int(first_number(composition_summary.get("prototype_pass_count"), 0)),
        int(first_number(v4_learned_summary.get("prototype_pass_count"), 0)),
        int(first_number(post_v4_learned_summary.get("prototype_pass_count"), 0)),
        int(first_number(v5_learned_summary.get("prototype_pass_count"), 0)),
        int(first_number(unseen_learned_summary.get("prototype_pass_count"), 0)),
        int(first_number(frontier_summary.get("prototype_pass_count"), 0)),
        int(first_number(contract_blind_summary.get("prototype_pass_count"), 0)),
    ]
    agent_lane_core_ready = bool(
        agent_repo.get("transfer_consumer_ready") is True
        and agent_tool.get("case_ready") is True
        and agent_tool.get("transfer_consumer_ready") is True
        and agent_sts.get("named_consumer_effect") is True
        and int(first_number(agent_tool.get("case_count"), 0)) >= 64
    )
    agent_lane_cross_domain_capsules = int(first_number(get_path(state, ["cross_domain_sts_capsules", "summary", "capsule_count"]), 0))
    # The agent gate reads cross-domain capsules directly through sts_consumption,
    # but older reports may omit capsule counts. Fall back to the current capsule
    # report so the governor can queue the correct next private lane.
    if agent_lane_cross_domain_capsules <= 0:
        cross_domain_capsules = read_json(REPORTS / "cross_domain_sts_capsules.json", {})
        agent_lane_cross_domain_capsules = int(first_number(get_path(cross_domain_capsules, ["summary", "capsule_count"]), 0))
    agent_lane_breadth_ready = bool(
        agent_lane_core_ready
        and (
            agent_conversation.get("graduated") is True
            or agent_puffer.get("native_policy_ready") is True
        )
    )
    agent_blockers: list[str] = []
    if not agent_lane_core_ready:
        agent_blockers.append("core_tool_use_or_sts_consumer_missing")
    if agent_conversation.get("graduated") is not True:
        agent_blockers.append("conversation_transfer_not_graduated")
    if agent_puffer.get("native_policy_ready") is not True:
        agent_blockers.append("rl_policy_transfer_not_ready")
    return {
        "public": {
            "calibration": rel(PUBLIC_CALIBRATION),
            "matrix": rel(PUBLIC_MATRIX),
            "traces": rel(PUBLIC_TRACES),
            "residual": rel(PUBLIC_RESIDUAL),
            "pass_rate": public_pass_rate,
            "task_count": public_task_count,
            "cards_below_floor": cards_below_floor,
            "dominant_residual_categories": readiness_summary.get("dominant_public_residual_categories") or [],
            "next_public_blocker": readiness_summary.get("next_public_blocker"),
            "promotion_candidate_card_count": matrix_summary.get("promotion_candidate_card_count"),
            "floor": PUBLIC_FLOOR,
        },
        "readiness": {
            "post_distillation_trigger_state": readiness.get("trigger_state"),
            "readiness_packet_trigger_state": packet.get("trigger_state"),
            "packet_mode": packet.get("mode"),
            "packet_technical_ready": bool(packet.get("technical_ready_for_one_bounded_public_calibration") or packet_summary.get("technical_ready")),
            "operator_dry_run_trigger_state": state["operator_dry_run"].get("trigger_state"),
            "operator_dry_run_ready": bool(dry_run_summary.get("ready_for_operator_approval")),
            "operator_dry_run_executed": bool(dry_run_summary.get("executed")),
            "operator_lock_active": PUBLIC_LOCK.exists(),
            "public_calibration_allowed": bool(packet.get("public_calibration_allowed") or readiness.get("public_calibration_allowed")),
            "post_v4_public_artifact_state": post_v4_public_state,
            "post_v4_public_artifacts_present": post_v4_present,
            "forbidden_post_v4_public_artifacts_present": forbidden_present,
            "stale": stale,
        },
        "private": {
            "broad_green": broad_green,
            "broad_private_pass_rate": first_number(broad_summary.get("heldout_pass_rate"), 0.0),
            "broad_learned_only_pass_rate": first_number(broad_learned_summary.get("learned_only_pass_rate"), 0.0),
            "broad_strict_novel_learned_only_pass_rate": broad_strict_learned_pass_rate,
            "broad_strict_novel_learned_only_pass_count": broad_strict_learned_pass_count,
            "broad_exact_train_body_memory_pass_count": broad_exact_body_memory_pass_count,
            "broad_learned_pass_normalized_ast_unique_count": int(first_number(broad_learned_structure.get("pass_normalized_ast_unique_count"), 0)),
            "broad_learned_pass_ast_shape_count": int(first_number(broad_learned_structure.get("pass_ast_shape_count"), 0)),
            "broad_learned_pass_top_duplicate_rate": first_number(broad_learned_structure.get("pass_top_normalized_ast_duplicate_rate"), 1.0),
            "broad_learned_control_structure_ready": broad_learned_structure.get("control_structure_coverage_ready") is True,
            "broad_learned_train_ast_overlap_rate": first_number(broad_learned_novelty.get("exact_train_normalized_ast_overlap_rate"), 1.0),
            "broad_learned_train_body_overlap_rate": first_number(broad_learned_novelty.get("exact_train_body_normalized_ast_overlap_rate"), 1.0),
            "broad_learned_train_novelty_ready": broad_learned_novelty.get("novelty_ready") is True,
            "semantic_alias_green": alias_green,
            "semantic_alias_pass_rate": first_number(alias_summary.get("pass_rate"), 0.0),
            "semantic_alias_learned_only_pass_rate": first_number(alias_summary.get("learned_only_pass_rate"), 0.0),
            "semantic_alias_inferred_token_pass_count": int(first_number(alias_summary.get("semantic_alias_inferred_token_pass_count"), 0)),
            "semantic_alias_row_count": int(first_number(alias_summary.get("alias_row_count"), 0)),
            "novel_composition_green": composition_green,
            "novel_composition_pass_rate": first_number(composition_summary.get("pass_rate"), 0.0),
            "novel_composition_only_pass_rate": first_number(composition_summary.get("composition_only_pass_rate"), 0.0),
            "novel_composition_token_pass_count": int(first_number(composition_summary.get("composition_token_pass_count"), 0)),
            "novel_composition_row_count": int(first_number(composition_summary.get("row_count"), 0)),
            "v4_learned_green": v4_green,
            "v4_private_pass_rate": first_number(v4_score_summary.get("pass_rate"), 0.0),
            "v4_learned_only_pass_rate": first_number(v4_learned_summary.get("learned_only_pass_rate"), 0.0),
            "v4_learned_maturity": v4_learned_maturity,
            "post_v4_shadow_green": post_v4_green,
            "post_v4_shadow_pass_rate": first_number(post_v4_summary.get("pass_rate"), 0.0),
            "post_v4_shadow_learned_only_pass_rate": first_number(post_v4_summary.get("learned_only_pass_rate"), 0.0),
            "post_v4_shadow_learned_maturity": post_v4_learned_maturity,
            "post_v4_shadow_heldout_rows": int(first_number(post_v4_summary.get("private_heldout_row_count"), 0)),
            "post_v4_shadow_cap_reached": bool(post_v4_scaling_summary.get("scale_cap_reached") or get_path(state["post_v4_autopilot"], ["summary", "scale_efficiency", "scale_cap_reached"], False)),
            "post_v4_scaling_recommendation": post_v4_scaling_summary.get("recommendation"),
            "v5_private_ecology_pass_rate": first_number(v5_score_summary.get("pass_rate"), 0.0),
            "v5_learned_only_pass_rate": first_number(v5_learned_summary.get("learned_only_pass_rate"), 0.0),
            "v5_learned_maturity": v5_learned_maturity,
            "v5_refresh_green": state["v5_refresh"].get("trigger_state") == "GREEN",
            "v5_refresh_completion": v5_refresh_summary.get("completion_evidence_status"),
            "v5_refresh_fresh": get_path(state["v5_refresh"], ["summary", "freshness", "fresh"], False),
            "v5_green": v5_green,
            "unseen_transfer_challenge_green": unseen_green,
            "unseen_transfer_challenge_pass_rate": first_number(unseen_summary.get("pass_rate"), 0.0),
            "unseen_transfer_challenge_learned_only_pass_rate": first_number(unseen_summary.get("learned_only_pass_rate"), 0.0),
            "unseen_transfer_learned_maturity": unseen_learned_maturity,
            "unseen_transfer_challenge_rows": int(first_number(unseen_summary.get("challenge_row_count"), 0)),
            "unseen_transfer_exact_semantic_replay_count": int(first_number(unseen_summary.get("exact_semantic_key_replay_count"), 0)),
            "residual_frontier_green": frontier_green,
            "residual_frontier_pass_rate": first_number(frontier_summary.get("pass_rate"), 0.0),
            "residual_frontier_only_pass_rate": first_number(frontier_summary.get("frontier_only_pass_rate"), 0.0),
            "residual_frontier_token_pass_count": int(first_number(frontier_summary.get("frontier_token_pass_count"), 0)),
            "residual_frontier_rows": int(first_number(frontier_summary.get("row_count"), 0)),
            "residual_frontier_spec_count": int(first_number(frontier_summary.get("frontier_spec_count"), 0)),
            "contract_blind_transfer_green": contract_blind_green,
            "contract_blind_transfer_rows": int(first_number(contract_blind_summary.get("row_count"), 0)),
            "contract_blind_transfer_pass_rate": first_number(contract_blind_summary.get("pass_rate"), 0.0),
            "contract_blind_transfer_strict_learned_only_pass_rate": first_number(contract_blind_summary.get("strict_novel_learned_only_pass_rate"), 0.0),
            "contract_blind_transfer_learned_token_pass_count": int(first_number(contract_blind_summary.get("learned_token_pass_count"), 0)),
            "contract_blind_transfer_semantic_names_withheld_count": int(first_number(contract_blind_summary.get("semantic_names_withheld_count"), 0)),
            "contract_blind_learned_maturity": contract_blind_learned_maturity,
            "contract_blind_learned_only_task_count": int(first_number(contract_blind_learned_summary.get("learned_only_task_count"), 0)),
            "contract_blind_learned_source_release_fresh": contract_blind_learned_summary.get("decoder_source_release_fresh") is True,
            "contract_blind_exact_train_body_memory_candidate_rows": int(first_number(object_field(contract_blind_learned_summary, "candidate_inventory").get("exact_train_body_memory_candidate_rows"), 0)),
            "learned_token_pass_count_total": sum(learned_counts),
            "prototype_pass_count_total": sum(prototype_counts),
        },
        "safety": {
            "public_tests_used": any_true("public_tests_used", state),
            "public_solutions_used": any_true("public_solutions_used", state),
            "external_inference_calls": external_calls,
            "public_lock": rel(PUBLIC_LOCK),
        },
        "downstream": {
            "architecture_guidance_green": bool(
                architecture_guidance.get("trigger_state") == "GREEN"
                and top_external_calls(architecture_guidance) == 0
                and len(architecture_guidance.get("experiments") or []) > 0
            ),
            "architecture_guidance_experiment_count": len(architecture_guidance.get("experiments") or [])
            if isinstance(architecture_guidance.get("experiments"), list)
            else 0,
            "architecture_guidance_teacher_status": get_path(architecture_guidance, ["teacher", "status"]),
            "architecture_experiment_recommended": get_path(architecture_experiment, ["recommended_next_experiment", "id"]),
            "architecture_experiment_status": get_path(architecture_experiment, ["recommended_next_experiment", "status"]),
            "architecture_change_allowed": bool(architecture_experiment.get("architecture_change_allowed")),
            "teacher_preflight_ok": bool(
                teacher_preflight.get("ok") is True
                and int(first_number(teacher_summary.get("blocker_count"), 999)) == 0
                and teacher_summary.get("apply_mode_blocked") is True
                and teacher_summary.get("worker_teacher_invariant") is True
            ),
            "teacher_preflight_trigger_state": teacher_preflight.get("trigger_state"),
            "teacher_live_status": teacher_summary.get("teacher_live_status"),
            "causal_architecture_delta_green": bool(
                causal_delta.get("trigger_state") == "GREEN"
                and causal_delta.get("status") == "completed_with_capability_delta"
                and causal_delta.get("promotion_evidence") is True
                and causal_summary.get("public_calibration_allowed") is False
                and int(first_number(causal_summary.get("public_task_count"), 0)) == 0
                and top_external_calls(causal_delta) == 0
            ),
            "causal_architecture_delta_status": causal_delta.get("status"),
            "causal_best_target_delta": causal_summary.get("best_target_delta"),
            "causal_private_semantic_test_delta": causal_summary.get("private_semantic_test_passed_task_rate_delta"),
            "causal_private_semantic_positive_family_count": causal_summary.get("private_semantic_positive_family_count"),
            "causal_private_semantic_regressed_family_count": causal_summary.get("private_semantic_regressed_family_count"),
            "causal_public_task_count": causal_summary.get("public_task_count"),
            "student_first_audit_clear": bool(
                student_first.get("trigger_state") in {"GREEN", "YELLOW"}
                and all(row.get("passed") for row in student_first.get("gates", []) if isinstance(row, dict))
                and student_first_summary.get("student_first_public_transfer_valid") is True
                and student_first_summary.get("promotion_allowed_by_evidence") is False
                and top_external_calls(student_first) == 0
            ),
            "student_first_trigger_state": student_first.get("trigger_state"),
            "student_first_public_transfer_valid": student_first_summary.get("student_first_public_transfer_valid"),
            "student_first_public_pass_rate": student_first_summary.get("public_task_pass_rate"),
            "residual_ratchet_green": bool(
                residual_ratchet.get("trigger_state") in {"GREEN", "YELLOW"}
                and ratchet_summary.get("public_calibration_allowed") is False
                and int(first_number(ratchet_summary.get("queue_item_count"), 0)) > 0
                and top_external_calls(residual_ratchet) == 0
            ),
            "residual_ratchet_trigger_state": residual_ratchet.get("trigger_state"),
            "residual_ratchet_decision": ratchet_summary.get("decision"),
            "residual_ratchet_top_action": ratchet_summary.get("top_private_action"),
            "residual_ratchet_queue_item_count": ratchet_summary.get("queue_item_count"),
            "agent_lane_trigger_state": agent_lane.get("trigger_state"),
            "agent_lane_core_transfer_ready": agent_lane_core_ready,
            "agent_lane_breadth_extension_ready": agent_lane_breadth_ready,
            "agent_lane_terminal_tool_use_cases": int(first_number(agent_tool.get("case_count"), 0)),
            "agent_lane_terminal_tool_use_pass_rate": first_number(agent_tool.get("pass_rate"), 0.0),
            "agent_lane_terminal_tool_use_transfer_ready": bool(agent_tool.get("transfer_consumer_ready")),
            "agent_lane_cross_domain_capsule_count": agent_lane_cross_domain_capsules,
            "agent_lane_sts_named_consumer_effect": bool(agent_sts.get("named_consumer_effect")),
            "agent_lane_sts_named_consumer_effect_source": agent_sts.get("named_consumer_effect_source"),
            "agent_lane_conversation_graduated": bool(agent_conversation.get("graduated")),
            "agent_lane_puffer_native_policy_ready": bool(agent_puffer.get("native_policy_ready")),
            "agent_lane_remaining_blockers": agent_blockers,
        },
        "artifacts": {
            "post_v4_queue": rel(POST_V4_QUEUE),
            "v5_queue": rel(V5_QUEUE),
            "unseen_transfer_challenge": rel(UNSEEN_TRANSFER_CHALLENGE),
            "governor_queue": rel(DEFAULT_QUEUE),
            "semantic_alias_gate": rel(SEMANTIC_ALIAS_GATE),
            "novel_composition_gate": rel(NOVEL_COMPOSITION_GATE),
            "v5_refresh": rel(V5_REFRESH),
            "architecture_guidance": rel(ARCH_GUIDANCE_POST_V4),
            "architecture_experiment_governance": rel(ARCH_EXPERIMENT_GOVERNANCE),
            "teacher_preflight": rel(TEACHER_PREFLIGHT),
            "causal_architecture_delta": rel(CAUSAL_ARCHITECTURE_DELTA),
            "student_first_audit": rel(STUDENT_FIRST_AUDIT),
            "residual_ratchet": rel(RESIDUAL_RATCHET),
            "residual_frontier": rel(RESIDUAL_FRONTIER),
            "agent_lane_transfer": rel(AGENT_LANE_TRANSFER),
        },
    }


def build_gates(state: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gate("public_calibration_operator_lock_active", evidence["readiness"]["operator_lock_active"], evidence["readiness"]["public_lock"] if "public_lock" in evidence["readiness"] else rel(PUBLIC_LOCK), "hard"),
        gate("public_calibration_not_allowed_by_reports", not evidence["readiness"]["public_calibration_allowed"], evidence["readiness"]["public_calibration_allowed"], "hard"),
        gate(
            "post_v4_public_artifacts_approved_or_absent",
            bool(evidence["readiness"]["post_v4_public_artifact_state"].get("allowed"))
            or not evidence["readiness"]["post_v4_public_artifacts_present"],
            evidence["readiness"]["post_v4_public_artifact_state"],
            "hard",
        ),
        gate("operator_dry_run_not_executed", not evidence["readiness"]["operator_dry_run_executed"], evidence["readiness"]["operator_dry_run_executed"], "hard"),
        gate("public_tests_and_solutions_not_used", not evidence["safety"]["public_tests_used"] and not evidence["safety"]["public_solutions_used"], evidence["safety"], "hard"),
        gate("external_inference_zero", evidence["safety"]["external_inference_calls"] == 0, evidence["safety"]["external_inference_calls"], "hard"),
        gate("spent_public_calibration_present", evidence["public"]["task_count"] >= 160, evidence["public"], "warning"),
        gate("post_distillation_readiness_current", not evidence["readiness"]["stale"]["post_distillation_readiness"], evidence["readiness"]["stale"], "warning"),
        gate("readiness_packet_current", not evidence["readiness"]["stale"]["readiness_packet"], evidence["readiness"]["stale"], "warning"),
        gate("private_broad_learned_evidence_green", evidence["private"]["broad_green"], evidence["private"], "warning"),
        gate("semantic_alias_private_transfer_green", evidence["private"]["semantic_alias_green"], evidence["private"], "warning"),
        gate("novel_composition_private_transfer_green", evidence["private"]["novel_composition_green"], evidence["private"], "warning"),
        gate("v4_learned_evidence_green", evidence["private"]["v4_learned_green"], evidence["private"], "warning"),
        gate("post_v4_shadow_cap_evidence_green", evidence["private"]["post_v4_shadow_green"] and evidence["private"]["post_v4_shadow_cap_reached"], evidence["private"], "warning"),
        gate("v5_private_ecology_evidence_green", evidence["private"]["v5_green"], evidence["private"], "warning"),
        gate("private_unseen_transfer_challenge_green", evidence["private"]["unseen_transfer_challenge_green"], evidence["private"], "warning"),
        gate("private_residual_frontier_green", evidence["private"]["residual_frontier_green"], evidence["private"], "warning"),
        gate("private_contract_blind_transfer_green", evidence["private"]["contract_blind_transfer_green"], evidence["private"], "warning"),
        gate("prototype_pass_count_zero", evidence["private"]["prototype_pass_count_total"] == 0, evidence["private"]["prototype_pass_count_total"], "warning"),
        gate("architecture_guidance_from_spent_residuals_green", evidence["downstream"]["architecture_guidance_green"], evidence["downstream"], "warning"),
        gate("architecture_experiment_governance_has_recommendation", bool(evidence["downstream"]["architecture_experiment_recommended"]), evidence["downstream"], "warning"),
        gate("teacher_preflight_hard_gates_clear", evidence["downstream"]["teacher_preflight_ok"], evidence["downstream"], "warning"),
        gate("causal_architecture_delta_private_evidence_green", evidence["downstream"]["causal_architecture_delta_green"], evidence["downstream"], "warning"),
        gate("student_first_evidence_audit_clear", evidence["downstream"]["student_first_audit_clear"], evidence["downstream"], "warning"),
        gate("private_residual_self_improvement_ratchet_green", evidence["downstream"]["residual_ratchet_green"], evidence["downstream"], "warning"),
        gate("agent_lane_core_transfer_ready", evidence["downstream"]["agent_lane_core_transfer_ready"], evidence["downstream"], "warning"),
        gate("agent_lane_breadth_extension_ready", evidence["downstream"]["agent_lane_breadth_extension_ready"], evidence["downstream"], "warning"),
        gate("public_transfer_floor_cleared", evidence["public"]["pass_rate"] >= PUBLIC_FLOOR and not evidence["public"]["cards_below_floor"], evidence["public"], "warning"),
    ]


def build_queue(evidence: dict[str, Any], gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed = {row["gate"]: row for row in gates if not row["passed"]}
    rows: list[dict[str, Any]] = []
    if any(row["severity"] == "hard" for row in failed.values()):
        if "public_calibration_operator_lock_active" in failed:
            rows.append(queue_item("restore_public_calibration_operator_lock", "Restore the operator lock before any training or calibration decision.", [], priority=1, safe=False))
        if "post_v4_public_artifacts_approved_or_absent" in failed:
            rows.append(queue_item("quarantine_unapproved_public_calibration_artifacts", "Unapproved post-v4 public artifacts exist; stop and quarantine before continuing.", [], priority=2, safe=False))
        if "public_tests_and_solutions_not_used" in failed or "external_inference_zero" in failed:
            rows.append(queue_item("audit_public_boundary_or_external_inference", "Audit public-boundary and external-inference contamination before using any evidence.", [sys.executable, "scripts/external_inference_audit.py", "--out", "reports/external_inference_audit.json"], priority=3, safe=True))
        return rows

    if "post_distillation_readiness_current" in failed:
        rows.append(queue_item("refresh_post_distillation_readiness_no_execute", "Refresh the public-transfer readiness report without running public calibration.", [sys.executable, "scripts/post_distillation_public_transfer_readiness_v1.py"], priority=10, safe=True))
    if "readiness_packet_current" in failed:
        rows.append(queue_item("refresh_public_readiness_packet_no_execute", "Refresh the operator packet in post-distillation mode without public execution.", [sys.executable, "scripts/public_calibration_readiness_packet.py", "--mode", "post-distillation"], priority=11, safe=True))
    if "private_broad_learned_evidence_green" in failed:
        rows.append(queue_item("refresh_broad_private_learned_distillation", "Rerun the broad private learned-only gate before making transfer decisions.", [sys.executable, "scripts/broad_private_learned_distillation_gate_v1.py"], priority=20, safe=True))
    if "semantic_alias_private_transfer_green" in failed:
        rows.append(queue_item("run_broad_private_semantic_alias_gate", "Run the full private semantic-alias transfer gate so exact semantic-family lookup is not mistaken for broad transfer.", [sys.executable, "scripts/broad_private_semantic_alias_gate_v1.py", "--execute", "--task-limit", "0", "--min-alias-rows", str(SEMANTIC_ALIAS_MIN_ROWS)], priority=20, safe=True))
    if "novel_composition_private_transfer_green" in failed:
        rows.append(queue_item("run_broad_private_novel_composition_gate", "Run the full private novel-composition transfer gate so one-body semantic lookup is not mistaken for reusable reasoning.", [sys.executable, "scripts/broad_private_novel_composition_gate_v1.py", "--execute", "--rows", str(NOVEL_COMPOSITION_MIN_ROWS), "--min-rows", str(NOVEL_COMPOSITION_MIN_ROWS)], priority=20, safe=True))
    if "v4_learned_evidence_green" in failed:
        rows.append(
            queue_item(
                "refresh_v4_learned_distillation",
                "Rerun the v4 learned-only gate with structural and train-novelty checks before making transfer decisions.",
                [
                    sys.executable,
                    "scripts/broad_private_learned_distillation_gate_v1.py",
                    "--heldout",
                    "data/training_data/high_transfer/private_eval/public_safe_broad_transfer_maturity_v4_heldout_code_lm_tasks.jsonl",
                    "--candidates",
                    "reports/code_lm_private_candidates_public_safe_broad_transfer_maturity_v4_heldout.jsonl",
                    "--control-candidates",
                    "reports/code_lm_private_candidates_public_safe_broad_transfer_maturity_v4_heldout_sts_off.jsonl",
                    "--score",
                    "reports/public_safe_broad_transfer_maturity_v4_score.json",
                    "--private-train",
                    "data/training_data/high_transfer/private_train/public_safe_broad_transfer_maturity_v4_code_lm_tasks.jsonl",
                    "--learned-only-candidates-out",
                    "reports/code_lm_private_candidates_public_safe_broad_transfer_maturity_v4_heldout_learned_only.jsonl",
                    "--learned-only-score-out",
                    "reports/public_safe_broad_transfer_maturity_v4_learned_only_score.json",
                    "--learned-only-score-markdown-out",
                    "reports/public_safe_broad_transfer_maturity_v4_learned_only_score.md",
                    "--min-heldout-rows",
                    "1008",
                    "--out",
                    "reports/public_safe_broad_transfer_maturity_v4_learned_distillation_gate.json",
                    "--markdown-out",
                    "reports/public_safe_broad_transfer_maturity_v4_learned_distillation_gate.md",
                ],
                priority=21,
                safe=True,
            )
        )
    if "post_v4_shadow_cap_evidence_green" in failed:
        rows.append(
            queue_item(
                "refresh_post_v4_shadow_private_autopilot",
                "Refresh post-v4 private shadow candidates, score, learned-only gate, and queue without running public calibration.",
                [
                    sys.executable,
                    "scripts/post_v4_generalization_autopilot_v1.py",
                    "--execute",
                    "--train-rows",
                    "12000",
                    "--heldout-rows",
                    "2400",
                    "--private-eval-limit",
                    "2400",
                    "--max-hours",
                    "8",
                ],
                priority=22,
                safe=True,
            )
        )
    if "v5_private_ecology_evidence_green" in failed:
        rows.append(
            queue_item(
                "refresh_private_ecology_v5_full_gate",
                "Refresh v5 private ecology rows, STS streams, fanout, score, and learned-only gate without public calibration.",
                [
                    sys.executable,
                    "scripts/private_ecology_generalization_v5_refresh.py",
                    "--execute",
                    "--train-rows",
                    "1200",
                    "--heldout-rows",
                    "480",
                    "--private-eval-limit",
                    "480",
                    "--max-hours",
                    "6",
                ],
                priority=23,
                safe=True,
            )
        )
    if "private_unseen_transfer_challenge_green" in failed:
        rows.append(
            queue_item(
                "run_private_unseen_transfer_challenge",
                "Run a private OOD transfer challenge with exact semantic keys withheld before any operator public-review step.",
                [
                    sys.executable,
                    "scripts/private_unseen_transfer_challenge_v1.py",
                    "--execute",
                    "--rows",
                    "120",
                    "--max-hours",
                    "2",
                ],
                priority=24,
                safe=True,
            )
        )
    if "private_residual_frontier_green" in failed:
        rows.append(
            queue_item(
                "run_private_residual_frontier",
                "Run private residual-frontier composition pressure from aggregate public residual categories; public calibration remains locked.",
                [
                    sys.executable,
                    "scripts/private_residual_frontier_v1.py",
                    "--execute",
                    "--rows",
                    str(RESIDUAL_FRONTIER_MIN_ROWS),
                    "--min-rows",
                    str(RESIDUAL_FRONTIER_MIN_ROWS),
                    "--max-hours",
                    "6",
                ],
                priority=25,
                safe=True,
            )
        )
    if "private_residual_self_improvement_ratchet_green" in failed:
        rows.append(private_residual_ratchet_item())
    if "agent_lane_core_transfer_ready" in failed:
        rows.append(agent_lane_core_item())
    if "agent_lane_breadth_extension_ready" in failed:
        rows.append(agent_lane_breadth_item(evidence))

    if not rows:
        if not evidence["downstream"]["architecture_guidance_green"]:
            rows.append(architecture_guidance_item())
        if not evidence["downstream"]["architecture_experiment_recommended"]:
            rows.append(architecture_experiment_item())
        if not evidence["downstream"]["teacher_preflight_ok"]:
            rows.append(teacher_preflight_item())
        if (
            evidence["downstream"]["architecture_guidance_green"]
            and evidence["downstream"]["architecture_experiment_recommended"]
            and evidence["downstream"]["teacher_preflight_ok"]
            and not evidence["downstream"]["causal_architecture_delta_green"]
        ):
            rows.append(
                queue_item(
                    "run_private_causal_architecture_delta",
                    "Run the private-only same-seed architecture delta loop; this does not execute public calibration.",
                    [
                        sys.executable,
                        "scripts/causal_architecture_delta_loop.py",
                        "--execute-ablation",
                        "--task-limit",
                        "24",
                        "--candidates-per-task",
                        "4",
                    ],
                    priority=60,
                    safe=True,
                )
            )
        if (
            evidence["downstream"]["architecture_guidance_green"]
            and evidence["downstream"]["architecture_experiment_recommended"]
            and evidence["downstream"]["teacher_preflight_ok"]
            and evidence["downstream"]["causal_architecture_delta_green"]
            and not evidence["downstream"]["student_first_audit_clear"]
        ):
            rows.append(
                queue_item(
                    "student_first_evidence_audit",
                    "Audit that public-transfer claims are still student-generated and not adapter/template/loop shortcuts.",
                    [
                        sys.executable,
                        "scripts/student_first_evidence_audit.py",
                        "--out",
                        "reports/student_first_evidence_audit.json",
                        "--markdown-out",
                        "reports/student_first_evidence_audit.md",
                    ],
                    priority=70,
                    safe=True,
                )
            )
        if (
            evidence["downstream"]["architecture_guidance_green"]
            and evidence["downstream"]["architecture_experiment_recommended"]
            and evidence["downstream"]["teacher_preflight_ok"]
            and evidence["downstream"]["causal_architecture_delta_green"]
            and evidence["downstream"]["student_first_audit_clear"]
            and evidence["private"]["unseen_transfer_challenge_green"]
            and evidence["private"]["residual_frontier_green"]
            and evidence["downstream"]["agent_lane_breadth_extension_ready"]
            and evidence["downstream"]["residual_ratchet_green"]
            and evidence["downstream"]["residual_ratchet_decision"] == "retry_private"
            and bool(evidence["downstream"]["residual_ratchet_top_action"])
        ):
            rows.append(
                queue_item(
                    "execute_private_residual_self_improvement_queue",
                    "The residual ratchet says broad public transfer is still below floor; execute one safe private residual action and refresh the governor.",
                    [
                        sys.executable,
                        "scripts/private_residual_self_improvement_ratchet_v1.py",
                        "--execute",
                        "--max-actions",
                        "1",
                    ],
                    priority=80,
                    safe=True,
                )
            )
        if (
            evidence["downstream"]["architecture_guidance_green"]
            and evidence["downstream"]["architecture_experiment_recommended"]
            and evidence["downstream"]["teacher_preflight_ok"]
            and evidence["downstream"]["causal_architecture_delta_green"]
            and evidence["downstream"]["student_first_audit_clear"]
            and evidence["private"]["unseen_transfer_challenge_green"]
            and evidence["private"]["residual_frontier_green"]
            and evidence["downstream"]["agent_lane_breadth_extension_ready"]
            and (
                not evidence["downstream"]["residual_ratchet_green"]
                or evidence["downstream"]["residual_ratchet_decision"] != "retry_private"
                or not evidence["downstream"]["residual_ratchet_top_action"]
            )
        ):
            rows.append(
                queue_item(
                    "operator_review_public_calibration_locked",
                    "Private learned evidence is current, but public calibration remains locked and below floor; explicit operator approval is required before any bounded public run.",
                    [],
                    priority=90,
                    safe=False,
                    requires_operator_public_unlock=True,
                )
            )
    return rows


def architecture_guidance_item() -> dict[str, Any]:
    return queue_item(
        "architecture_guidance_from_spent_public_residuals_no_teacher",
        "Convert the spent public residuals into private architecture experiment specs; do not call teacher and do not execute public calibration.",
        [
            sys.executable,
            "scripts/architecture_guidance_loop.py",
            "--real-code-report",
            rel(PUBLIC_CALIBRATION),
            "--trace-in",
            rel(PUBLIC_TRACES),
            "--out",
            "reports/architecture_guidance_loop_post_v4_generalization.json",
            "--markdown-out",
            "reports/architecture_guidance_loop_post_v4_generalization.md",
            "--teacher-prompt-out",
            "reports/teacher_architecture_guidance_prompt_post_v4_generalization.md",
            "--experiments-out",
            "reports/architecture_guided_experiments_post_v4_generalization.json",
        ],
        priority=30,
        safe=True,
    )


def architecture_experiment_item() -> dict[str, Any]:
    return queue_item(
        "architecture_experiment_governance_refresh",
        "Rank bounded decoder/architecture experiments from current evidence before more private row generation.",
        [
            sys.executable,
            "scripts/architecture_experiment_governor.py",
            "--out",
            "reports/architecture_experiment_governance.json",
            "--markdown-out",
            "reports/architecture_experiment_governance.md",
        ],
        priority=40,
        safe=True,
    )


def teacher_preflight_item() -> dict[str, Any]:
    return queue_item(
        "teacher_preflight_proposal_only",
        "Verify the teacher path is proposal-only and worker chunks remain teacher-free before any architecture-wall escalation.",
        [
            sys.executable,
            "scripts/full_training_teacher_preflight.py",
            "--profile",
            "smoke",
            "--skip-autonomy-readiness",
            "--out",
            "reports/full_training_teacher_preflight.json",
            "--markdown-out",
            "reports/full_training_teacher_preflight.md",
        ],
        priority=50,
        safe=True,
    )


def private_residual_ratchet_item() -> dict[str, Any]:
    return queue_item(
        "refresh_private_residual_self_improvement_ratchet",
        "Build the current private residual self-improvement queue from aggregate residual summaries without running public calibration.",
        [
            sys.executable,
            "scripts/private_residual_self_improvement_ratchet_v1.py",
            "--out",
            "reports/private_residual_self_improvement_ratchet_v1.json",
            "--markdown-out",
            "reports/private_residual_self_improvement_ratchet_v1.md",
            "--queue-out",
            "reports/private_residual_self_improvement_ratchet_v1_queue.jsonl",
        ],
        priority=75,
        safe=True,
    )


def agent_lane_core_item() -> dict[str, Any]:
    return queue_item(
        "refresh_agent_lane_tool_use_and_capsules",
        "Refresh private terminal/tool-use transfer, cross-domain STS capsules, and the agent-lane gate before public-review decisions.",
        [
            sys.executable,
            "scripts/agent_lane_private_refresh.py",
            "--max-tool-cases",
            "64",
            "--max-capsules",
            "256",
        ],
        priority=81,
        safe=True,
    )


def agent_lane_breadth_item(evidence: dict[str, Any]) -> dict[str, Any]:
    blockers = set(evidence["downstream"].get("agent_lane_remaining_blockers") or [])
    if "conversation_transfer_not_graduated" in blockers:
        return queue_item(
            "run_conversation_hard_v4_agent_transfer",
            "Run the private hard-v4 conversation lane so tool-use/state-memory capsules transfer into user-facing agent behavior.",
            [
                sys.executable,
                "scripts/multi_turn_conversation_benchmark.py",
                "--suite-mode",
                "hard_v4",
                "--out",
                "reports/high_transfer_multi_turn_conversation_hard_v4.json",
                "--markdown-out",
                "reports/high_transfer_multi_turn_conversation_hard_v4.md",
                "--workers",
                "4",
            ],
            priority=82,
            safe=True,
        )
    return queue_item(
        "run_pufferlib_rl_agent_transfer",
        "Run the private RL lane so legal-action masking, reward credit, and rollout memory join the agent transfer frontier.",
        [
            sys.executable,
            "scripts/pufferlib4_rl_lane.py",
            "--out",
            "reports/pufferlib4_rl_lane.json",
            "--markdown-out",
            "reports/pufferlib4_rl_lane.md",
        ],
        priority=83,
        safe=True,
    )


def queue_item(
    kind: str,
    title: str,
    command: list[str],
    *,
    priority: int,
    safe: bool,
    requires_operator_public_unlock: bool = False,
) -> dict[str, Any]:
    return {
        "policy": "project_theseus_generalization_governor_queue_item_v1",
        "queue": "theseus_generalization_governor_v1",
        "kind": kind,
        "title": title,
        "priority": priority,
        "status": "pending",
        "command": command,
        "public_calibration_allowed": False,
        "safe_to_execute_without_operator_public_approval": bool(safe and not requires_operator_public_unlock),
        "requires_operator_public_unlock": bool(requires_operator_public_unlock),
    }


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    failed = [row for row in report.get("gates", []) if not row.get("passed")]
    lines = [
        "# Theseus Generalization Governor v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- wall: `{get_path(report, ['wall', 'name'])}` / `{get_path(report, ['wall', 'status'])}`",
        f"- public_pass_rate: `{summary.get('public_pass_rate')}` floor=`{summary.get('public_floor')}` tasks=`{summary.get('public_task_count')}`",
        f"- cards_below_floor: `{summary.get('cards_below_floor')}`",
        f"- operator_lock_active: `{summary.get('operator_lock_active')}`",
        f"- public_calibration_allowed: `{summary.get('public_calibration_allowed')}`",
        f"- private_broad_green: `{summary.get('private_broad_green')}`",
        f"- semantic_alias_green: `{summary.get('semantic_alias_green')}`",
        f"- semantic_alias_inferred_token_pass_count: `{summary.get('semantic_alias_inferred_token_pass_count')}`",
        f"- novel_composition_green: `{summary.get('novel_composition_green')}`",
        f"- novel_composition_only_pass_rate: `{summary.get('novel_composition_only_pass_rate')}`",
        f"- novel_composition_token_pass_count: `{summary.get('novel_composition_token_pass_count')}`",
        f"- v4_learned_green: `{summary.get('v4_learned_green')}`",
        f"- v4_learned_maturity_ready: `{summary.get('v4_learned_maturity_ready')}` body_overlap=`{summary.get('v4_learned_train_body_overlap_rate')}`",
        f"- post_v4_shadow_cap_reached: `{summary.get('post_v4_shadow_cap_reached')}`",
        f"- post_v4_shadow_learned_maturity_ready: `{summary.get('post_v4_shadow_learned_maturity_ready')}` body_overlap=`{summary.get('post_v4_shadow_train_body_overlap_rate')}`",
        f"- v5_private_ecology_green: `{summary.get('v5_private_ecology_green')}`",
        f"- v5_learned_maturity_ready: `{summary.get('v5_learned_maturity_ready')}` body_overlap=`{summary.get('v5_train_body_overlap_rate')}`",
        f"- unseen_transfer_learned_maturity_ready: `{summary.get('unseen_transfer_learned_maturity_ready')}` body_overlap=`{summary.get('unseen_transfer_train_body_overlap_rate')}`",
        f"- residual_frontier_green: `{summary.get('residual_frontier_green')}`",
        f"- residual_frontier_only_pass_rate: `{summary.get('residual_frontier_only_pass_rate')}`",
        f"- residual_frontier_token_pass_count: `{summary.get('residual_frontier_token_pass_count')}`",
        f"- residual_frontier_spec_count: `{summary.get('residual_frontier_spec_count')}`",
        f"- contract_blind_transfer_green: `{summary.get('contract_blind_transfer_green')}` rows=`{summary.get('contract_blind_transfer_rows')}` strict_learned=`{summary.get('contract_blind_transfer_strict_learned_only_pass_rate')}`",
        f"- contract_blind_learned_maturity_ready: `{summary.get('contract_blind_learned_maturity_ready')}` ast_shapes=`{summary.get('contract_blind_pass_ast_shape_count')}` unique_ast=`{summary.get('contract_blind_pass_normalized_ast_unique_count')}` body_overlap=`{summary.get('contract_blind_train_body_overlap_rate')}`",
        f"- architecture_guidance_green: `{summary.get('architecture_guidance_green')}`",
        f"- architecture_experiment_recommended: `{summary.get('architecture_experiment_recommended')}`",
        f"- teacher_preflight_ok: `{summary.get('teacher_preflight_ok')}`",
        f"- causal_architecture_delta_green: `{summary.get('causal_architecture_delta_green')}`",
        f"- causal_best_target_delta: `{summary.get('causal_best_target_delta')}`",
        f"- causal_private_semantic_test_delta: `{summary.get('causal_private_semantic_test_delta')}`",
        f"- causal_private_semantic_positive_family_count: `{summary.get('causal_private_semantic_positive_family_count')}`",
        f"- causal_private_semantic_regressed_family_count: `{summary.get('causal_private_semantic_regressed_family_count')}`",
        f"- student_first_audit_clear: `{summary.get('student_first_audit_clear')}`",
        f"- residual_ratchet_green: `{summary.get('residual_ratchet_green')}`",
        f"- residual_ratchet_decision: `{summary.get('residual_ratchet_decision')}`",
        f"- residual_ratchet_top_action: `{summary.get('residual_ratchet_top_action')}`",
        f"- prototype_pass_count_total: `{summary.get('prototype_pass_count_total')}`",
        f"- learned_token_pass_count_total: `{summary.get('learned_token_pass_count_total')}`",
        f"- external_inference_calls: `{summary.get('external_inference_calls')}`",
        "",
        "## Failed Gates",
        "",
    ]
    if failed:
        for row in failed:
            lines.append(f"- `{row.get('gate')}` ({row.get('severity')})")
    else:
        lines.append("- None.")
    lines.extend(["", "## Queue", ""])
    for row in report.get("queue", []) or []:
        command = " ".join(str(item) for item in row.get("command") or [])
        if command:
            lines.append(f"- `{row.get('kind')}`: {row.get('title')} Command: `{command}`")
        else:
            lines.append(f"- `{row.get('kind')}`: {row.get('title')}")
    lines.extend(["", "## Rules", ""])
    for key, value in (report.get("rules") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def any_true(key: str, state: dict[str, Any]) -> bool:
    for value in state.values():
        if isinstance(value, dict):
            if value.get(key) is True:
                return True
            summary = object_field(value, "summary")
            if summary.get(key) is True:
                return True
    return False


def top_external_calls(report: dict[str, Any]) -> int:
    try:
        total = int(report.get("external_inference_calls") or 0)
    except (TypeError, ValueError):
        total = 0
    summary = object_field(report, "summary")
    try:
        total += int(summary.get("external_inference_calls") or 0)
    except (TypeError, ValueError):
        pass
    return total


def learned_maturity(summary: dict[str, Any]) -> dict[str, Any]:
    structure = object_field(summary, "learned_structural_inventory")
    novelty = object_field(summary, "learned_train_novelty_inventory")
    normalized_ast_unique_count = int(first_number(structure.get("pass_normalized_ast_unique_count"), 0))
    min_normalized_ast_unique_count = int(first_number(structure.get("min_pass_normalized_ast_unique_count"), 999))
    ast_shape_count = int(first_number(structure.get("pass_ast_shape_count"), 0))
    min_ast_shape_count = int(first_number(structure.get("min_pass_ast_shape_count"), 999))
    top_duplicate_rate = first_number(structure.get("pass_top_normalized_ast_duplicate_rate"), 1.0)
    max_top_duplicate_rate = first_number(structure.get("max_pass_top_duplicate_rate"), 0.0)
    structural_ready = bool(
        normalized_ast_unique_count >= min_normalized_ast_unique_count
        and ast_shape_count >= min_ast_shape_count
        and top_duplicate_rate <= max_top_duplicate_rate
        and structure.get("control_structure_coverage_ready") is True
    )
    novelty_ready = novelty.get("novelty_ready") is True
    fresh = summary.get("decoder_source_release_fresh") is True
    prototype_pass_count = int(first_number(summary.get("prototype_pass_count"), 999))
    return {
        "ready": bool(fresh and structural_ready and novelty_ready and prototype_pass_count == 0),
        "fresh": fresh,
        "structural_ready": structural_ready,
        "novelty_ready": novelty_ready,
        "prototype_pass_count": prototype_pass_count,
        "pass_normalized_ast_unique_count": normalized_ast_unique_count,
        "min_pass_normalized_ast_unique_count": min_normalized_ast_unique_count,
        "pass_ast_shape_count": ast_shape_count,
        "min_pass_ast_shape_count": min_ast_shape_count,
        "pass_top_duplicate_rate": top_duplicate_rate,
        "max_pass_top_duplicate_rate": max_top_duplicate_rate,
        "train_ast_overlap_rate": first_number(novelty.get("exact_train_normalized_ast_overlap_rate"), 1.0),
        "train_body_overlap_rate": first_number(novelty.get("exact_train_body_normalized_ast_overlap_rate"), 1.0),
        "private_train": novelty.get("private_train") or "",
        "completion_evidence_status": summary.get("completion_evidence_status") or "",
    }


def stale_report(path: Path, stale_seconds: int) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) > stale_seconds
    except OSError:
        return True


def post_v4_public_artifact_state() -> dict[str, Any]:
    present = [rel(path) for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()]
    if not present:
        return {
            "allowed": True,
            "mode": "absent",
            "present_artifacts": [],
            "approval_valid": False,
            "execute_report_valid": False,
            "required_outputs_present": False,
            "operator_lock_active": PUBLIC_LOCK.exists(),
            "rules": "post-v4 public artifacts may exist only after the approved one-shot calibration completed and relocked",
        }
    approval = read_json(OPERATOR_APPROVAL, {})
    execute = read_json(OPERATOR_EXECUTE, {})
    approval_valid = (
        approval.get("policy") == "project_theseus_public_calibration_operator_approval_v1"
        and approval.get("approved") is True
        and approval.get("proposed_slug") == "post_v4_seed23_5x32"
        and int(first_number(approval.get("max_runs"), 0)) == 1
    )
    execute_valid = (
        execute.get("policy") == "project_theseus_operator_bounded_public_calibration_v1"
        and execute.get("trigger_state") == "GREEN"
        and get_path(execute, ["summary", "executed"]) is True
        and get_path(execute, ["summary", "proposed_slug"]) == "post_v4_seed23_5x32"
        and get_path(execute, ["summary", "output_exists_after"]) is True
        and get_path(execute, ["summary", "operator_lock_present_after"]) is True
        and int(first_number(get_path(execute, ["summary", "run_returncode"]), -1)) == 0
    )
    required_outputs_present = all(path.exists() for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS[:3])
    allowed = approval_valid and execute_valid and required_outputs_present and PUBLIC_LOCK.exists()
    return {
        "allowed": allowed,
        "mode": "approved_spent_one_shot" if allowed else "unapproved_or_incomplete",
        "present_artifacts": present,
        "approval_valid": approval_valid,
        "execute_report_valid": execute_valid,
        "required_outputs_present": required_outputs_present,
        "operator_lock_active": PUBLIC_LOCK.exists(),
        "rules": "post-v4 public artifacts may exist only after the approved one-shot calibration completed and relocked",
    }


def read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return value if isinstance(value, dict) else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    field = value.get(key) if isinstance(value, dict) else {}
    return field if isinstance(field, dict) else {}


def list_field(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def get_path(value: Any, path: list[Any], default: Any = None) -> Any:
    cur = value
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(value).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
