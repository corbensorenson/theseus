#!/usr/bin/env python3
"""No-execute public transfer readiness refresh.

This report freezes the next public benchmark contract and reconciles it with
fresh private transfer evidence. It never runs public calibration and never
reads public benchmark tests or solutions beyond the existing calibration
metadata already governed by the benchmark loaders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_CONTRACT = ROOT / "configs" / "public_benchmark_contract_v1.json"
DEFAULT_OUT = REPORTS / "public_transfer_readiness_refresh_v1.json"
DEFAULT_MARKDOWN = REPORTS / "public_transfer_readiness_refresh_v1.md"
OPERATOR_REVIEW_FAMILY = "operator_reviewed_bounded_public_calibration_packet"
OPERATOR_REVIEW_REASON = "operator_review_required"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=rel(DEFAULT_CONTRACT))
    parser.add_argument("--post-readiness", default="reports/post_distillation_public_transfer_readiness_v1.json")
    parser.add_argument("--capacity-dry-run", default="reports/broad_transfer_closure_runner_public_contract_capacity_dry_run.json")
    parser.add_argument("--public-readiness-packet", default="reports/public_calibration_readiness_packet.json")
    parser.add_argument("--operator-dry-run", default="reports/operator_bounded_public_calibration_dry_run.json")
    parser.add_argument("--sts-ranker", default="reports/sts_ranker_policy_v1.json")
    parser.add_argument("--structural-action", default="reports/neural_seed_structural_action_ablation_report.json")
    parser.add_argument("--full-body-admissibility", default="reports/private_full_body_candidate_admissibility_gate_v1.json")
    parser.add_argument("--full-body-semantic-ablation", default="reports/private_full_body_semantic_quality_ablation_v1.json")
    parser.add_argument("--maturity-audit", default="reports/maturity_integrity_audit.json")
    parser.add_argument("--governor", default="reports/theseus_generalization_governor_v1.json")
    parser.add_argument("--private-residual-target-consumer", default="reports/private_residual_target_consumer_v1.json")
    parser.add_argument("--alignment-preflight", default="reports/public_calibration_alignment_preflight.json")
    parser.add_argument("--operator-lock", default="reports/public_calibration_operator_lock.flag")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = resolve(args.contract)
    post_path = resolve(args.post_readiness)
    capacity_path = resolve(args.capacity_dry_run)
    packet_path = resolve(args.public_readiness_packet)
    operator_dry_run_path = resolve(args.operator_dry_run)
    sts_ranker_path = resolve(args.sts_ranker)
    structural_path = resolve(args.structural_action)
    full_body_path = resolve(args.full_body_admissibility)
    semantic_ablation_path = resolve(args.full_body_semantic_ablation)
    maturity_path = resolve(args.maturity_audit)
    governor_path = resolve(args.governor)
    private_residual_target_consumer_path = resolve(args.private_residual_target_consumer)
    alignment_preflight_path = resolve(args.alignment_preflight)
    lock_path = resolve(args.operator_lock)

    contract = read_json(contract_path, {})
    post = read_json(post_path, {})
    capacity = read_json(capacity_path, {})
    packet = read_json(packet_path, {})
    operator_dry_run = read_json(operator_dry_run_path, {})
    sts_ranker = read_json(sts_ranker_path, {})
    structural = read_json(structural_path, {})
    full_body = read_json(full_body_path, {})
    semantic_ablation = read_json(semantic_ablation_path, {})
    maturity = read_json(maturity_path, {})
    governor = read_json(governor_path, {})
    private_residual_target_consumer = read_json(private_residual_target_consumer_path, {})
    alignment_preflight = read_json(alignment_preflight_path, {})

    post_summary = object_field(post, "summary")
    capacity_summary = object_field(capacity, "summary")
    sts_summary = object_field(sts_ranker, "summary")
    structural_summary = object_field(structural, "summary")
    full_body_summary = object_field(full_body, "summary")
    semantic_ablation_summary = object_field(semantic_ablation, "summary")
    maturity_summary = object_field(maturity, "summary")
    governor_summary = object_field(governor, "summary")
    private_residual_target_consumer_summary = object_field(private_residual_target_consumer, "summary")
    alignment_summary = object_field(alignment_preflight, "summary")
    contract_stage = object_field(contract, "stage_1_code_generation_surface")

    hard_gates = [
        gate("contract_file_present", contract_path.exists(), rel(contract_path)),
        gate("contract_policy_valid", contract.get("policy") == "project_theseus_public_benchmark_contract_v1", contract.get("policy")),
        gate("public_operator_lock_active", lock_path.exists(), rel(lock_path)),
        gate("post_readiness_hard_gates_clear", post.get("trigger_state") in {"YELLOW", "GREEN"} and not list_field(post.get("hard_blockers")), {
            "path": rel(post_path),
            "trigger_state": post.get("trigger_state"),
            "hard_blocker_count": len(list_field(post.get("hard_blockers"))),
        }),
        gate("fresh_broad_private_current", post_summary.get("decoder_source_release_fresh") is True, {
            "decoder_source_release_fresh": post_summary.get("decoder_source_release_fresh"),
            "stale_reasons": post_summary.get("decoder_source_release_stale_reasons"),
        }),
        gate("capacity_dry_run_not_execute", capacity_summary.get("execute") is False, {
            "path": rel(capacity_path),
            "execute": capacity_summary.get("execute"),
            "trigger_state": capacity.get("trigger_state"),
        }),
        gate("public_capacity_sufficient_for_contract", capacity_gate_passed(capacity), {
            "required_per_card": contract_stage.get("cases_per_card"),
            "cards": contract_stage.get("cards"),
            "capacity": capacity.get("capacity"),
        }),
        gate("no_public_calibration_executed_by_refresh", True, "this script only reads reports and writes readiness artifacts"),
        gate("external_inference_zero", external_inference_zero(post, capacity, sts_ranker, structural, full_body, semantic_ablation, maturity, governor, private_residual_target_consumer, alignment_preflight), {
            "post_readiness": external_calls(post),
            "capacity": external_calls(capacity),
            "sts_ranker": external_calls(sts_ranker),
            "structural": external_calls(structural),
            "full_body_admissibility": external_calls(full_body),
            "full_body_semantic_ablation": external_calls(semantic_ablation),
            "maturity": external_calls(maturity),
            "governor": external_calls(governor),
            "private_residual_target_consumer": external_calls(private_residual_target_consumer),
            "alignment_preflight": external_calls(alignment_preflight),
        }),
    ]

    evidence_gates = [
        gate("public_calibration_alignment_preflight_ready", alignment_preflight_ready(alignment_preflight), {
            "path": rel(alignment_preflight_path),
            "trigger_state": alignment_preflight.get("trigger_state"),
            "alignment_preflight_ready": alignment_preflight.get("alignment_preflight_ready"),
            "case_manifest": alignment_summary.get("case_manifest"),
            "case_manifest_row_count": alignment_summary.get("case_manifest_row_count"),
            "case_manifest_card_counts": alignment_summary.get("case_manifest_card_counts"),
            "candidate_manifest_bound_to_case_manifest": alignment_summary.get("candidate_manifest_bound_to_case_manifest"),
            "candidate_manifest_preexists_before_run": alignment_summary.get("candidate_manifest_preexists_before_run"),
            "training_rows_written": alignment_summary.get("training_rows_written"),
            "external_inference_calls": alignment_summary.get("external_inference_calls"),
        }),
        gate("sts_ranker_policy_green", sts_ranker.get("trigger_state") == "GREEN", {
            "path": rel(sts_ranker_path),
            "trigger_state": sts_ranker.get("trigger_state"),
            "selected_pass_rate": first_present(sts_summary, "selected_pass_rate", "sts_policy_selected_pass_rate"),
            "matched_non_sts_pass_rate": first_present(
                sts_summary,
                "matched_non_sts_pass_rate",
                "non_sts_policy_selected_pass_rate",
            ),
        }),
        gate("sts_ranker_no_fallback_or_public_leakage", (
            int_number(sts_summary.get("fallback_return_candidate_count")) == 0
            and int_number(sts_summary.get("public_leakage_count")) == 0
        ), {
            "fallback_return_candidate_count": sts_summary.get("fallback_return_candidate_count"),
            "public_leakage_count": sts_summary.get("public_leakage_count"),
        }),
        gate("structural_action_decoder_green", structural.get("trigger_state") == "GREEN", {
            "path": rel(structural_path),
            "trigger_state": structural.get("trigger_state"),
            "symliquid_pass_rate": first_present(
                structural_summary,
                "symliquid_pass_rate",
                "symliquid_sts_on_pass_rate",
            ),
            "transformer_pass_rate": first_present(
                structural_summary,
                "transformer_pass_rate",
                "transformer_sts_on_pass_rate",
            ),
        }),
        gate("structural_action_no_fallback_public_or_teacher", structural_no_fallback_public_teacher(structural), structural_gate_snapshot(structural)),
        gate("full_body_candidate_admissibility_green", full_body_admissibility_current(full_body), {
            "path": rel(full_body_path),
            "trigger_state": full_body.get("trigger_state"),
            "candidate_row_count": full_body_summary.get("candidate_row_count"),
            "full_body_token_candidate_count": full_body_summary.get("full_body_token_candidate_count"),
            "benchmark_promotion_eligible_candidate_count": full_body_summary.get("benchmark_promotion_eligible_candidate_count"),
            "no_admissible_task_rate": full_body_summary.get("no_admissible_task_rate"),
            "fallback_return_candidate_count": full_body_summary.get("fallback_return_candidate_count"),
            "template_like_candidate_count": full_body_summary.get("template_like_candidate_count"),
            "public_leakage_count": full_body_summary.get("public_leakage_count"),
        }),
        gate("full_body_semantic_quality_green", full_body_semantic_quality_current(semantic_ablation), {
            "path": rel(semantic_ablation_path),
            "trigger_state": semantic_ablation.get("trigger_state"),
            "best_selected_pass_rate": semantic_ablation_summary.get("best_private_public_shaped_selected_pass_rate"),
            "best_pass_if_any_rate": semantic_ablation_summary.get("best_private_public_shaped_pass_if_any_rate"),
            "post_v4_default_semantic_dead": first_present(
                semantic_ablation_summary,
                "post_v4_default_semantic_dead",
                "post_v4_token_beam_semantic_dead",
            ),
            "v4_full_body_selected_pass_rate": semantic_ablation_summary.get("v4_full_body_selected_pass_rate"),
            "v4_strict_novel_learned_only_pass_rate": semantic_ablation_summary.get("v4_strict_novel_learned_only_pass_rate"),
            "fallback_return_candidate_count": semantic_ablation_summary.get("fallback_return_candidate_count"),
            "public_leakage_count": semantic_ablation_summary.get("public_leakage_count"),
        }),
        gate("maturity_audit_no_hard_blockers", not list_field(maturity.get("hard_blockers")) and int_number(maturity_summary.get("hard_blocker_count")) == 0, {
            "path": rel(maturity_path),
            "trigger_state": maturity.get("trigger_state"),
            "hard_blockers": maturity.get("hard_blockers"),
            "hard_blocker_count": maturity_summary.get("hard_blocker_count"),
            "maturity_blockers": first_present(maturity_summary, "maturity_blockers") or maturity.get("maturity_blockers"),
        }),
        gate("promotion_integrity_receipt_ready", maturity_summary.get("promotion_integrity_ready") is True, {
            "path": rel(maturity_path),
            "source": maturity_summary.get("promotion_integrity_source"),
            "promotion_integrity_ready": maturity_summary.get("promotion_integrity_ready"),
            "promotion_integrity_verified_candidate_count": maturity_summary.get("promotion_integrity_verified_candidate_count"),
            "promotion_integrity_viea_record_count": maturity_summary.get("promotion_integrity_viea_record_count"),
            "rule": "public-transfer readiness inherits the independent candidate-promotion integrity receipt from the maturity audit",
        }),
        gate("governor_no_hard_failures", int_number(governor_summary.get("hard_failed_gate_count")) == 0, {
            "path": rel(governor_path),
            "trigger_state": governor.get("trigger_state"),
            "hard_failed_gate_count": governor_summary.get("hard_failed_gate_count"),
            "warning_failed_gate_count": governor_summary.get("warning_failed_gate_count"),
        }),
        gate("operator_dry_run_did_not_execute", object_field(operator_dry_run, "summary").get("mode") in {None, "dry_run"}, {
            "path": rel(operator_dry_run_path),
            "mode": object_field(operator_dry_run, "summary").get("mode"),
            "executed": object_field(operator_dry_run, "summary").get("executed"),
        }),
        gate("public_readiness_packet_not_execute", packet.get("public_calibration_allowed") is False, {
            "path": rel(packet_path),
            "trigger_state": packet.get("trigger_state"),
            "public_calibration_allowed": packet.get("public_calibration_allowed"),
        }),
    ]

    failed_private_fix_evidence = post_summary.get("failed_private_fix_evidence")
    private_residual_unresolved_count = int_number(
        private_residual_target_consumer_summary.get("unresolved_target_count")
    )
    private_residual_current = private_residual_fix_current(failed_private_fix_evidence) and private_residual_unresolved_count == 0
    private_fix_review_ready = bool(private_residual_current and alignment_preflight_ready(alignment_preflight))
    public_transfer_ready = post.get("public_transfer_ready_for_new_calibration") is True
    transfer_gates = [
        gate("current_public_transfer_floor_cleared", public_transfer_ready, {
            "public_pass_rate": post_summary.get("public_pass_rate"),
            "public_floor": post_summary.get("public_floor"),
            "cards_below_floor": post_summary.get("cards_below_floor"),
            "latest_public_task_count": post_summary.get("public_task_count"),
        }),
        gate("next_private_residual_fix_current", private_residual_current, {
            "recommended_private_fix_family": post_summary.get("recommended_private_fix_family"),
            "failed_private_fix_evidence": failed_private_fix_evidence,
            "operator_review_required": operator_review_required(failed_private_fix_evidence),
            "private_residual_target_consumer_state": private_residual_target_consumer.get("trigger_state"),
            "unresolved_target_count": private_residual_unresolved_count,
            "unresolved_target_category_counts": private_residual_target_consumer_summary.get(
                "unresolved_target_category_counts", {}
            ),
        }),
        gate("full_body_private_semantic_nonzero", full_body_semantic_nonzero(full_body, semantic_ablation), {
            "path": rel(full_body_path),
            "selected_pass_rate": full_body_summary.get("selected_pass_rate"),
            "pass_if_any_rate": full_body_summary.get("pass_if_any_rate"),
            "semantic_ablation_path": rel(semantic_ablation_path),
            "best_selected_pass_rate": semantic_ablation_summary.get("best_private_public_shaped_selected_pass_rate"),
            "best_pass_if_any_rate": semantic_ablation_summary.get("best_private_public_shaped_pass_if_any_rate"),
            "recommendation": full_body.get("recommendation"),
        }),
    ]

    hard_failed = [row for row in hard_gates if not row["passed"]]
    evidence_failed = [row for row in evidence_gates if not row["passed"]]
    transfer_failed = [row for row in transfer_gates if not row["passed"]]
    trigger_state = "RED" if hard_failed else ("GREEN" if not evidence_failed and not transfer_failed else "YELLOW")

    contract_cards = list_field(contract_stage.get("cards"))
    report = {
        "policy": "project_theseus_public_transfer_readiness_refresh_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "public_calibration_allowed": False,
        "ready_for_one_new_public_calibration": bool(trigger_state == "GREEN" and public_transfer_ready and not lock_path.exists()),
        "inputs": {
            "contract": rel(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "post_readiness": rel(post_path),
            "capacity_dry_run": rel(capacity_path),
            "public_readiness_packet": rel(packet_path),
            "operator_dry_run": rel(operator_dry_run_path),
            "sts_ranker": rel(sts_ranker_path),
            "structural_action": rel(structural_path),
            "full_body_admissibility": rel(full_body_path),
            "full_body_semantic_ablation": rel(semantic_ablation_path),
            "maturity_audit": rel(maturity_path),
            "governor": rel(governor_path),
            "private_residual_target_consumer": rel(private_residual_target_consumer_path),
            "alignment_preflight": rel(alignment_preflight_path),
            "operator_lock": rel(lock_path),
        },
        "summary": {
            "contract_slug": contract_stage.get("slug"),
            "contract_cards": contract_cards,
            "contract_task_count": contract_stage.get("total_task_count"),
            "contract_cases_per_card": contract_stage.get("cases_per_card"),
            "capacity_sufficient": capacity_gate_passed(capacity),
            "capacity_task_count": len(contract_cards) * int_number(contract_stage.get("cases_per_card")),
            "broad_private_fresh": post_summary.get("decoder_source_release_fresh"),
            "learned_only_private_pass_rate": post_summary.get("learned_only_private_pass_rate"),
            "learned_token_pass_count": post_summary.get("learned_token_pass_count"),
            "prototype_pass_count": post_summary.get("prototype_pass_count"),
            "sts_ranker_selected_pass_rate": first_present(sts_summary, "selected_pass_rate", "sts_policy_selected_pass_rate"),
            "sts_ranker_non_sts_selected_pass_rate": first_present(
                sts_summary,
                "matched_non_sts_pass_rate",
                "non_sts_policy_selected_pass_rate",
            ),
            "fallback_return_candidate_count": sts_summary.get("fallback_return_candidate_count"),
            "public_leakage_count": sts_summary.get("public_leakage_count"),
            "structural_action_trigger_state": structural.get("trigger_state"),
            "structural_action_symliquid_pass_rate": first_present(
                structural_summary,
                "symliquid_pass_rate",
                "symliquid_sts_on_pass_rate",
            ),
            "structural_action_transformer_pass_rate": first_present(
                structural_summary,
                "transformer_pass_rate",
                "transformer_sts_on_pass_rate",
            ),
            "full_body_admissibility_trigger_state": full_body.get("trigger_state"),
            "full_body_candidate_row_count": full_body_summary.get("candidate_row_count"),
            "full_body_token_candidate_count": full_body_summary.get("full_body_token_candidate_count"),
            "full_body_benchmark_promotion_eligible_candidate_count": full_body_summary.get("benchmark_promotion_eligible_candidate_count"),
            "full_body_no_admissible_task_rate": full_body_summary.get("no_admissible_task_rate"),
            "full_body_selected_pass_rate": full_body_summary.get("selected_pass_rate"),
            "full_body_pass_if_any_rate": full_body_summary.get("pass_if_any_rate"),
            "full_body_semantic_ablation_trigger_state": semantic_ablation.get("trigger_state"),
            "full_body_semantic_best_selected_pass_rate": semantic_ablation_summary.get("best_private_public_shaped_selected_pass_rate"),
            "full_body_semantic_best_pass_if_any_rate": semantic_ablation_summary.get("best_private_public_shaped_pass_if_any_rate"),
            "full_body_post_v4_default_semantic_dead": first_present(
                semantic_ablation_summary,
                "post_v4_default_semantic_dead",
                "post_v4_token_beam_semantic_dead",
            ),
            "full_body_v4_full_body_selected_pass_rate": semantic_ablation_summary.get("v4_full_body_selected_pass_rate"),
            "full_body_v4_strict_novel_learned_only_pass_rate": semantic_ablation_summary.get("v4_strict_novel_learned_only_pass_rate"),
            "full_body_fallback_return_candidate_count": full_body_summary.get("fallback_return_candidate_count"),
            "full_body_public_leakage_count": full_body_summary.get("public_leakage_count"),
            "full_body_recommendation": full_body.get("recommendation"),
            "maturity_trigger_state": maturity.get("trigger_state"),
            "maturity_blockers": first_present(maturity_summary, "maturity_blockers") or maturity.get("maturity_blockers"),
            "promotion_integrity_ready": maturity_summary.get("promotion_integrity_ready"),
            "promotion_integrity_source": maturity_summary.get("promotion_integrity_source"),
            "promotion_integrity_verified_candidate_count": maturity_summary.get("promotion_integrity_verified_candidate_count"),
            "promotion_integrity_viea_record_count": maturity_summary.get("promotion_integrity_viea_record_count"),
            "governor_trigger_state": governor.get("trigger_state"),
            "latest_public_pass_rate": post_summary.get("public_pass_rate"),
            "latest_public_task_count": post_summary.get("public_task_count"),
            "latest_public_cards_below_floor": post_summary.get("cards_below_floor"),
            "recommended_private_fix_family": post_summary.get("recommended_private_fix_family"),
            "failed_private_fix_evidence": post_summary.get("failed_private_fix_evidence"),
            "private_residual_target_consumer_state": private_residual_target_consumer.get("trigger_state"),
            "private_residual_target_rows": private_residual_target_consumer_summary.get("target_rows"),
            "private_residual_covered_target_count": private_residual_target_consumer_summary.get("covered_target_count"),
            "private_residual_unresolved_target_count": private_residual_unresolved_count,
            "private_residual_unresolved_target_category_counts": private_residual_target_consumer_summary.get(
                "unresolved_target_category_counts", {}
            ),
            "private_residual_fix_current": private_residual_current,
            "private_fix_review_ready": private_fix_review_ready,
            "alignment_preflight_state": alignment_preflight.get("trigger_state"),
            "alignment_preflight_ready": alignment_preflight.get("alignment_preflight_ready"),
            "alignment_case_manifest": alignment_summary.get("case_manifest"),
            "alignment_case_manifest_row_count": alignment_summary.get("case_manifest_row_count"),
            "alignment_candidate_manifest_bound_to_case_manifest": alignment_summary.get(
                "candidate_manifest_bound_to_case_manifest"
            ),
            "hard_failed_gate_count": len(hard_failed),
            "evidence_failed_gate_count": len(evidence_failed),
            "transfer_failed_gate_count": len(transfer_failed),
            "calibration_review_ready_after_private_fixes": bool(
                not hard_failed
                and not evidence_failed
                and private_fix_review_ready
                and full_body_semantic_nonzero(full_body, semantic_ablation)
            ),
        },
        "gates": {
            "hard": hard_gates,
            "evidence": evidence_gates,
            "transfer": transfer_gates,
        },
        "next_actions": next_actions(
            trigger_state,
            post_summary,
            transfer_failed,
            evidence_failed,
            private_residual_target_consumer_summary,
        ),
        "rules": {
            "public_calibration": "This refresh never executes public calibration and keeps public_calibration_allowed=false.",
            "benchmark_contract": "The next public spend must match the frozen contract or produce a new contract before approval.",
            "training_boundary": "Do not train on public prompts, tests, hidden tests, solutions, traces, answer templates, or score labels.",
            "fallback_boundary": "Fallback return candidates are disallowed for promotion and public claims.",
        },
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }
    return report


def capacity_gate_passed(capacity: dict[str, Any]) -> bool:
    rows = capacity.get("capacity") if isinstance(capacity.get("capacity"), list) else []
    return bool(rows) and all(row.get("capacity_sufficient") is True for row in rows if isinstance(row, dict))


def structural_no_fallback_public_teacher(report: dict[str, Any]) -> bool:
    gates = report.get("gates") if isinstance(report.get("gates"), list) else []
    wanted = {
        "fallback_return_rate_zero",
        "no_public_or_teacher_use",
    }
    seen = {str(row.get("gate") or row.get("name") or ""): row for row in gates if isinstance(row, dict)}
    return all(seen.get(name, {}).get("passed") is True for name in wanted)


def structural_gate_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    gates = report.get("gates") if isinstance(report.get("gates"), list) else []
    out: dict[str, Any] = {}
    for row in gates:
        if not isinstance(row, dict):
            continue
        name = str(row.get("gate") or row.get("name") or "")
        if name in {"fallback_return_rate_zero", "no_public_or_teacher_use", "compiler_syntax_validity_nonzero"}:
            out[name] = row.get("passed")
    return out


def full_body_admissibility_current(report: dict[str, Any]) -> bool:
    summary = object_field(report, "summary")
    return bool(
        report.get("trigger_state") == "GREEN"
        and int_number(summary.get("candidate_row_count")) > 0
        and int_number(summary.get("full_body_token_candidate_count")) > 0
        and int_number(summary.get("benchmark_promotion_eligible_candidate_count")) > 0
        and number(summary.get("no_admissible_task_rate")) <= 0.03
        and int_number(summary.get("fallback_return_candidate_count")) == 0
        and int_number(summary.get("template_like_candidate_count")) == 0
        and int_number(summary.get("public_leakage_count")) == 0
        and external_calls(report) == 0
    )


def full_body_semantic_quality_current(report: dict[str, Any]) -> bool:
    summary = object_field(report, "summary")
    return bool(
        report.get("trigger_state") == "GREEN"
        and number(summary.get("best_private_public_shaped_selected_pass_rate")) > 0.0
        and number(summary.get("best_private_public_shaped_pass_if_any_rate")) > 0.0
        and external_calls(report) == 0
    )


def full_body_semantic_nonzero(report: dict[str, Any], semantic_ablation: dict[str, Any] | None = None) -> bool:
    summary = object_field(report, "summary")
    if number(summary.get("selected_pass_rate")) > 0.0 and number(summary.get("pass_if_any_rate")) > 0.0:
        return True
    ablation_summary = object_field(semantic_ablation or {}, "summary")
    return bool(
        (semantic_ablation or {}).get("trigger_state") == "GREEN"
        and number(ablation_summary.get("best_private_public_shaped_selected_pass_rate")) > 0.0
        and number(ablation_summary.get("best_private_public_shaped_pass_if_any_rate")) > 0.0
    )


def alignment_preflight_ready(report: dict[str, Any]) -> bool:
    summary = object_field(report, "summary")
    return bool(
        report.get("policy") == "project_theseus_public_calibration_alignment_preflight_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and report.get("alignment_preflight_ready") is True
        and int_number(summary.get("case_manifest_row_count")) == 320
        and summary.get("candidate_manifest_bound_to_case_manifest") is True
        and summary.get("candidate_manifest_preexists_before_run") is False
        and summary.get("public_tests_used") is False
        and summary.get("public_solutions_used") is False
        and int_number(summary.get("training_rows_written")) == 0
        and external_calls(report) == 0
    )


def external_inference_zero(*reports: dict[str, Any]) -> bool:
    return all(external_calls(report) == 0 for report in reports)


def external_calls(report: dict[str, Any]) -> int:
    return max(
        int_number(report.get("external_inference_calls")),
        int_number(object_field(report, "summary").get("external_inference_calls")),
    )


def operator_review_required(failed_private_fix_evidence: Any) -> bool:
    if not isinstance(failed_private_fix_evidence, dict):
        return False
    return (
        failed_private_fix_evidence.get("family") == OPERATOR_REVIEW_FAMILY
        and failed_private_fix_evidence.get("reason") == OPERATOR_REVIEW_REASON
    )


def private_residual_fix_current(failed_private_fix_evidence: Any) -> bool:
    return failed_private_fix_evidence in ({}, None) or operator_review_required(failed_private_fix_evidence)


def next_actions(
    trigger_state: str,
    post_summary: dict[str, Any],
    transfer_failed: list[dict[str, Any]],
    evidence_failed: list[dict[str, Any]],
    private_residual_target_consumer_summary: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    if trigger_state == "RED":
        actions.extend(
            [
                "Fix hard integrity gates before any calibration review.",
                "Keep the public operator lock active.",
            ]
        )
    if evidence_failed:
        actions.append("Refresh failed private evidence gates before proposing public calibration.")
    if transfer_failed:
        family = str(post_summary.get("recommended_private_fix_family") or "unknown_private_residual_fix")
        transfer_failed_names = {str(row.get("gate") or "") for row in transfer_failed if isinstance(row, dict)}
        if "next_private_residual_fix_current" in transfer_failed_names:
            unresolved = int_number(private_residual_target_consumer_summary.get("unresolved_target_count"))
            if unresolved > 0:
                actions.append(
                    "Close the private residual target queue before public review: "
                    f"{unresolved} target(s) remain unresolved, with categories "
                    f"{private_residual_target_consumer_summary.get('unresolved_target_category_counts', {})}."
                )
            else:
                actions.append(f"Fix and refresh private residual family `{family}` under the current decoder/release.")
            actions.append("Do not spend a new public calibration until that private fix is current and readiness is rerun.")
        if "current_public_transfer_floor_cleared" in transfer_failed_names:
            if operator_review_required(post_summary.get("failed_private_fix_evidence")):
                if int_number(private_residual_target_consumer_summary.get("unresolved_target_count")) > 0:
                    actions.append(
                        "Private semantic evidence is current, but public review is premature until the calibration slice/candidate manifest alignment target is closed."
                    )
                else:
                    actions.append("Private residual evidence is current; the remaining transfer blocker is an operator-reviewed bounded public calibration decision.")
            else:
                actions.append("Keep mining current private residual failures before proposing another public calibration.")
        if "full_body_private_semantic_nonzero" in transfer_failed_names:
            actions.append("Full-body candidate admissibility is repaired, but private pass-if-any remains zero; repair learned candidate semantics before any public calibration review.")
    if not actions:
        actions.append("If the operator lock is intentionally unlocked for exactly one run, execute only the frozen contract through the guarded runner and relock immediately.")
    actions.append("Keep public benchmark prompts/tests/solutions/traces out of training rows.")
    return actions


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    lines = [
        "# Public Transfer Readiness Refresh v1",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Ready for one new public calibration: `{report.get('ready_for_one_new_public_calibration')}`",
        f"- Public calibration allowed: `{report.get('public_calibration_allowed')}`",
        f"- Contract: `{summary.get('contract_slug')}` / `{summary.get('contract_task_count')}` tasks",
        f"- Capacity sufficient: `{summary.get('capacity_sufficient')}`",
        f"- Broad private fresh: `{summary.get('broad_private_fresh')}`",
        f"- Learned-only private pass rate: `{summary.get('learned_only_private_pass_rate')}`",
        f"- Latest public pass rate: `{summary.get('latest_public_pass_rate')}`",
        f"- Cards below floor: `{', '.join(summary.get('latest_public_cards_below_floor') or [])}`",
        f"- Full-body admissibility: `{summary.get('full_body_admissibility_trigger_state')}` no-admissible=`{summary.get('full_body_no_admissible_task_rate')}`",
        f"- Full-body private pass-if-any: `{summary.get('full_body_pass_if_any_rate')}`",
        f"- Full-body semantic ablation: `{summary.get('full_body_semantic_ablation_trigger_state')}` best selected/pass-if-any=`{summary.get('full_body_semantic_best_selected_pass_rate')}` / `{summary.get('full_body_semantic_best_pass_if_any_rate')}`",
        f"- Promotion integrity receipt ready: `{summary.get('promotion_integrity_ready')}`",
        f"- Promotion integrity verified candidates: `{summary.get('promotion_integrity_verified_candidate_count')}`",
        f"- Recommended private fix: `{summary.get('recommended_private_fix_family')}`",
        f"- Private residual target consumer: `{summary.get('private_residual_target_consumer_state')}`",
        f"- Private residual unresolved targets: `{summary.get('private_residual_unresolved_target_count')}`",
        f"- Private residual unresolved categories: `{summary.get('private_residual_unresolved_target_category_counts')}`",
        "",
        "## Gates",
    ]
    for group, rows in object_field(report, "gates").items():
        lines.append(f"### {group}")
        for row in rows if isinstance(rows, list) else []:
            lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}`")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions") if isinstance(report.get("next_actions"), list) else []:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def object_field(obj: dict[str, Any], key: str) -> dict[str, Any]:
    value = obj.get(key)
    return value if isinstance(value, dict) else {}


def list_field(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def int_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def first_present(obj: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value
    return None


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
