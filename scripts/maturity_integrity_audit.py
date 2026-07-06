#!/usr/bin/env python3
"""Maturity and anti-cheat audit for Project Theseus.

This report is deliberately conservative. It does not try to make the system
look capable; it looks for ways the system could give itself credit without
real causal evidence. Promotion, model growth, and public calibration should be
blocked whenever this audit finds leakage, template substitution, report-only
claims, or missing transfer proof.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_CODE_FLOOR = 0.70
PUBLIC_CARDS = {
    "source_mbpp",
    "source_evalplus",
    "source_bigcodebench",
    "source_livecodebench",
    "humaneval",
    "human_eval",
}
PUBLIC_LEAK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bpublic[_ -]?(test|solution|answer)s?\b",
        r"\bsource_(mbpp|evalplus|bigcodebench|livecodebench)\b",
        r"\bhuman[_ -]?eval\b",
        r"\bbenchmark[_ -]?answer\b",
    )
]
TRAINING_MANIFEST_GLOBS = [
    "reports/*private*curriculum*.jsonl",
    "reports/*private*train*.jsonl",
    "reports/*residual*curriculum*.jsonl",
    "training_data/high_transfer/private_train/**/*.jsonl",
    "data/private_code_curriculum/**/*.jsonl",
    "data/sts_learning/**/*.jsonl",
]
CALIBRATION_ONLY_PATH_PARTS = {
    "/data/public_code_benchmark_manifests/",
    "/data/old_project_benchmarks/",
    "/benchmarks/",
}
CALIBRATION_ONLY_REPORT_PREFIXES = (
    "code_lm_public_tasks_",
    "real_code_benchmark_traces_",
    "student_code_candidates_",
    "code_lm_sts_public_generations_",
)
CALIBRATION_ONLY_REPORT_SUFFIXES = (
    "_phase_ledger.jsonl",
    "_public_candidate_manifest.jsonl",
)
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_ROWS = 2000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--decoder-gate", default="reports/decoder_v2_private_ablation_gate.json")
    parser.add_argument("--private-public-transfer", default="reports/capability_transfer_closure_v1.json")
    parser.add_argument("--real-code", default="reports/real_code_benchmark_graduation.json")
    parser.add_argument("--sts-causal", default="reports/sts_causal_decoder_ablation.json")
    parser.add_argument("--sts-control", default="reports/sts_decoder_control_contract.json")
    parser.add_argument("--sts-ranker-policy", default="reports/sts_ranker_policy_v1.json")
    parser.add_argument(
        "--student-candidate-manifest",
        default="reports/student_code_candidates_private_pressure_private_recovery_train_once_fanout_v1.jsonl",
    )
    parser.add_argument("--out", default="reports/maturity_integrity_audit.json")
    parser.add_argument("--markdown-out", default="reports/maturity_integrity_audit.md")
    args = parser.parse_args()

    state = load_state(args)
    manifest_scan = scan_training_manifests()
    checks = build_checks(state, manifest_scan)
    hard_blockers = [row for row in checks if row["severity"] == "hard" and not row["passed"]]
    maturity_blockers = [row for row in checks if row["severity"] == "maturity" and not row["passed"]]
    evidence_blockers = [row for row in checks if row["severity"] == "evidence" and not row["passed"]]
    trigger_state = "RED" if hard_blockers else "YELLOW" if maturity_blockers or evidence_blockers else "GREEN"
    public_ready = trigger_state == "GREEN" and transfer_floor_cleared(state)
    promotion_ready = public_ready and bool(get_path(state, ["coherence", "allows_candidate_promotion"], False))
    recovery_evidence = floor_recovery_evidence(state)
    promotion_integrity = promotion_integrity_evidence(state)
    report = {
        "policy": "project_theseus_maturity_integrity_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "passed": trigger_state == "GREEN",
        "summary": {
            "hard_blocker_count": len(hard_blockers),
            "maturity_blocker_count": len(maturity_blockers),
            "evidence_blocker_count": len(evidence_blockers),
            "public_calibration_allowed": public_ready,
            "candidate_promotion_allowed": promotion_ready,
            "model_growth_allowed": False if trigger_state != "GREEN" else bool(get_path(state, ["model_growth", "model_growth_allowed"], False)),
            "broad_public_pass_rate": broad_public_pass_rate(state),
            "weakest_public_card_rate": weakest_public_card_rate(state),
            "floor_recovery_evidence": recovery_evidence,
            "promotion_integrity_ready": promotion_integrity["ready"],
            "promotion_integrity_source": promotion_integrity["source"],
            "promotion_integrity_verified_candidate_count": promotion_integrity["integrity_verified_candidate_count"],
            "promotion_integrity_viea_record_count": promotion_integrity["viea_record_count"],
            "closed_loop_residual_ratchet_decision": object_field(
                state.get("closed_loop_residual_ratchet"), "summary"
            ).get("decision"),
            "closed_loop_residual_ratchet_reason": object_field(
                state.get("closed_loop_residual_ratchet"), "summary"
            ).get("decision_reason"),
            "manifest_public_leak_hit_count": manifest_scan["hit_count"],
            "manifest_files_scanned": manifest_scan["files_scanned"],
            "integrity_principle": "No report, template, teacher output, public benchmark, or scaffold counts as capability without private causal evidence and transfer proof.",
        },
        "checks": checks,
        "hard_blockers": [row["name"] for row in hard_blockers],
        "maturity_blockers": [row["name"] for row in maturity_blockers],
        "evidence_blockers": [row["name"] for row in evidence_blockers],
        "floor_recovery_evidence": recovery_evidence,
        "next_actions": next_actions(hard_blockers, maturity_blockers, evidence_blockers, state),
        "manifest_scan": manifest_scan,
        "rules": {
            "public_data": "Public benchmark tasks are calibration-only; public tests, answers, and card identities must not become private training rows.",
            "templates": "Templates/scaffolds may be diagnostic only; promotion evidence must come from learned candidates with exact interfaces and executable bodies.",
            "reports": "Reports are views over evidence. A report is not capability unless a named consumer changes behavior and an ablation shows lift.",
            "teacher": "Teacher may be used only as governed training-time proposal/distillation input; runtime external serving and teacher apply mode remain forbidden.",
            "growth": "Capacity growth is blocked until transfer, coherence, integrity, and architecture-delta evidence are all clean.",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    ingest_self(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state != "RED" else 2


def load_state(args: argparse.Namespace | None = None) -> dict[str, Any]:
    def arg_path(name: str, default: str) -> Path:
        if args is None:
            return REPORTS / default
        return resolve(str(getattr(args, name)))

    return {
        "a_plus": read_json(REPORTS / "a_plus_operating_scorecard.json"),
        "architecture_results": read_json(REPORTS / "architecture_experiment_results.json"),
        "broad": read_json(arg_path("broad", "broad_transfer_matrix.json")),
        "broad_floor_recovery": read_json(REPORTS / "broad_public_code_transfer_floor_recovery.json"),
        "closed_loop_residual_ratchet": read_json(REPORTS / "closed_loop_residual_ratchet.json"),
        "candidate": read_json(REPORTS / "candidate_promotion_gate.json"),
        "coherence": read_json(REPORTS / "coherence_delirium_gate.json"),
        "contract_preflight": read_json(REPORTS / "code_lm_closure_public_contract_preflight_seed23_32.json"),
        "cross_domain": read_json(REPORTS / "cross_domain_sts_capsules.json"),
        "decoder_gate": read_json(arg_path("decoder_gate", "decoder_v2_private_ablation_gate.json")),
        "execution_shape": latest_json("execution_shape_candidate_coverage*.json")
        or latest_json("execution_shape_private_ablation*.json"),
        "external_inference": read_json(REPORTS / "external_inference_audit.json"),
        "model_growth": read_json(REPORTS / "model_growth_gate.json"),
        "performance": read_json(REPORTS / "performance_optimizer.json"),
        "private_public_transfer": read_json(arg_path("private_public_transfer", "private_public_transfer_proof.json")),
        "real_code": read_json(arg_path("real_code", "real_code_benchmark_graduation.json")),
        "report_store": read_json(REPORTS / "report_evidence_store.json"),
        "report_store_db": REPORTS / "report_evidence_store.sqlite",
        "sts_causal": read_json(arg_path("sts_causal", "sts_causal_decoder_ablation.json")),
        "sts_control": read_json(arg_path("sts_control", "sts_decoder_control_contract.json")),
        "sts_ranker_policy": read_json(arg_path("sts_ranker_policy", "sts_ranker_policy_v1.json")),
        "agent_lane_transfer": read_json(REPORTS / "agent_lane_transfer_gate.json"),
        "teacher_last": read_json(REPORTS / "teacher_oracle_last.json"),
        "teacher_distillation": read_json(REPORTS / "teacher_distillation_gate.json"),
        "candidate_manifest_path": (
            resolve(str(args.student_candidate_manifest))
            if args is not None
            else REPORTS / "student_code_candidates_private_pressure_private_recovery_train_once_fanout_v1.jsonl"
        ),
    }


def build_checks(state: dict[str, Any], manifest_scan: dict[str, Any]) -> list[dict[str, Any]]:
    broad_rate = broad_public_pass_rate(state)
    weakest_card = weakest_public_card_rate(state)
    decoder_summary = object_field(state["decoder_gate"], "summary")
    execution_summary = object_field(state["execution_shape"], "summary")
    real_code_summary = object_field(state["real_code"], "summary")
    transfer_summary = object_field(state["private_public_transfer"], "summary")
    transfer_candidate_coverage = candidate_coverage_evidence(decoder_summary, transfer_summary)
    preflight = contract_preflight_evidence(state["contract_preflight"])
    promotion_integrity = promotion_integrity_evidence(state)
    architecture_status = str(state["architecture_results"].get("status") or "")
    architecture_delta = max_abs_delta(state["architecture_results"].get("score_delta"))
    architecture_targeted_delta = architecture_targeted_delta_ready(state["architecture_results"])
    sts_summary = object_field(state["sts_causal"], "summary")
    sts_delta = max_abs_delta(
        {
            "coverage": sts_summary.get("sts_public_eligible_coverage_delta"),
            "pass_rate": sts_summary.get("sts_public_pass_rate_delta"),
            "candidate": sts_summary.get("sts_candidate_distribution_delta"),
        }
    )
    report_store_db = state["report_store_db"]
    report_store_rows = sqlite_count(report_store_db, "report_runs") if report_store_db.exists() else 0
    candidate_promotes = bool(state["candidate"].get("promote"))
    model_growth_allowed = bool(state["model_growth"].get("model_growth_allowed"))
    ready_claims = readiness_claims(state)
    template_counts = {
        "execution_shape_diagnostic_template_candidate_count": int(number(execution_summary.get("diagnostic_template_candidate_count"))),
        "real_code_template_like_candidate_count": int(number(real_code_summary.get("template_like_candidate_count"))),
        "decoder_template_like_candidate_count": int(number(decoder_summary.get("template_like_candidate_count"))),
    }
    no_public_leak_metrics = [
        int(number(object_field(state["contract_preflight"], "summary").get("public_leak_flag_count"))),
        int(number(real_code_summary.get("public_leak_flag_count"))),
        int(number(decoder_summary.get("public_leak_flag_count"))),
        manifest_scan["hit_count"],
    ]
    grammar_masked_promotion_evidence = promotion_candidate_grammar_evidence(real_code_summary, state)
    return [
        check(
            "public_training_leak_absent",
            sum(no_public_leak_metrics) == 0,
            "hard",
            {"public_leak_metrics": no_public_leak_metrics, "manifest_hits": manifest_scan["hits"][:10]},
        ),
        check(
            "public_benchmark_claims_calibration_only",
            public_claims_are_calibration_only(state),
            "hard",
            {
                "real_code_claim": state["real_code"].get("public_benchmark_score_claim"),
                "private_public_claim": transfer_summary.get("public_benchmark_score_claim"),
            },
        ),
        check(
            "contract_preflight_blocks_erased_interfaces",
            preflight.get("trigger_state") == "GREEN"
            and int(number(preflight.get("varargs_task_count"))) == 0
            and int(number(preflight.get("weak_required_construct_count"))) == 0
            and int(number(preflight.get("weak_full_body_count"))) == 0
            and not preflight.get("hard_blockers"),
            "hard",
            {
                "state": preflight.get("trigger_state"),
                "varargs_task_count": preflight.get("varargs_task_count"),
                "weak_required_construct_count": preflight.get("weak_required_construct_count"),
                "weak_full_body_count": preflight.get("weak_full_body_count"),
                "hard_blockers": preflight.get("hard_blockers"),
            },
        ),
        check(
            "templates_are_not_promotion_evidence",
            sum(template_counts.values()) == 0 and not template_promotion_claimed(state),
            "hard",
            template_counts,
        ),
        check(
            "promotion_facing_candidates_are_grammar_masked_learned_tokens",
            bool(grammar_masked_promotion_evidence.get("passed")),
            "hard",
            grammar_masked_promotion_evidence,
        ),
        check(
            "promotion_integrity_receipt_required",
            bool(promotion_integrity["ready"]),
            "hard",
            promotion_integrity,
        ),
        check(
            "public_calibration_locked_until_private_transfer_proof",
            not any(ready_claims.values()) or both_private_gates_ready(state),
            "hard",
            ready_claims,
        ),
        check(
            "candidate_promotion_blocked_until_transfer_and_coherence",
            (not candidate_promotes)
            or (
                transfer_floor_cleared(state)
                and bool(get_path(state, ["coherence", "allows_candidate_promotion"], False))
                and state["coherence"].get("trigger_state") == "GREEN"
            ),
            "hard",
            {
                "candidate_promotes": candidate_promotes,
                "broad_rate": broad_rate,
                "weakest_card": weakest_card,
                "coherence_state": state["coherence"].get("trigger_state"),
                "allows_candidate_promotion": state["coherence"].get("allows_candidate_promotion"),
            },
        ),
        check(
            "model_growth_blocked_until_integrity_and_transfer",
            (not model_growth_allowed) or transfer_floor_cleared(state),
            "hard",
            {
                "model_growth_allowed": model_growth_allowed,
                "model_growth_missing_evidence": state["model_growth"].get("missing_evidence"),
                "broad_rate": broad_rate,
                "weakest_card": weakest_card,
            },
        ),
        check(
            "teacher_runtime_serving_forbidden_and_training_governed",
            teacher_boundary_is_governed(state),
            "hard",
            teacher_boundary_evidence(state),
        ),
        check(
            "report_store_is_append_only_evidence",
            report_store_rows >= 100,
            "evidence",
            {"db_exists": report_store_db.exists(), "stored_run_count": report_store_rows},
        ),
        check(
            "scorecard_weakest_domain_caps_overall",
            bool(get_path(state, ["a_plus", "summary", "weakest_domain_caps_overall"], False))
            or int(number(get_path(state, ["a_plus", "summary", "blocking_wall_count"], 0))) == 0,
            "evidence",
            object_field(state["a_plus"], "summary"),
        ),
        check(
            "public_transfer_floor_cleared",
            transfer_floor_cleared(state),
            "maturity",
            {
                "broad_public_pass_rate": broad_rate,
                "weakest_card_rate": weakest_card,
                "floor": PUBLIC_CODE_FLOOR,
                "recovery": floor_recovery_evidence(state),
            },
        ),
        check(
            "candidate_coverage_transfer_gate_is_fresh",
            candidate_coverage_ready(decoder_summary, transfer_summary),
            "maturity",
            transfer_candidate_coverage,
        ),
        check(
            "architecture_experiments_have_delta_before_promotion",
            (not bool(state["architecture_results"].get("promotion_evidence")))
            or (architecture_status == "completed_with_capability_delta" and architecture_targeted_delta),
            "hard",
            {
                "status": architecture_status,
                "promotion_evidence": state["architecture_results"].get("promotion_evidence"),
                "max_abs_delta": architecture_delta,
                "targeted_delta_ready": architecture_targeted_delta,
                "promotion_decision": state["architecture_results"].get("promotion_decision"),
                "minimum_delta": 0.01,
            },
        ),
        check(
            "architecture_delta_exists_for_compounding_claim",
            architecture_status == "completed_with_capability_delta" and architecture_targeted_delta,
            "maturity",
            {
                "status": architecture_status,
                "max_abs_delta": architecture_delta,
                "targeted_delta_ready": architecture_targeted_delta,
                "promotion_decision": state["architecture_results"].get("promotion_decision"),
                "minimum_delta": 0.01,
            },
        ),
        check(
            "sts_capsules_have_causal_consumer",
            sts_has_causal_consumer(state, sts_delta),
            "maturity",
            {
                "sts_causal_state": state["sts_causal"].get("trigger_state"),
                "sts_max_abs_delta": sts_delta,
                "sts_control_state": state["sts_control"].get("trigger_state"),
                "sts_control_rows_written": state["sts_control"].get("control_rows_written"),
                "sts_control_consumer_count": len(get_path(state, ["sts_control", "consumer_contract", "consumers"], [])),
                "sts_ranker_policy": sts_ranker_policy_evidence(state),
                "agent_lane_transfer_state": state["agent_lane_transfer"].get("trigger_state"),
                "cross_domain_capsules": get_path(state, ["cross_domain", "summary", "capsule_count"], None),
            },
        ),
        check(
            "gpu_path_not_cpu_bound_by_policy",
            performance_is_not_claiming_full_gpu_when_idle(state),
            "maturity",
            {
                "performance_state": state["performance"].get("trigger_state"),
                "bottlenecks": get_path(state, ["performance", "bottlenecks"], []),
                "preferred_backend": get_path(state, ["performance", "preferred_backend"], None),
            },
        ),
    ]


def check(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def promotion_integrity_evidence(state: dict[str, Any]) -> dict[str, Any]:
    candidate_report = state.get("candidate") if isinstance(state.get("candidate"), dict) else {}
    receipt = object_field(candidate_report, "promotion_integrity")
    viea_receipt = object_field(receipt, "viea_spine_consumer_receipt")
    non_promotion_claims = receipt.get("non_promotion_claims_by_family")
    if not isinstance(non_promotion_claims, dict):
        non_promotion_claims = {}
    family_counts = receipt.get("family_counts")
    if not isinstance(family_counts, dict):
        family_counts = {}
    verified_by_family = receipt.get("promotion_verified_by_family")
    if not isinstance(verified_by_family, dict):
        verified_by_family = {}
    no_cheat_fault_count = int(number(viea_receipt.get("no_cheat_fault_count")))
    integrity_mismatch_count = int(number(receipt.get("integrity_mismatch_count")))
    ready = (
        receipt.get("policy") == "project_theseus_candidate_promotion_integrity_receipt_v1"
        and receipt.get("ready_for_promotion_claims") is True
        and viea_receipt.get("ready") is True
        and no_cheat_fault_count == 0
        and integrity_mismatch_count == 0
        and int(number(receipt.get("integrity_verified_candidate_count"))) > 0
        and not non_promotion_claims
        and int(number(receipt.get("public_training_rows_written"))) == 0
        and int(number(receipt.get("external_inference_calls"))) == 0
        and int(number(receipt.get("fallback_return_count"))) == 0
    )
    return {
        "source": "reports/candidate_promotion_gate.json",
        "ready": ready,
        "policy": receipt.get("policy"),
        "candidate_integrity_policy": receipt.get("candidate_integrity_policy"),
        "candidate_integrity_trigger_state": receipt.get("candidate_integrity_trigger_state"),
        "viea_ready": viea_receipt.get("ready"),
        "viea_record_count": viea_receipt.get("record_count"),
        "viea_missing_required_groups": viea_receipt.get("missing_required_groups"),
        "viea_no_cheat_fault_count": no_cheat_fault_count,
        "candidate_count": receipt.get("candidate_count"),
        "audited_candidate_count": receipt.get("audited_candidate_count"),
        "family_counts": family_counts,
        "promotion_verified_by_family": verified_by_family,
        "non_promotion_claims_by_family": non_promotion_claims,
        "integrity_verified_candidate_count": receipt.get("integrity_verified_candidate_count"),
        "integrity_mismatch_count": integrity_mismatch_count,
        "public_training_rows_written": receipt.get("public_training_rows_written"),
        "external_inference_calls": receipt.get("external_inference_calls"),
        "fallback_return_count": receipt.get("fallback_return_count"),
        "rule": (
            "Maturity/public-transfer readiness must consume the independent candidate-promotion "
            "integrity receipt. Self-declared candidate flags, routers, tools, fallback/template "
            "families, and structural adapters do not support learned-generation claims."
        ),
    }


def public_claims_are_calibration_only(state: dict[str, Any]) -> bool:
    claims = [
        state["real_code"].get("public_benchmark_score_claim"),
        get_path(state, ["private_public_transfer", "summary", "public_benchmark_score_claim"], None),
    ]
    for claim in claims:
        if claim and "calibration_only" not in str(claim):
            return False
    return True


def contract_preflight_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Normalize the canonical Code LM closure preflight report shape.

    ``code_lm_closure.py --preflight-only`` records the decoder-contract
    counters under ``summary.public_decoder_contract_preflight``. Older audit
    snapshots placed the same counters at the report top level. Promotion
    maturity should accept either shape, but it should still require the same
    zero-erasure evidence.
    """
    nested = get_path(report, ["summary", "public_decoder_contract_preflight"], {})
    if isinstance(nested, dict) and nested:
        return {
            "trigger_state": report.get("trigger_state"),
            "varargs_task_count": nested.get("varargs_task_count"),
            "weak_required_construct_count": nested.get("weak_required_construct_count"),
            "weak_full_body_count": nested.get("weak_full_body_count"),
            "hard_blockers": nested.get("hard_blockers"),
            "public_tests_used": nested.get("public_tests_used"),
            "public_solutions_used": nested.get("public_solutions_used"),
            "public_task_count": nested.get("public_task_count"),
        }
    return report


def template_promotion_claimed(state: dict[str, Any]) -> bool:
    if not state["candidate"].get("promote"):
        return False
    real_summary = object_field(state["real_code"], "summary")
    return int(number(real_summary.get("template_like_candidate_count"))) > 0


def promotion_candidate_grammar_evidence(real_code_summary: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Use fresh canonical candidate evidence when public calibration reports are stale."""
    rule = (
        "public-transfer promotion evidence must be full-body learned-token decode under "
        "parser/contract masks, not prompt-program or diagnostic comparator rows"
    )
    transfer_summary = object_field(state["private_public_transfer"], "summary")
    closure_evidence = candidate_coverage_evidence(object_field(state["decoder_gate"], "summary"), transfer_summary)
    if closure_evidence.get("source") == "capability_transfer_closure_v1":
        return {
            "source": "capability_transfer_closure_v1",
            "passed": bool(closure_evidence.get("ready")),
            "rule": rule,
            "candidate_coverage": closure_evidence,
        }

    real_eligible = int(number(real_code_summary.get("benchmark_promotion_eligible_candidate_count")))
    real_grammar = int(number(real_code_summary.get("grammar_masked_learned_token_candidate_count")))
    real_evidence = {
        "source": "real_code_benchmark_graduation",
        "grammar_masked_learned_token_candidate_count": real_code_summary.get("grammar_masked_learned_token_candidate_count"),
        "benchmark_promotion_eligible_candidate_count": real_code_summary.get("benchmark_promotion_eligible_candidate_count"),
    }
    if real_eligible == 0 or (real_grammar > 0 and real_grammar >= real_eligible):
        return {**real_evidence, "passed": True, "rule": rule}

    manifest_path = state.get("candidate_manifest_path")
    if not isinstance(manifest_path, Path):
        manifest_path = REPORTS / "student_code_candidates_private_pressure_private_recovery_train_once_fanout_v1.jsonl"
    manifest_evidence = scan_candidate_manifest_for_promotion_grammar(manifest_path)
    manifest_eligible = int(number(manifest_evidence.get("benchmark_promotion_eligible_candidate_count")))
    manifest_grammar = int(number(manifest_evidence.get("grammar_masked_learned_token_candidate_count")))
    manifest_full_body = int(number(manifest_evidence.get("full_body_token_candidate_count")))
    manifest_token = int(number(manifest_evidence.get("token_level_code_generation_learned_count")))
    manifest_clean = (
        int(number(manifest_evidence.get("template_like_candidate_count"))) == 0
        and int(number(manifest_evidence.get("public_tests_used_count"))) == 0
        and int(number(manifest_evidence.get("public_solutions_used_count"))) == 0
        and int(number(manifest_evidence.get("external_inference_used_count"))) == 0
    )
    manifest_integrity_passed = (
        bool(manifest_evidence.get("exists"))
        and (manifest_eligible == 0 or (manifest_grammar >= manifest_eligible > 0))
        and (manifest_eligible == 0 or (manifest_full_body >= manifest_eligible and manifest_token >= manifest_eligible))
        and manifest_clean
    )
    transfer_gates_ready = bool(state["decoder_gate"].get("ready_for_public_calibration")) and both_private_gates_ready(state)
    return {
        **real_evidence,
        "source": "canonical_train_once_fanout_manifest",
        "real_code_summary_missing_or_stale_grammar_count": True,
        "manifest": manifest_evidence,
        "decoder_gate_ready": bool(state["decoder_gate"].get("ready_for_public_calibration")),
        "private_public_transfer_ready": both_private_gates_ready(state),
        "candidate_integrity_passed": manifest_integrity_passed,
        "transfer_gates_ready": transfer_gates_ready,
        "promotion_ready": manifest_integrity_passed and transfer_gates_ready,
        "passed": manifest_integrity_passed,
        "rule": (
            f"{rule}; decoder/transfer readiness is reported separately so missing transfer proof "
            "does not masquerade as candidate-integrity failure"
        ),
    }


def scan_candidate_manifest_for_promotion_grammar(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "path": rel(path),
        "exists": path.exists(),
        "row_count": 0,
        "benchmark_promotion_eligible_candidate_count": 0,
        "grammar_masked_learned_token_candidate_count": 0,
        "full_body_token_candidate_count": 0,
        "token_level_code_generation_learned_count": 0,
        "template_like_candidate_count": 0,
        "public_tests_used_count": 0,
        "public_solutions_used_count": 0,
        "external_inference_used_count": 0,
        "bad_promotion_rows": [],
    }
    if not path.exists():
        return evidence
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return evidence
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        evidence["row_count"] += 1
        eligible = row_bool_any(row, "benchmark_promotion_eligible", "benchmark_candidate_eligible")
        grammar = row_bool_any(row, "grammar_masked_learned_token_candidate")
        full_body = row_bool_any(row, "full_body_token_candidate", "full_body_candidate")
        token = row_bool_any(row, "token_level_code_generation_learned", "token_level_generated")
        template_like = row_bool_any(row, "template_like_candidate", "placeholder_scaffold_body")
        public_tests = row_bool_any(row, "public_tests_used", "public_tests_visible_to_generator", "tests_used")
        public_solutions = row_bool_any(row, "public_solutions_used", "public_solution_used", "canonical_solution_seen_by_solver")
        external_inference = row_bool_any(row, "external_inference_used") or number(row_deep_get(row, "external_inference_calls")) > 0
        if eligible:
            evidence["benchmark_promotion_eligible_candidate_count"] += 1
        if grammar:
            evidence["grammar_masked_learned_token_candidate_count"] += 1
        if full_body:
            evidence["full_body_token_candidate_count"] += 1
        if token:
            evidence["token_level_code_generation_learned_count"] += 1
        if template_like:
            evidence["template_like_candidate_count"] += 1
        if public_tests:
            evidence["public_tests_used_count"] += 1
        if public_solutions:
            evidence["public_solutions_used_count"] += 1
        if external_inference:
            evidence["external_inference_used_count"] += 1
        if eligible and not (grammar and full_body and token) and len(evidence["bad_promotion_rows"]) < 10:
            evidence["bad_promotion_rows"].append(
                {
                    "line": index,
                    "task_id": row.get("task_id"),
                    "candidate_mode": row.get("candidate_generation_mode") or row.get("candidate_mode"),
                    "grammar_masked": grammar,
                    "full_body": full_body,
                    "token_level": token,
                }
            )
    return evidence


def row_bool_any(row: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if row_deep_get(row, key) is True:
            return True
    return False


def row_deep_get(row: dict[str, Any], key: str) -> Any:
    if row.get(key) is not None:
        return row.get(key)
    provenance = row.get("provenance")
    if isinstance(provenance, dict) and provenance.get(key) is not None:
        return provenance.get(key)
    loop = row.get("program_synthesis_loop_v1")
    if isinstance(loop, dict):
        decode_control = loop.get("decode_control")
        if isinstance(decode_control, dict) and decode_control.get(key) is not None:
            return decode_control.get(key)
    return None


def readiness_claims(state: dict[str, Any]) -> dict[str, bool]:
    transfer_summary = object_field(state["private_public_transfer"], "summary")
    return {
        "decoder_gate_ready": bool(state["decoder_gate"].get("ready_for_public_calibration")),
        "private_public_transfer_ready": bool(state["private_public_transfer"].get("ready_for_public_calibration")),
        "private_public_transfer_summary_ready": bool(get_path(state, ["private_public_transfer", "summary", "ready_for_public_calibration"], False)),
        "capability_transfer_private_semantic_ready": bool(transfer_summary.get("private_semantic_ready")),
        "capability_transfer_candidate_coverage_ready": bool(transfer_summary.get("public_candidate_coverage_ready")),
        "capability_transfer_no_cheat_clean": bool(transfer_summary.get("no_cheat_clean")),
    }


def both_private_gates_ready(state: dict[str, Any]) -> bool:
    transfer_summary = object_field(state["private_public_transfer"], "summary")
    capability_transfer_ready = (
        bool(transfer_summary.get("private_semantic_ready"))
        and bool(transfer_summary.get("public_candidate_coverage_ready"))
        and bool(transfer_summary.get("no_cheat_clean"))
        and candidate_coverage_ready(object_field(state["decoder_gate"], "summary"), transfer_summary)
    )
    legacy_transfer_ready = bool(
        state["private_public_transfer"].get("ready_for_public_calibration")
        or get_path(state, ["private_public_transfer", "summary", "ready_for_public_calibration"], False)
    )
    return capability_transfer_ready or (
        bool(state["decoder_gate"].get("ready_for_public_calibration")) and legacy_transfer_ready
    )


def candidate_coverage_ready(decoder_summary: dict[str, Any], transfer_summary: dict[str, Any]) -> bool:
    evidence = candidate_coverage_evidence(decoder_summary, transfer_summary)
    if evidence.get("source") == "capability_transfer_closure_v1":
        return bool(evidence.get("ready"))
    coverage = first_number(
        decoder_summary.get("public_eligible_task_coverage"),
        transfer_summary.get("public_eligible_task_coverage"),
        transfer_summary.get("eligible_candidate_coverage"),
    )
    no_admissible = first_number(
        decoder_summary.get("public_no_admissible_task_rate"),
        transfer_summary.get("public_no_admissible_task_rate"),
        transfer_summary.get("no_admissible_candidate_rate"),
    )
    program_loop = first_number(
        decoder_summary.get("public_program_synthesis_loop_present_rate"),
        transfer_summary.get("public_program_synthesis_loop_present_rate"),
    )
    program_ready = first_number(
        decoder_summary.get("public_program_synthesis_promotion_ready_rate"),
        transfer_summary.get("public_program_synthesis_promotion_ready_rate"),
    )
    return (
        coverage is not None
        and no_admissible is not None
        and coverage >= 0.60
        and no_admissible <= 0.25
        and program_loop is not None
        and program_loop >= 0.60
        and program_ready is not None
        and program_ready >= 0.50
    )


def candidate_coverage_evidence(decoder_summary: dict[str, Any], transfer_summary: dict[str, Any]) -> dict[str, Any]:
    public_candidate_coverage = object_field(transfer_summary, "public_candidate_coverage")
    if public_candidate_coverage:
        task_count = int(number(public_candidate_coverage.get("task_count")))
        candidate_count = int(number(public_candidate_coverage.get("candidate_count")))
        promotion_eligible = int(number(public_candidate_coverage.get("benchmark_promotion_eligible_candidate_count")))
        full_body = int(number(public_candidate_coverage.get("full_body_token_candidate_count")))
        grammar_masked = int(number(public_candidate_coverage.get("grammar_masked_learned_token_candidate_count")))
        fallback = int(number(public_candidate_coverage.get("expression_memory_fallback_count")))
        template_like = int(number(public_candidate_coverage.get("template_like_candidate_count")))
        loop_closure = int(number(public_candidate_coverage.get("loop_closure_candidate_count")))
        external = int(number(public_candidate_coverage.get("external_inference_calls")))
        public_tests_visible = public_candidate_coverage.get("public_tests_visible_to_generator")
        canonical_solution_seen = public_candidate_coverage.get("canonical_solution_seen_by_solver")
        ready = (
            bool(transfer_summary.get("private_semantic_ready"))
            and bool(transfer_summary.get("no_cheat_clean"))
            and public_candidate_coverage.get("trigger_state") == "GREEN"
            and bool(public_candidate_coverage.get("ready"))
            and task_count >= 160
            and candidate_count >= task_count
            and promotion_eligible > 0
            and full_body >= promotion_eligible
            and grammar_masked >= promotion_eligible
            and fallback == 0
            and template_like == 0
            and loop_closure == 0
            and external == 0
            and public_tests_visible is False
            and canonical_solution_seen is False
        )
        return {
            "source": "capability_transfer_closure_v1",
            "ready": ready,
            "private_semantic_ready": transfer_summary.get("private_semantic_ready"),
            "no_cheat_clean": transfer_summary.get("no_cheat_clean"),
            "task_count": task_count,
            "candidate_count": candidate_count,
            "benchmark_promotion_eligible_candidate_count": promotion_eligible,
            "full_body_token_candidate_count": full_body,
            "grammar_masked_learned_token_candidate_count": grammar_masked,
            "expression_memory_fallback_count": fallback,
            "template_like_candidate_count": template_like,
            "loop_closure_candidate_count": loop_closure,
            "external_inference_calls": external,
            "public_tests_visible_to_generator": public_tests_visible,
            "canonical_solution_seen_by_solver": canonical_solution_seen,
        }
    coverage = first_number(
        decoder_summary.get("public_eligible_task_coverage"),
        transfer_summary.get("public_eligible_task_coverage"),
        transfer_summary.get("eligible_candidate_coverage"),
    )
    no_admissible = first_number(
        decoder_summary.get("public_no_admissible_task_rate"),
        transfer_summary.get("public_no_admissible_task_rate"),
        transfer_summary.get("no_admissible_candidate_rate"),
    )
    program_loop = first_number(
        decoder_summary.get("public_program_synthesis_loop_present_rate"),
        transfer_summary.get("public_program_synthesis_loop_present_rate"),
    )
    program_ready = first_number(
        decoder_summary.get("public_program_synthesis_promotion_ready_rate"),
        transfer_summary.get("public_program_synthesis_promotion_ready_rate"),
    )
    return {
        "source": "legacy_private_public_transfer_proof",
        "ready": (
            coverage is not None
            and no_admissible is not None
            and coverage >= 0.60
            and no_admissible <= 0.25
            and program_loop is not None
            and program_loop >= 0.60
            and program_ready is not None
            and program_ready >= 0.50
        ),
        "public_eligible_task_coverage": coverage,
        "public_no_admissible_task_rate": no_admissible,
        "public_program_synthesis_loop_present_rate": program_loop,
        "public_program_synthesis_promotion_ready_rate": program_ready,
        "private_public_transfer_ready": transfer_summary.get("ready_for_public_calibration"),
    }


def teacher_is_proposal_only(state: dict[str, Any]) -> bool:
    mode = str(state["teacher_last"].get("mode") or "proposal")
    blocked_reason = str(state["teacher_last"].get("blocked_reason") or "")
    return mode in {"", "proposal"} and blocked_reason != "teacher_must_remain_proposal_only"


def teacher_boundary_is_governed(state: dict[str, Any]) -> bool:
    gate = state.get("teacher_distillation") if isinstance(state.get("teacher_distillation"), dict) else {}
    summary = object_field(gate, "summary")
    if gate.get("trigger_state") == "GREEN":
        return (
            bool(summary.get("runtime_external_tokens_forbidden"))
            and bool(summary.get("manifest_admission_safety_checks_clean"))
            and int(number(summary.get("manifest_public_overlap_hits"))) == 0
            and int(number(summary.get("manifest_holdout_overlap_hits"))) == 0
            and float(number(summary.get("manifest_verifier_pass_rate"))) >= 0.95
        )
    return teacher_is_proposal_only(state)


def teacher_boundary_evidence(state: dict[str, Any]) -> dict[str, Any]:
    gate = state.get("teacher_distillation") if isinstance(state.get("teacher_distillation"), dict) else {}
    summary = object_field(gate, "summary")
    return {
        "teacher_last_mode": state["teacher_last"].get("mode"),
        "teacher_last_status": state["teacher_last"].get("status"),
        "teacher_last_blocked_reason": state["teacher_last"].get("blocked_reason"),
        "distillation_gate_state": gate.get("trigger_state"),
        "distillation_allowed": summary.get("distillation_allowed"),
        "manifest_row_count": summary.get("manifest_row_count"),
        "manifest_verifier_pass_rate": summary.get("manifest_verifier_pass_rate"),
        "manifest_admission_safety_checks_clean": summary.get("manifest_admission_safety_checks_clean"),
        "manifest_public_overlap_hits": summary.get("manifest_public_overlap_hits"),
        "manifest_holdout_overlap_hits": summary.get("manifest_holdout_overlap_hits"),
        "runtime_external_tokens_forbidden": summary.get("runtime_external_tokens_forbidden"),
        "teacher_accepted_row_share": summary.get("teacher_accepted_row_share"),
    }


def sts_has_causal_consumer(state: dict[str, Any], sts_delta: float) -> bool:
    if state["agent_lane_transfer"].get("trigger_state") == "GREEN" and sts_delta >= 0.01:
        return True
    if state["sts_causal"].get("trigger_state") == "GREEN" and sts_delta >= 0.01:
        return True
    if sts_ranker_policy_ready(state):
        return True
    return False


def sts_ranker_policy_ready(state: dict[str, Any]) -> bool:
    report = state.get("sts_ranker_policy")
    if not isinstance(report, dict) or report.get("trigger_state") != "GREEN":
        return False
    summary = object_field(report, "summary")
    integration = object_field(report, "guarded_integration")
    recommendation = object_field(report, "recommendation")
    return (
        bool(integration.get("eligible_for_guarded_runtime_use"))
        and integration.get("enabled_by_default") is False
        and integration.get("allow_public_calibration") is False
        and int(number(summary.get("surface_count"))) >= 2
        and float(number(summary.get("selected_pass_delta_sts_policy_minus_non_sts_policy"))) > 0.0
        and int(number(summary.get("sts_policy_vs_original_regression_count"))) == 0
        and int(number(summary.get("fallback_return_candidate_count"))) == 0
        and int(number(summary.get("public_leakage_count"))) == 0
        and int(number(summary.get("external_inference_calls"))) == 0
        and recommendation.get("public_calibration_auto_run") is False
    )


def sts_ranker_policy_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("sts_ranker_policy")
    if not isinstance(report, dict):
        report = {}
    summary = object_field(report, "summary")
    integration = object_field(report, "guarded_integration")
    recommendation = object_field(report, "recommendation")
    return {
        "trigger_state": report.get("trigger_state"),
        "eligible_for_guarded_runtime_use": integration.get("eligible_for_guarded_runtime_use"),
        "enabled_by_default": integration.get("enabled_by_default"),
        "allow_public_calibration": integration.get("allow_public_calibration"),
        "surface_count": summary.get("surface_count"),
        "task_count": summary.get("task_count"),
        "selected_pass_delta_sts_policy_minus_non_sts_policy": summary.get("selected_pass_delta_sts_policy_minus_non_sts_policy"),
        "sts_policy_vs_original_regression_count": summary.get("sts_policy_vs_original_regression_count"),
        "fallback_return_candidate_count": summary.get("fallback_return_candidate_count"),
        "public_leakage_count": summary.get("public_leakage_count"),
        "external_inference_calls": summary.get("external_inference_calls"),
        "recommendation": recommendation.get("decision"),
    }


def performance_is_not_claiming_full_gpu_when_idle(state: dict[str, Any]) -> bool:
    performance = state["performance"]
    if performance.get("trigger_state") == "RED":
        return False
    bottlenecks = performance.get("bottlenecks") if isinstance(performance.get("bottlenecks"), list) else []
    names = {str(row.get("name") or row.get("id") or row) for row in bottlenecks if isinstance(row, dict)}
    return "accelerator_idle_during_training" not in names


def transfer_floor_cleared(state: dict[str, Any]) -> bool:
    broad_rate = broad_public_pass_rate(state)
    weakest = weakest_public_card_rate(state)
    return broad_rate is not None and broad_rate >= PUBLIC_CODE_FLOOR and weakest is not None and weakest >= PUBLIC_CODE_FLOOR


def broad_public_pass_rate(state: dict[str, Any]) -> float | None:
    summary = object_field(state["broad"], "summary")
    return first_number(
        summary.get("real_public_pass_rate"),
        summary.get("aggregate_pass_rate"),
        get_path(state, ["broad", "public_transfer", "real_public_task_pass_rate"], None),
    )


def weakest_public_card_rate(state: dict[str, Any]) -> float | None:
    rows = state["broad"].get("rows") if isinstance(state["broad"].get("rows"), list) else []
    rates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rate = first_number(
            row.get("real_public_pass_rate"),
            row.get("public_pass_rate"),
            row.get("pass_rate"),
            row.get("real_public_task_pass_rate"),
        )
        if rate is not None:
            rates.append(rate)
    return min(rates) if rates else None


def floor_recovery_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("broad_floor_recovery")
    if not isinstance(report, dict):
        report = {}
    summary = object_field(report, "summary")
    family_audit = object_field(summary, "same_seed_semantic_family_delta_audit")
    return {
        "path": "reports/broad_public_code_transfer_floor_recovery.json",
        "trigger_state": report.get("trigger_state"),
        "status": report.get("status"),
        "remaining_gap_explained": bool(summary.get("remaining_gap_explained")),
        "private_pressure_row_count": summary.get("private_pressure_row_count"),
        "same_seed_ablation_status": summary.get("same_seed_ablation_status"),
        "same_seed_private_semantic_lift": summary.get("same_seed_private_semantic_lift"),
        "same_seed_semantic_family_regressed_count": family_audit.get("regressed_family_count"),
        "same_seed_semantic_family_flat_count": family_audit.get("flat_family_count"),
        "same_seed_semantic_family_regressed": [
            row.get("family")
            for row in family_audit.get("regressed_families", [])
            if isinstance(row, dict)
        ],
        "same_seed_semantic_family_flat": [
            row.get("family")
            for row in family_audit.get("flat_families", [])
            if isinstance(row, dict)
        ],
        "public_tests_used": summary.get("public_tests_used"),
        "public_solutions_used": summary.get("public_solutions_used"),
        "score_semantics": "diagnostic recovery evidence only; public floor still requires bounded calibration to change",
    }


def scan_training_manifests() -> dict[str, Any]:
    seen: set[Path] = set()
    files: list[Path] = []
    skipped_calibration_report_count = 0
    for pattern in TRAINING_MANIFEST_GLOBS:
        for path in ROOT.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            if calibration_only_path(path):
                skipped_calibration_report_count += 1
                seen.add(path)
                continue
            seen.add(path)
            files.append(path)
    hits = []
    rows_scanned = 0
    bytes_scanned = 0
    for path in sorted(files):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_MANIFEST_BYTES:
            continue
        try:
            for index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if index > MAX_MANIFEST_ROWS:
                    break
                rows_scanned += 1
                bytes_scanned += len(line)
                if private_manifest_line_has_public_leak(line):
                    hits.append({"path": rel(path), "line": index, "excerpt": safe_excerpt(line)})
                    if len(hits) >= 50:
                        break
        except OSError:
            continue
    return {
        "files_scanned": len(files),
        "rows_scanned": rows_scanned,
        "bytes_scanned": bytes_scanned,
        "hit_count": len(hits),
        "hits": hits,
        "skipped_large_file_count": sum(1 for path in files if path.exists() and path.stat().st_size > MAX_MANIFEST_BYTES),
        "skipped_calibration_report_count": skipped_calibration_report_count,
    }


def calibration_only_path(path: Path) -> bool:
    normalized = "/" + rel(path).lower().strip("/").replace("\\", "/")
    if any(part in normalized for part in CALIBRATION_ONLY_PATH_PARTS):
        return True
    if "/reports/" in normalized:
        name = path.name.lower()
        if name.startswith(CALIBRATION_ONLY_REPORT_PREFIXES):
            return True
        if name.endswith(CALIBRATION_ONLY_REPORT_SUFFIXES):
            return True
    return False


def private_manifest_line_has_public_leak(line: str) -> bool:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return private_manifest_row_has_public_leak(value)
    lowered = line.lower()
    if "calibration_only" in lowered or "public_leak_flag_count" in lowered:
        return False
    return any(pattern.search(line) for pattern in PUBLIC_LEAK_PATTERNS)


def private_manifest_row_has_public_leak(row: dict[str, Any]) -> bool:
    evidence_level = str(row.get("benchmark_evidence_level") or "").lower()
    card_id = str(row.get("card_id") or "").lower()
    if evidence_level.startswith("private") or card_id.startswith("private"):
        explicit_source = " ".join(
            str(row.get(key) or "")
            for key in (
                "source_card",
                "source_benchmark",
                "public_source",
                "benchmark_source",
                "benchmark_id",
                "dataset",
            )
        ).lower()
        if any(card in explicit_source for card in PUBLIC_CARDS):
            return True
        for key in ("public_tests", "public_solution", "public_answer", "benchmark_answer"):
            if key in row and row.get(key):
                return True
        return False
    text = json.dumps(row, sort_keys=True)
    if "calibration_only" in text.lower():
        return False
    return any(pattern.search(text) for pattern in PUBLIC_LEAK_PATTERNS)


def next_actions(
    hard_blockers: list[dict[str, Any]],
    maturity_blockers: list[dict[str, Any]],
    evidence_blockers: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    names = {row["name"] for row in hard_blockers + maturity_blockers + evidence_blockers}
    if "public_training_leak_absent" in names:
        actions.append("Quarantine any private rows that mention public benchmark cards/tests/answers; regenerate from private residuals only.")
    if "public_transfer_floor_cleared" in names or "candidate_coverage_transfer_gate_is_fresh" in names:
        recovery_summary = object_field(state.get("broad_floor_recovery"), "summary")
        family_audit = object_field(recovery_summary, "same_seed_semantic_family_delta_audit")
        regressed = [
            str(row.get("family"))
            for row in family_audit.get("regressed_families", [])
            if isinstance(row, dict) and row.get("family")
        ]
        flat = [
            str(row.get("family"))
            for row in family_audit.get("flat_families", [])
            if isinstance(row, dict) and row.get("family")
        ]
        if regressed or flat:
            actions.append(
                "Keep public calibration locked; repair private same-seed semantic family deltas before promotion-ready recovery evidence: "
                f"regressed={regressed or []}, flat={flat or []}."
            )
        closed_loop_action = closed_loop_ratchet_action_text(state)
        if closed_loop_action:
            actions.append(closed_loop_action)
        elif "public_transfer_floor_cleared" in names and "candidate_coverage_transfer_gate_is_fresh" not in names:
            transfer_summary = object_field(state.get("private_public_transfer"), "summary")
            transfer_ready = bool(
                transfer_summary.get("private_semantic_ready")
                and transfer_summary.get("public_candidate_coverage_ready")
                and transfer_summary.get("no_cheat_clean")
            )
            if transfer_ready:
                actions.append(
                    "Fresh private transfer and public-shaped candidate coverage are clean; do not manufacture more private rows just to avoid the public wall. "
                    "The remaining floor check requires exactly one governed public calibration under the frozen contract if the run-specific operator unlock exists; otherwise stop on the locked public-transfer wall."
                )
            else:
                actions.append(
                    "Keep public calibration locked; build private residual repair rows for verifier mismatch, "
                    "remaining no-admissible candidates, return-shape fidelity, and algorithmic planning; rerun "
                    "private decoder/transfer/STS gates before any future bounded public calibration."
                )
        else:
            actions.append(
                "Keep public calibration locked; run only private closure shards and decoder/private transfer proof "
                "until coverage and no-admissible gates pass."
            )
    if "sts_capsules_have_causal_consumer" in names:
        control_ready = bool(
            state["sts_control"].get("trigger_state") in {"GREEN", "YELLOW"}
            and int(number(state["sts_control"].get("control_rows_written"))) > 0
        )
        if control_ready:
            actions.append("Run a fresh private closure that consumes sts_decoder_control_rows.jsonl, then rerun same-seed STS A/B for measured delta.")
        else:
            actions.append("Make STS/SymLiquid choose decoder priors, skeleton families, retry policy, or route selection, then run same-seed A/B.")
    if "architecture_delta_exists_for_compounding_claim" in names:
        actions.append("Run one narrow architecture experiment from a residual cluster with private heldout delta, rollback, and promotion rules.")
    if "gpu_path_not_cpu_bound_by_policy" in names:
        actions.append("Move hot candidate scoring/generation work to batched CUDA and keep CPU work to orchestration.")
    if not actions and not hard_blockers:
        actions.append("Integrity is clean; continue private-to-public transfer proof before any model growth or public calibration.")
    if hard_blockers:
        actions.insert(0, "Do not promote, grow, or public-calibrate until hard integrity blockers are cleared.")
    return actions[:8]


def closed_loop_ratchet_action_text(state: dict[str, Any]) -> str:
    report = state.get("closed_loop_residual_ratchet")
    if not isinstance(report, dict) or report.get("trigger_state") not in {"GREEN", "YELLOW"}:
        return ""
    summary = object_field(report, "summary")
    decision = object_field(report, "decision")
    kind = str(summary.get("decision") or decision.get("kind") or "")
    if kind not in {"promote", "rollback", "retry_private", "stop_blocker"}:
        return ""
    reason = str(summary.get("decision_reason") or decision.get("reason") or "closed-loop residual ratchet decision")
    if kind == "stop_blocker":
        return f"Closed-loop residual ratchet stop blocker: {reason}"
    if kind == "retry_private":
        return f"Closed-loop residual ratchet requires a private retry: {reason}"
    if kind == "rollback":
        return f"Closed-loop residual ratchet requires rollback/demotion: {reason}"
    return f"Closed-loop residual ratchet is promotion-ready, but maturity/governor gates still own promotion: {reason}"


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    lines = [
        "# Theseus Maturity Integrity Audit",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Hard blockers: `{summary.get('hard_blocker_count')}`",
        f"- Maturity blockers: `{summary.get('maturity_blocker_count')}`",
        f"- Evidence blockers: `{summary.get('evidence_blocker_count')}`",
        f"- Public calibration allowed: `{summary.get('public_calibration_allowed')}`",
        f"- Candidate promotion allowed: `{summary.get('candidate_promotion_allowed')}`",
        f"- Promotion integrity receipt ready: `{summary.get('promotion_integrity_ready')}`",
        f"- Promotion integrity verified candidates: `{summary.get('promotion_integrity_verified_candidate_count')}`",
        "",
        "## Failed Checks",
        "",
    ]
    failed = [row for row in report.get("checks", []) if isinstance(row, dict) and not row.get("passed")]
    if not failed:
        lines.append("- None.")
    for row in failed:
        lines.append(f"- `{row.get('severity')}` {row.get('name')}")
    lines.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def latest_json(pattern: str) -> dict[str, Any]:
    matches = sorted(REPORTS.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        return {}
    return read_json(matches[-1])


def object_field(value: Any, key: str) -> dict[str, Any]:
    item = value.get(key) if isinstance(value, dict) else {}
    return item if isinstance(item, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def first_number(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def max_abs_delta(value: Any) -> float:
    if isinstance(value, dict):
        return max([max_abs_delta(item) for item in value.values()] or [0.0])
    if isinstance(value, list):
        return max([max_abs_delta(item) for item in value] or [0.0])
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def architecture_targeted_delta_ready(report: dict[str, Any]) -> bool:
    contract = object_field(report, "residual_delta_contract")
    decision = object_field(report, "promotion_decision")
    observed = contract.get("observed_improvements") if isinstance(contract.get("observed_improvements"), dict) else {}
    return (
        bool(contract.get("targeted_improvement_observed"))
        and bool(observed)
        and decision.get("decision") in {"promote", "planned"}
    )


def sqlite_count(path: Path, table: str) -> int:
    try:
        with sqlite3.connect(str(path)) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0] if row else 0)
    except sqlite3.Error:
        return 0


def safe_excerpt(line: str) -> str:
    text = line.strip().replace("\\", "/")
    return text[:220] + ("..." if len(text) > 220 else "")


def ingest_self(path: Path, payload: dict[str, Any]) -> None:
    try:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        import report_evidence_store  # type: ignore

        report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, path, payload=payload)
    except Exception:
        return


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
