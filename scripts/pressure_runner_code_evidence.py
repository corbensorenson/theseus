"""Code-benchmark evidence scoring helpers for pressure_runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pressure_runner_utils import check, read_jsonl


def learned_manifest_matches_card(path: Path, card_id: str) -> bool:
    if not path.exists():
        return False
    prefix = f"{card_id}_"
    for row in read_jsonl(path):
        if str(row.get("candidate_source") or "") not in {
            "student_learning_checkpoint_v1",
            "student_neural_checkpoint_v1",
            "student_token_generator_checkpoint_v1",
            "student_code_lm_checkpoint_v1",
        }:
            continue
        if str(row.get("task_id") or "").startswith(prefix):
            return True
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        if str(provenance.get("card_id") or "") == card_id:
            return True
    return False


def code_repair_score_bonus(organism: dict[str, Any]) -> float:
    if not organism.get("ran"):
        return 0.0
    bonus = 0.04
    if organism.get("transfer_loaded"):
        bonus += 0.03
    if organism.get("transfer_altered_behavior"):
        bonus += 0.04
    if float(organism.get("pass_rate_delta") or 0.0) > 0:
        bonus += 0.02
    return min(0.12, bonus)


def public_multistream_score_bonus(public_multistream: dict[str, Any]) -> float:
    if not public_multistream.get("ran"):
        return 0.0
    if public_multistream.get("public_benchmark_score_claim") != "forbidden":
        return 0.0
    if int(public_multistream.get("external_inference_calls") or 0) != 0:
        return 0.0
    bonus = 0.02
    if float(public_multistream.get("pass_rate_delta") or 0.0) > 0:
        bonus += 0.02
    if int(public_multistream.get("task_level_improvements_over_single_stream") or 0) > 0:
        bonus += 0.02
    if int(public_multistream.get("task_level_regressions_vs_single_stream") or 0) == 0:
        bonus += 0.01
    return min(0.07, bonus)


def real_code_graduation_score_bonus(real_code: dict[str, Any]) -> float:
    if not real_code.get("ran"):
        return 0.0
    if not bool(real_code.get("student_candidate_benchmark_integrity_valid")):
        return 0.0
    if not bool(real_code.get("token_level_code_generation_learned")):
        return 0.0
    if int(real_code.get("template_like_candidate_count") or 0) != 0:
        return 0.0
    if int(real_code.get("loop_closure_candidate_count") or 0) != 0:
        return 0.0
    if real_code.get("candidate_source") not in {
        "local_theseus_student_checkpoint",
        "student_learning_checkpoint_v1",
        "student_neural_checkpoint_v1",
        "student_token_generator_checkpoint_v1",
        "student_code_lm_checkpoint_v1",
    }:
        return 0.0
    if real_code.get("public_benchmark_score_claim") not in {
        "student_checkpoint_public_task_calibration_only",
        "student_learning_checkpoint_public_task_calibration_only",
        "student_neural_checkpoint_public_task_calibration_only",
        "student_token_generator_checkpoint_public_task_calibration_only",
        "student_code_lm_checkpoint_public_task_calibration_only",
    }:
        return 0.0
    if int(real_code.get("external_inference_calls") or 0) != 0:
        return 0.0
    bonus = 0.02
    if int(real_code.get("public_task_count") or 0) > 0:
        bonus += 0.02
    if float(real_code.get("pass_rate_delta") or 0.0) >= 0.0:
        bonus += 0.01
    if int(real_code.get("task_level_regressions_vs_single_stream") or 0) == 0:
        bonus += 0.01
    return min(0.06, bonus)


def organism_checks(organism: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        check("code_repair_organism_ran", bool(organism.get("ran")), organism.get("report") or organism.get("skipped_reason") or "not_run"),
        check("code_transfer_artifacts_loaded", bool(organism.get("transfer_loaded")), f"report={organism.get('report')}"),
        check(
            "code_transfer_altered_behavior",
            bool(organism.get("transfer_altered_behavior")),
            f"baseline={organism.get('baseline_pass_rate')} transfer={organism.get('transfer_pass_rate')} delta={organism.get('pass_rate_delta')}",
        ),
        check("code_repair_patch_trace_written", bool(organism.get("patch_trace")), organism.get("patch_trace") or ""),
    ]


def public_multistream_checks(public_multistream: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        check(
            "public_code_multistream_manifest_built",
            int(public_multistream.get("case_count") or 0) > 0,
            public_multistream.get("manifest") or public_multistream.get("skipped_reason") or "not_run",
        ),
        check(
            "public_code_multistream_runner_ran",
            bool(public_multistream.get("ran")),
            public_multistream.get("runner_report") or public_multistream.get("skipped_reason") or "not_run",
        ),
        check(
            "public_code_multistream_delta_positive",
            float(public_multistream.get("pass_rate_delta") or 0.0) > 0,
            f"delta={public_multistream.get('pass_rate_delta')}",
        ),
        check(
            "public_code_multistream_no_task_regressions",
            int(public_multistream.get("task_level_regressions_vs_single_stream") or 0) == 0,
            f"regressions={public_multistream.get('task_level_regressions_vs_single_stream')}",
        ),
        check(
            "public_code_score_claim_quarantined",
            public_multistream.get("public_benchmark_score_claim") == "forbidden",
            str(public_multistream.get("public_benchmark_score_claim")),
        ),
    ]


def real_code_graduation_checks(real_code: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        check(
            "real_code_graduation_runner_ran",
            bool(real_code.get("ran")),
            real_code.get("report") or real_code.get("skipped_reason") or "not_run",
        ),
        check(
            "real_code_public_tasks_or_loader_cases_present",
            int(real_code.get("public_task_count") or 0) > 0 or int(real_code.get("loader_regression_case_count") or 0) > 0,
            f"public_tasks={real_code.get('public_task_count')} loader_cases={real_code.get('loader_regression_case_count')}",
        ),
        check(
            "real_code_same_case_delta_reported",
            "pass_rate_delta" in real_code,
            f"delta={real_code.get('pass_rate_delta')}",
        ),
        check(
            "real_code_no_task_regressions",
            int(real_code.get("task_level_regressions_vs_single_stream") or 0) == 0,
            f"regressions={real_code.get('task_level_regressions_vs_single_stream')}",
        ),
        check(
            "real_code_score_claim_quarantined",
            real_code.get("public_benchmark_score_claim")
            in {
                "student_checkpoint_public_task_calibration_only",
                "student_learning_checkpoint_public_task_calibration_only",
                "student_neural_checkpoint_public_task_calibration_only",
                "student_token_generator_checkpoint_public_task_calibration_only",
                "student_code_lm_checkpoint_public_task_calibration_only",
            },
            str(real_code.get("public_benchmark_score_claim")),
        ),
        check(
            "real_code_student_candidate_source",
            real_code.get("candidate_source")
            in {
                "local_theseus_student_checkpoint",
                "student_learning_checkpoint_v1",
                "student_neural_checkpoint_v1",
                "student_token_generator_checkpoint_v1",
                "student_code_lm_checkpoint_v1",
            },
            f"candidate_source={real_code.get('candidate_source')} student_candidates={real_code.get('student_candidate_count')}",
        ),
        check(
            "real_code_token_level_student_generation",
            bool(real_code.get("token_level_code_generation_learned"))
            and bool(real_code.get("student_candidate_benchmark_integrity_valid"))
            and int(real_code.get("benchmark_promotion_eligible_candidate_count") or 0) > 0,
            (
                f"token_level={real_code.get('token_level_code_generation_learned')} "
                f"integrity={real_code.get('student_candidate_benchmark_integrity_valid')} "
                f"eligible={real_code.get('benchmark_promotion_eligible_candidate_count')}"
            ),
        ),
        check(
            "real_code_no_template_or_loop_distilled_candidates",
            int(real_code.get("template_like_candidate_count") or 0) == 0
            and int(real_code.get("loop_closure_candidate_count") or 0) == 0,
            f"template_like={real_code.get('template_like_candidate_count')} loop_closure={real_code.get('loop_closure_candidate_count')}",
        ),
    ]


def organism_metrics(organism: dict[str, Any]) -> dict[str, Any]:
    return {
        "ran": bool(organism.get("ran")),
        "ok": bool(organism.get("ok")),
        "report": organism.get("report"),
        "patch_trace": organism.get("patch_trace"),
        "transfer_evidence": organism.get("transfer_evidence"),
        "transfer_loaded": bool(organism.get("transfer_loaded")),
        "transfer_consumed": bool(organism.get("transfer_consumed")),
        "transfer_altered_behavior": bool(organism.get("transfer_altered_behavior")),
        "baseline_pass_rate": organism.get("baseline_pass_rate"),
        "transfer_pass_rate": organism.get("transfer_pass_rate"),
        "pass_rate_delta": organism.get("pass_rate_delta"),
        "task_count": organism.get("task_count"),
        "residual_count": organism.get("residual_count"),
    }


def public_multistream_metrics(public_multistream: dict[str, Any]) -> dict[str, Any]:
    return {
        "ran": bool(public_multistream.get("ran")),
        "ok": bool(public_multistream.get("ok")),
        "benchmark_evidence_level": public_multistream.get("benchmark_evidence_level"),
        "public_benchmark_score_claim": public_multistream.get("public_benchmark_score_claim"),
        "builder_report": public_multistream.get("builder_report"),
        "manifest": public_multistream.get("manifest"),
        "runner_report": public_multistream.get("runner_report"),
        "trace": public_multistream.get("trace"),
        "verifier": public_multistream.get("verifier"),
        "single_stream_baseline": public_multistream.get("single_stream_baseline"),
        "patch_selection_transfer_artifact": public_multistream.get("patch_selection_transfer_artifact"),
        "case_count": public_multistream.get("case_count"),
        "single_stream_transfer_pass_rate": public_multistream.get("single_stream_transfer_pass_rate"),
        "multi_stream_pass_rate": public_multistream.get("multi_stream_pass_rate"),
        "pass_rate_delta": public_multistream.get("pass_rate_delta"),
        "task_level_improvements_over_single_stream": public_multistream.get("task_level_improvements_over_single_stream"),
        "task_level_regressions_vs_single_stream": public_multistream.get("task_level_regressions_vs_single_stream"),
        "patch_stream_synthesis_used_count": public_multistream.get("patch_stream_synthesis_used_count"),
        "avg_patch_candidates_tested": public_multistream.get("avg_patch_candidates_tested"),
        "monitorability_coverage": public_multistream.get("monitorability_coverage"),
        "verifier_score": public_multistream.get("verifier_score"),
        "apples_to_apples_overlap": public_multistream.get("apples_to_apples_overlap"),
    }


def real_code_graduation_metrics(real_code: dict[str, Any]) -> dict[str, Any]:
    return {
        "ran": bool(real_code.get("ran")),
        "ok": bool(real_code.get("ok")),
        "report": real_code.get("report"),
        "trace": real_code.get("trace"),
        "transfer_artifact": real_code.get("transfer_artifact"),
        "trigger_state": real_code.get("trigger_state"),
        "candidate_source": real_code.get("candidate_source"),
        "score_semantics": real_code.get("score_semantics"),
        "benchmark_evidence_level": real_code.get("benchmark_evidence_level"),
        "public_benchmark_score_claim": real_code.get("public_benchmark_score_claim"),
        "promotion_allowed": bool(real_code.get("promotion_allowed")),
        "public_task_count": real_code.get("public_task_count"),
        "loader_regression_case_count": real_code.get("loader_regression_case_count"),
        "total_case_count": real_code.get("total_case_count"),
        "single_stream_pass_rate": real_code.get("single_stream_pass_rate"),
        "multi_stream_pass_rate": real_code.get("multi_stream_pass_rate"),
        "real_public_task_pass_rate": real_code.get("real_public_task_pass_rate"),
        "pass_rate_delta": real_code.get("pass_rate_delta"),
        "task_level_improvements_over_single_stream": real_code.get("task_level_improvements_over_single_stream"),
        "task_level_regressions_vs_single_stream": real_code.get("task_level_regressions_vs_single_stream"),
        "transfer_artifacts_loaded": real_code.get("transfer_artifacts_loaded"),
        "transfer_behavior_changed_suites": real_code.get("transfer_behavior_changed_suites"),
        "student_candidate_count": real_code.get("student_candidate_count"),
        "student_candidate_manifest_exists": real_code.get("student_candidate_manifest_exists"),
        "student_candidate_provenance_valid": real_code.get("student_candidate_provenance_valid"),
        "student_candidate_benchmark_integrity_valid": real_code.get("student_candidate_benchmark_integrity_valid"),
        "template_like_candidate_count": real_code.get("template_like_candidate_count"),
        "loop_closure_candidate_count": real_code.get("loop_closure_candidate_count"),
        "token_level_code_generation_learned": real_code.get("token_level_code_generation_learned"),
        "token_level_learned_candidate_count": real_code.get("token_level_learned_candidate_count"),
        "benchmark_promotion_eligible_candidate_count": real_code.get("benchmark_promotion_eligible_candidate_count"),
        "candidate_generation_modes": real_code.get("candidate_generation_modes"),
    }
