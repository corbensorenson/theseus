#!/usr/bin/env python3
"""Private-only Decoder V2 ablation gate before public receiver calibration.

This script does not run public benchmarks. It summarizes private candidate
manifests from Code LM closure and checks whether contract-guided skeletons and
STS-conditioned skeletons are actually improving verifier-grounded candidate
quality before the scheduler is allowed to spend another public 4-card run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "decoder_v2_private_ablation_gate.json"
DEFAULT_MARKDOWN = REPORTS / "decoder_v2_private_ablation_gate.md"
DEFAULT_TRAIN_ONCE_FANOUT = REPORTS / "code_lm_train_once_fanout.json"
DEFAULT_CLOSURE_REPORTS = [
    REPORTS / "code_lm_closure_private_pressure_private.json",
    REPORTS / "code_lm_closure_private_pressure_private_recovery_train_once_fanout_v1.json",
    REPORTS / "code_lm_closure_edge_contract_v2_private.json",
    REPORTS / "code_lm_closure_edge_case_full_body_private_v1.json",
    REPORTS / "code_lm_closure_edge_contract_balanced_4card_private_v2.json",
]


def training_data_root() -> Path:
    configured = os.environ.get("THESEUS_TRAINING_DATA_ROOT", "").strip()
    if configured:
        return Path(configured)
    if sys.platform.startswith("win"):
        return Path("D:/ProjectTheseus/training_data")
    return ROOT / "data" / "training_data"


def training_data_path(*parts: str) -> Path:
    return training_data_root().joinpath(*parts)


DEFAULT_NO_ADMISSIBLE_PACKET = REPORTS / "no_admissible_candidate_residuals.json"
DEFAULT_NO_ADMISSIBLE_JSONL = REPORTS / "no_admissible_candidate_residuals.jsonl"
DEFAULT_PARTIAL_ARTIFACT_SCORE = REPORTS / "code_lm_partial_artifact_score.json"
DEFAULT_NO_ADMISSIBLE_POLICY_ROWS = training_data_path(
    "candidate_coverage", "private_train", "no_admissible_repair_policy_rows.jsonl"
)
MIN_PROMOTION_PUBLIC_TASKS = 32
MIN_PROMOTION_PUBLIC_CANDIDATES = 96
MIN_PROMOTION_PUBLIC_CANDIDATES_PER_TASK = 3
DECODER_SOURCE = ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure.rs"
DECODER_SOURCE_DIR = ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure"
DECODER_FINGERPRINT_MARKERS = (
    "semantic_decoder_v2",
    "execution_shape_skeleton",
    "edge_exec_repair",
    "typed_edge_exec_receiver",
    "decoder_contract",
    "contract_guided_skeleton",
    "contract_guided_token",
    "local_adapter_edge_skeleton",
    "sts_causal_skeleton",
    "candidate_floor",
    "body_token_allowed",
    "syntax_constrained_body",
    "invalid_inline_block_header_body",
    "callable_keyword_argument",
    "archive_context_manager",
    "invalid_overcomposed_generated_line",
)

sys.path.insert(0, str(ROOT / "scripts"))
import report_evidence_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-report", action="append", default=[])
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--no-admissible-out", default=str(DEFAULT_NO_ADMISSIBLE_PACKET.relative_to(ROOT)))
    parser.add_argument("--no-admissible-jsonl-out", default=str(DEFAULT_NO_ADMISSIBLE_JSONL.relative_to(ROOT)))
    parser.add_argument("--no-admissible-policy-rows-out", default=str(DEFAULT_NO_ADMISSIBLE_POLICY_ROWS))
    args = parser.parse_args()

    started = time.perf_counter()
    closure_paths = [resolve(path) for path in args.closure_report] if args.closure_report else default_closure_reports()
    closure_reports = [closure_summary(path) for path in closure_paths]
    latest = latest_usable_closure(closure_reports)
    latest_partial = latest_diagnostic_partial_closure(closure_reports)
    partial_artifact_score = read_json(DEFAULT_PARTIAL_ARTIFACT_SCORE, {})
    private_rows = read_jsonl(resolve(str(latest.get("private_candidate_manifest") or ""))) if latest else []
    public_rows = read_jsonl(resolve(str(latest.get("public_candidate_manifest") or ""))) if latest else []
    groups = group_candidates(private_rows)
    public_groups = group_candidates(public_rows)
    fail_reasons = {name: dict(reason_counts(rows)) for name, rows in groups.items()}
    contract = group_metrics(groups.get("contract_guided", []))
    baseline = group_metrics(groups.get("baseline", []))
    sts = group_metrics(groups.get("sts_conditioned_skeleton", []))
    semantic = group_metrics(groups.get("semantic_plan", []))
    closure_private_delta = private_delta(latest.get("closure_report") if latest else {})
    private_signal_positive = (
        closure_private_delta > 0.0
        if closure_private_delta is not None
        else any(metric["verifier_pass_rate"] > 0.0 for metric in [contract, baseline, sts, semantic])
    )
    decoder_fingerprint = decoder_relevant_source_fingerprint()
    decoder_source_mtime = decoder_relevant_source_mtime()
    closure_mtime = float(latest.get("mtime") or 0.0) if latest else 0.0
    report_fingerprint = str(latest.get("decoder_relevant_source_fingerprint") or "") if latest else ""
    decoder_current = closure_decoder_current(
        report_fingerprint=report_fingerprint,
        current_fingerprint=decoder_fingerprint,
        closure_mtime=closure_mtime,
        decoder_source_mtime=decoder_source_mtime,
    )
    contract_observed = contract["candidate_count"] > 0
    sts_observed = sts["candidate_count"] > 0
    contract_non_regressive = (
        contract["verifier_pass_rate"] >= max(baseline["verifier_pass_rate"], semantic["verifier_pass_rate"]) - 0.03
        if contract_observed
        else False
    )
    sts_non_regressive = (
        sts["verifier_pass_rate"] >= max(contract["verifier_pass_rate"], semantic["verifier_pass_rate"]) - 0.05
        if sts_observed
        else False
    )
    public_clean = public_candidate_clean(public_rows)
    public_generation = public_candidate_generation_quality(public_rows)
    public_provenance = candidate_provenance_quality(public_rows)
    public_no_admissible = public_no_admissible_diagnostics(public_rows)
    no_admissible_packet = materialize_no_admissible_residuals(
        public_rows=public_rows,
        private_rows=private_rows,
        public_diagnostics=public_no_admissible,
        packet_out=resolve(args.no_admissible_out),
        residual_jsonl_out=resolve(args.no_admissible_jsonl_out),
        policy_rows_out=resolve(args.no_admissible_policy_rows_out),
    )
    train_once_wrapper = read_json(DEFAULT_TRAIN_ONCE_FANOUT, {})
    train_once_freshness = train_once_freshness_for_latest(latest, train_once_wrapper)
    execution_shape_coverage = latest_execution_shape_candidate_coverage_report()
    execution_shape_summary = object_field(execution_shape_coverage, "summary")
    execution_shape_no_admissible = execution_shape_summary.get("learned_token_decoder_no_admissible_candidate_rate")
    execution_shape_no_admissible = (
        1.0 if execution_shape_no_admissible is None else number(execution_shape_no_admissible)
    )
    execution_shape_private_ready = (
        execution_shape_coverage.get("trigger_state") == "GREEN"
        and float(number(execution_shape_summary.get("learned_token_decoder_pass_rate")) or 0.0) >= 0.70
        and float(execution_shape_no_admissible) <= 0.25
        and int(number(execution_shape_summary.get("diagnostic_template_candidate_count")) or 0) == 0
    )
    gates = [
        gate("private_closure_present", bool(latest), latest.get("path") if latest else "missing"),
        gate(
            "private_closure_completed_cleanly",
            bool(latest) and latest.get("run_status") == "completed" and not latest.get("diagnostic_only"),
            {
                "latest_completed": latest.get("path") if latest else None,
                "latest_partial_or_failed": latest_partial.get("path") if latest_partial else None,
                "latest_partial_run_status": latest_partial.get("run_status") if latest_partial else None,
                "partial_artifact_score": rel_or_abs(DEFAULT_PARTIAL_ARTIFACT_SCORE)
                if DEFAULT_PARTIAL_ARTIFACT_SCORE.exists()
                else None,
                "rule": "timed-out partial manifests can diagnose candidate coverage but cannot unlock public calibration",
            },
        ),
        gate(
            "private_closure_checkpoint_current_against_private_training_rows",
            bool(latest) and bool(latest.get("checkpoint_current_against_private_training_rows")),
            latest.get("checkpoint_freshness") if latest else "missing",
        ),
        gate(
            "train_once_wrapper_current_when_applicable",
            bool(train_once_freshness.get("passed")),
            train_once_freshness,
        ),
        gate("private_candidate_manifest_present", len(private_rows) > 0, len(private_rows)),
        gate(
            "private_closure_delta_positive",
            private_signal_positive,
            "explicit_delta_not_reported_candidate_verifier_signal_used"
            if closure_private_delta is None
            else round(closure_private_delta, 6),
        ),
        gate(
            "decoder_fingerprint_current",
            decoder_current,
            {
                "closure_report_fingerprint": report_fingerprint or None,
                "current_source_fingerprint": decoder_fingerprint,
                "closure_mtime": closure_mtime or None,
                "decoder_source_mtime": decoder_source_mtime or None,
                "rule": "prefer an explicit closure-recorded decoder fingerprint; otherwise require the closure report to be newer than decoder-relevant sources",
            },
        ),
        gate("contract_guided_skeleton_observed", contract_observed, contract),
        gate("contract_guided_non_regressive", contract_non_regressive, {"contract": contract, "baseline": baseline, "semantic_plan": semantic}),
        gate("sts_conditioned_skeleton_observed", sts_observed, sts),
        gate("sts_conditioned_non_regressive", sts_non_regressive, {"sts": sts, "contract": contract, "semantic_plan": semantic}),
        gate(
            "private_execution_shape_candidate_coverage_recovered",
            execution_shape_private_ready,
            {
                "report": execution_shape_coverage.get("source_report_path"),
                "trigger_state": execution_shape_coverage.get("trigger_state"),
                "learned_token_decoder_pass_rate": execution_shape_summary.get("learned_token_decoder_pass_rate"),
                "learned_token_decoder_no_admissible_candidate_rate": execution_shape_summary.get("learned_token_decoder_no_admissible_candidate_rate"),
                "diagnostic_template_candidate_count": execution_shape_summary.get("diagnostic_template_candidate_count"),
                "public_gate_basis": execution_shape_summary.get("public_gate_basis"),
            },
        ),
        gate("public_candidate_manifest_calibration_only", public_clean["ok"], public_clean),
        gate(
            "public_surface_scale_sufficient_for_calibration",
            bool(latest) and not bool(latest.get("scale_diagnostic_only")),
            {
                "latest_closure": latest.get("path") if latest else None,
                "scale_diagnostic_only": bool(latest.get("scale_diagnostic_only")) if latest else None,
                "public_task_count": latest.get("public_task_count") if latest else None,
                "public_candidate_count": latest.get("public_candidate_count") if latest else None,
                "minimum_public_task_count": MIN_PROMOTION_PUBLIC_TASKS,
                "minimum_public_candidate_count": min_promotion_public_candidate_count(
                    int(number(latest.get("public_task_count")) or 0) if latest else 0
                ),
                "rule": (
                    "small public metadata surfaces may prove private closure/candidate behavior, "
                    "but cannot unlock public calibration"
                ),
            },
        ),
        gate(
            "public_candidate_generation_coverage",
            public_generation["actual_token_task_coverage"] >= 0.60
            and public_generation["eligible_task_coverage"] >= 0.25,
            public_generation,
        ),
        gate(
            "no_admissible_candidate_rate_bounded",
            public_generation["no_admissible_task_rate"] <= 0.25,
            public_generation,
        ),
        gate(
            "public_candidate_provenance_and_quality",
            public_provenance["actual_token_candidate_count"] > 0
            and public_provenance["provenance_present_rate"] >= 0.90
            and public_provenance["program_synthesis_loop_present_rate"] >= 0.90
            and public_provenance["program_synthesis_contract_ir_rate"] >= 0.90
            and public_provenance["program_synthesis_ast_plan_rate"] >= 0.90
            and public_provenance["program_synthesis_parser_mask_rate"] >= 0.90
            and public_provenance["program_synthesis_ranker_rate"] >= 0.90
            and public_provenance["program_synthesis_promotion_ready_rate"] >= 0.70
            and public_provenance["ast_valid_rate"] >= 0.95
            and public_provenance["entry_point_match_rate"] >= 0.95
            and public_provenance["quality_gate_pass_rate"] >= 0.90
            and public_provenance["bogus_return_local_callable_count"] == 0
            and public_provenance["bogus_return_attribute_count"] == 0,
            public_provenance,
        ),
        gate(
            "no_admissible_failures_materialized",
            no_admissible_packet["residual_record_count"] > 0
            or public_generation["no_admissible_task_count"] == 0
            or public_generation["task_count"] == 0,
            no_admissible_packet,
        ),
    ]
    ready = all(item["passed"] for item in gates)
    report = {
        "policy": "project_theseus_decoder_v2_private_ablation_gate_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if ready else "YELLOW",
        "ready_for_public_calibration": ready,
        "summary": {
            "ready_for_public_calibration": ready,
            "latest_closure": latest.get("path") if latest else None,
            "latest_closure_private_only": bool(latest.get("private_only")) if latest else False,
            "latest_closure_scale_diagnostic_only": bool(latest.get("scale_diagnostic_only")) if latest else False,
            "latest_diagnostic_partial_closure": latest_partial.get("path") if latest_partial else None,
            "diagnostic_partial_artifact_score": rel_or_abs(DEFAULT_PARTIAL_ARTIFACT_SCORE)
            if DEFAULT_PARTIAL_ARTIFACT_SCORE.exists()
            else None,
            "partial_artifacts_public_candidate_rows": get_path(
                partial_artifact_score, ["summary", "public_candidate_rows"], None
            ),
            "partial_artifacts_private_candidate_rows": get_path(
                partial_artifact_score, ["summary", "private_candidate_rows"], None
            ),
            "private_candidate_count": len(private_rows),
            "public_candidate_count": len(public_rows),
            "checkpoint_current_against_private_training_rows": bool(
                latest.get("checkpoint_current_against_private_training_rows")
            )
            if latest
            else False,
            "train_once_wrapper_current_when_applicable": bool(train_once_freshness.get("passed")),
            "train_once_wrapper_freshness": train_once_freshness,
            "private_closure_pass_delta": round(closure_private_delta, 6) if closure_private_delta is not None else None,
            "private_signal_positive": private_signal_positive,
            "baseline_verifier_pass_rate": baseline["verifier_pass_rate"],
            "semantic_plan_verifier_pass_rate": semantic["verifier_pass_rate"],
            "contract_guided_verifier_pass_rate": contract["verifier_pass_rate"],
            "sts_conditioned_verifier_pass_rate": sts["verifier_pass_rate"],
            "contract_guided_candidate_count": contract["candidate_count"],
            "sts_conditioned_candidate_count": sts["candidate_count"],
            "public_task_count": public_generation["task_count"],
            "public_actual_token_task_coverage": public_generation["actual_token_task_coverage"],
            "public_eligible_task_coverage": public_generation["eligible_task_coverage"],
            "public_no_admissible_task_rate": public_generation["no_admissible_task_rate"],
            "public_no_admissible_task_count": public_generation["no_admissible_task_count"],
            "public_diagnostic_no_admissible_task_rate": public_generation["diagnostic_no_admissible_task_rate"],
            "public_diagnostic_no_admissible_task_count": public_generation["diagnostic_no_admissible_task_count"],
            "public_no_admissible_shadowed_by_eligible_task_count": public_generation[
                "no_admissible_shadowed_by_eligible_task_count"
            ],
            "public_no_admissible_shadowed_by_eligible_task_rate": public_generation[
                "no_admissible_shadowed_by_eligible_task_rate"
            ],
            "public_candidate_provenance_present_rate": public_provenance["provenance_present_rate"],
            "public_program_synthesis_loop_present_rate": public_provenance["program_synthesis_loop_present_rate"],
            "public_program_synthesis_contract_ir_rate": public_provenance["program_synthesis_contract_ir_rate"],
            "public_program_synthesis_ast_plan_rate": public_provenance["program_synthesis_ast_plan_rate"],
            "public_program_synthesis_parser_mask_rate": public_provenance["program_synthesis_parser_mask_rate"],
            "public_program_synthesis_ranker_rate": public_provenance["program_synthesis_ranker_rate"],
            "public_program_synthesis_promotion_ready_rate": public_provenance["program_synthesis_promotion_ready_rate"],
            "public_candidate_quality_gate_pass_rate": public_provenance["quality_gate_pass_rate"],
            "public_candidate_ast_valid_rate": public_provenance["ast_valid_rate"],
            "public_candidate_entry_point_match_rate": public_provenance["entry_point_match_rate"],
            "public_candidate_bogus_return_local_callable_count": public_provenance["bogus_return_local_callable_count"],
            "public_no_admissible_top_reasons": public_no_admissible["top_rejection_reasons"],
            "public_no_admissible_top_missing_capability_families": public_no_admissible["top_missing_capability_families"],
            "no_admissible_residual_record_count": no_admissible_packet["residual_record_count"],
            "no_admissible_policy_row_count": no_admissible_packet["policy_row_count"],
            "no_admissible_residuals_out": no_admissible_packet["packet_out"],
            "no_admissible_policy_rows_out": no_admissible_packet["policy_rows_out"],
            "private_execution_shape_candidate_coverage_ready": execution_shape_private_ready,
            "private_execution_shape_candidate_coverage_report": execution_shape_coverage.get("source_report_path"),
            "private_execution_shape_learned_token_pass_rate": execution_shape_summary.get("learned_token_decoder_pass_rate"),
            "private_execution_shape_no_admissible_rate": execution_shape_summary.get("learned_token_decoder_no_admissible_candidate_rate"),
            "decoder_relevant_source_fingerprint": decoder_fingerprint,
            "decoder_relevant_source_mtime": decoder_source_mtime or None,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "closure_reports": closure_reports,
        "candidate_groups": {name: group_metrics(rows) for name, rows in sorted(groups.items())},
        "public_candidate_groups": {name: group_metrics(rows) for name, rows in sorted(public_groups.items())},
        "public_candidate_provenance": public_provenance,
        "public_no_admissible_diagnostics": public_no_admissible,
        "no_admissible_residual_materialization": no_admissible_packet,
        "private_fail_reasons_by_group": fail_reasons,
        "gates": gates,
        "rules": {
            "public_benchmarks": "not executed here; public task manifests are inspected only as calibration candidate metadata",
            "promotion": "a public 4-card run is allowed only after private closure delta is positive and contract/STS skeleton groups are observed and non-regressive",
            "teacher": "teacher may propose architecture experiments, but this gate requires local private evidence",
        },
        "next_actions": next_actions(ready, gates),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, resolve(args.out), payload=report)
    print(json.dumps(report, indent=2))
    return 0


def closure_summary(path: Path) -> dict[str, Any]:
    report = read_json(path, {})
    rust_report_path = resolve(str(report.get("rust_report") or ""))
    rust = read_json(rust_report_path, {})
    private_candidate_manifest = first_string(
        [
            rust.get("private_candidate_manifest"),
            report.get("private_candidate_manifest"),
            get_path(report, ["summary", "private_candidate_manifest"], ""),
            get_path(report, ["paths", "private_candidates"], ""),
        ]
    )
    public_candidate_manifest = first_string(
        [
            rust.get("public_candidate_manifest"),
            report.get("public_candidate_manifest"),
            get_path(report, ["summary", "public_candidate_manifest"], ""),
            get_path(report, ["paths", "public_candidates"], ""),
        ]
    )
    candidate_manifests_present = bool(private_candidate_manifest and public_candidate_manifest)
    rust_completed = rust.get("run_status") == "completed" and candidate_manifests_present
    effective_run_status = "completed" if rust_completed else report.get("run_status")
    effective_trigger_state = rust.get("trigger_state") if rust_completed else report.get("trigger_state")
    decoder_fingerprint = first_string(
        [
            get_path(report, ["decoder_relevant_source_fingerprint"], ""),
            get_path(report, ["summary", "decoder_relevant_source_fingerprint"], ""),
            get_path(rust, ["decoder_relevant_source_fingerprint"], ""),
            get_path(rust, ["summary", "decoder_relevant_source_fingerprint"], ""),
        ]
    )
    summary_payload = object_field(report, "summary")
    rust_summary_payload = object_field(rust, "summary")
    public_manifest_diagnostics = object_field(summary_payload, "public_candidate_manifest_diagnostics")
    checkpoint_freshness = closure_checkpoint_freshness(report)
    private_only = bool(
        report.get("private_only")
        or summary_payload.get("private_only")
        or rust.get("private_only")
        or rust_summary_payload.get("private_only")
    )
    private_candidate_count = first_number(
        [
            report.get("private_candidate_count"),
            summary_payload.get("private_candidate_count"),
            rust.get("private_candidate_count"),
            rust_summary_payload.get("private_candidate_count"),
        ]
    )
    public_candidate_count = first_number(
        [
            report.get("public_candidate_count"),
            summary_payload.get("public_candidate_count"),
            rust.get("public_candidate_count"),
            rust_summary_payload.get("public_candidate_count"),
        ]
    )
    private_eval_task_count = first_number(
        [
            summary_payload.get("private_eval_task_count"),
            rust_summary_payload.get("private_eval_task_count"),
        ]
    )
    public_task_count = first_number(
        [
            summary_payload.get("public_task_count"),
            rust_summary_payload.get("public_task_count"),
            public_manifest_diagnostics.get("task_count"),
        ]
    )
    return {
        "path": rel_or_abs(path),
        "exists": path.exists(),
        "mtime": path.stat().st_mtime if path.exists() else None,
        "trigger_state": effective_trigger_state,
        "run_status": effective_run_status,
        "outer_run_status": report.get("run_status"),
        "rust_completed_without_outer_wrapper": rust_completed and report.get("run_status") != "completed",
        "diagnostic_only": effective_run_status != "completed"
        or rust.get("run_status") == "timed_out_process_tree_killed",
        "rust_report": rel_or_abs(rust_report_path) if rust_report_path.exists() else str(report.get("rust_report") or ""),
        "rust_trigger_state": rust.get("trigger_state"),
        "rust_run_status": rust.get("run_status"),
        "private_candidate_manifest": private_candidate_manifest,
        "public_candidate_manifest": public_candidate_manifest,
        "private_only": private_only,
        "partial_private_candidate_rows": get_path(rust, ["summary", "partial_private_candidate_rows"], None),
        "partial_public_candidate_rows": get_path(rust, ["summary", "partial_public_candidate_rows"], None),
        "private_candidate_count": private_candidate_count,
        "public_candidate_count": public_candidate_count,
        "private_eval_task_count": private_eval_task_count,
        "public_task_count": public_task_count,
        "train_once_checkpoint_fanout": bool(summary_payload.get("train_once_checkpoint_fanout")),
        "checkpoint_cuda_readout_used": bool(summary_payload.get("checkpoint_cuda_readout_used")),
        "checkpoint_current_against_private_training_rows": checkpoint_freshness["fresh"],
        "checkpoint_freshness": checkpoint_freshness,
        "public_token_level_candidate_count": first_number(
            [
                summary_payload.get("public_token_level_candidate_count"),
                public_manifest_diagnostics.get("token_level_candidate_count"),
            ]
        ),
        "public_program_synthesis_loop_rate": first_number(
            [
                summary_payload.get("public_program_synthesis_loop_present_rate"),
                public_manifest_diagnostics.get("program_synthesis_loop_rate"),
            ]
        ),
        "public_program_synthesis_promotion_ready_rate": first_number(
            [
                summary_payload.get("public_program_synthesis_promotion_ready_rate"),
                public_manifest_diagnostics.get("promotion_ready_rate"),
            ]
        ),
        "public_verifier_pass_rate": first_number(
            [
                summary_payload.get("public_candidate_quality_gate_pass_rate"),
                public_manifest_diagnostics.get("verifier_pass_rate"),
            ]
        ),
        "template_like_candidate_count": first_number(
            [
                summary_payload.get("template_like_candidate_count"),
                public_manifest_diagnostics.get("template_like_candidate_count"),
            ]
        ),
        "scale_diagnostic_only": closure_scale_diagnostic_only(
            path=path,
            private_only=private_only,
            private_candidate_count=private_candidate_count,
            public_candidate_count=public_candidate_count,
            private_eval_task_count=private_eval_task_count,
            public_task_count=public_task_count,
        ),
        "closure_report": report,
        "rust_report_payload": rust,
        "decoder_relevant_source_fingerprint": decoder_fingerprint,
    }


def closure_checkpoint_freshness(report: dict[str, Any]) -> dict[str, Any]:
    summary_payload = object_field(report, "summary")
    private_input_freshness = object_field(summary_payload, "private_input_freshness")
    checkpoint_gate = named_gate_passed(report, "checkpoint_current_against_private_training_rows")
    fanout_gate = named_gate_passed(report, "fanout_artifacts_current_against_source_binary_provenance")
    private_inputs_fresh = private_input_freshness.get("fresh")
    fresh = True
    if private_input_freshness:
        fresh = fresh and bool(private_inputs_fresh)
    if checkpoint_gate is not None:
        fresh = fresh and bool(checkpoint_gate)
    if fanout_gate is not None:
        fresh = fresh and bool(fanout_gate)
    return {
        "fresh": fresh,
        "private_input_freshness": private_input_freshness or None,
        "checkpoint_current_gate": checkpoint_gate,
        "fanout_current_gate": fanout_gate,
        "rule": (
            "private closure evidence can describe candidate quality, but cannot unlock public calibration "
            "unless the materialized private curriculum, checkpoint, checkpoint report, fanout manifests, "
            "and release binary/source provenance are current"
        ),
    }


def train_once_freshness_for_latest(latest: dict[str, Any] | None, wrapper: dict[str, Any]) -> dict[str, Any]:
    """Require the canonical train-once wrapper to agree when it owns the latest closure."""
    if not latest:
        return {
            "passed": True,
            "applicable": False,
            "reason": "no latest closure; private_closure_present gate owns this failure",
        }
    if not wrapper:
        return {
            "passed": True,
            "applicable": False,
            "reason": "canonical train-once wrapper report missing; closure-level freshness gates apply",
        }
    wrapper_paths = object_field(wrapper, "paths")
    wrapper_closure_raw = str(wrapper_paths.get("closure_report") or "")
    wrapper_closure = resolve(wrapper_closure_raw) if wrapper_closure_raw else Path()
    latest_path = resolve(str(latest.get("path") or ""))
    slug = str(wrapper.get("slug") or "").strip()
    latest_key = normalized_path_key(latest_path)
    wrapper_key = normalized_path_key(wrapper_closure) if wrapper_closure_raw else ""
    applies = bool(
        (wrapper_key and latest_key == wrapper_key)
        or (wrapper_closure_raw and latest_path.name == Path(wrapper_closure_raw).name)
        or (slug and slug.lower() in latest_key)
    )
    if not applies:
        return {
            "passed": True,
            "applicable": False,
            "latest_closure": latest.get("path"),
            "train_once_slug": slug,
            "train_once_closure_report": rel_or_abs(wrapper_closure) if wrapper_closure_raw else "",
            "rule": "wrapper freshness is enforced only when the selected closure belongs to that train-once slug",
        }
    private_input_freshness = get_path(wrapper, ["artifact_provenance", "private_input_freshness"], {})
    if not isinstance(private_input_freshness, dict):
        private_input_freshness = {}
    if not private_input_freshness:
        private_input_freshness = object_field(wrapper, "private_input_freshness")
    completed = wrapper.get("run_status") in {"completed", "completed_existing_artifacts"}
    inputs_fresh = bool(
        private_input_freshness.get("fresh")
        or get_path(wrapper, ["artifact_provenance", "summary", "private_inputs_fresh"], False)
    )
    closure_report_available = bool(wrapper_closure_raw and wrapper_paths.get("closure_report"))
    passed = bool(completed and inputs_fresh and closure_report_available)
    return {
        "passed": passed,
        "applicable": True,
        "train_once_slug": slug,
        "train_once_trigger_state": wrapper.get("trigger_state"),
        "train_once_run_status": wrapper.get("run_status"),
        "train_once_current_phase": wrapper.get("current_phase"),
        "train_once_closure_report": rel_or_abs(wrapper_closure) if wrapper_closure_raw else "",
        "latest_closure": latest.get("path"),
        "private_input_freshness": private_input_freshness or None,
        "completed": completed,
        "private_inputs_fresh": inputs_fresh,
        "closure_report_available": closure_report_available,
        "rule": (
            "when the selected closure is owned by the canonical train-once wrapper, decoder gates must use "
            "the wrapper's live freshness state; stale private inputs make old closure manifests diagnostic-only"
        ),
    }


def normalized_path_key(path: Path) -> str:
    try:
        return str(path.resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path).replace("\\", "/").lower()


def named_gate_passed(report: dict[str, Any], name: str) -> bool | None:
    rows = report.get("gates")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("gate") == name:
            return bool(row.get("passed"))
    return None


def default_closure_reports() -> list[Path]:
    discovered = [
        path
        for path in REPORTS.glob("code_lm_closure*.json")
        if ".heartbeat." not in path.name
        and "public_contract_preflight" not in path.name
        and "partial_artifact" not in path.name
    ]
    return unique_paths([*DEFAULT_CLOSURE_REPORTS, *discovered])


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def latest_usable_closure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get("exists")
        and row.get("run_status") == "completed"
        and not row.get("diagnostic_only")
        and row.get("trigger_state") in {"GREEN", "YELLOW"}
        and row.get("private_candidate_manifest")
        and row.get("public_candidate_manifest")
    ]
    if not usable:
        return {}
    return max(usable, key=closure_selection_score)


def closure_scale_diagnostic_only(
    *,
    path: Path,
    private_only: bool,
    private_candidate_count: Any,
    public_candidate_count: Any,
    private_eval_task_count: Any,
    public_task_count: Any,
) -> bool:
    name = path.name.lower()
    if "smoke" in name or "current_source_smoke" in name:
        return True
    if "checkpoint" in name and "fanout" not in name:
        return True
    public_candidates = int(number(public_candidate_count) or 0)
    private_candidates = int(number(private_candidate_count) or 0)
    private_eval_tasks = int(number(private_eval_task_count) or 0)
    public_tasks = int(number(public_task_count) or 0)
    if private_only:
        return private_candidates < 256 or private_eval_tasks < 32
    return (
        public_tasks < MIN_PROMOTION_PUBLIC_TASKS
        or public_candidates < min_promotion_public_candidate_count(public_tasks)
        or private_candidates < 256
    )


def min_promotion_public_candidate_count(public_task_count: int) -> int:
    """Minimum learned-token public inventory for a bounded promotion fanout."""
    if public_task_count < MIN_PROMOTION_PUBLIC_TASKS:
        return MIN_PROMOTION_PUBLIC_CANDIDATES
    return max(
        MIN_PROMOTION_PUBLIC_CANDIDATES,
        public_task_count * MIN_PROMOTION_PUBLIC_CANDIDATES_PER_TASK,
    )


def closure_selection_score(row: dict[str, Any]) -> tuple[Any, ...]:
    name = Path(str(row.get("path") or "")).name.lower()
    smoke_or_tiny = bool(row.get("scale_diagnostic_only")) or "smoke" in name or "checkpoint" in name
    public_candidate_count = int(number(row.get("public_candidate_count")) or 0)
    private_candidate_count = int(number(row.get("private_candidate_count")) or 0)
    public_task_count = int(number(row.get("public_task_count")) or 0)
    private_eval_task_count = int(number(row.get("private_eval_task_count")) or 0)
    public_token_count = int(number(row.get("public_token_level_candidate_count")) or 0)
    program_loop_rate = float(number(row.get("public_program_synthesis_loop_rate")) or 0.0)
    promotion_ready_rate = float(number(row.get("public_program_synthesis_promotion_ready_rate")) or 0.0)
    verifier_pass_rate = float(number(row.get("public_verifier_pass_rate")) or 0.0)
    template_like_count = int(number(row.get("template_like_candidate_count")) or 0)
    source_current = closure_decoder_current(
        report_fingerprint=str(row.get("decoder_relevant_source_fingerprint") or ""),
        current_fingerprint=decoder_relevant_source_fingerprint(),
        closure_mtime=float(row.get("mtime") or 0.0),
        decoder_source_mtime=decoder_relevant_source_mtime(),
    )
    return (
        not smoke_or_tiny,
        source_current,
        float(row.get("mtime") or 0.0),
        bool(row.get("train_once_checkpoint_fanout")),
        bool(row.get("checkpoint_cuda_readout_used")),
        private_candidate_count >= 256,
        private_eval_task_count >= 32,
        program_loop_rate >= 0.75,
        promotion_ready_rate >= 0.50,
        verifier_pass_rate >= 0.75,
        template_like_count == 0,
        public_task_count >= MIN_PROMOTION_PUBLIC_TASKS,
        public_candidate_count >= min_promotion_public_candidate_count(public_task_count),
        public_task_count,
        public_candidate_count,
        private_eval_task_count,
        private_candidate_count,
        public_token_count,
        program_loop_rate,
        promotion_ready_rate,
        verifier_pass_rate,
    )


def latest_diagnostic_partial_closure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    partial = [
        row
        for row in rows
        if row.get("exists")
        and row.get("diagnostic_only")
        and (row.get("private_candidate_manifest") or row.get("public_candidate_manifest"))
    ]
    if not partial:
        return {}
    return max(partial, key=lambda row: float(row.get("mtime") or 0.0))


def private_delta(report: dict[str, Any]) -> float | None:
    candidates = [
        ["summary", "private_pass_rate_delta"],
        ["summary", "private_eval_pass_rate_delta"],
        ["summary", "private_eval_delta"],
        ["summary", "improvement_contract", "private_delta"],
        ["rust_report_payload", "summary", "private_pass_rate_delta"],
        ["rust_report_payload", "summary", "private_eval_pass_rate_delta"],
    ]
    for path in candidates:
        value = get_path(report, path, None)
        if value is not None:
            return number(value)
    before = get_path(report, ["summary", "private_eval_before", "pass_rate"], None)
    after = get_path(report, ["summary", "private_eval_after", "pass_rate"], None)
    if before is not None and after is not None:
        return number(after) - number(before)
    return None


def group_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = str(row.get("candidate_generation_mode") or row.get("generation_mode") or row.get("mode") or row.get("candidate_mode") or "")
        lowered = mode.lower()
        if "sts_conditioned" in lowered and (
            "skeleton" in lowered
            or "semantic_plan_v2" in lowered
            or "edge_exec_repair" in lowered
            or "contract_guided_token" in lowered
            or "contract_guided" in lowered
        ):
            groups["sts_conditioned_skeleton"].append(row)
        elif non_sts_contract_guided_token_inventory(row, lowered):
            groups["contract_guided"].append(row)
        elif "contract_guided_token" in lowered:
            groups["contract_guided"].append(row)
        elif "contract_guided_skeleton" in lowered:
            groups["contract_guided"].append(row)
        elif "local_adapter_edge_skeleton" in lowered:
            groups["contract_guided"].append(row)
        elif "execution_shape_skeleton" in lowered or "edge_exec_repair" in lowered:
            groups["contract_guided"].append(row)
        elif "semantic_plan_v2" in lowered:
            groups["semantic_plan"].append(row)
        else:
            groups["baseline"].append(row)
    return dict(groups)


def non_sts_contract_guided_token_inventory(row: dict[str, Any], lowered_mode: str) -> bool:
    if "sts_conditioned" in lowered_mode:
        return False
    if any(
        needle in lowered_mode
        for needle in (
            "same_seed_non_sts_comparator",
            "contract_transduced_token_decoder",
            "prompt_program_decoder",
            "skeleton",
            "prototype",
            "ngram",
            "semantic_plan",
        )
    ):
        return False
    if "_token_decoder" not in lowered_mode and "token_decoder" not in lowered_mode:
        return False
    if not (
        "eligible_receiver_inventory_router_v1" in lowered_mode
        or "private_to_public_receiver_inventory_bridge_v1" in lowered_mode
    ):
        return False
    if row.get("template_like_candidate") or row.get("expression_memory_fallback"):
        return False
    if row.get("public_tests_visible_to_generator") or row.get("canonical_solution_seen_by_solver"):
        return False
    loop = row.get("program_synthesis_loop_v1")
    has_program_loop = isinstance(loop, dict) and bool(loop.get("contract_ir")) and bool(loop.get("ranker"))
    return bool(row.get("token_level_code_generation_learned")) or has_program_loop


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    passed = sum(1 for row in rows if bool(row.get("decoder_contract_verifier_v1_passed")))
    task_ids = {str(row.get("task_id") or row.get("source_task_id") or "") for row in rows if row.get("task_id") or row.get("source_task_id")}
    modes = Counter(str(row.get("candidate_generation_mode") or row.get("generation_mode") or row.get("mode") or "unknown") for row in rows)
    reasons = reason_counts(rows)
    return {
        "candidate_count": count,
        "task_count": len(task_ids),
        "verifier_pass_count": passed,
        "verifier_fail_count": count - passed,
        "verifier_pass_rate": round(passed / count, 6) if count else 0.0,
        "top_modes": dict(modes.most_common(10)),
        "top_fail_reasons": dict(reasons.most_common(10)),
    }


def reason_counts(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        reasons = row.get("decoder_contract_verifier_v1_reasons")
        if isinstance(reasons, str):
            reasons = [reasons]
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            text = str(reason)
            if text:
                counts[text] += 1
    return counts


def public_candidate_clean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad = []
    for row in rows:
        if row.get("canonical_solution_seen_by_solver") or row.get("public_tests_visible_to_generator"):
            bad.append(str(row.get("task_id") or row.get("row_id") or "unknown"))
    return {
        "ok": not bad,
        "candidate_count": len(rows),
        "bad_rows": bad[:20],
        "rule": "public candidate metadata may be inspected; public solutions/tests cannot feed generation or training",
    }


def no_admissible_candidate_row(row: dict[str, Any]) -> bool:
    mode = candidate_mode(row).lower()
    scope = str(row.get("candidate_program_scope") or "").lower()
    contract = str(row.get("candidate_generation_contract") or "").lower()
    source_type = str(row.get("source_type") or "").lower()
    if mode == "student_decoder_no_admissible_candidate_residual":
        return True
    if scope == "no_admissible_candidate_residual":
        return True
    if source_type == "no_admissible_candidate_residual":
        return True
    if contract == "student_decoder_no_admissible_candidate_residual_not_promotion_evidence":
        return True
    return False


def actual_learned_token_candidate(row: dict[str, Any]) -> bool:
    return (
        grammar_masked_learned_token_candidate(row)
        and row.get("deterministic_guardrail_passed") is not False
        and not no_admissible_candidate_row(row)
    )


def promotion_eligible_learned_token_candidate(row: dict[str, Any]) -> bool:
    return (
        actual_learned_token_candidate(row)
        and bool(row.get("benchmark_promotion_eligible"))
        and row.get("decoder_contract_verifier_v1_passed") is not False
        and not bool(row.get("template_like_candidate"))
        and not bool(row.get("loop_closure_generated"))
        and not bool(row.get("expression_memory_fallback"))
        and not bool(row.get("placeholder_scaffold_body"))
        and not bogus_return_attribute_code(candidate_code(row))
        and not bogus_return_local_callable_code(candidate_code(row))
    )


def public_candidate_generation_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = {
        str(row.get("task_id") or row.get("source_task_id") or "")
        for row in rows
        if row.get("task_id") or row.get("source_task_id")
    }
    actual_token_tasks = set()
    eligible_tasks = set()
    diagnostic_no_admissible_tasks = set()
    guardrail_failed_tasks = set()
    mode_counts: Counter[str] = Counter()
    for row in rows:
        task_id = str(row.get("task_id") or row.get("source_task_id") or "")
        if not task_id:
            continue
        mode = str(row.get("candidate_generation_mode") or row.get("generation_mode") or row.get("mode") or "unknown")
        mode_counts[mode] += 1
        deterministic_ok = row.get("deterministic_guardrail_passed") is not False
        no_admissible = no_admissible_candidate_row(row)
        if no_admissible:
            diagnostic_no_admissible_tasks.add(task_id)
        if not deterministic_ok:
            guardrail_failed_tasks.add(task_id)
        actual_token = actual_learned_token_candidate(row)
        if actual_token:
            actual_token_tasks.add(task_id)
        if promotion_eligible_learned_token_candidate(row):
            eligible_tasks.add(task_id)
    task_count = max(1, len(task_ids))
    unresolved_no_admissible_tasks = diagnostic_no_admissible_tasks - eligible_tasks
    shadowed_no_admissible_tasks = diagnostic_no_admissible_tasks & eligible_tasks
    return {
        "task_count": len(task_ids),
        "actual_token_task_count": len(actual_token_tasks),
        "eligible_task_count": len(eligible_tasks),
        "no_admissible_task_count": len(unresolved_no_admissible_tasks),
        "diagnostic_no_admissible_task_count": len(diagnostic_no_admissible_tasks),
        "no_admissible_shadowed_by_eligible_task_count": len(shadowed_no_admissible_tasks),
        "guardrail_failed_task_count": len(guardrail_failed_tasks),
        "actual_token_task_coverage": round(len(actual_token_tasks) / task_count, 6),
        "eligible_task_coverage": round(len(eligible_tasks) / task_count, 6),
        "no_admissible_task_rate": round(len(unresolved_no_admissible_tasks) / task_count, 6),
        "diagnostic_no_admissible_task_rate": round(len(diagnostic_no_admissible_tasks) / task_count, 6),
        "no_admissible_shadowed_by_eligible_task_rate": round(len(shadowed_no_admissible_tasks) / task_count, 6),
        "guardrail_failed_task_rate": round(len(guardrail_failed_tasks) / task_count, 6),
        "top_modes": dict(mode_counts.most_common(8)),
        "rule": "receiver calibration requires broad learned token-candidate coverage; no_admissible_task_rate counts unresolved tasks only, while diagnostic_no_admissible_task_rate preserves shadowed branch diagnostics",
    }


def candidate_provenance_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual_rows = [
        row
        for row in rows
        if grammar_masked_learned_token_candidate(row) and not no_admissible_candidate_row(row)
    ]
    provenance_present = 0
    ast_valid = 0
    entry_point_match = 0
    public_boundary_clean = 0
    quality_pass = 0
    program_loop_present = 0
    program_contract_ir = 0
    program_ast_plan = 0
    program_constrained_decode = 0
    program_parser_mask = 0
    program_verifier_repair = 0
    program_ranker = 0
    program_promotion_ready = 0
    program_clean_boundary = 0
    bogus_attr = 0
    bogus_local_call = 0
    promotion_quality_candidate_count = 0
    non_promotion_diagnostic_count = 0
    same_seed_comparator_diagnostic_count = 0
    quality_failure_reasons: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    diagnostic_samples: list[dict[str, Any]] = []
    for row in actual_rows:
        code = candidate_code(row)
        provenance = row.get("provenance")
        if isinstance(provenance, dict) and provenance:
            provenance_present += 1
        program_loop = program_synthesis_loop(row)
        contract_ir = object_field(program_loop, "contract_ir")
        ast_plan = object_field(program_loop, "ast_plan_latent")
        decode_control = object_field(program_loop, "decode_control")
        verifier_repair = object_field(program_loop, "verifier_repair")
        ranker = object_field(program_loop, "ranker")
        program_loop_ok = bool(program_loop)
        contract_ir_ok = bool(contract_ir.get("entry_point")) and isinstance(contract_ir.get("visible_args"), list)
        ast_plan_ok = bool(ast_plan.get("causal_contract_order")) and bool(ast_plan.get("semantic_plan"))
        constrained_decode_ok = bool(decode_control.get("constrained_token_decode"))
        parser_mask_ok = bool(decode_control.get("parser_contract_mask"))
        verifier_repair_ok = bool(verifier_repair.get("stage_present"))
        ranker_ok = all(
            value is not None
            for value in [
                ranker.get("beautiful_code_score"),
                ranker.get("body_transfer_score"),
                ranker.get("contract_guided_score"),
            ]
        )
        promotion_ready_ok = bool(program_loop.get("promotion_ready"))
        boundary_ok = bool(contract_ir.get("public_tests_used") is False and contract_ir.get("public_solutions_used") is False)
        if program_loop_ok:
            program_loop_present += 1
        if contract_ir_ok:
            program_contract_ir += 1
        if ast_plan_ok:
            program_ast_plan += 1
        if constrained_decode_ok:
            program_constrained_decode += 1
        if parser_mask_ok:
            program_parser_mask += 1
        if verifier_repair_ok:
            program_verifier_repair += 1
        if ranker_ok:
            program_ranker += 1
        if promotion_ready_ok:
            program_promotion_ready += 1
        if boundary_ok:
            program_clean_boundary += 1
        parsed = parse_code(code)
        if parsed is not None:
            ast_valid += 1
        entry = str(row.get("entry_point") or nested_visible_task_field(row, "entry_point") or "")
        entry_point_ok = parsed is not None and function_entry_point_matches(parsed, entry)
        if entry_point_ok:
            entry_point_match += 1
        if not row.get("canonical_solution_seen_by_solver") and not row.get("public_tests_visible_to_generator"):
            public_boundary_clean += 1
        bad_attr = bogus_return_attribute_code(code)
        bad_local = bogus_return_local_callable_code(code)
        if bad_attr:
            bogus_attr += 1
        if bad_local:
            bogus_local_call += 1
        score = number(row.get("beautiful_code_score"))
        promotion_candidate = bool(row.get("benchmark_promotion_eligible")) and grammar_masked_learned_token_candidate(row)
        same_seed_comparator = bool(row.get("same_seed_non_sts_comparator")) or (
            "same_seed_non_sts_comparator" in candidate_mode(row).lower()
        )
        if promotion_candidate:
            promotion_quality_candidate_count += 1
        else:
            non_promotion_diagnostic_count += 1
            if same_seed_comparator:
                same_seed_comparator_diagnostic_count += 1
        quality_ok = (
            parsed is not None
            and promotion_candidate
            and row.get("decoder_contract_verifier_v1_passed") is not False
            and row.get("deterministic_guardrail_passed") is not False
            and not bool(row.get("template_like_candidate"))
            and not bool(row.get("placeholder_scaffold_body"))
            and not bool(row.get("expression_memory_fallback"))
            and not bool(row.get("loop_closure_generated"))
            and program_loop_ok
            and contract_ir_ok
            and ast_plan_ok
            and constrained_decode_ok
            and parser_mask_ok
            and verifier_repair_ok
            and ranker_ok
            and promotion_ready_ok
            and boundary_ok
            and not bad_attr
            and not bad_local
            and (score is None or float(score) >= 1.0)
        )
        if quality_ok:
            quality_pass += 1
        if not quality_ok and promotion_candidate:
            reasons = []
            if parsed is None:
                reasons.append("python_ast_parse_failed")
            elif not entry_point_ok:
                reasons.append("entry_point_mismatch")
            if row.get("decoder_contract_verifier_v1_passed") is False:
                reasons.append("decoder_contract_failed")
            if row.get("deterministic_guardrail_passed") is False:
                reasons.append("deterministic_guardrail_failed")
            if bool(row.get("template_like_candidate")):
                reasons.append("template_like_candidate")
            if bool(row.get("placeholder_scaffold_body")):
                reasons.append("placeholder_scaffold_body")
            if bool(row.get("expression_memory_fallback")):
                reasons.append("expression_memory_fallback")
            if bool(row.get("loop_closure_generated")):
                reasons.append("loop_closure_generated")
            if not program_loop_ok:
                reasons.append("program_synthesis_loop_missing")
            if not contract_ir_ok:
                reasons.append("contract_ir_incomplete")
            if not ast_plan_ok:
                reasons.append("ast_plan_incomplete")
            if not parser_mask_ok:
                reasons.append("parser_contract_mask_missing")
            if not constrained_decode_ok:
                reasons.append("constrained_token_decode_missing")
            if not verifier_repair_ok:
                reasons.append("verifier_repair_missing")
            if not ranker_ok:
                reasons.append("ranker_incomplete")
            if not promotion_ready_ok:
                reasons.append("program_synthesis_not_promotion_ready")
            if not boundary_ok:
                reasons.append("public_boundary_not_clean")
            if bad_attr:
                reasons.append("bogus_return_attribute")
            if bad_local:
                reasons.append("bogus_return_local_callable")
            if score is not None and float(score) < 1.0:
                reasons.append("beautiful_code_score_below_floor")
            if not reasons:
                reasons.append("quality_gate_failed_unknown")
            quality_failure_reasons.update(reasons)
        if not quality_ok and promotion_candidate and len(samples) < 16:
            samples.append(
                {
                    "task_id": row.get("task_id") or row.get("source_task_id"),
                    "entry_point": entry,
                    "mode": candidate_mode(row),
                    "quality_failure_reasons": reasons,
                    "promotion_eligible": row.get("benchmark_promotion_eligible"),
                    "decoder_contract_passed": row.get("decoder_contract_verifier_v1_passed"),
                    "deterministic_guardrail_passed": row.get("deterministic_guardrail_passed"),
                    "beautiful_code_score": row.get("beautiful_code_score"),
                    "program_synthesis_loop_present": program_loop_ok,
                    "program_synthesis_contract_ir": contract_ir_ok,
                    "program_synthesis_ast_plan": ast_plan_ok,
                    "program_synthesis_parser_mask": parser_mask_ok,
                    "program_synthesis_ranker": ranker_ok,
                    "program_synthesis_promotion_ready": promotion_ready_ok,
                    "bogus_return_attribute": bad_attr,
                    "bogus_return_local_callable": bad_local,
                    "body_preview": " | ".join(part.strip() for part in code.splitlines() if part.strip())[:360],
                }
            )
        if not promotion_candidate and len(diagnostic_samples) < 8:
            diagnostic_samples.append(
                {
                    "task_id": row.get("task_id") or row.get("source_task_id"),
                    "entry_point": entry,
                    "mode": candidate_mode(row),
                    "same_seed_non_sts_comparator": same_seed_comparator,
                    "candidate_generation_contract": row.get("candidate_generation_contract"),
                    "promotion_eligible": row.get("benchmark_promotion_eligible"),
                    "quality_accounting": "diagnostic_only_excluded_from_promotion_quality_denominator",
                    "body_preview": " | ".join(part.strip() for part in code.splitlines() if part.strip())[:240],
                }
            )
    denom = max(1, len(actual_rows))
    quality_denom = max(1, promotion_quality_candidate_count)
    return {
        "actual_token_candidate_count": len(actual_rows),
        "promotion_quality_candidate_count": promotion_quality_candidate_count,
        "non_promotion_diagnostic_candidate_count": non_promotion_diagnostic_count,
        "same_seed_non_sts_comparator_diagnostic_count": same_seed_comparator_diagnostic_count,
        "provenance_present_count": provenance_present,
        "ast_valid_count": ast_valid,
        "entry_point_match_count": entry_point_match,
        "public_boundary_clean_count": public_boundary_clean,
        "quality_gate_pass_count": quality_pass,
        "program_synthesis_loop_present_count": program_loop_present,
        "program_synthesis_contract_ir_count": program_contract_ir,
        "program_synthesis_ast_plan_count": program_ast_plan,
        "program_synthesis_constrained_decode_count": program_constrained_decode,
        "program_synthesis_parser_mask_count": program_parser_mask,
        "program_synthesis_verifier_repair_count": program_verifier_repair,
        "program_synthesis_ranker_count": program_ranker,
        "program_synthesis_promotion_ready_count": program_promotion_ready,
        "program_synthesis_clean_boundary_count": program_clean_boundary,
        "bogus_return_attribute_count": bogus_attr,
        "bogus_return_local_callable_count": bogus_local_call,
        "provenance_present_rate": round(provenance_present / denom, 6),
        "ast_valid_rate": round(ast_valid / denom, 6),
        "entry_point_match_rate": round(entry_point_match / denom, 6),
        "public_boundary_clean_rate": round(public_boundary_clean / denom, 6),
        "quality_gate_pass_rate": round(quality_pass / quality_denom, 6),
        "quality_gate_denominator": promotion_quality_candidate_count,
        "quality_failure_reasons": dict(quality_failure_reasons.most_common()),
        "program_synthesis_loop_present_rate": round(program_loop_present / denom, 6),
        "program_synthesis_contract_ir_rate": round(program_contract_ir / denom, 6),
        "program_synthesis_ast_plan_rate": round(program_ast_plan / denom, 6),
        "program_synthesis_constrained_decode_rate": round(program_constrained_decode / denom, 6),
        "program_synthesis_parser_mask_rate": round(program_parser_mask / denom, 6),
        "program_synthesis_verifier_repair_rate": round(program_verifier_repair / denom, 6),
        "program_synthesis_ranker_rate": round(program_ranker / denom, 6),
        "program_synthesis_promotion_ready_rate": round(program_promotion_ready / denom, 6),
        "program_synthesis_clean_boundary_rate": round(program_clean_boundary / denom, 6),
        "failed_quality_samples": samples,
        "non_promotion_diagnostic_samples": diagnostic_samples,
        "rule": "public receiver unlock quality is measured over benchmark-promotion candidates; same-seed non-STS comparator rows remain diagnostic causal evidence and are reported separately",
    }


def public_no_admissible_diagnostics(rows: list[dict[str, Any]], sample_limit: int = 16) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or row.get("source_task_id") or "")
        if task_id:
            by_task[task_id].append(row)

    rejection_reasons: Counter[str] = Counter()
    missing_families: Counter[str] = Counter()
    required_constructs: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    diagnostic_no_admissible_task_count = 0
    unresolved_no_admissible_task_count = 0
    shadowed_no_admissible_task_count = 0
    zero_eligible_task_count = 0

    for task_id, task_rows in sorted(by_task.items()):
        accepted = [
            row
            for row in task_rows
            if row.get("decoder_contract_verifier_v1_passed") is not False
            and "no_admissible" not in candidate_mode(row).lower()
            and not bool(row.get("placeholder_scaffold_body"))
            and not bool(row.get("template_like_candidate"))
            and not bool(row.get("expression_memory_fallback"))
            and not bogus_return_attribute_code(candidate_code(row))
            and not bogus_return_local_callable_code(candidate_code(row))
        ]
        promotion_eligible = [row for row in task_rows if promotion_eligible_learned_token_candidate(row)]
        no_admissible_rows = [row for row in task_rows if no_admissible_candidate_row(row)]
        if not promotion_eligible:
            zero_eligible_task_count += 1
        if no_admissible_rows:
            diagnostic_no_admissible_task_count += 1
            if promotion_eligible:
                shadowed_no_admissible_task_count += 1
            else:
                unresolved_no_admissible_task_count += 1

        if no_admissible_rows and promotion_eligible:
            status = "shadowed_by_promotion_eligible_candidate"
        elif no_admissible_rows:
            status = "unresolved_no_admissible"
        elif not promotion_eligible:
            status = "zero_promotion_eligible_without_no_admissible_diagnostic"
        else:
            status = "resolved"

        if status != "resolved":
            for row in task_rows:
                mode_counts[candidate_mode(row)] += 1
                for reason in row_rejection_reasons(row):
                    rejection_reasons[reason] += 1
                family = str(row.get("missing_capability_family") or "")
                if family:
                    missing_families[family] += 1
                plan = object_field(row, "semantic_decoder_v2_plan")
                for construct in list_field(plan, "required_constructs"):
                    required_constructs[str(construct)] += 1

        if (not promotion_eligible or no_admissible_rows) and len(samples) < sample_limit:
            exemplar = no_admissible_rows[0] if no_admissible_rows else task_rows[0]
            samples.append(
                {
                    "task_id": task_id,
                    "source_task_id": exemplar.get("source_task_id") or get_path(exemplar, ["provenance", "visible_task", "source_task_id"], ""),
                    "category": exemplar.get("category") or nested_visible_task_field(exemplar, "category"),
                    "entry_point": exemplar.get("entry_point") or nested_visible_task_field(exemplar, "entry_point"),
                    "accepted_candidate_count": len(accepted),
                    "promotion_eligible_candidate_count": len(promotion_eligible),
                    "diagnostic_no_admissible_row_count": len(no_admissible_rows),
                    "no_admissible_status": status,
                    "raw_candidate_count": len(task_rows),
                    "raw_modes": dict(Counter(candidate_mode(row) for row in task_rows).most_common(8)),
                    "rejection_reasons": dict(Counter(reason for row in task_rows for reason in row_rejection_reasons(row)).most_common(8)),
                    "missing_capability_family": exemplar.get("missing_capability_family"),
                    "task_contract": task_contract_summary(exemplar),
                    "sample_bodies": sample_candidate_bodies(task_rows),
                }
            )

    task_count = max(1, len(by_task))
    return {
        "task_count": len(by_task),
        "zero_eligible_task_count": zero_eligible_task_count,
        "zero_eligible_task_rate": round(zero_eligible_task_count / task_count, 6),
        "no_admissible_task_count": unresolved_no_admissible_task_count,
        "no_admissible_task_rate": round(unresolved_no_admissible_task_count / task_count, 6),
        "diagnostic_no_admissible_task_count": diagnostic_no_admissible_task_count,
        "diagnostic_no_admissible_task_rate": round(diagnostic_no_admissible_task_count / task_count, 6),
        "no_admissible_shadowed_by_eligible_task_count": shadowed_no_admissible_task_count,
        "no_admissible_shadowed_by_eligible_task_rate": round(shadowed_no_admissible_task_count / task_count, 6),
        "top_rejection_reasons": dict(rejection_reasons.most_common(16)),
        "top_missing_capability_families": dict(missing_families.most_common(16)),
        "top_required_constructs_on_blocked_tasks": dict(required_constructs.most_common(16)),
        "top_modes": dict(mode_counts.most_common(16)),
        "sample_limit": sample_limit,
        "samples": samples,
        "rule": "diagnostic only: public benchmark prompts/metadata may explain missing coverage, but public tests/solutions remain unavailable to training; unresolved no-admissible excludes diagnostic rows shadowed by promotion-eligible learned-token candidates",
    }


def candidate_mode(row: dict[str, Any]) -> str:
    return str(row.get("candidate_generation_mode") or row.get("generation_mode") or row.get("mode") or "unknown")


def grammar_masked_learned_token_candidate(row: dict[str, Any]) -> bool:
    mode = candidate_mode(row).lower()
    if no_admissible_candidate_row(row):
        return False
    if any(
        token in mode
        for token in [
            "prompt_program_decoder",
            "same_seed_non_sts_comparator",
            "skeleton",
            "prototype",
            "ngram",
            "semantic_plan",
            "native_sts_stream_expression",
        ]
    ):
        return False
    has_learned_token_mode = any(
        token in mode
        for token in [
            "token_decoder",
            "contract_transduced_token_decoder",
            "full_body_token_beam",
            "greedy_body_token_decoder",
        ]
    )
    loop = program_synthesis_loop(row)
    decode_control = object_field(loop, "decode_control")
    return bool(
        has_learned_token_mode
        and bool(row.get("full_body_token_candidate"))
        and bool(row.get("compositional_token_candidate"))
        and not bool(row.get("template_like_candidate"))
        and not bool(row.get("expression_memory_fallback"))
        and not bool(row.get("sts_candidate_expression_used"))
        and not bool(row.get("same_seed_non_sts_comparator"))
        and bool(decode_control.get("constrained_token_decode"))
        and bool(decode_control.get("parser_contract_mask"))
        and bool(decode_control.get("exact_interface_claim"))
        and not bool(decode_control.get("template_or_memory_fallback"))
    )


def row_rejection_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in [
        "decoder_contract_verifier_v1_reasons",
        "deterministic_guardrail_reasons",
        "candidate_rejection_reasons",
    ]:
        value = row.get(key)
        if isinstance(value, str):
            reasons.append(value)
        elif isinstance(value, list):
            reasons.extend(str(item) for item in value if str(item))
    counts = row.get("candidate_rejection_counts")
    if isinstance(counts, dict):
        for reason, count in counts.items():
            try:
                n = int(count)
            except Exception:
                n = 1
            reasons.extend([str(reason)] * max(1, min(n, 8)))
    if "no_admissible" in candidate_mode(row).lower() and not reasons:
        reasons.append("no_admissible_candidate")
    return [reason for reason in reasons if reason]


def program_synthesis_loop(row: dict[str, Any]) -> dict[str, Any]:
    direct = object_field(row, "program_synthesis_loop_v1")
    if direct:
        return direct
    provenance = object_field(row, "provenance")
    nested = object_field(provenance, "program_synthesis_loop_v1")
    if nested:
        return nested
    visible = object_field(row, "visible_task")
    nested = object_field(visible, "program_synthesis_loop_v1")
    if nested:
        return nested
    nested = object_field(object_field(provenance, "visible_task"), "program_synthesis_loop_v1")
    return nested


def task_contract_summary(row: dict[str, Any]) -> dict[str, Any]:
    plan = object_field(row, "semantic_decoder_v2_plan")
    if not plan:
        plan = object_field(row, "decoder_contract_summary")
    if not plan:
        plan = object_field(object_field(row, "provenance"), "decoder_contract_summary")
    if not plan:
        plan = object_field(object_field(row, "visible_task"), "decoder_contract_summary")
    if not plan:
        plan = object_field(object_field(object_field(row, "provenance"), "visible_task"), "decoder_contract_summary")
    return {
        "return_shape": row.get("return_shape") or plan.get("return_shape"),
        "type_family": plan.get("type_family"),
        "visible_arg_count": plan.get("visible_arg_count"),
        "argument_roles": plan.get("argument_roles"),
        "required_constructs": list_field(plan, "required_constructs"),
        "plan_hints": list_field(plan, "plan_hints")[:24],
        "sts_hints": list_field(plan, "sts_hints")[:16],
    }


def nested_visible_task_field(row: dict[str, Any], key: str) -> Any:
    direct = get_path(row, ["visible_task", key], None)
    if direct is not None:
        return direct
    return get_path(row, ["provenance", "visible_task", key], "")


def candidate_code(row: dict[str, Any]) -> str:
    return str(row.get("code") or row.get("body") or row.get("candidate_body") or "")


def parse_code(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def function_entry_point_matches(tree: ast.Module, entry_point: str) -> bool:
    if not entry_point:
        return any(isinstance(node, ast.FunctionDef) for node in tree.body)
    return any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body)


def bogus_return_attribute_code(code: str) -> bool:
    blocked = {
        "isinstance",
        "list",
        "dict",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "set",
        "len",
        "sum",
        "min",
        "max",
        "sorted",
        "range",
        "append",
        "extend",
        "insert",
        "remove",
        "pop",
        "sort",
        "reverse",
        "items",
        "keys",
        "values",
        "get",
        "split",
        "strip",
        "lower",
        "upper",
        "replace",
        "join",
    }
    tree = parse_code(code)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.attr in blocked:
                return True
    return False


def bogus_return_local_callable_code(code: str) -> bool:
    allowed_callables = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "pow",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
    tree = parse_code(code)
    if tree is None:
        return False
    for fn in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        assigned: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
                assigned.add(node.target.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name):
                        assigned.add(item.optional_vars.id)
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in assigned
                and node.value.func.id not in allowed_callables
            ):
                return True
    return False


def sample_candidate_bodies(rows: list[dict[str, Any]], limit: int = 4) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for row in rows:
        body = str(candidate_code(row) or row.get("expr") or "")
        if not body:
            body = str(row.get("candidate_return_expr") or "")
        preview = " | ".join(part.strip() for part in body.splitlines() if part.strip())
        samples.append(
            {
                "mode": candidate_mode(row),
                "reason": ", ".join(row_rejection_reasons(row)[:4]) or "accepted_or_unlabeled",
                "body_preview": preview[:360],
            }
        )
        if len(samples) >= limit:
            break
    return samples


def materialize_no_admissible_residuals(
    *,
    public_rows: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    public_diagnostics: dict[str, Any],
    packet_out: Path,
    residual_jsonl_out: Path,
    policy_rows_out: Path,
) -> dict[str, Any]:
    private_diagnostics = public_no_admissible_diagnostics(private_rows)
    public_records = residual_records_from_diagnostics(public_diagnostics, source_scope="public_calibration_metadata_only")
    private_records = residual_records_from_diagnostics(private_diagnostics, source_scope="private_candidate_manifest")
    records = public_records + private_records
    policy_rows = [no_admissible_policy_row(row) for row in records]
    packet = {
        "policy": "project_theseus_no_admissible_candidate_residuals_v1",
        "created_utc": now(),
        "training_use_state": "decoder_control_policy_only_not_code_answer_training",
        "contamination_boundary": "public records carry metadata, hashes, contracts, rejection reasons, and generated candidate previews only; no public tests or solutions",
        "summary": {
            "residual_record_count": len(records),
            "public_residual_record_count": len(public_records),
            "private_residual_record_count": len(private_records),
            "policy_row_count": len(policy_rows),
            "public_unresolved_no_admissible_task_count": public_diagnostics.get("no_admissible_task_count"),
            "public_diagnostic_no_admissible_task_count": public_diagnostics.get("diagnostic_no_admissible_task_count"),
            "public_no_admissible_shadowed_by_eligible_task_count": public_diagnostics.get(
                "no_admissible_shadowed_by_eligible_task_count"
            ),
            "private_unresolved_no_admissible_task_count": private_diagnostics.get("no_admissible_task_count"),
            "private_diagnostic_no_admissible_task_count": private_diagnostics.get("diagnostic_no_admissible_task_count"),
            "private_no_admissible_shadowed_by_eligible_task_count": private_diagnostics.get(
                "no_admissible_shadowed_by_eligible_task_count"
            ),
            "top_rejection_reasons": merge_counts(
                public_diagnostics.get("top_rejection_reasons"),
                private_diagnostics.get("top_rejection_reasons"),
            ),
            "top_missing_capability_families": merge_counts(
                public_diagnostics.get("top_missing_capability_families"),
                private_diagnostics.get("top_missing_capability_families"),
            ),
        },
        "records": records[:160],
        "outputs": {
            "packet_out": rel_or_abs(packet_out),
            "residual_jsonl_out": rel_or_abs(residual_jsonl_out),
            "policy_rows_out": rel_or_abs(policy_rows_out),
        },
        "rules": {
            "no_templates": "rows describe why learned-token generation failed; they do not provide replacement bodies",
            "consumer": "SymLiquid and Code LM closure may use these rows as control pressure for candidate coverage recovery",
            "public_benchmarks": "public benchmark rows remain calibration metadata only",
        },
        "external_inference_calls": 0,
    }
    write_json(packet_out, packet)
    write_jsonl(residual_jsonl_out, records)
    write_jsonl(policy_rows_out, policy_rows)
    return {
        "packet_out": rel_or_abs(packet_out),
        "residual_jsonl_out": rel_or_abs(residual_jsonl_out),
        "policy_rows_out": rel_or_abs(policy_rows_out),
        "residual_record_count": len(records),
        "public_residual_record_count": len(public_records),
        "private_residual_record_count": len(private_records),
        "policy_row_count": len(policy_rows),
        "top_rejection_reasons": packet["summary"]["top_rejection_reasons"],
        "top_missing_capability_families": packet["summary"]["top_missing_capability_families"],
    }


def residual_records_from_diagnostics(diagnostics: dict[str, Any], *, source_scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in diagnostics.get("samples", []):
        if not isinstance(sample, dict):
            continue
        task_key = {
            "scope": source_scope,
            "task_id": sample.get("task_id"),
            "source_task_id": sample.get("source_task_id"),
            "entry_point": sample.get("entry_point"),
            "category": sample.get("category"),
        }
        rows.append(
            {
                "row_id": "no_admissible_residual_" + hashlib.sha256(
                    json.dumps(task_key, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:16],
                "policy": "project_theseus_no_admissible_candidate_residual_v1",
                "created_utc": now(),
                "source_scope": source_scope,
                "task_hash": hashlib.sha256(json.dumps(task_key, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16],
                "task_id": sample.get("task_id"),
                "source_task_id": sample.get("source_task_id"),
                "entry_point": sample.get("entry_point"),
                "category": sample.get("category"),
                "accepted_candidate_count": sample.get("accepted_candidate_count"),
                "promotion_eligible_candidate_count": sample.get("promotion_eligible_candidate_count"),
                "diagnostic_no_admissible_row_count": sample.get("diagnostic_no_admissible_row_count"),
                "no_admissible_status": sample.get("no_admissible_status"),
                "raw_candidate_count": sample.get("raw_candidate_count"),
                "raw_modes": sample.get("raw_modes"),
                "rejection_reasons": sample.get("rejection_reasons"),
                "missing_capability_family": sample.get("missing_capability_family"),
                "task_contract": sample.get("task_contract"),
                "generated_candidate_previews": sample.get("sample_bodies"),
                "training_use_state": "decoder_control_policy_only_not_code_answer_training",
                "public_benchmark_training_data_used": False,
                "raw_public_prompt_or_tests_copied": False,
                "external_inference_calls": 0,
            }
        )
    return rows


def no_admissible_policy_row(record: dict[str, Any]) -> dict[str, Any]:
    reasons = record.get("rejection_reasons") if isinstance(record.get("rejection_reasons"), dict) else {}
    contract = record.get("task_contract") if isinstance(record.get("task_contract"), dict) else {}
    family = str(record.get("missing_capability_family") or "unknown_capability_family")
    required = contract.get("required_constructs") if isinstance(contract.get("required_constructs"), list) else []
    status = str(record.get("no_admissible_status") or "unresolved_no_admissible")
    shadowed = status == "shadowed_by_promotion_eligible_candidate"
    source_type = "shadowed_no_admissible_candidate_diagnostic" if shadowed else "no_admissible_candidate_residual"
    prompt_prefix = (
        "A decoder branch emitted a no-admissible diagnostic, but another learned-token branch produced a "
        "promotion-eligible candidate. Treat this as branch-ranking and efficiency pressure, not a coverage failure. "
        if shadowed
        else "A learned-token decoder failed to produce a promotion-eligible admissible candidate. "
    )
    answer_prefix = (
        "Improve branch ranking and early exit: prefer the promotion-eligible learned-token branch already found, "
        "and keep the diagnostic no-admissible branch as low-weight residual pressure. "
        if shadowed
        else "Repair candidate coverage before scoring: preserve the exact interface, generate a complete AST-valid full body, "
    )
    answer = (
        answer_prefix
        + "Use all required arguments, satisfy return shape, and target missing capability family "
        + f"{family}. Required constructs: {', '.join(str(item) for item in required[:8]) or 'infer from contract'}."
    )
    return {
        "row_id": "decoder_control_" + hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16],
        "dataset_id": "dataset.no_admissible_decoder_control.v1",
        "source_type": source_type,
        "split": "control",
        "prompt": (
            prompt_prefix +
            f"Scope={record.get('source_scope')} entry_point={record.get('entry_point')} "
            f"family={family} reasons={dict(list(reasons.items())[:8])}. "
            "Choose decoder repair pressure, not a benchmark answer."
        ),
        "answer": answer,
        "missing_capability_family": family,
        "required_constructs": required,
        "no_admissible_status": status,
        "decoder_control_weight": 0.25 if shadowed else 1.0,
        "task_hash": record.get("task_hash"),
        "training_use_state": "decoder_control_policy_only_not_code_answer_training",
        "public_benchmark_training_data_used": False,
        "raw_public_prompt_or_tests_copied": False,
        "external_inference_calls": 0,
        "created_utc": now(),
    }


def merge_counts(first: Any, second: Any, limit: int = 16) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for payload in [first, second]:
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            try:
                counts[str(key)] += int(value)
            except Exception:
                counts[str(key)] += 1
    return dict(counts.most_common(limit))


def next_actions(ready: bool, gates: list[dict[str, Any]]) -> list[str]:
    failed = [row["name"] for row in gates if not row["passed"]]
    if ready:
        return [
            "Allow exactly one same-seed 4-card public receiver calibration.",
            "If receiver transfer stays flat, feed the exact residual cluster to teacher-as-architect for one experiment spec.",
        ]
    if "private_closure_completed_cleanly" in failed or "private_closure_present" in failed:
        return [
            "Preserve timed-out manifests as diagnostic-only evidence, then run a smaller bounded private closure with unique recovery artifacts.",
            "Public calibration remains locked until the decoder gate sees a completed private closure report.",
        ]
    if "train_once_wrapper_current_when_applicable" in failed:
        return [
            "The canonical train-once wrapper marks the selected closure stale against private training inputs.",
            "Run a fresh CUDA train-once private checkpoint before decoder or transfer gates consume those manifests.",
        ]
    if "decoder_fingerprint_current" in failed:
        return [
            "Private execution-shape coverage is useful, but the closure/candidate manifests are stale relative to the decoder source.",
            "Run a fresh private closure with the current decoder, then rerun this gate; do not public-calibrate from stale receiver manifests.",
        ]
    if "private_execution_shape_candidate_coverage_recovered" in failed:
        return [
            "Keep public calibration locked; recover learned-token private execution-shape coverage before another private closure or public run.",
            "Focus parser/AST-constrained learned generation on no-admissible residuals and execution-shaped bodies.",
        ]
    if "contract_guided_skeleton_observed" in failed:
        return ["Run a private Code LM closure after the contract-guided skeleton decoder patch; do not public-calibrate yet."]
    if "sts_conditioned_skeleton_observed" in failed:
        return ["Run private closure with STS enabled and confirm STS-conditioned skeleton rows appear."]
    if "contract_guided_non_regressive" in failed or "sts_conditioned_non_regressive" in failed:
        return ["Patch Decoder V2 skeleton selection before another public run; verifier-guided groups are not yet private-clean."]
    if (
        "public_candidate_generation_coverage" in failed
        or "no_admissible_candidate_rate_bounded" in failed
        or "public_candidate_provenance_and_quality" in failed
    ):
        return [
            "Patch local adapter generation and token candidate inventory so every receiver task family gets at least one admissible full-body token candidate with program-synthesis loop evidence.",
            "Focus the next private closure on no-admissible residuals, two-argument interfaces, recursive/nested structures, list/string transforms, execution-shaped bodies, and syntactically valid nonsense returns.",
        ]
    if "public_surface_scale_sufficient_for_calibration" in failed:
        return [
            "Keep public calibration locked until the visible calibration metadata surface reaches at least 32 tasks and 96 candidates.",
            "Build a broader metadata-only public surface, rerun train-once fanout, then rerun this decoder gate before any public benchmark scoring.",
        ]
    return ["Keep private closure/evidence generation running; public calibration remains blocked by the private ablation gate."]


def latest_execution_shape_candidate_coverage_report() -> dict[str, Any]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in REPORTS.glob("execution_shape_candidate_coverage*.json"):
        name = path.name
        if name.endswith("_rust.json") or name.endswith("_checkpoint.json"):
            continue
        report = read_json(path, {})
        if not isinstance(report, dict) or not report:
            continue
        report.setdefault("source_report_path", rel_or_abs(path))
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, report))
    if not candidates:
        return {}

    def score(item: tuple[float, dict[str, Any]]) -> tuple[int, float, float, float]:
        mtime, report = item
        summary = object_field(report, "summary")
        trigger_bonus = 1 if report.get("trigger_state") == "GREEN" else 0
        pass_rate = number(summary.get("learned_token_decoder_pass_rate"))
        no_admissible = summary.get("learned_token_decoder_no_admissible_candidate_rate")
        no_admissible = 1.0 if no_admissible is None else number(no_admissible)
        return (trigger_bonus, float(pass_rate or 0.0), -float(no_admissible or 0.0), mtime)

    return max(candidates, key=score)[1]


def decoder_relevant_source_fingerprint() -> str:
    paths = decoder_source_paths()
    if not paths:
        return ""
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths)
    relevant = "\n".join(
        line for line in text.splitlines() if any(marker in line for marker in DECODER_FINGERPRINT_MARKERS)
    )
    return hashlib.sha256(relevant.encode("utf-8")).hexdigest()[:16]


def decoder_relevant_source_mtime() -> float:
    return max((path.stat().st_mtime for path in decoder_source_paths()), default=0.0)


def decoder_source_paths() -> list[Path]:
    if DECODER_SOURCE.exists():
        return [DECODER_SOURCE]
    if DECODER_SOURCE_DIR.exists():
        return sorted(DECODER_SOURCE_DIR.glob("*.rs"))
    return []


def closure_decoder_current(
    *,
    report_fingerprint: str,
    current_fingerprint: str,
    closure_mtime: float,
    decoder_source_mtime: float,
) -> bool:
    if decoder_source_mtime and (not closure_mtime or closure_mtime < decoder_source_mtime):
        return False
    if report_fingerprint:
        return bool(current_fingerprint and report_fingerprint == current_fingerprint)
    if not closure_mtime:
        return False
    if not decoder_source_mtime:
        return True
    return closure_mtime >= decoder_source_mtime


def first_string(values: Iterable[Any]) -> str:
    for value in values:
        text = str(value or "")
        if text:
            return text
    return ""


def first_number(values: Iterable[Any]) -> float:
    for value in values:
        if value is None:
            continue
        parsed = number(value)
        if parsed != 0.0 or str(value).strip() in {"0", "0.0"}:
            return parsed
    return 0.0


def gate(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Decoder V2 Private Ablation Gate",
        "",
        f"- Status: **{report['trigger_state']}**",
        f"- Ready for public calibration: `{report['ready_for_public_calibration']}`",
        f"- Latest closure: `{report['summary']['latest_closure']}`",
        f"- Latest diagnostic partial closure: `{report['summary'].get('latest_diagnostic_partial_closure')}`",
        f"- Private closure delta: `{report['summary']['private_closure_pass_delta']}`",
        "",
        "## Gates",
        "",
    ]
    for row in report["gates"]:
        marker = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {marker}: `{row['name']}`")
    lines.extend(["", "## Candidate Groups", ""])
    for name, metrics in report["candidate_groups"].items():
        lines.append(f"- `{name}`: pass rate `{metrics['verifier_pass_rate']}`, candidates `{metrics['candidate_count']}`")
    lines.extend(["", "## Next Actions", ""])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def object_field(payload: Any, key: str) -> dict[str, Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else {}


def list_field(payload: Any, key: str) -> list[Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, list) else []


def get_path(payload: Any, path: list[str], default: Any = None) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
