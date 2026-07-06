#!/usr/bin/env python3
"""Private full-body contract-transfer recovery comparator.

This is a no-public-payload recovery harness for the current public-transfer
wall. It compares the promotion-grade full-body token generator with full
visible private contract context against the same generator with minimal visible
contract context, then records aggregate old-path evidence for orientation.

It does not run public calibration, read public traces, export tests/solutions
to generation, call a teacher, or promote a model.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from candidate_floor_v2_private_token_probe import (  # noqa: E402
    DEFAULT_EVAL,
    DEFAULT_TRAINING_SOURCES,
    normalize_candidate_phases,
    row_is_private_eval,
    rust_command,
    summarize,
    visible_task_manifest,
)
from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402
from theseus_archive_resolver import is_archive_pointer, resolve_archived_path  # noqa: E402


REPORTS = ROOT / "reports"
DEFAULT_PREVIOUS_CANDIDATE_FLOOR = REPORTS / "candidate_floor_v2_private_token_probe.json"
DEFAULT_OLD_PUBLIC_AGGREGATE = REPORTS / "real_code_benchmark_graduation_wide_public_seed23_5x32_interface_floor_v1.json"
DEFAULT_NEW_PUBLIC_AGGREGATE = REPORTS / "real_code_benchmark_graduation_industry_code_transfer_seed14_5x64_v1.json"
DEFAULT_DECODER_ABLATION = REPORTS / "decoder_v2_private_ablation_gate.json"
DEFAULT_SYMLIQUID_COMPARATOR = REPORTS / "candidate_floor_v2_survival_lane_seed101_comparator.json"
DEFAULT_SYMLIQUID_SAME_SLICE_COMPARATOR = (
    REPORTS / "full_body_contract_transfer_recovery_v1_symliquid_same_slice_comparator.json"
)
DEFAULT_STRUCTURAL_VCM_ABLATION = REPORTS / "broad_capability_structural_vcm_ablation_v1.json"
DEFAULT_BROAD_VCM_FEATURE_ABLATION = REPORTS / "broad_capability_vcm_feature_ablation_v1.json"
DEFAULT_OLD_CLOSURE_CHECKPOINT = (
    REPORTS / "student_code_lm_checkpoint_private_broad_floor_transfer_repair_closure_v16_private_sts.json"
)
DEFAULT_OLD_STS_TRAINING_DATA = ROOT / "data/sts_learning/sts_code_streams_seed14.jsonl"
DEFAULT_OUT = REPORTS / "full_body_contract_transfer_recovery_v1.json"
DEFAULT_MD = REPORTS / "full_body_contract_transfer_recovery_v1.md"
DEFAULT_REPAIR_TARGETS = REPORTS / "full_body_contract_transfer_recovery_v1_private_repair_targets.jsonl"


PUBLIC_FLAG_KEYS = {
    "public_benchmark",
    "public_benchmark_row",
    "public_benchmark_solutions_included",
    "public_prompts_included",
    "public_score_labels_included",
    "public_tests_included",
}
FORBIDDEN_EXPORT_KEYS = {
    "tests",
    "test",
    "hidden_tests",
    "canonical_solution",
    "solution",
    "solution_body",
    "reference_solution",
    "trace",
    "public_trace",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", action="append", default=[])
    parser.add_argument("--training-sources", default=DEFAULT_TRAINING_SOURCES)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--max-eval-rows", type=int, default=96)
    parser.add_argument("--max-candidates-per-task", type=int, default=8)
    parser.add_argument("--max-training-rows-per-source", type=int, default=1400)
    parser.add_argument("--max-project-files", type=int, default=160)
    parser.add_argument("--required-readiness-eval-rows", type=int, default=192)
    parser.add_argument("--min-selected-pass-rate", type=float, default=0.95)
    parser.add_argument("--max-no-candidate-rate", type=float, default=0.01)
    parser.add_argument("--previous-candidate-floor", default=rel(DEFAULT_PREVIOUS_CANDIDATE_FLOOR))
    parser.add_argument("--old-public-aggregate", default=rel(DEFAULT_OLD_PUBLIC_AGGREGATE))
    parser.add_argument("--new-public-aggregate", default=rel(DEFAULT_NEW_PUBLIC_AGGREGATE))
    parser.add_argument("--decoder-ablation", default=rel(DEFAULT_DECODER_ABLATION))
    parser.add_argument("--symliquid-comparator", default=rel(DEFAULT_SYMLIQUID_COMPARATOR))
    parser.add_argument("--symliquid-same-slice-comparator", default=rel(DEFAULT_SYMLIQUID_SAME_SLICE_COMPARATOR))
    parser.add_argument("--structural-vcm-ablation", default=rel(DEFAULT_STRUCTURAL_VCM_ABLATION))
    parser.add_argument("--broad-vcm-feature-ablation", default=rel(DEFAULT_BROAD_VCM_FEATURE_ABLATION))
    parser.add_argument("--old-closure-checkpoint", default=rel(DEFAULT_OLD_CLOSURE_CHECKPOINT))
    parser.add_argument("--old-sts-training-data", default=rel(DEFAULT_OLD_STS_TRAINING_DATA))
    parser.add_argument("--rust-build-profile", choices=["release", "debug"], default="release")
    parser.add_argument("--skip-old-closure-lane", action="store_true")
    parser.add_argument("--artifact-prefix", default="reports/full_body_contract_transfer_recovery_v1")
    parser.add_argument("--repair-targets-out", default=rel(DEFAULT_REPAIR_TARGETS))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    eval_paths = [resolve(path) for path in (args.eval_jsonl or DEFAULT_EVAL)]
    rows = select_private_eval_rows(
        eval_paths,
        max_rows=max(1, int(args.max_eval_rows)),
        seed=int(args.seed),
        previous_candidate_floor=resolve(args.previous_candidate_floor),
    )
    arms = []
    if not bool(args.skip_old_closure_lane):
        arms.append(run_old_closure_arm(args, rows))
    arms.extend(
        [
            run_arm(args, rows, arm_name="full_contract_context", visible_contract_mode="full"),
            run_arm(args, rows, arm_name="minimal_contract_context", visible_contract_mode="minimal"),
        ]
    )
    arm_by_name = {str(arm["arm_name"]): arm for arm in arms}
    full = arm_by_name["full_contract_context"]
    minimal = arm_by_name["minimal_contract_context"]
    full_summary = object_field(full, "summary")
    minimal_summary = object_field(minimal, "summary")

    historical = historical_context(args)
    no_cheat = no_cheat_summary(rows, arms)
    comparison = comparison_summary(full_summary, minimal_summary)
    side_by_side = side_by_side_recovery_matrix(
        arms,
        historical,
        max_candidates_per_task=int(args.max_candidates_per_task),
    )
    readiness = readiness_summary(args, rows, full_summary, comparison, no_cheat)
    repair_targets = build_repair_targets(rows, full, minimal)
    write_jsonl(resolve(args.repair_targets_out), repair_targets)
    gates = [
        gate("private_eval_rows_loaded", bool(rows), {"rows": len(rows), "sources": [rel(path) for path in eval_paths]}, "hard"),
        gate(
            "contract_blind_failures_included",
            weak_contract_blind_inclusion(rows, resolve(args.previous_candidate_floor))["included_all"],
            weak_contract_blind_inclusion(rows, resolve(args.previous_candidate_floor)),
            "warning",
        ),
        gate("all_configured_equal_budget_arms_ran", all(arm.get("returncode") == 0 for arm in arms), arm_status(arms), "hard"),
        gate(
            "old_closure_same_slice_lane_ran",
            bool(args.skip_old_closure_lane) or arm_by_name.get("old_sts_closure_path", {}).get("returncode") == 0,
            arm_status([arm_by_name["old_sts_closure_path"]]) if "old_sts_closure_path" in arm_by_name else {"skipped": bool(args.skip_old_closure_lane)},
            "hard",
        ),
        gate(
            "old_closure_same_slice_sts_conditioning_present",
            bool(args.skip_old_closure_lane)
            or int_number(get_path(arm_by_name.get("old_sts_closure_path", {}), ["sts_conditioning", "conditioned_private_task_count"], 0))
            >= len(rows),
            get_path(arm_by_name.get("old_sts_closure_path", {}), ["sts_conditioning"], {}),
            "warning",
        ),
        gate("no_public_payload_exported", no_cheat["forbidden_export_key_count"] == 0, no_cheat, "hard"),
        gate("no_public_training_rows", no_cheat["public_training_rows"] == 0, no_cheat["public_training_rows"], "hard"),
        gate("fallback_count_zero", no_cheat["fallback_return_count"] == 0, no_cheat["fallback_return_count"], "hard"),
        gate("template_like_count_zero", no_cheat["template_like_candidate_count"] == 0, no_cheat["template_like_candidate_count"], "hard"),
        gate("external_inference_zero", no_cheat["external_inference_calls"] == 0, no_cheat["external_inference_calls"], "hard"),
        gate(
            "full_contract_near_zero_no_candidate_private",
            comparison["full_no_candidate_rate"] <= float(args.max_no_candidate_rate),
            comparison,
            "maturity",
        ),
        gate(
            "full_contract_near_zero_no_admissible_private",
            comparison["full_no_admissible_task_rate"] <= float(args.max_no_candidate_rate),
            comparison,
            "maturity",
        ),
        gate(
            "full_contract_selected_pass_floor",
            comparison["full_selected_pass_rate"] >= float(args.min_selected_pass_rate),
            comparison,
            "maturity",
        ),
        gate(
            "full_contract_not_worse_than_minimal",
            comparison["selected_pass_delta_full_minus_minimal"] >= 0.0
            and comparison["no_candidate_delta_full_minus_minimal"] <= 0.0,
            comparison,
            "warning",
        ),
        gate(
            "per_family_regressions_recorded_and_zero",
            comparison["per_family_regression_count_vs_minimal"] == 0,
            {
                "regression_count": comparison["per_family_regression_count_vs_minimal"],
                "improvement_count": comparison["per_family_improvement_count_vs_minimal"],
                "regressions": comparison["per_family_regressions_vs_minimal"],
            },
            "warning",
        ),
        gate(
            "readiness_packet_decision_recorded",
            readiness["decision"] in {"ready_for_future_governed_public_calibration", "not_ready_private_repair_needed"},
            readiness,
            "hard",
        ),
        gate(
            "side_by_side_recovery_matrix_recorded",
            side_by_side["required_lanes_present"],
            side_by_side,
            "hard",
        ),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failed = [
        row for row in gates if row["severity"] in {"warning", "maturity"} and not row["passed"]
    ]
    trigger_state = "RED" if hard_failed else "YELLOW" if warning_failed or not readiness["ready"] else "GREEN"
    return {
        "policy": "project_theseus_full_body_contract_transfer_recovery_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "private_eval_rows": len(rows),
            "required_readiness_eval_rows": int(args.required_readiness_eval_rows),
            "max_candidates_per_task": int(args.max_candidates_per_task),
            "full_contract_selected_pass_rate": comparison["full_selected_pass_rate"],
            "minimal_contract_selected_pass_rate": comparison["minimal_selected_pass_rate"],
            "selected_pass_delta_full_minus_minimal": comparison["selected_pass_delta_full_minus_minimal"],
            "full_no_candidate_rate": comparison["full_no_candidate_rate"],
            "minimal_no_candidate_rate": comparison["minimal_no_candidate_rate"],
            "full_no_admissible_task_rate": comparison["full_no_admissible_task_rate"],
            "minimal_no_admissible_task_rate": comparison["minimal_no_admissible_task_rate"],
            "per_family_regression_count_vs_minimal": comparison["per_family_regression_count_vs_minimal"],
            "per_family_improvement_count_vs_minimal": comparison["per_family_improvement_count_vs_minimal"],
            "fallback_return_count": no_cheat["fallback_return_count"],
            "template_like_candidate_count": no_cheat["template_like_candidate_count"],
            "public_training_rows": no_cheat["public_training_rows"],
            "external_inference_calls": no_cheat["external_inference_calls"],
            "readiness_decision": readiness["decision"],
            "ready_for_future_governed_public_calibration": readiness["ready"],
            "private_repair_target_rows": len(repair_targets),
            "vcm_delta_status": get_path(historical, ["vcm_delta_context", "recommended_action"], ""),
        },
        "inputs": {
            "eval_jsonl": [rel(path) for path in eval_paths],
            "training_sources": rel(resolve(args.training_sources)),
            "previous_candidate_floor": rel(resolve(args.previous_candidate_floor)),
            "old_public_aggregate": rel(resolve(args.old_public_aggregate)),
            "new_public_aggregate": rel(resolve(args.new_public_aggregate)),
            "decoder_ablation": rel(resolve(args.decoder_ablation)),
            "symliquid_comparator": rel(resolve(args.symliquid_comparator)),
            "symliquid_same_slice_comparator": rel(resolve(args.symliquid_same_slice_comparator)),
            "structural_vcm_ablation": rel(resolve(args.structural_vcm_ablation)),
            "broad_vcm_feature_ablation": rel(resolve(args.broad_vcm_feature_ablation)),
            "old_closure_checkpoint": rel(resolve(args.old_closure_checkpoint)),
            "old_sts_training_data": rel(resolve(args.old_sts_training_data)),
        },
        "arms": arms,
        "comparison": comparison,
        "side_by_side_recovery_matrix": side_by_side,
        "historical_context": historical,
        "readiness": readiness,
        "private_repair_targets": {
            "path": rel(resolve(args.repair_targets_out)),
            "rows": len(repair_targets),
            "category_counts": dict(sorted(Counter(str(row.get("residual_concept") or "unknown") for row in repair_targets).items())),
            "score_semantics": "Private task IDs and residual labels only; no tests, solutions, public payloads, or candidate code.",
        },
        "no_cheat_audit": no_cheat,
        "gates": gates,
        "rules": {
            "public_calibration_run": False,
            "public_payload_training": False,
            "public_prompts_tests_solutions_traces_read": False,
            "candidate_code_in_report": False,
            "teacher_calls": False,
            "model_promotion_allowed": False,
            "score_semantics": (
                "Private equal-budget recovery comparator only. Historical public reports are used as aggregate "
                "score context, not as training data or candidate-generation input."
            ),
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def select_private_eval_rows(
    paths: list[Path],
    *,
    max_rows: int,
    seed: int,
    previous_candidate_floor: Path,
) -> list[dict[str, Any]]:
    all_rows = []
    for path in paths:
        for row in read_jsonl(path):
            if row_is_private_eval(row):
                item = dict(row)
                item["split"] = "eval"
                all_rows.append(item)
    all_rows = dedupe_rows(all_rows)
    weak_families = set(weak_contract_blind_families(previous_candidate_floor))
    priority = [
        row
        for row in all_rows
        if str(row.get("residual_concept") or row.get("category") or "") in weak_families
    ]
    rest = [
        row
        for row in all_rows
        if str(row.get("residual_concept") or row.get("category") or "") not in weak_families
    ]
    priority.sort(key=lambda row: stable_key({"seed": seed, "priority": row.get("task_id")}))
    rest.sort(key=lambda row: stable_key({"seed": seed, "task_id": row.get("task_id"), "entry_point": row.get("entry_point")}))
    selected = dedupe_rows(priority + rest)
    return selected[:max_rows]


def run_arm(
    args: argparse.Namespace,
    private_rows: list[dict[str, Any]],
    *,
    arm_name: str,
    visible_contract_mode: str,
) -> dict[str, Any]:
    prefix = str(args.artifact_prefix).rstrip("/")
    arm_args = argparse.Namespace(
        training_sources=args.training_sources,
        seed=int(args.seed),
        max_candidates_per_task=int(args.max_candidates_per_task),
        max_training_rows_per_source=int(args.max_training_rows_per_source),
        max_project_files=int(args.max_project_files),
        task_manifest_out=f"{prefix}_{arm_name}_tasks.jsonl",
        candidate_manifest_out=f"{prefix}_{arm_name}_candidates.jsonl",
        checkpoint_out=f"{prefix}_{arm_name}_checkpoint.json",
        rust_report_out=f"{prefix}_{arm_name}_rust.json",
    )
    task_manifest = visible_task_manifest(private_rows, visible_contract_mode=visible_contract_mode)
    write_jsonl(resolve(arm_args.task_manifest_out), task_manifest)
    for path in [
        resolve(arm_args.candidate_manifest_out),
        resolve(arm_args.checkpoint_out),
        resolve(arm_args.rust_report_out),
    ]:
        if path.exists():
            path.unlink()
    command = rust_command(arm_args)
    started = time.perf_counter()
    timeout_seconds = max(240, min(1800, 90 + len(private_rows) * max(1, int(args.max_candidates_per_task))))
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout_seconds)
        returncode = result.returncode
        stdout_tail = result.stdout[-1600:]
        stderr_tail = result.stderr[-1600:]
        error = ""
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout_tail = (exc.stdout or "")[-1600:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-1600:] if isinstance(exc.stderr, str) else ""
        error = f"rust token generator timed out after {timeout_seconds}s"
    except OSError as exc:
        returncode = 127
        stdout_tail = ""
        stderr_tail = ""
        error = str(exc)
    raw_candidates = read_jsonl(resolve(arm_args.candidate_manifest_out))
    candidates = normalize_candidate_phases(raw_candidates)
    for candidate in candidates:
        candidate["candidate_recovery_arm"] = arm_name
        candidate["visible_contract_mode"] = visible_contract_mode
    if candidates != raw_candidates:
        write_jsonl(resolve(arm_args.candidate_manifest_out), candidates)
    rust_report = read_json(resolve(arm_args.rust_report_out), {})
    private_eval = evaluate_private_candidates(private_rows, candidates) if private_rows and candidates else {}
    summary = summarize(private_rows, candidates, rust_report, private_eval)
    summary.update(extra_arm_metrics(private_rows, candidates, private_eval, task_manifest))
    return {
        "arm_name": arm_name,
        "visible_contract_mode": visible_contract_mode,
        "returncode": returncode,
        "summary": summary,
        "artifacts": {
            "task_manifest": rel(resolve(arm_args.task_manifest_out)),
            "candidate_manifest": rel(resolve(arm_args.candidate_manifest_out)),
            "checkpoint": rel(resolve(arm_args.checkpoint_out)),
            "rust_report": rel(resolve(arm_args.rust_report_out)),
        },
        "command": command,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "generation_error": error,
        "private_verifier": compact_private_eval(private_eval),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def run_old_closure_arm(args: argparse.Namespace, private_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the old closure fanout on the same visible private eval rows.

    This lane is diagnostic evidence for the old STS/contract-aware path. The
    manifest intentionally carries split/contract/signature metadata only; tests
    and solutions remain withheld from generation.
    """

    arm_name = "old_sts_closure_path"
    prefix = str(args.artifact_prefix).rstrip("/")
    task_manifest_out = f"{prefix}_{arm_name}_tasks.jsonl"
    candidate_manifest_out = f"{prefix}_{arm_name}_candidates.jsonl"
    public_candidate_out = f"{prefix}_{arm_name}_public_empty_candidates.jsonl"
    rust_report_out = f"{prefix}_{arm_name}_rust.json"
    sts_input_out = f"{prefix}_{arm_name}_sts_input.jsonl"
    sts_generation_out = f"{prefix}_{arm_name}_sts_streams.jsonl"
    sts_checkpoint_out = f"{prefix}_{arm_name}_sts_checkpoint.json"
    sts_report_out = f"{prefix}_{arm_name}_sts_report.json"
    task_manifest = old_closure_visible_task_manifest(private_rows)
    write_jsonl(resolve(task_manifest_out), task_manifest)
    for path in [
        resolve(candidate_manifest_out),
        resolve(public_candidate_out),
        resolve(rust_report_out),
        resolve(sts_input_out),
        resolve(sts_generation_out),
        resolve(sts_checkpoint_out),
        resolve(sts_report_out),
    ]:
        if path.exists():
            path.unlink()
    started = time.perf_counter()
    sts_conditioning = run_same_slice_sts_conditioning(
        args,
        task_manifest,
        sts_input_out=sts_input_out,
        sts_generation_out=sts_generation_out,
        sts_checkpoint_out=sts_checkpoint_out,
        sts_report_out=sts_report_out,
    )
    checkpoint_path = resolve_old_closure_checkpoint(resolve(args.old_closure_checkpoint))
    command = [
        *rust_cli_prefix(args),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        rel(resolve(task_manifest_out)),
        "--public-task-manifest",
        rel(write_empty_public_manifest(prefix, arm_name)),
        "--checkpoint-in",
        rel(checkpoint_path),
        "--seed",
        str(int(args.seed)),
        "--candidates-per-task",
        str(max(1, int(args.max_candidates_per_task))),
        "--private-candidate-out",
        rel(resolve(candidate_manifest_out)),
        "--public-candidate-out",
        rel(resolve(public_candidate_out)),
        "--report-out",
        rel(resolve(rust_report_out)),
        "--private-eval-limit",
        str(len(task_manifest)),
        "--public-task-limit",
        "0",
    ]
    if sts_conditioning.get("safe") and sts_conditioning.get("generation_path"):
        command.extend(["--sts-streams", rel(resolve(str(sts_conditioning["generation_path"])))])
    timeout_seconds = max(300, min(2400, 120 + len(task_manifest) * max(2, int(args.max_candidates_per_task)) * 2))
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout_seconds)
        returncode = result.returncode
        stdout_tail = result.stdout[-1600:]
        stderr_tail = result.stderr[-1600:]
        error = ""
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout_tail = (exc.stdout or "")[-1600:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-1600:] if isinstance(exc.stderr, str) else ""
        error = f"old closure fanout timed out after {timeout_seconds}s"
    except OSError as exc:
        returncode = 127
        stdout_tail = ""
        stderr_tail = ""
        error = str(exc)
    raw_candidates = read_jsonl(resolve(candidate_manifest_out))
    candidates = normalize_candidate_phases(raw_candidates)
    for candidate in candidates:
        candidate["candidate_recovery_arm"] = arm_name
        candidate["visible_contract_mode"] = "old_closure_full_contract"
        candidate["old_closure_same_slice_lane"] = True
        candidate["promotion_evidence_for_repaired_path"] = False
        candidate["full_body_token_candidate"] = True
        candidate["external_inference_calls"] = int_number(candidate.get("external_inference_calls"))
    if candidates != raw_candidates:
        write_jsonl(resolve(candidate_manifest_out), candidates)
    rust_report = read_json(resolve(rust_report_out), {})
    private_eval = evaluate_private_candidates(private_rows, candidates) if private_rows and candidates else {}
    summary = summarize(private_rows, candidates, rust_report, private_eval)
    summary.update(extra_arm_metrics(private_rows, candidates, private_eval, task_manifest))
    summary["same_slice_sts_conditioned_task_count"] = int_number(
        sts_conditioning.get("conditioned_private_task_count")
    )
    summary["diagnostic_old_closure_lane"] = True
    summary["promotion_evidence_for_repaired_path"] = False
    return {
        "arm_name": arm_name,
        "visible_contract_mode": "old_closure_full_contract",
        "returncode": returncode,
        "summary": summary,
        "artifacts": {
            "task_manifest": rel(resolve(task_manifest_out)),
            "candidate_manifest": rel(resolve(candidate_manifest_out)),
            "public_candidate_manifest": rel(resolve(public_candidate_out)),
            "rust_report": rel(resolve(rust_report_out)),
            "resolved_checkpoint": rel(checkpoint_path),
            "sts_input": rel(resolve(sts_input_out)),
            "sts_generation": rel(resolve(sts_generation_out)),
            "sts_report": rel(resolve(sts_report_out)),
        },
        "command": command,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "generation_error": error,
        "sts_conditioning": sts_conditioning,
        "private_verifier": compact_private_eval(private_eval),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def old_closure_visible_task_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = visible_task_manifest(rows, visible_contract_mode="full")
    out = []
    for row in manifest:
        item = dict(row)
        item["split"] = "eval"
        item["solution_expr"] = ""
        item["solution_body"] = ""
        item["old_closure_same_slice_visible_manifest"] = True
        item["candidate_expression_eligible"] = False
        item["private_solution_body_used"] = False
        out.append(item)
    return out


def run_same_slice_sts_conditioning(
    args: argparse.Namespace,
    task_manifest: list[dict[str, Any]],
    *,
    sts_input_out: str,
    sts_generation_out: str,
    sts_checkpoint_out: str,
    sts_report_out: str,
) -> dict[str, Any]:
    eval_rows = [same_slice_sts_row(row) for row in task_manifest]
    training_rows = [
        row
        for row in read_jsonl(resolve(args.old_sts_training_data))
        if str(row.get("split") or "") == "train" and not bool(row.get("public_benchmark_solutions_included"))
    ]
    conditioning_rows = eval_rows + training_rows
    write_jsonl(resolve(sts_input_out), conditioning_rows)
    command = [
        *rust_cli_prefix(args),
        "train-sts-parallel-decoder",
        "--input",
        rel(resolve(sts_input_out)),
        "--seed",
        str(int(args.seed)),
        "--hv-dim",
        "384",
        "--max-vocab",
        "640",
        "--epochs",
        "3",
        "--lr",
        "0.06",
        "--max-generate-steps",
        "32",
        "--max-train-rows",
        str(min(1400, max(64, len(training_rows)))),
        "--max-eval-rows",
        str(max(1, len(eval_rows))),
        "--max-generate-rows",
        str(max(1, len(eval_rows))),
        "--checkpoint-out",
        rel(resolve(sts_checkpoint_out)),
        "--generation-out",
        rel(resolve(sts_generation_out)),
        "--report-out",
        rel(resolve(sts_report_out)),
    ]
    timeout_seconds = max(300, min(1800, 120 + len(eval_rows) * 4))
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout_seconds)
        returncode = result.returncode
        stdout_tail = result.stdout[-1200:]
        stderr_tail = result.stderr[-1200:]
        error = ""
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout_tail = (exc.stdout or "")[-1200:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-1200:] if isinstance(exc.stderr, str) else ""
        error = f"same-slice STS conditioning timed out after {timeout_seconds}s"
    except OSError as exc:
        returncode = 127
        stdout_tail = ""
        stderr_tail = ""
        error = str(exc)
    generated = read_jsonl(resolve(sts_generation_out))
    wanted = {str(row.get("task_id") or "") for row in task_manifest}
    conditioned = {
        str(row.get("task_id") or "")
        for row in generated
        if str(row.get("task_id") or "") in wanted
        and isinstance(row.get("streams"), dict)
        and not bool(row.get("public_benchmark_solutions_included"))
    }
    report = read_json(resolve(sts_report_out), {})
    safe = returncode == 0 and wanted.issubset(conditioned) and report.get("trigger_state") in {"GREEN", "YELLOW"}
    return {
        "safe": safe,
        "returncode": returncode,
        "generation_path": rel(resolve(sts_generation_out)) if safe else "",
        "conditioning_input": rel(resolve(sts_input_out)),
        "report": rel(resolve(sts_report_out)),
        "conditioned_private_task_count": len(conditioned),
        "required_private_task_count": len(wanted),
        "missing_task_count": len(wanted - conditioned),
        "missing_task_ids": sorted(wanted - conditioned)[:20],
        "training_rows": len(training_rows),
        "eval_rows": len(eval_rows),
        "trigger_state": report.get("trigger_state"),
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "external_inference_calls": 0,
        "command": command,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "error": error,
    }


def same_slice_sts_row(task: dict[str, Any]) -> dict[str, Any]:
    contract = task.get("decoder_contract") if isinstance(task.get("decoder_contract"), dict) else {}
    roles = contract.get("argument_roles") if isinstance(contract.get("argument_roles"), dict) else {}
    return_contract = contract.get("return_contract") if isinstance(contract.get("return_contract"), dict) else {}
    plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
    label = str(task.get("concept_residual_label") or task.get("residual_concept") or task.get("category") or "")
    context = {
        "entry_point": task.get("entry_point"),
        "prompt": task.get("prompt"),
        "category": task.get("category"),
        "residual_concept": task.get("residual_concept"),
        "argument_roles": roles,
        "return_contract": return_contract,
        "type_family": contract.get("type_family"),
        "return_shape": contract.get("return_shape"),
        "required_constructs": contract.get("required_constructs"),
        "skeleton_bias": plan.get("skeleton_bias"),
    }
    return {
        "policy": "project_theseus_same_slice_private_sts_conditioning_v1",
        "task_id": str(task.get("task_id") or ""),
        "source_task_id": str(task.get("source_task_id") or ""),
        "split": "eval",
        "benchmark_evidence_level": "private_same_slice_sts_context_no_solution_targets",
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "private_solution_body_used": False,
        "visible_task_only": True,
        "input_streams": {
            "context_stream": json.dumps(context, sort_keys=True),
            "solver_stream": "infer full Python body from visible prompt, signature, decoder contract, roles, and return contract",
            "critic_stream": f"check residual={label}; reject shallow wrappers, fallbacks, and wrong return shape",
            "tool_stream": "use parser and decoder-contract signals only; do not use public tests, solutions, traces, or answer templates",
            "residual_stream": label,
        },
        "target_streams": {
            "solver_stream": f"contract roles -> algorithm plan -> executable full body for {label}",
            "critic_stream": f"verify argument use, return shape, edge cases, and {label}",
            "tool_stream": "ast.parse; import/load; private verifier only after generation",
            "patch_stream": "",
            "residual_stream": label,
            "visible_report_stream": "same-slice private STS context generated without solution targets",
        },
    }


def write_empty_public_manifest(prefix: str, arm_name: str) -> Path:
    path = resolve(f"{prefix}_{arm_name}_public_empty_tasks.jsonl")
    write_text(path, "")
    return path


def resolve_old_closure_checkpoint(path: Path) -> Path:
    value = read_json(path, {})
    if isinstance(value.get("model_artifacts_v1"), dict):
        return path
    archive_path = value.get("archive_path")
    original_bytes = int_number(value.get("original_bytes"))
    if archive_path:
        archive = resolve(str(archive_path))
        target = ROOT / "runtime" / "candidate_replay_contract_v1" / f"{path.stem}_resolved.json"
        if is_archive_pointer(target):
            archived_target = resolve_archived_path(target)
            if archived_target.exists() and archived_target.suffix != ".gz":
                return archived_target
            if archived_target.exists() and archived_target.suffix == ".gz":
                target.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(archived_target, "rb") as src, target.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                return target
        if target.exists() and (original_bytes <= 0 or target.stat().st_size >= max(1, int(original_bytes * 0.9))):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        if archive.suffix == ".gz":
            with gzip.open(archive, "rb") as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        else:
            target.write_bytes(archive.read_bytes())
        return target
    return path


def rust_cli_prefix(args: argparse.Namespace) -> list[str]:
    profile = str(getattr(args, "rust_build_profile", "release") or "release")
    candidates = [
        ROOT / "target" / profile / "symliquid-cli",
        ROOT / "target" / profile / "symliquid-cli.exe",
    ]
    source_files = [
        ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "sts_parallel_decoder.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure" / "part_00.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure" / "candidate_fanout.rs",
    ]
    for exe in candidates:
        if exe.exists() and all(exe.stat().st_mtime >= path.stat().st_mtime for path in source_files if path.exists()):
            return [str(exe)]
    if profile == "release":
        return ["cargo", "run", "--release", "-p", "symliquid-cli", "--"]
    return ["cargo", "run", "-p", "symliquid-cli", "--"]


def extra_arm_metrics(
    private_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    private_eval: dict[str, Any],
    task_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    private_task_ids = {str(row.get("task_id") or "") for row in private_rows if str(row.get("task_id") or "")}
    task_count = len(private_task_ids)
    candidate_task_ids = {str(row.get("task_id") or "") for row in candidates if str(row.get("task_id") or "")}
    no_candidate_task_count = max(0, task_count - len(candidate_task_ids))
    no_candidate_rate = ratio(no_candidate_task_count, task_count)
    admissible_task_ids = {
        str(row.get("task_id") or "")
        for row in candidates
        if str(row.get("task_id") or "") in private_task_ids
        and row.get("benchmark_promotion_eligible") is True
        and row.get("full_body_token_candidate") is True
        and row.get("template_like_candidate") is not True
        and row.get("expression_memory_fallback") is not True
    }
    no_admissible_task_count = max(0, task_count - len(admissible_task_ids))
    verification = object_field(private_eval, "private_verification")
    return {
        "selected_pass_rate": number(private_eval.get("trained_pass_rate")),
        "selected_passed": int_number(private_eval.get("trained_passed")),
        "pass_if_any_rate": number(private_eval.get("trained_pass_rate")),
        "no_candidate_rate": no_candidate_rate,
        "no_candidate_task_count": no_candidate_task_count,
        "admissible_task_count": len(admissible_task_ids),
        "no_admissible_task_count": no_admissible_task_count,
        "no_admissible_task_rate": ratio(no_admissible_task_count, task_count),
        "verifier_load_rate": number(verification.get("runtime_load_rate")),
        "verifier_compile_rate": number(verification.get("compile_pass_rate")),
        "verifier_lint_rate": number(verification.get("lint_pass_rate")),
        "verifier_attempt_count": int_number(verification.get("candidate_attempt_count")),
        "stage_counts": verification.get("stage_counts") if isinstance(verification.get("stage_counts"), dict) else {},
        "forbidden_export_key_count": forbidden_export_key_count(task_manifest),
        "public_flagged_task_rows": public_flagged_rows(private_rows),
        "public_flagged_candidate_rows": public_flagged_rows(candidates),
    }


def comparison_summary(full: dict[str, Any], minimal: dict[str, Any]) -> dict[str, Any]:
    full_pass = number(full.get("selected_pass_rate") or full.get("private_trained_pass_rate"))
    minimal_pass = number(minimal.get("selected_pass_rate") or minimal.get("private_trained_pass_rate"))
    full_no_candidate = number(full.get("no_candidate_rate"))
    minimal_no_candidate = number(minimal.get("no_candidate_rate"))
    full_no_admissible = number(full.get("no_admissible_task_rate"))
    minimal_no_admissible = number(minimal.get("no_admissible_task_rate"))
    full_verifier = number(full.get("verifier_load_rate"))
    minimal_verifier = number(minimal.get("verifier_load_rate"))
    family_regressions = per_family_regressions(full, minimal)
    return {
        "full_selected_pass_rate": full_pass,
        "minimal_selected_pass_rate": minimal_pass,
        "selected_pass_delta_full_minus_minimal": round(full_pass - minimal_pass, 6),
        "full_no_candidate_rate": full_no_candidate,
        "minimal_no_candidate_rate": minimal_no_candidate,
        "no_candidate_delta_full_minus_minimal": round(full_no_candidate - minimal_no_candidate, 6),
        "full_no_admissible_task_rate": full_no_admissible,
        "minimal_no_admissible_task_rate": minimal_no_admissible,
        "no_admissible_delta_full_minus_minimal": round(full_no_admissible - minimal_no_admissible, 6),
        "per_family_regression_count_vs_minimal": family_regressions["regression_count"],
        "per_family_improvement_count_vs_minimal": family_regressions["improvement_count"],
        "per_family_equal_count_vs_minimal": family_regressions["equal_count"],
        "per_family_regressions_vs_minimal": family_regressions["regressions"],
        "per_family_improvements_vs_minimal": family_regressions["improvements"],
        "full_verifier_load_rate": full_verifier,
        "minimal_verifier_load_rate": minimal_verifier,
        "verifier_load_delta_full_minus_minimal": round(full_verifier - minimal_verifier, 6),
        "full_candidate_count": int_number(full.get("candidate_count")),
        "minimal_candidate_count": int_number(minimal.get("candidate_count")),
        "full_full_body_candidates": int_number(full.get("full_body_token_candidate_count")),
        "minimal_full_body_candidates": int_number(minimal.get("full_body_token_candidate_count")),
        "full_stage_counts": full.get("stage_counts") if isinstance(full.get("stage_counts"), dict) else {},
        "minimal_stage_counts": minimal.get("stage_counts") if isinstance(minimal.get("stage_counts"), dict) else {},
    }


def per_family_regressions(full: dict[str, Any], minimal: dict[str, Any]) -> dict[str, Any]:
    full_rates = full.get("private_family_pass_rates")
    minimal_rates = minimal.get("private_family_pass_rates")
    full_rates = full_rates if isinstance(full_rates, dict) else {}
    minimal_rates = minimal_rates if isinstance(minimal_rates, dict) else {}
    regressions = []
    improvements = []
    equal_count = 0
    for family in sorted(set(full_rates) | set(minimal_rates)):
        full_rate = number(full_rates.get(family))
        minimal_rate = number(minimal_rates.get(family))
        delta = round(full_rate - minimal_rate, 6)
        row = {
            "family": family,
            "full_contract_rate": full_rate,
            "minimal_contract_rate": minimal_rate,
            "delta_full_minus_minimal": delta,
        }
        if delta < 0:
            regressions.append(row)
        elif delta > 0:
            improvements.append(row)
        else:
            equal_count += 1
    return {
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "equal_count": equal_count,
        "regressions": regressions,
        "improvements": improvements,
    }


def historical_context(args: argparse.Namespace) -> dict[str, Any]:
    old_public = read_json(resolve(args.old_public_aggregate), {})
    new_public = read_json(resolve(args.new_public_aggregate), {})
    decoder = read_json(resolve(args.decoder_ablation), {})
    symliquid = read_json(resolve(args.symliquid_comparator), {})
    symliquid_same_slice = read_json(resolve(args.symliquid_same_slice_comparator), {})
    structural_vcm = read_json(resolve(args.structural_vcm_ablation), {})
    broad_vcm = read_json(resolve(args.broad_vcm_feature_ablation), {})
    old_summary = object_field(old_public, "summary")
    new_summary = object_field(new_public, "summary")
    decoder_summary = object_field(decoder, "summary")
    sym_summary = object_field(symliquid, "summary")
    same_slice_summary = object_field(symliquid_same_slice, "summary")
    same_slice_comparisons = object_field(symliquid_same_slice, "comparisons")
    same_slice_by_arm = object_field(same_slice_comparisons, "by_arm")
    same_slice_sym = object_field(same_slice_by_arm, "symliquid_style")
    same_slice_trans = object_field(same_slice_by_arm, "transformer_control")
    structural_vcm_summary = object_field(structural_vcm, "summary")
    broad_vcm_summary = object_field(broad_vcm, "summary")
    return {
        "old_public_aggregate_reference": {
            "path": rel(resolve(args.old_public_aggregate)),
            "task_count": int_number(old_summary.get("public_task_count")),
            "pass_rate": number(old_summary.get("real_public_task_pass_rate")),
            "no_candidate_events": get_path(old_summary, ["verification_cascade_summary", "verification_stage_counts", "no_candidate"], 0),
            "pass_origins": old_summary.get("multi_stream_pass_origin_counts") if isinstance(old_summary.get("multi_stream_pass_origin_counts"), dict) else {},
            "scope": "aggregate_score_context_only_no_public_payload_ingestion",
        },
        "new_public_aggregate_reference": {
            "path": rel(resolve(args.new_public_aggregate)),
            "task_count": int_number(new_summary.get("public_task_count")),
            "pass_rate": number(new_summary.get("real_public_task_pass_rate")),
            "no_candidate_events": get_path(new_summary, ["verification_cascade_summary", "verification_stage_counts", "no_candidate"], 0),
            "pass_origins": new_summary.get("multi_stream_pass_origin_counts") if isinstance(new_summary.get("multi_stream_pass_origin_counts"), dict) else {},
            "scope": "aggregate_score_context_only_consumed_public_surface_not_rerun",
        },
        "old_private_decoder_contract_reference": {
            "path": rel(resolve(args.decoder_ablation)),
            "trigger_state": decoder.get("trigger_state"),
            "contract_guided_candidate_count": int_number(decoder_summary.get("contract_guided_candidate_count")),
            "contract_guided_verifier_pass_rate": number(decoder_summary.get("contract_guided_verifier_pass_rate")),
            "sts_conditioned_candidate_count": int_number(decoder_summary.get("sts_conditioned_candidate_count")),
            "sts_conditioned_verifier_pass_rate": number(decoder_summary.get("sts_conditioned_verifier_pass_rate")),
            "ready_for_public_calibration": bool(decoder_summary.get("ready_for_public_calibration")),
            "scope": "old_path_private_contract_awareness_reference_not_promotion_evidence_for_new_path",
        },
        "symliquid_matched_compute_reference": {
            "path": rel(resolve(args.symliquid_comparator)),
            "trigger_state": symliquid.get("trigger_state"),
            "comparison_level": sym_summary.get("comparison_level"),
            "best_sts_on_arm": sym_summary.get("best_sts_on_arm_by_verifier_pass_rate"),
            "symliquid_minus_transformer": sym_summary.get("symliquid_minus_transformer_sts_on_verifier_pass_rate"),
            "parameter_match_delta": sym_summary.get("parameter_match_delta"),
            "scope": "matched_compute_body_selector_reference_not_full_body_promotion_claim",
        },
        "symliquid_same_slice_matched_compute": {
            "path": rel(resolve(args.symliquid_same_slice_comparator)),
            "trigger_state": symliquid_same_slice.get("trigger_state"),
            "comparison_level": same_slice_summary.get("comparison_level"),
            "eval_rows": same_slice_summary.get("eval_rows"),
            "candidate_rows": same_slice_summary.get("candidate_rows"),
            "fanout_top_k": get_path(symliquid_same_slice, ["matched_budget", "fanout_top_k"], 0),
            "parameter_match_delta": same_slice_summary.get("parameter_match_delta"),
            "symliquid_sts_on_pass_rate": same_slice_sym.get("sts_on_verifier_pass_rate"),
            "transformer_sts_on_pass_rate": same_slice_trans.get("sts_on_verifier_pass_rate"),
            "symliquid_sts_delta": same_slice_sym.get("sts_delta"),
            "transformer_sts_delta": same_slice_trans.get("sts_delta"),
            "winner": same_slice_comparisons.get("winner_by_sts_on_verifier_pass_rate"),
            "symliquid_minus_transformer": same_slice_comparisons.get("symliquid_minus_transformer_sts_on_verifier_pass_rate"),
            "public_training_rows": same_slice_summary.get("public_training_rows"),
            "external_inference_calls": same_slice_summary.get("external_inference_calls"),
            "teacher_used": same_slice_summary.get("teacher_used"),
            "scope": "fresh_same_slice_matched_compute_body_template_discovery_lane_not_full_body_promotion_claim",
        },
        "vcm_delta_context": {
            "structural_vcm_ablation": {
                "path": rel(resolve(args.structural_vcm_ablation)),
                "trigger_state": structural_vcm.get("trigger_state"),
                "transformer_structural_only_delta": structural_vcm_summary.get("transformer_structural_only_delta"),
                "transformer_augmented_delta": structural_vcm_summary.get("transformer_augmented_delta"),
                "symliquid_structural_only_delta": structural_vcm_summary.get("symliquid_structural_only_delta"),
                "symliquid_augmented_delta": structural_vcm_summary.get("symliquid_augmented_delta"),
                "recommended_action": structural_vcm_summary.get("recommended_action"),
            },
            "broad_body_template_vcm_ablation": {
                "path": rel(resolve(args.broad_vcm_feature_ablation)),
                "trigger_state": broad_vcm.get("trigger_state"),
                "transformer_delta": get_path(broad_vcm_summary, ["deltas", "transformer_control"], 0),
                "symliquid_delta": get_path(broad_vcm_summary, ["deltas", "symliquid_style"], 0),
                "recommended_action": broad_vcm_summary.get("recommended_action"),
                "fallback_return_count": broad_vcm_summary.get("fallback_return_count"),
                "public_training_rows_written": broad_vcm_summary.get("public_training_rows_written"),
                "external_inference_calls": broad_vcm_summary.get("external_inference_calls"),
            },
            "recommended_action": (
                "enable_vcm_for_structural_path_keep_disabled_for_old_body_template_selector_until_positive_same_surface_lift"
            ),
            "score_semantics": (
                "VCM deltas are same-surface private ablation context. They do not use public benchmark payloads "
                "and do not grant promotion by themselves."
            ),
        },
    }


def side_by_side_recovery_matrix(
    arms: list[dict[str, Any]],
    historical: dict[str, Any],
    *,
    max_candidates_per_task: int,
) -> dict[str, Any]:
    by_arm = {str(arm.get("arm_name") or ""): arm for arm in arms}
    old = object_field(by_arm.get("old_sts_closure_path", {}), "summary")
    old_sts = object_field(by_arm.get("old_sts_closure_path", {}), "sts_conditioning")
    full = object_field(by_arm.get("full_contract_context", {}), "summary")
    minimal = object_field(by_arm.get("minimal_contract_context", {}), "summary")
    old_public = object_field(historical, "old_public_aggregate_reference")
    new_public = object_field(historical, "new_public_aggregate_reference")
    old_private = object_field(historical, "old_private_decoder_contract_reference")
    symliquid = object_field(historical, "symliquid_matched_compute_reference")
    symliquid_same_slice = object_field(historical, "symliquid_same_slice_matched_compute")
    lanes = [
        {
            "lane": "old_sts_contract_interface_receiver_bridge_private_equal_budget",
            "evidence_scope": "actual_private_same_slice_old_closure_fanout_with_same_slice_sts_if_safe",
            "task_count": old.get("private_eval_task_count"),
            "candidate_count": old.get("candidate_count"),
            "max_candidates_per_task": max_candidates_per_task,
            "selected_pass_rate": old.get("selected_pass_rate"),
            "selected_passed": old.get("selected_passed"),
            "no_candidate_rate": old.get("no_candidate_rate"),
            "no_admissible_task_rate": old.get("no_admissible_task_rate"),
            "no_admissible_task_count": old.get("no_admissible_task_count"),
            "same_slice_sts_conditioned_task_count": old.get("same_slice_sts_conditioned_task_count"),
            "same_slice_sts_required_task_count": old_sts.get("required_private_task_count"),
            "same_slice_sts_safe": old_sts.get("safe"),
            "fallback_return_count": old.get("expression_memory_fallback_count"),
            "template_like_candidate_count": old.get("template_like_candidate_count"),
            "external_inference_calls": old.get("external_inference_calls"),
            "public_flagged_candidate_rows": old.get("public_flagged_candidate_rows"),
            "public_flagged_task_rows": old.get("public_flagged_task_rows"),
            "equal_private_candidate_budget": bool(old),
            "public_payload_training": False,
            "promotion_evidence_for_repaired_path": False,
        },
        {
            "lane": "old_sts_contract_interface_receiver_bridge_public_reference",
            "evidence_scope": str(old_public.get("scope") or "aggregate_reference_only"),
            "path": old_public.get("path"),
            "task_count": old_public.get("task_count"),
            "pass_rate": old_public.get("pass_rate"),
            "no_candidate_events": old_public.get("no_candidate_events"),
            "pass_origins": old_public.get("pass_origins", {}),
            "equal_private_candidate_budget": False,
            "public_payload_training": False,
            "promotion_evidence_for_repaired_path": False,
        },
        {
            "lane": "old_private_sts_contract_awareness_reference",
            "evidence_scope": str(old_private.get("scope") or "private_reference_only"),
            "path": old_private.get("path"),
            "trigger_state": old_private.get("trigger_state"),
            "contract_guided_candidate_count": old_private.get("contract_guided_candidate_count"),
            "contract_guided_verifier_pass_rate": old_private.get("contract_guided_verifier_pass_rate"),
            "sts_conditioned_candidate_count": old_private.get("sts_conditioned_candidate_count"),
            "sts_conditioned_verifier_pass_rate": old_private.get("sts_conditioned_verifier_pass_rate"),
            "equal_private_candidate_budget": False,
            "public_payload_training": False,
            "promotion_evidence_for_repaired_path": False,
        },
        {
            "lane": "current_full_body_public_regression_reference",
            "evidence_scope": str(new_public.get("scope") or "consumed_public_aggregate_reference_only"),
            "path": new_public.get("path"),
            "task_count": new_public.get("task_count"),
            "pass_rate": new_public.get("pass_rate"),
            "no_candidate_events": new_public.get("no_candidate_events"),
            "pass_origins": new_public.get("pass_origins", {}),
            "equal_private_candidate_budget": False,
            "public_payload_training": False,
            "promotion_evidence_for_repaired_path": False,
        },
        {
            "lane": "current_full_body_token_beam_private_equal_budget_control",
            "evidence_scope": (
                "actual_private_equal_budget_current_full_body_token_generator_with_minimal_visible_contract_context"
            ),
            "task_count": minimal.get("private_eval_task_count"),
            "candidate_count": minimal.get("candidate_count"),
            "max_candidates_per_task": max_candidates_per_task,
            "selected_pass_rate": minimal.get("selected_pass_rate"),
            "selected_passed": minimal.get("selected_passed"),
            "no_candidate_rate": minimal.get("no_candidate_rate"),
            "no_admissible_task_rate": minimal.get("no_admissible_task_rate"),
            "no_admissible_task_count": minimal.get("no_admissible_task_count"),
            "fallback_return_count": minimal.get("expression_memory_fallback_count"),
            "template_like_candidate_count": minimal.get("template_like_candidate_count"),
            "external_inference_calls": minimal.get("external_inference_calls"),
            "public_flagged_candidate_rows": minimal.get("public_flagged_candidate_rows"),
            "public_flagged_task_rows": minimal.get("public_flagged_task_rows"),
            "equal_private_candidate_budget": True,
            "public_payload_training": False,
            "promotion_evidence_for_repaired_path": False,
        },
        {
            "lane": "repaired_transformer_hybrid_full_body_private_equal_budget",
            "evidence_scope": "actual_private_equal_budget_recovery_arm",
            "task_count": full.get("private_eval_task_count"),
            "candidate_count": full.get("candidate_count"),
            "max_candidates_per_task": max_candidates_per_task,
            "selected_pass_rate": full.get("selected_pass_rate"),
            "selected_passed": full.get("selected_passed"),
            "no_candidate_rate": full.get("no_candidate_rate"),
            "no_admissible_task_rate": full.get("no_admissible_task_rate"),
            "no_admissible_task_count": full.get("no_admissible_task_count"),
            "fallback_return_count": full.get("expression_memory_fallback_count"),
            "template_like_candidate_count": full.get("template_like_candidate_count"),
            "external_inference_calls": full.get("external_inference_calls"),
            "public_flagged_candidate_rows": full.get("public_flagged_candidate_rows"),
            "public_flagged_task_rows": full.get("public_flagged_task_rows"),
            "equal_private_candidate_budget": True,
            "public_payload_training": False,
            "promotion_evidence_for_repaired_path": True,
        },
        {
            "lane": "minimal_contract_context_private_equal_budget_ablation",
            "evidence_scope": "actual_private_equal_budget_ablation_arm",
            "task_count": minimal.get("private_eval_task_count"),
            "candidate_count": minimal.get("candidate_count"),
            "max_candidates_per_task": max_candidates_per_task,
            "selected_pass_rate": minimal.get("selected_pass_rate"),
            "selected_passed": minimal.get("selected_passed"),
            "no_candidate_rate": minimal.get("no_candidate_rate"),
            "no_admissible_task_rate": minimal.get("no_admissible_task_rate"),
            "no_admissible_task_count": minimal.get("no_admissible_task_count"),
            "equal_private_candidate_budget": True,
            "public_payload_training": False,
            "promotion_evidence_for_repaired_path": False,
        },
        {
            "lane": "symliquid_matched_compute_same_slice_private_equal_budget",
            "evidence_scope": str(symliquid_same_slice.get("scope") or "missing_same_slice_symliquid_comparator"),
            "path": symliquid_same_slice.get("path"),
            "trigger_state": symliquid_same_slice.get("trigger_state"),
            "comparison_level": symliquid_same_slice.get("comparison_level"),
            "task_count": symliquid_same_slice.get("eval_rows"),
            "candidate_rows": symliquid_same_slice.get("candidate_rows"),
            "max_candidates_per_task": symliquid_same_slice.get("fanout_top_k"),
            "parameter_match_delta": symliquid_same_slice.get("parameter_match_delta"),
            "symliquid_sts_on_pass_rate": symliquid_same_slice.get("symliquid_sts_on_pass_rate"),
            "transformer_sts_on_pass_rate": symliquid_same_slice.get("transformer_sts_on_pass_rate"),
            "symliquid_sts_delta": symliquid_same_slice.get("symliquid_sts_delta"),
            "transformer_sts_delta": symliquid_same_slice.get("transformer_sts_delta"),
            "symliquid_minus_transformer": symliquid_same_slice.get("symliquid_minus_transformer"),
            "winner": symliquid_same_slice.get("winner"),
            "equal_private_candidate_budget": symliquid_same_slice.get("trigger_state") in {"GREEN", "YELLOW"},
            "public_payload_training": False,
            "public_training_rows": symliquid_same_slice.get("public_training_rows"),
            "external_inference_calls": symliquid_same_slice.get("external_inference_calls"),
            "teacher_used": symliquid_same_slice.get("teacher_used"),
            "promotion_evidence_for_repaired_path": False,
        },
        {
            "lane": "symliquid_matched_compute_discovery_reference",
            "evidence_scope": str(symliquid.get("scope") or "matched_compute_reference_only"),
            "path": symliquid.get("path"),
            "trigger_state": symliquid.get("trigger_state"),
            "comparison_level": symliquid.get("comparison_level"),
            "parameter_match_delta": symliquid.get("parameter_match_delta"),
            "symliquid_minus_transformer": symliquid.get("symliquid_minus_transformer"),
            "best_sts_on_arm": symliquid.get("best_sts_on_arm"),
            "equal_private_candidate_budget": False,
            "public_payload_training": False,
            "promotion_evidence_for_repaired_path": False,
        },
    ]
    required = {
        "old_sts_contract_interface_receiver_bridge_private_equal_budget",
        "old_sts_contract_interface_receiver_bridge_public_reference",
        "old_private_sts_contract_awareness_reference",
        "current_full_body_public_regression_reference",
        "current_full_body_token_beam_private_equal_budget_control",
        "repaired_transformer_hybrid_full_body_private_equal_budget",
        "minimal_contract_context_private_equal_budget_ablation",
        "symliquid_matched_compute_same_slice_private_equal_budget",
        "symliquid_matched_compute_discovery_reference",
    }
    present = {str(row.get("lane") or "") for row in lanes}
    return {
        "policy": "project_theseus_full_body_candidate_recovery_side_by_side_matrix_v1",
        "required_lanes_present": required.issubset(present),
        "required_lanes": sorted(required),
        "present_lanes": sorted(present),
        "lanes": lanes,
        "score_semantics": (
            "The old closure, repaired/minimal full-body, and SymLiquid/transformer body-template discovery arms are "
            "fresh private same-slice lanes when their summaries are present. Consumed public scores remain aggregate "
            "references only; the SymLiquid lane is protected discovery evidence, not full-body promotion evidence."
        ),
    }


def readiness_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    full_summary: dict[str, Any],
    comparison: dict[str, Any],
    no_cheat: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if len(rows) < int(args.required_readiness_eval_rows):
        blockers.append("private_eval_slice_below_readiness_row_floor")
    if comparison["full_no_candidate_rate"] > float(args.max_no_candidate_rate):
        blockers.append("full_contract_no_candidate_rate_above_floor")
    if comparison["full_no_admissible_task_rate"] > float(args.max_no_candidate_rate):
        blockers.append("full_contract_no_admissible_rate_above_floor")
    if comparison["full_selected_pass_rate"] < float(args.min_selected_pass_rate):
        blockers.append("full_contract_selected_pass_rate_below_floor")
    if comparison["per_family_regression_count_vs_minimal"] > 0:
        blockers.append("per_family_regressions_vs_minimal_present")
    if no_cheat["fallback_return_count"] != 0:
        blockers.append("fallback_returns_present")
    if no_cheat["template_like_candidate_count"] != 0:
        blockers.append("template_like_candidates_present")
    if no_cheat["public_training_rows"] != 0 or no_cheat["forbidden_export_key_count"] != 0:
        blockers.append("public_boundary_not_clean")
    if int_number(full_summary.get("verifier_attempt_count")) == 0:
        blockers.append("private_verifier_attempts_missing")
    ready = not blockers
    return {
        "ready": ready,
        "decision": "ready_for_future_governed_public_calibration" if ready else "not_ready_private_repair_needed",
        "blockers": blockers,
        "next_private_repair_targets": next_private_repair_targets(comparison, full_summary),
        "readiness_floor": {
            "required_eval_rows": int(args.required_readiness_eval_rows),
            "min_selected_pass_rate": float(args.min_selected_pass_rate),
            "max_no_candidate_rate": float(args.max_no_candidate_rate),
        },
    }


def build_repair_targets(
    private_rows: list[dict[str, Any]],
    full_arm: dict[str, Any],
    minimal_arm: dict[str, Any],
) -> list[dict[str, Any]]:
    full_counts = object_field(full_arm, "private_verifier").get("concept_residual_counts")
    minimal_counts = object_field(minimal_arm, "private_verifier").get("concept_residual_counts")
    full_counts = full_counts if isinstance(full_counts, dict) else {}
    minimal_counts = minimal_counts if isinstance(minimal_counts, dict) else {}
    failing_labels = {str(label) for label, count in full_counts.items() if int_number(count) > 0}
    targets = []
    for row in private_rows:
        family = str(row.get("residual_concept") or row.get("category") or "")
        label = str(row.get("concept_residual_label") or family)
        if family not in failing_labels and label not in failing_labels:
            continue
        targets.append(
            {
                "policy": "project_theseus_full_body_contract_transfer_recovery_private_repair_target_v1",
                "task_id": str(row.get("task_id") or ""),
                "source_task_id": str(row.get("source_task_id") or row.get("task_id") or ""),
                "category": str(row.get("category") or ""),
                "residual_concept": family,
                "concept_residual_label": label,
                "entry_point": str(row.get("entry_point") or ""),
                "repair_focus": repair_focus_for_family(family),
                "full_context_failed": True,
                "minimal_context_failed": int_number(minimal_counts.get(family) or minimal_counts.get(label)) > 0,
                "public_benchmark": False,
                "public_tests_used": False,
                "public_solutions_used": False,
                "tests_exported": False,
                "solutions_exported": False,
                "candidate_code_exported": False,
            }
        )
    targets.sort(key=lambda row: stable_key({"task_id": row.get("task_id"), "family": row.get("residual_concept")}))
    return targets


def repair_focus_for_family(family: str) -> list[str]:
    if family.startswith("contract_blind_unit_"):
        return [
            "contract_blind_semantic_planning",
            "verifier_contract_alignment",
            "return_shape_from_prompt_only",
        ]
    if "stdin" in family:
        return ["io_contract_stdin_stdout", "verifier_contract_alignment"]
    if "return" in family or "type" in family:
        return ["return_type_shape", "verifier_contract_alignment"]
    return ["semantic_algorithmic_planning", "verifier_contract_alignment"]


def next_private_repair_targets(comparison: dict[str, Any], full_summary: dict[str, Any]) -> list[str]:
    targets = []
    if comparison["full_no_candidate_rate"] > 0:
        targets.append("candidate_coverage_no_candidate")
    if comparison["full_no_admissible_task_rate"] > 0:
        targets.append("promotion_grade_admissible_full_body_coverage")
    if comparison["full_selected_pass_rate"] < 0.95:
        stage_counts = full_summary.get("stage_counts") if isinstance(full_summary.get("stage_counts"), dict) else {}
        if int_number(stage_counts.get("runtime_loaded")) > 0:
            targets.append("semantic_algorithmic_planning")
        if int_number(stage_counts.get("lint_parse_failed")) or int_number(stage_counts.get("candidate_compile_failed")):
            targets.append("syntax_and_return_shape")
        targets.append("verifier_contract_alignment")
    if comparison["selected_pass_delta_full_minus_minimal"] < 0:
        targets.append("contract_context_feature_deweighting")
    if comparison["per_family_regression_count_vs_minimal"] > 0:
        targets.append("per_family_regression_repair")
    if not targets:
        targets.append("scale_private_eval_slice_to_readiness_floor")
    return targets


def no_cheat_summary(private_rows: list[dict[str, Any]], arms: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = []
    task_manifests = []
    for arm in arms:
        artifacts = object_field(arm, "artifacts")
        candidate_rows.extend(read_jsonl(resolve(str(artifacts.get("candidate_manifest") or ""))))
        task_manifests.extend(read_jsonl(resolve(str(artifacts.get("task_manifest") or ""))))
    return {
        "public_training_rows": public_flagged_rows(private_rows) + public_flagged_rows(candidate_rows),
        "public_task_manifest_rows": public_flagged_rows(task_manifests),
        "forbidden_export_key_count": forbidden_export_key_count(task_manifests),
        "fallback_return_count": sum(1 for row in candidate_rows if row.get("expression_memory_fallback") is True),
        "template_like_candidate_count": sum(1 for row in candidate_rows if row.get("template_like_candidate") is True),
        "external_inference_calls": sum(int_number(row.get("external_inference_calls")) for row in candidate_rows),
        "candidate_rows_scanned": len(candidate_rows),
        "task_manifest_rows_scanned": len(task_manifests),
    }


def weak_contract_blind_inclusion(rows: list[dict[str, Any]], previous_candidate_floor: Path) -> dict[str, Any]:
    wanted = weak_contract_blind_families(previous_candidate_floor)
    present = sorted(
        {
            str(row.get("residual_concept") or row.get("category") or "")
            for row in rows
            if str(row.get("residual_concept") or row.get("category") or "") in set(wanted)
        }
    )
    missing = sorted(set(wanted) - set(present))
    return {
        "wanted": wanted,
        "present": present,
        "missing": missing,
        "included_all": not missing,
    }


def weak_contract_blind_families(previous_candidate_floor: Path) -> list[str]:
    report = read_json(previous_candidate_floor, {})
    summary = object_field(report, "summary")
    rates = summary.get("private_family_pass_rates")
    if not isinstance(rates, dict):
        return []
    out = []
    for family, rate in rates.items():
        if str(family).startswith("contract_blind_unit_") and number(rate) <= 0.0:
            out.append(str(family))
    return sorted(out)


def arm_status(arms: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(arm.get("arm_name")): {
            "returncode": arm.get("returncode"),
            "candidate_count": get_path(arm, ["summary", "candidate_count"], 0),
            "selected_pass_rate": get_path(arm, ["summary", "selected_pass_rate"], 0),
            "no_candidate_rate": get_path(arm, ["summary", "no_candidate_rate"], 0),
        }
        for arm in arms
    }


def compact_private_eval(private_eval: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_task_count": private_eval.get("eval_task_count"),
        "trained_passed": private_eval.get("trained_passed"),
        "trained_pass_rate": private_eval.get("trained_pass_rate"),
        "baseline_pass_rate": private_eval.get("baseline_pass_rate"),
        "sts_off_pass_rate": private_eval.get("sts_off_pass_rate"),
        "residual_count": private_eval.get("residual_count"),
        "concept_residual_counts": private_eval.get("concept_residual_counts"),
        "concept_family_pass_rates": private_eval.get("concept_family_pass_rates"),
        "private_verification": private_eval.get("private_verification"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    readiness = object_field(report, "readiness")
    comparison = object_field(report, "comparison")
    lines = [
        "# Full-Body Contract Transfer Recovery v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- private eval rows: `{summary.get('private_eval_rows')}`",
        f"- full contract selected pass rate: `{summary.get('full_contract_selected_pass_rate')}`",
        f"- minimal contract selected pass rate: `{summary.get('minimal_contract_selected_pass_rate')}`",
        f"- selected pass delta full-minus-minimal: `{summary.get('selected_pass_delta_full_minus_minimal')}`",
        f"- full no-candidate rate: `{summary.get('full_no_candidate_rate')}`",
        f"- full no-admissible task rate: `{summary.get('full_no_admissible_task_rate')}`",
        f"- per-family regressions vs minimal: `{summary.get('per_family_regression_count_vs_minimal')}`",
        f"- per-family improvements vs minimal: `{summary.get('per_family_improvement_count_vs_minimal')}`",
        f"- readiness decision: `{summary.get('readiness_decision')}`",
        f"- readiness blockers: `{readiness.get('blockers')}`",
        "",
        "## No-Cheat Audit",
        f"- fallback returns: `{summary.get('fallback_return_count')}`",
        f"- template-like candidates: `{summary.get('template_like_candidate_count')}`",
        f"- public training rows: `{summary.get('public_training_rows')}`",
        f"- external inference calls: `{summary.get('external_inference_calls')}`",
        "",
        "## Comparison",
        f"- full verifier load rate: `{comparison.get('full_verifier_load_rate')}`",
        f"- minimal verifier load rate: `{comparison.get('minimal_verifier_load_rate')}`",
        f"- full candidates: `{comparison.get('full_candidate_count')}`",
        f"- minimal candidates: `{comparison.get('minimal_candidate_count')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        if isinstance(row, dict):
            lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}` ({row.get('severity')})")
    lines.append("")
    return "\n".join(lines)


def forbidden_export_key_count(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        total += sum(1 for key in FORBIDDEN_EXPORT_KEYS if key in row and row.get(key) not in (None, "", [], {}))
    return total


def public_flagged_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if any(row.get(key) is True for key in PUBLIC_FLAG_KEYS))


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        key = str(row.get("task_id") or row.get("entry_point") or stable_key(row))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def object_field(value: Any, key: str) -> dict[str, Any]:
    item = value.get(key) if isinstance(value, dict) else None
    return item if isinstance(item, dict) else {}


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def int_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def stable_key(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def read_json(path: Path, default: Any | None = None) -> Any:
    default = {} if default is None else default
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
