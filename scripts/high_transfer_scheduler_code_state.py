"""Private code-transfer gate and closure state helpers for the scheduler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from high_transfer_scheduler_common import *  # noqa: F401,F403


def execution_shape_private_gate(ablation: dict[str, Any]) -> dict[str, Any]:
    """Require clean private residual movement before public receiver reruns."""

    summary = ablation.get("summary") if isinstance(ablation.get("summary"), dict) else {}
    family = get_path(ablation, ["family_results", "execution_shape_skeleton_decoder_private_v1"], {})
    if not isinstance(family, dict):
        family = {}
    pass_rate = safe_float(summary.get("execution_shape_skeleton_pass_rate") or family.get("private_pass_rate"))
    min_pass_rate = safe_float(summary.get("skeleton_public_gate_min_pass_rate") or 0.70)
    no_admissible = int(family.get("no_admissible_candidate_count") or summary.get("skeleton_no_admissible_candidate_count") or 0)
    zero_categories = summary.get("skeleton_zero_pass_categories") or [
        category
        for category, rate in sorted((family.get("per_category_pass_rate") or {}).items())
        if safe_float(rate) <= 0.0
    ]
    checks = {
        "trigger_green": ablation.get("trigger_state") == "GREEN",
        "skeleton_competitive_with_semantic": bool(summary.get("skeleton_competitive_with_semantic")),
        "skeleton_min_pass_rate": pass_rate >= min_pass_rate,
        "no_admissible_candidates_cleared": no_admissible == 0,
        "all_private_execution_shape_categories_covered": not zero_categories,
    }
    ready = all(checks.values())
    if ready:
        reason = "private_ablation_clean_enough_for_public_receiver_calibration"
    elif not checks["trigger_green"]:
        reason = "private_ablation_report_not_green"
    elif not checks["skeleton_min_pass_rate"]:
        reason = "private_skeleton_pass_rate_below_public_gate"
    elif not checks["no_admissible_candidates_cleared"]:
        reason = "private_no_admissible_candidate_residuals_remain"
    elif not checks["all_private_execution_shape_categories_covered"]:
        reason = "private_zero_pass_execution_shape_categories_remain"
    else:
        reason = "private_ablation_gate_not_clean"
    return {
        "ready_for_public_calibration": ready,
        "reason": reason,
        "checks": checks,
        "skeleton_pass_rate": pass_rate,
        "minimum_skeleton_pass_rate": min_pass_rate,
        "skeleton_no_admissible_candidate_count": no_admissible,
        "skeleton_zero_pass_categories": list(zero_categories),
    }


def decoder_v2_private_ablation_gate_state() -> dict[str, Any]:
    """Require private ablation proof after private closure and decoder edits."""

    ablation = read_json(DECODER_V2_PRIVATE_ABLATION_REPORT, {})
    ablation_mtime = file_mtime(DECODER_V2_PRIVATE_ABLATION_REPORT)
    closure_candidates = [
        BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_CLOSURE_REPORT,
        BROAD_PRIVATE_PRESSURE_CLOSURE_REPORT,
        EDGE_CONTRACT_V2_CLOSURE_REPORT,
        EDGE_CASE_FULL_BODY_CLOSURE_REPORT,
        BALANCED_EDGE_CONTRACT_CLOSURE_REPORT,
        EDGE_CONTRACT_CLOSURE_REPORT,
    ]
    existing_closures = [path for path in closure_candidates if path.exists()]
    latest_closure = max(existing_closures, key=lambda path: path.stat().st_mtime) if existing_closures else None
    latest_closure_mtime = latest_closure.stat().st_mtime if latest_closure else 0.0
    ready = bool(ablation.get("ready_for_public_calibration"))
    trigger_state = str(ablation.get("trigger_state") or "")
    decoder_fingerprint = decoder_relevant_source_fingerprint()
    reported_fingerprint = str(get_path(ablation, ["closure_reports", 0, "decoder_relevant_source_fingerprint"], "") or "")
    source_changed_after_gate = bool(
        decoder_relevant_source_mtime()
        and ablation_mtime
        and decoder_relevant_source_mtime() > ablation_mtime
    )
    latest_closure_stale_for_decoder = bool(
        decoder_relevant_source_mtime()
        and latest_closure_mtime
        and decoder_relevant_source_mtime() > latest_closure_mtime
    )
    gate_current = bool(
        ready
        and ablation_mtime
        and latest_closure_mtime
        and ablation_mtime >= latest_closure_mtime
        and not source_changed_after_gate
        and not latest_closure_stale_for_decoder
    )
    needs_ablation = bool(
        latest_closure_mtime
        and not latest_closure_stale_for_decoder
        and (not ablation_mtime or ablation_mtime < latest_closure_mtime or source_changed_after_gate or trigger_state != "GREEN")
    )
    if gate_current:
        reason = "decoder_v2_private_ablation_gate_current"
    elif latest_closure_stale_for_decoder:
        reason = "private_closure_stale_for_decoder_source"
    elif source_changed_after_gate:
        reason = "decoder_source_newer_than_decoder_v2_private_ablation_gate"
    elif latest_closure_mtime and not ablation_mtime:
        reason = "missing_decoder_v2_private_ablation_gate"
    elif latest_closure_mtime and ablation_mtime < latest_closure_mtime:
        reason = "private_closure_newer_than_decoder_v2_private_ablation_gate"
    elif trigger_state != "GREEN":
        reason = "decoder_v2_private_ablation_gate_not_green"
    else:
        reason = "decoder_v2_private_ablation_gate_waiting_for_private_closure"
    return {
        "ready_for_public_calibration": gate_current,
        "needs_ablation": needs_ablation,
        "reason": reason,
        "ablation_report": str(DECODER_V2_PRIVATE_ABLATION_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "ablation_report_exists": DECODER_V2_PRIVATE_ABLATION_REPORT.exists(),
        "ablation_report_mtime": ablation_mtime or None,
        "ablation_trigger_state": trigger_state or None,
        "latest_private_closure_report": str(latest_closure.relative_to(ROOT)).replace("\\", "/") if latest_closure else None,
        "latest_private_closure_mtime": latest_closure_mtime or None,
        "latest_private_closure_stale_for_decoder": latest_closure_stale_for_decoder,
        "source_changed_after_gate": source_changed_after_gate,
        "decoder_relevant_source_fingerprint": decoder_fingerprint,
        "reported_decoder_relevant_source_fingerprint": reported_fingerprint or None,
        "gate_summary": ablation.get("summary") if isinstance(ablation.get("summary"), dict) else {},
    }


def private_type_shape_receiver_ablation_state() -> dict[str, Any]:
    """Track the teacher-requested private type/shape receiver ablation gate."""

    report = read_json(PRIVATE_TYPE_SHAPE_RECEIVER_ABLATION_REPORT, {})
    report_mtime = file_mtime(PRIVATE_TYPE_SHAPE_RECEIVER_ABLATION_REPORT)
    teacher_path = REPORTS / "teacher_public_transfer_residual_last.json"
    teacher = read_json(teacher_path, {})
    teacher_mtime = file_mtime(teacher_path)
    teacher_spec_present = "private_type_shape_receiver_veto_ablation_v1" in json.dumps(
        teacher,
        sort_keys=True,
    ).lower()
    candidate_manifest = REPORTS / "code_lm_private_candidates.jsonl"
    candidate_mtime = file_mtime(candidate_manifest)
    closure_candidates = [
        BROAD_PRIVATE_PRESSURE_CLOSURE_REPORT,
        EDGE_CONTRACT_V2_CLOSURE_REPORT,
        EDGE_CASE_FULL_BODY_CLOSURE_REPORT,
        BALANCED_EDGE_CONTRACT_CLOSURE_REPORT,
        EDGE_CONTRACT_CLOSURE_REPORT,
    ]
    existing_closures = [path for path in closure_candidates if path.exists()]
    latest_closure = max(existing_closures, key=lambda path: path.stat().st_mtime) if existing_closures else None
    latest_closure_mtime = latest_closure.stat().st_mtime if latest_closure else 0.0
    private_closure_ready = bool(latest_closure_mtime and latest_closure_mtime >= decoder_relevant_source_mtime())
    policy_current = str(report.get("policy") or "") == "project_theseus_private_type_shape_receiver_veto_ablation_v1"
    trigger_green = report.get("trigger_state") == "GREEN"
    ready = bool(report.get("ready_for_public_calibration"))
    required_mtime = max(
        teacher_mtime if teacher_spec_present else 0.0,
        candidate_mtime,
        latest_closure_mtime,
    )
    ablation_current = bool(
        policy_current
        and trigger_green
        and ready
        and report_mtime
        and (not required_mtime or report_mtime >= required_mtime)
    )
    needs_ablation = bool(
        teacher_spec_present
        and private_closure_ready
        and candidate_mtime
        and not ablation_current
    )
    if ablation_current:
        reason = "private_type_shape_receiver_ablation_current"
    elif not teacher_spec_present:
        reason = "teacher_private_type_shape_receiver_spec_not_present"
    elif not private_closure_ready:
        reason = "waiting_for_current_private_closure_before_receiver_ablation"
    elif not candidate_mtime:
        reason = "missing_private_candidate_manifest_for_receiver_ablation"
    elif not report_mtime:
        reason = "missing_private_type_shape_receiver_ablation"
    elif report_mtime < required_mtime:
        reason = "private_type_shape_receiver_ablation_stale_for_teacher_candidates_or_closure"
    elif not policy_current:
        reason = "private_type_shape_receiver_ablation_wrong_policy"
    elif not trigger_green:
        reason = "private_type_shape_receiver_ablation_not_green"
    elif not ready:
        reason = "private_type_shape_receiver_ablation_not_ready_for_public_calibration"
    else:
        reason = "private_type_shape_receiver_ablation_not_current"
    return {
        "ready_for_public_calibration": ablation_current,
        "needs_ablation": needs_ablation,
        "reason": reason,
        "teacher_spec_present": teacher_spec_present,
        "teacher_spec_report": str(teacher_path.relative_to(ROOT)).replace("\\", "/"),
        "teacher_spec_mtime": teacher_mtime or None,
        "ablation_report": str(PRIVATE_TYPE_SHAPE_RECEIVER_ABLATION_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "ablation_report_exists": PRIVATE_TYPE_SHAPE_RECEIVER_ABLATION_REPORT.exists(),
        "ablation_report_mtime": report_mtime or None,
        "ablation_trigger_state": report.get("trigger_state"),
        "ablation_policy": report.get("policy"),
        "candidate_manifest": str(candidate_manifest.relative_to(ROOT)).replace("\\", "/"),
        "candidate_manifest_mtime": candidate_mtime or None,
        "latest_private_closure_report": str(latest_closure.relative_to(ROOT)).replace("\\", "/") if latest_closure else None,
        "latest_private_closure_mtime": latest_closure_mtime or None,
        "private_closure_ready": private_closure_ready,
        "required_mtime": required_mtime or None,
        "gate_deltas": report.get("deltas") if isinstance(report.get("deltas"), dict) else {},
        "gate_summary": {
            "matched_eval_task_count": get_path(report, ["summary", "matched_eval_task_count"], None),
            "candidate_count": get_path(report, ["summary", "candidate_count"], None),
            "enabled_private_pass_rate": get_path(report, ["summary", "receiver_enabled_private_pass_rate"], None),
            "disabled_private_pass_rate": get_path(report, ["summary", "receiver_disabled_private_pass_rate"], None),
            "private_pass_rate_delta": get_path(report, ["deltas", "private_pass_rate_delta"], None),
            "body_exec_pass_rate_delta": get_path(report, ["deltas", "body_exec_pass_rate_delta"], None),
            "return_shape_ok": get_path(report, ["summary", "return_shape_ok"], None),
            "accepted_candidate_coverage_ratio": get_path(report, ["deltas", "accepted_candidate_coverage_ratio"], None),
        },
    }


def edge_exec_repair_lifecycle(edge_report: dict[str, Any], broad: dict[str, Any]) -> dict[str, Any]:
    """Classify the teacher edge-exec experiment without using public answers.

    Graduation here means "this exact experiment produced its intended receiver
    signal and should stop being re-run as the active scheduler task." It does
    not mean broad code transfer is solved.
    """
    summary = edge_report.get("summary") if isinstance(edge_report.get("summary"), dict) else {}
    rates: dict[str, float] = {}
    regressions: dict[str, int] = {}
    for suite in edge_report.get("suites", []):
        if not isinstance(suite, dict):
            continue
        card = str(suite.get("card_id") or "")
        if not card:
            continue
        rates[card] = float(suite.get("multi_stream_pass_rate") or 0.0)
        regressions[card] = int(suite.get("task_level_regressions") or 0)
    public_task_count = int(summary.get("public_task_count") or 0)
    template_like = int(summary.get("template_like_candidate_count") or 0)
    loop_closure = int(summary.get("loop_closure_candidate_count") or 0)
    external_calls = int(summary.get("external_inference_calls") or edge_report.get("external_inference_calls") or 0)
    task_regressions = int(summary.get("task_level_regressions_vs_single_stream") or 0) + sum(regressions.values())
    cards_below_floor = get_path(broad, ["summary", "cards_below_floor"], []) or []
    human_eval_ok = "source_human_eval" not in cards_below_floor
    bigcodebench_rate = float(rates.get("source_bigcodebench", 0.0) or 0.0)
    livecodebench_rate = float(rates.get("source_livecodebench", 0.0) or 0.0)
    # LiveCodeBench moved off zero in the first edge-exec pass, but
    # BigCodeBench is the remaining hard-zero receiver. Keep this experiment
    # active until the systems-style receiver also produces a clean pass.
    receiver_signal = bigcodebench_rate > 0.0
    clean = (
        public_task_count >= 128
        and template_like == 0
        and loop_closure == 0
        and external_calls == 0
        and task_regressions == 0
        and human_eval_ok
    )
    graduated = bool(receiver_signal and clean)
    if graduated:
        reason = "edge_exec_receiver_signal_with_clean_calibration"
    elif public_task_count == 0:
        reason = "missing_edge_exec_calibration"
    elif not receiver_signal:
        reason = "bigcodebench_zero"
    elif not clean:
        reason = "edge_exec_cleanliness_or_regression_gate_failed"
    else:
        reason = "edge_exec_frontier_open"
    return {
        "graduated": graduated,
        "reason": reason,
        "public_task_count": public_task_count,
        "receiver_rates": rates,
        "receiver_signal": receiver_signal,
        "receiver_off_zero": bigcodebench_rate > 0.0 or livecodebench_rate > 0.0,
        "bigcodebench_rate": bigcodebench_rate,
        "livecodebench_rate": livecodebench_rate,
        "template_like_candidate_count": template_like,
        "loop_closure_candidate_count": loop_closure,
        "external_inference_calls": external_calls,
        "task_regressions": task_regressions,
        "human_eval_ok": human_eval_ok,
    }


def four_card_calibration_mtime() -> float:
    """Return the freshest completed four-card calibration evidence timestamp."""

    evidence_specs: list[tuple[Path, Any]] = [
        (
            REPORTS
            / "broad_transfer_closure_runner_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.json",
            lambda data: data.get("trigger_state") == "GREEN"
            and int(get_path(data, ["summary", "public_task_count"], 0) or 0) >= 128,
        ),
        (
            REPORTS
            / "real_code_benchmark_graduation_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.json",
            lambda data: data.get("trigger_state") == "GREEN"
            and int(get_path(data, ["summary", "public_task_count"], 0) or 0) >= 128,
        ),
        (
            REPORTS
            / "code_lm_closure_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.json",
            lambda data: data.get("run_status") == "completed"
            and int(get_path(data, ["summary", "public_task_count"], 0) or 0) >= 128,
        ),
    ]
    for path in REPORTS.glob("real_code_benchmark_graduation_*4card.json"):
        evidence_specs.append(
            (
                path,
                lambda data: data.get("trigger_state") == "GREEN"
                and int(get_path(data, ["summary", "public_task_count"], 0) or 0) >= 128,
            )
        )
    for path in REPORTS.glob("code_lm_closure_*4card.json"):
        evidence_specs.append(
            (
                path,
                lambda data: data.get("run_status") == "completed"
                and int(get_path(data, ["summary", "public_task_count"], 0) or 0) >= 128,
            )
        )
    mtimes: list[float] = []
    for path, is_complete in evidence_specs:
        if not path.exists():
            continue
        data = read_json(path, {})
        if isinstance(data, dict) and is_complete(data):
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def edge_contract_v2_public_calibration_limit_state(calibration_mtime: float | None = None) -> dict[str, Any]:
    """Track the one-shot public receiver calibration allowed by the v2 gate."""

    verifier = read_json(EDGE_CONTRACT_V2_VERIFIER_REPORT, {})
    verifier_ready = bool(verifier.get("ready_for_public_calibration"))
    verifier_mtime = file_mtime(EDGE_CONTRACT_V2_VERIFIER_REPORT)
    calibration = float(calibration_mtime or four_card_calibration_mtime() or 0.0)
    consumed = bool(verifier_ready and verifier_mtime and calibration and calibration > verifier_mtime)
    return {
        "limit_reached": consumed,
        "verifier_ready_for_public_calibration": verifier_ready,
        "verifier_mtime": verifier_mtime or None,
        "last_four_card_calibration_mtime": calibration or None,
        "reason": "post_edge_contract_v2_public_calibration_limit_reached" if consumed else "post_edge_contract_v2_public_calibration_available",
    }


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def decoder_relevant_source_fingerprint() -> str:
    """Hash decoder-relevant Rust lines so unrelated bookkeeping edits do not reopen gates."""

    if not DECODER_SOURCE.exists():
        return ""
    text = DECODER_SOURCE.read_text(encoding="utf-8", errors="replace")
    relevant = "\n".join(
        line for line in text.splitlines() if any(marker in line for marker in DECODER_FINGERPRINT_MARKERS)
    )
    return hashlib.sha256(relevant.encode("utf-8")).hexdigest()[:16]


def decoder_relevant_source_mtime() -> float:
    """Return the mtime for source that can causally change decoder generation.

    Python wrappers and report writers are tracked through task/report freshness.
    Treating every wrapper edit as decoder drift wastes completed private
    closures when the edit only improves evidence plumbing.
    """

    return file_mtime(DECODER_SOURCE)


def balanced_edge_contract_experiment_state(guidance: dict[str, Any]) -> dict[str, Any]:
    """Track the teacher-requested private-first balanced edge-contract experiment."""

    teacher_last = read_json(REPORTS / "teacher_architecture_guidance_last.json", {})
    recommendation = " ".join(
        str(item or "")
        for item in [
            get_path(guidance, ["teacher", "response_json", "recommended_intervention"], ""),
            get_path(guidance, ["teacher", "response_json", "diagnosis"], ""),
            get_path(teacher_last, ["response_json", "recommended_intervention"], ""),
            get_path(teacher_last, ["response_json", "diagnosis"], ""),
        ]
    ).lower()
    teacher_requested = (
        "balanced" in recommendation
        and "edge" in recommendation
        and "contract" in recommendation
        and "private" in recommendation
    )
    pressure_report = BALANCED_EDGE_CONTRACT_PRESSURE_REPORT
    closure_report = BALANCED_EDGE_CONTRACT_CLOSURE_REPORT
    pressure = read_json(pressure_report, {})
    closure = read_json(closure_report, {})
    pressure_mtime = file_mtime(pressure_report)
    closure_mtime = file_mtime(closure_report)
    decoder_source_mtime = decoder_relevant_source_mtime()
    decoder_source_changed_after_closure = bool(
        decoder_source_mtime
        and (not closure_mtime or decoder_source_mtime > closure_mtime)
    )
    pressure_current = bool(
        pressure_report.exists()
        and pressure.get("trigger_state") == "GREEN"
        and int(get_path(pressure, ["summary", "private_row_count"], 0) or 0) > 0
        and safe_float(get_path(pressure, ["summary", "execution_shaped_programs_share"], 1.0)) <= 0.25
        and int(get_path(pressure, ["summary", "benchmark_named_private_rows"], 1) or 0) == 0
    )
    closure_current = bool(
        pressure_current
        and closure_report.exists()
        and closure_mtime >= pressure_mtime
        and closure.get("trigger_state") == "GREEN"
        and closure.get("run_status") == "completed"
        and not decoder_source_changed_after_closure
    )
    needs_pressure = bool(teacher_requested and not pressure_current)
    needs_private_closure = bool(teacher_requested and pressure_current and not closure_current)
    return {
        "teacher_requested": teacher_requested,
        "teacher_guidance_created_utc": guidance.get("created_utc") or teacher_last.get("created_utc"),
        "pressure_report": str(pressure_report.relative_to(ROOT)).replace("\\", "/"),
        "pressure_report_exists": pressure_report.exists(),
        "pressure_report_mtime": pressure_mtime or None,
        "pressure_trigger_state": pressure.get("trigger_state"),
        "pressure_current": pressure_current,
        "private_train_jsonl": str(BALANCED_EDGE_CONTRACT_TRAIN_JSONL).replace("\\", "/"),
        "closure_report": str(closure_report.relative_to(ROOT)).replace("\\", "/"),
        "closure_report_exists": closure_report.exists(),
        "closure_report_mtime": closure_mtime or None,
        "closure_trigger_state": closure.get("trigger_state"),
        "closure_run_status": closure.get("run_status"),
        "closure_current": closure_current,
        "needs_pressure": needs_pressure,
        "needs_private_closure": needs_private_closure,
        "blocks_public_recalibration": bool(teacher_requested and not closure_current),
        "execution_shaped_programs_share": get_path(pressure, ["summary", "execution_shaped_programs_share"], None),
        "benchmark_named_private_rows": get_path(pressure, ["summary", "benchmark_named_private_rows"], None),
        "decoder_relevant_source_mtime": decoder_source_mtime or None,
        "decoder_source_changed_after_closure": decoder_source_changed_after_closure,
        "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
    }


def edge_case_full_body_experiment_state(guidance: dict[str, Any]) -> dict[str, Any]:
    """Track the teacher-requested private-first full-body edge-case experiment."""

    teacher_last = read_json(REPORTS / "teacher_architecture_guidance_last.json", {})
    text_parts = [
        get_path(guidance, ["teacher", "response_json", "recommended_intervention"], ""),
        get_path(guidance, ["teacher", "response_json", "diagnosis"], ""),
        get_path(guidance, ["teacher", "response_json", "experiment_spec", "id"], ""),
        get_path(teacher_last, ["response_json", "recommended_intervention"], ""),
        get_path(teacher_last, ["response_json", "diagnosis"], ""),
        get_path(teacher_last, ["response_json", "experiment_spec", "id"], ""),
    ]
    recommendation = " ".join(str(item or "") for item in text_parts).lower()
    teacher_requested = (
        ("edge-case" in recommendation or "edge case" in recommendation or "edge_case" in recommendation)
        and "full-body" in recommendation
        and "private" in recommendation
    ) or "residual_targeted_private_edge_case_full_body_curriculum_v1" in recommendation
    pressure_report = EDGE_CASE_FULL_BODY_PRESSURE_REPORT
    closure_report = EDGE_CASE_FULL_BODY_CLOSURE_REPORT
    pressure = read_json(pressure_report, {})
    closure = read_json(closure_report, {})
    pressure_mtime = file_mtime(pressure_report)
    closure_mtime = file_mtime(closure_report)
    decoder_source_mtime = decoder_relevant_source_mtime()
    decoder_source_changed_after_closure = bool(
        decoder_source_mtime
        and (not closure_mtime or decoder_source_mtime > closure_mtime)
    )
    pressure_current = bool(
        pressure_report.exists()
        and pressure.get("trigger_state") == "GREEN"
        and int(get_path(pressure, ["summary", "private_row_count"], 0) or 0) > 0
        and int(get_path(pressure, ["summary", "private_solution_test_failures"], 1) or 0) == 0
        and int(get_path(pressure, ["summary", "edge_case_full_body_contract_rows"], 0) or 0) > 0
        and int(get_path(pressure, ["summary", "benchmark_named_private_rows"], 1) or 0) == 0
    )
    private_baseline = safe_float(get_path(closure, ["summary", "private_baseline_pass_rate"], 0.0))
    private_trained = safe_float(get_path(closure, ["summary", "private_trained_pass_rate"], 0.0))
    private_delta = safe_float(get_path(closure, ["summary", "private_pass_rate_delta"], 0.0))
    next_token_delta = safe_float(get_path(closure, ["summary", "next_token_accuracy_delta"], 0.0))
    sts_regressions = int(get_path(closure, ["summary", "private_sts_repair_task_level_regressions"], 0) or 0)
    private_gate = {
        "ready_for_public_calibration": bool(
            closure_report.exists()
            and closure.get("trigger_state") == "GREEN"
            and closure.get("run_status") == "completed"
            and private_trained >= private_baseline
            and private_delta >= 0.05
            and next_token_delta >= 0.0
            and sts_regressions == 0
        ),
        "private_baseline_pass_rate": private_baseline,
        "private_trained_pass_rate": private_trained,
        "private_pass_rate_delta": private_delta,
        "next_token_accuracy_delta": next_token_delta,
        "private_sts_repair_task_level_regressions": sts_regressions,
        "minimum_private_pass_rate_delta": 0.05,
        "score_semantics": "held-out private gate only; public benchmarks remain calibration-only",
    }
    closure_current = bool(
        pressure_current
        and closure_report.exists()
        and closure_mtime >= pressure_mtime
        and closure.get("trigger_state") == "GREEN"
        and closure.get("run_status") == "completed"
        and private_gate["ready_for_public_calibration"]
        and not decoder_source_changed_after_closure
    )
    needs_pressure = bool(teacher_requested and not pressure_current)
    needs_private_closure = bool(teacher_requested and pressure_current and not closure_current)
    return {
        "teacher_requested": teacher_requested,
        "teacher_guidance_created_utc": guidance.get("created_utc") or teacher_last.get("created_utc"),
        "pressure_report": str(pressure_report.relative_to(ROOT)).replace("\\", "/"),
        "pressure_report_exists": pressure_report.exists(),
        "pressure_report_mtime": pressure_mtime or None,
        "pressure_trigger_state": pressure.get("trigger_state"),
        "pressure_current": pressure_current,
        "private_train_jsonl": str(EDGE_CASE_FULL_BODY_TRAIN_JSONL).replace("\\", "/"),
        "closure_report": str(closure_report.relative_to(ROOT)).replace("\\", "/"),
        "closure_report_exists": closure_report.exists(),
        "closure_report_mtime": closure_mtime or None,
        "closure_trigger_state": closure.get("trigger_state"),
        "closure_run_status": closure.get("run_status"),
        "closure_current": closure_current,
        "needs_pressure": needs_pressure,
        "needs_private_closure": needs_private_closure,
        "blocks_public_recalibration": bool(teacher_requested and not closure_current),
        "private_gate": private_gate,
        "edge_case_full_body_contract_rows": get_path(pressure, ["summary", "edge_case_full_body_contract_rows"], None),
        "benchmark_named_private_rows": get_path(pressure, ["summary", "benchmark_named_private_rows"], None),
        "decoder_relevant_source_mtime": decoder_source_mtime or None,
        "decoder_source_changed_after_closure": decoder_source_changed_after_closure,
        "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
    }


def edge_contract_v2_experiment_state(guidance: dict[str, Any]) -> dict[str, Any]:
    """Track the private-first edge-contract v2 curriculum and verifier gate."""

    teacher_last = read_json(REPORTS / "teacher_architecture_guidance_last.json", {})
    text_parts = [
        get_path(guidance, ["teacher", "response_json", "recommended_intervention"], ""),
        get_path(guidance, ["teacher", "response_json", "diagnosis"], ""),
        get_path(guidance, ["teacher", "response_json", "experiment_spec", "id"], ""),
        get_path(teacher_last, ["response_json", "recommended_intervention"], ""),
        get_path(teacher_last, ["response_json", "diagnosis"], ""),
        get_path(teacher_last, ["response_json", "experiment_spec", "id"], ""),
    ]
    recommendation = " ".join(str(item or "") for item in text_parts).lower()
    teacher_requested = (
        "edge_contract_v2" in recommendation
        or "edge contract v2" in recommendation
        or ("private residual curriculum" in recommendation and "verifier" in recommendation)
    )
    # The user explicitly promoted this teacher intervention, so keep it
    # eligible even if the latest teacher report has already rotated.
    user_promoted = True
    pressure_report = EDGE_CONTRACT_V2_PRESSURE_REPORT
    closure_report = EDGE_CONTRACT_V2_CLOSURE_REPORT
    verifier_report = EDGE_CONTRACT_V2_VERIFIER_REPORT
    pressure = read_json(pressure_report, {})
    closure = read_json(closure_report, {})
    verifier = read_json(verifier_report, {})
    pressure_mtime = file_mtime(pressure_report)
    closure_mtime = file_mtime(closure_report)
    verifier_mtime = file_mtime(verifier_report)
    decoder_source_mtime = decoder_relevant_source_mtime()
    decoder_source_changed_after_closure = bool(
        decoder_source_mtime
        and (not closure_mtime or decoder_source_mtime > closure_mtime)
    )
    pressure_current = bool(
        pressure_report.exists()
        and pressure.get("trigger_state") == "GREEN"
        and int(get_path(pressure, ["summary", "private_row_count"], 0) or 0) > 0
        and int(get_path(pressure, ["summary", "private_solution_test_failures"], 1) or 0) == 0
        and int(get_path(pressure, ["summary", "edge_contract_v2_rows"], 0) or 0) > 0
        and int(get_path(pressure, ["summary", "edge_contract_v2_generation_plan_rows"], 0) or 0)
        == int(get_path(pressure, ["summary", "edge_contract_v2_rows"], -1) or -1)
        and int(get_path(pressure, ["summary", "benchmark_named_private_rows"], 1) or 0) == 0
    )
    private_baseline = safe_float(get_path(closure, ["summary", "private_baseline_pass_rate"], 0.0))
    private_trained = safe_float(get_path(closure, ["summary", "private_trained_pass_rate"], 0.0))
    private_delta = safe_float(get_path(closure, ["summary", "private_pass_rate_delta"], 0.0))
    next_token_delta = safe_float(get_path(closure, ["summary", "next_token_accuracy_delta"], 0.0))
    sts_regressions = int(get_path(closure, ["summary", "private_sts_repair_task_level_regressions"], 0) or 0)
    verifier_ready = bool(verifier.get("ready_for_public_calibration"))
    private_gate = {
        "ready_for_public_calibration": bool(
            closure_report.exists()
            and closure.get("trigger_state") == "GREEN"
            and closure.get("run_status") == "completed"
            and private_trained >= private_baseline
            and private_delta >= 0.05
            and next_token_delta >= 0.0
            and sts_regressions == 0
            and verifier_ready
        ),
        "private_baseline_pass_rate": private_baseline,
        "private_trained_pass_rate": private_trained,
        "private_pass_rate_delta": private_delta,
        "next_token_accuracy_delta": next_token_delta,
        "private_sts_repair_task_level_regressions": sts_regressions,
        "verifier_ready_for_public_calibration": verifier_ready,
        "minimum_private_pass_rate_delta": 0.05,
        "score_semantics": "edge-contract-v2 held-out private gate only; public benchmarks remain calibration-only",
    }
    closure_current = bool(
        pressure_current
        and closure_report.exists()
        and closure_mtime >= pressure_mtime
        and verifier_report.exists()
        and verifier_mtime >= closure_mtime
        and closure.get("trigger_state") == "GREEN"
        and closure.get("run_status") == "completed"
        and private_gate["ready_for_public_calibration"]
        and not decoder_source_changed_after_closure
    )
    requested = teacher_requested or user_promoted
    needs_pressure = bool(requested and not pressure_current)
    needs_private_closure = bool(requested and pressure_current and not closure_current)
    return {
        "teacher_requested": requested,
        "teacher_guidance_created_utc": guidance.get("created_utc") or teacher_last.get("created_utc"),
        "pressure_report": str(pressure_report.relative_to(ROOT)).replace("\\", "/"),
        "pressure_report_exists": pressure_report.exists(),
        "pressure_report_mtime": pressure_mtime or None,
        "pressure_trigger_state": pressure.get("trigger_state"),
        "pressure_current": pressure_current,
        "private_train_jsonl": str(EDGE_CONTRACT_V2_TRAIN_JSONL).replace("\\", "/"),
        "closure_report": str(closure_report.relative_to(ROOT)).replace("\\", "/"),
        "closure_report_exists": closure_report.exists(),
        "closure_report_mtime": closure_mtime or None,
        "closure_trigger_state": closure.get("trigger_state"),
        "closure_run_status": closure.get("run_status"),
        "verifier_report": str(verifier_report.relative_to(ROOT)).replace("\\", "/"),
        "verifier_report_exists": verifier_report.exists(),
        "verifier_report_mtime": verifier_mtime or None,
        "verifier_trigger_state": verifier.get("trigger_state"),
        "closure_current": closure_current,
        "needs_pressure": needs_pressure,
        "needs_private_closure": needs_private_closure,
        "blocks_public_recalibration": bool(requested and not closure_current),
        "private_gate": private_gate,
        "edge_contract_v2_rows": get_path(pressure, ["summary", "edge_contract_v2_rows"], None),
        "edge_contract_v2_generation_plan_rows": get_path(pressure, ["summary", "edge_contract_v2_generation_plan_rows"], None),
        "benchmark_named_private_rows": get_path(pressure, ["summary", "benchmark_named_private_rows"], None),
        "decoder_relevant_source_mtime": decoder_source_mtime or None,
        "decoder_source_changed_after_closure": decoder_source_changed_after_closure,
        "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
    }


def private_pressure_recalibration_state() -> dict[str, Any]:
    pressure_reports = list(RECEIVER_RECALIBRATION_PRESSURE_REPORTS)
    calibration_mtime = four_card_calibration_mtime()
    existing = [path for path in pressure_reports if path.exists()]
    missing = [path for path in pressure_reports if not path.exists()]
    if not existing:
        return {
            "ready_for_recalibration": False,
            "reason": "missing_private_pressure_reports",
            "missing": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
            "fresh_private_pressure_reports": [],
            "latest_private_pressure_report": None,
            "latest_private_pressure_mtime": None,
            "last_four_card_calibration_mtime": calibration_mtime or None,
            "rotation_epoch": None,
        }
    latest = max(existing, key=lambda path: path.stat().st_mtime)
    latest_mtime = latest.stat().st_mtime
    fresh = [path for path in existing if path.stat().st_mtime > calibration_mtime]
    stale = [path for path in existing if path.stat().st_mtime <= calibration_mtime]
    typed_closure = typed_interface_private_closure_state()
    broad_private_closure = private_pressure_private_closure_state()
    edge_contract_closure = edge_contract_private_closure_state()
    edge_full_body = edge_case_full_body_experiment_state(
        read_json(REPORTS / "architecture_guidance_loop_edge_contract_4card_private_4card_seed31_32.json", {})
    )
    edge_v2 = edge_contract_v2_experiment_state(
        read_json(REPORTS / "architecture_guidance_loop_semantic_decoder_v2_4card.json", {})
    )
    broad_closure_mtime = float(broad_private_closure.get("closure_report_mtime") or 0.0)
    broad_closure_after_calibration = bool(
        broad_private_closure.get("allows_public_recalibration")
        and broad_closure_mtime
        and (not calibration_mtime or broad_closure_mtime > calibration_mtime)
    )
    edge_closure_mtime = float(edge_contract_closure.get("closure_report_mtime") or 0.0)
    edge_closure_after_calibration = bool(
        edge_contract_closure.get("allows_public_recalibration")
        and edge_closure_mtime
        and (not calibration_mtime or edge_closure_mtime > calibration_mtime)
    )
    edge_full_mtime = float(edge_full_body.get("closure_report_mtime") or 0.0)
    edge_full_closure_after_calibration = bool(
        edge_full_body.get("closure_current")
        and edge_full_mtime
        and (not calibration_mtime or edge_full_mtime > calibration_mtime)
    )
    edge_v2_mtime = float(edge_v2.get("closure_report_mtime") or 0.0)
    edge_v2_closure_after_calibration = bool(
        edge_v2.get("closure_current")
        and edge_v2_mtime
        and (not calibration_mtime or edge_v2_mtime > calibration_mtime)
    )
    type_shape_receiver_gate = private_type_shape_receiver_ablation_state()
    type_shape_receiver_gate_mtime = float(type_shape_receiver_gate.get("ablation_report_mtime") or 0.0)
    type_shape_receiver_gate_after_calibration = bool(
        type_shape_receiver_gate.get("ready_for_public_calibration")
        and type_shape_receiver_gate_mtime
        and (not calibration_mtime or type_shape_receiver_gate_mtime > calibration_mtime)
    )
    closure_supports_recalibration = (
        broad_closure_after_calibration
        or edge_closure_after_calibration
        or edge_full_closure_after_calibration
        or edge_v2_closure_after_calibration
        or type_shape_receiver_gate_after_calibration
    )
    ready = bool(fresh) or closure_supports_recalibration
    typed_report_rel = "reports/high_transfer_typed_interface_skeleton_code_residual_curriculum.json"
    typed_is_fresh = typed_report_rel in {str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh}
    edge_report_rel = "reports/high_transfer_edge_contract_4card_code_residual_curriculum.json"
    edge_is_fresh = edge_report_rel in {str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh}
    edge_full_report_rel = "reports/high_transfer_edge_case_full_body_private_curriculum_v1_code_residual_curriculum.json"
    edge_full_is_fresh = edge_full_report_rel in {str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh}
    edge_v2_report_rel = "reports/high_transfer_edge_contract_v2_private_residual_curriculum_code_residual_curriculum.json"
    edge_v2_is_fresh = edge_v2_report_rel in {str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh}
    if typed_is_fresh and not (
        typed_closure["allows_public_recalibration"]
        or broad_private_closure["allows_public_recalibration"]
        or edge_contract_closure["allows_public_recalibration"]
    ):
        return {
            "ready_for_recalibration": False,
            "reason": (
                "typed_interface_private_closure_required_before_public_recalibration"
                if not typed_closure["closure_current"]
                else "typed_interface_private_closure_yellow_blocks_public_recalibration"
            ),
            "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/"),
            "latest_private_pressure_mtime": latest_mtime,
            "last_four_card_calibration_mtime": calibration_mtime,
            "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
            "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
            "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
            "typed_interface_private_closure_state": typed_closure,
            "private_pressure_private_closure_state": broad_private_closure,
            "edge_contract_private_closure_state": edge_contract_closure,
            "edge_case_full_body_state": edge_full_body,
            "required_private_closure_before_public": True,
            "rotation_epoch": None,
        }
    if edge_is_fresh and not (
        broad_private_closure["allows_public_recalibration"]
        or edge_contract_closure["allows_public_recalibration"]
    ):
        return {
            "ready_for_recalibration": False,
            "reason": (
                "edge_contract_private_closure_required_before_public_recalibration"
                if not edge_contract_closure["closure_current"]
                else "edge_contract_private_closure_yellow_blocks_public_recalibration"
            ),
            "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/"),
            "latest_private_pressure_mtime": latest_mtime,
            "last_four_card_calibration_mtime": calibration_mtime,
            "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
            "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
            "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
            "typed_interface_private_closure_state": typed_closure,
            "private_pressure_private_closure_state": broad_private_closure,
            "edge_contract_private_closure_state": edge_contract_closure,
            "edge_case_full_body_state": edge_full_body,
            "required_private_closure_before_public": True,
            "rotation_epoch": None,
        }
    if (
        edge_full_is_fresh
        and edge_full_body["blocks_public_recalibration"]
        and not broad_private_closure["allows_public_recalibration"]
    ):
        return {
            "ready_for_recalibration": False,
            "reason": (
                "edge_case_full_body_private_closure_required_before_public_recalibration"
                if not edge_full_body["closure_current"]
                else "edge_case_full_body_private_closure_yellow_blocks_public_recalibration"
            ),
            "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/"),
            "latest_private_pressure_mtime": latest_mtime,
            "last_four_card_calibration_mtime": calibration_mtime,
            "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
            "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
            "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
            "typed_interface_private_closure_state": typed_closure,
            "private_pressure_private_closure_state": broad_private_closure,
            "edge_contract_private_closure_state": edge_contract_closure,
            "edge_case_full_body_state": edge_full_body,
            "edge_contract_v2_state": edge_v2,
            "required_private_closure_before_public": True,
            "rotation_epoch": None,
        }
    if (
        edge_v2_is_fresh
        and edge_v2["blocks_public_recalibration"]
        and not broad_private_closure["allows_public_recalibration"]
    ):
        return {
            "ready_for_recalibration": False,
            "reason": (
                "edge_contract_v2_private_closure_required_before_public_recalibration"
                if not edge_v2["closure_current"]
                else "edge_contract_v2_private_closure_yellow_blocks_public_recalibration"
            ),
            "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/"),
            "latest_private_pressure_mtime": latest_mtime,
            "last_four_card_calibration_mtime": calibration_mtime,
            "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
            "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
            "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
            "typed_interface_private_closure_state": typed_closure,
            "private_pressure_private_closure_state": broad_private_closure,
            "edge_contract_private_closure_state": edge_contract_closure,
            "edge_case_full_body_state": edge_full_body,
            "edge_contract_v2_state": edge_v2,
            "required_private_closure_before_public": True,
            "rotation_epoch": None,
        }
    if ready and not closure_supports_recalibration:
        return {
            "ready_for_recalibration": False,
            "reason": (
                "private_pressure_private_closure_required_before_public_recalibration"
                if not broad_private_closure["closure_current"]
                else "private_pressure_private_closure_yellow_blocks_public_recalibration"
            ),
            "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/"),
            "latest_private_pressure_mtime": latest_mtime,
            "last_four_card_calibration_mtime": calibration_mtime,
            "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
            "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
            "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
            "typed_interface_private_closure_state": typed_closure,
            "private_pressure_private_closure_state": broad_private_closure,
            "edge_contract_private_closure_state": edge_contract_closure,
            "edge_case_full_body_state": edge_full_body,
            "edge_contract_v2_state": edge_v2,
            "required_private_closure_before_public": True,
            "rotation_epoch": None,
        }
    decoder_source_mtime = decoder_relevant_source_mtime()
    decoder_source_changed_after_calibration = bool(
        decoder_source_mtime
        and (not calibration_mtime or decoder_source_mtime > calibration_mtime)
    )
    decoder_v2_ablation_gate = decoder_v2_private_ablation_gate_state()
    if ready and not decoder_v2_ablation_gate["ready_for_public_calibration"]:
        return {
            "ready_for_recalibration": False,
            "reason": "decoder_v2_private_ablation_gate_required_before_public_recalibration",
            "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/"),
            "latest_private_pressure_mtime": latest_mtime,
            "last_four_card_calibration_mtime": calibration_mtime,
            "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
            "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
            "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
            "typed_interface_private_closure_state": typed_closure,
            "private_pressure_private_closure_state": broad_private_closure,
            "edge_contract_private_closure_state": edge_contract_closure,
            "edge_case_full_body_state": edge_full_body,
            "edge_contract_v2_state": edge_v2,
            "decoder_v2_private_ablation_gate": decoder_v2_ablation_gate,
            "private_pressure_closure_after_calibration": broad_closure_after_calibration,
            "edge_contract_closure_after_calibration": edge_closure_after_calibration,
            "edge_case_full_body_closure_after_calibration": edge_full_closure_after_calibration,
            "edge_contract_v2_closure_after_calibration": edge_v2_closure_after_calibration,
            "decoder_relevant_source_mtime": decoder_source_mtime or None,
            "decoder_source_changed_after_calibration": decoder_source_changed_after_calibration,
            "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
            "rotation_epoch": None,
        }
    if (
        ready
        and type_shape_receiver_gate["teacher_spec_present"]
        and not type_shape_receiver_gate["ready_for_public_calibration"]
    ):
        return {
            "ready_for_recalibration": False,
            "reason": "private_type_shape_receiver_ablation_required_before_public_recalibration",
            "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/"),
            "latest_private_pressure_mtime": latest_mtime,
            "last_four_card_calibration_mtime": calibration_mtime,
            "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
            "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
            "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
            "typed_interface_private_closure_state": typed_closure,
            "private_pressure_private_closure_state": broad_private_closure,
            "edge_contract_private_closure_state": edge_contract_closure,
            "edge_case_full_body_state": edge_full_body,
            "edge_contract_v2_state": edge_v2,
            "decoder_v2_private_ablation_gate": decoder_v2_ablation_gate,
            "private_type_shape_receiver_ablation_gate": type_shape_receiver_gate,
            "private_pressure_closure_after_calibration": broad_closure_after_calibration,
            "edge_contract_closure_after_calibration": edge_closure_after_calibration,
            "edge_case_full_body_closure_after_calibration": edge_full_closure_after_calibration,
            "edge_contract_v2_closure_after_calibration": edge_v2_closure_after_calibration,
            "decoder_relevant_source_mtime": decoder_source_mtime or None,
            "decoder_source_changed_after_calibration": decoder_source_changed_after_calibration,
            "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
            "rotation_epoch": None,
        }
    edge_v2_public_limit = edge_contract_v2_public_calibration_limit_state(calibration_mtime)
    if ready and edge_v2_public_limit["limit_reached"] and not decoder_source_changed_after_calibration:
        if type_shape_receiver_gate_after_calibration:
            edge_v2_public_limit = {
                **edge_v2_public_limit,
                "limit_reached": False,
                "superseded_by_private_type_shape_receiver_gate": True,
            }
        else:
            return {
                "ready_for_recalibration": False,
                "reason": "post_edge_contract_v2_public_calibration_limit_reached",
                "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/"),
                "latest_private_pressure_mtime": latest_mtime,
                "last_four_card_calibration_mtime": calibration_mtime,
                "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
                "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
                "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
                "typed_interface_private_closure_state": typed_closure,
                "private_pressure_private_closure_state": broad_private_closure,
                "edge_contract_private_closure_state": edge_contract_closure,
                "edge_case_full_body_state": edge_full_body,
                "edge_contract_v2_state": edge_v2,
                "edge_contract_v2_public_calibration_limit": edge_v2_public_limit,
                "private_pressure_closure_after_calibration": broad_closure_after_calibration,
                "edge_contract_closure_after_calibration": edge_closure_after_calibration,
                "edge_case_full_body_closure_after_calibration": edge_full_closure_after_calibration,
                "edge_contract_v2_closure_after_calibration": edge_v2_closure_after_calibration,
                "private_type_shape_receiver_gate_after_calibration": type_shape_receiver_gate_after_calibration,
                "private_type_shape_receiver_ablation_gate": type_shape_receiver_gate,
                "decoder_relevant_source_mtime": decoder_source_mtime or None,
                "decoder_source_changed_after_calibration": decoder_source_changed_after_calibration,
                "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
                "rotation_epoch": None,
            }
    if (
        ready
        and calibration_mtime
        and not decoder_source_changed_after_calibration
        and not closure_supports_recalibration
    ):
        return {
            "ready_for_recalibration": False,
            "reason": "public_recalibration_requires_decoder_or_generator_source_change_after_last_four_card_calibration",
            "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/"),
            "latest_private_pressure_mtime": latest_mtime,
            "last_four_card_calibration_mtime": calibration_mtime,
            "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
            "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
            "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
            "typed_interface_private_closure_state": typed_closure,
            "private_pressure_private_closure_state": broad_private_closure,
            "edge_contract_private_closure_state": edge_contract_closure,
            "private_pressure_closure_after_calibration": broad_closure_after_calibration,
            "edge_contract_closure_after_calibration": edge_closure_after_calibration,
            "edge_case_full_body_closure_after_calibration": edge_full_closure_after_calibration,
            "edge_contract_v2_closure_after_calibration": edge_v2_closure_after_calibration,
            "decoder_relevant_source_mtime": decoder_source_mtime or None,
            "decoder_source_changed_after_calibration": decoder_source_changed_after_calibration,
            "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
            "rotation_epoch": None,
        }
    payload = {
        "latest": str(latest.relative_to(ROOT)).replace("\\", "/"),
        "latest_mtime": latest_mtime,
        "calibration_mtime": calibration_mtime,
        "reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in existing],
        "fresh": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
        "stale": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
        "missing": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
        "decoder_source_mtime": decoder_source_mtime,
        "decoder_source_changed_after_calibration": decoder_source_changed_after_calibration,
        "private_pressure_closure_mtime": broad_closure_mtime,
        "private_pressure_closure_after_calibration": broad_closure_after_calibration,
        "edge_contract_closure_mtime": edge_closure_mtime,
        "edge_contract_closure_after_calibration": edge_closure_after_calibration,
        "edge_case_full_body_closure_mtime": edge_full_mtime,
        "edge_case_full_body_closure_after_calibration": edge_full_closure_after_calibration,
        "edge_contract_v2_closure_mtime": edge_v2_mtime,
        "edge_contract_v2_closure_after_calibration": edge_v2_closure_after_calibration,
        "private_type_shape_receiver_gate_mtime": type_shape_receiver_gate_mtime,
        "private_type_shape_receiver_gate_after_calibration": type_shape_receiver_gate_after_calibration,
        "edge_contract_v2_public_calibration_limit": edge_v2_public_limit,
        "decoder_v2_private_ablation_gate": decoder_v2_ablation_gate,
        "private_type_shape_receiver_ablation_gate": type_shape_receiver_gate,
    }
    return {
        "ready_for_recalibration": ready,
        "reason": (
            "private_pressure_closure_newer_than_last_four_card_calibration"
            if broad_closure_after_calibration
            else
            "edge_contract_closure_newer_than_last_four_card_calibration"
            if edge_closure_after_calibration
            else
            "edge_case_full_body_closure_newer_than_last_four_card_calibration"
            if edge_full_closure_after_calibration
            else
            "edge_contract_v2_closure_newer_than_last_four_card_calibration"
            if edge_v2_closure_after_calibration
            else
            "private_type_shape_receiver_ablation_newer_than_last_four_card_calibration"
            if type_shape_receiver_gate_after_calibration
            else
            "private_pressure_newer_than_last_four_card_calibration"
            if ready
            else "all_private_pressure_reports_already_calibrated"
            if stale
            else "four_card_calibration_newer_than_private_pressure"
        ),
        "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/"),
        "latest_private_pressure_mtime": latest_mtime,
        "last_four_card_calibration_mtime": calibration_mtime,
        "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
        "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
        "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
        "typed_interface_private_closure_state": typed_closure,
        "private_pressure_private_closure_state": broad_private_closure,
        "edge_contract_private_closure_state": edge_contract_closure,
        "edge_case_full_body_state": edge_full_body,
        "edge_contract_v2_state": edge_v2,
        "private_pressure_closure_mtime": broad_closure_mtime or None,
        "private_pressure_closure_after_calibration": broad_closure_after_calibration,
        "edge_contract_closure_mtime": edge_closure_mtime or None,
        "edge_contract_closure_after_calibration": edge_closure_after_calibration,
        "edge_case_full_body_closure_mtime": edge_full_mtime or None,
        "edge_case_full_body_closure_after_calibration": edge_full_closure_after_calibration,
        "edge_contract_v2_closure_mtime": edge_v2_mtime or None,
        "edge_contract_v2_closure_after_calibration": edge_v2_closure_after_calibration,
        "private_type_shape_receiver_gate_mtime": type_shape_receiver_gate_mtime or None,
        "private_type_shape_receiver_gate_after_calibration": type_shape_receiver_gate_after_calibration,
        "decoder_v2_private_ablation_gate": decoder_v2_ablation_gate,
        "private_type_shape_receiver_ablation_gate": type_shape_receiver_gate,
        "decoder_relevant_source_mtime": decoder_source_mtime or None,
        "decoder_source_changed_after_calibration": decoder_source_changed_after_calibration,
        "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
        "rotation_epoch": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        if ready
        else None,
    }


def private_pressure_private_closure_state() -> dict[str, Any]:
    closure_candidates = [
        path
        for path in (*BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_CLOSURE_REPORTS, BROAD_PRIVATE_PRESSURE_CLOSURE_REPORT)
        if path.exists()
    ]
    closure_report = (
        max(closure_candidates, key=lambda path: path.stat().st_mtime)
        if closure_candidates
        else BROAD_PRIVATE_PRESSURE_TRAIN_ONCE_CLOSURE_REPORT
    )
    existing = [path for path in PRIVATE_PRESSURE_REPORTS if path.exists()]
    missing = [path for path in PRIVATE_PRESSURE_REPORTS if not path.exists()]
    missing_required = [path for path in PRIVATE_PRESSURE_REQUIRED_REPORTS if not path.exists()]
    closure_mtime = file_mtime(closure_report)
    closure = read_json(closure_report, {})
    closure_consumed_plan_ir_code_lm = closure_consumed_train_path(
        closure, DECODER_PLAN_IR_CODE_LM_TRAIN_JSONL
    ) or train_once_closure_consumes_canonical_private_rows(closure)
    edge_obligation = read_json(EDGE_OBLIGATION_PRIVATE_PRESSURE_REPORT, {})
    edge_obligation_mtime = file_mtime(EDGE_OBLIGATION_PRIVATE_PRESSURE_REPORT)
    edge_obligation_ready = bool(edge_obligation.get("ready_for_public_calibration"))
    closure_embeds_edge_obligation_ready = bool(
        get_path(closure, ["summary", "edge_obligation_decode_gate_ready"], False)
        or get_path(closure, ["edge_obligation_decode_gate", "ready_for_public_calibration"], False)
    )
    edge_obligation_current = bool(
        edge_obligation_ready
        and (edge_obligation_mtime >= closure_mtime or closure_embeds_edge_obligation_ready)
    )
    latest_mtime = max((path.stat().st_mtime for path in existing), default=0.0)
    latest = max(existing, key=lambda path: path.stat().st_mtime) if existing else None
    decoder_source_mtime = decoder_relevant_source_mtime()
    fresh = [path for path in existing if path.stat().st_mtime > closure_mtime]
    stale = [path for path in existing if path.stat().st_mtime <= closure_mtime]
    decoder_source_changed_after_closure = bool(
        decoder_source_mtime
        and (not closure_mtime or decoder_source_mtime > closure_mtime)
    )
    closure_public_gate_usable = private_closure_public_gate_usable(closure)
    closure_current = bool(
        existing
        and not missing_required
        and latest_mtime
        and closure_mtime >= latest_mtime
        and not decoder_source_changed_after_closure
        and closure.get("trigger_state") in {"GREEN", "YELLOW"}
        and closure.get("run_status") == "completed"
        and closure_consumed_plan_ir_code_lm
        and edge_obligation_ready
        and edge_obligation_current
        and closure_public_gate_usable
    )
    allows_public_recalibration = bool(closure_current)
    needs_private_closure = bool(existing and not missing_required and not closure_current)
    if missing_required:
        reason = "missing_required_private_pressure_reports"
    elif decoder_source_changed_after_closure:
        reason = "decoder_or_generator_source_newer_than_private_closure"
    elif closure.get("run_status") == "completed" and not edge_obligation_ready:
        reason = "edge_obligation_decode_gate_required_for_private_pressure_closure"
    elif closure.get("run_status") == "completed" and not edge_obligation_current:
        reason = "edge_obligation_decode_gate_stale_for_private_pressure_closure"
    elif closure.get("run_status") == "completed" and not closure_consumed_plan_ir_code_lm:
        reason = "private_closure_missing_decoder_plan_ir_code_lm_rows"
    elif closure_current:
        reason = "private_pressure_private_closure_current"
    elif fresh:
        reason = "fresh_private_pressure_newer_than_private_closure"
    elif not existing:
        reason = "missing_private_pressure_reports"
    else:
        reason = "private_pressure_private_closure_not_current"
    payload = {
        "latest_mtime": latest_mtime,
        "closure_mtime": closure_mtime,
        "fresh": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
        "missing_required": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing_required],
        "closure_trigger_state": closure.get("trigger_state"),
        "closure_run_status": closure.get("run_status"),
        "decoder_relevant_source_mtime": decoder_source_mtime,
        "decoder_source_changed_after_closure": decoder_source_changed_after_closure,
        "edge_obligation_report": str(EDGE_OBLIGATION_PRIVATE_PRESSURE_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "edge_obligation_mtime": edge_obligation_mtime or None,
        "edge_obligation_ready": edge_obligation_ready,
        "edge_obligation_current": edge_obligation_current,
        "closure_embeds_edge_obligation_ready": closure_embeds_edge_obligation_ready,
        "closure_consumed_decoder_plan_ir_code_lm": closure_consumed_plan_ir_code_lm,
        "closure_public_gate_usable": closure_public_gate_usable,
    }
    return {
        "closure_report": str(closure_report.relative_to(ROOT)).replace("\\", "/"),
        "closure_report_exists": closure_report.exists(),
        "closure_report_mtime": closure_mtime or None,
        "closure_trigger_state": closure.get("trigger_state"),
        "closure_run_status": closure.get("run_status"),
        "closure_current": closure_current,
        "allows_public_recalibration": allows_public_recalibration,
        "needs_private_closure": needs_private_closure,
        "reason": reason,
        "latest_private_pressure_report": str(latest.relative_to(ROOT)).replace("\\", "/") if latest else None,
        "latest_private_pressure_mtime": latest_mtime or None,
        "decoder_relevant_source_mtime": decoder_source_mtime or None,
        "decoder_source_changed_after_closure": decoder_source_changed_after_closure,
        "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
        "fresh_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in fresh],
        "stale_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stale],
        "missing_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing],
        "missing_required_private_pressure_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in missing_required],
        "closure_consumed_decoder_plan_ir_code_lm": closure_consumed_plan_ir_code_lm,
        "edge_obligation_current": edge_obligation_current,
        "closure_embeds_edge_obligation_ready": closure_embeds_edge_obligation_ready,
        "closure_public_gate_usable": closure_public_gate_usable,
        "rotation_epoch": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        if needs_private_closure
        else None,
    }


def private_closure_public_gate_usable(closure: dict[str, Any]) -> bool:
    """Allow completed private closures with real private lift to feed ablation.

    A closure can be YELLOW for diagnostic warnings, such as one STS repair
    regression or a newly introduced private row label. Public calibration is
    still blocked by the Decoder V2 ablation gate, so this only prevents the
    board from rerunning the same private closure instead of verifying it.
    """

    if closure.get("run_status") != "completed":
        return False
    if closure.get("trigger_state") not in {"GREEN", "YELLOW"}:
        return False
    if closure.get("hard_operational_failures"):
        return False
    summary = closure.get("summary") if isinstance(closure.get("summary"), dict) else {}
    if bool(summary.get("train_once_checkpoint_fanout")):
        return train_once_closure_public_gate_usable(closure)
    private_delta = safe_float(summary.get("private_pass_rate_delta"))
    sts_delta = safe_float(summary.get("private_sts_repair_pass_rate_delta"))
    sts_regressions = int(summary.get("private_sts_repair_task_level_regressions") or 0)
    high_transfer_rows = int(summary.get("high_transfer_private_train_task_count") or 0)
    return bool(
        private_delta >= 0.05
        and sts_delta >= 0.0
        and sts_regressions <= 1
        and high_transfer_rows > 0
        and closure_consumed_train_path(closure, DECODER_PLAN_IR_CODE_LM_TRAIN_JSONL)
    )


def train_once_closure_public_gate_usable(closure: dict[str, Any]) -> bool:
    summary = closure.get("summary") if isinstance(closure.get("summary"), dict) else {}
    private_input_fresh = bool(get_path(summary, ["private_input_freshness", "fresh"], False))
    return bool(
        closure.get("run_status") == "completed"
        and closure.get("trigger_state") in {"GREEN", "YELLOW"}
        and bool(summary.get("train_once_checkpoint_fanout"))
        and not bool(summary.get("repeated_training_per_candidate_shard"))
        and bool(summary.get("checkpoint_cuda_readout_used"))
        and int(summary.get("private_candidate_count") or 0) > 0
        and int(summary.get("private_token_level_candidate_count") or 0) > 0
        and int(summary.get("high_transfer_private_train_task_count") or 0) > 0
        and private_input_fresh
        and train_once_closure_consumes_canonical_private_rows(closure)
    )


def train_once_closure_consumes_canonical_private_rows(closure: dict[str, Any]) -> bool:
    summary = closure.get("summary") if isinstance(closure.get("summary"), dict) else {}
    if not bool(summary.get("train_once_checkpoint_fanout")):
        return False
    raw_paths = summary.get("high_transfer_private_train_jsonl")
    if not isinstance(raw_paths, list):
        return False
    observed = {normalize_path_for_compare(path) for path in raw_paths}
    required = normalize_path_for_compare(DECODER_PLAN_IR_CODE_LM_TRAIN_JSONL)
    return required in observed


def closure_consumed_train_path(closure: dict[str, Any], required_path: Path) -> bool:
    summary = closure.get("summary") if isinstance(closure.get("summary"), dict) else {}
    raw_paths = summary.get("high_transfer_private_train_jsonl")
    observed: list[str] = []
    if isinstance(raw_paths, list):
        observed.extend(str(item) for item in raw_paths)
    elif isinstance(raw_paths, str):
        observed.extend(chunk.strip() for chunk in raw_paths.replace(",", ";").split(";") if chunk.strip())
    normalized_required = normalize_path_for_compare(required_path)
    return any(normalize_path_for_compare(path) == normalized_required for path in observed)


def normalize_path_for_compare(path: str | Path) -> str:
    return str(path).replace("\\", "/").lower()


def decoder_plan_ir_code_lm_adapter_state() -> dict[str, Any]:
    plan_report = DECODER_PLAN_IR_REPORT
    adapter_report = DECODER_PLAN_IR_CODE_LM_ADAPTER_REPORT
    rows_path = DECODER_PLAN_IR_CODE_LM_TRAIN_JSONL
    plan = read_json(plan_report, {})
    adapter = read_json(adapter_report, {})
    plan_mtime = file_mtime(plan_report)
    adapter_mtime = file_mtime(adapter_report)
    rows_mtime = file_mtime(rows_path)
    adapter_summary = adapter.get("summary") if isinstance(adapter.get("summary"), dict) else {}
    adapter_rows = int(adapter_summary.get("code_lm_row_count") or adapter_summary.get("joined_row_count") or 0)
    adapter_contract_rate = float(adapter_summary.get("contract_row_rate") or 0.0)
    adapter_leak_flags = int(adapter_summary.get("public_leak_flag_count") or 0)
    plan_ready = bool(plan_report.exists() and plan.get("trigger_state") == "GREEN")
    adapter_current = bool(
        plan_ready
        and adapter_report.exists()
        and rows_path.exists()
        and adapter_mtime >= plan_mtime
        and rows_mtime >= plan_mtime
        and adapter.get("trigger_state") == "GREEN"
        and adapter_rows >= 1000
        and adapter_contract_rate >= 0.98
        and adapter_leak_flags == 0
    )
    needs_adapter = bool(plan_ready and not adapter_current)
    if not plan_report.exists():
        reason = "missing_decoder_plan_ir_private_pressure_report"
    elif not plan_ready:
        reason = "decoder_plan_ir_private_pressure_not_green"
    elif not adapter_report.exists() or not rows_path.exists():
        reason = "missing_decoder_plan_ir_code_lm_adapter"
    elif adapter_mtime < plan_mtime or rows_mtime < plan_mtime:
        reason = "decoder_plan_ir_code_lm_adapter_stale_for_plan_ir"
    elif adapter.get("trigger_state") != "GREEN":
        reason = "decoder_plan_ir_code_lm_adapter_not_green"
    elif adapter_rows < 1000:
        reason = "decoder_plan_ir_code_lm_rows_below_floor"
    elif adapter_contract_rate < 0.98:
        reason = "decoder_plan_ir_contract_rate_below_floor"
    elif adapter_leak_flags:
        reason = "decoder_plan_ir_public_leak_flags_present"
    else:
        reason = "decoder_plan_ir_code_lm_adapter_current"
    payload = {
        "plan_report_mtime": plan_mtime or None,
        "adapter_report_mtime": adapter_mtime or None,
        "rows_mtime": rows_mtime or None,
        "adapter_rows": adapter_rows,
        "adapter_contract_rate": adapter_contract_rate,
        "adapter_public_leak_flag_count": adapter_leak_flags,
        "adapter_trigger_state": adapter.get("trigger_state"),
        "reason": reason,
    }
    return {
        "plan_report": str(plan_report.relative_to(ROOT)).replace("\\", "/"),
        "adapter_report": str(adapter_report.relative_to(ROOT)).replace("\\", "/"),
        "rows_path": str(rows_path).replace("\\", "/"),
        "plan_report_exists": plan_report.exists(),
        "adapter_report_exists": adapter_report.exists(),
        "rows_path_exists": rows_path.exists(),
        "plan_report_mtime": plan_mtime or None,
        "adapter_report_mtime": adapter_mtime or None,
        "rows_mtime": rows_mtime or None,
        "adapter_current": adapter_current,
        "needs_adapter": needs_adapter,
        "reason": reason,
        "adapter_rows": adapter_rows,
        "adapter_contract_rate": adapter_contract_rate,
        "adapter_public_leak_flag_count": adapter_leak_flags,
        "rotation_epoch": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        if needs_adapter
        else None,
    }


def edge_contract_private_closure_state() -> dict[str, Any]:
    pressure_report = EDGE_CONTRACT_PRESSURE_REPORT
    closure_report = EDGE_CONTRACT_CLOSURE_REPORT
    pressure_mtime = file_mtime(pressure_report)
    closure_mtime = file_mtime(closure_report)
    closure = read_json(closure_report, {})
    decoder_source_mtime = decoder_relevant_source_mtime()
    decoder_source_changed_after_closure = bool(
        decoder_source_mtime
        and (not closure_mtime or decoder_source_mtime > closure_mtime)
    )
    closure_current = bool(
        pressure_mtime
        and closure_mtime >= pressure_mtime
        and not decoder_source_changed_after_closure
        and closure.get("trigger_state") != "RED"
        and closure.get("run_status") == "completed"
    )
    allows_public_recalibration = bool(closure_current and closure.get("trigger_state") == "GREEN")
    needs_private_closure = bool(pressure_report.exists() and not closure_current)
    if not pressure_report.exists():
        reason = "missing_edge_contract_pressure_report"
    elif decoder_source_changed_after_closure:
        reason = "decoder_or_generator_source_newer_than_edge_contract_closure"
    elif closure_current:
        reason = "edge_contract_private_closure_current"
    elif closure_mtime and pressure_mtime > closure_mtime:
        reason = "fresh_edge_contract_pressure_newer_than_private_closure"
    else:
        reason = "edge_contract_private_closure_not_current"
    payload = {
        "edge_contract_pressure_mtime": pressure_mtime,
        "closure_mtime": closure_mtime,
        "closure_trigger_state": closure.get("trigger_state"),
        "closure_run_status": closure.get("run_status"),
        "decoder_relevant_source_mtime": decoder_source_mtime,
        "decoder_source_changed_after_closure": decoder_source_changed_after_closure,
    }
    return {
        "edge_contract_pressure_report": str(pressure_report.relative_to(ROOT)).replace("\\", "/"),
        "edge_contract_pressure_exists": pressure_report.exists(),
        "edge_contract_pressure_mtime": pressure_mtime or None,
        "closure_report": str(closure_report.relative_to(ROOT)).replace("\\", "/"),
        "closure_report_exists": closure_report.exists(),
        "closure_report_mtime": closure_mtime or None,
        "closure_trigger_state": closure.get("trigger_state"),
        "closure_run_status": closure.get("run_status"),
        "closure_current": closure_current,
        "allows_public_recalibration": allows_public_recalibration,
        "needs_private_closure": needs_private_closure,
        "reason": reason,
        "decoder_relevant_source_mtime": decoder_source_mtime or None,
        "decoder_source_changed_after_closure": decoder_source_changed_after_closure,
        "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
        "rotation_epoch": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        if needs_private_closure
        else None,
    }


def typed_interface_private_closure_state() -> dict[str, Any]:
    typed_report = REPORTS / "high_transfer_typed_interface_skeleton_code_residual_curriculum.json"
    closure_report = REPORTS / "code_lm_closure_typed_interface_private.json"
    typed_mtime = file_mtime(typed_report)
    closure_mtime = file_mtime(closure_report)
    closure = read_json(closure_report, {})
    closure_current = bool(
        typed_mtime
        and closure_mtime >= typed_mtime
        and closure.get("trigger_state") != "RED"
        and closure.get("run_status") == "completed"
    )
    allows_public_recalibration = bool(closure_current and closure.get("trigger_state") == "GREEN")
    return {
        "typed_private_pressure_report": str(typed_report.relative_to(ROOT)).replace("\\", "/"),
        "typed_private_pressure_exists": typed_report.exists(),
        "typed_private_pressure_mtime": typed_mtime or None,
        "closure_report": str(closure_report.relative_to(ROOT)).replace("\\", "/"),
        "closure_report_exists": closure_report.exists(),
        "closure_report_mtime": closure_mtime or None,
        "closure_trigger_state": closure.get("trigger_state"),
        "closure_run_status": closure.get("run_status"),
        "closure_current": closure_current,
        "allows_public_recalibration": allows_public_recalibration,
        "needs_private_closure": bool(typed_mtime and not closure_current),
    }


def concept_private_pressure_state(concept: str) -> dict[str, Any]:
    safe = concept.replace("-", "_")
    report = REPORTS / f"high_transfer_{safe}_code_residual_curriculum.json"
    calibration_mtime = four_card_calibration_mtime()
    if concept == "type_contract_diagnostic":
        return {
            "waiting_for_recalibration": False,
            "reason": "type_contract_diagnostic_uses_explicit_feedback_lifecycle",
            "report": str(report.relative_to(ROOT)).replace("\\", "/"),
            "report_mtime": report.stat().st_mtime if report.exists() else None,
            "calibration_mtime": calibration_mtime or None,
        }
    if not report.exists():
        return {
            "waiting_for_recalibration": False,
            "reason": "concept_private_pressure_missing",
            "report": str(report.relative_to(ROOT)).replace("\\", "/"),
            "report_mtime": None,
            "calibration_mtime": calibration_mtime or None,
        }
    report_mtime = report.stat().st_mtime
    waiting = report_mtime > calibration_mtime
    return {
        "waiting_for_recalibration": waiting,
        "reason": "concept_private_pressure_newer_than_last_calibration"
        if waiting
        else "last_calibration_newer_than_concept_private_pressure",
        "report": str(report.relative_to(ROOT)).replace("\\", "/"),
        "report_mtime": report_mtime,
        "calibration_mtime": calibration_mtime,
    }


