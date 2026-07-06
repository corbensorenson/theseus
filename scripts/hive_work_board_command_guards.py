"""Safety guards for Hive work board command routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FOUR_CARD_RECEIVER_SLUG = "source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32"
CODE_CONTRACT_PREFLIGHT_REPORT = REPORTS / "code_lm_closure_public_contract_preflight_seed23_32.json"
EXECUTION_SHAPE_NO_TEMPLATE_SMOKE_REPORT = REPORTS / "execution_shape_private_ablation_smoke.json"
PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG = "frontier_private_transfer_private_only_train_once_v1"
PRIVATE_PRESSURE_TRAIN_ONCE_LEGACY_SLUG = "private_pressure_private_recovery_train_once_fanout_v1"
DECODER_RELEVANT_SOURCES = (
    ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure.rs",
    ROOT / "scripts" / "code_lm_closure.py",
    ROOT / "scripts" / "code_residual_curriculum.py",
    ROOT / "scripts" / "type_contract_diagnostic.py",
)


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def private_pressure_private_closure_needed() -> bool:
    """Return true when Decoder V2 changed after the private closure evidence.

    The public-gate order is strict:
    fresh private pressure closure -> private ablation gate -> one public 4-card
    calibration. This guard keeps the board from running the ablation gate
    against stale closure artifacts after decoder/scheduler changes.
    """

    scheduler = read_json(REPORTS / "high_transfer_curriculum_scheduler.json", {})
    concepts = scheduler.get("concepts") if isinstance(scheduler.get("concepts"), list) else []
    for row in concepts:
        if not isinstance(row, dict):
            continue
        if str(row.get("concept") or "").lower() != "private_pressure_private_closure":
            continue
        state = get_path(row, ["evidence", "private_pressure_private_closure_state"], {})
        if isinstance(state, dict):
            return bool(state.get("needs_private_closure") or not state.get("closure_current"))

    closure_candidates = [
        REPORTS / f"code_lm_closure_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.json",
        REPORTS / f"code_lm_closure_{PRIVATE_PRESSURE_TRAIN_ONCE_LEGACY_SLUG}.json",
        REPORTS / "code_lm_closure_private_pressure_private.json",
    ]
    existing = [path for path in closure_candidates if path.exists()]
    if not existing:
        return True
    closure = max(existing, key=lambda path: path.stat().st_mtime)
    payload = read_json(closure, {})
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    private_inputs_fresh = bool(get_path(summary, ["private_input_freshness", "fresh"], True))
    complete = bool(
        (payload.get("run_status") == "completed" or payload.get("trigger_state") == "GREEN")
        and private_inputs_fresh
    )
    return (not complete) or decoder_relevant_source_mtime() > closure.stat().st_mtime

def private_type_shape_receiver_ablation_needed() -> bool:
    """Return true when teacher requested the receiver gate and it is not current."""

    scheduler = read_json(REPORTS / "high_transfer_curriculum_scheduler.json", {})
    concepts = scheduler.get("concepts") if isinstance(scheduler.get("concepts"), list) else []
    for row in concepts:
        if not isinstance(row, dict):
            continue
        if str(row.get("concept") or "").lower() != "private_type_shape_receiver_veto_ablation":
            continue
        state = get_path(row, ["evidence", "private_type_shape_receiver_ablation_state"], {})
        if isinstance(state, dict):
            return bool(state.get("needs_ablation") or not state.get("ready_for_public_calibration"))

    teacher = read_json(REPORTS / "teacher_public_transfer_residual_last.json", {})
    if "private_type_shape_receiver_veto_ablation_v1" not in json.dumps(teacher, sort_keys=True).lower():
        return False
    report_path = REPORTS / "private_type_shape_receiver_ablation.json"
    if not report_path.exists():
        return True
    report = read_json(report_path, {})
    return not bool(
        report.get("trigger_state") == "GREEN"
        and report.get("ready_for_public_calibration")
        and report.get("policy") == "project_theseus_private_type_shape_receiver_veto_ablation_v1"
    )

def post_edge_contract_v2_public_calibration_limit_reached(task: dict[str, Any]) -> bool:
    concept = str(get_path(task, ["evidence", "concept"], "") or "").lower()
    if concept != "private_pressure_four_card_recalibration":
        return False
    verifier_path = REPORTS / "edge_contract_v2_private_verifier.json"
    verifier = read_json(verifier_path, {})
    if not verifier.get("ready_for_public_calibration") or not verifier_path.exists():
        return False
    verifier_mtime = verifier_path.stat().st_mtime
    calibration_mtime = four_card_receiver_calibration_mtime()
    if not calibration_mtime or calibration_mtime <= verifier_mtime:
        return False
    receiver_path = REPORTS / "private_type_shape_receiver_ablation.json"
    receiver = read_json(receiver_path, {})
    receiver_mtime = receiver_path.stat().st_mtime if receiver_path.exists() else 0.0
    if (
        receiver.get("ready_for_public_calibration")
        and receiver.get("trigger_state") == "GREEN"
        and receiver.get("policy") == "project_theseus_private_type_shape_receiver_veto_ablation_v1"
        and receiver_mtime > calibration_mtime
    ):
        return False
    decoder_mtime = decoder_relevant_source_mtime()
    return not bool(decoder_mtime and decoder_mtime > calibration_mtime)

def private_public_calibration_guard() -> dict[str, Any]:
    """Allow exactly one public receiver calibration after a current private gate."""

    gate_path = REPORTS / "decoder_v2_private_ablation_gate.json"
    gate = read_json(gate_path, {})
    gate_mtime = gate_path.stat().st_mtime if gate_path.exists() else 0.0
    source_mtime = decoder_relevant_source_mtime()
    private_paths = [
        REPORTS / f"code_lm_closure_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.json",
        REPORTS / f"code_lm_closure_{PRIVATE_PRESSURE_TRAIN_ONCE_LEGACY_SLUG}.json",
        REPORTS / "code_lm_closure_private_pressure_private.json",
        REPORTS / "code_lm_closure_edge_contract_v2_private.json",
        REPORTS / "code_lm_closure_edge_case_full_body_private_v1.json",
        REPORTS / "code_lm_closure_edge_contract_balanced_4card_private_v2.json",
        REPORTS / "code_lm_closure_edge_contract_4card_private.json",
    ]
    private_mtimes = []
    for path in private_paths:
        if not path.exists():
            continue
        payload = read_json(path, {})
        if payload.get("run_status") == "completed" or payload.get("trigger_state") == "GREEN":
            private_mtimes.append(path.stat().st_mtime)
    latest_private_mtime = max(private_mtimes) if private_mtimes else 0.0
    calibration_mtime = four_card_receiver_calibration_mtime()
    ready = bool(gate.get("ready_for_public_calibration"))
    source_current = bool(gate_mtime and gate_mtime >= source_mtime)
    private_current = bool(gate_mtime and gate_mtime >= latest_private_mtime)
    receiver_gate_path = REPORTS / "private_type_shape_receiver_ablation.json"
    receiver_gate = read_json(receiver_gate_path, {})
    receiver_gate_mtime = receiver_gate_path.stat().st_mtime if receiver_gate_path.exists() else 0.0
    teacher = read_json(REPORTS / "teacher_public_transfer_residual_last.json", {})
    receiver_gate_required = "private_type_shape_receiver_veto_ablation_v1" in json.dumps(
        teacher,
        sort_keys=True,
    ).lower()
    receiver_gate_ready = bool(
        receiver_gate.get("ready_for_public_calibration")
        and receiver_gate.get("trigger_state") == "GREEN"
        and receiver_gate.get("policy") == "project_theseus_private_type_shape_receiver_veto_ablation_v1"
        and receiver_gate_mtime >= latest_private_mtime
    )
    public_gate_mtime = max(gate_mtime, receiver_gate_mtime if receiver_gate_required else 0.0)
    one_shot_unused = bool(not calibration_mtime or calibration_mtime <= public_gate_mtime)
    receiver_one_shot_unused = bool(
        not receiver_gate_required
        or not calibration_mtime
        or calibration_mtime <= receiver_gate_mtime
    )
    execution_shape_gate_path = REPORTS / "execution_shape_private_ablation.json"
    execution_shape_gate = read_json(execution_shape_gate_path, {})
    execution_shape_gate_summary = (
        execution_shape_gate.get("summary")
        if isinstance(execution_shape_gate.get("summary"), dict)
        else {}
    )
    execution_shape_gate_mtime = execution_shape_gate_path.stat().st_mtime if execution_shape_gate_path.exists() else 0.0
    execution_shape_gate_ready = bool(
        execution_shape_gate.get("ready_for_public_calibration")
        or execution_shape_gate.get("private_ablation_public_gate_ready")
        or execution_shape_gate_summary.get("ready_for_public_calibration")
        or execution_shape_gate_summary.get("private_ablation_public_gate_ready")
    )
    execution_shape_gate_current_diagnostic = bool(
        execution_shape_gate_mtime
        and latest_private_mtime
        and execution_shape_gate_mtime >= latest_private_mtime
        and execution_shape_gate.get("policy") == "project_theseus_execution_shape_private_ablation_v1"
        and execution_shape_gate.get("trigger_state") == "YELLOW"
        and not execution_shape_gate_ready
        and int(execution_shape_gate_summary.get("private_eval_task_count") or 0) > 0
    )
    allowed = (
        ready
        and source_current
        and private_current
        and (not receiver_gate_required or receiver_gate_ready)
        and one_shot_unused
        and receiver_one_shot_unused
        and not execution_shape_gate_current_diagnostic
    )
    blockers = []
    if not ready:
        blockers.append("decoder_v2_private_ablation_gate_not_ready")
    if not source_current:
        blockers.append("decoder_or_scheduler_source_newer_than_private_gate")
    if not private_current:
        blockers.append("private_closure_newer_than_private_gate")
    if receiver_gate_required and not receiver_gate_ready:
        blockers.append("private_type_shape_receiver_ablation_gate_not_ready")
    if not one_shot_unused:
        blockers.append("public_receiver_calibration_already_consumed_this_gate")
    if not receiver_one_shot_unused:
        blockers.append("public_receiver_calibration_already_consumed_receiver_gate")
    if execution_shape_gate_current_diagnostic:
        blockers.append("execution_shape_private_ablation_gate_not_ready")
    return {
        "allowed": allowed,
        "blockers": blockers,
        "gate_path": rel(gate_path),
        "gate_mtime": gate_mtime,
        "receiver_gate_path": rel(receiver_gate_path),
        "receiver_gate_required": receiver_gate_required,
        "receiver_gate_ready": receiver_gate_ready,
        "receiver_gate_mtime": receiver_gate_mtime,
        "execution_shape_gate_path": rel(execution_shape_gate_path),
        "execution_shape_gate_mtime": execution_shape_gate_mtime,
        "execution_shape_gate_ready": execution_shape_gate_ready,
        "execution_shape_gate_current_diagnostic": execution_shape_gate_current_diagnostic,
        "public_gate_mtime": public_gate_mtime,
        "source_mtime": source_mtime,
        "latest_private_mtime": latest_private_mtime,
        "calibration_mtime": calibration_mtime,
        "ready_for_public_calibration": ready,
        "policy": "private_gates_then_one_public_receiver_calibration_v2",
    }

def code_contract_preflight_command() -> list[str]:
    return [
        sys.executable,
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
    ]

def code_contract_preflight_guard() -> dict[str, Any]:
    report = read_json(CODE_CONTRACT_PREFLIGHT_REPORT, {})
    preflight = get_path(report, ["summary", "public_decoder_contract_preflight"], {})
    if not isinstance(preflight, dict):
        preflight = {}
    report_mtime = CODE_CONTRACT_PREFLIGHT_REPORT.stat().st_mtime if CODE_CONTRACT_PREFLIGHT_REPORT.exists() else 0.0
    source_mtime = decoder_relevant_source_mtime()
    hard_blockers = preflight.get("hard_blockers") if isinstance(preflight.get("hard_blockers"), list) else []
    varargs = int(preflight.get("varargs_task_count") or 0)
    weak_required = int(preflight.get("weak_required_construct_count") or 0)
    weak_full_body = int(preflight.get("weak_full_body_count") or 0)
    arithmetic_obligations = int(get_path(preflight, ["construct_counts", "arithmetic_formula"], 0) or 0)
    current = bool(report_mtime and source_mtime and report_mtime >= source_mtime)
    passed = bool(
        report.get("policy") == "project_theseus_code_lm_closure_preflight_v1"
        and report.get("trigger_state") == "GREEN"
        and report.get("run_status") == "completed"
        and preflight.get("passed") is True
        and varargs == 0
        and weak_required == 0
        and weak_full_body == 0
        and arithmetic_obligations > 0
        and not hard_blockers
    )
    if not CODE_CONTRACT_PREFLIGHT_REPORT.exists():
        reason = "code_contract_preflight_missing"
    elif not current:
        reason = "code_contract_preflight_stale_after_decoder_source_change"
    elif not passed:
        reason = "code_contract_preflight_failed"
    else:
        reason = "code_contract_preflight_green_current"
    return {
        "allowed": bool(passed and current),
        "reason": reason,
        "report": rel(CODE_CONTRACT_PREFLIGHT_REPORT),
        "report_mtime": report_mtime or None,
        "source_mtime": source_mtime or None,
        "varargs_task_count": varargs,
        "weak_required_construct_count": weak_required,
        "weak_full_body_count": weak_full_body,
        "arithmetic_formula_obligation_count": arithmetic_obligations,
        "hard_blockers": hard_blockers,
    }

def execution_shape_no_template_smoke_guard() -> dict[str, Any]:
    report = read_json(EXECUTION_SHAPE_NO_TEMPLATE_SMOKE_REPORT, {})
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    report_mtime = (
        EXECUTION_SHAPE_NO_TEMPLATE_SMOKE_REPORT.stat().st_mtime
        if EXECUTION_SHAPE_NO_TEMPLATE_SMOKE_REPORT.exists()
        else 0.0
    )
    source_mtime = decoder_relevant_source_mtime()
    current = bool(report_mtime and source_mtime and report_mtime >= source_mtime)
    no_templates = int(summary.get("diagnostic_template_candidate_count") or 0) == 0
    public_ready = bool(
        report.get("ready_for_public_calibration")
        or report.get("private_ablation_public_gate_ready")
        or summary.get("ready_for_public_calibration")
        or summary.get("private_ablation_public_gate_ready")
        or summary.get("learned_token_public_gate_ready")
    )
    pass_rate = float(summary.get("learned_token_decoder_pass_rate") or 0.0)
    zero_categories = [
        str(item)
        for item in (summary.get("learned_token_decoder_zero_pass_categories") or [])
        if str(item).strip()
    ]
    targeted_closure = targeted_execution_shape_zero_category_closure(
        zero_categories,
        source_mtime=source_mtime,
    )
    remediated_public_ready = bool(
        public_ready
        or (
            current
            and no_templates
            and pass_rate >= 0.70
            and zero_categories
            and targeted_closure.get("closed")
        )
    )
    hard_failures = [
        str(row.get("gate") or "")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    if not EXECUTION_SHAPE_NO_TEMPLATE_SMOKE_REPORT.exists():
        reason = "execution_shape_no_template_smoke_missing"
    elif not current:
        reason = "execution_shape_no_template_smoke_stale_after_decoder_source_change"
    elif not no_templates:
        reason = "execution_shape_no_template_smoke_template_candidates_present"
    elif not remediated_public_ready:
        reason = "execution_shape_no_template_smoke_private_gate_not_ready"
    elif not public_ready and targeted_closure.get("closed"):
        reason = "execution_shape_no_template_smoke_green_with_targeted_zero_category_closure"
    else:
        reason = "execution_shape_no_template_smoke_green_current"
    return {
        "allowed": bool(current and no_templates and remediated_public_ready),
        "reason": reason,
        "report": rel(EXECUTION_SHAPE_NO_TEMPLATE_SMOKE_REPORT),
        "report_mtime": report_mtime or None,
        "source_mtime": source_mtime or None,
        "learned_token_decoder_pass_rate": pass_rate,
        "learned_token_public_gate_ready": bool(summary.get("learned_token_public_gate_ready")),
        "private_ablation_public_gate_ready": remediated_public_ready,
        "diagnostic_template_candidate_count": int(summary.get("diagnostic_template_candidate_count") or 0),
        "zero_pass_categories": zero_categories,
        "targeted_zero_category_closure": targeted_closure,
        "hard_failed_gates": hard_failures,
    }

def targeted_execution_shape_zero_category_closure(
    zero_categories: list[str],
    *,
    source_mtime: float,
) -> dict[str, Any]:
    if not zero_categories:
        return {"closed": False, "reason": "no_zero_categories_to_close", "reports": []}
    covered: set[str] = set()
    reports: list[dict[str, Any]] = []
    for path in REPORTS.glob("execution_shape_private_ablation_*_patch_smoke.json"):
        data = read_json(path, {})
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        report_mtime = path.stat().st_mtime if path.exists() else 0.0
        filters = (
            data.get("inputs", {}).get("category_filter")
            if isinstance(data.get("inputs"), dict)
            else []
        )
        categories = [str(item) for item in (filters or []) if str(item).strip()]
        no_templates = int(summary.get("diagnostic_template_candidate_count") or 0) == 0
        ready = bool(
            summary.get("learned_token_public_gate_ready")
            or summary.get("private_ablation_public_gate_ready")
        )
        fresh = bool(report_mtime and source_mtime and report_mtime >= source_mtime)
        learned_no_admissible = float(summary.get("learned_token_decoder_no_admissible_candidate_rate") or 0.0)
        passed = bool(
            fresh
            and no_templates
            and ready
            and learned_no_admissible == 0.0
            and float(summary.get("learned_token_decoder_pass_rate") or 0.0) >= 0.70
        )
        reports.append(
            {
                "report": rel(path),
                "categories": categories,
                "fresh": fresh,
                "passed": passed,
                "learned_token_decoder_pass_rate": float(summary.get("learned_token_decoder_pass_rate") or 0.0),
                "learned_token_decoder_no_admissible_candidate_rate": learned_no_admissible,
                "diagnostic_template_candidate_count": int(summary.get("diagnostic_template_candidate_count") or 0),
            }
        )
        if passed:
            covered.update(categories)
    missing = [category for category in zero_categories if category not in covered]
    return {
        "closed": not missing,
        "reason": "targeted_zero_categories_closed" if not missing else "targeted_zero_categories_missing",
        "covered": sorted(covered),
        "missing": missing,
        "reports": reports,
    }

def code_lm_training_command_gate(command_spec: dict[str, Any]) -> dict[str, Any]:
    command = command_spec.get("command") if isinstance(command_spec.get("command"), list) else []
    command_text = " ".join(str(part) for part in command).replace("\\", "/")
    is_code_lm_training = "scripts/code_lm_closure.py" in command_text and "--preflight-only" not in command
    if not is_code_lm_training:
        return {"allowed": True, "reason": "not_code_lm_training_command"}
    preflight_guard = code_contract_preflight_guard()
    smoke_guard = execution_shape_no_template_smoke_guard()
    allowed = bool(preflight_guard.get("allowed") and smoke_guard.get("allowed"))
    blockers = []
    if not preflight_guard.get("allowed"):
        blockers.append(str(preflight_guard.get("reason") or "code_contract_preflight_not_ready"))
    if not smoke_guard.get("allowed"):
        blockers.append(str(smoke_guard.get("reason") or "execution_shape_no_template_smoke_not_ready"))
    return {
        "allowed": allowed,
        "reason": "code_lm_training_gate_ready" if allowed else "code_lm_training_blocked_by_private_no_template_gate",
        "blockers": blockers,
        "code_contract_preflight_guard": preflight_guard,
        "execution_shape_no_template_smoke_guard": smoke_guard,
    }

def execution_shape_no_template_smoke_command() -> list[str]:
    return [
        sys.executable,
        "scripts/execution_shape_private_ablation.py",
        "--seed",
        "14",
        "--train-rows",
        "80",
        "--eval-rows",
        "8",
        "--max-work-steps",
        "140000",
        "--out",
        "reports/execution_shape_private_ablation_smoke.json",
        "--markdown-out",
        "reports/execution_shape_private_ablation_smoke.md",
        "--candidate-out",
        "reports/execution_shape_private_ablation_smoke_candidates.jsonl",
        "--public-candidate-out",
        "reports/execution_shape_private_ablation_smoke_public_candidates.jsonl",
        "--checkpoint-out",
        "reports/execution_shape_private_ablation_smoke_checkpoint.json",
        "--rust-report-out",
        "reports/execution_shape_private_ablation_smoke_rust.json",
        "--curriculum-out",
        "data/private_code_curriculum/execution_shape_private_ablation_smoke_seed14.jsonl",
        "--public-manifest-out",
        "reports/execution_shape_private_ablation_smoke_visible_manifest.jsonl",
    ]

def four_card_receiver_calibration_mtime() -> float:
    def typed_edge_receiver_enabled(data: dict[str, Any]) -> bool:
        return bool(get_path(data, ["summary", "typed_edge_exec_receiver_v1_enabled"], False))

    candidates = [
        (
            REPORTS / f"broad_transfer_closure_runner_{FOUR_CARD_RECEIVER_SLUG}.json",
            lambda data: data.get("trigger_state") == "GREEN"
            and int(get_path(data, ["summary", "public_task_count"], 0) or 0) >= 128,
            True,
        ),
        (
            REPORTS / f"code_lm_closure_{FOUR_CARD_RECEIVER_SLUG}.json",
            lambda data: data.get("run_status") == "completed"
            and int(get_path(data, ["summary", "public_task_count"], 0) or 0) >= 128,
            True,
        ),
    ]
    mtimes: list[float] = []
    for path, is_complete, require_typed_edge in candidates:
        if not path.exists():
            continue
        data = read_json(path, {})
        if (
            isinstance(data, dict)
            and is_complete(data)
            and (not require_typed_edge or typed_edge_receiver_enabled(data))
        ):
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0

def decoder_relevant_source_mtime() -> float:
    mtimes = [path.stat().st_mtime for path in DECODER_RELEVANT_SOURCES if path.exists()]
    return max(mtimes) if mtimes else 0.0
