"""Read broad-transfer closure residuals before choosing the next intervention.

This script intentionally separates operational faults from model-quality
walls. A missing public report, in-progress Code LM report, or stale lock is not
semantic evidence and should be fixed before the teacher is asked for an
architecture diagnosis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_FLOOR_RECOVERY_REPORT = "reports/broad_public_code_transfer_floor_recovery.json"
DEFAULT_CURRENT_CALIBRATION_REPORT = (
    "reports/real_code_benchmark_graduation_private_pressure_private_recovery_train_once_fanout_v1_public_calibration.json"
)
DEFAULT_CURRENT_CALIBRATION_TRACE = (
    "reports/real_code_benchmark_traces_private_pressure_private_recovery_train_once_fanout_v1_public_calibration.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-report", default="")
    parser.add_argument("--floor-recovery-report", default=DEFAULT_FLOOR_RECOVERY_REPORT)
    parser.add_argument("--calibration-report", default=DEFAULT_CURRENT_CALIBRATION_REPORT)
    parser.add_argument("--calibration-trace", default=DEFAULT_CURRENT_CALIBRATION_TRACE)
    parser.add_argument("--out", default="reports/broad_transfer_residual_reader.json")
    parser.add_argument("--markdown-out", default="reports/broad_transfer_residual_reader.md")
    args = parser.parse_args()

    closure_path = resolve(args.closure_report) if args.closure_report else latest_closure_report_path()
    floor_recovery_path = resolve(args.floor_recovery_report)
    calibration_report_path = resolve(args.calibration_report)
    calibration_trace_path = resolve(args.calibration_trace)

    closure = read_json(closure_path, {})
    outputs = closure.get("outputs") if isinstance(closure.get("outputs"), dict) else {}
    code_lm = read_json(resolve(str(outputs.get("code_lm_report") or "")), {})
    rust = read_json(resolve(str(outputs.get("rust_report") or "")), {})
    closure_public_report = read_json(resolve(str(outputs.get("public_report") or "")), {})
    closure_public_trace = read_jsonl(resolve(str(outputs.get("public_trace") or "")))
    floor_recovery = read_json(floor_recovery_path, {})
    calibration_report = read_json(calibration_report_path, {})
    calibration_trace = read_jsonl(calibration_trace_path)
    private_eval = code_lm.get("private_eval") if isinstance(code_lm.get("private_eval"), dict) else {}

    use_floor_recovery = should_prefer_floor_recovery(floor_recovery_path, closure_path, floor_recovery)
    public_report = calibration_report if use_floor_recovery and calibration_report else closure_public_report
    public_trace = calibration_trace if use_floor_recovery and calibration_trace else closure_public_trace

    closure_operational_faults = detect_operational_faults(closure, code_lm, rust, public_report)
    operational_faults = [] if use_floor_recovery and floor_recovery else closure_operational_faults
    trace_summary = summarize_public_trace(public_trace)
    floor_summary = summarize_floor_recovery(floor_recovery)
    if use_floor_recovery and floor_summary["residual_counts"]:
        trace_summary = merge_public_summaries(trace_summary, floor_summary)
    private_summary = summarize_private_eval(private_eval)
    dominant = dominant_wall(operational_faults, trace_summary, private_summary)
    recommendation = recommendation_for(dominant)

    gates = [
        gate("closure_report_loaded", bool(closure) or bool(floor_recovery), {"closure": rel(closure_path), "floor_recovery": rel(floor_recovery_path)}),
        gate("operational_faults_separated", True, operational_faults),
        gate("public_answers_absent", True, "only task ids, residual labels, counts, and report states are read"),
        gate("broad_floor_recovery_loaded", bool(floor_recovery), rel(floor_recovery_path)),
        gate("teacher_ready_only_for_model_wall", dominant["wall_type"] == "model_quality_wall", dominant),
    ]
    report = {
        "policy": "project_theseus_broad_transfer_residual_reader_v1",
        "created_utc": now(),
        "trigger_state": "YELLOW" if operational_faults or dominant["wall_type"] != "model_quality_wall" else "GREEN",
        "source": {
            "mode": "broad_floor_recovery" if use_floor_recovery else "closure_trace",
            "closure_report": rel(closure_path),
            "code_lm_report": outputs.get("code_lm_report"),
            "rust_report": outputs.get("rust_report"),
            "public_report": rel(calibration_report_path) if use_floor_recovery and calibration_report else outputs.get("public_report"),
            "public_trace": rel(calibration_trace_path) if use_floor_recovery and calibration_trace else outputs.get("public_trace"),
            "floor_recovery_report": rel(floor_recovery_path),
        },
        "summary": {
            "wall_type": dominant["wall_type"],
            "dominant_residual": dominant["dominant_residual"],
            "dominant_count": dominant["dominant_count"],
            "operational_fault_count": len(operational_faults),
            "public_failed_attempt_count": trace_summary["failed_attempt_count"],
            "private_failed_attempt_count": private_summary["failed_attempt_count"],
            "broad_public_pass_rate": floor_summary.get("broad_public_pass_rate"),
            "public_floor": floor_summary.get("public_floor"),
            "private_pressure_row_count": floor_summary.get("private_pressure_row_count"),
            "same_seed_private_semantic_lift": floor_summary.get("same_seed_private_semantic_lift"),
            "remaining_gap_explained": floor_summary.get("remaining_gap_explained"),
            "sts_flat": bool(dominant.get("sts_flat")),
            "teacher_should_run": dominant["wall_type"] == "model_quality_wall",
            "promotion_evidence": False,
        },
        "operational_faults": operational_faults,
        "stale_closure_operational_faults": closure_operational_faults if use_floor_recovery else [],
        "public_trace_summary": trace_summary,
        "floor_recovery_summary": floor_summary,
        "private_eval_summary": private_summary,
        "dominant": dominant,
        "recommendation": recommendation,
        "gates": gates,
        "score_semantics": "diagnostic routing only; public benchmark data remains calibration-only",
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0


def latest_closure_report_path() -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in REPORTS.glob("broad_transfer_closure_runner_source_*.json"):
        payload = read_json(path, {})
        outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
        if outputs.get("public_report") and outputs.get("public_trace"):
            candidates.append((safe_mtime(path), path))
    if not candidates:
        return REPORTS / "broad_transfer_closure_runner_source_evalplus.json"
    return max(candidates, key=lambda item: item[0])[1]


def should_prefer_floor_recovery(floor_path: Path, closure_path: Path, floor_recovery: dict[str, Any]) -> bool:
    if not floor_recovery:
        return False
    summary = floor_recovery.get("summary") if isinstance(floor_recovery.get("summary"), dict) else {}
    if summary.get("remaining_gap_explained") or summary.get("fresh_calibration_residual_families"):
        return True
    return safe_mtime(floor_path) >= safe_mtime(closure_path)


def safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def detect_operational_faults(
    closure: dict[str, Any],
    code_lm: dict[str, Any],
    rust: dict[str, Any],
    public_report: dict[str, Any],
) -> list[dict[str, Any]]:
    faults: list[dict[str, Any]] = []
    if not closure:
        faults.append({"fault": "closure_report_missing", "kind": "operational"})
        return faults
    if code_lm.get("run_status") == "in_progress":
        faults.append(
            {
                "fault": "code_lm_closure_in_progress_or_interrupted",
                "kind": "operational",
                "progress_stage": code_lm.get("progress_stage"),
            }
        )
    if code_lm.get("trigger_state") == "RED":
        faults.append(
            {
                "fault": "code_lm_closure_red",
                "kind": "operational",
                "hard_operational_failures": code_lm.get("hard_operational_failures", []),
            }
        )
    if rust.get("run_status") == "in_progress":
        faults.append(
            {
                "fault": "rust_symliquid_in_progress_or_interrupted",
                "kind": "operational",
                "progress_stage": rust.get("progress_stage"),
            }
        )
    if closure.get("summary", {}).get("execute") and not public_report:
        faults.append({"fault": "public_calibration_missing", "kind": "operational"})
    for row in closure.get("steps", []) if isinstance(closure.get("steps"), list) else []:
        if int(row.get("returncode") or 0) != 0 and not row.get("allow_failure"):
            faults.append(
                {
                    "fault": "runner_step_failed",
                    "kind": "operational",
                    "step": row.get("name"),
                    "error": row.get("error"),
                    "returncode": row.get("returncode"),
                }
            )
    return faults


def summarize_public_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    residuals: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    task_ids: set[str] = set()
    for row in rows:
        if row.get("event") != "real_code_candidate_test" or row.get("passed") is True:
            continue
        residual = normalize_residual(str(row.get("residual_class") or "wrong_answer"))
        residuals[residual] += 1
        origin = str(row.get("candidate_origin") or "unknown")
        origins[origin.split(":", 2)[1] if ":" in origin else origin] += 1
        task_ids.add(str(row.get("task_id") or row.get("source_task_id") or "unknown"))
    return {
        "failed_attempt_count": sum(residuals.values()),
        "residual_counts": dict(residuals),
        "failed_origin_counts": dict(origins),
        "failed_task_count": len(task_ids),
        "failed_task_hashes": sorted(short_hash(item) for item in task_ids)[:64],
    }


def summarize_floor_recovery(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    fresh_counts = summary.get("fresh_calibration_residual_families")
    matrix_counts = summary.get("dominant_residual_families")
    source_counts = fresh_counts if isinstance(fresh_counts, dict) and fresh_counts else matrix_counts
    residuals: Counter[str] = Counter()
    if isinstance(source_counts, dict):
        for key, value in source_counts.items():
            residuals[normalize_residual(str(key))] += int(value or 0)
    next_blocker = summary.get("next_blocker") if isinstance(summary.get("next_blocker"), dict) else {}
    primary = str(next_blocker.get("primary") or "")
    if primary and not residuals:
        residuals[normalize_residual(primary)] += int(sum(next_blocker.get("counts", {}).values()) or 1)
    return {
        "failed_attempt_count": sum(residuals.values()),
        "residual_counts": dict(residuals),
        "failed_origin_counts": {},
        "failed_task_count": 0,
        "failed_task_hashes": [],
        "broad_public_pass_rate": summary.get("broad_public_pass_rate"),
        "public_floor": summary.get("public_floor"),
        "private_pressure_row_count": summary.get("private_pressure_row_count"),
        "same_seed_private_semantic_lift": summary.get("same_seed_private_semantic_lift"),
        "remaining_gap_explained": summary.get("remaining_gap_explained"),
        "weak_cards": summary.get("weak_cards"),
        "next_blocker": next_blocker,
    }


def merge_public_summaries(trace_summary: dict[str, Any], floor_summary: dict[str, Any]) -> dict[str, Any]:
    residuals = Counter()
    residuals.update(trace_summary.get("residual_counts", {}))
    residuals.update(floor_summary.get("residual_counts", {}))
    origins = Counter()
    origins.update(trace_summary.get("failed_origin_counts", {}))
    origins.update(floor_summary.get("failed_origin_counts", {}))
    return {
        "failed_attempt_count": sum(residuals.values()),
        "residual_counts": dict(residuals),
        "failed_origin_counts": dict(origins),
        "failed_task_count": trace_summary.get("failed_task_count", 0),
        "failed_task_hashes": trace_summary.get("failed_task_hashes", []),
        "broad_floor_recovery_counts_included": True,
        "score_semantics": "public calibration residual labels plus broad-floor aggregate residual counts; no public tests or answers are read into training",
    }


def summarize_private_eval(private_eval: dict[str, Any]) -> dict[str, Any]:
    counts = Counter()
    families = Counter()
    for row in private_eval.get("residuals", []) if isinstance(private_eval.get("residuals"), list) else []:
        label = normalize_residual(str(row.get("concept_residual_label") or row.get("residual_class") or "wrong_answer"))
        counts[label] += 1
        families[str(row.get("category") or "unknown")] += 1
    if not counts and isinstance(private_eval.get("concept_residual_counts"), dict):
        for key, value in private_eval["concept_residual_counts"].items():
            counts[normalize_residual(str(key))] += int(value or 0)
    return {
        "failed_attempt_count": sum(counts.values()),
        "residual_counts": dict(counts),
        "family_counts": dict(families),
    }


def dominant_wall(
    operational_faults: list[dict[str, Any]],
    public_summary: dict[str, Any],
    private_summary: dict[str, Any],
) -> dict[str, Any]:
    if operational_faults:
        counter = Counter(str(row.get("fault") or "operational_fault") for row in operational_faults)
        residual, count = counter.most_common(1)[0]
        return {
            "wall_type": "operational_closure_wall",
            "dominant_residual": residual,
            "dominant_count": count,
            "sts_flat": False,
        }
    combined = Counter()
    combined.update(public_summary.get("residual_counts", {}))
    combined.update(private_summary.get("residual_counts", {}))
    if not combined:
        return {
            "wall_type": "no_residual_evidence",
            "dominant_residual": "no_failed_attempts_loaded",
            "dominant_count": 0,
            "sts_flat": True,
        }
    residual, count = combined.most_common(1)[0]
    return {
        "wall_type": "model_quality_wall",
        "dominant_residual": residual,
        "dominant_count": count,
        "sts_flat": residual in {"sts_flat", "sts_not_causal"} or public_summary.get("failed_attempt_count", 0) == 0,
    }


def recommendation_for(dominant: dict[str, Any]) -> dict[str, Any]:
    residual = dominant.get("dominant_residual")
    if dominant.get("wall_type") == "operational_closure_wall":
        return {
            "next_action": "fix_execution_boundary_then_rerun_same_card",
            "decoder_patch": "none_until_real_candidate_residuals_exist",
            "teacher": "do_not_call_teacher_for_operational_fault",
        }
    if residual in {"no_admissible_candidate", "runtime"}:
        return {
            "next_action": "patch_ast_skeleton_and_admissibility",
            "decoder_patch": "strengthen branch/loop/local return-shape skeletons",
            "teacher": "optional_after_private_eval_confirms_recurrence",
        }
    if residual in {"type_handling", "type_or_name_error", "return_shape"}:
        return {
            "next_action": "patch_type_return_shape_planner",
            "decoder_patch": "condition decoding on argument and return contracts",
            "teacher": "proposal_only_if repeated across public cards",
        }
    if residual in {"wrong_answer", "semantic_wrong_answer"}:
        return {
            "next_action": "train_private_semantic_family",
            "decoder_patch": "add residual family pressure for collection/math/string semantics",
            "teacher": "proposal_only_architecture_diagnosis_allowed",
        }
    if residual in {"edge_case", "edge_contract"}:
        return {
            "next_action": "train_private_edge_contract_and_intended_behavior_family",
            "decoder_patch": "make edge obligations alter AST plan, branch guards, loop bounds, and return contracts before token decode",
            "teacher": "proposal_only_if same_seed private edge lift stalls",
        }
    if residual in {"local_code_generation_adapter_needed", "external_dependency_missing"}:
        return {
            "next_action": "patch_private_adapter_runtime_dependency_family",
            "decoder_patch": "route dependency and adapter constraints into import guards, local fallbacks, and verifier prefilters",
            "teacher": "proposal_only_if private adapter ablation stalls",
        }
    return {
        "next_action": "read_cluster_and_choose_smallest_private_experiment",
        "decoder_patch": "residual_specific",
        "teacher": "proposal_only_after residual evidence",
    }


def normalize_residual(value: str) -> str:
    low = value.lower()
    if "edge" in low:
        return "edge_case"
    if "adapter" in low or "dependency" in low:
        return "local_code_generation_adapter_needed"
    if "algorithm" in low:
        return "algorithm_choice"
    if "no admissible" in low or "emitted no admissible" in low:
        return "no_admissible_candidate"
    if "type" in low or "name" in low:
        return "type_handling"
    if "syntax" in low or "parse" in low:
        return "syntax_or_parse"
    if "wrong" in low or "assert" in low:
        return "wrong_answer"
    if "runtime" in low:
        return "no_admissible_candidate"
    return low or "unknown"


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    rec = report.get("recommendation", {})
    source = report.get("source", {})
    return "\n".join(
        [
            "# Broad Transfer Residual Reader",
            "",
            f"- trigger_state: `{report.get('trigger_state')}`",
            f"- source_mode: `{source.get('mode')}`",
            f"- wall_type: `{summary.get('wall_type')}`",
            f"- dominant_residual: `{summary.get('dominant_residual')}`",
            f"- broad_public_pass_rate: `{summary.get('broad_public_pass_rate')}`",
            f"- private_pressure_row_count: `{summary.get('private_pressure_row_count')}`",
            f"- same_seed_private_semantic_lift: `{summary.get('same_seed_private_semantic_lift')}`",
            f"- operational_fault_count: `{summary.get('operational_fault_count')}`",
            f"- next_action: `{rec.get('next_action')}`",
            f"- teacher: `{rec.get('teacher')}`",
            "",
        ]
    )


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    if not path or str(path) == "." or not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path or str(path) == "." or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
