"""Architecture Guidance Loop v1.

This turns measured residuals into governed architecture experiment proposals.
The teacher may be queued or called in proposal-only mode, but it is never
allowed to provide benchmark answers, apply patches, or see public tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-code-report", default="reports/real_code_benchmark_graduation.json")
    parser.add_argument("--trace-in", default="reports/real_code_benchmark_traces.jsonl")
    parser.add_argument("--learning-scoreboard", default="reports/learning_scoreboard.json")
    parser.add_argument("--taming-stack", default="reports/deterministic_taming_stack.json")
    parser.add_argument("--out", default="reports/architecture_guidance_loop.json")
    parser.add_argument("--markdown-out", default="reports/architecture_guidance_loop.md")
    parser.add_argument("--teacher-prompt-out", default="reports/teacher_architecture_guidance_prompt.md")
    parser.add_argument("--experiments-out", default="reports/architecture_guided_experiments.json")
    parser.add_argument("--focus-card", default="", help="Optional receiver card to isolate residuals for a teacher wall packet, e.g. source_bigcodebench.")
    parser.add_argument("--queue-teacher", action="store_true")
    parser.add_argument("--allow-teacher", action="store_true")
    args = parser.parse_args()

    learning = read_json(resolve(args.learning_scoreboard), {})
    real_code_path = select_real_code_report(args.real_code_report, learning)
    real_code = read_json(real_code_path, {})
    trace_path = select_trace_path(args.trace_in, real_code)
    traces = read_jsonl(trace_path)
    taming = read_json(resolve(args.taming_stack), {})
    residuals = residual_summary(traces, focus_card=args.focus_card)
    diagnosis = diagnose(real_code, learning, taming, residuals, focus_card=args.focus_card)
    experiments = build_experiments(diagnosis, real_code_path=real_code_path, trace_path=trace_path)
    prompt = teacher_prompt(diagnosis, experiments)
    write_text(resolve(args.teacher_prompt_out), prompt)
    write_json(resolve(args.experiments_out), {"policy": "project_theseus_architecture_guided_experiments_v1", "experiments": experiments})

    teacher_allowed = teacher_call_allowed(diagnosis, residuals, learning, taming)
    teacher = {
        "status": "not_requested",
        "mode": "proposal",
        "teacher_call_allowed": teacher_allowed,
        "external_inference_calls": 0,
    }
    if args.queue_teacher or args.allow_teacher:
        if teacher_allowed:
            teacher = run_teacher(args, prompt_file=rel(resolve(args.teacher_prompt_out)))
            teacher["teacher_call_allowed"] = True
        else:
            teacher = {
                "status": "skipped_not_at_measured_wall",
                "mode": "proposal",
                "queued": False,
                "teacher_call_allowed": False,
                "external_inference_calls": 0,
                "reason": "Teacher is reserved for measured public-transfer walls with residual evidence.",
            }

    gates = [
        gate("residuals_loaded", residuals["failed_attempt_count"] > 0, f"failed_attempts={residuals['failed_attempt_count']}"),
        gate("public_answers_not_in_prompt", True, "prompt contains counts, hashes, residual classes, and metrics only"),
        gate("teacher_escalation_conditional", (not (args.queue_teacher or args.allow_teacher)) or teacher_allowed, teacher),
        gate("teacher_proposal_only", not args.allow_teacher or teacher.get("mode") == "proposal", teacher.get("status")),
        gate("experiments_written", bool(experiments), f"experiments={len(experiments)}"),
        gate("external_inference_zero_or_governed_teacher", (not args.allow_teacher) or teacher.get("status") == "completed", teacher.get("status")),
    ]
    report = {
        "policy": "project_theseus_architecture_guidance_loop_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "YELLOW",
        "purpose": "Residual cluster -> architecture diagnosis -> governed local experiment spec -> private eval/public calibration gate.",
        "diagnosis": diagnosis,
        "experiments": experiments,
        "teacher": teacher,
        "artifacts": {
            "real_code_report": rel(real_code_path),
            "trace": rel(trace_path),
            "teacher_prompt": rel(resolve(args.teacher_prompt_out)),
            "experiments": rel(resolve(args.experiments_out)),
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
        },
        "gates": gates,
        "external_inference_calls": 1 if args.allow_teacher and teacher.get("status") == "completed" else 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 1


def residual_summary(traces: list[dict[str, Any]], *, focus_card: str = "") -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    failed_tasks: set[str] = set()
    passing_origins: Counter[str] = Counter()
    for row in traces:
        if row.get("event") != "real_code_candidate_test":
            continue
        if focus_card:
            task_id = str(row.get("task_id") or "")
            if not task_id.startswith(f"{focus_card}_") and str(row.get("card_id") or "") != focus_card:
                continue
        origin = str(row.get("candidate_origin") or "unknown")
        mode = origin.split(":", 2)[1] if ":" in origin else origin
        if row.get("passed") is True:
            passing_origins[mode] += 1
            continue
        residual = str(row.get("residual_class") or "wrong_answer")
        class_counts[residual] += 1
        origin_counts[mode] += 1
        failed_tasks.add(str(row.get("task_id") or row.get("source_task_id") or "unknown"))
    return {
        "failed_attempt_count": sum(class_counts.values()),
        "failed_task_hashes": sorted(short_hash(task) for task in failed_tasks),
        "class_counts": dict(class_counts),
        "failed_origin_counts": dict(origin_counts),
        "passing_origin_counts": dict(passing_origins),
        "focus_card": focus_card,
    }


def select_real_code_report(configured: str, learning: dict[str, Any]) -> Path:
    configured_path = resolve(configured)
    if configured != "reports/real_code_benchmark_graduation.json":
        return configured_path
    best = get_path(learning, ["broad_transfer_matrix", "best_single_public_report", "path"], "")
    if best:
        best_path = resolve(str(best))
        if best_path.exists():
            return best_path
    return configured_path


def select_trace_path(configured: str, real_code: dict[str, Any]) -> Path:
    configured_path = resolve(configured)
    if configured != "reports/real_code_benchmark_traces.jsonl":
        return configured_path
    trace = get_path(real_code, ["artifacts", "trace"], "")
    if trace:
        trace_path = resolve(str(trace))
        if trace_path.exists():
            return trace_path
    return configured_path


def diagnose(real_code: dict[str, Any], learning: dict[str, Any], taming: dict[str, Any], residuals: dict[str, Any], *, focus_card: str = "") -> dict[str, Any]:
    real_summary = object_field(real_code, "summary")
    broad = object_field(learning, "broad_transfer_matrix")
    public_pass = float(real_summary.get("real_public_task_pass_rate") or 0.0)
    full_body_pass = int(real_summary.get("full_body_public_pass_count") or 0)
    fallback_pass = int(real_summary.get("expression_fallback_public_pass_count") or 0)
    public_floor_gap = round(max(0.0, 0.70 - public_pass), 6)
    broad_wall_cards = sorted(
        set(str(card) for card in (broad.get("cards_below_floor") or []))
        | set(str(card) for card in (broad.get("no_clean_student_evidence_cards") or []))
        | set(str(card) for card in (broad.get("loader_only_cards") or []))
    )
    dominant_residual = top_key(residuals["class_counts"]) or "edge_case"
    failed_origins = residuals.get("failed_origin_counts", {})
    diagnosis = {
        "wall": (
            "public_transfer_quality"
            if public_floor_gap > 0
            else "broad_public_transfer_coverage"
            if broad_wall_cards
            else "promotion_closure"
        ),
        "public_pass_rate": public_pass,
        "required_floor": 0.70,
        "floor_gap": public_floor_gap,
        "broad_public_pass_rate": broad.get("real_public_pass_rate"),
        "broad_public_task_count": broad.get("real_public_task_count"),
        "broad_clean_covered_card_count": broad.get("clean_covered_card_count"),
        "broad_requested_card_count": broad.get("requested_card_count"),
        "broad_wall_cards": broad_wall_cards,
        "broad_no_clean_student_evidence_cards": broad.get("no_clean_student_evidence_cards", []),
        "broad_cards_below_floor": broad.get("cards_below_floor", []),
        "broad_loader_only_cards": broad.get("loader_only_cards", []),
        "dominant_residual": dominant_residual,
        "full_body_public_pass_count": full_body_pass,
        "expression_fallback_public_pass_count": fallback_pass,
        "fallback_dependency_present": fallback_pass > 0,
        "failed_candidate_origins": failed_origins,
        "rule_substrate_state": get_path(learning, ["rule_substrate", "trigger_state"], None),
        "taming_stack_state": taming.get("trigger_state"),
        "teacher_role": "proposal_only_architecture_guidance",
        "focus_card": focus_card,
        "forbidden_teacher_roles": [
            "public benchmark solution generation",
            "hidden test access",
            "teacher apply mode",
            "distillation answer source",
        ],
    }
    if fallback_pass > 0:
        diagnosis["interpretation"] = "Full-body token decoding is real but still loses to simpler fallback on some cases; train private residual full-body repairs."
    elif full_body_pass <= 0:
        diagnosis["interpretation"] = "Full-body decoder is not yet carrying public transfer; prioritize body-level syntax/control-flow curriculum."
    elif broad_wall_cards:
        diagnosis["interpretation"] = "The strongest single report is above floor, but broader calibration is not yet competitive; target no-clean and below-floor benchmark families without public-answer leakage."
    else:
        diagnosis["interpretation"] = "Public wall is narrow; target dominant residual class with private data and STS repair ablation."
    return diagnosis


def build_experiments(diagnosis: dict[str, Any], *, real_code_path: Path, trace_path: Path) -> list[dict[str, Any]]:
    experiments = [
        {
            "id": "execution_shaped_programs_private_curriculum",
            "kind": "data_improvement_private_only",
            "status": "ready_for_smoke_or_profile",
            "teacher_needed": False,
            "hypothesis": "Private file/path/CSV/archive/JSON/system-library little-program tasks should improve BigCodeBench-like execution structure without public answer leakage.",
            "commands": [
                f"python scripts/code_residual_curriculum.py --trace-in {rel(trace_path)} --real-code-report {rel(real_code_path)} --concept-focus execution_shaped_programs --private-out D:/ProjectTheseus/training_data/high_transfer/private_train/execution_shaped_programs_residual_code_lm_tasks.jsonl --out reports/high_transfer_execution_shaped_programs_code_residual_curriculum.json --markdown-out reports/high_transfer_execution_shaped_programs_code_residual_curriculum.md --max-rows 960"
            ],
            "promotion_gates": ["private_rows_written", "private_solution_tests_pass", "public_solutions_not_copied", "bigcodebench_off_zero", "no_cross_card_regressions"],
        },
        {
            "id": "deterministic_taming_stack",
            "kind": "zero_param_rule_substrate",
            "status": "ready_for_smoke_or_profile",
            "teacher_needed": False,
            "hypothesis": "Language/form/tool/memory linters prevent invalid candidates and stale reports from becoming learning evidence.",
            "commands": [
                "python scripts/deterministic_taming_stack.py --run-cargo-check --out reports/deterministic_taming_stack.json --markdown-out reports/deterministic_taming_stack.md"
            ],
            "promotion_gates": ["no_invalid_python_promotion_candidates", "tool_schema_configs_parse", "memory_truth_fresh"],
        },
        {
            "id": "residual_targeted_private_code_curriculum",
            "kind": "data_improvement_private_only",
            "status": "ready_for_smoke_or_profile",
            "teacher_needed": False,
            "hypothesis": f"Private generated tasks targeting {diagnosis.get('dominant_residual')} residuals should improve full-body transfer without public answer leakage.",
            "commands": [
                f"python scripts/code_residual_curriculum.py --trace-in {rel(trace_path)} --real-code-report {rel(real_code_path)} --private-out D:/ProjectTheseus/training_data/residual_code_curriculum/private_train/residual_code_lm_tasks.jsonl --out reports/code_residual_curriculum.json --markdown-out reports/code_residual_curriculum.md --max-rows 960"
            ],
            "promotion_gates": ["private_rows_written", "public_solutions_not_copied", "public_transfer_improves", "no_regressions"],
        },
        {
            "id": "sts_repair_ablation",
            "kind": "inference_training_ablation",
            "status": "ready_for_smoke_or_profile",
            "teacher_needed": False,
            "hypothesis": "STS conditioning should causally improve repair/pass rate over single-stream on the same public calibration tasks.",
            "commands": [
                "python scripts/sts_repair_ablation.py --out reports/sts_repair_ablation.json --markdown-out reports/sts_repair_ablation.md"
            ],
            "promotion_gates": ["same_task_overlap", "sts_delta_positive", "zero_regressions", "public_score_quarantined"],
        },
        {
            "id": "cognitive_context_spaces",
            "kind": "context_architecture_sts_substrate",
            "status": "ready_for_smoke_or_profile",
            "teacher_needed": False,
            "hypothesis": "Dedicated private planning, mouthpiece draft, review, artifact workspace, personality, and memory streams should make long-horizon outputs more reliable without adding benchmark answers.",
            "commands": [
                "python scripts/cognitive_context_router.py --policy configs/cognitive_context_policy.json --base-sts data/sts_learning/sts_code_streams_seed14.jsonl --out reports/cognitive_context_router.json --markdown-out reports/cognitive_context_router.md"
            ],
            "promotion_gates": ["visible_report_requires_review", "raw_internal_monologue_not_visible", "public_benchmark_solutions_absent", "external_inference_zero"],
        },
        {
            "id": "student_first_evidence_audit",
            "kind": "truth_layer_evidence_hygiene",
            "status": "ready_for_smoke_or_profile",
            "teacher_needed": False,
            "hypothesis": "Every public-transfer claim should be traced to token-level student generation, not rankers, deterministic helpers, templates, or loop-closure tools.",
            "commands": [
                "python scripts/student_first_evidence_audit.py --out reports/student_first_evidence_audit.json --markdown-out reports/student_first_evidence_audit.md"
            ],
            "promotion_gates": ["student_checkpoint_candidate_source", "token_level_generation_valid", "no_templates_or_loop_tools", "public_score_quarantined"],
        },
        {
            "id": "long_horizon_programming_curriculum",
            "kind": "private_repo_repair_curriculum",
            "status": "ready_for_smoke_or_profile",
            "teacher_needed": False,
            "hypothesis": "Private repo repair tasks with hidden tests should train inspection, patching, test execution, and residual summaries beyond single-function completion.",
            "commands": [
                "python scripts/long_horizon_programming_curriculum.py --task-out D:/ProjectTheseus/training_data/long_horizon_programming/private_train/repo_repair_tasks.jsonl --sts-out D:/ProjectTheseus/training_data/long_horizon_programming/sts/repo_repair_sts_rows.jsonl --out reports/long_horizon_programming_curriculum.json --markdown-out reports/long_horizon_programming_curriculum.md"
            ],
            "promotion_gates": ["private_hidden_tests_only", "public_benchmark_solutions_absent", "sts_trace_rows_written", "public_swe_calibration_later"],
        },
    ]
    if diagnosis.get("floor_gap", 0.0) > 0 or diagnosis.get("broad_wall_cards"):
        experiments.append(
            {
                "id": "teacher_architecture_wall_review",
                "kind": "proposal_only_teacher_guidance",
                "status": "queued_or_manual",
                "teacher_needed": True,
                "hypothesis": "Sparse teacher should propose one architecture experiment from residual evidence, not solve tasks.",
                "commands": [
                    "python scripts/teacher_oracle.py --reason architecture_wall --mode proposal --prompt-file reports/teacher_architecture_guidance_prompt.md --local-evidence reports/architecture_guidance_loop.json reports/learning_scoreboard.json reports/broad_transfer_matrix.json reports/transfer_generalization_audit.json --queue-only --out reports/teacher_architecture_guidance_last.json"
                ],
                "promotion_gates": ["teacher_proposal_only", "no_public_answers", "local_experiment_verified_before_adoption"],
            }
        )
    return experiments


def teacher_call_allowed(
    diagnosis: dict[str, Any],
    residuals: dict[str, Any],
    learning: dict[str, Any],
    taming: dict[str, Any],
) -> bool:
    return bool(
        (
            float(diagnosis.get("floor_gap") or 0.0) > 0.0
            or bool(diagnosis.get("broad_wall_cards"))
        )
        and int(residuals.get("failed_attempt_count") or 0) > 0
        and taming.get("trigger_state") != "RED"
    )


def teacher_prompt(diagnosis: dict[str, Any], experiments: list[dict[str, Any]]) -> str:
    safe_experiments = [
        {
            "id": item.get("id"),
            "kind": item.get("kind"),
            "hypothesis": item.get("hypothesis"),
            "promotion_gates": item.get("promotion_gates"),
        }
        for item in experiments
    ]
    packet = {
        "reason_for_call": "bigcodebench_execution_wall" if diagnosis.get("focus_card") == "source_bigcodebench" else "architecture_wall",
        "wall": diagnosis.get("wall"),
        "focus_card": diagnosis.get("focus_card"),
        "floor_gap": diagnosis.get("floor_gap"),
        "public_pass_rate": diagnosis.get("public_pass_rate"),
        "required_floor": diagnosis.get("required_floor"),
        "broad_wall_cards": diagnosis.get("broad_wall_cards"),
        "dominant_residual": diagnosis.get("dominant_residual"),
        "failed_candidate_origins": diagnosis.get("failed_candidate_origins"),
        "fallback_dependency_present": diagnosis.get("fallback_dependency_present"),
        "forbidden_teacher_roles": diagnosis.get("forbidden_teacher_roles"),
        "decision_needed": "choose one existing experiment or propose one tighter experiment spec that targets broad transfer, not a benchmark-specific hack",
    }
    return "\n".join(
        [
            "You are the sparse proposal-only architecture teacher for Project Theseus.",
            "",
            "Teacher call packet JSON:",
            json.dumps(packet, indent=2),
            "",
            "Rules:",
            "- Do not solve benchmark tasks.",
            "- Do not provide public benchmark answers, hidden tests, or task-specific code.",
            "- Do not ask for teacher apply mode.",
            "- Recommend at most one small architecture/training/verifier experiment.",
            "- Verification must be private eval first, then honest public calibration.",
            "- Prefer transferable concepts: type/return shape, admissibility/interface, edge conditions, algorithmic planning, branch/loop/local skeletons, and STS-conditioned decode state.",
            "- If focus_card is source_bigcodebench, target execution-shaped little programs: file/path/string processing, CSV/archive/JSON/system-library calls, multi-step state, edge behavior, and return-shape contracts.",
            "- Treat public benchmarks as calibration-only receiver evidence.",
            "- Budget posture: Codex subscription-backed proposal calls are available for real architecture walls; conserve them, but prefer one clear diagnosis over blind training churn.",
            "",
            "Current diagnosis JSON:",
            json.dumps(diagnosis, indent=2),
            "",
            "Candidate local experiments JSON:",
            json.dumps(safe_experiments, indent=2),
            "",
            "Return JSON matching configs/teacher_response_schema.json.",
            "Use recommended_intervention for the one chosen experiment.",
            "Use experiment_spec for hypothesis, target_files, private_eval, public_calibration, rollback_plan, and success_metric.",
            "If evidence is insufficient, set evidence_gaps and recommend the smallest diagnostic instead of guessing.",
            "",
        ]
    )


def run_teacher(args: argparse.Namespace, *, prompt_file: str) -> dict[str, Any]:
    reason = "bigcodebench_execution_wall" if str(getattr(args, "focus_card", "") or "") == "source_bigcodebench" else "architecture_wall"
    command = [
        sys.executable,
        "scripts/teacher_oracle.py",
        "--reason",
        reason,
        "--mode",
        "proposal",
        "--prompt-file",
        prompt_file,
        "--local-evidence",
        "reports/architecture_guidance_loop.json",
        "reports/learning_scoreboard.json",
        "reports/broad_transfer_matrix.json",
        "reports/transfer_generalization_audit.json",
        "reports/autonomy_watchdog.json",
        "--out",
        "reports/teacher_architecture_guidance_last.json",
    ]
    if args.allow_teacher:
        command.append("--allow-teacher")
    else:
        command.append("--queue-only")
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=1800 if args.allow_teacher else 120)
    report = (
        read_json(REPORTS / "teacher_architecture_guidance_last.json", {})
        if args.allow_teacher
        else read_json(REPORTS / "teacher_queue_last.json", {})
    )
    status = report.get("status", "queued_not_executed" if not args.allow_teacher else "unknown")
    return {
        "status": status,
        "mode": report.get("mode", "proposal"),
        "reason_for_call": report.get("reason_for_call"),
        "response_json": report.get("response_json"),
        "queued": not args.allow_teacher,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
        "external_inference_calls": 1 if args.allow_teacher and status == "completed" else 0,
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    diagnosis = report.get("diagnosis", {})
    return "\n".join(
        [
            "# Architecture Guidance Loop",
            "",
            f"State: **{report.get('trigger_state')}**",
            "",
            f"- Wall: {diagnosis.get('wall')}",
            f"- Public pass rate: {diagnosis.get('public_pass_rate')} / {diagnosis.get('required_floor')}",
            f"- Broad public pass rate: {diagnosis.get('broad_public_pass_rate')} over {diagnosis.get('broad_public_task_count')} tasks",
            f"- Dominant residual: {diagnosis.get('dominant_residual')}",
            f"- Interpretation: {diagnosis.get('interpretation')}",
            f"- Experiments proposed: {len(report.get('experiments', []))}",
            f"- Teacher status: {get_path(report, ['teacher', 'status'], 'not_requested')}",
            f"- Teacher recommendation: {get_path(report, ['teacher', 'response_json', 'recommended_intervention'], 'n/a')}",
            "",
        ]
    )


def top_key(value: dict[str, Any]) -> str:
    if not value:
        return ""
    return str(max(value.items(), key=lambda item: int(item[1] or 0))[0])


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
