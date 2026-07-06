#!/usr/bin/env python3
"""Score Private Residual Repair v3 heldout rows against candidate manifests.

This is a private-only scorer. It executes generated candidates against private
synthetic heldout tests and emits the gate metrics needed before any future
public calibration can be proposed.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus_archive_resolver import read_jsonl_follow_pointer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_residual_repair_v3_heldout_code_lm_tasks.jsonl"
DEFAULT_CANDIDATES = ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_student_repair.jsonl"
DEFAULT_CONTROL = ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_student_repair_sts_off_control.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default=str(DEFAULT_HELDOUT.relative_to(ROOT)))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--control-candidates", default=str(DEFAULT_CONTROL.relative_to(ROOT)))
    parser.add_argument(
        "--task-limit",
        type=int,
        default=0,
        help="Optional private heldout task limit for canary scoring. Zero scores the full heldout set.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=2)
    parser.add_argument(
        "--adapter-off",
        action="store_true",
        help=(
            "Score capability with private-v3 diagnostic semantic adapters excluded from pass credit. "
            "Adapters are still executed and reported as diagnostic evidence."
        ),
    )
    parser.add_argument("--out", default="reports/private_residual_repair_v3_heldout_score.json")
    parser.add_argument("--markdown-out", default="reports/private_residual_repair_v3_heldout_score.md")
    args = parser.parse_args()

    report = score_heldout(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def score_heldout(args: argparse.Namespace) -> dict[str, Any]:
    heldout_path = resolve(args.heldout)
    candidate_path = resolve(args.candidates)
    rows = read_jsonl(heldout_path)
    heldout_total_before_limit = len(rows)
    if int(args.task_limit) > 0:
        rows = rows[: int(args.task_limit)]
    candidates = read_jsonl(candidate_path)
    control_path = resolve(args.control_candidates) if args.control_candidates else None
    control_candidates = read_jsonl(control_path) if control_path is not None else []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        task_id = str(candidate.get("task_id") or "")
        if task_id:
            by_task[task_id].append(candidate)
    results = []
    family_counts: Counter[str] = Counter()
    family_passes: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_passes: Counter[str] = Counter()
    no_admissible = 0
    live_stdin_passes = 0
    sts_task_passes = 0
    non_sts_task_passes = 0
    learned_candidate_task_passes = 0
    structural_action_task_passes = 0
    diagnostic_adapter_task_passes = 0
    diagnostic_adapter_only_tasks = 0
    task_regressions = 0
    fallback_flag_count = 0
    for row in rows:
        family = str(row.get("targeted_private_residual_family_v3") or "unknown")
        category = str(row.get("category") or "unknown")
        family_counts[family] += 1
        category_counts[category] += 1
        task_candidates = by_task.get(str(row.get("task_id") or ""), [])
        if not task_candidates:
            no_admissible += 1
            results.append(task_result(row, False, "no_candidate", "", False, False, False, False, 0, []))
            continue
        no_admissible_residual_only = all(no_admissible_residual_candidate(candidate) for candidate in task_candidates)
        if no_admissible_residual_only:
            no_admissible += 1
        candidate_errors = []
        passed = False
        pass_mode = ""
        pass_sts = False
        learned_candidate_passed = False
        structural_action_passed = False
        diagnostic_adapter_passed = False
        diagnostic_pass_without_credit = False
        for index, candidate in enumerate(task_candidates):
            if fallback_return_candidate(candidate):
                fallback_flag_count += 1
            ok, error = run_candidate(row, candidate, timeout_seconds=max(1, int(args.timeout_seconds)))
            if ok:
                candidate_mode = str(candidate.get("candidate_generation_mode") or "")
                candidate_sts = bool(
                    candidate.get("sts_stream_conditioned")
                    or candidate.get("sts_candidate_expression_used")
                    or "sts_conditioned" in candidate_mode.lower()
                )
                candidate_diagnostic_adapter = private_residual_v3_diagnostic_adapter_mode(candidate_mode)
                candidate_structural_action = structural_action_candidate(candidate)
                candidate_learned = learned_student_candidate(candidate) or candidate_structural_action
                if candidate_diagnostic_adapter:
                    diagnostic_adapter_passed = True
                    if bool(args.adapter_off):
                        diagnostic_pass_without_credit = True
                        continue
                else:
                    if candidate_learned:
                        learned_candidate_passed = True
                    if candidate_structural_action:
                        structural_action_passed = True
                if not passed:
                    passed = True
                    pass_mode = candidate_mode
                    pass_sts = candidate_sts
                continue
            if len(candidate_errors) < 3:
                candidate_errors.append(error)
        if passed:
            family_passes[family] += 1
            category_passes[category] += 1
            if family == "livecodebench_stdin_proxy_v1":
                live_stdin_passes += 1
            if pass_sts:
                sts_task_passes += 1
            else:
                non_sts_task_passes += 1
            if learned_candidate_passed:
                learned_candidate_task_passes += 1
            if structural_action_passed:
                structural_action_task_passes += 1
            if diagnostic_adapter_passed:
                diagnostic_adapter_task_passes += 1
        else:
            if any(candidate.get("same_seed_non_sts_comparator") for candidate in task_candidates):
                task_regressions += 1
            if diagnostic_pass_without_credit:
                diagnostic_adapter_only_tasks += 1
        results.append(
            task_result(
                row,
                passed,
                "passed"
                if passed
                else "diagnostic_adapter_only_adapter_off_failed"
                if diagnostic_pass_without_credit
                else "failed_private_tests",
                pass_mode,
                pass_sts,
                learned_candidate_passed,
                structural_action_passed,
                diagnostic_adapter_passed,
                len(task_candidates),
                candidate_errors,
            )
        )
    task_count = len(rows)
    pass_count = sum(1 for row in results if row["passed"])
    pass_rate = round(pass_count / max(1, task_count), 6)
    control_summary = score_control(rows, control_candidates, timeout_seconds=max(1, int(args.timeout_seconds)))
    if control_summary["task_count"] > 0:
        control_passes = {
            str(row.get("task_id") or "")
            for row in control_summary["results"]
            if bool(row.get("passed"))
        }
        current_passes = {str(row.get("task_id") or "") for row in results if bool(row.get("passed"))}
        task_regressions = len(control_passes - current_passes)
        sts_delta = round(pass_rate - float(control_summary["pass_rate"]), 6)
    else:
        sts_delta = None
    family_rates = {
        family: {
            "task_count": family_counts[family],
            "pass_count": family_passes[family],
            "pass_rate": round(family_passes[family] / max(1, family_counts[family]), 6),
        }
        for family in sorted(family_counts)
    }
    category_rates = {
        category: {
            "task_count": category_counts[category],
            "pass_count": category_passes[category],
            "pass_rate": round(category_passes[category] / max(1, category_counts[category]), 6),
        }
        for category in sorted(category_counts)
    }
    no_admissible_rate = round(no_admissible / max(1, task_count), 6)
    gates = [
        gate("heldout_rows_present", task_count > 0, task_count),
        gate("candidate_rows_present", len(candidates) > 0, len(candidates)),
        gate("private_residual_v3_heldout_pass_rate_floor", pass_rate >= 0.70, {"observed": pass_rate, "minimum": 0.70}),
        gate("adapter_off_scoring_enabled", bool(args.adapter_off), bool(args.adapter_off)),
        gate(
            "private_residual_v3_learned_candidate_pass_rate_floor",
            (learned_candidate_task_passes / max(1, task_count)) >= 0.70,
            {
                "observed": round(learned_candidate_task_passes / max(1, task_count), 6),
                "minimum": 0.70,
                "diagnostic_adapter_task_pass_rate": round(diagnostic_adapter_task_passes / max(1, task_count), 6),
            },
        ),
        gate("no_admissible_task_rate_floor", no_admissible_rate <= 0.03, {"observed": no_admissible_rate, "maximum": 0.03}),
        gate("livecodebench_private_stdin_proxy_nonzero", live_stdin_passes >= 1, live_stdin_passes),
        gate("fallback_returns_zero", fallback_flag_count == 0, fallback_flag_count),
        gate(
            "same_seed_control_present_and_nonregressive",
            control_summary["task_count"] == task_count and task_regressions == 0,
            {
                "sts_delta": sts_delta,
                "sts_lift_claim_allowed": bool(sts_delta is not None and sts_delta > 0.0 and task_regressions == 0),
                "task_regressions": task_regressions,
                "control_task_count": control_summary["task_count"],
                "required_control_task_count": task_count,
                "score_semantics": "Neutral STS delta is not a blocker when the learned STS-on path and matched non-STS control both pass; it only blocks a positive STS-lift claim.",
            },
        ),
        gate("public_tests_not_used", True, "heldout tests are private synthetic v3 tests"),
        gate("public_solutions_not_used", True, "candidate scoring reads no public benchmark solution files"),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    return {
        "policy": "project_theseus_private_residual_repair_v3_heldout_score_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "heldout": rel(heldout_path),
            "candidates": rel(candidate_path),
            "control_candidates": rel(control_path) if control_path is not None else "",
            "timeout_seconds": int(args.timeout_seconds),
            "task_limit": int(args.task_limit),
        },
        "summary": {
            "private_residual_v3_heldout_task_count": task_count,
            "private_residual_v3_heldout_total_before_limit": heldout_total_before_limit,
            "private_residual_v3_heldout_task_limit": int(args.task_limit),
            "private_residual_v3_heldout_pass_count": pass_count,
            "private_residual_v3_heldout_pass_rate": pass_rate,
            "no_admissible_task_count": no_admissible,
            "no_admissible_task_rate": no_admissible_rate,
            "livecodebench_private_stdin_proxy_pass_count": live_stdin_passes,
            "private_residual_v3_sts_delta": sts_delta,
            "private_residual_v3_sts_regressions": task_regressions,
            "private_residual_v3_sts_lift_claim_allowed": bool(
                sts_delta is not None and sts_delta > 0.0 and task_regressions == 0
            ),
            "private_residual_v3_same_seed_control_nonregressive": bool(
                control_summary["task_count"] == task_count and task_regressions == 0
            ),
            "private_residual_v3_sts_control_task_count": control_summary["task_count"],
            "private_residual_v3_sts_control_pass_count": control_summary["pass_count"],
            "private_residual_v3_sts_control_pass_rate": control_summary["pass_rate"],
            "sts_task_passes": sts_task_passes,
            "non_sts_task_passes": non_sts_task_passes,
            "learned_candidate_task_passes": learned_candidate_task_passes,
            "learned_candidate_task_pass_rate": round(learned_candidate_task_passes / max(1, task_count), 6),
            "structural_action_candidate_task_passes": structural_action_task_passes,
            "structural_action_candidate_task_pass_rate": round(structural_action_task_passes / max(1, task_count), 6),
            "diagnostic_adapter_task_passes": diagnostic_adapter_task_passes,
            "diagnostic_adapter_task_pass_rate": round(diagnostic_adapter_task_passes / max(1, task_count), 6),
            "diagnostic_adapter_only_adapter_off_task_count": diagnostic_adapter_only_tasks,
            "candidate_row_count": len(candidates),
            "adapter_off_scoring": bool(args.adapter_off),
            "fallback_return_candidate_count": fallback_flag_count,
            "family_rates": family_rates,
            "category_rates": category_rates,
            "public_tests_used": False,
            "public_solutions_used": False,
            "external_inference_calls": 0,
        },
        "gates": gates,
        "results": results,
        "control_results": control_summary["results"],
        "next_actions": next_actions(
            pass_rate,
            no_admissible_rate,
            live_stdin_passes,
            sts_delta,
            category_rates,
            learned_candidate_task_passes,
            diagnostic_adapter_task_passes,
            task_count,
        ),
        "external_inference_calls": 0,
    }


def score_control(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not candidates:
        return {"task_count": 0, "pass_count": 0, "pass_rate": 0.0, "results": []}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        task_id = str(candidate.get("task_id") or "")
        if task_id:
            by_task[task_id].append(candidate)
    results = []
    for row in rows:
        passed = False
        for candidate in by_task.get(str(row.get("task_id") or ""), []):
            ok, _error = run_candidate(row, candidate, timeout_seconds=timeout_seconds)
            if ok:
                passed = True
                break
        results.append({"task_id": row.get("task_id"), "passed": passed})
    pass_count = sum(1 for row in results if row["passed"])
    task_count = len(rows)
    return {
        "task_count": task_count,
        "pass_count": pass_count,
        "pass_rate": round(pass_count / max(1, task_count), 6),
        "results": results,
    }


def run_candidate(row: dict[str, Any], candidate: dict[str, Any], *, timeout_seconds: int) -> tuple[bool, str]:
    code = str(candidate.get("code") or "")
    tests = str(row.get("tests") or "")
    if not code.strip() or not tests.strip():
        return False, "missing candidate code or tests"
    namespace: dict[str, Any] = {}

    if not hasattr(signal, "SIGALRM"):
        try:
            exec(code, namespace, namespace)
            exec(tests, namespace, namespace)
            return True, ""
        except Exception as exc:  # pragma: no cover - report detail is the test oracle
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            return False, detail[:240]

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"candidate exceeded {timeout_seconds}s")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        exec(code, namespace, namespace)
        exec(tests, namespace, namespace)
        return True, ""
    except Exception as exc:  # pragma: no cover - report detail is the test oracle
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return False, detail[:240]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def no_admissible_residual_candidate(candidate: dict[str, Any]) -> bool:
    mode = str(candidate.get("candidate_generation_mode") or "").lower()
    code = str(candidate.get("code") or "").lower()
    return mode == "student_decoder_no_admissible_candidate_residual" or (
        "no_admissible_candidate" in mode
        and "student decoder emitted no admissible candidate" in code
    )


def private_residual_v3_diagnostic_adapter_mode(mode: str) -> bool:
    normalized = mode.lower()
    return "private_residual_v3" in normalized and "semantic_adapter" in normalized


def learned_student_candidate(candidate: dict[str, Any]) -> bool:
    mode = str(candidate.get("candidate_generation_mode") or "").lower()
    if no_admissible_residual_candidate(candidate):
        return False
    if candidate.get("same_seed_non_sts_comparator"):
        return False
    if private_residual_v3_diagnostic_adapter_mode(mode):
        return False
    if fallback_return_candidate(candidate):
        return False
    return bool(
        candidate.get("token_level_code_generation_learned")
        or candidate.get("grammar_masked_learned_token_candidate")
        or "contract_guided_token_decoder" in mode
        or "full_body_token_beam" in mode
        or "greedy_body_token_decoder" in mode
        or "private_residual_v3_train_induced_structural_token_decoder" in mode
    )


def structural_action_candidate(candidate: dict[str, Any]) -> bool:
    mode = str(candidate.get("candidate_generation_mode") or "").lower()
    return bool(
        candidate.get("structural_action_candidate")
        or candidate.get("private_residual_v3_train_induced_structural_token_stage")
        or "structural_action" in mode
        or "train_induced_structural_token_decoder" in mode
    )


def fallback_return_candidate(candidate: dict[str, Any]) -> bool:
    mode = str(candidate.get("candidate_generation_mode") or "").lower()
    return bool(candidate.get("expression_memory_fallback") or ("fallback" in mode and "fallback_skipped" not in mode))


def task_result(
    row: dict[str, Any],
    passed: bool,
    status: str,
    pass_mode: str,
    pass_sts: bool,
    learned_candidate_passed: bool,
    structural_action_passed: bool,
    diagnostic_adapter_passed: bool,
    candidate_count: int,
    sample_errors: list[str],
) -> dict[str, Any]:
    return {
        "task_id": row.get("task_id"),
        "category": row.get("category"),
        "family": row.get("targeted_private_residual_family_v3"),
        "passed": bool(passed),
        "status": status,
        "pass_candidate_mode": pass_mode,
        "pass_sts_conditioned": bool(pass_sts),
        "learned_candidate_passed": bool(learned_candidate_passed),
        "structural_action_candidate_passed": bool(structural_action_passed),
        "diagnostic_adapter_passed": bool(diagnostic_adapter_passed),
        "candidate_count": candidate_count,
        "sample_errors": sample_errors,
    }


def next_actions(
    pass_rate: float,
    no_admissible_rate: float,
    live_stdin_passes: int,
    sts_delta: float | None,
    category_rates: dict[str, dict[str, Any]],
    learned_candidate_task_passes: int,
    diagnostic_adapter_task_passes: int,
    task_count: int,
) -> list[str]:
    actions = []
    if pass_rate < 0.70:
        actions.append("Private v3 heldout pass rate is below 0.70; repair decoder semantics before public calibration.")
    if no_admissible_rate > 0.03:
        zero_categories = [
            category
            for category, data in sorted(category_rates.items())
            if int(data.get("task_count") or 0) > 0 and int(data.get("pass_count") or 0) == 0
        ]
        suffix = f" Zero-pass categories: {', '.join(zero_categories)}." if zero_categories else ""
        actions.append(f"No-admissible rate is above 0.03; repair candidate-floor generation for missing v3 families.{suffix}")
    if live_stdin_passes < 1:
        actions.append("LiveCodeBench-style stdin proxy still has zero passes; prioritize stdin parser/output-format candidates.")
    if sts_delta is None:
        actions.append("STS control evidence is missing for this heldout scorer; provide a same-task control manifest before claiming STS improvement.")
    elif sts_delta < 0.0:
        actions.append("STS did not beat non-STS on this private heldout scorer; keep STS controls and improve semantic routing.")
    learned_rate = learned_candidate_task_passes / max(1, task_count)
    if diagnostic_adapter_task_passes > 0 and learned_rate < 0.70:
        actions.append(
            "Diagnostic adapters cleared candidate-floor behavior, but learned/student candidates remain below floor; do not treat adapter passes as student learning."
        )
    if not actions:
        actions.append("Private v3 heldout gates are clear; rerun the full decoder/transfer/maturity stack before public calibration.")
    return actions


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Private Residual Repair V3 Heldout Score",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Heldout pass rate: {summary.get('private_residual_v3_heldout_pass_rate')}",
        f"- Heldout passes: {summary.get('private_residual_v3_heldout_pass_count')}/{summary.get('private_residual_v3_heldout_task_count')}",
        f"- No-admissible rate: {summary.get('no_admissible_task_rate')}",
        f"- LiveCodeBench stdin proxy passes: {summary.get('livecodebench_private_stdin_proxy_pass_count')}",
        f"- STS delta: {summary.get('private_residual_v3_sts_delta')}",
        f"- STS regressions: {summary.get('private_residual_v3_sts_regressions')}",
        f"- STS lift claim allowed: {summary.get('private_residual_v3_sts_lift_claim_allowed')}",
        f"- Same-seed control non-regressive: {summary.get('private_residual_v3_same_seed_control_nonregressive')}",
        f"- Adapter-off scoring: {summary.get('adapter_off_scoring')}",
        f"- Learned/student candidate passes: {summary.get('learned_candidate_task_passes')}/{summary.get('private_residual_v3_heldout_task_count')} = {summary.get('learned_candidate_task_pass_rate')}",
        f"- Structural-action candidate passes: {summary.get('structural_action_candidate_task_passes')}/{summary.get('private_residual_v3_heldout_task_count')} = {summary.get('structural_action_candidate_task_pass_rate')}",
        f"- Diagnostic adapter passes: {summary.get('diagnostic_adapter_task_passes')}/{summary.get('private_residual_v3_heldout_task_count')} = {summary.get('diagnostic_adapter_task_pass_rate')}",
        f"- Fallback return flags: {summary.get('fallback_return_candidate_count')}",
        f"- Candidate rows: {summary.get('candidate_row_count')}",
        "",
        "## Family Rates",
    ]
    for family, data in (summary.get("family_rates") or {}).items():
        lines.append(f"- `{family}`: {data.get('pass_count')}/{data.get('task_count')} = {data.get('pass_rate')}")
    lines.extend(["", "## Category Rates"])
    for category, data in (summary.get("category_rates") or {}).items():
        lines.append(f"- `{category}`: {data.get('pass_count')}/{data.get('task_count')} = {data.get('pass_rate')}")
    lines.append("")
    lines.append("Public benchmark tests and solutions are not used by this scorer.")
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
