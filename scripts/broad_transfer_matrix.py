"""Broad public-transfer matrix for Project Theseus code evidence.

This report prevents one small smoke slice from being mistaken for overall
coding ability. It selects the strongest clean report available for each public
code family, records STS deltas and residual families, and keeps promotion
semantics separate from calibration semantics. This is intentionally a
best-evidence matrix, not a single-checkpoint promotion claim.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_CODE_FLOOR = 0.70
DEFAULT_CARDS = [
    "source_human_eval",
    "source_mbpp",
    "source_evalplus",
    "source_bigcodebench",
    "source_livecodebench",
]
ALLOWED_CANDIDATE_SOURCES = {
    "student_code_lm_checkpoint_v1",
    "student_token_generator_checkpoint_v1",
}
ALLOWED_SCORE_CLAIMS = {
    "student_code_lm_checkpoint_public_task_calibration_only",
    "student_token_generator_checkpoint_public_task_calibration_only",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", nargs="*", default=DEFAULT_CARDS)
    parser.add_argument("--min-public-tasks", type=int, default=32)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--out", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--markdown-out", default="reports/broad_transfer_matrix.md")
    args = parser.parse_args()

    reports_dir = resolve(args.reports_dir)
    reports = load_real_code_reports(reports_dir)
    rows = [build_card_row(card, reports, args.min_public_tasks) for card in args.cards]
    best_single = best_single_public_report(reports)
    payload = build_payload(rows, args.cards, best_single, args.min_public_tasks)
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def load_real_code_reports(reports_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(reports_dir.glob("real_code_benchmark*.json")):
        payload = read_json(path)
        if payload.get("policy") != "project_theseus_real_code_benchmark_graduation_v1":
            continue
        reports.append((path, payload))
    return reports


def build_card_row(
    card: str,
    reports: list[tuple[Path, dict[str, Any]]],
    min_public_tasks: int,
) -> dict[str, Any]:
    candidates = [(path, payload) for path, payload in reports if card in report_cards(payload)]
    if not candidates:
        return {
            "card_id": card,
            "status": "missing",
            "trigger_state": "YELLOW",
            "selected_report": "",
            "case_count": 0,
            "public_task_count": 0,
            "loader_regression_case_count": 0,
            "multi_stream_pass_rate": 0.0,
            "single_stream_pass_rate": 0.0,
            "pass_rate_delta": 0.0,
            "no_cheat_valid": False,
            "no_cheat_violations": ["missing_report"],
            "residual_family_counts": {},
        }
    candidates.sort(key=lambda item: report_rank(item[0], item[1], card), reverse=True)
    path, payload = candidates[0]
    suite = suite_for_card(payload, card)
    summary = object_field(payload, "summary")
    audit = no_cheat_audit(payload, suite)
    has_clean_candidate = any(
        not no_cheat_audit(candidate_payload, suite_for_card(candidate_payload, card))["violations"]
        for _candidate_path, candidate_payload in candidates
    )
    suite_evidence = str(suite.get("benchmark_evidence_level") or payload.get("benchmark_evidence_level") or "")
    if suite_evidence == "public_benchmark_task_regression":
        public_task_count = int(suite.get("case_count") or 0)
    else:
        public_task_count = int(summary.get("public_task_count") or 0) if len(report_cards(payload)) == 1 else 0
    loader_count = int(summary.get("loader_regression_case_count") or 0)
    if public_task_count == 0 and "loader" in suite_evidence:
        loader_count = int(suite.get("case_count") or loader_count)
    case_count = int(suite.get("case_count") or summary.get("total_case_count") or 0)
    residual_counts = residual_family_counts(suite.get("residuals", []), payload.get("residuals", []), card)
    coverage_warnings = []
    if public_task_count > 0 and public_task_count < min_public_tasks:
        coverage_warnings.append(f"public_task_count_below_{min_public_tasks}")
    if public_task_count == 0 and loader_count > 0:
        coverage_warnings.append("loader_only_no_public_task_score")
    single_rate = numeric_value(suite, "single_stream_pass_rate", numeric_value(summary, "single_stream_pass_rate", 0.0))
    multi_rate = numeric_value(suite, "multi_stream_pass_rate", numeric_value(summary, "multi_stream_pass_rate", 0.0))
    pass_delta = numeric_value(suite, "pass_rate_delta", round(multi_rate - single_rate, 6))
    if multi_rate < PUBLIC_CODE_FLOOR:
        coverage_warnings.append("below_public_code_floor")
    clean_evidence_blockers = classify_clean_evidence_blockers(
        audit["violations"],
        coverage_warnings,
        suite_evidence=suite_evidence,
    )
    row_state = "GREEN"
    if audit["violations"] or coverage_warnings:
        row_state = "YELLOW"
    return {
        "card_id": card,
        "status": "selected" if not audit["violations"] else "no_clean_student_evidence",
        "trigger_state": row_state,
        "selected_report": display_path(path),
        "created_utc": payload.get("created_utc"),
        "candidate_source": payload.get("candidate_source"),
        "public_benchmark_score_claim": payload.get("public_benchmark_score_claim"),
        "score_semantics": payload.get("score_semantics"),
        "benchmark_evidence_level": suite_evidence,
        "case_count": case_count,
        "public_task_count": public_task_count,
        "loader_regression_case_count": loader_count,
        "single_stream_passed": int(suite.get("single_stream_passed") or round(case_count * float(suite.get("single_stream_pass_rate") or 0.0))),
        "multi_stream_passed": int(suite.get("multi_stream_passed") or round(case_count * float(suite.get("multi_stream_pass_rate") or 0.0))),
        "single_stream_pass_rate": single_rate,
        "multi_stream_pass_rate": multi_rate,
        "pass_rate_delta": pass_delta,
        "task_level_improvements": int(suite.get("task_level_improvements") or 0),
        "task_level_regressions": int(suite.get("task_level_regressions") or 0),
        "sts_transfer_behavior_changed": bool(suite.get("transfer_behavior_changed")),
        "no_cheat_valid": not audit["violations"],
        "clean_student_evidence_available": has_clean_candidate,
        "no_cheat_checks": audit["checks"],
        "no_cheat_violations": audit["violations"],
        "coverage_warnings": coverage_warnings,
        "clean_evidence_blockers": clean_evidence_blockers,
        "residual_family_counts": residual_counts,
        "residual_task_ids": residual_task_ids(suite.get("residuals", [])),
    }


def build_payload(
    rows: list[dict[str, Any]],
    requested_cards: list[str],
    best_single: dict[str, Any],
    min_public_tasks: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("status") != "missing"]
    public_rows = [
        row
        for row in selected
        if row.get("no_cheat_valid") and int(row.get("public_task_count") or 0) > 0
    ]
    total_public_tasks = sum(int(row.get("public_task_count") or 0) for row in public_rows)
    total_multi_passed = sum(int(row.get("multi_stream_passed") or 0) for row in public_rows)
    total_single_passed = sum(int(row.get("single_stream_passed") or 0) for row in public_rows)
    violation_count = sum(len(row.get("no_cheat_violations") or []) for row in rows)
    no_clean_cards = [row["card_id"] for row in selected if not row.get("no_cheat_valid")]
    below_floor_cards = [
        row["card_id"]
        for row in public_rows
        if float(row.get("multi_stream_pass_rate") or 0.0) < PUBLIC_CODE_FLOOR
    ]
    missing_cards = [row["card_id"] for row in rows if row.get("status") == "missing"]
    loader_only_cards = [
        row["card_id"]
        for row in selected
        if int(row.get("public_task_count") or 0) == 0 and int(row.get("loader_regression_case_count") or 0) > 0
    ]
    coverage_warning_cards = [
        row["card_id"]
        for row in selected
        if row.get("coverage_warnings")
    ]
    trigger_state = "GREEN"
    if missing_cards or below_floor_cards or loader_only_cards or coverage_warning_cards or no_clean_cards:
        trigger_state = "YELLOW"
    if not selected:
        trigger_state = "RED"
    aggregate_pass = safe_div(total_multi_passed, total_public_tasks)
    aggregate_single = safe_div(total_single_passed, total_public_tasks)
    summary = {
        "requested_card_count": len(requested_cards),
        "covered_card_count": len(selected),
        "clean_covered_card_count": sum(1 for row in selected if row.get("no_cheat_valid")),
        "missing_cards": missing_cards,
        "no_clean_student_evidence_cards": no_clean_cards,
        "real_public_task_count": total_public_tasks,
        "real_public_multi_passed": total_multi_passed,
        "real_public_single_passed": total_single_passed,
        "real_public_pass_rate": round(aggregate_pass, 6),
        "real_public_single_stream_pass_rate": round(aggregate_single, 6),
        "real_public_sts_delta": round(aggregate_pass - aggregate_single, 6),
        "total_regressions": sum(int(row.get("task_level_regressions") or 0) for row in selected),
        "no_cheat_violation_count": violation_count,
        "cards_below_floor": below_floor_cards,
        "loader_only_cards": loader_only_cards,
        "coverage_warning_cards": coverage_warning_cards,
        "min_public_tasks_per_promotion_card": min_public_tasks,
        "promotion_candidate_card_count": sum(
            1
            for row in selected
            if not row.get("no_cheat_violations")
            and int(row.get("public_task_count") or 0) >= min_public_tasks
            and float(row.get("multi_stream_pass_rate") or 0.0) >= PUBLIC_CODE_FLOOR
            and int(row.get("task_level_regressions") or 0) == 0
        ),
    }
    return {
        "policy": "project_theseus_broad_transfer_matrix_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "thesis": "Broad public calibration is separate from promotion. No one small slice is overall coding ability.",
        "requested_cards": requested_cards,
        "required_public_task_floor": PUBLIC_CODE_FLOOR,
        "summary": summary,
        "best_single_public_report": best_single,
        "rows": rows,
        "promotion_evidence": False,
        "score_semantics": "broad calibration matrix; promotion must use one clean checkpoint/report with no-cheat evidence and sufficient coverage",
        "external_inference_calls": 0,
    }


def report_rank(path: Path, payload: dict[str, Any], card: str) -> tuple[Any, ...]:
    suite = suite_for_card(payload, card)
    summary = object_field(payload, "summary")
    audit = no_cheat_audit(payload, suite)
    suite_evidence = str(suite.get("benchmark_evidence_level") or payload.get("benchmark_evidence_level") or "")
    public_tasks = int(suite.get("case_count") or 0) if suite_evidence == "public_benchmark_task_regression" else int(summary.get("public_task_count") or 0)
    if "loader" in suite_evidence:
        public_tasks = 0
    total_cases = int(summary.get("total_case_count") or suite.get("case_count") or 0)
    multi_rate = numeric_value(suite, "multi_stream_pass_rate", numeric_value(summary, "real_public_task_pass_rate", 0.0))
    pass_delta = numeric_value(suite, "pass_rate_delta", numeric_value(summary, "pass_rate_delta", 0.0))
    regressions = int(suite.get("task_level_regressions") or summary.get("task_level_regressions_vs_single_stream") or 0)
    return (
        1 if not audit["violations"] else 0,
        1 if regressions == 0 else 0,
        public_tasks,
        multi_rate,
        pass_delta,
        total_cases,
        str(payload.get("created_utc") or ""),
        path.name,
    )


def best_single_public_report(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    clean: list[tuple[int, float, str, str, Path, dict[str, Any]]] = []
    for path, payload in reports:
        summary = object_field(payload, "summary")
        suite = suite_for_card(payload, report_cards(payload)[0] if report_cards(payload) else "")
        audit = no_cheat_audit(payload, suite)
        public_tasks = int(summary.get("public_task_count") or 0)
        if audit["violations"] or public_tasks <= 0:
            continue
        clean.append(
            (
                public_tasks,
                float(summary.get("real_public_task_pass_rate") or 0.0),
                str(payload.get("created_utc") or ""),
                path.name,
                path,
                payload,
            )
        )
    if not clean:
        return {}
    clean.sort(reverse=True)
    public_tasks, pass_rate, _, _, path, payload = clean[0]
    summary = object_field(payload, "summary")
    return {
        "path": display_path(path),
        "public_task_count": public_tasks,
        "real_public_task_pass_rate": pass_rate,
        "candidate_source": payload.get("candidate_source"),
        "score_claim": payload.get("public_benchmark_score_claim"),
        "pass_rate_delta": summary.get("pass_rate_delta"),
        "task_level_regressions": summary.get("task_level_regressions_vs_single_stream"),
    }


def no_cheat_audit(payload: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(payload, "summary")
    checks = {
        "policy_ok": payload.get("policy") == "project_theseus_real_code_benchmark_graduation_v1",
        "trigger_state_usable": payload.get("trigger_state") in {"GREEN", "YELLOW"},
        "candidate_source_allowed": payload.get("candidate_source") in ALLOWED_CANDIDATE_SOURCES,
        "score_claim_allowed": payload.get("public_benchmark_score_claim") in ALLOWED_SCORE_CLAIMS,
        "token_level_code_generation_learned": bool(summary.get("token_level_code_generation_learned")),
        "full_body_token_generation_present": int(summary.get("full_body_token_candidate_count") or 0) > 0,
        "benchmark_promotion_candidates_present": int(summary.get("benchmark_promotion_eligible_candidate_count") or 0) > 0,
        "candidate_provenance_valid": bool(summary.get("student_candidate_provenance_valid")),
        "benchmark_integrity_valid": bool(summary.get("student_candidate_benchmark_integrity_valid")),
        "template_like_candidate_count_zero": int(summary.get("template_like_candidate_count") or 0) == 0,
        "loop_closure_candidate_count_zero": int(summary.get("loop_closure_candidate_count") or 0) == 0,
        "expression_memory_fallback_count_zero": int(summary.get("expression_memory_fallback_count") or 0) == 0,
        "external_inference_zero": int(payload.get("external_inference_calls") or summary.get("external_inference_calls") or 0) == 0,
        "same_cases_compared": bool(suite.get("same_cases_compared", True)),
    }
    return {"checks": checks, "violations": [key for key, passed in checks.items() if not passed]}


def classify_clean_evidence_blockers(
    violations: list[str],
    coverage_warnings: list[str],
    *,
    suite_evidence: str,
) -> list[str]:
    blockers: list[str] = []
    violation_set = set(violations)
    if "candidate_source_allowed" in violation_set or "score_claim_allowed" in violation_set:
        blockers.append("student_candidate_source_not_allowed_for_broad_promotion")
    if "token_level_code_generation_learned" in violation_set:
        blockers.append("missing_learned_token_level_generation")
    if "full_body_token_generation_present" in violation_set:
        blockers.append("missing_full_body_token_candidates")
    if "benchmark_promotion_candidates_present" in violation_set:
        blockers.append("missing_benchmark_promotion_eligible_candidates")
    if "candidate_provenance_valid" in violation_set or "benchmark_integrity_valid" in violation_set:
        blockers.append("student_candidate_integrity_not_clean")
    if "loader" in suite_evidence or "loader_only_no_public_task_score" in coverage_warnings:
        blockers.append("loader_only_no_public_task_score")
    if any(item.startswith("public_task_count_below_") for item in coverage_warnings):
        blockers.append("public_task_slice_below_broad_minimum")
    if "below_public_code_floor" in coverage_warnings:
        blockers.append("below_public_code_floor")
    return blockers


def suite_for_card(payload: dict[str, Any], card: str) -> dict[str, Any]:
    suites = payload.get("suites")
    if not isinstance(suites, list):
        return {}
    for suite in suites:
        if isinstance(suite, dict) and suite.get("card_id") == card:
            return suite
    return suites[0] if suites and isinstance(suites[0], dict) else {}


def report_cards(payload: dict[str, Any]) -> list[str]:
    for key in ("cards", "requested_cards"):
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    suites = payload.get("suites")
    if isinstance(suites, list):
        return [str(suite.get("card_id")) for suite in suites if isinstance(suite, dict) and suite.get("card_id")]
    return []


def residual_family_counts(primary: Any, secondary: Any, card: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    residuals = primary if isinstance(primary, list) and primary else secondary
    if not isinstance(residuals, list):
        return counts
    for item in residuals:
        if not isinstance(item, dict):
            continue
        if item.get("card_id") not in {None, "", card}:
            continue
        family = str(item.get("concept_residual") or item.get("type") or item.get("category") or "unknown")
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def residual_task_ids(residuals: Any) -> list[str]:
    if not isinstance(residuals, list):
        return []
    return [
        str(item.get("task_id"))
        for item in residuals
        if isinstance(item, dict) and item.get("task_id")
    ][:64]


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def numeric_value(value: dict[str, Any], key: str, default: float = 0.0) -> float:
    if key not in value or value.get(key) is None:
        return float(default)
    try:
        return float(value.get(key))
    except (TypeError, ValueError):
        return float(default)


def safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Broad Transfer Matrix",
        "",
        f"Generated: {payload.get('created_utc')}",
        f"Trigger: **{payload.get('trigger_state')}**",
        "",
        f"- Public tasks covered: {summary.get('real_public_task_count')}",
        f"- Aggregate public pass rate: {summary.get('real_public_pass_rate')}",
        f"- STS delta: {summary.get('real_public_sts_delta')}",
        f"- Missing cards: {', '.join(summary.get('missing_cards') or []) or 'none'}",
        f"- No clean student evidence: {', '.join(summary.get('no_clean_student_evidence_cards') or []) or 'none'}",
        f"- Below-floor cards: {', '.join(summary.get('cards_below_floor') or []) or 'none'}",
        f"- Loader-only cards: {', '.join(summary.get('loader_only_cards') or []) or 'none'}",
        f"- No-cheat violations: {summary.get('no_cheat_violation_count')}",
        "",
        "| Card | Report | Public Tasks | Multi | Single | Delta | State | Blockers | Residuals |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        residuals = ", ".join(f"{key}:{value}" for key, value in (row.get("residual_family_counts") or {}).items())
        blockers = ", ".join(str(item) for item in (row.get("clean_evidence_blockers") or []))
        lines.append(
            "| {card} | {report} | {tasks} | {multi} | {single} | {delta} | {state} | {blockers} | {residuals} |".format(
                card=row.get("card_id"),
                report=row.get("selected_report") or row.get("status"),
                tasks=row.get("public_task_count", 0),
                multi=row.get("multi_stream_pass_rate", 0.0),
                single=row.get("single_stream_pass_rate", 0.0),
                delta=row.get("pass_rate_delta", 0.0),
                state=row.get("trigger_state"),
                blockers=blockers or "none",
                residuals=residuals or "none",
            )
        )
    lines.extend(
        [
            "",
            "This matrix is best-per-card calibration truth, not a promotion artifact. Promotion still requires a single clean checkpoint report with sufficient coverage and no regressions.",
            "",
        ]
    )
    return "\n".join(lines)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
