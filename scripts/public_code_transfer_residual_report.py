#!/usr/bin/env python3
"""Summarize public code-transfer calibration residuals without public content.

The report produced here is a control signal for private/source-level repair
only. It groups failures by public card and broad residual category while
keeping public tests, solutions, prompts, and candidate code out of the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

CATEGORIES = [
    "platform_path_normalization",
    "interface_fidelity",
    "return_shape",
    "algorithmic_planning",
    "dependency_runtime_handling",
    "verifier_mismatch",
    "no_admissible_candidate_regression",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-report",
        default="reports/real_code_benchmark_graduation_broad_floor_public4card_calibration_v3.json",
    )
    parser.add_argument(
        "--trace-in",
        default="reports/real_code_benchmark_traces_broad_floor_public4card_calibration_v3.jsonl",
    )
    parser.add_argument(
        "--student-candidate-manifest",
        default="reports/student_code_candidates_private_pressure_private_recovery_broad_floor_train_once_v3.jsonl",
    )
    parser.add_argument("--out", default="reports/public_code_transfer_residual_report.json")
    parser.add_argument("--markdown-out", default="reports/public_code_transfer_residual_report.md")
    args = parser.parse_args()

    report_path = resolve(args.calibration_report)
    trace_path = resolve(args.trace_in)
    candidate_path = resolve(args.student_candidate_manifest)
    calibration = read_json(report_path, {})
    traces = read_jsonl(trace_path)
    student_candidates = read_jsonl(candidate_path)
    payload = build_report(
        calibration,
        traces,
        student_candidates,
        report_path=report_path,
        trace_path=trace_path,
        candidate_path=candidate_path,
    )
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(
    calibration: dict[str, Any],
    traces: list[dict[str, Any]],
    student_candidates: list[dict[str, Any]],
    *,
    report_path: Path,
    trace_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    suites = [row for row in calibration.get("suites", []) if isinstance(row, dict)]
    residuals = [row for row in calibration.get("residuals", []) if isinstance(row, dict)]
    if not residuals:
        for suite in suites:
            residuals.extend(row for row in suite.get("residuals", []) if isinstance(row, dict))

    trace_by_task = task_trace_summary(traces)
    grouped: dict[str, dict[str, Any]] = {}
    aggregate_categories: Counter[str] = Counter()
    aggregate_raw: Counter[str] = Counter()
    hashed_failures: list[dict[str, Any]] = []

    candidate_index = index_candidates(student_candidates)
    availability = no_admissible_candidate_availability(residuals, candidate_index)

    source_cards = ordered_source_cards(suites, residuals)
    for card in source_cards:
        suite = next((row for row in suites if row.get("card_id") == card), {})
        card_residuals = [row for row in residuals if row.get("card_id") == card]
        category_counts: Counter[str] = Counter()
        raw_counts: Counter[str] = Counter()
        samples: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
        for residual in card_residuals:
            raw_type = str(residual.get("type") or "unknown")
            detail = str(residual.get("detail") or "")
            task_id = str(residual.get("task_id") or "")
            category = classify_residual(raw_type, detail)
            raw_counts[raw_type] += 1
            category_counts[category] += 1
            aggregate_categories[category] += 1
            aggregate_raw[raw_type] += 1
            if len(samples[category]) < 8:
                samples[category].append(
                    {
                        "task_id": task_id,
                        "task_hash": stable_hash(f"{card}:{task_id}")[:16],
                        "raw_type": raw_type,
                        "stage": trace_by_task.get(task_id, {}).get("stage"),
                        "detail_summary": summarize_detail(detail),
                    }
                )
            if len(hashed_failures) < 128:
                hashed_failures.append(
                    {
                        "card_id": card,
                        "task_hash": stable_hash(f"{card}:{task_id}")[:16],
                        "category": category,
                        "raw_type": raw_type,
                    }
                )

        grouped[card] = {
            "card_id": card,
            "benchmark_evidence_level": suite.get("benchmark_evidence_level"),
            "case_count": int(suite.get("case_count") or 0),
            "multi_stream_passed": int(suite.get("multi_stream_passed") or 0),
            "multi_stream_pass_rate": safe_float(suite.get("multi_stream_pass_rate")),
            "residual_count": len(card_residuals),
            "category_counts": {category: category_counts.get(category, 0) for category in CATEGORIES},
            "raw_residual_counts": dict(raw_counts),
            "sample_failures": {key: value for key, value in samples.items() if value},
            "dominant_category": category_counts.most_common(1)[0][0] if category_counts else "",
            "next_private_fix_family": private_fix_family(category_counts),
            "adapter_adjusted_no_admissible": availability["by_card"].get(
                card,
                empty_availability_counts(),
            ),
        }

    summary = calibration.get("summary") if isinstance(calibration.get("summary"), dict) else {}
    adjusted_categories = Counter(aggregate_categories)
    adjusted_categories["no_admissible_candidate_regression"] = max(
        0,
        adjusted_categories.get("no_admissible_candidate_regression", 0)
        - availability["summary"]["eligible_candidate_available_tasks"],
    )
    gates = [
        gate("calibration_report_loaded", calibration.get("policy") == "project_theseus_real_code_benchmark_graduation_v1", rel(report_path)),
        gate("trace_loaded", bool(traces), {"path": rel(trace_path), "trace_rows": len(traces)}),
        gate("student_candidate_manifest_loaded", bool(student_candidates), {"path": rel(candidate_path), "rows": len(student_candidates)}),
        gate("source_cards_grouped", all(card in grouped for card in source_cards), source_cards),
        gate("public_content_not_embedded", not public_content_embedded(grouped), "task ids/hashes, counts, stages, and residual labels only"),
        gate("calibration_only", calibration.get("promotion_allowed") is False, calibration.get("public_benchmark_score_claim")),
    ]
    return {
        "policy": "project_theseus_public_code_transfer_residual_report_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
        "calibration_report": rel(report_path),
        "trace_report": rel(trace_path),
        "student_candidate_manifest": rel(candidate_path),
        "summary": {
            "real_public_task_pass_rate": summary.get("real_public_task_pass_rate"),
            "multi_stream_pass_rate": summary.get("multi_stream_pass_rate"),
            "public_task_count": summary.get("public_task_count"),
            "loader_regression_case_count": summary.get("loader_regression_case_count"),
            "total_case_count": summary.get("total_case_count"),
            "student_candidate_count": summary.get("student_candidate_count"),
            "candidate_attempt_count": (summary.get("verification_cascade_summary") or {}).get("candidate_attempt_count")
            if isinstance(summary.get("verification_cascade_summary"), dict)
            else None,
            "dominant_categories": aggregate_categories.most_common(),
            "adapter_adjusted_dominant_categories": adjusted_categories.most_common(),
            "raw_residual_counts": aggregate_raw.most_common(),
            "adapter_adjusted_no_admissible": availability["summary"],
            "cards_with_missing_candidates": [
                card
                for card, row in grouped.items()
                if row["category_counts"].get("no_admissible_candidate_regression", 0) > 0
            ],
            "next_blocker": aggregate_categories.most_common(1)[0][0] if aggregate_categories else "",
            "next_blocker_after_current_adapter": adjusted_categories.most_common(1)[0][0] if adjusted_categories else "",
            "recommended_private_fix_family": recommended_private_fix(aggregate_categories),
            "recommended_private_fix_family_after_current_adapter": recommended_private_fix(adjusted_categories),
            "public_tests_or_solutions_embedded": False,
            "public_prompts_embedded": False,
            "external_inference_calls": 0,
        },
        "cards": grouped,
        "hashed_failure_index": hashed_failures,
        "gates": gates,
        "rules": {
            "public_benchmarks": "calibration-only; public tests and solutions are not emitted",
            "training": "do not train on this report directly; route categories into private/source-level architecture fixes",
            "promotion": "candidate promotion and model growth remain locked unless integrity gates stay GREEN after private repair",
        },
        "external_inference_calls": 0,
    }


def ordered_source_cards(suites: list[dict[str, Any]], residuals: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "source_mbpp",
        "source_evalplus",
        "source_bigcodebench",
        "source_human_eval",
        "source_livecodebench",
    ]
    seen = {
        str(row.get("card_id") or "")
        for row in suites + residuals
        if isinstance(row, dict) and str(row.get("card_id") or "")
    }
    ordered = [card for card in preferred if card in seen]
    ordered.extend(sorted(card for card in seen if card not in preferred))
    return ordered


def index_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for key in (str(row.get("task_id") or ""), str(row.get("source_task_id") or "")):
            if key:
                indexed[key].append(row)
    return indexed


def no_admissible_candidate_availability(
    residuals: list[dict[str, Any]],
    candidate_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_card: dict[str, Counter[str]] = defaultdict(Counter)
    hashed_tasks: list[dict[str, str]] = []
    for residual in residuals:
        if not isinstance(residual, dict):
            continue
        raw_type = str(residual.get("type") or "")
        detail = str(residual.get("detail") or "")
        if classify_residual(raw_type, detail) != "no_admissible_candidate_regression":
            continue
        card = str(residual.get("card_id") or "")
        task_id = str(residual.get("task_id") or "")
        rows = candidate_index.get(task_id, [])
        status = candidate_availability_status(rows)
        by_card[card][status] += 1
        by_card[card]["total_no_admissible_residual_tasks"] += 1
        if len(hashed_tasks) < 128:
            hashed_tasks.append(
                {
                    "card_id": card,
                    "task_hash": stable_hash(f"{card}:{task_id}")[:16],
                    "status": status,
                }
            )

    summary = Counter()
    for counts in by_card.values():
        summary.update(counts)
    return {
        "summary": {
            "total_no_admissible_residual_tasks": summary.get("total_no_admissible_residual_tasks", 0),
            "eligible_candidate_available_tasks": summary.get("eligible_candidate_available", 0),
            "comparator_only_candidate_tasks": summary.get("comparator_only_candidate", 0),
            "residual_stub_only_tasks": summary.get("residual_stub_only", 0),
            "no_manifest_candidate_rows_tasks": summary.get("no_manifest_candidate_rows", 0),
            "true_remaining_no_admissible_tasks": (
                summary.get("comparator_only_candidate", 0)
                + summary.get("residual_stub_only", 0)
                + summary.get("no_manifest_candidate_rows", 0)
            ),
            "score_semantics": "adapter-adjusted candidate availability only; this does not change the public calibration score",
        },
        "by_card": {card: dict(counts) for card, counts in by_card.items()},
        "hashed_task_status": hashed_tasks,
    }


def empty_availability_counts() -> dict[str, int]:
    return {
        "total_no_admissible_residual_tasks": 0,
        "eligible_candidate_available": 0,
        "comparator_only_candidate": 0,
        "residual_stub_only": 0,
        "no_manifest_candidate_rows": 0,
    }


def candidate_availability_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no_manifest_candidate_rows"
    if any(candidate_is_benchmark_eligible(row) for row in rows):
        return "eligible_candidate_available"
    if any(str(row.get("candidate_generation_mode") or "").startswith("student_decoder_no_admissible") for row in rows):
        return "residual_stub_only"
    return "comparator_only_candidate"


def candidate_is_benchmark_eligible(row: dict[str, Any]) -> bool:
    return bool(
        truthy(row.get("benchmark_promotion_eligible"))
        and truthy(row.get("token_level_code_generation_learned"))
        and truthy(row.get("full_body_token_candidate"))
        and truthy(row.get("deterministic_guardrail_passed"))
        and row.get("decoder_contract_verifier_v1_passed") is not False
        and not truthy(row.get("placeholder_scaffold_body"))
        and not truthy(row.get("expression_memory_fallback"))
        and not truthy(row.get("loop_closure_generated"))
        and not truthy(row.get("template_like_candidate"))
    )


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def task_trace_summary(traces: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in traces:
        if row.get("event") != "real_code_candidate_test":
            continue
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        current = out.setdefault(
            task_id,
            {
                "attempts": 0,
                "stage_counts": Counter(),
                "max_reward": 0.0,
                "stage": "",
            },
        )
        current["attempts"] += 1
        stage = str(row.get("verification_stage") or "")
        current["stage_counts"][stage] += 1
        current["max_reward"] = max(float(current["max_reward"]), safe_float(row.get("verification_reward")))
        current["stage"] = stage
    return {
        key: {
            "attempts": value["attempts"],
            "stage_counts": dict(value["stage_counts"]),
            "max_reward": value["max_reward"],
            "stage": value["stage"],
        }
        for key, value in out.items()
    }


def classify_residual(raw_type: str, detail: str) -> str:
    text = f"{raw_type} {detail}".lower()
    if (
        "d:/projecttheseus/tmp" in text
        or "d:\\projecttheseus\\tmp" in text
        or ("can't open file" in text and "theseus_real_code_grad" in text and "projecttheseus/tmp" in text)
    ):
        return "platform_path_normalization"
    if "missing local theseus student checkpoint candidate" in text or "no_candidate" in text:
        return "no_admissible_candidate_regression"
    if "no_admissible" in text or "no admissible" in text:
        return "no_admissible_candidate_regression"
    if "visible_argument" in text or "entry_point" in text or "signature" in text or "interface" in text:
        return "interface_fidelity"
    if "return_shape" in text or "type_handling" in text or "type shape" in text:
        return "return_shape"
    if (
        "external_dependency" in text
        or "unavailable_external_import" in text
        or "runtime_load" in text
        or "runtime_failed" in text
        or "import_or_runtime" in text
        or "psutil" in text
        or "pandas" in text
        or "seaborn" in text
    ):
        return "dependency_runtime_handling"
    if "algorithm_choice" in text or "formula" in text or "missing_required_skeleton" in text:
        return "algorithmic_planning"
    if "intended_behavior_failed" in text or "assertionerror" in text or "edge_case" in text:
        return "verifier_mismatch"
    return "algorithmic_planning"


def private_fix_family(counts: Counter[str]) -> str:
    if not counts:
        return ""
    category = counts.most_common(1)[0][0]
    return {
        "platform_path_normalization": "real_code_benchmark_support.runtime_tmp_dir_cross_platform",
        "interface_fidelity": "typed_interface_skeleton",
        "return_shape": "type_and_return_shape",
        "algorithmic_planning": "algorithmic_planning",
        "dependency_runtime_handling": "admissibility_and_interface",
        "verifier_mismatch": "edge_contract_v2_private_residual_curriculum",
        "no_admissible_candidate_regression": "candidate_floor_v2_private_residual_curriculum",
    }.get(category, "candidate_floor_v2_private_residual_curriculum")


def recommended_private_fix(counts: Counter[str]) -> str:
    return private_fix_family(counts)


def summarize_detail(detail: str) -> str:
    text = " ".join(str(detail or "").split())
    if not text:
        return ""
    replacements = [
        ("Traceback", "traceback"),
        ("AssertionError", "assertion_error"),
        ("beautiful_code_lint_failed:", "lint_failed:"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text[:180]


def public_content_embedded(value: Any) -> bool:
    text = json.dumps(value, sort_keys=True).lower()
    banned = [
        "canonical_solution",
        "hidden_test",
        "tests_source",
        "def solution",
        "candidate_code",
        "\"prompt\"",
    ]
    return any(token in text for token in banned)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    except (OSError, json.JSONDecodeError):
        return rows
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Public Code Transfer Residual Report",
        "",
        f"- State: `{payload.get('trigger_state')}`",
        f"- Calibration: `{payload.get('calibration_report')}`",
        f"- Real public pass rate: `{summary.get('real_public_task_pass_rate')}`",
        f"- Total cases: `{summary.get('total_case_count')}`",
        f"- Next blocker: `{summary.get('next_blocker')}`",
        f"- Recommended private fix family: `{summary.get('recommended_private_fix_family')}`",
        "",
        "## Dominant Categories",
        "",
    ]
    for category, count in summary.get("dominant_categories", []):
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Cards", ""])
    for card, row in payload.get("cards", {}).items():
        lines.append(
            f"### {card}\n"
            f"- Evidence: `{row.get('benchmark_evidence_level')}`\n"
            f"- Passes: `{row.get('multi_stream_passed')}/{row.get('case_count')}` "
            f"rate `{row.get('multi_stream_pass_rate')}`\n"
            f"- Dominant category: `{row.get('dominant_category')}`\n"
            f"- Private fix family: `{row.get('next_private_fix_family')}`"
        )
        for category, count in row.get("category_counts", {}).items():
            if count:
                lines.append(f"  - `{category}`: {count}")
        lines.append("")
    lines.append("Public benchmark content remains calibration-only; this report contains residual categories, counts, stages, task ids/hashes, and no tests/solutions/prompts.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
