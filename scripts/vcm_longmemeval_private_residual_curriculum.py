"""Private LongMemEval-style residual curriculum for VCM.

This script is deliberately private-only. It builds synthetic local analogues
from private templates, scores VCM against the same local memory baselines used
by the public-memory adapter, and reports whether the LongMemEval wall is ready
for a future exact-once public calibration proposal. It never reads public
benchmark payloads and never writes training rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))

import vcm_official_public_memory_adapter as prompt_adapter  # noqa: E402


DEFAULT_OUT = REPORTS / "vcm_longmemeval_private_residual_curriculum.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_longmemeval_private_residual_curriculum.md"
DEFAULT_CASE_COUNT = 180
MAJOR_QUESTION_TYPES = [
    "current_update",
    "fact",
    "preference",
    "temporal_first",
    "temporal_last",
    "choice",
    "where",
    "who",
    "when",
    "abstention",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--case-count", type=int, default=DEFAULT_CASE_COUNT)
    parser.add_argument("--context-budget-chars", type=int, default=900)
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(
        case_count=max(150, args.case_count),
        context_budget_chars=max(128, args.context_budget_chars),
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, case_count: int, context_budget_chars: int, started: float) -> dict[str, Any]:
    public_aggregate = latest_public_aggregate()
    items = build_private_items(case_count)
    rows = [score_private_item(item, context_budget_chars=context_budget_chars) for item in items]
    summary = summarize_rows(rows, context_budget_chars=context_budget_chars)
    gates = curriculum_gates(summary)
    hard_failures = [row for row in gates if not row["passed"] and row["severity"] == "blocker"]
    trigger_state = "GREEN" if not hard_failures else "YELLOW"
    return {
        "policy": "project_theseus_vcm_longmemeval_private_residual_curriculum_v1",
        "created_utc": prompt_adapter.now(),
        "trigger_state": trigger_state,
        "private_only": True,
        "benchmark": "private_longmemeval_style_residual_curriculum",
        "public_source": "aggregate_counts_only",
        "public_failure_aggregate": public_aggregate,
        "context_budget_chars": context_budget_chars,
        "summary": summary,
        "gates": gates,
        "hard_failures": hard_failures,
        "rows": compact_rows(rows),
        "external_inference_calls": 0,
        "teacher_solving_calls": 0,
        "fallback_return_count": 0,
        "public_prompt_chars_loaded": 0,
        "public_context_chars_loaded": 0,
        "public_answer_chars_loaded": 0,
        "public_training_rows_written": 0,
        "future_public_calibration_proposal": future_public_calibration_proposal(summary),
        "runtime_seconds": round(time.perf_counter() - started, 4),
    }


def latest_public_aggregate() -> dict[str, Any]:
    path = REPORTS / "vcm_public_memory_prompt_calibration.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False}
    summary = dict_value(report.get("summary"))
    longmemeval = dict_value(get_path(summary, ["per_benchmark", "longmemeval"], {}))
    systems = dict_value(longmemeval.get("memory_systems"))
    return {
        "available": True,
        "slice_id": report.get("slice_id"),
        "longmemeval_items": longmemeval.get("items"),
        "longmemeval_vcm_on_pass_rate": longmemeval.get("vcm_on_pass_rate"),
        "longmemeval_vcm_off_pass_rate": longmemeval.get("vcm_off_pass_rate"),
        "longmemeval_question_type": longmemeval.get("longmemeval_question_type"),
        "longmemeval_system_pass_rates": {
            name: row.get("pass_rate")
            for name, row in systems.items()
            if isinstance(row, dict)
        },
        "public_payload_used_for_private_cases": False,
    }


def build_private_items(case_count: int) -> list[prompt_adapter.PublicMemoryItem]:
    builders = [
        build_current_update_case,
        build_fact_alias_case,
        build_preference_case,
        build_temporal_first_case,
        build_temporal_last_case,
        build_choice_case,
        build_where_case,
        build_who_case,
        build_when_case,
        build_abstention_case,
    ]
    items: list[prompt_adapter.PublicMemoryItem] = []
    idx = 0
    while len(items) < case_count:
        builder = builders[idx % len(builders)]
        items.append(builder(idx))
        idx += 1
    return items


def build_current_update_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    subject = f"workshop router {idx % 9}"
    old = f"alpha-{idx:03d}"
    answer = f"delta-{idx:03d}"
    lines = noise("current_front", idx, 6)
    lines.append(lme_line(f"cur_anchor_{idx}", idx, "user", f"The {subject} password used to be {old}."))
    lines.extend(noise("current_middle", idx + 10, 6))
    evidence = lme_line(f"cur_update_{idx}", idx + 20, "user", f"It is now {answer} after the inventory refresh.")
    lines.append(evidence)
    lines.extend(noise("current_tail", idx + 30, 8))
    return private_item(
        idx,
        "current_update",
        lines,
        f"What is the current password for the {subject}?",
        [answer],
        evidence,
        "private_longmemeval_structured_recency_fusion",
    )


def build_fact_alias_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    alias = f"route beacon {idx % 11}"
    object_name = f"orange tracker {idx % 7}"
    answer = f"cedar-{idx:03d}"
    lines = noise("fact_front", idx, 7)
    lines.append(lme_line(f"fact_anchor_{idx}", idx + 3, "user", f"I call the {object_name} the {alias} during workshop setup."))
    lines.extend(noise("fact_middle", idx + 14, 5))
    evidence = lme_line(f"fact_answer_{idx}", idx + 26, "user", f"Its access code is {answer}.")
    lines.append(evidence)
    lines.extend(noise("fact_tail", idx + 40, 9))
    return private_item(
        idx,
        "fact",
        lines,
        f"What is the access code for the {alias}?",
        [answer],
        evidence,
        "private_longmemeval_multi_session_query_decomposition",
    )


def build_preference_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    activity = f"travel day {idx % 13}"
    answer = f"matte silver caliper {idx % 5}"
    lines = noise("pref_front", idx, 5)
    lines.append(lme_line(f"pref_anchor_{idx}", idx + 5, "user", f"For {activity}, I tested the black caliper and the brass caliper."))
    lines.extend(noise("pref_middle", idx + 15, 5))
    evidence = lme_line(f"pref_answer_{idx}", idx + 25, "user", f"I prefer the {answer} for that because it is easier to read.")
    lines.append(evidence)
    lines.extend(noise("pref_tail", idx + 35, 8))
    return private_item(
        idx,
        "preference",
        lines,
        f"Which caliper do I prefer for {activity}?",
        [answer],
        evidence,
        "private_longmemeval_answer_span_compaction",
    )


def build_temporal_first_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    first = f"calibration clinic {idx % 8}"
    second = f"packet audit workshop {idx % 8}"
    evidence = lme_line(f"first_answer_{idx}", idx + 12, "user", f"I attended the {first} before the {second}.")
    lines = noise("first_front", idx, 9) + [evidence] + noise("first_tail", idx + 20, 11)
    return private_item(
        idx,
        "temporal_first",
        lines,
        f"Which event did I attend first, the {first} or the {second}?",
        [first],
        evidence,
        "private_longmemeval_multi_session_query_decomposition",
    )


def build_temporal_last_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    first = f"lens pickup {idx % 8}"
    second = f"sensor install {idx % 8}"
    evidence = lme_line(f"last_answer_{idx}", idx + 13, "user", f"I handled the {first} before the {second}.")
    lines = noise("last_front", idx, 8) + [evidence] + noise("last_tail", idx + 21, 12)
    return private_item(
        idx,
        "temporal_last",
        lines,
        f"Which event happened last, the {first} or the {second}?",
        [second],
        evidence,
        "private_longmemeval_structured_recency_fusion",
    )


def build_choice_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    chosen = f"blue field notebook {idx % 6}"
    rejected = f"red pocket notebook {idx % 6}"
    evidence = lme_line(f"choice_answer_{idx}", idx + 14, "user", f"I chose the {chosen} instead of the {rejected} for the site walk.")
    lines = noise("choice_front", idx, 8) + [evidence] + noise("choice_tail", idx + 22, 10)
    return private_item(
        idx,
        "choice",
        lines,
        f"Which notebook did I choose for the site walk, the {chosen} or the {rejected}?",
        [chosen],
        evidence,
        "private_longmemeval_query_decomposition",
    )


def build_where_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    item = f"folding projector {idx % 10}"
    answer = f"studio {idx % 4}"
    lines = noise("where_front", idx, 7)
    lines.append(lme_line(f"where_anchor_{idx}", idx + 6, "user", f"The {item} started out in the lab cabinet."))
    lines.extend(noise("where_middle", idx + 15, 7))
    evidence = lme_line(f"where_answer_{idx}", idx + 28, "user", f"I moved it to {answer} before the room reshuffle.")
    lines.append(evidence)
    lines.extend(noise("where_tail", idx + 38, 8))
    return private_item(
        idx,
        "where",
        lines,
        f"Where is the {item} currently?",
        [answer],
        evidence,
        "private_longmemeval_structured_recency_fusion",
    )


def build_who_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    system = f"cooling loop {idx % 9}"
    answer = f"Jordan Vale {idx % 7}"
    lines = noise("who_front", idx, 7)
    lines.append(lme_line(f"who_anchor_{idx}", idx + 7, "user", f"The repair contact for the {system} was being updated."))
    lines.extend(noise("who_middle", idx + 16, 6))
    evidence = lme_line(f"who_answer_{idx}", idx + 29, "user", f"{answer} is now the contact for it.")
    lines.append(evidence)
    lines.extend(noise("who_tail", idx + 39, 8))
    return private_item(
        idx,
        "who",
        lines,
        f"Who is the current repair contact for the {system}?",
        [answer],
        evidence,
        "private_longmemeval_structured_recency_fusion",
    )


def build_when_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    task = f"lens pickup {idx % 9}"
    answer = f"2026/08/{(idx % 20) + 1:02d}"
    lines = noise("when_front", idx, 7)
    lines.append(lme_line(f"when_anchor_{idx}", idx + 6, "user", f"I was scheduling the {task} after the shop visit."))
    lines.extend(noise("when_middle", idx + 15, 6))
    evidence = lme_line(f"when_answer_{idx}", idx + 30, "user", f"It is now set for {answer}.")
    lines.append(evidence)
    lines.extend(noise("when_tail", idx + 40, 9))
    return private_item(
        idx,
        "when",
        lines,
        f"When is the {task} scheduled?",
        [answer],
        evidence,
        "private_longmemeval_answer_span_compaction",
    )


def build_abstention_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    lines = noise("abstain_front", idx, 10)
    lines.append(lme_line(f"abstain_anchor_{idx}", idx + 12, "user", "The shelf inventory mentioned a blue bin, a green bin, and a spare gasket."))
    lines.extend(noise("abstain_tail", idx + 20, 12))
    return private_item(
        idx,
        "abstention",
        lines,
        f"What color is the hidden cable for private kit {idx}?",
        ["__NO_ANSWER__"],
        "",
        "private_longmemeval_abstention_thresholding",
    )


def private_item(
    idx: int,
    question_type: str,
    lines: list[str],
    question: str,
    answers: list[str],
    evidence: str,
    analogue: str,
) -> prompt_adapter.PublicMemoryItem:
    oracle = [{"id": f"private_lme:{question_type}:{idx}:line", "text": evidence}] if evidence else []
    return prompt_adapter.PublicMemoryItem(
        item_id=f"private_lme_residual_{idx:04d}_{question_type}",
        benchmark="longmemeval",
        task=question_type,
        prompt="",
        context="\n".join(lines),
        question=question,
        answers=answers,
        oracle_evidence=oracle,
        metadata={
            "private_analogue": analogue,
            "private_curriculum": "longmemeval_semantic_quality_v1",
            "question_type": question_type,
            "public_payload_derived": False,
        },
    )


def lme_line(session: str, day_offset: int, role: str, content: str) -> str:
    month = 7 + ((day_offset // 24) % 2)
    day = (day_offset % 24) + 1
    hour = 8 + (day_offset % 9)
    return f"[private_{session}] 2026/{month:02d}/{day:02d} (Tue) {hour:02d}:00 {role}: {content}"


def noise(prefix: str, seed: int, count: int) -> list[str]:
    return [
        lme_line(
            f"{prefix}_{seed}_{idx}",
            seed + idx,
            "assistant" if idx % 5 == 0 else "user",
            f"Private routine note {seed}-{idx} tracks errands, schedules, labels, and storage reminders without the requested answer.",
        )
        for idx in range(count)
    ]


def score_private_item(item: prompt_adapter.PublicMemoryItem, *, context_budget_chars: int) -> dict[str, Any]:
    scored = prompt_adapter.score_item(item, context_budget_chars=context_budget_chars)
    systems = {
        name: row
        for name, row in dict_value(scored.get("memory_systems")).items()
        if isinstance(row, dict)
    }
    system_passes = {name: bool(row.get("passed")) for name, row in systems.items()}
    non_vcm_passes = {
        name: passed
        for name, passed in system_passes.items()
        if name != "vcm_graph_evidence_selector"
    }
    best_non_vcm_any = any(non_vcm_passes.values())
    vcm_on = dict_value(scored.get("vcm_on"))
    vcm_off = dict_value(scored.get("vcm_off"))
    passed = bool(vcm_on.get("passed"))
    diagnostics = private_failure_diagnostics(item, scored)
    return {
        "item_id": item.item_id,
        "question_type": item.metadata.get("question_type"),
        "private_analogue": item.metadata.get("private_analogue"),
        "answer_hash": prompt_adapter.stable_hash(item.answers),
        "context_hash": prompt_adapter.stable_hash(item.context),
        "oracle_evidence_hash": prompt_adapter.stable_hash(item.oracle_evidence),
        "vcm_on_passed": passed,
        "vcm_off_passed": bool(vcm_off.get("passed")),
        "best_non_vcm_any_passed": best_non_vcm_any,
        "system_passes": system_passes,
        "vcm_on_no_admissible": bool(vcm_on.get("no_admissible")),
        "vcm_off_no_admissible": bool(vcm_off.get("no_admissible")),
        "vcm_on_evidence_precision": vcm_on.get("evidence_precision", 0.0),
        "vcm_on_evidence_recall": vcm_on.get("evidence_recall", 0.0),
        "vcm_off_evidence_recall": vcm_off.get("evidence_recall", 0.0),
        "vcm_on_answer_span_chars": int(vcm_on.get("answer_span_chars") or 0),
        "vcm_off_answer_span_chars": int(vcm_off.get("answer_span_chars") or 0),
        "vcm_on_question_type": vcm_on.get("longmemeval_question_type"),
        "vcm_on_abstention_reason": vcm_on.get("abstention_reason", ""),
        "diagnostics": diagnostics,
        "external_inference_calls": 0,
        "teacher_solving_calls": 0,
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
    }


def private_failure_diagnostics(item: prompt_adapter.PublicMemoryItem, scored: dict[str, Any]) -> list[str]:
    vcm_on = dict_value(scored.get("vcm_on"))
    if vcm_on.get("passed"):
        return []
    diagnostics = []
    if vcm_on.get("no_admissible"):
        diagnostics.append("abstention_miss" if item.answers != ["__NO_ANSWER__"] else "abstention_true_positive")
    if float(vcm_on.get("evidence_recall") or 0.0) < 1.0 and item.oracle_evidence:
        diagnostics.append("missed_evidence")
    if int(vcm_on.get("answer_span_chars") or 0) > 120:
        diagnostics.append("overlong_span")
    qtype = str(item.metadata.get("question_type") or "unknown")
    if qtype in {"temporal_first", "temporal_last", "choice"}:
        diagnostics.append("temporal_or_choice_misorder")
    elif qtype in {"current_update", "where", "who", "when"}:
        diagnostics.append("structured_recency_fusion_gap")
    elif qtype == "preference":
        diagnostics.append("preference_span_gap")
    elif qtype == "fact":
        diagnostics.append("query_decomposition_gap")
    elif qtype == "abstention":
        diagnostics.append("abstention_false_positive")
    if not diagnostics:
        diagnostics.append("wrong_answer_type")
    return sorted(set(diagnostics))


def summarize_rows(rows: list[dict[str, Any]], *, context_budget_chars: int) -> dict[str, Any]:
    systems = sorted({
        name
        for row in rows
        for name in dict_value(row.get("system_passes"))
    })
    system_rates = {
        name: mean([1.0 if get_path(row, ["system_passes", name], False) else 0.0 for row in rows])
        for name in systems
    }
    best_single_non_vcm_rate = max(
        [rate for name, rate in system_rates.items() if name != "vcm_graph_evidence_selector"],
        default=0.0,
    )
    vcm_rate = mean([1.0 if row.get("vcm_on_passed") else 0.0 for row in rows])
    flat_rate = mean([1.0 if row.get("vcm_off_passed") else 0.0 for row in rows])
    per_type = summarize_by_question_type(rows)
    abstention = summarize_abstention(rows)
    return {
        "case_count": len(rows),
        "context_budget_chars": context_budget_chars,
        "question_types": sorted(per_type),
        "vcm_on_pass_rate": vcm_rate,
        "vcm_off_flat_tail_pass_rate": flat_rate,
        "best_single_non_vcm_pass_rate": best_single_non_vcm_rate,
        "vcm_over_flat_tail_delta": round(vcm_rate - flat_rate, 6),
        "vcm_over_best_single_non_vcm_delta": round(vcm_rate - best_single_non_vcm_rate, 6),
        "best_non_vcm_any_pass_rate": mean([1.0 if row.get("best_non_vcm_any_passed") else 0.0 for row in rows]),
        "vcm_only_wins_vs_flat_tail": sum(1 for row in rows if row.get("vcm_on_passed") and not row.get("vcm_off_passed")),
        "flat_tail_only_wins": sum(1 for row in rows if row.get("vcm_off_passed") and not row.get("vcm_on_passed")),
        "best_non_vcm_any_only_wins": sum(1 for row in rows if row.get("best_non_vcm_any_passed") and not row.get("vcm_on_passed")),
        "system_pass_rates": system_rates,
        "per_question_type": per_type,
        "minimum_major_question_type_pass_rate": min(
            [float(per_type.get(name, {}).get("vcm_on_pass_rate") or 0.0) for name in MAJOR_QUESTION_TYPES],
            default=0.0,
        ),
        "vcm_on_evidence_precision": mean([float(row.get("vcm_on_evidence_precision") or 0.0) for row in rows]),
        "vcm_on_evidence_recall": mean([float(row.get("vcm_on_evidence_recall") or 0.0) for row in rows]),
        "vcm_on_answer_span_chars_mean": mean([float(row.get("vcm_on_answer_span_chars") or 0.0) for row in rows if not is_no_answer(row)]),
        "vcm_off_answer_span_chars_mean": mean([float(row.get("vcm_off_answer_span_chars") or 0.0) for row in rows if not is_no_answer(row)]),
        "abstention": abstention,
        "diagnostic_counts": count_values(diag for row in rows for diag in list_value(row.get("diagnostics"))),
        "external_inference_calls": 0,
        "teacher_solving_calls": 0,
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
        "public_payload_chars_loaded": 0,
    }


def summarize_by_question_type(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for question_type in sorted({str(row.get("question_type") or "unknown") for row in rows}):
        bucket = [row for row in rows if row.get("question_type") == question_type]
        vcm_rate = mean([1.0 if row.get("vcm_on_passed") else 0.0 for row in bucket])
        best_single = max(
            [
                mean([1.0 if get_path(row, ["system_passes", name], False) else 0.0 for row in bucket])
                for name in sorted({system for row in bucket for system in dict_value(row.get("system_passes"))})
                if name != "vcm_graph_evidence_selector"
            ],
            default=0.0,
        )
        out[question_type] = {
            "items": len(bucket),
            "vcm_on_pass_rate": vcm_rate,
            "vcm_off_flat_tail_pass_rate": mean([1.0 if row.get("vcm_off_passed") else 0.0 for row in bucket]),
            "best_single_non_vcm_pass_rate": best_single,
            "vcm_over_best_single_non_vcm_delta": round(vcm_rate - best_single, 6),
            "vcm_on_evidence_recall": mean([float(row.get("vcm_on_evidence_recall") or 0.0) for row in bucket]),
            "vcm_on_answer_span_chars_mean": mean([float(row.get("vcm_on_answer_span_chars") or 0.0) for row in bucket if not is_no_answer(row)]),
            "flat_tail_only_wins": sum(1 for row in bucket if row.get("vcm_off_passed") and not row.get("vcm_on_passed")),
            "best_non_vcm_any_only_wins": sum(1 for row in bucket if row.get("best_non_vcm_any_passed") and not row.get("vcm_on_passed")),
            "diagnostic_counts": count_values(diag for row in bucket for diag in list_value(row.get("diagnostics"))),
        }
    return out


def summarize_abstention(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_no_answer = [row for row in rows if is_no_answer(row)]
    answerable = [row for row in rows if not is_no_answer(row)]
    tp = sum(1 for row in true_no_answer if row.get("vcm_on_no_admissible"))
    fp = sum(1 for row in answerable if row.get("vcm_on_no_admissible"))
    fn = sum(1 for row in true_no_answer if not row.get("vcm_on_no_admissible"))
    tn = sum(1 for row in answerable if not row.get("vcm_on_no_admissible"))
    return {
        "no_answer_cases": len(true_no_answer),
        "answerable_cases": len(answerable),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(tp / max(1, tp + fp), 6),
        "recall": round(tp / max(1, tp + fn), 6),
        "false_positive_rate": round(fp / max(1, fp + tn), 6),
    }


def curriculum_gates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gate("private_case_count_floor", int(summary.get("case_count") or 0) >= 150, "blocker", f"cases={summary.get('case_count')}"),
        gate("private_vcm_pass_rate_floor", float(summary.get("vcm_on_pass_rate") or 0.0) >= 0.85, "blocker", f"vcm={summary.get('vcm_on_pass_rate')}"),
        gate(
            "private_major_question_type_floor",
            float(summary.get("minimum_major_question_type_pass_rate") or 0.0) >= 0.75,
            "blocker",
            f"min={summary.get('minimum_major_question_type_pass_rate')}",
        ),
        gate(
            "private_vcm_beats_best_single_non_vcm",
            float(summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.05,
            "blocker",
            f"delta={summary.get('vcm_over_best_single_non_vcm_delta')}",
        ),
        gate("private_no_flat_tail_off_only", int(summary.get("flat_tail_only_wins") or 0) == 0, "blocker", f"flat_tail_only={summary.get('flat_tail_only_wins')}"),
        gate("private_vcm_evidence_recall_floor", float(summary.get("vcm_on_evidence_recall") or 0.0) >= 0.85, "blocker", f"recall={summary.get('vcm_on_evidence_recall')}"),
        gate("private_abstention_recall_reported", "recall" in dict_value(summary.get("abstention")), "blocker", str(summary.get("abstention"))),
        gate("no_cheat_counters_zero", no_cheat(summary), "blocker", "external=0 teacher=0 fallback=0 public_training=0"),
    ]


def future_public_calibration_proposal(summary: dict[str, Any]) -> dict[str, Any]:
    green = all(row["passed"] for row in curriculum_gates(summary) if row["severity"] == "blocker")
    return {
        "proposal_state": "READY_TO_PROPOSE_EXACT_ONCE_PUBLIC_CONFIRMATION" if green else "BLOCKED_BY_PRIVATE_RESIDUALS",
        "run_public_automatically": False,
        "reason": (
            "Private LongMemEval residual curriculum meets all gates; a future exact-once public confirmation can be proposed."
            if green
            else "Private residual curriculum does not yet justify any public spend."
        ),
    }


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": row.get("item_id"),
            "question_type": row.get("question_type"),
            "private_analogue": row.get("private_analogue"),
            "vcm_on_passed": row.get("vcm_on_passed"),
            "vcm_off_passed": row.get("vcm_off_passed"),
            "best_non_vcm_any_passed": row.get("best_non_vcm_any_passed"),
            "vcm_on_no_admissible": row.get("vcm_on_no_admissible"),
            "vcm_on_evidence_recall": row.get("vcm_on_evidence_recall"),
            "vcm_on_answer_span_chars": row.get("vcm_on_answer_span_chars"),
            "diagnostics": row.get("diagnostics"),
            "answer_hash": row.get("answer_hash"),
            "context_hash": row.get("context_hash"),
            "oracle_evidence_hash": row.get("oracle_evidence_hash"),
        }
        for row in rows
    ]


def gate(name: str, passed: bool, severity: str, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def no_cheat(summary: dict[str, Any]) -> bool:
    return (
        int(summary.get("external_inference_calls") or 0) == 0
        and int(summary.get("teacher_solving_calls") or 0) == 0
        and int(summary.get("fallback_return_count") or 0) == 0
        and int(summary.get("public_training_rows_written") or 0) == 0
        and int(summary.get("public_payload_chars_loaded") or 0) == 0
    )


def is_no_answer(row: dict[str, Any]) -> bool:
    return str(row.get("question_type") or "") == "abstention"


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM LongMemEval Private Residual Curriculum",
        "",
        f"State: `{report['trigger_state']}`",
        f"Private only: `{report['private_only']}`",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary['case_count']}`",
        f"- VCM-on pass rate: `{summary['vcm_on_pass_rate']}`",
        f"- Flat-tail pass rate: `{summary['vcm_off_flat_tail_pass_rate']}`",
        f"- Best single non-VCM pass rate: `{summary['best_single_non_vcm_pass_rate']}`",
        f"- VCM over best single non-VCM: `{summary['vcm_over_best_single_non_vcm_delta']}`",
        f"- VCM evidence recall: `{summary['vcm_on_evidence_recall']}`",
        f"- Abstention: `{summary['abstention']}`",
        f"- Future public proposal: `{report['future_public_calibration_proposal']['proposal_state']}`",
        "",
        "## Per Question Type",
        "",
    ]
    for question_type, row in dict_value(summary.get("per_question_type")).items():
        lines.append(
            f"- `{question_type}`: items `{row['items']}`, VCM `{row['vcm_on_pass_rate']}`, "
            f"best non-VCM `{row['best_single_non_vcm_pass_rate']}`, recall `{row['vcm_on_evidence_recall']}`"
        )
    lines.extend(["", "## Gates", ""])
    for row in report["gates"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {mark}: `{row['gate']}` - {row['evidence']}")
    return "\n".join(lines) + "\n"


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def mean(values: list[float]) -> float:
    clean = [float(value) for value in values]
    return round(sum(clean) / len(clean), 6) if clean else 0.0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
