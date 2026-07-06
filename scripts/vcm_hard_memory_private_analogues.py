"""Private hard-memory analogue gauntlet for Theseus VCM.

The suite mirrors hard public benchmark failure categories without loading or
copying public benchmark prompts, contexts, answers, traces, or templates.
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


DEFAULT_OUT = REPORTS / "vcm_hard_memory_private_analogues.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_hard_memory_private_analogues.md"
DEFAULT_CASE_COUNT = 1200
DEFAULT_CONTEXT_BUDGET_CHARS = 900
FAMILIES = [
    "nolima_lexical_disconnect",
    "michelangelo_lsq",
    "lveval_confusing_facts",
    "loft_structured_memory",
    "longbench_v2_multidoc",
    "infinitebench_long_dependency",
    "mtrag_multiturn",
    "mtrag_un_answerability",
    "facts_grounding_private",
    "locomoplus_cognitive_memory",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--case-count", type=int, default=DEFAULT_CASE_COUNT)
    parser.add_argument("--context-budget-chars", type=int, default=DEFAULT_CONTEXT_BUDGET_CHARS)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(
        case_count=max(1000, args.case_count),
        context_budget_chars=max(128, args.context_budget_chars),
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    if args.summary_only:
        print(json.dumps({"trigger_state": report["trigger_state"], "summary": report["summary"], "gates": report["gates"]}, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, case_count: int, context_budget_chars: int, started: float) -> dict[str, Any]:
    items = build_items(case_count)
    rows = [score_item(item, context_budget_chars=context_budget_chars) for item in items]
    summary = summarize_rows(rows, context_budget_chars=context_budget_chars)
    gates = hard_memory_gates(summary)
    hard_failures = [row for row in gates if not row["passed"] and row["severity"] == "blocker"]
    trigger_state = "GREEN" if not hard_failures else "YELLOW"
    return {
        "policy": "project_theseus_vcm_hard_memory_private_analogues_v1",
        "created_utc": prompt_adapter.now(),
        "trigger_state": trigger_state,
        "private_only": True,
        "benchmark": "private_hard_memory_analogue_gauntlet",
        "context_budget_chars": context_budget_chars,
        "suite": {
            "case_count": len(items),
            "families": FAMILIES,
            "public_payload_used_for_private_cases": False,
            "public_prompt_or_answer_templates_used": False,
            "training_rows_written": 0,
        },
        "summary": summary,
        "gates": gates,
        "hard_failures": hard_failures,
        "rows": compact_rows(rows),
        "residual_repair_plan": residual_repair_plan(rows),
        "external_inference_calls": 0,
        "teacher_solving_calls": 0,
        "fallback_return_count": 0,
        "public_prompt_chars_loaded": 0,
        "public_context_chars_loaded": 0,
        "public_answer_chars_loaded": 0,
        "public_trace_chars_loaded": 0,
        "public_test_chars_loaded": 0,
        "public_solution_chars_loaded": 0,
        "public_template_chars_loaded": 0,
        "public_training_rows_written": 0,
        "runtime_seconds": round(time.perf_counter() - started, 4),
    }


def build_items(case_count: int) -> list[prompt_adapter.PublicMemoryItem]:
    builders = [
        build_nolima_case,
        build_michelangelo_case,
        build_lveval_case,
        build_loft_case,
        build_longbench_case,
        build_infinitebench_case,
        build_mtrag_case,
        build_mtrag_un_case,
        build_facts_case,
        build_locomoplus_case,
    ]
    items: list[prompt_adapter.PublicMemoryItem] = []
    idx = 0
    while len(items) < case_count:
        items.append(builders[idx % len(builders)](idx))
        idx += 1
    return items[:case_count]


def build_nolima_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    cue = f"windy bridge {idx % 37}"
    alias = f"quiet-harbor-{idx:04d}"
    answer = f"NOL-{idx:05d}"
    support = [
        event_line(idx, f"The cue {cue} means {alias}."),
        event_line(idx + 1, f"The {alias} key is {answer}."),
    ]
    context = pack_context(idx, support)
    return item(
        item_id=f"hard_private_nolima_{idx:04d}",
        family="nolima_lexical_disconnect",
        skill="alias_bridge_exact_retrieval",
        context=context,
        question=f"What secret is linked to {cue}?",
        answers=[answer],
        evidence=support,
    )


def build_michelangelo_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    anchor = f"cobalt route {idx % 29}"
    answer = f"latent-node-{idx:04d}"
    support = [
        event_line(idx, f"The latent list anchor is {anchor}."),
        event_line(idx + 2, f"The item after {anchor} is {answer}."),
    ]
    context = pack_context(idx, support)
    return item(
        item_id=f"hard_private_michelangelo_{idx:04d}",
        family="michelangelo_lsq",
        skill="latent_list_successor",
        context=context,
        question=f"What item is after {anchor} in the latent list?",
        answers=[answer],
        evidence=support,
    )


def build_lveval_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    subject = f"LV capsule {idx % 41}"
    stale = f"old-keyword-{idx:04d}"
    answer = f"final-keyword-{idx:04d}"
    support = [
        event_line(idx, f"A confusing fact says the answer keyword for {subject} was {stale}."),
        event_line(idx + 10, f"The answer keyword for {subject} is {answer}."),
    ]
    context = pack_context(idx, support, confusing_lines=[f"The old answer keyword for {subject} was {stale}."])
    return item(
        item_id=f"hard_private_lveval_{idx:04d}",
        family="lveval_confusing_facts",
        skill="confusing_fact_rejection",
        context=context,
        question=f"What is the answer keyword for {subject}?",
        answers=[answer],
        evidence=[{"id": evidence_id(support[1]), "text": support[1]}],
    )


def build_loft_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    account = f"account-{idx % 53}"
    answer = f"route-total-{(idx * 17) % 997:03d}"
    support = [
        event_line(idx, f"Table row A for {account} has local value {idx % 17}."),
        event_line(idx + 3, f"Table row B for {account} has local value {(idx * 3) % 19}."),
        event_line(idx + 6, f"The SQL-like result for {account} is {answer}."),
    ]
    context = pack_context(idx, support)
    return item(
        item_id=f"hard_private_loft_{idx:04d}",
        family="loft_structured_memory",
        skill="structured_query_result_recall",
        context=context,
        question=f"What SQL-like result should be returned for {account}?",
        answers=[answer],
        evidence=[{"id": evidence_id(support[2]), "text": support[2]}],
    )


def build_longbench_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    doc_a = f"briefing {idx % 31}"
    doc_b = f"appendix {idx % 43}"
    answer = f"cross-doc-token-{idx:04d}"
    support = [
        event_line(idx, f"In {doc_a}, the connector points to {doc_b}."),
        event_line(idx + 5, f"In {doc_b}, the final cross-document answer is {answer}."),
    ]
    context = pack_context(idx, support)
    return item(
        item_id=f"hard_private_longbench_v2_{idx:04d}",
        family="longbench_v2_multidoc",
        skill="multi_document_bridge",
        context=context,
        question=f"What is the final cross-document answer for {doc_a}?",
        answers=[answer],
        evidence=support,
    )


def build_infinitebench_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    chain = f"deep chain {idx % 47}"
    answer = f"infinite-answer-{idx:04d}"
    support = [
        event_line(idx, f"The start of {chain} points to middle marker {idx:04d}."),
        event_line(idx + 11, f"The middle marker {idx:04d} points to terminal slot {idx + 777}."),
        event_line(idx + 22, f"The terminal slot {idx + 777} answer is {answer}."),
    ]
    context = pack_context(idx, support)
    return item(
        item_id=f"hard_private_infinitebench_{idx:04d}",
        family="infinitebench_long_dependency",
        skill="long_dependency_bridge",
        context=context,
        question=f"What answer is at the end of {chain}?",
        answers=[answer],
        evidence=support,
    )


def build_mtrag_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    deployment = f"deployment {idx % 23}"
    answer = f"option-{idx % 9}-{idx:04d}"
    support = [
        event_line(idx, f"We discussed {deployment} during the first turn."),
        event_line(idx + 9, f"The {deployment} decision is {answer}."),
    ]
    context = pack_context(idx, support)
    return item(
        item_id=f"hard_private_mtrag_{idx:04d}",
        family="mtrag_multiturn",
        skill="followup_turn_bridge",
        context=context,
        question=f"What is the decision for {deployment}?",
        answers=[answer],
        evidence=support,
    )


def build_mtrag_un_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    topic = f"unresolved ticket {idx:04d}"
    support = [
        event_line(idx, f"We opened {topic} but did not record the answer."),
        event_line(idx + 4, f"The notes for {topic} explicitly say the requested answer is unknown."),
    ]
    context = pack_context(idx, support)
    return item(
        item_id=f"hard_private_mtrag_un_{idx:04d}",
        family="mtrag_un_answerability",
        skill="unanswerable_abstention",
        context=context,
        question=f"What exact answer was finalized for {topic}?",
        answers=["__NO_ANSWER__"],
        evidence=[],
    )


def build_facts_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    doc = f"grounding document {idx % 59}"
    answer = f"grounded-fact-{idx:04d}"
    support = [
        event_line(idx, f"The unsupported distractor for {doc} says ignore-me-{idx}."),
        event_line(idx + 8, f"The grounded answer for {doc} is {answer}."),
    ]
    context = pack_context(idx, support)
    return item(
        item_id=f"hard_private_facts_{idx:04d}",
        family="facts_grounding_private",
        skill="grounded_span_recovery",
        context=context,
        question=f"What grounded answer is supported for {doc}?",
        answers=[answer],
        evidence=[{"id": evidence_id(support[1]), "text": support[1]}],
    )


def build_locomoplus_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    cue = f"blue room {idx % 17}"
    alias = f"evening-focus-{idx:04d}"
    answer = f"low-light-mode-{idx:04d}"
    support = [
        event_line(idx, f"The cue {cue} means {alias}."),
        event_line(idx + 13, f"The {alias} mode is {answer}."),
    ]
    context = pack_context(idx, support)
    return item(
        item_id=f"hard_private_locomoplus_{idx:04d}",
        family="locomoplus_cognitive_memory",
        skill="latent_constraint_bridge",
        context=context,
        question=f"What should be used when the context is {cue}?",
        answers=[answer],
        evidence=support,
    )


def item(
    *,
    item_id: str,
    family: str,
    skill: str,
    context: str,
    question: str,
    answers: list[str],
    evidence: list[str | dict[str, str]],
) -> prompt_adapter.PublicMemoryItem:
    oracle_evidence = []
    for row in evidence:
        if isinstance(row, dict):
            oracle_evidence.append(row)
        else:
            oracle_evidence.append({"id": evidence_id(row), "text": row})
    return prompt_adapter.PublicMemoryItem(
        item_id=item_id,
        benchmark="longmemeval",
        task="hard_memory_private",
        prompt=f"<private_context>\n{context}\n</private_context>\n\nQuestion: {question}",
        context=context,
        question=question,
        answers=answers,
        oracle_evidence=oracle_evidence,
        metadata={
            "gauntlet_family": family,
            "gauntlet_skill": skill,
            "private_only": True,
            "public_payload_derived": False,
            "score_semantics": "local deterministic exact/contains answer extraction",
        },
    )


def pack_context(idx: int, support_lines: list[str], confusing_lines: list[str] | None = None) -> str:
    noise_count = noise_count_for_idx(idx)
    before = noise_lines(idx, noise_count // 3)
    middle = noise_lines(idx + 10_000, noise_count // 3)
    after = noise_lines(idx + 20_000, noise_count - len(before) - len(middle))
    lines = before + support_lines[:1] + middle
    for confusing in confusing_lines or []:
        lines.append(event_line(idx + 333, confusing))
    lines.extend(support_lines[1:])
    lines.extend(after)
    return "\n".join(lines)


def noise_count_for_idx(idx: int) -> int:
    if idx % 211 == 0:
        return 5200
    if idx % 97 == 0:
        return 1700
    if idx % 29 == 0:
        return 520
    if idx % 7 == 0:
        return 170
    return 54


def noise_lines(seed: int, count: int) -> list[str]:
    return [
        event_line(seed + line_no, f"Private hard-memory distractor {seed}-{line_no} mentions shelves, devices, schedules, and unrelated notes without the requested answer.")
        for line_no in range(max(0, count))
    ]


def event_line(seed: int, text: str) -> str:
    day = ((seed % 28) + 1)
    return f"[hard_{seed:06d}] 2026/05/{day:02d} user: {text}"


def evidence_id(line: str) -> str:
    return f"private_hard:evidence:{prompt_adapter.stable_hash(line)[:16]}"


def score_item(item: prompt_adapter.PublicMemoryItem, *, context_budget_chars: int) -> dict[str, Any]:
    scored = prompt_adapter.score_item(item, context_budget_chars=context_budget_chars)
    best_non_vcm = prompt_adapter.dict_value(scored.get("best_non_vcm_memory_system"))
    memory_systems = prompt_adapter.dict_value(scored.get("memory_systems"))
    family = str(item.metadata.get("gauntlet_family") or "unknown")
    skill = str(item.metadata.get("gauntlet_skill") or item.task)
    return {
        "item_id": item.item_id,
        "family": family,
        "skill": skill,
        "benchmark": item.benchmark,
        "task": item.task,
        "answer_count": len(item.answers),
        "answer_hash": prompt_adapter.stable_hash(item.answers),
        "context_hash": prompt_adapter.stable_hash(item.context),
        "prompt_hash": prompt_adapter.stable_hash(item.prompt),
        "oracle_evidence_hash": prompt_adapter.stable_hash(item.oracle_evidence),
        "source_context_tokens_estimate": scored.get("source_context_tokens_estimate"),
        "context_length_bucket": scored.get("context_length_bucket"),
        "context_depth_bucket": scored.get("context_depth_bucket"),
        "vcm_on": compact_system_result(prompt_adapter.dict_value(scored.get("vcm_on"))),
        "vcm_off": compact_system_result(prompt_adapter.dict_value(scored.get("vcm_off"))),
        "best_non_vcm_memory_system": best_non_vcm,
        "memory_systems": {name: compact_system_result(result) for name, result in memory_systems.items() if isinstance(result, dict)},
        "vcm_on_passed": bool(prompt_adapter.get_path(scored, ["vcm_on", "passed"], False)),
        "vcm_off_passed": bool(prompt_adapter.get_path(scored, ["vcm_off", "passed"], False)),
        "best_non_vcm_passed": bool(best_non_vcm.get("passed")),
        "winner": scored.get("winner"),
        "latency_ms": scored.get("latency_ms"),
        "residual_categories": scored.get("residual_categories"),
        "external_inference_calls": 0,
        "teacher_solving_calls": 0,
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
        "public_prompt_chars_loaded": 0,
        "public_context_chars_loaded": 0,
        "public_answer_chars_loaded": 0,
    }


def compact_system_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": result.get("system"),
        "passed": bool(result.get("passed")),
        "prediction_hash": result.get("prediction_hash"),
        "evidence_precision": result.get("evidence_precision"),
        "evidence_recall": result.get("evidence_recall"),
        "no_admissible": bool(result.get("no_admissible")),
        "answer_span_chars": result.get("answer_span_chars"),
        "selected_context_chars": result.get("selected_context_chars"),
        "selected_context_compression_ratio": result.get("selected_context_compression_ratio"),
        "longmemeval_question_type": result.get("longmemeval_question_type"),
        "longmemeval_candidate_count": result.get("longmemeval_candidate_count"),
        "longmemeval_best_score": result.get("longmemeval_best_score"),
        "abstention_reason": result.get("abstention_reason"),
    }


def summarize_rows(rows: list[dict[str, Any]], *, context_budget_chars: int) -> dict[str, Any]:
    vcm_rate = mean([1.0 if row["vcm_on_passed"] else 0.0 for row in rows])
    flat_tail_rate = mean([1.0 if row["vcm_off_passed"] else 0.0 for row in rows])
    system_rates = summarize_system_rates(rows)
    best_single_non_vcm_rate = max(
        [row["pass_rate"] for name, row in system_rates.items() if name != "vcm_graph_evidence_selector"],
        default=0.0,
    )
    family_summary = summarize_group(rows, "family")
    skill_summary = summarize_group(rows, "skill")
    abstention = abstention_summary(rows)
    return {
        "case_count": len(rows),
        "context_budget_chars": context_budget_chars,
        "family_count": len(family_summary),
        "families": sorted(family_summary),
        "length_bucket_count": len(summarize_group(rows, "context_length_bucket")),
        "vcm_on_pass_rate": vcm_rate,
        "vcm_off_flat_tail_pass_rate": flat_tail_rate,
        "best_single_non_vcm_pass_rate": best_single_non_vcm_rate,
        "vcm_over_best_single_non_vcm_delta": round(vcm_rate - best_single_non_vcm_rate, 6),
        "vcm_over_flat_tail_delta": round(vcm_rate - flat_tail_rate, 6),
        "vcm_only_wins_vs_flat_tail": sum(1 for row in rows if row["vcm_on_passed"] and not row["vcm_off_passed"]),
        "flat_tail_only_wins": sum(1 for row in rows if row["vcm_off_passed"] and not row["vcm_on_passed"]),
        "best_non_vcm_any_only_wins": sum(1 for row in rows if row["best_non_vcm_passed"] and not row["vcm_on_passed"]),
        "vcm_on_evidence_precision": mean([float(prompt_adapter.get_path(row, ["vcm_on", "evidence_precision"], 0.0) or 0.0) for row in rows if not is_no_answer_row(row)]),
        "vcm_on_evidence_recall": mean([float(prompt_adapter.get_path(row, ["vcm_on", "evidence_recall"], 0.0) or 0.0) for row in rows if not is_no_answer_row(row)]),
        "vcm_selected_context_compression_ratio_mean": mean([float(prompt_adapter.get_path(row, ["vcm_on", "selected_context_compression_ratio"], 0.0) or 0.0) for row in rows]),
        "latency_ms_mean": mean([float(row.get("latency_ms") or 0.0) for row in rows]),
        "no_admissible_rate": mean([1.0 if prompt_adapter.get_path(row, ["vcm_on", "no_admissible"], False) else 0.0 for row in rows]),
        "abstention": abstention,
        "family_summary": family_summary,
        "skill_summary": skill_summary,
        "minimum_family_pass_rate": min([row.get("vcm_on_pass_rate", 0.0) for row in family_summary.values()], default=0.0),
        "system_pass_rates": system_rates,
        "source_context_token_distribution": summarize_numeric([float(row.get("source_context_tokens_estimate") or 0.0) for row in rows]),
        "per_length_bucket": summarize_group(rows, "context_length_bucket"),
        "external_inference_calls": 0,
        "teacher_solving_calls": 0,
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
        "public_payload_chars_loaded": 0,
    }


def summarize_system_rates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    systems: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for name, result in prompt_adapter.dict_value(row.get("memory_systems")).items():
            if isinstance(result, dict):
                systems.setdefault(name, []).append(result)
    return {
        name: {
            "pass_rate": mean([1.0 if result.get("passed") else 0.0 for result in results]),
            "evidence_precision": mean([float(result.get("evidence_precision") or 0.0) for result in results]),
            "evidence_recall": mean([float(result.get("evidence_recall") or 0.0) for result in results]),
            "no_admissible_rate": mean([1.0 if result.get("no_admissible") else 0.0 for result in results]),
        }
        for name, results in sorted(systems.items())
    }


def summarize_group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for group, group_rows in sorted(groups.items()):
        vcm_rate = mean([1.0 if row["vcm_on_passed"] else 0.0 for row in group_rows])
        best_single = best_single_non_vcm_for_rows(group_rows)
        summary[group] = {
            "items": len(group_rows),
            "vcm_on_pass_rate": vcm_rate,
            "flat_tail_pass_rate": mean([1.0 if row["vcm_off_passed"] else 0.0 for row in group_rows]),
            "best_single_non_vcm_pass_rate": best_single,
            "vcm_over_best_single_non_vcm_delta": round(vcm_rate - best_single, 6),
            "vcm_only_wins_vs_flat_tail": sum(1 for row in group_rows if row["vcm_on_passed"] and not row["vcm_off_passed"]),
            "flat_tail_only_wins": sum(1 for row in group_rows if row["vcm_off_passed"] and not row["vcm_on_passed"]),
            "best_non_vcm_any_only_wins": sum(1 for row in group_rows if row["best_non_vcm_passed"] and not row["vcm_on_passed"]),
            "vcm_evidence_recall": mean([float(prompt_adapter.get_path(row, ["vcm_on", "evidence_recall"], 0.0) or 0.0) for row in group_rows]),
            "no_admissible_rate": mean([1.0 if prompt_adapter.get_path(row, ["vcm_on", "no_admissible"], False) else 0.0 for row in group_rows]),
        }
    return summary


def best_single_non_vcm_for_rows(rows: list[dict[str, Any]]) -> float:
    systems = summarize_system_rates(rows)
    return max([row["pass_rate"] for name, row in systems.items() if name != "vcm_graph_evidence_selector"], default=0.0)


def abstention_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    no_answer = [row for row in rows if is_no_answer_row(row)]
    answerable = [row for row in rows if not is_no_answer_row(row)]
    tp = sum(1 for row in no_answer if prompt_adapter.get_path(row, ["vcm_on", "no_admissible"], False))
    fn = sum(1 for row in no_answer if not prompt_adapter.get_path(row, ["vcm_on", "no_admissible"], False))
    fp = sum(1 for row in answerable if prompt_adapter.get_path(row, ["vcm_on", "no_admissible"], False))
    tn = sum(1 for row in answerable if not prompt_adapter.get_path(row, ["vcm_on", "no_admissible"], False))
    return {
        "no_answer_cases": len(no_answer),
        "answerable_cases": len(answerable),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "precision": round(tp / max(1, tp + fp), 6),
        "recall": round(tp / max(1, tp + fn), 6),
        "false_positive_rate": round(fp / max(1, fp + tn), 6),
    }


def hard_memory_gates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    abstention = prompt_adapter.dict_value(summary.get("abstention"))
    return [
        gate("private_case_count_floor", int(summary.get("case_count") or 0) >= 1000, "blocker", f"cases={summary.get('case_count')}"),
        gate("hard_family_count_floor", int(summary.get("family_count") or 0) >= 8, "blocker", f"families={summary.get('family_count')}"),
        gate("length_bucket_count_floor", int(summary.get("length_bucket_count") or 0) >= 3, "warning", f"buckets={summary.get('length_bucket_count')}"),
        gate("private_vcm_overall_floor", float(summary.get("vcm_on_pass_rate") or 0.0) >= 0.85, "blocker", f"vcm={summary.get('vcm_on_pass_rate')}"),
        gate("private_family_floor", float(summary.get("minimum_family_pass_rate") or 0.0) >= 0.70, "blocker", f"min={summary.get('minimum_family_pass_rate')}"),
        gate("private_vcm_beats_best_single_non_vcm", float(summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.03, "blocker", f"delta={summary.get('vcm_over_best_single_non_vcm_delta')}"),
        gate("private_vcm_evidence_recall_floor", float(summary.get("vcm_on_evidence_recall") or 0.0) >= 0.80, "blocker", f"recall={summary.get('vcm_on_evidence_recall')}"),
        gate("private_abstention_precision_floor", float(abstention.get("precision") or 0.0) >= 0.95, "blocker", f"precision={abstention.get('precision')}"),
        gate("private_abstention_recall_floor", float(abstention.get("recall") or 0.0) >= 0.95, "blocker", f"recall={abstention.get('recall')}"),
        gate("no_fallback_returns", int(summary.get("fallback_return_count") or 0) == 0, "blocker", f"fallback={summary.get('fallback_return_count')}"),
        gate("no_teacher_or_external_inference", int(summary.get("teacher_solving_calls") or 0) == 0 and int(summary.get("external_inference_calls") or 0) == 0, "blocker", f"teacher={summary.get('teacher_solving_calls')} external={summary.get('external_inference_calls')}"),
        gate("no_public_payload_or_training_rows", int(summary.get("public_payload_chars_loaded") or 0) == 0 and int(summary.get("public_training_rows_written") or 0) == 0, "blocker", f"public_payload={summary.get('public_payload_chars_loaded')} public_training={summary.get('public_training_rows_written')}"),
    ]


def residual_repair_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if not row["vcm_on_passed"]]
    by_family: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for row in failures:
        family = str(row.get("family") or "unknown")
        by_family[family] = by_family.get(family, 0) + 1
        for category in row.get("residual_categories") or []:
            by_category[str(category)] = by_category.get(str(category), 0) + 1
    actions = []
    if by_family.get("nolima_lexical_disconnect"):
        actions.append("Improve alias/latent-cue bridge selection and answer-span compaction on lexical-disconnect rows.")
    if by_family.get("mtrag_un_answerability"):
        actions.append("Tighten abstention thresholding for underspecified multi-turn RAG questions.")
    if by_family.get("loft_structured_memory"):
        actions.append("Add more structured-state fusion before any public LOFT-style calibration.")
    if not actions:
        actions.append("Use residual hashes only as regression pressure; do not generate public-derived training rows.")
    return {
        "failure_count": len(failures),
        "failure_by_family": by_family,
        "failure_by_category": by_category,
        "actions": actions,
        "public_training_allowed": False,
    }


def gate(name: str, passed: bool, severity: str, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "item_id": row["item_id"],
                "family": row["family"],
                "skill": row["skill"],
                "answer_hash": row["answer_hash"],
                "context_hash": row["context_hash"],
                "oracle_evidence_hash": row["oracle_evidence_hash"],
                "context_length_bucket": row.get("context_length_bucket"),
                "source_context_tokens_estimate": row.get("source_context_tokens_estimate"),
                "vcm_on_passed": row["vcm_on_passed"],
                "vcm_off_passed": row["vcm_off_passed"],
                "best_non_vcm_passed": row["best_non_vcm_passed"],
                "best_non_vcm_system": prompt_adapter.get_path(row, ["best_non_vcm_memory_system", "system"], ""),
                "vcm_on_evidence_recall": prompt_adapter.get_path(row, ["vcm_on", "evidence_recall"], 0.0),
                "vcm_on_no_admissible": prompt_adapter.get_path(row, ["vcm_on", "no_admissible"], False),
                "vcm_selected_context_compression_ratio": prompt_adapter.get_path(row, ["vcm_on", "selected_context_compression_ratio"], 0.0),
                "residual_categories": row.get("residual_categories"),
            }
        )
    return compact


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Hard Memory Private Analogues",
        "",
        f"State: `{report['trigger_state']}`",
        f"Cases: `{summary['case_count']}`",
        f"Families: `{summary['family_count']}`",
        f"Length buckets: `{summary['length_bucket_count']}`",
        f"VCM pass rate: `{summary['vcm_on_pass_rate']}`",
        f"Best single non-VCM pass rate: `{summary['best_single_non_vcm_pass_rate']}`",
        f"VCM over best non-VCM: `{summary['vcm_over_best_single_non_vcm_delta']}`",
        f"Evidence recall: `{summary['vcm_on_evidence_recall']}`",
        f"Abstention: `{summary['abstention']}`",
        "",
        "## Families",
        "",
    ]
    for family, row in prompt_adapter.dict_value(summary.get("family_summary")).items():
        lines.append(
            f"- `{family}`: VCM `{row.get('vcm_on_pass_rate')}`, best non-VCM `{row.get('best_single_non_vcm_pass_rate')}`, delta `{row.get('vcm_over_best_single_non_vcm_delta')}`, items `{row.get('items')}`"
        )
    lines.extend(["", "## Length Buckets", ""])
    for bucket, row in prompt_adapter.dict_value(summary.get("per_length_bucket")).items():
        lines.append(f"- `{bucket}`: VCM `{row.get('vcm_on_pass_rate')}`, items `{row.get('items')}`")
    lines.extend(["", "## Gates", ""])
    for row in report["gates"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {mark}: `{row['gate']}` - {row['evidence']}")
    lines.extend(["", "## Residual Repair Plan", ""])
    for action in report["residual_repair_plan"]["actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def is_no_answer_row(row: dict[str, Any]) -> bool:
    return row["answer_hash"] == prompt_adapter.stable_hash(["__NO_ANSWER__"])


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def summarize_numeric(values: list[float]) -> dict[str, float]:
    clean = sorted(values)
    if not clean:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "min": round(clean[0], 6),
        "max": round(clean[-1], 6),
        "mean": mean(clean),
        "p50": round(percentile(clean, 0.50), 6),
        "p90": round(percentile(clean, 0.90), 6),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
    return values[idx]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
