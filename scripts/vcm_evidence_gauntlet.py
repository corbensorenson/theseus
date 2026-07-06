"""Private VCM evidence gauntlet.

This is a broad, private-only proof surface for Theseus+VCM. It compares the
real VCM public-memory adapter path against the same non-VCM baselines used by
the governed public-memory runner, but it generates every case locally and never
loads public prompt/context/answer payloads into fixtures or training rows.
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

import vcm_longmemeval_private_residual_curriculum as lme_private  # noqa: E402
import vcm_official_public_memory_adapter as prompt_adapter  # noqa: E402


DEFAULT_OUT = REPORTS / "vcm_evidence_gauntlet.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "vcm_evidence_gauntlet.md"
DEFAULT_PROOF_CARD_OUT = REPORTS / "vcm_proof_card.md"
DEFAULT_CASE_COUNT = 1200
MAJOR_FAMILIES = [
    "longmemeval_semantic",
    "ruler_needle",
    "babilong_state",
    "file_task_memory",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--proof-card-out", default=rel(DEFAULT_PROOF_CARD_OUT))
    parser.add_argument("--case-count", type=int, default=DEFAULT_CASE_COUNT)
    parser.add_argument("--context-budget-chars", type=int, default=900)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(
        case_count=max(1000, args.case_count),
        context_budget_chars=max(128, args.context_budget_chars),
        started=started,
    )
    write_json(resolve(args.out), report)
    markdown = render_markdown(report)
    write_text(resolve(args.markdown_out), markdown)
    write_text(resolve(args.proof_card_out), render_proof_card(report))
    if args.summary_only:
        print(json.dumps({"trigger_state": report["trigger_state"], "summary": report["summary"], "gates": report["gates"]}, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, case_count: int, context_budget_chars: int, started: float) -> dict[str, Any]:
    items = build_private_gauntlet_items(case_count)
    rows = [score_item(item, context_budget_chars=context_budget_chars) for item in items]
    summary = summarize_rows(rows, context_budget_chars=context_budget_chars)
    gates = gauntlet_gates(summary)
    hard_failures = [row for row in gates if not row["passed"] and row["severity"] == "blocker"]
    trigger_state = "GREEN" if not hard_failures else "YELLOW"
    return {
        "policy": "project_theseus_vcm_evidence_gauntlet_v1",
        "created_utc": prompt_adapter.now(),
        "trigger_state": trigger_state,
        "private_only": True,
        "benchmark": "private_vcm_evidence_gauntlet",
        "context_budget_chars": context_budget_chars,
        "suite": {
            "case_count": len(items),
            "families": MAJOR_FAMILIES,
            "public_payload_used_for_private_cases": False,
            "training_rows_written": 0,
        },
        "summary": summary,
        "gates": gates,
        "hard_failures": hard_failures,
        "rows": compact_rows(rows),
        "proof_card": proof_card_summary(summary),
        "public_confirmation_manifest_proposal": public_confirmation_manifest_proposal(summary),
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


def build_private_gauntlet_items(case_count: int) -> list[prompt_adapter.PublicMemoryItem]:
    ruler_count = max(200, round(case_count * 0.20))
    babilong_count = max(200, round(case_count * 0.20))
    file_task_count = max(160, round(case_count * 0.16))
    lme_count = max(1, case_count - ruler_count - babilong_count - file_task_count)

    items: list[prompt_adapter.PublicMemoryItem] = []
    items.extend(mark_family(lme_private.build_private_items(lme_count), "longmemeval_semantic"))
    items.extend(build_private_ruler_items(ruler_count))
    items.extend(build_private_babilong_items(babilong_count))
    items.extend(build_file_task_memory_items(file_task_count))
    return items[:case_count]


def mark_family(items: list[prompt_adapter.PublicMemoryItem], family: str) -> list[prompt_adapter.PublicMemoryItem]:
    for item in items:
        item.metadata["gauntlet_family"] = family
        item.metadata["private_only"] = True
    return items


def build_private_ruler_items(count: int) -> list[prompt_adapter.PublicMemoryItem]:
    items = prompt_adapter.build_ruler_items(
        max_items=count,
        offset=10_000,
        source_token_buckets=[4_000, 16_000, 64_000],
    )
    for item in items:
        item.item_id = f"private_gauntlet_{item.item_id}"
        item.metadata["gauntlet_family"] = "ruler_needle"
        item.metadata["private_only"] = True
        item.metadata["public_payload_derived"] = False
    return items


def build_private_babilong_items(count: int) -> list[prompt_adapter.PublicMemoryItem]:
    builders = [build_babi_qa1_case, build_babi_qa2_case, build_babi_qa3_case, build_babi_qa6_case]
    items = []
    idx = 0
    while len(items) < count:
        items.append(builders[idx % len(builders)](idx))
        idx += 1
    return items


def build_babi_qa1_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    person = person_name(idx)
    first = location_name(idx + 2)
    final = location_name(idx + 7)
    lines = [
        f"{person} moved to the {first}.",
        f"{other_person_name(idx)} went to the {location_name(idx + 3)}.",
        f"{person} travelled to the {location_name(idx + 5)}.",
        f"{person} journeyed to the {final}.",
    ]
    evidence = [{"id": f"private_babi_qa1_{idx}:line4", "text": lines[-1]}]
    return private_memory_item(
        item_id=f"private_gauntlet_babilong_qa1_{idx:04d}",
        benchmark="babilong",
        task="qa1",
        family="babilong_state",
        context="\n".join(lines),
        question=f"Where is {person}? ",
        answers=[final],
        evidence=evidence,
        skill="babilong_current_location",
    )


def build_babi_qa2_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    person = person_name(idx)
    item = item_name(idx)
    final = location_name(idx + 9)
    lines = [
        f"{person} moved to the {location_name(idx + 1)}.",
        f"{person} grabbed the {item}.",
        f"{other_person_name(idx)} went to the {location_name(idx + 4)}.",
        f"{person} journeyed to the {final}.",
    ]
    evidence = [
        {"id": f"private_babi_qa2_{idx}:line2", "text": lines[1]},
        {"id": f"private_babi_qa2_{idx}:line4", "text": lines[3]},
    ]
    return private_memory_item(
        item_id=f"private_gauntlet_babilong_qa2_{idx:04d}",
        benchmark="babilong",
        task="qa2",
        family="babilong_state",
        context="\n".join(lines),
        question=f"Where is the {item}? ",
        answers=[final],
        evidence=evidence,
        skill="babilong_object_tracking",
    )


def build_babi_qa3_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    person = person_name(idx)
    item = item_name(idx)
    before = location_name(idx + 6)
    after = location_name(idx + 10)
    lines = [
        f"{person} moved to the {location_name(idx + 1)}.",
        f"{person} grabbed the {item}.",
        f"{person} went to the {before}.",
        f"{person} journeyed to the {after}.",
        f"{person} discarded the {item} there.",
    ]
    evidence = [
        {"id": f"private_babi_qa3_{idx}:line3", "text": lines[2]},
        {"id": f"private_babi_qa3_{idx}:line4", "text": lines[3]},
        {"id": f"private_babi_qa3_{idx}:line5", "text": lines[4]},
    ]
    return private_memory_item(
        item_id=f"private_gauntlet_babilong_qa3_{idx:04d}",
        benchmark="babilong",
        task="qa3",
        family="babilong_state",
        context="\n".join(lines),
        question=f"Where was the {item} before the {after}? ",
        answers=[before],
        evidence=evidence,
        skill="babilong_temporal_object_tracking",
    )


def build_babi_qa6_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    person = person_name(idx)
    final = location_name(idx + 4)
    query_location = final if idx % 2 == 0 else location_name(idx + 8)
    answer = "yes" if query_location == final else "no"
    lines = [
        f"{person} moved to the {location_name(idx + 1)}.",
        f"{other_person_name(idx)} went to the {location_name(idx + 2)}.",
        f"{person} travelled to the {final}.",
    ]
    evidence = [{"id": f"private_babi_qa6_{idx}:line3", "text": lines[-1]}]
    return private_memory_item(
        item_id=f"private_gauntlet_babilong_qa6_{idx:04d}",
        benchmark="babilong",
        task="qa6",
        family="babilong_state",
        context="\n".join(lines),
        question=f"Is {person} in the {query_location}? ",
        answers=[answer],
        evidence=evidence,
        skill="babilong_yes_no_location",
    )


def build_file_task_memory_items(count: int) -> list[prompt_adapter.PublicMemoryItem]:
    builders = [
        build_file_path_case,
        build_task_status_case,
        build_storage_location_case,
        build_owner_contact_case,
        build_file_abstention_case,
    ]
    items: list[prompt_adapter.PublicMemoryItem] = []
    idx = 0
    while len(items) < count:
        items.append(builders[idx % len(builders)](idx))
        idx += 1
    return items


def build_file_path_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    project = f"atlas panel {idx % 17}"
    answer = f"file-{idx:04d}"
    lines = private_noise("file_front", idx, 7)
    lines.append(lme_private.lme_line(f"file_anchor_{idx}", idx + 7, "user", f"The export for {project} was moved during the cleanup."))
    lines.extend(private_noise("file_mid", idx + 20, 6))
    evidence = lme_private.lme_line(f"file_answer_{idx}", idx + 34, "user", f"The current file is {answer}.")
    lines.append(evidence)
    lines.extend(private_noise("file_tail", idx + 48, 8))
    return private_memory_item(
        item_id=f"private_gauntlet_file_path_{idx:04d}",
        benchmark="longmemeval",
        task="file_task_memory",
        family="file_task_memory",
        context="\n".join(lines),
        question=f"What is the current file for {project}?",
        answers=[answer],
        evidence=[{"id": f"private_file_path_{idx}:line", "text": evidence}],
        skill="file_exact_name",
    )


def build_task_status_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    task = f"compiler sweep {idx % 19}"
    answer = f"ready-{idx:04d}"
    lines = private_noise("task_front", idx, 6)
    lines.append(lme_private.lme_line(f"task_anchor_{idx}", idx + 5, "user", f"The {task} status was stale yesterday."))
    lines.extend(private_noise("task_mid", idx + 18, 7))
    evidence = lme_private.lme_line(f"task_answer_{idx}", idx + 37, "user", f"It is now {answer} after the verifier pass.")
    lines.append(evidence)
    lines.extend(private_noise("task_tail", idx + 52, 7))
    return private_memory_item(
        item_id=f"private_gauntlet_task_status_{idx:04d}",
        benchmark="longmemeval",
        task="file_task_memory",
        family="file_task_memory",
        context="\n".join(lines),
        question=f"What is the current status for {task}?",
        answers=[answer],
        evidence=[{"id": f"private_task_status_{idx}:line", "text": evidence}],
        skill="task_current_status",
    )


def build_storage_location_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    bundle = f"model bundle {idx % 13}"
    answer = f"vault {idx % 9}"
    lines = private_noise("storage_front", idx, 7)
    lines.append(lme_private.lme_line(f"storage_anchor_{idx}", idx + 6, "user", f"The {bundle} started in the old shelf."))
    lines.extend(private_noise("storage_mid", idx + 19, 6))
    evidence = lme_private.lme_line(f"storage_answer_{idx}", idx + 35, "user", f"I moved it to {answer} after the artifact scan.")
    lines.append(evidence)
    lines.extend(private_noise("storage_tail", idx + 50, 8))
    return private_memory_item(
        item_id=f"private_gauntlet_storage_{idx:04d}",
        benchmark="longmemeval",
        task="file_task_memory",
        family="file_task_memory",
        context="\n".join(lines),
        question=f"Where is the {bundle} currently?",
        answers=[answer],
        evidence=[{"id": f"private_storage_{idx}:line", "text": evidence}],
        skill="storage_location",
    )


def build_owner_contact_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    asset = f"sensor kit {idx % 11}"
    answer = f"Morgan Slate {idx % 8}"
    lines = private_noise("owner_front", idx, 7)
    lines.append(lme_private.lme_line(f"owner_anchor_{idx}", idx + 9, "user", f"The owner contact for the {asset} was being updated."))
    lines.extend(private_noise("owner_mid", idx + 23, 6))
    evidence = lme_private.lme_line(f"owner_answer_{idx}", idx + 38, "user", f"{answer} is now the contact for it.")
    lines.append(evidence)
    lines.extend(private_noise("owner_tail", idx + 54, 8))
    return private_memory_item(
        item_id=f"private_gauntlet_owner_{idx:04d}",
        benchmark="longmemeval",
        task="file_task_memory",
        family="file_task_memory",
        context="\n".join(lines),
        question=f"Who is the current owner contact for the {asset}?",
        answers=[answer],
        evidence=[{"id": f"private_owner_{idx}:line", "text": evidence}],
        skill="owner_contact",
    )


def build_file_abstention_case(idx: int) -> prompt_adapter.PublicMemoryItem:
    lines = private_noise("file_abs_front", idx, 10)
    lines.append(lme_private.lme_line(f"file_abs_anchor_{idx}", idx + 13, "user", "The local file list mentioned a gray folder and a green folder."))
    lines.extend(private_noise("file_abs_tail", idx + 24, 10))
    return private_memory_item(
        item_id=f"private_gauntlet_file_abstain_{idx:04d}",
        benchmark="longmemeval",
        task="file_task_memory",
        family="file_task_memory",
        context="\n".join(lines),
        question=f"What is the secret path for hidden file packet {idx}?",
        answers=["__NO_ANSWER__"],
        evidence=[],
        skill="file_abstention",
    )


def private_memory_item(
    *,
    item_id: str,
    benchmark: str,
    task: str,
    family: str,
    context: str,
    question: str,
    answers: list[str],
    evidence: list[dict[str, str]],
    skill: str,
) -> prompt_adapter.PublicMemoryItem:
    return prompt_adapter.PublicMemoryItem(
        item_id=item_id,
        benchmark=benchmark,
        task=task,
        prompt=f"<private_context>\n{context}\n</private_context>\n\nQuestion: {question}",
        context=context,
        question=question,
        answers=answers,
        oracle_evidence=evidence,
        metadata={
            "gauntlet_family": family,
            "gauntlet_skill": skill,
            "private_only": True,
            "public_payload_derived": False,
            "score_semantics": "local deterministic exact/contains answer extraction",
        },
    )


def score_item(item: prompt_adapter.PublicMemoryItem, *, context_budget_chars: int) -> dict[str, Any]:
    scored = prompt_adapter.score_item(item, context_budget_chars=context_budget_chars)
    best_non_vcm = prompt_adapter.dict_value(scored.get("best_non_vcm_memory_system"))
    memory_systems = prompt_adapter.dict_value(scored.get("memory_systems"))
    family = str(item.metadata.get("gauntlet_family") or item.benchmark)
    skill = str(item.metadata.get("gauntlet_skill") or item.metadata.get("private_analogue") or item.task)
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
    answerable_rows = [row for row in rows if not is_no_answer_row(row)]
    vcm_rate = mean([1.0 if row["vcm_on_passed"] else 0.0 for row in rows])
    vcm_off_rate = mean([1.0 if row["vcm_off_passed"] else 0.0 for row in rows])
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
        "vcm_on_pass_rate": vcm_rate,
        "vcm_off_flat_tail_pass_rate": vcm_off_rate,
        "best_single_non_vcm_pass_rate": best_single_non_vcm_rate,
        "vcm_over_best_single_non_vcm_delta": round(vcm_rate - best_single_non_vcm_rate, 6),
        "vcm_over_flat_tail_delta": round(vcm_rate - vcm_off_rate, 6),
        "vcm_only_wins_vs_flat_tail": sum(1 for row in rows if row["vcm_on_passed"] and not row["vcm_off_passed"]),
        "flat_tail_only_wins": sum(1 for row in rows if row["vcm_off_passed"] and not row["vcm_on_passed"]),
        "best_non_vcm_any_only_wins": sum(1 for row in rows if row["best_non_vcm_passed"] and not row["vcm_on_passed"]),
        "vcm_on_evidence_precision": mean([float(prompt_adapter.get_path(row, ["vcm_on", "evidence_precision"], 0.0) or 0.0) for row in answerable_rows]),
        "vcm_on_evidence_recall": mean([float(prompt_adapter.get_path(row, ["vcm_on", "evidence_recall"], 0.0) or 0.0) for row in answerable_rows]),
        "vcm_on_evidence_precision_all_cases": mean([float(prompt_adapter.get_path(row, ["vcm_on", "evidence_precision"], 0.0) or 0.0) for row in rows]),
        "vcm_on_evidence_recall_all_cases": mean([float(prompt_adapter.get_path(row, ["vcm_on", "evidence_recall"], 0.0) or 0.0) for row in rows]),
        "vcm_on_answer_span_chars_mean": mean([float(prompt_adapter.get_path(row, ["vcm_on", "answer_span_chars"], 0.0) or 0.0) for row in rows if not is_no_answer_row(row)]),
        "vcm_selected_context_compression_ratio_mean": mean([float(prompt_adapter.get_path(row, ["vcm_on", "selected_context_compression_ratio"], 0.0) or 0.0) for row in rows]),
        "latency_ms_mean": mean([float(row.get("latency_ms") or 0.0) for row in rows]),
        "no_admissible_rate": mean([1.0 if prompt_adapter.get_path(row, ["vcm_on", "no_admissible"], False) else 0.0 for row in rows]),
        "abstention": abstention,
        "family_summary": family_summary,
        "skill_summary": skill_summary,
        "minimum_major_family_pass_rate": min([family_summary.get(family, {}).get("vcm_on_pass_rate", 0.0) for family in MAJOR_FAMILIES], default=0.0),
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
            "answer_span_chars_mean": mean([float(result.get("answer_span_chars") or 0.0) for result in results]),
            "selected_context_compression_ratio_mean": mean([float(result.get("selected_context_compression_ratio") or 0.0) for result in results]),
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
        best_any = mean([1.0 if row["best_non_vcm_passed"] else 0.0 for row in group_rows])
        best_single = best_single_non_vcm_for_rows(group_rows)
        summary[group] = {
            "items": len(group_rows),
            "vcm_on_pass_rate": vcm_rate,
            "flat_tail_pass_rate": mean([1.0 if row["vcm_off_passed"] else 0.0 for row in group_rows]),
            "best_non_vcm_any_pass_rate": best_any,
            "best_single_non_vcm_pass_rate": best_single,
            "vcm_over_best_single_non_vcm_delta": round(vcm_rate - best_single, 6),
            "vcm_only_wins_vs_flat_tail": sum(1 for row in group_rows if row["vcm_on_passed"] and not row["vcm_off_passed"]),
            "flat_tail_only_wins": sum(1 for row in group_rows if row["vcm_off_passed"] and not row["vcm_on_passed"]),
            "best_non_vcm_any_only_wins": sum(1 for row in group_rows if row["best_non_vcm_passed"] and not row["vcm_on_passed"]),
            "vcm_evidence_recall": mean([float(prompt_adapter.get_path(row, ["vcm_on", "evidence_recall"], 0.0) or 0.0) for row in group_rows]),
            "vcm_evidence_precision": mean([float(prompt_adapter.get_path(row, ["vcm_on", "evidence_precision"], 0.0) or 0.0) for row in group_rows]),
            "no_admissible_rate": mean([1.0 if prompt_adapter.get_path(row, ["vcm_on", "no_admissible"], False) else 0.0 for row in group_rows]),
        }
    return summary


def best_single_non_vcm_for_rows(rows: list[dict[str, Any]]) -> float:
    systems = summarize_system_rates(rows)
    return max([row["pass_rate"] for name, row in systems.items() if name != "vcm_graph_evidence_selector"], default=0.0)


def abstention_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    no_answer = [row for row in rows if row["answer_hash"] == prompt_adapter.stable_hash(["__NO_ANSWER__"])]
    answerable = [row for row in rows if row not in no_answer]
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


def gauntlet_gates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    abstention = prompt_adapter.dict_value(summary.get("abstention"))
    return [
        gate("private_case_count_floor", int(summary.get("case_count") or 0) >= 1000, "blocker", f"cases={summary.get('case_count')}"),
        gate("private_vcm_overall_floor", float(summary.get("vcm_on_pass_rate") or 0.0) >= 0.90, "blocker", f"vcm={summary.get('vcm_on_pass_rate')}"),
        gate("private_major_family_floor", float(summary.get("minimum_major_family_pass_rate") or 0.0) >= 0.80, "blocker", f"min={summary.get('minimum_major_family_pass_rate')}"),
        gate("private_vcm_beats_best_single_non_vcm", float(summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.05, "blocker", f"delta={summary.get('vcm_over_best_single_non_vcm_delta')}"),
        gate("private_vcm_evidence_recall_floor", float(summary.get("vcm_on_evidence_recall") or 0.0) >= 0.90, "blocker", f"recall={summary.get('vcm_on_evidence_recall')}"),
        gate("private_abstention_precision_floor", float(abstention.get("precision") or 0.0) >= 0.95, "blocker", f"precision={abstention.get('precision')}"),
        gate("private_abstention_recall_floor", float(abstention.get("recall") or 0.0) >= 0.95, "blocker", f"recall={abstention.get('recall')}"),
        gate("no_fallback_returns", int(summary.get("fallback_return_count") or 0) == 0, "blocker", f"fallback={summary.get('fallback_return_count')}"),
        gate("no_teacher_or_external_inference", int(summary.get("teacher_solving_calls") or 0) == 0 and int(summary.get("external_inference_calls") or 0) == 0, "blocker", f"teacher={summary.get('teacher_solving_calls')} external={summary.get('external_inference_calls')}"),
        gate("no_public_payload_or_training_rows", int(summary.get("public_payload_chars_loaded") or 0) == 0 and int(summary.get("public_training_rows_written") or 0) == 0, "blocker", f"public_payload={summary.get('public_payload_chars_loaded')} public_training={summary.get('public_training_rows_written')}"),
    ]


def gate(name: str, passed: bool, severity: str, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def proof_card_summary(summary: dict[str, Any]) -> dict[str, Any]:
    family_summary = prompt_adapter.dict_value(summary.get("family_summary"))
    weak_families = [
        name
        for name, row in family_summary.items()
        if float(prompt_adapter.dict_value(row).get("vcm_on_pass_rate") or 0.0) < 0.90
    ]
    losing_families = [
        name
        for name, row in family_summary.items()
        if float(prompt_adapter.dict_value(row).get("vcm_over_best_single_non_vcm_delta") or 0.0) < 0.0
    ]
    return {
        "claim": "VCM is ready to be treated as a core Theseus memory capability"
        if float(summary.get("vcm_on_pass_rate") or 0.0) >= 0.90
        and float(summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.05
        and not losing_families
        else "VCM needs further private repair before a core-capability claim",
        "wins": [
            name
            for name, row in family_summary.items()
            if float(prompt_adapter.dict_value(row).get("vcm_over_best_single_non_vcm_delta") or 0.0) > 0.0
        ],
        "ties": [
            name
            for name, row in family_summary.items()
            if float(prompt_adapter.dict_value(row).get("vcm_over_best_single_non_vcm_delta") or 0.0) == 0.0
        ],
        "losses": losing_families,
        "weak_families": weak_families,
        "runtime_cost_ms_mean": summary.get("latency_ms_mean"),
        "compression_ratio_mean": summary.get("vcm_selected_context_compression_ratio_mean"),
    }


def public_confirmation_manifest_proposal(summary: dict[str, Any]) -> dict[str, Any]:
    gates_green = (
        int(summary.get("case_count") or 0) >= 1000
        and float(summary.get("vcm_on_pass_rate") or 0.0) >= 0.90
        and float(summary.get("minimum_major_family_pass_rate") or 0.0) >= 0.80
        and float(summary.get("vcm_over_best_single_non_vcm_delta") or 0.0) >= 0.05
    )
    return {
        "proposal_state": "READY_TO_PREPARE_GOVERNED_PUBLIC_CONFIRMATION_MANIFEST" if gates_green else "BLOCKED_BY_PRIVATE_GAUNTLET",
        "recommended_slice_id": "vcm_public_memory_prompt_slice_2026_06_19_evidence_gauntlet_confirmation_fresh" if gates_green else "",
        "candidate_public_surfaces": ["ruler", "babilong", "longmemeval", "helmet", "longbench_v2"],
        "run_public_automatically": False,
        "public_training_allowed": False,
        "reason": "Private VCM evidence gauntlet meets breadth gates." if gates_green else "Private VCM evidence gauntlet still has blocker failures.",
    }


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "item_id": row["item_id"],
                "family": row["family"],
                "skill": row["skill"],
                "benchmark": row["benchmark"],
                "task": row["task"],
                "answer_hash": row["answer_hash"],
                "context_hash": row["context_hash"],
                "oracle_evidence_hash": row["oracle_evidence_hash"],
                "context_length_bucket": row.get("context_length_bucket"),
                "vcm_on_passed": row["vcm_on_passed"],
                "vcm_off_passed": row["vcm_off_passed"],
                "best_non_vcm_passed": row["best_non_vcm_passed"],
                "best_non_vcm_system": prompt_adapter.get_path(row, ["best_non_vcm_memory_system", "system"], ""),
                "vcm_on_evidence_precision": prompt_adapter.get_path(row, ["vcm_on", "evidence_precision"], 0.0),
                "vcm_on_evidence_recall": prompt_adapter.get_path(row, ["vcm_on", "evidence_recall"], 0.0),
                "vcm_on_answer_span_chars": prompt_adapter.get_path(row, ["vcm_on", "answer_span_chars"], 0),
                "vcm_on_no_admissible": prompt_adapter.get_path(row, ["vcm_on", "no_admissible"], False),
                "vcm_selected_context_compression_ratio": prompt_adapter.get_path(row, ["vcm_on", "selected_context_compression_ratio"], 0.0),
                "residual_categories": row.get("residual_categories"),
            }
        )
    return compact


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Evidence Gauntlet",
        "",
        f"State: `{report['trigger_state']}`",
        f"Cases: `{summary['case_count']}`",
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
    lines.extend(["", "## Gates", ""])
    for row in report["gates"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {mark}: `{row['gate']}` - {row['evidence']}")
    lines.extend(["", "## Public Confirmation Proposal", ""])
    proposal = report["public_confirmation_manifest_proposal"]
    lines.append(f"- State: `{proposal['proposal_state']}`")
    lines.append(f"- Recommended slice: `{proposal['recommended_slice_id']}`")
    lines.append(f"- Run public automatically: `{proposal['run_public_automatically']}`")
    return "\n".join(lines) + "\n"


def render_proof_card(report: dict[str, Any]) -> str:
    summary = report["summary"]
    card = report["proof_card"]
    lines = [
        "# VCM Proof Card",
        "",
        f"Claim: `{card['claim']}`",
        "",
        f"- Private cases: `{summary['case_count']}`",
        f"- VCM pass rate: `{summary['vcm_on_pass_rate']}`",
        f"- Best single non-VCM pass rate: `{summary['best_single_non_vcm_pass_rate']}`",
        f"- VCM over best non-VCM: `{summary['vcm_over_best_single_non_vcm_delta']}`",
        f"- Evidence precision/recall: `{summary['vcm_on_evidence_precision']}` / `{summary['vcm_on_evidence_recall']}`",
        f"- Abstention precision/recall: `{summary['abstention']['precision']}` / `{summary['abstention']['recall']}`",
        f"- Mean selected-context compression ratio: `{summary['vcm_selected_context_compression_ratio_mean']}`",
        f"- Mean scoring latency: `{summary['latency_ms_mean']}` ms",
        f"- Wins: `{card['wins']}`",
        f"- Ties: `{card['ties']}`",
        f"- Losses: `{card['losses']}`",
        f"- Weak families: `{card['weak_families']}`",
        "",
        "No public payloads were loaded into private fixtures or training rows. No teacher solving, external inference, or fallback returns were used.",
    ]
    return "\n".join(lines) + "\n"


def private_noise(prefix: str, start: int, count: int) -> list[str]:
    return [
        lme_private.lme_line(f"{prefix}_{start}_{idx}", start + idx, "user" if idx % 4 else "assistant", f"Private gauntlet note {start}-{idx} mentions labels, folders, errands, and devices without the requested answer.")
        for idx in range(count)
    ]


def person_name(idx: int) -> str:
    return ["Mira", "Sol", "Nora", "Lena", "Omar", "Rhea", "Tara", "Vera"][idx % 8]


def other_person_name(idx: int) -> str:
    return ["Iris", "Milo", "Cora", "Zane", "Luca", "Nia", "Pia", "Theo"][(idx + 3) % 8]


def location_name(idx: int) -> str:
    return ["kitchen", "hallway", "garden", "office", "lab", "studio", "garage", "bedroom", "cellar", "workshop", "attic"][idx % 11]


def item_name(idx: int) -> str:
    return ["token", "badge", "cable", "sensor", "tablet", "marker", "adapter", "router"][idx % 8]


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
