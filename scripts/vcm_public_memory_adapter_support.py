"""Support helpers for the VCM official public-memory adapter.

This module keeps scoring, reporting, residual shaping, and IO helpers out of
``vcm_official_public_memory_adapter.py`` so the adapter remains focused on
public-source staging and resolver orchestration.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus_archive_resolver import is_archive_pointer, read_jsonl_follow_pointer, resolve_archived_path


ROOT = Path(__file__).resolve().parents[1]


def score_prediction(answers: list[str], prediction: str) -> bool:
    if not answers:
        return False
    expected = {normalize_answer(answer) for answer in answers}
    if expected == {"__no_answer__"}:
        return prediction == ""
    if not prediction:
        return False
    predicted = {normalize_answer(part) for part in re.split(r",| and ", prediction) if part.strip()}
    if len(expected) == 1:
        return next(iter(expected)) in normalize_answer(prediction)
    return expected == predicted


def evidence_metrics(item: PublicMemoryItem, selected_ids: list[str]) -> dict[str, float]:
    expected = {row["id"] for row in item.oracle_evidence}
    selected = set(selected_ids)
    overlap = expected & selected
    precision = len(overlap) / max(1, len(selected))
    recall = len(overlap) / max(1, len(expected))
    return {"precision": round(precision, 6), "recall": round(recall, 6)}


def annotate_resolver_result(result: dict[str, Any], length_metrics: dict[str, Any]) -> dict[str, Any]:
    source_chars = int(length_metrics.get("source_context_chars") or 0)
    selected_chars = int(result.get("selected_context_chars") or 0)
    return {
        **result,
        "source_context_chars": source_chars,
        "source_context_tokens_estimate": int(length_metrics.get("source_context_tokens_estimate") or 0),
        "context_length_bucket": length_metrics.get("context_length_bucket"),
        "context_depth_bucket": length_metrics.get("context_depth_bucket"),
        "selected_context_compression_ratio": round(selected_chars / max(1, source_chars), 6),
    }


def item_length_metrics(item: PublicMemoryItem) -> dict[str, Any]:
    source_chars = len(item.context)
    source_tokens = estimate_tokens(item.context)
    return {
        "source_context_chars": source_chars,
        "source_context_tokens_estimate": source_tokens,
        "context_length_bucket": length_bucket(source_tokens),
        "context_depth_bucket": context_depth_bucket(item),
        "oracle_evidence_count": len(item.oracle_evidence),
    }


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    char_estimate = max(1, round(len(text) / 4))
    wordish_estimate = len(re.findall(r"\S+", text))
    return max(char_estimate, wordish_estimate)


def length_bucket(tokens: int) -> str:
    if tokens < 8_000:
        return "lt_8k"
    if tokens < 32_000:
        return "8k_to_32k"
    if tokens < 128_000:
        return "32k_to_128k"
    return "128k_plus"


def context_depth_bucket(item: PublicMemoryItem) -> str:
    if not item.context or not item.oracle_evidence:
        return "unknown"
    positions = []
    for row in item.oracle_evidence:
        text = str(row.get("text") or "")
        if not text:
            continue
        pos = item.context.find(text)
        if pos >= 0:
            positions.append(pos)
    if not positions:
        return "unknown"
    span = (max(positions) - min(positions)) / max(1, len(item.context))
    if span >= 0.5:
        return "multi_span"
    depth = min(positions) / max(1, len(item.context))
    if depth < 0.2:
        return "front"
    if depth < 0.4:
        return "early_middle"
    if depth < 0.6:
        return "middle"
    if depth < 0.8:
        return "late_middle"
    return "tail"


def residual_categories(item: PublicMemoryItem, vcm_on: dict[str, Any], passed: bool, evidence: dict[str, float]) -> list[str]:
    if passed:
        return []
    categories = []
    if vcm_on.get("no_admissible"):
        categories.append("no_admissible")
    if evidence["recall"] < 1.0:
        categories.append("missed_evidence")
    if item.benchmark == "longmemeval":
        question_type = str(vcm_on.get("longmemeval_question_type") or "unknown")
        if question_type in {"temporal_first", "temporal_last", "current_update"}:
            categories.append("longmemeval_structured_recency_fusion")
        if question_type in {"choice", "fact", "preference", "where", "who", "when"}:
            categories.append("longmemeval_query_decomposition")
        if int(vcm_on.get("answer_span_chars") or 0) > 120:
            categories.append("longmemeval_answer_span_compaction")
        if vcm_on.get("no_admissible"):
            categories.append("longmemeval_abstention_thresholding")
    if item.benchmark == "longbench_v2":
        categories.append("longbench_v2_choice_evidence_selection")
        if vcm_on.get("no_admissible"):
            categories.append("longbench_v2_abstention_thresholding")
    if item.benchmark == "needlebench_opencompass":
        categories.append("needlebench_deep_needle_retrieval")
        if vcm_on.get("no_admissible"):
            categories.append("needlebench_answer_format_extraction")
    if item.benchmark == "infinitebench":
        categories.append("infinitebench_key_value_retrieval")
        if vcm_on.get("no_admissible"):
            categories.append("infinitebench_no_admissible_key_lookup")
    if item.task in {"qa1", "qa3", "qa6"}:
        categories.append("temporal_update_failure")
    if item.task in {"qa2", "qa3"}:
        categories.append("state_tracking_failure")
    if not categories:
        categories.append("answer_shape_mismatch")
    return sorted(set(categories))


def stale_update_deletion_eval(item: PublicMemoryItem, vcm_on: dict[str, Any], passed: bool) -> dict[str, Any]:
    if item.benchmark == "babilong" and item.task in {"qa1", "qa3", "qa6"}:
        return {
            "covered": True,
            "stale_update_behavior": "pass" if passed else "fail",
            "deletion_behavior": "not_applicable",
        }
    return {"covered": False, "stale_update_behavior": "not_applicable", "deletion_behavior": "not_applicable"}


def summarize_rows(rows: list[dict[str, Any]], items: list[PublicMemoryItem], context_budget_chars: int) -> dict[str, Any]:
    scored = [row for row in rows if row.get("status") == "scored"]
    by_benchmark: dict[str, dict[str, Any]] = {}
    for benchmark in sorted({row.get("benchmark") for row in scored}):
        bench_rows = [row for row in scored if row.get("benchmark") == benchmark]
        by_benchmark[str(benchmark)] = summarize_score_rows(bench_rows)
    memory_systems = summarize_memory_systems(scored)
    vcm_system = memory_systems.get("vcm_graph_evidence_selector", {})
    flat_system = memory_systems.get("flat_tail_window_baseline", {})
    non_vcm = {name: row for name, row in memory_systems.items() if name != "vcm_graph_evidence_selector"}
    best_non_vcm = max(non_vcm.values(), key=lambda row: row.get("pass_rate", 0.0), default={})
    win_counts = {
        "vcm_on": len([row for row in scored if row.get("winner") == "vcm_on"]),
        "vcm_off": len([row for row in scored if row.get("winner") == "vcm_off"]),
        "tie": len([row for row in scored if row.get("winner") == "tie"]),
    }
    public_chars = payload_char_counts(items)
    length_distribution = summarize_numeric([float(row.get("source_context_tokens_estimate") or 0.0) for row in scored])
    selected_distribution = summarize_numeric([float(get_path(row, ["vcm_on", "selected_context_chars"], 0.0) or 0.0) for row in scored])
    compression_distribution = summarize_numeric([float(get_path(row, ["vcm_on", "selected_context_compression_ratio"], 0.0) or 0.0) for row in scored])
    return {
        "item_count": len(rows),
        "scored_item_count": len(scored),
        "queued_item_count": len([row for row in rows if row.get("status") == "queued"]),
        "benchmark_count": len(by_benchmark),
        "context_budget_chars": context_budget_chars,
        "vcm_on_pass_rate": mean([1.0 if get_path(row, ["vcm_on", "passed"], False) else 0.0 for row in scored]),
        "vcm_off_pass_rate": mean([1.0 if get_path(row, ["vcm_off", "passed"], False) else 0.0 for row in scored]),
        "vcm_on_evidence_precision": mean([float(get_path(row, ["vcm_on", "evidence_precision"], 0.0) or 0.0) for row in scored]),
        "vcm_off_evidence_precision": mean([float(get_path(row, ["vcm_off", "evidence_precision"], 0.0) or 0.0) for row in scored]),
        "vcm_on_evidence_recall": mean([float(get_path(row, ["vcm_on", "evidence_recall"], 0.0) or 0.0) for row in scored]),
        "vcm_off_evidence_recall": mean([float(get_path(row, ["vcm_off", "evidence_recall"], 0.0) or 0.0) for row in scored]),
        "no_admissible_rate": mean([1.0 if get_path(row, ["vcm_on", "no_admissible"], False) else 0.0 for row in scored]),
        "win_counts": win_counts,
        "memory_systems": memory_systems,
        "best_non_vcm_memory_system": best_non_vcm,
        "source_context_token_distribution": length_distribution,
        "vcm_selected_context_char_distribution": selected_distribution,
        "vcm_selected_context_compression_ratio_distribution": compression_distribution,
        "per_length_bucket": summarize_bucket_rows(scored, "context_length_bucket"),
        "per_depth_bucket": summarize_bucket_rows(scored, "context_depth_bucket"),
        "vcm_over_flat_tail_delta": round(
            float(vcm_system.get("pass_rate") or 0.0) - float(flat_system.get("pass_rate") or 0.0),
            6,
        ),
        "vcm_over_best_non_vcm_delta": round(
            float(vcm_system.get("pass_rate") or 0.0) - float(best_non_vcm.get("pass_rate") or 0.0),
            6,
        ),
        "per_benchmark": by_benchmark,
        "stale_update_deletion": summarize_stale_update_deletion(scored),
        "latency_ms_mean": mean([float(row.get("latency_ms") or 0.0) for row in scored]),
        "public_payload_counters": public_chars,
    }


def summarize_score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    memory_systems = summarize_memory_systems(rows)
    vcm_system = memory_systems.get("vcm_graph_evidence_selector", {})
    flat_system = memory_systems.get("flat_tail_window_baseline", {})
    longmemeval_rows = [row for row in rows if row.get("benchmark") == "longmemeval"]
    return {
        "items": len(rows),
        "vcm_on_pass_rate": mean([1.0 if get_path(row, ["vcm_on", "passed"], False) else 0.0 for row in rows]),
        "vcm_off_pass_rate": mean([1.0 if get_path(row, ["vcm_off", "passed"], False) else 0.0 for row in rows]),
        "vcm_on_evidence_precision": mean([float(get_path(row, ["vcm_on", "evidence_precision"], 0.0) or 0.0) for row in rows]),
        "vcm_off_evidence_precision": mean([float(get_path(row, ["vcm_off", "evidence_precision"], 0.0) or 0.0) for row in rows]),
        "vcm_on_evidence_recall": mean([float(get_path(row, ["vcm_on", "evidence_recall"], 0.0) or 0.0) for row in rows]),
        "vcm_off_evidence_recall": mean([float(get_path(row, ["vcm_off", "evidence_recall"], 0.0) or 0.0) for row in rows]),
        "vcm_only_wins": len([row for row in rows if row.get("winner") == "vcm_on"]),
        "off_only_wins": len([row for row in rows if row.get("winner") == "vcm_off"]),
        "memory_systems": memory_systems,
        "source_context_token_distribution": summarize_numeric([float(row.get("source_context_tokens_estimate") or 0.0) for row in rows]),
        "vcm_selected_context_compression_ratio_distribution": summarize_numeric([float(get_path(row, ["vcm_on", "selected_context_compression_ratio"], 0.0) or 0.0) for row in rows]),
        "per_length_bucket": summarize_bucket_rows(rows, "context_length_bucket"),
        "per_depth_bucket": summarize_bucket_rows(rows, "context_depth_bucket"),
        "longmemeval_question_type": summarize_longmemeval_question_types(longmemeval_rows),
        "vcm_answer_span_chars": summarize_numeric([float(get_path(row, ["vcm_on", "answer_span_chars"], 0.0) or 0.0) for row in rows]),
        "vcm_over_flat_tail_delta": round(
            float(vcm_system.get("pass_rate") or 0.0) - float(flat_system.get("pass_rate") or 0.0),
            6,
        ),
        "no_admissible_rate": mean([1.0 if get_path(row, ["vcm_on", "no_admissible"], False) else 0.0 for row in rows]),
        "stale_update_deletion": summarize_stale_update_deletion(rows),
        "latency_ms_mean": mean([float(row.get("latency_ms") or 0.0) for row in rows]),
    }


def summarize_memory_systems(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    systems: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for name, result in dict_value(row.get("memory_systems")).items():
            if isinstance(result, dict):
                systems.setdefault(name, []).append(result)
    summary: dict[str, dict[str, Any]] = {}
    for name, results in sorted(systems.items()):
        summary[name] = {
            "system": name,
            "items": len(results),
            "pass_rate": mean([1.0 if result.get("passed") else 0.0 for result in results]),
            "evidence_precision": mean([float(result.get("evidence_precision") or 0.0) for result in results]),
            "evidence_recall": mean([float(result.get("evidence_recall") or 0.0) for result in results]),
            "no_admissible_rate": mean([1.0 if result.get("no_admissible") else 0.0 for result in results]),
            "answer_span_chars_mean": mean([float(result.get("answer_span_chars") or 0.0) for result in results]),
            "longmemeval_candidate_count_mean": mean([float(result.get("longmemeval_candidate_count") or 0.0) for result in results]),
            "selected_context_chars_mean": mean([float(result.get("selected_context_chars") or 0.0) for result in results]),
            "selected_context_compression_ratio_mean": mean([float(result.get("selected_context_compression_ratio") or 0.0) for result in results]),
        }
    return summary


def summarize_longmemeval_question_types(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        question_type = str(get_path(row, ["vcm_on", "longmemeval_question_type"], "unknown") or "unknown")
        buckets.setdefault(question_type, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for question_type, bucket_rows in sorted(buckets.items()):
        vcm_rate = mean([1.0 if get_path(row, ["vcm_on", "passed"], False) else 0.0 for row in bucket_rows])
        best_non_vcm_rate = mean([1.0 if any_non_vcm_passed(row) else 0.0 for row in bucket_rows])
        summary[question_type] = {
            "items": len(bucket_rows),
            "vcm_on_pass_rate": vcm_rate,
            "best_non_vcm_pass_rate": best_non_vcm_rate,
            "vcm_over_best_non_vcm_delta": round(vcm_rate - best_non_vcm_rate, 6),
            "no_admissible_rate": mean([1.0 if get_path(row, ["vcm_on", "no_admissible"], False) else 0.0 for row in bucket_rows]),
            "answer_span_chars_mean": mean([float(get_path(row, ["vcm_on", "answer_span_chars"], 0.0) or 0.0) for row in bucket_rows]),
        }
    return summary


def summarize_bucket_rows(rows: list[dict[str, Any]], bucket_field: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = str(row.get(bucket_field) or "unknown")
        buckets.setdefault(bucket, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for bucket, bucket_rows in sorted(buckets.items()):
        vcm_rate = mean([1.0 if get_path(row, ["vcm_on", "passed"], False) else 0.0 for row in bucket_rows])
        flat_rate = mean([1.0 if get_path(row, ["vcm_off", "passed"], False) else 0.0 for row in bucket_rows])
        best_non_vcm_rate = mean([1.0 if any_non_vcm_passed(row) else 0.0 for row in bucket_rows])
        summary[bucket] = {
            "items": len(bucket_rows),
            "vcm_on_pass_rate": vcm_rate,
            "vcm_off_flat_tail_pass_rate": flat_rate,
            "best_non_vcm_pass_rate": best_non_vcm_rate,
            "vcm_over_flat_tail_delta": round(vcm_rate - flat_rate, 6),
            "vcm_over_best_non_vcm_delta": round(vcm_rate - best_non_vcm_rate, 6),
            "vcm_only_wins": len([row for row in bucket_rows if row.get("winner") == "vcm_on"]),
            "off_only_wins": len([row for row in bucket_rows if row.get("winner") == "vcm_off"]),
            "no_admissible_rate": mean([1.0 if get_path(row, ["vcm_on", "no_admissible"], False) else 0.0 for row in bucket_rows]),
            "source_context_tokens_mean": mean([float(row.get("source_context_tokens_estimate") or 0.0) for row in bucket_rows]),
            "vcm_selected_context_compression_ratio_mean": mean([float(get_path(row, ["vcm_on", "selected_context_compression_ratio"], 0.0) or 0.0) for row in bucket_rows]),
        }
    return summary


def any_non_vcm_passed(row: dict[str, Any]) -> bool:
    for name, result in dict_value(row.get("memory_systems")).items():
        if name == "vcm_graph_evidence_selector":
            continue
        if isinstance(result, dict) and result.get("passed"):
            return True
    return False


def summarize_numeric(values: list[float]) -> dict[str, float]:
    clean = sorted(value for value in values if value >= 0.0)
    if not clean:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(clean[0], 6),
        "p50": round(percentile(clean, 0.5), 6),
        "p90": round(percentile(clean, 0.9), 6),
        "max": round(clean[-1], 6),
        "mean": mean(clean),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[idx]


def summarize_stale_update_deletion(rows: list[dict[str, Any]]) -> dict[str, int]:
    covered = [row for row in rows if get_path(row, ["stale_update_deletion", "covered"], False)]
    return {
        "covered": len(covered),
        "stale_update_pass": len([
            row for row in covered if get_path(row, ["stale_update_deletion", "stale_update_behavior"], "") == "pass"
        ]),
        "stale_update_fail": len([
            row for row in covered if get_path(row, ["stale_update_deletion", "stale_update_behavior"], "") == "fail"
        ]),
        "deletion_pass": len([
            row for row in covered if get_path(row, ["stale_update_deletion", "deletion_behavior"], "") == "pass"
        ]),
        "deletion_fail": len([
            row for row in covered if get_path(row, ["stale_update_deletion", "deletion_behavior"], "") == "fail"
        ]),
    }


def payload_char_counts(items: list[PublicMemoryItem]) -> dict[str, int]:
    return {
        "public_prompt_chars_loaded": sum(len(item.prompt) for item in items if item.prompt),
        "public_context_chars_loaded": sum(len(item.context) for item in items if item.context),
        "public_answer_chars_loaded": sum(len(json.dumps(item.answers)) for item in items if item.answers),
        "public_trace_chars_loaded": 0,
        "public_solution_chars_loaded": 0,
        "public_template_chars_loaded": 0,
        "public_tests_loaded": 0,
        "public_training_rows_written": 0,
    }


def build_private_residuals(rows: list[dict[str, Any]], path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    categories = aggregate_residuals(rows)
    residuals = []
    for idx, category in enumerate(categories, start=1):
        residuals.append(private_residual_fixture(category, idx))
    benchmark_plan = aggregate_private_repair_plan(rows)
    report = {
        "policy": "project_theseus_vcm_public_memory_private_residual_repair_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "private_only": True,
        "repair_needed": bool(categories),
        "public_prompt_chars": 0,
        "public_answer_chars": 0,
        "public_training_rows_written": 0,
        "residual_categories": categories,
        "aggregate_benchmark_plan": benchmark_plan,
        "fixture_count": len(residuals),
        "fixtures": rel(path),
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return residuals, report


def aggregate_private_repair_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("status") == "scored"]
    by_benchmark: dict[str, dict[str, Any]] = {}
    for benchmark in sorted({str(row.get("benchmark") or "") for row in scored}):
        bench_rows = [row for row in scored if row.get("benchmark") == benchmark]
        failed = [row for row in bench_rows if not get_path(row, ["vcm_on", "passed"], False)]
        best_non_vcm_rate = mean([1.0 if any_non_vcm_passed(row) else 0.0 for row in bench_rows])
        vcm_rate = mean([1.0 if get_path(row, ["vcm_on", "passed"], False) else 0.0 for row in bench_rows])
        by_benchmark[benchmark] = {
            "items": len(bench_rows),
            "vcm_on_pass_rate": vcm_rate,
            "best_non_vcm_pass_rate": best_non_vcm_rate,
            "vcm_over_best_non_vcm_delta": round(vcm_rate - best_non_vcm_rate, 6),
            "failed_items": len(failed),
            "off_only_wins": len([row for row in bench_rows if row.get("winner") == "vcm_off"]),
            "residual_category_counts": count_values(cat for row in failed for cat in list_value(row.get("residual_categories"))),
            "length_bucket_fail_counts": count_values(str(row.get("context_length_bucket") or "unknown") for row in failed),
            "depth_bucket_fail_counts": count_values(str(row.get("context_depth_bucket") or "unknown") for row in failed),
        }
    longmemeval = by_benchmark.get("longmemeval", {})
    next_private_targets = []
    if longmemeval and float(longmemeval.get("vcm_over_best_non_vcm_delta") or 0.0) < 0.0:
        next_private_targets.extend(
            [
                "private_longmemeval_multi_session_query_decomposition",
                "private_longmemeval_answer_span_compaction",
                "private_longmemeval_structured_recency_fusion",
                "private_longmemeval_abstention_thresholding",
            ]
        )
    if any("answer_shape_mismatch" in list_value(row.get("residual_categories")) for row in scored):
        next_private_targets.append("private_answer_shape_exact_span_extraction")
    if any("missed_evidence" in list_value(row.get("residual_categories")) for row in scored):
        next_private_targets.append("private_evidence_selection_recall_ladder")
    if any("longbench_v2_choice_evidence_selection" in list_value(row.get("residual_categories")) for row in scored):
        next_private_targets.append("private_longbench_v2_style_choice_evidence_selection")
    if any("needlebench_deep_needle_retrieval" in list_value(row.get("residual_categories")) for row in scored):
        next_private_targets.append("private_needlebench_style_deep_needle_retrieval")
    if any("infinitebench_key_value_retrieval" in list_value(row.get("residual_categories")) for row in scored):
        next_private_targets.append("private_infinitebench_style_key_value_retrieval")
    return {
        "public_source": "aggregate_counts_only",
        "public_prompt_chars": 0,
        "public_answer_chars": 0,
        "by_benchmark": by_benchmark,
        "next_private_targets": sorted(set(next_private_targets)),
    }


def private_residual_fixture(category: str, idx: int) -> dict[str, Any]:
    templates = {
        "missed_evidence": {
            "context": "The archive shelf is noisy. Corben put the blue adapter report in drawer C. The old note says drawer A.",
            "question": "Where is the blue adapter report now?",
            "answer": "drawer C",
        },
        "temporal_update_failure": {
            "context": "The workshop node was in idle mode. Later the workshop node moved to training mode.",
            "question": "What is the current workshop node mode?",
            "answer": "training mode",
        },
        "state_tracking_failure": {
            "context": "Mira picked up the token in the lab. Mira moved to the office. Mira handed the token to Sol.",
            "question": "Who has the token now?",
            "answer": "Sol",
        },
        "answer_shape_mismatch": {
            "context": "The private run produced score 17 and score 19 for the same arm.",
            "question": "List the scores for the arm.",
            "answer": "17,19",
        },
        "no_admissible": {
            "context": "A note states the cache shard is in bay seven.",
            "question": "Which bay contains the cache shard?",
            "answer": "bay seven",
        },
        "longbench_v2_choice_evidence_selection": {
            "context": "The blue arm improved after the sparse pass. The red arm regressed after the dense pass.",
            "question": "Which arm improved after the sparse pass? A. red arm B. blue arm C. green arm D. gold arm",
            "answer": "B",
        },
        "longbench_v2_abstention_thresholding": {
            "context": "The report discusses unrelated build logs and does not identify a selected arm.",
            "question": "Which arm was selected? A. red arm B. blue arm C. green arm D. gold arm",
            "answer": "__no_answer__",
        },
        "needlebench_deep_needle_retrieval": {
            "context": "Long filler. The first lab node to finish the adapter run was the cedar node.",
            "question": "Which lab node finished first? Please answer in the format 'The first lab node to finish was________.'",
            "answer": "The first lab node to finish was the cedar node.",
        },
        "needlebench_answer_format_extraction": {
            "context": "The checkpoint drawer for the airport run was drawer Q.",
            "question": "Which drawer held the checkpoint? Please answer in the format 'The checkpoint drawer was________.'",
            "answer": "The checkpoint drawer was drawer Q.",
        },
        "infinitebench_key_value_retrieval": {
            "context": "{\"private_key_001\": \"north-cache\", \"private_key_002\": \"south-cache\"}",
            "question": "Extract the value corresponding to the specified key in the JSON object: \"private_key_002\".",
            "answer": "south-cache",
        },
        "infinitebench_no_admissible_key_lookup": {
            "context": "{\"private_key_001\": \"north-cache\"}",
            "question": "Extract the value corresponding to the specified key in the JSON object: \"private_key_missing\".",
            "answer": "__no_answer__",
        },
    }
    spec = templates.get(category, templates["answer_shape_mismatch"])
    return {
        "policy": "project_theseus_vcm_private_residual_fixture_v1",
        "fixture_id": f"vcm_private_residual_{idx:03d}_{category}",
        "category": category,
        "public_source": "aggregate_category_only",
        "public_prompt_chars": 0,
        "public_answer_chars": 0,
        "private_only": True,
        "context": spec["context"],
        "question": spec["question"],
        "answer": spec["answer"],
        "training_use_allowed": False,
        "requires_vcm_training_admission_bridge": True,
    }


def aggregate_residuals(rows: list[dict[str, Any]]) -> list[str]:
    categories = sorted({cat for row in rows for cat in list_value(row.get("residual_categories"))})
    return categories


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def payload_row(item: PublicMemoryItem) -> dict[str, Any]:
    return {
        "policy": "project_theseus_vcm_public_memory_quarantined_payload_v1",
        "item_id": item.item_id,
        "benchmark": item.benchmark,
        "task": item.task,
        "prompt": item.prompt,
        "context": item.context,
        "question": item.question,
        "answers": item.answers,
        "oracle_evidence": item.oracle_evidence,
        "metadata": item.metadata,
        "private_training_allowed": False,
        "external_inference_calls": 0,
    }


def item_manifest_row(item: PublicMemoryItem) -> dict[str, Any]:
    metrics = item_length_metrics(item)
    return {
        "item_id": item.item_id,
        "benchmark": item.benchmark,
        "task": item.task,
        "prompt_hash": stable_hash(item.prompt),
        "context_hash": stable_hash(item.context),
        "answer_hash": stable_hash(item.answers),
        "oracle_evidence_hash": stable_hash(item.oracle_evidence),
        **metrics,
        "metadata_public_safe": {
            "official_family": item.metadata.get("official_family"),
            "hf_dataset": item.metadata.get("hf_dataset"),
            "split": item.metadata.get("split"),
            "source_task_config": item.metadata.get("source_task_config"),
            "source_file": item.metadata.get("source_file"),
            "question_type": item.metadata.get("question_type"),
            "domain": item.metadata.get("domain"),
            "sub_domain": item.metadata.get("sub_domain"),
            "target_source_tokens": item.metadata.get("target_source_tokens"),
            "target_source_token_bucket": item.metadata.get("target_source_token_bucket"),
            "needle_depth": item.metadata.get("needle_depth"),
        },
        "private_training_allowed": False,
    }


def unscored_item_row(item: PublicMemoryItem) -> dict[str, Any]:
    metrics = item_length_metrics(item)
    return {
        "item_id": item.item_id,
        "benchmark": item.benchmark,
        "task": item.task,
        "status": item.metadata.get("status", "unscored"),
        "reason": item.metadata.get("reason", "not scored because blockers are present"),
        **metrics,
        "prompt_hash": stable_hash(item.prompt),
        "context_hash": stable_hash(item.context),
        "answer_hash": stable_hash(item.answers),
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
    }


def queued_row(item: PublicMemoryItem) -> dict[str, Any]:
    metrics = item_length_metrics(item)
    return {
        "item_id": item.item_id,
        "benchmark": item.benchmark,
        "task": item.task,
        "status": "queued",
        "reason": item.metadata.get("reason"),
        **metrics,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "public_training_rows_written": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Prompt-Level Public Memory Calibration",
        "",
        f"State: `{report['trigger_state']}`",
        "",
        "## Summary",
        "",
        f"- Calibration mode: `{report['calibration_mode']}`",
        f"- Scored items: `{summary['scored_item_count']}`",
        f"- Item offset per benchmark: `{summary.get('item_offset_per_benchmark')}`",
        f"- VCM-on pass rate: `{summary['vcm_on_pass_rate']}`",
        f"- VCM-off pass rate: `{summary['vcm_off_pass_rate']}`",
        f"- VCM-on evidence precision: `{summary['vcm_on_evidence_precision']}`",
        f"- VCM-off evidence precision: `{summary['vcm_off_evidence_precision']}`",
        f"- VCM-on evidence recall: `{summary['vcm_on_evidence_recall']}`",
        f"- VCM-off evidence recall: `{summary['vcm_off_evidence_recall']}`",
        f"- VCM-only wins: `{summary['win_counts']['vcm_on']}`",
        f"- Off-only wins: `{summary['win_counts']['vcm_off']}`",
        f"- VCM over flat-tail delta: `{summary.get('vcm_over_flat_tail_delta')}`",
        f"- VCM over best non-VCM delta: `{summary.get('vcm_over_best_non_vcm_delta')}`",
        f"- No-admissible rate: `{summary['no_admissible_rate']}`",
        f"- Source token distribution: `{summary.get('source_context_token_distribution')}`",
        f"- VCM compression ratio distribution: `{summary.get('vcm_selected_context_compression_ratio_distribution')}`",
        f"- Mean latency ms: `{summary['latency_ms_mean']}`",
        f"- External inference calls: `{summary['external_inference_calls']}`",
        f"- Public training rows written: `{summary['public_training_rows_written']}`",
        f"- Fallback return count: `{summary['fallback_return_count']}`",
        f"- Private residual fixtures: `{summary['private_residual_fixture_count']}`",
        "",
        "## Memory Systems",
        "",
    ]
    for name, row in summary.get("memory_systems", {}).items():
        lines.append(
            f"- `{name}`: pass `{row['pass_rate']}`, evidence precision `{row['evidence_precision']}`, "
            f"evidence recall `{row['evidence_recall']}`, no-admissible `{row['no_admissible_rate']}`, "
            f"mean selected chars `{row['selected_context_chars_mean']}`, "
            f"mean compression `{row.get('selected_context_compression_ratio_mean')}`"
        )
    lines.extend([
        "",
        "## Length Buckets",
        "",
    ])
    for bucket, row in summary.get("per_length_bucket", {}).items():
        lines.append(
            f"- `{bucket}`: items `{row['items']}`, VCM-on `{row['vcm_on_pass_rate']}`, "
            f"flat-tail `{row['vcm_off_flat_tail_pass_rate']}`, best non-VCM `{row['best_non_vcm_pass_rate']}`, "
            f"delta-flat `{row['vcm_over_flat_tail_delta']}`, delta-best `{row['vcm_over_best_non_vcm_delta']}`, "
            f"compression `{row['vcm_selected_context_compression_ratio_mean']}`"
        )
    lines.extend([
        "",
        "## Depth Buckets",
        "",
    ])
    for bucket, row in summary.get("per_depth_bucket", {}).items():
        lines.append(
            f"- `{bucket}`: items `{row['items']}`, VCM-on `{row['vcm_on_pass_rate']}`, "
            f"flat-tail `{row['vcm_off_flat_tail_pass_rate']}`, best non-VCM `{row['best_non_vcm_pass_rate']}`, "
            f"off-only `{row['off_only_wins']}`"
        )
    lines.extend([
        "",
        "## Per Benchmark",
        "",
    ])
    for benchmark, row in summary["per_benchmark"].items():
        lines.append(
            f"- `{benchmark}`: items `{row['items']}`, VCM-on `{row['vcm_on_pass_rate']}`, "
            f"VCM-off `{row['vcm_off_pass_rate']}`, evidence precision `{row['vcm_on_evidence_precision']}`/`{row['vcm_off_evidence_precision']}`, "
            f"evidence recall `{row['vcm_on_evidence_recall']}`/`{row['vcm_off_evidence_recall']}`, "
            f"no-admissible `{row['no_admissible_rate']}`, latency ms `{row['latency_ms_mean']}`, "
            f"stale/update `{row['stale_update_deletion']}`, VCM-only `{row['vcm_only_wins']}`, off-only `{row['off_only_wins']}`"
        )
    lines.extend(["", "## Boundaries", ""])
    lines.append("- Public prompt/context/answer payloads are quarantined under ignored `data/public_benchmarks/`.")
    lines.append("- Public payloads are not admitted to training rows.")
    if "longmemeval" in summary.get("per_benchmark", {}):
        lines.append("- LongMemEval is scored locally through deterministic extraction over quarantined official rows; no model judge or external inference is used.")
    else:
        lines.append("- LongMemEval remains queued until its data/evaluator can be staged without external model judging.")
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker.get('severity')}` `{blocker.get('kind')}`: {blocker.get('detail')}")
    return "\n".join(lines) + "\n"


def ruler_query_keys(question: str) -> list[str]:
    match = re.search(r"for (.*?) mentioned", question)
    if not match:
        return []
    return [part.strip().strip(",") for part in re.split(r", and |,| and ", match.group(1)) if part.strip()]


def first_capitalized_name(text: str) -> str:
    question_words = {"Where", "What", "Who", "When", "Why", "How", "Is", "Are", "Was", "Were", "Did", "Does"}
    for match in re.finditer(r"\b([A-Z][a-z]+)\b", text):
        value = match.group(1)
        if value not in question_words:
            return value
    return ""


def person_from_where_question(question: str) -> str:
    match = re.search(r"\bWhere (?:is|was) ([A-Z][a-z]+)\b", question)
    return match.group(1) if match else first_capitalized_name(question)


def object_from_question(question: str) -> str:
    match = re.search(r"(?:Where is|Where was|What is) the ([a-z]+)", question)
    return match.group(1) if match else ""


def before_location_query(question: str) -> tuple[str, str]:
    match = re.search(r"Where was the ([a-z]+) before the ([a-z]+)", question)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def compact_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_")[:96] or stable_hash(value)[-16:]


def parse_int_csv(value: str, *, default: list[int]) -> list[int]:
    parsed = []
    for part in value.split(","):
        try:
            parsed.append(max(1, int(part.strip())))
        except ValueError:
            continue
    return parsed or default


def mean(values: list[float]) -> float:
    return round(sum(values) / max(1, len(values)), 6)


def run(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True, timeout=120)
    return result.stdout.strip()


def git_head(path: Path) -> str:
    if not (path / ".git").exists():
        return ""
    try:
        return run(["git", "rev-parse", "HEAD"], cwd=path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ""


def forbidden_item_overlaps(
    *,
    ledger_path: Path,
    quarantine_root: Path,
    slice_ids: list[str],
    current_item_ids: list[str],
) -> dict[str, list[str]]:
    current = set(current_item_ids)
    overlaps: dict[str, list[str]] = {}
    for slice_id in slice_ids:
        prior = load_slice_item_ids(ledger_path=ledger_path, quarantine_root=quarantine_root, slice_id=slice_id)
        overlaps[slice_id] = sorted(current & prior)
    return overlaps


def load_slice_item_ids(*, ledger_path: Path, quarantine_root: Path, slice_id: str) -> set[str]:
    item_ids: set[str] = set()
    candidate_paths = [
        quarantine_root / slice_id / "item_manifest.json",
        quarantine_root / slice_id / "payloads.jsonl",
    ]
    for row in read_jsonl(ledger_path):
        if row.get("slice_id") != slice_id:
            continue
        for key in ["item_manifest", "payload_manifest"]:
            path_text = str(row.get(key) or "")
            if path_text:
                candidate_paths.append(resolve(path_text))
    for path in candidate_paths:
        if not path.exists():
            continue
        if path.name == "item_manifest.json":
            manifest = read_json(path)
            for row in list_value(manifest.get("items")):
                if isinstance(row, dict) and row.get("item_id"):
                    item_ids.add(str(row["item_id"]))
        elif path.name == "manifest.json":
            manifest = read_json(path)
            payload_path = resolve(str(manifest.get("payload_path") or ""))
            if payload_path.exists():
                candidate_paths.append(payload_path)
        elif path.suffix == ".jsonl":
            for row in read_jsonl(path):
                if row.get("item_id"):
                    item_ids.add(str(row["item_id"]))
    return item_ids


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    if is_archive_pointer(path):
        pointer = read_json(path)
        original_sha256 = str(pointer.get("original_sha256") or "")
        if original_sha256:
            return "sha256:" + original_sha256
    path = resolve_archived_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cursor = value
    for key in path:
        if not isinstance(cursor, dict):
            return default
        cursor = cursor.get(key)
    return default if cursor is None else cursor


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
