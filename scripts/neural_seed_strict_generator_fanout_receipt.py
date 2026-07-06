#!/usr/bin/env python3
"""Replay receipt for strict neural seed full-body candidates.

This gate measures the practical transformer/hybrid survival lane without
crediting body templates, fixed renderers, semantic routers, tools, public
benchmark data, or fallback returns as learned generation.
"""

from __future__ import annotations

import argparse
import ast
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from candidate_integrity import PROMOTION_FAMILIES, recompute_candidate_integrity  # noqa: E402
from code_lm_private_verifier import run_any, runtime_tmp_dir  # noqa: E402
from neural_seed_candidate_evidence_summary import no_cheat_exclusion_reasons  # noqa: E402
from neural_seed_code_proposer_comparator import load_private_rows  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
DEFAULT_CANDIDATES = ROOT / "reports" / "neural_seed_token_decoder_candidates_strict_body_tokens.jsonl"
DEFAULT_OUT = ROOT / "reports" / "neural_seed_strict_generator_fanout_receipt.json"
DEFAULT_MD = ROOT / "reports" / "neural_seed_strict_generator_fanout_receipt.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--eval-jsonl", default="")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--max-tasks", type=int, default=96)
    parser.add_argument("--max-candidates-per-task", type=int, default=4)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = read_json(config_path)
    data_cfg = dict(config.get("data") if isinstance(config.get("data"), dict) else {})
    eval_path = resolve(args.eval_jsonl or str(data_cfg.get("eval_jsonl") or ""))
    rows = read_jsonl(resolve(args.candidates))
    eval_rows = load_private_rows(eval_path, data_cfg)
    report = build_report(
        rows,
        eval_rows,
        config_path=config_path,
        candidate_path=resolve(args.candidates),
        eval_path=eval_path,
        max_tasks=max(0, int(args.max_tasks)),
        max_candidates_per_task=max(1, int(args.max_candidates_per_task)),
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))
    if args.gate and report["trigger_state"] == "RED":
        return 2
    return 0


def build_report(
    rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    config_path: Path,
    candidate_path: Path,
    eval_path: Path,
    max_tasks: int,
    max_candidates_per_task: int,
) -> dict[str, Any]:
    annotated: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    syntax_counts: Counter[str] = Counter()
    eligible_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_by_arm_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    eligible_candidate_count = 0
    fallback_return_count = 0
    external_inference_calls = 0
    public_training_rows_written = 0

    for row in rows:
        integrity = recompute_candidate_integrity(row)
        family = str(integrity.get("recomputed_candidate_family") or "unknown")
        family_counts[family] += 1
        for mismatch in integrity.get("integrity_mismatches") or []:
            mismatch_counts[str(mismatch)] += 1
        reasons = no_cheat_exclusion_reasons(row)
        for reason in reasons:
            exclusion_counts[str(reason)] += 1
        syntax_ok = python_syntax_ok(str(row.get("code") or ""))
        syntax_counts["syntax_valid" if syntax_ok else "syntax_invalid"] += 1
        fallback_return_count += int("fallback_return" in reasons)
        external_inference_calls += int(row.get("external_inference_calls") or 0)
        public_training_rows_written += int(row.get("public_training_rows_written") or 0)

        eligible = bool(
            str(row.get("phase") or "") == "private_eval"
            and family in PROMOTION_FAMILIES
            and bool(integrity.get("integrity_verified"))
            and not reasons
            and syntax_ok
        )
        if eligible:
            eligible_candidate_count += 1
            task_id = str(row.get("task_id") or row.get("source_task_id") or "")
            arm = str(row.get("substrate_arm") or "unknown")
            row_copy = dict(row)
            row_copy["candidate_integrity_receipt"] = compact_integrity(integrity)
            eligible_by_task[task_id].append(row_copy)
            eligible_by_arm_task[arm][task_id].append(row_copy)
        if len(annotated) < 64:
            annotated.append(
                {
                    "task_id": row.get("task_id"),
                    "phase": row.get("phase"),
                    "substrate_arm": row.get("substrate_arm"),
                    "candidate_generation_mode": row.get("candidate_generation_mode"),
                    "family": family,
                    "integrity_verified": bool(integrity.get("integrity_verified")),
                    "syntax_ok": syntax_ok,
                    "excluded_reasons": reasons,
                    "candidate_sha256": row.get("candidate_sha256") or integrity.get("candidate_sha256"),
                }
            )

    capped_by_task = {
        task_id: sorted(candidates, key=candidate_rank_key)[:max_candidates_per_task]
        for task_id, candidates in eligible_by_task.items()
    }
    task_ids = sorted(capped_by_task)
    if max_tasks:
        task_ids = task_ids[:max_tasks]
    selected_eval_rows = [row for row in eval_rows if str(row.get("task_id") or "") in set(task_ids)]
    combined_verification = run_verification(selected_eval_rows, capped_by_task)
    semantic_residual_diagnosis = build_blind_semantic_residual_diagnosis(
        selected_eval_rows,
        capped_by_task,
        combined_verification,
    )
    selector_ablation = build_blind_selector_ablation(
        selected_eval_rows,
        eligible_by_task,
        baseline_verification=combined_verification,
        max_candidates_per_task=max_candidates_per_task,
    )
    by_arm: dict[str, Any] = {}
    for arm, task_map in sorted(eligible_by_arm_task.items()):
        capped = {
            task_id: sorted(candidates, key=candidate_rank_key)[:max_candidates_per_task]
            for task_id, candidates in task_map.items()
            if task_id in set(task_ids)
        }
        by_arm[arm] = run_verification(selected_eval_rows, capped)

    summary = {
        "candidate_manifest": rel(candidate_path),
        "config": rel(config_path),
        "private_eval_jsonl": rel(eval_path),
        "candidate_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "integrity_mismatch_count": sum(mismatch_counts.values()),
        "integrity_mismatch_counts": dict(mismatch_counts.most_common()),
        "exclusion_reason_counts": dict(exclusion_counts.most_common()),
        "syntax_valid_count": int(syntax_counts["syntax_valid"]),
        "syntax_invalid_count": int(syntax_counts["syntax_invalid"]),
        "syntax_valid_rate": ratio(syntax_counts["syntax_valid"], len(rows)),
        "eligible_full_body_candidate_count": eligible_candidate_count,
        "eligible_task_count": len(capped_by_task),
        "replayed_task_count": len(selected_eval_rows),
        "max_tasks": max_tasks,
        "max_candidates_per_task": max_candidates_per_task,
        "combined": combined_verification,
        "semantic_residual_diagnosis": semantic_residual_diagnosis,
        "selector_ablation": selector_ablation,
        "by_arm": by_arm,
        "public_training_rows_written": public_training_rows_written,
        "external_inference_calls": external_inference_calls,
        "fallback_return_count": fallback_return_count,
        "candidate_generation_credit": eligible_candidate_count,
        "learned_generation_claim_allowed": False,
        "model_promotion_allowed": False,
    }
    hard_gaps = []
    if not rows:
        hard_gaps.append("candidate_manifest_empty")
    if eligible_candidate_count <= 0:
        hard_gaps.append("no_integrity_verified_learned_full_body_candidates")
    if summary["integrity_mismatch_count"] != 0:
        hard_gaps.append("candidate_integrity_mismatches_present")
    if public_training_rows_written != 0:
        hard_gaps.append("public_training_rows_written_nonzero")
    if external_inference_calls != 0:
        hard_gaps.append("external_inference_calls_nonzero")
    eligible_fallbacks = sum(
        1
        for task_candidates in capped_by_task.values()
        for candidate in task_candidates
        if "fallback_return" in no_cheat_exclusion_reasons(candidate)
    )
    if eligible_fallbacks != 0:
        hard_gaps.append("fallback_return_in_eligible_candidates")
    if selected_eval_rows and combined_verification["runtime_load_task_rate"] <= 0.0:
        hard_gaps.append("no_runtime_loadable_learned_full_body_candidates")
    trigger_state = "RED" if hard_gaps else "GREEN"
    if trigger_state == "GREEN" and combined_verification["intended_behavior_pass_rate"] <= 0.0:
        trigger_state = "YELLOW"

    records = build_viea_records(summary, trigger_state=trigger_state)
    return {
        "policy": "project_theseus_neural_seed_strict_generator_fanout_receipt_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "hard_gaps": hard_gaps,
        "summary": summary,
        "sampled_candidate_audit": annotated,
        "viea_strict_generator_fanout_records": records,
        "score_semantics": (
            "Private replay receipt for strict prompt/signature full-body token candidates. "
            "It measures syntax, private runtime loadability, and private intended behavior separately. "
            "It does not train, run public calibration, call a teacher, serve external tokens, or credit "
            "templates, semantic renderers, structural action renderers, routers, tools, or fallback returns as learned generation."
        ),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def run_verification(eval_rows: list[dict[str, Any]], candidates_by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="theseus_strict_generator_receipt_", dir=runtime_tmp_dir()) as tmp:
        root = Path(tmp)
        for task in eval_rows:
            task_id = str(task.get("task_id") or "")
            candidates = candidates_by_task.get(task_id, [])
            result = run_any(root, task, candidates, phase="private_eval")
            traces = list(result.get("attempt_traces") or [])
            attempts.extend(traces)
            task_results.append(
                {
                    "task_id": task_id,
                    "passed": bool(result.get("passed")),
                    "compile_any": any(bool(trace.get("compile_passed")) for trace in traces),
                    "runtime_load_any": any(bool(trace.get("runtime_loaded")) for trace in traces),
                    "attempt_count": len(traces),
                    "final_stage": result.get("verification_stage"),
                    "final_reward": result.get("verification_reward"),
                }
            )
    task_count = len(task_results)
    compile_tasks = sum(1 for row in task_results if row["compile_any"])
    runtime_tasks = sum(1 for row in task_results if row["runtime_load_any"])
    passed_tasks = sum(1 for row in task_results if row["passed"])
    stage_counts = Counter(str(row.get("verification_stage") or "") for row in attempts)
    return {
        "task_count": task_count,
        "attempt_count": len(attempts),
        "compile_task_count": compile_tasks,
        "runtime_load_task_count": runtime_tasks,
        "intended_behavior_passed_task_count": passed_tasks,
        "compile_task_rate": ratio(compile_tasks, task_count),
        "runtime_load_task_rate": ratio(runtime_tasks, task_count),
        "intended_behavior_pass_rate": ratio(passed_tasks, task_count),
        "stage_counts": dict(stage_counts.most_common()),
        "attempt_sample": attempts[:32],
        "task_results": task_results,
        "task_result_sample": task_results[:32],
    }


def build_blind_semantic_residual_diagnosis(
    eval_rows: list[dict[str, Any]],
    candidates_by_task: dict[str, list[dict[str, Any]]],
    verification: dict[str, Any],
) -> dict[str, Any]:
    task_results = {
        str(row.get("task_id") or ""): row
        for row in list(verification.get("task_results") or [])
        if isinstance(row, dict)
    }
    prompt_feature_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    failed_issue_counts: Counter[str] = Counter()
    passed_issue_counts: Counter[str] = Counter()
    candidate_feature_counts: Counter[str] = Counter()
    task_samples: list[dict[str, Any]] = []
    failed_runtime_loaded = 0
    failed_no_candidate = 0
    passed_task_count = 0

    for task in eval_rows:
        task_id = str(task.get("task_id") or "")
        prompt = str(task.get("prompt") or "")
        prompt_tags = prompt_feature_tags(prompt)
        for tag in prompt_tags:
            prompt_feature_counts[tag] += 1
        result = task_results.get(task_id, {})
        passed = bool(result.get("passed"))
        runtime_loaded = bool(result.get("runtime_load_any"))
        if passed:
            passed_task_count += 1
        elif runtime_loaded:
            failed_runtime_loaded += 1
        else:
            failed_no_candidate += 1

        task_issues: Counter[str] = Counter()
        candidate_summaries: list[dict[str, Any]] = []
        for candidate in candidates_by_task.get(task_id, []):
            features = candidate_blind_code_features(str(candidate.get("code") or ""))
            candidate_issues = blind_candidate_issue_labels(features, prompt_tags)
            for label in candidate_issues:
                issue_counts[label] += 1
                task_issues[label] += 1
                if passed:
                    passed_issue_counts[label] += 1
                else:
                    failed_issue_counts[label] += 1
            for label, value in features.items():
                if isinstance(value, bool) and value:
                    candidate_feature_counts[label] += 1
            if len(candidate_summaries) < 3:
                candidate_summaries.append(
                    {
                        "candidate_sha256": candidate.get("candidate_sha256"),
                        "substrate_arm": candidate.get("substrate_arm"),
                        "rank": candidate.get("rank"),
                        "rank_score": candidate.get("rank_score"),
                        "features": features,
                        "issue_labels": candidate_issues,
                    }
                )
        if len(task_samples) < 16:
            task_samples.append(
                {
                    "task_id": task_id,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_feature_tags": prompt_tags,
                    "passed": passed,
                    "runtime_loaded": runtime_loaded,
                    "attempt_count": int(result.get("attempt_count") or 0),
                    "issue_counts": dict(task_issues.most_common()),
                    "candidate_summaries": candidate_summaries,
                }
            )

    task_count = len(eval_rows)
    dominant_failed_issues = [label for label, _ in failed_issue_counts.most_common(8)]
    return {
        "policy": "project_theseus_strict_generator_blind_semantic_residual_diagnosis_v1",
        "task_count": task_count,
        "passed_task_count": passed_task_count,
        "failed_runtime_loaded_task_count": failed_runtime_loaded,
        "failed_no_candidate_or_no_load_task_count": failed_no_candidate,
        "prompt_feature_counts": dict(prompt_feature_counts.most_common()),
        "candidate_feature_counts": dict(candidate_feature_counts.most_common()),
        "issue_counts": dict(issue_counts.most_common()),
        "failed_issue_counts": dict(failed_issue_counts.most_common()),
        "passed_issue_counts": dict(passed_issue_counts.most_common()),
        "dominant_failed_issue_labels": dominant_failed_issues,
        "recommended_private_repair_targets": blind_repair_targets(dominant_failed_issues),
        "task_samples": task_samples,
        "blind_input_fields_used": ["task_id", "prompt", "candidate.code", "candidate.rank", "candidate.rank_score", "verifier_stage_outcome"],
        "forbidden_fields_excluded": [
            "tests",
            "hidden_tests",
            "solution",
            "solution_body",
            "solution_expr",
            "expected",
            "category",
            "source_task_id",
            "decoder_contract.return_shape",
            "decoder_contract.type_family",
            "decoder_contract.required_constructs",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "score_semantics": (
            "Private blind residual diagnosis for the strict generator. It uses natural-language prompt "
            "feature tags, candidate AST/static features, and verifier stage outcomes. It does not use "
            "tests, solutions, answer labels, decoder-contract hidden target fields, public benchmark data, "
            "or teacher output, and it writes no training rows."
        ),
    }


def build_blind_selector_ablation(
    eval_rows: list[dict[str, Any]],
    eligible_by_task: dict[str, list[dict[str, Any]]],
    *,
    baseline_verification: dict[str, Any],
    max_candidates_per_task: int,
) -> dict[str, Any]:
    blind_capped_by_task: dict[str, list[dict[str, Any]]] = {}
    full_pool_by_task: dict[str, list[dict[str, Any]]] = {}
    task_receipts: list[dict[str, Any]] = []
    changed_top1_count = 0
    changed_topk_count = 0
    total_full_pool_candidates = 0
    max_pool_size = 0
    for task in eval_rows:
        task_id = str(task.get("task_id") or "")
        prompt = str(task.get("prompt") or "")
        prompt_tags = prompt_feature_tags(prompt)
        candidates = list(eligible_by_task.get(task_id, []))
        baseline_sorted = sorted(candidates, key=candidate_rank_key)
        blind_sorted = sorted(candidates, key=lambda row: blind_static_selector_key(row, prompt_tags))
        baseline_capped = baseline_sorted[:max_candidates_per_task]
        blind_capped = blind_sorted[:max_candidates_per_task]
        blind_capped_by_task[task_id] = blind_capped
        full_pool_by_task[task_id] = baseline_sorted
        total_full_pool_candidates += len(baseline_sorted)
        max_pool_size = max(max_pool_size, len(baseline_sorted))
        baseline_ids = [str(row.get("candidate_sha256") or "") for row in baseline_capped]
        blind_ids = [str(row.get("candidate_sha256") or "") for row in blind_capped]
        if baseline_ids[:1] != blind_ids[:1]:
            changed_top1_count += 1
        if baseline_ids != blind_ids:
            changed_topk_count += 1
        if len(task_receipts) < 16:
            task_receipts.append(
                {
                    "task_id": task_id,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_feature_tags": prompt_tags,
                    "candidate_pool_size": len(baseline_sorted),
                    "baseline_top_candidate_sha256": baseline_ids[0] if baseline_ids else "",
                    "blind_static_top_candidate_sha256": blind_ids[0] if blind_ids else "",
                    "top1_changed": baseline_ids[:1] != blind_ids[:1],
                    "topk_changed": baseline_ids != blind_ids,
                    "blind_static_top_candidate_score": (
                        blind_static_selector_score(blind_capped[0], prompt_tags)
                        if blind_capped
                        else None
                    ),
                }
            )

    blind_verification = run_verification(eval_rows, blind_capped_by_task)
    full_pool_verification = run_verification(eval_rows, full_pool_by_task)
    baseline_passed = int(baseline_verification.get("intended_behavior_passed_task_count") or 0)
    blind_passed = int(blind_verification.get("intended_behavior_passed_task_count") or 0)
    oracle_passed = int(full_pool_verification.get("intended_behavior_passed_task_count") or 0)
    task_count = len(eval_rows)
    if oracle_passed > baseline_passed:
        diagnosis = "selector_quality_gap"
    elif blind_passed > baseline_passed:
        diagnosis = "blind_static_selector_promising"
    else:
        diagnosis = "candidate_pool_semantic_quality_gap"
    return {
        "policy": "project_theseus_strict_generator_blind_selector_ablation_v1",
        "selector_modes": [
            "baseline_model_rank",
            "blind_prompt_ast_static_rank",
            "full_pool_pass_if_any_oracle",
        ],
        "task_count": task_count,
        "max_candidates_per_task": int(max_candidates_per_task),
        "candidate_pool_task_count": len(full_pool_by_task),
        "total_full_pool_candidates": total_full_pool_candidates,
        "max_candidate_pool_size": max_pool_size,
        "top1_changed_task_count": changed_top1_count,
        "topk_changed_task_count": changed_topk_count,
        "baseline_model_rank": selector_mode_summary(baseline_verification),
        "blind_prompt_ast_static_rank": selector_mode_summary(blind_verification),
        "full_pool_pass_if_any_oracle": selector_mode_summary(full_pool_verification),
        "baseline_to_blind_pass_delta": blind_passed - baseline_passed,
        "baseline_to_oracle_pass_delta": oracle_passed - baseline_passed,
        "selector_diagnosis": diagnosis,
        "task_receipts": task_receipts,
        "blind_ranker_input_fields": ["prompt", "candidate.code", "candidate.rank", "candidate.rank_score"],
        "oracle_use_boundary": (
            "The full-pool oracle is private evaluation-only pass-if-any analysis. It is not used to rank "
            "future candidates, train rows, or support learned-generation promotion."
        ),
        "forbidden_fields_excluded": [
            "tests",
            "hidden_tests",
            "solution",
            "solution_body",
            "solution_expr",
            "expected",
            "category",
            "source_task_id",
            "decoder_contract.return_shape",
            "decoder_contract.type_family",
            "decoder_contract.required_constructs",
            "public_benchmark_payloads",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "score_semantics": (
            "Selector ablation only. Baseline and blind-static rankers choose candidates using visible "
            "prompt and candidate AST/static features. The full-pool oracle uses private verifier outcomes "
            "only to diagnose whether any generated candidate already passes; it does not train, route, "
            "or promote the generator."
        ),
    }


def selector_mode_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_count": int(report.get("task_count") or 0),
        "attempt_count": int(report.get("attempt_count") or 0),
        "compile_task_rate": float(report.get("compile_task_rate") or 0.0),
        "runtime_load_task_rate": float(report.get("runtime_load_task_rate") or 0.0),
        "intended_behavior_passed_task_count": int(report.get("intended_behavior_passed_task_count") or 0),
        "intended_behavior_pass_rate": float(report.get("intended_behavior_pass_rate") or 0.0),
        "stage_counts": dict(report.get("stage_counts") if isinstance(report.get("stage_counts"), dict) else {}),
    }


def blind_static_selector_key(row: dict[str, Any], prompt_tags: list[str]) -> tuple[float, int, float, str]:
    score = blind_static_selector_score(row, prompt_tags)
    rank, neg_rank_score, sha = candidate_rank_key(row)
    return (-score, rank, neg_rank_score, sha)


def blind_static_selector_score(row: dict[str, Any], prompt_tags: list[str]) -> float:
    features = candidate_blind_code_features(str(row.get("code") or ""))
    issues = blind_candidate_issue_labels(features, prompt_tags)
    score = 0.0
    score += 2.0 if bool(features.get("syntax_valid")) else -20.0
    score += 1.5 if bool(features.get("uses_parameter")) else -3.0
    score += 1.0 if bool(features.get("has_return")) else -5.0
    score += 0.4 * min(6, int(features.get("statement_count") or 0))
    if bool(features.get("returns_none")):
        score -= 6.0
    if bool(features.get("returns_constant")) and not bool(features.get("returns_collection_literal")):
        score -= 4.0
    if bool(features.get("returns_parameter_directly")):
        score -= 4.0
    if any(tag in prompt_tags for tag in ["parsing_text", "grouping_or_dedup", "window_or_sequence", "aggregation"]):
        score += 1.25 if bool(features.get("has_loop")) else -1.25
    if any(tag in prompt_tags for tag in ["filtering", "parsing_text", "graph_or_grid"]):
        score += 1.0 if bool(features.get("has_branch")) else -1.0
    if "parsing_text" in prompt_tags:
        score += 1.25 if bool(features.get("has_string_method_call")) else -1.25
    if "structured_output" in prompt_tags:
        score += 1.0 if bool(features.get("has_collection_literal")) else -1.0
    score -= 0.75 * sum(1 for issue in issues if issue != "blind_static_shape_plausible")
    try:
        model_score = float(row.get("rank_score") or 0.0)
    except (TypeError, ValueError):
        model_score = 0.0
    score += max(-0.5, min(0.5, model_score))
    return round(score, 6)


def prompt_feature_tags(prompt: str) -> list[str]:
    text = f" {prompt.lower()} "
    tags: list[str] = []
    groups = {
        "parsing_text": ["parse", "extract", "text", "string", "query", "encode", "decode", "regex", "noisy"],
        "aggregation": ["count", "sum", "total", "max", "min", "average", "mean", "median", "score"],
        "ordering": ["sort", "order", "rank", "top", "largest", "smallest"],
        "filtering": ["filter", "threshold", "without", "remove", "ignore", "valid", "invalid"],
        "grouping_or_dedup": ["group", "dedup", "unique", "frequency", "histogram", "pairs"],
        "window_or_sequence": ["window", "subsequence", "substring", "consecutive", "adjacent", "run-length"],
        "graph_or_grid": ["graph", "grid", "matrix", "path", "neighbor", "connected"],
        "structured_output": ["dict", "list", "tuple", "pairs", "mapping", "key", "values"],
    }
    for tag, needles in groups.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    return tags or ["generic_transform"]


def candidate_blind_code_features(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "syntax_valid": False,
            "has_loop": False,
            "has_branch": False,
            "has_return": False,
            "returns_none": False,
            "returns_constant": False,
            "returns_parameter_directly": False,
            "uses_parameter": False,
            "has_collection_literal": False,
            "has_subscript": False,
            "has_string_method_call": False,
            "has_numeric_or_len_call": False,
            "statement_count": 0,
        }
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    args = [arg.arg for arg in getattr(getattr(function, "args", None), "args", [])] if function else []
    arg_names = set(args)
    names_loaded = {
        node.id
        for node in ast.walk(function or tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    returns = [node for node in ast.walk(function or tree) if isinstance(node, ast.Return)]
    call_names = [call_name(node) for node in ast.walk(function or tree) if isinstance(node, ast.Call)]
    collection_literals = (ast.List, ast.Dict, ast.Set, ast.Tuple, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)
    return_values = [node.value for node in returns if node.value is not None]
    return_name_values = [node.id for node in return_values if isinstance(node, ast.Name)]
    return_constants = [node for node in return_values if isinstance(node, ast.Constant)]
    return_collections = [node for node in return_values if isinstance(node, collection_literals)]
    return_subscripts = [node for node in return_values if isinstance(node, ast.Subscript)]
    string_methods = {"split", "strip", "lstrip", "rstrip", "isdigit", "isalpha", "isalnum", "lower", "upper", "replace", "join", "startswith", "endswith"}
    numeric_or_len = {"len", "sum", "max", "min", "sorted", "range", "enumerate", "zip", "int", "float", "abs"}
    return {
        "syntax_valid": True,
        "has_loop": any(isinstance(node, (ast.For, ast.While, ast.comprehension)) for node in ast.walk(function or tree)),
        "has_branch": any(isinstance(node, (ast.If, ast.IfExp, ast.Try, ast.BoolOp, ast.Compare)) for node in ast.walk(function or tree)),
        "has_return": bool(returns),
        "returns_none": any(isinstance(node, ast.Constant) and node.value is None for node in return_values),
        "returns_constant": bool(return_constants),
        "returns_parameter_directly": any(name in arg_names for name in return_name_values),
        "returns_collection_literal": bool(return_collections),
        "returns_subscript": bool(return_subscripts),
        "uses_parameter": bool(arg_names & names_loaded),
        "has_collection_literal": any(isinstance(node, collection_literals) for node in ast.walk(function or tree)),
        "has_subscript": any(isinstance(node, ast.Subscript) for node in ast.walk(function or tree)),
        "has_string_method_call": any(name in string_methods for name in call_names),
        "has_numeric_or_len_call": any(name in numeric_or_len for name in call_names),
        "statement_count": len(getattr(function, "body", []) if function else tree.body),
    }


def blind_candidate_issue_labels(features: dict[str, Any], prompt_tags: list[str]) -> list[str]:
    labels: list[str] = []
    if not bool(features.get("syntax_valid")):
        return ["syntax_invalid"]
    if not bool(features.get("has_return")):
        labels.append("missing_return")
    if bool(features.get("returns_none")):
        labels.append("returns_none")
    if bool(features.get("returns_constant")) and not bool(features.get("returns_collection_literal")):
        labels.append("constant_scalar_return")
    if bool(features.get("returns_parameter_directly")):
        labels.append("returns_input_parameter_directly")
    if not bool(features.get("uses_parameter")):
        labels.append("does_not_use_input_parameter")
    if any(tag in prompt_tags for tag in ["parsing_text", "window_or_sequence", "grouping_or_dedup"]) and not bool(features.get("has_loop")):
        labels.append("prompt_implies_iteration_but_no_loop")
    if any(tag in prompt_tags for tag in ["filtering", "parsing_text", "graph_or_grid"]) and not bool(features.get("has_branch")):
        labels.append("prompt_implies_branching_but_no_branch")
    if "parsing_text" in prompt_tags and not bool(features.get("has_string_method_call")):
        labels.append("prompt_implies_string_processing_but_no_string_ops")
    if any(tag in prompt_tags for tag in ["aggregation", "ordering", "window_or_sequence"]) and not (
        bool(features.get("has_loop")) or bool(features.get("has_numeric_or_len_call"))
    ):
        labels.append("prompt_implies_aggregation_but_no_loop_or_numeric_call")
    if "structured_output" in prompt_tags and not (
        bool(features.get("has_collection_literal")) or bool(features.get("returns_subscript"))
    ):
        labels.append("prompt_implies_structured_output_but_no_collection_construction")
    return labels or ["blind_static_shape_plausible"]


def blind_repair_targets(labels: list[str]) -> list[str]:
    targets: list[str] = []
    if any(label in labels for label in ["prompt_implies_string_processing_but_no_string_ops", "prompt_implies_iteration_but_no_loop"]):
        targets.append("add_prompt_to_algorithm_operator_auxiliary_for_parse_iterate_transform_without_solution_or_tests")
    if any(label in labels for label in ["returns_none", "constant_scalar_return", "returns_input_parameter_directly"]):
        targets.append("increase_private_penalty_for_constant_or_input_echo_returns_in_direct_body_token_training")
    if "prompt_implies_structured_output_but_no_collection_construction" in labels:
        targets.append("train_blind_structured_output_construction_features_from_private_licensed_rows")
    if any(label in labels for label in ["prompt_implies_branching_but_no_branch", "prompt_implies_aggregation_but_no_loop_or_numeric_call"]):
        targets.append("add_visible_prompt_control_flow_feature_head_and_ranker_ablation")
    return targets or ["mine_loaded_behavior_failures_for_prompt_ast_feature_mismatch_before_public_calibration"]


def call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def build_viea_records(summary: dict[str, Any], *, trigger_state: str) -> list[dict[str, Any]]:
    run_suffix = sha256_json(
        {
            "candidate_manifest": summary.get("candidate_manifest"),
            "eligible": summary.get("eligible_full_body_candidate_count"),
            "combined": summary.get("combined"),
        }
    )[:16]
    run_id = f"strict_generator_fanout_receipt-{run_suffix}"
    claim_id = f"claim_strict_generator_fanout_receipt-{run_suffix}"
    support_state = "SUPPORTED" if trigger_state in {"GREEN", "YELLOW"} else "BLOCKED"
    combined = summary.get("combined") if isinstance(summary.get("combined"), dict) else {}
    common = {
        "run_id": run_id,
        "producer_surface": "neural_seed_strict_generator_fanout_receipt",
        "support_state": support_state,
        "candidate_count": int(summary.get("candidate_count") or 0),
        "candidate_attempt_count": int(combined.get("attempt_count") or 0),
        "integrity_verified_candidate_count": int(summary.get("eligible_full_body_candidate_count") or 0),
        "integrity_mismatch_count": int(summary.get("integrity_mismatch_count") or 0),
        "syntax_invalid_count": int(summary.get("syntax_invalid_count") or 0),
        "runtime_load_rate": float(combined.get("runtime_load_task_rate") or 0.0),
        "intended_behavior_pass_rate": float(combined.get("intended_behavior_pass_rate") or 0.0),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return [
        {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": f"strict_generator_authority-{run_suffix}",
            "authority_scope": ["private_eval_rows", "local_private_verifier", "candidate_integrity_replay"],
            "state": "local_private_replay_no_public_calibration_no_external_inference",
        },
        {
            **common,
            "record_type": "runtime_adapter_invocation",
            "record_id": f"strict_generator_runtime_adapter-{run_suffix}",
            "adapter_id": "code_lm_private_verifier.run_any",
            "status": trigger_state,
        },
        {
            **common,
            "record_type": "resource_budget",
            "record_id": f"strict_generator_resource_budget-{run_suffix}",
            "budget_id": f"strict_generator_replay_budget-{run_suffix}",
            "replay_limits": {
                "max_tasks": summary.get("max_tasks"),
                "max_candidates_per_task": summary.get("max_candidates_per_task"),
            },
            "heavy_training_started_by_record": False,
        },
        {
            **common,
            "record_type": "generation_mode",
            "record_id": f"strict_generator_generation_mode-{run_suffix}",
            "candidate_generation_credit": int(summary.get("eligible_full_body_candidate_count") or 0),
            "learned_generation_claim_allowed": False,
            "promotion_state": "replay_evidence_only_not_model_promotion",
            "non_claim": "No renderer/router/template/tool/fallback row is credited as learned full-body generation.",
        },
        {
            **common,
            "record_type": "failure_boundary",
            "record_id": f"strict_generator_failure_boundary-{run_suffix}",
            "fallback_return_used": False,
            "structured_non_solved": float(combined.get("intended_behavior_pass_rate") or 0.0) <= 0.0,
            "terminal": False,
            "status": trigger_state,
        },
        {
            **common,
            "record_type": "artifact_graph_record",
            "record_id": f"strict_generator_artifact-{run_suffix}",
            "artifact_kind": "strict_generator_fanout_replay_receipt",
            "source_refs": [str(summary.get("candidate_manifest")), str(summary.get("private_eval_jsonl"))],
            "content_hash": sha256_json(summary),
            "evidence_ref": rel(DEFAULT_OUT),
        },
        {
            **common,
            "record_type": "proof_carrying_claim",
            "record_id": f"strict_generator_proof_claim-{run_suffix}",
            "proof_claim_id": claim_id,
            "proof_status": support_state,
            "verifier_surface": "code_lm_private_verifier",
            "candidate_family": "learned_full_body_token",
            "evidence_ref": rel(DEFAULT_OUT),
        },
        {
            **common,
            "record_type": "claim_record",
            "record_id": f"strict_generator_claim-{run_suffix}",
            "claim_id": claim_id,
            "state": "strict_full_body_token_replay_measured",
            "status": trigger_state,
            "claim_boundary": "syntax_loadability_behavior_measurement_not_public_transfer_or_promotion",
            "evidence_ref": rel(DEFAULT_OUT),
        },
        {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": f"strict_generator_evidence_transition-{run_suffix}",
            "state": "candidate_manifest_to_integrity_audit_to_private_verifier_replay",
            "status": support_state,
            "evidence_ref": rel(DEFAULT_OUT),
        },
    ]


def compact_integrity(integrity: dict[str, Any]) -> dict[str, Any]:
    return {
        "recomputed_candidate_family": integrity.get("recomputed_candidate_family"),
        "integrity_verified": bool(integrity.get("integrity_verified")),
        "pure_learned_generation": bool(integrity.get("pure_learned_generation")),
        "candidate_family_confidence": integrity.get("candidate_family_confidence"),
        "candidate_sha256": integrity.get("candidate_sha256"),
    }


def candidate_rank_key(row: dict[str, Any]) -> tuple[int, float, str]:
    try:
        rank = int(row.get("rank") or 9999)
    except (TypeError, ValueError):
        rank = 9999
    try:
        score = float(row.get("rank_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return rank, -score, str(row.get("candidate_sha256") or "")


def python_syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def ratio(num: int | float, den: int | float) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    combined = summary.get("combined") if isinstance(summary.get("combined"), dict) else {}
    selector = summary.get("selector_ablation") if isinstance(summary.get("selector_ablation"), dict) else {}
    lines = [
        "# Neural Seed Strict Generator Fanout Receipt",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- candidate_count: `{summary.get('candidate_count')}`",
        f"- eligible_full_body_candidate_count: `{summary.get('eligible_full_body_candidate_count')}`",
        f"- replayed_task_count: `{summary.get('replayed_task_count')}`",
        f"- syntax_valid_rate: `{summary.get('syntax_valid_rate')}`",
        f"- runtime_load_task_rate: `{combined.get('runtime_load_task_rate')}`",
        f"- intended_behavior_pass_rate: `{combined.get('intended_behavior_pass_rate')}`",
        f"- selector_diagnosis: `{selector.get('selector_diagnosis')}`",
        f"- baseline_to_oracle_pass_delta: `{selector.get('baseline_to_oracle_pass_delta')}`",
        f"- hard_gaps: `{report.get('hard_gaps')}`",
        "",
        "## Semantics",
        "",
        str(report.get("score_semantics") or ""),
        "",
    ]
    return "\n".join(lines)


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    combined = summary.get("combined") if isinstance(summary.get("combined"), dict) else {}
    by_arm = summary.get("by_arm") if isinstance(summary.get("by_arm"), dict) else {}
    selector = summary.get("selector_ablation") if isinstance(summary.get("selector_ablation"), dict) else {}
    return {
        "trigger_state": report.get("trigger_state"),
        "hard_gaps": report.get("hard_gaps"),
        "candidate_count": summary.get("candidate_count"),
        "eligible_full_body_candidate_count": summary.get("eligible_full_body_candidate_count"),
        "replayed_task_count": summary.get("replayed_task_count"),
        "syntax_valid_rate": summary.get("syntax_valid_rate"),
        "combined": {
            "task_count": combined.get("task_count"),
            "attempt_count": combined.get("attempt_count"),
            "runtime_load_task_rate": combined.get("runtime_load_task_rate"),
            "intended_behavior_pass_rate": combined.get("intended_behavior_pass_rate"),
            "stage_counts": combined.get("stage_counts"),
        },
        "selector_ablation": {
            "selector_diagnosis": selector.get("selector_diagnosis"),
            "baseline_to_blind_pass_delta": selector.get("baseline_to_blind_pass_delta"),
            "baseline_to_oracle_pass_delta": selector.get("baseline_to_oracle_pass_delta"),
        },
        "by_arm": {
            arm: {
                "task_count": row.get("task_count"),
                "attempt_count": row.get("attempt_count"),
                "runtime_load_task_rate": row.get("runtime_load_task_rate"),
                "intended_behavior_pass_rate": row.get("intended_behavior_pass_rate"),
            }
            for arm, row in by_arm.items()
            if isinstance(row, dict)
        },
    }


def sha256_json(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
