#!/usr/bin/env python3
"""Gate whether broad-private transfer is learned-token behavior.

The broad-private score gate can be GREEN while the pass path is still a
private-train prototype bridge. This report keeps that useful private evidence
from being mistaken for promotion-grade learned decoder behavior.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus_archive_resolver import read_jsonl_follow_pointer
from readiness_freshness import freshness_report


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PRIVATE_TRAIN = ROOT / "data" / "training_data" / "high_transfer" / "private_train"
DEFAULT_HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "broad_private_generalization_ladder_v1_heldout_code_lm_tasks.jsonl"
DEFAULT_CANDIDATES = REPORTS / "code_lm_private_candidates_broad_private_generalization_ladder_v1_heldout.jsonl"
DEFAULT_CONTROL = REPORTS / "code_lm_private_candidates_broad_private_generalization_ladder_v1_heldout_sts_off.jsonl"
DEFAULT_SCORE = REPORTS / "broad_private_generalization_score_v1.json"
DEFAULT_LEARNED_ONLY = REPORTS / "code_lm_private_candidates_broad_private_generalization_ladder_v1_heldout_learned_only.jsonl"
DEFAULT_LEARNED_SCORE = REPORTS / "broad_private_generalization_score_v1_learned_only.json"
DEFAULT_LEARNED_SCORE_MD = REPORTS / "broad_private_generalization_score_v1_learned_only.md"
DEFAULT_STRICT_LEARNED_ONLY = REPORTS / "code_lm_private_candidates_broad_private_generalization_ladder_v1_heldout_strict_novel_learned_only.jsonl"
DEFAULT_STRICT_LEARNED_SCORE = REPORTS / "broad_private_generalization_score_v1_strict_novel_learned_only.json"
DEFAULT_STRICT_LEARNED_SCORE_MD = REPORTS / "broad_private_generalization_score_v1_strict_novel_learned_only.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default=rel(DEFAULT_HELDOUT))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--control-candidates", default=rel(DEFAULT_CONTROL))
    parser.add_argument("--score", default=rel(DEFAULT_SCORE))
    parser.add_argument("--private-train", default="")
    parser.add_argument("--learned-only-candidates-out", default=rel(DEFAULT_LEARNED_ONLY))
    parser.add_argument("--learned-only-score-out", default=rel(DEFAULT_LEARNED_SCORE))
    parser.add_argument("--learned-only-score-markdown-out", default=rel(DEFAULT_LEARNED_SCORE_MD))
    parser.add_argument("--strict-novel-learned-only-candidates-out", default=rel(DEFAULT_STRICT_LEARNED_ONLY))
    parser.add_argument("--strict-novel-learned-only-score-out", default=rel(DEFAULT_STRICT_LEARNED_SCORE))
    parser.add_argument("--strict-novel-learned-only-score-markdown-out", default=rel(DEFAULT_STRICT_LEARNED_SCORE_MD))
    parser.add_argument("--timeout-seconds", type=int, default=2)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--min-heldout-rows", type=int, default=1000)
    parser.add_argument("--max-train-normalized-overlap-rate", type=float, default=0.05)
    parser.add_argument("--skip-learned-score", action="store_true")
    parser.add_argument("--out", default="reports/broad_private_learned_distillation_gate_v1.json")
    parser.add_argument("--markdown-out", default="reports/broad_private_learned_distillation_gate_v1.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = resolve(args.candidates)
    heldout_path = resolve(args.heldout)
    private_train_path = resolve(args.private_train) if str(args.private_train or "").strip() else infer_private_train_path(heldout_path)
    score_path = resolve(args.score)
    learned_path = resolve(args.learned_only_candidates_out)
    learned_score_path = resolve(args.learned_only_score_out)
    learned_score_md_path = resolve(args.learned_only_score_markdown_out)
    strict_learned_path = resolve(args.strict_novel_learned_only_candidates_out)
    strict_learned_score_path = resolve(args.strict_novel_learned_only_score_out)
    strict_learned_score_md_path = resolve(args.strict_novel_learned_only_score_markdown_out)

    candidates = read_jsonl(candidate_path)
    private_train_rows = read_jsonl(private_train_path) if private_train_path is not None and private_train_path.exists() else []
    private_train_core_body_signatures = {
        normalized_core_body_signature(solution_code(row))
        for row in private_train_rows
    }
    private_train_core_body_signatures.discard("")
    learned_candidate_inventory = [row for row in candidates if learned_token_candidate(row)]
    exact_train_body_memory_rows = [
        row
        for row in learned_candidate_inventory
        if exact_train_body_memory_candidate(row, private_train_core_body_signatures)
    ]
    strict_novel_learned_only = [
        row
        for row in learned_candidate_inventory
        if not exact_train_body_memory_candidate(row, private_train_core_body_signatures)
    ]
    # Learned-proof manifests intentionally exclude exact private train-body
    # memory rows. Those rows remain visible in candidate_inventory as
    # diagnostic evidence, but cannot carry learned-token proof.
    learned_only = strict_novel_learned_only
    write_jsonl(learned_path, learned_only)
    write_jsonl(strict_learned_path, strict_novel_learned_only)

    commands: list[dict[str, Any]] = []
    if not args.skip_learned_score:
        cmd = [
            sys.executable,
            "scripts/broad_private_generalization_score_v1.py",
            "--heldout",
            rel(heldout_path),
            "--candidates",
            rel(learned_path),
            "--control-candidates",
            rel(resolve(args.control_candidates)),
            "--timeout-seconds",
            str(max(1, int(args.timeout_seconds))),
            "--task-limit",
            str(max(0, int(args.task_limit))),
            "--min-heldout-rows",
            str(max(1, int(args.min_heldout_rows))),
            "--out",
            rel(learned_score_path),
            "--markdown-out",
            rel(learned_score_md_path),
        ]
        commands.append(run_command(cmd))
        strict_cmd = [
            sys.executable,
            "scripts/broad_private_generalization_score_v1.py",
            "--heldout",
            rel(heldout_path),
            "--candidates",
            rel(strict_learned_path),
            "--control-candidates",
            rel(resolve(args.control_candidates)),
            "--timeout-seconds",
            str(max(1, int(args.timeout_seconds))),
            "--task-limit",
            str(max(0, int(args.task_limit))),
            "--min-heldout-rows",
            str(max(1, int(args.min_heldout_rows))),
            "--out",
            rel(strict_learned_score_path),
            "--markdown-out",
            rel(strict_learned_score_md_path),
        ]
        commands.append(run_command(strict_cmd))

    score = read_json(score_path, {})
    learned_score = read_json(learned_score_path, {})
    strict_learned_score = read_json(strict_learned_score_path, {})
    score_summary = score.get("summary") if isinstance(score.get("summary"), dict) else {}
    learned_summary = learned_score.get("summary") if isinstance(learned_score.get("summary"), dict) else {}
    strict_learned_summary = strict_learned_score.get("summary") if isinstance(strict_learned_score.get("summary"), dict) else {}
    full_results = score.get("results") if isinstance(score.get("results"), list) else []
    strict_results = strict_learned_score.get("results") if isinstance(strict_learned_score.get("results"), list) else []
    full_pass_count = int(score_summary.get("pass_count") or 0)
    full_task_count = int(score_summary.get("heldout_task_count") or 0)
    candidate_index = index_candidates(candidates)
    strict_candidate_index = index_candidates(strict_novel_learned_only)
    pass_inventory = pass_inventory_summary(full_results, candidate_index, private_train_core_body_signatures)
    candidate_inventory = candidate_inventory_summary(
        candidates,
        learned_candidate_inventory,
        learned_only,
        exact_train_body_memory_rows,
        strict_novel_learned_only,
    )
    learned_pass_rows = learned_pass_candidate_rows(full_results, candidate_index)
    strict_learned_pass_rows = learned_pass_candidate_rows(strict_results, strict_candidate_index)
    structural_inventory = learned_structural_inventory_summary(strict_novel_learned_only, strict_learned_pass_rows, full_task_count)
    train_novelty_inventory = learned_train_novelty_summary(
        strict_learned_pass_rows,
        private_train_path,
        max_overlap_rate=max(0.0, float(args.max_train_normalized_overlap_rate)),
    )

    mode_passes = score_summary.get("mode_passes") if isinstance(score_summary.get("mode_passes"), dict) else {}
    learned_pass_rate = numeric(learned_summary.get("pass_rate"), 0.0)
    strict_learned_pass_rate = numeric(strict_learned_summary.get("pass_rate"), 0.0)
    prototype_pass_count = int(pass_inventory.get("prototype_pass_count") or 0)
    exact_train_body_memory_pass_count = int(pass_inventory.get("exact_train_body_memory_pass_count") or 0)
    prototype_rows = int(candidate_inventory.get("prototype_rows") or 0)
    exact_train_body_memory_rate = numeric(candidate_inventory.get("exact_train_body_memory_candidate_rate"), 0.0)
    prototype_verifier_rate = numeric(candidate_inventory.get("prototype_verifier_admissible_rate"), 0.0)
    mode_accounting_sum = sum(int(value or 0) for value in mode_passes.values())
    current_evidence = current_decoder_evidence(
        candidate_path,
        score_path,
        learned_path,
        learned_score_path,
        strict_learned_path,
        strict_learned_score_path,
        resolve(args.control_candidates),
    )

    gates = [
        gate("full_broad_private_score_present", bool(score_summary), rel(score_path)),
        gate("learned_distillation_artifacts_current_for_decoder_source_and_release", current_evidence["fresh"], current_evidence),
        gate("full_broad_private_pass_rate_ge_070", numeric(score_summary.get("pass_rate"), 0.0) >= 0.70, score_summary.get("pass_rate")),
        gate("mode_pass_accounting_complete", mode_accounting_sum == full_pass_count, {"mode_passes_sum": mode_accounting_sum, "pass_count": full_pass_count}),
        gate("learned_only_score_present", bool(learned_summary), rel(learned_score_path)),
        gate("learned_token_pass_rate_ge_070", learned_pass_rate >= 0.70, {"observed": learned_pass_rate, "minimum": 0.70}),
        gate("strict_novel_learned_only_score_present", bool(strict_learned_summary), rel(strict_learned_score_path)),
        gate("strict_novel_learned_token_pass_rate_ge_070", strict_learned_pass_rate >= 0.70, {"observed": strict_learned_pass_rate, "minimum": 0.70}),
        gate(
            "exact_train_body_memory_not_dominant",
            exact_train_body_memory_pass_count == 0,
            {
                "exact_train_body_memory_pass_count": exact_train_body_memory_pass_count,
                "exact_train_body_memory_candidate_rate": exact_train_body_memory_rate,
                "exact_train_body_memory_candidate_rows": len(exact_train_body_memory_rows),
                "excluded_from_learned_proof": True,
                "strict_rule": "strict novel learned evidence excludes exact private train solution-body replay",
            },
        ),
        gate("learned_pass_normalized_ast_diversity_ge_min", int(structural_inventory["pass_normalized_ast_unique_count"]) >= int(structural_inventory["min_pass_normalized_ast_unique_count"]), structural_inventory),
        gate("learned_pass_ast_shape_diversity_ge_min", int(structural_inventory["pass_ast_shape_count"]) >= int(structural_inventory["min_pass_ast_shape_count"]), structural_inventory),
        gate("learned_pass_duplicate_concentration_le_max", numeric(structural_inventory["pass_top_normalized_ast_duplicate_rate"], 1.0) <= numeric(structural_inventory["max_pass_top_duplicate_rate"], 0.0), structural_inventory),
        gate("learned_pass_control_structure_coverage", structural_inventory["control_structure_coverage_ready"], structural_inventory),
        gate("learned_pass_private_train_novelty_ready", train_novelty_inventory["novelty_ready"], train_novelty_inventory),
        gate("prototype_dependency_not_dominant", prototype_pass_count == 0, {"prototype_pass_count": prototype_pass_count, "full_pass_count": full_pass_count}),
        gate("verifier_admissible_prototype_rate_ge_095", prototype_rows == 0 or prototype_verifier_rate >= 0.95, {"prototype_rows": prototype_rows, "prototype_verifier_admissible_rate": candidate_inventory.get("prototype_verifier_admissible_rate")}),
        gate("public_data_not_used", true(score_summary.get("public_tests_used")) is False and true(score_summary.get("public_solutions_used")) is False, {"public_tests_used": score_summary.get("public_tests_used"), "public_solutions_used": score_summary.get("public_solutions_used")}),
        gate("external_inference_zero", int(score_summary.get("external_inference_calls") or 0) == 0, score_summary.get("external_inference_calls")),
    ]
    hard_failures = [row for row in gates if not row["passed"] and row["gate"] in {"full_broad_private_score_present", "public_data_not_used", "external_inference_zero"}]
    blockers = [row for row in gates if not row["passed"]]
    trigger_state = "RED" if hard_failures else ("GREEN" if not blockers else "YELLOW")
    completion_evidence_status = "learned_distillation_ready"
    if trigger_state != "GREEN":
        if not current_evidence.get("fresh"):
            completion_evidence_status = "stale_learned_distillation_evidence"
        elif exact_train_body_memory_pass_count > 0:
            completion_evidence_status = "exact_train_body_memory_blocker"
        elif strict_learned_pass_rate < 0.70:
            completion_evidence_status = "strict_novel_learned_transfer_below_floor"
        elif any(str(row.get("gate") or "").startswith("learned_pass_") for row in blockers):
            completion_evidence_status = "learned_structural_maturity_blocker"
        elif prototype_pass_count > 0 or prototype_rows > 0:
            completion_evidence_status = "prototype_dependent_transfer_blocker"
        else:
            completion_evidence_status = "learned_transfer_below_floor"
    return {
        "policy": "project_theseus_broad_private_learned_distillation_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "score": rel(score_path),
            "candidates": rel(candidate_path),
            "heldout": rel(heldout_path),
            "private_train": rel(private_train_path) if private_train_path else "",
            "control_candidates": rel(resolve(args.control_candidates)),
            "public_tests_used": False,
            "public_solutions_used": False,
            "task_limit": max(0, int(args.task_limit)),
            "min_heldout_rows": max(1, int(args.min_heldout_rows)),
        },
        "outputs": {
            "learned_only_candidates": rel(learned_path),
            "learned_only_score": rel(learned_score_path),
            "learned_only_score_markdown": rel(learned_score_md_path),
            "strict_novel_learned_only_candidates": rel(strict_learned_path),
            "strict_novel_learned_only_score": rel(strict_learned_score_path),
            "strict_novel_learned_only_score_markdown": rel(strict_learned_score_md_path),
        },
        "summary": {
            "completion_evidence_status": completion_evidence_status,
            "full_pass_count": full_pass_count,
            "full_task_count": full_task_count,
            "full_pass_rate": score_summary.get("pass_rate"),
            "learned_only_pass_count": learned_summary.get("pass_count"),
            "learned_only_task_count": learned_summary.get("heldout_task_count"),
            "learned_only_pass_rate": learned_summary.get("pass_rate"),
            "learned_only_no_admissible_task_rate": learned_summary.get("no_admissible_task_rate"),
            "strict_novel_learned_only_pass_count": strict_learned_summary.get("pass_count"),
            "strict_novel_learned_only_task_count": strict_learned_summary.get("heldout_task_count"),
            "strict_novel_learned_only_pass_rate": strict_learned_summary.get("pass_rate"),
            "strict_novel_learned_only_no_admissible_task_rate": strict_learned_summary.get("no_admissible_task_rate"),
            "prototype_pass_count": prototype_pass_count,
            "exact_train_body_memory_pass_count": exact_train_body_memory_pass_count,
            "learned_token_pass_count": pass_inventory.get("learned_token_pass_count"),
            "verifier_admissible_pass_count": pass_inventory.get("verifier_admissible_pass_count"),
            "mode_passes_sum": mode_accounting_sum,
            "mode_passes_count": len(mode_passes),
            "candidate_inventory": candidate_inventory,
            "pass_inventory": pass_inventory,
            "learned_structural_inventory": structural_inventory,
            "learned_train_novelty_inventory": train_novelty_inventory,
            "decoder_source_release_fresh": current_evidence["fresh"],
            "decoder_source_release_stale_reasons": current_evidence["stale_reasons"],
            "blocker_count": len(blockers),
        },
        "gates": gates,
        "blockers": blockers,
        "commands": commands,
        "next_actions": next_actions(trigger_state, blockers, candidate_inventory, learned_summary, strict_learned_summary, current_evidence),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }


def current_decoder_evidence(
    candidate_path: Path,
    score_path: Path,
    learned_path: Path,
    learned_score_path: Path,
    strict_learned_path: Path,
    strict_learned_score_path: Path,
    control_path: Path,
) -> dict[str, Any]:
    artifacts = {
        "full_candidates": candidate_path,
        "full_score": score_path,
        "learned_only_candidates": learned_path,
        "learned_only_score": learned_score_path,
        "strict_novel_learned_only_candidates": strict_learned_path,
        "strict_novel_learned_only_score": strict_learned_score_path,
    }
    if str(control_path):
        artifacts["control_candidates"] = control_path
    return freshness_report(
        artifacts,
        root=ROOT,
        rule=(
            "learned-distillation evidence must be regenerated after decoder "
            "source changes or after rebuilding target/release/symliquid-cli"
        ),
    )


def learned_token_candidate(row: dict[str, Any]) -> bool:
    mode = str(row.get("candidate_generation_mode") or "").lower()
    return (
        true(row.get("token_level_code_generation_learned"))
        and true(row.get("candidate_syntax_lint_passed"))
        and row.get("deterministic_guardrail_passed") is not False
        and row.get("decoder_contract_verifier_v1_passed") is not False
        and not true(row.get("broad_private_train_prototype_stage"))
        and not true(row.get("broad_private_generalization_semantic_adapter_stage"))
        and not true(row.get("private_residual_v3_semantic_adapter_stage"))
        and not true(row.get("contract_transduced_stage"))
        and not true(row.get("same_seed_non_sts_comparator"))
        and "contract_transduced_token_decoder" not in mode
        and "body_memory_replay" not in mode
    )


def exact_train_body_memory_candidate(row: dict[str, Any], private_train_core_body_signatures: set[str]) -> bool:
    if not private_train_core_body_signatures:
        return False
    return normalized_core_body_signature(str(row.get("code") or "")) in private_train_core_body_signatures


def index_candidates(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("task_id") or ""), str(row.get("candidate_generation_mode") or ""))
        out.setdefault(key, []).append(row)
    return out


def pass_inventory_summary(
    results: list[dict[str, Any]],
    index: dict[tuple[str, str], list[dict[str, Any]]],
    private_train_core_body_signatures: set[str],
) -> dict[str, Any]:
    out = {
        "passed_result_count": 0,
        "prototype_pass_count": 0,
        "diagnostic_adapter_pass_count": 0,
        "learned_token_pass_count": 0,
        "exact_train_body_memory_pass_count": 0,
        "verifier_admissible_pass_count": 0,
        "unindexed_pass_count": 0,
        "ambiguous_pass_mode_count": 0,
    }
    examples = []
    for result in results:
        if not true(result.get("passed")):
            continue
        out["passed_result_count"] += 1
        task_id = str(result.get("task_id") or "")
        mode = str(result.get("pass_candidate_mode") or "")
        rows = index.get((task_id, mode), [])
        if not rows:
            out["unindexed_pass_count"] += 1
            if len(examples) < 5:
                examples.append({"task_id": task_id, "mode": mode})
            continue
        if len(rows) > 1:
            out["ambiguous_pass_mode_count"] += 1
        if any(true(row.get("broad_private_train_prototype_stage")) for row in rows):
            out["prototype_pass_count"] += 1
        if any(true(row.get("broad_private_generalization_semantic_adapter_stage")) or true(row.get("private_residual_v3_semantic_adapter_stage")) for row in rows):
            out["diagnostic_adapter_pass_count"] += 1
        if any(
            learned_token_candidate(row)
            and not exact_train_body_memory_candidate(row, private_train_core_body_signatures)
            for row in rows
        ):
            out["learned_token_pass_count"] += 1
        if any(exact_train_body_memory_candidate(row, private_train_core_body_signatures) for row in rows):
            out["exact_train_body_memory_pass_count"] += 1
        if any(row.get("decoder_contract_verifier_v1_passed") is True and row.get("deterministic_guardrail_passed") is not False for row in rows):
            out["verifier_admissible_pass_count"] += 1
    out["unindexed_pass_examples"] = examples
    return out


def learned_pass_candidate_rows(results: list[dict[str, Any]], index: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if not true(result.get("passed")):
            continue
        task_id = str(result.get("task_id") or "")
        mode = str(result.get("pass_candidate_mode") or "")
        for candidate in index.get((task_id, mode), []):
            if learned_token_candidate(candidate):
                rows.append(candidate)
                break
    return rows


def candidate_inventory_summary(
    candidates: list[dict[str, Any]],
    learned_candidate_inventory: list[dict[str, Any]],
    learned_only: list[dict[str, Any]],
    exact_train_body_memory_rows: list[dict[str, Any]],
    strict_novel_learned_only: list[dict[str, Any]],
) -> dict[str, Any]:
    prototype = [row for row in candidates if true(row.get("broad_private_train_prototype_stage"))]
    prototype_verifier = [row for row in prototype if row.get("decoder_contract_verifier_v1_passed") is True and row.get("deterministic_guardrail_passed") is not False]
    prototype_rate = round(len(prototype_verifier) / len(prototype), 6) if prototype else 1.0
    exact_memory_rate = round(len(exact_train_body_memory_rows) / max(1, len(learned_candidate_inventory)), 6)
    return {
        "candidate_rows": len(candidates),
        "raw_learned_candidate_rows": len(learned_candidate_inventory),
        "learned_only_candidate_rows": len(learned_only),
        "strict_novel_learned_only_candidate_rows": len(strict_novel_learned_only),
        "exact_train_body_memory_candidate_rows": len(exact_train_body_memory_rows),
        "exact_train_body_memory_candidate_rate": exact_memory_rate,
        "exact_train_body_memory_rows_excluded_from_learned_proof": len(exact_train_body_memory_rows),
        "token_level_code_generation_learned_rows": sum(true(row.get("token_level_code_generation_learned")) for row in candidates),
        "prototype_rows": len(prototype),
        "prototype_verifier_admissible_rows": len(prototype_verifier),
        "prototype_verifier_admissible_rate": prototype_rate,
        "prototype_verifier_failure_reasons": reason_counts(row.get("decoder_contract_verifier_v1_reasons") for row in prototype if row.get("decoder_contract_verifier_v1_passed") is not True),
    }


def learned_structural_inventory_summary(
    learned_only: list[dict[str, Any]],
    learned_pass_rows: list[dict[str, Any]],
    full_task_count: int,
) -> dict[str, Any]:
    pass_count = len(learned_pass_rows)
    min_unique = max(4, min(8, max(1, full_task_count) // 24))
    min_shapes = max(4, min(8, max(1, full_task_count) // 24))
    max_top_duplicate_rate = 0.15
    min_control_rate = 0.75
    pass_normalized = Counter(normalized_ast_signature(str(row.get("code") or "")) for row in learned_pass_rows)
    generated_normalized = Counter(normalized_ast_signature(str(row.get("code") or "")) for row in learned_only)
    shape_counts = Counter(ast_shape_signature(str(row.get("code") or "")) for row in learned_pass_rows)
    coverage = structural_coverage(learned_pass_rows)
    top_pass_duplicate = pass_normalized.most_common(1)[0][1] if pass_normalized else 0
    top_generated_duplicate = generated_normalized.most_common(1)[0][1] if generated_normalized else 0
    return {
        "learned_only_candidate_rows": len(learned_only),
        "learned_pass_candidate_rows": pass_count,
        "pass_normalized_ast_unique_count": len(pass_normalized),
        "pass_normalized_ast_unique_rate": round(len(pass_normalized) / max(1, pass_count), 6),
        "pass_top_normalized_ast_duplicate_count": top_pass_duplicate,
        "pass_top_normalized_ast_duplicate_rate": round(top_pass_duplicate / max(1, pass_count), 6),
        "generated_normalized_ast_unique_count": len(generated_normalized),
        "generated_top_normalized_ast_duplicate_rate": round(top_generated_duplicate / max(1, len(learned_only)), 6),
        "pass_ast_shape_count": len(shape_counts),
        "pass_ast_shape_top_count": shape_counts.most_common(1)[0][1] if shape_counts else 0,
        "pass_loop_rate": coverage["loop_rate"],
        "pass_branch_rate": coverage["branch_rate"],
        "pass_local_assignment_rate": coverage["local_assignment_rate"],
        "pass_call_rate": coverage["call_rate"],
        "min_pass_normalized_ast_unique_count": min_unique,
        "min_pass_ast_shape_count": min_shapes,
        "max_pass_top_duplicate_rate": max_top_duplicate_rate,
        "min_control_structure_rate": min_control_rate,
        "control_structure_coverage_ready": (
            pass_count > 0
            and coverage["loop_rate"] >= min_control_rate
            and coverage["branch_rate"] >= min_control_rate
            and coverage["local_assignment_rate"] >= min_control_rate
            and coverage["call_rate"] >= min_control_rate
        ),
        "score_semantics": "private learned pass-path maturity; normalized AST ignores function names, variable names, and scalar constants",
    }


def learned_train_novelty_summary(
    learned_pass_rows: list[dict[str, Any]],
    private_train_path: Path | None,
    *,
    max_overlap_rate: float,
) -> dict[str, Any]:
    train_rows = read_jsonl(private_train_path) if private_train_path is not None and private_train_path.exists() else []
    pass_count = len(learned_pass_rows)
    train_full = Counter(normalized_ast_signature(solution_code(row)) for row in train_rows)
    train_body = Counter(normalized_core_body_signature(solution_code(row)) for row in train_rows)
    pass_full = Counter(normalized_ast_signature(str(row.get("code") or "")) for row in learned_pass_rows)
    pass_body = Counter(normalized_core_body_signature(str(row.get("code") or "")) for row in learned_pass_rows)
    exact_full_overlap = sum(count for signature, count in pass_full.items() if signature in train_full)
    exact_body_overlap = sum(count for signature, count in pass_body.items() if signature in train_body)
    evaluable = bool(private_train_path and private_train_path.exists() and train_rows and learned_pass_rows)
    full_rate = round(exact_full_overlap / max(1, pass_count), 6)
    body_rate = round(exact_body_overlap / max(1, pass_count), 6)
    novelty_ready = bool(not evaluable or (full_rate <= max_overlap_rate and body_rate <= max_overlap_rate))
    return {
        "private_train": rel(private_train_path) if private_train_path else "",
        "private_train_exists": bool(private_train_path and private_train_path.exists()),
        "train_row_count": len(train_rows),
        "train_normalized_ast_unique_count": len(train_full),
        "train_body_normalized_ast_unique_count": len(train_body),
        "learned_pass_candidate_rows": pass_count,
        "learned_pass_normalized_ast_unique_count": len(pass_full),
        "learned_pass_body_normalized_ast_unique_count": len(pass_body),
        "exact_train_normalized_ast_overlap_count": exact_full_overlap,
        "exact_train_normalized_ast_overlap_rate": full_rate,
        "exact_train_body_normalized_ast_overlap_count": exact_body_overlap,
        "exact_train_body_normalized_ast_overlap_rate": body_rate,
        "max_train_normalized_overlap_rate": max_overlap_rate,
        "novelty_evaluable": evaluable,
        "novelty_ready": novelty_ready,
        "score_semantics": "private train-overlap novelty; core-body AST strips the standard candidate wrapper and ignores function names, argument names, variable names, and scalar constants",
    }


def infer_private_train_path(heldout_path: Path) -> Path | None:
    name = heldout_path.name
    mapping = {
        "broad_private_generalization_ladder_v1_heldout_code_lm_tasks.jsonl": "broad_private_generalization_ladder_v1_code_lm_tasks.jsonl",
        "broad_private_generalization_ladder_v1_semantic_alias_heldout_code_lm_tasks.jsonl": "broad_private_generalization_ladder_v1_code_lm_tasks.jsonl",
        "public_safe_broad_transfer_maturity_v4_heldout_code_lm_tasks.jsonl": "public_safe_broad_transfer_maturity_v4_code_lm_tasks.jsonl",
        "post_v4_private_shadow_transfer_v1_heldout_code_lm_tasks.jsonl": "post_v4_private_shadow_transfer_v1_code_lm_tasks.jsonl",
        "private_ecology_generalization_v5_heldout_code_lm_tasks.jsonl": "private_ecology_generalization_v5_code_lm_tasks.jsonl",
        "private_unseen_transfer_challenge_v1_code_lm_tasks.jsonl": "private_ecology_generalization_v5_code_lm_tasks.jsonl",
    }
    train_name = mapping.get(name)
    return PRIVATE_TRAIN / train_name if train_name else None


def solution_code(row: dict[str, Any]) -> str:
    entry = str(row.get("entry_point") or "candidate")
    body = str(row.get("solution_body") or "")
    if not body.strip():
        expr = str(row.get("solution_expr") or "").strip()
        body = f"return {expr}" if expr else "return None"
    lines = [f"def {entry}(*args):"]
    for line in body.splitlines() or ["return None"]:
        lines.append(f"    {line}" if line.strip() else "")
    return "\n".join(lines) + "\n"


class _AstNormalizer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = "fn"
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = "arg"
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = "name"
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (str, int, float, bool, type(None))):
            node.value = type(node.value).__name__
        return node


def normalized_ast_signature(code: str) -> str:
    try:
        tree = ast.parse(code)
        normalized = _AstNormalizer().visit(tree)
        ast.fix_missing_locations(normalized)
        payload = ast.dump(normalized, include_attributes=False)
    except SyntaxError:
        payload = f"syntax_error:{code[:120]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_function_body_signature(code: str) -> str:
    try:
        tree = ast.parse(code)
        normalized = _AstNormalizer().visit(tree)
        ast.fix_missing_locations(normalized)
        functions = [node for node in ast.walk(normalized) if isinstance(node, ast.FunctionDef)]
        if functions:
            payload_node: ast.AST = ast.Module(body=functions[0].body, type_ignores=[])
            ast.fix_missing_locations(payload_node)
        else:
            payload_node = normalized
        payload = ast.dump(payload_node, include_attributes=False)
    except SyntaxError:
        payload = f"syntax_error:{code[:120]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_core_body_signature(code: str) -> str:
    try:
        tree = ast.parse(code)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        body = list(functions[0].body) if functions else list(tree.body)
        while body and wrapper_prelude_assignment(body[0]):
            body.pop(0)
        payload_node: ast.AST = ast.Module(body=body, type_ignores=[])
        normalized = _AstNormalizer().visit(payload_node)
        ast.fix_missing_locations(normalized)
        payload = ast.dump(normalized, include_attributes=False)
    except SyntaxError:
        payload = f"syntax_error:{code[:120]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def wrapper_prelude_assignment(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign):
        return False
    targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
    return bool(targets) and all(target in {"data", "other", "extra"} for target in targets)


def ast_shape_signature(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "syntax_error"
    interesting = {
        "FunctionDef",
        "For",
        "While",
        "If",
        "Assign",
        "Return",
        "Call",
        "ListComp",
        "DictComp",
        "Compare",
        "BinOp",
        "Subscript",
        "BoolOp",
        "UnaryOp",
    }
    counts = Counter(type(node).__name__ for node in ast.walk(tree))
    return "|".join(f"{name}:{counts[name]}" for name in sorted(interesting) if counts[name])


def structural_coverage(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = {"loop": 0, "branch": 0, "local_assignment": 0, "call": 0}
    for row in rows:
        try:
            tree = ast.parse(str(row.get("code") or ""))
        except SyntaxError:
            continue
        nodes = list(ast.walk(tree))
        counts["loop"] += int(any(isinstance(node, (ast.For, ast.While)) for node in nodes))
        counts["branch"] += int(any(isinstance(node, ast.If) for node in nodes))
        counts["local_assignment"] += int(any(isinstance(node, ast.Assign) for node in nodes))
        counts["call"] += int(any(isinstance(node, ast.Call) for node in nodes))
    total = max(1, len(rows))
    return {f"{name}_rate": round(value / total, 6) for name, value in counts.items()}


def reason_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        reasons = value if isinstance(value, list) and value else ["unspecified"]
        key = ",".join(str(item) for item in reasons)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def next_actions(
    trigger_state: str,
    blockers: list[dict[str, Any]],
    candidate_inventory: dict[str, Any],
    learned_summary: dict[str, Any],
    strict_learned_summary: dict[str, Any],
    current_evidence: dict[str, Any],
) -> list[str]:
    if trigger_state == "GREEN":
        return ["Rerun maturity/readiness before proposing any operator-approved public calibration."]
    actions = []
    if not current_evidence.get("fresh"):
        actions.extend([
            "Rerun broad_private_generalization_unattended_v1.py --execute under the current decoder/release, then rerun this learned-distillation gate.",
            "Do not use stale learned-only pass evidence as transfer proof.",
        ])
    if numeric(candidate_inventory.get("exact_train_body_memory_candidate_rate"), 0.0) > 0.0:
        actions.append(
            "Exact private train-body replay rows are present but excluded from learned-token proof; reduce them at generation source in a future cleanup if they become numerous or ever pass."
        )
    if numeric(strict_learned_summary.get("pass_rate"), 0.0) < 0.70:
        actions.append(
            "Train or refresh the grammar-masked broad-private token path until strict novel learned-only heldout pass rate clears 0.70 without exact train-body memory."
        )
    if numeric(learned_summary.get("pass_rate"), 0.0) < 0.70:
        if int(candidate_inventory.get("prototype_rows") or 0) > 0:
            actions.append("Distill broad-private prototype bodies into the grammar-masked token decoder; learned-only heldout pass rate is below floor.")
        else:
            actions.append("Improve learned-token decoder breadth; learned-only heldout pass rate is below floor without prototype dependency.")
    if any(str(row.get("gate") or "").startswith("learned_pass_") for row in blockers):
        actions.append("Increase learned pass-path maturity: normalized AST diversity, AST-shape coverage, duplicate concentration, or loop/branch/local/call coverage is below the private structural floor.")
    if int(candidate_inventory.get("prototype_rows") or 0) > 0 and numeric(candidate_inventory.get("prototype_verifier_admissible_rate"), 0.0) < 0.95:
        actions.append("Repair verifier-admissible prototype coverage before using prototype rows as distillation targets.")
    if blockers:
        actions.append(f"First failing gate: {blockers[0].get('gate')}.")
    return actions or ["Refresh broad-private candidate manifests and rerun this gate."]


def run_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "green"}
    return bool(value)


def numeric(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Broad Private Learned Distillation Gate v1",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Completion: `{summary.get('completion_evidence_status')}`",
        f"- Full broad-private pass rate: `{summary.get('full_pass_rate')}`",
        f"- Learned-only pass rate: `{summary.get('learned_only_pass_rate')}`",
        f"- Strict novel learned-only pass rate: `{summary.get('strict_novel_learned_only_pass_rate')}`",
        f"- Prototype pass count: `{summary.get('prototype_pass_count')}`",
        f"- Exact train-body memory pass count: `{summary.get('exact_train_body_memory_pass_count')}`",
        f"- Learned-token pass count: `{summary.get('learned_token_pass_count')}`",
        f"- Prototype verifier-admissible rate: `{summary.get('candidate_inventory', {}).get('prototype_verifier_admissible_rate')}`",
        f"- Exact train-body memory candidate rate: `{summary.get('candidate_inventory', {}).get('exact_train_body_memory_candidate_rate')}`",
        f"- Learned pass normalized AST unique count: `{summary.get('learned_structural_inventory', {}).get('pass_normalized_ast_unique_count')}`",
        f"- Learned pass AST shape count: `{summary.get('learned_structural_inventory', {}).get('pass_ast_shape_count')}`",
        f"- Learned pass top normalized duplicate rate: `{summary.get('learned_structural_inventory', {}).get('pass_top_normalized_ast_duplicate_rate')}`",
        f"- Learned pass train AST overlap rate: `{summary.get('learned_train_novelty_inventory', {}).get('exact_train_normalized_ast_overlap_rate')}`",
        f"- Learned pass train body overlap rate: `{summary.get('learned_train_novelty_inventory', {}).get('exact_train_body_normalized_ast_overlap_rate')}`",
        f"- Decoder source/release fresh: `{summary.get('decoder_source_release_fresh')}`",
        "",
        "## Blockers",
    ]
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    if not blockers:
        lines.append("- None.")
    else:
        for row in blockers:
            lines.append(f"- `{row.get('gate')}` evidence `{row.get('evidence')}`")
    lines.append("")
    lines.append("## Next Actions")
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
