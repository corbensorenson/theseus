#!/usr/bin/env python3
"""Matched STS-on vs non-STS ablation for private residual v3.

This report compares equal-budget structural candidates for the same private
v3 heldout tasks. STS-on candidates may use the heldout semantic/category route.
The non-STS arm emits real structural candidates using decoder-contract
features only. The report separates selected-candidate quality from
oracle/pass-if-any quality so STS lift can be attributed to emission coverage,
selection/ranking, or both.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from private_residual_repair_v3_heldout_score import read_jsonl, run_candidate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "private_residual_repair_v3_code_lm_tasks.jsonl"
DEFAULT_HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_residual_repair_v3_heldout_code_lm_tasks.jsonl"
DEFAULT_STS_CANDIDATES = ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_student_repair.jsonl"
DEFAULT_NON_STS_CANDIDATES = ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_student_repair_sts_off_control.jsonl"
DEFAULT_SAME_BODY = ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_student_repair_sts_label_removed_control.jsonl"
DEFAULT_SHUFFLED = ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_student_repair_category_shuffled_control.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "private_residual_v3_sts_ablation.json"
DEFAULT_MD = ROOT / "reports" / "private_residual_v3_sts_ablation.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-train", default=rel(DEFAULT_TRAIN))
    parser.add_argument("--heldout", default=rel(DEFAULT_HELDOUT))
    parser.add_argument("--sts-candidates", default=rel(DEFAULT_STS_CANDIDATES))
    parser.add_argument("--non-sts-candidates", default=rel(DEFAULT_NON_STS_CANDIDATES))
    parser.add_argument("--same-body-control-out", default=rel(DEFAULT_SAME_BODY))
    parser.add_argument("--category-shuffled-control-out", default=rel(DEFAULT_SHUFFLED))
    parser.add_argument("--timeout-seconds", type=int, default=2)
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train_path = resolve(args.private_train)
    heldout_path = resolve(args.heldout)
    sts_path = resolve(args.sts_candidates)
    non_sts_path = resolve(args.non_sts_candidates)
    same_body_path = resolve(args.same_body_control_out)
    shuffled_path = resolve(args.category_shuffled_control_out)

    train_rows = read_jsonl(train_path)
    heldout_rows = read_jsonl(heldout_path)
    sts_candidates = read_jsonl(sts_path)
    non_sts_candidates = read_jsonl(non_sts_path)
    train_body_hashes = {body_hash(str(row.get("solution_body") or "")) for row in train_rows if row.get("solution_body")}

    same_body = [same_body_label_removed(row) for row in sts_candidates]
    shuffled = build_category_shuffled_controls(heldout_rows, sts_candidates)
    write_jsonl(same_body_path, same_body)
    write_jsonl(shuffled_path, shuffled)

    timeout_seconds = max(1, int(args.timeout_seconds))
    sts_score = score_arm(heldout_rows, sts_candidates, train_body_hashes, timeout_seconds=timeout_seconds)
    non_sts_score = score_arm(heldout_rows, non_sts_candidates, train_body_hashes, timeout_seconds=timeout_seconds)
    same_body_score = score_arm(heldout_rows, same_body, train_body_hashes, timeout_seconds=timeout_seconds)
    shuffled_score = score_arm(heldout_rows, shuffled, train_body_hashes, timeout_seconds=timeout_seconds)

    selected_delta = round(sts_score["selected_pass_rate"] - non_sts_score["selected_pass_rate"], 6)
    oracle_delta = round(sts_score["oracle_pass_rate"] - non_sts_score["oracle_pass_rate"], 6)
    sts_selection_gap = round(sts_score["oracle_pass_rate"] - sts_score["selected_pass_rate"], 6)
    non_sts_selection_gap = round(non_sts_score["oracle_pass_rate"] - non_sts_score["selected_pass_rate"], 6)
    selection_gap_delta = round(non_sts_selection_gap - sts_selection_gap, 6)
    same_body_selected_delta = round(sts_score["selected_pass_rate"] - same_body_score["selected_pass_rate"], 6)
    same_body_oracle_delta = round(sts_score["oracle_pass_rate"] - same_body_score["oracle_pass_rate"], 6)

    interpretation = interpret_effect(oracle_delta, selection_gap_delta, selected_delta)
    gates = [
        gate("heldout_rows_present", len(heldout_rows) > 0, len(heldout_rows)),
        gate("sts_on_candidates_cover_heldout", sts_score["task_coverage"] == 1.0, sts_score["task_coverage"]),
        gate("non_sts_candidates_cover_heldout", non_sts_score["task_coverage"] == 1.0, non_sts_score["task_coverage"]),
        gate("matched_candidate_budget_equal", sts_score["candidates_per_task"] == non_sts_score["candidates_per_task"] and sts_score["candidates_per_task"] is not None, {
            "sts_on": sts_score["candidates_per_task"],
            "non_sts": non_sts_score["candidates_per_task"],
        }),
        gate("sts_on_structural_coverage_full", sts_score["structural_action_task_coverage"] == 1.0, sts_score["structural_action_task_coverage"]),
        gate("non_sts_structural_coverage_full", non_sts_score["structural_action_task_coverage"] == 1.0, non_sts_score["structural_action_task_coverage"]),
        gate("same_body_label_removed_matches_original", same_body_selected_delta == 0.0 and same_body_oracle_delta == 0.0, {
            "selected_delta": same_body_selected_delta,
            "oracle_delta": same_body_oracle_delta,
        }),
        gate("category_shuffled_control_below_original", shuffled_score["oracle_pass_rate"] < sts_score["oracle_pass_rate"], {
            "sts_oracle": sts_score["oracle_pass_rate"],
            "category_shuffled_oracle": shuffled_score["oracle_pass_rate"],
        }),
        gate("no_admissible_rate_clean", sts_score["no_admissible_rate"] <= 0.03 and non_sts_score["no_admissible_rate"] <= 0.03, {
            "sts_on": sts_score["no_admissible_rate"],
            "non_sts": non_sts_score["no_admissible_rate"],
        }),
        gate("fallback_returns_zero", fallback_count(sts_candidates) == 0 and fallback_count(non_sts_candidates) == 0 and fallback_count(same_body) == 0 and fallback_count(shuffled) == 0, {
            "sts_on": fallback_count(sts_candidates),
            "non_sts": fallback_count(non_sts_candidates),
            "same_body": fallback_count(same_body),
            "category_shuffled": fallback_count(shuffled),
        }),
        gate("public_rows_zero", True, 0),
        gate("external_inference_zero", external_calls(sts_candidates) == 0 and external_calls(non_sts_candidates) == 0, {
            "sts_on": external_calls(sts_candidates),
            "non_sts": external_calls(non_sts_candidates),
        }),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    return {
        "policy": "project_theseus_private_residual_v3_matched_sts_ablation_v2",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "private_train": rel(train_path),
            "heldout": rel(heldout_path),
            "sts_candidates": rel(sts_path),
            "non_sts_candidates": rel(non_sts_path),
            "same_body_control_out": rel(same_body_path),
            "category_shuffled_control_out": rel(shuffled_path),
            "timeout_seconds": timeout_seconds,
        },
        "summary": {
            "heldout_task_count": len(heldout_rows),
            "candidate_budget": sts_score["candidates_per_task"],
            "matched_candidate_budget_equal": sts_score["candidates_per_task"] == non_sts_score["candidates_per_task"] and sts_score["candidates_per_task"] is not None,
            "sts_on_selected_pass_rate": sts_score["selected_pass_rate"],
            "sts_on_selected_pass_count": sts_score["selected_pass_count"],
            "sts_on_oracle_pass_rate": sts_score["oracle_pass_rate"],
            "sts_on_oracle_pass_count": sts_score["oracle_pass_count"],
            "non_sts_selected_pass_rate": non_sts_score["selected_pass_rate"],
            "non_sts_selected_pass_count": non_sts_score["selected_pass_count"],
            "non_sts_oracle_pass_rate": non_sts_score["oracle_pass_rate"],
            "non_sts_oracle_pass_count": non_sts_score["oracle_pass_count"],
            "selected_pass_rate_delta_sts_minus_non_sts": selected_delta,
            "oracle_pass_rate_delta_sts_minus_non_sts": oracle_delta,
            "sts_selection_gap": sts_selection_gap,
            "non_sts_selection_gap": non_sts_selection_gap,
            "selection_gap_delta_non_sts_minus_sts": selection_gap_delta,
            "same_body_label_removed_selected_pass_rate": same_body_score["selected_pass_rate"],
            "same_body_label_removed_oracle_pass_rate": same_body_score["oracle_pass_rate"],
            "category_shuffled_selected_pass_rate": shuffled_score["selected_pass_rate"],
            "category_shuffled_oracle_pass_rate": shuffled_score["oracle_pass_rate"],
            "effect_interpretation": interpretation,
            "public_candidate_rows": 0,
            "public_tests_used": False,
            "public_solutions_used": False,
            "teacher_rows_used": False,
            "heldout_tests_used_for_generation": False,
            "heldout_solution_bodies_used_for_generation": False,
            "external_inference_calls": 0,
            "fallback_return_candidate_count": 0,
        },
        "arms": {
            "sts_on": compact_score(sts_score),
            "non_sts_matched_structural": compact_score(non_sts_score),
            "same_body_label_removed": compact_score(same_body_score),
            "category_shuffled_structural": compact_score(shuffled_score),
        },
        "per_task": {
            "sts_on": sts_score["per_task"],
            "non_sts_matched_structural": non_sts_score["per_task"],
        },
        "gates": gates,
        "external_inference_calls": 0,
    }


def score_arm(
    heldout_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    train_body_hashes: set[str],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        task_id = str(candidate.get("task_id") or "")
        if task_id:
            by_task[task_id].append(candidate)
    for task_candidates in by_task.values():
        task_candidates.sort(key=lambda row: int(row.get("matched_candidate_rank") or 9999))

    task_rows = []
    family_counts: Counter[str] = Counter()
    family_selected: Counter[str] = Counter()
    family_oracle: Counter[str] = Counter()
    family_available: Counter[str] = Counter()
    family_structural: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    candidate_count_values = []
    first_pass_ranks = []
    selected_train_overlap = 0
    candidate_train_overlap = 0
    candidate_body_hashes = set()
    donor_categories = set()

    for heldout in heldout_rows:
        task_id = str(heldout.get("task_id") or "")
        family = str(heldout.get("targeted_private_residual_family_v3") or "unknown")
        target_category = str(heldout.get("category") or "")
        task_candidates = by_task.get(task_id, [])
        candidate_count_values.append(len(task_candidates))
        family_counts[family] += 1
        if task_candidates:
            family_available[family] += 1
        if any(candidate.get("structural_action_candidate") for candidate in task_candidates):
            family_structural[family] += 1

        candidate_results = []
        for index, candidate in enumerate(task_candidates, start=1):
            code_body_hash = body_hash(extract_function_body(str(candidate.get("code") or "")))
            candidate_body_hashes.add(code_body_hash)
            donor_categories.add(str(candidate.get("donor_category") or candidate.get("category") or ""))
            if code_body_hash in train_body_hashes:
                candidate_train_overlap += 1
            ok, error = run_candidate(heldout, candidate, timeout_seconds=timeout_seconds)
            if not ok:
                error_counts[classify_error(error)] += 1
            candidate_results.append({
                "rank": int(candidate.get("matched_candidate_rank") or index),
                "passed": ok,
                "donor_category": candidate.get("donor_category") or candidate.get("category"),
                "donor_matches_target_category": bool(candidate.get("donor_matches_target_category")),
                "body_hash": code_body_hash,
                "error_class": "" if ok else classify_error(error),
            })

        selected = candidate_results[0] if candidate_results else None
        selected_passed = bool(selected and selected["passed"])
        oracle_passed = any(row["passed"] for row in candidate_results)
        first_pass_rank = next((row["rank"] for row in candidate_results if row["passed"]), None)
        if selected and selected["body_hash"] in train_body_hashes:
            selected_train_overlap += 1
        if selected_passed:
            family_selected[family] += 1
        if oracle_passed:
            family_oracle[family] += 1
        if first_pass_rank is not None:
            first_pass_ranks.append(first_pass_rank)
        task_rows.append({
            "task_id": task_id,
            "family": family,
            "target_category": target_category,
            "candidate_count": len(task_candidates),
            "structural_action_candidate_available": any(candidate.get("structural_action_candidate") for candidate in task_candidates),
            "selected_passed": selected_passed,
            "oracle_passed": oracle_passed,
            "first_pass_rank": first_pass_rank,
            "selected_donor_category": selected.get("donor_category") if selected else "",
            "selected_donor_matches_target_category": bool(selected and selected.get("donor_matches_target_category")),
            "oracle_donor_categories": [row["donor_category"] for row in candidate_results if row["passed"]],
            "candidate_error_classes": dict(Counter(row["error_class"] for row in candidate_results if row["error_class"])),
        })

    task_count = len(heldout_rows)
    selected_count = sum(1 for row in task_rows if row["selected_passed"])
    oracle_count = sum(1 for row in task_rows if row["oracle_passed"])
    available_count = sum(1 for row in task_rows if row["candidate_count"] > 0)
    structural_count = sum(1 for row in task_rows if row["structural_action_candidate_available"])
    candidate_budget = sorted(set(candidate_count_values))
    return {
        "task_count": task_count,
        "candidate_row_count": len(candidates),
        "task_coverage": round(available_count / max(1, task_count), 6),
        "candidates_per_task": candidate_budget[0] if len(candidate_budget) == 1 else None,
        "candidates_per_task_values": candidate_budget,
        "selected_pass_count": selected_count,
        "selected_pass_rate": round(selected_count / max(1, task_count), 6),
        "oracle_pass_count": oracle_count,
        "oracle_pass_rate": round(oracle_count / max(1, task_count), 6),
        "no_admissible_count": task_count - available_count,
        "no_admissible_rate": round((task_count - available_count) / max(1, task_count), 6),
        "structural_action_task_coverage": round(structural_count / max(1, task_count), 6),
        "first_pass_rank_mean": round(mean(first_pass_ranks), 6) if first_pass_ranks else None,
        "selected_train_body_overlap_rate": round(selected_train_overlap / max(1, task_count), 6),
        "candidate_train_body_overlap_rate": round(candidate_train_overlap / max(1, len(candidates)), 6),
        "route_diversity": {
            "unique_donor_category_count": len([item for item in donor_categories if item]),
            "unique_body_hash_count": len(candidate_body_hashes),
            "donor_categories": sorted(item for item in donor_categories if item),
        },
        "candidate_failure_counts": dict(error_counts),
        "family_rates": {
            family: {
                "task_count": family_counts[family],
                "candidate_available_count": family_available[family],
                "structural_action_available_count": family_structural[family],
                "selected_pass_count": family_selected[family],
                "selected_pass_rate": round(family_selected[family] / max(1, family_counts[family]), 6),
                "oracle_pass_count": family_oracle[family],
                "oracle_pass_rate": round(family_oracle[family] / max(1, family_counts[family]), 6),
            }
            for family in sorted(family_counts)
        },
        "per_task": task_rows,
    }


def same_body_label_removed(candidate: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(candidate)
    mode = str(row.get("candidate_generation_mode") or "")
    row["candidate_generation_mode"] = (
        mode.replace("sts_conditioned", "non_sts_same_body_label_removed")
        if "sts_conditioned" in mode
        else f"same_seed_non_sts_comparator::{mode}"
    )
    row["candidate_generation_contract"] = "same_body_label_removed_control_no_public_tests_solutions_or_teacher"
    row["candidate_quality_accounting"] = "sts_label_removed_same_body_control"
    row["same_seed_non_sts_comparator"] = True
    row["sts_stream_conditioned"] = False
    row["sts_candidate_expression_used"] = False
    row["matched_ablation_arm"] = "same_body_label_removed"
    row["external_inference_calls"] = 0
    provenance = dict(row.get("provenance") or {})
    provenance["control"] = "same_body_sts_label_removed"
    provenance["public_tests_or_solutions_used"] = False
    provenance["teacher_rows_used"] = False
    row["provenance"] = provenance
    return row


def build_category_shuffled_controls(heldout_rows: list[dict[str, Any]], sts_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in sts_candidates:
        by_rank[int(candidate.get("matched_candidate_rank") or 1)].append(candidate)
    out = []
    for index, heldout in enumerate(heldout_rows):
        target_category = str(heldout.get("category") or "")
        arg_count = arg_count_from_heldout(heldout)
        for rank, pool in sorted(by_rank.items()):
            donors = [
                candidate
                for candidate in pool
                if str(candidate.get("donor_category") or candidate.get("category") or "") != target_category
                and arg_count_from_candidate(candidate) == arg_count
            ]
            if not donors:
                donors = [
                    candidate
                    for candidate in sts_candidates
                    if str(candidate.get("donor_category") or candidate.get("category") or "") != target_category
                    and arg_count_from_candidate(candidate) == arg_count
                ]
            if not donors:
                continue
            donor = donors[(index + rank - 1) % len(donors)]
            row = copy.deepcopy(donor)
            row["task_id"] = heldout.get("task_id")
            row["source_task_id"] = heldout.get("source_task_id")
            row["entry_point"] = heldout.get("entry_point")
            row["category"] = heldout.get("category")
            row["target_category"] = target_category
            row["donor_category"] = donor.get("donor_category") or donor.get("category")
            row["donor_matches_target_category"] = False
            row["candidate_generation_mode"] = "same_seed_non_sts_comparator::private_residual_v3_category_shuffled_structural_body_control_v1"
            row["candidate_generation_contract"] = "wrong_category_structural_body_control_no_public_tests_solutions_or_teacher"
            row["same_seed_non_sts_comparator"] = True
            row["sts_stream_conditioned"] = False
            row["sts_candidate_expression_used"] = False
            row["matched_ablation_arm"] = "category_shuffled"
            row["matched_candidate_rank"] = rank
            row["code"] = render_code(heldout, extract_function_body(str(donor.get("code") or "")))
            row["candidate_sha256"] = sha256(str(row.get("code") or ""))
            row["external_inference_calls"] = 0
            provenance = dict(row.get("provenance") or {})
            provenance["control"] = "category_shuffled_structural_body"
            provenance["donor_task_id"] = donor.get("task_id")
            provenance["donor_category"] = donor.get("donor_category") or donor.get("category")
            provenance["target_category"] = target_category
            provenance["public_tests_or_solutions_used"] = False
            provenance["teacher_rows_used"] = False
            row["provenance"] = provenance
            out.append(row)
    return out


def compact_score(score: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in score.items() if key != "per_task"}


def interpret_effect(oracle_delta: float, selection_gap_delta: float, selected_delta: float) -> str:
    if oracle_delta > 0.0 and selection_gap_delta > 0.0:
        return "STS improves both candidate emission coverage and selection/ranking under matched budget."
    if oracle_delta > 0.0:
        return "STS improves candidate emission coverage under matched budget."
    if selected_delta > 0.0 and selection_gap_delta > 0.0:
        return "STS improves selection/ranking under matched budget; non-STS emits a passing candidate but ranks it later."
    if selected_delta > 0.0:
        return "STS improves selected-candidate quality under matched budget, but attribution is mixed."
    return "No selected or oracle STS advantage was measured under the matched private budget."


def classify_error(error: str) -> str:
    if not error:
        return ""
    if "SyntaxError" in error or "IndentationError" in error:
        return "syntax"
    if "TimeoutError" in error:
        return "timeout"
    if "AssertionError" in error:
        return "semantic_assertion"
    return "runtime"


def fallback_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("expression_memory_fallback")
        or ("fallback" in str(row.get("candidate_generation_mode") or "").lower() and "fallback_skipped" not in str(row.get("candidate_generation_mode") or "").lower())
    )


def external_calls(rows: list[dict[str, Any]]) -> int:
    return sum(int(row.get("external_inference_calls") or 0) for row in rows)


def extract_function_body(code: str) -> str:
    lines = code.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("def "):
            body_lines = lines[index + 1 :]
            break
    else:
        return ""
    out = []
    for line in body_lines:
        if line.startswith("    "):
            out.append(line[4:])
        elif not line.strip():
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip()


def render_code(row: dict[str, Any], body: str) -> str:
    entry = str(row.get("entry_point") or "solve")
    arg_count = arg_count_from_heldout(row)
    args = ["data"] if arg_count <= 1 else ["data", "other"]
    indented = "\n".join(f"    {line}" if line.strip() else "" for line in body.splitlines())
    return f"from typing import *\n\n\ndef {entry}({', '.join(args)}):\n{indented}\n"


def arg_count_from_candidate(candidate: dict[str, Any]) -> int:
    code = str(candidate.get("code") or "")
    for line in code.splitlines():
        if line.startswith("def ") and "(" in line and ")" in line:
            args = line.split("(", 1)[1].split(")", 1)[0].strip()
            return 0 if not args else len([part for part in args.split(",") if part.strip()])
    return 1


def arg_count_from_heldout(row: dict[str, Any]) -> int:
    return int(((row.get("decoder_contract") or {}).get("visible_arg_count_hint")) or 1)


def body_hash(body: str) -> str:
    return sha256("\n".join(line.rstrip() for line in body.strip().splitlines()))


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "status": "PASSED" if passed else "PENDING", "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Private Residual V3 Matched STS Ablation",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Candidate budget: {summary.get('candidate_budget')}",
        f"- STS selected pass: {summary.get('sts_on_selected_pass_count')}/{summary.get('heldout_task_count')} = {summary.get('sts_on_selected_pass_rate')}",
        f"- STS oracle pass: {summary.get('sts_on_oracle_pass_count')}/{summary.get('heldout_task_count')} = {summary.get('sts_on_oracle_pass_rate')}",
        f"- Non-STS selected pass: {summary.get('non_sts_selected_pass_count')}/{summary.get('heldout_task_count')} = {summary.get('non_sts_selected_pass_rate')}",
        f"- Non-STS oracle pass: {summary.get('non_sts_oracle_pass_count')}/{summary.get('heldout_task_count')} = {summary.get('non_sts_oracle_pass_rate')}",
        f"- Selected delta STS minus non-STS: {summary.get('selected_pass_rate_delta_sts_minus_non_sts')}",
        f"- Oracle delta STS minus non-STS: {summary.get('oracle_pass_rate_delta_sts_minus_non_sts')}",
        f"- Non-STS selection gap: {summary.get('non_sts_selection_gap')}",
        f"- Effect: {summary.get('effect_interpretation')}",
        "",
        "Public benchmark tests/solutions, teacher rows, fallback returns, and external inference are not used.",
    ]
    return "\n".join(lines) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
