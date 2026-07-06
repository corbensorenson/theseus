#!/usr/bin/env python3
"""Leakage-checked private held-out transfer baseline.

This gate freezes an existing private train/eval pair and consumes an existing
replay report. It does not generate candidates, train, call a teacher, or run
public calibration. The output is one reproducible transfer number plus exact
prompt/code/template hash and ngram overlap evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus_archive_resolver import read_jsonl_follow_pointer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = (
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_train"
    / "post_v4_seed23_5x32_private_residual_repair_v3_code_lm_tasks.jsonl"
)
DEFAULT_HELDOUT = (
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_eval"
    / "post_v4_seed23_5x32_private_residual_repair_v3_heldout_code_lm_tasks.jsonl"
)
DEFAULT_REPLAY = ROOT / "reports" / "private_candidate_replay_contract_audit_reality_harness_token_learned_full_body.json"
DEFAULT_STRUCTURAL_REPLAY = ROOT / "reports" / "private_candidate_replay_contract_audit_reality_harness_structural_adapter.json"
DEFAULT_NGRAM_REPLAY = ROOT / "reports" / "private_candidate_replay_contract_audit_reality_harness_token_private_ngram.json"
DEFAULT_COMBINED_REPLAY = ROOT / "reports" / "private_candidate_replay_contract_audit_reality_harness_token_combined_promotion.json"
DEFAULT_OUT = ROOT / "reports" / "private_heldout_transfer_baseline_v1.json"
DEFAULT_MD = ROOT / "reports" / "private_heldout_transfer_baseline_v1.md"
DEFAULT_DERIVED_TRAIN = ROOT / "reports" / "private_heldout_transfer_baseline_v1_disjoint_train.jsonl"
DEFAULT_DERIVED_HELDOUT = ROOT / "reports" / "private_heldout_transfer_baseline_v1_disjoint_eval.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=rel(DEFAULT_TRAIN))
    parser.add_argument("--heldout", default=rel(DEFAULT_HELDOUT))
    parser.add_argument("--replay-report", default=rel(DEFAULT_REPLAY))
    parser.add_argument("--learned-replay-report", default=rel(DEFAULT_REPLAY))
    parser.add_argument("--structural-replay-report", default=rel(DEFAULT_STRUCTURAL_REPLAY))
    parser.add_argument("--ngram-replay-report", default=rel(DEFAULT_NGRAM_REPLAY))
    parser.add_argument("--combined-replay-report", default=rel(DEFAULT_COMBINED_REPLAY))
    parser.add_argument("--derive-disjoint-from", default="", help="Optional existing split JSONL to derive a strict disjoint train/eval pair from.")
    parser.add_argument("--derive-min-eval-rows", type=int, default=24)
    parser.add_argument("--derive-max-eval-rows", type=int, default=64, help="0 means keep every clean eval row.")
    parser.add_argument("--derived-train-out", default=rel(DEFAULT_DERIVED_TRAIN))
    parser.add_argument("--derived-heldout-out", default=rel(DEFAULT_DERIVED_HELDOUT))
    parser.add_argument("--ngram-size", type=int, default=8)
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train_path = resolve(args.train)
    heldout_path = resolve(args.heldout)
    replay_path = resolve(args.replay_report)
    learned_replay_path = resolve(args.learned_replay_report)
    structural_replay_path = resolve(args.structural_replay_report)
    ngram_replay_path = resolve(args.ngram_replay_report)
    combined_replay_path = resolve(args.combined_replay_report)
    derived_split = None
    if str(args.derive_disjoint_from or "").strip():
        derived_split = derive_disjoint_split(
            resolve(args.derive_disjoint_from),
            min_eval_rows=int(args.derive_min_eval_rows),
            max_eval_rows=int(args.derive_max_eval_rows),
        )
        train_rows = derived_split["train_rows"]
        heldout_rows = derived_split["heldout_rows"]
        train_path = resolve(args.derived_train_out)
        heldout_path = resolve(args.derived_heldout_out)
        write_jsonl(train_path, train_rows)
        write_jsonl(heldout_path, heldout_rows)
    else:
        train_rows = read_rows(train_path)
        heldout_rows = read_rows(heldout_path)
    replay = read_json(replay_path, {})
    learned_replay = read_json(learned_replay_path, {})
    structural_replay = read_json(structural_replay_path, {})
    ngram_replay = read_json(ngram_replay_path, {})
    combined_replay = read_json(combined_replay_path, {})
    ngram_size = max(3, int(args.ngram_size))

    overlap = overlap_audit(train_rows, heldout_rows, ngram_size=ngram_size)
    replay_summary = replay.get("summary") if isinstance(replay.get("summary"), dict) else {}
    task_count = int(replay_summary.get("task_count") or 0)
    functional_count = int(
        replay_summary.get("selected_functional_promotion_count")
        if replay_summary.get("selected_functional_promotion_count") is not None
        else replay_summary.get("selected_intended_behavior_pass_count") or 0
    )
    transfer_number = ratio(functional_count, task_count)
    transfer_ci = wilson_ci(functional_count, task_count)
    ablations = family_ablation_summary(
        {
            "learned_full_body_token_only": (learned_replay_path, learned_replay),
            "structural_adapter_only": (structural_replay_path, structural_replay),
            "private_ngram_body_only": (ngram_replay_path, ngram_replay),
            "combined_survival_lane": (combined_replay_path, combined_replay),
        }
    )
    public_training_rows = sum(1 for row in train_rows if truthy(row.get("public_benchmark")))
    public_heldout_rows = sum(1 for row in heldout_rows if truthy(row.get("public_benchmark")))
    forbidden_public_payload_rows = sum(
        1
        for row in train_rows + heldout_rows
        if truthy(row.get("public_tests_included")) or truthy(row.get("public_benchmark_solutions_included"))
    )
    split_overlap_clean = (
        overlap["exact_prompt_hash_overlap_count"] == 0
        and overlap["exact_code_hash_overlap_count"] == 0
        and overlap["exact_template_hash_overlap_count"] == 0
    )
    replay_inputs = replay.get("inputs") if isinstance(replay.get("inputs"), dict) else {}
    replay_heldout = str(replay_inputs.get("heldout") or "")
    replay_heldout_matches = paths_match(replay_heldout, heldout_path)
    heldout_claim_valid = split_overlap_clean and replay_heldout_matches
    score_semantics = (
        "valid private held-out transfer claim; numerator is selected functional promotion count from replay, "
        "denominator is replay task count"
        if heldout_claim_valid
        else invalid_score_semantics(split_overlap_clean=split_overlap_clean, replay_heldout_matches=replay_heldout_matches)
    )
    gates = [
        gate("train_rows_present", len(train_rows) > 0, {"path": rel(train_path), "rows": len(train_rows)}),
        gate("heldout_rows_present", len(heldout_rows) > 0, {"path": rel(heldout_path), "rows": len(heldout_rows)}),
        gate("replay_report_present", replay.get("policy") == "project_theseus_private_candidate_replay_contract_audit_v1", replay.get("policy")),
        gate("public_training_rows_zero", public_training_rows == 0, public_training_rows),
        gate("public_heldout_rows_zero", public_heldout_rows == 0, public_heldout_rows),
        gate("public_payload_flags_zero", forbidden_public_payload_rows == 0, forbidden_public_payload_rows),
        gate("exact_prompt_hash_overlap_zero", overlap["exact_prompt_hash_overlap_count"] == 0, overlap["exact_prompt_hash_overlap_count"]),
        gate("exact_code_hash_overlap_zero", overlap["exact_code_hash_overlap_count"] == 0, overlap["exact_code_hash_overlap_count"]),
        gate("exact_template_hash_overlap_zero", overlap["exact_template_hash_overlap_count"] == 0, overlap["exact_template_hash_overlap_count"]),
        gate(
            "replay_heldout_matches_baseline",
            replay_heldout_matches,
            {"replay_heldout": replay_heldout, "baseline_heldout": rel(heldout_path)},
        ),
        gate("ngram_overlap_audit_present", bool(overlap["ngram_size"] and overlap["heldout_row_count"]), overlap["max_prompt_ngram_jaccard"]),
        gate("single_transfer_number_present", task_count > 0, {"functional_count": functional_count, "task_count": task_count}),
        gate("family_ablation_reports_present", ablations["all_reports_present"], ablations["report_paths"]),
        gate("external_inference_zero", int(replay.get("external_inference_calls") or 0) == 0, replay.get("external_inference_calls")),
    ]
    failed = [row for row in gates if not row["passed"]]
    trigger_state = "GREEN" if not failed and functional_count > 0 else "YELLOW" if train_rows and heldout_rows else "RED"
    return {
        "policy": "project_theseus_private_heldout_transfer_baseline_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "train": rel(train_path),
            "heldout": rel(heldout_path),
            "derive_disjoint_from": rel(resolve(args.derive_disjoint_from)) if str(args.derive_disjoint_from or "").strip() else "",
            "replay_report": rel(replay_path),
            "ngram_size": ngram_size,
        },
        "frozen_manifest": {
            "train_sha256": sha256_file(train_path),
            "heldout_sha256": sha256_file(heldout_path),
            "replay_report_sha256": sha256_file(replay_path) if replay_path.exists() else "",
            "train_row_count": len(train_rows),
            "heldout_row_count": len(heldout_rows),
            "row_id_hash": sha256_text(
                "\n".join(
                    sorted(str(row.get("task_id") or row.get("source_task_id") or "") for row in train_rows + heldout_rows)
                )
            ),
        },
        "summary": {
            "train_row_count": len(train_rows),
            "heldout_row_count": len(heldout_rows),
            "replay_task_count": task_count,
            "functional_promotion_count": functional_count,
            "private_heldout_transfer_pass_rate": transfer_number,
            "private_heldout_transfer_fraction": fraction(functional_count, task_count),
            "private_heldout_transfer_ci95": transfer_ci,
            "heldout_claim_valid": heldout_claim_valid,
            "split_overlap_clean": split_overlap_clean,
            "replay_heldout_matches_baseline": replay_heldout_matches,
            "replay_heldout": replay_heldout,
            "derived_split": derived_split_summary(derived_split),
            "score_semantics": score_semantics,
            "recommendation": baseline_recommendation(
                heldout_claim_valid,
                functional_count,
                task_count,
                ablations,
                split_overlap_clean=split_overlap_clean,
                replay_heldout_matches=replay_heldout_matches,
            ),
            "family_ablation": ablations,
            "public_training_rows": public_training_rows,
            "public_heldout_rows": public_heldout_rows,
            "forbidden_public_payload_rows": forbidden_public_payload_rows,
            **overlap,
        },
        "gates": gates,
        "public_calibration_run": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
    }


def family_ablation_summary(reports: dict[str, tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, (path, report) in reports.items():
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        task_count = int(summary.get("task_count") or 0)
        pass_count = int(
            summary.get("selected_functional_promotion_count")
            if summary.get("selected_functional_promotion_count") is not None
            else summary.get("selected_intended_behavior_pass_count") or 0
        )
        pass_if_any_count = int(summary.get("pass_if_any_count") or 0)
        rows[name] = {
            "path": rel(path),
            "present": report.get("policy") == "project_theseus_private_candidate_replay_contract_audit_v1",
            "trigger_state": report.get("trigger_state"),
            "task_count": task_count,
            "selected_functional_promotion_count": pass_count,
            "selected_pass_fraction": fraction(pass_count, task_count),
            "selected_pass_rate": ratio(pass_count, task_count),
            "selected_pass_rate_ci95": wilson_ci(pass_count, task_count),
            "pass_if_any_count": pass_if_any_count,
            "pass_if_any_fraction": fraction(pass_if_any_count, task_count),
            "pass_if_any_rate": ratio(pass_if_any_count, task_count),
            "candidate_family_counts": summary.get("candidate_family_counts", {}),
            "functional_promotion_by_family": summary.get("functional_promotion_by_family", {}),
            "integrity_mismatch_count": summary.get("candidate_integrity_mismatch_count", 0),
        }
    learned = rows["learned_full_body_token_only"]
    structural = rows["structural_adapter_only"]
    ngram = rows["private_ngram_body_only"]
    combined = rows["combined_survival_lane"]
    learned_count = int(learned.get("selected_functional_promotion_count") or 0)
    combined_count = int(combined.get("selected_functional_promotion_count") or 0)
    structural_count = int(structural.get("selected_functional_promotion_count") or 0)
    ngram_count = int(ngram.get("selected_functional_promotion_count") or 0)
    lift_count = combined_count - learned_count
    if lift_count > 0:
        conclusion = "structural_or_ngram_families_lift_selected_functional_pass"
    elif structural_count == 0 and ngram_count == 0:
        conclusion = "structural_and_ngram_are_count_inflation_under_current_replay"
    else:
        conclusion = "no_combined_lift_over_learned_under_current_replay"
    return {
        "report_paths": {name: row["path"] for name, row in rows.items()},
        "all_reports_present": all(bool(row.get("present")) for row in rows.values()),
        "rows": rows,
        "combined_minus_learned_selected_functional_count": lift_count,
        "combined_minus_learned_selected_functional_rate": round(
            float(combined.get("selected_pass_rate") or 0.0) - float(learned.get("selected_pass_rate") or 0.0),
            6,
        ),
        "conclusion": conclusion,
        "score_semantics": "family ablations compare selected functional promotion counts; they do not train, tune thresholds, or run public calibration",
    }


def baseline_recommendation(
    heldout_claim_valid: bool,
    functional_count: int,
    task_count: int,
    ablations: dict[str, Any],
    *,
    split_overlap_clean: bool,
    replay_heldout_matches: bool,
) -> str:
    if not heldout_claim_valid:
        if split_overlap_clean and not replay_heldout_matches:
            return (
                "The private split is exact-overlap clean, but the replay report was run on a different heldout manifest. "
                "Generate candidates and rerun learned_full_body_token replay on this derived heldout before citing transfer."
            )
        return (
            "Do not cite this as held-out transfer. Freeze or derive a private eval slice with zero exact prompt, "
            "code, and template overlap before using the replay number for promotion."
        )
    if task_count <= 0:
        return "No replay denominator is available; rerun the matched private replay before making a transfer claim."
    if functional_count <= 0:
        return "Heldout split is clean, but functional transfer is zero; repair semantic candidate quality before calibration."
    if ablations.get("conclusion") == "structural_and_ngram_are_count_inflation_under_current_replay":
        return (
            "Functional transfer is nonzero, but structural/ngram families do not lift selected pass; improve learned "
            "full-body semantics before broad calibration."
        )
    return "Heldout transfer is nonzero under clean overlap gates; compare against fresh matched controls before promotion."


def invalid_score_semantics(*, split_overlap_clean: bool, replay_heldout_matches: bool) -> str:
    reasons = []
    if not split_overlap_clean:
        reasons.append("train/eval exact-overlap gates failed")
    if not replay_heldout_matches:
        reasons.append("replay report was not run on the audited heldout manifest")
    joined = "; ".join(reasons) if reasons else "claim gate failed"
    return f"invalid held-out transfer claim because {joined}; replay score is diagnostic only"


def derive_disjoint_split(source_path: Path, *, min_eval_rows: int, max_eval_rows: int) -> dict[str, Any]:
    source_rows = read_rows(source_path)
    train_rows = [row for row in source_rows if str(row.get("split") or "") == "train"]
    eval_rows = [row for row in source_rows if str(row.get("split") or "") == "eval"]
    train_prompt_hashes = {sha256_text(prompt_text(row)) for row in train_rows if prompt_text(row)}
    train_code_hashes = {sha256_text(code_text(row)) for row in train_rows if code_text(row)}
    train_template_hashes = {sha256_text(template_text(row)) for row in train_rows if template_text(row)}
    clean_eval = [
        row
        for row in eval_rows
        if sha256_text(prompt_text(row)) not in train_prompt_hashes
        and sha256_text(code_text(row)) not in train_code_hashes
        and sha256_text(template_text(row)) not in train_template_hashes
    ]
    selected = diverse_eval_selection(clean_eval, max_eval_rows=max_eval_rows)
    if len(selected) < int(min_eval_rows):
        selected = []
    return {
        "source": rel(source_path),
        "source_row_count": len(source_rows),
        "source_train_row_count": len(train_rows),
        "source_eval_row_count": len(eval_rows),
        "clean_eval_candidate_count": len(clean_eval),
        "min_eval_rows": int(min_eval_rows),
        "max_eval_rows": int(max_eval_rows),
        "train_rows": train_rows,
        "heldout_rows": selected,
        "selected_eval_row_count": len(selected),
        "selected_category_count": len({str(row.get("category") or "") for row in selected}),
    }


def diverse_eval_selection(rows: list[dict[str, Any]], *, max_eval_rows: int) -> list[dict[str, Any]]:
    if max_eval_rows <= 0:
        return list(rows)
    selected = []
    used_categories: set[str] = set()
    for row in rows:
        category = str(row.get("category") or "")
        if category in used_categories:
            continue
        selected.append(row)
        used_categories.add(category)
        if len(selected) >= max_eval_rows:
            return selected
    for row in rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= max_eval_rows:
            break
    return selected


def derived_split_summary(derived_split: dict[str, Any] | None) -> dict[str, Any]:
    if not derived_split:
        return {"derived": False}
    return {
        "derived": True,
        "source": derived_split.get("source", ""),
        "source_row_count": derived_split.get("source_row_count", 0),
        "source_train_row_count": derived_split.get("source_train_row_count", 0),
        "source_eval_row_count": derived_split.get("source_eval_row_count", 0),
        "clean_eval_candidate_count": derived_split.get("clean_eval_candidate_count", 0),
        "selected_eval_row_count": derived_split.get("selected_eval_row_count", 0),
        "selected_category_count": derived_split.get("selected_category_count", 0),
        "min_eval_rows": derived_split.get("min_eval_rows", 0),
        "max_eval_rows": derived_split.get("max_eval_rows", 0),
    }


def overlap_audit(train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]], *, ngram_size: int) -> dict[str, Any]:
    train_prompt_hashes = hash_counter(prompt_text(row) for row in train_rows)
    heldout_prompt_hashes = hash_counter(prompt_text(row) for row in heldout_rows)
    train_code_hashes = hash_counter(code_text(row) for row in train_rows)
    heldout_code_hashes = hash_counter(code_text(row) for row in heldout_rows)
    train_template_hashes = hash_counter(template_text(row) for row in train_rows)
    heldout_template_hashes = hash_counter(template_text(row) for row in heldout_rows)
    train_prompt_ngrams = ngram_set("\n".join(prompt_text(row) for row in train_rows), ngram_size)
    train_code_ngrams = ngram_set("\n".join(code_text(row) for row in train_rows), ngram_size)
    train_template_ngrams = ngram_set("\n".join(template_text(row) for row in train_rows), ngram_size)
    prompt_jaccards = []
    code_jaccards = []
    template_jaccards = []
    prompt_ngram_hits = 0
    code_ngram_hits = 0
    template_ngram_hits = 0
    for row in heldout_rows:
        prompt_ngrams = ngram_set(prompt_text(row), ngram_size)
        code_ngrams = ngram_set(code_text(row), ngram_size)
        template_ngrams = ngram_set(template_text(row), ngram_size)
        prompt_jaccards.append(jaccard(prompt_ngrams, train_prompt_ngrams))
        code_jaccards.append(jaccard(code_ngrams, train_code_ngrams))
        template_jaccards.append(jaccard(template_ngrams, train_template_ngrams))
        prompt_ngram_hits += len(prompt_ngrams & train_prompt_ngrams)
        code_ngram_hits += len(code_ngrams & train_code_ngrams)
        template_ngram_hits += len(template_ngrams & train_template_ngrams)
    return {
        "heldout_row_count": len(heldout_rows),
        "ngram_size": ngram_size,
        "exact_prompt_hash_overlap_count": overlap_count(train_prompt_hashes, heldout_prompt_hashes),
        "exact_code_hash_overlap_count": overlap_count(train_code_hashes, heldout_code_hashes),
        "exact_template_hash_overlap_count": overlap_count(train_template_hashes, heldout_template_hashes),
        "heldout_prompt_hash_count": len(heldout_prompt_hashes),
        "heldout_code_hash_count": len(heldout_code_hashes),
        "heldout_template_hash_count": len(heldout_template_hashes),
        "prompt_ngram_hit_count": prompt_ngram_hits,
        "code_ngram_hit_count": code_ngram_hits,
        "template_ngram_hit_count": template_ngram_hits,
        "max_prompt_ngram_jaccard": round(max(prompt_jaccards or [0.0]), 6),
        "max_code_ngram_jaccard": round(max(code_jaccards or [0.0]), 6),
        "max_template_ngram_jaccard": round(max(template_jaccards or [0.0]), 6),
        "mean_prompt_ngram_jaccard": round(sum(prompt_jaccards) / len(prompt_jaccards), 6) if prompt_jaccards else 0.0,
        "mean_code_ngram_jaccard": round(sum(code_jaccards) / len(code_jaccards), 6) if code_jaccards else 0.0,
        "mean_template_ngram_jaccard": round(sum(template_jaccards) / len(template_jaccards), 6) if template_jaccards else 0.0,
    }


def prompt_text(row: dict[str, Any]) -> str:
    return normalize_text(row.get("prompt"))


def code_text(row: dict[str, Any]) -> str:
    return normalize_text("\n".join([str(row.get("solution_body") or ""), str(row.get("solution_expr") or "")]))


def template_text(row: dict[str, Any]) -> str:
    decoder = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    return normalize_text(
        json.dumps(
            {
                "category": row.get("category"),
                "tags": row.get("tags"),
                "entry_point": row.get("entry_point"),
                "targeted_private_residual_family_v3": row.get("targeted_private_residual_family_v3"),
                "decoder_policy": decoder.get("policy"),
                "required_constructs": decoder.get("required_constructs"),
            },
            sort_keys=True,
        )
    )


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def hash_counter(values: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    for value in values:
        if str(value).strip():
            counter[sha256_text(str(value))] += 1
    return counter


def overlap_count(left: Counter[str], right: Counter[str]) -> int:
    return sum(min(left[key], right[key]) for key in set(left) & set(right))


def ngram_set(text: str, size: int) -> set[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", normalize_text(text))
    if len(tokens) < size:
        return set()
    return {" ".join(tokens[index : index + size]) for index in range(0, len(tokens) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]


def read_json(path: Path, default: Any = None) -> Any:
    default = {} if default is None else default
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    failed = [row["gate"] for row in report.get("gates", []) if not row.get("passed")]
    return "\n".join(
        [
            "# Private Held-Out Transfer Baseline v1",
            "",
            f"- State: `{report.get('trigger_state')}`",
            f"- Transfer: `{summary.get('private_heldout_transfer_fraction')}` rate=`{summary.get('private_heldout_transfer_pass_rate')}` ci95=`{summary.get('private_heldout_transfer_ci95')}`",
            f"- Heldout claim valid: `{summary.get('heldout_claim_valid')}`",
            f"- Score semantics: `{summary.get('score_semantics')}`",
            f"- Recommendation: `{summary.get('recommendation')}`",
            f"- Train / heldout rows: `{summary.get('train_row_count')}` / `{summary.get('heldout_row_count')}`",
            f"- Exact prompt/code/template overlaps: `{summary.get('exact_prompt_hash_overlap_count')}` / `{summary.get('exact_code_hash_overlap_count')}` / `{summary.get('exact_template_hash_overlap_count')}`",
            f"- Max prompt/code/template ngram Jaccard: `{summary.get('max_prompt_ngram_jaccard')}` / `{summary.get('max_code_ngram_jaccard')}` / `{summary.get('max_template_ngram_jaccard')}`",
            f"- Public rows/payload flags: `{summary.get('public_training_rows')}` / `{summary.get('public_heldout_rows')}` / `{summary.get('forbidden_public_payload_rows')}`",
            f"- Failed gates: `{failed}`",
            "",
        ]
    )


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def fraction(num: int, den: int) -> str:
    return f"{int(num)}/{int(den)}"


def wilson_ci(num: int, den: int, z: float = 1.959963984540054) -> dict[str, Any]:
    num = int(num)
    den = int(den)
    if den <= 0:
        return {"count": num, "denominator": den, "low": 0.0, "high": 0.0}
    p = num / den
    z2 = z * z
    denom = 1.0 + z2 / den
    center = (p + z2 / (2.0 * den)) / denom
    margin = z * ((p * (1.0 - p) / den + z2 / (4.0 * den * den)) ** 0.5) / denom
    return {
        "count": num,
        "denominator": den,
        "low": round(max(0.0, center - margin), 6),
        "high": round(min(1.0, center + margin), 6),
    }


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def paths_match(left: str | Path, right: str | Path) -> bool:
    if not str(left or "").strip():
        return False
    return rel(resolve(left)) == rel(resolve(right))


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
