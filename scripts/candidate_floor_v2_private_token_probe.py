#!/usr/bin/env python3
"""Probe Candidate Floor v2 with learned full-body token candidates on private tasks.

This is the private analogue for the public-transfer wall. It sends only visible
private task metadata to the Rust token generator, never tests or solutions, and
then verifies the emitted full-body candidates against private heldout tests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402


DEFAULT_EVAL = [
    "data/training_data/high_transfer/private_eval/public_safe_broad_transfer_maturity_v4_heldout_code_lm_tasks.jsonl",
    "data/training_data/high_transfer/private_eval/private_residual_repair_v3_heldout_code_lm_tasks.jsonl",
    "data/training_data/high_transfer/private_eval/edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum_heldout_code_lm_tasks.jsonl",
    "data/training_data/high_transfer/private_eval/private_contract_blind_transfer_v1_code_lm_tasks.jsonl",
    "data/training_data/high_transfer/private_eval/broad_private_generalization_ladder_v1_heldout_code_lm_tasks.jsonl",
    "data/training_data/high_transfer/private_eval/broad_private_novel_composition_v1_heldout_code_lm_tasks.jsonl",
    "data/training_data/high_transfer/private_eval/dependency_free_candidate_private_curriculum_heldout_code_lm_tasks.jsonl",
]
DEFAULT_TRAINING_SOURCES = "data/training_sources/broad_capability_curriculum_v1_training_sources.json"
QUEUE_OPEN_STATES = {"needs_private_ablation", "open", "yellow", "unresolved"}
QUEUE_CATEGORY_ALIASES = {
    "algorithmic_planning": "algorithm_choice",
    "algorithm_choice": "algorithm_choice",
    "semantic_wrong_answer": "algorithm_choice",
    "wrong_algorithm": "algorithm_choice",
    "type_handling": "return_type_shape",
    "return_shape": "return_type_shape",
    "return_type_shape": "return_type_shape",
    "shape_mismatch": "return_type_shape",
    "edge_case": "edge_cases",
    "edge_cases": "edge_cases",
    "boundary_case": "edge_cases",
    "external_dependency_missing": "external_dependency_missing",
    "dependency_missing": "external_dependency_missing",
    "dependency_free": "external_dependency_missing",
}
QUEUE_CATEGORY_TERMS = {
    "algorithm_choice": {
        "algorithm_choice",
        "algorithmic_planning_contracts",
        "semantic_ranker_selection",
        "semantic_wrong_answer",
        "wrong_algorithm",
        "multi_step_control_flow",
    },
    "return_type_shape": {
        "return_type_shape",
        "return_interface",
        "return_interface_fidelity",
        "return_interface_fidelity_v3",
        "type_handling",
        "shape_str",
        "type_conversion",
    },
    "edge_cases": {
        "edge_case",
        "edge_cases",
        "duplicate_handling",
    },
    "external_dependency_missing": {
        "external_dependency_missing",
        "dependency_free",
        "standard_library",
        "standard-library",
        "stdlib",
        "dependency",
        "import_avoidance",
        "external_package",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", action="append", default=[])
    parser.add_argument("--training-sources", default=DEFAULT_TRAINING_SOURCES)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-eval-rows", type=int, default=192)
    parser.add_argument("--max-candidates-per-task", type=int, default=8)
    parser.add_argument("--max-training-rows-per-source", type=int, default=1400)
    parser.add_argument("--max-project-files", type=int, default=160)
    parser.add_argument(
        "--residual-queue",
        default="",
        help=(
            "Optional private repair queue generated from public residual categories. "
            "Only category labels are consumed; public prompts/tests/solutions remain excluded."
        ),
    )
    parser.add_argument(
        "--visible-contract-mode",
        choices=["full", "minimal"],
        default="full",
        help=(
            "full preserves safe private decoder-contract context; minimal strips "
            "role/return/plan/composition fields for VCM-off generation ablations"
        ),
    )
    parser.add_argument("--task-manifest-out", default="reports/candidate_floor_v2_private_token_probe_tasks.jsonl")
    parser.add_argument("--candidate-manifest-out", default="reports/candidate_floor_v2_private_token_probe_candidates.jsonl")
    parser.add_argument("--checkpoint-out", default="reports/candidate_floor_v2_private_token_probe_checkpoint.json")
    parser.add_argument("--rust-report-out", default="reports/candidate_floor_v2_private_token_probe_rust.json")
    parser.add_argument("--out", default="reports/candidate_floor_v2_private_token_probe.json")
    parser.add_argument("--markdown-out", default="reports/candidate_floor_v2_private_token_probe.md")
    args = parser.parse_args()

    started = time.perf_counter()
    report = run_probe(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def run_probe(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    eval_paths = [resolve(path) for path in (args.eval_jsonl or DEFAULT_EVAL)]
    residual_queue_path = resolve(args.residual_queue) if args.residual_queue else None
    residual_queue_rows = load_residual_queue(residual_queue_path) if residual_queue_path else []
    residual_queue_profile = build_residual_queue_profile(residual_queue_rows)
    all_private_rows = load_private_eval_rows(eval_paths, max_rows=0, seed=int(args.seed))
    if residual_queue_profile["open_categories"]:
        private_rows, row_matches = filter_private_eval_rows_for_queue(
            all_private_rows,
            residual_queue_profile["open_categories"],
        )
    else:
        private_rows = all_private_rows
        row_matches = {}
    private_rows = select_private_rows_for_eval_limit(
        private_rows,
        row_matches=row_matches,
        max_rows=max(1, int(args.max_eval_rows)),
        open_categories=residual_queue_profile["open_categories"],
    )
    row_matches = {str(row.get("task_id") or ""): row_matches.get(str(row.get("task_id") or ""), []) for row in private_rows}
    residual_queue_eval = summarize_residual_queue_eval(
        private_rows,
        row_matches=row_matches,
        residual_queue_profile=residual_queue_profile,
    )
    task_manifest = visible_task_manifest(private_rows, visible_contract_mode=str(args.visible_contract_mode))
    write_jsonl(resolve(args.task_manifest_out), task_manifest)

    removed = remove_stale(
        [
            resolve(args.candidate_manifest_out),
            resolve(args.checkpoint_out),
            resolve(args.rust_report_out),
        ]
    )
    command = rust_command(args)
    timeout_seconds = max(240, min(1800, 60 + len(task_manifest) * max(1, int(args.max_candidates_per_task))))
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout_seconds)
        returncode = result.returncode
        stdout_tail = result.stdout[-2000:]
        stderr_tail = result.stderr[-2000:]
        error = ""
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout_tail = (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else ""
        error = f"rust token generator timed out after {timeout_seconds}s"
    except OSError as exc:
        returncode = 127
        stdout_tail = ""
        stderr_tail = ""
        error = str(exc)

    candidates_raw = read_jsonl(resolve(args.candidate_manifest_out))
    candidates = normalize_candidate_phases(candidates_raw)
    if candidates != candidates_raw:
        write_jsonl(resolve(args.candidate_manifest_out), candidates)
    rust_report = read_json(resolve(args.rust_report_out), {})
    private_eval = evaluate_private_candidates(private_rows, candidates) if private_rows and candidates else {}
    summary = summarize(
        private_rows,
        candidates,
        rust_report,
        private_eval,
        residual_queue_eval=residual_queue_eval,
    )
    gates = [
        gate("private_eval_rows_loaded", bool(private_rows), {"rows": len(private_rows), "sources": [rel(path) for path in eval_paths]}, "hard"),
        gate("visible_task_manifest_omits_tests", all("tests" not in row for row in task_manifest), len(task_manifest), "hard"),
        gate("visible_task_manifest_omits_solution_body", all("solution_body" not in row for row in task_manifest), len(task_manifest), "hard"),
        gate("rust_token_generator_completed", returncode == 0, {"returncode": returncode, "error": error}, "hard"),
        gate("full_body_candidates_for_every_task", summary["candidate_coverage_rate"] >= 1.0, summary["candidate_coverage_rate"], "hard"),
        gate("no_admissible_private_task_count_zero", summary["no_candidate_task_count"] == 0, summary["no_candidate_task_count"], "hard"),
        gate("learned_full_body_candidates_emitted", summary["full_body_token_candidate_count"] > 0, summary["full_body_token_candidate_count"], "hard"),
        gate("grammar_masked_candidates_emitted", summary["grammar_masked_learned_token_candidate_count"] > 0, summary["grammar_masked_learned_token_candidate_count"], "hard"),
        gate("template_like_count_zero", summary["template_like_candidate_count"] == 0, summary["template_like_candidate_count"], "hard"),
        gate("fallback_count_zero", summary["expression_memory_fallback_count"] == 0, summary["expression_memory_fallback_count"], "hard"),
        gate("runtime_external_inference_zero", summary["external_inference_calls"] == 0, summary["external_inference_calls"], "hard"),
        gate("semantic_private_pass_nonzero", summary["private_trained_pass_rate"] > 0.0, summary["private_trained_pass_rate"], "warning"),
    ]
    if residual_queue_path:
        gates.extend(
            [
                gate(
                    "residual_queue_loaded",
                    bool(residual_queue_rows),
                    {"rows": len(residual_queue_rows), "path": rel(residual_queue_path)},
                    "hard",
                ),
                gate(
                    "residual_queue_open_categories_detected",
                    bool(residual_queue_profile["open_categories"]),
                    residual_queue_profile["open_categories"],
                    "warning",
                ),
                gate(
                    "residual_queue_private_eval_rows_loaded",
                    residual_queue_eval["targeted_private_eval_task_count"] > 0,
                    residual_queue_eval["targeted_private_eval_task_count"],
                    "hard",
                ),
                gate(
                    "residual_queue_categories_have_private_eval",
                    not residual_queue_eval["missing_open_categories"],
                    residual_queue_eval["missing_open_categories"],
                    "warning",
                ),
            ]
        )
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failed = [row for row in gates if row["severity"] == "warning" and not row["passed"]]
    trigger_state = "RED" if hard_failed else "YELLOW" if warning_failed else "GREEN"
    return {
        "policy": "project_theseus_candidate_floor_v2_private_token_probe_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "inputs": {
            "eval_jsonl": [rel(path) for path in eval_paths],
            "training_sources": rel(resolve(args.training_sources)),
            "residual_queue": rel(residual_queue_path) if residual_queue_path else "",
        },
        "artifacts": {
            "task_manifest": rel(resolve(args.task_manifest_out)),
            "candidate_manifest": rel(resolve(args.candidate_manifest_out)),
            "checkpoint": rel(resolve(args.checkpoint_out)),
            "rust_report": rel(resolve(args.rust_report_out)),
        },
        "command": command,
        "returncode": returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "generation_error": error,
        "removed_stale_artifacts": removed,
        "private_verifier": compact_private_eval(private_eval),
        "gates": gates,
        "rules": {
            "public_calibration_run": False,
            "public_payload_training": False,
            "public_residual_content_consumed": False,
            "public_residual_category_labels_consumed": bool(residual_queue_path),
            "tests_visible_to_generator": False,
            "solutions_visible_to_generator": False,
            "candidate_floor_semantics": "Every private eval row must receive genuine learned full-body candidates before semantic scores are considered.",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def load_private_eval_rows(paths: list[Path], *, max_rows: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            if not row_is_private_eval(row):
                continue
            item = dict(row)
            item["split"] = "eval"
            rows.append(item)
    rows = dedupe_rows(rows)
    rows.sort(key=lambda row: stable_key({"seed": seed, "task_id": row.get("task_id"), "entry_point": row.get("entry_point")}))
    return rows[:max_rows] if max_rows > 0 else rows


def load_residual_queue(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = []
    for row in read_jsonl(path):
        if row.get("public_prompt_embedded") is True:
            continue
        if row.get("public_tests_embedded") is True:
            continue
        if row.get("public_solution_embedded") is True:
            continue
        if row.get("candidate_code_embedded") is True:
            continue
        rows.append(row)
    return rows


def build_residual_queue_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    open_rows = [row for row in rows if row_requests_fresh_private_probe(row)]
    open_categories = sorted(
        {
            canonical_queue_category(str(row.get("canonical_category") or ""))
            for row in open_rows
            if canonical_queue_category(str(row.get("canonical_category") or ""))
        }
    )
    return {
        "row_count": len(rows),
        "open_row_count": len(open_rows),
        "open_categories": open_categories,
        "open_category_counts": dict(
            sorted(Counter(canonical_queue_category(str(row.get("canonical_category") or "")) for row in open_rows).items())
        ),
    }


def row_requests_fresh_private_probe(row: dict[str, Any]) -> bool:
    if str(row.get("state") or "").lower() in QUEUE_OPEN_STATES:
        return True
    evidence = row.get("current_private_evidence")
    if isinstance(evidence, dict):
        try:
            return int(evidence.get("fresh_public_residual_count") or 0) > 0
        except (TypeError, ValueError):
            return False
    return False


def canonical_queue_category(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return QUEUE_CATEGORY_ALIASES.get(key, key)


def filter_private_eval_rows_for_queue(
    rows: list[dict[str, Any]],
    open_categories: list[str],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    out: list[dict[str, Any]] = []
    matches_by_task: dict[str, list[str]] = {}
    for row in rows:
        matches = matched_queue_categories_for_row(row, open_categories)
        if not matches:
            continue
        task_id = str(row.get("task_id") or "")
        matches_by_task[task_id] = matches
        item = dict(row)
        item["fresh_public_residual_private_ablation_categories"] = matches
        out.append(item)
    return out, matches_by_task


def select_private_rows_for_eval_limit(
    rows: list[dict[str, Any]],
    *,
    row_matches: dict[str, list[str]],
    max_rows: int,
    open_categories: list[str],
) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        return rows
    if not open_categories:
        return rows[:max_rows]
    indexed = list(enumerate(rows))
    by_category: dict[str, list[int]] = {category: [] for category in open_categories}
    for idx, row in indexed:
        task_id = str(row.get("task_id") or "")
        matches = set(row_matches.get(task_id, []))
        for category in open_categories:
            if category in matches:
                by_category[category].append(idx)
    category_order = sorted(open_categories, key=lambda category: (len(by_category.get(category, [])), category))
    selected_order: list[int] = []
    selected: set[int] = set()
    positions = {category: 0 for category in category_order}
    while len(selected) < max_rows:
        progressed = False
        for category in category_order:
            indices = by_category.get(category, [])
            pos = positions[category]
            while pos < len(indices) and indices[pos] in selected:
                pos += 1
            positions[category] = pos
            if pos >= len(indices):
                continue
            idx = indices[pos]
            selected.add(idx)
            selected_order.append(idx)
            positions[category] = pos + 1
            progressed = True
            if len(selected) >= max_rows:
                break
        if not progressed:
            break
    if len(selected) < max_rows:
        for idx, _row in indexed:
            if idx in selected:
                continue
            selected.add(idx)
            selected_order.append(idx)
            if len(selected) >= max_rows:
                break
    return [row for idx, row in enumerate(rows) if idx in selected]


def matched_queue_categories_for_row(row: dict[str, Any], open_categories: list[str]) -> list[str]:
    text = row_metadata_text(row)
    matches = []
    for category in open_categories:
        terms = QUEUE_CATEGORY_TERMS.get(category, {category})
        if any(term in text for term in terms):
            matches.append(category)
    return matches


def row_metadata_text(row: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in [
        "task_id",
        "source_task_id",
        "card_id",
        "source_id",
        "category",
        "residual_concept",
        "targeted_private_residual_family_v3",
        "concept_residual_label",
        "benchmark_evidence_level",
        "broad_private_family_v1",
        "public_safe_maturity_family_v4",
        "edge_contract_v3_family",
    ]:
        value = row.get(key)
        if isinstance(value, (str, int, float, bool)):
            pieces.append(str(value))
    tags = row.get("tags")
    if isinstance(tags, list):
        pieces.extend(str(tag) for tag in tags if isinstance(tag, (str, int, float, bool)))
    contract = row.get("decoder_contract")
    if isinstance(contract, dict):
        pieces.extend(decoder_contract_metadata_terms(contract))
    return " ".join(pieces).lower().replace("-", "_")


def decoder_contract_metadata_terms(contract: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ["policy", "semantic_family", "residual_label_hint", "type_family", "return_shape", "score_semantics"]:
        value = contract.get(key)
        if isinstance(value, (str, int, float, bool)):
            terms.append(str(value))
    for key in ["required_constructs", "composition_steps"]:
        value = contract.get(key)
        if isinstance(value, list):
            terms.extend(json.dumps(item, sort_keys=True) for item in value if isinstance(item, (dict, list, str, int, float, bool)))
    plan = contract.get("generation_plan")
    if isinstance(plan, dict):
        for key in ["policy"]:
            value = plan.get(key)
            if isinstance(value, (str, int, float, bool, dict, list)):
                terms.append(json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value))
        for key in ["skeleton_bias"]:
            value = plan.get(key)
            if isinstance(value, list):
                terms.extend(str(item) for item in value if isinstance(item, (str, int, float, bool)))
    return terms


def row_is_private_eval(row: dict[str, Any]) -> bool:
    if row.get("public_benchmark") is True or row.get("public_benchmark_row") is True:
        return False
    if row.get("public_tests_included") is True or row.get("public_benchmark_solutions_included") is True:
        return False
    if int(row.get("external_inference_calls") or 0) != 0:
        return False
    return bool(row.get("task_id") and row.get("prompt") and row.get("entry_point") and row.get("tests"))


def visible_task_manifest(rows: list[dict[str, Any]], *, visible_contract_mode: str = "full") -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "task_id": str(row.get("task_id") or ""),
                "source_task_id": str(row.get("source_task_id") or row.get("task_id") or ""),
                "card_id": str(row.get("card_id") or "private_candidate_floor_v2_probe"),
                "source_id": str(row.get("source_id") or "private_candidate_floor_v2_probe"),
                "case_type": str(row.get("case_type") or "private_eval"),
                "prompt": str(row.get("prompt") or ""),
                "entry_point": str(row.get("entry_point") or ""),
                "category": str(row.get("category") or ""),
                "residual_concept": str(row.get("residual_concept") or ""),
                "targeted_private_residual_family_v3": str(row.get("targeted_private_residual_family_v3") or ""),
                "concept_residual_label": str(row.get("concept_residual_label") or ""),
                "decoder_contract": safe_decoder_contract(
                    row.get("decoder_contract"),
                    visible_contract_mode=visible_contract_mode,
                ),
                "tags": [str(tag) for tag in row.get("tags", [])] if isinstance(row.get("tags"), list) else [],
                "benchmark_evidence_level": str(row.get("benchmark_evidence_level") or "private_candidate_floor_v2_probe"),
                "visible_task_only": True,
                "tests_exported": False,
                "canonical_solution_exported": False,
                "public_benchmark": False,
                "public_tests_used": False,
                "public_solutions_used": False,
            }
        )
    return out


def safe_decoder_contract(value: Any, *, visible_contract_mode: str = "full") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    minimal = visible_contract_mode == "minimal"
    allowed_scalar = {
        "policy",
        "semantic_family",
        "residual_label_hint",
        "type_family",
        "return_shape",
        "score_semantics",
        "visible_arg_count_hint",
        "full_body_required",
        "guardrail_only",
    }
    out: dict[str, Any] = {}
    for key in sorted(allowed_scalar):
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[key] = item
    if minimal:
        return out
    if isinstance(value.get("argument_roles"), dict):
        out["argument_roles"] = {
            str(key): str(role)
            for key, role in value["argument_roles"].items()
            if isinstance(key, str) and isinstance(role, (str, int, float, bool))
        }
    if isinstance(value.get("required_constructs"), list):
        out["required_constructs"] = [
            str(item)
            for item in value["required_constructs"]
            if isinstance(item, (str, int, float, bool))
        ]
    if isinstance(value.get("return_contract"), dict):
        return_contract = value["return_contract"]
        out["return_contract"] = {
            str(key): item
            for key, item in return_contract.items()
            if isinstance(key, str) and (isinstance(item, (str, int, float, bool)) or item is None)
        }
    if isinstance(value.get("generation_plan"), dict):
        generation_plan = value["generation_plan"]
        clean_plan: dict[str, Any] = {}
        for key in ["policy", "repair_strategy", "semantic_ranker_target"]:
            item = generation_plan.get(key)
            if isinstance(item, (str, int, float, bool)) or item is None:
                clean_plan[key] = item
        if isinstance(generation_plan.get("skeleton_bias"), list):
            clean_plan["skeleton_bias"] = [
                str(item)
                for item in generation_plan["skeleton_bias"]
                if isinstance(item, (str, int, float, bool))
            ]
        if isinstance(generation_plan.get("verifier_feedback"), list):
            clean_plan["verifier_feedback"] = [
                str(item)
                for item in generation_plan["verifier_feedback"]
                if isinstance(item, (str, int, float, bool))
            ]
        if clean_plan:
            out["generation_plan"] = clean_plan
    if isinstance(value.get("composition_steps"), list):
        steps: list[dict[str, str]] = []
        for step in value["composition_steps"]:
            if not isinstance(step, dict):
                continue
            clean_step: dict[str, str] = {}
            for key in ["semantic_family", "category"]:
                item = step.get(key)
                if isinstance(item, (str, int, float, bool)):
                    clean_step[key] = str(item)
            if clean_step:
                steps.append(clean_step)
        if steps:
            out["composition_steps"] = steps
    return out


def rust_command(args: argparse.Namespace) -> list[str]:
    exe = native_symliquid_cli()
    source_files = [
        ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "code_token_generator.rs",
    ]
    module_dir = ROOT / "crates" / "symliquid-cli" / "src" / "code_token_generator"
    if module_dir.exists():
        source_files.extend(sorted(module_dir.glob("*.rs")))
    exe_fresh = exe.exists() and all(exe.stat().st_mtime >= path.stat().st_mtime for path in source_files if path.exists())
    prefix = [str(exe)] if exe_fresh else ["cargo", "run", "-p", "symliquid-cli", "--"]
    return [
        *prefix,
        "train-code-token-generator",
        "--task-manifest",
        rel(resolve(args.task_manifest_out)),
        "--training-sources",
        rel(resolve(args.training_sources)),
        "--project-code-roots",
        "scripts,crates",
        "--seed",
        str(int(args.seed)),
        "--max-training-rows-per-source",
        str(max(1, int(args.max_training_rows_per_source))),
        "--max-project-files",
        str(max(1, int(args.max_project_files))),
        "--max-candidates-per-task",
        str(max(1, int(args.max_candidates_per_task))),
        "--checkpoint-out",
        rel(resolve(args.checkpoint_out)),
        "--out",
        rel(resolve(args.candidate_manifest_out)),
        "--report-out",
        rel(resolve(args.rust_report_out)),
    ]


def native_symliquid_cli() -> Path:
    native = ROOT / "target" / "release" / "symliquid-cli"
    windows = ROOT / "target" / "release" / "symliquid-cli.exe"
    return native if native.exists() else windows


def normalize_candidate_phases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["phase"] = "private_eval"
        item["private_candidate_floor_v2_probe"] = True
        item["public_benchmark_training_allowed"] = False
        item["external_inference_calls"] = int(item.get("external_inference_calls") or 0)
        out.append(item)
    return out


def summarize(
    private_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    rust_report: dict[str, Any],
    private_eval: dict[str, Any],
    *,
    residual_queue_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_ids = {str(row.get("task_id") or "") for row in private_rows}
    candidate_task_ids = {str(row.get("task_id") or "") for row in candidates if row.get("full_body_token_candidate") is True}
    by_family = Counter(str(row.get("residual_concept") or row.get("category") or "unknown") for row in private_rows)
    pass_rates = private_eval.get("concept_family_pass_rates") if isinstance(private_eval.get("concept_family_pass_rates"), dict) else {}
    residual_queue_eval = residual_queue_eval or {}
    summary = {
        "private_eval_task_count": len(private_rows),
        "candidate_task_count": len(candidate_task_ids),
        "no_candidate_task_count": len(task_ids - candidate_task_ids),
        "candidate_coverage_rate": len(candidate_task_ids) / len(task_ids) if task_ids else 0.0,
        "candidate_count": len(candidates),
        "full_body_token_candidate_count": sum(1 for row in candidates if row.get("full_body_token_candidate") is True),
        "grammar_masked_learned_token_candidate_count": sum(1 for row in candidates if row.get("grammar_masked_learned_token_candidate") is True),
        "benchmark_promotion_eligible_candidate_count": sum(1 for row in candidates if row.get("benchmark_promotion_eligible") is True),
        "template_like_candidate_count": sum(1 for row in candidates if row.get("template_like_candidate") is True),
        "loop_closure_candidate_count": sum(1 for row in candidates if row.get("loop_closure_generated") is True),
        "expression_memory_fallback_count": sum(1 for row in candidates if row.get("expression_memory_fallback") is True),
        "external_inference_calls": sum(int(row.get("external_inference_calls") or 0) for row in candidates),
        "rust_trigger_state": rust_report.get("trigger_state"),
        "training_rows_used": get_path(rust_report, ["summary", "training_rows_used"], 0),
        "ready_training_sources": get_path(rust_report, ["summary", "ready_training_sources"], 0),
        "private_trained_pass_rate": float(private_eval.get("trained_pass_rate") or 0.0),
        "private_trained_passed": int(private_eval.get("trained_passed") or 0),
        "private_residual_count": int(private_eval.get("residual_count") or 0),
        "private_family_counts": dict(sorted(by_family.items())),
        "private_family_pass_rates": pass_rates,
        "public_prompts_embedded": False,
        "public_tests_embedded": False,
        "public_solutions_embedded": False,
        "runtime_external_inference_calls": 0,
    }
    if residual_queue_eval:
        summary["residual_queue_eval"] = attach_residual_queue_pass_rates(
            residual_queue_eval,
            pass_rates=pass_rates,
        )
    return summary


def summarize_residual_queue_eval(
    private_rows: list[dict[str, Any]],
    *,
    row_matches: dict[str, list[str]],
    residual_queue_profile: dict[str, Any],
) -> dict[str, Any]:
    open_categories = [str(item) for item in residual_queue_profile.get("open_categories", [])]
    category_counts: Counter[str] = Counter()
    category_families: dict[str, set[str]] = defaultdict(set)
    for row in private_rows:
        task_id = str(row.get("task_id") or "")
        matches = row_matches.get(task_id) or []
        family = str(row.get("residual_concept") or row.get("category") or "unknown")
        for category in matches:
            category_counts[category] += 1
            category_families[category].add(family)
    missing = sorted(category for category in open_categories if category_counts.get(category, 0) <= 0)
    return {
        "residual_queue_row_count": int(residual_queue_profile.get("row_count") or 0),
        "residual_queue_open_row_count": int(residual_queue_profile.get("open_row_count") or 0),
        "open_categories": open_categories,
        "open_category_counts": residual_queue_profile.get("open_category_counts") or {},
        "targeted_private_eval_task_count": len(private_rows),
        "targeted_private_eval_category_counts": dict(sorted(category_counts.items())),
        "targeted_private_eval_families_by_category": {
            category: sorted(families)
            for category, families in sorted(category_families.items())
        },
        "missing_open_categories": missing,
    }


def attach_residual_queue_pass_rates(
    residual_queue_eval: dict[str, Any],
    *,
    pass_rates: dict[str, Any],
) -> dict[str, Any]:
    out = dict(residual_queue_eval)
    families_by_category = out.get("targeted_private_eval_families_by_category")
    if not isinstance(families_by_category, dict):
        return out
    category_pass_rates: dict[str, dict[str, float | None]] = {}
    category_min_pass_rate: dict[str, float | None] = {}
    for category, families in families_by_category.items():
        if not isinstance(families, list):
            continue
        rates: dict[str, float | None] = {}
        numeric_rates = []
        for family in families:
            value = pass_rates.get(str(family))
            rate = float(value) if isinstance(value, (int, float)) else None
            rates[str(family)] = rate
            if rate is not None:
                numeric_rates.append(rate)
        category_pass_rates[str(category)] = rates
        category_min_pass_rate[str(category)] = min(numeric_rates) if numeric_rates else None
    out["targeted_private_eval_family_pass_rates_by_category"] = category_pass_rates
    out["targeted_private_eval_min_family_pass_rate_by_category"] = category_min_pass_rate
    return out


def compact_private_eval(private_eval: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_task_count": private_eval.get("eval_task_count"),
        "trained_passed": private_eval.get("trained_passed"),
        "trained_pass_rate": private_eval.get("trained_pass_rate"),
        "residual_count": private_eval.get("residual_count"),
        "concept_residual_counts": private_eval.get("concept_residual_counts"),
        "concept_family_pass_rates": private_eval.get("concept_family_pass_rates"),
        "private_verification": private_eval.get("private_verification"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    queue_eval = summary.get("residual_queue_eval") if isinstance(summary.get("residual_queue_eval"), dict) else {}
    lines = [
        "# Candidate Floor v2 Private Token Probe",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- private eval tasks: `{summary.get('private_eval_task_count')}`",
        f"- candidate coverage: `{summary.get('candidate_coverage_rate')}`",
        f"- no-candidate tasks: `{summary.get('no_candidate_task_count')}`",
        f"- full-body candidates: `{summary.get('full_body_token_candidate_count')}`",
        f"- grammar-masked learned candidates: `{summary.get('grammar_masked_learned_token_candidate_count')}`",
        f"- private trained pass rate: `{summary.get('private_trained_pass_rate')}`",
        f"- fallback/template counts: `{summary.get('expression_memory_fallback_count')}` / `{summary.get('template_like_candidate_count')}`",
    ]
    if queue_eval:
        lines.extend(
            [
                f"- residual queue open categories: `{queue_eval.get('open_categories')}`",
                f"- targeted private eval rows: `{queue_eval.get('targeted_private_eval_task_count')}`",
                f"- targeted private eval category counts: `{queue_eval.get('targeted_private_eval_category_counts')}`",
                f"- missing queue categories: `{queue_eval.get('missing_open_categories')}`",
            ]
        )
    lines.extend(["", "## Gates"])
    for row in report.get("gates", []):
        if isinstance(row, dict):
            lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}` ({row.get('severity')})")
    lines.append("")
    return "\n".join(lines)


def remove_stale(paths: list[Path]) -> list[str]:
    removed = []
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                removed.append(rel(path))
        except OSError:
            continue
    return removed


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        key = str(row.get("task_id") or row.get("entry_point") or stable_key(row))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def stable_key(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def read_json(path: Path, default: Any | None = None) -> Any:
    default = {} if default is None else default
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
