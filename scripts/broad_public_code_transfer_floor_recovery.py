#!/usr/bin/env python3
"""Private-only recovery plan for the broad public code-transfer floor.

This script does not run public calibration. It reads public calibration
reports only as aggregate metrics and residual family labels, creates
source-agnostic private pressure rows for the current weak-card concepts, and
optionally runs a same-seed private A/B over the decoder/ranker/verifier patch.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from code_residual_curriculum import build_private_rows, verify_private_solution_rows


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_FLOOR = 0.70
WEAK_CARD_TARGETS = [
    "source_mbpp",
    "source_evalplus",
    "source_bigcodebench",
    "source_livecodebench",
]
INTENDED_BEHAVIOR_FOCUS = "intended_behavior_transfer_private_curriculum"
SEMANTIC_BROAD_FLOOR_FOCUS = "broad_floor_semantic_transfer_private_curriculum"
LEGACY_INTENDED_BEHAVIOR_FOCUS = "livecodebench_intended_behavior_private_curriculum"
PRIVATE_ABLATION_FOCUS_ORDER = [
    SEMANTIC_BROAD_FLOOR_FOCUS,
    "adapter_runtime_dependency_handling",
    INTENDED_BEHAVIOR_FOCUS,
    "type_and_return_shape",
    "algorithmic_planning",
    "edge_contract_v2_private_residual_curriculum",
    "admissibility_and_interface",
]
PRIVATE_SEMANTIC_TARGET_FAMILIES = [
    "edge_case",
    "local_code_generation_adapter_needed",
    "type_handling",
    "external_dependency_missing",
    "algorithm_choice",
]
NON_PROMOTABLE_PRIVATE_CONCEPTS = {
    "candidate_floor_v2",
    "typed_interface_skeleton",
}
RESIDUAL_TO_FOCUS = {
    "type_handling": SEMANTIC_BROAD_FLOOR_FOCUS,
    "edge_case": SEMANTIC_BROAD_FLOOR_FOCUS,
    "local_code_generation_adapter_needed": SEMANTIC_BROAD_FLOOR_FOCUS,
    "algorithm_choice": "algorithmic_planning",
    "no_admissible_candidate": SEMANTIC_BROAD_FLOOR_FOCUS,
    "interface_fidelity": "admissibility_and_interface",
    "runtime_load_failure": SEMANTIC_BROAD_FLOOR_FOCUS,
    "external_dependency_missing": SEMANTIC_BROAD_FLOOR_FOCUS,
    "verification_cascade_compile": "admissibility_and_interface",
}
RESIDUAL_TO_FOCUSES = {
    "type_handling": [SEMANTIC_BROAD_FLOOR_FOCUS, "type_and_return_shape"],
    "edge_case": [SEMANTIC_BROAD_FLOOR_FOCUS, "edge_contract_v2_private_residual_curriculum"],
    "local_code_generation_adapter_needed": [
        SEMANTIC_BROAD_FLOOR_FOCUS,
        "admissibility_and_interface",
        "adapter_runtime_dependency_handling",
    ],
    "algorithm_choice": ["algorithmic_planning"],
    "no_admissible_candidate": [SEMANTIC_BROAD_FLOOR_FOCUS, "admissibility_and_interface"],
    "interface_fidelity": ["admissibility_and_interface", "type_and_return_shape"],
    "runtime_load_failure": [
        SEMANTIC_BROAD_FLOOR_FOCUS,
        "adapter_runtime_dependency_handling",
        "admissibility_and_interface",
    ],
    "external_dependency_missing": [
        SEMANTIC_BROAD_FLOOR_FOCUS,
        "adapter_runtime_dependency_handling",
        "admissibility_and_interface",
    ],
    "verification_cascade_compile": ["admissibility_and_interface"],
}


def parse_args() -> argparse.Namespace:
    default_private = Path(
        "D:/ProjectTheseus/training_data/high_transfer/private_train/"
        "broad_public_code_transfer_floor_recovery_v1_residual_code_lm_tasks.jsonl"
    )
    if not default_private.anchor or not Path(default_private.anchor).exists():
        default_private = REPORTS / "broad_public_code_transfer_floor_recovery_private_rows.jsonl"
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--decoder-gate", default="reports/decoder_v2_private_ablation_gate.json")
    parser.add_argument("--transfer-proof", default="reports/private_public_transfer_proof.json")
    parser.add_argument(
        "--calibration-report",
        default="auto",
        help="Optional calibration diagnostics, or 'auto' to use the newest clean weak-card public calibration. Only residual type counts are read; prompts/tests/solutions are never copied into private rows.",
    )
    parser.add_argument(
        "--residual-report",
        default="reports/public_code_transfer_residual_report.json",
        help="Optional sanitized residual diagnosis. Used only for adapter-adjusted residual counts; never as public training content.",
    )
    parser.add_argument("--private-out", default=str(default_private))
    parser.add_argument("--manifest-out", default="reports/broad_public_code_transfer_floor_recovery_private_manifest.jsonl")
    parser.add_argument("--ablation-out", default="reports/broad_public_code_transfer_floor_recovery_ablation.json")
    parser.add_argument("--ablation-markdown-out", default="reports/broad_public_code_transfer_floor_recovery_ablation.md")
    parser.add_argument("--out", default="reports/broad_public_code_transfer_floor_recovery.json")
    parser.add_argument("--markdown-out", default="reports/broad_public_code_transfer_floor_recovery.md")
    parser.add_argument("--rows-per-focus", type=int, default=72)
    parser.add_argument("--task-limit", type=int, default=24)
    parser.add_argument("--candidates-per-task", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7459)
    parser.add_argument("--min-semantic-lift", type=float, default=0.03)
    parser.add_argument("--execute-ablation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = read_json(resolve(args.matrix), {})
    decoder_gate = read_json(resolve(args.decoder_gate), {})
    transfer_proof = read_json(resolve(args.transfer_proof), {})
    calibration_report_path = select_calibration_report(args.calibration_report)
    calibration_report = read_json(calibration_report_path, {})
    residual_report = read_json(resolve(args.residual_report), {})
    clusters = residual_clusters(matrix)
    calibration_residuals = calibration_residual_type_counts(calibration_report, residual_report)
    pressure_rows = build_pressure_rows(
        clusters,
        rows_per_focus=max(8, args.rows_per_focus),
        seed=args.seed,
        calibration_residuals=calibration_residuals,
    )
    private_check = verify_private_solution_rows(pressure_rows)
    private_out = resolve(args.private_out)
    manifest_out = resolve(args.manifest_out)
    write_jsonl(private_out, pressure_rows)
    eval_rows = stratified_private_eval_rows(pressure_rows, max(1, args.task_limit))
    write_jsonl(manifest_out, eval_rows)
    ablation = run_ablation(args, manifest_out) if args.execute_ablation else {"status": "SKIPPED"}
    payload = build_payload(
        args=args,
        matrix=matrix,
        decoder_gate=decoder_gate,
        transfer_proof=transfer_proof,
        calibration_report=calibration_report,
        calibration_report_path=calibration_report_path,
        residual_report=residual_report,
        clusters=clusters,
        pressure_rows=pressure_rows,
        private_check=private_check,
        private_out=private_out,
        manifest_out=manifest_out,
        ablation=ablation,
    )
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def residual_clusters(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = matrix.get("rows") if isinstance(matrix.get("rows"), list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        card = str(row.get("card_id") or "")
        if card not in WEAK_CARD_TARGETS:
            continue
        public_task_count = int(row.get("public_task_count") or 0)
        multi_passed = int(row.get("multi_stream_passed") or 0)
        pass_rate = safe_float(row.get("multi_stream_pass_rate"), 0.0)
        residuals = row.get("residual_family_counts") if isinstance(row.get("residual_family_counts"), dict) else {}
        required_passes = math.ceil(PUBLIC_FLOOR * max(1, public_task_count))
        additional_needed = max(0, required_passes - multi_passed)
        top_residuals = [
            {
                "family": str(family),
                "count": int(count or 0),
                "private_focus": primary_focus_for_residual(str(family), card),
                "private_focuses": focuses_for_residual(str(family), card),
            }
            for family, count in sorted(residuals.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))
        ]
        out.append(
            {
                "card_id": card,
                "pass_rate": pass_rate,
                "public_task_count": public_task_count,
                "multi_stream_passed": multi_passed,
                "floor": PUBLIC_FLOOR,
                "additional_passes_needed_for_floor": additional_needed,
                "selected_report": row.get("selected_report"),
                "top_residuals": top_residuals[:6],
                "next_private_patches": next_private_patches(card, top_residuals),
                "public_tests_used": False,
                "public_solutions_used": False,
            }
        )
    return out


def next_private_patches(card: str, top_residuals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patches = []
    for item in top_residuals[:4]:
        family = str(item.get("family") or "unknown")
        focus = str(item.get("private_focus") or primary_focus_for_residual(family, card))
        focuses = item.get("private_focuses") if isinstance(item.get("private_focuses"), list) else [focus]
        decoder_patch = "broad_public_floor_recovery_v1 ranker/prefilter keeps structurally valid full-body candidates ahead of generic passthroughs"
        verifier_patch = "staged contract verifier remains fail-fast: visible args -> return shape -> required constructs -> semantic family before sandbox/full tests"
        if family in {"runtime_load_failure", "external_dependency_missing", "local_code_generation_adapter_needed"}:
            decoder_patch = "private admissibility/runtime pressure requires guarded optional imports with deterministic pure-Python fallbacks before ranking"
            verifier_patch = "optional dependency verifier rejects unguarded imports and rewards fallback-safe adapter bodies before expensive sandbox work"
        patches.append(
            {
                "patch_id": f"{family}_private_pressure_for_{card}",
                "residual_family": family,
                "private_focus": focus,
                "private_focuses": focuses,
                "decoder_patch": decoder_patch,
                "verifier_patch": verifier_patch,
                "public_benchmark_content": "aggregate residual family labels only; no public tests, public solutions, or copied public prompts",
            }
        )
    return patches


def build_pressure_rows(
    clusters: list[dict[str, Any]],
    *,
    rows_per_focus: int,
    seed: int,
    calibration_residuals: Counter[str] | None = None,
) -> list[dict[str, Any]]:
    class_counts: Counter[str] = Counter()
    focus_counts: Counter[str] = Counter()
    for cluster in clusters:
        card = str(cluster.get("card_id") or "")
        for item in cluster.get("top_residuals", []):
            if isinstance(item, dict):
                family = str(item.get("family") or "unknown")
                count = int(item.get("count") or 0)
                class_counts[family] += count
                for focus in focuses_for_residual(family, card):
                    focus_counts[focus] += count
    for family, count in (calibration_residuals or Counter()).items():
        if family in RESIDUAL_TO_FOCUS:
            class_counts[str(family)] += int(count or 0)
            for focus in focuses_for_residual(str(family), None):
                focus_counts[focus] += int(count or 0)
    if not class_counts:
        class_counts.update({"type_handling": 1, "edge_case": 1, "algorithm_choice": 1})
        for family, count in class_counts.items():
            for focus in focuses_for_residual(family, None):
                focus_counts[focus] += count
    ordered_focuses = [focus for focus, _count in focus_counts.most_common()]
    for focus in [
        INTENDED_BEHAVIOR_FOCUS,
        SEMANTIC_BROAD_FLOOR_FOCUS,
        "adapter_runtime_dependency_handling",
        "type_and_return_shape",
        "edge_contract_v2_private_residual_curriculum",
        "admissibility_and_interface",
        "algorithmic_planning",
    ]:
        if focus not in ordered_focuses:
            ordered_focuses.append(focus)
    ordered_focuses = [focus for focus in ordered_focuses if focus in set(PRIVATE_ABLATION_FOCUS_ORDER)]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    summary = {
        "class_counts": dict(class_counts),
        "failed_task_hashes": [],
        "targets": [],
    }
    broad_context = {
        "cards_below_floor": WEAK_CARD_TARGETS,
        "active_card": "",
        "active_residual_family_counts": dict(class_counts),
        "no_clean_student_evidence_cards": [],
        "loader_only_cards": [],
    }
    for focus_index, focus in enumerate(ordered_focuses):
        focus_rows = focus_row_budget(focus, focus_counts, rows_per_focus)
        generated = build_private_rows(
            summary,
            seed=seed + focus_index * 101,
            max_rows=focus_rows,
            broad_context=broad_context,
            concept_focus=focus,
        )
        for row in generated:
            row = dict(row)
            if private_row_concept(row) in NON_PROMOTABLE_PRIVATE_CONCEPTS:
                continue
            base_id = str(row.get("task_id") or f"row_{len(rows):04d}")
            row["task_id"] = f"broad_public_floor_recovery_v1_{focus}_{base_id}"
            if row["task_id"] in seen:
                continue
            seen.add(row["task_id"])
            row["source_id"] = "local_generated_broad_public_floor_recovery_private_pressure"
            row["split"] = "train"
            tags = [str(item) for item in row.get("tags") or []]
            tags.extend(["broad_public_code_transfer_floor_recovery_v1", "private_only_floor_recovery"])
            row["tags"] = sorted(set(tags))
            contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
            plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
            plan = {
                **plan,
                "policy": "broad_public_code_transfer_floor_recovery_v1",
                "repair_strategy": "visible_interface_return_shape_required_construct_then_semantic_family",
                "skeleton_bias": sorted({focus, str(row.get("residual_concept") or ""), *map(str, contract.get("required_constructs") or [])}),
                "verifier_feedback": [
                    "fail_fast_visible_args",
                    "fail_fast_return_shape",
                    "fail_fast_required_constructs",
                    "semantic_family_before_full_sandbox",
                ],
            }
            contract["generation_plan"] = plan
            row["decoder_contract"] = contract
            provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
            provenance.update(
                {
                    "policy": "project_theseus_broad_public_code_transfer_floor_recovery_v1",
                    "private_focus": focus,
                    "focus_row_budget": focus_rows,
                    "private_only": True,
                    "public_task_ids_hashed_only": True,
                    "public_prompts_used": False,
                    "public_tests_used": False,
                    "public_solutions_used": False,
                    "source_public_signal": "aggregate residual family counts only",
                }
            )
            row["provenance"] = provenance
            row["benchmark_evidence_level"] = "private_broad_public_floor_recovery_train_only"
            row["public_benchmark"] = False
            row["public_benchmark_solutions_included"] = False
            row["public_tests_included"] = False
            rows.append(row)
    return rows


def stratified_private_eval_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Build a small private eval manifest that covers every active recovery focus.

    The previous first-N slice could accidentally verify only one focus because
    private rows are generated focus-by-focus. The broad-floor goal needs the
    same-seed heldout check to touch type/return, algorithmic planning, edge
    contracts, and admissibility/runtime pressure whenever those rows exist.
    """

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fallback: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        row_id = str(row.get("task_id") or "")
        if not row_id or row_id in seen:
            continue
        seen.add(row_id)
        focus = recovery_focus(row)
        prepared = dict(row, split="eval")
        if focus in PRIVATE_ABLATION_FOCUS_ORDER:
            buckets[focus].append(prepared)
        else:
            fallback.append(prepared)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    while len(selected) < limit and any(buckets.values()):
        made_progress = False
        for focus in PRIVATE_ABLATION_FOCUS_ORDER:
            if len(selected) >= limit:
                break
            if not buckets[focus]:
                continue
            candidate = buckets[focus].pop(0)
            candidate_id = str(candidate.get("task_id") or "")
            if candidate_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate_id)
            made_progress = True
        if not made_progress:
            break

    for candidate in fallback:
        if len(selected) >= limit:
            break
        candidate_id = str(candidate.get("task_id") or "")
        if candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate_id)
    if len(selected) < limit:
        for focus in PRIVATE_ABLATION_FOCUS_ORDER:
            for candidate in buckets[focus]:
                if len(selected) >= limit:
                    break
                candidate_id = str(candidate.get("task_id") or "")
                if candidate_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate_id)
    return selected


def recovery_focus(row: dict[str, Any]) -> str:
    text_parts = [
        str(row.get("task_id") or ""),
        str(row.get("residual_concept") or ""),
        str(row.get("category") or ""),
        " ".join(map(str, row.get("tags") or [])),
    ]
    contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
    text_parts.extend(map(str, plan.get("skeleton_bias") or []))
    text = " ".join(text_parts).lower()
    if (
        INTENDED_BEHAVIOR_FOCUS in text
        or "intended_behavior_transfer" in text
        or LEGACY_INTENDED_BEHAVIOR_FOCUS in text
        or "livecodebench_intended_behavior" in text
    ):
        return INTENDED_BEHAVIOR_FOCUS
    if "adapter_runtime_dependency_handling" in text or "optional_dependency" in text or "runtime_load_failure" in text:
        return "adapter_runtime_dependency_handling"
    if (
        "admissibility_and_interface" in text
        or "local_code_generation_adapter_needed" in text
    ):
        return "admissibility_and_interface"
    if "type_and_return_shape" in text or "receiver_return_shape_contract" in text:
        return "type_and_return_shape"
    if "algorithmic_planning" in text or "algorithm_choice" in text:
        return "algorithmic_planning"
    if "edge_contract_v2_private_residual_curriculum" in text or "edge_contract_v2" in text:
        return "edge_contract_v2_private_residual_curriculum"
    return "unknown"


def private_row_concept(row: dict[str, Any]) -> str:
    return str(row.get("residual_concept") or row.get("category") or "unknown")


def semantic_target_family_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    """Classify private pressure rows by the semantic transfer failures they target."""

    counts: Counter[str] = Counter()
    for row in rows:
        contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
        plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
        parts = [
            str(row.get("task_id") or ""),
            str(row.get("residual_concept") or ""),
            str(row.get("category") or ""),
            str(row.get("prompt") or ""),
            str(row.get("tests") or ""),
            " ".join(map(str, row.get("tags") or [])),
            " ".join(map(str, contract.get("required_constructs") or [])),
            " ".join(map(str, plan.get("skeleton_bias") or [])),
            " ".join(map(str, plan.get("verifier_feedback") or [])),
        ]
        text = " ".join(parts).lower()
        if any(token in text for token in ("edge_case", "edge_contract", "boundary", "empty", "singleton", "tie")):
            counts["edge_case"] += 1
        if any(
            token in text
            for token in (
                "local_code_generation_adapter_needed",
                "admissibility_and_interface",
                "adapter_runtime",
                "runtime_load_failure",
                "visible_args",
            )
        ):
            counts["local_code_generation_adapter_needed"] += 1
        if any(
            token in text
            for token in (
                "type_handling",
                "type_and_return_shape",
                "return_shape",
                "type_contract",
                "receiver_return_shape_contract",
            )
        ):
            counts["type_handling"] += 1
        if any(
            token in text
            for token in (
                "external_dependency_missing",
                "optional_dependency",
                "pure-python fallback",
                "fallback",
                "pandas",
                "numpy",
                "requests",
                "sklearn",
                "matplotlib",
                "beautifulsoup",
                "nltk",
            )
        ):
            counts["external_dependency_missing"] += 1
        if any(
            token in text
            for token in (
                "algorithm_choice",
                "algorithmic_planning",
                "execution_shaped_programs",
                "sliding_window",
                "frequency",
                "graph",
                "search",
                "dynamic_programming",
            )
        ):
            counts["algorithm_choice"] += 1
    return counts


def run_ablation(args: argparse.Namespace, manifest_out: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/broad_transfer_residual_decoder_ablation.py",
        "--manifest-in",
        rel(manifest_out),
        "--manifest-out",
        "reports/broad_public_code_transfer_floor_recovery_ablation_manifest.jsonl",
        "--task-limit",
        str(max(1, args.task_limit)),
        "--candidates-per-task",
        str(max(1, args.candidates_per_task)),
        "--seed",
        str(args.seed),
        "--out",
        args.ablation_out,
        "--markdown-out",
        args.ablation_markdown_out,
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    report = read_json(resolve(args.ablation_out), {})
    return {
        "status": "COMPLETED" if completed.returncode == 0 else "FAILED",
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
        "report": rel(resolve(args.ablation_out)),
        "report_status": report.get("status"),
        "delta": report.get("delta") if isinstance(report.get("delta"), dict) else {},
        "gates": report.get("gates") if isinstance(report.get("gates"), list) else [],
        "private_only": get_path(report, ["manifest", "public_task_count"], 1) == 0,
    }


def build_payload(
    *,
    args: argparse.Namespace,
    matrix: dict[str, Any],
    decoder_gate: dict[str, Any],
    transfer_proof: dict[str, Any],
    calibration_report: dict[str, Any],
    calibration_report_path: Path,
    residual_report: dict[str, Any],
    clusters: list[dict[str, Any]],
    pressure_rows: list[dict[str, Any]],
    private_check: dict[str, Any],
    private_out: Path,
    manifest_out: Path,
    ablation: dict[str, Any],
) -> dict[str, Any]:
    matrix_summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
    public_rate = safe_float(matrix_summary.get("real_public_pass_rate"), 0.0)
    floor_cleared = public_rate >= PUBLIC_FLOOR
    aggregate_public_tasks = int(matrix_summary.get("real_public_task_count") or 0)
    aggregate_passed = int(matrix_summary.get("real_public_multi_passed") or 0)
    aggregate_needed = max(0, math.ceil(PUBLIC_FLOOR * max(1, aggregate_public_tasks)) - aggregate_passed)
    residual_counter: Counter[str] = Counter()
    focus_counter: Counter[str] = Counter()
    for cluster in clusters:
        for item in cluster.get("top_residuals", []):
            if isinstance(item, dict):
                residual_counter[str(item.get("family") or "unknown")] += int(item.get("count") or 0)
                for focus in item.get("private_focuses") or [item.get("private_focus") or "unknown"]:
                    focus_counter[str(focus)] += int(item.get("count") or 0)
    fresh_calibration_residuals = calibration_residual_type_counts(calibration_report, residual_report)
    adapter_adjustment = calibration_adapter_adjustment(residual_report)
    next_blocker = classify_next_blocker(residual_counter, fresh_calibration_residuals, public_rate)
    pressure_counter = Counter(str(row.get("residual_concept") or "unknown") for row in pressure_rows)
    non_promotable_counter = Counter(
        private_row_concept(row)
        for row in pressure_rows
        if private_row_concept(row) in NON_PROMOTABLE_PRIVATE_CONCEPTS
    )
    private_rows_with_tests = sum(1 for row in pressure_rows if str(row.get("tests") or "").strip())
    semantic_target_counts = semantic_target_family_counts(pressure_rows)
    missing_semantic_targets = [
        family for family in PRIVATE_SEMANTIC_TARGET_FAMILIES if semantic_target_counts.get(family, 0) == 0
    ]
    manifest_rows = read_jsonl(manifest_out)
    manifest_focus_counts = Counter(recovery_focus(row) for row in manifest_rows)
    missing_manifest_focuses = [
        focus
        for focus in PRIVATE_ABLATION_FOCUS_ORDER
        if any(recovery_focus(row) == focus for row in pressure_rows)
        and manifest_focus_counts.get(focus, 0) == 0
    ]
    ablation_delta = ablation.get("delta") if isinstance(ablation.get("delta"), dict) else {}
    ablation_completed = ablation.get("status") in {"COMPLETED", "SKIPPED"}
    ablation_executed = ablation.get("status") == "COMPLETED"
    semantic_lift = safe_float(ablation_delta.get("semantic_test_passed_task_rate_delta"), 0.0)
    semantic_lift_ready = ablation_executed and semantic_lift >= float(args.min_semantic_lift)
    semantic_family_audit = semantic_family_delta_audit(ablation_delta)
    ablation_summary = read_json(resolve(ablation.get("report", args.ablation_out)), {}) if ablation.get("report") else {}
    patched_summary = get_path(ablation_summary, ["patched", "summary"], {})
    patched_learned_token_only = (
        isinstance(patched_summary, dict)
        and int(patched_summary.get("private_receiver_eligible_candidate_count") or 0) > 0
        and int(patched_summary.get("non_grammar_private_receiver_eligible_candidate_count") or 0) == 0
        and int(patched_summary.get("grammar_masked_learned_token_candidate_count") or 0)
        >= int(patched_summary.get("private_receiver_eligible_candidate_count") or 0)
    )
    targeted_private_signal = (
        safe_float(ablation_delta.get("passed_task_rate_delta"), 0.0) >= 0.0
        and safe_float(ablation_delta.get("no_admissible_rate_delta"), 0.0) <= 0.0
        and (
            safe_float(ablation_delta.get("passed_task_rate_delta"), 0.0) > 0.0
            or safe_float(ablation_delta.get("broad_transfer_residual_task_count_delta"), 0.0) > 0.0
            or safe_float(ablation_delta.get("eligible_receiver_inventory_task_count_delta"), 0.0) > 0.0
            or not ablation_executed
        )
    )
    gap_explained = bool(clusters and residual_counter and all(cluster.get("next_private_patches") for cluster in clusters))
    gates = [
        gate("decoder_private_gate_green", decoder_gate.get("trigger_state") == "GREEN" and bool(decoder_gate.get("ready_for_public_calibration")), decoder_gate.get("summary")),
        gate("private_public_transfer_proof_green", transfer_proof.get("trigger_state") == "GREEN" and bool(transfer_proof.get("ready_for_public_calibration")), transfer_proof.get("summary")),
        gate("weak_cards_selected", sorted(card["card_id"] for card in clusters) == sorted(WEAK_CARD_TARGETS), [card["card_id"] for card in clusters]),
        gate("private_pressure_rows_written", len(pressure_rows) > 0 and private_check.get("failure_count") == 0, {"rows": len(pressure_rows), "solution_check": private_check}),
        gate(
            "template_skeleton_private_rows_absent",
            not non_promotable_counter,
            {
                "non_promotable_private_concept_counts": dict(non_promotable_counter),
                "blocked_concepts": sorted(NON_PROMOTABLE_PRIVATE_CONCEPTS),
            },
        ),
        gate(
            "private_rows_have_behavior_tests",
            len(pressure_rows) > 0 and private_rows_with_tests == len(pressure_rows),
            {"private_rows_with_tests": private_rows_with_tests, "private_pressure_rows": len(pressure_rows)},
        ),
        gate("public_training_leak_absent", public_leak_count(pressure_rows) == 0, {"public_leak_count": public_leak_count(pressure_rows)}),
        gate(
            "same_seed_private_ablation_manifest_covers_active_focuses",
            not missing_manifest_focuses,
            {"manifest_focus_counts": dict(manifest_focus_counts.most_common()), "missing_focuses": missing_manifest_focuses},
        ),
        gate(
            "private_pressure_targets_required_semantic_families",
            not missing_semantic_targets,
            {"semantic_target_family_counts": dict(semantic_target_counts), "missing_targets": missing_semantic_targets},
        ),
        gate("same_seed_private_ablation_completed_or_skipped", ablation_completed, {"status": ablation.get("status"), "returncode": ablation.get("returncode")}),
        gate("same_seed_private_ablation_private_only", not ablation_executed or bool(ablation.get("private_only")), {"ablation_status": ablation.get("status")}),
        gate(
            "same_seed_private_semantic_lift_required_before_promotion",
            semantic_lift_ready,
            {
                "semantic_test_passed_task_rate_delta": semantic_lift,
                "minimum": args.min_semantic_lift,
                "semantic_test_passed_task_count_delta": ablation_delta.get("semantic_test_passed_task_count_delta"),
                "rule": "coverage/distribution changes are not enough; the patched same-seed arm must pass more private behavioral tests",
            },
        ),
        gate(
            "same_seed_private_semantic_family_delta_clean",
            ablation_executed
            and bool(semantic_family_audit.get("all_target_families_positive"))
            and bool(semantic_family_audit.get("all_target_families_non_regressing")),
            {
                "semantic_family_delta_audit": semantic_family_audit,
                "rule": "aggregate lift is not enough; every active private residual target family must improve without regression before recovery evidence can be treated as promotion-ready",
            },
        ),
        gate(
            "patched_candidates_use_grammar_masked_learned_token_source",
            patched_learned_token_only,
            {
                "patched_summary": {
                    "grammar_masked_learned_token_candidate_count": get_path(patched_summary, ["grammar_masked_learned_token_candidate_count"], None)
                    if isinstance(patched_summary, dict)
                    else None,
                    "private_receiver_eligible_candidate_count": get_path(patched_summary, ["private_receiver_eligible_candidate_count"], None)
                    if isinstance(patched_summary, dict)
                    else None,
                    "non_grammar_private_receiver_eligible_candidate_count": get_path(patched_summary, ["non_grammar_private_receiver_eligible_candidate_count"], None)
                    if isinstance(patched_summary, dict)
                    else None,
                    "prompt_program_candidate_count": get_path(patched_summary, ["prompt_program_candidate_count"], None)
                    if isinstance(patched_summary, dict)
                    else None,
                },
                "rule": "promotion-facing private repair candidates must be grammar-masked learned-token full-body candidates",
            },
        ),
        gate("targeted_private_signal_or_gap_explained", targeted_private_signal or gap_explained, {"delta": ablation_delta, "gap_explained": gap_explained}),
        gate("public_calibration_not_run", True, "operator lock is respected; this script never runs public calibration"),
        gate("floor_cleared_or_remaining_gap_explained", floor_cleared or gap_explained, {"floor_cleared": floor_cleared, "gap_explained": gap_explained}),
    ]
    trigger = "GREEN" if all(item["passed"] for item in gates) else "YELLOW"
    if not clusters or public_leak_count(pressure_rows) > 0:
        trigger = "RED"
    return {
        "policy": "project_theseus_broad_public_code_transfer_floor_recovery_v1",
        "created_utc": now(),
        "trigger_state": trigger,
        "status": "floor_cleared" if floor_cleared else "remaining_gap_explained_with_private_patches" if gap_explained else "incomplete",
        "summary": {
            "broad_public_pass_rate": public_rate,
            "public_floor": PUBLIC_FLOOR,
            "public_transfer_floor_cleared": floor_cleared,
            "aggregate_public_tasks": aggregate_public_tasks,
            "aggregate_public_passed": aggregate_passed,
            "aggregate_additional_passes_needed_for_floor": aggregate_needed,
            "weak_cards": [cluster["card_id"] for cluster in clusters],
            "dominant_residual_families": dict(residual_counter.most_common()),
            "fresh_calibration_residual_families": dict(fresh_calibration_residuals.most_common()),
            "fresh_calibration_adapter_adjustment": adapter_adjustment,
            "private_focus_counts": dict(focus_counter.most_common()),
            "semantic_target_family_counts": dict(semantic_target_counts),
            "missing_semantic_target_families": missing_semantic_targets,
            "next_blocker": next_blocker,
            "private_pressure_row_count": len(pressure_rows),
            "private_pressure_rows_with_behavior_tests": private_rows_with_tests,
            "private_pressure_residual_concepts": dict(pressure_counter.most_common(12)),
            "non_promotable_private_concepts": dict(non_promotable_counter.most_common()),
            "same_seed_ablation_manifest_focus_counts": dict(manifest_focus_counts.most_common()),
            "same_seed_ablation_status": ablation.get("status"),
            "same_seed_private_signal": targeted_private_signal,
            "same_seed_private_semantic_lift": semantic_lift,
            "same_seed_private_semantic_lift_ready": semantic_lift_ready,
            "same_seed_semantic_family_delta_audit": semantic_family_audit,
            "patched_candidates_learned_token_only": patched_learned_token_only,
            "remaining_gap_explained": gap_explained,
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_calibration_run": False,
            "model_growth_allowed": False,
        },
        "inputs": {
            "matrix": args.matrix,
            "decoder_gate": args.decoder_gate,
            "transfer_proof": args.transfer_proof,
            "calibration_report": rel(calibration_report_path),
            "residual_report": args.residual_report,
        },
        "outputs": {
            "private_pressure_rows": rel(private_out),
            "private_ablation_manifest": rel(manifest_out),
            "same_seed_ablation": ablation.get("report", args.ablation_out),
        },
        "residual_clusters": clusters,
        "same_seed_ablation": ablation,
        "gates": gates,
        "rules": {
            "public_benchmarks": "aggregate metrics and residual family labels only; no public tests/solutions/prompts enter training",
            "public_calibration": "operator locked; this recovery pass does not run it",
            "promotion": "private pressure and A/B evidence diagnose the path to the floor; broad public score changes only after an approved bounded calibration",
        },
        "next_actions": next_actions(
            floor_cleared,
            clusters,
            semantic_lift_ready=semantic_lift_ready,
            semantic_family_audit=semantic_family_audit,
            patched_learned_token_only=patched_learned_token_only,
        ),
        "external_inference_calls": 0,
    }


def next_actions(
    floor_cleared: bool,
    clusters: list[dict[str, Any]],
    *,
    semantic_lift_ready: bool = False,
    semantic_family_audit: dict[str, Any] | None = None,
    patched_learned_token_only: bool = False,
) -> list[str]:
    if floor_cleared:
        return [
            "Refresh candidate_promotion_gate and coherence_delirium_gate before any model growth or promotion.",
        ]
    actions = []
    if not patched_learned_token_only:
        actions.append(
            "Fix the learned-token decoder evidence path first: promotion-facing repair candidates must be grammar-masked learned-token candidates, not templates or prompt-program rows."
        )
    if not semantic_lift_ready:
        actions.append(
            "Keep training/model growth blocked and patch the private semantic repair loop until the same-seed patched arm passes more private behavioral tests than baseline."
        )
    family_audit = semantic_family_audit if isinstance(semantic_family_audit, dict) else {}
    for row in family_audit.get("regressed_families", []) or []:
        if not isinstance(row, dict):
            continue
        actions.append(
            f"Repair regressed private semantic family `{row.get('family')}` via `{row.get('private_focus')}` before using this recovery as promotion-ready evidence."
        )
    for row in family_audit.get("flat_families", []) or []:
        if not isinstance(row, dict):
            continue
        actions.append(
            f"Add private learned-token pressure for flat semantic family `{row.get('family')}` via `{row.get('private_focus')}`; aggregate lift is not enough."
        )
    if semantic_lift_ready and patched_learned_token_only:
        actions.extend(
            [
                "Use the generated private pressure rows in the next train-once private closure; keep public calibration locked.",
                "After the next private closure, rerun decoder_v2_private_ablation_gate and private_public_transfer_proof.",
            ]
        )
    for cluster in clusters:
        card = cluster.get("card_id")
        top_residuals = cluster.get("top_residuals") if isinstance(cluster.get("top_residuals"), list) else []
        top = top_residuals[0] if top_residuals and isinstance(top_residuals[0], dict) else {}
        actions.append(
            f"Patch private {top.get('private_focus', SEMANTIC_BROAD_FLOOR_FOCUS)} pressure for {card}: needs {cluster.get('additional_passes_needed_for_floor')} more public-calibration passes to clear its floor."
        )
    return actions


def semantic_family_delta_audit(ablation_delta: dict[str, Any]) -> dict[str, Any]:
    families = ablation_delta.get("semantic_task_family_deltas")
    if not isinstance(families, dict):
        return {
            "present": False,
            "rows": [],
            "improved_families": [],
            "flat_families": [],
            "regressed_families": [],
            "all_target_families_positive": False,
            "all_target_families_non_regressing": False,
        }
    rows: list[dict[str, Any]] = []
    for family, metrics in sorted(families.items()):
        metric_row = metrics if isinstance(metrics, dict) else {}
        count_delta = int(safe_float(metric_row.get("semantic_passed_task_count_delta"), 0.0))
        rate_delta = safe_float(metric_row.get("semantic_passed_task_rate_delta"), 0.0)
        tested = int(safe_float(metric_row.get("semantic_tested_task_count"), 0.0))
        if count_delta < 0 or rate_delta < 0.0:
            status = "regressed"
        elif count_delta > 0 or rate_delta > 0.0:
            status = "improved"
        else:
            status = "flat"
        rows.append(
            {
                "family": str(family),
                "status": status,
                "semantic_passed_task_count_delta": count_delta,
                "semantic_passed_task_rate_delta": rate_delta,
                "semantic_tested_task_count": tested,
                "private_focus": primary_focus_for_residual(str(family), None),
                "private_focuses": focuses_for_residual(str(family), None),
                "public_tests_used": False,
                "public_solutions_used": False,
                "repair_rule": "private learned-token repair only; no public prompts/tests/solutions/templates/fallback returns",
            }
        )
    improved = [row for row in rows if row["status"] == "improved"]
    flat = [row for row in rows if row["status"] == "flat"]
    regressed = [row for row in rows if row["status"] == "regressed"]
    return {
        "present": True,
        "family_count": len(rows),
        "rows": rows,
        "improved_families": improved,
        "flat_families": flat,
        "regressed_families": regressed,
        "improved_family_count": len(improved),
        "flat_family_count": len(flat),
        "regressed_family_count": len(regressed),
        "all_target_families_positive": bool(rows) and not flat and not regressed,
        "all_target_families_non_regressing": bool(rows) and not regressed,
        "score_semantics": "private same-seed semantic behavioral-test deltas only; public calibration remains locked",
    }


def calibration_residual_type_counts(report: dict[str, Any], residual_report: dict[str, Any] | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    residuals = report.get("residuals") if isinstance(report.get("residuals"), list) else []
    for item in residuals:
        if not isinstance(item, dict):
            continue
        card = str(item.get("card_id") or "")
        if card not in WEAK_CARD_TARGETS:
            continue
        family = str(item.get("type") or item.get("family") or "").strip()
        if not family:
            continue
        counts[family] += 1
    adjustment = calibration_adapter_adjustment(residual_report or {})
    if adjustment["eligible_candidate_available_tasks"] > 0 and counts.get("local_code_generation_adapter_needed", 0) > 0:
        counts["local_code_generation_adapter_needed"] = min(
            counts["local_code_generation_adapter_needed"],
            adjustment["true_remaining_no_admissible_tasks"],
        )
    return counts


def select_calibration_report(spec: str) -> Path:
    if str(spec or "").strip().lower() not in {"", "auto", "latest"}:
        return resolve(spec)
    candidates = sorted(REPORTS.glob("real_code_benchmark_graduation*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if "skipped" in path.name.lower():
            continue
        report = read_json(path, {})
        if clean_weak_card_calibration(report):
            return path
    fallback = REPORTS / "real_code_benchmark_graduation_private_pressure_private_recovery_broad_floor_v2_public_calibration.json"
    return fallback if fallback.exists() else resolve("reports/real_code_benchmark_graduation_broad_floor_public4card_calibration_v3.json")


def clean_weak_card_calibration(report: dict[str, Any]) -> bool:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    public_tasks = int(summary.get("public_task_count") or summary.get("total_case_count") or 0)
    if public_tasks < 32:
        return False
    if int(summary.get("template_like_candidate_count") or 0) > 0:
        return False
    if int(summary.get("loop_closure_candidate_count") or 0) > 0:
        return False
    if bool(summary.get("student_candidate_provenance_valid")) is False:
        return False
    if bool(summary.get("student_candidate_benchmark_integrity_valid")) is False:
        return False
    suites = report.get("suites") if isinstance(report.get("suites"), list) else []
    card_ids = {str(suite.get("card_id") or "") for suite in suites if isinstance(suite, dict)}
    weak_overlap = len(card_ids.intersection(WEAK_CARD_TARGETS))
    return weak_overlap >= 2 or public_tasks >= 96


def primary_focus_for_residual(family: str, card: str | None = None) -> str:
    focuses = focuses_for_residual(family, card)
    return focuses[0] if focuses else SEMANTIC_BROAD_FLOOR_FOCUS


def focuses_for_residual(family: str, card: str | None = None) -> list[str]:
    focuses = list(RESIDUAL_TO_FOCUSES.get(family, [RESIDUAL_TO_FOCUS.get(family, SEMANTIC_BROAD_FLOOR_FOCUS)]))
    if card == "source_livecodebench" and family in {"edge_case", "type_handling", "local_code_generation_adapter_needed"}:
        focuses.insert(0, INTENDED_BEHAVIOR_FOCUS)
    return list(dict.fromkeys(focuses))


def focus_row_budget(focus: str, focus_counts: Counter[str], rows_per_focus: int) -> int:
    base = max(8, int(rows_per_focus))
    if not focus_counts:
        return base
    count = int(focus_counts.get(focus, 0) or 0)
    max_count = max(1, max(int(value or 0) for value in focus_counts.values()))
    if count <= 0:
        return max(24, base // 2)
    scale = count / max_count
    return max(24, int(round(base * (0.5 + 1.5 * scale))))


def calibration_adapter_adjustment(residual_report: dict[str, Any]) -> dict[str, Any]:
    summary = residual_report.get("summary") if isinstance(residual_report.get("summary"), dict) else {}
    adjusted = summary.get("adapter_adjusted_no_admissible") if isinstance(summary.get("adapter_adjusted_no_admissible"), dict) else {}
    return {
        "total_no_admissible_residual_tasks": int(adjusted.get("total_no_admissible_residual_tasks") or 0),
        "eligible_candidate_available_tasks": int(adjusted.get("eligible_candidate_available_tasks") or 0),
        "true_remaining_no_admissible_tasks": int(adjusted.get("true_remaining_no_admissible_tasks") or adjusted.get("total_no_admissible_residual_tasks") or 0),
        "score_semantics": "sanitized adapter-adjusted routing only; public calibration score is unchanged",
    }


def classify_next_blocker(
    matrix_residuals: Counter[str],
    calibration_residuals: Counter[str],
    public_rate: float,
) -> dict[str, Any]:
    combined = Counter(matrix_residuals)
    combined.update(calibration_residuals)
    runtime_count = (
        combined.get("runtime_load_failure", 0)
        + combined.get("external_dependency_missing", 0)
        + combined.get("local_code_generation_adapter_needed", 0)
    )
    named_counts = {
        "type/return shape": combined.get("type_handling", 0),
        "algorithm choice": combined.get("algorithm_choice", 0),
        "edge contract": combined.get("edge_case", 0),
        "adapter/runtime dependency handling": runtime_count,
        "benchmark transfer": 1 if not combined and public_rate < PUBLIC_FLOOR else 0,
        "coherence": 0,
    }
    primary = max(named_counts.items(), key=lambda item: (item[1], item[0]))[0]
    if calibration_residuals.get("runtime_load_failure", 0) > 0:
        primary = "adapter/runtime dependency handling"
    return {
        "primary": primary,
        "counts": named_counts,
        "basis": {
            "matrix_residual_family_counts": dict(matrix_residuals.most_common()),
            "fresh_calibration_residual_family_counts": dict(calibration_residuals.most_common()),
            "public_floor": PUBLIC_FLOOR,
            "public_rate": public_rate,
        },
        "public_tests_used_for_training": False,
        "public_solutions_used_for_training": False,
        "score_semantics": "routing diagnosis only; private rows are generated from private curricula and aggregate residual labels, not public tests or answers",
    }


def public_leak_count(rows: list[dict[str, Any]]) -> int:
    hits = 0
    for row in rows:
        text = json.dumps(row, sort_keys=True).lower()
        if '"public_tests_included": true' in text or '"public_benchmark_solutions_included": true' in text:
            hits += 1
        if '"public_prompts_used": true' in text or '"public_solutions_used": true' in text:
            hits += 1
    return hits


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Broad Public Code Transfer Floor Recovery",
        "",
        f"- State: `{payload.get('trigger_state')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Broad public pass rate: `{summary.get('broad_public_pass_rate')}` / `{summary.get('public_floor')}`",
        f"- Remaining gap explained: `{summary.get('remaining_gap_explained')}`",
        f"- Private pressure rows: `{summary.get('private_pressure_row_count')}`",
        f"- Next blocker: `{get_path(summary, ['next_blocker', 'primary'], 'unknown')}`",
        f"- Same-seed ablation: `{summary.get('same_seed_ablation_status')}`",
        f"- Same-seed semantic lift: `{summary.get('same_seed_private_semantic_lift')}`",
        f"- Semantic family regressions: `{get_path(summary, ['same_seed_semantic_family_delta_audit', 'regressed_family_count'], 0)}`",
        f"- Semantic family flat rows: `{get_path(summary, ['same_seed_semantic_family_delta_audit', 'flat_family_count'], 0)}`",
        f"- Public calibration run: `{summary.get('public_calibration_run')}`",
        "",
        "## Weak Cards",
    ]
    for cluster in payload.get("residual_clusters", []):
        if not isinstance(cluster, dict):
            continue
        residuals = ", ".join(
            f"{item.get('family')}:{item.get('count')}" for item in cluster.get("top_residuals", [])[:4] if isinstance(item, dict)
        )
        lines.append(
            f"- `{cluster.get('card_id')}` pass `{cluster.get('pass_rate')}` needs `+{cluster.get('additional_passes_needed_for_floor')}`: {residuals}"
        )
    family_audit = summary.get("same_seed_semantic_family_delta_audit") if isinstance(summary.get("same_seed_semantic_family_delta_audit"), dict) else {}
    if family_audit:
        lines.extend(["", "## Same-Seed Semantic Family Deltas"])
        for row in family_audit.get("rows", []) or []:
            if isinstance(row, dict):
                lines.append(
                    f"- `{row.get('family')}` `{row.get('status')}`: count delta `{row.get('semantic_passed_task_count_delta')}`, rate delta `{row.get('semantic_passed_task_rate_delta')}`, focus `{row.get('private_focus')}`"
                )
    lines.extend(["", "## Gates"])
    for item in payload.get("gates", []):
        if isinstance(item, dict):
            lines.append(f"- `{item.get('gate')}`: `{'PASS' if item.get('passed') else 'FAIL'}`")
    lines.extend(["", "## Next Actions"])
    lines.extend(f"- {item}" for item in payload.get("next_actions", []))
    return "\n".join(lines) + "\n"


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: float) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_path(payload: Any, path: list[str], default: Any = None) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


if __name__ == "__main__":
    raise SystemExit(main())
