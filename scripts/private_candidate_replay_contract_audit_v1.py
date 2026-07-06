#!/usr/bin/env python3
"""Private candidate replay contract audit.

This audit verifies the accounting contract between a private task manifest,
candidate manifest, and execution verifier. It does not train, generate new
candidates, run public calibration, call a teacher, or embed candidate code in
the output report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from private_full_body_candidate_admissibility_gate_v1 import (  # noqa: E402
    fallback_return_candidate,
    normalize_candidate,
    private_semantic_candidate_eligible,
    unconditional_constant_return_candidate,
)
from candidate_integrity import recompute_candidate_integrity  # noqa: E402
from real_code_benchmark_runtime import (  # noqa: E402
    build_task_verification_context,
    run_candidate as run_runtime_candidate,
)
from theseus_archive_resolver import read_jsonl_follow_pointer  # noqa: E402


DEFAULT_HELDOUT = (
    ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_eval"
    / "post_v4_seed23_5x32_private_residual_repair_v3_heldout_code_lm_tasks.jsonl"
)
DEFAULT_CANDIDATES = (
    ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_post_v4_heldout_current_release.jsonl"
)
DEFAULT_OUT = ROOT / "reports" / "private_candidate_replay_contract_audit_v1.json"
DEFAULT_MD = ROOT / "reports" / "private_candidate_replay_contract_audit_v1.md"
DEFAULT_REPLAY_DIR = ROOT / "runtime" / "candidate_replay_contract_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default=rel(DEFAULT_HELDOUT))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--trace", default="")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--max-candidates-per-task", type=int, default=0, help="0 means replay all eligible candidates.")
    parser.add_argument("--family-filter", default="", help="Optional recomputed candidate family to replay.")
    parser.add_argument(
        "--candidate-task-scope",
        choices=["all", "present"],
        default="all",
        help="When set to present, audit only heldout tasks that have candidate manifest rows before applying --task-limit.",
    )
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--replay-dir", default=rel(DEFAULT_REPLAY_DIR))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    heldout_path = resolve(args.heldout)
    candidate_path = resolve(args.candidates)
    trace_path = resolve(args.trace) if args.trace else None
    replay_dir = resolve(args.replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(heldout_path)
    total_before_limit = len(tasks)
    raw_candidates = read_jsonl(candidate_path)
    candidates = []
    candidate_family_counts: Counter[str] = Counter()
    integrity_mismatch_counts: Counter[str] = Counter()
    for index, row in enumerate(raw_candidates):
        candidate = normalize_candidate(row)
        integrity = candidate.get("candidate_integrity") if isinstance(candidate.get("candidate_integrity"), dict) else recompute_candidate_integrity(candidate)
        candidate["candidate_integrity"] = integrity
        candidate["recomputed_candidate_family"] = integrity.get("recomputed_candidate_family") or "unknown"
        candidate["candidate_integrity_verified"] = integrity_verified(integrity)
        candidate["candidate_integrity_mismatches"] = integrity.get("integrity_mismatches") or []
        candidate_family_counts[str(candidate["recomputed_candidate_family"])] += 1
        for mismatch in candidate["candidate_integrity_mismatches"]:
            integrity_mismatch_counts[str(mismatch)] += 1
        candidate["_manifest_index"] = index
        candidate["_candidate_id"] = candidate_id(candidate, index)
        candidates.append(candidate)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        task_id = str(candidate.get("task_id") or "")
        if task_id:
            by_task[task_id].append(candidate)
    if str(args.candidate_task_scope or "all") == "present":
        candidate_task_ids = set(by_task)
        tasks = [task for task in tasks if str(task.get("task_id") or "") in candidate_task_ids]
    if int(args.task_limit) > 0:
        tasks = tasks[: int(args.task_limit)]

    trace_summary = summarize_trace(trace_path) if trace_path else {"available": False}
    task_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_selected_pass: Counter[str] = Counter()
    category_oracle_pass: Counter[str] = Counter()
    eligible_family_counts: Counter[str] = Counter()
    replayed_family_counts: Counter[str] = Counter()
    integrity_verified_family_counts: Counter[str] = Counter()
    functional_promotion_family_counts: Counter[str] = Counter()

    unexplained_no_candidate = 0
    selected_pass_count = 0
    pass_if_any_count = 0
    functional_promotion_count = 0
    selected_functional_promotion_count = 0
    selected_runtime_loaded = 0
    selected_compile_passed = 0
    selected_lint_passed = 0
    eligible_candidate_count = 0
    replayed_candidate_count = 0
    fallback_count = 0
    constant_return_count = 0
    public_boundary_violations = public_boundary_violations_for(tasks, candidates)

    for task_index, task in enumerate(tasks):
        task_id = str(task.get("task_id") or "")
        category = str(task.get("category") or "unknown")
        category_counts[category] += 1
        task_candidates = by_task.get(task_id, [])
        family_filter = str(args.family_filter or "").strip()
        if family_filter:
            eligible = [
                candidate
                for candidate in task_candidates
                if family_ablation_candidate_eligible(candidate, family_filter)
            ]
        else:
            eligible = [candidate for candidate in task_candidates if private_semantic_candidate_eligible(candidate)]
        for candidate in eligible:
            eligible_family_counts[str(candidate.get("recomputed_candidate_family") or "unknown")] += 1
            if bool(candidate.get("candidate_integrity_verified")):
                integrity_verified_family_counts[str(candidate.get("recomputed_candidate_family") or "unknown")] += 1
        eligible_candidate_count += len(eligible)
        if not task_candidates:
            unexplained_no_candidate += 1
            task_rows.append(task_row(task, [], [], "missing_manifest_candidates", False, False, []))
            continue
        if not eligible:
            task_rows.append(
                task_row(
                    task,
                    task_candidates,
                    [],
                    "no_private_semantic_eligible_candidates",
                    False,
                    False,
                    ineligible_reasons(task_candidates[:5]),
                )
            )
            continue

        replay_limit = int(args.max_candidates_per_task)
        selected_candidates = eligible if replay_limit <= 0 else eligible[:replay_limit]
        selected_result = None
        task_pass_if_any = False
        task_replay_rows = []
        context = build_task_verification_context(replay_dir, task)
        # Private task prompts are natural-language contracts, not executable
        # helper preludes. The public benchmark runtime keeps prompt prelude for
        # HumanEval-style visible helper definitions; replaying private rows
        # through that path must suppress it or the prompt itself becomes a
        # false lint failure before the candidate is tested.
        context["visible_prompt_prelude"] = ""
        for replay_index, candidate in enumerate(selected_candidates):
            result = run_runtime_candidate(
                replay_dir,
                task,
                candidate,
                mode="private_replay_contract",
                attempt_index=(task_index * 10000) + replay_index,
                verification_context=context,
            )
            replayed_candidate_count += 1
            replayed_family_counts[str(candidate.get("recomputed_candidate_family") or "unknown")] += 1
            stage = str(result.get("verification_stage") or "unknown")
            stage_counts[stage] += 1
            row = replay_row(candidate, result, replay_index)
            task_replay_rows.append(row)
            replay_rows.append(row)
            if replay_index == 0:
                selected_result = result
            if bool(result.get("intended_behavior_passed")):
                task_pass_if_any = True
                if bool(candidate.get("candidate_integrity_verified")):
                    functional_promotion_count += 1
                    functional_promotion_family_counts[str(candidate.get("recomputed_candidate_family") or "unknown")] += 1
            if fallback_return_candidate(candidate):
                fallback_count += 1
            if unconditional_constant_return_candidate(candidate):
                constant_return_count += 1

        selected_passed = bool(selected_result and selected_result.get("intended_behavior_passed"))
        if selected_result and bool(selected_result.get("lint_passed")):
            selected_lint_passed += 1
        if selected_result and bool(selected_result.get("compile_passed")):
            selected_compile_passed += 1
        if selected_result and bool(selected_result.get("runtime_loaded")):
            selected_runtime_loaded += 1
        if selected_passed:
            selected_pass_count += 1
            category_selected_pass[category] += 1
            if selected_candidates and bool(selected_candidates[0].get("candidate_integrity_verified")):
                selected_functional_promotion_count += 1
        if task_pass_if_any:
            pass_if_any_count += 1
            category_oracle_pass[category] += 1
        task_rows.append(
            task_row(
                task,
                task_candidates,
                eligible,
                "replayed",
                selected_passed,
                task_pass_if_any,
                task_replay_rows[:3],
            )
        )

    task_count = len(tasks)
    no_unexplained = unexplained_no_candidate == 0
    gates = [
        gate("private_heldout_rows_present", task_count > 0, {"heldout": rel(heldout_path), "rows": task_count}),
        gate("candidate_rows_present", len(candidates) > 0, {"candidates": rel(candidate_path), "rows": len(candidates)}),
        gate("public_boundary_clean", public_boundary_violations == 0, {"violations": public_boundary_violations}),
        gate("no_unexplained_no_candidate", no_unexplained, {"unexplained_no_candidate": unexplained_no_candidate}),
        gate("selected_candidates_compile", selected_compile_passed == task_count, {"selected_compile_passed": selected_compile_passed, "task_count": task_count}),
        gate("selected_candidates_runtime_load", selected_runtime_loaded == task_count, {"selected_runtime_loaded": selected_runtime_loaded, "task_count": task_count}),
        gate("fallback_returns_zero", fallback_count == 0, fallback_count),
        gate("constant_returns_zero", constant_return_count == 0, constant_return_count),
        gate(
            "functional_promotion_requires_integrity_and_behavior",
            functional_promotion_count > 0,
            {
                "functional_promotion_count": functional_promotion_count,
                "replayed_candidate_count": replayed_candidate_count,
                "selected_functional_promotion_count": selected_functional_promotion_count,
                "task_count": task_count,
            },
        ),
        gate("public_calibration_not_run", True, "audit consumes existing private manifests only"),
        gate("external_inference_zero", True, 0),
    ]
    hard_failed = [row for row in gates if not row["passed"]]
    trigger_state = "GREEN" if not hard_failed else "YELLOW" if task_count and len(candidates) else "RED"
    return {
        "policy": "project_theseus_private_candidate_replay_contract_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "task_count": task_count,
            "heldout_total_before_limit": total_before_limit,
            "candidate_task_scope": str(args.candidate_task_scope or "all"),
            "heldout_rows_after_candidate_scope": len(tasks),
            "candidate_row_count": len(candidates),
            "family_filter": str(args.family_filter or ""),
            "candidate_family_counts": dict(sorted(candidate_family_counts.items())),
            "eligible_family_counts": dict(sorted(eligible_family_counts.items())),
            "replayed_family_counts": dict(sorted(replayed_family_counts.items())),
            "integrity_verified_by_family": dict(sorted(integrity_verified_family_counts.items())),
            "functional_promotion_by_family": dict(sorted(functional_promotion_family_counts.items())),
            "candidate_integrity_mismatch_count": sum(integrity_mismatch_counts.values()),
            "candidate_integrity_mismatch_counts": dict(sorted(integrity_mismatch_counts.items())),
            "eligible_candidate_count": eligible_candidate_count,
            "replayed_candidate_count": replayed_candidate_count,
            "integrity_verified_candidate_count": sum(integrity_verified_family_counts.values()),
            "functional_promotion_count": functional_promotion_count,
            "functional_promotion_rate": ratio(functional_promotion_count, replayed_candidate_count),
            "functional_promotion_rate_ci95": wilson_ci(functional_promotion_count, replayed_candidate_count),
            "functional_promotion_fraction": fraction(functional_promotion_count, replayed_candidate_count),
            "selected_functional_promotion_count": selected_functional_promotion_count,
            "selected_functional_promotion_rate": ratio(selected_functional_promotion_count, task_count),
            "selected_functional_promotion_rate_ci95": wilson_ci(selected_functional_promotion_count, task_count),
            "selected_functional_promotion_fraction": fraction(selected_functional_promotion_count, task_count),
            "tasks_with_manifest_candidates": sum(1 for row in task_rows if int(row.get("candidate_count") or 0) > 0),
            "tasks_with_eligible_candidates": sum(1 for row in task_rows if int(row.get("eligible_candidate_count") or 0) > 0),
            "unexplained_no_candidate_count": unexplained_no_candidate,
            "selected_lint_pass_count": selected_lint_passed,
            "selected_compile_pass_count": selected_compile_passed,
            "selected_runtime_load_count": selected_runtime_loaded,
            "selected_intended_behavior_pass_count": selected_pass_count,
            "pass_if_any_count": pass_if_any_count,
            "rate_denominator_task_count": task_count,
            "selected_lint_pass_rate": ratio(selected_lint_passed, task_count),
            "selected_lint_pass_rate_ci95": wilson_ci(selected_lint_passed, task_count),
            "selected_compile_pass_rate": ratio(selected_compile_passed, task_count),
            "selected_compile_pass_rate_ci95": wilson_ci(selected_compile_passed, task_count),
            "selected_runtime_load_rate": ratio(selected_runtime_loaded, task_count),
            "selected_runtime_load_rate_ci95": wilson_ci(selected_runtime_loaded, task_count),
            "selected_intended_behavior_pass_rate": ratio(selected_pass_count, task_count),
            "selected_intended_behavior_pass_rate_ci95": wilson_ci(selected_pass_count, task_count),
            "selected_intended_behavior_pass_fraction": fraction(selected_pass_count, task_count),
            "pass_if_any_rate": ratio(pass_if_any_count, task_count),
            "pass_if_any_rate_ci95": wilson_ci(pass_if_any_count, task_count),
            "pass_if_any_fraction": fraction(pass_if_any_count, task_count),
            "stage_counts": dict(sorted(stage_counts.items())),
            "category_rates": category_rates(category_counts, category_selected_pass, category_oracle_pass),
            "fallback_return_candidate_count": fallback_count,
            "unconditional_constant_return_candidate_count": constant_return_count,
            "public_boundary_violation_count": public_boundary_violations,
            "trace_reconciliation": trace_summary,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "inputs": {
            "heldout": rel(heldout_path),
            "candidates": rel(candidate_path),
            "trace": rel(trace_path) if trace_path else "",
            "task_limit": int(args.task_limit),
            "max_candidates_per_task": int(args.max_candidates_per_task),
            "family_filter": str(args.family_filter or ""),
            "candidate_task_scope": str(args.candidate_task_scope or "all"),
            "replay_dir": rel(replay_dir),
        },
        "gates": gates,
        "task_rows": task_rows,
        "candidate_replay_rows_sample": replay_rows[:200],
        "rules": {
            "private_only": True,
            "public_calibration_run": False,
            "public_training_rows_written": 0,
            "candidate_code_embedded_in_report": False,
            "external_inference_calls": 0,
        },
        "external_inference_calls": 0,
    }


def replay_row(candidate: dict[str, Any], result: dict[str, Any], replay_index: int) -> dict[str, Any]:
    code = str(candidate.get("code") or "")
    manifest_sha = str(candidate.get("candidate_sha256") or "")
    sha = sha256_text(code)
    return {
        "candidate_id": candidate.get("_candidate_id"),
        "task_id_hash": sha256_text(str(candidate.get("task_id") or ""))[:16],
        "source_task_id_hash": sha256_text(str(candidate.get("source_task_id") or ""))[:16],
        "source_family": candidate.get("candidate_generation_mode"),
        "recomputed_candidate_family": candidate.get("recomputed_candidate_family"),
        "candidate_integrity_verified": bool(candidate.get("candidate_integrity_verified")),
        "candidate_integrity_mismatch_count": len(candidate.get("candidate_integrity_mismatches") or []),
        "candidate_source": candidate.get("candidate_source"),
        "rank": int(candidate.get("_manifest_index") or 0),
        "task_local_rank": replay_index,
        "body_hash": sha,
        "manifest_candidate_hash": manifest_sha,
        "loaded": bool(code.strip()),
        "reconstructed": bool(code.strip()),
        "lint_status": "pass" if result.get("lint_passed") else "fail",
        "compile_status": "pass" if result.get("compile_passed") else "fail",
        "import_status": "pass" if result.get("runtime_loaded") else "fail",
        "verifier_status": "pass" if result.get("intended_behavior_passed") else "fail",
        "verification_stage": result.get("verification_stage"),
        "returncode": result.get("returncode"),
        "runtime_ms": result.get("runtime_ms"),
        "fallback_return_candidate": bool(fallback_return_candidate(candidate)),
        "unconditional_constant_return_candidate": bool(unconditional_constant_return_candidate(candidate)),
    }


def task_row(
    task: dict[str, Any],
    candidates: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    status: str,
    selected_passed: bool,
    pass_if_any: bool,
    evidence: list[Any],
) -> dict[str, Any]:
    return {
        "task_id_hash": sha256_text(str(task.get("task_id") or ""))[:16],
        "source_task_id_hash": sha256_text(str(task.get("source_task_id") or ""))[:16],
        "category": task.get("category"),
        "family": task.get("targeted_private_residual_family_v3"),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected_candidate_id": eligible[0].get("_candidate_id") if eligible else "",
        "selected_body_hash": sha256_text(str(eligible[0].get("code") or "")) if eligible else "",
        "status": status,
        "selected_passed": bool(selected_passed),
        "pass_if_any": bool(pass_if_any),
        "evidence": evidence,
    }


def ineligible_reasons(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        integrity = candidate.get("candidate_integrity") if isinstance(candidate.get("candidate_integrity"), dict) else {}
        self_declared = (
            integrity.get("self_declared_flags")
            if isinstance(integrity.get("self_declared_flags"), dict)
            else {}
        )
        rows.append(
            {
                "candidate_id": candidate.get("_candidate_id"),
                "body_hash": sha256_text(str(candidate.get("code") or "")),
                "mode": candidate.get("candidate_generation_mode"),
                "recomputed_candidate_family": candidate.get("recomputed_candidate_family"),
                "candidate_integrity_mismatches": candidate.get("candidate_integrity_mismatches") or [],
                "full_body": truthy(self_declared.get("full_body_token_candidate")),
                "grammar_masked": truthy(self_declared.get("grammar_masked_learned_token_candidate")),
                "guardrail": truthy(candidate.get("deterministic_guardrail_passed")),
                "fallback": fallback_return_candidate(candidate),
                "constant_return": unconditional_constant_return_candidate(candidate),
                "external_inference_calls": int(candidate.get("external_inference_calls") or 0),
            }
        )
    return rows


def family_ablation_candidate_eligible(candidate: dict[str, Any], family_filter: str) -> bool:
    family = str(candidate.get("recomputed_candidate_family") or "unknown")
    if family != family_filter:
        return False
    integrity = candidate.get("candidate_integrity") if isinstance(candidate.get("candidate_integrity"), dict) else {}
    if candidate.get("candidate_integrity_mismatches"):
        return False
    if int(candidate.get("external_inference_calls") or 0) != 0:
        return False
    if truthy(candidate.get("expression_memory_fallback")):
        return False
    if fallback_return_candidate(candidate) or unconditional_constant_return_candidate(candidate):
        return False
    if not truthy(candidate.get("full_body_token_candidate")):
        return False
    if candidate.get("deterministic_guardrail_passed") is False:
        return False
    if candidate.get("decoder_contract_verifier_v1_passed") is False:
        return False
    if family in {"learned_full_body_token", "transformer_hybrid", "symliquid"}:
        return integrity_verified(integrity)
    return True


def summarize_trace(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    rows = read_jsonl(path)
    candidate_rows = [row for row in rows if row.get("event") == "real_code_candidate_test"]
    stages = Counter(str(row.get("verification_stage") or "unknown") for row in candidate_rows)
    return {
        "available": True,
        "path": rel(path),
        "candidate_test_rows": len(candidate_rows),
        "no_candidate_rows": stages.get("no_candidate", 0),
        "stage_counts": dict(sorted(stages.items())),
        "candidate_sha256_rows": sum(1 for row in candidate_rows if row.get("candidate_sha256")),
    }


def public_boundary_violations_for(tasks: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> int:
    count = 0
    for task in tasks:
        if truthy(task.get("public_benchmark")):
            count += 1
        if truthy(task.get("public_tests_included")) or truthy(task.get("public_benchmark_solutions_included")):
            count += 1
    for candidate in candidates:
        if truthy(candidate.get("public_tests_visible_to_generator")):
            count += 1
        if truthy(candidate.get("canonical_solution_seen_by_solver")):
            count += 1
        if int(candidate.get("external_inference_calls") or 0) != 0:
            count += 1
    return count


def candidate_id(candidate: dict[str, Any], index: int) -> str:
    task = str(candidate.get("task_id") or "")
    code = str(candidate.get("code") or "")
    source = str(candidate.get("candidate_generation_mode") or candidate.get("candidate_source") or "")
    return "cand_" + sha256_text(f"{task}\n{source}\n{index}\n{sha256_text(code)}")[:20]


def category_rates(
    counts: Counter[str],
    selected_passes: Counter[str],
    oracle_passes: Counter[str],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "task_count": counts[key],
            "selected_pass_count": selected_passes[key],
            "selected_pass_rate": ratio(selected_passes[key], counts[key]),
            "pass_if_any_count": oracle_passes[key],
            "pass_if_any_rate": ratio(oracle_passes[key], counts[key]),
            "selected_pass_rate_ci95": wilson_ci(selected_passes[key], counts[key]),
            "pass_if_any_rate_ci95": wilson_ci(oracle_passes[key], counts[key]),
            "selected_pass_fraction": fraction(selected_passes[key], counts[key]),
            "pass_if_any_fraction": fraction(oracle_passes[key], counts[key]),
        }
        for key in sorted(counts)
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for row in read_jsonl_follow_pointer(path):
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Private Candidate Replay Contract Audit v1",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Tasks: {summary.get('task_count')}",
        f"- Candidate rows: {summary.get('candidate_row_count')}",
        f"- Family filter: {summary.get('family_filter')}",
        f"- Candidate families: {summary.get('candidate_family_counts')}",
        f"- Eligible families: {summary.get('eligible_family_counts')}",
        f"- Eligible candidates: {summary.get('eligible_candidate_count')}",
        f"- Replayed candidates: {summary.get('replayed_candidate_count')}",
        f"- Unexplained no-candidate: {summary.get('unexplained_no_candidate_count')}",
        f"- Selected compile pass rate: {summary.get('selected_compile_pass_rate')}",
        f"- Selected runtime load rate: {summary.get('selected_runtime_load_rate')}",
        f"- Selected intended-behavior: {summary.get('selected_intended_behavior_pass_fraction')} rate={summary.get('selected_intended_behavior_pass_rate')} ci95={summary.get('selected_intended_behavior_pass_rate_ci95')}",
        f"- Pass-if-any: {summary.get('pass_if_any_fraction')} rate={summary.get('pass_if_any_rate')} ci95={summary.get('pass_if_any_rate_ci95')}",
        f"- Functional promotion: {summary.get('functional_promotion_fraction')} rate={summary.get('functional_promotion_rate')} ci95={summary.get('functional_promotion_rate_ci95')}",
        f"- Fallback return candidates: {summary.get('fallback_return_candidate_count')}",
        f"- Constant return candidates: {summary.get('unconditional_constant_return_candidate_count')}",
        f"- Public boundary violations: {summary.get('public_boundary_violation_count')}",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        status = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {status}: `{row.get('gate')}`")
    return "\n".join(lines) + "\n"


def ratio(num: int, den: int) -> float:
    return round(num / max(1, den), 6)


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


def integrity_verified(integrity: dict[str, Any]) -> bool:
    return bool(integrity.get("integrity_verified", integrity.get("promotion_verified", False)))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path | None) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return str(candidate)


if __name__ == "__main__":
    raise SystemExit(main())
