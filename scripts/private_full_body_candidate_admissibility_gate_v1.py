#!/usr/bin/env python3
"""Private-only full-body candidate admissibility gate.

This gate exists to verify the candidate-generator repair before any future
public calibration is even proposed. It reads private residual heldout rows and
student candidate manifests, checks a private semantic full-body eligibility
shape, and executes only private tests.

Private semantic eligibility is intentionally separate from public benchmark
promotion eligibility. A candidate can prove private learned full-body transfer
without claiming that it is ready to count on a public calibration surface.
"""

from __future__ import annotations

import argparse
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

from private_residual_repair_v3_heldout_score import run_candidate  # noqa: E402
from real_code_benchmark_runtime import (  # noqa: E402
    benchmark_candidate_eligible,
    bogus_return_attribute_body,
    bogus_return_local_callable_body,
    is_template_like_candidate,
    normalize_student_candidate,
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
DEFAULT_CONTROL_CANDIDATES = (
    ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_post_v4_heldout_current_release_sts_off.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default=rel(DEFAULT_HELDOUT))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--control-candidates", default=rel(DEFAULT_CONTROL_CANDIDATES))
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=2)
    parser.add_argument("--out", default="reports/private_full_body_candidate_admissibility_gate_v1.json")
    parser.add_argument("--markdown-out", default="reports/private_full_body_candidate_admissibility_gate_v1.md")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    heldout_path = resolve(args.heldout)
    candidate_path = resolve(args.candidates)
    heldout_rows = read_jsonl(heldout_path)
    heldout_total_before_limit = len(heldout_rows)
    if int(args.task_limit) > 0:
        heldout_rows = heldout_rows[: int(args.task_limit)]
    candidates_raw = read_jsonl(candidate_path)
    candidates = [normalize_candidate(row) for row in candidates_raw]
    control_candidates = read_jsonl(resolve(args.control_candidates)) if args.control_candidates else []

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        task_id = str(candidate.get("task_id") or "")
        if task_id:
            by_task[task_id].append(candidate)

    selected_passes = 0
    oracle_passes = 0
    no_admissible = 0
    tasks_with_candidates = 0
    result_rows = []
    family_counts: Counter[str] = Counter()
    family_oracle_passes: Counter[str] = Counter()
    family_selected_passes: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_oracle_passes: Counter[str] = Counter()
    category_selected_passes: Counter[str] = Counter()

    for row in heldout_rows:
        task_id = str(row.get("task_id") or "")
        family = str(row.get("targeted_private_residual_family_v3") or "unknown")
        category = str(row.get("category") or "unknown")
        family_counts[family] += 1
        category_counts[category] += 1
        task_candidates = by_task.get(task_id, [])
        if task_candidates:
            tasks_with_candidates += 1
        eligible = [candidate for candidate in task_candidates if private_semantic_candidate_eligible(candidate)]
        if not eligible:
            no_admissible += 1
            result_rows.append(task_result(row, len(task_candidates), 0, False, False, []))
            continue

        selected_ok, selected_error = run_candidate(
            row,
            eligible[0],
            timeout_seconds=max(1, int(args.timeout_seconds)),
        )
        if selected_ok:
            selected_passes += 1
            family_selected_passes[family] += 1
            category_selected_passes[category] += 1

        oracle_ok = selected_ok
        sample_errors = [] if selected_ok else [selected_error]
        if not oracle_ok:
            for candidate in eligible[1:]:
                ok, error = run_candidate(
                    row,
                    candidate,
                    timeout_seconds=max(1, int(args.timeout_seconds)),
                )
                if ok:
                    oracle_ok = True
                    break
                if len(sample_errors) < 3:
                    sample_errors.append(error)
        if oracle_ok:
            oracle_passes += 1
            family_oracle_passes[family] += 1
            category_oracle_passes[category] += 1
        result_rows.append(
            task_result(
                row,
                len(task_candidates),
                len(eligible),
                selected_ok,
                oracle_ok,
                sample_errors,
            )
        )

    task_count = len(heldout_rows)
    selected_pass_rate = ratio(selected_passes, task_count)
    oracle_pass_rate = ratio(oracle_passes, task_count)
    no_admissible_rate = ratio(no_admissible, task_count)
    control_summary = summarize_control(heldout_rows, control_candidates, max(1, int(args.timeout_seconds)))
    if control_summary.get("available") and control_summary.get("control_selected_pass_rate") is not None:
        control_summary["sts_delta_selected_pass_rate"] = round(
            selected_pass_rate - float(control_summary["control_selected_pass_rate"]),
            6,
        )
    fallback_return_count = sum(1 for candidate in candidates if fallback_return_candidate(candidate))
    unconditional_constant_return_count = sum(
        1 for candidate in candidates if unconditional_constant_return_candidate(candidate)
    )
    template_like_count = sum(1 for candidate in candidates if is_template_like_candidate(candidate))
    public_leakage_count = public_leakage_row_count(heldout_rows, candidates_raw)
    external_inference_calls = sum(int(candidate.get("external_inference_calls") or 0) for candidate in candidates)
    full_body_count = sum(1 for candidate in candidates if truthy(candidate.get("full_body_token_candidate")))
    learned_count = sum(1 for candidate in candidates if truthy(candidate.get("token_level_code_generation_learned")))
    countable_integrity_count = sum(1 for candidate in candidates if countable_integrity(candidate))
    private_semantic_eligible_count = sum(1 for candidate in candidates if private_semantic_candidate_eligible(candidate))
    promotion_eligible_count = sum(1 for candidate in candidates if public_promotion_candidate_admissible(candidate))

    gates = [
        gate("heldout_rows_present", task_count > 0, task_count),
        gate("candidate_rows_present", len(candidates) > 0, len(candidates)),
        gate("learned_token_candidates_present", learned_count > 0, learned_count),
        gate("full_body_candidates_present", full_body_count > 0, full_body_count),
        gate("private_semantic_learned_candidates_present", private_semantic_eligible_count > 0, private_semantic_eligible_count),
        gate(
            "public_promotion_boundary_reported",
            True,
            {
                "strict_public_promotion_eligible_candidates": promotion_eligible_count,
                "countable_public_integrity_candidates": countable_integrity_count,
            },
        ),
        gate("no_admissible_task_rate_reduced", no_admissible_rate <= 0.03, {"observed": no_admissible_rate, "maximum": 0.03}),
        gate("semantic_pass_if_any_nonzero", oracle_pass_rate > 0.0, {"observed": oracle_pass_rate, "minimum": "> 0.0"}),
        gate("selected_semantic_pass_nonzero", selected_pass_rate > 0.0, {"observed": selected_pass_rate, "minimum": "> 0.0"}),
        gate("fallback_return_candidates_zero", fallback_return_count == 0, fallback_return_count),
        gate(
            "unconditional_constant_return_candidates_zero",
            unconditional_constant_return_count == 0,
            unconditional_constant_return_count,
        ),
        gate("template_like_candidates_zero", template_like_count == 0, template_like_count),
        gate("public_leakage_zero", public_leakage_count == 0, public_leakage_count),
        gate("external_inference_zero", external_inference_calls == 0, external_inference_calls),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW" if len(candidates) > 0 else "RED"
    return {
        "policy": "project_theseus_private_full_body_candidate_admissibility_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "heldout": rel(heldout_path),
            "candidates": rel(candidate_path),
            "control_candidates": rel(resolve(args.control_candidates)) if args.control_candidates else "",
            "timeout_seconds": int(args.timeout_seconds),
            "task_limit": int(args.task_limit),
            "heldout_total_before_limit": heldout_total_before_limit,
        },
        "summary": {
            "candidate_row_count": len(candidates),
            "task_count": task_count,
            "tasks_with_candidates": tasks_with_candidates,
            "learned_token_candidate_count": learned_count,
            "full_body_token_candidate_count": full_body_count,
            "private_semantic_eligible_candidate_count": private_semantic_eligible_count,
            "countable_integrity_candidate_count": countable_integrity_count,
            "benchmark_promotion_eligible_candidate_count": promotion_eligible_count,
            "no_admissible_task_count": no_admissible,
            "no_admissible_task_rate": no_admissible_rate,
            "selected_pass_count": selected_passes,
            "selected_pass_rate": selected_pass_rate,
            "pass_if_any_count": oracle_passes,
            "pass_if_any_rate": oracle_pass_rate,
            "sts_on_vs_matched_sts_off": control_summary,
            "fallback_return_candidate_count": fallback_return_count,
            "unconditional_constant_return_candidate_count": unconditional_constant_return_count,
            "template_like_candidate_count": template_like_count,
            "public_leakage_count": public_leakage_count,
            "external_inference_calls": external_inference_calls,
            "family_rates": rate_table(family_counts, family_selected_passes, family_oracle_passes),
            "category_rates": rate_table(category_counts, category_selected_passes, category_oracle_passes),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "gates": gates,
        "results": result_rows,
        "recommendation": recommendation(selected_pass_rate, oracle_pass_rate, no_admissible_rate, promotion_eligible_count),
        "external_inference_calls": external_inference_calls,
    }


def normalize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_student_candidate(row)
    merged = dict(row)
    merged.update(normalized)
    preserve_truthy_source_flags = [
        "token_level_code_generation_learned",
        "full_body_token_candidate",
        "grammar_masked_learned_token_candidate",
        "structural_action_candidate",
        "private_residual_v3_train_induced_structural_token_stage",
    ]
    for key in preserve_truthy_source_flags:
        if truthy(row.get(key)):
            merged[key] = row.get(key)
    if "benchmark_integrity" in row:
        merged["benchmark_integrity"] = row["benchmark_integrity"]
    return merged


def public_promotion_candidate_admissible(candidate: dict[str, Any]) -> bool:
    return bool(benchmark_candidate_eligible(candidate) and countable_integrity(candidate))


def private_semantic_candidate_eligible(candidate: dict[str, Any]) -> bool:
    mode = str(candidate.get("candidate_generation_mode") or "").lower()
    code = str(candidate.get("code") or "")
    provenance = candidate.get("provenance") if isinstance(candidate.get("provenance"), dict) else {}
    learned_source = bool(
        truthy(candidate.get("token_level_code_generation_learned"))
        or truthy(candidate.get("structural_action_candidate"))
        or truthy(candidate.get("private_residual_v3_train_induced_structural_token_stage"))
        or "structural_token_decoder" in mode
    )
    integrity = candidate.get("benchmark_integrity") if isinstance(candidate.get("benchmark_integrity"), dict) else {}
    adapter_off_source = bool(
        str(candidate.get("candidate_quality_accounting") or "") == "adapter_off_learned_student_candidate"
        or truthy(candidate.get("structural_action_candidate"))
        or truthy(candidate.get("private_residual_v3_train_induced_structural_token_stage"))
        or truthy(candidate.get("benchmark_promotion_eligible"))
        or truthy(integrity.get("may_count_for_public_benchmark_promotion"))
        or "full_body_token_beam" in mode
    )
    diagnostic_adapter = bool(
        truthy(candidate.get("private_residual_v3_semantic_adapter_stage"))
        or "diagnostic_adapter" in mode
        or str(candidate.get("candidate_quality_accounting") or "") == "diagnostic_adapter_candidate"
    )
    forbidden_provenance = [
        "tests_used",
        "canonical_solution_used",
        "heldout_tests_used",
        "heldout_solution_body_used",
        "public_tests_or_solutions_used",
        "teacher_rows_used",
    ]
    provenance_clean = not any(truthy(provenance.get(key)) for key in forbidden_provenance)
    return bool(
        learned_source
        and adapter_off_source
        and not diagnostic_adapter
        and truthy(candidate.get("full_body_token_candidate"))
        and truthy(candidate.get("grammar_masked_learned_token_candidate"))
        and str(candidate.get("candidate_program_scope") or "full_function_body") == "full_function_body"
        and truthy(candidate.get("deterministic_guardrail_passed"))
        and candidate.get("decoder_contract_verifier_v1_passed") is not False
        and not truthy(candidate.get("placeholder_scaffold_body"))
        and not bogus_return_attribute_body(code)
        and not bogus_return_local_callable_body(code)
        and not fallback_return_candidate(candidate)
        and not unconditional_constant_return_candidate(candidate)
        and not truthy(candidate.get("loop_closure_generated"))
        and not is_template_like_candidate(candidate)
        and int(candidate.get("external_inference_calls") or 0) == 0
        and not truthy(candidate.get("public_tests_visible_to_generator"))
        and not truthy(candidate.get("canonical_solution_seen_by_solver"))
        and provenance_clean
    )


def countable_integrity(candidate: dict[str, Any]) -> bool:
    integrity = candidate.get("benchmark_integrity") if isinstance(candidate.get("benchmark_integrity"), dict) else {}
    if truthy(integrity.get("may_count_for_public_benchmark_promotion")):
        return True
    provenance = candidate.get("provenance") if isinstance(candidate.get("provenance"), dict) else {}
    integrity = provenance.get("benchmark_integrity") if isinstance(provenance.get("benchmark_integrity"), dict) else {}
    return truthy(integrity.get("may_count_for_public_benchmark_promotion"))


def fallback_return_candidate(candidate: dict[str, Any]) -> bool:
    mode = str(candidate.get("candidate_generation_mode") or "").lower()
    code = str(candidate.get("code") or "").lower()
    return bool(
        truthy(candidate.get("expression_memory_fallback"))
        or ("fallback" in mode and "fallback_skipped" not in mode)
        or "return none" in code
        or "result = none" in code
    )


def unconditional_constant_return_candidate(candidate: dict[str, Any]) -> bool:
    """Reject bodies that are only a hard-coded default answer."""
    code = str(candidate.get("code") or "")
    body = code_body_lines(code)
    if len(body) != 1:
        return False
    line = body[0].strip()
    if not line.startswith("return "):
        return False
    expr = line.removeprefix("return ").strip()
    return expr in {
        "0",
        "0.0",
        "1",
        "1.0",
        "False",
        "True",
        "None",
        "[]",
        "{}",
        "()",
        "''",
        '""',
    }


def code_body_lines(code: str) -> list[str]:
    lines = []
    for raw in code.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("def "):
            continue
        if stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def public_leakage_row_count(heldout_rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> int:
    count = 0
    for row in heldout_rows:
        if truthy(row.get("public_benchmark")):
            count += 1
        if truthy(row.get("public_tests_included")) or truthy(row.get("public_benchmark_solutions_included")):
            count += 1
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        if truthy(provenance.get("public_prompts_used")):
            count += 1
        if truthy(provenance.get("public_tests_used")) or truthy(provenance.get("public_benchmark_answers_used")):
            count += 1
    for candidate in candidates:
        if truthy(candidate.get("public_tests_visible_to_generator")) or truthy(candidate.get("canonical_solution_seen_by_solver")):
            count += 1
        provenance = candidate.get("provenance") if isinstance(candidate.get("provenance"), dict) else {}
        if truthy(provenance.get("tests_used")) or truthy(provenance.get("canonical_solution_used")):
            count += 1
    return count


def summarize_control(rows: list[dict[str, Any]], candidates: list[dict[str, Any]], timeout_seconds: int) -> dict[str, Any]:
    if not candidates:
        return {
            "available": False,
            "control_task_count": 0,
            "control_selected_pass_count": 0,
            "control_selected_pass_rate": None,
            "sts_delta_selected_pass_rate": None,
        }
    normalized = [normalize_candidate(row) for row in candidates]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in normalized:
        task_id = str(candidate.get("task_id") or "")
        if task_id:
            by_task[task_id].append(candidate)
    passes = 0
    for row in rows:
        task_candidates = [
            candidate
            for candidate in by_task.get(str(row.get("task_id") or ""), [])
            if private_semantic_candidate_eligible(candidate)
        ]
        if not task_candidates:
            continue
        ok, _error = run_candidate(row, task_candidates[0], timeout_seconds=timeout_seconds)
        if ok:
            passes += 1
    task_count = len(rows)
    return {
        "available": True,
        "control_task_count": task_count,
        "control_selected_pass_count": passes,
        "control_selected_pass_rate": ratio(passes, task_count),
        "sts_delta_selected_pass_rate": None,
    }


def task_result(
    row: dict[str, Any],
    candidate_count: int,
    eligible_count: int,
    selected_passed: bool,
    oracle_passed: bool,
    sample_errors: list[str],
) -> dict[str, Any]:
    return {
        "task_hash": sha256_text(str(row.get("task_id") or ""))[:16],
        "category": str(row.get("category") or ""),
        "family": str(row.get("targeted_private_residual_family_v3") or ""),
        "candidate_count": candidate_count,
        "eligible_candidate_count": eligible_count,
        "selected_passed": bool(selected_passed),
        "pass_if_any": bool(oracle_passed),
        "sample_errors": sample_errors[:3],
    }


def rate_table(
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
        }
        for key in sorted(counts)
    }


def recommendation(
    selected_pass_rate: float,
    oracle_pass_rate: float,
    no_admissible_rate: float,
    public_promotion_eligible_count: int,
) -> str:
    if no_admissible_rate > 0.03:
        return "Not ready for public calibration review: full-body admissibility is still missing on too many private heldout tasks."
    if oracle_pass_rate <= 0.0:
        return "Full-body admissibility is repaired, but semantic private repair remains before any public calibration review."
    if selected_pass_rate < oracle_pass_rate:
        return "Admissibility is repaired and at least one candidate can pass; improve selector/ranker before public calibration review."
    if public_promotion_eligible_count <= 0:
        return (
            "Private semantic full-body transfer is repaired, but public calibration remains blocked: "
            "no strict public-promotion candidate manifest is being claimed."
        )
    return "Admissibility and selected private execution are nonzero; prepare an operator review packet before any one-shot public calibration."


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def ratio(num: int, den: int) -> float:
    return round(num / max(1, den), 6)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Private Full-Body Candidate Admissibility Gate v1",
        "",
        f"Trigger: **{report['trigger_state']}**",
        "",
        "## Summary",
        "",
        f"- tasks: {summary['task_count']}",
        f"- candidate rows: {summary['candidate_row_count']}",
        f"- full-body candidates: {summary['full_body_token_candidate_count']}",
        f"- private semantic eligible candidates: {summary['private_semantic_eligible_candidate_count']}",
        f"- countable integrity candidates: {summary['countable_integrity_candidate_count']}",
        f"- strict public-promotion eligible candidates: {summary['benchmark_promotion_eligible_candidate_count']}",
        f"- no-admissible task rate: {summary['no_admissible_task_rate']}",
        f"- selected pass rate: {summary['selected_pass_rate']}",
        f"- pass-if-any rate: {summary['pass_if_any_rate']}",
        f"- fallback return candidates: {summary['fallback_return_candidate_count']}",
        f"- unconditional constant return candidates: {summary['unconditional_constant_return_candidate_count']}",
        f"- template-like candidates: {summary['template_like_candidate_count']}",
        f"- public leakage count: {summary['public_leakage_count']}",
        f"- external inference calls: {summary['external_inference_calls']}",
        "",
        "## Recommendation",
        "",
        str(report["recommendation"]),
        "",
        "## Gates",
        "",
    ]
    for row in report["gates"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {mark}: {row['gate']} - {row['evidence']}")
    lines.append("")
    return "\n".join(lines)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
