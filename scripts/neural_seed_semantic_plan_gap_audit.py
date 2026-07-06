#!/usr/bin/env python3
"""Audit semantic-plan routing gaps for the neural seed token decoder.

This is diagnostic-only. It may use private eval tests and solution bodies to
label residuals, but those labels are never fed back into candidate generation,
training, teacher distillation, public calibration, or promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import classify_failure, run_any, runtime_tmp_dir  # noqa: E402
from neural_seed_code_proposer_comparator import (  # noqa: E402
    deterministic_sample,
    dict_or_empty,
    get_path,
    load_private_rows,
    ratio,
    rel,
)
from neural_seed_token_decoder_comparator import (  # noqa: E402
    DEFAULT_CANDIDATES,
    DEFAULT_CONFIG,
    gap_family,
    return_shape_for_task,
    semantic_plan_from_body,
)


DEFAULT_OUT = ROOT / "reports" / "neural_seed_semantic_plan_gap_audit.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "neural_seed_semantic_plan_gap_audit.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--candidate-manifest", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config))
    candidates = read_jsonl(resolve(args.candidate_manifest))
    seed = args.seed if args.seed is not None else infer_seed(config, candidates)
    report = build_audit(config, candidates, seed=seed, candidate_manifest=args.candidate_manifest, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_audit(
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    candidate_manifest: str,
    started: float,
) -> dict[str, Any]:
    data_cfg = dict_or_empty(config.get("data"))
    budget = dict_or_empty(config.get("matched_budget"))
    eval_rows_all = load_private_rows(resolve(str(data_cfg.get("eval_jsonl") or "")), data_cfg)
    eval_rows = deterministic_sample(eval_rows_all, int(data_cfg.get("max_eval_rows") or 24), seed + 1009)
    by_task = {str(row.get("task_id") or ""): row for row in eval_rows}

    generated = [row for row in candidates if row.get("phase") != "private_baseline"]
    by_arm_task_phase: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_arm_task_phase[
            (
                str(row.get("substrate_arm") or ""),
                str(row.get("task_id") or ""),
                str(row.get("phase") or ""),
            )
        ].append(row)
    for rows in by_arm_task_phase.values():
        rows.sort(key=lambda row: (int(row.get("rank") or 999), -float(row.get("rank_score") or 0.0)))

    task_rows: list[dict[str, Any]] = []
    pass_counts: Counter[tuple[str, str]] = Counter()
    total_counts: Counter[tuple[str, str]] = Counter()
    expected_plan_matches: Counter[tuple[str, str]] = Counter()
    expected_plan_known: Counter[tuple[str, str]] = Counter()
    plan_counts: Counter[tuple[str, str, str]] = Counter()
    terminal_plan_counts: Counter[tuple[str, str, str]] = Counter()
    plan_confusions: Counter[tuple[str, str, str]] = Counter()
    wrong_shape_counts: Counter[tuple[str, str, str]] = Counter()
    family_totals: Counter[tuple[str, str, str]] = Counter()
    family_passes: Counter[tuple[str, str, str]] = Counter()
    family_plan_failures: Counter[tuple[str, str, str, str]] = Counter()
    contract_beam_available: Counter[tuple[str, str]] = Counter()
    contract_beam_selected: Counter[tuple[str, str]] = Counter()
    learned_route_available: Counter[tuple[str, str]] = Counter()
    learned_route_selected: Counter[tuple[str, str]] = Counter()
    learned_route_strategy_available: Counter[tuple[str, str, str]] = Counter()
    learned_route_strategy_selected: Counter[tuple[str, str, str]] = Counter()
    gap_counts: Counter[str] = Counter()
    sts_repairs: Counter[str] = Counter()
    sts_regressions: Counter[str] = Counter()

    old_timeout = os.environ.get("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS")
    os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = str(
        max(1, int(budget.get("private_candidate_timeout_seconds") or 4))
    )
    try:
        with tempfile.TemporaryDirectory(prefix="theseus_semantic_plan_audit_", dir=runtime_tmp_dir()) as tmp:
            root = Path(tmp)
            for task in eval_rows:
                task_id = str(task.get("task_id") or "")
                family = gap_family(task)
                expected_plan = semantic_plan_from_body(str(task.get("solution_body") or ""))
                expected_shape = return_shape_for_task(task)
                task_row: dict[str, Any] = {
                    "task_id": task_id,
                    "entry_point": task.get("entry_point"),
                    "family": family,
                    "expected_plan_diagnostic_only": expected_plan,
                    "expected_return_shape_diagnostic_only": expected_shape,
                    "arms": {},
                }
                for arm in ["symliquid_style", "transformer_control"]:
                    arm_row = {}
                    phase_results = {}
                    for phase in ["private_eval_sts_off", "private_eval"]:
                        rows = by_arm_task_phase.get((arm, task_id, phase), [])
                        result = run_any(root, task, rows, phase=phase)
                        selected = selected_candidate(rows, result)
                        top = rows[0] if rows else {}
                        selected_decode = dict_or_empty(selected.get("body_structure_decode"))
                        top_decode = dict_or_empty(top.get("body_structure_decode"))
                        has_contract_beam = any(bool(row.get("visible_contract_semantic_beam")) for row in rows)
                        has_learned_route = any(bool(row.get("learned_internal_semantic_route")) for row in rows)
                        available_route_strategies = sorted(
                            {
                                strategy
                                for route_row in rows
                                for strategy in [learned_route_strategy(route_row)]
                                if strategy
                            }
                        )
                        selected_contract_beam = bool(selected.get("visible_contract_semantic_beam")) or bool(
                            selected_decode.get("visible_contract_semantic_beam")
                        )
                        selected_learned_route = bool(selected.get("learned_internal_semantic_route")) or bool(
                            selected_decode.get("learned_internal_semantic_route")
                        )
                        selected_route_strategy = learned_route_strategy(selected) if selected_learned_route else ""
                        selected_plan = str(selected_decode.get("semantic_plan") or "")
                        top_plan = str(top_decode.get("semantic_plan") or "")
                        wrong_shape = classify_wrong_answer_shape(
                            task,
                            result,
                            selected,
                            expected_plan=expected_plan,
                            selected_plan=selected_plan,
                            expected_shape=expected_shape,
                        )
                        passed = bool(result.get("passed"))
                        total_counts[(arm, phase)] += 1
                        pass_counts[(arm, phase)] += int(passed)
                        if expected_plan:
                            expected_plan_known[(arm, phase)] += 1
                            expected_plan_matches[(arm, phase)] += int(selected_plan == expected_plan)
                            plan_confusions[(arm, phase, f"{expected_plan}->{selected_plan or 'missing'}")] += 1
                        plan_counts[(arm, phase, top_plan or "missing")] += 1
                        terminal_plan_counts[(arm, phase, selected_plan or "missing")] += 1
                        wrong_shape_counts[(arm, phase, wrong_shape)] += 1
                        contract_beam_available[(arm, phase)] += int(has_contract_beam)
                        contract_beam_selected[(arm, phase)] += int(selected_contract_beam)
                        learned_route_available[(arm, phase)] += int(has_learned_route)
                        learned_route_selected[(arm, phase)] += int(selected_learned_route)
                        for strategy in available_route_strategies:
                            learned_route_strategy_available[(arm, phase, strategy)] += 1
                        if selected_route_strategy:
                            learned_route_strategy_selected[(arm, phase, selected_route_strategy)] += 1
                        family_totals[(arm, phase, family)] += 1
                        family_passes[(arm, phase, family)] += int(passed)
                        if not passed:
                            family_plan_failures[(arm, phase, family, selected_plan or "missing")] += 1
                        phase_results[phase] = passed
                        arm_row[phase] = {
                            "passed": passed,
                            "verification_stage": result.get("verification_stage"),
                            "verification_reward": result.get("verification_reward"),
                            "top_rank": top.get("rank"),
                            "top_plan": top_plan,
                            "selected_rank": selected.get("rank"),
                            "selected_plan": selected_plan,
                            "selected_semantic_plan_supported": bool(selected_decode.get("semantic_plan_supported")),
                            "selected_predicted_return_shape": selected_decode.get("predicted_return_shape"),
                            "selected_visible_contract_semantic_beam": selected_contract_beam,
                            "selected_learned_internal_semantic_route": selected_learned_route,
                            "selected_learned_internal_semantic_route_strategy": selected_route_strategy,
                            "visible_contract_semantic_beam_available": has_contract_beam,
                            "learned_internal_semantic_route_available": has_learned_route,
                            "learned_internal_semantic_route_available_strategies": available_route_strategies,
                            "wrong_answer_shape": wrong_shape,
                            "stderr_tail": str(result.get("stderr") or "")[-360:],
                        }
                    if phase_results.get("private_eval") and not phase_results.get("private_eval_sts_off"):
                        sts_repairs[arm] += 1
                    if phase_results.get("private_eval_sts_off") and not phase_results.get("private_eval"):
                        sts_regressions[arm] += 1
                    task_row["arms"][arm] = arm_row
                sym_pass = bool(get_path(task_row, ["arms", "symliquid_style", "private_eval", "passed"], False))
                tx_pass = bool(get_path(task_row, ["arms", "transformer_control", "private_eval", "passed"], False))
                if sym_pass and tx_pass:
                    status = "both_pass"
                elif sym_pass:
                    status = "symliquid_only_win"
                elif tx_pass:
                    status = "transformer_only_win"
                else:
                    status = "both_fail"
                task_row["gap_status"] = status
                gap_counts[status] += 1
                task_rows.append(task_row)
    finally:
        if old_timeout is None:
            os.environ.pop("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS", None)
        else:
            os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = old_timeout

    by_arm = build_arm_summary(
        pass_counts,
        total_counts,
        expected_plan_matches,
        expected_plan_known,
        plan_counts,
        terminal_plan_counts,
        plan_confusions,
        wrong_shape_counts,
        family_totals,
        family_passes,
        family_plan_failures,
        contract_beam_available,
        contract_beam_selected,
        learned_route_available,
        learned_route_selected,
        learned_route_strategy_available,
        learned_route_strategy_selected,
    )
    bottleneck = diagnose_bottleneck(by_arm, gap_counts)
    gates = build_gates(config, candidates, generated, eval_rows, task_rows)
    trigger = "GREEN" if all(row["passed"] for row in gates if row["severity"] == "hard") else "RED"
    if trigger == "GREEN" and not bottleneck.get("symliquid_gap_closed"):
        trigger = "YELLOW"
    report = {
        "policy": "project_theseus_neural_seed_semantic_plan_gap_audit_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "config": rel(resolve(str(config.get("_path", "")))) if config.get("_path") else "",
        "candidate_manifest": rel(resolve(candidate_manifest)),
        "seed": seed,
        "summary": {
            "eval_rows": len(eval_rows),
            "candidate_rows": len(candidates),
            "generated_candidate_rows": len(generated),
            "gap_counts": dict(gap_counts),
            "sts_repairs": dict(sts_repairs),
            "sts_regressions": dict(sts_regressions),
            "symliquid_private_eval_pass_rate": get_path(by_arm, ["symliquid_style", "by_phase", "private_eval", "pass_rate"]),
            "transformer_private_eval_pass_rate": get_path(by_arm, ["transformer_control", "by_phase", "private_eval", "pass_rate"]),
            "symliquid_private_eval_plan_match_rate": get_path(by_arm, ["symliquid_style", "by_phase", "private_eval", "expected_plan_match_rate"]),
            "transformer_private_eval_plan_match_rate": get_path(by_arm, ["transformer_control", "by_phase", "private_eval", "expected_plan_match_rate"]),
            "bottleneck": bottleneck,
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "model_promotion_allowed": False,
        },
        "by_arm": by_arm,
        "task_gap_examples": {
            "both_pass": first_examples(task_rows, "both_pass"),
            "both_fail": first_examples(task_rows, "both_fail"),
            "transformer_only_win": first_examples(task_rows, "transformer_only_win"),
            "symliquid_only_win": first_examples(task_rows, "symliquid_only_win"),
        },
        "task_rows": task_rows,
        "gates": gates,
        "score_semantics": (
            "Private semantic-plan diagnostic only. Private eval tests and solution bodies are used only to label "
            "wrong-answer residuals and expected-plan confusions after generation. They are not candidate-generation "
            "features, training targets, public calibration evidence, teacher data, distillation data, or promotion evidence."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    return report


def selected_candidate(candidates: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    traces = result.get("attempt_traces") if isinstance(result.get("attempt_traces"), list) else []
    selected_index = 0
    if traces:
        passed = next((row for row in traces if row.get("passed")), None)
        selected_index = int((passed or traces[-1]).get("attempt_index") or 1) - 1
    selected_index = max(0, min(selected_index, len(candidates) - 1))
    return candidates[selected_index] if candidates else {}


def learned_route_strategy(candidate: dict[str, Any]) -> str:
    route = dict_or_empty(candidate.get("learned_internal_semantic_route"))
    strategy = str(route.get("strategy") or "").strip()
    if strategy:
        return strategy
    provenance = dict_or_empty(candidate.get("provenance"))
    if provenance.get("semantic_route_source") == "learned_internal_semantic_route":
        return "unknown_learned_internal_route"
    return ""


def classify_wrong_answer_shape(
    task: dict[str, Any],
    result: dict[str, Any],
    candidate: dict[str, Any],
    *,
    expected_plan: str,
    selected_plan: str,
    expected_shape: str,
) -> str:
    if result.get("passed"):
        return "passed"
    if not candidate:
        return "no_candidate"
    decode = dict_or_empty(candidate.get("body_structure_decode"))
    repair = dict_or_empty(candidate.get("grammar_repair"))
    if repair.get("fallback_return_used") or decode.get("fallback_return_used"):
        return "fallback_return_policy_violation"
    if decode and not decode.get("semantic_plan_supported"):
        return "unsupported_semantic_plan"
    predicted_shape = str(decode.get("predicted_return_shape") or "").lower()
    if expected_shape and predicted_shape and predicted_shape != expected_shape:
        return f"return_shape_mismatch:{predicted_shape}->{expected_shape}"
    if expected_plan and selected_plan and selected_plan != expected_plan:
        return f"plan_mismatch:{expected_plan}->{selected_plan}"
    stage = str(result.get("verification_stage") or "")
    stderr = str(result.get("stderr") or "")
    failure = classify_failure(stderr)
    if stage and stage != "runtime_loaded":
        return f"{stage}:{failure}"
    return failure or "wrong_answer"


def build_arm_summary(
    pass_counts: Counter[tuple[str, str]],
    total_counts: Counter[tuple[str, str]],
    expected_plan_matches: Counter[tuple[str, str]],
    expected_plan_known: Counter[tuple[str, str]],
    plan_counts: Counter[tuple[str, str, str]],
    terminal_plan_counts: Counter[tuple[str, str, str]],
    plan_confusions: Counter[tuple[str, str, str]],
    wrong_shape_counts: Counter[tuple[str, str, str]],
    family_totals: Counter[tuple[str, str, str]],
    family_passes: Counter[tuple[str, str, str]],
    family_plan_failures: Counter[tuple[str, str, str, str]],
    contract_beam_available: Counter[tuple[str, str]],
    contract_beam_selected: Counter[tuple[str, str]],
    learned_route_available: Counter[tuple[str, str]],
    learned_route_selected: Counter[tuple[str, str]],
    learned_route_strategy_available: Counter[tuple[str, str, str]],
    learned_route_strategy_selected: Counter[tuple[str, str, str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ["symliquid_style", "transformer_control"]:
        arm_row = {"by_phase": {}}
        for phase in ["private_eval_sts_off", "private_eval"]:
            total = total_counts[(arm, phase)]
            families = {}
            for (_arm, _phase, family), count in family_totals.items():
                if _arm != arm or _phase != phase:
                    continue
                families[family] = {
                    "task_count": count,
                    "pass_rate": ratio(family_passes[(arm, phase, family)], count),
                }
            family_failures = {
                f"{family}:{plan}": count
                for (_arm, _phase, family, plan), count in family_plan_failures.items()
                if _arm == arm and _phase == phase
            }
            arm_row["by_phase"][phase] = {
                "task_count": total,
                "pass_count": pass_counts[(arm, phase)],
                "pass_rate": ratio(pass_counts[(arm, phase)], total),
                "expected_plan_known_count": expected_plan_known[(arm, phase)],
                "expected_plan_match_count": expected_plan_matches[(arm, phase)],
                "expected_plan_match_rate": ratio(expected_plan_matches[(arm, phase)], expected_plan_known[(arm, phase)]),
                "top_plan_counts": counter_slice(plan_counts, arm, phase),
                "selected_plan_counts": counter_slice(terminal_plan_counts, arm, phase),
                "plan_confusion_counts": counter_slice(plan_confusions, arm, phase, limit=24),
                "wrong_answer_shape_counts": counter_slice(wrong_shape_counts, arm, phase, limit=24),
                "visible_contract_semantic_beam_available_count": contract_beam_available[(arm, phase)],
                "visible_contract_semantic_beam_available_rate": ratio(contract_beam_available[(arm, phase)], total),
                "visible_contract_semantic_beam_selected_count": contract_beam_selected[(arm, phase)],
                "visible_contract_semantic_beam_selected_rate": ratio(contract_beam_selected[(arm, phase)], total),
                "learned_internal_semantic_route_available_count": learned_route_available[(arm, phase)],
                "learned_internal_semantic_route_available_rate": ratio(learned_route_available[(arm, phase)], total),
                "learned_internal_semantic_route_selected_count": learned_route_selected[(arm, phase)],
                "learned_internal_semantic_route_selected_rate": ratio(learned_route_selected[(arm, phase)], total),
                "learned_internal_semantic_route_strategy_available_rates": strategy_rate_map(
                    learned_route_strategy_available, arm, phase, total
                ),
                "learned_internal_semantic_route_strategy_selected_rates": strategy_rate_map(
                    learned_route_strategy_selected, arm, phase, total
                ),
                "family_pass_rates": families,
                "family_plan_failure_counts": dict(sorted(family_failures.items(), key=lambda item: (-item[1], item[0]))[:24]),
            }
        out[arm] = arm_row
    return out


def strategy_rate_map(counter: Counter[tuple[str, str, str]], arm: str, phase: str, total: int) -> dict[str, float | None]:
    rows = {
        key: ratio(count, total)
        for (_arm, _phase, key), count in counter.items()
        if _arm == arm and _phase == phase
    }
    return dict(sorted(rows.items()))


def counter_slice(counter: Counter[tuple[str, str, str]], arm: str, phase: str, *, limit: int = 16) -> dict[str, int]:
    rows = {
        key: count
        for (_arm, _phase, key), count in counter.items()
        if _arm == arm and _phase == phase
    }
    return dict(sorted(rows.items(), key=lambda item: (-item[1], item[0]))[:limit])


def diagnose_bottleneck(by_arm: dict[str, Any], gap_counts: Counter[str]) -> dict[str, Any]:
    sym_pass = float(get_path(by_arm, ["symliquid_style", "by_phase", "private_eval", "pass_rate"], 0.0) or 0.0)
    tx_pass = float(get_path(by_arm, ["transformer_control", "by_phase", "private_eval", "pass_rate"], 0.0) or 0.0)
    sym_plan = float(get_path(by_arm, ["symliquid_style", "by_phase", "private_eval", "expected_plan_match_rate"], 0.0) or 0.0)
    tx_plan = float(get_path(by_arm, ["transformer_control", "by_phase", "private_eval", "expected_plan_match_rate"], 0.0) or 0.0)
    pass_gap = round(sym_pass - tx_pass, 6)
    plan_gap = round(sym_plan - tx_plan, 6)
    if pass_gap >= -0.1:
        label = "symliquid_gap_mostly_closed"
    elif plan_gap < -0.1:
        label = "symliquid_semantic_plan_selection"
    else:
        label = "shared_renderer_or_behavior_semantics_after_plan_selection"
    return {
        "label": label,
        "symliquid_gap_closed": pass_gap >= -0.1,
        "symliquid_minus_transformer_pass_rate": pass_gap,
        "symliquid_minus_transformer_expected_plan_match_rate": plan_gap,
        "gap_counts": dict(gap_counts),
    }


def build_gates(
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        gate("candidate_manifest_loaded", bool(candidates), {"rows": len(candidates)}, "hard"),
        gate("eval_rows_loaded", bool(eval_rows), {"rows": len(eval_rows)}, "hard"),
        gate("task_rows_recorded", len(task_rows) == len(eval_rows), {"task_rows": len(task_rows), "eval_rows": len(eval_rows)}, "hard"),
        gate("semantic_plan_recorded", all(get_path(row, ["body_structure_decode", "semantic_plan"], "") for row in generated), {"generated_rows": len(generated)}, "hard"),
        gate("fallback_return_rate_zero", all(not get_path(row, ["grammar_repair", "fallback_return_used"], False) and not get_path(row, ["body_structure_decode", "fallback_return_used"], False) for row in generated), {"generated_rows": len(generated)}, "hard"),
        gate("eval_tests_not_visible_to_generator", all(not row.get("eval_tests_visible_to_generator") and not row.get("eval_solution_visible_to_generator") for row in candidates), True, "hard"),
        gate("public_data_not_visible_to_generator", all(not row.get("public_tests_visible_to_generator") and not row.get("public_solutions_visible_to_generator") for row in candidates), True, "hard"),
        gate("external_inference_zero", sum(int(row.get("external_inference_calls") or 0) for row in candidates) == 0, 0, "hard"),
        gate("teacher_public_promotion_locked", True, {"teacher_used": False, "public_training_rows": 0, "model_promotion_allowed": False}, "hard"),
    ]


def first_examples(rows: list[dict[str, Any]], status: str, limit: int = 8) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if row.get("gap_status") != status:
            continue
        examples.append(
            {
                "task_id": row.get("task_id"),
                "family": row.get("family"),
                "expected_plan_diagnostic_only": row.get("expected_plan_diagnostic_only"),
                "symliquid": get_path(row, ["arms", "symliquid_style", "private_eval"], {}),
                "transformer": get_path(row, ["arms", "transformer_control", "private_eval"], {}),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def infer_seed(config: dict[str, Any], candidates: list[dict[str, Any]]) -> int:
    for row in candidates:
        seed = get_path(row, ["provenance", "seed"])
        if seed is not None:
            return int(seed)
    seeds = get_path(config, ["matched_budget", "seeds"], [23])
    return int((seeds or [23])[0])


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    bottleneck = dict_or_empty(summary.get("bottleneck"))
    lines = [
        "# Neural Seed Semantic Plan Gap Audit",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- seed: `{report.get('seed')}`",
        f"- eval_rows: `{summary.get('eval_rows')}`",
        f"- candidate_rows: `{summary.get('candidate_rows')}`",
        f"- gap_counts: `{summary.get('gap_counts')}`",
        f"- symliquid_private_eval_pass_rate: `{summary.get('symliquid_private_eval_pass_rate')}`",
        f"- transformer_private_eval_pass_rate: `{summary.get('transformer_private_eval_pass_rate')}`",
        f"- symliquid_private_eval_plan_match_rate: `{summary.get('symliquid_private_eval_plan_match_rate')}`",
        f"- transformer_private_eval_plan_match_rate: `{summary.get('transformer_private_eval_plan_match_rate')}`",
        f"- bottleneck: `{bottleneck.get('label')}`",
        f"- symliquid_minus_transformer_pass_rate: `{bottleneck.get('symliquid_minus_transformer_pass_rate')}`",
        f"- symliquid_minus_transformer_expected_plan_match_rate: `{bottleneck.get('symliquid_minus_transformer_expected_plan_match_rate')}`",
        "",
        "## Wrong-Answer Shapes",
        "",
    ]
    for arm, row in dict_or_empty(report.get("by_arm")).items():
        phase = get_path(row, ["by_phase", "private_eval"], {})
        lines.append(f"### {arm}")
        lines.append(
            f"- visible_contract_semantic_beam_selected_rate: "
            f"`{phase.get('visible_contract_semantic_beam_selected_rate')}`"
        )
        lines.append(
            f"- visible_contract_semantic_beam_available_rate: "
            f"`{phase.get('visible_contract_semantic_beam_available_rate')}`"
        )
        lines.append(
            f"- learned_internal_semantic_route_selected_rate: "
            f"`{phase.get('learned_internal_semantic_route_selected_rate')}`"
        )
        lines.append(
            f"- learned_internal_semantic_route_available_rate: "
            f"`{phase.get('learned_internal_semantic_route_available_rate')}`"
        )
        lines.append(
            f"- learned_internal_semantic_route_strategy_selected_rates: "
            f"`{phase.get('learned_internal_semantic_route_strategy_selected_rates')}`"
        )
        lines.append(
            f"- learned_internal_semantic_route_strategy_available_rates: "
            f"`{phase.get('learned_internal_semantic_route_strategy_available_rates')}`"
        )
        for key, count in dict_or_empty(phase.get("wrong_answer_shape_counts")).items():
            lines.append(f"- `{key}`: `{count}`")
        lines.append("")
    lines.extend(["## Semantics", "", str(report.get("score_semantics", "")), ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
