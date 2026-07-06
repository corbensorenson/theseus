#!/usr/bin/env python3
"""Audit SymLiquid/transformer complementarity across private decoder reports.

This is an evidence reducer only. It reads route-independence ablation reports
and their per-seed semantic-plan audits, then reports whether SymLiquid provides
stable unique wins or bounded union value over the matched transformer control.
It does not train, call a teacher, use public calibration data, or promote.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION = ROOT / "reports" / "neural_seed_token_decoder_route_independence_ablation.json"
DEFAULT_OUT = ROOT / "reports" / "neural_seed_token_decoder_complementarity_audit.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "neural_seed_token_decoder_complementarity_audit.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", default=rel(DEFAULT_ABLATION))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.execute:
        report = run_audit(resolve(args.ablation), started)
    else:
        report = planned_report(args.ablation, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def run_audit(ablation_path: Path, started: float) -> dict[str, Any]:
    ablation = read_json(ablation_path)
    variant_rows = ablation.get("variant_rows") if isinstance(ablation.get("variant_rows"), list) else []
    by_variant: dict[str, Any] = {}
    global_sym_only_families: Counter[str] = Counter()
    global_tx_only_families: Counter[str] = Counter()
    loaded_variant_reports = 0
    loaded_seed_audits = 0
    missing_seed_audits: list[str] = []
    gap_mismatches = 0

    for variant in variant_rows:
        variant_id = str(variant.get("id") or "")
        variant_report_path = resolve(str(variant.get("report") or ""))
        variant_report = read_json(variant_report_path)
        if variant_report:
            loaded_variant_reports += 1
        summary, missing, mismatches = summarize_variant(variant_id, variant, variant_report)
        by_variant[variant_id] = summary
        loaded_seed_audits += summary.get("loaded_seed_audit_count", 0)
        missing_seed_audits.extend(missing)
        gap_mismatches += mismatches
        global_sym_only_families.update(summary.get("stable_symliquid_only_family_counts", {}))
        global_tx_only_families.update(summary.get("stable_transformer_only_family_counts", {}))

    full = by_variant.get("full_learned_beam_off", {})
    recommendation = build_recommendation(ablation, full, by_variant)
    safety = safety_summary(ablation, variant_rows)
    gates = build_gates(
        ablation,
        variant_rows,
        by_variant,
        loaded_variant_reports,
        loaded_seed_audits,
        missing_seed_audits,
        gap_mismatches,
        safety,
        recommendation,
    )
    trigger = "GREEN" if all(row["passed"] for row in gates if row["severity"] == "hard") else "RED"
    return {
        "policy": "project_theseus_private_decoder_complementarity_audit_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "ablation_report": rel(ablation_path),
        "baseline_attribution": ablation.get("attribution"),
        "by_variant": by_variant,
        "overall": {
            "stable_symliquid_only_family_counts": dict(global_sym_only_families.most_common()),
            "stable_transformer_only_family_counts": dict(global_tx_only_families.most_common()),
            "max_union_gain_vs_best_single": max_number(
                row.get("bounded_union_oracle", {}).get("union_gain_vs_best_single") for row in by_variant.values()
            ),
            "recommendation": recommendation,
        },
        "safety": safety,
        "gates": gates,
        "score_semantics": (
            "Private evidence reducer over existing token-decoder ablation reports. Gap status is recomputed from "
            "private verifier pass/fail rows and used only to audit complementarity. No public calibration, teacher "
            "call, distillation, external inference, fallback terminal return, or model promotion occurs."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def summarize_variant(
    variant_id: str,
    variant_row: dict[str, Any],
    variant_report: dict[str, Any],
) -> tuple[dict[str, Any], list[str], int]:
    seed_rows = variant_report.get("seed_rows") if isinstance(variant_report.get("seed_rows"), list) else []
    gap_counts: Counter[str] = Counter()
    family_gap_counts: Counter[str] = Counter()
    sym_strategy_counts: Counter[str] = Counter()
    tx_strategy_counts: Counter[str] = Counter()
    task_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_family: dict[str, str] = {}
    sym_only_examples: list[dict[str, Any]] = []
    tx_only_examples: list[dict[str, Any]] = []
    both_fail_examples: list[dict[str, Any]] = []
    missing_audits: list[str] = []
    gap_mismatches = 0
    loaded_seed_audits = 0

    for seed_row in seed_rows:
        seed = seed_row.get("seed")
        audit_path = resolve(str(seed_row.get("semantic_plan_audit") or ""))
        audit = read_json(audit_path)
        if not audit:
            missing_audits.append(rel(audit_path))
            continue
        loaded_seed_audits += 1
        task_rows = audit.get("task_rows") if isinstance(audit.get("task_rows"), list) else []
        for task in task_rows:
            task_id = str(task.get("task_id") or "")
            family = str(task.get("family") or "unknown")
            expected_plan = task.get("expected_plan_diagnostic_only")
            sym_eval = dict_or_empty(get_path(task, ["arms", "symliquid_style", "private_eval"], {}))
            tx_eval = dict_or_empty(get_path(task, ["arms", "transformer_control", "private_eval"], {}))
            status = computed_gap_status(bool(sym_eval.get("passed")), bool(tx_eval.get("passed")))
            if status != str(task.get("gap_status") or ""):
                gap_mismatches += 1
            gap_counts[status] += 1
            family_gap_counts[f"{status}:{family}"] += 1
            task_status_counts[task_id][status] += 1
            task_family[task_id] = family
            sym_strategy_counts[str(sym_eval.get("selected_learned_internal_semantic_route_strategy") or "none")] += 1
            tx_strategy_counts[str(tx_eval.get("selected_learned_internal_semantic_route_strategy") or "none")] += 1
            example = {
                "seed": seed,
                "task_id": task_id,
                "family": family,
                "expected_plan_diagnostic_only": expected_plan,
                "sym_selected_plan": sym_eval.get("selected_plan"),
                "transformer_selected_plan": tx_eval.get("selected_plan"),
                "sym_strategy": sym_eval.get("selected_learned_internal_semantic_route_strategy") or "none",
                "transformer_strategy": tx_eval.get("selected_learned_internal_semantic_route_strategy") or "none",
            }
            if status == "symliquid_only_win" and len(sym_only_examples) < 12:
                sym_only_examples.append(example)
            elif status == "transformer_only_win" and len(tx_only_examples) < 12:
                tx_only_examples.append(example)
            elif status == "both_fail" and len(both_fail_examples) < 12:
                both_fail_examples.append(example)

    total = sum(gap_counts.values())
    sym_pass = gap_counts["both_pass"] + gap_counts["symliquid_only_win"]
    tx_pass = gap_counts["both_pass"] + gap_counts["transformer_only_win"]
    union_pass = total - gap_counts["both_fail"]
    stable_threshold = stable_task_threshold(loaded_seed_audits)
    stable_tasks = stable_task_summary(task_status_counts, task_family, stable_threshold)
    bounded_union = {
        "task_row_count": total,
        "symliquid_pass_rate": rate(sym_pass, total),
        "transformer_pass_rate": rate(tx_pass, total),
        "union_pass_rate": rate(union_pass, total),
        "best_single_pass_rate": rate(max(sym_pass, tx_pass), total),
        "union_gain_vs_best_single": subtract(rate(union_pass, total), rate(max(sym_pass, tx_pass), total)),
        "union_extra_pass_count_vs_best_single": union_pass - max(sym_pass, tx_pass),
        "score_semantics": "Oracle union only: counts tasks where either matched arm passed. It is not a runtime router.",
    }
    return (
        {
            "label": variant_row.get("label"),
            "report": variant_row.get("report"),
            "trigger_state": variant_row.get("trigger_state"),
            "symliquid_sts_on_mean": variant_row.get("symliquid_sts_on_mean"),
            "transformer_sts_on_mean": variant_row.get("transformer_sts_on_mean"),
            "symliquid_minus_transformer_sts_on_mean": variant_row.get("symliquid_minus_transformer_sts_on_mean"),
            "symliquid_learned_internal_semantic_route_strategy_selected_means": variant_row.get(
                "symliquid_learned_internal_semantic_route_strategy_selected_means"
            ),
            "gap_counts": dict(gap_counts.most_common()),
            "family_gap_counts": dict(family_gap_counts.most_common(32)),
            "symliquid_strategy_counts": dict(sym_strategy_counts.most_common()),
            "transformer_strategy_counts": dict(tx_strategy_counts.most_common()),
            "bounded_union_oracle": bounded_union,
            "loaded_seed_audit_count": loaded_seed_audits,
            "stable_task_threshold": stable_threshold,
            "stable_tasks": stable_tasks,
            "stable_symliquid_only_family_counts": stable_tasks["family_counts_by_status"].get("symliquid_only_win", {}),
            "stable_transformer_only_family_counts": stable_tasks["family_counts_by_status"].get(
                "transformer_only_win", {}
            ),
            "examples": {
                "symliquid_only_win": sym_only_examples,
                "transformer_only_win": tx_only_examples,
                "both_fail": both_fail_examples,
            },
        },
        missing_audits,
        gap_mismatches,
    )


def stable_task_threshold(seed_count: int) -> int:
    if seed_count <= 1:
        return 1
    return max(2, math.ceil(seed_count * 0.6))


def stable_task_summary(
    task_status_counts: dict[str, Counter[str]],
    task_family: dict[str, str],
    threshold: int,
) -> dict[str, Any]:
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_counts_by_status: dict[str, Counter[str]] = defaultdict(Counter)
    for task_id, counts in sorted(task_status_counts.items()):
        status, count = counts.most_common(1)[0]
        if count < threshold:
            continue
        family = task_family.get(task_id, "unknown")
        row = {
            "task_id": task_id,
            "family": family,
            "status": status,
            "count": count,
            "status_counts": dict(counts.most_common()),
        }
        by_status[status].append(row)
        family_counts_by_status[status][family] += 1
    return {
        "by_status": {status: rows[:24] for status, rows in by_status.items()},
        "counts_by_status": {status: len(rows) for status, rows in by_status.items()},
        "family_counts_by_status": {
            status: dict(counts.most_common()) for status, counts in family_counts_by_status.items()
        },
    }


def computed_gap_status(sym_passed: bool, tx_passed: bool) -> str:
    if sym_passed and tx_passed:
        return "both_pass"
    if sym_passed:
        return "symliquid_only_win"
    if tx_passed:
        return "transformer_only_win"
    return "both_fail"


def build_recommendation(ablation: dict[str, Any], full: dict[str, Any], by_variant: dict[str, Any]) -> dict[str, Any]:
    attribution = dict_or_empty(ablation.get("attribution"))
    full_sym = number(full.get("symliquid_sts_on_mean"))
    full_tx = number(full.get("transformer_sts_on_mean"))
    full_union_gain = number(get_path(full, ["bounded_union_oracle", "union_gain_vs_best_single"])) or 0.0
    stable_sym_only = int(get_path(full, ["stable_tasks", "counts_by_status", "symliquid_only_win"], 0) or 0)
    stable_tx_only = int(get_path(full, ["stable_tasks", "counts_by_status", "transformer_only_win"], 0) or 0)
    visible_rate = number(attribution.get("symliquid_full_visible_text_prototype_selected_rate"))
    plan_head = number(attribution.get("symliquid_plan_head_only_mean"))
    no_visible = number(attribution.get("symliquid_no_visible_text_memory_mean"))
    dropout = number(attribution.get("symliquid_route_dropout_half_mean"))
    keep = (
        full_sym is not None
        and full_tx is not None
        and full_sym >= 0.80
        and (full_sym >= full_tx or stable_sym_only > 0 or full_union_gain >= 0.02)
    )
    if keep:
        decision = "keep_symliquid_as_discovery_lane"
        if (
            visible_rate is not None
            and no_visible is not None
            and full_sym is not None
            and visible_rate <= 0.20
            and no_visible >= full_sym - 0.025
        ):
            rationale = (
                "Full-route SymLiquid remains at the survival threshold with matched transformer parity, and the "
                "no-visible variant preserves the same score. The protected discovery lane should continue, but the "
                "remaining evidence pressure is the thin dropout margin and full-route both-fail families rather "
                "than more visible-text displacement."
            )
        else:
            rationale = (
                "Full-route SymLiquid is at or above the survival threshold and has either parity, stable unique wins, "
                "or bounded union value. It should remain protected, but visible-text prototype dependence still needs "
                "to be driven down before claiming substrate independence."
            )
    else:
        decision = "do_not_expand_symliquid_without_new_evidence"
        rationale = (
            "Current evidence does not show enough stable unique value after matched controls. Keep transformer as "
            "the survival baseline and only continue SymLiquid through narrow, auditable discovery experiments."
        )
    return {
        "decision": decision,
        "rationale": rationale,
        "full_symliquid_sts_on_mean": full_sym,
        "full_transformer_sts_on_mean": full_tx,
        "full_union_gain_vs_best_single": round(full_union_gain, 6),
        "stable_full_symliquid_only_task_count": stable_sym_only,
        "stable_full_transformer_only_task_count": stable_tx_only,
        "symliquid_full_visible_text_prototype_selected_rate": visible_rate,
        "symliquid_plan_head_only_mean": plan_head,
        "symliquid_no_visible_text_memory_mean": no_visible,
        "symliquid_route_dropout_half_mean": dropout,
    }


def safety_summary(ablation: dict[str, Any], variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    hard_failures = [
        row for row in ablation.get("gates", []) if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    return {
        "ablation_trigger_state": ablation.get("trigger_state"),
        "ablation_hard_gate_failure_count": len(hard_failures),
        "variant_count": len(variant_rows),
        "fallback_return_rate_zero": all(bool(row.get("fallback_return_rate_zero")) for row in variant_rows),
        "teacher_used": any(bool(row.get("teacher_used")) for row in variant_rows),
        "public_training_rows": sum(int(row.get("public_training_rows") or 0) for row in variant_rows),
        "model_promotion_allowed": any(bool(row.get("model_promotion_allowed")) for row in variant_rows),
        "external_inference_calls": sum(int(row.get("external_inference_calls") or 0) for row in variant_rows),
        "hard_gate_failures": hard_failures,
    }


def build_gates(
    ablation: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    by_variant: dict[str, Any],
    loaded_variant_reports: int,
    loaded_seed_audits: int,
    missing_seed_audits: list[str],
    gap_mismatches: int,
    safety: dict[str, Any],
    recommendation: dict[str, Any],
) -> list[dict[str, Any]]:
    total_task_rows = sum(
        int(get_path(row, ["bounded_union_oracle", "task_row_count"], 0) or 0) for row in by_variant.values()
    )
    return [
        gate("ablation_report_loaded", bool(ablation), {"trigger_state": ablation.get("trigger_state")}, "hard"),
        gate(
            "variant_reports_loaded",
            loaded_variant_reports == len(variant_rows) and loaded_variant_reports > 0,
            {"loaded": loaded_variant_reports, "expected": len(variant_rows)},
            "hard",
        ),
        gate(
            "seed_audits_loaded",
            loaded_seed_audits > 0 and not missing_seed_audits,
            {"loaded": loaded_seed_audits, "missing": missing_seed_audits[:12]},
            "hard",
        ),
        gate("task_rows_loaded", total_task_rows > 0, {"task_rows": total_task_rows}, "hard"),
        gate("gap_status_recomputed_without_mismatch", gap_mismatches == 0, {"mismatches": gap_mismatches}, "hard"),
        gate(
            "fallback_terminal_returns_remain_zero",
            bool(safety.get("fallback_return_rate_zero")),
            safety.get("fallback_return_rate_zero"),
            "hard",
        ),
        gate("external_inference_zero", int(safety.get("external_inference_calls") or 0) == 0, safety, "hard"),
        gate("teacher_not_used", not bool(safety.get("teacher_used")), safety, "hard"),
        gate("public_training_rows_zero", int(safety.get("public_training_rows") or 0) == 0, safety, "hard"),
        gate("model_promotion_locked", not bool(safety.get("model_promotion_allowed")), safety, "hard"),
        gate(
            "recommendation_recorded",
            bool(recommendation.get("decision")),
            {"decision": recommendation.get("decision")},
            "hard",
        ),
    ]


def planned_report(ablation: str, started: float) -> dict[str, Any]:
    return {
        "policy": "project_theseus_private_decoder_complementarity_audit_v0",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "ablation_report": ablation,
        "summary": {
            "execute_required": True,
            "command": "python3 scripts/neural_seed_token_decoder_complementarity_audit.py --execute",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    overall = dict_or_empty(report.get("overall"))
    recommendation = dict_or_empty(overall.get("recommendation"))
    lines = [
        "# Neural Seed Token Decoder Complementarity Audit",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- recommendation: `{recommendation.get('decision')}`",
        f"- full_symliquid_sts_on_mean: `{recommendation.get('full_symliquid_sts_on_mean')}`",
        f"- full_transformer_sts_on_mean: `{recommendation.get('full_transformer_sts_on_mean')}`",
        f"- full_union_gain_vs_best_single: `{recommendation.get('full_union_gain_vs_best_single')}`",
        f"- stable_full_symliquid_only_task_count: `{recommendation.get('stable_full_symliquid_only_task_count')}`",
        f"- stable_full_transformer_only_task_count: `{recommendation.get('stable_full_transformer_only_task_count')}`",
        f"- symliquid_full_visible_text_prototype_selected_rate: "
        f"`{recommendation.get('symliquid_full_visible_text_prototype_selected_rate')}`",
        "",
        "## Rationale",
        "",
        str(recommendation.get("rationale") or ""),
        "",
        "## Variants",
        "",
    ]
    for variant_id, row in dict_or_empty(report.get("by_variant")).items():
        union = dict_or_empty(row.get("bounded_union_oracle"))
        stable = dict_or_empty(row.get("stable_tasks"))
        lines.append(
            f"- `{variant_id}`: sym=`{row.get('symliquid_sts_on_mean')}`, "
            f"tx=`{row.get('transformer_sts_on_mean')}`, "
            f"gap_counts=`{row.get('gap_counts')}`, "
            f"union_gain=`{union.get('union_gain_vs_best_single')}`, "
            f"stable_counts=`{stable.get('counts_by_status')}`"
        )
    lines.extend(["", "## Stable SymLiquid-Only Families", ""])
    for key, value in dict_or_empty(overall.get("stable_symliquid_only_family_counts")).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    lines.extend(["", "## Semantics", "", str(report.get("score_semantics") or ""), ""])
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def max_number(values: Any) -> float | None:
    numbers = [parsed for parsed in (number(value) for value in values) if parsed is not None]
    return round(max(numbers), 6) if numbers else None


def rate(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 6)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
