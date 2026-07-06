#!/usr/bin/env python3
"""Mine private residuals for context-route improvements.

This reducer aligns the current route-independence ablation variants per
seed/task and reports where the full route, no-visible route, and dropout route
diverge. It is evidence only: no training, teacher calls, public calibration,
promotion, or fallback terminal returns are performed here.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION = ROOT / "reports" / "neural_seed_token_decoder_route_independence_ablation.json"
DEFAULT_OUT = ROOT / "reports" / "neural_seed_token_decoder_residual_context_miner.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "neural_seed_token_decoder_residual_context_miner.md"
KEY_VARIANTS = [
    "full_learned_beam_off",
    "route_dropout_half_beam_off",
    "no_visible_text_memory_beam_off",
    "plan_head_only_beam_off",
    "no_internal_beam_off",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", default=rel(DEFAULT_ABLATION))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.execute:
        report = run_miner(resolve(args.ablation), started)
    else:
        report = planned_report(args.ablation, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def run_miner(ablation_path: Path, started: float) -> dict[str, Any]:
    ablation = read_json(ablation_path)
    variants = load_variant_rows(ablation)
    aligned, load_summary = align_task_rows(variants)
    buckets = mine_buckets(aligned)
    safety = safety_summary(ablation, variants)
    recommendation = build_recommendation(ablation, buckets)
    gates = build_gates(ablation, variants, aligned, buckets, safety, recommendation, load_summary)
    trigger = "GREEN" if all(row["passed"] for row in gates if row["severity"] == "hard") else "RED"
    return {
        "policy": "project_theseus_private_residual_context_miner_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "ablation_report": rel(ablation_path),
        "variant_reports": {variant_id: row.get("report") for variant_id, row in variants.items()},
        "baseline_attribution": ablation.get("attribution"),
        "load_summary": load_summary,
        "summary": summarize_buckets(buckets),
        "buckets": buckets,
        "recommendation": recommendation,
        "safety": safety,
        "gates": gates,
        "score_semantics": (
            "Private residual context-route mining over existing route-independence ablation reports. "
            "Private eval tests/solutions are used only through already-written semantic-plan diagnostic audits "
            "to label residuals after generation; they are not training features, teacher data, public evidence, "
            "or promotion evidence."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def load_variant_rows(ablation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = ablation.get("variant_rows") if isinstance(ablation.get("variant_rows"), list) else []
    return {str(row.get("id") or ""): row for row in rows if str(row.get("id") or "") in KEY_VARIANTS}


def align_task_rows(variants: dict[str, dict[str, Any]]) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    aligned: dict[tuple[int, str], dict[str, Any]] = defaultdict(dict)
    audit_paths: list[str] = []
    missing_variant_reports: list[str] = []
    missing_audits: list[str] = []
    loaded_task_rows = 0
    for variant_id, variant in variants.items():
        report_path = resolve(str(variant.get("report") or ""))
        report = read_json(report_path)
        if not report:
            missing_variant_reports.append(rel(report_path))
            continue
        seed_rows = report.get("seed_rows") if isinstance(report.get("seed_rows"), list) else []
        for seed_row in seed_rows:
            seed = int(seed_row.get("seed") or 0)
            audit_path = resolve(str(seed_row.get("semantic_plan_audit") or ""))
            audit = read_json(audit_path)
            if not audit:
                missing_audits.append(rel(audit_path))
                continue
            audit_paths.append(rel(audit_path))
            rows = audit.get("task_rows") if isinstance(audit.get("task_rows"), list) else []
            for task in rows:
                key = (seed, str(task.get("task_id") or ""))
                aligned[key][variant_id] = summarize_task(task, seed=seed, variant_id=variant_id)
                loaded_task_rows += 1
    return aligned, {
        "variant_count": len(variants),
        "aligned_seed_task_count": len(aligned),
        "loaded_task_rows": loaded_task_rows,
        "audit_count": len(audit_paths),
        "missing_variant_reports": missing_variant_reports,
        "missing_audits": missing_audits,
    }


def summarize_task(task: dict[str, Any], *, seed: int, variant_id: str) -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "seed": seed,
        "task_id": str(task.get("task_id") or ""),
        "family": str(task.get("family") or "unknown"),
        "entry_point": task.get("entry_point"),
        "expected_plan": task.get("expected_plan_diagnostic_only"),
        "expected_return_shape": task.get("expected_return_shape_diagnostic_only"),
        "gap_status": str(task.get("gap_status") or "unknown"),
        "arms": {
            arm: summarize_arm(get_path(task, ["arms", arm, "private_eval"], {}))
            for arm in ["symliquid_style", "transformer_control"]
        },
    }


def summarize_arm(phase: Any) -> dict[str, Any]:
    phase = dict_or_empty(phase)
    return {
        "passed": bool(phase.get("passed")),
        "selected_rank": phase.get("selected_rank"),
        "selected_plan": phase.get("selected_plan"),
        "selected_predicted_return_shape": phase.get("selected_predicted_return_shape"),
        "selected_strategy": phase.get("selected_learned_internal_semantic_route_strategy") or "none",
        "selected_learned_internal_semantic_route": bool(phase.get("selected_learned_internal_semantic_route")),
        "wrong_answer_shape": phase.get("wrong_answer_shape") or "unknown",
        "verification_stage": phase.get("verification_stage"),
        "stderr_tail": phase.get("stderr_tail"),
    }


def mine_buckets(aligned: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_seed, _task_id), by_variant in sorted(aligned.items()):
        full = by_variant.get("full_learned_beam_off")
        dropout = by_variant.get("route_dropout_half_beam_off")
        no_visible = by_variant.get("no_visible_text_memory_beam_off")
        plan_head = by_variant.get("plan_head_only_beam_off")
        no_internal = by_variant.get("no_internal_beam_off")
        if not full or not no_visible:
            continue
        if full.get("gap_status") == "both_fail":
            bucket_rows["full_route_both_fail"].append(compare_row(full, dropout, no_visible, plan_head, no_internal))
        if no_visible.get("gap_status") == "symliquid_only_win":
            bucket_rows["no_visible_symliquid_only_win"].append(compare_row(no_visible, full, dropout, plan_head, no_internal))
        if no_visible.get("gap_status") == "transformer_only_win":
            bucket_rows["no_visible_transformer_only_win"].append(compare_row(no_visible, full, dropout, plan_head, no_internal))
        if no_visible.get("gap_status") == "both_pass" and (
            (full and full.get("gap_status") != "both_pass") or (dropout and dropout.get("gap_status") != "both_pass")
        ):
            bucket_rows["no_visible_both_pass_full_or_dropout_regression"].append(
                compare_row(no_visible, full, dropout, plan_head, no_internal)
            )
        if full and dropout:
            for arm in ["symliquid_style", "transformer_control"]:
                if arm_passed(full, arm) and not arm_passed(dropout, arm):
                    bucket_rows[f"dropout_regression_{arm}"].append(
                        compare_row(dropout, full, no_visible, plan_head, no_internal, focus_arm=arm)
                    )
        if full and no_visible:
            full_sym = get_path(full, ["arms", "symliquid_style"], {})
            no_vis_sym = get_path(no_visible, ["arms", "symliquid_style"], {})
            if bool(full_sym.get("passed")) and str(full_sym.get("selected_strategy")) == "visible_text_prototype_memory":
                name = "full_visible_text_symliquid_pass"
                if not bool(no_vis_sym.get("passed")):
                    name = "full_visible_text_rescues_symliquid_over_no_visible"
                bucket_rows[name].append(compare_row(full, no_visible, dropout, plan_head, no_internal))
            if bool(no_vis_sym.get("passed")) and str(full_sym.get("selected_strategy")) == "visible_text_prototype_memory":
                bucket_rows["context_can_pass_but_full_selects_visible_symliquid"].append(
                    compare_row(full, no_visible, dropout, plan_head, no_internal)
                )
    return {name: summarize_bucket(name, rows) for name, rows in sorted(bucket_rows.items())}


def compare_row(
    primary: dict[str, Any],
    alternate_a: dict[str, Any] | None,
    alternate_b: dict[str, Any] | None,
    plan_head: dict[str, Any] | None,
    no_internal: dict[str, Any] | None,
    *,
    focus_arm: str = "symliquid_style",
) -> dict[str, Any]:
    row = {
        "seed": primary.get("seed"),
        "task_id": primary.get("task_id"),
        "family": primary.get("family"),
        "entry_point": primary.get("entry_point"),
        "expected_plan": primary.get("expected_plan"),
        "expected_return_shape": primary.get("expected_return_shape"),
        "focus_arm": focus_arm,
        "variants": {
            "primary": variant_snapshot(primary),
        },
    }
    if alternate_a:
        row["variants"]["alternate_a"] = variant_snapshot(alternate_a)
    if alternate_b:
        row["variants"]["alternate_b"] = variant_snapshot(alternate_b)
    if plan_head:
        row["variants"]["plan_head_only_beam_off"] = variant_snapshot(plan_head)
    if no_internal:
        row["variants"]["no_internal_beam_off"] = variant_snapshot(no_internal)
    return row


def variant_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant_id": row.get("variant_id"),
        "gap_status": row.get("gap_status"),
        "symliquid": row.get("arms", {}).get("symliquid_style"),
        "transformer": row.get("arms", {}).get("transformer_control"),
    }


def summarize_bucket(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    sym_wrong: Counter[str] = Counter()
    tx_wrong: Counter[str] = Counter()
    sym_strategy: Counter[str] = Counter()
    plan_confusions: Counter[str] = Counter()
    for row in rows:
        family_counts[str(row.get("family") or "unknown")] += 1
        task_counts[str(row.get("task_id") or "")] += 1
        primary = dict_or_empty(row.get("variants", {}).get("primary"))
        sym = dict_or_empty(primary.get("symliquid"))
        tx = dict_or_empty(primary.get("transformer"))
        sym_wrong[str(sym.get("wrong_answer_shape") or "unknown")] += 1
        tx_wrong[str(tx.get("wrong_answer_shape") or "unknown")] += 1
        sym_strategy[str(sym.get("selected_strategy") or "none")] += 1
        expected = str(row.get("expected_plan") or "missing")
        selected = str(sym.get("selected_plan") or "missing")
        if expected != selected:
            plan_confusions[f"{expected}->{selected}"] += 1
    return {
        "row_count": len(rows),
        "family_counts": dict(family_counts.most_common(24)),
        "stable_task_counts": dict((task, count) for task, count in task_counts.most_common(24) if count >= 2),
        "symliquid_wrong_answer_shape_counts": dict(sym_wrong.most_common(24)),
        "transformer_wrong_answer_shape_counts": dict(tx_wrong.most_common(16)),
        "symliquid_selected_strategy_counts": dict(sym_strategy.most_common()),
        "symliquid_plan_confusion_counts": dict(plan_confusions.most_common(24)),
        "examples": rows[:16],
    }


def build_recommendation(ablation: dict[str, Any], buckets: dict[str, Any]) -> dict[str, Any]:
    attribution = dict_or_empty(ablation.get("attribution"))
    visible_rate = number(attribution.get("symliquid_full_visible_text_prototype_selected_rate"))
    no_visible = number(attribution.get("symliquid_no_visible_text_memory_mean"))
    dropout = number(attribution.get("symliquid_route_dropout_half_mean"))
    full_score = number(attribution.get("symliquid_beam_off_score"))
    if full_score is None:
        for row in ablation.get("variant_rows", []) if isinstance(ablation.get("variant_rows"), list) else []:
            if isinstance(row, dict) and row.get("id") == "full_learned_beam_off":
                full_score = number(row.get("symliquid_sts_on_mean"))
                break
    context_can_pass = int(get_path(buckets, ["context_can_pass_but_full_selects_visible_symliquid", "row_count"], 0) or 0)
    visible_rescue = int(get_path(buckets, ["full_visible_text_rescues_symliquid_over_no_visible", "row_count"], 0) or 0)
    no_visible_sym_only = int(get_path(buckets, ["no_visible_symliquid_only_win", "row_count"], 0) or 0)
    dropout_regressions = int(get_path(buckets, ["dropout_regression_symliquid_style", "row_count"], 0) or 0)
    visible_displaced = (
        visible_rate is not None
        and no_visible is not None
        and full_score is not None
        and visible_rate <= 0.35
        and no_visible >= max(0.8, full_score - 0.025)
        and visible_rescue == 0
    )
    if visible_displaced and dropout is not None and dropout >= 0.80 and dropout_regressions:
        next_action = "harden_remaining_dropout_plan_head_regressions_and_full_route_both_fail_families"
        rationale = (
            "The no-visible variant preserves the full-route score and the route-dropout mean has reached the "
            "target, but the margin is thin and residual dropout regressions remain. The next pressure should be "
            "the remaining plan-head dropout misses plus full-route both-fail families, not more visible-text "
            "displacement."
        )
    elif visible_displaced:
        next_action = "treat_contract_context_route_as_current_replacement_for_visible_text_memory"
        rationale = (
            "The no-visible variant preserves the full-route score while visible-text selected rate is below target. "
            "The next pressure should be dropout robustness and remaining full-route both-fail families, not more "
            "visible-text displacement."
        )
    elif context_can_pass:
        next_action = "prefer_context_when_it_already_passes_before_visible_text_route"
        rationale = (
            "There are full-route SymLiquid passes selecting visible text even though the no-visible context route "
            "also passes. The next change should adjust route scoring/order so context successes win before visible "
            "prototype memory, without removing visible memory entirely."
        )
    elif no_visible_sym_only:
        next_action = "mine_no_visible_symliquid_only_families_for_context_prototypes"
        rationale = (
            "No-visible SymLiquid-only rows exist but do not currently surface as full-route unique wins. Improve "
            "context prototype construction for those families, then retest full-route selection."
        )
    else:
        next_action = "treat_visible_route_as_current_score_wall"
        rationale = (
            "Residuals do not show enough already-solved context cases to displace visible-text routing safely. "
            "Further progress likely needs stronger source representation or return-shape-aware context prototypes."
        )
    return {
        "next_action": next_action,
        "rationale": rationale,
        "current_visible_text_selected_rate": visible_rate,
        "current_no_visible_symliquid_mean": no_visible,
        "current_full_symliquid_mean": full_score,
        "current_route_dropout_symliquid_mean": dropout,
        "context_can_pass_but_full_selects_visible_count": context_can_pass,
        "full_visible_text_rescue_count": visible_rescue,
        "no_visible_symliquid_only_count": no_visible_sym_only,
        "dropout_regression_symliquid_count": dropout_regressions,
    }


def summarize_buckets(buckets: dict[str, Any]) -> dict[str, Any]:
    return {
        "bucket_counts": {name: row.get("row_count", 0) for name, row in buckets.items()},
        "top_context_can_pass_families": get_path(
            buckets, ["context_can_pass_but_full_selects_visible_symliquid", "family_counts"], {}
        ),
        "top_visible_rescue_families": get_path(
            buckets, ["full_visible_text_rescues_symliquid_over_no_visible", "family_counts"], {}
        ),
        "top_no_visible_symliquid_only_families": get_path(
            buckets, ["no_visible_symliquid_only_win", "family_counts"], {}
        ),
        "top_dropout_regression_families": get_path(
            buckets, ["dropout_regression_symliquid_style", "family_counts"], {}
        ),
    }


def safety_summary(ablation: dict[str, Any], variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hard_failures = [
        row for row in ablation.get("gates", []) if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    return {
        "ablation_trigger_state": ablation.get("trigger_state"),
        "ablation_hard_gate_failure_count": len(hard_failures),
        "fallback_return_rate_zero": all(bool(row.get("fallback_return_rate_zero")) for row in variants.values()),
        "external_inference_calls": sum(int(row.get("external_inference_calls") or 0) for row in variants.values()),
        "teacher_used": any(bool(row.get("teacher_used")) for row in variants.values()),
        "public_training_rows": sum(int(row.get("public_training_rows") or 0) for row in variants.values()),
        "model_promotion_allowed": any(bool(row.get("model_promotion_allowed")) for row in variants.values()),
        "hard_gate_failures": hard_failures,
    }


def build_gates(
    ablation: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    aligned: dict[tuple[int, str], dict[str, Any]],
    buckets: dict[str, Any],
    safety: dict[str, Any],
    recommendation: dict[str, Any],
    load_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    required_variants = set(KEY_VARIANTS)
    loaded_variants = set(variants)
    return [
        gate("ablation_report_loaded", bool(ablation), {"trigger_state": ablation.get("trigger_state")}, "hard"),
        gate("required_variants_present", required_variants.issubset(loaded_variants), sorted(loaded_variants), "hard"),
        gate(
            "variant_reports_and_audits_loaded",
            not load_summary.get("missing_variant_reports") and not load_summary.get("missing_audits"),
            load_summary,
            "hard",
        ),
        gate("aligned_task_rows_loaded", bool(aligned), {"aligned_seed_task_count": len(aligned)}, "hard"),
        gate("residual_buckets_recorded", bool(buckets), {"bucket_names": sorted(buckets)}, "hard"),
        gate("fallback_terminal_returns_remain_zero", bool(safety.get("fallback_return_rate_zero")), safety, "hard"),
        gate("external_inference_zero", int(safety.get("external_inference_calls") or 0) == 0, safety, "hard"),
        gate("teacher_not_used", not bool(safety.get("teacher_used")), safety, "hard"),
        gate("public_training_rows_zero", int(safety.get("public_training_rows") or 0) == 0, safety, "hard"),
        gate("model_promotion_locked", not bool(safety.get("model_promotion_allowed")), safety, "hard"),
        gate("route_recommendation_recorded", bool(recommendation.get("next_action")), recommendation, "hard"),
    ]


def planned_report(ablation: str, started: float) -> dict[str, Any]:
    return {
        "policy": "project_theseus_private_residual_context_miner_v0",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "ablation_report": ablation,
        "summary": {
            "execute_required": True,
            "command": "python3 scripts/neural_seed_token_decoder_residual_context_miner.py --execute",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    recommendation = dict_or_empty(report.get("recommendation"))
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Neural Seed Residual Context Miner",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- next_action: `{recommendation.get('next_action')}`",
        f"- current_visible_text_selected_rate: `{recommendation.get('current_visible_text_selected_rate')}`",
        f"- current_full_symliquid_mean: `{recommendation.get('current_full_symliquid_mean')}`",
        f"- current_no_visible_symliquid_mean: `{recommendation.get('current_no_visible_symliquid_mean')}`",
        f"- current_route_dropout_symliquid_mean: `{recommendation.get('current_route_dropout_symliquid_mean')}`",
        f"- context_can_pass_but_full_selects_visible_count: "
        f"`{recommendation.get('context_can_pass_but_full_selects_visible_count')}`",
        f"- full_visible_text_rescue_count: `{recommendation.get('full_visible_text_rescue_count')}`",
        f"- no_visible_symliquid_only_count: `{recommendation.get('no_visible_symliquid_only_count')}`",
        f"- dropout_regression_symliquid_count: `{recommendation.get('dropout_regression_symliquid_count')}`",
        "",
        "## Rationale",
        "",
        str(recommendation.get("rationale") or ""),
        "",
        "## Bucket Counts",
        "",
    ]
    for name, count in dict_or_empty(summary.get("bucket_counts")).items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Bucket Details", ""])
    for name, bucket in dict_or_empty(report.get("buckets")).items():
        lines.append(f"### {name}")
        lines.append(f"- row_count: `{bucket.get('row_count')}`")
        lines.append(f"- family_counts: `{bucket.get('family_counts')}`")
        lines.append(f"- symliquid_wrong_answer_shape_counts: `{bucket.get('symliquid_wrong_answer_shape_counts')}`")
        lines.append(f"- symliquid_selected_strategy_counts: `{bucket.get('symliquid_selected_strategy_counts')}`")
        lines.append("")
    lines.extend(["## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    lines.extend(["", "## Semantics", "", str(report.get("score_semantics") or ""), ""])
    return "\n".join(lines)


def arm_passed(row: dict[str, Any], arm: str) -> bool:
    return bool(get_path(row, ["arms", arm, "passed"], False))


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
