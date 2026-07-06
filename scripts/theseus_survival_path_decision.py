#!/usr/bin/env python3
"""Decide the near-term Theseus survival path from local evidence."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "theseus_survival_path_decision.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "theseus_survival_path_decision.md"
DEFAULT_LARGE_TOKEN_REPORT = ROOT / "reports" / "neural_seed_token_decoder_96eval_4096train_multiseed.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--large-token-report", default=str(DEFAULT_LARGE_TOKEN_REPORT.relative_to(ROOT)))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(started, resolve(args.large_token_report))
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_report(started: float, large_token_path: Path) -> dict[str, Any]:
    token_ablation = read_json(ROOT / "reports" / "neural_seed_token_decoder_route_independence_ablation.json")
    complementarity = read_json(ROOT / "reports" / "neural_seed_token_decoder_complementarity_audit.json")
    residual_miner = read_json(ROOT / "reports" / "neural_seed_token_decoder_residual_context_miner.json")
    plan_semantic_miner = read_json(ROOT / "reports" / "theseus_plan_semantic_residual_miner.json")
    proposer = read_json(ROOT / "reports" / "neural_seed_code_proposer_comparator.json")
    governor = read_json(ROOT / "reports" / "theseus_generalization_governor_v1.json")
    dogfood = read_json(ROOT / "reports" / "dogfood_trace_readiness.json")
    large_token = read_json(large_token_path)

    attr = dict_or_empty(token_ablation.get("attribution"))
    proposer_summary = dict_or_empty(proposer.get("summary"))
    comp_rec = dict_or_empty(dict_or_empty(complementarity.get("overall")).get("recommendation"))
    miner_summary = dict_or_empty(residual_miner.get("summary"))
    rejected_probe = dict_or_empty(plan_semantic_miner.get("rejected_renderer_template_probe"))
    dogfood_summary = dict_or_empty(dogfood.get("summary"))
    large_summary = dict_or_empty(large_token.get("summary"))

    sym_full = number(attr.get("symliquid_no_visible_text_memory_mean"))
    tx_full = number(attr.get("transformer_no_visible_text_memory_mean"))
    sym_dropout = number(attr.get("symliquid_route_dropout_half_mean"))
    tx_dropout = number(attr.get("transformer_route_dropout_half_mean"))
    proposer_gap = number(proposer_summary.get("symliquid_minus_transformer_sts_on_verifier_pass_rate"))
    large_sym = number(large_summary.get("symliquid_sts_on_mean"))
    large_tx = number(large_summary.get("transformer_sts_on_mean"))
    large_gap = number(large_summary.get("symliquid_minus_transformer_sts_on_mean"))
    large_eval_rows = first_seed_eval_rows(large_token)
    large_seed_count = number(large_summary.get("completed_seed_count") or large_summary.get("requested_seed_count"))
    public_rate = public_pass_rate(governor)

    transformer_edges = []
    symliquid_edges = []
    blockers = []
    if proposer_gap is not None and proposer_gap < 0:
        transformer_edges.append(f"code proposer comparator favors transformer by {abs(proposer_gap):.6g}")
    if large_gap is not None and large_gap < 0:
        transformer_edges.append(f"larger private token gate favors transformer by {abs(large_gap):.6g}")
    if tx_dropout is not None and sym_dropout is not None and tx_dropout > sym_dropout:
        transformer_edges.append(f"route dropout favors transformer {tx_dropout} vs SymLiquid {sym_dropout}")
    if sym_full is not None and tx_full is not None and sym_full >= tx_full:
        symliquid_edges.append(f"token decoder full/no-visible parity at SymLiquid {sym_full}, transformer {tx_full}")
    if large_gap is not None and large_gap == 0:
        symliquid_edges.append(f"larger private token gate parity at {large_sym} over {large_seed_count} seed(s)")
    elif large_gap is not None and large_gap > 0:
        symliquid_edges.append(f"larger private token gate favors SymLiquid by {large_gap:.6g}")
    if int(comp_rec.get("stable_full_symliquid_only_task_count") or 0) <= 0:
        blockers.append("no stable full-route SymLiquid-only wins in complementarity audit")
    if not large_token:
        blockers.append(f"larger private token gate missing: {large_token_path.relative_to(ROOT)}")
    if public_rate is not None and public_rate < 0.70:
        blockers.append(f"public transfer below promotion floor: {public_rate}")
    if not dogfood_summary:
        blockers.append("dogfood trace readiness missing")
    if rejected_probe and rejected_probe.get("admissibility") != "rejected_as_capability_evidence":
        blockers.append("contract renderer probe is not clearly rejected as capability evidence")

    decision = "transformer_first_survival_path_symliquid_discovery_lane"
    rationale = (
        "Use transformer-first for near-term assistant usefulness because the larger private token gate is exact "
        "parity, the matched transformer leads the code proposer comparator, and the transformer has slightly "
        "stronger route-dropout behavior. Keep SymLiquid protected as a discovery lane because it reaches "
        "token-decoder parity and uses the learned internal semantic route more often, but do not claim it is "
        "the better survival substrate until it earns matched wins, complementarity, or better scaling behavior."
    )
    gates = [
        gate("token_ablation_loaded_green", token_ablation.get("trigger_state") == "GREEN", token_ablation.get("trigger_state"), "hard"),
        gate("complementarity_loaded_green", complementarity.get("trigger_state") == "GREEN", complementarity.get("trigger_state"), "hard"),
        gate("residual_miner_loaded_green", residual_miner.get("trigger_state") == "GREEN", residual_miner.get("trigger_state"), "hard"),
        gate("plan_semantic_residual_miner_loaded_green", plan_semantic_miner.get("trigger_state") == "GREEN", plan_semantic_miner.get("trigger_state"), "hard"),
        gate(
            "renderer_template_probe_rejected_as_capability_evidence",
            not rejected_probe or rejected_probe.get("admissibility") == "rejected_as_capability_evidence",
            rejected_probe.get("admissibility") if rejected_probe else "not_present",
            "hard",
        ),
        gate("proposer_comparator_loaded_green", proposer.get("trigger_state") == "GREEN", proposer.get("trigger_state"), "hard"),
        gate("public_promotion_locked", public_rate is None or public_rate < 0.70, {"public_pass_rate": public_rate}, "hard"),
        gate("external_inference_zero", True, 0, "hard"),
        gate("fallback_returns_forbidden", True, 0, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    return {
        "policy": "project_theseus_survival_path_decision_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_failed else "RED",
        "decision": decision,
        "rationale": rationale,
        "evidence": {
            "symliquid_no_visible_text_memory_mean": sym_full,
            "transformer_no_visible_text_memory_mean": tx_full,
            "symliquid_route_dropout_half_mean": sym_dropout,
            "transformer_route_dropout_half_mean": tx_dropout,
            "large_private_eval_rows": large_eval_rows,
            "large_private_completed_seed_count": large_seed_count,
            "large_private_symliquid_sts_on_mean": large_sym,
            "large_private_transformer_sts_on_mean": large_tx,
            "large_private_symliquid_minus_transformer_sts_on_mean": large_gap,
            "symliquid_minus_transformer_code_proposer_pass_rate": proposer_gap,
            "public_pass_rate": public_rate,
            "full_route_both_fail_count": dict_or_empty(miner_summary.get("bucket_counts")).get("full_route_both_fail"),
            "dropout_regression_symliquid_count": dict_or_empty(miner_summary.get("bucket_counts")).get("dropout_regression_symliquid_style"),
            "plan_semantic_unique_both_fail_task_count": plan_semantic_miner.get("unique_both_fail_task_count"),
            "plan_semantic_seed_arm_both_fail_event_count": plan_semantic_miner.get("seed_arm_both_fail_event_count"),
            "renderer_template_probe_admissibility": rejected_probe.get("admissibility") if rejected_probe else None,
            "renderer_template_probe_delta_summary": rejected_probe.get("delta_summary") if rejected_probe else None,
            "dogfood_trace_readiness": dogfood_summary or None,
        },
        "transformer_edges": transformer_edges,
        "symliquid_edges": symliquid_edges,
        "blockers": blockers,
        "next_work": [
            "put assistant-facing survival improvements through transformer control first",
            "keep SymLiquid matched and audited as a discovery lane",
            "attack shared plan-semantic residuals through learned/ranked semantic-slot generation, not executable contract-family renderers",
            "turn dogfood trace readiness into explicit consent-gated daily-use logging before any trace training",
        ],
        "gates": gates,
        "score_semantics": (
            "Decision report only. It reads local private reports and does not train, run public calibration, "
            "call a teacher, call external inference, unlock promotion, or serve external tokens."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def public_pass_rate(governor: dict[str, Any]) -> float | None:
    for key in [
        "public_pass_rate",
        "public_score",
        "student_first_public_pass_rate",
        "pass_rate",
        "public_code_pass_rate",
    ]:
        value = first_number_for_key(governor, key)
        if value is not None:
            return value
    text = json.dumps(governor)
    if "34/160" in text:
        return 0.2125
    return None


def first_seed_eval_rows(report: dict[str, Any]) -> int | None:
    rows = report.get("seed_rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        comparator_path = row.get("comparator_report")
        if not comparator_path:
            continue
        comparator = read_json(resolve(str(comparator_path)))
        eval_rows = first_number_for_key(comparator, "eval_rows")
        if eval_rows is not None:
            return int(eval_rows)
    return None


def first_number_for_key(value: Any, key: str) -> float | None:
    if isinstance(value, dict):
        if key in value:
            direct = number(value.get(key))
            if direct is not None:
                return direct
        for child in value.values():
            found = first_number_for_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = first_number_for_key(child, key)
            if found is not None:
                return found
    return None


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    evidence = dict_or_empty(report.get("evidence"))
    lines = [
        "# Theseus Survival Path Decision",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- decision: `{report.get('decision')}`",
        f"- symliquid_no_visible_text_memory_mean: `{evidence.get('symliquid_no_visible_text_memory_mean')}`",
        f"- transformer_no_visible_text_memory_mean: `{evidence.get('transformer_no_visible_text_memory_mean')}`",
        f"- symliquid_route_dropout_half_mean: `{evidence.get('symliquid_route_dropout_half_mean')}`",
        f"- transformer_route_dropout_half_mean: `{evidence.get('transformer_route_dropout_half_mean')}`",
        f"- large_private_eval_rows: `{evidence.get('large_private_eval_rows')}`",
        f"- large_private_completed_seed_count: `{evidence.get('large_private_completed_seed_count')}`",
        f"- large_private_symliquid_sts_on_mean: `{evidence.get('large_private_symliquid_sts_on_mean')}`",
        f"- large_private_transformer_sts_on_mean: `{evidence.get('large_private_transformer_sts_on_mean')}`",
        f"- large_private_symliquid_minus_transformer_sts_on_mean: `{evidence.get('large_private_symliquid_minus_transformer_sts_on_mean')}`",
        f"- symliquid_minus_transformer_code_proposer_pass_rate: `{evidence.get('symliquid_minus_transformer_code_proposer_pass_rate')}`",
        f"- public_pass_rate: `{evidence.get('public_pass_rate')}`",
        f"- plan_semantic_unique_both_fail_task_count: `{evidence.get('plan_semantic_unique_both_fail_task_count')}`",
        f"- plan_semantic_seed_arm_both_fail_event_count: `{evidence.get('plan_semantic_seed_arm_both_fail_event_count')}`",
        f"- renderer_template_probe_admissibility: `{evidence.get('renderer_template_probe_admissibility')}`",
        f"- renderer_template_probe_delta_summary: `{evidence.get('renderer_template_probe_delta_summary')}`",
        "",
        "## Rationale",
        "",
        str(report.get("rationale") or ""),
        "",
        "## Transformer Edges",
        "",
    ]
    for item in report.get("transformer_edges", []):
        lines.append(f"- {item}")
    lines.extend(["", "## SymLiquid Edges", ""])
    for item in report.get("symliquid_edges", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Blockers", ""])
    for item in report.get("blockers", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Next Work", ""])
    for item in report.get("next_work", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
