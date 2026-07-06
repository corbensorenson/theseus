#!/usr/bin/env python3
"""Run beam-on/off and learned-routing ablations for the neural seed decoder.

The goal is attribution, not promotion: each variant runs the normal private
multi-seed token-decoder smoke and keeps public calibration, teacher calls,
fallback terminal returns, and promotion locked.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
DEFAULT_OUT = ROOT / "reports" / "neural_seed_token_decoder_route_independence_ablation.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "neural_seed_token_decoder_route_independence_ablation.md"
REFERENCE_PRE_BEAM_SYMLIQUID_MEAN = 0.166667
REFERENCE_PRE_BEAM_TRANSFORMER_MEAN = 0.2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--seeds", default="23,29,31,37,41")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    base_config = read_json(config_path)
    if not args.execute:
        report = planned_report(args, started)
    else:
        report = run_ablation(base_config, config_path, args.seeds, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def run_ablation(base_config: dict[str, Any], config_path: Path, seeds: str, started: float) -> dict[str, Any]:
    variants = [
        {
            "id": "full_learned_beam_on",
            "label": "full learned routing + visible-contract beam",
            "internal_semantic_routing_enabled": True,
            "visible_contract_semantic_beam_enabled": True,
            "routing_overrides": {},
        },
        {
            "id": "full_learned_beam_off",
            "label": "full learned routing, beam disabled",
            "internal_semantic_routing_enabled": True,
            "visible_contract_semantic_beam_enabled": False,
            "routing_overrides": {},
        },
        {
            "id": "route_dropout_half_beam_off",
            "label": "full learned routing, beam disabled, prototype routes kept at 50%",
            "internal_semantic_routing_enabled": True,
            "visible_contract_semantic_beam_enabled": False,
            "routing_overrides": {
                "prototype_route_keep_rate": 0.5,
                "contract_fingerprint_route_keep_rate": 0.5,
                "contract_feature_route_keep_rate": 0.5,
                "visible_text_prototype_route_keep_rate": 0.5,
            },
        },
        {
            "id": "no_visible_text_memory_beam_off",
            "label": "no visible-text prototype memory, beam disabled",
            "internal_semantic_routing_enabled": True,
            "visible_contract_semantic_beam_enabled": False,
            "routing_overrides": {
                "visible_text_prototype_route_memory": False,
                "visible_text_prototype_route_top_k": 0,
                "visible_text_prototype_route_keep_rate": 0.0,
            },
        },
        {
            "id": "plan_head_only_beam_off",
            "label": "plan-head only, no prototype memory, beam disabled",
            "internal_semantic_routing_enabled": True,
            "visible_contract_semantic_beam_enabled": False,
            "routing_overrides": {
                "prototype_route_memory": False,
                "prototype_route_weight": 0.0,
                "prototype_route_keep_rate": 0.0,
                "contract_fingerprint_route_memory": False,
                "contract_fingerprint_route_top_k": 0,
                "contract_fingerprint_route_keep_rate": 0.0,
                "contract_feature_route_memory": False,
                "contract_feature_route_top_k": 0,
                "contract_feature_route_keep_rate": 0.0,
                "visible_text_prototype_route_memory": False,
                "visible_text_prototype_route_top_k": 0,
                "visible_text_prototype_route_keep_rate": 0.0,
            },
        },
        {
            "id": "no_internal_beam_off",
            "label": "no learned internal routing, beam disabled",
            "internal_semantic_routing_enabled": False,
            "visible_contract_semantic_beam_enabled": False,
            "routing_overrides": {},
        },
    ]
    variant_rows = []
    for variant in variants:
        variant_config = ablation_config(
            base_config,
            routing_enabled=bool(variant["internal_semantic_routing_enabled"]),
            beam_enabled=bool(variant["visible_contract_semantic_beam_enabled"]),
            routing_overrides=dict_or_empty(variant.get("routing_overrides")),
        )
        with tempfile.NamedTemporaryFile("w", suffix=f"-theseus-token-decoder-ablation-{variant['id']}.json", delete=False) as handle:
            json.dump(variant_config, handle, indent=2)
            handle.write("\n")
            temp_config = Path(handle.name)
        out = ROOT / "reports" / f"neural_seed_token_decoder_ablation_{variant['id']}.json"
        markdown = ROOT / "reports" / f"neural_seed_token_decoder_ablation_{variant['id']}.md"
        artifact_prefix = f"neural_seed_token_decoder_ablation_{variant['id']}"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "neural_seed_token_decoder_multiseed_smoke.py"),
            "--config",
            str(temp_config),
            "--out",
            str(out),
            "--markdown-out",
            str(markdown),
            "--artifact-prefix",
            artifact_prefix,
            "--seeds",
            seeds,
            "--execute",
        ]
        command = run_command(cmd)
        variant_report = read_json(out)
        variant_rows.append(variant_summary(variant, command, variant_report, out, markdown, artifact_prefix))
        temp_config.unlink(missing_ok=True)

    attribution = attribution_summary(variant_rows)
    residuals = residual_failure_summary(variant_rows)
    dogfood = dogfood_trace_summary()
    gates = build_gates(variant_rows, attribution, residuals, dogfood)
    trigger = "GREEN" if all(row["passed"] for row in gates if row["severity"] == "hard") else "RED"
    if trigger == "GREEN" and any(not row["passed"] for row in gates):
        trigger = "YELLOW"
    return {
        "policy": "project_theseus_neural_seed_token_decoder_route_independence_ablation_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "config": rel(config_path),
        "starting_git_commit": git_commit(),
        "seeds": parse_seeds(seeds),
        "reference_pre_beam": {
            "symliquid_sts_on_mean": REFERENCE_PRE_BEAM_SYMLIQUID_MEAN,
            "transformer_sts_on_mean": REFERENCE_PRE_BEAM_TRANSFORMER_MEAN,
            "source": "pre-visible-contract-beam five-seed token-decoder smoke",
        },
        "variant_rows": variant_rows,
        "attribution": attribution,
        "residual_failure_summary": residuals,
        "dogfood_trace_compatibility": dogfood,
        "gates": gates,
        "score_semantics": (
            "Private route-independence ablation over the normal multi-seed token-decoder smoke. The learned-routing "
            "variants train a matched plan router in both arms from private train semantic-slot targets and STS-on "
            "visible contract fields, then separately enable or remove visible-text prototype memory, model-context "
            "prototype memory, and deterministic visible-contract beams. No public calibration, teacher calls, "
            "fallback terminal returns, distillation, promotion, or external inference are allowed."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def ablation_config(
    base_config: dict[str, Any],
    *,
    routing_enabled: bool,
    beam_enabled: bool,
    routing_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    structure = config.setdefault("body_structure_decoder", {})
    beam = structure.setdefault("visible_contract_semantic_beam", {})
    beam["enabled"] = bool(beam_enabled)
    routing = structure.setdefault("internal_semantic_routing", {})
    routing["enabled"] = bool(routing_enabled)
    routing["matched_for_both_arms"] = True
    routing["uses_eval_tests_or_solutions"] = False
    routing["uses_public_data"] = False
    for key, value in dict_or_empty(routing_overrides).items():
        routing[key] = value
    if not routing_enabled:
        routing["auxiliary_plan_loss_weight"] = 0.0
        routing["plan_router_scale"] = 0.0
        routing["learned_route_top_k"] = 0
        routing["prototype_route_memory"] = False
        routing["prototype_route_weight"] = 0.0
        routing["prototype_route_keep_rate"] = 0.0
        routing["contract_fingerprint_route_memory"] = False
        routing["contract_fingerprint_route_top_k"] = 0
        routing["contract_fingerprint_route_keep_rate"] = 0.0
        routing["contract_feature_route_memory"] = False
        routing["contract_feature_route_top_k"] = 0
        routing["contract_feature_route_keep_rate"] = 0.0
        routing["visible_text_prototype_route_memory"] = False
        routing["visible_text_prototype_route_top_k"] = 0
        routing["visible_text_prototype_route_keep_rate"] = 0.0
    return config


def variant_summary(
    variant: dict[str, Any],
    command: dict[str, Any],
    report: dict[str, Any],
    report_path: Path,
    markdown_path: Path,
    artifact_prefix: str,
) -> dict[str, Any]:
    summary = dict_or_empty(report.get("summary"))
    seed_rows = report.get("seed_rows") if isinstance(report.get("seed_rows"), list) else []
    hard_failures = [
        row
        for seed_row in seed_rows
        for row in seed_row.get("hard_gate_failures", [])
        if isinstance(row, dict) and not row.get("passed")
    ]
    audit_hard_failures = [
        row
        for seed_row in seed_rows
        for row in seed_row.get("audit_hard_gate_failures", [])
        if isinstance(row, dict) and not row.get("passed")
    ]
    fallback_zero = all(
        float(seed_row.get("symliquid_fallback_rate") or 0.0) == 0.0
        and float(seed_row.get("transformer_fallback_rate") or 0.0) == 0.0
        for seed_row in seed_rows
    )
    return {
        "id": variant["id"],
        "label": variant["label"],
        "internal_semantic_routing_enabled": bool(variant["internal_semantic_routing_enabled"]),
        "visible_contract_semantic_beam_enabled": bool(variant["visible_contract_semantic_beam_enabled"]),
        "routing_overrides": dict_or_empty(variant.get("routing_overrides")),
        "returncode": command.get("returncode"),
        "trigger_state": report.get("trigger_state"),
        "report": rel(report_path),
        "markdown": rel(markdown_path),
        "artifact_prefix": artifact_prefix,
        "completed_seed_count": summary.get("completed_seed_count"),
        "requested_seed_count": summary.get("requested_seed_count"),
        "symliquid_sts_on_mean": summary.get("symliquid_sts_on_mean"),
        "symliquid_sts_on_stdev": summary.get("symliquid_sts_on_stdev"),
        "transformer_sts_on_mean": summary.get("transformer_sts_on_mean"),
        "transformer_sts_on_stdev": summary.get("transformer_sts_on_stdev"),
        "symliquid_minus_transformer_sts_on_mean": summary.get("symliquid_minus_transformer_sts_on_mean"),
        "symliquid_expected_plan_match_mean": summary.get("symliquid_expected_plan_match_mean"),
        "transformer_expected_plan_match_mean": summary.get("transformer_expected_plan_match_mean"),
        "symliquid_visible_contract_semantic_beam_selected_mean": summary.get("symliquid_visible_contract_semantic_beam_selected_mean"),
        "transformer_visible_contract_semantic_beam_selected_mean": summary.get("transformer_visible_contract_semantic_beam_selected_mean"),
        "symliquid_visible_contract_semantic_beam_available_mean": summary.get("symliquid_visible_contract_semantic_beam_available_mean"),
        "transformer_visible_contract_semantic_beam_available_mean": summary.get("transformer_visible_contract_semantic_beam_available_mean"),
        "symliquid_learned_internal_semantic_route_selected_mean": summary.get("symliquid_learned_internal_semantic_route_selected_mean"),
        "transformer_learned_internal_semantic_route_selected_mean": summary.get("transformer_learned_internal_semantic_route_selected_mean"),
        "symliquid_learned_internal_semantic_route_available_mean": summary.get("symliquid_learned_internal_semantic_route_available_mean"),
        "transformer_learned_internal_semantic_route_available_mean": summary.get("transformer_learned_internal_semantic_route_available_mean"),
        "symliquid_learned_internal_semantic_route_strategy_selected_means": summary.get("symliquid_learned_internal_semantic_route_strategy_selected_means"),
        "transformer_learned_internal_semantic_route_strategy_selected_means": summary.get("transformer_learned_internal_semantic_route_strategy_selected_means"),
        "symliquid_learned_internal_semantic_route_strategy_available_means": summary.get("symliquid_learned_internal_semantic_route_strategy_available_means"),
        "transformer_learned_internal_semantic_route_strategy_available_means": summary.get("transformer_learned_internal_semantic_route_strategy_available_means"),
        "winner_counts": summary.get("winner_counts"),
        "fallback_return_rate_zero": fallback_zero,
        "external_inference_calls": summary.get("external_inference_calls", 0),
        "teacher_used": bool(summary.get("teacher_used")),
        "public_training_rows": summary.get("public_training_rows", 0),
        "model_promotion_allowed": bool(summary.get("model_promotion_allowed")),
        "hard_gate_failure_count": len(hard_failures),
        "audit_hard_gate_failure_count": len(audit_hard_failures),
        "hard_gate_failures": hard_failures[:8],
        "audit_hard_gate_failures": audit_hard_failures[:8],
        "stdout_tail": command.get("stdout_tail"),
        "stderr_tail": command.get("stderr_tail"),
    }


def attribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("id")): row for row in rows}
    beam_on = by_id.get("full_learned_beam_on", {})
    beam_off = by_id.get("full_learned_beam_off", {})
    dropout = by_id.get("route_dropout_half_beam_off", {})
    no_visible = by_id.get("no_visible_text_memory_beam_off", {})
    plan_head = by_id.get("plan_head_only_beam_off", {})
    no_internal = by_id.get("no_internal_beam_off", {})
    sym_beam_on = number(beam_on.get("symliquid_sts_on_mean"))
    sym_beam_off = number(beam_off.get("symliquid_sts_on_mean"))
    sym_dropout = number(dropout.get("symliquid_sts_on_mean"))
    sym_no_visible = number(no_visible.get("symliquid_sts_on_mean"))
    sym_plan_head = number(plan_head.get("symliquid_sts_on_mean"))
    sym_no_internal = number(no_internal.get("symliquid_sts_on_mean"))
    tx_beam_on = number(beam_on.get("transformer_sts_on_mean"))
    tx_beam_off = number(beam_off.get("transformer_sts_on_mean"))
    tx_dropout = number(dropout.get("transformer_sts_on_mean"))
    tx_no_visible = number(no_visible.get("transformer_sts_on_mean"))
    tx_plan_head = number(plan_head.get("transformer_sts_on_mean"))
    tx_no_internal = number(no_internal.get("transformer_sts_on_mean"))
    sym_selected = number(beam_on.get("symliquid_visible_contract_semantic_beam_selected_mean"))
    tx_selected = number(beam_on.get("transformer_visible_contract_semantic_beam_selected_mean"))
    sym_learned_selected = number(beam_off.get("symliquid_learned_internal_semantic_route_selected_mean"))
    tx_learned_selected = number(beam_off.get("transformer_learned_internal_semantic_route_selected_mean"))
    sym_full_strategies = dict_or_empty(beam_off.get("symliquid_learned_internal_semantic_route_strategy_selected_means"))
    tx_full_strategies = dict_or_empty(beam_off.get("transformer_learned_internal_semantic_route_strategy_selected_means"))
    sym_dropout_strategies = dict_or_empty(dropout.get("symliquid_learned_internal_semantic_route_strategy_selected_means"))
    sym_no_visible_strategies = dict_or_empty(no_visible.get("symliquid_learned_internal_semantic_route_strategy_selected_means"))
    sym_plan_head_strategies = dict_or_empty(plan_head.get("symliquid_learned_internal_semantic_route_strategy_selected_means"))
    best_reduced = best_reduced_prototype_variant([dropout, no_visible, plan_head])
    return {
        "symliquid_learned_internal_routing_delta_vs_no_internal_beam_off": subtract(sym_beam_off, sym_no_internal),
        "transformer_learned_internal_routing_delta_vs_no_internal_beam_off": subtract(tx_beam_off, tx_no_internal),
        "symliquid_route_dropout_delta_vs_full_beam_off": subtract(sym_dropout, sym_beam_off),
        "transformer_route_dropout_delta_vs_full_beam_off": subtract(tx_dropout, tx_beam_off),
        "symliquid_visible_text_prototype_memory_delta": subtract(sym_beam_off, sym_no_visible),
        "transformer_visible_text_prototype_memory_delta": subtract(tx_beam_off, tx_no_visible),
        "symliquid_context_prototype_memory_delta": subtract(sym_no_visible, sym_plan_head),
        "transformer_context_prototype_memory_delta": subtract(tx_no_visible, tx_plan_head),
        "symliquid_plan_head_only_delta_vs_no_internal": subtract(sym_plan_head, sym_no_internal),
        "transformer_plan_head_only_delta_vs_no_internal": subtract(tx_plan_head, tx_no_internal),
        "symliquid_shared_beam_delta_vs_internal_beam_off": subtract(sym_beam_on, sym_beam_off),
        "transformer_shared_beam_delta_vs_internal_beam_off": subtract(tx_beam_on, tx_beam_off),
        "symliquid_beam_off_recovery_delta_vs_pre_beam_reference": subtract(sym_beam_off, REFERENCE_PRE_BEAM_SYMLIQUID_MEAN),
        "transformer_beam_off_recovery_delta_vs_pre_beam_reference": subtract(tx_beam_off, REFERENCE_PRE_BEAM_TRANSFORMER_MEAN),
        "symliquid_plan_head_only_mean": sym_plan_head,
        "transformer_plan_head_only_mean": tx_plan_head,
        "symliquid_no_visible_text_memory_mean": sym_no_visible,
        "transformer_no_visible_text_memory_mean": tx_no_visible,
        "symliquid_route_dropout_half_mean": sym_dropout,
        "transformer_route_dropout_half_mean": tx_dropout,
        "symliquid_beam_on_selected_rate": sym_selected,
        "transformer_beam_on_selected_rate": tx_selected,
        "mean_beam_on_selected_rate": mean(numbers([sym_selected, tx_selected])),
        "symliquid_beam_off_learned_route_selected_rate": sym_learned_selected,
        "transformer_beam_off_learned_route_selected_rate": tx_learned_selected,
        "mean_beam_off_learned_route_selected_rate": mean(numbers([sym_learned_selected, tx_learned_selected])),
        "symliquid_full_visible_text_prototype_selected_rate": number(sym_full_strategies.get("visible_text_prototype_memory")),
        "transformer_full_visible_text_prototype_selected_rate": number(tx_full_strategies.get("visible_text_prototype_memory")),
        "symliquid_full_context_prototype_selected_rate": number(sym_full_strategies.get("plan_head_plus_context_prototype_memory")),
        "transformer_full_context_prototype_selected_rate": number(tx_full_strategies.get("plan_head_plus_context_prototype_memory")),
        "symliquid_full_contract_fingerprint_selected_rate": number(sym_full_strategies.get("contract_fingerprint_context_memory")),
        "transformer_full_contract_fingerprint_selected_rate": number(tx_full_strategies.get("contract_fingerprint_context_memory")),
        "symliquid_full_contract_feature_selected_rate": number(sym_full_strategies.get("contract_feature_context_memory")),
        "transformer_full_contract_feature_selected_rate": number(tx_full_strategies.get("contract_feature_context_memory")),
        "symliquid_full_plan_head_selected_rate": number(sym_full_strategies.get("plan_head")),
        "transformer_full_plan_head_selected_rate": number(tx_full_strategies.get("plan_head")),
        "symliquid_dropout_visible_text_prototype_selected_rate": number(sym_dropout_strategies.get("visible_text_prototype_memory")),
        "symliquid_dropout_contract_fingerprint_selected_rate": number(sym_dropout_strategies.get("contract_fingerprint_context_memory")),
        "symliquid_dropout_contract_feature_selected_rate": number(sym_dropout_strategies.get("contract_feature_context_memory")),
        "symliquid_no_visible_context_prototype_selected_rate": number(sym_no_visible_strategies.get("plan_head_plus_context_prototype_memory")),
        "symliquid_no_visible_contract_fingerprint_selected_rate": number(sym_no_visible_strategies.get("contract_fingerprint_context_memory")),
        "symliquid_no_visible_contract_feature_selected_rate": number(sym_no_visible_strategies.get("contract_feature_context_memory")),
        "symliquid_plan_head_only_selected_rate": number(sym_plan_head_strategies.get("plan_head")),
        "best_non_collapsed_prototype_reduced_variant": best_reduced,
        "symliquid_beam_off_gap_vs_transformer": subtract(sym_beam_off, tx_beam_off),
        "symliquid_beam_on_gap_vs_transformer": subtract(sym_beam_on, tx_beam_on),
        "likely_primary_source": likely_source(sym_beam_off, sym_no_internal, sym_beam_on, sym_no_visible, sym_plan_head),
    }


def likely_source(
    sym_beam_off: float | None,
    sym_no_internal: float | None,
    sym_beam_on: float | None,
    sym_no_visible: float | None,
    sym_plan_head: float | None,
) -> str:
    learned_delta = subtract(sym_beam_off, sym_no_internal)
    beam_delta = subtract(sym_beam_on, sym_beam_off)
    visible_memory_delta = subtract(sym_beam_off, sym_no_visible)
    plan_head_delta = subtract(sym_plan_head, sym_no_internal)
    reference_delta = subtract(sym_no_internal, REFERENCE_PRE_BEAM_SYMLIQUID_MEAN)
    if visible_memory_delta is not None and visible_memory_delta >= 0.2:
        return "visible_text_prototype_route_memory"
    no_visible_delta = subtract(sym_no_visible, sym_plan_head)
    if no_visible_delta is not None and no_visible_delta >= 0.15:
        return "contract_or_context_route_memory"
    if plan_head_delta is not None and plan_head_delta >= 0.15:
        return "learned_neural_plan_head"
    if beam_delta is not None and beam_delta >= 0.05:
        return "shared_visible_contract_semantic_beam"
    if learned_delta is not None and learned_delta >= 0.05:
        return "mixed_internal_semantic_routing"
    if reference_delta is not None and reference_delta >= 0.05:
        return "renderer_or_capacity_change_without_internal_router"
    return "mixed_or_unresolved"


def best_reduced_prototype_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for row in rows:
        sym = number(row.get("symliquid_sts_on_mean"))
        if sym is None:
            continue
        strategies = dict_or_empty(row.get("symliquid_learned_internal_semantic_route_strategy_selected_means"))
        visible_rate = number(strategies.get("visible_text_prototype_memory")) or 0.0
        candidates.append(
            {
                "id": row.get("id"),
                "symliquid_sts_on_mean": sym,
                "transformer_sts_on_mean": row.get("transformer_sts_on_mean"),
                "symliquid_visible_text_prototype_selected_rate": visible_rate,
                "symliquid_strategy_selected_means": strategies,
            }
        )
    candidates.sort(key=lambda item: (float(item.get("symliquid_sts_on_mean") or 0.0), -float(item.get("symliquid_visible_text_prototype_selected_rate") or 0.0)), reverse=True)
    return candidates[0] if candidates else {}


def build_gates(
    rows: list[dict[str, Any]],
    attribution: dict[str, Any],
    residuals: dict[str, Any],
    dogfood: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {str(row.get("id")): row for row in rows}
    required_ids = {
        "full_learned_beam_on",
        "full_learned_beam_off",
        "route_dropout_half_beam_off",
        "no_visible_text_memory_beam_off",
        "plan_head_only_beam_off",
        "no_internal_beam_off",
    }
    beam_on = by_id.get("full_learned_beam_on", {})
    beam_off = by_id.get("full_learned_beam_off", {})
    plan_head = by_id.get("plan_head_only_beam_off", {})
    no_internal = by_id.get("no_internal_beam_off", {})
    best_reduced = dict_or_empty(attribution.get("best_non_collapsed_prototype_reduced_variant"))
    return [
        gate("all_required_ablation_variants_ran", required_ids.issubset(set(by_id)) and all(row.get("completed_seed_count") == row.get("requested_seed_count") for row in rows), row_counts(rows), "hard"),
        gate("fallback_return_rate_zero_all_variants", all(bool(row.get("fallback_return_rate_zero")) for row in rows), {row.get("id"): row.get("fallback_return_rate_zero") for row in rows}, "hard"),
        gate("external_inference_zero_all_variants", all(int(row.get("external_inference_calls") or 0) == 0 for row in rows), {row.get("id"): row.get("external_inference_calls") for row in rows}, "hard"),
        gate("teacher_public_promotion_locked_all_variants", all(not row.get("teacher_used") and int(row.get("public_training_rows") or 0) == 0 and not row.get("model_promotion_allowed") for row in rows), safety_evidence(rows), "hard"),
        gate("no_comparator_hard_gate_failures", all(int(row.get("hard_gate_failure_count") or 0) == 0 for row in rows), {row.get("id"): row.get("hard_gate_failures") for row in rows}, "hard"),
        gate("no_audit_hard_gate_failures", all(int(row.get("audit_hard_gate_failure_count") or 0) == 0 for row in rows), {row.get("id"): row.get("audit_hard_gate_failures") for row in rows}, "hard"),
        gate(
            "full_beam_off_symliquid_stays_close_to_current_mean",
            number(beam_off.get("symliquid_sts_on_mean")) is not None
            and float(beam_off.get("symliquid_sts_on_mean")) >= 0.75,
            {
                "beam_off_symliquid_mean": beam_off.get("symliquid_sts_on_mean"),
                "current_reference": 0.85,
                "delta": attribution.get("symliquid_beam_off_recovery_delta_vs_pre_beam_reference"),
            },
            "hard",
        ),
        gate(
            "learned_routing_improves_symliquid_beam_off_vs_no_internal",
            number(attribution.get("symliquid_learned_internal_routing_delta_vs_no_internal_beam_off")) is not None
            and float(attribution.get("symliquid_learned_internal_routing_delta_vs_no_internal_beam_off")) >= 0.05,
            attribution,
            "hard",
        ),
        gate(
            "beam_on_selected_rate_materially_reduced",
            number(attribution.get("mean_beam_on_selected_rate")) is not None
            and float(attribution.get("mean_beam_on_selected_rate")) <= 0.2,
            {
                "symliquid": attribution.get("symliquid_beam_on_selected_rate"),
                "transformer": attribution.get("transformer_beam_on_selected_rate"),
                "mean": attribution.get("mean_beam_on_selected_rate"),
            },
            "hard",
        ),
        gate(
            "symliquid_near_parity_or_ahead_with_beam_on",
            number(beam_on.get("symliquid_minus_transformer_sts_on_mean")) is not None
            and float(beam_on.get("symliquid_minus_transformer_sts_on_mean")) >= -0.1,
            beam_on.get("symliquid_minus_transformer_sts_on_mean"),
            "hard",
        ),
        gate(
            "symliquid_beam_off_not_collapsed",
            number(beam_off.get("symliquid_sts_on_mean")) is not None
            and float(beam_off.get("symliquid_sts_on_mean")) >= max(0.3, REFERENCE_PRE_BEAM_SYMLIQUID_MEAN + 0.1),
            beam_off.get("symliquid_sts_on_mean"),
            "hard",
        ),
        gate(
            "plan_head_only_or_blocker_reported",
            bool(plan_head) and bool(residuals),
            {
                "plan_head_only_symliquid_mean": plan_head.get("symliquid_sts_on_mean"),
                "target": 0.35,
                "residual_summary_recorded": bool(residuals),
            },
            "hard",
        ),
        gate(
            "plan_head_only_independence_target",
            number(plan_head.get("symliquid_sts_on_mean")) is not None
            and float(plan_head.get("symliquid_sts_on_mean")) >= 0.35,
            {
                "plan_head_only_symliquid_mean": plan_head.get("symliquid_sts_on_mean"),
                "no_internal_symliquid_mean": no_internal.get("symliquid_sts_on_mean"),
                "delta": attribution.get("symliquid_plan_head_only_delta_vs_no_internal"),
                "blocker_if_failed": residuals.get("top_symliquid_plan_head_only_failures"),
            },
            "soft",
        ),
        gate(
            "visible_text_prototype_dependence_reduced_in_best_variant",
            number(best_reduced.get("symliquid_sts_on_mean")) is not None
            and float(best_reduced.get("symliquid_sts_on_mean")) >= 0.35
            and float(best_reduced.get("symliquid_visible_text_prototype_selected_rate") or 0.0) <= 0.5,
            best_reduced,
            "soft",
        ),
        gate(
            "residual_failures_mined",
            bool(residuals.get("by_variant")),
            {"variants": list(dict_or_empty(residuals.get("by_variant")).keys())},
            "hard",
        ),
        gate(
            "dogfood_trace_compatibility_checked",
            bool(dogfood.get("checked")) or bool(dogfood.get("not_checked_reason")),
            dogfood,
            "soft",
        ),
        gate(
            "no_internal_beam_off_kept_as_attribution_control",
            bool(no_internal),
            no_internal.get("report"),
            "soft",
        ),
    ]


def row_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {row.get("id"): {"completed": row.get("completed_seed_count"), "requested": row.get("requested_seed_count")} for row in rows}


def safety_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        row.get("id"): {
            "teacher_used": row.get("teacher_used"),
            "public_training_rows": row.get("public_training_rows"),
            "model_promotion_allowed": row.get("model_promotion_allowed"),
        }
        for row in rows
    }


def residual_failure_summary(variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    plan_head_failures: Counter[str] = Counter()
    for variant in variant_rows:
        variant_id = str(variant.get("id") or "")
        report_path = resolve(str(variant.get("report") or ""))
        report = read_json(report_path)
        family_counts: Counter[str] = Counter()
        wrong_shape_counts: Counter[str] = Counter()
        strategy_failure_counts: Counter[str] = Counter()
        gap_counts: Counter[str] = Counter()
        examples = []
        for seed_row in report.get("seed_rows", []) if isinstance(report.get("seed_rows"), list) else []:
            audit_path = resolve(str(seed_row.get("semantic_plan_audit") or ""))
            audit = read_json(audit_path)
            for task in audit.get("task_rows", []) if isinstance(audit.get("task_rows"), list) else []:
                family = str(task.get("family") or "unknown")
                gap_status = str(task.get("gap_status") or "unknown")
                gap_counts[gap_status] += 1
                for arm in ["symliquid_style", "transformer_control"]:
                    phase = dict_or_empty(get_path(task, ["arms", arm, "private_eval"], {}))
                    if phase.get("passed"):
                        continue
                    wrong = str(phase.get("wrong_answer_shape") or "unknown")
                    strategy = str(phase.get("selected_learned_internal_semantic_route_strategy") or "none")
                    key = f"{arm}:{family}:{wrong}"
                    family_counts[f"{arm}:{family}"] += 1
                    wrong_shape_counts[key] += 1
                    strategy_failure_counts[f"{arm}:{strategy}"] += 1
                    if variant_id == "plan_head_only_beam_off" and arm == "symliquid_style":
                        plan_head_failures[f"{family}:{wrong}"] += 1
                    if len(examples) < 12:
                        examples.append(
                            {
                                "seed": seed_row.get("seed"),
                                "task_id": task.get("task_id"),
                                "family": family,
                                "gap_status": gap_status,
                                "arm": arm,
                                "wrong_answer_shape": wrong,
                                "selected_strategy": strategy,
                                "selected_plan": phase.get("selected_plan"),
                                "expected_plan_diagnostic_only": task.get("expected_plan_diagnostic_only"),
                            }
                        )
        by_variant[variant_id] = {
            "report": variant.get("report"),
            "gap_counts": dict(gap_counts.most_common()),
            "family_failure_counts": dict(family_counts.most_common(24)),
            "wrong_answer_shape_counts": dict(wrong_shape_counts.most_common(24)),
            "strategy_failure_counts": dict(strategy_failure_counts.most_common(12)),
            "examples": examples,
        }
    return {
        "policy": "project_theseus_private_route_independence_residual_summary_v0",
        "by_variant": by_variant,
        "top_symliquid_plan_head_only_failures": dict(plan_head_failures.most_common(16)),
        "uses_public_data": False,
        "teacher_used": False,
        "external_inference_calls": 0,
    }


def dogfood_trace_summary() -> dict[str, Any]:
    readiness_path = ROOT / "reports" / "dogfood_trace_readiness.json"
    readiness = read_json(readiness_path) if readiness_path.exists() else {}
    candidates = []
    for pattern_root in [ROOT / "reports", ROOT / "data", ROOT / "runtime"]:
        if not pattern_root.exists():
            continue
        for path in pattern_root.rglob("*"):
            if len(candidates) >= 40:
                break
            name = path.name.lower()
            if path.is_file() and ("dogfood" in name or "daily_use" in name or "daily-use" in name or "usage_trace" in name):
                candidates.append(rel(path))
    if readiness:
        summary = dict_or_empty(readiness.get("summary"))
        return {
            "checked": readiness.get("trigger_state") == "GREEN",
            "readiness_report": rel(readiness_path),
            "readiness_trigger_state": readiness.get("trigger_state"),
            "capture_enabled": bool(summary.get("capture_enabled")),
            "training_enabled": bool(summary.get("training_enabled")),
            "raw_text_capture_enabled": bool(summary.get("raw_text_capture_enabled")),
            "candidate_trace_artifacts": candidates[:40],
            "trained_on_user_text": bool(summary.get("trained_on_user_text")),
            "external_inference_calls": int(readiness.get("external_inference_calls") or 0),
            "score_semantics": readiness.get("score_semantics"),
        }
    if not candidates:
        return {
            "checked": False,
            "not_checked_reason": "no governed real dogfood trace artifact found under reports/, data/, or runtime/",
            "candidate_trace_artifacts": [],
            "trained_on_user_text": False,
            "external_inference_calls": 0,
        }
    return {
        "checked": False,
        "not_checked_reason": "candidate trace artifacts exist, but no governed local token-decoder compatibility runner is configured for them yet",
        "candidate_trace_artifacts": candidates[:40],
        "trained_on_user_text": False,
        "external_inference_calls": 0,
    }


def planned_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    return {
        "policy": "project_theseus_neural_seed_token_decoder_route_independence_ablation_v0",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "config": args.config,
        "starting_git_commit": git_commit(),
        "seeds": parse_seeds(args.seeds),
        "summary": {
            "execute_required": True,
            "command": "python3 scripts/neural_seed_token_decoder_ablation.py --execute",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def run_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2400:],
        "stderr_tail": proc.stderr[-2400:],
    }


def git_commit() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def parse_seeds(value: str) -> list[int]:
    out = []
    for part in str(value or "").replace(";", ",").split(","):
        if part.strip():
            out.append(int(part.strip()))
    return out or [23, 29, 31, 37, 41]


def numbers(values: Any) -> list[float]:
    out = []
    for value in values:
        parsed = number(value)
        if parsed is not None:
            out.append(parsed)
    return out


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def subtract(left: Any, right: Any) -> float | None:
    left_num = number(left)
    right_num = number(right)
    if left_num is None or right_num is None:
        return None
    return round(left_num - right_num, 6)


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def stdev(values: list[float]) -> float | None:
    return round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0 if values else None


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


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


def render_markdown(report: dict[str, Any]) -> str:
    attribution = dict_or_empty(report.get("attribution"))
    lines = [
        "# Neural Seed Token Decoder Ablation",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- seeds: `{report.get('seeds')}`",
        f"- likely_primary_source: `{attribution.get('likely_primary_source')}`",
        f"- symliquid_learned_internal_routing_delta_vs_no_internal_beam_off: "
        f"`{attribution.get('symliquid_learned_internal_routing_delta_vs_no_internal_beam_off')}`",
        f"- symliquid_shared_beam_delta_vs_internal_beam_off: "
        f"`{attribution.get('symliquid_shared_beam_delta_vs_internal_beam_off')}`",
        f"- symliquid_beam_off_recovery_delta_vs_pre_beam_reference: "
        f"`{attribution.get('symliquid_beam_off_recovery_delta_vs_pre_beam_reference')}`",
        f"- symliquid_visible_text_prototype_memory_delta: "
        f"`{attribution.get('symliquid_visible_text_prototype_memory_delta')}`",
        f"- symliquid_plan_head_only_mean: `{attribution.get('symliquid_plan_head_only_mean')}`",
        f"- symliquid_plan_head_only_delta_vs_no_internal: "
        f"`{attribution.get('symliquid_plan_head_only_delta_vs_no_internal')}`",
        f"- symliquid_route_dropout_half_mean: `{attribution.get('symliquid_route_dropout_half_mean')}`",
        f"- symliquid_full_visible_text_prototype_selected_rate: "
        f"`{attribution.get('symliquid_full_visible_text_prototype_selected_rate')}`",
        f"- symliquid_full_contract_fingerprint_selected_rate: "
        f"`{attribution.get('symliquid_full_contract_fingerprint_selected_rate')}`",
        f"- symliquid_full_contract_feature_selected_rate: "
        f"`{attribution.get('symliquid_full_contract_feature_selected_rate')}`",
        f"- symliquid_dropout_visible_text_prototype_selected_rate: "
        f"`{attribution.get('symliquid_dropout_visible_text_prototype_selected_rate')}`",
        f"- symliquid_dropout_contract_feature_selected_rate: "
        f"`{attribution.get('symliquid_dropout_contract_feature_selected_rate')}`",
        f"- symliquid_no_visible_contract_fingerprint_selected_rate: "
        f"`{attribution.get('symliquid_no_visible_contract_fingerprint_selected_rate')}`",
        f"- symliquid_no_visible_contract_feature_selected_rate: "
        f"`{attribution.get('symliquid_no_visible_contract_feature_selected_rate')}`",
        f"- mean_beam_on_selected_rate: `{attribution.get('mean_beam_on_selected_rate')}`",
        f"- mean_beam_off_learned_route_selected_rate: "
        f"`{attribution.get('mean_beam_off_learned_route_selected_rate')}`",
        "",
        "## Variants",
        "",
    ]
    for row in report.get("variant_rows", []):
        lines.append(
            f"- `{row.get('id')}`: sym=`{row.get('symliquid_sts_on_mean')}`, "
            f"tx=`{row.get('transformer_sts_on_mean')}`, "
            f"gap=`{row.get('symliquid_minus_transformer_sts_on_mean')}`, "
            f"sym_beam_selected=`{row.get('symliquid_visible_contract_semantic_beam_selected_mean')}`, "
            f"sym_learned_route=`{row.get('symliquid_learned_internal_semantic_route_selected_mean')}`, "
            f"sym_route_strategies=`{row.get('symliquid_learned_internal_semantic_route_strategy_selected_means')}`, "
            f"fallback_zero=`{row.get('fallback_return_rate_zero')}`, "
            f"trigger=`{row.get('trigger_state')}`"
        )
    residuals = dict_or_empty(report.get("residual_failure_summary"))
    if residuals:
        lines.extend(["", "## Residual Blockers", ""])
        for key, count in dict_or_empty(residuals.get("top_symliquid_plan_head_only_failures")).items():
            lines.append(f"- `{key}`: `{count}`")
    dogfood = dict_or_empty(report.get("dogfood_trace_compatibility"))
    if dogfood:
        lines.extend(["", "## Dogfood Trace Compatibility", ""])
        lines.append(f"- checked: `{dogfood.get('checked')}`")
        lines.append(f"- not_checked_reason: `{dogfood.get('not_checked_reason')}`")
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    lines.extend(["", "## Semantics", "", str(report.get("score_semantics", "")), ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
