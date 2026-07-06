#!/usr/bin/env python3
"""Summarize the current overnight neural-seed handoff.

The report is intentionally conservative: it reads existing gates and states
what is cleared, what remains blocked, and what the next bounded local run
should do. It does not launch training, spend public calibration, or call the
teacher.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/neural_seed_overnight_handoff.json")
    parser.add_argument("--markdown-out", default="reports/neural_seed_overnight_handoff.md")
    args = parser.parse_args()

    state = load_state()
    report = build_report(state)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0


def load_state() -> dict[str, Any]:
    return {
        "external": read_json(REPORTS / "external_inference_audit.json"),
        "teacher_distillation": read_json(REPORTS / "teacher_distillation_gate.json"),
        "model_growth": read_json(REPORTS / "model_growth_gate.json"),
        "overnight": read_json(REPORTS / "overnight_learning_readiness.json"),
        "architecture_delta": read_json(REPORTS / "causal_architecture_delta_loop.json"),
        "architecture_governance": read_json(REPORTS / "architecture_experiment_governance.json"),
        "genesis": read_json(REPORTS / "genesis_kernel" / "report.json"),
        "adapter": read_json(REPORTS / "benchmark_adapter_factory.json"),
        "code_forge": read_json(REPORTS / "code_residual_forge.json"),
        "local_repair": read_latest_json(REPORTS, "local_code_repair_organism_*_seed*.json"),
        "multi_stream": read_latest_json(REPORTS, "multi_stream_code_pressure_*_seed*.json"),
        "self_edit": read_json(REPORTS / "self_edit_experiment_lane.json"),
        "memory": read_json(REPORTS / "long_horizon_memory_probe.json"),
        "sparkstream": read_json(REPORTS / "sparkstream_status.json"),
        "frontier": read_json(REPORTS / "frontier_policy_status.json"),
        "neural_seed_growth": read_json(REPORTS / "neural_seed_growth_gate.json"),
    }


def build_report(state: dict[str, Any]) -> dict[str, Any]:
    model_growth_missing = list(state["model_growth"].get("missing_evidence") or [])
    model_growth_hard = list(state["model_growth"].get("hard_blockers") or [])
    cleared = cleared_evidence(state)
    neural_summary = neural_seed_summary(state["neural_seed_growth"])
    next_step = neural_seed_next_step(neural_summary)
    offline_runbook = {
        "one_cycle_smoke": "python3 scripts/sparkstream_daemon.py --offline --once",
        "bounded_overnight_dry_run": "python3 scripts/sparkstream_daemon.py --offline --duration-hours 8",
        "bounded_overnight_execute": (
            "python3 scripts/sparkstream_daemon.py --offline --execute "
            "--duration-hours 8"
        ),
        "notes": [
            "--offline forces allow_teacher=false and allow_network_fetch=false even when the policy defaults are permissive",
            "use --execute only when local power, cooling, and disk headroom are acceptable",
            "public promotion remains blocked until the public transfer floor is honestly met",
        ],
    }
    return {
        "policy": "project_theseus_neural_seed_overnight_handoff_v1",
        "created_utc": now(),
        "north_star": (
            "Private, locally trained daily-use model; zero external inference at serving time; "
            "teacher share trends to zero."
        ),
        "overnight_launch_ready": bool(state["overnight"].get("overnight_launch_ready")),
        "overnight_trigger_state": state["overnight"].get("trigger_state"),
        "promotion_ready": bool(state["overnight"].get("promotion_ready")),
        "model_growth_allowed": bool(state["model_growth"].get("model_growth_allowed")),
        "model_growth_hard_blockers": model_growth_hard,
        "model_growth_missing_evidence": model_growth_missing,
        "cleared_evidence": cleared,
        "remaining_blockers": remaining_blockers(state),
        "offline_boundary": {
            "external_inference_audit_ok": bool(state["external"].get("ok")),
            "teacher_distillation_allowed": bool(state["teacher_distillation"].get("distillation_allowed")),
            "last_cycle_allow_teacher": get_path(state, ["frontier", "allow_teacher"], None),
            "last_cycle_allow_network_fetch": get_path(state, ["frontier", "allow_network_fetch"], None),
            "last_cycle_offline": get_path(state, ["frontier", "offline"], None),
            "sparkstream_phase": state["sparkstream"].get("phase"),
        },
        "architecture_delta": architecture_summary(state["architecture_delta"]),
        "neural_seed_experiment": neural_summary,
        "neural_seed_next_step": next_step,
        "offline_runbook": offline_runbook,
        "deleted_generated_artifacts": [
            "reports/post_v4_generalization_autopilot_v1_archive"
        ],
        "next_best_action": next_best_action(state),
        "external_inference_calls": 0,
    }


def cleared_evidence(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    add = rows.append
    add({"name": "offline_daemon_smoke", "status": state["sparkstream"].get("phase"), "passed": state["sparkstream"].get("phase") == "cycle_complete"})
    add({"name": "overnight_readiness", "status": state["overnight"].get("trigger_state"), "passed": bool(state["overnight"].get("overnight_launch_ready"))})
    add({"name": "architecture_ladder_private_delta", **architecture_summary(state["architecture_delta"])})
    add({"name": "code_residual_forge", "status": state["code_forge"].get("trigger_state"), "passed": state["code_forge"].get("trigger_state") == "GREEN"})
    add({"name": "multi_stream_code_pressure", "status": state["multi_stream"].get("status"), "passed": float(get_path(state, ["multi_stream", "summary", "pass_rate_delta"], 0.0) or 0.0) > 0.0})
    add({"name": "local_code_repair_organism", "status": state["local_repair"].get("trigger_state"), "passed": float(get_path(state, ["local_repair", "summary", "pass_rate_delta"], 0.0) or 0.0) > 0.0})
    add({"name": "self_edit_experiment_lane", "status": state["self_edit"].get("trigger_state"), "passed": state["self_edit"].get("trigger_state") == "GREEN"})
    add({"name": "long_horizon_memory_probe", "status": state["memory"].get("trigger_state"), "passed": float(get_path(state, ["memory", "score", "overall"], 0.0) or 0.0) >= 1.0})
    growth_checks = {
        row.get("name"): row
        for row in state["model_growth"].get("checks", [])
        if isinstance(row, dict)
    }
    for name in ["adapter_pressure_available", "genesis_artifact_substrate_ready"]:
        row = growth_checks.get(name, {})
        if row:
            add({"name": name, "status": "cleared" if row.get("passed") else "blocked", "passed": bool(row.get("passed"))})
    return rows


def remaining_blockers(state: dict[str, Any]) -> list[dict[str, Any]]:
    checks = {
        row.get("name"): row
        for row in state["model_growth"].get("checks", [])
        if isinstance(row, dict)
    }
    names = list(state["model_growth"].get("hard_blockers") or []) + list(
        state["model_growth"].get("missing_evidence") or []
    )
    blockers = []
    for name in names:
        row = checks.get(name, {})
        blockers.append(
            {
                "name": name,
                "status": "blocked",
                "severity": row.get("severity"),
                "evidence": row.get("evidence", {}),
                "next_fix": fix_for_blocker(name),
            }
        )
    return blockers


def fix_for_blocker(name: str) -> str:
    fixes = {
        "adapter_pressure_available": "Run or repair smoke adapters until at least one benchmark adapter card is ready.",
        "genesis_artifact_substrate_ready": "Fix Genesis source report loading and high-risk claim accounting.",
        "external_inference_teacher_only": "Remove or gate external inference violations before overnight work.",
    }
    return fixes.get(name, "Regenerate the owning gate report and clear this model-growth blocker.")


def neural_seed_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "present": False,
            "spec_ready": False,
            "neural_student_ready": False,
            "path": "reports/neural_seed_growth_gate.json",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "spec_ready": bool(report.get("spec_ready") or summary.get("spec_ready")),
        "execute_allowed": bool(report.get("execute_allowed") or summary.get("execute_allowed")),
        "neural_student_ready": bool(report.get("neural_student_ready") or summary.get("neural_student_ready")),
        "matched_arms": summary.get("matched_arms"),
        "substrate_comparator": summary.get("substrate_comparator"),
        "code_proposer_comparator": summary.get("code_proposer_comparator"),
        "code_proposer_gap_report": summary.get("code_proposer_gap_report"),
        "token_decoder_comparator": summary.get("token_decoder_comparator"),
        "token_decoder_multiseed": summary.get("token_decoder_multiseed"),
        "semantic_plan_gap_audit": summary.get("semantic_plan_gap_audit"),
        "architecture_sweep": summary.get("architecture_sweep"),
        "residual_mining": summary.get("residual_mining"),
        "primary_backend": get_path(report, ["macos_constraints", "primary_apple_silicon_backend"], None),
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_growth_gate.json",
    }


def neural_seed_next_step(neural_summary: dict[str, Any]) -> dict[str, Any]:
    token_multiseed = neural_summary.get("token_decoder_multiseed")
    token_multiseed = token_multiseed if isinstance(token_multiseed, dict) else {}
    if token_multiseed.get("multiseed_smoke_ready"):
        gap_closed = bool(token_multiseed.get("symliquid_gap_closed"))
        if gap_closed:
            return {
                "id": "neural_seed_shared_semantic_behavior_repair",
                "summary": (
                    "The five-seed semantic-slot token-decoder smoke mostly closes the SymLiquid routing gap "
                    f"(mean SymLiquid={token_multiseed.get('symliquid_sts_on_mean')}, transformer={token_multiseed.get('transformer_sts_on_mean')}, "
                    f"gap={token_multiseed.get('symliquid_minus_transformer_sts_on_mean')}). Next, repair shared low absolute "
                    "behavior through renderer/body semantics, residual-family coverage, and a small capacity ladder while keeping "
                    "fallback returns, public calibration, teacher calls, and promotion locked."
                ),
                "must_not_do": [
                    "no public calibration spend",
                    "no teacher calls or distillation",
                    "no model promotion",
                    "no bulk data fetches",
                    "no fallback terminal returns counted as capability",
                ],
                "required_inputs": [
                    "reports/neural_seed_token_decoder_multiseed_smoke.json",
                    "reports/neural_seed_semantic_plan_gap_audit*.json",
                    "reports/neural_seed_token_decoder_candidates_seed_*.jsonl",
                    "private verifier wrong-answer residuals",
                ],
                "measurement_contract": [
                    "keep five or more seeds before claiming substrate movement",
                    "fallback-return use must remain exactly zero",
                    "raw syntax and semantic-plan support must remain nonzero for both arms",
                    "track absolute verifier pass rate, not only SymLiquid-vs-transformer gap",
                    "do not use eval tests or solutions as generation features",
                ],
                "multiseed_summary": token_multiseed,
            }
    token_comparator = neural_summary.get("token_decoder_comparator")
    token_comparator = token_comparator if isinstance(token_comparator, dict) else {}
    if token_comparator.get("token_decoder_smoke_ready"):
        syntax_rates = token_comparator.get("syntax_pass_rate_sts_on") if isinstance(token_comparator.get("syntax_pass_rate_sts_on"), dict) else {}
        raw_syntax_rates = token_comparator.get("raw_syntax_pass_rate_sts_on") if isinstance(token_comparator.get("raw_syntax_pass_rate_sts_on"), dict) else {}
        fallback_rates = token_comparator.get("grammar_repair_fallback_rate_sts_on") if isinstance(token_comparator.get("grammar_repair_fallback_rate_sts_on"), dict) else {}
        skeleton_rates = token_comparator.get("statement_skeleton_render_rate_sts_on") if isinstance(token_comparator.get("statement_skeleton_render_rate_sts_on"), dict) else {}
        semantic_rates = token_comparator.get("semantic_slot_render_rate_sts_on") if isinstance(token_comparator.get("semantic_slot_render_rate_sts_on"), dict) else {}
        semantic_supported_rates = token_comparator.get("semantic_plan_supported_rate_sts_on") if isinstance(token_comparator.get("semantic_plan_supported_rate_sts_on"), dict) else {}
        by_arm = token_comparator.get("by_arm") if isinstance(token_comparator.get("by_arm"), dict) else {}
        verifier_rates = {
            arm: get_path(row, ["sts_on_verifier_pass_rate"], 0.0)
            for arm, row in by_arm.items()
            if isinstance(row, dict)
        }
        syntax_repaired = bool(syntax_rates) and all(float(value or 0.0) > 0.0 for value in syntax_rates.values())
        raw_syntax_nonzero = bool(raw_syntax_rates) and all(float(value or 0.0) > 0.0 for value in raw_syntax_rates.values())
        all_fallback = bool(fallback_rates) and all(float(value or 0.0) >= 0.99 for value in fallback_rates.values())
        no_fallback = bool(fallback_rates) and all(float(value or 0.0) == 0.0 for value in fallback_rates.values())
        skeleton_rendered = bool(skeleton_rates) and all(float(value or 0.0) > 0.0 for value in skeleton_rates.values())
        semantic_rendered = bool(semantic_rates) and all(float(value or 0.0) > 0.0 for value in semantic_rates.values())
        semantic_supported = bool(semantic_supported_rates) and all(float(value or 0.0) > 0.0 for value in semantic_supported_rates.values())
        behavior_zero = bool(verifier_rates) and all(float(value or 0.0) <= 0.0 for value in verifier_rates.values())
        behavior_nonzero = bool(verifier_rates) and any(float(value or 0.0) > 0.0 for value in verifier_rates.values())
        if syntax_repaired and all_fallback and behavior_zero:
            residual_mining = neural_summary.get("residual_mining") if isinstance(neural_summary.get("residual_mining"), dict) else {}
            architecture_sweep = neural_summary.get("architecture_sweep") if isinstance(neural_summary.get("architecture_sweep"), dict) else {}
            return {
                "id": "neural_seed_token_decoder_body_structure_repair",
                "summary": (
                    "Grammar-constrained repair now produces syntactically valid candidates, but raw syntax is still "
                    "zero and both arms reach behavior only through shape-compatible fallback returns. The proposer "
                    f"seed sweep is ready={architecture_sweep.get('seed_sweep_ready')} and residual mining found "
                    f"{residual_mining.get('symliquid_only_win_count')} SymLiquid-only wins. Next, train and constrain "
                    "the decoder to emit non-fallback statement structure before scaling or calibrating."
                ),
                "must_not_do": [
                    "no public calibration spend",
                    "no teacher calls or distillation",
                    "no model promotion",
                    "no bulk data fetches",
                    "no long unattended training on a fallback-only decoder",
                ],
                "required_inputs": [
                    "reports/neural_seed_token_decoder_comparator.json",
                    "reports/neural_seed_token_decoder_candidates.jsonl",
                    "grammar_repair fallback metrics",
                    "private verifier wrong-answer residuals",
                ],
                "measurement_contract": [
                    "same private train/eval split and STS controls",
                    "both arms must emit token-decoded candidate rows, not body templates",
                    "report raw syntax, repaired syntax, fallback rate, verifier pass, and wrong-answer residuals",
                    "do not count fallback-only syntax as capability",
                    "keep model growth and teacher distillation locked until separate gates allow them",
                ],
            }
        if syntax_repaired and raw_syntax_nonzero and no_fallback and semantic_rendered and semantic_supported and behavior_nonzero:
            residual_mining = neural_summary.get("residual_mining") if isinstance(neural_summary.get("residual_mining"), dict) else {}
            return {
                "id": "neural_seed_semantic_plan_gap_repair",
                "summary": (
                    "The token decoder now emits non-fallback semantic slots and reaches nonzero private verifier "
                    f"behavior ({verifier_rates}). The matched transformer control leads; next, repair SymLiquid semantic-plan "
                    "routing and residual-family plan specificity without public calibration, teacher calls, or fallback returns."
                ),
                "must_not_do": [
                    "no public calibration spend",
                    "no teacher calls or distillation",
                    "no model promotion",
                    "no bulk data fetches",
                    "no fallback terminal returns counted as capability",
                ],
                "required_inputs": [
                    "reports/neural_seed_token_decoder_comparator.json",
                    "reports/neural_seed_token_decoder_candidates.jsonl",
                    "reports/neural_seed_residual_mining.json",
                    "private verifier wrong-answer residuals",
                ],
                "measurement_contract": [
                    "fallback-return use must remain exactly zero",
                    "raw syntax and semantic-plan support must remain nonzero for both arms",
                    "compare SymLiquid and transformer under matched parameter/compute budget",
                    "prioritize residual families from mining, especially collection_logic, state_machine, interface_fidelity, project_memory, long_horizon_plan, device_routing, and stdin parsing",
                    "do not use eval tests or solutions as generation features",
                ],
                "residual_pressure": residual_mining.get("next_private_pressure"),
            }
        if syntax_repaired and raw_syntax_nonzero and no_fallback and skeleton_rendered:
            residual_mining = neural_summary.get("residual_mining") if isinstance(neural_summary.get("residual_mining"), dict) else {}
            return {
                "id": "neural_seed_statement_skeleton_semantic_repair",
                "summary": (
                    "The token decoder now emits syntactically valid non-fallback statement-structure candidates. "
                    f"Fallback-return rate is zero and residual mining shows {residual_mining.get('symliquid_only_win_count')} "
                    "SymLiquid-only wins plus shared both-fail families. Next, improve semantic body updates and wrong-answer "
                    "residual handling under the same matched private verifier before any scale, teacher, or public calibration claim."
                ),
                "must_not_do": [
                    "no public calibration spend",
                    "no teacher calls or distillation",
                    "no model promotion",
                    "no bulk data fetches",
                    "no fallback terminal returns counted as capability",
                ],
                "required_inputs": [
                    "reports/neural_seed_token_decoder_comparator.json",
                    "reports/neural_seed_token_decoder_candidates.jsonl",
                    "reports/neural_seed_residual_mining.json",
                    "private verifier wrong-answer residuals",
                ],
                "measurement_contract": [
                    "same private train/eval split and STS controls",
                    "both arms must emit token-decoded candidate rows, not body templates",
                    "fallback-return use must remain exactly zero",
                    "report raw syntax, repaired syntax, verifier pass, and wrong-answer residuals separately",
                    "target semantic updates for SymLiquid-only and both-fail families without using eval tests or solutions as generation features",
                ],
            }
        return {
            "id": "neural_seed_grammar_constrained_token_decoder_repair",
            "summary": (
                "The matched token-decoder smoke is recorded and both arms currently fail at syntax. "
                "Next, repair the tokenization/detokenization and grammar-constrained decode path before "
                "doing more training or public calibration."
            ),
            "must_not_do": [
                "no public calibration spend",
                "no teacher calls or distillation",
                "no model promotion",
                "no bulk data fetches",
                "no long unattended training on the current invalid-token decoder",
            ],
            "required_inputs": [
                "reports/neural_seed_code_proposer_gap_report.json",
                "reports/neural_seed_token_decoder_comparator.json",
                "reports/neural_seed_token_decoder_candidates.jsonl",
                "private verifier syntax-stage failures",
            ],
            "measurement_contract": [
                "same private train/eval split and STS controls",
                "both arms emit token-decoded candidate rows, not body templates",
                "syntax pass must be explicitly measured before behavior pass",
                "do not count syntax-repair fallback as capability without separate reporting",
                "keep model growth and teacher distillation locked until separate gates allow them",
            ],
        }
    code_comparator = neural_summary.get("code_proposer_comparator")
    code_comparator = code_comparator if isinstance(code_comparator, dict) else {}
    if code_comparator.get("code_proposer_smoke_ready"):
        return {
            "id": "neural_seed_token_decoder_parity_and_symliquid_gap_repair",
            "summary": (
                "The first true private candidate-code comparator is recorded. Next, replace "
                "the body-template selector with matched token-level code decoders for both "
                "arms and use the current report to focus SymLiquid gap repair."
            ),
            "must_not_do": [
                "no public calibration spend",
                "no teacher calls or distillation",
                "no model promotion",
                "no bulk data fetches",
                "no unattended long training until the token-decoder adapter gate is explicit",
            ],
            "required_inputs": [
                "reports/neural_seed_code_proposer_comparator.json",
                "reports/neural_seed_code_proposer_candidates.jsonl",
                "private code LM verifier contract",
                "same private train/eval split and STS controls",
            ],
            "measurement_contract": [
                "both arms emit candidate code rows, not labels",
                "same verifier, fanout, ranker, seed, token budget, and training step budget",
                "report syntax pass, verifier pass, accepted candidate rate, residual movement, and regressions",
                "report whether MLX is used or why it is unavailable",
                "keep model growth and teacher distillation locked until separate gates allow them",
            ],
        }
    comparator = neural_summary.get("substrate_comparator")
    comparator = comparator if isinstance(comparator, dict) else {}
    if comparator.get("substrate_smoke_ready") and not comparator.get("code_proposer_comparison_ready"):
        return {
            "id": "neural_seed_full_code_proposer_adapter_parity",
            "summary": (
                "The first private substrate smoke is recorded. Next, promote it to a true "
                "code-proposer comparison by adding SymLiquid and transformer adapters that "
                "emit candidate code rows into the same verifier/fanout/ranker path."
            ),
            "must_not_do": [
                "no public calibration spend",
                "no teacher calls or distillation",
                "no model promotion",
                "no bulk data fetches",
            ],
            "required_inputs": [
                "reports/neural_seed_substrate_comparator.json",
                "private code LM verifier contract",
                "shared candidate-code row schema",
                "same private train/eval split and STS controls",
            ],
            "measurement_contract": [
                "same private task rows for both code-proposer adapters",
                "same fanout seeds, verifier, and ranker",
                "matched parameter, token, step, and wall-clock accounting",
                "report syntax pass, verifier pass, accepted candidate rate, residual movement, and regressions",
                "keep model growth and teacher distillation locked until separate gates allow them",
            ],
        }
    return {
        "id": "neural_seed_symliquid_vs_matched_control",
        "summary": (
            "Build the measured small neural proposer lane behind the existing "
            "verifier/fanout/STS harness, with a SymLiquid substrate arm and a "
            "parameter/compute-matched transformer control."
        ),
        "must_not_do": [
            "no public calibration spend",
            "no teacher calls",
            "no arbitrary remote execution",
            "no new saturated private ecology/shadow/frontier suites",
        ],
        "required_inputs": [
            "private residual code curriculum",
            "STS control rows",
            "current verifier/fanout harness",
            "causal architecture delta report",
        ],
        "measurement_contract": [
            "same private train/eval rows for both arms",
            "matched parameter and compute budget",
            "report wall-clock, backend, memory, and accepted candidate rate",
            "report verifier pass rate, residual class movement, and regressions",
            "do not promote without public-calibration operator unlock after private gate",
        ],
    }


def next_best_action(state: dict[str, Any]) -> str:
    growth_action = state["model_growth"].get("next_action") or "Keep model growth locked until the owning gates allow it."
    hard = state["model_growth"].get("hard_blockers") or []
    missing = state["model_growth"].get("missing_evidence") or []
    neural = neural_seed_summary(state["neural_seed_growth"])
    token_comparator = neural.get("token_decoder_comparator")
    token_comparator = token_comparator if isinstance(token_comparator, dict) else {}
    token_multiseed = neural.get("token_decoder_multiseed")
    token_multiseed = token_multiseed if isinstance(token_multiseed, dict) else {}
    code_comparator = neural.get("code_proposer_comparator")
    code_comparator = code_comparator if isinstance(code_comparator, dict) else {}
    comparator = neural.get("substrate_comparator")
    comparator = comparator if isinstance(comparator, dict) else {}
    if hard or missing:
        return growth_action
    if not neural.get("spec_ready"):
        return "Write the measured SymLiquid-vs-matched-control neural seed spec; keep growth and teacher distillation locked."
    if token_multiseed.get("multiseed_smoke_ready") and token_multiseed.get("symliquid_gap_closed"):
        return (
            "The five-seed token-decoder smoke mostly closes the SymLiquid semantic-plan routing gap. Next, improve "
            "absolute private verifier behavior with shared semantic renderer/body repairs and a small capacity ladder; "
            "keep fallback returns, public calibration, teacher, and promotion locked."
        )
    if token_comparator.get("token_decoder_smoke_ready"):
        syntax_rates = token_comparator.get("syntax_pass_rate_sts_on") if isinstance(token_comparator.get("syntax_pass_rate_sts_on"), dict) else {}
        raw_syntax_rates = token_comparator.get("raw_syntax_pass_rate_sts_on") if isinstance(token_comparator.get("raw_syntax_pass_rate_sts_on"), dict) else {}
        fallback_rates = token_comparator.get("grammar_repair_fallback_rate_sts_on") if isinstance(token_comparator.get("grammar_repair_fallback_rate_sts_on"), dict) else {}
        skeleton_rates = token_comparator.get("statement_skeleton_render_rate_sts_on") if isinstance(token_comparator.get("statement_skeleton_render_rate_sts_on"), dict) else {}
        semantic_rates = token_comparator.get("semantic_slot_render_rate_sts_on") if isinstance(token_comparator.get("semantic_slot_render_rate_sts_on"), dict) else {}
        semantic_supported_rates = token_comparator.get("semantic_plan_supported_rate_sts_on") if isinstance(token_comparator.get("semantic_plan_supported_rate_sts_on"), dict) else {}
        by_arm = token_comparator.get("by_arm") if isinstance(token_comparator.get("by_arm"), dict) else {}
        verifier_rates = [
            float(get_path(row, ["sts_on_verifier_pass_rate"], 0.0) or 0.0)
            for row in by_arm.values()
            if isinstance(row, dict)
        ]
        if (
            syntax_rates
            and all(float(value or 0.0) > 0.0 for value in syntax_rates.values())
            and fallback_rates
            and all(float(value or 0.0) >= 0.99 for value in fallback_rates.values())
            and verifier_rates
            and all(value <= 0.0 for value in verifier_rates)
        ):
            return (
                "Syntax repair is now working, but the token decoder is fallback-only and still behavior-zero. "
                "Next, repair learned statement/body structure under the same matched SymLiquid-vs-transformer "
                "private verifier path; keep public calibration, teacher, and promotion locked."
            )
        if (
            syntax_rates
            and all(float(value or 0.0) > 0.0 for value in syntax_rates.values())
            and raw_syntax_rates
            and all(float(value or 0.0) > 0.0 for value in raw_syntax_rates.values())
            and fallback_rates
            and all(float(value or 0.0) == 0.0 for value in fallback_rates.values())
            and semantic_rates
            and all(float(value or 0.0) > 0.0 for value in semantic_rates.values())
            and semantic_supported_rates
            and all(float(value or 0.0) > 0.0 for value in semantic_supported_rates.values())
            and verifier_rates
            and any(value > 0.0 for value in verifier_rates)
        ):
            return (
                "Semantic-slot decoding is now non-fallback and behavior-positive. Next, repair the SymLiquid "
                "semantic-plan routing gap against the matched transformer control while keeping public calibration, "
                "teacher, promotion, and fallback returns locked."
            )
        if (
            syntax_rates
            and all(float(value or 0.0) > 0.0 for value in syntax_rates.values())
            and raw_syntax_rates
            and all(float(value or 0.0) > 0.0 for value in raw_syntax_rates.values())
            and fallback_rates
            and all(float(value or 0.0) == 0.0 for value in fallback_rates.values())
            and skeleton_rates
            and all(float(value or 0.0) > 0.0 for value in skeleton_rates.values())
        ):
            return (
                "Non-fallback statement-structure decoding is now syntactically alive. Next, repair semantic "
                "body updates and wrong-answer residuals under the same matched SymLiquid-vs-transformer private "
                "verifier path; keep fallback returns, public calibration, teacher, and promotion locked."
            )
        return (
            "Token-decoder comparator is recorded, but both arms are currently blocked at syntax. "
            "Repair grammar-constrained decoding/tokenization before more training; keep model growth, "
            "teacher distillation, and public calibration locked."
        )
    if code_comparator.get("code_proposer_smoke_ready"):
        return (
            "Private candidate-code comparator is recorded; next implement matched token-level code decoders "
            "and repair the SymLiquid gap shown by the smoke. Keep model growth, teacher distillation, "
            "and public calibration locked."
        )
    if comparator.get("substrate_smoke_ready") and not comparator.get("code_proposer_comparison_ready"):
        return (
            "Private substrate smoke is recorded; next implement full candidate-code adapters for "
            "SymLiquid and transformer arms under the same verifier/fanout/ranker path. Keep model "
            "growth, teacher distillation, and public calibration locked."
        )
    if not bool(state["model_growth"].get("model_growth_allowed")):
        return f"Neural seed spec is ready for review, but do not train or promote yet: {growth_action}"
    return "Run only the bounded neural seed smoke behind verifier/fanout/STS gates; public calibration and teacher distillation remain separately locked."


def architecture_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    failed_gates = [
        gate.get("gate") or gate.get("name")
        for gate in report.get("gates", [])
        if isinstance(gate, dict) and not gate.get("passed")
    ]
    return {
        "status": report.get("status"),
        "trigger_state": report.get("trigger_state"),
        "passed": report.get("status") == "completed_with_capability_delta" and report.get("trigger_state") == "GREEN" and not failed_gates,
        "best_target_delta": summary.get("best_target_delta"),
        "private_heldout_pass_rate_delta": summary.get("private_heldout_pass_rate_delta"),
        "private_receiver_eligible_task_rate_delta": summary.get("private_receiver_eligible_task_rate_delta"),
        "private_semantic_test_passed_task_rate_delta": summary.get("private_semantic_test_passed_task_rate_delta"),
        "public_task_count": summary.get("public_task_count"),
        "public_tests_or_solutions_used": summary.get("public_tests_or_solutions_used"),
        "failed_gates": failed_gates,
    }


def render_markdown(report: dict[str, Any]) -> str:
    cleared = "\n".join(
        f"- `{row['name']}`: {'PASS' if row.get('passed') else 'HOLD'} ({row.get('status')})"
        for row in report["cleared_evidence"]
    )
    blockers = "\n".join(
        f"- `{row['name']}`: {row['next_fix']}"
        for row in report["remaining_blockers"]
    )
    comparator = get_path(report, ["neural_seed_experiment", "substrate_comparator"], {})
    comparator_lines = "\n".join(
        [
            f"- present: `{comparator.get('present')}`",
            f"- comparison_level: `{comparator.get('comparison_level')}`",
            f"- substrate_smoke_ready: `{comparator.get('substrate_smoke_ready')}`",
            f"- code_proposer_comparison_ready: `{comparator.get('code_proposer_comparison_ready')}`",
            f"- best_sts_on_arm_by_verifier_pass_rate: `{comparator.get('best_sts_on_arm_by_verifier_pass_rate')}`",
            f"- symliquid_minus_transformer_sts_on_verifier_pass_rate: `{comparator.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
        ]
    )
    code_comparator = get_path(report, ["neural_seed_experiment", "code_proposer_comparator"], {})
    code_comparator_lines = "\n".join(
        [
            f"- present: `{code_comparator.get('present')}`",
            f"- comparison_level: `{code_comparator.get('comparison_level')}`",
            f"- code_proposer_smoke_ready: `{code_comparator.get('code_proposer_smoke_ready')}`",
            f"- candidate_rows: `{code_comparator.get('candidate_rows')}`",
            f"- best_sts_on_arm_by_verifier_pass_rate: `{code_comparator.get('best_sts_on_arm_by_verifier_pass_rate')}`",
            f"- symliquid_minus_transformer_sts_on_verifier_pass_rate: `{code_comparator.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
        ]
    )
    gap = get_path(report, ["neural_seed_experiment", "code_proposer_gap_report"], {})
    gap_lines = "\n".join(
        [
            f"- present: `{gap.get('present')}`",
            f"- gap_counts: `{gap.get('gap_counts')}`",
            f"- sts_repairs: `{gap.get('sts_repairs')}`",
            f"- sts_regressions: `{gap.get('sts_regressions')}`",
            f"- failure_cause_counts: `{gap.get('failure_cause_counts')}`",
        ]
    )
    token_comparator = get_path(report, ["neural_seed_experiment", "token_decoder_comparator"], {})
    token_comparator_lines = "\n".join(
        [
            f"- present: `{token_comparator.get('present')}`",
            f"- comparison_level: `{token_comparator.get('comparison_level')}`",
            f"- token_decoder_smoke_ready: `{token_comparator.get('token_decoder_smoke_ready')}`",
            f"- candidate_rows: `{token_comparator.get('candidate_rows')}`",
            f"- target_mode: `{token_comparator.get('target_mode')}`",
            f"- syntax_pass_rate_sts_on: `{token_comparator.get('syntax_pass_rate_sts_on')}`",
            f"- raw_syntax_pass_rate_sts_on: `{token_comparator.get('raw_syntax_pass_rate_sts_on')}`",
            f"- grammar_repair_fallback_rate_sts_on: `{token_comparator.get('grammar_repair_fallback_rate_sts_on')}`",
            f"- statement_skeleton_render_rate_sts_on: `{token_comparator.get('statement_skeleton_render_rate_sts_on')}`",
            f"- semantic_slot_render_rate_sts_on: `{token_comparator.get('semantic_slot_render_rate_sts_on')}`",
            f"- semantic_plan_supported_rate_sts_on: `{token_comparator.get('semantic_plan_supported_rate_sts_on')}`",
            f"- predicted_return_shape_rate_sts_on: `{token_comparator.get('predicted_return_shape_rate_sts_on')}`",
            f"- best_sts_on_arm_by_verifier_pass_rate: `{token_comparator.get('best_sts_on_arm_by_verifier_pass_rate')}`",
            f"- symliquid_minus_transformer_sts_on_verifier_pass_rate: `{token_comparator.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
            f"- symliquid_gap_vs_body_template: `{token_comparator.get('symliquid_gap_vs_body_template')}`",
            f"- transformer_gap_vs_body_template: `{token_comparator.get('transformer_gap_vs_body_template')}`",
        ]
    )
    token_multiseed = get_path(report, ["neural_seed_experiment", "token_decoder_multiseed"], {})
    token_multiseed_lines = "\n".join(
        [
            f"- present: `{token_multiseed.get('present')}`",
            f"- multiseed_smoke_ready: `{token_multiseed.get('multiseed_smoke_ready')}`",
            f"- completed_seed_count: `{token_multiseed.get('completed_seed_count')}`",
            f"- symliquid_sts_on_mean: `{token_multiseed.get('symliquid_sts_on_mean')}`",
            f"- transformer_sts_on_mean: `{token_multiseed.get('transformer_sts_on_mean')}`",
            f"- symliquid_minus_transformer_sts_on_mean: `{token_multiseed.get('symliquid_minus_transformer_sts_on_mean')}`",
            f"- symliquid_expected_plan_match_mean: `{token_multiseed.get('symliquid_expected_plan_match_mean')}`",
            f"- transformer_expected_plan_match_mean: `{token_multiseed.get('transformer_expected_plan_match_mean')}`",
            f"- symliquid_gap_closed: `{token_multiseed.get('symliquid_gap_closed')}`",
            f"- bottleneck: `{token_multiseed.get('bottleneck')}`",
            f"- winner_counts: `{token_multiseed.get('winner_counts')}`",
        ]
    )
    semantic_audit = get_path(report, ["neural_seed_experiment", "semantic_plan_gap_audit"], {})
    semantic_audit_lines = "\n".join(
        [
            f"- present: `{semantic_audit.get('present')}`",
            f"- semantic_plan_gap_audit_ready: `{semantic_audit.get('semantic_plan_gap_audit_ready')}`",
            f"- seed: `{semantic_audit.get('seed')}`",
            f"- gap_counts: `{semantic_audit.get('gap_counts')}`",
            f"- symliquid_private_eval_plan_match_rate: `{semantic_audit.get('symliquid_private_eval_plan_match_rate')}`",
            f"- transformer_private_eval_plan_match_rate: `{semantic_audit.get('transformer_private_eval_plan_match_rate')}`",
            f"- bottleneck: `{semantic_audit.get('bottleneck')}`",
        ]
    )
    architecture_sweep = get_path(report, ["neural_seed_experiment", "architecture_sweep"], {})
    architecture_sweep_lines = "\n".join(
        [
            f"- present: `{architecture_sweep.get('present')}`",
            f"- seed_sweep_ready: `{architecture_sweep.get('seed_sweep_ready')}`",
            f"- seed_count: `{architecture_sweep.get('seed_count')}`",
            f"- run_count: `{architecture_sweep.get('run_count')}`",
            f"- single_seed_claims_disallowed: `{architecture_sweep.get('single_seed_claims_disallowed')}`",
            f"- aggregate: `{architecture_sweep.get('aggregate')}`",
        ]
    )
    residual_mining = get_path(report, ["neural_seed_experiment", "residual_mining"], {})
    residual_mining_lines = "\n".join(
        [
            f"- present: `{residual_mining.get('present')}`",
            f"- residual_mining_ready: `{residual_mining.get('residual_mining_ready')}`",
            f"- symliquid_only_win_count: `{residual_mining.get('symliquid_only_win_count')}`",
            f"- transformer_only_win_count: `{residual_mining.get('transformer_only_win_count')}`",
            f"- both_fail_count: `{residual_mining.get('both_fail_count')}`",
            f"- next_private_pressure: `{residual_mining.get('next_private_pressure')}`",
        ]
    )
    if report.get("neural_seed_next_step", {}).get("id") in {
        "neural_seed_full_code_proposer_adapter_parity",
        "neural_seed_token_decoder_parity_and_symliquid_gap_repair",
        "neural_seed_grammar_constrained_token_decoder_repair",
        "neural_seed_token_decoder_body_structure_repair",
        "neural_seed_statement_skeleton_semantic_repair",
        "neural_seed_semantic_plan_gap_repair",
    }:
        command_label = "Immediate Command"
        if report.get("neural_seed_next_step", {}).get("id") == "neural_seed_token_decoder_body_structure_repair":
            command_text = "No unattended training command is recommended from this handoff; the next work is learned body-structure decoder repair."
        elif report.get("neural_seed_next_step", {}).get("id") == "neural_seed_statement_skeleton_semantic_repair":
            command_text = "No unattended training command is recommended from this handoff; the next work is semantic wrong-answer repair on the non-fallback statement decoder."
        elif report.get("neural_seed_next_step", {}).get("id") == "neural_seed_semantic_plan_gap_repair":
            command_text = "No unattended training command is recommended from this handoff; the next work is SymLiquid semantic-plan routing repair under the matched private verifier."
        else:
            command_text = "No unattended training command is recommended from this handoff; the next work is source grammar/token-decoder repair."
    else:
        command_label = "Offline Command"
        command_text = f"`{report['offline_runbook']['bounded_overnight_execute']}`"
    return "\n".join(
        [
            "# Neural Seed Overnight Handoff",
            "",
            f"- created_utc: `{report['created_utc']}`",
            f"- overnight_launch_ready: `{report['overnight_launch_ready']}`",
            f"- model_growth_allowed: `{report['model_growth_allowed']}`",
            f"- missing_evidence: `{', '.join(report['model_growth_missing_evidence']) or 'none'}`",
            "",
            "## Cleared Evidence",
            "",
            cleared,
            "",
            "## Remaining Blockers",
            "",
            blockers,
            "",
            "## Substrate Comparator",
            "",
            comparator_lines,
            "",
            "## Code-Proposer Comparator",
            "",
            code_comparator_lines,
            "",
            "## Code-Proposer Gap Report",
            "",
            gap_lines,
            "",
            "## Token Decoder Comparator",
            "",
            token_comparator_lines,
            "",
            "## Token Decoder Multi-Seed",
            "",
            token_multiseed_lines,
            "",
            "## Semantic Plan Gap Audit",
            "",
            semantic_audit_lines,
            "",
            "## Architecture Seed Sweep",
            "",
            architecture_sweep_lines,
            "",
            "## Residual Mining",
            "",
            residual_mining_lines,
            "",
            "## Next Step",
            "",
            report["neural_seed_next_step"]["summary"],
            "",
            f"## {command_label}",
            "",
            command_text,
            "",
        ]
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_latest_json(directory: Path, pattern: str) -> dict[str, Any]:
    matches = [path for path in directory.glob(pattern) if path.is_file()]
    if not matches:
        return {}
    return read_json(max(matches, key=lambda path: path.stat().st_mtime))


def get_path(data: Any, keys: list[str], default: Any = None) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
