#!/usr/bin/env python3
"""Gate the C1 correctness/RL/generator survival lane.

This gate is deliberately narrow. It records whether a bounded private
verifier-driven learned body-token experiment is clean enough to count as C1
architecture evidence. A clean falsifying wall is acceptable evidence here; it
is not a promotion, public-transfer, or learned-generation success claim.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_EXPERIMENTS = ROOT / "configs" / "correctness_in_loop_generator_experiments.json"
DEFAULT_REPLAY = REPORTS / "assistant_code_private_replay_probe.json"
DEFAULT_INTEGRITY = REPORTS / "candidate_integrity_strict_generator_mlx_decode_eval_15m_capacity_b64_param_nontrivial_top_level_smoke_v1.json"
DEFAULT_BLIND_AUDIT = REPORTS / "blind_information_flow_audit_strict_generator_mlx_decode_eval_15m_capacity_b64_param_nontrivial_top_level_smoke_v1.json"
DEFAULT_POLICY = REPORTS / "policy_optimization_program.json"
DEFAULT_GENERATION_MODE = REPORTS / "generation_mode_registry.json"
DEFAULT_FANOUT = REPORTS / "neural_seed_strict_generator_fanout_receipt.json"
DEFAULT_OUT = REPORTS / "correctness_generator_survival_lane_gate.json"

ALLOWED_ELIGIBLE_FAMILIES = {"learned_full_body_token", "transformer_hybrid", "symliquid"}
FORBIDDEN_LEARNED_CREDIT_FAMILIES = {
    "deterministic_tool",
    "fallback_or_template",
    "hand_authored_contract_body",
    "neural_action_selector",
    "private_ngram_body",
    "structural_adapter",
    "unknown",
}
FORBIDDEN_INFERENCE_FIELDS = {
    "category",
    "solution",
    "solution_expr",
    "solution_body",
    "tests",
    "hidden_tests",
    "expected",
    "answer",
    "source_task_id",
    "return_shape",
    "type_family",
    "required_constructs",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default=rel(DEFAULT_EXPERIMENTS))
    parser.add_argument("--replay", default=rel(DEFAULT_REPLAY))
    parser.add_argument("--integrity", default=rel(DEFAULT_INTEGRITY))
    parser.add_argument("--blind-audit", default=rel(DEFAULT_BLIND_AUDIT))
    parser.add_argument("--policy", default=rel(DEFAULT_POLICY))
    parser.add_argument("--generation-mode", default=rel(DEFAULT_GENERATION_MODE))
    parser.add_argument("--fanout", default=rel(DEFAULT_FANOUT))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    report = build_report(
        experiments=read_json(resolve(args.experiments)),
        replay=read_json(resolve(args.replay)),
        integrity=read_json(resolve(args.integrity)),
        blind_audit=read_json(resolve(args.blind_audit)),
        policy=read_json(resolve(args.policy)),
        generation_mode=read_json(resolve(args.generation_mode)),
        fanout=read_json(resolve(args.fanout)),
        evidence_refs=[
            rel(resolve(args.experiments)),
            rel(resolve(args.replay)),
            rel(resolve(args.integrity)),
            rel(resolve(args.blind_audit)),
            rel(resolve(args.policy)),
            rel(resolve(args.generation_mode)),
            rel(resolve(args.fanout)),
        ],
    )
    write_json(resolve(args.out), report)
    print(json.dumps({"trigger_state": report["trigger_state"], "summary": report["summary"]}, indent=2, sort_keys=True))
    return 1 if args.gate and report["trigger_state"] != "GREEN" else 0


def build_report(
    *,
    experiments: dict[str, Any],
    replay: dict[str, Any],
    integrity: dict[str, Any],
    blind_audit: dict[str, Any],
    policy: dict[str, Any],
    generation_mode: dict[str, Any],
    fanout: dict[str, Any],
    evidence_refs: list[str],
) -> dict[str, Any]:
    replay_summary = dict_value(replay.get("summary"))
    replay_rules = dict_value(replay.get("rules"))
    integrity_summary = dict_value(integrity.get("summary"))
    blind_summary = dict_value(blind_audit.get("summary"))
    blind_rules = dict_value(blind_audit.get("rules"))
    policy_summary = dict_value(policy.get("summary"))
    generation_summary = dict_value(generation_mode.get("summary"))
    fanout_summary = dict_value(fanout.get("summary"))
    experiments_summary = dict_value(experiments.get("summary"))
    eligible_families = set(str(x) for x in dict_value(replay_summary.get("eligible_family_counts")).keys())
    replayed_families = set(str(x) for x in dict_value(replay_summary.get("replayed_family_counts")).keys())
    integrity_families = set(str(x) for x in dict_value(integrity_summary.get("integrity_verified_by_family")).keys())
    forbidden_fields = set(str(x) for x in list_values(blind_rules.get("forbidden_inference_fields")))
    semantic_wall = semantic_wall_summary(replay_summary, fanout_summary)
    no_cheat = no_cheat_counters(replay=replay, replay_summary=replay_summary, replay_rules=replay_rules, blind_summary=blind_summary, fanout_summary=fanout_summary)
    gates = [
        gate("preregistered_private_experiment_present", experiments.get("trigger_state") == "GREEN" and int_or(experiments_summary.get("experiment_count"), 0) >= 1, experiments_summary),
        gate("public_training_forbidden_by_experiment", experiments_summary.get("public_benchmark_training_allowed") is False, experiments_summary),
        gate("fallback_credit_forbidden_by_experiment", experiments_summary.get("fallback_credit_allowed") is False, experiments_summary),
        gate("private_replay_clean_or_yellow_wall", replay.get("trigger_state") in {"GREEN", "YELLOW"}, replay.get("trigger_state")),
        gate("replay_is_private_only", replay_rules.get("private_only") is True and replay_rules.get("public_calibration_run") is False, replay_rules),
        gate("eligible_families_are_learned_direct_only", bool(eligible_families) and eligible_families.issubset(ALLOWED_ELIGIBLE_FAMILIES), sorted(eligible_families)),
        gate("forbidden_families_not_eligible", not (eligible_families & FORBIDDEN_LEARNED_CREDIT_FAMILIES), sorted(eligible_families & FORBIDDEN_LEARNED_CREDIT_FAMILIES)),
        gate("replayed_families_match_eligible_scope", bool(replayed_families) and replayed_families.issubset(ALLOWED_ELIGIBLE_FAMILIES), sorted(replayed_families)),
        gate("integrity_recomputed_for_eligible_family", eligible_families.issubset(integrity_families) and int_or(replay_summary.get("candidate_integrity_mismatch_count"), -1) == 0, {"eligible": sorted(eligible_families), "integrity": sorted(integrity_families), "mismatches": replay_summary.get("candidate_integrity_mismatch_count")}),
        gate("task_and_candidate_coverage_present", int_or(replay_summary.get("task_count"), 0) >= 8 and int_or(replay_summary.get("eligible_candidate_count"), 0) > 0 and int_or(replay_summary.get("tasks_with_manifest_candidates"), 0) == int_or(replay_summary.get("task_count"), -1), replay_summary),
        gate("strict_body_tokens_loadability_measured", float_or(replay_summary.get("selected_compile_pass_rate"), 0.0) > 0.0 and float_or(replay_summary.get("selected_runtime_load_rate"), 0.0) > 0.0, replay_summary),
        gate("semantic_result_measured_without_promotion_claim", semantic_wall["measured"] and (semantic_wall["improved"] or semantic_wall["falsifying_wall_recorded"]), semantic_wall),
        gate("candidate_integrity_report_green", integrity.get("trigger_state") == "GREEN" and int_or(integrity_summary.get("integrity_mismatch_count"), -1) == 0, integrity_summary),
        gate("blind_information_flow_clean", blind_audit.get("trigger_state") == "GREEN" and int_or(blind_summary.get("static_information_flow_violation_count"), -1) == 0 and int_or(blind_summary.get("config_information_flow_violation_count"), -1) == 0, blind_summary),
        gate("forbidden_inference_fields_declared", FORBIDDEN_INFERENCE_FIELDS.issubset(forbidden_fields), sorted(FORBIDDEN_INFERENCE_FIELDS - forbidden_fields)),
        gate("policy_optimization_governed", policy.get("trigger_state") == "GREEN" and int_or(policy_summary.get("hard_gap_count"), -1) == 0, policy_summary),
        gate("generation_mode_accounted_not_promoted", generation_mode.get("trigger_state") in {"GREEN", "YELLOW"} and int_or(generation_summary.get("promotable_comparison_count"), -1) == 0 and int_or(generation_summary.get("hard_gap_count"), -1) == 0, generation_summary),
        gate("fanout_receipt_records_current_semantic_wall", fanout.get("trigger_state") == "GREEN" and float_or(nested(fanout_summary, "combined", "intended_behavior_pass_rate"), -1.0) >= 0.0, fanout_summary),
        gate("no_public_external_fallback_or_boundary_faults", all(value == 0 for value in no_cheat.values()), no_cheat),
    ]
    hard_gaps = [row for row in gates if not row["passed"]]
    expected_invalid_controls = c1_expected_invalid_controls(
        experiments_summary=experiments_summary,
        replay_summary=replay_summary,
        replay_rules=replay_rules,
        integrity=integrity,
        integrity_summary=integrity_summary,
        blind_audit=blind_audit,
        blind_summary=blind_summary,
        forbidden_fields=forbidden_fields,
        eligible_families=eligible_families,
        replayed_families=replayed_families,
        integrity_families=integrity_families,
        semantic_wall=semantic_wall,
        generation_mode=generation_mode,
        generation_summary=generation_summary,
        no_cheat=no_cheat,
    )
    synthetic_support_ready = (
        not hard_gaps
        and all(row["rejected"] for row in expected_invalid_controls)
        and semantic_wall["measured"]
        and (semantic_wall["improved"] or semantic_wall["falsifying_wall_recorded"])
        and all(value == 0 for value in no_cheat.values())
    )
    state = "GREEN" if not hard_gaps else "RED"
    support_state = (
        "synthetic-test-backed"
        if synthetic_support_ready
        else ("prototype-backed" if state == "GREEN" else "not_yet_supported")
    )
    summary = {
        "c1_correctness_generator_survival_lane_state": state,
        "c1_correctness_generator_survival_lane_support_state": support_state,
        "c1_synthetic_support_ready": synthetic_support_ready,
        "c1_expected_invalid_control_count": len(expected_invalid_controls),
        "c1_expected_invalid_rejected_count": sum(1 for row in expected_invalid_controls if row["rejected"]),
        "experiment_preregistered": experiments.get("trigger_state") == "GREEN",
        "replay_trigger_state": replay.get("trigger_state"),
        "eligible_candidate_count": replay_summary.get("eligible_candidate_count"),
        "eligible_family_counts": replay_summary.get("eligible_family_counts"),
        "task_count": replay_summary.get("task_count"),
        "tasks_with_manifest_candidates": replay_summary.get("tasks_with_manifest_candidates"),
        "selected_compile_pass_rate": replay_summary.get("selected_compile_pass_rate"),
        "selected_runtime_load_rate": replay_summary.get("selected_runtime_load_rate"),
        "selected_intended_behavior_pass_rate": replay_summary.get("selected_intended_behavior_pass_rate"),
        "pass_if_any_rate": replay_summary.get("pass_if_any_rate"),
        "functional_promotion_rate": replay_summary.get("functional_promotion_rate"),
        "falsifying_wall_recorded": semantic_wall["falsifying_wall_recorded"],
        "improvement_recorded": semantic_wall["improved"],
        "candidate_integrity_mismatch_count": replay_summary.get("candidate_integrity_mismatch_count"),
        "public_boundary_violation_count": replay_summary.get("public_boundary_violation_count"),
        "fallback_return_candidate_count": replay_summary.get("fallback_return_candidate_count"),
        "unconditional_constant_return_candidate_count": replay_summary.get("unconditional_constant_return_candidate_count"),
        "public_training_rows_written": no_cheat["public_training_rows_written"],
        "external_inference_calls": no_cheat["external_inference_calls"],
        "fallback_return_count": no_cheat["fallback_return_count"],
        "hard_gap_count": len(hard_gaps),
    }
    return {
        "policy": "project_theseus_correctness_generator_survival_lane_gate_v1",
        "created_utc": now(),
        "trigger_state": state,
        "summary": summary,
        "gates": gates,
        "hard_gaps": hard_gaps,
        "evidence_refs": evidence_refs,
        "support_state_basis": {
            "synthetic_support_ready": synthetic_support_ready,
            "expected_invalid_control_count": len(expected_invalid_controls),
            "expected_invalid_rejected_count": sum(1 for row in expected_invalid_controls if row["rejected"]),
            "eligible_families": sorted(eligible_families),
            "replayed_families": sorted(replayed_families),
            "integrity_families": sorted(integrity_families),
            "task_count": replay_summary.get("task_count"),
            "eligible_candidate_count": replay_summary.get("eligible_candidate_count"),
            "selected_compile_pass_rate": replay_summary.get("selected_compile_pass_rate"),
            "selected_runtime_load_rate": replay_summary.get("selected_runtime_load_rate"),
            "selected_intended_behavior_pass_rate": replay_summary.get("selected_intended_behavior_pass_rate"),
            "pass_if_any_rate": replay_summary.get("pass_if_any_rate"),
            "functional_promotion_rate": replay_summary.get("functional_promotion_rate"),
            "semantic_wall": semantic_wall,
            "no_cheat_counters": no_cheat,
        },
        "expected_invalid_controls": expected_invalid_controls,
        "semantic_wall": semantic_wall,
        "no_cheat_counters": no_cheat,
        "non_claims": [
            "C1 synthetic-test-backed means one bounded private verifier-driven learned body-token experiment has replay, integrity, blind-flow, generation-mode, policy, and expected-invalid controls; it does not claim promotion-grade code generation.",
            "The current replay records a falsifying semantic wall: integrity-clean transformer/hybrid candidates compile/load sometimes, but selected/pass-if-any functional behavior remains zero on this probe.",
            "Routers, templates, ngrams, semantic renderers, deterministic tools, and fallback returns are not eligible learned-generation credit.",
            "Public benchmark payloads remain calibration-only and no public prompts/tests/solutions/traces are written as training rows.",
        ],
        "next_private_repair_target": {
            "target": "semantic candidate construction before public calibration",
            "allowed_inputs": ["natural-language prompt", "callable signature", "allowed private/licensed training rows", "generated-prefix AST/state features"],
            "forbidden_inputs": sorted(FORBIDDEN_INFERENCE_FIELDS | {"public_benchmark_payloads", "teacher_runtime_tokens", "fallback_return_templates"}),
            "repair_focus": [
                "algorithm choice",
                "loop exit and local return synthesis",
                "structured output construction",
                "string/list/dict operation selection",
                "return-shape behavior inferred only from prompt/signature, not hidden labels",
            ],
        },
    }


def c1_expected_invalid_controls(
    *,
    experiments_summary: dict[str, Any],
    replay_summary: dict[str, Any],
    replay_rules: dict[str, Any],
    integrity: dict[str, Any],
    integrity_summary: dict[str, Any],
    blind_audit: dict[str, Any],
    blind_summary: dict[str, Any],
    forbidden_fields: set[str],
    eligible_families: set[str],
    replayed_families: set[str],
    integrity_families: set[str],
    semantic_wall: dict[str, Any],
    generation_mode: dict[str, Any],
    generation_summary: dict[str, Any],
    no_cheat: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            "control": "missing_preregistration_blocks_c1_synthetic",
            "rejected": int_or(experiments_summary.get("experiment_count"), 0) >= 1
            and experiments_summary.get("public_benchmark_training_allowed") is False
            and experiments_summary.get("fallback_credit_allowed") is False,
            "reason": "synthetic C1 evidence must be preregistered and must forbid public-training and fallback-credit shortcuts",
        },
        {
            "control": "public_or_external_or_fallback_fault_blocks_c1",
            "rejected": all(value == 0 for value in no_cheat.values())
            and replay_rules.get("private_only") is True
            and replay_rules.get("public_calibration_run") is False,
            "reason": "C1 cannot pass with public-training rows, runtime external inference, boundary violations, fallback returns, or constant-return shortcuts",
        },
        {
            "control": "forbidden_family_credit_blocks_c1",
            "rejected": bool(eligible_families)
            and eligible_families.issubset(ALLOWED_ELIGIBLE_FAMILIES)
            and not (eligible_families & FORBIDDEN_LEARNED_CREDIT_FAMILIES)
            and bool(replayed_families)
            and replayed_families.issubset(ALLOWED_ELIGIBLE_FAMILIES),
            "reason": "tools, templates, routers, ngrams, adapters, and unknown families cannot count as learned body-token evidence",
        },
        {
            "control": "integrity_mismatch_blocks_c1",
            "rejected": integrity.get("trigger_state") == "GREEN"
            and eligible_families.issubset(integrity_families)
            and int_or(integrity_summary.get("integrity_mismatch_count"), -1) == 0
            and int_or(replay_summary.get("candidate_integrity_mismatch_count"), -1) == 0,
            "reason": "candidate family and learned-generation eligibility must be recomputed by independent integrity audit",
        },
        {
            "control": "blind_information_flow_violation_blocks_c1",
            "rejected": blind_audit.get("trigger_state") == "GREEN"
            and int_or(blind_summary.get("static_information_flow_violation_count"), -1) == 0
            and int_or(blind_summary.get("config_information_flow_violation_count"), -1) == 0
            and FORBIDDEN_INFERENCE_FIELDS.issubset(forbidden_fields),
            "reason": "generation and ranking must not see answer-identifying fields or hidden target metadata",
        },
        {
            "control": "missing_loadability_measurement_blocks_c1",
            "rejected": float_or(replay_summary.get("selected_compile_pass_rate"), 0.0) > 0.0
            and float_or(replay_summary.get("selected_runtime_load_rate"), 0.0) > 0.0,
            "reason": "C1 needs actual learned body-token candidate loadability measurement, not a manifest-only claim",
        },
        {
            "control": "missing_semantic_measurement_blocks_c1",
            "rejected": bool(semantic_wall.get("measured"))
            and (bool(semantic_wall.get("improved")) or bool(semantic_wall.get("falsifying_wall_recorded"))),
            "reason": "C1 must either record private behavior improvement or retain a falsifying semantic wall",
        },
        {
            "control": "promotion_laundering_blocks_c1",
            "rejected": generation_mode.get("trigger_state") in {"GREEN", "YELLOW"}
            and int_or(generation_summary.get("promotable_comparison_count"), -1) == 0
            and float_or(replay_summary.get("functional_promotion_rate"), -1.0) == 0.0,
            "reason": "a falsifying-wall run cannot be laundered into a promotion or generation-mode claim",
        },
        {
            "control": "too_thin_private_fixture_blocks_c1",
            "rejected": int_or(replay_summary.get("task_count"), 0) >= 8
            and int_or(replay_summary.get("eligible_candidate_count"), 0) > 0
            and int_or(replay_summary.get("tasks_with_manifest_candidates"), 0) == int_or(replay_summary.get("task_count"), -1),
            "reason": "C1 synthetic evidence needs enough private fixture coverage to exercise the survival-lane contract",
        },
    ]


def semantic_wall_summary(replay_summary: dict[str, Any], fanout_summary: dict[str, Any]) -> dict[str, Any]:
    selected = float_or(replay_summary.get("selected_intended_behavior_pass_rate"), -1.0)
    pass_if_any = float_or(replay_summary.get("pass_if_any_rate"), -1.0)
    promotion = float_or(replay_summary.get("functional_promotion_rate"), -1.0)
    fanout_behavior = float_or(nested(fanout_summary, "combined", "intended_behavior_pass_rate"), -1.0)
    measured = selected >= 0.0 and pass_if_any >= 0.0 and promotion >= 0.0
    improved = selected > 0.0 or pass_if_any > 0.0 or promotion > 0.0
    falsifying_wall = measured and not improved and int_or(replay_summary.get("eligible_candidate_count"), 0) > 0
    return {
        "measured": measured,
        "improved": improved,
        "falsifying_wall_recorded": falsifying_wall,
        "selected_intended_behavior_pass_rate": selected,
        "pass_if_any_rate": pass_if_any,
        "functional_promotion_rate": promotion,
        "fanout_intended_behavior_pass_rate": fanout_behavior,
        "diagnosis": "semantic_candidate_quality_wall" if falsifying_wall else "private_behavior_improvement_recorded" if improved else "not_measured",
    }


def no_cheat_counters(
    *,
    replay: dict[str, Any],
    replay_summary: dict[str, Any],
    replay_rules: dict[str, Any],
    blind_summary: dict[str, Any],
    fanout_summary: dict[str, Any],
) -> dict[str, int]:
    return {
        "public_training_rows_written": int_or(replay_rules.get("public_training_rows_written"), 0) + int_or(fanout_summary.get("public_training_rows_written"), 0),
        "external_inference_calls": int_or(replay.get("external_inference_calls"), 0) + int_or(replay_rules.get("external_inference_calls"), 0) + int_or(fanout_summary.get("external_inference_calls"), 0),
        "fallback_return_count": int_or(replay_summary.get("fallback_return_candidate_count"), 0) + int_or(fanout_summary.get("fallback_return_count"), 0),
        "public_boundary_violation_count": int_or(replay_summary.get("public_boundary_violation_count"), 0),
        "candidate_integrity_mismatch_count": int_or(replay_summary.get("candidate_integrity_mismatch_count"), 0) + int_or(blind_summary.get("candidate_overclaim_count"), 0),
        "unconditional_constant_return_candidate_count": int_or(replay_summary.get("unconditional_constant_return_candidate_count"), 0),
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
