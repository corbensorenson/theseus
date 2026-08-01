#!/usr/bin/env python3
"""Audit the prospective, one-shot D1 qualification instrument.

This module deliberately does not acquire a D1 source.  Source membership may
open only after the sealed P4-v2r2 terminal disposition identifies an eligible
survivor.  The audit exists now so the statistical design, completion policy,
controls, and inference boundary cannot be chosen after seeing D1 tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_d1_fresh_qualification_instrument.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_d1_fresh_qualification_instrument_audit.json"
POLICY = "project_theseus_d1_fresh_source_disjoint_qualification_instrument_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    args = parser.parse_args()
    config_path = resolve(args.config)
    report = build_report(config_path)
    write_json(resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report(
    config_path: Path = DEFAULT_CONFIG,
    *,
    disposition_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = read_json(config_path)
    faults = validate_config(config)
    activation = mapping(config.get("activation"))
    disposition_path = resolve(str(activation.get("p4_terminal_disposition") or ""))
    disposition = (
        disposition_override
        if disposition_override is not None
        else read_json(disposition_path) if disposition_path.is_file() else {}
    )
    survivor_checks = {
        "terminal_disposition_present": bool(disposition),
        "terminal_disposition_green": disposition.get("trigger_state")
        == activation.get("required_trigger_state"),
        "decision_relevant_survivor": disposition.get("scientific_status")
        == activation.get("required_scientific_status"),
        "claim_identity_matches": disposition.get("claim_id")
        == activation.get("required_claim_id"),
        "D1_eligibility_explicit": mapping(disposition.get("consumption")).get(
            "eligible_for_D1"
        )
        is activation.get("required_eligible_for_D1"),
        "survivor_effect_rule_passed": mapping(
            disposition.get("decision_rule")
        ).get("survivor_effect_rule_passed")
        is True,
        "effect_decision_authorized": mapping(
            disposition.get("decision_rule")
        ).get("effect_decision_authorized")
        is True,
    }
    activation_ready = not faults and all(survivor_checks.values())
    prior_repositories, registry_faults = prior_repository_inventory(config)
    faults.extend(registry_faults)
    activation_state = (
        "READY_FOR_AUTONOMOUS_D1_SOURCE_MEMBERSHIP_FREEZE"
        if activation_ready
        else "WAITING_FOR_GREEN_DECISION_RELEVANT_P4V2R2_SURVIVOR"
    )
    design = recompute_power_design(mapping(config.get("power_design")))
    return {
        "policy": "project_theseus_d1_fresh_qualification_instrument_audit_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "instrument": artifact(config_path),
        "instrument_policy": config.get("policy"),
        "activation_state": activation_state,
        "execution_authorized": False,
        "source_acquisition_authorized": activation_ready,
        "candidate_or_control_calls_authorized": False,
        "survivor_checks": survivor_checks,
        "p4_terminal_disposition": (
            artifact(disposition_path) if disposition_path.is_file() else {}
        ),
        "power_design_recomputation": design,
        "prior_source_inventory": {
            "registry_count": len(
                list(mapping(config.get("source_surface")).get(
                    "source_disjoint_registry_paths"
                ) or [])
            ),
            "repository_count": len(prior_repositories),
            "repositories_sha256": stable_hash(prior_repositories),
            "repositories": prior_repositories,
        },
        "completion_policy": mapping(config.get("generation_completion")),
        "authority": mapping(config.get("one_shot_authority")),
        "maximum_inference": (
            "GREEN means the prospective D1 instrument is internally consistent. "
            "It does not mean P4 produced a survivor, does not acquire or consume a "
            "D1 task, and does not authorize candidate generation. Source acquisition "
            "opens automatically only when every survivor check is true; candidate "
            "calls remain closed until the fresh cohort and evaluator are sealed."
        ),
    }


def validate_config(config: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("instrument_policy_invalid")
    if config.get("state") != (
        "PROSPECTIVELY_BOUND_BEFORE_P4_SURVIVOR_AND_D1_SOURCE_ACQUISITION"
    ):
        faults.append("instrument_state_invalid")
    activation = mapping(config.get("activation"))
    if activation.get("user_or_operator_approval_required") is not False:
        faults.append("user_or_operator_gate_present")
    if activation.get("source_acquisition_before_activation_forbidden") is not True:
        faults.append("preactivation_source_acquisition_not_forbidden")
    completion = mapping(config.get("generation_completion"))
    if completion.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if completion.get("normal_completion") != ["parser_complete", "model_eos"]:
        faults.append("normal_completion_contract_invalid")
    if completion.get("sole_numeric_boundary") != (
        "pinned_model_declared_context_window_minus_exact_prompt_tokens"
    ):
        faults.append("numeric_boundary_not_exact_context_residual")
    if completion.get(
        "boundary_or_host_stop_is_model_mechanism_candidate_or_evaluator_failure"
    ) is not False:
        faults.append("boundary_misclassified_as_negative_evidence")
    surface = mapping(config.get("source_surface"))
    registries = list(surface.get("source_disjoint_registry_paths") or [])
    if len(registries) != 8 or len(set(registries)) != 8:
        faults.append("source_disjoint_registry_set_invalid")
    if surface.get("membership_fixed_before_archive_fetch") is not True:
        faults.append("membership_not_fixed_before_fetch")
    if surface.get("replacement_after_any_candidate_or_control_call") is not False:
        faults.append("post_generation_task_replacement_allowed")
    if surface.get("consume_exact_identity_once") is not True:
        faults.append("one_shot_consumption_missing")
    authority = mapping(config.get("one_shot_authority"))
    if authority.get("user_or_operator_gate") is not False:
        faults.append("one_shot_user_gate_present")
    if authority.get("rerun_consumed_identity_allowed") is not False:
        faults.append("consumed_surface_rerun_allowed")
    if any(int(authority.get(key) or 0) != 0 for key in (
        "external_inference_calls", "teacher_calls", "training_rows_written"
    )):
        faults.append("forbidden_external_or_training_authority_present")
    design = recompute_power_design(mapping(config.get("power_design")))
    if not design["matches_declared_design"]:
        faults.append("power_design_recomputation_mismatch")
    decision = mapping(config.get("decision_rule"))
    if decision.get("automatic_book_support_promotion") is not False:
        faults.append("automatic_book_support_promotion_allowed")
    if decision.get("serving_training_D2_or_teacher_authority") is not False:
        faults.append("cross_stage_authority_present")
    return faults


def recompute_power_design(design: dict[str, Any]) -> dict[str, Any]:
    null = float(design.get("null_treatment_win_probability") or 0.0)
    alternative = float(
        design.get("minimum_worthwhile_discordant_win_probability") or 0.0
    )
    alpha = float(design.get("alpha") or 0.0)
    minimum_power = float(design.get("minimum_power") or 0.0)
    required_pairs, required_wins, achieved_alpha, achieved_power = (0, 0, 1.0, 0.0)
    if 0.0 < null < alternative < 1.0 and 0.0 < alpha < 1.0:
        for pairs in range(1, 1000):
            for wins in range(pairs // 2 + 1, pairs + 1):
                false_positive = binomial_upper_tail(pairs, wins, null)
                power = binomial_upper_tail(pairs, wins, alternative)
                if false_positive <= alpha and power >= minimum_power:
                    required_pairs = pairs
                    required_wins = wins
                    achieved_alpha = false_positive
                    achieved_power = power
                    break
            if required_pairs:
                break
    discordance_floor = float(design.get("conservative_discordance_rate_floor") or 0.0)
    reach_probability = float(
        design.get("minimum_probability_of_reaching_required_discordant_pairs") or 0.0
    )
    cohort_size = 0
    if required_pairs and 0.0 < discordance_floor <= 1.0:
        for tasks in range(required_pairs, 10000):
            probability = binomial_upper_tail(
                tasks, required_pairs, discordance_floor
            )
            if probability >= reach_probability:
                cohort_size = tasks
                break
    matches = (
        required_pairs == int(design.get("design_required_discordant_pairs") or 0)
        and required_wins
        == int(design.get("design_required_treatment_wins_at_required_discordant_pairs") or 0)
        and cohort_size == int(design.get("design_derived_cohort_size") or 0)
    )
    return {
        "required_discordant_pairs": required_pairs,
        "required_treatment_wins_at_required_pairs": required_wins,
        "achieved_type_I_error": round(achieved_alpha, 10),
        "achieved_power_at_minimum_worthwhile_effect": round(achieved_power, 10),
        "design_derived_cohort_size": cohort_size,
        "probability_of_reaching_required_pairs_at_discordance_floor": round(
            binomial_upper_tail(cohort_size, required_pairs, discordance_floor), 10
        ) if cohort_size else 0.0,
        "matches_declared_design": matches,
        "output_token_budget_involved": False,
    }


def binomial_upper_tail(trials: int, successes: int, probability: float) -> float:
    if successes > trials:
        return 0.0
    return sum(
        math.comb(trials, count)
        * probability ** count
        * (1.0 - probability) ** (trials - count)
        for count in range(successes, trials + 1)
    )


def prior_repository_inventory(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    repositories: set[str] = set()
    faults: list[str] = []
    paths = list(mapping(config.get("source_surface")).get(
        "source_disjoint_registry_paths"
    ) or [])
    for value in paths:
        path = resolve(str(value))
        if not path.is_file():
            faults.append(f"prior_source_registry_missing:{relative(path)}")
            continue
        payload = read_json(path)
        found = collect_repositories(payload)
        for key in ("task", "p2a_instrument_adequacy_task"):
            referenced = payload.get(key)
            if not isinstance(referenced, str) or not referenced.endswith(".json"):
                continue
            referenced_path = resolve(referenced)
            if not referenced_path.is_file():
                faults.append(
                    f"prior_source_task_missing:{relative(path)}:{referenced}"
                )
                continue
            found.update(collect_repositories(read_json(referenced_path)))
        if not found:
            faults.append(f"prior_source_registry_has_no_repository_identity:{relative(path)}")
        repositories.update(found)
    return sorted(repositories), faults


def collect_repositories(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "repository" and isinstance(item, str) and "/" in item:
                found.add(item.strip().lower())
            else:
                found.update(collect_repositories(item))
    elif isinstance(value, list):
        for item in value:
            found.update(collect_repositories(item))
    return found


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "activation_state": report.get("activation_state"),
        "source_acquisition_authorized": report.get("source_acquisition_authorized"),
        "candidate_or_control_calls_authorized": report.get(
            "candidate_or_control_calls_authorized"
        ),
        "design_derived_cohort_size": mapping(
            report.get("power_design_recomputation")
        ).get("design_derived_cohort_size"),
        "faults": report.get("faults"),
    }


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
