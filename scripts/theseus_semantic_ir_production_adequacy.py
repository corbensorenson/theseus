#!/usr/bin/env python3
"""Independently audit the Semantic-IR production adequacy experiment.

This owner does not generate candidates, repair Semantic IR, or score hidden
behavior.  It recomputes the prospective design and, once present, audits a
source-disjoint task pool and model-run receipt without trusting candidate
claims.  The preregistration path performs no network or model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_audit.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_audit_v1"
CONFIG_POLICY = "project_theseus_semantic_ir_production_adequacy_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--pool", default="")
    parser.add_argument("--results", default="")
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    args = parser.parse_args()
    report = audit(
        resolve(args.config),
        pool_path=resolve(args.pool) if args.pool else None,
        results_path=resolve(args.results) if args.results else None,
    )
    write_json(resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def audit(
    config_path: Path,
    *,
    pool_path: Path | None = None,
    results_path: Path | None = None,
) -> dict[str, Any]:
    config = read_json(config_path)
    faults = audit_config(config, config_path)
    design = mapping(config.get("competence_design"))
    trials = integer(design.get("panel_size"))
    required = integer(design.get("minimum_successes"))
    null = number(design.get("null_mechanics_probability"))
    alternative = number(design.get("adequate_mechanics_probability"))
    false_positive = binomial_upper_tail(trials, required, null)
    power = binomial_upper_tail(trials, required, alternative)
    pool_audit: dict[str, Any] | None = None
    result_audit: dict[str, Any] | None = None
    if pool_path is not None:
        pool_audit = audit_pool(read_json(pool_path), config)
        faults.extend(pool_audit["faults"])
    if results_path is not None:
        if pool_audit is None:
            faults.append("results_require_independently_audited_pool")
        else:
            result_audit = audit_results(read_json(results_path), config, pool_audit)
            faults.extend(result_audit["faults"])
    stage = (
        "terminal_adequacy_audit"
        if results_path is not None
        else "sealed_pool_audit"
        if pool_path is not None
        else "preregistration_audit"
    )
    if faults:
        trigger = "RED"
    elif result_audit is not None:
        trigger = "GREEN" if result_audit["adequate"] else "YELLOW"
    elif pool_audit is not None:
        trigger = "GREEN"
    else:
        trigger = "GREEN"
    return {
        "policy": POLICY,
        "stage": stage,
        "trigger_state": trigger,
        "faults": sorted(set(faults)),
        "config_sha256": sha256_file(config_path),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "design_recomputation": {
            "panel_size": trials,
            "minimum_successes": required,
            "null_mechanics_probability": null,
            "adequate_mechanics_probability": alternative,
            "one_sided_false_positive_probability": false_positive,
            "power_at_adequate_probability": power,
        },
        "pool_audit": pool_audit,
        "result_audit": result_audit,
        "counters": {
            "network_calls": 0,
            "local_model_calls": 0,
            "external_inference_calls": 0,
            "hidden_evaluator_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "maximum_inference": (
            "A GREEN preregistration audit establishes only that the independent "
            "adequacy design is internally consistent and source-bound. A GREEN "
            "pool audit additionally establishes source, license, temporal, and "
            "information-flow eligibility before generation. Only a terminal "
            "results audit can disposition this exact implementation as adequate; "
            "none of these states is a subsystem treatment effect, D1, D2, or book support."
        ),
    }


def audit_config(config: dict[str, Any], config_path: Path) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    if config.get("state") != "PREREGISTERED_METADATA_ACQUISITION_ONLY":
        faults.append("config_state_invalid")
    for path_key, hash_key in (
        ("production_owner", "production_owner_sha256"),
        ("production_config", "production_config_sha256"),
        ("bounded_canary_report", "bounded_canary_report_sha256"),
    ):
        path = resolve(str(config.get(path_key) or ""))
        if not path.is_file() or sha256_file(path) != str(config.get(hash_key) or ""):
            faults.append(f"binding_invalid:{path_key}")
    model = mapping(config.get("frozen_model"))
    if model.get("repo_id") != "mlx-community/Tmax-9B-MLX-8bit":
        faults.append("frozen_model_repo_invalid")
    if model.get("revision") != "33812d6cf04f88856f25eb828de4f3144a194560":
        faults.append("frozen_model_revision_invalid")
    design = mapping(config.get("competence_design"))
    trials = integer(design.get("panel_size"))
    required = integer(design.get("minimum_successes"))
    null = number(design.get("null_mechanics_probability"))
    alternative = number(design.get("adequate_mechanics_probability"))
    alpha = number(design.get("one_sided_alpha"))
    target_power = number(design.get("minimum_power"))
    if (trials, required, null, alternative, alpha, target_power) != (
        18, 13, 0.5, 0.8, 0.05, 0.8
    ):
        faults.append("competence_design_not_frozen_exact")
    if binomial_upper_tail(trials, required, null) > alpha:
        faults.append("false_positive_probability_above_alpha")
    if binomial_upper_tail(trials, required, alternative) < target_power:
        faults.append("power_below_minimum")
    strata = dictionaries(design.get("strata"))
    if len(strata) != 6 or any(integer(row.get("task_count")) != 3 for row in strata):
        faults.append("six_by_three_strata_missing")
    if len({str(row.get("id") or "") for row in strata}) != 6:
        faults.append("stratum_ids_not_unique")
    completion = mapping(config.get("generation_completion"))
    if completion.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if completion.get("normal_completion") != ["parser_complete", "model_eos"]:
        faults.append("normal_completion_invalid")
    authority = mapping(config.get("authority"))
    false_fields = (
        "external_inference_authorized",
        "teacher_calls_authorized",
        "training_rows_authorized",
        "serving_authorized",
        "D1_authorized",
        "D2_authorized",
        "book_support_promotion_authorized",
        "user_or_operator_gate",
    )
    if any(authority.get(key) is not False for key in false_fields):
        faults.append("authority_boundary_invalid")
    if authority.get("public_source_metadata_and_archive_acquisition_authorized") is not True:
        faults.append("public_source_acquisition_not_authorized")
    if config_path != DEFAULT_CONFIG and not config_path.is_file():
        faults.append("config_missing")
    return faults


def audit_pool(pool: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    tasks = dictionaries(pool.get("tasks"))
    design = mapping(config.get("competence_design"))
    expected = integer(design.get("panel_size"))
    if pool.get("state") != "SEALED_BEFORE_ANY_MODEL_OR_CONTROL_CALL":
        faults.append("pool_not_sealed_pre_generation")
    if len(tasks) != expected:
        faults.append("panel_size_mismatch")
    repositories = [str(row.get("repository") or "").lower() for row in tasks]
    if len(set(repositories)) != len(repositories):
        faults.append("repositories_not_distinct")
    excluded = {str(value).lower() for value in config.get("excluded_repositories") or []}
    if excluded.intersection(repositories):
        faults.append("source_overlap_with_prior_theseus_repository")
    snapshot = str(mapping(config.get("temporal_guard")).get("model_snapshot_observed_utc") or "")
    strata = {str(row.get("id")): integer(row.get("task_count")) for row in dictionaries(design.get("strata"))}
    observed_strata: dict[str, int] = {key: 0 for key in strata}
    for row in tasks:
        if str(row.get("merged_utc") or "") <= snapshot:
            faults.append("task_not_strictly_post_snapshot")
        if row.get("license_verified") is not True:
            faults.append("task_license_not_verified")
        if row.get("parent_failed_independent_evaluator") is not True:
            faults.append("parent_not_known_negative")
        if row.get("target_passed_independent_evaluator") is not True:
            faults.append("target_not_known_positive")
        if row.get("candidate_visible_target_or_hidden_data") is not False:
            faults.append("candidate_information_flow_invalid")
        stratum = str(row.get("stratum") or "")
        if stratum not in observed_strata:
            faults.append("unknown_task_stratum")
        else:
            observed_strata[stratum] += 1
    if observed_strata != strata:
        faults.append("stratum_balance_mismatch")
    counters = mapping(pool.get("counters"))
    for key in (
        "local_model_calls",
        "external_inference_calls",
        "teacher_calls",
        "training_rows_written",
        "D1_cases_consumed",
        "D2_cases_consumed",
    ):
        if integer(counters.get(key)) != 0:
            faults.append(f"pre_generation_counter_nonzero:{key}")
    return {
        "faults": sorted(set(faults)),
        "task_count": len(tasks),
        "distinct_repositories": len(set(repositories)),
        "strata": observed_strata,
        "pool_sha256": stable_hash(pool),
    }


def audit_results(
    results: dict[str, Any],
    config: dict[str, Any],
    pool_audit: dict[str, Any],
) -> dict[str, Any]:
    faults: list[str] = []
    rows = dictionaries(results.get("tasks"))
    design = mapping(config.get("competence_design"))
    expected = integer(design.get("panel_size"))
    required = integer(design.get("minimum_successes"))
    if len(rows) != expected:
        faults.append("result_task_count_mismatch")
    successes = sum(row.get("production_mechanics_passed") is True for row in rows)
    if any(row.get("candidate_integrity_recomputed_independently") is not True for row in rows):
        faults.append("independent_candidate_integrity_missing")
    if any(row.get("physical_context_boundary_hit") is True for row in rows):
        faults.append("physical_context_boundary_hit")
    if any(row.get("unsafe_or_unauthorized_effect") is True for row in rows):
        faults.append("unsafe_or_unauthorized_effect")
    expected_strata = {str(row["id"]): 2 for row in dictionaries(design.get("strata"))}
    stratum_successes = {key: 0 for key in expected_strata}
    for row in rows:
        if row.get("production_mechanics_passed") is True:
            key = str(row.get("stratum") or "")
            if key in stratum_successes:
                stratum_successes[key] += 1
    weak_tail_passed = all(
        stratum_successes[key] >= threshold for key, threshold in expected_strata.items()
    )
    adequate = (
        not faults
        and pool_audit.get("faults") == []
        and successes >= required
        and weak_tail_passed
    )
    return {
        "faults": sorted(set(faults)),
        "successes": successes,
        "required_successes": required,
        "stratum_successes": stratum_successes,
        "minimum_successes_per_stratum": expected_strata,
        "weak_tail_passed": weak_tail_passed,
        "adequate": adequate,
        "disposition": "ADEQUATE_FOR_ONE_FRESH_CLAIM_CAMPAIGN" if adequate else "INCONCLUSIVE_EXPERIMENT",
    }


def binomial_upper_tail(trials: int, successes: int, probability: float) -> float:
    if trials < 0 or successes < 0 or successes > trials or not 0.0 <= probability <= 1.0:
        return 1.0
    return sum(
        math.comb(trials, value)
        * probability**value
        * (1.0 - probability) ** (trials - value)
        for value in range(successes, trials + 1)
    )


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value or [] if isinstance(row, dict)]


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"),
        "design_recomputation": report.get("design_recomputation"),
        "result_disposition": mapping(report.get("result_audit")).get("disposition"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
