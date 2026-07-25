#!/usr/bin/env python3
"""Run and aggregate the three-seed canonical RDC/KERC matched adequacy panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import host_resource_safety
import pretraining_candidate_canary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "rdc_kerc_matched_adequacy.json"
DEFAULT_REPORT = ROOT / "reports" / "rdc_kerc_matched_adequacy.json"


class KercAdequacyFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KercAdequacyFault(f"json_unavailable:{path}") from exc
    if not isinstance(value, dict):
        raise KercAdequacyFault(f"json_object_required:{path}")
    return value


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(path)
    if config.get("policy") != "project_theseus_rdc_kerc_matched_adequacy_v1":
        raise KercAdequacyFault("policy_invalid")
    if config.get("targets") != ["english_kerc", "english_surface_control"]:
        raise KercAdequacyFault("matched_targets_invalid")
    seeds = [int(value) for value in config.get("seeds") or []]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise KercAdequacyFault("three_unique_seeds_required")
    if int(config.get("steps_per_target") or 0) <= 0:
        raise KercAdequacyFault("step_budget_invalid")
    hard = config.get("hard_boundaries") or {}
    for key in (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "fallback_template_router_tool_credit",
        "confirmation_surface_consumption",
    ):
        if int(hard.get(key, -1)) != 0:
            raise KercAdequacyFault(f"hard_boundary_nonzero:{key}")
    if hard.get("production_checkpoint_mutation") is not False:
        raise KercAdequacyFault("production_checkpoint_mutation_allowed")
    config["_config_path"] = str(path)
    return config


def render(pattern: str, seed: int) -> str:
    return pattern.replace("{seed}", str(seed))


def aggregate_candidate_receipts(
    lease: dict[str, Any],
    components: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    receipts = [(target_id, report["candidate_canary_resource_receipt"]) for target_id, report in components]
    faults = [
        f"{target_id}:{fault}"
        for target_id, receipt in receipts
        for fault in receipt.get("faults") or []
    ]
    observations = [
        {"target_id": target_id, **observation}
        for target_id, receipt in receipts
        for observation in receipt.get("observation_prefix") or []
    ]
    return {
        "policy": "project_theseus_candidate_canary_resource_receipt_v1",
        "lease_digest": lease["lease_digest"],
        "candidate_id": lease["candidate_id"],
        "observed_optimizer_positions": sum(
            int(receipt.get("observed_optimizer_positions") or 0)
            for _target_id, receipt in receipts
        ),
        "observed_maximum_step": max(
            int(receipt.get("observed_maximum_step") or 0)
            for _target_id, receipt in receipts
        ),
        "wall_seconds": round(
            sum(float(receipt.get("wall_seconds") or 0.0) for _target_id, receipt in receipts),
            6,
        ),
        "peak_rss_mib": max(float(receipt.get("peak_rss_mib") or 0.0) for _target_id, receipt in receipts),
        "peak_mlx_active_mib": max(float(receipt.get("peak_mlx_active_mib") or 0.0) for _target_id, receipt in receipts),
        "peak_mlx_cache_mib": max(float(receipt.get("peak_mlx_cache_mib") or 0.0) for _target_id, receipt in receipts),
        "peak_mlx_allocator_mib": max(float(receipt.get("peak_mlx_allocator_mib") or 0.0) for _target_id, receipt in receipts),
        "memory_budget_measurement": "maximum_of_process_isolated_target_receipts",
        "memory_budget_enforcement": "external_host_live_reserve_and_swap_watchdog_per_isolated_target",
        "declared_peak_memory_mib": float(lease["budgets"]["max_peak_memory_mib"]),
        "declared_peak_memory_exceeded": any(
            receipt.get("declared_peak_memory_exceeded") is True
            for _target_id, receipt in receipts
        ),
        "observation_prefix": observations[:64],
        "peak_scratch_disk_mib": max(float(receipt.get("peak_scratch_disk_mib") or 0.0) for _target_id, receipt in receipts),
        "faults": faults,
        "passed": all(receipt.get("passed") is True for _target_id, receipt in receipts) and not faults,
        "aggregation": "process_isolated_matched_pair_v1",
        "component_receipts": {
            target_id: receipt for target_id, receipt in receipts
        },
    }


def aggregate_host_receipts(
    components: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    receipts = [(target_id, report["host_resource_safety_receipt"]) for target_id, report in components]
    faults = [
        f"{target_id}:{receipt.get('fault')}"
        for target_id, receipt in receipts
        if receipt.get("fault")
    ]
    observations = [
        {"target_id": target_id, **observation}
        for target_id, receipt in receipts
        for observation in receipt.get("observation_prefix") or []
    ]
    return {
        "policy": host_resource_safety.POLICY,
        "command": ["process_isolated_matched_pair_v1"],
        "passed": all(receipt.get("passed") is True for _target_id, receipt in receipts) and not faults,
        "child_started": all(receipt.get("child_started") is True for _target_id, receipt in receipts),
        "terminated_by_guard": any(receipt.get("terminated_by_guard") is True for _target_id, receipt in receipts),
        "fault": ",".join(faults),
        "returncode": 0 if all(int(receipt.get("returncode") or 0) == 0 for _target_id, receipt in receipts) else 2,
        "wall_seconds": round(sum(float(receipt.get("wall_seconds") or 0.0) for _target_id, receipt in receipts), 3),
        "physical_memory_mib": max(float(receipt.get("physical_memory_mib") or 0.0) for _target_id, receipt in receipts),
        "initial_reclaimable_available_mib": min(float(receipt.get("initial_reclaimable_available_mib") or 0.0) for _target_id, receipt in receipts),
        "minimum_reclaimable_available_mib": min(float(receipt.get("minimum_reclaimable_available_mib") or 0.0) for _target_id, receipt in receipts),
        "maximum_process_rss_mib": max(float(receipt.get("maximum_process_rss_mib") or 0.0) for _target_id, receipt in receipts),
        "maximum_inferred_unified_memory_mib": max(float(receipt.get("maximum_inferred_unified_memory_mib") or 0.0) for _target_id, receipt in receipts),
        "maximum_swapout_growth_mib": max(float(receipt.get("maximum_swapout_growth_mib") or 0.0) for _target_id, receipt in receipts),
        "reserve_breach_observations_required": max(int(receipt.get("reserve_breach_observations_required") or 0) for _target_id, receipt in receipts),
        "maximum_consecutive_reserve_breaches": max(int(receipt.get("maximum_consecutive_reserve_breaches") or 0) for _target_id, receipt in receipts),
        "limits": receipts[0][1].get("limits") or {},
        "observation_prefix": observations[:64],
        "aggregation": "process_isolated_matched_pair_v1",
        "component_receipts": {
            target_id: receipt for target_id, receipt in receipts
        },
    }


def assemble_seed_report(
    config: dict[str, Any],
    seed: int,
    *,
    report_path: Path,
    scratch: Path,
    candidate_contract: dict[str, Any],
    components: list[tuple[str, dict[str, Any]]],
) -> Path:
    pair_lease = pretraining_candidate_canary.candidate_lease(
        candidate_id=config["candidate_id"],
        max_steps=int(config["steps_per_target"]),
        scratch_checkpoint_root=scratch,
        targets=list(config["targets"]),
        phase=config["phase"],
        resume=False,
        selected_seed=seed,
        contract=candidate_contract,
    )
    if pair_lease.get("authorized") is not True:
        raise KercAdequacyFault(
            f"pair_lease_unauthorized:{seed}:{pair_lease.get('faults')}"
        )
    results = [
        result_by_target(component, target_id)
        for target_id, component in components
    ]
    durable_candidate = read_json(
        scratch / "english_kerc" / "training_receipt.json"
    )
    results[0]["candidate_initialization"] = durable_candidate[
        "candidate_initialization"
    ]
    hard_gaps = sorted(
        set(
            gap
            for _target_id, component in components
            for gap in component.get("hard_gaps") or []
        )
    )
    candidate_resource_receipt = aggregate_candidate_receipts(
        pair_lease, components
    )
    host_resource_receipt = aggregate_host_receipts(components)
    bounded_steps_complete = (
        all(
            component.get("trigger_state") == "GREEN"
            and int(result.get("optimizer_steps") or 0)
            == int(config["steps_per_target"])
            for (_target_id, component), result in zip(components, results)
        )
        and candidate_resource_receipt.get("passed") is True
        and host_resource_receipt.get("passed") is True
    )
    payload = {
        **components[0][1],
        "candidate_canary_lease": pair_lease,
        "candidate_canary_resource_receipt": candidate_resource_receipt,
        "host_resource_safety_receipt": host_resource_receipt,
        "executed_targets": list(config["targets"]),
        "results": results,
        "hard_gaps": hard_gaps,
        "all_requested_targets_complete": all(
            row.get("complete") for row in results
        ),
        "bounded_candidate_steps_complete": bounded_steps_complete,
    }
    payload["trigger_state"] = (
        "GREEN" if bounded_steps_complete and not hard_gaps else "RED"
    )
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path


def run_seed(config: dict[str, Any], seed: int) -> Path:
    report_path = resolve(render(config["report_pattern"], seed))
    scratch = resolve(render(config["scratch_root_pattern"], seed))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_contract = read_json(resolve(config["candidate_contract"]))
    candidate_row = next(
        row
        for row in candidate_contract["canaries"]
        if row["candidate_id"] == config["candidate_id"]
    )
    safety_mapping = {
        **candidate_contract["host_safety_policy"],
        **dict(candidate_row.get("host_safety_overrides") or {}),
    }
    safety_policy = host_resource_safety.policy_from_mapping(
        safety_mapping,
        maximum_wall_seconds=float(
            min(
                int(config.get("maximum_seed_wall_seconds") or 21600),
                int(candidate_row["max_wall_seconds"]),
            )
        ),
    )
    state_path = scratch / "candidate_initialization_state.json"
    state_path.unlink(missing_ok=True)
    components: list[tuple[str, dict[str, Any]]] = []
    for target_id in config["targets"]:
        component_path = report_path.with_name(
            f"{report_path.stem}.{target_id}.component{report_path.suffix}"
        )
        command = [
            sys.executable,
            config["trainer_source"],
            "--config",
            config["trainer_config"],
            "--out",
            str(component_path),
            "--execute",
            "--target",
            target_id,
            "--max-steps",
            str(config["steps_per_target"]),
            "--architecture-candidate-id",
            config["candidate_id"],
            "--candidate-seed",
            str(seed),
            "--phase",
            config["phase"],
            "--scratch-checkpoint-root",
            str(scratch),
            "--candidate-initialization-state",
            str(state_path),
        ]
        process = host_resource_safety.run_guarded(
            command,
            cwd=ROOT,
            policy=safety_policy,
            env={"THESEUS_GUARDED_ACCELERATOR_CHILD": "1"},
        )
        guard_path = report_path.with_name(
            f"{report_path.stem}.{target_id}.host_resource_safety{report_path.suffix}"
        )
        guard_path.write_text(
            json.dumps(process.receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if process.returncode != 0 or process.receipt.get("passed") is not True:
            raise KercAdequacyFault(
                f"seed_run_failed:{seed}:{target_id}:{process.returncode}:"
                f"guard={process.receipt.get('fault')}:{process.stderr[-2000:]}"
            )
        component = read_json(component_path)
        component["host_resource_safety_receipt"] = process.receipt
        components.append((target_id, component))

    return assemble_seed_report(
        config,
        seed,
        report_path=report_path,
        scratch=scratch,
        candidate_contract=candidate_contract,
        components=components,
    )


def assemble_existing_seed(config: dict[str, Any], seed: int) -> Path:
    report_path = resolve(render(config["report_pattern"], seed))
    scratch = resolve(render(config["scratch_root_pattern"], seed))
    components: list[tuple[str, dict[str, Any]]] = []
    for target_id in config["targets"]:
        component_path = report_path.with_name(
            f"{report_path.stem}.{target_id}.component{report_path.suffix}"
        )
        guard_path = report_path.with_name(
            f"{report_path.stem}.{target_id}.host_resource_safety{report_path.suffix}"
        )
        component = read_json(component_path)
        guard = read_json(guard_path)
        if component.get("trigger_state") != "GREEN" or guard.get("passed") is not True:
            raise KercAdequacyFault(
                f"existing_component_invalid:{seed}:{target_id}"
            )
        component["host_resource_safety_receipt"] = guard
        components.append((target_id, component))
    return assemble_seed_report(
        config,
        seed,
        report_path=report_path,
        scratch=scratch,
        candidate_contract=read_json(resolve(config["candidate_contract"])),
        components=components,
    )


def result_by_target(report: dict[str, Any], target_id: str) -> dict[str, Any]:
    rows = [
        row for row in report.get("results") or []
        if isinstance(row, dict) and row.get("target_id") == target_id
    ]
    if len(rows) != 1:
        raise KercAdequacyFault(f"target_result_count:{target_id}:{len(rows)}")
    return rows[0]


def training_seconds(result: dict[str, Any]) -> float:
    return sum(
        float((phase or {}).get("device_step_seconds_total") or 0.0)
        for phase in (result.get("phases") or {}).values()
        if isinstance(phase, dict)
    )


def assert_zero(row: dict[str, Any], key: str, scope: str) -> None:
    if int(row.get(key, 0)) != 0:
        raise KercAdequacyFault(f"no_cheat_counter_nonzero:{scope}:{key}")


def summarize_seed(config: dict[str, Any], seed: int, report: dict[str, Any]) -> dict[str, Any]:
    lease = report.get("candidate_canary_lease") or {}
    if (
        report.get("policy") != "project_theseus_moecot_language_arm_training_plan_v1"
        or report.get("trigger_state") != "GREEN"
        or lease.get("candidate_id") != config["candidate_id"]
        or int(lease.get("selected_seed") or 0) != seed
        or lease.get("seed_execution_mode") != "single_bound_seed"
        or lease.get("targets") != config["targets"]
    ):
        raise KercAdequacyFault(f"seed_report_contract_invalid:{seed}")
    for key in (
        "public_training_rows_written",
        "external_inference_calls",
        "fallback_return_count",
        "templates_renderers_routers_tools_credit",
    ):
        assert_zero(report, key, f"seed-{seed}")
    receipt = report.get("candidate_canary_resource_receipt") or {}
    if receipt.get("passed") is not True or receipt.get("faults"):
        raise KercAdequacyFault(f"resource_receipt_failed:{seed}")
    host_receipt = report.get("host_resource_safety_receipt") or {}
    if (
        host_receipt.get("policy") != host_resource_safety.POLICY
        or host_receipt.get("passed") is not True
        or host_receipt.get("terminated_by_guard") is not False
    ):
        raise KercAdequacyFault(f"host_resource_safety_receipt_failed:{seed}")

    summaries: dict[str, Any] = {}
    results: dict[str, Any] = {}
    for target_id in config["targets"]:
        result = result_by_target(report, target_id)
        if int(result.get("candidate_seed") or 0) != seed:
            raise KercAdequacyFault(f"result_seed_mismatch:{seed}:{target_id}")
        evaluation = result.get("evaluation") or {}
        bounded = evaluation.get("bounded_candidate_evaluation") or {}
        if (
            int(evaluation.get("row_count") or 0)
            != int(config["adoption_gate"]["required_rows_per_target"])
            or bounded.get("active") is not True
            or bounded.get("selection_uses_model_outcomes") is not False
            or bounded.get("selection_uses_answer_text") is not False
            or evaluation.get("generator_visible_fields") != ["prompt"]
        ):
            raise KercAdequacyFault(f"behavior_contract_invalid:{seed}:{target_id}")
        for key in (
            "templates_renderers_routers_tools_credit",
            "public_training_rows_written",
            "public_benchmark_payload_count",
            "external_inference_calls",
            "fallback_return_count",
        ):
            assert_zero(evaluation, key, f"{seed}:{target_id}")
        summaries[target_id] = evaluation["summary"]
        results[target_id] = result

    candidate = results["english_kerc"]
    control = results["english_surface_control"]
    candidate_initialization = candidate.get("candidate_initialization") or {}
    control_initialization = control.get("candidate_initialization") or {}
    candidate_common = candidate_initialization.get("common_tensor_manifest") or {}
    control_common = control_initialization.get("common_tensor_manifest") or {}
    candidate_specific = (
        candidate_initialization.get("architecture_specific_tensor_manifest") or {}
    )
    control_specific = (
        control_initialization.get("architecture_specific_tensor_manifest") or {}
    )
    if (
        candidate_initialization.get("policy")
        != "project_theseus_candidate_common_initialization_v1"
        or control_initialization.get("policy")
        != "project_theseus_candidate_common_initialization_v1"
        or candidate_initialization.get("role") != "reference"
        or control_initialization.get("role") != "aligned"
        or int(candidate_initialization.get("seed") or 0) != seed
        or int(control_initialization.get("seed") or 0) != seed
        or candidate_initialization.get("exact_alignment") is not True
        or control_initialization.get("exact_alignment") is not True
        or candidate_initialization.get("architecture_specific_tensors_unchanged")
        is not True
        or control_initialization.get("architecture_specific_tensors_unchanged")
        is not True
        or int(candidate_common.get("tensor_count") or 0) <= 0
        or candidate_common != control_common
        or int(candidate_specific.get("tensor_count") or 0) <= 0
        or int(control_specific.get("tensor_count") or 0) <= 0
    ):
        raise KercAdequacyFault(f"common_initialization_invalid:{seed}")
    candidate_summary = summaries["english_kerc"]
    control_summary = summaries["english_surface_control"]
    parameter_ratio = max(int(candidate["parameter_count"]), int(control["parameter_count"])) / min(
        int(candidate["parameter_count"]), int(control["parameter_count"])
    )
    candidate_seconds = training_seconds(candidate)
    control_seconds = training_seconds(control)
    return {
        "seed": seed,
        "candidate_summary": candidate_summary,
        "control_summary": control_summary,
        "exact_match_gain_count": int(candidate_summary["exact_match_count"]) - int(control_summary["exact_match_count"]),
        "mean_target_similarity_gain": float(candidate_summary["mean_target_sequence_similarity"]) - float(control_summary["mean_target_sequence_similarity"]),
        "nonempty_rate_gain": float(candidate_summary["nonempty_rate"]) - float(control_summary["nonempty_rate"]),
        "byte_serialization_valid_rate_gain": float(candidate_summary["byte_serialization_valid_rate"]) - float(control_summary["byte_serialization_valid_rate"]),
        "parameter_count_ratio": parameter_ratio,
        "candidate_optimizer_steps": int(candidate.get("optimizer_steps") or 0),
        "control_optimizer_steps": int(control.get("optimizer_steps") or 0),
        "candidate_training_seconds": candidate_seconds,
        "control_training_seconds": control_seconds,
        "training_wall_time_ratio": candidate_seconds / max(control_seconds, 1e-9),
        "common_initialization": {
            "exact": True,
            "manifest": candidate_common,
            "candidate_architecture_specific": candidate_specific,
            "control_architecture_specific": control_specific,
        },
        "resource_receipt": receipt,
        "host_resource_safety_receipt": host_receipt,
    }


def aggregate(config: dict[str, Any], reports: list[tuple[int, Path, dict[str, Any]]]) -> dict[str, Any]:
    rows = [summarize_seed(config, seed, report) for seed, _path, report in reports]
    k5_path = resolve(config["k5_learnability_prerequisite"])
    k5 = read_json(k5_path)
    k5_profiles = ((k5.get("autoregressive_pipeline") or {}).get("profiles") or {})
    known_tiny_profile_steps = max(
        [
            int(((profile or {}).get("training") or {}).get("optimizer_steps") or 0)
            for profile in k5_profiles.values()
        ]
        or [0]
    )
    gate = config["adoption_gate"]
    mean_exact = statistics.fmean(row["exact_match_gain_count"] for row in rows)
    seed_win_fraction = sum(row["exact_match_gain_count"] > 0 for row in rows) / len(rows)
    mean_similarity = statistics.fmean(row["mean_target_similarity_gain"] for row in rows)
    checks = {
        "seed_count": len(rows) == len(config["seeds"]),
        "matched_parameter_count": max(row["parameter_count_ratio"] for row in rows) <= float(gate["maximum_parameter_count_ratio"]),
        "equal_optimizer_steps": all(row["candidate_optimizer_steps"] == row["control_optimizer_steps"] for row in rows),
        "mean_exact_match_gain": mean_exact >= float(gate["minimum_mean_exact_match_gain_count"]),
        "exact_seed_win_fraction": seed_win_fraction >= float(gate["minimum_exact_seed_win_fraction"]),
        "mean_target_similarity_gain": mean_similarity >= float(gate["minimum_mean_target_similarity_gain"]),
        "nonempty_nonregression": min(row["nonempty_rate_gain"] for row in rows) >= 0.0,
        "byte_serialization_nonregression": min(row["byte_serialization_valid_rate_gain"] for row in rows) >= 0.0,
        "lifecycle_cost": max(row["training_wall_time_ratio"] for row in rows) <= float(gate["maximum_training_wall_time_ratio"]),
        "common_initialization_exact": all(
            row["common_initialization"]["exact"] for row in rows
        ),
    }
    required = list(checks)
    if not gate.get("require_equal_optimizer_steps", True):
        required.remove("equal_optimizer_steps")
    if not gate.get("require_nonempty_rate_nonregression", True):
        required.remove("nonempty_nonregression")
    if not gate.get("require_byte_serialization_nonregression", True):
        required.remove("byte_serialization_nonregression")
    adopted = all(checks[key] for key in required)
    candidate_generation_absent = all(
        float(row["candidate_summary"].get("nonempty_rate") or 0.0) == 0.0
        for row in rows
    )
    under_known_k5_budget = (
        k5.get("trigger_state") == "GREEN"
        and known_tiny_profile_steps > int(config["steps_per_target"])
    )
    config_path = resolve(config.get("_config_path") or DEFAULT_CONFIG)
    source_artifacts = {
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
            "bytes": config_path.stat().st_size,
        },
        "trainer_config": {
            "path": config["trainer_config"],
            "sha256": sha256(resolve(config["trainer_config"])),
            "bytes": resolve(config["trainer_config"]).stat().st_size,
        },
        "trainer_source": {
            "path": config["trainer_source"],
            "sha256": sha256(resolve(config["trainer_source"])),
            "bytes": resolve(config["trainer_source"]).stat().st_size,
        },
        "aggregator_source": {
            "path": config["aggregator_source"],
            "sha256": sha256(resolve(config["aggregator_source"])),
            "bytes": resolve(config["aggregator_source"]).stat().st_size,
        },
        "candidate_contract": {
            "path": config["candidate_contract"],
            "sha256": sha256(resolve(config["candidate_contract"])),
            "bytes": resolve(config["candidate_contract"]).stat().st_size,
        },
        "k5_learnability_prerequisite": {
            "path": config["k5_learnability_prerequisite"],
            "sha256": sha256(k5_path),
            "bytes": k5_path.stat().st_size,
        },
    }
    for seed, path, _report in reports:
        source_artifacts[f"seed_{seed}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return {
        "policy": config["policy"],
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN",
        "support_state": "private-source-disjoint-three-seed-canonical-matched-experiment",
        "disposition": (
            "ADOPT_RDC_KERC_FIRST_CAMPAIGN"
            if adopted
            else (
                "INCONCLUSIVE_EXPERIMENT_REPAIR_K5_BEFORE_K8"
                if candidate_generation_absent and under_known_k5_budget
                else "SCOPED_DISCOVERY_EXCLUDED_FROM_FIRST_CAMPAIGN"
            )
        ),
        "scientific_falsification_claimed": False,
        "source_artifacts": source_artifacts,
        "checks": checks,
        "metrics": {
            "seed_count": len(rows),
            "mean_exact_match_gain_count": mean_exact,
            "exact_seed_win_fraction": seed_win_fraction,
            "mean_target_similarity_gain": mean_similarity,
            "maximum_parameter_count_ratio": max(row["parameter_count_ratio"] for row in rows),
            "maximum_training_wall_time_ratio": max(row["training_wall_time_ratio"] for row in rows),
        },
        "adequacy_assessment": {
            "candidate_generation_absent_all_seeds": candidate_generation_absent,
            "known_tiny_profile_autoregressive_steps": known_tiny_profile_steps,
            "matched_campaign_steps_per_target": int(config["steps_per_target"]),
            "matched_budget_below_known_k5_profile": under_known_k5_budget,
            "state": (
                "INCONCLUSIVE_EXPERIMENT"
                if candidate_generation_absent and under_known_k5_budget
                else "ADEQUACY_NOT_REJECTED_BY_THIS_CHECK"
            ),
        },
        "runs": rows,
        "hard_boundaries": config["hard_boundaries"],
        "reentry_condition": "A larger prospectively frozen verifier-bearing KERC campaign with the same source custody and matched total-system cost.",
        "claim_boundary": "This direct private three-seed panel can select first-campaign engineering spend; it does not establish general KERC utility or scientific falsification.",
        "public_training_rows": 0,
        "public_evaluation_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "production_checkpoint_mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_REPORT))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--assemble-components", action="store_true")
    args = parser.parse_args()
    if args.execute and args.assemble_components:
        parser.error("--execute and --assemble-components are mutually exclusive")
    config_path = resolve(args.config)
    config = load_config(config_path)
    reports: list[tuple[int, Path, dict[str, Any]]] = []
    for seed in config["seeds"]:
        if args.execute:
            path = run_seed(config, int(seed))
        elif args.assemble_components:
            path = assemble_existing_seed(config, int(seed))
        else:
            path = resolve(render(config["report_pattern"], int(seed)))
        reports.append((int(seed), path, read_json(path)))
    report = aggregate(config, reports)
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"trigger_state": report["trigger_state"], "disposition": report["disposition"], "checks": report["checks"], "metrics": report["metrics"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
