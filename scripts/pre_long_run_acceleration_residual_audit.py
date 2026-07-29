#!/usr/bin/env python3
"""Close the finite, evidence-triggered pre-long-run acceleration docket."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/pre_long_run_acceleration_residual_audit.json"
POLICY = "project_theseus_pre_long_run_acceleration_residual_audit_v1"


class ResidualAuditFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResidualAuditFault(f"json_object_required:{relative(path)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trainer_recipe(trainer: dict[str, Any]) -> dict[str, Any]:
    execution = trainer["training"]["execution_policy"]
    pretraining = execution["pretraining"]
    topology = trainer["topology"]
    return {
        "compute_dtype": execution["compute_dtype"],
        "fp32_master": execution["fp32_master"],
        "token_loss_compute_dtype": execution["token_loss_compute_dtype"],
        "training_step_mode": pretraining["training_step_mode"],
        "compiled_microbatch_size": int(pretraining["compiled_microbatch_size"]),
        "compile_width_quantum": int(pretraining["compile_width_quantum"]),
        "optimizer_id": trainer["training"]["optimizer_id"],
        "training_rope_kernel": trainer["training"]["training_rope_kernel"],
        "self_attention_projection": topology.get(
            "self_attention_projection", "separate"
        ),
        "feed_forward_activation": topology.get(
            "feed_forward_activation", "swiglu"
        ),
        "residual_policy": topology.get(
            "residual_policy", "sequential_unscaled"
        ),
        "per_head_muon": trainer["training"]["optimizer_id"]
        == "per_head_muon_mlx",
    }


def validate_lineage(
    lineage_root: Path,
    *,
    expected_count: int,
    first_step: int,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    manifests = sorted(lineage_root.glob("step-*_to_*/manifest.json"))
    faults: list[str] = []
    prior_after: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = []
    for index, path in enumerate(manifests):
        manifest = read_json(path)
        before = manifest.get("before_identity") or {}
        after = manifest.get("after_identity") or {}
        if index == 0 and int(before.get("optimizer_steps") or -1) != first_step:
            faults.append("prospective_anchor_start_mismatch")
        if prior_after is not None and before != prior_after:
            faults.append(f"lineage_identity_discontinuity:{relative(path)}")
        for name, row in (manifest.get("artifacts") or {}).items():
            artifact_path = resolve(row["path"])
            if not artifact_path.is_file():
                faults.append(f"lineage_artifact_missing:{name}:{relative(path)}")
            elif sha256_file(artifact_path) != row["sha256"]:
                faults.append(f"lineage_artifact_hash_mismatch:{name}:{relative(path)}")
        evidence.append(
            {
                "path": relative(path),
                "sha256": sha256_file(path),
                "before_step": before.get("optimizer_steps"),
                "after_step": after.get("optimizer_steps"),
            }
        )
        prior_after = after
    if len(manifests) != expected_count:
        faults.append(f"lineage_manifest_count:{len(manifests)}")
    if prior_after is None:
        faults.append("lineage_empty")
    else:
        terminal_expectations = {
            "optimizer_steps": receipt.get("optimizer_steps"),
            "optimizer_positions": receipt.get("optimizer_positions"),
            "checkpoint_sha256": receipt.get("checkpoint_sha256"),
            "optimizer_state_sha256": receipt.get("optimizer_state_sha256"),
            "mlx_rng_state_sha256": receipt.get("mlx_rng_state_sha256"),
        }
        for key, value in terminal_expectations.items():
            if prior_after.get(key) != value:
                faults.append(f"lineage_terminal_mismatch:{key}")
    return {
        "passed": not faults,
        "faults": faults,
        "manifest_count": len(manifests),
        "first_prospective_step": (
            evidence[0]["before_step"] if evidence else None
        ),
        "terminal_step": evidence[-1]["after_step"] if evidence else None,
        "pre_anchor_chain_available": False,
        "pre_anchor_boundary": (
            "The append-only evidence begins at the prospective step-9048 "
            "anchor; it makes no complete-chain claim before that point."
        ),
        "manifests": evidence,
    }


def execute(
    config_path: Path = DEFAULT_CONFIG,
    *,
    publish_report: bool = True,
) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("policy") != POLICY:
        raise ResidualAuditFault("config_policy_invalid")

    evidence: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name, value in config["evidence"].items():
        path = resolve(value)
        evidence[name] = (path, read_json(path))

    trainer_path = resolve(config["production_trainer_config"])
    trainer = read_json(trainer_path)
    candidate_path = resolve(config["candidate_contract"])
    candidate_contract = read_json(candidate_path)
    review_path = resolve(config["pre_long_run_review"])
    review = read_json(review_path)
    checkpoint_root = resolve(config["checkpoint_root"])
    receipt_path = checkpoint_root / "training_receipt.json"
    receipt = read_json(receipt_path)
    expected = config["expected_checkpoint"]

    checkpoint_files = {
        "model": resolve(receipt["checkpoint"]),
        "optimizer": resolve(receipt["optimizer_state"]),
        "mlx_rng": resolve(receipt["mlx_rng_state"]),
        "receipt": receipt_path,
    }
    checkpoint_hashes = {
        name: sha256_file(path) for name, path in checkpoint_files.items()
    }
    checkpoint_hash_match = {
        "model": checkpoint_hashes["model"] == receipt["checkpoint_sha256"],
        "optimizer": (
            checkpoint_hashes["optimizer"] == receipt["optimizer_state_sha256"]
        ),
        "mlx_rng": (
            checkpoint_hashes["mlx_rng"] == receipt["mlx_rng_state_sha256"]
        ),
    }
    alias_pairs = {
        "model": (checkpoint_root / "weights.safetensors", checkpoint_files["model"]),
        "optimizer": (
            checkpoint_root / "optimizer.safetensors",
            checkpoint_files["optimizer"],
        ),
        "mlx_rng": (
            checkpoint_root / "optimizer.mlx-rng.safetensors",
            checkpoint_files["mlx_rng"],
        ),
    }
    alias_hash_match = {
        name: sha256_file(alias) == sha256_file(versioned)
        for name, (alias, versioned) in alias_pairs.items()
    }
    lineage = validate_lineage(
        resolve(config["lineage_root"]),
        expected_count=int(expected["lineage_manifest_count"]),
        first_step=int(expected["first_prospective_step"]),
        receipt=receipt,
    )

    selector = evidence["final_selector"][1]
    sustained = evidence["sustained_route"][1]
    fresh = evidence["fresh_process_replay"][1]
    ane_join = evidence["ane_exact_block_join"][1]
    ane_bakeoff = evidence["ane_backend_bakeoff"][1]
    joined_data = evidence["joined_data_path"][1]
    data_stage = evidence["data_stage_qualification"][1]
    qkv = evidence["fused_qkv_full_route"][1]
    per_head = evidence["per_head_muon"][1]
    attnres = evidence["kimi_attnres"][1]
    situ = evidence["kimi_situ_glu"][1]

    global_safety = candidate_contract["host_safety_policy"]
    inline_reserves_zero = all(
        float((row.get("host_safety_overrides") or {}).get(key, 0)) == 0.0
        for row in candidate_contract["canaries"]
        for key in (
            "minimum_available_before_launch_mib",
            "minimum_available_during_run_mib",
        )
    )
    transaction_bytes = sum(path.stat().st_size for path in checkpoint_files.values())
    required_disk_bytes = transaction_bytes * 2
    disk_free_bytes = shutil.disk_usage(checkpoint_root).free
    recipe = trainer_recipe(trainer)
    reconciliations = {
        row["id"]: row for row in review["completed_evidence_reconciliations"]
    }
    docket = {row["id"]: row for row in review["finite_candidate_docket"]}

    domains = {
        "mlx_metal": {
            "passed": (
                selector.get("trigger_state") == "GREEN"
                and all(selector.get("gates", {}).values())
                and selector["campaign_disposition"][
                    "launch_recipe_changed_by_challenger"
                ]
                is False
                and recipe == config["selected_recipe"]
            ),
            "disposition": "compiled_fp32_mlx_retained",
        },
        "cpu_rust": {
            "passed": selector["gates"][
                "rust_host_rewrite_rejected_by_measured_amdahl_bound"
            ]
            is True,
            "disposition": "host_rewrite_closed_without_new_cpu_profile",
        },
        "ane_accelerate_metal": {
            "passed": (
                ane_join["gates"]["complete_decoder_block_mechanics"] is True
                and ane_join["gates"]["replay_exact"] is True
                and ane_bakeoff["state"] == "GREEN_MATCHED_BAKEOFF_RETAIN_MLX"
                and ane_bakeoff["selection"]["native_selected"] is False
                and ane_bakeoff["selection"]["retained_backend"] == "compiled_mlx"
                and ane_bakeoff["gates"]["replay_and_stability"] is True
                and ane_bakeoff["gates"]["native_joined_wall_beats_mlx_mean"]
                is False
                and ane_bakeoff["gates"][
                    "native_joined_wall_beats_mlx_conservative"
                ]
                is False
            ),
            "disposition": "exact_native_decoder_block_not_selected",
        },
        "data_pipeline": {
            "passed": (
                joined_data.get("trigger_state") == "GREEN"
                and joined_data.get("canonical_lineage_unchanged") is True
                and joined_data["final_resume_validation"]["state"] == "GREEN"
                and data_stage.get("trigger_state") == "GREEN"
                and data_stage["source_conditioned_pretraining"]["state"] == "GREEN"
                and data_stage["supervision"]["state"] == "GREEN"
                and reconciliations["source_conditioned_and_supervision_execution"][
                    "state"
                ]
                == "QUALIFIED_SAFE_NOT_CALENDAR_DOMINANT"
            ),
            "disposition": "joined_auxiliary_path_qualified_not_calendar_dominant",
        },
        "checkpoint_replay": {
            "passed": (
                receipt.get("optimizer_steps") == expected["optimizer_steps"]
                and receipt.get("optimizer_positions")
                == expected["optimizer_positions"]
                and receipt.get("capability_claim") == expected["capability_claim"]
                and all(checkpoint_hash_match.values())
                and all(alias_hash_match.values())
                and lineage["passed"] is True
                and sustained.get("trigger_state") == "GREEN"
                and sustained.get("canonical_lineage_unchanged") is True
                and sustained.get("exact_resume_each_segment") is True
                and fresh.get("trigger_state") == "GREEN"
                and fresh.get("canonical_lineage_unchanged") is True
                and fresh.get("exact_resume_validation") is True
                and fresh.get("independent_segmented_replay_numeric_parity")
                is True
            ),
            "disposition": "exact_step_11416_custody_and_replay_green",
        },
        "memory_disk": {
            "passed": (
                trainer["host_resource_safety"]["memory_guard_mode"]
                == "predicted_exhaustion"
                and float(
                    trainer["host_resource_safety"][
                        "minimum_available_before_launch_mib"
                    ]
                )
                == 0.0
                and float(
                    trainer["host_resource_safety"][
                        "minimum_available_during_run_mib"
                    ]
                )
                == 0.0
                and trainer["host_resource_safety"]["swapout_growth_action"]
                == "report_only"
                and float(
                    trainer["host_resource_safety"]["maximum_wall_seconds"]
                )
                == 0.0
                and global_safety["memory_guard_mode"] == "predicted_exhaustion"
                and float(global_safety["minimum_available_before_launch_mib"])
                == 0.0
                and float(global_safety["minimum_available_during_run_mib"])
                == 0.0
                and global_safety["swapout_growth_action"] == "report_only"
                and inline_reserves_zero
                and disk_free_bytes >= required_disk_bytes
            ),
            "disposition": "predicted_exhaustion_and_derived_two_transaction_disk_custody",
            "disk_free_bytes": disk_free_bytes,
            "checkpoint_transaction_bytes": transaction_bytes,
            "two_transaction_required_bytes": required_disk_bytes,
            "fixed_available_memory_floor_mib": 0,
        },
        "thermal": {
            "passed": (
                sustained["thermal_stability"]["terminal"] is True
                and sustained["thermal_stability"]["thermal_warning_observed"]
                is False
                and sustained["thermal_stability"]["state"]
                == "STABLE_WITHIN_OBSERVED_REPLICATE_UNCERTAINTY"
                and sustained.get("elapsed_time_requirement") is None
                and sustained.get("arbitrary_percentage_tolerance") is None
            ),
            "disposition": "stable_within_observed_replicate_uncertainty",
        },
        "joined_wall": {
            "passed": (
                qkv["selection"]["candidate_selected"] is False
                and qkv["selection"]["production_route_changed"] is False
                and qkv["selection"]["arbitrary_percentage_hurdle"] is False
                and qkv["timing"]["pair_count"] == 3
                and qkv["timing"]["candidate_win_count"] < 3
                and qkv["gates"]["pooled_direction_positive"] is False
                and qkv["production_authority"]["live_checkpoint_mutated"] is False
            ),
            "disposition": "separate_qkv_retained_after_full_route_pairs",
        },
        "architecture_candidates": {
            "passed": (
                per_head.get("trigger_state") == "GREEN"
                and per_head["comparisons"]["per_head_muon_mlx"]["disposition"]
                == "NOT_SELECTED_FIRST_CAMPAIGN"
                and attnres.get("trigger_state") == "GREEN"
                and attnres["campaign_disposition"]["selected_architecture"]
                == "control"
                and situ.get("trigger_state") == "GREEN"
                and situ["campaign_disposition"]["selected_architecture"]
                == "control"
                and all(
                    docket[candidate]["state"].startswith("COMPLETED_NOT_SELECTED")
                    for candidate in (
                        "kimi_k3_per_head_muon",
                        "kimi_k3_attention_residuals",
                        "kimi_k3_situ_glu",
                    )
                )
            ),
            "disposition": "all_finite_kimi_k3_candidates_completed_not_selected",
        },
        "custody_hold": {
            "passed": (
                resolve(config["yield_control"]).is_file()
                and review.get("state") == "HOLD_FOR_FINITE_REVIEW"
                and review["execution_hold"]["in_flight_transaction_interrupted"]
                is False
                and review["execution_hold"]["new_long_segment_authorized"]
                is False
                and review["execution_hold"][
                    "new_architecture_may_touch_live_checkpoint"
                ]
                is False
                and review["resource_policy"]["clock_of_day_windows_allowed"]
                is False
                and review["resource_policy"]["arbitrary_hour_limit_allowed"]
                is False
                and review["resource_policy"]["arbitrary_memory_floor_allowed"]
                is False
            ),
            "disposition": "long_training_remains_prohibited",
        },
    }
    required_domains = list(config["required_domains"])
    missing_domains = sorted(set(required_domains) - set(domains))
    failed_domains = [
        name for name in required_domains if not domains.get(name, {}).get("passed")
    ]
    trigger_state = (
        "GREEN" if not missing_domains and not failed_domains else "RED"
    )
    report = {
        "policy": POLICY,
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "trigger_state": trigger_state,
        "support_state": (
            "SUPPORTED_FINITE_ACCELERATION_DOCKET_CLOSED"
            if trigger_state == "GREEN"
            else "UNSUPPORTED_OPEN_RESIDUAL_OR_CUSTODY_FAULT"
        ),
        "required_domains": required_domains,
        "missing_domains": missing_domains,
        "failed_domains": failed_domains,
        "domains": domains,
        "selected_recipe": recipe,
        "checkpoint_custody": {
            "optimizer_steps": receipt.get("optimizer_steps"),
            "optimizer_positions": receipt.get("optimizer_positions"),
            "capability_claim": receipt.get("capability_claim"),
            "files": {
                name: {
                    "path": relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": checkpoint_hashes[name],
                }
                for name, path in checkpoint_files.items()
            },
            "versioned_hashes_match_receipt": checkpoint_hash_match,
            "canonical_alias_hashes_match_versioned": alias_hash_match,
            "lineage": lineage,
        },
        "evidence": {
            name: {
                "path": relative(path),
                "sha256": sha256_file(path),
            }
            for name, (path, _report) in evidence.items()
        },
        "authority": {
            "production_trainer_config": {
                "path": relative(trainer_path),
                "sha256": sha256_file(trainer_path),
            },
            "candidate_contract": {
                "path": relative(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "pre_long_run_review": {
                "path": relative(review_path),
                "sha256": sha256_file(review_path),
            },
            "yield_control": {
                "path": relative(resolve(config["yield_control"])),
                "sha256": sha256_file(resolve(config["yield_control"])),
            },
        },
        "selection_claim": config["selection_claim"],
        "reentry_rule": (
            "Reopen only for a materially new source, mechanism, hardware "
            "capability, or measured critical-path residual with a plausible "
            "joined-wall benefit; preregister one bounded matched test."
        ),
        "long_training_started_or_resumed": False,
        "capability_claim": "NONE_ACCELERATION_READINESS_AND_CUSTODY_ONLY",
        "non_claims": config["non_claims"],
    }
    if publish_report:
        report_path = resolve(config["report"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    report = execute(resolve(args.config), publish_report=not args.no_publish)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["trigger_state"] == "GREEN" else 1)


if __name__ == "__main__":
    main()
