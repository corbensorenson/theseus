#!/usr/bin/env python3
"""Independently audit the finite pre-long-run selection and custody claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/pre_long_run_independent_readiness_audit.json"
POLICY = "project_theseus_pre_long_run_independent_readiness_audit_v1"


class IndependentAuditFault(ValueError):
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
        raise IndependentAuditFault(f"json_object_required:{relative(path)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def production_recipe(trainer: dict[str, Any]) -> dict[str, Any]:
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


def recompute_lineage(
    root: Path,
    *,
    expected_count: int,
    anchor_step: int,
    terminal: dict[str, Any],
) -> dict[str, Any]:
    manifests = sorted(root.glob("step-*_to_*/manifest.json"))
    faults: list[str] = []
    previous: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(manifests):
        row = read_json(path)
        before = row.get("before_identity") or {}
        after = row.get("after_identity") or {}
        if index == 0 and before.get("optimizer_steps") != anchor_step:
            faults.append("anchor_step_mismatch")
        if previous is not None and before != previous:
            faults.append(f"identity_discontinuity:{relative(path)}")
        for artifact in (row.get("artifacts") or {}).values():
            artifact_path = resolve(artifact["path"])
            if not artifact_path.is_file():
                faults.append(f"artifact_missing:{relative(artifact_path)}")
            elif sha256_file(artifact_path) != artifact["sha256"]:
                faults.append(
                    f"artifact_hash_mismatch:{relative(artifact_path)}"
                )
        rows.append(
            {
                "path": relative(path),
                "sha256": sha256_file(path),
                "before": before,
                "after": after,
            }
        )
        previous = after
    if len(rows) != expected_count:
        faults.append(f"manifest_count_mismatch:{len(rows)}")
    terminal_keys = (
        "optimizer_steps",
        "optimizer_positions",
        "checkpoint_sha256",
        "optimizer_state_sha256",
        "mlx_rng_state_sha256",
    )
    if previous is None:
        faults.append("lineage_empty")
    else:
        for key in terminal_keys:
            if previous.get(key) != terminal.get(key):
                faults.append(f"terminal_identity_mismatch:{key}")
    return {
        "passed": not faults,
        "faults": faults,
        "manifest_count": len(rows),
        "anchor_step": rows[0]["before"].get("optimizer_steps") if rows else None,
        "terminal_step": rows[-1]["after"].get("optimizer_steps") if rows else None,
        "pre_anchor_full_chain_available": False,
        "rows": rows,
    }


def matching_consumption_rows(
    registry_path: Path,
    freeze: dict[str, Any],
    freeze_sha256: str,
) -> list[dict[str, Any]]:
    needles = {
        freeze_sha256,
        str(freeze["candidate_id"]),
        str(freeze["candidate_packet_sha256"]),
        str(freeze["case_contract_sha256"]),
    }
    matches: list[dict[str, Any]] = []
    if not registry_path.is_file():
        return matches
    for line_number, line in enumerate(
        registry_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        serialized = json.dumps(row, sort_keys=True)
        if any(needle in serialized for needle in needles):
            matches.append({"line": line_number, "row": row})
    return matches


def execute(
    config_path: Path = DEFAULT_CONFIG,
    *,
    publish_report: bool = True,
) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("policy") != POLICY:
        raise IndependentAuditFault("config_policy_invalid")

    residual_path = resolve(config["residual_audit"])
    residual = read_json(residual_path)
    residual_config_path = resolve(config["residual_audit_config"])
    trainer_path = resolve(config["production_trainer_config"])
    trainer = read_json(trainer_path)
    candidate_contract_path = resolve(config["candidate_contract"])
    candidate_contract = read_json(candidate_contract_path)
    review_path = resolve(config["pre_long_run_review"])
    review = read_json(review_path)
    freeze_path = resolve(config["functional_freeze"])
    freeze = read_json(freeze_path)
    registry_path = resolve(config["functional_consumption_registry"])
    candidate_reports = {
        name: (resolve(path), read_json(resolve(path)))
        for name, path in config["candidate_reports"].items()
    }
    replay_reports = {
        name: (resolve(path), read_json(resolve(path)))
        for name, path in config["replay_reports"].items()
    }
    expected = config["expected"]

    residual_hash_faults: list[str] = []
    if residual.get("config_sha256") != sha256_file(residual_config_path):
        residual_hash_faults.append("residual_config_hash_mismatch")
    for section in ("evidence", "authority"):
        for name, row in (residual.get(section) or {}).items():
            path = resolve(row["path"])
            if not path.is_file():
                residual_hash_faults.append(
                    f"{section}_missing:{name}:{relative(path)}"
                )
            elif sha256_file(path) != row["sha256"]:
                residual_hash_faults.append(
                    f"{section}_hash_mismatch:{name}:{relative(path)}"
                )

    checkpoint_root = resolve(config["checkpoint_root"])
    receipt_path = checkpoint_root / "training_receipt.json"
    receipt = read_json(receipt_path)
    versioned = {
        "model": resolve(receipt["checkpoint"]),
        "optimizer": resolve(receipt["optimizer_state"]),
        "mlx_rng": resolve(receipt["mlx_rng_state"]),
    }
    expected_hashes = {
        "model": receipt["checkpoint_sha256"],
        "optimizer": receipt["optimizer_state_sha256"],
        "mlx_rng": receipt["mlx_rng_state_sha256"],
    }
    versioned_hashes = {
        name: sha256_file(path) for name, path in versioned.items()
    }
    aliases = {
        "model": checkpoint_root / "weights.safetensors",
        "optimizer": checkpoint_root / "optimizer.safetensors",
        "mlx_rng": checkpoint_root / "optimizer.mlx-rng.safetensors",
    }
    alias_hashes = {name: sha256_file(path) for name, path in aliases.items()}
    checkpoint_hashes_pass = all(
        versioned_hashes[name] == expected_hashes[name]
        and alias_hashes[name] == versioned_hashes[name]
        for name in versioned
    )
    terminal = {
        "optimizer_steps": receipt.get("optimizer_steps"),
        "optimizer_positions": receipt.get("optimizer_positions"),
        "checkpoint_sha256": expected_hashes["model"],
        "optimizer_state_sha256": expected_hashes["optimizer"],
        "mlx_rng_state_sha256": expected_hashes["mlx_rng"],
    }
    lineage = recompute_lineage(
        resolve(config["lineage_root"]),
        expected_count=int(expected["lineage_manifest_count"]),
        anchor_step=int(expected["prospective_anchor_step"]),
        terminal=terminal,
    )

    fresh = replay_reports["fresh_process"][1]
    sustained = replay_reports["sustained_route"][1]
    global_safety = candidate_contract["host_safety_policy"]
    inline_reserves_zero = all(
        float((row.get("host_safety_overrides") or {}).get(key, 0)) == 0.0
        for row in candidate_contract["canaries"]
        for key in (
            "minimum_available_before_launch_mib",
            "minimum_available_during_run_mib",
        )
    )
    transaction_bytes = (
        sum(path.stat().st_size for path in versioned.values())
        + receipt_path.stat().st_size
    )
    disk_free_bytes = shutil.disk_usage(checkpoint_root).free

    freeze_sha256 = sha256_file(freeze_path)
    consumption_matches = matching_consumption_rows(
        registry_path, freeze, freeze_sha256
    )
    per_head = candidate_reports["per_head_muon"][1]
    attnres = candidate_reports["attention_residuals"][1]
    situ = candidate_reports["situ_glu"][1]
    qkv = candidate_reports["fused_qkv"][1]
    scoped_candidates = (per_head, attnres, situ)
    selected_recipe = production_recipe(trainer)

    audits = {
        "evidence_integrity": {
            "passed": (
                not residual_hash_faults
                and residual.get("trigger_state") == "GREEN"
                and residual.get("failed_domains") == []
                and residual.get("long_training_started_or_resumed") is False
            ),
            "faults": residual_hash_faults,
            "independence": (
                "Every path and digest asserted by the residual audit was "
                "recomputed from disk; its GREEN label alone was insufficient."
            ),
        },
        "lineage_custody": {
            "passed": (
                checkpoint_hashes_pass
                and lineage["passed"] is True
                and receipt.get("optimizer_steps") == expected["optimizer_steps"]
                and receipt.get("optimizer_positions")
                == expected["optimizer_positions"]
                and receipt.get("capability_claim") == expected["capability_claim"]
            ),
            "checkpoint_hashes_pass": checkpoint_hashes_pass,
            "versioned_hashes": versioned_hashes,
            "alias_hashes": alias_hashes,
            "lineage": lineage,
        },
        "replay_integrity": {
            "passed": (
                fresh.get("trigger_state") == "GREEN"
                and fresh.get("canonical_lineage_unchanged") is True
                and fresh.get("exact_resume_validation") is True
                and fresh.get("independent_segmented_replay_numeric_parity")
                is True
                and fresh.get("host_resource_guard_passed") is True
                and sustained.get("trigger_state") == "GREEN"
                and sustained.get("canonical_lineage_unchanged") is True
                and sustained.get("exact_resume_each_segment") is True
                and sustained.get("hard_gaps") == []
            ),
            "fresh_process_report_sha256": sha256_file(
                replay_reports["fresh_process"][0]
            ),
            "sustained_report_sha256": sha256_file(
                replay_reports["sustained_route"][0]
            ),
        },
        "resource_integrity": {
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
                and disk_free_bytes >= 2 * transaction_bytes
                and sustained["thermal_stability"]["terminal"] is True
                and sustained["thermal_stability"]["thermal_warning_observed"]
                is False
                and sustained.get("elapsed_time_requirement") is None
                and sustained.get("arbitrary_percentage_tolerance") is None
            ),
            "disk_free_bytes": disk_free_bytes,
            "checkpoint_transaction_bytes": transaction_bytes,
            "two_transaction_required_bytes": 2 * transaction_bytes,
            "fixed_available_memory_floor_mib": 0,
        },
        "evaluation_nonconsumption": {
            "passed": (
                freeze.get("candidate_id")
                == "moecot_mlx_57m_active_preregistered_v1"
                and freeze.get("immutable") is True
                and freeze.get("source_disjoint") is True
                and freeze.get("consumed_case_count") == 0
                and freeze.get("evaluation_state") == "NOT_EVALUATED"
                and freeze.get("capability_claim") == "NOT_EVALUATED"
                and freeze.get("public_training_rows_written") == 0
                and freeze.get("external_inference_calls") == 0
                and not consumption_matches
                and receipt.get("public_training_rows_written") == 0
                and receipt.get("external_inference_calls") == 0
            ),
            "freeze_path": relative(freeze_path),
            "freeze_sha256": freeze_sha256,
            "registry_path": relative(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "matching_consumption_rows": consumption_matches,
        },
        "negative_evidence_scope": {
            "passed": (
                all(
                    report["gates"]["source_disjoint"] is True
                    and report["gates"]["no_public_external_or_fallback"] is True
                    and report["campaign_disposition"][
                        "scientific_falsification_claimed"
                    ]
                    is False
                    for report in scoped_candidates
                )
                and per_head["campaign_disposition"]["selected_optimizer"]
                == "adamw_mlx"
                and attnres["campaign_disposition"]["selected_architecture"]
                == "control"
                and situ["campaign_disposition"]["selected_architecture"]
                == "control"
                and qkv["selection"]["candidate_selected"] is False
                and qkv["selection"]["production_route_changed"] is False
                and qkv["production_authority"]["live_checkpoint_mutated"] is False
            ),
            "candidate_report_sha256s": {
                name: sha256_file(path)
                for name, (path, _report) in candidate_reports.items()
            },
            "scientific_falsification_claimed": False,
        },
        "claim_boundary": {
            "passed": (
                residual.get("selection_claim")
                == (
                    "No currently source- or measured-profile-justified open "
                    "acceleration residual remains among the tested eligible routes."
                )
                and residual.get("capability_claim")
                == "NONE_ACCELERATION_READINESS_AND_CUSTODY_ONLY"
                and any(
                    "does not prove" in row.lower()
                    and "fastest" in row.lower()
                    for row in residual.get("non_claims", [])
                )
                and selected_recipe == expected["selected_recipe"]
                and receipt.get("capability_claim") == "NOT_EVALUATED"
            ),
            "authorized_claim": residual.get("selection_claim"),
            "forbidden_claims": [
                "fastest physically possible",
                "capability or utility",
                "architecture superiority",
                "broad scientific falsification",
                "long-campaign outcome",
            ],
            "selected_recipe": selected_recipe,
        },
        "execution_hold": {
            "passed": (
                resolve(config["yield_control"]).is_file()
                and review.get("state") == "HOLD_FOR_FINITE_REVIEW"
                and review["execution_hold"]["new_long_segment_authorized"]
                is False
                and review["execution_hold"][
                    "new_architecture_may_touch_live_checkpoint"
                ]
                is False
                and review["execution_hold"][
                    "in_flight_transaction_interrupted"
                ]
                is False
            ),
            "yield_control_path": config["yield_control"],
            "yield_control_sha256": sha256_file(
                resolve(config["yield_control"])
            ),
            "long_training_started_or_resumed": False,
        },
    }
    required = list(config["required_audits"])
    missing = sorted(set(required) - set(audits))
    failed = [name for name in required if not audits.get(name, {}).get("passed")]
    trigger_state = "GREEN" if not missing and not failed else "RED"
    report = {
        "policy": POLICY,
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "trigger_state": trigger_state,
        "support_state": (
            "SUPPORTED_INDEPENDENT_PRE_LONG_RUN_READINESS_AUDIT"
            if trigger_state == "GREEN"
            else "UNSUPPORTED_INTEGRITY_OR_CLAIM_BOUNDARY_FAULT"
        ),
        "required_audits": required,
        "missing_audits": missing,
        "failed_audits": failed,
        "audits": audits,
        "authority": {
            "residual_audit": {
                "path": relative(residual_path),
                "sha256": sha256_file(residual_path),
            },
            "production_trainer_config": {
                "path": relative(trainer_path),
                "sha256": sha256_file(trainer_path),
            },
            "candidate_contract": {
                "path": relative(candidate_contract_path),
                "sha256": sha256_file(candidate_contract_path),
            },
            "pre_long_run_review": {
                "path": relative(review_path),
                "sha256": sha256_file(review_path),
            },
            "functional_freeze": {
                "path": relative(freeze_path),
                "sha256": freeze_sha256,
            },
        },
        "decision": (
            "INDEPENDENT_AUDITS_GREEN_READY_FOR_CONTENT_ADDRESSED_REPLACEMENT_FREEZE"
            if trigger_state == "GREEN"
            else "BLOCK_REPLACEMENT_FREEZE"
        ),
        "long_training_authorized": False,
        "capability_claim": "NONE_INDEPENDENT_READINESS_AUDIT_ONLY",
        "non_claims": [
            "This audit does not prove physical performance optimality.",
            "This audit does not evaluate model capability or utility.",
            "This audit does not authorize or start long training.",
        ],
    }
    if publish_report:
        output_path = resolve(config["report"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
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
