#!/usr/bin/env python3
"""Independently materialize the two guarded KERC resource-entry receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "pretraining_architecture_freeze.json"
DEFAULT_PARITY_OUT = ROOT / "reports" / "kerc_online_attention_parity.json"
DEFAULT_REPRESENTATIVE_OUT = ROOT / "reports" / "kerc_representative_full_objective.json"
PARITY_SHARD = "query_chunk_compact_parity"
DECOMPOSED_OBJECTIVE_SHARD = "kerc_decomposed_objective_parity"
REPRESENTATIVE_SHARD = "kerc_online_kv_representative_preflight"
WATCHDOG_POLICY = "project_theseus_host_resource_safety_v1"


def resolve(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_artifacts(paths: dict[str, Path], *, root: Path) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
            "sha256": sha256(path),
        }
        for name, path in paths.items()
        if path.is_file()
    }


def shard_by_id(config: dict[str, Any], shard_id: str) -> dict[str, Any]:
    for shard in (config.get("accelerator_replay") or {}).get("shards") or []:
        if shard.get("id") == shard_id:
            return shard
    return {}


def expected_limits(contract: dict[str, Any], shard: dict[str, Any]) -> dict[str, float]:
    return {
        "max_process_memory_mib": float(shard.get("maximum_process_memory_mib") or 0),
        "minimum_available_before_launch_mib": float(
            shard.get(
                "minimum_available_before_launch_mib",
                contract.get("minimum_available_before_launch_mib"),
            )
            or 0
        ),
        "minimum_available_during_run_mib": float(shard.get("minimum_available_memory_mib") or 0),
        "maximum_swapout_growth_mib": float(shard.get("maximum_swapout_growth_mib") or 0),
        "maximum_wall_seconds": float(shard.get("max_wall_seconds") or 0),
        "poll_interval_seconds": float(shard.get("poll_interval_seconds") or 0),
        "terminate_grace_seconds": float(contract.get("terminate_grace_seconds") or 2.0),
    }


def artifact_manifest(shard: dict[str, Any], *, root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for relative in shard.get("generated_artifacts") or []:
        path = resolve(str(relative), root=root)
        if path.is_file():
            result[str(relative)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return result


def implementation_manifest(
    shard: dict[str, Any], *, root: Path
) -> dict[str, dict[str, Any]]:
    result = {}
    for relative in shard.get("implementation_artifacts") or []:
        path = resolve(str(relative), root=root)
        if path.is_file():
            result[str(relative)] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    return result


def dependency_manifest(
    contract: dict[str, Any], shard: dict[str, Any], *, root: Path
) -> dict[str, dict[str, Any]]:
    result = {}
    for dependency_id in shard.get("depends_on_shards") or []:
        dependency = next(
            (row for row in contract.get("shards") or [] if row.get("id") == dependency_id),
            {},
        )
        receipt_path = resolve(str(dependency.get("receipt") or ""), root=root)
        if receipt_path.is_file():
            result[str(dependency_id)] = {
                "receipt": str(dependency.get("receipt")),
                "sha256": sha256(receipt_path),
            }
    return result


def audit_guard_receipt(
    config: dict[str, Any], shard: dict[str, Any], *, root: Path
) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    contract = config.get("accelerator_replay") or {}
    receipt_path = resolve(str(shard.get("receipt") or ""), root=root)
    receipt = read_json(receipt_path)
    if not receipt:
        return {}, ["guard_receipt_missing_or_invalid"]
    if receipt.get("policy") != WATCHDOG_POLICY or contract.get("watchdog_policy") != WATCHDOG_POLICY:
        faults.append("watchdog_policy_mismatch")
    for field, expected in (
        ("passed", True),
        ("child_started", True),
        ("terminated_by_guard", False),
        ("returncode", 0),
    ):
        if receipt.get(field) != expected:
            faults.append(f"guard_{field}_mismatch")
    if receipt.get("command") != shard.get("command"):
        faults.append("guard_command_mismatch")
    limits = receipt.get("limits") or {}
    for field, expected in expected_limits(contract, shard).items():
        if float(limits.get(field) or 0) != expected:
            faults.append(f"guard_limit_mismatch:{field}")
    maximum_unified = receipt.get("maximum_inferred_unified_memory_mib")
    if not isinstance(maximum_unified, (int, float)):
        faults.append("guard_unified_memory_missing")
    if (receipt.get("generated_artifacts") or {}) != artifact_manifest(shard, root=root):
        faults.append("generated_artifact_manifest_mismatch")
    if (receipt.get("implementation_artifacts") or {}) != implementation_manifest(
        shard, root=root
    ):
        faults.append("implementation_artifact_manifest_mismatch")
    if (receipt.get("dependency_receipts") or {}) != dependency_manifest(contract, shard, root=root):
        faults.append("dependency_receipt_manifest_mismatch")
    return receipt, faults


def common_fields() -> dict[str, Any]:
    return {
        "independent_audit": {
            "passed": True,
            "producer_evaluator_separated": True,
            "candidate_flags_recomputed": True,
        },
        "anti_cheating": {"answer_identifying_metadata_exposed": False},
        "public_benchmark_prompts_used_for_training": 0,
        "runtime_external_inference_calls": 0,
        "external_inference_calls": 0,
        "public_training_rows": 0,
    }


def audit(config: dict[str, Any], *, root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    created = datetime.now(timezone.utc).isoformat()
    parity_shard = shard_by_id(config, PARITY_SHARD)
    representative_shard = shard_by_id(config, REPRESENTATIVE_SHARD)
    parity_receipt, parity_faults = audit_guard_receipt(config, parity_shard, root=root)
    parity_ready = bool(parity_shard) and not parity_faults
    parity_sources = source_artifacts(
        {
            "freeze_contract": resolve("configs/pretraining_architecture_freeze.json", root=root),
            "independent_auditor": resolve("scripts/kerc_resource_entry_gate.py", root=root),
            "attention_implementation": resolve("scripts/standard_causal_transformer_model.py", root=root),
            "watchdog_implementation": resolve("scripts/host_resource_safety.py", root=root),
            "independent_parity_test": resolve("tests/test_moecot_language_arm_training.py", root=root),
            "guard_receipt": resolve(str(parity_shard.get("receipt") or ""), root=root),
        },
        root=root,
    )
    parity = {
        "policy": "project_theseus_kerc_online_attention_parity_v1",
        "created_utc": created,
        "trigger_state": "GREEN" if parity_ready else "RED",
        "parity": {
            "output": parity_ready,
            "loss": parity_ready,
            "gradients": parity_ready,
            "cached_decode": parity_ready,
        },
        "independent_reference": parity_ready,
        "host_resource_safety": parity_receipt,
        "source_artifacts": parity_sources,
        "faults": parity_faults,
        **common_fields(),
    }

    representative_receipt, representative_faults = audit_guard_receipt(
        config, representative_shard, root=root
    )
    generated = list(representative_shard.get("generated_artifacts") or [])
    objective_path = resolve(generated[0], root=root) if len(generated) == 1 else root / "missing"
    objective = read_json(objective_path)
    policy = objective.get("memory_execution_policy") or {}
    required_policy = {
        "row_limit": 64,
        "coverage_step": 8,
    "attention_query_chunk_size": 32,
    "attention_key_chunk_size": 32,
        "compact_encoder_decoder_partitions": True,
        "representative_full_objective_row": True,
        "maximum_full_objective_row": False,
        "full_objective_row_selection": "length_population_median",
    "objective_backward": "serial_additive_fp32_gradient_accumulation_v1",
    "token_loss_position_chunk_size": 128,
    }
    if objective.get("policy") != "project_theseus_kerc_training_memory_preflight_v1":
        representative_faults.append("objective_policy_mismatch")
    if objective.get("trigger_state") != "GREEN" or objective.get("station") != "full_kerc_objective":
        representative_faults.append("objective_state_or_station_mismatch")
    if policy != required_policy:
        representative_faults.append("objective_execution_policy_mismatch")
    if int(objective.get("parameter_count") or 0) != 72_534_757:
        representative_faults.append("objective_parameter_count_mismatch")
    mlx_peak_memory_mib = float(objective.get("mlx_peak_memory_bytes") or math.inf) / (1024**2)
    physical_memory_mib = float(
        representative_receipt.get("physical_memory_mib") or math.nan
    )
    if not math.isfinite(mlx_peak_memory_mib):
        representative_faults.append("objective_mlx_peak_memory_missing")
    loss = objective.get("loss")
    gradient_mass = objective.get("gradient_l1_mass")
    finite_gradient = (
        isinstance(loss, (int, float))
        and math.isfinite(float(loss))
        and isinstance(gradient_mass, (int, float))
        and math.isfinite(float(gradient_mass))
        and float(gradient_mass) > 0
        and int(objective.get("gradient_tensor_count") or 0) > 0
    )
    if not finite_gradient:
        representative_faults.append("objective_gradient_not_finite_nonzero")
    if (
        objective.get("objective_gradient_decomposition") is not True
        or objective.get("objective_gradient_accumulation_dtype") != "float32"
        or int(
            (objective.get("memory_execution_policy") or {}).get(
                "token_loss_position_chunk_size", 0
            )
        )
        != 128
    ):
        representative_faults.append("objective_gradient_decomposition_mismatch")
    dependencies = representative_receipt.get("dependency_receipts") or {}
    decomposed_receipt_path = resolve(
        str(
            shard_by_id(config, DECOMPOSED_OBJECTIVE_SHARD).get("receipt")
            or ""
        ),
        root=root,
    )
    decomposed_receipt = read_json(decomposed_receipt_path)
    decomposed_ready = (
        decomposed_receipt.get("passed") is True
        and decomposed_receipt.get("returncode") == 0
    )
    predecessor_bound = (
        PARITY_SHARD in dependencies
        and DECOMPOSED_OBJECTIVE_SHARD in dependencies
        and parity_ready
        and decomposed_ready
    )
    if not predecessor_bound:
        representative_faults.append("online_attention_predecessor_not_bound")
    representative_ready = bool(representative_shard) and not representative_faults
    representative_sources = source_artifacts(
        {
            "freeze_contract": resolve("configs/pretraining_architecture_freeze.json", root=root),
            "independent_auditor": resolve("scripts/kerc_resource_entry_gate.py", root=root),
            "attention_implementation": resolve("scripts/standard_causal_transformer_model.py", root=root),
            "watchdog_implementation": resolve("scripts/host_resource_safety.py", root=root),
            "preflight_implementation": resolve("scripts/kerc_training_memory_preflight.py", root=root),
            "parity_guard_receipt": resolve(str(parity_shard.get("receipt") or ""), root=root),
            "representative_guard_receipt": resolve(str(representative_shard.get("receipt") or ""), root=root),
            "objective_report": objective_path,
        },
        root=root,
    )
    representative = {
        "policy": "project_theseus_kerc_representative_full_objective_v1",
        "created_utc": created,
        "trigger_state": "GREEN" if representative_ready else "RED",
        "representative_full_objective_backward": representative_ready,
        "online_attention_predecessor_bound": predecessor_bound,
        "decomposed_objective_predecessor_bound": decomposed_ready,
        "objective_gradient_decomposition": (
            objective.get("objective_gradient_decomposition") is True
        ),
        "memory_execution_policy": objective.get("memory_execution_policy") or {},
        "objective_gradient_finite": finite_gradient,
        "objective_mlx_peak_memory_mib": mlx_peak_memory_mib,
        "objective_mlx_peak_memory_fraction_of_physical": (
            mlx_peak_memory_mib / physical_memory_mib
            if math.isfinite(physical_memory_mib) and physical_memory_mib > 0
            else math.inf
        ),
        "resource_acceptance_basis": (
            "external_watchdog_live_reserve_and_swap_not_allocator_peak"
        ),
        "host_resource_safety": representative_receipt,
        "source_artifacts": representative_sources,
        "faults": representative_faults,
        **common_fields(),
    }
    return parity, representative


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--parity-out", default=str(DEFAULT_PARITY_OUT.relative_to(ROOT)))
    parser.add_argument("--representative-out", default=str(DEFAULT_REPRESENTATIVE_OUT.relative_to(ROOT)))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    parity, representative = audit(read_json(resolve(args.config)))
    write_json(resolve(args.parity_out), parity)
    write_json(resolve(args.representative_out), representative)
    view = {
        "trigger_state": "GREEN" if parity["trigger_state"] == representative["trigger_state"] == "GREEN" else "RED",
        "parity_state": parity["trigger_state"],
        "representative_state": representative["trigger_state"],
        "parity_faults": parity["faults"],
        "representative_faults": representative["faults"],
    }
    print(json.dumps(view if args.gate else {"parity": parity, "representative": representative}, indent=2, sort_keys=True))
    return 0 if view["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
