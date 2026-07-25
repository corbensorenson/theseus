from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_resource_entry_gate as gate


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def fixture(tmp_path: Path) -> dict:
    for relative in (
        "configs/pretraining_architecture_freeze.json",
        "scripts/kerc_resource_entry_gate.py",
        "scripts/standard_causal_transformer_model.py",
        "scripts/kerc_training_memory_preflight.py",
        "tests/test_moecot_language_arm_training.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    contract = {
        "watchdog_policy": gate.WATCHDOG_POLICY,
        "minimum_available_before_launch_mib": 6144,
        "terminate_grace_seconds": 2,
        "shards": [],
    }
    parity = {
        "id": gate.PARITY_SHARD,
        "command": ["python", "-m", "pytest", "parity"],
        "receipt": "reports/accelerator_replay/parity.json",
        "max_wall_seconds": 120,
        "maximum_process_memory_mib": 1536,
        "minimum_available_before_launch_mib": 5632,
        "minimum_available_memory_mib": 4096,
        "maximum_swapout_growth_mib": 16,
        "poll_interval_seconds": 0.1,
    }
    decomposed = {
        "id": gate.DECOMPOSED_OBJECTIVE_SHARD,
        "command": ["python", "-m", "pytest", "decomposed"],
        "receipt": "reports/accelerator_replay/decomposed.json",
        "depends_on_shards": [gate.PARITY_SHARD],
        "max_wall_seconds": 180,
        "maximum_process_memory_mib": 1024,
        "minimum_available_before_launch_mib": 3072,
        "minimum_available_memory_mib": 2048,
        "maximum_swapout_growth_mib": 16,
        "poll_interval_seconds": 0.1,
    }
    representative = {
        "id": gate.REPRESENTATIVE_SHARD,
        "command": ["python", "preflight"],
        "receipt": "reports/accelerator_replay/representative.json",
        "depends_on_shards": [gate.PARITY_SHARD, gate.DECOMPOSED_OBJECTIVE_SHARD],
        "generated_artifacts": ["reports/objective.json"],
        "max_wall_seconds": 1800,
        "maximum_process_memory_mib": 5120,
        "maximum_mlx_peak_memory_fraction_of_physical": 0.4,
        "minimum_available_memory_mib": 4096,
        "maximum_swapout_growth_mib": 16,
        "poll_interval_seconds": 0.1,
    }
    contract["shards"] = [parity, decomposed, representative]
    config = {"accelerator_replay": contract}
    write(tmp_path / "configs/pretraining_architecture_freeze.json", config)
    parity_receipt_path = tmp_path / parity["receipt"]
    parity_receipt = {
        "policy": gate.WATCHDOG_POLICY,
        "passed": True,
        "child_started": True,
        "terminated_by_guard": False,
        "returncode": 0,
        "command": parity["command"],
        "limits": gate.expected_limits(contract, parity),
        "generated_artifacts": {},
        "dependency_receipts": {},
        "maximum_process_rss_mib": 200,
        "maximum_inferred_unified_memory_mib": 240,
        "maximum_swapout_growth_mib": 0,
        "physical_memory_mib": 16384,
    }
    write(parity_receipt_path, parity_receipt)
    decomposed_receipt_path = tmp_path / decomposed["receipt"]
    decomposed_receipt = {
        "policy": gate.WATCHDOG_POLICY,
        "passed": True,
        "child_started": True,
        "terminated_by_guard": False,
        "returncode": 0,
        "command": decomposed["command"],
        "limits": gate.expected_limits(contract, decomposed),
        "generated_artifacts": {},
        "dependency_receipts": {
            gate.PARITY_SHARD: {
                "receipt": parity["receipt"],
                "sha256": gate.sha256(parity_receipt_path),
            }
        },
        "maximum_process_rss_mib": 220,
        "maximum_inferred_unified_memory_mib": 280,
        "maximum_swapout_growth_mib": 0,
        "physical_memory_mib": 16384,
    }
    write(decomposed_receipt_path, decomposed_receipt)
    objective_path = tmp_path / "reports/objective.json"
    objective = {
        "policy": "project_theseus_kerc_training_memory_preflight_v1",
        "trigger_state": "GREEN",
        "station": "full_kerc_objective",
        "parameter_count": 72_534_757,
        "memory_execution_policy": {
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
        },
        "objective_gradient_decomposition": True,
        "objective_gradient_accumulation_dtype": "float32",
        "loss": 1.0,
        "gradient_l1_mass": 2.0,
        "gradient_tensor_count": 3,
        "mlx_peak_memory_bytes": 4800 * 1024 * 1024,
    }
    write(objective_path, objective)
    dependency = {
        gate.PARITY_SHARD: {
            "receipt": parity["receipt"],
            "sha256": gate.sha256(parity_receipt_path),
        },
        gate.DECOMPOSED_OBJECTIVE_SHARD: {
            "receipt": decomposed["receipt"],
            "sha256": gate.sha256(decomposed_receipt_path),
        }
    }
    representative_receipt = {
        "policy": gate.WATCHDOG_POLICY,
        "passed": True,
        "child_started": True,
        "terminated_by_guard": False,
        "returncode": 0,
        "command": representative["command"],
        "limits": gate.expected_limits(contract, representative),
        "generated_artifacts": {
            "reports/objective.json": {
                "sha256": gate.sha256(objective_path),
                "bytes": objective_path.stat().st_size,
            }
        },
        "dependency_receipts": dependency,
        "maximum_process_rss_mib": 4096,
        "maximum_inferred_unified_memory_mib": 4800,
        "maximum_swapout_growth_mib": 0,
        "physical_memory_mib": 16384,
    }
    write(tmp_path / representative["receipt"], representative_receipt)
    return config


def test_valid_resource_entry_receipts_are_independently_materialized(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    parity, representative = gate.audit(config, root=tmp_path)

    assert parity["trigger_state"] == "GREEN"
    assert gate.expected_limits(
        config["accelerator_replay"],
        config["accelerator_replay"]["shards"][0],
    )["minimum_available_before_launch_mib"] == 5632
    assert parity["parity"] == {"output": True, "loss": True, "gradients": True, "cached_decode": True}
    assert representative["trigger_state"] == "GREEN"
    assert representative["online_attention_predecessor_bound"] is True
    assert representative["objective_gradient_finite"] is True


def test_tampered_objective_fails_closed(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    objective = tmp_path / "reports/objective.json"
    payload = json.loads(objective.read_text(encoding="utf-8"))
    payload["gradient_l1_mass"] = 0
    write(objective, payload)

    _parity, representative = gate.audit(config, root=tmp_path)

    assert representative["trigger_state"] == "RED"
    assert "generated_artifact_manifest_mismatch" in representative["faults"]
    assert "objective_gradient_not_finite_nonzero" in representative["faults"]


def test_missing_parity_receipt_blocks_dependent_objective(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    (tmp_path / "reports/accelerator_replay/parity.json").unlink()

    parity, representative = gate.audit(config, root=tmp_path)

    assert parity["trigger_state"] == "RED"
    assert representative["trigger_state"] == "RED"
    assert "online_attention_predecessor_not_bound" in representative["faults"]
