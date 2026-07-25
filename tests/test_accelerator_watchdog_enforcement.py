from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dynamic_byte_patch_codec as dynamic_patch  # noqa: E402
import generation_architecture_contracts as generation  # noqa: E402
import generation_mode_gate  # noqa: E402
import kerc_adequacy_canary  # noqa: E402
import kerc_residual_allocator_qualification  # noqa: E402
import kerc_structured_drafting as kerc_drafting  # noqa: E402
import kerc_training_memory_preflight  # noqa: E402
import neural_seed_50m_scale_preregistration as scale_preregistration  # noqa: E402
import policy_objective_contracts as policy_objectives  # noqa: E402
import policy_optimization_gate  # noqa: E402
import rdc_relation_learning as relation_learning  # noqa: E402
import relational_dimension_compiler as rdc  # noqa: E402
import soap_full_shape_resource_preflight as soap_preflight  # noqa: E402
import target_authoritative_drafting as target_drafting  # noqa: E402


def _forbid_subprocess(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
    raise AssertionError("accelerator subprocess launched without host watchdog")


def test_accelerator_canaries_fail_closed_before_native_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("THESEUS_GUARDED_ACCELERATOR_CHILD", raising=False)
    for module in (
        dynamic_patch,
        generation,
        kerc_drafting,
        policy_objectives,
        relation_learning,
        soap_preflight,
        target_drafting,
    ):
        monkeypatch.setattr(module.subprocess, "run", _forbid_subprocess)

    receipts = (
        dynamic_patch.mlx_dynamic_patch_adequacy_canary(),
        generation.mlx_mtp_canary(),
        generation.mlx_mtp_adequacy_canary(),
        kerc_drafting.mlx_structured_drafting_canary(),
        policy_objectives.mlx_parity_probe(policy_objectives.reference_pair()),
        relation_learning.mlx_relation_learning_canary(),
        target_drafting.mlx_drafting_adequacy_canary(),
        target_drafting.mlx_target_kv_authority_canary(),
    )
    assert all(row["available"] is False for row in receipts)
    assert all(row["fault"] == "ACCELERATOR_WATCHDOG_REQUIRED" for row in receipts)
    with pytest.raises(soap_preflight.SoapPreflightFault, match="watchdog_required"):
        soap_preflight.mlx_eigh_support_and_timing({})


def test_sparse_mlx_lowering_denies_ambient_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("THESEUS_GUARDED_ACCELERATOR_CHILD", raising=False)
    with pytest.raises(rdc.RDCFault, match="accelerator_watchdog_required"):
        rdc.factorized_relation_pool_mlx(
            {}, entity_features=[], role_factors=[], schema_factors=[]
        )


def test_kerc_memory_station_refuses_before_native_execution_or_report_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("THESEUS_GUARDED_ACCELERATOR_CHILD", raising=False)
    output = tmp_path / "forbidden-memory-station.json"

    def forbid_execute(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("unguarded memory station executed")

    def forbid_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unguarded memory station wrote a report")

    monkeypatch.setattr(
        kerc_training_memory_preflight, "execute_station", forbid_execute
    )
    monkeypatch.setattr(
        kerc_training_memory_preflight.arm, "write_json_atomic", forbid_write
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kerc_training_memory_preflight.py",
            "--station",
            "full_kerc_objective",
            "--out",
            str(output),
        ],
    )
    with pytest.raises(
        kerc_training_memory_preflight.host_resource_safety.HostResourceSafetyFault,
        match="direct MLX station execution is denied",
    ):
        kerc_training_memory_preflight.main()
    assert not output.exists()


def test_aggregate_accelerator_entrypoints_refuse_before_report_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("THESEUS_GUARDED_ACCELERATOR_CHILD", raising=False)

    def forbid_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unguarded accelerator entrypoint mutated a report")

    for module, argv in (
        (generation_mode_gate, ["generation_mode_gate.py"]),
        (policy_optimization_gate, ["policy_optimization_gate.py"]),
        (
            scale_preregistration,
            ["neural_seed_50m_scale_preregistration.py", "--execute-canaries"],
        ),
    ):
        monkeypatch.setattr(module, "write_json", forbid_write)
        monkeypatch.setattr(sys, "argv", argv)
        assert module.main() == 2

    for module, argv in (
        (
            kerc_adequacy_canary,
            ["kerc_adequacy_canary.py", "--execute"],
        ),
        (
            kerc_residual_allocator_qualification,
            ["kerc_residual_allocator_qualification.py"],
        ),
    ):
        monkeypatch.setattr(module, "write_json_atomic", forbid_write)
        monkeypatch.setattr(sys, "argv", argv)
        assert module.main() == 2
