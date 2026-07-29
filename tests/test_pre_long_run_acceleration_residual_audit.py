from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pre_long_run_acceleration_residual_audit as audit


def test_config_covers_every_cross_domain_residual() -> None:
    config = json.loads(
        (
            ROOT / "configs/pre_long_run_acceleration_residual_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert set(config["required_domains"]) == {
        "mlx_metal",
        "cpu_rust",
        "ane_accelerate_metal",
        "data_pipeline",
        "checkpoint_replay",
        "memory_disk",
        "thermal",
        "joined_wall",
        "architecture_candidates",
        "custody_hold",
    }
    assert config["selected_recipe"]["training_step_mode"] == "compiled"
    assert config["selected_recipe"]["self_attention_projection"] == "separate"
    assert config["selected_recipe"]["per_head_muon"] is False


def test_current_finite_residual_audit_is_green_and_keeps_hold() -> None:
    report = audit.execute(
        ROOT / "configs/pre_long_run_acceleration_residual_audit.json",
        publish_report=False,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["failed_domains"] == []
    assert report["missing_domains"] == []
    assert all(row["passed"] for row in report["domains"].values())
    assert report["checkpoint_custody"]["optimizer_steps"] == 11416
    assert report["checkpoint_custody"]["optimizer_positions"] == 87441996
    assert report["checkpoint_custody"]["capability_claim"] == "NOT_EVALUATED"
    assert report["checkpoint_custody"]["lineage"]["manifest_count"] == 37
    assert report["checkpoint_custody"]["lineage"]["pre_anchor_chain_available"] is False
    assert report["domains"]["memory_disk"]["fixed_available_memory_floor_mib"] == 0
    assert report["long_training_started_or_resumed"] is False
    assert (
        ROOT / "runtime/control/neural_seed_yield_after_segment"
    ).is_file()


def test_lineage_audit_rejects_terminal_identity_tampering(tmp_path: Path) -> None:
    lineage_root = tmp_path / "lineage"
    segment = lineage_root / "step-00009048_to_00009112"
    segment.mkdir(parents=True)
    artifact = segment / "after.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = {
        "before_identity": {"optimizer_steps": 9048},
        "after_identity": {
            "optimizer_steps": 9112,
            "optimizer_positions": 1,
            "checkpoint_sha256": "wrong",
            "optimizer_state_sha256": "wrong",
            "mlx_rng_state_sha256": "wrong",
        },
        "artifacts": {
            "after": {
                "path": str(artifact),
                "sha256": audit.sha256_file(artifact),
            }
        },
    }
    (segment / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    result = audit.validate_lineage(
        lineage_root,
        expected_count=1,
        first_step=9048,
        receipt={
            "optimizer_steps": 9112,
            "optimizer_positions": 2,
            "checkpoint_sha256": "expected",
            "optimizer_state_sha256": "expected",
            "mlx_rng_state_sha256": "expected",
        },
    )
    assert result["passed"] is False
    assert "lineage_terminal_mismatch:optimizer_positions" in result["faults"]
    assert "lineage_terminal_mismatch:checkpoint_sha256" in result["faults"]
