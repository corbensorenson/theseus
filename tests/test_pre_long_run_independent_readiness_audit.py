from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pre_long_run_independent_readiness_audit as audit
import moecot_language_arm_training as training


def test_independent_audit_config_covers_all_required_boundaries() -> None:
    config = json.loads(
        (
            ROOT / "configs/pre_long_run_independent_readiness_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert set(config["required_audits"]) == {
        "evidence_integrity",
        "lineage_custody",
        "replay_integrity",
        "resource_integrity",
        "evaluation_nonconsumption",
        "negative_evidence_scope",
        "claim_boundary",
        "execution_hold",
    }
    assert config["expected"]["optimizer_steps"] == 11416
    assert config["expected"]["prospective_anchor_step"] == 9048


def test_current_independent_audit_is_green_without_authorizing_training() -> None:
    report = audit.execute(
        ROOT / "configs/pre_long_run_independent_readiness_audit.json",
        publish_report=False,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["failed_audits"] == []
    assert report["missing_audits"] == []
    assert all(row["passed"] for row in report["audits"].values())
    assert report["audits"]["lineage_custody"]["lineage"]["manifest_count"] == 37
    assert (
        report["audits"]["lineage_custody"]["lineage"][
            "pre_anchor_full_chain_available"
        ]
        is False
    )
    assert (
        report["audits"]["evaluation_nonconsumption"][
            "matching_consumption_rows"
        ]
        == []
    )
    assert report["audits"]["resource_integrity"]["fixed_available_memory_floor_mib"] == 0
    assert report["long_training_authorized"] is False


def test_exact_step_11416_plan_migration_is_accepted_without_state_reset() -> None:
    config_path = ROOT / "configs/moecot_language_arm_training.json"
    config = training.bind_scale_preregistration(
        training.read_json(config_path)
    )
    plan = training.build_plan(config, config_path=config_path)
    receipt = training.read_json(
        ROOT
        / "checkpoints/moecot_mlx_57m_active_preregistered_v1"
        / "shared_trunk/training_receipt.json"
    )
    target = plan["targets"][training.SHARED_TRUNK_ID]
    migration = training.accepted_plan_identity_migration(
        receipt, plan, target
    )
    assert plan["plan_sha256"] == (
        "5d5b77e5d47814e9c3e8a8842b6f20966668d8d9f483f3584561fc550f196b1d"
    )
    assert migration is not None
    assert migration["migration_id"] == (
        "shared_trunk_step11416_finite_candidate_and_guard_closure_v1"
    )
    assert migration["legacy_optimizer_steps"] == 11416
    assert migration["legacy_optimizer_positions"] == 87441996
    assert migration["reset_data_cursor_phase"] is None
    assert migration["reset_data_cursor_seed"] is None


def test_consumption_matcher_detects_current_identity(tmp_path: Path) -> None:
    freeze = {
        "candidate_id": "candidate",
        "candidate_packet_sha256": "packet",
        "case_contract_sha256": "cases",
    }
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps({"identity": {"candidate_id": "candidate"}}) + "\n",
        encoding="utf-8",
    )
    matches = audit.matching_consumption_rows(registry, freeze, "freeze")
    assert len(matches) == 1
    assert matches[0]["line"] == 1


def test_lineage_recomputation_rejects_artifact_tampering(
    tmp_path: Path,
) -> None:
    lineage = tmp_path / "lineage"
    segment = lineage / "step-00009048_to_00009112"
    segment.mkdir(parents=True)
    artifact = segment / "receipt.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = {
        "before_identity": {"optimizer_steps": 9048},
        "after_identity": {
            "optimizer_steps": 9112,
            "optimizer_positions": 1,
            "checkpoint_sha256": "model",
            "optimizer_state_sha256": "optimizer",
            "mlx_rng_state_sha256": "rng",
        },
        "artifacts": {
            "receipt": {
                "path": str(artifact),
                "sha256": "wrong",
            }
        },
    }
    (segment / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    result = audit.recompute_lineage(
        lineage,
        expected_count=1,
        anchor_step=9048,
        terminal={
            "optimizer_steps": 9112,
            "optimizer_positions": 1,
            "checkpoint_sha256": "model",
            "optimizer_state_sha256": "optimizer",
            "mlx_rng_state_sha256": "rng",
        },
    )
    assert result["passed"] is False
    assert any(fault.startswith("artifact_hash_mismatch") for fault in result["faults"])
