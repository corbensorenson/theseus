from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pre_long_run_independent_readiness_audit as audit  # noqa: E402
import moecot_language_arm_training as training  # noqa: E402


def test_independent_audit_config_covers_all_required_boundaries() -> None:
    config = json.loads(
        (ROOT / "configs/pre_long_run_independent_readiness_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(config["required_audits"]) == {
        "evidence_integrity",
        "lineage_custody",
        "replay_integrity",
        "resource_integrity",
        "evaluation_nonconsumption",
        "evaluation_surface_freshness",
        "negative_evidence_scope",
        "claim_boundary",
        "execution_hold",
    }
    assert config["expected"]["optimizer_steps"] == 11416
    assert config["expected"]["prospective_anchor_step"] == 9048


def test_current_independent_audit_recomputes_fresh_surface_without_authorizing_it() -> (
    None
):
    report = audit.execute(
        ROOT / "configs/pre_long_run_independent_readiness_audit.json",
        publish_report=False,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["failed_audits"] == []
    assert report["missing_audits"] == []
    assert report["audits"]["evaluation_nonconsumption"]["passed"] is True
    assert report["audits"]["evaluation_surface_freshness"]["passed"] is True
    assert report["audits"]["lineage_custody"]["lineage"]["manifest_count"] == 37
    assert (
        report["audits"]["lineage_custody"]["lineage"][
            "pre_anchor_full_chain_available"
        ]
        is False
    )
    assert (
        report["audits"]["evaluation_nonconsumption"]["matching_consumption_rows"] == []
    )
    freshness = report["audits"]["evaluation_surface_freshness"]
    assert freshness["state"] == "VALID_FRESH_PRIVATE_SURFACE"
    recomputed = freshness["independent_recomputation"]
    assert recomputed["passed"] is True
    assert recomputed["current"]["contract_sha256"] == (
        "d48875c5acdb883ef8dfc251914109186c21ff9d07eaf6c06b9b00927ca1c676"
    )
    assert recomputed["current"]["consumption_registry_lines"] == []
    assert recomputed["exact_prompt_overlaps"] == []
    assert recomputed["normalized_prompt_overlaps"] == []
    assert recomputed["historical_packets"][0]["contract_sha256"] == (
        "d724363eca913129cb1701105b06c8f51ebc644d1a7485994bc1a50b54bdc792"
    )
    assert recomputed["historical_packets"][0]["consumption_registry_lines"] != []
    assert freshness["evaluation_authorized"] is False
    assert (
        report["audits"]["resource_integrity"]["fixed_available_memory_floor_mib"] == 0
    )
    assert report["long_training_authorized"] is False


def test_exact_step_11416_plan_migration_is_accepted_without_state_reset() -> None:
    config_path = ROOT / "configs/moecot_language_arm_training.json"
    config = training.bind_scale_preregistration(training.read_json(config_path))
    plan = training.build_plan(config, config_path=config_path)
    receipt = training.read_json(
        ROOT
        / "checkpoints/moecot_mlx_57m_active_preregistered_v1"
        / "shared_trunk/training_receipt.json"
    )
    target = plan["targets"][training.SHARED_TRUNK_ID]
    migration = training.accepted_plan_identity_migration(receipt, plan, target)
    assert plan["plan_sha256"] == (
        "46de0ea6a115a2c5ee4a065dbb9892d66bbff4f12fdc43b35e297490eafa32e1"
    )
    assert migration is not None
    assert migration["migration_id"] == (
        "shared_trunk_step11416_uncapped_d2_contract_rebind_v1"
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


def test_consumption_matcher_detects_equivalent_consumed_contract(
    tmp_path: Path,
) -> None:
    freeze = {
        "candidate_id": "new-candidate",
        "candidate_packet_sha256": "new-packet",
        "case_contract_sha256": "new-cases",
    }
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps({"identity": {"case_contract_sha256": "old-cases"}}) + "\n",
        encoding="utf-8",
    )

    matches = audit.matching_consumption_rows(
        registry,
        freeze,
        "new-freeze",
        ["old-cases"],
    )

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
    (segment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
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
