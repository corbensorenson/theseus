from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import core_evidence_campaign as campaign  # noqa: E402


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "core_evidence_campaign.json").read_text(encoding="utf-8"))


def test_e0_preregistration_is_green_and_prospective() -> None:
    report = campaign.build_preregistration(
        load_config(),
        ROOT / "configs" / "core_evidence_campaign.json",
    )

    assert report["trigger_state"] == "GREEN"
    assert report["preregistration_state"] == "FROZEN_PROSPECTIVE"
    assert report["hard_gaps"] == []
    assert report["sealed_evaluator_summary"]["targets_opened_to_worker"] == 0
    assert report["sealed_evaluator_summary"]["D2_cases_consumed"] == 0
    assert report["sealed_evaluator_summary"]["public_calibration_cases_consumed"] == 0
    assert report["sealed_evaluator_summary"]["external_inference_calls"] == 0
    assert report["sealed_evaluator_summary"]["teacher_calls"] == 0


def test_public_task_projection_contains_no_hidden_fields() -> None:
    report = campaign.build_preregistration(
        load_config(),
        ROOT / "configs" / "core_evidence_campaign.json",
    )
    public_tasks = report["public_packet"]["tasks"]

    assert public_tasks
    for task in public_tasks:
        assert not campaign.FORBIDDEN_VISIBLE_FIELDS.intersection(task)
        assert set(task) == {
            "opaque_task_id",
            "partition",
            "denominator",
            "family",
            "natural_request",
            "parent_source_commit",
            "allowed_runtime_context",
            "authority_grant",
            "effect_class",
        }
        assert task["parent_source_commit"]
        assert task["natural_request"]


def test_e2_and_e3_targets_are_source_disjoint() -> None:
    config = load_config()

    assert campaign.disjoint_targets(config["tasks"], "D1_E2", "D1_E3")
    assert len(campaign.target_set(config["tasks"], "D1_E2")) >= 4
    assert len(campaign.target_set(config["tasks"], "D1_E3")) >= 4
    assert campaign.repeated_family_present(config["tasks"], "D1_E3")


def test_hidden_field_leak_invalidates_preregistration() -> None:
    config = load_config()
    config["information_flow"]["candidate_visible_fields"].append("target_commit")

    report = campaign.build_preregistration(
        config,
        ROOT / "configs" / "core_evidence_campaign.json",
    )

    assert report["trigger_state"] == "RED"
    failed = {row["name"] for row in report["hard_gaps"]}
    assert "visible_fields_exact" in failed
    assert "visible_hidden_disjoint" in failed
    assert "no_forbidden_visible_field" in failed


def test_worker_credit_or_D2_access_invalidates_preregistration() -> None:
    config = load_config()
    config["identities"]["worker_learned_credit"] = True
    config["boundaries"]["D2_consumption"] = "allowed"

    report = campaign.build_preregistration(
        config,
        ROOT / "configs" / "core_evidence_campaign.json",
    )

    assert report["trigger_state"] == "RED"
    failed = {row["name"] for row in report["hard_gaps"]}
    assert "worker_has_no_learned_credit" in failed
    assert "D2_forbidden" in failed


def test_route_or_floor_mutation_invalidates_preregistration() -> None:
    config = load_config()
    config["matched_routes"] = config["matched_routes"][:-1]
    config["decision_rules"]["competence_floor"]["minimum_useful_rate"] = 0

    report = campaign.build_preregistration(
        config,
        ROOT / "configs" / "core_evidence_campaign.json",
    )

    assert report["trigger_state"] == "RED"
    failed = {row["name"] for row in report["hard_gaps"]}
    assert "matched_routes_exact" in failed
    assert "competence_floor_frozen" in failed


def test_history_task_subject_or_partition_mutation_invalidates_preregistration() -> None:
    config = load_config()
    config["tasks"][0]["natural_request"] = "Authored success path"
    for task in config["tasks"]:
        if task["partition"] == "development":
            task["partition"] = "calibration"

    report = campaign.build_preregistration(
        config,
        ROOT / "configs" / "core_evidence_campaign.json",
    )

    assert report["trigger_state"] == "RED"
    failed = {row["name"] for row in report["hard_gaps"]}
    assert "commit_subjects_match_natural_requests" in failed
    assert "partitions_complete" in failed
    assert "development_floor_has_tasks" in failed


def test_report_digest_is_stable_across_timestamp_changes() -> None:
    config = load_config()
    first = campaign.build_preregistration(
        config,
        ROOT / "configs" / "core_evidence_campaign.json",
    )
    second = campaign.build_preregistration(
        config,
        ROOT / "configs" / "core_evidence_campaign.json",
    )

    assert first["preregistration_sha256"] == second["preregistration_sha256"]
    assert first["report_payload_sha256"] == second["report_payload_sha256"]
