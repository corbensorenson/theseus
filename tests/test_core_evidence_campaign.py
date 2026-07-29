from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import core_evidence_campaign as campaign  # noqa: E402
import core_evidence_worker as worker  # noqa: E402


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


def test_evaluator_must_require_a_real_independently_verified_patch() -> None:
    config = load_config()
    config["evaluator_contract"]["completion_predicate"]["patch_required"] = False
    config["evaluator_contract"]["candidate_output_schema"]["candidate_emitted_integrity_flags_trusted"] = True

    report = campaign.build_preregistration(
        config,
        ROOT / "configs" / "core_evidence_campaign.json",
    )

    assert report["trigger_state"] == "RED"
    failed = {row["name"] for row in report["hard_gaps"]}
    assert "candidate_output_schema_frozen" in failed
    assert "completion_requires_real_verified_patch" in failed


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


def test_e1_packet_joins_allowed_blocked_revoked_and_exact_rollback() -> None:
    config = load_config()
    report = campaign.build_e1_packet(
        config,
        ROOT / "configs" / "core_evidence_campaign.json",
        source_commit="unit-test-source",
        checkout_root=ROOT,
        gate_results={
            "registry": {"trigger_state": "GREEN", "returncode": 0},
            "roadmap": {"trigger_state": "YELLOW", "returncode": 0},
        },
        clean_checkout=True,
    )

    assert report["trigger_state"] == "GREEN"
    assert report["disposition"] == "REPLAYABLE_REFERENCE_BACKED"
    assert report["hard_gaps"] == []
    assert report["allowed_effect_trace"]["ready"] is True
    assert report["allowed_effect_trace"]["rollback"]["complete"] is True
    assert report["allowed_effect_trace"]["rollback"]["before_identity"] == report["allowed_effect_trace"]["rollback"]["final_identity"]
    assert report["blocked_effect_trace"]["ready"] is False
    assert report["revoked_effect_trace"]["ready"] is False
    assert report["independent_effect_audit"]["valid"] is True
    assert report["counters"]["exact_rollback_count"] == 1
    assert report["counters"]["D2_cases_consumed"] == 0
    assert report["counters"]["external_inference_calls"] == 0
    assert report["counters"]["learned_generation_credit"] == 0


def test_e1_replay_fails_if_e0_config_identity_changes(tmp_path: Path) -> None:
    config = load_config()
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps({**config, "question": "changed"}), encoding="utf-8")
    report = campaign.build_e1_packet(
        config,
        changed,
        source_commit="unit-test-source",
        checkout_root=ROOT,
        gate_results={
            "registry": {"trigger_state": "GREEN", "returncode": 0},
            "roadmap": {"trigger_state": "YELLOW", "returncode": 0},
        },
        clean_checkout=True,
    )

    assert report["trigger_state"] == "RED"
    assert report["disposition"] == "REPLAY_FAILED"
    assert "E0_config_identity" in {row["name"] for row in report["hard_gaps"]}


def test_e1_replay_fails_if_source_gates_are_red() -> None:
    report = campaign.build_e1_packet(
        load_config(),
        ROOT / "configs" / "core_evidence_campaign.json",
        source_commit="unit-test-source",
        checkout_root=ROOT,
        gate_results={
            "registry": {"trigger_state": "RED", "returncode": 2},
            "roadmap": {"trigger_state": "RED", "returncode": 2},
        },
        clean_checkout=True,
    )

    assert report["trigger_state"] == "RED"
    assert report["disposition"] == "REPLAY_FAILED"
    failed = {row["name"] for row in report["hard_gaps"]}
    assert "registry_gate_green" in failed
    assert "roadmap_gate_result_bound" not in failed
    assert any(row["artifact_id"] == "roadmap_implementation_gate" for row in report["artifact_gaps"])


def test_e1_replay_fails_if_roadmap_gate_result_is_missing() -> None:
    report = campaign.build_e1_packet(
        load_config(),
        ROOT / "configs" / "core_evidence_campaign.json",
        source_commit="unit-test-source",
        checkout_root=ROOT,
        gate_results={
            "registry": {"trigger_state": "GREEN", "returncode": 0},
            "roadmap": {"trigger_state": None, "returncode": 1},
        },
        clean_checkout=True,
    )

    assert report["trigger_state"] == "RED"
    assert "roadmap_gate_result_bound" in {row["name"] for row in report["hard_gaps"]}


def test_e1_local_evidence_capsule_is_complete_and_digest_only(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    registry = json.loads((ROOT / "configs" / "project_manifest_registry.json").read_text(encoding="utf-8"))
    for contract in registry["route_evidence_contracts"]:
        for requirement in contract.get("requirements", []):
            for relative_path in requirement.get("source_paths", []):
                source = ROOT / relative_path
                destination = checkout / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

    capsule = campaign.materialize_e1_evidence_capsule(ROOT, checkout)

    assert capsule["missing_required_paths"] == []
    assert capsule["source_timestamp_faults"] == []
    assert capsule["source_timestamp_overlays"]
    assert capsule["entry_count"] >= 20
    assert capsule["total_bytes"] > 0
    assert len(capsule["capsule_manifest_sha256"]) == 64
    for entry in capsule["entries"]:
        assert set(entry) == {
            "path",
            "bytes",
            "sha256",
            "raw_content_embedded_in_public_packet",
            "sensitivity",
        }
        assert entry["raw_content_embedded_in_public_packet"] is False
        assert (checkout / entry["path"]).is_file()
        assert campaign.sha256_bytes((checkout / entry["path"]).read_bytes()) == entry["sha256"]
    assert all(row["content_changed"] is False for row in capsule["source_timestamp_overlays"])


def test_local_worker_is_target_blind_and_emits_no_capability_credit(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    (snapshot / "scripts").mkdir(parents=True)
    (snapshot / "scripts" / "resource_policy.py").write_text(
        "def resource_policy():\n    return 'bounded'\n",
        encoding="utf-8",
    )
    visible = {
        "natural_request": "Tighten the resource policy and verify it.",
        "parent_source_commit": "a" * 40,
        "allowed_runtime_context": ["repository_parent_snapshot"],
        "authority_grant": "temporary_effect_with_exact_rollback",
    }

    result = worker.run_worker(visible, snapshot)

    assert result["patch_unified_diff"] == ""
    assert result["learned_generation_credit"] == 0
    assert result["external_inference_calls"] == 0
    assert result["teacher_calls"] == 0
    assert result["D2_cases_consumed"] == 0
    assert result["public_calibration_cases_consumed"] == 0
    assert not campaign.FORBIDDEN_VISIBLE_FIELDS.intersection(result)
    assert result["proposed_paths"] == ["scripts/resource_policy.py"]


def test_local_worker_rejects_extra_hidden_input_fields(tmp_path: Path) -> None:
    visible = {
        "natural_request": "Do the task.",
        "parent_source_commit": "a" * 40,
        "allowed_runtime_context": ["repository_parent_snapshot"],
        "authority_grant": "read_only",
        "target_commit": "b" * 40,
    }

    try:
        worker.run_worker(visible, tmp_path)
    except ValueError as exc:
        assert "input fields must be exactly" in str(exc)
    else:
        raise AssertionError("worker accepted a hidden target field")


def test_e2_stops_at_frozen_competence_floor_without_opening_heldout() -> None:
    report = campaign.run_e2_comparison(
        load_config(),
        ROOT / "configs" / "core_evidence_campaign.json",
    )

    assert report["trigger_state"] == "GREEN"
    assert report["terminal_disposition"] == "INCONCLUSIVE_WORKER_INADEQUATE"
    assert report["hard_gaps"] == []
    assert report["competence_floor"]["attempted"] == 3
    assert report["competence_floor"]["useful"] == 0
    assert report["competence_floor"]["passed"] is False
    assert report["heldout"]["opened"] == 0
    assert report["counters"]["E2_heldout_tasks_opened"] == 0
    assert report["counters"]["D2_cases_consumed"] == 0
    assert report["counters"]["external_inference_calls"] == 0
    assert len(report["route_summaries"]) == 5
    assert all(row["attempted"] == 3 for row in report["route_summaries"])
    assert all(
        task["candidate_seal"]["target_opened_before_seal"] is False
        for task in report["task_results"]
    )
    assert all(
        task["independent_evaluation"]["useful_completed_task"] is False
        for task in report["task_results"]
    )
