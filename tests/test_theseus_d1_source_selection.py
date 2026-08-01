from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_d1_source_selection as selection  # noqa: E402


CONFIG = ROOT / "configs" / "theseus_d1_source_selection.json"


def survivor() -> dict[str, object]:
    return {
        "created_utc": "2026-08-01T12:00:00Z",
        "trigger_state": "GREEN",
        "scientific_status": "P4V2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "consumption": {"eligible_for_D1": True},
        "decision_rule": {
            "survivor_effect_rule_passed": True,
            "effect_decision_authorized": True,
        },
    }


def candidate(index: int) -> dict[str, object]:
    digit = f"{index + 1000:040x}"[-40:]
    target = f"{index + 2000:040x}"[-40:]
    merge = f"{index + 3000:040x}"[-40:]
    repository = f"d1-owner-{index}/d1-repo-{index}"
    return {
        "repository": repository,
        "repository_url": f"https://github.com/{repository}",
        "license_spdx": "MIT",
        "primary_language": "Python",
        "pull_request": index + 1,
        "pull_request_url": f"https://github.com/{repository}/pull/{index + 1}",
        "pull_request_title": f"Repair behavior {index}",
        "merged_utc": "2026-08-01T13:00:00Z",
        "parent_revision": digit,
        "target_revision": target,
        "merge_revision": merge,
        "changed_paths": [f"package_{index}/core.py", f"tests/test_core_{index}.py"],
        "changed_files": [
            {
                "filename": f"package_{index}/core.py",
                "status": "modified",
                "previous_filename": "",
                "additions": 2,
                "deletions": 1,
                "changes": 3,
            },
            {
                "filename": f"tests/test_core_{index}.py",
                "status": "modified",
                "previous_filename": "",
                "additions": 3,
                "deletions": 1,
                "changes": 4,
            },
        ],
        "metadata_retrieved_utc": "2026-08-02T00:00:00Z",
        "public_benchmark_source": False,
        "training_or_distillation_overlap": False,
        "metadata_only_selection": True,
    }


def ledger(count: int = 44) -> dict[str, object]:
    disposition = survivor()
    return {
        "policy": "project_theseus_d1_online_metadata_frame_v1",
        "state": "COMPLETE_QUERY_PARTITIONS_SEALED",
        "activation_disposition_sha256": selection.d1.stable_hash(disposition),
        "acquisition_opened_utc": "2026-08-01T12:00:01Z",
        "frame_start_utc": "2026-07-01T00:00:00Z",
        "frame_end_utc": "2026-08-01T00:00:00Z",
        "query_partitions": [
            {
                "id": "python-2026-07",
                "complete": True,
                "raw_response_sha256": "a" * 64,
            }
        ],
        "boundaries": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "candidate_or_control_calls": 0,
            "parent_target_oracle_or_evaluator_executions": 0,
            "archive_fetches": 0,
        },
        "rows": [candidate(index) for index in range(count)],
    }


def test_preactivation_does_not_even_ingest_metadata() -> None:
    report = selection.build_report(
        CONFIG,
        disposition_override={},
        ledger_override=ledger(),
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["source_metadata_ledger_ingested"] is False
    assert report["source_acquisition_authorized"] is False
    assert report["registry_ready"] is False


def test_survivor_freezes_metadata_only_power_derived_cohort() -> None:
    report = selection.build_report(
        CONFIG,
        disposition_override=survivor(),
        ledger_override=ledger(),
    )
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["registry_ready"] is True
    registry = report["registry_candidate"]
    assert registry["task_count"] == 44
    assert registry["distinct_repository_count"] == 44
    assert registry["boundaries"]["archive_fetches"] == 0
    assert registry["boundaries"]["candidate_or_control_calls"] == 0
    assert registry["replacement_after_membership_freeze"] is False
    assert registry["source_disjoint_from"]["training"] == (
        "all_selected_D1_tasks_permanently_excluded"
    )
    assert report["candidate_or_control_calls_authorized"] is False
    assert report["archive_fetch_authorized"] is False


def test_selection_is_deterministic_and_input_order_independent() -> None:
    first = ledger(50)
    second = dict(first)
    second["rows"] = list(reversed(first["rows"]))
    left = selection.build_report(
        CONFIG, disposition_override=survivor(), ledger_override=first
    )
    right = selection.build_report(
        CONFIG, disposition_override=survivor(), ledger_override=second
    )
    left_ids = [row["selection_digest"] for row in left["registry_candidate"]["tasks"]]
    right_ids = [row["selection_digest"] for row in right["registry_candidate"]["tasks"]]
    assert left_ids == right_ids


def test_underfilled_frame_pauses_without_smaller_substitute() -> None:
    report = selection.build_report(
        CONFIG,
        disposition_override=survivor(),
        ledger_override=ledger(43),
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["registry_ready"] is False
    assert "eligible_distinct_repository_count_below_design:43/44" in report["faults"]


def test_candidate_or_evaluator_outcome_in_metadata_invalidates_frame() -> None:
    value = ledger()
    value["rows"][0]["candidate_output"] = "forbidden"
    report = selection.build_report(
        CONFIG,
        disposition_override=survivor(),
        ledger_override=value,
    )
    assert report["registry_ready"] is False
    assert "candidate_control_or_evaluator_outcome_in_metadata_frame" in report["faults"]


def test_answer_identifying_metadata_invalidates_frame() -> None:
    value = ledger()
    value["rows"][0]["source_task_id"] = "leaky-target-identity"
    report = selection.build_report(
        CONFIG,
        disposition_override=survivor(),
        ledger_override=value,
    )
    assert report["registry_ready"] is False
    assert "candidate_control_or_evaluator_outcome_in_metadata_frame" in report["faults"]


def test_pre_snapshot_task_is_independently_excluded() -> None:
    value = ledger(45)
    value["rows"][0]["merged_utc"] = "2026-07-30T04:53:03Z"
    report = selection.build_report(
        CONFIG,
        disposition_override=survivor(),
        ledger_override=value,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["selection"]["exclusion_reason_counts"][
        "task_not_merged_strictly_after_frozen_model_snapshot_observation"
    ] == 1
    assert report["temporal_contamination_guard"][
        "candidate_emitted_contamination_flags_trusted"
    ] is False


def test_temporal_guard_is_bound_to_the_exact_p4v2r2_tmax_identity() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    audit, faults = selection.audit_temporal_guard(config)
    assert faults == []
    assert audit["model_identity"] == {
        "repo_id": "mlx-community/Tmax-9B-MLX-8bit",
        "revision": "33812d6cf04f88856f25eb828de4f3144a194560",
        "snapshot_manifest_sha256": (
            "a399b12d768ebf45ff5ce1f873fefc5525c980d953379394f4d5deb3201cb3dc"
        ),
    }
    assert audit["model_snapshot_observed_utc"] == "2026-07-30T12:35:41.833397Z"


def test_metadata_acquired_before_survivor_is_rejected() -> None:
    value = ledger()
    value["acquisition_opened_utc"] = "2026-08-01T11:59:59Z"
    value["rows"][0]["metadata_retrieved_utc"] = "2026-08-01T11:59:59Z"
    report = selection.build_report(
        CONFIG,
        disposition_override=survivor(),
        ledger_override=value,
    )
    assert report["registry_ready"] is False
    assert "metadata_acquisition_not_proven_post_survivor" in report["faults"]
    assert "metadata_row_not_retrieved_post_survivor:0" in report["faults"]


def test_prior_consumed_repository_is_excluded() -> None:
    value = ledger(45)
    value["rows"][0]["repository"] = "jd/tenacity"
    value["rows"][0]["repository_url"] = "https://github.com/jd/tenacity"
    value["rows"][0]["pull_request_url"] = "https://github.com/jd/tenacity/pull/1"
    report = selection.build_report(
        CONFIG,
        disposition_override=survivor(),
        ledger_override=value,
    )
    assert report["trigger_state"] == "GREEN"
    repositories = {
        row["repository"].lower() for row in report["registry_candidate"]["tasks"]
    }
    assert "jd/tenacity" not in repositories
    assert report["selection"]["exclusion_reason_counts"][
        "prior_source_repository_overlap"
    ] == 1


def test_public_benchmark_repository_is_independently_excluded() -> None:
    value = ledger(45)
    value["rows"][0]["repository"] = "SWE-bench/SWE-bench"
    value["rows"][0]["repository_url"] = "https://github.com/SWE-bench/SWE-bench"
    value["rows"][0]["pull_request_url"] = (
        "https://github.com/SWE-bench/SWE-bench/pull/1"
    )
    report = selection.build_report(
        CONFIG,
        disposition_override=survivor(),
        ledger_override=value,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["selection"]["exclusion_reason_counts"][
        "public_benchmark_repository_overlap"
    ] == 1
    assert report["temporal_contamination_guard"][
        "public_benchmark_repository_count"
    ] > 0


def test_selection_contract_has_no_user_gate_or_postfreeze_replacement() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["selection"]["user_or_operator_approval_required"] is False
    assert config["selection"]["replacement_after_membership_freeze"] is False
    assert config["authority"]["candidate_calls_after_registry_write"] is False
    assert config["discovery_frame"]["external_inference_calls"] == 0


def test_D1_selector_inherits_only_the_mechanics_qualified_Python_scope() -> None:
    instrument_config = json.loads(
        (ROOT / "configs/theseus_d1_fresh_qualification_instrument.json").read_text(
            encoding="utf-8"
        )
    )
    languages = selection.normalized_languages(
        instrument_config["source_surface"]["programming_language_scope"]
    )
    assert languages == {"python"}
