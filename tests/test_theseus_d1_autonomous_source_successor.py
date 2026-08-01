from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_d1_autonomous_source_successor as successor  # noqa: E402


CONFIG_PATH = ROOT / "configs/theseus_d1_autonomous_source_successor.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def survivor() -> dict:
    return {
        "policy": "project_theseus_p4v2r2_terminal_disposition_v1",
        "created_utc": "2026-08-01T18:00:00Z",
        "trigger_state": "GREEN",
        "scientific_status": "P4V2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "consumption": {"eligible_for_D1": True},
        "decision_rule": {
            "survivor_effect_rule_passed": True,
            "effect_decision_authorized": True,
        },
    }


def non_survivor() -> dict:
    return {
        "created_utc": "2026-08-01T18:00:00Z",
        "trigger_state": "GREEN",
        "scientific_status": "P4V2R2_ADEQUATE_NO_SURVIVOR",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
    }


def registry() -> dict:
    tasks = [
        {"campaign_index": index, "repository": f"owner/repo-{index}"}
        for index in range(1, 45)
    ]
    return {
        "policy": "project_theseus_d1_online_source_registry_v1",
        "state": "FIXED_BEFORE_ARCHIVE_FETCH_PARENT_TARGET_ORACLE_EVALUATOR_OR_CANDIDATE_EXECUTION",
        "task_count": 44,
        "tasks": tasks,
        "boundaries": {
            "archive_fetches": 0,
            "parent_target_oracle_or_evaluator_executions": 0,
            "candidate_or_control_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
        },
        "replacement_after_membership_freeze": False,
    }


def test_config_binds_exact_owners_without_user_token_or_cross_stage_authority() -> None:
    value = config()
    assert successor.validate_config(value) == []
    assert successor.audit_bindings(value)["passed"] is True
    authority = value["authority"]
    assert authority["user_or_operator_approval_required"] is False
    assert authority["wait_deadline_seconds"] is None
    assert authority["project_selected_quality_token_cap_allowed"] is False
    assert authority["archive_fetch_authorized"] is False
    assert authority["candidate_or_control_calls_authorized"] is False
    assert authority["book_support_promotion_authorized"] is False


def test_preterminal_p4_waits_without_calls_or_execution() -> None:
    report = successor.preflight(
        config(),
        config_path=CONFIG_PATH,
        disposition_override={},
        lease_exists_override=False,
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["activation_state"] == "WAITING_FOR_TERMINAL_P4V2R2"
    assert report["terminal"] is False
    assert report["execution_authorized"] is False
    assert report["network_calls"] == report["candidate_or_control_calls"] == 0


def test_non_survivor_closes_d1_without_network_or_user_gate() -> None:
    report = successor.preflight(
        config(),
        config_path=CONFIG_PATH,
        disposition_override=non_survivor(),
        lease_exists_override=False,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["activation_state"] == "CLOSED_P4V2R2_NON_SURVIVOR"
    assert report["terminal"] is True
    assert report["execution_authorized"] is False
    assert report["network_calls"] == 0


def test_exact_survivor_authorizes_metadata_only_after_complete_interval() -> None:
    report = successor.preflight(
        config(),
        config_path=CONFIG_PATH,
        disposition_override=survivor(),
        ledger_override={},
        now_override=datetime(2026, 8, 2, 12, tzinfo=timezone.utc),
        lease_exists_override=False,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["activation_state"] == "D1_METADATA_ACQUISITION_READY"
    assert report["execution_authorized"] is True
    assert report["next_action"] == "acquire_metadata_then_freeze_if_complete"
    assert report["archive_fetches"] == 0
    assert report["candidate_or_control_calls"] == 0


def test_same_complete_interval_is_not_reacquired_when_cohort_is_short() -> None:
    ledger = {
        "frame_end_utc": "2026-08-02T00:00:00Z",
        "rows": [],
    }
    report = successor.preflight(
        config(),
        config_path=CONFIG_PATH,
        disposition_override=survivor(),
        ledger_override=ledger,
        now_override=datetime(2026, 8, 2, 12, tzinfo=timezone.utc),
        lease_exists_override=False,
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["activation_state"] == "WAITING_FOR_NEW_COMPLETE_UTC_INTERVAL"
    assert report["execution_authorized"] is False


def test_frozen_registry_is_terminal_and_never_replaced() -> None:
    report = successor.preflight(
        config(),
        config_path=CONFIG_PATH,
        disposition_override=survivor(),
        registry_override=registry(),
        lease_exists_override=False,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["activation_state"] == "D1_SOURCE_REGISTRY_FROZEN"
    assert report["terminal"] is True
    assert report["execution_authorized"] is False
    assert report["source_registry_audit"]["passed"] is True
