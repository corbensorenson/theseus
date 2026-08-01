from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4v2r2_campaign as campaign  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p4v2r2_campaign_is_exact_bound_and_completion_based() -> None:
    report = campaign.audit_campaign()

    assert report["trigger_state"] == "RED"
    assert "partial_unsealed_runtime_receipts:p4v2r2_01_flask_5917" in report["faults"]
    assert report["complete_tasks"] + report["pending_tasks"] == 10
    assert report["model_calls_retained"] == report["complete_tasks"] * 6
    assert (
        report["parser_complete_calls"] + report["model_eos_calls"]
        == report["model_calls_retained"]
    )
    assert report["safety_ceiling_hits"] == 0
    assert report["project_selected_quality_token_cap"] is None
    assert report["pool_sha256"] == campaign.POOL_SHA256
    assert report["pool_seal_commit"] == campaign.POOL_SEAL_COMMIT
    assert report["instrument_sha256"] == campaign.INSTRUMENT_SHA256
    assert report["hosted_reference"]["calls"] == 0
    repair = report["preconsumption_bootstrap_repair"]
    assert repair["passed"] is True
    assert repair["candidate_or_control_calls"] == 0
    assert repair["tasks_consumed"] == 0
    assert repair["surface_reuse_authorized"] is True


def test_p4v2r2_result_paths_are_campaign_specific_and_unopened() -> None:
    pool = campaign.p2a.read_json(campaign.POOL)
    paths = [campaign.result_paths(row) for row in pool["tasks"]]

    assert len({row["run"] for row in paths}) == 10
    assert len({row["evaluation"] for row in paths}) == 10
    assert all(
        row["run"].name.startswith("theseus_p4v2r2_attempt2_") for row in paths
    )
    audit = campaign.audit_campaign()
    assert audit["model_calls_retained"] in {
        0,
        6,
        12,
        18,
        24,
        30,
        36,
        42,
        48,
        54,
        60,
    }
    assert audit["runtime_attempt_namespace"] == "p4v2r2_attempt1"


def test_p4v2r2_pool_binds_zero_calls_and_all_mechanics_floors() -> None:
    pool = campaign.p2a.read_json(campaign.POOL)

    assert pool["green_evaluator_audits"] == 10
    assert pool["v2r2_oracle_replays_green"] == 10
    assert pool["dependency_corruptions_rejected"] == 10
    assert pool["generation_budget"]["project_selected_quality_token_cap"] is None
    assert pool["counters"]["local_model_calls"] == 0
    assert pool["counters"]["hosted_model_calls"] == 0


def test_attempt2_interruption_is_retained_as_scoped_implementation_failure() -> None:
    report = json.loads(
        (ROOT / "reports" / "theseus_p4v2r2_attempt2_interruption.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["scientific_status"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert report["denominators"]["model_calls_observed"] == 2
    assert report["denominators"]["candidate_outputs_released"] == 0
    assert report["denominators"]["consumed_tasks"] == 1
    assert report["denominators"]["candidate_unseen_tasks"] == 9
    assert report["denominators"]["project_selected_quality_token_cap"] is None
    assert report["disposition"]["same_denominator_resume_authorized"] is False
    assert report["disposition"]["consumed_task_replay_authorized"] is False
    for receipt in report["runtime_receipts"]:
        assert sha256_file(ROOT / receipt["path"]) == receipt["sha256"]
    assert not (ROOT / "runtime" / "control" / "theseus_p4v2r2_campaign_lease.json").exists()
