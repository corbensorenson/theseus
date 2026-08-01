from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_assistant_p2a_evaluator as evaluator  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p3_pool_is_sealed_before_any_candidate_generation() -> None:
    pool = json.loads(
        (ROOT / "configs" / "theseus_p3_task_pool.json").read_text(encoding="utf-8")
    )

    assert pool["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION"
    assert pool["instrument_freeze_commit"] == "d08bf94653ee3a5ca508a2457ca21d58a4010a98"
    assert pool["candidate_generation_opened"] is False
    assert pool["task_count"] == 10
    assert pool["green_evaluator_audits"] == 10
    assert pool["distinct_repositories"] == 10
    assert pool["faults"] == []
    assert pool["counters"]["candidate_model_calls"] == 0
    assert pool["counters"]["external_inference_calls"] == 0
    assert digest(ROOT / pool["source_registry"]) == pool["source_registry_sha256"]
    assert digest(ROOT / pool["instrument"]) == pool["instrument_sha256"]
    assert digest(ROOT / pool["instrument_audit"]) == pool["instrument_audit_sha256"]


def test_p3_tasks_are_distinct_blind_and_parent_fail_target_pass() -> None:
    pool = json.loads(
        (ROOT / "configs" / "theseus_p3_task_pool.json").read_text(encoding="utf-8")
    )
    repositories: set[str] = set()

    for index, row in enumerate(pool["tasks"], 1):
        task_path = ROOT / row["task"]
        evaluator_path = ROOT / row["evaluator"]
        audit_path = ROOT / row["evaluator_audit"]
        task = json.loads(task_path.read_text(encoding="utf-8"))
        evaluator_config = json.loads(evaluator_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

        assert row["campaign_index"] == index
        repositories.add(row["repository"])
        assert digest(task_path) == row["task_sha256"]
        assert digest(evaluator_path) == row["evaluator_sha256"]
        assert digest(audit_path) == row["evaluator_audit_sha256"]
        assert p2a.audit_task(task_path)["trigger_state"] == "GREEN"
        assert evaluator_config["blindness"]["candidate_generation_may_read_this_manifest"] is False
        assert evaluator_config["blindness"]["route_label_passed_to_scoring"] is False
        assert audit["trigger_state"] == "GREEN"
        assert audit["baseline_verification"]["passed"] is False
        assert audit["target_verification"]["passed"] is True
        screen = task["contamination_screen"]
        assert screen["public_benchmark"] is False
        assert screen["previous_theseus_surface"] is False
        assert screen["source_disjoint_from_p2a_p2b_p2c"] is True
        assert screen["later_patch_or_tests_candidate_visible"] is False
        assert screen["development_task_eligible_for_training"] is False
        assert screen["development_task_eligible_for_D1_or_D2"] is False

    assert len(repositories) == 10


def test_p3_live_evaluators_still_bind_parent_fail_target_pass() -> None:
    pool = json.loads(
        (ROOT / "configs" / "theseus_p3_task_pool.json").read_text(encoding="utf-8")
    )

    for row in pool["tasks"]:
        assert evaluator.audit_evaluator(ROOT / row["evaluator"])["trigger_state"] == "GREEN"
