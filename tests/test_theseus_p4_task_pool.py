from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4_cognitive_compilation_evaluator as p4_evaluator  # noqa: E402
import theseus_p4_task_pool as task_pool  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p4_pool_is_sealed_before_any_candidate_or_control_call() -> None:
    pool = read(ROOT / "configs" / "theseus_p4_task_pool.json")

    assert pool["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION"
    assert pool["instrument_freeze_commit"] == "4ef352303a4d4c93288d1db3da659a874663c6d3"
    assert pool["candidate_generation_opened"] is False
    assert pool["task_count"] == 10
    assert pool["green_evaluator_audits"] == 10
    assert pool["distinct_repositories"] == 10
    assert pool["faults"] == []
    assert pool["counters"]["local_model_calls"] == 0
    assert pool["counters"]["hosted_model_calls"] == 0
    assert pool["counters"]["deterministic_request_compiler_calls"] == 0
    assert pool["counters"]["compiler_oracle_evaluator_audits"] == 10
    assert digest(ROOT / pool["source_registry"]) == pool["source_registry_sha256"]
    assert digest(ROOT / pool["instrument"]) == pool["instrument_sha256"]
    assert digest(ROOT / pool["instrument_audit"]) == pool["instrument_audit_sha256"]


def test_p4_registry_is_distinct_source_disjoint_and_pre_candidate() -> None:
    registry = read(ROOT / "configs" / "theseus_p4_task_sources.json")
    assert task_pool.audit_registry(registry) == []
    tasks = registry["tasks"]
    repositories = {row["repository"].lower() for row in tasks}
    prior = {
        value.lower()
        for value in task_pool.P2_REPOSITORIES | task_pool.p3_repositories()
    }

    assert len(tasks) == 10
    assert len(repositories) == 10
    assert repositories.isdisjoint(prior)
    assert registry["boundaries"] == {
        "candidate_generation_opened": False,
        "local_model_calls": 0,
        "hosted_model_calls": 0,
        "deterministic_request_compiler_calls": 0,
        "compiler_oracle_calls_before_selection": 0,
        "user_task_or_label_dependency": False,
    }


def test_p4_tasks_bind_obligations_blind_evaluators_and_green_evidence() -> None:
    pool = read(ROOT / "configs" / "theseus_p4_task_pool.json")
    for index, row in enumerate(pool["tasks"], 1):
        task_path = ROOT / row["task"]
        evaluator_path = ROOT / row["evaluator"]
        oracle_path = ROOT / row["oracle_ir"]
        audit_path = ROOT / row["evaluator_audit"]
        task = read(task_path)
        evaluator = read(evaluator_path)
        audit = read(audit_path)

        assert row["campaign_index"] == index
        assert digest(task_path) == row["task_sha256"]
        assert digest(evaluator_path) == row["evaluator_sha256"]
        assert digest(oracle_path) == row["oracle_ir_sha256"]
        assert digest(audit_path) == row["evaluator_audit_sha256"]
        assert p4.audit_task(task_path)["trigger_state"] == "GREEN"
        assert len(task["obligations"]) >= 3
        assert any(item["kind"] == "preserve" for item in task["obligations"])
        assert evaluator["blindness"]["candidate_generation_may_read_this_manifest"] is False
        assert evaluator["blindness"]["route_label_passed_to_scoring"] is False
        assert evaluator["blindness"]["hidden_test_candidate_visible"] is False
        assert evaluator["blindness"]["oracle_candidate_visible"] is False
        assert audit["trigger_state"] == "GREEN"
        assert audit["baseline_verification"]["hidden_passed"] is False
        assert audit["target_verification"]["hidden_passed"] is True
        assert audit["compiler_oracle_verification"]["visible_passed"] is True
        assert audit["compiler_oracle_verification"]["hidden_passed"] is True
        assert all(audit["corruption_intervention_rejections"].values())
        screen = task["contamination_screen"]
        assert screen["public_benchmark"] is False
        assert screen["previous_theseus_surface"] is False
        assert screen["source_disjoint_from_p2_p3"] is True
        assert screen["later_patch_hidden_test_or_oracle_candidate_visible"] is False
        assert screen["development_task_eligible_for_training"] is False
        assert screen["development_task_eligible_for_D1_or_D2"] is False


def test_archive_omissions_are_symmetric_links_outside_required_paths() -> None:
    registry = read(ROOT / "configs" / "theseus_p4_task_sources.json")
    for source in registry["tasks"]:
        stem = source["stem"]
        parent_report = read(ROOT / "reports" / f"theseus_{stem}_parent_archive_sanitization.json")
        target_report = read(ROOT / "reports" / f"theseus_{stem}_target_archive_sanitization.json")
        fixture = ROOT / "tests" / "fixtures" / "theseus_p4_online"
        faults = task_pool.audit_sanitization_pair(
            parent_report,
            target_report,
            fixture / f"{stem}_parent.tar.gz",
            fixture / f"{stem}_parent_upstream.tar.gz",
            fixture / f"{stem}_target.tar.gz",
            fixture / f"{stem}_target_upstream.tar.gz",
            source["source_root"],
            source["target_root"],
            source,
        )
        assert faults == []


def test_pytest_task_exercises_selective_dependency_local_repair() -> None:
    task = read(ROOT / "configs" / "theseus_p4_task_02_pytest_14586.json")
    selected = p4.implicated_obligations(
        task,
        {"faults": []},
        {
            "apply_faults": [],
            "visible_verifier": {
                "passed": False,
                "stdout_tail": "",
                "stderr_tail": "P4_VISIBLE_pytest_PRIMARY",
            },
        },
    )
    assert selected == {"O1", "O3"}
    first = [
        {"id": "U1", "obligation_ids": ["O1", "O3"], "operation": "REPLACE", "path": "src/_pytest/logging.py", "node_id": "N1", "replacement_sha256": "a"},
        {"id": "U2", "obligation_ids": ["O2", "O4"], "operation": "REPLACE", "path": "src/_pytest/logging.py", "node_id": "N2", "replacement_sha256": "b"},
    ]
    permitted = [dict(first[0], replacement_sha256="c"), first[1]]
    unrelated = [first[0], dict(first[1], replacement_sha256="d")]
    assert p4.repair_locality_faults(first, permitted, selected) == []
    assert p4.repair_locality_faults(first, unrelated, selected) == [
        "semantic_repair_not_dependency_local"
    ]


def test_p4_live_evaluators_remain_green() -> None:
    pool = read(ROOT / "configs" / "theseus_p4_task_pool.json")
    for row in pool["tasks"]:
        assert p4_evaluator.audit_evaluator(ROOT / row["evaluator"])["trigger_state"] == "GREEN"
