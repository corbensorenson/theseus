from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import core_evidence_local_8b_qualification as qualification  # noqa: E402


PUBLIC_PATH = (
    ROOT / "configs" / "core_evidence_tmax_fresh_v1_qualification_public.json"
)
EVALUATOR_PATH = (
    ROOT
    / "configs"
    / "core_evidence_tmax_fresh_v1_qualification_evaluator.json"
)
FREEZE_PATH = (
    ROOT / "configs" / "core_evidence_tmax_fresh_v1_qualification_freeze.json"
)
ALIGNMENT_PATH = (
    ROOT / "reports" / "core_evidence_tmax_fresh_v1_alignment_audit.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def requested_scripts(manifest: dict) -> set[str]:
    return {
        match
        for task in manifest.get("tasks", [])
        for match in re.findall(
            r"scripts/[A-Za-z0-9_./-]+\.py",
            str(task.get("natural_request") or ""),
        )
    }


def test_tmax_freeze_binds_sealed_candidate_sources_and_detects_later_drift() -> None:
    public = read_json(PUBLIC_PATH)
    freeze = read_json(FREEZE_PATH)
    sealed_candidates = read_json(
        ROOT
        / "reports"
        / "core_evidence_tmax_fresh_v1_1_qualification_candidates.json"
    )
    worker_config = read_json(
        ROOT / freeze["candidate_worker_config_path"]
    )

    assert (
        sealed_candidates["source_identities"]
        == freeze["candidate_source_identities"]
    )
    with pytest.raises(
        ValueError,
        match="candidate_source_mutated_after_freeze",
    ):
        qualification.validate_frozen_inputs(public, freeze, PUBLIC_PATH)

    assert worker_config["policy"] == (
        "project_theseus_local_8b_stack_worker_v1"
    )
    assert freeze["candidate_worker_config_path"] == (
        "configs/core_evidence_tmax_9b_worker_control_v3.json"
    )
    assert freeze["competence_floor"] == {
        "minimum_attempted_tasks": 3,
        "minimum_useful_rate": 0.5,
        "minimum_weakest_family_rate": 0.34,
    }
    assert freeze["terminal_rules"]["run_once"] is True


def test_tmax_cohort_is_source_disjoint_and_public_manifest_is_inert() -> None:
    public = read_json(PUBLIC_PATH)
    current = requested_scripts(public)
    historical: set[str] = set()
    for path in sorted((ROOT / "configs").glob("*qualification_public*.json")):
        if path == PUBLIC_PATH:
            continue
        historical.update(requested_scripts(read_json(path)))

    assert current == {
        "scripts/external_inference_audit.py",
        "scripts/hive_fleet_readiness.py",
        "scripts/self_evolution_governor.py",
    }
    assert current.isdisjoint(historical)
    serialized = json.dumps(public["tasks"], sort_keys=True)
    assert "hidden_test" not in serialized
    assert "target_patch" not in serialized
    assert "target_commit" not in serialized


def test_tmax_alignment_and_runtime_qualification_are_green() -> None:
    alignment = read_json(ALIGNMENT_PATH)
    assert alignment["trigger_state"] == "GREEN"
    assert alignment["summary"] == {
        "aligned_task_count": 3,
        "target_commit_count": 0,
        "target_patch_count": 0,
        "task_count": 3,
    }
    assert all(
        task["baseline_failed_as_expected"]
        for task in alignment["tasks"]
    )
    for name in (
        "core_evidence_tmax_9b_runtime_preflight.json",
        "core_evidence_tmax_9b_long_context_8k_qualification.json",
        "core_evidence_tmax_9b_long_context_15k_qualification.json",
    ):
        assert read_json(ROOT / "reports" / name)["trigger_state"] == "GREEN"


def test_tmax_evaluator_identity_is_frozen_without_opening_it_to_candidate() -> None:
    freeze = read_json(FREEZE_PATH)
    assert freeze["evaluator_manifest_sha256"] == (
        qualification.sha256_file(EVALUATOR_PATH)
    )
    assert freeze["counters_at_freeze"]["candidate_generation_calls"] == 0
    assert freeze["counters_at_freeze"]["target_commits_opened"] == 0
    assert freeze["counters_at_freeze"]["target_patches_opened"] == 0
