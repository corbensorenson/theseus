from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_assistant_p2a_evaluator as evaluator  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p2c_pool_binds_frozen_instrument_task_and_blind_evaluator() -> None:
    pool = json.loads((ROOT / "configs" / "theseus_p2c_task_pool.json").read_text(encoding="utf-8"))
    task_path = ROOT / pool["task"]
    evaluator_path = ROOT / pool["evaluator"]
    audit_path = ROOT / pool["evaluator_audit"]

    assert pool["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION"
    assert pool["instrument_freeze_commit"] == "9b87c861ce37febdd6daed3cdc700da08f8d6471"
    assert pool["candidate_generation_opened"] is False
    assert pool["counters"]["candidate_model_calls"] == 0
    assert digest(ROOT / pool["instrument"]) == pool["instrument_sha256"]
    assert digest(task_path) == pool["task_sha256"]
    assert digest(evaluator_path) == pool["evaluator_sha256"]
    assert digest(audit_path) == pool["evaluator_audit_sha256"]
    assert p2a.audit_task(task_path)["trigger_state"] == "GREEN"
    assert evaluator.audit_evaluator(evaluator_path)["trigger_state"] == "GREEN"


def test_p2c_click_task_uses_repo_relative_paths_and_retains_upstream_archives() -> None:
    task = json.loads((ROOT / "configs" / "theseus_p2c_task_click_3578.json").read_text(encoding="utf-8"))
    evaluator_config = json.loads(
        (ROOT / "configs" / "theseus_p2c_evaluator_click_3578.json").read_text(encoding="utf-8")
    )

    assert task["source_archive_root"] == "pallets-click-8929d39"
    assert task["allowed_effect_paths"] == ["src/click/core.py"]
    assert all(not path.startswith(task["source_archive_root"] + "/") for path in task["allowed_effect_paths"])
    assert task["source_provenance"]["license_spdx"] == "BSD-3-Clause"
    assert digest(ROOT / task["source_provenance"]["upstream_archive"]) == task["source_provenance"]["upstream_archive_sha256"]
    assert digest(ROOT / evaluator_config["target_provenance"]["upstream_archive"]) == evaluator_config["target_provenance"]["upstream_archive_sha256"]
    with tarfile.open(ROOT / task["source_archive"]) as archive:
        members = archive.getmembers()
    assert members
    assert all(not member.issym() and not member.islnk() for member in members)


def test_p2c_click_task_is_source_disjoint_and_never_training_or_claim_evidence() -> None:
    pool = json.loads((ROOT / "configs" / "theseus_p2c_task_pool.json").read_text(encoding="utf-8"))
    task = json.loads((ROOT / "configs" / "theseus_p2c_task_click_3578.json").read_text(encoding="utf-8"))
    screen = task["contamination_screen"]

    assert set(pool["source_disjoint_from"]) == {
        "p2a-typing-qualifier-inheritance-001",
        "p2b-multipart-dynamic-read-wrapper-001",
    }
    assert screen["public_benchmark"] is False
    assert screen["previous_theseus_surface"] is False
    assert screen["source_disjoint_from_p2a_and_p2b"] is True
    assert screen["later_patch_or_tests_candidate_visible"] is False
    assert screen["development_task_eligible_for_training"] is False
    assert screen["development_task_eligible_for_D1_or_D2"] is False
