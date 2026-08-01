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


def test_sealed_online_pool_binds_green_task_and_evaluator() -> None:
    pool = json.loads((ROOT / "configs" / "theseus_p2a_task_pool.json").read_text(encoding="utf-8"))
    task_path = ROOT / pool["p2a_instrument_adequacy_task"]
    evaluator_path = ROOT / pool["p2a_evaluator"]
    audit_path = ROOT / pool["p2a_evaluator_audit"]

    assert pool["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION"
    assert pool["candidate_generation_opened"] is False
    assert pool["task_count"] == 1
    assert digest(task_path) == pool["p2a_instrument_adequacy_task_sha256"]
    assert digest(evaluator_path) == pool["p2a_evaluator_sha256"]
    assert digest(audit_path) == pool["p2a_evaluator_audit_sha256"]
    assert p2a.audit_task(task_path)["trigger_state"] == "GREEN"
    assert evaluator.audit_evaluator(evaluator_path)["trigger_state"] == "GREEN"


def test_parent_and_target_archives_are_exact_licensed_evaluator_boundaries() -> None:
    task = json.loads(
        (ROOT / "configs" / "theseus_p2a_task_typing_extensions_677.json").read_text(encoding="utf-8")
    )
    evaluator_config = json.loads(
        (ROOT / "configs" / "theseus_p2a_evaluator_typing_extensions_677.json").read_text(encoding="utf-8")
    )
    parent = ROOT / task["source_archive"]
    target = ROOT / evaluator_config["target_archive"]

    assert task["source_provenance"]["license_spdx"] == "PSF-2.0"
    assert digest(parent) == task["source_archive_sha256"]
    assert digest(target) == evaluator_config["target_archive_sha256"]
    with tarfile.open(parent) as archive:
        names = set(archive.getnames())
    assert "typing_extensions_677_parent/LICENSE" in names
    assert "typing_extensions_677_parent/src/typing_extensions.py" in names
    assert "target_archive" not in task
    assert "target_provenance" not in task


def test_task_is_development_only_and_not_a_consumed_surface() -> None:
    task = json.loads(
        (ROOT / "configs" / "theseus_p2a_task_typing_extensions_677.json").read_text(encoding="utf-8")
    )
    screen = task["contamination_screen"]
    prior_p2 = (ROOT / "configs" / "theseus_assistant_p2_canary.json").read_text(encoding="utf-8")

    assert screen["public_benchmark"] is False
    assert screen["previous_theseus_surface"] is False
    assert screen["task_selected_before_candidate_generation"] is True
    assert screen["later_patch_or_tests_candidate_visible"] is False
    assert screen["development_task_eligible_for_training"] is False
    assert screen["development_task_eligible_for_D1_or_D2"] is False
    assert task["opaque_task_id"] not in prior_p2
