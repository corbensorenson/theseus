from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import core_evidence_local_8b_freeze as freeze_builder  # noqa: E402
import core_evidence_local_8b_qualification as qualification  # noqa: E402


PUBLIC = ROOT / "configs" / "core_evidence_local_8b_qualification_public.json"
EVALUATOR = (
    ROOT / "configs" / "core_evidence_local_8b_qualification_evaluator.json"
)
AUDIT = ROOT / "reports" / "core_evidence_local_8b_alignment_audit.json"
FREEZE = ROOT / "configs" / "core_evidence_local_8b_qualification_freeze.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_completed_frozen_qualification_cannot_rerun_after_source_change() -> None:
    with pytest.raises(ValueError, match="candidate_source_mutated_after_freeze"):
        qualification.validate_frozen_inputs(read(PUBLIC), read(FREEZE), PUBLIC)
    completed = read(
        ROOT / "reports" / "core_evidence_local_8b_qualification_candidates.json"
    )
    assert completed["source_identities"] == read(FREEZE)[
        "candidate_source_identities"
    ]


def test_freeze_binds_green_target_free_alignment_audit() -> None:
    value = freeze_builder.build(PUBLIC, EVALUATOR, AUDIT)
    assert value["trigger_state"] == "GREEN"
    assert len(value["task_identities"]) == 3
    assert value["counters_at_freeze"]["target_commits_opened"] == 0
    assert value["counters_at_freeze"]["target_patches_opened"] == 0
    assert len(
        value["evaluator_source_identities"]["hidden_test_sources"]
    ) == 3


def test_public_task_mutation_is_rejected(tmp_path: Path) -> None:
    public = read(PUBLIC)
    public["tasks"][0]["natural_request"] += " changed"
    changed = tmp_path / "public.json"
    changed.write_text(json.dumps(public), encoding="utf-8")
    with pytest.raises(ValueError, match="public_manifest_mutated_after_freeze"):
        qualification.validate_frozen_inputs(public, read(FREEZE), changed)


def test_public_evaluator_request_mismatch_is_rejected() -> None:
    evaluator = copy.deepcopy(read(EVALUATOR))
    evaluator["tasks"][0]["natural_request"] += " changed"
    with pytest.raises(
        ValueError,
        match="public_evaluator_natural_request_mismatch",
    ):
        freeze_builder.validate_pair(read(PUBLIC), evaluator, read(AUDIT))
