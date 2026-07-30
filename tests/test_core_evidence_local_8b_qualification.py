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
    assert value["candidate_worker_config_path"] == (
        "configs/core_evidence_local_8b_worker.json"
    )


def test_freeze_can_bind_a_distinct_repository_worker_config(
    tmp_path: Path,
) -> None:
    config = tmp_path / "outside.json"
    config.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worker_config_outside_repository"):
        freeze_builder.build(PUBLIC, EVALUATOR, AUDIT, config)

    value = freeze_builder.build(
        PUBLIC,
        EVALUATOR,
        AUDIT,
        ROOT / "configs" / "core_evidence_qwen35_9b_worker_planned.json",
    )
    assert value["candidate_worker_config_path"] == (
        "configs/core_evidence_qwen35_9b_worker_planned.json"
    )
    assert value["candidate_source_identities"]["worker_config_sha256"] == (
        qualification.sha256_file(
            ROOT / "configs" / "core_evidence_qwen35_9b_worker_planned.json"
        )
    )


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


def test_runner_passes_the_frozen_worker_config_to_every_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_config = (
        ROOT / "configs" / "core_evidence_qwen35_9b_worker_planned.json"
    )
    public = read(
        ROOT / "configs" / "core_evidence_qwen35_fresh_qualification_public.json"
    )
    freeze = {
        "candidate_worker_config_path": str(selected_config.relative_to(ROOT)),
    }
    public_path = tmp_path / "public.json"
    freeze_path = tmp_path / "freeze.json"
    public_path.write_text(json.dumps(public), encoding="utf-8")
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    observed: list[Path] = []

    monkeypatch.setattr(
        qualification,
        "validate_frozen_inputs",
        lambda *_args: None,
    )

    def fake_run_task(
        task: dict,
        config: dict,
        *,
        config_path: Path,
    ) -> dict:
        assert config == read(selected_config)
        observed.append(config_path)
        return {
            "opaque_task_id": task["opaque_task_id"],
            "sealed_before_target_open": True,
            "learned_generation_credit": 1,
        }

    monkeypatch.setattr(
        qualification.development,
        "run_task",
        fake_run_task,
    )
    report = qualification.run(public_path, freeze_path)

    assert report["trigger_state"] == "GREEN"
    assert observed == [selected_config] * 3
