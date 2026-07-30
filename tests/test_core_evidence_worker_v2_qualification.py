from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import core_evidence_worker_v2_evaluator as evaluator  # noqa: E402
import core_evidence_worker_v2_qualification as qualification  # noqa: E402


def public_manifest() -> dict:
    return json.loads(
        (
            ROOT
            / "configs"
            / "core_evidence_worker_v2_qualification_public.json"
        ).read_text()
    )


def source_identities() -> dict[str, str]:
    return {
        "worker_sha256": qualification.sha256_file(qualification.WORKER),
        "worker_config_sha256": qualification.sha256_file(
            qualification.WORKER_CONFIG
        ),
        "qualification_runner_sha256": qualification.sha256_file(
            Path(qualification.__file__)
        ),
        "evaluator_sha256": qualification.sha256_file(
            qualification.EVALUATOR
        ),
    }


def test_fresh_requests_pass_adequacy_before_target_open() -> None:
    manifest = public_manifest()
    assert all(
        qualification.request_adequacy(task["natural_request"])["passed"]
        for task in manifest["tasks"]
    )
    assert manifest["request_adequacy_policy"][
        "target_patch_inspected_before_freeze"
    ] is False


def test_terse_commit_title_cannot_enter_fresh_qualification() -> None:
    audit = qualification.request_adequacy(
        "Remove arbitrary training launch limits"
    )
    assert not audit["passed"]
    assert "fewer_than_24_words" in audit["faults"]
    assert "verification_expectation_missing" in audit["faults"]


def test_manifest_validation_rejects_changed_competence_floor(
    tmp_path: Path,
) -> None:
    manifest = public_manifest()
    path = tmp_path / "public.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    freeze = {
        "policy": "project_theseus_worker_v2_fresh_qualification_freeze_v1",
        "source_identities": source_identities(),
        "public_manifest_sha256": qualification.sha256_file(path),
    }
    qualification.validate_manifest(manifest, freeze, path)
    manifest["competence_floor"]["minimum_useful_rate"] = 0.49
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    freeze["public_manifest_sha256"] = qualification.sha256_file(path)
    with pytest.raises(ValueError, match="competence floor changed"):
        qualification.validate_manifest(manifest, freeze, path)


def test_fresh_evaluator_rejects_nonzero_boundary_counter(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "sealed_before_target_open": True,
            "candidate_seal": {"target_opened_before_seal": False},
        }
        for _ in range(3)
    ]
    report = {
        "policy": (
            "project_theseus_worker_v2_fresh_qualification_candidates_v1"
        ),
        "trigger_state": "GREEN",
        "faults": [],
        "tasks": rows,
        "counters": {
            "D2_cases_consumed": 1,
        },
    }
    report_path = tmp_path / "candidate.json"
    report_path.write_text(json.dumps(report))
    with pytest.raises(
        evaluator.EvaluationFault,
        match="boundary counter nonzero",
    ):
        evaluator.validate_fresh_candidate_set(
            report,
            report_path,
            {},
            tmp_path / "campaign.json",
            ROOT,
        )
