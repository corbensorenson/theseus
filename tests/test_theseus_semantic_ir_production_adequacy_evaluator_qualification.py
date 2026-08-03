from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_semantic_ir_production_adequacy_evaluator_qualification as qualification  # noqa: E402


def test_full_evaluator_qualification_is_green_and_pre_model() -> None:
    report = qualification.qualify()
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["task_count"] == 18
    assert report["green_task_count"] == 18
    assert report["panel_admitted_for_task_packet_materialization"] is True
    assert report["candidate_or_model_exposure_authorized"] is False
    for key in (
        "candidate_or_control_calls",
        "local_model_calls",
        "external_inference_calls",
        "teacher_calls",
        "training_rows_written",
        "D1_cases_consumed",
        "D2_cases_consumed",
    ):
        assert report["counters"][key] == 0


def test_every_task_rejects_parent_and_controls_but_accepts_target_and_benign() -> None:
    report = qualification.qualify()
    expected = {
        "parent_negative": True,
        "target_positive": True,
        "benign_equivalent_positive": True,
        "required_mechanism_mutation_rejected": True,
        "missing_required_path_rejected": True,
        "unauthorized_path_rejected": True,
    }
    assert [row["index"] for row in report["rows"]] == list(range(1, 19))
    assert all(row["checks"] == expected for row in report["rows"])


def test_persisted_report_binds_current_evaluator_owner() -> None:
    report = json.loads(
        (
            ROOT
            / "reports"
            / "theseus_semantic_ir_production_adequacy_evaluator_qualification.json"
        ).read_text()
    )
    assert report["evaluator_owner"]["sha256"] == qualification.sha256_file(
        Path(qualification.__file__).resolve()
    )
