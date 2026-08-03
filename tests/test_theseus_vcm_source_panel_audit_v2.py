from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_vcm_source_panel_audit_v2 as audit  # noqa: E402


def test_repaired_real_source_panel_is_green() -> None:
    report = audit.audit()
    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "SIXTY_TWO_ENGLISH_SOURCE_TASKS_ADMITTED_BEFORE_EVALUATOR_EXECUTION"
    assert report["source_panel_admitted"] is True
    assert report["assembled_task_count"] == 62
    assert report["unique_repository_count"] == 62
    assert report["english_eligible_task_count"] == 62
    assert report["archive_receipt_count"] == 248
    assert report["member_receipt_count"] == 681
    assert report["selected_source_difference_count"] == 62
    assert report["selected_verifier_difference_count"] == 62
    assert report["superseded_non_english_archive_count_preserved_but_ignored"] == 24
    assert report["candidate_packet_materialization_opened"] is False
    assert report["parent_target_or_evaluator_executions"] == 0
    assert report["local_model_calls"] == 0
    assert report["external_reference_calls"] == 0
    assert report["faults"] == []


def test_only_declared_slots_are_substituted() -> None:
    report = audit.audit()
    rows = {row["index"]: row for row in report["assembled_rows"]}
    assert rows[28]["repository"] == "moritan777/chat-trpg-gm-mvp"
    assert {rows[index]["repository"] for index in audit.LANGUAGE_REPLACEMENT_INDICES} == {
        "endoftheline818/opensourcesai-cmdcenter",
        "rickli0822-prog/windows-ai-login-doctor",
        "Ganesh-403/semantic-plagiarism-detector",
        "pokle/glidecomp",
        "inkeep/open-knowledge",
        "AbsoluteMode/session-recall",
    }
    original = audit.p2a.read_json(ROOT / "reports" / "theseus_vcm_source_materialization.json")
    for row in original["rows"]:
        if row["index"] not in audit.LANGUAGE_REPLACEMENT_INDICES | {28}:
            assert rows[row["index"]]["repository"] == row["repository"]
