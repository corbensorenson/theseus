from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import autonomy_watchdog_actions as watchdog_actions  # noqa: E402
import run_training_ratchet_profile as ratchet  # noqa: E402
import vcm_official_public_memory_adapter as memory_adapter  # noqa: E402
import viea_growth_surfaces as growth_surfaces  # noqa: E402


def test_watchdog_recent_correction_parses_iso_timestamp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created = watchdog_actions.now()
    report = {
        "ok": True,
        "created_utc": created,
        "decision": {"reason": "teacher_blocked_local_correction"},
    }
    (tmp_path / "autonomy_cycle_watchdog_correction.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    monkeypatch.setattr(watchdog_actions, "REPORTS", tmp_path)

    assert watchdog_actions.recent_teacher_blocked_correction(max_age_seconds=60)


def test_student_candidate_paths_use_the_existing_card_slug() -> None:
    assert (
        ratchet.student_code_candidate_manifest_path("source/example-card", 23)
        == "reports/student_code_candidates_source_example_card_seed23.jsonl"
    )
    step = ratchet.student_code_candidate_generator_step(
        "source/example-card",
        23,
        max_cases_per_card=1,
    )
    assert "source_example_card" in " ".join(step["command"])


def test_babilong_person_query_uses_shared_question_parser() -> None:
    context = "Mary went to the hallway.\nMary travelled to the kitchen."

    assert (
        memory_adapter.resolve_babilong(context, "Where is Mary?", "qa1") == "kitchen"
    )


def test_growth_surface_state_aggregation_accepts_iterables() -> None:
    payloads = ({"trigger_state": state} for state in ("GREEN", "YELLOW"))

    assert growth_surfaces.aggregate_state(payloads) == "YELLOW"
