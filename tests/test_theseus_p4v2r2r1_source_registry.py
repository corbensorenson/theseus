from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r1_source_registry as registry  # noqa: E402


def test_recovery_source_registry_is_green_and_exactly_ten() -> None:
    report = registry.audit()

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["carried_candidate_unseen_tasks"] == 9
    assert report["replacement_tasks"] == 1
    assert report["sealed_successor_tasks"] == 10


def test_replacement_is_source_disjoint_and_preconsumption() -> None:
    report = registry.audit()

    assert report["replacement_repository"] == "pydantic/pydantic-ai"
    assert report["replacement_source_disjoint"] is True
    assert report["new_archive_fetches"] == 0
    assert report["successor_candidate_or_control_calls"] == 0
    assert report["project_selected_quality_token_cap"] is None
