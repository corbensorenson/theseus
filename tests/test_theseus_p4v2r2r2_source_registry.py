from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2r2_source_registry as registry  # noqa: E402


def test_fresh_source_registry_is_prospective_disjoint_and_uncapped() -> None:
    report = registry.audit()

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["task_count"] == 10
    assert report["distinct_repository_count"] == 10
    assert report["prior_repository_count"] == 50
    assert report["prior_repository_overlap"] == []
    assert report["candidate_or_control_calls"] == 0
    assert report["archive_fetches"] == 0
    assert report["project_selected_quality_token_cap"] is None
    assert report["instrument"]["binding_mode"] == "prospective_pre_generation_repair"


def test_fresh_source_registry_has_complexity_gradient_and_no_user_gate() -> None:
    report = registry.audit()
    value = p2a.read_json(registry.REGISTRY)

    assert report["behavioral_unit_distribution"] == {"1": 5, "2": 4, "3": 1}
    assert value["boundaries"]["user_task_label_or_approval_dependency"] is False
    assert value["boundaries"]["candidate_generation_opened"] is False
    assert all(
        set(row["allowed_effect_paths"]).issubset(row["patch_changed_files"])
        for row in value["tasks"]
    )
