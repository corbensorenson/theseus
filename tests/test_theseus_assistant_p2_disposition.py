from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_disposition", ROOT / "scripts" / "theseus_assistant_p2_disposition.py")
assert SPEC and SPEC.loader
disposition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(disposition)


def test_terminal_disposition_consumes_the_single_repair_and_requires_p2a() -> None:
    report = disposition.build_disposition(
        ROOT / "reports" / "theseus_assistant_p2_canary_r1.json",
        ROOT / "reports" / "theseus_assistant_p2_evaluation_r1.json",
        ROOT / "configs" / "theseus_assistant_p2_canary_repair_r1.json",
    )
    assert report["trigger_state"] == "GREEN", report["failed_checks"]
    assert report["disposition"] == "P2A_INSTRUMENT_INADEQUATE_REBUILD_REQUIRED"
    assert report["stage_states"]["P2"] == "COMPLETE_INCONCLUSIVE_INSTRUMENT"
    assert report["stage_states"]["P2A"] == "REBUILD_REQUIRED"
    assert report["stage_states"]["P3"] == "BLOCKED_ON_P2A_ADEQUACY"
    assert report["scientific_scope"]["Theseus_subsystem_result"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert report["scientific_scope"]["TMax_model_generally_falsified"] is False
    assert report["ablation_disposition"]["subsystem_specific_utility_ablation_run"] is False


def test_both_arms_remain_scoped_as_shared_cap_failures() -> None:
    report = disposition.build_disposition(
        ROOT / "reports" / "theseus_assistant_p2_canary_r1.json",
        ROOT / "reports" / "theseus_assistant_p2_evaluation_r1.json",
        ROOT / "configs" / "theseus_assistant_p2_canary_repair_r1.json",
    )
    arms = report["arm_observations"]
    assert set(arms) == {"direct_local_model", "integrated_local_model"}
    assert all(row["generated_tokens"] == 512 for row in arms.values())
    assert all(row["patch_applied"] == 0 for row in arms.values())
    assert all(row["unsafe"] == 0 for row in arms.values())
