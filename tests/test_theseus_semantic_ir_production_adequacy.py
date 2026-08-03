from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_semantic_ir_production_adequacy as adequacy  # noqa: E402


def test_preregistration_is_source_bound_powered_and_zero_call() -> None:
    report = adequacy.audit(
        ROOT / "configs" / "theseus_semantic_ir_production_adequacy.json"
    )
    assert report["trigger_state"] == "GREEN"
    assert report["stage"] == "preregistration_audit"
    assert report["faults"] == []
    design = report["design_recomputation"]
    assert design["panel_size"] == 18
    assert design["minimum_successes"] == 13
    assert round(design["one_sided_false_positive_probability"], 10) == 0.0481262207
    assert design["power_at_adequate_probability"] >= 0.8
    assert set(report["counters"].values()) == {0}


def test_pool_audit_rejects_prior_source_and_pre_generation_calls() -> None:
    config = json.loads(
        (ROOT / "configs" / "theseus_semantic_ir_production_adequacy.json").read_text()
    )
    tasks = []
    for index, stratum in enumerate(config["competence_design"]["strata"]):
        for offset in range(3):
            tasks.append({
                "repository": f"fresh/repo-{index}-{offset}",
                "merged_utc": "2026-08-01T00:00:00Z",
                "license_verified": True,
                "parent_failed_independent_evaluator": True,
                "target_passed_independent_evaluator": True,
                "candidate_visible_target_or_hidden_data": False,
                "stratum": stratum["id"],
            })
    pool = {
        "state": "SEALED_BEFORE_ANY_MODEL_OR_CONTROL_CALL",
        "tasks": tasks,
        "counters": {
            "local_model_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
    }
    assert adequacy.audit_pool(pool, config)["faults"] == []
    pool["tasks"][0]["repository"] = "pydantic/pydantic"
    pool["counters"]["local_model_calls"] = 1
    faults = adequacy.audit_pool(pool, config)["faults"]
    assert "source_overlap_with_prior_theseus_repository" in faults
    assert "pre_generation_counter_nonzero:local_model_calls" in faults


def test_terminal_result_requires_overall_and_weak_tail_floors() -> None:
    config = json.loads(
        (ROOT / "configs" / "theseus_semantic_ir_production_adequacy.json").read_text()
    )
    rows = []
    for stratum in config["competence_design"]["strata"]:
        rows.extend({
            "stratum": stratum["id"],
            "production_mechanics_passed": offset < 2,
            "candidate_integrity_recomputed_independently": True,
            "physical_context_boundary_hit": False,
            "unsafe_or_unauthorized_effect": False,
        } for offset in range(3))
    rows[2]["production_mechanics_passed"] = True
    report = adequacy.audit_results(
        {"tasks": rows}, config, {"faults": []}
    )
    assert report["successes"] == 13
    assert report["weak_tail_passed"] is True
    assert report["adequate"] is True
    rows[3]["production_mechanics_passed"] = False
    rows[4]["production_mechanics_passed"] = False
    rows[5]["production_mechanics_passed"] = False
    report = adequacy.audit_results(
        {"tasks": rows}, config, {"faults": []}
    )
    assert report["weak_tail_passed"] is False
    assert report["adequate"] is False
