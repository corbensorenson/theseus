from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2_task_pool as pool  # noqa: E402


def registry() -> dict:
    return json.loads(pool.SOURCE_REGISTRY.read_text(encoding="utf-8"))


def test_p4v2r2_pool_binds_committed_source_and_evaluator_transactions() -> None:
    assert pool.SOURCE_SELECTION_COMMIT == (
        "8cebe4a65bb03965e9f62efa8249f2f9ddb8fc08"
    )
    assert pool.SOURCE_ACQUISITION_COMMIT == (
        "3ea8d770f4d59061f7b9ae128e8877917b8fd570"
    )
    assert pool.EVALUATOR_SURFACE_COMMIT == (
        "c9ab9f1346fa96ebd06ab70cc49fcf087fbd6daa"
    )
    assert pool.audit_registry(registry()) == []


def test_p4v2r2_pool_rejects_opened_or_nonzero_selection_boundary() -> None:
    opened = copy.deepcopy(registry())
    opened["boundaries"]["candidate_generation_opened"] = True
    nonzero = copy.deepcopy(registry())
    nonzero["boundaries"]["local_model_calls"] = 1

    assert "candidate_generation_already_opened" in pool.audit_registry(opened)
    assert "selection_boundary_nonzero:local_model_calls" in pool.audit_registry(
        nonzero
    )


def test_p4v2r2_selector_correction_is_exact_and_non_membership_bearing() -> None:
    correction = pool.selector_correction_map()
    assert correction == {
        ("p4v2r2_03_textual_6592", "U2"): {
            "node_type": "FunctionDef",
            "qualified_name": "XTermParser.parse",
        }
    }
    assert pool.obligation_correction_map() == {
        ("p4v2r2_03_textual_6592", "U1"): ["O1", "O2"]
    }
    assert pool.unit_order_correction_map() == {
        "p4v2r2_03_textual_6592": ["U4", "U3", "U2", "U1"]
    }
    assert pool.target_decorator_correction_set() == {
        ("p4v2r2_03_textual_6592", "U2")
    }
    assert pool.baseline_failure_marker_map() == {
        "p4v2r2_05_pyflakes_765": [
            "TypeError: 'bool' object is not subscriptable"
        ]
    }


def test_p4v2r2_pool_has_no_quality_token_cap_or_user_gate_after_seal() -> None:
    path = ROOT / "configs" / "theseus_p4v2r2_task_pool.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["candidate_generation_opened"] is False
    assert value["generation_budget"]["project_selected_quality_token_cap"] is None
    assert value["counters"]["local_model_calls"] == 0
    assert value["counters"]["hosted_model_calls"] == 0
    assert value["task_count"] == value["distinct_repositories"] == 10


def test_p4v2r2_sealed_pool_meets_every_mechanics_floor() -> None:
    value = json.loads(
        (ROOT / "configs" / "theseus_p4v2r2_task_pool.json").read_text(
            encoding="utf-8"
        )
    )
    assert value["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION"
    assert value["faults"] == []
    assert value["candidate_generation_opened"] is False
    assert value["task_count"] == 10
    assert value["green_evaluator_audits"] == 10
    assert value["v2r2_oracle_replays_green"] == 10
    assert value["dependency_corruptions_rejected"] == 10
    for row in value["tasks"]:
        assert row["evaluator_audit_trigger_state"] == "GREEN"
        assert row["baseline_parent_failed"] is True
        assert row["upstream_target_passed"] is True
        assert row["compiler_oracle_v1_passed"] is True
        assert row["four_base_corruptions_rejected"] is True
        assert row["dependency_corruption"]["rejected"] is True
        assert row["v2r2_oracle_replay"]["trigger_state"] == "GREEN"
        assert row["v2r2_oracle_replay"]["v1_v2r2_actions_equivalent"] is True
        assert row["v2r2_oracle_replay"]["visible_passed"] is True
        assert row["v2r2_oracle_replay"]["hidden_passed"] is True


def test_p4v2r2_pool_binds_both_oracle_transports_without_candidate_visibility() -> None:
    value = json.loads(
        (ROOT / "configs" / "theseus_p4v2r2_task_pool.json").read_text(
            encoding="utf-8"
        )
    )
    for row in value["tasks"]:
        v1 = ROOT / row["oracle_ir"]
        v2 = ROOT / row["treatment_transport_oracle_ir"]
        assert v1.read_text(encoding="utf-8").startswith("THESEUS_SEMANTIC_IR_V1\n")
        assert v2.read_text(encoding="utf-8").startswith("THESEUS_SEMANTIC_IR_V2\n")
        evaluator = json.loads((ROOT / row["evaluator"]).read_text(encoding="utf-8"))
        assert evaluator["blindness"]["oracle_candidate_visible"] is False
        assert evaluator["oracle_ir_file"] == row["oracle_ir"]
        assert (
            evaluator["treatment_transport_oracle_ir_file"]
            == row["treatment_transport_oracle_ir"]
        )
