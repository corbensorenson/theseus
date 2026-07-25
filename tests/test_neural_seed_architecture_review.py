from __future__ import annotations

import copy
import json
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import neural_seed_architecture_review as review  # noqa: E402
import moecot_language_arm_training as training  # noqa: E402


def test_first_rung_allocation_is_matched_and_weak_tail_balanced() -> None:
    config = json.loads((ROOT / "configs/neural_seed_architecture_review.json").read_text())
    allocation = config["candidate_budgets"]["100000000"]
    phases = config["component_phase_budgets"]["100000000"]
    assert review.validate_allocation(allocation, 100_000_000, phases) == []
    assert sum(allocation["moecot_system"].values()) == 100_000_000
    assert len({allocation["moecot_system"][arm] for arm in training.ARM_IDS}) == 1
    assert allocation["dense_active_parameter"] == {
        "dense_active_parameter": 100_000_000
    }
    assert allocation["dense_total_parameter"] == {
        "dense_total_parameter": 100_000_000
    }
    assert sum(row["total"] for row in phases.values()) == 300_000_000
    assert all(
        row["total"]
        == row["pretraining"]
        + row["source_conditioned_pretraining"]
        + row["supervision"]
        for row in phases.values()
    )


def test_allocation_rejects_hidden_extra_budget_and_weak_tail_mismatch() -> None:
    config = json.loads((ROOT / "configs/neural_seed_architecture_review.json").read_text())
    allocation = copy.deepcopy(config["candidate_budgets"]["100000000"])
    allocation["moecot_system"]["english"] += 1
    phases = config["component_phase_budgets"]["100000000"]
    gaps = review.validate_allocation(allocation, 100_000_000, phases)
    assert "moecot_total_position_budget_mismatch" in gaps
    assert "moecot_weak_tail_arm_budget_mismatch" in gaps


def test_review_plan_is_isolated_and_preserves_canonical_architecture() -> None:
    config_path = ROOT / "configs/neural_seed_architecture_review.json"
    config = json.loads(config_path.read_text())
    training_path = ROOT / config["training_config"]
    training_config = training.bind_scale_preregistration(
        json.loads(training_path.read_text())
    )
    canonical = training.build_plan(training_config, config_path=training_path)
    allocation = config["candidate_budgets"]["100000000"]
    phases = config["component_phase_budgets"]["100000000"]
    planned = review.build_review_plan(
        canonical, config, allocation, phases, 100_000_000
    )

    assert planned["canonical_plan_sha256"] == canonical["plan_sha256"]
    assert planned["plan_sha256"] != canonical["plan_sha256"]
    assert planned["models"] == canonical["models"]
    assert planned["stage"]["stage_signature"] == canonical["stage"]["stage_signature"]
    for target_id in review.COMPONENT_IDS:
        target = planned["targets"][target_id]
        assert target["review_only"] is True
        assert target["plan_sha256"] == planned["plan_sha256"]
        assert target["checkpoint"].startswith(
            "checkpoints/neural_seed_57m_architecture_review_v3/100000000/"
        )
        assert target["checkpoint"] != canonical["targets"][target_id]["checkpoint"]
        assert target["review_component_total_optimizer_positions"] == phases[target_id]["total"]
        assert target["optimizer_target_positions"] == phases[target_id]["pretraining"]


def test_review_surface_is_private_disjoint_and_generator_blind() -> None:
    config_path = ROOT / "configs/neural_seed_architecture_review.json"
    config = json.loads(config_path.read_text())
    contract = review.build_contract(config, config_path, 100_000_000)
    assert contract["source_disjoint_audit"]["hard_gaps"] == []
    assert len(contract["candidate_packet"]["rows"]) == 160
    assert all(
        set(row) == {"case_id", "arm_id", "prompt"}
        for row in contract["candidate_packet"]["rows"]
    )
    assert contract["boundaries"]["confirmation_surface_consumed"] is False
    assert contract["boundaries"]["public_surface_consumed"] is False
    assert contract["boundaries"]["candidate_outputs_training_eligible"] is False


def test_freeze_rejects_semantic_mutation() -> None:
    semantic = {
        "policy": "project_theseus_architecture_review_semantic_identity_v1",
        "plan_sha256": "a" * 64,
    }
    frozen = {
        "policy": review.FREEZE_POLICY,
        "immutable": True,
        "review_optimizer_positions": [100_000_000],
        "semantic_identity": semantic,
        "training_eligible": False,
    }
    assert review.validate_freeze(frozen, semantic, 100_000_000) == []
    changed = dict(semantic, plan_sha256="b" * 64)
    assert "review_semantic_identity_mismatch" in review.validate_freeze(
        frozen, changed, 100_000_000
    )


def test_next_action_respects_shared_trunk_dependency() -> None:
    rows = [
        {
            "target_id": target_id,
            "state": "NOT_STARTED",
            "optimizer_positions": 0,
            "target_optimizer_positions": 1,
            "faults": [],
        }
        for target_id in review.COMPONENT_IDS
    ]
    action = review.next_action(rows, {"complete": False, "rows": []})
    assert action["kind"] == "train_review_component"
    assert action["target_id"] == training.SHARED_TRUNK_ID


def test_parallel_verifier_preserves_case_order_and_bounds_tool_concurrency(
    monkeypatch,
) -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_verify(case, output, _config):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {
            "case_id": case["case_id"],
            "arm_id": case["arm_id"],
            "verifier_kind": case["verifier"]["kind"],
            "passed": output == "ok",
            "duration_ms": 20.0,
        }

    monkeypatch.setattr(review, "verify_candidate", fake_verify)
    cases = [
        {
            "case_id": f"case-{index}",
            "arm_id": "python",
            "verifier": {"kind": "python_isolated_tests_v1"},
        }
        for index in range(6)
    ]
    contract = {
        "config_payload": {
            "evaluation": {
                "verifier_parallelism": {
                    "maximum_workers": 4,
                    "per_verifier_maximum": {"python_isolated_tests_v1": 2},
                }
            }
        },
        "functional_config": {},
    }
    prepared = [
        {
            "candidate_id": "candidate",
            "cases": cases,
            "outputs": {row["case_id"]: "ok" for row in cases},
        }
    ]

    observed = review.verify_candidate_bundles(contract, prepared)

    assert maximum_active == 2
    assert [row["case_id"] for row in observed["rows"]["candidate"]] == [
        row["case_id"] for row in cases
    ]
    assert observed["case_count"] == 6
    assert observed["sum_case_duration_ms"] == 120.0
