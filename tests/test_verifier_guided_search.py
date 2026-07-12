from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verifier_guided_search import (  # noqa: E402
    Proposal,
    SearchBudget,
    run_search,
    validate_replay,
)
from strict_generator_semantic_ir_repair_apply import verifier_guided_search_receipts  # noqa: E402
from theseus_plan_compiler import verifier_guided_search_contract_for  # noqa: E402


def feedback(*, passed: bool, reward: float, fault: str = "") -> dict:
    return {
        "passed": passed,
        "verification_stage": "intended_behavior" if passed else "runtime_loaded",
        "verification_reward": reward,
        "fault_codes": [fault] if fault else [],
        "repair_scope": ["return.value"] if fault else [],
        "message_code": "pass" if passed else "wrong_answer",
        "evidence_hash": "a" * 64,
        "verifier_id": "private.visible.contract.v1",
    }


def no_repair(_proposal: Proposal, _result: dict):
    return []


def verified_integrity(proposal: Proposal) -> dict:
    import hashlib

    return {
        "independently_recomputed": True,
        "valid": True,
        "candidate_sha256": hashlib.sha256(proposal.code.encode()).hexdigest(),
        "family": "test_fixture",
        "fallback_or_template": False,
        "origin_independently_recomputed": True,
        "verified_origin": proposal.origin,
    }


_run_search = run_search


def run_search(*args, **kwargs):  # type: ignore[no-redef]
    kwargs.setdefault("integrity", verified_integrity)
    return _run_search(*args, **kwargs)


def test_one_shot_exact_is_model_only_and_replayable() -> None:
    receipt = run_search(
        [Proposal("return data", "model_one_shot", metadata={"learned": False, "fallback": True})],
        verify=lambda _proposal: feedback(passed=True, reward=1.0),
        repair=no_repair,
        task_ref_hash="task-a",
    )
    assert receipt["summary"]["model_one_shot_pass"] is True
    assert receipt["summary"]["learned_search_guided_pass"] is False
    assert receipt["summary"]["assisted_repair_pass"] is False
    assert receipt["claim_accounting"]["candidate_self_declared_flags_trusted"] is False
    assert validate_replay(receipt)["passed"] is True


def test_learned_repair_is_search_guided_not_one_shot() -> None:
    def verify(proposal: Proposal) -> dict:
        return feedback(passed="fixed" in proposal.code, reward=1.0 if "fixed" in proposal.code else 0.4, fault="wrong_answer")

    def repair(proposal: Proposal, result: dict):
        assert result["fault_codes"] == ["wrong_answer"]
        return [Proposal(proposal.code + " # fixed", "model_repair", model_receipt_hash="b" * 64)]

    receipt = run_search([Proposal("return data", "model_one_shot")], verify=verify, repair=repair)
    assert receipt["summary"]["model_one_shot_pass"] is False
    assert receipt["summary"]["learned_search_guided_pass"] is True
    assert receipt["summary"]["selected_depth"] == 1


def test_deterministic_repair_is_assisted_and_zero_credit() -> None:
    receipt = run_search(
        [Proposal("bad", "model_one_shot")],
        verify=lambda proposal: feedback(passed=proposal.code == "good", reward=1.0 if proposal.code == "good" else 0.1),
        repair=lambda _proposal, _result: [Proposal("good", "deterministic_repair")],
    )
    assert receipt["summary"]["assisted_repair_pass"] is True
    assert receipt["claim_accounting"]["model_only_search_guided"] is False
    assert receipt["claim_accounting"]["deterministic_repair_generation_credit"] == 0


@pytest.mark.parametrize("forbidden", ["tests", "hidden_tests", "solution", "expected"])
def test_forbidden_verifier_feedback_is_quarantined(forbidden: str) -> None:
    raw = feedback(passed=False, reward=0.0)
    raw[forbidden] = ["secret"]
    receipt = run_search(
        [Proposal("bad", "model_one_shot")],
        verify=lambda _proposal: raw,
        repair=no_repair,
    )
    assert receipt["summary"]["verified_exact_count"] == 0
    assert receipt["faults"][0]["fault_type"] == "FORBIDDEN_VERIFIER_FEEDBACK"
    assert any(row["state"] == "quarantined" for row in receipt["events"])


def test_unknown_feedback_field_fails_closed() -> None:
    raw = feedback(passed=False, reward=0.0)
    raw["stderr"] = "opaque"
    receipt = run_search([Proposal("bad", "model_one_shot")], verify=lambda _: raw, repair=no_repair)
    assert receipt["faults"][0]["fault_type"] == "UNKNOWN_VERIFIER_FEEDBACK_FIELD"


def test_integrity_must_be_independently_recomputed() -> None:
    receipt = run_search(
        [Proposal("bad", "model_one_shot")],
        verify=lambda _: feedback(passed=True, reward=1.0),
        repair=no_repair,
        integrity=lambda _: {"valid": True},
    )
    assert receipt["summary"]["verifier_call_count"] == 0
    assert receipt["faults"][0]["fault_type"] == "INTEGRITY_NOT_INDEPENDENT"


def test_duplicate_code_is_suppressed_across_origins() -> None:
    receipt = run_search(
        [Proposal("same", "model_one_shot"), Proposal("same", "tool_assisted")],
        verify=lambda _: feedback(passed=False, reward=0.0),
        repair=no_repair,
    )
    assert receipt["summary"]["proposal_count"] == 1
    assert any(row["state"] == "duplicate" for row in receipt["events"])


def test_verifier_budget_is_exact() -> None:
    receipt = run_search(
        [Proposal(f"candidate-{index}", "model_one_shot") for index in range(5)],
        verify=lambda _: feedback(passed=False, reward=0.0),
        repair=no_repair,
        budget=SearchBudget(max_verifier_calls=2, max_proposals=5),
    )
    assert receipt["summary"]["verifier_call_count"] == 2
    assert receipt["summary"]["stop_reason"] == "verifier_budget_exhausted"
    assert validate_replay(receipt)["budget_valid"] is True


def test_depth_budget_stops_recursive_repair() -> None:
    receipt = run_search(
        [Proposal("depth-0", "model_one_shot")],
        verify=lambda _: feedback(passed=False, reward=0.2),
        repair=lambda proposal, _: [Proposal(proposal.code + "+", "model_repair")],
        budget=SearchBudget(max_depth=1),
    )
    assert max(row["depth"] for row in receipt["events"]) == 1
    assert any(row["state"] == "depth_exhausted" for row in receipt["events"])


def test_wall_budget_stops_without_verifying() -> None:
    times = iter([0.0, 0.002, 0.002, 0.002])
    receipt = run_search(
        [Proposal("candidate", "model_one_shot")],
        verify=lambda _: feedback(passed=True, reward=1.0),
        repair=no_repair,
        budget=SearchBudget(max_wall_ms=1),
        monotonic=lambda: next(times),
    )
    assert receipt["summary"]["verifier_call_count"] == 0
    assert receipt["summary"]["stop_reason"] == "wall_budget_exhausted"


def test_tool_fault_is_typed_and_does_not_fabricate_result() -> None:
    def broken(_proposal: Proposal) -> dict:
        raise RuntimeError("tool down")

    receipt = run_search([Proposal("candidate", "model_one_shot")], verify=broken, repair=no_repair)
    assert receipt["selected"] == {}
    assert receipt["faults"][0]["fault_type"] == "VERIFIER_FAULT"


def test_replay_detects_tampering() -> None:
    receipt = run_search(
        [Proposal("return data", "model_one_shot")],
        verify=lambda _: feedback(passed=True, reward=1.0),
        repair=no_repair,
    )
    tampered = copy.deepcopy(receipt)
    tampered["events"][0]["origin"] = "tool_assisted"
    replay = validate_replay(tampered)
    assert replay["passed"] is False
    assert replay["event_digest_valid"] is False


def test_invalid_origin_is_a_typed_fault() -> None:
    receipt = run_search(
        [Proposal("candidate", "self_declared_learned")],
        verify=lambda _: feedback(passed=True, reward=1.0),
        repair=no_repair,
    )
    assert receipt["summary"]["proposal_count"] == 0
    assert receipt["faults"][0]["fault_type"] == "PROPOSAL_ORIGIN_INVALID"


def test_semantic_ir_runtime_adapter_replays_assisted_repair() -> None:
    source_code = "def solve(data):\n    return [item for item in data]\n"
    repaired_code = "def solve(data):\n    return list(data)\n"
    source_sha = __import__("hashlib").sha256(source_code.encode()).hexdigest()
    repaired_sha = __import__("hashlib").sha256(repaired_code.encode()).hexdigest()
    source = {
        "task_id": "private-visible-task",
        "code": source_code,
        "candidate_sha256": source_sha,
        "candidate_source": "mlx_transformer_hybrid",
        "candidate_generation_mode": "token_level_code_decoder",
    }
    repaired = {
        "task_id": "private-visible-task",
        "code": repaired_code,
        "candidate_sha256": repaired_sha,
        "semantic_ir_repair_apply": {
            "source_candidate_sha256": source_sha,
            "changed_atom_ids": ["atom.return.value"],
            "policy": "project_theseus_strict_generator_semantic_ir_repair_apply_v1",
            "candidate_generation_credit": 0,
            "learned_generation_claim_allowed": False,
        },
    }
    source_report = {
        "verification_attempt_labels": [
            {
                "code_sha256": source_sha,
                "passed": False,
                "intended_behavior_passed": False,
                "verification_stage": "runtime_loaded",
                "verification_reward": 0.6,
                "failure_class": "wrong_answer",
            }
        ]
    }
    repaired_report = {
        "verification_attempt_labels": [
            {
                "code_sha256": repaired_sha,
                "passed": True,
                "intended_behavior_passed": True,
                "verification_stage": "intended_behavior",
                "verification_reward": 1.0,
                "failure_class": "none",
            }
        ]
    }
    receipts = verifier_guided_search_receipts(
        [source],
        [repaired],
        source_report,
        repaired_report,
        execute=True,
    )
    assert len(receipts) == 1
    assert receipts[0]["summary"]["assisted_repair_pass"] is True
    assert receipts[0]["claim_accounting"]["deterministic_repair_generation_credit"] == 0
    assert receipts[0]["replay"]["passed"] is True


def test_plan_compiler_assigns_risk_bounded_search_contract() -> None:
    config = {
        "verifier_guided_search": {
            "policy": "test_search_policy",
            "eligible_ops": ["VERIFY"],
            "budgets_by_risk": {
                "medium": {"max_proposals": 8, "max_verifier_calls": 4, "max_depth": 1, "max_repair_branches": 2, "max_wall_ms": 1000},
                "high": {"max_proposals": 96, "max_verifier_calls": 48, "max_depth": 3, "max_repair_branches": 6, "max_wall_ms": 20000},
            },
            "feedback_allowlist": ["passed", "fault_codes"],
            "candidate_integrity_independent": True,
            "public_hidden_verifier_feedback_allowed": False,
            "fallback_returns_allowed": False,
        }
    }
    contract = verifier_guided_search_contract_for(
        node_id="goal.verify",
        atom={"op": "VERIFY", "risk_tier": "high", "required_capabilities": []},
        goal={"risk_tier": "medium"},
        config=config,
        route={"blocked": False},
    )
    assert contract["decision"] == "eligible_bounded"
    assert contract["budget"]["max_verifier_calls"] == 48
    assert contract["public_hidden_verifier_feedback_allowed"] is False
    assert contract["fallback_returns_allowed"] is False
    assert contract["candidate_integrity_independent"] is True


def test_plan_compiler_blocks_search_when_route_is_blocked() -> None:
    config = {
        "verifier_guided_search": {
            "eligible_ops": ["EXECUTE"],
            "budgets_by_risk": {"medium": {}},
            "candidate_integrity_independent": True,
            "public_hidden_verifier_feedback_allowed": False,
            "fallback_returns_allowed": False,
        }
    }
    contract = verifier_guided_search_contract_for(
        node_id="goal.execute",
        atom={"op": "EXECUTE"},
        goal={},
        config=config,
        route={"blocked": True, "block_reason": "authority lease missing"},
    )
    assert contract["decision"] == "blocked_by_route"
    assert contract["reason"] == "authority lease missing"
