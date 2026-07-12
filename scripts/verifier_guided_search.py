#!/usr/bin/env python3
"""Bounded verifier-guided candidate search with strict claim accounting."""

from __future__ import annotations

import hashlib
import heapq
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable


POLICY = "project_theseus_verifier_guided_search_v1"
ORIGINS = {
    "model_one_shot",
    "model_repair",
    "deterministic_repair",
    "tool_assisted",
}
LEARNED_ORIGINS = {"model_one_shot", "model_repair"}
ASSISTED_ORIGINS = {"deterministic_repair", "tool_assisted"}
TERMINAL_STATES = {
    "verified_exact",
    "quarantined",
    "duplicate",
    "depth_exhausted",
    "budget_exhausted",
}
ALLOWED_FEEDBACK_KEYS = {
    "passed",
    "verification_stage",
    "verification_reward",
    "fault_codes",
    "repair_scope",
    "message_code",
    "evidence_hash",
    "verifier_id",
}
FORBIDDEN_FEEDBACK_TOKENS = {
    "answer",
    "category",
    "expected",
    "hidden_test",
    "hidden_tests",
    "solution",
    "solution_body",
    "solution_expr",
    "source_task_id",
    "test",
    "tests",
    "traceback",
}


class SearchContractFault(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail

    def record(self) -> dict[str, Any]:
        return {
            "fault_type": self.code,
            "detail": self.detail,
            "failure_behavior": "quarantine_without_fallback",
        }


@dataclass(frozen=True)
class SearchBudget:
    max_proposals: int = 32
    max_verifier_calls: int = 16
    max_depth: int = 2
    max_repair_branches: int = 4
    max_wall_ms: int = 5_000
    stop_on_first_exact: bool = True

    def normalized(self) -> "SearchBudget":
        values = {
            "max_proposals": max(1, int(self.max_proposals)),
            "max_verifier_calls": max(1, int(self.max_verifier_calls)),
            "max_depth": max(0, int(self.max_depth)),
            "max_repair_branches": max(0, int(self.max_repair_branches)),
            "max_wall_ms": max(1, int(self.max_wall_ms)),
            "stop_on_first_exact": bool(self.stop_on_first_exact),
        }
        return SearchBudget(**values)


@dataclass(frozen=True)
class Proposal:
    code: str
    origin: str
    proposal_id: str = ""
    parent_id: str = ""
    model_receipt_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


VerifyFn = Callable[[Proposal], dict[str, Any]]
RepairFn = Callable[[Proposal, dict[str, Any]], Iterable[Proposal]]
IntegrityFn = Callable[[Proposal], dict[str, Any]]


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_hash(code: str) -> str:
    return hashlib.sha256(str(code).encode("utf-8")).hexdigest()


def sanitize_verifier_feedback(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SearchContractFault("VERIFIER_FEEDBACK_NOT_OBJECT", type(raw).__name__)
    forbidden = sorted(
        str(key)
        for key in raw
        if str(key).lower() in FORBIDDEN_FEEDBACK_TOKENS
        or any(token in str(key).lower() for token in ("solution", "hidden_test", "expected_answer"))
    )
    unknown = sorted(str(key) for key in raw if str(key) not in ALLOWED_FEEDBACK_KEYS)
    if forbidden:
        raise SearchContractFault("FORBIDDEN_VERIFIER_FEEDBACK", ",".join(forbidden))
    if unknown:
        raise SearchContractFault("UNKNOWN_VERIFIER_FEEDBACK_FIELD", ",".join(unknown))

    fault_codes = raw.get("fault_codes") or []
    repair_scope = raw.get("repair_scope") or []
    if not isinstance(fault_codes, list) or not all(isinstance(item, str) for item in fault_codes):
        raise SearchContractFault("FAULT_CODES_NOT_STRING_LIST", type(fault_codes).__name__)
    if not isinstance(repair_scope, list) or not all(isinstance(item, str) for item in repair_scope):
        raise SearchContractFault("REPAIR_SCOPE_NOT_STRING_LIST", type(repair_scope).__name__)
    reward = raw.get("verification_reward", 0.0)
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise SearchContractFault("VERIFICATION_REWARD_NOT_NUMERIC", type(reward).__name__)
    reward_value = float(reward)
    if not 0.0 <= reward_value <= 1.0:
        raise SearchContractFault("VERIFICATION_REWARD_OUT_OF_RANGE", str(reward_value))
    sanitized = {
        "passed": bool(raw.get("passed")),
        "verification_stage": str(raw.get("verification_stage") or "unknown"),
        "verification_reward": reward_value,
        "fault_codes": sorted(set(fault_codes)),
        "repair_scope": sorted(set(repair_scope)),
        "message_code": str(raw.get("message_code") or ""),
        "evidence_hash": str(raw.get("evidence_hash") or ""),
        "verifier_id": str(raw.get("verifier_id") or ""),
    }
    serialized = json.dumps(sanitized, sort_keys=True).lower()
    if any(f'"{token}"' in serialized for token in FORBIDDEN_FEEDBACK_TOKENS):
        raise SearchContractFault("FORBIDDEN_VERIFIER_FEEDBACK_VALUE", "reserved token")
    return sanitized


def normalize_proposal(
    proposal: Proposal,
    *,
    parent_id: str = "",
) -> Proposal:
    if not isinstance(proposal, Proposal):
        raise SearchContractFault("PROPOSAL_TYPE_INVALID", type(proposal).__name__)
    origin = str(proposal.origin or "")
    if origin not in ORIGINS:
        raise SearchContractFault("PROPOSAL_ORIGIN_INVALID", origin)
    code = str(proposal.code or "")
    if not code.strip():
        raise SearchContractFault("PROPOSAL_CODE_EMPTY", origin)
    assigned_parent = str(parent_id or proposal.parent_id or "")
    identity = stable_hash(
        {
            "origin": origin,
            "code_sha256": code_hash(code),
            "parent_id": assigned_parent,
            "model_receipt_hash": str(proposal.model_receipt_hash or ""),
        }
    )
    return Proposal(
        code=code,
        origin=origin,
        proposal_id=f"proposal.{identity[:24]}",
        parent_id=assigned_parent,
        model_receipt_hash=str(proposal.model_receipt_hash or ""),
        metadata={},  # Candidate-emitted claim flags never cross the search boundary.
    )


def default_integrity(proposal: Proposal) -> dict[str, Any]:
    return {
        "independently_recomputed": True,
        "valid": bool(proposal.code.strip()),
        "candidate_sha256": code_hash(proposal.code),
        "family": "unclassified",
        "fallback_or_template": False,
        "origin_independently_recomputed": False,
        "verified_origin": "",
    }


def run_search(
    initial_proposals: Iterable[Proposal],
    *,
    verify: VerifyFn,
    repair: RepairFn,
    integrity: IntegrityFn = default_integrity,
    budget: SearchBudget | None = None,
    task_ref_hash: str = "",
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    policy_budget = (budget or SearchBudget()).normalized()
    started = monotonic()
    frontier: list[tuple[tuple[float, int, str], int, Proposal, int]] = []
    event_rows: list[dict[str, Any]] = []
    faults: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    seen_code_hashes: set[str] = set()
    proposal_count = 0
    verifier_calls = 0
    repair_calls = 0
    sequence = 0

    def elapsed_ms() -> int:
        return max(0, int((monotonic() - started) * 1000))

    def event(proposal: Proposal, depth: int, state: str, **extra: Any) -> None:
        row = {
            "sequence": len(event_rows),
            "proposal_id": proposal.proposal_id,
            "parent_id": proposal.parent_id,
            "origin": proposal.origin,
            "depth": depth,
            "state": state,
            "candidate_sha256": code_hash(proposal.code),
            **extra,
        }
        event_rows.append(row)

    def enqueue(raw: Proposal, depth: int, parent_id: str = "", priority: float = 0.0) -> None:
        nonlocal proposal_count, sequence
        if proposal_count >= policy_budget.max_proposals:
            return
        try:
            proposal = normalize_proposal(raw, parent_id=parent_id)
        except SearchContractFault as exc:
            faults.append(exc.record())
            return
        digest = code_hash(proposal.code)
        if digest in seen_code_hashes:
            event(proposal, depth, "duplicate")
            return
        seen_code_hashes.add(digest)
        proposal_count += 1
        sequence += 1
        # Higher verifier-derived repair priority is popped first; shallower nodes win ties.
        heapq.heappush(frontier, ((-float(priority), depth, proposal.proposal_id), sequence, proposal, depth))
        event(proposal, depth, "generated")

    for initial in initial_proposals:
        enqueue(initial, 0)

    stop_reason = "frontier_exhausted"
    while frontier:
        if elapsed_ms() >= policy_budget.max_wall_ms:
            stop_reason = "wall_budget_exhausted"
            break
        if verifier_calls >= policy_budget.max_verifier_calls:
            stop_reason = "verifier_budget_exhausted"
            break
        _priority, _sequence, proposal, depth = heapq.heappop(frontier)
        try:
            integrity_result = integrity(proposal)
            if not isinstance(integrity_result, dict) or integrity_result.get("independently_recomputed") is not True:
                raise SearchContractFault("INTEGRITY_NOT_INDEPENDENT", proposal.proposal_id)
            if integrity_result.get("valid") is not True:
                event(proposal, depth, "quarantined", fault_type="candidate_integrity_rejected")
                continue
            if str(integrity_result.get("candidate_sha256") or "") != code_hash(proposal.code):
                raise SearchContractFault("INTEGRITY_DIGEST_MISMATCH", proposal.proposal_id)
            if integrity_result.get("origin_independently_recomputed") is not True:
                raise SearchContractFault("ORIGIN_NOT_INDEPENDENTLY_RECOMPUTED", proposal.proposal_id)
            if str(integrity_result.get("verified_origin") or "") != proposal.origin:
                raise SearchContractFault(
                    "ORIGIN_CLASS_MISMATCH",
                    f"claimed={proposal.origin},verified={integrity_result.get('verified_origin')}",
                )
        except Exception as exc:  # The boundary converts tool failures into typed search faults.
            fault = exc.record() if isinstance(exc, SearchContractFault) else SearchContractFault("INTEGRITY_FAULT", type(exc).__name__).record()
            faults.append(fault)
            event(proposal, depth, "quarantined", fault_type=fault["fault_type"])
            continue

        verifier_calls += 1
        try:
            raw_feedback = verify(proposal)
            feedback = sanitize_verifier_feedback(raw_feedback)
        except Exception as exc:
            fault = exc.record() if isinstance(exc, SearchContractFault) else SearchContractFault("VERIFIER_FAULT", type(exc).__name__).record()
            faults.append(fault)
            event(proposal, depth, "quarantined", fault_type=fault["fault_type"])
            continue

        state = "verified_exact" if feedback["passed"] else "verified_lossy"
        event(proposal, depth, state, verifier_feedback=feedback)
        if feedback["passed"]:
            winner = {
                "proposal_id": proposal.proposal_id,
                "origin": proposal.origin,
                "depth": depth,
                "candidate_sha256": code_hash(proposal.code),
                "verification": feedback,
                "code": proposal.code,
            }
            winners.append(winner)
            if policy_budget.stop_on_first_exact:
                stop_reason = "verified_exact_found"
                break
            continue
        if depth >= policy_budget.max_depth:
            event(proposal, depth, "depth_exhausted")
            continue
        if policy_budget.max_repair_branches <= 0:
            continue
        try:
            repair_calls += 1
            repairs = list(repair(proposal, feedback))[: policy_budget.max_repair_branches]
            for repaired in repairs:
                enqueue(
                    repaired,
                    depth + 1,
                    parent_id=proposal.proposal_id,
                    priority=float(feedback["verification_reward"]),
                )
        except Exception as exc:
            fault = exc.record() if isinstance(exc, SearchContractFault) else SearchContractFault("REPAIR_FAULT", type(exc).__name__).record()
            faults.append(fault)
            event(proposal, depth, "quarantined", fault_type=fault["fault_type"])

    if frontier and stop_reason.endswith("exhausted"):
        for _priority, _sequence, proposal, depth in frontier:
            event(proposal, depth, "budget_exhausted", stop_reason=stop_reason)

    selected = sorted(
        winners,
        key=lambda row: (
            -float(row["verification"].get("verification_reward") or 0.0),
            int(row["depth"]),
            str(row["proposal_id"]),
        ),
    )[0] if winners else {}
    runtime_ms = elapsed_ms()
    event_digest = stable_hash(event_rows)
    model_one_shot_pass = bool(selected and selected.get("origin") == "model_one_shot")
    learned_search_pass = bool(selected and selected.get("origin") == "model_repair")
    assisted_pass = bool(selected and selected.get("origin") in ASSISTED_ORIGINS)
    receipt = {
        "policy": POLICY,
        "task_ref_hash": str(task_ref_hash or ""),
        "budget": asdict(policy_budget),
        "summary": {
            "proposal_count": proposal_count,
            "unique_candidate_count": len(seen_code_hashes),
            "verifier_call_count": verifier_calls,
            "repair_call_count": repair_calls,
            "verified_exact_count": len(winners),
            "model_one_shot_pass": model_one_shot_pass,
            "learned_search_guided_pass": learned_search_pass,
            "assisted_repair_pass": assisted_pass,
            "selected_origin": str(selected.get("origin") or ""),
            "selected_depth": selected.get("depth"),
            "fault_count": len(faults),
            "stop_reason": stop_reason,
            "runtime_ms": runtime_ms,
            "useful_solution_per_second": round(bool(selected) / max(runtime_ms / 1000.0, 1e-9), 6),
            "fallback_return_count": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
        },
        "selected": selected,
        "events": event_rows,
        "faults": faults,
        "event_digest": event_digest,
        "claim_accounting": {
            "model_only_one_shot": model_one_shot_pass,
            "model_only_search_guided": learned_search_pass,
            "deterministic_or_tool_assisted": assisted_pass,
            "deterministic_repair_generation_credit": 0,
            "tool_assisted_generation_credit": 0,
            "candidate_self_declared_flags_trusted": False,
        },
        "boundaries": {
            "verifier_feedback_allowlist": sorted(ALLOWED_FEEDBACK_KEYS),
            "forbidden_feedback_tokens": sorted(FORBIDDEN_FEEDBACK_TOKENS),
            "hidden_evaluation_tests_may_not_be_used_for_model_only_or_public_claims": True,
            "public_benchmark_search_requires_visible_or_independent_nonhidden_verifier": True,
        },
    }
    receipt["replay_hash"] = stable_hash(replay_payload(receipt))
    return receipt


def replay_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": receipt.get("policy"),
        "task_ref_hash": receipt.get("task_ref_hash"),
        "budget": receipt.get("budget"),
        "summary": {
            key: value
            for key, value in dict(receipt.get("summary") or {}).items()
            if key not in {"runtime_ms", "useful_solution_per_second"}
        },
        "selected": receipt.get("selected"),
        "events": receipt.get("events"),
        "faults": receipt.get("faults"),
        "event_digest": receipt.get("event_digest"),
        "claim_accounting": receipt.get("claim_accounting"),
        "boundaries": receipt.get("boundaries"),
    }


def validate_replay(receipt: dict[str, Any]) -> dict[str, Any]:
    events = list(receipt.get("events") or [])
    sequence_valid = [int(row.get("sequence", -1)) for row in events] == list(range(len(events)))
    digest_valid = str(receipt.get("event_digest") or "") == stable_hash(events)
    replay_valid = str(receipt.get("replay_hash") or "") == stable_hash(replay_payload(receipt))
    budget = dict(receipt.get("budget") or {})
    summary = dict(receipt.get("summary") or {})
    budget_valid = (
        int(summary.get("proposal_count") or 0) <= int(budget.get("max_proposals") or 0)
        and int(summary.get("verifier_call_count") or 0) <= int(budget.get("max_verifier_calls") or 0)
    )
    passed = sequence_valid and digest_valid and replay_valid and budget_valid
    return {
        "policy": "project_theseus_verifier_guided_search_replay_v1",
        "passed": passed,
        "sequence_valid": sequence_valid,
        "event_digest_valid": digest_valid,
        "replay_hash_valid": replay_valid,
        "budget_valid": budget_valid,
        "candidate_generation_credit": 0,
    }
