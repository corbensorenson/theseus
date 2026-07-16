from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import reflexive_dispatch as reflex


def event(
    payload: str = "help me plan this",
    *,
    origin: str = "local_user_control",
    authenticated: bool = True,
    authorities: list[str] | None = None,
    principal: str = "local-user",
) -> dict:
    return {
        "event_id": f"event:{principal}:{reflex.digest(payload)}",
        "principal": principal,
        "authenticated": authenticated,
        "origin": origin,
        "received_at": "2026-07-16T20:00:00+00:00",
        "valid_time": "2026-07-16T20:00:00+00:00",
        "authority_refs": authorities if authorities is not None else ["local_assistant_read", "local_tool_read"],
        "context_handles": ["vcm://test/context"],
        "deadline_ms": 30000,
        "literal_payload": payload,
        "literal_payload_digest": reflex.digest(payload),
    }


class ReflexiveDispatchTests(unittest.TestCase):
    def test_contract_is_content_bound_and_has_no_parallel_authority(self) -> None:
        contract = reflex.load_contract()
        self.assertEqual(contract["architecture_scope"], "existing_assistant_scf_octopus_viea_vcm_procedural_owners_only")
        self.assertEqual(len(contract["source_evidence"]["paper_sha256"]), 64)
        self.assertIn("verification_failed", contract["terminal_outcomes"])
        self.assertTrue(all("shell" not in row for row in contract["user_commands"]))

    def test_command_text_from_untrusted_origins_is_literal_only(self) -> None:
        for origin in reflex.load_contract()["ingress"]["literal_only_origins"]:
            trace = reflex.dispatch(event("/tool inspect this", origin=origin), intent="chat")
            self.assertTrue(trace["ingress"]["command_shaped_literal_isolated"])
            self.assertEqual(trace["ingress"]["mode"], "automatic")
            self.assertEqual(trace["selection"]["terminal_outcome"], "prepared")
            selected = set(trace["selection"]["selected_proposal_ids"])
            self.assertTrue(any(row["capability_id"] == "assistant.chat_checkpoint" and row["proposal_id"] in selected for row in trace["proposals"]))

    def test_authenticated_direct_command_bypasses_inference_but_not_qualification(self) -> None:
        trace = reflex.dispatch(event("/tool"), intent="chat")
        self.assertEqual(trace["ingress"]["mode"], "direct_command")
        self.assertTrue(trace["ingress"]["inference_bypassed"])
        selected = set(trace["selection"]["selected_proposal_ids"])
        self.assertTrue(any(row["capability_id"] == "assistant.deterministic_tool" and row["proposal_id"] in selected for row in trace["proposals"]))
        self.assertFalse(trace["effect"]["effect_authority_granted"])

        denied = reflex.dispatch(event("/tool", authenticated=False), intent="chat")
        self.assertEqual(denied["selection"]["terminal_outcome"], "prepared")
        self.assertTrue(denied["ingress"]["command_shaped_literal_isolated"])
        self.assertFalse(denied["ingress"]["inference_bypassed"])

    def test_forced_route_fails_closed_without_implicit_fallback(self) -> None:
        denied = reflex.dispatch(
            event(authorities=["local_assistant_read"]),
            intent="chat",
            requested_route="assistant.deterministic_tool",
            fallback_policy="no_fallback",
        )
        self.assertEqual(denied["selection"]["terminal_outcome"], "unauthorized")
        self.assertFalse(denied["selection"]["fallback_used"])
        self.assertEqual(denied["plan_nodes"], [])
        self.assertEqual(denied["no_cheat"]["fallback_return_count"], 0)

        untrusted = reflex.dispatch(
            event(origin="model_output"),
            intent="chat",
            requested_route="assistant.chat_checkpoint",
        )
        self.assertEqual(untrusted["selection"]["terminal_outcome"], "unauthorized")

    def test_qualification_precedes_cost_and_stale_route_is_rejected(self) -> None:
        stale = reflex.dispatch(
            event(),
            intent="planning",
            route_health={"assistant.plan_dag": False},
        )
        self.assertEqual(stale["selection"]["terminal_outcome"], "unsupported")
        self.assertIn("implementation_stale_or_blocked", stale["qualification"][0]["failures"])

        trace = reflex.dispatch(
            event(),
            intent="chat",
            learned_proposals=[
                {"proposal_id": "cheap-ood", "capability_id": "assistant.deterministic_tool", "score": 0.99, "ood": True},
                {"proposal_id": "low-confidence", "capability_id": "assistant.plan_dag", "score": 0.2, "ood": False},
            ],
        )
        selected = set(trace["selection"]["selected_proposal_ids"])
        selected_rows = [row for row in trace["proposals"] if row["proposal_id"] in selected]
        self.assertEqual([row["capability_id"] for row in selected_rows], ["assistant.chat_checkpoint"])

    def test_registry_rejects_overlap_text_macros_and_unknown_capabilities(self) -> None:
        contract = reflex.load_contract()
        overlap = copy.deepcopy(contract)
        overlap["reflexes"].append(copy.deepcopy(overlap["reflexes"][0]))
        overlap["reflexes"][-1]["reflex_id"] = "different-id-same-match"
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_RULE_UNRESOLVED_OVERLAP"):
            reflex.validate_contract(overlap)

        macro = copy.deepcopy(contract)
        macro["user_commands"][0]["shell"] = "rm -rf /"
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_COMMAND_TEXT_MACRO_FORBIDDEN"):
            reflex.validate_contract(macro)

        unknown = copy.deepcopy(contract)
        unknown["user_commands"][0]["capability_id"] = "capability.unknown"
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_COMMAND_CAPABILITY_UNKNOWN"):
            reflex.validate_contract(unknown)

    def test_composite_dag_rejects_cycles_cost_bombs_and_unselected_capabilities(self) -> None:
        limits = reflex.load_contract()["resource_limits"]
        cycle = [
            {"node_id": "a", "capability_id": "assistant.plan_dag", "dependencies": ["b"], "effect_class": "none", "verifier_ref": "viea_plan_contract", "cost_units": 1, "retry_limit": 0, "completion_policy": "all_or_nothing"},
            {"node_id": "b", "capability_id": "assistant.plan_dag", "dependencies": ["a"], "effect_class": "none", "verifier_ref": "viea_plan_contract", "cost_units": 1, "retry_limit": 0, "completion_policy": "all_or_nothing"},
        ]
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_DAG_CYCLE"):
            reflex.validate_plan_dag(cycle, limits)

        cost_bomb = copy.deepcopy(cycle[:1])
        cost_bomb[0]["dependencies"] = []
        cost_bomb[0]["cost_units"] = limits["max_aggregate_cost_units"] + 1
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_DAG_COST_BUDGET_EXCEEDED"):
            reflex.validate_plan_dag(cost_bomb, limits)

        alien = copy.deepcopy(cost_bomb)
        alien[0].update({"capability_id": "assistant.code_candidate", "cost_units": 1, "verifier_ref": "private_code_verifier"})
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_DAG_CAPABILITY_NOT_SELECTED"):
            reflex.dispatch(event(), intent="planning", plan_nodes=alien)

    def test_registered_workflow_selects_and_orders_multiple_capabilities(self) -> None:
        trace = reflex.dispatch(event("/plan-tool"), intent="chat")
        selected = {
            capability_id
            for row in trace["proposals"]
            if row["proposal_id"] in trace["selection"]["selected_proposal_ids"]
            for capability_id in reflex.proposal_capability_ids(row)
        }
        self.assertEqual(trace["selection"]["kind"], "dag")
        self.assertEqual(trace["selection"]["terminal_outcome"], "prepared")
        self.assertEqual(selected, {"assistant.deterministic_tool", "assistant.plan_dag"})
        self.assertEqual([row["node_id"] for row in trace["plan_nodes"]], ["tool-evidence", "compile-plan"])
        self.assertEqual(trace["plan_nodes"][1]["dependencies"], ["tool-evidence"])
        self.assertEqual(trace["metrics"]["total_cost_units"], 30)
        self.assertEqual(reflex.verify_trace(trace)["state"], "VERIFIED")

    def test_workflow_registry_rejects_alias_collision_cycle_and_contract_mismatch(self) -> None:
        contract = reflex.load_contract()
        collision = copy.deepcopy(contract)
        collision["workflows"][0]["alias"] = "/tool"
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_WORKFLOW_NAMESPACE_CONFLICT"):
            reflex.validate_contract(collision)

        cycle = copy.deepcopy(contract)
        cycle["workflows"][0]["nodes"][0]["dependencies"] = ["compile-plan"]
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_DAG_CYCLE"):
            reflex.validate_contract(cycle)

        mismatch = copy.deepcopy(contract)
        mismatch["workflows"][0]["nodes"][0]["verifier_ref"] = "unregistered_verifier"
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_WORKFLOW_NODE_CONTRACT_MISMATCH"):
            reflex.validate_contract(mismatch)

    def test_effort_profiles_are_realized_and_fail_closed_at_the_budget_boundary(self) -> None:
        direct = reflex.dispatch(event(), intent="code", effort_profile="direct")
        balanced = reflex.dispatch(event(), intent="code", effort_profile="balanced")
        self.assertEqual(direct["effort"]["requested_profile"], "direct")
        self.assertTrue(direct["effort"]["profile_fidelity"])
        self.assertEqual(direct["selection"]["terminal_outcome"], "resource_exceeded")
        self.assertEqual(direct["plan_nodes"], [])
        self.assertEqual(balanced["selection"]["terminal_outcome"], "prepared")
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_EFFORT_PROFILE_UNKNOWN"):
            reflex.dispatch(event(), intent="chat", effort_profile="unbounded")

        tampered = copy.deepcopy(balanced)
        tampered["effort"]["realized_limits"]["max_nodes"] += 1
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_TRACE_DIGEST_INVALID"):
            reflex.verify_trace(tampered)

    def test_effect_capability_requires_explicit_authority_and_never_grants_it_from_proposal(self) -> None:
        denied = reflex.dispatch(
            event(authorities=["local_assistant_read", "local_tool_read"]),
            intent="chat",
            requested_route="assistant.route_authority_effect",
        )
        self.assertEqual(denied["selection"]["terminal_outcome"], "unauthorized")

        prepared = reflex.dispatch(
            event(authorities=["local_assistant_read", "local_tool_read", "local_effect_write"]),
            intent="chat",
            requested_route="assistant.route_authority_effect",
        )
        self.assertEqual(prepared["selection"]["terminal_outcome"], "prepared")
        self.assertTrue(prepared["effect"]["required"])
        self.assertEqual(prepared["effect"]["state"], "prepared")
        self.assertFalse(prepared["effect"]["effect_authority_granted"])
        self.assertFalse(prepared["proposals"][0]["proposal_grants_authority"])

    def test_composite_route_fails_when_any_member_is_stale(self) -> None:
        trace = reflex.dispatch(
            event("/plan-tool"),
            intent="chat",
            route_health={"assistant.deterministic_tool": True, "assistant.plan_dag": False},
        )
        self.assertEqual(trace["selection"]["terminal_outcome"], "unsupported")
        self.assertEqual(trace["plan_nodes"], [])
        self.assertEqual(
            trace["qualification"][0]["capability_failures"]["assistant.plan_dag"],
            ["implementation_stale_or_blocked"],
        )

    def test_malformed_learned_composite_is_typed_unsupported_not_an_exception(self) -> None:
        trace = reflex.dispatch(
            event(),
            intent="chat",
            learned_proposals=[
                {
                    "proposal_id": "malformed-composite",
                    "capability_ids": ["assistant.deterministic_tool", "assistant.plan_dag"],
                    "score": 0.99,
                    "composite": True,
                    "plan_nodes": [
                        {
                            "node_id": "alien",
                            "capability_id": "assistant.code_candidate",
                            "dependencies": [],
                            "effect_class": "none",
                            "verifier_ref": "private_code_verifier",
                            "cost_units": 1,
                            "retry_limit": 0,
                            "completion_policy": "all_or_nothing",
                        }
                    ],
                }
            ],
        )
        learned = next(row for row in trace["qualification"] if len(row["capability_ids"]) == 2)
        self.assertFalse(learned["qualified"])
        self.assertIn("composite_plan_invalid:REFLEX_DAG_CAPABILITY_NOT_SELECTED", learned["failures"])
        self.assertEqual(trace["selection"]["terminal_outcome"], "prepared")
        self.assertEqual(trace["no_cheat"]["learned_generation_credit"], 0)

    def test_chronicle_is_bitemporal_append_only_and_keeps_claims_separate(self) -> None:
        base = {
            "record_id": "chronicle:a",
            "record_type": "state",
            "valid_time": "2026-07-15T00:00:00Z",
            "transaction_time": "2026-07-16T00:00:00Z",
            "epistemic_state": "observed",
            "source_identity": "sensor:a",
            "payload_digest": reflex.digest("state-a"),
            "correction_of": "",
        }
        ledger = reflex.append_chronicle_record([], base)
        correction = {**base, "record_id": "chronicle:b", "transaction_time": "2026-07-16T01:00:00Z", "payload_digest": reflex.digest("state-b"), "correction_of": "chronicle:a"}
        ledger = reflex.append_chronicle_record(ledger, correction)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger[0]["payload_digest"], reflex.digest("state-a"))

        poisoned = {**base, "record_id": "chronicle:claim", "record_type": "claim", "epistemic_state": "observed"}
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_CHRONICLE_CLAIM_STATE_COLLAPSE"):
            reflex.append_chronicle_record(ledger, poisoned)

    def test_cache_identity_closes_over_authority_privacy_versions_and_freshness(self) -> None:
        kwargs = {
            "task_semantics": "retrieve current plan",
            "principal": "local-user",
            "tenant": "private",
            "authority_refs": ["local_assistant_read"],
            "entity_scope": ["project:theseus"],
            "time_scope": "current",
            "privacy_view": "owner",
            "source_versions": {"roadmap": "sha256:a"},
            "schema_versions": {"chronicle": "v1"},
            "capability_versions": {"assistant.plan_dag": "v1"},
            "model_version": "model:v1",
            "policy_version": "policy:v1",
            "freshness_epoch": "epoch:1",
        }
        cache = reflex.cache_identity(**kwargs)
        self.assertFalse(reflex.invalidate_cache(cache, {"policy_version": "policy:v1"})["invalidated"])
        invalidated = reflex.invalidate_cache(cache, {"policy_version": "policy:v2"})
        self.assertTrue(invalidated["invalidated"])
        self.assertTrue(invalidated["descendants_must_invalidate"])
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_CACHE_IDENTITY_INCOMPLETE"):
            reflex.cache_identity(**{**kwargs, "privacy_view": ""})

    def test_compilation_requires_diverse_verified_traces_negative_space_and_economics(self) -> None:
        traces = []
        for principal in ("user-a", "user-b", "user-a"):
            trace = reflex.dispatch(event(principal=principal), intent="planning")
            trace["result"]["verification_state"] = "passed"
            traces.append(trace)
        candidate = reflex.compile_reflex_candidate(
            traces,
            negative_case_refs=["negative:a", "negative:b"],
            differential_passed=True,
            shadow_passed=True,
            canary_passed=True,
            expected_reuse_value=20.0,
            lifecycle_cost=5.0,
        )
        self.assertEqual(candidate["state"], "qualified")
        retired = reflex.decompile_reflex(candidate, reason="dependency drift", changed_dependencies=["policy:v2"])
        self.assertEqual(retired["state"], "decompiled")
        self.assertTrue(retired["descendants_invalidated"])

        premature = reflex.compile_reflex_candidate(
            traces[:1],
            negative_case_refs=[],
            differential_passed=False,
            shadow_passed=False,
            canary_passed=False,
            expected_reuse_value=1.0,
            lifecycle_cost=5.0,
        )
        self.assertEqual(premature["state"], "quarantined")
        self.assertIn("verified_trace_floor", premature["rejection_reasons"])

    def test_trace_integrity_and_terminal_state_cannot_be_rewritten(self) -> None:
        trace = reflex.dispatch(event(), intent="planning")
        verified = reflex.verify_trace(trace)
        self.assertEqual(verified["state"], "VERIFIED")
        self.assertFalse(verified["effect_authority_granted"])
        tampered = copy.deepcopy(trace)
        tampered["selection"]["terminal_outcome"] = "resolved"
        with self.assertRaisesRegex(reflex.ReflexiveDispatchFault, "REFLEX_TRACE_DIGEST_INVALID"):
            reflex.verify_trace(tampered)


if __name__ == "__main__":
    unittest.main()
