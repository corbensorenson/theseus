from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import roadmap_implementation_gate as gate  # noqa: E402


def matrix(required_status: str = "qualified") -> dict:
    return {
        "phases": [
            {"phase": 0, "title": "Registry", "status": required_status, "missing_items": [], "required_gates": ["gate"], "current_evidence": ["evidence"], "integration_smoke": ["smoke"]},
            {"phase": 10, "title": "Training", "status": "partial", "missing_items": ["learn behavior"]},
            {"phase": 9, "title": "Peers", "status": "frozen", "missing_items": ["external peer not reachable"]},
        ],
        "pre_training_architecture_contract": {
            "required_phase_ids": [0],
            "training_or_behavior_qualification_phase_ids": [10],
            "external_environment_phase_ids": [9],
        },
        "claim_support_ladder": [],
        "book_reference_core_before_training": {"required_slices": []},
        "out_of_scope_now": [
            "public_benchmark_training",
            "serve_external_inference",
            "count_router_as_learned_generation",
            "count_template_as_learned_generation",
            "long_training_as_implementation_proof",
            "training_score_chase_before_book_reference_core",
            "capability_claim_from_assisted_or_tool_output",
        ],
    }


class PreTrainingArchitectureGateTests(unittest.TestCase):
    def test_deferred_kerc_campaign_exclusion_is_machine_checked(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "campaign.json"
            config = {
                "comparison_contract": {
                    "first_campaign_candidate_ids": ["english", "python"]
                },
                "training": {"kernel_english_optimizer_repetitions": 0},
                "kernel_english_training": {
                    "required": False,
                    "disposition": {
                        "state": "DEFERRED_FROM_FIRST_LONG_RUN",
                        "full_kerc_training_enabled": False,
                        "first_campaign_topology_exposure": 0,
                        "first_campaign_optimizer_repetitions": 0,
                    },
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            binding = {
                "first_practical_campaign_exclusion": {
                    "policy": "project_theseus_kerc_first_campaign_exclusion_v1",
                    "config": str(config_path),
                    "required_disposition_state": "DEFERRED_FROM_FIRST_LONG_RUN",
                    "forbidden_candidate_ids": ["kernel_english", "kerc"],
                    "required_zero_fields": [
                        "kernel_english_training.disposition.first_campaign_topology_exposure",
                        "kernel_english_training.disposition.first_campaign_optimizer_repetitions",
                        "training.kernel_english_optimizer_repetitions",
                    ],
                }
            }
            report = gate.audit_kerc_first_campaign_exclusion(binding)
            self.assertTrue(report["ready"])
            self.assertEqual(report["faults"], [])

            config["comparison_contract"]["first_campaign_candidate_ids"].append(
                "kernel_english"
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")
            exposed = gate.audit_kerc_first_campaign_exclusion(binding)
            self.assertFalse(exposed["ready"])
            self.assertIn(
                "kerc_candidate_exposed:kernel_english", exposed["faults"]
            )

    def test_training_phase_does_not_circularly_block_architecture(self) -> None:
        report = gate.audit_pre_training_architecture_readiness(
            matrix=matrix(),
            phase_reports=[],
            book_contract_report={},
            current_hard_gap_count=0,
        )

        self.assertTrue(report["ready"])
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["deferred_unfinished_phases"][0]["phase"], 10)

    def test_unfinished_architecture_phase_still_blocks(self) -> None:
        report = gate.audit_pre_training_architecture_readiness(
            matrix=matrix(required_status="partial"),
            phase_reports=[],
            book_contract_report={},
            current_hard_gap_count=0,
        )

        self.assertFalse(report["ready"])
        self.assertEqual(report["blockers"][0]["kind"], "unfinished_architecture_prerequisite_phases")

    def test_implemented_or_wired_is_not_qualified(self) -> None:
        for status in ("implemented", "wired"):
            report = gate.audit_pre_training_architecture_readiness(
                matrix=matrix(required_status=status),
                phase_reports=[],
                book_contract_report={},
                current_hard_gap_count=0,
            )

            self.assertFalse(report["ready"], status)
            self.assertTrue(
                any(row["kind"] == "unfinished_architecture_prerequisite_phases" for row in report["blockers"]),
                status,
            )

    def test_kerc_mandatory_replacement_ladder_blocks_until_complete(self) -> None:
        import hashlib
        import tempfile

        binding = {
            "mandatory_replacement_qualification": {
                "policy": "project_theseus_kerc_mandatory_replacement_qualification_v1",
                "state": "ACTIVE_BLOCKING",
                "required_ladder": list(gate.REQUIRED_KERC_REPLACEMENT_LADDER),
                "common_acceptance": list(gate.REQUIRED_KERC_COMMON_ACCEPTANCE),
                "acceptance_by_ladder": {
                    step: list(gate.REQUIRED_KERC_REPLACEMENT_ACCEPTANCE[step])
                    for step in gate.REQUIRED_KERC_REPLACEMENT_LADDER
                },
                "completed_ladder": [],
                "evidence_by_ladder": {},
                "resource_rule": "RESOURCE_DEFERRED_ON_THIS_HOST is diagnostic evidence, not an exit state; optimize or redesign the implementation.",
            }
        }
        active = gate.audit_kerc_mandatory_replacement_qualification(binding)
        self.assertFalse(active["ready"])
        self.assertEqual(active["remaining_ladder"], list(gate.REQUIRED_KERC_REPLACEMENT_LADDER))
        strengthened_paths = {
            step: {
                predicate["path"]
                for predicate in gate.REQUIRED_KERC_REPLACEMENT_ACCEPTANCE[step]
            }
            for step in gate.REQUIRED_KERC_REPLACEMENT_LADDER
        }
        self.assertIn(
            "proposal.denominator_complete",
            strengthened_paths[gate.REQUIRED_KERC_REPLACEMENT_LADDER[3]],
        )
        self.assertIn(
            "lifecycle.contraction_out_of_envelope_expands",
            strengthened_paths[gate.REQUIRED_KERC_REPLACEMENT_LADDER[4]],
        )
        self.assertIn(
            "order_route_regret_reported",
            strengthened_paths[gate.REQUIRED_KERC_REPLACEMENT_LADDER[5]],
        )
        self.assertIn(
            "matched.lower_order_rescue",
            strengthened_paths[gate.REQUIRED_KERC_REPLACEMENT_LADDER[6]],
        )
        self.assertIn(
            "decomposed_objective_predecessor_bound",
            strengthened_paths[gate.REQUIRED_KERC_REPLACEMENT_LADDER[1]],
        )
        self.assertIn(
            "objective_gradient_decomposition",
            strengthened_paths[gate.REQUIRED_KERC_REPLACEMENT_LADDER[1]],
        )
        self.assertIn(
            "memory_execution_policy.token_loss_position_chunk_size",
            strengthened_paths[gate.REQUIRED_KERC_REPLACEMENT_LADDER[1]],
        )

        reordered = copy.deepcopy(binding)
        reordered_contract = reordered["mandatory_replacement_qualification"]
        reordered_contract["completed_ladder"] = [gate.REQUIRED_KERC_REPLACEMENT_LADDER[2]]
        reordered_contract["evidence_by_ladder"] = {
            gate.REQUIRED_KERC_REPLACEMENT_LADDER[2]: {}
        }
        reordered_audit = gate.audit_kerc_mandatory_replacement_qualification(reordered)
        self.assertIn("completed_ladder_not_ordered_prefix", reordered_audit["faults"])

        weakened = copy.deepcopy(binding)
        weakened_contract = weakened["mandatory_replacement_qualification"]
        weakened_contract["acceptance_by_ladder"][gate.REQUIRED_KERC_REPLACEMENT_LADDER[0]].pop()
        weakened_audit = gate.audit_kerc_mandatory_replacement_qualification(weakened)
        self.assertIn(
            "acceptance_contract_missing_or_drifted:"
            + gate.REQUIRED_KERC_REPLACEMENT_LADDER[0],
            weakened_audit["faults"],
        )

        scoped = copy.deepcopy(binding)
        scoped_contract = scoped["mandatory_replacement_qualification"]
        scoped_contract["state"] = (
            "FIRST_CAMPAIGN_SCOPE_EXCLUDED_INCONCLUSIVE_EXPERIMENT"
        )
        scoped_contract["completed_ladder"] = []
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "kerc_final_exposure.json"
            report_path.write_text(
                json.dumps(
                    {
                        "policy": "project_theseus_kerc_k5_stage_learnability_probe_v1",
                        "trigger_state": "GREEN",
                        "qualification_state": "LEARNABILITY_SANITY_FAILED",
                        "capability_claim": "NONE_TRAINING_ROW_OVERFIT_DIAGNOSTIC_ONLY",
                        "exact_match_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            scoped_contract["first_campaign_scope_disposition"] = {
                "policy": "project_theseus_kerc_first_campaign_scope_disposition_v1",
                "classification": "INCONCLUSIVE_EXPERIMENT",
                "scientific_falsification_claimed": False,
                "incomplete_ladder_preserved": True,
                "first_campaign_optimizer_exposure": 0,
                "exact_scope": "first matched 57M campaign only",
                "reentry_condition": "reopen the preserved K4-K8 ladder prospectively",
                "evidence": {
                    "path": str(report_path),
                    "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                },
            }
            scoped_ready = gate.audit_kerc_mandatory_replacement_qualification(scoped)
            self.assertTrue(scoped_ready["ready"])
            self.assertEqual(
                scoped_ready["remaining_ladder"],
                list(gate.REQUIRED_KERC_REPLACEMENT_LADDER),
            )

            scoped_contract["first_campaign_scope_disposition"][
                "scientific_falsification_claimed"
            ] = True
            scoped_invalid = gate.audit_kerc_mandatory_replacement_qualification(scoped)
            self.assertFalse(scoped_invalid["ready"])
            self.assertIn(
                "scope_must_not_claim_scientific_falsification",
                scoped_invalid["faults"],
            )

        contract = binding["mandatory_replacement_qualification"]
        contract["state"] = "QUALIFIED_NOT_SELECTED"
        contract["completed_ladder"] = list(gate.REQUIRED_KERC_REPLACEMENT_LADDER)
        terminal_without_receipts = gate.audit_kerc_mandatory_replacement_qualification(binding)
        self.assertFalse(terminal_without_receipts["ready"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.txt"
            source.write_text("bound", encoding="utf-8")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            evidence_by_ladder = {}
            for step in gate.REQUIRED_KERC_REPLACEMENT_LADDER:
                policy = f"{step}_receipt_v1"
                report_path = root / f"{step}.json"
                report = {
                    "policy": policy,
                    "trigger_state": "GREEN",
                    "source_artifacts": {
                        "source": {"path": str(source), "sha256": source_sha}
                    },
                }
                predicates = (
                    gate.REQUIRED_KERC_COMMON_ACCEPTANCE
                    + gate.REQUIRED_KERC_REPLACEMENT_ACCEPTANCE[step]
                )
                for predicate in predicates:
                    parts = predicate["path"].split(".")
                    owner = report
                    for part in parts[:-1]:
                        owner = owner.setdefault(part, {})
                    operator = predicate["operator"]
                    expected = predicate["value"]
                    if operator == "gt":
                        actual = expected + 1
                    elif operator == "lt":
                        actual = expected - 1
                    elif operator in {"gte", "lte", "equals"}:
                        actual = expected
                    elif operator == "in":
                        actual = (
                            contract["state"]
                            if predicate["path"] == "disposition"
                            else expected[0]
                        )
                    else:
                        self.fail(f"unhandled test predicate operator: {operator}")
                    owner[parts[-1]] = actual
                report_path.write_text(
                    json.dumps(report),
                    encoding="utf-8",
                )
                evidence_by_ladder[step] = {
                    "path": str(report_path),
                    "policy": policy,
                    "required_trigger_state": "GREEN",
                }
            contract["evidence_by_ladder"] = evidence_by_ladder
            complete = gate.audit_kerc_mandatory_replacement_qualification(binding)
            self.assertTrue(complete["ready"])
            self.assertEqual(complete["faults"], [])

    def test_phase_partition_must_cover_every_phase_once(self) -> None:
        payload = matrix()
        payload["pre_training_architecture_contract"]["training_or_behavior_qualification_phase_ids"] = []
        report = gate.audit_pre_training_architecture_readiness(
            matrix=payload,
            phase_reports=[],
            book_contract_report={},
            current_hard_gap_count=0,
        )

        self.assertFalse(report["ready"])
        partition = next(row for row in report["blockers"] if row["kind"].endswith("partition_invalid"))
        self.assertEqual(partition["missing_phase_ids"], [10])

    def test_required_cross_phase_backlog_blocks_until_pretraining_boundary_is_wired(self) -> None:
        payload = matrix()
        payload["pre_training_architecture_contract"].update(
            {
                "required_backlog_ids": ["planned.kernel_v1"],
                "ready_backlog_statuses": ["pretraining_wired_behavior_qualification_pending"],
            }
        )
        payload["planned_codex_test_backlog"] = [
            {
                "backlog_id": "planned.kernel_v1",
                "status": "pre_training_architecture_required",
                "pre_training_acceptance_boundary": "Implement exact substrate and freeze the campaign.",
            }
        ]

        report = gate.audit_pre_training_architecture_readiness(
            matrix=payload,
            phase_reports=[],
            book_contract_report={},
            current_hard_gap_count=0,
        )

        self.assertFalse(report["ready"])
        blocker = next(row for row in report["blockers"] if row["kind"] == "unfinished_pre_training_backlog_contracts")
        self.assertEqual(blocker["contracts"][0]["backlog_id"], "planned.kernel_v1")

        payload["planned_codex_test_backlog"][0]["status"] = "pretraining_wired_behavior_qualification_pending"
        report = gate.audit_pre_training_architecture_readiness(
            matrix=payload,
            phase_reports=[],
            book_contract_report={},
            current_hard_gap_count=0,
        )

        self.assertTrue(report["ready"])
        self.assertTrue(report["required_backlog_contracts"][0]["ready"])

        payload["planned_codex_test_backlog"][0]["pre_training_acceptance_boundary"] = ""
        report = gate.audit_pre_training_architecture_readiness(
            matrix=payload,
            phase_reports=[],
            book_contract_report={},
            current_hard_gap_count=0,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["required_backlog_contracts"][0]["pre_training_acceptance_boundary_present"])

    def test_required_cross_phase_backlog_must_exist(self) -> None:
        payload = matrix()
        payload["pre_training_architecture_contract"].update(
            {
                "required_backlog_ids": ["planned.missing_v1"],
                "ready_backlog_statuses": ["implemented"],
            }
        )

        report = gate.audit_pre_training_architecture_readiness(
            matrix=payload,
            phase_reports=[],
            book_contract_report={},
            current_hard_gap_count=0,
        )

        self.assertFalse(report["ready"])
        blocker = next(row for row in report["blockers"] if row["kind"] == "missing_required_pre_training_backlog_contracts")
        self.assertEqual(blocker["backlog_ids"], ["planned.missing_v1"])

    def test_declared_backlog_evidence_must_be_green_and_source_bound(self) -> None:
        import tempfile

        payload = matrix()
        payload["pre_training_architecture_contract"].update(
            {
                "required_backlog_ids": ["planned.kernel_v1"],
                "ready_backlog_statuses": ["retired_by_pretraining_verdict"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.txt"
            source.write_text("bound", encoding="utf-8")
            import hashlib

            report_path = root / "receipt.json"
            report_path.write_text(json.dumps({
                "policy": "fixture_disposition_v1",
                "trigger_state": "GREEN",
                "disposition": "retired",
                "source_artifacts": {
                    "source": {
                        "path": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                },
            }), encoding="utf-8")
            payload["planned_codex_test_backlog"] = [{
                "backlog_id": "planned.kernel_v1",
                "status": "retired_by_pretraining_verdict",
                "pre_training_acceptance_boundary": "Retired by a source-bound verdict.",
                "negative_disposition_contract": {
                    "kind": "campaign_scope_only",
                    "scientific_falsification_claimed": False,
                    "exact_scope": "first campaign only",
                    "reentry_condition": "run a separate matched campaign",
                },
                "pre_training_evidence": {
                    "path": str(report_path),
                    "policy": "fixture_disposition_v1",
                    "required_trigger_state": "GREEN",
                    "required_disposition": "retired",
                },
            }]
            ready = gate.audit_pre_training_architecture_readiness(
                matrix=payload, phase_reports=[], book_contract_report={}, current_hard_gap_count=0
            )
            self.assertTrue(ready["ready"])
            source.write_text("tampered", encoding="utf-8")
            stale = gate.audit_pre_training_architecture_readiness(
                matrix=payload, phase_reports=[], book_contract_report={}, current_hard_gap_count=0
            )
            self.assertFalse(stale["ready"])
            contract = stale["required_backlog_contracts"][0]
            self.assertIn("source_artifacts_stale:source", contract["evidence"]["faults"])

    def test_proxy_failure_cannot_retire_a_mechanism(self) -> None:
        payload = matrix()
        payload["pre_training_architecture_contract"].update(
            {
                "required_backlog_ids": ["planned.kernel_v1"],
                "ready_backlog_statuses": ["retired_by_pretraining_verdict"],
            }
        )
        row = {
            "backlog_id": "planned.kernel_v1",
            "status": "retired_by_pretraining_verdict",
            "pre_training_acceptance_boundary": "A toy proxy failed.",
        }
        payload["planned_codex_test_backlog"] = [row]

        missing = gate.audit_pre_training_architecture_readiness(
            matrix=payload, phase_reports=[], book_contract_report={}, current_hard_gap_count=0
        )
        self.assertFalse(missing["ready"])
        self.assertFalse(
            missing["required_backlog_contracts"][0]["negative_disposition"]["ready"]
        )

        row["negative_disposition_contract"] = {
            "kind": "campaign_scope_only",
            "scientific_falsification_claimed": False,
            "exact_scope": "the first campaign only",
            "reentry_condition": "run a faithful separately preregistered campaign",
        }
        scoped = gate.audit_pre_training_architecture_readiness(
            matrix=payload, phase_reports=[], book_contract_report={}, current_hard_gap_count=0
        )
        self.assertTrue(scoped["ready"])

        row["negative_disposition_contract"]["scientific_falsification_claimed"] = True
        overclaimed = gate.audit_pre_training_architecture_readiness(
            matrix=payload, phase_reports=[], book_contract_report={}, current_hard_gap_count=0
        )
        self.assertFalse(overclaimed["ready"])

    def test_strict_architecture_first_contract_is_machine_enforced(self) -> None:
        payload = matrix()
        payload["pre_training_architecture_contract"].update(
            {
                "strict_architecture_first_enforcement": True,
                "execution_priority": "architecture_before_long_training",
                "training_authority_state": "denied_until_finite_docket_and_freeze_package_are_green",
                "binding_disposition_kinds": [
                    "include_in_frozen_campaign",
                    "exclude_by_falsification_or_retirement",
                    "wire_complete_contract_and_defer_only_learned_efficacy",
                ],
                "required_backlog_ids": ["planned.router_v1"],
                "ready_backlog_statuses": ["implemented"],
                "dependency_order": [
                    "planned.router_v1",
                    "final_cross_owner_replay_and_architecture_freeze_package",
                    "unchanged_final_mlx_mechanics_canaries_and_joint_campaign_preregistration",
                ],
                "completion_evidence_rule": "Require canonical integration and independent evidence.",
                "architecture_change_intake_rule": "Admit only campaign-invalidating architecture changes.",
                "sequence_rule": "Disposition then freeze then training.",
            }
        )
        payload["planned_codex_test_backlog"] = [
            {
                "backlog_id": "planned.router_v1",
                "status": "implemented",
                "pre_training_acceptance_boundary": "Canonical route integration is replayable.",
            }
        ]

        report = gate.audit_pre_training_architecture_readiness(
            matrix=payload,
            phase_reports=[],
            book_contract_report={},
            current_hard_gap_count=0,
        )
        self.assertTrue(report["ready"])
        self.assertTrue(report["strict_architecture_first_enforcement"])

        for mutation in ("bad_priority", "missing_order", "duplicate_order", "bad_authority"):
            broken = copy.deepcopy(payload)
            contract = broken["pre_training_architecture_contract"]
            if mutation == "bad_priority":
                contract["execution_priority"] = "train_first"
            elif mutation == "missing_order":
                contract["dependency_order"].remove("planned.router_v1")
                broken["planned_codex_test_backlog"][0]["status"] = "pre_training_architecture_required"
            elif mutation == "duplicate_order":
                contract["dependency_order"].insert(1, "planned.router_v1")
            else:
                contract["training_authority_state"] = "authorized"
            report = gate.audit_pre_training_architecture_readiness(
                matrix=broken,
                phase_reports=[],
                book_contract_report={},
                current_hard_gap_count=0,
            )
            self.assertFalse(report["ready"], mutation)
            self.assertTrue(
                any(row["kind"] == "architecture_first_enforcement_contract_invalid" for row in report["blockers"]),
                mutation,
            )


if __name__ == "__main__":
    unittest.main()
