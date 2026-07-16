from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import roadmap_implementation_gate as gate  # noqa: E402


def matrix(required_status: str = "implemented") -> dict:
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
