from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()

