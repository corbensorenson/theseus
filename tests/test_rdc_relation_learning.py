from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rdc_relation_learning as learning


class RDCRelationLearningTests(unittest.TestCase):
    def test_denominator_and_budget_are_fail_closed(self) -> None:
        batch = learning.reference_batch()
        self.assertEqual(4, learning.validate_batch(batch)["denominator_count"])
        with self.assertRaisesRegex(learning.RelationLearningFault, "proposal_budget_invalid"):
            learning.select_from_learned_scores(
                batch, proposal_scores=(1, 2, 3, 4), qualification_scores=(1, 2, 3, 4),
                proposal_budget=5, qualification_threshold=0,
            )

    def test_evaluator_labels_cannot_change_inference_selection(self) -> None:
        batch = learning.reference_batch()
        common = dict(
            proposal_scores=(0.9, 0.8, 0.2, 0.1),
            qualification_scores=(0.9, 0.2, 0.8, 0.1),
            proposal_budget=2,
            qualification_threshold=0.5,
        )
        first = learning.select_from_learned_scores(batch, known_relevant_ids=("transfer:1",), **common)
        second = learning.select_from_learned_scores(batch, known_relevant_ids=("time:1",), **common)
        self.assertEqual(first.proposed_relation_ids, second.proposed_relation_ids)
        self.assertEqual(first.qualified_relation_ids, second.qualified_relation_ids)
        self.assertNotEqual(first.proposal_recall, second.proposal_recall)

    def test_mlx_proposer_and_independent_qualifier_learn_and_reload(self) -> None:
        receipt = learning.mlx_relation_learning_canary(optimizer_steps=128)
        self.assertTrue(receipt["available"], receipt.get("stderr_tail"))
        self.assertTrue(receipt["passed"], receipt)
        self.assertEqual(0, receipt["public_training_rows"])
        self.assertEqual("NOT_EVALUATED", receipt["capability_claim"])

    def test_reference_suite_is_green_without_truth_claim(self) -> None:
        report = learning.run_reference_suite()
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertIn("not_relation_truth", report["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
