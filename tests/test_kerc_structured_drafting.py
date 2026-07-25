from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_structured_drafting as drafting


class KercStructuredDraftingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = drafting.reference_vocabulary()
        self.first = drafting.StructuredSemanticUnit(
            "REPORT", "ASSERTED", "AFFIRMED", "NONE",
            (("VALUE", "@N1"), ("SOURCE", "@Q1")), "exact", True, False,
        )
        self.second = drafting.StructuredSemanticUnit(
            "COMPARE", "ASSERTED", "AFFIRMED", "NONE",
            (("LEFT", "@N1"), ("RIGHT", "@E1")), "faithful", True, True,
        )

    def test_incremental_grammar_accepts_complete_units_and_rejects_bad_order(self) -> None:
        receipt = drafting.validate_unit_sequence((self.first, self.second), self.vocabulary)
        self.assertTrue(receipt["program_closed"])
        grammar = drafting.IncrementalStructuredGrammar(2)
        grammar.advance("UNIT_BEGIN")
        with self.assertRaisesRegex(drafting.StructuredDraftFault, "transition_invalid"):
            grammar.advance("POINTER")

    def test_typed_pointer_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(drafting.StructuredDraftFault, "role_pointer_type_mismatch"):
            drafting.validate_unit(
                self.first,
                self.vocabulary,
                role_pointer_types={"VALUE": {"number"}, "SOURCE": {"quote"}},
                pointer_types={"@N1": "quote", "@Q1": "quote"},
            )

    def test_target_acceptance_discards_wrong_suffix(self) -> None:
        wrong = drafting.StructuredSemanticUnit(
            "COMPARE", "POSSIBLE", "AFFIRMED", "NONE",
            (("LEFT", "@N1"), ("RIGHT", "@E1")), "faithful", True, True,
        )
        receipt = drafting.verify_structured_draft(
            (self.first, wrong),
            (self.first, self.second),
            vocabulary=self.vocabulary,
            semantic_verifier=lambda unit: drafting.validate_unit(unit, self.vocabulary)["valid"],
        )
        self.assertEqual(1, receipt.accepted_count)
        self.assertEqual(1, receipt.target_mismatch_count)
        self.assertEqual(self.second, receipt.target_unit_committed)

    def test_mlx_heads_learn_reload_and_use_hidden_state(self) -> None:
        receipt = drafting.mlx_structured_drafting_canary(optimizer_steps=96)
        self.assertTrue(receipt["available"], receipt.get("stderr_tail"))
        self.assertTrue(receipt["passed"], receipt)
        self.assertEqual(0, receipt["grammar_pointer_renderer_learned_credit"])
        self.assertEqual("NOT_EVALUATED", receipt["capability_claim"])

    def test_reference_suite_is_green_without_capability_claim(self) -> None:
        report = drafting.run_reference_suite()
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertIn("not_semantic_truth", report["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
