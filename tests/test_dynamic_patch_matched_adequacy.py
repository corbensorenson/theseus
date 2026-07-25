from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dynamic_patch_matched_adequacy as adequacy


class DynamicPatchMatchedAdequacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = adequacy.load_config()

    def test_preflight_uses_source_disjoint_governed_rows(self) -> None:
        report = adequacy.preflight()
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertEqual(6, report["candidate_count"])
        self.assertEqual(3, report["seed_count"])
        self.assertEqual(50, report["train_rows"])
        self.assertEqual(20, report["heldout_rows"])
        self.assertEqual(0, report["source_disjoint"]["cross_split_overlap_count"])

    def test_entropy_model_uses_only_supplied_training_rows(self) -> None:
        train = [{"payload": b"aaaa"}, {"payload": b"aaab"}]
        model = adequacy.fit_prefix_entropy(train)
        familiar = adequacy.prefix_uncertainty(b"aaaa", model)
        unfamiliar = adequacy.prefix_uncertainty(b"zzzz", model)
        self.assertLess(sum(familiar), sum(unfamiliar))

    def test_boundary_families_are_causal_and_bounded(self) -> None:
        payload = b"alpha beta(x); gamma"
        entropy = adequacy.fit_prefix_entropy([{"payload": payload}])
        uncertainty = adequacy.prefix_uncertainty(payload, entropy)
        for source in (
            "every_byte",
            "fixed_width",
            "prefix_entropy",
            "learned_prefix_entropy_prediction",
            "visible_whitespace_and_code_punctuation",
        ):
            targets = adequacy.boundary_targets(
                payload,
                source,
                uncertainty=uncertainty,
                fixed_width=8,
                entropy_threshold=0.58,
            )
            points = adequacy.points_for_targets(targets, 16)
            self.assertEqual(0, points[0])
            self.assertEqual(len(payload), points[-1])
            self.assertLessEqual(max(b - a for a, b in zip(points, points[1:])), 16)

    def test_patch_inputs_mask_only_target_bytes(self) -> None:
        rows = [{
            "payload": b"prompt-answer",
            "target_byte_start": 7,
            "target_byte_count": 6,
        }]
        entropy = adequacy.fit_prefix_entropy(rows)
        packed = adequacy.patch_inputs(rows, "fixed_width", entropy, self.config)
        self.assertEqual(13, int(packed["mask"].sum()))
        self.assertEqual(6, int(packed["target_mask"].sum()))
        self.assertEqual(0, int(packed["target_mask"][0, :7].sum()))

    def test_candidate_contract_forbids_loss_only_selection(self) -> None:
        self.assertFalse(
            self.config["hard_boundaries"]["selection_from_compression_or_loss_alone"]
        )
        self.assertTrue(
            self.config["prospective_decision"]["functional_verifier_required_for_adoption"]
        )

    def test_fractional_batch_schedule_matches_requested_mean(self) -> None:
        schedule = adequacy.fractional_batch_schedule(
            1.6, steps=50, maximum_batch_size=8
        )
        self.assertEqual(50, len(schedule))
        self.assertEqual({1, 2}, set(schedule))
        self.assertAlmostEqual(1.6, sum(schedule) / len(schedule), places=6)


if __name__ == "__main__":
    unittest.main()
