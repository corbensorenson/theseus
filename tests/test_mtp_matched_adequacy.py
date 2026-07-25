from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mtp_matched_adequacy as adequacy


class MtpMatchedAdequacyTests(unittest.TestCase):
    def test_config_is_prospective_and_no_cheat(self) -> None:
        config = adequacy.load_config()
        self.assertEqual(5, len(config["candidates"]))
        self.assertTrue(config["prospective_decision"]["functional_verifier_required_for_adoption"])
        self.assertFalse(config["hard_boundaries"]["heldout_labels_visible_to_optimizer"])
        self.assertFalse(config["hard_boundaries"]["architecture_selection_from_auxiliary_loss_alone"])

    def test_source_disjoint_receipt_rejects_overlap(self) -> None:
        train = {"english": [{"source_identity": "same"}]}
        heldout = {"english": [{"source_identity": "same"}]}
        receipt = adequacy.source_disjoint_receipt(train, heldout)
        self.assertFalse(receipt["passed"])
        self.assertEqual(1, receipt["cross_split_overlap_count"])

    def test_balanced_batch_has_one_row_per_arm_and_is_deterministic(self) -> None:
        rows = {
            arm: [{"source_identity": f"{arm}-{index}"} for index in range(3)]
            for arm in ("english", "python", "javascript_typescript", "html_css", "rust")
        }
        first = adequacy.balanced_batches(rows, steps=4, seed=7)
        second = adequacy.balanced_batches(rows, steps=4, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(all(len(batch) == 5 for batch in first))

    def test_sham_and_mtp_routes_use_same_schedule_but_different_gradient_authority(self) -> None:
        config = adequacy.load_config()
        by_id = {row["id"]: row for row in config["candidates"]}
        sham = by_id["ar_independent_sham"]
        mtp = by_id["mtp_conventional_independent"]
        self.assertEqual(sham["head_mode"], mtp["head_mode"])
        self.assertEqual(sham["schedule"], mtp["schedule"])
        self.assertNotEqual(sham["gradient_route"], mtp["gradient_route"])
        self.assertEqual(
            adequacy.candidate_scale(sham, 0, config["training"]),
            adequacy.candidate_scale(mtp, 0, config["training"]),
        )

    def test_curriculum_has_a_bounded_first_active_step(self) -> None:
        config = adequacy.load_config()
        candidate = next(
            row for row in config["candidates"]
            if row["id"] == "mtp_curriculum_independent"
        )
        scales = [
            adequacy.candidate_scale(candidate, step, config["training"])
            for step in range(config["training"]["steps"])
        ]
        first_active = next(index for index, value in enumerate(scales) if value > 0.0)
        self.assertEqual(config["training"]["warmup_steps"], first_active)
        self.assertLess(first_active, config["training"]["steps"])

    def test_functional_verifier_absence_forces_inconclusive(self) -> None:
        config = adequacy.load_config()
        arms = {
            arm: {"ntp_loss": 1.0}
            for arm in config["scoped_arms"]
        }
        controls = []
        candidates = []
        for seed in config["seeds"]:
            controls.append(
                {
                    "seed": seed,
                    "training_wall_seconds": 1.0,
                    "final_heldout": {
                        "ntp_loss": 1.0,
                        "greedy_token_accuracy": 0.1,
                        "by_arm": copy.deepcopy(arms),
                        "functional_verifier_available": False,
                    },
                }
            )
            candidates.append(
                {
                    "seed": seed,
                    "training_wall_seconds": 1.0,
                    "final_heldout": {
                        "ntp_loss": 0.8,
                        "greedy_token_accuracy": 0.2,
                        "by_arm": {
                            arm: {"ntp_loss": 0.8}
                            for arm in config["scoped_arms"]
                        },
                        "functional_verifier_available": False,
                    },
                }
            )
        summary = adequacy.summarize_pair(
            candidates, controls, config["prospective_decision"]
        )
        self.assertEqual("INCONCLUSIVE_EXPERIMENT", summary["disposition"])
        self.assertFalse(summary["gates"]["functional_verifier"])

    def test_no_loss_signal_scopes_mtp_out_without_claiming_falsification(self) -> None:
        comparisons = {
            candidate_id: {
                "disposition": "INCONCLUSIVE_EXPERIMENT",
                "mean_relative_ntp_loss_improvement": -0.001,
                "gates": {"functional_verifier": False},
            }
            for candidate_id in (
                "mtp_conventional_independent",
                "mtp_curriculum_independent",
                "mtp_register_curriculum",
            )
        }
        disposition = adequacy.campaign_disposition(comparisons)
        self.assertEqual(
            "SCOPED_DISCOVERY_EXCLUDED_FROM_FIRST_CAMPAIGN",
            disposition["kind"],
        )
        self.assertFalse(disposition["scientific_falsification_claimed"])
        self.assertIn("verifier-bearing", disposition["reentry_condition"])

    def test_config_mutation_enabling_auxiliary_only_selection_fails_closed(self) -> None:
        config = json.loads(adequacy.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["hard_boundaries"]["architecture_selection_from_auxiliary_loss_alone"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(adequacy.MtpAdequacyFault, "selection_boundary_invalid"):
                adequacy.load_config(path)

    def test_candidate_lease_is_authorized_only_in_scratch_namespace(self) -> None:
        config = adequacy.load_config()
        lease = adequacy.pretraining_candidate_canary.candidate_lease(
            candidate_id=config["candidate_lease_id"],
            max_steps=int(config["training"]["steps"]),
            scratch_checkpoint_root=adequacy.resolve(config["scratch_root"]),
            targets=["shared_trunk"],
            phase="pretraining",
            resume=False,
        )
        self.assertTrue(lease["authorized"])
        self.assertEqual([], lease["faults"])


if __name__ == "__main__":
    unittest.main()
