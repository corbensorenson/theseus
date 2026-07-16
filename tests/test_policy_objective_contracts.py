from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_objective_contracts as objectives


class PolicyObjectiveContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = objectives.load_contract()
        self.pair = objectives.reference_pair()
        self.group = objectives.reference_rollout_group()

    def test_all_named_objectives_are_executable_and_finite(self) -> None:
        offline = {name: objectives.preference_loss(name, self.pair, self.contract) for name in self.contract["offline_preference_objectives"]}
        reward = {name: objectives.verifier_reward_loss(name, self.group, self.contract) for name in self.contract["verifier_reward_objectives"]}
        self.assertEqual({"dpo", "ipo", "orpo", "kto", "simpo"}, set(offline))
        self.assertEqual({"grpo", "rloo", "remax", "rlvr"}, set(reward))
        self.assertTrue(all(math.isfinite(value) for value in offline.values()))
        self.assertTrue(all(math.isfinite(row["loss"]) for row in reward.values()))

    def test_pair_and_rollout_receipts_are_content_bound(self) -> None:
        pair = objectives.validate_preference_pair(self.pair, self.contract)
        group = objectives.validate_rollout_group(self.group, self.contract)
        self.assertTrue(pair["pair_digest"].startswith("sha256:"))
        self.assertTrue(group["group_digest"].startswith("sha256:"))

    def test_checkpoint_reference_optimizer_migration_and_cleanup(self) -> None:
        receipt = objectives.checkpoint_roundtrip(self.contract)
        self.assertTrue(receipt["roundtrip_exact"])
        self.assertTrue(receipt["reference_identity_frozen"])
        self.assertTrue(receipt["optimizer_state_present"])
        self.assertTrue(receipt["migration_exact"])
        self.assertTrue(receipt["cleanup_complete"])

    def test_update_lease_rolls_back_exactly_and_seals_epoch(self) -> None:
        receipt = objectives.policy_lease_rollback_receipt()
        self.assertEqual("generator", receipt["target_id"])
        self.assertTrue(receipt["rollback_exact"])
        self.assertTrue(receipt["epoch_sealed"])

    def test_objective_selection_is_disabled_without_behavior_evidence(self) -> None:
        receipt = objectives.activation_receipt({}, self.contract)
        self.assertFalse(receipt["authorized"])
        self.assertIn("objectives_disabled_by_frozen_contract", receipt["blockers"])

    def test_reward_hacking_and_integrity_mutations_fail_closed(self) -> None:
        controls = objectives.mutation_controls(self.contract)
        self.assertEqual(controls["case_count"], controls["passed_count"])
        self.assertGreaterEqual(controls["case_count"], 12)

    def test_capacity_overrun_is_rejected(self) -> None:
        group = copy.deepcopy(self.group)
        group["verifier_capacity_observed"]["compute_units"] = 101
        with self.assertRaisesRegex(objectives.ObjectiveContractFault, "verifier_capacity_exceeded"):
            objectives.verifier_reward_loss("grpo", group, self.contract)

    def test_reference_suite_has_mlx_parity_and_zero_training_exposure(self) -> None:
        report = objectives.run_reference_suite(self.contract)
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertTrue(report["summary"]["mlx_parity"])
        self.assertEqual(0, report["summary"]["optimizer_exposure_steps"])
        self.assertEqual(0, report["summary"]["public_training_rows_written"])
        self.assertEqual(0, report["summary"]["fallback_or_template_credit"])


if __name__ == "__main__":
    unittest.main()
