from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import open_ended_improvement_campaign as campaigns


class OpenEndedImprovementCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = campaigns.load_contract()
        self.manifest = campaigns.reference_manifest(self.contract)
        self.campaign = campaigns.open_campaign(self.manifest, self.contract, architecture_fixture=True)

    def test_runtime_starts_disabled_without_behavior_positive_evidence(self) -> None:
        receipt = campaigns.activation_receipt({}, self.contract)
        self.assertFalse(receipt["authorized"])
        self.assertIn("runtime_disabled_by_frozen_contract", receipt["blockers"])
        with self.assertRaisesRegex(campaigns.CampaignFault, "runtime_campaign_not_authorized"):
            campaigns.open_campaign(self.manifest, self.contract)

    def test_authorities_holdout_and_best_final_roles_are_distinct(self) -> None:
        self.assertEqual(5, len(set(self.campaign["authorities"].values())))
        self.assertEqual(self.contract["fixed_holdout"]["content_digest"], self.campaign["holdout_digest"])
        self.assertNotEqual(self.campaign["best_digest"], self.campaign["final_digest"])

    def test_single_axis_matched_challenger_can_be_independently_promoted(self) -> None:
        campaign = campaigns.propose_challenger(self.campaign, campaigns.reference_proposal(self.campaign, "qualified", "context_policy", 2), self.contract)
        campaign = campaigns.evaluate_challenger(campaign, campaigns.reference_evaluation(campaign, utility=0.54, coverage=0.84, weak_tail=0.42), self.contract)
        campaign = campaigns.decide_challenger(campaign, self.contract)
        self.assertEqual("promoted", campaign["history"][-1]["state"])
        self.assertNotEqual(campaign["baseline_champion_digest"], campaign["champion_digest"])
        self.assertTrue(campaigns.verify_journal(campaign))

    def test_weak_tail_regression_is_rejected_and_retained(self) -> None:
        campaign = campaigns.propose_challenger(self.campaign, campaigns.reference_proposal(self.campaign, "tail", "router_policy", 2), self.contract)
        campaign = campaigns.evaluate_challenger(campaign, campaigns.reference_evaluation(campaign, utility=0.60, coverage=0.90, weak_tail=0.20), self.contract)
        campaign = campaigns.decide_challenger(campaign, self.contract)
        self.assertEqual("rejected", campaign["history"][-1]["state"])
        self.assertEqual("weak_tail_regression", campaign["negative_knowledge"][-1]["reason"])

    def test_negative_knowledge_prevents_exact_failed_repeat(self) -> None:
        proposal = campaigns.reference_proposal(self.campaign, "repeat", "router_policy", 2)
        proposed = campaigns.propose_challenger(self.campaign, proposal, self.contract)
        fingerprint = proposed["active_challenger"]["fingerprint"]
        campaign = copy.deepcopy(self.campaign)
        campaign["negative_knowledge"] = [{"fingerprint": fingerprint}]
        with self.assertRaisesRegex(campaigns.CampaignFault, "known_failed_candidate_repeated"):
            campaigns.propose_challenger(campaign, proposal, self.contract)

    def test_debt_ceiling_forces_authorized_stop_and_handoff(self) -> None:
        campaign = campaigns.propose_challenger(self.campaign, campaigns.reference_proposal(self.campaign, "debt", "objective", 2), self.contract)
        evaluation = campaigns.reference_evaluation(campaign, utility=0.54, coverage=0.84, weak_tail=0.42)
        evaluation["costs"]["verification"] = self.contract["debt_ceilings"]["verification"] + 1
        campaign = campaigns.evaluate_challenger(campaign, evaluation, self.contract)
        self.assertEqual("stopped", campaign["state"])
        self.assertEqual("debt_ceiling_exceeded", campaign["terminal_reason"])
        self.assertEqual(0, campaign["shutdown_handoff"]["unresolved_challenger_count"])

    def test_stop_seals_epoch_and_rollback_restores_exact_baseline(self) -> None:
        stopped = campaigns.stop_campaign(self.campaign, "shutdown_handoff", self.campaign["authorities"]["stop"], self.contract)
        self.assertTrue(stopped["epoch_token"].startswith("sealed:"))
        rolled = campaigns.rollback_campaign(stopped, stopped["authorities"]["rollback"], reason="test")
        self.assertTrue(rolled["rollback_exact"])
        self.assertEqual(rolled["champion_digest"], rolled["baseline_champion_digest"])

    def test_mutation_controls_fail_closed(self) -> None:
        controls = campaigns.mutation_controls(self.contract)
        self.assertEqual(controls["case_count"], controls["passed_count"])
        self.assertGreaterEqual(controls["case_count"], 12)

    def test_reference_campaign_is_green_without_training_or_effects(self) -> None:
        report = campaigns.run_reference_campaign(self.contract)
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertEqual(0, report["summary"]["optimizer_exposure_steps"])
        self.assertEqual(0, report["summary"]["runtime_effect_count"])
        self.assertFalse(report["summary"]["runtime_authorized"])
        self.assertTrue(report["summary"]["rollback_exact"])


if __name__ == "__main__":
    unittest.main()
