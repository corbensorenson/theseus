from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_update_lease as leases


class PolicyUpdateLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = leases.load_contract()
        self.target_id = "planner"
        self.target = self.contract["targets"][self.target_id]
        self.request = leases.reference_request(self.target_id, self.target)

    def test_contract_covers_all_update_targets(self) -> None:
        self.assertEqual({"planner", "router", "vcm_selector", "verifier", "executor", "generator", "generation_mode"}, set(self.contract["targets"]))

    def test_lease_delta_monitor_and_commit_are_hash_chained(self) -> None:
        lease = leases.issue_lease(self.request, self.contract)
        state = copy.deepcopy(lease["current_state"]); state["revision"] = 2
        lease = leases.apply_delta(lease, {"target_id": self.target_id, "state_path": lease["state_path"], "before_digest": lease["current_digest"], "next_state": state, "cost_observed": {"verification": 1, "repair": 0, "human_cleanup": 0, "compute": 1, "energy": 1}}, self.contract)
        for suffix in ("a", "b"):
            lease = leases.observe(lease, {"observation_id": suffix, "heldout_contract": lease["heldout_contract"], "drift": 0, "heldout_delta": 0.1, "verification_escape_count": 0, "authority_violation_count": 0}, self.contract)
        committed = leases.commit(lease)
        self.assertEqual("committed", committed["state"])
        self.assertTrue(leases.verify_journal(committed))

    def test_monitor_breach_seals_epoch_and_restores_baseline(self) -> None:
        lease = leases.issue_lease(self.request, self.contract)
        state = copy.deepcopy(lease["current_state"]); state["revision"] = 2
        lease = leases.apply_delta(lease, {"target_id": self.target_id, "state_path": lease["state_path"], "before_digest": lease["current_digest"], "next_state": state, "cost_observed": {"verification": 1, "repair": 0, "human_cleanup": 0, "compute": 1, "energy": 1}}, self.contract)
        rolled = leases.observe(lease, {"observation_id": "breach", "heldout_contract": lease["heldout_contract"], "drift": 1, "heldout_delta": -1, "verification_escape_count": 1, "authority_violation_count": 0}, self.contract)
        self.assertEqual("rolled_back", rolled["state"])
        self.assertTrue(rolled["rollback_exact"])
        self.assertTrue(rolled["epoch_token"].startswith("sealed:"))

    def test_authority_feedback_cost_and_cross_target_mutations_fail_closed(self) -> None:
        controls = leases.mutation_controls(self.contract)
        self.assertEqual(controls["case_count"], controls["passed_count"])

    def test_reference_matrix_commits_every_target_and_keeps_costs_visible(self) -> None:
        report = leases.run_reference_matrix(self.contract)
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertEqual(7, report["summary"]["committed_target_count"])
        self.assertTrue(report["summary"]["rollback_canary_exact"])
        self.assertTrue(all(not row["authority_delta"] for row in report["target_receipts"]))
        self.assertTrue(all(set(row["cost_observed"]) == {"verification", "repair", "human_cleanup", "compute", "energy"} for row in report["target_receipts"]))


if __name__ == "__main__":
    unittest.main()
