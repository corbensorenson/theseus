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

import full_state_update_causality as causality


class FullStateUpdateCausalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = causality.load_contract()
        self.inventory = causality.build_reference_inventory(self.contract)

    def test_inventory_covers_every_mutable_state_and_has_distinct_authority(self) -> None:
        summary = causality.validate_inventory(self.inventory, self.contract)
        self.assertEqual(set(self.contract["required_artifact_kinds"]), {row["kind"] for row in self.inventory["artifacts"]})
        self.assertEqual(len(self.contract["required_artifact_kinds"]), summary["artifact_kind_count"])
        self.assertNotEqual(self.inventory["best_checkpoint_id"], self.inventory["final_checkpoint_id"])

    def test_package_roundtrip_is_content_addressed_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "package"
            manifest = causality.write_state_package(self.inventory, package, self.contract)
            replayed = causality.read_state_package(package, self.contract)
            self.assertEqual(self.inventory["inventory_digest"], replayed["inventory_digest"])
            target = package / manifest["artifacts"][0]["replicas"][0]["path"]
            target.write_text(json.dumps({"tampered": True}), encoding="utf-8")
            with self.assertRaisesRegex(causality.FullStateCausalityFault, "package_artifact_file_mismatch"):
                causality.read_state_package(package, self.contract)

    def test_update_binds_exposure_changes_state_and_rolls_back_exactly(self) -> None:
        prepared = causality.prepare_update(self.inventory, admitted_candidate_ids=["source:row-001"], optimizer_steps=2, contract=self.contract)
        committed = causality.apply_reference_update(prepared, self.contract)
        self.assertNotEqual(committed["pre_inventory_digest"], committed["post_inventory_digest"])
        self.assertTrue(committed["best_selection_receipt"]["roles_distinct"])
        rolled_back = causality.rollback_update(committed, self.contract)
        self.assertTrue(rolled_back["exact_pre_state_restored"])
        self.assertEqual(self.inventory["inventory_digest"], rolled_back["restored_inventory_digest"])

    def test_update_rejects_unknown_lineage_and_unbounded_steps(self) -> None:
        with self.assertRaisesRegex(causality.FullStateCausalityFault, "optimizer_exposure_lineage_unknown"):
            causality.prepare_update(self.inventory, admitted_candidate_ids=["source:unknown"], optimizer_steps=1, contract=self.contract)
        with self.assertRaisesRegex(causality.FullStateCausalityFault, "optimizer_step_authority_exceeded"):
            causality.prepare_update(self.inventory, admitted_candidate_ids=["source:row-001"], optimizer_steps=9, contract=self.contract)

    def test_deletion_closes_descendants_and_separates_claims(self) -> None:
        plan = causality.plan_deletion(self.inventory, {"source:row-001"}, self.contract)
        self.assertTrue(plan["closure_complete"])
        self.assertFalse(plan["physical_erasure_claim_allowed"])
        self.assertFalse(plan["behavioral_unlearning_claim_allowed"])
        kinds = {row["kind"] for row in plan["actions"]}
        self.assertIn("checkpoint_backup", kinds)
        self.assertIn("external_effect_receipt", kinds)
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "package"
            causality.write_state_package(self.inventory, package, self.contract)
            receipt = causality.execute_package_deletion(package, self.inventory, plan)
        self.assertTrue(receipt["all_target_files_absent"])
        self.assertTrue(receipt["all_declared_replicas_accounted"])
        checkpoint_erasure = next(row for row in receipt["erased"] if row["artifact_id"] == "checkpoint:final-000")
        self.assertEqual({"primary", "local_backup"}, {row["replica_id"] for row in checkpoint_erasure["replicas"]})
        self.assertTrue(checkpoint_erasure["all_replica_paths_absent"])
        self.assertEqual("bounded_local_package_only", receipt["privacy_erasure_scope"])
        self.assertTrue(receipt["behavioral_influence_state"].startswith("unverified"))

    def test_best_final_rng_lineage_and_effect_mutations_fail_closed(self) -> None:
        mutations = []
        missing_rng = copy.deepcopy(self.inventory)
        missing_rng["artifacts"] = [row for row in missing_rng["artifacts"] if row["kind"] != "rng_state"]
        mutations.append((missing_rng, "required_artifact_kinds_missing"))
        same_checkpoint = copy.deepcopy(self.inventory)
        same_checkpoint["best_checkpoint_id"] = same_checkpoint["final_checkpoint_id"]
        mutations.append((same_checkpoint, "best_final_authority_not_distinct"))
        effect = copy.deepcopy(self.inventory)
        next(row for row in effect["artifacts"] if row["kind"] == "external_effect_receipt")["compensation"] = "none"
        mutations.append((effect, "external_effect_compensation_missing"))
        for candidate, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(causality.FullStateCausalityFault, message):
                    causality.validate_inventory(candidate, self.contract)

    def test_complete_reference_fixture_is_green_without_unlearning_overclaim(self) -> None:
        report = causality.run_reference_fixture(self.contract)
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertEqual(report["summary"]["gate_count"], report["summary"]["passed_gate_count"])
        self.assertEqual(report["summary"]["mutation_case_count"], report["summary"]["mutation_passed_count"])
        self.assertTrue(any("Behavioral influence" in claim for claim in report["non_claims"]))


if __name__ == "__main__":
    unittest.main()
