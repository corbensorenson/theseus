from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generation_architecture_contracts as generation
from standard_causal_transformer_objectives import mtp_curriculum_scale


class GenerationArchitectureContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = generation.load_contract()

    def test_every_named_mode_has_a_complete_record(self) -> None:
        expected = {"autoregressive", "mtp", "medusa", "eagle", "speculative", "layerskip", "sketch_first_llada"}
        self.assertEqual(expected, set(self.contract["modes"]))
        for mode_id in expected:
            record = generation.generation_mode_record(mode_id, self.contract)
            self.assertTrue(generation.validate_mode_record(record, self.contract)["valid"])

    def test_checkpoint_roundtrip_migration_and_retirement_exclusion(self) -> None:
        receipt = generation.checkpoint_roundtrip(self.contract)
        self.assertTrue(receipt["roundtrip_exact"])
        self.assertTrue(receipt["migration_exact"])
        self.assertTrue(receipt["cleanup_complete"])
        self.assertTrue(receipt["retired_modes_absent_from_optimizer"])

    def test_speculative_helper_binds_target_and_accepted_prefix_cache(self) -> None:
        checkpoint = generation.checkpoint_contract(self.contract)
        draft = {"draft_revision": "draft:1", "target_model_revision": checkpoint["model_revision"], "target_base_parameter_digest": checkpoint["base_parameter_digest"], "draft_checkpoint_digest": generation.digest("draft"), "cache_commit_policy": "accepted_prefix_only"}
        receipt = generation.speculative_loader_receipt(checkpoint, draft)
        self.assertTrue(receipt["compatible"])
        self.assertFalse(receipt["enabled"])
        self.assertFalse(receipt["target_topology_changed"])

    def test_mtp_executes_on_mlx_with_shape_safe_future_offsets(self) -> None:
        receipt = generation.mlx_mtp_canary(self.contract)
        self.assertTrue(receipt["available"])
        self.assertTrue(receipt["passed"])
        self.assertEqual(0, receipt["optimizer_steps"])
        self.assertEqual([7, 6, 5], receipt["observed"]["valid_positions"])

    def test_adequate_mtp_candidates_train_heads_and_reload_exactly(self) -> None:
        receipt = generation.mlx_mtp_adequacy_canary(
            self.contract, optimizer_steps=24
        )
        self.assertTrue(receipt["available"])
        self.assertTrue(receipt["passed"])
        self.assertEqual(
            {"conventional_independent", "register_conditioned"},
            set(receipt["candidate_checks"]),
        )
        self.assertEqual(0, receipt["public_training_rows"])
        self.assertEqual("NOT_EVALUATED", receipt["capability_claim"])
        for checks in receipt["candidate_checks"].values():
            self.assertTrue(all(checks.values()))

    def test_mtp_curriculum_starts_with_ntp_and_ramps_monotonically(self) -> None:
        scales = [
            mtp_curriculum_scale(
                step, warmup_steps=2, ramp_steps=4, maximum=0.5
            )
            for step in range(8)
        ]
        self.assertEqual([0.0, 0.0], scales[:2])
        self.assertEqual(0.5, scales[-1])
        self.assertEqual(sorted(scales), scales)

    def test_legacy_rank_one_is_not_a_selectable_candidate(self) -> None:
        candidate = self.contract["mtp_candidates"]["legacy_shared_rank1"]
        self.assertEqual(
            "compatibility_fixture_not_selectable", candidate["selection_state"]
        )
        with self.assertRaisesRegex(
            generation.GenerationArchitectureFault, "mtp_candidate_unknown"
        ):
            generation.mtp_candidate_model_config("missing", self.contract)

    def test_retired_modes_have_explicit_reentry_conditions(self) -> None:
        retired = [mode for mode in self.contract["modes"].values() if mode["first_campaign_disposition"].startswith("retired")]
        self.assertEqual(5, len(retired))
        self.assertTrue(all(mode.get("reentry_condition") for mode in retired))

    def test_draft_candidates_share_target_authoritative_abi(self) -> None:
        self.assertEqual(
            {"medusa_tree", "eagle_feature", "separate_draft", "kerc_structured_unit"},
            set(self.contract["draft_candidates"]),
        )
        self.assertEqual(
            "target_accepted_prefix_only",
            self.contract["draft_abi"]["cache_commit_policy"],
        )

    def test_mode_selection_requires_behavior_positive_receipt(self) -> None:
        receipt = generation.activation_receipt({}, self.contract)
        self.assertFalse(receipt["authorized"])
        self.assertIn("behavior_positive_generation_evidence_missing", receipt["blockers"])

    def test_mutation_controls_fail_closed(self) -> None:
        controls = generation.mutation_controls(self.contract)
        self.assertEqual(controls["case_count"], controls["passed_count"])
        self.assertGreaterEqual(controls["case_count"], 11)

    def test_reference_suite_is_green_without_training_or_credit(self) -> None:
        report = generation.run_reference_suite(self.contract)
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertEqual(0, report["summary"]["optimizer_exposure_steps"])
        self.assertEqual(0, report["summary"]["public_training_rows_written"])
        self.assertEqual(0, report["summary"]["fallback_or_template_credit"])


if __name__ == "__main__":
    unittest.main()
