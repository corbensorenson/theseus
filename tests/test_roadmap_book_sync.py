#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import roadmap_implementation_gate as gate  # noqa: E402


class RoadmapBookSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix_path = ROOT / "configs" / "roadmap_implementation_matrix.json"
        cls.book_root = ROOT.parent / "AI_book"
        cls.matrix = json.loads(cls.matrix_path.read_text(encoding="utf-8"))

    def audit(self, matrix: dict) -> dict:
        return gate.audit_book_implementation_contract(matrix, self.book_root)

    def gap_kinds(self, report: dict) -> set[str]:
        return {str(row.get("kind") or "") for row in report["hard_gaps"]}

    def test_subsystem_first_program_selection_is_machine_bound(self) -> None:
        recenter = self.matrix["research_program_recenter"]
        flagship = self.matrix["flagship_lane_governance"]
        neural = recenter["neural_hold"]

        self.assertEqual("SUBSYSTEM_PROOF_ACTIVE_NEURAL_HOLD", recenter["state"])
        self.assertEqual("ASI_STACK_SUBSYSTEM_CAUSAL_PROOF", recenter["active_track"])
        self.assertEqual(
            "virtual-context-abi.core",
            recenter["active_claim"]["claim_id"],
        )
        self.assertEqual(
            "VCM_V3_K3_TERMINAL_INCONCLUSIVE_EXPERIMENT_HOST_OPERABILITY_REDESIGN_REQUIRED",
            recenter["active_claim"]["state"],
        )
        self.assertFalse(recenter["active_claim"]["fresh_claim_pool_authorized"])
        self.assertEqual(
            1,
            recenter["claim_selection_policy"][
                "maximum_simultaneously_active_claims"
            ],
        )
        self.assertIn(
            "Luna reference outputs",
            recenter["claim_selection_policy"]["forbidden_inputs"],
        )
        self.assertEqual(
            "F1_asi_stack_subsystem_causal_proof",
            flagship["active_flagship_lane_id"],
        )
        self.assertEqual(
            "F_memory_planning_tool_substrate", flagship["active_track_id"]
        )
        self.assertEqual("HOLD_SUBSYSTEM_PROOF_FIRST", neural["state"])
        self.assertFalse(neural["launch_allowed"])
        self.assertEqual(11992, neural["optimizer_steps"])
        self.assertFalse(neural["D2_consumed"])
        self.assertTrue(recenter["neural_reentry"]["all_conditions_required"])
        self.assertGreaterEqual(len(recenter["neural_reentry"]["conditions"]), 6)
        self.assertFalse(recenter["autonomy"]["routine_user_approval_required"])

    def test_crosswalk_summary_counts_all_84_current_rows(self) -> None:
        summary = self.matrix["book_chapter_crosswalk_summary"]
        rows = self.matrix["book_chapter_implementation_crosswalk"]
        derived_tracks: dict[str, int] = {}
        for row in rows:
            track = row["primary_track_id"]
            derived_tracks[track] = derived_tracks.get(track, 0) + 1

        self.assertEqual(84, summary["book_chapter_count"])
        self.assertEqual(84, len(rows))
        self.assertEqual(derived_tracks, summary["track_counts"])
        self.assertEqual(84, sum(summary["current_state_counts"].values()))
        self.assertEqual(84, sum(summary["support_state_target_counts"].values()))

    def test_current_crosswalk_matches_manifest_exactly(self) -> None:
        report = self.audit(self.matrix)
        summary = report["summary"]
        self.assertTrue(summary["book_manifest_order_match"])
        self.assertTrue(summary["book_manifest_digest_match"])
        self.assertEqual("pinned_git_commit", summary["book_manifest_source"])
        self.assertEqual(
            self.matrix["latest_ai_book_reconciliation"]["book_commit"],
            summary["book_manifest_commit"],
        )
        self.assertEqual(0, summary["book_manifest_source_field_drift_count"])
        self.assertEqual(84, summary["book_manifest_chapter_count"])
        self.assertEqual(636, summary["book_codex_test_count"])
        self.assertEqual(204, summary["book_pending_or_partial_codex_test_count"])

    def test_reordered_rows_fail_closed(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        rows = matrix["book_chapter_implementation_crosswalk"]
        rows[0], rows[1] = rows[1], rows[0]
        report = self.audit(matrix)
        self.assertIn("book_manifest_chapter_id_order_mismatch", self.gap_kinds(report))

    def test_book_owned_field_drift_fails_closed(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        matrix["book_chapter_implementation_crosswalk"][0]["book_invariants"] = ["changed"]
        report = self.audit(matrix)
        self.assertIn("book_manifest_source_field_drift", self.gap_kinds(report))
        self.assertGreater(report["summary"]["book_manifest_source_field_drift_count"], 0)

    def test_manifest_digest_drift_fails_closed(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        matrix["latest_ai_book_reconciliation"]["manifest_sha256"] = "0" * 64
        report = self.audit(matrix)
        self.assertIn("book_manifest_digest_mismatch", self.gap_kinds(report))

    def test_live_book_worktree_drift_does_not_replace_pinned_manifest(self) -> None:
        report = self.audit(self.matrix)
        summary = report["summary"]
        self.assertTrue(summary["book_manifest_digest_match"])
        if summary["live_book_manifest_differs_from_pin"]:
            warning_kinds = {str(row.get("kind") or "") for row in report["warnings"]}
            self.assertIn("live_book_worktree_differs_from_pinned_snapshot", warning_kinds)
        self.assertNotIn("book_manifest_digest_mismatch", self.gap_kinds(report))

    def test_missing_pinned_commit_fails_closed(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        matrix["latest_ai_book_reconciliation"]["book_commit"] = "0" * 40
        report = self.audit(matrix)
        self.assertIn("pinned_book_manifest_unavailable", self.gap_kinds(report))

    def test_84_chapter_reconciliation_is_exact_owned_and_non_authorizing(self) -> None:
        review = self.matrix[
            "latest_deep_technical_and_asi_stack_review_reconciliation"
        ]
        audit = review["live_source_audit"]
        intake = self.matrix["asi_stack_completion_program"]["live_book_intake"]

        self.assertEqual(84, audit["book_committed_chapter_count"])
        self.assertEqual(84, audit["authoritative_theseus_crosswalk_row_count"])
        self.assertEqual(0, audit["unmapped_current_chapter_count"])
        self.assertEqual([], audit["unmapped_current_chapter_ids"])
        self.assertEqual(84, intake["observed_chapter_count"])
        self.assertEqual(84, intake["authoritative_crosswalk_row_count"])
        self.assertEqual(0, intake["unmapped_current_chapter_count"])
        self.assertFalse(
            self.matrix["asi_stack_completion_program"][
                "authoritative_book_pin_unchanged"
            ]
        )
        self.assertTrue(
            self.matrix["asi_stack_completion_program"][
                "authoritative_book_pin_advanced"
            ]
        )
        self.assertEqual(
            "source_binding_complete",
            next(
                row["state"]
                for row in self.matrix["asi_stack_completion_program"][
                    "work_packages"
                ]
                if row["id"] == "ASI-00"
            ),
        )
        self.assertIn(
            "changes neither runtime authority nor book support",
            intake["rule"],
        )

        required_disposition_fields = {
            "shared_field_disposition",
            "owner_work_package_id",
            "mechanism_maturity",
            "evidence_maturity",
            "route_maturity",
            "activation_gate",
            "bound_test",
            "residual",
            "maximum_inference",
        }
        rows = self.matrix["book_chapter_implementation_crosswalk"]
        self.assertEqual(84, len(rows))
        for row in rows:
            self.assertFalse(
                [field for field in required_disposition_fields if not row.get(field)],
                row["chapter_id"],
            )

    def test_security_and_evaluator_review_work_has_bounded_owners(self) -> None:
        program = self.matrix["asi_stack_completion_program"]
        packages = {row["id"]: row for row in program["work_packages"]}

        self.assertEqual("required_now", packages["ASI-31"]["state"])
        self.assertEqual("pretraining_contract", packages["ASI-32"]["state"])
        self.assertIn(
            "ASI-31", program["execution_waves"][0]["work_package_ids"]
        )
        self.assertIn(
            "ASI-32", program["execution_waves"][1]["work_package_ids"]
        )
        self.assertIn(
            "runtime forbidden-field taint",
            packages["ASI-32"]["acceptance_boundary"],
        )

    def test_d1_source_successor_is_executable_but_downstream_gap_is_explicit(self) -> None:
        flagship = self.matrix["asi_stack_completion_program"][
            "p4v2r2_cognitive_compilation_instrument"
        ]
        successor = flagship["prospective_D1_successor"]
        self.assertTrue(flagship["complete_first_call_artifact_visible_to_second_call"])
        self.assertTrue(flagship["complete_visible_verifier_feedback_visible_to_second_call"])
        self.assertIsNone(flagship["project_selected_quality_token_cap"])
        self.assertIsNone(flagship["project_selected_first_artifact_character_cap"])
        self.assertIsNone(flagship["project_selected_first_artifact_token_cap"])
        self.assertIsNone(flagship["project_selected_verifier_feedback_character_cap"])
        self.assertEqual(
            "complete_waiting_indefinitely_without_user_gate",
            successor["source_stage_implementation"],
        )
        self.assertEqual(
            "design_derived_206_repository_initial_metadata_frame_order_frozen_before_archive_fetch_then_first_44_independently_evaluator_qualified_before_candidate_calls",
            successor["source_stage_terminal_boundary"],
        )
        self.assertEqual(
            "complete_waiting_on_frozen_registry_no_user_gate",
            successor["source_materialization_implementation"],
        )
        self.assertEqual(
            "green_exact_local_denial_canaries",
            successor["evaluator_sandbox_qualification_state"],
        )
        self.assertFalse(
            successor[
                "untrusted_repository_execution_authorized_before_evaluator_seal"
            ]
        )
        self.assertEqual(
            "blind_campaign_independent_integrity_recomputation_exact_cost_custody_and_terminal_disposition_implemented_prospectively_autonomous_pipeline_controller_not_implemented_do_not_claim_D1_execution_ready",
            successor["downstream_evaluator_campaign_and_disposition_state"],
        )
        self.assertEqual(2, len(successor["remaining_D1_execution_owners"]))
        for key in (
            "autonomous_source_successor_config",
            "autonomous_source_successor",
            "autonomous_source_successor_test",
            "source_materialization_config",
            "source_materialization",
            "source_materialization_test",
            "evaluator_sandbox_config",
            "evaluator_sandbox",
            "evaluator_sandbox_test",
            "evaluator_sandbox_qualification",
            "evaluator_seal_config",
            "evaluator_seal",
            "evaluator_seal_test",
            "exact_prompt_token_counter",
            "exact_prompt_token_counter_test",
            "candidate_adapter",
            "candidate_adapter_test",
            "blind_evaluator",
            "blind_evaluator_test",
            "campaign_config",
            "campaign",
            "campaign_test",
            "terminal_disposition_config",
            "terminal_disposition",
            "terminal_disposition_test",
        ):
            self.assertTrue((ROOT / successor[key]).is_file())
        sandbox_report = json.loads(
            (ROOT / successor["evaluator_sandbox_qualification"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("GREEN", sandbox_report["trigger_state"])
        self.assertTrue(sandbox_report["untrusted_execution_authorized"])
        self.assertFalse(sandbox_report["faults"])
        self.assertTrue(all(sandbox_report["canary"].values()))
        self.assertFalse(sandbox_report["run_receipt"]["boundary_hit"])
        self.assertTrue(sandbox_report["run_receipt"]["stdout_complete"])
        self.assertTrue(sandbox_report["run_receipt"]["stderr_complete"])
        self.assertIsNone(
            sandbox_report["run_receipt"]["project_selected_character_cap"]
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / successor["evaluator_sandbox_config"]).read_bytes()
            ).hexdigest(),
            sandbox_report["config"]["sha256"],
        )
        self.assertFalse(successor["candidate_or_control_calls_authorized"])
        self.assertFalse(successor["automatic_book_support_promotion"])

    def test_high_leverage_claim_bindings_are_exact_and_decision_bounded(self) -> None:
        campaign = self.matrix["core_evidence_campaign"]
        policy = campaign["claim_binding_policy"]
        bindings = campaign["claim_test_bindings"]
        binding_ids = [row["claim_id"] for row in bindings]

        self.assertEqual(13, len(bindings))
        self.assertEqual(campaign["claim_ids"], binding_ids)
        self.assertEqual(13, len(set(binding_ids)))
        self.assertEqual(
            self.matrix["latest_ai_book_reconciliation"]["book_commit"],
            policy["source_book_commit"],
        )
        self.assertEqual(
            self.matrix["latest_ai_book_reconciliation"]["manifest_sha256"],
            policy["source_book_manifest_sha256"],
        )

        raw = subprocess.run(
            [
                "git",
                "-C",
                str(self.book_root),
                "show",
                f"{policy['source_book_commit']}:book_structure.json",
            ],
            check=True,
            capture_output=True,
        ).stdout
        manifest = json.loads(raw)
        book_claims = {
            f"{chapter['id']}.core": chapter["core_claim"]
            for part in manifest["parts"]
            for chapter in part["chapters"]
        }

        required_fields = {
            "claim_id",
            "core_claim_sha256",
            "role",
            "owner_work_package_id",
            "stage",
            "causal_variable",
            "primary_estimand",
            "faithful_mechanism",
            "adequacy_requirements",
            "matched_controls",
            "decision_rule",
            "maximum_inference_positive",
            "maximum_inference_negative",
        }
        for row in bindings:
            self.assertFalse(
                [field for field in required_fields if not row.get(field)],
                row["claim_id"],
            )
            self.assertIn(row["claim_id"], book_claims)
            self.assertEqual(
                hashlib.sha256(book_claims[row["claim_id"]].encode()).hexdigest(),
                row["core_claim_sha256"],
                row["claim_id"],
            )
            self.assertGreaterEqual(len(row["faithful_mechanism"]), 3)
            self.assertGreaterEqual(len(row["adequacy_requirements"]), 3)
            self.assertGreaterEqual(len(row["matched_controls"]), 3)

        self.assertEqual(
            2, sum(row["role"] == "integrity_prerequisite" for row in bindings)
        )
        self.assertEqual(
            8, sum(row["role"].startswith("p4_causal_candidate") for row in bindings)
        )
        self.assertEqual(
            1, sum(row["role"] == "independent_d2_neural_claim" for row in bindings)
        )
        self.assertEqual(
            2, sum(row["role"].endswith("synthesis_claim") for row in bindings)
        )
        self.assertEqual(
            policy["ordered_p4_candidate_claim_ids"],
            [row["claim_id"] for row in bindings if row["role"].startswith("p4")],
        )
        self.assertNotIn(
            "project-theseus-as-report-first-implementation-reference.core",
            binding_ids,
        )
        self.assertEqual(
            "none until independent book-side claim review accepts a claim-specific evidence transition",
            policy["global_decision_rule"]["SUPPORT_EFFECT"],
        )

    def test_claim_handoff_is_executable_pinned_and_non_authorizing(self) -> None:
        handoff = self.matrix["flagship_lane_governance"]["book_handoff_contract"]
        implementation = handoff["implementation"]

        self.assertEqual(
            self.matrix["latest_ai_book_reconciliation"]["book_commit"],
            implementation["book_pin_commit"],
        )
        self.assertEqual(
            "cognitive-compilation-and-semantic-ir.core",
            implementation["claim_id"],
        )
        self.assertEqual(
            "required_once_only_if_P4V2R2R3_survives",
            implementation["d1_requirement"],
        )
        self.assertEqual(
            "green_ready_for_governed_book_review_without_D1",
            implementation["state"],
        )
        self.assertEqual("GREEN", implementation["report_trigger_state"])
        self.assertEqual(
            "READY_FOR_GOVERNED_BOOK_REVIEW_WITHOUT_D1",
            implementation["activation_state"],
        )
        self.assertTrue(implementation["packet_ready"])
        self.assertTrue(implementation["public_safe_aggregate_only"])
        self.assertFalse(implementation["automatic_support_transition_proposed"])
        self.assertEqual("none", implementation["support_state_effect"])
        self.assertEqual("none", implementation["publication_authority"])
        self.assertEqual("none", implementation["release_authority"])
        for key in ("config", "builder", "test", "report"):
            self.assertTrue((ROOT / implementation[key]).is_file())

        report = self.audit(self.matrix)
        self.assertNotIn(
            "shared_flagship_book_handoff_implementation_invalid",
            self.gap_kinds(report),
        )

    def test_claim_handoff_support_laundering_fails_closed(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        implementation = matrix["flagship_lane_governance"][
            "book_handoff_contract"
        ]["implementation"]
        implementation["automatic_support_transition_proposed"] = True
        implementation["support_state_effect"] = "empirical-test-backed"

        report = self.audit(matrix)
        self.assertIn(
            "shared_flagship_book_handoff_implementation_invalid",
            self.gap_kinds(report),
        )


if __name__ == "__main__":
    unittest.main()
