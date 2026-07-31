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


if __name__ == "__main__":
    unittest.main()
