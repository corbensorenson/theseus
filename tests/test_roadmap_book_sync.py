#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
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
        self.assertEqual(54, summary["book_manifest_chapter_count"])
        self.assertEqual(511, summary["book_codex_test_count"])
        self.assertEqual(109, summary["book_pending_or_partial_codex_test_count"])

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


if __name__ == "__main__":
    unittest.main()
