from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import training_data_lineage_audit as audit


class TrainingDataLineageAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source_path = self.root / "private.jsonl"
        self.source_path.write_text("", encoding="utf-8")
        self.source = {
            "source_id": "private-source-v1",
            "path": str(self.source_path),
            "sha256": audit.file_sha256(self.source_path),
            "source_kind": "private_training_rows",
            "license_status": "allowed_cc0-1.0",
            "provenance_status": "project_internal_path_provenance",
        }
        self.public_fixture = {
            "prompt": "Compute a stable weighted total from signed integer records while skipping invalid labels.",
            "solution_body": "total = 0\nfor record in records:\n    if record.valid:\n        total += record.weight * abs(record.value)\nreturn total",
        }
        self.contamination = audit.contamination_index_from_rows([self.public_fixture])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def receipt(self, row: dict, *, index: int = 0, source: dict | None = None) -> dict:
        return audit.candidate_receipt(
            row,
            source=source or self.source,
            row_index=index,
            contamination_index=self.contamination,
        )

    def write_ledger(self, path: Path, receipts: list[dict]) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for receipt in receipts:
                handle.write(json.dumps(receipt, sort_keys=True) + "\n")

    def test_clean_candidate_is_admitted_without_storing_payload(self) -> None:
        row = {
            "prompt": "Return the median timestamp from a validated private event stream.",
            "solution_body": "values = sorted(events)\nreturn values[len(values) // 2]",
            "license_spdx": "CC0-1.0",
            "split": "train",
        }
        receipt = self.receipt(row)
        self.assertEqual("admit", receipt["decision"])
        self.assertTrue(receipt["authority"]["training_allowed"])
        self.assertFalse(receipt["raw_payload_stored"])
        self.assertNotIn(row["prompt"], json.dumps(receipt))
        self.assertEqual(audit.row_sha256(row), receipt["row_sha256"])

    def test_adversarial_controls_cover_all_policy_boundaries(self) -> None:
        controls = audit.run_adversary_controls()
        self.assertEqual(9, controls["case_count"])
        self.assertEqual(9, controls["passed_count"])
        self.assertFalse(controls["raw_fixture_text_emitted"])
        observed = {row["case_id"]: row["observed"] for row in controls["results"]}
        self.assertEqual("quarantine", observed["exact_overlap"])
        self.assertEqual("quarantine", observed["semantic_overlap"])
        for case_id in ("public_flag", "missing_license", "teacher_unverified", "raw_user", "fallback", "heldout"):
            self.assertEqual("reject", observed[case_id])

    def test_teacher_candidate_requires_governed_manifest_and_has_parent_lineage(self) -> None:
        manifest_path = self.root / "teacher.json"
        manifest = {
            "admission_safety_checks_clean": True,
            "public_overlap_hits": 0,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        row = {
            "request_id": "teacher-request-1",
            "license_spdx": "project-internal",
            "external_inference_calls": 1,
            "training_row": {
                "prompt": "Derive a private bounded queue invariant.",
                "solution_body": "return capacity >= size >= 0",
                "license_spdx": "project-internal",
            },
            "local_verifier": {"accepted": True},
            "admission_checks": {
                "leakage_audited": True,
                "license_checked": True,
                "provenance_retained": True,
                "public_benchmark_excluded": True,
                "runtime_serving_forbidden": True,
                "verifier_accepted": True,
            },
        }
        receipt = audit.teacher_candidate_receipt(
            row,
            row_index=0,
            manifest=manifest,
            manifest_path=manifest_path,
            contamination_index=self.contamination,
        )
        self.assertEqual("admit", receipt["decision"])
        self.assertTrue(receipt["provenance"]["lineage_complete"])
        self.assertTrue(receipt["provenance"]["parent_ref_hashes"])
        self.assertFalse(receipt["authority"]["runtime_direct_serving_allowed"])

        legacy = json.loads(json.dumps(receipt))
        legacy["provenance"]["lineage_complete"] = False
        legacy["provenance"]["parent_ref_hashes"] = []
        ledger = self.root / "teacher-ledger.jsonl.gz"
        self.write_ledger(ledger, [legacy])
        migrated = audit.upgrade_teacher_lineage_receipts(
            ledger,
            teacher_manifest={**manifest, "rows": [row]},
            teacher_manifest_path=manifest_path,
            contamination_index=self.contamination,
        )
        self.assertEqual(1, migrated)
        with gzip.open(ledger, "rt", encoding="utf-8") as handle:
            repaired = json.loads(next(handle))
        self.assertTrue(repaired["provenance"]["lineage_complete"])
        self.assertTrue(repaired["provenance"]["parent_ref_hashes"])

    def test_continual_policy_and_deletion_negative_control_are_replayable(self) -> None:
        workload = [
            {
                "candidate_id": f"candidate-{index}",
                "family": f"family-{index % 4}",
                "decision": "admit",
                "order": index + 1,
                "lineage_complete": True,
                "synthetic_depth": index % 3,
                "simulated_revoked": index == 3,
            }
            for index in range(40)
        ]
        comparison = audit.continual_learning_comparison(workload)
        self.assertTrue(comparison["comparison_ready"])
        self.assertEqual(5, comparison["policy_count"])
        self.assertEqual(
            {"replacement", "accumulation", "targeted_replay", "quarantine", "full_retraining"},
            {row["policy_id"] for row in comparison["policies"]},
        )
        deletion = audit.descendant_deletion_closure_fixture()
        self.assertEqual(11, deletion["artifact_kind_count"])
        self.assertTrue(deletion["positive_fixture_closed"])
        self.assertTrue(deletion["expected_invalid_fixture_rejected"])
        self.assertTrue(deletion["expected_invalid_fixture"]["unverified_descendants"])
        self.assertFalse(deletion["real_deletion_executed"])

    def test_canonical_lineage_owner_consumes_full_state_causality(self) -> None:
        report = audit.full_state_update_causality.run_reference_fixture()
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertTrue(report["rollback"]["exact_pre_state_restored"])
        self.assertNotEqual(
            report["summary"]["best_checkpoint_id"],
            report["summary"]["final_checkpoint_id"],
        )
        records = audit.build_viea_records(
            {
                "hard_gap_count": 0,
                "candidate_receipt_count": 1,
                "admitted_candidate_count": 1,
                "lineage_edge_count": 1,
                "exact_public_overlap_candidate_count": 0,
                "semantic_public_overlap_candidate_count": 0,
                "teacher_candidate_count": 0,
            },
            {"sha256": "a" * 64},
            {"recommended_policy": "targeted_replay", "workload_hash": "workload", "policy_count": 5, "comparison_ready": True},
            {"artifact_kind_count": 11, "positive_fixture_closed": True, "expected_invalid_fixture_rejected": True},
            {"passed_count": 9, "case_count": 9},
            report,
        )
        full_state_record = next(row for row in records if row["record_type"] == "full_state_update_transaction")
        self.assertTrue(full_state_record["exact_rollback"])
        self.assertFalse(full_state_record["behavioral_unlearning_claim_allowed"])

    def test_ledger_recovery_is_content_bound_and_rejects_duplicate_identity(self) -> None:
        row = {
            "prompt": "Return a private queue watermark.",
            "solution_body": "return max(values)",
            "license_spdx": "CC0-1.0",
        }
        receipt = self.receipt(row)
        ledger = self.root / "ledger.jsonl.gz"
        self.write_ledger(ledger, [receipt])
        recovered = audit.recover_ledger_state(
            ledger,
            source_rows=[self.source],
            teacher_manifest_path=self.root / "missing-teacher.json",
            contamination_digest=self.contamination["digest"],
            expected_count=1,
        )
        self.assertEqual(1, recovered["processed"])
        self.assertEqual(1, recovered["metrics"]["admit"])

        changed_source = dict(self.source, sha256="0" * 64)
        self.assertFalse(audit.recover_ledger_state(
            ledger,
            source_rows=[changed_source],
            teacher_manifest_path=self.root / "missing-teacher.json",
            contamination_digest=self.contamination["digest"],
            expected_count=1,
        ))
        self.write_ledger(ledger, [receipt, receipt])
        self.assertFalse(audit.recover_ledger_state(
            ledger,
            source_rows=[self.source],
            teacher_manifest_path=self.root / "missing-teacher.json",
            contamination_digest=self.contamination["digest"],
            expected_count=2,
        ))

    def test_admitted_hash_loader_rejects_ledger_mutation(self) -> None:
        admitted = self.receipt({
            "prompt": "Return a private queue watermark.",
            "solution_body": "return max(values)",
            "license_spdx": "CC0-1.0",
        })
        rejected = self.receipt({
            "prompt": "Private heldout queue watermark.",
            "solution_body": "return max(values)",
            "license_spdx": "CC0-1.0",
            "split": "test",
        }, index=1)
        ledger = self.root / "ledger.jsonl.gz"
        self.write_ledger(ledger, [admitted, rejected])
        admission = {"candidate_lineage": {"candidate_receipt_ledger": {
            "path": str(ledger),
            "sha256": audit.file_sha256(ledger),
        }}}
        self.assertEqual({admitted["row_sha256"]}, audit.load_admitted_candidate_hashes(admission))
        with ledger.open("ab") as handle:
            handle.write(b"tamper")
        self.assertEqual(set(), audit.load_admitted_candidate_hashes(admission))

    def test_incremental_rebuild_rescans_only_changed_source_group(self) -> None:
        rows_a = [{
            "prompt": "Return private value A.",
            "solution_body": "return value_a",
            "license_spdx": "CC0-1.0",
        }]
        rows_b = [{
            "prompt": "Return private value B.",
            "solution_body": "return value_b",
            "license_spdx": "CC0-1.0",
        }]
        path_a = self.root / "a.jsonl"
        path_b = self.root / "b.jsonl"
        path_a.write_text("\n".join(json.dumps(row) for row in rows_a) + "\n", encoding="utf-8")
        path_b.write_text("\n".join(json.dumps(row) for row in rows_b) + "\n", encoding="utf-8")
        source_a = dict(self.source, source_id="source-a", path=str(path_a), sha256=audit.file_sha256(path_a), row_count=1)
        source_b = dict(self.source, source_id="source-b", path=str(path_b), sha256=audit.file_sha256(path_b), row_count=1)
        ledger = self.root / "incremental.jsonl.gz"
        self.write_ledger(ledger, [
            audit.candidate_receipt(rows_a[0], source=source_a, row_index=0, contamination_index=self.contamination),
            audit.candidate_receipt(rows_b[0], source=source_b, row_index=0, contamination_index=self.contamination),
        ])
        rows_b.append({
            "prompt": "Return private value B2.",
            "solution_body": "return value_b2",
            "license_spdx": "CC0-1.0",
        })
        path_b.write_text("\n".join(json.dumps(row) for row in rows_b) + "\n", encoding="utf-8")
        source_b = dict(source_b, sha256=audit.file_sha256(path_b), row_count=2)
        teacher_manifest = self.root / "teacher-empty.json"
        teacher_manifest.write_text(json.dumps({"rows": []}), encoding="utf-8")
        result = audit.incremental_rebuild_ledger(
            ledger,
            source_rows=[source_a, source_b],
            teacher_manifest={"rows": []},
            teacher_manifest_path=teacher_manifest,
            contamination_index=self.contamination,
            expected_count=3,
        )
        self.assertEqual(3, result["processed"])
        self.assertEqual(1, result["reused_source_count"])
        self.assertEqual(2, result["rescanned_source_count"])
        self.assertEqual(1, result["reused_receipt_count"])
        self.assertEqual(2, result["rescanned_receipt_count"])
        self.assertTrue(result["ledger_receipt"]["replay_valid"])


if __name__ == "__main__":
    unittest.main()
