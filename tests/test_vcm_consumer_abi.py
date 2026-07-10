from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import vcm_consumer_abi as abi


def governor_payload() -> dict:
    return {
        "trigger_state": "GREEN",
        "created_utc": "2026-07-10T00:00:00Z",
        "summary": {
            "hard_gap_count": 0,
            "mission_brief_status": "ready",
            "deletion_closure_status": "closed",
            "deletion_closure_fault_count": 0,
            "scif_status": "ready",
            "context_abi_fixture_status": "ready",
            "context_abi_fixture_count": 5,
            "context_abi_fixture_passed_count": 5,
            "context_resolver_status": "ready",
            "context_resolver_request_count": 7,
            "context_resolver_passed_count": 7,
            "representation_certificate_status": "ready",
            "representation_certificate_count": 7,
            "representation_certificate_passed_count": 7,
            "snapshot_branch_status": "ready",
            "snapshot_branch_count": 7,
            "snapshot_branch_passed_count": 7,
        },
    }


class VcmConsumerAbiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.governor = self.root / "governor.json"
        self.semantic_index = self.root / "semantic_index.json"
        self.source = self.root / "source.json"
        self.output = self.root / "out.json"
        self.governor.write_text(json.dumps(governor_payload()), encoding="utf-8")
        self.source.write_text("{}", encoding="utf-8")
        self.semantic_index.write_text(json.dumps({
            "policy": "test_vcm_index",
            "pages": [{
                "address": "vcm://test/context/source@v1",
                "aliases": [],
                "source_path": str(self.source),
                "status": "active",
                "model_visible": True,
                "taints": [],
            }],
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def packet(self, **overrides):
        kwargs = {
            "consumer_id": "unit.consumer",
            "purpose": "unit_test",
            "read_set": [str(self.source)],
            "write_set": [str(self.output)],
            "authority_ceiling": ["local_read"],
            "permitted_uses": ["unit_test_read"],
            "governor_path": self.governor,
            "semantic_index_path": self.semantic_index,
            "now_utc": datetime(2026, 7, 10, 0, 0, tzinfo=timezone.utc),
        }
        kwargs.update(overrides)
        return abi.build_consumer_packet(**kwargs)

    def test_positive_packet_is_replayable_and_complete(self):
        packet = self.packet()
        self.assertTrue(packet["ready"])
        self.assertTrue(packet["validation"]["passed"])
        self.assertTrue(packet["snapshot_branch"]["copy_on_write"])
        self.assertFalse(packet["representation_certificate"]["consumer_policy"]["best_effort_materialization_allowed"])
        self.assertEqual(
            {"context_abi_record", "context_transaction", "context_adequacy", "context_lease_receipt", "authority_use_receipt", "failure_boundary"},
            {row["record_type"] for row in packet["records"]},
        )

    def test_missing_required_context_fails_closed(self):
        packet = self.packet(read_set=[str(self.root / "missing.json")])
        self.assertFalse(packet["ready"])
        self.assertIn("CONTEXT_REQUIRED_MISSING", packet["typed_faults"])

    def test_stale_context_fails_closed(self):
        packet = self.packet(
            context_refs=[{
                "ref": str(self.source),
                "exists": True,
                "required": True,
                "created_utc": "2026-07-09T00:00:00Z",
                "max_age_seconds": 60,
            }]
        )
        self.assertFalse(packet["ready"])
        self.assertIn("CONTEXT_REQUIRED_STALE", packet["typed_faults"])

    def test_tainted_training_context_fails_closed(self):
        packet = self.packet(taint_labels=["public_benchmark_payload"])
        self.assertFalse(packet["ready"])
        self.assertIn("CONTEXT_TAINT_DENIED", packet["typed_faults"])

    def test_contradiction_fails_closed(self):
        packet = self.packet(contradiction_refs=["claim://conflict"])
        self.assertFalse(packet["ready"])
        self.assertIn("CONTEXT_CONTRADICTION_UNRESOLVED", packet["typed_faults"])

    def test_revoked_or_deleted_context_fails_closed(self):
        for field in ("revoked", "deleted"):
            with self.subTest(field=field):
                packet = self.packet(context_refs=[{
                    "ref": str(self.source),
                    "exists": True,
                    "required": True,
                    field: True,
                }])
                self.assertFalse(packet["ready"])
                self.assertIn("CONTEXT_REVOKED_OR_DELETED", packet["typed_faults"])

    def test_semantic_address_is_resolved_independently(self):
        packet = self.packet(context_refs=[{
            "kind": "semantic_address",
            "ref": "vcm://test/context/source@v1",
            "required": True,
            "exists": False,
        }])
        self.assertTrue(packet["ready"])
        source_ref = packet["representation_certificate"]["source_refs"][0]
        self.assertTrue(source_ref["resolved_from_index"])
        self.assertTrue(source_ref["exists"])
        self.assertTrue(source_ref["source_sha256"])

    def test_forged_semantic_exists_flag_fails_closed(self):
        packet = self.packet(context_refs=[{
            "kind": "semantic_address",
            "ref": "vcm://test/context/forged@v1",
            "required": True,
            "exists": True,
        }])
        self.assertFalse(packet["ready"])
        self.assertIn("CONTEXT_SEMANTIC_REF_UNRESOLVED", packet["typed_faults"])

    def test_index_taint_cannot_be_dropped_by_consumer(self):
        payload = json.loads(self.semantic_index.read_text(encoding="utf-8"))
        payload["pages"][0]["taints"] = ["public_benchmark_payload"]
        self.semantic_index.write_text(json.dumps(payload), encoding="utf-8")
        packet = self.packet(context_refs=[{
            "kind": "semantic_address",
            "ref": "vcm://test/context/source@v1",
            "required": True,
            "exists": True,
            "taint_labels": [],
        }])
        self.assertFalse(packet["ready"])
        self.assertIn("CONTEXT_TAINT_DENIED", packet["typed_faults"])

    def test_over_compression_fails_closed(self):
        packet = self.packet(compression_loss=0.6, max_compression_loss=0.35)
        self.assertFalse(packet["ready"])
        self.assertIn("CONTEXT_OVER_COMPRESSED", packet["typed_faults"])

    def test_authority_widening_is_rejected_independently(self):
        packet = self.packet()
        tampered = copy.deepcopy(packet)
        tampered["representation_certificate"]["materialized_authority_labels"].append("network_write")
        result = abi.validate_consumer_packet(tampered)
        self.assertFalse(result["passed"])
        self.assertIn("CONTEXT_AUTHORITY_WIDENING", result["faults"])

    def test_source_mutation_is_rejected(self):
        packet = self.packet(write_set=[str(self.source)])
        self.assertFalse(packet["ready"])
        self.assertIn("CONTEXT_SOURCE_MUTATION_ATTEMPT", packet["typed_faults"])

    def test_governor_partial_certificate_coverage_fails_closed(self):
        payload = governor_payload()
        payload["summary"]["representation_certificate_passed_count"] = 6
        self.governor.write_text(json.dumps(payload), encoding="utf-8")
        packet = self.packet()
        self.assertFalse(packet["ready"])
        self.assertIn("VCM_GOVERNOR_REPRESENTATION_CERTIFICATE_PASSED_COUNT_INCOMPLETE", packet["typed_faults"])

    def test_compact_receipt_preserves_audit_identity_without_full_payload(self):
        packet = self.packet()
        receipt = abi.compact_consumer_packet(packet)
        self.assertTrue(receipt["ready"])
        self.assertEqual(packet["packet_id"], receipt["packet_id"])
        self.assertEqual(packet["representation_certificate"]["certificate_id"], receipt["representation_certificate_id"])
        self.assertEqual(1, receipt["source_ref_count"])
        self.assertNotIn("representation_certificate", receipt)
        self.assertNotIn("records", receipt)


if __name__ == "__main__":
    unittest.main()
