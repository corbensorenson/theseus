from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import vcm_semantic_memory as semantic  # noqa: E402


def page(address: str, text: str, *, page_type: str = "evidence", status: str = "active") -> dict:
    source_path = f"memory/{address.rsplit('/', 1)[-1].split('@', 1)[0]}.md"
    return {
        "address": address,
        "immutable_version": address.rsplit("@", 1)[-1],
        "content_hash": f"sha256:{text}",
        "type": page_type,
        "execution_class": "evidence_observation",
        "status": status,
        "title": text,
        "source": {
            "source_path": source_path,
            "source_hash": f"sha256:source-{text}",
            "source_role": "private_test_fixture",
        },
        "representations": {
            "L1": {"materialized_text": text},
            "L2": {"materialized_text": text},
            "L3": {"materialized_text": text},
        },
        "taints": [],
    }


def production_page(address: str, text: str, *, source_path: str, page_type: str = "evidence") -> dict:
    row = page(address, text, page_type=page_type)
    source = row.pop("source")
    source["source_path"] = source_path
    row["authoritative_sources"] = [source]
    row["metadata"] = {"semantic_identity_kind": "policy" if page_type == "policy" else "proposition"}
    return row


class DurableSemanticMemoryTests(unittest.TestCase):
    def test_qcsa_identity_address_and_route_are_distinct_and_authority_safe(self) -> None:
        first_page = production_page(
            "vcm://theseus/index-a/current-wall@v1",
            "current wall",
            source_path="docs/PROJECT_STATE.md",
            page_type="policy",
        )
        first = semantic.build_semantic_memory(
            [first_page],
            {"edges": [], "invalidation": {"invalidated_addresses": []}},
            task="inspect current wall",
        )
        second_page = production_page(
            "vcm://theseus/index-b/moved-wall@v2",
            "current wall revised",
            source_path="docs/PROJECT_STATE.md",
            page_type="policy",
        )
        second = semantic.build_semantic_memory(
            [second_page],
            {"edges": [], "invalidation": {"invalidated_addresses": []}},
            previous=first,
            task="inspect current wall",
        )

        soid = second["objects"][0]["semantic_object_id"]
        self.assertTrue(soid.startswith("soid:"))
        self.assertEqual(first["objects"][0]["semantic_object_id"], soid)
        self.assertNotIn("index-b", soid)
        self.assertTrue(second["identity_registry"]["identity_is_separate_from_address"])
        self.assertEqual(len(second["semantic_address_atlas"]["facets"]), 3)
        certificate = second["semantic_address_certificates"][0]
        verified = semantic.verify_semantic_address_certificate(
            certificate,
            second,
            requested_use="context_retrieval",
            requested_authority="read_context",
            expected_task="inspect current wall",
            expected_consumer="vcm_context_compiler",
        )
        self.assertEqual(verified["soid"], soid)
        route = semantic.translate_semantic_address(second, certificate)
        self.assertEqual(route["physical_address"], second_page["address"])
        self.assertFalse(route["effect_authority_granted"])
        self.assertTrue(route["requires_separate_scf_effect_authorization"])
        superseded = copy.deepcopy(second)
        superseded["objects"][0]["lifecycle_state"] = "superseded"
        superseded_certificate = copy.deepcopy(certificate)
        superseded_certificate["validity"]["lifecycle_state"] = "superseded"
        unsigned = copy.deepcopy(superseded_certificate)
        unsigned.pop("certificate_id")
        unsigned.pop("certificate_digest")
        superseded_certificate["certificate_digest"] = semantic._digest(unsigned)
        superseded_certificate["certificate_id"] = semantic._stable_id(
            "sac", superseded_certificate["certificate_digest"]
        )
        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_ROUTE_LIFECYCLE_DENIED"):
            semantic.translate_semantic_address(superseded, superseded_certificate)
        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_AUTHORITY_CEILING_EXCEEDED"):
            semantic.translate_semantic_address(
                second,
                certificate,
                requested_authority="release_effect",
            )

    def test_qcsa_packet_occurrences_do_not_collapse_on_shared_backing_report(self) -> None:
        first = production_page(
            "vcm://theseus/context/packet-a@v1",
            "packet a",
            source_path="reports/autonomy_cycle_last.json",
        )
        second = production_page(
            "vcm://theseus/context/packet-b@v1",
            "packet b",
            source_path="reports/autonomy_cycle_last.json",
        )
        first["metadata"].update({"source_kind": "context_packet", "packet_id": "packet-a"})
        second["metadata"].update({"source_kind": "context_packet", "packet_id": "packet-b"})
        state = semantic.build_semantic_memory(
            [first, second],
            {"edges": [], "invalidation": {"invalidated_addresses": []}},
            task="packet identity",
        )
        self.assertEqual(len(state["objects"]), 2)
        self.assertEqual(len({row["semantic_object_id"] for row in state["objects"]}), 2)
        self.assertEqual(state["identity_registry"]["object_count"], 2)

        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_ADDRESS_COLLISION"):
            semantic.build_semantic_memory(
                [first, copy.deepcopy(first)],
                {"edges": [], "invalidation": {"invalidated_addresses": []}},
                task="duplicate physical address",
            )

        retargeted = copy.deepcopy(first)
        retargeted["metadata"]["packet_id"] = "packet-retargeted"
        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_SILENT_ADDRESS_RETARGET"):
            semantic.build_semantic_memory(
                [retargeted, second],
                {"edges": [], "invalidation": {"invalidated_addresses": []}},
                previous=state,
                task="packet identity",
            )

    def test_qcsa_certificate_tamper_stale_epoch_and_consumer_laundering_fail_closed(self) -> None:
        state = semantic.build_semantic_memory(
            [production_page("vcm://theseus/evidence/a@v1", "evidence a", source_path="reports/a.json")],
            {"edges": [], "invalidation": {"invalidated_addresses": []}},
            task="review evidence",
        )
        certificate = state["semantic_address_certificates"][0]
        tampered = copy.deepcopy(certificate)
        tampered["allowed_uses"].append("effect_authorization")
        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_CERTIFICATE_DIGEST_INVALID"):
            semantic.verify_semantic_address_certificate(
                tampered,
                state,
                requested_use="context_retrieval",
                requested_authority="read_context",
            )

        stale = copy.deepcopy(certificate)
        stale["atlas_epoch"] = "vcm-atlas-stale"
        unsigned = copy.deepcopy(stale)
        unsigned.pop("certificate_id")
        unsigned.pop("certificate_digest")
        stale["certificate_digest"] = semantic._digest(unsigned)
        stale["certificate_id"] = semantic._stable_id("sac", stale["certificate_digest"])
        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_CERTIFICATE_EPOCH_STALE"):
            semantic.verify_semantic_address_certificate(
                stale,
                state,
                requested_use="context_retrieval",
                requested_authority="read_context",
            )

        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_CERTIFICATE_CONSUMER_MISMATCH"):
            semantic.verify_semantic_address_certificate(
                certificate,
                state,
                requested_use="context_retrieval",
                requested_authority="read_context",
                expected_consumer="effect_executor",
            )

        stale_object = copy.deepcopy(state)
        stale_object["objects"][0]["current_revision"]["content_hash"] = "sha256:stale"
        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_CERTIFICATE_OBJECT_STATE_STALE"):
            semantic.verify_semantic_address_certificate(
                certificate,
                stale_object,
                requested_use="context_retrieval",
                requested_authority="read_context",
            )

        tampered_atlas = copy.deepcopy(state)
        soid = certificate["soid"]
        tampered_atlas["semantic_address_atlas"]["paths"][soid]["task_retrieval"][0] += "-tampered"
        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_ATLAS_DIGEST_INVALID"):
            semantic.verify_semantic_address_certificate(
                certificate,
                tampered_atlas,
                requested_use="context_retrieval",
                requested_authority="read_context",
            )

    def test_qcsa_atlas_migration_preserves_soid_and_rolls_back_exactly(self) -> None:
        original_page = production_page(
            "vcm://theseus/atlas-a/object@v1",
            "object",
            source_path="memory/object.json",
        )
        state = semantic.build_semantic_memory(
            [original_page],
            {"edges": [], "invalidation": {"invalidated_addresses": []}},
            task="migrate object",
        )
        soid = state["objects"][0]["semantic_object_id"]
        migrated, receipt = semantic.migrate_semantic_atlas(
            state,
            target_epoch="vcm-atlas-1.1.0",
            changes=[
                {
                    "mode": "readdress",
                    "soid": soid,
                    "new_address": "vcm://theseus/atlas-b/object@v2",
                },
                {"mode": "fail", "reason": "orphan_address"},
            ],
            inventory={
                "descendants": ["context-packet:a"],
                "caches": ["c-tlb:a"],
                "backups": ["snapshot:a"],
                "receipts": ["receipt:a"],
            },
            shadow_passed=True,
        )
        self.assertEqual(migrated["objects"][0]["semantic_object_id"], soid)
        self.assertEqual(migrated["objects"][0]["current_address"], "vcm://theseus/atlas-b/object@v2")
        self.assertTrue(receipt["same_soid_preserved"])
        self.assertEqual(len(receipt["typed_failures"]), 1)
        restored, rollback = semantic.rollback_semantic_atlas(migrated, receipt)
        self.assertTrue(rollback["matches_pre_migration"])
        self.assertEqual(semantic._qcsa_state_digest(restored), semantic._qcsa_state_digest(state))

        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_SILENT_MIGRATION_RETARGET"):
            semantic.migrate_semantic_atlas(
                state,
                target_epoch="vcm-atlas-1.1.0",
                changes=[
                    {
                        "mode": "readdress",
                        "soid": soid,
                        "new_soid": "soid:" + "0" * 24,
                        "new_address": "vcm://theseus/atlas-b/object@v2",
                    }
                ],
                inventory={"descendants": ["a"], "caches": ["a"], "backups": ["a"], "receipts": ["a"]},
                shadow_passed=True,
            )
        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_MIGRATION_INVENTORY_INCOMPLETE"):
            semantic.migrate_semantic_atlas(
                state,
                target_epoch="vcm-atlas-1.1.0",
                changes=[],
                inventory={"descendants": [], "caches": ["a"], "backups": ["a"], "receipts": ["a"]},
                shadow_passed=True,
            )

    def test_qcsa_disposition_is_content_bound_and_does_not_hide_replay_fault(self) -> None:
        config = semantic.load_qcsa_config()
        self.assertIn("adaptive_active_question_policy", config["retired_from_first_long_run"])
        self.assertEqual(config["evidence"]["replay_state"]["evaluation_byte_replay"], "RED_ONE_MICRO_ROUNDING_DRIFT")

        tampered = copy.deepcopy(config)
        tampered["evidence"]["measurements"]["operation_ratio"] = 1.0
        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_EVIDENCE_MEASUREMENTS_INVALID"):
            semantic.validate_qcsa_config(tampered)
        tampered = copy.deepcopy(config)
        tampered["evidence"]["replay_state"]["evaluation_byte_replay"] = "GREEN"
        with self.assertRaisesRegex(semantic.QCSAFault, "VCM_QCSA_REPLAY_BOUNDARY_INVALID"):
            semantic.validate_qcsa_config(tampered)

    def test_object_identity_survives_revision_and_restart(self) -> None:
        first = semantic.build_semantic_memory(
            [page("vcm://theseus/project/current-wall@v1", "current project wall policy")],
            {"edges": [], "invalidation": {"invalidated_addresses": []}},
            task="inspect current wall",
        )
        second = semantic.build_semantic_memory(
            [page("vcm://theseus/project/current-wall@v2", "updated current project wall policy")],
            {"edges": [], "invalidation": {"invalidated_addresses": []}},
            previous=first,
            task="inspect current wall",
        )

        self.assertEqual(first["objects"][0]["semantic_object_id"], second["objects"][0]["semantic_object_id"])
        self.assertEqual(len(second["objects"][0]["revision_history"]), 2)
        self.assertTrue(second["restart_replay"]["state_digest_match"])
        self.assertTrue(second["restart_replay"]["query_replay_match"])
        self.assertTrue(second["ontology_migrations"][0]["preserves_object_ids"])

    def test_retracted_and_poisoned_memory_fails_closed_in_retrieval(self) -> None:
        active = page("vcm://theseus/policy/charter@v1", "trusted charter governance", page_type="policy")
        poisoned = page("vcm://theseus/evidence/poisoned@v1", "trusted charter secret", status="quarantined")
        state = semantic.build_semantic_memory(
            [active, poisoned],
            {
                "edges": [],
                "invalidation": {"invalidated_addresses": [active["address"]]},
            },
            task="trusted charter",
        )

        self.assertEqual(semantic.query_semantic_memory(state, "trusted charter"), [])
        lifecycle = {row["current_address"]: row["lifecycle_state"] for row in state["objects"]}
        self.assertEqual(lifecycle[active["address"]], "retracted")
        self.assertEqual(lifecycle[poisoned["address"]], "quarantined")
        actions = {row["semantic_object_id"]: row["action"] for row in state["consolidation_records"]}
        active_object_id = next(
            row["semantic_object_id"] for row in state["objects"] if row["current_address"] == active["address"]
        )
        self.assertIn("tombstone", actions[active_object_id])

    def test_hybrid_retrieval_is_explainable_and_snapshot_is_bounded(self) -> None:
        policy = page("vcm://theseus/policy/charter@v1", "project charter governance", page_type="policy")
        evidence = page("vcm://theseus/evidence/run@v1", "training run evidence")
        state = semantic.build_semantic_memory(
            [policy, evidence],
            {
                "edges": [
                    {"from": policy["address"], "to": evidence["address"], "type": "supports", "created_utc": "2026-01-01T00:00:00Z"}
                ],
                "invalidation": {"invalidated_addresses": []},
            },
            task="project charter",
        )

        results = semantic.query_semantic_memory(state, "project charter")
        self.assertEqual(results[0]["address"], policy["address"])
        self.assertGreater(results[0]["score_components"]["sparse_bm25"], 0)
        self.assertIn("graph_degree", results[0]["score_components"])
        self.assertLessEqual(len(state["bounded_snapshot"]["object_ids"]), 32)
        self.assertFalse(state["claims"]["dense_embedding_retrieval"])
        self.assertFalse(state["claims"]["parametric_unlearning"])

    def test_lifecycle_merge_compaction_and_rejection_are_transactional(self) -> None:
        target = page("vcm://theseus/memory/target@v1", "merged durable target", page_type="policy")
        source = page("vcm://theseus/memory/source@v1", "legacy source evidence")
        state = semantic.build_semantic_memory(
            [target, source],
            {"edges": [], "invalidation": {"invalidated_addresses": []}},
            task="merge memory",
        )
        by_address = {row["current_address"]: row for row in state["objects"]}
        target_id = by_address[target["address"]]["semantic_object_id"]
        source_id = by_address[source["address"]]["semantic_object_id"]

        objects, relations, transactions = semantic.apply_lifecycle_transactions(
            state["objects"],
            state["relations"],
            [
                {"operation": "merge", "target_object_id": target_id, "source_object_ids": [source_id], "reason": "test_merge"},
                {"operation": "compact", "target_object_id": target_id, "reason": "invalid_hot_compaction"},
            ],
        )

        updated = {row["semantic_object_id"]: row for row in objects}
        self.assertEqual(updated[source_id]["lifecycle_state"], "superseded")
        self.assertEqual(updated[source_id]["merged_into_object_id"], target_id)
        self.assertEqual(updated[target_id]["merge_source_object_ids"], [source_id])
        self.assertEqual(transactions[0]["status"], "committed")
        self.assertEqual(transactions[1]["status"], "rejected")
        self.assertEqual(transactions[1]["typed_failure"], "compaction_requires_cold_tier")
        self.assertTrue(any(row["relation_type"] == "derived_from" for row in relations))

    def test_prior_ontology_persists_through_fresh_process_replay(self) -> None:
        old = semantic.build_semantic_memory(
            [page("vcm://theseus/project/state@v1", "current project policy")],
            {"edges": [], "invalidation": {"invalidated_addresses": []}},
            task="current project",
        )
        old["ontology"]["version"] = "0.9.0"
        migrated = semantic.build_semantic_memory(
            [page("vcm://theseus/project/state@v2", "current project policy revised")],
            {"edges": [], "invalidation": {"invalidated_addresses": []}},
            previous=old,
            task="current project",
        )
        self.assertEqual(migrated["ontology_migrations"][0]["from_version"], "0.9.0")
        self.assertEqual(migrated["ontology_migrations"][0]["mode"], "additive_projection")

        with tempfile.TemporaryDirectory() as temp:
            graph_path = Path(temp) / "graph.json"
            pages_path = Path(temp) / "pages.json"
            receipt_path = Path(temp) / "receipt.json"
            graph_path.write_text(
                json.dumps({"semantic_memory": old, "edges": [], "invalidation": {"invalidated_addresses": []}}),
                encoding="utf-8",
            )
            pages_path.write_text(
                json.dumps([page("vcm://theseus/project/state@v2", "current project policy revised")]),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "vcm_semantic_memory.py"),
                    "migrate-replay",
                    "--graph",
                    str(graph_path),
                    "--pages",
                    str(pages_path),
                    "--out",
                    str(receipt_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

        self.assertTrue(receipt["process_replay"])
        self.assertTrue(receipt["state_digest_match"])
        self.assertTrue(receipt["query_replay_match"])
        self.assertEqual(receipt["ontology_version"], "1.1.0")
        self.assertEqual(receipt["migration"]["from_version"], "0.9.0")
        self.assertEqual(receipt["migration"]["to_version"], "1.1.0")


if __name__ == "__main__":
    unittest.main()
