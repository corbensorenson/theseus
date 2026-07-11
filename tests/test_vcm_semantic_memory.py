from __future__ import annotations

import sys
import json
import subprocess
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


class DurableSemanticMemoryTests(unittest.TestCase):
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
        self.assertEqual(receipt["ontology_version"], "1.0.0")
        self.assertEqual(receipt["migration"]["from_version"], "0.9.0")
        self.assertEqual(receipt["migration"]["to_version"], "1.0.0")


if __name__ == "__main__":
    unittest.main()
