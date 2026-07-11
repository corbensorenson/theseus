from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
