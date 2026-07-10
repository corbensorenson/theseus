from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from governed_conversation_stream import (  # noqa: E402
    AtomicConversationShardWriter,
    ConversationDeduper,
    admit_conversation,
    reconstruct_oasst_conversations,
    redact_sensitive_text,
    source_plan,
)


EMPTY_PUBLIC_INDEX = {
    "exact_hashes": set(),
    "entries": [],
    "buckets": {},
    "digest": "fixture",
    "text_count": 1,
}


class GovernedConversationStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "id": "human_fixture",
            "dataset": "fixture/human",
            "license_spdx": "apache-2.0",
            "provenance_class": "human_contributed",
        }

    def test_sensitive_literals_are_redacted_before_admission(self) -> None:
        text, counts = redact_sensitive_text(
            "Email me at person@example.com or call +1 (555) 222-9876. api_key=abcdEFGH12345678"
        )
        self.assertNotIn("person@example.com", text)
        self.assertNotIn("222-9876", text)
        self.assertNotIn("abcdEFGH12345678", text)
        self.assertEqual(1, counts["email"])
        self.assertEqual(1, counts["secret"])

    def test_admission_emits_receipt_and_never_persists_raw_sensitive_text(self) -> None:
        result = admit_conversation(
            [
                {"role": "user", "content": "Explain how to sort a list. My email is person@example.com."},
                {"role": "assistant", "content": "Use sorted(values) for a new list, or values.sort() to mutate it."},
            ],
            source=self.source,
            provenance={"adapter": "fixture"},
            public_index=EMPTY_PUBLIC_INDEX,
            deduper=ConversationDeduper(),
            max_chars=4000,
        )
        self.assertEqual("admit", result["receipt"]["decision"])
        self.assertFalse(result["receipt"]["raw_unredacted_text_persisted"])
        self.assertNotIn("person@example.com", result["train_row"]["causal_text"])
        self.assertEqual(
            result["receipt"]["receipt_id"],
            result["train_row"]["data_admission_receipt_id"],
        )

    def test_exact_and_near_duplicates_are_quarantined(self) -> None:
        deduper = ConversationDeduper(max_hamming_distance=6)
        messages = [
            {"role": "user", "content": "Describe deterministic sorting for integer values."},
            {"role": "assistant", "content": "Sort a copied list with sorted(values), which leaves the original unchanged."},
        ]
        first = admit_conversation(
            messages,
            source=self.source,
            provenance={},
            public_index=EMPTY_PUBLIC_INDEX,
            deduper=deduper,
            max_chars=4000,
        )
        second = admit_conversation(
            messages,
            source=self.source,
            provenance={},
            public_index=EMPTY_PUBLIC_INDEX,
            deduper=deduper,
            max_chars=4000,
        )
        self.assertEqual("admit", first["receipt"]["decision"])
        self.assertEqual("quarantine", second["receipt"]["decision"])
        self.assertIn("exact_duplicate", second["receipt"]["decision_reasons"])

    def test_oasst_tree_reconstruction_preserves_ranked_ancestry(self) -> None:
        rows = [
            {"message_id": "u1", "parent_id": None, "role": "prompter", "text": "Explain sorting.", "lang": "en", "review_result": True, "deleted": False, "rank": None},
            {"message_id": "a1", "parent_id": "u1", "role": "assistant", "text": "Sorting orders values by a key.", "lang": "en", "review_result": True, "deleted": False, "rank": 0},
            {"message_id": "a2", "parent_id": "u1", "role": "assistant", "text": "Low-ranked response.", "lang": "en", "review_result": True, "deleted": False, "rank": 1},
            {"message_id": "u2", "parent_id": "a1", "role": "prompter", "text": "Show Python.", "lang": "en", "review_result": True, "deleted": False, "rank": None},
            {"message_id": "a3", "parent_id": "u2", "role": "assistant", "text": "Use sorted(values) to return a copy.", "lang": "en", "review_result": True, "deleted": False, "rank": 0},
        ]
        conversations = list(
            reconstruct_oasst_conversations(
                rows,
                {"required_langs": ["en"], "max_assistant_rank": 0, "max_scan_rows": 100},
            )
        )
        self.assertEqual(2, len(conversations))
        longest, provenance = conversations[-1]
        self.assertEqual(["user", "assistant", "user", "assistant"], [row["role"] for row in longest])
        self.assertFalse(provenance["raw_message_ids_emitted"])

    def test_atomic_shards_resume_without_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = AtomicConversationShardWriter(root, shard_rows=1, max_disk_bytes=1_000_000)
            receipt = {"receipt_id": "r1", "metrics": {"token_positions": 12}}
            writer.add({"task_id": "t1", "causal_text": "<|user|> hello\n<|assistant|> a sufficiently long answer"}, {"task_id": "s1"}, receipt)
            writer.mark_source_complete("source-a")
            writer.close()
            resumed = AtomicConversationShardWriter(root, shard_rows=1, max_disk_bytes=1_000_000)
            self.assertEqual(1, resumed.total_rows)
            self.assertEqual(12, resumed.total_token_positions)
            self.assertIn("source-a", resumed.completed_sources)
            manifest = json.loads((root / "scalable_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(1, len(manifest["shards"]))

    def test_external_teacher_source_fails_closed_without_gate(self) -> None:
        plan = source_plan(
            {
                "id": "teacher",
                "dataset": "fixture/teacher",
                "license_spdx": "mit",
                "provenance_class": "external_teacher_generated",
            },
            allowed_licenses={"mit"},
            teacher_allowed=False,
        )
        self.assertEqual("teacher_distillation_gate_not_admitted_for_conversation", plan["decision"])


if __name__ == "__main__":
    unittest.main()
