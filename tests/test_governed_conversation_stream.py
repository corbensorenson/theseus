from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from governed_conversation_stream import (  # noqa: E402
    AtomicConversationShardWriter,
    AtomicDocumentShardWriter,
    ConversationDeduper,
    admit_conversation,
    admit_document,
    chunk_document,
    document_source_plan,
    reconstruct_oasst_conversations,
    run_document_intake,
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

    def test_high_quality_static_model_corpus_is_data_not_live_teacher_inference(self) -> None:
        plan = source_plan(
            {
                "id": "licensed-static",
                "dataset": "fixture/static",
                "license_spdx": "apache-2.0",
                "provenance_class": "external_teacher_generated",
                "static_open_corpus": True,
                "quality_tier": "curated_high",
            },
            allowed_licenses={"apache-2.0"},
            teacher_allowed=False,
            licensed_static_policy={
                "enabled": True,
                "required_quality_tiers": ["curated_high"],
                "static_open_corpus_required": True,
                "live_provider_calls_allowed": False,
            },
        )
        self.assertEqual("eligible_for_intake", plan["decision"])
        self.assertTrue(plan["static_model_corpus_allowed"])

    def test_public_domain_document_admission_is_pretraining_only_and_deduplicated(self) -> None:
        source = {
            "id": "gutenberg_fixture",
            "dataset": "fixture/gutenberg",
            "license_spdx": "public-domain",
            "provenance_class": "human_public_domain",
        }
        provenance = {
            "source_url": "https://www.gutenberg.org/ebooks/1",
            "title": "Fixture",
            "author": "Public Domain Author",
            "row_license": "Public Domain",
        }
        text = ("This is a human-authored public-domain paragraph with sufficient lexical content. " * 15).strip()
        deduper = ConversationDeduper()
        admitted = admit_document(
            text,
            source=source,
            provenance=provenance,
            public_index=EMPTY_PUBLIC_INDEX,
            deduper=deduper,
            min_chars=200,
            max_chars=4000,
        )
        duplicate = admit_document(
            text,
            source=source,
            provenance=provenance,
            public_index=EMPTY_PUBLIC_INDEX,
            deduper=deduper,
            min_chars=200,
            max_chars=4000,
        )
        self.assertEqual("admit", admitted["receipt"]["decision"])
        self.assertEqual("natural_language_document", admitted["train_row"]["modality"])
        self.assertEqual("broad_english_self_supervised_pretraining_only", admitted["train_row"]["training_use"])
        self.assertEqual("quarantine", duplicate["receipt"]["decision"])
        self.assertIn("exact_duplicate", duplicate["receipt"]["decision_reasons"])

    def test_document_chunking_and_source_policy_fail_closed(self) -> None:
        chunks = chunk_document(("Paragraph text. " * 200) + "\n\n" + ("Second paragraph. " * 200), max_chars=1000, min_chars=200)
        self.assertTrue(chunks)
        self.assertTrue(all(200 <= len(chunk) <= 1000 for chunk in chunks))
        blocked = document_source_plan(
            {
                "id": "synthetic",
                "license_spdx": "public-domain",
                "provenance_class": "external_teacher_generated",
                "format": "document_text",
                "require_row_public_domain_license": True,
            },
            allowed_licenses={"public-domain"},
        )
        self.assertEqual("provenance_class_not_admitted", blocked["decision"])

    def test_atomic_document_shards_resume_without_entering_sft_or_sts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = AtomicDocumentShardWriter(root, shard_rows=1, max_disk_bytes=1_000_000)
            row = {"task_id": "d1", "causal_text": "A sufficiently long public-domain document chunk."}
            receipt = {"receipt_id": "r1", "metrics": {"token_positions": 8}}
            writer.add(row, receipt)
            writer.mark_source_complete("source-a", complete=True)
            writer.close()
            resumed = AtomicDocumentShardWriter(root, shard_rows=1, max_disk_bytes=1_000_000)
            self.assertEqual(1, resumed.total_rows)
            self.assertEqual(8, resumed.total_token_positions)
            self.assertFalse((root / "sts_streams").exists())
            manifest = json.loads((root / "scalable_manifest.json").read_text())
            self.assertEqual("project_theseus_governed_document_stream_state_v1", manifest["policy"])

    def test_document_intake_satisfied_target_resume_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = AtomicDocumentShardWriter(root, shard_rows=1, max_disk_bytes=1_000_000)
            writer.add(
                {"task_id": "d1", "causal_text": "A sufficiently long public-domain document chunk."},
                {"receipt_id": "r1", "metrics": {"token_positions": 8}},
            )
            writer.close()
            config = {
                "allowed_licenses": [],
                "broad_text_intake": {"target_one_pass_token_positions": 8},
                "broad_text_sources": [
                    {
                        "id": "fixture",
                        "dataset": "network/must-not-be-read",
                        "license_spdx": "public-domain",
                        "provenance_class": "human_public_domain",
                        "format": "document_text",
                        "require_row_public_domain_license": True,
                        "enabled": True,
                        "scalable": True,
                    }
                ],
            }
            with patch(
                "governed_conversation_stream.build_public_index",
                return_value={"text_count": 1, "digest": "fixture"},
            ):
                report = run_document_intake(config, root=root, execute=True)
            self.assertEqual(1, report["summary"]["total_rows"])
            self.assertEqual("resume_target_already_met", report["sources"][0]["decision"])


if __name__ == "__main__":
    unittest.main()
