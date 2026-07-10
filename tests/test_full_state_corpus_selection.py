from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.neural_seed_code_proposer_comparator import stable_hash
from scripts.neural_seed_full_state_pretraining import (
    collect_full_state_python_examples,
    filter_complete_target_examples,
    python_function_body_pretraining_examples,
)


class FullStateCorpusSelectionTests(unittest.TestCase):
    def test_selection_is_bottom_k_over_full_manifest_not_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = []
            expected_rows = []
            for index in range(12):
                path = root / f"source_{index:02d}.py"
                text = (
                    f'def transform_{index}(values):\n'
                    f'    """Transform values using offset {index}."""\n'
                    "    total = 0\n"
                    "    for value in values:\n"
                    f"        total += value + {index}\n"
                    "    return total\n"
                )
                path.write_text(text, encoding="utf-8")
                sources.append(
                    {
                        "path": str(path),
                        "admitted": True,
                        "license_allowed": True,
                        "char_count": len(text),
                        "sha256": stable_hash(text),
                    }
                )
                expected = python_function_body_pretraining_examples(
                    str(path),
                    text,
                    max_function_body_chars=2000,
                    source_text_style="prompt_signature_metadata_v2",
                )[0]
                source_text = str(expected["source_text"])
                body = str(expected["body"])
                source_body_hash = stable_hash(json.dumps([source_text, body], separators=(",", ":"), ensure_ascii=True))
                rank = int(stable_hash(f"17:{source_body_hash}:{path}:transform_{index}"), 16)
                expected_rows.append((rank, f"transform_{index}"))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"sources": sources}), encoding="utf-8")

            rows, summary = collect_full_state_python_examples(
                {
                    "corpus_manifest": str(manifest),
                    "source_text_style": "prompt_signature_metadata_v2",
                    "quality_filter": {"enabled": False},
                },
                max_files=12,
                max_examples=3,
                min_target_tokens=1,
                max_function_body_chars=2000,
                seed=17,
            )

            expected = [name for _rank, name in sorted(expected_rows)[:3]]
            self.assertEqual(expected, [row["function"] for row in rows])
            self.assertEqual(12, summary["eligible_example_count_before_sampling"])
            self.assertEqual("deterministic_full_stream_bottom_k_v1", summary["selection_policy"])
            self.assertGreater(summary["selected_single_pass_target_token_positions"], 0)

    def test_exact_source_body_duplicates_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = (
                'def total(values):\n'
                '    """Sum all values."""\n'
                "    result = 0\n"
                "    for value in values:\n"
                "        result += value\n"
                "    return result\n"
            )
            sources = []
            for index in range(2):
                path = root / f"duplicate_{index}.py"
                path.write_text(text, encoding="utf-8")
                sources.append(
                    {
                        "path": str(path),
                        "admitted": True,
                        "license_allowed": True,
                        "char_count": len(text),
                        "sha256": stable_hash(text),
                    }
                )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"sources": sources}), encoding="utf-8")

            rows, summary = collect_full_state_python_examples(
                {"corpus_manifest": str(manifest), "quality_filter": {"enabled": False}},
                max_files=2,
                max_examples=10,
                min_target_tokens=1,
                max_function_body_chars=2000,
                seed=3,
            )

            self.assertEqual(1, len(rows))
            self.assertEqual(1, summary["exact_source_body_duplicate_count"])

    def test_oversized_targets_are_rejected_without_prefix_truncation(self) -> None:
        examples = [
            {"body": "return data", "function": "short"},
            {
                "body": "\n".join(["result = []", *["result.append(data)" for _ in range(20)], "return result"]),
                "function": "long",
            },
        ]

        kept, summary = filter_complete_target_examples(
            examples,
            max_target=16,
            target_mode="body_tokens",
        )

        self.assertEqual(["short"], [row["function"] for row in kept])
        self.assertEqual(1, summary["oversized_example_count"])
        self.assertEqual(0, summary["target_sequence_truncation_count"])
        self.assertEqual(0, summary["target_encoding_fault_count"])


if __name__ == "__main__":
    unittest.main()
