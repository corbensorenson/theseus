from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from moecot_language_supervision import (  # noqa: E402
    BoundedRows,
    admit_row,
    localized_change_excerpt,
    normalize_code,
    normalize_english,
    split_overlap,
    validate_config,
)


def config() -> dict:
    return json.loads((ROOT / "configs" / "moecot_language_arm_training.json").read_text())


def test_frozen_source_revisions_and_prompt_only_visibility() -> None:
    cfg = validate_config(config())
    assert cfg["generator_visible_fields"] == ["prompt"]
    assert set(cfg["evaluator_only_fields"]) == {"target", "target_sha256", "source_identity"}
    assert cfg["code_source"]["revision"] == "fc56fe33c030c6daa414c2b112c932b8eed085e6"
    assert cfg["english_source"]["revision"] == "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"


def test_code_normalization_retains_provenance_and_hides_target_from_prompt() -> None:
    cfg = validate_config(config())
    row, reason = normalize_code(
        {
            "subject": "Return the incremented value from this function",
            "old_contents": "export const add = (x) => x;\n",
            "new_contents": "export const add = (x) => x + 1;\n",
            "license": "mit",
            "commit": "abc123",
            "repos": "example/repo",
            "old_file": "add.ts",
            "new_file": "add.ts",
        },
        cfg,
        cfg["code_source"],
        arm="javascript_typescript",
        language="typescript",
    )
    assert reason == ""
    assert row["target"] not in row["prompt"]
    assert "Return only the complete revised excerpt" in row["prompt"]
    assert row["license_spdx"] == "mit"
    assert row["provenance"]["commit"] == "abc123"
    assert row["public_benchmark"] is False
    assert row["external_inference_calls"] == 0


def test_disallowed_license_and_benchmark_marker_fail_closed() -> None:
    cfg = validate_config(config())
    base = {
        "subject": "Apply this useful source code change",
        "old_contents": "fn value() -> i32 { 1 }\n",
        "new_contents": "fn value() -> i32 { 2 }\n",
        "commit": "abc",
    }
    _row, reason = normalize_code(
        {**base, "license": "agpl-3.0"}, cfg, cfg["code_source"], arm="rust", language="rust"
    )
    assert reason == "row_license_not_allowed"
    _row, reason = normalize_code(
        {**base, "license": "mit", "subject": "Copy the HumanEval answer"},
        cfg,
        cfg["code_source"],
        arm="rust",
        language="rust",
    )
    assert reason == "public_benchmark_marker"


def test_english_normalization_uses_human_authored_frozen_source() -> None:
    cfg = validate_config(config())
    row, reason = normalize_english(
        {
            "instruction": "Explain why a checksum is useful in one paragraph.",
            "context": "A file crosses an unreliable transport.",
            "response": "A checksum lets the receiver detect accidental changes to the file.",
            "category": "general_qa",
        },
        cfg,
        cfg["english_source"],
    )
    assert reason == ""
    assert row["arm_id"] == "english"
    assert "Context:" in row["prompt"]
    assert row["target"] not in row["prompt"]


def test_hash_bounded_selection_and_split_are_order_invariant() -> None:
    cfg = validate_config(config())
    rows = []
    for index in range(100):
        prompt = f"Explain deterministic item {index} in enough detail."
        target = f"Deterministic answer number {index}."
        row, reason = normalize_english(
            {"instruction": prompt, "context": "", "response": target, "category": "general_qa"},
            cfg,
            cfg["english_source"],
        )
        assert reason == ""
        rows.append(row)

    def select(values: list[dict]) -> tuple[list[str], list[str]]:
        selectors = {"private_train": BoundedRows(7), "private_eval": BoundedRows(3)}
        prompts: set[str] = set()
        targets: set[str] = set()
        from collections import Counter

        rejected: Counter[str] = Counter()
        for row in values:
            admit_row(dict(row), selectors, cfg, prompts, targets, rejected)
        return (
            [row["row_id"] for row in selectors["private_train"].rows()],
            [row["row_id"] for row in selectors["private_eval"].rows()],
        )

    assert select(rows) == select(list(reversed(rows)))


def test_split_overlap_audit_detects_prompt_or_target_reuse() -> None:
    prompts = {}
    targets = {}
    for arm in ("english", "python", "javascript_typescript", "html_css", "rust"):
        prompts[f"{arm}:private_train"] = {f"p-{arm}"}
        prompts[f"{arm}:private_eval"] = set()
        targets[f"{arm}:private_train"] = {f"t-{arm}"}
        targets[f"{arm}:private_eval"] = set()
    prompts["rust:private_eval"] = {"p-rust"}
    audit = split_overlap(prompts, targets)
    assert audit["prompt_overlap_count"] == 1
    assert audit["target_overlap_count"] == 0


def test_localized_edit_contract_preserves_changed_span_and_context() -> None:
    old = "one\ntwo\nthree\nfour\nfive\n"
    new = "one\ntwo\nTHREE\nfour\nfive\n"
    old_excerpt, new_excerpt, receipt = localized_change_excerpt(old, new, context_lines=1)
    assert old_excerpt == "two\nthree\nfour\n"
    assert new_excerpt == "two\nTHREE\nfour\n"
    assert receipt["old_line_range"] == [1, 4]
    assert receipt["new_line_range"] == [1, 4]
