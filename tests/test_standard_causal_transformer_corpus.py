from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from standard_causal_transformer_corpus import (  # noqa: E402
    category_targets,
    code_quality_rejection_reasons,
    load_pretrain_memmaps,
    materialize_pretrain_stage,
    pretrain_array_paths,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path: Path) -> dict:
    code_rows = []
    for language, suffix, text in (
        ("python", ".py", "def alpha(value):\n    return value + 1\n"),
        ("typescript", ".ts", "bravo typescript interface maps event payload fields"),
        ("css", ".css", "charlie stylesheet grid color margin padding display"),
        ("rust", ".rs", "delta rust iterator ownership result match lifetime"),
    ):
        code_rows.append(
            {
                "language": language,
                "path": f"sample{suffix}",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    code_path = tmp_path / "code" / "samples.jsonl"
    write_jsonl(code_path, code_rows)
    code_manifest = tmp_path / "code" / "manifest.json"
    code_manifest.write_text(
        json.dumps({"sample_jsonl": "samples.jsonl", "sample_jsonl_sha256": digest(code_path)}),
        encoding="utf-8",
    )

    conversation_path = tmp_path / "conversation" / "conversation.jsonl"
    write_jsonl(
        conversation_path,
        [{"causal_text": "echo user asks a careful question assistant answers with useful detail"}],
    )
    conversation_manifest = tmp_path / "conversation" / "manifest.json"
    conversation_manifest.write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "train_path": "conversation.jsonl",
                        "train_sha256": digest(conversation_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    broad_path = tmp_path / "broad" / "document.jsonl"
    write_jsonl(
        broad_path,
        [{"causal_text": "foxtrot history science engineering culture language evidence explanation"}],
    )
    broad_manifest = tmp_path / "broad" / "manifest.json"
    broad_manifest.write_text(
        json.dumps(
            {"shards": [{"train_path": "document.jsonl", "train_sha256": digest(broad_path)}]}
        ),
        encoding="utf-8",
    )
    intake_policy = tmp_path / "intake-policy.json"
    intake_policy.write_text(
        json.dumps(
            {
                "sources": [{"id": "conversation", "enabled": True, "required_langs": ["en"]}],
                "broad_text_sources": [
                    {"id": "broad", "enabled": True, "required_language": "en"}
                ],
            }
        ),
        encoding="utf-8",
    )
    repo_policy = tmp_path / "repo-policy.json"
    repo_policy.write_text(json.dumps({"repos": [{"repo": "example/project"}]}), encoding="utf-8")
    return {
        "tokenization": {"max_sequence_tokens": 4},
        "moecot_language_seed_contract": {
            "arms": [
                {"id": "english", "categories": ["english_conversation_instruction", "english_broad"]},
                {"id": "python", "categories": ["python"]},
                {"id": "javascript_typescript", "categories": ["javascript_typescript"]},
                {"id": "html_css", "categories": ["html_css"]},
                {"id": "rust", "categories": ["rust"]},
            ]
        },
        "data_model_scaling_contract": {
            "required_unique_positions": 24,
            "domain_minimum_positions": {
                "english_natural_language_total": 6,
                "code_total": 12,
                "flexible_tail_reserve": 6,
            },
            "subset_minimum_positions": {"english_conversation_instruction": 3},
            "code_language_minimum_positions": {
                "python": 3,
                "javascript_typescript": 3,
                "html_css": 3,
                "rust": 3,
            },
        },
        "canonical_corpus": {
            "near_duplicate_hamming_distance": 0,
            "natural_language_scope": {
                "allowed_languages": ["en"],
                "non_allowed_action": "quarantine",
                "intake_policy": str(intake_policy),
            },
            "programming_language_scope": [
                "python",
                "javascript_typescript",
                "html_css",
                "rust",
            ],
            "code_quality_policy": {
                "policy": "project_theseus_curated_code_quality_v1",
                "curated_repo_config": str(repo_policy),
                "minimum_logical_tokens": 1,
                "maximum_line_characters": 2000,
                "maximum_mean_nonempty_line_characters": 300,
                "minimum_unique_token_ratio": 0.01,
                "excluded_path_parts": ["vendor"],
                "excluded_name_markers": [".min."],
                "generated_text_markers": ["do not edit"],
            },
            "code_shard_manifests": [str(code_manifest)],
            "conversation_root": str(conversation_path.parent),
            "conversation_manifest": str(conversation_manifest),
            "broad_text_root": str(broad_path.parent),
            "broad_text_manifest": str(broad_manifest),
        },
    }


def encode(text: str, _category: str) -> tuple[list[str], list[int], dict]:
    tokens = text.split()
    return tokens, list(range(1, len(tokens) + 1)), {"unknown_token_count": 0}


def test_category_targets_partition_frozen_contract_exactly(tmp_path: Path) -> None:
    targets = category_targets(fixture(tmp_path)["data_model_scaling_contract"])

    assert targets == {
        "english_conversation_instruction": 4,
        "english_broad": 4,
        "python": 4,
        "javascript_typescript": 4,
        "html_css": 4,
        "rust": 4,
    }
    assert sum(targets.values()) == 24


def test_code_quality_rejects_generated_vendored_and_minified_payloads(tmp_path: Path) -> None:
    corpus = fixture(tmp_path)["canonical_corpus"]

    assert "excluded_path" in code_quality_rejection_reasons(
        corpus,
        path="vendor/library.py",
        text="def useful(value):\n    return value + 1\n",
        category="python",
    )
    assert "generated_marker" in code_quality_rejection_reasons(
        corpus,
        path="src/generated.rs",
        text="// DO NOT EDIT\npub fn value() -> i32 { 1 }",
        category="rust",
    )
    assert "minified_or_bundle_name" in code_quality_rejection_reasons(
        corpus,
        path="web/app.min.js",
        text="function value(){return 1}",
        category="javascript_typescript",
    )


def test_materialization_is_content_bound_balanced_and_disk_backed(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    inputs, labels, mask, report = materialize_pretrain_stage(
        config,
        root=tmp_path,
        stage_dir=stage_dir,
        target_vocab={},
        target_offset=100,
        tokenize_and_encode=encode,
        eval_body_patterns=set(),
    )

    assert inputs.shape == labels.shape == mask.shape == (6, 4)
    assert int(mask.sum()) == 24
    assert report["category_positions"] == report["category_targets"]
    assert report["materialized_positions"] == 24
    assert report["non_overlapping_windows"] is True
    assert report["arm_views"]["arms"]["english"]["target_positions"] == 8
    assert report["arm_views"]["arms"]["python"]["row_ranges"] == [{"start": 2, "stop": 3}]
    assert report["arm_views"]["mixed_dense_control"]["target_positions"] == 24
    assert report["public_training_rows_written"] == 0
    assert all(row["sha256"] == digest(Path(row["path"])) for row in report["array_artifacts"].values())

    loaded = load_pretrain_memmaps(
        pretrain_array_paths(stage_dir),
        (6, 4),
        expected=report["array_artifacts"],
    )
    assert int(loaded[2].sum()) == 24

    with pretrain_array_paths(stage_dir)["mask"].open("r+b") as handle:
        handle.write(b"\x00")
    with pytest.raises(ValueError, match="array identity mismatch"):
        load_pretrain_memmaps(
            pretrain_array_paths(stage_dir),
            (6, 4),
            expected=report["array_artifacts"],
        )


def test_materialization_refuses_stale_source_shard(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    code_manifest = Path(config["canonical_corpus"]["code_shard_manifests"][0])
    code_path = code_manifest.parent / "samples.jsonl"
    code_path.write_text(code_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    with pytest.raises(ValueError, match="source identity mismatch"):
        materialize_pretrain_stage(
            config,
            root=tmp_path,
            stage_dir=stage_dir,
            target_vocab={},
            target_offset=100,
            tokenize_and_encode=encode,
            eval_body_patterns=set(),
        )


def test_materialization_refuses_non_english_natural_language_source(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    policy_path = Path(config["canonical_corpus"]["natural_language_scope"]["intake_policy"])
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["sources"][0]["required_langs"] = ["es"]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    with pytest.raises(ValueError, match="non-English"):
        materialize_pretrain_stage(
            config,
            root=tmp_path,
            stage_dir=stage_dir,
            target_vocab={},
            target_offset=100,
            tokenize_and_encode=encode,
            eval_body_patterns=set(),
        )
