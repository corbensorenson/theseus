from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from moecot_source_conditioned_pretraining import (  # noqa: E402
    delete_spans,
    denoising_rows,
    source_rejection,
    validate_config,
)
from neural_seed_open_vocab import reserve_byte_fallback_tokens  # noqa: E402


def config() -> dict:
    return {
        "source_conditioned_pretraining": {
            "policy": "project_theseus_moecot_source_conditioned_pretraining_v1",
            "rows_by_arm": {
                "english": 0,
                "python": 2,
                "javascript_typescript": 2,
                "html_css": 2,
                "rust": 2,
            },
            "maximum_windows_per_document": 2,
            "deletion_fraction": 0.2,
            "maximum_deletion_spans": 2,
            "minimum_target_logical_tokens": 8,
            "maximum_target_logical_tokens": 24,
            "maximum_source_encoded_tokens": 200,
            "maximum_target_encoded_tokens": 100,
            "seed": 7,
            "allowed_licenses": ["mit"],
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
    }


def test_config_rejects_cross_arm_data_and_nonzero_boundaries() -> None:
    cfg = config()
    validate_config(cfg)
    cfg["source_conditioned_pretraining"]["rows_by_arm"]["english"] = 1
    with pytest.raises(ValueError, match="English arm"):
        validate_config(cfg)
    cfg = config()
    cfg["source_conditioned_pretraining"]["public_training_rows_written"] = 1
    with pytest.raises(ValueError, match="no-cheat"):
        validate_config(cfg)


def test_corruption_is_deterministic_bounded_and_not_identity() -> None:
    tokens = list("abcdefghijklmnopqrstuvwxyz")
    cfg = config()["source_conditioned_pretraining"]
    first = delete_spans(tokens, cfg, "1" * 64)
    second = delete_spans(tokens, cfg, "1" * 64)
    assert first == second
    assert first != tokens
    assert len(first) == len(tokens) - round(len(tokens) * cfg["deletion_fraction"])


def test_denoising_rows_keep_target_out_of_prompt_and_preserve_provenance() -> None:
    cfg = config()["source_conditioned_pretraining"]
    cfg["maximum_source_encoded_tokens"] = 1000
    cfg["maximum_target_encoded_tokens"] = 500
    source_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    target_vocab = dict(source_vocab)
    reserve_byte_fallback_tokens(source_vocab, max_vocab=270, stream="source")
    reserve_byte_fallback_tokens(target_vocab, max_vocab=270, stream="target")
    source = {
        "text": "fn main() { let value = 42; println!(\"{}\", value); }\n" * 3,
        "text_sha256": "a" * 64,
        "language": "rust",
        "repo": "example/repo",
        "path": "src/main.rs",
        "license_spdx": "MIT",
    }
    rows = denoising_rows(source, "rust", cfg, source_vocab, target_vocab)
    assert rows
    assert all(row["target"] not in row["prompt"] for row in rows)
    assert all(row["public_benchmark"] is False for row in rows)
    assert all(row["source_identity"]["text_sha256"] == "a" * 64 for row in rows)
    assert all(0.0 < row["target_token_copy_fraction"] < 1.0 for row in rows)


def test_source_rejection_fails_closed_on_license_and_public_payloads() -> None:
    cfg = config()["source_conditioned_pretraining"]
    clean = {
        "text": "fn main() {}",
        "license_spdx": "MIT",
        "public_benchmark": False,
        "public_tests_included": False,
        "public_benchmark_solutions_included": False,
    }
    assert source_rejection(clean, cfg) == ""
    assert source_rejection({**clean, "license_spdx": "unknown"}, cfg) == "license_not_allowed"
    assert source_rejection({**clean, "public_benchmark": True}, cfg).startswith("public_")
