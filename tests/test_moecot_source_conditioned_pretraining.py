from __future__ import annotations

import json
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
    inspect_kernel_english,
    kernel_english_split_overlap,
    materialize_kernel_english,
    source_rejection,
    validate_config,
    validate_kernel_english_config,
)
import kernel_english_protocol as kernel  # noqa: E402
import vcm_semantic_memory as memory  # noqa: E402
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


def kernel_config(tmp_path: Path) -> dict:
    return {
        "stage_dir": str(tmp_path / "stage"),
        "kernel_english_training": {
            "policy": "project_theseus_moecot_kernel_english_stage_v1",
            "required": True,
            "stage_root": str(tmp_path / "kernel-stage"),
            "report": str(tmp_path / "kernel-report.json"),
            "records_jsonl": str(tmp_path / "records.jsonl"),
            "verification_ledger_jsonl": str(tmp_path / "verification-ledger.jsonl"),
            "objective_order": list(kernel.TRAINING_OBJECTIVES),
            "records_by_split": {
                "private_train": 1,
                "private_dev": 1,
                "private_eval": 1,
            },
            "maximum_sequence_tokens": 20000,
            "batch_size": 1,
            "allowed_licenses": ["cc0-1.0"],
            "public_training_rows_written": 0,
            "public_benchmark_payload_count": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "template_credit": 0,
            "deterministic_renderer_credit": 0,
        },
    }


def kernel_record(split: str, index: int) -> dict:
    source = f"Approval request {index} may proceed."
    scope = {
        "user": "training-user",
        "project": "theseus",
        "conversation": f"training-{split}-{index}",
        "privacy": "private_local",
    }
    state = memory.create_hierarchical_residual_state(
        f"training-{split}-{index}", scope=scope
    )
    program = {
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": "PROCEED",
                "modality": "POSSIBLE",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 0.8,
                "derivation": "compiler_inference",
                "source_spans": [[0, len(source)]],
                "arguments": [],
            }
        ],
    }
    packet = kernel.build_kernel_packet(
        source,
        program,
        hrl_state=state,
        provenance={"source": "private_test_fixture"},
    )
    answer = {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": "PROCEED",
                "modality": "POSSIBLE",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 0.8,
                "arguments": [],
            }
        ],
        "required_terms": [],
        "required_caveats": ["The outcome remains uncertain."],
        "style": {"register": "plain"},
    }
    record = {
        "policy": kernel.TRAINING_RECORD_POLICY,
        "split": split,
        "language": "en",
        "source_text": source,
        "kernel_packet": packet,
        "hrl_state": state,
        "answer_packet": answer,
        "surface_target": f"Request {index} may proceed, but the outcome remains uncertain.",
        "provenance": {
            "source_id": f"source-{split}-{index}",
            "source_group": f"group-{split}-{index}",
            "license_spdx": "CC0-1.0",
            "permitted_use": "model_training",
        },
        "verification_receipt": {
            "policy": kernel.TRAINING_VERIFICATION_POLICY,
            "receipt_id": f"receipt-{split}-{index}",
            "accepted": True,
            "verifier_id": "fixture-verifier-v1",
            "reviewer_independent_of_record_producer": True,
            "method": "human_dual_review",
            "evidence_sha256": "sha256:" + f"{index:064x}"[-64:],
        },
        "public_benchmark": False,
        "public_tests_included": False,
        "public_benchmark_solutions_included": False,
        "external_inference": False,
        "fallback_return_count": 0,
        "template_credit": 0,
        "deterministic_renderer_credit": 0,
        "candidate_generation_credit": 0,
    }
    record["verification_receipt"]["semantic_payload_sha256"] = (
        kernel.training_semantic_payload_sha256(record)
    )
    return record


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


def test_kernel_stage_materializes_replays_and_cleans_atomic_files(tmp_path: Path) -> None:
    cfg = kernel_config(tmp_path)
    validate_kernel_english_config(cfg)
    source_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    target_vocab = dict(source_vocab)
    for token in kernel.TRAINING_TASK_TAGS.values():
        source_vocab[token] = len(source_vocab)
    reserve_byte_fallback_tokens(source_vocab, max_vocab=274, stream="source")
    reserve_byte_fallback_tokens(target_vocab, max_vocab=270, stream="target")
    stage = Path(cfg["stage_dir"])
    stage.mkdir(parents=True)
    (stage / "stage_metadata_v1.json").write_text(
        json.dumps({"source_vocab": source_vocab, "target_vocab": target_vocab}),
        encoding="utf-8",
    )
    records = [
        kernel_record("private_train", 1),
        kernel_record("private_dev", 2),
        kernel_record("private_eval", 3),
    ]
    records_path = Path(cfg["kernel_english_training"]["records_jsonl"])
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    ledger_path = Path(
        cfg["kernel_english_training"]["verification_ledger_jsonl"]
    )
    ledger_path.write_text(
        "".join(
            json.dumps(row["verification_receipt"], sort_keys=True) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )

    report = materialize_kernel_english(cfg, tmp_path / "config.json")
    assert report["trigger_state"] == "GREEN"
    assert report["unique_raw_source_count"] == 3
    assert report["derived_view_unique_data_credit"] == 0
    assert report["derived_view_optimizer_exposure_count"] == 12
    assert all(value == 3 for value in report["compiled_view_count_by_objective"].values())
    assert inspect_kernel_english(cfg, tmp_path / "config.json")["trigger_state"] == "GREEN"
    assert not list(Path(cfg["kernel_english_training"]["stage_root"]).glob("*.partial"))

    train_path = Path(report["artifacts"]["english:private_train"]["path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    train_path.write_text(train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    replay = inspect_kernel_english(cfg, tmp_path / "config.json")
    assert replay["trigger_state"] == "RED"
    assert any("artifact_identity" in gap for gap in replay["hard_gaps"])


def test_kernel_split_overlap_rejects_group_and_content_reuse() -> None:
    train = kernel.validate_training_record(kernel_record("private_train", 10))
    dev = kernel.validate_training_record(kernel_record("private_dev", 11))
    dev["provenance"]["source_group"] = train["provenance"]["source_group"]
    dev["raw_source_sha256"] = train["raw_source_sha256"]
    audit = kernel_english_split_overlap(
        {"private_train": [train], "private_dev": [dev], "private_eval": []}
    )
    assert audit["content_bound_disjoint"] is False
    assert audit["source_group_overlap_count"] == 1
    assert audit["raw_source_overlap_count"] == 1
