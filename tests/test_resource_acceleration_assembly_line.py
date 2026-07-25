from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from resource_acceleration_assembly_line import (  # noqa: E402
    ARCHITECTURE_DECISION_STATIONS,
    SERVING_STATIONS,
    TRAINING_STATIONS,
    build_assembly_line,
    generation_quality_receipt,
)


def test_fault_parity_never_becomes_capability_grade_speed_evidence() -> None:
    empty = hashlib.sha256(b"").hexdigest()
    receipt = generation_quality_receipt(
        {
            "case_count": 2,
            "exact_parity_case_count": 2,
            "cases": [
                {
                    "arm_id": "python",
                    "generation_state": "FAULT",
                    "generation_reason": "byte_serialization_fault",
                    "output_sha256": empty,
                },
                {
                    "arm_id": "rust",
                    "generation_state": "FAULT",
                    "generation_reason": "byte_serialization_fault",
                    "output_sha256": empty,
                },
            ],
        }
    )

    assert receipt["mechanics_parity"] is True
    assert receipt["successful_nonempty_case_count"] == 0
    assert receipt["capability_grade_speed_evidence"] is False


def test_successful_nonempty_exact_parity_is_capability_grade() -> None:
    receipt = generation_quality_receipt(
        {
            "case_count": 2,
            "exact_parity_case_count": 2,
            "cases": [
                {
                    "arm_id": "python",
                    "generation_state": "GREEN",
                    "generation_reason": None,
                    "output_sha256": hashlib.sha256(b"def f(): return 1").hexdigest(),
                },
                {
                    "arm_id": "english",
                    "generation_state": "GREEN",
                    "generation_reason": None,
                    "output_sha256": hashlib.sha256(b"A useful answer").hexdigest(),
                },
            ],
        }
    )

    assert receipt["successful_nonempty_case_count"] == 2
    assert receipt["capability_grade_speed_evidence"] is True


def test_assembly_line_keeps_missing_stations_visible_and_ordered() -> None:
    report = {
        "training": {"paired_canary": {}},
        "checkpoint_storage": {},
        "checkpoint_load": {},
        "inference": {"case_count": 0, "cases": []},
        "resident_runtime": {},
        "assistant_context_refresh": {},
        "architecture_decision_control": {},
    }

    observed = build_assembly_line(report)

    assert observed["measurement_complete"] is False
    assert observed["quality_complete"] is False
    assert tuple(
        row["station_id"]
        for row in observed["lanes"]["training_feedback"]["stations"]
    ) == TRAINING_STATIONS
    assert tuple(
        row["station_id"]
        for row in observed["lanes"]["architecture_decision"]["stations"]
    ) == ARCHITECTURE_DECISION_STATIONS
    assert tuple(
        row["station_id"]
        for row in observed["lanes"]["interactive_serving"]["stations"]
    ) == SERVING_STATIONS
    assert "data_admission" in observed["lanes"]["training_feedback"]["unmeasured_stations"]
    assert "autoregressive_decode" in observed["lanes"]["interactive_serving"]["quality_blocked_stations"]


def test_assembly_line_does_not_mislabel_partial_checkpoint_cost_as_complete() -> None:
    report = {
        "training": {"paired_canary": {}},
        "checkpoint_storage": {
            "formats": {
                "safetensors": {
                    "serialization_seconds": 0.2,
                    "content_hash_seconds": 0.1,
                    "bytes": 100,
                }
            }
        },
        "checkpoint_load": {},
        "inference": {"case_count": 0, "cases": []},
        "resident_runtime": {},
        "assistant_context_refresh": {},
        "architecture_decision_control": {},
    }

    observed = build_assembly_line(report)
    row = next(
        item
        for item in observed["lanes"]["training_feedback"]["stations"]
        if item["station_id"] == "checkpoint_publication"
    )

    assert row["timing"]["cold_seconds"] == 0.3
    assert row["measurement_kind"] == "PARTIAL"
    assert "checkpoint_publication" in observed["lanes"]["training_feedback"][
        "partial_or_derived_stations"
    ]


def test_assembly_line_consumes_exact_corpus_to_tensor_evidence() -> None:
    report = {
        "training": {"paired_canary": {}},
        "corpus_to_tensor": {
            "state": "GREEN",
            "sample": {"input_utf8_bytes": 4000},
            "parity": {"exact_byte_token_row_tensor_receipt_identity": True},
            "performance": {
                "rust_median_wall_seconds": 0.25,
                "rust_runs": [{"materialized_positions": 10000}],
            },
        },
        "checkpoint_storage": {},
        "checkpoint_load": {},
        "inference": {"case_count": 0, "cases": []},
        "resident_runtime": {},
        "assistant_context_refresh": {},
        "architecture_decision_control": {},
    }

    observed = build_assembly_line(report)
    row = next(
        item
        for item in observed["lanes"]["training_feedback"]["stations"]
        if item["station_id"] == "stage_materialization"
    )

    assert row["measurement_kind"] == "MEASURED"
    assert row["timing"]["cold_seconds"] == 0.25
    assert row["throughput_units_per_second"] == 40000.0
    assert row["bytes_processed"] == 4000
    assert row["quality_ready"] is True


def test_candidate_ledger_rejects_nonreproducible_bf16_and_keeps_gaps_explicit() -> None:
    report = {
        "training": {
            "bf16_checkpoint_resume": {"state": "RED"},
            "fp32_checkpoint_resume": {"state": "GREEN"},
            "paired_canary": {},
        },
        "adoption": {
            "rust_exact_corpus_to_tensor": "QUALIFIED_FROZEN_LINEAGE",
            "rust_kerc_dual_space_encoding": "QUALIFIED_TYPED_PREPROCESSING",
        },
        "corpus_to_tensor": {},
        "checkpoint_storage": {},
        "checkpoint_load": {},
        "inference": {"case_count": 0, "cases": []},
        "resident_runtime": {},
        "assistant_context_refresh": {},
        "architecture_decision_control": {},
    }

    ledger = build_assembly_line(report)["candidate_disposition_ledger"]
    by_id = {row["candidate_id"]: row for row in ledger["rows"]}

    assert ledger["all_candidates_dispositioned"] is True
    assert by_id["exact_corpus_to_tensor_rust"]["disposition"] == "ADOPTED"
    assert by_id["compiled_fp32_microbatch4"]["disposition"] == "ADOPTED"
    assert by_id["bf16_compute_fp32_master"]["disposition"] == "REJECTED"
    assert by_id["compressed_jsonl_direct_ingestion"]["disposition"] == "DEFERRED"
    assert by_id["parquet_source_ingestion"]["disposition"] == "INAPPLICABLE_CURRENT_LINEAGE"


def test_candidate_ledger_consumes_scanner_compression_and_parallel_dispositions() -> None:
    report = {
        "training": {"paired_canary": {}},
        "adoption": {},
        "corpus_to_tensor": {
            "ingestion_parallelism_and_scanner": {
                "scanner": {"disposition": "REJECTED"},
                "compressed_input": {"exact_artifact_parity": True},
                "bounded_parallel_materialization": {"disposition": "ADOPTED"},
                "content_addressed_cache": {"state": "GREEN"},
            }
        },
        "checkpoint_storage": {},
        "checkpoint_load": {},
        "inference": {"case_count": 0, "cases": []},
        "resident_runtime": {},
        "assistant_context_refresh": {},
        "architecture_decision_control": {},
    }

    ledger = build_assembly_line(report)["candidate_disposition_ledger"]
    by_id = {row["candidate_id"]: row for row in ledger["rows"]}

    assert by_id["simd_swar_exact_scanner"]["disposition"] == "REJECTED"
    assert by_id["compressed_jsonl_direct_ingestion"]["disposition"] == "ADOPTED"
    assert by_id["bounded_parallel_corpus_encoding"]["disposition"] == "ADOPTED"
    assert by_id["persistent_hostwide_encoded_shard_cache"]["disposition"] == "ADOPTED"
