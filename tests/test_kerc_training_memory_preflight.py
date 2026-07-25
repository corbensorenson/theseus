from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_training_memory_preflight as preflight  # noqa: E402


def test_memory_preflight_requires_complete_identity_matched_panel() -> None:
    rows = [
        {
            "station": station,
            "row_sha256": "same",
            "sequence_width": 1248,
            "parameter_count": 72_534_757,
            "mlx_peak_memory_bytes": (index + 1) * 1024 * 1024,
            "memory_execution_policy": {
                "row_limit": 64,
                "coverage_step": 8,
                "attention_query_chunk_size": 256,
                "attention_key_chunk_size": 256,
                "compact_encoder_decoder_partitions": True,
            },
        }
        for index, station in enumerate(preflight.STATIONS)
    ]
    report = preflight.aggregate(rows)
    assert report["trigger_state"] == "GREEN"
    assert report["capability_credit"] == "NONE_RESOURCE_PREFLIGHT_ONLY"
    assert report["scientific_falsification_claimed"] is False
    assert report["peak_memory_mib_by_station"]["decoder_core"] == 1.0
    assert report["memory_execution_policy"]["attention_key_chunk_size"] == 256

    broken = [dict(row) for row in rows]
    broken[-1]["row_sha256"] = "different"
    with pytest.raises(ValueError, match="changed row or model identity"):
        preflight.aggregate(broken)

    mismatched_policy = [dict(row) for row in rows]
    mismatched_policy[-1]["memory_execution_policy"] = dict(
        mismatched_policy[-1]["memory_execution_policy"],
        attention_key_chunk_size=128,
    )
    with pytest.raises(ValueError, match="changed row or model identity"):
        preflight.aggregate(mismatched_policy)
