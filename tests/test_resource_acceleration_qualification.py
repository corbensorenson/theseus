from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from resource_acceleration_qualification import (  # noqa: E402
    distribution,
    select_qualification_rows,
    semantic_receipt,
    tensor_mapping_manifest,
    validate_packet,
)


def test_selection_covers_arms_before_filling_remaining_slots() -> None:
    rows = [
        {"case_id": f"{arm}-{index}", "arm_id": arm, "prompt": "prompt"}
        for arm in ("english", "python", "rust")
        for index in range(3)
    ]
    selected = select_qualification_rows(rows, 5)

    assert len(selected) == 5
    assert {row["arm_id"] for row in selected} == {"english", "python", "rust"}
    assert selected == select_qualification_rows(list(reversed(rows)), 5)


def test_packet_rejects_hidden_evaluator_fields() -> None:
    assert validate_packet(
        [{"case_id": "a", "arm_id": "python", "prompt": "p", "expected": "x"}]
    ) == ["private_prompt_packet_contains_evaluator_or_target_fields"]


def test_semantic_receipt_ignores_only_acceleration_route_fields() -> None:
    reference = {
        "state": "FAULT",
        "reason": "byte_serialization_fault",
        "beam_advance": "serial",
        "logit_filter": "host",
        "preprune_beam_expansions": False,
        "decode_receipt": {"state": "INVALID", "token_index": 8},
    }
    optimized = {
        **reference,
        "beam_advance": "batched",
        "logit_filter": "device",
        "preprune_beam_expansions": True,
    }

    assert semantic_receipt(reference) == semantic_receipt(optimized)


def test_distribution_reports_nearest_rank_p95() -> None:
    summary = distribution([1.0, 2.0, 3.0, 4.0])

    assert summary["p50"] == 2.5
    assert summary["p95"] == 4.0
    assert summary["total"] == 10.0


def test_tensor_mapping_manifest_is_order_independent_and_content_bound() -> None:
    import numpy as np

    first = {
        "b": np.array([[1.0, 2.0]], dtype=np.float32),
        "a": np.array([3, 4], dtype=np.int32),
    }
    reordered = {"a": first["a"], "b": first["b"]}
    changed = {**reordered, "b": np.array([[1.0, 2.5]], dtype=np.float32)}

    manifest = tensor_mapping_manifest(first)
    assert manifest == tensor_mapping_manifest(reordered)
    assert manifest["tensor_count"] == 2
    assert manifest["element_count"] == 4
    assert manifest["payload_bytes"] == 16
    assert manifest["sha256"] != tensor_mapping_manifest(changed)["sha256"]
