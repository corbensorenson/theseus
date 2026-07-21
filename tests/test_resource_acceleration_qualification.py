from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from resource_acceleration_qualification import (  # noqa: E402
    aggregate_training_routes,
    distribution,
    process_resource_delta,
    select_qualification_rows,
    semantic_receipt,
    tensor_mapping_manifest,
    tree_numeric_receipt,
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


def test_process_resource_delta_preserves_peak_and_differences_counters() -> None:
    before = {
        "maximum_resident_set_bytes": 100,
        "block_input_operations": 2,
        "block_output_operations": 3,
        "voluntary_context_switches": 5,
        "involuntary_context_switches": 7,
    }
    after = {
        "maximum_resident_set_bytes": 150,
        "block_input_operations": 11,
        "block_output_operations": 13,
        "voluntary_context_switches": 17,
        "involuntary_context_switches": 19,
    }
    observed = process_resource_delta(before, after)
    assert observed["maximum_resident_set_bytes"] == 150
    assert observed["block_input_operations_delta"] == 9
    assert observed["block_output_operations_delta"] == 10
    assert observed["voluntary_context_switches_delta"] == 12
    assert observed["involuntary_context_switches_delta"] == 12


def test_tree_numeric_receipt_tracks_precision_and_nonfinite_state() -> None:
    import mlx.core as mx
    import mlx.utils as mlx_utils

    finite = tree_numeric_receipt(
        {
            "compute": mx.array([1.0, 2.0], dtype=mx.bfloat16),
            "step": mx.array([1], dtype=mx.uint64),
        },
        mx=mx,
        mlx_utils=mlx_utils,
    )
    invalid = tree_numeric_receipt(
        {"weight": mx.array([float("nan")], dtype=mx.float32)},
        mx=mx,
        mlx_utils=mlx_utils,
    )

    assert finite["all_finite"] is True
    assert finite["dtypes"] == ["mlx.core.bfloat16", "mlx.core.uint64"]
    assert invalid["all_finite"] is False


def test_training_route_aggregation_uses_pooled_work_and_preserves_runs() -> None:
    rows = [
        {
            "training_step_execution": "compiled",
            "optimizer_steps": 4,
            "optimizer_positions": 40,
            "warmup_excluded_positions": 30,
            "warmup_excluded_seconds": 2.0,
            "warmup_excluded_positions_per_second": 15.0,
            "mlx_memory": {"peak_bytes": 100},
        },
        {
            "training_step_execution": "compiled",
            "optimizer_steps": 4,
            "optimizer_positions": 60,
            "warmup_excluded_positions": 50,
            "warmup_excluded_seconds": 2.0,
            "warmup_excluded_positions_per_second": 25.0,
            "mlx_memory": {"peak_bytes": 120},
        },
    ]

    observed = aggregate_training_routes(rows)

    assert observed["optimizer_steps_total"] == 8
    assert observed["warmup_excluded_positions_total"] == 80
    assert observed["warmup_excluded_seconds_total"] == 4.0
    assert observed["pooled_positions_per_second"] == 20.0
    assert observed["peak_mlx_bytes_maximum"] == 120
    assert observed["runs"] == rows
