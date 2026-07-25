from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_k5_candidate_evaluator as evaluator


def test_operation_specific_floor_uses_bound_probe_peak_not_training_floor(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "probe.host_resource_safety.json"
    receipt_path.write_text(
        json.dumps(
            {
                "passed": True,
                "fault": "",
                "command": ["python", "scripts/kerc_k5_stage_learnability_probe.py"],
                "maximum_inferred_unified_memory_mib": 1253.344,
                "maximum_swapout_growth_mib": 0.0,
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    contract = {
        "host_safety_policy": {
            "minimum_available_before_launch_mib": 6144,
            "minimum_available_during_run_mib": 4096,
        },
        "canaries": [
            {
                "candidate_id": "rdc_kerc_k5_adequacy",
                "host_safety_overrides": {
                    "minimum_available_before_launch_mib": 2048,
                    "minimum_available_during_run_mib": 2048,
                    "maximum_swapout_growth_mib": 16,
                },
            }
        ],
    }
    mapping = evaluator.operation_specific_host_safety_mapping(
        candidate_id="rdc_kerc_k5_adequacy",
        candidate_contract=contract,
        receipt_path=str(receipt_path),
        receipt_sha256=digest,
        command_marker="kerc_k5_stage_learnability_probe.py",
    )
    assert mapping["minimum_available_before_launch_mib"] == 3302
    assert mapping["minimum_available_during_run_mib"] == 2048
    assert mapping["measured_operation_preflight"][
        "maximum_inferred_unified_memory_mib"
    ] == 1253.344
    with pytest.raises(ValueError, match="identity mismatch"):
        evaluator.operation_specific_host_safety_mapping(
            candidate_id="rdc_kerc_k5_adequacy",
            candidate_contract=contract,
            receipt_path=str(receipt_path),
            receipt_sha256="0" * 64,
            command_marker="kerc_k5_stage_learnability_probe.py",
        )


def test_summarize_evaluation_recomputes_pipeline_state(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "nonempty": True,
                        "generation": {"state": "GREEN"},
                    },
                    {
                        "nonempty": False,
                        "generation": {
                            "state": "FAULT",
                            "reason": "KERC_STAGE_INVALID",
                        },
                    },
                    {
                        "nonempty": False,
                        "generation": {
                            "state": "FAULT",
                            "fault": {
                                "fault_type": "KERC_STAGE_INVALID",
                                "detail": json.dumps(
                                    {
                                        "semantic_selection": {
                                            "complete_candidate_count": 4,
                                            "rejected_candidate_count": 3,
                                            "rejection_counts": {
                                                "KERC_ALIGNMENT_SPAN_INVALID": 3
                                            },
                                            "selected_semantically_valid": True,
                                        }
                                    }
                                ),
                            },
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = evaluator.summarize_evaluation(
        {"rows": {"path": str(rows_path)}}
    )
    assert summary["row_count"] == 3
    assert summary["pipeline_success_count"] == 1
    assert summary["pipeline_success_rate"] == 1 / 3
    assert summary["fault_or_state_counts"] == {
        "GREEN": 1,
        "KERC_STAGE_INVALID": 2,
    }
    assert summary["semantic_proposal_accounting"] == {
        "complete_candidate_count": 4,
        "rejected_candidate_count": 3,
        "rejection_counts": {"KERC_ALIGNMENT_SPAN_INVALID": 3},
        "valid_selected_candidate_count": 1,
        "denominator_complete_for_reported_faults": True,
        "invalid_candidates_repaired_or_rewritten": False,
    }
    assert summary["raw_generated_text_retained"] is False


def test_frozen_surface_excludes_every_prior_evaluated_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "private_dev.jsonl"
    rows = [
        {
            "row_id": f"row-{index}",
            "split": "private_dev",
            "public_benchmark": False,
        }
        for index in range(8)
    ]
    artifact.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    training_report = {
        "targets": {
            "english_kerc": {
                "supervision_artifacts": {
                    "private_dev": {
                        "path": str(artifact),
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "row_count": len(rows),
                    }
                }
            }
        }
    }
    monkeypatch.setattr(
        evaluator, "prior_kerc_evaluation_row_ids", lambda: {"row-0", "row-1"}
    )
    manifest_path = tmp_path / "surface.json"
    surface = evaluator.frozen_surface_manifest(
        training_report=training_report,
        surface_id="fresh-surface-test-v1",
        maximum_rows=4,
        output_path=manifest_path,
    )
    assert len(surface["selected_row_ids"]) == 4
    assert not {"row-0", "row-1"}.intersection(surface["selected_row_ids"])
    loaded = evaluator.load_frozen_surface(
        path=manifest_path,
        surface_id="fresh-surface-test-v1",
        training_report=training_report,
        maximum_rows=4,
    )
    assert loaded == surface


def test_consumed_surface_registry_refuses_exact_surface_reuse(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.jsonl"
    evaluator.append_jsonl(
        registry,
        {
            "policy": evaluator.REGISTRY_POLICY,
            "surface_id": "surface-a",
            "consumed": True,
        },
    )
    evaluator.append_jsonl(
        registry,
        {
            "policy": evaluator.REGISTRY_POLICY,
            "surface_id": "surface-b",
            "consumed": False,
        },
    )
    assert evaluator.consumed_surface_ids(registry) == {"surface-a"}
