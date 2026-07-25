from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_k5_stage_learnability_probe as probe  # noqa: E402


def test_probe_checkpoint_counterfactual_is_content_bound_and_explicit(
    tmp_path: Path,
) -> None:
    authoritative = tmp_path / "authoritative.safetensors"
    diagnostic = tmp_path / "diagnostic.safetensors"
    authoritative.write_bytes(b"authoritative")
    diagnostic.write_bytes(b"diagnostic")
    result = {
        "checkpoint": str(authoritative),
        "checkpoint_sha256": probe.sha256(authoritative),
    }

    selected, receipt = probe.resolve_probe_checkpoint(result)
    assert selected == authoritative
    assert receipt["counterfactual_checkpoint"] is False

    selected, receipt = probe.resolve_probe_checkpoint(
        result,
        diagnostic_checkpoint=str(diagnostic),
        diagnostic_checkpoint_sha256=probe.sha256(diagnostic),
    )
    assert selected == diagnostic
    assert receipt["counterfactual_checkpoint"] is True
    assert receipt["matched_row_selection_source"] == "training_report"

    try:
        probe.resolve_probe_checkpoint(
            result,
            diagnostic_checkpoint=str(diagnostic),
            diagnostic_checkpoint_sha256="0" * 64,
        )
    except ValueError as exc:
        assert "diagnostic checkpoint identity mismatch" in str(exc)
    else:
        raise AssertionError("mismatched diagnostic checkpoint was accepted")


def test_objective_balanced_exposure_batches_replays_warmup_and_joint_tail() -> None:
    batches = probe.objective_balanced_exposure_batches(
        row_count=4,
        optimizer_steps=8,
        single_objective_warmup_steps=4,
    )

    assert batches == (
        (1,),
        (2,),
        (3,),
        (0,),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
    )
    assert probe.objective_balanced_exposure_batches(
        row_count=4,
        optimizer_steps=2,
        single_objective_warmup_steps=0,
    ) == ((0, 1, 2, 3), (0, 1, 2, 3))
    assert probe.objective_balanced_exposure_batches(
        row_count=4,
        optimizer_steps=3,
        single_objective_warmup_steps=0,
        fixed_objective_index=3,
    ) == ((3,), (3,), (3,))


def test_objective_exposure_projection_reports_weak_tail_and_zero_rows() -> None:
    projection = probe.objective_exposure_projection(
        (
            ("objective:compiler",),
            ("objective:compiler", "length:short"),
            ("objective:renderer",),
        ),
        ("c1", "c2", "r1"),
        {"c1": 2, "r1": 4},
    )

    assert projection["compiler"] == {
        "population_row_count": 2,
        "sampled_unique_row_count": 1,
        "unsampled_row_count": 1,
        "total_optimizer_updates": 2,
        "minimum_row_exposures": 0,
        "maximum_row_exposures": 2,
        "mean_row_exposures": 1.0,
    }
    assert projection["renderer"]["minimum_row_exposures"] == 4
    assert projection["renderer"]["unsampled_row_count"] == 0


def test_resource_stress_prefix_replay_appends_maximum_target_row() -> None:
    inputs = np.asarray(
        [
            [4, 5, 0, 0, 0],
            [6, 7, 8, 9, 0],
            [10, 11, 12, 13, 14],
        ],
        dtype=np.int32,
    )
    mask = np.asarray(
        [
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    prefix, receipt = probe.resource_stress_prefix_replay(
        inputs=inputs,
        mask=mask,
        progress_mask=mask,
        coverage_indices=(0, 1),
        enabled=True,
        capacity=3,
    )

    assert prefix == [0, 1, 2]
    assert receipt["active"] is True
    assert receipt["target_positions"] == 4
    assert receipt["active_width"] == 5
    assert receipt["already_in_coverage_prefix"] is False
    assert receipt["selected_row_count"] == 1
    assert receipt["rows"][0]["roles"] == [
        "maximum_target_positions",
        "maximum_active_width",
    ]


def test_resource_stress_prefix_replay_preserves_existing_stress_row() -> None:
    rows = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
    mask = np.asarray([[0, 1], [1, 1]], dtype=np.float32)

    prefix, receipt = probe.resource_stress_prefix_replay(
        inputs=rows,
        mask=mask,
        progress_mask=mask,
        coverage_indices=(1, 0),
        enabled=True,
        capacity=2,
    )

    assert prefix == [1, 0]
    assert receipt["already_in_coverage_prefix"] is True


def test_resource_stress_prefix_replay_includes_distinct_widest_row() -> None:
    inputs = np.asarray(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 8],
        ],
        dtype=np.int32,
    )
    mask = np.asarray(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    prefix, receipt = probe.resource_stress_prefix_replay(
        inputs=inputs,
        mask=mask,
        progress_mask=mask,
        coverage_indices=(),
        enabled=True,
        capacity=2,
    )

    assert prefix == [0, 1]
    assert receipt["selected_row_count"] == 2
    assert receipt["rows"][0]["roles"] == ["maximum_target_positions"]
    assert receipt["rows"][1]["roles"] == ["maximum_active_width"]
    assert receipt["rows"][1]["active_width"] == 5


def test_teacher_forced_alignment_accepts_full_sequence_predictions() -> None:
    predictions, expected = probe.align_teacher_forced_predictions(
        np.asarray([90, 91, 12, 13, 14]),
        np.asarray([10, 11, 12, 13, 14]),
        supervised_start=2,
    )

    assert predictions.tolist() == [12, 13, 14]
    assert expected.tolist() == [12, 13, 14]


def test_teacher_forced_alignment_accepts_compact_target_predictions() -> None:
    predictions, expected = probe.align_teacher_forced_predictions(
        np.asarray([12, 13, 14]),
        np.asarray([10, 11, 12, 13, 14]),
        supervised_start=2,
    )

    assert predictions.tolist() == [12, 13, 14]
    assert expected.tolist() == [12, 13, 14]


def test_teacher_forced_alignment_rejects_execution_shape_drift() -> None:
    try:
        probe.align_teacher_forced_predictions(
            np.asarray([12, 13]),
            np.asarray([10, 11, 12, 13, 14]),
            supervised_start=2,
        )
    except ValueError as exc:
        assert "full or compact execution" in str(exc)
    else:
        raise AssertionError("teacher-forced execution shape drift was accepted")


def test_subset_supervision_stage_rows_preserves_exact_token_authority_scope() -> None:
    stage = probe.SimpleNamespace(
        inputs=probe.training.RaggedRows(
            [
                np.asarray([1, 2], dtype=np.int32),
                np.asarray([3], dtype=np.int32),
                np.asarray([4, 5, 6], dtype=np.int32),
            ],
            dtype=np.int32,
            standard_width=2,
        ),
        labels=np.asarray([[1], [2], [3]], dtype=np.int32),
        mask=np.asarray([[1], [0], [1]], dtype=np.float32),
        loss_mask=np.asarray([[1], [0], [1]], dtype=np.float32),
        training_row_ids=("row-0", "row-1", "row-2"),
        kerc_coverage_labels=(("a",), ("b",), ("c",)),
        receipt={"policy": "source"},
        invariant="preserved",
    )

    selected = probe.subset_supervision_stage_rows(
        stage, probe.training.token_supervised_row_indices(stage.loss_mask)
    )

    assert len(selected.inputs) == 2
    assert np.asarray(selected.inputs[0]).tolist() == [1, 2]
    assert np.asarray(selected.inputs[1]).tolist() == [4, 5, 6]
    assert selected.training_row_ids == ("row-0", "row-2")
    assert selected.kerc_coverage_labels == (("a",), ("c",))
    assert selected.invariant == "preserved"
    assert selected.receipt["probe_token_supervised_row_projection"][
        "selected_row_count"
    ] == 2


def test_stage_scope_replay_selects_overfit_row_before_token_authority_audit() -> None:
    stage = probe.SimpleNamespace(
        inputs=probe.training.RaggedRows(
            [
                np.asarray([1, 2, 3], dtype=np.int32),
                np.asarray([4, 5], dtype=np.int32),
            ],
            dtype=np.int32,
            standard_width=3,
        ),
        labels=probe.training.RaggedRows(
            [
                np.asarray([2, 3, 4], dtype=np.int32),
                np.asarray([5, 6], dtype=np.int32),
            ],
            dtype=np.int32,
            standard_width=3,
        ),
        mask=probe.training.RaggedRows(
            [
                np.asarray([0, 1, 1], dtype=np.uint8),
                np.asarray([0, 1], dtype=np.uint8),
            ],
            dtype=np.uint8,
            standard_width=3,
        ),
        loss_mask=probe.training.RaggedRows(
            [
                np.asarray([0, 1, 1], dtype=np.float32),
                np.asarray([0, 1], dtype=np.float32),
            ],
            dtype=np.float32,
            standard_width=3,
        ),
        sample_weights=np.asarray([1.0, 1.0], dtype=np.float32),
        training_row_ids=("compiler-long", "compiler-short"),
        kerc_coverage_labels=(
            ("objective:surface_to_kernel_program_v1",),
            ("objective:surface_to_kernel_program_v1",),
        ),
        receipt={"policy": "fixture"},
    )

    selected, scope = probe.replay_training_stage_row_scope(
        stage,
        {
            "kerc_overfit_rows_per_objective": 1,
            "kerc_delta_stage_only": 1,
        },
    )

    assert selected.training_row_ids == ("compiler-short",)
    assert scope == {
        "stage_only_token_supervised_row_count": 1,
        "stage_only_zero_token_authority_rows_excluded": 0,
    }


def test_kernel_phase_replay_seed_includes_exact_prior_optimizer_steps() -> None:
    assert probe.kernel_phase_replay_seed(
        {
            "effective_training_seed": 20260722,
            "optimizer_steps": 4598,
        },
        {"optimizer_steps": 32},
        0,
    ) == 20265288
    assert probe.kernel_phase_replay_seed(
        {
            "effective_training_seed": 20260722,
            "optimizer_steps": 4598,
        },
        {
            "optimizer_steps": 32,
            "data_cursor_start": {
                "policy": "project_theseus_training_data_cursor_v1",
                "seed": 20260722,
            },
        },
        0,
    ) == 20260722


def test_segmented_sampler_replay_contract_recovers_cumulative_authority() -> None:
    assert probe.segmented_sampler_replay_contract(
        {
            "optimizer_steps": 4598,
            "current_kernel_phase_optimizer_positions": 11900,
            "current_kernel_phase_position_accounting_reset": True,
        },
        {
            "optimizer_steps": 1,
            "target_positions_consumed": 406,
            "coverage_first_sampling": {
                "cumulative_across_fresh_process_segments": True,
                "planning_capacity": 24,
            },
        },
        {
            "candidate_scratch_resume_policy": (
                "exact_fresh_process_segment_v1"
            ),
            "continuation_source_optimizer_steps": 4566,
        },
    ) == {
        "optimizer_steps": 32,
        "optimizer_positions": 11900,
        "planning_capacity": 24,
        "segment_steps": 1,
    }


def test_segmented_sampler_replay_contract_rebases_a_new_objective_lineage() -> None:
    assert probe.segmented_sampler_replay_contract(
        {
            "plan_sha256": "new-plan",
            "optimizer_steps": 4630,
            "current_kernel_phase_optimizer_positions": 23800,
            "current_kernel_phase_position_accounting_reset": True,
        },
        {
            "optimizer_steps": 1,
            "target_positions_consumed": 406,
            "coverage_first_sampling": {
                "cumulative_across_fresh_process_segments": True,
                "planning_capacity": 24,
            },
        },
        {
            "candidate_scratch_resume_policy": (
                "exact_fresh_process_segment_v1"
            ),
            "continuation_source_optimizer_steps": 4598,
        },
        source_result={
            "plan_sha256": "old-plan",
            "current_kernel_phase_optimizer_positions": 11900,
            "current_kernel_phase_position_accounting_reset": True,
            "phases": {
                "kernel_english": {
                    "coverage_first_sampling": {
                        "observed_label_counts": {
                            "objective:compiler": 32
                        }
                    }
                }
            },
        },
    ) == {
        "optimizer_steps": 32,
        "optimizer_positions": 11900,
        "planning_capacity": 24,
        "segment_steps": 1,
        "source_plan_sha256": "old-plan",
        "source_current_kernel_phase_optimizer_positions": 11900,
        "source_coverage_observed_label_counts": {
            "objective:compiler": 32
        },
    }


def test_segmented_sampler_replay_contract_keeps_already_reset_destination() -> None:
    assert probe.segmented_sampler_replay_contract(
        {
            "plan_sha256": "new-plan",
            "optimizer_steps": 4662,
            "current_kernel_phase_optimizer_positions": 11900,
            "current_kernel_phase_position_accounting_reset": True,
        },
        {
            "optimizer_steps": 1,
            "target_positions_consumed": 406,
            "coverage_first_sampling": {
                "cumulative_across_fresh_process_segments": True,
                "planning_capacity": 24,
            },
        },
        {
            "candidate_scratch_resume_policy": (
                "exact_fresh_process_segment_v1"
            ),
            "continuation_source_optimizer_steps": 4630,
        },
        source_result={
            "plan_sha256": "old-plan",
            "current_kernel_phase_optimizer_positions": 23800,
            "current_kernel_phase_position_accounting_reset": True,
        },
    ) == {
        "optimizer_steps": 32,
        "optimizer_positions": 11900,
        "planning_capacity": 24,
        "segment_steps": 1,
    }


def test_one_step_probe_preserves_population_coverage_planning_capacity() -> None:
    assert probe.exact_coverage_planning_capacity(
        {
            "optimizer_steps": 1,
            "coverage_first_sampling": {
                "capacity": 1,
                "planning_capacity": 24,
            },
        },
        requested_steps=1,
        replay_step_limit=1,
        stage_row_count=24,
        segmented_contract=None,
    ) == 24
    with pytest.raises(ValueError, match="outside the exact staged population"):
        probe.exact_coverage_planning_capacity(
            {
                "coverage_first_sampling": {
                    "planning_capacity": 25,
                },
            },
            requested_steps=1,
            replay_step_limit=1,
            stage_row_count=24,
            segmented_contract=None,
        )


def test_segmented_repeat_replay_caps_required_coverage_at_population() -> None:
    assert probe.exact_coverage_planning_capacity(
        {
            "optimizer_steps": 1,
            "coverage_first_sampling": {
                "capacity": 1,
                "planning_capacity": 24,
            },
        },
        requested_steps=1,
        replay_step_limit=32,
        stage_row_count=24,
        segmented_contract={
            "optimizer_steps": 32,
            "optimizer_positions": 11900,
            "planning_capacity": 24,
            "segment_steps": 1,
        },
    ) == 24
    with pytest.raises(ValueError, match="outside the exact staged population"):
        probe.exact_coverage_planning_capacity(
            {
                "coverage_first_sampling": {
                    "planning_capacity": 23,
                },
            },
            requested_steps=1,
            replay_step_limit=32,
            stage_row_count=24,
            segmented_contract={
                "optimizer_steps": 32,
                "optimizer_positions": 11900,
                "planning_capacity": 23,
                "segment_steps": 1,
            },
        )


def test_retained_row_probe_cli_is_free_generation_only() -> None:
    source = inspect.getsource(probe.main)
    assert "--retained-row-report" in source
    assert "explicit retained row report requires free generation" in source
    assert "K5 row-selection reports are mutually exclusive" in source


def test_training_row_panel_requires_exact_admitted_rows() -> None:
    row = {
        "admitted_training_row": True,
        "row_id": "row-a",
    }
    report = {
        "policy": probe.POLICY,
        "qualification_state": "TEACHER_FORCED_DIAGNOSTIC_ONLY",
        "rows": [row, {**row, "row_id": "row-b"}],
        "public_benchmark_prompts_used": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
    }
    assert probe.exact_training_row_panel_ids(report) == (
        "row-a",
        "row-b",
    )
    with pytest.raises(ValueError, match="exact admitted training panel"):
        probe.exact_training_row_panel_ids(
            {
                **report,
                "rows": [row, {**row, "row_id": "row-a"}],
            }
        )
    with pytest.raises(ValueError, match="exact admitted training panel"):
        probe.exact_training_row_panel_ids(
            {
                **report,
                "rows": [
                    row,
                    {
                        **row,
                        "row_id": "row-b",
                        "admitted_training_row": False,
                    },
                ],
            }
        )


def test_segmented_sampler_replay_contract_is_inactive_for_ordinary_run() -> None:
    assert probe.segmented_sampler_replay_contract(
        {"optimizer_steps": 32},
        {
            "optimizer_steps": 32,
            "coverage_first_sampling": {
                "cumulative_across_fresh_process_segments": False,
            },
        },
        {},
    ) is None


def test_segmented_sampler_replay_contract_rejects_missing_cumulative_state() -> None:
    with pytest.raises(
        ValueError,
        match="lacks cumulative sampler authority",
    ):
        probe.segmented_sampler_replay_contract(
            {
                "optimizer_steps": 4598,
                "current_kernel_phase_optimizer_positions": 406,
                "current_kernel_phase_position_accounting_reset": True,
            },
            {
                "optimizer_steps": 1,
                "target_positions_consumed": 406,
                "coverage_first_sampling": {
                    "cumulative_across_fresh_process_segments": True,
                    "planning_capacity": 24,
                },
            },
            {
                "candidate_scratch_resume_policy": (
                    "exact_fresh_process_segment_v1"
                ),
                "continuation_source_optimizer_steps": 4566,
            },
        )


def test_dense_training_epoch_order_preserves_mutating_shuffle_cursor() -> None:
    first = probe.dense_training_epoch_order(
        list(range(24)),
        seed=20260722,
        probabilities=None,
        minimum_stratum_coverage=False,
    )
    second = probe.dense_training_epoch_order(
        first,
        seed=20260723,
        probabilities=None,
        minimum_stratum_coverage=False,
    )
    reset_second = probe.dense_training_epoch_order(
        list(range(24)),
        seed=20260723,
        probabilities=None,
        minimum_stratum_coverage=False,
    )

    assert second != reset_second
    assert sorted(second) == list(range(24))


def test_exact_stage_teacher_forced_arrays_uses_transport_authority() -> None:
    inputs, expected, authority = probe.exact_stage_teacher_forced_arrays(
        np.asarray([7, 8, 9, 10], dtype=np.int32),
        np.asarray([8, 9, 10, 11], dtype=np.int32),
        np.asarray([0, 1, 1, 0], dtype=np.float32),
    )

    assert inputs.tolist() == [7, 8, 9, 10]
    assert expected.tolist() == [9, 10]
    assert authority.tolist() == [False, True, True, False]
    assert probe.align_exact_stage_predictions(
        np.asarray([20, 21, 22, 23]), authority
    ).tolist() == [21, 22]
    assert probe.align_exact_stage_predictions(
        np.asarray([21, 22]), authority
    ).tolist() == [21, 22]


def test_gradient_parameter_groups_keep_bridge_and_stage_ownership_separate() -> None:
    assert (
        probe.gradient_parameter_group("layers.0.source_attention.query_proj.weight")
        == "source_conditioned_bridge"
    )
    assert (
        probe.gradient_parameter_group("kerc_stage_adapters.2.down.weight")
        == "stage_conditioning"
    )
    assert (
        probe.gradient_parameter_group("kerc_kernel_output.weight")
        == "kernel_output"
    )
    assert (
        probe.gradient_parameter_group("kerc_surface_output.bias")
        == "surface_output"
    )


def test_gradient_pair_metrics_report_conflict_and_group_cosines() -> None:
    left = {
        "kerc_kernel_output.weight": np.asarray([1.0, 0.0], dtype=np.float32),
        "kerc_stage_adapters.0.down.weight": np.asarray([0.0, 2.0], dtype=np.float32),
    }
    right = {
        "kerc_kernel_output.weight": np.asarray([-1.0, 0.0], dtype=np.float32),
        "kerc_stage_adapters.0.down.weight": np.asarray([0.0, 2.0], dtype=np.float32),
    }
    metrics = probe.gradient_pair_metrics(left, right)
    assert metrics["kernel_output"]["cosine"] == -1.0
    assert metrics["stage_conditioning"]["cosine"] == 1.0
    assert metrics["all_trainable"]["cosine"] == 0.6


def test_gradient_pair_metrics_reject_inventory_drift() -> None:
    try:
        probe.gradient_pair_metrics(
            {"kerc_kernel_output.weight": np.ones(1, dtype=np.float32)},
            {"kerc_surface_output.weight": np.ones(1, dtype=np.float32)},
        )
    except ValueError as exc:
        assert "exact parameter inventory" in str(exc)
    else:
        raise AssertionError("gradient inventory drift was accepted")


def test_teacher_forced_aggregate_preserves_objective_and_frequency_denominators() -> None:
    rows = [
        {
            "objective": "surface_to_kernel_program_v1",
            "sampled_training_row": True,
            "sampled_optimizer_step_count": 2,
            "teacher_forced_top1_correct": 3,
            "teacher_forced_top1_total": 5,
            "teacher_forced_eos_top1": True,
            "teacher_forced_accuracy_by_target_frequency": {
                "singleton": {"correct": 1, "total": 2},
                "repeated": {"correct": 2, "total": 3},
            },
        },
        {
            "objective": "surface_to_kernel_program_v1",
            "sampled_training_row": False,
            "sampled_optimizer_step_count": 0,
            "teacher_forced_top1_correct": 2,
            "teacher_forced_top1_total": 5,
            "teacher_forced_eos_top1": False,
            "teacher_forced_accuracy_by_target_frequency": {
                "singleton": {"correct": 0, "total": 2},
                "repeated": {"correct": 2, "total": 3},
            },
        },
    ]

    summary = probe.aggregate_teacher_forced_rows(rows)[
        "surface_to_kernel_program_v1"
    ]

    assert summary["row_count"] == 2
    assert summary["sampled_row_count"] == 1
    assert summary["optimizer_step_count"] == 2
    assert summary["top1_accuracy"] == 0.5
    assert summary["eos_top1_rate"] == 0.5
    assert summary["frequency"]["singleton"] == {
        "correct": 1,
        "total": 4,
        "accuracy": 0.25,
    }
    assert summary["frequency"]["repeated"] == {
        "correct": 4,
        "total": 6,
        "accuracy": 0.66666667,
    }
