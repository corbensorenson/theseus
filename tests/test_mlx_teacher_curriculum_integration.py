from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import strict_generator_mlx_pretraining_probe as pretraining  # noqa: E402
import strict_generator_mlx_private_adaptation as adaptation  # noqa: E402


def teacher_row(task_id: str = "teacher.private.v1") -> dict:
    return {
        "task_id": task_id,
        "prompt": "Return a running private state digest for visible records.",
        "entry_point": "solve_private",
        "solution_body": "state = 0\nfor item in records:\n    state += item\nreturn state",
        "tests": "assert solve_private([1, 2]) == 3",
        "source_kind": "teacher_distillation",
        "teacher_generated": True,
        "teacher_manifest_row_id": f"manifest.{task_id}",
        "public_benchmark": False,
        "public_prompt": False,
    }


def test_fresh_pretraining_injects_teacher_rows_without_source_leakage() -> None:
    staged = {
        "active": True,
        "examples": [
            {"path": "base.py", "function": "base", "source_text": "Implement base", "body": "return 1"}
        ],
        "source_vocab_extension_texts": ["Implement base"],
        "target_vocab_extension_bodies": ["return 1"],
        "summary": {"row_example_count": 1},
    }

    result = pretraining.inject_governed_teacher_examples(
        staged,
        teacher_rows=[teacher_row()],
        teacher_summary={
            "enabled": True,
            "gate_green": True,
            "available_code_lm_training_rows": 1,
            "holdout_family_code_lm_training_rows": 0,
            "external_inference_calls": 1,
            "manifest": "reports/teacher_distillation_manifest.json",
        },
        text_views={"sts_on": ["prompt", "entry_point"]},
    )

    teacher_example = result["examples"][-1]
    assert result["summary"]["governed_teacher_injected_row_count"] == 1
    assert teacher_example["teacher_generated"] is True
    assert "running private state digest" in teacher_example["source_text"]
    assert "assert solve_private" not in teacher_example["source_text"]
    assert teacher_example["body"] not in teacher_example["source_text"]
    assert result["teacher_training"]["runtime_external_inference_calls"] == 0
    assert result["teacher_training"]["public_training_rows"] == 0


def test_adaptation_reserves_teacher_rows_for_train_only() -> None:
    teacher = teacher_row()
    base = [
        {"task_id": f"base.{index}", "prompt": f"base {index}", "entry_point": "base", "solution_body": "return 1"}
        for index in range(10)
    ]

    train_rows, eval_rows, audit = adaptation.reserve_governed_teacher_train_rows(
        selected_rows=base,
        eligible_rows=[*base, teacher],
        injected_teacher_rows=[teacher],
        max_train_rows=6,
        max_eval_rows=2,
    )

    assert train_rows[0]["task_id"] == teacher["task_id"]
    assert len(train_rows) == 6
    assert all(row["task_id"] != teacher["task_id"] for row in eval_rows)
    assert audit["reserved_train_position_count"] == 1
    assert audit["teacher_rows_in_eval_count"] == 0
