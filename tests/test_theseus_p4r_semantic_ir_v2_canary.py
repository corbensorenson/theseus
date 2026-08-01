from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4r_semantic_ir_v2_canary as canary  # noqa: E402


def case() -> dict:
    return {
        "case_id": "tuple_add",
        "natural_request": "Change VALUES from (1, 2) to (1, 2, 3).",
        "source": "VALUES = (1, 2)\n",
        "expected_source": "VALUES = (1, 2, 3)\n",
        "target_node_type": "Tuple",
        "obligations": [
            {"id": "O1", "kind": "require", "text": "add integer 3"},
            {"id": "O2", "kind": "preserve", "text": "preserve 1 and 2"},
            {"id": "O3", "kind": "non_goal", "text": "no other changes"},
        ],
        "obligation_dependencies": [{"before": "O2", "after": "O1"}],
    }


def test_canary_selects_one_stable_target_and_renders_labeled_fields(
    tmp_path: Path,
) -> None:
    value = case()
    (tmp_path / "sample.py").write_text(value["source"], encoding="utf-8")
    task = canary.build_task(value)
    symbols = p4.semantic_symbol_table(tmp_path, task)
    target = canary.select_target(symbols, "Tuple")
    prompt = canary.render_prompt(value, task, symbols, target)

    assert f"TARGET NODE {target['id']}" in prompt
    assert f"TARGET NODE_SHA {target['sha256']}" in prompt
    assert "ALL_OBLIGATIONS <comma-separated exact obligation ids>" in prompt
    assert "Use exactly one UNIT" in prompt
    assert "without surrounding leading indentation" in prompt


def test_canary_task_contains_no_expected_answer_field() -> None:
    task = canary.build_task(case())

    assert "expected_source" not in task
    assert "solution" not in task
    assert [row["id"] for row in task["obligations"]] == ["O1", "O2", "O3"]
