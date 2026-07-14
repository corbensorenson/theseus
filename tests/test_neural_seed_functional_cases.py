from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_functional_cases import ARMS, materialize_cases


def config() -> dict:
    return json.loads((ROOT / "configs/neural_seed_functional_utility.json").read_text())


def test_case_contract_is_balanced_deterministic_and_answer_blind() -> None:
    first = materialize_cases(config())
    second = materialize_cases(config())

    assert first == second
    assert len(first) == 160
    assert len({row["case_id"] for row in first}) == 160
    for arm in ARMS:
        assert sum(row["arm_id"] == arm for row in first) == 32
    forbidden = {
        "task_family", "verifier", "expected", "tests", "hidden_tests",
        "return_shape", "required_constructs", "solution", "answer",
    }
    for row in first:
        assert set(row["model_visible"]) == {"case_id", "arm_id", "prompt"}
        assert not (set(row["model_visible"]) & forbidden)


def test_javascript_prompts_do_not_contain_typescript_annotations() -> None:
    cases = materialize_cases(config())
    javascript = [
        row for row in cases
        if row["arm_id"] == "javascript_typescript"
        and row["variant"] % 2 == 0
        and row["task_family"] != "repository_edit"
    ]
    assert javascript
    assert all(": number" not in row["prompt"] and "Record<" not in row["prompt"] for row in javascript)
