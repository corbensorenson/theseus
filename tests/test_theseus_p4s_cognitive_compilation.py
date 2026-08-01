from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4s_cognitive_compilation as p4s  # noqa: E402


def test_completion_accepts_labeled_v2_and_existing_control_envelopes() -> None:
    labeled = (
        "THESEUS_SEMANTIC_IR_V2\nSOURCE " + "a" * 64
        + "\nALL_OBLIGATIONS O1\nUNIT U1\nOBLIGATIONS O1\nOP REPLACE\n"
        "PATH sample.py\nNODE N-ONE\nNODE_SHA " + "b" * 64
        + "\n<<<\nx = 2\n>>>\nEND_UNIT\nLOSS NONE\nEND"
    )
    direct = "THESEUS_EDIT_V1\nREPLACE sample.py 1 1\n<<<\nx = 2\n>>>\nEND"

    assert p4s.candidate_envelope_complete(labeled)
    assert p4s.candidate_envelope_complete(direct)


def test_semantic_scope_table_excludes_nested_statements_and_assignments() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sample.py").write_text(
            "MODE = 'old'\n\ndef change(value):\n    local = value + 1\n    if local:\n        return local\n",
            encoding="utf-8",
        )
        task = {
            "allowed_effect_paths": ["sample.py"],
            "candidate_visible_context": {
                "reads": [{"path": "sample.py", "start_line": 1, "end_line": 100}]
            },
            "semantic_ir_contract": {"maximum_symbol_nodes": 80},
            "obligations": [{"id": "O1", "kind": "require", "text": "change"}],
            "obligation_dependencies": [],
        }

        table = p4s.semantic_scope_symbol_table(root, task)

    assert [row["node_type"] for row in table["nodes"]] == ["Assign", "FunctionDef"]
    assert table["semantic_unit_policy"] == "p4s_complete_scope_v1"


def test_semantic_prompt_ends_in_labeled_output_shape() -> None:
    prompt = p4s.render_arm_prompt(
        p4.SEMANTIC,
        {"natural_request": "change behavior"},
        "[INFORMATION_MATCHED_OBLIGATIONS]\nO1 REQUIRE: change",
        {},
    )

    assert prompt.startswith(p4s.SEMANTIC_PROMPT_MARKER)
    assert prompt.endswith("LOSS NONE\nEND")
    assert "THESEUS_SEMANTIC_IR_V2" in prompt
    assert "complete replacement source" in prompt


def test_p4s_runner_has_no_project_selected_quality_cap() -> None:
    assert p4s.MODEL_CONTEXT_TOKENS == 262144
    assert "fresh p4 decision" in p4s.__doc__.lower()
