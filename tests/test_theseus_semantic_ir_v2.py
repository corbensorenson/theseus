from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_semantic_ir_v2 as ir_v2  # noqa: E402


def task() -> dict:
    return {
        "allowed_effect_paths": ["sample.py"],
        "candidate_visible_context": {
            "reads": [{"path": "sample.py", "start_line": 1, "end_line": 5}]
        },
        "semantic_ir_contract": {"maximum_symbol_nodes": 40},
        "obligations": [
            {"id": "O1", "kind": "require", "text": "add three"},
            {"id": "O2", "kind": "preserve", "text": "keep one and two"},
            {"id": "O3", "kind": "non_goal", "text": "do not change choose"},
        ],
        "obligation_dependencies": [{"before": "O2", "after": "O1"}],
    }


def write_source(root: Path) -> None:
    (root / "sample.py").write_text(
        "VALUES = (1, 2)\n\ndef choose(value):\n    return value in VALUES\n",
        encoding="utf-8",
    )


def valid_ir(root: Path) -> str:
    symbols = p4.semantic_symbol_table(root, task())
    target = next(row for row in symbols["nodes"] if row["node_type"] == "Tuple")
    return (
        f"{ir_v2.HEADER}\n"
        f"SOURCE {symbols['source_digest']}\n"
        "ALL_OBLIGATIONS O1,O2,O3\n"
        "UNIT U1\n"
        "OBLIGATIONS O1,O2,O3\n"
        "OP REPLACE\n"
        "PATH sample.py\n"
        f"NODE {target['id']}\n"
        f"NODE_SHA {target['sha256']}\n"
        "<<<\n(1, 2, 3)\n>>>\n"
        "END_UNIT\n"
        "LOSS NONE\n"
        "END"
    )


def test_labeled_v2_canonicalizes_into_strict_v1_lowerer(tmp_path: Path) -> None:
    write_source(tmp_path)
    text = valid_ir(tmp_path)

    assert ir_v2.complete(text) is True
    result = ir_v2.parse(text, task(), tmp_path)
    assert result["faults"] == []
    assert result["canonical_v1"].startswith(p4.IR_HEADER)
    assert result["semantic_receipt"]["transport"] == (
        "theseus_semantic_ir_v2_labeled"
    )
    assert p2a.apply_actions(tmp_path, result["actions"]) == []
    assert "VALUES = (1, 2, 3)" in (tmp_path / "sample.py").read_text(
        encoding="utf-8"
    )


def test_v2_rejects_field_omission_identity_and_unparsed_text(tmp_path: Path) -> None:
    write_source(tmp_path)
    text = valid_ir(tmp_path)
    symbols = p4.semantic_symbol_table(tmp_path, task())
    mutations = (
        text.replace("OP REPLACE\n", "", 1),
        text.replace(symbols["source_digest"], "0" * 64, 1),
        text.replace("ALL_OBLIGATIONS O1,O2,O3", "ALL_OBLIGATIONS O1,O2", 1),
        text.replace("LOSS NONE", "UNPARSED\nLOSS NONE", 1),
    )

    for mutation in mutations:
        assert ir_v2.parse(mutation, task(), tmp_path)["faults"]


def test_v2_completion_is_syntax_only_but_requires_terminal_structure(
    tmp_path: Path,
) -> None:
    write_source(tmp_path)
    text = valid_ir(tmp_path)

    assert ir_v2.complete(text.removesuffix("\nEND")) is False
    assert ir_v2.complete("commentary\n" + text) is False
    assert ir_v2.complete("```text\n" + text + "\n```") is True
    assert "NODE_SHA" in ir_v2.grammar()
