from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_semantic_ir_v2 as v2  # noqa: E402
import theseus_semantic_ir_v2r1 as v2r1  # noqa: E402


def task() -> dict:
    return {
        "allowed_effect_paths": ["sample.py"],
        "candidate_visible_context": {
            "reads": [{"path": "sample.py", "start_line": 1, "end_line": 5}]
        },
        "semantic_ir_contract": {"maximum_symbol_nodes": 40},
        "obligations": [
            {"id": "O1", "kind": "require", "text": "set safe"},
            {"id": "O2", "kind": "preserve", "text": "keep MODE"},
            {"id": "O3", "kind": "non_goal", "text": "no extras"},
        ],
        "obligation_dependencies": [{"before": "O2", "after": "O1"}],
    }


def headerless(root: Path) -> str:
    symbols = p4.semantic_symbol_table(root, task())
    target = next(row for row in symbols["nodes"] if row["node_type"] == "Assign")
    return (
        f"SOURCE {symbols['source_digest']}\n"
        "ALL_OBLIGATIONS O1,O2,O3\n"
        "UNIT U1\nOBLIGATIONS O1,O2,O3\nOP REPLACE\nPATH sample.py\n"
        f"NODE {target['id']}\nNODE_SHA {target['sha256']}\n"
        "<<<\nMODE = \"safe\"\n>>>\nEND_UNIT\nLOSS NONE\nEND"
    )


def test_headerless_bound_transport_is_inferred_and_lowered(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text('MODE = "fast"\n', encoding="utf-8")
    text = headerless(tmp_path)

    assert v2.complete(text) is False
    assert v2r1.complete(text) is True
    result = v2r1.parse(text, task(), tmp_path)
    assert result["faults"] == []
    assert result["semantic_receipt"]["version_header_inferred_from_bound_parser"] is True


def test_header_inference_does_not_accept_untyped_or_unterminated_text() -> None:
    assert v2r1.complete("SOURCE abc\nEND") is False
    assert v2r1.complete("MODE = safe") is False
    normalized, inferred = v2r1.normalize("MODE = safe")
    assert normalized == "MODE = safe"
    assert inferred is False

