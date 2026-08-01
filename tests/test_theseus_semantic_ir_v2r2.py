from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_semantic_ir_v2r2 as v2r2  # noqa: E402


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


def test_obligation_list_surface_variants_normalize_without_semantic_change(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.py").write_text('MODE = "fast"\n', encoding="utf-8")
    canonical = headerless(tmp_path)
    variants = (
        ("[O1, O2, O3]", "unquoted_bracket_list"),
        ('["O1", "O2", "O3"]', "quoted_bracket_list"),
        ("['O1', 'O2', 'O3']", "quoted_bracket_list"),
        ("O1 O2 O3", "whitespace_list"),
        ("O1, O2, O3", "comma_list"),
    )
    for surface, surface_class in variants:
        text = canonical.replace("O1,O2,O3", surface)
        normalized, receipt = v2r2.normalize_with_receipt(text)
        assert "ALL_OBLIGATIONS O1,O2,O3" in normalized
        assert "OBLIGATIONS O1,O2,O3" in normalized
        assert receipt["rejected_field_count"] == 0
        assert {row["surface_class"] for row in receipt["fields"]} == {
            surface_class
        }
        assert receipt["answer_bearing_transformation"] is False
        assert receipt["identifier_values_invented"] == 0
        assert receipt["identifier_order_preserved"] is True
        assert v2r2.complete(text) is True
        parsed = v2r2.parse(text, task(), tmp_path)
        assert parsed["faults"] == []
        custody = parsed["semantic_receipt"]["obligation_list_normalization"]
        assert custody["path_node_operation_source_digest_touched"] is False
        assert custody["replacement_source_touched"] is False


def test_normalizer_rejects_ambiguous_or_executable_surfaces() -> None:
    rejected = (
        "[O1, O2",
        "O1,,O2",
        "O1, O2 O3",
        "['O1', O2]",
        "[__import__('os')]",
        "{'O1': 'O2'}",
        "O1;O2",
        "o1,o2",
        "\"O1\",\"O2\"",
    )
    for surface in rejected:
        assert v2r2.canonical_obligation_list(surface) is None


def test_normalizer_preserves_order_duplicates_and_replacement_bytes(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.py").write_text('MODE = "fast"\n', encoding="utf-8")
    text = headerless(tmp_path).replace(
        "ALL_OBLIGATIONS O1,O2,O3",
        "ALL_OBLIGATIONS [O3, O1, O1]",
    ).replace(
        'MODE = "safe"',
        'MODE = "safe"\nOBLIGATIONS [O9, O8]',
    )

    normalized, receipt = v2r2.normalize_with_receipt(text)

    assert "ALL_OBLIGATIONS O3,O1,O1" in normalized
    assert 'MODE = "safe"\nOBLIGATIONS [O9, O8]' in normalized
    assert receipt["identifier_order_preserved"] is True
    assert receipt["replacement_source_touched"] is False
    assert v2r2.parse(text, task(), tmp_path)["faults"]


def test_unknown_obligation_is_not_repaired_or_silently_admitted(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.py").write_text('MODE = "fast"\n', encoding="utf-8")
    text = headerless(tmp_path).replace("O1,O2,O3", "[O1, O2, O999]")

    normalized, receipt = v2r2.normalize_with_receipt(text)

    assert "O1,O2,O999" in normalized
    assert receipt["identifier_values_invented"] == 0
    assert v2r2.parse(text, task(), tmp_path)["faults"]
