from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4s_cognitive_compilation as p4s  # noqa: E402
import theseus_semantic_ir_production as production  # noqa: E402


def task() -> dict:
    return {
        "natural_request": "Make choose accept three while preserving prior values.",
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


def valid_ir(root: Path, *, path_surface: str = "sample.py") -> str:
    symbols = p4s.semantic_scope_symbol_table(root, task())
    target = next(row for row in symbols["nodes"] if row["node_type"] == "Assign")
    return (
        f"{production.HEADER}\n"
        f"SOURCE {symbols['source_digest']}\n"
        "ALL_OBLIGATIONS O1,O2,O3\n"
        "CHANGE_OBLIGATIONS O1\n"
        "PRESERVE_OBLIGATIONS O2\n"
        "NON_GOAL_OBLIGATIONS O3\n"
        "UNIT U1\n"
        "OBLIGATIONS O1,O2\n"
        "OP REPLACE\n"
        f"PATH {path_surface}\n"
        f"NODE {target['id']}\n"
        f"NODE_SHA {target['sha256']}\n"
        "<<<\nVALUES = (1, 2, 3)\n>>>\n"
        "END_UNIT\n"
        "LOSS NONE\n"
        "END"
    )


def test_role_aware_schema_lowers_change_without_mutating_non_goal(tmp_path: Path) -> None:
    write_source(tmp_path)
    parsed = production.parse(valid_ir(tmp_path), task(), tmp_path)

    assert parsed["faults"] == []
    assert parsed["semantic_receipt"]["obligation_roles"] == {
        "change": ["O1"],
        "preserve": ["O2"],
        "non_goal": ["O3"],
    }
    assert parsed["units"][0]["change_obligation_ids"] == ["O1"]
    assert parsed["units"][0]["preserve_obligation_ids"] == ["O2"]
    assert p2a.apply_actions(tmp_path, parsed["actions"]) == []
    assert "VALUES = (1, 2, 3)" in (tmp_path / "sample.py").read_text(
        encoding="utf-8"
    )


def test_change_coverage_dependency_and_non_goal_attachment_fail_closed(tmp_path: Path) -> None:
    write_source(tmp_path)
    base = valid_ir(tmp_path)
    mutations = {
        "semantic_change_obligation_coverage_incomplete": base.replace(
            "OBLIGATIONS O1,O2", "OBLIGATIONS O2"
        ),
        "semantic_unit_dependency_not_closed": base.replace(
            "OBLIGATIONS O1,O2", "OBLIGATIONS O1"
        ),
        "semantic_non_goal_attached_to_mutation": base.replace(
            "OBLIGATIONS O1,O2", "OBLIGATIONS O1,O2,O3"
        ),
    }
    for expected, text in mutations.items():
        result = production.parse(text, task(), tmp_path)
        assert expected in result["faults"]
        assert result["actions"] == []


def test_role_ledger_is_exact_and_cannot_reclassify_obligations(tmp_path: Path) -> None:
    write_source(tmp_path)
    base = valid_ir(tmp_path)
    for mutation in (
        base.replace("CHANGE_OBLIGATIONS O1", "CHANGE_OBLIGATIONS O2"),
        base.replace("PRESERVE_OBLIGATIONS O2", "PRESERVE_OBLIGATIONS O1"),
        base.replace("NON_GOAL_OBLIGATIONS O3", "NON_GOAL_OBLIGATIONS O2"),
        base.replace("ALL_OBLIGATIONS O1,O2,O3", "ALL_OBLIGATIONS O1,O2"),
    ):
        assert production.parse(mutation, task(), tmp_path)["faults"]


def test_redundant_path_coordinates_require_exact_candidate_visible_identity(
    tmp_path: Path,
) -> None:
    write_source(tmp_path)
    symbols = p4s.semantic_scope_symbol_table(tmp_path, task())
    target = next(row for row in symbols["nodes"] if row["node_type"] == "Assign")
    coordinate_path = (
        f"sample.py:{target['start_line']}:{target['start_col']}-"
        f"{target['end_line']}:{target['end_col']}"
    )
    parsed = production.parse(
        valid_ir(tmp_path, path_surface=coordinate_path), task(), tmp_path
    )
    assert parsed["faults"] == []
    assert parsed["actions"][0]["path"] == "sample.py"
    assert parsed["semantic_receipt"]["path_normalization"][0]["state"] == (
        "REDUNDANT_EXACT_COORDINATES_REMOVED"
    )

    wrong = valid_ir(tmp_path, path_surface=coordinate_path.replace(":0-", ":1-"))
    assert "semantic_target_identity_invalid" in production.parse(
        wrong, task(), tmp_path
    )["faults"]


def test_malformed_hash_invented_operation_and_unparsed_text_are_rejected(
    tmp_path: Path,
) -> None:
    write_source(tmp_path)
    base = valid_ir(tmp_path)
    node_hash = next(
        line.split(" ", 1)[1]
        for line in base.splitlines()
        if line.startswith("NODE_SHA ")
    )
    mutations = (
        base.replace(node_hash, node_hash[:-8]),
        base.replace("OP REPLACE", "OP COPY"),
        base.replace("LOSS NONE", "COMMENTARY\nLOSS NONE"),
    )
    for mutation in mutations:
        result = production.parse(mutation, task(), tmp_path)
        assert result["faults"]
        assert result["actions"] == []


def test_canonicalizer_changes_delimiters_only_and_preserves_replacement_bytes(
    tmp_path: Path,
) -> None:
    write_source(tmp_path)
    text = valid_ir(tmp_path).replace(
        "ALL_OBLIGATIONS O1,O2,O3", "ALL_OBLIGATIONS ['O1', 'O2', 'O3']"
    ).replace("OBLIGATIONS O1,O2", "OBLIGATIONS O1 O2")
    normalized, receipt = production.canonicalize_with_receipt(text)

    assert "ALL_OBLIGATIONS O1,O2,O3" in normalized
    assert "OBLIGATIONS O1,O2" in normalized
    assert "VALUES = (1, 2, 3)" in normalized
    assert receipt["identifier_values_invented"] == 0
    assert receipt["replacement_source_touched"] is False
    assert production.parse(text, task(), tmp_path)["faults"] == []


def test_complete_repair_prompt_has_no_project_selected_tail_cap(tmp_path: Path) -> None:
    write_source(tmp_path)
    original = production.render_prompt(task(), "candidate-visible context")
    first = valid_ir(tmp_path) + ("\n" + "x" * 20000)
    stdout = "BEGIN" + ("v" * 5000) + "END"
    prompt = production.render_repair_prompt(
        original,
        first,
        ["semantic_change_obligation_coverage_incomplete"],
        {
            "apply_faults": [],
            "visible_verifier": {
                "returncode": 1,
                "stdout_tail": stdout,
                "stderr_tail": "",
            },
        },
    )
    assert first in prompt
    assert stdout in prompt
    assert "THESEUS_SEMANTIC_IR_V3" in prompt


def test_empty_preserve_or_non_goal_roles_are_explicit_none() -> None:
    value = task()
    value["obligations"] = [value["obligations"][0]]
    value["obligation_dependencies"] = []
    prompt = production.render_prompt(value, "context")
    assert "PRESERVE_OBLIGATIONS NONE" in prompt
    assert "NON_GOAL_OBLIGATIONS NONE" in prompt
