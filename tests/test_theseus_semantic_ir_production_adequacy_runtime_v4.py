from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_production_adequacy_runtime_v4 as runtime  # noqa: E402


TASK = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_replacement_04_task.json"


def extract_consumed_parent(tmp_path: Path) -> tuple[Path, dict]:
    task = json.loads(TASK.read_text(encoding="utf-8"))
    root = tmp_path / "source"
    with tarfile.open(ROOT / task["source_archive"]) as archive:
        archive.extractall(root, filter="data")
    return root / task["source_archive_root"], task


def test_statement_inventory_exposes_small_target_inside_large_function(
    tmp_path: Path,
) -> None:
    root, task = extract_consumed_parent(tmp_path)
    task["semantic_ir_contract"]["maximum_symbol_nodes"] = 1_000_000
    task["semantic_ir_contract"]["maximum_semantic_scope_nodes"] = 1_000_000
    table = runtime.semantic_scope_symbol_table(root, task)

    target = next(
        row
        for row in table["nodes"]
        if row["path"] == "skbio/alignment/_pair.py"
        and row["start_line"] == 424
        and row["node_type"] == "Assign"
    )
    enclosing = next(
        row
        for row in table["nodes"]
        if row["node_type"] == "FunctionDef" and row["start_line"] == 35
    )

    assert target["end_line"] == 424
    assert enclosing["end_line"] == 467
    assert table["inventory_complete"] is True
    assert table["inventory_truncated"] is False
    assert table["statement_scope_metrics"]["project_selected_node_cap_applied"] is False


def test_local_insertion_lowers_against_one_statement_not_whole_function(
    tmp_path: Path,
) -> None:
    root, task = extract_consumed_parent(tmp_path)
    task["semantic_ir_contract"]["maximum_symbol_nodes"] = 1_000_000
    task["semantic_ir_contract"]["maximum_semantic_scope_nodes"] = 1_000_000
    table = runtime.semantic_scope_symbol_table(root, task)
    target = next(
        row
        for row in table["nodes"]
        if row["start_line"] == 424 and row["node_type"] == "Assign"
    )
    artifact = (
        f"{runtime.HEADER}\n"
        f"SOURCE {table['source_digest']}\n"
        "ALL_OBLIGATIONS O1,O2,O3\n"
        "CHANGE_OBLIGATIONS O1\n"
        "PRESERVE_OBLIGATIONS O2\n"
        "NON_GOAL_OBLIGATIONS O3\n"
        "UNIT U1\n"
        "OBLIGATIONS O1,O2\n"
        "OP INSERT_BEFORE\n"
        "PATH skbio/alignment/_pair.py\n"
        f"NODE {target['id']}\n"
        f"NODE_SHA {target['sha256']}\n"
        "<<<\n"
        "    if atol is None:\n"
        "        atol = 0.0\n"
        "    atol = dtype(atol)\n"
        ">>>\n"
        "END_UNIT\n"
        "LOSS NONE\n"
        "END"
    )
    parsed = runtime.parse(artifact, task, root)

    assert parsed["faults"] == []
    assert parsed["actions"][0]["start_line"] == 424
    assert parsed["actions"][0]["end_line"] == 424
    assert len(parsed["actions"][0]["replacement"].splitlines()) == 4


def test_statement_inventory_fails_closed_instead_of_truncating(tmp_path: Path) -> None:
    root, task = extract_consumed_parent(tmp_path)
    task["semantic_ir_contract"]["maximum_symbol_nodes"] = 1
    task["semantic_ir_contract"]["maximum_semantic_scope_nodes"] = 1
    with pytest.raises(p2a.InstrumentFault, match="would_be_truncated"):
        runtime.semantic_scope_symbol_table(root, task)


def test_prompt_requires_smallest_sufficient_statement_target() -> None:
    task = json.loads(TASK.read_text(encoding="utf-8"))
    prompt = runtime.render_prompt(task, "[VISIBLE]")
    assert "smallest exact node" in prompt
    assert "do not reproduce a containing function" in prompt
