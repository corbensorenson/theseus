from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_production_adequacy_runtime as runtime  # noqa: E402


def task() -> dict:
    return {
        "natural_request": "Add helper.py while preserving sample.py.",
        "allowed_effect_paths": ["sample.py", "helper.py"],
        "candidate_visible_context": {
            "reads": [{"path": "sample.py", "start_line": 1, "end_line": 4}],
            "missing_allowed_effect_paths": ["helper.py"],
        },
        "semantic_ir_contract": {
            "maximum_symbol_nodes": 40,
            "maximum_semantic_scope_nodes": 40,
        },
        "obligations": [
            {"id": "O1", "kind": "require", "text": "add helper"},
            {"id": "O2", "kind": "preserve", "text": "preserve sample"},
            {"id": "O3", "kind": "non_goal", "text": "no other files"},
        ],
        "obligation_dependencies": [{"before": "O2", "after": "O1"}],
    }


def write_source(root: Path) -> None:
    (root / "sample.py").write_text(
        "VALUES = (1, 2)\n\ndef choose(value):\n    return value in VALUES\n",
        encoding="utf-8",
    )


def valid_ir(root: Path, *, path: str = "helper.py") -> str:
    symbols = runtime.semantic_scope_symbol_table(root, task())
    return (
        f"{runtime.HEADER}\n"
        f"SOURCE {symbols['source_digest']}\n"
        "ALL_OBLIGATIONS O1,O2,O3\n"
        "CHANGE_OBLIGATIONS O1\n"
        "PRESERVE_OBLIGATIONS O2\n"
        "NON_GOAL_OBLIGATIONS O3\n"
        "UNIT U1\n"
        "OBLIGATIONS O1,O2\n"
        "OP CREATE_FILE\n"
        f"PATH {path}\n"
        "<<<\n"
        "def helper(value):\n"
        "    return value\n"
        ">>>\n"
        "END_UNIT\n"
        "LOSS NONE\n"
        "END"
    )


def test_create_file_unit_is_identity_bound_and_applies_in_disposable_root(
    tmp_path: Path,
) -> None:
    write_source(tmp_path)
    parsed = runtime.parse(valid_ir(tmp_path), task(), tmp_path)

    assert parsed["faults"] == []
    assert parsed["actions"] == [{
        "op": "CREATE_FILE",
        "path": "helper.py",
        "replacement": "def helper(value):\n    return value",
    }]
    assert parsed["units"][0]["node_id"] is None
    assert parsed["semantic_receipt"]["path_normalization"][0]["state"] == (
        "EXACT_DECLARED_MISSING_PATH"
    )
    assert runtime.apply_actions(tmp_path, parsed["actions"]) == []
    assert (tmp_path / "helper.py").read_text(encoding="utf-8") == (
        "def helper(value):\n    return value\n"
    )


def test_create_file_rejects_existing_undeclared_traversal_and_duplicate_targets(
    tmp_path: Path,
) -> None:
    write_source(tmp_path)
    base = valid_ir(tmp_path)
    unit = base[base.index("UNIT U1") : base.index("LOSS NONE")]
    cases = (
        base.replace("PATH helper.py", "PATH sample.py"),
        base.replace("PATH helper.py", "PATH undeclared.py"),
        base.replace("PATH helper.py", "PATH ../escape.py"),
        base.replace("LOSS NONE", unit + "LOSS NONE"),
    )
    for text in cases:
        parsed = runtime.parse(text, task(), tmp_path)
        assert parsed["faults"]
        assert parsed["actions"] == []


def test_create_file_requires_declared_missing_identity(tmp_path: Path) -> None:
    write_source(tmp_path)
    value = task()
    value["candidate_visible_context"]["missing_allowed_effect_paths"] = []
    parsed = runtime.parse(valid_ir(tmp_path), value, tmp_path)
    assert "semantic_symbol_table_invalid" in parsed["faults"]
    assert parsed["actions"] == []


def test_action_applier_rejects_unbound_create_effects(tmp_path: Path) -> None:
    write_source(tmp_path)
    (tmp_path / "existing.py").write_text("x = 1\n", encoding="utf-8")
    assert runtime.apply_actions(tmp_path, [{
        "op": "CREATE_FILE", "path": "existing.py", "replacement": "x = 2"
    }]) == ["action_0_create_target_unsafe_or_exists"]
    assert runtime.apply_actions(tmp_path, [{
        "op": "CREATE_FILE", "path": "missing/helper.py", "replacement": "x = 2"
    }]) == ["action_0_create_parent_missing"]
    assert runtime.apply_actions(tmp_path, [{
        "op": "CREATE_FILE", "path": "../escape.py", "replacement": "x = 2"
    }]) == ["action_0_create_path_unsafe"]


def test_historical_no_create_symbol_identity_is_unchanged(tmp_path: Path) -> None:
    import theseus_p4s_cognitive_compilation as p4s

    write_source(tmp_path)
    value = task()
    value["allowed_effect_paths"] = ["sample.py"]
    value["candidate_visible_context"].pop("missing_allowed_effect_paths")
    assert runtime.semantic_scope_symbol_table(tmp_path, value) == (
        p4s.semantic_scope_symbol_table(tmp_path, value)
    )
