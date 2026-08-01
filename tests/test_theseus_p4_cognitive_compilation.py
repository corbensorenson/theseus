from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402


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
        "visible_feedback_map": [
            {"marker": "three missing", "obligation_ids": ["O1"]}
        ],
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
        f"{p4.IR_HEADER}\nSOURCE {symbols['source_digest']}\n"
        "OBLIGATIONS O1,O2,O3\n"
        f"UNIT U1 O1,O2,O3 REPLACE sample.py {target['id']} {target['sha256']}\n"
        "<<<\n(1, 2, 3)\n>>>\nLOSS NONE\nEND"
    )


def test_semantic_ir_lowers_stable_ast_identity_to_typed_edit(tmp_path: Path) -> None:
    write_source(tmp_path)
    first_symbols = p4.semantic_symbol_table(tmp_path, task())
    second_symbols = p4.semantic_symbol_table(tmp_path, task())
    assert first_symbols == second_symbols

    parsed = p4.parse_semantic_ir(valid_ir(tmp_path), task(), tmp_path)
    assert parsed["faults"] == []
    assert len(parsed["actions"]) == 1
    assert parsed["semantic_receipt"]["obligation_ids"] == ["O1", "O2", "O3"]
    assert parsed["semantic_receipt"]["loss_obligation_ids"] == []
    assert p2a.apply_actions(tmp_path, parsed["actions"]) == []
    assert "VALUES = (1, 2, 3)" in (tmp_path / "sample.py").read_text(encoding="utf-8")


def test_semantic_ir_rejects_identity_loss_coverage_and_unparsed_text(tmp_path: Path) -> None:
    write_source(tmp_path)
    ir = valid_ir(tmp_path)
    source_digest = p4.semantic_symbol_table(tmp_path, task())["source_digest"]
    target_hash = p4.IR_UNIT_RE.search(ir).group(6)  # type: ignore[union-attr]
    mutations = (
        ir.replace(source_digest, "0" * 64, 1),
        ir.replace("OBLIGATIONS O1,O2,O3", "OBLIGATIONS O1,O2", 1),
        ir.replace(target_hash, "0" * 64, 1),
        ir.replace("LOSS NONE", "LOSS O1", 1),
        ir.replace("LOSS NONE", "EXTRA\nLOSS NONE", 1),
    )
    for mutation in mutations:
        assert p4.parse_semantic_ir(mutation, task(), tmp_path)["faults"]


def test_semantic_ir_enforces_dependency_closure_and_nonoverlap(tmp_path: Path) -> None:
    write_source(tmp_path)
    ir = valid_ir(tmp_path)
    missing_dependency = ir.replace("UNIT U1 O1,O2,O3", "UNIT U1 O1,O3")
    assert "semantic_unit_dependency_not_closed" in p4.parse_semantic_ir(
        missing_dependency, task(), tmp_path
    )["faults"]

    duplicate = ir.replace("\nLOSS NONE", "\n" + ir.split("\n", 3)[3].split("\nLOSS NONE")[0] + "\nLOSS NONE")
    faults = p4.parse_semantic_ir(duplicate, task(), tmp_path)["faults"]
    assert "semantic_units_overlap" in faults
    assert "semantic_unit_identity_duplicate" in faults


def test_dependency_local_repair_rejects_unrelated_change_and_identity_churn() -> None:
    first = [{
        "id": "U1", "obligation_ids": ["O1", "O2"], "operation": "REPLACE",
        "path": "sample.py", "node_id": "N-ONE", "replacement_sha256": "a",
    }, {
        "id": "U2", "obligation_ids": ["O3"], "operation": "REPLACE",
        "path": "sample.py", "node_id": "N-TWO", "replacement_sha256": "b",
    }]
    unrelated = [dict(first[0], replacement_sha256="c"), dict(first[1], replacement_sha256="d")]
    assert p4.repair_locality_faults(first, unrelated, {"O1", "O2"}) == [
        "semantic_repair_not_dependency_local"
    ]
    churn = [dict(first[0], id="U9"), first[1]]
    assert "semantic_repair_unit_identity_churn" in p4.repair_locality_faults(
        first, churn, {"O1", "O2", "O3"}
    )


def test_three_arm_order_uses_latin_rotation() -> None:
    assert p4.arm_order(1) == (p4.DIRECT, p4.PLAN, p4.SEMANTIC)
    assert p4.arm_order(2) == (p4.PLAN, p4.SEMANTIC, p4.DIRECT)
    assert p4.arm_order(3) == (p4.SEMANTIC, p4.DIRECT, p4.PLAN)
    assert p4.arm_order(4) == p4.arm_order(1)


def test_deterministic_request_compiler_exact_literal_replace(tmp_path: Path) -> None:
    write_source(tmp_path)
    value = task() | {
        "natural_request": "Replace exact literal `return value in VALUES` with `return value not in VALUES`."
    }
    actions, faults, rule = p4.deterministic_request_compile(value, tmp_path)
    assert faults == []
    assert rule == "exact_literal_replace"
    assert actions[0]["replacement"] == "    return value not in VALUES"


def test_deterministic_request_compiler_collection_edit(tmp_path: Path) -> None:
    write_source(tmp_path)
    value = task() | {"natural_request": "Add `3` to `VALUES`."}
    actions, faults, rule = p4.deterministic_request_compile(value, tmp_path)
    assert faults == []
    assert rule == "collection_literal_edit"
    assert actions[0]["replacement"] == "VALUES = (1, 2, 3)"


def test_deterministic_request_compiler_abstains_outside_fixed_grammar(tmp_path: Path) -> None:
    write_source(tmp_path)
    value = task() | {"natural_request": "Refactor choose for clarity."}
    actions, faults, rule = p4.deterministic_request_compile(value, tmp_path)
    assert actions == []
    assert faults == ["deterministic_request_pattern_unsupported"]
    assert rule == "abstain"


def test_mechanics_audit_is_green() -> None:
    report = p4.mechanics_audit()
    assert report["trigger_state"] == "GREEN"
    assert report["ready"] is True
    assert all(report["corruption_rejections"].values())


def test_frozen_instrument_audit_is_green() -> None:
    report = p4.audit_instrument(
        ROOT / "configs" / "theseus_p4_cognitive_compilation_instrument.json"
    )
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
