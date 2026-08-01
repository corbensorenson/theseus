from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4r_intervention_locality_canary as canary  # noqa: E402


def test_controlled_fault_injection_changes_only_selected_unit() -> None:
    original = (
        "THESEUS_SEMANTIC_IR_V2\nSOURCE " + "a" * 64
        + "\nALL_OBLIGATIONS O1,O2,O3,O4\n"
        "UNIT U1\nOBLIGATIONS O1,O2\nOP REPLACE\nPATH sample.py\nNODE N-ONE\n"
        "NODE_SHA " + "b" * 64 + "\n<<<\nPRIMARY = \"new\"\n>>>\nEND_UNIT\n"
        "UNIT U2\nOBLIGATIONS O3,O4\nOP REPLACE\nPATH sample.py\nNODE N-TWO\n"
        "NODE_SHA " + "c" * 64 + "\n<<<\nSECONDARY = \"steady\"\n>>>\nEND_UNIT\n"
        "LOSS NONE\nEND"
    )
    mutated = canary.inject_unit_fault(original, "U1", 'PRIMARY = "__FAULT__"')

    assert 'PRIMARY = "__FAULT__"' in mutated
    assert 'PRIMARY = "new"' not in mutated
    assert 'SECONDARY = "steady"' in mutated


def test_static_intervention_ladder_rejects_all_bound_corruptions() -> None:
    case = {
        "source": 'PRIMARY = "old"\nSECONDARY = "hold"\n',
        "expected_source": 'PRIMARY = "new"\nSECONDARY = "steady"\n',
        "feedback_marker": "FAIL_O1_PRIMARY",
        "feedback_obligation_ids": ["O1"],
        "obligations": [
            {"id": "O1", "kind": "require", "text": "set PRIMARY to new"},
            {"id": "O2", "kind": "preserve", "text": "preserve PRIMARY name"},
            {"id": "O3", "kind": "require", "text": "set SECONDARY to steady"},
            {"id": "O4", "kind": "preserve", "text": "preserve SECONDARY name"},
        ],
        "obligation_dependencies": [
            {"before": "O2", "after": "O1"},
            {"before": "O4", "after": "O3"},
        ],
        "units": [
            {
                "unit_id": "U1", "obligation_ids": ["O1", "O2"],
                "target_line": 1, "expected_replacement": 'PRIMARY = "new"',
            },
            {
                "unit_id": "U2", "obligation_ids": ["O3", "O4"],
                "target_line": 2, "expected_replacement": 'SECONDARY = "steady"',
            },
        ],
    }

    report = canary.run_static_intervention_ladder(case)

    assert report["trigger_state"] == "GREEN"
    assert report["valid_fixture_action_count"] == 2
    assert all(report["required_corruption_rejections"].values())


def test_no_project_selected_quality_token_cap_constant() -> None:
    assert canary.MODEL_CONTEXT_TOKENS == 262144
    assert "non-claim" in canary.__doc__.lower()
