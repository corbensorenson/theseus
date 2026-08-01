from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4r_semantic_ir_v2_canary as v2  # noqa: E402
import theseus_p4r_semantic_ir_v2r1_canary as v2r1  # noqa: E402


def test_repair_prompt_fixes_observed_header_and_operation_failures(
    tmp_path: Path,
) -> None:
    case = {
        "case_id": "return_membership",
        "natural_request": "Change membership to not in.",
        "source": "def is_member(value):\n    return value in (1, 2)\n",
        "target_node_type": "Return",
        "obligations": [
            {"id": "O1", "kind": "require", "text": "use not in"},
            {"id": "O2", "kind": "preserve", "text": "preserve tuple"},
            {"id": "O3", "kind": "non_goal", "text": "no signature change"},
        ],
        "obligation_dependencies": [{"before": "O2", "after": "O1"}],
    }
    source = tmp_path / "sample.py"
    source.write_text(case["source"], encoding="utf-8")
    task = v2.build_task(case)
    symbols = p4.semantic_symbol_table(tmp_path, task)
    target = v2.select_target(symbols, "Return")
    prompt = v2r1.render_prompt(case, task, symbols, target, source)

    assert prompt.startswith("FIRST LINE MUST BE EXACTLY: THESEUS_SEMANTIC_IR_V2")
    assert "REQUIRED OP: REPLACE" in prompt
    assert "TARGET_NODE_SOURCE\nreturn value in (1, 2)" in prompt
    assert "without surrounding indentation or enclosing source" in prompt


def test_repair_canary_closes_transport_but_not_semantic_unit_adequacy() -> None:
    report = json.loads(
        (
            ROOT / "reports" / "theseus_p4r_semantic_ir_v2r1_mechanics_canary.json"
        ).read_text(encoding="utf-8")
    )

    assert report["trigger_state"] == "YELLOW"
    assert report["state"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert report["parse_and_lower"] == "3/3"
    assert report["verified"] == "1/3"
    assert report["model_loads"] == 1
    assert report["model_calls"] == 3
    assert report["safety_ceiling_hits"] == 0
