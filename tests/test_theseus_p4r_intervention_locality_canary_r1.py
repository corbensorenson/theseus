from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4r_intervention_locality_canary_r1 as canary  # noqa: E402


def test_bound_skeleton_is_last_and_contains_two_complete_units() -> None:
    case = {
        "natural_request": "change both values",
        "source": 'PRIMARY = "old"\nSECONDARY = "hold"\n',
        "obligations": [
            {"id": "O1", "kind": "require", "text": "first"},
            {"id": "O2", "kind": "preserve", "text": "first name"},
            {"id": "O3", "kind": "require", "text": "second"},
            {"id": "O4", "kind": "preserve", "text": "second name"},
        ],
        "obligation_dependencies": [
            {"before": "O2", "after": "O1"},
            {"before": "O4", "after": "O3"},
        ],
    }
    targets = {
        "U1": {"id": "N-ONE", "sha256": "a" * 64},
        "U2": {"id": "N-TWO", "sha256": "b" * 64},
    }
    task = {
        "obligations": case["obligations"],
        "obligation_dependencies": case["obligation_dependencies"],
    }

    prompt = canary.render_first_prompt(
        case, task, {"source_digest": "c" * 64}, targets, ROOT / "README.md"
    )

    assert prompt.endswith("LOSS NONE\nEND")
    assert prompt.count("UNIT U1\n") == 1
    assert prompt.count("UNIT U2\n") == 1
    assert prompt.count("END_UNIT\n") == 2
    assert "WRITE_COMPLETE_U1_ASSIGNMENT_HERE" in prompt
    assert "WRITE_COMPLETE_U2_ASSIGNMENT_HERE" in prompt


def test_r1_is_scoped_to_non_claim_transport_repair() -> None:
    assert canary.POLICY.endswith("canary_r1_v1")
    assert "bound-skeleton repair" in canary.__doc__.lower()
    assert canary.MODEL_CONTEXT_TOKENS == 262144
