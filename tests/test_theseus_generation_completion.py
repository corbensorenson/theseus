from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_generation_completion as completion  # noqa: E402


def test_direct_plan_and_semantic_envelopes_complete_only_at_terminal_end() -> None:
    direct = "THESEUS_EDIT_V1\nREPLACE a.py 1 1\n<<<\nx = 1\n>>>\nEND"
    plan = "THESEUS_PLAN_V1\nPLAN\nedit x\nTARGET\n" + direct
    semantic = (
        "THESEUS_SEMANTIC_IR_V1\nSOURCE abc\nOBLIGATIONS O1\n"
        "UNIT U1 O1 REPLACE a.py N1 deadbeef\n<<<\nx = 1\n>>>\nLOSS NONE\nEND"
    )

    assert completion.candidate_envelope_complete(direct)
    assert completion.candidate_envelope_complete(plan)
    assert completion.candidate_envelope_complete(semantic)
    assert not completion.candidate_envelope_complete(direct.removesuffix("\nEND"))
    assert not completion.candidate_envelope_complete("commentary\n" + direct)


def test_fenced_envelope_waits_for_closing_fence() -> None:
    direct = "THESEUS_EDIT_V1\nREPLACE a.py 1 1\n<<<\nx = 1\n>>>\nEND"

    assert not completion.candidate_envelope_complete("```text\n" + direct)
    assert completion.candidate_envelope_complete("```text\n" + direct + "\n```")
