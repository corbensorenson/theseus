from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p2b_disposition as disposition  # noqa: E402


def test_p2b_is_inconclusive_because_prompt_and_parser_newline_transports_differ() -> None:
    report = disposition.build(
        ROOT / "reports" / "theseus_assistant_p2b_requests_7502_run.json",
        ROOT / "reports" / "theseus_assistant_p2b_requests_7502_evaluation.json",
        ROOT / "configs" / "theseus_assistant_p2b_instrument.json",
        ROOT / "configs" / "theseus_p2b_task_requests_7502.json",
    )

    assert report["trigger_state"] == "GREEN"
    assert report["scientific_status"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert report["terminal_disposition"] == "P2B_TERMINAL_INCONCLUSIVE_LITERAL_GRAMMAR_TRANSPORT"
    assert report["grammar_transport"]["contains_literal_backslash_n"] is True
    assert report["grammar_transport"]["contains_actual_newline"] is False
    assert report["recomputed_checks"]["escaped_grammar_transport_reproduced_in_every_output"] is True
    assert report["recomputed_checks"]["three_of_four_outputs_contain_replace_token"] is True
    assert report["recomputed_checks"]["every_output_uses_authorized_repo_relative_path"] is True
    assert report["recomputed_checks"]["correctness_evaluated_candidates"] == 0
    assert report["next_stage"]["id"] == "P2C_NEW_INSTRUMENT_AND_SOURCE_DISJOINT_TASK"
    assert report["consumption"]["eligible_for_exact_rerun"] is False
