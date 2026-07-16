from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gvr_state_machine as gvr


def test_reference_fixture_separates_learned_assisted_and_fallback_credit() -> None:
    report = gvr.run_reference_fixture()
    assert report["trigger_state"] == "GREEN"
    assert report["repaired_receipt"]["learned_generation_credit"] == 0
    assert report["repaired_receipt"]["assisted_repair_credit"] == 1
    assert report["rollback_receipt"]["rollback_exact"] is True
    assert report["summary"]["mutation_case_count"] == report["summary"]["mutation_passed_count"]


def test_transition_history_is_tamper_evident() -> None:
    base = gvr.create_candidate(code_sha256="a" * 64, generator_revision="g", checkpoint_id="c", source_context_digest="s")
    receipt = {"verifier_id": "v", "verifier_revision": "1", "candidate_id": base["candidate_id"], "artifact_sha256": base["current_artifact_sha256"], "independent": True, "tests_digest": "private", "verdict": "exact"}
    exact = gvr.transition(base, "verified_exact", receipt)
    assert gvr.verify_history(exact)
    exact["transitions"][0]["artifact_sha256"] = "tampered"
    assert not gvr.verify_history(exact)
