import copy

import scripts.gvr_state_machine as gvr


def verified_candidate() -> dict:
    candidate = gvr.create_candidate(
        code_sha256="a" * 64,
        generator_revision="generator:v1",
        checkpoint_id="checkpoint:v1",
        source_context_digest="context:v1",
    )
    receipt = {
        "verifier_id": "verifier:independent",
        "verifier_revision": "v1",
        "candidate_id": candidate["candidate_id"],
        "artifact_sha256": candidate["current_artifact_sha256"],
        "independent": True,
        "tests_digest": "tests:private",
        "verdict": "exact",
    }
    return gvr.transition(candidate, "verified_exact", receipt)


def test_candidate_identity_remains_bound() -> None:
    candidate = verified_candidate()
    candidate["candidate_id"] = "sha256:" + "f" * 64
    assert not gvr.verify_history(
        candidate
    ), "request_contract:gvr_candidate_identity_binding"


def test_final_artifact_matches_transition_chain() -> None:
    candidate = verified_candidate()
    candidate["current_artifact_sha256"] = "b" * 64
    assert not gvr.verify_history(
        candidate
    ), "request_contract:gvr_final_artifact_binding"


def test_malformed_transition_returns_false() -> None:
    candidate = copy.deepcopy(verified_candidate())
    candidate["transitions"] = [None]
    try:
        observed = gvr.verify_history(candidate)
    except Exception:
        observed = "raised"
    assert observed is False, "request_contract:gvr_malformed_history_fail_closed"
