import copy

import scripts.policy_update_lease as lease


def applied_lease() -> dict:
    contract = lease.load_contract()
    target_id = "planner"
    current = lease.issue_lease(
        lease.reference_request(target_id, contract["targets"][target_id]),
        contract,
    )
    next_state = copy.deepcopy(current["current_state"])
    next_state["revision"] = 2
    return lease.apply_delta(
        current,
        {
            "target_id": target_id,
            "state_path": current["state_path"],
            "before_digest": current["current_digest"],
            "next_state": next_state,
            "cost_observed": {
                "verification": 1,
                "repair": 0,
                "human_cleanup": 0,
                "compute": 1,
                "energy": 1,
            },
        },
        contract,
    )


def resign_last_entry(candidate: dict) -> None:
    row = candidate["journal"][-1]
    row["entry_digest"] = lease.digest({
        key: value for key, value in row.items()
        if key != "entry_digest"
    })


def test_journal_target_remains_bound() -> None:
    wrong_target = applied_lease()
    wrong_target["journal"][-1]["target_id"] = "router"
    resign_last_entry(wrong_target)
    assert not lease.verify_journal(
        wrong_target
    ), "request_contract:journal_target_binding"


def test_journal_state_path_remains_bound() -> None:
    wrong_path = applied_lease()
    wrong_path["journal"][-1]["state_path"] = "runtime/wrong.json"
    resign_last_entry(wrong_path)
    assert not lease.verify_journal(
        wrong_path
    ), "request_contract:journal_state_path_binding"


def test_final_after_digest_matches_current_digest() -> None:
    candidate = applied_lease()
    candidate["current_digest"] = lease.digest({"tampered": True})
    assert not lease.verify_journal(
        candidate
    ), "request_contract:journal_final_digest_binding"


def test_malformed_journal_row_returns_false() -> None:
    candidate = applied_lease()
    candidate["journal"] = [None]
    try:
        observed = lease.verify_journal(candidate)
    except Exception:
        observed = "raised"
    assert observed is False, "request_contract:malformed_journal_fail_closed"
