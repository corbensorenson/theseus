from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.procedural_memory_assets import (
    append_lifecycle_ledger,
    build_assets,
    build_trie,
    lifecycle_receipt,
    lookup,
)
from scripts import theseus_assistant_runtime


NOW = "2026-07-10T12:00:00Z"
POLICY = {
    "monitor_window_events": 8,
    "minimum_recent_useful_rate": 0.9,
    "retire_after_consecutive_failures": 2,
    "max_stale_days": 30,
}


def event(*, outcome: str = "completed", created_utc: str = NOW) -> dict:
    return {
        "surface": "local_assistant",
        "assistant_lane": "planning_assistant",
        "intent": "planning",
        "outcome": outcome,
        "created_utc": created_utc,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def candidate() -> dict:
    return {
        "id": "procedural.assistant_trace.test",
        "source_traces": [
            {
                "surface": "local_assistant",
                "assistant_lane": "planning_assistant",
                "intent_bucket": "planning",
            }
        ],
        "preconditions": ["metadata only"],
        "postconditions": ["completed"],
        "monitoring": ["outcome"],
        "retirement_criteria": ["two failures"],
    }


def route() -> dict:
    return {
        "id": "canary.test",
        "candidate_id": "procedural.assistant_trace.test",
        "replay_fixture_id": "replay.test",
        "canary_route_eligible": True,
    }


def test_active_asset_is_retrievable_and_noncredit() -> None:
    report = build_assets(
        candidates=[candidate()],
        replay_results=[{"id": "replay.test", "candidate_id": candidate()["id"], "passed": True}],
        canary_routes=[route()],
        events=[event(), event(), event()],
        lifecycle_policy=POLICY,
        created_utc=NOW,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["summary"]["active_asset_count"] == 1
    asset = report["assets"][0]
    assert asset["candidate_generation_credit"] == "none"
    assert asset["learned_generation_claim_allowed"] is False
    assert lookup(report["lookahead_trie"], asset["lookup_binding"])["asset_id"] == asset["id"]


def test_ambiguous_and_unknown_bindings_abstain() -> None:
    report = build_assets(
        candidates=[candidate()],
        replay_results=[{"id": "replay.test", "candidate_id": candidate()["id"], "passed": True}],
        canary_routes=[route()],
        events=[event()],
        lifecycle_policy=POLICY,
        created_utc=NOW,
    )
    asset = report["assets"][0]
    duplicate = dict(asset)
    duplicate["id"] = "procedural.asset.collision"
    ambiguous = lookup(build_trie([asset, duplicate]), asset["lookup_binding"])
    unknown = lookup(report["lookahead_trie"], {"surface": "x", "assistant_lane": "y", "intent": "z"})
    assert ambiguous["state"] == "AMBIGUOUS"
    assert ambiguous["asset_id"] == ""
    assert unknown["state"] == "NO_ADMISSIBLE"


def test_stale_and_drifted_procedures_retire() -> None:
    kwargs = {
        "candidate": candidate(),
        "route": route(),
        "replay": {"passed": True},
        "binding": {"surface": "local_assistant", "assistant_lane": "planning_assistant", "intent": "planning"},
        "policy": POLICY,
        "created_utc": NOW,
    }
    stale = lifecycle_receipt(events=[event(created_utc="2025-01-01T00:00:00Z")], **kwargs)
    drift = lifecycle_receipt(events=[event(outcome="missed"), event(outcome="ignored")], **kwargs)
    assert stale["lifecycle_state"] == "retired_stale"
    assert stale["rollback_triggered"] is True
    assert drift["lifecycle_state"] == "retired_drift"
    assert drift["rollback_triggered"] is True


def test_no_cheat_fault_blocks_asset() -> None:
    bad = event()
    bad["fallback_return_count"] = 1
    receipt = lifecycle_receipt(
        candidate=candidate(),
        route=route(),
        replay={"passed": True},
        binding={"surface": "local_assistant", "assistant_lane": "planning_assistant", "intent": "planning"},
        events=[bad],
        policy=POLICY,
        created_utc=NOW,
    )
    assert receipt["lifecycle_state"] == "blocked"
    assert receipt["rollback_triggered"] is True


def test_lifecycle_ledger_is_append_only_and_idempotent(tmp_path: Path) -> None:
    receipt = lifecycle_receipt(
        candidate=candidate(),
        route=route(),
        replay={"passed": True},
        binding={"surface": "local_assistant", "assistant_lane": "planning_assistant", "intent": "planning"},
        events=[event()],
        policy=POLICY,
        created_utc=NOW,
    )
    path = tmp_path / "lifecycle.jsonl"
    assert append_lifecycle_ledger(path, [receipt]) == 1
    assert append_lifecycle_ledger(path, [receipt]) == 0
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [receipt]


def test_runtime_selects_each_bound_route_without_first_route_fallthrough(tmp_path: Path, monkeypatch) -> None:
    routes = []
    route_configs = {
        "planning": {"assistant_lane": "planning_assistant", "vcm_task_family": "autonomy_governance"},
        "chat": {"assistant_lane": "chat_checkpoint", "vcm_task_family": "operator_chat"},
        "tool": {"assistant_lane": "tool_assistant", "vcm_task_family": "operator_chat"},
    }
    for intent, route_config in route_configs.items():
        routes.append(
            {
                "id": f"default.{intent}",
                "default_route_adopted": True,
                "learned_generation_claim_allowed": False,
                "continued_regression_guard": {"armed": True},
                "route_binding_contract": {
                    "assistant_surfaces": ["local_assistant"],
                    "assistant_intents": [intent],
                    "assistant_lanes": [route_config["assistant_lane"]],
                    "vcm_task_families": [route_config["vcm_task_family"]],
                    "runtime_consumers": ["theseus_assistant_runtime"],
                    "selection_keys": ["intent", "assistant_lane", "vcm_task_family"],
                },
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            }
        )
    report = {
        "trigger_state": "GREEN",
        "summary": {
            "hard_gap_count": 0,
            "default_route_adopted_count": 3,
            "default_route_guarded_count": 3,
            "learned_generation_claim_count": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
        "default_routes": routes,
    }
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "adoption.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(theseus_assistant_runtime, "ROOT", tmp_path)
    config = {
        "procedural_memory_default_route": {
            "enabled": True,
            "report": "reports/adoption.json",
            "eligible_intents": list(route_configs),
            "required_for_intents": ["planning"],
            "selection_mode": "route_binding_contract",
            "required_runtime_consumer": "theseus_assistant_runtime",
        }
    }
    for intent, route_config in route_configs.items():
        packet = theseus_assistant_runtime.procedural_default_route_packet(intent, route_config, config)
        assert packet["ready"] is True
        assert packet["selected_route"]["id"] == f"default.{intent}"
        assert packet["selection"]["matched"] is True
    wrong_surface = theseus_assistant_runtime.procedural_default_route_packet(
        "planning",
        route_configs["planning"],
        config,
        surface="hive_operator",
    )
    assert wrong_surface["ready"] is False
    assert wrong_surface["selected_route"] == {}
