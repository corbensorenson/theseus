from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_vcm_language_replacements_v2 as v2  # noqa: E402
import theseus_vcm_source_materialization as source  # noqa: E402


def test_real_v2_preflight_is_green_and_scientifically_equal() -> None:
    report = v2.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "SIX_LANGUAGE_REPLACEMENT_V2_TRANSPORT_PREFLIGHT_GREEN"
    assert report["transport_repair"] == {
        "predecessor_graphql_node_batch_size": 40,
        "successor_graphql_node_batch_size": 20,
        "scientific_selection_rules_changed": False,
        "attempted_role_accounting_required": True,
    }
    assert all(value == 0 for value in report["counters"].values())


def test_acquire_changes_only_effective_graphql_batch_size(tmp_path: Path, monkeypatch) -> None:
    config = v2.p2a.read_json(v2.DEFAULT_CONFIG)
    config_path = tmp_path / "config.json"
    v2.p2a.write_json(config_path, config)
    observed: dict[str, object] = {}

    def fake_acquire(effective_config_path, ledger, client, retry_policy):
        effective = v2.p2a.read_json(effective_config_path)
        scientific = v2.p2a.read_json(v2.p2a.resolve(effective["scientific_selection_config"]))
        observed["batch_size"] = scientific["graphql_transport"]["node_batch_size"]
        observed["view"] = v2.replacement_scientific_view(effective)
        return {"trigger_state": "GREEN", "state": "fixture", "counters": source.zero_counters()}

    monkeypatch.setattr(v2.v1r, "acquire", fake_acquire)
    retry_policy = config["transport_retry_policy"]
    ledger = source.SourceLedger(tmp_path / "checkpoint.json", config_path, retry_policy)
    client = v2.InstrumentedSourceClient(ledger, retry_policy)
    report = v2.acquire(config_path, ledger, client, retry_policy)
    predecessor = v2.p2a.read_json(ROOT / config["predecessor_config"])
    assert observed["batch_size"] == 20
    observed_view = dict(observed["view"])
    predecessor_view = v2.replacement_scientific_view(predecessor)
    observed_view.pop("scientific_selection_config")
    predecessor_view.pop("scientific_selection_config")
    assert observed_view == predecessor_view
    assert report["policy"] == v2.POLICY
    assert report["transport_repair"]["scientific_selection_rules_changed"] is False


def test_attempted_role_accounting_sums_to_checkpoint_logical_requests(tmp_path: Path) -> None:
    config = v2.p2a.read_json(v2.DEFAULT_CONFIG)
    ledger = source.SourceLedger(tmp_path / "checkpoint.json", v2.DEFAULT_CONFIG, config["transport_retry_policy"])
    client = v2.InstrumentedSourceClient(ledger, config["transport_retry_policy"])
    for _ in range(7):
        ledger.begin()
    client.title_logical_attempts = 2
    client.source_logical_attempts = 3
    report = v2.attach_attempted_role_accounting({"counters": source.zero_counters()}, ledger, client)
    assert report["attempted_request_role_accounting"] == {
        "metadata": 2,
        "title": 2,
        "source_content": 3,
        "sum_equals_checkpoint_logical_requests": True,
    }
