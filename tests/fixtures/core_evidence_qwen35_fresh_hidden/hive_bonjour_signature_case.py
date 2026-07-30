from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import hive_node_peer_registry as registry  # noqa: E402


def signed_txt(monkeypatch) -> tuple[dict, dict]:
    monkeypatch.setenv("THESEUS_HIVE_DISCOVERY_SECRET", "fixture-secret")
    policy = {
        "security": {
            "discovery_secret_env": "THESEUS_HIVE_DISCOVERY_SECRET",
        }
    }
    txt = {
        "txtvers": "1",
        "hive_id": "fixture-hive",
        "node_id": "worker-a",
        "node_name": "Worker A",
        "api_url": "http://127.0.0.1:8787",
        "relay_url": "http://127.0.0.1:8788",
        "roles": "worker,evaluator",
        "caps": "mlx,metal",
    }
    txt["sig"] = registry.bonjour_signature(policy, txt)
    txt["sig_alg"] = "hmac-sha256"
    assert registry.verify_bonjour_txt(policy, txt)
    return policy, txt


def test_bonjour_signature_binds_every_advertised_peer_field(monkeypatch) -> None:
    policy, original = signed_txt(monkeypatch)
    mutations = {
        "txtvers": "2",
        "node_name": "Attacker",
        "relay_url": "http://attacker.invalid:9999",
        "roles": "coordinator",
        "caps": "unbounded-execution",
    }
    for field, replacement in mutations.items():
        tampered = dict(original)
        tampered[field] = replacement
        assert not registry.verify_bonjour_txt(
            policy,
            tampered,
        ), f"request_contract:bonjour_signature_binds_advertised_fields:{field}"


def test_bonjour_signature_algorithm_is_enforced(monkeypatch) -> None:
    policy, txt = signed_txt(monkeypatch)
    txt["sig_alg"] = "none"
    assert not registry.verify_bonjour_txt(
        policy,
        txt,
    ), "request_contract:bonjour_signature_algorithm_enforced"
