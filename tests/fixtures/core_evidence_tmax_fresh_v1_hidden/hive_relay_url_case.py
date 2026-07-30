from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import hive_fleet_readiness as readiness  # noqa: E402


POLICY = {
    "policy": "project_theseus_hive_policy_v0",
    "federation": {"tiers": {"private": {"remote_tasks": True}}},
    "task_kinds": {},
}
NO_REMOTE = {
    "summary": {"trusted_node_count": 2},
    "nodes": [],
}


def blockers(relay_url: str, registry: dict = NO_REMOTE) -> list[str]:
    return readiness.blockers_for(
        "darwin",
        POLICY,
        secret_present=True,
        relay_url=relay_url,
        registry=registry,
    )


def test_invalid_or_unsafe_relay_urls_do_not_satisfy_connectivity() -> None:
    invalid = [
        "not-a-url",
        "http://relay.example/private",
        "https:///missing-host",
        "https://user:secret@relay.example/private",
        "https://relay.example/private#fragment",
    ]
    assert all(
        "same_lan_or_private_tunnel_required_without_relay"
        in blockers(value)
        for value in invalid
    ), "request_contract:unsafe_relay_urls_retain_connectivity_blocker"


def test_valid_https_relay_satisfies_connectivity() -> None:
    assert (
        "same_lan_or_private_tunnel_required_without_relay"
        not in blockers("https://relay.example/private")
    ), "request_contract:valid_https_relay_accepted"


def test_trusted_remote_preserves_existing_relay_bypass() -> None:
    registry = {
        "summary": {"trusted_node_count": 2},
        "nodes": [
            {
                "is_local": False,
                "trust": {"trusted": True},
            }
        ],
    }
    assert (
        "same_lan_or_private_tunnel_required_without_relay"
        not in blockers("", registry)
    ), "request_contract:trusted_remote_preserves_relay_bypass"
