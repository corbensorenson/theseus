from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import hive_node_registry as registry  # noqa: E402


def test_naive_peer_timestamp_fails_closed_without_raising() -> None:
    try:
        observed = registry.timestamp_age_seconds("2026-01-01T00:00:00")
    except Exception:
        observed = "raised"
    assert observed is None, (
        "request_contract:naive_peer_timestamp_fails_closed"
    )


def test_materially_future_peer_timestamp_is_rejected() -> None:
    future = datetime.now(timezone.utc) + timedelta(minutes=1)
    assert registry.timestamp_age_seconds(
        future.isoformat()
    ) is None, "request_contract:future_peer_timestamp_rejected"


def test_timezone_aware_peer_timestamp_preserves_elapsed_time() -> None:
    past = datetime.now(timezone.utc) - timedelta(seconds=30)
    offset = past.astimezone(timezone(timedelta(hours=2)))
    age = registry.timestamp_age_seconds(offset.isoformat())
    assert isinstance(age, float) and 25.0 <= age <= 40.0
