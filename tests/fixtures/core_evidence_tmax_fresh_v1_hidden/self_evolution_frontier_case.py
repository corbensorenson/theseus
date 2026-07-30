from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import self_evolution_governor as governor  # noqa: E402


def frontier(name: str, residual: object, report: str = "") -> dict:
    return {
        "lifecycle": "frontier",
        "benchmark_name": name,
        "best_report": report,
        "residual": residual,
    }


def test_malformed_boolean_and_nonfinite_residuals_are_ignored() -> None:
    ledger = [
        frontier("bad-text", "not-a-number"),
        frontier("bad-bool", True),
        frontier("bad-nan", "nan"),
        frontier("bad-inf", float("inf")),
        frontier("valid", "2.5"),
    ]
    try:
        selected = governor.active_frontier(ledger)
    except Exception:
        selected = {}
    assert selected.get("benchmark_name") == "valid", (
        "request_contract:invalid_frontier_residuals_ignored"
    )
    assert governor.active_frontier([
        frontier("bad", "nan"),
        frontier("also-bad", False),
    ]) == {}, "request_contract:no_finite_frontier_returns_empty"


def test_equal_residual_ties_use_ascending_stable_identity() -> None:
    ledger = [
        frontier("zeta", 3, "reports/z.json"),
        frontier("alpha", "3.0", "reports/b.json"),
        frontier("alpha", 3.0, "reports/a.json"),
    ]
    selected = governor.active_frontier(ledger)
    assert (
        selected.get("benchmark_name"),
        selected.get("best_report"),
    ) == (
        "alpha",
        "reports/a.json",
    ), "request_contract:frontier_tie_break_is_deterministic"


def test_preferred_family_filtering_precedes_global_fallback() -> None:
    ledger = [
        frontier("coding_small", 2.0),
        frontier("web_agent_large", 9.0),
    ]
    selected = governor.active_frontier(
        ledger,
        preferred_family="coding_local_sandbox",
    )
    assert selected.get("benchmark_name") == "coding_small"
