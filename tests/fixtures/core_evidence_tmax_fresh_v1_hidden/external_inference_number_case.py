from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import external_inference_audit as audit  # noqa: E402


def test_boolean_and_nonfinite_audit_numbers_are_rejected() -> None:
    rejected = [
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        "nan",
        "+Infinity",
        "-inf",
    ]
    observed = [audit.number(value) for value in rejected]
    assert observed == [0.0] * len(rejected) and all(
        math.isfinite(value) for value in observed
    ), "request_contract:boolean_and_nonfinite_audit_numbers_rejected"


def test_finite_audit_numbers_and_numeric_strings_are_preserved() -> None:
    cases = [
        (7, 7.0),
        (-2.5, -2.5),
        ("3.25", 3.25),
        ("-4e2", -400.0),
    ]
    assert [
        audit.number(value) for value, _ in cases
    ] == [
        expected for _, expected in cases
    ], "request_contract:finite_audit_numbers_preserved"
