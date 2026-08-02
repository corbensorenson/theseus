#!/usr/bin/env python3
"""Independent hidden checks for the fresh P4 recovery pool."""

from __future__ import annotations

import sys

from theseus_p4v2r2r2_evaluator_core import evaluate


def main() -> int:
    case = sys.argv[1] if len(sys.argv) == 2 else ""
    try:
        evaluate(case, "hidden")
        print(f"P4V2R2R2_HIDDEN_PASS_{case.upper()}")
        return 0
    except Exception as exc:  # noqa: BLE001 - independent verifier fails closed.
        print(f"P4V2R2R2_HIDDEN_FAIL_{case.upper()}:{type(exc).__name__}:{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
