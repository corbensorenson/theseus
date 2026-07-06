#!/usr/bin/env python3
"""Standalone macOS runtime doctor wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import theseus_runtime  # noqa: E402


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose source, installed app, LaunchAgent, and MLX runtime state on macOS.")
    parser.add_argument("--out", default="reports/macos_runtime_doctor.json")
    args = parser.parse_args()

    report = theseus_runtime.runtime_doctor_report(write_report=True)
    if args.out:
        out_path = resolve(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
