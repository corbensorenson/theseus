#!/usr/bin/env python3
"""Fail closed when repository and Cargo license declarations diverge."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPDX = "Apache-2.0"


def audit(root: Path = ROOT) -> dict[str, Any]:
    gaps = []
    license_path = root / "LICENSE"
    license_text = (
        license_path.read_text(encoding="utf-8", errors="replace")
        if license_path.is_file()
        else ""
    )
    if not license_text.startswith("Apache License\nVersion 2.0"):
        gaps.append("root_license_is_not_apache_2_0")
    cargo_rows = []
    workspace_manifests = [
        root / "Cargo.toml",
        *sorted((root / "crates").glob("*/Cargo.toml")),
    ]
    for path in workspace_manifests:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        declarations = re.findall(
            r"(?m)^\s*license\s*=\s*\"([^\"]+)\"\s*$",
            text,
        )
        for declaration in declarations:
            cargo_rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "license": declaration,
                }
            )
            if declaration != EXPECTED_SPDX:
                gaps.append(
                    f"cargo_license_mismatch:{path.relative_to(root)}:{declaration}"
                )
    root_cargo = root / "Cargo.toml"
    if not any(row["path"] == "Cargo.toml" for row in cargo_rows):
        gaps.append("root_cargo_license_missing")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if "licensed under Apache-2.0" not in readme:
        gaps.append("readme_apache_license_statement_missing")
    return {
        "policy": "project_theseus_repository_license_alignment_v1",
        "trigger_state": "GREEN" if not gaps else "RED",
        "expected_spdx": EXPECTED_SPDX,
        "cargo_declarations": cargo_rows,
        "root_license": "LICENSE",
        "hard_gaps": gaps,
        "non_claim": (
            "This checks repository metadata alignment; it is not legal advice "
            "or a third-party dependency license review."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    report = audit()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else (2 if args.gate else 0)


if __name__ == "__main__":
    raise SystemExit(main())
