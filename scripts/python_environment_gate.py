#!/usr/bin/env python3
"""Fail-closed verifier for the canonical Theseus Python environments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "python_environment_contract.json"
DEFAULT_REPORT = ROOT / "reports" / "python_environment_gate.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("policy") != "project_theseus_python_environment_v1":
        raise ValueError("python_environment_contract_policy_invalid")
    return payload


def profile_packages(contract: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    profiles = contract.get("profiles") or {}
    if profile not in profiles:
        raise ValueError(f"unknown_python_environment_profile:{profile}")
    ordered: list[dict[str, Any]] = []
    visited: set[str] = set()

    def add(name: str) -> None:
        if name in visited:
            return
        row = profiles.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"unknown_inherited_python_environment_profile:{name}")
        for parent in row.get("inherits") or []:
            add(str(parent))
        visited.add(name)
        ordered.extend(item for item in row.get("packages") or [] if isinstance(item, dict))

    add(profile)
    return ordered


def requirement_pins(path: Path, visited: set[Path] | None = None) -> dict[str, str]:
    visited = visited or set()
    path = path.resolve()
    if path in visited:
        raise ValueError(f"recursive_requirements_include:{path}")
    visited.add(path)
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            child = line.split(maxsplit=1)[1]
            pins.update(requirement_pins(path.parent / child, visited))
            continue
        if line.startswith(("-c ", "--constraint ")):
            continue
        if "==" not in line or line.count("==") != 1:
            raise ValueError(f"non_exact_requirement:{path}:{line}")
        name, version = line.split("==", 1)
        key = name.strip().lower().replace("_", "-")
        if not key or not version.strip():
            raise ValueError(f"invalid_requirement:{path}:{line}")
        if key in pins and pins[key] != version.strip():
            raise ValueError(f"conflicting_requirement:{key}")
        pins[key] = version.strip()
    visited.remove(path)
    return pins


def declared_constraints(path: Path, visited: set[Path] | None = None) -> set[Path]:
    """Return every constraint file reached through a requirements tree."""
    visited = visited or set()
    path = path.resolve()
    if path in visited:
        raise ValueError(f"recursive_requirements_include:{path}")
    visited.add(path)
    constraints: set[Path] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            child = line.split(maxsplit=1)[1]
            constraints.update(declared_constraints(path.parent / child, visited))
        elif line.startswith(("-c ", "--constraint ")):
            child = line.split(maxsplit=1)[1]
            constraints.add((path.parent / child).resolve())
    visited.remove(path)
    return constraints


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_inventory() -> dict[str, str]:
    inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").lower().replace("_", "-")
        if name:
            inventory[name] = distribution.version
    return inventory


def isolated_import(module: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stderr": result.stderr[-1000:],
    }


def audit(
    contract: dict[str, Any],
    profile: str,
    *,
    python_version: tuple[int, int] | None = None,
    version_lookup: Callable[[str], str | None] = installed_version,
    import_lookup: Callable[[str], dict[str, Any]] = isolated_import,
    inventory_lookup: Callable[[], dict[str, str]] = installed_inventory,
) -> dict[str, Any]:
    expected_python = contract.get("python") or {}
    actual_python = python_version or (sys.version_info.major, sys.version_info.minor)
    python_ready = actual_python == (
        int(expected_python.get("major", -1)),
        int(expected_python.get("minor", -1)),
    )
    profile_row = (contract.get("profiles") or {}).get(profile) or {}
    requirements_path = resolve(str(profile_row.get("requirements") or ""))
    lock_path = resolve(str(profile_row.get("lock") or ""))
    expected = {
        str(row.get("distribution") or "").lower().replace("_", "-"): str(row.get("version") or "")
        for row in profile_packages(contract, profile)
    }
    requirement_fault = ""
    try:
        pinned = requirement_pins(requirements_path)
        constraints = declared_constraints(requirements_path)
    except (OSError, ValueError) as exc:
        pinned = {}
        constraints = set()
        requirement_fault = str(exc)
    requirements_match = not requirement_fault and pinned == expected
    lock_fault = ""
    try:
        lock_pins = requirement_pins(lock_path)
        lock_digest = sha256_file(lock_path)
    except (OSError, ValueError) as exc:
        lock_pins = {}
        lock_digest = ""
        lock_fault = str(exc)
    expected_lock_digest = str(profile_row.get("lock_sha256") or "")
    lock_declared = lock_path.resolve() in constraints if str(profile_row.get("lock") or "") else False
    lock_digest_matches = bool(expected_lock_digest) and lock_digest == expected_lock_digest
    lock_versions = []
    for distribution, expected_version in sorted(lock_pins.items()):
        try:
            actual_version = version_lookup(distribution)
        except (KeyError, LookupError):
            actual_version = None
        lock_versions.append(
            {
                "distribution": distribution,
                "expected_version": expected_version,
                "actual_version": actual_version,
                "ready": actual_version == expected_version,
            }
        )
    lock_mismatches = [row["distribution"] for row in lock_versions if not row["ready"]]
    installed = {name.lower().replace("_", "-"): version for name, version in inventory_lookup().items()}
    tooling_allowlist = {"pip", "setuptools", "wheel"}
    unexpected_installed = sorted(set(installed) - set(lock_pins) - tooling_allowlist)
    lock_ready = (
        not lock_fault
        and bool(lock_pins)
        and lock_declared
        and lock_digest_matches
        and not lock_mismatches
        and not unexpected_installed
    )
    guarded_accelerator = os.environ.get("THESEUS_GUARDED_ACCELERATOR_CHILD") == "1"
    packages = []
    for row in profile_packages(contract, profile):
        distribution = str(row.get("distribution") or "")
        expected_version = str(row.get("version") or "")
        actual_version = version_lookup(distribution)
        accelerator = row.get("accelerator") is True
        metadata_ready = actual_version == expected_version
        if accelerator and not guarded_accelerator:
            import_receipt = {"passed": None, "skipped": True, "reason": "watchdog_authority_required"}
        elif metadata_ready:
            import_receipt = import_lookup(str(row.get("module") or ""))
        else:
            import_receipt = {"passed": False, "skipped": True, "reason": "version_missing_or_mismatched"}
        packages.append(
            {
                "distribution": distribution,
                "module": row.get("module"),
                "expected_version": expected_version,
                "actual_version": actual_version,
                "accelerator": accelerator,
                "metadata_ready": metadata_ready,
                "import": import_receipt,
                "ready": metadata_ready and (accelerator and not guarded_accelerator or import_receipt.get("passed") is True),
            }
        )
    missing_or_mismatched = [row["distribution"] for row in packages if not row["metadata_ready"]]
    import_failures = [row["distribution"] for row in packages if row["metadata_ready"] and row["ready"] is False]
    ready = python_ready and requirements_match and lock_ready and not missing_or_mismatched and not import_failures
    install_command = f"{sys.executable} -m pip install -r {requirements_path}"
    return {
        "policy": "project_theseus_python_environment_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_state": "GREEN" if ready else "RED",
        "profile": profile,
        "python": {
            "expected": f"{expected_python.get('major')}.{expected_python.get('minor')}.x",
            "actual": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "ready": python_ready,
        },
        "requirements": {
            "path": str(requirements_path.relative_to(ROOT)) if requirements_path.is_relative_to(ROOT) else str(requirements_path),
            "exact_pins_only": True,
            "matches_contract": requirements_match,
            "fault": requirement_fault,
        },
        "lock": {
            "path": str(lock_path.relative_to(ROOT)) if lock_path.is_relative_to(ROOT) else str(lock_path),
            "declared_by_requirements": lock_declared,
            "expected_sha256": expected_lock_digest,
            "actual_sha256": lock_digest,
            "digest_matches": lock_digest_matches,
            "exact_pins_only": not lock_fault,
            "package_count": len(lock_pins),
            "installed_mismatches": lock_mismatches,
            "unexpected_installed": unexpected_installed,
            "ready": lock_ready,
            "fault": lock_fault,
        },
        "packages": packages,
        "summary": {
            "ready": ready,
            "package_count": len(packages),
            "locked_package_count": len(lock_pins),
            "lock_installed_mismatches": lock_mismatches,
            "unexpected_installed": unexpected_installed,
            "missing_or_mismatched": missing_or_mismatched,
            "import_failures": import_failures,
            "ambient_accelerator_import_count": 0,
            "install_command": install_command,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT.relative_to(ROOT)))
    parser.add_argument("--profile", choices=("cpu", "data", "mlx"), default="cpu")
    parser.add_argument("--out", default="")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    report = audit(load_contract(resolve(args.contract)), args.profile)
    if args.out:
        path = resolve(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    view = {
        "trigger_state": report["trigger_state"],
        "profile": report["profile"],
        "python": report["python"],
        "requirements": report["requirements"],
        "lock": report["lock"],
        "summary": report["summary"],
    }
    print(json.dumps(view if args.gate else report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
