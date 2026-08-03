from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_toolchain_compatibility_audit as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_toolchain_compatibility_audit.json")


def test_all_locked_tasks_receive_static_compatibility_classification() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["observations"]["locked_task_count"] == 48
    assert len({row["index"] for row in REPORT["rows"]}) == 48
    assert REPORT["dependency_prefetch_executions"] == 0
    assert REPORT["repository_executions"] == 0


def test_task_three_requires_the_qualified_successor_profile() -> None:
    row = next(row for row in REPORT["rows"] if row["index"] == 3)
    assert row["state"] == "COMPATIBLE_DECLARED_REQUIREMENTS"
    assert "node22_20_npm10_9_3" in row["compatible_profile_ids"]
    assert "node22_15_npm10_9_2" not in row["compatible_profile_ids"]


def test_semver_parser_covers_manifest_range_forms() -> None:
    assert owner.semver_satisfies("22.20.0", ">=22.20.0 <25.0.0") is True
    assert owner.semver_satisfies("22.15.0", ">=22.20.0 <25.0.0") is False
    assert owner.semver_satisfies("10.13.1", "^10.0.0") is True
    assert owner.semver_satisfies("1.22.22", "1.x") is True
