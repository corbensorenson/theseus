from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import autonomy_cycle_runtime as runtime  # noqa: E402


def test_teacher_and_network_authority_default_deny_in_policy() -> None:
    policy = json.loads(
        (ROOT / "configs" / "autonomy_policy.json").read_text(encoding="utf-8")
    )

    assert policy["allow_teacher_by_default"] is False
    assert policy["allow_network_fetch_by_default"] is False


def test_request_local_authority_never_inherits_or_overrides_forbid() -> None:
    assert not runtime.request_local_authority(
        requested=False,
        forbidden=False,
    )
    assert runtime.request_local_authority(
        requested=True,
        forbidden=False,
    )
    assert not runtime.request_local_authority(
        requested=True,
        forbidden=True,
    )


def test_execution_entrypoints_do_not_read_policy_authority_defaults() -> None:
    for relative in (
        "scripts/autonomy_cycle.py",
        "scripts/sparkstream_daemon.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "policy.get(\"allow_teacher_by_default\"" not in source
        assert "policy.get(\"allow_network_fetch_by_default\"" not in source
        assert "request_local_authority(" in source


def test_runtime_uses_shared_frontier_family_classifier() -> None:
    assert (
        runtime.row_frontier_family(
            {
                "benchmark_name": "coding_local_repair",
                "benchmark_type": "local",
                "best_report": "reports/coding_local_repair.json",
            }
        )
        == "coding_local_sandbox"
    )
