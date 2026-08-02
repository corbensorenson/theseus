#!/usr/bin/env python3
"""Machine-authorize the complete-artifact P4 repair campaign."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import theseus_assistant_p2a as p2a
import theseus_p4v2r2r2_autonomous_launch as predecessor
import theseus_p4v2r2r3_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p4v2r2r3_autonomous_launch_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_p4v2r2r3_autonomous_launch.json"
BOUND_CAMPAIGN_COMMIT = "45b06f8a140e45cc8b27f89b99aad16f1979d72f"


@contextmanager
def bind_predecessor() -> Iterator[None]:
    rebound = {
        "POLICY": POLICY,
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "BOUND_CAMPAIGN_COMMIT": BOUND_CAMPAIGN_COMMIT,
        "campaign": campaign,
    }
    original = {name: getattr(predecessor, name) for name in rebound}
    try:
        for name, value in rebound.items():
            setattr(predecessor, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(predecessor, name, value)


def validate_config(config: dict[str, Any]) -> list[str]:
    with bind_predecessor():
        return predecessor.validate_config(config)


def audit_bindings(config: dict[str, Any]) -> dict[str, Any]:
    with bind_predecessor():
        return predecessor.audit_bindings(config)


def preflight(
    config: dict[str, Any],
    *,
    config_path: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with bind_predecessor():
        return predecessor.preflight(
            config,
            config_path=config_path,
            overrides=overrides,
        )


def execute_once(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    with bind_predecessor():
        return predecessor.execute_once(config, config_path=config_path)


def wait_and_execute(
    config: dict[str, Any],
    *,
    config_path: Path,
    poll_seconds: float = 20.0,
) -> dict[str, Any]:
    with bind_predecessor():
        return predecessor.wait_and_execute(
            config,
            config_path=config_path,
            poll_seconds=poll_seconds,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-for-machine", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    config = p2a.read_json(config_path)
    if args.wait_for_machine and not args.execute:
        parser.error("--wait-for-machine requires --execute")
    if args.wait_for_machine:
        report = wait_and_execute(
            config,
            config_path=config_path,
            poll_seconds=args.poll_seconds,
        )
    elif args.execute:
        report = execute_once(config, config_path=config_path)
    else:
        report = preflight(config, config_path=config_path)
    p2a.write_json(p2a.resolve(str(config["report"])), report)
    campaign_audit = p2a.mapping(
        report.get("final_campaign_audit") or report.get("campaign_audit")
    )
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "launch_authorized": report["launch_authorized"],
                "complete_tasks": campaign_audit.get("complete_tasks"),
                "pending_tasks": campaign_audit.get("pending_tasks"),
                "failed_gates": report.get("failed_gates", []),
                "faults": report.get("faults", []),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
