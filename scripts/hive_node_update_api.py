"""Update and version helpers for the Project Theseus Hive node."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import update_manager
from hive_node_common import append_jsonl, event, read_json


ROOT = Path(__file__).resolve().parents[1]


def startup_update_check() -> None:
    try:
        policy = update_policy()
        args = argparse.Namespace(
            catalog_url="",
            update_id="",
            apply=False,
            if_enabled_on_start=True,
            respect_interval=True,
        )
        update_manager.check_for_updates(policy, args)
    except Exception as exc:  # Keep Hive startup alive even when an update source is down.
        append_jsonl(ROOT / "reports" / "update_events.jsonl", event("startup_update_check_failed", {"error": str(exc)}))


def update_policy() -> dict[str, Any]:
    return update_manager.read_json(update_manager.POLICY_PATH, {})


def hive_version_catalog() -> dict[str, Any]:
    catalog = read_json(ROOT / "reports" / "hive_update_catalog.json", {})
    return catalog if isinstance(catalog, dict) else {}


def update_configure_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(
        mode=str(payload.get("mode") or ""),
        channel=str(payload.get("channel") or ""),
        track=str(payload.get("track") or ""),
        catalog_url=str(payload.get("catalog_url") or ""),
        check_on_start=bool(payload.get("check_on_start")),
        no_check_on_start=bool(payload.get("no_check_on_start")),
        auto_install_soft=payload.get("auto_install_soft"),
        no_auto_install_soft=bool(payload.get("no_auto_install_soft")),
        auto_install_hard=bool(payload.get("auto_install_hard")),
        no_auto_install_hard=bool(payload.get("no_auto_install_hard")),
        allow_prerelease=bool(payload.get("allow_prerelease")),
        no_allow_prerelease=bool(payload.get("no_allow_prerelease")),
    )
    return update_manager.configure_client(update_policy(), args)


def update_check_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(
        catalog_url=str(payload.get("catalog_url") or ""),
        update_id=str(payload.get("update_id") or ""),
        apply=bool(payload.get("apply")),
        if_enabled_on_start=bool(payload.get("if_enabled_on_start")),
        respect_interval=bool(payload.get("respect_interval")),
    )
    return update_manager.check_for_updates(update_policy(), args)


def update_apply_from_payload(payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    args = argparse.Namespace(
        mode=mode,
        execute=bool(payload.get("execute", True)),
        allow_hard=bool(payload.get("allow_hard")),
        restart=bool(payload.get("restart")),
        offer=str(payload.get("offer") or ""),
    )
    return update_manager.apply_update(update_policy(), args)
