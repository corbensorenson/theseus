"""Candidate-tied update manager for Project Theseus.

Accepted candidates become update offers. Soft updates only activate the
candidate metadata/checkpoint state and notify clients. Hard updates materialize
the checkpoint and copy source/app changes only when explicitly executed with
the hard-update guard, while skipping protected local/company paths and arms.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError
from urllib.parse import urlparse

import checkpoint_registry  # noqa: E402
import license_manager  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "update_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status")
    status.add_argument("--out", default="")

    check = sub.add_parser("check")
    check.add_argument("--catalog-url", default="")
    check.add_argument("--update-id", default="")
    check.add_argument("--apply", action="store_true")
    check.add_argument("--if-enabled-on-start", action="store_true")
    check.add_argument("--respect-interval", action="store_true")
    check.add_argument("--out", default="")

    configure = sub.add_parser("configure")
    configure.add_argument("--mode", choices=["manual", "notify", "auto_soft", "auto_safe"], default="")
    configure.add_argument("--channel", default="")
    configure.add_argument("--track", choices=["stable", "beta", "dev"], default="")
    configure.add_argument("--catalog-url", default="")
    configure.add_argument("--check-on-start", action="store_true")
    configure.add_argument("--no-check-on-start", action="store_true")
    configure.add_argument("--auto-install-soft", action="store_true")
    configure.add_argument("--no-auto-install-soft", action="store_true")
    configure.add_argument("--auto-install-hard", action="store_true")
    configure.add_argument("--no-auto-install-hard", action="store_true")
    configure.add_argument("--allow-prerelease", action="store_true")
    configure.add_argument("--no-allow-prerelease", action="store_true")
    configure.add_argument("--out", default="")

    catalog = sub.add_parser("catalog")
    catalog.add_argument("--out", default="")

    create = sub.add_parser("create")
    create.add_argument("--checkpoint-id", default="")
    create.add_argument("--if-promoted", action="store_true")
    create.add_argument("--out", default="")

    apply = sub.add_parser("apply")
    apply.add_argument("--mode", choices=["auto", "soft", "hard"], default="auto")
    apply.add_argument("--execute", action="store_true")
    apply.add_argument("--allow-hard", action="store_true")
    apply.add_argument("--restart", action="store_true")
    apply.add_argument("--offer", default="")
    apply.add_argument("--out", default="")

    protect_arm = sub.add_parser("protect-arm")
    protect_arm.add_argument("arm_name")
    protect_arm.add_argument("--reason", default="")
    protect_arm.add_argument("--out", default="")

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command == "create":
        report = create_offer(policy, args)
    elif args.command == "apply":
        report = apply_update(policy, args)
    elif args.command == "check":
        report = check_for_updates(policy, args)
    elif args.command == "configure":
        report = configure_client(policy, args)
    elif args.command == "catalog":
        report = public_catalog(policy)
    elif args.command == "protect-arm":
        report = protect_arm(policy, args.arm_name, args.reason)
    else:
        report = status_report(policy=policy, write_report=True)
    out = getattr(args, "out", "") or ""
    if out:
        write_json(ROOT / out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def create_offer(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidate = read_json(ROOT / get_path(policy, ["candidate_trigger", "candidate_gate"], "reports/candidate_promotion_gate.json"), {})
    promoted = bool(candidate.get("promote"))
    checkpoint = resolve_checkpoint(policy, args.checkpoint_id)
    checkpoint_id = str(checkpoint.get("checkpoint_id") or args.checkpoint_id or "")
    required_status = str(get_path(policy, ["candidate_trigger", "require_checkpoint_status"], "promoted") or "")
    checkpoint_promoted = bool(required_status and checkpoint.get("status") == required_status)
    base = {
        "ok": True,
        "policy": "project_theseus_update_offer_report_v0",
        "created_utc": now(),
        "candidate_promote": promoted,
        "candidate_gate": f"{candidate.get('passed')}/{candidate.get('total')}",
        "checkpoint_id": checkpoint_id,
        "checkpoint_status": checkpoint.get("status"),
        "external_inference_calls": 0,
    }
    if args.if_promoted and not promoted and not checkpoint_promoted:
        report = {
            **base,
            "status": "skipped_not_promoted",
            "reason": "Neither the current candidate gate nor the resolved checkpoint is promoted, so no update offer is created.",
        }
        write_status(policy, {**status_report(policy=policy), "last_create": report})
        return report
    if not checkpoint:
        return {**base, "ok": False, "status": "blocked_missing_checkpoint", "reason": "No checkpoint manifest could be resolved."}
    if args.if_promoted and required_status and checkpoint.get("status") != required_status:
        return {
            **base,
            "ok": False,
            "status": "blocked_checkpoint_not_promoted",
            "reason": f"Checkpoint status must be {required_status!r} for candidate update creation.",
        }

    snapshot_paths = checkpoint_snapshot_paths(checkpoint)
    hard_paths = sorted(path for path in snapshot_paths if is_hard_path(policy, path))
    protected_hits = sorted(path for path in snapshot_paths if is_protected_path(policy, path))
    protected_arms = detect_protected_arms(policy)
    hard_kind = bool(hard_paths)
    update_id = stable_update_id(checkpoint, candidate)
    offer = {
        **base,
        "status": "offer_ready",
        "update_id": update_id,
        "channel": recommended_channel(),
        "update_kind": "hard_required" if hard_kind else "soft",
        "soft_available": True,
        "hard_available": hard_kind,
        "restart_required": hard_kind,
        "auto_install_soft": bool(get_path(policy, ["channels", recommended_channel(), "auto_install_soft"], True)),
        "auto_install_hard": bool(get_path(policy, ["channels", recommended_channel(), "auto_install_hard"], False)),
        "checkpoint": checkpoint_summary(checkpoint),
        "checkpoint_scores": checkpoint.get("scores", {}),
        "improvement_summary": build_improvement_summary(policy, checkpoint, candidate),
        "payload": {
            "snapshot_files": len(snapshot_paths),
            "hard_paths": hard_paths[:80],
            "hard_path_count": len(hard_paths),
            "protected_hits": protected_hits[:80],
            "protected_hit_count": len(protected_hits),
            "protected_arms": protected_arms,
        },
        "install_guidance": install_guidance(hard_kind),
        "external_inference_calls": 0,
    }
    offer_path = path_from_policy(policy, ["paths", "current_offer"], "reports/update_offer_current.json")
    write_json(offer_path, offer)
    append_jsonl(path_from_policy(policy, ["paths", "history"], "reports/update_history.jsonl"), compact_offer_history(offer))
    append_jsonl(path_from_policy(policy, ["paths", "events"], "reports/update_events.jsonl"), event("offer_created", {"update_id": update_id, "checkpoint_id": checkpoint_id, "kind": offer["update_kind"]}))
    write_status(policy, {**status_report(policy=policy), "last_create": offer})
    return offer


def apply_update(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    offer_path = ROOT / args.offer if args.offer else path_from_policy(policy, ["paths", "current_offer"], "reports/update_offer_current.json")
    offer = read_json(offer_path, {})
    if not offer or offer.get("status") not in {"offer_ready", "soft_installed", "hard_staged", "hard_installed"}:
        return {"ok": False, "status": "blocked_missing_offer", "reason": "No update offer is available to apply."}
    channel = str(offer.get("channel") or recommended_channel())
    license_feature = str(get_path(policy, ["channels", channel, "license_feature"], "local_research"))
    license_check = license_manager.check_feature(license_feature, write_report=True)
    if not license_check.get("allowed") and license_feature == "private_update_channel":
        license_check = license_manager.check_feature("local_research", write_report=True)
    if not license_check.get("allowed"):
        return {
            "ok": False,
            "status": "blocked_license",
            "feature": license_feature,
            "license": compact_license_check(license_check),
        }

    mode = args.mode
    if mode == "auto":
        mode = "hard" if offer.get("hard_available") and bool(get_path(policy, ["channels", channel, "auto_install_hard"], False)) else "soft"
    if mode == "soft":
        return apply_soft(policy, offer, execute=bool(args.execute), license_check=license_check)
    return apply_hard(policy, offer, execute=bool(args.execute), allow_hard=bool(args.allow_hard), restart=bool(args.restart), license_check=license_check)


def configure_client(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    path = path_from_policy(policy, ["paths", "local_config"], "configs/update_client.local.json")
    cfg = read_json(path, {})
    cfg = cfg if isinstance(cfg, dict) else {}
    cfg.setdefault("policy", "project_theseus_update_client_local_v0")
    cfg["updated_utc"] = now()
    if args.mode:
        cfg["mode"] = args.mode
    if args.channel:
        cfg["channel"] = args.channel
    if args.track:
        cfg["track"] = args.track
    if args.catalog_url:
        cfg["catalog_url"] = args.catalog_url
    if args.check_on_start:
        cfg["check_on_start"] = True
    if args.no_check_on_start:
        cfg["check_on_start"] = False
    if args.auto_install_soft:
        cfg["auto_install_soft"] = True
    if args.no_auto_install_soft:
        cfg["auto_install_soft"] = False
    if args.auto_install_hard:
        cfg["auto_install_hard"] = True
    if args.no_auto_install_hard:
        cfg["auto_install_hard"] = False
    if args.allow_prerelease:
        cfg["allow_prerelease"] = True
    if args.no_allow_prerelease:
        cfg["allow_prerelease"] = False
    defaults = client_defaults(policy)
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    # Keep hard updates opt-in unless the user explicitly set the local switch.
    if cfg.get("mode") in {"auto_soft", "auto_safe"}:
        cfg["auto_install_soft"] = True
    cfg["auto_install_hard"] = bool(cfg.get("auto_install_hard", False))
    write_json(path, cfg)
    append_jsonl(path_from_policy(policy, ["paths", "events"], "reports/update_events.jsonl"), event("client_configured", public_client_config(cfg)))
    report = status_report(policy=policy, write_report=True)
    report["client_config_written"] = rel_path(path)
    return report


def check_for_updates(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = effective_client_config(policy)
    if getattr(args, "catalog_url", ""):
        cfg["catalog_url"] = args.catalog_url
    last_checkin = read_json(path_from_policy(policy, ["paths", "checkin"], "reports/update_checkin.json"), {})
    if getattr(args, "if_enabled_on_start", False) and not bool(cfg.get("check_on_start", True)):
        return skipped_checkin(policy, "skipped_startup_checks_disabled", cfg, last_checkin, "Startup update checks are disabled by the local update settings.")
    if getattr(args, "respect_interval", False):
        age = checkin_age_seconds(last_checkin)
        interval = int(cfg.get("auto_check_interval_seconds") or 21600)
        if age is not None and age < interval:
            return skipped_checkin(
                policy,
                "skipped_recent_check",
                cfg,
                last_checkin,
                f"Last update check was {int(age)} seconds ago; interval is {interval} seconds.",
            )
    catalog_report = fetch_catalog_report(policy, cfg)
    selected = select_catalog_offer(policy, catalog_report.get("catalog", {}), cfg, update_id=str(getattr(args, "update_id", "") or ""))
    installed = read_json(path_from_policy(policy, ["paths", "installed_state"], "configs/update_installed.local.json"), {})
    apply_report: dict[str, Any] = {}
    current_offer: dict[str, Any] = {}
    if selected.get("ok") and selected.get("offer"):
        current_offer = normalize_catalog_offer(policy, selected["offer"], catalog_report)
        write_json(path_from_policy(policy, ["paths", "current_offer"], "reports/update_offer_current.json"), current_offer)
        append_jsonl(
            path_from_policy(policy, ["paths", "events"], "reports/update_events.jsonl"),
            event("catalog_offer_selected", {"update_id": current_offer.get("update_id"), "catalog_source": catalog_report.get("source")}),
        )
        if should_auto_apply(cfg, current_offer) or getattr(args, "apply", False):
            apply_args = argparse.Namespace(
                mode="soft",
                execute=True,
                allow_hard=False,
                restart=False,
                offer=str(path_from_policy(policy, ["paths", "current_offer"], "reports/update_offer_current.json")),
            )
            apply_report = apply_update(policy, apply_args)
    checkin = {
        "ok": bool(catalog_report.get("ok")) and bool(selected.get("ok", True)),
        "policy": "project_theseus_update_checkin_v0",
        "created_utc": now(),
        "client": public_client_config(cfg),
        "catalog": compact_catalog(catalog_report.get("catalog", {})),
        "catalog_source": catalog_report.get("source"),
        "catalog_url": catalog_report.get("url", ""),
        "catalog_ok": bool(catalog_report.get("ok")),
        "catalog_error": catalog_report.get("error", ""),
        "selected": selected,
        "installed": compact_installed(installed),
        "current_offer": compact_offer(current_offer or read_json(path_from_policy(policy, ["paths", "current_offer"], "reports/update_offer_current.json"), {})),
        "auto_apply": apply_report,
        "next_action": update_next_action(cfg, selected, current_offer, apply_report),
    }
    write_json(path_from_policy(policy, ["paths", "catalog_cache"], "reports/update_catalog_cache.json"), catalog_report)
    write_json(path_from_policy(policy, ["paths", "checkin"], "reports/update_checkin.json"), checkin)
    write_status(policy, status_report(policy=policy, write_report=False))
    return checkin


def public_catalog(policy: dict[str, Any]) -> dict[str, Any]:
    offer = read_json(path_from_policy(policy, ["paths", "current_offer"], "reports/update_offer_current.json"), {})
    installed = read_json(path_from_policy(policy, ["paths", "installed_state"], "configs/update_installed.local.json"), {})
    app = app_version_info()
    offers = []
    if offer.get("update_id"):
        offers.append(public_catalog_offer(offer, app))
    return {
        "ok": True,
        "policy": "project_theseus_public_update_catalog_v0",
        "created_utc": now(),
        "catalog_id": "local-hive",
        "channel": recommended_channel(),
        "track": "stable",
        "latest_app_version": app,
        "latest_model_checkpoint": compact_installed(installed),
        "offers": offers,
        "communication": {
            "catalog_path": get_path(policy, ["public_catalog", "hive_catalog_path"], "/api/hive/update-catalog"),
            "status_path": "/api/hive/status",
            "artifact_index_path": "/api/hive/artifacts",
        },
    }


def public_catalog_offer(offer: dict[str, Any], app: dict[str, Any]) -> dict[str, Any]:
    payload = offer.get("payload") if isinstance(offer.get("payload"), dict) else {}
    improvement = offer.get("improvement_summary") if isinstance(offer.get("improvement_summary"), dict) else {}
    return {
        "ok": True,
        "policy": "project_theseus_update_offer_report_v0",
        "status": offer.get("status", "offer_ready"),
        "update_id": offer.get("update_id"),
        "channel": offer.get("channel") or recommended_channel(),
        "track": offer.get("track") or "stable",
        "update_kind": offer.get("update_kind"),
        "soft_available": bool(offer.get("soft_available", True)),
        "hard_available": bool(offer.get("hard_available", False)),
        "restart_required": bool(offer.get("restart_required", False)),
        "auto_install_soft": bool(offer.get("auto_install_soft", False)),
        "auto_install_hard": bool(offer.get("auto_install_hard", False)),
        "checkpoint_id": offer.get("checkpoint_id"),
        "checkpoint_status": offer.get("checkpoint_status"),
        "candidate_gate": offer.get("candidate_gate"),
        "checkpoint": checkpoint_summary(offer.get("checkpoint") if isinstance(offer.get("checkpoint"), dict) else {}),
        "improvement_summary": {
            "headline": improvement.get("headline", ""),
            "candidate_gate": improvement.get("candidate_gate", {}),
            "what_users_should_notice": improvement.get("what_users_should_notice", []),
        },
        "payload": {
            "snapshot_files": payload.get("snapshot_files"),
            "hard_path_count": payload.get("hard_path_count"),
            "protected_hit_count": payload.get("protected_hit_count"),
            "protected_arm_count": len(payload.get("protected_arms") if isinstance(payload.get("protected_arms"), list) else []),
        },
        "install_guidance": offer.get("install_guidance"),
        "app_version": app.get("version"),
        "created_utc": offer.get("created_utc"),
        "published_utc": offer.get("published_utc") or offer.get("created_utc"),
        "source": "local_hive_catalog",
        "external_inference_calls": 0,
    }


def apply_soft(policy: dict[str, Any], offer: dict[str, Any], *, execute: bool, license_check: dict[str, Any]) -> dict[str, Any]:
    installed = read_json(path_from_policy(policy, ["paths", "installed_state"], "configs/update_installed.local.json"), {})
    report = {
        "ok": True,
        "policy": "project_theseus_update_apply_report_v0",
        "created_utc": now(),
        "mode": "soft",
        "execute": execute,
        "status": "soft_installed" if execute else "soft_dry_run_ready",
        "update_id": offer.get("update_id"),
        "checkpoint_id": offer.get("checkpoint_id"),
        "restart_required": False,
        "previous_update_id": installed.get("active_update_id"),
        "improvement_summary": user_improvement_summary(offer),
        "protected": protection_summary(policy),
        "license": compact_license_check(license_check),
        "external_inference_calls": 0,
    }
    if execute:
        state = {
            "policy": "project_theseus_installed_update_state_v0",
            "updated_utc": now(),
            "active_update_id": offer.get("update_id"),
            "active_checkpoint_id": offer.get("checkpoint_id"),
            "active_checkpoint_status": offer.get("checkpoint_status"),
            "soft_installed_utc": now(),
            "hard_installed_utc": installed.get("hard_installed_utc"),
            "restart_required": False,
            "update_kind": offer.get("update_kind"),
            "improvement_summary": offer.get("improvement_summary", {}),
            "scores": offer.get("checkpoint_scores", {}),
            "protected": protection_summary(policy),
        }
        write_json(path_from_policy(policy, ["paths", "installed_state"], "configs/update_installed.local.json"), state)
        install_root = path_from_policy(policy, ["paths", "installed_manifests"], "updates/installed")
        write_json(install_root / f"{offer.get('update_id')}.json", offer)
        append_jsonl(path_from_policy(policy, ["paths", "events"], "reports/update_events.jsonl"), event("soft_update_installed", {"update_id": offer.get("update_id"), "checkpoint_id": offer.get("checkpoint_id")}))
    write_status(policy, status_report(policy=policy, write_report=False))
    return report


def apply_hard(policy: dict[str, Any], offer: dict[str, Any], *, execute: bool, allow_hard: bool, restart: bool, license_check: dict[str, Any]) -> dict[str, Any]:
    checkpoint_id = str(offer.get("checkpoint_id") or "")
    if not checkpoint_id:
        return {"ok": False, "status": "blocked_missing_checkpoint_id"}
    if not execute:
        return {
            "ok": True,
            "status": "hard_dry_run_ready",
            "mode": "hard",
            "checkpoint_id": checkpoint_id,
            "restart_required": True,
            "guidance": "Re-run with --execute --allow-hard to stage/copy app/source changes.",
            "protected": protection_summary(policy),
            "license": compact_license_check(license_check),
        }
    if not allow_hard:
        return {"ok": False, "status": "blocked_allow_hard_required", "reason": "Hard updates require --allow-hard."}

    materialized = materialize_update(policy, checkpoint_id)
    if not materialized.get("ok"):
        return {"ok": False, "status": "blocked_materialize_failed", "materialize": materialized}
    source_root = Path(materialized["out"]).resolve()
    copied, skipped, errors = copy_materialized_workspace(policy, source_root)
    status = "hard_installed_restart_required" if not errors else "hard_partially_installed"
    installed = read_json(path_from_policy(policy, ["paths", "installed_state"], "configs/update_installed.local.json"), {})
    installed.update(
        {
            "policy": "project_theseus_installed_update_state_v0",
            "updated_utc": now(),
            "active_update_id": offer.get("update_id"),
            "active_checkpoint_id": checkpoint_id,
            "hard_installed_utc": now(),
            "restart_required": True,
            "restart_requested": bool(restart),
            "update_kind": offer.get("update_kind"),
            "improvement_summary": offer.get("improvement_summary", {}),
            "scores": offer.get("checkpoint_scores", {}),
            "protected": protection_summary(policy),
        }
    )
    write_json(path_from_policy(policy, ["paths", "installed_state"], "configs/update_installed.local.json"), installed)
    append_jsonl(path_from_policy(policy, ["paths", "events"], "reports/update_events.jsonl"), event("hard_update_installed", {"update_id": offer.get("update_id"), "checkpoint_id": checkpoint_id, "copied": len(copied), "skipped": len(skipped), "errors": len(errors)}))
    report = {
        "ok": not errors,
        "policy": "project_theseus_update_apply_report_v0",
        "created_utc": now(),
        "status": status,
        "mode": "hard",
        "checkpoint_id": checkpoint_id,
        "update_id": offer.get("update_id"),
        "restart_required": True,
        "restart_requested": bool(restart),
        "materialized": materialized,
        "copied": copied[:200],
        "copied_count": len(copied),
        "skipped": skipped[:200],
        "skipped_count": len(skipped),
        "errors": errors[:50],
        "license": compact_license_check(license_check),
        "external_inference_calls": 0,
    }
    write_status(policy, status_report(policy=policy, write_report=False))
    return report


def status_report(*, policy: dict[str, Any] | None = None, write_report: bool = False) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    offer = read_json(path_from_policy(policy, ["paths", "current_offer"], "reports/update_offer_current.json"), {})
    installed = read_json(path_from_policy(policy, ["paths", "installed_state"], "configs/update_installed.local.json"), {})
    checkin = read_json(path_from_policy(policy, ["paths", "checkin"], "reports/update_checkin.json"), {})
    cfg = effective_client_config(policy)
    channel = str(offer.get("channel") or recommended_channel())
    feature = str(get_path(policy, ["channels", channel, "license_feature"], "local_research"))
    license_check = license_manager.check_feature(feature, write_report=True)
    if not license_check.get("allowed") and feature == "private_update_channel":
        license_check = license_manager.check_feature("local_research", write_report=True)
    update_available = bool(offer.get("update_id") and offer.get("update_id") != installed.get("active_update_id"))
    catalog_summary = checkin_catalog_summary(checkin)
    report = {
        "ok": True,
        "policy": "project_theseus_update_status_v0",
        "created_utc": now(),
        "enabled": bool(policy.get("enabled", True)),
        "channel": channel,
        "app_version": app_version_info(),
        "client": public_client_config(cfg),
        "catalog": catalog_summary,
        "catalog_offers": catalog_offer_options(policy),
        "last_checkin": compact_checkin(checkin),
        "update_available": update_available,
        "soft_update_available": bool(update_available and offer.get("soft_available", True)),
        "hard_update_available": bool(update_available and offer.get("hard_available")),
        "restart_required": bool(installed.get("restart_required") or (update_available and offer.get("restart_required"))),
        "current_offer": compact_offer(offer),
        "installed": compact_installed(installed),
        "license": compact_license_check(license_check),
        "protected": protection_summary(policy),
        "communication": update_communication_status(policy, cfg, checkin),
        "last_events": read_jsonl_tail(path_from_policy(policy, ["paths", "events"], "reports/update_events.jsonl"), 20),
        "next_action": status_next_action(cfg, update_available, offer, checkin),
        "external_inference_calls": 0,
    }
    if write_report:
        write_status(policy, report)
    return report


def effective_client_config(policy: dict[str, Any]) -> dict[str, Any]:
    cfg = read_json(path_from_policy(policy, ["paths", "local_config"], "configs/update_client.local.json"), {})
    cfg = cfg if isinstance(cfg, dict) else {}
    defaults = client_defaults(policy)
    merged = {**defaults, **cfg}
    merged["mode"] = str(merged.get("mode") or defaults["mode"])
    merged["channel"] = str(merged.get("channel") or recommended_channel())
    merged["track"] = str(merged.get("track") or "stable")
    merged["catalog_url"] = str(merged.get("catalog_url") or discover_catalog_url(policy, merged))
    merged["auto_install_soft"] = bool(merged.get("auto_install_soft") or merged["mode"] in {"auto_soft", "auto_safe"})
    merged["auto_install_hard"] = bool(merged.get("auto_install_hard", False))
    return merged


def client_defaults(policy: dict[str, Any]) -> dict[str, Any]:
    defaults = get_path(policy, ["client_defaults"], {})
    if not isinstance(defaults, dict):
        defaults = {}
    return {
        "mode": str(defaults.get("mode") or "notify"),
        "channel": str(defaults.get("channel") or recommended_channel()),
        "track": str(defaults.get("track") or "stable"),
        "check_on_start": bool(defaults.get("check_on_start", True)),
        "auto_check_interval_seconds": int(defaults.get("auto_check_interval_seconds") or 21600),
        "auto_install_soft": bool(defaults.get("auto_install_soft", False)),
        "auto_install_hard": bool(defaults.get("auto_install_hard", False)),
        "allow_prerelease": bool(defaults.get("allow_prerelease", False)),
        "friendly_beginner_mode": bool(defaults.get("friendly_beginner_mode", True)),
        "catalog_url": "",
    }


def public_client_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": cfg.get("mode"),
        "channel": cfg.get("channel"),
        "track": cfg.get("track"),
        "catalog_url_configured": bool(cfg.get("catalog_url")),
        "catalog_url": cfg.get("catalog_url") or "",
        "check_on_start": bool(cfg.get("check_on_start")),
        "auto_install_soft": bool(cfg.get("auto_install_soft")),
        "auto_install_hard": bool(cfg.get("auto_install_hard")),
        "allow_prerelease": bool(cfg.get("allow_prerelease")),
    }


def discover_catalog_url(policy: dict[str, Any], cfg: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["public_catalog", "url_env"], "THESEUS_UPDATE_CATALOG_URL"))
    if os.environ.get(env_name):
        return str(os.environ[env_name])
    default_url = str(get_path(policy, ["public_catalog", "default_url"], "") or "")
    if default_url:
        return default_url
    hive_url = discover_hive_catalog_url(policy)
    if hive_url:
        return hive_url
    public_cfg = read_json(ROOT / "configs" / "public_hive_contribution.local.json", {})
    gateway = str(public_cfg.get("gateway_url") or "").rstrip("/")
    if gateway:
        return gateway + str(get_path(policy, ["public_catalog", "gateway_catalog_path"], "/api/theseus/updates/catalog"))
    return ""


def discover_hive_catalog_url(policy: dict[str, Any]) -> str:
    """Find the private Hive update catalog from local join/profile state."""
    catalog_path = str(get_path(policy, ["public_catalog", "hive_catalog_path"], "/api/hive/update-catalog") or "/api/hive/update-catalog")
    candidates: list[str] = []
    join = read_json(ROOT / "configs" / "hive_join.local.json", {})
    add_hive_endpoint_candidates(candidates, join)
    profiles = read_json(ROOT / "configs" / "hive_profiles.local.json", {})
    active_profile_id = str(profiles.get("active_profile_id") or "") if isinstance(profiles, dict) else ""
    rows = profiles.get("profiles") if isinstance(profiles.get("profiles"), list) else []
    for profile in rows:
        if isinstance(profile, dict) and (not active_profile_id or str(profile.get("profile_id") or "") == active_profile_id):
            add_hive_endpoint_candidates(candidates, profile)
    status = read_json(ROOT / "reports" / "hive_status.json", {})
    if isinstance(status, dict):
        candidates.append(str(status.get("api_url") or ""))
    for candidate in candidates:
        url = catalog_url_from_base(candidate, catalog_path)
        if url:
            return url
    return ""


def add_hive_endpoint_candidates(candidates: list[str], payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    for key in ["coordinator_url", "relay_url", "api_url"]:
        value = str(payload.get(key) or "").strip()
        if value:
            candidates.append(value)
    for key in ["coordinator_urls", "relay_urls", "node_urls"]:
        values = payload.get(key)
        if isinstance(values, list):
            candidates.extend(str(item).strip() for item in values if str(item).strip())


def catalog_url_from_base(base: str, catalog_path: str) -> str:
    base = str(base or "").strip()
    if not base:
        return ""
    if base.endswith(catalog_path):
        return base
    if "/api/hive/" in base:
        base = base.split("/api/hive/", 1)[0]
    return base.rstrip("/") + catalog_path


def fetch_catalog_report(policy: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    url = str(cfg.get("catalog_url") or "")
    if not url:
        catalog = public_catalog(policy)
        return {
            "ok": True,
            "policy": "project_theseus_update_catalog_fetch_v0",
            "created_utc": now(),
            "source": "local_hive",
            "url": "",
            "catalog": catalog,
        }
    parsed = urlparse(url)
    try:
        if parsed.scheme in {"http", "https"}:
            if parsed.scheme != "https" and not is_private_catalog_host(parsed.hostname or ""):
                return {"ok": False, "source": "remote_http_blocked", "url": url, "error": "remote update catalogs must use https unless they are private LAN/loopback hosts"}
            timeout = int(get_path(policy, ["public_catalog", "timeout_seconds"], 8))
            max_bytes = int(get_path(policy, ["public_catalog", "max_catalog_bytes"], 1_048_576))
            request = urlrequest.Request(url, headers={"User-Agent": "ProjectTheseusUpdateClient/0.1"})
            with urlrequest.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-configured update catalog.
                raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                return {"ok": False, "source": "remote", "url": url, "error": "catalog_too_large"}
            catalog = json.loads(raw.decode("utf-8"))
            source = "remote"
        else:
            path = Path(url).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            catalog = read_json(path, {})
            source = "file"
        if not isinstance(catalog, dict):
            return {"ok": False, "source": source, "url": url, "error": "catalog_not_object"}
        return {
            "ok": True,
            "policy": "project_theseus_update_catalog_fetch_v0",
            "created_utc": now(),
            "source": source,
            "url": url,
            "catalog": catalog,
        }
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "policy": "project_theseus_update_catalog_fetch_v0",
            "created_utc": now(),
            "source": "remote" if parsed.scheme in {"http", "https"} else "file",
            "url": url,
            "error": str(exc),
            "catalog": public_catalog(policy),
            "fallback_source": "local_hive",
        }


def is_private_catalog_host(host: str) -> bool:
    return (
        host in {"localhost", "127.0.0.1", "::1"}
        or host.startswith("127.")
        or host.startswith("10.")
        or host.startswith("192.168.")
        or any(host.startswith(f"172.{idx}.") for idx in range(16, 32))
    )


def select_catalog_offer(policy: dict[str, Any], catalog: dict[str, Any], cfg: dict[str, Any], *, update_id: str = "") -> dict[str, Any]:
    offers = catalog.get("offers") if isinstance(catalog.get("offers"), list) else []
    installed = read_json(path_from_policy(policy, ["paths", "installed_state"], "configs/update_installed.local.json"), {})
    active_update = str(installed.get("active_update_id") or "")
    channel = str(cfg.get("channel") or recommended_channel())
    track = str(cfg.get("track") or "stable")
    allow_prerelease = bool(cfg.get("allow_prerelease"))
    candidates: list[dict[str, Any]] = []
    requested: dict[str, Any] = {}
    for offer in offers:
        if not isinstance(offer, dict) or not offer.get("update_id"):
            continue
        if update_id and str(offer.get("update_id")) == update_id:
            requested = offer
        if str(offer.get("update_id")) == active_update:
            continue
        if offer.get("channel") and str(offer.get("channel")) != channel:
            continue
        offer_track = str(offer.get("track") or catalog.get("track") or "stable")
        if offer_track != track and not (allow_prerelease and offer_track in {"beta", "dev"}):
            continue
        if bool(offer.get("prerelease")) and not allow_prerelease:
            continue
        candidates.append(offer)
    if update_id:
        if not requested:
            return {"ok": False, "status": "requested_offer_not_found", "update_id": update_id, "candidate_count": len(candidates)}
        if not any(str(row.get("update_id")) == update_id for row in candidates):
            return {
                "ok": False,
                "status": "requested_offer_not_compatible",
                "update_id": update_id,
                "reason": "The selected update is already installed or does not match this channel/track/prerelease policy.",
                "candidate_count": len(candidates),
            }
        selected = requested
        return {
            "ok": True,
            "status": "update_available",
            "candidate_count": len(candidates),
            "requested": True,
            "update_id": selected.get("update_id"),
            "checkpoint_id": selected.get("checkpoint_id"),
            "offer": selected,
        }
    if not candidates:
        return {"ok": True, "status": "no_update", "reason": "No compatible newer offer was found.", "candidate_count": 0}
    candidates.sort(key=lambda row: str(row.get("published_utc") or row.get("created_utc") or ""), reverse=True)
    selected = candidates[0]
    return {
        "ok": True,
        "status": "update_available",
        "candidate_count": len(candidates),
        "update_id": selected.get("update_id"),
        "checkpoint_id": selected.get("checkpoint_id"),
        "offer": selected,
    }


def normalize_catalog_offer(policy: dict[str, Any], offer: dict[str, Any], catalog_report: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(offer)
    normalized.setdefault("ok", True)
    normalized.setdefault("policy", "project_theseus_update_offer_report_v0")
    normalized.setdefault("status", "offer_ready")
    normalized.setdefault("created_utc", offer.get("published_utc") or offer.get("created_utc") or now())
    normalized.setdefault("channel", recommended_channel())
    normalized.setdefault("update_kind", "hard_required" if normalized.get("hard_available") else "soft")
    normalized.setdefault("soft_available", True)
    normalized.setdefault("hard_available", False)
    normalized.setdefault("restart_required", bool(normalized.get("hard_available")))
    normalized.setdefault("auto_install_soft", False)
    normalized.setdefault("auto_install_hard", False)
    normalized.setdefault("checkpoint_status", get_path(normalized, ["checkpoint", "status"], "public_catalog"))
    normalized.setdefault("improvement_summary", {})
    improvement = normalized["improvement_summary"] if isinstance(normalized["improvement_summary"], dict) else {}
    improvement.setdefault("headline", normalized.get("headline") or f"Project Theseus update {normalized.get('update_id')} is available.")
    improvement.setdefault("what_users_should_notice", normalized.get("what_users_should_notice", []))
    normalized["improvement_summary"] = improvement
    normalized["catalog_source"] = {
        "source": catalog_report.get("source"),
        "url": catalog_report.get("url", ""),
        "checked_utc": catalog_report.get("created_utc"),
    }
    return normalized


def should_auto_apply(cfg: dict[str, Any], offer: dict[str, Any]) -> bool:
    if cfg.get("mode") not in {"auto_soft", "auto_safe"}:
        return False
    if offer.get("soft_available", True):
        return bool(cfg.get("auto_install_soft", True))
    return False


def update_next_action(cfg: dict[str, Any], selected: dict[str, Any], offer: dict[str, Any], apply_report: dict[str, Any]) -> str:
    if apply_report.get("status") == "soft_installed":
        return "Soft update installed. Keep using Project Theseus normally."
    if selected.get("status") == "update_available":
        if cfg.get("mode") == "manual":
            return "An update is available. Open Updates and choose when to install it."
        if cfg.get("mode") == "notify":
            return "An update is available. Project Theseus will notify you and wait for your choice."
        if offer.get("hard_available"):
            return "The safe part of the update can be installed automatically; app/source changes still need your approval."
        return "The update is ready for automatic soft install."
    if selected.get("status") == "no_update":
        return "This install is current for the selected update channel."
    return "Update check completed."


def status_next_action(cfg: dict[str, Any], update_available: bool, offer: dict[str, Any], checkin: dict[str, Any]) -> str:
    if update_available and cfg.get("mode") == "manual":
        return "Update available. Review it and install when ready."
    if update_available and cfg.get("mode") == "notify":
        return "Update available. Auto-update is off; choose Install when ready."
    if update_available and cfg.get("mode") in {"auto_soft", "auto_safe"}:
        return "Update available. Safe automatic update will run on the next check-in."
    if not checkin:
        return "No public catalog check has run yet."
    if checkin.get("catalog_ok"):
        return "This install is current for the selected channel."
    return "Could not reach the configured update catalog; local/offline update state is still available."


def skipped_checkin(policy: dict[str, Any], status: str, cfg: dict[str, Any], last_checkin: dict[str, Any], reason: str) -> dict[str, Any]:
    report = {
        "ok": True,
        "policy": "project_theseus_update_checkin_v0",
        "created_utc": now(),
        "status": status,
        "reason": reason,
        "client": public_client_config(cfg),
        "last_checkin": compact_checkin(last_checkin),
        "catalog": checkin_catalog_summary(last_checkin),
        "catalog_source": last_checkin.get("catalog_source") if isinstance(last_checkin, dict) else "",
        "catalog_url": last_checkin.get("catalog_url") if isinstance(last_checkin, dict) else "",
        "catalog_ok": bool(last_checkin.get("catalog_ok")) if isinstance(last_checkin, dict) else False,
        "current_offer": compact_offer(read_json(path_from_policy(policy, ["paths", "current_offer"], "reports/update_offer_current.json"), {})),
        "next_action": reason,
    }
    write_status(policy, status_report(policy=policy, write_report=False))
    return report


def checkin_age_seconds(checkin: dict[str, Any]) -> float | None:
    if not isinstance(checkin, dict) or not checkin.get("created_utc"):
        return None
    try:
        created = datetime.fromisoformat(str(checkin["created_utc"]).replace("Z", "+00:00"))
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - created).total_seconds())


def checkin_catalog_summary(checkin: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(checkin, dict):
        return {}
    catalog = checkin.get("catalog") if isinstance(checkin.get("catalog"), dict) else {}
    if not catalog:
        return {}
    if "offer_count" in catalog or "latest_offer" in catalog:
        return catalog
    return compact_catalog(catalog)


def catalog_offer_options(policy: dict[str, Any]) -> list[dict[str, Any]]:
    cache = read_json(path_from_policy(policy, ["paths", "catalog_cache"], "reports/update_catalog_cache.json"), {})
    catalog = cache.get("catalog") if isinstance(cache.get("catalog"), dict) else {}
    offers = catalog.get("offers") if isinstance(catalog.get("offers"), list) else []
    rows = [catalog_offer_option(row) for row in offers if isinstance(row, dict) and row.get("update_id")]
    return sorted(rows, key=lambda row: str(row.get("published_utc") or row.get("created_utc") or ""), reverse=True)[:20]


def catalog_offer_option(offer: dict[str, Any]) -> dict[str, Any]:
    compact = compact_offer(offer)
    return {
        **compact,
        "published_utc": offer.get("published_utc") or offer.get("created_utc"),
        "channel": offer.get("channel"),
        "track": offer.get("track"),
        "app_version": offer.get("app_version"),
        "prerelease": bool(offer.get("prerelease")),
    }


def compact_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(catalog, dict):
        return {}
    offers = catalog.get("offers") if isinstance(catalog.get("offers"), list) else []
    return {
        "policy": catalog.get("policy"),
        "catalog_id": catalog.get("catalog_id"),
        "created_utc": catalog.get("created_utc"),
        "channel": catalog.get("channel"),
        "track": catalog.get("track"),
        "latest_app_version": catalog.get("latest_app_version"),
        "latest_model_checkpoint": catalog.get("latest_model_checkpoint"),
        "offer_count": len(offers),
        "latest_offer": compact_offer(offers[0]) if offers and isinstance(offers[0], dict) else {},
    }


def compact_checkin(checkin: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(checkin, dict) or not checkin:
        return {}
    return {
        "created_utc": checkin.get("created_utc"),
        "catalog_ok": checkin.get("catalog_ok"),
        "catalog_source": checkin.get("catalog_source"),
        "catalog_url": checkin.get("catalog_url"),
        "selected_status": get_path(checkin, ["selected", "status"], ""),
        "next_action": checkin.get("next_action"),
    }


def update_communication_status(policy: dict[str, Any], cfg: dict[str, Any], checkin: dict[str, Any]) -> dict[str, Any]:
    return {
        "official_catalog_configured": bool(cfg.get("catalog_url")),
        "catalog_url": cfg.get("catalog_url") or "",
        "last_check_ok": bool(checkin.get("catalog_ok")) if checkin else False,
        "last_check_utc": checkin.get("created_utc") if isinstance(checkin, dict) else "",
        "hive_catalog_endpoint": get_path(policy, ["public_catalog", "hive_catalog_path"], "/api/hive/update-catalog"),
        "artifact_endpoint": "/api/hive/artifacts",
        "private_hive_fallback": True,
    }


def app_version_info() -> dict[str, Any]:
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8") if (ROOT / "Cargo.toml").exists() else ""
    version = "0.0.0"
    for line in cargo.splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped:
            version = stripped.split("=", 1)[1].strip().strip('"')
            break
    return {
        "app": "Project Theseus",
        "version": version,
        "source": "Cargo.toml",
    }


def materialize_update(policy: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    root = path_from_policy(policy, ["paths", "materialized_root"], "updates/materialized").resolve()
    out = root / checkpoint_id
    if not is_within(out, root):
        return {"ok": False, "error": "unsafe_materialize_path", "out": str(out)}
    if out.exists():
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()
    checkpoint_policy = checkpoint_registry.read_json(ROOT / "configs" / "autonomy_policy.json")
    return checkpoint_registry.materialize_checkpoint(checkpoint_policy, checkpoint_id, out, force=False)


def copy_materialized_workspace(policy: dict[str, Any], source_root: Path) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    materialized_root = path_from_policy(policy, ["paths", "materialized_root"], "updates/materialized").resolve()
    if not is_within(source_root, materialized_root):
        return [], [], [{"path": str(source_root), "error": "source_not_under_materialized_root"}]
    copied: list[str] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        rel = str(source.relative_to(source_root)).replace("\\", "/")
        target = (ROOT / rel).resolve()
        if not is_within(target, ROOT.resolve()):
            errors.append({"path": rel, "error": "target_outside_root"})
            continue
        if is_protected_path(policy, rel) or not is_hard_path(policy, rel):
            skipped.append({"path": rel, "reason": "protected_or_not_hard_payload"})
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(rel)
        except OSError as exc:
            errors.append({"path": rel, "error": str(exc)})
    return copied, skipped, errors


def resolve_checkpoint(policy: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    checkpoint_policy = checkpoint_registry.read_json(ROOT / "configs" / "autonomy_policy.json")
    if checkpoint_id:
        return checkpoint_registry.load_checkpoint_manifest(checkpoint_policy, checkpoint_id)
    checkpoint_last = read_json(ROOT / get_path(policy, ["candidate_trigger", "checkpoint_last"], "reports/checkpoint_last.json"), {})
    if checkpoint_last.get("checkpoint_id"):
        return checkpoint_last
    registry = read_json(ROOT / get_path(policy, ["candidate_trigger", "checkpoint_registry"], "reports/checkpoint_registry.json"), {})
    for item in reversed(registry.get("checkpoints", []) if isinstance(registry.get("checkpoints"), list) else []):
        if item.get("status") == "promoted" or item.get("promote"):
            return checkpoint_registry.load_checkpoint_manifest(checkpoint_policy, str(item.get("checkpoint_id") or ""))
    return {}


def checkpoint_snapshot_paths(checkpoint: dict[str, Any]) -> list[str]:
    files = get_path(checkpoint, ["snapshot", "files"], {})
    if isinstance(files, dict):
        return sorted(str(path).replace("\\", "/") for path in files)
    return []


def is_hard_path(policy: dict[str, Any], rel: str) -> bool:
    patterns = get_path(policy, ["hard_update", "source_path_patterns"], [])
    return any(fnmatch.fnmatch(rel, str(pattern)) for pattern in patterns)


def is_protected_path(policy: dict[str, Any], rel: str) -> bool:
    local = read_json(path_from_policy(policy, ["paths", "local_config"], "configs/update_client.local.json"), {})
    patterns = list(get_path(policy, ["protected_paths"], []))
    patterns.extend(str(item) for item in local.get("protected_paths", []) if item)
    return any(fnmatch.fnmatch(rel, str(pattern)) for pattern in patterns)


def detect_protected_arms(policy: dict[str, Any]) -> list[dict[str, Any]]:
    local = read_json(path_from_policy(policy, ["paths", "local_config"], "configs/update_client.local.json"), {})
    protected_names = set(str(item) for item in local.get("protected_arms", []) if item)
    markers = set(str(item) for item in get_path(policy, ["protected_arm_markers"], []))
    owner_values = set(str(item) for item in get_path(policy, ["protected_arm_owner_values"], []))
    rows: list[dict[str, Any]] = []
    for path in [ROOT / "reports" / "arm_registry.json", ROOT / "configs" / "arm_registry.local.json", ROOT / "configs" / "company_arms.local.json"]:
        registry = read_json(path, {})
        arms = registry.get("arms") if isinstance(registry, dict) else []
        for arm in arms if isinstance(arms, list) else []:
            if not isinstance(arm, dict):
                continue
            name = str(arm.get("name") or arm.get("arm_name") or arm.get("id") or "")
            owner = str(arm.get("owner") or arm.get("ownership") or "").lower()
            marker_hit = any(bool(arm.get(marker)) for marker in markers)
            owner_hit = owner in owner_values or any(value in owner for value in owner_values)
            if name in protected_names or marker_hit or owner_hit:
                rows.append({"name": name, "owner": owner, "source": rel_path(path), "reason": "local_or_company_protected"})
    for name in protected_names:
        if not any(row.get("name") == name for row in rows):
            rows.append({"name": name, "source": rel_path(path_from_policy(policy, ["paths", "local_config"], "configs/update_client.local.json")), "reason": "local_config"})
    return sorted(rows, key=lambda row: str(row.get("name") or ""))


def protect_arm(policy: dict[str, Any], arm_name: str, reason: str) -> dict[str, Any]:
    config_path = path_from_policy(policy, ["paths", "local_config"], "configs/update_client.local.json")
    cfg = read_json(config_path, {})
    protected = [str(item) for item in cfg.get("protected_arms", []) if item]
    if arm_name not in protected:
        protected.append(arm_name)
    cfg.update(
        {
            "policy": "project_theseus_update_client_local_v0",
            "updated_utc": now(),
            "protected_arms": sorted(protected),
            "protected_arm_notes": {**(cfg.get("protected_arm_notes") if isinstance(cfg.get("protected_arm_notes"), dict) else {}), arm_name: reason or "user protected"},
        }
    )
    write_json(config_path, cfg)
    append_jsonl(path_from_policy(policy, ["paths", "events"], "reports/update_events.jsonl"), event("arm_protected", {"arm_name": arm_name, "reason": reason}))
    return {"ok": True, "status": "arm_protected", "arm_name": arm_name, "config": rel_path(config_path)}


def build_improvement_summary(policy: dict[str, Any], checkpoint: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    installed = read_json(path_from_policy(policy, ["paths", "installed_state"], "configs/update_installed.local.json"), {})
    previous_scores = installed.get("scores") if isinstance(installed.get("scores"), dict) else {}
    current_scores = checkpoint.get("scores") if isinstance(checkpoint.get("scores"), dict) else {}
    improvements: list[dict[str, Any]] = []
    min_delta = float(get_path(policy, ["offer_summary", "min_score_delta_to_call_improved"], 0.0001))
    for key, row in current_scores.items():
        if not isinstance(row, dict):
            continue
        current = first_number(row, ["score", "accuracy", "public_accuracy", "active_frontier_accuracy", "seed55_frontier_accuracy"])
        prev = None
        if isinstance(previous_scores.get(key), dict):
            prev = first_number(previous_scores[key], ["score", "accuracy", "public_accuracy", "active_frontier_accuracy", "seed55_frontier_accuracy"])
        if current is not None and (prev is None or current - prev >= min_delta):
            improvements.append({"benchmark": key, "score": current, "previous": prev, "delta": None if prev is None else round(current - prev, 6)})
    candidate_scores = candidate.get("scores") if isinstance(candidate.get("scores"), dict) else {}
    for key, value in candidate_scores.items():
        if isinstance(value, (int, float)):
            improvements.append({"benchmark": f"candidate:{key}", "score": float(value), "previous": None, "delta": None})
    max_items = int(get_path(policy, ["offer_summary", "max_improvements"], 12))
    residual = read_json(ROOT / "reports" / "residual_escrow.json", {})
    capability = read_json(ROOT / "reports" / "capability_matrix.json", {})
    return {
        "headline": improvement_headline(candidate, improvements),
        "candidate_gate": {
            "promote": candidate.get("promote"),
            "passed": candidate.get("passed"),
            "total": candidate.get("total"),
            "failed_gates": [row.get("gate") for row in candidate.get("checks", []) if isinstance(row, dict) and not row.get("passed")],
        },
        "scores": candidate_scores,
        "improvements": improvements[:max_items],
        "residuals": summarize_residuals(residual),
        "capabilities": {
            "average_maturity": get_path(capability, ["summary", "average_maturity"], None),
            "ready_or_active": get_path(capability, ["summary", "ready_or_active"], None),
            "top_gaps": get_path(capability, ["summary", "top_gaps"], [])[:5],
        },
        "what_users_should_notice": user_visible_improvements(candidate_scores, improvements),
    }


def improvement_headline(candidate: dict[str, Any], improvements: list[dict[str, Any]]) -> str:
    if candidate.get("promote"):
        return f"Accepted candidate passed {candidate.get('passed')}/{candidate.get('total')} promotion gates."
    if improvements:
        best = improvements[0]
        return f"Candidate update captured measured progress on {best.get('benchmark')}."
    return "Candidate update captured a new checkpoint with preserved governance metadata."


def user_visible_improvements(scores: dict[str, Any], improvements: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    if "seed55_frontier_accuracy" in scores:
        rows.append(f"Seed55 mutated frontier accuracy tracked at {float(scores['seed55_frontier_accuracy']):.3f}.")
    if "seed49_regression_accuracy" in scores:
        rows.append(f"Seed49 regression hold tracked at {float(scores['seed49_regression_accuracy']):.3f}.")
    if "public_accuracy" in scores:
        rows.append(f"Public comparator accuracy tracked at {float(scores['public_accuracy']):.3f}.")
    for item in improvements[:4]:
        name = item.get("benchmark")
        score = item.get("score")
        if isinstance(score, (int, float)):
            rows.append(f"{name} score available at {score:.3f}.")
    return rows[:8]


def summarize_residuals(residual: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(residual, dict):
        return {}
    return {
        "clusters": get_path(residual, ["summary", "clusters"], get_path(residual, ["clusters"], None)),
        "active": get_path(residual, ["summary", "active"], None),
        "critical": get_path(residual, ["summary", "critical"], None),
        "promotion_status": get_path(residual, ["summary", "promotion_status"], None),
    }


def install_guidance(hard_kind: bool) -> str:
    if hard_kind:
        return "Soft update can activate the accepted checkpoint now. Hard app/source changes are staged and require an intentional restart."
    return "Soft update can activate this accepted checkpoint without restart."


def user_improvement_summary(offer: dict[str, Any]) -> dict[str, Any]:
    improvement = offer.get("improvement_summary") if isinstance(offer.get("improvement_summary"), dict) else {}
    return {
        "headline": improvement.get("headline", ""),
        "candidate_gate": improvement.get("candidate_gate", {}),
        "what_users_should_notice": improvement.get("what_users_should_notice", []),
    }


def protection_summary(policy: dict[str, Any]) -> dict[str, Any]:
    local = read_json(path_from_policy(policy, ["paths", "local_config"], "configs/update_client.local.json"), {})
    return {
        "protected_path_patterns": len(get_path(policy, ["protected_paths"], [])) + len(local.get("protected_paths", []) if isinstance(local.get("protected_paths"), list) else []),
        "protected_arms": detect_protected_arms(policy),
        "company_arms_protected": bool(get_path(policy, ["channels", "company", "protect_company_arms"], True)),
    }


def checkpoint_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "label": checkpoint.get("label"),
        "status": checkpoint.get("status"),
        "profile": checkpoint.get("profile"),
        "created_utc": checkpoint.get("created_utc"),
        "snapshot_kind": get_path(checkpoint, ["snapshot", "kind"], checkpoint.get("snapshot_kind")),
        "chain_depth": get_path(checkpoint, ["snapshot", "chain_depth"], checkpoint.get("chain_depth")),
        "state_hash": get_path(checkpoint, ["snapshot", "state_hash"], checkpoint.get("state_hash")),
        "chain_hash": get_path(checkpoint, ["snapshot", "chain_hash"], checkpoint.get("chain_hash")),
    }


def compact_offer(offer: dict[str, Any]) -> dict[str, Any]:
    if not offer or not offer.get("update_id"):
        return {}
    return {
        "update_id": offer.get("update_id"),
        "status": offer.get("status"),
        "checkpoint_id": offer.get("checkpoint_id"),
        "checkpoint_status": offer.get("checkpoint_status"),
        "candidate_gate": offer.get("candidate_gate"),
        "update_kind": offer.get("update_kind"),
        "soft_available": offer.get("soft_available"),
        "hard_available": offer.get("hard_available"),
        "restart_required": offer.get("restart_required"),
        "created_utc": offer.get("created_utc"),
        "headline": get_path(offer, ["improvement_summary", "headline"], ""),
        "what_users_should_notice": get_path(offer, ["improvement_summary", "what_users_should_notice"], []),
    }


def compact_installed(installed: dict[str, Any]) -> dict[str, Any]:
    if not installed:
        return {}
    return {
        "active_update_id": installed.get("active_update_id"),
        "active_checkpoint_id": installed.get("active_checkpoint_id"),
        "soft_installed_utc": installed.get("soft_installed_utc"),
        "hard_installed_utc": installed.get("hard_installed_utc"),
        "restart_required": installed.get("restart_required"),
        "headline": get_path(installed, ["improvement_summary", "headline"], ""),
    }


def compact_offer_history(offer: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_utc": now(),
        "update_id": offer.get("update_id"),
        "checkpoint_id": offer.get("checkpoint_id"),
        "update_kind": offer.get("update_kind"),
        "candidate_gate": offer.get("candidate_gate"),
        "headline": get_path(offer, ["improvement_summary", "headline"], ""),
    }


def compact_license_check(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed": report.get("allowed", True),
        "feature": get_path(report, ["feature_check", "feature"], ""),
        "tier": get_path(report, ["entitlement", "tier"], ""),
        "source": get_path(report, ["entitlement", "source"], ""),
        "next_action": report.get("next_action"),
    }


def recommended_channel() -> str:
    license_status = license_manager.status_report(write_report=False)
    tier = str(get_path(license_status, ["hive", "tier"], "") or get_path(license_status, ["entitlement", "tier"], ""))
    if tier == "company":
        return "company"
    if tier == "public":
        return "public"
    return "community"


def stable_update_id(checkpoint: dict[str, Any], candidate: dict[str, Any]) -> str:
    seed = "|".join(
        [
            str(checkpoint.get("checkpoint_id") or ""),
            str(get_path(checkpoint, ["snapshot", "chain_hash"], "")),
            str(candidate.get("passed") or ""),
            str(candidate.get("total") or ""),
        ]
    )
    return "theseus-update-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def first_number(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def path_from_policy(policy: dict[str, Any], path_keys: list[str], default: str) -> Path:
    return ROOT / str(get_path(policy, path_keys, default))


def write_status(policy: dict[str, Any], report: dict[str, Any]) -> None:
    write_json(path_from_policy(policy, ["paths", "status"], "reports/update_status.json"), report)


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl_tail(path: Path, count: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-count:]


def event(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"created_utc": now(), "kind": kind, **payload}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
