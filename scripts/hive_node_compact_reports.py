"""Compact report-shaping helpers for the Theseus Hive node."""

from __future__ import annotations

from typing import Any

from hive_node_common import get_path


def compact_update_status(status: dict[str, Any]) -> dict[str, Any]:
    client = status.get("client") if isinstance(status.get("client"), dict) else {}
    offer = status.get("current_offer") if isinstance(status.get("current_offer"), dict) else {}
    checkin = status.get("last_checkin") if isinstance(status.get("last_checkin"), dict) else {}
    communication = status.get("communication") if isinstance(status.get("communication"), dict) else {}
    return {
        "mode": client.get("mode"),
        "channel": client.get("channel"),
        "track": client.get("track"),
        "check_on_start": client.get("check_on_start"),
        "auto_install_soft": client.get("auto_install_soft"),
        "auto_install_hard": client.get("auto_install_hard"),
        "update_available": status.get("update_available"),
        "soft_update_available": status.get("soft_update_available"),
        "hard_update_available": status.get("hard_update_available"),
        "restart_required": status.get("restart_required"),
        "offer_update_id": offer.get("update_id"),
        "offer_checkpoint_id": offer.get("checkpoint_id"),
        "headline": offer.get("headline"),
        "last_check_utc": checkin.get("created_utc"),
        "catalog_ok": checkin.get("catalog_ok"),
        "catalog_source": checkin.get("catalog_source"),
        "catalog_url_configured": communication.get("official_catalog_configured"),
        "next_action": status.get("next_action"),
    }

def compact_hive_version(status: dict[str, Any]) -> dict[str, Any]:
    local = status.get("local") if isinstance(status.get("local"), dict) else {}
    verified = status.get("verified_version") if isinstance(status.get("verified_version"), dict) else {}
    installers = status.get("installers") if isinstance(status.get("installers"), dict) else {}
    convergence = status.get("convergence") if isinstance(status.get("convergence"), dict) else {}
    return {
        "local_version_id": local.get("version_id"),
        "app_version": local.get("app_version"),
        "git_commit": get_path(local, ["git", "short_commit"], ""),
        "dirty": get_path(local, ["git", "dirty"], None),
        "verified_version_id": verified.get("version_id"),
        "verified_ok": verified.get("ok"),
        "installer_artifact_count": installers.get("artifact_count"),
        "target_version_id": convergence.get("target_version_id"),
        "local_matches_target": convergence.get("local_matches_target"),
    }

def compact_storage(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": status.get("enabled"),
        "share_count": status.get("share_count"),
        "shares": [
            {
                "share_id": share.get("share_id"),
                "name": share.get("name"),
                "kind": share.get("kind"),
                "accessible": share.get("accessible"),
                "tags": share.get("tags") or [],
            }
            for share in (status.get("shares") or [])[:12]
            if isinstance(share, dict) and share.get("enabled")
        ],
        "limits": status.get("limits") or {},
    }

def compact_remote_control(status: dict[str, Any]) -> dict[str, Any]:
    providers = status.get("providers") if isinstance(status.get("providers"), list) else []
    return {
        "enabled": status.get("enabled"),
        "ready_provider_count": status.get("ready_provider_count"),
        "preferred_provider_id": status.get("preferred_provider_id"),
        "providers": [
            {
                "id": provider.get("id"),
                "label": provider.get("label"),
                "ready": provider.get("ready"),
                "installed": provider.get("installed"),
                "configured": provider.get("configured"),
                "role": provider.get("role"),
                "rustdesk_id": provider.get("rustdesk_id"),
                "connect": provider.get("connect") or {},
            }
            for provider in providers
            if isinstance(provider, dict)
        ],
        "local": status.get("local") or {},
        "security": status.get("security") or {},
    }

def compact_rented_compute(status: dict[str, Any]) -> dict[str, Any]:
    last_plan = status.get("last_plan") if isinstance(status.get("last_plan"), dict) else {}
    return {
        "enabled": status.get("enabled"),
        "configured_profile_count": status.get("configured_profile_count"),
        "aws_cli_installed": get_path(status, ["prerequisites", "aws_cli_installed"], False),
        "gcloud_cli_installed": get_path(status, ["prerequisites", "gcloud_cli_installed"], False),
        "azure_cli_installed": get_path(status, ["prerequisites", "azure_cli_installed"], False),
        "vastai_cli_installed": get_path(status, ["prerequisites", "vastai_cli_installed"], False),
        "curl_installed": get_path(status, ["prerequisites", "curl_installed"], False),
        "join_config_available": get_path(status, ["prerequisites", "join_config_available"], False),
        "last_plan": {
            "decision": last_plan.get("decision"),
            "provider": last_plan.get("provider"),
            "profile": last_plan.get("profile"),
            "estimated_total_usd": last_plan.get("estimated_total_usd"),
        },
        "next_actions": status.get("next_actions", [])[:5] if isinstance(status.get("next_actions"), list) else [],
    }

def compact_utilization(status: dict[str, Any]) -> dict[str, Any]:
    summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
    return {
        "trigger_state": status.get("trigger_state"),
        "idle_slots": summary.get("idle_slots") or {},
        "busy_slots": summary.get("busy_slots") or {},
        "planned_actions": summary.get("planned_actions"),
        "executed_actions": summary.get("executed_actions"),
        "blocked": summary.get("blocked"),
        "active_or_planned_nodes": summary.get("active_or_planned_nodes"),
        "safe_idle_uncovered_nodes": summary.get("safe_idle_uncovered_nodes"),
        "blocked_nodes": summary.get("blocked_nodes"),
        "always_active_mode": get_path(status, ["always_active", "mode"], ""),
        "pause_flags": get_path(status, ["always_active", "pause_flags"], []),
        "stop_flags": get_path(status, ["always_active", "stop_flags"], []),
    }

def compact_operator_overnight(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "created_utc": report.get("created_utc"),
        "window_hours": report.get("window_hours"),
        "worker_report_count": summary.get("worker_report_count"),
        "promotion_count": summary.get("promotion_count"),
        "failed_count": summary.get("failed_count"),
        "stale_lease_count": summary.get("stale_lease_count"),
        "next_actions": (report.get("next_actions") if isinstance(report.get("next_actions"), list) else [])[:3],
    }

def compact_promoted(rows: list[Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows[-8:]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "arm_id": row.get("arm_id"),
                "job_id": row.get("job_id"),
                "round_id": row.get("round_id"),
                "score": row.get("score"),
                "backend": row.get("backend"),
                "promoted_model_path": row.get("promoted_model_path"),
            }
        )
    return out

def compact_training_execution(rows: list[Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows[-8:]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "job_id": row.get("job_id"),
                "task_kind": row.get("task_kind"),
                "target_node_id": row.get("target_node_id"),
                "ok": row.get("ok"),
                "duplicate": row.get("duplicate"),
                "error": row.get("error") or row.get("message"),
            }
        )
    return out

def compact_license(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "registration_complete": report.get("registration_complete"),
        "tier": get_path(report, ["entitlement", "tier"], ""),
        "source": get_path(report, ["entitlement", "source"], ""),
        "paid": get_path(report, ["entitlement", "paid"], False),
        "nodes_used": get_path(report, ["hive", "node_count"], None),
        "node_limit": get_path(report, ["entitlement", "node_limit"], None),
        "can_run_worker_chunks": get_path(report, ["feature_summary", "can_run_worker_chunks"], False),
        "can_create_company_hive": get_path(report, ["feature_summary", "can_create_company_hive"], False),
    }

def compact_runtime(report: dict[str, Any]) -> dict[str, Any]:
    paths = report.get("paths") if isinstance(report.get("paths"), dict) else {}
    return {
        "runtime_root": get_path(paths, ["runtime_root", "path"], ""),
        "data_dir": get_path(paths, ["data_dir", "path"], ""),
        "cache_dir": get_path(paths, ["cache_dir", "path"], ""),
        "reports_dir": get_path(paths, ["reports_dir", "path"], ""),
        "checkpoints_dir": get_path(paths, ["checkpoints_dir", "path"], ""),
        "cargo_target_dir": get_path(paths, ["cargo_target_dir", "path"], ""),
    }

def compact_market(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": report.get("mode"),
        "currency_symbol": get_path(report, ["currency", "symbol"], "TWC"),
        "exchange_enabled": bool(report.get("exchange_enabled")),
        "available_micro_twc": get_path(report, ["balances", "available_micro_twc"], 0),
        "earned_micro_twc": get_path(report, ["balances", "earned_micro_twc"], 0),
        "can_account_public_work": get_path(report, ["license", "can_account_public_work"], False),
    }
