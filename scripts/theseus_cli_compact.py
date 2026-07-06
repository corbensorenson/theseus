"""Status compaction helpers for the Project Theseus CLI."""

from __future__ import annotations

from typing import Any


def compact_probe(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": probe.get("node_id"),
        "node_name": probe.get("node_name"),
        "platform": probe.get("platform"),
        "capabilities": probe.get("capabilities", []),
    }


def compact_node(node: dict[str, Any]) -> dict[str, Any]:
    resources = node.get("resources") if isinstance(node.get("resources"), dict) else {}
    return {
        "node_id": node.get("node_id"),
        "node_name": node.get("node_name"),
        "api_url": node.get("api_url"),
        "hive_id": node.get("hive_id"),
        "tier": node.get("federation_tier"),
        "capabilities": node.get("capabilities", []),
        "resources": {
            "cpu": resources.get("cpu", {}),
            "memory": resources.get("memory", {}),
            "disk": resources.get("disk", {}),
            "nvidia": resources.get("nvidia", {}),
            "mlx": resources.get("mlx", {}),
        },
        "slots": node.get("slots", []),
        "storage": node.get("storage", {}),
        "voice_following": node.get("voice_following", {}),
        "runtime_paths": node.get("runtime_paths", {}),
    }


def compact_dashboard(status: dict[str, Any]) -> dict[str, Any]:
    daemon = status.get("daemon") if isinstance(status.get("daemon"), dict) else {}
    autonomy = status.get("autonomy") if isinstance(status.get("autonomy"), dict) else {}
    last = autonomy.get("last") if isinstance(autonomy.get("last"), dict) else {}
    return {
        "ok": bool(status),
        "created_utc": status.get("created_utc"),
        "daemon": {
            "running": daemon.get("running"),
            "pid": daemon.get("pid"),
            "profile": daemon.get("profile"),
            "stale": daemon.get("stale"),
        },
        "autonomy": {
            "cycle_id": last.get("cycle_id"),
            "created_utc": last.get("created_utc"),
            "ok": last.get("ok"),
            "profile": last.get("profile"),
            "teacher_needed": last.get("teacher_needed"),
            "teacher_used": last.get("teacher_used"),
            "frontier_family": get_path(last, ["decision", "frontier_family"], ""),
            "failed_candidate_gates": get_path(last, ["decision", "failed_candidate_gates"], []),
            "pause_flag": autonomy.get("pause_flag"),
            "stop_flag": autonomy.get("stop_flag"),
        },
    }


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    checks = candidate.get("checks") if isinstance(candidate.get("checks"), list) else []
    failed = [str(row.get("gate")) for row in checks if isinstance(row, dict) and not row.get("passed")]
    return {
        "promote": candidate.get("promote"),
        "passed": candidate.get("passed"),
        "total": candidate.get("total"),
        "failed_gates": failed,
    }


def compact_benchmarks(ledger: Any) -> dict[str, Any]:
    rows = ledger if isinstance(ledger, list) else []
    frontier = [row for row in rows if isinstance(row, dict) and row.get("lifecycle") == "frontier"]
    regression = [row for row in rows if isinstance(row, dict) and row.get("lifecycle") == "regression"]
    return {
        "total": len(rows),
        "frontier": len(frontier),
        "regression": len(regression),
        "frontier_names": [str(row.get("benchmark_name")) for row in frontier[:10]],
    }


def compact_peers(peers: dict[str, Any]) -> dict[str, Any]:
    return {
        "peer_count": peers.get("peer_count", 0),
        "peers": [
            {
                "node_name": row.get("node_name"),
                "api_url": row.get("api_url"),
                "capabilities": row.get("capabilities", []),
            }
            for row in (peers.get("peers") if isinstance(peers.get("peers"), list) else [])[:12]
            if isinstance(row, dict)
        ],
    }


def compact_scheduler(scheduler: dict[str, Any]) -> dict[str, Any]:
    summary = scheduler.get("summary") if isinstance(scheduler.get("summary"), dict) else {}
    return {
        "ok": bool(scheduler),
        "best_training_node": summary.get("best_training_node"),
        "best_cuda_node": summary.get("best_cuda_node"),
        "best_mlx_node": summary.get("best_mlx_node"),
        "best_inference_node": summary.get("best_inference_node"),
        "real_worker_chunks": summary.get("real_worker_chunks"),
        "placements": scheduler.get("placements", [])[:10] if isinstance(scheduler.get("placements"), list) else [],
    }


def compact_public(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": status.get("mode"),
        "enabled": status.get("enabled"),
        "can_connect_now": status.get("can_connect_now"),
        "gateway_url_configured": status.get("gateway_url_configured"),
        "next_action": status.get("next_action"),
    }


def compact_market(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": status.get("mode"),
        "currency_symbol": get_path(status, ["currency", "symbol"], "TWC"),
        "exchange_enabled": status.get("exchange_enabled"),
        "tradable_token_enabled": status.get("tradable_token_enabled"),
        "available_micro_twc": get_path(status, ["balances", "available_micro_twc"], 0),
        "earned_micro_twc": get_path(status, ["balances", "earned_micro_twc"], 0),
        "ledger_tail_events": get_path(status, ["summary", "ledger_tail_events"], 0),
        "can_account_public_work": get_path(status, ["license", "can_account_public_work"], False),
        "next_action": status.get("next_action"),
    }


def compact_rented_compute(status: dict[str, Any]) -> dict[str, Any]:
    last_plan = status.get("last_plan") if isinstance(status.get("last_plan"), dict) else {}
    providers = status.get("providers") if isinstance(status.get("providers"), list) else []
    return {
        "enabled": status.get("enabled"),
        "configured_profile_count": status.get("configured_profile_count"),
        "aws_cli_installed": get_path(status, ["prerequisites", "aws_cli_installed"], False),
        "gcloud_cli_installed": get_path(status, ["prerequisites", "gcloud_cli_installed"], False),
        "azure_cli_installed": get_path(status, ["prerequisites", "azure_cli_installed"], False),
        "vastai_cli_installed": get_path(status, ["prerequisites", "vastai_cli_installed"], False),
        "curl_installed": get_path(status, ["prerequisites", "curl_installed"], False),
        "join_config_available": get_path(status, ["prerequisites", "join_config_available"], False),
        "last_plan_decision": last_plan.get("decision"),
        "last_plan_provider": last_plan.get("provider"),
        "last_plan_profile": last_plan.get("profile"),
        "implemented_providers": [row.get("provider") for row in providers if isinstance(row, dict) and row.get("implemented")],
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
        "always_active_mode": get_path(status, ["always_active", "mode"], ""),
        "blockers": status.get("blockers") if isinstance(status.get("blockers"), list) else [],
    }


def compact_remote(status: dict[str, Any]) -> dict[str, Any]:
    relay = status.get("relay") if isinstance(status.get("relay"), dict) else {}
    local = status.get("local") if isinstance(status.get("local"), dict) else {}
    cost = status.get("cost_policy") if isinstance(status.get("cost_policy"), dict) else {}
    return {
        "relay_configured": relay.get("configured"),
        "relay_url": relay.get("url"),
        "relay_scope": relay.get("scope"),
        "relay_live": local.get("relay_live"),
        "wireguard_tools_installed": local.get("wireguard_tools_installed"),
        "paid_dependency_required": cost.get("paid_dependency_required", False),
        "next_actions": status.get("next_actions", [])[:5] if isinstance(status.get("next_actions"), list) else [],
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
            }
            for provider in providers
            if isinstance(provider, dict)
        ],
        "next_actions": status.get("next_actions", [])[:5] if isinstance(status.get("next_actions"), list) else [],
    }


def compact_voice(status: dict[str, Any]) -> dict[str, Any]:
    room = status.get("room") if isinstance(status.get("room"), dict) else {}
    microphone = status.get("microphone") if isinstance(status.get("microphone"), dict) else {}
    speaker = status.get("speaker") if isinstance(status.get("speaker"), dict) else {}
    presence = status.get("presence") if isinstance(status.get("presence"), dict) else {}
    route = status.get("route") if isinstance(status.get("route"), dict) else {}
    return {
        "enabled": status.get("enabled"),
        "room_id": room.get("room_id"),
        "room_name": room.get("name"),
        "microphone_ready": microphone.get("ready"),
        "speaker_ready": speaker.get("ready"),
        "microphone_available": microphone.get("available"),
        "speaker_available": speaker.get("available"),
        "presence_score": presence.get("score"),
        "presence_age_seconds": presence.get("age_seconds"),
        "route_state": route.get("state"),
        "respond_node_name": get_path(route, ["respond_node", "node_name"], ""),
        "next_actions": status.get("next_actions", [])[:5] if isinstance(status.get("next_actions"), list) else [],
    }


def compact_license(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "registration_complete": status.get("registration_complete"),
        "tier": get_path(status, ["entitlement", "tier"], ""),
        "source": get_path(status, ["entitlement", "source"], ""),
        "paid": get_path(status, ["entitlement", "paid"], False),
        "node_limit": get_path(status, ["entitlement", "node_limit"], None),
        "nodes_used": get_path(status, ["hive", "node_count"], None),
        "can_create_company_hive": get_path(status, ["feature_summary", "can_create_company_hive"], False),
        "can_operate_public_gateway": get_path(status, ["feature_summary", "can_operate_public_gateway"], False),
        "next_action": status.get("next_action"),
    }


def compact_runtime(status: dict[str, Any]) -> dict[str, Any]:
    paths = status.get("paths") if isinstance(status.get("paths"), dict) else {}
    migration = status.get("migration") if isinstance(status.get("migration"), dict) else {}
    rows = migration.get("managed_directories") if isinstance(migration.get("managed_directories"), list) else []
    return {
        "runtime_root": get_path(paths, ["runtime_root", "path"], ""),
        "reports_dir": get_path(paths, ["reports_dir", "path"], ""),
        "checkpoints_dir": get_path(paths, ["checkpoints_dir", "path"], ""),
        "cargo_target_dir": get_path(paths, ["cargo_target_dir", "path"], ""),
        "managed_directories": [
            {
                "name": row.get("name"),
                "status": row.get("status"),
                "source_size_gib": row.get("source_size_gib"),
                "target": row.get("target"),
            }
            for row in rows
        ],
    }


def compact_openai(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": status.get("enabled"),
        "live": status.get("live"),
        "base_url": status.get("base_url"),
        "model": status.get("model"),
        "checkpoint_id": status.get("checkpoint_id"),
        "require_token": status.get("require_token"),
        "license_allowed": get_path(status, ["license", "allowed"], None),
        "external_inference_calls": status.get("external_inference_calls"),
    }


def compact_updates(status: dict[str, Any]) -> dict[str, Any]:
    offer = status.get("current_offer") if isinstance(status.get("current_offer"), dict) else {}
    installed = status.get("installed") if isinstance(status.get("installed"), dict) else {}
    client = status.get("client") if isinstance(status.get("client"), dict) else {}
    catalog = status.get("catalog") if isinstance(status.get("catalog"), dict) else {}
    communication = status.get("communication") if isinstance(status.get("communication"), dict) else {}
    return {
        "update_available": status.get("update_available"),
        "soft_update_available": status.get("soft_update_available"),
        "hard_update_available": status.get("hard_update_available"),
        "restart_required": status.get("restart_required"),
        "mode": client.get("mode"),
        "check_on_start": client.get("check_on_start"),
        "auto_install_soft": client.get("auto_install_soft"),
        "auto_install_hard": client.get("auto_install_hard"),
        "catalog_ok": get_path(status, ["last_checkin", "catalog_ok"], None),
        "catalog_source": get_path(status, ["last_checkin", "catalog_source"], ""),
        "catalog_offer_count": catalog.get("offer_count"),
        "catalog_url_configured": communication.get("official_catalog_configured"),
        "offer_update_id": offer.get("update_id"),
        "offer_checkpoint_id": offer.get("checkpoint_id"),
        "headline": offer.get("headline"),
        "installed_update_id": installed.get("active_update_id"),
        "installed_checkpoint_id": installed.get("active_checkpoint_id"),
        "license_allowed": get_path(status, ["license", "allowed"], None),
        "next_action": status.get("next_action"),
    }


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
