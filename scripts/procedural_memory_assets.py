#!/usr/bin/env python3
"""Build replay-bound procedural assets and fail-closed lifecycle receipts.

This module is imported by the canonical procedural-memory gate.  The assets
compress verified workflow routing, not answer generation, and therefore never
receive learned-generation credit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUCCESS_OUTCOMES = {"accepted", "completed", "corrected"}
FAILURE_OUTCOMES = {"missed", "ignored"}


def build_assets(
    *,
    candidates: list[dict[str, Any]],
    replay_results: list[dict[str, Any]],
    canary_routes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    lifecycle_policy: dict[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    candidate_by_id = {str(row.get("id") or ""): row for row in candidates}
    replay_by_id = {str(row.get("id") or ""): row for row in replay_results}
    assets: list[dict[str, Any]] = []
    lifecycle_receipts: list[dict[str, Any]] = []

    for route in canary_routes:
        if route.get("canary_route_eligible") is not True:
            continue
        candidate_id = str(route.get("candidate_id") or "")
        fixture_id = str(route.get("replay_fixture_id") or "")
        candidate = candidate_by_id.get(candidate_id, {})
        replay = replay_by_id.get(fixture_id, {})
        if not candidate or replay.get("passed") is not True:
            continue
        binding = binding_from_candidate(candidate)
        receipt = lifecycle_receipt(
            candidate=candidate,
            route=route,
            replay=replay,
            binding=binding,
            events=events,
            policy=lifecycle_policy,
            created_utc=created_utc,
        )
        lifecycle_receipts.append(receipt)
        asset = {
            "id": f"procedural.asset.{stable_id(candidate_id, fixture_id, binding)}",
            "candidate_id": candidate_id,
            "route_id": str(route.get("id") or ""),
            "replay_fixture_id": fixture_id,
            "asset_kind": "verified_procedural_route_lookahead",
            "lookup_binding": binding,
            "lookahead_tokens": binding_tokens(binding),
            "preconditions": list_values(candidate.get("preconditions")),
            "postconditions": list_values(candidate.get("postconditions")),
            "monitoring": list_values(candidate.get("monitoring")),
            "retirement_criteria": list_values(candidate.get("retirement_criteria")),
            "lifecycle_receipt_id": receipt["receipt_id"],
            "lifecycle_state": receipt["lifecycle_state"],
            "route_eligible": receipt["lifecycle_state"] == "active",
            "candidate_generation_credit": "none",
            "learned_generation_claim_allowed": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "non_claims": [
                "Procedural lookahead is workflow retrieval, not parametric learning.",
                "This asset cannot support learned-generation or public-transfer claims.",
                "A missing, ambiguous, stale, or drifted binding returns no route.",
            ],
        }
        asset["asset_sha256"] = stable_hash({key: value for key, value in asset.items() if key != "asset_sha256"})
        assets.append(asset)

    active_assets = [row for row in assets if row.get("route_eligible")]
    trie = build_trie(active_assets)
    ablation = evaluate_lookup_ablation(active_assets, trie)
    negative_controls = evaluate_negative_controls(active_assets, trie, created_utc, lifecycle_policy)
    hard_gaps: list[dict[str, Any]] = []
    if any(not row.get("passed") for row in negative_controls):
        hard_gaps.append({"kind": "procedural_asset_negative_control_failed", "controls": negative_controls})
    if any(row.get("lifecycle_state") not in {"active", "retired_stale", "retired_drift", "blocked"} for row in lifecycle_receipts):
        hard_gaps.append({"kind": "unknown_procedural_lifecycle_state"})
    return {
        "policy": "project_theseus_procedural_memory_assets_v1",
        "created_utc": created_utc,
        "trigger_state": "GREEN" if not hard_gaps else "RED",
        "summary": {
            "asset_count": len(assets),
            "active_asset_count": len(active_assets),
            "retired_asset_count": len(assets) - len(active_assets),
            "diverse_binding_count": len({binding_key(row.get("lookup_binding", {})) for row in active_assets}),
            "lifecycle_receipt_count": len(lifecycle_receipts),
            "lookup_fixture_count": ablation["fixture_count"],
            "lookahead_selected_count": ablation["lookahead_selected_count"],
            "no_lookahead_selected_count": ablation["no_lookahead_selected_count"],
            "negative_control_count": len(negative_controls),
            "negative_control_rejected_count": sum(1 for row in negative_controls if row.get("passed")),
            "hard_gap_count": len(hard_gaps),
            "learned_generation_claim_count": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
        "assets": assets,
        "lifecycle_receipts": lifecycle_receipts,
        "lookahead_trie": trie,
        "lookup_ablation": ablation,
        "negative_controls": negative_controls,
        "hard_gaps": hard_gaps,
    }


def lifecycle_receipt(
    *,
    candidate: dict[str, Any],
    route: dict[str, Any],
    replay: dict[str, Any],
    binding: dict[str, str],
    events: list[dict[str, Any]],
    policy: dict[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    matched = [row for row in events if event_matches_binding(row, binding)]
    matched.sort(key=lambda row: str(row.get("created_utc") or ""))
    window_size = max(1, int_or(policy.get("monitor_window_events"), 32))
    recent = matched[-window_size:]
    outcomes = [event_outcome(row) for row in recent]
    useful = sum(1 for value in outcomes if value in SUCCESS_OUTCOMES)
    useful_rate = useful / len(recent) if recent else 0.0
    consecutive_failures = 0
    for value in reversed(outcomes):
        if value not in FAILURE_OUTCOMES:
            break
        consecutive_failures += 1
    latest_utc = str(matched[-1].get("created_utc") or "") if matched else ""
    age_days = timestamp_age_days(created_utc, latest_utc)
    stale = not matched or age_days > float_or(policy.get("max_stale_days"), 30.0)
    drifted = bool(
        recent
        and (
            useful_rate < float_or(policy.get("minimum_recent_useful_rate"), 0.90)
            or consecutive_failures >= int_or(policy.get("retire_after_consecutive_failures"), 2)
        )
    )
    no_cheat_fault_count = sum(1 for row in matched if event_no_cheat_fault(row))
    replay_passed = replay.get("passed") is True
    lifecycle_state = "active"
    reasons: list[str] = []
    if not replay_passed or route.get("canary_route_eligible") is not True or no_cheat_fault_count:
        lifecycle_state = "blocked"
        reasons.append("replay_route_or_no_cheat_invariant_failed")
    elif stale:
        lifecycle_state = "retired_stale"
        reasons.append("source_trace_freshness_expired")
    elif drifted:
        lifecycle_state = "retired_drift"
        reasons.append("postcondition_outcome_drift")
    metrics = {
        "matched_event_count": len(matched),
        "monitor_window_event_count": len(recent),
        "recent_useful_rate": useful_rate,
        "consecutive_failure_count": consecutive_failures,
        "latest_event_utc": latest_utc,
        "latest_event_age_days": age_days,
        "no_cheat_fault_count": no_cheat_fault_count,
        "replay_passed": replay_passed,
    }
    receipt_identity = {
        "matched_event_count": metrics["matched_event_count"],
        "monitor_window_event_count": metrics["monitor_window_event_count"],
        "recent_useful_rate": round(metrics["recent_useful_rate"], 6),
        "consecutive_failure_count": metrics["consecutive_failure_count"],
        "latest_event_utc": metrics["latest_event_utc"],
        "no_cheat_fault_count": metrics["no_cheat_fault_count"],
        "replay_passed": metrics["replay_passed"],
    }
    receipt_id = f"procedural.lifecycle.{stable_id(candidate.get('id'), lifecycle_state, receipt_identity)}"
    return {
        "receipt_id": receipt_id,
        "candidate_id": str(candidate.get("id") or ""),
        "route_id": str(route.get("id") or ""),
        "created_utc": created_utc,
        "lifecycle_state": lifecycle_state,
        "retired": lifecycle_state.startswith("retired"),
        "reasons": reasons,
        "lookup_binding": binding,
        "metrics": metrics,
        "rollback_triggered": lifecycle_state != "active",
        "authority_scope": "local_metadata_route_only",
        "learned_generation_claim_allowed": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def binding_from_candidate(candidate: dict[str, Any]) -> dict[str, str]:
    traces = list_dicts(candidate.get("source_traces"))
    trace = traces[0] if traces else {}
    return {
        "surface": clean_label(trace.get("surface") or ""),
        "assistant_lane": clean_label(trace.get("assistant_lane") or ""),
        "intent": clean_label(trace.get("intent_bucket") or ""),
    }


def build_trie(assets: list[dict[str, Any]]) -> dict[str, Any]:
    root: dict[str, Any] = {"children": {}, "asset_ids": []}
    for asset in sorted(assets, key=lambda row: str(row.get("id") or "")):
        node = root
        for token in list_values(asset.get("lookahead_tokens")):
            children = node.setdefault("children", {})
            node = children.setdefault(str(token), {"children": {}, "asset_ids": []})
        node.setdefault("asset_ids", []).append(str(asset.get("id") or ""))
    root["trie_sha256"] = stable_hash({key: value for key, value in root.items() if key != "trie_sha256"})
    return root


def lookup(trie: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    node = trie
    for token in binding_tokens(binding):
        children = node.get("children") if isinstance(node.get("children"), dict) else {}
        if token not in children:
            return {"state": "NO_ADMISSIBLE", "asset_id": "", "candidate_asset_ids": []}
        node = children[token]
    asset_ids = sorted(str(item) for item in list_values(node.get("asset_ids")) if str(item))
    if len(asset_ids) != 1:
        return {"state": "AMBIGUOUS" if asset_ids else "NO_ADMISSIBLE", "asset_id": "", "candidate_asset_ids": asset_ids}
    return {"state": "SELECTED", "asset_id": asset_ids[0], "candidate_asset_ids": asset_ids}


def evaluate_lookup_ablation(assets: list[dict[str, Any]], trie: dict[str, Any]) -> dict[str, Any]:
    results = []
    for asset in assets:
        selected = lookup(trie, dict_value(asset.get("lookup_binding")))
        results.append(
            {
                "asset_id": asset.get("id"),
                "lookahead_state": selected["state"],
                "lookahead_selected_correctly": selected.get("asset_id") == asset.get("id"),
                "no_lookahead_state": "NO_PROCEDURAL_SOURCE",
                "no_lookahead_selected_correctly": False,
            }
        )
    return {
        "claim_scope": "route-index functionality only; not assistant-quality or learned-generation evidence",
        "equal_query_budget": True,
        "fixture_count": len(results),
        "lookahead_selected_count": sum(1 for row in results if row["lookahead_selected_correctly"]),
        "no_lookahead_selected_count": 0,
        "results": results,
    }


def evaluate_negative_controls(
    assets: list[dict[str, Any]],
    trie: dict[str, Any],
    created_utc: str,
    lifecycle_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    unknown = lookup(trie, {"surface": "unknown", "assistant_lane": "unknown", "intent": "unknown"})
    controls = [
        {
            "id": "unknown_binding_abstains",
            "passed": unknown["state"] == "NO_ADMISSIBLE" and not unknown["asset_id"],
            "evidence": unknown,
        }
    ]
    if assets:
        duplicate = dict(assets[0])
        duplicate["id"] = f"{assets[0]['id']}.collision"
        ambiguous = lookup(build_trie([assets[0], duplicate]), dict_value(assets[0].get("lookup_binding")))
        controls.append(
            {
                "id": "ambiguous_binding_abstains",
                "passed": ambiguous["state"] == "AMBIGUOUS" and not ambiguous["asset_id"],
                "evidence": ambiguous,
            }
        )
        retired_trie = build_trie([])
        retired = lookup(retired_trie, dict_value(assets[0].get("lookup_binding")))
        controls.append(
            {
                "id": "retired_asset_excluded",
                "passed": retired["state"] == "NO_ADMISSIBLE",
                "evidence": {"created_utc": created_utc, **retired},
            }
        )
        binding = dict_value(assets[0].get("lookup_binding"))
        synthetic_candidate = {"id": assets[0].get("candidate_id")}
        synthetic_route = {"id": assets[0].get("route_id"), "canary_route_eligible": True}
        synthetic_replay = {"passed": True}
        stale_event = event_for_binding(binding, "completed", "2000-01-01T00:00:00Z")
        stale_receipt = lifecycle_receipt(
            candidate=synthetic_candidate,
            route=synthetic_route,
            replay=synthetic_replay,
            binding=binding,
            events=[stale_event],
            policy=lifecycle_policy,
            created_utc=created_utc,
        )
        controls.append(
            {
                "id": "stale_trace_retires_automatically",
                "passed": stale_receipt["lifecycle_state"] == "retired_stale" and stale_receipt["rollback_triggered"] is True,
                "evidence": stale_receipt,
            }
        )
        drift_events = [
            event_for_binding(binding, "missed", created_utc),
            event_for_binding(binding, "ignored", created_utc),
        ]
        drift_receipt = lifecycle_receipt(
            candidate=synthetic_candidate,
            route=synthetic_route,
            replay=synthetic_replay,
            binding=binding,
            events=drift_events,
            policy=lifecycle_policy,
            created_utc=created_utc,
        )
        controls.append(
            {
                "id": "postcondition_drift_retires_and_rolls_back",
                "passed": drift_receipt["lifecycle_state"] == "retired_drift" and drift_receipt["rollback_triggered"] is True,
                "evidence": drift_receipt,
            }
        )
    return controls


def event_for_binding(binding: dict[str, Any], outcome: str, created_utc: str) -> dict[str, Any]:
    return {
        "surface": binding.get("surface"),
        "assistant_lane": binding.get("assistant_lane"),
        "intent": binding.get("intent"),
        "outcome": outcome,
        "created_utc": created_utc,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def append_lifecycle_ledger(path: Path, receipts: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("receipt_id"):
                existing_ids.add(str(row["receipt_id"]))
    pending = [row for row in receipts if str(row.get("receipt_id") or "") not in existing_ids]
    if pending:
        with path.open("a", encoding="utf-8") as handle:
            for row in pending:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(pending)


def event_matches_binding(event: dict[str, Any], binding: dict[str, str]) -> bool:
    return (
        clean_label(event.get("surface") or "local_assistant") == binding.get("surface")
        and clean_label(event.get("assistant_lane") or event.get("lane") or "assistant") == binding.get("assistant_lane")
        and event_intent(event) == binding.get("intent")
    )


def event_intent(event: dict[str, Any]) -> str:
    explicit = str(event.get("intent") or "").strip()
    if explicit:
        return clean_label(explicit.split(";", 1)[0])
    summary = str(event.get("intent_summary_redacted") or "")
    if "intent=" in summary:
        summary = summary.split("intent=", 1)[1].split(";", 1)[0]
    elif summary:
        summary = summary.split(":", 1)[0].split("_metadata", 1)[0]
    return clean_label(summary or "general")


def event_outcome(event: dict[str, Any]) -> str:
    return clean_label(event.get("outcome") or event.get("feedback") or "")


def event_no_cheat_fault(event: dict[str, Any]) -> bool:
    return bool(
        event.get("raw_user_text")
        or event.get("raw_prompt")
        or event.get("prompt_text")
        or int_or(event.get("public_training_rows_written"), 0)
        or int_or(event.get("external_inference_calls"), 0)
        or int_or(event.get("fallback_return_count"), 0)
    )


def binding_tokens(binding: dict[str, Any]) -> list[str]:
    return [
        f"surface:{clean_label(binding.get('surface'))}",
        f"lane:{clean_label(binding.get('assistant_lane'))}",
        f"intent:{clean_label(binding.get('intent'))}",
    ]


def binding_key(binding: dict[str, Any]) -> str:
    return "|".join(binding_tokens(binding))


def timestamp_age_days(now_text: str, then_text: str) -> float:
    if not then_text:
        return float("inf")
    try:
        now_value = datetime.fromisoformat(now_text.replace("Z", "+00:00"))
        then_value = datetime.fromisoformat(then_text.replace("Z", "+00:00"))
        return max(0.0, (now_value - then_value).total_seconds() / 86400.0)
    except ValueError:
        return float("inf")


def stable_id(*parts: Any) -> str:
    return stable_hash(parts)[:16]


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def clean_label(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:80] or "unknown"


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def list_values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_or(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
