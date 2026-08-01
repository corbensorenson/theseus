#!/usr/bin/env python3
"""Successor route identity with explicit model-context generation custody."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import theseus_assistant_route_integrity as v1
from theseus_assistant_route_integrity import *  # noqa: F401,F403


ROUTE_POLICY = "project_theseus_live_route_integrity_v2"


def load_model_contract(
    worker_config_path: str | Path,
    runtime_preflight_path: str | Path,
    *,
    maximum_tokens: int,
    required_repo_id: str = "",
    required_revision: str = "",
    required_snapshot_manifest_sha256: str = "",
) -> dict[str, Any]:
    """Extend the historical identity only for explicit successor boundaries."""

    contract = v1.load_model_contract(
        worker_config_path,
        runtime_preflight_path,
        maximum_tokens=maximum_tokens,
        required_repo_id=required_repo_id,
        required_revision=required_revision,
        required_snapshot_manifest_sha256=required_snapshot_manifest_sha256,
    )
    worker = v1.read_json(v1.resolve(worker_config_path))
    boundary = v1.as_dict(worker.get("generation_boundary"))
    if "model_declared_context_window_tokens" not in boundary:
        return contract

    faults = list(contract.get("faults") or [])
    card = v1.as_dict(worker.get("model"))
    configured_maximum = int(card.get("maximum_action_tokens") or 0)
    declared_context = int(boundary.get("model_declared_context_window_tokens") or 0)
    if boundary.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if boundary.get("numeric_ceiling_source") != "model_declared_context_window":
        faults.append("generation_numeric_ceiling_not_model_declared_context")
    if declared_context <= 0 or configured_maximum != declared_context:
        faults.append("worker_boundary_not_equal_model_declared_context")
    if boundary.get("ceiling_hit_is_instrument_invalid") is not True:
        faults.append("generation_boundary_hit_not_invalidated")
    normalized = {
        "policy": str(boundary.get("policy") or ""),
        "numeric_ceiling_source": str(boundary.get("numeric_ceiling_source") or ""),
        "model_declared_context_window_tokens": declared_context,
        "project_selected_quality_token_cap": None,
        "ceiling_hit_is_instrument_invalid": boundary.get(
            "ceiling_hit_is_instrument_invalid"
        )
        is True,
    }
    identity = dict(contract.get("identity") or {})
    identity["generation_boundary"] = normalized
    identity.pop("identity_sha256", None)
    identity["identity_sha256"] = v1.stable_hash(identity)
    contract["identity"] = identity
    contract["faults"] = sorted(set(faults))
    contract["ready"] = not contract["faults"]
    return contract
