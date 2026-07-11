#!/usr/bin/env python3
"""Fail-closed provider policy for governed teacher calls."""

from __future__ import annotations

from typing import Any


HARD_ALLOWED_PROVIDERS = frozenset({"openai", "chatgpt", "codex", "codex_cli"})
HARD_ALLOWED_MODEL_PREFIXES = ("gpt-", "chatgpt", "codex")
HARD_FORBIDDEN_MARKERS = frozenset({"anthropic", "claude", "haiku", "opus", "sonnet"})
CODEX_EXECUTABLE_NAMES = frozenset({"codex", "codex.exe"})


def teacher_provider_decision(policy: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Validate a teacher identity against config and immutable project limits."""

    provider_policy = policy.get("provider_policy") if isinstance(policy.get("provider_policy"), dict) else {}
    provider = str(row.get("provider") or "").strip().lower()
    model = str(row.get("model") or "").strip().lower()
    configured_providers = {
        str(value).strip().lower() for value in provider_policy.get("allowed_providers", [])
    }
    configured_prefixes = tuple(
        str(value).strip().lower() for value in provider_policy.get("allowed_model_prefixes", [])
    )
    configured_forbidden = {
        str(value).strip().lower() for value in provider_policy.get("forbidden_markers", [])
    }
    reject_reasons: list[str] = []
    if provider_policy.get("fail_closed") is not True:
        reject_reasons.append("teacher_provider_policy_not_fail_closed")
    if provider_policy.get("require_explicit_provider") is not True or not provider:
        reject_reasons.append("teacher_provider_missing")
    if provider_policy.get("require_explicit_model") is not True or not model:
        reject_reasons.append("teacher_model_missing")
    if provider not in configured_providers or provider not in HARD_ALLOWED_PROVIDERS:
        reject_reasons.append("teacher_provider_not_approved")
    if (
        not configured_prefixes
        or not model.startswith(configured_prefixes)
        or not model.startswith(HARD_ALLOWED_MODEL_PREFIXES)
    ):
        reject_reasons.append("teacher_model_not_approved")
    combined = f"{provider} {model}"
    forbidden_markers = configured_forbidden | HARD_FORBIDDEN_MARKERS
    if any(marker and marker in combined for marker in forbidden_markers):
        reject_reasons.append("forbidden_teacher_provider_or_model")
    return {
        "accepted": not reject_reasons,
        "provider": provider,
        "model": model,
        "reject_reasons": sorted(set(reject_reasons)),
    }


def teacher_launch_decision(
    provider_policy: dict[str, Any],
    teacher_policy: dict[str, Any],
) -> dict[str, Any]:
    """Validate identity and executable before an external process can start."""

    decision = teacher_provider_decision(
        provider_policy,
        {"provider": teacher_policy.get("provider"), "model": teacher_policy.get("model")},
    )
    reject_reasons = list(decision["reject_reasons"])
    provider = decision["provider"]
    configured_command = str(teacher_policy.get("codex_command") or "").strip()
    command_name = configured_command.lower()
    if provider != "codex_cli":
        reject_reasons.append("teacher_oracle_requires_codex_cli_provider")
    if command_name not in CODEX_EXECUTABLE_NAMES:
        reject_reasons.append("teacher_oracle_requires_codex_executable")
    return {
        **decision,
        "accepted": not reject_reasons,
        "configured_command": configured_command,
        "reject_reasons": sorted(set(reject_reasons)),
    }
