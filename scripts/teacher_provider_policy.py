#!/usr/bin/env python3
"""Fail-closed provider policy for governed teacher calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HARD_ALLOWED_PROVIDERS = frozenset({"openai", "chatgpt", "codex", "codex_cli"})
HARD_ALLOWED_MODEL_PREFIXES = ("gpt-", "chatgpt", "codex")
HARD_ALLOWED_LIVE_MODELS = frozenset({"gpt-5.6-sol"})
HARD_ALLOWED_LIVE_REASONING_EFFORTS = frozenset({"medium", "high"})
HARD_FORBIDDEN_MARKERS = frozenset({"anthropic", "claude", "haiku", "opus", "sonnet"})
CODEX_EXECUTABLE_NAMES = frozenset({"codex", "codex.exe"})
IDENTITY_FIELD_MARKERS = (
    "provider",
    "vendor",
    "model",
    "executable",
    "command",
    "endpoint",
    "api_base",
)


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
    model = decision["model"]
    reasoning_effort = str(teacher_policy.get("reasoning_effort") or "").strip().lower()
    approved_live_models = {
        str(value).strip().lower()
        for value in provider_policy.get("provider_policy", {}).get("approved_live_models", [])
    }
    allowed_reasoning_efforts = {
        str(value).strip().lower()
        for value in provider_policy.get("provider_policy", {}).get(
            "allowed_live_reasoning_efforts", []
        )
    }
    configured_command = str(teacher_policy.get("codex_command") or "").strip()
    command_name = configured_command.lower()
    if provider != "codex_cli":
        reject_reasons.append("teacher_oracle_requires_codex_cli_provider")
    if command_name not in CODEX_EXECUTABLE_NAMES:
        reject_reasons.append("teacher_oracle_requires_codex_executable")
    if (
        not approved_live_models
        or model not in approved_live_models
        or model not in HARD_ALLOWED_LIVE_MODELS
    ):
        reject_reasons.append("live_teacher_model_not_approved")
    if (
        not allowed_reasoning_efforts
        or reasoning_effort not in allowed_reasoning_efforts
        or reasoning_effort not in HARD_ALLOWED_LIVE_REASONING_EFFORTS
    ):
        reject_reasons.append("live_teacher_reasoning_effort_not_approved")
    return {
        **decision,
        "accepted": not reject_reasons,
        "configured_command": configured_command,
        "reasoning_effort": reasoning_effort,
        "reject_reasons": sorted(set(reject_reasons)),
    }


def teacher_receipt_decision(
    provider_policy: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    """Verify that a retained external call proves the declared OpenAI route.

    Provider and model labels are not provenance. A real external call must
    retain the exact executable argv and the model selected by that argv.
    Zero-inference local fixtures may omit argv, but cannot claim a teacher
    call was made.
    """

    decision = teacher_provider_decision(provider_policy, row)
    reject_reasons = list(decision["reject_reasons"])
    external_inference_calls = _external_inference_calls(row)
    command_value = row.get("command")
    command = [str(value) for value in command_value] if isinstance(command_value, list) else []
    command_sha256 = (
        hashlib.sha256(json.dumps(command, separators=(",", ":")).encode("utf-8")).hexdigest()
        if command
        else ""
    )

    if external_inference_calls > 0:
        if decision["provider"] != "codex_cli":
            reject_reasons.append("external_teacher_receipt_requires_codex_cli_provider")
        if not command:
            reject_reasons.append("external_teacher_receipt_command_missing")
        else:
            executable = Path(command[0]).name.lower()
            if executable not in CODEX_EXECUTABLE_NAMES:
                reject_reasons.append("external_teacher_receipt_executable_not_codex")
            if len(command) < 2 or command[1] != "exec":
                reject_reasons.append("external_teacher_receipt_not_codex_exec")
            command_model = _command_option(command, "-m")
            if not command_model:
                reject_reasons.append("external_teacher_receipt_model_argument_missing")
            elif command_model.strip().lower() != decision["model"]:
                reject_reasons.append("external_teacher_receipt_model_mismatch")
            combined_command = " ".join(command).lower()
            if any(marker in combined_command for marker in HARD_FORBIDDEN_MARKERS):
                reject_reasons.append("forbidden_teacher_executable_or_argument")
    elif command:
        combined_command = " ".join(command).lower()
        if any(marker in combined_command for marker in HARD_FORBIDDEN_MARKERS):
            reject_reasons.append("forbidden_teacher_executable_or_argument")

    forbidden_identity_values = _forbidden_identity_values(row)
    if forbidden_identity_values:
        reject_reasons.append("forbidden_teacher_identity_in_receipt_provenance")

    return {
        **decision,
        "accepted": not reject_reasons,
        "external_inference_calls": external_inference_calls,
        "command_retained": bool(command),
        "command_sha256": command_sha256,
        "executable": Path(command[0]).name if command else "",
        "command_model": _command_option(command, "-m") if command else "",
        "forbidden_identity_values": forbidden_identity_values,
        "reject_reasons": sorted(set(reject_reasons)),
    }


def _external_inference_calls(row: dict[str, Any]) -> int:
    recorded = row.get("external_inference_calls")
    if recorded is not None:
        try:
            return max(0, int(recorded))
        except (TypeError, ValueError):
            return 0
    if row.get("status") == "completed" and str(row.get("provider") or "") == "codex_cli":
        return 1
    return 0


def _command_option(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError:
        return ""
    return command[index + 1] if index + 1 < len(command) else ""


def _forbidden_identity_values(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Find forbidden vendors/models in identity-bearing receipt metadata.

    Response and prompt text are intentionally excluded: a valid OpenAI teacher
    may discuss another provider. Provider provenance may not be relabelled or
    hidden in a nested receipt, usage, or transport object.
    """

    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).strip().lower()
            child_path = (*path, key_text)
            identity_field = any(marker in key_text for marker in IDENTITY_FIELD_MARKERS)
            if identity_field and isinstance(child, (str, int, float, bool)):
                child_text = str(child).strip().lower()
                if any(marker in child_text for marker in HARD_FORBIDDEN_MARKERS):
                    findings.append(f"{'.'.join(child_path)}={child_text[:120]}")
            elif identity_field and isinstance(child, list):
                child_text = " ".join(str(item) for item in child).strip().lower()
                if any(marker in child_text for marker in HARD_FORBIDDEN_MARKERS):
                    findings.append(f"{'.'.join(child_path)}={child_text[:120]}")
            if isinstance(child, (dict, list)):
                findings.extend(_forbidden_identity_values(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                findings.extend(_forbidden_identity_values(child, (*path, str(index))))
    return sorted(set(findings))
