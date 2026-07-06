"""Markdown rendering for train-once Code LM fanout reports."""

from __future__ import annotations

from typing import Any


def render_markdown(payload: dict[str, Any]) -> str:
    summary = _object_field(payload, "summary")
    phase_summary = payload.get("phase_ledger_summary")
    if not isinstance(phase_summary, dict):
        phase_summary = summary.get("phase_ledger_summary") if isinstance(summary.get("phase_ledger_summary"), dict) else {}
    control = payload.get("control_signal_contract")
    if not isinstance(control, dict):
        control = summary.get("control_signal_contract") if isinstance(summary.get("control_signal_contract"), dict) else {}
    lines = [
        "# Code LM Train-Once Fanout",
        "",
        f"- Trigger: `{payload.get('trigger_state')}`",
        f"- Status: `{payload.get('run_status')}`",
        f"- Phase: `{payload.get('current_phase', 'planned')}`",
        f"- Slug: `{payload.get('slug')}`",
        f"- STS mode: `{payload.get('sts_conditioning_mode', summary.get('sts_conditioning_mode', ''))}`",
        f"- STS used: `{summary.get('sts_conditioning_used', False)}`",
        f"- Closure report: `{payload.get('closure_report', '')}`",
        f"- Repeated training per candidate shard: `{summary.get('repeated_training_per_candidate_shard', False)}`",
        f"- Public calibration allowed: `{summary.get('public_calibration_allowed', False)}`",
        f"- Report semantics: `{control.get('semantics', 'control_signal')}`",
        f"- Phase ledger events: `{phase_summary.get('event_count', 0)}`",
    ]
    sts_policy = payload.get("sts_default_policy")
    if not isinstance(sts_policy, dict):
        sts_policy = summary.get("sts_default_policy") if isinstance(summary.get("sts_default_policy"), dict) else {}
    if sts_policy:
        lines.extend(
            [
                "",
                "## STS Default Policy",
                f"- Default: `{sts_policy.get('default', '')}`",
                f"- Disable flag: `{sts_policy.get('disable_flag', '')}`",
                f"- STS-off role: `{sts_policy.get('sts_off_role', '')}`",
                f"- Same-seed non-STS comparator preserved: `{sts_policy.get('same_seed_non_sts_comparator_preserved', False)}`",
            ]
        )
    phases = phase_summary.get("phases") if isinstance(phase_summary.get("phases"), dict) else {}
    if phases:
        lines.extend(["", "## Phase Ledger"])
        for phase, row in phases.items():
            lines.append(
                f"- `{phase}` latest `{row.get('latest_event')}` elapsed `{row.get('elapsed_seconds')}`s consumer `{row.get('consumer')}`"
            )
    targets = summary.get("slow_phase_targets") if isinstance(summary.get("slow_phase_targets"), list) else []
    if targets:
        lines.extend(["", "## Optimizer Targets"])
        for item in targets[:8]:
            lines.append(f"- `{item.get('id')}`: {item.get('recommended_action')}")
    provenance = payload.get("artifact_provenance")
    if not isinstance(provenance, dict):
        provenance = summary.get("artifact_provenance") if isinstance(summary.get("artifact_provenance"), dict) else {}
    if provenance:
        prov_summary = provenance.get("summary") if isinstance(provenance.get("summary"), dict) else {}
        lines.extend(
            [
                "",
                "## Artifact Provenance",
                f"- Fanout fresh: `{prov_summary.get('fanout_fresh')}`",
                f"- Checkpoint sha256: `{str(prov_summary.get('checkpoint_sha256', ''))[:16]}`",
                f"- Release binary sha256: `{str(prov_summary.get('release_binary_sha256', ''))[:16]}`",
                f"- Source fingerprint: `{str(prov_summary.get('source_combined_sha256', ''))[:16]}`",
            ]
        )
    categories = summary.get("phase_timing_categories") if isinstance(summary.get("phase_timing_categories"), dict) else {}
    if categories:
        lines.extend(["", "## Phase Timing Categories"])
        for name in ["candidate_expansion", "sts_conditioning", "ranker_prefilter", "verifier_cache", "artifact_write"]:
            row = categories.get(name) if isinstance(categories.get(name), dict) else {}
            lines.append(f"- `{name}` private `{row.get('private_ms', 0)}`ms public `{row.get('public_ms', 0)}`ms")
    verification = payload.get("staged_verification_contract") or summary.get("staged_verification_contract") or []
    if isinstance(verification, list) and verification:
        lines.extend(["", "## Staged Verification Contract"])
        for row in verification:
            if isinstance(row, dict):
                lines.append(f"- `{row.get('stage')}` -> `{row.get('pass_signal')}` fail_fast=`{row.get('fail_fast')}`")
    next_actions = payload.get("next_actions") or []
    if next_actions:
        lines.extend(["", "## Next Actions"])
        lines.extend(f"- {item}" for item in next_actions)
    return "\n".join(lines) + "\n"


def _object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}
