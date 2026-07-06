#!/usr/bin/env python3
"""Report, file IO, and markdown helpers for the neural seed comparator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from neural_seed_code_proposer_comparator import dict_or_empty, get_path  # noqa: E402


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return [row for row in rows if isinstance(row, dict)]


def stable_hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def planned_report(config: dict[str, Any], config_path: str, gap_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": "project_theseus_neural_seed_token_decoder_comparator_report_v0",
        "created_utc": now(),
        "config": config_path,
        "trigger_state": "PLANNED",
        "execute": False,
        "summary": {
            "comparison_level": config.get("comparison_level"),
            "token_decoder_smoke_ready": False,
            "external_inference_calls": 0,
        },
        "gap_report": {
            "trigger_state": gap_report.get("trigger_state"),
            "summary": gap_report.get("summary", {}),
        },
        "adapter_boundary": config.get("adapter_boundary", {}),
        "external_inference_calls": 0,
    }



def token_score_semantics(target_mode: str) -> str:
    if target_mode == "body_tokens":
        target_text = (
            "Both arms train on direct private solution-body tokens and emit decoded Python body-token candidates. "
            "No semantic-slot body renderer, contract semantic beam, structural-action adapter, body-template selector, "
            "or fallback terminal return is credited as evidence."
        )
    elif target_mode == "semantic_slots_v1":
        target_text = (
            "Both arms train on private solution-body AST semantic-slot targets. Semantic-slot rendered bodies are "
            "diagnostic only and are excluded from no-cheat learned-generation evidence."
        )
    elif target_mode == "statement_skeleton_v1":
        target_text = (
            "Both arms train on private statement-skeleton targets. Skeleton-rendered bodies are diagnostic only and "
            "are excluded from no-cheat learned-generation evidence unless a separate gate admits them."
        )
    elif target_mode == "strict_action_tokens_v1":
        target_text = (
            "Both arms train on private strict-action statement targets. Strict-action rendered bodies are diagnostic "
            "syntax/action evidence only and are excluded from no-cheat learned-generation evidence. They do not count "
            "as promotion-grade direct learned full-body Python generation."
        )
    else:
        target_text = f"Both arms train with target_mode={target_mode!r}."
    return (
        "Private token-level code decoder smoke only. "
        + target_text
        + " Inference features are restricted by the config and blind information-flow audit. "
        "Candidates are scored only through the private verifier. Raw syntax, repaired syntax, fallback-return use, "
        "verifier pass, semantic beams, learned routing config, and residual failures are reported separately. "
        "The no-cheat evidence view excludes null baselines, fallback returns, visible-contract semantic priors, "
        "semantic family body renderers, body-template selectors, task-id keyed candidates, public/eval leakage, "
        "candidate-time teacher/external-inference rows, and unexpected promotion-eligible diagnostics. "
        "No public calibration, live teacher call, network fetch, runtime external serving, or model promotion occurred. "
        "If governed teacher-distillation rows are enabled, they are admitted training-time rows only and are reported separately."
    )


def render_gap_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Neural Seed Code-Proposer Gap Report",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- eval_rows: `{summary.get('eval_rows')}`",
        f"- candidate_rows: `{summary.get('candidate_rows')}`",
        f"- gap_counts: `{summary.get('gap_counts')}`",
        f"- sts_repairs: `{summary.get('sts_repairs')}`",
        f"- sts_regressions: `{summary.get('sts_regressions')}`",
        f"- failure_cause_counts: `{summary.get('failure_cause_counts')}`",
        "",
        "## Top Confusions",
        "",
    ]
    for key, count in dict_or_empty(summary.get("top_confusions")).items():
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(["", "## Semantics", "", str(report.get("score_semantics", "")), ""])
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Neural Seed Token Decoder Comparator",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- comparison_level: `{summary.get('comparison_level')}`",
        f"- token_decoder_smoke_ready: `{summary.get('token_decoder_smoke_ready')}`",
        f"- train_rows: `{summary.get('train_rows')}`",
        f"- eval_rows: `{summary.get('eval_rows')}`",
        f"- candidate_rows: `{summary.get('candidate_rows')}`",
        f"- target_mode: `{summary.get('target_mode')}`",
        f"- target_vocab_size: `{summary.get('target_vocab_size')}`",
        f"- parameter_match_delta: `{summary.get('parameter_match_delta')}`",
        f"- best_sts_on_arm_by_verifier_pass_rate: `{summary.get('best_sts_on_arm_by_verifier_pass_rate')}`",
        f"- symliquid_minus_transformer_sts_on_verifier_pass_rate: `{summary.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
        "",
        "## Score Semantics",
        "",
        str(report.get("score_semantics", "")),
        "",
        "## STS-On Token Decoder Results",
        "",
    ]
    for arm_id, arm in dict_or_empty(report.get("arms")).items():
        row = dict_or_empty(arm.get("summary"))
        backend = dict_or_empty(row.get("backend"))
        lines.append(
            f"- `{arm_id}`: verifier_pass_rate=`{row.get('sts_on_verifier_pass_rate')}`, "
            f"accepted_candidate_rate=`{row.get('accepted_candidate_rate')}`, "
            f"syntax_pass_rate=`{row.get('syntax_pass_rate_sts_on')}`, "
            f"raw_syntax_pass_rate=`{row.get('raw_syntax_pass_rate_sts_on')}`, "
            f"repair_changed_rate=`{row.get('grammar_repair_changed_rate_sts_on')}`, "
            f"fallback_rate=`{row.get('grammar_repair_fallback_rate_sts_on')}`, "
            f"statement_skeleton_rate=`{row.get('statement_skeleton_render_rate_sts_on')}`, "
            f"semantic_slot_rate=`{row.get('semantic_slot_render_rate_sts_on')}`, "
            f"semantic_plan_supported_rate=`{row.get('semantic_plan_supported_rate_sts_on')}`, "
            f"predicted_return_shape_rate=`{row.get('predicted_return_shape_rate_sts_on')}`, "
            f"sts_delta=`{row.get('sts_delta')}`, "
            f"regressions=`{row.get('sts_task_level_regressions')}`, "
            f"params=`{row.get('parameter_count')}`, "
            f"backend=`{backend.get('framework')}:{backend.get('device')}`"
        )
    comparisons = dict_or_empty(report.get("comparisons"))
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- winner_by_sts_on_verifier_pass_rate: `{comparisons.get('winner_by_sts_on_verifier_pass_rate')}`",
            f"- symliquid_minus_transformer_sts_on_verifier_pass_rate: `{comparisons.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
            f"- symliquid_gap_vs_body_template: `{summary.get('symliquid_gap_vs_body_template')}`",
            f"- transformer_gap_vs_body_template: `{summary.get('transformer_gap_vs_body_template')}`",
            "",
            "## Gates",
            "",
        ]
    )
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    return "\n".join(lines) + "\n"
