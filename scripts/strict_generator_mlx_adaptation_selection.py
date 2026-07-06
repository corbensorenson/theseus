#!/usr/bin/env python3
"""Private-row selection and guard weighting for strict MLX adaptation.

These helpers audit admitted private target bodies and select existing private
training rows for adaptation. They do not generate candidates, inspect eval
verifier tests, use public benchmark data, call teachers, or grant learned-code
capability credit.
"""

from __future__ import annotations

import json
from typing import Any

from neural_seed_code_proposer_comparator import deterministic_sample, stable_hash  # noqa: E402
from neural_seed_decode_static_guard import decode_static_guard  # noqa: E402
from strict_generator_mlx_replay_selection import (  # noqa: E402
    private_train_replay_tier_inventory,
    private_train_replay_tier_match,
)
from strict_generator_mlx_pretraining_probe import allowed_parameter_names_from_source_text  # noqa: E402


def strict_target_guard_rows(
    bodies: list[str],
    source_texts: list[str],
    *,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    pass_count = 0
    for index, body in enumerate(bodies):
        source_text = source_texts[index] if index < len(source_texts) else ""
        allowed = allowed_parameter_names_from_source_text(source_text)
        if not allowed:
            allowed = {"data"}
        guard = decode_static_guard(
            str(body or ""),
            allowed_names=allowed,
            require_parameter_use=True,
            require_nontrivial_return=True,
            require_top_level_return=True,
        )
        passed = bool(guard.get("passed"))
        if passed:
            pass_count += 1
        for failure in list(guard.get("failures") or []):
            key = str(failure)
            failure_counts[key] = failure_counts.get(key, 0) + 1
        rows.append(
            {
                "index": index,
                "passed": passed,
                "failures": [str(item) for item in list(guard.get("failures") or [])],
                "allowed_parameter_count": len(allowed),
                "dependency": guard.get("dependency"),
                "definite_assignment": guard.get("definite_assignment"),
                "control_flow_pathology": guard.get("control_flow_pathology"),
            }
        )
    total = len(rows)
    examples = [
        {
            "index": row["index"],
            "failures": row["failures"],
            "dependency": row.get("dependency"),
            "definite_assignment": row.get("definite_assignment"),
            "control_flow_pathology": row.get("control_flow_pathology"),
        }
        for row in rows
        if not bool(row.get("passed"))
    ][:8]
    return rows, {
        "enabled": True,
        "policy": "private_target_body_strict_decode_guard_audit_v1",
        "split": split_name,
        "rows": total,
        "passed": pass_count,
        "failed": total - pass_count,
        "pass_rate": round(pass_count / max(1, total), 6),
        "failure_counts": dict(sorted(failure_counts.items())),
        "failure_examples": examples,
        "score_semantics": (
            "Audits admitted private target bodies against the same task-blind strict decode guard "
            "used for MLX candidate admission. The guard sees only target body text and visible "
            "callable parameter names from source text; it does not inspect tests, eval solutions, "
            "public benchmarks, verifier results, or answer metadata."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def apply_guard_clean_target_weights(
    token_weight_rows: list[list[float]],
    target_guard_rows: list[dict[str, Any]],
    *,
    guard_clean_boost: float,
    rejected_weight: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    clean_boost = max(0.0, float(guard_clean_boost if guard_clean_boost is not None else 1.0))
    rejected = max(0.0, float(rejected_weight if rejected_weight is not None else 1.0))
    if clean_boost == 1.0 and rejected == 1.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "private_target_body_strict_decode_guard_loss_weighting_v1",
            "reason": "identity_weights",
            "guard_clean_target_loss_boost": clean_boost,
            "guard_rejected_target_loss_weight": rejected,
            "rows": len(token_weight_rows),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    adjusted: list[list[float]] = []
    clean_rows = 0
    rejected_rows = 0
    weighted_token_count = 0
    for index, row in enumerate(token_weight_rows):
        guard = target_guard_rows[index] if index < len(target_guard_rows) else {}
        multiplier = clean_boost if bool(guard.get("passed")) else rejected
        if bool(guard.get("passed")):
            clean_rows += 1
        else:
            rejected_rows += 1
        weighted_row = [float(value) * multiplier for value in row]
        if multiplier != 1.0:
            weighted_token_count += len(weighted_row)
        adjusted.append(weighted_row)
    return adjusted, {
        "enabled": True,
        "policy": "private_target_body_strict_decode_guard_loss_weighting_v1",
        "guard_clean_target_loss_boost": clean_boost,
        "guard_rejected_target_loss_weight": rejected,
        "rows": len(token_weight_rows),
        "guard_clean_rows": clean_rows,
        "guard_rejected_rows": rejected_rows,
        "weighted_token_count": weighted_token_count,
        "score_semantics": (
            "Optional supervised loss weighting over admitted private target rows, derived only from "
            "task-blind strict-guard audit of the private target body and visible callable parameter "
            "names. It does not synthesize code, inspect tests/solutions, use public data, alter "
            "verifier scoring, or grant candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def select_private_train_tier_rows(rows: list[dict[str, Any]], *, tier: str) -> dict[str, Any]:
    tier_name = str(tier or "any").strip() or "any"
    inventory = private_train_replay_tier_inventory(rows)
    if tier_name == "any":
        selected = list(rows)
    else:
        selected = [row for row in rows if private_train_replay_tier_match(row, tier_name)]
    return {
        "enabled": tier_name != "any",
        "tier": tier_name,
        "policy": "private_adaptation_existing_row_tier_selection_v1",
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "inventory": inventory,
        "rows": selected,
        "score_semantics": (
            "Selects existing private train rows by private solution-body structural complexity before "
            "sampling adaptation rows. This affects only which admitted private targets are used for "
            "training/eval loss. Generation source text remains strict prompt/signature-only, and this "
            "is not heldout transfer evidence, public calibration, or learned-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def tier_balanced_private_train_sample(rows: list[dict[str, Any]], *, limit: int, seed: int) -> dict[str, Any]:
    requested = max(0, int(limit or 0))
    tier_names = ["simple_return", "loop_accumulate", "algorithmic_small"]
    buckets: dict[str, list[dict[str, Any]]] = {
        tier_name: [row for row in rows if private_train_replay_tier_match(row, tier_name)]
        for tier_name in tier_names
    }
    active_tiers = [tier_name for tier_name in tier_names if buckets.get(tier_name)]
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    tier_counts: dict[str, int] = {tier_name: 0 for tier_name in tier_names}
    target_per_tier = max(1, requested // max(1, len(active_tiers))) if requested else 0
    remainder_budget = max(0, requested - (target_per_tier * max(1, len(active_tiers))))

    for tier_index, tier_name in enumerate(active_tiers):
        bucket = buckets[tier_name]
        quota = min(len(bucket), target_per_tier + (1 if tier_index < remainder_budget else 0))
        sampled = deterministic_sample(bucket, quota, seed + 1009 * (tier_index + 1))
        for row in sampled:
            key = private_train_row_key(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            tier_counts[tier_name] += 1

    if len(selected) < requested:
        remaining = [row for row in rows if private_train_row_key(row) not in selected_keys]
        fill = deterministic_sample(remaining, requested - len(selected), seed + 7919)
        for row in fill:
            key = private_train_row_key(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            matched = [tier_name for tier_name in tier_names if private_train_replay_tier_match(row, tier_name)]
            tier_counts[matched[0] if matched else "unclassified"] = tier_counts.get(matched[0] if matched else "unclassified", 0) + 1

    selected.sort(key=lambda row: stable_hash(f"tier-balanced:{seed}:{private_train_row_key(row)}"))
    return {
        "enabled": True,
        "policy": "private_adaptation_existing_row_tier_balanced_sampling_v1",
        "requested_rows": requested,
        "input_rows": len(rows),
        "active_tiers": active_tiers,
        "target_per_active_tier": target_per_tier,
        "selected_rows": len(selected),
        "tier_counts": tier_counts,
        "rows": selected,
        "score_semantics": (
            "Deterministically samples admitted private training rows across existing structural replay tiers "
            "before train/eval splitting so one abundant family cannot dominate the learned prefix/body update. "
            "This does not create synthetic rows, inspect public data, inspect eval tests/solutions, or grant "
            "candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def private_train_row_key(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return stable_hash(repr(row))
    preferred = row.get("task_id") or row.get("source_task_id") or row.get("entry_point")
    if preferred:
        return str(preferred)
    return stable_hash(json.dumps(row, sort_keys=True, default=str))


def private_train_balanced_sample_summary(selection: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in selection.items() if key != "rows"}


def private_train_tier_summary(selection: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in selection.items() if key != "rows"}


def private_train_tier_vocab_summary(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(selection.get("enabled")),
        "tier": selection.get("tier"),
        "policy": selection.get("policy"),
        "input_rows": int(selection.get("input_rows") or 0),
        "selected_rows": int(selection.get("selected_rows") or 0),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }
