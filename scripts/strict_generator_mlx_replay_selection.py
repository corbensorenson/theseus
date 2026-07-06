#!/usr/bin/env python3
"""Private replay and heldout row selection for strict MLX generation.

This module selects existing private rows for replay and governed private
heldout splits. It does not generate candidates, inspect eval tests for
generation, use public benchmark data, call teachers, or grant promotion
credit.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from neural_seed_code_proposer_comparator import (  # noqa: E402
    deterministic_sample,
    dict_or_empty,
    get_path,
    load_private_rows,
    resolve,
    stable_hash,
)
from neural_seed_token_decoder_support import target_tokens  # noqa: E402
from neural_seed_visible_source import (  # noqa: E402
    deterministic_family_balanced_sample,
    family_disjoint_split_audit,
    strict_disjoint_family_key,
)


def select_private_train_replay_rows(config: dict[str, Any], *, max_rows: int, tier: str = "any") -> dict[str, Any]:
    """Select private train rows for correctness-label replay.

    This split is for verifier-labeled training signal generation, not heldout
    evaluation or promotion evidence. It copies private train rows into a
    verifier-only eval view because the shared private verifier evaluates rows
    with ``split == "eval"``. Configured family-disjoint holdout families are
    excluded so replay artifacts do not contaminate heldout transfer claims.
    """

    data_cfg = dict_or_empty(config.get("data"))
    budget = dict_or_empty(config.get("matched_budget"))
    seed = int((budget.get("seeds") or [23])[0]) + 23017
    rows_all = load_private_rows(resolve(str(data_cfg.get("train_jsonl") or "")), data_cfg)
    rows_pool, exclusion = exclude_configured_holdout_rows_for_replay(config, rows_all)
    tier_name = str(tier or "any").strip() or "any"
    tier_inventory = private_train_replay_tier_inventory(rows_pool)
    if tier_name != "any":
        tiered_rows = [row for row in rows_pool if private_train_replay_tier_match(row, tier_name)]
    else:
        tiered_rows = rows_pool
    limit = max_rows or min(32, int(data_cfg.get("max_train_rows") or 1024))
    sampled = deterministic_sample(tiered_rows, limit, seed)
    replay_rows: list[dict[str, Any]] = []
    for row in sampled:
        copy = dict(row)
        copy["split"] = "eval"
        copy["replay_source_split"] = str(row.get("split") or "train")
        copy["private_train_replay_row"] = True
        copy["private_train_replay_tier"] = tier_name
        replay_rows.append(copy)
    return {
        "active": bool(replay_rows),
        "split": "private_train_replay",
        "seed": seed,
        "train_rows": [],
        "eval_rows": replay_rows,
        "family_key": exclusion.get("family_key") or "concept_residual_label",
        "split_audit": {
            "policy": "private_train_replay_split_audit_v1",
            "overlap": {},
            "holdout_exclusion": exclusion,
            "tier_selection": {
                "enabled": tier_name != "any",
                "tier": tier_name,
                "policy": "private_train_replay_existing_row_tier_selection_v1",
                "inventory": tier_inventory,
                "rows_after_tier_filter": len(tiered_rows),
                "selected_rows": len(replay_rows),
                "score_semantics": (
                    "Private replay tier selection uses existing private train rows and private "
                    "solution-body structural complexity only to choose replay/training-signal rows. "
                    "Generation still receives only the strict prompt/signature source text; this "
                    "selection is not heldout evidence, not public calibration, and not a learned "
                    "generation claim."
                ),
                "uses_eval_tests_or_solutions_for_generation": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "uses_eval_tests_or_solutions_for_generation": False,
            "uses_public_data": False,
            "score_semantics": (
                "Private train rows are replayed through the verifier solely to produce correctness "
                "labels for future private training. They are not heldout evidence, not public "
                "calibration, and not promotion evidence."
            ),
        },
        "summary": {
            "train_rows": 0,
            "eval_rows": len(replay_rows),
            "source_rows_before_holdout_exclusion": len(rows_all),
            "source_rows_after_holdout_exclusion": len(rows_pool),
            "train_replay_tier": tier_name,
            "source_rows_after_tier_filter": len(tiered_rows),
            "tier_inventory": tier_inventory,
            "family_disjoint_holdout_exclusion": exclusion,
            "evidence_role": "private_train_correctness_replay_not_heldout_eval",
        },
    }


def private_train_replay_tier_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tiers = ["simple_return", "loop_accumulate", "algorithmic_small"]
    counts = {tier: 0 for tier in tiers}
    family_counts: dict[str, int] = {}
    for row in rows:
        family = str(row.get("concept_residual_label") or row.get("category") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        for tier in tiers:
            if private_train_replay_tier_match(row, tier):
                counts[tier] += 1
    return {
        "policy": "private_train_replay_existing_row_tier_inventory_v1",
        "row_count": len(rows),
        "tier_counts": counts,
        "top_families": sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))[:12],
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def private_train_replay_tier_match(row: dict[str, Any], tier: str) -> bool:
    if str(tier or "any") == "any":
        return True
    features = private_solution_body_features(str(row.get("solution_body") or ""))
    if not features.get("parse_ok"):
        return False
    if tier == "simple_return":
        return (
            int(features.get("function_body_statement_count") or 0) <= 3
            and int(features.get("return_count") or 0) >= 1
            and int(features.get("loop_count") or 0) == 0
            and int(features.get("if_count") or 0) <= 1
            and int(features.get("call_count") or 0) <= 3
            and int(features.get("node_count") or 0) <= 40
        )
    if tier == "loop_accumulate":
        return (
            1 <= int(features.get("loop_count") or 0) <= 1
            and int(features.get("return_count") or 0) >= 1
            and int(features.get("function_body_statement_count") or 0) <= 6
            and int(features.get("node_count") or 0) <= 95
            and int(features.get("comprehension_count") or 0) <= 1
        )
    if tier == "algorithmic_small":
        return (
            int(features.get("node_count") or 0) <= 140
            and int(features.get("loop_count") or 0) <= 2
            and int(features.get("if_count") or 0) <= 4
            and int(features.get("return_count") or 0) >= 1
        )
    return False


def private_solution_body_features(body: str) -> dict[str, Any]:
    wrapper = "def __theseus_private_body_probe__(data, other=None, extra=None):\n"
    for line in str(body or "").splitlines() or ["pass"]:
        wrapper += f"    {line}\n" if line.strip() else "\n"
    try:
        tree = ast.parse(wrapper)
    except SyntaxError as exc:
        return {
            "parse_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:200],
        }
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    body_nodes = list(function.body) if function is not None else []
    return {
        "parse_ok": True,
        "function_body_statement_count": len(body_nodes),
        "node_count": sum(1 for _ in ast.walk(function)) if function is not None else 0,
        "loop_count": sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(function)) if function is not None else 0,
        "if_count": sum(isinstance(node, ast.If) for node in ast.walk(function)) if function is not None else 0,
        "return_count": sum(isinstance(node, ast.Return) for node in ast.walk(function)) if function is not None else 0,
        "call_count": sum(isinstance(node, ast.Call) for node in ast.walk(function)) if function is not None else 0,
        "comprehension_count": sum(isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)) for node in ast.walk(function)) if function is not None else 0,
    }


def exclude_configured_holdout_rows_for_replay(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        selection = select_family_disjoint_rows(config, max_rows=1)
    except Exception as exc:
        return [], {
            "enabled": False,
            "clean": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:400],
            "score_semantics": (
                "Private train replay could not resolve family-disjoint holdout families. "
                "It fails closed rather than risking contaminated transfer evidence."
            ),
        }
    if not bool(selection.get("active")):
        return rows, {
            "enabled": False,
            "clean": True,
            "reason": dict_or_empty(selection.get("summary")).get("reason") or "family_disjoint_selection_inactive",
            "excluded_row_count": 0,
            "uses_eval_tests_or_solutions_for_generation": False,
            "uses_public_data": False,
        }
    family_key = str(selection.get("family_key") or "concept_residual_label")
    holdout = {str(value) for value in selection.get("holdout_families") or []}
    if not holdout:
        return [], {
            "enabled": True,
            "clean": False,
            "reason": "missing_holdout_families",
            "family_key": family_key,
            "excluded_row_count": 0,
        }
    kept = [row for row in rows if strict_disjoint_family_key(row, family_key) not in holdout]
    excluded = len(rows) - len(kept)
    return kept, {
        "enabled": True,
        "clean": bool(kept),
        "policy": "exclude_configured_family_disjoint_holdout_families_from_private_train_replay_v1",
        "family_key": family_key,
        "holdout_families": sorted(holdout),
        "input_row_count": len(rows),
        "excluded_row_count": excluded,
        "remaining_row_count": len(kept),
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
    }


def select_family_disjoint_rows(config: dict[str, Any], *, max_rows: int) -> dict[str, Any]:
    data_cfg = dict_or_empty(config.get("data"))
    disjoint_cfg = dict_or_empty(data_cfg.get("family_disjoint_eval"))
    if not bool(disjoint_cfg.get("enabled", False)):
        return {"active": False, "summary": {"reason": "family_disjoint_eval_disabled"}}
    target_mode = str(get_path(config, ["body_structure_decoder", "target_mode"], "body_tokens"))
    budget = dict_or_empty(config.get("matched_budget"))
    seed = int(disjoint_cfg.get("split_seed") or (int((budget.get("seeds") or [23])[0]) + 711))
    train_rows_all = load_private_rows(resolve(str(data_cfg.get("train_jsonl") or "")), data_cfg)
    eval_rows_all = load_private_rows(resolve(str(data_cfg.get("eval_jsonl") or "")), data_cfg)
    min_holdout_families = max(1, int(disjoint_cfg.get("min_holdout_families") or 6))
    family_key_name = str(disjoint_cfg.get("family_key") or "concept_residual_label")
    train_families_all = {strict_disjoint_family_key(row, family_key_name) for row in train_rows_all}
    eval_families_all = {strict_disjoint_family_key(row, family_key_name) for row in eval_rows_all}
    common_families = sorted(family for family in train_families_all & eval_families_all if family and family != "unknown")
    holdout_families = sorted(common_families, key=lambda family: stable_hash(f"{seed}:family_disjoint:{family}"))[:min_holdout_families]
    if len(holdout_families) < min_holdout_families:
        return {
            "active": False,
            "summary": {
                "reason": "insufficient_common_families",
                "common_family_count": len(common_families),
                "required_holdout_family_count": min_holdout_families,
            },
        }
    holdout = set(holdout_families)
    train_pool = [row for row in train_rows_all if strict_disjoint_family_key(row, family_key_name) not in holdout]
    eval_pool = [row for row in eval_rows_all if strict_disjoint_family_key(row, family_key_name) in holdout]
    train_rows = deterministic_sample(train_pool, int(disjoint_cfg.get("max_train_rows") or data_cfg.get("max_train_rows") or 512), seed)
    eval_limit = max_rows or int(disjoint_cfg.get("max_eval_rows") or data_cfg.get("max_eval_rows") or 24)
    eval_rows = deterministic_sample(eval_pool, eval_limit, seed + 1009)
    split_audit = family_disjoint_split_audit(
        train_rows,
        eval_rows,
        holdout_families=holdout_families,
        family_key_name=family_key_name,
        target_mode=target_mode,
    )
    return {
        "active": bool(train_rows and eval_rows),
        "split": "family_disjoint",
        "seed": seed,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "holdout_families": holdout_families,
        "family_key": family_key_name,
        "split_audit": split_audit,
        "summary": {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "holdout_family_count": len(holdout_families),
            "holdout_families": holdout_families,
        },
    }


def select_broad_private_heldout_rows(config: dict[str, Any], *, max_rows: int) -> dict[str, Any]:
    data_cfg = dict_or_empty(config.get("data"))
    broad_cfg = dict_or_empty(data_cfg.get("broad_private_heldout_eval"))
    if not bool(broad_cfg.get("enabled", False)):
        return {"active": False, "summary": {"reason": "broad_private_heldout_eval_disabled"}}
    target_mode = str(get_path(config, ["body_structure_decoder", "target_mode"], "body_tokens"))
    budget = dict_or_empty(config.get("matched_budget"))
    seed = int(broad_cfg.get("split_seed") or (int((budget.get("seeds") or [23])[0]) + 17017))
    train_rows_all = load_private_rows(resolve(str(data_cfg.get("train_jsonl") or "")), data_cfg)
    eval_paths_cfg = broad_cfg.get("eval_jsonl")
    eval_paths = [resolve(str(path)) for path in eval_paths_cfg] if isinstance(eval_paths_cfg, list) else [resolve(str(eval_paths_cfg or ""))]
    eval_data_cfg = dict(data_cfg)
    eval_data_cfg["forbidden_row_flags"] = data_cfg.get("forbidden_row_flags", [])
    eval_rows_unfiltered: list[dict[str, Any]] = []
    seen_eval_keys: set[str] = set()
    for eval_path in eval_paths:
        for row in load_private_rows(eval_path, eval_data_cfg):
            key = str(row.get("task_id") or row.get("source_task_id") or stable_hash(json.dumps(row, sort_keys=True)))
            if key in seen_eval_keys:
                continue
            seen_eval_keys.add(key)
            eval_rows_unfiltered.append(row)
    family_key_name = str(broad_cfg.get("family_key") or "concept_residual_label")
    train_limit = int(broad_cfg.get("max_train_rows") or data_cfg.get("max_train_rows") or 1024)
    train_rows = deterministic_sample(train_rows_all, train_limit, seed)
    train_solution_hashes = {
        stable_hash(str(row.get("solution_body") or "").strip())
        for row in train_rows_all
        if str(row.get("solution_body") or "").strip()
    }
    train_families_all = {strict_disjoint_family_key(row, family_key_name) for row in train_rows_all}
    train_token_hashes = {
        stable_hash(" ".join(target_tokens(str(row.get("solution_body") or ""), target_mode=target_mode)))
        for row in train_rows_all
        if str(row.get("solution_body") or "").strip()
    }
    eval_rows_all = []
    overlap_rejected = 0
    family_overlap_rejected = 0
    for row in eval_rows_unfiltered:
        if strict_disjoint_family_key(row, family_key_name) in train_families_all:
            family_overlap_rejected += 1
            continue
        body = str(row.get("solution_body") or "").strip()
        body_hash = stable_hash(body) if body else ""
        token_hash = stable_hash(" ".join(target_tokens(body, target_mode=target_mode))) if body else ""
        if body_hash in train_solution_hashes or token_hash in train_token_hashes:
            overlap_rejected += 1
            continue
        eval_rows_all.append(row)
    eval_limit = max_rows or int(broad_cfg.get("max_eval_rows") or 200)
    eval_rows = deterministic_family_balanced_sample(
        eval_rows_all,
        eval_limit,
        seed + 1009,
        family_key_name=family_key_name,
    )
    split_audit = family_disjoint_split_audit(
        train_rows,
        eval_rows,
        holdout_families=sorted({strict_disjoint_family_key(row, family_key_name) for row in eval_rows}),
        family_key_name=family_key_name,
        target_mode=target_mode,
    )
    return {
        "active": bool(train_rows and eval_rows),
        "split": "broad_private_heldout",
        "seed": seed,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "family_key": family_key_name,
        "split_audit": split_audit,
        "summary": {
            "train_rows": len(train_rows),
            "eval_rows_before_overlap_filter": len(eval_rows_unfiltered),
            "eval_rows_after_overlap_filter": len(eval_rows_all),
            "eval_rows": len(eval_rows),
            "eval_family_count": int(split_audit.get("eval_family_count") or 0),
            "eval_rows_rejected_for_train_family_overlap": family_overlap_rejected,
            "eval_rows_rejected_for_train_solution_or_token_overlap": overlap_rejected,
        },
    }
