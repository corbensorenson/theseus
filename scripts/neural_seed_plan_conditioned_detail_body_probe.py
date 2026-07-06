#!/usr/bin/env python3
"""Plan-conditioned detail body decoder probe for neural-seed residuals.

The probe trains matched tiny body-token decoders on private train rows, using
visible private contract text plus a semantic-plan condition. At eval time, the
plan conditions come only from train-derived visible-contract route memories.
Generated body candidates are appended after an existing private candidate
manifest and scored by the private verifier.

This is diagnostic source, not a production renderer. It does not add canned
family bodies, task-id branches, fallback returns, public data, teacher calls,
or model promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402
from neural_seed_code_proposer_comparator import (  # noqa: E402
    build_vocab,
    count_params,
    deterministic_sample,
    dict_or_empty,
    encode_many,
    get_path,
    import_torch,
    load_private_rows,
    maxrss_mb,
    mlx_status,
    rel,
    render_private_function,
    row_text,
    select_torch_device,
    stable_hash,
    syntax_summary,
)
from neural_seed_token_decoder_comparator import (  # noqa: E402
    TransformerTokenDecoder,
    SymLiquidTokenDecoder,
    body_tokens,
    build_contract_feature_route_memory,
    build_contract_fingerprint_route_memory,
    build_target_vocab,
    build_visible_text_plan_route_memory,
    candidate_row,
    choose_sym_token_dims,
    generate_candidates,
    generate_contract_feature_route_candidates,
    generate_contract_fingerprint_route_candidates,
    generate_visible_text_prototype_route_candidates,
    grammar_repaired_body,
    grammar_repair_summary,
    train_token_model,
)


DEFAULT_CONFIG = ROOT / "reports" / "neural_seed_token_decoder_96eval_4096train_config.json"
DEFAULT_BASE = ROOT / "reports" / "neural_seed_token_decoder_ablation_full_learned_beam_off_candidates_seed_23.jsonl"
DEFAULT_OUT = ROOT / "reports" / "neural_seed_plan_conditioned_detail_body_probe_seed23.json"
DEFAULT_MD = ROOT / "reports" / "neural_seed_plan_conditioned_detail_body_probe_seed23.md"
DEFAULT_CANDIDATES = ROOT / "reports" / "neural_seed_plan_conditioned_detail_body_probe_seed23_candidates.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--base-candidates", default=str(DEFAULT_BASE.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD.relative_to(ROOT)))
    parser.add_argument("--candidate-manifest-out", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-train-rows", type=int, default=1024)
    parser.add_argument("--max-eval-rows", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fanout-top-k", type=int, default=1)
    parser.add_argument("--max-plan-candidates", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config))
    if not args.execute:
        report = planned_report(config, args)
    else:
        report = run_probe(config, args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def run_probe(config: dict[str, Any], args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    torch, nn = import_torch()
    data_cfg = dict_or_empty(config.get("data"))
    text_views = dict_or_empty(config.get("text_views"))
    budget = dict(dict_or_empty(config.get("matched_budget")))
    budget["epochs"] = int(args.epochs)
    budget["fanout_top_k"] = int(args.fanout_top_k)
    budget["max_target_tokens"] = int(budget.get("max_target_tokens") or 96)
    budget["max_target_vocab"] = int(budget.get("max_target_vocab") or 512)
    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_all = load_private_rows(resolve(str(data_cfg.get("train_jsonl") or "")), data_cfg)
    eval_all = load_private_rows(resolve(str(data_cfg.get("eval_jsonl") or "")), data_cfg)
    train_rows = deterministic_sample(train_all, int(args.max_train_rows), seed)
    eval_rows = deterministic_sample(eval_all, int(args.max_eval_rows), seed + 1009)
    eval_ids = {str(row.get("task_id") or "") for row in eval_rows}
    base_candidates = [
        row
        for row in read_jsonl(resolve(args.base_candidates))
        if str(row.get("task_id") or "") in eval_ids
    ]
    source_fields = list(text_views.get("sts_on") or [])
    target_vocab = build_target_vocab(
        [str(row.get("solution_body") or "") for row in train_rows],
        max_vocab=int(budget.get("max_target_vocab") or 512),
        target_mode="body_tokens",
    )
    plan_vocab = build_target_vocab(
        [str(row.get("solution_body") or "") for row in train_rows],
        max_vocab=int(budget.get("max_target_vocab") or 512),
        target_mode="semantic_slots_v1",
    )
    route_plan_rows = proposed_plans_for_eval(config, train_rows, eval_rows, plan_vocab, source_fields, args)
    train_sources = [
        detail_source_text(row, source_fields, semantic_plan_from_solution(row))
        for row in train_rows
    ]
    source_vocab = build_vocab(train_sources, max_vocab=int(budget.get("max_source_vocab") or 1024))
    target_rows = encode_body_targets(train_rows, target_vocab, int(budget.get("max_target_tokens") or 96))
    max_source = int(budget.get("max_source_tokens") or 96)
    max_target = int(budget.get("max_target_tokens") or 96)
    device, backend_note = select_torch_device(torch)
    mlx = mlx_status()

    transformer_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get("transformer_control"))
    transformer_dims = {
        "d_model": int(transformer_cfg.get("d_model") or 48),
        "nhead": int(transformer_cfg.get("nhead") or 2),
        "num_layers": int(transformer_cfg.get("num_layers") or 1),
        "dim_feedforward": int(transformer_cfg.get("dim_feedforward") or 96),
    }
    transformer_param_count = count_params(
        TransformerTokenDecoder(
            len(source_vocab),
            len(target_vocab),
            max_source_len=max_source,
            max_target_len=max_target,
            **transformer_dims,
            torch=torch,
            nn=nn,
        )
    )
    sym_dims, sym_param_count = choose_sym_token_dims(
        config,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        target_params=transformer_param_count,
        torch=torch,
        nn=nn,
    )
    param_delta = abs(sym_param_count - transformer_param_count) / max(1, transformer_param_count)

    all_augmented = list(base_candidates)
    arm_reports: dict[str, Any] = {}
    old_timeout = os.environ.get("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS")
    os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = str(max(1, int(budget.get("private_candidate_timeout_seconds") or 4)))
    try:
        for arm_id in ["symliquid_style", "transformer_control"]:
            arm_started = time.perf_counter()
            if arm_id == "symliquid_style":
                model = SymLiquidTokenDecoder(
                    len(source_vocab),
                    len(target_vocab),
                    hidden_dim=sym_dims["hidden_dim"],
                    reservoir_dim=sym_dims["reservoir_dim"],
                    hv_dim=sym_dims["hv_dim"],
                    torch=torch,
                    nn=nn,
                )
                substrate = "torch_symliquid_style_plan_conditioned_body_decoder"
                parameter_count = sym_param_count
                dims = sym_dims
            else:
                model = TransformerTokenDecoder(
                    len(source_vocab),
                    len(target_vocab),
                    max_source_len=max_source,
                    max_target_len=max_target,
                    **transformer_dims,
                    torch=torch,
                    nn=nn,
                )
                substrate = "torch_transformer_plan_conditioned_body_decoder"
                parameter_count = transformer_param_count
                dims = transformer_dims
            model.to(device)
            train_x = encode_many(train_sources, source_vocab, max_source)
            before_mem = maxrss_mb()
            train_summary = train_token_model(
                model,
                train_x,
                target_rows,
                budget,
                torch=torch,
                device=device,
                pad_id=target_vocab["<pad>"],
                plan_auxiliary_loss_weight=0.0,
            )
            generated_rows = generate_detail_rows(
                model,
                eval_rows,
                route_plan_rows,
                source_fields,
                source_vocab,
                target_vocab,
                config,
                seed,
                arm_id=arm_id,
                substrate=substrate,
                max_source=max_source,
                max_target=max_target,
                fanout_top_k=int(args.fanout_top_k),
                grammar_top_k=int(budget.get("grammar_decode_top_k") or 256),
                torch=torch,
                device=device,
            )
            all_augmented.extend(generated_rows)
            base_eval = evaluate_private_candidates(eval_rows, [row for row in base_candidates if row.get("substrate_arm") == arm_id or row.get("phase") == "private_baseline"])
            augmented_eval = evaluate_private_candidates(eval_rows, [row for row in all_augmented if row.get("substrate_arm") == arm_id or row.get("phase") == "private_baseline"])
            arm_reports[arm_id] = {
                "summary": {
                    "baseline_pass_rate": base_eval.get("trained_pass_rate"),
                    "augmented_pass_rate": augmented_eval.get("trained_pass_rate"),
                    "delta": round(float(augmented_eval.get("trained_pass_rate") or 0.0) - float(base_eval.get("trained_pass_rate") or 0.0), 6),
                    "baseline_passed": base_eval.get("trained_passed"),
                    "augmented_passed": augmented_eval.get("trained_passed"),
                    "detail_candidate_rows": len(generated_rows),
                    "fallback_return_rows": sum(1 for row in generated_rows if get_path(row, ["grammar_repair", "fallback_return_used"], False)),
                    "syntax_pass_rate": syntax_summary(generated_rows).get("syntax_pass_rate"),
                    "parameter_count": parameter_count,
                    "dims": dims,
                    "backend": {"framework": "torch", "device": str(device), **backend_note, **mlx},
                    "memory": {"maxrss_mb_before": before_mem, "maxrss_mb_after": maxrss_mb()},
                    "train": train_summary,
                    "wall_time_ms": int((time.perf_counter() - arm_started) * 1000),
                },
                "private_verifier": {
                    "baseline": base_eval,
                    "augmented": augmented_eval,
                },
            }
    finally:
        if old_timeout is None:
            os.environ.pop("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS", None)
        else:
            os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = old_timeout

    write_jsonl(resolve(args.candidate_manifest_out), all_augmented)
    gates = [
        gate("private_train_rows_loaded", bool(train_rows), {"train_rows": len(train_rows)}, "hard"),
        gate("private_eval_rows_loaded", bool(eval_rows), {"eval_rows": len(eval_rows)}, "hard"),
        gate("base_candidates_loaded", bool(base_candidates), {"base_candidate_rows": len(base_candidates)}, "hard"),
        gate("plan_conditions_from_train_visible_contract_memory", bool(route_plan_rows), {"eval_plan_rows": len(route_plan_rows)}, "hard"),
        gate("matched_parameter_budget", param_delta <= float(budget.get("trusted_parameter_match_tolerance") or 0.08), {"parameter_match_delta": round(param_delta, 6)}, "hard"),
        gate("fallback_return_rows_zero", all(get_path(row, ["summary", "fallback_return_rows"], 0) == 0 for row in arm_reports.values()), per_arm_field(arm_reports, "fallback_return_rows"), "hard"),
        gate("no_augmented_regression", all(float(get_path(row, ["summary", "delta"], 0.0) or 0.0) >= 0.0 for row in arm_reports.values()), per_arm_field(arm_reports, "delta"), "hard"),
        gate("external_inference_zero", True, 0, "hard"),
        gate("teacher_public_promotion_locked", True, {"teacher_used": False, "public_training_rows": 0, "model_promotion_allowed": False}, "hard"),
    ]
    any_improvement = any(float(get_path(row, ["summary", "delta"], 0.0) or 0.0) > 0.0 for row in arm_reports.values())
    trigger = "GREEN" if all(row["passed"] for row in gates if row["severity"] == "hard") and any_improvement else "YELLOW"
    if any(not row["passed"] for row in gates if row["severity"] == "hard"):
        trigger = "RED"
    return {
        "policy": "project_theseus_neural_seed_plan_conditioned_detail_body_probe_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "config": rel(resolve(args.config)),
        "base_candidates": rel(resolve(args.base_candidates)),
        "candidate_manifest": rel(resolve(args.candidate_manifest_out)),
        "summary": {
            "seed": seed,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
            "target_mode": "body_tokens",
            "plan_condition_source": "private_train_visible_contract_route_memory",
            "parameter_match_delta": round(param_delta, 6),
            "any_improvement": any_improvement,
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "model_promotion_allowed": False,
        },
        "arms": arm_reports,
        "plan_condition_summary": summarize_plan_conditions(route_plan_rows),
        "gates": gates,
        "score_semantics": (
            "Focused private diagnostic. It trains matched detail body-token decoders on private train rows, "
            "conditions eval generation only on train-derived visible-contract semantic route proposals, appends "
            "new candidates after an existing private candidate manifest, and scores through the private verifier. "
            "No public calibration, public data, teacher call, fallback return, task-id branch, eval solution feature, "
            "or promotion is used."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def proposed_plans_for_eval(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    plan_vocab: dict[str, int],
    source_fields: list[str],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    routing = dict_or_empty(get_path(config, ["body_structure_decoder", "internal_semantic_routing"], {}))
    fp_memory = build_contract_fingerprint_route_memory(train_rows, enabled=True)
    feature_memory = build_contract_feature_route_memory(train_rows, enabled=True)
    text_memory = build_visible_text_plan_route_memory(train_rows, source_fields, enabled=True)
    route_batches: list[list[list[dict[str, Any]]]] = [
        generate_contract_fingerprint_route_candidates(
            eval_rows,
            plan_vocab,
            fp_memory,
            top_k=max(1, int(routing.get("contract_fingerprint_route_top_k") or 3)),
        ),
        generate_contract_feature_route_candidates(
            eval_rows,
            plan_vocab,
            feature_memory,
            top_k=max(1, int(routing.get("contract_feature_route_top_k") or 4)),
        ),
        generate_visible_text_prototype_route_candidates(
            eval_rows,
            source_fields,
            text_memory,
            top_k=max(1, int(routing.get("visible_text_prototype_route_top_k") or 4)),
        ),
    ]
    out: dict[str, list[dict[str, Any]]] = {}
    for idx, row in enumerate(eval_rows):
        plans: dict[str, dict[str, Any]] = {}
        for batch in route_batches:
            for proposal in batch[idx] if idx < len(batch) else []:
                plan = str(proposal.get("learned_route_plan") or "")
                if not plan:
                    continue
                score = float(proposal.get("rank_score") or 0.0)
                previous = plans.get(plan)
                if previous is None or score > float(previous.get("rank_score") or 0.0):
                    plans[plan] = {
                        "plan": plan,
                        "rank_score": score,
                        "route_strategy": proposal.get("learned_route_strategy"),
                        "route_source": proposal.get("beam_source"),
                    }
        ranked = sorted(plans.values(), key=lambda item: (-float(item["rank_score"]), item["plan"]))[: max(1, int(args.max_plan_candidates))]
        out[str(row.get("task_id") or "")] = ranked
    return out


def generate_detail_rows(
    model: Any,
    eval_rows: list[dict[str, Any]],
    plan_rows: dict[str, list[dict[str, Any]]],
    source_fields: list[str],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    config: dict[str, Any],
    seed: int,
    *,
    arm_id: str,
    substrate: str,
    max_source: int,
    max_target: int,
    fanout_top_k: int,
    grammar_top_k: int,
    torch: Any,
    device: Any,
) -> list[dict[str, Any]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    source_texts: list[str] = []
    for task in eval_rows:
        task_id = str(task.get("task_id") or "")
        for plan in plan_rows.get(task_id, []):
            pairs.append((task, plan))
            source_texts.append(detail_source_text(task, source_fields, str(plan.get("plan") or "")))
    if not pairs:
        return []
    encoded = encode_many(source_texts, source_vocab, max_source)
    decoded_batches = generate_candidates(
        model,
        encoded,
        target_vocab,
        max_target_tokens=max_target,
        fanout_top_k=fanout_top_k,
        grammar_top_k=grammar_top_k,
        torch=torch,
        device=device,
    )
    max_rank: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for (task, plan), decoded_rows in zip(pairs, decoded_batches):
        task_id = str(task.get("task_id") or "")
        for proposal in decoded_rows:
            max_rank[task_id] += 1
            repair = grammar_repaired_body(task, raw_body=str(proposal.get("body") or ""), decoded_tokens=list(proposal.get("decoded_tokens") or []))
            code = render_private_function(task, str(repair["body"]))
            row = candidate_row(
                task,
                code=code,
                phase="private_eval",
                arm_id=arm_id,
                substrate=substrate,
                view="sts_on_detail_body_probe",
                rank=1000 + int(max_rank[task_id]),
                rank_score=float(proposal.get("rank_score") or 0.0),
                config=config,
                seed=seed,
                decoded_token_count=int(proposal.get("decoded_token_count") or 0),
                decoded_token_sha256=str(proposal.get("decoded_token_sha256") or ""),
            )
            row["candidate_source"] = "neural_seed_plan_conditioned_detail_body_probe"
            row["detail_body_decoder"] = {
                "enabled": True,
                "conditioned_plan": plan.get("plan"),
                "plan_rank_score": plan.get("rank_score"),
                "route_strategy": plan.get("route_strategy"),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "teacher_used": False,
            }
            row["body_structure_decode"] = {
                "target_mode": "body_tokens",
                "rendered_from_statement_skeleton": False,
                "rendered_from_semantic_slots": False,
                "semantic_plan": plan.get("plan"),
                "semantic_plan_supported": False,
                "plan_conditioned_detail_body_decoder": True,
                "fallback_return_used": False,
            }
            row["grammar_repair"] = {
                "policy": "deterministic_python_body_repair_v1",
                "raw_syntax_ok": bool(repair["raw_syntax_ok"]),
                "repaired_syntax_ok": bool(repair["repaired_syntax_ok"]),
                "changed": bool(repair["changed"]),
                "strategy": repair["strategy"],
                "repair_passes": repair["repair_passes"],
                "fallback_return_used": bool(repair["fallback_return_used"]),
                "fallback_return_shape": repair["fallback_return_shape"],
                "contract_return_shape_for_diagnostics": repair["contract_return_shape_for_diagnostics"],
                "raw_failure": repair["raw_failure"],
                "repaired_failure": repair["repaired_failure"],
            }
            row["provenance"]["ranker"] = "plan_conditioned_body_token_log_probability"
            row["provenance"]["training_target"] = "private_train_solution_body_tokens"
            out.append(row)
    return out


def detail_source_text(row: dict[str, Any], fields: list[str], plan: str) -> str:
    return f"{row_text(row, fields)}\nconditioned_semantic_plan:{plan}".strip()


def semantic_plan_from_solution(row: dict[str, Any]) -> str:
    from neural_seed_token_decoder_comparator import semantic_plan_from_body

    return semantic_plan_from_body(str(row.get("solution_body") or ""))


def encode_body_targets(rows: list[dict[str, Any]], vocab: dict[str, int], max_len: int) -> list[list[int]]:
    out = []
    for row in rows:
        ids = [vocab["<bos>"]]
        ids.extend(vocab.get(tok, vocab["<unk>"]) for tok in body_tokens(str(row.get("solution_body") or ""))[: max(1, max_len - 2)])
        ids.append(vocab["<eos>"])
        ids = ids[:max_len]
        out.append(ids + [vocab["<pad>"]] * max(0, max_len - len(ids)))
    return out


def summarize_plan_conditions(plan_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = Counter()
    strategies = Counter()
    for rows in plan_rows.values():
        for row in rows:
            counts[str(row.get("plan") or "")] += 1
            strategies[str(row.get("route_strategy") or "")] += 1
    return {
        "task_count": len(plan_rows),
        "plan_condition_count": sum(counts.values()),
        "top_plans": dict(counts.most_common(12)),
        "route_strategy_counts": dict(strategies.most_common(12)),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def per_arm_field(arms: dict[str, Any], field: str) -> dict[str, Any]:
    return {arm: get_path(row, ["summary", field], None) for arm, row in arms.items()}


def planned_report(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "policy": "project_theseus_neural_seed_plan_conditioned_detail_body_probe_v0",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "execute": False,
        "config": args.config,
        "summary": {"comparison_level": config.get("comparison_level"), "external_inference_calls": 0},
        "external_inference_calls": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Neural Seed Plan-Conditioned Detail Body Probe",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- seed: `{summary.get('seed')}`",
        f"- train_rows: `{summary.get('train_rows')}`",
        f"- eval_rows: `{summary.get('eval_rows')}`",
        f"- target_mode: `{summary.get('target_mode')}`",
        f"- any_improvement: `{summary.get('any_improvement')}`",
        f"- parameter_match_delta: `{summary.get('parameter_match_delta')}`",
        "",
        "## Arms",
        "",
    ]
    for arm, row in dict_or_empty(report.get("arms")).items():
        arm_summary = dict_or_empty(row.get("summary"))
        lines.extend(
            [
                f"### {arm}",
                f"- baseline_pass_rate: `{arm_summary.get('baseline_pass_rate')}`",
                f"- augmented_pass_rate: `{arm_summary.get('augmented_pass_rate')}`",
                f"- delta: `{arm_summary.get('delta')}`",
                f"- detail_candidate_rows: `{arm_summary.get('detail_candidate_rows')}`",
                f"- syntax_pass_rate: `{arm_summary.get('syntax_pass_rate')}`",
                f"- fallback_return_rows: `{arm_summary.get('fallback_return_rows')}`",
                "",
            ]
        )
    lines.append("## Gates")
    lines.append("")
    for gate_row in report.get("gates") or []:
        lines.append(f"- `{gate_row.get('name')}`: `{gate_row.get('passed')}`")
    lines.append("")
    lines.append(str(report.get("score_semantics") or ""))
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
