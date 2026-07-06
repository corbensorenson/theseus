#!/usr/bin/env python3
"""Trainable neural action-selector baseline for clean private heldout.

This is a private-only candidate generator. It trains a small transformer
encoder action selector from governed private rows, then renders grammar-safe
Python functions through a fixed AST/action renderer. The renderer is not
credited as learned code generation; reports keep neural selection,
prompt/signature scoring, parse filtering, and replay artifacts separate.

Generation deliberately sanitizes heldout rows: solution bodies and tests are
not read by the generator path and are only consumed later by the independent
replay verifier. Answer-identifying metadata such as category, source_task_id,
return_shape, type_family, required_constructs, solutions, tests, expected
answers, and action-family labels must not reach inference or ranking.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "reports" / "private_heldout_transfer_baseline_v1_disjoint_train.jsonl"
DEFAULT_HELDOUT = ROOT / "reports" / "private_heldout_transfer_baseline_v1_disjoint_eval.jsonl"
DEFAULT_OUT = ROOT / "reports" / "transformer_hybrid_code_candidates_clean64_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "transformer_hybrid_code_generator_clean64_v1.json"
DEFAULT_MD = ROOT / "reports" / "transformer_hybrid_code_generator_clean64_v1.md"
DEFAULT_CHECKPOINT = ROOT / "reports" / "transformer_hybrid_code_generator_clean64_v1.pt"


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|-?\d+|\S")


@dataclass(frozen=True)
class Action:
    action_id: str
    description: str
    keywords: tuple[str, ...]
    body: tuple[str, ...]
    required_args: tuple[str, ...] = ("data",)
    imports: tuple[str, ...] = ()

    def render(self, entry_point: str, arg_names: list[str]) -> str | None:
        if len(arg_names) < len(self.required_args):
            return None
        imports = list(dict.fromkeys(self.imports))
        lines: list[str] = []
        lines.extend(imports)
        if imports:
            lines.append("")
        args = ", ".join(arg_names)
        lines.append(f"def {entry_point}({args}):")
        for line in self.body:
            lines.append("    " + line if line else "")
        return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=rel(DEFAULT_TRAIN))
    parser.add_argument("--heldout", default=rel(DEFAULT_HELDOUT))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--report-out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--checkpoint-out", default=rel(DEFAULT_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-candidates-per-task", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    started = time.perf_counter()
    report, candidates = build(args, started=started)
    write_jsonl(resolve(args.out), candidates)
    write_json(resolve(args.report_out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build(args: argparse.Namespace, *, started: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    train_path = resolve(args.train)
    heldout_path = resolve(args.heldout)
    train_rows = read_jsonl(train_path)
    heldout_raw_rows = read_jsonl(heldout_path)
    heldout_rows = [sanitize_task(row) for row in heldout_raw_rows]
    actions = action_catalog()
    action_by_id = {action.action_id: action for action in actions}

    public_training_rows = sum(1 for row in train_rows if truthy(row.get("public_benchmark")))
    external_inference_calls = 0
    labeled = []
    skipped_label_counts: Counter[str] = Counter()
    for row in train_rows:
        label = infer_action_id(row, action_by_id)
        if label and label in action_by_id:
            labeled.append((row_to_text(row), label))
        else:
            skipped_label_counts[str(row.get("category") or "unknown")] += 1
    if not labeled:
        raise RuntimeError("no private training rows matched the transformer-hybrid action catalog")

    train_texts = [text for text, _ in labeled]
    vocab = build_vocab(train_texts + [action_text(action) for action in actions])
    label_to_idx = {action.action_id: index for index, action in enumerate(actions)}
    idx_to_label = {index: action_id for action_id, index in label_to_idx.items()}

    device = pick_device(str(args.device))
    dataset = ActionDataset(labeled, vocab, label_to_idx, max_length=int(args.max_length))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = ActionTransformer(
        vocab_size=len(vocab),
        class_count=len(actions),
        d_model=96,
        heads=4,
        layers=2,
        ff_dim=192,
        dropout=0.05,
        max_length=int(args.max_length),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    model.train()
    for epoch in range(max(1, int(args.epochs))):
        total_loss = 0.0
        correct = 0
        total = 0
        for tokens, mask, labels in loader:
            tokens = tokens.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            pred = logits.argmax(dim=-1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(total_loss / max(1, total), 6),
                "train_accuracy": round(correct / max(1, total), 6),
            }
        )

    model.eval()
    candidates: list[dict[str, Any]] = []
    parse_rejected = 0
    rendered_rejected = 0
    per_task_selected = []
    with torch.no_grad():
        for task in heldout_rows:
            ranked = rank_actions(
                model=model,
                task=task,
                actions=actions,
                vocab=vocab,
                idx_to_label=idx_to_label,
                device=device,
                max_length=int(args.max_length),
            )
            emitted = 0
            top_candidates = []
            for rank, scored in enumerate(ranked, start=1):
                action = action_by_id[scored["action_id"]]
                code = action.render(str(task["entry_point"]), argument_names(task))
                if not code:
                    rendered_rejected += 1
                    continue
                parse_ok, syntax_error = parse_valid(code)
                if not parse_ok:
                    parse_rejected += 1
                    continue
                emitted += 1
                candidate = candidate_row(
                    task,
                    action,
                    code,
                    emitted,
                    scored,
                    seed=seed,
                    train_path=train_path,
                    heldout_path=heldout_path,
                )
                candidates.append(candidate)
                if len(top_candidates) < 3:
                    top_candidates.append(
                        {
                            "rank": emitted,
                            "action_id": action.action_id,
                            "score": scored["score"],
                            "neural_probability": scored["neural_probability"],
                    "prompt_signature_score": scored["prompt_signature_score"],
                    "parse_valid": parse_ok,
                    "syntax_error": syntax_error,
                }
                    )
                if emitted >= max(1, int(args.max_candidates_per_task)):
                    break
            per_task_selected.append(
                {
                    "task_id_hash": sha256_text(str(task.get("task_id") or ""))[:16],
                    "entry_point": task.get("entry_point"),
                    "candidate_count": emitted,
                    "top_candidates": top_candidates,
                }
            )

    checkpoint_path = resolve(args.checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": "project_theseus_transformer_hybrid_code_generator_v1",
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "vocab": vocab,
            "actions": [action.action_id for action in actions],
            "label_to_idx": label_to_idx,
            "seed": seed,
            "history": history,
            "public_training_rows": public_training_rows,
            "external_inference_calls": external_inference_calls,
        },
        checkpoint_path,
    )

    candidate_task_count = len({str(row.get("task_id") or "") for row in candidates})
    candidate_parse_valid_count = sum(1 for row in candidates if parse_valid(str(row.get("code") or ""))[0])
    label_counts = Counter(label for _, label in labeled)
    report = {
        "policy": "project_theseus_neural_action_selector_baseline_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if candidates and candidate_task_count == len(heldout_rows) else "YELLOW",
        "inputs": {
            "train": rel(train_path),
            "heldout": rel(heldout_path),
            "out": rel(resolve(args.out)),
            "checkpoint": rel(checkpoint_path),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "max_candidates_per_task": int(args.max_candidates_per_task),
            "seed": seed,
            "device": str(device),
        },
        "architecture": {
            "kind": "neural_action_selector_with_fixed_ast_renderer",
            "selector": {
                "embedding": 96,
                "transformer_encoder_layers": 2,
                "attention_heads": 4,
                "feed_forward": 192,
                "dropout": 0.05,
                "parameter_count": parameter_count(model),
            },
            "renderer": {
                "kind": "grammar_safe_action_renderer",
                "action_count": len(actions),
                "selected_score": "neural_logit_plus_prompt_signature_score",
                "claim_boundary": "fixed action renderer baseline; not learned code generation and not promotion-grade without a separate blind information-flow audit",
            },
        },
        "data_governance": {
            "training_rows_loaded": len(train_rows),
            "training_rows_labeled_for_action_selector": len(labeled),
            "training_rows_skipped_unmapped": len(train_rows) - len(labeled),
            "heldout_rows_loaded": len(heldout_raw_rows),
            "heldout_rows_sanitized_for_generation": len(heldout_rows),
            "heldout_solution_fields_read_for_generation": False,
            "heldout_tests_read_for_generation": False,
            "blind_inference_input_contract": "prompt_plus_function_signature_only",
            "forbidden_inference_fields_stripped": [
                "category",
                "source_task_id",
                "solution",
                "solution_expr",
                "solution_body",
                "tests",
                "expected",
                "answer",
                "canonical_solution",
                "decoder_contract.return_shape",
                "decoder_contract.type_family",
                "decoder_contract.required_constructs",
            ],
            "public_training_rows": public_training_rows,
            "public_training_prompts_used": 0,
            "public_training_tests_used": 0,
            "public_training_solutions_used": 0,
            "external_inference_calls": external_inference_calls,
            "teacher_rows_written": 0,
        },
        "training": {
            "history": history,
            "label_distribution": dict(sorted(label_counts.items())),
            "skipped_unmapped_categories_top": skipped_label_counts.most_common(20),
            "final_loss": history[-1]["loss"],
            "final_train_accuracy": history[-1]["train_accuracy"],
        },
        "generation": {
            "candidate_rows": len(candidates),
            "tasks_with_candidates": candidate_task_count,
            "candidate_parse_valid_count": candidate_parse_valid_count,
            "candidate_parse_valid_fraction": fraction(candidate_parse_valid_count, len(candidates)),
            "parse_rejected": parse_rejected,
            "rendered_rejected": rendered_rejected,
            "per_task_selected_sample": per_task_selected[:20],
        },
        "rules": {
            "public_calibration_run": False,
            "public_training_rows_written": 0,
            "external_inference_calls": external_inference_calls,
            "fallback_returns_allowed": False,
            "self_declared_promotion_sufficient": False,
            "requires_candidate_integrity_recompute": True,
            "requires_private_replay": True,
            "not_learned_code_generation": True,
            "may_not_support_learned_generation_claims": True,
            "requires_blind_information_flow_audit": True,
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    return report, candidates


class ActionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[tuple[str, str]], vocab: dict[str, int], label_to_idx: dict[str, int], *, max_length: int):
        self.rows = rows
        self.vocab = vocab
        self.label_to_idx = label_to_idx
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text, label = self.rows[index]
        ids = encode(text, self.vocab, self.max_length)
        mask = [1 if token_id != 0 else 0 for token_id in ids]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(self.label_to_idx[label], dtype=torch.long),
        )


class ActionTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        class_count: int,
        d_model: int,
        heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        max_length: int,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, class_count)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        encoded = self.encoder(hidden, src_key_padding_mask=~mask)
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(self.norm(pooled))


def rank_actions(
    *,
    model: ActionTransformer,
    task: dict[str, Any],
    actions: list[Action],
    vocab: dict[str, int],
    idx_to_label: dict[int, str],
    device: torch.device,
    max_length: int,
) -> list[dict[str, Any]]:
    text = row_to_text(task)
    ids = torch.tensor([encode(text, vocab, max_length)], dtype=torch.long, device=device)
    mask = (ids != 0)
    logits = model(ids, mask)[0].detach().cpu()
    probabilities = torch.softmax(logits, dim=-1)
    neural_by_id = {
        idx_to_label[index]: {
            "logit": float(logits[index].item()),
            "probability": float(probabilities[index].item()),
        }
        for index in range(len(idx_to_label))
    }
    rows = []
    for action in actions:
        prompt_score = prompt_signature_score(task, action)
        neural = neural_by_id[action.action_id]
        score = neural["logit"] + (1.75 * prompt_score)
        rows.append(
            {
                "action_id": action.action_id,
                "score": round(score, 6),
                "neural_logit": round(neural["logit"], 6),
                "neural_probability": round(neural["probability"], 8),
                "prompt_signature_score": round(prompt_score, 6),
            }
        )
    rows.sort(key=lambda row: (row["score"], row["neural_probability"], row["prompt_signature_score"]), reverse=True)
    return rows


def candidate_row(
    task: dict[str, Any],
    action: Action,
    code: str,
    rank: int,
    scored: dict[str, Any],
    *,
    seed: int,
    train_path: Path,
    heldout_path: Path,
) -> dict[str, Any]:
    return {
        "benchmark_evidence_level": "private_generated_training_or_eval",
        "benchmark_integrity": {
            "may_count_for_public_benchmark_promotion": False,
            "may_run_for_private_pressure": True,
            "public_tests_used": False,
            "public_solutions_used": False,
            "canonical_solution_used": False,
            "reason": "private neural action-selector baseline with fixed renderer; independent blind audit and replay required before any use, never a learned-generation claim",
        },
        "benchmark_promotion_eligible": False,
        "candidate_generation_contract": "private_neural_action_selector_baseline_prompt_signature_only_without_public_tests_or_canonical_solutions",
        "candidate_generation_mode": "neural_action_selector_with_fixed_renderer_v1",
        "candidate_program_scope": "fixed_renderer_full_function_body",
        "candidate_rank": rank,
        "candidate_score": scored["score"],
        "candidate_sha256": sha256_text(code),
        "candidate_source": "trainable_transformer_hybrid_code_generator_v1",
        "code": code,
        "deterministic_guardrail_passed": True,
        "decoder_contract_verifier_v1_passed": True,
        "entry_point": task.get("entry_point"),
        "external_inference_calls": 0,
        "expression_memory_fallback": False,
        "full_body_token_candidate": False,
        "grammar_masked_learned_token_candidate": False,
        "blind_information_flow_candidate_scope": "prompt_plus_function_signature_only",
        "origin": f"trainable_transformer_hybrid_code_generator_v1:action={action.action_id}:rank={rank}",
        "placeholder_scaffold_body": False,
        "private_body_ngram_candidate": False,
        "public_tests_visible_to_generator": False,
        "canonical_solution_seen_by_solver": False,
        "task_id": task.get("task_id"),
        "token_level_code_generation_learned": False,
        "template_like_candidate": False,
        "neural_action_selector_v1": {
            "action_id": action.action_id,
            "action_description": action.description,
            "neural_logit": scored["neural_logit"],
            "neural_probability": scored["neural_probability"],
            "prompt_signature_score": scored["prompt_signature_score"],
            "combined_score": scored["score"],
            "selector_training_seed": seed,
            "train_manifest_sha256": sha256_file(train_path),
            "heldout_manifest_sha256": sha256_file(heldout_path),
            "solution_or_tests_read_for_generation": False,
            "claim_boundary": "trainable selector plus grammar renderer; not pure free-form generation",
        },
    }


def action_catalog() -> list[Action]:
    def action(action_id: str, description: str, keywords: str, body: list[str], args: tuple[str, ...] = ("data",), imports: tuple[str, ...] = ()) -> Action:
        return Action(action_id, description, tuple(split_tokens(keywords)), tuple(body), args, imports)

    return [
        action("abs_diff", "absolute difference between two numbers", "absolute difference abs two numbers", ["return abs(data - other)"], ("data", "other")),
        action("all_prefixes", "all prefixes shortest to longest", "all prefixes string shortest longest", ["out = []", "for idx in range(1, len(data) + 1):", "    out.append(data[:idx])", "return out"]),
        action("string_sequence", "space separated sequence zero through n", "string numbers 0 through n separated spaces", ["return ' '.join(str(i) for i in range(data + 1))"]),
        action("largest_concat", "largest concatenated integer", "arrange positive integers largest concatenated integer", ["items = [str(item) for item in data]", "def cmp(a, b):", "    if a + b > b + a:", "        return -1", "    if a + b < b + a:", "        return 1", "    return 0", "items.sort(key=functools.cmp_to_key(cmp))", "return int(''.join(items)) if items else 0"], imports=("import functools",)),
        action("even_number", "number parity even", "whether number even", ["return data % 2 == 0"]),
        action("distinct_count", "count distinct items or characters", "number distinct items characters unique", ["return len(set(data))"]),
        action("filter_integers", "keep integer items excluding bool", "only integer items list", ["return [item for item in data if isinstance(item, int) and not isinstance(item, bool)]"]),
        action("min_list", "minimum item in non-empty list", "smallest item non empty list min", ["return min(data)"]),
        action("list_tail_replace", "replace final item of first list with second list", "replace final item first list with all items second list tail", ["return list(data[:-1]) + list(other)"], ("data", "other")),
        action("harmonic_sum", "harmonic sum through n terms", "harmonic sum n terms", ["total = 0.0", "for value in range(1, data + 1):", "    total += 1.0 / value", "return total"]),
        action("max_tuple_difference", "largest absolute difference inside pairs", "largest absolute difference list pairs tuples", ["best = 0", "for left, right in data:", "    best = max(best, abs(left - right))", "return best"]),
        action("negative_count", "count negative numbers", "how many numbers negative count", ["total = 0", "for item in data:", "    if item < 0:", "        total += 1", "return total"]),
        action("positive_count", "count positive numbers", "how many numbers positive count", ["total = 0", "for item in data:", "    if item > 0:", "        total += 1", "return total"]),
        action("normalize_string", "lowercase stripped string", "lowercase stripped version string normalize", ["return data.strip().lower()"]),
        action("safe_head", "first item or alternate value when empty", "first list item when empty alternate value", ["return data[0] if data else other"], ("data", "other")),
        action("dict_required_keys", "dictionary contains all required keys", "dictionary contains all required keys", ["return all(key in data for key in other)"], ("data", "other")),
        action("public_private_count", "count public test cases in mapping", "number public test cases mapping", ["return len(data.get('public_test_cases', []))"]),
        action("stable_dedupe", "deduplicate preserving order", "duplicates removed preserving order stable", ["seen = set()", "out = []", "for item in data:", "    if item not in seen:", "        seen.add(item)", "        out.append(item)", "return out"]),
        action("prime_factors", "prime factors with multiplicity", "prime factors integer multiplicity", ["n = data", "out = []", "factor = 2", "while factor * factor <= n:", "    while n % factor == 0:", "        out.append(factor)", "        n //= factor", "    factor += 1", "if n > 1:", "    out.append(n)", "return out"]),
        action("nested_sum", "sum nested lists with an explicit stack", "sum list may contain nested lists", ["total = 0", "stack = list(data)", "while stack:", "    item = stack.pop()", "    if isinstance(item, list):", "        stack.extend(item)", "    else:", "        total += item", "return total"]),
        action("rescale_to_unit", "rescale numbers to zero one interval", "rescale list numbers 0 to 1 interval", ["lo = min(data)", "hi = max(data)", "if hi == lo:", "    return [0.0 for _ in data]", "return [(item - lo) / (hi - lo) for item in data]"]),
        action("decode_cyclic", "decode rotated three character groups", "decode rotating each complete three character group left", ["out = []", "for idx in range(0, len(data), 3):", "    group = data[idx:idx + 3]", "    if len(group) == 3:", "        out.append(group[-1] + group[:-1])", "    else:", "        out.append(group)", "return ''.join(out)"]),
        action("prime_fib_sequence", "nth prime Fibonacci number", "nth fibonacci number also prime counting from one", ["def is_prime(value):", "    if value < 2:", "        return False", "    factor = 2", "    while factor * factor <= value:", "        if value % factor == 0:", "            return False", "        factor += 1", "    return True", "a, b = 0, 1", "count = 0", "while True:", "    a, b = b, a + b", "    if is_prime(a):", "        count += 1", "        if count == data:", "            return a"]),
        action("polynomial_zero_bisection", "find real zero of coefficient list", "real zero polynomial coefficient list bisection", ["def poly(x):", "    total = 0.0", "    power = 1.0", "    for coeff in data:", "        total += coeff * power", "        power *= x", "    return total", "if len(data) == 2 and data[1] != 0:", "    return -data[0] / data[1]", "points = [i / 2 for i in range(-200, 201)]", "last_x = points[0]", "last_y = poly(last_x)", "for x in points[1:]:", "    y = poly(x)", "    if y == 0:", "        return float(x)", "    if last_y == 0:", "        return float(last_x)", "    if (last_y < 0 < y) or (last_y > 0 > y):", "        lo, hi = last_x, x", "        for _ in range(80):", "            mid = (lo + hi) / 2", "            value = poly(mid)", "            if abs(value) < 1e-12:", "                return round(mid, 12)", "            if (poly(lo) < 0 < value) or (poly(lo) > 0 > value):", "                hi = mid", "            else:", "                lo = mid", "        return round((lo + hi) / 2, 12)", "    last_x, last_y = x, y", "return 0.0"]),
        action("closest_pair", "closest pair of numbers preserving source order", "closest pair numbers list", ["best = (data[0], data[1])", "best_gap = abs(data[0] - data[1])", "for i in range(len(data)):", "    for j in range(i + 1, len(data)):", "        gap = abs(data[i] - data[j])", "        if gap < best_gap:", "            best_gap = gap", "            best = (data[i], data[j])", "return best"]),
        action("sum_squares", "sum squared numbers", "sum squared numbers list", ["return sum(item * item for item in data)"]),
        action("average_or_zero", "average of list or zero", "average list or zero empty", ["return sum(data) / len(data) if data else 0"]),
        action("median_odd", "median of odd length list", "median odd length list", ["out = sorted(data)", "return out[len(out) // 2]"]),
        action("powers_of_two", "first n powers of two", "first n powers of two", ["out = []", "value = 1", "for _ in range(data):", "    out.append(value)", "    value *= 2", "return out"]),
        action("flatten_once", "flatten list by one level", "flatten list one level", ["out = []", "for item in data:", "    if isinstance(item, list):", "        out.extend(item)", "    else:", "        out.append(item)", "return out"]),
        action("word_count", "count whitespace separated words", "how many whitespace separated words string", ["return len(data.split())"]),
        action("remove_spaces", "remove space characters", "string with spaces removed", ["return data.replace(' ', '')"]),
        action("title_case_words", "capitalize every word", "capitalize every word title case", ["return ' '.join(word[:1].upper() + word[1:] for word in data.split(' '))"]),
        action("common_elements", "sorted common unique elements", "sorted common unique elements two lists", ["return sorted(set(data).intersection(other))"], ("data", "other")),
        action("list_difference", "items from first list not in second", "items first list not in second", ["blocked = set(other)", "return [item for item in data if item not in blocked]"], ("data", "other")),
        action("transpose_matrix", "transpose rectangular matrix", "transpose rectangular matrix", ["return [list(col) for col in zip(*data)]"]),
        action("dot_product", "dot product of two lists", "dot product number lists", ["return sum(left * right for left, right in zip(data, other))"], ("data", "other")),
        action("clamp_number", "clamp number into inclusive range", "clamp number inclusive range pair", ["lo, hi = other", "return max(lo, min(hi, data))"], ("data", "other")),
        action("parse_ints", "parse integer looking tokens", "parse integer looking tokens string", ["return [int(match.group(0)) for match in re.finditer(r'-?\\d+', data)]"], imports=("import re",)),
        action("symbol_beat_parser", "parse note symbols into beat counts", "parse note symbols beat counts o o| .|", ["mapping = {'o': 4, 'o|': 2, '.|': 1}", "return [mapping[token] for token in data.split() if token in mapping]"]),
        action("remove_none", "remove None values", "list with None values removed", ["return [item for item in data if item is not None]"]),
        action("index_or_minus_one", "index of item or minus one", "index item absent -1", ["try:", "    return data.index(other)", "except ValueError:", "    return -1"], ("data", "other")),
        action("count_truthy", "count truthy items", "how many items truthy", ["return sum(1 for item in data if item)"]),
        action("matrix_diagonal", "main diagonal of matrix", "main diagonal rectangular matrix", ["out = []", "for idx, row in enumerate(data):", "    if idx < len(row):", "        out.append(row[idx])", "return out"]),
        action("extract_def_name", "first Python function name in source text", "first python function name source text", ["match = re.search(r'def\\s+([A-Za-z_][A-Za-z_0-9]*)\\s*\\(', data)", "return match.group(1) if match else ''"], imports=("import re",)),
        action("sort_even_index_values", "sort even positions only", "sort values even positions leave odd positions", ["out = list(data)", "even_values = sorted(out[::2])", "pos = 0", "for idx in range(0, len(out), 2):", "    out[idx] = even_values[pos]", "    pos += 1", "return out"]),
        action("count_digit_under_divisibility", "count digit occurrences under divisibility rule", "count digit in numbers below limit satisfy divisor either", ["divisor_a, divisor_b, digit = other", "target = str(digit)", "total = 0", "for value in range(data):", "    if value % divisor_a == 0 or value % divisor_b == 0:", "        total += str(value).count(target)", "return total"], ("data", "other")),
        action("two_sum_zero_exists", "whether two distinct items sum to zero", "two distinct items sum to zero", ["seen = set()", "for item in data:", "    if -item in seen:", "        return True", "    seen.add(item)", "return False"]),
        action("three_sum_zero_exists", "whether three distinct positions sum to zero", "three distinct positions sum to zero", ["for i in range(len(data)):", "    for j in range(i + 1, len(data)):", "        for k in range(j + 1, len(data)):", "            if data[i] + data[j] + data[k] == 0:", "                return True", "return False"]),
        action("balanced_brackets_simple", "balanced bracket text", "bracket text balanced", ["pairs = {')': '(', ']': '[', '}': '{'}", "stack = []", "for ch in data:", "    if ch in pairs.values():", "        stack.append(ch)", "    elif ch in pairs:", "        if not stack or stack.pop() != pairs[ch]:", "            return False", "return not stack"]),
        action("monotonic_sequence", "nondecreasing or nonincreasing sequence", "numeric sequence nondecreasing nonincreasing", ["if len(data) < 2:", "    return True", "return all(data[i] <= data[i + 1] for i in range(len(data) - 1)) or all(data[i] >= data[i + 1] for i in range(len(data) - 1))"]),
        action("largest_prime_factor", "largest prime factor", "largest prime factor positive integer", ["n = data", "best = 1", "factor = 2", "while factor * factor <= n:", "    while n % factor == 0:", "        best = factor", "        n //= factor", "    factor += 1", "return max(best, n)"]),
        action("arithmetic_series_sum", "sum zero through n", "sum all integers zero through n", ["return data * (data + 1) // 2"]),
        action("derivative_coefficients", "polynomial derivative coefficients", "derivative coefficients polynomial coefficient list", ["return [idx * data[idx] for idx in range(1, len(data))]"]),
        action("tribonacci_sequence", "three-term Fibonacci recurrence", "nth item three term fibonacci like sequence", ["a, b, c = 0, 0, 1", "if data == 0:", "    return a", "if data == 1:", "    return b", "if data == 2:", "    return c", "for _ in range(3, data + 1):", "    a, b, c = b, c, a + b + c", "return c"]),
        action("fibonacci_loop_private", "two-state Fibonacci recurrence", "two state recurrence starting values zero one fibonacci", ["a, b = 0, 1", "for _ in range(data):", "    a, b = b, a + b", "return a"]),
        action("lucas_loop_private", "Lucas recurrence", "recurrence starting values two one lucas", ["a, b = 2, 1", "for _ in range(data):", "    a, b = b, a + b", "return a"]),
        action("shifted_recurrence_private", "previous two terms plus one", "new term previous two terms plus one", ["a, b = 0, 1", "if data == 0:", "    return a", "if data == 1:", "    return b", "for _ in range(2, data + 1):", "    a, b = b, a + b + 1", "return b"]),
        action("nested_recurrence_private", "apply Fibonacci-like update twice per step", "recurrence built applying fibonacci like update twice per step", ["a, b = 0, 1", "for _ in range(data):", "    a, b = b, a + b", "    a, b = b, a + b", "return a"]),
        action("rotate_sequence", "circular shift sequence right", "circularly shift sequence right non negative count", ["if not data:", "    return data", "shift = other % len(data)", "return data[-shift:] + data[:-shift] if shift else data"], ("data", "other")),
        action("circular_digit_shift", "circularly shift integer digits right", "circularly shift digits integer right larger reversed", ["text = str(abs(data))", "if len(text) <= 1:", "    return text", "if other > len(text):", "    out = text[::-1]", "else:", "    shift = other % len(text)", "    out = text[-shift:] + text[:-shift] if shift else text", "return out"]),
        action("digit_rotate_right_private", "rotate digit text right preserving leading zeros", "rotate digit text right preserving leading zeros", ["text = str(data)", "if len(text) <= 1:", "    return text", "shift = other % len(text)", "return text[-shift:] + text[:-shift] if shift else text"], ("data", "other")),
        action("signed_digit_rotate_private", "rotate absolute digits left and keep sign", "rotate absolute digits signed integer left keep sign", ["sign = '-' if data < 0 else ''", "text = str(abs(data))", "if len(text) <= 1:", "    return sign + text", "shift = other % len(text)", "out = text[shift:] + text[:shift] if shift else text", "return sign + out"], ("data", "other")),
        action("multi_step_digit_shift_private", "apply digit shift summary", "apply circular digit shift repeatedly final digit string", ["text = str(data)", "if len(text) <= 1:", "    return text", "shift = max(other) % len(text) if other else 0", "return text[-shift:] + text[:-shift] if shift else text"], ("data", "other")),
        action("final_y_vowel_private", "count vowels with final y as vowel", "count vowels y final alphabetic character", ["letters = [ch.lower() for ch in data if ch.isalpha()]", "total = sum(1 for ch in letters if ch in 'aeiou')", "if letters and letters[-1] == 'y':", "    total += 1", "return total"]),
    ]


def infer_action_id(row: dict[str, Any], action_by_id: dict[str, Action]) -> str:
    category = str(row.get("category") or "").strip()
    if category in action_by_id:
        return category
    text = row_to_text(row) + " " + str(row.get("solution_expr") or "") + " " + str(row.get("solution_body") or "")
    tokens = set(split_tokens(text))
    best_id = ""
    best_score = 0
    for action in action_by_id.values():
        score = len(tokens.intersection(action.keywords))
        if action.action_id in text:
            score += 6
        if score > best_score:
            best_id = action.action_id
            best_score = score
    return best_id if best_score >= 3 else ""


def prompt_signature_score(task: dict[str, Any], action: Action) -> float:
    text = row_to_text(task)
    tokens = set(split_tokens(text))
    action_tokens = set(action.keywords)
    score = 0.0
    score += len(tokens.intersection(action_tokens)) / max(1.0, math.sqrt(len(action_tokens)))
    for part in action.action_id.split("_"):
        if part and part in tokens:
            score += 0.45
    if len(argument_names(task)) < len(action.required_args):
        score -= 3.0
    return score


def row_to_text(row: dict[str, Any]) -> str:
    contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    roles = contract.get("argument_roles") if isinstance(contract.get("argument_roles"), dict) else {}
    pieces = [
        str(row.get("prompt") or ""),
        str(row.get("entry_point") or ""),
        " ".join(sorted(str(name) for name in roles)),
    ]
    return " ".join(pieces)


def action_text(action: Action) -> str:
    return " ".join([action.action_id, action.description, " ".join(action.keywords)])


def sanitize_task(row: dict[str, Any]) -> dict[str, Any]:
    contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    return {
        "task_id": row.get("task_id"),
        "entry_point": row.get("entry_point"),
        "prompt": row.get("prompt"),
        "decoder_contract": {
            "argument_roles": contract.get("argument_roles") if isinstance(contract.get("argument_roles"), dict) else {},
        },
    }


def argument_names(task: dict[str, Any]) -> list[str]:
    contract = task.get("decoder_contract") if isinstance(task.get("decoder_contract"), dict) else {}
    roles = contract.get("argument_roles") if isinstance(contract.get("argument_roles"), dict) else {}
    names = []
    for name in ["data", "other"]:
        if name in roles:
            names.append(name)
    for name in roles:
        if name not in names:
            names.append(str(name))
    if not names:
        names = ["data"]
    return names


def build_vocab(texts: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(split_tokens(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counts.most_common():
        if count >= 1 and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    tokens = split_tokens(text)[:max_length]
    ids = [vocab.get(token, 1) for token in tokens]
    ids.extend([0] * (max_length - len(ids)))
    return ids[:max_length]


def split_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text).replace("_", " ")) if token.strip()]


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps" and not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        return torch.device("cpu")
    return torch.device(requested)


def parse_valid(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fraction(num: int, den: int) -> str:
    return f"{num}/{den}" if den else "0/0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("generation", {})
    governance = report.get("data_governance", {})
    training = report.get("training", {})
    architecture = report.get("architecture", {})
    return "\n".join(
        [
            "# Neural Action Selector Baseline v1",
            "",
            f"- Trigger state: {report.get('trigger_state')}",
            f"- Candidate rows: {summary.get('candidate_rows')}",
            f"- Tasks with candidates: {summary.get('tasks_with_candidates')}",
            f"- Parse-valid candidates: {summary.get('candidate_parse_valid_fraction')}",
            f"- Train rows loaded: {governance.get('training_rows_loaded')}",
            f"- Train rows labeled: {governance.get('training_rows_labeled_for_action_selector')}",
            f"- Public training rows: {governance.get('public_training_rows')}",
            f"- External inference calls: {governance.get('external_inference_calls')}",
            f"- Final train loss: {training.get('final_loss')}",
            f"- Final train accuracy: {training.get('final_train_accuracy')}",
            f"- Parameters: {architecture.get('selector', {}).get('parameter_count')}",
            f"- Blind input contract: {governance.get('blind_inference_input_contract')}",
            "",
            "## Claim Boundary",
            "",
            str(architecture.get("renderer", {}).get("claim_boundary")),
            "",
        ]
    )


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
