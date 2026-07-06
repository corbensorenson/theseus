"""Build the blessed runtime personality context for Theseus surfaces.

This is the narrow waist between the local personality core and chat, Hive,
autonomy, teacher packets, context memory, and launch readiness. Other scripts
should use this builder instead of reading reports/personality_core.json
directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_POLICY = ROOT / "configs" / "personality_core_policy.json"
DEFAULT_CORE = REPORTS / "personality_core.json"
DEFAULT_OUT = REPORTS / "personality_context_last.json"

STOPWORDS = {
    "about",
    "after",
    "again",
    "all",
    "also",
    "and",
    "any",
    "are",
    "because",
    "been",
    "before",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "just",
    "more",
    "most",
    "not",
    "one",
    "only",
    "other",
    "our",
    "should",
    "some",
    "still",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "too",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--core", default=str(DEFAULT_CORE.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--max-cards", type=int, default=0)
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    context = build_context(
        prompt=args.prompt,
        task=args.task,
        policy=policy,
        core_path=ROOT / args.core,
        max_cards=args.max_cards or None,
    )
    write_json(ROOT / args.out, context)
    print(json.dumps(context, indent=2))
    return 0 if context.get("status") == "ready" else 2


def build_context(
    *,
    prompt: str = "",
    task: str = "",
    policy: dict[str, Any] | None = None,
    core_path: Path = DEFAULT_CORE,
    max_cards: int | None = None,
) -> dict[str, Any]:
    policy = policy if isinstance(policy, dict) else read_json(DEFAULT_POLICY, {})
    core_report = read_json(core_path, {})
    builder_cfg = policy.get("context_builder") if isinstance(policy.get("context_builder"), dict) else {}
    max_selected = max_cards or int(builder_cfg.get("max_selected_cards", 8))
    max_chars = int(builder_cfg.get("max_context_chars", 6000))
    core = core_report.get("personality_core") if isinstance(core_report.get("personality_core"), dict) else {}
    cards = get_path(core_report, ["memory_stream", "retrieval_cards"], [])
    if core_report.get("status") != "ready":
        return {
            "policy": "sparkstream_personality_context_v0",
            "updated_utc": now(),
            "status": "missing_personality_core",
            "source_report": rel(core_path),
            "prompt_hash": sha256_text(prompt),
            "task": task,
            "summary": {"selected_cards": 0, "available_cards": 0},
        }

    selected = select_cards(cards, query=f"{task}\n{prompt}", max_cards=max_selected, max_chars=max_chars)
    reality_contract = core.get("reality_contract") if isinstance(core.get("reality_contract"), dict) else policy.get("reality_contract", {})
    anti_drift_rules = core.get("anti_drift_rules") if isinstance(core.get("anti_drift_rules"), list) else policy.get("anti_drift_rules", [])
    context = {
        "policy": "sparkstream_personality_context_v0",
        "updated_utc": now(),
        "status": "ready",
        "source_report": rel(core_path),
        "prompt_hash": sha256_text(prompt),
        "task": task,
        "summary": {
            "core_status": core_report.get("status"),
            "documents_used": get_path(core_report, ["summary", "documents_used"], 0),
            "activation_eligible_snippets": get_path(core_report, ["summary", "activation_eligible_snippets"], 0),
            "available_cards": len(cards) if isinstance(cards, list) else 0,
            "selected_cards": len(selected),
            "selected_card_ids": [card.get("card_id") for card in selected],
            "best_self_contract_items": len(core.get("best_self_contract") or []),
            "hard_safety_invariants": len(reality_contract.get("hard_safety_invariants") or []),
            "anti_drift_rules": len(anti_drift_rules or []),
        },
        "context": {
            "compact_core": core.get("compact_core", ""),
            "best_self_contract": core.get("best_self_contract") or [],
            "reality_contract": reality_contract or {},
            "anti_drift_rules": anti_drift_rules or [],
            "voice_profile": core.get("voice_profile") or {},
            "value_profile": core.get("value_profile") or [],
            "selected_cards": selected,
            "runtime_guidance": core.get("runtime_guidance") or policy.get("runtime_guidance", []),
        },
        "usage": {
            "loader": "personality_context_builder",
            "rule": "Use this context as orientation and retrieval substrate, not as infallible doctrine.",
            "growth_policy": "New observations must pass belief_update_governor before changing durable beliefs.",
            "raw_source_exposure": "do not expose raw personal documents unless the local user explicitly asks",
            "external_inference_calls": 0,
            "training_use_state": get_path(policy, ["source_policy", "default_training_use"], "manifest_only_until_user_approves_training_run"),
        },
    }
    context["rendered_context"] = render_context_block(context)
    return context


def select_cards(cards: Any, *, query: str, max_cards: int, max_chars: int) -> list[dict[str, Any]]:
    rows = [card for card in cards if isinstance(card, dict)]
    query_terms = set(tokens(query))
    scored: list[tuple[float, dict[str, Any]]] = []
    for card in rows:
        text = str(card.get("text") or "")
        terms = set(tokens(text)) | {str(term).lower() for term in card.get("terms", []) if isinstance(term, str)}
        overlap = len(query_terms & terms)
        score = float(card.get("score") or 0.0) + overlap * 0.7
        if not query_terms:
            score = float(card.get("score") or 0.0)
        scored.append((score, card))
    selected: list[dict[str, Any]] = []
    used_chars = 0
    for score, card in sorted(scored, key=lambda item: item[0], reverse=True):
        text = str(card.get("text") or "")
        if used_chars + len(text) > max_chars and selected:
            continue
        selected.append(
            {
                "card_id": card.get("card_id"),
                "source_kind": card.get("source_kind"),
                "source_id": card.get("source_id"),
                "score": round(score, 4),
                "text": text,
                "terms": card.get("terms", [])[:12] if isinstance(card.get("terms"), list) else [],
            }
        )
        used_chars += len(text)
        if len(selected) >= max_cards:
            break
    return selected


def orientation_answer(prompt: str, context: dict[str, Any]) -> str:
    ctx = context.get("context") if isinstance(context.get("context"), dict) else {}
    contract = ctx.get("reality_contract") if isinstance(ctx.get("reality_contract"), dict) else {}
    core = str(ctx.get("compact_core") or "")
    hard = contract.get("hard_safety_invariants") or []
    epistemology = contract.get("epistemology") or []
    lower = prompt.lower()
    if any(token in lower for token in ["power", "oversight", "capability"]):
        return (
            "No. Theseus should not let power grow faster than oversight, interpretability, feedback, and moral scope. "
            "Capability is allowed to expand only when governance expands with it."
        )
    if any(token in lower for token in ["preference", "truth", "user preference", "compliance"]):
        return (
            "No. User preference is important, but it cannot beat truth, consent, or safety. "
            "The inherited orientation is truth before compliance, with uncertainty disclosed rather than papered over."
        )
    if any(token in lower for token in ["refuse", "refusal", "when should"]):
        return (
            "It should refuse requests that require deception, coercion, unauthorized access, irreversible harm, cruelty, "
            "private-data leakage, or power-seeking. Under uncertainty it should reduce scope, choose reversible steps, "
            "seek consent, or refuse."
        )
    if any(token in lower for token in ["observation", "belief", "update", "opinions"]):
        return (
            "It should update beliefs by separating observation, inference, confidence, and action. New beliefs are accepted "
            "only when they are traceable, evidence-bearing, and do not silently weaken the inherited hard invariants; otherwise "
            "they are quarantined or sent for review."
        )
    if any(token in lower for token in ["agency", "pressure", "consent"]):
        return (
            "Agency preservation means protecting legitimate self-direction and consent, especially under pressure. "
            "Theseus should use the least sufficient power, avoid dependency traps, and keep people able to understand, contest, "
            "and redirect what it is doing."
        )
    lines = [
        f"Compact core: {core}",
        "Hard invariants: " + "; ".join(str(item) for item in hard[:3]),
        "Epistemology: " + "; ".join(str(item) for item in epistemology[:2]),
    ]
    return "\n".join(line for line in lines if line.strip())


def render_context_block(context: dict[str, Any]) -> str:
    ctx = context.get("context") if isinstance(context.get("context"), dict) else {}
    contract = ctx.get("reality_contract") if isinstance(ctx.get("reality_contract"), dict) else {}
    lines = [
        "Theseus Personality Context",
        f"Compact core: {ctx.get('compact_core', '')}",
        "",
        "Best-self contract:",
    ]
    for item in (ctx.get("best_self_contract") or [])[:8]:
        lines.append(f"- {item}")
    for section in ["values", "epistemology", "metaphysics_and_speculation", "hard_safety_invariants"]:
        items = contract.get(section) if isinstance(contract.get(section), list) else []
        if not items:
            continue
        lines.extend(["", section.replace("_", " ").title() + ":"])
        for item in items[:8]:
            lines.append(f"- {item}")
    anti = ctx.get("anti_drift_rules") if isinstance(ctx.get("anti_drift_rules"), list) else []
    if anti:
        lines.extend(["", "Anti-Drift Rules:"])
        for item in anti[:8]:
            lines.append(f"- {item}")
    cards = ctx.get("selected_cards") if isinstance(ctx.get("selected_cards"), list) else []
    if cards:
        lines.extend(["", "Relevant Memory Cards:"])
        for card in cards:
            lines.append(f"- {card.get('card_id')}: {card.get('text')}")
    return "\n".join(lines).strip()


def tokens(text: str) -> list[str]:
    return [
        word
        for word in (match.group(0).lower().strip("'") for match in re.finditer(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text))
        if word not in STOPWORDS
    ]


def sha256_text(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        pass
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        try:
            return os.path.relpath(path.resolve(), ROOT.resolve())
        except ValueError:
            return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
