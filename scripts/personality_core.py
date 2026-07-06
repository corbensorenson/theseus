"""Build a governed personality core from local user-owned documents.

The core is intentionally layered: source inventory, distilled values/voice,
retrieval cards, and a future training manifest. It is not a giant prompt and
it does not call external inference.
"""

from __future__ import annotations

import argparse
import fnmatch
import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "personality_core_policy.json"

STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "all",
    "also",
    "and",
    "any",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "don",
    "each",
    "every",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "him",
    "his",
    "how",
    "into",
    "its",
    "just",
    "like",
    "more",
    "most",
    "must",
    "not",
    "one",
    "only",
    "other",
    "our",
    "out",
    "over",
    "own",
    "should",
    "same",
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
    "isn",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--max-snippets", type=int, default=0)
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    limits = policy.get("limits") if isinstance(policy.get("limits"), dict) else {}
    max_snippets = args.max_snippets or int(limits.get("max_snippets", 1800))
    report = build_personality_core(policy, max_snippets=max_snippets)

    out = ROOT / (args.out or policy.get("report") or "reports/personality_core.json")
    markdown_out = ROOT / (args.markdown_out or policy.get("markdown_report") or "reports/personality_core.md")
    manifest_out = ROOT / (args.manifest_out or policy.get("manifest_report") or "reports/personality_core_training_manifest.jsonl")
    write_json(out, report)
    write_text(markdown_out, render_markdown(report))
    write_jsonl(manifest_out, build_manifest(report, int(limits.get("max_manifest_rows", 500))))
    print(json.dumps(report, indent=2))
    return 0


def build_personality_core(policy: dict[str, Any], *, max_snippets: int) -> dict[str, Any]:
    root = ROOT / str(policy.get("root") or "personality-documents")
    limits = policy.get("limits") if isinstance(policy.get("limits"), dict) else {}
    max_cards = int(limits.get("max_retrieval_cards", 96))
    if not root.exists():
        return {
            "policy": "sparkstream_personality_core_v0",
            "updated_utc": now(),
            "root": rel(root),
            "status": "missing_source_root",
            "summary": {"documents_scanned": 0, "snippets": 0},
        }

    snippets: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    excluded_limit = int(limits.get("excluded_sample_limit", 120))
    per_source_limit = int(limits.get("max_snippets_per_source", 280))

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        decision = classify_source(path, root, policy)
        row = {
            "path": rel(path),
            "bytes": path.stat().st_size,
            "source_kind": decision["source_kind"],
            "included": decision["included"],
            "reason": decision["reason"],
        }
        file_rows.append(row)
        if not decision["included"]:
            if len(excluded) < excluded_limit:
                excluded.append({"path": rel(path), "reason": decision["reason"]})
            continue
        parsed = parse_source(path, decision["source_kind"], policy)
        for item in parsed[:per_source_limit]:
            item["source_path"] = rel(path)
            item["source_kind"] = decision["source_kind"]
            snippets.append(item)
            if len(snippets) >= max_snippets:
                break
        if len(snippets) >= max_snippets:
            break

    term_counts = Counter()
    source_counts: Counter[str] = Counter()
    for item in snippets:
        source_counts[item["source_kind"]] += 1
        term_counts.update(tokens(item.get("text", "")))

    value_profile = score_values(snippets, policy)
    voice_profile = build_voice_profile(snippets)
    cards = build_retrieval_cards(snippets, policy, max_cards=max_cards)
    suppressed = [item for item in snippets if not item.get("activation_eligible", True)]
    raw_bytes = sum(int(row["bytes"]) for row in file_rows)
    included_files = [row for row in file_rows if row["included"]]

    return {
        "policy": "sparkstream_personality_core_v0",
        "updated_utc": now(),
        "status": "ready",
        "root": rel(root),
        "source_policy": policy.get("source_policy", {}),
        "summary": {
            "documents_scanned": len(file_rows),
            "documents_used": len(included_files),
            "raw_bytes_scanned": raw_bytes,
            "snippets": len(snippets),
            "activation_eligible_snippets": len(snippets) - len(suppressed),
            "suppressed_snippets": len(suppressed),
            "source_counts": dict(source_counts),
            "top_terms": [{"term": term, "count": count} for term, count in term_counts.most_common(30)],
        },
        "personality_core": {
            "compact_core": compact_core(policy),
            "best_self_contract": policy.get("best_self_contract", []),
            "value_profile": value_profile,
            "voice_profile": voice_profile,
            "reality_contract": policy.get("reality_contract", {}),
            "anti_drift_rules": policy.get("anti_drift_rules", []),
            "runtime_guidance": policy.get("runtime_guidance", []),
        },
        "memory_stream": {
            "retrieval_cards": cards,
            "card_count": len(cards),
            "selection": "top scored activation-eligible snippets, URL-redacted and length-capped",
        },
        "adaptation_plan": policy.get("adaptation_lanes", []),
        "files": file_rows[:400],
        "excluded_samples": excluded,
        "privacy": {
            "direct_messages_included": False,
            "ad_engagements_included": False,
            "grok_chats_included": False,
            "raw_source_committed_policy": "source files are user-supplied; generated core reports stay under ignored reports/",
        },
    }


def classify_source(path: Path, root: Path, policy: dict[str, Any]) -> dict[str, Any]:
    rel_path = rel(path).replace("\\", "/")
    rel_lower = rel_path.lower()
    for pattern in policy.get("exclude_path_globs", []):
        if fnmatch.fnmatch(rel_lower, str(pattern).lower()):
            return {"included": False, "source_kind": "excluded", "reason": f"excluded_by_policy:{pattern}"}
    suffix = path.suffix.lower()
    if suffix in set(policy.get("include_text_extensions", [".md", ".txt"])):
        if "/data/" in rel_lower and rel_lower.endswith("/readme.txt"):
            return {"included": False, "source_kind": "twitter_archive_metadata", "reason": "archive_metadata_not_voice"}
        return {"included": True, "source_kind": "document_text", "reason": "text_document"}
    if suffix == ".js" and path.name in set(policy.get("include_twitter_archive_files", [])):
        return {"included": True, "source_kind": f"twitter_archive:{path.name}", "reason": "allowed_twitter_archive_file"}
    return {"included": False, "source_kind": "unsupported", "reason": "not_a_supported_personality_source"}


def parse_source(path: Path, source_kind: str, policy: dict[str, Any]) -> list[dict[str, Any]]:
    if source_kind == "document_text":
        return parse_text_document(path, policy)
    if source_kind.endswith("tweets.js"):
        return parse_tweets(path, policy)
    if source_kind.endswith("note-tweet.js"):
        return parse_note_tweets(path, policy)
    if source_kind.endswith("article.js"):
        return parse_articles(path, policy)
    return []


def parse_text_document(path: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
    text = clean_text(read_text(path))
    rows = []
    for index, chunk in enumerate(chunk_text(text, policy)):
        rows.append(record(chunk, source_id=f"paragraph:{index}", created_at="", policy=policy))
    return rows


def parse_tweets(path: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    payload = load_twitter_js(path)
    if not isinstance(payload, list):
        return rows
    for item in payload:
        tweet = item.get("tweet") if isinstance(item, dict) else None
        if not isinstance(tweet, dict):
            continue
        text = str(tweet.get("full_text") or "")
        if tweet.get("retweeted") or text.startswith("RT @"):
            continue
        rows.extend(
            record(
                chunk,
                source_id=str(tweet.get("id_str") or tweet.get("id") or ""),
                created_at=str(tweet.get("created_at") or ""),
                policy=policy,
            )
            for chunk in chunk_text(text, policy)
        )
    return rows


def parse_note_tweets(path: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    payload = load_twitter_js(path)
    if not isinstance(payload, list):
        return rows
    for item in payload:
        note = item.get("noteTweet") if isinstance(item, dict) else None
        core = note.get("core") if isinstance(note, dict) else None
        if not isinstance(core, dict):
            continue
        text = str(core.get("text") or "")
        rows.extend(
            record(
                chunk,
                source_id=str(note.get("noteTweetId") or ""),
                created_at=str(note.get("createdAt") or ""),
                policy=policy,
            )
            for chunk in chunk_text(text, policy)
        )
    return rows


def parse_articles(path: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    payload = load_twitter_js(path)
    if not isinstance(payload, list):
        return rows
    for item in payload:
        article = item.get("article") if isinstance(item, dict) else None
        content = article.get("content") if isinstance(article, dict) else None
        blocks = content.get("blocks") if isinstance(content, dict) else None
        if not isinstance(blocks, list):
            continue
        text = "\n\n".join(str(block.get("text") or "") for block in blocks if isinstance(block, dict))
        rows.extend(
            record(
                chunk,
                source_id=str(article.get("id") or ""),
                created_at=str(article.get("createdAt") or ""),
                policy=policy,
            )
            for chunk in chunk_text(text, policy)
        )
    return rows


def record(text: str, *, source_id: str, created_at: str, policy: dict[str, Any]) -> dict[str, Any]:
    clean = clean_text(text)
    eligible, reasons = activation_eligibility(clean, policy)
    return {
        "source_id": source_id,
        "created_at": created_at,
        "text": clean,
        "chars": len(clean),
        "activation_eligible": eligible,
        "suppression_reasons": reasons,
        "score": score_text(clean, eligible),
    }


def activation_eligibility(text: str, policy: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    lower = text.lower()
    privacy = policy.get("privacy") if isinstance(policy.get("privacy"), dict) else {}
    for pattern in privacy.get("sensitive_topic_patterns", []):
        try:
            if re.search(str(pattern), text):
                reasons.append("sensitive_or_harmful_tactical_fragment")
                break
        except re.error:
            continue
    if re.search(r"\b(api[_ -]?key|authorization:|bearer\s+|private key|password|secret|token)\b", lower):
        reasons.append("secret_like_text")
    if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.I):
        reasons.append("email_like_text")
    if re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text):
        reasons.append("ip_address_like_text")
    return (not reasons, reasons)


def load_twitter_js(path: Path) -> Any:
    text = read_text(path)
    equals = text.find("=")
    if equals == -1:
        return []
    payload = text[equals + 1 :].strip()
    if payload.endswith(";"):
        payload = payload[:-1].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return []


def chunk_text(text: str, policy: dict[str, Any]) -> list[str]:
    limits = policy.get("limits") if isinstance(policy.get("limits"), dict) else {}
    min_chars = int(limits.get("min_snippet_chars", 80))
    max_chars = int(limits.get("max_snippet_chars", 1400))
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = clean_text(para)
        if len(para) < min_chars:
            continue
        if len(para) <= max_chars:
            if not is_blocked_text(para, policy):
                chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for sentence in sentences:
            if not sentence:
                continue
            candidate = f"{buf} {sentence}".strip()
            if len(candidate) <= max_chars:
                buf = candidate
                continue
            if len(buf) >= min_chars and not is_blocked_text(buf, policy):
                chunks.append(buf)
            buf = sentence[:max_chars]
        if len(buf) >= min_chars and not is_blocked_text(buf, policy):
            chunks.append(buf)
    return chunks


def is_blocked_text(text: str, policy: dict[str, Any]) -> bool:
    privacy = policy.get("privacy") if isinstance(policy.get("privacy"), dict) else {}
    for pattern in privacy.get("blocked_text_patterns", []):
        try:
            if re.search(str(pattern), text):
                return True
        except re.error:
            continue
    return False


def build_retrieval_cards(snippets: list[dict[str, Any]], policy: dict[str, Any], *, max_cards: int) -> list[dict[str, Any]]:
    limits = policy.get("limits") if isinstance(policy.get("limits"), dict) else {}
    card_chars = int(limits.get("retrieval_card_chars", 700))
    ranked = sorted(
        (item for item in snippets if item.get("activation_eligible", True)),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    cards = []
    for index, item in enumerate(ranked[:max_cards], start=1):
        text = redact_snippet(str(item.get("text", "")), policy)[:card_chars].strip()
        cards.append(
            {
                "card_id": f"pcore_{index:04d}",
                "source_path": item.get("source_path"),
                "source_kind": item.get("source_kind"),
                "source_id": item.get("source_id"),
                "created_at": item.get("created_at"),
                "score": round(float(item.get("score", 0.0)), 4),
                "text": text,
                "terms": tokens(text)[:16],
            }
        )
    return cards


def build_manifest(report: dict[str, Any], max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    core = report.get("personality_core") if isinstance(report.get("personality_core"), dict) else {}
    contract = core.get("best_self_contract") or []
    for card in get_path(report, ["memory_stream", "retrieval_cards"], [])[:max_rows]:
        if not isinstance(card, dict):
            continue
        rows.append(
            {
                "kind": "personality_retrieval_card",
                "source_path": card.get("source_path"),
                "source_kind": card.get("source_kind"),
                "source_id": card.get("source_id"),
                "input": "Use this local memory only when relevant to the user, while preserving truth, agency, and safety.",
                "target": card.get("text"),
                "best_self_contract": contract,
                "license_spdx": "local-user-supplied-private",
                "training_use_state": "manifest_only_pending_user_approval",
            }
        )
    return rows


def score_values(snippets: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    lexicon = policy.get("value_lexicon") if isinstance(policy.get("value_lexicon"), dict) else {}
    joined = "\n".join(str(item.get("text", "")).lower() for item in snippets)
    scores = []
    for name, words in lexicon.items():
        count = 0
        for word in words if isinstance(words, list) else []:
            count += joined.count(str(word).lower())
        scores.append({"value": name, "score": count})
    scores.sort(key=lambda item: (-int(item["score"]), item["value"]))
    return scores


def build_voice_profile(snippets: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(item.get("text", "")) for item in snippets if item.get("activation_eligible", True)]
    joined = "\n".join(texts)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if s.strip()]
    sentence_lengths = [len(tokens(sentence, keep_stopwords=True)) for sentence in sentences]
    first_person = len(re.findall(r"\b(i|me|my|mine|myself|we|our|ours)\b", joined, flags=re.I))
    second_person = len(re.findall(r"\b(you|your|yours|yourself)\b", joined, flags=re.I))
    word_count = max(1, len(tokens(joined, keep_stopwords=True)))
    return {
        "eligible_text_count": len(texts),
        "sentence_count": len(sentences),
        "avg_sentence_words": round(sum(sentence_lengths) / len(sentence_lengths), 2) if sentence_lengths else 0,
        "median_sentence_words": median(sentence_lengths) if sentence_lengths else 0,
        "question_rate": round(joined.count("?") / max(1, len(sentences)), 4),
        "exclamation_rate": round(joined.count("!") / max(1, len(sentences)), 4),
        "first_person_mentions_per_1k_words": round(first_person * 1000 / word_count, 2),
        "second_person_mentions_per_1k_words": round(second_person * 1000 / word_count, 2),
        "style_observations": [
            "Often uses compact maxims and operational rules.",
            "Connects ethics, agency, systems, faith, and practical action.",
            "Prefers direct imperatives and cause-effect reasoning.",
            "Uses long-form constitutional/specification language for alignment topics.",
        ],
    }


def score_text(text: str, eligible: bool) -> float:
    lower = text.lower()
    score = min(len(text) / 800.0, 2.0)
    for marker in ["truth", "agency", "wisdom", "coherence", "responsibility", "family", "build", "learn", "calm", "gentle"]:
        if marker in lower:
            score += 0.45
    if re.search(r"\b(i|my|we|our)\b", lower):
        score += 0.2
    if not eligible:
        score *= 0.05
    return score


def compact_core(policy: dict[str, Any]) -> str:
    contract = policy.get("best_self_contract") or []
    if not contract:
        return "Truth-grounded, agency-preserving, responsible, warm, and corrigible."
    return " ".join(str(item).strip() for item in contract[:4])


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    core = report.get("personality_core") or {}
    values = core.get("value_profile") or []
    lines = [
        "# Theseus Personality Core",
        "",
        f"Updated UTC: {report.get('updated_utc', '')}",
        f"Status: {report.get('status', '')}",
        "",
        "## Summary",
        "",
        f"- Documents scanned: {summary.get('documents_scanned', 0)}",
        f"- Documents used: {summary.get('documents_used', 0)}",
        f"- Snippets: {summary.get('snippets', 0)}",
        f"- Activation eligible snippets: {summary.get('activation_eligible_snippets', 0)}",
        f"- Suppressed snippets: {summary.get('suppressed_snippets', 0)}",
        "",
        "## Compact Core",
        "",
        str(core.get("compact_core", "")),
        "",
        "## Top Values",
        "",
    ]
    for item in values[:10]:
        lines.append(f"- {item.get('value')}: {item.get('score')}")
    lines.extend(
        [
            "",
            "## Reality Contract",
            "",
        ]
    )
    contract = core.get("reality_contract") if isinstance(core.get("reality_contract"), dict) else {}
    for section, items in contract.items():
        title = str(section).replace("_", " ").title()
        lines.append(f"### {title}")
        lines.append("")
        for item in items if isinstance(items, list) else []:
            lines.append(f"- {item}")
        lines.append("")
    lines.extend(
        [
            "## Runtime Guidance",
            "",
        ]
    )
    for item in core.get("runtime_guidance") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Adaptation Plan",
            "",
        ]
    )
    for item in report.get("adaptation_plan") or []:
        if not isinstance(item, dict):
            continue
        lines.append(f"- {item.get('lane')}: {item.get('status')} - {item.get('description')}")
    lines.append("")
    return "\n".join(lines)


def clean_text(text: str) -> str:
    text = repair_mojibake(html.unescape(str(text)))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
    return text


def repair_mojibake(text: str) -> str:
    if not any(marker in text for marker in ["Ã", "â", "ð"]):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text
    if weirdness(repaired) < weirdness(text):
        return repaired
    return text


def weirdness(text: str) -> int:
    return sum(text.count(marker) for marker in ["Ã", "â", "�", "ð"])


def redact_snippet(text: str, policy: dict[str, Any]) -> str:
    privacy = policy.get("privacy") if isinstance(policy.get("privacy"), dict) else {}
    redacted = text
    if privacy.get("redact_urls", True):
        redacted = re.sub(r"https?://\S+", "[url]", redacted)
    redacted = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "[email]", redacted, flags=re.I)
    redacted = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "[ip]", redacted)
    return redacted


def tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    words = [word.lower().strip("'") for word in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text)]
    if keep_stopwords:
        return words
    return [word for word in words if word not in STOPWORDS and len(word) > 2]


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


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
