"""Source freshness helpers for Code LM decoder artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECODER_SOURCE = ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure.rs"
DECODER_SOURCE_DIR = ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure"
DECODER_FINGERPRINT_MARKERS = (
    "semantic_decoder_v2",
    "execution_shape_skeleton",
    "edge_exec_repair",
    "typed_edge_exec_receiver",
    "decoder_contract",
    "contract_guided_skeleton",
    "contract_guided_token",
    "local_adapter_edge_skeleton",
    "sts_causal_skeleton",
    "candidate_floor",
    "body_token_allowed",
    "syntax_constrained_body",
    "invalid_inline_block_header_body",
    "callable_keyword_argument",
    "archive_context_manager",
    "invalid_overcomposed_generated_line",
)


def decoder_source_paths() -> list[Path]:
    paths = [DECODER_SOURCE]
    if DECODER_SOURCE_DIR.exists():
        paths.extend(sorted(DECODER_SOURCE_DIR.glob("*.rs")))
    return paths


def decoder_relevant_source_fingerprint() -> str:
    chunks: list[str] = []
    for path in decoder_source_paths():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        chunks.extend(line for line in text.splitlines() if any(marker in line for marker in DECODER_FINGERPRINT_MARKERS))
    return hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest()[:16]


def decoder_relevant_source_mtime() -> float:
    mtimes = [path.stat().st_mtime for path in decoder_source_paths() if path.exists()]
    return max(mtimes) if mtimes else 0.0
