"""Governed tiny-sample ingestion for approved external training data.

This script is intentionally conservative. It samples only license-approved
Hugging Face dataset rows listed in the online source catalog, keeps the raw
sample in ignored candidate storage, derives a small local pairwise training
artifact, and writes a report with gates. It does not bulk download, scrape,
or call external inference providers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "autonomy_policy.json"
DEFAULT_CATALOG = ROOT / "configs" / "online_source_catalog.json"
DEFAULT_CATALOG_REPORT = ROOT / "reports" / "online_source_catalog_report.json"
DEFAULT_OUT = ROOT / "reports" / "training_data_sampler.json"
DEFAULT_SAMPLE_ROOT = Path("D:/ProjectTheseus/training_data/governed_samples")
TEXT_KEYS = ("text", "content", "document", "article", "body", "prompt")
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_FIRST_ROWS_URL = "https://datasets-server.huggingface.co/first-rows"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG.relative_to(ROOT)))
    parser.add_argument("--catalog-report", default=str(DEFAULT_CATALOG_REPORT.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--max-docs-per-source", type=int, default=0)
    parser.add_argument("--max-total-docs", type=int, default=0)
    parser.add_argument("--max-chars-per-doc", type=int, default=0)
    parser.add_argument("--sample-root", default=str(DEFAULT_SAMPLE_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    catalog = read_json(ROOT / args.catalog)
    catalog_report = read_json(ROOT / args.catalog_report)
    ingest_policy = policy.get("training_data") or {}
    license_policy = catalog.get("license_policy") or {}
    allowed_licenses = {normal_license(item) for item in license_policy.get("allowed_data_licenses", [])}
    max_docs_per_source = args.max_docs_per_source or int(ingest_policy.get("max_docs_per_source", 64))
    max_total_docs = args.max_total_docs or int(ingest_policy.get("max_total_docs_per_cycle", 128))
    max_chars_per_doc = args.max_chars_per_doc or int(ingest_policy.get("max_chars_per_doc", 4000))
    sample_enabled = bool(ingest_policy.get("autonomous_small_samples", False))
    network_allowed = bool(args.allow_network_fetch and sample_enabled)
    sample_root = Path(args.sample_root)

    eval_fingerprint = build_eval_fingerprint()
    source_status = {
        str(item.get("id")): item
        for item in catalog_report.get("training_data_candidates", [])
        if isinstance(item, dict) and item.get("id")
    }
    samples: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for source in catalog.get("sources", []):
        if len(samples) >= max_total_docs:
            break
        if not isinstance(source, dict) or source.get("category") != "training_data":
            continue
        source_id = str(source.get("id") or "").strip()
        if not source_id:
            continue
        sample_cfg = source.get("sampling") or {}
        status = source_status.get(source_id, {})
        remaining = max(0, max_total_docs - len(samples))
        row = sample_source(
            source=source,
            status=status,
            sample_cfg=sample_cfg,
            allowed_licenses=allowed_licenses,
            max_docs=min(max_docs_per_source, remaining),
            max_chars=max_chars_per_doc,
            eval_fingerprint=eval_fingerprint,
            sample_root=sample_root,
            network_allowed=network_allowed,
            dry_run=args.dry_run,
        )
        source_reports.append(row["report"])
        samples.extend(row["samples"])
        errors.extend(row["errors"])

    pairwise_rows = build_pairwise_rows(samples, eval_fingerprint, int(ingest_policy.get("max_pairwise_rows_per_cycle", 512)))
    report = build_report(
        policy=policy,
        catalog=catalog,
        network_allowed=network_allowed,
        sample_enabled=sample_enabled,
        sources=source_reports,
        samples=samples,
        pairwise_rows=pairwise_rows,
        errors=errors,
        eval_fingerprint=eval_fingerprint,
        sample_root=sample_root,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        sample_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(sample_root / "approved_training_mix.jsonl", samples)
        write_jsonl(sample_root / "approved_pairwise_distill.jsonl", pairwise_rows)
        write_json(sample_root / "approved_training_mix.dataset_card.json", dataset_card(report))
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["training_use_allowed"] or not sample_enabled else 1


def sample_source(
    *,
    source: dict[str, Any],
    status: dict[str, Any],
    sample_cfg: dict[str, Any],
    allowed_licenses: set[str],
    max_docs: int,
    max_chars: int,
    eval_fingerprint: dict[str, set[str]],
    sample_root: Path,
    network_allowed: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_id = str(source.get("id") or "")
    license_spdx = normal_license(source.get("license_spdx"))
    sample_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    checks = [
        check("category_training_data", source.get("category") == "training_data", str(source.get("category"))),
        check("license_allowed", license_spdx in allowed_licenses, str(source.get("license_spdx"))),
        check("metadata_only_source", source.get("import_policy") == "metadata_only", str(source.get("import_policy"))),
        check("source_kind_huggingface_dataset", source.get("source_kind") == "huggingface_dataset", str(source.get("source_kind"))),
        check("catalog_staged", bool(status.get("staged", False)), str(status.get("decision", ""))),
        check("sampling_enabled", bool(sample_cfg.get("enabled", False)), json.dumps(sample_cfg, sort_keys=True)),
    ]
    if not all(item["passed"] for item in checks):
        return {
            "samples": [],
            "errors": [],
            "report": source_report(source, sample_cfg, checks, [], "skipped_gate_failed"),
        }
    if not network_allowed:
        checks.append(check("network_fetch_allowed", False, "run with --allow-network-fetch and policy training_data.autonomous_small_samples=true"))
        return {
            "samples": [],
            "errors": [],
            "report": source_report(source, sample_cfg, checks, [], "skipped_network_disabled"),
        }
    dataset = str(source.get("name") or "").strip()
    config = str(sample_cfg.get("config") or "default")
    split = str(sample_cfg.get("split") or "train")
    offset = int(sample_cfg.get("offset", 0))
    length = max(1, min(max_docs, int(sample_cfg.get("length", max_docs))))
    try:
        payload = fetch_hf_rows(dataset=dataset, config=config, split=split, offset=offset, length=length)
    except Exception as exc:  # noqa: BLE001 - report and keep autonomy moving.
        errors.append({"source": source_id, "error": str(exc), "stage": "fetch_hf_rows"})
        checks.append(check("hf_rows_fetch", False, str(exc)))
        return {"samples": [], "errors": errors, "report": source_report(source, sample_cfg, checks, [], "fetch_failed")}

    seen: set[str] = set()
    for row in payload.get("rows", []):
        raw = row.get("row") if isinstance(row, dict) else None
        if not isinstance(raw, dict):
            continue
        text = clean_text(extract_text(raw), max_chars)
        if not text or len(text) < int(sample_cfg.get("min_chars", 160)):
            continue
        text_hash = sha256_norm(text)
        if text_hash in seen or text_hash in eval_fingerprint["sentences"]:
            continue
        if has_eval_sentence_overlap(text, eval_fingerprint):
            continue
        seen.add(text_hash)
        sample_rows.append(
            {
                "source_id": source_id,
                "source_name": dataset,
                "source_url": source.get("url"),
                "license_spdx": source.get("license_spdx"),
                "dataset_config": config,
                "dataset_split": split,
                "row_idx": row.get("row_idx"),
                "text": text,
                "text_sha256": text_hash,
                "fetched_utc": now(),
                "provenance": {
                    "fetch_api": "huggingface_datasets_server_rows",
                    "url": hf_rows_url(dataset, config, split, offset, length),
                    "row_fields": sorted(raw.keys()),
                    "original_id": raw.get("id"),
                    "original_url": raw.get("url"),
                },
                "governance": {
                    "autonomous_small_sample": True,
                    "bulk_download": False,
                    "training_use": "allowed_for_low_ratio_pairwise_distill_if_report_green",
                    "external_inference_calls": 0,
                },
            }
        )
    checks.extend(
        [
            check("sample_rows_present", bool(sample_rows), f"accepted={len(sample_rows)}"),
            check("no_exact_eval_text_overlap", True, "rows rejected if exact text or sentence overlap matched"),
            check("bounded_sample_size", len(sample_rows) <= max_docs, f"accepted={len(sample_rows)} max={max_docs}"),
        ]
    )
    if not dry_run:
        out_dir = sample_root / source_id
        write_jsonl(out_dir / "sample.jsonl", sample_rows)
        write_json(
            out_dir / "provenance.json",
            {
                "source_id": source_id,
                "source_name": dataset,
                "license_spdx": source.get("license_spdx"),
                "dataset_config": config,
                "dataset_split": split,
                "sample_rows": len(sample_rows),
                "fetched_utc": now(),
                "fetch_url": hf_rows_url(dataset, config, split, offset, length),
            },
        )
    return {"samples": sample_rows, "errors": errors, "report": source_report(source, sample_cfg, checks, sample_rows, "sampled")}


def fetch_hf_rows(*, dataset: str, config: str, split: str, offset: int, length: int) -> dict[str, Any]:
    url = hf_rows_url(dataset, config, split, offset, length)
    request = urllib.request.Request(url, headers={"User-Agent": "SparkStream-RMI-governed-sampler/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code == 429 and offset == 0:
            return fetch_hf_first_rows(dataset=dataset, config=config, split=split, length=length)
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def fetch_hf_first_rows(*, dataset: str, config: str, split: str, length: int) -> dict[str, Any]:
    params = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split})
    request = urllib.request.Request(
        f"{HF_FIRST_ROWS_URL}?{params}",
        headers={"User-Agent": "SparkStream-RMI-governed-sampler/0.1"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("rows", [])
    if isinstance(rows, list):
        payload["rows"] = rows[: max(1, length)]
    return payload


def hf_rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{HF_ROWS_URL}?{params}"


def build_pairwise_rows(
    samples: list[dict[str, Any]], eval_fingerprint: dict[str, set[str]], max_rows: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    for sample in samples:
        for sentence in split_sentences(sample.get("text", "")):
            good = normalize_sentence(sentence)
            if not (8 <= len(good.split()) <= 32):
                continue
            bad, rule = corrupt_sentence(good)
            if not bad:
                continue
            pair = {
                "sentence_good": good,
                "sentence_bad": bad,
                "rule": f"external_{rule}",
                "field": "external_text_distill",
                "linguistics_term": rule,
                "source": "external_open_sample_pairwise_distill",
                "source_id": sample.get("source_id"),
                "source_name": sample.get("source_name"),
                "source_text_sha256": sample.get("text_sha256"),
                "license_spdx": sample.get("license_spdx"),
                "generation_policy": "local_rule_corruption_no_external_inference",
                "generation_strategy": "open_text_to_minimal_pair",
                "training_origin": "external_open_sample_pairwise_governed",
            }
            key = pair_key(pair)
            if key in seen_pairs or key in eval_fingerprint["pairs"]:
                continue
            if sentence_key(good) in eval_fingerprint["sentences"] or sentence_key(bad) in eval_fingerprint["sentences"]:
                continue
            seen_pairs.add(key)
            pair["external_pair_id"] = f"external_pairwise_{len(rows):06d}"
            rows.append(pair)
            if len(rows) >= max_rows:
                return rows
    return rows


def corrupt_sentence(sentence: str) -> tuple[str, str]:
    rules = [
        (r"\bis\b", "are", "agreement_aux_is_are"),
        (r"\bare\b", "is", "agreement_aux_are_is"),
        (r"\bwas\b", "were", "agreement_aux_was_were"),
        (r"\bwere\b", "was", "agreement_aux_were_was"),
        (r"\bhas\b", "have", "agreement_have_has"),
        (r"\bhave\b", "has", "agreement_have_has"),
        (r"\bthis\b", "these", "determiner_number"),
        (r"\bthese\b", "this", "determiner_number"),
        (r"\bthat\b", "those", "determiner_number"),
        (r"\bthose\b", "that", "determiner_number"),
    ]
    for pattern, replacement, rule in rules:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            bad = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
            if bad != sentence:
                return bad, rule
    return "", ""


def build_eval_fingerprint() -> dict[str, set[str]]:
    paths = [
        ROOT / "data" / "babylm_blimp_filtered_eval.jsonl",
        ROOT / "data" / "public_blimp_eval.jsonl",
        ROOT / "benchmarks" / "bridges" / "babylm_wh_gap_bridge.jsonl",
    ]
    paths.extend((ROOT / "data").glob("babylm_mutated_holdout_seed*.jsonl"))
    pairs: set[str] = set()
    sentences: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    good = normalize_sentence(row.get("sentence_good", ""))
                    bad = normalize_sentence(row.get("sentence_bad", ""))
                    if good:
                        sentences.add(sentence_key(good))
                    if bad:
                        sentences.add(sentence_key(bad))
                    if good or bad:
                        pairs.add(pair_key({"sentence_good": good, "sentence_bad": bad}))
        except OSError:
            continue
    return {"pairs": pairs, "sentences": sentences}


def has_eval_sentence_overlap(text: str, eval_fingerprint: dict[str, set[str]]) -> bool:
    for sentence in split_sentences(text):
        if sentence_key(sentence) in eval_fingerprint["sentences"]:
            return True
    return False


def extract_text(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    values = [value for value in row.values() if isinstance(value, str)]
    values.sort(key=len, reverse=True)
    return values[0] if values else ""


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [normalize_sentence(part) for part in parts if normalize_sentence(part)]


def normalize_sentence(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = text.strip(" \t\r\n\"'")
    return text


def clean_text(value: str, max_chars: int) -> str:
    text = str(value).replace("\x00", " ")
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].strip()


def build_report(
    *,
    policy: dict[str, Any],
    catalog: dict[str, Any],
    network_allowed: bool,
    sample_enabled: bool,
    sources: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    eval_fingerprint: dict[str, set[str]],
    sample_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    checks = [
        check("autonomous_small_samples_enabled", sample_enabled, str(sample_enabled)),
        check("network_fetch_allowed_for_sampler", network_allowed, str(network_allowed)),
        check("sample_root_on_d_drive", str(sample_root).replace("\\", "/").lower().startswith("d:/"), rel(sample_root)),
        check("source_samples_present", bool(samples), f"samples={len(samples)}"),
        check("pairwise_rows_present", bool(pairwise_rows), f"pairs={len(pairwise_rows)}"),
        check("no_sampler_errors", not errors, f"errors={len(errors)}"),
    ]
    training_ok = all(item["passed"] for item in checks)
    return {
        "policy": "sparkstream_training_data_sampler_v0",
        "created_utc": now(),
        "dry_run": dry_run,
        "network_fetch_allowed": network_allowed,
        "external_inference_calls": 0,
        "training_use_allowed": training_ok,
        "usage_policy": {
            "allowed_for_training": training_ok,
            "allowed_profiles": ["smoke", "inner_loop"] if training_ok else [],
            "candidate_profile_requires_human_or_teacher_review": True,
            "bulk_download": False,
            "max_external_pair_ratio": get_path(policy, ["training_data", "max_external_pair_ratio"], 0.02),
        },
        "summary": {
            "sample_root": rel(sample_root),
            "sample_sources": len([source for source in sources if source.get("accepted_count", 0) > 0]),
            "sample_rows": len(samples),
            "pairwise_rows": len(pairwise_rows),
            "eval_pair_fingerprint_size": len(eval_fingerprint["pairs"]),
            "eval_sentence_fingerprint_size": len(eval_fingerprint["sentences"]),
            "errors": len(errors),
        },
        "checks": checks,
        "sources": sources,
        "errors": errors,
        "artifacts": {
            "approved_training_mix_jsonl": rel(sample_root / "approved_training_mix.jsonl"),
            "pairwise_training_jsonl": rel(sample_root / "approved_pairwise_distill.jsonl"),
            "dataset_card": rel(sample_root / "approved_training_mix.dataset_card.json"),
            "catalog": rel(ROOT / "configs" / "online_source_catalog.json"),
        },
        "governance": {
            "allowed_data_licenses": (catalog.get("license_policy") or {}).get("allowed_data_licenses", []),
            "knowledge_sources_lookup_only": True,
            "commercial_roms_forbidden": True,
            "teacher_generation_used": False,
        },
    }


def source_report(
    source: dict[str, Any],
    sample_cfg: dict[str, Any],
    checks: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    return {
        "id": source.get("id"),
        "name": source.get("name"),
        "status": status,
        "license_spdx": source.get("license_spdx"),
        "sampling": sample_cfg,
        "accepted_count": len(samples),
        "checks": checks,
    }


def dataset_card(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_card": "sparkstream_external_training_tiny_samples_v0",
        "created_utc": report.get("created_utc"),
        "purpose": "Tiny governed open-license sample for low-ratio local pairwise training augmentation.",
        "artifacts": report.get("artifacts"),
        "summary": report.get("summary"),
        "usage_policy": report.get("usage_policy"),
        "governance": report.get("governance"),
        "risks": [
            "Samples are web/open corpus slices and may contain noise.",
            "Use only at low ratio unless a stronger sampling plan and review pass.",
            "Not a private holdout and never an evaluation source.",
        ],
    }


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def pair_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        "\n".join([normalize_sentence(row.get("sentence_good", "")), normalize_sentence(row.get("sentence_bad", ""))])
        .lower()
        .encode("utf-8")
    ).hexdigest()


def sentence_key(sentence: Any) -> str:
    return hashlib.sha256(normalize_sentence(sentence).lower().encode("utf-8")).hexdigest()


def sha256_norm(text: str) -> str:
    return hashlib.sha256(normalize_sentence(text).lower().encode("utf-8")).hexdigest()


def normal_license(value: Any) -> str:
    return str(value or "").strip().lower()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def get_path(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
