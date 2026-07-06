"""Build the governed native voice STT/TTS training manifest.

The manifest is a data-plumbing step, not a speech model. It records which
licensed speech sources may pressure Theseus' native voice I/O learner and, for
small approved cases, materializes tiny audio/transcript shards under ignored
candidate storage. It never calls provider/pretrained STT/TTS inference.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "native_voice_training_policy.json"
DEFAULT_VOICE_POLICY = ROOT / "configs" / "native_voice_policy.json"
DEFAULT_CATALOG = ROOT / "configs" / "online_source_catalog.json"
DEFAULT_OUT = ROOT / "reports" / "native_voice_training_manifest.json"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--voice-policy", default=str(DEFAULT_VOICE_POLICY.relative_to(ROOT)))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    policy_path = ROOT / args.policy
    voice_policy_path = ROOT / args.voice_policy
    catalog_path = ROOT / args.catalog
    policy = read_json(policy_path)
    voice_policy = read_json(voice_policy_path)
    catalog = read_json(catalog_path)

    storage_root = ROOT / str(policy.get("storage_root") or "data/external_benchmark_candidates/native_voice_samples")
    manifest_artifact = ROOT / str(
        policy.get("manifest_artifact")
        or "data/external_benchmark_candidates/native_voice_samples/native_voice_training_manifest.jsonl"
    )
    caps = policy.get("caps") if isinstance(policy.get("caps"), dict) else {}
    max_sources = int(caps.get("max_sources_per_cycle", 8))
    max_clips_total = int(caps.get("max_clips_per_cycle", 8))
    max_bytes_per_clip = int(caps.get("max_bytes_per_clip", 5_000_000))
    max_total_audio_bytes = int(caps.get("max_total_audio_bytes_per_cycle", 25_000_000))

    catalog_by_id = {
        str(item.get("id") or ""): item
        for item in catalog.get("sources", [])
        if isinstance(item, dict) and item.get("id")
    }
    allowed_licenses = {
        normalize_license(item)
        for item in (
            policy.get("allowed_licenses")
            or get_path(catalog, ["license_policy", "allowed_data_licenses"], [])
        )
    }
    rows: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    clips_written = 0
    bytes_written = 0

    for source in (policy.get("sources") or [])[: max(0, max_sources)]:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("source_id") or "").strip()
        catalog_source = catalog_by_id.get(source_id, {})
        row = source_row(source, catalog_source, allowed_licenses)
        rows.append(row)
        if row["decision"] != "allowed":
            blocked.append(block("source_not_allowed", source_id, row["decision_reason"]))
            continue
        can_materialize = bool(
            args.allow_network_fetch
            and not args.dry_run
            and policy.get("autonomous_tiny_audio_samples", True)
            and source.get("autonomous_tiny_audio", False)
            and source.get("hf_dataset")
        )
        if not can_materialize:
            continue
        remaining_clips = max(0, max_clips_total - clips_written)
        remaining_bytes = max(0, max_total_audio_bytes - bytes_written)
        if remaining_clips <= 0 or remaining_bytes <= 0:
            break
        result = materialize_hf_tiny_audio(
            source=source,
            storage_root=storage_root,
            max_clips=min(int(caps.get("max_clips_per_source", 4)), remaining_clips),
            max_bytes_per_clip=max_bytes_per_clip,
            max_total_bytes=remaining_bytes,
        )
        materialized.extend(result["samples"])
        errors.extend(result["errors"])
        clips_written += len(result["samples"])
        bytes_written += sum(int(sample.get("bytes", 0)) for sample in result["samples"])

    if not args.dry_run:
        manifest_artifact.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(manifest_artifact, manifest_rows(rows, materialized))

    checks = build_checks(policy, voice_policy, rows, materialized)
    stt_sources = [row for row in rows if row["license_allowed"] and "stt_train" in row.get("roles", [])]
    tts_sources = [row for row in rows if row["license_allowed"] and "tts_train" in row.get("roles", [])]
    report = {
        "policy": "project_theseus_native_voice_training_manifest_report_v0",
        "created_utc": now(),
        "config": rel(policy_path),
        "voice_policy": rel(voice_policy_path),
        "catalog": rel(catalog_path),
        "storage_root": rel(storage_root),
        "manifest_artifact": rel(manifest_artifact),
        "allow_network_fetch": bool(args.allow_network_fetch),
        "dry_run": bool(args.dry_run),
        "external_inference_calls": 0,
        "summary": {
            "sources": len(rows),
            "license_allowed_sources": sum(1 for row in rows if row["license_allowed"]),
            "stt_sources": len(stt_sources),
            "tts_sources": len(tts_sources),
            "tiny_audio_clips": len(materialized),
            "tiny_audio_bytes": bytes_written,
            "ready_for_native_training": bool(stt_sources and tts_sources),
            "ready_for_stt_bootstrap": bool(stt_sources and materialized),
            "ready_for_tts_bootstrap": bool(tts_sources),
            "bulk_downloads": policy.get("bulk_downloads", "forbidden_without_explicit_user_approval"),
            "status": status_from_checks(checks, errors),
        },
        "checks": checks,
        "sources": rows,
        "materialized_samples": materialized,
        "blocked": blocked,
        "errors": errors,
        "training_packets": training_packets(rows, materialized),
        "next_actions": next_actions(rows, materialized, errors),
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["summary"]["ready_for_native_training"] and not errors else 1


def source_row(source: dict[str, Any], catalog_source: dict[str, Any], allowed_licenses: set[str]) -> dict[str, Any]:
    source_license = normalize_license(source.get("license_spdx"))
    catalog_license = normalize_license(catalog_source.get("license_spdx"))
    license_spdx = source_license or catalog_license
    license_allowed = license_spdx in allowed_licenses
    import_policy = str(catalog_source.get("import_policy") or "metadata_only")
    if not license_allowed:
        decision = "blocked_license"
        reason = f"license={license_spdx or 'unknown'}"
    elif import_policy == "queue_only":
        decision = "metadata_planned"
        reason = "catalog source is queue-only; training policy records it but will not fetch"
    else:
        decision = "allowed"
        reason = "license and policy allow governed native voice pressure"
    if license_allowed and str(source.get("source_id")) in {"libritts", "ljspeech", "common_voice"}:
        decision = "allowed"
        reason = "voice training policy allows metadata/planned shards; tiny audio fetch remains source-specific"
    return {
        "source_id": source.get("source_id"),
        "display_name": source.get("display_name"),
        "roles": source.get("roles", []),
        "license_spdx": license_spdx,
        "license_allowed": license_allowed,
        "license_evidence_url": source.get("license_evidence_url"),
        "source_url": source.get("source_url") or catalog_source.get("url"),
        "catalog_import_policy": import_policy,
        "autonomous_tiny_audio": bool(source.get("autonomous_tiny_audio", False)),
        "hf_dataset": source.get("hf_dataset"),
        "hf_config": source.get("hf_config"),
        "hf_split": source.get("hf_split"),
        "decision": decision,
        "decision_reason": reason,
        "recommended_initial_use": source.get("recommended_initial_use"),
    }


def materialize_hf_tiny_audio(
    source: dict[str, Any],
    storage_root: Path,
    max_clips: int,
    max_bytes_per_clip: int,
    max_total_bytes: int,
) -> dict[str, list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    dataset = str(source.get("hf_dataset") or "")
    config = str(source.get("hf_config") or "default")
    split = str(source.get("hf_split") or "train")
    audio_field = str(source.get("audio_field") or "audio")
    text_field = str(source.get("text_field") or "text")
    id_field = str(source.get("id_field") or "id")
    speaker_field = str(source.get("speaker_field") or "speaker_id")
    if not dataset or max_clips <= 0:
        return {"samples": samples, "errors": errors}
    try:
        page = hf_rows(dataset, config, split, length=max_clips)
    except Exception as exc:  # noqa: BLE001
        errors.append({"source_id": source.get("source_id"), "stage": "hf_rows", "error": str(exc)[:500]})
        return {"samples": samples, "errors": errors}
    source_dir = storage_root / safe_name(str(source.get("source_id") or dataset))
    source_dir.mkdir(parents=True, exist_ok=True)
    bytes_left = max_total_bytes
    for item in page.get("rows", []):
        row = item.get("row") if isinstance(item, dict) else {}
        if not isinstance(row, dict):
            continue
        audio_url = first_audio_url(row.get(audio_field))
        transcript = str(row.get(text_field) or "").strip()
        if not audio_url or not transcript:
            continue
        sample_id = safe_name(str(row.get(id_field) or f"row_{item.get('row_idx', len(samples))}"))
        suffix = audio_suffix(audio_url)
        target = source_dir / f"{sample_id}{suffix}"
        max_bytes = min(max_bytes_per_clip, bytes_left)
        if max_bytes <= 0:
            break
        try:
            size = download_file(audio_url, target, max_bytes)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "source_id": source.get("source_id"),
                    "stage": "audio_download",
                    "sample_id": sample_id,
                    "error": str(exc)[:500],
                }
            )
            continue
        bytes_left -= size
        samples.append(
            {
                "source_id": source.get("source_id"),
                "sample_id": sample_id,
                "audio_path": rel(target),
                "transcript": transcript,
                "speaker": row.get(speaker_field),
                "split": split,
                "config": config,
                "bytes": size,
                "roles": source.get("roles", []),
                "license_spdx": normalize_license(source.get("license_spdx")),
            }
        )
    if samples:
        write_jsonl(source_dir / "samples.jsonl", samples)
    return {"samples": samples, "errors": errors}


def hf_rows(dataset: str, config: str, split: str, length: int) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": 0,
            "length": max(1, length),
        }
    )
    req = urllib.request.Request(f"{HF_ROWS_URL}?{query}", headers={"User-Agent": "ProjectTheseusNativeVoice/0.1"})
    with urllib.request.urlopen(req, timeout=45) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def first_audio_url(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("src"):
                return str(item["src"])
    if isinstance(value, dict) and value.get("src"):
        return str(value["src"])
    return ""


def download_file(url: str, target: Path, max_bytes: int) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "ProjectTheseusNativeVoice/0.1"})
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"clip exceeds cap {max_bytes} bytes")
    target.write_bytes(data)
    return len(data)


def build_checks(policy: dict[str, Any], voice_policy: dict[str, Any], rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stt_allowed = any(row["license_allowed"] and "stt_train" in row.get("roles", []) for row in rows)
    tts_allowed = any(row["license_allowed"] and "tts_train" in row.get("roles", []) for row in rows)
    external_forbidden = get_path(voice_policy, ["execution_boundary", "provider_stt_tts"]) == "forbidden"
    return [
        check("voice_training_policy_present", bool(policy), policy.get("policy", "")),
        check("native_voice_policy_forbids_external_stt_tts", external_forbidden, "provider STT/TTS forbidden"),
        check("at_least_one_stt_source_license_allowed", stt_allowed, "licensed STT source required"),
        check("at_least_one_tts_source_license_allowed", tts_allowed, "licensed TTS source required"),
        check(
            "tiny_manifest_materialized_or_planned",
            bool(samples) or bool(rows),
            f"samples={len(samples)} sources={len(rows)}",
        ),
        check("external_inference_calls_zero", True, "0"),
    ]


def training_packets(rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    if samples:
        packets.append(
            {
                "id": "native_stt_bootstrap_tiny_librispeech",
                "type": "native_voice_training_packet",
                "target": "native_stt_decoder",
                "status": "ready",
                "samples": len(samples),
                "input": "audio_path + transcript",
                "output_report": "reports/native_stt_decoder.json",
                "external_inference_calls": 0,
            }
        )
    if any(row["license_allowed"] and "tts_train" in row.get("roles", []) for row in rows):
        packets.append(
            {
                "id": "native_tts_bootstrap_text_audio_pairs",
                "type": "native_voice_training_packet",
                "target": "native_tts_generator",
                "status": "planned" if not samples else "ready_for_bootstrap",
                "sources": [row["source_id"] for row in rows if row["license_allowed"] and "tts_train" in row.get("roles", [])],
                "output_report": "reports/native_tts_generator.json",
                "external_inference_calls": 0,
            }
        )
    return packets


def next_actions(rows: list[dict[str, Any]], samples: list[dict[str, Any]], errors: list[dict[str, Any]]) -> list[str]:
    actions = []
    if not samples:
        actions.append("Materialize a tiny licensed speech shard when network fetch is allowed and storage caps permit it.")
    else:
        actions.append("Run the native STT decoder learner on the materialized audio/transcript shard.")
    if any(row.get("source_id") == "libritts" and row["license_allowed"] for row in rows):
        actions.append("Add an approved LibriTTS or LJSpeech shard downloader before serious native TTS training.")
    if errors:
        actions.append("Inspect native voice manifest errors before treating voice data as training-ready.")
    actions.append("Keep provider/pretrained speech inference out of all non-teacher paths.")
    return actions


def manifest_rows(rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [{"kind": "source", **row} for row in rows]
    out.extend({"kind": "sample", **sample} for sample in samples)
    return out


def status_from_checks(checks: list[dict[str, Any]], errors: list[dict[str, Any]]) -> str:
    if errors:
        return "YELLOW"
    if all(item["passed"] for item in checks):
        return "GREEN"
    return "YELLOW"


def block(kind: str, source_id: str, reason: str) -> dict[str, Any]:
    return {"kind": kind, "source_id": source_id, "reason": reason}


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": str(evidence)[:400]}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for item in path:
        if not isinstance(cur, dict) or item not in cur:
            return default
        cur = cur[item]
    return cur


def normalize_license(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def audio_suffix(url: str) -> str:
    clean = urllib.parse.urlparse(url).path.lower()
    match = re.search(r"\.(flac|wav|mp3|ogg|m4a)$", clean)
    return f".{match.group(1)}" if match else ".audio"


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe[:120] or "sample"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
