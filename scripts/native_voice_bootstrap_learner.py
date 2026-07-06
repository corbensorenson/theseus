"""Bootstrap native Theseus STT/TTS component reports from the voice manifest.

This is intentionally a tiny local learner/indexer, not a general speech model.
It consumes materialized licensed audio/transcript samples, builds deterministic
local indexes, and writes component reports that future acoustic/codec learners
can replace. It uses zero external inference and does not import STT/TTS
packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/native_voice_training_manifest.json")
    parser.add_argument("--artifact-dir", default="data/external_benchmark_candidates/native_voice_samples/models")
    parser.add_argument("--stt-out", default="reports/native_stt_decoder.json")
    parser.add_argument("--tts-out", default="reports/native_tts_generator.json")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    artifact_dir = ROOT / args.artifact_dir
    manifest = read_json(manifest_path)
    samples = [
        item
        for item in manifest.get("materialized_samples", [])
        if isinstance(item, dict) and item.get("audio_path") and item.get("transcript")
    ]
    indexes = build_indexes(samples)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stt_index_path = artifact_dir / "native_stt_bootstrap_index.json"
    tts_index_path = artifact_dir / "native_tts_bootstrap_index.json"
    write_json(stt_index_path, indexes["stt"])
    write_json(tts_index_path, indexes["tts"])
    stt_report = component_report(
        component_id="native_stt_decoder",
        manifest_path=manifest_path,
        index_path=stt_index_path,
        sample_count=len(samples),
        role="speech_to_text",
        bootstrap_ready=bool(samples),
        metrics={
            "memorized_bootstrap_exact_transcripts": len(samples),
            "word_error_rate": None,
            "character_error_rate": None,
        },
        next_action="Train a real native acoustic decoder over licensed shards; this bootstrap index is only a local data-path proof.",
    )
    tts_report = component_report(
        component_id="native_tts_generator",
        manifest_path=manifest_path,
        index_path=tts_index_path,
        sample_count=len(samples),
        role="text_to_speech",
        bootstrap_ready=bool(samples),
        metrics={
            "paired_audio_text_examples": len(samples),
            "mel_or_waveform_reconstruction_loss": None,
            "intelligibility_proxy": None,
        },
        next_action="Train a real native speech generator over approved TTS shards; this bootstrap index only proves artifact plumbing.",
    )
    write_json(ROOT / args.stt_out, stt_report)
    write_json(ROOT / args.tts_out, tts_report)
    print(json.dumps({"stt": stt_report["summary"], "tts": tts_report["summary"]}, indent=2))
    return 0 if samples else 1


def build_indexes(samples: list[dict[str, Any]]) -> dict[str, Any]:
    stt_rows = []
    tts_rows = []
    for sample in samples:
        audio_path = ROOT / str(sample.get("audio_path"))
        digest = sha256_file(audio_path) if audio_path.exists() else ""
        row = {
            "sample_id": sample.get("sample_id"),
            "source_id": sample.get("source_id"),
            "audio_path": sample.get("audio_path"),
            "transcript": sample.get("transcript"),
            "speaker": sample.get("speaker"),
            "bytes": sample.get("bytes"),
            "sha256": digest,
            "license_spdx": sample.get("license_spdx"),
        }
        stt_rows.append(row)
        tts_rows.append(row)
    return {
        "stt": {
            "schema": "project_theseus_native_stt_bootstrap_index_v0",
            "created_utc": now(),
            "rows": stt_rows,
            "external_inference_calls": 0,
        },
        "tts": {
            "schema": "project_theseus_native_tts_bootstrap_index_v0",
            "created_utc": now(),
            "rows": tts_rows,
            "external_inference_calls": 0,
        },
    }


def component_report(
    component_id: str,
    manifest_path: Path,
    index_path: Path,
    sample_count: int,
    role: str,
    bootstrap_ready: bool,
    metrics: dict[str, Any],
    next_action: str,
) -> dict[str, Any]:
    return {
        "policy": "project_theseus_native_voice_component_report_v0",
        "created_utc": now(),
        "component_id": component_id,
        "role": role,
        "manifest": rel(manifest_path),
        "artifact": rel(index_path),
        "summary": {
            "status": "bootstrap_ready" if bootstrap_ready else "waiting_for_samples",
            "bootstrap_ready": bootstrap_ready,
            "native_model_ready": False,
            "sample_count": sample_count,
            "external_inference_calls": 0,
        },
        "metrics": metrics,
        "limits": [
            "This bootstrap is an index/plumbing proof, not a broad STT/TTS model.",
            "It must not be used as evidence of mastered voice capability.",
        ],
        "next_action": next_action,
        "external_inference_calls": 0,
    }


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
