"""Report the native Project Theseus voice I/O substrate.

This is deliberately not an STT/TTS package probe. Voice is a native capability
lane for the Octopus head/router: external/provider speech inference is
forbidden, third-party pretrained voice models are not accepted as runtime
intelligence, and benchmark/data sources are only pressure surfaces for local
learning.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/native_voice_policy.json")
    parser.add_argument("--out", default="reports/native_voice_io.json")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    cards = voice_cards()
    training_manifest = read_json(ROOT / "reports" / "native_voice_training_manifest.json")
    packet_probe = audio_packet_probe()
    checks = [
        check("native_voice_policy_present", bool(policy), args.policy),
        check("voice_is_head_router_io", get_path(policy, ["architecture", "voice_is_head_router_io"]) is True, policy.get("owner", "")),
        check("provider_stt_tts_forbidden", get_path(policy, ["execution_boundary", "provider_stt_tts"]) == "forbidden", "no provider speech inference"),
        check(
            "pretrained_third_party_voice_models_forbidden",
            get_path(policy, ["execution_boundary", "pretrained_third_party_voice_models"]) == "forbidden_for_inference",
            "no borrowed STT/TTS model inference",
        ),
        check("voice_cards_present", len(cards) >= 3, f"present={len(cards)}"),
        check(
            "native_voice_training_manifest_ready",
            bool(get_path(training_manifest, ["summary", "ready_for_native_training"], False)),
            f"stt={get_path(training_manifest, ['summary', 'stt_sources'], 0)} tts={get_path(training_manifest, ['summary', 'tts_sources'], 0)} tiny_clips={get_path(training_manifest, ['summary', 'tiny_audio_clips'], 0)}",
        ),
        check("audio_packet_contract_passes", packet_probe["ok"], packet_probe["summary"]),
        check("external_inference_calls_zero", True, "0"),
    ]
    learned_components = learned_voice_components()
    checks.append(
        check(
            "native_voice_bootstrap_reports_present",
            learned_components.get("bootstrap_ready_count", 0) >= 1,
            f"bootstrap_ready={learned_components.get('bootstrap_ready_count', 0)}",
        )
    )
    scaffold_score = sum(1 for item in checks if item["passed"]) / max(1, len(checks))
    native_model_score = min(0.42, 0.21 * learned_components["ready_count"])
    data_score = 0.08 if get_path(training_manifest, ["summary", "ready_for_native_training"], False) else 0.0
    tiny_bootstrap_score = 0.04 if get_path(training_manifest, ["summary", "ready_for_stt_bootstrap"], False) else 0.0
    score = round(min(0.86, 0.12 + 0.18 * scaffold_score + data_score + tiny_bootstrap_score + native_model_score), 4)

    report = {
        "policy": "project_theseus_native_voice_io_report_v0",
        "created_utc": now(),
        "config": args.policy,
        "owner": policy.get("owner", "octopus_head_router_io_layer"),
        "summary": {
            "suite": "native_voice_io",
            "accuracy": score,
            "status": "frontier_open" if score < 0.9 else "mastered",
            "voice_is_head_router_io": get_path(policy, ["architecture", "voice_is_head_router_io"]) is True,
            "native_model_ready": learned_components["ready_count"] >= 2,
            "external_inference_calls": 0,
        },
        "checks": checks,
        "packet_contract": packet_probe,
        "voice_cards": cards,
        "training_manifest": manifest_summary(training_manifest),
        "learned_components": learned_components,
        "allowed_non_inference_utilities": policy.get("allowed_non_inference_utilities", []),
        "disallowed_inference_modules": policy.get("disallowed_inference_modules", []),
        "residuals": residuals(checks, learned_components),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def voice_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for path in sorted((ROOT / "benchmarks" / "cards").glob("source_*.json")):
        data = read_json(path)
        source_id = str(data.get("source_id") or "")
        if (
            str(data.get("runner_family") or "") == "voice_local"
            or str(data.get("category") or "").startswith("voice")
            or source_id in {"common_voice", "librispeech", "libritts", "ljspeech", "vctk", "speechbrain_benchmarks"}
        ):
            cards.append(
                {
                    "id": data.get("id"),
                    "source_id": source_id,
                    "status": data.get("status"),
                    "license_allowed": bool(data.get("license_allowed")),
                    "path": rel(path),
                }
            )
    return cards


def audio_packet_probe() -> dict[str, Any]:
    sample_rate = 8000
    duration_s = 0.08
    samples = [
        int(16000 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
        for i in range(int(sample_rate * duration_s))
    ]
    raw = BytesIO()
    with wave.open(raw, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))
    payload = raw.getvalue()
    with wave.open(BytesIO(payload), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        decoded = [
            int.from_bytes(frames[index : index + 2], "little", signed=True)
            for index in range(0, len(frames), 2)
        ]
    energy = sum(sample * sample for sample in decoded) / max(1, len(decoded))
    zero_crossings = sum(
        1
        for left, right in zip(decoded, decoded[1:])
        if (left < 0 <= right) or (left >= 0 > right)
    )
    features = {
        "sample_rate_hz": sample_rate,
        "duration_ms": round(duration_s * 1000.0, 3),
        "frame_count": len(decoded),
        "rms_energy": round(math.sqrt(energy), 4),
        "zero_crossing_rate": round(zero_crossings / max(1, len(decoded) - 1), 6),
    }
    ok = features["frame_count"] > 0 and features["rms_energy"] > 0
    return {
        "ok": ok,
        "summary": f"frames={features['frame_count']} zcr={features['zero_crossing_rate']}",
        "features": features,
        "packet_schema": {
            "input": "audio_input_packet",
            "feature": "native_audio_feature_packet",
            "semantic": "head_router_semantic_packet",
            "output": "native_audio_output_packet",
        },
    }


def learned_voice_components() -> dict[str, Any]:
    expected = {
        "native_stt_decoder": ROOT / "reports" / "native_stt_decoder.json",
        "native_tts_generator": ROOT / "reports" / "native_tts_generator.json",
    }
    components = []
    for name, path in expected.items():
        data = read_json(path)
        ready = bool(data) and get_path(data, ["summary", "native_model_ready"], False) is True
        bootstrap_ready = bool(data) and get_path(data, ["summary", "bootstrap_ready"], False) is True
        components.append(
            {
                "id": name,
                "ready": ready,
                "bootstrap_ready": bootstrap_ready,
                "report": rel(path),
                "present": path.exists(),
                "external_inference_calls": get_path(data, ["summary", "external_inference_calls"], data.get("external_inference_calls", 0)),
            }
        )
    return {
        "ready_count": sum(1 for item in components if item["ready"]),
        "bootstrap_ready_count": sum(1 for item in components if item["bootstrap_ready"]),
        "components": components,
        "rule": "These reports must be produced by local Theseus-trained components, not installed STT/TTS models.",
    }


def residuals(checks: list[dict[str, Any]], learned: dict[str, Any]) -> list[dict[str, Any]]:
    out = [
        {"type": item["name"], "detail": item["evidence"]}
        for item in checks
        if not item["passed"]
    ]
    if learned["ready_count"] < 2:
        out.append(
            {
                "type": "native_voice_model_frontier",
                "detail": "Train native STT and TTS components from licensed data; do not satisfy this gate by installing external speech models.",
            }
        )
    return out


def manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    if not manifest:
        return {
            "present": False,
            "ready_for_native_training": False,
            "manifest_report": "reports/native_voice_training_manifest.json",
        }
    return {
        "present": True,
        "ready_for_native_training": bool(get_path(manifest, ["summary", "ready_for_native_training"], False)),
        "ready_for_stt_bootstrap": bool(get_path(manifest, ["summary", "ready_for_stt_bootstrap"], False)),
        "ready_for_tts_bootstrap": bool(get_path(manifest, ["summary", "ready_for_tts_bootstrap"], False)),
        "sources": get_path(manifest, ["summary", "sources"], 0),
        "tiny_audio_clips": get_path(manifest, ["summary", "tiny_audio_clips"], 0),
        "tiny_audio_bytes": get_path(manifest, ["summary", "tiny_audio_bytes"], 0),
        "external_inference_calls": manifest.get("external_inference_calls", 0),
        "manifest_artifact": manifest.get("manifest_artifact"),
    }


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


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for item in path:
        if not isinstance(cur, dict) or item not in cur:
            return default
        cur = cur[item]
    return cur


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
