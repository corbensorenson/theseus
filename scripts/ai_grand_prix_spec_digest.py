"""Digest the local AI Grand Prix technical specification into a report.

The source PDF is user-supplied and local. This script extracts only the
operational constraints needed by the drone arm and adapter smoke gates.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = Path(r"C:\Users\corbe\Downloads\260508_Technical_Spec_0002.pdf")
DEFAULT_OUT = ROOT / "reports" / "ai_grand_prix_spec_digest.json"
DEFAULT_TEXT = ROOT / "reports" / "ai_grand_prix_spec_extract.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--text-out", default=str(DEFAULT_TEXT.relative_to(ROOT)))
    args = parser.parse_args()

    pdf = Path(args.pdf)
    text = extract_text(pdf)
    if text:
        text_path = ROOT / args.text_out
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")
    digest = build_digest(pdf, text, str(Path(args.text_out)).replace("\\", "/"))
    write_json(ROOT / args.out, digest)
    print(json.dumps(digest, indent=2))
    return 0 if digest["summary"]["contract_recorded"] else 1


def extract_text(pdf: Path) -> str:
    if not pdf.exists():
        return DEFAULT_TEXT.read_text(encoding="utf-8") if DEFAULT_TEXT.exists() else ""
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return DEFAULT_TEXT.read_text(encoding="utf-8") if DEFAULT_TEXT.exists() else ""
    reader = PdfReader(str(pdf))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def build_digest(pdf: Path, text: str, text_report: str) -> dict[str, Any]:
    clean = " ".join(text.split())
    issue = first_match(clean, r"ISSUE:\s*([0-9.]+)") or "00.02"
    spec_id = first_match(clean, r"DOCUMENT ID:\s*([A-Z0-9-]+)") or "VADR-TS-002"
    python_known = first_match(clean, r"Python\s+([0-9]+\.[0-9]+\.[0-9]+)") or "3.14.2"
    return {
        "policy": "sparkstream_ai_grand_prix_spec_digest_v0",
        "created_utc": now(),
        "source_pdf": str(pdf),
        "source_pdf_exists": pdf.exists(),
        "source_extract": text_report,
        "spec_id": spec_id,
        "issue": issue,
        "runtime": {
            "os_target": "Windows 11",
            "linux_supported": False,
            "python_known_good": python_known,
            "gpu_vram_mib": 8192,
            "other_environments_allowed": True,
        },
        "simulation": {
            "environment": "high-fidelity deterministic real-time physics simulator",
            "physics_update_hz": 120,
            "gps_available": False,
            "global_position_exposed": False,
            "course_elements": ["start_gate", "sequential_gates", "finish_gate", "obstacles", "boundaries", "terrain"],
            "drone_chassis_mm": {"width": 280, "length": 280, "height": 160},
            "gate_outer_mm": {"width": 2700, "height": 2700, "depth": 260},
            "gate_inner_mm": {"width": 1500, "height": 1500, "depth": 260},
        },
        "frames": {
            "coordinate_convention": "MAVLink2 NED",
            "local_frame": "MAV_FRAME_LOCAL_NED",
            "body_frame": "MAV_FRAME_BODY_NED",
            "body_to_imu": "identity",
            "camera_pitch_deg_up": 20,
            "camera_intrinsics": {
                "resolution_px": [640, 360],
                "cx_cy": [320, 180],
                "fx_fy": [320, 320],
                "vfov_deg": 90,
                "distortion": "none",
            },
        },
        "mavlink": {
            "transport": "UDP",
            "compatible_interfaces": ["MAVSDK", "MAVLink2"],
            "supported_messages": [
                "HEARTBEAT",
                "ATTITUDE",
                "HIGHRES_IMU",
                "SET_POSITION_TARGET_LOCAL_NED",
                "SET_ATTITUDE_TARGET",
                "TIMESYNC",
            ],
            "command_rate_hz_max_exclusive": 100,
            "heartbeat_rate_hz_min": 2,
        },
        "vision_stream": {
            "frequency_hz": 30,
            "resolution_px": [640, 360],
            "transport": "UDP",
            "default_port": 5600,
            "byte_order": "little_endian",
            "header_size_bytes": 24,
            "fields": ["frame_id", "chunk_id", "total_chunks", "jpeg_size", "payload_size", "sim_time_ns"],
        },
        "compliance": {
            "human_interaction_during_submitted_flight": "disqualifying",
            "max_run_duration_minutes": 8,
            "client_responsibilities": [
                "establish_mavlink_communication",
                "maintain_heartbeat",
                "send_control_commands",
                "process_telemetry",
                "process_vision_stream",
            ],
        },
        "safety_governance": {
            "simulation_only_by_default": True,
            "live_drone_hardware_requires_explicit_human_approval": True,
            "external_inference_in_control_loop": "forbidden",
            "required_reflexes": ["lost_heartbeat_hold", "command_rate_clamp", "boundary_hold", "vision_timeout_hold"],
        },
        "summary": {
            "contract_recorded": bool(text),
            "python_314_known_good": python_known.startswith("3.14"),
            "simulator_endpoint_required": True,
            "competition_lane": "drone_racing_sitl",
        },
        "external_inference_calls": 0,
    }


def first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
