#!/usr/bin/env python3
"""Offline load/generation preflight for a pinned local Worker v2 model."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_evidence_worker_v2 import LocalMlxModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run(Path(args.config))
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "repo_id": report["model_identity"]["repo_id"],
        "load_wall_ms": report["runtime"]["load_wall_ms"],
        "generation_wall_ms": report["runtime"]["generation_wall_ms"],
        "peak_rss_mib": report["runtime"]["peak_rss_mib"],
    }, indent=2, sort_keys=True))
    return 0


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    card = dict(config["model"])
    card["maximum_action_tokens"] = min(
        96, int(card["maximum_action_tokens"])
    )
    swap_before = swap_used_mib()
    load_started = time.perf_counter()
    model = LocalMlxModel(card)
    load_wall_ms = (time.perf_counter() - load_started) * 1000.0
    generation_started = time.perf_counter()
    output = model.generate([
        {
            "role": "system",
            "content": (
                "Return exactly one compact JSON object and no prose. "
                "Use this schema: {\"action\":\"list\",\"prefix\":\"scripts\"}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Inspect the scripts directory. Return the requested JSON "
                "list action."
            ),
        },
    ])
    generation_wall_ms = (time.perf_counter() - generation_started) * 1000.0
    parsed = None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        pass
    valid_action = parsed == {"action": "list", "prefix": "scripts"}
    swap_after = swap_used_mib()
    return {
        "policy": "project_theseus_local_model_runtime_preflight_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if valid_action else "RED",
        "model_identity": {
            "repo_id": card["repo_id"],
            "revision": card["revision"],
            "snapshot_manifest_sha256": model.snapshot_manifest_sha256,
        },
        "runtime": {
            "load_wall_ms": round(load_wall_ms, 3),
            "generation_wall_ms": round(generation_wall_ms, 3),
            "peak_rss_mib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 * 1024),
                3,
            ),
            "swap_before_mib": swap_before,
            "swap_after_mib": swap_after,
            "swap_growth_mib": (
                None if swap_before is None or swap_after is None
                else round(swap_after - swap_before, 3)
            ),
            **model.last_generation_metrics,
        },
        "output": {
            "raw_sha256": __import__("hashlib").sha256(
                output.encode()
            ).hexdigest(),
            "characters": len(output),
            "exact_action_valid": valid_action,
        },
        "counters": {
            "local_model_inference_calls": 1,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "user_facing_effects": 0,
        },
        "maximum_inference": (
            "This proves only offline runtime compatibility and exact action "
            "format on one trivial prompt, not repository competence."
        ),
    }


def swap_used_mib() -> float | None:
    result = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return None
    import re
    match = re.search(r"used = ([0-9.]+)M", result.stdout)
    return float(match.group(1)) if match else None


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
