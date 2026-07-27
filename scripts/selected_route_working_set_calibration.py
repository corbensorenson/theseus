#!/usr/bin/env python3
"""Bind successful exact-route receipts into a memory working-set envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_selected_route_working_set_calibration_v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def calibrate(
    pairs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    if len(pairs) < 2:
        raise ValueError("two successful exact-route receipts are required")
    rows = []
    command_shapes = set()
    for route_path, host_path in pairs:
        route = read_json(route_path)
        wrapper = read_json(host_path)
        host = wrapper.get("host_resource_safety")
        if not isinstance(host, dict):
            host = wrapper
        command = [str(value) for value in host.get("command") or []]
        try:
            steps = command[command.index("--steps") + 1]
            config = command[command.index("--config") + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError("guarded command shape is incomplete") from exc
        command_shapes.add((Path(config).name, steps))
        if (
            wrapper.get("passed") is not True
            or host.get("passed") is not True
            or host.get("fault")
            or route.get("resume_validation") != "GREEN"
            or int(route.get("optimizer_steps") or 0) != 64
            or float(
                host.get("maximum_inferred_unified_memory_mib") or 0.0
            )
            <= 0.0
        ):
            raise ValueError("only successful exact 64-step routes may calibrate")
        rows.append(
            {
                "route": {
                    "path": relative(route_path),
                    "sha256": sha256_file(route_path),
                    "optimizer_steps": int(route["optimizer_steps"]),
                    "resume_validation": route["resume_validation"],
                    "checkpoint_sha256": route.get("checkpoint_sha256"),
                    "optimizer_state_sha256": route.get(
                        "optimizer_state_sha256"
                    ),
                },
                "host": {
                    "path": relative(host_path),
                    "sha256": sha256_file(host_path),
                    "maximum_inferred_unified_memory_mib": float(
                        host["maximum_inferred_unified_memory_mib"]
                    ),
                    "minimum_reclaimable_available_mib": float(
                        host["minimum_reclaimable_available_mib"]
                    ),
                    "maximum_swapout_growth_mib": float(
                        host["maximum_swapout_growth_mib"]
                    ),
                    "raw_receipt": host,
                },
            }
        )
    if command_shapes != {("moecot_language_arm_training.json", "64")}:
        raise ValueError("calibration command shapes do not match")
    return {
        "policy": POLICY,
        "passed": True,
        "route_id": "compiled_fp32_microbatch4_width512_materialized_state",
        "successful_receipt_count": len(rows),
        "maximum_inferred_unified_memory_mib": max(
            row["host"]["maximum_inferred_unified_memory_mib"]
            for row in rows
        ),
        "minimum_reclaimable_available_mib_observed": min(
            row["host"]["minimum_reclaimable_available_mib"]
            for row in rows
        ),
        "selection_rule": (
            "Suppress decline-rate extrapolation only while the current exact "
            "route remains inside this independently completed working-set "
            "envelope and initial reclaimable memory exceeds the envelope. "
            "There is no fixed remaining-memory floor."
        ),
        "receipts": rows,
        "capability_claim": "NONE_RESOURCE_CALIBRATION_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("ROUTE", "HOST"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = calibrate(
        [
            (Path(route).resolve(), Path(host).resolve())
            for route, host in args.pair
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "successful_receipt_count": report[
                    "successful_receipt_count"
                ],
                "maximum_inferred_unified_memory_mib": report[
                    "maximum_inferred_unified_memory_mib"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
