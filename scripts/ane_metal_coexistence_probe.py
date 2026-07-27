#!/usr/bin/env python3
"""Measure whether independent ANE and Metal commands actually overlap.

Commands are argv arrays, never shell strings.  The report records alternating
standalone latency and concurrently observed latency so a faster combined wall
time cannot conceal damage to the canonical Metal workload.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


POLICY = "project_theseus_ane_metal_coexistence_probe_v1"


class CoexistenceFault(ValueError):
    """Raised for unsafe or malformed command packets."""


def validate_command(command: Any, label: str) -> list[str]:
    if not isinstance(command, list) or not command:
        raise CoexistenceFault(f"{label}_must_be_nonempty_argv_array")
    if any(not isinstance(part, str) or not part for part in command):
        raise CoexistenceFault(f"{label}_contains_invalid_argv")
    return list(command)


def run_command(command: Sequence[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = time.perf_counter() - started
    output = completed.stdout or b""
    return {
        "seconds": duration,
        "returncode": completed.returncode,
        "output_bytes": len(output),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "output_tail": output[-2048:].decode("utf-8", errors="replace"),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
        "mean_seconds": statistics.fmean(values),
    }


def execute(
    gpu_command: Sequence[str],
    ane_command: Sequence[str],
    *,
    rounds: int,
) -> dict[str, Any]:
    if rounds < 2:
        raise CoexistenceFault("rounds_must_be_at_least_two")
    standalone: dict[str, list[dict[str, Any]]] = {"gpu": [], "ane": []}
    concurrent_pairs: list[dict[str, Any]] = []

    for round_index in range(rounds):
        serial_order = (
            (("gpu", gpu_command), ("ane", ane_command))
            if round_index % 2 == 0
            else (("ane", ane_command), ("gpu", gpu_command))
        )
        for label, command in serial_order:
            standalone[label].append(run_command(command))

        pair_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            gpu_future = executor.submit(run_command, gpu_command)
            ane_future = executor.submit(run_command, ane_command)
            gpu_result = gpu_future.result()
            ane_result = ane_future.result()
        concurrent_pairs.append(
            {
                "round": round_index,
                "wall_seconds": time.perf_counter() - pair_started,
                "gpu": gpu_result,
                "ane": ane_result,
            }
        )

    gpu_standalone = _summary(
        [float(result["seconds"]) for result in standalone["gpu"]]
    )
    ane_standalone = _summary(
        [float(result["seconds"]) for result in standalone["ane"]]
    )
    concurrent_wall = _summary(
        [float(pair["wall_seconds"]) for pair in concurrent_pairs]
    )
    concurrent_gpu = _summary(
        [float(pair["gpu"]["seconds"]) for pair in concurrent_pairs]
    )
    concurrent_ane = _summary(
        [float(pair["ane"]["seconds"]) for pair in concurrent_pairs]
    )
    all_results = (
        standalone["gpu"]
        + standalone["ane"]
        + [pair["gpu"] for pair in concurrent_pairs]
        + [pair["ane"] for pair in concurrent_pairs]
    )
    commands_green = all(result["returncode"] == 0 for result in all_results)
    serial_sum_median = float(gpu_standalone["median_seconds"]) + float(
        ane_standalone["median_seconds"]
    )
    overlap_speedup = serial_sum_median / float(
        concurrent_wall["median_seconds"]
    )
    gpu_slowdown = float(concurrent_gpu["median_seconds"]) / float(
        gpu_standalone["median_seconds"]
    )
    ane_slowdown = float(concurrent_ane["median_seconds"]) / float(
        ane_standalone["median_seconds"]
    )
    actual_overlap_observed = (
        float(concurrent_wall["maximum_seconds"])
        < float(gpu_standalone["minimum_seconds"])
        + float(ane_standalone["minimum_seconds"])
    )
    return {
        "policy": POLICY,
        "trigger_state": (
            "GREEN_MECHANICAL_OVERLAP"
            if commands_green and actual_overlap_observed
            else "INCONCLUSIVE_OR_RED"
        ),
        "rounds": rounds,
        "standalone": {
            "gpu": gpu_standalone,
            "ane": ane_standalone,
        },
        "concurrent": {
            "wall": concurrent_wall,
            "gpu": concurrent_gpu,
            "ane": concurrent_ane,
        },
        "overlap_speedup_vs_serial_sum": overlap_speedup,
        "gpu_latency_slowdown": gpu_slowdown,
        "ane_latency_slowdown": ane_slowdown,
        "actual_overlap_observed": actual_overlap_observed,
        "all_commands_green": commands_green,
        "raw": {
            "standalone": standalone,
            "concurrent_pairs": concurrent_pairs,
        },
        "claim_scope": (
            "Process coexistence only. This does not prove zero-copy tensor "
            "interop, numerical parity, training replay, or end-to-end gain."
        ),
        "canonical_backend_changed": False,
        "checkpoint_mutated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet = json.loads(args.commands.read_text(encoding="utf-8"))
    gpu_command = validate_command(packet.get("gpu"), "gpu")
    ane_command = validate_command(packet.get("ane"), "ane")
    report = execute(gpu_command, ane_command, rounds=args.rounds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_commands_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
