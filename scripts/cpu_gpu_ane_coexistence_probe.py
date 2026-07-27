#!/usr/bin/env python3
"""Measure three-engine CPU, GPU, and ANE process coexistence.

The commands must each emit a final JSON object.  This harness alternates the
standalone order, then launches all three concurrently.  It records both whole
process wall time and worker-reported kernel timing so imports/compilation
cannot masquerade as accelerator overlap.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


POLICY = "project_theseus_cpu_gpu_ane_coexistence_probe_v1"
LABELS = ("cpu", "gpu", "ane")


class CoexistenceFault(ValueError):
    """Raised for malformed command packets or worker evidence."""


def validate_command(command: Any, label: str) -> list[str]:
    if not isinstance(command, list) or not command:
        raise CoexistenceFault(f"{label}_must_be_nonempty_argv_array")
    if any(not isinstance(part, str) or not part for part in command):
        raise CoexistenceFault(f"{label}_contains_invalid_argv")
    return list(command)


def final_json(output: bytes) -> dict[str, Any] | None:
    for raw_line in reversed(output.decode("utf-8", errors="replace").splitlines()):
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def worker_mean_milliseconds(label: str, payload: dict[str, Any]) -> float:
    if label == "cpu":
        value = payload.get("mean_milliseconds")
    else:
        value = (payload.get("runtime") or {}).get("mean_milliseconds")
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise CoexistenceFault(f"{label}_worker_mean_milliseconds_missing")
    return float(value)


def run_command(label: str, command: Sequence[str]) -> dict[str, Any]:
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
    payload = final_json(output)
    parse_error = ""
    internal_mean_milliseconds = 0.0
    if payload is None:
        parse_error = "final_json_missing"
    else:
        try:
            internal_mean_milliseconds = worker_mean_milliseconds(
                label, payload
            )
        except CoexistenceFault as error:
            parse_error = str(error)
    trigger_state = str((payload or {}).get("trigger_state") or "")
    return {
        "seconds": duration,
        "returncode": completed.returncode,
        "output_bytes": len(output),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "output_tail_on_error": (
            output[-1024:].decode("utf-8", errors="replace")
            if completed.returncode != 0 or parse_error
            else ""
        ),
        "worker_policy": str((payload or {}).get("policy") or ""),
        "worker_trigger_state": trigger_state,
        "worker_shape": (payload or {}).get("shape"),
        "worker_internal_mean_milliseconds": internal_mean_milliseconds,
        "worker_parse_error": parse_error,
    }


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def execute(
    commands: dict[str, Sequence[str]],
    *,
    rounds: int,
) -> dict[str, Any]:
    if rounds < 2:
        raise CoexistenceFault("rounds_must_be_at_least_two")
    if set(commands) != set(LABELS):
        raise CoexistenceFault("commands_must_bind_cpu_gpu_and_ane")

    standalone: dict[str, list[dict[str, Any]]] = {
        label: [] for label in LABELS
    }
    concurrent_rounds: list[dict[str, Any]] = []
    for round_index in range(rounds):
        order = LABELS[round_index % len(LABELS) :] + LABELS[
            : round_index % len(LABELS)
        ]
        for label in order:
            standalone[label].append(
                run_command(label, commands[label])
            )

        concurrent_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                label: executor.submit(
                    run_command, label, commands[label]
                )
                for label in LABELS
            }
            results = {
                label: futures[label].result()
                for label in LABELS
            }
        concurrent_rounds.append(
            {
                "round": round_index,
                "wall_seconds": time.perf_counter() - concurrent_started,
                **results,
            }
        )

    process_summaries = {
        label: {
            "standalone_seconds": summary(
                [float(row["seconds"]) for row in standalone[label]]
            ),
            "concurrent_seconds": summary(
                [
                    float(row[label]["seconds"])
                    for row in concurrent_rounds
                ]
            ),
            "standalone_internal_mean_milliseconds": summary(
                [
                    float(row["worker_internal_mean_milliseconds"])
                    for row in standalone[label]
                ]
            ),
            "concurrent_internal_mean_milliseconds": summary(
                [
                    float(
                        row[label]["worker_internal_mean_milliseconds"]
                    )
                    for row in concurrent_rounds
                ]
            ),
        }
        for label in LABELS
    }
    for label in LABELS:
        process_summaries[label]["process_slowdown"] = (
            float(
                process_summaries[label]["concurrent_seconds"]["median"]
            )
            / float(
                process_summaries[label]["standalone_seconds"]["median"]
            )
        )
        process_summaries[label]["kernel_slowdown"] = (
            float(
                process_summaries[label][
                    "concurrent_internal_mean_milliseconds"
                ]["median"]
            )
            / float(
                process_summaries[label][
                    "standalone_internal_mean_milliseconds"
                ]["median"]
            )
        )

    concurrent_wall = summary(
        [float(row["wall_seconds"]) for row in concurrent_rounds]
    )
    standalone_serial_median = sum(
        float(
            process_summaries[label]["standalone_seconds"]["median"]
        )
        for label in LABELS
    )
    all_results = [
        row
        for label in LABELS
        for row in standalone[label]
    ] + [
        row[label]
        for row in concurrent_rounds
        for label in LABELS
    ]
    commands_green = all(
        int(row["returncode"]) == 0
        and not row["worker_parse_error"]
        and (
            row["worker_trigger_state"] == "GREEN"
            or row["worker_trigger_state"].startswith("GREEN_")
        )
        for row in all_results
    )
    overlap_speedup = (
        standalone_serial_median / float(concurrent_wall["median"])
    )
    actual_overlap = (
        float(concurrent_wall["maximum"])
        < sum(
            float(
                process_summaries[label]["standalone_seconds"]["minimum"]
            )
            for label in LABELS
        )
    )
    return {
        "policy": POLICY,
        "trigger_state": (
            "GREEN_THREE_ENGINE_MECHANICAL_OVERLAP"
            if commands_green and actual_overlap
            else "INCONCLUSIVE_OR_RED"
        ),
        "rounds": rounds,
        "workers": process_summaries,
        "concurrent_wall_seconds": concurrent_wall,
        "standalone_serial_median_seconds": standalone_serial_median,
        "overlap_speedup_vs_serial_sum": overlap_speedup,
        "actual_overlap_observed": actual_overlap,
        "all_commands_green": commands_green,
        "raw": {
            "standalone": standalone,
            "concurrent_rounds": concurrent_rounds,
        },
        "claim_scope": (
            "Three-process CPU Accelerate, MLX/Metal, and Core ML/ANE "
            "coexistence mechanics only. Workloads are shape-related but "
            "not one matched Theseus optimizer step; rates cannot be summed "
            "into a training speedup claim."
        ),
        "canonical_checkpoint_mutated": False,
        "public_benchmark_rows_read": 0,
        "external_inference_calls": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet = json.loads(args.commands.read_text(encoding="utf-8"))
    commands = {
        label: validate_command(packet.get(label), label)
        for label in LABELS
    }
    report = execute(commands, rounds=args.rounds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.out)
    print(json.dumps(report, sort_keys=True))
    return (
        0
        if report["trigger_state"]
        == "GREEN_THREE_ENGINE_MECHANICAL_OVERLAP"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
