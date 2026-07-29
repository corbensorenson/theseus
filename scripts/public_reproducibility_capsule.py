#!/usr/bin/env python3
"""Run the licensed, public-safe tiny Theseus evidence-protocol capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "examples" / "public_repro_capsule" / "config.json"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def initial_state() -> dict[str, Any]:
    return {
        "step": 0,
        "weight": Decimal("0"),
        "bias": Decimal("0"),
        "weight_momentum": Decimal("0"),
        "bias_momentum": Decimal("0"),
        "cursor": 0,
    }


def train(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    updates: int,
    learning_rate: Decimal,
    momentum: Decimal,
) -> dict[str, Any]:
    current = dict(state)
    while int(current["step"]) < updates:
        row = rows[int(current["cursor"]) % len(rows)]
        x = Decimal(str(row["x"]))
        y = Decimal(str(row["y"]))
        prediction = current["weight"] * x + current["bias"]
        error = prediction - y
        weight_gradient = Decimal("2") * error * x
        bias_gradient = Decimal("2") * error
        current["weight_momentum"] = (
            momentum * current["weight_momentum"]
            + (Decimal("1") - momentum) * weight_gradient
        )
        current["bias_momentum"] = (
            momentum * current["bias_momentum"]
            + (Decimal("1") - momentum) * bias_gradient
        )
        current["weight"] -= learning_rate * current["weight_momentum"]
        current["bias"] -= learning_rate * current["bias_momentum"]
        current["step"] = int(current["step"]) + 1
        current["cursor"] = (int(current["cursor"]) + 1) % len(rows)
    return current


def serialized_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": "project_theseus_public_tiny_checkpoint_v1",
        "step": int(state["step"]),
        "model": {
            "weight": str(state["weight"]),
            "bias": str(state["bias"]),
        },
        "optimizer": {
            "kind": "momentum_sgd",
            "weight_momentum": str(state["weight_momentum"]),
            "bias_momentum": str(state["bias_momentum"]),
        },
        "sampler": {
            "cursor": int(state["cursor"]),
            "order": "cyclic_source_order",
        },
    }


def restored_state(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(value["step"]),
        "weight": Decimal(value["model"]["weight"]),
        "bias": Decimal(value["model"]["bias"]),
        "weight_momentum": Decimal(value["optimizer"]["weight_momentum"]),
        "bias_momentum": Decimal(value["optimizer"]["bias_momentum"]),
        "cursor": int(value["sampler"]["cursor"]),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def execute(config_path: Path, out_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    getcontext().prec = int(config["decimal_precision"])
    data_path = ROOT / config["dataset"]
    rows = read_rows(data_path)
    learning_rate = Decimal(config["learning_rate"])
    momentum = Decimal(config["momentum"])
    midpoint = int(config["checkpoint_update"])
    updates = int(config["updates"])

    midpoint_state = train(
        initial_state(),
        rows,
        updates=midpoint,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    midpoint_path = out_dir / "checkpoint-midpoint.json"
    write_json(midpoint_path, serialized_state(midpoint_state))
    resumed = train(
        restored_state(json.loads(midpoint_path.read_text())),
        rows,
        updates=updates,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    uninterrupted = train(
        initial_state(),
        rows,
        updates=updates,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    resumed_payload = serialized_state(resumed)
    uninterrupted_payload = serialized_state(uninterrupted)
    final_path = out_dir / "checkpoint-final.json"
    write_json(final_path, resumed_payload)

    candidates = []
    tolerance = Decimal(config["absolute_tolerance"])
    for index, (raw_x, raw_expected) in enumerate(
        zip(config["candidate_inputs"], config["expected_outputs"])
    ):
        x = Decimal(raw_x)
        expected = Decimal(raw_expected)
        observed = resumed["weight"] * x + resumed["bias"]
        candidates.append(
            {
                "candidate_id": f"tiny-linear-candidate-{index}",
                "input": str(x),
                "output": str(observed),
                "output_sha256": digest_bytes(str(observed).encode()),
                "expected": str(expected),
                "absolute_error": str(abs(observed - expected)),
                "passed": abs(observed - expected) <= tolerance,
            }
        )
    candidate_packet = {
        "policy": "project_theseus_public_tiny_candidate_packet_v1",
        "rows": candidates,
        "training_eligible": False,
        "capability_claim": "NONE_EVIDENCE_PROTOCOL_ONLY",
    }
    candidate_path = out_dir / "candidate-packet.json"
    write_json(candidate_path, candidate_packet)
    verifier = {
        "policy": "project_theseus_public_tiny_verifier_v1",
        "tolerance": str(tolerance),
        "passed": all(row["passed"] for row in candidates),
        "candidate_packet_sha256": digest_file(candidate_path),
        "case_count": len(candidates),
    }
    verifier_path = out_dir / "verifier-output.json"
    write_json(verifier_path, verifier)

    replay_equal = resumed_payload == uninterrupted_payload
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest_file(path),
        }
        for path in (
            midpoint_path,
            final_path,
            candidate_path,
            verifier_path,
        )
    ]
    ready = replay_equal and verifier["passed"]
    report = {
        "policy": config["policy"],
        "trigger_state": "GREEN" if ready else "RED",
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "decimal_precision": getcontext().prec,
        },
        "inputs": {
            "config": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": digest_file(config_path),
            "dataset": data_path.relative_to(ROOT).as_posix(),
            "dataset_sha256": digest_file(data_path),
            "row_count": len(rows),
            "license": config["license"],
        },
        "training": {
            "updates": updates,
            "checkpoint_update": midpoint,
            "resume_exact_match": replay_equal,
            "final_state_sha256": digest_bytes(canonical_bytes(resumed_payload)),
        },
        "verification": verifier,
        "artifacts": files,
        "runtime": {
            "wall_seconds": round(time.perf_counter() - started, 6),
            "measurement": "local_monotonic_observation_not_performance_claim",
        },
        "boundaries": config["boundaries"],
        "non_claims": [
            "This capsule verifies an evidence protocol, not useful learned capability.",
            "Its runtime does not predict MLX, Metal, CUDA, or production training speed.",
            "Its synthetic rows are not a public benchmark or private Theseus curriculum.",
        ],
    }
    write_json(out_dir / "capsule-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    report = execute(config_path, Path(args.out_dir).resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else (2 if args.gate else 0)


if __name__ == "__main__":
    sys.exit(main())
