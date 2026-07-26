#!/usr/bin/env python3
"""Build one content-bound MoECOT auxiliary memmap cache without Metal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import moecot_language_arm_training as training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--artifact-field",
        choices=("source_conditioned_artifacts", "supervision_artifacts"),
        required=True,
    )
    args = parser.parse_args()

    config_path = training.resolve(args.config)
    config = training.bind_scale_preregistration(
        training.read_json(config_path)
    )
    plan = training.build_plan(config, config_path=config_path)
    target = (plan.get("targets") or {}).get(args.target)
    if not isinstance(target, dict):
        raise ValueError(f"unknown training target: {args.target}")
    stage_dir = training.resolve(str(config["stage_dir"]))
    metadata = training.read_json(stage_dir / "stage_metadata_v1.json")
    base = training.read_json(training.resolve(str(config["base_config"])))
    receipt_policy = (
        "project_theseus_moecot_source_conditioned_arrays_v1"
        if args.artifact_field == "source_conditioned_artifacts"
        else "project_theseus_moecot_exact_supervision_arrays_v1"
    )
    cache_path = training.write_auxiliary_stage_cache(
        config,
        base,
        target,
        metadata=metadata,
        artifact_field=args.artifact_field,
        receipt_policy=receipt_policy,
    )
    print(
        json.dumps(
            {
                "policy": "project_theseus_auxiliary_stage_memmap_cache_v1",
                "target_id": args.target,
                "artifact_field": args.artifact_field,
                "cache_path": training.relative(cache_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
