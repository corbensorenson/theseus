#!/usr/bin/env python3
"""Drive canonical neural-seed pretraining through qualified fresh processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import moecot_language_arm_training as training  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/moecot_language_arm_training.json"
DEFAULT_OUT = ROOT / "reports/neural_seed_training_campaign.json"
POLICY = "project_theseus_fresh_process_training_campaign_v1"


def campaign_state(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = training.bind_scale_preregistration(
        training.read_json(config_path)
    )
    plan = training.build_plan(config, config_path=config_path)
    if plan.get("trigger_state") == "RED":
        raise RuntimeError(
            "canonical plan is not GREEN: "
            + ",".join(plan.get("hard_gaps") or [])
        )
    target = plan["targets"][training.SHARED_TRUNK_ID]
    receipt_path = training.resolve(str(target["receipt"]))
    receipt = (
        training.read_json(receipt_path)
        if receipt_path.is_file()
        else {}
    )
    return config, plan, target, receipt


def estimate(
    config: dict[str, Any],
    target: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    policy = dict(
        config["architecture_training_authority"]["fresh_process_segments"]
    )
    qualification_path = training.resolve(
        str(policy["qualification_report"])
    )
    qualification = (
        training.read_json(qualification_path)
        if qualification_path.is_file()
        else {}
    )
    rows = list(qualification.get("fresh_process_segments") or [])
    device_positions = sum(
        int(
            row.get("pretrain_optimizer_positions")
            or row.get("optimizer_positions")
            or 0
        )
        for row in rows[-1:]
    ) - (
        int(
            rows[0].get("pretrain_optimizer_positions")
            or rows[0].get("optimizer_positions")
            or 0
        )
        if len(rows) > 1
        else 0
    )
    device_seconds = sum(
        float(row.get("device_step_seconds_total") or 0.0)
        for row in rows[1:] if len(rows) > 1
    )
    measured_pps = (
        device_positions / device_seconds
        if device_positions > 0 and device_seconds > 0
        else 0.0
    )
    completed = int(receipt.get("pretrain_optimizer_positions") or 0)
    target_positions = int(target["optimizer_target_positions"])
    remaining = max(0, target_positions - completed)
    seconds = remaining / measured_pps if measured_pps > 0 else None
    return {
        "completed_pretrain_optimizer_positions": completed,
        "target_pretrain_optimizer_positions": target_positions,
        "remaining_pretrain_optimizer_positions": remaining,
        "measured_device_positions_per_second": round(measured_pps, 3),
        "estimated_remaining_device_seconds": (
            round(seconds, 3) if seconds is not None else None
        ),
        "estimated_remaining_device_days": (
            round(seconds / 86400.0, 3)
            if seconds is not None
            else None
        ),
        "qualification_report": training.relative(qualification_path),
        "qualification_trigger_state": qualification.get("trigger_state"),
    }


def run_campaign(
    *,
    config_path: Path,
    out: Path,
    max_segments: int,
) -> dict[str, Any]:
    config, plan, target, receipt = campaign_state(config_path)
    policy = dict(
        config["architecture_training_authority"]["fresh_process_segments"]
    )
    steps = int(policy["maximum_optimizer_steps"])
    authority = training.architecture_training_authority(
        config,
        max_steps=steps,
        targets=[training.SHARED_TRUNK_ID],
        phase="pretraining",
        resume=True,
        campaign_segment=True,
    )
    if authority.get("trigger_state") != "GREEN":
        raise RuntimeError(
            "fresh-process segment authority denied: "
            + str(authority.get("reason") or "")
        )
    segment_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    while True:
        _, _, target, before = campaign_state(config_path)
        remaining = max(
            0,
            int(target["optimizer_target_positions"])
            - int(before.get("pretrain_optimizer_positions") or 0),
        )
        if remaining == 0 or (
            max_segments > 0 and len(segment_rows) >= max_segments
        ):
            break
        command = [
            str(
                training.resolve(
                    str(config["host_resource_safety"]["qualified_python"])
                )
            ),
            str(ROOT / "scripts" / "moecot_language_arm_training.py"),
            "--config",
            str(config_path),
            "--out",
            str(out.with_name(out.stem + ".segment-latest.json")),
            "--guarded",
            "--execute",
            "--target",
            training.SHARED_TRUNK_ID,
            "--phase",
            "pretraining",
            "--resume",
            "--max-steps",
            str(steps),
            "--campaign-segment",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        _, _, _, after = campaign_state(config_path)
        before_steps = int(before.get("optimizer_steps") or 0)
        after_steps = int(after.get("optimizer_steps") or 0)
        row = {
            "segment_index": len(segment_rows) + 1,
            "command": command,
            "returncode": completed.returncode,
            "optimizer_steps_before": before_steps,
            "optimizer_steps_after": after_steps,
            "optimizer_step_delta": after_steps - before_steps,
            "pretrain_positions_before": int(
                before.get("pretrain_optimizer_positions") or 0
            ),
            "pretrain_positions_after": int(
                after.get("pretrain_optimizer_positions") or 0
            ),
            "checkpoint_sha256_after": after.get("checkpoint_sha256"),
            "optimizer_state_sha256_after": after.get(
                "optimizer_state_sha256"
            ),
            "mlx_rng_state_sha256_after": after.get(
                "mlx_rng_state_sha256"
            ),
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }
        segment_rows.append(row)
        if (
            completed.returncode != 0
            or row["optimizer_step_delta"] <= 0
            or row["optimizer_step_delta"] > steps
        ):
            break
        interim = {
            "policy": POLICY,
            "created_utc": training.now(),
            "trigger_state": "RUNNING",
            "plan_sha256": plan["plan_sha256"],
            "segment_policy": policy,
            "segments": segment_rows,
            "progress": estimate(config, target, after),
        }
        training.write_json_atomic(out, interim)
    _, final_plan, final_target, final_receipt = campaign_state(config_path)
    failed = bool(
        segment_rows
        and (
            segment_rows[-1]["returncode"] != 0
            or segment_rows[-1]["optimizer_step_delta"] <= 0
            or segment_rows[-1]["optimizer_step_delta"] > steps
        )
    )
    return {
        "policy": POLICY,
        "created_utc": training.now(),
        "trigger_state": "RED" if failed else "GREEN",
        "plan_sha256": final_plan["plan_sha256"],
        "segment_policy": policy,
        "segment_authority": authority,
        "segments_executed": len(segment_rows),
        "segments": segment_rows,
        "progress": estimate(config, final_target, final_receipt),
        "pretraining_complete": (
            int(final_receipt.get("pretrain_optimizer_positions") or 0)
            >= int(final_target["optimizer_target_positions"])
        ),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "capability_claim": "NOT_EVALUATED",
        "hard_gaps": (
            ["fresh_process_segment_failed"] if failed else []
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Zero runs until pretraining completion; positive values bound this invocation.",
    )
    args = parser.parse_args()
    if args.max_segments < 0:
        parser.error("--max-segments cannot be negative")
    config_path = Path(args.config).resolve()
    out = Path(args.out).resolve()
    if args.execute:
        report = run_campaign(
            config_path=config_path,
            out=out,
            max_segments=args.max_segments,
        )
    else:
        config, plan, target, receipt = campaign_state(config_path)
        report = {
            "policy": POLICY,
            "created_utc": training.now(),
            "trigger_state": "STATUS_ONLY",
            "plan_sha256": plan["plan_sha256"],
            "progress": estimate(config, target, receipt),
            "capability_claim": "NOT_EVALUATED",
        }
    training.write_json_atomic(out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report.get("trigger_state") == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
