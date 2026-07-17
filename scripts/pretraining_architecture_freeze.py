#!/usr/bin/env python3
"""Build and replay the content-addressed pretraining architecture freeze package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "pretraining_architecture_freeze.json"
DEFAULT_REPORT = ROOT / "reports" / "pretraining_architecture_freeze_package.json"


class ArchitectureFreezeFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def negative_disposition_ready(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "")
    if status not in {"falsified_pretraining", "retired_by_pretraining_verdict"}:
        return True
    contract = row.get("negative_disposition_contract") or {}
    kind = str(contract.get("kind") or "")
    if status == "retired_by_pretraining_verdict" and kind == "campaign_scope_only":
        return (
            contract.get("scientific_falsification_claimed") is False
            and bool(str(contract.get("exact_scope") or ""))
            and bool(str(contract.get("reentry_condition") or ""))
        )
    required = (
        "mechanism_fidelity_audited",
        "learnability_sanity_passed",
        "matched_opportunity_audited",
        "independent_construct_valid_evaluation",
        "multi_seed_uncertainty_and_power_reported",
        "replicated",
    )
    return (
        kind == "decision_grade_negative"
        and all(contract.get(key) is True for key in required)
        and bool(str(contract.get("exact_claim_scope") or ""))
    )


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy") != "project_theseus_pretraining_architecture_freeze_v1":
        raise ArchitectureFreezeFault("policy_invalid")
    if len(config.get("required_artifacts") or []) < 40:
        raise ArchitectureFreezeFault("artifact_closure_too_small")
    if len(config.get("replay_commands") or []) < 6:
        raise ArchitectureFreezeFault("replay_closure_too_small")
    boundaries = config.get("boundaries") or {}
    if boundaries.get("long_optimizer_run_allowed_during_freeze") is not False or boundaries.get("public_calibration_allowed_during_freeze") is not False:
        raise ArchitectureFreezeFault("freeze_boundary_invalid")
    if any(int(boundaries.get(key, -1)) != 0 for key in ("public_training_rows", "external_inference_calls", "fallback_or_template_credit")):
        raise ArchitectureFreezeFault("no_cheat_boundary_nonzero")
    return config


def architecture_dispositions(config: dict[str, Any]) -> dict[str, Any]:
    matrix_path = resolve(config["matrix"])
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    architecture = matrix.get("pre_training_architecture_contract") or {}
    required = set(architecture.get("required_backlog_ids") or [])
    ready_statuses = set(config["required_ready_backlog_statuses"])
    backlog = {
        row.get("backlog_id"): row
        for row in matrix.get("planned_codex_test_backlog") or []
        if isinstance(row, dict) and row.get("backlog_id") in required
    }
    missing = sorted(required - set(backlog))
    unready = sorted(
        backlog_id for backlog_id, row in backlog.items()
        if row.get("status") not in ready_statuses
        or not row.get("pre_training_acceptance_boundary")
        or not negative_disposition_ready(row)
    )
    if missing or unready:
        raise ArchitectureFreezeFault(
            "architecture_disposition_incomplete:missing=" + ",".join(missing)
            + ";unready=" + ",".join(unready)
        )
    rows = {
        backlog_id: {
            "status": backlog[backlog_id]["status"],
            "acceptance_boundary_digest": digest(backlog[backlog_id]["pre_training_acceptance_boundary"]),
            "evidence": backlog[backlog_id].get("pre_training_evidence"),
            "negative_disposition": backlog[backlog_id].get(
                "negative_disposition_contract"
            ),
        }
        for backlog_id in sorted(required)
    }
    return {
        "required_count": len(required),
        "ready_count": len(rows),
        "rows": rows,
        "matrix_sha256": sha256(matrix_path),
    }


def artifact_manifest(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = {}
    for value in config["required_artifacts"]:
        path = resolve(value)
        if not path.is_file():
            raise ArchitectureFreezeFault(f"required_artifact_missing:{value}")
        manifest[value] = {"path": value, "sha256": sha256(path), "bytes": path.stat().st_size}
    return manifest


def run_replays(config: dict[str, Any]) -> list[dict[str, Any]]:
    receipts = []
    for index, command in enumerate(config["replay_commands"]):
        started = time.perf_counter()
        process = subprocess.run(
            [str(value) for value in command],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=180,
        )
        receipt = {
            "index": index,
            "command": command,
            "returncode": process.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_sha256": hashlib.sha256(process.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr.encode("utf-8")).hexdigest(),
            "stdout_tail": process.stdout[-1200:],
            "stderr_tail": process.stderr[-1200:],
        }
        receipts.append(receipt)
        if process.returncode != 0:
            raise ArchitectureFreezeFault(f"replay_failed:{index}:{' '.join(command)}")
    return receipts


def build_report(config: dict[str, Any], *, execute_replays: bool) -> dict[str, Any]:
    dispositions = architecture_dispositions(config)
    replays = run_replays(config) if execute_replays else []
    if not execute_replays:
        raise ArchitectureFreezeFault("independent_replay_required")
    manifest = artifact_manifest(config)
    package_identity = digest({
        "campaign_id": config["campaign_id"],
        "artifacts": manifest,
        "dispositions": dispositions,
        "commands": config["replay_commands"],
        "boundaries": config["boundaries"],
    })
    return {
        "policy": config["policy"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN",
        "support_state": "replayable-reference-backed",
        "disposition": "architecture_frozen_training_not_started",
        "campaign_id": config["campaign_id"],
        "package_identity": package_identity,
        "source_artifacts": manifest,
        "architecture_dispositions": dispositions,
        "replay_receipts": replays,
        "summary": {
            "artifact_count": len(manifest),
            "architecture_contract_count": dispositions["required_count"],
            "ready_architecture_contract_count": dispositions["ready_count"],
            "replay_count": len(replays),
            "replay_pass_count": sum(row["returncode"] == 0 for row in replays),
            "long_optimizer_steps": 0,
            "public_calibrations": 0,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_or_template_credit": 0,
        },
        "boundaries": config["boundaries"],
        "non_claims": [
            "Architecture freeze and bounded mechanics replay are not model training or capability evidence.",
            "The package authorizes only the exact content-addressed campaign after the external readiness gate revalidates this report.",
            "Any listed artifact change invalidates this package and requires a new replay before long training.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_REPORT))
    parser.add_argument("--execute-replays", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    report = build_report(load_config(config_path), execute_replays=args.execute_replays)
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"trigger_state": report["trigger_state"], "package_identity": report["package_identity"], "summary": report["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
