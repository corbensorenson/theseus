#!/usr/bin/env python3
"""Execute the sealed 57M D2 comparison once under machine-only authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import moecot_language_arm_training as training  # noqa: E402
import neural_seed_functional_consumption as consumption  # noqa: E402
import neural_seed_functional_utility as utility  # noqa: E402
import neural_seed_local_english_raters as local_raters  # noqa: E402
import neural_seed_training_campaign as training_campaign  # noqa: E402


POLICY = "project_theseus_neural_seed_d2_autonomous_one_shot_evaluation_v1"
DEFAULT_CONFIG = ROOT / "configs/neural_seed_d2_autonomous_evaluation_controller.json"
TARGETS = ("moecot_system", "dense_active_parameter", "dense_total_parameter")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report_path = resolve(args.out or str(config["report"]))
    if args.execute:
        recover_stale_lease(config)
        report = execute_once(config, config_path=config_path)
    else:
        report = preflight(config, config_path=config_path)
    write_json_atomic(report_path, report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def preflight(
    config: dict[str, Any],
    *,
    config_path: Path,
    source_override: dict[str, Any] | None = None,
    process_override: list[dict[str, Any]] | None = None,
    freeze_override: dict[str, Any] | None = None,
    manifest_override: dict[str, Any] | None = None,
    checkpoint_override: dict[str, Any] | None = None,
    registry_rows_override: list[dict[str, Any]] | None = None,
    local_models_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_config(config)
    freeze_path = resolve(str(config["freeze"]))
    functional_config_path = resolve(str(config["functional_config"]))
    freeze = freeze_override or read_json(freeze_path)
    functional_config = read_json(functional_config_path)
    manifest = manifest_override or utility.build_manifest(
        functional_config,
        functional_config_path,
    )
    freeze_gaps = utility.validate_freeze(manifest, freeze)
    source = source_override or source_state()
    processes = process_override
    if processes is None:
        processes = training_campaign.active_accelerator_jobs(
            list(config["exclusive_accelerator_process_patterns"])
        )
    checkpoints = checkpoint_override or checkpoint_audit(config, freeze)
    registry_path = resolve(str(config["consumption_registry"]))
    registry_rows = (
        registry_rows_override
        if registry_rows_override is not None
        else consumption.read_registry(registry_path)
    )
    freeze_sha = stable_hash(freeze)
    matching_registry_rows = [
        row
        for row in registry_rows
        if (row.get("identity") or {}).get("freeze_sha256") == freeze_sha
    ]
    local_models = local_models_override or local_model_audit()
    outputs = protected_outputs(config)
    existing_outputs = [relative(path) for path in outputs if path.exists()]
    gates = {
        "source_clean": source.get("clean_at_generation") is True,
        "freeze_not_evaluated": (
            freeze.get("evaluation_state") == "NOT_EVALUATED"
            and int(freeze.get("consumed_case_count") or 0) == 0
        ),
        "freeze_matches_exact_sources": (
            manifest.get("trigger_state") == "GREEN" and not freeze_gaps
        ),
        "all_checkpoints_complete_and_bound": checkpoints.get("trigger_state")
        == "GREEN",
        "surface_identity_unconsumed": not matching_registry_rows,
        "no_result_artifact_collision": not existing_outputs,
        "all_local_rater_snapshots_available": local_models.get("trigger_state")
        == "GREEN",
        "no_competing_accelerator_job": not processes,
        "no_active_lease": not resolve(str(config["active_lease"])).exists(),
        "qualified_python_available": resolve(
            str(config["qualified_python"])
        ).is_file(),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "policy": POLICY,
        "created_utc": training.now(),
        "trigger_state": "GREEN" if not failed else "PAUSED",
        "execution_authorized": not failed,
        "config": artifact(config_path),
        "source_binding": source,
        "freeze": artifact(freeze_path),
        "freeze_sha256": freeze_sha,
        "freeze_gaps": freeze_gaps,
        "manifest_state": manifest.get("trigger_state"),
        "checkpoint_audit": checkpoints,
        "local_model_audit": local_models,
        "matching_registry_row_count": len(matching_registry_rows),
        "existing_protected_outputs": existing_outputs,
        "competing_accelerator_jobs": processes,
        "gates": gates,
        "failed_gates": failed,
        "authority": config["authority"],
        "maximum_inference": (
            "GREEN authorizes exactly one sealed D2 transaction. It does not "
            "authorize a rerun, serving, training-row admission, external "
            "inference, public calibration, architecture promotion, or book "
            "support promotion. A context or host boundary invalidates the "
            "observation and the consumed transaction remains terminal."
        ),
    }


def execute_once(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    before = preflight(config, config_path=config_path)
    acquisition: dict[str, Any] = {}
    if before.get("failed_gates") == ["all_local_rater_snapshots_available"]:
        acquisition = acquire_local_rater_models(config)
        before = preflight(config, config_path=config_path)
    if before["trigger_state"] != "GREEN":
        return {**before, "local_rater_model_acquisition": acquisition}
    lease_id = uuid.uuid4().hex
    active_lease = resolve(str(config["active_lease"]))
    archive_dir = resolve(str(config["lease_archive_directory"]))
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{lease_id}.json"
    lease = {
        "policy": POLICY,
        "lease_id": lease_id,
        "pid": os.getpid(),
        "created_utc": training.now(),
        "state": "RUNNING",
        "source_binding": before["source_binding"],
        "freeze_sha256": before["freeze_sha256"],
        "steps": [],
    }
    try:
        write_json_exclusive(active_lease, lease)
    except FileExistsError:
        raced = dict(before)
        raced["trigger_state"] = "PAUSED"
        raced["execution_authorized"] = False
        raced["failed_gates"] = sorted(set([*raced["failed_gates"], "no_active_lease"]))
        raced["lease_acquisition_race"] = True
        return raced

    error = ""
    try:
        for step_id, command in command_plan(config):
            receipt = run_step(step_id, command)
            lease["steps"].append(receipt)
            write_json_atomic(active_lease, lease)
            if receipt["returncode"] != 0:
                raise RuntimeError(
                    f"step_failed:{step_id}:returncode={receipt['returncode']}"
                )
        verdict = read_json(resolve(str(config["architecture_verdict"])))
        if verdict.get("trigger_state") != "GREEN":
            raise RuntimeError(
                f"architecture_verdict_invalid:{verdict.get('trigger_state')}"
            )
        lease["state"] = "COMPLETED_ONCE"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}:{exc}"[:4000]
        lease["state"] = "TERMINAL_INCOMPLETE_OR_INVALID_OBSERVATION"
    lease["completed_utc"] = training.now()
    lease["error"] = error
    lease["rerun_authorized"] = False
    lease["consumed_failure_is_model_or_architecture_negative_evidence"] = False
    write_json_atomic(archive_path, lease)
    try:
        active_lease.unlink()
    except FileNotFoundError:
        pass
    return {
        "policy": POLICY,
        "created_utc": training.now(),
        "trigger_state": "GREEN" if not error else "RED",
        "execution_complete": not error,
        "lease": artifact(archive_path),
        "source_binding": before["source_binding"],
        "freeze_sha256": before["freeze_sha256"],
        "steps": lease["steps"],
        "architecture_verdict": (
            artifact(resolve(str(config["architecture_verdict"])))
            if resolve(str(config["architecture_verdict"])).is_file()
            else {}
        ),
        "error": error,
        "rerun_authorized": False,
        "user_or_operator_action_required": False,
        "maximum_inference": before["maximum_inference"],
        "local_rater_model_acquisition": acquisition,
    }


def command_plan(config: dict[str, Any]) -> list[tuple[str, list[str]]]:
    python = str(resolve(str(config["qualified_python"])))
    freeze = str(resolve(str(config["freeze"])))
    functional_config = str(resolve(str(config["functional_config"])))
    manifest = str(resolve(str(config["manifest"])))
    packet = str(resolve(str(config["candidate_packet"])))
    exact = str(resolve(str(config["exact_diagnostic"])))
    exact_markdown = str(resolve(str(config["exact_diagnostic_markdown"])))
    plan: list[tuple[str, list[str]]] = [
        (
            "exact_recovery_diagnostic",
            [
                python,
                str(resolve("scripts/moecot_dense_exact_recovery_diagnostic.py")),
                "--freeze",
                freeze,
                "--out",
                exact,
                "--markdown-out",
                exact_markdown,
                "--gate",
            ],
        )
    ]
    for target in TARGETS:
        bundle = str(resolve(config["candidate_bundles"][target]))
        blind = str(resolve(config["blind_packets"][target]))
        qualification = str(resolve(config["qualifications"][target]))
        plan.append(
            (
                f"candidate_generation:{target}",
                [
                    python,
                    str(resolve("scripts/neural_seed_functional_generate.py")),
                    "--target",
                    target,
                    "--config",
                    str(resolve(str(config["training_config"]))),
                    "--freeze",
                    freeze,
                    "--packet",
                    packet,
                    "--out",
                    bundle,
                ],
            )
        )
        plan.append(
            (
                f"code_verification:{target}",
                [
                    python,
                    str(resolve("scripts/neural_seed_functional_utility.py")),
                    "--config",
                    functional_config,
                    "--freeze-out",
                    freeze,
                    "--manifest-out",
                    manifest,
                    "--packet-out",
                    packet,
                    "--evaluate-candidates",
                    bundle,
                    "--blind-english-packet-out",
                    blind,
                    "--out",
                    qualification,
                ],
            )
        )
    rater_command = [
        python,
        str(resolve("scripts/neural_seed_local_english_raters.py")),
        "--judgment-dir",
        str(resolve(str(config["judgment_directory"]))),
        "--receipt-out",
        str(resolve(str(config["judgment_receipt"]))),
    ]
    for target in TARGETS:
        rater_command.extend(
            [
                "--packet",
                f"{config['judgment_labels'][target]}={resolve(config['blind_packets'][target])}",
            ]
        )
    plan.append(("blind_local_english_scoring", rater_command))
    for target in TARGETS:
        label = str(config["judgment_labels"][target])
        plan.append(
            (
                f"final_qualification:{target}",
                [
                    python,
                    str(resolve("scripts/neural_seed_functional_utility.py")),
                    "--config",
                    functional_config,
                    "--freeze-out",
                    freeze,
                    "--manifest-out",
                    manifest,
                    "--packet-out",
                    packet,
                    "--evaluate-candidates",
                    str(resolve(config["candidate_bundles"][target])),
                    "--judgments",
                    str(resolve(str(config["judgment_directory"]))) + f"/{label}.jsonl",
                    "--judgment-receipt",
                    str(resolve(str(config["judgment_receipt"]))),
                    "--judgment-label",
                    label,
                    "--out",
                    str(resolve(config["qualifications"][target])),
                ],
            )
        )
    plan.append(
        (
            "architecture_verdict",
            [
                python,
                str(resolve("scripts/neural_seed_functional_utility.py")),
                "--config",
                functional_config,
                "--freeze-out",
                freeze,
                "--manifest-out",
                manifest,
                "--packet-out",
                packet,
                "--compare-results",
                *[str(resolve(config["qualifications"][target])) for target in TARGETS],
                "--exact-diagnostic",
                exact,
                "--out",
                str(resolve(str(config["architecture_verdict"]))),
            ],
        )
    )
    return plan


def checkpoint_audit(config: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    training_config_path = resolve(str(config["training_config"]))
    training_config = read_json(training_config_path)
    plan = training.build_plan(training_config, config_path=training_config_path)
    gaps = []
    rows = []
    for target_id in (
        "shared_trunk",
        *training.ARM_IDS,
        "dense_active_parameter",
        "dense_total_parameter",
    ):
        target = (plan.get("targets") or {}).get(target_id) or {}
        receipt_path = resolve(str(target.get("receipt") or ""))
        receipt = read_json(receipt_path) if receipt_path.is_file() else {}
        checkpoint = resolve(
            str(receipt.get("checkpoint") or target.get("checkpoint") or "")
        )
        row_gaps = []
        if not receipt.get("complete"):
            row_gaps.append("training_incomplete")
        plan_binding = checkpoint_plan_binding(receipt, plan, target)
        if plan_binding["state"] == "UNBOUND_PLAN_MISMATCH":
            row_gaps.append("plan_mismatch")
        if receipt.get("stage_signature") != freeze.get("training_stage_signature"):
            row_gaps.append("stage_mismatch")
        if not checkpoint.is_file() or sha256_file(checkpoint) != receipt.get(
            "checkpoint_sha256"
        ):
            row_gaps.append("checkpoint_identity_mismatch")
        gaps.extend(f"{target_id}:{gap}" for gap in row_gaps)
        rows.append(
            {
                "target_id": target_id,
                "receipt": relative(receipt_path),
                "checkpoint": relative(checkpoint),
                "plan_binding": plan_binding,
                "hard_gaps": row_gaps,
            }
        )
    return {
        "trigger_state": "GREEN" if not gaps else "PAUSED",
        "targets": rows,
        "hard_gaps": gaps,
    }


def checkpoint_plan_binding(
    receipt: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    receipt_sha256 = str(receipt.get("plan_sha256") or "")
    current_sha256 = str(plan.get("plan_sha256") or "")
    if receipt_sha256 and receipt_sha256 == current_sha256:
        return {
            "state": "EXACT_CURRENT_PLAN",
            "receipt_plan_sha256": receipt_sha256,
            "current_plan_sha256": current_sha256,
            "migration": {},
        }
    migration = training.accepted_plan_identity_migration(receipt, plan, target)
    if migration is not None:
        return {
            "state": "ACCEPTED_EXACT_IDENTITY_MIGRATION",
            "receipt_plan_sha256": receipt_sha256,
            "current_plan_sha256": current_sha256,
            "migration": migration,
        }
    return {
        "state": "UNBOUND_PLAN_MISMATCH",
        "receipt_plan_sha256": receipt_sha256,
        "current_plan_sha256": current_sha256,
        "migration": {},
    }


def local_model_audit() -> dict[str, Any]:
    config = read_json(utility.LOCAL_RATER_CONFIG)
    cache_root = resolve(str(config["model_cache_root"]))
    gaps = []
    rows = []
    for card in [*config["primary_raters"], config["adjudicator"]]:
        snapshot, fault = local_raters.local_snapshot(
            card,
            cache_root=cache_root,
        )
        if fault:
            gaps.append(f"{card['rater_id']}:{fault}")
        rows.append(
            {
                "rater_id": card["rater_id"],
                "snapshot": str(snapshot or ""),
                "available": snapshot is not None,
            }
        )
    return {
        "trigger_state": "GREEN" if not gaps else "PAUSED",
        "models": rows,
        "hard_gaps": gaps,
    }


def acquire_local_rater_models(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("local_rater_model_acquisition") or {}
    report_path = resolve(str(policy.get("report") or ""))
    gaps = []
    rows = []
    if policy.get("enabled") is not True:
        gaps.append("automatic_local_rater_acquisition_disabled")
    if policy.get("automatic_only_when_every_other_gate_is_green") is not True:
        gaps.append("local_rater_acquisition_gate_scope_invalid")
    if policy.get("static_weight_download_only") is not True:
        gaps.append("local_rater_acquisition_role_invalid")
    if int(policy.get("external_inference_calls", -1)) != 0:
        gaps.append("local_rater_acquisition_external_inference_nonzero")
    if int(policy.get("training_rows_written", -1)) != 0:
        gaps.append("local_rater_acquisition_training_rows_nonzero")
    rater_config = read_json(utility.LOCAL_RATER_CONFIG)
    configured_cache = resolve(str(rater_config["model_cache_root"]))
    policy_cache = resolve(str(policy.get("cache_root") or ""))
    if configured_cache != policy_cache:
        gaps.append("local_rater_acquisition_cache_binding_mismatch")
    if not gaps:
        from huggingface_hub import snapshot_download

        policy_cache.mkdir(parents=True, exist_ok=True)
        for card in [
            *rater_config["primary_raters"],
            rater_config["adjudicator"],
        ]:
            try:
                snapshot = Path(
                    snapshot_download(
                        repo_id=str(card["repo_id"]),
                        revision=str(card["revision"]),
                        cache_dir=str(policy_cache),
                        local_files_only=False,
                    )
                ).resolve()
                identity = local_raters.snapshot_identity(snapshot)
                rows.append(
                    {
                        "rater_id": card["rater_id"],
                        "repo_id": card["repo_id"],
                        "revision": card["revision"],
                        "license": card["license"],
                        "snapshot_identity": identity,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                gaps.append(f"{card['rater_id']}:{type(exc).__name__}:{exc}"[:2000])
                break
    report = {
        "policy": "project_theseus_neural_seed_d2_local_rater_acquisition_v1",
        "created_utc": training.now(),
        "trigger_state": "GREEN" if not gaps and len(rows) == 3 else "RED",
        "cache_root": relative(policy_cache),
        "models": rows,
        "hard_gaps": gaps,
        "static_weight_download_only": True,
        "external_inference_calls": 0,
        "training_rows_written": 0,
        "user_or_operator_approval_required": False,
    }
    if report_path:
        write_json_atomic(report_path, report)
    return report


def protected_outputs(config: dict[str, Any]) -> list[Path]:
    rows = [
        *[resolve(path) for path in config["candidate_bundles"].values()],
        *[resolve(path) for path in config["blind_packets"].values()],
        *[resolve(path) for path in config["qualifications"].values()],
        resolve(str(config["judgment_receipt"])),
        resolve(str(config["exact_diagnostic"])),
        resolve(str(config["exact_diagnostic_markdown"])),
        resolve(str(config["architecture_verdict"])),
    ]
    judgment_dir = resolve(str(config["judgment_directory"]))
    rows.extend(
        judgment_dir / f"{label}.jsonl" for label in config["judgment_labels"].values()
    )
    return rows


def run_step(step_id: str, command: list[str]) -> dict[str, Any]:
    started = training.now()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "step_id": step_id,
        "started_utc": started,
        "completed_utc": training.now(),
        "command": command,
        "returncode": completed.returncode,
        "output_tail": completed.stdout[-12000:],
    }


def recover_stale_lease(config: dict[str, Any]) -> None:
    active = resolve(str(config["active_lease"]))
    if not active.is_file():
        return
    lease = read_json(active)
    pid = int(lease.get("pid") or 0)
    if pid > 0 and process_alive(pid):
        return
    lease["state"] = "RECOVERED_STALE_TERMINAL_LEASE"
    lease["recovered_utc"] = training.now()
    lease["rerun_authorized"] = False
    archive_dir = resolve(str(config["lease_archive_directory"]))
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{lease.get('lease_id') or uuid.uuid4().hex}-stale.json"
    write_json_atomic(archive, lease)
    active.unlink()


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != POLICY:
        raise ValueError("D2 controller policy mismatch")
    for key in (
        "candidate_bundles",
        "blind_packets",
        "judgment_labels",
        "qualifications",
    ):
        if tuple((config.get(key) or {}).keys()) != TARGETS:
            raise ValueError(f"D2 controller target map mismatch:{key}")
    authority = config.get("authority") or {}
    required_true = (
        "require_clean_source",
        "require_exact_freeze_source_hashes",
        "require_all_checkpoints_complete",
        "require_no_competing_accelerator_job",
    )
    if any(authority.get(key) is not True for key in required_true):
        raise ValueError("D2 controller machine authority boundary missing")
    required_false = (
        "user_or_operator_approval_required",
        "rerun_consumed_identity_allowed",
        "physical_boundary_is_negative_evidence",
        "project_selected_quality_token_cap_allowed",
        "public_calibration_authorized",
        "external_inference_authorized",
        "serving_authorized",
        "training_row_admission_authorized",
    )
    if any(authority.get(key) is not False for key in required_false):
        raise ValueError("D2 controller forbidden authority enabled")
    acquisition = config.get("local_rater_model_acquisition") or {}
    if (
        acquisition.get("enabled") is not True
        or acquisition.get("automatic_only_when_every_other_gate_is_green") is not True
        or acquisition.get("static_weight_download_only") is not True
        or acquisition.get("exact_repo_revision_and_license_required") is not True
        or int(acquisition.get("external_inference_calls", -1)) != 0
        or int(acquisition.get("training_rows_written", -1)) != 0
    ):
        raise ValueError("D2 local-rater acquisition boundary invalid")


def source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "branch": branch,
        "clean_at_generation": not status,
        "dirty_path_count": len(status),
        "dirty_paths": status,
    }


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "sha256": sha256_file(path) if path.is_file() else "",
    }


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "policy",
            "created_utc",
            "trigger_state",
            "execution_authorized",
            "execution_complete",
            "failed_gates",
            "error",
            "rerun_authorized",
        )
        if key in report
    }


if __name__ == "__main__":
    sys.exit(main())
