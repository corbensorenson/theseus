"""Private custody bundles for active, unpromoted Project Theseus checkpoints.

This complements the accepted-candidate backup manager. It never publishes or
pushes model state. A custody bundle is written only to an explicitly
acknowledged private destination and contains:

* a complete Git bundle for one clean source ref;
* content-addressed checkpoint and evidence objects;
* a manifest binding every logical path to exact bytes; and
* an independently runnable verifier and restore path.

The production policy requires the destination to be outside the repository and
on a different filesystem. Tests may disable the different-filesystem rule in a
temporary fixture policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "checkpoint_backup_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "create", "verify", "restore"])
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--local-config", default="")
    parser.add_argument("--destination", default="")
    parser.add_argument("--bundle", default="")
    parser.add_argument("--restore-root", default="")
    parser.add_argument("--source-ref", default="")
    parser.add_argument(
        "--acknowledge-private-encrypted-destination", action="store_true"
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    policy = require_object(read_json(resolve_from(ROOT, args.policy)), "backup_policy")
    custody = require_object(policy.get("custody"), "custody_policy")
    out = resolve_report_path(ROOT, custody, args.out)

    try:
        if args.command in {"plan", "create"}:
            source_ref = args.source_ref or str(custody.get("source_ref") or "HEAD")
            plan = build_custody_plan(ROOT, policy, source_ref=source_ref)
            if args.command == "plan":
                report = plan
            else:
                local_config_path = args.local_config or str(
                    custody.get("local_config")
                    or "configs/checkpoint_custody.local.json"
                )
                local_config = read_json(resolve_from(ROOT, local_config_path))
                destination = args.destination or str(
                    get_path(local_config, ["custody", "destination_root"], "")
                )
                acknowledged = bool(
                    args.acknowledge_private_encrypted_destination
                    or get_path(
                        local_config,
                        ["custody", "operator_attests_private_encrypted_destination"],
                        False,
                    )
                )
                report = create_custody_bundle(
                    ROOT,
                    policy,
                    plan,
                    destination=destination,
                    acknowledged_private_encrypted=acknowledged,
                )
        elif args.command == "verify":
            if not args.bundle:
                raise CustodyError("--bundle is required for verify")
            report = verify_custody_bundle(resolve_from(Path.cwd(), args.bundle))
        else:
            if not args.bundle or not args.restore_root:
                raise CustodyError(
                    "--bundle and --restore-root are required for restore"
                )
            report = restore_custody_bundle(
                resolve_from(Path.cwd(), args.bundle),
                resolve_from(Path.cwd(), args.restore_root),
            )
    except (CustodyError, OSError, subprocess.SubprocessError, ValueError) as exc:
        report = {
            "policy": "project_theseus_checkpoint_custody_report_v1",
            "created_utc": now(),
            "ok": False,
            "status": "blocked",
            "command": args.command,
            "reason": str(exc),
            "external_inference_calls": 0,
        }

    if out is not None:
        write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


class CustodyError(RuntimeError):
    """Fail-closed custody preflight or verification error."""


def build_custody_plan(
    root: Path, policy: dict[str, Any], *, source_ref: str
) -> dict[str, Any]:
    custody = require_object(policy.get("custody"), "custody_policy")
    freeze_path = safe_project_path(root, str(custody.get("active_freeze") or ""))
    freeze = require_object(read_json(freeze_path), "active_freeze")
    freeze_report_path = safe_project_path(root, str(freeze.get("report") or ""))
    freeze_report = require_object(
        read_json(freeze_report_path), "active_freeze_report"
    )
    checkpoint_root = safe_project_path(root, str(freeze.get("checkpoint_root") or ""))
    receipt_path = checkpoint_root / "training_receipt.json"
    receipt = require_object(read_json(receipt_path), "training_receipt")

    expected = require_object(freeze.get("expected"), "active_freeze.expected")
    expected_step = int(expected.get("optimizer_steps") or 0)
    receipt_step = int(receipt.get("optimizer_steps") or 0)
    required_names = render_required_checkpoint_files(custody, expected_step)

    gates: list[dict[str, Any]] = []
    gate(gates, "expected_step_is_positive", expected_step > 0, expected_step)
    gate(
        gates,
        "receipt_step_matches_freeze",
        receipt_step == expected_step,
        receipt_step,
    )
    gate(
        gates,
        "operator_hold_present",
        get_path(freeze_report, ["gates", "operator_hold", "hold_present"], False)
        is True,
        get_path(freeze_report, ["gates", "operator_hold"], {}),
    )
    gate(
        gates,
        "operator_hold_not_removed",
        get_path(freeze_report, ["gates", "operator_hold", "hold_removed"], True)
        is False,
        get_path(freeze_report, ["gates", "operator_hold"], {}),
    )
    functional_surface = require_object(
        freeze_report.get("functional_surface"), "functional_surface"
    )
    surface_integrity = require_object(
        functional_surface.get("integrity"), "functional_surface.integrity"
    )
    matching_registry_lines = functional_surface.get("matching_registry_lines") or []
    integrity_state = str(surface_integrity.get("state") or "")
    equivalent_prior_consumption = bool(
        surface_integrity.get("equivalent_consumed_case_contract_sha256s")
        or matching_registry_lines
    )
    integrity_recorded = (
        integrity_state
        in {"VALID_FRESH_PRIVATE_SURFACE", "INVALID_EXACT_SURFACE_REUSE"}
        and surface_integrity.get("evaluation_authorized") is False
        and (
            not equivalent_prior_consumption
            or integrity_state == "INVALID_EXACT_SURFACE_REUSE"
        )
    )
    gate(
        gates,
        "functional_surface_integrity_recorded",
        integrity_recorded,
        {
            "state": integrity_state,
            "equivalent_prior_consumption": equivalent_prior_consumption,
            "evaluation_authorized": surface_integrity.get("evaluation_authorized"),
        },
    )
    gate(
        gates,
        "freeze_external_inference_zero",
        int(get_path(freeze, ["boundaries", "external_inference_calls"], -1)) == 0,
        get_path(freeze, ["boundaries", "external_inference_calls"], None),
    )
    gate(
        gates,
        "receipt_external_inference_zero",
        receipt.get("external_inference_calls") == 0,
        receipt.get("external_inference_calls"),
    )

    missing_checkpoint = [
        name for name in required_names if not (checkpoint_root / name).is_file()
    ]
    gate(
        gates,
        "required_checkpoint_files_present",
        not missing_checkpoint,
        missing_checkpoint,
    )
    checkpoint_files = regular_files(checkpoint_root)
    checkpoint_names = {
        path.relative_to(checkpoint_root).as_posix() for path in checkpoint_files
    }
    gate(
        gates,
        "required_checkpoint_files_in_inventory",
        set(required_names).issubset(checkpoint_names),
        sorted(set(required_names) - checkpoint_names),
    )

    evidence_files = collect_evidence_files(root, custody, freeze_path, freeze)
    lineage_root = safe_project_path(
        root,
        str(freeze.get(str(custody.get("lineage_root_key") or "lineage_root")) or ""),
    )
    lineage_manifests = (
        sorted(lineage_root.glob("*/manifest.json")) if lineage_root.is_dir() else []
    )
    expected_lineage = int(expected.get("lineage_manifest_count") or 0)
    gate(
        gates,
        "lineage_manifest_count_matches",
        len(lineage_manifests) == expected_lineage,
        {"expected": expected_lineage, "observed": len(lineage_manifests)},
    )

    git = git_state(root, source_ref)
    gate(gates, "git_available", git.get("available") is True, git)
    gate(gates, "source_ref_resolves", bool(git.get("commit")), git.get("commit"))
    if bool(custody.get("require_clean_git", True)):
        gate(
            gates,
            "git_worktree_clean",
            git.get("dirty") is False,
            git.get("porcelain", []),
        )

    all_files = dedupe_paths([*checkpoint_files, *evidence_files])
    entries, objects = inventory_files(root, all_files, checkpoint_root=checkpoint_root)
    receipt_gates = checkpoint_receipt_gates(receipt, entries, expected_step)
    gates.extend(receipt_gates)
    ok = all(row["passed"] for row in gates)
    unique_bytes = sum(int(row["bytes"]) for row in objects)
    logical_bytes = sum(int(row["bytes"]) for row in entries)
    return {
        "policy": "project_theseus_checkpoint_custody_plan_v1",
        "created_utc": now(),
        "ok": ok,
        "status": "custody_plan_ready" if ok else "blocked_preflight",
        "source": git,
        "source_ref": source_ref,
        "active_freeze": relative(root, freeze_path),
        "active_freeze_report": relative(root, freeze_report_path),
        "checkpoint_root": relative(root, checkpoint_root),
        "optimizer_steps": expected_step,
        "entries": entries,
        "objects": objects,
        "summary": {
            "logical_file_count": len(entries),
            "unique_object_count": len(objects),
            "logical_bytes": logical_bytes,
            "unique_object_bytes": unique_bytes,
            "deduplicated_bytes": logical_bytes - unique_bytes,
            "lineage_manifest_count": len(lineage_manifests),
        },
        "gates": gates,
        "boundaries": {
            "operator_hold_preserved": True,
            "functional_surface_current_contract_consumed": (
                int(functional_surface.get("consumed_case_count") or 0) > 0
            ),
            "functional_surface_equivalent_prior_consumption_detected": (
                equivalent_prior_consumption
            ),
            "functional_surface_integrity_state": integrity_state,
            "functional_surface_evaluation_authorized": False,
            "training_started_or_resumed": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "capability_claim": False,
        },
    }


def create_custody_bundle(
    root: Path,
    policy: dict[str, Any],
    plan: dict[str, Any],
    *,
    destination: str,
    acknowledged_private_encrypted: bool,
) -> dict[str, Any]:
    custody = require_object(policy.get("custody"), "custody_policy")
    if not plan.get("ok"):
        raise CustodyError("custody plan is not green")
    if not destination:
        raise CustodyError(
            "No private destination configured. Set custody.destination_root in "
            "configs/checkpoint_custody.local.json or pass --destination."
        )
    if (
        bool(custody.get("require_private_encrypted_acknowledgement", True))
        and not acknowledged_private_encrypted
    ):
        raise CustodyError(
            "Private/encrypted destination acknowledgement is required; use the ignored local "
            "config or --acknowledge-private-encrypted-destination."
        )

    destination_root = Path(destination).expanduser().resolve()
    destination_report = validate_destination(root, destination_root, custody)
    reserve_bytes = int(custody.get("destination_free_reserve_bytes") or 0)
    required_bytes = int(get_path(plan, ["summary", "unique_object_bytes"], 0))
    required_bytes += (
        int(custody.get("source_bundle_estimate_bytes") or 0) + reserve_bytes
    )
    free_bytes = shutil.disk_usage(destination_root).free
    if free_bytes < required_bytes:
        raise CustodyError(
            f"Destination has {free_bytes} free bytes but custody requires at least {required_bytes}."
        )

    commit = str(get_path(plan, ["source", "commit"], ""))
    step = int(plan.get("optimizer_steps") or 0)
    bundle_name = f"theseus-step-{step:08d}-{commit[:12]}"
    final = destination_root / bundle_name
    if final.exists():
        raise CustodyError(f"Custody bundle already exists: {final}")
    partial = destination_root / f".{bundle_name}.partial-{uuid.uuid4().hex}"
    partial.mkdir(parents=False, exist_ok=False)
    try:
        source_bundle = partial / "source.bundle"
        run_git(
            root,
            [
                "bundle",
                "create",
                str(source_bundle),
                str(plan.get("source_ref") or "HEAD"),
            ],
        )
        source_verify = run_git(root, ["bundle", "verify", str(source_bundle)])
        source_row = file_identity(source_bundle)

        object_root = partial / "objects" / "sha256"
        for row in plan.get("objects", []):
            digest = str(row["sha256"])
            source = safe_project_path(root, str(row["source_path"]))
            target = object_root / digest[:2] / digest
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied = file_identity(target)
            if copied["sha256"] != digest or copied["bytes"] != int(row["bytes"]):
                raise CustodyError(
                    f"Object copy verification failed for {row['source_path']}"
                )
            os.chmod(target, 0o400)

        manifest = {
            "policy": "project_theseus_checkpoint_custody_manifest_v1",
            "created_utc": now(),
            "bundle_name": bundle_name,
            "source": {
                **plan["source"],
                "ref": plan.get("source_ref"),
                "bundle": "source.bundle",
                "bundle_bytes": source_row["bytes"],
                "bundle_sha256": source_row["sha256"],
                "bundle_verify": {
                    "returncode": source_verify.returncode,
                    "stdout_tail": source_verify.stdout[-2000:],
                    "stderr_tail": source_verify.stderr[-2000:],
                },
            },
            "active_freeze": plan.get("active_freeze"),
            "active_freeze_report": plan.get("active_freeze_report"),
            "checkpoint_root": plan.get("checkpoint_root"),
            "optimizer_steps": step,
            "entries": plan.get("entries", []),
            "objects": [
                {
                    "sha256": row["sha256"],
                    "bytes": row["bytes"],
                    "object_path": f"objects/sha256/{str(row['sha256'])[:2]}/{row['sha256']}",
                }
                for row in plan.get("objects", [])
            ],
            "summary": plan.get("summary", {}),
            "preflight_gates": plan.get("gates", []),
            "destination": destination_report,
            "boundaries": plan.get("boundaries", {}),
        }
        manifest_path = partial / "manifest.json"
        write_json(manifest_path, manifest)
        manifest_sha256 = sha256_file(manifest_path)
        (partial / "manifest.sha256").write_text(
            f"{manifest_sha256}  manifest.json\n", encoding="ascii"
        )
        os.chmod(manifest_path, 0o400)
        os.chmod(partial / "manifest.sha256", 0o400)
        partial.rename(final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    verification = verify_custody_bundle(final)
    if not verification.get("ok"):
        raise CustodyError(
            f"Created bundle failed independent verification: {verification.get('faults')}"
        )
    return {
        "policy": "project_theseus_checkpoint_custody_report_v1",
        "created_utc": now(),
        "ok": True,
        "status": "custody_bundle_created_and_verified",
        "bundle": str(final),
        "manifest_sha256": verification.get("manifest_sha256"),
        "source_commit": commit,
        "optimizer_steps": step,
        "summary": plan.get("summary", {}),
        "destination": destination_report,
        "verification": verification,
        "external_inference_calls": 0,
    }


def verify_custody_bundle(bundle: Path) -> dict[str, Any]:
    bundle = bundle.expanduser().resolve()
    faults: list[str] = []
    manifest_path = bundle / "manifest.json"
    digest_path = bundle / "manifest.sha256"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        manifest = {}
        faults.append("manifest_missing_or_invalid")
    expected_manifest_digest = ""
    if digest_path.is_file():
        expected_manifest_digest = (
            digest_path.read_text(encoding="ascii").strip().split()[0]
        )
    else:
        faults.append("manifest_digest_missing")
    observed_manifest_digest = (
        sha256_file(manifest_path) if manifest_path.is_file() else ""
    )
    if expected_manifest_digest != observed_manifest_digest:
        faults.append("manifest_digest_mismatch")

    object_rows = (
        manifest.get("objects") if isinstance(manifest.get("objects"), list) else []
    )
    verified_objects = 0
    for row in object_rows:
        if not isinstance(row, dict):
            faults.append("invalid_object_row")
            continue
        try:
            object_path = safe_bundle_path(bundle, str(row.get("object_path") or ""))
        except CustodyError as exc:
            faults.append(str(exc))
            continue
        if not object_path.is_file() or object_path.is_symlink():
            faults.append(f"object_missing_or_unsafe:{row.get('object_path')}")
            continue
        identity = file_identity(object_path)
        if identity["sha256"] != row.get("sha256") or identity["bytes"] != int(
            row.get("bytes") or -1
        ):
            faults.append(f"object_identity_mismatch:{row.get('object_path')}")
            continue
        verified_objects += 1

    entries = (
        manifest.get("entries") if isinstance(manifest.get("entries"), list) else []
    )
    known_objects = {
        str(row.get("sha256")) for row in object_rows if isinstance(row, dict)
    }
    for row in entries:
        if (
            not isinstance(row, dict)
            or str(row.get("sha256") or "") not in known_objects
        ):
            faults.append(
                f"entry_object_missing:{row.get('path') if isinstance(row, dict) else 'invalid'}"
            )

    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    try:
        source_bundle = safe_bundle_path(bundle, str(source.get("bundle") or ""))
    except CustodyError as exc:
        source_bundle = Path()
        faults.append(str(exc))
    source_verify: dict[str, Any] = {}
    if not source_bundle.is_file():
        faults.append("source_bundle_missing")
    else:
        source_identity = file_identity(source_bundle)
        if source_identity["sha256"] != source.get("bundle_sha256"):
            faults.append("source_bundle_digest_mismatch")
        if source_identity["bytes"] != int(source.get("bundle_bytes") or -1):
            faults.append("source_bundle_size_mismatch")
        result = verify_git_bundle(source_bundle)
        source_verify = {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
        if result.returncode != 0:
            faults.append("source_bundle_git_verify_failed")

    return {
        "policy": "project_theseus_checkpoint_custody_verification_v1",
        "created_utc": now(),
        "ok": not faults,
        "status": "custody_bundle_verified" if not faults else "custody_bundle_invalid",
        "bundle": str(bundle),
        "manifest_sha256": observed_manifest_digest,
        "source_commit": source.get("commit"),
        "optimizer_steps": manifest.get("optimizer_steps"),
        "entry_count": len(entries),
        "object_count": len(object_rows),
        "verified_object_count": verified_objects,
        "source_bundle_verify": source_verify,
        "faults": faults,
        "external_inference_calls": 0,
    }


def restore_custody_bundle(bundle: Path, restore_root: Path) -> dict[str, Any]:
    verification = verify_custody_bundle(bundle)
    if not verification.get("ok"):
        raise CustodyError("Custody bundle verification failed before restore.")
    if restore_root.exists() and any(restore_root.iterdir()):
        raise CustodyError(f"Restore root must be absent or empty: {restore_root}")
    restore_root.mkdir(parents=True, exist_ok=True)

    manifest = require_object(read_json(bundle / "manifest.json"), "custody_manifest")
    source = require_object(manifest.get("source"), "custody_manifest.source")
    source_dir = restore_root / "source"
    clone = subprocess.run(
        ["git", "clone", str(bundle / str(source["bundle"])), str(source_dir)],
        text=True,
        capture_output=True,
    )
    if clone.returncode != 0:
        raise CustodyError(f"Git bundle restore failed: {clone.stderr[-2000:]}")
    restored_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_dir, text=True
    ).strip()
    if restored_commit != str(source.get("commit") or ""):
        raise CustodyError(
            f"Restored source commit {restored_commit} does not match manifest {source.get('commit')}."
        )

    payload_root = restore_root / "payload"
    restored_link_groups: dict[str, Path] = {}
    restored_object_digests: set[str] = set()
    restored_count = 0
    for row in manifest.get("entries", []):
        if not isinstance(row, dict):
            raise CustodyError("Invalid entry row during restore.")
        relative_path = safe_relative_path(str(row.get("path") or ""))
        target = payload_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = str(row.get("sha256") or "")
        link_group = str(row.get("link_group") or "")
        if not link_group:
            raise CustodyError(f"Missing link group during restore: {relative_path}")
        if link_group in restored_link_groups:
            os.link(restored_link_groups[link_group], target)
        else:
            object_path = bundle / "objects" / "sha256" / digest[:2] / digest
            shutil.copyfile(object_path, target)
            restored_link_groups[link_group] = target
        restored_object_digests.add(digest)
        os.chmod(target, int(row.get("mode") or 0o600))
        identity = file_identity(target)
        if identity["sha256"] != digest or identity["bytes"] != int(
            row.get("bytes") or -1
        ):
            raise CustodyError(f"Restored payload verification failed: {relative_path}")
        restored_count += 1

    report = {
        "policy": "project_theseus_checkpoint_custody_restore_v1",
        "created_utc": now(),
        "ok": True,
        "status": "custody_restore_verified",
        "bundle": str(bundle),
        "restore_root": str(restore_root),
        "source_commit": restored_commit,
        "optimizer_steps": manifest.get("optimizer_steps"),
        "restored_entry_count": restored_count,
        "restored_unique_object_count": len(restored_object_digests),
        "restored_link_group_count": len(restored_link_groups),
        "pre_restore_verification": verification,
        "external_inference_calls": 0,
    }
    write_json(restore_root / "restore_receipt.json", report)
    return report


def collect_evidence_files(
    root: Path,
    custody: dict[str, Any],
    freeze_path: Path,
    freeze: dict[str, Any],
) -> list[Path]:
    paths = [freeze_path]
    for key in custody.get("evidence_from_freeze_keys") or []:
        value = freeze.get(str(key))
        if isinstance(value, str) and value:
            paths.append(safe_project_path(root, value))
    for key in custody.get("evidence_maps_from_freeze_keys") or []:
        value = freeze.get(str(key))
        if isinstance(value, dict):
            for path_value in value.values():
                if isinstance(path_value, str) and path_value:
                    paths.append(safe_project_path(root, path_value))
    lineage_key = str(custody.get("lineage_root_key") or "lineage_root")
    lineage_value = freeze.get(lineage_key)
    if isinstance(lineage_value, str) and lineage_value:
        lineage_root = safe_project_path(root, lineage_value)
        if not lineage_root.is_dir():
            raise CustodyError(f"Lineage root is missing: {lineage_value}")
        paths.extend(regular_files(lineage_root))
    missing = [relative(root, path) for path in paths if not path.is_file()]
    if missing:
        raise CustodyError(f"Required custody evidence is missing: {missing}")
    return dedupe_paths(paths)


def inventory_files(
    root: Path,
    paths: Iterable[Path],
    *,
    checkpoint_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    object_sources: dict[str, dict[str, Any]] = {}
    inode_groups: dict[tuple[int, int], str] = {}
    for path in sorted(paths, key=lambda item: relative(root, item)):
        if path.is_symlink() or not path.is_file():
            raise CustodyError(
                f"Custody input must be a regular non-symlink file: {relative(root, path)}"
            )
        stat = path.stat()
        inode_key = (stat.st_dev, stat.st_ino)
        if inode_key not in inode_groups:
            inode_groups[inode_key] = f"link-{len(inode_groups) + 1:05d}"
        identity = file_identity(path)
        project_path = relative(root, path)
        row = {
            "path": project_path,
            "kind": "checkpoint"
            if is_relative_to(path, checkpoint_root)
            else "evidence",
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
            "mode": stat.st_mode & 0o777,
            "link_group": inode_groups[inode_key],
        }
        entries.append(row)
        object_sources.setdefault(
            identity["sha256"],
            {
                "sha256": identity["sha256"],
                "bytes": identity["bytes"],
                "source_path": project_path,
            },
        )
    objects = sorted(object_sources.values(), key=lambda row: str(row["sha256"]))
    return entries, objects


def checkpoint_receipt_gates(
    receipt: dict[str, Any],
    entries: list[dict[str, Any]],
    expected_step: int,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    identities = {str(row["path"]): row for row in entries}
    declared = [
        ("weights", receipt.get("checkpoint"), receipt.get("checkpoint_sha256")),
        (
            "optimizer",
            receipt.get("optimizer_state"),
            receipt.get("optimizer_state_sha256"),
        ),
        ("mlx_rng", receipt.get("mlx_rng_state"), receipt.get("mlx_rng_state_sha256")),
    ]
    for label, path_value, digest in declared:
        path_text = str(path_value or "")
        observed = identities.get(path_text, {}).get("sha256")
        gate(
            gates,
            f"receipt_{label}_identity_matches",
            bool(path_text and digest and observed == digest),
            {"path": path_text, "declared": digest, "observed": observed},
        )

    root_value = Path(str(receipt.get("checkpoint") or "")).parent.as_posix()
    aliases = [
        (
            "weights",
            "weights.safetensors",
            f"weights.step-{expected_step:08d}.safetensors",
        ),
        (
            "optimizer",
            "optimizer.safetensors",
            f"optimizer.step-{expected_step:08d}.safetensors",
        ),
        (
            "mlx_rng",
            "optimizer.mlx-rng.safetensors",
            f"optimizer.step-{expected_step:08d}.mlx-rng.safetensors",
        ),
    ]
    for label, current_name, step_name in aliases:
        current = identities.get(f"{root_value}/{current_name}", {})
        versioned = identities.get(f"{root_value}/{step_name}", {})
        gate(
            gates,
            f"current_and_step_{label}_identical",
            bool(
                current
                and versioned
                and current.get("sha256") == versioned.get("sha256")
            ),
            {
                "current": current.get("sha256"),
                "versioned": versioned.get("sha256"),
            },
        )
    return gates


def validate_destination(
    root: Path, destination: Path, custody: dict[str, Any]
) -> dict[str, Any]:
    if not destination.exists() or not destination.is_dir():
        raise CustodyError(
            f"Custody destination must already exist as a directory: {destination}"
        )
    if is_relative_to(destination, root) or destination == root:
        raise CustodyError("Custody destination must be outside the repository.")
    root_device = root.stat().st_dev
    destination_device = destination.stat().st_dev
    distinct_device = root_device != destination_device
    if bool(custody.get("require_distinct_device", True)) and not distinct_device:
        raise CustodyError(
            "Custody destination must be on a different filesystem/device."
        )
    return {
        "path": str(destination),
        "outside_repository": True,
        "distinct_device": distinct_device,
        "private_encrypted_destination_acknowledged": True,
    }


def render_required_checkpoint_files(custody: dict[str, Any], step: int) -> list[str]:
    values = custody.get("required_checkpoint_files") or []
    return [str(value).format(optimizer_step=step) for value in values]


def git_state(root: Path, source_ref: str) -> dict[str, Any]:
    try:
        branch = git_output(root, ["branch", "--show-current"])
        commit = git_output(root, ["rev-parse", source_ref])
        porcelain = git_output(root, ["status", "--short"]).splitlines()
        return {
            "available": True,
            "branch": branch,
            "commit": commit,
            "dirty": bool(porcelain),
            "porcelain_count": len(porcelain),
            "porcelain": porcelain,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc), "commit": ""}


def run_git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if result.returncode != 0:
        raise CustodyError(
            f"git {' '.join(args)} failed: {(result.stderr or result.stdout)[-2000:]}"
        )
    return result


def verify_git_bundle(bundle: Path) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="theseus-custody-verify-") as temp:
        bare = Path(temp) / "verify.git"
        init = subprocess.run(
            ["git", "init", "--bare", "-q", str(bare)],
            text=True,
            capture_output=True,
        )
        if init.returncode != 0:
            return init
        return subprocess.run(
            ["git", "-C", str(bare), "bundle", "verify", str(bundle)],
            text=True,
            capture_output=True,
        )


def git_output(root: Path, args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def regular_files(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise CustodyError(f"Custody directory is missing or unsafe: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CustodyError(f"Custody directory contains symlink: {path}")
        if path.is_file():
            files.append(path)
    return files


def file_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any) -> None:
    rows.append({"name": name, "passed": bool(passed), "observed": observed})


def safe_project_path(root: Path, value: str) -> Path:
    if not value:
        raise CustodyError("Empty project path in custody policy.")
    path = resolve_from(root, value)
    if not is_relative_to(path, root):
        raise CustodyError(f"Custody input escapes project root: {value}")
    return path


def safe_bundle_path(bundle: Path, value: str) -> Path:
    relative_path = safe_relative_path(value)
    path = (bundle / relative_path).resolve()
    if not is_relative_to(path, bundle):
        raise CustodyError(f"Bundle path escapes bundle root: {value}")
    return path


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise CustodyError(f"Unsafe relative path: {value}")
    return path


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_report_path(root: Path, custody: dict[str, Any], value: str) -> Path | None:
    configured = value or str(get_path(custody, ["reports", "last"], ""))
    return resolve_from(root, configured) if configured else None


def relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    return [
        Path(value) for value in dict.fromkeys(str(path.resolve()) for path in paths)
    ]


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise CustodyError(f"{label} is missing or invalid")
    return value


def read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
