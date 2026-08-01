from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checkpoint_custody.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_custody", SCRIPT)
assert SPEC and SPEC.loader
custody = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(custody)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def fixture_project(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "project"
    root.mkdir()
    run_git(root, "init", "-q")
    run_git(root, "config", "user.name", "Custody Test")
    run_git(root, "config", "user.email", "custody@example.invalid")

    step = 8
    checkpoint_root = root / "checkpoints" / "active" / "shared_trunk"
    checkpoint_root.mkdir(parents=True)
    payloads = {
        "weights": b"weights-v1",
        "optimizer": b"optimizer-v1",
        "rng": b"rng-v1",
    }
    names = {
        "weights": ("weights.safetensors", f"weights.step-{step:08d}.safetensors"),
        "optimizer": (
            "optimizer.safetensors",
            f"optimizer.step-{step:08d}.safetensors",
        ),
        "rng": (
            "optimizer.mlx-rng.safetensors",
            f"optimizer.step-{step:08d}.mlx-rng.safetensors",
        ),
    }
    for key, (current, versioned) in names.items():
        versioned_path = checkpoint_root / versioned
        versioned_path.write_bytes(payloads[key])
        os.link(versioned_path, checkpoint_root / current)

    receipt = {
        "policy": "test_receipt",
        "optimizer_steps": step,
        "checkpoint": f"checkpoints/active/shared_trunk/{names['weights'][1]}",
        "checkpoint_sha256": sha256(checkpoint_root / names["weights"][1]),
        "optimizer_state": f"checkpoints/active/shared_trunk/{names['optimizer'][1]}",
        "optimizer_state_sha256": sha256(checkpoint_root / names["optimizer"][1]),
        "mlx_rng_state": f"checkpoints/active/shared_trunk/{names['rng'][1]}",
        "mlx_rng_state_sha256": sha256(checkpoint_root / names["rng"][1]),
        "external_inference_calls": 0,
    }
    write_json(checkpoint_root / "training_receipt.json", receipt)
    write_json(checkpoint_root / "training_heartbeat.json", {"global_step": step})

    lineage_root = root / "reports" / "lineage"
    for index in range(2):
        write_json(
            lineage_root / f"segment-{index}" / "manifest.json", {"segment": index}
        )
        write_json(lineage_root / f"segment-{index}" / "receipt.json", {"step": index})
    write_json(
        root / "reports" / "freeze.json",
        {
            "gates": {"operator_hold": {"hold_present": True, "hold_removed": False}},
            "functional_surface": {
                "consumed_case_count": 0,
                "matching_registry_lines": [],
                "integrity": {
                    "state": "VALID_FRESH_PRIVATE_SURFACE",
                    "equivalent_consumed_case_contract_sha256s": [],
                    "evaluation_authorized": False,
                },
            },
        },
    )
    write_json(root / "reports" / "audit.json", {"ok": True})
    write_json(root / "runtime" / "control" / "hold", {"hold": True})
    write_json(root / "configs" / "trainer.json", {"trainer": "test"})
    freeze = {
        "policy": "test_freeze",
        "report": "reports/freeze.json",
        "checkpoint_root": "checkpoints/active/shared_trunk",
        "lineage_root": "reports/lineage",
        "audit": "reports/audit.json",
        "trainer": "configs/trainer.json",
        "hold_marker": "runtime/control/hold",
        "expected": {"optimizer_steps": step, "lineage_manifest_count": 2},
        "boundaries": {"external_inference_calls": 0},
    }
    write_json(root / "configs" / "freeze.json", freeze)

    policy: dict[str, object] = {
        "custody": {
            "active_freeze": "configs/freeze.json",
            "source_ref": "HEAD",
            "require_clean_git": True,
            "require_distinct_device": False,
            "require_private_encrypted_acknowledgement": True,
            "destination_free_reserve_bytes": 0,
            "source_bundle_estimate_bytes": 0,
            "required_checkpoint_files": [
                "weights.safetensors",
                "weights.step-{optimizer_step:08d}.safetensors",
                "optimizer.safetensors",
                "optimizer.step-{optimizer_step:08d}.safetensors",
                "optimizer.mlx-rng.safetensors",
                "optimizer.step-{optimizer_step:08d}.mlx-rng.safetensors",
                "training_receipt.json",
                "training_heartbeat.json",
            ],
            "evidence_from_freeze_keys": [
                "report",
                "audit",
                "trainer",
                "hold_marker",
            ],
            "evidence_maps_from_freeze_keys": [],
            "lineage_root_key": "lineage_root",
        }
    }
    write_json(root / "configs" / "policy.json", policy)
    (root / ".gitignore").write_text(
        "/checkpoints/\n/reports/\n/runtime/\n", encoding="utf-8"
    )
    run_git(root, "add", "configs", ".gitignore")
    run_git(root, "commit", "-q", "-m", "fixture source")
    return root, policy


def test_plan_binds_exact_checkpoint_evidence_hold_and_source(tmp_path: Path) -> None:
    root, policy = fixture_project(tmp_path)
    plan = custody.build_custody_plan(root, policy, source_ref="HEAD")

    assert plan["ok"] is True
    assert plan["optimizer_steps"] == 8
    assert plan["summary"]["lineage_manifest_count"] == 2
    assert (
        plan["summary"]["logical_file_count"] > plan["summary"]["unique_object_count"]
    )
    assert plan["summary"]["deduplicated_bytes"] > 0
    assert all(row["passed"] for row in plan["gates"])


def test_plan_preserves_invalid_surface_evidence_without_claiming_freshness(
    tmp_path: Path,
) -> None:
    root, policy = fixture_project(tmp_path)
    freeze_report_path = root / "reports" / "freeze.json"
    freeze_report = json.loads(freeze_report_path.read_text(encoding="utf-8"))
    freeze_report["functional_surface"]["matching_registry_lines"] = [1, 3]
    freeze_report["functional_surface"]["integrity"] = {
        "state": "INVALID_EXACT_SURFACE_REUSE",
        "equivalent_consumed_case_contract_sha256s": ["old-contract"],
        "evaluation_authorized": False,
    }
    write_json(freeze_report_path, freeze_report)

    plan = custody.build_custody_plan(root, policy, source_ref="HEAD")

    assert plan["ok"] is True
    assert (
        plan["boundaries"]["functional_surface_equivalent_prior_consumption_detected"]
        is True
    )
    assert (
        plan["boundaries"]["functional_surface_integrity_state"]
        == "INVALID_EXACT_SURFACE_REUSE"
    )
    assert plan["boundaries"]["functional_surface_evaluation_authorized"] is False


def test_plan_fails_when_checkpoint_bytes_drift_from_receipt(tmp_path: Path) -> None:
    root, policy = fixture_project(tmp_path)
    current = root / "checkpoints" / "active" / "shared_trunk" / "weights.safetensors"
    current.unlink()
    current.write_bytes(b"drifted-current-weights")

    plan = custody.build_custody_plan(root, policy, source_ref="HEAD")

    assert plan["ok"] is False
    failed = {row["name"] for row in plan["gates"] if not row["passed"]}
    assert "current_and_step_weights_identical" in failed


def test_create_verify_and_restore_round_trip(tmp_path: Path) -> None:
    root, policy = fixture_project(tmp_path)
    plan = custody.build_custody_plan(root, policy, source_ref="HEAD")
    destination = tmp_path / "private-destination"
    destination.mkdir()

    created = custody.create_custody_bundle(
        root,
        policy,
        plan,
        destination=str(destination),
        acknowledged_private_encrypted=True,
    )

    assert created["ok"] is True
    bundle = Path(created["bundle"])
    verified = custody.verify_custody_bundle(bundle)
    assert verified["ok"] is True
    assert verified["verified_object_count"] == verified["object_count"]

    restore_root = tmp_path / "restore"
    restored = custody.restore_custody_bundle(bundle, restore_root)
    assert restored["ok"] is True
    assert restored["source_commit"] == run_git(root, "rev-parse", "HEAD")
    restored_weights = (
        restore_root
        / "payload"
        / "checkpoints"
        / "active"
        / "shared_trunk"
        / "weights.safetensors"
    )
    assert restored_weights.read_bytes() == b"weights-v1"
    restored_versioned_weights = (
        restore_root
        / "payload"
        / "checkpoints"
        / "active"
        / "shared_trunk"
        / "weights.step-00000008.safetensors"
    )
    assert restored_weights.stat().st_ino == restored_versioned_weights.stat().st_ino
    assert (restore_root / "restore_receipt.json").is_file()


def test_create_rejects_destination_inside_repository(tmp_path: Path) -> None:
    root, policy = fixture_project(tmp_path)
    plan = custody.build_custody_plan(root, policy, source_ref="HEAD")
    destination = root / "private"
    destination.mkdir()

    try:
        custody.create_custody_bundle(
            root,
            policy,
            plan,
            destination=str(destination),
            acknowledged_private_encrypted=True,
        )
    except custody.CustodyError as exc:
        assert "outside the repository" in str(exc)
    else:
        raise AssertionError("repository-local custody destination was accepted")


def test_create_requires_private_destination_acknowledgement(tmp_path: Path) -> None:
    root, policy = fixture_project(tmp_path)
    plan = custody.build_custody_plan(root, policy, source_ref="HEAD")
    destination = tmp_path / "destination"
    destination.mkdir()

    try:
        custody.create_custody_bundle(
            root,
            policy,
            plan,
            destination=str(destination),
            acknowledged_private_encrypted=False,
        )
    except custody.CustodyError as exc:
        assert "acknowledgement is required" in str(exc)
    else:
        raise AssertionError("unacknowledged custody destination was accepted")


def test_verify_rejects_tampered_object(tmp_path: Path) -> None:
    root, policy = fixture_project(tmp_path)
    plan = custody.build_custody_plan(root, policy, source_ref="HEAD")
    destination = tmp_path / "destination"
    destination.mkdir()
    created = custody.create_custody_bundle(
        root,
        policy,
        plan,
        destination=str(destination),
        acknowledged_private_encrypted=True,
    )
    bundle = Path(created["bundle"])
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    object_path = bundle / manifest["objects"][0]["object_path"]
    object_path.chmod(0o600)
    object_path.write_bytes(b"tampered")

    verified = custody.verify_custody_bundle(bundle)

    assert verified["ok"] is False
    assert any(
        fault.startswith("object_identity_mismatch:") for fault in verified["faults"]
    )
