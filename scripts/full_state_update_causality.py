#!/usr/bin/env python3
"""Content-addressed full-state update, rollback, and deletion mechanics.

This module is consumed by the canonical training-data lineage audit. It does
not train a model or claim behavioral unlearning. It proves that every state
class which a future update can mutate has an identity, authority, lineage,
storage, rollback, and deletion disposition before long training begins.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "full_state_update_causality.json"


class FullStateCausalityFault(ValueError):
    """A fail-closed full-state contract violation."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FullStateCausalityFault("contract_not_object")
    required = {
        "policy", "schema_version", "owner", "required_artifact_kinds", "authority",
        "required_state_fields", "deletion_actions", "terminal_states", "claim_boundaries",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise FullStateCausalityFault(f"contract_missing_fields:{','.join(missing)}")
    kinds = payload["required_artifact_kinds"]
    if not isinstance(kinds, list) or len(kinds) != len(set(kinds)) or not kinds:
        raise FullStateCausalityFault("required_artifact_kinds_invalid")
    if set(kinds) != set(payload["deletion_actions"]):
        raise FullStateCausalityFault("deletion_action_coverage_mismatch")
    if int(payload.get("bounded_canary_max_optimizer_steps") or 0) > 8:
        raise FullStateCausalityFault("bounded_canary_exceeds_architecture_limit")
    return payload


def artifact(
    artifact_id: str,
    kind: str,
    payload: Any,
    *,
    parents: list[str],
    authority: str,
    deletion_action: str,
    replicas: list[str] | None = None,
    compensation: str = "none",
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "content_sha256": digest(payload),
        "payload": copy.deepcopy(payload),
        "parent_ids": list(parents),
        "storage_replicas": list(replicas or ["primary"]),
        "authority": authority,
        "lifecycle_state": "active",
        "deletion_action": deletion_action,
        "compensation": compensation,
    }


def build_reference_inventory(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    actions = contract["deletion_actions"]
    rows = [
        artifact("source:row-001", "source_candidate", {"row_sha256": "11" * 32}, parents=[], authority="data_governance_controller", deletion_action=actions["source_candidate"]),
        artifact("dataset:snapshot-001", "dataset_snapshot", {"candidate_ids": ["source:row-001"], "split": "train"}, parents=["source:row-001"], authority="data_governance_controller", deletion_action=actions["dataset_snapshot"]),
        artifact("model:pre", "model_parameters", {"w": [0.125, -0.25], "step": 0}, parents=["dataset:snapshot-001"], authority="canonical_training_controller", deletion_action=actions["model_parameters"]),
        artifact("optimizer:pre", "optimizer_state", {"m": [0.0, 0.0], "v": [0.0, 0.0], "step": 0}, parents=["model:pre"], authority="canonical_training_controller", deletion_action=actions["optimizer_state"]),
        artifact("scheduler:pre", "scheduler_state", {"step": 0, "learning_rate": 0.001}, parents=["optimizer:pre"], authority="canonical_training_controller", deletion_action=actions["scheduler_state"]),
        artifact("rng:pre", "rng_state", {"python": 13, "numpy": 17, "torch": 19, "mlx": 23}, parents=["model:pre"], authority="canonical_training_controller", deletion_action=actions["rng_state"]),
        artifact("cache:pre", "runtime_cache", {"keys": ["dataset:snapshot-001", "model:pre"]}, parents=["dataset:snapshot-001", "model:pre"], authority="runtime_cache_controller", deletion_action=actions["runtime_cache"]),
        artifact("index:pre", "retrieval_index", {"document_ids": ["source:row-001"]}, parents=["source:row-001"], authority="vcm_index_controller", deletion_action=actions["retrieval_index"]),
        artifact("checkpoint:final-000", "checkpoint_final", {"model": "model:pre", "optimizer": "optimizer:pre", "rng": "rng:pre"}, parents=["model:pre", "optimizer:pre", "scheduler:pre", "rng:pre"], authority="canonical_training_controller", deletion_action=actions["checkpoint_final"], replicas=["primary", "local_backup"]),
        artifact("checkpoint:best-000", "checkpoint_best", {"model": "model:pre", "metric": 0.0, "metric_contract": "frozen-private-utility-v1"}, parents=["model:pre", "checkpoint:final-000"], authority="frozen_independent_verifier", deletion_action=actions["checkpoint_best"], replicas=["primary", "local_backup"]),
        artifact("checkpoint:backup-000", "checkpoint_backup", {"checkpoint": "checkpoint:final-000"}, parents=["checkpoint:final-000"], authority="replacement_transaction_kernel", deletion_action=actions["checkpoint_backup"], replicas=["local_backup"]),
        artifact("report:baseline-000", "derived_report", {"checkpoint": "checkpoint:best-000", "support": "mechanics_only"}, parents=["checkpoint:best-000"], authority="evidence_store", deletion_action=actions["derived_report"]),
        artifact("effect:receipt-000", "external_effect_receipt", {"effect": "publish_local_pointer", "target": "checkpoint:best-000"}, parents=["checkpoint:best-000"], authority="replacement_transaction_kernel", deletion_action=actions["external_effect_receipt"], compensation="restore_previous_pointer"),
    ]
    inventory = {
        "policy": contract["policy"],
        "schema_version": contract["schema_version"],
        "created_utc": now(),
        "selection_metric_contract": "frozen-private-utility-v1",
        "best_checkpoint_id": "checkpoint:best-000",
        "final_checkpoint_id": "checkpoint:final-000",
        "artifacts": rows,
    }
    inventory["inventory_digest"] = inventory_digest(inventory)
    validate_inventory(inventory, contract)
    return inventory


def inventory_digest(inventory: dict[str, Any]) -> str:
    stable = {
        key: value for key, value in inventory.items()
        if key not in {"created_utc", "inventory_digest"}
    }
    return digest(stable)


def validate_inventory(inventory: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    rows = inventory.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise FullStateCausalityFault("inventory_artifacts_missing")
    required_fields = set(contract["required_state_fields"])
    by_id: dict[str, dict[str, Any]] = {}
    kinds: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise FullStateCausalityFault("artifact_not_object")
        missing = required_fields.difference(row)
        if missing:
            raise FullStateCausalityFault(f"artifact_missing_fields:{','.join(sorted(missing))}")
        artifact_id = str(row["artifact_id"])
        if not artifact_id or artifact_id in by_id:
            raise FullStateCausalityFault("artifact_identity_duplicate_or_empty")
        if row["content_sha256"] != digest(row.get("payload")):
            raise FullStateCausalityFault(f"artifact_content_digest_mismatch:{artifact_id}")
        if not row["storage_replicas"] or len(row["storage_replicas"]) != len(set(row["storage_replicas"])):
            raise FullStateCausalityFault(f"storage_replicas_invalid:{artifact_id}")
        if row["kind"] == "external_effect_receipt" and row.get("compensation") in {None, "", "none"}:
            raise FullStateCausalityFault("external_effect_compensation_missing")
        by_id[artifact_id] = row
        kinds.add(str(row["kind"]))
    missing_kinds = set(contract["required_artifact_kinds"]).difference(kinds)
    if missing_kinds:
        raise FullStateCausalityFault(f"required_artifact_kinds_missing:{','.join(sorted(missing_kinds))}")
    for row in rows:
        unknown = set(row["parent_ids"]).difference(by_id)
        if unknown:
            raise FullStateCausalityFault(f"unknown_parent:{row['artifact_id']}:{','.join(sorted(unknown))}")
    topological_order(by_id)
    best_id = str(inventory.get("best_checkpoint_id") or "")
    final_id = str(inventory.get("final_checkpoint_id") or "")
    if best_id == final_id or best_id not in by_id or final_id not in by_id:
        raise FullStateCausalityFault("best_final_authority_not_distinct")
    if by_id[best_id]["kind"] != "checkpoint_best" or by_id[final_id]["kind"] != "checkpoint_final":
        raise FullStateCausalityFault("best_final_kind_mismatch")
    authority = contract["authority"]
    if by_id[best_id]["authority"] != authority["best_checkpoint_selector"]:
        raise FullStateCausalityFault("best_checkpoint_selector_authority_mismatch")
    if by_id[final_id]["authority"] != authority["final_checkpoint_writer"]:
        raise FullStateCausalityFault("final_checkpoint_writer_authority_mismatch")
    if not inventory.get("selection_metric_contract"):
        raise FullStateCausalityFault("selection_metric_contract_missing")
    if inventory.get("inventory_digest") and inventory["inventory_digest"] != inventory_digest(inventory):
        raise FullStateCausalityFault("inventory_digest_mismatch")
    return {
        "artifact_count": len(rows),
        "artifact_kind_count": len(kinds),
        "lineage_edge_count": sum(len(row["parent_ids"]) for row in rows),
        "inventory_digest": inventory_digest(inventory),
        "topological_order": topological_order(by_id),
    }


def topological_order(by_id: dict[str, dict[str, Any]]) -> list[str]:
    children: dict[str, list[str]] = defaultdict(list)
    indegree = {artifact_id: 0 for artifact_id in by_id}
    for artifact_id, row in by_id.items():
        for parent in row["parent_ids"]:
            children[parent].append(artifact_id)
            indegree[artifact_id] += 1
    queue = deque(sorted(node for node, count in indegree.items() if count == 0))
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(by_id):
        raise FullStateCausalityFault("lineage_cycle_detected")
    return order


def write_state_package(inventory: dict[str, Any], directory: Path, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    validation = validate_inventory(inventory, contract)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for row in inventory["artifacts"]:
        filename = hashlib.sha256(row["artifact_id"].encode("utf-8")).hexdigest() + ".json"
        replica_refs = []
        for replica_id in row["storage_replicas"]:
            replica_dir = directory / "replicas" / replica_id
            replica_dir.mkdir(parents=True, exist_ok=True)
            target = replica_dir / filename
            temporary = target.with_suffix(".tmp")
            temporary.write_text(canonical(row) + "\n", encoding="utf-8")
            os.replace(temporary, target)
            replica_refs.append({
                "replica_id": replica_id,
                "path": str(target.relative_to(directory)),
                "file_sha256": file_sha256(target),
            })
        manifest_rows.append({"artifact_id": row["artifact_id"], "replicas": replica_refs})
    manifest = {
        "policy": contract["policy"],
        "inventory_digest": validation["inventory_digest"],
        "best_checkpoint_id": inventory["best_checkpoint_id"],
        "final_checkpoint_id": inventory["final_checkpoint_id"],
        "selection_metric_contract": inventory["selection_metric_contract"],
        "artifacts": manifest_rows,
    }
    manifest["package_digest"] = digest({key: value for key, value in manifest.items() if key != "package_digest"})
    target = directory / "manifest.json"
    temporary = directory / "manifest.tmp"
    temporary.write_text(canonical(manifest) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return manifest


def read_state_package(directory: Path, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_package = digest({key: value for key, value in manifest.items() if key != "package_digest"})
    if manifest.get("package_digest") != expected_package:
        raise FullStateCausalityFault("package_manifest_digest_mismatch")
    rows = []
    for ref in manifest.get("artifacts", []):
        replicas = ref.get("replicas")
        if not isinstance(replicas, list) or not replicas:
            raise FullStateCausalityFault(f"package_replica_manifest_missing:{ref.get('artifact_id')}")
        replica_rows = []
        for replica in replicas:
            path = directory / replica["path"]
            if not path.is_file() or file_sha256(path) != replica["file_sha256"]:
                raise FullStateCausalityFault(f"package_artifact_file_mismatch:{ref.get('artifact_id')}:{replica.get('replica_id')}")
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("artifact_id") != ref.get("artifact_id"):
                raise FullStateCausalityFault("package_artifact_identity_mismatch")
            replica_rows.append(row)
        if any(canonical(row) != canonical(replica_rows[0]) for row in replica_rows[1:]):
            raise FullStateCausalityFault(f"package_replica_divergence:{ref.get('artifact_id')}")
        declared = set(replica_rows[0].get("storage_replicas") or [])
        materialized = {str(replica.get("replica_id") or "") for replica in replicas}
        if declared != materialized:
            raise FullStateCausalityFault(f"package_replica_coverage_mismatch:{ref.get('artifact_id')}")
        rows.append(replica_rows[0])
    inventory = {
        "policy": manifest["policy"],
        "schema_version": contract["schema_version"],
        "selection_metric_contract": manifest["selection_metric_contract"],
        "best_checkpoint_id": manifest["best_checkpoint_id"],
        "final_checkpoint_id": manifest["final_checkpoint_id"],
        "artifacts": rows,
    }
    inventory["inventory_digest"] = inventory_digest(inventory)
    validate_inventory(inventory, contract)
    if inventory["inventory_digest"] != manifest["inventory_digest"]:
        raise FullStateCausalityFault("package_inventory_digest_mismatch")
    return inventory


def prepare_update(inventory: dict[str, Any], *, admitted_candidate_ids: list[str], optimizer_steps: int, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    validation = validate_inventory(inventory, contract)
    if optimizer_steps < 1 or optimizer_steps > int(contract["bounded_canary_max_optimizer_steps"]):
        raise FullStateCausalityFault("optimizer_step_authority_exceeded")
    known = {row["artifact_id"] for row in inventory["artifacts"] if row["kind"] == "source_candidate"}
    if not admitted_candidate_ids or set(admitted_candidate_ids).difference(known):
        raise FullStateCausalityFault("optimizer_exposure_lineage_unknown")
    return {
        "transaction_id": digest([validation["inventory_digest"], admitted_candidate_ids, optimizer_steps]),
        "state": "prepared",
        "pre_inventory": copy.deepcopy(inventory),
        "pre_inventory_digest": validation["inventory_digest"],
        "admitted_candidate_ids": sorted(admitted_candidate_ids),
        "optimizer_steps": optimizer_steps,
        "selection_metric_contract": inventory["selection_metric_contract"],
        "authority": contract["authority"]["update_owner"],
        "created_utc": now(),
    }


def apply_reference_update(transaction: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    if transaction.get("state") != "prepared":
        raise FullStateCausalityFault("transaction_not_prepared")
    updated = copy.deepcopy(transaction["pre_inventory"])
    rows = {row["artifact_id"]: row for row in updated["artifacts"]}
    steps = int(transaction["optimizer_steps"])
    mutations = {
        "model:pre": {"w": [0.125 + 0.001 * steps, -0.25 - 0.001 * steps], "step": steps},
        "optimizer:pre": {"m": [0.01, -0.01], "v": [0.001, 0.001], "step": steps},
        "scheduler:pre": {"step": steps, "learning_rate": 0.001 / (1 + steps)},
        "rng:pre": {"python": 13 + steps, "numpy": 17 + steps, "torch": 19 + steps, "mlx": 23 + steps},
        "cache:pre": {"keys": ["dataset:snapshot-001", "model:pre"], "epoch": steps},
        "index:pre": {"document_ids": ["source:row-001"], "epoch": steps},
    }
    for artifact_id, payload in mutations.items():
        rows[artifact_id]["payload"] = payload
        rows[artifact_id]["content_sha256"] = digest(payload)
    final = rows[updated["final_checkpoint_id"]]
    final["payload"] = {"model": "model:pre", "optimizer": "optimizer:pre", "rng": "rng:pre", "step": steps}
    final["content_sha256"] = digest(final["payload"])
    best = rows[updated["best_checkpoint_id"]]
    best["payload"] = {"model": "model:pre", "metric": 0.5, "metric_contract": transaction["selection_metric_contract"], "selected_step": steps}
    best["content_sha256"] = digest(best["payload"])
    updated["created_utc"] = now()
    updated["inventory_digest"] = inventory_digest(updated)
    validate_inventory(updated, contract)
    result = copy.deepcopy(transaction)
    result.update({
        "state": "committed",
        "post_inventory": updated,
        "post_inventory_digest": updated["inventory_digest"],
        "best_selection_receipt": {
            "selector": contract["authority"]["best_checkpoint_selector"],
            "metric_contract": transaction["selection_metric_contract"],
            "selected_checkpoint_id": updated["best_checkpoint_id"],
            "final_checkpoint_id": updated["final_checkpoint_id"],
            "roles_distinct": updated["best_checkpoint_id"] != updated["final_checkpoint_id"],
        },
    })
    return result


def rollback_update(transaction: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    if transaction.get("state") != "committed":
        raise FullStateCausalityFault("transaction_not_committed")
    restored = copy.deepcopy(transaction["pre_inventory"])
    validation = validate_inventory(restored, contract)
    if validation["inventory_digest"] != transaction.get("pre_inventory_digest"):
        raise FullStateCausalityFault("rollback_identity_mismatch")
    return {
        "transaction_id": transaction["transaction_id"],
        "state": "rolled_back",
        "restored_inventory": restored,
        "restored_inventory_digest": validation["inventory_digest"],
        "exact_pre_state_restored": validation["inventory_digest"] == transaction["pre_inventory_digest"],
        "rollback_authority": contract["authority"]["rollback_owner"],
    }


def plan_deletion(inventory: dict[str, Any], root_ids: set[str], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    validation = validate_inventory(inventory, contract)
    by_id = {row["artifact_id"]: row for row in inventory["artifacts"]}
    unknown = root_ids.difference(by_id)
    if unknown:
        raise FullStateCausalityFault(f"deletion_root_unknown:{','.join(sorted(unknown))}")
    children: dict[str, list[str]] = defaultdict(list)
    for row in inventory["artifacts"]:
        for parent in row["parent_ids"]:
            children[parent].append(row["artifact_id"])
    reached = set(root_ids)
    queue = deque(sorted(root_ids))
    while queue:
        parent = queue.popleft()
        for child in sorted(children[parent]):
            if child not in reached:
                reached.add(child)
                queue.append(child)
    actions = []
    for artifact_id in validation["topological_order"]:
        if artifact_id not in reached:
            continue
        row = by_id[artifact_id]
        actions.append({
            "artifact_id": artifact_id,
            "kind": row["kind"],
            "action": row["deletion_action"],
            "replicas": list(row["storage_replicas"]),
            "requires_behavioral_influence_measurement": row["kind"] == "model_parameters",
            "requires_retraction_or_compensation": row["kind"] in {"derived_report", "external_effect_receipt"},
        })
    return {
        "policy": contract["policy"],
        "inventory_digest": validation["inventory_digest"],
        "root_ids": sorted(root_ids),
        "reached_ids": sorted(reached),
        "closure_complete": len(reached) == len(by_id),
        "actions": actions,
        "logical_revocation_complete": bool(actions),
        "physical_erasure_claim_allowed": False,
        "behavioral_unlearning_claim_allowed": False,
    }


def execute_package_deletion(package_dir: Path, inventory: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    refs = {row["artifact_id"]: row for row in manifest["artifacts"]}
    erased = []
    retained = []
    for action in plan["actions"]:
        artifact_id = action["artifact_id"]
        replica_refs = refs[artifact_id].get("replicas") or []
        if action["action"].startswith("retain_with_"):
            retained.append({
                "artifact_id": artifact_id,
                "action": action["action"],
                "replicas": [
                    {
                        "replica_id": replica["replica_id"],
                        "file_sha256": file_sha256(package_dir / replica["path"]),
                    }
                    for replica in replica_refs
                ],
            })
        else:
            replica_receipts = []
            for replica in replica_refs:
                path = package_dir / replica["path"]
                if path.exists():
                    path.unlink()
                replica_receipts.append({
                    "replica_id": replica["replica_id"],
                    "path_absent": not path.exists(),
                })
            erased.append({
                "artifact_id": artifact_id,
                "action": action["action"],
                "replicas": replica_receipts,
                "all_replica_paths_absent": bool(replica_receipts) and all(row["path_absent"] for row in replica_receipts),
                "replica_coverage_exact": {row["replica_id"] for row in replica_receipts} == set(action["replicas"]),
            })
    receipt = {
        "policy": "project_theseus_storage_deletion_receipt_v1",
        "inventory_digest": inventory["inventory_digest"],
        "root_ids": plan["root_ids"],
        "erased": erased,
        "retained_with_retraction_or_compensation": retained,
        "all_target_files_absent": all(row["all_replica_paths_absent"] for row in erased),
        "all_declared_replicas_accounted": all(row["replica_coverage_exact"] for row in erased),
        "behavioral_influence_state": "unverified_requires_retraining_or_unlearning_evaluation",
        "privacy_erasure_scope": "bounded_local_package_only",
    }
    receipt["receipt_digest"] = digest(receipt)
    return receipt


def mutation_controls(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    base = build_reference_inventory(contract)
    controls: list[dict[str, Any]] = []

    def expect_fault(case_id: str, mutator: Any, expected: str) -> None:
        candidate = copy.deepcopy(base)
        mutator(candidate)
        observed = "accepted"
        try:
            validate_inventory(candidate, contract)
        except FullStateCausalityFault as exc:
            observed = str(exc)
        controls.append({"case_id": case_id, "passed": expected in observed, "expected": expected, "observed": observed})

    expect_fault("missing_rng", lambda value: value.update(artifacts=[row for row in value["artifacts"] if row["kind"] != "rng_state"]), "required_artifact_kinds_missing")
    expect_fault("implicit_best_equals_final", lambda value: value.update(best_checkpoint_id=value["final_checkpoint_id"]), "best_final_authority_not_distinct")
    expect_fault("lineage_cycle", lambda value: next(row for row in value["artifacts"] if row["artifact_id"] == "source:row-001")["parent_ids"].append("effect:receipt-000"), "lineage_cycle_detected")
    expect_fault("payload_tamper", lambda value: next(row for row in value["artifacts"] if row["artifact_id"] == "model:pre")["payload"].update(step=99), "artifact_content_digest_mismatch")
    expect_fault("missing_effect_compensation", lambda value: next(row for row in value["artifacts"] if row["kind"] == "external_effect_receipt").update(compensation="none"), "external_effect_compensation_missing")
    expect_fault("unknown_lineage_parent", lambda value: next(row for row in value["artifacts"] if row["kind"] == "checkpoint_backup")["parent_ids"].append("checkpoint:missing"), "unknown_parent")

    update_step_rejected = False
    try:
        prepare_update(base, admitted_candidate_ids=["source:row-001"], optimizer_steps=9, contract=contract)
    except FullStateCausalityFault as exc:
        update_step_rejected = str(exc) == "optimizer_step_authority_exceeded"
    controls.append({"case_id": "unbounded_optimizer_canary", "passed": update_step_rejected, "expected": "optimizer_step_authority_exceeded", "observed": "optimizer_step_authority_exceeded" if update_step_rejected else "accepted"})

    incomplete = plan_deletion(base, {"checkpoint:final-000"}, contract)
    controls.append({"case_id": "incomplete_deletion_root", "passed": not incomplete["closure_complete"], "expected": "closure_incomplete", "observed": "closure_incomplete" if not incomplete["closure_complete"] else "closed"})
    return {"case_count": len(controls), "passed_count": sum(bool(row["passed"]) for row in controls), "results": controls}


def run_reference_fixture(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    baseline = build_reference_inventory(contract)
    baseline_validation = validate_inventory(baseline, contract)
    transaction = prepare_update(baseline, admitted_candidate_ids=["source:row-001"], optimizer_steps=2, contract=contract)
    committed = apply_reference_update(transaction, contract)
    rollback = rollback_update(committed, contract)
    deletion_plan = plan_deletion(baseline, {"source:row-001"}, contract)
    controls = mutation_controls(contract)
    with tempfile.TemporaryDirectory(prefix="theseus-full-state-") as temporary:
        package_dir = Path(temporary) / "state-package"
        manifest = write_state_package(baseline, package_dir, contract)
        replayed = read_state_package(package_dir, contract)
        deletion_receipt = execute_package_deletion(package_dir, replayed, deletion_plan)
    gates = {
        "required_kinds_complete": baseline_validation["artifact_kind_count"] == len(contract["required_artifact_kinds"]),
        "checkpoint_package_roundtrip_exact": replayed["inventory_digest"] == baseline["inventory_digest"],
        "update_changes_full_state_identity": committed["post_inventory_digest"] != committed["pre_inventory_digest"],
        "best_final_authority_distinct": committed["best_selection_receipt"]["roles_distinct"],
        "rollback_exact": rollback["exact_pre_state_restored"],
        "deletion_descendant_closure_complete": deletion_plan["closure_complete"],
        "local_storage_erasure_receipts_complete": deletion_receipt["all_target_files_absent"] and deletion_receipt["all_declared_replicas_accounted"],
        "behavioral_unlearning_not_overclaimed": deletion_receipt["behavioral_influence_state"].startswith("unverified"),
        "mutation_controls_complete": controls["passed_count"] == controls["case_count"],
    }
    return {
        "policy": contract["policy"],
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "summary": {
            "artifact_count": baseline_validation["artifact_count"],
            "artifact_kind_count": baseline_validation["artifact_kind_count"],
            "lineage_edge_count": baseline_validation["lineage_edge_count"],
            "optimizer_steps": transaction["optimizer_steps"],
            "best_checkpoint_id": baseline["best_checkpoint_id"],
            "final_checkpoint_id": baseline["final_checkpoint_id"],
            "mutation_case_count": controls["case_count"],
            "mutation_passed_count": controls["passed_count"],
            "gate_count": len(gates),
            "passed_gate_count": sum(bool(value) for value in gates.values()),
        },
        "gates": gates,
        "baseline_inventory_digest": baseline["inventory_digest"],
        "package_digest": manifest["package_digest"],
        "committed_inventory_digest": committed["post_inventory_digest"],
        "rollback": {key: value for key, value in rollback.items() if key != "restored_inventory"},
        "best_selection_receipt": committed["best_selection_receipt"],
        "deletion_plan": deletion_plan,
        "storage_deletion_receipt": deletion_receipt,
        "mutation_controls": controls,
        "non_claims": [
            "This bounded fixture is architecture mechanics, not a trained-model result.",
            "Logical revocation and bounded local file erasure do not prove removal from undeclared external systems.",
            "Behavioral influence and machine unlearning remain unverified until measured after training.",
            "A best-checkpoint selection receipt does not establish model quality.",
        ],
    }


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()
