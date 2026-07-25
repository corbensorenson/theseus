#!/usr/bin/env python3
"""Learned proposal and independent qualification for the canonical RDC ABI.

The proposer may rank only an explicit source-visible candidate denominator.
The qualifier is separately parameterized and may reject proposed relations.
Evaluator relevance labels are accepted only after selection for measurement and
never enter either inference path.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import host_resource_safety
from relational_dimension_compiler import ProposalReceipt, proposal_receipt


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_rdc_relation_learning_v1"


class RelationLearningFault(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RelationCandidateBatch:
    request: dict[str, Any]
    candidate_ids: tuple[str, ...]
    source_features: tuple[tuple[float, ...], ...]
    evidence_features: tuple[tuple[float, ...], ...]


def validate_batch(batch: RelationCandidateBatch) -> dict[str, Any]:
    count = len(batch.candidate_ids)
    if not count or len(set(batch.candidate_ids)) != count:
        raise RelationLearningFault("relation_candidate_denominator_invalid")
    if len(batch.source_features) != count or len(batch.evidence_features) != count:
        raise RelationLearningFault("relation_candidate_feature_count_mismatch")
    source_widths = {len(row) for row in batch.source_features}
    evidence_widths = {len(row) for row in batch.evidence_features}
    if len(source_widths) != 1 or not next(iter(source_widths), 0):
        raise RelationLearningFault("relation_source_feature_width_invalid")
    if len(evidence_widths) != 1 or not next(iter(evidence_widths), 0):
        raise RelationLearningFault("relation_evidence_feature_width_invalid")
    values = [value for rows in (batch.source_features, batch.evidence_features) for row in rows for value in row]
    if any(not isinstance(value, (int, float)) or value != value or abs(value) == float("inf") for value in values):
        raise RelationLearningFault("relation_feature_nonfinite")
    return {
        "valid": True,
        "denominator_count": count,
        "source_feature_width": next(iter(source_widths)),
        "evidence_feature_width": next(iter(evidence_widths)),
        "batch_digest": digest(asdict(batch)),
    }


def build_relation_proposer_qualifier(
    *,
    source_feature_dim: int,
    evidence_feature_dim: int,
    hidden_dim: int,
    mx: Any,
    nn: Any,
) -> Any:
    if min(source_feature_dim, evidence_feature_dim, hidden_dim) <= 0:
        raise ValueError("relation learner dimensions must be positive")

    class RelationProposerQualifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proposer_input = nn.Linear(source_feature_dim, hidden_dim, bias=True)
            self.proposer_output = nn.Linear(hidden_dim, 1, bias=True)
            self.qualifier_input = nn.Linear(
                source_feature_dim + evidence_feature_dim, hidden_dim, bias=True
            )
            self.qualifier_output = nn.Linear(hidden_dim, 1, bias=True)

        def __call__(self, source_features: Any, evidence_features: Any) -> dict[str, Any]:
            if source_features.shape[:-1] != evidence_features.shape[:-1]:
                raise ValueError("relation learner candidate shape mismatch")
            proposal_hidden = nn.silu(self.proposer_input(source_features))
            qualification_hidden = nn.silu(
                self.qualifier_input(mx.concatenate([source_features, evidence_features], axis=-1))
            )
            return {
                "proposal_logits": self.proposer_output(proposal_hidden)[..., 0],
                "qualification_logits": self.qualifier_output(qualification_hidden)[..., 0],
            }

    return RelationProposerQualifier()


def select_from_learned_scores(
    batch: RelationCandidateBatch,
    *,
    proposal_scores: Sequence[float],
    qualification_scores: Sequence[float],
    proposal_budget: int,
    qualification_threshold: float,
    known_relevant_ids: Sequence[str] = (),
) -> ProposalReceipt:
    """Select first, then attach evaluator-only recall labels to the receipt."""

    validate_batch(batch)
    count = len(batch.candidate_ids)
    if len(proposal_scores) != count or len(qualification_scores) != count:
        raise RelationLearningFault("relation_score_count_mismatch")
    if not 1 <= proposal_budget <= count:
        raise RelationLearningFault("relation_proposal_budget_invalid")
    ranking = sorted(range(count), key=lambda index: (-float(proposal_scores[index]), batch.candidate_ids[index]))
    proposed_indices = ranking[:proposal_budget]
    proposed = tuple(batch.candidate_ids[index] for index in proposed_indices)
    qualified = tuple(
        batch.candidate_ids[index]
        for index in proposed_indices
        if float(qualification_scores[index]) >= qualification_threshold
    )
    return proposal_receipt(
        request=batch.request,
        denominator_ids=batch.candidate_ids,
        proposed_ids=proposed,
        qualified_ids=qualified,
        known_relevant_ids=known_relevant_ids,
        proposal_sources=("learned_source_visible_proposer", "independent_learned_qualifier"),
    )


def mlx_relation_learning_canary(*, optimizer_steps: int = 128) -> dict[str, Any]:
    if not host_resource_safety.accelerator_child_authorized():
        return {
            "available": False,
            "passed": False,
            "fault": "ACCELERATOR_WATCHDOG_REQUIRED",
            "optimizer_steps": 0,
            "capability_claim": "NOT_EVALUATED",
        }
    payload = {"scripts": str(ROOT / "scripts"), "optimizer_steps": int(optimizer_steps)}
    code = r'''
import json, sys, tempfile
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
p=json.loads(sys.stdin.read()); sys.path.insert(0,p["scripts"])
from rdc_relation_learning import build_relation_proposer_qualifier
mx.random.seed(20260722)
source=mx.array([
 [1,0,0,0,1,0],[0,1,0,1,0,0],[0,0,1,0,0,1],[1,1,0,0,0,0],
 [0,1,1,0,1,0],[1,0,1,1,0,0],[1,1,1,0,0,1],[0,0,0,1,1,1],
],dtype=mx.float32)
evidence=mx.array([
 [1,0,1,0],[0,1,0,1],[1,1,0,0],[0,0,1,1],
 [1,0,0,1],[0,1,1,0],[1,1,1,0],[0,0,0,1],
],dtype=mx.float32)
proposal_y=mx.array([1,0,1,0,1,0,1,0],dtype=mx.float32)
qualify_y=mx.array([1,0,0,1,1,0,1,0],dtype=mx.float32)
model=build_relation_proposer_qualifier(source_feature_dim=6,evidence_feature_dim=4,hidden_dim=32,mx=mx,nn=nn)
optimizer=optim.AdamW(learning_rate=0.02)
def loss_fn(candidate):
 out=candidate(source,evidence)
 return mx.mean(nn.losses.binary_cross_entropy(out["proposal_logits"],proposal_y,with_logits=True))+mx.mean(nn.losses.binary_cross_entropy(out["qualification_logits"],qualify_y,with_logits=True))
value_and_grad=nn.value_and_grad(model,loss_fn); initial=loss_fn(model); mx.eval(initial)
proposal_gradient=False; qualifier_gradient=False
for _ in range(p["optimizer_steps"]):
 loss,grads=value_and_grad(model); optimizer.update(model,grads); mx.eval(loss,model.parameters(),optimizer.state)
 for name,value in utils.tree_flatten(grads):
  active=float(mx.max(mx.abs(value)).item())>0
  proposal_gradient=proposal_gradient or (active and name.startswith("proposer_"))
  qualifier_gradient=qualifier_gradient or (active and name.startswith("qualifier_"))
final=loss_fn(model); output=model(source,evidence); changed_source=model(source+mx.array([[0,0,0,0,0.5,0]],dtype=mx.float32),evidence); changed_evidence=model(source,evidence+mx.array([[0,0,0.5,0]],dtype=mx.float32)); mx.eval(final,output,changed_source,changed_evidence)
source_delta=float(mx.max(mx.abs(output["proposal_logits"]-changed_source["proposal_logits"])).item())
evidence_proposal_delta=float(mx.max(mx.abs(output["proposal_logits"]-changed_evidence["proposal_logits"])).item())
evidence_qualifier_delta=float(mx.max(mx.abs(output["qualification_logits"]-changed_evidence["qualification_logits"])).item())
with tempfile.TemporaryDirectory(prefix="theseus-rdc-learning-") as tmp:
 path=Path(tmp)/"model.safetensors"; model.save_weights(str(path)); reloaded=build_relation_proposer_qualifier(source_feature_dim=6,evidence_feature_dim=4,hidden_dim=32,mx=mx,nn=nn); reloaded.load_weights(str(path)); replay=reloaded(source,evidence); mx.eval(replay)
 reload_delta=max(float(mx.max(mx.abs(output[key]-replay[key])).item()) for key in output)
print(json.dumps({"initial_loss":float(initial.item()),"final_loss":float(final.item()),"proposal_gradient":proposal_gradient,"qualifier_gradient":qualifier_gradient,"source_proposal_delta":source_delta,"evidence_proposal_delta":evidence_proposal_delta,"evidence_qualifier_delta":evidence_qualifier_delta,"checkpoint_reload_max_abs_delta":reload_delta},sort_keys=True))
'''
    process = subprocess.run(
        [sys.executable, "-c", code], input=json.dumps(payload), text=True,
        capture_output=True, timeout=180,
    )
    if process.returncode:
        return {"available": False, "passed": False, "stderr_tail": process.stderr[-2000:]}
    observed = json.loads(process.stdout)
    checks = {
        "proposer_gradient": observed["proposal_gradient"],
        "qualifier_gradient": observed["qualifier_gradient"],
        "representative_overfit": observed["final_loss"] < observed["initial_loss"] * 0.1,
        "source_intervention": observed["source_proposal_delta"] > 1e-6,
        "qualifier_evidence_intervention": observed["evidence_qualifier_delta"] > 1e-6,
        "proposer_evidence_noninterference": observed["evidence_proposal_delta"] == 0.0,
        "checkpoint_reload": observed["checkpoint_reload_max_abs_delta"] == 0.0,
    }
    return {
        "available": True,
        "passed": all(checks.values()),
        "checks": checks,
        "observed": observed,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "capability_claim": "NOT_EVALUATED",
    }


def reference_batch() -> RelationCandidateBatch:
    return RelationCandidateBatch(
        request={"query": "who received the report"},
        candidate_ids=("transfer:1", "transfer:2", "ownership:1", "time:1"),
        source_features=((1, 0), (0.8, 0.1), (0.2, 0.7), (0.1, 0.9)),
        evidence_features=((1, 0), (0, 1), (1, 1), (0, 0)),
    )


def run_reference_suite() -> dict[str, Any]:
    batch = reference_batch()
    first = select_from_learned_scores(
        batch,
        proposal_scores=(0.9, 0.8, 0.2, 0.1),
        qualification_scores=(0.9, 0.2, 0.8, 0.1),
        proposal_budget=2,
        qualification_threshold=0.5,
        known_relevant_ids=("transfer:1",),
    )
    changed_labels = select_from_learned_scores(
        batch,
        proposal_scores=(0.9, 0.8, 0.2, 0.1),
        qualification_scores=(0.9, 0.2, 0.8, 0.1),
        proposal_budget=2,
        qualification_threshold=0.5,
        known_relevant_ids=("time:1",),
    )
    gates = {
        "denominator_observable": first.denominator_count == 4,
        "proposal_budget_exact": len(first.proposed_relation_ids) == 2,
        "qualification_is_subset": set(first.qualified_relation_ids).issubset(first.proposed_relation_ids),
        "evaluator_label_noninterference": first.proposed_relation_ids == changed_labels.proposed_relation_ids and first.qualified_relation_ids == changed_labels.qualified_relation_ids,
    }
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "gates": gates,
        "receipt": asdict(first) | {"proposal_recall": first.proposal_recall},
        "no_cheat_counters": {"public_training_rows": 0, "external_inference_calls": 0, "evaluator_labels_visible_to_inference": 0},
        "claim_boundary": "learned_selection_mechanics_only_not_relation_truth_recall_or_utility",
    }


if __name__ == "__main__":
    print(json.dumps(run_reference_suite(), indent=2, sort_keys=True))
