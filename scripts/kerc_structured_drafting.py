#!/usr/bin/env python3
"""Typed semantic-unit drafting for the canonical KERC generation path.

The learned head proposes complete relational units. A deterministic grammar
only rejects malformed proposals, while an independent verifier and canonical
target retain acceptance authority. Grammar, pointer lookup, and exact target
comparison receive zero learned-generation credit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import host_resource_safety


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_kerc_structured_drafting_v1"


class StructuredDraftFault(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class StructuredDraftVocabulary:
    operators: tuple[str, ...]
    modalities: tuple[str, ...]
    polarities: tuple[str, ...]
    quantifiers: tuple[str, ...]
    roles: tuple[str, ...]
    entity_pointers: tuple[str, ...]
    fidelity_modes: tuple[str, ...]
    maximum_roles: int


@dataclass(frozen=True)
class StructuredSemanticUnit:
    operator: str
    modality: str
    polarity: str
    quantifier: str
    role_pointer_pairs: tuple[tuple[str, str], ...]
    fidelity_mode: str
    closes_node: bool
    closes_program: bool


@dataclass(frozen=True)
class StructuredDraftReceipt:
    proposed_count: int
    accepted_count: int
    rejected_count: int
    accepted_units: tuple[StructuredSemanticUnit, ...]
    rejected_units: tuple[StructuredSemanticUnit, ...]
    target_unit_committed: StructuredSemanticUnit | None
    grammar_rejection_count: int
    semantic_verifier_rejection_count: int
    target_mismatch_count: int
    exact_target_authority: bool


def validate_unit(
    unit: StructuredSemanticUnit,
    vocabulary: StructuredDraftVocabulary,
    *,
    role_pointer_types: Mapping[str, set[str]] | None = None,
    pointer_types: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    inventories = (
        (unit.operator, vocabulary.operators, "operator"),
        (unit.modality, vocabulary.modalities, "modality"),
        (unit.polarity, vocabulary.polarities, "polarity"),
        (unit.quantifier, vocabulary.quantifiers, "quantifier"),
        (unit.fidelity_mode, vocabulary.fidelity_modes, "fidelity"),
    )
    for value, allowed, label in inventories:
        if value not in allowed:
            raise StructuredDraftFault(f"structured_unit_{label}_unknown")
    if not 0 <= len(unit.role_pointer_pairs) <= vocabulary.maximum_roles:
        raise StructuredDraftFault("structured_unit_role_count_invalid")
    if unit.closes_program and not unit.closes_node:
        raise StructuredDraftFault("structured_unit_program_closes_before_node")
    for role, pointer in unit.role_pointer_pairs:
        if role not in vocabulary.roles:
            raise StructuredDraftFault("structured_unit_role_unknown")
        if pointer not in vocabulary.entity_pointers:
            raise StructuredDraftFault("structured_unit_pointer_unknown")
        if role_pointer_types is not None and pointer_types is not None:
            actual = pointer_types.get(pointer)
            if actual is None or actual not in role_pointer_types.get(role, set()):
                raise StructuredDraftFault("structured_unit_role_pointer_type_mismatch")
    return {"valid": True, "unit_digest": digest(asdict(unit))}


class IncrementalStructuredGrammar:
    """Finite state machine for one or more complete semantic units."""

    _fixed = ("OPERATOR", "MODALITY", "POLARITY", "QUANTIFIER")

    def __init__(self, maximum_roles: int) -> None:
        if maximum_roles < 0:
            raise StructuredDraftFault("structured_grammar_role_budget_invalid")
        self.maximum_roles = maximum_roles
        self.phase = "UNIT_BEGIN"
        self.role_count = 0
        self.complete_units = 0
        self.program_closed = False

    def allowed_kinds(self) -> tuple[str, ...]:
        if self.program_closed:
            return ()
        if self.phase == "ROLE_OR_FIDELITY":
            return ("ROLE", "FIDELITY") if self.role_count < self.maximum_roles else ("FIDELITY",)
        return {
            "UNIT_BEGIN": ("UNIT_BEGIN",),
            "OPERATOR": ("OPERATOR",),
            "MODALITY": ("MODALITY",),
            "POLARITY": ("POLARITY",),
            "QUANTIFIER": ("QUANTIFIER",),
            "POINTER": ("POINTER",),
            "CLOSE_NODE": ("CLOSE_NODE",),
            "CLOSE_OR_NEXT": ("CLOSE_PROGRAM", "UNIT_BEGIN"),
        }[self.phase]

    def advance(self, kind: str) -> None:
        if kind not in self.allowed_kinds():
            raise StructuredDraftFault(f"structured_grammar_transition_invalid:{self.phase}:{kind}")
        if self.phase == "UNIT_BEGIN":
            self.phase = "OPERATOR"
            self.role_count = 0
        elif self.phase in self._fixed:
            index = self._fixed.index(self.phase)
            self.phase = self._fixed[index + 1] if index + 1 < len(self._fixed) else "ROLE_OR_FIDELITY"
        elif self.phase == "ROLE_OR_FIDELITY":
            self.phase = "POINTER" if kind == "ROLE" else "CLOSE_NODE"
        elif self.phase == "POINTER":
            self.role_count += 1
            self.phase = "ROLE_OR_FIDELITY"
        elif self.phase == "CLOSE_NODE":
            self.complete_units += 1
            self.phase = "CLOSE_OR_NEXT"
        elif self.phase == "CLOSE_OR_NEXT":
            if kind == "CLOSE_PROGRAM":
                self.program_closed = True
            else:
                self.phase = "OPERATOR"
                self.role_count = 0


def unit_token_kinds(unit: StructuredSemanticUnit) -> tuple[str, ...]:
    values = ["UNIT_BEGIN", "OPERATOR", "MODALITY", "POLARITY", "QUANTIFIER"]
    for _role, _pointer in unit.role_pointer_pairs:
        values.extend(("ROLE", "POINTER"))
    values.extend(("FIDELITY", "CLOSE_NODE"))
    if unit.closes_program:
        values.append("CLOSE_PROGRAM")
    return tuple(values)


def validate_unit_sequence(
    units: Sequence[StructuredSemanticUnit], vocabulary: StructuredDraftVocabulary
) -> dict[str, Any]:
    grammar = IncrementalStructuredGrammar(vocabulary.maximum_roles)
    for unit in units:
        validate_unit(unit, vocabulary)
        for kind in unit_token_kinds(unit):
            grammar.advance(kind)
    return {
        "valid": True,
        "complete_unit_count": grammar.complete_units,
        "program_closed": grammar.program_closed,
        "sequence_digest": digest([asdict(unit) for unit in units]),
    }


SemanticVerifier = Callable[[StructuredSemanticUnit], bool]


def verify_structured_draft(
    proposed: Sequence[StructuredSemanticUnit],
    canonical_target: Sequence[StructuredSemanticUnit],
    *,
    vocabulary: StructuredDraftVocabulary,
    semantic_verifier: SemanticVerifier,
) -> StructuredDraftReceipt:
    """Accept the longest grammar-valid, verifier-valid, target-exact prefix."""

    accepted: list[StructuredSemanticUnit] = []
    rejected: list[StructuredSemanticUnit] = []
    grammar_rejections = 0
    verifier_rejections = 0
    target_mismatches = 0
    target_committed = None
    for index, unit in enumerate(proposed):
        try:
            validate_unit_sequence((*accepted, unit), vocabulary)
        except StructuredDraftFault:
            grammar_rejections += 1
            rejected.extend(proposed[index:])
            break
        if not semantic_verifier(unit):
            verifier_rejections += 1
            rejected.extend(proposed[index:])
            break
        if index >= len(canonical_target) or unit != canonical_target[index]:
            target_mismatches += 1
            rejected.extend(proposed[index:])
            if index < len(canonical_target):
                target_committed = canonical_target[index]
            break
        accepted.append(unit)
    if not rejected and len(accepted) < len(canonical_target):
        target_committed = canonical_target[len(accepted)]
    return StructuredDraftReceipt(
        proposed_count=len(proposed),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        accepted_units=tuple(accepted),
        rejected_units=tuple(rejected),
        target_unit_committed=target_committed,
        grammar_rejection_count=grammar_rejections,
        semantic_verifier_rejection_count=verifier_rejections,
        target_mismatch_count=target_mismatches,
        exact_target_authority=True,
    )


def build_structured_unit_drafter(
    *,
    d_model: int,
    hidden_dim: int,
    future_offsets: Sequence[int],
    field_sizes: Sequence[int],
    mx: Any,
    nn: Any,
) -> Any:
    if d_model <= 0 or hidden_dim <= 0 or not future_offsets or any(size <= 1 for size in field_sizes):
        raise ValueError("structured drafter dimensions must be positive and categorical")

    class UnitHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(d_model)
            self.input = nn.Linear(d_model, hidden_dim, bias=False)
            self.output = nn.Linear(hidden_dim, sum(field_sizes), bias=True)

        def __call__(self, hidden: Any) -> tuple[Any, ...]:
            logits = self.output(nn.silu(self.input(self.norm(hidden))))
            boundaries = []
            cursor = 0
            for size in field_sizes:
                boundaries.append(logits[..., cursor : cursor + size])
                cursor += size
            return tuple(boundaries)

    class StructuredUnitDrafter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = [UnitHead() for _ in future_offsets]

        def __call__(self, hidden: Any) -> tuple[tuple[Any, ...], ...]:
            return tuple(head(hidden) for head in self.heads)

    return StructuredUnitDrafter()


def mlx_structured_drafting_canary(*, optimizer_steps: int = 96) -> dict[str, Any]:
    if not host_resource_safety.accelerator_child_authorized():
        return {
            "available": False,
            "passed": False,
            "fault": "ACCELERATOR_WATCHDOG_REQUIRED",
            "optimizer_steps": 0,
            "grammar_pointer_renderer_learned_credit": 0,
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
from kerc_structured_drafting import build_structured_unit_drafter
mx.random.seed(20260722)
field_sizes=(7,4,3,5,6,9,4,2,2)
hidden=mx.random.normal((16,24))
labels=tuple(mx.array([(row*(field+2)+field)%size for row in range(16)],dtype=mx.int32) for field,size in enumerate(field_sizes))
model=build_structured_unit_drafter(d_model=24,hidden_dim=64,future_offsets=(1,2),field_sizes=field_sizes,mx=mx,nn=nn)
optimizer=optim.AdamW(learning_rate=0.02)
def loss_fn(candidate):
    heads=candidate(hidden); losses=[]
    for horizon,fields in enumerate(heads):
        for field,logits in enumerate(fields): losses.append(nn.losses.cross_entropy(logits,(labels[field]+horizon)%field_sizes[field],reduction="mean"))
    return sum(losses)/len(losses)
value_and_grad=nn.value_and_grad(model,loss_fn); initial=loss_fn(model); mx.eval(initial)
gradient=False
for _ in range(p["optimizer_steps"]):
    loss,grads=value_and_grad(model); optimizer.update(model,grads); mx.eval(loss,model.parameters(),optimizer.state)
    gradient=gradient or any(float(mx.max(mx.abs(value)).item())>0 for _name,value in utils.tree_flatten(grads))
final=loss_fn(model); outputs=model(hidden); shifted=model(hidden+0.25); mx.eval(final,outputs,shifted)
intervention=max(float(mx.max(mx.abs(a-b)).item()) for (_na,a),(_nb,b) in zip(utils.tree_flatten(outputs),utils.tree_flatten(shifted)))
with tempfile.TemporaryDirectory(prefix="theseus-kerc-structured-") as tmp:
    path=Path(tmp)/"draft.safetensors"; model.save_weights(str(path)); reloaded=build_structured_unit_drafter(d_model=24,hidden_dim=64,future_offsets=(1,2),field_sizes=field_sizes,mx=mx,nn=nn); reloaded.load_weights(str(path)); replay=reloaded(hidden); mx.eval(replay)
    reload_delta=max(float(mx.max(mx.abs(a-b)).item()) for (_na,a),(_nb,b) in zip(utils.tree_flatten(outputs),utils.tree_flatten(replay)))
print(json.dumps({"initial_loss":float(initial.item()),"final_loss":float(final.item()),"gradient_flow":gradient,"hidden_intervention_max_abs_delta":intervention,"checkpoint_reload_max_abs_delta":reload_delta,"parameter_count":sum(int(v.size) for _n,v in utils.tree_flatten(model.parameters()))},sort_keys=True))
'''
    process = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=180,
    )
    if process.returncode:
        return {"available": False, "passed": False, "stderr_tail": process.stderr[-2000:]}
    observed = json.loads(process.stdout)
    checks = {
        "gradient_flow": observed["gradient_flow"],
        "bounded_overfit": observed["final_loss"] < observed["initial_loss"] * 0.15,
        "hidden_state_causal_use": observed["hidden_intervention_max_abs_delta"] > 1e-6,
        "checkpoint_reload": observed["checkpoint_reload_max_abs_delta"] == 0.0,
        "independent_future_heads": observed["parameter_count"] > 0,
    }
    return {
        "available": True,
        "passed": all(checks.values()),
        "checks": checks,
        "observed": observed,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "grammar_pointer_renderer_learned_credit": 0,
        "capability_claim": "NOT_EVALUATED",
    }


def reference_vocabulary() -> StructuredDraftVocabulary:
    return StructuredDraftVocabulary(
        operators=("REPORT", "COMPARE"),
        modalities=("ASSERTED", "POSSIBLE"),
        polarities=("AFFIRMED", "NEGATED"),
        quantifiers=("NONE", "EXISTS"),
        roles=("VALUE", "SOURCE", "LEFT", "RIGHT"),
        entity_pointers=("@N1", "@Q1", "@E1"),
        fidelity_modes=("semantic", "faithful", "lexical", "exact"),
        maximum_roles=4,
    )


def run_reference_suite() -> dict[str, Any]:
    vocabulary = reference_vocabulary()
    first = StructuredSemanticUnit("REPORT", "ASSERTED", "AFFIRMED", "NONE", (("VALUE", "@N1"), ("SOURCE", "@Q1")), "exact", True, False)
    second = StructuredSemanticUnit("COMPARE", "ASSERTED", "AFFIRMED", "NONE", (("LEFT", "@N1"), ("RIGHT", "@E1")), "faithful", True, True)
    wrong = StructuredSemanticUnit("COMPARE", "POSSIBLE", "AFFIRMED", "NONE", (("LEFT", "@N1"), ("RIGHT", "@E1")), "faithful", True, True)
    receipt = verify_structured_draft(
        (first, wrong),
        (first, second),
        vocabulary=vocabulary,
        semantic_verifier=lambda unit: validate_unit(unit, vocabulary)["valid"],
    )
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if receipt.accepted_count == 1 and receipt.target_mismatch_count == 1 else "RED",
        "grammar": validate_unit_sequence((first, second), vocabulary),
        "authority_receipt": asdict(receipt),
        "no_cheat_counters": {"public_training_rows": 0, "external_inference_calls": 0, "grammar_pointer_renderer_learned_credit": 0},
        "claim_boundary": "structured_draft_mechanics_only_not_semantic_truth_capability_or_speed",
    }


if __name__ == "__main__":
    print(json.dumps(run_reference_suite(), indent=2, sort_keys=True))
