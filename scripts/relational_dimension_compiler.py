#!/usr/bin/env python3
"""Typed relational-state ABI for the Theseus RDC/KERC candidate.

This owner separates semantic, primitive-compute, and storage arity. It provides
exact relation reification, branch/lifecycle custody, observable proposal and
operator receipts, reversible query-relative contraction, and columnar lowering.
It does not discover true relations or grant execution authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import host_resource_safety


POLICY = "project_theseus_relational_dimension_compiler_v1"
SCHEMA_VERSION = "RDC-1.0"
LIFECYCLE_STATES = {
    "proposed",
    "qualified",
    "believed",
    "observed",
    "executed",
    "weakened",
    "contradicted",
    "superseded",
    "archived",
    "retired",
}
ACTUALITY_STATES = {"observed", "executed"}
ALLOWED_TRANSITIONS = {
    "proposed": {"qualified", "contradicted", "archived", "retired"},
    "qualified": {"believed", "observed", "executed", "weakened", "contradicted", "archived"},
    "believed": {"observed", "executed", "weakened", "contradicted", "superseded", "archived"},
    "observed": {"weakened", "contradicted", "superseded", "archived"},
    "executed": {"weakened", "contradicted", "superseded", "archived"},
    "weakened": {"qualified", "contradicted", "superseded", "archived"},
    "contradicted": {"superseded", "archived", "retired"},
    "superseded": {"archived", "retired"},
    "archived": {"retired"},
    "retired": set(),
}


class RDCFault(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class DimensionAxis:
    axis_id: str
    carrier: str
    semantics: str
    symmetry: str
    variance_law: str
    legal_operations: tuple[str, ...]
    metric_or_unit: str = "none"


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    entity_type: str
    branch_id: str
    provenance_digest: str
    lifecycle_state: str = "qualified"


@dataclass(frozen=True)
class RoleSpec:
    role_id: str
    participant_type: str
    minimum_count: int = 1
    maximum_count: int = 1
    permutation_class: str = "ordered"


@dataclass(frozen=True)
class RelationSchema:
    schema_id: str
    version: int
    roles: tuple[RoleSpec, ...]
    semantic_arity: int
    symmetry: str
    dimensional_axes: tuple[str, ...]


@dataclass(frozen=True)
class TypedIncidence:
    relation_id: str
    role_id: str
    participant_id: str
    ordinal: int
    branch_id: str


@dataclass(frozen=True)
class RelationInstance:
    relation_id: str
    schema_id: str
    schema_version: int
    branch_id: str
    lifecycle_state: str
    confidence: float
    uncertainty_state: str
    provenance_digests: tuple[str, ...]
    defeater_digests: tuple[str, ...]
    incidences: tuple[TypedIncidence, ...]
    effect_authority: str = "none"


@dataclass(frozen=True)
class BranchRecord:
    branch_id: str
    parent_branch_id: str | None
    branch_kind: str
    actuality: bool
    revision: int


@dataclass(frozen=True)
class ProposalReceipt:
    request_digest: str
    denominator_count: int
    proposed_relation_ids: tuple[str, ...]
    qualified_relation_ids: tuple[str, ...]
    known_relevant_relation_ids: tuple[str, ...]
    proposal_sources: tuple[str, ...]

    @property
    def proposal_recall(self) -> float | None:
        if not self.known_relevant_relation_ids:
            return None
        proposed = set(self.proposed_relation_ids)
        return len(proposed.intersection(self.known_relevant_relation_ids)) / len(
            set(self.known_relevant_relation_ids)
        )


@dataclass(frozen=True)
class OperatorCard:
    operator_id: str
    primitive_arity: int
    supported_schema_ids: tuple[str, ...]
    approximation_class: str
    input_contract: str
    output_contract: str
    symmetry_contract: str
    estimated_compute: float
    estimated_memory: float
    estimated_latency: float
    verifier_cost: float
    failure_modes: tuple[str, ...]


@dataclass(frozen=True)
class OrderDecision:
    request_digest: str
    operator_id: str | None
    primitive_arity: int | None
    total_cost: float | None
    abstained: bool
    rejected_operator_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ContractionCertificate:
    certificate_id: str
    source_relation_ids: tuple[str, ...]
    macro_entity_id: str
    query_family_ids: tuple[str, ...]
    environment_class: str
    discrepancy_tolerance: float
    boundary_entity_ids: tuple[str, ...]
    provenance_digests: tuple[str, ...]
    expansion_triggers: tuple[str, ...]
    source_complex_digest: str


@dataclass(frozen=True)
class SchemaMigration:
    schema_id: str
    from_version: int
    to_version: int
    role_map: tuple[tuple[str, str], ...]
    default_participants: tuple[tuple[str, tuple[str, ...]], ...] = ()
    reversible: bool = True


@dataclass(frozen=True)
class SchemaMigrationReceipt:
    migration_digest: str
    source_relation_digest: str
    target_relation_digest: str
    source_schema_digest: str
    target_schema_digest: str
    exact_rollback_verified: bool


@dataclass(frozen=True)
class SpecialistObservation:
    request_digest: str
    specialist_output_digest: str
    slow_path_output_digest: str
    verified: bool
    latency_saved: float


@dataclass(frozen=True)
class CompiledSpecialist:
    specialist_id: str
    schema_id: str
    schema_version: int
    query_family_ids: tuple[str, ...]
    environment_class: str
    source_relation_ids: tuple[str, ...]
    source_complex_digest: str
    slow_path_operator_id: str
    minimum_revision: int
    expiry_revision: int
    maximum_discrepancy_rate: float
    minimum_verified_observations: int
    retained_slow_path: bool
    qualification_digest: str


def validate_axis(axis: DimensionAxis) -> None:
    if not axis.axis_id or not axis.carrier or not axis.semantics:
        raise RDCFault("axis_identity_incomplete")
    if not axis.legal_operations:
        raise RDCFault("axis_legal_operations_empty")


def validate_schema(schema: RelationSchema, axes: Mapping[str, DimensionAxis]) -> None:
    if schema.version <= 0 or schema.semantic_arity <= 0:
        raise RDCFault("schema_version_or_arity_invalid")
    if len(schema.roles) != schema.semantic_arity:
        raise RDCFault("schema_semantic_arity_mismatch")
    role_ids = [role.role_id for role in schema.roles]
    if len(role_ids) != len(set(role_ids)):
        raise RDCFault("schema_role_identity_duplicate")
    for role in schema.roles:
        if role.minimum_count < 0 or role.maximum_count < max(1, role.minimum_count):
            raise RDCFault("schema_role_cardinality_invalid")
        if role.permutation_class not in {"ordered", "symmetric"}:
            raise RDCFault("schema_role_permutation_invalid")
    if any(axis_id not in axes for axis_id in schema.dimensional_axes):
        raise RDCFault("schema_dimension_axis_unknown")


def reify_relation(
    *,
    relation_id: str,
    schema: RelationSchema,
    participants: Mapping[str, Sequence[str]],
    entities: Mapping[str, EntityRecord],
    branch: BranchRecord,
    provenance_digests: Sequence[str],
    confidence: float,
    uncertainty_state: str,
) -> RelationInstance:
    if not 0.0 <= confidence <= 1.0:
        raise RDCFault("relation_confidence_invalid")
    if uncertainty_state not in {"resolved", "ambiguous", "unknown"}:
        raise RDCFault("relation_uncertainty_invalid")
    extra_roles = set(participants).difference(role.role_id for role in schema.roles)
    if extra_roles:
        raise RDCFault("relation_role_unknown")
    incidences = []
    for role in schema.roles:
        participant_ids = list(participants.get(role.role_id, ()))
        if not role.minimum_count <= len(participant_ids) <= role.maximum_count:
            raise RDCFault("relation_role_cardinality_mismatch")
        if role.permutation_class == "symmetric":
            participant_ids.sort()
        for ordinal, participant_id in enumerate(participant_ids):
            entity = entities.get(participant_id)
            if entity is None:
                raise RDCFault("relation_participant_unknown")
            if entity.entity_type != role.participant_type:
                raise RDCFault("relation_participant_type_mismatch")
            if entity.branch_id != branch.branch_id:
                raise RDCFault("relation_branch_leakage")
            incidences.append(
                TypedIncidence(
                    relation_id=relation_id,
                    role_id=role.role_id,
                    participant_id=participant_id,
                    ordinal=ordinal,
                    branch_id=branch.branch_id,
                )
            )
    return RelationInstance(
        relation_id=relation_id,
        schema_id=schema.schema_id,
        schema_version=schema.version,
        branch_id=branch.branch_id,
        lifecycle_state="proposed",
        confidence=confidence,
        uncertainty_state=uncertainty_state,
        provenance_digests=tuple(sorted(set(provenance_digests))),
        defeater_digests=(),
        incidences=tuple(incidences),
    )


def reconstruct_roles(relation: RelationInstance) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for incidence in relation.incidences:
        if incidence.relation_id != relation.relation_id or incidence.branch_id != relation.branch_id:
            raise RDCFault("incidence_identity_mismatch")
        grouped.setdefault(incidence.role_id, []).append(
            (incidence.ordinal, incidence.participant_id)
        )
    return {
        role_id: tuple(participant for _ordinal, participant in sorted(values))
        for role_id, values in sorted(grouped.items())
    }


def migrate_relation_schema(
    relation: RelationInstance,
    *,
    source_schema: RelationSchema,
    target_schema: RelationSchema,
    migration: SchemaMigration,
    entities: Mapping[str, EntityRecord],
    branch: BranchRecord,
) -> tuple[RelationInstance, SchemaMigrationReceipt]:
    """Migrate one relation through an explicit, content-bound role mapping.

    Migration never edits a relation in place. The returned instance keeps the
    stable relation id and lifecycle/provenance state while receiving a new
    schema version. Reversible migrations must prove an exact role round trip.
    """

    for schema in (source_schema, target_schema):
        if schema.version <= 0 or len(schema.roles) != schema.semantic_arity:
            raise RDCFault("schema_migration_schema_shape_invalid")
        if len({role.role_id for role in schema.roles}) != len(schema.roles):
            raise RDCFault("schema_migration_schema_role_duplicate")
    if (
        source_schema.schema_id != target_schema.schema_id
        or migration.schema_id != source_schema.schema_id
        or relation.schema_id != source_schema.schema_id
    ):
        raise RDCFault("schema_migration_identity_mismatch")
    if (
        relation.schema_version != source_schema.version
        or migration.from_version != source_schema.version
        or migration.to_version != target_schema.version
        or migration.from_version == migration.to_version
    ):
        raise RDCFault("schema_migration_version_mismatch")
    if relation.branch_id != branch.branch_id:
        raise RDCFault("schema_migration_branch_mismatch")

    source_roles = reconstruct_roles(relation)
    role_map = dict(migration.role_map)
    if set(role_map) != {role.role_id for role in source_schema.roles}:
        raise RDCFault("schema_migration_source_role_map_incomplete")
    if len(set(role_map.values())) != len(role_map):
        raise RDCFault("schema_migration_target_role_collision")
    target_role_ids = {role.role_id for role in target_schema.roles}
    if not set(role_map.values()).issubset(target_role_ids):
        raise RDCFault("schema_migration_target_role_unknown")
    defaults = dict(migration.default_participants)
    if not set(defaults).issubset(target_role_ids.difference(role_map.values())):
        raise RDCFault("schema_migration_default_role_invalid")

    participants = {
        target_role: source_roles.get(source_role, ())
        for source_role, target_role in role_map.items()
    }
    participants.update(defaults)
    migrated = reify_relation(
        relation_id=relation.relation_id,
        schema=target_schema,
        participants=participants,
        entities=entities,
        branch=branch,
        provenance_digests=relation.provenance_digests,
        confidence=relation.confidence,
        uncertainty_state=relation.uncertainty_state,
    )
    migrated = replace(
        migrated,
        lifecycle_state=relation.lifecycle_state,
        defeater_digests=relation.defeater_digests,
        effect_authority=relation.effect_authority,
    )

    exact_rollback = False
    if migration.reversible:
        if defaults:
            raise RDCFault("schema_migration_reversible_defaults_forbidden")
        inverse = {target: source for source, target in role_map.items()}
        restored = {
            inverse[target_role]: participants_for_role
            for target_role, participants_for_role in reconstruct_roles(migrated).items()
            if target_role in inverse
        }
        exact_rollback = restored == source_roles
        if not exact_rollback:
            raise RDCFault("schema_migration_rollback_not_exact")

    migration_payload = asdict(migration)
    return migrated, SchemaMigrationReceipt(
        migration_digest=digest(migration_payload),
        source_relation_digest=digest(asdict(relation)),
        target_relation_digest=digest(asdict(migrated)),
        source_schema_digest=digest(asdict(source_schema)),
        target_schema_digest=digest(asdict(target_schema)),
        exact_rollback_verified=exact_rollback,
    )


def rollback_relation_schema(
    relation: RelationInstance,
    *,
    source_schema: RelationSchema,
    target_schema: RelationSchema,
    migration: SchemaMigration,
    receipt: SchemaMigrationReceipt,
    entities: Mapping[str, EntityRecord],
    branch: BranchRecord,
) -> RelationInstance:
    """Reverse a previously verified migration and reject mutated state."""

    if not migration.reversible or not receipt.exact_rollback_verified:
        raise RDCFault("schema_migration_not_reversible")
    if (
        receipt.migration_digest != digest(asdict(migration))
        or receipt.target_relation_digest != digest(asdict(relation))
        or receipt.source_schema_digest != digest(asdict(source_schema))
        or receipt.target_schema_digest != digest(asdict(target_schema))
    ):
        raise RDCFault("schema_migration_receipt_mismatch")
    reverse = SchemaMigration(
        schema_id=migration.schema_id,
        from_version=migration.to_version,
        to_version=migration.from_version,
        role_map=tuple((target, source) for source, target in migration.role_map),
        reversible=True,
    )
    restored, _reverse_receipt = migrate_relation_schema(
        relation,
        source_schema=target_schema,
        target_schema=source_schema,
        migration=reverse,
        entities=entities,
        branch=branch,
    )
    if digest(asdict(restored)) != receipt.source_relation_digest:
        raise RDCFault("schema_migration_source_not_restored")
    return restored


def transition_relation(
    relation: RelationInstance,
    *,
    next_state: str,
    branch: BranchRecord,
    event_kind: str,
    evidence_digest: str,
) -> tuple[RelationInstance, dict[str, Any]]:
    if next_state not in LIFECYCLE_STATES:
        raise RDCFault("relation_lifecycle_state_unknown")
    if relation.branch_id != branch.branch_id:
        raise RDCFault("relation_lifecycle_branch_mismatch")
    if next_state not in ALLOWED_TRANSITIONS[relation.lifecycle_state]:
        raise RDCFault("relation_lifecycle_transition_invalid")
    if next_state in ACTUALITY_STATES and not branch.actuality:
        raise RDCFault("hypothetical_branch_actualization_forbidden")
    if next_state in ACTUALITY_STATES and event_kind not in {"observation", "execution", "reconciliation"}:
        raise RDCFault("actuality_event_authority_missing")
    updated = replace(relation, lifecycle_state=next_state)
    return updated, {
        "relation_id": relation.relation_id,
        "branch_id": branch.branch_id,
        "from_state": relation.lifecycle_state,
        "to_state": next_state,
        "event_kind": event_kind,
        "evidence_digest": evidence_digest,
        "effect_authority_unchanged": updated.effect_authority == "none",
        "event_digest": digest(
            [relation.relation_id, branch.branch_id, relation.lifecycle_state, next_state, event_kind, evidence_digest]
        ),
    }


def proposal_receipt(
    *,
    request: Mapping[str, Any],
    denominator_ids: Sequence[str],
    proposed_ids: Sequence[str],
    qualified_ids: Sequence[str],
    known_relevant_ids: Sequence[str] = (),
    proposal_sources: Sequence[str],
) -> ProposalReceipt:
    denominator = tuple(dict.fromkeys(denominator_ids))
    proposed = tuple(dict.fromkeys(proposed_ids))
    qualified = tuple(dict.fromkeys(qualified_ids))
    if not set(proposed).issubset(denominator):
        raise RDCFault("proposal_outside_denominator")
    if not set(qualified).issubset(proposed):
        raise RDCFault("qualification_outside_proposals")
    return ProposalReceipt(
        request_digest=digest(request),
        denominator_count=len(denominator),
        proposed_relation_ids=proposed,
        qualified_relation_ids=qualified,
        known_relevant_relation_ids=tuple(dict.fromkeys(known_relevant_ids)),
        proposal_sources=tuple(dict.fromkeys(proposal_sources)),
    )


def select_operator(
    *,
    request: Mapping[str, Any],
    schema_id: str,
    minimum_primitive_arity: int,
    operators: Sequence[OperatorCard],
    maximum_total_cost: float,
    weights: Mapping[str, float],
) -> OrderDecision:
    candidates = []
    rejected = []
    for operator in operators:
        if schema_id not in operator.supported_schema_ids or operator.primitive_arity < minimum_primitive_arity:
            rejected.append(operator.operator_id)
            continue
        cost = (
            operator.estimated_compute * float(weights.get("compute", 1.0))
            + operator.estimated_memory * float(weights.get("memory", 1.0))
            + operator.estimated_latency * float(weights.get("latency", 1.0))
            + operator.verifier_cost * float(weights.get("verifier", 1.0))
        )
        if cost <= maximum_total_cost:
            candidates.append((cost, operator.primitive_arity, operator.operator_id))
        else:
            rejected.append(operator.operator_id)
    if not candidates:
        return OrderDecision(
            request_digest=digest(request),
            operator_id=None,
            primitive_arity=None,
            total_cost=None,
            abstained=True,
            rejected_operator_ids=tuple(sorted(set(rejected))),
            reason="no_qualified_operator_within_total_cost",
        )
    cost, arity, operator_id = min(candidates)
    return OrderDecision(
        request_digest=digest(request),
        operator_id=operator_id,
        primitive_arity=arity,
        total_cost=cost,
        abstained=False,
        rejected_operator_ids=tuple(sorted(set(rejected))),
        reason="least_total_cost_qualified_operator",
    )


def contract_relations(
    *,
    relations: Sequence[RelationInstance],
    macro_entity_id: str,
    query_family_ids: Sequence[str],
    environment_class: str,
    discrepancy_tolerance: float,
    boundary_entity_ids: Sequence[str],
    expansion_triggers: Sequence[str],
) -> ContractionCertificate:
    if not relations or not query_family_ids or not expansion_triggers:
        raise RDCFault("contraction_contract_incomplete")
    if not 0.0 <= discrepancy_tolerance <= 1.0:
        raise RDCFault("contraction_tolerance_invalid")
    branches = {relation.branch_id for relation in relations}
    if len(branches) != 1:
        raise RDCFault("contraction_cross_branch_forbidden")
    if any(relation.lifecycle_state not in {"qualified", "believed", "observed", "executed"} for relation in relations):
        raise RDCFault("contraction_relation_unqualified")
    source_payload = [asdict(relation) for relation in sorted(relations, key=lambda item: item.relation_id)]
    source_digest = digest(source_payload)
    values = {
        "source_relation_ids": tuple(sorted(relation.relation_id for relation in relations)),
        "macro_entity_id": macro_entity_id,
        "query_family_ids": tuple(sorted(set(query_family_ids))),
        "environment_class": environment_class,
        "discrepancy_tolerance": float(discrepancy_tolerance),
        "boundary_entity_ids": tuple(sorted(set(boundary_entity_ids))),
        "provenance_digests": tuple(sorted({value for relation in relations for value in relation.provenance_digests})),
        "expansion_triggers": tuple(sorted(set(expansion_triggers))),
        "source_complex_digest": source_digest,
    }
    return ContractionCertificate(certificate_id=digest(values), **values)


def resolve_contraction(
    certificate: ContractionCertificate,
    *,
    query_family_id: str,
    environment_class: str,
    observed_discrepancy: float,
    active_triggers: Sequence[str] = (),
) -> dict[str, Any]:
    reasons = []
    if query_family_id not in certificate.query_family_ids:
        reasons.append("query_outside_certificate")
    if environment_class != certificate.environment_class:
        reasons.append("environment_shift")
    if observed_discrepancy > certificate.discrepancy_tolerance:
        reasons.append("discrepancy_exceeded")
    reasons.extend(sorted(set(active_triggers).intersection(certificate.expansion_triggers)))
    return {
        "certificate_id": certificate.certificate_id,
        "use_macro_object": not reasons,
        "expand_source_complex": bool(reasons),
        "reasons": reasons,
        "source_relation_ids": list(certificate.source_relation_ids) if reasons else [],
    }


def compile_monitored_specialist(
    *,
    certificate: ContractionCertificate,
    schema_id: str,
    schema_version: int,
    slow_path_operator_id: str,
    observations: Sequence[SpecialistObservation],
    current_revision: int,
    lifetime_revisions: int,
    maximum_discrepancy_rate: float,
    minimum_verified_observations: int,
) -> CompiledSpecialist:
    """Compile a K4 specialist only from paired slow-path observations.

    Every qualifying observation carries both specialist and retained slow-path
    output identities. A specialist can amortize work, but cannot erase the
    reference path that qualifies and monitors it.
    """

    if schema_version <= 0 or current_revision < 0 or lifetime_revisions <= 0:
        raise RDCFault("specialist_revision_contract_invalid")
    if not 0.0 <= maximum_discrepancy_rate <= 1.0:
        raise RDCFault("specialist_discrepancy_threshold_invalid")
    if minimum_verified_observations < 1:
        raise RDCFault("specialist_observation_floor_invalid")
    if len({row.request_digest for row in observations}) != len(observations):
        raise RDCFault("specialist_observation_request_duplicate")
    verified = [row for row in observations if row.verified]
    if len(verified) < minimum_verified_observations:
        raise RDCFault("specialist_verified_observations_insufficient")
    mismatches = sum(
        row.specialist_output_digest != row.slow_path_output_digest for row in verified
    )
    discrepancy_rate = mismatches / len(verified)
    if discrepancy_rate > maximum_discrepancy_rate:
        raise RDCFault("specialist_discrepancy_exceeded")
    if sum(row.latency_saved for row in verified) <= 0.0:
        raise RDCFault("specialist_no_observed_amortization")

    qualification = {
        "certificate_id": certificate.certificate_id,
        "schema_id": schema_id,
        "schema_version": schema_version,
        "slow_path_operator_id": slow_path_operator_id,
        "observations": [asdict(row) for row in observations],
        "current_revision": current_revision,
        "lifetime_revisions": lifetime_revisions,
        "maximum_discrepancy_rate": maximum_discrepancy_rate,
        "minimum_verified_observations": minimum_verified_observations,
    }
    qualification_digest = digest(qualification)
    return CompiledSpecialist(
        specialist_id=f"rdc-specialist-{qualification_digest[:16]}",
        schema_id=schema_id,
        schema_version=schema_version,
        query_family_ids=certificate.query_family_ids,
        environment_class=certificate.environment_class,
        source_relation_ids=certificate.source_relation_ids,
        source_complex_digest=certificate.source_complex_digest,
        slow_path_operator_id=slow_path_operator_id,
        minimum_revision=current_revision,
        expiry_revision=current_revision + lifetime_revisions,
        maximum_discrepancy_rate=maximum_discrepancy_rate,
        minimum_verified_observations=minimum_verified_observations,
        retained_slow_path=True,
        qualification_digest=qualification_digest,
    )


def route_compiled_specialist(
    specialist: CompiledSpecialist,
    *,
    query_family_id: str,
    environment_class: str,
    schema_id: str,
    schema_version: int,
    current_revision: int,
    source_complex_digest: str,
    monitoring_observations: Sequence[SpecialistObservation] = (),
) -> dict[str, Any]:
    """Route through a specialist or explicitly fall back to its slow path."""

    reasons = []
    if query_family_id not in specialist.query_family_ids:
        reasons.append("query_outside_specialist")
    if environment_class != specialist.environment_class:
        reasons.append("environment_shift")
    if schema_id != specialist.schema_id or schema_version != specialist.schema_version:
        reasons.append("schema_shift")
    if not specialist.minimum_revision <= current_revision <= specialist.expiry_revision:
        reasons.append("specialist_expired")
    if source_complex_digest != specialist.source_complex_digest:
        reasons.append("source_complex_changed")
    verified = [row for row in monitoring_observations if row.verified]
    if verified:
        discrepancy_rate = sum(
            row.specialist_output_digest != row.slow_path_output_digest for row in verified
        ) / len(verified)
        if discrepancy_rate > specialist.maximum_discrepancy_rate:
            reasons.append("monitoring_discrepancy_exceeded")
    else:
        discrepancy_rate = None
    use_specialist = not reasons
    return {
        "specialist_id": specialist.specialist_id,
        "use_specialist": use_specialist,
        "operator_id": specialist.specialist_id if use_specialist else specialist.slow_path_operator_id,
        "retained_slow_path": specialist.retained_slow_path,
        "rollback_reasons": reasons,
        "monitoring_discrepancy_rate": discrepancy_rate,
        "route_receipt_digest": digest(
            {
                "specialist": asdict(specialist),
                "query_family_id": query_family_id,
                "environment_class": environment_class,
                "schema_id": schema_id,
                "schema_version": schema_version,
                "current_revision": current_revision,
                "source_complex_digest": source_complex_digest,
                "monitoring": [asdict(row) for row in monitoring_observations],
                "reasons": reasons,
            }
        ),
    }


def lower_structure_of_arrays(
    entities: Sequence[EntityRecord], relations: Sequence[RelationInstance]
) -> dict[str, Any]:
    entity_rows = sorted(entities, key=lambda item: item.entity_id)
    relation_rows = sorted(relations, key=lambda item: item.relation_id)
    incidence_rows = sorted(
        (incidence for relation in relation_rows for incidence in relation.incidences),
        key=lambda item: (item.relation_id, item.role_id, item.ordinal),
    )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "entities": {
            "entity_id": [row.entity_id for row in entity_rows],
            "entity_type": [row.entity_type for row in entity_rows],
            "branch_id": [row.branch_id for row in entity_rows],
            "lifecycle_state": [row.lifecycle_state for row in entity_rows],
            "provenance_digest": [row.provenance_digest for row in entity_rows],
        },
        "relations": {
            "relation_id": [row.relation_id for row in relation_rows],
            "schema_id": [row.schema_id for row in relation_rows],
            "schema_version": [row.schema_version for row in relation_rows],
            "branch_id": [row.branch_id for row in relation_rows],
            "lifecycle_state": [row.lifecycle_state for row in relation_rows],
            "confidence": [row.confidence for row in relation_rows],
            "uncertainty_state": [row.uncertainty_state for row in relation_rows],
        },
        "incidences": {
            "relation_id": [row.relation_id for row in incidence_rows],
            "role_id": [row.role_id for row in incidence_rows],
            "participant_id": [row.participant_id for row in incidence_rows],
            "ordinal": [row.ordinal for row in incidence_rows],
            "branch_id": [row.branch_id for row in incidence_rows],
        },
    }
    packet["packet_digest"] = digest(packet)
    return packet


def validate_lowering(packet: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(packet)
    expected_digest = record.pop("packet_digest", None)
    if record.get("schema_version") != SCHEMA_VERSION or digest(record) != expected_digest:
        raise RDCFault("lowering_digest_mismatch")
    for table_name in ("entities", "relations", "incidences"):
        columns = record.get(table_name)
        if not isinstance(columns, dict) or not columns:
            raise RDCFault("lowering_table_missing")
        lengths = {len(values) for values in columns.values()}
        if len(lengths) != 1:
            raise RDCFault("lowering_column_length_mismatch")
    entity_ids = set(record["entities"]["entity_id"])
    relation_ids = set(record["relations"]["relation_id"])
    if not set(record["incidences"]["participant_id"]).issubset(entity_ids):
        raise RDCFault("lowering_incidence_entity_unknown")
    if not set(record["incidences"]["relation_id"]).issubset(relation_ids):
        raise RDCFault("lowering_incidence_relation_unknown")
    return {
        "valid": True,
        "entity_count": len(entity_ids),
        "relation_count": len(relation_ids),
        "incidence_count": len(record["incidences"]["relation_id"]),
        "packet_digest": expected_digest,
    }


def lower_sparse_relation_batch(
    packet: Mapping[str, Any],
    *,
    entity_type_vocabulary: Sequence[str],
    schema_vocabulary: Sequence[str],
    role_vocabulary: Sequence[str],
) -> dict[str, Any]:
    """Lower the content-bound SoA packet to factorized integer tensors.

    The representation is sparse in incidences and never materializes a dense
    entity-by-relation-by-role tensor. Vocabularies are explicit checkpoint ABI.
    """

    validate_lowering(packet)
    entities = packet["entities"]
    relations = packet["relations"]
    incidences = packet["incidences"]
    entity_type_ids = {value: index for index, value in enumerate(entity_type_vocabulary)}
    schema_ids = {value: index for index, value in enumerate(schema_vocabulary)}
    role_ids = {value: index for index, value in enumerate(role_vocabulary)}
    if len(entity_type_ids) != len(entity_type_vocabulary):
        raise RDCFault("sparse_lowering_entity_vocabulary_duplicate")
    if len(schema_ids) != len(schema_vocabulary):
        raise RDCFault("sparse_lowering_schema_vocabulary_duplicate")
    if len(role_ids) != len(role_vocabulary):
        raise RDCFault("sparse_lowering_role_vocabulary_duplicate")

    entity_index = {value: index for index, value in enumerate(entities["entity_id"])}
    relation_index = {value: index for index, value in enumerate(relations["relation_id"])}
    try:
        lowered = {
            "schema_version": SCHEMA_VERSION,
            "source_packet_digest": packet["packet_digest"],
            "vocabularies": {
                "entity_type": list(entity_type_vocabulary),
                "schema": list(schema_vocabulary),
                "role": list(role_vocabulary),
            },
            "entity_type_id": [entity_type_ids[value] for value in entities["entity_type"]],
            "relation_schema_id": [schema_ids[value] for value in relations["schema_id"]],
            "incidence_relation_index": [relation_index[value] for value in incidences["relation_id"]],
            "incidence_role_id": [role_ids[value] for value in incidences["role_id"]],
            "incidence_participant_index": [entity_index[value] for value in incidences["participant_id"]],
            "incidence_ordinal": list(incidences["ordinal"]),
        }
    except KeyError as exc:
        raise RDCFault(f"sparse_lowering_vocabulary_or_identity_missing:{exc.args[0]}") from exc
    relation_count = len(relations["relation_id"])
    counts = [0] * relation_count
    for index in lowered["incidence_relation_index"]:
        counts[index] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    lowered["relation_incidence_offsets"] = offsets
    lowered["tensor_digest"] = digest(lowered)
    return lowered


def validate_sparse_relation_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(batch)
    expected = record.pop("tensor_digest", None)
    if record.get("schema_version") != SCHEMA_VERSION or digest(record) != expected:
        raise RDCFault("sparse_lowering_digest_mismatch")
    incidence_fields = (
        "incidence_relation_index",
        "incidence_role_id",
        "incidence_participant_index",
        "incidence_ordinal",
    )
    lengths = {len(record[field]) for field in incidence_fields}
    if len(lengths) != 1:
        raise RDCFault("sparse_lowering_incidence_length_mismatch")
    relation_count = len(record["relation_schema_id"])
    entity_count = len(record["entity_type_id"])
    offsets = record["relation_incidence_offsets"]
    if len(offsets) != relation_count + 1 or offsets[0] != 0 or offsets[-1] != next(iter(lengths), 0):
        raise RDCFault("sparse_lowering_offsets_invalid")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise RDCFault("sparse_lowering_offsets_not_monotonic")
    if any(not 0 <= value < relation_count for value in record["incidence_relation_index"]):
        raise RDCFault("sparse_lowering_relation_index_invalid")
    if any(not 0 <= value < entity_count for value in record["incidence_participant_index"]):
        raise RDCFault("sparse_lowering_participant_index_invalid")
    return {
        "valid": True,
        "entity_count": entity_count,
        "relation_count": relation_count,
        "incidence_count": next(iter(lengths), 0),
        "tensor_digest": expected,
    }


def factorized_relation_pool_numpy(
    batch: Mapping[str, Any],
    *,
    entity_features: Any,
    role_factors: Any,
    schema_factors: Any,
) -> Any:
    """Reference O(incidences * hidden) relational contraction."""

    import numpy as np

    validate_sparse_relation_batch(batch)
    entities = np.asarray(entity_features, dtype=np.float32)
    roles = np.asarray(role_factors, dtype=np.float32)
    schemas = np.asarray(schema_factors, dtype=np.float32)
    if entities.ndim != 2 or roles.ndim != 2 or schemas.ndim != 2:
        raise RDCFault("factorized_relation_rank_invalid")
    hidden = entities.shape[1]
    if roles.shape[1] != hidden or schemas.shape[1] != hidden:
        raise RDCFault("factorized_relation_hidden_mismatch")
    if len(entities) != len(batch["entity_type_id"]):
        raise RDCFault("factorized_relation_entity_count_mismatch")
    if len(roles) != len(batch["vocabularies"]["role"]):
        raise RDCFault("factorized_relation_role_count_mismatch")
    if len(schemas) != len(batch["vocabularies"]["schema"]):
        raise RDCFault("factorized_relation_schema_count_mismatch")
    outputs = np.zeros((len(batch["relation_schema_id"]), hidden), dtype=np.float32)
    for relation_index, role_id, participant_index in zip(
        batch["incidence_relation_index"],
        batch["incidence_role_id"],
        batch["incidence_participant_index"],
    ):
        outputs[relation_index] += entities[participant_index] * roles[role_id]
    counts = np.diff(np.asarray(batch["relation_incidence_offsets"], dtype=np.int32))
    outputs /= np.maximum(counts[:, None], 1)
    return outputs * schemas[np.asarray(batch["relation_schema_id"], dtype=np.int32)]


def factorized_relation_pool_mlx(
    batch: Mapping[str, Any],
    *,
    entity_features: Any,
    role_factors: Any,
    schema_factors: Any,
) -> Any:
    """MLX segment contraction with the same sparse semantics as the reference."""

    if not host_resource_safety.accelerator_child_authorized():
        raise RDCFault("accelerator_watchdog_required")

    import mlx.core as mx

    validate_sparse_relation_batch(batch)
    entities = mx.array(entity_features, dtype=mx.float32)
    roles = mx.array(role_factors, dtype=mx.float32)
    schemas = mx.array(schema_factors, dtype=mx.float32)
    if entities.ndim != 2 or roles.ndim != 2 or schemas.ndim != 2:
        raise RDCFault("factorized_relation_rank_invalid")
    hidden = entities.shape[1]
    if roles.shape[1] != hidden or schemas.shape[1] != hidden:
        raise RDCFault("factorized_relation_hidden_mismatch")
    relation_ids = mx.array(batch["incidence_relation_index"], dtype=mx.int32)
    role_ids = mx.array(batch["incidence_role_id"], dtype=mx.int32)
    participant_ids = mx.array(batch["incidence_participant_index"], dtype=mx.int32)
    incidence_values = entities[participant_ids] * roles[role_ids]
    relation_count = len(batch["relation_schema_id"])
    pooled = mx.zeros((relation_count, hidden), dtype=mx.float32)
    pooled = pooled.at[relation_ids].add(incidence_values)
    counts = mx.array(
        [right - left for left, right in zip(batch["relation_incidence_offsets"], batch["relation_incidence_offsets"][1:])],
        dtype=mx.float32,
    )
    pooled = pooled / mx.maximum(counts[:, None], mx.array(1.0, dtype=mx.float32))
    schema_ids = mx.array(batch["relation_schema_id"], dtype=mx.int32)
    output = pooled * schemas[schema_ids]
    mx.eval(output)
    return output


def reference_fixture() -> dict[str, Any]:
    axes = {
        axis.axis_id: axis
        for axis in (
            DimensionAxis("semantic_role", "typed_role", "participant function", "schema_declared", "identity", ("bind", "permute_if_symmetric")),
            DimensionAxis("branch", "branch_dag", "hypothesis or actuality", "none", "explicit_reconciliation", ("fork", "compare", "reconcile")),
            DimensionAxis("epistemic", "lifecycle_state", "support and uncertainty", "none", "evidence_event", ("qualify", "weaken", "contradict")),
        )
    }
    for axis in axes.values():
        validate_axis(axis)
    schema = RelationSchema(
        schema_id="transfer",
        version=1,
        roles=(
            RoleSpec("sender", "agent"),
            RoleSpec("object", "artifact"),
            RoleSpec("recipient", "agent"),
            RoleSpec("time", "time"),
        ),
        semantic_arity=4,
        symmetry="ordered",
        dimensional_axes=tuple(axes),
    )
    validate_schema(schema, axes)
    branch = BranchRecord("actual", None, "actual", True, 1)
    entities = {
        row.entity_id: row
        for row in (
            EntityRecord("alice", "agent", "actual", "p:a"),
            EntityRecord("report", "artifact", "actual", "p:r"),
            EntityRecord("bob", "agent", "actual", "p:b"),
            EntityRecord("noon", "time", "actual", "p:t"),
        )
    }
    relation = reify_relation(
        relation_id="transfer:1",
        schema=schema,
        participants={"sender": ["alice"], "object": ["report"], "recipient": ["bob"], "time": ["noon"]},
        entities=entities,
        branch=branch,
        provenance_digests=("source:1",),
        confidence=0.8,
        uncertainty_state="resolved",
    )
    qualified, event = transition_relation(
        relation,
        next_state="qualified",
        branch=branch,
        event_kind="qualification",
        evidence_digest="evidence:1",
    )
    lowering = lower_structure_of_arrays(tuple(entities.values()), (qualified,))
    return {
        "axes": axes,
        "schema": schema,
        "branch": branch,
        "entities": entities,
        "relation": qualified,
        "event": event,
        "lowering": lowering,
    }


def run_reference_suite() -> dict[str, Any]:
    fixture = reference_fixture()
    relation = fixture["relation"]
    source_schema = fixture["schema"]
    target_schema = RelationSchema(
        schema_id="transfer",
        version=2,
        roles=(
            RoleSpec("agent_from", "agent"),
            RoleSpec("payload", "artifact"),
            RoleSpec("agent_to", "agent"),
            RoleSpec("at_time", "time"),
        ),
        semantic_arity=4,
        symmetry="ordered",
        dimensional_axes=source_schema.dimensional_axes,
    )
    migration = SchemaMigration(
        schema_id="transfer",
        from_version=1,
        to_version=2,
        role_map=(("sender", "agent_from"), ("object", "payload"), ("recipient", "agent_to"), ("time", "at_time")),
    )
    migrated, migration_receipt = migrate_relation_schema(
        relation,
        source_schema=source_schema,
        target_schema=target_schema,
        migration=migration,
        entities=fixture["entities"],
        branch=fixture["branch"],
    )
    restored = rollback_relation_schema(
        migrated,
        source_schema=source_schema,
        target_schema=target_schema,
        migration=migration,
        receipt=migration_receipt,
        entities=fixture["entities"],
        branch=fixture["branch"],
    )
    proposals = proposal_receipt(
        request={"consumer": "planner", "schema": "transfer"},
        denominator_ids=("transfer:1", "transfer:2", "transfer:3"),
        proposed_ids=("transfer:1", "transfer:2"),
        qualified_ids=("transfer:1",),
        known_relevant_ids=("transfer:1",),
        proposal_sources=("schema_completion", "planning_demand"),
    )
    operators = (
        OperatorCard("pairwise_join", 2, ("transfer",), "exact_reified", "incidences", "relation", "role_preserving", 2, 1, 2, 1, ("surrogate_miss",)),
        OperatorCard("triadic_kernel", 3, ("transfer",), "learned_approximation", "entity_features", "relation_score", "ordered", 7, 4, 3, 2, ("calibration",)),
    )
    decision = select_operator(
        request={"relation": relation.relation_id},
        schema_id="transfer",
        minimum_primitive_arity=2,
        operators=operators,
        maximum_total_cost=10,
        weights={"compute": 1, "memory": 1, "latency": 1, "verifier": 1},
    )
    certificate = contract_relations(
        relations=(relation,),
        macro_entity_id="macro:transfer",
        query_family_ids=("who_received",),
        environment_class="fixture-v1",
        discrepancy_tolerance=0.05,
        boundary_entity_ids=("alice", "bob"),
        expansion_triggers=("role_changed", "evidence_revoked"),
    )
    expand = resolve_contraction(
        certificate,
        query_family_id="different_query",
        environment_class="fixture-v1",
        observed_discrepancy=0.0,
    )
    observations = tuple(
        SpecialistObservation(f"request:{index}", f"answer:{index}", f"answer:{index}", True, 1.0)
        for index in range(4)
    )
    specialist = compile_monitored_specialist(
        certificate=certificate,
        schema_id="transfer",
        schema_version=1,
        slow_path_operator_id="pairwise_join",
        observations=observations,
        current_revision=1,
        lifetime_revisions=4,
        maximum_discrepancy_rate=0.0,
        minimum_verified_observations=4,
    )
    specialist_route = route_compiled_specialist(
        specialist,
        query_family_id="who_received",
        environment_class="fixture-v1",
        schema_id="transfer",
        schema_version=1,
        current_revision=2,
        source_complex_digest=certificate.source_complex_digest,
    )
    sparse_batch = lower_sparse_relation_batch(
        fixture["lowering"],
        entity_type_vocabulary=("agent", "artifact", "time"),
        schema_vocabulary=("transfer",),
        role_vocabulary=("object", "recipient", "sender", "time"),
    )
    gates = {
        "semantic_arity_reified": len(relation.incidences) == source_schema.semantic_arity,
        "role_roundtrip": reconstruct_roles(relation)
        == {"object": ("report",), "recipient": ("bob",), "sender": ("alice",), "time": ("noon",)},
        "proposal_denominator_observable": proposals.denominator_count == 3,
        "proposal_recall_observable": proposals.proposal_recall == 1.0,
        "least_cost_operator_selected": decision.operator_id == "pairwise_join",
        "query_relative_expansion": expand["expand_source_complex"],
        "columnar_lowering_valid": validate_lowering(fixture["lowering"])["valid"],
        "schema_migration_exact_rollback": restored == relation and migration_receipt.exact_rollback_verified,
        "slow_path_retained_by_specialist": specialist.retained_slow_path and specialist_route["use_specialist"],
        "sparse_factorized_lowering_valid": validate_sparse_relation_batch(sparse_batch)["valid"],
        "effect_authority_absent": relation.effect_authority == "none",
    }
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "gates": gates,
        "proposal_receipt": asdict(proposals) | {"proposal_recall": proposals.proposal_recall},
        "order_decision": asdict(decision),
        "contraction_certificate": asdict(certificate),
        "expansion_receipt": expand,
        "schema_migration_receipt": asdict(migration_receipt),
        "compiled_specialist": asdict(specialist),
        "specialist_route": specialist_route,
        "lowering_receipt": validate_lowering(fixture["lowering"]),
        "sparse_lowering_receipt": validate_sparse_relation_batch(sparse_batch),
        "no_cheat_counters": {"public_training_rows": 0, "external_inference_calls": 0, "deterministic_generation_credit": 0},
        "claim_boundary": "representation_and_governance_mechanics_only_not_relation_truth_learning_efficiency_or_kerc_advantage",
    }


if __name__ == "__main__":
    print(json.dumps(run_reference_suite(), indent=2, sort_keys=True))
