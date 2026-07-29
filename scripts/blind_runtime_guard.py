"""Runtime capability boundary for blind candidate generation and ranking.

Static source inspection remains useful lint. This module is the executable
boundary: consumers receive an allowlisted view and forbidden fields cannot be
read through mapping access, attributes, helpers, aliases, cached envelopes, or
ranking inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterator


FORBIDDEN_FIELDS = frozenset(
    {
        "action_id",
        "answer",
        "answers",
        "benchmark_card",
        "canonical_solution",
        "category",
        "expected",
        "expected_output",
        "family",
        "hidden_tests",
        "required_constructs",
        "return_shape",
        "solution",
        "solution_body",
        "solution_expr",
        "source_task_id",
        "tests",
        "type_family",
    }
)


class BlindRuntimeFault(PermissionError):
    """A runtime consumer attempted to cross the blind-flow boundary."""


@dataclass(frozen=True)
class BlindCapability:
    allowed_fields: frozenset[str]
    purpose: str


class BlindView(Mapping[str, Any]):
    """An immutable allowlisted projection over a private raw record."""

    __slots__ = ("__raw", "__capability")

    def __init__(
        self,
        raw: Mapping[str, Any],
        capability: BlindCapability,
    ) -> None:
        if FORBIDDEN_FIELDS.intersection(capability.allowed_fields):
            raise BlindRuntimeFault("capability_contains_forbidden_fields")
        object.__setattr__(self, "_BlindView__raw", raw)
        object.__setattr__(self, "_BlindView__capability", capability)

    def __getitem__(self, key: str) -> Any:
        name = str(key)
        if name in FORBIDDEN_FIELDS:
            raise BlindRuntimeFault(f"forbidden_field_access:{name}")
        if name not in self.__capability.allowed_fields:
            raise BlindRuntimeFault(f"field_outside_capability:{name}")
        return inert_copy(self.__raw[name], path=name)

    def __iter__(self) -> Iterator[str]:
        return iter(
            sorted(key for key in self.__capability.allowed_fields if key in self.__raw)
        )

    def __len__(self) -> int:
        return sum(key in self.__raw for key in self.__capability.allowed_fields)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def get(self, key: str, default: Any = None) -> Any:
        name = str(key)
        if name in self.__raw:
            return self[name]
        if name in FORBIDDEN_FIELDS:
            raise BlindRuntimeFault(f"forbidden_field_access:{name}")
        if name not in self.__capability.allowed_fields:
            raise BlindRuntimeFault(f"field_outside_capability:{name}")
        return inert_copy(default, path=f"default.{name}")


def issue_blind_capability(
    *,
    allowed_fields: set[str] | frozenset[str],
    purpose: str,
) -> BlindCapability:
    fields = frozenset(str(field) for field in allowed_fields)
    if not fields or not purpose.strip():
        raise BlindRuntimeFault("nonempty_capability_and_purpose_required")
    if FORBIDDEN_FIELDS.intersection(fields):
        raise BlindRuntimeFault("capability_contains_forbidden_fields")
    return BlindCapability(fields, purpose.strip())


def blind_view(
    raw: Mapping[str, Any],
    capability: BlindCapability,
) -> BlindView:
    if not isinstance(raw, Mapping):
        raise BlindRuntimeFault("blind_view_requires_mapping")
    return BlindView(raw, capability)


def validate_cached_artifact(
    value: Any,
    capability: BlindCapability,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BlindRuntimeFault("cached_artifact_requires_mapping")
    declared = value.get("visible_fields")
    if not isinstance(declared, list):
        raise BlindRuntimeFault("cached_artifact_visible_fields_required")
    declared_fields = frozenset(str(field) for field in declared)
    if declared_fields != capability.allowed_fields:
        raise BlindRuntimeFault("cached_artifact_capability_mismatch")
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise BlindRuntimeFault("cached_artifact_payload_required")
    forbidden = sorted(FORBIDDEN_FIELDS.intersection(flat_keys(payload)))
    if forbidden:
        raise BlindRuntimeFault(
            f"cached_artifact_forbidden_fields:{','.join(forbidden)}"
        )
    unknown = sorted(set(payload) - capability.allowed_fields)
    if unknown:
        raise BlindRuntimeFault(
            f"cached_artifact_fields_outside_capability:{','.join(unknown)}"
        )
    return {
        str(key): inert_copy(payload[key], path=f"payload.{key}")
        for key in sorted(payload)
    }


def guard_ranking_inputs(
    rows: list[Mapping[str, Any]],
    capability: BlindCapability,
) -> list[BlindView]:
    return [blind_view(row, capability) for row in rows]


def inert_copy(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        forbidden = sorted(FORBIDDEN_FIELDS.intersection(flat_keys(value)))
        if forbidden:
            raise BlindRuntimeFault(
                f"nested_forbidden_fields:{path}:{','.join(forbidden)}"
            )
        return {
            str(key): inert_copy(child, path=f"{path}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            inert_copy(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            inert_copy(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    return value


def flat_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(flat_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.update(flat_keys(child))
    return keys
