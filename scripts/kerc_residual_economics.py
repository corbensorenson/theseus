#!/usr/bin/env python3
"""Exact residual coding and rate-distortion economics for KERC.

The codec is a deterministic order-1 adaptive arithmetic coder. Its initial
finite-state counts are seeded from the Kernel and higher-level residual state,
so each level is coded under an explicit conditional model instead of being
estimated from JSON length. The implementation is deliberately independent of
the learned allocator so codec accounting can audit learned decisions.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Iterable


CODEC_POLICY = "project_theseus_kerc_conditional_residual_codec_v1"
ALLOCATION_POLICY = "project_theseus_kerc_rate_distortion_allocator_v1"
PROMOTION_POLICY = "project_theseus_kerc_residual_promotion_economics_v1"
FIDELITY_ORDER = ("semantic", "faithful", "lexical", "exact")
STATE_BITS = 32
MAX_TOTAL = 1 << 15

# Stable field identifiers keep the residual wire representation compact without
# coupling the readable/auditable packet ABI to an external serialization package.
# Unknown keys remain losslessly representable through id 0 plus their UTF-8 bytes.
WIRE_SCHEMA = "kerc_residual_wire_v1"
WIRE_KEYS = (
    "access_policy",
    "actor_id",
    "aliases",
    "authority",
    "authority_rank",
    "byte_end",
    "byte_start",
    "character_end",
    "character_start",
    "content_ref",
    "copy_policy",
    "encoding",
    "exactness",
    "expiry",
    "fidelity",
    "formatting",
    "frame_name",
    "frame_roles",
    "global",
    "handle",
    "inline_bytes_b64",
    "language",
    "lexical_unit",
    "locked",
    "morphology",
    "object_sha256",
    "object_type",
    "privacy",
    "promotion_economics",
    "protection_source",
    "provenance_hash",
    "realization_ref",
    "record_type",
    "security_labels",
    "source_span",
    "style",
    "tag",
    "target_spans",
    "terminology",
    "units",
    "value",
)
WIRE_KEY_IDS = {key: index + 1 for index, key in enumerate(WIRE_KEYS)}


class ResidualEconomicsFault(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def _uvarint(value: int) -> bytes:
    if value < 0:
        raise ResidualEconomicsFault("KERC_RESIDUAL_WIRE_INTEGER_INVALID", str(value))
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _wire_bytes(value: Any) -> bytes:
    """Encode JSON-like residual values into a deterministic typed byte stream."""

    if value is None:
        return b"\x00"
    if value is False:
        return b"\x01"
    if value is True:
        return b"\x02"
    if isinstance(value, int) and not isinstance(value, bool):
        zigzag = value * 2 if value >= 0 else (-value * 2) - 1
        return b"\x03" + _uvarint(zigzag)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResidualEconomicsFault(
                "KERC_RESIDUAL_WIRE_FLOAT_INVALID", str(value)
            )
        return b"\x04" + struct.pack(">d", value)
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return b"\x05" + _uvarint(len(payload)) + payload
    if isinstance(value, bytes):
        return b"\x06" + _uvarint(len(value)) + value
    if isinstance(value, (list, tuple)):
        return b"\x07" + _uvarint(len(value)) + b"".join(
            _wire_bytes(item) for item in value
        )
    if isinstance(value, dict):
        rows = []
        for raw_key in sorted(value, key=str):
            if not isinstance(raw_key, str):
                raise ResidualEconomicsFault(
                    "KERC_RESIDUAL_WIRE_KEY_INVALID", str(type(raw_key))
                )
            key_id = WIRE_KEY_IDS.get(raw_key, 0)
            key = _uvarint(key_id)
            if key_id == 0:
                encoded_key = raw_key.encode("utf-8")
                key += _uvarint(len(encoded_key)) + encoded_key
            rows.append(key + _wire_bytes(value[raw_key]))
        return b"\x08" + _uvarint(len(rows)) + b"".join(rows)
    raise ResidualEconomicsFault(
        "KERC_RESIDUAL_WIRE_TYPE_INVALID", str(type(value))
    )


def residual_wire_bytes(value: Any) -> bytes:
    return WIRE_SCHEMA.encode("ascii") + b"\x00" + _wire_bytes(value)


class _BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def write(self, bit: int) -> None:
        self.bits.append(1 if bit else 0)

    def finish(self) -> tuple[bytes, int]:
        bit_length = len(self.bits)
        padded = [*self.bits]
        padded.extend([0] * ((8 - len(padded) % 8) % 8))
        payload = bytes(
            sum(padded[index + offset] << (7 - offset) for offset in range(8))
            for index in range(0, len(padded), 8)
        )
        return payload, bit_length


class _BitReader:
    def __init__(self, payload: bytes, bit_length: int) -> None:
        self.payload = payload
        self.bit_length = bit_length
        self.offset = 0

    def read(self) -> int:
        if self.offset >= self.bit_length:
            return 0
        byte = self.payload[self.offset // 8]
        bit = (byte >> (7 - self.offset % 8)) & 1
        self.offset += 1
        return bit


class _FrequencyRow:
    """Mutable 256-symbol distribution with logarithmic prefix operations."""

    def __init__(self) -> None:
        self.counts = [1] * 256
        self.tree = [0, *(index & -index for index in range(1, 257))]
        self.total = 256

    def _rebuild(self) -> None:
        self.tree = [0] * 257
        self.total = 0
        for symbol, count in enumerate(self.counts):
            self._add(symbol, count)

    def _add(self, symbol: int, delta: int) -> None:
        self.total += delta
        index = symbol + 1
        while index < len(self.tree):
            self.tree[index] += delta
            index += index & -index

    def prefix(self, symbol: int) -> int:
        total = 0
        index = symbol
        while index:
            total += self.tree[index]
            index -= index & -index
        return total

    def interval(self, symbol: int) -> tuple[int, int, int]:
        low = self.prefix(symbol)
        return low, low + self.counts[symbol], self.total

    def find(self, scaled: int) -> int:
        if not 0 <= scaled < self.total:
            raise ResidualEconomicsFault(
                "KERC_RESIDUAL_CODEC_SCALED_VALUE_INVALID", str(scaled)
            )
        index = 0
        running = 0
        bit = 1 << 8
        while bit:
            candidate = index + bit
            if candidate <= 256 and running + self.tree[candidate] <= scaled:
                index = candidate
                running += self.tree[candidate]
            bit >>= 1
        return index

    def observe(self, symbol: int, increment: int = 1) -> None:
        self.counts[symbol] += increment
        self._add(symbol, increment)
        if self.total >= MAX_TOTAL:
            self.counts = [max(1, (value + 1) // 2) for value in self.counts]
            self._rebuild()


class _AdaptiveOrderOneModel:
    """Finite-state byte model with deterministic conditioning and adaptation."""

    BOS = 256

    def __init__(self, conditioning: bytes) -> None:
        self.counts: dict[int, _FrequencyRow] = {}
        previous = self.BOS
        for symbol in conditioning:
            row = self._row(previous)
            row.observe(symbol, 4)
            previous = symbol

    def _row(self, context: int) -> _FrequencyRow:
        row = self.counts.get(context)
        if row is None:
            row = _FrequencyRow()
            self.counts[context] = row
        return row

    def interval(self, context: int, symbol: int) -> tuple[int, int, int]:
        return self._row(context).interval(symbol)

    def find(self, context: int, scaled: int) -> int:
        return self._row(context).find(scaled)

    def observe(self, context: int, symbol: int) -> None:
        self._row(context).observe(symbol)


def _encode_arithmetic(payload: bytes, conditioning: bytes) -> tuple[bytes, int]:
    full = 1 << STATE_BITS
    half = full >> 1
    quarter = half >> 1
    three_quarter = quarter * 3
    low = 0
    high = full - 1
    pending = 0
    writer = _BitWriter()
    model = _AdaptiveOrderOneModel(conditioning)
    context = model.BOS

    def emit(bit: int) -> None:
        nonlocal pending
        writer.write(bit)
        for _ in range(pending):
            writer.write(1 - bit)
        pending = 0

    for symbol in payload:
        cumulative_low, cumulative_high, total = model.interval(context, symbol)
        width = high - low + 1
        high = low + (width * cumulative_high // total) - 1
        low = low + (width * cumulative_low // total)
        while True:
            if high < half:
                emit(0)
            elif low >= half:
                emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarter:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low = low << 1
            high = (high << 1) | 1
        model.observe(context, symbol)
        context = symbol
    pending += 1
    emit(0 if low < quarter else 1)
    return writer.finish()


def _decode_arithmetic(
    encoded: bytes, bit_length: int, byte_count: int, conditioning: bytes
) -> bytes:
    full = 1 << STATE_BITS
    half = full >> 1
    quarter = half >> 1
    three_quarter = quarter * 3
    reader = _BitReader(encoded, bit_length)
    low = 0
    high = full - 1
    code = 0
    for _ in range(STATE_BITS):
        code = (code << 1) | reader.read()
    model = _AdaptiveOrderOneModel(conditioning)
    context = model.BOS
    output = bytearray()
    for _ in range(byte_count):
        total = model._row(context).total
        width = high - low + 1
        scaled = ((code - low + 1) * total - 1) // width
        symbol = model.find(context, scaled)
        if not 0 <= symbol < 256:
            raise ResidualEconomicsFault(
                "KERC_RESIDUAL_CODEC_SYMBOL_INVALID", str(symbol)
            )
        cumulative_low, cumulative_high, interval_total = model.interval(
            context, symbol
        )
        if interval_total != total:
            raise ResidualEconomicsFault(
                "KERC_RESIDUAL_CODEC_MODEL_DRIFT", f"{interval_total}!={total}"
            )
        high = low + (width * cumulative_high // total) - 1
        low = low + (width * cumulative_low // total)
        while True:
            if high < half:
                pass
            elif low >= half:
                low -= half
                high -= half
                code -= half
            elif low >= quarter and high < three_quarter:
                low -= quarter
                high -= quarter
                code -= quarter
            else:
                break
            low = low << 1
            high = (high << 1) | 1
            code = ((code << 1) | reader.read()) & (full - 1)
        output.append(symbol)
        model.observe(context, symbol)
        context = symbol
    return bytes(output)


def encode_conditional_payload(
    payload: bytes, conditioning: bytes, *, verify_roundtrip: bool = True
) -> dict[str, Any]:
    encoded, bit_length = _encode_arithmetic(payload, conditioning)
    if verify_roundtrip:
        decoded = _decode_arithmetic(
            encoded, bit_length, len(payload), conditioning
        )
        if decoded != payload:
            raise ResidualEconomicsFault(
                "KERC_RESIDUAL_CODEC_ROUNDTRIP_MISMATCH", digest_bytes(payload)
            )
    record = {
        "policy": CODEC_POLICY,
        "model": "adaptive_order1_byte_fsm_conditioned_arithmetic_v1",
        "state_bits": STATE_BITS,
        "conditioning_sha256": digest_bytes(conditioning),
        "payload_sha256": digest_bytes(payload),
        "payload_byte_count": len(payload),
        "uncompressed_bits": len(payload) * 8,
        "encoded_bits": bit_length,
        "encoded_byte_count": len(encoded),
        "encoded_base64": base64.b64encode(encoded).decode("ascii"),
        "exact_roundtrip": True,
        "fallback_return_count": 0,
    }
    record["record_sha256"] = digest(record)
    return record


def decode_conditional_payload(record: dict[str, Any], conditioning: bytes) -> bytes:
    core = {key: copy.deepcopy(value) for key, value in record.items() if key != "record_sha256"}
    if record.get("policy") != CODEC_POLICY or record.get("record_sha256") != digest(core):
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_CODEC_RECORD_INVALID", str(record.get("record_sha256"))
        )
    if record.get("conditioning_sha256") != digest_bytes(conditioning):
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_CODEC_CONDITION_MISMATCH", digest_bytes(conditioning)
        )
    try:
        encoded = base64.b64decode(str(record["encoded_base64"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_CODEC_PAYLOAD_INVALID", str(exc)
        ) from exc
    decoded = _decode_arithmetic(
        encoded,
        int(record["encoded_bits"]),
        int(record["payload_byte_count"]),
        conditioning,
    )
    if digest_bytes(decoded) != record.get("payload_sha256"):
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_CODEC_DECODE_DIGEST_MISMATCH", digest_bytes(decoded)
        )
    return decoded


def build_residual_codec(
    *,
    kernel_program: dict[str, Any],
    global_state: dict[str, Any],
    segment_residual: dict[str, Any],
    token_residuals: list[dict[str, Any]],
    exact_objects: dict[str, Any],
) -> dict[str, Any]:
    kernel = residual_wire_bytes(kernel_program)
    global_payload = residual_wire_bytes(global_state)
    segment_payload = residual_wire_bytes(segment_residual)
    token_payload = residual_wire_bytes(token_residuals)
    exact_payload = residual_wire_bytes(exact_objects)
    payloads = {
        "interaction": (global_payload, kernel),
        "segment": (segment_payload, kernel + b"\x00" + global_payload),
        "token": (
            token_payload,
            kernel + b"\x00" + global_payload + b"\x00" + segment_payload,
        ),
        "exact": (
            exact_payload,
            kernel + b"\x00" + global_payload + b"\x00" + segment_payload,
        ),
    }
    channels = {
        name: encode_conditional_payload(
            payload, conditioning, verify_roundtrip=False
        )
        for name, (payload, conditioning) in payloads.items()
    }
    result = {
        "policy": CODEC_POLICY,
        "codec_authority": "exact_accounting_and_roundtrip_not_learned_utility",
        "wire_schema": WIRE_SCHEMA,
        "channel_order": ["interaction", "segment", "token", "exact"],
        "channels": channels,
        "total_encoded_bits": sum(row["encoded_bits"] for row in channels.values()),
        "total_uncompressed_bits": sum(
            row["uncompressed_bits"] for row in channels.values()
        ),
        "encoded_storage_bytes": sum(
            row["encoded_byte_count"] for row in channels.values()
        ),
        "cleartext_abi_storage_bytes": sum(
            len(canonical_bytes(value))
            for value in (
                global_state,
                segment_residual,
                token_residuals,
                exact_objects,
            )
        ),
        "cleartext_abi_copy_charged_to_wire_bits": False,
        "cleartext_abi_copy_charged_to_storage": True,
        "fallback_return_count": 0,
    }
    result["contract_sha256"] = digest(result)
    return result


def validate_residual_codec(
    record: dict[str, Any],
    *,
    kernel_program: dict[str, Any],
    global_state: dict[str, Any],
    segment_residual: dict[str, Any],
    token_residuals: list[dict[str, Any]],
    exact_objects: dict[str, Any],
) -> dict[str, Any]:
    core = {key: copy.deepcopy(value) for key, value in record.items() if key != "contract_sha256"}
    if record.get("contract_sha256") != digest(core):
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_CODEC_CONTRACT_INVALID", str(record.get("contract_sha256"))
        )
    if record.get("wire_schema") != WIRE_SCHEMA:
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_WIRE_SCHEMA_INVALID", str(record.get("wire_schema"))
        )
    kernel = residual_wire_bytes(kernel_program)
    global_payload = residual_wire_bytes(global_state)
    segment_payload = residual_wire_bytes(segment_residual)
    token_payload = residual_wire_bytes(token_residuals)
    exact_payload = residual_wire_bytes(exact_objects)
    expected_payloads = {
        "interaction": (global_payload, kernel),
        "segment": (segment_payload, kernel + b"\x00" + global_payload),
        "token": (
            token_payload,
            kernel + b"\x00" + global_payload + b"\x00" + segment_payload,
        ),
        "exact": (
            exact_payload,
            kernel + b"\x00" + global_payload + b"\x00" + segment_payload,
        ),
    }
    channels = record.get("channels") if isinstance(record.get("channels"), dict) else {}
    if set(channels) != set(expected_payloads):
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_CODEC_CHANNEL_INVENTORY_INVALID", ",".join(sorted(channels))
        )
    for channel, (payload, conditioning) in expected_payloads.items():
        decoded = decode_conditional_payload(channels[channel], conditioning)
        if decoded != payload:
            raise ResidualEconomicsFault(
                "KERC_RESIDUAL_CODEC_REPLAY_MISMATCH", channel
            )
    encoded_total = sum(int(row["encoded_bits"]) for row in channels.values())
    uncompressed_total = sum(
        int(row["uncompressed_bits"]) for row in channels.values()
    )
    if (
        encoded_total != int(record.get("total_encoded_bits", -1))
        or uncompressed_total != int(record.get("total_uncompressed_bits", -1))
        or sum(int(row["encoded_byte_count"]) for row in channels.values())
        != int(record.get("encoded_storage_bytes", -1))
        or sum(
            len(canonical_bytes(value))
            for value in (
                global_state,
                segment_residual,
                token_residuals,
                exact_objects,
            )
        )
        != int(record.get("cleartext_abi_storage_bytes", -1))
        or record.get("cleartext_abi_copy_charged_to_wire_bits") is not False
        or record.get("cleartext_abi_copy_charged_to_storage") is not True
    ):
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_CODEC_TOTAL_MISMATCH",
            f"{encoded_total}:{uncompressed_total}",
        )
    return {
        "state": "READY",
        "exact_roundtrip": True,
        "total_encoded_bits": int(record["total_encoded_bits"]),
        "total_uncompressed_bits": int(record["total_uncompressed_bits"]),
        "contract_sha256": record["contract_sha256"],
        "fallback_return_count": 0,
    }


def allocate_rate_distortion(
    candidates: Iterable[dict[str, Any]],
    *,
    importance: float,
    lambda_value: float,
    minimum_fidelity: str = "semantic",
) -> dict[str, Any]:
    if minimum_fidelity not in FIDELITY_ORDER:
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_MINIMUM_FIDELITY_INVALID", minimum_fidelity
        )
    if not math.isfinite(importance) or importance < 0.0:
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_IMPORTANCE_INVALID", str(importance)
        )
    if not math.isfinite(lambda_value) or lambda_value < 0.0:
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_LAMBDA_INVALID", str(lambda_value)
        )
    minimum_index = FIDELITY_ORDER.index(minimum_fidelity)
    normalized: list[dict[str, Any]] = []
    for raw in candidates:
        level = str(raw.get("fidelity") or "")
        if level not in FIDELITY_ORDER:
            raise ResidualEconomicsFault(
                "KERC_RESIDUAL_CANDIDATE_FIDELITY_INVALID", level
            )
        bits = int(raw.get("encoded_bits", -1))
        distortion = float(raw.get("distortion", float("nan")))
        if bits < 0 or not math.isfinite(distortion) or distortion < 0.0:
            raise ResidualEconomicsFault(
                "KERC_RESIDUAL_CANDIDATE_COST_INVALID", canonical_bytes(raw).decode()
            )
        hard_blocked = FIDELITY_ORDER.index(level) < minimum_index
        objective = (
            float("inf")
            if hard_blocked
            else bits + lambda_value * importance * distortion
        )
        normalized.append(
            {
                "fidelity": level,
                "encoded_bits": bits,
                "distortion": distortion,
                "hard_blocked": hard_blocked,
                "objective": objective,
                "evidence_sha256": str(raw.get("evidence_sha256") or ""),
            }
        )
    if {row["fidelity"] for row in normalized} != set(FIDELITY_ORDER):
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_CANDIDATE_INVENTORY_INCOMPLETE",
            ",".join(sorted(row["fidelity"] for row in normalized)),
        )
    selected = min(
        normalized,
        key=lambda row: (row["objective"], FIDELITY_ORDER.index(row["fidelity"])),
    )
    result = {
        "policy": ALLOCATION_POLICY,
        "importance": importance,
        "lambda": lambda_value,
        "minimum_fidelity": minimum_fidelity,
        "candidates": normalized,
        "selected_fidelity": selected["fidelity"],
        "selected_encoded_bits": selected["encoded_bits"],
        "selected_distortion": selected["distortion"],
        "selection_rule": "argmin_encoded_bits_plus_lambda_times_importance_times_distortion",
        "fallback_return_count": 0,
    }
    result["allocation_sha256"] = digest(result)
    return result


def build_structural_rate_distortion_allocation(
    *,
    kernel_program: dict[str, Any],
    global_state: dict[str, Any],
    segment_residual: dict[str, Any],
    token_residuals: list[dict[str, Any]],
    exact_objects: dict[str, Any],
    importance: float,
    lambda_value: float,
    exact_codec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure four actual residual candidates and allocate under structural loss.

    Distortion here is intentionally narrow: it counts packet-owned residual
    structures omitted at each fidelity. It does not claim semantic equivalence.
    Protected exact objects impose a hard exact-fidelity constraint.
    """

    channel_views = {
        "semantic": ({}, [], {}),
        "faithful": (segment_residual, [], {}),
        "lexical": (segment_residual, token_residuals, {}),
        "exact": (segment_residual, token_residuals, exact_objects),
    }
    component_weight = {
        "segment": 1.0 if segment_residual else 0.0,
        "token": float(len(token_residuals)),
        "exact": float(len(exact_objects)),
    }
    denominator = max(1.0, sum(component_weight.values()))
    candidates = []
    codec_identities = {}
    codecs_by_payload: dict[str, dict[str, Any]] = {}
    for fidelity in FIDELITY_ORDER:
        candidate_segment, candidate_tokens, candidate_objects = channel_views[fidelity]
        payload_identity = digest(
            {
                "segment": candidate_segment,
                "token": candidate_tokens,
                "exact": candidate_objects,
            }
        )
        codec = codecs_by_payload.get(payload_identity)
        if codec is None:
            if fidelity == "exact" and exact_codec is not None:
                validate_residual_codec(
                    exact_codec,
                    kernel_program=kernel_program,
                    global_state=global_state,
                    segment_residual=candidate_segment,
                    token_residuals=candidate_tokens,
                    exact_objects=candidate_objects,
                )
                codec = copy.deepcopy(exact_codec)
            else:
                codec = build_residual_codec(
                    kernel_program=kernel_program,
                    global_state=global_state,
                    segment_residual=candidate_segment,
                    token_residuals=candidate_tokens,
                    exact_objects=candidate_objects,
                )
            codecs_by_payload[payload_identity] = codec
        omitted = 0.0
        if fidelity == "semantic":
            omitted += component_weight["segment"] + component_weight["token"] + component_weight["exact"]
        elif fidelity == "faithful":
            omitted += component_weight["token"] + component_weight["exact"]
        elif fidelity == "lexical":
            omitted += component_weight["exact"]
        distortion = omitted / denominator
        evidence = {
            "fidelity": fidelity,
            "codec_contract_sha256": codec["contract_sha256"],
            "encoded_bits": codec["total_encoded_bits"],
            "distortion": distortion,
            "distortion_policy": "omitted_packet_owned_residual_mass_v1",
        }
        evidence_sha256 = digest(evidence)
        codec_identities[fidelity] = {
            **evidence,
            "evidence_sha256": evidence_sha256,
        }
        candidates.append(
            {
                "fidelity": fidelity,
                "encoded_bits": codec["total_encoded_bits"],
                "distortion": distortion,
                "evidence_sha256": evidence_sha256,
            }
        )
    minimum = "exact" if exact_objects else "semantic"
    allocation = allocate_rate_distortion(
        candidates,
        importance=importance,
        lambda_value=lambda_value,
        minimum_fidelity=minimum,
    )
    result = {
        **allocation,
        "candidate_evidence": codec_identities,
        "distortion_authority": "source_bound_structural_omission_not_semantic_utility",
        "protected_exact_object_count": len(exact_objects),
        "rate_distortion_optimality_scope": "measured_structural_candidates_only",
        "capability_or_efficiency_claim": False,
    }
    result["allocation_sha256"] = digest(
        {key: value for key, value in result.items() if key != "allocation_sha256"}
    )
    return result


def validate_structural_rate_distortion_allocation(
    record: dict[str, Any],
    *,
    kernel_program: dict[str, Any],
    global_state: dict[str, Any],
    segment_residual: dict[str, Any],
    token_residuals: list[dict[str, Any]],
    exact_objects: dict[str, Any],
    exact_codec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = build_structural_rate_distortion_allocation(
        kernel_program=kernel_program,
        global_state=global_state,
        segment_residual=segment_residual,
        token_residuals=token_residuals,
        exact_objects=exact_objects,
        importance=float(record.get("importance", float("nan"))),
        lambda_value=float(record.get("lambda", float("nan"))),
        exact_codec=exact_codec,
    )
    if record != expected:
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_ALLOCATION_RECEIPT_INVALID", digest(record)
        )
    return expected


def reallocate_structural_receipt(
    record: dict[str, Any], *, lambda_value: float
) -> dict[str, Any]:
    candidates = [
        {
            "fidelity": row["fidelity"],
            "encoded_bits": row["encoded_bits"],
            "distortion": row["distortion"],
            "evidence_sha256": row["evidence_sha256"],
        }
        for row in record.get("candidates") or []
    ]
    allocation = allocate_rate_distortion(
        candidates,
        importance=float(record.get("importance", float("nan"))),
        lambda_value=lambda_value,
        minimum_fidelity=str(record.get("minimum_fidelity") or ""),
    )
    result = {
        **allocation,
        "candidate_evidence": copy.deepcopy(record.get("candidate_evidence") or {}),
        "distortion_authority": "source_bound_structural_omission_not_semantic_utility",
        "protected_exact_object_count": int(
            record.get("protected_exact_object_count", 0)
        ),
        "rate_distortion_optimality_scope": "measured_structural_candidates_only",
        "capability_or_efficiency_claim": False,
    }
    result["allocation_sha256"] = digest(
        {key: value for key, value in result.items() if key != "allocation_sha256"}
    )
    return result


def calibrate_allocation_lambda(
    development_receipts: Iterable[dict[str, Any]],
    *,
    lambda_grid: Iterable[float],
    maximum_importance_weighted_distortion: float,
) -> dict[str, Any]:
    receipts = list(development_receipts)
    grid = sorted(set(float(value) for value in lambda_grid))
    if (
        not receipts
        or not grid
        or any(not math.isfinite(value) or value <= 0.0 for value in grid)
        or not math.isfinite(maximum_importance_weighted_distortion)
        or not 0.0 <= maximum_importance_weighted_distortion <= 1.0
    ):
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_LAMBDA_CALIBRATION_CONTRACT_INVALID",
            f"{len(receipts)}:{grid}:{maximum_importance_weighted_distortion}",
        )
    curve = []
    selected_lambda = None
    for value in grid:
        weighted_distortion = 0.0
        importance_total = 0.0
        counts = {fidelity: 0 for fidelity in FIDELITY_ORDER}
        exact_constraint_violations = 0
        for receipt in receipts:
            allocated = reallocate_structural_receipt(receipt, lambda_value=value)
            importance = float(allocated["importance"])
            weighted_distortion += importance * float(
                allocated["selected_distortion"]
            )
            importance_total += importance
            counts[allocated["selected_fidelity"]] += 1
            if (
                int(allocated["protected_exact_object_count"]) > 0
                and allocated["selected_fidelity"] != "exact"
            ):
                exact_constraint_violations += 1
        mean = weighted_distortion / max(importance_total, 1e-12)
        passed = (
            mean <= maximum_importance_weighted_distortion
            and exact_constraint_violations == 0
        )
        curve.append(
            {
                "lambda": value,
                "importance_weighted_structural_distortion": mean,
                "selected_fidelity_counts": counts,
                "protected_exact_constraint_violations": exact_constraint_violations,
                "passes": passed,
            }
        )
        if passed and selected_lambda is None:
            selected_lambda = value
    if selected_lambda is None:
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_LAMBDA_CALIBRATION_UNSATISFIED",
            str(maximum_importance_weighted_distortion),
        )
    result = {
        "policy": "project_theseus_kerc_dev_only_lambda_calibration_v1",
        "selection_rule": "smallest_frozen_grid_value_meeting_dev_weighted_structural_distortion_ceiling",
        "fit_split": "private_dev",
        "lambda_grid": grid,
        "maximum_importance_weighted_structural_distortion": maximum_importance_weighted_distortion,
        "selected_lambda": selected_lambda,
        "curve": curve,
        "public_benchmark_used": False,
        "final_evaluation_used_for_selection": False,
        "semantic_utility_claim": False,
    }
    result["calibration_sha256"] = digest(result)
    return result


def promotion_economics(
    *,
    definition_bits: int,
    direct_bits: int,
    reference_bits: int,
    observed_uses: int,
) -> dict[str, Any]:
    if min(definition_bits, direct_bits, reference_bits, observed_uses) < 0:
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_PROMOTION_COST_INVALID",
            f"{definition_bits}:{direct_bits}:{reference_bits}:{observed_uses}",
        )
    savings_per_use = direct_bits - reference_bits
    minimum_uses = (
        math.floor(definition_bits / savings_per_use) + 1
        if savings_per_use > 0
        else None
    )
    should_promote = minimum_uses is not None and observed_uses >= minimum_uses
    result = {
        "policy": PROMOTION_POLICY,
        "definition_bits": definition_bits,
        "direct_bits": direct_bits,
        "reference_bits": reference_bits,
        "observed_uses": observed_uses,
        "savings_per_use": savings_per_use,
        "minimum_uses_strict_break_even": minimum_uses,
        "direct_total_bits": observed_uses * direct_bits,
        "shared_total_bits": definition_bits + observed_uses * reference_bits,
        "should_promote": should_promote,
        "fallback_return_count": 0,
    }
    result["economics_sha256"] = digest(result)
    return result


def validate_promotion_economics(record: dict[str, Any]) -> dict[str, Any]:
    expected = promotion_economics(
        definition_bits=int(record.get("definition_bits", -1)),
        direct_bits=int(record.get("direct_bits", -1)),
        reference_bits=int(record.get("reference_bits", -1)),
        observed_uses=int(record.get("observed_uses", -1)),
    )
    if record != expected:
        raise ResidualEconomicsFault(
            "KERC_RESIDUAL_PROMOTION_RECEIPT_INVALID", digest(record)
        )
    return expected
