#!/usr/bin/env python3
"""Run two pinned local blind English raters and conditional adjudication."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/neural_seed_local_english_raters.json"
DEFAULT_JUDGMENT_DIR = ROOT / "reports/private_functional_english_judgments"
DEFAULT_RECEIPT = ROOT / "reports/private_functional_english_judgment_receipt.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--packet",
        action="append",
        default=[],
        help="Opaque label=blind-packet.json; repeat for each candidate bundle",
    )
    parser.add_argument("--judgment-dir", default=str(DEFAULT_JUDGMENT_DIR))
    parser.add_argument("--receipt-out", default=str(DEFAULT_RECEIPT))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = read_json(config_path)
    packet_specs = parse_packet_specs(args.packet)
    contract = build_contract(config, config_path, packet_specs)
    if args.gate or contract["trigger_state"] != "GREEN":
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0 if contract["trigger_state"] == "GREEN" else 2
    receipt = execute(
        config,
        config_path,
        packet_specs,
        judgment_dir=resolve(args.judgment_dir),
    )
    write_json(resolve(args.receipt_out), receipt)
    print(json.dumps({key: receipt[key] for key in ("policy", "created_utc", "trigger_state", "packet_count", "judgment_count", "adjudicated_case_count", "hard_gaps")}, indent=2, sort_keys=True))
    return 0 if receipt["trigger_state"] == "GREEN" else 2


def parse_packet_specs(values: list[str]) -> list[tuple[str, Path]]:
    rows = []
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", label):
            raise ValueError("--packet must be opaque_label=path")
        rows.append((label, resolve(raw_path)))
    if len({label for label, _path in rows}) != len(rows):
        raise ValueError("packet labels must be unique")
    return rows


def build_contract(
    config: dict[str, Any], config_path: Path, packet_specs: list[tuple[str, Path]]
) -> dict[str, Any]:
    gaps = validate_config(config)
    if not packet_specs:
        gaps.append("blind_packet_missing")
    packet_rows = []
    for label, path in packet_specs:
        packet = read_json(path) if path.is_file() else {}
        packet_gaps = validate_packet(packet)
        gaps.extend(f"{label}:{gap}" for gap in packet_gaps)
        packet_rows.append(
            {
                "label": label,
                "path": relative(path),
                "sha256": sha256_file(path) if path.is_file() else "",
                "packet_sha256": packet.get("packet_sha256"),
                "item_count": packet.get("item_count"),
            }
        )
    model_rows = []
    for card in [*config.get("primary_raters", []), config.get("adjudicator") or {}]:
        snapshot, error = local_snapshot(card)
        if error:
            gaps.append(f"local_model_unavailable:{card.get('rater_id')}:{error}")
        model_rows.append(
            {
                "rater_id": card.get("rater_id"),
                "repo_id": card.get("repo_id"),
                "revision": card.get("revision"),
                "snapshot": str(snapshot) if snapshot else "",
                "available": snapshot is not None,
            }
        )
    return {
        "policy": "project_theseus_local_blind_english_rater_contract_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not gaps else "RED",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation": relative(Path(__file__)),
        "implementation_sha256": sha256_file(Path(__file__)),
        "packets": packet_rows,
        "models": model_rows,
        "hard_gaps": sorted(set(gaps)),
        "boundaries": config.get("boundaries") or {},
    }


def execute(
    config: dict[str, Any],
    config_path: Path,
    packet_specs: list[tuple[str, Path]],
    *,
    judgment_dir: Path,
) -> dict[str, Any]:
    contract = build_contract(config, config_path, packet_specs)
    if contract["trigger_state"] != "GREEN":
        return {**contract, "policy": "project_theseus_local_blind_english_judgment_receipt_v1"}
    packets = {label: read_json(path) for label, path in packet_specs}
    entries = []
    for label, packet in packets.items():
        for item in packet["items"]:
            entries.append({"packet_label": label, **item})
    order = list(range(len(entries)))
    random.Random(int(config["seed"])).shuffle(order)
    ordered = [entries[index] for index in order]
    all_judgments: dict[str, list[dict[str, Any]]] = {label: [] for label in packets}
    model_receipts = []
    hard_gaps = []
    for card in config["primary_raters"]:
        judgments, model_receipt = score_with_model(card, ordered, config, adjudicator=False)
        model_receipts.append(model_receipt)
        hard_gaps.extend(model_receipt["hard_gaps"])
        for row in judgments:
            all_judgments[row.pop("packet_label")].append(row)
    disagreement_keys = adjudication_keys(all_judgments, config)
    if disagreement_keys:
        selected = [
            row
            for row in ordered
            if (row["packet_label"], row["case_id"]) in disagreement_keys
        ]
        judgments, model_receipt = score_with_model(
            config["adjudicator"], selected, config, adjudicator=True
        )
        model_receipts.append(model_receipt)
        hard_gaps.extend(model_receipt["hard_gaps"])
        for row in judgments:
            all_judgments[row.pop("packet_label")].append(row)
    judgment_dir.mkdir(parents=True, exist_ok=True)
    judgment_files = []
    for label, path in packet_specs:
        rows = sorted(
            all_judgments[label],
            key=lambda row: (
                str(row["case_id"]),
                bool(row.get("adjudicator")),
                str(row["rater_id"]),
            ),
        )
        output = judgment_dir / f"{label}.jsonl"
        write_jsonl(output, rows)
        judgment_files.append(
            {
                "label": label,
                "path": relative(output),
                "sha256": sha256_file(output),
                "row_count": len(rows),
                "blind_packet_path": relative(path),
                "blind_packet_sha256": sha256_file(path),
                "blind_packet_contract_sha256": packets[label]["packet_sha256"],
            }
        )
    expected_primary = len(entries) * int(config["scoring"]["primary_raters_required"])
    observed_primary = sum(
        row.get("adjudicator") is not True
        for rows in all_judgments.values()
        for row in rows
    )
    if observed_primary != expected_primary:
        hard_gaps.append("primary_judgment_count_mismatch")
    observed_adjudication = sum(
        row.get("adjudicator") is True
        for rows in all_judgments.values()
        for row in rows
    )
    if observed_adjudication != len(disagreement_keys):
        hard_gaps.append("adjudication_count_mismatch")
    return {
        "policy": "project_theseus_local_blind_english_judgment_receipt_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_gaps else "RED",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation": relative(Path(__file__)),
        "implementation_sha256": sha256_file(Path(__file__)),
        "packet_count": len(packets),
        "blind_item_count": len(entries),
        "judgment_count": observed_primary + observed_adjudication,
        "adjudicated_case_count": len(disagreement_keys),
        "judgment_files": judgment_files,
        "model_receipts": model_receipts,
        "hard_gaps": sorted(set(hard_gaps)),
        "local_evaluator_inference_calls": sum(
            int(row["inference_calls"]) for row in model_receipts
        ),
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "judgments_admitted_to_training": False,
        "raw_model_responses_retained": False,
    }


def score_with_model(
    card: dict[str, Any],
    entries: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    adjudicator: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    snapshot, error = local_snapshot(card)
    if error or snapshot is None:
        return [], model_failure_receipt(card, f"local_model_unavailable:{error}")
    identity = snapshot_identity(snapshot)
    load_started = time.perf_counter()
    model, tokenizer = load(str(snapshot), lazy=False)
    mx.eval(model.parameters())
    load_ms = round((time.perf_counter() - load_started) * 1000.0, 6)
    sampler = make_sampler(temp=float(config["generation"]["temperature"]))
    judgments = []
    calls = 0
    retries = 0
    faults = []
    generation_ms = 0.0
    try:
        for entry in entries:
            prompt = rating_prompt(entry, config)
            parsed = None
            last_error = ""
            response_hash = ""
            maximum_attempts = 1 + int(config["generation"]["maximum_format_retries"])
            for attempt in range(maximum_attempts):
                active_prompt = prompt if attempt == 0 else format_retry_prompt(prompt, last_error)
                messages = [
                    {"role": "system", "content": system_prompt(config)},
                    {"role": "user", "content": active_prompt},
                ]
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **dict(card.get("chat_template_kwargs") or {}),
                )
                started = time.perf_counter()
                response = generate(
                    model,
                    tokenizer,
                    rendered,
                    max_tokens=int(config["generation"]["maximum_output_tokens"]),
                    sampler=sampler,
                    verbose=False,
                )
                generation_ms += (time.perf_counter() - started) * 1000.0
                calls += 1
                response_hash = hashlib.sha256(response.encode()).hexdigest()
                parsed, last_error = parse_scores(response, config)
                if parsed is not None:
                    break
                retries += int(attempt + 1 < maximum_attempts)
            if parsed is None:
                faults.append(
                    f"invalid_rater_output:{card['rater_id']}:{entry['packet_label']}:{entry['case_id']}:{last_error}"
                )
                continue
            row = {
                "case_id": entry["case_id"],
                "blind_item_id": entry["blind_item_id"],
                "candidate_sha256": entry["candidate_sha256"],
                "rater_id": card["rater_id"],
                "scores": parsed,
                "local_model_response_sha256": response_hash,
                "packet_label": entry["packet_label"],
            }
            if adjudicator:
                row["adjudicator"] = True
            judgments.append(row)
    finally:
        del model
        del tokenizer
        gc.collect()
        mx.clear_cache()
    return judgments, {
        "rater_id": card["rater_id"],
        "role": "adjudicator" if adjudicator else "primary",
        "repo_id": card["repo_id"],
        "revision": card["revision"],
        "snapshot_identity": identity,
        "load_duration_ms": load_ms,
        "generation_duration_ms": round(generation_ms, 6),
        "inference_calls": calls,
        "format_retries": retries,
        "judgment_count": len(judgments),
        "hard_gaps": faults,
        "raw_model_responses_retained": False,
    }


def rating_prompt(entry: dict[str, Any], config: dict[str, Any]) -> str:
    dimensions = list(config["scoring"]["dimensions"])
    schema = json.dumps({dimension: 0 for dimension in dimensions}, separators=(",", ":"))
    return (
        "Evaluate the candidate response to the user request. Score each dimension from 0 to 4. "
        "Use only the request and candidate shown here. Return exactly one JSON object with integer values and no other text.\n\n"
        f"USER REQUEST:\n{entry['prompt']}\n\n"
        f"CANDIDATE RESPONSE:\n{entry['candidate_output']}\n\n"
        f"JSON SHAPE:\n{schema}"
    )


def system_prompt(config: dict[str, Any]) -> str:
    return (
        "You are a strict, architecture-blind evaluator. Never infer or discuss model identity. "
        "Scores: 0 is unusable or contradictory; 1 has major failures; 2 is partially useful but materially incomplete; "
        "3 is correct and useful with at most minor issues; 4 is excellent, precise, and complete. "
        "instruction_fulfillment measures compliance with explicit constraints; correctness_and_grounding measures factual/logical support from supplied context; "
        "conversation_state measures retention and correct revision of prior-turn state; calibration measures appropriate uncertainty, clarification, or abstention; "
        "clarity measures concise, coherent communication. Output JSON only."
    )


def format_retry_prompt(original: str, error: str) -> str:
    return (
        original
        + "\n\nYour prior response was rejected for formatting: "
        + error
        + ". Return only the required JSON object."
    )


def parse_scores(response: str, config: dict[str, Any]) -> tuple[dict[str, int] | None, str]:
    dimensions = set(config["scoring"]["dimensions"])
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        try:
            value, _end = decoder.raw_decode(response[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or set(value) != dimensions:
            continue
        if all(
            isinstance(value[key], int)
            and int(config["scoring"]["score_minimum"])
            <= value[key]
            <= int(config["scoring"]["score_maximum"])
            for key in dimensions
        ):
            return {key: int(value[key]) for key in config["scoring"]["dimensions"]}, ""
    return None, "no_exact_integer_score_object"


def adjudication_keys(
    judgments: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> set[tuple[str, str]]:
    delta = int(config["scoring"]["adjudication_required_score_delta"])
    dimensions = list(config["scoring"]["dimensions"])
    required = set()
    for label, rows in judgments.items():
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row.get("adjudicator") is not True:
                by_case.setdefault(str(row["case_id"]), []).append(row)
        for case_id, primary in by_case.items():
            if len(primary) == 2 and any(
                abs(int(primary[0]["scores"][dimension]) - int(primary[1]["scores"][dimension])) >= delta
                for dimension in dimensions
            ):
                required.add((label, case_id))
    return required


def validate_config(config: dict[str, Any]) -> list[str]:
    gaps = []
    if config.get("policy") != "project_theseus_local_blind_english_raters_v1":
        gaps.append("policy_mismatch")
    primaries = config.get("primary_raters") or []
    adjudicator = config.get("adjudicator") or {}
    cards = [*primaries, adjudicator]
    ids = [str(card.get("rater_id") or "") for card in cards]
    revisions = [str(card.get("revision") or "") for card in cards]
    if len(primaries) != 2 or int((config.get("scoring") or {}).get("primary_raters_required") or 0) != 2:
        gaps.append("exactly_two_primary_raters_required")
    if len(set(ids)) != 3 or any(not value for value in ids):
        gaps.append("rater_ids_not_distinct")
    if len(set(revisions)) != 3 or any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in revisions):
        gaps.append("rater_revisions_not_distinct_or_pinned")
    scoring = config.get("scoring") or {}
    if scoring.get("dimensions") != [
        "instruction_fulfillment",
        "correctness_and_grounding",
        "conversation_state",
        "calibration",
        "clarity",
    ]:
        gaps.append("scoring_dimensions_mismatch")
    for key in ("model_identity_hidden", "checkpoint_identity_hidden", "reference_answer_hidden"):
        if scoring.get(key) is not True:
            gaps.append(f"blind_boundary_missing:{key}")
    boundaries = config.get("boundaries") or {}
    if boundaries.get("local_inference_only") is not True or int(boundaries.get("external_inference_calls", -1)) != 0:
        gaps.append("local_only_boundary_missing")
    if boundaries.get("raw_model_responses_retained") is not False:
        gaps.append("raw_response_retention_must_be_false")
    return gaps


def validate_packet(packet: dict[str, Any]) -> list[str]:
    gaps = []
    if packet.get("policy") != "project_theseus_blind_english_judgment_packet_v1":
        gaps.append("packet_policy_mismatch")
    if packet.get("trigger_state") != "GREEN" or int(packet.get("item_count") or 0) != 32:
        gaps.append("packet_incomplete")
    for key in ("model_identity_present", "checkpoint_identity_present", "reference_answer_present"):
        if packet.get(key) is not False:
            gaps.append(f"packet_blind_boundary_failed:{key}")
    items = packet.get("items") if isinstance(packet.get("items"), list) else []
    for item in items:
        if any(key in item for key in ("model_id", "checkpoint_id", "architecture", "reference_answer")):
            gaps.append("packet_item_identity_or_reference_exposed")
            break
    return gaps


def local_snapshot(card: dict[str, Any]) -> tuple[Path | None, str]:
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=str(card["repo_id"]),
            revision=str(card["revision"]),
            local_files_only=True,
        )
        return Path(path).resolve(), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"


def snapshot_identity(snapshot: Path) -> dict[str, Any]:
    files = []
    for path in sorted(value for value in snapshot.rglob("*") if value.is_file()):
        files.append(
            {
                "path": str(path.relative_to(snapshot)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "snapshot": str(snapshot),
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
        "manifest_sha256": stable_hash(files),
    }


def model_failure_receipt(card: dict[str, Any], fault: str) -> dict[str, Any]:
    return {
        "rater_id": card.get("rater_id"),
        "role": "unknown",
        "repo_id": card.get("repo_id"),
        "revision": card.get("revision"),
        "snapshot_identity": {},
        "load_duration_ms": 0.0,
        "generation_duration_ms": 0.0,
        "inference_calls": 0,
        "format_retries": 0,
        "judgment_count": 0,
        "hard_gaps": [fault],
        "raw_model_responses_retained": False,
    }


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
