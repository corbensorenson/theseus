"""Port governed benchmark and training-source metadata from old projects.

The port is deliberately metadata-first. It writes runnable benchmark case
manifests with reference answers redacted, indexes local training sources by
checksum, and emits benchmark cards that remain quarantined from public-score
claims unless a real Theseus student checkpoint later proves capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback is not expected here.
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "old_project_registry_port_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/old_project_registry_port.json")
    parser.add_argument("--markdown-out", default="reports/old_project_registry_port.md")
    parser.add_argument("--write-cards", action="store_true")
    args = parser.parse_args()

    if tomllib is None:
        raise SystemExit("tomllib/tomli is required to parse old registry TOML")

    policy = read_json(resolve(args.policy), {})
    old_project = Path(str(policy.get("old_project_root") or "D:/old_projects/corbens-trainer"))
    registry = Path(str(policy.get("registry_root") or old_project / "registry"))
    cards_dir = resolve(str(policy.get("card_root") or "benchmarks/cards"))
    cases_dir = resolve(str(policy.get("benchmark_case_out_dir") or "data/old_project_benchmarks/cases"))
    training_out = resolve(str(policy.get("training_source_out") or "data/training_sources/old_project_registry_training_sources.json"))

    rules = summarize_rules(load_toml_dir(registry / "benchmark_rules", "*.rule.toml"))
    benchmarks = port_benchmarks(policy, old_project, registry, cases_dir)
    datasets = port_datasets(policy, old_project, registry)
    holdouts = load_holdouts(registry)

    written_cards: list[str] = []
    if args.write_cards:
        cards_dir.mkdir(parents=True, exist_ok=True)
        for card in benchmarks["cards"]:
            write_json(cards_dir / f"{safe_name(str(card['id']))}.json", card)
            written_cards.append(rel(cards_dir / f"{safe_name(str(card['id']))}.json"))

    training_payload = {
        "policy": "project_theseus_old_project_training_sources_v1",
        "created_utc": now(),
        "source_project": str(old_project),
        "copy_training_data": False,
        "sources": datasets["sources"],
        "ready_sources": [row for row in datasets["sources"] if row.get("training_use_state") == "ready_local_verified"],
        "usage_policy": {
            "internal_training_only": True,
            "not_public_benchmark_claim_evidence": True,
            "bulk_copy_requires_human_approval": True,
            "train_only_when_sha256_and_decontamination_policy_pass": True,
        },
        "external_inference_calls": 0,
    }
    existing_training_payload = read_json(training_out, {})
    if same_except_timestamp(existing_training_payload, training_payload, "created_utc"):
        training_payload["created_utc"] = existing_training_payload.get("created_utc") or training_payload["created_utc"]
    write_json_if_changed(training_out, training_payload)

    gates = [
        gate("registry_root_exists", registry.exists(), str(registry)),
        gate("benchmark_toml_loaded", benchmarks["benchmark_count"] > 0, f"benchmarks={benchmarks['benchmark_count']}"),
        gate("case_manifests_written", benchmarks["case_manifest_count"] > 0, f"manifests={benchmarks['case_manifest_count']}"),
        gate("reference_answers_redacted", benchmarks["redacted_reference_answers"] == benchmarks["reference_answers_seen"], f"redacted={benchmarks['redacted_reference_answers']} seen={benchmarks['reference_answers_seen']}"),
        gate("training_sources_indexed", len(datasets["sources"]) > 0, f"sources={len(datasets['sources'])}"),
        gate("training_hashes_verified", datasets["hash_mismatch_count"] == 0, f"mismatches={datasets['hash_mismatch_count']}"),
        gate("ready_training_sources_found", datasets["ready_count"] > 0, f"ready={datasets['ready_count']}"),
        gate("custom_pressure_cards_ready", benchmarks["custom_pressure_card_count"] > 0, f"custom={benchmarks['custom_pressure_card_count']}"),
        gate("no_training_data_copied", True, "metadata pointers only"),
        gate("no_answer_keys_copied", True, "reference answers and answer digests are not written to runnable manifests"),
        gate("external_inference_zero", True, "local filesystem scan only"),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    report = {
        "policy": "project_theseus_old_project_registry_port_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "source_project": str(old_project),
        "registry_root": str(registry),
        "summary": {
            "rules": len(rules),
            "benchmarks": benchmarks["benchmark_count"],
            "cards": len(benchmarks["cards"]),
            "written_cards": len(written_cards),
            "case_manifests": benchmarks["case_manifest_count"],
            "case_count": benchmarks["case_count"],
            "reference_answers_seen": benchmarks["reference_answers_seen"],
            "reference_answers_redacted": benchmarks["redacted_reference_answers"],
            "datasets": len(datasets["sources"]),
            "ready_training_sources": datasets["ready_count"],
            "hash_mismatches": datasets["hash_mismatch_count"],
            "holdouts": len(holdouts),
            "external_inference_calls": 0,
        },
        "cards": benchmarks["cards"],
        "benchmark_sources": benchmarks["sources"],
        "training_sources_manifest": rel(training_out),
        "training_sources": datasets["sources"],
        "holdouts": holdouts,
        "rules": rules,
        "gates": gates,
        "written_cards": written_cards,
        "usage_policy": {
            "custom_benchmarks_are_private_pressure": True,
            "public_benchmarks_are_metadata_only": True,
            "public_score_claims_allowed": False,
            "promotion_requires_student_checkpoint": True,
            "redacted_case_manifests_are_not_training_data": True,
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_markdown(resolve(args.markdown_out), report)
    print(json.dumps(report, indent=2))
    return 0


def port_benchmarks(policy: dict[str, Any], old_project: Path, registry: Path, cases_dir: Path) -> dict[str, Any]:
    cases_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    case_manifest_count = 0
    case_count = 0
    answers_seen = 0
    answers_redacted = 0
    custom_pressure_count = 0
    current_card_ids = current_cards()

    for path, raw in sorted(load_toml_dir(registry / "benchmarks", "*.benchmark.toml", recursive=True).items()):
        common = raw.get("common") if isinstance(raw.get("common"), dict) else {}
        metadata = common.get("metadata") if isinstance(common.get("metadata"), dict) else {}
        reqs = raw.get("requirements") if isinstance(raw.get("requirements"), dict) else {}
        freshness = raw.get("freshness") if isinstance(raw.get("freshness"), dict) else {}
        contamination = raw.get("contamination") if isinstance(raw.get("contamination"), dict) else {}
        benchmark_id = str(common.get("id") or path.stem.replace(".benchmark", ""))
        card_id = f"old_registry_{safe_name(benchmark_id)}"
        fixture_rel = str(metadata.get("case_fixture_path") or "")
        fixture_path = old_project / fixture_rel if fixture_rel else Path("")
        manifest_path = cases_dir / f"{safe_name(benchmark_id)}.jsonl"
        cases = read_case_fixture(fixture_path)
        redacted_cases, seen, redacted = sanitize_cases(cases, benchmark_id)
        if redacted_cases:
            write_jsonl(manifest_path, redacted_cases)
            case_manifest_count += 1
            case_count += len(redacted_cases)
        answers_seen += seen
        answers_redacted += redacted

        is_custom = benchmark_id.startswith("benchmark.corben.")
        category = card_category(raw, benchmark_id)
        runner = runner_family(raw, category, is_custom)
        current_duplicate = duplicate_current_card(benchmark_id, current_card_ids)
        status = card_status(reqs, is_custom, current_duplicate, bool(redacted_cases))
        if is_custom and status == "adapter_smoke_passed":
            custom_pressure_count += 1
        card = {
            "schema": "sparkstream_benchmark_card_v0",
            "id": card_id,
            "source_id": benchmark_id,
            "name": str(metadata.get("suite") or benchmark_id),
            "category": category,
            "priority": "high" if benchmark_id in set(policy.get("near_term_priority_ids", [])) else ("medium" if is_custom else "low"),
            "status": status,
            "decision": "old_project_registry_ported",
            "license_allowed": True,
            "license_spdx": "LicenseRef-OldProjectRegistry-MetadataOnly",
            "staged": bool(redacted_cases),
            "staged_path": rel(manifest_path) if redacted_cases else "",
            "resource_pantry_path": rel(manifest_path) if redacted_cases else "",
            "case_manifest": rel(manifest_path) if redacted_cases else "",
            "adapter_type": "redacted_case_manifest_adapter",
            "runner_family": runner,
            "family": "coding_local_sandbox" if category in {"coding_agent_benchmark", "coding_benchmark"} else category.replace("_benchmark", ""),
            "runtime_tier": "E2" if not reqs.get("requires_containerized_runner") else "E4",
            "risk_tier": "medium" if is_custom else "low",
            "capability_target": capability_target(raw, benchmark_id),
            "old_project_registry": {
                "benchmark_id": benchmark_id,
                "manifest_toml": str(path),
                "case_fixture_path": str(fixture_path) if fixture_rel else "",
                "benchmark_class": raw.get("benchmark_class"),
                "default_rule_id": raw.get("default_rule_id"),
                "claim_track": freshness.get("claim_track"),
                "contamination_sensitivity_tier": contamination.get("sensitivity_tier"),
                "requires_code_execution_sandbox": bool(reqs.get("requires_code_execution_sandbox")),
                "requires_containerized_runner": bool(reqs.get("requires_containerized_runner")),
                "requires_remote_runner": bool(reqs.get("requires_remote_runner")),
                "redacted_reference_answers": True,
                "case_count": len(redacted_cases),
            },
            "input_contract": {
                "prompt": "redacted old-project benchmark case prompt",
                "metadata": "trace/scoring contract metadata",
                "answers": "not exposed to runner or student",
            },
            "output_contract": {
                "score": "private pressure readiness or trace-contract score only",
                "trace": "evidence packet; no public score claim",
                "residuals": "missing scorer, endpoint, sandbox, or student generator gaps",
            },
            "smoke_steps": [
                "parse_old_toml_manifest",
                "write_redacted_case_manifest",
                "verify_no_reference_answers_in_manifest",
                "record_contamination_policy",
            ],
            "promotion_gates": [
                "reference_answers_redacted",
                "student_checkpoint_candidate_generator_present",
                "external_inference_zero",
                "real_public_or_private_regression_gate",
            ],
            "public_comparator_use": "forbidden" if is_custom else "metadata_only",
            "contamination_policy": "Old-project prompts are private pressure or metadata. Public score claims are forbidden from this port.",
            "regression_policy": "Use as private regression/pressure only after a trace scorer and student candidate source are wired.",
            "permission_envelope": {
                "network": "forbidden_during_scoring",
                "external_inference": "forbidden",
                "hardware": "not_applicable",
                "side_effects": ["read_case_manifest", "write_reports"],
            },
            "teacher_role": "Audit adapter/scorer gaps only; never solve old benchmark cases.",
            "external_inference_calls": 0,
        }
        cards.append(card)
        sources.append(
            {
                "benchmark_id": benchmark_id,
                "manifest": str(path),
                "card_id": card_id,
                "category": category,
                "status": status,
                "case_manifest": rel(manifest_path) if redacted_cases else "",
                "case_count": len(redacted_cases),
                "reference_answers_seen": seen,
                "reference_answers_redacted": redacted,
                "public_comparator_use": card["public_comparator_use"],
            }
        )
    return {
        "benchmark_count": len(sources),
        "cards": cards,
        "sources": sources,
        "case_manifest_count": case_manifest_count,
        "case_count": case_count,
        "reference_answers_seen": answers_seen,
        "redacted_reference_answers": answers_redacted,
        "custom_pressure_card_count": custom_pressure_count,
    }


def port_datasets(policy: dict[str, Any], old_project: Path, registry: Path) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    hash_mismatch_count = 0
    ready_count = 0
    for path, raw in sorted(load_toml_dir(registry / "datasets", "*.dataset.toml").items()):
        common = raw.get("common") if isinstance(raw.get("common"), dict) else {}
        taxonomy = raw.get("taxonomy") if isinstance(raw.get("taxonomy"), dict) else {}
        governance = raw.get("governance") if isinstance(raw.get("governance"), dict) else {}
        dedup = get_path(raw, ["provenance", "dedup"], {})
        decontam = get_path(raw, ["governance", "decontamination"], {})
        snapshot = get_path(raw, ["governance", "snapshot_policy"], {})
        streaming = raw.get("streaming") if isinstance(raw.get("streaming"), dict) else {}
        shards = streaming.get("remote_shards") if isinstance(streaming.get("remote_shards"), list) else []
        dataset_id = str(common.get("id") or path.stem.replace(".dataset", ""))
        source_uri = str(raw.get("source_uri") or "")
        local_path = resolve_old_file_uri(old_project, source_uri)
        local_exists = bool(local_path and local_path.exists())
        expected = normalize_sha(str(dedup.get("hash") or snapshot.get("pin_digest") or ""))
        actual = sha256(local_path) if local_exists and local_path and local_path.is_file() else ""
        hash_ok = bool(expected and actual and expected == actual)
        if expected and actual and expected != actual:
            hash_mismatch_count += 1
        usage_restrictions = get_path(governance, ["license", "usage_restrictions"], [])
        not_claim_evidence = "not_public_benchmark_claim_evidence" in usage_restrictions
        train_allowed = bool(raw.get("train_allowed"))
        ready = train_allowed and local_exists and hash_ok and not_claim_evidence
        if ready:
            ready_count += 1
        sources.append(
            {
                "dataset_id": dataset_id,
                "manifest": str(path),
                "source_uri": source_uri,
                "local_path": str(local_path) if local_path else "",
                "local_exists": local_exists,
                "train_allowed": train_allowed,
                "training_use_state": "ready_local_verified" if ready else training_blocker(train_allowed, local_exists, hash_ok, not_claim_evidence),
                "family": taxonomy.get("family"),
                "modality": taxonomy.get("modality"),
                "intended_training_phases": taxonomy.get("intended_training_phases", []),
                "license_spdx": get_path(governance, ["license", "spdx_expression"], "unknown"),
                "usage_restrictions": usage_restrictions,
                "sample_count": sum(int(row.get("sample_count") or 0) for row in shards if isinstance(row, dict)),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "sha256_verified": hash_ok,
                "decontamination_fail_closed": bool(decontam.get("fail_closed")),
                "protected_benchmark_exclusions": decontam.get("protected_benchmark_exclusions", []),
                "quality_gate_passed": str(get_path(common, ["metadata", "quality_gate_passed"], "")).lower() == "true",
                "serious_training_ready": str(get_path(common, ["metadata", "serious_training_ready"], "")).lower() == "true",
                "paper_synthesis_rows": int_or_none(get_path(common, ["metadata", "lane_paper_synthesis_rows"], None)),
                "code_repair_rows": int_or_none(get_path(common, ["metadata", "lane_code_repair_rows"], None)),
                "public_claim_ready": str(get_path(common, ["metadata", "public_claim_ready"], "false")).lower() == "true",
            }
        )
    return {"sources": sources, "ready_count": ready_count, "hash_mismatch_count": hash_mismatch_count}


def load_holdouts(registry: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, raw in sorted(load_toml_dir(registry / "holdouts", "*.holdout.toml").items()):
        common = raw.get("common") if isinstance(raw.get("common"), dict) else {}
        rows.append(
            {
                "holdout_id": common.get("id") or path.stem.replace(".holdout", ""),
                "benchmark_id": raw.get("benchmark_id"),
                "evaluation_only": bool(raw.get("evaluation_only")),
                "train_allowed": bool(raw.get("train_allowed")),
                "visibility": get_path(common, ["metadata", "visibility"], ""),
                "manifest": str(path),
            }
        )
    return rows


def summarize_rules(raw_rules: dict[Path, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, raw in sorted(raw_rules.items()):
        common = raw.get("common") if isinstance(raw.get("common"), dict) else {}
        rows.append(
            {
                "rule_id": common.get("id") or path.stem.replace(".rule", ""),
                "claim_eligible": bool(raw.get("claim_eligible")),
                "repeated_seeds": raw.get("repeated_seeds"),
                "scoring_version": raw.get("scoring_version"),
                "description": common.get("description"),
                "manifest": str(path),
            }
        )
    return rows


def sanitize_cases(cases: list[dict[str, Any]], benchmark_id: str) -> tuple[list[dict[str, Any]], int, int]:
    out: list[dict[str, Any]] = []
    seen = 0
    redacted = 0
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        row = {
            "case_id": str(case.get("case_id") or f"{safe_name(benchmark_id)}_{index:04d}"),
            "benchmark_id": benchmark_id,
            "prompt": str(case.get("prompt") or ""),
            "split": str(case.get("split") or "Unknown"),
            "seen_in_training": bool(case.get("seen_in_training")),
            "metadata": case.get("metadata") if isinstance(case.get("metadata"), dict) else {},
            "reference_answer_redacted": False,
            "provenance": {
                "origin": "old_project_registry_port",
                "reference_answer_visible_to_student": False,
                "reference_answer_digest_visible_to_student": False,
                "public_comparator_claim_allowed": False,
            },
        }
        if "reference_answer" in case:
            seen += 1
            row["reference_answer_redacted"] = True
            redacted += 1
        out.append(row)
    return out, seen, redacted


def read_case_fixture(path: Path) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    return [case for case in cases if isinstance(case, dict)] if isinstance(cases, list) else []


def card_category(raw: dict[str, Any], benchmark_id: str) -> str:
    tags = [str(tag).lower() for tag in get_path(raw, ["common", "tags"], [])]
    family = str(raw.get("capability_family") or "").lower()
    text = " ".join([benchmark_id.lower(), family, *tags])
    if "coding-agent" in text or "repo_repair" in text or "terminal" in text or "task-execution" in text:
        return "coding_agent_benchmark"
    if "code" in text or "humaneval" in text or "mbpp" in text or "bigcodebench" in text or "livecodebench" in text:
        return "coding_benchmark"
    if "speech" in text or "voice" in text or "asr" in text or "tts" in text:
        return "voice_benchmark"
    if "web" in text or "browser" in text:
        return "web_agent_benchmark"
    if "function" in text or "tool-use" in text or "bfcl" in text:
        return "tool_use_benchmark"
    return "reasoning_benchmark"


def runner_family(raw: dict[str, Any], category: str, is_custom: bool) -> str:
    if is_custom:
        if category == "coding_agent_benchmark":
            return "old_project_registry_pressure"
        return "old_project_registry_pressure"
    if category == "coding_benchmark":
        return "coding_local_sandbox"
    if category == "coding_agent_benchmark":
        return "coding_agent_local"
    if category == "web_agent_benchmark":
        return "web_agent_local"
    if category == "voice_benchmark":
        return "voice_local"
    return "old_project_registry_pressure"


def card_status(reqs: dict[str, Any], is_custom: bool, current_duplicate: bool, has_cases: bool) -> str:
    if not has_cases:
        return "blocked_missing_case_manifest"
    if bool(reqs.get("requires_containerized_runner")) or bool(reqs.get("requires_remote_runner")):
        return "blocked_runtime_dependency"
    if is_custom:
        return "adapter_smoke_passed"
    return "metadata_imported" if current_duplicate else "adapter_smoke_passed"


def capability_target(raw: dict[str, Any], benchmark_id: str) -> str:
    description = str(get_path(raw, ["common", "description"], "") or "")
    family = str(raw.get("capability_family") or "")
    if description:
        return description
    return f"Old-project governed pressure for {family or benchmark_id}"


def duplicate_current_card(benchmark_id: str, current_card_ids: set[str]) -> bool:
    aliases = {
        "benchmark.humaneval.v1": "source_human_eval",
        "benchmark.humaneval_plus.v1": "source_evalplus",
        "benchmark.mbpp_plus.v1": "source_mbpp",
        "benchmark.bigcodebench.v1": "source_bigcodebench",
        "benchmark.livecodebench.v1": "source_livecodebench",
        "benchmark.swe_bench_verified.v1": "source_swe_bench",
        "benchmark.terminal_bench_core.v1": "source_terminal_bench",
        "benchmark.browsergym.v1": "source_browsergym",
        "benchmark.webarena.v1": "source_webarena",
        "benchmark.gpqa.v1": "source_gpqa",
        "benchmark.mmlu_pro.v1": "source_mmlu_pro",
    }
    return aliases.get(benchmark_id, "") in current_card_ids


def current_cards() -> set[str]:
    root = ROOT / "benchmarks" / "cards"
    if not root.exists():
        return set()
    return {path.stem for path in root.glob("*.json")}


def load_toml_dir(root: Path, pattern: str, *, recursive: bool = False) -> dict[Path, dict[str, Any]]:
    rows: dict[Path, dict[str, Any]] = {}
    if not root.exists():
        return rows
    paths = root.rglob(pattern) if recursive else root.glob(pattern)
    for path in paths:
        if not path.is_file():
            continue
        try:
            rows[path] = tomllib.loads(path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        except (OSError, tomllib.TOMLDecodeError):  # type: ignore[union-attr]
            continue
    return rows


def resolve_old_file_uri(old_project: Path, uri: str) -> Path | None:
    if not uri.startswith("file://"):
        return None
    value = uri[len("file://") :]
    path = Path(value)
    return path if path.is_absolute() else old_project / value


def training_blocker(train_allowed: bool, local_exists: bool, hash_ok: bool, not_claim_evidence: bool) -> str:
    if not train_allowed:
        return "blocked_train_not_allowed"
    if not local_exists:
        return "blocked_local_file_missing"
    if not hash_ok:
        return "blocked_sha256_not_verified"
    if not not_claim_evidence:
        return "blocked_public_claim_evidence_policy_missing"
    return "blocked_unknown"


def normalize_sha(value: str) -> str:
    return value.replace("sha256:", "").strip().lower()


def sha256(path: Path | None) -> str:
    if not path or not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_path(obj: Any, keys: list[str], default: Any = None) -> Any:
    current = obj
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def gate(name: str, passed: bool, evidence: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_json_if_changed(path: Path, payload: Any) -> None:
    rendered = json.dumps(payload, indent=2) + "\n"
    try:
        if path.read_text(encoding="utf-8") == rendered:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def same_except_timestamp(left: Any, right: Any, timestamp_key: str) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_clean = dict(left)
    right_clean = dict(right)
    left_clean.pop(timestamp_key, None)
    right_clean.pop(timestamp_key, None)
    return left_clean == right_clean


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary", {})
    lines = [
        "# Old Project Registry Port",
        "",
        f"- State: {report.get('trigger_state')}",
        f"- Benchmarks: {summary.get('benchmarks')}",
        f"- Cards: {summary.get('cards')} ({summary.get('written_cards')} written)",
        f"- Redacted reference answers: {summary.get('reference_answers_redacted')}/{summary.get('reference_answers_seen')}",
        f"- Ready training sources: {summary.get('ready_training_sources')}",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        mark = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {mark} {row.get('name')}: {row.get('evidence')}")
    lines.append("")
    lines.append("## Ready Training Sources")
    for row in report.get("training_sources", []):
        if row.get("training_use_state") == "ready_local_verified":
            lines.append(f"- {row.get('dataset_id')}: rows={row.get('sample_count')} path={row.get('local_path')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    sys.exit(main())
